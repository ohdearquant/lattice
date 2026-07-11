//! Per-row symmetric INT8 (Q8) weight quantization for Qwen3.5-2B.
//!
//! The model stores linear weights in row-major `[out_features, in_features]` order and
//! evaluates them through `matmul_bt`, which computes `C = A @ B^T`.
//! That layout makes **per-row / per-output-channel** scaling the natural choice:
//!
//! ```text
//! output[j] = scale[j] * sum_k(input[k] * q[j, k])
//! ```
//!
//! This module provides:
//! - a compact `Q8Matrix` representation,
//! - per-row symmetric quantization,
//! - a quantized `matmul_bt_q8` kernel,
//! - quantized mirrors of the major Qwen3.5-2B weight structures,
//! - conversion helpers, and
//! - memory accounting utilities.

#![forbid(unsafe_code)]

use crate::attention::gdn::GatedDeltaNetWeights;
use crate::error::InferenceError;
use crate::model::qwen35::{
    AttentionWeights, CommonLayerWeights, FeedForwardWeights, FullAttentionLayerWeights,
    ModelWeights,
};
use crate::model::qwen35_config::{LayerType, Qwen35Config};
use std::mem::size_of;

/// **Unstable**: per-row symmetric INT8 weight matrix; storage layout and field names may change.
///
/// A quantized weight matrix backed by `i8` values plus one scale per row.
#[derive(Debug, Clone)]
pub struct Q8Matrix {
    /// Quantized weight values in row-major `[rows, cols]` order.
    pub data: Vec<i8>,
    /// Per-row symmetric scale factors, one per output channel.
    pub scales: Vec<f32>,
    /// Number of rows (`out_features`).
    pub rows: usize,
    /// Number of columns (`in_features`).
    pub cols: usize,
}

/// **Unstable**: quantize a row-major f32 matrix into Q8; quantization scheme may change.
///
/// Quantize a single row-major f32 matrix into per-row symmetric Q8.
///
/// For each row `i`:
/// - `scale[i] = max(abs(row)) / 127.0`
/// - if the row is all zeros, `scale[i] = 1.0`
/// - `q[i, j] = clamp(round(w[i, j] / scale[i]), -128, 127)`
///
/// Returns `Err(InvalidInput)` if any source row contains `NaN` or `±inf`.
/// A non-finite source value produces a non-finite scale (`inf / 127 = inf`),
/// which then turns every matmul output for that output channel into `NaN`
/// (`0 * inf`). Rejecting here attributes the error to the bad source weight
/// rather than to a downstream numerical guard.
pub fn quantize_matrix(w: &[f32], rows: usize, cols: usize) -> Result<Q8Matrix, InferenceError> {
    if w.len() != rows * cols {
        return Err(InferenceError::InvalidInput(format!(
            "matrix length {} does not match rows({rows}) * cols({cols})",
            w.len()
        )));
    }

    let mut data = Vec::with_capacity(w.len());
    let mut scales = Vec::with_capacity(rows);

    for row_idx in 0..rows {
        let start = row_idx * cols;
        let end = start + cols;
        let row = &w[start..end];

        let mut max_abs = 0.0f32;
        for &v in row {
            // IEEE 754: `NaN > x` is FALSE for any x, so a NaN value would
            // silently leave max_abs unchanged. The scale then stays finite,
            // and the NaN element is quantized to 0 via Rust's saturating
            // f32-as-i8 cast — wrong data, no error. Reject the row here so
            // the error points at the source weight, not a downstream matmul.
            if !v.is_finite() {
                return Err(InferenceError::InvalidInput(format!(
                    "Q8 weight row {row_idx} contains a non-finite value ({v}); \
                     source weights must be finite"
                )));
            }
            let abs_v = v.abs();
            if abs_v > max_abs {
                max_abs = abs_v;
            }
        }

        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };

        // Defensive guard: the loop above already rejects NaN/±inf inputs,
        // so a non-finite scale here would indicate an internal logic error.
        if !scale.is_finite() {
            return Err(InferenceError::InvalidInput(format!(
                "Q8 weight row {row_idx} has non-finite scale ({scale}); \
                 source row contains NaN or ±inf"
            )));
        }

        scales.push(scale);

        for &v in row {
            let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
            data.push(q);
        }
    }

    Ok(Q8Matrix {
        data,
        scales,
        rows,
        cols,
    })
}

/// **Unstable**: quantized matmul_bt kernel; tile size and dispatch strategy may change.
///
/// Compute `C = A @ B_q^T` where `B_q` is stored as per-row quantized INT8.
///
/// Shapes:
/// - `A`: `[m, k]` in row-major f32
/// - `B_q`: `[n, k]` in row-major Q8
/// - `C`: `[m, n]` in row-major f32
///
/// Each output is computed as:
///
/// ```text
/// c[i, j] = scale[j] * sum_t(a[i, t] * q[j, t])
/// ```
///
/// The activation input remains f32, so the inner product is accumulated in f32 and the
/// per-row scale is applied once per output element.
pub fn matmul_bt_q8(a: &[f32], b_q: &Q8Matrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    // Guard the dimension products against usize overflow BEFORE any `m * k` /
    // `n * k` / `m * n` is evaluated below: in release those multiplies wrap
    // silently, so an overflowing shape could pass a length check and drive an
    // out-of-bounds access. Mirrors the same guards in `matmul_bt`
    // (forward/cpu/matmul.rs); `matmul_bt_q8` was missing them.
    assert!(m.checked_mul(k).is_some(), "matmul shape overflow: m*k");
    assert!(n.checked_mul(k).is_some(), "matmul shape overflow: n*k");
    assert!(m.checked_mul(n).is_some(), "matmul shape overflow: m*n");
    // Oversized-scratch-prefix allow-list (ADR-080 C4): `>=`, not `assert_eq!`. Every access
    // to `a` below is a bounded sub-slice (`a[i*k..(i+1)*k]`), so a caller-supplied `a`
    // longer than `m*k` is sound — unifies with `matmul_bt`/`matmul_bt_f16`'s contract
    // instead of this Q8 path alone rejecting valid oversized callers. `b_q.rows`/`b_q.cols`
    // are the matrix's own declared shape (not a raw buffer length), so those stay an exact
    // shape-mismatch check, not a buffer-size check.
    assert!(a.len() >= m * k, "A length does not match m * k");
    assert_eq!(b_q.rows, n, "B_q rows do not match n");
    assert_eq!(b_q.cols, k, "B_q cols do not match k");
    assert_eq!(
        b_q.data.len(),
        n * k,
        "B_q data length does not match n * k"
    );
    assert_eq!(b_q.scales.len(), n, "B_q scales length does not match n");
    assert!(c.len() >= m * n, "C buffer is too small");

    // On macOS we tile the N (output-channel) dimension, dequantize each tile
    // of B rows from i8→f32 (applying per-row scales), then delegate to the
    // Accelerate-backed matmul_bt which hits the AMX coprocessor.
    //
    // Tile sizing: 64 rows * 2048 cols * 4 bytes = 512 KB fits comfortably in
    // the M-series 16 MB L2 cache. For M=1 (decode) we write tile results
    // directly into c; for M>1 we scatter from a contiguous tile output buffer.
    #[cfg(target_os = "macos")]
    {
        const TILE_N: usize = 64;
        // The tile scratch buffers are sized by `TILE_N * k` and `m * TILE_N`,
        // which are NOT covered by the `m*k` / `n*k` / `m*n` guards above: when
        // `n < TILE_N` (including `n == 0`) these products can overflow while the
        // guarded ones do not, and a wrapped size would under-allocate the
        // scratch and drive an OOB scatter below. Guard them on the same
        // fail-closed contract.
        assert!(
            TILE_N.checked_mul(k).is_some(),
            "matmul shape overflow: TILE_N*k"
        );
        assert!(
            m.checked_mul(TILE_N).is_some(),
            "matmul shape overflow: m*TILE_N"
        );
        let mut b_f32 = vec![0.0f32; TILE_N * k];
        let mut c_tile = vec![0.0f32; m * TILE_N];

        for tile_start in (0..n).step_by(TILE_N) {
            let tile_n = (n - tile_start).min(TILE_N);

            // Dequantize: convert i8 -> f32 and fold in the per-row scale.
            for j in 0..tile_n {
                let global_j = tile_start + j;
                let q_row = &b_q.data[global_j * k..(global_j + 1) * k];
                let scale = b_q.scales[global_j];
                let dst = &mut b_f32[j * k..(j + 1) * k];
                for t in 0..k {
                    dst[t] = q_row[t] as f32 * scale;
                }
            }

            // Accelerate BLAS GEMM: C_tile[m, tile_n] = A[m, k] @ B_tile[tile_n, k]^T
            crate::forward::cpu::matmul_bt(
                a,
                &b_f32[..tile_n * k],
                &mut c_tile[..m * tile_n],
                m,
                k,
                tile_n,
            );

            // Scatter tile results into correct columns of c.
            if m == 1 {
                // Fast path for decode: c is [1, n], tile writes a contiguous run.
                c[tile_start..tile_start + tile_n].copy_from_slice(&c_tile[..tile_n]);
            } else {
                for i in 0..m {
                    let c_row = &mut c[i * n + tile_start..i * n + tile_start + tile_n];
                    let tile_row = &c_tile[i * tile_n..(i + 1) * tile_n];
                    c_row.copy_from_slice(tile_row);
                }
            }
        }
    }

    // Scalar fallback for non-macOS platforms.
    #[cfg(not(target_os = "macos"))]
    {
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let c_row = &mut c[i * n..(i + 1) * n];

            for j in 0..n {
                let q_row = &b_q.data[j * k..(j + 1) * k];
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += a_row[t] * q_row[t] as f32;
                }
                c_row[j] = acc * b_q.scales[j];
            }
        }
    }
}

/// **Unstable**: quantized GatedDeltaNet weights; field set may change with architecture updates.
///
/// Quantized weights for a GatedDeltaNet layer.
///
/// Only the large projection matrices are quantized. Small or numerically sensitive
/// vectors remain in f32.
#[derive(Debug, Clone)]
pub struct Q8GatedDeltaNetWeights {
    /// Quantized QKV projection `[qkv_dim, hidden]`.
    pub in_proj_qkv: Q8Matrix,
    /// Quantized output-gate projection `[output_dim, hidden]`.
    pub in_proj_z: Q8Matrix,
    /// Quantized update-rate projection `[num_value_heads, hidden]`.
    pub in_proj_b: Q8Matrix,
    /// Quantized decay-input projection `[num_value_heads, hidden]`.
    pub in_proj_a: Q8Matrix,
    /// Learnable log-decay values `[num_value_heads]`, kept in f32.
    pub a_log: Vec<f32>,
    /// Learnable timestep biases `[num_value_heads]`, kept in f32.
    pub dt_bias: Vec<f32>,
    /// Depthwise conv1d weights, kept in f32.
    pub conv1d_weight: Vec<f32>,
    /// Conv1d channel count.
    pub conv_dim: usize,
    /// Conv1d kernel size.
    pub kernel_size: usize,
    /// Per-head gated RMSNorm gamma, kept in f32 exactly as loaded.
    pub norm_weight: Vec<f32>,
    /// Quantized output projection `[hidden, output_dim]`.
    pub out_proj: Q8Matrix,
}

/// **Unstable**: quantized full-attention (GQA) layer weights; field set may change with model variants.
#[derive(Debug, Clone)]
pub struct Q8FullAttentionLayerWeights {
    /// Quantized Q-and-gate projection `[2 * q_dim, hidden]`.
    pub q_proj: Q8Matrix,
    /// Quantized K projection `[kv_dim, hidden]`.
    pub k_proj: Q8Matrix,
    /// Quantized V projection `[kv_dim, hidden]`.
    pub v_proj: Q8Matrix,
    /// Quantized output projection `[hidden, q_dim]`.
    pub o_proj: Q8Matrix,
    /// Q normalization weights, kept in f32.
    pub q_norm: Vec<f32>,
    /// K normalization weights, kept in f32.
    pub k_norm: Vec<f32>,
}

/// **Unstable**: quantized common layer weights (MLP); field set may change with model variants.
///
/// Quantized common layer weights shared by both attention types.
#[derive(Debug, Clone)]
pub struct Q8CommonLayerWeights {
    /// Input RMSNorm weights, kept in f32.
    pub input_layernorm: Vec<f32>,
    /// Post-attention RMSNorm weights, kept in f32.
    pub post_attention_layernorm: Vec<f32>,
    /// Quantized SwiGLU gate projection `[intermediate, hidden]`.
    pub gate_proj: Q8Matrix,
    /// Quantized SwiGLU up projection `[intermediate, hidden]`.
    pub up_proj: Q8Matrix,
    /// Quantized down projection `[hidden, intermediate]`.
    pub down_proj: Q8Matrix,
}

/// **Unstable**: quantized per-layer attention storage; variants may change with model support.
///
/// Quantized per-layer attention storage.
#[derive(Debug, Clone)]
pub enum Q8AttentionWeights {
    /// Quantized GatedDeltaNet weights.
    Linear(Q8GatedDeltaNetWeights),
    /// Quantized full-attention weights.
    Full(Q8FullAttentionLayerWeights),
}

/// **Unstable**: all quantized model weights; layout may change with new layer types.
///
/// All quantized model weights.
#[derive(Debug, Clone)]
pub struct Q8ModelWeights {
    /// Embedding table, kept in f32 because it is tied to the LM head.
    pub embed_tokens: Vec<f32>,
    /// Final RMSNorm weights, kept in f32.
    pub final_norm: Vec<f32>,
    /// Per-layer quantized weights.
    pub layers: Vec<(Q8AttentionWeights, Q8CommonLayerWeights)>,
}

/// Quantize all model weights into their Q8 representations.
///
/// The embedding table and final norm are cloned as-is (f32) since the embedding
/// is tied to the LM head and norms are numerically sensitive. All large projection
/// matrices are quantized per-row symmetric INT8.
pub(crate) fn quantize_model_weights(
    weights: &ModelWeights,
    cfg: &Qwen35Config,
) -> Result<Q8ModelWeights, InferenceError> {
    let layers = weights
        .layers
        .iter()
        .map(|(attn, common)| {
            let q8_attn = match attn {
                AttentionWeights::Linear(gdn) => {
                    Q8AttentionWeights::Linear(quantize_gdn_weights(gdn)?)
                }
                AttentionWeights::Full(full) => {
                    Q8AttentionWeights::Full(quantize_full_attn_weights(full, cfg)?)
                }
            };
            let q8_common = quantize_common_weights(common)?;
            Ok((q8_attn, q8_common))
        })
        .collect::<Result<Vec<_>, InferenceError>>()?;

    Ok(Q8ModelWeights {
        embed_tokens: weights.embed_tokens.clone(),
        final_norm: weights.final_norm.clone(),
        layers,
    })
}

/// **Unstable**: convert GatedDeltaNet weights to Q8; output type may change.
///
/// Quantize a `GatedDeltaNetWeights` bundle into its Q8 representation.
///
/// Returns `Err` (never panics) when the decay parameter shapes are inconsistent:
/// `in_proj_b` and `in_proj_a` data lengths must equal their declared `rows * cols`;
/// `a_log` and `dt_bias` lengths must equal `in_proj_b_rows` (the value-head count).
pub fn quantize_gdn_weights(
    w: &GatedDeltaNetWeights,
) -> Result<Q8GatedDeltaNetWeights, InferenceError> {
    let value_head_rows = w.in_proj_b_rows;

    if w.in_proj_b.len() != w.in_proj_b_rows * w.in_proj_b_cols {
        return Err(InferenceError::InvalidInput(format!(
            "GDN in_proj_b data length {} does not match rows({}) * cols({})",
            w.in_proj_b.len(),
            w.in_proj_b_rows,
            w.in_proj_b_cols
        )));
    }
    if w.in_proj_a.len() != w.in_proj_a_rows * w.in_proj_a_cols {
        return Err(InferenceError::InvalidInput(format!(
            "GDN in_proj_a data length {} does not match rows({}) * cols({})",
            w.in_proj_a.len(),
            w.in_proj_a_rows,
            w.in_proj_a_cols
        )));
    }
    if w.in_proj_a_rows != value_head_rows {
        return Err(InferenceError::InvalidInput(format!(
            "GDN in_proj_a_rows ({}) does not match in_proj_b_rows ({})",
            w.in_proj_a_rows, value_head_rows
        )));
    }
    if w.a_log.len() != value_head_rows {
        return Err(InferenceError::InvalidInput(format!(
            "GDN a_log length ({}) does not match value_head rows ({})",
            w.a_log.len(),
            value_head_rows
        )));
    }
    if w.dt_bias.len() != value_head_rows {
        return Err(InferenceError::InvalidInput(format!(
            "GDN dt_bias length ({}) does not match value_head rows ({})",
            w.dt_bias.len(),
            value_head_rows
        )));
    }

    let in_proj_qkv = quantize_matrix(&w.in_proj_qkv, w.in_proj_qkv_rows, w.in_proj_qkv_cols)?;
    let in_proj_z = quantize_matrix(&w.in_proj_z, w.in_proj_z_rows, w.in_proj_z_cols)?;
    let in_proj_b = quantize_matrix(&w.in_proj_b, w.in_proj_b_rows, w.in_proj_b_cols)?;
    let in_proj_a = quantize_matrix(&w.in_proj_a, w.in_proj_a_rows, w.in_proj_a_cols)?;
    let out_proj = quantize_matrix(&w.out_proj, w.out_proj_rows, w.out_proj_cols)?;

    Ok(Q8GatedDeltaNetWeights {
        in_proj_qkv,
        in_proj_z,
        in_proj_b,
        in_proj_a,
        a_log: w.a_log.clone(),
        dt_bias: w.dt_bias.clone(),
        conv1d_weight: w.conv1d_weight.clone(),
        conv_dim: w.conv_dim,
        kernel_size: w.kernel_size,
        norm_weight: w.norm_weight.clone(),
        out_proj,
    })
}

/// Quantize a full-attention layer into its Q8 representation.
///
/// The original full-attention weight struct does not carry explicit row/column metadata,
/// so the matrix shapes are inferred from the config's attention topology.
///
/// Returns `Err(InvalidInput)` if any weight matrix contains `NaN` or `±inf` values.
pub(crate) fn quantize_full_attn_weights(
    w: &FullAttentionLayerWeights,
    cfg: &Qwen35Config,
) -> Result<Q8FullAttentionLayerWeights, InferenceError> {
    let ((q_rows, hidden), (kv_rows, _), (_, _), (o_rows, o_cols)) =
        infer_full_attention_shapes(w, cfg);

    let q_proj = quantize_matrix(&w.q_proj, q_rows, hidden)?;
    let k_proj = quantize_matrix(&w.k_proj, kv_rows, hidden)?;
    let v_proj = quantize_matrix(&w.v_proj, kv_rows, hidden)?;
    let o_proj = quantize_matrix(&w.o_proj, o_rows, o_cols)?;

    Ok(Q8FullAttentionLayerWeights {
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_norm: w.q_norm.clone(),
        k_norm: w.k_norm.clone(),
    })
}

/// Quantize the common per-layer MLP weights into their Q8 representation (dense only).
pub(crate) fn quantize_common_weights(
    w: &CommonLayerWeights,
) -> Result<Q8CommonLayerWeights, InferenceError> {
    let dense = match &w.ffn {
        FeedForwardWeights::Dense(d) => d,
        FeedForwardWeights::Moe(_) => {
            return Err(InferenceError::UnsupportedModel(
                "Q8 quantization is dense-only; MoE layers are not supported".to_string(),
            ));
        }
    };
    let hidden = w.input_layernorm.len();
    let ((gate_rows, _), (up_rows, _), (down_rows, down_cols)) = infer_dense_shapes(dense, hidden);

    let gate_proj = quantize_matrix(&dense.gate_proj, gate_rows, hidden)?;
    let up_proj = quantize_matrix(&dense.up_proj, up_rows, hidden)?;
    let down_proj = quantize_matrix(&dense.down_proj, down_rows, down_cols)?;

    Ok(Q8CommonLayerWeights {
        input_layernorm: w.input_layernorm.clone(),
        post_attention_layernorm: w.post_attention_layernorm.clone(),
        gate_proj,
        up_proj,
        down_proj,
    })
}

/// **Unstable**: compute Q8 vs f32 memory usage; return tuple shape may change.
///
/// Compute memory usage for all weight matrices that are quantized by this module.
///
/// Returns:
/// - total f32 bytes for those matrices,
/// - total Q8 bytes (`i8` data + per-row scales),
/// - compression ratio `f32_bytes / q8_bytes`.
pub fn memory_report(cfg: &Qwen35Config) -> (usize, usize, f32) {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;

    let qkv_dim = cfg.linear_qkv_dim();
    let linear_output_dim = cfg.linear_output_dim();
    let linear_heads = cfg.linear_num_value_heads();

    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();

    let mut f32_bytes = 0usize;
    let mut q8_bytes = 0usize;

    for layer_type in &cfg.layer_types {
        add_matrix_bytes(inter, hidden, &mut f32_bytes, &mut q8_bytes); // gate_proj
        add_matrix_bytes(inter, hidden, &mut f32_bytes, &mut q8_bytes); // up_proj
        add_matrix_bytes(hidden, inter, &mut f32_bytes, &mut q8_bytes); // down_proj

        match layer_type {
            LayerType::LinearAttention => {
                add_matrix_bytes(qkv_dim, hidden, &mut f32_bytes, &mut q8_bytes);
                add_matrix_bytes(linear_output_dim, hidden, &mut f32_bytes, &mut q8_bytes);
                add_matrix_bytes(linear_heads, hidden, &mut f32_bytes, &mut q8_bytes);
                add_matrix_bytes(linear_heads, hidden, &mut f32_bytes, &mut q8_bytes);
                add_matrix_bytes(hidden, linear_output_dim, &mut f32_bytes, &mut q8_bytes);
            }
            LayerType::FullAttention => {
                add_matrix_bytes(2 * q_dim, hidden, &mut f32_bytes, &mut q8_bytes);
                add_matrix_bytes(kv_dim, hidden, &mut f32_bytes, &mut q8_bytes);
                add_matrix_bytes(kv_dim, hidden, &mut f32_bytes, &mut q8_bytes);
                add_matrix_bytes(hidden, q_dim, &mut f32_bytes, &mut q8_bytes);
            }
        }
    }

    let savings_ratio = if q8_bytes == 0 {
        f32::INFINITY
    } else {
        f32_bytes as f32 / q8_bytes as f32
    };

    (f32_bytes, q8_bytes, savings_ratio)
}

#[inline]
fn add_matrix_bytes(rows: usize, cols: usize, f32_bytes: &mut usize, q8_bytes: &mut usize) {
    *f32_bytes += rows * cols * size_of::<f32>();
    *q8_bytes += rows * cols * size_of::<i8>() + rows * size_of::<f32>();
}

fn infer_dense_shapes(
    dense: &crate::model::qwen35::DenseFfnWeights,
    hidden: usize,
) -> ((usize, usize), (usize, usize), (usize, usize)) {
    assert_eq!(
        dense.gate_proj.len() % hidden,
        0,
        "gate_proj length must be divisible by hidden"
    );
    let inter = dense.gate_proj.len() / hidden;
    assert_eq!(
        dense.up_proj.len(),
        inter * hidden,
        "up_proj length does not match inferred [intermediate, hidden] shape"
    );
    assert_eq!(
        dense.down_proj.len(),
        hidden * inter,
        "down_proj length does not match inferred [hidden, intermediate] shape"
    );
    ((inter, hidden), (inter, hidden), (hidden, inter))
}

type ShapePair = (usize, usize);

fn infer_full_attention_shapes(
    w: &FullAttentionLayerWeights,
    cfg: &Qwen35Config,
) -> (ShapePair, ShapePair, ShapePair, ShapePair) {
    let head_dim = w.q_norm.len();

    assert!(head_dim > 0, "q_norm/head_dim must be non-zero");
    assert_eq!(
        w.k_norm.len(),
        head_dim,
        "k_norm length must match q_norm/head_dim"
    );

    let q_dim = cfg.num_attention_heads * head_dim;
    let kv_dim = cfg.num_key_value_heads * head_dim;
    let q_rows = 2 * q_dim;

    assert_eq!(
        w.q_proj.len() % q_rows,
        0,
        "q_proj length must be divisible by inferred row count"
    );
    let hidden = w.q_proj.len() / q_rows;

    assert_eq!(
        w.k_proj.len(),
        kv_dim * hidden,
        "k_proj length does not match inferred [kv_dim, hidden] shape"
    );
    assert_eq!(
        w.v_proj.len(),
        kv_dim * hidden,
        "v_proj length does not match inferred [kv_dim, hidden] shape"
    );
    assert_eq!(
        w.o_proj.len(),
        hidden * q_dim,
        "o_proj length does not match inferred [hidden, q_dim] shape"
    );

    (
        (q_rows, hidden),
        (kv_dim, hidden),
        (kv_dim, hidden),
        (hidden, q_dim),
    )
}

#[cfg(test)]
fn dequantize_matrix(q: &Q8Matrix) -> Vec<f32> {
    let mut out = vec![0.0f32; q.rows * q.cols];
    for row in 0..q.rows {
        let scale = q.scales[row];
        let start = row * q.cols;
        let end = start + q.cols;
        for idx in start..end {
            out[idx] = q.data[idx] as f32 * scale;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::cpu::matmul_bt;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn xorshift32(state: &mut u32) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        x
    }

    fn uniform_signed(state: &mut u32) -> f32 {
        let x = xorshift32(state);
        let u = x as f32 / u32::MAX as f32;
        u * 2.0 - 1.0
    }

    #[test]
    fn test_quantize_identity() {
        let rows = 4;
        let cols = 4;
        let w = vec![
            1.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 0.0, // row 2
            0.0, 0.0, 0.0, 1.0, // row 3
        ];

        let q = quantize_matrix(&w, rows, cols).unwrap();
        let dq = dequantize_matrix(&q);

        assert_eq!(q.rows, rows);
        assert_eq!(q.cols, cols);
        assert_eq!(q.scales.len(), rows);
        assert_eq!(q.data.len(), rows * cols);

        for &scale in &q.scales {
            assert!(approx_eq(scale, 1.0 / 127.0, 1e-8));
        }
        for (orig, recon) in w.iter().zip(dq.iter()) {
            assert!(approx_eq(*orig, *recon, 1e-6));
        }
    }

    /// Build a GatedDeltaNetWeights whose decay params are internally consistent at
    /// value-head granularity: a_log/dt_bias length == in_proj_b_rows == in_proj_a_rows.
    fn valid_gdn_weights(value_heads: usize, hidden: usize) -> GatedDeltaNetWeights {
        let qkv_rows = value_heads * 4;
        let z_rows = value_heads * 2;
        GatedDeltaNetWeights {
            in_proj_qkv: vec![0.0; qkv_rows * hidden],
            in_proj_qkv_rows: qkv_rows,
            in_proj_qkv_cols: hidden,
            in_proj_z: vec![0.0; z_rows * hidden],
            in_proj_z_rows: z_rows,
            in_proj_z_cols: hidden,
            in_proj_b: vec![0.0; value_heads * hidden],
            in_proj_b_rows: value_heads,
            in_proj_b_cols: hidden,
            in_proj_a: vec![0.0; value_heads * hidden],
            in_proj_a_rows: value_heads,
            in_proj_a_cols: hidden,
            a_log: vec![0.0; value_heads],
            dt_bias: vec![0.0; value_heads],
            conv1d_weight: vec![0.0; qkv_rows * 4],
            conv_dim: qkv_rows,
            kernel_size: 4,
            norm_weight: vec![1.0; z_rows],
            out_proj: vec![0.0; hidden * z_rows],
            out_proj_rows: hidden,
            out_proj_cols: z_rows,
        }
    }

    #[test]
    fn quantize_gdn_weights_accepts_consistent_value_head_shapes() {
        let w = valid_gdn_weights(3, 8);
        let q = quantize_gdn_weights(&w).expect("consistent value-head shapes must quantize");
        assert_eq!(q.a_log.len(), 3);
        assert_eq!(q.dt_bias.len(), 3);
    }

    #[test]
    fn quantize_gdn_weights_rejects_decay_shape_mismatch() {
        // a_log length (4) disagrees with the value-head row count (3). The #262 loader
        // bug would have masked an asymmetric mismatch; the quantizer must fail closed
        // (return Err — not panic via quantize_matrix's assert, not silently proceed).
        let mut w = valid_gdn_weights(3, 8);
        w.a_log = vec![0.0; 4];
        match quantize_gdn_weights(&w) {
            Err(InferenceError::InvalidInput(msg)) => {
                assert!(msg.contains("a_log"), "error must name a_log, got: {msg}");
            }
            Err(e) => panic!("expected InvalidInput, got: {e}"),
            Ok(_) => panic!("expected Err for decay shape mismatch, got Ok"),
        }
    }

    #[test]
    fn test_quantize_roundtrip_accuracy() {
        let rows = 32;
        let cols = 64;
        let mut seed = 0x1234_5678u32;
        let mut w = Vec::with_capacity(rows * cols);

        for _ in 0..rows * cols {
            w.push(uniform_signed(&mut seed) * 0.03);
        }

        let q = quantize_matrix(&w, rows, cols).unwrap();
        let dq = dequantize_matrix(&q);

        let mut max_abs_err = 0.0f32;
        let mut mean_abs_err = 0.0f32;
        for (orig, recon) in w.iter().zip(dq.iter()) {
            let err = (orig - recon).abs();
            if err > max_abs_err {
                max_abs_err = err;
            }
            mean_abs_err += err;
        }
        mean_abs_err /= (rows * cols) as f32;

        assert!(max_abs_err < 2.0e-4, "max_abs_err={max_abs_err}");
        assert!(mean_abs_err < 7.0e-5, "mean_abs_err={mean_abs_err}");
    }

    #[test]
    fn test_matmul_bt_q8_matches_f32() {
        let m = 2;
        let k = 256;
        let n = 16;

        let mut seed = 0xCAFE_BABEu32;
        let mut a = Vec::with_capacity(m * k);
        let mut b = Vec::with_capacity(n * k);

        for _ in 0..m * k {
            a.push(uniform_signed(&mut seed) * 0.1);
        }

        for row in 0..n {
            let scale = (row as f32 + 1.0) / 8_192.0;
            b.push(127.0 * scale);
            for col in 1..k {
                let qv = ((row * 17 + col * 13) % 255) as i32 - 127;
                b.push(qv as f32 * scale);
            }
        }

        let b_q = quantize_matrix(&b, n, k).unwrap();

        let mut c_f32 = vec![0.0f32; m * n];
        let mut c_q8 = vec![0.0f32; m * n];
        matmul_bt(&a, &b, &mut c_f32, m, k, n);
        matmul_bt_q8(&a, &b_q, &mut c_q8, m, k, n);

        let mut max_abs_err = 0.0f32;
        let mut max_rel_err = 0.0f32;
        let mut large_outputs = 0usize;

        for (&ref_v, &q_v) in c_f32.iter().zip(c_q8.iter()) {
            let abs_err = (ref_v - q_v).abs();
            if abs_err > max_abs_err {
                max_abs_err = abs_err;
            }
            if ref_v.abs() > 0.01 {
                large_outputs += 1;
                let rel_err = abs_err / ref_v.abs();
                if rel_err > max_rel_err {
                    max_rel_err = rel_err;
                }
            }
        }

        assert!(large_outputs > 0, "expected some non-trivial outputs");
        assert!(max_abs_err < 1.0e-4, "max_abs_err={max_abs_err}");
        assert!(max_rel_err < 0.01, "max_rel_err={max_rel_err}");
    }

    #[test]
    fn test_matmul_bt_q8_known_values() {
        let a = vec![1.0, 2.0, -1.0]; // [1, 3]
        let b = vec![
            1.27, -0.64, 0.0, // row 0, scale = 0.01, q = [127, -64, 0]
            -2.54, 0.0, 1.28, // row 1, scale = 0.02, q = [-127, 0, 64]
        ];

        let q = quantize_matrix(&b, 2, 3).unwrap();
        assert!(approx_eq(q.scales[0], 0.01, 1e-8));
        assert!(approx_eq(q.scales[1], 0.02, 1e-8));
        assert_eq!(&q.data[0..3], &[127i8, -64i8, 0i8]);
        assert_eq!(&q.data[3..6], &[-127i8, 0i8, 64i8]);

        let mut c = vec![0.0f32; 2];
        matmul_bt_q8(&a, &q, &mut c, 1, 3, 2);

        let expected0 = 0.01 * (1.0 * 127.0 + 2.0 * -64.0 + -0.0);
        let expected1 = 0.02 * (1.0 * -127.0 + 2.0 * 0.0 + -64.0);

        assert!(approx_eq(c[0], expected0, 1e-6));
        assert!(approx_eq(c[1], expected1, 1e-6));
        assert!(approx_eq(c[0], -0.01, 1e-6));
        assert!(approx_eq(c[1], -3.82, 1e-6));
    }

    /// Tiny valid Q8Matrix for the overflow-guard tests: the guards fire before
    /// `b_q` is inspected, so shape need only be self-consistent.
    fn dummy_q8_1x1() -> Q8Matrix {
        Q8Matrix {
            data: vec![0i8],
            scales: vec![0.0f32],
            rows: 1,
            cols: 1,
        }
    }

    /// Overflow guard: `m * k` must be rejected before the length `assert_eq!`
    /// wraps it. Mutation-sensitive — dropping the `m*k` guard makes this fall
    /// through to the `A length does not match m * k` assert (m*k wraps in
    /// release), changing the panic message and failing this `expected`.
    #[test]
    #[should_panic(expected = "matmul shape overflow: m*k")]
    fn test_matmul_bt_q8_rejects_mk_overflow() {
        let q = dummy_q8_1x1();
        let mut c = vec![0.0f32; 1];
        // m*k overflows; n*k and m*n do not.
        matmul_bt_q8(&[], &q, &mut c, usize::MAX, 2, 1);
    }

    /// Overflow guard: `n * k` (m*k passes first).
    #[test]
    #[should_panic(expected = "matmul shape overflow: n*k")]
    fn test_matmul_bt_q8_rejects_nk_overflow() {
        let q = dummy_q8_1x1();
        let mut c = vec![0.0f32; 1];
        // m*k = 1*2 ok; n*k = MAX*2 overflows.
        matmul_bt_q8(&[0.0, 0.0], &q, &mut c, 1, 2, usize::MAX);
    }

    /// Overflow guard: `m * n` (m*k and n*k pass first).
    #[test]
    #[should_panic(expected = "matmul shape overflow: m*n")]
    fn test_matmul_bt_q8_rejects_mn_overflow() {
        let q = dummy_q8_1x1();
        let mut c = vec![0.0f32; 1];
        // m*k = MAX*1 ok; n*k = 2*1 ok; m*n = MAX*2 overflows.
        matmul_bt_q8(&[], &q, &mut c, usize::MAX, 1, 2);
    }

    /// Overflow guard for the macOS tile-scratch size `TILE_N * k`. With
    /// `m=0, k=MAX, n=0` every earlier product (`m*k`, `n*k`, `m*n`) and every
    /// length assert is zero/empty, so control reaches the tile block; `TILE_N*k`
    /// then overflows. Mutation-sensitive: dropping this guard makes the
    /// `vec![0.0f32; TILE_N * k]` allocation panic with "capacity overflow"
    /// instead, failing this `expected`.
    #[cfg(target_os = "macos")]
    #[test]
    #[should_panic(expected = "matmul shape overflow: TILE_N*k")]
    fn test_matmul_bt_q8_rejects_tile_nk_overflow() {
        let q = Q8Matrix {
            data: vec![],
            scales: vec![],
            rows: 0,
            cols: usize::MAX,
        };
        matmul_bt_q8(&[], &q, &mut [], 0, usize::MAX, 0);
    }

    /// Overflow guard for the macOS tile-scratch size `m * TILE_N`. With
    /// `m=MAX, k=0, n=0` the earlier products and length asserts are all
    /// zero/empty, so control reaches the tile block; `m*TILE_N` then overflows.
    /// Mutation-sensitive by the same "capacity overflow" message mismatch.
    #[cfg(target_os = "macos")]
    #[test]
    #[should_panic(expected = "matmul shape overflow: m*TILE_N")]
    fn test_matmul_bt_q8_rejects_m_tile_n_overflow() {
        let q = Q8Matrix {
            data: vec![],
            scales: vec![],
            rows: 0,
            cols: 0,
        };
        matmul_bt_q8(&[], &q, &mut [], usize::MAX, 0, 0);
    }

    #[test]
    fn test_zero_row_handling() {
        let b = vec![
            0.0, 0.0, 0.0, 0.0, // zero row
            0.5, -0.5, 0.25, -0.25,
        ];
        let q = quantize_matrix(&b, 2, 4).unwrap();

        assert!(approx_eq(q.scales[0], 1.0, 1e-8));
        assert_eq!(&q.data[0..4], &[0i8, 0i8, 0i8, 0i8]);

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0f32; 2];
        matmul_bt_q8(&a, &q, &mut c, 1, 4, 2);

        assert!(approx_eq(c[0], 0.0, 1e-8));
        assert!(c[1].abs() > 0.0);
    }

    #[test]
    fn test_scale_computation() {
        let w = vec![
            1.27, -0.63, 0.0, // max abs = 1.27 -> scale = 0.01
            -2.54, 2.0, 0.0, // max abs = 2.54 -> scale = 0.02
        ];
        let q = quantize_matrix(&w, 2, 3).unwrap();

        assert!(approx_eq(q.scales[0], 0.01, 1e-8));
        assert!(approx_eq(q.scales[1], 0.02, 1e-8));
        assert_eq!(&q.data[0..3], &[127i8, -63i8, 0i8]);
        assert_eq!(&q.data[3..6], &[-127i8, 100i8, 0i8]);
    }

    #[test]
    fn test_memory_report() {
        let cfg = Qwen35Config::qwen35_2b();
        let (f32_bytes, q8_bytes, ratio) = memory_report(&cfg);

        assert_eq!(f32_bytes, 5_490_868_224);
        assert_eq!(q8_bytes, 1_375_004_928);
        assert!(q8_bytes < f32_bytes);
        assert!(ratio > 3.9 && ratio < 4.05, "ratio={ratio}");
    }

    #[test]
    fn test_quantize_large_matrix() {
        let rows = 6_144;
        let cols = 2_048;
        let mut w = Vec::with_capacity(rows * cols);

        for r in 0..rows {
            let row_scale = 0.005 + (r % 17) as f32 * 0.0005;
            for c in 0..cols {
                let bucket = ((r * 131 + c * 17) % 257) as i32 - 128;
                w.push(bucket as f32 * row_scale / 128.0);
            }
        }

        let q = quantize_matrix(&w, rows, cols).unwrap();

        assert_eq!(q.rows, rows);
        assert_eq!(q.cols, cols);
        assert_eq!(q.data.len(), rows * cols);
        assert_eq!(q.scales.len(), rows);
        assert!(q.scales.iter().all(|&s| s.is_finite() && s > 0.0));
        assert!(q.data.iter().any(|&v| v != 0));

        let first_row_scale = q.scales[0];
        let last_row_scale = q.scales[rows - 1];
        assert!(first_row_scale > 0.0);
        assert!(last_row_scale > 0.0);
    }

    #[test]
    fn test_quantize_common_weights_moe_returns_err_not_panic() {
        use crate::error::InferenceError;
        use crate::model::qwen35::{
            CommonLayerWeights, FeedForwardWeights, MoeLayerWeights, MoeRouter, RoutedExperts,
            SharedExpert,
        };

        let hidden = 4usize;
        let num_experts = 2usize;
        let num_experts_per_tok = 1usize;
        let inter = 2usize;

        let router = MoeRouter::new(
            vec![0.0f32; num_experts * hidden],
            num_experts,
            num_experts_per_tok,
            hidden,
        )
        .expect("valid router shape");

        let experts = RoutedExperts::new(
            vec![0.0f32; num_experts * 2 * inter * hidden],
            vec![0.0f32; num_experts * hidden * inter],
            num_experts,
            hidden,
            inter,
        )
        .expect("valid experts shape");

        let shared_expert = SharedExpert::new(
            vec![0.0f32; inter * hidden],
            vec![0.0f32; inter * hidden],
            vec![0.0f32; hidden * inter],
            vec![0.0f32; hidden],
            hidden,
            inter,
        )
        .expect("valid shared expert shape");

        let moe_common = CommonLayerWeights {
            input_layernorm: vec![0.0f32; hidden],
            post_attention_layernorm: vec![0.0f32; hidden],
            ffn: FeedForwardWeights::Moe(MoeLayerWeights {
                router,
                experts,
                shared_expert,
            }),
        };

        let result = quantize_common_weights(&moe_common);
        assert!(
            matches!(result, Err(InferenceError::UnsupportedModel(_))),
            "expected Err(UnsupportedModel), got: {result:?}"
        );
    }

    /// `quantize_matrix` must reject a weight matrix whose source row contains
    /// `+inf` or `NaN`. Such a row produces `scale = inf / 127 = inf`, which then
    /// poisons every matmul output for that channel via `0 * inf = NaN`.
    ///
    /// Mutation check: removing the `!scale.is_finite()` guard in `quantize_matrix`
    /// causes this call to return `Ok` instead of `Err`, failing `expect_err`.
    #[test]
    fn test_quantize_matrix_rejects_nonfinite_source_row() {
        // Row 0 is finite; row 1 contains +inf → scale for row 1 = inf / 127 = inf.
        let w = vec![
            1.0,
            2.0,
            3.0, // row 0: finite, scale ≈ 3/127
            1.0,
            f32::INFINITY,
            0.0, // row 1: +inf → scale = inf
        ];
        match quantize_matrix(&w, 2, 3) {
            Err(InferenceError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("non-finite"),
                    "error must describe the non-finite scale; got: {msg}"
                );
                assert!(
                    msg.contains("1"),
                    "error must identify the offending row index; got: {msg}"
                );
            }
            Err(e) => panic!("expected InvalidInput for non-finite scale, got: {e}"),
            Ok(_) => panic!("expected Err for non-finite source row, got Ok"),
        }
    }

    /// `quantize_matrix` must reject a weight matrix that contains `NaN`.
    ///
    /// Unlike `±inf`, NaN does NOT propagate through the `> max_abs` comparison
    /// in IEEE 754 — `NaN > max_abs` evaluates to `false` — so a NaN value
    /// never updates `max_abs`.  The scale then stays finite and the NaN element
    /// is silently quantized to 0 via Rust's saturating `f32 as i8` cast.
    ///
    /// Mutation check: removing the `!v.is_finite()` loop guard inside
    /// `quantize_matrix` makes this test return `Ok` (NaN quietly becomes 0)
    /// instead of `Err`, failing the `expect_err`-style match.
    #[test]
    fn test_quantize_matrix_rejects_nan_input() {
        // Row 0 is finite; row 1 contains NaN in lane 1.
        // Without the loop guard the NaN never updates max_abs, the scale is
        // finite, and the result is Ok (NaN → 0).  With the guard it is Err.
        let w = vec![
            1.0,
            2.0,
            3.0, // row 0: finite
            1.0,
            f32::NAN,
            0.0, // row 1: NaN lane
        ];
        match quantize_matrix(&w, 2, 3) {
            Err(InferenceError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("non-finite"),
                    "error must describe the non-finite value; got: {msg}"
                );
                assert!(
                    msg.contains("1"),
                    "error must identify the offending row index; got: {msg}"
                );
            }
            Err(e) => panic!("expected InvalidInput for NaN input, got: {e}"),
            Ok(_) => panic!("expected Err for NaN input, got Ok"),
        }
    }

    /// `quantize_gdn_weights` propagates the non-finite-scale error from
    /// `quantize_matrix`. A single `+inf` value in `in_proj_qkv` is enough.
    ///
    /// Mutation check: removing the `!scale.is_finite()` guard in `quantize_matrix`
    /// causes this to return `Ok`, failing `expect_err`.
    #[test]
    fn test_quantize_gdn_weights_rejects_nonfinite_scale() {
        let mut w = valid_gdn_weights(3, 8);
        // row 0 of in_proj_qkv gets max_abs = +inf → scale = +inf / 127 = +inf
        w.in_proj_qkv[0] = f32::INFINITY;
        match quantize_gdn_weights(&w) {
            Err(InferenceError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("non-finite"),
                    "error must describe the non-finite scale; got: {msg}"
                );
            }
            Err(e) => panic!("expected InvalidInput for non-finite scale, got: {e}"),
            Ok(_) => panic!("expected Err for non-finite source weight, got Ok"),
        }
    }

    #[test]
    fn test_quantize_common_weights_dense_returns_ok() {
        use crate::model::qwen35::{CommonLayerWeights, DenseFfnWeights, FeedForwardWeights};

        let hidden = 4usize;
        let inter = 2usize;

        let dense_common = CommonLayerWeights {
            input_layernorm: vec![0.0f32; hidden],
            post_attention_layernorm: vec![0.0f32; hidden],
            ffn: FeedForwardWeights::Dense(DenseFfnWeights {
                gate_proj: vec![0.0f32; inter * hidden],
                up_proj: vec![0.0f32; inter * hidden],
                down_proj: vec![0.0f32; hidden * inter],
            }),
        };

        let result = quantize_common_weights(&dense_common);
        assert!(
            result.is_ok(),
            "expected Ok for dense layer, got: {result:?}"
        );
    }
}
