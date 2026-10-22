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
use crate::weights::ingress::{IngestedTensor, validate_ingested_tensor};
use std::mem::size_of;

const Q8_MATRIX_SOURCE: &str = "in-memory Q8 quantization";
const Q8_GDN_SOURCE: &str = "GatedDeltaNet Q8 quantization";
const Q8_ATTENTION_SOURCE: &str = "full-attention Q8 quantization";
const Q8_FFN_SOURCE: &str = "dense FFN Q8 quantization";

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
/// Returns `Err(InvalidInput)` if the declared geometry overflows or does not
/// match the source length, any source value is `NaN` or `±inf`, or a derived
/// scale is non-positive or non-finite. Validation is performed through the
/// shared weight-ingress contract so failures identify the tensor and first
/// offending element or row.
pub fn quantize_matrix(w: &[f32], rows: usize, cols: usize) -> Result<Q8Matrix, InferenceError> {
    quantize_named_matrix(w, rows, cols, Q8_MATRIX_SOURCE, "matrix")
}

fn validate_source_geometry(
    len: usize,
    rows: usize,
    cols: usize,
    source: &str,
    tensor_name: &str,
) -> Result<(), InferenceError> {
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "{source}: tensor {tensor_name} shape [{rows}, {cols}] overflows usize element count"
        ))
    })?;
    if len != expected {
        return Err(InferenceError::InvalidInput(format!(
            "{source}: tensor {tensor_name} (Q8) source element count {len} does not match shape \
             [{rows}, {cols}] (expected {expected})"
        )));
    }
    Ok(())
}

/// Validate a declared tensor shape against the shape derived from `cfg` — the single source
/// of truth every downstream Q8 matmul indexes by. Never infer geometry from checkpoint data
/// (norm lengths, projection lengths); every caller here passes cfg-derived expectations.
fn validate_cfg_shape(
    rows: usize,
    expected_rows: usize,
    cols: usize,
    expected_cols: usize,
    source: &str,
    tensor_name: &str,
) -> Result<(), InferenceError> {
    if rows != expected_rows || cols != expected_cols {
        return Err(InferenceError::InvalidInput(format!(
            "{source}: tensor {tensor_name} shape [{rows}, {cols}] does not match config-derived \
             shape [{expected_rows}, {expected_cols}]"
        )));
    }
    Ok(())
}

/// Validate a 1-D vector length (a norm gamma) against the cfg-derived expected length.
///
/// `pub(crate)` so the NEON Q8 ingress path (`forward::neon_forward`) shares this check
/// with the CPU Q8 path instead of re-deriving its own length-comparison logic.
pub(crate) fn validate_cfg_len(
    len: usize,
    expected: usize,
    source: &str,
    tensor_name: &str,
) -> Result<(), InferenceError> {
    if len != expected {
        return Err(InferenceError::InvalidInput(format!(
            "{source}: tensor {tensor_name} length {len} does not match config-derived length \
             {expected}"
        )));
    }
    Ok(())
}

fn quantize_named_matrix(
    w: &[f32],
    rows: usize,
    cols: usize,
    source: &str,
    tensor_name: &str,
) -> Result<Q8Matrix, InferenceError> {
    validate_source_geometry(w.len(), rows, cols, source, tensor_name)?;

    let shape = [rows, cols];

    let mut data = Vec::with_capacity(w.len());
    let mut scales = Vec::with_capacity(rows);
    let mut has_nonfinite = false;

    // Single pass over the source matrix: finite-value validation is fused with the
    // max-abs scale computation (both already touch every element in row-major
    // order), instead of an earlier full pre-pass plus this loop re-reading the
    // same bytes. max(|x_i|) is order-independent and involves no FP accumulation,
    // so fusing cannot change the computed scale. A non-finite element still lets
    // `abs_v > max_abs` run (NaN comparisons are false, +-inf propagates), but the
    // resulting `max_abs`/`scale`/`data` are discarded below once `has_nonfinite`
    // triggers the exact pre-fusion diagnostic via a full re-scan (error path only).
    for row_idx in 0..rows {
        let start = row_idx * cols;
        let end = start + cols;
        let row = &w[start..end];

        let mut max_abs = 0.0f32;
        for &v in row {
            has_nonfinite |= !v.is_finite();
            let abs_v = v.abs();
            if abs_v > max_abs {
                max_abs = abs_v;
            }
        }

        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        scales.push(scale);

        for &v in row {
            let q = (v / scale).round().clamp(-128.0, 127.0) as i8;
            data.push(q);
        }
    }

    if has_nonfinite {
        // Re-scan (once, on the malformed-input path only) to reproduce the
        // exact shared-ingress diagnostic (offending element index and value).
        validate_ingested_tensor(IngestedTensor::q8_source(source, tensor_name, &shape, w))?;
        unreachable!("has_nonfinite implies validate_ingested_tensor rejects tensor {tensor_name}");
    }

    validate_ingested_tensor(IngestedTensor::q8(
        source,
        tensor_name,
        &shape,
        &data,
        &scales,
    ))?;

    Ok(Q8Matrix {
        data,
        scales,
        rows,
        cols,
    })
}

#[cfg(target_os = "macos")]
const TILE_N: usize = 64;

#[derive(Debug, Default)]
pub(crate) struct Q8MatmulScratch {
    #[cfg(target_os = "macos")]
    b_f32: Vec<f32>,
    #[cfg(target_os = "macos")]
    c_tile: Vec<f32>,
    #[cfg(test)]
    calls: usize,
}

impl Q8MatmulScratch {
    pub(crate) fn prepare(&mut self, max_m: usize, max_k: usize) -> Result<(), InferenceError> {
        #[cfg(target_os = "macos")]
        {
            // The tile scratch buffers are sized by `TILE_N * k` and `m * TILE_N`,
            // which are NOT covered by the `m*k` / `n*k` / `m*n` guards in the
            // caller: when `n < TILE_N` (including `n == 0`) these products can
            // overflow while the guarded ones do not, and a wrapped size would
            // under-allocate the scratch and drive an OOB scatter below. Guard
            // them on the same fail-closed contract.
            let b_len = TILE_N.checked_mul(max_k).ok_or_else(|| {
                InferenceError::InvalidInput("matmul shape overflow: TILE_N*k".into())
            })?;
            let c_len = max_m.checked_mul(TILE_N).ok_or_else(|| {
                InferenceError::InvalidInput("matmul shape overflow: m*TILE_N".into())
            })?;
            for (len, label) in [(b_len, "TILE_N*k"), (c_len, "m*TILE_N")] {
                if len
                    .checked_mul(size_of::<f32>())
                    .is_none_or(|bytes| bytes > isize::MAX as usize)
                {
                    return Err(InferenceError::InvalidInput(format!(
                        "matmul scratch byte extent exceeds isize::MAX: {label}"
                    )));
                }
            }
            if self.b_f32.len() < b_len {
                self.b_f32.resize(b_len, 0.0);
            }
            if self.c_tile.len() < c_len {
                self.c_tile.resize(c_len, 0.0);
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = (max_m, max_k);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn call_count(&self) -> usize {
        self.calls
    }

    #[cfg(test)]
    pub(crate) fn storage_snapshot(&self) -> [[usize; 3]; 2] {
        #[cfg(target_os = "macos")]
        {
            [
                [
                    self.b_f32.as_ptr() as usize,
                    self.b_f32.len(),
                    self.b_f32.capacity(),
                ],
                [
                    self.c_tile.as_ptr() as usize,
                    self.c_tile.len(),
                    self.c_tile.capacity(),
                ],
            ]
        }
        #[cfg(not(target_os = "macos"))]
        {
            [[0; 3]; 2]
        }
    }

    #[cfg(test)]
    pub(crate) fn poison(&mut self) {
        #[cfg(target_os = "macos")]
        {
            self.b_f32.fill(f32::NAN);
            self.c_tile.fill(f32::NAN);
        }
    }
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
    matmul_bt_q8_with_scratch(a, b_q, c, m, k, n, &mut Q8MatmulScratch::default());
}

pub(crate) fn matmul_bt_q8_with_scratch(
    a: &[f32],
    b_q: &Q8Matrix,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scratch: &mut Q8MatmulScratch,
) {
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
        if let Err(error) = scratch.prepare(m, k) {
            panic!("{error}");
        }
        let b_f32 = &mut scratch.b_f32[..TILE_N * k];
        let c_tile = &mut scratch.c_tile[..m * TILE_N];

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
        let _ = &scratch;
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

    #[cfg(test)]
    {
        scratch.calls += 1;
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
                    Q8AttentionWeights::Linear(quantize_gdn_weights(gdn, cfg)?)
                }
                AttentionWeights::Full(full) => {
                    Q8AttentionWeights::Full(quantize_full_attn_weights(full, cfg)?)
                }
            };
            let q8_common = quantize_common_weights(common, cfg)?;
            Ok((q8_attn, q8_common))
        })
        .collect::<Result<Vec<_>, InferenceError>>()?;

    Ok(Q8ModelWeights {
        embed_tokens: weights.embed_tokens.clone(),
        final_norm: weights.final_norm.clone(),
        layers,
    })
}

/// Validate every GatedDeltaNet projection's declared shape against `cfg` — the same
/// `linear_qkv_dim()` / `linear_output_dim()` / `linear_num_value_heads()` / `hidden_size`
/// accessors the CPU and NEON forward passes index by — instead of trusting the
/// self-reported `*_rows` / `*_cols` metadata a caller attached to `w`. Shared by
/// `quantize_gdn_weights` (CPU Q8) and the NEON `quantize_model` packing path so a
/// caller-supplied or checkpoint-derived `GatedDeltaNetWeights` incompatible with `cfg`
/// is rejected at ingress on both paths, not just one.
pub(crate) fn validate_gdn_shapes(
    w: &GatedDeltaNetWeights,
    cfg: &Qwen35Config,
) -> Result<(), InferenceError> {
    let hidden = cfg.hidden_size;
    let value_heads = cfg.linear_num_value_heads();
    let qkv_dim = cfg.checked_linear_qkv_dim()?;
    let output_dim = cfg.checked_linear_output_dim()?;

    validate_cfg_shape(
        w.in_proj_qkv_rows,
        qkv_dim,
        w.in_proj_qkv_cols,
        hidden,
        Q8_GDN_SOURCE,
        "in_proj_qkv",
    )?;
    validate_cfg_shape(
        w.in_proj_z_rows,
        output_dim,
        w.in_proj_z_cols,
        hidden,
        Q8_GDN_SOURCE,
        "in_proj_z",
    )?;
    validate_cfg_shape(
        w.in_proj_b_rows,
        value_heads,
        w.in_proj_b_cols,
        hidden,
        Q8_GDN_SOURCE,
        "in_proj_b",
    )?;
    validate_cfg_shape(
        w.in_proj_a_rows,
        value_heads,
        w.in_proj_a_cols,
        hidden,
        Q8_GDN_SOURCE,
        "in_proj_a",
    )?;
    validate_cfg_shape(
        w.out_proj_rows,
        hidden,
        w.out_proj_cols,
        output_dim,
        Q8_GDN_SOURCE,
        "out_proj",
    )?;

    validate_source_geometry(
        w.in_proj_qkv.len(),
        w.in_proj_qkv_rows,
        w.in_proj_qkv_cols,
        Q8_GDN_SOURCE,
        "in_proj_qkv",
    )?;
    validate_source_geometry(
        w.in_proj_z.len(),
        w.in_proj_z_rows,
        w.in_proj_z_cols,
        Q8_GDN_SOURCE,
        "in_proj_z",
    )?;
    validate_source_geometry(
        w.in_proj_b.len(),
        w.in_proj_b_rows,
        w.in_proj_b_cols,
        Q8_GDN_SOURCE,
        "in_proj_b",
    )?;
    validate_source_geometry(
        w.in_proj_a.len(),
        w.in_proj_a_rows,
        w.in_proj_a_cols,
        Q8_GDN_SOURCE,
        "in_proj_a",
    )?;
    validate_source_geometry(
        w.out_proj.len(),
        w.out_proj_rows,
        w.out_proj_cols,
        Q8_GDN_SOURCE,
        "out_proj",
    )?;

    validate_cfg_len(w.a_log.len(), value_heads, Q8_GDN_SOURCE, "a_log")?;
    validate_cfg_len(w.dt_bias.len(), value_heads, Q8_GDN_SOURCE, "dt_bias")?;

    // `w.conv_dim` is a separate self-reported field from `conv1d_weight`'s length; the NEON
    // forward path (`gdn_step_q8_neon`) indexes `qkv_proj`/`conv_buffer` through `0..conv_dim`,
    // so a `conv_dim` that disagrees with `qkv_dim` (even if `conv1d_weight.len()` is
    // internally consistent with it) must be rejected here, not discovered as an
    // out-of-bounds index on the first decoded token.
    if w.conv_dim != qkv_dim {
        return Err(InferenceError::InvalidInput(format!(
            "{Q8_GDN_SOURCE}: tensor conv_dim {} does not match config-derived linear_qkv_dim {qkv_dim}",
            w.conv_dim
        )));
    }

    let expected_conv_len = cfg.checked_linear_conv_len()?;
    validate_cfg_len(
        w.conv1d_weight.len(),
        expected_conv_len,
        Q8_GDN_SOURCE,
        "conv1d_weight",
    )?;
    validate_cfg_len(
        w.norm_weight.len(),
        cfg.linear_value_head_dim,
        Q8_GDN_SOURCE,
        "norm_weight",
    )?;

    // `a_log`/`dt_bias`/`conv1d_weight`/`norm_weight` are retained verbatim (never
    // quantized), so they never pass through `quantize_named_matrix`'s finite-value
    // scan. Without this, NaN/Inf here reaches the GDN decay/conv/gated-norm
    // recurrence and poisons state instead of failing closed. Checked here, in the
    // shape-validation choke point both `quantize_gdn_weights` (CPU Q8) and the NEON
    // `pack_gdn_weights` call before packing, so a caller-supplied or
    // checkpoint-derived `GatedDeltaNetWeights` with non-finite retained vectors is
    // rejected on both paths, not just one.
    validate_ingested_tensor(IngestedTensor::q8_source(
        Q8_GDN_SOURCE,
        "a_log",
        &[w.a_log.len()],
        &w.a_log,
    ))?;
    validate_ingested_tensor(IngestedTensor::q8_source(
        Q8_GDN_SOURCE,
        "dt_bias",
        &[w.dt_bias.len()],
        &w.dt_bias,
    ))?;
    validate_ingested_tensor(IngestedTensor::q8_source(
        Q8_GDN_SOURCE,
        "conv1d_weight",
        &[w.conv1d_weight.len()],
        &w.conv1d_weight,
    ))?;
    validate_ingested_tensor(IngestedTensor::q8_source(
        Q8_GDN_SOURCE,
        "norm_weight",
        &[w.norm_weight.len()],
        &w.norm_weight,
    ))?;

    Ok(())
}

/// **Unstable**: convert GatedDeltaNet weights to Q8; output type may change.
///
/// Quantize a `GatedDeltaNetWeights` bundle into its Q8 representation.
///
/// Returns `Err` (never panics) when any projection shape disagrees with `cfg` (shapes are
/// validated against config-derived dimensions using checked arithmetic — `conv_dim` must
/// equal `cfg.linear_qkv_dim()`), when data lengths disagree with declared `rows * cols`, or
/// when any element of `a_log`, `dt_bias`, `conv1d_weight`, or `norm_weight` is non-finite
/// (`NaN` or `±inf`) — these four vectors are retained as f32 rather than quantized, so
/// finiteness is checked explicitly instead of falling out of per-row Q8 scale computation.
pub fn quantize_gdn_weights(
    w: &GatedDeltaNetWeights,
    cfg: &Qwen35Config,
) -> Result<Q8GatedDeltaNetWeights, InferenceError> {
    // `validate_gdn_shapes` covers both the projection geometry and the finite-value
    // scan of the four retained (never-quantized) vectors — a_log, dt_bias,
    // conv1d_weight, norm_weight — so this is the only pass over those buffers, not a
    // separate scan on top of an earlier shape-only check.
    validate_gdn_shapes(w, cfg)?;

    let in_proj_qkv = quantize_named_matrix(
        &w.in_proj_qkv,
        w.in_proj_qkv_rows,
        w.in_proj_qkv_cols,
        Q8_GDN_SOURCE,
        "in_proj_qkv",
    )?;
    let in_proj_z = quantize_named_matrix(
        &w.in_proj_z,
        w.in_proj_z_rows,
        w.in_proj_z_cols,
        Q8_GDN_SOURCE,
        "in_proj_z",
    )?;
    let in_proj_b = quantize_named_matrix(
        &w.in_proj_b,
        w.in_proj_b_rows,
        w.in_proj_b_cols,
        Q8_GDN_SOURCE,
        "in_proj_b",
    )?;
    let in_proj_a = quantize_named_matrix(
        &w.in_proj_a,
        w.in_proj_a_rows,
        w.in_proj_a_cols,
        Q8_GDN_SOURCE,
        "in_proj_a",
    )?;
    let out_proj = quantize_named_matrix(
        &w.out_proj,
        w.out_proj_rows,
        w.out_proj_cols,
        Q8_GDN_SOURCE,
        "out_proj",
    )?;

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
/// so the matrix shapes are inferred from the config's attention topology using checked
/// arithmetic (`num_attention_heads * head_dim` etc. reject overflow instead of wrapping).
///
/// Returns `Err(InvalidInput)` if any weight matrix contains `NaN` or `±inf` values, if a
/// weight matrix's declared source length disagrees with its config-derived shape, or if
/// the config-derived dimensions overflow `usize`.
pub(crate) fn quantize_full_attn_weights(
    w: &FullAttentionLayerWeights,
    cfg: &Qwen35Config,
) -> Result<Q8FullAttentionLayerWeights, InferenceError> {
    let ((q_rows, hidden), (kv_rows, _), (_, _), (o_rows, o_cols)) =
        infer_full_attention_shapes(w, cfg)?;

    let q_proj = quantize_named_matrix(&w.q_proj, q_rows, hidden, Q8_ATTENTION_SOURCE, "q_proj")?;
    let k_proj = quantize_named_matrix(&w.k_proj, kv_rows, hidden, Q8_ATTENTION_SOURCE, "k_proj")?;
    let v_proj = quantize_named_matrix(&w.v_proj, kv_rows, hidden, Q8_ATTENTION_SOURCE, "v_proj")?;
    let o_proj = quantize_named_matrix(&w.o_proj, o_rows, o_cols, Q8_ATTENTION_SOURCE, "o_proj")?;

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
    cfg: &Qwen35Config,
) -> Result<Q8CommonLayerWeights, InferenceError> {
    let dense = match &w.ffn {
        FeedForwardWeights::Dense(d) => d,
        FeedForwardWeights::Moe(_) => {
            return Err(InferenceError::UnsupportedModel(
                "Q8 quantization is dense-only; MoE layers are not supported".to_string(),
            ));
        }
    };
    // Both RMSNorm vectors are validated against cfg.hidden_size: in release builds
    // qwen35_rms_norm zips gamma against the activation, so a short
    // post_attention_layernorm would silently leave trailing hidden values
    // unnormalized instead of failing loudly.
    let hidden = cfg.hidden_size;
    validate_cfg_len(
        w.input_layernorm.len(),
        hidden,
        Q8_FFN_SOURCE,
        "input_layernorm",
    )?;
    validate_cfg_len(
        w.post_attention_layernorm.len(),
        hidden,
        Q8_FFN_SOURCE,
        "post_attention_layernorm",
    )?;
    let ((gate_rows, _), (up_rows, _), (down_rows, down_cols)) = infer_dense_shapes(dense, cfg)?;

    let gate_proj = quantize_named_matrix(
        &dense.gate_proj,
        gate_rows,
        hidden,
        Q8_FFN_SOURCE,
        "gate_proj",
    )?;
    let up_proj = quantize_named_matrix(&dense.up_proj, up_rows, hidden, Q8_FFN_SOURCE, "up_proj")?;
    let down_proj = quantize_named_matrix(
        &dense.down_proj,
        down_rows,
        down_cols,
        Q8_FFN_SOURCE,
        "down_proj",
    )?;

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

type ShapePair = (usize, usize);

fn infer_dense_shapes(
    dense: &crate::model::qwen35::DenseFfnWeights,
    cfg: &Qwen35Config,
) -> Result<(ShapePair, ShapePair, ShapePair), InferenceError> {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    // `hidden == 0` is not caught by `validate_source_geometry` below when the source
    // tensors are also empty (0 == rows * 0 passes vacuously): an all-empty dense layer
    // would otherwise reach ingestion successfully and panic on the first forward pass,
    // where `qwen35_rms_norm` divides `x.len() / hidden`. Reject the zero hidden
    // dimension explicitly, as a typed error, before that division is ever reached.
    if hidden == 0 {
        return Err(InferenceError::InvalidInput(format!(
            "{Q8_FFN_SOURCE}: cfg.hidden_size must be > 0"
        )));
    }
    validate_source_geometry(
        dense.gate_proj.len(),
        inter,
        hidden,
        Q8_FFN_SOURCE,
        "gate_proj",
    )?;
    validate_source_geometry(dense.up_proj.len(), inter, hidden, Q8_FFN_SOURCE, "up_proj")?;
    validate_source_geometry(
        dense.down_proj.len(),
        hidden,
        inter,
        Q8_FFN_SOURCE,
        "down_proj",
    )?;
    Ok(((inter, hidden), (inter, hidden), (hidden, inter)))
}

fn infer_full_attention_shapes(
    w: &FullAttentionLayerWeights,
    cfg: &Qwen35Config,
) -> Result<(ShapePair, ShapePair, ShapePair, ShapePair), InferenceError> {
    // Every dimension below comes from `cfg` — the same accessors `matmul_bt_q8` and the
    // rest of the forward pass index by — never from checkpoint tensor lengths. A
    // checkpoint that disagrees with cfg is rejected here, at ingress, instead of
    // reaching a shape assertion mid-matmul on the first full-attention token.
    let head_dim = cfg.head_dim;
    let hidden = cfg.hidden_size;
    let q_dim = cfg.checked_full_q_dim()?;
    let kv_dim = cfg.checked_full_kv_dim()?;
    let q_rows = crate::model::qwen35_config::checked_double(q_dim, "full_q_dim")?;

    validate_cfg_len(w.q_norm.len(), head_dim, Q8_ATTENTION_SOURCE, "q_norm")?;
    validate_cfg_len(w.k_norm.len(), head_dim, Q8_ATTENTION_SOURCE, "k_norm")?;
    validate_source_geometry(
        w.q_proj.len(),
        q_rows,
        hidden,
        Q8_ATTENTION_SOURCE,
        "q_proj",
    )?;
    validate_source_geometry(
        w.k_proj.len(),
        kv_dim,
        hidden,
        Q8_ATTENTION_SOURCE,
        "k_proj",
    )?;
    validate_source_geometry(
        w.v_proj.len(),
        kv_dim,
        hidden,
        Q8_ATTENTION_SOURCE,
        "v_proj",
    )?;
    validate_source_geometry(w.o_proj.len(), hidden, q_dim, Q8_ATTENTION_SOURCE, "o_proj")?;

    Ok((
        (q_rows, hidden),
        (kv_dim, hidden),
        (kv_dim, hidden),
        (hidden, q_dim),
    ))
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
#[path = "q8_matmul_scratch_tests.rs"]
mod scratch_tests;

#[cfg(test)]
mod tests;
