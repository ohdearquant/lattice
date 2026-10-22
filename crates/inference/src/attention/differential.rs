//! Differential attention for the Differential Transformer (Ye et al., ICLR 2025).
//!
//! Implements the exact reference algorithm from Microsoft's
//! `unilm/Diff-Transformer/multihead_flashdiff_1.py`, adapted to Lattice conventions:
//!
//! - Q and K are split into two halves along the head dimension (`2*num_heads` packed heads).
//! - Two independent causal softmax attention maps are computed.
//! - The second map (scaled by a learnable `lambda_full`) is subtracted from the first.
//! - A sub-layer RMSNorm is applied over the `2*head_dim`-wide value context.
//! - The result is scaled by `(1 - lambda_init)` to compensate for the subtraction.
//!
//! **Caller responsibility**: Q and K must already have RoPE applied. Linear projections are
//! the caller's responsibility. This module owns the dual-softmax-subtract math, the lambda
//! computation, the sub-RMSNorm, and the final scale.
//!
//! See arXiv:2410.05258 for the full algorithm.

// ===================================================================
// Configuration
// ===================================================================

/// **Unstable**: configuration for one differential attention layer.
///
/// `embed_dim = 2 * num_heads * head_dim` in the reference architecture.
/// Q/K projections produce `2*num_heads` packed heads of `head_dim` each.
/// V projections produce `num_kv_heads` heads, each `2*head_dim` wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiffAttnConfig {
    /// Post-split logical head count (pairs). `num_heads` Q-pairs → `2*num_heads` packed Q heads.
    pub num_heads: usize,
    /// KV head count for GQA. `num_heads % num_kv_heads == 0` must hold.
    pub num_kv_heads: usize,
    /// Per-half head dimension. Each packed head (Q or K) has `head_dim` elements.
    pub head_dim: usize,
    /// 0-based layer index, used to compute `lambda_init`.
    pub layer_depth: usize,
}

impl DiffAttnConfig {
    /// **Unstable**: depth-scheduled lambda initializer: `0.8 - 0.6 * exp(-0.3 * depth)`.
    ///
    /// At depth 0: `0.8 - 0.6 = 0.2`. Increases monotonically toward 0.8.
    #[inline]
    pub fn lambda_init(&self) -> f32 {
        0.8 - 0.6 * (-0.3 * self.layer_depth as f32).exp()
    }

    /// **Unstable**: total packed Q-head count: `2 * num_heads`.
    #[inline]
    pub fn q_heads_packed(&self) -> usize {
        2 * self.num_heads
    }

    /// **Unstable**: total packed KV-head count: `2 * num_kv_heads`.
    #[inline]
    pub fn kv_heads_packed(&self) -> usize {
        2 * self.num_kv_heads
    }

    /// **Unstable**: GQA repeat factor per KV head: `num_heads / num_kv_heads`.
    ///
    /// # Panics
    ///
    /// Panics if `num_kv_heads == 0` or `num_heads` is not divisible by it. Real
    /// `assert!`s (not `debug_assert`) — this is a public method, so a release
    /// caller passing an invalid config must fail loudly rather than divide by
    /// zero or silently floor a non-divisible ratio.
    #[inline]
    pub fn n_rep(&self) -> usize {
        assert!(self.num_kv_heads > 0, "num_kv_heads must be > 0");
        assert_eq!(
            self.num_heads % self.num_kv_heads,
            0,
            "num_heads ({}) must be divisible by num_kv_heads ({})",
            self.num_heads,
            self.num_kv_heads
        );
        self.num_heads / self.num_kv_heads
    }

    /// **Unstable**: total Q-buffer dimension: `2 * num_heads * head_dim`.
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.q_heads_packed() * self.head_dim
    }

    /// **Unstable**: total K-buffer dimension: `2 * num_kv_heads * head_dim`.
    #[inline]
    pub fn k_dim(&self) -> usize {
        self.kv_heads_packed() * self.head_dim
    }

    /// **Unstable**: total V-buffer dimension: `num_kv_heads * 2 * head_dim`.
    /// V heads are `2*head_dim` wide (not split).
    #[inline]
    pub fn v_dim(&self) -> usize {
        self.num_kv_heads * 2 * self.head_dim
    }

    /// **Unstable**: output dimension: `num_heads * 2 * head_dim`.
    #[inline]
    pub fn out_dim(&self) -> usize {
        self.num_heads * 2 * self.head_dim
    }
}

// ===================================================================
// Lambda reparameterization
// ===================================================================

/// **Unstable**: learnable lambda reparameterization vectors for one layer.
///
/// `lambda_full = exp(lq1·lk1) - exp(lq2·lk2) + lambda_init`
///
/// Each vector has `head_dim` elements. Initialized near zero so that at init
/// `lambda_full ≈ lambda_init`.
pub struct DiffLambdaParams {
    /// Shape: `[head_dim]`
    pub lambda_q1: Vec<f32>,
    /// Shape: `[head_dim]`
    pub lambda_k1: Vec<f32>,
    /// Shape: `[head_dim]`
    pub lambda_q2: Vec<f32>,
    /// Shape: `[head_dim]`
    pub lambda_k2: Vec<f32>,
}

/// **Unstable**: compute `lambda_full = exp(lq1·lk1) - exp(lq2·lk2) + lambda_init`.
///
/// The dot products are over `head_dim`. The result can be any real value; it is
/// NOT clamped (negative lambda_full is valid and meaningful in the reference).
///
/// # Panics
///
/// Panics if the four lambda vectors do not all have equal length. A real
/// `assert_eq!` (not `debug_assert`) — a misloaded checkpoint with mismatched
/// shapes must fail loudly, not silently truncate via `zip()` in release.
#[inline]
pub fn compute_lambda_full(params: &DiffLambdaParams, lambda_init: f32) -> f32 {
    let len = params.lambda_q1.len();
    assert_eq!(params.lambda_k1.len(), len, "lambda_k1 length != lambda_q1");
    assert_eq!(params.lambda_q2.len(), len, "lambda_q2 length != lambda_q1");
    assert_eq!(params.lambda_k2.len(), len, "lambda_k2 length != lambda_q1");

    let dot1: f32 = params
        .lambda_q1
        .iter()
        .zip(params.lambda_k1.iter())
        .map(|(&q, &k)| q * k)
        .sum();
    let dot2: f32 = params
        .lambda_q2
        .iter()
        .zip(params.lambda_k2.iter())
        .map(|(&q, &k)| q * k)
        .sum();

    dot1.exp() - dot2.exp() + lambda_init
}

// ===================================================================
// Scratch buffers
// ===================================================================

/// **Unstable**: pre-allocated scratch buffers for differential attention; layout may grow.
#[derive(Default, Clone, Debug)]
pub struct DiffAttnScratch {
    /// Scores for the first softmax map (packed head 2h): `[num_heads, seq_len, seq_len]`.
    scores1: Vec<f32>,
    /// Scores for the second softmax map (packed head 2h+1): `[num_heads, seq_len, seq_len]`.
    scores2: Vec<f32>,
    /// V head transposed for one KV head: `[2*head_dim, seq_len]`.
    v_head_t: Vec<f32>,
    /// Extracted contiguous K packed head: `[seq_len, head_dim]`.
    k_packed: Vec<f32>,
    /// Extracted contiguous Q packed head: `[seq_len, head_dim]`.
    q_packed: Vec<f32>,
    /// Per-head context before subln: `[num_heads, seq_len, 2*head_dim]`.
    context: Vec<f32>,
}

impl DiffAttnScratch {
    /// **Unstable**: resize all scratch buffers to accommodate the given sequence length and config.
    pub fn reserve_for(&mut self, seq_len: usize, cfg: &DiffAttnConfig) {
        let n_pairs = cfg.num_heads;
        let v_head_dim = 2 * cfg.head_dim; // V is 2*head_dim wide per KV head
        self.scores1.resize(n_pairs * seq_len * seq_len, 0.0_f32);
        self.scores2.resize(n_pairs * seq_len * seq_len, 0.0_f32);
        self.v_head_t.resize(v_head_dim * seq_len, 0.0_f32);
        self.k_packed.resize(seq_len * cfg.head_dim, 0.0_f32);
        self.q_packed.resize(seq_len * cfg.head_dim, 0.0_f32);
        self.context.resize(n_pairs * seq_len * v_head_dim, 0.0_f32);
    }
}

// ===================================================================
// Core kernel
// ===================================================================

const MASK_VALUE: f32 = -10_000.0_f32;

/// **Unstable**: apply causal differential attention (prefill, multi-token).
///
/// Implements the exact DIFF Transformer algorithm from arXiv:2410.05258 §3.
///
/// # Buffer layouts
///
/// - `q_buf`: `[seq_len, 2*num_heads*head_dim]` — packed Q heads, interleaved per token
/// - `k_buf`: `[seq_len, 2*num_kv_heads*head_dim]` — packed K heads
/// - `v_buf`: `[seq_len, num_kv_heads*(2*head_dim)]` — V heads, each `2*head_dim` wide
/// - `subln_weight`: `[2*head_dim]` — RMSNorm gamma for the sub-layer norm
/// - `attn_out`: `[seq_len, num_heads*(2*head_dim)]` — output, interleaved per pair
///
/// Q and K must already have RoPE applied. Causal masking is applied internally via
/// additive `-10000.0` (ADR-010 design choice #2).
///
/// # Panics
///
/// Panics if buffer lengths are inconsistent with `seq_len` and `cfg`.
#[allow(clippy::too_many_arguments)]
pub fn apply_differential_attention(
    q_buf: &[f32],
    k_buf: &[f32],
    v_buf: &[f32],
    lambda_params: &DiffLambdaParams,
    subln_weight: &[f32],
    subln_eps: f32,
    attn_out: &mut [f32],
    seq_len: usize,
    cfg: &DiffAttnConfig,
    scratch: &mut DiffAttnScratch,
) {
    use crate::forward::cpu::matmul_bt;

    // Config validation — must precede the buffer-length asserts below. `cfg.q_dim()`,
    // `k_dim()`, etc. silently collapse to 0 when `num_heads` or `head_dim` is 0, which
    // would make those length checks vacuously pass; and `num_kv_heads == 0` would reach
    // a raw divide-by-zero in the `%` check and in `cfg.n_rep()`.
    assert!(cfg.num_heads > 0, "num_heads must be > 0");
    assert!(cfg.num_kv_heads > 0, "num_kv_heads must be > 0");
    assert!(cfg.head_dim > 0, "head_dim must be > 0");
    assert_eq!(
        cfg.num_heads % cfg.num_kv_heads,
        0,
        "num_heads ({}) must be divisible by num_kv_heads ({})",
        cfg.num_heads,
        cfg.num_kv_heads
    );

    assert_eq!(
        q_buf.len(),
        seq_len * cfg.q_dim(),
        "q_buf length mismatch: expected {} got {}",
        seq_len * cfg.q_dim(),
        q_buf.len()
    );
    assert_eq!(
        k_buf.len(),
        seq_len * cfg.k_dim(),
        "k_buf length mismatch: expected {} got {}",
        seq_len * cfg.k_dim(),
        k_buf.len()
    );
    assert_eq!(
        v_buf.len(),
        seq_len * cfg.v_dim(),
        "v_buf length mismatch: expected {} got {}",
        seq_len * cfg.v_dim(),
        v_buf.len()
    );
    assert_eq!(
        subln_weight.len(),
        2 * cfg.head_dim,
        "subln_weight length must be 2*head_dim"
    );
    assert_eq!(
        attn_out.len(),
        seq_len * cfg.out_dim(),
        "attn_out length mismatch: expected {} got {}",
        seq_len * cfg.out_dim(),
        attn_out.len()
    );
    for (name, v) in [
        ("lambda_q1", &lambda_params.lambda_q1),
        ("lambda_k1", &lambda_params.lambda_k1),
        ("lambda_q2", &lambda_params.lambda_q2),
        ("lambda_k2", &lambda_params.lambda_k2),
    ] {
        assert_eq!(
            v.len(),
            cfg.head_dim,
            "{name} length must equal head_dim ({}), got {}",
            cfg.head_dim,
            v.len()
        );
    }

    if seq_len == 0 {
        return;
    }

    scratch.reserve_for(seq_len, cfg);

    let lambda_init = cfg.lambda_init();
    let lambda_full = compute_lambda_full(lambda_params, lambda_init);
    let scale = (cfg.head_dim as f32).powf(-0.5);

    // n_pairs = cfg.num_heads (logical head pairs) — used for scratch reservation in reserve_for
    let head_dim = cfg.head_dim;
    let v_head_dim = 2 * head_dim;
    let n_rep = cfg.n_rep();

    // Strides for the packed buffers.
    // q_buf row: [2*num_heads * head_dim] — packed heads interleaved by token
    let q_row_stride = cfg.q_dim(); // 2 * num_heads * head_dim
    // k_buf row: [2*num_kv_heads * head_dim]
    let k_row_stride = cfg.k_dim(); // 2 * num_kv_heads * head_dim
    // v_buf row: [num_kv_heads * 2 * head_dim]
    let v_row_stride = cfg.v_dim(); // num_kv_heads * 2 * head_dim
    // attn_out row: [num_heads * 2 * head_dim]
    let out_row_stride = cfg.out_dim(); // num_heads * 2 * head_dim

    // For each logical pair h (0..n_pairs), packed heads 2h and 2h+1 form the pair.
    // With GQA, pair h maps to KV head h / n_rep. Each KV head covers n_rep pairs.
    // We iterate per KV head to share the K/V extraction.
    for kv_h in 0..cfg.num_kv_heads {
        // ------------------------------------------------------------------
        // Extract and transpose V for this KV head: [2*head_dim, seq_len]
        // V head kv_h occupies columns [kv_h * v_head_dim .. (kv_h+1) * v_head_dim]
        // ------------------------------------------------------------------
        let v_head_offset = kv_h * v_head_dim;
        let v_t = &mut scratch.v_head_t[..v_head_dim * seq_len];
        for pos in 0..seq_len {
            let src_off = pos * v_row_stride + v_head_offset;
            let v_row = &v_buf[src_off..src_off + v_head_dim];
            for d in 0..v_head_dim {
                v_t[d * seq_len + pos] = v_row[d];
            }
        }

        // For each of the n_rep Q pairs that share this KV head:
        let pair_start = kv_h * n_rep;
        let pair_end = pair_start + n_rep;

        for pair_h in pair_start..pair_end {
            // Packed Q head indices for this pair: 2*pair_h and 2*pair_h+1
            let q_h0 = 2 * pair_h; // first packed Q head index
            let q_h1 = 2 * pair_h + 1; // second packed Q head index
            // Packed K head indices for this KV head: 2*kv_h and 2*kv_h+1
            let k_h0 = 2 * kv_h;
            let k_h1 = 2 * kv_h + 1;

            // ------------------------------------------------------------------
            // scores1[pair_h]: Q head 2*pair_h @ K head 2*kv_h ^T
            // scores2[pair_h]: Q head 2*pair_h+1 @ K head 2*kv_h+1 ^T
            // ------------------------------------------------------------------
            for (q_ph, k_ph, scores_target) in [
                (q_h0, k_h0, &mut scratch.scores1),
                (q_h1, k_h1, &mut scratch.scores2),
            ] {
                // Extract contiguous Q packed head: [seq_len, head_dim]
                let q_packed = &mut scratch.q_packed[..seq_len * head_dim];
                for pos in 0..seq_len {
                    let src_off = pos * q_row_stride + q_ph * head_dim;
                    q_packed[pos * head_dim..pos * head_dim + head_dim]
                        .copy_from_slice(&q_buf[src_off..src_off + head_dim]);
                }

                // Extract contiguous K packed head: [seq_len, head_dim]
                let k_packed = &mut scratch.k_packed[..seq_len * head_dim];
                for pos in 0..seq_len {
                    let src_off = pos * k_row_stride + k_ph * head_dim;
                    k_packed[pos * head_dim..pos * head_dim + head_dim]
                        .copy_from_slice(&k_buf[src_off..src_off + head_dim]);
                }

                // GEMM: [seq_len, head_dim] @ [seq_len, head_dim]^T -> [seq_len, seq_len]
                let score_off = pair_h * seq_len * seq_len;
                let score_slice = &mut scores_target[score_off..score_off + seq_len * seq_len];
                matmul_bt(q_packed, k_packed, score_slice, seq_len, head_dim, seq_len);

                // Scale + additive causal mask
                for qi in 0..seq_len {
                    let row = &mut score_slice[qi * seq_len..(qi + 1) * seq_len];
                    for (ki, v) in row.iter_mut().enumerate() {
                        if ki <= qi {
                            *v *= scale;
                        } else {
                            *v = MASK_VALUE;
                        }
                    }
                }

                // Softmax (numerically stable, two-pass). ADR-080 C1: route the
                // fail-closed final decision through the shared row-finalizer
                // instead of the bare `sum > 0.0` check, which had no `else`
                // branch and left a NaN/+inf-poisoned row un-zeroed (#780).
                for qi in 0..seq_len {
                    let row = &mut score_slice[qi * seq_len..(qi + 1) * seq_len];
                    let valid = qi + 1;
                    let (max_val, any_nan) =
                        crate::attention::softmax_row::row_max_and_any_nan(&row[..valid]);
                    if crate::attention::softmax_row::row_fails_closed_pre_exp(max_val, any_nan) {
                        row.fill(0.0);
                        continue;
                    }
                    let mut sum = 0.0_f32;
                    for v in &mut row[..valid] {
                        *v = (*v - max_val).exp();
                        sum += *v;
                    }
                    crate::attention::softmax_row::finalize_row(&mut row[..valid], sum);
                    row[valid..].fill(0.0);
                }
            }

            // ------------------------------------------------------------------
            // Differential subtraction: attn_weights = scores1[pair_h] - lambda_full * scores2[pair_h]
            // Result is [seq_len, seq_len]. Can be negative — no clamp.
            // ------------------------------------------------------------------
            let s1_off = pair_h * seq_len * seq_len;
            let s2_off = pair_h * seq_len * seq_len;
            // Build the differential scores in place in scores1 (we no longer need s1 separately).
            for i in 0..(seq_len * seq_len) {
                scratch.scores1[s1_off + i] -= lambda_full * scratch.scores2[s2_off + i];
            }

            // ------------------------------------------------------------------
            // Context = differential_scores @ V^T: [seq_len, 2*head_dim]
            // scores1: [seq_len, seq_len] (row-major)
            // v_t: [2*head_dim, seq_len] (transposed V)
            // matmul_bt(A[M,K], B[N,K]^T, C[M,N]) computes A @ B^T
            // We want C[seq_len, v_head_dim] = scores[seq_len, seq_len] @ V[seq_len, v_head_dim]
            // Rewrite as C = scores @ V^T^T — but V is already transposed as v_t[v_head_dim, seq_len]
            // matmul_bt(scores[seq,seq], v_t[v_hd,seq], ctx[seq,v_hd], seq, seq, v_hd)
            // = scores @ v_t^T which is scores @ V  ✓
            // ------------------------------------------------------------------
            let ctx_off = pair_h * seq_len * v_head_dim;
            let ctx_slice = &mut scratch.context[ctx_off..ctx_off + seq_len * v_head_dim];
            let diff_scores = &scratch.scores1[s1_off..s1_off + seq_len * seq_len];
            matmul_bt(
                diff_scores,
                &scratch.v_head_t[..v_head_dim * seq_len],
                ctx_slice,
                seq_len,
                seq_len,
                v_head_dim,
            );

            // ------------------------------------------------------------------
            // Sub-layer RMSNorm: over 2*head_dim, applied per (head, position)
            // using crate rms_norm which handles num_tokens rows of `hidden` width.
            // ------------------------------------------------------------------
            // rms_norm(x, gamma, hidden, eps) operates row-by-row, treating
            // ctx_slice as [seq_len, v_head_dim] tokens.
            {
                use crate::forward::cpu::rms_norm;
                rms_norm(ctx_slice, subln_weight, v_head_dim, subln_eps);
            }

            // ------------------------------------------------------------------
            // Scale by (1 - lambda_init) and write to attn_out
            // ------------------------------------------------------------------
            let scale_factor = 1.0 - lambda_init;
            for pos in 0..seq_len {
                let src_off = pos * v_head_dim;
                let dst_off = pos * out_row_stride + pair_h * v_head_dim;
                let src = &ctx_slice[src_off..src_off + v_head_dim];
                let dst = &mut attn_out[dst_off..dst_off + v_head_dim];
                for (d, s) in dst.iter_mut().zip(src.iter()) {
                    *d = s * scale_factor;
                }
            }
        }
    }
}

// ===================================================================
// Tests
// ===================================================================

#[cfg(test)]
mod tests;
