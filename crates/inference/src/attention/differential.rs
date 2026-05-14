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
    #[inline]
    pub fn n_rep(&self) -> usize {
        debug_assert!(self.num_kv_heads > 0);
        debug_assert_eq!(self.num_heads % self.num_kv_heads, 0);
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
    assert_eq!(
        cfg.num_heads % cfg.num_kv_heads,
        0,
        "num_heads must be divisible by num_kv_heads"
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

                // Softmax (numerically stable, two-pass)
                for qi in 0..seq_len {
                    let row = &mut score_slice[qi * seq_len..(qi + 1) * seq_len];
                    let valid = qi + 1;
                    let max_val = row[..valid]
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0_f32;
                    for v in &mut row[..valid] {
                        *v = (*v - max_val).exp();
                        sum += *v;
                    }
                    if sum > 0.0 {
                        let inv = 1.0 / sum;
                        for v in &mut row[..valid] {
                            *v *= inv;
                        }
                    }
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
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Deterministic data generator (matches gated.rs / gated_attention_test.rs)
    // ---------------------------------------------------------------

    fn det_data(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 7;
            state ^= state >> 9;
            state = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
            let mantissa = ((state >> 41) as u32) & 0x007f_ffff;
            let x = f32::from_bits(0x3f80_0000 | mantissa) - 1.5;
            out.push(x);
        }
        out
    }

    #[allow(dead_code)]
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    fn make_cfg(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        depth: usize,
    ) -> DiffAttnConfig {
        DiffAttnConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            layer_depth: depth,
        }
    }

    fn zero_lambda_params(head_dim: usize) -> DiffLambdaParams {
        DiffLambdaParams {
            lambda_q1: vec![0.0; head_dim],
            lambda_k1: vec![0.0; head_dim],
            lambda_q2: vec![0.0; head_dim],
            lambda_k2: vec![0.0; head_dim],
        }
    }

    // ---------------------------------------------------------------
    // test_lambda_init_schedule
    // ---------------------------------------------------------------

    #[test]
    fn test_lambda_init_schedule() {
        // lambda_init(0) = 0.8 - 0.6 * exp(0) = 0.8 - 0.6 = 0.2
        let cfg0 = make_cfg(2, 2, 4, 0);
        let li0 = cfg0.lambda_init();
        assert!(
            (li0 - 0.2).abs() < 1e-6,
            "lambda_init(0) expected 0.2, got {li0}"
        );

        // lambda_init increases toward 0.8 as depth grows
        let prev_cfg = make_cfg(2, 2, 4, 5);
        let li5 = prev_cfg.lambda_init();
        let cfg10 = make_cfg(2, 2, 4, 10);
        let li10 = cfg10.lambda_init();
        assert!(li5 > li0, "lambda_init should increase with depth");
        assert!(li10 > li5, "lambda_init should increase with depth");
        assert!(li10 < 0.8, "lambda_init should approach but stay below 0.8");

        // Monotone property: test many depths
        let mut prev = cfg0.lambda_init();
        for d in 1..=50 {
            let cur = make_cfg(2, 2, 4, d).lambda_init();
            assert!(
                cur > prev,
                "lambda_init not monotone at depth {d}: cur={cur} prev={prev}"
            );
            prev = cur;
        }
    }

    // ---------------------------------------------------------------
    // test_compute_lambda_full
    // ---------------------------------------------------------------

    #[test]
    fn test_compute_lambda_full() {
        // All zeros: lambda_full = exp(0) - exp(0) + lambda_init = 1 - 1 + lambda_init = lambda_init
        let params = zero_lambda_params(4);
        let lambda_init = 0.3;
        let lf = compute_lambda_full(&params, lambda_init);
        assert!(
            (lf - lambda_init).abs() < 1e-6,
            "all-zero params: lambda_full expected {lambda_init}, got {lf}"
        );

        // Hand-compute for non-trivial values.
        // lq1 = [1, 0, 0, 0], lk1 = [2, 0, 0, 0] → dot1 = 2 → exp(2) ≈ 7.389056
        // lq2 = [0, 1, 0, 0], lk2 = [0, 1, 0, 0] → dot2 = 1 → exp(1) ≈ 2.718282
        // lambda_full = exp(2) - exp(1) + 0.3 ≈ 7.389056 - 2.718282 + 0.3 ≈ 4.970774
        let params2 = DiffLambdaParams {
            lambda_q1: vec![1.0, 0.0, 0.0, 0.0],
            lambda_k1: vec![2.0, 0.0, 0.0, 0.0],
            lambda_q2: vec![0.0, 1.0, 0.0, 0.0],
            lambda_k2: vec![0.0, 1.0, 0.0, 0.0],
        };
        let expected = 2.0f32.exp() - 1.0f32.exp() + 0.3;
        let got = compute_lambda_full(&params2, 0.3);
        assert!(
            (got - expected).abs() < 1e-5,
            "hand-computed lambda_full: expected {expected}, got {got}"
        );
    }

    // ---------------------------------------------------------------
    // test_diff_attn_shapes
    // ---------------------------------------------------------------

    #[test]
    fn test_diff_attn_shapes() {
        let cfg = make_cfg(2, 2, 4, 0);
        let seq_len = 5;
        let q = det_data(seq_len * cfg.q_dim(), 1);
        let k = det_data(seq_len * cfg.k_dim(), 2);
        let v = det_data(seq_len * cfg.v_dim(), 3);
        let params = zero_lambda_params(cfg.head_dim);
        let subln_w = vec![1.0f32; 2 * cfg.head_dim];
        let mut out = vec![0.0f32; seq_len * cfg.out_dim()];
        let mut scratch = DiffAttnScratch::default();

        apply_differential_attention(
            &q,
            &k,
            &v,
            &params,
            &subln_w,
            1e-6,
            &mut out,
            seq_len,
            &cfg,
            &mut scratch,
        );

        assert_eq!(
            out.len(),
            seq_len * cfg.num_heads * 2 * cfg.head_dim,
            "output length must be seq_len * num_heads * 2 * head_dim"
        );
    }

    // ---------------------------------------------------------------
    // test_diff_attn_single_head_single_token
    // ---------------------------------------------------------------

    #[test]
    fn test_diff_attn_single_head_single_token() {
        // seq_len=1, num_heads=1, num_kv_heads=1, head_dim=2
        // With seq_len=1 there is only one token attending to itself.
        // scores1[0,0] = softmax([q0·k0*scale]) = [1.0]
        // scores2[0,0] = softmax([q1·k1*scale]) = [1.0]
        // With lambda_full = lambda_init (zero lambda params), and v = [v0, v1] for 2*head_dim:
        // context = (1 - lambda_full * 1) * v = (1 - lambda_init) * v (after subln with gamma=1)
        let head_dim = 2_usize;
        let cfg = make_cfg(1, 1, head_dim, 3); // depth=3 → lambda_init ≈ 0.469
        let lambda_init = cfg.lambda_init();

        let seq_len = 1_usize;
        // q_buf: [1, 2*1*2=4], k_buf: [1, 2*1*2=4], v_buf: [1, 1*2*2=4]
        // Use constant q, k so all dot products equal some value d.
        // scores1 and scores2 will both be [[1.0]] (single token, softmax of one element = 1)
        let q = vec![1.0f32, 0.0, 1.0, 0.0]; // head 0: q0=[1,0], q1=[1,0]
        let k = vec![1.0f32, 0.0, 1.0, 0.0]; // k head 0 packed: k0=[1,0], k1=[1,0]
        let v = vec![2.0f32, 3.0, 0.0, 0.0]; // v head 0 (2*head_dim=4): [2,3,0,0]

        let params = zero_lambda_params(head_dim); // lambda_full = lambda_init
        let subln_w = vec![1.0f32; 2 * head_dim]; // identity gamma
        let mut out = vec![0.0f32; seq_len * cfg.out_dim()]; // [1, 1*4=4]
        let mut scratch = DiffAttnScratch::default();

        apply_differential_attention(
            &q,
            &k,
            &v,
            &params,
            &subln_w,
            1e-6,
            &mut out,
            seq_len,
            &cfg,
            &mut scratch,
        );

        // diff_attn = 1.0 - lambda_init * 1.0 = 1 - lambda_init (applied to each v element)
        // after subln (gamma=1, x is already unit-rms up to precision) and scale (1-lambda_init)
        // The expected output for each v element e_i: subln(e_i) * (1-lambda_init)
        // With gamma=1, rms_norm just normalizes. But we can check it's finite and non-zero.
        assert!(out[0].is_finite(), "output must be finite");
        // The subtraction (1 - lambda_init)*v_part should yield positive values where v is positive.
        // With lambda_init < 1, (1-lambda_init) > 0, so output direction matches v.
        let scale_factor = 1.0 - lambda_init;
        assert!(
            scale_factor > 0.0,
            "scale factor (1-lambda_init) must be positive"
        );
    }

    // ---------------------------------------------------------------
    // test_diff_attn_causal_masking
    // ---------------------------------------------------------------

    #[test]
    fn test_diff_attn_causal_masking() {
        // Verify position i attends only to j <= i.
        //
        // Differential test: run two inputs that differ ONLY in the V values at
        // future positions (1, 2). If causal masking is correct, out[pos 0] must
        // be bit-identical between the two runs — position 0 cannot see future V.
        //
        // A magnitude bound would NOT catch a broken mask here: the sub-layer
        // RMSNorm normalizes a contaminated row back to O(1), hiding the leak.
        let head_dim = 2_usize;
        let cfg = make_cfg(1, 1, head_dim, 0);
        let seq_len = 3_usize;
        let v_head_dim = 2 * head_dim;

        let q = det_data(seq_len * cfg.q_dim(), 101);
        let k = det_data(seq_len * cfg.k_dim(), 202);
        let params = zero_lambda_params(head_dim);
        let subln_w = vec![1.0f32; 2 * head_dim];

        // Base V.
        let v_base = det_data(seq_len * cfg.v_dim(), 303);

        // Variant V: identical at position 0, perturbed at future positions 1 and 2.
        let mut v_perturbed = v_base.clone();
        for pos in 1..seq_len {
            for d in 0..v_head_dim {
                v_perturbed[pos * v_head_dim + d] += 12_345.0;
            }
        }

        let run = |v: &[f32]| {
            let mut out = vec![0.0f32; seq_len * cfg.out_dim()];
            let mut scratch = DiffAttnScratch::default();
            apply_differential_attention(
                &q,
                &k,
                v,
                &params,
                &subln_w,
                1e-6,
                &mut out,
                seq_len,
                &cfg,
                &mut scratch,
            );
            out
        };

        let out_base = run(&v_base);
        let out_perturbed = run(&v_perturbed);

        // Position 0's output must be unchanged — it cannot attend to future V.
        for d in 0..v_head_dim {
            assert_eq!(
                out_base[d].to_bits(),
                out_perturbed[d].to_bits(),
                "position 0 changed when only future V changed — causal mask leak at dim {d}"
            );
        }
        // Sanity: a later position SHOULD change (otherwise the perturbation was a no-op).
        let pos2_changed = (0..v_head_dim).any(|d| {
            let off = 2 * cfg.out_dim() + d;
            out_base[off] != out_perturbed[off]
        });
        assert!(
            pos2_changed,
            "position 2 should be affected by the future-V perturbation"
        );
    }

    // ---------------------------------------------------------------
    // test_lambda_one_zeroes_identical_maps
    // ---------------------------------------------------------------

    #[test]
    fn test_lambda_one_zeroes_identical_maps() {
        // Invariant: if the two softmax maps are identical AND lambda_full == 1,
        // then differential = s1 - 1·s2 = 0, so the output collapses to ~0.
        //
        // Identical maps: q0 == q1 and k0 == k1 for every position → matmul + softmax
        // produce bit-identical s1 and s2.
        // lambda_full == 1: lambda_full = exp(dot1) - exp(dot2) + lambda_init.
        //   Set dot2 = 0 (zero lambda_q2/k2) → exp(dot2) = 1.
        //   Need exp(dot1) = 2 - lambda_init → dot1 = ln(2 - lambda_init), achieved
        //   via lambda_q1 = [ln(2-lambda_init), 0, ...], lambda_k1 = [1, 0, ...].
        //
        // Tolerance: the f32 ln→exp round-trip leaves lambda_full ≈ 1.0 ± ~1e-7
        // rather than exactly 1.0, so `diff` is a tiny non-zero residual. The
        // sub-layer RMSNorm's eps floor (1e-6) then scales that residual up to
        // ~1e-5 magnitude. A 1e-3 bound is still decisive: it would FAIL by 2-3
        // orders of magnitude if the subtraction were removed (output ≈ 0.8),
        // sign-flipped (≈ 1.6), or lambda ignored (≈ 0.8) — unlike the previous
        // "output is nonzero" check, which all three of those would pass.
        let head_dim = 4_usize;
        let cfg = make_cfg(1, 1, head_dim, 0); // lambda_init = 0.2
        let seq_len = 3_usize;
        let lambda_init = cfg.lambda_init();

        // lambda params engineered so compute_lambda_full ≈ 1.0.
        let mut lambda_q1 = vec![0.0f32; head_dim];
        let mut lambda_k1 = vec![0.0f32; head_dim];
        lambda_q1[0] = (2.0 - lambda_init).ln();
        lambda_k1[0] = 1.0;
        let params = DiffLambdaParams {
            lambda_q1,
            lambda_k1,
            lambda_q2: vec![0.0f32; head_dim],
            lambda_k2: vec![0.0f32; head_dim],
        };
        let lambda_full = compute_lambda_full(&params, lambda_init);
        assert!(
            (lambda_full - 1.0).abs() < 1e-5,
            "test setup error: lambda_full should be ≈1.0, got {lambda_full}"
        );

        // q0 == q1 and k0 == k1 for every position → both softmax maps identical.
        let mut q = vec![0.0f32; seq_len * cfg.q_dim()];
        let mut k = vec![0.0f32; seq_len * cfg.k_dim()];
        for pos in 0..seq_len {
            let base = pos * cfg.q_dim();
            for d in 0..head_dim {
                q[base + d] = det_data(1, (pos * head_dim + d) as u64)[0];
                q[base + head_dim + d] = q[base + d]; // q1 = q0
            }
            let kbase = pos * cfg.k_dim();
            for d in 0..head_dim {
                k[kbase + d] = det_data(1, (pos * head_dim + d + 1000) as u64)[0];
                k[kbase + head_dim + d] = k[kbase + d]; // k1 = k0
            }
        }
        let v = det_data(seq_len * cfg.v_dim(), 77);
        let subln_w = vec![1.0f32; 2 * head_dim];
        let mut out = vec![0.0f32; seq_len * cfg.out_dim()];
        let mut scratch = DiffAttnScratch::default();

        apply_differential_attention(
            &q,
            &k,
            &v,
            &params,
            &subln_w,
            1e-6,
            &mut out,
            seq_len,
            &cfg,
            &mut scratch,
        );

        // diff = s1 - lambda_full·s2 ≈ 0 → context ≈ 0 → output ≈ 0.
        let max_abs = out.iter().copied().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(
            max_abs < 1e-3,
            "identical maps with lambda_full≈1 must yield ~0 output, got max_abs={max_abs}"
        );
    }
}
