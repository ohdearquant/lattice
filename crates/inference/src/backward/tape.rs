//! Forward-cache structs and helper forwards for RMSNorm and SwiGLU.
// Forward pass that caches all activations needed for the backward pass.
// The scope for this milestone: only layer-23 (the last GQA layer) is trained.
// Layers 0..23 are a frozen prefix — we run them forward and save the residual
// stream at layer-23's input. From there we cache everything needed to
// differentiate through layer-23 and the final head.

/// Cached activations for a single GQA layer (layer-23 in the full model).
pub struct LayerCache {
    /// Residual stream input before pre-attention RMSNorm: [hidden]
    pub residual_pre_attn: Vec<f32>,
    /// Pre-attention RMSNorm output (i.e. the normed hidden): [hidden]
    pub normed_pre_attn: Vec<f32>,
    /// inv_rms values for the pre-attention RMSNorm: scalar per token, stored [1]
    pub inv_rms_pre_attn: f32,
    /// Pre-FFN residual: [hidden]
    pub residual_pre_ffn: Vec<f32>,
    /// Normed pre-FFN: [hidden]
    pub normed_pre_ffn: Vec<f32>,
    /// inv_rms for pre-FFN norm: scalar
    pub inv_rms_pre_ffn: f32,
    /// Gate pre-activation (before silu) for SwiGLU: [inter]
    pub gate_pre: Vec<f32>,
    /// Up pre-activation: [inter]
    pub up_pre: Vec<f32>,
    /// Attention output (after o_proj) before residual add: [hidden]
    pub attn_out: Vec<f32>,
    /// FFN output (after down_proj) before residual add: [hidden]
    pub ffn_out: Vec<f32>,
}

/// Activations from the full sequence forward pass through one GQA layer.
/// Stored per-token so the backward can iterate.
pub struct SequenceLayerCache {
    pub tokens: Vec<LayerCache>,
    /// Q pre-rope per-token, packed: [seq_len * q_dim]
    pub q_pre_rope: Vec<f32>,
    /// K pre-rope per-token: [seq_len * kv_dim]
    pub k_pre_rope: Vec<f32>,
    /// V per-token: [seq_len * kv_dim]
    pub v: Vec<f32>,
    /// Softmax probs per position: vec of length seq_len, each element is
    /// a flat [num_q_heads * (t+1)] probs vector for position t.
    pub softmax_probs: Vec<Vec<f32>>,
    /// Context (after softmax-weighted sum, before o_proj): [seq_len * q_dim]
    pub context: Vec<f32>,
    /// h_q = A_q x per-token: [seq_len * rank]
    pub h_q: Vec<f32>,
    /// h_v = A_v x per-token: [seq_len * rank]
    pub h_v: Vec<f32>,
    /// Q after rope: per-token head layout [seq_len][q_dim]
    pub q_after_rope: Vec<Vec<f32>>,
}

/// Top-level activation tape covering the trained segment.
pub struct BackwardTape {
    /// Hidden state at the input of layer-23 (= output of layer-22 after residual).
    /// Shape [hidden] — we treat the prefix as frozen so we only need this boundary.
    pub layer23_input: Vec<f32>,
    /// Per-sequence cache for layer-23.
    pub layer23: SequenceLayerCache,
    /// Final RMSNorm normed output: [hidden]
    pub final_normed: Vec<f32>,
    /// inv_rms for the final norm: scalar
    pub inv_rms_final: f32,
    /// Logits: [vocab_size]
    pub logits: Vec<f32>,
}

/// RMSNorm forward + cache: returns (normed, inv_rms).
pub fn rms_norm_forward(x: &[f32], w: &[f32], eps: f32) -> (Vec<f32>, f32) {
    let d = x.len();
    let mean_sq: f32 = x.iter().map(|xi| xi * xi).sum::<f32>() / d as f32;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    let normed: Vec<f32> = x
        .iter()
        .zip(w.iter())
        .map(|(xi, wi)| xi * wi * inv_rms)
        .collect();
    (normed, inv_rms)
}

/// SwiGLU forward: returns (output [hidden], gate_pre [inter], up_pre [inter]).
pub fn swiglu_forward(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    hidden: usize,
    inter: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Release-active, overflow-first, oversized-scratch-allowed contract (ADR-080 C4, held
    // finding): gate_pre/up_pre are each an `A(1,hidden) @ B(inter,hidden)^T` matvec (`x`
    // plays A, `w_gate`/`w_up` play B). Previously this silently truncated a too-short `x`
    // via `.zip(x.iter())` (no panic, just a wrong, under-summed dot product) instead of
    // rejecting the shape like `matmul_bt`/the materialized GQA training forward do.
    crate::forward::cpu::validate_gemm_bt(
        x.len(),
        w_gate.len(),
        inter,
        1,
        hidden,
        inter,
        "swiglu_forward:gate",
    );
    crate::forward::cpu::validate_gemm_bt(
        x.len(),
        w_up.len(),
        inter,
        1,
        hidden,
        inter,
        "swiglu_forward:up",
    );
    let mut gate_pre = vec![0.0f32; inter];
    let mut up_pre = vec![0.0f32; inter];
    for i in 0..inter {
        gate_pre[i] = w_gate[i * hidden..(i + 1) * hidden]
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum();
        up_pre[i] = w_up[i * hidden..(i + 1) * hidden]
            .iter()
            .zip(x.iter())
            .map(|(a, b)| a * b)
            .sum();
    }

    // silu(gate) * up
    let mixed: Vec<f32> = gate_pre
        .iter()
        .zip(up_pre.iter())
        .map(|(&g, &u)| {
            let s = 1.0 / (1.0 + (-g).exp());
            g * s * u
        })
        .collect();

    // down_proj: `out = A(1,inter) @ B(hidden,inter)^T` (`mixed` plays A, `w_down` plays B).
    // `mixed` is freshly computed above at exactly `inter` elements, but `w_down` is
    // caller-supplied and gets the same release-active shape check as gate/up above.
    crate::forward::cpu::validate_gemm_bt(
        mixed.len(),
        w_down.len(),
        hidden,
        1,
        inter,
        hidden,
        "swiglu_forward:down",
    );
    let mut out = vec![0.0f32; hidden];
    for i in 0..hidden {
        out[i] = w_down[i * inter..(i + 1) * inter]
            .iter()
            .zip(mixed.iter())
            .map(|(a, b)| a * b)
            .sum();
    }
    (out, gate_pre, up_pre)
}

/// SwiGLU forward over a contiguous block of positions in one pass.
///
/// `x` holds the normalised input rows of `rows` positions, `[rows, hidden]` row-major;
/// the caller slices out exactly the positions it wants computed. Each of the three linear
/// maps is a single `matmul_bt` over all rows instead of `rows` separate matvecs.
///
/// Returns `(out [rows, hidden], gate_pre [rows, inter], up_pre [rows, inter])`. Row `i`
/// equals what [`swiglu_forward`] returns for `x[i * hidden..(i + 1) * hidden]`, up to
/// floating-point reassociation of the dot products.
pub fn swiglu_forward_seq(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    rows: usize,
    hidden: usize,
    inter: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Checked before the buffers below are sized from these products; `matmul_bt`
    // validates the operand lengths themselves (release-active).
    assert!(
        rows.checked_mul(inter).is_some() && rows.checked_mul(hidden).is_some(),
        "swiglu_forward_seq: shape overflow: rows*inter or rows*hidden"
    );
    if rows == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let mut gate_pre = vec![0.0f32; rows * inter];
    let mut up_pre = vec![0.0f32; rows * inter];
    crate::forward::cpu::matmul_bt(x, w_gate, &mut gate_pre, rows, hidden, inter);
    crate::forward::cpu::matmul_bt(x, w_up, &mut up_pre, rows, hidden, inter);

    // silu(gate) * up, same expression as `swiglu_forward`.
    let mixed: Vec<f32> = gate_pre
        .iter()
        .zip(up_pre.iter())
        .map(|(&g, &u)| {
            let s = 1.0 / (1.0 + (-g).exp());
            g * s * u
        })
        .collect();

    let mut out = vec![0.0f32; rows * hidden];
    crate::forward::cpu::matmul_bt(&mixed, w_down, &mut out, rows, inter, hidden);
    (out, gate_pre, up_pre)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_forward_roundtrip() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let w = vec![1.0f32; 4];
        let (normed, inv_rms) = rms_norm_forward(&x, &w, 1e-6);
        let mean_sq: f32 = x.iter().map(|xi| xi * xi).sum::<f32>() / 4.0;
        let expected_inv = 1.0 / (mean_sq + 1e-6f32).sqrt();
        assert!((inv_rms - expected_inv).abs() < 1e-6, "inv_rms mismatch");
        let expected_norm: Vec<f32> = x.iter().map(|xi| xi * expected_inv).collect();
        for (a, b) in normed.iter().zip(expected_norm.iter()) {
            assert!((a - b).abs() < 1e-5, "normed mismatch {a} vs {b}");
        }
    }

    #[test]
    fn swiglu_forward_smoke() {
        let hidden = 2;
        let inter = 3;
        let x = vec![1.0f32, -0.5];
        let w_gate = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
        let w_up = vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];
        let w_down = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let (out, gate_pre, up_pre) = swiglu_forward(&x, &w_gate, &w_up, &w_down, hidden, inter);
        assert_eq!(out.len(), hidden);
        assert_eq!(gate_pre.len(), inter);
        assert_eq!(up_pre.len(), inter);
    }

    // ADR-080 C4 held finding: `swiglu_forward` previously truncated a too-short `x` via
    // `.zip(x.iter())` silently instead of rejecting the shape like `matmul_bt` does.
    #[test]
    #[should_panic(expected = "a too short for m*k")]
    fn swiglu_forward_rejects_short_activation() {
        let hidden = 2;
        let inter = 3;
        let x = vec![1.0f32]; // too short: needs hidden = 2
        let w_gate = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
        let w_up = vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];
        let w_down = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let _ = swiglu_forward(&x, &w_gate, &w_up, &w_down, hidden, inter);
    }

    #[test]
    #[should_panic(expected = "b too short for n*k")]
    fn swiglu_forward_rejects_short_w_gate() {
        let hidden = 2;
        let inter = 3;
        let x = vec![1.0f32, -0.5];
        let w_gate = vec![1.0f32, 0.0]; // too short: needs inter*hidden = 6
        let w_up = vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];
        let w_down = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let _ = swiglu_forward(&x, &w_gate, &w_up, &w_down, hidden, inter);
    }

    #[test]
    #[should_panic(expected = "b too short for n*k")]
    fn swiglu_forward_rejects_short_w_down() {
        let hidden = 2;
        let inter = 3;
        let x = vec![1.0f32, -0.5];
        let w_gate = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
        let w_up = vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];
        let w_down = vec![1.0f32, 0.0]; // too short: needs hidden*inter = 6
        let _ = swiglu_forward(&x, &w_gate, &w_up, &w_down, hidden, inter);
    }

    #[test]
    fn swiglu_forward_accepts_oversized_activation() {
        let hidden = 2;
        let inter = 3;
        let x = vec![1.0f32, -0.5, 99.0]; // oversized by 1
        let w_gate = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
        let w_up = vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];
        let w_down = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let (out, gate_pre, up_pre) = swiglu_forward(&x, &w_gate, &w_up, &w_down, hidden, inter);
        assert_eq!(out.len(), hidden);
        assert_eq!(gate_pre.len(), inter);
        assert_eq!(up_pre.len(), inter);
    }

    // Sequence-level forward vs the single-position `swiglu_forward` reference. `matmul_bt`
    // reassociates the dot products, so parity is a relative tolerance, not bit-exact; the
    // tolerance and error formula match the parity tests in `ops.rs`.
    const PARITY_TOL: f64 = 1e-4;

    fn xorshift_fill(seed: u64, n: usize, amp: f32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * amp
            })
            .collect()
    }

    fn rel_err(reference: &[f32], got: &[f32]) -> f64 {
        assert_eq!(reference.len(), got.len(), "length mismatch");
        let diff_sq: f64 = reference
            .iter()
            .zip(got.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum();
        let norm_sq: f64 = reference.iter().map(|&a| (a as f64).powi(2)).sum();
        (diff_sq / norm_sq.max(1e-30)).sqrt()
    }

    /// Runs `swiglu_forward_seq` on rows `start..end` of a larger `[seq, hidden]` input and
    /// compares every returned row with `swiglu_forward` on the same position.
    fn assert_seq_matches_per_position(start: usize, end: usize) {
        // Odd, unequal dims: a transposed or swapped operand cannot pass by symmetry.
        let (seq, hidden, inter) = (11usize, 37usize, 53usize);
        // `xorshift_fill` forces the seed odd, so the seeds here are distinct odd numbers:
        // adjacent even/odd seeds would give gate and up the identical matrix, and a swap
        // of the two would then pass unnoticed.
        let x = xorshift_fill(101, seq * hidden, 1.0);
        let w_gate = xorshift_fill(103, inter * hidden, 0.3);
        let w_up = xorshift_fill(105, inter * hidden, 0.3);
        let w_down = xorshift_fill(107, hidden * inter, 0.3);

        let rows = end - start;
        let (out, gate_pre, up_pre) = swiglu_forward_seq(
            &x[start * hidden..end * hidden],
            &w_gate,
            &w_up,
            &w_down,
            rows,
            hidden,
            inter,
        );
        assert_eq!(out.len(), rows * hidden);
        assert_eq!(gate_pre.len(), rows * inter);
        assert_eq!(up_pre.len(), rows * inter);

        let mut ref_out = Vec::new();
        let mut ref_gate = Vec::new();
        let mut ref_up = Vec::new();
        for t in start..end {
            let (o, g, u) = swiglu_forward(
                &x[t * hidden..(t + 1) * hidden],
                &w_gate,
                &w_up,
                &w_down,
                hidden,
                inter,
            );
            ref_out.extend_from_slice(&o);
            ref_gate.extend_from_slice(&g);
            ref_up.extend_from_slice(&u);
        }

        assert!(
            rel_err(&ref_gate, &ref_up) > 0.1,
            "reference gate_pre and up_pre must differ, or a gate/up swap is undetectable"
        );
        for (name, reference, got) in [
            ("out", &ref_out, &out),
            ("gate_pre", &ref_gate, &gate_pre),
            ("up_pre", &ref_up, &up_pre),
        ] {
            assert!(
                reference.iter().any(|v| v.abs() > 1e-3),
                "{name}: reference is all ~0, comparison would be vacuous"
            );
            let err = rel_err(reference, got);
            eprintln!("swiglu_forward_seq {start}..{end} {name} rel_err={err:.2e}");
            assert!(
                err < PARITY_TOL,
                "swiglu_forward_seq {start}..{end} {name} vs per-position rel_err {err:.2e} >= {PARITY_TOL:.2e}"
            );
        }
    }

    #[test]
    fn swiglu_forward_seq_parity_full_range() {
        assert_seq_matches_per_position(0, 11);
    }

    // The terminal-layer shape: a block that does not start at position 0 and stops short of
    // the end of the sequence.
    #[test]
    fn swiglu_forward_seq_parity_range_not_starting_at_zero() {
        assert_seq_matches_per_position(4, 10);
    }

    #[test]
    fn swiglu_forward_seq_parity_single_row_at_last_position() {
        assert_seq_matches_per_position(10, 11);
    }

    #[test]
    fn swiglu_forward_seq_empty_block_returns_empty() {
        let (out, gate_pre, up_pre) = swiglu_forward_seq(&[], &[], &[], &[], 0, 4, 6);
        assert!(out.is_empty() && gate_pre.is_empty() && up_pre.is_empty());
    }

    #[test]
    #[should_panic(expected = "a too short for m*k")]
    fn swiglu_forward_seq_rejects_short_activation() {
        let (hidden, inter) = (2usize, 3usize);
        let x = vec![1.0f32, -0.5, 0.25]; // 2 rows need 4 values
        let w_gate = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
        let w_up = vec![0.5f32; 6];
        let w_down = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let _ = swiglu_forward_seq(&x, &w_gate, &w_up, &w_down, 2, hidden, inter);
    }
}
