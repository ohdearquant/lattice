//! Self-contained integration test for differential attention.
//!
//! # Reference
//!
//! HuggingFace `transformers` does not ship a DIFF Transformer model. The reference
//! is Microsoft's `unilm/Diff-Transformer/multihead_flashdiff_1.py` algorithm,
//! transcribed exactly to Rust in `ref_differential_attention` below. The implementation
//! under test (`apply_differential_attention`) is verified against this reference within
//! `max_abs_diff <= 1e-4`.
//!
//! # Buffer conventions
//!
//! - `q_buf`:  `[seq_len, 2*num_heads*head_dim]`
//! - `k_buf`:  `[seq_len, 2*num_kv_heads*head_dim]`
//! - `v_buf`:  `[seq_len, num_kv_heads*(2*head_dim)]`
//! - `out`:    `[seq_len, num_heads*(2*head_dim)]`
//!
//! Run:
//!   cargo test --test differential_attention_test

use lattice_inference::attention::differential::{
    DiffAttnConfig, DiffAttnScratch, DiffLambdaParams, apply_differential_attention,
    compute_lambda_full,
};
use lattice_inference::forward::cpu::matmul_bt;

// ===================================================================
// Deterministic PRNG (xorshift64 + float extraction)
//
// Matches the generator used in gated_attention_test.rs and all other
// integration tests in this crate.
// ===================================================================

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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in max_abs_diff");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

// ===================================================================
// Reference implementation
//
// A naive, unoptimized, exact-math transcription of the Microsoft
// DIFF Transformer algorithm (multihead_flashdiff_1.py).
// No approximations; uses f32::exp throughout.
// ===================================================================

/// Compute a single row of softmax over the causal prefix.
///
/// `row` has length `seq_len`; only indices `0..=qi` are valid (causal).
/// Invalid positions are masked with `-10000.0` before softmax.
fn ref_softmax_causal_row(row: &mut [f32], qi: usize) {
    let seq_len = row.len();
    for ki in 0..seq_len {
        if ki > qi {
            row[ki] = -10_000.0_f32;
        }
    }
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
    // Zero out the masked tail (they hold -10000 before exp; set to 0 after softmax).
    row[valid..].fill(0.0);
}

/// RMSNorm a single row in-place.
///
/// `row` has length `hidden = 2*head_dim`. `gamma` has the same length.
fn ref_rms_norm_row(row: &mut [f32], gamma: &[f32], eps: f32) {
    let hidden = row.len();
    let sum_sq: f32 = row.iter().map(|&x| x * x).sum();
    let rms = (sum_sq / hidden as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for (v, &g) in row.iter_mut().zip(gamma.iter()) {
        *v = *v * inv_rms * g;
    }
}

/// Reference: naive differential attention matching multihead_flashdiff_1.py.
///
/// Inputs:
/// - `q_buf`:  `[seq_len, 2*num_heads*head_dim]`
/// - `k_buf`:  `[seq_len, 2*num_kv_heads*head_dim]`
/// - `v_buf`:  `[seq_len, num_kv_heads*2*head_dim]`
/// - `subln_weight`: `[2*head_dim]` RMSNorm gamma
///
/// Output: `[seq_len, num_heads*2*head_dim]`
#[allow(clippy::too_many_arguments)]
fn ref_differential_attention(
    q_buf: &[f32],
    k_buf: &[f32],
    v_buf: &[f32],
    lambda_params: &DiffLambdaParams,
    subln_weight: &[f32],
    subln_eps: f32,
    seq_len: usize,
    cfg: &DiffAttnConfig,
) -> Vec<f32> {
    let num_heads = cfg.num_heads;
    let num_kv_heads = cfg.num_kv_heads;
    let head_dim = cfg.head_dim;
    let v_head_dim = 2 * head_dim;
    let n_rep = num_heads / num_kv_heads;

    let lambda_init = cfg.lambda_init();
    let lambda_full = compute_lambda_full(lambda_params, lambda_init);
    let scale = (head_dim as f32).powf(-0.5);

    // Packed buffer strides
    let q_stride = 2 * num_heads * head_dim; // q_buf row width
    let k_stride = 2 * num_kv_heads * head_dim; // k_buf row width
    let v_stride = num_kv_heads * v_head_dim; // v_buf row width
    let out_stride = num_heads * v_head_dim; // output row width

    let mut out = vec![0.0_f32; seq_len * out_stride];

    for pair_h in 0..num_heads {
        let kv_h = pair_h / n_rep;
        let q_h0 = 2 * pair_h; // first packed Q head
        let q_h1 = 2 * pair_h + 1; // second packed Q head
        let k_h0 = 2 * kv_h; // first packed K head
        let k_h1 = 2 * kv_h + 1; // second packed K head

        // Extract Q heads: [seq_len, head_dim]
        let mut q0 = vec![0.0_f32; seq_len * head_dim];
        let mut q1 = vec![0.0_f32; seq_len * head_dim];
        for pos in 0..seq_len {
            let src0 = pos * q_stride + q_h0 * head_dim;
            let src1 = pos * q_stride + q_h1 * head_dim;
            q0[pos * head_dim..pos * head_dim + head_dim]
                .copy_from_slice(&q_buf[src0..src0 + head_dim]);
            q1[pos * head_dim..pos * head_dim + head_dim]
                .copy_from_slice(&q_buf[src1..src1 + head_dim]);
        }

        // Extract K heads: [seq_len, head_dim]
        let mut k0 = vec![0.0_f32; seq_len * head_dim];
        let mut k1 = vec![0.0_f32; seq_len * head_dim];
        for pos in 0..seq_len {
            let src0 = pos * k_stride + k_h0 * head_dim;
            let src1 = pos * k_stride + k_h1 * head_dim;
            k0[pos * head_dim..pos * head_dim + head_dim]
                .copy_from_slice(&k_buf[src0..src0 + head_dim]);
            k1[pos * head_dim..pos * head_dim + head_dim]
                .copy_from_slice(&k_buf[src1..src1 + head_dim]);
        }

        // Extract V head: [seq_len, v_head_dim]
        let mut v_head = vec![0.0_f32; seq_len * v_head_dim];
        for pos in 0..seq_len {
            let src = pos * v_stride + kv_h * v_head_dim;
            v_head[pos * v_head_dim..pos * v_head_dim + v_head_dim]
                .copy_from_slice(&v_buf[src..src + v_head_dim]);
        }

        // Compute attention scores: [seq_len, seq_len] for each half
        let mut scores1 = vec![0.0_f32; seq_len * seq_len];
        let mut scores2 = vec![0.0_f32; seq_len * seq_len];

        // scores = Q @ K^T, scale + causal mask + softmax
        matmul_bt(&q0, &k0, &mut scores1, seq_len, head_dim, seq_len);
        matmul_bt(&q1, &k1, &mut scores2, seq_len, head_dim, seq_len);

        for qi in 0..seq_len {
            // scale
            for ki in 0..=qi {
                scores1[qi * seq_len + ki] *= scale;
                scores2[qi * seq_len + ki] *= scale;
            }
            ref_softmax_causal_row(&mut scores1[qi * seq_len..(qi + 1) * seq_len], qi);
            ref_softmax_causal_row(&mut scores2[qi * seq_len..(qi + 1) * seq_len], qi);
        }

        // Differential subtraction: scores1 - lambda_full * scores2
        for i in 0..(seq_len * seq_len) {
            scores1[i] -= lambda_full * scores2[i];
        }

        // Context: [seq_len, v_head_dim] = diff_scores @ V
        let mut context = vec![0.0_f32; seq_len * v_head_dim];
        for qi in 0..seq_len {
            for kj in 0..seq_len {
                let w = scores1[qi * seq_len + kj];
                for d in 0..v_head_dim {
                    context[qi * v_head_dim + d] += w * v_head[kj * v_head_dim + d];
                }
            }
        }

        // Sub-layer RMSNorm per position
        for qi in 0..seq_len {
            let row = &mut context[qi * v_head_dim..(qi + 1) * v_head_dim];
            ref_rms_norm_row(row, subln_weight, subln_eps);
        }

        // Scale by (1 - lambda_init) and write output
        let scale_factor = 1.0 - lambda_init;
        for pos in 0..seq_len {
            let src_off = pos * v_head_dim;
            let dst_off = pos * out_stride + pair_h * v_head_dim;
            for d in 0..v_head_dim {
                out[dst_off + d] = context[src_off + d] * scale_factor;
            }
        }
    }

    out
}

// ===================================================================
// Helper: build test config and random inputs
// ===================================================================

fn make_inputs(
    cfg: &DiffAttnConfig,
    seq_len: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, DiffLambdaParams, Vec<f32>) {
    let q = det_data(seq_len * cfg.q_dim(), seed);
    let k = det_data(seq_len * cfg.k_dim(), seed + 1);
    let v = det_data(seq_len * cfg.v_dim(), seed + 2);
    let params = DiffLambdaParams {
        lambda_q1: det_data(cfg.head_dim, seed + 3),
        lambda_k1: det_data(cfg.head_dim, seed + 4),
        lambda_q2: det_data(cfg.head_dim, seed + 5),
        lambda_k2: det_data(cfg.head_dim, seed + 6),
    };
    // Scale down lambda params so exp(dot) stays reasonable
    let scale_down = |v: Vec<f32>| v.into_iter().map(|x| x * 0.1).collect::<Vec<_>>();
    let params = DiffLambdaParams {
        lambda_q1: scale_down(params.lambda_q1),
        lambda_k1: scale_down(params.lambda_k1),
        lambda_q2: scale_down(params.lambda_q2),
        lambda_k2: scale_down(params.lambda_k2),
    };
    let subln_w = vec![1.0_f32; 2 * cfg.head_dim];
    (q, k, v, params, subln_w)
}

// ===================================================================
// Integration tests
// ===================================================================

/// Test case parameters: (seq_len, num_heads, num_kv_heads, head_dim, layer_depth, name)
const TEST_CASES: &[(usize, usize, usize, usize, usize, &str)] = &[
    // MHA: single head, short sequence
    (1, 1, 1, 4, 0, "mha-1h-seq1"),
    (3, 1, 1, 4, 0, "mha-1h-seq3"),
    // MHA: multiple heads
    (5, 2, 2, 4, 1, "mha-2h-seq5"),
    (8, 4, 4, 8, 3, "mha-4h-seq8"),
    // GQA: num_kv_heads < num_heads
    (4, 4, 2, 4, 2, "gqa-4h-2kv-seq4"),
    (6, 4, 1, 4, 5, "gqa-4h-1kv-seq6"),
    // Larger head_dim
    (5, 2, 1, 8, 10, "gqa-2h-1kv-hd8-seq5"),
];

/// Core correctness test: optimized kernel matches naive reference within 1e-4.
#[test]
fn test_diff_attn_matches_reference() {
    const TOLERANCE: f32 = 1e-4;

    for &(seq_len, num_heads, num_kv_heads, head_dim, layer_depth, name) in TEST_CASES {
        let cfg = DiffAttnConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            layer_depth,
        };
        let (q, k, v, params, subln_w) = make_inputs(&cfg, seq_len, (seq_len as u64) * 1009);

        let ref_out =
            ref_differential_attention(&q, &k, &v, &params, &subln_w, 1e-6, seq_len, &cfg);

        let mut opt_out = vec![0.0_f32; seq_len * cfg.out_dim()];
        let mut scratch = DiffAttnScratch::default();
        apply_differential_attention(
            &q,
            &k,
            &v,
            &params,
            &subln_w,
            1e-6,
            &mut opt_out,
            seq_len,
            &cfg,
            &mut scratch,
        );

        let diff = max_abs_diff(&opt_out, &ref_out);
        assert!(
            diff <= TOLERANCE,
            "case '{name}': max_abs_diff={diff} exceeds {TOLERANCE}"
        );
    }
}

/// Verify the output has the correct length for all test cases.
#[test]
fn test_diff_attn_output_length() {
    for &(seq_len, num_heads, num_kv_heads, head_dim, layer_depth, name) in TEST_CASES {
        let cfg = DiffAttnConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            layer_depth,
        };
        let expected_len = seq_len * num_heads * 2 * head_dim;
        let (q, k, v, params, subln_w) = make_inputs(&cfg, seq_len, 42);
        let mut out = vec![0.0_f32; expected_len];
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
            expected_len,
            "case '{name}': output len mismatch"
        );
    }
}

/// Verify causal masking: position 0's output must not depend on future V.
///
/// Differential test: run two inputs that differ ONLY in V at positions 1..seq_len.
/// `out[pos 0]` must be bit-identical between the runs. A magnitude bound would NOT
/// catch a leak — the sub-layer RMSNorm renormalizes a contaminated row back to O(1).
#[test]
fn test_causal_masking_integration() {
    let cfg = DiffAttnConfig {
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        layer_depth: 0,
    };
    let seq_len = 4;
    let (q, k, v_base, params, subln_w) = make_inputs(&cfg, seq_len, 777);

    let v_head_dim = 2 * cfg.head_dim;
    let v_stride = cfg.num_kv_heads * v_head_dim;

    // Variant: identical at position 0, perturbed at all future positions.
    let mut v_perturbed = v_base.clone();
    for pos in 1..seq_len {
        for d in 0..v_stride {
            v_perturbed[pos * v_stride + d] += 9_876.0;
        }
    }

    let run = |v: &[f32]| {
        let mut out = vec![0.0_f32; seq_len * cfg.out_dim()];
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
    let out_stride = cfg.out_dim();

    // Position 0 must be unchanged — it cannot attend to future V.
    for d in 0..out_stride {
        assert_eq!(
            out_base[d].to_bits(),
            out_perturbed[d].to_bits(),
            "position 0 changed when only future V changed — causal mask leak at dim {d}"
        );
    }
    // Sanity: the last position SHOULD change, or the perturbation was a no-op.
    let last_changed = (0..out_stride).any(|d| {
        let off = (seq_len - 1) * out_stride + d;
        out_base[off] != out_perturbed[off]
    });
    assert!(
        last_changed,
        "last position should be affected by the future-V perturbation"
    );
}

/// Verify that scratch buffer reuse across calls does not corrupt results.
///
/// Call `apply_differential_attention` twice with different inputs on the same scratch.
/// Both outputs must match their respective references.
#[test]
fn test_scratch_reuse() {
    const TOL: f32 = 1e-4;

    let cfg = DiffAttnConfig {
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 4,
        layer_depth: 2,
    };
    let mut scratch = DiffAttnScratch::default();

    for (seed, seq_len) in [(100_u64, 3_usize), (200, 5)] {
        let (q, k, v, params, subln_w) = make_inputs(&cfg, seq_len, seed);
        let ref_out =
            ref_differential_attention(&q, &k, &v, &params, &subln_w, 1e-6, seq_len, &cfg);
        let mut opt_out = vec![0.0_f32; seq_len * cfg.out_dim()];
        apply_differential_attention(
            &q,
            &k,
            &v,
            &params,
            &subln_w,
            1e-6,
            &mut opt_out,
            seq_len,
            &cfg,
            &mut scratch,
        );
        let diff = max_abs_diff(&opt_out, &ref_out);
        assert!(
            diff <= TOL,
            "scratch reuse seed={seed} seq_len={seq_len}: max_abs_diff={diff} exceeds {TOL}"
        );
    }
}

/// Independent check of the GQA head-mapping convention.
///
/// The naive reference (`ref_differential_attention`) derives the KV head for
/// query-pair `h` as `h / n_rep` — the *same* integer-division mapping the kernel
/// uses. A shared misunderstanding of the grouping (contiguous vs. interleaved)
/// would pass both, so that oracle alone cannot validate it.
///
/// This test removes the `/ n_rep` derivation from the oracle path entirely. GQA
/// is *defined* as MHA with each KV head physically replicated `n_rep` times in
/// contiguous order (`repeat_kv`; in Microsoft's reference this is what
/// `flash_attn_func` does internally). So:
///
///   1. Run a GQA config (`num_kv_heads < num_heads`) on K/V data with distinct
///      KV heads.
///   2. Build an MHA config (`num_kv_heads == num_heads`, `n_rep == 1` — no
///      grouping logic in the compute path) whose K/V buffers are the GQA buffers
///      with each KV head materialized `n_rep` times, using a *hardcoded* pair→KV
///      table rather than `/ n_rep`.
///   3. The two outputs must be bit-identical: query-pair `h` sees the same Q/K/V
///      in both runs, so every matmul/softmax/norm is the same arithmetic on the
///      same bytes.
///
/// If the kernel's grouping convention disagreed with contiguous `repeat_kv`, the
/// GQA run would diverge from the materialized-MHA run.
///
/// This does not replace a capture from the actual flash-attn reference (which
/// needs CUDA hardware), but it closes the "shared derivation" gap: the oracle
/// here decides the grouping by explicit data placement, not by the kernel's own
/// indexing formula.
#[test]
fn test_gqa_equals_materialized_mha() {
    // GQA config: 4 query-pairs, 2 KV heads → n_rep = 2.
    let gqa_cfg = DiffAttnConfig {
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        layer_depth: 3,
    };
    // Contiguous repeat_kv for num_heads=4, num_kv_heads=2: query-pair h is served
    // by KV head h/2 → [0, 0, 1, 1]. Written as a literal, NOT computed via
    // `/ n_rep`, so this oracle shares no grouping derivation with the kernel.
    const PAIR_TO_KV: [usize; 4] = [0, 0, 1, 1];

    let seq_len = 5;
    let head_dim = gqa_cfg.head_dim;
    let v_head_dim = 2 * head_dim;

    let (q, k_gqa, v_gqa, params, subln_w) = make_inputs(&gqa_cfg, seq_len, 4242);

    // MHA config: one KV head per query-pair (n_rep = 1). head_dim is unchanged,
    // so `params` and `subln_w` from make_inputs are valid for both configs.
    let mha_cfg = DiffAttnConfig {
        num_heads: gqa_cfg.num_heads,
        num_kv_heads: gqa_cfg.num_heads,
        head_dim,
        layer_depth: gqa_cfg.layer_depth,
    };

    // Materialize K: MHA KV head `m` holds GQA KV head `PAIR_TO_KV[m]`'s two
    // packed K heads.
    let gqa_k_stride = gqa_cfg.k_dim();
    let mha_k_stride = mha_cfg.k_dim();
    let mut k_mha = vec![0.0_f32; seq_len * mha_k_stride];
    for pos in 0..seq_len {
        for (mha_kv, &gqa_kv) in PAIR_TO_KV.iter().enumerate() {
            for half in 0..2 {
                let src = pos * gqa_k_stride + (2 * gqa_kv + half) * head_dim;
                let dst = pos * mha_k_stride + (2 * mha_kv + half) * head_dim;
                k_mha[dst..dst + head_dim].copy_from_slice(&k_gqa[src..src + head_dim]);
            }
        }
    }

    // Materialize V: MHA KV head `m` holds GQA KV head `PAIR_TO_KV[m]`'s V head.
    let gqa_v_stride = gqa_cfg.v_dim();
    let mha_v_stride = mha_cfg.v_dim();
    let mut v_mha = vec![0.0_f32; seq_len * mha_v_stride];
    for pos in 0..seq_len {
        for (mha_kv, &gqa_kv) in PAIR_TO_KV.iter().enumerate() {
            let src = pos * gqa_v_stride + gqa_kv * v_head_dim;
            let dst = pos * mha_v_stride + mha_kv * v_head_dim;
            v_mha[dst..dst + v_head_dim].copy_from_slice(&v_gqa[src..src + v_head_dim]);
        }
    }

    // Q is never grouped, and q_dim/out_dim depend only on num_heads (identical
    // between the two configs) — so the Q buffer and output shape are shared.
    assert_eq!(gqa_cfg.q_dim(), mha_cfg.q_dim());
    assert_eq!(gqa_cfg.out_dim(), mha_cfg.out_dim());

    let run = |cfg: &DiffAttnConfig, k: &[f32], v: &[f32]| {
        let mut out = vec![0.0_f32; seq_len * cfg.out_dim()];
        let mut scratch = DiffAttnScratch::default();
        apply_differential_attention(
            &q,
            k,
            v,
            &params,
            &subln_w,
            1e-6,
            &mut out,
            seq_len,
            cfg,
            &mut scratch,
        );
        out
    };

    let out_gqa = run(&gqa_cfg, &k_gqa, &v_gqa);
    let out_mha = run(&mha_cfg, &k_mha, &v_mha);

    for (i, (g, m)) in out_gqa.iter().zip(out_mha.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            m.to_bits(),
            "GQA output diverged from materialized-MHA at index {i}: {g} vs {m} \
             — kernel head-grouping disagrees with contiguous repeat_kv"
        );
    }
}
