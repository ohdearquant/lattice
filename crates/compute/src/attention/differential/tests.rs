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

// NaN-honest max|a-b|: a `.fold(0.0, f32::max)` silently drops a NaN/Inf
// operand (IEEE maxNum keeps the non-NaN side), letting a catastrophically
// wrong output read as 0.0 and slip past a `<= tol` gate. Surface it instead.
#[allow(dead_code)]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if !d.is_finite() {
            return d;
        }
        if d > max {
            max = d;
        }
    }
    max
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
// ADR-080 C1: NaN score must fail closed (zero the poisoned row), not
// propagate. RED before the fix: the softmax loop's `if sum > 0.0` had no
// `else` branch, so a NaN-poisoned row was left holding NaN/Inf exponentials
// instead of being zeroed (#780).
// ---------------------------------------------------------------

#[test]
fn test_diff_attn_nan_score_fails_closed() {
    let head_dim = 2_usize;
    let cfg = make_cfg(1, 1, head_dim, 0);
    let seq_len = 3_usize;
    let q_dim = cfg.q_dim();
    let out_dim = cfg.out_dim();

    // Differential attention packs TWO Q sub-vectors per head (Q1 -> scores1,
    // Q2 -> scores2; see the `(q_h0, k_h0, scores1), (q_h1, k_h1, scores2)`
    // pair loop above). Poisoning only ONE sub-vector's dim (as the original
    // regression test did) only fails ONE of the two differential softmaxes
    // closed, so `context = attn1 - lambda*attn2` is `0 - lambda*attn2`, not
    // exactly zero -- `is_finite()` passed on that output for the wrong
    // reason (a still-finite nonzero value), not because the row was zeroed.
    // Poison the WHOLE token-1 Q row (both sub-vectors) so BOTH softmaxes
    // fail closed and the differential context is `0 - lambda*0 == 0`
    // exactly, letting this test pin "zero in full" instead of "finite".
    let mut q = det_data(seq_len * q_dim, 11);
    for d in 0..q_dim {
        q[q_dim + d] = f32::NAN;
    }
    let k = det_data(seq_len * cfg.k_dim(), 12);
    let v = det_data(seq_len * cfg.v_dim(), 13);

    let params = zero_lambda_params(head_dim);
    let subln_w = vec![1.0f32; 2 * head_dim];
    let mut out = vec![0.0f32; seq_len * out_dim];
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

    let poisoned_row = &out[out_dim..2 * out_dim];
    assert!(
        poisoned_row.iter().all(|&x| x == 0.0),
        "differential attention's poisoned query row must be exactly \
             all-zero (fail-closed), not merely finite; got {poisoned_row:?}"
    );

    // Sibling rows (query 0 and query 2) share no poisoned Q/K/V and must
    // stay finite and non-degenerate (a real normalized result, not
    // incidentally zero).
    for t in [0usize, 2usize] {
        let row = &out[t * out_dim..(t + 1) * out_dim];
        assert!(
            row.iter().all(|x| x.is_finite()),
            "query {t}'s row must stay finite, got {row:?}"
        );
        assert!(
            row.iter().any(|&x| x != 0.0),
            "query {t}'s row must be a real (non-degenerate) result, got {row:?}"
        );
    }
}

// ---------------------------------------------------------------
// test_diff_attn_partial_nan_row_fails_closed_in_full
// ---------------------------------------------------------------

/// #785: poisoning a WHOLE Q row (as
/// `test_diff_attn_nan_score_fails_closed` does) makes every lane of that
/// row NaN together, so it can't distinguish "zero the whole row" from a
/// buggy "zero only the NaN lane(s), normalize the rest" mutation -- both
/// produce the same all-zero result there. This test poisons a single K
/// TIME STEP instead (both packed K halves at position 0), which causal
/// masking puts in every later query's row but leaves the *other* lanes of
/// that row (k1, k2, ...) untouched: query 1's row is `[NaN, finite]`, a
/// genuinely mixed row. A one-lane-only-zero mutation would zero just lane
/// 0 and renormalize lane 1 to a nonzero weight, producing a nonzero
/// combined context; the real fail-closed contract must zero the WHOLE
/// row so the combined context is exactly zero.
#[test]
fn test_diff_attn_partial_nan_row_fails_closed_in_full() {
    let head_dim = 2_usize;
    let cfg = make_cfg(1, 1, head_dim, 0);
    let seq_len = 2_usize;
    let out_dim = cfg.out_dim();

    let q = det_data(seq_len * cfg.q_dim(), 21);
    // Poison K time-step 0 across BOTH packed K halves (k_h0 -> scores1,
    // k_h1 -> scores2) so both differential softmax paths see the same
    // partial-NaN pattern for every query that includes position 0.
    let mut k = det_data(seq_len * cfg.k_dim(), 22);
    for d in 0..cfg.k_dim() {
        k[d] = f32::NAN;
    }
    let v = det_data(seq_len * cfg.v_dim(), 23);

    let params = zero_lambda_params(head_dim);
    let subln_w = vec![1.0f32; 2 * head_dim];
    let mut out = vec![0.0f32; seq_len * out_dim];
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

    // Query 1's row is `[k0=NaN, k1=finite]` in both scores1 and scores2 --
    // exactly the mixed-lane case a per-lane-only fix would mishandle.
    let mixed_row = &out[out_dim..2 * out_dim];
    assert!(
        mixed_row.iter().all(|&x| x == 0.0),
        "a row with any NaN lane must fail closed in FULL (all lanes zero), \
             not just the poisoned lane(s); got {mixed_row:?}"
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

// ---------------------------------------------------------------
// Config validation guards
//
// Each guard fires before any buffer is inspected, so empty buffers
// are sufficient to reach (and only reach) the assertion under test.
// ---------------------------------------------------------------

#[test]
#[should_panic(expected = "num_heads must be > 0")]
fn test_zero_num_heads_panics() {
    let cfg = make_cfg(0, 1, 4, 0);
    let params = zero_lambda_params(4);
    let mut out: Vec<f32> = vec![];
    let mut scratch = DiffAttnScratch::default();
    apply_differential_attention(
        &[],
        &[],
        &[],
        &params,
        &[],
        1e-6,
        &mut out,
        1,
        &cfg,
        &mut scratch,
    );
}

#[test]
#[should_panic(expected = "num_kv_heads must be > 0")]
fn test_zero_num_kv_heads_panics() {
    let cfg = make_cfg(2, 0, 4, 0);
    let params = zero_lambda_params(4);
    let mut out: Vec<f32> = vec![];
    let mut scratch = DiffAttnScratch::default();
    apply_differential_attention(
        &[],
        &[],
        &[],
        &params,
        &[],
        1e-6,
        &mut out,
        1,
        &cfg,
        &mut scratch,
    );
}

#[test]
#[should_panic(expected = "head_dim must be > 0")]
fn test_zero_head_dim_panics() {
    let cfg = make_cfg(2, 2, 0, 0);
    let params = zero_lambda_params(0);
    let mut out: Vec<f32> = vec![];
    let mut scratch = DiffAttnScratch::default();
    apply_differential_attention(
        &[],
        &[],
        &[],
        &params,
        &[],
        1e-6,
        &mut out,
        1,
        &cfg,
        &mut scratch,
    );
}

#[test]
#[should_panic(expected = "num_kv_heads must be > 0")]
fn test_n_rep_zero_num_kv_heads_panics() {
    // `n_rep()` is public — a direct caller in release must fail loudly,
    // not hit a raw divide-by-zero.
    let _ = make_cfg(4, 0, 8, 0).n_rep();
}

#[test]
#[should_panic(expected = "must be divisible")]
fn test_n_rep_non_divisible_panics() {
    // Non-divisible config must panic, not silently floor 5/2 → 2.
    let _ = make_cfg(5, 2, 8, 0).n_rep();
}
