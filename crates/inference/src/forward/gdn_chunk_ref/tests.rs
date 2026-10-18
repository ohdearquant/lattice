use super::*;
use serde::Deserialize;
use std::path::PathBuf;

/// Fixture format written by `tests/fixtures/gdn_chunk/generate.py`.
#[derive(Deserialize)]
struct Fixture {
    length: usize,
    dk: usize,
    dv: usize,
    chunk_size: usize,
    q: Vec<Vec<f32>>,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
    h0: Vec<Vec<f32>>,
    out_seq: Vec<Vec<f32>>,
    h_seq: Vec<Vec<f32>>,
    out_chk: Vec<Vec<f32>>,
    h_chk: Vec<Vec<f32>>,
}

fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flat_map(|r| r.iter().copied()).collect()
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gdn_chunk")
}

fn load_fixture(name: &str) -> Fixture {
    let path = fixtures_dir().join(name);
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    serde_json::from_str(&data).unwrap_or_else(|e| panic!("bad JSON in {}: {e}", path.display()))
}

/// NaN/Inf-honest `max|a-b|`. A plain `.fold(0.0, f32::max)` silently drops
/// a NaN operand (IEEE `maxNum` returns the non-NaN side), so a
/// catastrophically non-finite output can read as a clean `0.0` and slip
/// through a `<= TOL` tolerance gate. This surfaces any non-finite
/// difference instead, so the gate fails closed on it. (This is exactly the
/// failure the strong-decay fixture exists to catch: the pre-fix linear
/// oracle produced NaN, which a max-fold gate reported as `0.0`.)
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch in max_abs_diff");
    let mut max = 0.0f32;
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

fn count_nonfinite(v: &[f32]) -> usize {
    v.iter().filter(|x| !x.is_finite()).count()
}

const TOL: f32 = 1.0e-5;

fn run_case(fixture_name: &str) {
    let fx = load_fixture(fixture_name);
    let q = flatten(&fx.q);
    let k = flatten(&fx.k);
    let v = flatten(&fx.v);
    let h0 = flatten(&fx.h0);
    let expected_out_seq = flatten(&fx.out_seq);
    let expected_h_seq = flatten(&fx.h_seq);
    let expected_out_chk = flatten(&fx.out_chk);
    let expected_h_chk = flatten(&fx.h_chk);

    let (rust_out_seq, rust_h_seq) = sequential_gdn(
        &q, &k, &v, &fx.beta, &fx.alpha, &h0, fx.length, fx.dk, fx.dv, true,
    );
    let (rust_out_chk, rust_h_chk) = chunkwise_gdn(
        &q,
        &k,
        &v,
        &fx.beta,
        &fx.alpha,
        &h0,
        fx.length,
        fx.dk,
        fx.dv,
        fx.chunk_size,
        true,
    );

    for (label, buf) in [
        ("rust_out_seq", &rust_out_seq),
        ("rust_h_seq", &rust_h_seq),
        ("rust_out_chk", &rust_out_chk),
        ("rust_h_chk", &rust_h_chk),
    ] {
        let n = count_nonfinite(buf);
        assert_eq!(
            n, 0,
            "{fixture_name}: {label} contains {n} non-finite (NaN/Inf) values — \
                 a decay-underflow regression the max-fold gate would silently pass"
        );
    }

    let d_seq_out = max_abs_diff(&rust_out_seq, &expected_out_seq);
    let d_seq_h = max_abs_diff(&rust_h_seq, &expected_h_seq);
    let d_chk_out = max_abs_diff(&rust_out_chk, &expected_out_chk);
    let d_chk_h = max_abs_diff(&rust_h_chk, &expected_h_chk);
    let d_internal_out = max_abs_diff(&rust_out_chk, &rust_out_seq);
    let d_internal_h = max_abs_diff(&rust_h_chk, &rust_h_seq);

    println!(
        "gdn_chunk_ref fixture={fixture_name} \
             rust_seq_vs_numpy_seq(out={d_seq_out:.3e}, h={d_seq_h:.3e}) \
             rust_chk_vs_numpy_chk(out={d_chk_out:.3e}, h={d_chk_h:.3e}) \
             rust_chk_vs_rust_seq(out={d_internal_out:.3e}, h={d_internal_h:.3e})"
    );

    assert!(
        d_seq_out <= TOL,
        "{fixture_name}: rust sequential_gdn out diverged from NumPy: max_abs={d_seq_out}"
    );
    assert!(
        d_seq_h <= TOL,
        "{fixture_name}: rust sequential_gdn h diverged from NumPy: max_abs={d_seq_h}"
    );
    assert!(
        d_chk_out <= TOL,
        "{fixture_name}: rust chunkwise_gdn out diverged from NumPy: max_abs={d_chk_out}"
    );
    assert!(
        d_chk_h <= TOL,
        "{fixture_name}: rust chunkwise_gdn h diverged from NumPy: max_abs={d_chk_h}"
    );
    assert!(
        d_internal_out <= TOL,
        "{fixture_name}: rust chunkwise_gdn out diverged from rust sequential_gdn (the \
             equivalence gate the Metal kernels rely on): max_abs={d_internal_out}"
    );
    assert!(
        d_internal_h <= TOL,
        "{fixture_name}: rust chunkwise_gdn h diverged from rust sequential_gdn (the \
             equivalence gate the Metal kernels rely on): max_abs={d_internal_h}"
    );
}

#[test]
fn gdn_chunk_ref_parity_seed7_chunk64() {
    run_case("case_seed7_len191_chunk64.json");
}

#[test]
fn gdn_chunk_ref_parity_seed11_chunk128() {
    run_case("case_seed11_len384_chunk128.json");
}

#[test]
fn gdn_chunk_ref_parity_seed23_uneven_tail() {
    run_case("case_seed23_len130_chunk64.json");
}

/// gdn175 S1 fix regression: a strongly-decaying head (alpha as low as
/// 1e-4) whose linear-space `cumprod(alpha)` underflows to exactly
/// `0.0f32` within one C=64 chunk on the pre-fix oracle (`gamma_inv =
/// 1/0.0 = inf`, `gamma[i]*gamma_inv[j] = 0*inf = NaN`). The log-space
/// reformulation must still reproduce `sequential_gdn` to <= 1e-5 here.
#[test]
fn gdn_chunk_ref_parity_seed31_strong_decay() {
    run_case("case_seed31_len160_chunk64_strongdecay.json");
}

/// Gate-integrity proof: `max_abs_diff` must never let a non-finite
/// difference read as a small in-tolerance value. A `.fold(0.0, f32::max)`
/// silently drops the NaN/Inf operand and returns `0.0` here, which would
/// pass a `<= TOL` assert on a catastrophically wrong output.
#[test]
fn max_abs_diff_is_nan_and_inf_honest() {
    let clean = [1.0f32, 2.0, 3.0];

    let nan_side = [1.0f32, f32::NAN, 3.0];
    let d = max_abs_diff(&clean, &nan_side);
    assert!(
        !d.is_finite(),
        "max_abs_diff silently dropped a NaN operand (got {d}); the gate is blind"
    );

    let inf_side = [1.0f32, f32::INFINITY, 3.0];
    let di = max_abs_diff(&clean, &inf_side);
    assert!(
        !di.is_finite(),
        "max_abs_diff silently dropped an Inf operand (got {di}); the gate is blind"
    );
}

/// Mutation-sensitivity proof for the strong-decay fixture: on the pre-fix
/// linear-space oracle, `cumprod(alpha)` underflows to exactly `0.0f32`
/// within the first C=64 chunk, so `1.0/gamma` is `+inf` and the decay
/// factor `gamma[i] * gamma_inv[j]` becomes `0*inf = NaN`. The log-space
/// reformulation shipped in this PR never forms `1.0/gamma`. If this assert
/// ever fails, the fixture no longer exercises the regression and the
/// `gdn_chunk_ref_parity_seed31_strong_decay` gate is decoration.
#[test]
fn seed31_fixture_triggers_linear_gamma_underflow() {
    let fx = load_fixture("case_seed31_len160_chunk64_strongdecay.json");
    let c = fx.chunk_size;
    let mut gamma = 1.0f32;
    let mut underflowed = false;
    for i in 0..c {
        gamma *= fx.alpha[i];
        if gamma == 0.0 {
            underflowed = true;
            break;
        }
    }
    assert!(
        underflowed,
        "seed31 strong-decay fixture no longer underflows the linear cumprod within a \
             chunk; it no longer guards the gamma-underflow regression this PR fixes"
    );
    let gamma_inv = 1.0f32 / gamma;
    assert!(
        !gamma_inv.is_finite(),
        "expected 1/0 = inf, got {gamma_inv}"
    );
    assert!(
        (0.0f32 * gamma_inv).is_nan(),
        "expected the pre-fix decay factor 0*inf to be NaN"
    );
}

/// ADR-080 C1 fail-closed (#850) table test for the chunkwise oracle's row
/// normalizer: a row with a non-finite lane must be left at its zero-
/// initialized default (whole-row zero by non-assignment, equivalent to
/// direct zero-assignment), never divided through a non-finite `denom`
/// (`NaN / x == NaN`, `x / inf == 0` only masks the corrupted row's OTHER
/// lanes while the corrupted lane itself stays non-finite). Finite rows
/// (all-zero, ordinary) must be numerically unchanged.
///
/// Mutation-sensitive: reverting the `if !sum_sq.is_finite() { continue; }`
/// guard back to unconditional division makes the `nan_row`/`inf_row` cases
/// below fail (the output row keeps a non-finite lane).
#[test]
fn l2_normalize_rows_fail_closed_table() {
    let cols = 3usize;
    let cases: &[(&str, &[f32], bool)] = &[
        ("nan_lane", &[f32::NAN, 1.0, 2.0], true),
        ("pos_inf_lane", &[f32::INFINITY, 1.0, 2.0], true),
        ("neg_inf_lane", &[f32::NEG_INFINITY, 1.0, 2.0], true),
        ("all_zero", &[0.0, 0.0, 0.0], true),
    ];
    for (label, row, expect_all_zero) in cases {
        let out = l2_normalize_rows(row, 1, cols, 1e-6);
        assert!(
            out.iter().all(|x| x.is_finite()),
            "case {label}: l2_normalize_rows output must be fully finite, got {out:?}"
        );
        if *expect_all_zero {
            assert!(
                out.iter().all(|x| *x == 0.0),
                "case {label}: expected the whole row zeroed, got {out:?}"
            );
        }
    }

    // ordinary finite row: numerically unchanged from the pre-#850 formula.
    let ordinary = l2_normalize_rows(&[3.0f32, 4.0], 1, 2, 0.0);
    assert!((ordinary[0] - 0.6).abs() < 1e-6);
    assert!((ordinary[1] - 0.8).abs() < 1e-6);

    // multi-row: a non-finite row must not contaminate an adjacent finite row.
    let multi = l2_normalize_rows(&[f32::NAN, 0.0, 0.0, 3.0, 4.0, 0.0], 2, 3, 0.0);
    assert!(
        multi[0..3].iter().all(|x| *x == 0.0),
        "row 0 (poisoned) must be all-zero, got {:?}",
        &multi[0..3]
    );
    assert!(
        (multi[3] - 0.6).abs() < 1e-6 && (multi[4] - 0.8).abs() < 1e-6,
        "row 1 (clean) must be unaffected by row 0's poisoning, got {:?}",
        &multi[3..6]
    );
}
