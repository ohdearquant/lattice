use super::*;

/// Build a `[seq_len, seq_len]` single-head score matrix where query row 0
/// holds `row` (padded/truncated to `seq_len`) and every other query row
/// holds benign, well-formed finite scores. Returns `(matrix, seq_len)`.
fn single_row_matrix(row: &[f32]) -> (Vec<f32>, usize) {
    let seq_len = row.len();
    let mut x = vec![0.0f32; seq_len * seq_len];
    x[..seq_len].copy_from_slice(row);
    for qi in 1..seq_len {
        for ki in 0..seq_len {
            x[qi * seq_len + ki] = 0.1 * (qi as f32 + 1.0) + 0.01 * (ki as f32);
        }
    }
    (x, seq_len)
}

/// A single NaN score anywhere in the row must zero the WHOLE row, not just
/// that lane (#740). RED before the fix: the row max fold silently dropped
/// the NaN operand, so only the NaN's own output lane stayed corrupted while
/// the rest of the row normalized around the (correct) finite max.
#[test]
fn scalar_nan_lane_zeros_whole_row() {
    let (mut x, seq_len) = single_row_matrix(&[1.0, f32::NAN, 2.0, 0.5]);
    softmax_attention_scalar(&mut x, seq_len, 1);
    assert!(
        x[..seq_len].iter().all(|&v| v == 0.0),
        "expected all-zero row: {:?}",
        &x[..seq_len]
    );
}

/// A `+inf` score must zero the whole row (fail-closed on non-finite max).
#[test]
fn scalar_pos_inf_zeros_whole_row() {
    let (mut x, seq_len) = single_row_matrix(&[1.0, f32::INFINITY, 2.0]);
    softmax_attention_scalar(&mut x, seq_len, 1);
    assert!(
        x[..seq_len].iter().all(|&v| v == 0.0),
        "expected all-zero row: {:?}",
        &x[..seq_len]
    );
}

/// An all-masked (`-inf`) row must zero out rather than compute `fast_exp(NaN)`.
#[test]
fn scalar_all_neg_inf_zeros_row() {
    let (mut x, seq_len) = single_row_matrix(&[f32::NEG_INFINITY; 4]);
    softmax_attention_scalar(&mut x, seq_len, 1);
    assert!(
        x[..seq_len].iter().all(|&v| v == 0.0),
        "expected all-zero row: {:?}",
        &x[..seq_len]
    );
}

/// A masked (`-inf`) lane mixed with valid scores must produce EXACT `0.0`,
/// not the small nonzero value `fast_exp`'s finite-floor clamp would
/// otherwise leak (#740). RED before the fix: `fast_exp(-inf - max)` clamped
/// to `fast_exp(-87.0)`, a tiny but nonzero weight.
#[test]
fn scalar_masked_lane_is_exact_zero() {
    let (mut x, seq_len) = single_row_matrix(&[1.0, f32::NEG_INFINITY, 0.5]);
    softmax_attention_scalar(&mut x, seq_len, 1);
    assert_eq!(x[1], 0.0, "masked lane must be exact zero, got {}", x[1]);
    assert!(x[..seq_len].iter().all(|v| v.is_finite()));
    let sum: f32 = x[..seq_len].iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "row must still sum to ~1: {:?}",
        &x[..seq_len]
    );
}

/// Multiple heads/rows in one call: only the poisoned row zeros, sibling
/// rows in the same batched buffer stay well-formed.
#[test]
fn scalar_only_poisoned_row_zeros() {
    let seq_len = 2;
    let num_heads = 1;
    // head 0, row 0 (query 0): NaN present. head 0, row 1 (query 1): clean.
    let mut x = vec![f32::NAN, 1.0, 0.3, 0.7];
    softmax_attention_scalar(&mut x, seq_len, num_heads);
    assert!(x[0..2].iter().all(|&v| v == 0.0), "row 0 must zero: {x:?}");
    assert!(x[2..4].iter().all(|v| v.is_finite() && *v > 0.0));
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;

    fn run_neon(x: &mut [f32], seq_len: usize, num_heads: usize) {
        // SAFETY: NEON is always available on aarch64.
        unsafe {
            softmax_attention_neon(x, seq_len, num_heads);
        }
    }

    /// NEON must match the scalar reference: a single NaN lane zeros the
    /// whole row. Uses a 17-wide row (one 16-wide vectorized chunk + a
    /// scalar tail) so both the chunked path and the tail path run.
    #[test]
    fn neon_nan_in_chunk_zeros_whole_row() {
        let mut row = vec![0.5f32; 17];
        row[5] = f32::NAN; // inside the 16-wide vectorized chunk
        let (mut x, seq_len) = single_row_matrix(&row);
        run_neon(&mut x, seq_len, 1);
        assert!(
            x[..seq_len].iter().all(|&v| v == 0.0),
            "expected all-zero row: {:?}",
            &x[..seq_len]
        );
    }

    /// NaN in the scalar tail (not caught by the vectorized chunk NaN scan)
    /// must also zero the whole row.
    #[test]
    fn neon_nan_in_tail_zeros_whole_row() {
        let mut row = vec![0.5f32; 17];
        row[16] = f32::NAN; // scalar tail lane
        let (mut x, seq_len) = single_row_matrix(&row);
        run_neon(&mut x, seq_len, 1);
        assert!(
            x[..seq_len].iter().all(|&v| v == 0.0),
            "expected all-zero row: {:?}",
            &x[..seq_len]
        );
    }

    /// A `+inf` score zeros the whole row.
    #[test]
    fn neon_pos_inf_zeros_whole_row() {
        let mut row = vec![0.5f32; 17];
        row[3] = f32::INFINITY;
        let (mut x, seq_len) = single_row_matrix(&row);
        run_neon(&mut x, seq_len, 1);
        assert!(
            x[..seq_len].iter().all(|&v| v == 0.0),
            "expected all-zero row: {:?}",
            &x[..seq_len]
        );
    }

    /// An all-masked (`-inf`) row zeros out.
    #[test]
    fn neon_all_neg_inf_zeros_row() {
        let row = vec![f32::NEG_INFINITY; 17];
        let (mut x, seq_len) = single_row_matrix(&row);
        run_neon(&mut x, seq_len, 1);
        assert!(
            x[..seq_len].iter().all(|&v| v == 0.0),
            "expected all-zero row: {:?}",
            &x[..seq_len]
        );
    }

    /// A masked (`-inf`) lane inside the vectorized chunk must be exact
    /// zero, not `neon_exp`'s finite-floor-clamp leakage (#740).
    #[test]
    fn neon_masked_lane_in_chunk_is_exact_zero() {
        let mut row = vec![0.5f32; 17];
        row[2] = f32::NEG_INFINITY;
        let (mut x, seq_len) = single_row_matrix(&row);
        run_neon(&mut x, seq_len, 1);
        assert_eq!(x[2], 0.0, "masked lane must be exact zero, got {}", x[2]);
        assert!(x[..seq_len].iter().all(|v| v.is_finite()));
        let sum: f32 = x[..seq_len].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "row must still sum to ~1: {:?}",
            &x[..seq_len]
        );
    }

    /// NEON must agree with the scalar reference on well-formed (all
    /// finite) rows, matching the scalar/NEON parity the file's own
    /// comments already document.
    #[test]
    fn neon_matches_scalar_on_finite_row() {
        let row: Vec<f32> = (0..20).map(|i| (i as f32 * 0.37).sin()).collect();
        let (mut x_scalar, seq_len) = single_row_matrix(&row);
        let mut x_neon = x_scalar.clone();
        softmax_attention_scalar(&mut x_scalar, seq_len, 1);
        run_neon(&mut x_neon, seq_len, 1);
        for (a, b) in x_scalar[..seq_len].iter().zip(x_neon[..seq_len].iter()) {
            assert!((a - b).abs() < 1e-4, "scalar={a} neon={b}");
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;

    fn run_avx2(x: &mut [f32], seq_len: usize, num_heads: usize) -> bool {
        if !simd_config().avx2_enabled {
            return false;
        }
        // SAFETY: gated on the runtime AVX2 feature check above.
        unsafe {
            softmax_attention_avx2(x, seq_len, num_heads);
        }
        true
    }

    #[test]
    fn avx2_nan_in_chunk_zeros_whole_row() {
        let mut row = vec![0.5f32; 9]; // one 8-wide chunk + a scalar tail
        row[3] = f32::NAN;
        let (mut x, seq_len) = single_row_matrix(&row);
        if !run_avx2(&mut x, seq_len, 1) {
            return;
        }
        assert!(
            x[..seq_len].iter().all(|&v| v == 0.0),
            "expected all-zero row: {:?}",
            &x[..seq_len]
        );
    }

    #[test]
    fn avx2_nan_in_tail_zeros_whole_row() {
        let mut row = vec![0.5f32; 9];
        row[8] = f32::NAN;
        let (mut x, seq_len) = single_row_matrix(&row);
        if !run_avx2(&mut x, seq_len, 1) {
            return;
        }
        assert!(
            x[..seq_len].iter().all(|&v| v == 0.0),
            "expected all-zero row: {:?}",
            &x[..seq_len]
        );
    }

    #[test]
    fn avx2_pos_inf_zeros_whole_row() {
        let mut row = vec![0.5f32; 9];
        row[0] = f32::INFINITY;
        let (mut x, seq_len) = single_row_matrix(&row);
        if !run_avx2(&mut x, seq_len, 1) {
            return;
        }
        assert!(
            x[..seq_len].iter().all(|&v| v == 0.0),
            "expected all-zero row: {:?}",
            &x[..seq_len]
        );
    }

    #[test]
    fn avx2_masked_lane_is_exact_zero() {
        let mut row = vec![0.5f32; 9];
        row[4] = f32::NEG_INFINITY;
        let (mut x, seq_len) = single_row_matrix(&row);
        if !run_avx2(&mut x, seq_len, 1) {
            return;
        }
        assert_eq!(x[4], 0.0, "masked lane must be exact zero, got {}", x[4]);
    }

    #[test]
    fn avx2_matches_scalar_on_finite_row() {
        let row: Vec<f32> = (0..20).map(|i| (i as f32 * 0.37).sin()).collect();
        let (mut x_scalar, seq_len) = single_row_matrix(&row);
        let mut x_avx2 = x_scalar.clone();
        softmax_attention_scalar(&mut x_scalar, seq_len, 1);
        if !run_avx2(&mut x_avx2, seq_len, 1) {
            return;
        }
        for (a, b) in x_scalar[..seq_len].iter().zip(x_avx2[..seq_len].iter()) {
            assert!((a - b).abs() < 1e-4, "scalar={a} avx2={b}");
        }
    }
}
