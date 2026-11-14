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

/// Row lengths covering an empty SIMD body, exact multiples of the 8- and
/// 16-wide chunks, and chunk-plus-tail shapes.
const ACCURACY_LENS: [usize; 9] = [1, 3, 8, 9, 15, 16, 17, 33, 128];

/// Relative tolerance against the f64 reference. The polynomial exp is within
/// ~15 ULP and f32 summation over 128 terms adds at most ~8e-6, while the
/// Schraudolph bit-trick exp is off by up to ~6%.
const ACCURACY_RTOL: f64 = 2e-5;

/// A `[seq_len, seq_len]` single-head matrix of deterministic scores in
/// roughly [-12, 12], so each row spans many powers of two after the max is
/// subtracted.
fn accuracy_matrix(seq_len: usize) -> Vec<f32> {
    (0..seq_len * seq_len)
        .map(|i| {
            let t = i as f32;
            12.0 * (t * 0.618_034).sin() * (t * 0.137).cos()
        })
        .collect()
}

fn softmax_f64_reference(row: &[f32]) -> Vec<f64> {
    let max = row
        .iter()
        .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
    let exps: Vec<f64> = row.iter().map(|&v| (f64::from(v) - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Every output weight of `run` must match the f64 softmax within
/// `ACCURACY_RTOL` relative error, on every row of every shape.
fn assert_matches_f64_reference(arm: &str, run: impl Fn(&mut [f32], usize, usize)) {
    for &seq_len in &ACCURACY_LENS {
        let input = accuracy_matrix(seq_len);
        let mut out = input.clone();
        run(&mut out, seq_len, 1);
        for r in 0..seq_len {
            let row = &input[r * seq_len..(r + 1) * seq_len];
            let reference = softmax_f64_reference(row);
            for (k, (&got, &want)) in out[r * seq_len..(r + 1) * seq_len]
                .iter()
                .zip(reference.iter())
                .enumerate()
            {
                let rel = (f64::from(got) - want).abs() / want;
                assert!(
                    rel <= ACCURACY_RTOL,
                    "{arm}: seq_len={seq_len} row={r} key={k}: got {got}, f64 reference {want}, relative error {rel:.3e}"
                );
            }
        }
    }
}

#[test]
fn scalar_matches_f64_reference() {
    assert_matches_f64_reference("scalar", softmax_attention_scalar);
}

#[test]
fn dispatched_matches_f64_reference() {
    assert_matches_f64_reference("dispatched", softmax_attention);
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

    // The grid spans [-ln(2)/2, ln(2)/2]. The 5e-6 bound accommodates
    // floating-point error for both the degree-4 and previous degree-5 forms
    // while remaining tighter than the 2e-5 softmax weight gate.
    #[test]
    fn neon_exp_reduced_interval_matches_f64_dense_grid() {
        use std::arch::aarch64::{vld1q_f32, vst1q_f32};
        const STEPS: usize = 100_000;
        const MAX_RELATIVE_ERROR: f64 = 5e-6;
        let half_ln2 = std::f64::consts::LN_2 / 2.0;
        for batch in 0..=STEPS / 4 {
            let mut input = [0.0f32; 4];
            for (lane, x) in input.iter_mut().enumerate() {
                let index = (batch * 4 + lane).min(STEPS);
                *x = (-half_ln2 + 2.0 * half_ln2 * index as f64 / STEPS as f64) as f32;
            }
            let mut output = [0.0f32; 4];
            // SAFETY: both arrays hold four f32 lanes; NEON is available on aarch64.
            unsafe {
                let vector = vld1q_f32(input.as_ptr());
                vst1q_f32(output.as_mut_ptr(), exp_neon_degree4(vector));
            }
            for (&x, &got) in input.iter().zip(&output) {
                let reference = f64::from(x).exp();
                let relative = (f64::from(got) - reference).abs() / reference;
                assert!(
                    relative <= MAX_RELATIVE_ERROR,
                    "x={x}, got={got}, reference={reference}, relative error={relative}"
                );
            }
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

    #[test]
    fn neon_matches_f64_reference() {
        assert_matches_f64_reference("neon", run_neon);
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
    fn avx2_matches_f64_reference() {
        if !simd_config().avx2_enabled {
            return;
        }
        assert_matches_f64_reference("avx2", |x, seq_len, num_heads| {
            run_avx2(x, seq_len, num_heads);
        });
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

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn assert_special_rows(arm: &str, run: impl Fn(&mut [f32], usize, usize)) {
    for &len in &ACCURACY_LENS {
        for lane in [0, len / 2, len - 1] {
            for value in [f32::NAN, f32::INFINITY] {
                let mut row = vec![0.5f32; len];
                row[lane] = value;
                let (mut matrix, _) = single_row_matrix(&row);
                run(&mut matrix, len, 1);
                assert!(
                    matrix[..len].iter().all(|&v| v == 0.0),
                    "{arm}: len={len} lane={lane} value={value}"
                );
            }
        }
        let (mut masked, _) = single_row_matrix(&vec![f32::NEG_INFINITY; len]);
        run(&mut masked, len, 1);
        assert!(masked[..len].iter().all(|&v| v == 0.0), "{arm}: len={len}");
        if len > 1 {
            let mut row = vec![0.5f32; len];
            row[0] = f32::NEG_INFINITY;
            let (mut matrix, _) = single_row_matrix(&row);
            run(&mut matrix, len, 1);
            assert_eq!(matrix[0], 0.0, "{arm}: len={len}");
            assert!(matrix[1..len].iter().all(|v| v.is_finite() && *v > 0.0));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_fma_matches_f64_reference() {
    let config = simd_config();
    if !config.avx2_enabled || !config.fma_enabled {
        return;
    }
    eprintln!("softmax avx2-fma: exercising SIMD body");
    let run = |x: &mut [f32], seq_len, num_heads| {
        // SAFETY: both AVX2 and FMA were detected above.
        unsafe { softmax_attention_avx2_fma(x, seq_len, num_heads) };
    };
    assert_matches_f64_reference("avx2-fma", run);
    assert_special_rows("avx2-fma", run);
}

#[cfg(target_arch = "x86_64")]
mod avx2_fma_accuracy {
    use super::*;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    unsafe fn evaluate_exp(input: &[f32; 8]) -> [f32; 8] {
        let mut output = [0.0; 8];
        // SAFETY: both arrays have eight lanes; the caller supplies AVX2 and FMA.
        unsafe {
            let result = exp_avx2_fma(_mm256_loadu_ps(input.as_ptr()));
            _mm256_storeu_ps(output.as_mut_ptr(), result);
        }
        output
    }

    fn assert_normalized_rows(input: &[f32], output: &[f32], seq_len: usize) {
        for (row_index, (row, got)) in input
            .chunks_exact(seq_len)
            .zip(output.chunks_exact(seq_len))
            .enumerate()
        {
            if row.iter().any(|v| v.is_nan() || *v == f32::INFINITY)
                || row.iter().all(|v| *v == f32::NEG_INFINITY)
            {
                assert!(got.iter().all(|v| v.to_bits() == 0), "row={row_index}");
                continue;
            }
            for (key, (&got, want)) in got.iter().zip(softmax_f64_reference(row)).enumerate() {
                if want == 0.0 {
                    assert_eq!(got.to_bits(), 0, "row={row_index} key={key}");
                    continue;
                }
                // These fixtures deliberately stay outside subnormal/clamped weights.
                assert!(want >= f64::from(f32::MIN_POSITIVE));
                let relative = (f64::from(got) - want).abs() / want;
                assert!(
                    relative <= ACCURACY_RTOL,
                    "seq_len={seq_len} row={row_index} key={key}: got {got}, reference {want}, relative error {relative:.3e}"
                );
            }
        }
    }

    #[test]
    fn helper_matches_f64_over_reduction_and_reconstruction_samples() {
        let config = simd_config();
        if !config.avx2_enabled || !config.fma_enabled {
            return;
        }
        let half_ln2 = std::f64::consts::LN_2 / 2.0;
        let mut samples = Vec::new();
        for (lo, hi) in [
            (-half_ln2, half_ln2),
            (f64::from(EXP_LO), f64::from(EXP_HI)),
        ] {
            for index in 0..=100_000 {
                samples.push((lo + (hi - lo) * f64::from(index) / 100_000.0) as f32);
            }
        }
        samples.extend([0.0, 0.0f32.next_down(), 0.0f32.next_up()]);
        // Nominal boundaries and their neighbors are stress samples, not a certificate.
        for exponent in -127..=127 {
            for offset in [0.0, 0.5] {
                let center = ((f64::from(exponent) + offset) * std::f64::consts::LN_2) as f32;
                let mut below = center;
                let mut above = center;
                samples.push(center);
                for _ in 0..4 {
                    below = below.next_down();
                    above = above.next_up();
                    samples.extend([below, above]);
                }
            }
        }
        samples.extend([EXP_LO.next_up(), EXP_HI.next_down()]);
        samples.retain(|&x| (EXP_LO..=EXP_HI).contains(&x));
        for chunk in samples.chunks(8) {
            let mut input = [0.0; 8];
            input[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: AVX2 and FMA were detected above.
            let output = unsafe { evaluate_exp(&input) };
            for (&x, &got) in chunk.iter().zip(output.iter()) {
                let want = f64::from(x).exp();
                let relative = (f64::from(got) - want).abs() / want;
                assert!(relative <= 5e-6, "x={x:?}: relative error {relative:.3e}");
            }
        }

        // Clamp compatibility is separate from the unclamped f64 accuracy oracle.
        let input = [
            EXP_LO - 1.0,
            EXP_LO,
            EXP_LO.next_down(),
            EXP_LO.next_up(),
            EXP_HI.next_down(),
            EXP_HI,
            EXP_HI.next_up(),
            EXP_HI + 1.0,
        ];
        // SAFETY: AVX2 and FMA were detected above.
        let output = unsafe { evaluate_exp(&input) };
        for lane in [0, 2] {
            assert_eq!(output[lane].to_bits(), output[1].to_bits());
        }
        for lane in [6, 7] {
            assert_eq!(output[lane].to_bits(), output[5].to_bits());
        }
        eprintln!("softmax avx2-fma: completed helper accuracy and clamp controls");
    }

    #[test]
    fn normalized_long_and_tail_rows_match_f64_and_dispatch() {
        let config = simd_config();
        if !config.avx2_enabled || !config.fma_enabled {
            return;
        }
        for seq_len in ACCURACY_LENS.into_iter().chain([256, 512]) {
            let mut input = vec![0.0; 2 * seq_len * seq_len];
            for (row_index, row) in input.chunks_exact_mut(seq_len).enumerate() {
                for (key, value) in row.iter_mut().enumerate() {
                    *value = match row_index % 5 {
                        0 => 3.25,
                        1 => {
                            if key == row_index % seq_len {
                                0.0
                            } else {
                                -12.0
                            }
                        }
                        2 => {
                            if key % 2 == 0 {
                                0.0
                            } else {
                                -8.0
                            }
                        }
                        3 if key != 0 => {
                            let gap = ((key + row_index) % 35) as f64 + 0.5;
                            let boundary = (-gap * std::f64::consts::LN_2) as f32;
                            match key % 3 {
                                0 => boundary.next_down(),
                                1 => boundary,
                                _ => boundary.next_up(),
                            }
                        }
                        3 => 0.0,
                        _ => {
                            if key == seq_len - 1 {
                                0.0
                            } else {
                                -(((key + row_index) % 25) as f32)
                            }
                        }
                    };
                }
            }
            let mut direct = input.clone();
            // SAFETY: AVX2 and FMA were detected above.
            unsafe { softmax_attention_avx2_fma(&mut direct, seq_len, 2) };
            assert_normalized_rows(&input, &direct, seq_len);
            let mut dispatched = input.clone();
            softmax_attention(&mut dispatched, seq_len, 2);
            for (index, (got, want)) in dispatched.iter().zip(&direct).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "seq_len={seq_len} index={index}"
                );
            }
        }
        eprintln!("softmax avx2-fma: completed normalized long/tail and dispatch controls");
    }

    #[test]
    fn special_lanes_preserve_masks_and_neighboring_rows() {
        let config = simd_config();
        if !config.avx2_enabled || !config.fma_enabled {
            return;
        }
        for seq_len in [9, 17, 33] {
            let mut lanes = vec![0, 7, 8, seq_len - 1];
            lanes.sort_unstable();
            lanes.dedup();
            for lane in lanes {
                for special in [f32::NEG_INFINITY, f32::NAN, f32::INFINITY] {
                    let mut input = vec![0.5; 2 * seq_len * seq_len];
                    input[lane] = special;
                    let last_row = input.len() - seq_len;
                    input[last_row + lane] = special;
                    let mut output = input.clone();
                    // SAFETY: AVX2 and FMA were detected above.
                    unsafe { softmax_attention_avx2_fma(&mut output, seq_len, 2) };
                    assert_normalized_rows(&input, &output, seq_len);
                }
            }
            let mut input = vec![0.5; 2 * seq_len * seq_len];
            input[..seq_len].fill(f32::NEG_INFINITY);
            let last_row = input.len() - seq_len;
            input[last_row..].fill(f32::NEG_INFINITY);
            let mut output = input.clone();
            // SAFETY: AVX2 and FMA were detected above.
            unsafe { softmax_attention_avx2_fma(&mut output, seq_len, 2) };
            assert_normalized_rows(&input, &output, seq_len);
        }
        eprintln!("softmax avx2-fma: completed special-lane and neighboring-row controls");
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_degree5_matches_f64_reference() {
    if !simd_config().avx2_enabled {
        return;
    }
    eprintln!("softmax avx2 degree5: exercising SIMD body");
    let run = |x: &mut [f32], seq_len, num_heads| {
        // SAFETY: AVX2 was detected above.
        unsafe { softmax_attention_avx2(x, seq_len, num_heads) };
    };
    assert_matches_f64_reference("avx2-degree5", run);
    assert_special_rows("avx2-degree5", run);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_degree5_matches_f64_reference() {
    eprintln!("softmax neon degree5: exercising SIMD body");
    let run = |x: &mut [f32], seq_len, num_heads| {
        // SAFETY: NEON is guaranteed on aarch64.
        unsafe { softmax_attention_neon(x, seq_len, num_heads) };
    };
    assert_matches_f64_reference("neon-degree5", run);
    assert_special_rows("neon-degree5", run);
}
