use super::*;

fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "index {i}: {x} vs {y} (delta {})",
            (x - y).abs()
        );
    }
}

#[test]
fn adjacent_block_sign_streams_are_not_shifted_copies() {
    // Regression test: with the naive
    // `seed + i * GAMMA` block-seed derivation, `signs_from_seed`'s own
    // per-output `+GAMMA` state advance made block i+1's sign vector
    // EXACTLY block i's shifted by one position (block_size - 1 shared
    // PRNG states). Mutation check: revert `derive_block_seed` to the
    // naive spaced form and the shifted-copy assertion below fails for
    // every (seed, block) pair; the avalanche-mixed derivation must not
    // exhibit the shift alignment for any adjacent pair.
    for seed in [0u64, 1, 42, 0xDEAD_BEEF] {
        for b in [64usize, 128, 256] {
            let s0 = signs_from_seed(derive_block_seed(seed, 0), b);
            let s1 = signs_from_seed(derive_block_seed(seed, 1), b);
            // The old defect: s1[..b-1] == s0[1..] elementwise.
            let shifted_copy = s1[..b - 1]
                .iter()
                .zip(&s0[1..])
                .all(|(a, c)| a.to_bits() == c.to_bits());
            assert!(
                !shifted_copy,
                "seed {seed}, block_size {b}: adjacent blocks' sign streams \
                     are one-position-shifted copies (shared PRNG states)"
            );
        }
    }
}

#[test]
fn block_hadamard_rejects_usize_max_length_without_allocating() {
    // #703 security finding: n=usize::MAX with block_size=1 previously
    // passed every existing check and then performed an infallible
    // `Vec::with_capacity`-style allocation for `usize::MAX` blocks,
    // aborting the process. The n-bound check must reject this before
    // any block allocation — this test must never actually allocate.
    let err = BlockHadamard::new(0, usize::MAX, 1).unwrap_err();
    assert!(
        format!("{err}").contains("MAX_BLOCK_HADAMARD_LEN")
            || format!("{err}").to_lowercase().contains("n <="),
        "expected an n-bound rejection, got: {err}"
    );
}

#[test]
#[ignore = "stress test: constructs a single block at the 16M-element \
                cap, deriving a 64 MiB sign buffer — run explicitly with \
                `cargo test -- --ignored` rather than on every default run"]
fn block_hadamard_accepts_n_at_the_cap_boundary() {
    // The cap itself must still be constructible — block_size ==
    // MAX_BLOCK_HADAMARD_LEN keeps num_blocks at exactly 1. Verifying
    // acceptance at the true boundary means constructing at the true
    // boundary; there is no way to assert this without materializing
    // the full sign buffer, hence `#[ignore]` rather than running it
    // on every default `cargo test`.
    let bh = BlockHadamard::new(0, MAX_BLOCK_HADAMARD_LEN, MAX_BLOCK_HADAMARD_LEN).unwrap();
    assert_eq!(bh.dim(), MAX_BLOCK_HADAMARD_LEN);
    assert_eq!(bh.num_blocks(), 1);
}

#[test]
fn block_hadamard_rejects_n_one_past_the_cap() {
    // cap+1 with block_size=1 must fail on the n-bound, not on
    // divisibility or block_size validity — this exercises the exact
    // boundary the maintainer specified.
    let err = BlockHadamard::new(0, MAX_BLOCK_HADAMARD_LEN + 1, 1).unwrap_err();
    assert!(
        format!("{err}").contains(&(MAX_BLOCK_HADAMARD_LEN).to_string()),
        "expected the cap value in the error, got: {err}"
    );
}

#[test]
fn block_hadamard_rejects_num_blocks_above_cap_without_allocating() {
    // n = 1 << 24, block_size = 1
    // passes the existing n-cap (n is exactly at MAX_BLOCK_HADAMARD_LEN)
    // and every block_size check, but requests num_blocks = 1 << 24
    // (16,777,216) blocks, each needing its own per-block seed
    // derivation. The num_blocks cap must reject this before the
    // per-block signs loop runs, so this test itself never performs
    // that work.
    let err = BlockHadamard::new(0, 1 << 24, 1).unwrap_err();
    assert!(
        format!("{err}").contains(&MAX_BLOCK_HADAMARD_BLOCKS.to_string())
            || format!("{err}").to_lowercase().contains("blocks"),
        "expected a num_blocks-bound rejection, got: {err}"
    );
}

#[test]
fn block_hadamard_accepts_num_blocks_at_the_cap_boundary() {
    // The num_blocks cap itself must still be constructible: exactly
    // MAX_BLOCK_HADAMARD_BLOCKS blocks of block_size 1 stays cheap
    // (small n) even though num_blocks is at the documented bound.
    let bh = BlockHadamard::new(0, MAX_BLOCK_HADAMARD_BLOCKS, 1).unwrap();
    assert_eq!(bh.num_blocks(), MAX_BLOCK_HADAMARD_BLOCKS);
    assert_eq!(bh.dim(), MAX_BLOCK_HADAMARD_BLOCKS);
}

#[test]
fn block_hadamard_accepts_large_valid_n_via_single_fallible_reservation() {
    // A large-but-valid single-block construction (num_blocks = 1, well
    // past any real model's rotation axis) must still succeed via one
    // fallible `try_reserve_exact` reservation, not an infallible
    // `Vec::with_capacity`, and its signs must match the per-block
    // reference exactly — proving the single-buffer path didn't change
    // sign VALUES while making the allocation fallible. (The exact
    // MAX_BLOCK_HADAMARD_LEN boundary is covered, construction-only, by
    // `block_hadamard_accepts_n_at_the_cap_boundary`; this uses a smaller
    // large `n` so the Walsh-Hadamard transform below stays cheap.)
    let n = 1 << 16;
    let bh = BlockHadamard::new(0x5EED, n, n).unwrap();
    assert_eq!(bh.num_blocks(), 1);
    assert_eq!(bh.dim(), n);

    let expected = RandomizedHadamard::new(derive_block_seed(0x5EED, 0), n).unwrap();
    let mut lhs = vec![1.0_f32; n];
    let mut rhs = vec![1.0_f32; n];
    bh.apply(&mut lhs).unwrap();
    expected.apply(&mut rhs).unwrap();
    assert_eq!(
        lhs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        rhs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "large-but-valid single-block construction must match the \
             per-block reference bit-for-bit"
    );
}

#[test]
fn block_hadamard_signs_match_per_block_randomized_hadamard() {
    // The flat contiguous `signs`
    // buffer must produce byte-identical rotations to the prior
    // `Vec<RandomizedHadamard>` layout (one struct + one signs `Vec` per
    // block). Prove it end-to-end rather than by comparing private sign
    // vectors: rotate the same input two ways — the new `BlockHadamard`,
    // and a manual per-block loop that reconstructs exactly what the old
    // layout did (`RandomizedHadamard::new(derive_block_seed(seed, i),
    // block_size)` applied to block `i`'s slice) — and assert the outputs
    // agree bit-for-bit. This covers both the per-block sign VALUES
    // (`signs_from_seed(derive_block_seed(...))`) and the `D`-then-`H`
    // operation order, so a drift in either would fail the assertion.
    for (seed, block_size, num_blocks) in [
        (0x51ED_u64, 4usize, 5usize),
        (42, 8, 3),
        (0xDEAD_BEEF, 2, 7),
    ] {
        let n = block_size * num_blocks;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();

        // New layout (flat contiguous signs).
        let bh = BlockHadamard::new(seed, n, block_size).unwrap();
        let mut new_out = input.clone();
        bh.apply(&mut new_out).unwrap();

        // Old layout, reconstructed: one RandomizedHadamard per block,
        // same `derive_block_seed(seed, i)`, applied to that block's slice.
        let mut old_out = input.clone();
        for (i, chunk) in old_out.chunks_mut(block_size).enumerate() {
            let rh = RandomizedHadamard::new(derive_block_seed(seed, i), block_size).unwrap();
            rh.apply(chunk).unwrap();
        }

        assert_eq!(
            new_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            old_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "seed {seed}, block_size {block_size}, num_blocks {num_blocks}: \
                 flat-signs BlockHadamard must match the per-block \
                 RandomizedHadamard layout bit-for-bit"
        );

        // The refactored inverse must still round-trip the forward rotation.
        let mut round = new_out.clone();
        bh.apply_inverse(&mut round).unwrap();
        approx_eq(&round, &input, 1e-5);
    }
}

#[test]
fn walsh_hadamard_size_1_is_identity() {
    let mut data = [3.5_f32];
    walsh_hadamard_in_place(&mut data).unwrap();
    approx_eq(&data, &[3.5], 1e-6);
}

#[test]
fn walsh_hadamard_size_2_known_result() {
    let mut data = [1.0_f32, 2.0];
    walsh_hadamard_in_place(&mut data).unwrap();
    approx_eq(&data, &[3.0, -1.0], 1e-6);
}

#[test]
fn walsh_hadamard_size_4_known_result() {
    let mut data = [1.0_f32, 0.0, 0.0, 0.0];
    walsh_hadamard_in_place(&mut data).unwrap();
    approx_eq(&data, &[1.0, 1.0, 1.0, 1.0], 1e-6);
}

#[test]
fn walsh_hadamard_double_application_scales_by_n() {
    let n = 16;
    let original: Vec<f32> = (0..n).map(|i| i as f32 - 8.0).collect();
    let mut data = original.clone();
    walsh_hadamard_in_place(&mut data).unwrap();
    walsh_hadamard_in_place(&mut data).unwrap();
    let scaled: Vec<f32> = original.iter().map(|&x| x * n as f32).collect();
    approx_eq(&data, &scaled, 1e-4);
}

#[test]
fn walsh_hadamard_orthonormal_is_isometry() {
    let original: Vec<f32> = (0..32).map(|i| (i as f32 * 0.137).sin()).collect();
    let original_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut data = original.clone();
    walsh_hadamard_orthonormal_in_place(&mut data).unwrap();
    let transformed_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (original_norm - transformed_norm).abs() < 1e-4,
        "||x||={original_norm} vs ||Hx||={transformed_norm}"
    );
}

#[test]
fn walsh_hadamard_orthonormal_is_involution() {
    let original: Vec<f32> = (0..64).map(|i| (i as f32 * 0.71).cos()).collect();
    let mut data = original.clone();
    walsh_hadamard_orthonormal_in_place(&mut data).unwrap();
    walsh_hadamard_orthonormal_in_place(&mut data).unwrap();
    approx_eq(&data, &original, 1e-4);
}

#[test]
fn walsh_hadamard_rejects_non_power_of_two() {
    let mut data = [1.0_f32, 2.0, 3.0];
    assert!(walsh_hadamard_in_place(&mut data).is_err());
}

#[test]
fn walsh_hadamard_rejects_empty() {
    let mut data: [f32; 0] = [];
    assert!(walsh_hadamard_in_place(&mut data).is_err());
}

#[test]
fn randomized_hadamard_is_orthogonal() {
    let r = RandomizedHadamard::new(42, 128).unwrap();
    let original: Vec<f32> = (0..128).map(|i| (i as f32 * 0.13).sin()).collect();
    let original_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut data = original.clone();
    r.apply(&mut data).unwrap();
    let transformed_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (original_norm - transformed_norm).abs() < 1e-3,
        "||x||={original_norm} vs ||Rx||={transformed_norm}"
    );
}

#[test]
fn randomized_hadamard_inverse_round_trips() {
    let r = RandomizedHadamard::new(0xDEAD_BEEF, 256).unwrap();
    let original: Vec<f32> = (0..256).map(|i| (i as f32 * 0.41).cos() + 0.3).collect();
    let mut data = original.clone();
    r.apply(&mut data).unwrap();
    r.apply_inverse(&mut data).unwrap();
    approx_eq(&data, &original, 1e-3);
}

#[test]
fn randomized_hadamard_seed_determinism() {
    let r1 = RandomizedHadamard::new(7, 64).unwrap();
    let r2 = RandomizedHadamard::new(7, 64).unwrap();
    let mut a: Vec<f32> = (0..64).map(|i| i as f32 + 0.5).collect();
    let mut b = a.clone();
    r1.apply(&mut a).unwrap();
    r2.apply(&mut b).unwrap();
    assert_eq!(a, b, "same seed must produce bit-identical output");
}

#[test]
fn randomized_hadamard_seed_differs() {
    let r1 = RandomizedHadamard::new(7, 64).unwrap();
    let r2 = RandomizedHadamard::new(8, 64).unwrap();
    let mut a: Vec<f32> = (0..64).map(|i| i as f32 + 0.5).collect();
    let mut b = a.clone();
    r1.apply(&mut a).unwrap();
    r2.apply(&mut b).unwrap();
    let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    assert!(
        diff > 1.0,
        "different seeds should produce different output"
    );
}

#[test]
fn randomized_hadamard_rejects_length_mismatch() {
    let r = RandomizedHadamard::new(1, 16).unwrap();
    let mut data = vec![0.0_f32; 8];
    assert!(r.apply(&mut data).is_err());
}

fn approx_eq_f64(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "index {i}: {x} vs {y} (delta {})",
            (x - y).abs()
        );
    }
}

#[test]
fn walsh_hadamard_f64_size_2_known_result() {
    let mut data = [1.0_f64, 2.0];
    walsh_hadamard_f64_in_place(&mut data).unwrap();
    approx_eq_f64(&data, &[3.0, -1.0], 1e-12);
}

#[test]
fn walsh_hadamard_f64_orthonormal_is_isometry() {
    let original: Vec<f64> = (0..32).map(|i| (i as f64 * 0.137).sin()).collect();
    let original_norm: f64 = original.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut data = original.clone();
    walsh_hadamard_orthonormal_f64_in_place(&mut data).unwrap();
    let transformed_norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        (original_norm - transformed_norm).abs() < 1e-12,
        "||x||={original_norm} vs ||Hx||={transformed_norm}"
    );
}

#[test]
fn walsh_hadamard_f64_orthonormal_is_involution() {
    let original: Vec<f64> = (0..64).map(|i| (i as f64 * 0.71).cos()).collect();
    let mut data = original.clone();
    walsh_hadamard_orthonormal_f64_in_place(&mut data).unwrap();
    walsh_hadamard_orthonormal_f64_in_place(&mut data).unwrap();
    approx_eq_f64(&data, &original, 1e-12);
}

#[test]
fn randomized_hadamard_f64_inverse_round_trips() {
    let r = RandomizedHadamard::new(0xDEAD_BEEF, 256).unwrap();
    let original: Vec<f64> = (0..256).map(|i| (i as f64 * 0.41).cos() + 0.3).collect();
    let mut data = original.clone();
    r.apply_f64(&mut data).unwrap();
    r.apply_inverse_f64(&mut data).unwrap();
    approx_eq_f64(&data, &original, 1e-12);
}

#[test]
fn randomized_hadamard_f32_f64_agree_in_precision() {
    let r = RandomizedHadamard::new(7, 256).unwrap();
    let mut f32_data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.13).sin()).collect();
    let mut f64_data: Vec<f64> = f32_data.iter().map(|&x| x as f64).collect();
    r.apply(&mut f32_data).unwrap();
    r.apply_f64(&mut f64_data).unwrap();
    for (i, (a, b)) in f32_data.iter().zip(f64_data.iter()).enumerate() {
        let delta = (*a as f64 - b).abs();
        assert!(delta < 1e-5, "index {i}: f32={a} vs f64={b}, delta={delta}");
    }
}

#[test]
fn randomized_hadamard_outlier_redistribution() {
    let r = RandomizedHadamard::new(123, 1024).unwrap();
    let mut data = vec![0.0_f32; 1024];
    data[42] = 100.0;
    let pre_max = data.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
    r.apply(&mut data).unwrap();
    let post_max = data.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
    assert!(
        post_max < pre_max * 0.5,
        "outlier should be redistributed: pre_max={pre_max}, post_max={post_max}"
    );
}

// --- BlockHadamard ---

fn synthetic_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 11) as u32;
            (bits as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

#[test]
fn block_hadamard_round_trips() {
    for &(n, b) in &[(3584usize, 64usize), (3584, 128), (3584, 256), (17408, 256)] {
        let bh = BlockHadamard::new(0xB10C_5EED, n, b).unwrap();
        let original = synthetic_vec(n, 7);
        let mut data = original.clone();
        bh.apply(&mut data).unwrap();
        bh.apply_inverse(&mut data).unwrap();
        approx_eq(&data, &original, 1e-3);
    }
}

#[test]
fn block_hadamard_preserves_norm() {
    let bh = BlockHadamard::new(42, 3584, 128).unwrap();
    let original = synthetic_vec(3584, 11);
    let original_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut data = original.clone();
    bh.apply(&mut data).unwrap();
    let transformed_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (original_norm - transformed_norm).abs() < 1e-2,
        "||x||={original_norm} vs ||Rx||={transformed_norm}"
    );
}

#[test]
fn block_hadamard_exact_partition_coverage() {
    // Every element belongs to exactly one block: perturbing element i
    // in isolation must only change output elements within i's own
    // block, for every valid (n, b) pair the design doc names.
    for &(n, b) in &[
        (3584usize, 64usize),
        (3584, 128),
        (3584, 256),
        (17408, 64),
        (17408, 128),
        (17408, 256),
    ] {
        assert_eq!(n % b, 0, "test precondition: b must divide n");
        let bh = BlockHadamard::new(0xC0FF_EE00, n, b).unwrap();
        assert_eq!(bh.num_blocks(), n / b);
        assert_eq!(bh.dim(), n);

        let base = synthetic_vec(n, 3);
        let mut perturbed = base.clone();
        let target = n / 2 + 3; // arbitrary interior element
        perturbed[target] += 5.0;

        let mut out_base = base.clone();
        bh.apply(&mut out_base).unwrap();
        let mut out_perturbed = perturbed.clone();
        bh.apply(&mut out_perturbed).unwrap();

        let block_start = (target / b) * b;
        let block_end = block_start + b;
        for i in 0..n {
            let changed = (out_base[i] - out_perturbed[i]).abs() > 1e-6;
            if i >= block_start && i < block_end {
                // inside target's block: allowed (not required) to change
            } else {
                assert!(
                    !changed,
                    "element {i} outside block [{block_start},{block_end}) \
                         changed when only element {target} was perturbed \
                         (n={n}, b={b}) — partition coverage violated"
                );
            }
        }
    }
}

#[test]
fn block_hadamard_rejects_block_size_not_dividing_n() {
    // 1024 is a power of two but 3584 (= 2^9 * 7) is not a multiple of
    // it (3584 / 1024 = 3.5) — this exercises the divisibility check on
    // its own. A non-power-of-two block_size like 100 would be caught
    // by the earlier power-of-two check regardless of divisibility,
    // making that check redundant and leaving this test green even if
    // the divisibility check were removed.
    assert!(BlockHadamard::new(1, 3584, 1024).is_err());
}

#[test]
fn block_hadamard_rejects_non_power_of_two_block_size() {
    // 3584 is divisible by 3584/7=512... use a divisor that isn't a
    // power of two: 3584 = 2^9 * 7, so block_size=7 divides evenly but
    // is not a power of two.
    assert!(BlockHadamard::new(1, 3584, 7).is_err());
}

#[test]
fn block_hadamard_rejects_zero_length() {
    assert!(BlockHadamard::new(1, 0, 64).is_err());
}

#[test]
fn block_hadamard_rejects_zero_block_size() {
    assert!(BlockHadamard::new(1, 3584, 0).is_err());
}

#[test]
fn block_hadamard_rejects_length_mismatch_on_apply() {
    let bh = BlockHadamard::new(1, 3584, 128).unwrap();
    let mut data = vec![0.0_f32; 100];
    assert!(bh.apply(&mut data).is_err());
}

/// A decoder built from the wrong seed must not reproduce the original
/// vector, and a decoder built from the matching seed must reproduce it
/// exactly — proving the round trip is sensitive to which seed derived
/// the block signs, not just whether a seed was supplied at all.
#[test]
fn block_hadamard_mutation_sensitive_round_trip() {
    let n = 3584;
    let b = 128;
    let good_seed = 0xB10C_5EED;
    let corrupted_seed = 0xB10C_5EED ^ 0x1; // flips the derived signs

    let original = synthetic_vec(n, 7);

    // A decoder seeded differently from the encoder must diverge from
    // the original vector on decode.
    let encoder = BlockHadamard::new(good_seed, n, b).unwrap();
    let corrupted_decoder = BlockHadamard::new(corrupted_seed, n, b).unwrap();
    let mut data = original.clone();
    encoder.apply(&mut data).unwrap();
    corrupted_decoder.apply_inverse(&mut data).unwrap();
    let corrupted_delta: f32 = data
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(
        corrupted_delta > 1e-2,
        "mismatched-seed round trip should diverge from the original, got delta {corrupted_delta}"
    );

    // A decoder seeded to match the encoder reproduces the original
    // exactly, confirming the divergence above traces to the seed
    // mismatch and not some other defect.
    let restored_decoder = BlockHadamard::new(good_seed, n, b).unwrap();
    let mut data_restored = original.clone();
    encoder.apply(&mut data_restored).unwrap();
    restored_decoder.apply_inverse(&mut data_restored).unwrap();
    approx_eq(&data_restored, &original, 1e-3);
}
