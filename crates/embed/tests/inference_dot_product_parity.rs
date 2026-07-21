//! Cross-crate synchronization test between `lattice-embed`'s dot product
//! and `lattice-inference`'s GEMM row kernel (ADR-013's scope-boundary
//! note).
//!
//! `lattice-inference::forward::cpu::matmul_bt` and
//! `lattice-embed::simd::dot_product` are two independently-tuned
//! implementations that intentionally do not share code (see ADR-013):
//! different accumulator counts, different SIMD widths, different
//! reassociation order. This test compares them on identical inputs at a
//! `1 x K` by `1 x K` reduction (where `matmul_bt` degenerates to a dot
//! product) using a scale-aware tolerance.
//!
//! This is a synchronization alarm, not a correctness oracle: it proves the
//! two implementations haven't silently drifted apart on shared inputs.
//! Each crate's own scalar-equivalence and architecture-specific tests
//! remain authoritative for that implementation's intrinsic correctness.
//!
//! Gated on the `native` feature (default-on): `lattice-inference` is an
//! optional dependency behind that feature, so this file is skipped
//! entirely on a `--no-default-features` build.

#![cfg(feature = "native")]

use lattice_embed::simd::dot_product;
use lattice_inference::forward::cpu::matmul_bt;

/// Run `matmul_bt` as a `1 x K` by `1 x K` transposed-B product, which
/// reduces to a single dot product for `m = n = 1`.
fn inference_dot(a: &[f32], b: &[f32]) -> f32 {
    let k = a.len();
    assert_eq!(b.len(), k, "test helper requires equal-length inputs");
    let mut c = [0.0f32; 1];
    matmul_bt(a, b, &mut c, 1, k, 1);
    c[0]
}

/// Scale-aware comparison: bit-exact agreement is not the contract (the two
/// kernels reassociate differently), so the tolerance scales with the
/// magnitude of the terms actually being summed rather than with the final
/// (possibly near-zero, under cancellation) result.
fn assert_kernels_agree(a: &[f32], b: &[f32], label: &str) {
    let embed = dot_product(a, b);
    let inference = inference_dot(a, b);
    let abs_sum: f32 = a.iter().zip(b).map(|(x, y)| (x * y).abs()).sum();
    let tol = 1e-5 * abs_sum + 1e-3;
    let diff = (embed - inference).abs();
    assert!(
        diff <= tol,
        "{label} (len={}): embed dot_product={embed:e} vs inference matmul_bt={inference:e}, \
         diff={diff:e} > tol={tol:e} (abs_sum={abs_sum:e})",
        a.len(),
    );
}

/// Deterministic pseudo-random vector in `[-1, 1]`.
fn generate_vector(dim: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..dim)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            let hash = hasher.finish();
            (hash as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn boundary_lengths_agree() {
    // Spans both crates' SIMD unroll widths (embed: 8-wide AVX2 with a
    // 4x/8x unroll and a NEON/WASM-SIMD128 path; inference: 4- and
    // 8-accumulator scalar unrolls plus NEON/AVX2/AVX-512/Accelerate
    // dispatch) and their scalar tails, from single-element up through
    // multi-kilobyte vectors.
    for len in [
        1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256,
        257, 383, 384, 385, 512, 769, 1000, 1536,
    ] {
        let a = generate_vector(len, 0xA11CE);
        let b = generate_vector(len, 0xB0B);
        assert_kernels_agree(&a, &b, "boundary_lengths_agree");
    }
}

#[test]
fn dim_384_specialization_agrees() {
    // `lattice_embed`'s AVX2 path has a dedicated 384-element kernel
    // distinct from its general chunked loop; 384 is also this
    // workspace's default embedding dimension (BGE-small).
    for seed in [1u64, 2, 3, 4, 5] {
        let a = generate_vector(384, seed);
        let b = generate_vector(384, seed.wrapping_mul(7919));
        assert_kernels_agree(&a, &b, "dim_384_specialization_agrees");
    }
}

#[test]
fn cancellation_heavy_inputs_agree() {
    // Large-magnitude, opposite-sign, near-cancelling terms: the true dot
    // product is small relative to the individual products, which is where
    // accumulator count and pairing order matter most and where a
    // reassociation or sign bug is easiest to hide behind a "looks close to
    // zero either way" false pass — hence the tolerance is scaled off the
    // summed absolute products (`abs_sum`), not off the near-zero result.
    for len in [8usize, 64, 384, 1000] {
        let mut a = Vec::with_capacity(len);
        let mut b = Vec::with_capacity(len);
        for i in 0..len {
            let mag = 1.0e4 * (1.0 + (i % 7) as f32);
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            a.push(sign * mag + (i as f32 * 1e-3));
            b.push(mag);
        }
        assert_kernels_agree(&a, &b, "cancellation_heavy_inputs_agree");
    }
}
