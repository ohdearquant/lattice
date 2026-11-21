//! Unsafe AVX2 tiled transposed-B matmul implementation.
// ---------------------------------------------------------------------------
// AVX2+FMA tiled matmul_bt — cache-blocked with 4×8 tiles of 2×4 AVX2 blocks
// ---------------------------------------------------------------------------
//
// Same tile structure as NEON (4i × 8j), but uses 256-bit AVX2 lanes (8 f32s).
// A full tile is computed as four 2×4 register blocks (eight accumulators each).
//

#[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
use super::arch_kernels::hsum_m256;
#[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
use super::tiled::{TILE_I, TILE_J, TILE_K};

// The fast path below hardcodes the four 2×4 blocks of a 4×8 tile.
#[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
const _: () = assert!(TILE_I == 4 && TILE_J == 8);

#[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
#[target_feature(enable = "avx2", enable = "fma")]
pub(super) unsafe fn matmul_bt_tiled_avx2(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    let mut j_start = 0;
    while j_start < n {
        let j_end = (j_start + TILE_J).min(n);
        let j_count = j_end - j_start;

        let mut i_start = 0;
        while i_start < m {
            let i_end = (i_start + TILE_I).min(m);
            let i_count = i_end - i_start;

            let mut k_start = 0;
            while k_start < k {
                let k_end = (k_start + TILE_K).min(k);
                let k_len = k_end - k_start;

                // Fast path: full 4×8 tile with k_len >= 16. The tile runs as four 2×4
                // blocks. A block needs eight accumulators, four B vectors and one A
                // vector, 13 of the 16 AVX2 registers, where the whole tile needed 32
                // accumulators. Every output still sums its K-tile in the same 8-lane FMA
                // chain and the same reduction as the 4×8 kernel this replaced.
                if i_count == TILE_I && j_count == TILE_J && k_len >= 16 {
                    let k_vec_end = k_start + (k_len / 8) * 8;
                    for ii in [0, 2] {
                        for jj in [0, 4] {
                            // SAFETY: rows i_start + ii .. + 2 lie in the full TILE_I tile and
                            // columns j_start + jj .. + 4 in the full TILE_J tile, and
                            // k_vec_end <= k_end <= k, so every 8-wide load is in bounds.
                            block_2x4(
                                a_ptr,
                                b_ptr,
                                c_ptr,
                                i_start + ii,
                                j_start + jj,
                                k,
                                n,
                                k_start,
                                k_vec_end,
                            );
                        }
                    }

                    // Scalar remainder for k not covered by AVX2 loads
                    let k_scalar_start = k_start + (k_len / 8) * 8;
                    if k_scalar_start < k_end {
                        for ii in 0..TILE_I {
                            let i = i_start + ii;
                            for jj in 0..TILE_J {
                                let j = j_start + jj;
                                let mut sum = 0.0f32;
                                for p in k_scalar_start..k_end {
                                    sum += *a_ptr.add(i * k + p) * *b_ptr.add(j * k + p);
                                }
                                *c_ptr.add(i * n + j) += sum;
                            }
                        }
                    }
                } else {
                    // Edge tiles: scalar fallback
                    for ii in 0..i_count {
                        let i = i_start + ii;
                        for jj in 0..j_count {
                            let j = j_start + jj;
                            let mut sum = 0.0f32;
                            for p in k_start..k_end {
                                sum += *a_ptr.add(i * k + p) * *b_ptr.add(j * k + p);
                            }
                            *c_ptr.add(i * n + j) += sum;
                        }
                    }
                }

                k_start += TILE_K;
            }
            i_start += TILE_I;
        }
        j_start += TILE_J;
    }
}

/// One 2×4 block of a full tile: C[i0..i0+2, j0..j0+4] += hsum over k_start..k_vec_end of
/// the 8-lane FMA chain A[i] · B[j], each output reduced by `hsum_m256`.
///
/// # Safety
///
/// AVX2 and FMA must be available. Rows `i0..i0 + 2` of `a` and `c`, rows `j0..j0 + 4` of
/// `b`, and columns `j0..j0 + 4` of `c` must exist, with `k_vec_end <= k`.
#[cfg(all(target_arch = "x86_64", not(target_os = "macos")))]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn block_2x4(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    i0: usize,
    j0: usize,
    k: usize,
    n: usize,
    k_start: usize,
    k_vec_end: usize,
) {
    use std::arch::x86_64::*;

    let a0 = a_ptr.add(i0 * k);
    let a1 = a_ptr.add((i0 + 1) * k);
    let b0 = b_ptr.add(j0 * k);
    let b1 = b_ptr.add((j0 + 1) * k);
    let b2 = b_ptr.add((j0 + 2) * k);
    let b3 = b_ptr.add((j0 + 3) * k);

    let mut r0 = [_mm256_setzero_ps(); 4];
    let mut r1 = [_mm256_setzero_ps(); 4];
    let mut ko = k_start;
    while ko < k_vec_end {
        let bv0 = _mm256_loadu_ps(b0.add(ko));
        let bv1 = _mm256_loadu_ps(b1.add(ko));
        let bv2 = _mm256_loadu_ps(b2.add(ko));
        let bv3 = _mm256_loadu_ps(b3.add(ko));

        let av = _mm256_loadu_ps(a0.add(ko));
        r0[0] = _mm256_fmadd_ps(av, bv0, r0[0]);
        r0[1] = _mm256_fmadd_ps(av, bv1, r0[1]);
        r0[2] = _mm256_fmadd_ps(av, bv2, r0[2]);
        r0[3] = _mm256_fmadd_ps(av, bv3, r0[3]);

        let av = _mm256_loadu_ps(a1.add(ko));
        r1[0] = _mm256_fmadd_ps(av, bv0, r1[0]);
        r1[1] = _mm256_fmadd_ps(av, bv1, r1[1]);
        r1[2] = _mm256_fmadd_ps(av, bv2, r1[2]);
        r1[3] = _mm256_fmadd_ps(av, bv3, r1[3]);

        ko += 8;
    }

    let c0 = c_ptr.add(i0 * n + j0);
    let c1 = c_ptr.add((i0 + 1) * n + j0);
    for jj in 0..4 {
        *c0.add(jj) += hsum_m256(r0[jj]);
        *c1.add(jj) += hsum_m256(r1[jj]);
    }
}
