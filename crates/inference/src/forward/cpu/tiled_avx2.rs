// ---------------------------------------------------------------------------
// AVX2+FMA tiled matmul_bt — cache-blocked with 4×8 AVX2 microkernel
// ---------------------------------------------------------------------------
//
// Same tile structure as NEON (4i × 8j), but uses 256-bit AVX2 lanes (8 f32s).
// Inner k-loop unrolled 2× (processes 16 k-values per iteration).
//

#[cfg(target_arch = "x86_64")]
use super::arch_kernels::hsum_m256;
#[cfg(target_arch = "x86_64")]
use super::tiled::{TILE_I, TILE_J, TILE_K};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub(super) unsafe fn matmul_bt_tiled_avx2(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use std::arch::x86_64::*;

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

                // Fast path: full 4×8 tile with k_len >= 16 (AVX2 2× unrolled)
                if i_count == TILE_I && j_count == TILE_J && k_len >= 16 {
                    // SAFETY: Same bounds reasoning as NEON path.
                    // AVX2 loads read 8 contiguous f32s; all offsets stay within bounds.
                    let mut acc0 = [_mm256_setzero_ps(); TILE_J];
                    let mut acc1 = [_mm256_setzero_ps(); TILE_J];
                    let mut acc2 = [_mm256_setzero_ps(); TILE_J];
                    let mut acc3 = [_mm256_setzero_ps(); TILE_J];

                    let a0_base = a_ptr.add(i_start * k);
                    let a1_base = a_ptr.add((i_start + 1) * k);
                    let a2_base = a_ptr.add((i_start + 2) * k);
                    let a3_base = a_ptr.add((i_start + 3) * k);

                    let b0_base = b_ptr.add(j_start * k);
                    let b1_base = b_ptr.add((j_start + 1) * k);
                    let b2_base = b_ptr.add((j_start + 2) * k);
                    let b3_base = b_ptr.add((j_start + 3) * k);
                    let b4_base = b_ptr.add((j_start + 4) * k);
                    let b5_base = b_ptr.add((j_start + 5) * k);
                    let b6_base = b_ptr.add((j_start + 6) * k);
                    let b7_base = b_ptr.add((j_start + 7) * k);

                    // Main loop: 16 k-values per iteration (2× unrolled 8-wide AVX2)
                    let k_pairs = k_len / 16;
                    for kp in 0..k_pairs {
                        let ko = k_start + kp * 16;

                        // --- First group of 8 k-values ---
                        let bv0a = _mm256_loadu_ps(b0_base.add(ko));
                        let bv1a = _mm256_loadu_ps(b1_base.add(ko));
                        let bv2a = _mm256_loadu_ps(b2_base.add(ko));
                        let bv3a = _mm256_loadu_ps(b3_base.add(ko));
                        let bv4a = _mm256_loadu_ps(b4_base.add(ko));
                        let bv5a = _mm256_loadu_ps(b5_base.add(ko));
                        let bv6a = _mm256_loadu_ps(b6_base.add(ko));
                        let bv7a = _mm256_loadu_ps(b7_base.add(ko));

                        let av = _mm256_loadu_ps(a0_base.add(ko));
                        acc0[0] = _mm256_fmadd_ps(av, bv0a, acc0[0]);
                        acc0[1] = _mm256_fmadd_ps(av, bv1a, acc0[1]);
                        acc0[2] = _mm256_fmadd_ps(av, bv2a, acc0[2]);
                        acc0[3] = _mm256_fmadd_ps(av, bv3a, acc0[3]);
                        acc0[4] = _mm256_fmadd_ps(av, bv4a, acc0[4]);
                        acc0[5] = _mm256_fmadd_ps(av, bv5a, acc0[5]);
                        acc0[6] = _mm256_fmadd_ps(av, bv6a, acc0[6]);
                        acc0[7] = _mm256_fmadd_ps(av, bv7a, acc0[7]);

                        let av = _mm256_loadu_ps(a1_base.add(ko));
                        acc1[0] = _mm256_fmadd_ps(av, bv0a, acc1[0]);
                        acc1[1] = _mm256_fmadd_ps(av, bv1a, acc1[1]);
                        acc1[2] = _mm256_fmadd_ps(av, bv2a, acc1[2]);
                        acc1[3] = _mm256_fmadd_ps(av, bv3a, acc1[3]);
                        acc1[4] = _mm256_fmadd_ps(av, bv4a, acc1[4]);
                        acc1[5] = _mm256_fmadd_ps(av, bv5a, acc1[5]);
                        acc1[6] = _mm256_fmadd_ps(av, bv6a, acc1[6]);
                        acc1[7] = _mm256_fmadd_ps(av, bv7a, acc1[7]);

                        let av = _mm256_loadu_ps(a2_base.add(ko));
                        acc2[0] = _mm256_fmadd_ps(av, bv0a, acc2[0]);
                        acc2[1] = _mm256_fmadd_ps(av, bv1a, acc2[1]);
                        acc2[2] = _mm256_fmadd_ps(av, bv2a, acc2[2]);
                        acc2[3] = _mm256_fmadd_ps(av, bv3a, acc2[3]);
                        acc2[4] = _mm256_fmadd_ps(av, bv4a, acc2[4]);
                        acc2[5] = _mm256_fmadd_ps(av, bv5a, acc2[5]);
                        acc2[6] = _mm256_fmadd_ps(av, bv6a, acc2[6]);
                        acc2[7] = _mm256_fmadd_ps(av, bv7a, acc2[7]);

                        let av = _mm256_loadu_ps(a3_base.add(ko));
                        acc3[0] = _mm256_fmadd_ps(av, bv0a, acc3[0]);
                        acc3[1] = _mm256_fmadd_ps(av, bv1a, acc3[1]);
                        acc3[2] = _mm256_fmadd_ps(av, bv2a, acc3[2]);
                        acc3[3] = _mm256_fmadd_ps(av, bv3a, acc3[3]);
                        acc3[4] = _mm256_fmadd_ps(av, bv4a, acc3[4]);
                        acc3[5] = _mm256_fmadd_ps(av, bv5a, acc3[5]);
                        acc3[6] = _mm256_fmadd_ps(av, bv6a, acc3[6]);
                        acc3[7] = _mm256_fmadd_ps(av, bv7a, acc3[7]);

                        // --- Second group of 8 k-values (unrolled) ---
                        let ko2 = ko + 8;
                        let bv0b = _mm256_loadu_ps(b0_base.add(ko2));
                        let bv1b = _mm256_loadu_ps(b1_base.add(ko2));
                        let bv2b = _mm256_loadu_ps(b2_base.add(ko2));
                        let bv3b = _mm256_loadu_ps(b3_base.add(ko2));
                        let bv4b = _mm256_loadu_ps(b4_base.add(ko2));
                        let bv5b = _mm256_loadu_ps(b5_base.add(ko2));
                        let bv6b = _mm256_loadu_ps(b6_base.add(ko2));
                        let bv7b = _mm256_loadu_ps(b7_base.add(ko2));

                        let av = _mm256_loadu_ps(a0_base.add(ko2));
                        acc0[0] = _mm256_fmadd_ps(av, bv0b, acc0[0]);
                        acc0[1] = _mm256_fmadd_ps(av, bv1b, acc0[1]);
                        acc0[2] = _mm256_fmadd_ps(av, bv2b, acc0[2]);
                        acc0[3] = _mm256_fmadd_ps(av, bv3b, acc0[3]);
                        acc0[4] = _mm256_fmadd_ps(av, bv4b, acc0[4]);
                        acc0[5] = _mm256_fmadd_ps(av, bv5b, acc0[5]);
                        acc0[6] = _mm256_fmadd_ps(av, bv6b, acc0[6]);
                        acc0[7] = _mm256_fmadd_ps(av, bv7b, acc0[7]);

                        let av = _mm256_loadu_ps(a1_base.add(ko2));
                        acc1[0] = _mm256_fmadd_ps(av, bv0b, acc1[0]);
                        acc1[1] = _mm256_fmadd_ps(av, bv1b, acc1[1]);
                        acc1[2] = _mm256_fmadd_ps(av, bv2b, acc1[2]);
                        acc1[3] = _mm256_fmadd_ps(av, bv3b, acc1[3]);
                        acc1[4] = _mm256_fmadd_ps(av, bv4b, acc1[4]);
                        acc1[5] = _mm256_fmadd_ps(av, bv5b, acc1[5]);
                        acc1[6] = _mm256_fmadd_ps(av, bv6b, acc1[6]);
                        acc1[7] = _mm256_fmadd_ps(av, bv7b, acc1[7]);

                        let av = _mm256_loadu_ps(a2_base.add(ko2));
                        acc2[0] = _mm256_fmadd_ps(av, bv0b, acc2[0]);
                        acc2[1] = _mm256_fmadd_ps(av, bv1b, acc2[1]);
                        acc2[2] = _mm256_fmadd_ps(av, bv2b, acc2[2]);
                        acc2[3] = _mm256_fmadd_ps(av, bv3b, acc2[3]);
                        acc2[4] = _mm256_fmadd_ps(av, bv4b, acc2[4]);
                        acc2[5] = _mm256_fmadd_ps(av, bv5b, acc2[5]);
                        acc2[6] = _mm256_fmadd_ps(av, bv6b, acc2[6]);
                        acc2[7] = _mm256_fmadd_ps(av, bv7b, acc2[7]);

                        let av = _mm256_loadu_ps(a3_base.add(ko2));
                        acc3[0] = _mm256_fmadd_ps(av, bv0b, acc3[0]);
                        acc3[1] = _mm256_fmadd_ps(av, bv1b, acc3[1]);
                        acc3[2] = _mm256_fmadd_ps(av, bv2b, acc3[2]);
                        acc3[3] = _mm256_fmadd_ps(av, bv3b, acc3[3]);
                        acc3[4] = _mm256_fmadd_ps(av, bv4b, acc3[4]);
                        acc3[5] = _mm256_fmadd_ps(av, bv5b, acc3[5]);
                        acc3[6] = _mm256_fmadd_ps(av, bv6b, acc3[6]);
                        acc3[7] = _mm256_fmadd_ps(av, bv7b, acc3[7]);
                    }

                    // Handle remaining 8-element chunk if k_len not divisible by 16
                    let k_vec_rem_start = k_start + k_pairs * 16;
                    if k_vec_rem_start + 8 <= k_end {
                        let ko = k_vec_rem_start;
                        let bv0 = _mm256_loadu_ps(b0_base.add(ko));
                        let bv1 = _mm256_loadu_ps(b1_base.add(ko));
                        let bv2 = _mm256_loadu_ps(b2_base.add(ko));
                        let bv3 = _mm256_loadu_ps(b3_base.add(ko));
                        let bv4 = _mm256_loadu_ps(b4_base.add(ko));
                        let bv5 = _mm256_loadu_ps(b5_base.add(ko));
                        let bv6 = _mm256_loadu_ps(b6_base.add(ko));
                        let bv7 = _mm256_loadu_ps(b7_base.add(ko));

                        let av = _mm256_loadu_ps(a0_base.add(ko));
                        acc0[0] = _mm256_fmadd_ps(av, bv0, acc0[0]);
                        acc0[1] = _mm256_fmadd_ps(av, bv1, acc0[1]);
                        acc0[2] = _mm256_fmadd_ps(av, bv2, acc0[2]);
                        acc0[3] = _mm256_fmadd_ps(av, bv3, acc0[3]);
                        acc0[4] = _mm256_fmadd_ps(av, bv4, acc0[4]);
                        acc0[5] = _mm256_fmadd_ps(av, bv5, acc0[5]);
                        acc0[6] = _mm256_fmadd_ps(av, bv6, acc0[6]);
                        acc0[7] = _mm256_fmadd_ps(av, bv7, acc0[7]);

                        let av = _mm256_loadu_ps(a1_base.add(ko));
                        acc1[0] = _mm256_fmadd_ps(av, bv0, acc1[0]);
                        acc1[1] = _mm256_fmadd_ps(av, bv1, acc1[1]);
                        acc1[2] = _mm256_fmadd_ps(av, bv2, acc1[2]);
                        acc1[3] = _mm256_fmadd_ps(av, bv3, acc1[3]);
                        acc1[4] = _mm256_fmadd_ps(av, bv4, acc1[4]);
                        acc1[5] = _mm256_fmadd_ps(av, bv5, acc1[5]);
                        acc1[6] = _mm256_fmadd_ps(av, bv6, acc1[6]);
                        acc1[7] = _mm256_fmadd_ps(av, bv7, acc1[7]);

                        let av = _mm256_loadu_ps(a2_base.add(ko));
                        acc2[0] = _mm256_fmadd_ps(av, bv0, acc2[0]);
                        acc2[1] = _mm256_fmadd_ps(av, bv1, acc2[1]);
                        acc2[2] = _mm256_fmadd_ps(av, bv2, acc2[2]);
                        acc2[3] = _mm256_fmadd_ps(av, bv3, acc2[3]);
                        acc2[4] = _mm256_fmadd_ps(av, bv4, acc2[4]);
                        acc2[5] = _mm256_fmadd_ps(av, bv5, acc2[5]);
                        acc2[6] = _mm256_fmadd_ps(av, bv6, acc2[6]);
                        acc2[7] = _mm256_fmadd_ps(av, bv7, acc2[7]);

                        let av = _mm256_loadu_ps(a3_base.add(ko));
                        acc3[0] = _mm256_fmadd_ps(av, bv0, acc3[0]);
                        acc3[1] = _mm256_fmadd_ps(av, bv1, acc3[1]);
                        acc3[2] = _mm256_fmadd_ps(av, bv2, acc3[2]);
                        acc3[3] = _mm256_fmadd_ps(av, bv3, acc3[3]);
                        acc3[4] = _mm256_fmadd_ps(av, bv4, acc3[4]);
                        acc3[5] = _mm256_fmadd_ps(av, bv5, acc3[5]);
                        acc3[6] = _mm256_fmadd_ps(av, bv6, acc3[6]);
                        acc3[7] = _mm256_fmadd_ps(av, bv7, acc3[7]);
                    }

                    // Horizontal reduction for AVX2: sum 8 lanes to scalar
                    let c0 = c_ptr.add(i_start * n + j_start);
                    let c1 = c_ptr.add((i_start + 1) * n + j_start);
                    let c2 = c_ptr.add((i_start + 2) * n + j_start);
                    let c3 = c_ptr.add((i_start + 3) * n + j_start);

                    for jj in 0..TILE_J {
                        *c0.add(jj) += hsum_m256(acc0[jj]);
                        *c1.add(jj) += hsum_m256(acc1[jj]);
                        *c2.add(jj) += hsum_m256(acc2[jj]);
                        *c3.add(jj) += hsum_m256(acc3[jj]);
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
