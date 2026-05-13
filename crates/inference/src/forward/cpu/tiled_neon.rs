// ---------------------------------------------------------------------------
// NEON tiled matmul_bt — cache-blocked with 4×8 NEON microkernel
// ---------------------------------------------------------------------------
//
// Microkernel design (4i × 8j):
//   - 32 float32x4_t accumulators (4 rows × 8 cols)
//   - Inner k-loop unrolled 2× (processes 8 k-values per iteration)
//   - B vectors loaded once per k-chunk, reused across all 4 i-rows
//   - Register budget: 8 B-vecs + 1 A-vec + 8 acc (current i) = 17 live regs
//     (remaining 15 hold acc for other i-rows or are spill slots)
//

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
use super::tiled::{TILE_I, TILE_J, TILE_K};

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
#[target_feature(enable = "neon")]
pub(super) unsafe fn matmul_bt_tiled_neon(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use std::arch::aarch64::*;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    // Loop order: J-tiles → I-tiles → K-tiles
    // B tiles stay hot in cache across all I iterations.
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

                // Fast path: full 4×8 tile with k_len >= 8 (enables 2× unrolled inner loop)
                if i_count == TILE_I && j_count == TILE_J && k_len >= 8 {
                    // SAFETY: We have verified i_count == 4 and j_count == 8.
                    // All pointer arithmetic stays within bounds of a (m*k), b (n*k), c (m*n):
                    //   - i ranges from i_start to i_start+3, each < m
                    //   - j ranges from j_start to j_start+7, each < n
                    //   - k offsets range from k_start to k_end-1, each < k
                    //   - NEON loads read 4 contiguous f32s; offset+3 < k for all loads

                    // 4×8 accumulator grid: each acc[ii][jj] holds partial dot products
                    // across 4 lanes of the K-dimension. After the k-loop, each is
                    // horizontally summed to produce the scalar C[i][j] contribution.
                    let mut acc0 = [vdupq_n_f32(0.0); TILE_J];
                    let mut acc1 = [vdupq_n_f32(0.0); TILE_J];
                    let mut acc2 = [vdupq_n_f32(0.0); TILE_J];
                    let mut acc3 = [vdupq_n_f32(0.0); TILE_J];

                    // Precompute row base pointers for the 4 A-rows and 8 B-rows.
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

                    // Main loop: process 8 k-values per iteration (2× unrolled).
                    // Each iteration does 2 rounds of 4-wide NEON FMA.
                    let k_pairs = k_len / 8;
                    for kp in 0..k_pairs {
                        let ko = k_start + kp * 8;

                        // --- First group of 4 k-values ---
                        let bv0a = vld1q_f32(b0_base.add(ko));
                        let bv1a = vld1q_f32(b1_base.add(ko));
                        let bv2a = vld1q_f32(b2_base.add(ko));
                        let bv3a = vld1q_f32(b3_base.add(ko));
                        let bv4a = vld1q_f32(b4_base.add(ko));
                        let bv5a = vld1q_f32(b5_base.add(ko));
                        let bv6a = vld1q_f32(b6_base.add(ko));
                        let bv7a = vld1q_f32(b7_base.add(ko));

                        let av = vld1q_f32(a0_base.add(ko));
                        acc0[0] = vfmaq_f32(acc0[0], av, bv0a);
                        acc0[1] = vfmaq_f32(acc0[1], av, bv1a);
                        acc0[2] = vfmaq_f32(acc0[2], av, bv2a);
                        acc0[3] = vfmaq_f32(acc0[3], av, bv3a);
                        acc0[4] = vfmaq_f32(acc0[4], av, bv4a);
                        acc0[5] = vfmaq_f32(acc0[5], av, bv5a);
                        acc0[6] = vfmaq_f32(acc0[6], av, bv6a);
                        acc0[7] = vfmaq_f32(acc0[7], av, bv7a);

                        let av = vld1q_f32(a1_base.add(ko));
                        acc1[0] = vfmaq_f32(acc1[0], av, bv0a);
                        acc1[1] = vfmaq_f32(acc1[1], av, bv1a);
                        acc1[2] = vfmaq_f32(acc1[2], av, bv2a);
                        acc1[3] = vfmaq_f32(acc1[3], av, bv3a);
                        acc1[4] = vfmaq_f32(acc1[4], av, bv4a);
                        acc1[5] = vfmaq_f32(acc1[5], av, bv5a);
                        acc1[6] = vfmaq_f32(acc1[6], av, bv6a);
                        acc1[7] = vfmaq_f32(acc1[7], av, bv7a);

                        let av = vld1q_f32(a2_base.add(ko));
                        acc2[0] = vfmaq_f32(acc2[0], av, bv0a);
                        acc2[1] = vfmaq_f32(acc2[1], av, bv1a);
                        acc2[2] = vfmaq_f32(acc2[2], av, bv2a);
                        acc2[3] = vfmaq_f32(acc2[3], av, bv3a);
                        acc2[4] = vfmaq_f32(acc2[4], av, bv4a);
                        acc2[5] = vfmaq_f32(acc2[5], av, bv5a);
                        acc2[6] = vfmaq_f32(acc2[6], av, bv6a);
                        acc2[7] = vfmaq_f32(acc2[7], av, bv7a);

                        let av = vld1q_f32(a3_base.add(ko));
                        acc3[0] = vfmaq_f32(acc3[0], av, bv0a);
                        acc3[1] = vfmaq_f32(acc3[1], av, bv1a);
                        acc3[2] = vfmaq_f32(acc3[2], av, bv2a);
                        acc3[3] = vfmaq_f32(acc3[3], av, bv3a);
                        acc3[4] = vfmaq_f32(acc3[4], av, bv4a);
                        acc3[5] = vfmaq_f32(acc3[5], av, bv5a);
                        acc3[6] = vfmaq_f32(acc3[6], av, bv6a);
                        acc3[7] = vfmaq_f32(acc3[7], av, bv7a);

                        // --- Second group of 4 k-values (unrolled) ---
                        let ko2 = ko + 4;
                        let bv0b = vld1q_f32(b0_base.add(ko2));
                        let bv1b = vld1q_f32(b1_base.add(ko2));
                        let bv2b = vld1q_f32(b2_base.add(ko2));
                        let bv3b = vld1q_f32(b3_base.add(ko2));
                        let bv4b = vld1q_f32(b4_base.add(ko2));
                        let bv5b = vld1q_f32(b5_base.add(ko2));
                        let bv6b = vld1q_f32(b6_base.add(ko2));
                        let bv7b = vld1q_f32(b7_base.add(ko2));

                        let av = vld1q_f32(a0_base.add(ko2));
                        acc0[0] = vfmaq_f32(acc0[0], av, bv0b);
                        acc0[1] = vfmaq_f32(acc0[1], av, bv1b);
                        acc0[2] = vfmaq_f32(acc0[2], av, bv2b);
                        acc0[3] = vfmaq_f32(acc0[3], av, bv3b);
                        acc0[4] = vfmaq_f32(acc0[4], av, bv4b);
                        acc0[5] = vfmaq_f32(acc0[5], av, bv5b);
                        acc0[6] = vfmaq_f32(acc0[6], av, bv6b);
                        acc0[7] = vfmaq_f32(acc0[7], av, bv7b);

                        let av = vld1q_f32(a1_base.add(ko2));
                        acc1[0] = vfmaq_f32(acc1[0], av, bv0b);
                        acc1[1] = vfmaq_f32(acc1[1], av, bv1b);
                        acc1[2] = vfmaq_f32(acc1[2], av, bv2b);
                        acc1[3] = vfmaq_f32(acc1[3], av, bv3b);
                        acc1[4] = vfmaq_f32(acc1[4], av, bv4b);
                        acc1[5] = vfmaq_f32(acc1[5], av, bv5b);
                        acc1[6] = vfmaq_f32(acc1[6], av, bv6b);
                        acc1[7] = vfmaq_f32(acc1[7], av, bv7b);

                        let av = vld1q_f32(a2_base.add(ko2));
                        acc2[0] = vfmaq_f32(acc2[0], av, bv0b);
                        acc2[1] = vfmaq_f32(acc2[1], av, bv1b);
                        acc2[2] = vfmaq_f32(acc2[2], av, bv2b);
                        acc2[3] = vfmaq_f32(acc2[3], av, bv3b);
                        acc2[4] = vfmaq_f32(acc2[4], av, bv4b);
                        acc2[5] = vfmaq_f32(acc2[5], av, bv5b);
                        acc2[6] = vfmaq_f32(acc2[6], av, bv6b);
                        acc2[7] = vfmaq_f32(acc2[7], av, bv7b);

                        let av = vld1q_f32(a3_base.add(ko2));
                        acc3[0] = vfmaq_f32(acc3[0], av, bv0b);
                        acc3[1] = vfmaq_f32(acc3[1], av, bv1b);
                        acc3[2] = vfmaq_f32(acc3[2], av, bv2b);
                        acc3[3] = vfmaq_f32(acc3[3], av, bv3b);
                        acc3[4] = vfmaq_f32(acc3[4], av, bv4b);
                        acc3[5] = vfmaq_f32(acc3[5], av, bv5b);
                        acc3[6] = vfmaq_f32(acc3[6], av, bv6b);
                        acc3[7] = vfmaq_f32(acc3[7], av, bv7b);
                    }

                    // Handle remaining 4-element chunk if k_len is not divisible by 8
                    // but is divisible by 4 (since k_len >= 8, we know k_pairs >= 1).
                    let k_vec_rem_start = k_start + k_pairs * 8;
                    if k_vec_rem_start + 4 <= k_end {
                        let ko = k_vec_rem_start;
                        let bv0 = vld1q_f32(b0_base.add(ko));
                        let bv1 = vld1q_f32(b1_base.add(ko));
                        let bv2 = vld1q_f32(b2_base.add(ko));
                        let bv3 = vld1q_f32(b3_base.add(ko));
                        let bv4 = vld1q_f32(b4_base.add(ko));
                        let bv5 = vld1q_f32(b5_base.add(ko));
                        let bv6 = vld1q_f32(b6_base.add(ko));
                        let bv7 = vld1q_f32(b7_base.add(ko));

                        let av = vld1q_f32(a0_base.add(ko));
                        acc0[0] = vfmaq_f32(acc0[0], av, bv0);
                        acc0[1] = vfmaq_f32(acc0[1], av, bv1);
                        acc0[2] = vfmaq_f32(acc0[2], av, bv2);
                        acc0[3] = vfmaq_f32(acc0[3], av, bv3);
                        acc0[4] = vfmaq_f32(acc0[4], av, bv4);
                        acc0[5] = vfmaq_f32(acc0[5], av, bv5);
                        acc0[6] = vfmaq_f32(acc0[6], av, bv6);
                        acc0[7] = vfmaq_f32(acc0[7], av, bv7);

                        let av = vld1q_f32(a1_base.add(ko));
                        acc1[0] = vfmaq_f32(acc1[0], av, bv0);
                        acc1[1] = vfmaq_f32(acc1[1], av, bv1);
                        acc1[2] = vfmaq_f32(acc1[2], av, bv2);
                        acc1[3] = vfmaq_f32(acc1[3], av, bv3);
                        acc1[4] = vfmaq_f32(acc1[4], av, bv4);
                        acc1[5] = vfmaq_f32(acc1[5], av, bv5);
                        acc1[6] = vfmaq_f32(acc1[6], av, bv6);
                        acc1[7] = vfmaq_f32(acc1[7], av, bv7);

                        let av = vld1q_f32(a2_base.add(ko));
                        acc2[0] = vfmaq_f32(acc2[0], av, bv0);
                        acc2[1] = vfmaq_f32(acc2[1], av, bv1);
                        acc2[2] = vfmaq_f32(acc2[2], av, bv2);
                        acc2[3] = vfmaq_f32(acc2[3], av, bv3);
                        acc2[4] = vfmaq_f32(acc2[4], av, bv4);
                        acc2[5] = vfmaq_f32(acc2[5], av, bv5);
                        acc2[6] = vfmaq_f32(acc2[6], av, bv6);
                        acc2[7] = vfmaq_f32(acc2[7], av, bv7);

                        let av = vld1q_f32(a3_base.add(ko));
                        acc3[0] = vfmaq_f32(acc3[0], av, bv0);
                        acc3[1] = vfmaq_f32(acc3[1], av, bv1);
                        acc3[2] = vfmaq_f32(acc3[2], av, bv2);
                        acc3[3] = vfmaq_f32(acc3[3], av, bv3);
                        acc3[4] = vfmaq_f32(acc3[4], av, bv4);
                        acc3[5] = vfmaq_f32(acc3[5], av, bv5);
                        acc3[6] = vfmaq_f32(acc3[6], av, bv6);
                        acc3[7] = vfmaq_f32(acc3[7], av, bv7);
                    }

                    // Horizontal reduction: sum each float32x4 accumulator to a scalar
                    // and add to C[i][j]. vaddvq_f32 sums all 4 lanes.
                    let c0 = c_ptr.add(i_start * n + j_start);
                    let c1 = c_ptr.add((i_start + 1) * n + j_start);
                    let c2 = c_ptr.add((i_start + 2) * n + j_start);
                    let c3 = c_ptr.add((i_start + 3) * n + j_start);

                    for jj in 0..TILE_J {
                        *c0.add(jj) += vaddvq_f32(acc0[jj]);
                        *c1.add(jj) += vaddvq_f32(acc1[jj]);
                        *c2.add(jj) += vaddvq_f32(acc2[jj]);
                        *c3.add(jj) += vaddvq_f32(acc3[jj]);
                    }

                    // Scalar remainder for k-values not covered by NEON loads
                    let k_scalar_start = k_start + (k_len / 4) * 4;
                    if k_scalar_start < k_end {
                        for ii in 0..TILE_I {
                            let i = i_start + ii;
                            for jj in 0..TILE_J {
                                let j = j_start + jj;
                                let mut sum = 0.0f32;
                                for p in k_scalar_start..k_end {
                                    // SAFETY: i < m, p < k, j < n — within bounds.
                                    sum += *a_ptr.add(i * k + p) * *b_ptr.add(j * k + p);
                                }
                                *c_ptr.add(i * n + j) += sum;
                            }
                        }
                    }
                } else {
                    // Edge tiles: partial I or J count, or k_len < 8.
                    // Use scalar accumulation for correctness at tile boundaries.
                    for ii in 0..i_count {
                        let i = i_start + ii;
                        for jj in 0..j_count {
                            let j = j_start + jj;
                            let mut sum = 0.0f32;
                            for p in k_start..k_end {
                                // SAFETY: i < m, p < k so i*k+p < m*k = a.len().
                                // j < n, p < k so j*k+p < n*k = b.len().
                                sum += *a_ptr.add(i * k + p) * *b_ptr.add(j * k + p);
                            }
                            // SAFETY: i < m, j < n so i*n+j < m*n = c.len().
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
