//! General matmul helpers, preallocated matmul, transposed-B matmul, scalar fallbacks, and m=1 specialization.
use super::gemm_validate::{validate_gemm_bt, validate_gemm_nn};
#[cfg(not(target_os = "macos"))]
use super::simd::simd_config;

#[cfg(all(not(target_os = "macos"), target_arch = "aarch64"))]
use super::arch_kernels::matmul_neon;
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
use super::arch_kernels::{matmul_avx2, matmul_avx512};
#[cfg(target_os = "macos")]
use super::blas::{accelerate_matmul, accelerate_matmul_bt};
#[cfg(not(target_os = "macos"))]
use super::tiled::matmul_bt_tiled;

/// **Unstable**: general matmul C = A*B; dispatches to platform BLAS or SIMD fallback.
///
/// General matrix multiplication: C = A * B.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // Guard the allocation itself: an overflowed m*n would silently allocate a tiny Vec
    // whose length would then pass matmul_into's c.len() >= m*n check (same wrapped value).
    // Checked here so the vec! and all downstream guards use the same trustworthy product.
    assert!(
        m.checked_mul(n).is_some(),
        "matmul output shape overflow: m*n"
    );
    let mut c = vec![0.0f32; m * n];
    matmul_into(a, b, &mut c, m, k, n);
    c
}

/// **Unstable**: matmul into pre-allocated buffer; dispatch logic may change.
///
/// Matrix multiply into a pre-allocated output buffer.
pub fn matmul_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Release-active, overflow-first, oversized-scratch-allowed contract (#368, ADR-080 C4) —
    // see `gemm_validate` for the shared rationale. Some callers pass reused scratch buffers
    // longer than the exact footprint; that is sound (the check is `>=`). Note the output
    // suffix beyond m*n is NOT part of the result and may be clobbered (matmul_scalar zeroes
    // the full c slice) — callers needing suffix preservation must pass &mut c[..m*n].
    validate_gemm_nn(a.len(), b.len(), c.len(), m, k, n, "matmul");

    // GPU dispatch is NOT in this hot path — per-call buffer creation is too slow.
    // GPU acceleration requires the full forward pass to run on-device.
    // See gpu_gemm.rs for standalone GPU GEMM (used in benchmarks).

    #[cfg(target_os = "macos")]
    {
        accelerate_matmul(a, b, c, m, n, k);
    }

    #[cfg(not(target_os = "macos"))]
    matmul_scalar(a, b, c, m, k, n);
}

/// **Unstable**: matmul C = A @ B^T; primary inference kernel, dispatch strategy evolving.
///
/// Matrix multiply with transposed B: C = A @ B^T.
pub fn matmul_bt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Release-active, overflow-first, oversized-scratch-allowed contract (#368, ADR-080 C4).
    // Note: B is stored transposed, so its footprint is n*k, not k*n. Some callers pass
    // reused scratch buffers longer than the exact footprint; that is sound (the check is
    // `>=`). The output suffix beyond m*n is NOT part of the result and is unspecified by
    // contract: on x86_64 the dispatch below writes only c[..m*n], while on the other
    // non-macOS targets matmul_bt_tiled zeroes the full c slice it is given, so callers
    // needing suffix preservation there must pass &mut c[..m*n].
    validate_gemm_bt(a.len(), b.len(), c.len(), m, k, n, "matmul_bt");

    // CPU path only — Accelerate AMX on macOS, SIMD/scalar elsewhere.
    #[cfg(target_os = "macos")]
    {
        accelerate_matmul_bt(a, b, c, m, n, k);
    }

    // x86_64 fallback: hand-written SIMD with tiling for large matrices.
    #[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
    {
        // Use cache-blocked (tiled) path for large matrices where blocking pays off.
        // Two conditions must be met:
        //   1. Total work m*n*k >= 1024*1024 (below this, overhead dominates).
        //   2. K >= 128 (the shared dimension must be large enough that B-rows don't fit
        //      in L1 cache naturally). When K is small (e.g. 32), each B-row is only
        //      128 bytes and fits in L1 without tiling. Tiling would only change the
        //      accumulation order and introduce unnecessary numerical differences.
        // The tiled AVX2 kernel computes a full TILE_I x TILE_J tile with SIMD only when
        // its K-tile has at least 16 elements; every other tile, including every partial
        // TILE_I row tile, runs a scalar loop (the whole tiled call is scalar when
        // AVX2+FMA is not detected). So the tiled path takes only the largest
        // multiple of TILE_I rows, and the remaining rows (all of them when m < TILE_I,
        // as in every decode step) go through the direct kernels below: AVX-512F, then
        // AVX2, then the scalar reference when no SIMD feature is detected. Partial
        // TILE_J column tiles and short K tiles inside the tiled part are still scalar.
        // The tiled call gets exactly c[..full * n] because it zeroes the whole slice it
        // is given.
        let total_work = (m as u64) * (n as u64) * (k as u64);
        let full = if total_work >= 1024 * 1024 && k >= super::tiled::TILE_K {
            m - m % super::tiled::TILE_I
        } else {
            0
        };
        if full > 0 {
            matmul_bt_tiled(&a[..full * k], b, &mut c[..full * n], full, k, n);
        }
        if full < m {
            matmul_bt_direct(
                &a[full * k..m * k],
                b,
                &mut c[full * n..m * n],
                m - full,
                k,
                n,
            );
        }
    }

    // Other non-macOS targets: hand-written SIMD with tiling for large matrices.
    #[cfg(all(not(target_os = "macos"), not(target_arch = "x86_64")))]
    {
        // Use cache-blocked (tiled) path for large matrices where blocking pays off.
        // Two conditions must be met:
        //   1. Total work m*n*k >= 1024*1024 (below this, overhead dominates).
        //   2. K >= 128 (the shared dimension must be large enough that B-rows don't fit
        //      in L1 cache naturally). When K is small (e.g. 32), each B-row is only
        //      128 bytes and fits in L1 without tiling. Tiling would only change the
        //      accumulation order and introduce unnecessary numerical differences.
        // All m rows go to the tiled kernel without a row split: on aarch64 its NEON edge
        // branch vectorises partial TILE_I row tiles along K, so small m is not forced
        // scalar there.
        let total_work = (m as u64) * (n as u64) * (k as u64);
        if total_work >= 1024 * 1024 && k >= super::tiled::TILE_K {
            matmul_bt_tiled(a, b, c, m, k, n);
            return;
        }

        let config = simd_config();

        #[cfg(target_arch = "aarch64")]
        {
            if config.neon_enabled {
                // SAFETY: NEON is available on aarch64 and the runtime gate ensures this path.
                unsafe {
                    matmul_neon(a, b, c, m, k, n);
                    return;
                }
            }
        }

        matmul_bt_scalar(a, b, c, m, k, n);
    }
}

/// Direct (untiled) transposed-B matmul on x86_64: the widest SIMD kernel the CPU
/// supports, else the scalar reference. Each output element depends only on its own
/// A row and B row.
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
fn matmul_bt_direct(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let config = simd_config();

    if config.avx512f_enabled && config.fma_enabled {
        // SAFETY: The runtime feature checks above guarantee AVX-512F+FMA support.
        unsafe {
            matmul_avx512(a, b, c, m, k, n);
            return;
        }
    }
    if config.avx2_enabled && config.fma_enabled {
        // SAFETY: The runtime feature checks above guarantee AVX2+FMA support.
        unsafe {
            matmul_avx2(a, b, c, m, k, n);
            return;
        }
    }

    matmul_bt_scalar(a, b, c, m, k, n);
}

/// **Unstable**: scalar matmul reference; used for non-SIMD targets and testing.
///
/// Scalar reference implementation of A * B.
pub fn matmul_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    c.fill(0.0);

    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            let b_row = &b[p * n..(p + 1) * n];
            let c_row = &mut c[i * n..(i + 1) * n];
            for j in 0..n {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

#[cfg_attr(target_os = "macos", allow(dead_code))]
pub fn matmul_bt_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 1 {
        matmul_bt_scalar_m1(a, b, c, k, n);
        return;
    }
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        for j in 0..n {
            let b_row = &b[j * k..(j + 1) * k];
            let mut s0 = 0.0f32;
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            let mut s3 = 0.0f32;
            let unrolled = k / 4;
            for p in 0..unrolled {
                let off = p * 4;
                s0 += a_row[off] * b_row[off];
                s1 += a_row[off + 1] * b_row[off + 1];
                s2 += a_row[off + 2] * b_row[off + 2];
                s3 += a_row[off + 3] * b_row[off + 3];
            }
            for p in (unrolled * 4)..k {
                s0 += a_row[p] * b_row[p];
            }
            c_row[j] = (s0 + s1) + (s2 + s3);
        }
    }
}

#[cfg_attr(target_os = "macos", allow(dead_code))]
#[inline]
fn matmul_bt_scalar_m1(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
    let a_row = &a[..k];
    let unrolled8 = k / 8;
    for j in 0..n {
        let b_row = &b[j * k..(j + 1) * k];
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let mut s4 = 0.0f32;
        let mut s5 = 0.0f32;
        let mut s6 = 0.0f32;
        let mut s7 = 0.0f32;
        for p in 0..unrolled8 {
            let off = p * 8;
            s0 += a_row[off] * b_row[off];
            s1 += a_row[off + 1] * b_row[off + 1];
            s2 += a_row[off + 2] * b_row[off + 2];
            s3 += a_row[off + 3] * b_row[off + 3];
            s4 += a_row[off + 4] * b_row[off + 4];
            s5 += a_row[off + 5] * b_row[off + 5];
            s6 += a_row[off + 6] * b_row[off + 6];
            s7 += a_row[off + 7] * b_row[off + 7];
        }
        for p in (unrolled8 * 8)..k {
            s0 += a_row[p] * b_row[p];
        }
        c[j] = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- release-active bounds guards (#368) ---

    /// A too-short `b` must panic in both debug AND release builds (release-active assert).
    /// m=1, k=2, n=2: b must have n*k=4 elements; passing [] triggers the guard.
    #[test]
    #[should_panic(expected = "too short for n*k")]
    fn matmul_bt_short_b_panics_in_release() {
        let a = [0.0f32; 2]; // a.len() = m*k = 1*2 = 2 ✓
        let b: [f32; 0] = []; // b.len() = 0 < n*k = 4  ✗
        let mut c = [0.0f32; 2];
        matmul_bt(&a, &b, &mut c, 1, 2, 2);
    }

    /// A shape product that would overflow usize must panic before any memory access.
    /// m=2, k=usize::MAX, n=2: m*k overflows on the first overflow check.
    #[test]
    #[should_panic(expected = "shape overflow")]
    fn matmul_shape_overflow_panics() {
        let a = [0.0f32; 2];
        let b = [0.0f32; 2];
        let mut c = [0.0f32; 1];
        matmul_bt(&a, &b, &mut c, 2, usize::MAX, 2);
    }

    /// Callers may pass reused scratch buffers whose length EXCEEDS the exact footprint.
    /// `>=` is the correct check: this must NOT panic and must produce correct results
    /// in c[0..m*n]. The content of extra slots is unspecified across platforms.
    #[test]
    fn matmul_bt_oversized_c_does_not_panic() {
        // m=1, k=2, n=2: result c is m*n=2 elements. Pass c of length 3 (one extra).
        // a = [1, 2], b = [[1, 0], [0, 1]] stored row-major (transposed B).
        // c[0] = dot(a, b_row0) = 1*1 + 2*0 = 1
        // c[1] = dot(a, b_row1) = 1*0 + 2*1 = 2
        let a = [1.0f32, 2.0];
        let b = [1.0f32, 0.0, 0.0, 1.0]; // n=2 rows of k=2
        let mut c = [0.0f32; 3]; // intentionally oversized (3 > m*n=2)
        matmul_bt(&a, &b, &mut c, 1, 2, 2);
        assert!(
            (c[0] - 1.0).abs() < 1e-6,
            "c[0] should be 1.0, got {}",
            c[0]
        );
        assert!(
            (c[1] - 2.0).abs() < 1e-6,
            "c[1] should be 2.0, got {}",
            c[1]
        );
        // Extra slot c[2]: content is platform-defined, we only guarantee no panic and
        // correct values in c[0..m*n]. This test proves the >= bound is correct.
    }

    // A zero dimension is an empty GEMM, not a shape error: Accelerate aborts the process on a
    // zero leading dimension, so each case runs through the public entry point on every backend.
    // The result region is c[..m*n]; the suffix beyond it is not part of the contract (see the
    // `matmul_bt` doc), so these tests assert nothing about it. The sentinel only proves that a
    // nonempty result region is overwritten.
    const SENTINEL: f32 = 7.5;

    #[test]
    fn matmul_bt_zero_rows_returns() {
        let (m, k, n) = (0usize, 3usize, 2usize);
        let b = [1.0f32; 6];
        let mut c = [SENTINEL; 4];
        matmul_bt(&[], &b, &mut c, m, k, n);
        // m*n == 0: the result region is empty, so the property is that the call returns.
    }

    #[test]
    fn matmul_bt_zero_columns_returns() {
        let (m, k, n) = (2usize, 3usize, 0usize);
        let a = [1.0f32; 6];
        let mut c = [SENTINEL; 4];
        matmul_bt(&a, &[], &mut c, m, k, n);
        // m*n == 0: the result region is empty, so the property is that the call returns.
    }

    #[test]
    fn matmul_bt_zero_inner_dimension_zero_fills_output() {
        let (m, k, n) = (3usize, 0usize, 2usize);
        let mut c = [SENTINEL; 9];
        matmul_bt(&[], &[], &mut c, m, k, n);
        assert!(
            c[..m * n].iter().all(|&v| v == 0.0),
            "k=0 must zero c[..m*n], got {:?}",
            &c[..m * n]
        );
    }

    #[test]
    fn matmul_bt_zero_inner_dimension_single_row_zero_fills_output() {
        let mut c = [SENTINEL; 5];
        matmul_bt(&[], &[], &mut c, 1, 0, 4);
        assert!(c[..4].iter().all(|&v| v == 0.0));
    }

    #[cfg(not(target_os = "macos"))]
    fn lcg_vec(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1u32 << 24) as f32) * 0.04 - 0.02
            })
            .collect()
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn every_small_m_matches_scalar_reference() {
        let (k, n) = (256usize, 1024usize);
        let b = lcg_vec(n * k, 0x0E55);
        for m in 1..=9usize {
            let a = lcg_vec(m * k, 0x0F66 + m as u32);
            let mut got = vec![0.0f32; m * n];
            let mut want = vec![0.0f32; m * n];
            matmul_bt(&a, &b, &mut got, m, k, n);
            matmul_bt_scalar(&a, &b, &mut want, m, k, n);
            for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() < 1e-4, "m={m} idx={idx}: got {g}, want {w}");
            }
        }
    }

    // --- x86_64: small-m rows must not run through the tiled kernel's scalar edge loop ---

    #[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
    mod small_m_dispatch {
        use super::*;

        fn bits(v: &[f32]) -> Vec<u32> {
            v.iter().map(|x| x.to_bits()).collect()
        }

        /// m=1 with n*k >= 1M: the whole call must equal column chunks that are each below
        /// the tiled threshold (direct kernel). The direct kernels compute every output
        /// column from its own A row and B row only, so the two are bit-identical.
        #[test]
        fn m1_large_matches_direct_kernel_per_column_chunk() {
            let (m, k, n) = (1usize, 1024usize, 2048usize);
            let chunk = 512usize;
            assert!((m * k * chunk) < 1024 * 1024 && (m * k * n) >= 1024 * 1024);
            let a = lcg_vec(m * k, 0x0A11);
            let b = lcg_vec(n * k, 0x0B22);

            let mut whole = vec![0.0f32; m * n];
            matmul_bt(&a, &b, &mut whole, m, k, n);

            let mut chunked = vec![0.0f32; m * n];
            for j0 in (0..n).step_by(chunk) {
                matmul_bt(
                    &a,
                    &b[j0 * k..(j0 + chunk) * k],
                    &mut chunked[j0..j0 + chunk],
                    m,
                    k,
                    chunk,
                );
            }
            assert_eq!(bits(&whole), bits(&chunked));
        }

        /// m=5 with work >= 1M: rows 0..4 take the tiled kernel exactly as a 4-row call
        /// does, and row 4 takes the direct kernel exactly as a 1-row call does.
        #[test]
        fn m5_splits_into_tiled_rows_and_direct_row() {
            let (m, k, n) = (5usize, 1024usize, 512usize);
            let a = lcg_vec(m * k, 0x0C33);
            let b = lcg_vec(n * k, 0x0D44);

            let mut whole = vec![0.0f32; m * n];
            matmul_bt(&a, &b, &mut whole, m, k, n);

            let mut head = vec![0.0f32; 4 * n];
            matmul_bt(&a[..4 * k], &b, &mut head, 4, k, n);
            assert_eq!(bits(&whole[..4 * n]), bits(&head));

            let mut tail = vec![0.0f32; n];
            matmul_bt(&a[4 * k..], &b, &mut tail, 1, k, n);
            assert_eq!(bits(&whole[4 * n..]), bits(&tail));
        }

        /// A `c` longer than m*n keeps its suffix untouched, for m < TILE_I and for an m
        /// with a remainder, at a size where the tiled path is selected.
        #[test]
        fn oversized_c_suffix_is_untouched() {
            let (k, n) = (1024usize, 1024usize);
            let b = lcg_vec(n * k, 0x1077);
            let sentinel = f32::from_bits(0x7FC0_BEEF);
            for m in [1usize, 3, 4, 5, 7] {
                let a = lcg_vec(m * k, 0x1188 + m as u32);
                let extra = 16usize;
                let mut c = vec![sentinel; m * n + extra];
                matmul_bt(&a, &b, &mut c, m, k, n);
                assert!(
                    c[m * n..].iter().all(|x| x.to_bits() == sentinel.to_bits()),
                    "m={m}: suffix beyond m*n was written"
                );
                assert!(
                    c[..m * n].iter().all(|x| x.is_finite()),
                    "m={m}: result region was not fully written"
                );
            }
        }
    }
}
