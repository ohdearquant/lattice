//! General matmul helpers, preallocated matmul, transposed-B matmul, scalar fallbacks, and m=1 specialization.
#[cfg(not(target_os = "macos"))]
use super::simd::simd_config;

#[cfg(all(not(target_os = "macos"), target_arch = "aarch64"))]
use super::arch_kernels::matmul_neon;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::arch_kernels::matmul_neon;
#[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
use super::arch_kernels::{matmul_avx2, matmul_avx512};
#[cfg(target_os = "macos")]
use super::blas::{accelerate_matmul, accelerate_matmul_bt};
#[cfg(not(target_os = "macos"))]
use super::tiled::matmul_bt_tiled;

/// Total multiply-add elements (`m*n*k`) below which a GEMM is "small": too little
/// work for a platform GEMM backend's fixed per-call dispatch cost (Apple Accelerate's
/// AMX setup on macOS, this crate's own cache-blocking decision elsewhere) to pay for
/// itself against a direct unrolled dot-product loop. Shared by `matmul_bt`'s own
/// tiled-vs-SIMD choice below and by [`matmul_bt_attention`]'s macOS small-GEMM bypass.
const SMALL_GEMM_ELEMENTS: u64 = 1024 * 1024;

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
    // Release-active bounds guards — see #368 (#218/#224 precedent).
    // Overflow checks first: a wrapped shape product can make a length check pass spuriously
    // (the #367 lesson). The unchecked multiplies below are safe only after these pass.
    assert!(m.checked_mul(k).is_some(), "matmul shape overflow: m*k");
    assert!(k.checked_mul(n).is_some(), "matmul shape overflow: k*n");
    assert!(m.checked_mul(n).is_some(), "matmul shape overflow: m*n");
    // Lower-bound (>=) is the memory-safety invariant: inputs are read only within their
    // shape footprint (a: [0, m*k), b: [0, k*n)) and the output requires c.len() >= m*n.
    // Some callers pass reused scratch buffers longer than the exact footprint; that is
    // sound, and assert_eq! would false-panic on them. Note the output suffix beyond m*n
    // is NOT part of the result and may be clobbered (matmul_scalar zeroes the full c
    // slice) — callers needing suffix preservation must pass &mut c[..m*n].
    assert!(a.len() >= m * k, "matmul: a too short for m*k");
    assert!(b.len() >= k * n, "matmul: b too short for k*n");
    assert!(c.len() >= m * n, "matmul: c too short for m*n");

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
    // Release-active bounds guards — see #368 (#218/#224 precedent).
    // Overflow checks first: a wrapped shape product can make a length check pass spuriously
    // (the #367 lesson). Note: B is stored transposed, so its footprint is n*k, not k*n.
    assert!(m.checked_mul(k).is_some(), "matmul shape overflow: m*k");
    assert!(n.checked_mul(k).is_some(), "matmul shape overflow: n*k");
    assert!(m.checked_mul(n).is_some(), "matmul shape overflow: m*n");
    // Lower-bound (>=) is the memory-safety invariant: inputs are read only within their
    // shape footprint (a: [0, m*k), b: [0, n*k) since B is transposed) and the output
    // requires c.len() >= m*n. Some callers pass reused scratch buffers longer than the
    // exact footprint; that is sound, and assert_eq! would false-panic on them. Note the
    // output suffix beyond m*n is NOT part of the result and may be clobbered
    // (matmul_bt_tiled zeroes the full c slice) — callers needing suffix preservation
    // must pass &mut c[..m*n].
    assert!(a.len() >= m * k, "matmul: a too short for m*k");
    assert!(b.len() >= n * k, "matmul_bt: b too short for n*k");
    assert!(c.len() >= m * n, "matmul: c too short for m*n");

    // CPU path only — Accelerate AMX on macOS, SIMD/scalar elsewhere.
    #[cfg(target_os = "macos")]
    {
        accelerate_matmul_bt(a, b, c, m, n, k);
    }

    // Non-macOS fallback: hand-written SIMD with tiling for large matrices.
    #[cfg(not(target_os = "macos"))]
    {
        // Use cache-blocked (tiled) path for large matrices where blocking pays off.
        // Two conditions must be met:
        //   1. Total work > 1M elements (below this, overhead dominates).
        //   2. K >= 128 (the shared dimension must be large enough that B-rows don't fit
        //      in L1 cache naturally). When K is small (e.g. 32), each B-row is only
        //      128 bytes and fits in L1 without tiling. Tiling would only change the
        //      accumulation order and introduce unnecessary numerical differences.
        let total_work = (m as u64) * (n as u64) * (k as u64);
        if total_work >= SMALL_GEMM_ELEMENTS && k >= super::tiled::TILE_K {
            matmul_bt_tiled(a, b, c, m, k, n);
            return;
        }

        let config = simd_config();

        #[cfg(target_arch = "x86_64")]
        {
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
        }

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

/// **Unstable**: dispatch for multi-head attention's per-head GEMMs (`Q @ K^T` and
/// `scores @ V`), distinct from `matmul_bt`'s general dispatch.
///
/// `matmul_bt` always routes through the platform GEMM backend (Apple Accelerate's
/// `cblas_sgemm` on macOS), which is the right choice for the FFN/QKV/output-projection
/// GEMMs that dominate a transformer layer's FLOPs: their `m*n*k` is large enough that
/// the backend's per-call dispatch cost is negligible next to its throughput advantage.
/// Multi-head attention's per-head score and context steps are a different regime: a
/// short-text encoder call makes `num_heads` separate `matmul_bt`-shaped calls per
/// layer, each only `seq_len * seq_len * head_dim` multiply-adds (a few thousand for a
/// handful of tokens), so the *same* backend dispatch cost is now paid `num_heads`
/// times per layer for work that is individually tiny. On macOS this routes calls below
/// `SMALL_GEMM_ELEMENTS` to a direct kernel that never leaves this process instead of
/// invoking Accelerate; above that size (long sequences, where each head's GEMM is
/// large in its own right) it defers to `matmul_bt` unchanged.
///
/// On non-macOS platforms this is a pure passthrough to `matmul_bt`: that function
/// already makes this same small-vs-large split internally (see its cache-blocking
/// condition above), so there is nothing additional to bypass.
pub(crate) fn matmul_bt_attention(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    #[cfg(target_os = "macos")]
    {
        let total_work = (m as u64) * (n as u64) * (k as u64);
        if total_work < SMALL_GEMM_ELEMENTS {
            matmul_bt_small(a, b, c, m, k, n);
            return;
        }
    }
    matmul_bt(a, b, c, m, k, n);
}

/// macOS-only small-GEMM kernel backing [`matmul_bt_attention`]: computes `C = A @ B^T`
/// directly, without Apple Accelerate, for GEMMs known to be below `SMALL_GEMM_ELEMENTS`.
///
/// Guarded with the same release-active bounds checks as `matmul_bt` (#368 precedent)
/// since callers may pass reused scratch buffers longer than the exact footprint.
#[cfg(target_os = "macos")]
fn matmul_bt_small(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert!(
        m.checked_mul(k).is_some(),
        "matmul_bt_small shape overflow: m*k"
    );
    assert!(
        n.checked_mul(k).is_some(),
        "matmul_bt_small shape overflow: n*k"
    );
    assert!(
        m.checked_mul(n).is_some(),
        "matmul_bt_small shape overflow: m*n"
    );
    assert!(a.len() >= m * k, "matmul_bt_small: a too short for m*k");
    assert!(b.len() >= n * k, "matmul_bt_small: b too short for n*k");
    assert!(c.len() >= m * n, "matmul_bt_small: c too short for m*n");

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON (Advanced SIMD) is a mandatory baseline feature of every AArch64
        // core, unlike x86_64's AVX2/AVX-512 which vary by CPU and need a runtime
        // feature check. The bounds asserted above satisfy matmul_neon's own contract.
        unsafe {
            matmul_neon(a, b, c, m, k, n);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // Conservative fallback for macOS on non-aarch64 (older Intel Macs): a safe
        // scalar kernel needs no runtime feature detection. Still strictly bypasses
        // Accelerate's per-call dispatch cost for these small GEMMs.
        matmul_bt_scalar(a, b, c, m, k, n);
    }
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
}
