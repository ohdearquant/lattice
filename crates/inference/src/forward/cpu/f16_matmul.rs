//! Mixed-precision CPU matrix multiplication with packed F16 weights.

use crate::weights::f16_weights::F16;

#[cfg_attr(target_os = "macos", allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DotKernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    AvxF16c,
}

#[cfg_attr(target_os = "macos", allow(dead_code))]
impl DotKernel {
    #[inline]
    fn select() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx")
                && std::arch::is_x86_feature_detected!("f16c")
            {
                return Self::AvxF16c;
            }
        }

        Self::Scalar
    }

    #[inline]
    fn dot(self, a: &[f32], b: &[u16]) -> f32 {
        match self {
            Self::Scalar => dot_f16_f32_scalar(a, b),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: DotKernel::select only returns AvxF16c after runtime
            // AVX + F16C detection; dot_f16_f32_avx validates equal lengths.
            Self::AvxF16c => unsafe { dot_f16_f32_avx(a, b) },
        }
    }
}

#[cfg_attr(target_os = "macos", allow(dead_code))]
#[inline]
fn dot_f16_f32_scalar(a: &[f32], b: &[u16]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * F16(b[i]).to_f32();
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn dot_f16_f32_avx(a: &[f32], b: &[u16]) -> f32 {
    use std::arch::x86_64::{
        __m128i, _mm_loadu_si128, _mm256_cvtph_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps,
    };

    debug_assert_eq!(a.len(), b.len());
    const LANES: usize = 8;

    let mut acc = 0.0f32;
    let mut i = 0usize;
    let mut prod = [0.0f32; 8];

    while i + LANES <= a.len() {
        debug_assert!(i + LANES <= a.len());
        debug_assert!(i + LANES <= b.len());
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let hv = _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i);
        let bv = _mm256_cvtph_ps(hv);
        let pv = _mm256_mul_ps(av, bv);
        _mm256_storeu_ps(prod.as_mut_ptr(), pv);
        acc += prod[0];
        acc += prod[1];
        acc += prod[2];
        acc += prod[3];
        acc += prod[4];
        acc += prod[5];
        acc += prod[6];
        acc += prod[7];
        i += LANES;
    }

    while i < a.len() {
        acc += a[i] * F16(b[i]).to_f32();
        i += 1;
    }

    acc
}

/// **Unstable**: mixed-precision matmul C = A @ B^T with f16 weights; dispatch may change.
///
/// Matrix multiply `C = A @ B^T` where `A` is `f32`, `B` is packed `f16`, and `C` is `f32`.
///
/// Layouts:
/// - `a`: `[m, k]` row-major `f32`
/// - `b`: `[n, k]` row-major packed half (`u16`)
/// - `c`: `[m, n]` row-major `f32`
///
/// On macOS: bulk-converts B tiles to f32 and dispatches to Accelerate cblas_sgemm (AMX).
/// On other platforms: per-element f16→f32 conversion with optional AVX F16C.
pub fn matmul_bt_f16(a: &[f32], b: &[u16], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Release-active, overflow-first, oversized-scratch-allowed contract (ADR-080 C4, held
    // finding: this previously used `assert_eq!` — an exact-length requirement — while the
    // canonical `matmul_bt` (and every access pattern in this function's own body, which
    // only ever takes bounded prefixes/sub-slices of `a`/`b`/`c`) allows oversized scratch
    // buffers. Unifies with `matmul_bt`/`matmul_bt_q8` so f32/f16/Q8 GDN projections share
    // the same oversized-input contract instead of the f16 path rejecting valid callers.
    crate::forward::cpu::validate_gemm_bt(a.len(), b.len(), c.len(), m, k, n, "matmul_bt_f16");

    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        c.fill(0.0);
        return;
    }

    // On macOS: convert f16 B tiles to f32, then use Accelerate cblas_sgemm for AMX throughput.
    // The conversion is O(tile_n * k), the GEMM is O(m * tile_n * k) — for small m (decode),
    // the conversion cost is comparable to the GEMM, but both are vastly faster than the
    // scalar per-element dot product path.
    #[cfg(target_os = "macos")]
    {
        // Tile size: convert TILE_N rows of B at a time to bound scratch memory.
        // For decode (m=1), tile_n=256 means 256*2048*4 = 2MB scratch — fits in L2.
        const TILE_N: usize = 256;
        let mut b_f32 = vec![0.0f32; TILE_N * k];

        let mut col = 0;
        while col < n {
            let tile_n = (n - col).min(TILE_N);
            let b_tile = &b[col * k..(col + tile_n) * k];
            let buf = &mut b_f32[..tile_n * k];

            // Bulk f16→f32 conversion
            for i in 0..tile_n * k {
                buf[i] = F16(b_tile[i]).to_f32();
            }

            // FP-052: use a separate tile output buffer to avoid overlap between
            // the contiguous GEMM result and previously-scattered rows in `c`.
            // Writing directly into c[col..col + m*tile_n] when m > 1 would
            // overwrite rows already scattered from the previous iteration.
            let mut c_tile = vec![0.0f32; m * tile_n];
            crate::forward::cpu::matmul_bt(a, buf, &mut c_tile, m, k, tile_n);

            // Scatter from contiguous c_tile[m, tile_n] into strided c[m, n].
            for row in 0..m {
                let src = &c_tile[row * tile_n..(row + 1) * tile_n];
                c[row * n + col..row * n + col + tile_n].copy_from_slice(src);
            }

            col += tile_n;
        }
    }

    // Non-macOS: per-element conversion with DotKernel
    #[cfg(not(target_os = "macos"))]
    {
        let kernel = DotKernel::select();

        if m == 1 {
            let a_row = &a[..k];
            for out in 0..n {
                let b_row = &b[out * k..(out + 1) * k];
                c[out] = kernel.dot(a_row, b_row);
            }
            return;
        }

        for row in 0..m {
            let a_row = &a[row * k..(row + 1) * k];
            let c_row = &mut c[row * n..(row + 1) * n];
            for out in 0..n {
                let b_row = &b[out * k..(out + 1) * k];
                c_row[out] = kernel.dot(a_row, b_row);
            }
        }
    }
}

#[cfg(test)]
fn matmul_bt_f32_reference(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(c.len(), m * n);

    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];
        let c_row = &mut c[row * n..(row + 1) * n];
        for out in 0..n {
            let b_row = &b[out * k..(out + 1) * k];
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += a_row[j] * b_row[j];
            }
            c_row[out] = acc;
        }
    }
}

#[cfg(test)]
mod tests;
