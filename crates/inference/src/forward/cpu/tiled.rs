// ===================================================================
// Cache-blocked (tiled) matrix multiplication: C = A @ B^T
// ===================================================================
//
// Tile sizes chosen to fit working set in L1 data cache (128KB on Apple M-series):
//   TILE_K = 128: shared dimension tile (512 bytes per row of A/B at f32)
//   TILE_I = 4:   output rows processed together
//   TILE_J = 8:   output columns (B rows) processed together
//
// On macOS these constants and tiled functions are bypassed by Accelerate but
// retained for non-macOS targets (Linux, WASM, etc.).
#[cfg(not(target_os = "macos"))]
use super::simd::simd_config;

#[cfg(not(target_os = "macos"))]
pub(super) const TILE_K: usize = 128;
#[cfg(not(target_os = "macos"))]
pub(super) const TILE_I: usize = 4;
#[cfg(not(target_os = "macos"))]
pub(super) const TILE_J: usize = 8;

/// Cache-blocked matmul_bt dispatcher. Selects SIMD or scalar microkernel.
#[cfg(not(target_os = "macos"))]
pub(super) fn matmul_bt_tiled(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Zero output — we accumulate partial sums from K-tiles.
    c.fill(0.0);

    let config = simd_config();

    #[cfg(target_arch = "aarch64")]
    {
        if config.neon_enabled {
            // SAFETY: NEON is available on aarch64 and the runtime gate ensures this path.
            unsafe {
                super::tiled_neon::matmul_bt_tiled_neon(a, b, c, m, k, n);
                return;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if config.avx2_enabled && config.fma_enabled {
            // SAFETY: The runtime feature checks above guarantee AVX2+FMA support.
            unsafe {
                super::tiled_avx2::matmul_bt_tiled_avx2(a, b, c, m, k, n);
                return;
            }
        }
    }

    matmul_bt_tiled_scalar(a, b, c, m, k, n);
}

/// Scalar cache-blocked matmul_bt. Used as fallback when no SIMD is available.
#[cfg(not(target_os = "macos"))]
fn matmul_bt_tiled_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Loop order: J-tiles (B columns) → I-tiles (A rows) → K-tiles (shared dim)
    // This keeps B tiles hot in cache across I iterations.
    let mut j_start = 0;
    while j_start < n {
        let j_end = (j_start + TILE_J).min(n);

        let mut i_start = 0;
        while i_start < m {
            let i_end = (i_start + TILE_I).min(m);

            let mut k_start = 0;
            while k_start < k {
                let k_end = (k_start + TILE_K).min(k);

                // Microkernel: accumulate C[i_start..i_end, j_start..j_end]
                // += A[i_start..i_end, k_start..k_end] @ B^T[j_start..j_end, k_start..k_end]
                for i in i_start..i_end {
                    let a_row = &a[i * k + k_start..i * k + k_end];
                    for j in j_start..j_end {
                        let b_row = &b[j * k + k_start..j * k + k_end];
                        let mut sum = 0.0f32;
                        for p in 0..a_row.len() {
                            sum += a_row[p] * b_row[p];
                        }
                        c[i * n + j] += sum;
                    }
                }

                k_start += TILE_K;
            }
            i_start += TILE_I;
        }
        j_start += TILE_J;
    }
}
