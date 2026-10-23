//! CPU matrix multiplication for per-row symmetric INT8 weights.

#![forbid(unsafe_code)]

use crate::error::InferenceError;
use crate::weights::q8_weights::Q8Matrix;
#[cfg(target_os = "macos")]
use std::mem::size_of;

#[cfg(target_os = "macos")]
const TILE_N: usize = 64;

#[derive(Debug, Default)]
pub(crate) struct Q8MatmulScratch {
    #[cfg(target_os = "macos")]
    b_f32: Vec<f32>,
    #[cfg(target_os = "macos")]
    c_tile: Vec<f32>,
    #[cfg(test)]
    calls: usize,
}

impl Q8MatmulScratch {
    pub(crate) fn prepare(&mut self, max_m: usize, max_k: usize) -> Result<(), InferenceError> {
        #[cfg(target_os = "macos")]
        {
            // The tile scratch buffers are sized by `TILE_N * k` and `m * TILE_N`,
            // which are NOT covered by the `m*k` / `n*k` / `m*n` guards in the
            // caller: when `n < TILE_N` (including `n == 0`) these products can
            // overflow while the guarded ones do not, and a wrapped size would
            // under-allocate the scratch and drive an OOB scatter below. Guard
            // them on the same fail-closed contract.
            let b_len = TILE_N.checked_mul(max_k).ok_or_else(|| {
                InferenceError::InvalidInput("matmul shape overflow: TILE_N*k".into())
            })?;
            let c_len = max_m.checked_mul(TILE_N).ok_or_else(|| {
                InferenceError::InvalidInput("matmul shape overflow: m*TILE_N".into())
            })?;
            for (len, label) in [(b_len, "TILE_N*k"), (c_len, "m*TILE_N")] {
                if len
                    .checked_mul(size_of::<f32>())
                    .is_none_or(|bytes| bytes > isize::MAX as usize)
                {
                    return Err(InferenceError::InvalidInput(format!(
                        "matmul scratch byte extent exceeds isize::MAX: {label}"
                    )));
                }
            }
            if self.b_f32.len() < b_len {
                self.b_f32.resize(b_len, 0.0);
            }
            if self.c_tile.len() < c_len {
                self.c_tile.resize(c_len, 0.0);
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = (max_m, max_k);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn call_count(&self) -> usize {
        self.calls
    }

    #[cfg(test)]
    pub(crate) fn storage_snapshot(&self) -> [[usize; 3]; 2] {
        #[cfg(target_os = "macos")]
        {
            [
                [
                    self.b_f32.as_ptr() as usize,
                    self.b_f32.len(),
                    self.b_f32.capacity(),
                ],
                [
                    self.c_tile.as_ptr() as usize,
                    self.c_tile.len(),
                    self.c_tile.capacity(),
                ],
            ]
        }
        #[cfg(not(target_os = "macos"))]
        {
            [[0; 3]; 2]
        }
    }

    #[cfg(test)]
    pub(crate) fn poison(&mut self) {
        #[cfg(target_os = "macos")]
        {
            self.b_f32.fill(f32::NAN);
            self.c_tile.fill(f32::NAN);
        }
    }
}

/// **Unstable**: quantized matmul_bt kernel; tile size and dispatch strategy may change.
///
/// Compute `C = A @ B_q^T` where `B_q` is stored as per-row quantized INT8.
///
/// Shapes:
/// - `A`: `[m, k]` in row-major f32
/// - `B_q`: `[n, k]` in row-major Q8
/// - `C`: `[m, n]` in row-major f32
///
/// Each output is computed as:
///
/// ```text
/// c[i, j] = scale[j] * sum_t(a[i, t] * q[j, t])
/// ```
///
/// The activation input remains f32, so the inner product is accumulated in f32 and the
/// per-row scale is applied once per output element.
pub fn matmul_bt_q8(a: &[f32], b_q: &Q8Matrix, c: &mut [f32], m: usize, k: usize, n: usize) {
    matmul_bt_q8_with_scratch(a, b_q, c, m, k, n, &mut Q8MatmulScratch::default());
}

pub(crate) fn matmul_bt_q8_with_scratch(
    a: &[f32],
    b_q: &Q8Matrix,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scratch: &mut Q8MatmulScratch,
) {
    // Guard the dimension products against usize overflow BEFORE any `m * k` /
    // `n * k` / `m * n` is evaluated below: in release those multiplies wrap
    // silently, so an overflowing shape could pass a length check and drive an
    // out-of-bounds access. Mirrors the same guards in `matmul_bt`
    // (forward/cpu/matmul.rs); `matmul_bt_q8` was missing them.
    assert!(m.checked_mul(k).is_some(), "matmul shape overflow: m*k");
    assert!(n.checked_mul(k).is_some(), "matmul shape overflow: n*k");
    assert!(m.checked_mul(n).is_some(), "matmul shape overflow: m*n");
    // Oversized-scratch-prefix allow-list (ADR-080 C4): `>=`, not `assert_eq!`. Every access
    // to `a` below is a bounded sub-slice (`a[i*k..(i+1)*k]`), so a caller-supplied `a`
    // longer than `m*k` is sound — unifies with `matmul_bt`/`matmul_bt_f16`'s contract
    // instead of this Q8 path alone rejecting valid oversized callers. `b_q.rows`/`b_q.cols`
    // are the matrix's own declared shape (not a raw buffer length), so those stay an exact
    // shape-mismatch check, not a buffer-size check.
    assert!(a.len() >= m * k, "A length does not match m * k");
    assert_eq!(b_q.rows, n, "B_q rows do not match n");
    assert_eq!(b_q.cols, k, "B_q cols do not match k");
    assert_eq!(
        b_q.data.len(),
        n * k,
        "B_q data length does not match n * k"
    );
    assert_eq!(b_q.scales.len(), n, "B_q scales length does not match n");
    assert!(c.len() >= m * n, "C buffer is too small");

    // On macOS we tile the N (output-channel) dimension, dequantize each tile
    // of B rows from i8→f32 (applying per-row scales), then delegate to the
    // Accelerate-backed matmul_bt which hits the AMX coprocessor.
    //
    // Tile sizing: 64 rows * 2048 cols * 4 bytes = 512 KB fits comfortably in
    // the M-series 16 MB L2 cache. For M=1 (decode) we write tile results
    // directly into c; for M>1 we scatter from a contiguous tile output buffer.
    #[cfg(target_os = "macos")]
    {
        if let Err(error) = scratch.prepare(m, k) {
            panic!("{error}");
        }
        let b_f32 = &mut scratch.b_f32[..TILE_N * k];
        let c_tile = &mut scratch.c_tile[..m * TILE_N];

        for tile_start in (0..n).step_by(TILE_N) {
            let tile_n = (n - tile_start).min(TILE_N);

            // Dequantize: convert i8 -> f32 and fold in the per-row scale.
            for j in 0..tile_n {
                let global_j = tile_start + j;
                let q_row = &b_q.data[global_j * k..(global_j + 1) * k];
                let scale = b_q.scales[global_j];
                let dst = &mut b_f32[j * k..(j + 1) * k];
                for t in 0..k {
                    dst[t] = q_row[t] as f32 * scale;
                }
            }

            // Accelerate BLAS GEMM: C_tile[m, tile_n] = A[m, k] @ B_tile[tile_n, k]^T
            crate::forward::cpu::matmul_bt(
                a,
                &b_f32[..tile_n * k],
                &mut c_tile[..m * tile_n],
                m,
                k,
                tile_n,
            );

            // Scatter tile results into correct columns of c.
            if m == 1 {
                // Fast path for decode: c is [1, n], tile writes a contiguous run.
                c[tile_start..tile_start + tile_n].copy_from_slice(&c_tile[..tile_n]);
            } else {
                for i in 0..m {
                    let c_row = &mut c[i * n + tile_start..i * n + tile_start + tile_n];
                    let tile_row = &c_tile[i * tile_n..(i + 1) * tile_n];
                    c_row.copy_from_slice(tile_row);
                }
            }
        }
    }

    // Scalar fallback for non-macOS platforms.
    #[cfg(not(target_os = "macos"))]
    {
        let _ = &scratch;
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let c_row = &mut c[i * n..(i + 1) * n];

            for j in 0..n {
                let q_row = &b_q.data[j * k..(j + 1) * k];
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += a_row[t] * q_row[t] as f32;
                }
                c_row[j] = acc * b_q.scales[j];
            }
        }
    }

    #[cfg(test)]
    {
        scratch.calls += 1;
    }
}

#[cfg(test)]
mod scratch_tests;

#[cfg(test)]
mod tests;
