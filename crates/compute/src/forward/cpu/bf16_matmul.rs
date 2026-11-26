//! CPU matrix multiplication with weights held as bf16 bits.
//!
//! [`matmul_bt_bf16`] widens [`BF16_TILE_ROWS`] rows of `B` at a time into caller-owned f32
//! scratch and multiplies each tile through [`matmul_bt`]. Widening a bf16 value to f32 is
//! exact, so the kernel sees the same values an f32 copy of a BF16 checkpoint holds, at half
//! the weight memory and the cost of one widen pass per call.

use super::{matmul_bt, validate_gemm_bt};

/// **Unstable**: rows of `B` that [`matmul_bt_bf16`] widens per tile.
pub const BF16_TILE_ROWS: usize = 1024;

/// **Unstable**: reusable f32 scratch for [`matmul_bt_bf16`]; layout may change.
///
/// A call grows the scratch only when it is too small for that call, so a scratch built with
/// [`Self::with_capacity`] for the largest product a caller runs is never reallocated.
#[derive(Debug, Default)]
pub struct Bf16MatmulScratch {
    b_tile: Vec<f32>,
    c_tile: Vec<f32>,
}

impl Bf16MatmulScratch {
    /// **Unstable**: an empty scratch that grows on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// **Unstable**: scratch for products with at most `max_m` rows of `A` and an inner
    /// dimension of at most `max_k`.
    ///
    /// # Panics
    ///
    /// When `max_m * BF16_TILE_ROWS` or `BF16_TILE_ROWS * max_k` overflows `usize`.
    pub fn with_capacity(max_m: usize, max_k: usize) -> Self {
        let Some(b_len) = BF16_TILE_ROWS.checked_mul(max_k) else {
            panic!("Bf16MatmulScratch: shape overflow: tile*k");
        };
        let c_len = if max_m > 1 {
            let Some(len) = max_m.checked_mul(BF16_TILE_ROWS) else {
                panic!("Bf16MatmulScratch: shape overflow: m*tile");
            };
            len
        } else {
            0
        };
        Self {
            b_tile: vec![0.0; b_len],
            c_tile: vec![0.0; c_len],
        }
    }

    /// **Unstable**: f32 elements the scratch currently holds, for memory accounting.
    pub fn held_elements(&self) -> usize {
        self.b_tile.len() + self.c_tile.len()
    }
}

#[inline]
fn widen_into(dst: &mut [f32], src: &[u16]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = f32::from_bits(u32::from(s) << 16);
    }
}

/// **Unstable**: `C = A @ B^T` with `B` held as bf16 bits; tiling may change.
///
/// Layouts:
/// - `a`: `[m, k]` row-major `f32`
/// - `b`: `[n, k]` row-major bf16 bits (`u16`)
/// - `c`: `[m, n]` row-major `f32`
///
/// Buffers longer than their logical footprint are accepted, as by [`matmul_bt`]; content of
/// `c` past `m * n` is not written. Each tile of `B` is multiplied by [`matmul_bt`], so the
/// result agrees with [`matmul_bt`] over the widened matrix to f32 rounding. It is not
/// guaranteed to be bit-identical: the backend may choose a different kernel for a tile than
/// for the whole product.
///
/// # Panics
///
/// On a shape whose products overflow `usize`, or a buffer shorter than its footprint.
pub fn matmul_bt_bf16(
    a: &[f32],
    b: &[u16],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scratch: &mut Bf16MatmulScratch,
) {
    validate_gemm_bt(a.len(), b.len(), c.len(), m, k, n, "matmul_bt_bf16");

    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        c[..m * n].fill(0.0);
        return;
    }

    // Both products below are bounded by `n * k` and `m * n`, which the validator checked.
    let tile_rows = n.min(BF16_TILE_ROWS);
    let b_need = tile_rows * k;
    if scratch.b_tile.len() < b_need {
        scratch.b_tile.resize(b_need, 0.0);
    }
    let a = &a[..m * k];

    if tile_rows == n {
        let b_tile = &mut scratch.b_tile[..b_need];
        widen_into(b_tile, &b[..n * k]);
        matmul_bt(a, b_tile, &mut c[..m * n], m, k, n);
        return;
    }

    let c_need = if m > 1 { m * tile_rows } else { 0 };
    if scratch.c_tile.len() < c_need {
        scratch.c_tile.resize(c_need, 0.0);
    }

    let mut col = 0;
    while col < n {
        let t = (n - col).min(BF16_TILE_ROWS);
        let b_tile = &mut scratch.b_tile[..t * k];
        widen_into(b_tile, &b[col * k..(col + t) * k]);
        if m == 1 {
            matmul_bt(a, b_tile, &mut c[col..col + t], 1, k, t);
        } else {
            let c_tile = &mut scratch.c_tile[..m * t];
            matmul_bt(a, b_tile, c_tile, m, k, t);
            for row in 0..m {
                c[row * n + col..row * n + col + t]
                    .copy_from_slice(&c_tile[row * t..(row + 1) * t]);
            }
        }
        col += t;
    }
}

#[cfg(test)]
mod tests;
