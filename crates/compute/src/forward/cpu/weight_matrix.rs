//! Row-major weight matrices held as f32 values or as bf16 bits.
//!
//! [`WeightMatrix::matmul_bt`] is `C = A @ W^T` over either store: the f32 store calls
//! [`matmul_bt`] directly and leaves the scratch untouched, and the bf16 store goes through
//! [`matmul_bt_bf16`]. Load-time consumers that need the values as f32 (packers, quantizers,
//! training) read them through [`WeightMatrix::to_f32`], which borrows the f32 store and widens
//! the bf16 store exactly.

use std::borrow::Cow;

use super::bf16_matmul::{Bf16MatmulScratch, matmul_bt_bf16};
use super::matmul_bt;

/// **Unstable**: a row-major weight matrix held as f32 values or as bf16 bits.
#[derive(Debug, Clone, PartialEq)]
pub enum WeightMatrix {
    /// f32 values.
    F32(Vec<f32>),
    /// bf16 bits, widened exactly to f32 wherever they are read.
    Bf16(Vec<u16>),
}

/// **Unstable**: how a CPU model holds its large weight matrices.
///
/// [`CpuWeightStore::Bf16`] holds a matrix as bf16 bits only when the checkpoint stores it as
/// BF16, so no value is ever rounded: every other dtype, and every small tensor such as a norm,
/// stays f32. Each product then widens the bits in tiles as [`matmul_bt_bf16`] does, which
/// halves the resident bytes of those matrices and costs time per product.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CpuWeightStore {
    /// Hold every matrix as f32 values.
    #[default]
    F32,
    /// Hold the matrices the checkpoint stores as BF16 as bf16 bits.
    Bf16,
}

impl CpuWeightStore {
    /// **Unstable**: the store's name as the command line spells it (`f32` or `bf16`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
        }
    }
}

impl std::fmt::Display for CpuWeightStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Default for WeightMatrix {
    fn default() -> Self {
        Self::F32(Vec::new())
    }
}

impl From<Vec<f32>> for WeightMatrix {
    fn from(values: Vec<f32>) -> Self {
        Self::F32(values)
    }
}

impl FromIterator<f32> for WeightMatrix {
    fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
        Self::F32(iter.into_iter().collect())
    }
}

#[inline]
fn widen(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

impl WeightMatrix {
    /// **Unstable**: number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::Bf16(v) => v.len(),
        }
    }

    /// **Unstable**: whether the matrix holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// **Unstable**: whether the matrix is held as bf16 bits.
    pub fn is_bf16(&self) -> bool {
        matches!(self, Self::Bf16(_))
    }

    /// **Unstable**: bytes the elements occupy: four per element at f32, two at bf16.
    pub fn held_bytes(&self) -> usize {
        match self {
            Self::F32(v) => std::mem::size_of_val(v.as_slice()),
            Self::Bf16(v) => std::mem::size_of_val(v.as_slice()),
        }
    }

    /// **Unstable**: the f32 values, or `None` for the bf16 store.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(v) => Some(v),
            Self::Bf16(_) => None,
        }
    }

    /// **Unstable**: the f32 values for writing, or `None` for the bf16 store.
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        match self {
            Self::F32(v) => Some(v),
            Self::Bf16(_) => None,
        }
    }

    /// **Unstable**: the values as f32: borrowed from the f32 store, widened from the bf16 store.
    pub fn to_f32(&self) -> Cow<'_, [f32]> {
        match self {
            Self::F32(v) => Cow::Borrowed(v),
            Self::Bf16(v) => Cow::Owned(v.iter().map(|&b| widen(b)).collect()),
        }
    }

    /// **Unstable**: copy row `row` of a matrix whose rows are `dst.len()` wide into `dst`.
    ///
    /// # Panics
    ///
    /// When the row lies past the end of the matrix.
    pub fn copy_row_into(&self, row: usize, dst: &mut [f32]) {
        let width = dst.len();
        let Some(start) = row.checked_mul(width) else {
            panic!("WeightMatrix::copy_row_into: shape overflow: row*width");
        };
        let end = start + width;
        match self {
            Self::F32(v) => dst.copy_from_slice(&v[start..end]),
            Self::Bf16(v) => {
                for (d, &s) in dst.iter_mut().zip(&v[start..end]) {
                    *d = widen(s);
                }
            }
        }
    }

    /// **Unstable**: `C = A @ W^T` with this matrix as `W`, laid out `[n, k]` row-major.
    ///
    /// The f32 store calls [`matmul_bt`] and never touches `scratch`; the bf16 store calls
    /// [`matmul_bt_bf16`] with it. Panics as those functions do on a short buffer or an
    /// overflowing shape.
    #[inline]
    pub fn matmul_bt(
        &self,
        a: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        scratch: &mut Bf16MatmulScratch,
    ) {
        match self {
            Self::F32(w) => matmul_bt(a, w, c, m, k, n),
            Self::Bf16(w) => matmul_bt_bf16(a, w, c, m, k, n, scratch),
        }
    }
}

#[cfg(test)]
mod tests;
