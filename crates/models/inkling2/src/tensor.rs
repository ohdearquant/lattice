//! Matrices as stored: dense rows in f32 or bf16, and banks of packed experts.

use lattice_inference_core::InferenceError;
use lattice_weights::half_bits::bf16_bits_to_f32;
use lattice_weights::mxfp8::{MXFP8_BLOCK_LEN, matvec_mxfp8};
use lattice_weights::nvfp4::{NVFP4_BLOCK_LEN, matvec_nvfp4};

use crate::ops::dot;

fn invalid(message: String) -> InferenceError {
    InferenceError::InvalidInput(message)
}

#[derive(Debug, Clone)]
enum Elements {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}

/// A row-major matrix kept in its stored element type. Rows are widened to f32 where they are
/// read.
#[derive(Debug, Clone)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    elements: Elements,
}

impl DenseMatrix {
    /// A matrix of f32 elements in row-major order.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidInput`] when a dimension is zero or `values` does not
    /// hold `rows * cols` elements.
    pub fn from_f32(rows: usize, cols: usize, values: Vec<f32>) -> Result<Self, InferenceError> {
        check_shape(rows, cols, values.len())?;
        Ok(Self {
            rows,
            cols,
            elements: Elements::F32(values),
        })
    }

    /// A matrix of bf16 elements in row-major order, given as their bit patterns.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidInput`] when a dimension is zero or `bits` does not
    /// hold `rows * cols` elements.
    pub fn from_bf16_bits(
        rows: usize,
        cols: usize,
        bits: Vec<u16>,
    ) -> Result<Self, InferenceError> {
        check_shape(rows, cols, bits.len())?;
        Ok(Self {
            rows,
            cols,
            elements: Elements::Bf16(bits),
        })
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The element at `(row, col)` widened to f32, or `None` outside the matrix.
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let index = row * self.cols + col;
        Some(match &self.elements {
            Elements::F32(values) => values[index],
            Elements::Bf16(bits) => bf16_bits_to_f32(bits[index]),
        })
    }

    /// Writes row `row`, widened to f32, into the first `cols` elements of `out`.
    pub(crate) fn row_into(&self, row: usize, out: &mut [f32]) {
        let range = row * self.cols..(row + 1) * self.cols;
        match &self.elements {
            Elements::F32(values) => out[..self.cols].copy_from_slice(&values[range]),
            Elements::Bf16(bits) => {
                for (o, &b) in out.iter_mut().zip(&bits[range]) {
                    *o = bf16_bits_to_f32(b);
                }
            }
        }
    }

    /// `out[r] = row r . x` for the first `out.len()` rows; `x` has `cols` elements.
    pub(crate) fn matvec(&self, x: &[f32], out: &mut [f32]) {
        match &self.elements {
            Elements::F32(values) => {
                for (o, row) in out.iter_mut().zip(values.chunks_exact(self.cols)) {
                    *o = dot(row, x);
                }
            }
            Elements::Bf16(bits) => {
                for (o, row) in out.iter_mut().zip(bits.chunks_exact(self.cols)) {
                    *o = row
                        .iter()
                        .zip(x)
                        .map(|(&b, &v)| bf16_bits_to_f32(b) * v)
                        .sum();
                }
            }
        }
    }
}

fn check_shape(rows: usize, cols: usize, len: usize) -> Result<(), InferenceError> {
    if rows == 0 || cols == 0 {
        return Err(invalid(format!(
            "matrix dimensions must be positive, got {rows} x {cols}"
        )));
    }
    if rows.checked_mul(cols) != Some(len) {
        return Err(invalid(format!(
            "a {rows} x {cols} matrix was given {len} elements"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
enum Packed {
    Nvfp4 {
        codes: Vec<u8>,
        scales: Vec<u8>,
        multipliers: Vec<f32>,
    },
    Mxfp8 {
        values: Vec<u8>,
        scales: Vec<u8>,
    },
}

/// One packed `rows x cols` matrix per routed expert, multiplied without being decoded to a
/// dense matrix.
#[derive(Debug, Clone)]
pub struct ExpertBank {
    experts: usize,
    rows: usize,
    cols: usize,
    packed: Packed,
}

impl ExpertBank {
    /// NVFP4 experts. `codes` is `[experts, rows, cols / 2]`, two E2M1 codes per byte with the
    /// even element in the low nibble; `scales` is `[experts, rows, cols / 16]` UE4M3 bytes, one
    /// per 16 consecutive elements of a row; every element of expert `e` is also multiplied by
    /// `multipliers[e]`.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidInput`] for a zero dimension, a column count that is
    /// not a multiple of 16, buffer lengths that do not match the shape, a scale with its sign
    /// bit set, or a multiplier that is negative or not finite.
    pub fn nvfp4(
        experts: usize,
        rows: usize,
        cols: usize,
        codes: Vec<u8>,
        scales: Vec<u8>,
        multipliers: Vec<f32>,
    ) -> Result<Self, InferenceError> {
        let elements = bank_elements(experts, rows, cols, NVFP4_BLOCK_LEN, "NVFP4")?;
        if codes.len() != elements / 2
            || scales.len() != elements / NVFP4_BLOCK_LEN
            || multipliers.len() != experts
        {
            return Err(invalid(format!(
                "an NVFP4 expert bank of {experts} x {rows} x {cols} was given {} code bytes, {} scales and {} multipliers",
                codes.len(),
                scales.len(),
                multipliers.len()
            )));
        }
        if let Some(index) = scales.iter().position(|&scale| scale & 0x80 != 0) {
            return Err(invalid(format!("NVFP4 scale {index} has its sign bit set")));
        }
        if let Some(index) = multipliers
            .iter()
            .position(|m| !m.is_finite() || m.is_sign_negative())
        {
            return Err(invalid(format!(
                "NVFP4 multiplier {index} is negative or not finite"
            )));
        }
        Ok(Self {
            experts,
            rows,
            cols,
            packed: Packed::Nvfp4 {
                codes,
                scales,
                multipliers,
            },
        })
    }

    /// MXFP8 experts. `values` is `[experts, rows, cols]` FP8 E4M3 bytes; `scales` is
    /// `[experts, rows, cols / 32]` E8M0 bytes, one per 32 consecutive elements of a row.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidInput`] for a zero dimension, a column count that is
    /// not a multiple of 32, buffer lengths that do not match the shape, or a scale of 255,
    /// which encodes NaN.
    pub fn mxfp8(
        experts: usize,
        rows: usize,
        cols: usize,
        values: Vec<u8>,
        scales: Vec<u8>,
    ) -> Result<Self, InferenceError> {
        let elements = bank_elements(experts, rows, cols, MXFP8_BLOCK_LEN, "MXFP8")?;
        if values.len() != elements || scales.len() != elements / MXFP8_BLOCK_LEN {
            return Err(invalid(format!(
                "an MXFP8 expert bank of {experts} x {rows} x {cols} was given {} values and {} scales",
                values.len(),
                scales.len()
            )));
        }
        if let Some(index) = scales.iter().position(|&scale| scale == u8::MAX) {
            return Err(invalid(format!(
                "MXFP8 scale {index} is 255, which encodes NaN"
            )));
        }
        Ok(Self {
            experts,
            rows,
            cols,
            packed: Packed::Mxfp8 { values, scales },
        })
    }

    /// Number of experts.
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// Rows of each expert matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns of each expert matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `out = W_expert x`, with `x` of `cols` and `out` of `rows` elements.
    pub(crate) fn matvec(
        &self,
        expert: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<(), InferenceError> {
        let elements = self.rows * self.cols;
        match &self.packed {
            Packed::Nvfp4 {
                codes,
                scales,
                multipliers,
            } => {
                let code_len = elements / 2;
                let scale_len = elements / NVFP4_BLOCK_LEN;
                matvec_nvfp4(
                    self.rows,
                    self.cols,
                    &codes[expert * code_len..(expert + 1) * code_len],
                    &scales[expert * scale_len..(expert + 1) * scale_len],
                    multipliers[expert],
                    x,
                    out,
                )
            }
            Packed::Mxfp8 { values, scales } => {
                let scale_len = elements / MXFP8_BLOCK_LEN;
                matvec_mxfp8(
                    self.rows,
                    self.cols,
                    &values[expert * elements..(expert + 1) * elements],
                    &scales[expert * scale_len..(expert + 1) * scale_len],
                    x,
                    out,
                )
            }
        }
    }
}

fn bank_elements(
    experts: usize,
    rows: usize,
    cols: usize,
    block: usize,
    format: &str,
) -> Result<usize, InferenceError> {
    if experts == 0 || rows == 0 || cols == 0 {
        return Err(invalid(format!(
            "{format} expert bank dimensions must be positive"
        )));
    }
    if !cols.is_multiple_of(block) {
        return Err(invalid(format!(
            "{format} expert columns must be a multiple of {block}, got {cols}"
        )));
    }
    experts
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(cols))
        .ok_or_else(|| invalid(format!("{format} expert bank size overflows")))
}
