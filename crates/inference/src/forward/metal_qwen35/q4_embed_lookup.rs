//! CPU row lookup for a Q4 embedding table read in place from its file.
//!
//! The Q4 Metal loader used to dequantize the whole embedding table to f16
//! at load time because the CPU lookup read that buffer. Decoding a row at
//! a time from the same payload the GPU logits GEMV addresses removes that
//! copy without changing the values the lookup produces.
//!
//! Only the row decode lives here so it is unit-testable without a device;
//! the table type that pairs it with the f16 Metal arm lives in `inner`,
//! where the Metal types are available.

use crate::weights::q4_weights::{
    Q4_BLOCK_BYTES, Q4_BLOCK_WEIGHTS, Q4BlockCheck, open_and_mmap_q4_file, q4_f16_to_f32,
    q4_f32_to_f16,
};

/// A `[rows, cols]` Q4 tensor whose blocks are read from a read-only mapping
/// of its `.q4` file.
pub(crate) struct Q4Embedding {
    mmap: memmap2::Mmap,
    payload_offset: usize,
    row_bytes: usize,
    rows: usize,
}

impl Q4Embedding {
    /// Map `path`, which must hold a `[rows, cols]` Q4 tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when `cols` is empty or does not end on a Q4 block
    /// boundary, when the file fails Q4 ingress validation, or when the
    /// mapped payload is shorter than the declared rows require.
    pub(crate) fn open(
        path: &std::path::Path,
        rows: usize,
        cols: usize,
        tensor_name: &str,
    ) -> Result<Self, String> {
        if rows == 0 {
            return Err(format!("{}: embedding table has no rows", path.display()));
        }
        if cols == 0 || !cols.is_multiple_of(Q4_BLOCK_WEIGHTS) {
            return Err(format!(
                "{}: embedding width {cols} is not a multiple of {Q4_BLOCK_WEIGHTS}; \
                 a row lookup needs each row to end on a block boundary",
                path.display()
            ));
        }
        let (header, mmap, _) =
            open_and_mmap_q4_file(path, Some(&[rows, cols]), Q4BlockCheck::Now { tensor_name })?;
        let payload_offset = usize::try_from(header.payload_offset)
            .map_err(|_| format!("{}: payload offset exceeds usize", path.display()))?;
        let row_bytes = (cols / Q4_BLOCK_WEIGHTS)
            .checked_mul(Q4_BLOCK_BYTES)
            .ok_or_else(|| format!("{}: embedding row size overflows usize", path.display()))?;
        let required = rows
            .checked_mul(row_bytes)
            .ok_or_else(|| format!("{}: embedding payload size overflows usize", path.display()))?;
        let payload_len = mmap.len().saturating_sub(payload_offset);
        if payload_len < required {
            return Err(format!(
                "{}: mapped payload is {payload_len} bytes, shorter than the {required} bytes \
                 needed by a [{rows}, {cols}] Q4 tensor",
                path.display()
            ));
        }
        Ok(Self {
            mmap,
            payload_offset,
            row_bytes,
            rows,
        })
    }

    /// Dequantize row `id` into `dst[..cols]`.
    ///
    /// Each weight is rounded through f16 exactly as the former load-time
    /// copy did: `nibble * scale + bias` in f32, narrowed to f16, widened
    /// back to f32.
    ///
    /// # Safety
    ///
    /// `dst` must point to at least `cols` writable f32 values, and must not
    /// alias this mapping.
    ///
    /// Callers validate `id` against the model's vocabulary size, which the
    /// constructor has already tied to this table's row count; an id outside
    /// that range is a caller bug; the lookup writes nothing rather than
    /// reading a different row.
    pub(crate) unsafe fn write_row(&self, id: usize, dst: *mut f32) {
        if id >= self.rows {
            debug_assert!(
                false,
                "embedding row {id} is out of range for {} rows",
                self.rows
            );
            return;
        }
        let start = self.payload_offset + id * self.row_bytes;
        let Some(row) = self.mmap.get(start..start + self.row_bytes) else {
            debug_assert!(false, "embedding row {id} lies outside the mapped payload");
            return;
        };
        let mut out = 0usize;
        for block in row.chunks_exact(Q4_BLOCK_BYTES) {
            let scale = q4_f16_to_f32(u16::from_ne_bytes([block[0], block[1]]));
            let bias = q4_f16_to_f32(u16::from_ne_bytes([block[2], block[3]]));
            for &byte in &block[4..] {
                for code in [byte & 0x0f, byte >> 4] {
                    let weight = code as f32 * scale + bias;
                    // SAFETY: the caller guarantees `cols` writable slots, and
                    // each row holds exactly `cols` weights.
                    unsafe { *dst.add(out) = q4_f16_to_f32(q4_f32_to_f16(weight)) };
                    out += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::q4_weights::save_q4_file;
    use crate::weights::q4_weights::signed_scale_fixture::negative_scale_tensor;

    fn expected_row(
        tensor: &crate::weights::q4_weights::Q4Tensor,
        id: usize,
        cols: usize,
    ) -> Vec<f32> {
        let blocks_per_row = cols / Q4_BLOCK_WEIGHTS;
        (0..cols)
            .map(|j| {
                let block = &tensor.blocks[id * blocks_per_row + j / Q4_BLOCK_WEIGHTS];
                let scale = q4_f16_to_f32(block.scale);
                let bias = q4_f16_to_f32(block.bias);
                let byte = block.packed[(j % Q4_BLOCK_WEIGHTS) / 2];
                let code = if j % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                q4_f16_to_f32(q4_f32_to_f16(code as f32 * scale + bias))
            })
            .collect()
    }

    #[test]
    fn row_lookup_is_bit_identical_to_the_f16_round_trip() {
        let rows = 7;
        let cols = 96;
        let tensor = negative_scale_tensor(rows, cols, 0x5eed);
        assert!(
            tensor.blocks.iter().any(|b| q4_f16_to_f32(b.scale) < 0.0),
            "fixture must exercise negative scales"
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("embed.q4");
        save_q4_file(&path, &tensor).expect("write fixture");
        let embedding =
            Q4Embedding::open(&path, rows, cols, "embed_tokens.weight").expect("open Q4 embedding");

        for id in 0..rows {
            let mut actual = vec![0.0f32; cols];
            // SAFETY: `actual` holds `cols` f32 values and is not part of the mapping.
            unsafe { embedding.write_row(id, actual.as_mut_ptr()) };
            let expected = expected_row(&tensor, id, cols);
            for (j, (a, e)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(a.to_bits(), e.to_bits(), "row {id} column {j}: {a} != {e}");
            }
        }
    }

    #[test]
    fn row_lookup_rejects_a_width_off_the_block_boundary() {
        let tensor = negative_scale_tensor(2, 96, 7);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("embed.q4");
        save_q4_file(&path, &tensor).expect("write fixture");
        let Err(err) = Q4Embedding::open(&path, 2, 95, "embed_tokens.weight") else {
            panic!("a width off the block boundary must be rejected");
        };
        assert!(err.contains("multiple of"), "unexpected error: {err}");
    }

    #[test]
    fn row_lookup_rejects_a_shape_that_does_not_match_the_file() {
        let tensor = negative_scale_tensor(2, 96, 11);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("embed.q4");
        save_q4_file(&path, &tensor).expect("write fixture");
        let Err(err) = Q4Embedding::open(&path, 3, 96, "embed_tokens.weight") else {
            panic!("a shape mismatch must be rejected");
        };
        assert!(err.contains("shape"), "unexpected error: {err}");
    }
}
