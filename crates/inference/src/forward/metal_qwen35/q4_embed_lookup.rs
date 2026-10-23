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

use crate::weights::q4_weights::Q4_BLOCK_BYTES;
use lattice_weights::q4_weights::{
    Q4_BLOCK_WEIGHTS, Q4BlockCheck, open_and_mmap_q4_file, q4_f16_to_f32, q4_f32_to_f16,
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

    /// Return the table's row count and stored width.
    pub(crate) fn dimensions(&self) -> (usize, usize) {
        (
            self.rows,
            self.row_bytes / Q4_BLOCK_BYTES * Q4_BLOCK_WEIGHTS,
        )
    }

    /// Dequantize row `id` into `dst`.
    ///
    /// Each weight is rounded through f16 exactly as the former load-time
    /// copy did: `nibble * scale + bias` in f32, narrowed to f16, widened
    /// back to f32.
    ///
    /// # Panics
    ///
    /// Panics before writing if `id` is outside the table or `dst` does not
    /// contain exactly the table's stored number of columns.
    pub(crate) fn write_row(&self, id: usize, dst: &mut [f32]) {
        let (rows, cols) = self.dimensions();
        assert!(
            id < rows,
            "embedding row {id} is out of range for {rows} rows"
        );
        assert_eq!(dst.len(), cols, "embedding destination width mismatch");
        let start = self.payload_offset + id * self.row_bytes;
        let row = &self.mmap[start..start + self.row_bytes];
        let mut out = 0usize;
        for block in row.chunks_exact(Q4_BLOCK_BYTES) {
            let scale = q4_f16_to_f32(u16::from_ne_bytes([block[0], block[1]]));
            let bias = q4_f16_to_f32(u16::from_ne_bytes([block[2], block[3]]));
            for &byte in &block[4..] {
                for code in [byte & 0x0f, byte >> 4] {
                    let weight = code as f32 * scale + bias;
                    dst[out] = q4_f16_to_f32(q4_f32_to_f16(weight));
                    out += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signed_scale_fixture::negative_scale_tensor;
    use crate::weights::q4_weights::save_q4_file;

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
            embedding.write_row(id, &mut actual);
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

    const SENTINEL_BITS: u32 = 0x7fc0_1234;

    fn open_fixture(rows: usize, cols: usize, seed: u64) -> (tempfile::TempDir, Q4Embedding) {
        let tensor = negative_scale_tensor(rows, cols, seed);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("embed.q4");
        save_q4_file(&path, &tensor).expect("write fixture");
        let embedding =
            Q4Embedding::open(&path, rows, cols, "embed_tokens.weight").expect("open Q4 embedding");
        (dir, embedding)
    }

    fn refused_lookup(embedding: &Q4Embedding, id: usize, len: usize) -> String {
        let mut dst = vec![f32::from_bits(SENTINEL_BITS); len];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            embedding.write_row(id, &mut dst)
        }));
        let payload = result.expect_err("the lookup must refuse this row and destination");
        assert!(
            dst.iter().all(|value| value.to_bits() == SENTINEL_BITS),
            "row {id}, destination width {len}: the lookup wrote before refusing"
        );
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                payload
                    .downcast_ref::<&str>()
                    .map(|text| (*text).to_string())
            })
            .unwrap_or_default()
    }

    #[test]
    fn row_lookup_reports_the_opened_geometry() {
        let (_dir, embedding) = open_fixture(5, 64, 3);
        assert_eq!(embedding.dimensions(), (5, 64));
    }

    #[test]
    fn row_lookup_refuses_an_out_of_range_row_before_writing() {
        let rows = 3;
        let cols = 96;
        let (_dir, embedding) = open_fixture(rows, cols, 5);
        for id in [rows, rows + 1, usize::MAX] {
            let message = refused_lookup(&embedding, id, cols);
            assert!(
                message.contains(&format!(
                    "embedding row {id} is out of range for {rows} rows"
                )),
                "row {id}: unexpected panic: {message}"
            );
        }
    }

    #[test]
    fn row_lookup_refuses_a_destination_of_the_wrong_width_before_writing() {
        let cols = 96;
        let (_dir, embedding) = open_fixture(2, cols, 9);
        for len in [0, 1, cols - 1, cols + 1, 2 * cols] {
            let message = refused_lookup(&embedding, 1, len);
            assert!(
                message.contains("embedding destination width mismatch"),
                "width {len}: unexpected panic: {message}"
            );
        }
    }
}
