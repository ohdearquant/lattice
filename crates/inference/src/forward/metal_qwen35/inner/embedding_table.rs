//! The `embed_tokens` source for the CPU row lookup, split out of `inner`.

use metal::Buffer;

use super::super::q4_embed_lookup::Q4Embedding;

/// Where `embed_tokens` rows come from for the CPU embedding lookup.
pub(crate) enum EmbeddingTable {
    /// f16 rows in a shared Metal buffer: the safetensors upload and every
    /// Q8_0 build, whose GPU logits path reads the same f16 buffer.
    F16(F16Embedding),
    /// Q4 blocks in a read-only mapping of the checkpoint's own file,
    /// decoded per row. GPU logits for a Q4 checkpoint read a separate
    /// mapping.
    Q4(Q4Embedding),
}

/// Checked geometry for the shared f16 embedding buffer.
pub(crate) struct F16Embedding {
    buffer: Buffer,
    rows: usize,
    cols: usize,
}

impl EmbeddingTable {
    /// Bind the dimensions of an f16 embedding buffer.
    pub(crate) fn f16(buffer: Buffer, rows: usize, cols: usize) -> Result<Self, String> {
        let bytes = rows
            .checked_mul(cols)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| "f16 embedding geometry overflows usize".to_string())?;
        if rows == 0 || cols == 0 || u64::try_from(bytes).ok() != Some(buffer.length()) {
            return Err("f16 embedding dimensions do not match the buffer length".to_string());
        }
        Ok(Self::F16(F16Embedding { buffer, rows, cols }))
    }

    /// Write row `id` as f32 values to `dst`.
    ///
    /// # Panics
    ///
    /// Panics before writing if `id` is outside the table or `dst` does not
    /// contain exactly the table's stored number of columns.
    ///
    /// # Safety
    ///
    /// The f16 buffer must hold initialized, CPU-readable shared storage.
    /// No CPU or GPU writer may access the source while it is read, and
    /// `dst` must not overlap the embedding storage.
    pub(crate) unsafe fn write_row(&self, id: usize, dst: &mut [f32]) {
        let (rows, cols) = match self {
            Self::F16(embedding) => (embedding.rows, embedding.cols),
            Self::Q4(embedding) => embedding.dimensions(),
        };
        assert!(
            id < rows,
            "embedding row {id} is out of range for {rows} rows"
        );
        assert_eq!(dst.len(), cols, "embedding destination width mismatch");
        match self {
            Self::F16(embedding) => {
                // SAFETY: construction checked the full geometry against the buffer;
                // id and dst bounds are checked above. The caller guarantees readable,
                // initialized storage with no concurrent writes or destination overlap.
                unsafe {
                    let src = (embedding.buffer.contents() as *const u16).add(id * cols);
                    for (i, value) in dst.iter_mut().enumerate() {
                        *value = crate::weights::half_bits::f16_bits_to_f32(*src.add(i));
                    }
                }
            }
            Self::Q4(embedding) => embedding.write_row(id, dst),
        }
    }

    /// The f16 rows bound to the GPU logits kernels on a Q8_0 build.
    ///
    /// Only Q8_0 dispatches call this, and every Q8_0 construction takes
    /// [`Self::F16`], so the Q4 arm cannot be reached from a valid engine.
    pub(crate) fn f16_buffer(&self) -> &Buffer {
        match self {
            Self::F16(embedding) => &embedding.buffer,
            Self::Q4(_) => unreachable!("a Q8_0 engine always builds the f16 embedding table"),
        }
    }
}

#[cfg(test)]
mod tests {
    use metal::{Buffer, Device, MTLResourceOptions};

    use super::EmbeddingTable;
    use crate::measurement::gpu_test_lock;

    const SENTINEL_BITS: u32 = 0x7fc0_1234;

    fn f16_buffer(device: &Device, bits: &[u16]) -> Buffer {
        device.new_buffer_with_data(
            bits.as_ptr().cast(),
            std::mem::size_of_val(bits) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    #[test]
    fn f16_table_refuses_geometry_that_does_not_match_the_buffer() {
        let _gpu_guard = gpu_test_lock();
        let device = Device::system_default().expect("Metal device");
        for (rows, cols, expected) in [
            (2, 3, "do not match"),
            (3, 4, "do not match"),
            (0, 8, "do not match"),
            (8, 0, "do not match"),
            (usize::MAX, 2, "overflows"),
        ] {
            let Err(err) = EmbeddingTable::f16(f16_buffer(&device, &[0; 8]), rows, cols) else {
                panic!("[{rows}, {cols}] must not bind an 8-element buffer");
            };
            assert!(
                err.contains(expected),
                "[{rows}, {cols}]: unexpected error: {err}"
            );
        }
        assert!(EmbeddingTable::f16(f16_buffer(&device, &[0; 8]), 2, 4).is_ok());
        assert!(EmbeddingTable::f16(f16_buffer(&device, &[0; 3]), 1, 3).is_ok());
    }

    #[test]
    fn f16_table_decodes_a_row_and_refuses_bad_lookups_before_writing() {
        let _gpu_guard = gpu_test_lock();
        let device = Device::system_default().expect("Metal device");
        let bits: [u16; 6] = [0x3c00, 0x4000, 0xbc00, 0x3800, 0x0000, 0x7bff];
        let table = EmbeddingTable::f16(f16_buffer(&device, &bits), 3, 2).expect("bind table");

        let mut row = [0.0f32; 2];
        // SAFETY: the buffer was initialized from `bits`, nothing writes it, and
        // `row` is a separate stack array.
        unsafe { table.write_row(1, &mut row) };
        assert_eq!(row, [-1.0, 0.5]);

        for (id, len, expected) in [
            (3, 2, "embedding row 3 is out of range for 3 rows"),
            (usize::MAX, 2, "is out of range for 3 rows"),
            (0, 1, "embedding destination width mismatch"),
            (0, 3, "embedding destination width mismatch"),
        ] {
            let mut dst = vec![f32::from_bits(SENTINEL_BITS); len];
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // SAFETY: as above; the call is expected to refuse before reading.
                unsafe { table.write_row(id, &mut dst) }
            }));
            let payload = result.expect_err("the lookup must refuse this row and destination");
            assert!(
                dst.iter().all(|value| value.to_bits() == SENTINEL_BITS),
                "row {id}, destination width {len}: the lookup wrote before refusing"
            );
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| {
                    payload
                        .downcast_ref::<&str>()
                        .map(|text| (*text).to_string())
                })
                .unwrap_or_default();
            assert!(
                message.contains(expected),
                "row {id}, width {len}: unexpected panic: {message}"
            );
        }
    }
}
