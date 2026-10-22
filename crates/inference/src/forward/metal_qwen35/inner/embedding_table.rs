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
        assert!(id < rows, "embedding row {id} is out of range for {rows} rows");
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
