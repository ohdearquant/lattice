//! The `embed_tokens` source for the CPU row lookup, split out of `inner`.

use metal::Buffer;

use super::super::q4_embed_lookup::Q4Embedding;

/// Where `embed_tokens` rows come from for the CPU embedding lookup.
pub(crate) enum EmbeddingTable {
    /// f16 rows in a shared Metal buffer: the safetensors upload and every
    /// Q8_0 build, whose GPU logits path reads the same f16 buffer.
    F16(Buffer),
    /// Q4 blocks in a read-only mapping of the checkpoint's own file,
    /// decoded per row. GPU logits for a Q4 checkpoint read a separate
    /// mapping.
    Q4(Q4Embedding),
}

impl EmbeddingTable {
    /// Write row `id` as f32 values to `dst`.
    ///
    /// # Safety
    ///
    /// `dst` must point to at least `cols` writable f32 values that do not
    /// alias the embedding storage.
    pub(crate) unsafe fn write_row(&self, id: usize, dst: *mut f32, cols: usize) {
        match self {
            Self::F16(buffer) => {
                // SAFETY: the buffer holds f16 rows of `cols` values, and
                // the caller guarantees `dst` holds `cols` f32 values.
                unsafe {
                    let src = (buffer.contents() as *const u16).add(id * cols);
                    for i in 0..cols {
                        *dst.add(i) = crate::weights::half_bits::f16_bits_to_f32(*src.add(i));
                    }
                }
            }
            Self::Q4(embedding) => {
                // SAFETY: forwarded caller guarantee; the Q4 table was
                // opened for exactly this many rows and columns.
                unsafe { embedding.write_row(id, dst) }
            }
        }
    }

    /// The f16 rows bound to the GPU logits kernels on a Q8_0 build.
    ///
    /// Only Q8_0 dispatches call this, and every Q8_0 construction takes
    /// [`Self::F16`], so the Q4 arm cannot be reached from a valid engine.
    pub(crate) fn f16_buffer(&self) -> &Buffer {
        match self {
            Self::F16(buffer) => buffer,
            Self::Q4(_) => unreachable!("a Q8_0 engine always builds the f16 embedding table"),
        }
    }
}
