//! Shared multimodal grid data.

/// Temporal, height and width dimensions of a patch grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridThw {
    /// Temporal patch count.
    pub t: usize,
    /// Patch-grid height.
    pub h: usize,
    /// Patch-grid width.
    pub w: usize,
}

impl GridThw {
    /// Total patch count.
    pub fn num_patches(&self) -> usize {
        self.t * self.h * self.w
    }
}

/// Encoder rows placed at placeholder-token positions (HF `masked_scatter`).
pub mod placeholder_scatter;

/// Experimental single-audio legacy merge; this is not model or serving support.
#[doc(hidden)]
pub mod qwen2_audio_merge;
