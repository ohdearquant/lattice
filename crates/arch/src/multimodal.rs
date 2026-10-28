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
