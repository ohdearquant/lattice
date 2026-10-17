//! Rotary Position Embeddings (RoPE) for decoder-only models.
//!
//! RoPE encodes position by rotating pairs of dimensions in Q and K
//! before the attention dot product. This module precomputes the
//! sin/cos tables and applies them in-place.

/// **Unstable**: internal RoPE table; only used inside forward kernels.
pub struct RopeTable {
    /// cos values: [max_seq_len, head_dim/2]
    cos: Vec<f32>,
    /// sin values: [max_seq_len, head_dim/2]
    sin: Vec<f32>,
    half_dim: usize,
}

impl RopeTable {
    /// **Unstable**: construct RoPE table; parameters and layout may change.
    ///
    /// `theta` is the base frequency (1_000_000.0 for Qwen3).
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64) -> Self {
        let half_dim = head_dim / 2;
        let mut cos = Vec::with_capacity(max_seq_len * half_dim);
        let mut sin = Vec::with_capacity(max_seq_len * half_dim);

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
                let angle = pos as f64 * freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }

        Self { cos, sin, half_dim }
    }

    /// **Unstable**: number of positions the precomputed table can serve.
    /// Callers that index `apply` / `cos_at` / `sin_at` beyond this will panic.
    #[inline]
    pub fn max_positions(&self) -> usize {
        self.cos.len().checked_div(self.half_dim).unwrap_or(0)
    }

    /// **Unstable**: row stride in the cos/sin tables — equals `head_dim / 2`.
    /// Callers performing partial RoPE need this to compute the per-position
    /// base offset correctly when the table is built for the full `head_dim`.
    #[inline]
    pub fn half_dim(&self) -> usize {
        self.half_dim
    }

    /// **Unstable**: raw cos accessor; exposed for testing only.
    #[inline]
    pub fn cos_at(&self, idx: usize) -> f32 {
        self.cos[idx]
    }

    /// **Unstable**: raw sin accessor; exposed for testing only.
    #[inline]
    pub fn sin_at(&self, idx: usize) -> f32 {
        self.sin[idx]
    }

    /// **Unstable**: apply RoPE in-place; called from forward kernels only.
    ///
    /// `x` has shape `[head_dim]` and is modified in-place.
    /// The first half_dim elements are paired with the second half_dim.
    #[inline]
    pub fn apply(&self, x: &mut [f32], position: usize) {
        let half = self.half_dim;
        debug_assert!(x.len() >= 2 * half);
        let base = position * half;

        for i in 0..half {
            let cos_val = self.cos[base + i];
            let sin_val = self.sin[base + i];
            let x0 = x[i];
            let x1 = x[half + i];
            x[i] = x0 * cos_val - x1 * sin_val;
            x[half + i] = x0 * sin_val + x1 * cos_val;
        }
    }
}

#[cfg(test)]
mod tests;
