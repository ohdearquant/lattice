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
    /// `x` has shape [head_dim] and is modified in-place.
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
mod tests {
    use super::*;

    #[test]
    fn test_rope_position_zero_is_identity() {
        let table = RopeTable::new(4, 16, 10000.0);
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let original = x.clone();
        table.apply(&mut x, 0);
        // At position 0, angle=0 for all dims, so cos=1, sin=0 → identity.
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "position 0 should be identity");
        }
    }

    #[test]
    fn test_rope_preserves_norm() {
        let table = RopeTable::new(128, 512, 1_000_000.0);
        let mut x: Vec<f32> = (0..128).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let norm_before: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        table.apply(&mut x, 42);
        let norm_after: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "RoPE should preserve vector norm"
        );
    }

    #[test]
    fn test_rope_different_positions_differ() {
        let table = RopeTable::new(4, 16, 10000.0);
        let base = vec![1.0, 0.0, 0.0, 1.0];
        let mut x1 = base.clone();
        let mut x2 = base.clone();
        table.apply(&mut x1, 1);
        table.apply(&mut x2, 5);
        assert_ne!(
            x1, x2,
            "different positions must produce different rotations"
        );
    }

    /// Verify f32 RoPE table precision at high positions (up to 256K).
    ///
    /// The table is computed in f64 and cast to f32. At position 262144 with
    /// theta=1e6 and head_dim=128, the highest-frequency dimension has
    /// angle = 262144.0 which requires accurate argument reduction in sin/cos.
    /// We verify the f32 table values stay within 1e-6 of the f64 ground truth.
    #[test]
    fn test_rope_precision_at_high_positions() {
        let head_dim = 128;
        let theta = 1_000_000.0f64;
        let half_dim = head_dim / 2;

        // Test positions spanning the full 256K context window
        let test_positions: &[usize] = &[0, 1, 100, 1_000, 10_000, 100_000, 262_143];
        let max_pos = *test_positions.last().unwrap() + 1;
        let table = RopeTable::new(head_dim, max_pos, theta);

        let mut max_cos_err: f64 = 0.0;
        let mut max_sin_err: f64 = 0.0;
        let mut worst_pos = 0usize;
        let mut worst_dim = 0usize;

        for &pos in test_positions {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
                let angle = pos as f64 * freq;
                let cos_f64 = angle.cos();
                let sin_f64 = angle.sin();

                let idx = pos * half_dim + i;
                let cos_f32 = table.cos_at(idx) as f64;
                let sin_f32 = table.sin_at(idx) as f64;

                let cos_err = (cos_f32 - cos_f64).abs();
                let sin_err = (sin_f32 - sin_f64).abs();

                if cos_err > max_cos_err {
                    max_cos_err = cos_err;
                    worst_pos = pos;
                    worst_dim = i;
                }
                if sin_err > max_sin_err {
                    max_sin_err = sin_err;
                }

                // f32 has ~7 decimal digits; after f64→f32 cast, error should be < 1e-6
                assert!(
                    cos_err < 1e-6,
                    "cos precision lost at pos={pos} dim={i}: f32={cos_f32} f64={cos_f64} err={cos_err}"
                );
                assert!(
                    sin_err < 1e-6,
                    "sin precision lost at pos={pos} dim={i}: f32={sin_f32} f64={sin_f64} err={sin_err}"
                );
            }
        }

        eprintln!(
            "[rope_precision] max_cos_err={max_cos_err:.2e} max_sin_err={max_sin_err:.2e} \
             worst_at pos={worst_pos} dim={worst_dim}"
        );
    }

    /// Verify RoPE preserves orthogonality at high positions.
    ///
    /// Two orthogonal input vectors should remain orthogonal after rotation
    /// (RoPE is a unitary transform). At high positions, accumulated f32 error
    /// could break this. We check dot(rotated_a, rotated_b) ≈ 0.
    #[test]
    fn test_rope_orthogonality_at_high_positions() {
        let head_dim = 128;
        let theta = 1_000_000.0f64;
        let max_pos = 262_144;
        let table = RopeTable::new(head_dim, max_pos, theta);

        // Two orthogonal vectors: e_0 and e_1
        let mut a = vec![0.0f32; head_dim];
        let mut b = vec![0.0f32; head_dim];
        a[0] = 1.0;
        b[1] = 1.0;

        for &pos in &[0usize, 1000, 100_000, 262_143] {
            let mut ra = a.clone();
            let mut rb = b.clone();
            table.apply(&mut ra, pos);
            table.apply(&mut rb, pos);

            let dot: f64 = ra
                .iter()
                .zip(rb.iter())
                .map(|(x, y)| *x as f64 * *y as f64)
                .sum();
            assert!(
                dot.abs() < 1e-5,
                "orthogonality broken at pos={pos}: dot={dot}"
            );
        }
    }
}
