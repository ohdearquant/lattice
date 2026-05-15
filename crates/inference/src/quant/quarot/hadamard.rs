//! Walsh-Hadamard transform and randomized Hadamard generators.
//!
//! Foundational primitive for QuaRot rotation absorption. All inputs must have
//! a length that is a power of two; the transform is in-place and O(n log n).
//!
//! Two factories are exposed:
//! - [`walsh_hadamard_in_place`] — deterministic structured Hadamard
//! - [`RandomizedHadamard`] — seeded `R = H · D` where `D` is diag(±1) and `H` is
//!   the orthonormal Walsh-Hadamard. See [`RandomizedHadamard`] for the explicit
//!   operation order on apply / apply_inverse.

use crate::error::InferenceError;

/// Apply the unnormalized Walsh-Hadamard transform in-place.
///
/// The transform produced is its own inverse up to scaling by `n`:
/// applying it twice yields `n * x` (so a normalized round-trip divides by `n`).
/// Use [`walsh_hadamard_orthonormal_in_place`] for an isometry (||y|| = ||x||).
///
/// Returns an error if `data.len()` is not a power of two.
pub fn walsh_hadamard_in_place(data: &mut [f32]) -> Result<(), InferenceError> {
    let n = data.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(InferenceError::Inference(format!(
            "walsh_hadamard requires a power-of-two length, got {n}"
        )));
    }

    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }
    Ok(())
}

/// Apply the orthonormal Walsh-Hadamard transform in-place (||y|| = ||x||).
///
/// Identical to [`walsh_hadamard_in_place`] followed by a `1/sqrt(n)` scale.
/// The orthonormal form is its own inverse: `walsh_hadamard_orthonormal_in_place`
/// applied twice returns the original vector (up to floating-point error).
pub fn walsh_hadamard_orthonormal_in_place(data: &mut [f32]) -> Result<(), InferenceError> {
    walsh_hadamard_in_place(data)?;
    let scale = 1.0_f32 / (data.len() as f32).sqrt();
    for v in data.iter_mut() {
        *v *= scale;
    }
    Ok(())
}

/// Deterministic seeded random sign vector — used as the diagonal `D` in the
/// `R = H · D` randomized Hadamard transform.
///
/// Uses a splitmix64 PRNG for reproducibility across platforms without pulling
/// in a `rand` dependency at the call site. Same `seed + n` always produces the
/// same signs.
fn signs_from_seed(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut signs = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        signs.push(if z & 1 == 0 { 1.0 } else { -1.0 });
    }
    signs
}

/// Randomized Hadamard rotation `R = H · D` where `D` is a seeded diagonal of
/// ±1 entries and `H` is the orthonormal Walsh-Hadamard transform. Both `D` and
/// `H` are symmetric and orthogonal, so `R` is orthogonal: `R^T R = (H·D)^T (H·D)
/// = D·H·H·D = D·D = I`.
///
/// [`Self::apply`] computes `R · x = H · (D · x)` — apply `D` first (sign flip
/// per coordinate), then `H` (orthonormal Walsh-Hadamard). [`Self::apply_inverse`]
/// computes `R^T · x = D · (H · x)` — `H` first, then `D`.
///
/// Applying [`Self::apply`] twice does NOT recover the input because
/// `H · D · H · D ≠ I` in general; use [`Self::apply_inverse`] to undo a rotation.
#[derive(Debug, Clone)]
pub struct RandomizedHadamard {
    signs: Vec<f32>,
}

impl RandomizedHadamard {
    /// Build a randomized Hadamard for vectors of length `n` from a seed.
    ///
    /// Returns an error if `n` is zero or not a power of two.
    pub fn new(seed: u64, n: usize) -> Result<Self, InferenceError> {
        if n == 0 || !n.is_power_of_two() {
            return Err(InferenceError::Inference(format!(
                "RandomizedHadamard requires a power-of-two length, got {n}"
            )));
        }
        Ok(Self {
            signs: signs_from_seed(seed, n),
        })
    }

    /// Dimension of the rotation.
    pub fn dim(&self) -> usize {
        self.signs.len()
    }

    /// Apply `R · x = H · (D · x)` to `data` in place: sign-flip by `D`, then
    /// orthonormal Walsh-Hadamard.
    ///
    /// Returns an error if `data.len() != self.dim()`.
    pub fn apply(&self, data: &mut [f32]) -> Result<(), InferenceError> {
        if data.len() != self.signs.len() {
            return Err(InferenceError::Inference(format!(
                "RandomizedHadamard::apply: length mismatch (have {}, want {})",
                data.len(),
                self.signs.len()
            )));
        }
        for (v, s) in data.iter_mut().zip(self.signs.iter()) {
            *v *= s;
        }
        walsh_hadamard_orthonormal_in_place(data)
    }

    /// Apply `R^T · x = D · (H · x)` to `data` in place — the inverse of
    /// [`Self::apply`]. Orthonormal Walsh-Hadamard, then sign-flip by `D`.
    pub fn apply_inverse(&self, data: &mut [f32]) -> Result<(), InferenceError> {
        if data.len() != self.signs.len() {
            return Err(InferenceError::Inference(format!(
                "RandomizedHadamard::apply_inverse: length mismatch (have {}, want {})",
                data.len(),
                self.signs.len()
            )));
        }
        walsh_hadamard_orthonormal_in_place(data)?;
        for (v, s) in data.iter_mut().zip(self.signs.iter()) {
            *v *= s;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "index {i}: {x} vs {y} (delta {})",
                (x - y).abs()
            );
        }
    }

    #[test]
    fn walsh_hadamard_size_1_is_identity() {
        let mut data = [3.5_f32];
        walsh_hadamard_in_place(&mut data).unwrap();
        approx_eq(&data, &[3.5], 1e-6);
    }

    #[test]
    fn walsh_hadamard_size_2_known_result() {
        let mut data = [1.0_f32, 2.0];
        walsh_hadamard_in_place(&mut data).unwrap();
        approx_eq(&data, &[3.0, -1.0], 1e-6);
    }

    #[test]
    fn walsh_hadamard_size_4_known_result() {
        let mut data = [1.0_f32, 0.0, 0.0, 0.0];
        walsh_hadamard_in_place(&mut data).unwrap();
        approx_eq(&data, &[1.0, 1.0, 1.0, 1.0], 1e-6);
    }

    #[test]
    fn walsh_hadamard_double_application_scales_by_n() {
        let n = 16;
        let original: Vec<f32> = (0..n).map(|i| i as f32 - 8.0).collect();
        let mut data = original.clone();
        walsh_hadamard_in_place(&mut data).unwrap();
        walsh_hadamard_in_place(&mut data).unwrap();
        let scaled: Vec<f32> = original.iter().map(|&x| x * n as f32).collect();
        approx_eq(&data, &scaled, 1e-4);
    }

    #[test]
    fn walsh_hadamard_orthonormal_is_isometry() {
        let original: Vec<f32> = (0..32).map(|i| (i as f32 * 0.137).sin()).collect();
        let original_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut data = original.clone();
        walsh_hadamard_orthonormal_in_place(&mut data).unwrap();
        let transformed_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (original_norm - transformed_norm).abs() < 1e-4,
            "||x||={original_norm} vs ||Hx||={transformed_norm}"
        );
    }

    #[test]
    fn walsh_hadamard_orthonormal_is_involution() {
        let original: Vec<f32> = (0..64).map(|i| (i as f32 * 0.71).cos()).collect();
        let mut data = original.clone();
        walsh_hadamard_orthonormal_in_place(&mut data).unwrap();
        walsh_hadamard_orthonormal_in_place(&mut data).unwrap();
        approx_eq(&data, &original, 1e-4);
    }

    #[test]
    fn walsh_hadamard_rejects_non_power_of_two() {
        let mut data = [1.0_f32, 2.0, 3.0];
        assert!(walsh_hadamard_in_place(&mut data).is_err());
    }

    #[test]
    fn walsh_hadamard_rejects_empty() {
        let mut data: [f32; 0] = [];
        assert!(walsh_hadamard_in_place(&mut data).is_err());
    }

    #[test]
    fn randomized_hadamard_is_orthogonal() {
        let r = RandomizedHadamard::new(42, 128).unwrap();
        let original: Vec<f32> = (0..128).map(|i| (i as f32 * 0.13).sin()).collect();
        let original_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut data = original.clone();
        r.apply(&mut data).unwrap();
        let transformed_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (original_norm - transformed_norm).abs() < 1e-3,
            "||x||={original_norm} vs ||Rx||={transformed_norm}"
        );
    }

    #[test]
    fn randomized_hadamard_inverse_round_trips() {
        let r = RandomizedHadamard::new(0xDEAD_BEEF, 256).unwrap();
        let original: Vec<f32> = (0..256).map(|i| (i as f32 * 0.41).cos() + 0.3).collect();
        let mut data = original.clone();
        r.apply(&mut data).unwrap();
        r.apply_inverse(&mut data).unwrap();
        approx_eq(&data, &original, 1e-3);
    }

    #[test]
    fn randomized_hadamard_seed_determinism() {
        let r1 = RandomizedHadamard::new(7, 64).unwrap();
        let r2 = RandomizedHadamard::new(7, 64).unwrap();
        let mut a: Vec<f32> = (0..64).map(|i| i as f32 + 0.5).collect();
        let mut b = a.clone();
        r1.apply(&mut a).unwrap();
        r2.apply(&mut b).unwrap();
        assert_eq!(a, b, "same seed must produce bit-identical output");
    }

    #[test]
    fn randomized_hadamard_seed_differs() {
        let r1 = RandomizedHadamard::new(7, 64).unwrap();
        let r2 = RandomizedHadamard::new(8, 64).unwrap();
        let mut a: Vec<f32> = (0..64).map(|i| i as f32 + 0.5).collect();
        let mut b = a.clone();
        r1.apply(&mut a).unwrap();
        r2.apply(&mut b).unwrap();
        let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(
            diff > 1.0,
            "different seeds should produce different output"
        );
    }

    #[test]
    fn randomized_hadamard_rejects_length_mismatch() {
        let r = RandomizedHadamard::new(1, 16).unwrap();
        let mut data = vec![0.0_f32; 8];
        assert!(r.apply(&mut data).is_err());
    }

    #[test]
    fn randomized_hadamard_outlier_redistribution() {
        let r = RandomizedHadamard::new(123, 1024).unwrap();
        let mut data = vec![0.0_f32; 1024];
        data[42] = 100.0;
        let pre_max = data.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
        r.apply(&mut data).unwrap();
        let post_max = data.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
        assert!(
            post_max < pre_max * 0.5,
            "outlier should be redistributed: pre_max={pre_max}, post_max={post_max}"
        );
    }
}
