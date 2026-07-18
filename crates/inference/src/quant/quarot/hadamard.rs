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

/// The splitmix64 stream increment (`0x9E3779B97F4A7C15`, the golden-ratio
/// constant `2^64 / phi`). Every seed-advancing site in this module MUST use
/// this exact constant: `signs_from_seed` and `derive_block_seed` advance
/// what is conceptually the same splitmix64 stream, and a mismatched
/// increment at one site (even a value that looks equivalent) silently
/// reintroduces the shifted-sign-stream defect `derive_block_seed`'s doc
/// comment describes — adjacent blocks would again share overlapping PRNG
/// states instead of landing at independent stream origins.
const SPLITMIX64_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

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

/// `f64` variant of [`walsh_hadamard_in_place`]. Same butterfly, double precision.
///
/// Used by the absorption step (planned in PR 2 of ADR-044) where rotation
/// math must be computed in `f64` to keep error well below the per-tensor `f16`
/// quantization-scale storage budget. See ADR-044 §Risks.
pub fn walsh_hadamard_f64_in_place(data: &mut [f64]) -> Result<(), InferenceError> {
    let n = data.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(InferenceError::Inference(format!(
            "walsh_hadamard_f64 requires a power-of-two length, got {n}"
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

/// `f64` variant of [`walsh_hadamard_orthonormal_in_place`].
pub fn walsh_hadamard_orthonormal_f64_in_place(data: &mut [f64]) -> Result<(), InferenceError> {
    walsh_hadamard_f64_in_place(data)?;
    let scale = 1.0_f64 / (data.len() as f64).sqrt();
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
    let mut signs = Vec::with_capacity(n);
    push_signs_from_seed(seed, n, &mut signs);
    signs
}

/// Push `n` signs derived from `seed` onto the end of `out`, without
/// allocating a separate `Vec` for the result. Used by [`BlockHadamard::new`]
/// to generate every block's signs directly into one pre-reserved buffer
/// instead of materializing one throwaway `Vec` per block.
fn push_signs_from_seed(seed: u64, n: usize, out: &mut Vec<f32>) {
    let mut state = seed.wrapping_add(SPLITMIX64_GAMMA);
    for _ in 0..n {
        state = state.wrapping_add(SPLITMIX64_GAMMA);
        out.push(if splitmix64_mix(state) & 1 == 0 {
            1.0
        } else {
            -1.0
        });
    }
}

/// The splitmix64 output finalizer: a full-avalanche bijective mix.
fn splitmix64_mix(state: u64) -> u64 {
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Per-block sign seed for [`BlockHadamard`]. The naive `seed + i * GAMMA`
/// derivation is UNSOUND here: `signs_from_seed`'s internal state also
/// advances by the same splitmix64 GAMMA per output, so linearly-spaced
/// seeds make adjacent blocks' sign streams exact one-position-shifted
/// copies sharing `block_size - 1` PRNG states. Running the spaced seed through the
/// full avalanche finalizer first decouples the streams: distinct block
/// indexes land at pseudo-random, non-linearly-related stream origins.
pub(crate) fn derive_block_seed(seed: u64, block_index: usize) -> u64 {
    splitmix64_mix(seed.wrapping_add((block_index as u64).wrapping_mul(SPLITMIX64_GAMMA)))
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

    /// `f64` variant of [`Self::apply`]. The sign vector `D` is stored once
    /// and cast to `f64` per element — same signs, double precision.
    pub fn apply_f64(&self, data: &mut [f64]) -> Result<(), InferenceError> {
        if data.len() != self.signs.len() {
            return Err(InferenceError::Inference(format!(
                "RandomizedHadamard::apply_f64: length mismatch (have {}, want {})",
                data.len(),
                self.signs.len()
            )));
        }
        for (v, s) in data.iter_mut().zip(self.signs.iter()) {
            *v *= f64::from(*s);
        }
        walsh_hadamard_orthonormal_f64_in_place(data)
    }

    /// `f64` variant of [`Self::apply_inverse`].
    pub fn apply_inverse_f64(&self, data: &mut [f64]) -> Result<(), InferenceError> {
        if data.len() != self.signs.len() {
            return Err(InferenceError::Inference(format!(
                "RandomizedHadamard::apply_inverse_f64: length mismatch (have {}, want {})",
                data.len(),
                self.signs.len()
            )));
        }
        walsh_hadamard_orthonormal_f64_in_place(data)?;
        for (v, s) in data.iter_mut().zip(self.signs.iter()) {
            *v *= f64::from(*s);
        }
        Ok(())
    }
}

/// Block-diagonal randomized Hadamard for lengths that are **not** themselves
/// a power of two, but that are evenly divided by a power-of-two block size
/// `b`. Splits `data` into `n / b` contiguous blocks of length `b` and
/// applies an independent [`RandomizedHadamard`] (its own seeded sign
/// vector) to each block.
///
/// This is the standard QuaRot fallback for axes like Qwen3.5-0.8B's
/// `intermediate_size = 3584` (not a power of two: `3584 = 2^9 · 7`) — see
/// issue #703 and QuaRot §4 (<https://arxiv.org/html/2404.00456>). Padding to
/// the next power of two is explicitly rejected by ADR-044 §Risks because it
/// is not orthogonal (the padded zeros still participate in the transform
/// and cannot be exactly un-padded after quantization noise is added); a
/// block-diagonal construction of orthogonal blocks stays exactly orthogonal
/// with zero padding, at the cost of not mixing outliers across block
/// boundaries.
///
/// Per-block signs are seeded via `derive_block_seed`: the block index is
/// spread by the golden-ratio constant and then passed through the full
/// splitmix64 avalanche finalizer. The finalizer step is load-bearing, not
/// cosmetic — spacing seeds linearly by the SAME constant that
/// `signs_from_seed` advances its internal state by makes adjacent blocks'
/// sign streams exact one-position-shifted copies (see `derive_block_seed`
/// and the `adjacent_block_sign_streams_are_not_shifted_copies` regression).
#[derive(Debug, Clone)]
pub struct BlockHadamard {
    block_size: usize,
    num_blocks: usize,
    /// Sign vectors for every block, stored in ONE contiguous buffer: block
    /// `i`'s signs live at `signs[i * block_size .. (i + 1) * block_size]`.
    /// This replaces a
    /// `Vec<RandomizedHadamard>` (one independently heap-allocated struct,
    /// each with its own signs `Vec`, per block) so allocation count is O(1)
    /// regardless of `num_blocks` — see [`BlockHadamard::new`].
    signs: Vec<f32>,
}

/// Upper bound on `n` accepted by [`BlockHadamard::new`].
///
/// `1 << 24` (16,777,216) is generous headroom over any real Qwen3.5/3.6
/// hidden or intermediate dimension this module rotates today: the largest
/// is Qwen3.6-27B's `intermediate_size` (a few thousand) and R3's axis is
/// `num_attention_heads` (tens). Even a future dense model with a
/// multi-hundred-thousand hidden size stays well under this cap, while
/// a single contiguous `signs: Vec<f32>` of `1 << 24` elements (the flat
/// buffer [`BlockHadamard::new`] reserves once, regardless of block count)
/// is a size that fails fast via a fallible reservation instead of
/// exhausting process memory. Anything beyond this is refused as an invalid
/// dimension rather than attempted — see the module's #703 security
/// finding: `n = usize::MAX, block_size = 1` previously passed all
/// validation and then performed an allocation request for `usize::MAX`
/// elements.
///
/// `pub(crate)` (not private) so the artifact-level R3/R4 plan gates in
/// `crate::quant::quarot::plan` can enforce this same bound ahead of
/// [`BlockHadamard::new`] — see
/// `plan::check_block_hadamard_num_blocks_cap`.
pub(crate) const MAX_BLOCK_HADAMARD_LEN: usize = 1 << 24;

/// Upper bound on `num_blocks = n / block_size` accepted by
/// [`BlockHadamard::new`], and the single source of truth for that bound
/// shared with the artifact-level R3/R4 validation gates in
/// `crate::quant::quarot::plan` (`OnlineRotationSpec::r4_dense_mlp` and
/// `OnlineRotationSpec::validate`) — a descriptor certified there must name
/// a `(dim, block_size)` pair this constructor can actually build.
///
/// [`MAX_BLOCK_HADAMARD_LEN`] alone does not bound allocation *size* per
/// block — `BlockHadamard::new(seed, 1 << 24, 1)` passes that cap (n is
/// exactly at the boundary) but requests `num_blocks = 1 << 24`
/// (16,777,216) blocks. Even with the single flat `signs` buffer this cap
/// exists alongside, an unreasonably large `num_blocks` means a `block_size`
/// too small to carry a meaningful rotation axis for any real model. This
/// cap bounds `num_blocks` directly, independent of how small `block_size`
/// is, and is checked before the signs buffer is reserved.
///
/// `4096` is generous headroom over the real geometry this module rotates
/// today: Qwen3.5-0.8B's `intermediate_size = 3584` with `block_size = 128`
/// needs 28 blocks; Qwen3.6-27B's `intermediate_size = 17408` with
/// `block_size = 256` needs 68 blocks (128 needs 136 — see
/// `plan::tests::r4_dense_mlp_accepts_block_size_within_num_blocks_cap`).
/// A future dense model would need a hidden/intermediate dimension in the
/// millions (at the smallest valid `block_size = 1`) to approach this cap,
/// at which point `block_size = 1` would not be a realistic choice for a
/// real rotation axis anyway.
pub(crate) const MAX_BLOCK_HADAMARD_BLOCKS: usize = 4096;

impl BlockHadamard {
    /// Build a block-diagonal randomized Hadamard for vectors of length `n`,
    /// using blocks of size `block_size`.
    ///
    /// Returns an error if `n == 0`, `n` exceeds
    /// `MAX_BLOCK_HADAMARD_LEN`, `num_blocks = n / block_size` exceeds
    /// `MAX_BLOCK_HADAMARD_BLOCKS`, `block_size == 0`, `block_size` is not
    /// a power of two, or `block_size` does not evenly divide `n`. Every
    /// bound check runs before any allocation, so an attacker-controlled or
    /// corrupted `(n, block_size)` pair (e.g. `n = usize::MAX` or
    /// `n = 1 << 24, block_size = 1`) fails closed with [`InferenceError`]
    /// instead of requesting an allocation the process cannot satisfy.
    pub fn new(seed: u64, n: usize, block_size: usize) -> Result<Self, InferenceError> {
        if n == 0 {
            return Err(InferenceError::Inference(
                "BlockHadamard requires a non-zero length".to_string(),
            ));
        }
        if n > MAX_BLOCK_HADAMARD_LEN {
            return Err(InferenceError::Inference(format!(
                "BlockHadamard requires n <= {MAX_BLOCK_HADAMARD_LEN} \
                 (generous headroom over any real model dimension), got {n} \
                 — refusing rather than attempting an unbounded allocation"
            )));
        }
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(InferenceError::Inference(format!(
                "BlockHadamard requires a power-of-two block_size, got {block_size}"
            )));
        }
        if !n.is_multiple_of(block_size) {
            return Err(InferenceError::Inference(format!(
                "BlockHadamard block_size {block_size} does not evenly divide length {n}"
            )));
        }
        let num_blocks = n / block_size;
        if num_blocks > MAX_BLOCK_HADAMARD_BLOCKS {
            return Err(InferenceError::Inference(format!(
                "BlockHadamard requires n / block_size <= {MAX_BLOCK_HADAMARD_BLOCKS} \
                 blocks (generous headroom over any real model geometry), got \
                 {num_blocks} ({n} / {block_size}) — refusing rather than \
                 deriving that many blocks' worth of per-block signs"
            )));
        }
        // Signs for every block are stored in ONE contiguous buffer rather
        // than one `RandomizedHadamard` (and its own signs `Vec`) per block:
        // allocation count is O(1) regardless of `num_blocks`. The reservation
        // itself is fallible (`try_reserve_exact`, not `Vec::with_capacity`),
        // so a cap-boundary `n` that clears every check above (e.g. n =
        // MAX_BLOCK_HADAMARD_LEN with num_blocks = 1) still fails closed with
        // `InferenceError` on an allocation the process cannot satisfy,
        // rather than aborting. Per-block signs are pushed directly into this
        // one buffer (`push_signs_from_seed`) instead of being built in a
        // throwaway per-block `Vec` and copied in, so there is exactly one
        // allocation regardless of `num_blocks`. Per-block sign values are
        // unchanged — `signs_from_seed(derive_block_seed(seed, i),
        // block_size)` is exactly what `RandomizedHadamard::new` computed per
        // block before this refactor (see
        // `block_hadamard_signs_match_per_block_randomized_hadamard`).
        let mut signs: Vec<f32> = Vec::new();
        signs.try_reserve_exact(n).map_err(|e| {
            InferenceError::Inference(format!(
                "BlockHadamard: failed to reserve {n} f32 signs \
                 ({} bytes) — refusing rather than aborting the process: {e}",
                n * std::mem::size_of::<f32>()
            ))
        })?;
        for i in 0..num_blocks {
            push_signs_from_seed(derive_block_seed(seed, i), block_size, &mut signs);
        }
        Ok(Self {
            block_size,
            num_blocks,
            signs,
        })
    }

    /// Total dimension `n = block_size * num_blocks`.
    pub fn dim(&self) -> usize {
        self.block_size * self.num_blocks
    }

    /// Block size `b` shared by every block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of blocks `n / b`.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Apply the block-diagonal rotation to `data` in place: each
    /// contiguous `block_size`-length slice is sign-flipped by its own
    /// segment of the flat `signs` buffer, then rotated by the orthonormal
    /// Walsh-Hadamard transform — the same `R = H · D` operation order as
    /// [`RandomizedHadamard::apply`], applied per block. Every element of
    /// `data` belongs to exactly one block (partition, not overlap).
    pub fn apply(&self, data: &mut [f32]) -> Result<(), InferenceError> {
        if data.len() != self.dim() {
            return Err(InferenceError::Inference(format!(
                "BlockHadamard::apply: length mismatch (have {}, want {})",
                data.len(),
                self.dim()
            )));
        }
        for (chunk, sign_chunk) in data
            .chunks_mut(self.block_size)
            .zip(self.signs.chunks(self.block_size))
        {
            for (v, s) in chunk.iter_mut().zip(sign_chunk.iter()) {
                *v *= s;
            }
            walsh_hadamard_orthonormal_in_place(chunk)?;
        }
        Ok(())
    }

    /// Inverse of [`Self::apply`] — undoes the block-diagonal rotation.
    pub fn apply_inverse(&self, data: &mut [f32]) -> Result<(), InferenceError> {
        if data.len() != self.dim() {
            return Err(InferenceError::Inference(format!(
                "BlockHadamard::apply_inverse: length mismatch (have {}, want {})",
                data.len(),
                self.dim()
            )));
        }
        for (chunk, sign_chunk) in data
            .chunks_mut(self.block_size)
            .zip(self.signs.chunks(self.block_size))
        {
            walsh_hadamard_orthonormal_in_place(chunk)?;
            for (v, s) in chunk.iter_mut().zip(sign_chunk.iter()) {
                *v *= s;
            }
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
    fn adjacent_block_sign_streams_are_not_shifted_copies() {
        // Regression test: with the naive
        // `seed + i * GAMMA` block-seed derivation, `signs_from_seed`'s own
        // per-output `+GAMMA` state advance made block i+1's sign vector
        // EXACTLY block i's shifted by one position (block_size - 1 shared
        // PRNG states). Mutation check: revert `derive_block_seed` to the
        // naive spaced form and the shifted-copy assertion below fails for
        // every (seed, block) pair; the avalanche-mixed derivation must not
        // exhibit the shift alignment for any adjacent pair.
        for seed in [0u64, 1, 42, 0xDEAD_BEEF] {
            for b in [64usize, 128, 256] {
                let s0 = signs_from_seed(derive_block_seed(seed, 0), b);
                let s1 = signs_from_seed(derive_block_seed(seed, 1), b);
                // The old defect: s1[..b-1] == s0[1..] elementwise.
                let shifted_copy = s1[..b - 1]
                    .iter()
                    .zip(&s0[1..])
                    .all(|(a, c)| a.to_bits() == c.to_bits());
                assert!(
                    !shifted_copy,
                    "seed {seed}, block_size {b}: adjacent blocks' sign streams \
                     are one-position-shifted copies (shared PRNG states)"
                );
            }
        }
    }

    #[test]
    fn block_hadamard_rejects_usize_max_length_without_allocating() {
        // #703 security finding: n=usize::MAX with block_size=1 previously
        // passed every existing check and then performed an infallible
        // `Vec::with_capacity`-style allocation for `usize::MAX` blocks,
        // aborting the process. The n-bound check must reject this before
        // any block allocation — this test must never actually allocate.
        let err = BlockHadamard::new(0, usize::MAX, 1).unwrap_err();
        assert!(
            format!("{err}").contains("MAX_BLOCK_HADAMARD_LEN")
                || format!("{err}").to_lowercase().contains("n <="),
            "expected an n-bound rejection, got: {err}"
        );
    }

    #[test]
    #[ignore = "stress test: constructs a single block at the 16M-element \
                cap, deriving a 64 MiB sign buffer — run explicitly with \
                `cargo test -- --ignored` rather than on every default run"]
    fn block_hadamard_accepts_n_at_the_cap_boundary() {
        // The cap itself must still be constructible — block_size ==
        // MAX_BLOCK_HADAMARD_LEN keeps num_blocks at exactly 1. Verifying
        // acceptance at the true boundary means constructing at the true
        // boundary; there is no way to assert this without materializing
        // the full sign buffer, hence `#[ignore]` rather than running it
        // on every default `cargo test`.
        let bh = BlockHadamard::new(0, MAX_BLOCK_HADAMARD_LEN, MAX_BLOCK_HADAMARD_LEN).unwrap();
        assert_eq!(bh.dim(), MAX_BLOCK_HADAMARD_LEN);
        assert_eq!(bh.num_blocks(), 1);
    }

    #[test]
    fn block_hadamard_rejects_n_one_past_the_cap() {
        // cap+1 with block_size=1 must fail on the n-bound, not on
        // divisibility or block_size validity — this exercises the exact
        // boundary the maintainer specified.
        let err = BlockHadamard::new(0, MAX_BLOCK_HADAMARD_LEN + 1, 1).unwrap_err();
        assert!(
            format!("{err}").contains(&(MAX_BLOCK_HADAMARD_LEN).to_string()),
            "expected the cap value in the error, got: {err}"
        );
    }

    #[test]
    fn block_hadamard_rejects_num_blocks_above_cap_without_allocating() {
        // n = 1 << 24, block_size = 1
        // passes the existing n-cap (n is exactly at MAX_BLOCK_HADAMARD_LEN)
        // and every block_size check, but requests num_blocks = 1 << 24
        // (16,777,216) blocks, each needing its own per-block seed
        // derivation. The num_blocks cap must reject this before the
        // per-block signs loop runs, so this test itself never performs
        // that work.
        let err = BlockHadamard::new(0, 1 << 24, 1).unwrap_err();
        assert!(
            format!("{err}").contains(&MAX_BLOCK_HADAMARD_BLOCKS.to_string())
                || format!("{err}").to_lowercase().contains("blocks"),
            "expected a num_blocks-bound rejection, got: {err}"
        );
    }

    #[test]
    fn block_hadamard_accepts_num_blocks_at_the_cap_boundary() {
        // The num_blocks cap itself must still be constructible: exactly
        // MAX_BLOCK_HADAMARD_BLOCKS blocks of block_size 1 stays cheap
        // (small n) even though num_blocks is at the documented bound.
        let bh = BlockHadamard::new(0, MAX_BLOCK_HADAMARD_BLOCKS, 1).unwrap();
        assert_eq!(bh.num_blocks(), MAX_BLOCK_HADAMARD_BLOCKS);
        assert_eq!(bh.dim(), MAX_BLOCK_HADAMARD_BLOCKS);
    }

    #[test]
    fn block_hadamard_accepts_large_valid_n_via_single_fallible_reservation() {
        // A large-but-valid single-block construction (num_blocks = 1, well
        // past any real model's rotation axis) must still succeed via one
        // fallible `try_reserve_exact` reservation, not an infallible
        // `Vec::with_capacity`, and its signs must match the per-block
        // reference exactly — proving the single-buffer path didn't change
        // sign VALUES while making the allocation fallible. (The exact
        // MAX_BLOCK_HADAMARD_LEN boundary is covered, construction-only, by
        // `block_hadamard_accepts_n_at_the_cap_boundary`; this uses a smaller
        // large `n` so the Walsh-Hadamard transform below stays cheap.)
        let n = 1 << 16;
        let bh = BlockHadamard::new(0x5EED, n, n).unwrap();
        assert_eq!(bh.num_blocks(), 1);
        assert_eq!(bh.dim(), n);

        let expected = RandomizedHadamard::new(derive_block_seed(0x5EED, 0), n).unwrap();
        let mut lhs = vec![1.0_f32; n];
        let mut rhs = vec![1.0_f32; n];
        bh.apply(&mut lhs).unwrap();
        expected.apply(&mut rhs).unwrap();
        assert_eq!(
            lhs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            rhs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "large-but-valid single-block construction must match the \
             per-block reference bit-for-bit"
        );
    }

    #[test]
    fn block_hadamard_signs_match_per_block_randomized_hadamard() {
        // The flat contiguous `signs`
        // buffer must produce byte-identical rotations to the prior
        // `Vec<RandomizedHadamard>` layout (one struct + one signs `Vec` per
        // block). Prove it end-to-end rather than by comparing private sign
        // vectors: rotate the same input two ways — the new `BlockHadamard`,
        // and a manual per-block loop that reconstructs exactly what the old
        // layout did (`RandomizedHadamard::new(derive_block_seed(seed, i),
        // block_size)` applied to block `i`'s slice) — and assert the outputs
        // agree bit-for-bit. This covers both the per-block sign VALUES
        // (`signs_from_seed(derive_block_seed(...))`) and the `D`-then-`H`
        // operation order, so a drift in either would fail the assertion.
        for (seed, block_size, num_blocks) in [
            (0x51ED_u64, 4usize, 5usize),
            (42, 8, 3),
            (0xDEAD_BEEF, 2, 7),
        ] {
            let n = block_size * num_blocks;
            let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();

            // New layout (flat contiguous signs).
            let bh = BlockHadamard::new(seed, n, block_size).unwrap();
            let mut new_out = input.clone();
            bh.apply(&mut new_out).unwrap();

            // Old layout, reconstructed: one RandomizedHadamard per block,
            // same `derive_block_seed(seed, i)`, applied to that block's slice.
            let mut old_out = input.clone();
            for (i, chunk) in old_out.chunks_mut(block_size).enumerate() {
                let rh = RandomizedHadamard::new(derive_block_seed(seed, i), block_size).unwrap();
                rh.apply(chunk).unwrap();
            }

            assert_eq!(
                new_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                old_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "seed {seed}, block_size {block_size}, num_blocks {num_blocks}: \
                 flat-signs BlockHadamard must match the per-block \
                 RandomizedHadamard layout bit-for-bit"
            );

            // The refactored inverse must still round-trip the forward rotation.
            let mut round = new_out.clone();
            bh.apply_inverse(&mut round).unwrap();
            approx_eq(&round, &input, 1e-5);
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

    fn approx_eq_f64(a: &[f64], b: &[f64], tol: f64) {
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
    fn walsh_hadamard_f64_size_2_known_result() {
        let mut data = [1.0_f64, 2.0];
        walsh_hadamard_f64_in_place(&mut data).unwrap();
        approx_eq_f64(&data, &[3.0, -1.0], 1e-12);
    }

    #[test]
    fn walsh_hadamard_f64_orthonormal_is_isometry() {
        let original: Vec<f64> = (0..32).map(|i| (i as f64 * 0.137).sin()).collect();
        let original_norm: f64 = original.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut data = original.clone();
        walsh_hadamard_orthonormal_f64_in_place(&mut data).unwrap();
        let transformed_norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (original_norm - transformed_norm).abs() < 1e-12,
            "||x||={original_norm} vs ||Hx||={transformed_norm}"
        );
    }

    #[test]
    fn walsh_hadamard_f64_orthonormal_is_involution() {
        let original: Vec<f64> = (0..64).map(|i| (i as f64 * 0.71).cos()).collect();
        let mut data = original.clone();
        walsh_hadamard_orthonormal_f64_in_place(&mut data).unwrap();
        walsh_hadamard_orthonormal_f64_in_place(&mut data).unwrap();
        approx_eq_f64(&data, &original, 1e-12);
    }

    #[test]
    fn randomized_hadamard_f64_inverse_round_trips() {
        let r = RandomizedHadamard::new(0xDEAD_BEEF, 256).unwrap();
        let original: Vec<f64> = (0..256).map(|i| (i as f64 * 0.41).cos() + 0.3).collect();
        let mut data = original.clone();
        r.apply_f64(&mut data).unwrap();
        r.apply_inverse_f64(&mut data).unwrap();
        approx_eq_f64(&data, &original, 1e-12);
    }

    #[test]
    fn randomized_hadamard_f32_f64_agree_in_precision() {
        let r = RandomizedHadamard::new(7, 256).unwrap();
        let mut f32_data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.13).sin()).collect();
        let mut f64_data: Vec<f64> = f32_data.iter().map(|&x| x as f64).collect();
        r.apply(&mut f32_data).unwrap();
        r.apply_f64(&mut f64_data).unwrap();
        for (i, (a, b)) in f32_data.iter().zip(f64_data.iter()).enumerate() {
            let delta = (*a as f64 - b).abs();
            assert!(delta < 1e-5, "index {i}: f32={a} vs f64={b}, delta={delta}");
        }
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

    // --- BlockHadamard ---

    fn synthetic_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (state >> 11) as u32;
                (bits as f32 / u32::MAX as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn block_hadamard_round_trips() {
        for &(n, b) in &[(3584usize, 64usize), (3584, 128), (3584, 256), (17408, 256)] {
            let bh = BlockHadamard::new(0xB10C_5EED, n, b).unwrap();
            let original = synthetic_vec(n, 7);
            let mut data = original.clone();
            bh.apply(&mut data).unwrap();
            bh.apply_inverse(&mut data).unwrap();
            approx_eq(&data, &original, 1e-3);
        }
    }

    #[test]
    fn block_hadamard_preserves_norm() {
        let bh = BlockHadamard::new(42, 3584, 128).unwrap();
        let original = synthetic_vec(3584, 11);
        let original_norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut data = original.clone();
        bh.apply(&mut data).unwrap();
        let transformed_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (original_norm - transformed_norm).abs() < 1e-2,
            "||x||={original_norm} vs ||Rx||={transformed_norm}"
        );
    }

    #[test]
    fn block_hadamard_exact_partition_coverage() {
        // Every element belongs to exactly one block: perturbing element i
        // in isolation must only change output elements within i's own
        // block, for every valid (n, b) pair the design doc names.
        for &(n, b) in &[
            (3584usize, 64usize),
            (3584, 128),
            (3584, 256),
            (17408, 64),
            (17408, 128),
            (17408, 256),
        ] {
            assert_eq!(n % b, 0, "test precondition: b must divide n");
            let bh = BlockHadamard::new(0xC0FF_EE00, n, b).unwrap();
            assert_eq!(bh.num_blocks(), n / b);
            assert_eq!(bh.dim(), n);

            let base = synthetic_vec(n, 3);
            let mut perturbed = base.clone();
            let target = n / 2 + 3; // arbitrary interior element
            perturbed[target] += 5.0;

            let mut out_base = base.clone();
            bh.apply(&mut out_base).unwrap();
            let mut out_perturbed = perturbed.clone();
            bh.apply(&mut out_perturbed).unwrap();

            let block_start = (target / b) * b;
            let block_end = block_start + b;
            for i in 0..n {
                let changed = (out_base[i] - out_perturbed[i]).abs() > 1e-6;
                if i >= block_start && i < block_end {
                    // inside target's block: allowed (not required) to change
                } else {
                    assert!(
                        !changed,
                        "element {i} outside block [{block_start},{block_end}) \
                         changed when only element {target} was perturbed \
                         (n={n}, b={b}) — partition coverage violated"
                    );
                }
            }
        }
    }

    #[test]
    fn block_hadamard_rejects_block_size_not_dividing_n() {
        // 1024 is a power of two but 3584 (= 2^9 * 7) is not a multiple of
        // it (3584 / 1024 = 3.5) — this exercises the divisibility check on
        // its own. A non-power-of-two block_size like 100 would be caught
        // by the earlier power-of-two check regardless of divisibility,
        // making that check redundant and leaving this test green even if
        // the divisibility check were removed.
        assert!(BlockHadamard::new(1, 3584, 1024).is_err());
    }

    #[test]
    fn block_hadamard_rejects_non_power_of_two_block_size() {
        // 3584 is divisible by 3584/7=512... use a divisor that isn't a
        // power of two: 3584 = 2^9 * 7, so block_size=7 divides evenly but
        // is not a power of two.
        assert!(BlockHadamard::new(1, 3584, 7).is_err());
    }

    #[test]
    fn block_hadamard_rejects_zero_length() {
        assert!(BlockHadamard::new(1, 0, 64).is_err());
    }

    #[test]
    fn block_hadamard_rejects_zero_block_size() {
        assert!(BlockHadamard::new(1, 3584, 0).is_err());
    }

    #[test]
    fn block_hadamard_rejects_length_mismatch_on_apply() {
        let bh = BlockHadamard::new(1, 3584, 128).unwrap();
        let mut data = vec![0.0_f32; 100];
        assert!(bh.apply(&mut data).is_err());
    }

    /// A decoder built from the wrong seed must not reproduce the original
    /// vector, and a decoder built from the matching seed must reproduce it
    /// exactly — proving the round trip is sensitive to which seed derived
    /// the block signs, not just whether a seed was supplied at all.
    #[test]
    fn block_hadamard_mutation_sensitive_round_trip() {
        let n = 3584;
        let b = 128;
        let good_seed = 0xB10C_5EED;
        let corrupted_seed = 0xB10C_5EED ^ 0x1; // flips the derived signs

        let original = synthetic_vec(n, 7);

        // A decoder seeded differently from the encoder must diverge from
        // the original vector on decode.
        let encoder = BlockHadamard::new(good_seed, n, b).unwrap();
        let corrupted_decoder = BlockHadamard::new(corrupted_seed, n, b).unwrap();
        let mut data = original.clone();
        encoder.apply(&mut data).unwrap();
        corrupted_decoder.apply_inverse(&mut data).unwrap();
        let corrupted_delta: f32 = data
            .iter()
            .zip(original.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            corrupted_delta > 1e-2,
            "mismatched-seed round trip should diverge from the original, got delta {corrupted_delta}"
        );

        // A decoder seeded to match the encoder reproduces the original
        // exactly, confirming the divergence above traces to the seed
        // mismatch and not some other defect.
        let restored_decoder = BlockHadamard::new(good_seed, n, b).unwrap();
        let mut data_restored = original.clone();
        encoder.apply(&mut data_restored).unwrap();
        restored_decoder.apply_inverse(&mut data_restored).unwrap();
        approx_eq(&data_restored, &original, 1e-3);
    }
}
