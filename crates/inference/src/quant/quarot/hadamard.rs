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
mod tests;
