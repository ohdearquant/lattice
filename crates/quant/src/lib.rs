//! Quantization for Lattice inference: the Walsh-Hadamard transform, the randomized and block
//! Hadamard rotations used by QuaRot, offline absorption of a rotation into linear-layer weights,
//! the offline QuaRot conversion of a Qwen3.5 checkpoint (rotation plan, RMSNorm fusion, LM-head
//! and LoRA handling, the source-tensor reader and the online-rotation artifact descriptor), and
//! the bounded reader for a Q4 checkpoint's `quantize_index.json` manifest.
//!
//! `lattice-inference` re-exports the supported items at their original
//! `lattice_inference::quant::q4_manifest` and `lattice_inference::quant::quarot` paths.

// Rotation kernels index several arrays at once; the same allowances as the inference crate.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

mod error {
    pub use lattice_inference_core::InferenceError;
}

pub mod q4_manifest;
pub mod quarot;

/// Quantization glue used by the `lattice-inference` facade.
///
/// Not part of the supported API. The facade calls these free functions by their full path; they are
/// not re-exported by that facade.
#[doc(hidden)]
pub mod __internal {
    use crate::quarot::plan::RotationTarget;
    use std::path::Path;

    /// The QuaRot rotation seed recorded in a Q4 checkpoint's `quantize_index.json`, validated against
    /// the target; `Ok(None)` when the checkpoint has no manifest.
    /// A legacy seed allows an object-form manifest to defer seed recovery to its caller.
    pub fn read_quarot_seed_from_index(
        q4_dir: &Path,
        target: &RotationTarget,
        legacy_seed: Option<u64>,
    ) -> Result<Option<u64>, String> {
        crate::quarot::convert::read_quarot_seed_from_index(q4_dir, target, legacy_seed)
    }
}
