//! Quantization primitives for Lattice inference: the Walsh-Hadamard transform, the randomized and
//! block Hadamard rotations used by QuaRot, offline absorption of a rotation into linear-layer
//! weights, and the bounded reader for a Q4 checkpoint's `quantize_index.json` manifest.
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
