//! Quantization primitives.
//!
//! Pre-quantization transforms that improve quantizability (rotation,
//! smoothing) live here. Currently:
//! - `quarot` — Walsh-Hadamard transform + randomized Hadamard rotation
//!   primitives, wired into [`crate::weights::q4_weights`] via
//!   `quarot::convert::convert_quarot_qwen35` (see ADR-044): the pipeline
//!   quantizes rotated tensors with `quantize_f64_to_q4` and writes them
//!   with `save_q4_file`, driven by the `bin/quantize_quarot` CLI.

pub mod q4_manifest;
pub mod quarot;
