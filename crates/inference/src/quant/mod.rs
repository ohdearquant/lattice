//! Quantization primitives.
//!
//! Pre-quantization transforms that improve quantizability (rotation,
//! smoothing) live here. Currently:
//! - [`quarot`] — Walsh-Hadamard transform + randomized Hadamard rotation
//!   primitives. No quantization integration yet; future PRs will wire these
//!   into [`crate::weights::q4_weights`] (see ADR-044).

pub mod quarot;
