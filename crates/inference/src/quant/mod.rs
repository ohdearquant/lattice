//! Quantization primitives.
//!
//! Built on top of [`crate::weights::q4_weights`]. Pre-quantization transforms
//! that improve quantizability (rotation, smoothing) live here.

pub mod quarot;
