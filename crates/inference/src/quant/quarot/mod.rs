//! QuaRot: Hadamard-rotated 4-bit quantization (Ashkboos et al., NeurIPS 2024).
//!
//! See [ADR-044](../../../../docs/adr/ADR-044-quarot-rotated-quantization.md).

pub mod hadamard;
pub mod io;
pub mod plan;
pub mod rotation;

pub use io::{QuarotTensorReader, SourceDType};
