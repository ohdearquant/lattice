//! QuaRot rotation primitives (Ashkboos et al., NeurIPS 2024): the Walsh-Hadamard transform, seeded
//! randomized and block Hadamard rotations, and their absorption into weight matrices.
//!
//! See [ADR-044](../../../docs/adr/ADR-044-quarot-rotated-quantization.md).

pub mod hadamard;
pub mod rotation;
