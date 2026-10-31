//! QuaRot: Hadamard-rotated 4-bit quantization (Ashkboos et al., NeurIPS 2024). The Walsh-Hadamard
//! transform, seeded randomized and block Hadamard rotations and their absorption into weight
//! matrices, the rotation plan for Qwen3.5, the RMSNorm fusion and LM-head passes, the converter that
//! writes a rotated Q4 checkpoint, its tensor reader, LoRA adapter rotation, and the f64
//! forward-equivalence check.
//!
//! See [ADR-044](../../../docs/adr/ADR-044-quarot-rotated-quantization.md).

pub mod convert;
pub mod forward_equivalence;
pub mod hadamard;
pub mod io;
pub mod lm_head;
pub mod lora;
pub mod pipeline;
pub mod plan;
#[cfg(test)]
mod r3_reference;
pub mod rmsnorm_fusion;
pub mod rotation;

pub use io::{ArtifactVersion, OnlineArtifactDescriptor, QuarotTensorReader, SourceDType};
pub use plan::{OnlineRotationSpec, OnlineTransformSite, RotationId};
