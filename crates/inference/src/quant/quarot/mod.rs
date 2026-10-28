//! QuaRot: Hadamard-rotated 4-bit quantization (Ashkboos et al., NeurIPS 2024), re-exported from
//! `lattice-quant`.
//!
//! See [ADR-044](../../../../docs/adr/ADR-044-quarot-rotated-quantization.md).
//!
//! **Compatibility path.** This module keeps the pre-split `lattice-inference` paths compiling; the
//! code it names lives in the crates that own it. These low-level paths are not a stable API and
//! will be removed at 1.0.

pub mod convert;
pub mod forward_equivalence;
pub mod hadamard;
pub mod io;
pub mod lm_head;
pub mod lora;
pub mod pipeline;
pub mod plan;
pub mod rmsnorm_fusion;
pub mod rotation;

pub use io::{ArtifactVersion, OnlineArtifactDescriptor, QuarotTensorReader, SourceDType};
pub use plan::{OnlineRotationSpec, OnlineTransformSite, RotationId};
