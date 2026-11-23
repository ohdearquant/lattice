//! The Inkling 2 looped mixture-of-experts text decoder, run on the CPU (ADR-106).
//!
//! [`Inkling2Model`] runs the text decoder from a validated
//! `lattice_arch::inkling2_config::Inkling2TextConfig` and the weights in [`Inkling2Weights`],
//! which [`Inkling2Weights::load`] reads from a checkpoint in the published layout and
//! [`Inkling2Model::open`] reads from a checkpoint directory.
//! Stored layers `core_start..core_end` run `num_passes` times per token. Every pass after the
//! first restarts from the state before the loop and reads messages from the previous pass, and
//! each pass keeps its own key, value and convolution caches in [`Inkling2Cache`].
//!
//! Experts stay packed (NVFP4 or MXFP8) and are multiplied by the packed matrix-vector kernels
//! of `lattice-weights`. Other matrices keep their stored element type and are widened to f32
//! as they are read. Arithmetic is f32 throughout.

mod attention;
mod cache;
mod conv;
mod loader;
mod mixture;
mod model;
mod moe;
mod ops;
mod tensor;
mod weights;

pub use cache::Inkling2Cache;
pub use loader::{ExpertFormat, Inkling2Checkpoint, TensorSpec, expected_text_tensors};
pub use model::Inkling2Model;
pub use tensor::{DenseMatrix, ExpertBank};
pub use weights::{
    AttentionWeights, Inkling2Weights, KvConvolution, LayerWeights, LoopGates, MessageGateWeights,
    MoeWeights,
};

#[cfg(test)]
mod loader_tests;
#[cfg(test)]
mod tests;
