//! `lattice-fann` provides small dense neural networks for CPU-first inference.
//! It targets roughly 100,000-parameter classifiers with a sub-5 ms CPU budget.
//!
//! Use [`NetworkBuilder`] to assemble layers, [`Network`] to run them, and
//! [`BackpropTrainer`] to train them. The CPU forward path reuses activation
//! buffers; `parallel`, `serde`, and `gpu` add batch inference, persistence, and
//! optional GPU acceleration. `simd` accelerates supported CPU kernels.
//!
//! See `docs/design.md` for the crate architecture and `docs/network.md` for
//! the network, activation, and binary-format reference.

#![warn(missing_docs)]

mod activation;
mod error;
mod layer;
pub mod lora;
mod network;
pub mod training;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use activation::Activation;
pub use error::{FannError, FannResult};
pub use layer::Layer;
pub use lora::LoraDescriptor;
pub use network::{Network, NetworkBuilder};
pub use training::{
    BackpropTrainer, GradientGuardStrategy, Trainer, TrainingConfig, TrainingResult,
};

#[cfg(feature = "gpu")]
pub use gpu::{GpuContext, GpuNetwork};

/// Prelude for common imports
pub mod prelude {
    pub use crate::{
        Activation, BackpropTrainer, FannError, FannResult, GradientGuardStrategy, Layer, Network,
        NetworkBuilder, Trainer, TrainingConfig, TrainingResult,
    };

    #[cfg(feature = "gpu")]
    pub use crate::gpu::{GpuContext, GpuNetwork};
}

#[cfg(test)]
mod tests;
