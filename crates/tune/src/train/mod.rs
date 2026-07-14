//! Training orchestration: the training loop, optimizer configuration, checkpointing,
//! early stopping, and GPU-accelerated training behind the `gpu` feature.
//!
//! See [`docs/design.md`](https://github.com/ohdearquant/lattice/blob/main/crates/tune/docs/design.md)
//! for the pipeline diagram and CPU/GPU training walkthroughs.
//!
//! # GPU weight-update gap (load-bearing)
//!
//! `GpuTrainer::train_batch` returns `Err(TuneError::Training(_))` for **every** optimizer
//! choice (Adam, AdamW, SGD-momentum, plain SGD, RMSprop): the GPU-shader optimizer dispatch
//! has no buffer bindings wired to the network's weight/gradient buffers, and the CPU-side
//! plain-SGD arm has neither real gradient plumbing nor a mutable weight write-back path.
//! `GpuTrainer::validate` is a forward-only, GPU-accelerated path unaffected by this gap.
//! See <https://github.com/ohdearquant/lattice/issues/797>; this note goes away once the
//! wiring lands.

mod config;
mod jit;
mod r#loop;

#[cfg(feature = "gpu")]
mod gpu;

pub use config::{
    EarlyStopping, LRSchedule, MAX_BATCH_SIZE, Optimizer, OptimizerConfig, RegularizationConfig,
    TrainingConfig,
};
pub use jit::{JitAdapter, JitConfig, JitResult, JitStrategy, freeze};
pub use r#loop::{
    Checkpoint, EpochMetrics, LoggingCallback, NoOpCallback, TrainingCallback, TrainingLoop,
    TrainingMetrics, TrainingState,
};

#[cfg(feature = "gpu")]
pub use gpu::{GpuTrainer, GpuTrainerBuilder};
