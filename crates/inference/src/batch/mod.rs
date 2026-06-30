//! Continuous batching engine for multi-sequence inference (ADR-048).
//!
//! This module implements iteration-level scheduling with chunked prefill
//! interleaved with decode steps. On Apple Silicon unified memory, the
//! "disaggregated" prefill/decode separation is achieved via chunked prefill
//! without any data transfer — KV pages and GDN state live in the same
//! physical memory throughout.
//!
//! # Module structure
//!
//! - [`config`] — [`BatchConfig`]: resource limits and scheduling parameters.
//! - [`sequence`] — [`Sequence`], [`SequenceManager`], [`SeqId`],
//!   [`SequenceState`], [`FinishReason`], [`AdapterKey`]: per-sequence state.
//! - [`scheduler`] — [`Scheduler`] trait, [`FifoScheduler`],
//!   [`SchedulerDecision`]: iteration-level batch selection.
//! - [`worker`] — [`BatchWorker`], [`GdnStatePool`],
//!   [`InferenceRequest`], [`InferenceToken`]: the continuous batching loop.
//!
//! # Usage sketch
//!
//! ```rust,ignore
//! use lattice_inference::batch::{BatchConfig, BatchWorker, InferenceRequest};
//! use lattice_inference::sampling::SamplingConfig;
//! use lattice_inference::kv_cache::{EvictionPolicy, PagedKVCacheConfig};
//! use lattice_inference::batch::worker::PagedKVCacheConfigExt;
//!
//! let kv_config = PagedKVCacheConfig { /* ... */ };
//! let mut worker = BatchWorker::try_new(
//!     BatchConfig::default(),
//!     kv_config,
//!     s_floats_per_slot,
//!     conv_floats_per_slot,
//!     Some(eos_token_id),
//! );
//!
//! let id = worker.submit(InferenceRequest {
//!     prompt_ids: vec![1, 2, 3],
//!     sampling: SamplingConfig::greedy(),
//!     lora_adapter: None,
//!     max_new_tokens: 64,
//! }).expect("valid request");
//!
//! while !worker.is_idle() {
//!     let tokens = worker.step(|input, gdn_pool| {
//!         // Run your model forward pass here.
//!         // input.token_ids: slice to process
//!         // input.start_pos: position for RoPE
//!         // input.gdn_slot: index into gdn_pool
//!         vec![0.0f32; vocab_size]
//!     });
//!     for token in tokens {
//!         println!("seq {} token {} finished={}", token.seq_id, token.token_id, token.finished);
//!     }
//! }
//! ```

pub mod config;
pub mod scheduler;
pub mod sequence;
pub mod worker;

// Convenience re-exports for the most common types.
pub use config::BatchConfig;
pub use scheduler::{FifoScheduler, Scheduler, SchedulerDecision};
pub use sequence::{AdapterKey, FinishReason, SeqId, Sequence, SequenceManager, SequenceState};
pub use worker::{
    BatchStepInput, BatchStepOutput, BatchWorker, GdnStatePool, InferenceRequest, InferenceToken,
    PagedKVCacheConfigExt,
};
