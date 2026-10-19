//! Metal-facing compatibility paths for the thread-affine serving worker.
//!
//! The worker host and its prompt-window helpers live in
//! [`crate::serving_worker`]. This module retains the vision exports,
//! measurement support, and the original paths used by the serving binaries
//! and their tests.

#[cfg(test)]
use crate::forward::metal_qwen35::ChatMessage;
#[cfg(test)]
use crate::forward::metal_qwen35::MetalQwen35State;
#[cfg(test)]
use crate::forward::metal_qwen35::format_chat_template;
#[cfg(any(test, feature = "bench-internals"))]
use crate::generation::GenerateConfig;
#[cfg(test)]
use crate::generation::GenerateOutput;
#[cfg(any(test, feature = "bench-internals"))]
use crate::serve::ApiError;
#[cfg(any(test, feature = "bench-internals"))]
use crate::serve::lora::ResidencyLimits;
#[cfg(test)]
use crate::serve::lora::{AdapterControlError, AdapterControlResult};
#[cfg(feature = "bench-internals")]
use crate::serve::prepare::PreparationHandle;
#[cfg(any(test, feature = "bench-internals"))]
use crate::serving_factory::ServingFactory;
#[cfg(test)]
use crate::tokenizer::Tokenizer as _;
#[cfg(test)]
use crate::tokenizer::bpe::BpeTokenizer;
#[cfg(test)]
use std::sync::Mutex;
#[cfg(test)]
use std::time::Instant;
#[cfg(test)]
use tokio::sync::{Semaphore, mpsc, oneshot};

#[cfg(test)]
use crate::model::qwen35_config::{Qwen35Config, VisionModelConfig};
pub use crate::serving_provider::providers::qwen::vision::VisionRuntime;
#[cfg(test)]
use crate::serving_provider::providers::qwen::vision::{
    JobRoute, VisionRequestBuild, build_vision_request, classify_job,
};
#[cfg(test)]
use crate::serving_provider::providers::qwen::vision::{
    VisionRuntimeLoad, build_vision_prompt_ids, build_vision_prompt_text, tokenize_text,
};
#[cfg(test)]
use crate::serving_runtime_contract::WorkerFailure;
pub use crate::serving_runtime_contract::{ContextWindowPolicy, WorkerMetadata};
#[cfg(test)]
use std::path::PathBuf;

pub use crate::serving_worker::AdapterCommand;
pub use crate::serving_worker::AffineWorker as MetalWorker;
pub use crate::serving_worker::ControlRequest;
pub use crate::serving_worker::DEFAULT_MAX_PENDING_JOBS;
pub use crate::serving_worker::StartupError;
pub use crate::serving_worker::WorkerClient as MetalWorkerClient;
pub use crate::serving_worker::WorkerEvent;
pub use crate::serving_worker::WorkerJob;
pub use crate::serving_worker::WorkerMessage;
pub use crate::serving_worker::WorkerOwner as MetalWorkerOwner;

#[cfg(any(test, feature = "test-utils"))]
pub use crate::serving_worker::{
    spawn_fake, spawn_fake_with_cap, spawn_fake_with_vision, test_client_and_jobs,
    test_client_and_jobs_with_adapters, test_client_and_jobs_with_cap,
};

#[cfg(test)]
use crate::serving_worker::{
    WorkerShutdown, lock_unpoisoned, run_guarded_worker, run_worker_loop,
    run_worker_loop_with_control,
};

pub(crate) use crate::serving_worker::{
    check_prompt_fits_window, render_text_prompt_within_window, serving_tokenizer,
};

/// Measurement access to worker internals, for the `bench_serve_prepare`
/// example only. Not a stable API.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod bench_support {
    use super::{ApiError, ContextWindowPolicy, GenerateConfig};

    /// Start a measurement worker whose own thread holds the GPU lock through
    /// model construction, every queued request, and runtime destruction.
    /// The lock remains held if the caller's bounded shutdown detaches it.
    /// The returned load duration excludes waiting for the measurement lock.
    pub fn spawn_with_measurement_lock(
        factory: super::ServingFactory,
        max_pending: usize,
        residency_limits: super::ResidencyLimits,
    ) -> Result<
        (
            super::MetalWorkerOwner,
            super::MetalWorkerClient,
            super::WorkerMetadata,
            super::PreparationHandle,
            std::time::Duration,
        ),
        super::StartupError,
    > {
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (owner, client, metadata, preparation) =
            crate::serving_worker::AffineWorker::spawn_with_guard(
                factory,
                max_pending,
                residency_limits,
                move || {
                    let guard = crate::measurement::gpu_test_lock();
                    let _ = started_tx.send(std::time::Instant::now());
                    guard
                },
            )?;
        let started = started_rx
            .recv()
            .map_err(|_| super::StartupError::ThreadExited)?;
        Ok((owner, client, metadata, preparation, started.elapsed()))
    }

    /// The worker's context-window admission check, exactly as the Metal
    /// worker applies it to a rendered, tokenized prompt before generation.
    pub fn check_prompt_fits_window(
        policy: ContextWindowPolicy,
        model_max_context: usize,
        prompt_len: usize,
        cfg: &GenerateConfig,
    ) -> Result<(), ApiError> {
        super::check_prompt_fits_window(policy, model_max_context, prompt_len, cfg)
    }
}

#[cfg(test)]
mod tests;
