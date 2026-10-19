//! Metal-facing compatibility paths for the thread-affine serving worker.
//!
//! The worker host lives in [`crate::serving_worker`]. This module retains
//! the Metal prompt helpers, vision exports, measurement support, and the
//! original paths used by the serving binaries and their tests.

use crate::forward::metal_qwen35::ChatMessage;
#[cfg(test)]
use crate::forward::metal_qwen35::MetalQwen35State;
use crate::forward::metal_qwen35::format_chat_template;
use crate::generation::GenerateConfig;
#[cfg(test)]
use crate::generation::GenerateOutput;
use crate::serve::ApiError;
#[cfg(any(test, feature = "bench-internals"))]
use crate::serve::lora::ResidencyLimits;
#[cfg(test)]
use crate::serve::lora::{AdapterControlError, AdapterControlResult};
#[cfg(feature = "bench-internals")]
use crate::serve::prepare::PreparationHandle;
use crate::serve::prompt_adapter::apply_qwen_thinking_mode;
#[cfg(any(test, feature = "bench-internals"))]
use crate::serving_factory::ServingFactory;
use crate::tokenizer::Tokenizer as _;
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

/// Adapter-selected KV-window invariant for Metal jobs (#656).
/// `lattice_serve` only knows the rendered prompt length on this worker, so
/// its full-window check runs here. `lattice` already checks the rendered
/// prompt in its HTTP preflight; repeating that adapter's exact formula here
/// prevents the shared worker from tightening its accepted boundary.
///
/// `lattice_serve.rs` keeps its pre-existing full-decode formula, including
/// reasoning tokens and one delimiter slot. `lattice.rs` keeps its
/// pre-existing HTTP formula, which accepts
/// `prompt_tokens + max_tokens == max_context`.
pub(crate) fn check_prompt_fits_window(
    policy: ContextWindowPolicy,
    model_max_context: usize,
    prompt_len: usize,
    cfg: &GenerateConfig,
) -> Result<(), ApiError> {
    // `lattice.rs`'s pre-refactor `check_context_window` also rejected an
    // empty rendered prompt (`prompt_token_count == 0`) as part of the same
    // predicate, independent of the window arithmetic; preserve that
    // conjunct for the policy that reproduces it.
    if matches!(policy, ContextWindowPolicy::PromptAndMaxTokens) && prompt_len == 0 {
        return Err(ApiError::BadRequest {
            message: format!(
                "prompt (0 tokens) plus max_tokens ({max_tokens}) exceeds model \
                 context window ({model_max_context})",
                max_tokens = cfg.max_new_tokens,
            ),
            code: "context_length_exceeded",
        });
    }
    let (decode_cap, delimiter_tokens) = match policy {
        ContextWindowPolicy::PromptAndMaxTokens => (cfg.max_new_tokens, 0),
        ContextWindowPolicy::PromptAndDecodeWithDelimiter => (
            cfg.max_new_tokens
                .saturating_add(cfg.effective_reasoning_budget().unwrap_or(0)),
            1,
        ),
    };
    let required = prompt_len
        .saturating_add(decode_cap)
        .saturating_add(delimiter_tokens);
    if required > model_max_context {
        let available = model_max_context.saturating_sub(prompt_len);
        let delimiter_clause = match delimiter_tokens {
            0 => String::new(),
            n => format!(" plus {n}"),
        };
        return Err(ApiError::BadRequest {
            message: format!(
                "prompt has {prompt_len} tokens, leaving {available} of the \
                 {model_max_context}-token context window for generation, but this \
                 request needs {decode_cap} generated tokens{delimiter_clause} (total {required}); \
                 reduce max_tokens/reasoning_budget or shorten the prompt"
            ),
            code: "context_length_exceeded",
        });
    }
    Ok(())
}

/// Raises the worker tokenizer's truncation cap to the model's context window.
///
/// A tokenizer loaded from `tokenizer.json` caps sequences at a fixed default
/// that can sit below the served context. Generation tokenizes the rendered
/// prompt with this same tokenizer, so a lower cap would silently drop the
/// prompt's tail (including the open assistant turn) for any prompt that
/// fits the window but exceeds the cap. Only ever raises the cap.
pub(crate) fn serving_tokenizer(tokenizer: BpeTokenizer, model_max_context: usize) -> BpeTokenizer {
    if tokenizer.max_seq_len() < model_max_context {
        tokenizer.with_max_seq_len(model_max_context)
    } else {
        tokenizer
    }
}

/// Renders a text-only job's prompt and admits it against the context window.
///
/// Admission uses the pre-truncation token count: a truncated count can never
/// exceed the tokenizer's cap, so an over-window prompt would pass the check
/// and then be generated from a shortened prefix.
pub(crate) fn render_text_prompt_within_window(
    tokenizer: &BpeTokenizer,
    messages: &[ChatMessage],
    policy: ContextWindowPolicy,
    model_max_context: usize,
    cfg: &GenerateConfig,
    _explicitly_thinking: bool,
) -> Result<(String, usize), ApiError> {
    let mut prompt = format_chat_template(messages);
    apply_qwen_thinking_mode(&mut prompt, cfg.enable_thinking);
    let prompt_len = tokenizer.tokenize(&prompt).pre_truncation_len;
    check_prompt_fits_window(policy, model_max_context, prompt_len, cfg)?;
    Ok((prompt, prompt_len))
}

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
