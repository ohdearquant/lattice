// ---------------------------------------------------------------------------
// serve subcommand: OpenAI-compatible HTTP API
// ---------------------------------------------------------------------------

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures::StreamExt as _;
use lattice_inference::Tokenizer;
// The canonical ChatML renderer is CPU-available (#668): this binary's
// CPU serve path renders normalized contract messages through the same
// formatter core the Metal worker uses, with no bespoke template copy.
use lattice_inference::forward::metal_qwen35::ChatMessage;
#[cfg(test)]
use lattice_inference::forward::metal_qwen35::format_chat_template;
#[cfg(feature = "metal-gpu")]
use lattice_inference::model::qwen35_config::GenerateConfig;
use lattice_inference::model::qwen35_config::{GenerateOutput, TokenLogprob};
use lattice_inference::serve::contract::{
    ChatRequest as ChatCompletionRequest, GenerationDefaults, ServeProfile,
    ValidatedChatRequest as ContractValidatedChatRequest,
    normalize_request_with_context_and_budget, validate_context_window_with_budget,
};
#[cfg(test)]
use lattice_inference::serve::contract::{
    ContentPart, Message, MessageContent, ResponseFormat, normalize_request,
};
use lattice_inference::serve::{format_normalized_chat_template, into_engine_chat_messages};
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Request body cap: 1 MiB.  Requests above this return HTTP 413.
/// ADR-080 C2 (#782): `lattice_inference::serve::REQUEST_BODY_LIMIT_BYTES`
/// is the single shared constant now; both binaries previously carried
/// this exact value independently.
use lattice_inference::serve::REQUEST_BODY_LIMIT_BYTES;

// -----------------------------------------------------------------------
// Model backend: CPU (safetensors) or Metal GPU (native Q4)
// -----------------------------------------------------------------------

/// Handle to the shared Metal GPU worker thread
/// (`lattice_inference::serve::metal_worker::MetalWorker`, issue #832:
/// the single shared owner module that replaces this binary's prior
/// private `MetalJob`/`MetalHandle` loop and `lattice_serve.rs`'s prior
/// private `Job`/`spawn_worker`/`run_worker_loop` loop). Cheaply `Clone`
/// (wraps a `MetalWorkerClient`, itself backed by an `mpsc` sender),
/// `Send + Sync`, so it can live in `AppState` like the CPU
/// `Arc<Qwen35Model>` does — only the underlying `MetalQwen35State`
/// inside the shared worker thread is confined to that thread.
///
/// This still serializes ALL Metal generation onto one thread: two
/// concurrent requests to a Q4-backed `lattice serve` run back-to-back,
/// not in parallel. That is correct for a single-GPU local engine (the
/// same default ollama uses) and is documented here rather than hidden
/// behind an innocuous-looking channel send.
#[cfg(feature = "metal-gpu")]
#[derive(Clone)]
pub struct MetalHandle {
    client: lattice_inference::serve::metal_worker::MetalWorkerClient,
}

#[cfg(feature = "metal-gpu")]
impl MetalHandle {
    fn supports_vision(&self) -> bool {
        self.client.supports_vision()
    }

    /// Normalizes
    /// [`lattice_inference::serve::metal_worker::WorkerEvent::Cancelled`]
    /// (the job was skipped at dequeue time: the client's `cancel`
    /// watch flag was already `true`, or this job's own event receiver
    /// was already closed) into the exact empty, interrupted
    /// `GenerateOutput` this binary's prior dequeue-cancellation reply
    /// already produced (#832: preserves this binary's own pre-existing
    /// observable shape — the shared `WorkerEvent::Cancelled` contract
    /// itself was modeled on it; see the shared module's own doc
    /// comment).
    fn normalize_cancelled(
        ev: lattice_inference::serve::metal_worker::WorkerEvent,
    ) -> lattice_inference::serve::metal_worker::WorkerEvent {
        use lattice_inference::serve::metal_worker::WorkerEvent;
        match ev {
            WorkerEvent::Cancelled => WorkerEvent::Complete(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: 0,
                generated_tokens: 0,
                stopped: false,
                stop_reason: Some(lattice_inference::StopReason::Interrupt),
                token_logprobs: vec![],
            }),
            other => other,
        }
    }

    /// Run one generation on the shared worker thread, forwarding each
    /// token delta to `on_token`. Returns the full `GenerateOutput`
    /// (including `stopped`/`stop_reason`) so callers can compute
    /// `finish_reason` with the exact same `finish_reason_for` helper
    /// the CPU path uses.
    ///
    /// Returns `Err` if the worker thread is unreachable
    /// (`ApiError::Internal`, "inference worker unavailable" — the same
    /// wording `lattice_serve.rs` uses for the identical condition on
    /// this shared worker contract, #832; this binary's prior distinct
    /// "not running" vs "dropped the request" phrasing collapses into
    /// one message here, since `MetalWorkerClient::submit` no longer
    /// exposes that distinction to callers), if the request cannot fit
    /// the model's context window (`ApiError::BadRequest`, surfaced by
    /// the shared worker's `check_prompt_fits_window` — new coverage
    /// for this binary; the HTTP-layer `check_context_window` preflight
    /// in `prepare_chat_request` already rejects the overwhelming
    /// majority of these before this call is ever reached), or if the
    /// underlying `generate_streaming` call itself fails closed (#611:
    /// e.g. a grammar mask that blocks every candidate token) —
    /// collapsed to `ApiError::Internal` here, matching the same
    /// generic "inference failed" 500 the CPU path already returns for
    /// any `generate()` error.
    async fn generate_streaming(
        &self,
        messages: Vec<ChatMessage>,
        gen_cfg: GenerateConfig,
        on_token: impl FnMut(&str) -> bool + Send + 'static,
    ) -> Result<GenerateOutput, ApiError> {
        // `cancel = never-fires` convenience form of
        // `generate_streaming_with_cancel`, for callers that do not wire
        // up disconnect cancellation.
        let (_never_cancels, cancel_rx) = tokio::sync::watch::channel(false);
        self.generate_streaming_with_cancel(messages, gen_cfg, on_token, cancel_rx)
            .await
    }

    /// Cancellation-aware sibling of [`Self::generate_streaming`]
    /// (ADR-080 C2, #744): `cancel` starts `false` and flips to `true`
    /// the moment the caller's paired
    /// `lattice_inference::serve::CancelOnDrop` guard is dropped (client
    /// disconnect). The shared worker (issue #832) checks it
    /// independently of `on_token`'s return value — before prefill,
    /// immediately after prefill, and at the top of every decode
    /// iteration — via
    /// `generate_streaming_with_prefix_cache_and_cancel`'s
    /// `should_cancel` predicate, and once more at dequeue time before
    /// paying for prefill on an already-abandoned job.
    ///
    /// `on_token`'s return value is still honored (this method stops
    /// calling it the first time it returns `false`, e.g. a
    /// disconnected `tx_delta`), but can no longer stop the worker
    /// thread directly the way it did when `MetalJob` embedded the
    /// callback inside the worker itself — the shared worker now lives
    /// behind a `WorkerEvent` channel this method drains from a
    /// separate async task, and `cancel` (checked independently by the
    /// worker) is the only signal that can reach across that boundary.
    /// In practice this is not a behavior change: `tx_delta` and the
    /// `cancel_guard` pairing `cancel` are dropped by the exact same
    /// axum stream-drop event at every call site, so `cancel` already
    /// catches a disconnect at essentially the same moment `on_token`
    /// would have.
    async fn generate_streaming_with_cancel(
        &self,
        messages: Vec<ChatMessage>,
        gen_cfg: GenerateConfig,
        on_token: impl FnMut(&str) -> bool + Send + 'static,
        cancel: tokio::sync::watch::Receiver<bool>,
    ) -> Result<GenerateOutput, ApiError> {
        // #932: the ONE way `MetalWorkerClient::submit` fails outwardly
        // -- the shared worker's outstanding-job admission cap is full.
        // Surfaces as an ordinary `ApiError::ServiceUnavailable` (503),
        // exactly like any other `Err` this method already returns.
        let mut rx = self.submit(messages, gen_cfg, cancel)?;
        Self::drain(&mut rx, on_token).await
    }

    /// Admission-only half of [`Self::generate_streaming_with_cancel`]
    /// (#939): runs `MetalWorkerClient::submit`'s synchronous admission
    /// check (issue #932's `Semaphore::try_acquire_owned`) and returns
    /// immediately, before any event is drained from the worker.
    ///
    /// Split out so `chat_completions`'s streaming arm can call this
    /// directly -- and propagate `Err(ApiError::ServiceUnavailable)`
    /// with `?` -- BEFORE building and returning the SSE response,
    /// instead of discovering admission failure only after a detached
    /// `tokio::spawn` task (which may not even have started running
    /// yet) reaches this call. Draining is [`Self::drain`], called
    /// separately once the response has been committed.
    fn submit(
        &self,
        messages: Vec<ChatMessage>,
        gen_cfg: GenerateConfig,
        cancel: tokio::sync::watch::Receiver<bool>,
    ) -> Result<
        tokio::sync::mpsc::UnboundedReceiver<lattice_inference::serve::metal_worker::WorkerEvent>,
        ApiError,
    > {
        self.client.submit(messages, gen_cfg, cancel)
    }

    /// Drains a receiver obtained from [`Self::submit`], forwarding
    /// token deltas to `on_token` and normalizing
    /// `WorkerEvent::Cancelled` -- exactly the loop
    /// `generate_streaming_with_cancel` ran inline before admission was
    /// split out of it (#939).
    async fn drain(
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<
            lattice_inference::serve::metal_worker::WorkerEvent,
        >,
        mut on_token: impl FnMut(&str) -> bool + Send + 'static,
    ) -> Result<GenerateOutput, ApiError> {
        use lattice_inference::serve::metal_worker::WorkerEvent;

        let mut deliver_deltas = true;
        loop {
            let Some(ev) = rx.recv().await else {
                return Err(ApiError::Internal {
                    message: "inference worker unavailable".to_string(),
                });
            };
            match Self::normalize_cancelled(ev) {
                WorkerEvent::Delta(delta) => {
                    if deliver_deltas && !on_token(&delta) {
                        deliver_deltas = false;
                    }
                }
                WorkerEvent::Complete(output) => return Ok(output),
                WorkerEvent::Rejected(api_err) => return Err(api_err),
                WorkerEvent::Failed(message) | WorkerEvent::ConstraintBlocked(message) => {
                    return Err(ApiError::Internal {
                        message: format!("generation failed: {message}"),
                    });
                }
                WorkerEvent::Cancelled => {
                    unreachable!("normalize_cancelled already rewrote Cancelled into Complete")
                }
                // Unrecognized future event kind: mirror the
                // `Failed`/`ConstraintBlocked` arm above and fail the
                // request with a generic internal error rather than
                // guessing.
                _ => {
                    return Err(ApiError::Internal {
                        message: "generation failed: unrecognized worker event".to_string(),
                    });
                }
            }
        }
    }
}

#[cfg(feature = "metal-gpu")]
fn map_metal_generation_error(error: ApiError) -> ApiError {
    match error {
        error @ (ApiError::ServiceUnavailable { .. } | ApiError::BadRequest { .. }) => error,
        other => {
            eprintln!("generation error (metal): {other:?}");
            ApiError::Internal {
                message: "inference failed".to_string(),
            }
        }
    }
}

/// The two ways `AppState` can run generation: the original CPU
/// (safetensors) path via `Arc<Qwen35Model>`, or the Metal GPU (native
/// Q4) path via a worker-thread handle. Both variants funnel into the
/// same request handler code below — `chat_completions` branches on this
/// enum in exactly two places (streaming and non-streaming) rather than
/// duplicating the handler.
#[derive(Clone)]
pub enum ModelBackend {
    Cpu(Arc<lattice_inference::model::qwen35::Qwen35Model>),
    #[cfg(feature = "metal-gpu")]
    Metal {
        handle: MetalHandle,
        tokenizer: Arc<lattice_inference::tokenizer::bpe::BpeTokenizer>,
        max_context: usize,
    },
    /// Test-only seam (ADR-080 C2): wraps a real tiny model for
    /// `tokenize_len`/`max_context`/
    /// `tokenizer` (so request validation stays realistic) but
    /// substitutes the CPU streaming generation call itself with an
    /// injected closure. This lets a test observe `should_cancel` being
    /// polled by the EXACT production composition in
    /// `chat_completions`'s CPU streaming arm -- the same `on_token`/
    /// `should_cancel` construction and `cancel_rx` wiring real
    /// requests use -- independently of a real decode loop's timing,
    /// isolating cancellation-signal wiring from `on_token`'s own
    /// failed-send stop condition (the exact ambiguity the disconnect
    /// test's mutation gap left open). Never constructible outside
    /// `--features test-utils`.
    #[cfg(all(feature = "test-utils", test))]
    CpuFakeGenerate {
        model: Arc<lattice_inference::model::qwen35::Qwen35Model>,
        #[allow(clippy::type_complexity)]
        generate: Arc<
            dyn Fn(
                    &str,
                    &lattice_inference::model::qwen35_config::GenerateConfig,
                    &mut dyn FnMut(&str) -> bool,
                    &mut dyn FnMut() -> bool,
                )
                    -> Result<GenerateOutput, lattice_inference::error::InferenceError>
                + Send
                + Sync,
        >,
    },
}

impl ModelBackend {
    pub fn tokenize_len(&self, text: &str) -> usize {
        match self {
            ModelBackend::Cpu(m) => m.tokenizer().tokenize(text).real_length,
            #[cfg(feature = "metal-gpu")]
            ModelBackend::Metal { tokenizer, .. } => tokenizer.tokenize(text).real_length,
            #[cfg(all(feature = "test-utils", test))]
            ModelBackend::CpuFakeGenerate { model, .. } => {
                model.tokenizer().tokenize(text).real_length
            }
        }
    }

    pub fn max_context(&self) -> usize {
        match self {
            ModelBackend::Cpu(m) => m.max_context(),
            #[cfg(feature = "metal-gpu")]
            ModelBackend::Metal { max_context, .. } => *max_context,
            #[cfg(all(feature = "test-utils", test))]
            ModelBackend::CpuFakeGenerate { model, .. } => model.max_context(),
        }
    }

    /// Tokenizer for this backend, used to render `logprobs` token ids
    /// back into text/bytes (#585).
    pub fn tokenizer(&self) -> &lattice_inference::tokenizer::bpe::BpeTokenizer {
        match self {
            ModelBackend::Cpu(m) => m.tokenizer(),
            #[cfg(feature = "metal-gpu")]
            ModelBackend::Metal { tokenizer, .. } => tokenizer,
            #[cfg(all(feature = "test-utils", test))]
            ModelBackend::CpuFakeGenerate { model, .. } => model.tokenizer(),
        }
    }

    pub fn supports_vision(&self) -> bool {
        match self {
            ModelBackend::Cpu(_) => false,
            #[cfg(feature = "metal-gpu")]
            ModelBackend::Metal { handle, .. } => handle.supports_vision(),
            #[cfg(all(feature = "test-utils", test))]
            ModelBackend::CpuFakeGenerate { .. } => false,
        }
    }

    /// Load a native Q4 checkpoint on the shared Metal worker thread
    /// (`lattice_inference::serve::metal_worker::MetalWorker`, issue
    /// #832 — the same shared owner `lattice_serve.rs` uses) and return
    /// the `ModelBackend::Metal` handle plus the resolved context
    /// window, for `main()`'s `Command::Serve` startup sequence.
    #[cfg(feature = "metal-gpu")]
    pub fn spawn_metal(
        model_dir: std::path::PathBuf,
        tokenizer_dir: Option<std::path::PathBuf>,
        max_pending: usize,
        preload_vision: bool,
    ) -> Result<(Self, usize), String> {
        use lattice_inference::serve::metal_worker::{
            ContextWindowPolicy, MetalWorker, StartupError, VisionRuntime, WorkerMetadata,
        };

        let tokenizer_path = tokenizer_dir
            .as_deref()
            .unwrap_or(&model_dir)
            .join("tokenizer.json");
        let tokenizer = Arc::new(
            lattice_inference::tokenizer::bpe::BpeTokenizer::from_tokenizer_json(&tokenizer_path)
                .map_err(|e| format!("tokenizer load failed ({}): {e}", tokenizer_path.display()))?,
        );
        // #832: the shared worker's loader needs its own owned
        // tokenizer (it renders + tokenizes the prompt once per request
        // for the KV-window check, `check_prompt_fits_window`).
        // Cloned from the `Arc` above rather than re-read from disk, so
        // `tokenizer.json` is still parsed exactly once; this is a
        // single, one-time, startup-only in-memory clone, not a
        // per-request cost.
        let tokenizer_for_worker = (*tokenizer).clone();
        // Preserves this binary's pre-existing behavior exactly: the
        // context window is this fixed cap, not re-derived from
        // `state.max_context()` after loading (unlike
        // `lattice_serve.rs`'s `load_model`). Passed to the shared
        // worker's `WorkerMetadata` too, so its internal
        // `check_prompt_fits_window` invariant agrees with the
        // HTTP-layer `check_context_window` preflight that already runs
        // first in `prepare_chat_request`.
        let max_context = crate::chat::chat_max_cache_len();
        let vision_config = crate::chat::load_q4_config(&model_dir)?;
        let mut vision_runtime =
            VisionRuntime::from_model_config(model_dir.clone(), &vision_config);
        if preload_vision {
            // issue #1336: eager-load now, on this startup thread, before the
            // Metal worker thread spawns. Lazy stays the default (see the
            // `--preload-vision` help text); a failure here must not abort
            // startup — warn and fall back to the normal lazy load on the
            // first image request, exactly as if this flag were absent.
            if let Err(err) = vision_runtime.preload() {
                eprintln!(
                    "Warning: --preload-vision failed, falling back to lazy vision loading: {err}"
                );
            }
        }
        let model_dir_for_loader = model_dir.clone();
        let tokenizer_path_for_loader = tokenizer_path.clone();
        let (owner, client, _meta) = MetalWorker::spawn_with_vision(
            move || {
                let cfg = crate::chat::load_q4_config(&model_dir_for_loader)?;
                let state =
                    lattice_inference::forward::metal_qwen35::MetalQwen35State::from_q4_dir(
                        &model_dir_for_loader,
                        &tokenizer_path_for_loader,
                        &cfg,
                        max_context,
                    )
                    .map_err(|e| format!("Q4 model load failed: {e}"))?;
                Ok((
                    state,
                    tokenizer_for_worker,
                    WorkerMetadata {
                        format: "q4".to_string(),
                        model_max_context: max_context,
                        context_window_policy: ContextWindowPolicy::PromptAndMaxTokens,
                    },
                ))
            },
            vision_runtime,
            max_pending,
        )
        .map_err(|e| match e {
            StartupError::Load(msg) => msg,
            // Preserves this binary's exact prior wording (distinct
            // from `MetalWorker::spawn`'s own generic
            // `StartupError::ThreadExited` `Display` text) for the same
            // condition: the worker thread exited/panicked before ever
            // sending a readiness signal.
            StartupError::ThreadExited => {
                "Metal worker thread exited before loading finished".to_string()
            }
            // #939: clap's own `value_parser` range already rejects an
            // out-of-range `--max-pending` before this call, so this
            // arm is unreachable in practice through the CLI -- kept
            // for exhaustiveness and as defense in depth against any
            // other future `spawn_metal` caller.
            err @ StartupError::InvalidMaxPending { .. } => err.to_string(),
            // Unrecognized future startup-failure kind: fall back to its
            // `Display` text rather than guessing at a more specific
            // wording, same as `InvalidMaxPending` above.
            err => err.to_string(),
        })?;
        // The explicit owner is not needed by this binary: every
        // production `MetalWorkerClient` retains an owner clone. The
        // clone held by `client` keeps the worker alive through the
        // router's lifetime; on a normal return, dropping the last
        // client closes the queue before the last owner performs its
        // bounded join. As with `lattice_serve.rs`, that guarantee does
        // not reach this file's fatal `std::process::exit` calls (e.g.
        // the server-error path a few lines below, after
        // `serve_until_shutdown` returns `Err`): process exit never
        // runs destructors, so any owner clone still alive at that
        // point is dropped without performing the bounded join.
        drop(owner);
        Ok((
            ModelBackend::Metal {
                handle: MetalHandle { client },
                tokenizer,
                max_context,
            },
            max_context,
        ))
    }
}

// -----------------------------------------------------------------------
// Shared application state
// -----------------------------------------------------------------------

/// State shared across all request handlers via axum's `State` extractor.
#[derive(Clone)]
pub struct AppState {
    /// The loaded model backend (CPU safetensors or Metal GPU Q4).
    pub model: ModelBackend,
    /// Default `max_tokens` value used when a request omits the field.
    /// Set from the `--max-tokens` CLI flag passed to `lattice serve`.
    pub default_max_tokens: usize,
    /// Hard upper bound on `max_tokens` accepted from any request.
    /// Prevents callers from requesting unbounded generation.
    pub max_tokens_cap: usize,
    /// Canonical model identifier echoed in every response.
    /// Derived from the `--model-id` flag or the model path basename.
    pub model_id: String,
    /// Monotonically increasing counter used to make response IDs unique
    /// across concurrent requests within the same second.
    pub request_counter: Arc<AtomicU64>,
    /// Pooled text/image embedding model for `/v1/embeddings`, loaded
    /// independently of `model` (a separate f16-packed checkpoint format --
    /// see `lattice_inference::serve::embeddings`'s module doc comment).
    /// `None` when no vision-language checkpoint was found at the served
    /// model directory; every `/v1/embeddings` request then fails closed
    /// with `vision_unsupported`.
    pub embedding_model: Option<Arc<lattice_inference::serve::embeddings::EmbeddingModel>>,
}

// -----------------------------------------------------------------------
// Error type (ADR-080 C2, #782): shared verbatim with `lattice_serve.rs`
// via `lattice_inference::serve::ApiError` -- this binary's local
// `ApiError`/`ErrorBody`/`ErrorDetail`/`IntoResponse` were byte-identical
// to the shared definition, so they are gone; every existing
// `ApiError::BadRequest { message, code }` / `PayloadTooLarge` /
// `Internal` construction site below is unaffected (same variant names
// and fields).
// -----------------------------------------------------------------------

use lattice_inference::serve::ApiError;

// -----------------------------------------------------------------------
// Request / response types
// -----------------------------------------------------------------------

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: ResponseMessage,
    pub finish_reason: String,
    /// Per-token log-probabilities (#585). `None` unless the request set
    /// `logprobs: true`, matching the OpenAI response shape where the
    /// field is omitted rather than `null` for a plain completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// `choices[].logprobs` — OpenAI chat-completions logprobs envelope (#585).
#[derive(Serialize)]
pub struct ChoiceLogprobs {
    pub content: Vec<TokenLogprobEntry>,
}

/// One sampled token's log-probability, plus its top-N alternatives.
#[derive(Serialize)]
pub struct TokenLogprobEntry {
    pub token: String,
    pub logprob: f32,
    /// Raw UTF-8 bytes of `token`. `None` when the token id could not be
    /// resolved back to vocabulary text (should not happen for a token
    /// this server just sampled, but fails closed rather than panicking).
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprobEntry>,
}

/// One alternative token considered at a sampled position.
#[derive(Serialize)]
pub struct TopLogprobEntry {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

#[derive(Serialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

// -----------------------------------------------------------------------
// SSE streaming types
// -----------------------------------------------------------------------

/// Internal channel message type for the streaming generation path.
///
/// `spawn_blocking` runs the sync `generate_streaming` call on a blocking
/// thread and sends incremental deltas through an unbounded channel.  The
/// async SSE handler reads from the other end and maps these messages to
/// OpenAI `chat.completion.chunk` events.
pub enum StreamMsg {
    /// One incremental text delta from the model.
    Delta(String),
    /// Generation finished normally; carries the OpenAI finish reason.
    Done { finish_reason: &'static str },
    /// Generation failed (invariant violation or engine error).
    Failed,
}

/// Top-level chunk object serialised into each `data:` SSE event.
#[derive(Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    /// Null while streaming, set on the final choice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

/// The `delta` field of a streaming chunk.
///
/// Exactly one of `role` / `content` is set per chunk:
/// - First chunk: `role = "assistant"`, no content.
/// - Subsequent content chunks: `content = <text>`, no role.
/// - Final finish chunk: both absent (empty delta `{}`).
#[derive(Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// -----------------------------------------------------------------------
// Validation helpers — pure functions, no model required, easily tested
// -----------------------------------------------------------------------

/// Resolve the effective `max_tokens`, rejecting zero, values above the
/// server cap, and conflicting `max_tokens` / `max_completion_tokens`.
#[cfg(test)]
fn validate_max_tokens(
    req_max: Option<usize>,
    req_max_completion: Option<usize>,
    default_max_tokens: usize,
    max_tokens_cap: usize,
) -> Result<usize, ApiError> {
    let effective = match (req_max, req_max_completion) {
        (None, None) => default_max_tokens,
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (Some(a), Some(b)) if a == b => a,
        (Some(a), Some(b)) => {
            return Err(ApiError::BadRequest {
                message: format!(
                    "max_tokens ({a}) and max_completion_tokens ({b}) differ; supply only one"
                ),
                code: "invalid_request",
            });
        }
    };
    // ADR-080 C2, #745: the zero-rejection itself is the shared contract
    // (`lattice_serve.rs`'s `build_cfg` silently clamped a
    // client-supplied `max_tokens: 0` through instead of rejecting it);
    // the cap-reject below stays local since the two binaries'
    // over-cap policies deliberately differ (this one rejects, the
    // daemon clamps to the model's context window).
    lattice_inference::serve::reject_zero_max_tokens(effective)?;
    if effective > max_tokens_cap {
        return Err(ApiError::BadRequest {
            message: format!("max_tokens {effective} exceeds server limit {max_tokens_cap}"),
            code: "max_tokens_exceeds_limit",
        });
    }
    Ok(effective)
}

/// Validate `temperature` is in `[0.0, 2.0]`.
#[cfg(test)]
fn validate_temperature(value: Option<f32>) -> Result<f32, ApiError> {
    lattice_inference::serve::contract::validate_temperature(
        value.unwrap_or(GenerationDefaults::standard(1).temperature),
    )
}

/// Validate `top_p` is in `(0.0, 1.0]`.
#[cfg(test)]
fn validate_top_p(value: Option<f32>) -> Result<f32, ApiError> {
    lattice_inference::serve::contract::validate_top_p(
        value.unwrap_or(GenerationDefaults::standard(1).top_p),
    )
}

/// Validate the `logprobs` / `top_logprobs` pair (#585) and resolve the
/// number of alternatives to capture per token.
///
/// - `logprobs` absent or `false` → `Ok(None)` (capture disabled, zero cost).
/// - `logprobs: true`, `top_logprobs` absent → `Ok(Some(0))` (per-token
///   logprob only, no alternatives — matches the OpenAI default).
/// - `logprobs: true`, `top_logprobs: Some(n)` with `0 <= n <= 20` → `Ok(Some(n))`.
/// - `top_logprobs` set without `logprobs: true` → rejected (matches OpenAI).
/// - `top_logprobs > 20` → rejected.
#[cfg(test)]
fn validate_logprobs(
    logprobs: Option<bool>,
    top_logprobs: Option<usize>,
) -> Result<Option<usize>, ApiError> {
    if !logprobs.unwrap_or(false) {
        if top_logprobs.is_some() {
            return Err(ApiError::BadRequest {
                message: "top_logprobs requires logprobs: true".to_string(),
                code: "invalid_request",
            });
        }
        return Ok(None);
    }
    let top_n = top_logprobs.unwrap_or(0);
    if top_n > 20 {
        return Err(ApiError::BadRequest {
            message: format!("top_logprobs {top_n} exceeds the maximum of 20"),
            code: "invalid_top_logprobs",
        });
    }
    Ok(Some(top_n))
}

/// Parse the OpenAI `stop` field into a `Vec<String>`.
///
/// Accepted forms:
/// - `null` / absent → empty vec (no string-level stops)
/// - a JSON string → `vec![s]`
/// - a JSON array of 1–4 non-empty strings → that vec
///
/// Returns `Err(BadRequest)` for:
/// - an empty array
/// - an array with more than 4 elements
/// - any array element that is not a string
/// - any stop string that is empty
#[cfg(test)]
fn parse_stop_strings(stop: &Option<Value>) -> Result<Vec<String>, ApiError> {
    lattice_inference::serve::contract::parse_stop_strings(stop)
}

/// Reject OpenAI fields that are parsed but not yet implemented.
///
/// Note: `stream=true` is now handled by the streaming path in `chat_completions`
/// and is intentionally NOT rejected here. `logprobs`/`top_logprobs` are
/// implemented on the non-streaming path only (#585); combined with
/// `stream: true` they are rejected below rather than silently ignored.
#[cfg(test)]
fn reject_unsupported(req: &ChatCompletionRequest) -> Result<(), ApiError> {
    if req.tools.is_some() || req.tool_choice.is_some() {
        return Err(ApiError::BadRequest {
            message: "tools and tool_choice are not supported by this server".to_string(),
            code: "unsupported_feature",
        });
    }
    if req.stream == Some(true) && req.logprobs.unwrap_or(false) {
        return Err(ApiError::BadRequest {
            message: "logprobs is not supported together with stream: true".to_string(),
            code: "unsupported_feature",
        });
    }
    if req.n.unwrap_or(1) > 1 {
        return Err(ApiError::BadRequest {
            message: "n > 1 is not supported".to_string(),
            code: "unsupported_feature",
        });
    }
    if let Some(fmt) = &req.response_format
        && fmt.r#type != "text"
    {
        return Err(ApiError::BadRequest {
            message: format!(
                "response_format.type '{}' is not supported; use 'text'",
                fmt.r#type
            ),
            code: "unsupported_feature",
        });
    }
    Ok(())
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// `#[cfg(test)]`-only adapter for fixtures that exercise the engine
/// renderer. Validation and contract-to-engine conversion both delegate
/// to the same shared paths production uses.
#[cfg(test)]
fn to_chat_messages(messages: &[Message]) -> Result<Vec<ChatMessage>, ApiError> {
    lattice_inference::serve::contract::normalize_messages(messages)
        .and_then(into_engine_chat_messages)
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Maps a `GenerateOutput` to the OpenAI `finish_reason` string (ADR-080
/// C2, #746): delegates to the shared `lattice_inference::serve::
/// finish_reason`, which both binaries now use so the mapping cannot
/// drift between them again -- `lattice_serve.rs`'s worker previously
/// hardcoded `"stop"` unconditionally instead of carrying the engine's
/// `stopped` flag through.
pub(super) fn finish_reason_for(
    output: &lattice_inference::model::qwen35_config::GenerateOutput,
) -> &'static str {
    lattice_inference::serve::finish_reason(output.stopped)
}

/// Decode-side token allowance for the post-generation length invariant
/// (#1334).
///
/// Mirrors `decode_cap` (`crates/inference/src/model/qwen35_config.rs`),
/// which this must stay in agreement with: when a positive
/// `reasoning_budget` is active and the model does not close its own
/// thinking block, the engine force-emits a `</think>` delimiter as an
/// extra generated token (`force_close_think` / `DecodePolicy::
/// apply_override`), so a completion that legitimately used the full
/// reasoning and answer allowance is `reasoning_budget + max_tokens + 1`
/// tokens long, not `reasoning_budget + max_tokens`. The invariant is
/// `generated_tokens > reasoning_budget + max_tokens + 1` for a positive
/// budget, and unchanged at `generated_tokens > max_tokens` when the
/// budget is absent or zero (no forced delimiter can occur).
fn decode_token_budget(max_tokens: usize, reasoning_budget: Option<usize>) -> usize {
    match reasoning_budget {
        Some(rb) if rb > 0 => rb.saturating_add(max_tokens).saturating_add(1),
        _ => max_tokens,
    }
}

/// Resolve a token id back to its OpenAI `logprobs` text/bytes representation (#585).
///
/// `token` uses the lossy UTF-8 rendering (matches OpenAI, which also shows
/// replacement characters for a token that is only part of a multi-byte
/// codepoint); `bytes` carries the exact original bytes so callers can
/// reconstruct byte-accurate output regardless of codepoint boundaries.
///
/// Every token id this server places into `token_logprobs` was just sampled
/// by this same tokenizer's vocabulary, so `token_for_id` returning `None`
/// is not expected in practice; the fallback fails closed with a visibly
/// synthetic token string and no bytes, rather than panicking.
fn render_token_logprob(
    tokenizer: &lattice_inference::tokenizer::bpe::BpeTokenizer,
    token_id: u32,
) -> (String, Option<Vec<u8>>) {
    match tokenizer.token_bytes_for_id(token_id) {
        Some(bytes) => (String::from_utf8_lossy(&bytes).into_owned(), Some(bytes)),
        None => (format!("<|unresolved_token_{token_id}|>"), None),
    }
}

/// Build the `choices[].logprobs` envelope from the engine's raw
/// `token_logprobs` (#585). `token_logprobs` is empty when `logprobs` was
/// not requested, in which case this returns an empty `content` — callers
/// only invoke this when the request set `logprobs: true`, so that case
/// does not arise in practice.
fn build_choice_logprobs(
    tokenizer: &lattice_inference::tokenizer::bpe::BpeTokenizer,
    token_logprobs: &[TokenLogprob],
) -> ChoiceLogprobs {
    let content = token_logprobs
        .iter()
        .map(|tl| {
            let (token, bytes) = render_token_logprob(tokenizer, tl.token_id);
            let top_logprobs = tl
                .top
                .iter()
                .map(|alt| {
                    let (token, bytes) = render_token_logprob(tokenizer, alt.token_id);
                    TopLogprobEntry {
                        token,
                        logprob: alt.logprob,
                        bytes,
                    }
                })
                .collect();
            TokenLogprobEntry {
                token,
                logprob: tl.logprob,
                bytes,
                top_logprobs,
            }
        })
        .collect();
    ChoiceLogprobs { content }
}

// Handlers
// -----------------------------------------------------------------------

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

/// `GET /` (ADR-080 C2): a minimal engine-
/// identity/endpoint-discovery document, in the same shape
/// `lattice_serve.rs` already served on its own daemon -- this binary
/// had no equivalent route at all, an undocumented route-set divergence
/// between the two binaries -- the routes did not actually match 1:1
/// until this route landed.
///
/// Built locally rather than returning the shared
/// [`lattice_inference::serve::root_body`] verbatim: this binary's
/// `/v1/embeddings` route (unlike `/v1/chat/completions`, `/v1/models`, and
/// `/health`) does not exist on `lattice_serve.rs` yet, so advertising it
/// through the byte-identical shared body would falsely claim the daemon
/// has it too.
pub async fn root() -> Json<Value> {
    let mut body = lattice_inference::serve::root_body();
    if let Some(endpoints) = body.get_mut("endpoints").and_then(Value::as_array_mut) {
        endpoints.push(Value::String("/v1/embeddings".to_string()));
    }
    Json(body)
}

/// `GET /v1/models` (ADR-080 C2, #746's sibling gap): advertises the
/// single loaded model, in the same shape `lattice_serve.rs` already
/// served on its own daemon -- this binary had no equivalent route at
/// all before this change.
pub async fn list_models(State(state): State<AppState>) -> Json<Value> {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Json(lattice_inference::serve::models_list_body(
        &state.model_id,
        created,
    ))
}

/// Test adapter for the shared normalization cascade without a real
/// context-window check. Production uses `prepare_chat_request`, which
/// supplies the prompt-aware check to the same shared function.
#[cfg(test)]
fn validate_chat_request(
    req: &ChatCompletionRequest,
    model_id: &str,
    default_max_tokens: usize,
    max_tokens_cap: usize,
) -> Result<ContractValidatedChatRequest, ApiError> {
    normalize_request(
        req,
        GenerationDefaults::standard(default_max_tokens),
        ServeProfile::lattice(model_id, max_tokens_cap),
    )
}

/// Output of the full pre-generation validation cascade, ready for
/// `gen_cfg` construction.
#[derive(Debug)]
struct PreparedChatRequest {
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    logprobs: Option<usize>,
    prompt: String,
    stop_strings: Vec<String>,
    reasoning_budget: Option<usize>,
    seed: Option<u64>,
    stream: bool,
}

/// Production entry point for the shared context-aware normalization
/// cascade: supplies the prompt-aware context-window check (rendering the
/// chat template, tokenizing it, then calling the shared
/// `validate_context_window_with_budget`) as the context check, in the
/// exact order the original inline `chat_completions` cascade used:
/// `stop` is validated *last*, after both the served-model hard
/// requirements and the context-window check that guards against a
/// panic in the blocking generation path. A request that is both
/// over-context and carries a malformed `stop` field must fail with
/// `context_length_exceeded`, not a stop-parsing error — pinned by
/// `cm_serve_context_window_checked_before_stop_parsing`.
///
/// `tokenize_len`/`max_context` are threaded through as thunks (rather
/// than a `&ModelBackend`) so this whole cascade — including the
/// ordering — is testable without constructing a real model: the
/// rendered `prompt` that `tokenize_len` needs only exists once
/// `validate_chat_request` has already run, so the thunk form lets a
/// test control the token count `check_context_window` sees without
/// having to fake a tokenizer.
fn prepare_chat_request(
    req: &ChatCompletionRequest,
    model_id: &str,
    default_max_tokens: usize,
    max_tokens_cap: usize,
    vision_supported: bool,
    tokenize_len: impl FnOnce(&str) -> usize,
    max_context: impl FnOnce() -> usize,
) -> Result<PreparedChatRequest, ApiError> {
    let (validated, prompt) = normalize_request_with_context_and_budget(
        req,
        GenerationDefaults::standard(default_max_tokens),
        ServeProfile::lattice(model_id, max_tokens_cap).with_vision_support(vision_supported),
        |messages, max_tokens, reasoning_budget| {
            let prompt = format_normalized_chat_template(messages);
            let prompt_token_count = tokenize_len(&prompt);
            validate_context_window_with_budget(
                prompt_token_count,
                max_tokens,
                reasoning_budget,
                max_context(),
            )?;
            Ok(prompt)
        },
    )?;
    let ContractValidatedChatRequest {
        messages,
        max_tokens,
        temperature,
        top_p,
        logprobs,
        stop_strings,
        reasoning_budget,
        seed,
        stream,
        ..
    } = validated;
    let messages = into_engine_chat_messages(messages)?;

    Ok(PreparedChatRequest {
        messages,
        max_tokens,
        temperature,
        top_p,
        logprobs,
        prompt,
        stop_strings,
        reasoning_budget,
        seed,
        stream,
    })
}

/// The `on_token`/`should_cancel` composition for CPU-style streaming
/// generation, constructed in exactly ONE place and shared by the real
/// `ModelBackend::Cpu` arm and the test-only `CpuFakeGenerate` arm below
/// (ADR-080 C2). Before this,
/// each arm rebuilt `move || *cancel_rx.borrow()` independently, so a
/// a mutation that broke ONLY the real `Cpu` arm's
/// predicate left the `CpuFakeGenerate`-only post-drop oracle green --
/// it was pinning a copy, not the production call site. Both arms now
/// funnel through this one function, so mutating the shared predicate
/// here is observed by the fake-arm-driven test too: there is no
/// separate copy left unmutated.
#[allow(clippy::type_complexity)]
fn spawn_cpu_style_streaming_generation(
    tx: futures::channel::mpsc::UnboundedSender<StreamMsg>,
    cancel_rx: tokio::sync::watch::Receiver<bool>,
    prompt: String,
    gen_cfg: lattice_inference::model::qwen35_config::GenerateConfig,
    generate: Arc<
        dyn Fn(
                &str,
                &lattice_inference::model::qwen35_config::GenerateConfig,
                &mut dyn FnMut(&str) -> bool,
                &mut dyn FnMut() -> bool,
            ) -> Result<GenerateOutput, lattice_inference::error::InferenceError>
            + Send
            + Sync,
    >,
    finish_streaming: impl FnOnce(GenerateOutput) + Send + 'static,
) {
    tokio::task::spawn_blocking(move || {
        let tx_delta = tx.clone();
        let mut on_token = move |delta: &str| {
            tx_delta
                .unbounded_send(StreamMsg::Delta(delta.to_string()))
                .is_ok()
        };
        let mut should_cancel = move || *cancel_rx.borrow();
        let result = generate(&prompt, &gen_cfg, &mut on_token, &mut should_cancel);
        match result {
            Ok(output) => finish_streaming(output),
            Err(e) => {
                eprintln!("generation error (streaming): {e}");
                let _ = tx.unbounded_send(StreamMsg::Failed);
            }
        }
    });
}

/// Axum route entry point. Takes the raw request body instead of
/// `Json<ChatCompletionRequest>` so `require_json_content_type` can run
/// against the raw headers before the body is read (see below). The
/// message-count bound is enforced inline during the single
/// `serde_json::from_slice::<ChatCompletionRequest>` parse below
/// (`serve::contract::deserialize_bounded_messages`), so a sub-body-cap
/// request built from tens of thousands of tiny messages is rejected
/// without materializing a `Vec<Message>` entry for each one -- there is
/// no separate raw-bytes pass over `messages` ahead of that parse.
///
/// `to_bytes(.., REQUEST_BODY_LIMIT_BYTES)` enforces the same cap the
/// router's `DefaultBodyLimit::max(REQUEST_BODY_LIMIT_BYTES)` layer
/// already applies to the underlying body stream, so the existing 413
/// behavior below is unchanged.
///
/// Switching from `Json` to a raw body also dropped `Json`'s own
/// Content-Type enforcement (a security gap: a body with a valid JSON
/// payload but `Content-Type: text/plain` -- or no `Content-Type` at
/// all -- previously got a free 415 from the `Json` extractor before
/// this handler ever ran). Restored via
/// [`lattice_inference::serve::require_json_content_type`], checked
/// against `headers` and the request rejected *before* the body is
/// read at all: unlike `Result<Bytes, BytesRejection>` (an axum
/// `FromRequest` extractor that fully buffers the body as part of
/// argument extraction, before this function body ever runs), taking
/// the raw `Body` here defers reading the body to the explicit
/// `to_bytes` call below, so an invalid-MIME request never pays the
/// buffering cost, mirroring `lattice_serve.rs`'s equivalent handler.
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
) -> Result<Response, ApiError> {
    lattice_inference::serve::require_json_content_type(&headers)?;

    // Surface a body-length-limit rejection as a structured 413
    // response; any other body-buffering failure (e.g. a client
    // disconnecting mid-stream) is not a size violation and gets the
    // same non-413 invalid-body response as a malformed JSON body.
    let bytes = axum::body::to_bytes(body, REQUEST_BODY_LIMIT_BYTES)
        .await
        .map_err(|err| {
            let is_length_limit = std::error::Error::source(&err)
                .is_some_and(<dyn std::error::Error>::is::<http_body_util::LengthLimitError>);
            if is_length_limit {
                return ApiError::PayloadTooLarge {
                    message: "request body exceeds 1 MiB limit".to_string(),
                };
            }
            eprintln!("invalid request body: {err}");
            ApiError::BadRequest {
                message: "invalid JSON request body".to_string(),
                code: "invalid_request_body",
            }
        })?;

    let req: ChatCompletionRequest = serde_json::from_slice(&bytes).map_err(|err| {
        if lattice_inference::serve::contract::is_message_flood_error(&err) {
            return ApiError::BadRequest {
                message: lattice_inference::serve::contract::message_flood_text(),
                code: "invalid_request_body",
            };
        }
        eprintln!("invalid request body: {err}");
        ApiError::BadRequest {
            message: "invalid JSON request body".to_string(),
            code: "invalid_request_body",
        }
    })?;

    chat_completions_with_request(State(state), req).await
}

/// The full chat-completions cascade, taking an already-parsed request.
/// Split out of [`chat_completions`] so tests can construct a
/// `ChatCompletionRequest` directly (a Rust struct literal) without
/// round-tripping it through JSON bytes -- those tests exercise
/// generation/streaming behavior, not the raw-bytes preflight, which is
/// covered separately in `serve::contract`'s own tests and this
/// binary's router-level tests.
async fn chat_completions_with_request(
    State(state): State<AppState>,
    req: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    let PreparedChatRequest {
        messages: _normalized_messages,
        max_tokens,
        temperature,
        top_p,
        logprobs,
        prompt,
        stop_strings,
        reasoning_budget,
        seed,
        stream,
    } = prepare_chat_request(
        &req,
        &state.model_id,
        state.default_max_tokens,
        state.max_tokens_cap,
        state.model.supports_vision(),
        |p| state.model.tokenize_len(p),
        || state.model.max_context(),
    )?;

    let gen_cfg = lattice_inference::model::qwen35_config::GenerateConfig {
        max_new_tokens: max_tokens,
        temperature,
        top_p,
        seed,
        stop_strings,
        reasoning_budget,
        logprobs,
        ..Default::default()
    };

    // Metal-only: reuse the exact messages normalized alongside the CPU
    // prompt, so role/content validation and allocation happen once.
    #[cfg(feature = "metal-gpu")]
    let chat_messages = _normalized_messages;
    // CPU-only builds never render `_normalized_messages` (the CPU
    // closures below capture only `cpu_model`/`prompt`/`gen_cfg`) -- drop
    // it here instead of letting it ride, unused, across the
    // `spawn_blocking(...).await` that follows.
    #[cfg(not(feature = "metal-gpu"))]
    drop(_normalized_messages);

    let model = state.model.clone();

    // Compute shared response metadata before branching on stream flag.
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let seq = state.request_counter.fetch_add(1, Ordering::Relaxed);
    let response_id = format!("chatcmpl-{created}-{seq}");

    if stream {
        // --- Streaming path ---
        //
        // `generate_streaming` is a synchronous blocking function.  We run it
        // on the blocking thread pool and feed incremental deltas into an
        // unbounded MPSC channel.  The async SSE handler drains the channel
        // and converts each message to an OpenAI `chat.completion.chunk` event.
        // An unbounded channel is acceptable here because the channel depth is
        // bounded by `max_tokens` (capped at `max_tokens_cap`): the producer
        // sends at most one `Delta` per generated token and generation halts at
        // the cap, so the worst-case buffer is a few thousand short strings —
        // the same order the non-streaming path already holds as one buffered
        // string.
        //
        // Disconnect cancellation (ADR-080 C2, #744): `cancel_guard` is
        // moved into the `body_stream` closure below, so it drops the
        // instant axum drops this response's stream (client disconnect).
        // Dropping it flips `cancel_rx` to `true`, which both backends
        // poll independently of `on_token` (before prefill, immediately
        // after prefill, and at the top of every decode iteration) —
        // closing the gap the old comment here used to document as "a
        // future refinement".
        let (tx, rx) = futures::channel::mpsc::unbounded::<StreamMsg>();
        let (cancel_guard, cancel_rx) = lattice_inference::serve::cancel_pair();

        let stream_id = response_id.clone();
        let stream_model = state.model_id.clone();

        // Both backends funnel their result through this closure so the
        // "generated_tokens > decode_token_budget invariant, then
        // finish_reason_for" logic is written exactly once and shared by
        // CPU and Metal.
        let finish_streaming = {
            let tx = tx.clone();
            move |output: GenerateOutput| {
                let budget = decode_token_budget(max_tokens, reasoning_budget);
                if output.generated_tokens > budget {
                    eprintln!(
                        "generation invariant violation: generated_tokens={} max_tokens={} reasoning_budget={:?}",
                        output.generated_tokens, max_tokens, reasoning_budget
                    );
                    let _ = tx.unbounded_send(StreamMsg::Failed);
                } else {
                    let finish_reason = finish_reason_for(&output);
                    let _ = tx.unbounded_send(StreamMsg::Done { finish_reason });
                }
            }
        };

        match model {
            ModelBackend::Cpu(cpu_model) => {
                // Delegates through the SAME shared helper the test-only
                // `CpuFakeGenerate` arm below uses -- only the generation call itself
                // (`cpu_model.generate_streaming_with_cancel` here, an
                // injected closure there) differs; the `should_cancel`
                // predicate is constructed once, by the helper, for both.
                #[allow(clippy::type_complexity)]
                let generate: Arc<
                    dyn Fn(
                            &str,
                            &lattice_inference::model::qwen35_config::GenerateConfig,
                            &mut dyn FnMut(&str) -> bool,
                            &mut dyn FnMut() -> bool,
                        )
                            -> Result<GenerateOutput, lattice_inference::error::InferenceError>
                        + Send
                        + Sync,
                > = Arc::new(move |p, c, on_token, should_cancel| {
                    cpu_model.generate_streaming_with_cancel(p, c, on_token, should_cancel)
                });
                spawn_cpu_style_streaming_generation(
                    tx,
                    cancel_rx,
                    prompt,
                    gen_cfg,
                    generate,
                    finish_streaming,
                );
            }
            #[cfg(feature = "metal-gpu")]
            ModelBackend::Metal { handle, .. } => {
                // #939: submit synchronously, before any SSE response is
                // built. `handle.submit` runs `MetalWorkerClient::submit`'s
                // admission check (#932) directly on this task -- not
                // inside the `tokio::spawn` below -- so an admission
                // rejection propagates via `?` as an ordinary
                // `ApiError::ServiceUnavailable` (503) before this
                // handler ever returns `Sse::new(...)`. Only draining the
                // already-admitted job happens in the detached task.
                let mut rx = handle.submit(chat_messages, gen_cfg, cancel_rx)?;
                tokio::spawn(async move {
                    let tx_delta = tx.clone();
                    let result = MetalHandle::drain(&mut rx, move |delta| {
                        tx_delta
                            .unbounded_send(StreamMsg::Delta(delta.to_string()))
                            .is_ok()
                    })
                    .await;
                    match result {
                        Ok(output) => finish_streaming(output),
                        Err(e) => {
                            eprintln!("generation error (streaming, metal): {e:?}");
                            let _ = tx.unbounded_send(StreamMsg::Failed);
                        }
                    }
                });
            }
            // ADR-080 C2: goes
            // through the exact same `spawn_cpu_style_streaming_generation`
            // helper as the real `Cpu` arm above -- there is no separate
            // `should_cancel` construction here to leave un-mutated. Only
            // the injected `generate` closure differs; this is the seam a
            // post-drop cancellation probe test substitutes.
            #[cfg(all(feature = "test-utils", test))]
            ModelBackend::CpuFakeGenerate { generate, .. } => {
                spawn_cpu_style_streaming_generation(
                    tx,
                    cancel_rx,
                    prompt,
                    gen_cfg,
                    generate,
                    finish_streaming,
                );
            }
        }

        // Build the SSE stream.
        //
        // Event order (OpenAI spec):
        //   1. Role chunk: `delta: {"role":"assistant"}`, finish_reason absent.
        //   2. One content chunk per Delta: `delta: {"content":"..."}`, finish_reason absent.
        //   3. Finish chunk: `delta: {}`, finish_reason set.
        //   4. Literal `data: [DONE]` sentinel.
        let role_chunk = {
            let chunk = ChatCompletionChunk {
                id: stream_id.clone(),
                object: "chat.completion.chunk",
                created,
                model: stream_model.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: Some("assistant"),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            let data = serde_json::to_string(&chunk).unwrap_or_default();
            Ok::<Event, std::convert::Infallible>(Event::default().data(data))
        };

        // Map each StreamMsg from the channel into one or two SSE events.
        let body_stream = rx.flat_map(move |msg| {
            // Keeps `cancel_guard` alive for exactly as long as this
            // stream is: the moment axum drops the whole SSE response
            // (client disconnect), this closure -- and the guard moved
            // into it -- drops too, flipping `cancel_rx` to `true`.
            let _cancel_guard_tied_to_stream_lifetime = &cancel_guard;
            let id = stream_id.clone();
            let mdl = stream_model.clone();
            match msg {
                StreamMsg::Delta(text) => {
                    let chunk = ChatCompletionChunk {
                        id,
                        object: "chat.completion.chunk",
                        created,
                        model: mdl,
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: Some(text),
                            },
                            finish_reason: None,
                        }],
                    };
                    let data = serde_json::to_string(&chunk).unwrap_or_default();
                    let events: Vec<Result<Event, std::convert::Infallible>> =
                        vec![Ok(Event::default().data(data))];
                    futures::stream::iter(events)
                }
                StreamMsg::Done { finish_reason } => {
                    let chunk = ChatCompletionChunk {
                        id,
                        object: "chat.completion.chunk",
                        created,
                        model: mdl,
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some(finish_reason),
                        }],
                    };
                    let data = serde_json::to_string(&chunk).unwrap_or_default();
                    let events: Vec<Result<Event, std::convert::Infallible>> = vec![
                        Ok(Event::default().data(data)),
                        Ok(Event::default().data("[DONE]")),
                    ];
                    futures::stream::iter(events)
                }
                StreamMsg::Failed => {
                    // The producer already logged the specific cause; keep
                    // client-visible detail generic while making failure
                    // distinguishable from a genuine stop condition.
                    let data = serde_json::json!({
                        "error": {
                            "message": "inference failed",
                            "type": "server_error",
                            "code": "internal_error",
                            "param": null,
                        }
                    })
                    .to_string();
                    let events: Vec<Result<Event, std::convert::Infallible>> = vec![
                        Ok(Event::default().data(data)),
                        Ok(Event::default().data("[DONE]")),
                    ];
                    futures::stream::iter(events)
                }
            }
        });

        let sse_stream = futures::stream::once(async move { role_chunk }).chain(body_stream);

        Ok(Sse::new(sse_stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // --- Non-streaming path (CPU leg byte-identical to the original) ---
        let output = match model {
            ModelBackend::Cpu(cpu_model) => {
                // `generate` is CPU-bound blocking work; run it on the blocking thread pool.
                tokio::task::spawn_blocking(move || cpu_model.generate(&prompt, &gen_cfg))
                    .await
                    .map_err(|e| {
                        eprintln!("task join error: {e}");
                        ApiError::Internal {
                            message: "inference failed".to_string(),
                        }
                    })?
                    .map_err(|e| {
                        eprintln!("generation error: {e}");
                        ApiError::Internal {
                            message: "inference failed".to_string(),
                        }
                    })?
            }
            #[cfg(feature = "metal-gpu")]
            ModelBackend::Metal { handle, .. } => handle
                .generate_streaming(chat_messages, gen_cfg, |_delta| true)
                .await
                // Preserve admission 503s and request-dependent worker
                // rejections (including image geometry/context 400s).
                .map_err(map_metal_generation_error)?,
            // ADR-080 C2 added
            // this variant for the streaming arm's cancellation probe
            // only, so non-streaming used to bypass the injected
            // `generate` closure entirely and delegate straight to the
            // real tiny model. Issue #828's field-level parity rows need
            // a NON-streaming seam too (deterministic content/usage
            // counts for `FieldExpectation::Eq` checks), so this now
            // goes through the exact same injected closure the
            // streaming arm uses -- `on_token`/`should_cancel` are
            // no-ops here (this arm is never the cancellation probe's
            // concern), matching how `model.generate()` itself has no
            // early-stop/cancel hooks either.
            #[cfg(all(feature = "test-utils", test))]
            ModelBackend::CpuFakeGenerate { generate, .. } => {
                tokio::task::spawn_blocking(move || {
                    generate(&prompt, &gen_cfg, &mut |_delta: &str| true, &mut || false)
                })
                .await
                .map_err(|e| {
                    eprintln!("task join error: {e}");
                    ApiError::Internal {
                        message: "inference failed".to_string(),
                    }
                })?
                .map_err(|e| {
                    eprintln!("generation error: {e}");
                    ApiError::Internal {
                        message: "inference failed".to_string(),
                    }
                })?
            }
        };

        // Distinguish "hit token cap" from "natural stop" (EOS / stop token / stop string).
        // `GenerateOutput.stopped` carries the explicit stop reason set by the library.
        // Log and return 500 if the invariant is violated.
        let budget = decode_token_budget(max_tokens, reasoning_budget);
        if output.generated_tokens > budget {
            eprintln!(
                "generation invariant violation: generated_tokens={} max_tokens={} reasoning_budget={:?}",
                output.generated_tokens, max_tokens, reasoning_budget
            );
            return Err(ApiError::Internal {
                message: "inference failed".to_string(),
            });
        }
        let finish_reason = finish_reason_for(&output);

        // #585: only render logprobs (and touch the tokenizer for it) when
        // the request actually asked for them — `logprobs` is `None` on
        // every other request, so this is a no-op on the default path.
        let choice_logprobs = logprobs
            .is_some()
            .then(|| build_choice_logprobs(state.model.tokenizer(), &output.token_logprobs));

        let response = ChatCompletionResponse {
            id: response_id,
            object: "chat.completion".to_string(),
            created,
            model: state.model_id.clone(),
            choices: vec![Choice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".to_string(),
                    content: output.text.clone(),
                },
                finish_reason: finish_reason.to_string(),
                logprobs: choice_logprobs,
            }],
            usage: Usage {
                prompt_tokens: output.prompt_tokens,
                completion_tokens: output.generated_tokens,
                total_tokens: output.prompt_tokens + output.generated_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

// -----------------------------------------------------------------------
// Embeddings
// -----------------------------------------------------------------------

/// `POST /v1/embeddings`: pooled text and image embeddings, OpenAI
/// `embeddings`-shaped request/response. See
/// `lattice_inference::serve::embeddings` for the wire contract and pooled
/// execution this handler wires into axum.
pub async fn embeddings(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
) -> Result<Response, ApiError> {
    use lattice_inference::serve::embeddings::{
        EmbeddingsRequest, embed_items, normalize_embedding_items, parse_pooling,
    };

    lattice_inference::serve::require_json_content_type(&headers)?;

    let bytes = axum::body::to_bytes(body, REQUEST_BODY_LIMIT_BYTES)
        .await
        .map_err(|err| {
            let is_length_limit = std::error::Error::source(&err)
                .is_some_and(<dyn std::error::Error>::is::<http_body_util::LengthLimitError>);
            if is_length_limit {
                return ApiError::PayloadTooLarge {
                    message: "request body exceeds 1 MiB limit".to_string(),
                };
            }
            eprintln!("invalid request body: {err}");
            ApiError::BadRequest {
                message: "invalid JSON request body".to_string(),
                code: "invalid_request_body",
            }
        })?;

    let req: EmbeddingsRequest = serde_json::from_slice(&bytes).map_err(|err| {
        eprintln!("invalid request body: {err}");
        ApiError::BadRequest {
            message: "invalid JSON request body".to_string(),
            code: "invalid_request_body",
        }
    })?;

    let pooling = parse_pooling(req.pooling.as_deref())?;
    let items = normalize_embedding_items(req.input.into_items())?;

    let Some(embedder) = state.embedding_model.clone() else {
        return Err(ApiError::BadRequest {
            message: "embeddings require a loaded vision-language checkpoint; restart this \
                      server with `--model` pointed at a vision-language checkpoint directory \
                      to enable this route"
                .to_string(),
            code: "vision_unsupported",
        });
    };
    let model_id = state.model_id.clone();

    let (data, usage) = tokio::task::spawn_blocking(move || embed_items(&embedder, items, pooling))
        .await
        .map_err(|e| {
            eprintln!("task join error: {e}");
            ApiError::Internal {
                message: "inference failed".to_string(),
            }
        })??;

    Ok(
        Json(lattice_inference::serve::embeddings::EmbeddingsResponse {
            object: "list",
            data,
            model: model_id,
            usage,
        })
        .into_response(),
    )
}

// -----------------------------------------------------------------------
// Router
// -----------------------------------------------------------------------

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .layer(DefaultBodyLimit::max(REQUEST_BODY_LIMIT_BYTES))
        .with_state(state)
}

// -----------------------------------------------------------------------
// Tests — pure helper functions; no model construction needed
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use lattice_inference::forward::metal_qwen35::ChatRole;

    /// #832 checklist: "Add a Metal-feature test asserting an explicit
    /// common-worker marker, so a silently-reintroduced per-binary
    /// fallback cannot pass green." `MetalHandle::client`'s field type
    /// is private, so this test can only be written from inside
    /// `mod serve` (same module, so private-field access is allowed) —
    /// it constructs a `MetalHandle` directly from a
    /// `lattice_inference::serve::metal_worker::MetalWorkerClient`,
    /// which only type-checks if `MetalHandle` still wraps that exact
    /// shared type. A private per-binary fallback worker (any type
    /// other than the shared `MetalWorkerClient`) would fail to compile
    /// here, not just fail some behavioral assertion at runtime.
    /// `test-utils`-gated: `test_client_and_jobs` (the only way to
    /// obtain a real `MetalWorkerClient` without a live GPU worker
    /// thread) lives behind that feature — see its own doc comment.
    #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
    #[test]
    fn metal_handle_is_backed_by_the_shared_metal_worker_client() {
        fn build_from_shared_client(
            client: lattice_inference::serve::metal_worker::MetalWorkerClient,
        ) -> MetalHandle {
            MetalHandle { client }
        }
        let (client, _jobs_rx) = lattice_inference::serve::metal_worker::test_client_and_jobs();
        let _handle: MetalHandle = build_from_shared_client(client);
    }

    /// #939 regression coverage: the unified `lattice serve` HTTP
    /// adapter's admission ordering. Before this fix, the Metal
    /// streaming arm called `MetalWorkerClient::submit` (the ONE way
    /// admission (#932) can reject a request) INSIDE a detached
    /// `tokio::spawn` task, after `chat_completions` had already
    /// returned `Sse::new(...).into_response()` -- so an admission
    /// rejection surfaced as a normal-looking stream that immediately
    /// ended, never as HTTP 503. The non-streaming arm separately
    /// collapsed `ApiError::ServiceUnavailable` into a generic 500. Both
    /// arms are driven here through the REAL `router()` + a real
    /// worker thread (`spawn_fake_with_cap`, cap=1, `generate` blocked
    /// on a channel so exactly one request is provably "in flight"),
    /// mirroring `lattice_serve.rs`'s own
    /// `real_router_admission_cap::chat_completions_returns_503_json_envelope_when_admission_cap_reached`
    /// -- that test alone did not catch this bug because it only
    /// covers the daemon's separate, already-correct adapter.
    #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
    mod admission_cap_939 {
        use super::*;
        use axum::body::Body;
        use axum::http::StatusCode;
        use lattice_inference::serve::metal_worker::{ContextWindowPolicy, spawn_fake_with_cap};
        use std::sync::mpsc as std_mpsc;
        use tower::ServiceExt as _;

        /// A real (if tiny) tokenizer -- NOT a hand-rolled minimal
        /// vocab. `lattice_serve.rs`'s equivalent
        /// `real_router_admission_cap::single_slot_blocking_state` uses
        /// the same `test_support::tiny_zero_model` tokenizer for
        /// exactly this reason: a hand-rolled single-entry vocab (e.g.
        /// just `{"hi": 0}`) lacks the byte-level fallback tokens a
        /// real BPE tokenizer's byte-encoder table relies on, so
        /// tokenizing the full rendered ChatML prompt against it can
        /// silently misbehave in ways a real tokenizer never does.
        fn tiny_tokenizer() -> lattice_inference::tokenizer::bpe::BpeTokenizer {
            lattice_inference::model::qwen35::test_support::tiny_zero_model()
                .tokenizer()
                .clone()
        }

        /// A real worker thread (cap=1) whose `generate` blocks on
        /// `unblock_rx.recv()` until the test releases it, so exactly
        /// one request can be "in flight" for as long as the test
        /// needs -- long enough to prove a second concurrent request is
        /// rejected by admission rather than racing to observe an
        /// already-freed slot. Mirrors `lattice_serve.rs`'s
        /// `real_router_admission_cap::single_slot_blocking_state`.
        fn single_slot_blocking_state() -> (AppState, std_mpsc::Sender<()>, std_mpsc::Receiver<()>)
        {
            let (unblock_tx, unblock_rx) = std_mpsc::channel::<()>();
            let unblock_rx = std::sync::Mutex::new(unblock_rx);
            let (started_tx, started_rx) = std_mpsc::channel::<()>();
            let client = spawn_fake_with_cap(
                1,
                ContextWindowPolicy::PromptAndMaxTokens,
                4096,
                tiny_tokenizer(),
                move |_messages, _cfg, prompt_tokens, _on_token, _should_cancel| {
                    let _ = started_tx.send(());
                    // Blocks the dedicated fake-worker OS thread (not
                    // the tokio runtime) until the test explicitly lets
                    // it proceed.
                    let _ = unblock_rx.lock().unwrap().recv();
                    Ok(GenerateOutput {
                        text: String::new(),
                        token_ids: vec![],
                        prompt_tokens,
                        generated_tokens: 0,
                        stopped: true,
                        stop_reason: None,
                        token_logprobs: vec![],
                    })
                },
            );
            let state = AppState {
                model: ModelBackend::Metal {
                    handle: MetalHandle { client },
                    tokenizer: Arc::new(tiny_tokenizer()),
                    max_context: 4096,
                },
                default_max_tokens: 16,
                max_tokens_cap: 4096,
                model_id: "test-model".to_string(),
                request_counter: Arc::new(AtomicU64::new(0)),
                embedding_model: None,
            };
            (state, unblock_tx, started_rx)
        }

        fn chat_request(stream: bool) -> axum::http::Request<Body> {
            let body = if stream {
                r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true}"#
            } else {
                r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#
            };
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("fixture request must build")
        }

        /// Asserts the shared 503 `server_busy` JSON envelope, AND that
        /// no SSE response was ever committed (issue #939's exact
        /// regression for the streaming arm: a 200 SSE response whose
        /// body happened to end immediately would previously slip past
        /// a status-code-only assertion).
        async fn assert_503_server_busy_no_sse(response: axum::http::Response<Body>, case: &str) {
            assert_eq!(
                response.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{case}: a request submitted while the admission cap is full must be \
                     rejected with HTTP 503, fail-fast, before any response -- streaming \
                     or not -- is committed"
            );
            let content_type = response
                .headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_string();
            assert!(
                !content_type.contains("text/event-stream"),
                "{case}: a 503 rejection must never carry an SSE content-type -- that \
                     would mean a streaming response had already committed before \
                     admission was checked: {content_type}"
            );
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("503 response body must be readable");
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
                panic!("{case}: 503 response must be JSON, not an SSE body: {e}")
            });
            assert_eq!(value["error"]["code"], "server_busy", "{case}: {value}");
            assert_eq!(value["error"]["type"], "server_error", "{case}: {value}");
            assert!(value["error"]["param"].is_null(), "{case}: {value}");
            assert!(
                !value["error"]["message"]
                    .as_str()
                    .unwrap_or_default()
                    .is_empty(),
                "{case}: 503 envelope must carry a human-readable message: {value}"
            );
        }

        #[tokio::test]
        async fn chat_completions_streaming_returns_503_before_sse_commit_at_cap() {
            let (state, unblock_tx, started_rx) = single_slot_blocking_state();
            let app = router(state);

            let app1 = app.clone();
            let handle1 = tokio::spawn(async move { app1.oneshot(chat_request(false)).await });
            tokio::task::spawn_blocking(move || started_rx.recv())
                .await
                .expect("blocking wait must not panic")
                .expect("request 1's worker-thread generate() must signal it started");

            let response2 = app
                .clone()
                .oneshot(chat_request(true))
                .await
                .expect("router must produce a response, not a transport error");
            assert_503_server_busy_no_sse(response2, "stream:true").await;

            unblock_tx.send(()).expect("unblock send must succeed");
            let response1 = handle1
                .await
                .expect("request 1's task must not panic")
                .expect("router must produce a response, not a transport error");
            assert_eq!(
                response1.status(),
                StatusCode::OK,
                "request 1 itself was never over any cap and must succeed normally"
            );
        }

        #[tokio::test]
        async fn chat_completions_non_streaming_returns_503_server_busy_not_500_at_cap() {
            let (state, unblock_tx, started_rx) = single_slot_blocking_state();
            let app = router(state);

            let app1 = app.clone();
            let handle1 = tokio::spawn(async move { app1.oneshot(chat_request(false)).await });
            tokio::task::spawn_blocking(move || started_rx.recv())
                .await
                .expect("blocking wait must not panic")
                .expect("request 1's worker-thread generate() must signal it started");

            let response2 = app
                .clone()
                .oneshot(chat_request(false))
                .await
                .expect("router must produce a response, not a transport error");
            assert_503_server_busy_no_sse(response2, "non-streaming").await;

            unblock_tx.send(()).expect("unblock send must succeed");
            let response1 = handle1
                .await
                .expect("request 1's task must not panic")
                .expect("router must produce a response, not a transport error");
            assert_eq!(
                response1.status(),
                StatusCode::OK,
                "request 1 itself was never over any cap and must succeed normally"
            );
        }
    }

    #[test]
    fn validate_max_tokens_rejects_zero() {
        let err = validate_max_tokens(Some(0), None, 256, 4096).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_max_tokens",
                ..
            }
        ));
    }

    #[test]
    fn validate_max_tokens_rejects_above_cap() {
        let err = validate_max_tokens(Some(9999), None, 256, 4096).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "max_tokens_exceeds_limit",
                ..
            }
        ));
    }

    #[test]
    fn validate_max_tokens_uses_default_when_absent() {
        assert_eq!(validate_max_tokens(None, None, 128, 4096).unwrap(), 128);
    }

    #[test]
    fn validate_max_tokens_alias_agrees() {
        assert_eq!(
            validate_max_tokens(Some(512), Some(512), 256, 4096).unwrap(),
            512
        );
    }

    #[test]
    fn validate_max_tokens_alias_conflict_rejected() {
        let err = validate_max_tokens(Some(100), Some(200), 256, 4096).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_request",
                ..
            }
        ));
    }

    #[test]
    fn validate_temperature_rejects_negative() {
        let err = validate_temperature(Some(-0.1)).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_temperature",
                ..
            }
        ));
    }

    #[test]
    fn validate_temperature_rejects_above_two() {
        let err = validate_temperature(Some(2.1)).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_temperature",
                ..
            }
        ));
    }

    #[test]
    fn validate_temperature_accepts_boundary() {
        assert_eq!(validate_temperature(Some(0.0)).unwrap(), 0.0);
        assert_eq!(validate_temperature(Some(2.0)).unwrap(), 2.0);
    }

    #[test]
    fn validate_top_p_rejects_zero() {
        let err = validate_top_p(Some(0.0)).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_top_p",
                ..
            }
        ));
    }

    #[test]
    fn validate_top_p_rejects_above_one() {
        let err = validate_top_p(Some(1.1)).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_top_p",
                ..
            }
        ));
    }

    #[test]
    fn validate_top_p_accepts_one() {
        assert_eq!(validate_top_p(Some(1.0)).unwrap(), 1.0);
    }

    #[test]
    fn chat_template_multi_message_chatml() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: MessageContent::Text("Be helpful.".to_string()),
            },
            Message {
                role: "user".to_string(),
                content: MessageContent::Text("Hello".to_string()),
            },
        ];
        let prompt = format_chat_template(&to_chat_messages(&messages).unwrap());
        assert!(prompt.contains("<|im_start|>system\nBe helpful.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    // Role/content-part rejection cases are covered by the
    // `to_chat_messages_rejects_*` tests below -- `to_chat_messages` is
    // the sole validation entry point the ChatML renderer sits behind
    // (#668), so there is no second renderer left to test separately.

    // Exercises finish_reason_for via the real helper function used by the handler.
    // A cap-reached output has stopped=false → "length".
    // A stop-condition output has stopped=true → "stop".
    #[test]
    fn finish_reason_length_only_at_cap() {
        use lattice_inference::model::qwen35_config::GenerateOutput;
        let cap = GenerateOutput {
            text: String::new(),
            token_ids: vec![],
            prompt_tokens: 10,
            generated_tokens: 64,
            stopped: false,
            stop_reason: Some(lattice_inference::StopReason::Length),
            token_logprobs: vec![],
        };
        assert_eq!(super::finish_reason_for(&cap), "length");

        let natural = GenerateOutput {
            text: "hello".into(),
            token_ids: vec![1, 2, 3],
            prompt_tokens: 10,
            generated_tokens: 3,
            stopped: true,
            stop_reason: Some(lattice_inference::StopReason::Eos),
            token_logprobs: vec![],
        };
        assert_eq!(super::finish_reason_for(&natural), "stop");
    }

    // M1 regression: a stop-string hit at exactly max_new_tokens must yield "stop",
    // not "length". The old token-count formula (generated == cap → "length") would
    // mislabel this case because the stop-completing token is included in generated_ids
    // before the stop is detected.
    //
    // This test calls the real finish_reason_for helper. It is RED when
    // finish_reason_for reverts to the old `generated_tokens == max_tokens` formula.
    #[test]
    fn finish_reason_stop_string_at_cap_is_stop_not_length() {
        use lattice_inference::model::qwen35_config::GenerateOutput;
        let max_tokens: usize = 4;
        // stop-string hit at exactly the token budget:
        // stopped=true because a stop string matched; generated_tokens==max_tokens
        // because the matching token is included in generated_ids before truncation.
        let output = GenerateOutput {
            text: "hi".into(),
            token_ids: vec![1, 2, 3, 4],
            prompt_tokens: 5,
            generated_tokens: max_tokens,
            stopped: true,
            stop_reason: Some(lattice_inference::StopReason::Eos),
            token_logprobs: vec![],
        };
        assert_eq!(
            super::finish_reason_for(&output),
            "stop",
            "stop-string hit at cap must yield finish_reason=stop, not length"
        );
    }

    // Natural length cap (no stop condition) must still yield "length".
    #[test]
    fn finish_reason_natural_length_cap_is_length() {
        use lattice_inference::model::qwen35_config::GenerateOutput;
        let output = GenerateOutput {
            text: "hi".into(),
            token_ids: vec![1, 2, 3, 4],
            prompt_tokens: 5,
            generated_tokens: 4,
            stopped: false,
            stop_reason: Some(lattice_inference::StopReason::Length),
            token_logprobs: vec![],
        };
        assert_eq!(super::finish_reason_for(&output), "length");
    }

    #[test]
    fn reject_unsupported_stream_true_ok() {
        // stream=true is now handled by the streaming path and must NOT be
        // rejected by reject_unsupported.
        let req = ChatCompletionRequest {
            model: Some("m".to_string()),
            messages: vec![],
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            reasoning_budget: None,
            stream: Some(true),
            stop: None,
            seed: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            logprobs: None,
            top_logprobs: None,
            n: None,
        };
        assert!(reject_unsupported(&req).is_ok());
    }

    #[test]
    fn reject_unsupported_stream_and_logprobs_rejected() {
        // #585: logprobs is implemented on the non-streaming path only;
        // combined with stream: true it must be rejected, not silently
        // ignored.
        let req = ChatCompletionRequest {
            stream: Some(true),
            logprobs: Some(true),
            ..bare_req()
        };
        let err = reject_unsupported(&req).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "unsupported_feature",
                ..
            }
        ));
    }

    // -----------------------------------------------------------------------
    // ChatCompletionChunk serialization
    // -----------------------------------------------------------------------

    #[test]
    fn chunk_content_delta_serializes_correctly() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-1-0".to_string(),
            object: "chat.completion.chunk",
            created: 1_000_000,
            model: "test-model".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: Some("Hello".to_string()),
                },
                finish_reason: None,
            }],
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(
            json.contains("\"object\":\"chat.completion.chunk\""),
            "must contain object field"
        );
        assert!(
            json.contains("\"delta\":{\"content\":\"Hello\"}"),
            "delta must contain only content when role is None"
        );
        // finish_reason must be absent (not null) when None
        assert!(
            !json.contains("finish_reason"),
            "finish_reason must be omitted when None"
        );
    }

    #[test]
    fn chunk_finish_delta_serializes_correctly() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-1-0".to_string(),
            object: "chat.completion.chunk",
            created: 1_000_000,
            model: "test-model".to_string(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop"),
            }],
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(
            json.contains("\"finish_reason\":\"stop\""),
            "finish chunk must include finish_reason"
        );
        // delta should be empty object since both role and content are None
        assert!(
            json.contains("\"delta\":{}"),
            "finish chunk delta must be empty object"
        );
    }

    #[test]
    fn reject_unsupported_n_gt_1() {
        let req = ChatCompletionRequest {
            model: Some("m".to_string()),
            messages: vec![],
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            reasoning_budget: None,
            stream: None,
            stop: None,
            seed: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            logprobs: None,
            top_logprobs: None,
            n: Some(3),
        };
        let err = reject_unsupported(&req).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "unsupported_feature",
                ..
            }
        ));
    }

    #[test]
    fn reject_unsupported_response_format_json() {
        let req = ChatCompletionRequest {
            model: Some("m".to_string()),
            messages: vec![],
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            reasoning_budget: None,
            stream: None,
            stop: None,
            seed: None,
            response_format: Some(ResponseFormat {
                r#type: "json_object".to_string(),
                json_schema: None,
            }),
            tools: None,
            tool_choice: None,
            logprobs: None,
            top_logprobs: None,
            n: None,
        };
        let err = reject_unsupported(&req).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "unsupported_feature",
                ..
            }
        ));
    }

    // -----------------------------------------------------------------------
    // reject_unsupported — remaining fields
    // -----------------------------------------------------------------------

    fn bare_req() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: Some("m".to_string()),
            messages: vec![],
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            reasoning_budget: None,
            stream: None,
            stop: None,
            seed: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            logprobs: None,
            top_logprobs: None,
            n: None,
        }
    }

    #[test]
    fn reject_unsupported_tools_rejected() {
        let req = ChatCompletionRequest {
            tools: Some(serde_json::json!([])),
            ..bare_req()
        };
        let err = reject_unsupported(&req).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "unsupported_feature",
                ..
            }
        ));
    }

    #[test]
    fn reject_unsupported_tool_choice_rejected() {
        let req = ChatCompletionRequest {
            tool_choice: Some(serde_json::json!("auto")),
            ..bare_req()
        };
        let err = reject_unsupported(&req).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "unsupported_feature",
                ..
            }
        ));
    }

    #[test]
    fn reject_unsupported_logprobs_true_ok() {
        // #585: logprobs is now implemented on the non-streaming path, so a
        // standalone `logprobs: true` (no `stream: true`) must be accepted
        // here — validation of the value itself is `validate_logprobs`'s job.
        let req = ChatCompletionRequest {
            logprobs: Some(true),
            ..bare_req()
        };
        assert!(reject_unsupported(&req).is_ok());
    }

    #[test]
    fn reject_unsupported_stop_now_accepted() {
        // stop is no longer rejected by reject_unsupported; it is parsed separately.
        let req = ChatCompletionRequest {
            stop: Some(serde_json::json!("</s>")),
            ..bare_req()
        };
        assert!(reject_unsupported(&req).is_ok());
    }

    // -----------------------------------------------------------------------
    // parse_stop_strings
    // -----------------------------------------------------------------------

    #[test]
    fn parse_stop_strings_null_gives_empty() {
        assert_eq!(parse_stop_strings(&None).unwrap(), Vec::<String>::new());
        assert_eq!(
            parse_stop_strings(&Some(serde_json::Value::Null)).unwrap(),
            Vec::<String>::new()
        );
    }

    #[test]
    fn parse_stop_strings_single_string_gives_vec_of_one() {
        let v = parse_stop_strings(&Some(serde_json::json!("</s>"))).unwrap();
        assert_eq!(v, vec!["</s>".to_string()]);
    }

    #[test]
    fn parse_stop_strings_array_of_two_accepted() {
        let v = parse_stop_strings(&Some(serde_json::json!(["</s>", "\nUser:"]))).unwrap();
        assert_eq!(v, vec!["</s>".to_string(), "\nUser:".to_string()]);
    }

    #[test]
    fn parse_stop_strings_empty_array_rejected() {
        let err = parse_stop_strings(&Some(serde_json::json!([]))).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_stop",
                ..
            }
        ));
    }

    #[test]
    fn parse_stop_strings_array_over_four_rejected() {
        let err =
            parse_stop_strings(&Some(serde_json::json!(["a", "b", "c", "d", "e"]))).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_stop",
                ..
            }
        ));
    }

    #[test]
    fn parse_stop_strings_array_with_number_rejected() {
        let err = parse_stop_strings(&Some(serde_json::json!(["ok", 42]))).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_stop",
                ..
            }
        ));
    }

    #[test]
    fn parse_stop_strings_empty_string_element_rejected() {
        let err = parse_stop_strings(&Some(serde_json::json!(["ok", ""]))).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_stop",
                ..
            }
        ));
    }

    #[test]
    fn parse_stop_strings_empty_string_scalar_rejected() {
        let err = parse_stop_strings(&Some(serde_json::json!(""))).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_stop",
                ..
            }
        ));
    }

    #[test]
    fn parse_stop_strings_array_exactly_four_accepted() {
        let v = parse_stop_strings(&Some(serde_json::json!(["a", "b", "c", "d"]))).unwrap();
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn reject_unsupported_stream_false_ok() {
        // stream=false must not trigger a rejection.
        let req = ChatCompletionRequest {
            stream: Some(false),
            ..bare_req()
        };
        assert!(reject_unsupported(&req).is_ok());
    }

    #[test]
    fn reject_unsupported_n_1_ok() {
        let req = ChatCompletionRequest {
            n: Some(1),
            ..bare_req()
        };
        assert!(reject_unsupported(&req).is_ok());
    }

    #[test]
    fn reject_unsupported_response_format_text_ok() {
        let req = ChatCompletionRequest {
            response_format: Some(ResponseFormat {
                r#type: "text".to_string(),
                json_schema: None,
            }),
            ..bare_req()
        };
        assert!(reject_unsupported(&req).is_ok());
    }

    #[test]
    fn reject_unsupported_logprobs_false_ok() {
        let req = ChatCompletionRequest {
            logprobs: Some(false),
            ..bare_req()
        };
        assert!(reject_unsupported(&req).is_ok());
    }

    // -----------------------------------------------------------------------
    // validate_max_tokens — additional edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn validate_max_tokens_at_exactly_cap_ok() {
        assert_eq!(
            validate_max_tokens(Some(4096), None, 256, 4096).unwrap(),
            4096
        );
    }

    #[test]
    fn validate_max_tokens_max_completion_only_ok() {
        assert_eq!(
            validate_max_tokens(None, Some(512), 256, 4096).unwrap(),
            512
        );
    }

    // -----------------------------------------------------------------------
    // validate_temperature — default path
    // -----------------------------------------------------------------------

    #[test]
    fn validate_temperature_none_uses_default() {
        assert_eq!(
            validate_temperature(None).unwrap(),
            GenerationDefaults::standard(1).temperature
        );
    }

    // -----------------------------------------------------------------------
    // validate_top_p — default path
    // -----------------------------------------------------------------------

    #[test]
    fn validate_top_p_none_uses_default() {
        assert_eq!(
            validate_top_p(None).unwrap(),
            GenerationDefaults::standard(1).top_p
        );
    }

    // -----------------------------------------------------------------------
    // format_chat_template (via to_chat_messages) — additional rendering cases
    // -----------------------------------------------------------------------

    #[test]
    fn chat_template_user_only() {
        let msgs = vec![Message {
            role: "user".to_string(),
            content: MessageContent::Text("hi".to_string()),
        }];
        let prompt = format_chat_template(&to_chat_messages(&msgs).unwrap());
        assert_eq!(
            prompt,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn chat_template_multi_turn_assistant() {
        let msgs = vec![
            Message {
                role: "user".to_string(),
                content: MessageContent::Text("q1".to_string()),
            },
            Message {
                role: "assistant".to_string(),
                content: MessageContent::Text("a1".to_string()),
            },
            Message {
                role: "user".to_string(),
                content: MessageContent::Text("q2".to_string()),
            },
        ];
        let prompt = format_chat_template(&to_chat_messages(&msgs).unwrap());
        assert!(prompt.contains("<|im_start|>user\nq1<|im_end|>"));
        assert!(prompt.contains("<|im_start|>assistant\na1<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nq2<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn chat_template_content_parts_text_ok() {
        let msgs = vec![Message {
            role: "user".to_string(),
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "hello".to_string(),
                },
                ContentPart::Text {
                    text: " world".to_string(),
                },
            ]),
        }];
        let prompt = format_chat_template(&to_chat_messages(&msgs).unwrap());
        assert!(prompt.contains("<|im_start|>user\nhello world<|im_end|>"));
    }

    // -----------------------------------------------------------------------
    // `to_chat_messages` validation (#661, CPU-available since #668) — the
    // sole validation entry point the ChatML renderer sits behind.
    // -----------------------------------------------------------------------

    #[test]
    fn to_chat_messages_rejects_invalid_role() {
        let messages = vec![Message {
            role: "function".to_string(),
            content: MessageContent::Text("data".to_string()),
        }];
        let err = to_chat_messages(&messages).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_role",
                ..
            }
        ));
    }

    #[test]
    fn to_chat_messages_rejects_tool_role() {
        let messages = vec![Message {
            role: "tool".to_string(),
            content: MessageContent::Text("result".to_string()),
        }];
        let err = to_chat_messages(&messages).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "unsupported_feature",
                ..
            }
        ));
    }

    #[test]
    fn to_chat_messages_rejects_developer_role() {
        let messages = vec![Message {
            role: "developer".to_string(),
            content: MessageContent::Text("system prompt".to_string()),
        }];
        let err = to_chat_messages(&messages).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "unsupported_feature",
                ..
            }
        ));
    }

    #[test]
    fn to_chat_messages_rejects_non_text_content_part() {
        let messages = vec![Message {
            role: "user".to_string(),
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: lattice_inference::serve::contract::ImageUrl {
                    url: "https://example.com/image.png".to_string(),
                    detail: None,
                },
            }]),
        }];
        let err = to_chat_messages(&messages).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "vision_unsupported",
                ..
            }
        ));
    }

    #[test]
    fn to_chat_messages_accepts_valid_roles() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: MessageContent::Text("Be helpful.".to_string()),
            },
            Message {
                role: "user".to_string(),
                content: MessageContent::Text("q1".to_string()),
            },
            Message {
                role: "assistant".to_string(),
                content: MessageContent::Text("a1".to_string()),
            },
        ];
        let chat_messages = to_chat_messages(&messages).unwrap();
        assert_eq!(chat_messages.len(), 3);
        assert_eq!(chat_messages[0].role, ChatRole::System);
        assert_eq!(chat_messages[0].content, "Be helpful.");
        assert_eq!(chat_messages[1].role, ChatRole::User);
        assert_eq!(chat_messages[2].role, ChatRole::Assistant);
    }

    // -----------------------------------------------------------------------
    // Error envelope JSON shape
    // -----------------------------------------------------------------------

    /// Drains an `axum::response::Response` body into a parsed `Value`,
    /// for asserting on the shared `lattice_inference::serve::ApiError`
    /// envelope shape (ADR-080 C2, #782) from this binary's own tests.
    async fn response_json(response: axum::response::Response) -> serde_json::Value {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body reads");
        serde_json::from_slice(&body).expect("response body is valid JSON")
    }

    #[tokio::test]
    async fn error_envelope_bad_request_shape() {
        let err = ApiError::BadRequest {
            message: "test error".to_string(),
            code: "invalid_request",
        };
        // Variant check kept separate so we know err itself was constructed correctly.
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_request",
                ..
            }
        ));
        // Verify the shared ApiError serialises to the OpenAI envelope shape:
        // {"error":{"message":"...","type":"invalid_request_error","code":"...","param":null}}
        let response = ApiError::BadRequest {
            message: "test error".to_string(),
            code: "invalid_request",
        }
        .into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let v = response_json(response).await;
        assert!(v["error"].is_object(), "top-level key must be 'error'");
        assert_eq!(v["error"]["message"], "test error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], "invalid_request");
        assert!(v["error"]["param"].is_null());
    }

    #[tokio::test]
    async fn error_envelope_payload_too_large_shape() {
        let response = ApiError::PayloadTooLarge {
            message: "request body exceeds 1 MiB limit".to_string(),
        }
        .into_response();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let v = response_json(response).await;
        assert_eq!(v["error"]["code"], "request_body_too_large");
    }

    #[tokio::test]
    async fn error_envelope_internal_shape() {
        let response = ApiError::Internal {
            message: "inference failed".to_string(),
        }
        .into_response();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let v = response_json(response).await;
        assert_eq!(v["error"]["type"], "server_error");
        assert_eq!(v["error"]["code"], "internal_error");
    }

    // -----------------------------------------------------------------------
    // message content normalization (via the shared normalize_messages,
    // exercised here through the `to_chat_messages` alias)
    // -----------------------------------------------------------------------

    #[test]
    fn message_text_plain_string() {
        let messages = [Message {
            role: "user".to_string(),
            content: MessageContent::Text("hello".to_string()),
        }];
        assert_eq!(to_chat_messages(&messages).unwrap()[0].content, "hello");
    }

    #[test]
    fn message_text_parts_concatenates() {
        let messages = [Message {
            role: "user".to_string(),
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "foo".to_string(),
                },
                ContentPart::Text {
                    text: "bar".to_string(),
                },
            ]),
        }];
        assert_eq!(to_chat_messages(&messages).unwrap()[0].content, "foobar");
    }

    #[test]
    fn message_text_parts_rejects_image() {
        let messages = [Message {
            role: "user".to_string(),
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: lattice_inference::serve::contract::ImageUrl {
                    url: "https://example.com/image.png".to_string(),
                    detail: None,
                },
            }]),
        }];
        let err = to_chat_messages(&messages).unwrap_err();
        match err {
            ApiError::BadRequest { message, code } => {
                assert_eq!(code, "vision_unsupported");
                assert_eq!(message, "image input requires a vision-capable model");
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn message_text_parts_rejects_unknown_part_type() {
        let messages = [Message {
            role: "user".to_string(),
            content: MessageContent::Parts(vec![ContentPart::Unsupported {
                kind: "file".to_string(),
            }]),
        }];
        let err = to_chat_messages(&messages).unwrap_err();
        match err {
            ApiError::BadRequest { message, code } => {
                assert_eq!(code, "unsupported_feature");
                assert_eq!(
                    message,
                    "content part type 'file' is not supported; only 'text' and 'image_url' \
                         parts are accepted"
                );
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // validate_logprobs (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn validate_logprobs_absent_disables_capture() {
        assert_eq!(validate_logprobs(None, None).unwrap(), None);
    }

    #[test]
    fn validate_logprobs_false_disables_capture() {
        assert_eq!(validate_logprobs(Some(false), None).unwrap(), None);
    }

    #[test]
    fn validate_logprobs_true_no_top_logprobs_defaults_to_zero() {
        // logprobs: true with no top_logprobs still captures the sampled
        // token's own logprob, just with no alternatives.
        assert_eq!(validate_logprobs(Some(true), None).unwrap(), Some(0));
    }

    #[test]
    fn validate_logprobs_true_with_top_logprobs_ok() {
        assert_eq!(validate_logprobs(Some(true), Some(5)).unwrap(), Some(5));
    }

    #[test]
    fn validate_logprobs_top_logprobs_at_boundary_twenty_ok() {
        assert_eq!(validate_logprobs(Some(true), Some(20)).unwrap(), Some(20));
    }

    #[test]
    fn validate_logprobs_top_logprobs_over_twenty_rejected() {
        let err = validate_logprobs(Some(true), Some(21)).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_top_logprobs",
                ..
            }
        ));
    }

    #[test]
    fn validate_logprobs_top_logprobs_without_logprobs_true_rejected() {
        let err = validate_logprobs(None, Some(5)).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_request",
                ..
            }
        ));
    }

    #[test]
    fn validate_logprobs_top_logprobs_with_logprobs_false_rejected() {
        let err = validate_logprobs(Some(false), Some(5)).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_request",
                ..
            }
        ));
    }

    // -----------------------------------------------------------------------
    // render_token_logprob / build_choice_logprobs (#585)
    // -----------------------------------------------------------------------

    /// Tiny in-memory tokenizer for logprob-rendering tests — no merges,
    /// just a fixed id -> token vocabulary large enough to exercise both a
    /// known and an unresolved token id. "Hello"/"world" are plain ASCII in
    /// the printable range the GPT-2 byte table maps to itself, so they
    /// round-trip byte-for-byte through `byte_decode_token[_bytes]`.
    fn logprob_test_tokenizer() -> lattice_inference::tokenizer::bpe::BpeTokenizer {
        let vocab: std::collections::HashMap<String, u32> =
            [("Hello".to_string(), 0u32), ("world".to_string(), 1u32)]
                .into_iter()
                .collect();
        lattice_inference::tokenizer::bpe::BpeTokenizer::from_vocab_and_merges(vocab, vec![])
            .expect("in-memory test vocab must construct")
    }

    #[test]
    fn render_token_logprob_resolves_known_token() {
        let tokenizer = logprob_test_tokenizer();
        let (token, bytes) = render_token_logprob(&tokenizer, 0);
        assert_eq!(token, "Hello");
        assert_eq!(bytes, Some(b"Hello".to_vec()));
    }

    #[test]
    fn render_token_logprob_unresolved_id_fails_closed() {
        // Token id 999 does not exist in the 2-entry test vocab: this must
        // fail closed with a visibly synthetic token and no bytes, never panic.
        let tokenizer = logprob_test_tokenizer();
        let (token, bytes) = render_token_logprob(&tokenizer, 999);
        assert_eq!(token, "<|unresolved_token_999|>");
        assert_eq!(bytes, None);
    }

    #[test]
    fn build_choice_logprobs_shapes_content_and_alternatives() {
        let tokenizer = logprob_test_tokenizer();
        let token_logprobs = vec![
            TokenLogprob {
                token_id: 0,
                logprob: -0.1,
                top: vec![
                    lattice_inference::model::qwen35_config::TopLogprob {
                        token_id: 0,
                        logprob: -0.1,
                    },
                    lattice_inference::model::qwen35_config::TopLogprob {
                        token_id: 1,
                        logprob: -2.3,
                    },
                ],
            },
            TokenLogprob {
                token_id: 1,
                logprob: -0.05,
                top: vec![],
            },
        ];
        let choice_logprobs = build_choice_logprobs(&tokenizer, &token_logprobs);
        assert_eq!(choice_logprobs.content.len(), 2);

        assert_eq!(choice_logprobs.content[0].token, "Hello");
        assert_eq!(choice_logprobs.content[0].logprob, -0.1);
        assert_eq!(choice_logprobs.content[0].top_logprobs.len(), 2);
        assert_eq!(choice_logprobs.content[0].top_logprobs[0].token, "Hello");
        assert_eq!(choice_logprobs.content[0].top_logprobs[1].token, "world");

        assert_eq!(choice_logprobs.content[1].token, "world");
        assert_eq!(choice_logprobs.content[1].logprob, -0.05);
        assert!(choice_logprobs.content[1].top_logprobs.is_empty());
    }

    // -----------------------------------------------------------------------
    // Choice.logprobs — JSON shape (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn choice_logprobs_omitted_from_json_when_none() {
        // The no-logprobs-requested response must be byte-identical to
        // before this feature existed: the key is absent, not `null`.
        let choice = Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: "hi".to_string(),
            },
            finish_reason: "stop".to_string(),
            logprobs: None,
        };
        let json = serde_json::to_string(&choice).unwrap();
        assert!(
            !json.contains("logprobs"),
            "logprobs key must be entirely absent when None, got: {json}"
        );
    }

    #[test]
    fn choice_logprobs_present_when_requested() {
        let choice = Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: "hi".to_string(),
            },
            finish_reason: "stop".to_string(),
            logprobs: Some(ChoiceLogprobs {
                content: vec![TokenLogprobEntry {
                    token: "hi".to_string(),
                    logprob: -0.2,
                    bytes: Some(b"hi".to_vec()),
                    top_logprobs: vec![],
                }],
            }),
        };
        let json = serde_json::to_string(&choice).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["logprobs"]["content"][0]["token"], "hi");
        assert_eq!(v["logprobs"]["content"][0]["logprob"], -0.2);
        assert_eq!(
            v["logprobs"]["content"][0]["bytes"],
            serde_json::json!([104, 105])
        );
        assert_eq!(
            v["logprobs"]["content"][0]["top_logprobs"],
            serde_json::json!([])
        );
    }

    // -----------------------------------------------------------------------
    // Capability-matrix fixtures (#654) — `validate_chat_request` cascade.
    //
    // Each `#[test]` fn name below is a fixture ID cited from
    // `docs/capability-matrix.md`'s Fixture column; `scripts/check-capability-
    // matrix.sh` greps this file for `fn <fixture_id>` and fails the build if
    // a matrix row cites an ID that no longer exists here. These three checks
    // (model-id match, empty messages, last-role-must-be-user) previously ran
    // only inline in `chat_completions` with no dedicated test at all.
    // -----------------------------------------------------------------------

    fn user_msg(text: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: MessageContent::Text(text.to_string()),
        }
    }

    #[test]
    fn cm_serve_model_mismatch_rejected() {
        let req = ChatCompletionRequest {
            model: Some("some-other-model".to_string()),
            messages: vec![user_msg("hi")],
            ..bare_req()
        };
        let err = validate_chat_request(&req, "served-model", 256, 4096).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "model_not_found",
                ..
            }
        ));
    }

    #[test]
    fn cm_serve_model_match_passes_model_check() {
        let req = ChatCompletionRequest {
            model: Some("served-model".to_string()),
            messages: vec![user_msg("hi")],
            ..bare_req()
        };
        assert!(validate_chat_request(&req, "served-model", 256, 4096).is_ok());
    }

    #[test]
    fn cm_serve_empty_messages_rejected() {
        let req = ChatCompletionRequest {
            model: Some("served-model".to_string()),
            messages: vec![],
            ..bare_req()
        };
        let err = validate_chat_request(&req, "served-model", 256, 4096).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_messages",
                ..
            }
        ));
    }

    #[test]
    fn cm_serve_last_message_not_user_rejected() {
        let req = ChatCompletionRequest {
            model: Some("served-model".to_string()),
            messages: vec![
                user_msg("hi"),
                Message {
                    role: "assistant".to_string(),
                    content: MessageContent::Text("hello".to_string()),
                },
            ],
            ..bare_req()
        };
        let err = validate_chat_request(&req, "served-model", 256, 4096).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "invalid_messages",
                ..
            }
        ));
    }

    #[test]
    fn cm_serve_unsupported_feature_rejected_before_model_check() {
        // `reject_unsupported` (tools/n/response_format/stream+logprobs) runs
        // first in the cascade: a request that both targets the wrong model
        // AND asks for `tools` must fail on the tools rejection, not the
        // model-mismatch check, so callers get the more specific error.
        let req = ChatCompletionRequest {
            model: Some("some-other-model".to_string()),
            messages: vec![user_msg("hi")],
            tools: Some(serde_json::json!([{"type": "function"}])),
            ..bare_req()
        };
        let err = validate_chat_request(&req, "served-model", 256, 4096).unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "unsupported_feature",
                ..
            }
        ));
    }

    #[test]
    fn cm_serve_stop_sequences_accepted_end_to_end() {
        // Full-cascade check that a well-formed request carrying `stop`
        // resolves through to `PreparedChatRequest.stop_strings` — the
        // capability matrix's "supported" claim for `stop` on this surface.
        let req = ChatCompletionRequest {
            model: Some("served-model".to_string()),
            messages: vec![user_msg("hi")],
            stop: Some(serde_json::json!(["\n\n"])),
            ..bare_req()
        };
        let prepared =
            prepare_chat_request(&req, "served-model", 256, 4096, false, |_| 1, || 4096).unwrap();
        assert_eq!(prepared.stop_strings, vec!["\n\n".to_string()]);
    }

    #[test]
    fn cm_serve_context_window_checked_before_stop_parsing() {
        // Regression fixture for a refactor bug: extracting stop-sequence
        // parsing into the pre-model validation cascade moved it ahead of
        // the context-window check. The pre-refactor inline sequence
        // checked the context window BEFORE parsing `stop`. A request that is both
        // over-context and carries a malformed `stop` field must
        // therefore fail with `context_length_exceeded`, not a
        // stop-parsing error.
        //
        // This drives `prepare_chat_request` itself (not just its
        // sub-functions in isolation), with a `tokenize_len` thunk that
        // reports the whole context window as already consumed by the
        // prompt — so it is sensitive to a future reordering of the
        // `check_context_window` / `parse_stop_strings` calls inside
        // `prepare_chat_request`, not just to whether each sub-function
        // works in isolation.
        let req = ChatCompletionRequest {
            model: Some("served-model".to_string()),
            messages: vec![user_msg("hi")],
            stop: Some(serde_json::json!([])), // malformed: empty array is rejected
            ..bare_req()
        };
        let err = prepare_chat_request(&req, "served-model", 256, 4096, false, |_| 4096, || 4096)
            .unwrap_err();
        assert!(matches!(
            err,
            ApiError::BadRequest {
                code: "context_length_exceeded",
                ..
            }
        ));
    }

    #[test]
    fn cm_serve_logprobs_resolved_end_to_end() {
        // Full-cascade check backing the matrix's "supported, non-streaming
        // only" `logprobs`/`top_logprobs` claim for `lattice serve`.
        let req = ChatCompletionRequest {
            model: Some("served-model".to_string()),
            messages: vec![user_msg("hi")],
            logprobs: Some(true),
            top_logprobs: Some(3),
            ..bare_req()
        };
        let validated = validate_chat_request(&req, "served-model", 256, 4096).unwrap();
        assert_eq!(validated.logprobs, Some(3));
    }

    // -----------------------------------------------------------------------
    // Shared `AppState` builder for the router-level test modules below --
    // both need a real (tiny, deterministic) CPU model, gated behind
    // `test-utils` (see `lattice_inference::model::qwen35::test_support`)
    // for the same reason: bin targets can't see this crate's own
    // `#[cfg(test)]`-only fixtures across the bin/lib compilation
    // boundary, only a real Cargo feature crosses it.
    // -----------------------------------------------------------------------
    #[cfg(feature = "test-utils")]
    fn tiny_state(max_tokens_cap: usize) -> AppState {
        let model = lattice_inference::model::qwen35::test_support::tiny_zero_model();
        AppState {
            model: ModelBackend::Cpu(Arc::new(model)),
            default_max_tokens: max_tokens_cap,
            max_tokens_cap,
            model_id: "test-model".to_string(),
            request_counter: Arc::new(AtomicU64::new(0)),
            embedding_model: None,
        }
    }

    #[cfg(all(feature = "metal-gpu", feature = "test-utils"))]
    mod vision_content_parts_1135 {
        use super::*;
        use axum::body::Body;
        use base64::Engine as _;
        use lattice_inference::serve::metal_worker::{ContextWindowPolicy, spawn_fake_with_vision};
        use std::sync::atomic::{AtomicBool, Ordering};
        use tower::ServiceExt as _;

        fn inline_png_data_uri() -> String {
            let image = image::RgbImage::new(32, 32);
            let mut bytes = Vec::new();
            image
                .write_to(
                    &mut std::io::Cursor::new(&mut bytes),
                    image::ImageFormat::Png,
                )
                .expect("PNG fixture must encode");
            format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(bytes)
            )
        }

        fn request(data_uri: &str) -> axum::http::Request<Body> {
            let body = serde_json::json!({
                "model": "test-model",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": "after"}
                    ]
                }]
            });
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request fixture must build")
        }

        #[test]
        fn worker_bad_request_remains_a_client_error() {
            let error = map_metal_generation_error(ApiError::BadRequest {
                message: "image geometry is unsupported".to_string(),
                code: "invalid_image",
            });
            assert!(matches!(
                error,
                ApiError::BadRequest {
                    code: "invalid_image",
                    ..
                }
            ));
        }

        #[tokio::test]
        async fn vision_model_accepts_image_and_enqueues_it_on_the_shared_worker() {
            let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                .tokenizer()
                .clone();
            let image_seen = Arc::new(AtomicBool::new(false));
            let seen = Arc::clone(&image_seen);
            let client = spawn_fake_with_vision(
                ContextWindowPolicy::PromptAndMaxTokens,
                4096,
                tokenizer.clone(),
                move |messages, _cfg, prompt_tokens, _on_token, _should_cancel| {
                    let image = messages[0]
                        .image
                        .as_ref()
                        .expect("image must reach the common worker job");
                    assert!(image.bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
                    assert_eq!(image.text_offset, "before".len());
                    assert_eq!(messages[0].content, "beforeafter");
                    seen.store(true, Ordering::SeqCst);
                    Ok(GenerateOutput {
                        text: "ok".to_string(),
                        token_ids: vec![0],
                        prompt_tokens,
                        generated_tokens: 1,
                        stopped: true,
                        stop_reason: None,
                        token_logprobs: vec![],
                    })
                },
            );
            let state = AppState {
                model: ModelBackend::Metal {
                    handle: MetalHandle { client },
                    tokenizer: Arc::new(tokenizer),
                    max_context: 4096,
                },
                default_max_tokens: 16,
                max_tokens_cap: 64,
                model_id: "test-model".to_string(),
                request_counter: Arc::new(AtomicU64::new(0)),
                embedding_model: None,
            };

            let response = router(state)
                .oneshot(request(&inline_png_data_uri()))
                .await
                .expect("router must return a response");
            assert_eq!(response.status(), StatusCode::OK);
            assert!(image_seen.load(Ordering::SeqCst));
        }

        #[tokio::test]
        async fn text_only_model_rejects_the_same_image_with_capability_code() {
            let response = router(tiny_state(64))
                .oneshot(request(&inline_png_data_uri()))
                .await
                .expect("router must return a response");
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("error body must be readable");
            let value: serde_json::Value =
                serde_json::from_slice(&body).expect("error body must be JSON");
            assert_eq!(value["error"]["code"], "vision_unsupported");
            assert_eq!(
                value["error"]["message"],
                "image input requires a vision-capable model"
            );
        }
    }

    // -----------------------------------------------------------------------
    // POST /v1/embeddings router-level contract tests.
    // -----------------------------------------------------------------------
    #[cfg(feature = "test-utils")]
    mod embeddings_route {
        use super::*;
        use axum::body::Body;
        use lattice_inference::serve::embeddings::test_support::{
            tiny_embedding_model, tiny_png_data_uri,
        };
        use tower::ServiceExt as _;

        fn post_embeddings(body: serde_json::Value) -> axum::http::Request<Body> {
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .expect("request fixture must build")
        }

        fn state_with_embedder() -> AppState {
            let mut state = tiny_state(64);
            state.embedding_model = Some(Arc::new(tiny_embedding_model()));
            state
        }

        async fn json_body(response: Response) -> serde_json::Value {
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("response body must be readable");
            serde_json::from_slice(&bytes).expect("response body must be JSON")
        }

        #[tokio::test]
        async fn no_embedding_model_loaded_fails_closed() {
            let response = router(tiny_state(64))
                .oneshot(post_embeddings(serde_json::json!({"input": "a"})))
                .await
                .expect("router must return a response");
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let value = json_body(response).await;
            assert_eq!(value["error"]["code"], "vision_unsupported");
        }

        #[tokio::test]
        async fn happy_path_text_only() {
            let response = router(state_with_embedder())
                .oneshot(post_embeddings(serde_json::json!({"input": "a"})))
                .await
                .expect("router must return a response");
            assert_eq!(response.status(), StatusCode::OK);
            let value = json_body(response).await;
            assert_eq!(value["object"], "list");
            assert_eq!(value["data"][0]["object"], "embedding");
            assert_eq!(value["data"][0]["index"], 0);
            assert_eq!(value["data"][0]["embedding"].as_array().unwrap().len(), 8);
        }

        #[tokio::test]
        async fn happy_path_image() {
            let response = router(state_with_embedder())
                .oneshot(post_embeddings(serde_json::json!({
                    "input": {"type": "image_url", "image_url": {"url": tiny_png_data_uri(0)}},
                })))
                .await
                .expect("router must return a response");
            assert_eq!(response.status(), StatusCode::OK);
            let value = json_body(response).await;
            let embedding = value["data"][0]["embedding"].as_array().unwrap();
            let norm: f64 = embedding
                .iter()
                .map(|x| x.as_f64().unwrap().powi(2))
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-3, "expected unit norm, got {norm}");
        }

        #[tokio::test]
        async fn mixed_batch_preserves_input_order() {
            let response = router(state_with_embedder())
                .oneshot(post_embeddings(serde_json::json!({
                    "input": [
                        "a",
                        {"type": "image_url", "image_url": {"url": tiny_png_data_uri(1)}},
                        "b",
                    ],
                })))
                .await
                .expect("router must return a response");
            assert_eq!(response.status(), StatusCode::OK);
            let value = json_body(response).await;
            let data = value["data"].as_array().unwrap();
            assert_eq!(data.len(), 3);
            assert_eq!(data[0]["index"], 0);
            assert_eq!(data[1]["index"], 1);
            assert_eq!(data[2]["index"], 2);
        }

        #[tokio::test]
        async fn remote_url_rejected() {
            let response = router(state_with_embedder())
                .oneshot(post_embeddings(serde_json::json!({
                    "input": {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                })))
                .await
                .expect("router must return a response");
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let value = json_body(response).await;
            assert_eq!(value["error"]["code"], "unsupported_image_url_scheme");
        }

        #[tokio::test]
        async fn malformed_data_uri_rejected() {
            let response = router(state_with_embedder())
                .oneshot(post_embeddings(serde_json::json!({
                    "input": {"type": "image_url", "image_url": {"url": "data:image/png,not-base64-marked"}},
                })))
                .await
                .expect("router must return a response");
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }

        #[tokio::test]
        async fn invalid_pooling_rejected() {
            let response = router(state_with_embedder())
                .oneshot(post_embeddings(
                    serde_json::json!({"input": "a", "pooling": "max"}),
                ))
                .await
                .expect("router must return a response");
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let value = json_body(response).await;
            assert_eq!(value["error"]["code"], "invalid_pooling");
        }

        #[tokio::test]
        async fn empty_input_rejected() {
            let response = router(state_with_embedder())
                .oneshot(post_embeddings(serde_json::json!({"input": []})))
                .await
                .expect("router must return a response");
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let value = json_body(response).await;
            assert_eq!(value["error"]["code"], "invalid_input");
        }

        #[tokio::test]
        async fn embeddings_route_advertised_at_root() {
            let response = router(tiny_state(64))
                .oneshot(
                    axum::http::Request::builder()
                        .method("GET")
                        .uri("/")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .expect("router must return a response");
            let value = json_body(response).await;
            assert!(
                value["endpoints"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|e| e == "/v1/embeddings")
            );
        }
    }

    // -----------------------------------------------------------------------
    // HTTP-level client-disconnect cancellation (ADR-080 C2) -- gated behind `test-utils` (see
    // `lattice_inference::model::qwen35::test_support`) because it needs a
    // real, tiny, deterministic CPU model to exercise the actual
    // `chat_completions` -> `body_stream`'s `cancel_guard` capture ->
    // `generate_streaming_with_cancel` composition end to end, not just the
    // primitive (already unit/mutation-tested in
    // `model/qwen35/generation.rs`) or the guard type (already unit-tested
    // in `serve/mod.rs`) in isolation. "The disclosed
    // HTTP-level disconnect test gap ... does not waive that gate."
    // -----------------------------------------------------------------------
    #[cfg(feature = "test-utils")]
    mod disconnect_cancellation {
        use super::*;
        use http_body_util::BodyExt;
        use std::time::Duration;

        /// The tiny test model's context window is a fixed 1024 tokens
        /// (`test_support::tiny_zero_model`'s `max_position_embeddings`),
        /// so `max_tokens` here must leave room for the rendered prompt
        /// (well under 1024) -- "effectively unbounded relative to this
        /// test's timeouts" rather than the vastly larger figure a real
        /// model's context window would allow.
        const NEAR_MAX_CONTEXT_TOKENS: usize = 900;

        fn tiny_state() -> AppState {
            super::tiny_state(NEAR_MAX_CONTEXT_TOKENS)
        }

        /// Proves the real `chat_completions` streaming composition --
        /// not just its primitives in isolation -- actually stops
        /// generation on client disconnect. A tiny all-zero-weight model
        /// with `max_tokens` near the tiny model's 1024-token context window means
        /// uncancelled generation would keep running far longer than
        /// this test's timeouts; it reads two real SSE frames (proving
        /// generation is genuinely under way) before dropping the
        /// response body to simulate a disconnect.
        ///
        /// Mutation-sensitive to a known regression: if
        /// `let _cancel_guard_tied_to_stream_lifetime = &cancel_guard;`
        /// is removed from `body_stream`'s `flat_map` closure in
        /// `chat_completions`, `cancel_guard` is no longer captured by
        /// anything and drops at the end of `chat_completions`'s own
        /// function body -- i.e. before the response is even returned to
        /// the caller, let alone before the client could disconnect.
        /// That flips `cancel_rx` immediately, so
        /// `generate_streaming_with_cancel`'s pre-prefill checkpoint
        /// returns `Interrupt` with zero tokens generated. Frame 1 (the
        /// role chunk) still arrives unconditionally either way, and a
        /// *second* frame still arrives either way too -- but under the
        /// mutation it is the finish chunk (`finish_reason: "length"` --
        /// the shared `finish_reason()` mapping has no `"interrupt"`
        /// string; a cancelled/interrupted result reports `"length"`
        /// exactly like a token-cap stop, see its own doc comment --
        /// no `content`) sent by `finish_streaming` for a zero-token
        /// result, not a real content delta. A test that only checks "a
        /// second frame arrived and was Ok" cannot tell these apart, so
        /// this asserts the frame's actual JSON payload shape: real code
        /// path yields `finish_reason: null` + a string `delta.content`;
        /// the mutated path yields a non-null `finish_reason` and no
        /// `content`, and the second `assert!` fails, so this test fails
        /// if the disconnect-stops-generation behavior regresses.
        #[tokio::test]
        async fn chat_completions_streaming_disconnect_stops_generation() {
            let req = ChatCompletionRequest {
                model: Some("test-model".to_string()),
                messages: vec![user_msg("hi")],
                max_tokens: Some(NEAR_MAX_CONTEXT_TOKENS),
                stream: Some(true),
                ..bare_req()
            };

            let response = chat_completions_with_request(State(tiny_state()), req)
                .await
                .expect("streaming request must be accepted");
            assert_eq!(response.status(), StatusCode::OK);

            let mut body = response.into_body();

            // Frame 1: the role chunk, emitted unconditionally before any
            // generation output (`futures::stream::once(role_chunk).chain(body_stream)`).
            tokio::time::timeout(Duration::from_secs(10), body.frame())
                .await
                .expect("role chunk frame must arrive quickly")
                .expect("role chunk frame must exist")
                .expect("role chunk frame must be Ok");

            // Frame 2: must be a genuine content delta from the tiny
            // model's decode loop, not the finish chunk. A frame merely
            // *arriving* here does not prove generation ran: if
            // `cancel_guard` is no longer tied to the stream, it drops
            // (flipping `cancel_rx`) before the blocking task's
            // pre-prefill checkpoint, `generate_streaming_with_cancel`
            // returns `Interrupt` with zero tokens, and
            // `finish_streaming` sends `StreamMsg::Done` immediately --
            // which *also* produces a well-formed, `Ok` SSE frame here,
            // just one carrying `finish_reason: "length"` (the shared
            // mapping has no `"interrupt"` string; see
            // `lattice_inference::serve::finish_reason`'s doc comment)
            // and no `content` instead of a real delta. So this asserts
            // the frame's actual payload shape, not just its arrival.
            let frame = tokio::time::timeout(Duration::from_secs(10), body.frame())
                .await
                .expect(
                    "a second frame must arrive quickly -- if this times out, \
                         nothing at all was sent after the role chunk",
                )
                .expect("second frame must exist")
                .expect("second frame must be Ok");
            let data = frame
                .into_data()
                .expect("second frame must carry data (not trailers)");
            let text = std::str::from_utf8(&data).expect("SSE frame must be UTF-8");
            let payload = text
                .strip_prefix("data: ")
                .unwrap_or(text)
                .trim_end_matches(['\n', '\r']);
            let chunk: serde_json::Value =
                serde_json::from_str(payload).expect("SSE frame must carry a JSON chunk");
            let choice = &chunk["choices"][0];
            assert!(
                choice["finish_reason"].is_null(),
                "second frame must be a content delta, not the finish chunk \
                     (generation was interrupted before producing any output, \
                     i.e. cancel_guard fired before the stream had a chance to \
                     run): {chunk}"
            );
            assert!(
                choice["delta"]["content"].is_string(),
                "second frame must carry delta.content (a real generated \
                     token), got: {chunk}"
            );

            // Client disconnect: drop the body. `cancel_guard` (captured
            // by `body_stream`'s `flat_map` closure) drops with it,
            // flipping the paired `cancel_rx` -- the CPU decode loop's
            // `generate_streaming_with_cancel` polls that flag at the top
            // of every iteration (already unit/mutation-tested in
            // `model/qwen35/generation.rs`) and stops within one step
            // instead of running toward `max_tokens` (900). This
            // test's own fast, deterministic completion (it does not hang
            // waiting on the now-unobservable background task) is the
            // practical proof: dropping never blocks, and the tiny
            // model's decode loop is fast enough on the blocking-thread
            // pool that a broken cancellation path would otherwise pin a
            // thread pool worker in an unbounded loop for the remainder
            // of the test process's life -- observable across this
            // binary's full test run as sporadic slowdowns/hangs in
            // later tests sharing the pool, not just this test in
            // isolation.
            drop(body);
        }
    }

    // -----------------------------------------------------------------------
    // Post-drop generator-side cancellation probe (ADR-080 C2):
    // `chat_completions_streaming_
    // disconnect_stops_generation` above proves guard retention BEFORE
    // the response is returned (frame 2 is a real content delta), but
    // does NOT prove `should_cancel` reaching the generator AFTER the
    // drop, independently of `on_token`'s own failed-send stop
    // condition -- the exact reverse mutation (`cancel_rx`
    // predicate replaced by `move || false`, `on_token`'s failed-send
    // path left intact) left that test green in 0.02s, because the
    // failed send alone stops a real decode loop just as fast as a
    // correctly wired cancellation would.
    //
    // This test isolates the two signals: `ModelBackend::CpuFakeGenerate`
    // (test-only, see its doc comment) sends exactly ONE delta -- so
    // `on_token` is never called again after that point -- then enters a
    // phase that polls ONLY `should_cancel`, in a loop that never touches
    // `on_token`/the reply channel at all, and reports which of two
    // outcomes ended that loop over a side channel `chat_completions`
    // itself never sees. Because the probe stops polling `on_token`
    // entirely after the first delta, `on_token`'s failed-send masking
    // effect (the exact thing that hid this bug from the original
    // test) cannot fire here -- only `should_cancel`'s own return value
    // can end the loop.
    // -----------------------------------------------------------------------
    #[cfg(feature = "test-utils")]
    mod post_drop_cancellation_probe {
        use super::*;
        use http_body_util::BodyExt;
        use std::sync::Mutex;
        use std::sync::mpsc::RecvTimeoutError;
        use std::time::Duration;

        /// One real delta, then a bounded should_cancel-only poll loop.
        const MAX_POLLS: usize = 4000;
        const POLL_INTERVAL: Duration = Duration::from_millis(5);
        const PROBE_POLL_BUDGET: Duration = POLL_INTERVAL.saturating_mul(MAX_POLLS as u32);
        const PROBE_COMPLETION_HEADROOM: Duration = Duration::from_secs(20);
        const PROBE_COMPLETION_TIMEOUT: Duration =
            PROBE_POLL_BUDGET.saturating_add(PROBE_COMPLETION_HEADROOM);
        const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

        fn await_checkpoint(
            rx: &std::sync::mpsc::Receiver<()>,
            checkpoint: &str,
            timeout: Duration,
        ) {
            match rx.recv_timeout(timeout) {
                Ok(()) => {}
                Err(RecvTimeoutError::Timeout) => {
                    panic!(
                        "fake generator timed out waiting for the {checkpoint} \
                             checkpoint after {timeout:?}"
                    );
                }
                Err(RecvTimeoutError::Disconnected) => {
                    panic!(
                        "fake generator's {checkpoint} checkpoint sender \
                             disconnected before signaling"
                    );
                }
            }
        }

        fn probe_completion_timeout_message() -> String {
            format!(
                "observed no ProbeOutcome within {PROBE_COMPLETION_TIMEOUT:?} \
                     after the client disconnect; possible causes include a \
                     generator scheduling delay or stall, or cancellation \
                     propagation failing to complete within the \
                     {PROBE_POLL_BUDGET:?} probe poll budget"
            )
        }

        #[test]
        fn completion_timeout_exceeds_probe_poll_budget() {
            assert_eq!(PROBE_POLL_BUDGET, Duration::from_secs(20));
            assert!(
                PROBE_COMPLETION_TIMEOUT > PROBE_POLL_BUDGET,
                "completion timeout {PROBE_COMPLETION_TIMEOUT:?} must exceed \
                     the complete probe poll budget {PROBE_POLL_BUDGET:?}"
            );
        }

        #[test]
        #[should_panic(expected = "post-handler-return checkpoint")]
        fn post_handler_return_checkpoint_timeout_fails_loudly() {
            let (tx, rx) = std::sync::mpsc::channel();
            await_checkpoint(&rx, "post-handler-return", Duration::ZERO);
            drop(tx);
        }

        #[test]
        #[should_panic(expected = "post-body-drop checkpoint")]
        fn post_body_drop_checkpoint_timeout_fails_loudly() {
            let (tx, rx) = std::sync::mpsc::channel();
            await_checkpoint(&rx, "post-body-drop", Duration::ZERO);
            drop(tx);
        }

        #[test]
        fn completion_timeout_message_reports_observation_and_candidate_causes() {
            let message = probe_completion_timeout_message();
            assert!(message.contains("observed no ProbeOutcome"));
            assert!(message.contains("generator scheduling delay or stall"));
            assert!(message.contains("cancellation propagation"));
        }

        /// Same rationale as `disconnect_cancellation`'s constant of the
        /// same value: the tiny test model's context window is a fixed
        /// 1024 tokens, so this stays comfortably under it while still
        /// being "effectively unbounded" relative to this test's own
        /// timeouts.
        const NEAR_MAX_CONTEXT_TOKENS: usize = 900;

        /// `should_cancel`'s reading taken after the test's `checkpoint1`
        /// signal (sent the instant `chat_completions(..).await`
        /// returns, before the test reads any SSE frame or drops
        /// anything). The generator acknowledges that reading over a
        /// oneshot channel, and the test waits for the acknowledgement
        /// before it can drop the body or send `go`. Must be `false` in
        /// correctly-wired code:
        /// `cancel_guard` is still alive at this point (held by
        /// `body_stream`'s `flat_map` closure, itself alive because the
        /// test hasn't dropped the body/receiver yet), so `cancel_rx`
        /// cannot have flipped. A `true` reading here is the direct,
        /// timing-independent signature of the guard-capture-removed
        /// mutation: `cancel_guard` is then just an unused
        /// local that drops the instant `chat_completions` returns --
        /// gating this read on `checkpoint1` (rather than reading it
        /// immediately after `on_token`, which raced against
        /// `chat_completions`'s own return on an early version of this
        /// test and non-deterministically read `false` even under the
        /// mutation) guarantees the read happens causally after that
        /// return, so an unused `cancel_guard` has unconditionally
        /// already dropped by the time this reads it.
        #[derive(Debug, PartialEq)]
        struct ProbeOutcome {
            pre_drop_cancelled: bool,
            post_drop: PostDropOutcome,
        }

        #[derive(Debug, PartialEq)]
        enum PostDropOutcome {
            /// `should_cancel` returned `true` after this many polls
            /// following the test's `go` signal (sent only after
            /// `drop(body)`) -- the correctly-wired behavior.
            CancelledAfterPolls(usize),
            /// The poll budget was exhausted without `should_cancel`
            /// ever returning `true` -- the `move || false` mutation's
            /// signature (an unthreaded/replaced predicate that can
            /// never observe the drop).
            ExhaustedWithoutCancel,
        }

        /// Builds an `AppState` whose CPU streaming generation is the
        /// injected `generate` closure instead of the real tiny model's
        /// decode loop, via `ModelBackend::CpuFakeGenerate`. Tokenizer/
        /// context-window behavior still comes from a real tiny model
        /// (`tiny_zero_model()`), so request validation ahead of the
        /// streaming branch is unchanged from the other HTTP-level
        /// tests in this file.
        fn tiny_state_with_fake_cpu_generate(
            max_tokens_cap: usize,
            generate: impl Fn(
                &str,
                &lattice_inference::model::qwen35_config::GenerateConfig,
                &mut dyn FnMut(&str) -> bool,
                &mut dyn FnMut() -> bool,
            )
                -> Result<GenerateOutput, lattice_inference::error::InferenceError>
            + Send
            + Sync
            + 'static,
        ) -> AppState {
            let model = lattice_inference::model::qwen35::test_support::tiny_zero_model();
            AppState {
                model: ModelBackend::CpuFakeGenerate {
                    model: Arc::new(model),
                    generate: Arc::new(generate),
                },
                default_max_tokens: max_tokens_cap,
                max_tokens_cap,
                model_id: "test-model".to_string(),
                request_counter: Arc::new(AtomicU64::new(0)),
                embedding_model: None,
            }
        }

        #[tokio::test]
        async fn chat_completions_streaming_failure_emits_error_event() {
            let state =
                tiny_state_with_fake_cpu_generate(64, |_prompt, _cfg, on_token, _should_cancel| {
                    let _ = on_token("partial");
                    Err(lattice_inference::error::InferenceError::InvalidInput(
                        "blocked by grammar".to_string(),
                    ))
                });
            let req = ChatCompletionRequest {
                model: Some("test-model".to_string()),
                messages: vec![user_msg("hi")],
                max_tokens: Some(64),
                stream: Some(true),
                ..bare_req()
            };

            let response = chat_completions_with_request(State(state), req)
                .await
                .expect("streaming request must be accepted");
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = response
                .into_body()
                .collect()
                .await
                .expect("SSE response body must be readable")
                .to_bytes();
            let text = String::from_utf8(bytes.to_vec()).expect("SSE body must be valid UTF-8");
            assert!(
                text.contains("\"content\":\"partial\""),
                "partial output must precede the error event; got: {text}"
            );
            let error_payload = text
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .find(|payload| payload.contains("\"error\""))
                .expect("a failed generation must emit an SSE error payload");
            let error: serde_json::Value =
                serde_json::from_str(error_payload).expect("SSE error payload must be valid JSON");
            assert_eq!(error["error"]["type"], "server_error");
            assert_eq!(error["error"]["code"], "internal_error");
            assert!(
                !text.contains("\"finish_reason\":\"stop\""),
                "generation failure must not masquerade as a clean stop; got: {text}"
            );
        }

        /// Mutation-sensitive to BOTH known regressions
        /// independently, via a two-way test<->generator handshake that
        /// removes the timing race a plain poll-and-time-it design would have
        /// (an early, timing-dependent version of this test observed
        /// `CancelledAfterPolls(1)` even under mutation (a), because
        /// `cancel_guard` -- unused once its capture is removed --
        /// drops at `chat_completions`'s own return, which can race
        /// ahead of or behind the generator's first poll depending on
        /// blocking-thread-pool scheduling):
        ///
        /// (a) removing `body_stream`'s
        ///     `let _cancel_guard_tied_to_stream_lifetime = &cancel_guard;`
        ///     capture flips `cancel_rx` the instant `chat_completions`
        ///     returns -- before the test even reads its first SSE
        ///     frame, let alone drops the body. `pre_drop_cancelled`
        ///     (read BEFORE the generator waits on the `go` signal,
        ///     which the test only sends after `drop(body)`) captures
        ///     exactly this: `true` here is impossible under correct
        ///     wiring regardless of scheduling, since `cancel_guard` is
        ///     provably still alive at that point.
        /// (b) replacing the CPU `should_cancel` predicate
        ///     (`move || *cancel_rx.borrow()`) with `move || false`
        ///     means the post-`go` poll loop never observes `true`, so
        ///     it exhausts `MAX_POLLS` and reports
        ///     `PostDropOutcome::ExhaustedWithoutCancel`.
        #[tokio::test]
        async fn chat_completions_streaming_disconnect_cancellation_reaches_generator_post_drop() {
            let (outcome_tx, outcome_rx) = tokio::sync::oneshot::channel::<ProbeOutcome>();
            let outcome_tx = Mutex::new(Some(outcome_tx));
            let (checkpoint1_tx, checkpoint1_rx) = std::sync::mpsc::channel::<()>();
            let checkpoint1_rx = Mutex::new(Some(checkpoint1_rx));
            let (pre_drop_observed_tx, pre_drop_observed_rx) =
                tokio::sync::oneshot::channel::<bool>();
            let pre_drop_observed_tx = Mutex::new(Some(pre_drop_observed_tx));
            let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
            let go_rx = Mutex::new(Some(go_rx));

            let state = tiny_state_with_fake_cpu_generate(
                NEAR_MAX_CONTEXT_TOKENS,
                move |_prompt, _cfg, on_token, should_cancel| {
                    // One real delta -- exactly what the disconnect test
                    // above reads as its second frame -- so
                    // `chat_completions`'s SSE framing has genuine
                    // content before this probe phase begins. Queued
                    // into the reply channel immediately; the test can
                    // read it as frame 2 regardless of whether this
                    // thread is later blocked waiting on `checkpoint1`.
                    on_token("probe");

                    // Block until the test confirms `chat_completions`
                    // itself has already returned (see `pre_drop_cancelled`'s
                    // doc comment on why this ordering, not an
                    // immediate post-`on_token` read, is what makes the
                    // next line's reading timing-independent).
                    if let Some(rx) = checkpoint1_rx.lock().unwrap().take() {
                        await_checkpoint(&rx, "post-handler-return", HANDSHAKE_TIMEOUT);
                    }
                    let pre_drop_cancelled = should_cancel();
                    if let Some(tx) = pre_drop_observed_tx.lock().unwrap().take() {
                        let _ = tx.send(pre_drop_cancelled);
                    }

                    // Block until the test signals it has dropped the
                    // body (or give up after 10s so a broken handshake
                    // fails this test instead of hanging the process).
                    if let Some(rx) = go_rx.lock().unwrap().take() {
                        await_checkpoint(&rx, "post-body-drop", HANDSHAKE_TIMEOUT);
                    }

                    let mut polls = 0usize;
                    let post_drop = loop {
                        if should_cancel() {
                            break PostDropOutcome::CancelledAfterPolls(polls);
                        }
                        if polls >= MAX_POLLS {
                            break PostDropOutcome::ExhaustedWithoutCancel;
                        }
                        polls += 1;
                        std::thread::sleep(POLL_INTERVAL);
                    };
                    if let Some(tx) = outcome_tx.lock().unwrap().take() {
                        let _ = tx.send(ProbeOutcome {
                            pre_drop_cancelled,
                            post_drop,
                        });
                    }
                    Ok(GenerateOutput {
                        text: "probe".to_string(),
                        token_ids: vec![],
                        prompt_tokens: 1,
                        generated_tokens: 1,
                        stopped: false,
                        stop_reason: Some(lattice_inference::stop_reason::StopReason::Interrupt),
                        token_logprobs: vec![],
                    })
                },
            );

            let req = ChatCompletionRequest {
                model: Some("test-model".to_string()),
                messages: vec![user_msg("hi")],
                max_tokens: Some(NEAR_MAX_CONTEXT_TOKENS),
                stream: Some(true),
                ..bare_req()
            };
            let response = chat_completions_with_request(State(state), req)
                .await
                .expect("streaming request must be accepted");
            // `chat_completions` has now unconditionally returned, so an
            // unused `cancel_guard` (mutation (a)) has already dropped;
            // signal the generator it may take its `pre_drop_cancelled`
            // reading.
            checkpoint1_tx
                .send(())
                .expect("generator must still be waiting for the post-return checkpoint");
            let observed_pre_drop_cancelled =
                tokio::time::timeout(HANDSHAKE_TIMEOUT, pre_drop_observed_rx)
                    .await
                    .expect(
                        "generator must acknowledge its pre-drop cancellation \
                             reading before the handshake timeout",
                    )
                    .expect(
                        "generator must not drop the pre-drop observation \
                             sender without acknowledging its reading",
                    );
            assert!(
                !observed_pre_drop_cancelled,
                "should_cancel must read false while the response body is \
                     still alive"
            );
            let mut body = response.into_body();

            // Frame 1: role chunk.
            tokio::time::timeout(Duration::from_secs(10), body.frame())
                .await
                .expect("role chunk frame must arrive quickly")
                .expect("role chunk frame must exist")
                .expect("role chunk frame must be Ok");

            // Frame 2: the fake generator's one real delta.
            tokio::time::timeout(Duration::from_secs(10), body.frame())
                .await
                .expect("delta frame must arrive quickly")
                .expect("delta frame must exist")
                .expect("delta frame must be Ok");

            // Client disconnect: drop the body. The fake generator is
            // waiting on `go_rx`, having already called `on_token` and
            // taken its `pre_drop_cancelled` reading.
            drop(body);
            go_tx
                .send(())
                .expect("generator must still be waiting for the post-drop checkpoint");

            let outcome = tokio::time::timeout(PROBE_COMPLETION_TIMEOUT, outcome_rx)
                .await
                .unwrap_or_else(|_| panic!("{}", probe_completion_timeout_message()))
                .expect("probe outcome sender must not be dropped without sending");

            assert!(
                !outcome.pre_drop_cancelled,
                "should_cancel must read false before the test has dropped the \
                     response body -- a true reading here means cancel_guard had \
                     already dropped (e.g. because it is no longer tied to the \
                     stream's lifetime) well before any disconnect happened: \
                     {outcome:?}"
            );
            assert!(
                matches!(outcome.post_drop, PostDropOutcome::CancelledAfterPolls(_)),
                "should_cancel must return true within the poll budget after \
                     the test drops the response body -- exhausting the budget \
                     means the disconnect signal never reached the generator at \
                     all: {outcome:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Streaming context-overflow status parity (ADR-080 C2): `lattice.rs` already gets this right
    // structurally -- `prepare_chat_request`'s context-window preflight
    // (`check_context_window`) runs unconditionally, before
    // `chat_completions` ever branches on `req.stream` -- so a `stream:
    // true` request that overflows the model's context window returns
    // HTTP 400 `context_length_exceeded` before any SSE stream is ever
    // built. `cm_serve_context_window_checked_before_stop_parsing` above
    // already pins the underlying cascade ordering as a pure function;
    // this drives the SAME contract through the real `Router`.
    //
    // ADR-080 C2: this now
    // builds its request from `lattice_inference::serve`'s shared
    // `OVERFLOW_PARITY_*` constants -- the SAME body/limits
    // `lattice_serve.rs`'s `real_router_overflow_parity` module drives
    // through its own real router and real worker -- so the two really
    // are the identical request the old doc comment here merely
    // claimed. The tiny test model's context window
    // (`test_support::tiny_zero_model`'s `max_position_embeddings`) is
    // fixed at `OVERFLOW_PARITY_CONTEXT_WINDOW` (1024) precisely so this
    // side's "effective context limit" matches the daemon side's
    // explicitly-configured `AppState.model_max_context` of the same
    // value.
    // -----------------------------------------------------------------------
    #[cfg(feature = "test-utils")]
    mod streaming_context_overflow {
        use super::*;
        use lattice_inference::serve::{
            OVERFLOW_PARITY_MAX_TOKENS_CAP, OVERFLOW_PARITY_REQUEST_BODY,
        };
        use tower::ServiceExt as _;

        #[tokio::test]
        async fn chat_completions_streaming_context_overflow_returns_400_before_committing_sse() {
            let body = axum::body::Body::from(OVERFLOW_PARITY_REQUEST_BODY.to_string());
            let request = axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(body)
                .expect("fixture request must build");
            let response = router(tiny_state(OVERFLOW_PARITY_MAX_TOKENS_CAP))
                .oneshot(request)
                .await
                .expect("router must produce a response, not a transport error");
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "an over-context stream:true request must be rejected with HTTP \
                     400 before any SSE stream is committed, matching \
                     lattice_serve.rs's equivalent preflight for the identical \
                     shared-fixture request body"
            );
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("error response body must be readable");
            let value: serde_json::Value =
                serde_json::from_slice(&bytes).expect("error response must be JSON");
            assert_eq!(value["error"]["code"], "context_length_exceeded");
        }
    }

    // -----------------------------------------------------------------------
    // Message-flood bound: proves the fix at the real HTTP layer, not just
    // at `serve::contract`'s own unit-test level -- the router's
    // `chat_completions` entry point must reject a body with more than
    // `MAX_MESSAGE_COUNT` tiny messages.
    // -----------------------------------------------------------------------
    #[cfg(feature = "test-utils")]
    mod message_flood {
        use super::*;
        use lattice_inference::serve::contract::MAX_MESSAGE_COUNT;
        use tower::ServiceExt as _;

        #[tokio::test]
        async fn chat_completions_rejects_message_flood() {
            // One more message than the bound, each as small as the wire
            // format allows: comfortably under the 1 MiB body cap, but
            // tens of thousands of entries. The message-count bound is
            // enforced inline while `ChatCompletionRequest::messages`
            // deserializes, so this never allocates a `Vec<Message>`
            // entry per message before rejecting -- see this test's
            // sibling unit tests in `serve::contract` for direct coverage
            // of that deserializer.
            let messages: Vec<String> = (0..MAX_MESSAGE_COUNT + 1)
                .map(|_| r#"{"role":"user","content":""}"#.to_string())
                .collect();
            let body = format!(
                r#"{{"model":"test-model","messages":[{}]}}"#,
                messages.join(",")
            );
            let request = axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body))
                .expect("fixture request must build");
            let response = router(tiny_state(64))
                .oneshot(request)
                .await
                .expect("router must produce a response, not a transport error");
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("error response body must be readable");
            let value: serde_json::Value =
                serde_json::from_slice(&bytes).expect("error response must be JSON");
            assert_eq!(value["error"]["code"], "invalid_request_body");
        }
    }

    // -----------------------------------------------------------------------
    // Content-Type precedence: an invalid-MIME request must be rejected by
    // `require_json_content_type` before the body is ever read as JSON.
    // -----------------------------------------------------------------------
    #[cfg(feature = "test-utils")]
    mod content_type_precedence {
        use super::*;
        use tower::ServiceExt as _;

        /// A body that is not valid JSON at all. Combined with a
        /// non-JSON `Content-Type`, this distinguishes ordering: if the
        /// Content-Type guard runs first, the response is 415
        /// `unsupported_media_type`; if it were reordered to run after
        /// the body is parsed, this body would instead fail JSON
        /// parsing first and surface as 400 `invalid_request_body`.
        #[tokio::test]
        async fn invalid_content_type_rejected_before_body_is_parsed() {
            let request = axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "text/plain")
                .body(axum::body::Body::from("this is not json"))
                .expect("fixture request must build");
            let response = router(tiny_state(64))
                .oneshot(request)
                .await
                .expect("router must produce a response, not a transport error");
            assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("error response body must be readable");
            let value: serde_json::Value =
                serde_json::from_slice(&bytes).expect("error response must be JSON");
            assert_eq!(value["error"]["code"], "unsupported_media_type");
        }
    }

    // -----------------------------------------------------------------------
    // Cross-binary `/v1/chat/completions` parity table (ADR-080 C2):
    // drives every fixture body in
    // `lattice_inference::serve::CHAT_COMPLETIONS_PARITY_CASES` through
    // THIS binary's real `Router` via `tower::ServiceExt::oneshot`, and
    // compares the resulting status + error code against the case's
    // `lattice`-side expectation. `lattice_serve.rs`'s own test module
    // runs the SAME table against its own router, asserting the
    // `lattice_serve`-side expectation -- together the two prove
    // same-input parity (or a documented, intentional divergence) at the
    // real HTTP layer, not just in each binary's private validation
    // helpers. Gated behind `test-utils` for the same reason as
    // `disconnect_cancellation`: `router()` needs a real `AppState`,
    // which needs a real (tiny) CPU model.
    // -----------------------------------------------------------------------
    #[cfg(feature = "test-utils")]
    mod parity_table {
        use super::*;
        use lattice_inference::StopReason;
        use lattice_inference::serve::{
            BASELINE_CANNED_COMPLETION_TOKENS, BASELINE_CANNED_PROMPT_TOKENS, BASELINE_CANNED_TEXT,
            Binary, CHAT_COMPLETIONS_PARITY_CASES, ExpectedResponse, check_sse_events,
        };
        use tower::ServiceExt as _;

        /// Small enough that `max_tokens_over_cap_reject_vs_clamp`'s
        /// 999999 genuinely exceeds both `default_max_tokens` and
        /// `max_tokens_cap`.
        const CAP: usize = 64;

        /// Deterministic CPU generation seam for every `Json`/`Sse`
        /// row (issue #828): the real request-parse/normalize/
        /// `GenerateConfig`-build/handler/serialization path all still
        /// runs unmodified -- only the actual model forward pass is
        /// replaced, via the SAME `ModelBackend::CpuFakeGenerate`
        /// injection seam the disconnect-cancellation probe uses.
        /// Content deltas are pushed through `on_token` (what the
        /// streaming arm reads) AND the returned `GenerateOutput.text`
        /// carries the identical concatenated text (what the
        /// non-streaming arm reads) so one closure serves both shapes.
        fn baseline_fake_state(max_tokens_cap: usize) -> AppState {
            let model = lattice_inference::model::qwen35::test_support::tiny_zero_model();
            #[allow(clippy::type_complexity)]
            let generate: Arc<
                dyn Fn(
                        &str,
                        &lattice_inference::model::qwen35_config::GenerateConfig,
                        &mut dyn FnMut(&str) -> bool,
                        &mut dyn FnMut() -> bool,
                    )
                        -> Result<GenerateOutput, lattice_inference::error::InferenceError>
                    + Send
                    + Sync,
            > = Arc::new(|_prompt, _cfg, on_token, _should_cancel| {
                for chunk in ["hello", " world"] {
                    if !on_token(chunk) {
                        break;
                    }
                }
                Ok(GenerateOutput {
                    text: BASELINE_CANNED_TEXT.to_string(),
                    token_ids: vec![1, 2],
                    prompt_tokens: BASELINE_CANNED_PROMPT_TOKENS as usize,
                    generated_tokens: BASELINE_CANNED_COMPLETION_TOKENS as usize,
                    stopped: true,
                    stop_reason: Some(StopReason::Eos),
                    token_logprobs: vec![],
                })
            });
            AppState {
                model: ModelBackend::CpuFakeGenerate {
                    model: Arc::new(model),
                    generate,
                },
                default_max_tokens: max_tokens_cap,
                max_tokens_cap,
                model_id: "test-model".to_string(),
                request_counter: Arc::new(AtomicU64::new(0)),
                embedding_model: None,
            }
        }

        #[tokio::test]
        async fn chat_completions_matches_shared_parity_table() {
            for case in CHAT_COMPLETIONS_PARITY_CASES {
                let expected = case.expected(Binary::Lattice);
                // Error-shaped rows never reach generation (rejected at
                // validation), so they keep using the plain real tiny
                // model exactly as before #828; only the new `Json`/
                // `Sse` rows need the deterministic generation seam.
                let app = match expected {
                    ExpectedResponse::Error { .. } => router(tiny_state(CAP)),
                    ExpectedResponse::Json { .. } | ExpectedResponse::Sse { .. } => {
                        router(baseline_fake_state(CAP))
                    }
                };
                let request = axum::http::Request::builder()
                    .method(case.method)
                    .uri(case.path)
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(case.body.build()))
                    .expect("fixture request must build");
                let response = app
                    .oneshot(request)
                    .await
                    .expect("router must produce a response, not a transport error");

                let status = response.status().as_u16();
                let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .expect("response body reads");
                let text = String::from_utf8_lossy(&body);

                assert_eq!(
                    status,
                    expected.status(),
                    "case '{}': expected status {}, got {status} (body: {text})",
                    case.name,
                    expected.status(),
                );

                match expected {
                    ExpectedResponse::Error { code, .. } => {
                        let value: serde_json::Value = serde_json::from_slice(&body)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "case '{}': non-2xx response body must be the shared \
                                         error envelope JSON: {e} (body: {text})",
                                    case.name,
                                )
                            });
                        assert_eq!(
                            value["error"]["code"], code,
                            "case '{}': expected error code '{code}', got {} \
                                 (full body: {value})",
                            case.name, value["error"]["code"]
                        );
                    }
                    ExpectedResponse::Json { fields, .. } => {
                        let value: serde_json::Value = serde_json::from_slice(&body)
                            .unwrap_or_else(|e| {
                                panic!(
                                    "case '{}': 2xx response body must be JSON: {e} \
                                         (body: {text})",
                                    case.name,
                                )
                            });
                        for field in fields {
                            field.check(&value).unwrap_or_else(|e| {
                                panic!("case '{}': field check failed: {e}", case.name)
                            });
                        }
                    }
                    ExpectedResponse::Sse { events, .. } => {
                        check_sse_events(&text, events).unwrap_or_else(|e| {
                            panic!("case '{}': SSE check failed: {e}", case.name)
                        });
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Production-adapter observation (issue #828): proves the shared
    // `ProductionAdapterObservation`/`GenerateConfigSnapshot` types
    // actually capture what THIS binary's real `chat_completions` ->
    // `prepare_chat_request` -> `GenerateConfig` construction produces,
    // not a value the test independently reconstructs. The injected
    // `CpuFakeGenerate` closure below runs strictly BELOW that real
    // path -- it records the `&GenerateConfig`/`&str` prompt it was
    // actually called with, then returns a canned result; it never
    // recomputes `build_cfg`/`validate_temperature`/etc. itself.
    //
    // DISPUTED (issue #828):
    // this observation captures `rendered_prompt`, not `messages`, and
    // that is the real shape of this seam, not an omission. `chat_completions`
    // computes `to_chat_messages(&req.messages)` (the normalized message
    // list) unconditionally whenever `feature = "metal-gpu"` is compiled
    // in, but that value is consumed ONLY by the `ModelBackend::Metal`
    // match arm (`handle.generate_streaming[_with_cancel](chat_messages,
    // ...)`); the `ModelBackend::Cpu`/`CpuFakeGenerate` arms this test
    // seam exercises never receive it -- their real `generate`/
    // `generate_streaming_with_cancel` calls take only `(&prompt,
    // &gen_cfg, ...)`. This mirrors `ProductionAdapterObservation`'s own
    // documented contract in `serve/mod.rs` ("exactly one of
    // `rendered_prompt`/`messages` is `Some` per capture, reflecting
    // which shape that binary's real adapter actually receives, not a
    // missing capture").
    //
    // Observing `messages` at the CPU seam authentically (not by
    // re-deriving `to_chat_messages` independently in the test, which
    // would be tautological -- exactly the bug this
    // module was written to fix) would require a `MetalFakeGenerate`
    // test double for `ModelBackend::Metal`. `MetalHandle::spawn`
    // hard-requires loading a real Q4 model directory onto a real Metal
    // GPU worker thread (`MetalQwen35State::from_q4_dir`) -- there is no
    // model-agnostic seam there the way `CpuFakeGenerate` mirrors
    // `ModelBackend::Cpu`. Building one would mean adding a new
    // production `ModelBackend` variant and mocking the async engine
    // handle's job-channel protocol: real production-code surface
    // expansion, not a test-only capture. That is out of scope for this
    // fix round; tracked as a follow-up if a Metal-path observation is
    // wanted (would need its own issue -- #828's fixture data and CI
    // environment target the CPU/tiny-tokenizer seam only).
    // -----------------------------------------------------------------------
    #[cfg(feature = "test-utils")]
    mod production_adapter_observation {
        use super::*;
        use lattice_inference::serve::{
            ExpectedObservation, GenerateConfigSnapshot, OBSERVATION_GOLDEN_USER_HI_THERE_CHATML,
            ProductionAdapterObservation, assert_observation_matches,
        };
        use std::sync::Mutex;
        use tower::ServiceExt as _;

        /// Builds the fixture state + fires the fixed `{"messages":[{"role":
        /// "user","content":"hi there"}],"temperature":1.3,"top_p":0.55,
        /// "seed":7,"max_tokens":9}` request against a real router, with the
        /// injected `CpuFakeGenerate` closure recording a
        /// `ProductionAdapterObservation` -- strictly below the real
        /// request-parse/`format_chat_template`/`GenerateConfig`-construction path
        /// (issue #828). `stopped` is threaded through a single local
        /// variable into both the recorded observation and the returned
        /// `GenerateOutput`, so a caller of this helper can vary it and prove
        /// the observation genuinely mirrors what the seam returned rather
        /// than an independent hardcoded literal.
        const OBSERVATION_GOLDEN_REQUEST_BODY: &str = r#"{"model":"test-model","messages":[{"role":"user","content":"hi there"}],"temperature":1.3,"top_p":0.55,"seed":7,"max_tokens":9}"#;

        async fn run_observed(stopped: bool, body: &str) -> ProductionAdapterObservation {
            let model = lattice_inference::model::qwen35::test_support::tiny_zero_model();
            let tokenizer = model.tokenizer().clone();
            let observed: Arc<Mutex<Option<ProductionAdapterObservation>>> =
                Arc::new(Mutex::new(None));
            let observed_for_closure = Arc::clone(&observed);
            #[allow(clippy::type_complexity)]
            let generate: Arc<
                dyn Fn(
                        &str,
                        &lattice_inference::model::qwen35_config::GenerateConfig,
                        &mut dyn FnMut(&str) -> bool,
                        &mut dyn FnMut() -> bool,
                    )
                        -> Result<GenerateOutput, lattice_inference::error::InferenceError>
                    + Send
                    + Sync,
            > = Arc::new(move |prompt, cfg, _on_token, _should_cancel| {
                let prompt_tokens = tokenizer.tokenize(prompt).real_length;
                *observed_for_closure
                    .lock()
                    .expect("observation mutex poisoned") = Some(ProductionAdapterObservation {
                    rendered_prompt: Some(prompt.to_string()),
                    messages: None,
                    gen_cfg: GenerateConfigSnapshot::from(cfg),
                    prompt_tokens,
                    stopped,
                });
                Ok(GenerateOutput {
                    text: "ok".to_string(),
                    token_ids: vec![1],
                    prompt_tokens,
                    generated_tokens: 1,
                    stopped,
                    stop_reason: if stopped {
                        Some(lattice_inference::StopReason::Eos)
                    } else {
                        None
                    },
                    token_logprobs: vec![],
                })
            });
            let state = AppState {
                model: ModelBackend::CpuFakeGenerate {
                    model: Arc::new(model),
                    generate,
                },
                default_max_tokens: 64,
                max_tokens_cap: 64,
                model_id: "test-model".to_string(),
                request_counter: Arc::new(AtomicU64::new(0)),
                embedding_model: None,
            };
            let request = axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_string()))
                .expect("fixture request must build");
            let response = router(state)
                .oneshot(request)
                .await
                .expect("router must produce a response, not a transport error");
            assert_eq!(response.status(), StatusCode::OK);

            observed
                .lock()
                .expect("observation mutex poisoned")
                .clone()
                .expect("the injected generate closure must have recorded an observation")
        }

        /// The `GenerateConfig` `lattice.rs`'s real `chat_completions` ->
        /// `prepare_chat_request`/`build_cfg`-equivalent construction (see
        /// its `let gen_cfg = ...` literal in this file) must produce for the
        /// fixed request body `run_observed` sends: every explicitly-set
        /// field mirrors the request, every other field is
        /// `GenerateConfig::default()` -- exactly like production's own
        /// `..Default::default()` tail.
        fn expected_gen_cfg() -> GenerateConfigSnapshot {
            GenerateConfigSnapshot::from(&lattice_inference::model::qwen35_config::GenerateConfig {
                max_new_tokens: 9,
                temperature: 1.3,
                top_p: 0.55,
                seed: Some(7),
                stop_strings: vec![],
                logprobs: None,
                ..Default::default()
            })
        }

        #[tokio::test]
        async fn chat_completions_non_streaming_observation_captures_real_config_and_prompt() {
            let obs = run_observed(true, OBSERVATION_GOLDEN_REQUEST_BODY).await;
            let expected_prompt_tokens = {
                let tokenizer = lattice_inference::model::qwen35::test_support::tiny_zero_model()
                    .tokenizer()
                    .clone();
                tokenizer
                    .tokenize(OBSERVATION_GOLDEN_USER_HI_THERE_CHATML)
                    .real_length
            };
            assert_observation_matches(
                &obs,
                &ExpectedObservation {
                    gen_cfg: expected_gen_cfg(),
                    rendered_prompt: Some(OBSERVATION_GOLDEN_USER_HI_THERE_CHATML),
                    messages: None,
                    prompt_tokens: expected_prompt_tokens,
                    stopped: true,
                },
            );
        }

        /// Proves `ProductionAdapterObservation::stopped` is genuinely
        /// derived from what the generation seam returned, not an
        /// independent hardcoded literal (the
        /// pre-fix `lattice_serve.rs` observation stored `stopped: true`
        /// unconditionally). Running the exact same request through
        /// `run_observed(false)` must observe `stopped == false`.
        #[tokio::test]
        async fn chat_completions_non_streaming_observation_captures_real_stopped_false() {
            let obs = run_observed(false, OBSERVATION_GOLDEN_REQUEST_BODY).await;
            assert!(
                !obs.stopped,
                "observation must report the seam's actual stopped=false, not a hardcoded true"
            );
        }

        /// #831 config-capture regression guard: a request that sets
        /// `reasoning_budget` must reach the real `GenerateConfig` this
        /// binary hands its generation adapter, not get silently dropped
        /// between `prepare_chat_request` and `gen_cfg` construction.
        /// Mutation-sensitive: dropping the `reasoning_budget` field
        /// assignment in `chat_completions_with_request`'s `gen_cfg`
        /// literal, or zeroing the parsed value, turns this `Some(5)`
        /// into `None`.
        #[tokio::test]
        async fn chat_completions_non_streaming_observation_captures_real_reasoning_budget() {
            let body = r#"{"model":"test-model","messages":[{"role":"user","content":"hi there"}],"temperature":1.3,"top_p":0.55,"seed":7,"max_tokens":9,"reasoning_budget":5}"#;
            let obs = run_observed(true, body).await;
            assert_eq!(
                obs.gen_cfg.reasoning_budget,
                Some(5),
                "reasoning_budget from the request must reach the real GenerateConfig"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Completion-length invariant accounts for reasoning_budget (#1334)
    //
    // Admission (`serve::contract::validate_context_window_with_budget`)
    // accepts `prompt + max_tokens + reasoning_budget + 1 <= max_context`,
    // but the post-generation invariant checked only `generated_tokens >
    // max_tokens` -- so a completion that legitimately spent its reasoning
    // allowance passed preflight, generated correctly, and was then reported
    // as an inference failure. These tests drive the real
    // `chat_completions_with_request` cascade (through the real `router()`,
    // for streaming's SSE framing) with an injected `CpuFakeGenerate`
    // closure that reports a caller-chosen `generated_tokens`, so the
    // engine's *reported* completion length can be placed exactly at or one
    // token past the decode budget independent of what the tiny model would
    // actually decode.
    // -----------------------------------------------------------------------
    #[cfg(feature = "test-utils")]
    mod completion_length_invariant_1334 {
        use super::*;
        use http_body_util::BodyExt as _;
        use tower::ServiceExt as _;

        /// `AppState` whose CPU generation is a canned `GenerateOutput`
        /// reporting exactly `generated_tokens`, regardless of the request.
        fn state_returning(max_tokens_cap: usize, generated_tokens: usize) -> AppState {
            let model = lattice_inference::model::qwen35::test_support::tiny_zero_model();
            AppState {
                model: ModelBackend::CpuFakeGenerate {
                    model: Arc::new(model),
                    generate: Arc::new(move |_prompt, _cfg, _on_token, _should_cancel| {
                        Ok(GenerateOutput {
                            text: "x".repeat(generated_tokens),
                            token_ids: vec![0; generated_tokens],
                            prompt_tokens: 1,
                            generated_tokens,
                            stopped: true,
                            stop_reason: Some(lattice_inference::StopReason::Length),
                            token_logprobs: vec![],
                        })
                    }),
                },
                default_max_tokens: max_tokens_cap,
                max_tokens_cap,
                model_id: "test-model".to_string(),
                request_counter: Arc::new(AtomicU64::new(0)),
                embedding_model: None,
            }
        }

        /// Builds the fixed `{"model":...,"messages":[...],"max_tokens":M[,
        /// "reasoning_budget":R][,"stream":true]}` request body. Omitting
        /// `reasoning_budget` (the "unset" half of the zero/unset cases)
        /// exercises a different code path through `prepare_chat_request`
        /// than sending an explicit `0` does, so callers choose which one a
        /// given case needs.
        fn request_body(
            max_tokens: usize,
            reasoning_budget: Option<usize>,
            stream: bool,
        ) -> String {
            let mut body = format!(
                r#"{{"model":"test-model","messages":[{{"role":"user","content":"hi"}}],"max_tokens":{max_tokens}"#
            );
            if let Some(budget) = reasoning_budget {
                body.push_str(&format!(r#","reasoning_budget":{budget}"#));
            }
            if stream {
                body.push_str(r#","stream":true"#);
            }
            body.push('}');
            body
        }

        async fn post(state: AppState, body: String) -> axum::http::Response<axum::body::Body> {
            let request = axum::http::Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body))
                .expect("fixture request must build");
            router(state)
                .oneshot(request)
                .await
                .expect("router must produce a response, not a transport error")
        }

        const MAX_TOKENS: usize = 10;
        const REASONING_BUDGET: usize = 4;

        // --- Non-streaming ---

        async fn assert_non_streaming(
            reasoning_budget: Option<usize>,
            generated_tokens: usize,
            expect_success: bool,
            case: &str,
        ) {
            let state = state_returning(64, generated_tokens);
            let body = request_body(MAX_TOKENS, reasoning_budget, false);
            let response = post(state, body).await;
            if expect_success {
                assert_eq!(
                    response.status(),
                    StatusCode::OK,
                    "{case}: generated_tokens={generated_tokens} at/under the decode budget \
                     must be accepted, not reported as an inference failure"
                );
            } else {
                assert_eq!(
                    response.status(),
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "{case}: generated_tokens={generated_tokens}, one past the decode budget, \
                     must still be rejected by the invariant"
                );
            }
        }

        #[tokio::test]
        async fn non_streaming_exact_acceptance_positive_budget() {
            // The +1 is the forced </think> delimiter (#1372): a positive
            // reasoning_budget allows rb + max_tokens + 1 generated tokens,
            // not rb + max_tokens.
            assert_non_streaming(
                Some(REASONING_BUDGET),
                MAX_TOKENS + REASONING_BUDGET + 1,
                true,
                "non-streaming, positive reasoning_budget, exact budget",
            )
            .await;
        }

        #[tokio::test]
        async fn non_streaming_one_past_rejection_positive_budget() {
            assert_non_streaming(
                Some(REASONING_BUDGET),
                MAX_TOKENS + REASONING_BUDGET + 2,
                false,
                "non-streaming, positive reasoning_budget, one past budget",
            )
            .await;
        }

        #[tokio::test]
        async fn non_streaming_exact_acceptance_unset_budget() {
            assert_non_streaming(
                None,
                MAX_TOKENS,
                true,
                "non-streaming, unset reasoning_budget, exact budget",
            )
            .await;
        }

        #[tokio::test]
        async fn non_streaming_one_past_rejection_unset_budget() {
            assert_non_streaming(
                None,
                MAX_TOKENS + 1,
                false,
                "non-streaming, unset reasoning_budget, one past budget",
            )
            .await;
        }

        // --- Streaming ---

        async fn assert_streaming(
            reasoning_budget: Option<usize>,
            generated_tokens: usize,
            expect_success: bool,
            case: &str,
        ) {
            let state = state_returning(64, generated_tokens);
            let body = request_body(MAX_TOKENS, reasoning_budget, true);
            let response = post(state, body).await;
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "{case}: an SSE response commits with 200 regardless of how generation \
                 concludes -- failure surfaces as an in-stream error event"
            );
            let bytes = response
                .into_body()
                .collect()
                .await
                .expect("SSE response body must be readable")
                .to_bytes();
            let text = String::from_utf8(bytes.to_vec()).expect("SSE body must be valid UTF-8");
            let has_error_event = text
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .any(|payload| payload.contains("\"error\""));
            // The fixture's `GenerateOutput.stopped` is `true` (`finish_reason_for`
            // -> `lattice_inference::serve::finish_reason` maps `stopped: true` to
            // `"stop"`, not `"length"` -- `stop_reason: Length` is set alongside it
            // but `finish_reason_for` derives only from `stopped`).
            let has_finish_reason = text.contains("\"finish_reason\":\"stop\"");
            if expect_success {
                assert!(
                    !has_error_event,
                    "{case}: generated_tokens={generated_tokens} at/under the decode budget \
                     must not emit an SSE error event; got: {text}"
                );
                assert!(
                    has_finish_reason,
                    "{case}: a successful completion must emit finish_reason \"stop\"; \
                     got: {text}"
                );
            } else {
                assert!(
                    has_error_event,
                    "{case}: generated_tokens={generated_tokens}, one past the decode budget, \
                     must still emit an SSE error event; got: {text}"
                );
                assert!(
                    !has_finish_reason,
                    "{case}: an invariant violation must not also emit a clean finish_reason; \
                     got: {text}"
                );
            }
        }

        #[tokio::test]
        async fn streaming_exact_acceptance_positive_budget() {
            // The +1 is the forced </think> delimiter (#1372): a positive
            // reasoning_budget allows rb + max_tokens + 1 generated tokens,
            // not rb + max_tokens.
            assert_streaming(
                Some(REASONING_BUDGET),
                MAX_TOKENS + REASONING_BUDGET + 1,
                true,
                "streaming, positive reasoning_budget, exact budget",
            )
            .await;
        }

        #[tokio::test]
        async fn streaming_one_past_rejection_positive_budget() {
            assert_streaming(
                Some(REASONING_BUDGET),
                MAX_TOKENS + REASONING_BUDGET + 2,
                false,
                "streaming, positive reasoning_budget, one past budget",
            )
            .await;
        }

        #[tokio::test]
        async fn streaming_exact_acceptance_unset_budget() {
            assert_streaming(
                None,
                MAX_TOKENS,
                true,
                "streaming, unset reasoning_budget, exact budget",
            )
            .await;
        }

        #[tokio::test]
        async fn streaming_one_past_rejection_unset_budget() {
            assert_streaming(
                None,
                MAX_TOKENS + 1,
                false,
                "streaming, unset reasoning_budget, one past budget",
            )
            .await;
        }
    }
}
