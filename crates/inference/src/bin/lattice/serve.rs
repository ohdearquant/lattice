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
// The canonical ChatML renderer is CPU-available (#668): this binary's
// CPU serve path renders normalized contract messages through the same
// formatter core the Metal worker uses, with no bespoke template copy.
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
use lattice_inference::forward::metal_qwen35::ChatMessage;
#[cfg(test)]
use lattice_inference::forward::metal_qwen35::format_chat_template;
use lattice_inference::serve::contract::ChatRequest as ChatCompletionRequest;
#[cfg(test)]
use lattice_inference::serve::contract::{
    ContentPart, GenerationDefaults, Message, MessageContent, ResponseFormat, ServeProfile,
    ValidatedChatRequest as ContractValidatedChatRequest, normalize_request,
};
#[cfg(test)]
use lattice_inference::serve::into_engine_chat_messages;
use lattice_inference::serve::prepare::PreparedChatRequest;
#[cfg(test)]
use lattice_inference::serve::prepare::prepare_chat_request;
use lattice_inference::serving_host::ServingHandle;
use lattice_inference::{GenerateOutput, TokenLogprob};
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Semaphore;

/// Request body cap: 1 MiB.  Requests above this return HTTP 413.
/// ADR-080 C2 (#782): `lattice_inference::serve::REQUEST_BODY_LIMIT_BYTES`
/// is the single shared constant now; both binaries previously carried
/// this exact value independently.
use lattice_inference::serve::REQUEST_BODY_LIMIT_BYTES;

// -----------------------------------------------------------------------
// Startup handoff: shared CPU host or affine worker
// -----------------------------------------------------------------------

/// Handle used while constructing the unified server's serving facade.
#[cfg(feature = "metal-gpu")]
#[derive(Clone)]
pub struct MetalHandle {
    client: lattice_inference::serve::metal_worker::MetalWorkerClient,
}

#[cfg(all(test, feature = "metal-gpu", feature = "test-utils"))]
fn map_metal_generation_error(error: ApiError) -> ApiError {
    lattice_inference::serving_host::map_worker_generation_error(error)
}

/// Startup result used to construct the HTTP serving facade.
#[derive(Clone)]
pub enum ModelBackend {
    Cpu(lattice_inference::serving_cpu_host::SharedCpuHandle),
    #[cfg(feature = "metal-gpu")]
    Metal {
        handle: MetalHandle,
        tokenizer: Arc<lattice_inference::tokenizer::bpe::BpeTokenizer>,
        max_context: usize,
    },
}

impl ModelBackend {
    /// Build the shared request facade after the startup route has loaded.
    pub(super) fn into_serving_handle(
        self,
        generation_admission: Arc<Semaphore>,
    ) -> lattice_inference::serving_host::ServingHandle {
        match self {
            Self::Cpu(cpu) => lattice_inference::serving_host::ServingHandle::shared_cpu(
                cpu,
                generation_admission,
            ),
            #[cfg(feature = "metal-gpu")]
            Self::Metal {
                handle,
                tokenizer,
                max_context,
            } => lattice_inference::serving_host::ServingHandle::affine_cli(
                handle.client,
                tokenizer,
                max_context,
            ),
        }
    }

    /// Load a native Q4 checkpoint on the shared Metal worker thread
    /// (`lattice_inference::serve::metal_worker::MetalWorker`, issue
    /// #832 — the same shared owner `lattice_serve.rs` uses) and return
    /// the startup handle plus the resolved context
    /// window, for `main()`'s `Command::Serve` startup sequence.
    #[cfg(feature = "metal-gpu")]
    pub(super) fn spawn_metal(
        model_dir: std::path::PathBuf,
        tokenizer_dir: Option<std::path::PathBuf>,
        max_pending: usize,
        residency_limits: lattice_inference::serve::lora::ResidencyLimits,
        preload_vision: bool,
    ) -> Result<(Self, usize), String> {
        use lattice_inference::serve::metal_worker::{
            ContextWindowPolicy, MetalWorker, StartupError, VisionRuntime, WorkerMetadata,
        };
        use lattice_inference::serving_factory::ServingFactory;

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
        // `providers::qwen::worker::load_model`). Passed to the shared
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
        let (owner, client, _meta, _preparation) = MetalWorker::spawn_with_vision(
            ServingFactory::qwen_metal(
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
            ),
            max_pending,
            residency_limits,
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

/// How many routed completions the episode store holds.
///
/// A window, not a history. Sized so an ordinary interactive session's
/// feedback arrives while its episode is still present; a deployment that
/// needs a longer one needs a durable store, which is a schema and a retention
/// policy and is not decided by ADR-096.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub const DEFAULT_EPISODE_CAPACITY: usize = 4096;

/// A fresh episode store at the default capacity.
///
/// One constructor rather than a literal per `AppState` site: a second
/// capacity written at a call site is a second policy nobody knows exists.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub fn episode_store() -> Arc<std::sync::Mutex<lattice_inference::router_episode::EpisodeStore>> {
    Arc::new(std::sync::Mutex::new(
        lattice_inference::router_episode::EpisodeStore::new(DEFAULT_EPISODE_CAPACITY),
    ))
}

#[cfg(test)]
pub type DefaultAppStateModel = ModelBackend;
#[cfg(not(test))]
pub type DefaultAppStateModel = lattice_inference::serving_host::ServingHandle;

/// A construction-time model value converted to the serving facade at startup.
pub(super) trait IntoServingHandle {
    fn into_serving_handle(
        self,
        generation_admission: Arc<Semaphore>,
    ) -> lattice_inference::serving_host::ServingHandle;
}

impl IntoServingHandle for lattice_inference::serving_host::ServingHandle {
    fn into_serving_handle(
        self,
        _generation_admission: Arc<Semaphore>,
    ) -> lattice_inference::serving_host::ServingHandle {
        self
    }
}

#[cfg(test)]
impl IntoServingHandle for ModelBackend {
    fn into_serving_handle(
        self,
        generation_admission: Arc<Semaphore>,
    ) -> lattice_inference::serving_host::ServingHandle {
        ModelBackend::into_serving_handle(self, generation_admission)
    }
}

/// State shared across all request handlers via axum's `State` extractor.
#[derive(Clone)]
pub struct AppState<M = DefaultAppStateModel> {
    /// The shared execution and capability facade.
    pub model: M,
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
    /// Bounded concurrent-job admission for `/v1/embeddings`'s pooled
    /// `embed_items` `spawn_blocking` calls (issue #1397), mirroring
    /// `lattice_serve.rs`'s `EmbeddingState::admission` for its own
    /// `/v1/embeddings` route: see [`EMBEDDING_MAX_CONCURRENT_JOBS`].
    /// Always constructed, independent of whether `embedding_model` is
    /// `Some` -- when no vision-language checkpoint is loaded, `embeddings()`
    /// already fails closed on `embedding_model` before this is ever
    /// touched.
    pub embedding_admission: Arc<Semaphore>,
    /// Bounded concurrent-generation admission for the CPU chat backend
    /// (see [`CPU_GENERATION_MAX_CONCURRENT_JOBS`]). A resource guard, not a
    /// throughput setting: each admitted generation holds one serving runtime
    /// and its KV cache, and concurrent generations contend for the same CPU
    /// cores. The shared serving facade receives this same semaphore for CPU
    /// generations while affine workers retain their own admission permits.
    pub cpu_generation_admission: Arc<Semaphore>,
    /// The router gate loaded at startup from `--router-state`, or `None` on a
    /// server with no router configured.
    ///
    /// `None` is a served state, not a failure: a request that omits `lora`
    /// selects the base model, which is what omitting the field has always
    /// done. What is NOT a served state is a configured router that would not
    /// load — startup refuses that rather than arriving here as `None`, since
    /// the two are indistinguishable from this field alone.
    ///
    /// Gated to Metal builds for the same reason every other adapter surface
    /// here is: routing selects resident adapters, and a build without Metal
    /// cannot make one resident (see `adapter_unsupported_build`). A
    /// `--router-state` passed to such a build is refused at startup rather
    /// than loaded into a field nothing on that build could ever apply.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub router_state: Option<Arc<lattice_inference::serve::routing::ServedRouter>>,
    /// Routed completions and the judgements against them (ADR-096 D3).
    ///
    /// Present on every Metal build whether or not a router is configured, and
    /// it stays empty on a server with no gate: `route_selection` is the only
    /// producer, and it does not run without one. A field that appeared and
    /// disappeared with the router would make "no episodes" and "no store"
    /// two states the feedback route answers identically.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub episodes: Arc<std::sync::Mutex<lattice_inference::router_episode::EpisodeStore>>,
}

impl<M: IntoServingHandle> AppState<M> {
    fn into_serving_handle(self) -> AppState<lattice_inference::serving_host::ServingHandle> {
        let Self {
            model,
            default_max_tokens,
            max_tokens_cap,
            model_id,
            request_counter,
            embedding_model,
            embedding_admission,
            cpu_generation_admission,
            #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            router_state,
            #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            episodes,
        } = self;
        let model = model.into_serving_handle(Arc::clone(&cpu_generation_admission));
        AppState {
            model,
            default_max_tokens,
            max_tokens_cap,
            model_id,
            request_counter,
            embedding_model,
            embedding_admission,
            cpu_generation_admission,
            #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            router_state,
            #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            episodes,
        }
    }
}

/// See [`AppState::embedding_admission`]. Same cap `lattice_serve.rs` uses
/// for its own pooled embedding route (`EMBEDDING_MAX_CONCURRENT_JOBS`
/// there): each admitted job here is additionally bounded to
/// `lattice_inference::serve::embeddings::MAX_EMBEDDING_TOTAL_TOKENS`
/// aggregate scaffold tokens by the budget check `embeddings()` runs inside
/// the admitted `spawn_blocking` job (see
/// `check_pooled_embedding_total_tokens`'s doc comment for why that check
/// runs after admission rather than before it, unlike the sibling route), so
/// aggregate pooled-decoder work across concurrent requests is bounded by
/// (this cap) x (that per-request budget) rather than growing with however
/// many requests arrive at once. Fixed rather than a CLI flag: like the
/// sibling route, CPU-bound pooled forward passes have no hardware queue
/// depth to model, this cap exists purely to bound aggregate memory and CPU
/// contention.
pub(crate) const EMBEDDING_MAX_CONCURRENT_JOBS: usize = 4;

/// Acquires one pooled-embedding admission slot for
/// [`AppState::embedding_admission`], mirroring `lattice_serve.rs`'s
/// `try_acquire_embedding_slot`. Synchronous and non-blocking: a full
/// admission cap is rejected immediately, before any tokenization, image
/// preprocessing, or `spawn_blocking` work runs.
///
/// # Errors
///
/// Returns [`ApiError::ServiceUnavailable`] (`server_busy`) when
/// [`EMBEDDING_MAX_CONCURRENT_JOBS`] jobs are already outstanding.
fn try_acquire_embedding_slot(
    admission: &Arc<Semaphore>,
) -> Result<tokio::sync::OwnedSemaphorePermit, ApiError> {
    admission
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::ServiceUnavailable {
            message: "too many outstanding /v1/embeddings requests; retry shortly".to_string(),
        })
}

/// Default cap on concurrent CPU chat generations
/// ([`AppState::cpu_generation_admission`]), overridable with `lattice serve
/// --max-cpu-generations`. Small on purpose: it is a resource guard rather
/// than a throughput setting, bounding the memory each admitted generation
/// holds (one serving runtime and its KV cache) and the CPU contention
/// between concurrent generations. An operator with spare cores and memory can
/// raise it.
pub const CPU_GENERATION_MAX_CONCURRENT_JOBS: usize = 2;

/// Runs one pooled-embedding job (`check_pooled_embedding_total_tokens` then
/// `embed_items`) on the blocking thread pool with `permit` moved into the
/// closure, so the admission slot releases
/// when the job finishes rather than when the caller's `.await` is dropped
/// -- mirrors `lattice_serve.rs`'s `run_embedding_encode`, for the identical
/// reason: a `spawn_blocking` task already under way is not cancelled by
/// dropping its `JoinHandle` (e.g. a disconnected client dropping the
/// request future), so a permit held only in the caller's stack frame would
/// release on disconnect while the job kept running -- letting repeated
/// disconnects admit unbounded concurrent jobs past the cap. Generic over
/// the closure's return type (unlike the sibling, which is concrete to its
/// own `Vec<Vec<f32>>` result) since `embed_items` returns an
/// `(EmbeddingDatum, EmbeddingsUsage)` pair already wrapped in `ApiError`
/// rather than `InferenceError`.
async fn run_pooled_embedding_job<T: Send + 'static>(
    permit: tokio::sync::OwnedSemaphorePermit,
    work: impl FnOnce() -> Result<T, ApiError> + Send + 'static,
) -> Result<Result<T, ApiError>, tokio::task::JoinError> {
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        work()
    })
    .await
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
pub(super) fn finish_reason_for(output: &lattice_inference::GenerateOutput) -> &'static str {
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
pub async fn list_models(
    State(state): State<AppState<lattice_inference::serving_host::ServingHandle>>,
) -> Json<Value> {
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
/// `to_bytes(.., REQUEST_BODY_LIMIT_BYTES)` below is the SOLE enforcement
/// of the request body cap on this route, not a second layer on top of
/// the router's `DefaultBodyLimit::max(REQUEST_BODY_LIMIT_BYTES)`.
/// `DefaultBodyLimit` works by inserting a request extension that only
/// extractors built on `Bytes`/`BytesMut` -- and anything layered on
/// them, e.g. `Json<T>`, `Form<T>`, multipart -- read via
/// `RequestExt::into_limited_body` (axum-core 0.8's
/// `extract::request_parts` and `ext_traits::request`). `Body`'s own
/// `FromRequest` impl returns the request body unwrapped
/// (`req.into_body()`) and never consults that extension, so the layer
/// does not bound a handler that takes `body: axum::body::Body`
/// directly, this one included. The existing 413 behavior below comes
/// entirely from this explicit `to_bytes` call.
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
    State(state): State<AppState<lattice_inference::serving_host::ServingHandle>>,
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
#[allow(clippy::field_reassign_with_default)]
async fn chat_completions_with_request<M: IntoServingHandle>(
    State(state): State<AppState<M>>,
    req: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    chat_completions_with_serving_request(State(state.into_serving_handle()), req).await
}

#[allow(clippy::field_reassign_with_default)]
async fn chat_completions_with_serving_request(
    State(state): State<AppState<lattice_inference::serving_host::ServingHandle>>,
    req: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    // Resolved ONCE here, upstream of every branch below, so the two submit
    // sites in this handler and the two `registry.apply` sites in the worker all
    // read one value. ADR-094's amendment splits the field into three states;
    // resolving per branch would reintroduce the sibling-path drift the shared
    // resolver exists to remove.
    let requested = lattice_inference::serve::lora::requested_adapters(req.lora.clone());
    lattice_inference::serve::lora::validate_scales(requested.selection())?;
    if !requested.selection().is_empty() {
        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        adapter_client(&state)?.validate_lora(requested.selection())?;
        #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
        return Err(adapter_unsupported_build());
    }
    let (
        PreparedChatRequest {
            messages: _normalized_messages,
            max_tokens,
            logprobs,
            prompt,
            reasoning_budget,
            stream,
            ..
        },
        gen_cfg,
    ) = state.model.prepare_lattice_chat_request(
        &req,
        &state.model_id,
        state.default_max_tokens,
        state.max_tokens_cap,
    )?;
    #[cfg(feature = "metal-gpu")]
    let explicitly_thinking = lattice_inference::serve::prepare::explicitly_enabled_thinking(&req)?;
    #[cfg(not(feature = "metal-gpu"))]
    let explicitly_thinking = false;

    // Metal-only: reuse the exact messages normalized alongside the CPU
    // prompt, so role/content validation and allocation happen once.
    #[cfg(feature = "metal-gpu")]
    let chat_messages = _normalized_messages;

    // ADR-093 decision 1: the selection is decided ONCE, here at prefill, and held for every token
    // this request generates. Placed after normalization so the gate reads the same text the model
    // will, rather than a second rendering of the request that could drift from it.
    //
    // Rebinds the RESOLVED value rather than writing back to `req.lora`. Every consumer below
    // reads `requested`, which was resolved above, so a write to the raw field here would be
    // discarded in silence -- the request would route, report that it routed, and serve the base
    // model.
    //
    // `is_routable` is the whole condition. An explicit list is a stated intent and an explicit
    // `[]` is a caller pinning the base model; routing over either would overrule the caller while
    // looking like a default.
    //
    // The routed selection takes the same two checks the caller's field took above, because it is
    // a different producer of the same value and those checks ran before it existed. An empty
    // result means the gate declined (no user turn to read), which is the base model -- the
    // original resolution already says that, so it stands rather than being restated as an
    // `Explicit` list that names nothing.
    //
    // Declared here and filled only on the branch that actually applies a
    // routed mixture. A request that was not routed, or was routed to nothing,
    // leaves this `None` and records no episode: putting it in the learner's
    // population would be a completion no gate produced.
    //
    // Shared response metadata, minted BEFORE routing rather than after. It
    // used to sit below the routing block, where it read more naturally:
    // nothing above needed it. An exploration draw does. The seed is derived
    // from the completion id so that an episode replays from the episode alone,
    // which requires the id to exist when the draw happens. The cost of the
    // move is that a request refused during routing now consumes a sequence
    // number, so the ids a server emits can skip -- `seq` exists to make the id
    // unique, nothing counts with it, and a gap is the honest record of a
    // request that was admitted and then refused.
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let seq = state.request_counter.fetch_add(1, Ordering::Relaxed);
    let response_id = format!("chatcmpl-{created}-{seq}");

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let mut routing_provenance: Option<lattice_inference::serve::routing::RoutedRequest> = None;
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    let requested = match state
        .router_state
        .clone()
        .filter(|_| requested.is_routable())
    {
        None => requested,
        Some(served) => {
            let decision = route_selection(&state, &served, &chat_messages, &response_id).await?;
            match &decision {
                Some(routed) if !routed.selections.is_empty() => {
                    lattice_inference::serve::lora::validate_scales(&routed.selections)?;
                    adapter_client(&state)?.validate_lora(&routed.selections)?;
                    routing_provenance = decision.clone();
                    lattice_inference::serve::lora::RequestedAdapters::Explicit(
                        routed.selections.clone(),
                    )
                }
                _ => requested,
            }
        }
    };
    let execution_request = lattice_inference::serving_host::LatticeExecutionRequest {
        prompt,
        config: gen_cfg,
        #[cfg(feature = "metal-gpu")]
        messages: Some(chat_messages),
        #[cfg(not(feature = "metal-gpu"))]
        messages: None,
        selection: requested.selection().to_vec(),
        explicitly_thinking,
    };

    // Recorded here because this is the first moment both halves exist: the
    // id a judgement will name was minted before routing, and the routing
    // decision it seeded was made above. Recording later, after generation,
    // would lose every episode whose completion failed -- and a mixture that
    // produced a failure is exactly the one a learner should be able to hear
    // about.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    if let (Some(routed), Some(served)) = (&routing_provenance, &state.router_state) {
        use lattice_inference::router_episode::{PolicyIdentity, RouterEpisode};
        let (artifact_version, schema_id) = served.with_report(|report| {
            (
                report.artifact.version_label(),
                report
                    .artifact
                    .provenance
                    .lineage()
                    .map(|l| l.schema_id.clone())
                    // A legacy artifact records no schema. Named as absent
                    // rather than defaulted to the empty string, which would
                    // read as a schema whose name is unprintable.
                    .unwrap_or_else(|| "unrecorded".to_string()),
            )
        });
        let episode = RouterEpisode {
            completion_id: response_id.clone(),
            policy: PolicyIdentity {
                artifact_version,
                schema_id,
                // The epoch the decision was made under, carried out of the
                // routing call rather than read again here: between the two
                // reads a reload could land, and this episode would be dated
                // to a policy that did not produce it.
                epoch: routed.policy_epoch,
            },
            adapter_names: served.with_report(|report| report.artifact.adapter_names.clone()),
            context_vector: routed.context_vector.clone(),
            selected_scores: routed.selected_scores.clone(),
            sampled_scores: routed.sampled_scores.clone(),
            pre_floor_weights: routed.pre_floor_weights.clone(),
            applied_weights: routed.applied_weights.clone(),
            dropped: routed.dropped.clone(),
            tau: routed.tau,
            #[cfg(feature = "mixture")]
            exploration: routed.exploration.clone(),
        };
        // A poisoned store loses episodes and must not lose the request: the
        // completion is the product, the episode is telemetry about it.
        match state.episodes.lock() {
            Ok(mut store) => store.record(episode),
            Err(_) => eprintln!(
                "[router] the episode store panicked on an earlier request; \
                 completion {response_id} is served but not recorded"
            ),
        }
    }

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

        // Keep the completion-budget invariant and finish reason shared by
        // both execution hosts.
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

        let delta_tx = tx.clone();
        let failure_tx = tx.clone();
        state
            .model
            .stream_lattice(
                execution_request,
                cancel_rx,
                move |delta| {
                    delta_tx
                        .unbounded_send(StreamMsg::Delta(delta.to_owned()))
                        .is_ok()
                },
                move |result| match result {
                    Ok(output) => finish_streaming(output),
                    Err(_) => {
                        let _ = failure_tx.unbounded_send(StreamMsg::Failed);
                    }
                },
            )
            .await?;

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
        // --- Non-streaming path ---
        let output = state.model.complete_lattice(execution_request).await?;

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
        let choice_logprobs = match (logprobs, state.model.tokenizer()) {
            (Some(_), Some(tokenizer)) => {
                Some(build_choice_logprobs(tokenizer, &output.token_logprobs))
            }
            _ => None,
        };

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
    State(state): State<AppState<lattice_inference::serving_host::ServingHandle>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
) -> Result<Response, ApiError> {
    use lattice_inference::serve::embeddings::{
        EmbeddingsRequest, check_pooled_embedding_total_tokens, embed_items,
        normalize_embedding_items, parse_pooling,
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

    // #1397: bounded concurrent-job admission FIRST, mirroring
    // `lattice_serve.rs`'s `try_acquire_embedding_slot` / `run_embedding_encode`
    // -- rejects with 503 `server_busy` before any tokenization, image
    // preprocessing, or `spawn_blocking` work runs, and the permit is moved
    // into the blocking closure so a disconnected client cannot free the
    // slot before the job itself finishes. Deliberately NOT mirroring the
    // sibling route's within-request ORDER of budget-then-admission: see
    // `check_pooled_embedding_total_tokens`'s doc comment for why this
    // route's aggregate token-budget check has to run AFTER admission,
    // inside the job below, instead of before it.
    let permit = try_acquire_embedding_slot(&state.embedding_admission)?;

    let (data, usage) = run_pooled_embedding_job(permit, move || {
        // #1397: aggregate token-budget check, mirroring `lattice_serve.rs`'s
        // `check_embeddings_total_tokens` -- runs inside the admitted job,
        // before `embed_items`, so a request already admitted past the
        // concurrency cap is still rejected before its actual pooled-decoder
        // work starts.
        check_pooled_embedding_total_tokens(&embedder, &items)?;
        embed_items(&embedder, items, pooling)
    })
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
// Runtime LoRA adapter control (Ocean 2026-09-16)
// -----------------------------------------------------------------------

/// Why this server cannot take an adapter.
///
/// Adapter loading is a Metal-engine capability: the single-slot adapter API lives
/// on `MetalQwen35State` and there is no CPU equivalent, so a CPU-backed server, or
/// a build without Metal at all, refuses rather than pretending. The refusal is a
/// 400 naming the condition, the same shape `/v1/embeddings` uses when the loaded
/// checkpoint cannot serve it: the route exists, and its unavailability is a
/// discoverable server state rather than a bare 404.
fn lora_unsupported_backend(message: &'static str) -> ApiError {
    ApiError::BadRequest {
        message: message.to_string(),
        code: "lora_unsupported_backend",
    }
}

/// The Metal worker client behind a Metal-backed server, or the refusal above.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn adapter_client(
    state: &AppState<lattice_inference::serving_host::ServingHandle>,
) -> Result<&lattice_inference::serve::metal_worker::MetalWorkerClient, ApiError> {
    state.model.adapter_client_or(lora_unsupported_backend(
        "runtime LoRA adapters require the Metal backend; restart this server with a \
         Metal Q4 checkpoint to enable this route",
    ))
}

/// The refusal a build without Metal owes every adapter request.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
fn adapter_unsupported_build() -> ApiError {
    lora_unsupported_backend(
        "this server was built without Metal support; runtime LoRA adapters require a macOS \
         Metal build",
    )
}

/// List confirmed resident adapters and the currently applied mixture.
pub async fn lora_list<M: IntoServingHandle>(
    State(state): State<AppState<M>>,
) -> Result<Response, ApiError> {
    let state = state.into_serving_handle();
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        lora_list_with_backend(
            Ok(()),
            || adapter_client(&state),
            lattice_inference::serve::metal_worker::MetalWorkerClient::adapter_index,
            state.router_state.as_deref(),
        )
    }
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        let _ = state;
        lora_list_with_backend::<()>(
            Err(adapter_unsupported_build()),
            || Err(adapter_unsupported_build()),
            |()| lattice_inference::serve::lora::AdapterIndex::default(),
            None,
        )
    }
}

/// Choose the adapters for one request from the learned gate.
///
/// FAIL CLOSED WHEN THE TRAINED SET IS NOT RESIDENT, which is a decision and not a detail. The
/// gate's columns are labelled by the artifact's adapter names, so a set that does not match
/// cannot be routed at all. The alternative -- skip routing and serve the base model -- is the
/// failure this ADR family refuses everywhere else: the operator configured routing, the server
/// says routing is enabled, and the only place the truth appears is in the answers. The refusal
/// names both lists side by side, so "which adapter moved" is answered where it is asked.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
async fn route_selection(
    state: &AppState<lattice_inference::serving_host::ServingHandle>,
    served: &Arc<lattice_inference::serve::routing::ServedRouter>,
    messages: &[ChatMessage],
    completion_id: &str,
) -> Result<Option<lattice_inference::serve::routing::RoutedRequest>, ApiError> {
    use lattice_inference::serve::routing::{PromptSource, context_text};

    // No user turn means no text under the rule the gate was trained on. Serving the base model is
    // right here and is NOT the silent degradation above: there is nothing to route ON, rather
    // than a routable request being quietly skipped.
    let turns = messages.iter().map(|m| {
        (
            matches!(
                m.role,
                lattice_inference::forward::metal_qwen35::ChatRole::User
            ),
            m.content.as_str(),
        )
    });
    let Some(text) = context_text(PromptSource::SERVED, turns) else {
        // `None` rather than an empty selection: a request with nothing to
        // route on produced no routing decision at all, and recording an
        // episode for it would put a completion in the learner's population
        // that no gate ever saw.
        return Ok(None);
    };

    // Startup refuses a router without an embedding model, so this is unreachable rather than
    // merely unlikely -- and it says so instead of unwrapping.
    let embedder = state.embedding_model.clone().ok_or_else(|| ApiError::Internal {
        message: "routing is configured but no embedding model is loaded; startup should have                   refused this configuration"
            .to_string(),
    })?;

    let pooling = served.pooling();
    let owned = text.to_string();
    // Same convention as the embeddings route: the pooled forward pass is synchronous CPU work and
    // does not belong on an async worker thread.
    let context_vector = tokio::task::spawn_blocking(move || embedder.embed_text(&owned, pooling))
        .await
        .map_err(|_| ApiError::Internal {
            message: "the routing embedder panicked".to_string(),
        })?
        .map_err(|err| ApiError::Internal {
            message: format!("could not embed the routing context: {err}"),
        })?;

    let resident = adapter_client(state)?.adapter_index();
    // Every trained column. Top-k would need a number this decision has no evidence for, and
    // `trained_order` has already required the resident set to match the artifact's, so this width
    // is always routable.
    //
    // What this does NOT do, stated because an earlier version of this comment said the gate
    // decides the weight here: it does not, and that is ADR-091 decision 1, not an oversight.
    // Uniform `1/k` is the default until that ADR's evidence gate passes. `ServingRouter::new`
    // builds the router with `AdapterRouter::new`, whose `WeightPolicy::default()` is `Uniform`;
    // `set_weight_policy` is called only by `apply_recorded_policy`, which requires the artifact to
    // record a learned policy AND `--router-learn`. Absent both, every selected adapter receives
    // `1.0 / k` and, with `k` equal to the column count, the gate's scores change neither the
    // selection nor the weights: a response from this path is the same for any gate, a zeroed one
    // included, which is what any acceptance run over it can and cannot claim. Turning the softmax
    // on is ADR-091's evidence gate, made an explicit startup flag rather than a default.
    let k = served.with_report(|report| report.artifact.adapter_names.len());
    // The vector the gate consumed goes in by value, so the decision carries
    // the input it was made from. ADR-096 decision 3: the server holds this,
    // and a caller cannot supply one in its place -- a caller's vector would
    // describe a mixture the server never formed.
    // `completion_id` seeds the draw when the recorded policy explores; it is
    // the id the response will carry, so the seed the episode records replays
    // from the episode alone.
    Ok(Some(served.route(
        &resident,
        context_vector,
        k,
        completion_id,
    )?))
}

/// `POST /v1/router/feedback` — judge one routed completion (ADR-096 D3).
///
/// The body carries a completion id and a signal, and nothing else. Everything
/// a learner needs about the decision was captured by the server when the
/// decision was made, because a completion produced by a weighted mixture does
/// not identify an adapter index and a caller cannot supply one without
/// inventing credit the feedback never gave.
///
/// Three successes and two refusals, and the three successes are different
/// answers rather than one. `recorded` is an on-policy judgement. `idempotent`
/// is a retry of the same signal and changed nothing. `stale` means the
/// judgement is real and counted and will not be used as an on-policy sample,
/// because the policy epoch it belongs to has ended. Collapsing stale into
/// success would silently feed a learner samples from a gate that no longer
/// serves; collapsing it into a failure would tell a caller its feedback was
/// lost, which is also untrue.
pub async fn router_feedback(
    State(state): State<AppState<lattice_inference::serving_host::ServingHandle>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
) -> Result<Json<serde_json::Value>, ApiError> {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        let _ = (&state, headers, body);
        Err(adapter_unsupported_build())
    }
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        use lattice_inference::router_episode::FeedbackRequest;

        lattice_inference::serve::require_json_content_type(&headers)?;
        let bytes = axum::body::to_bytes(body, REQUEST_BODY_LIMIT_BYTES)
            .await
            .map_err(|_| ApiError::BadRequest {
                message: "invalid request body".into(),
                code: "invalid_request",
            })?;
        let request: FeedbackRequest =
            serde_json::from_slice(&bytes).map_err(|err| ApiError::BadRequest {
                message: format!("invalid feedback body: {err}"),
                code: "invalid_feedback_body",
            })?;
        let Some(signal) = request.parsed_signal() else {
            return Err(ApiError::BadRequest {
                message: format!(
                    "signal {:?} is not one this server can act on; it is \"helpful\" or \"unhelpful\"",
                    request.signal
                ),
                code: "invalid_feedback_signal",
            });
        };

        // Read before the store's lock is taken, never while holding it: the
        // two locks are never nested in either order, here or on the chat
        // path. `None` when no gate is loaded, which the store reads as "the
        // policy that produced this has ended" -- a stand-in number here would
        // mark every episode on-policy for a router that does not exist.
        let current_epoch = state
            .router_state
            .as_ref()
            .map(|served| served.policy_epoch());
        let mut store = state.episodes.lock().map_err(|_| ApiError::Internal {
            message: "the episode store panicked on an earlier request and is no longer recording"
                .to_string(),
        })?;
        let outcome = store
            .judge(&request.completion_id, signal, current_epoch)
            .map_err(|err| ApiError::BadRequest {
                code: lattice_inference::router_episode::judgement_error_code(&err),
                message: err.to_string(),
            })?;
        Ok(Json(lattice_inference::router_episode::feedback_response(
            &request.completion_id,
            signal,
            outcome,
        )))
    }
}

/// `GET /v1/router/episodes` — the episode census (ADR-096 D3).
///
/// Counts, not a rate. The honest denominator for a feedback rate is
/// contested, and judged-over-recorded describes response behaviour rather
/// than utility; reporting only that invites reading it as coverage of the
/// population. `unjudged` is published alongside because missing feedback is
/// missing and not zero, and `evicted` because a census taken over a bounded
/// window is a window, which a reader cannot otherwise tell.
pub async fn router_episodes(
    State(state): State<AppState<lattice_inference::serving_host::ServingHandle>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        let _ = &state;
        Err(adapter_unsupported_build())
    }
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        let store = state.episodes.lock().map_err(|_| ApiError::Internal {
            message: "the episode store panicked on an earlier request and is no longer recording"
                .to_string(),
        })?;
        let census = store.census();
        Ok(Json(lattice_inference::router_episode::census_response(
            &census,
        )))
    }
}

/// Make a PEFT or MLX adapter resident without applying it to generation.
///
/// Repeated exact `(name, path)` identities reuse the resident id. One unload
/// removes it for every caller; aliases and changed file contents are not detected.
/// This unauthenticated route reads a caller-selected path on the server host.
pub async fn lora_load(
    State(state): State<AppState<lattice_inference::serving_host::ServingHandle>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
) -> Result<Response, ApiError> {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        let _ = state;
        lora_load_with_backend::<(), ()>(
            headers,
            body,
            Err(adapter_unsupported_build()),
            || Err(adapter_unsupported_build()),
            |_, _| Err(adapter_unsupported_build()),
            |(), ()| Err(adapter_unsupported_build()),
        )
        .await
    }
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        lora_load_with_backend(
            headers,
            body,
            Ok(()),
            || adapter_client(&state),
            lattice_inference::serve::lora::prepare_adapter_load,
            |client, prepared| client.submit_adapter_command(prepared.command),
        )
        .await
    }
}

/// Remove one resident identifier, refusing unknown ids.
pub async fn lora_unload(
    State(state): State<AppState<lattice_inference::serving_host::ServingHandle>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
) -> Result<Response, ApiError> {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        let _ = state;
        lora_unload_with_backend::<()>(
            headers,
            body,
            Err(adapter_unsupported_build()),
            || Err(adapter_unsupported_build()),
            |(), _| Err(adapter_unsupported_build()),
        )
        .await
    }
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        lora_unload_with_backend(
            headers,
            body,
            Ok(()),
            || adapter_client(&state),
            |client, id| {
                client.submit_adapter_command(
                    lattice_inference::serve::metal_worker::AdapterCommand::Unload { id },
                )
            },
        )
        .await
    }
}

async fn lora_load_with_backend<B, P>(
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
    build_admission: Result<(), ApiError>,
    resolve_backend: impl FnOnce() -> Result<B, ApiError>,
    prepare_load: impl FnOnce(&str, &str) -> Result<P, ApiError>,
    submit_load: impl FnOnce(
        B,
        P,
    )
        -> Result<lattice_inference::serve::lora::AdapterControlReceiver, ApiError>,
) -> Result<Response, ApiError> {
    lattice_inference::serve::lora::load_request(
        headers,
        body,
        build_admission,
        resolve_backend,
        prepare_load,
        submit_load,
    )
    .await
}

async fn lora_unload_with_backend<B>(
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
    build_admission: Result<(), ApiError>,
    resolve_backend: impl FnOnce() -> Result<B, ApiError>,
    submit_unload: impl FnOnce(
        B,
        u32,
    ) -> Result<
        lattice_inference::serve::lora::AdapterControlReceiver,
        ApiError,
    >,
) -> Result<Response, ApiError> {
    lattice_inference::serve::lora::unload_request(
        headers,
        body,
        build_admission,
        resolve_backend,
        submit_unload,
    )
    .await
}

fn lora_list_with_backend<B>(
    build_admission: Result<(), ApiError>,
    resolve_backend: impl FnOnce() -> Result<B, ApiError>,
    snapshot: impl FnOnce(B) -> lattice_inference::serve::lora::AdapterIndex,
    router: Option<&lattice_inference::serve::routing::ServedRouter>,
) -> Result<Response, ApiError> {
    lattice_inference::serve::lora::list_request(build_admission, resolve_backend, snapshot, router)
}

// -----------------------------------------------------------------------
// Router
// -----------------------------------------------------------------------

pub(super) fn router<M: IntoServingHandle>(state: AppState<M>) -> Router {
    let state = state.into_serving_handle();
    Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/lora", get(lora_list::<ServingHandle>))
        .route("/v1/lora/load", post(lora_load))
        .route("/v1/lora/unload", post(lora_unload))
        .route("/v1/router/feedback", post(router_feedback))
        .route("/v1/router/episodes", get(router_episodes))
        .layer(DefaultBodyLimit::max(REQUEST_BODY_LIMIT_BYTES))
        .with_state(state)
}

// -----------------------------------------------------------------------
// Tests — pure helper functions; no model construction needed
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests;

// The CPU-panic HTTP/SSE response tests live in their own file so that rustfmt
// formats them: rustfmt does not follow `include!`. Declaring the file as a
// module (rather than splicing its tokens into `post_drop_cancellation_probe`)
// also gives the tests their own namespace instead of the unrelated probe's.
// `#[path]` is relative to this file's own directory; the file sits beside it.
#[cfg(all(test, feature = "test-utils"))]
#[path = "serve_cpu_panic_tests.rs"]
mod cpu_panic_tests;
