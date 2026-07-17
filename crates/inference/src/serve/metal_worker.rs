//! Shared Metal GPU worker owner for the serve layer (issue #832, ADR-080
//! cluster C2/C3): the single dedicated thread that owns the `!Send`
//! `MetalQwen35State` for the whole process lifetime, replacing the two
//! previously-independent copies of this loop --
//! `lattice.rs`'s `MetalJob`/`MetalHandle` and `lattice_serve.rs`'s
//! `Job`/`spawn_worker`/`run_worker_loop`.
//!
//! # Why this existed twice
//!
//! Both binaries implement the same lifecycle: load the model on a
//! dedicated OS thread (the Metal state can never cross a thread boundary),
//! serve `Job`s FIFO from an unbounded channel, check a per-job
//! disconnect-cancellation signal before paying for any prefill work, reuse
//! the single process-wide [`CrossTurnSlotId::DEFAULT`] cache slot, and
//! stream token deltas back to the HTTP handler. Only comments -- not
//! shared code -- kept the two copies in sync, and they had already drifted:
//! on dequeue-time cancellation, `lattice.rs`'s worker sent an empty
//! interrupted `GenerateOutput` before moving on; `lattice_serve.rs`'s
//! worker silently dropped the job with no reply at all. This module picks
//! ONE contract -- an explicit [`WorkerEvent::Cancelled`] terminal event --
//! and both binaries now go through the exact same loop to get it.
//!
//! # Shutdown scope (explicitly out of scope here -- see the companion
//! lifecycle issue in this cluster, #833)
//!
//! [`MetalWorkerOwner`] retains the worker thread's `JoinHandle` so a future
//! graceful-shutdown PR has an obvious place to add a join/shutdown method,
//! but this module does not add one: neither `lattice.rs`'s prior
//! `MetalHandle` nor `lattice_serve.rs`'s prior bare
//! `mpsc::UnboundedSender<Job>` ever joined or explicitly shut down their
//! worker thread either (the process exits and the OS reaps the detached
//! thread), so today's behavior is preserved exactly rather than replaced
//! with a new, ad hoc shutdown state machine.
//!
//! # Testability without a GPU
//!
//! [`run_worker_loop`] -- the FIFO/cancellation/terminal-event state
//! machine -- is generic over an injected `generate` closure, exactly like
//! `lattice_serve.rs`'s pre-existing `run_worker_loop` was. [`MetalWorker::spawn`]
//! wires a REAL closure (calling `MetalQwen35State::generate_streaming_with_prefix_cache_and_cancel`)
//! into it for production; this module's own tests inject a fake generator
//! instead, so the state machine is fully covered without a Metal device.
//! `MetalWorker::spawn`'s `loader` failure path is also GPU-free: a loader
//! that returns `Err` before ever constructing a `MetalQwen35State`
//! typechecks and runs with no device involved. The real `spawn` -> real
//! `generate` success path has no equivalent GPU-free test (mirrors
//! precedent: PR #666's `MetalHandle` wiring shipped without a call-site
//! test requiring a real Q4 checkpoint fixture that doesn't exist, relying
//! on `metal_qwen35.rs`'s own exhaustive Device-gated tests for the
//! underlying `generate_streaming_with_prefix_cache_and_cancel` call).

use crate::forward::metal_qwen35::{ChatMessage, MetalQwen35State, format_chat_template};
use crate::kv_cache::CrossTurnSlotId;
use crate::model::qwen35_config::{GenerateConfig, GenerateOutput};
use crate::serve::ApiError;
use crate::tokenizer::Tokenizer as _;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::vision::VisionError;
use crate::vision::checkpoint::Qwen35VisionWeights;
use crate::vision::multimodal::Qwen35VisionRequest;
use crate::vision::qwen35_merger::qwen35_merger_forward;
use crate::vision::qwen35_vit::preprocess_qwen35_image;
use crate::vision::qwen35_vit_metal::qwen35_vit_forward_metal;
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, watch};

/// Default cap on outstanding (queued + in-flight) jobs a [`MetalWorkerClient`]
/// admits before rejecting new submissions (issue #932). Conservative on
/// purpose: this worker serializes ALL generation onto one dedicated thread
/// (see the module docs), so a queue depth in the hundreds/thousands under
/// bursty load just means O(N * request_size) memory growth (retained
/// messages, sampling config, and an open SSE/event channel per queued job)
/// with no matching throughput benefit — the extra jobs cannot run any
/// sooner. Both binaries expose this as an overridable `--max-pending` flag;
/// this constant is only the default when that flag is omitted.
pub const DEFAULT_MAX_PENDING_JOBS: usize = 32;

/// Selects the context-window formula enforced before Metal generation.
/// Each serve adapter supplies the policy matching its pre-worker contract.
#[derive(Debug, Clone, Copy)]
pub enum ContextWindowPolicy {
    /// Enforce `prompt_tokens + max_new_tokens <= model_max_context`.
    PromptAndMaxTokens,
    /// Enforce `prompt_tokens + max_new_tokens + reasoning_budget + 1
    /// <= model_max_context`.
    PromptAndDecodeWithDelimiter,
}

/// Everything a successful [`MetalWorker::spawn`] resolves to describe the
/// loaded model, beyond the client handle itself: the format string, the
/// actual KV context the loader allocated, and the adapter's window policy.
#[derive(Debug, Clone)]
pub struct WorkerMetadata {
    pub format: String,
    pub model_max_context: usize,
    pub context_window_policy: ContextWindowPolicy,
}

/// One token-stream event from the worker back to a request handler.
/// Replaces `lattice.rs`'s oneshot-reply `MetalJob` contract and
/// `lattice_serve.rs`'s private `Ev` enum with a single shared shape.
#[derive(Debug)]
pub enum WorkerEvent {
    /// One streamed token delta.
    Delta(String),
    /// Generation completed (naturally or via the engine's own internal
    /// `should_cancel` observation mid-decode -- that distinction lives in
    /// `GenerateOutput::stopped`/`stop_reason`, unchanged from both binaries'
    /// prior contract).
    Complete(GenerateOutput),
    /// The request itself cannot fit the model's KV window, caught before
    /// any generation work starts (#656). Carries a ready-to-return
    /// [`ApiError`] (`BadRequest`, code `context_length_exceeded`) instead
    /// of a raw string, so every caller maps it identically.
    Rejected(ApiError),
    /// Generation failed closed instead of completing for a reason other
    /// than a grammar-blocked mask -- an ordinary internal failure. Carries
    /// the underlying error message for server-side logging.
    Failed(String),
    /// Generation failed closed because a grammar mask blocked every
    /// candidate token (#611), distinct from [`WorkerEvent::Failed`] at the
    /// type level so a caller offering structured-output admission can
    /// report its dedicated `blocked_constraint` HTTP machine code without
    /// pattern-matching the message text (round-1 structured-output-v0
    /// review, medium finding 2: a backend wording change must not be able
    /// to silently degrade that code to `internal_error`). Carries the
    /// underlying error message for server-side logging only.
    ConstraintBlocked(String),
    /// The job was skipped before any prompt work started because the
    /// client was already gone: `cancel`'s watch flag was `true`, or this
    /// event receiver was already closed, at dequeue time. The single
    /// shared contract this refactor picks for that case (#832) -- neither
    /// binary's prior ad hoc behavior (an empty interrupted `GenerateOutput`
    /// reply vs. total silence) survives independently.
    Cancelled,
}

/// Failure classification internal to [`run_worker_loop`]'s injected
/// `generate` closure -- never exposed outside this module. Keeps the
/// `Rejected` vs. `Failed` distinction (#656 vs. #611) at the type level
/// instead of `lattice_serve.rs`'s prior string-prefix-sniffing convention
/// (`PROMPT_EXCEEDS_WINDOW_PREFIX`).
enum WorkerFailure {
    Rejected(ApiError),
    Failed(String),
    /// Mirrors [`WorkerEvent::ConstraintBlocked`] -- see that variant's doc
    /// comment. Kept distinct from `Failed` from the moment the generation
    /// call returns, all the way to the `WorkerEvent` sent back to the
    /// caller, so no stage in between has to sniff the message text.
    ConstraintBlocked(String),
}

impl From<crate::error::InferenceError> for WorkerFailure {
    /// Classifies a generation-time [`InferenceError`](crate::error::InferenceError)
    /// into the worker's own failure shape. `GrammarConstraintBlocked` is
    /// the one variant with a dedicated `WorkerEvent`; every other variant
    /// (including `InvalidInput`'s many unrelated uses) stays a generic
    /// `Failed` exactly as before this change.
    fn from(err: crate::error::InferenceError) -> Self {
        match err {
            crate::error::InferenceError::GrammarConstraintBlocked(message) => {
                WorkerFailure::ConstraintBlocked(message)
            }
            other => WorkerFailure::Failed(other.to_string()),
        }
    }
}

/// Worker startup failure: either the `loader` itself returned `Err`
/// (model/tokenizer load failure), the worker thread exited/panicked
/// before ever sending a readiness signal, or the requested admission cap
/// (issue #939) was outside `Semaphore::new`'s valid range.
#[derive(Debug)]
pub enum StartupError {
    Load(String),
    ThreadExited,
    /// `max_pending` was `0` (admits nothing -- every request would fail
    /// admission before any generation work could ever run) or greater
    /// than `Semaphore::MAX_PERMITS` (`Semaphore::new` panics outright on
    /// such a value). Caught here, before `Semaphore::new` is ever called,
    /// as an ordinary configuration error instead of a startup panic.
    InvalidMaxPending {
        max_pending: usize,
    },
}

impl std::fmt::Display for StartupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartupError::Load(message) => write!(f, "{message}"),
            StartupError::ThreadExited => {
                write!(f, "worker thread exited before loading finished")
            }
            StartupError::InvalidMaxPending { max_pending } => write!(
                f,
                "--max-pending must be between 1 and {} (got {max_pending})",
                Semaphore::MAX_PERMITS
            ),
        }
    }
}

impl std::error::Error for StartupError {}

/// One generation request handed to the worker thread.
///
/// `pub` (unconditionally) so its type can appear in the `test-utils`-gated
/// cross-binary test seam's public signatures below (a private type in a
/// public function's return position does not compile) -- its FIELDS stay
/// private always; only the `test-utils`-gated `impl` block further down
/// can construct or read one.
pub struct WorkerJob {
    messages: Vec<ChatMessage>,
    cfg: GenerateConfig,
    tx: mpsc::UnboundedSender<WorkerEvent>,
    cancel: watch::Receiver<bool>,
    /// Admission slot for this job (issue #932), held from
    /// [`MetalWorkerClient::submit`] until `run_worker_loop` finishes with
    /// this job (whatever the outcome — `Complete`, `Rejected`, `Failed`, or
    /// a dequeue-time `Cancelled`) and drops it, exactly once, via ordinary
    /// struct-field `Drop` — never released early, never released twice,
    /// and never forgotten on any of those paths because nothing in
    /// `run_worker_loop` ever moves it out of `job` or calls
    /// `mem::forget`/`mem::drop` on it directly. The leading underscore
    /// silences "field is never read" (this field's only job is to exist
    /// and be dropped) without needing `#[allow(dead_code)]`.
    _admission_permit: OwnedSemaphorePermit,
}

/// Owns the dedicated worker thread's `JoinHandle`. See the module docs'
/// "Shutdown scope" section: intentionally inert today beyond retaining the
/// handle -- issue #833 is where a join/shutdown method belongs.
#[derive(Debug)]
pub struct MetalWorkerOwner {
    #[allow(dead_code)]
    join_handle: Option<std::thread::JoinHandle<()>>,
}

/// Cheaply `Clone` (an `mpsc` sender) handle used to submit generation
/// requests to the worker thread. `Send + Sync` so it lives in a binary's
/// `AppState` the same way the CPU backend's `Arc<Qwen35Model>` does --
/// only the underlying `MetalQwen35State` inside the worker thread is
/// confined to that thread.
#[derive(Debug, Clone)]
pub struct MetalWorkerClient {
    jobs: mpsc::UnboundedSender<WorkerJob>,
    /// Bounded-admission cap (issue #932): `Semaphore::new(max_pending)`, one
    /// permit per outstanding job (queued + in-flight, i.e. from `submit`
    /// until `run_worker_loop` is done with it). `Arc`-shared with every
    /// clone of this client so the cap is process-wide, not per-clone.
    admission: Arc<Semaphore>,
}

impl MetalWorkerClient {
    /// Submit one generation request; the worker thread processes jobs
    /// strictly FIFO. Returns the event receiver on success -- if the
    /// worker thread is no longer running, the returned receiver closes
    /// with zero events (`recv()` resolves to `None` on the first poll).
    /// Callers must treat that the same as an explicit "worker unavailable"
    /// error, mirroring each binary's prior `jobs.send(..).is_err()` check.
    ///
    /// Returns `Err(ApiError::ServiceUnavailable)` -- the ONE way this
    /// method is allowed to fail outwardly -- when the outstanding-job cap
    /// (issue #932) is already full: `max_pending` jobs are currently
    /// either queued or in-flight on the shared worker thread. This check
    /// runs synchronously, before the job is enqueued at all, so a caller
    /// rejected here has done zero tokenization/model work and the worker
    /// thread never sees the request -- admission is a pure "should this
    /// job exist at all" gate, never a mid-stream failure. Every other
    /// `MetalWorkerClient::submit` failure mode (worker gone, context
    /// window overflow, generation error) still flows through the
    /// zero-events-on-`rx`/`WorkerEvent::Rejected`/`WorkerEvent::Failed`
    /// contract unchanged.
    pub fn submit(
        &self,
        messages: Vec<ChatMessage>,
        gen_cfg: GenerateConfig,
        cancel: watch::Receiver<bool>,
    ) -> Result<mpsc::UnboundedReceiver<WorkerEvent>, ApiError> {
        let permit = self.admission.clone().try_acquire_owned().map_err(|_| {
            ApiError::ServiceUnavailable {
                message: "too many outstanding requests; the inference worker's pending-job \
                          queue is full, retry shortly"
                    .to_string(),
            }
        })?;
        let (tx, rx) = mpsc::unbounded_channel();
        let job = WorkerJob {
            messages,
            cfg: gen_cfg,
            tx,
            cancel,
            _admission_permit: permit,
        };
        // On failure `job` (including `tx` and the admission permit) is
        // simply dropped here, closing `rx` with zero events and freeing
        // the slot immediately -- see the doc comment above.
        let _ = self.jobs.send(job);
        Ok(rx)
    }

    /// Live snapshot of the admission semaphore's free slots (issue #932's
    /// cap). `/metrics` (issue #583) computes queue depth / in-flight jobs
    /// as `max_pending - available_permits()`: a permit is held from
    /// `submit` until `run_worker_loop` is fully done with the job (see this
    /// type's own doc comment above), so this reflects real outstanding
    /// work rather than a separately-tracked counter that could drift from
    /// the actual admission state.
    pub fn available_permits(&self) -> usize {
        self.admission.available_permits()
    }
}

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
fn check_prompt_fits_window(
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
                .saturating_add(cfg.reasoning_budget.unwrap_or(0)),
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

/// Render `messages` into the physical token id stream a [`Qwen35VisionRequest`]
/// needs (ADR-069 S6): every turn is ChatML-rendered exactly like
/// [`format_chat_template`], except the turn at `image_message_index` splices
/// `vision_start_token_id`, `num_pads` copies of `image_token_id`, then
/// `vision_end_token_id` immediately after that turn's role header and ahead
/// of its own text -- the same "vision block precedes text" layout
/// `vision::pooled_embed::embed_image_from_bytes_f16` uses, and the layout
/// the committed `#989` HF differential golden encodes (see
/// `tests/fixtures/vision/README.md`). Tokenizes in exactly two calls (the
/// text before the spliced ids, then the text after) rather than one call
/// per ChatML segment, so only the one unavoidable seam next to the
/// numeric image-pad ids is exposed to cross-call BPE boundary effects --
/// every other turn boundary stays inside a single `tokenizer.tokenize`
/// call, identical to the plain-text path.
fn build_vision_prompt_ids(
    messages: &[ChatMessage],
    image_message_index: usize,
    tokenizer: &BpeTokenizer,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    image_token_id: u32,
    num_pads: usize,
) -> Vec<u32> {
    let tokenize = |text: &str| -> Vec<u32> {
        let out = tokenizer.tokenize(text);
        out.input_ids[..out.real_length].to_vec()
    };

    let render_turn = |out: &mut String, m: &ChatMessage| {
        out.push_str("<|im_start|>");
        out.push_str(m.role.as_str());
        out.push('\n');
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    };

    let mut before = String::new();
    for m in &messages[..image_message_index] {
        render_turn(&mut before, m);
    }
    before.push_str("<|im_start|>");
    before.push_str(messages[image_message_index].role.as_str());
    before.push('\n');

    let mut after = String::new();
    after.push_str(&messages[image_message_index].content);
    after.push_str("<|im_end|>\n");
    for m in &messages[image_message_index + 1..] {
        render_turn(&mut after, m);
    }
    after.push_str("<|im_start|>assistant\n");

    let mut ids = tokenize(&before);
    ids.push(vision_start_token_id);
    ids.extend(std::iter::repeat_n(image_token_id, num_pads));
    ids.push(vision_end_token_id);
    ids.extend(tokenize(&after));
    ids
}

/// Non-[`WorkerFailure`] outcome of [`build_vision_request`]: cancellation observed between its
/// expensive phases (decode/preprocess, ViT, merger) must not surface as an ordinary failure.
/// [`run_worker_loop`]'s shared cancellation contract classifies mid-flight cancellation (after
/// a job has already been dequeued and started) as an `Ok(GenerateOutput)` carrying
/// `stop_reason: Some(StopReason::Interrupt)` -- exactly how the pre-existing text-generation
/// path already reports a should-cancel observed mid-prefill/mid-decode -- never a
/// `WorkerEvent::Failed`. Keeping this distinct from [`WorkerFailure`] at the type level means
/// the call site cannot accidentally collapse the two the way the pre-fix code did (returning
/// `WorkerFailure::Failed("cancelled before vision generation started")`, which classified a
/// disconnected client the same as a genuine internal error).
enum VisionBuildStop {
    /// `should_cancel` observed `true` between two of `build_vision_request`'s phases; the
    /// caller must synthesize an interrupted [`GenerateOutput`] instead of propagating this.
    Cancelled,
    Failure(WorkerFailure),
}

/// Build a [`Qwen35VisionRequest`] for `messages` (exactly one of which
/// carries an image, at `image_message_index`) against `vision_weights`
/// (ADR-069 S6): preprocess -> Metal ViT forward (S3b, transparently
/// CPU-fallback per-GEMM when the Metal dispatch threshold isn't cleared --
/// see `qwen35_vit_metal`'s module docs) -> merger -> the expanded token-id
/// stream from [`build_vision_prompt_ids`]. Returns [`VisionBuildStop::Failure`] wrapping
/// [`WorkerFailure::Rejected`] for a caller-fixable problem (bad image bytes, oversized decoded
/// dimensions, misaligned dimensions, missing checkpoint vision metadata),
/// [`WorkerFailure::Failed`] for an internal forward-pass error, and
/// [`VisionBuildStop::Cancelled`] (issue found in PR #1021 review round 1) if `should_cancel`
/// observes the caller is gone at any point between decode/preprocess, ViT, and merger -- so a
/// disconnected client stops paying for vision work at the earliest checkpoint instead of only
/// after the full pass completes.
#[allow(clippy::too_many_arguments)]
fn build_vision_request(
    vision_weights: &Qwen35VisionWeights,
    cfg: &crate::model::qwen35_config::Qwen35Config,
    tokenizer: &BpeTokenizer,
    messages: &[ChatMessage],
    image_message_index: usize,
    image_bytes: &[u8],
    should_cancel: &mut dyn FnMut() -> bool,
) -> Result<Qwen35VisionRequest, VisionBuildStop> {
    let bad_request = |message: String| {
        VisionBuildStop::Failure(WorkerFailure::Rejected(ApiError::BadRequest {
            message,
            code: "invalid_image",
        }))
    };

    let vision_cfg = cfg.vision_config.as_ref().ok_or_else(|| {
        VisionBuildStop::Failure(WorkerFailure::Rejected(ApiError::BadRequest {
            message: "this checkpoint has no vision_config; image input is unsupported".into(),
            code: "vision_unsupported",
        }))
    })?;
    let image_token_id = cfg
        .image_token_id
        .ok_or_else(|| bad_request("checkpoint has no image_token_id".into()))?;
    let vision_start = cfg
        .vision_start_token_id
        .ok_or_else(|| bad_request("checkpoint has no vision_start_token_id".into()))?;
    let vision_end = cfg
        .vision_end_token_id
        .ok_or_else(|| bad_request("checkpoint has no vision_end_token_id".into()))?;

    // Checked before any decode/preprocess work: a client that disconnected between dequeue and
    // here must not pay for the (potentially large) image decode below (PR #1021 review round 1
    // major finding).
    if should_cancel() {
        return Err(VisionBuildStop::Cancelled);
    }
    let (pixel_values, grid) = preprocess_qwen35_image(image_bytes, vision_cfg).map_err(|e| {
        // The dimension guard (ADR-069 S6 review round 1 blocker) gets its own HTTP code,
        // distinct from ordinary image-rejection reasons, so a caller can tell "this image is
        // fundamentally too large to serve" apart from "this image/data URI was malformed".
        let code = if matches!(e, VisionError::DimensionsExceeded(_)) {
            "image_dimensions_exceeded"
        } else {
            "invalid_image"
        };
        VisionBuildStop::Failure(WorkerFailure::Rejected(ApiError::BadRequest {
            message: format!("image preprocessing failed: {e}"),
            code,
        }))
    })?;

    if should_cancel() {
        return Err(VisionBuildStop::Cancelled);
    }
    let pre_merger = qwen35_vit_forward_metal(vision_weights, vision_cfg, &pixel_values, grid)
        .map_err(|e| {
            VisionBuildStop::Failure(WorkerFailure::Failed(format!("ViT forward failed: {e}")))
        })?;

    if should_cancel() {
        return Err(VisionBuildStop::Cancelled);
    }
    let post_merger = qwen35_merger_forward(&vision_weights.merger, vision_cfg, &pre_merger)
        .map_err(|e| {
            VisionBuildStop::Failure(WorkerFailure::Failed(format!("merger forward failed: {e}")))
        })?;

    if should_cancel() {
        return Err(VisionBuildStop::Cancelled);
    }

    let merge_sq = vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size;
    if merge_sq == 0 || !grid.num_patches().is_multiple_of(merge_sq) {
        return Err(bad_request(format!(
            "image dimensions produce a patch grid {grid:?} not divisible by \
             spatial_merge_size^2 ({merge_sq})"
        )));
    }
    let num_pads = grid.num_patches() / merge_sq;

    let input_ids = build_vision_prompt_ids(
        messages,
        image_message_index,
        tokenizer,
        vision_start,
        vision_end,
        image_token_id,
        num_pads,
    );

    Ok(Qwen35VisionRequest {
        input_ids,
        image_grids: vec![grid],
        post_merger_rows: post_merger,
        image_token_id,
        spatial_merge_size: vision_cfg.spatial_merge_size,
        decoder_hidden_size: cfg.hidden_size,
    })
}

/// The `GenerateOutput` [`run_worker_loop`]'s shared cancellation contract requires for a job
/// that started (was dequeued) but whose client disconnected before any tokens were produced --
/// mirrors the identically-shaped literal the text-generation path returns from inside
/// `MetalQwen35State::generate_streaming_with_prefix_cache_and_cancel` when `should_cancel` is
/// observed `true` before prefill. `prompt_tokens: 0` here (unlike that text-path literal, which
/// already knows the tokenized prompt length at that point) because the vision path can be
/// interrupted before `build_vision_request` ever tokenizes anything; this is a best-effort
/// terminal event, not a usage-accounting one.
fn vision_cancelled_output() -> GenerateOutput {
    GenerateOutput {
        text: String::new(),
        token_ids: vec![],
        prompt_tokens: 0,
        generated_tokens: 0,
        stopped: false,
        stop_reason: Some(crate::stop_reason::StopReason::Interrupt),
        token_logprobs: vec![],
    }
}

/// Dequeue -> cancel-check -> generate -> reply, serialized on whatever
/// thread calls this (the dedicated Metal worker thread in production; a
/// plain `std::thread::spawn` in this module's own tests).
///
/// `generate` is injected so tests can swap in a fake, GPU-free generator
/// while exercising the exact same queue/cancellation logic production
/// uses (mirrors `lattice_serve.rs`'s pre-existing `run_worker_loop`
/// design, generalized so it is no longer specific to that one binary). It
/// must call `on_token` for each generated delta and stop as soon as
/// `on_token` returns `false`; it must also poll `should_cancel`
/// independently of `on_token` -- including during any phase that never
/// calls `on_token` at all (a prefill-like section) -- and stop as soon as
/// `should_cancel` returns `true`.
///
/// In order, every job gets: FIFO dequeue; a cancel check (`cancel`'s watch
/// flag, OR this job's event receiver already closed) BEFORE any prompt
/// work, sending exactly [`WorkerEvent::Cancelled`] and skipping to the
/// next job if it fires; otherwise a call to `generate`, and exactly one
/// terminal event (`Complete`, `Rejected`, or `Failed`) after zero or more
/// `Delta` events.
fn run_worker_loop(
    mut job_rx: mpsc::UnboundedReceiver<WorkerJob>,
    mut generate: impl FnMut(
        &[ChatMessage],
        &GenerateConfig,
        &mut dyn FnMut(&str, u32) -> bool,
        &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure>,
) {
    while let Some(job) = job_rx.blocking_recv() {
        // Dequeue-time cancel check, independent of token-callback return
        // values (#744/#606): a client that disconnected while this job was
        // still queued behind an earlier one -- or whose event receiver is
        // already gone for any other reason -- must not pay for prefill at
        // all. Exactly one terminal event either way.
        if *job.cancel.borrow() || job.tx.is_closed() {
            let _ = job.tx.send(WorkerEvent::Cancelled);
            continue;
        }

        let cb_tx = job.tx.clone();
        let cancel_for_token = job.cancel.clone();
        let mut on_token = move |delta: &str, _token_id: u32| {
            if *cancel_for_token.borrow() {
                return false;
            }
            // `send` also fails once the client hangs up; kept as a second,
            // independent check so a job whose cancellation notification is
            // somehow delayed still stops the instant its event receiver is
            // gone.
            cb_tx.send(WorkerEvent::Delta(delta.to_string())).is_ok()
        };

        // Separate from `on_token`: this is what reaches a generator's
        // prefill gap and any empty-delta decode iterations, neither of
        // which ever calls `on_token`.
        let cancel_for_predicate = job.cancel.clone();
        let tx_for_predicate = job.tx.clone();
        let mut should_cancel =
            move || *cancel_for_predicate.borrow() || tx_for_predicate.is_closed();

        match generate(&job.messages, &job.cfg, &mut on_token, &mut should_cancel) {
            Ok(output) => {
                let _ = job.tx.send(WorkerEvent::Complete(output));
            }
            Err(WorkerFailure::Rejected(api_err)) => {
                let _ = job.tx.send(WorkerEvent::Rejected(api_err));
            }
            Err(WorkerFailure::Failed(message)) => {
                eprintln!("[metal-worker] generation error: {message}");
                let _ = job.tx.send(WorkerEvent::Failed(message));
            }
            Err(WorkerFailure::ConstraintBlocked(message)) => {
                eprintln!("[metal-worker] generation error: {message}");
                let _ = job.tx.send(WorkerEvent::ConstraintBlocked(message));
            }
        }
    }
}

/// Namespace for [`MetalWorker::spawn`] -- a zero-sized marker type (never
/// constructed) so the shared worker's entry point reads as
/// `MetalWorker::spawn(..)` at every call site, matching the association
/// `lattice.rs`'s prior `MetalHandle::spawn` and `lattice_serve.rs`'s prior
/// `spawn_worker` free function both had with "the Metal worker".
pub struct MetalWorker;

impl MetalWorker {
    /// Spawn the dedicated thread that owns the `!Send` Metal state for the
    /// whole process lifetime. `loader` runs ON the worker thread itself --
    /// constructing `MetalQwen35State` there means the `!Send` state never
    /// crosses a thread boundary -- and its `Ok` metadata becomes both this
    /// call's return value and the actual KV context every job is checked
    /// against.
    ///
    /// Blocks the calling thread until `loader` finishes (successfully or
    /// not), mirroring both binaries' pre-existing "load, then bind, then
    /// listen" startup ordering (`lattice.rs`'s `MetalHandle::spawn`,
    /// `lattice_serve.rs`'s `spawn_worker` + its separate `ready` channel):
    /// a caller never binds its HTTP listener before the model is confirmed
    /// ready, and never gets a `MetalWorkerClient` it could submit jobs to
    /// before that point either.
    ///
    /// `max_pending` (issue #932) is the returned `MetalWorkerClient`'s
    /// outstanding-job admission cap -- see [`MetalWorkerClient::submit`].
    /// Both binaries pass their own `--max-pending`-derived value (default
    /// [`DEFAULT_MAX_PENDING_JOBS`]); this function applies no default of
    /// its own.
    pub fn spawn(
        loader: impl FnOnce() -> Result<
            (
                MetalQwen35State,
                BpeTokenizer,
                WorkerMetadata,
                Option<Qwen35VisionWeights>,
            ),
            String,
        > + Send
        + 'static,
        max_pending: usize,
    ) -> Result<(MetalWorkerOwner, MetalWorkerClient, WorkerMetadata), StartupError> {
        // #939: validate BEFORE `Semaphore::new`, which panics outright for
        // `max_pending > Semaphore::MAX_PERMITS` and would otherwise let
        // `max_pending == 0` silently build a worker that admits nothing.
        if max_pending == 0 || max_pending > Semaphore::MAX_PERMITS {
            return Err(StartupError::InvalidMaxPending { max_pending });
        }
        let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerJob>();
        let admission = Arc::new(Semaphore::new(max_pending));
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<WorkerMetadata, String>>();

        let join_handle = std::thread::spawn(move || match loader() {
            Ok((mut state, tokenizer, meta, vision_weights)) => {
                let _ = ready_tx.send(Ok(meta.clone()));
                run_worker_loop(job_rx, move |messages, cfg, on_token, should_cancel| {
                    // ADR-069 S6: a job carrying an image takes a separate
                    // path -- one Metal ViT+merger pass to build a
                    // `Qwen35VisionRequest`, then the vision-aware decode
                    // entry point -- instead of the plain ChatML render +
                    // text `generate_streaming_with_prefix_cache_and_cancel`
                    // below. `image_message_index` is `None` for every
                    // plain-text request, which is the overwhelming
                    // majority, so that request shape is completely
                    // unaffected by this branch.
                    let mut image_positions = messages
                        .iter()
                        .enumerate()
                        .filter(|(_, m)| m.image.is_some());
                    let image_message_index = image_positions.next().map(|(i, _)| i);
                    if image_positions.next().is_some() {
                        return Err(WorkerFailure::Rejected(ApiError::BadRequest {
                            message: "only a single image is supported per request".into(),
                            code: "vision_unsupported",
                        }));
                    }

                    if let Some(image_message_index) = image_message_index {
                        let Some(vision_weights) = vision_weights.as_ref() else {
                            return Err(WorkerFailure::Rejected(ApiError::BadRequest {
                                message: "image input requires a vision-capable model; this \
                                          checkpoint has no vision weights loaded"
                                    .into(),
                                code: "vision_unsupported",
                            }));
                        };
                        // Checked before entering `build_vision_request` at all (PR #1021
                        // review round 1 major finding): the prior code only checked
                        // `should_cancel` after the full decode/preprocess/ViT/merger pass
                        // had already run, so a disconnected client still paid for the
                        // entire vision prefill.
                        if should_cancel() {
                            return Ok(vision_cancelled_output());
                        }
                        let image_bytes = &messages[image_message_index]
                            .image
                            .as_ref()
                            .expect("image_message_index selected only messages with Some(image)")
                            .bytes;
                        let request = match build_vision_request(
                            vision_weights,
                            &state.engine.config,
                            &tokenizer,
                            messages,
                            image_message_index,
                            image_bytes,
                            &mut *should_cancel,
                        ) {
                            Ok(request) => request,
                            Err(VisionBuildStop::Cancelled) => {
                                return Ok(vision_cancelled_output());
                            }
                            Err(VisionBuildStop::Failure(failure)) => return Err(failure),
                        };
                        check_prompt_fits_window(
                            meta.context_window_policy,
                            meta.model_max_context,
                            request.input_ids.len(),
                            cfg,
                        )
                        .map_err(WorkerFailure::Rejected)?;
                        if should_cancel() {
                            return Ok(vision_cancelled_output());
                        }
                        // `generate_multimodal_vision` is a single blocking
                        // call (no incremental `on_token` hook exists on
                        // this entry point yet) -- the full answer arrives
                        // as one delta rather than a token-by-token stream
                        // (known ADR-069 S6 v0 limitation).
                        let output = state
                            .generate_multimodal_vision(&request, &tokenizer, cfg)
                            .map_err(WorkerFailure::from)?;
                        if !output.text.is_empty() {
                            on_token(&output.text, 0);
                        }
                        return Ok(output);
                    }

                    // Render the ChatML prompt exactly once (#828/#832: the
                    // prior `lattice_serve.rs` path rendered it a second
                    // time inside its own window preflight); reused for
                    // both the window check and the generation call below.
                    let prompt = format_chat_template(messages);
                    let prompt_len = tokenizer.tokenize(&prompt).real_length;
                    check_prompt_fits_window(
                        meta.context_window_policy,
                        meta.model_max_context,
                        prompt_len,
                        cfg,
                    )
                    .map_err(WorkerFailure::Rejected)?;

                    // Cache-aware + cancellation-aware call (#462/#744):
                    // reuses the previous turn's shared token prefix
                    // instead of a full re-prefill on every request, and
                    // observes client disconnect before prefill,
                    // immediately after prefill, and at the top of every
                    // decode iteration. This worker thread owns one
                    // `MetalQwen35State` for the whole process lifetime, so
                    // `CrossTurnSlotId::DEFAULT` is the only slot that
                    // exists; the planner re-verifies the retained prefix
                    // against this request's prompt on every call and
                    // falls back to `PrefixReuseMode::FullRefill` whenever
                    // they diverge, so correctness never depends on
                    // distinguishing clients.
                    let cached = state.generate_streaming_with_prefix_cache_and_cancel(
                        CrossTurnSlotId::DEFAULT,
                        &prompt,
                        &tokenizer,
                        cfg,
                        on_token,
                        should_cancel,
                    );
                    if let Ok(c) = &cached {
                        eprintln!(
                            "[metal-worker] cross-turn cache: mode={:?} reused={} \
                             prefetched={} prompt={}",
                            c.cache.mode,
                            c.cache.reused_tokens,
                            c.cache.prefetched_tokens,
                            c.cache.prompt_tokens,
                        );
                    }
                    cached.map(|c| c.output).map_err(WorkerFailure::from)
                });
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        });

        match ready_rx.recv() {
            Ok(Ok(meta)) => Ok((
                MetalWorkerOwner {
                    join_handle: Some(join_handle),
                },
                MetalWorkerClient {
                    jobs: job_tx,
                    admission,
                },
                meta,
            )),
            Ok(Err(e)) => Err(StartupError::Load(e)),
            Err(_) => Err(StartupError::ThreadExited),
        }
    }
}

// ─── test-only cross-binary seam (issue #832) ─────────────────────────────
//
// `lattice.rs` and `lattice_serve.rs` each carry their own router-level test
// suite that drives a fake worker through the real `chat_completions`
// handler and real `AppState`/job-queue plumbing. Before this module
// existed, each binary's own private `Job`/`run_worker_loop` was directly
// visible to its own `#[cfg(test)]` module (same crate, same compilation
// unit). Now that both binaries share this module instead, their tests are
// a *separate* compilation unit each (a bin target links against this
// library crate as an ordinary dependency and cannot see `#[cfg(test)]`-only
// internals) -- only a real Cargo feature crosses that boundary, matching
// this crate's pre-existing `test-utils` convention (see
// `lattice_inference::model::qwen35::test_support`'s own doc comment for the
// same reasoning spelled out in full).

#[cfg(any(test, feature = "test-utils"))]
impl WorkerJob {
    /// Reply to this job with one event, exactly as the production worker
    /// loop would via its own `job.tx.send(..)`. Returns `false` once the
    /// submitting caller's event receiver is gone. Test-only: production
    /// code always routes replies through [`run_worker_loop`], never
    /// directly.
    pub fn reply(&self, event: WorkerEvent) -> bool {
        self.tx.send(event).is_ok()
    }
}

/// A [`MetalWorkerClient`] wired to a plain, unattached job receiver, for
/// tests that want to fully control every reply by hand (a fake worker
/// task/thread, or none at all -- see [`WorkerJob::reply`]). Mirrors
/// `lattice_serve.rs`'s pre-existing `test_app_state_with_jobs` helper,
/// generalized so both binaries' test suites build on one shared seam
/// instead of each rolling its own raw `mpsc::unbounded_channel::<Job>()`
/// pair.
#[cfg(any(test, feature = "test-utils"))]
pub fn test_client_and_jobs() -> (MetalWorkerClient, mpsc::UnboundedReceiver<WorkerJob>) {
    // A large, effectively-unbounded cap: the overwhelming majority of
    // existing callers of this seam predate the #932 admission cap and
    // exercise request validation / routing / cancellation, not admission
    // itself -- they must keep behaving as if the queue were unbounded.
    // Tests that specifically exercise the cap use
    // `test_client_and_jobs_with_cap` instead.
    test_client_and_jobs_with_cap(TEST_EFFECTIVELY_UNBOUNDED_CAP)
}

/// Same as [`test_client_and_jobs`], with an explicit admission cap (issue
/// #932) instead of the effectively-unbounded default -- for tests that
/// exercise `MetalWorkerClient::submit`'s admission rejection itself.
#[cfg(any(test, feature = "test-utils"))]
pub fn test_client_and_jobs_with_cap(
    max_pending: usize,
) -> (MetalWorkerClient, mpsc::UnboundedReceiver<WorkerJob>) {
    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerJob>();
    (
        MetalWorkerClient {
            jobs: job_tx,
            admission: Arc::new(Semaphore::new(max_pending)),
        },
        job_rx,
    )
}

/// See [`test_client_and_jobs`]'s doc comment: the cap
/// `test_client_and_jobs`/`spawn_fake` (the two test-utils seams that predate
/// issue #932) use so pre-existing callers keep seeing effectively-unbounded
/// admission unless they opt into the `_with_cap` variant.
#[cfg(any(test, feature = "test-utils"))]
const TEST_EFFECTIVELY_UNBOUNDED_CAP: usize = 1_000_000;

/// A [`MetalWorkerClient`] backed by a REAL background thread running the
/// exact production FIFO/cancellation loop ([`run_worker_loop`]) and the
/// exact production [`check_prompt_fits_window`] invariant (real
/// chat-template render, real tokenizer) -- only the terminal "call into
/// Metal" step is replaced by `generate`, a caller-supplied fake. A mutation
/// to the shared window-check or FIFO loop is observed by whichever
/// binary's test drives this seam, not two independent per-binary copies of
/// the check (mirrors `lattice_serve.rs`'s pre-existing
/// `real_worker_state`/`baseline_fake_worker_state` test helpers,
/// generalized here so `lattice.rs`'s equivalent tests share it instead of
/// carrying a second, independently-written copy).
///
/// `generate` receives the already-tokenized `prompt_tokens` count (the same
/// value the real window-check computed) alongside `messages`/`cfg`, so a
/// caller can build a faithful `GenerateOutput`/observation without
/// re-deriving that count independently.
#[cfg(any(test, feature = "test-utils"))]
#[allow(clippy::type_complexity)]
pub fn spawn_fake(
    context_window_policy: ContextWindowPolicy,
    model_max_context: usize,
    tokenizer: BpeTokenizer,
    generate: impl FnMut(
        &[ChatMessage],
        &GenerateConfig,
        usize,
        &mut dyn FnMut(&str, u32) -> bool,
        &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, String>
    + Send
    + 'static,
) -> MetalWorkerClient {
    // See `test_client_and_jobs`'s doc comment: effectively-unbounded so
    // this seam's many pre-#932 callers (request validation / routing /
    // cancellation fixtures, not admission itself) keep behaving as before.
    spawn_fake_with_cap(
        TEST_EFFECTIVELY_UNBOUNDED_CAP,
        context_window_policy,
        model_max_context,
        tokenizer,
        generate,
    )
}

/// Same as [`spawn_fake`], with an explicit admission cap (issue #932)
/// instead of the effectively-unbounded default -- for tests that exercise
/// `MetalWorkerClient::submit`'s admission rejection at the real-router
/// (HTTP) layer.
#[cfg(any(test, feature = "test-utils"))]
#[allow(clippy::type_complexity)]
pub fn spawn_fake_with_cap(
    max_pending: usize,
    context_window_policy: ContextWindowPolicy,
    model_max_context: usize,
    tokenizer: BpeTokenizer,
    mut generate: impl FnMut(
        &[ChatMessage],
        &GenerateConfig,
        usize,
        &mut dyn FnMut(&str, u32) -> bool,
        &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, String>
    + Send
    + 'static,
) -> MetalWorkerClient {
    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerJob>();
    std::thread::spawn(move || {
        run_worker_loop(job_rx, move |messages, cfg, on_token, should_cancel| {
            let prompt = format_chat_template(messages);
            let prompt_tokens = tokenizer.tokenize(&prompt).real_length;
            check_prompt_fits_window(context_window_policy, model_max_context, prompt_tokens, cfg)
                .map_err(WorkerFailure::Rejected)?;
            generate(messages, cfg, prompt_tokens, on_token, should_cancel)
                .map_err(WorkerFailure::Failed)
        });
    });
    MetalWorkerClient {
        jobs: job_tx,
        admission: Arc::new(Semaphore::new(max_pending)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    // ── GPU-free fakes, ported from lattice_serve.rs's pre-existing
    //    `run_worker_loop` test suite (#832 migrates them here) ──────────

    #[allow(clippy::type_complexity)]
    fn fake_generate(
        cap: usize,
        started: Arc<AtomicUsize>,
        ran_tokens: Arc<AtomicUsize>,
    ) -> impl FnMut(
        &[ChatMessage],
        &GenerateConfig,
        &mut dyn FnMut(&str, u32) -> bool,
        &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        move |_messages, _cfg, on_token, should_cancel| {
            started.fetch_add(1, Ordering::SeqCst);
            let mut n = 0usize;
            for i in 0..cap {
                std::thread::sleep(Duration::from_millis(5));
                if should_cancel() {
                    break;
                }
                if !on_token("x", i as u32) {
                    break;
                }
                n += 1;
                ran_tokens.fetch_add(1, Ordering::SeqCst);
            }
            Ok(GenerateOutput {
                text: "x".repeat(n),
                token_ids: vec![0; n],
                prompt_tokens: 1,
                generated_tokens: n,
                stopped: false,
                stop_reason: None,
                token_logprobs: vec![],
            })
        }
    }

    #[allow(clippy::type_complexity)]
    fn fake_generate_with_prefill_gap(
        prefill_steps: usize,
        decode_cap: usize,
        entered_decode: Arc<AtomicBool>,
    ) -> impl FnMut(
        &[ChatMessage],
        &GenerateConfig,
        &mut dyn FnMut(&str, u32) -> bool,
        &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        move |_messages, _cfg, on_token, should_cancel| {
            for _ in 0..prefill_steps {
                std::thread::sleep(Duration::from_millis(5));
                if should_cancel() {
                    return Ok(GenerateOutput {
                        text: String::new(),
                        token_ids: vec![],
                        prompt_tokens: 1,
                        generated_tokens: 0,
                        stopped: false,
                        stop_reason: None,
                        token_logprobs: vec![],
                    });
                }
            }
            entered_decode.store(true, Ordering::SeqCst);
            let mut n = 0usize;
            for i in 0..decode_cap {
                std::thread::sleep(Duration::from_millis(5));
                if should_cancel() {
                    break;
                }
                if !on_token("x", i as u32) {
                    break;
                }
                n += 1;
            }
            Ok(GenerateOutput {
                text: "x".repeat(n),
                token_ids: vec![0; n],
                prompt_tokens: 1,
                generated_tokens: n,
                stopped: false,
                stop_reason: None,
                token_logprobs: vec![],
            })
        }
    }

    #[allow(clippy::type_complexity)]
    fn fake_generate_fails_once_then_succeeds(
        message: &'static str,
        call_count: Arc<AtomicUsize>,
    ) -> impl FnMut(
        &[ChatMessage],
        &GenerateConfig,
        &mut dyn FnMut(&str, u32) -> bool,
        &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        move |_messages, _cfg, on_token, _should_cancel| {
            if call_count.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(WorkerFailure::Failed(message.to_string()));
            }
            let _ = on_token("x", 0);
            Ok(GenerateOutput {
                text: "x".to_string(),
                token_ids: vec![0],
                prompt_tokens: 1,
                generated_tokens: 1,
                stopped: true,
                stop_reason: None,
                token_logprobs: vec![],
            })
        }
    }

    // ── PR #1021 review round 1 major finding: `build_vision_request`
    //    cancellation ordering ─────────────────────────────────────────────

    fn tiny_vision_cfg() -> crate::model::qwen35_config::VisionModelConfig {
        crate::model::qwen35_config::VisionModelConfig {
            depth: 1,
            hidden_size: 8,
            num_heads: 2,
            patch_size: 2,
            spatial_merge_size: 2,
            out_hidden_size: 8,
            temporal_patch_size: 1,
            num_position_embeddings: 16,
            in_channels: 1,
            deepstack_visual_indexes: vec![],
        }
    }

    /// Deliberately empty/zero-length weight vectors: both cancellation tests below must never
    /// touch these (the whole point is that `should_cancel` short-circuits before the ViT/merger
    /// forward passes that would read them), so their contents don't matter.
    fn empty_vision_weights() -> Qwen35VisionWeights {
        use crate::vision::checkpoint::VisualMergerWeights;
        Qwen35VisionWeights {
            patch_embed_weight: vec![],
            patch_embed_weight_shape: vec![],
            patch_embed_bias: vec![],
            pos_embed: vec![],
            blocks: vec![],
            merger: VisualMergerWeights {
                fc1_weight: vec![],
                fc1_bias: vec![],
                fc2_weight: vec![],
                fc2_bias: vec![],
                norm_weight: vec![],
                norm_bias: vec![],
            },
        }
    }

    fn vision_test_qwen35_config() -> crate::model::qwen35_config::Qwen35Config {
        crate::model::qwen35_config::Qwen35Config {
            vision_config: Some(tiny_vision_cfg()),
            image_token_id: Some(100),
            vision_start_token_id: Some(101),
            vision_end_token_id: Some(102),
            hidden_size: 8,
            ..Default::default()
        }
    }

    fn minimal_tokenizer() -> BpeTokenizer {
        // The constructor requires a dense id range, so a single-entry vocab is
        // the smallest valid tokenizer. These tests cancel before any
        // tokenization happens; the tokenizer is never actually exercised.
        let vocab = std::collections::HashMap::from([("a".to_string(), 0u32)]);
        BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
            .expect("single-entry vocab must construct a tokenizer")
    }

    fn make_black_test_png(w: u32, h: u32) -> Vec<u8> {
        use image::RgbImage;
        let img = RgbImage::new(w, h);
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    /// PR #1021 review round 1 major finding: the pre-fix code only checked `should_cancel`
    /// *after* `build_vision_request` had already run the full decode/preprocess/ViT/merger
    /// pass, so a disconnected client still paid for the entire vision prefill. Proves the fix:
    /// with `should_cancel` already `true`, `build_vision_request` must stop before touching the
    /// (deliberately invalid) image bytes at all -- if it decoded first, this would fail with an
    /// image-decode error instead of `Cancelled`, and `should_cancel` would be observed more than
    /// once.
    ///
    /// Mutation-sensitive: commenting out the first `if should_cancel() { return
    /// Err(VisionBuildStop::Cancelled); }` in `build_vision_request` turns this from an
    /// instant `Cancelled` into an `invalid_image` rejection from the garbage bytes reaching
    /// `preprocess_qwen35_image` -- verified locally by reverting the check and watching this
    /// assertion fail, then restoring it (not committed).
    #[test]
    fn build_vision_request_stops_before_any_work_when_already_cancelled() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();
        let mut should_cancel = move || {
            calls2.fetch_add(1, Ordering::SeqCst);
            true
        };

        let vision_weights = empty_vision_weights();
        let cfg = vision_test_qwen35_config();
        let tokenizer = minimal_tokenizer();
        let messages = vec![ChatMessage::user("hi")];
        let garbage_bytes = b"not an image";

        let result = build_vision_request(
            &vision_weights,
            &cfg,
            &tokenizer,
            &messages,
            0,
            garbage_bytes,
            &mut should_cancel,
        );

        assert!(
            matches!(result, Err(VisionBuildStop::Cancelled)),
            "expected Cancelled before any decode/preprocess work"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "should_cancel must be checked exactly once, before any other work"
        );
    }

    /// Defense-in-depth checkpoint: cancellation observed right after decode/preprocess
    /// completes must stop `build_vision_request` before it ever calls the Metal-only
    /// `qwen35_vit_forward_metal` (unreachable in this non-GPU gate) -- proven by the exact
    /// `should_cancel` call count (one before preprocess, one immediately after) rather than by
    /// exercising the GPU path itself.
    #[test]
    fn build_vision_request_stops_after_preprocess_before_vit_when_cancelled_mid_flight() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls2 = calls.clone();
        let mut should_cancel = move || calls2.fetch_add(1, Ordering::SeqCst) >= 1;

        let vision_weights = empty_vision_weights();
        let cfg = vision_test_qwen35_config();
        let tokenizer = minimal_tokenizer();
        let messages = vec![ChatMessage::user("hi")];
        let png = make_black_test_png(8, 8); // 8x8, aligned to tiny_vision_cfg's factor=4

        let result = build_vision_request(
            &vision_weights,
            &cfg,
            &tokenizer,
            &messages,
            0,
            &png,
            &mut should_cancel,
        );

        assert!(
            matches!(result, Err(VisionBuildStop::Cancelled)),
            "expected Cancelled between preprocess and the ViT forward pass"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "should_cancel must be checked once before preprocess and once after -- proving the \
             Metal ViT forward was never entered"
        );
    }

    /// Builds a `WorkerJob` plus the receiver its worker replies on and the
    /// guard that cancels it when dropped (the same guard a real handler
    /// moves into the SSE stream / keeps local for non-streaming, standing
    /// in here for "the client is still connected").
    fn make_job() -> (
        WorkerJob,
        mpsc::UnboundedReceiver<WorkerEvent>,
        crate::serve::CancelOnDrop,
    ) {
        let (tx, rx) = mpsc::unbounded_channel::<WorkerEvent>();
        let (cancel_guard, cancel_rx) = crate::serve::cancel_pair();
        // These FIFO/cancellation-loop tests drive `WorkerJob` directly
        // (bypassing `MetalWorkerClient::submit`'s admission check
        // entirely), so each job gets its own throwaway one-permit
        // semaphore rather than sharing a real admission cap -- these tests
        // are not exercising #932's admission behavior at all.
        let permit = Arc::new(Semaphore::new(1))
            .try_acquire_owned()
            .expect("fresh single-permit semaphore must have a permit available");
        let job = WorkerJob {
            messages: vec![ChatMessage::user("hi")],
            cfg: GenerateConfig::default(),
            tx,
            cancel: cancel_rx,
            _admission_permit: permit,
        };
        (job, rx, cancel_guard)
    }

    #[test]
    fn queued_job_cancelled_before_dequeue_sends_exactly_one_cancelled_event() {
        let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerJob>();
        let started = Arc::new(AtomicUsize::new(0));
        let ran_tokens = Arc::new(AtomicUsize::new(0));

        // Job 1 occupies the worker (50 fake tokens, 5ms apart = ~250ms)
        // long enough that job 2 is still sitting in the queue, untouched,
        // when we cancel it a few lines down.
        let (job1, rx1, _guard1) = make_job();
        job_tx.send(job1).unwrap();

        // Job 2: cancelled client-side (guard dropped) immediately, while
        // it is still queued behind job 1.
        let (job2, mut rx2, guard2) = make_job();
        job_tx.send(job2).unwrap();
        drop(guard2);

        // Job 3: submitted after the cancelled one, to prove the worker
        // moves on and keeps serving correctly afterward.
        let (job3, rx3, _guard3) = make_job();
        job_tx.send(job3).unwrap();
        drop(job_tx);

        let started2 = started.clone();
        let ran2 = ran_tokens.clone();
        let handle =
            std::thread::spawn(move || run_worker_loop(job_rx, fake_generate(50, started2, ran2)));

        let completion_tokens_of = |mut rx: mpsc::UnboundedReceiver<WorkerEvent>| -> Option<usize> {
            let mut ct = None;
            while let Some(ev) = rx.blocking_recv() {
                if let WorkerEvent::Complete(output) = ev {
                    ct = Some(output.generated_tokens);
                }
            }
            ct
        };

        assert_eq!(
            completion_tokens_of(rx1),
            Some(50),
            "job 1 should run to completion undisturbed"
        );

        // Job 2 must produce exactly one event: Cancelled -- the single
        // shared contract (#832) this refactor picks, replacing both
        // binaries' prior divergent behavior (an empty interrupted
        // GenerateOutput reply vs. total silence).
        match rx2.blocking_recv() {
            Some(WorkerEvent::Cancelled) => {}
            other => panic!("expected exactly one Cancelled event, got {other:?}"),
        }
        assert!(
            rx2.blocking_recv().is_none(),
            "cancelled queued job must produce no further events after Cancelled"
        );

        assert_eq!(
            completion_tokens_of(rx3),
            Some(50),
            "worker must survive cancelling job 2 and serve job 3 normally afterward"
        );

        handle.join().expect("worker thread must not panic");

        assert_eq!(
            started.load(Ordering::SeqCst),
            2,
            "generate() must run exactly twice (job 1, job 3) -- never for cancelled job 2"
        );
        assert_eq!(
            ran_tokens.load(Ordering::SeqCst),
            100,
            "50 real fake-tokens each for job 1 and job 3, zero for cancelled job 2"
        );
    }

    #[test]
    fn job_whose_event_receiver_is_already_closed_is_cancelled_without_running_generate() {
        // Distinct from a `cancel`-guard drop: this job's `cancel` watch
        // stays `false` forever (the guard is kept alive), but its event
        // receiver is dropped before the worker ever dequeues it. The
        // dequeue-time check must catch this independently (#832: "cancel
        // OR event_receiver_closed").
        let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerJob>();
        let (tx, rx) = mpsc::unbounded_channel::<WorkerEvent>();
        drop(rx);
        let (_guard, cancel_rx) = crate::serve::cancel_pair();
        let permit = Arc::new(Semaphore::new(1))
            .try_acquire_owned()
            .expect("fresh single-permit semaphore must have a permit available");
        let job = WorkerJob {
            messages: vec![ChatMessage::user("hi")],
            cfg: GenerateConfig::default(),
            tx,
            cancel: cancel_rx,
            _admission_permit: permit,
        };
        job_tx.send(job).unwrap();
        drop(job_tx);

        let started = Arc::new(AtomicUsize::new(0));
        let ran_tokens = Arc::new(AtomicUsize::new(0));
        let started2 = started.clone();
        let ran2 = ran_tokens.clone();
        let handle =
            std::thread::spawn(move || run_worker_loop(job_rx, fake_generate(50, started2, ran2)));
        handle.join().expect("worker thread must not panic");

        assert_eq!(
            started.load(Ordering::SeqCst),
            0,
            "generate() must never run for a job whose event receiver was already closed"
        );
        assert_eq!(ran_tokens.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn running_job_cancelled_midstream_stops_early_and_worker_survives() {
        let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerJob>();
        let started = Arc::new(AtomicUsize::new(0));
        let ran_tokens = Arc::new(AtomicUsize::new(0));

        let (job1, mut rx1, guard1) = make_job();
        job_tx.send(job1).unwrap();
        let mut guard1 = Some(guard1);

        let (job2, mut rx2, _guard2) = make_job();
        job_tx.send(job2).unwrap();
        drop(job_tx);

        let started2 = started.clone();
        let ran2 = ran_tokens.clone();
        let handle = std::thread::spawn(move || {
            run_worker_loop(job_rx, fake_generate(2000, started2, ran2))
        });

        let mut seen = 0;
        loop {
            match rx1.blocking_recv() {
                Some(WorkerEvent::Delta(_)) => {
                    seen += 1;
                    if seen == 5 {
                        guard1.take();
                    }
                }
                Some(WorkerEvent::Complete(output)) => {
                    assert!(
                        output.generated_tokens < 2000,
                        "job 1 must stop well short of its 2000-token cap after \
                         cancellation, got {}",
                        output.generated_tokens
                    );
                    assert!(
                        output.generated_tokens < 100,
                        "job 1 must stop within a handful of tokens of the client \
                         disconnecting, not run on regardless; got {}",
                        output.generated_tokens
                    );
                    break;
                }
                Some(WorkerEvent::Failed(message)) => {
                    panic!("fake_generate never fails; unexpected Failed: {message}")
                }
                Some(WorkerEvent::ConstraintBlocked(message)) => {
                    panic!(
                        "fake_generate never blocks on a grammar constraint; unexpected \
                         ConstraintBlocked: {message}"
                    )
                }
                Some(WorkerEvent::Rejected(err)) => {
                    panic!("fake_generate never rejects; unexpected Rejected: {err:?}")
                }
                Some(WorkerEvent::Cancelled) => {
                    panic!("job 1 was already running -- Cancelled is a dequeue-only event")
                }
                None => panic!("job 1's reply channel closed before a Complete event"),
            }
        }

        let mut n2 = None;
        while let Some(ev) = rx2.blocking_recv() {
            if let WorkerEvent::Complete(output) = ev {
                n2 = Some(output.generated_tokens);
            }
        }
        assert_eq!(
            n2,
            Some(2000),
            "worker must survive mid-stream cancellation and serve the next job to completion"
        );

        handle.join().expect("worker thread must not panic");
    }

    #[test]
    fn running_job_cancelled_during_prefill_like_phase_never_calls_on_token() {
        let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerJob>();
        let entered_decode = Arc::new(AtomicBool::new(false));

        let (job1, mut rx1, guard1) = make_job();
        job_tx.send(job1).unwrap();
        job_tx.send(make_job().0).unwrap_or(()); // keep queue non-trivial; unused receiver dropped
        drop(job_tx);

        let entered2 = entered_decode.clone();
        let handle = std::thread::spawn(move || {
            run_worker_loop(job_rx, fake_generate_with_prefill_gap(400, 50, entered2))
        });

        std::thread::sleep(Duration::from_millis(20));
        drop(guard1);

        match rx1.blocking_recv() {
            Some(WorkerEvent::Delta(_)) => panic!(
                "on_token must never be called: cancellation happened while the fake \
                 generator was still in its prefill-like phase, which does not call \
                 on_token at all"
            ),
            Some(WorkerEvent::Complete(output)) => {
                assert_eq!(
                    output.generated_tokens, 0,
                    "job cancelled during the prefill-like phase must produce zero tokens, \
                     got {}",
                    output.generated_tokens
                );
            }
            Some(WorkerEvent::Failed(message)) => {
                panic!("fake_generate_with_prefill_gap never fails; unexpected Failed: {message}")
            }
            Some(WorkerEvent::ConstraintBlocked(message)) => {
                panic!(
                    "fake_generate_with_prefill_gap never blocks on a grammar constraint; \
                     unexpected ConstraintBlocked: {message}"
                )
            }
            Some(WorkerEvent::Rejected(err)) => {
                panic!("fake_generate_with_prefill_gap never rejects; unexpected Rejected: {err:?}")
            }
            Some(WorkerEvent::Cancelled) => {
                panic!("job 1 was already dequeued and running -- not a dequeue-time cancel")
            }
            None => panic!("job 1's reply channel closed before a Complete event"),
        }

        handle.join().expect("worker thread must not panic");

        assert!(
            !entered_decode.load(Ordering::SeqCst),
            "should_cancel alone (on_token is never called during this phase) must stop \
             the job before the decode phase is ever reached"
        );
    }

    #[test]
    fn generation_failure_is_reported_as_failed_not_complete() {
        let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerJob>();

        let (job1, mut rx1, _guard1) = make_job();
        job_tx.send(job1).unwrap();
        let (job2, mut rx2, _guard2) = make_job();
        job_tx.send(job2).unwrap();
        drop(job_tx);

        let call_count = Arc::new(AtomicUsize::new(0));
        let handle = std::thread::spawn({
            let call_count = call_count.clone();
            move || {
                run_worker_loop(
                    job_rx,
                    fake_generate_fails_once_then_succeeds(
                        "grammar constraint blocked every token; no legal continuation \
                         exists in the current grammar state",
                        call_count,
                    ),
                )
            }
        });

        match rx1.blocking_recv() {
            Some(WorkerEvent::Failed(message)) => {
                assert!(
                    message.contains("grammar constraint blocked every token"),
                    "Failed must carry the underlying error message, got: {message}"
                );
            }
            Some(WorkerEvent::Complete(_)) => panic!(
                "a failed generation must never be reported as Complete -- that would \
                 silently hand the HTTP layer a fabricated result for a request that \
                 produced no legal output"
            ),
            other => panic!("expected Failed as the first and only event, got {other:?}"),
        }

        let mut done = None;
        while let Some(ev) = rx2.blocking_recv() {
            if let WorkerEvent::Complete(output) = ev {
                done = Some(output.generated_tokens);
            }
        }
        assert_eq!(
            done,
            Some(1),
            "worker thread must survive a failed generation and serve the next job \
             normally afterward"
        );

        handle
            .join()
            .expect("worker thread must not panic on a generation error");
    }

    #[test]
    fn queue_closure_lets_the_worker_thread_exit_and_join() {
        let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerJob>();
        let started = Arc::new(AtomicUsize::new(0));
        let ran_tokens = Arc::new(AtomicUsize::new(0));
        let started2 = started.clone();
        let ran2 = ran_tokens.clone();
        let handle =
            std::thread::spawn(move || run_worker_loop(job_rx, fake_generate(1, started2, ran2)));
        // No jobs submitted at all: dropping every sender must let
        // `job_rx.blocking_recv()` return `None` immediately and the loop
        // (and thread) exit cleanly.
        drop(job_tx);
        handle
            .join()
            .expect("worker thread must exit and be joinable once every job sender drops");
        assert_eq!(started.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn owner_shutdown_joins_cleanly_once_the_queue_closes() {
        // "Owner shutdown" without going through the real `MetalWorker::spawn`
        // (which needs a real Metal device to reach `Ok`): builds a
        // `MetalWorkerOwner` directly around a `run_worker_loop` thread, the
        // same shape `MetalWorker::spawn` constructs internally.
        let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerJob>();
        let started = Arc::new(AtomicUsize::new(0));
        let ran_tokens = Arc::new(AtomicUsize::new(0));
        let started2 = started.clone();
        let ran2 = ran_tokens.clone();
        let join_handle =
            std::thread::spawn(move || run_worker_loop(job_rx, fake_generate(1, started2, ran2)));
        let mut owner = MetalWorkerOwner {
            join_handle: Some(join_handle),
        };
        drop(job_tx);
        let handle = owner
            .join_handle
            .take()
            .expect("owner must retain the join handle (issue #833's seam)");
        handle
            .join()
            .expect("owner's worker thread must join cleanly once the queue closes");
    }

    #[test]
    fn loader_failure_before_readiness_is_reported_without_touching_a_device() {
        // The `Ok` arm's type (`MetalQwen35State`) is never constructed --
        // this typechecks and runs with zero GPU involvement.
        let result = MetalWorker::spawn(
            || Err("simulated load failure".to_string()),
            DEFAULT_MAX_PENDING_JOBS,
        );
        match result {
            Err(StartupError::Load(message)) => {
                assert_eq!(message, "simulated load failure");
            }
            other => panic!("expected StartupError::Load, got {other:?}"),
        }
    }

    #[test]
    fn startup_error_display_matches_each_variant() {
        assert_eq!(StartupError::Load("boom".to_string()).to_string(), "boom");
        assert_eq!(
            StartupError::ThreadExited.to_string(),
            "worker thread exited before loading finished"
        );
        assert_eq!(
            StartupError::InvalidMaxPending { max_pending: 0 }.to_string(),
            format!(
                "--max-pending must be between 1 and {} (got 0)",
                Semaphore::MAX_PERMITS
            )
        );
    }

    // ── #939 max_pending boundary tests ───────────────────────────────────
    //
    // Validated BEFORE `Semaphore::new` in `MetalWorker::spawn`, so -- like
    // `loader_failure_before_readiness_is_reported_without_touching_a_device`
    // above -- these never construct a real `MetalQwen35State` and need no
    // GPU: an out-of-range `max_pending` returns `Err` before `loader` would
    // even be called (a loader that panics if invoked proves that).

    #[test]
    fn max_pending_zero_is_rejected_before_semaphore_new() {
        let result = MetalWorker::spawn(
            || -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata, Option<Qwen35VisionWeights>), String> {
                panic!("loader must not run: max_pending=0 must be rejected first")
            },
            0,
        );
        match result {
            Err(StartupError::InvalidMaxPending { max_pending: 0 }) => {}
            other => panic!("expected InvalidMaxPending{{max_pending: 0}}, got {other:?}"),
        }
    }

    #[test]
    fn max_pending_above_max_permits_is_rejected_before_semaphore_new() {
        let too_big = Semaphore::MAX_PERMITS + 1;
        let result = MetalWorker::spawn(
            || -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata, Option<Qwen35VisionWeights>), String> {
                panic!("loader must not run: max_pending above MAX_PERMITS must be rejected first")
            },
            too_big,
        );
        match result {
            Err(StartupError::InvalidMaxPending { max_pending }) => {
                assert_eq!(max_pending, too_big);
            }
            other => panic!("expected InvalidMaxPending, got {other:?}"),
        }
    }

    // ── check_prompt_fits_window, ported from lattice_serve.rs's
    //    pre-existing `check_prompt_fits_window` test suite ───────────────

    fn cfg_with(max_new_tokens: usize, reasoning_budget: Option<usize>) -> GenerateConfig {
        GenerateConfig {
            max_new_tokens,
            reasoning_budget,
            ..Default::default()
        }
    }

    #[test]
    fn check_prompt_fits_window_rejects_when_prompt_plus_decode_overflows() {
        // model_max_context=8, prompt_len=2, max_new_tokens=7, reasoning_budget=None:
        // 2 (prompt) + 7 (decode) + 1 (delimiter) = 10 > 8 -- must reject.
        let cfg = cfg_with(7, None);
        let err = check_prompt_fits_window(
            ContextWindowPolicy::PromptAndDecodeWithDelimiter,
            8,
            2,
            &cfg,
        )
        .unwrap_err();
        match err {
            ApiError::BadRequest { message, code } => {
                assert_eq!(code, "context_length_exceeded");
                assert!(
                    message.contains("2 tokens") && message.contains("8-token"),
                    "error must name the actual prompt length and window: {message}"
                );
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn lattice_context_boundary_accepts_exact_window_and_rejects_one_past() {
        let cfg = cfg_with(7, None);
        assert!(
            check_prompt_fits_window(ContextWindowPolicy::PromptAndMaxTokens, 8, 1, &cfg).is_ok()
        );
        assert!(
            check_prompt_fits_window(ContextWindowPolicy::PromptAndMaxTokens, 8, 2, &cfg).is_err()
        );
    }

    /// `lattice.rs`'s original `check_context_window` rejects a zero-token
    /// prompt independent of the window arithmetic; the policy that
    /// reproduces that predicate must too, even when `max_new_tokens`
    /// alone fits the window. The delimiter policy never had that
    /// conjunct and must keep accepting a zero-length prompt that fits.
    #[test]
    fn lattice_policy_rejects_zero_token_prompt_even_when_window_fits() {
        let cfg = cfg_with(7, None);
        let err = check_prompt_fits_window(ContextWindowPolicy::PromptAndMaxTokens, 8, 0, &cfg)
            .unwrap_err();
        match err {
            ApiError::BadRequest { message, code } => {
                assert_eq!(code, "context_length_exceeded");
                assert!(
                    message.contains("0 tokens"),
                    "error must name the zero-length prompt: {message}"
                );
            }
            other => panic!("expected BadRequest, got {other:?}"),
        }

        assert!(
            check_prompt_fits_window(
                ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                9,
                0,
                &cfg_with(7, None),
            )
            .is_ok()
        );
    }

    #[test]
    fn lattice_serve_context_boundary_accepts_exact_window_and_rejects_one_past() {
        let at_boundary = cfg_with(5, Some(1));
        assert!(
            check_prompt_fits_window(
                ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                8,
                1,
                &at_boundary,
            )
            .is_ok()
        );

        let one_past = cfg_with(6, Some(1));
        assert!(
            check_prompt_fits_window(
                ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                8,
                1,
                &one_past,
            )
            .is_err()
        );
    }

    #[test]
    fn check_prompt_fits_window_accepts_ordinary_prompt_unclamped() {
        let cfg = cfg_with(50, None);
        assert!(
            check_prompt_fits_window(
                ContextWindowPolicy::PromptAndDecodeWithDelimiter,
                4096,
                100,
                &cfg,
            )
            .is_ok()
        );
    }

    // ── admission cap / backpressure (issue #932) ─────────────────────────

    /// Cap enforcement: with `max_pending=2`, job 1 (dequeued immediately,
    /// running) plus job 2 (queued behind it) fill the cap; a 3rd submission
    /// must be rejected with `ApiError::ServiceUnavailable` before it ever
    /// reaches the job channel.
    ///
    /// Mutation-verified by hand (issue #932 implementation): temporarily
    /// raising the cap passed to `test_client_and_jobs_with_cap` below from
    /// 2 to 3 makes the 3rd submission succeed and this test's
    /// `expect_err` panic -- confirming the assertion actually depends on
    /// the cap value rather than trivially passing regardless.
    #[test]
    fn submit_rejects_once_admission_cap_reached() {
        let cap = 2;
        let (client, job_rx) = test_client_and_jobs_with_cap(cap);
        let started = Arc::new(AtomicUsize::new(0));
        let ran_tokens = Arc::new(AtomicUsize::new(0));
        let started2 = started.clone();
        let ran2 = ran_tokens.clone();
        let handle = std::thread::spawn(move || {
            run_worker_loop(job_rx, fake_generate(2000, started2, ran2))
        });

        // Job 1: admitted, immediately dequeued (nothing else queued yet),
        // and running fake_generate's 2000-iteration/5ms-per-iteration
        // loop -- long enough to stay in-flight for the rest of this test.
        let (guard1, cancel1) = crate::serve::cancel_pair();
        let rx1 = client
            .submit(
                vec![ChatMessage::user("hi")],
                GenerateConfig::default(),
                cancel1,
            )
            .expect("job 1 must be admitted: cap=2, 0 outstanding");
        std::thread::sleep(Duration::from_millis(30));

        // Job 2: admitted (2nd of 2 permits); sits queued behind job 1
        // since the single worker thread is still busy with it.
        let (guard2, cancel2) = crate::serve::cancel_pair();
        let rx2 = client
            .submit(
                vec![ChatMessage::user("hi")],
                GenerateConfig::default(),
                cancel2,
            )
            .expect("job 2 must be admitted: cap=2, 1 outstanding");

        // Job 3: cap is now full (job 1 in-flight + job 2 queued == 2 ==
        // cap) -- must be rejected, and must never reach the job channel
        // (no tokenization/model work for a rejected admission).
        let (_guard3, cancel3) = crate::serve::cancel_pair();
        let err = client
            .submit(
                vec![ChatMessage::user("hi")],
                GenerateConfig::default(),
                cancel3,
            )
            .expect_err("job 3 must be rejected once the cap is reached");
        match err {
            ApiError::ServiceUnavailable { message } => {
                assert!(
                    message.contains("outstanding") || message.contains("pending"),
                    "rejection message should explain admission capacity: {message}"
                );
            }
            other => panic!("expected ServiceUnavailable, got {other:?}"),
        }

        // Cleanup: cancel jobs 1 and 2 so fake_generate's should_cancel
        // check stops them quickly, then drain and join.
        drop(guard1);
        drop(guard2);
        drop(rx1);
        drop(rx2);
        drop(client);
        handle.join().expect("worker thread must not panic");
    }

    /// Slot release on NORMAL completion: a cap=1 client must admit a
    /// second job only after the first job's terminal `Complete` event has
    /// been delivered and `run_worker_loop` has moved past it (dropping the
    /// `WorkerJob`, and with it the admission permit it owns).
    #[test]
    fn admission_slot_is_released_when_a_job_completes() {
        let cap = 1;
        let (client, job_rx) = test_client_and_jobs_with_cap(cap);
        let started = Arc::new(AtomicUsize::new(0));
        let ran_tokens = Arc::new(AtomicUsize::new(0));
        let started2 = started.clone();
        let ran2 = ran_tokens.clone();
        let handle =
            std::thread::spawn(move || run_worker_loop(job_rx, fake_generate(5, started2, ran2)));

        let (_guard1, cancel1) = crate::serve::cancel_pair();
        let mut rx1 = client
            .submit(
                vec![ChatMessage::user("hi")],
                GenerateConfig::default(),
                cancel1,
            )
            .expect("job 1 must be admitted");

        // Drain job 1 to its terminal Complete event -- fake_generate(5, ..)
        // runs to completion in ~25ms and is never cancelled.
        let mut completed = false;
        while let Some(ev) = rx1.blocking_recv() {
            if matches!(ev, WorkerEvent::Complete(_)) {
                completed = true;
            }
        }
        assert!(completed, "job 1 must complete normally");

        // The permit `run_worker_loop` held for job 1 is dropped along with
        // `job` at the end of that loop iteration, essentially immediately
        // after the `Complete` send above -- retry briefly rather than
        // assume that has already happened on this exact instruction by the
        // time this (different) thread observes the event.
        let mut admitted = false;
        for _ in 0..50 {
            let (_guard2, cancel2) = crate::serve::cancel_pair();
            match client.submit(
                vec![ChatMessage::user("hi")],
                GenerateConfig::default(),
                cancel2,
            ) {
                Ok(_rx2) => {
                    admitted = true;
                    break;
                }
                Err(_) => std::thread::sleep(Duration::from_millis(5)),
            }
        }
        assert!(
            admitted,
            "slot must be released once job 1 completes, admitting job 2 at the same cap=1"
        );

        drop(client);
        handle.join().expect("worker thread must not panic");
    }

    /// THE REGRESSION THIS TEST GUARDS (issue #932): a client-cancelled
    /// job that is still sitting in the queue (not yet dequeued) must NOT
    /// release its admission slot early -- it is still real, unprocessed
    /// work occupying a place in the FIFO queue -- but once the worker
    /// actually dequeues it and observes the cancellation (sending exactly
    /// one `WorkerEvent::Cancelled`, the existing #832 dequeue-time-cancel
    /// contract), its slot MUST be released, same as any other terminal
    /// outcome. A permit leaked specifically on this path would let the
    /// outstanding-job count only ever grow -- every cancelled queued
    /// request would permanently cost one admission slot, eventually
    /// wedging admission shut with zero real work outstanding.
    #[test]
    fn admission_slot_is_released_when_a_queued_job_is_cancelled() {
        let cap = 2;
        let (client, job_rx) = test_client_and_jobs_with_cap(cap);
        let started = Arc::new(AtomicUsize::new(0));
        let ran_tokens = Arc::new(AtomicUsize::new(0));
        let started2 = started.clone();
        let ran2 = ran_tokens.clone();
        let handle = std::thread::spawn(move || {
            run_worker_loop(job_rx, fake_generate(2000, started2, ran2))
        });

        // Job 1: admitted, immediately dequeued, running.
        let (guard1, cancel1) = crate::serve::cancel_pair();
        let rx1 = client
            .submit(
                vec![ChatMessage::user("hi")],
                GenerateConfig::default(),
                cancel1,
            )
            .expect("job 1 must be admitted");
        std::thread::sleep(Duration::from_millis(30));

        // Job 2: admitted (2nd of 2 permits), queued behind job 1. Cancel
        // it immediately, client-side, WHILE it is still sitting in the
        // queue, unprocessed.
        let (guard2, cancel2) = crate::serve::cancel_pair();
        let mut rx2 = client
            .submit(
                vec![ChatMessage::user("hi")],
                GenerateConfig::default(),
                cancel2,
            )
            .expect("job 2 must be admitted");
        drop(guard2);

        // Cap is full (2/2) right now: a 3rd submit must be rejected --
        // proving a client-cancelled-but-still-queued job legitimately
        // still occupies its slot before it has actually been dequeued.
        let (_guard3, cancel3) = crate::serve::cancel_pair();
        client
            .submit(
                vec![ChatMessage::user("hi")],
                GenerateConfig::default(),
                cancel3,
            )
            .expect_err("cap must still be full: job 2's slot isn't released until dequeued");

        // Let job 1 finish (cancel it too) so the worker dequeues job 2
        // next, observes its cancel flag, and emits exactly one Cancelled
        // event for it.
        drop(guard1);
        match rx2.blocking_recv() {
            Some(WorkerEvent::Cancelled) => {}
            other => panic!("expected job 2's exactly-one Cancelled event, got {other:?}"),
        }

        // The slot must now be free -- and specifically BOTH slots, not
        // just one. A single successful 4th admission (the original form
        // of this assertion) does not distinguish "job 2's queued-cancel
        // path correctly released its own permit" from "only job 1's
        // ordinary completion released a permit and job 2's leaked": at
        // this point job 1 has already finished (releasing one permit
        // unconditionally, regression or not), so a leak confined to job
        // 2's queued-cancel path still leaves exactly one usable permit --
        // enough for one admission to spuriously succeed. Poll the
        // semaphore's own count directly (this test module is a child of
        // `metal_worker`, so `client.admission` -- private outside this
        // file -- is visible here) rather than relying on dequeue timing
        // for a second, indirect proof.
        let mut permits_restored = false;
        for _ in 0..50 {
            if client.admission.available_permits() == cap {
                permits_restored = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            permits_restored,
            "both permits (job 1's own release AND job 2's queued-cancel release) must be \
             free once job 2's Cancelled event has fired, got {} of {cap}",
            client.admission.available_permits()
        );

        // And the caller-observable contract still holds: a fresh
        // admission at full cap succeeds.
        let (_guard4, cancel4) = crate::serve::cancel_pair();
        client
            .submit(
                vec![ChatMessage::user("hi")],
                GenerateConfig::default(),
                cancel4,
            )
            .expect("job 2's slot must be released after its Cancelled event, not leaked");

        drop(rx1);
        drop(client);
        handle.join().expect("worker thread must not panic");
    }
}
