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
//! # Bounded shutdown
//!
//! Every production [`MetalWorkerClient`] retains a clone of
//! [`MetalWorkerOwner`]. Dropping the last client first closes the job
//! queue, then dropping the last owner waits for the worker to exit and
//! joins it. The wait has a two-second deadline: a backend call that stops
//! polling cancellation can delay the worker, but cannot hang process
//! shutdown indefinitely. On timeout the join handle is detached and the
//! process remains free to exit.
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

use crate::forward::metal_qwen35::{
    ChatMessage, MetalQwen35State, format_chat_template, push_chat_turn_close, push_chat_turn_open,
};
use crate::kv_cache::CrossTurnSlotId;
use crate::model::qwen35_config::{
    GenerateConfig, GenerateOutput, Qwen35Config, VisionModelConfig,
};
use crate::serve::ApiError;
use crate::tokenizer::Tokenizer as _;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::vision::checkpoint::{
    Qwen35VisionWeights, load_qwen35_vision_weights_with_cancel,
    validate_qwen35_vision_weight_inventory,
};
use crate::vision::multimodal::Qwen35VisionRequest;
use crate::vision::qwen35_merger::qwen35_merger_forward_with_cancel;
use crate::vision::qwen35_vit::preprocess_qwen35_image_for_serve;
use crate::vision::qwen35_vit_metal::qwen35_vit_forward_metal_with_cancel;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
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

const WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
enum WorkerShutdown {
    Joined,
    AlreadyStopped,
    TimedOut,
    Panicked,
}

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
    /// pattern-matching the message text (a backend wording change must not
    /// be able to silently degrade that code to `internal_error`). Carries the
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
#[derive(Debug)]
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

/// Shared owner for the dedicated worker thread.
///
/// Production [`MetalWorkerClient`] values retain an owner clone. Because
/// the client's queue sender is declared before that clone, dropping the
/// final client closes the queue before the final owner begins its bounded
/// join. The last owner's `Drop` is the sole production shutdown trigger;
/// there is no explicit method that can detach the join handle while a
/// client still keeps the queue open.
#[derive(Debug, Clone)]
pub struct MetalWorkerOwner {
    _inner: Arc<MetalWorkerOwnerInner>,
}

#[derive(Debug)]
struct MetalWorkerOwnerInner {
    join_handle: Mutex<Option<std::thread::JoinHandle<()>>>,
    drop_timeout: Duration,
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl MetalWorkerOwnerInner {
    fn wait_for_exit(&self, timeout: Duration) -> WorkerShutdown {
        let handle = {
            let mut join_handle = lock_unpoisoned(&self.join_handle);
            let Some(handle) = join_handle.take() else {
                return WorkerShutdown::AlreadyStopped;
            };
            handle
        };

        let started = Instant::now();
        while !handle.is_finished() {
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                drop(handle);
                return WorkerShutdown::TimedOut;
            }
            std::thread::sleep(remaining.min(Duration::from_millis(5)));
        }

        if handle.join().is_ok() {
            WorkerShutdown::Joined
        } else {
            WorkerShutdown::Panicked
        }
    }
}

impl Drop for MetalWorkerOwnerInner {
    fn drop(&mut self) {
        match self.wait_for_exit(self.drop_timeout) {
            WorkerShutdown::Joined | WorkerShutdown::AlreadyStopped => {}
            WorkerShutdown::TimedOut => {
                let _ = writeln!(
                    std::io::stderr().lock(),
                    "[metal-worker] shutdown timed out after {} ms; detaching worker thread",
                    self.drop_timeout.as_millis()
                );
            }
            WorkerShutdown::Panicked => {
                let _ = writeln!(
                    std::io::stderr().lock(),
                    "[metal-worker] worker thread panicked during shutdown"
                );
            }
        }
    }
}

impl MetalWorkerOwner {
    fn from_handle(join_handle: std::thread::JoinHandle<()>) -> Self {
        Self::from_handle_with_timeout(join_handle, WORKER_SHUTDOWN_TIMEOUT)
    }

    fn from_handle_with_timeout(
        join_handle: std::thread::JoinHandle<()>,
        drop_timeout: Duration,
    ) -> Self {
        Self {
            _inner: Arc::new(MetalWorkerOwnerInner {
                join_handle: Mutex::new(Some(join_handle)),
                drop_timeout,
            }),
        }
    }

    #[cfg(any(test, feature = "test-utils"))]
    fn unattached_for_test() -> Self {
        Self {
            _inner: Arc::new(MetalWorkerOwnerInner {
                join_handle: Mutex::new(None),
                drop_timeout: WORKER_SHUTDOWN_TIMEOUT,
            }),
        }
    }
}

/// Cheaply `Clone` (an `mpsc` sender) handle used to submit generation
/// requests to the worker thread. `Send + Sync` so it lives in a binary's
/// `AppState` the same way the CPU backend's `Arc<Qwen35Model>` does --
/// only the underlying `MetalQwen35State` inside the worker thread is
/// confined to that thread.
#[derive(Debug, Clone)]
pub struct MetalWorkerClient {
    // Field order is the shutdown contract: the final sender closes before
    // the final owner clone starts its bounded join.
    jobs: mpsc::UnboundedSender<WorkerJob>,
    /// Bounded-admission cap (issue #932): `Semaphore::new(max_pending)`, one
    /// permit per outstanding job (queued + in-flight, i.e. from `submit`
    /// until `run_worker_loop` is done with it). `Arc`-shared with every
    /// clone of this client so the cap is process-wide, not per-clone.
    admission: Arc<Semaphore>,
    vision_supported: Arc<AtomicBool>,
    /// Keeps the worker join owner alive for exactly as long as the queue
    /// can accept jobs. Test-only clients without a worker carry an owner
    /// whose join slot is already empty.
    _owner: MetalWorkerOwner,
}

impl MetalWorkerClient {
    fn with_owner(
        jobs: mpsc::UnboundedSender<WorkerJob>,
        admission: Arc<Semaphore>,
        vision_supported: Arc<AtomicBool>,
        owner: MetalWorkerOwner,
    ) -> Self {
        Self {
            jobs,
            admission,
            vision_supported,
            _owner: owner,
        }
    }

    #[cfg(any(test, feature = "test-utils"))]
    fn unattached_for_test(
        jobs: mpsc::UnboundedSender<WorkerJob>,
        admission: Arc<Semaphore>,
    ) -> Self {
        Self::with_owner(
            jobs,
            admission,
            Arc::new(AtomicBool::new(false)),
            MetalWorkerOwner::unattached_for_test(),
        )
    }

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

    /// Whether this worker currently accepts image content.
    ///
    /// Starts `true` only for a concrete vision-capable checkpoint and
    /// flips to `false` if its first lazy vision-weight load fails
    /// terminally.
    pub fn supports_vision(&self) -> bool {
        self.vision_supported.load(Ordering::Acquire)
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

enum VisionState {
    Unsupported,
    Pending {
        model_dir: PathBuf,
        config: VisionModelConfig,
    },
    Loaded(Qwen35VisionWeights),
    Failed(String),
}

#[derive(Debug)]
enum VisionRuntimeLoad<'a> {
    Ready(&'a Qwen35VisionWeights),
    Unsupported,
    Cancelled,
}

/// Worker-local vision capability and lazily loaded vision weights.
///
/// Loading stays on the same dedicated thread that owns the Metal decoder
/// and happens only for the first admitted image request. Text-only servers
/// and text-only traffic retain the pre-vision startup and memory profile.
pub struct VisionRuntime {
    state: VisionState,
    vision_supported: Arc<AtomicBool>,
}

impl VisionRuntime {
    /// Resolve a runtime from concrete checkpoint metadata.
    pub fn from_model_config(model_dir: PathBuf, config: &Qwen35Config) -> Self {
        let token_metadata_present = match (
            config.image_token_id,
            config.vision_start_token_id,
            config.vision_end_token_id,
        ) {
            (Some(image), Some(start), Some(end)) => {
                [image, start, end]
                    .into_iter()
                    .all(|token| (token as usize) < config.vocab_size)
                    && image != start
                    && image != end
                    && start != end
            }
            _ => false,
        };
        let supported_weight_source = token_metadata_present
            && config.vision_config.as_ref().is_some_and(|vision_config| {
                validate_qwen35_vision_weight_inventory(&model_dir, vision_config).is_ok()
            });
        let state = match (
            &config.vision_config,
            token_metadata_present,
            supported_weight_source,
        ) {
            (Some(vision_config), true, true) => VisionState::Pending {
                model_dir,
                config: vision_config.clone(),
            },
            _ => VisionState::Unsupported,
        };
        let vision_supported =
            matches!(&state, VisionState::Pending { .. } | VisionState::Loaded(_));
        Self {
            state,
            vision_supported: Arc::new(AtomicBool::new(vision_supported)),
        }
    }

    /// Text-only runtime used by the compatibility [`MetalWorker::spawn`]
    /// entry point and GPU-free worker tests.
    pub fn unsupported() -> Self {
        Self {
            state: VisionState::Unsupported,
            vision_supported: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Whether request normalization may admit image content parts.
    pub fn is_supported(&self) -> bool {
        self.vision_supported.load(Ordering::Acquire)
    }

    fn shared_capability(&self) -> Arc<AtomicBool> {
        self.vision_supported.clone()
    }

    fn get_or_load(
        &mut self,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<VisionRuntimeLoad<'_>, String> {
        if let VisionState::Pending { model_dir, config } = &self.state {
            let model_dir = model_dir.clone();
            let config = config.clone();
            match load_qwen35_vision_weights_with_cancel(&model_dir, &config, should_cancel) {
                Ok(Some(weights)) => self.state = VisionState::Loaded(weights),
                Ok(None) => return Ok(VisionRuntimeLoad::Cancelled),
                Err(err) => {
                    let message = format!("vision weights failed to load: {err}");
                    self.state = VisionState::Failed(message.clone());
                    self.vision_supported.store(false, Ordering::Release);
                    return Err(message);
                }
            }
        }
        match &self.state {
            VisionState::Unsupported => Ok(VisionRuntimeLoad::Unsupported),
            VisionState::Loaded(weights) => Ok(VisionRuntimeLoad::Ready(weights)),
            VisionState::Failed(message) => Err(message.clone()),
            VisionState::Pending { .. } => {
                self.vision_supported.store(false, Ordering::Release);
                Err("vision weights remained pending after a load attempt".to_string())
            }
        }
    }
}

#[cfg(test)]
fn tokenize_text(tokenizer: &BpeTokenizer, text: &str) -> Vec<u32> {
    let encoded = tokenizer.tokenize(text);
    encoded.input_ids[..encoded.real_length].to_vec()
}

#[allow(clippy::too_many_arguments)]
fn build_vision_prompt_ids(
    messages: &[ChatMessage],
    image_message_index: usize,
    tokenizer: &BpeTokenizer,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    image_token_id: u32,
    image_pad_count: usize,
) -> Result<Vec<u32>, WorkerFailure> {
    let image_message = &messages[image_message_index];
    let image = image_message
        .image
        .as_ref()
        .ok_or_else(|| WorkerFailure::Failed("vision dispatch lost its image payload".into()))?;
    if image.text_offset > image_message.content.len()
        || !image_message.content.is_char_boundary(image.text_offset)
    {
        return Err(WorkerFailure::Failed(
            "normalized image text offset is not a UTF-8 boundary".into(),
        ));
    }

    let mut before = String::new();
    for message in &messages[..image_message_index] {
        push_chat_turn_open(&mut before, message.role.as_str());
        before.push_str(&message.content);
        push_chat_turn_close(&mut before);
    }
    push_chat_turn_open(&mut before, image_message.role.as_str());
    before.push_str(&image_message.content[..image.text_offset]);

    let mut after = String::new();
    after.push_str(&image_message.content[image.text_offset..]);
    push_chat_turn_close(&mut after);
    for message in &messages[image_message_index + 1..] {
        push_chat_turn_open(&mut after, message.role.as_str());
        after.push_str(&message.content);
        push_chat_turn_close(&mut after);
    }
    after.push_str("<|im_start|>assistant\n");

    let mut inserted_ids = Vec::with_capacity(image_pad_count.saturating_add(2));
    inserted_ids.push(vision_start_token_id);
    inserted_ids.extend(std::iter::repeat_n(image_token_id, image_pad_count));
    inserted_ids.push(vision_end_token_id);
    let ids = tokenizer.tokenize_fragments_with_inserted_ids(&before, &inserted_ids, &after);
    if ids.iter().filter(|&&id| id == image_token_id).count() != image_pad_count {
        return Err(WorkerFailure::Rejected(ApiError::BadRequest {
            message: "message text must not contain the checkpoint's reserved image token"
                .to_string(),
            code: "invalid_messages",
        }));
    }
    Ok(ids)
}

enum VisionRequestBuild {
    Ready {
        request: Qwen35VisionRequest,
        metal_dispatches: usize,
        gemm_calls: usize,
    },
    Cancelled,
}

fn build_vision_request(
    runtime: &mut VisionRuntime,
    config: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    messages: &[ChatMessage],
    image_message_index: usize,
    should_cancel: &mut dyn FnMut() -> bool,
    window_preflight: impl FnOnce(usize) -> Result<(), ApiError>,
) -> Result<VisionRequestBuild, WorkerFailure> {
    if should_cancel() {
        return Ok(VisionRequestBuild::Cancelled);
    }
    let vision_config = config.vision_config.as_ref().ok_or_else(|| {
        WorkerFailure::Rejected(ApiError::BadRequest {
            message: "image input requires a vision-capable model".to_string(),
            code: "vision_unsupported",
        })
    })?;
    let image_token_id = config.image_token_id.ok_or_else(|| {
        WorkerFailure::Failed("vision checkpoint has no image_token_id".to_string())
    })?;
    let vision_start_token_id = config.vision_start_token_id.ok_or_else(|| {
        WorkerFailure::Failed("vision checkpoint has no vision_start_token_id".to_string())
    })?;
    let vision_end_token_id = config.vision_end_token_id.ok_or_else(|| {
        WorkerFailure::Failed("vision checkpoint has no vision_end_token_id".to_string())
    })?;
    let image = messages[image_message_index]
        .image
        .as_ref()
        .ok_or_else(|| WorkerFailure::Failed("vision dispatch lost its image payload".into()))?;

    let (pixel_values, grid) = preprocess_qwen35_image_for_serve(&image.bytes, vision_config)
        .map_err(|err| {
            WorkerFailure::Rejected(ApiError::BadRequest {
                message: format!("image preprocessing failed: {err}"),
                code: "invalid_image",
            })
        })?;
    if should_cancel() {
        return Ok(VisionRequestBuild::Cancelled);
    }
    let merge_area = vision_config
        .spatial_merge_size
        .checked_mul(vision_config.spatial_merge_size)
        .filter(|&area| area > 0)
        .ok_or_else(|| WorkerFailure::Failed("vision spatial_merge_size is invalid".to_string()))?;
    if !grid.num_patches().is_multiple_of(merge_area) {
        return Err(WorkerFailure::Rejected(ApiError::BadRequest {
            message: "image patch grid is incompatible with the checkpoint merge size".to_string(),
            code: "invalid_image",
        }));
    }
    let image_pad_count = grid.num_patches() / merge_area;
    let input_ids = build_vision_prompt_ids(
        messages,
        image_message_index,
        tokenizer,
        vision_start_token_id,
        vision_end_token_id,
        image_token_id,
        image_pad_count,
    )?;
    if should_cancel() {
        return Ok(VisionRequestBuild::Cancelled);
    }
    window_preflight(input_ids.len()).map_err(WorkerFailure::Rejected)?;
    if should_cancel() {
        return Ok(VisionRequestBuild::Cancelled);
    }

    let weights = match runtime
        .get_or_load(should_cancel)
        .map_err(WorkerFailure::Failed)?
    {
        VisionRuntimeLoad::Ready(weights) => weights,
        VisionRuntimeLoad::Unsupported => {
            return Err(WorkerFailure::Rejected(ApiError::BadRequest {
                message: "image input requires a vision-capable model".to_string(),
                code: "vision_unsupported",
            }));
        }
        VisionRuntimeLoad::Cancelled => return Ok(VisionRequestBuild::Cancelled),
    };
    if should_cancel() {
        return Ok(VisionRequestBuild::Cancelled);
    }
    let Some(vit_output) = qwen35_vit_forward_metal_with_cancel(
        weights,
        vision_config,
        &pixel_values,
        grid,
        should_cancel,
    )
    .map_err(|err| WorkerFailure::Failed(format!("vision forward failed: {err}")))?
    else {
        return Ok(VisionRequestBuild::Cancelled);
    };
    let Some(post_merger) = qwen35_merger_forward_with_cancel(
        &weights.merger,
        vision_config,
        &vit_output.hidden_states,
        should_cancel,
    )
    .map_err(|err| WorkerFailure::Failed(format!("vision merger failed: {err}")))?
    else {
        return Ok(VisionRequestBuild::Cancelled);
    };
    if should_cancel() {
        return Ok(VisionRequestBuild::Cancelled);
    }

    Ok(VisionRequestBuild::Ready {
        request: Qwen35VisionRequest {
            input_ids,
            image_grids: vec![grid],
            post_merger_rows: post_merger,
            image_token_id,
            spatial_merge_size: vision_config.spatial_merge_size,
            decoder_hidden_size: config.hidden_size,
        },
        metal_dispatches: vit_output.metal_dispatches,
        gemm_calls: vit_output.gemm_calls,
    })
}

fn cancelled_output() -> GenerateOutput {
    GenerateOutput {
        text: String::new(),
        token_ids: Vec::new(),
        prompt_tokens: 0,
        generated_tokens: 0,
        stopped: false,
        stop_reason: Some(crate::StopReason::Interrupt),
        token_logprobs: Vec::new(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobRoute {
    Text,
    Vision { message_index: usize },
}

fn classify_job(messages: &[ChatMessage]) -> Result<JobRoute, WorkerFailure> {
    let mut image_positions = messages
        .iter()
        .enumerate()
        .filter(|(_, message)| message.image.is_some());
    let first = image_positions.next();
    if image_positions.next().is_some() {
        return Err(WorkerFailure::Rejected(ApiError::BadRequest {
            message: "only one image is supported per request".to_string(),
            code: "multiple_images_unsupported",
        }));
    }
    Ok(match first {
        Some((_message_index, message))
            if message.role != crate::forward::metal_qwen35::ChatRole::User =>
        {
            return Err(WorkerFailure::Rejected(ApiError::BadRequest {
                message: "image content is supported only on user messages".to_string(),
                code: "invalid_image_role",
            }));
        }
        Some((message_index, _)) => JobRoute::Vision { message_index },
        None => JobRoute::Text,
    })
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
        loader: impl FnOnce() -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String>
        + Send
        + 'static,
        max_pending: usize,
    ) -> Result<(MetalWorkerOwner, MetalWorkerClient, WorkerMetadata), StartupError> {
        Self::spawn_with_vision(loader, VisionRuntime::unsupported(), max_pending)
    }

    /// Vision-capable sibling of [`Self::spawn`].
    ///
    /// `vision_runtime` is derived from the same concrete checkpoint config
    /// as `loader`. It remains worker-local and loads vision tensors only
    /// when the first image-bearing job is actually dispatched.
    pub fn spawn_with_vision(
        loader: impl FnOnce() -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String>
        + Send
        + 'static,
        mut vision_runtime: VisionRuntime,
        max_pending: usize,
    ) -> Result<(MetalWorkerOwner, MetalWorkerClient, WorkerMetadata), StartupError> {
        let vision_supported = vision_runtime.shared_capability();
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
            Ok((mut state, tokenizer, meta)) => {
                let _ = ready_tx.send(Ok(meta.clone()));
                run_worker_loop(job_rx, move |messages, cfg, on_token, should_cancel| {
                    if let JobRoute::Vision {
                        message_index: image_message_index,
                    } = classify_job(messages)?
                    {
                        if should_cancel() {
                            return Ok(cancelled_output());
                        }
                        let config = state.engine.config.clone();
                        let (request, metal_dispatches, gemm_calls) = match build_vision_request(
                            &mut vision_runtime,
                            &config,
                            &tokenizer,
                            messages,
                            image_message_index,
                            should_cancel,
                            |prompt_len| {
                                check_prompt_fits_window(
                                    meta.context_window_policy,
                                    meta.model_max_context,
                                    prompt_len,
                                    cfg,
                                )
                            },
                        )? {
                            VisionRequestBuild::Ready {
                                request,
                                metal_dispatches,
                                gemm_calls,
                            } => (request, metal_dispatches, gemm_calls),
                            VisionRequestBuild::Cancelled => return Ok(cancelled_output()),
                        };
                        if should_cancel() {
                            return Ok(cancelled_output());
                        }
                        eprintln!(
                            "[metal-worker] route=vision dispatch=multimodal \
                             metal_gemm_dispatches={metal_dispatches} \
                             metal_gemm_calls={gemm_calls}"
                        );
                        let output = state
                            .generate_multimodal_vision_with_cancel(
                                &request,
                                &tokenizer,
                                cfg,
                                should_cancel,
                            )
                            .map_err(WorkerFailure::from)?;
                        if !output.text.is_empty() {
                            let _ = on_token(&output.text, 0);
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
                    //
                    // DEPLOYMENT ASSUMPTION, stated because it is currently
                    // true only by the accident that no multi-tenant consumer
                    // exists: this path assumes a single tenant, or clients
                    // that mutually trust one another. Reuse-versus-refill is
                    // externally visible as latency, so while no request can
                    // read another's content, a client CAN observe that some
                    // other request recently shared a prefix with its own.
                    // A shared inference endpoint serving mutually distrusting
                    // clients must key the slot per tenant via
                    // `CrossTurnSlotId::new`, not inherit `DEFAULT`.
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

        let owner = MetalWorkerOwner::from_handle(join_handle);
        match ready_rx.recv() {
            Ok(Ok(meta)) => {
                let client = MetalWorkerClient::with_owner(
                    job_tx,
                    admission,
                    vision_supported,
                    owner.clone(),
                );
                Ok((owner, client, meta))
            }
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
        MetalWorkerClient::unattached_for_test(job_tx, Arc::new(Semaphore::new(max_pending))),
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
    spawn_fake_with_capability(
        max_pending,
        context_window_policy,
        model_max_context,
        tokenizer,
        false,
        generate,
    )
}

/// Vision-admitting sibling of [`spawn_fake`] for cross-binary HTTP tests.
///
/// The fake still substitutes only the terminal Metal call, but advertises
/// vision capability so request normalization can prove that an admitted
/// image survives both binary adapters and reaches their common worker job.
#[cfg(any(test, feature = "test-utils"))]
#[allow(clippy::type_complexity)]
pub fn spawn_fake_with_vision(
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
    spawn_fake_with_capability(
        TEST_EFFECTIVELY_UNBOUNDED_CAP,
        context_window_policy,
        model_max_context,
        tokenizer,
        true,
        generate,
    )
}

#[cfg(any(test, feature = "test-utils"))]
#[allow(clippy::type_complexity)]
fn spawn_fake_with_capability(
    max_pending: usize,
    context_window_policy: ContextWindowPolicy,
    model_max_context: usize,
    tokenizer: BpeTokenizer,
    vision_supported: bool,
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
    let join_handle = std::thread::spawn(move || {
        run_worker_loop(job_rx, move |messages, cfg, on_token, should_cancel| {
            let prompt = format_chat_template(messages);
            let prompt_tokens = tokenizer.tokenize(&prompt).real_length;
            check_prompt_fits_window(context_window_policy, model_max_context, prompt_tokens, cfg)
                .map_err(WorkerFailure::Rejected)?;
            generate(messages, cfg, prompt_tokens, on_token, should_cancel)
                .map_err(WorkerFailure::Failed)
        });
    });
    let owner = MetalWorkerOwner::from_handle(join_handle);
    MetalWorkerClient::with_owner(
        job_tx,
        Arc::new(Semaphore::new(max_pending)),
        Arc::new(AtomicBool::new(vision_supported)),
        owner,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    fn test_owner(
        join_handle: std::thread::JoinHandle<()>,
        drop_timeout: Duration,
    ) -> MetalWorkerOwner {
        MetalWorkerOwner::from_handle_with_timeout(join_handle, drop_timeout)
    }

    fn tiny_tokenizer() -> BpeTokenizer {
        BpeTokenizer::from_vocab_and_merges(
            HashMap::from([
                ("a".to_string(), 0),
                ("b".to_string(), 1),
                ("e".to_string(), 2),
                ("f".to_string(), 3),
                ("o".to_string(), 4),
                ("r".to_string(), 5),
            ]),
            Vec::new(),
        )
        .expect("tiny tokenizer must construct")
    }

    fn tiny_vision_config(out_hidden_size: usize) -> VisionModelConfig {
        VisionModelConfig {
            depth: 1,
            hidden_size: 8,
            num_heads: 2,
            patch_size: 2,
            spatial_merge_size: 2,
            out_hidden_size,
            temporal_patch_size: 1,
            num_position_embeddings: 16,
            in_channels: 3,
            deepstack_visual_indexes: Vec::new(),
            intermediate_size: None,
        }
    }

    fn make_test_png(width: u32, height: u32) -> Vec<u8> {
        let mut image = image::RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let value = ((x + y) % 256) as u8;
                image.put_pixel(x, y, image::Rgb([value, value, value]));
            }
        }
        let mut bytes = Vec::new();
        image
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .expect("test PNG encode");
        bytes
    }

    #[test]
    fn image_job_classification_enforces_role_and_multiplicity() {
        assert_eq!(
            classify_job(&[ChatMessage::user("hello")]).expect("text job"),
            JobRoute::Text
        );
        let image = ChatMessage::user_with_image("beforeafter", vec![1, 2, 3], 6);
        assert_eq!(
            classify_job(&[ChatMessage::system("policy"), image.clone()]).expect("one image job"),
            JobRoute::Vision { message_index: 1 }
        );
        let err = classify_job(&[image.clone(), image]).expect_err("two images must fail");
        assert!(matches!(
            err,
            WorkerFailure::Rejected(ApiError::BadRequest {
                code: "multiple_images_unsupported",
                ..
            })
        ));
        for role in [
            crate::forward::metal_qwen35::ChatRole::System,
            crate::forward::metal_qwen35::ChatRole::Assistant,
        ] {
            let mut non_user_image = ChatMessage::user_with_image("beforeafter", vec![1, 2, 3], 6);
            non_user_image.role = role;
            let err = classify_job(&[non_user_image]).expect_err("non-user image role must fail");
            assert!(matches!(
                err,
                WorkerFailure::Rejected(ApiError::BadRequest {
                    code: "invalid_image_role",
                    ..
                })
            ));
        }
    }

    #[test]
    fn vision_prompt_splices_tokens_at_original_content_part_position() {
        let tokenizer = tiny_tokenizer();
        let messages = vec![
            ChatMessage::system("policy"),
            ChatMessage::user_with_image("beforeafter", vec![1], "before".len()),
            ChatMessage::assistant("prior"),
        ];
        let actual = build_vision_prompt_ids(&messages, 1, &tokenizer, 90, 91, 92, 3)
            .expect("vision prompt");

        let before = "<|im_start|>system\npolicy<|im_end|>\n<|im_start|>user\nbefore";
        let after =
            "after<|im_end|>\n<|im_start|>assistant\nprior<|im_end|>\n<|im_start|>assistant\n";
        let mut expected = tokenize_text(&tokenizer, before);
        expected.extend([90, 92, 92, 92, 91]);
        expected.extend(tokenize_text(&tokenizer, after));
        assert_eq!(actual, expected);
    }

    #[test]
    fn vision_prompt_tokenizes_the_complete_sequence_before_window_validation() {
        let capped = tiny_tokenizer().with_max_seq_len(4);
        let unbounded = capped.with_max_seq_len(usize::MAX);
        let before = "before".repeat(64);
        let messages = vec![ChatMessage::user_with_image(
            format!("{before}after"),
            vec![1],
            before.len(),
        )];
        let actual = build_vision_prompt_ids(&messages, 0, &capped, 90, 91, 92, 3)
            .expect("capped tokenizer must not truncate prompt fragments");
        let expected = build_vision_prompt_ids(&messages, 0, &unbounded, 90, 91, 92, 3)
            .expect("unbounded tokenizer");
        assert_eq!(actual, expected);
        assert!(actual.len() > 4);
    }

    fn vision_build_config() -> Qwen35Config {
        let mut config = Qwen35Config::qwen35_0_8b();
        config.vision_config = Some(tiny_vision_config(config.hidden_size));
        config.image_token_id = Some(90);
        config.vision_start_token_id = Some(91);
        config.vision_end_token_id = Some(92);
        config
    }

    #[test]
    fn vision_request_build_cancels_before_image_preprocessing() {
        let mut runtime = VisionRuntime::unsupported();
        let config = vision_build_config();
        let messages = vec![ChatMessage::user_with_image(
            "beforeafter",
            b"not an image".to_vec(),
            "before".len(),
        )];
        let mut polls = 0;
        let result = build_vision_request(
            &mut runtime,
            &config,
            &tiny_tokenizer(),
            &messages,
            0,
            &mut || {
                polls += 1;
                true
            },
            |_| panic!("window preflight must not run after cancellation"),
        )
        .expect("cancellation is not a worker failure");
        assert!(matches!(result, VisionRequestBuild::Cancelled));
        assert_eq!(polls, 1);
    }

    #[test]
    fn vision_request_build_cancels_after_preprocess_and_prompt_before_window_check() {
        let mut runtime = VisionRuntime::unsupported();
        let config = vision_build_config();
        let messages = vec![ChatMessage::user_with_image(
            "beforeafter",
            make_test_png(8, 8),
            "before".len(),
        )];
        let mut polls = 0;
        let mut window_checked = false;
        let result = build_vision_request(
            &mut runtime,
            &config,
            &tiny_tokenizer(),
            &messages,
            0,
            &mut || {
                polls += 1;
                polls == 3
            },
            |_| {
                window_checked = true;
                Ok(())
            },
        )
        .expect("cancellation is not a worker failure");
        assert!(matches!(result, VisionRequestBuild::Cancelled));
        assert_eq!(polls, 3);
        assert!(!window_checked);
    }

    #[test]
    fn vision_request_build_cancels_after_window_check_before_lazy_load() {
        let mut runtime = VisionRuntime::unsupported();
        let config = vision_build_config();
        let messages = vec![ChatMessage::user_with_image(
            "beforeafter",
            make_test_png(8, 8),
            "before".len(),
        )];
        let mut polls = 0;
        let mut window_checked = false;
        let result = build_vision_request(
            &mut runtime,
            &config,
            &tiny_tokenizer(),
            &messages,
            0,
            &mut || {
                polls += 1;
                polls == 4
            },
            |_| {
                window_checked = true;
                Ok(())
            },
        )
        .expect("cancellation is not a worker failure");
        assert!(matches!(result, VisionRequestBuild::Cancelled));
        assert_eq!(polls, 4);
        assert!(window_checked);
    }

    #[test]
    fn vision_runtime_capability_requires_config_token_metadata_and_weight_source() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("quantize_index.json"), "[]")
            .expect("weight-source marker");
        let mut config = Qwen35Config::qwen35_0_8b();
        assert!(
            !VisionRuntime::from_model_config(temp.path().to_path_buf(), &config).is_supported()
        );
        config.vision_config = Some(tiny_vision_config(config.hidden_size));
        config.image_token_id = Some(10);
        config.vision_start_token_id = Some(11);
        config.vision_end_token_id = Some(12);
        assert!(
            !VisionRuntime::from_model_config(temp.path().to_path_buf(), &config).is_supported(),
            "an empty manifest must not advertise vision capability"
        );

        let mut names = vec![
            "model.visual.patch_embed.proj.weight".to_string(),
            "model.visual.patch_embed.proj.bias".to_string(),
            "model.visual.pos_embed.weight".to_string(),
            "model.visual.merger.linear_fc1.weight".to_string(),
            "model.visual.merger.linear_fc1.bias".to_string(),
            "model.visual.merger.linear_fc2.weight".to_string(),
            "model.visual.merger.linear_fc2.bias".to_string(),
            "model.visual.merger.norm.weight".to_string(),
            "model.visual.merger.norm.bias".to_string(),
        ];
        for suffix in [
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "mlp.linear_fc1.weight",
            "mlp.linear_fc1.bias",
            "mlp.linear_fc2.weight",
            "mlp.linear_fc2.bias",
            "norm1.weight",
            "norm1.bias",
            "norm2.weight",
            "norm2.bias",
        ] {
            names.push(format!("model.visual.blocks.0.{suffix}"));
        }
        std::fs::write(temp.path().join("visual.bin"), b"inventory preflight")
            .expect("visual tensor marker");
        let entries: Vec<_> = names
            .iter()
            .map(|name| serde_json::json!({"name": name, "file": "visual.bin"}))
            .collect();
        std::fs::write(
            temp.path().join("quantize_index.json"),
            serde_json::to_vec(&entries).expect("manifest fixture"),
        )
        .expect("complete vision manifest");
        let mut invalid_tokens = config.clone();
        invalid_tokens.image_token_id = Some(config.vocab_size as u32);
        assert!(
            !VisionRuntime::from_model_config(temp.path().to_path_buf(), &invalid_tokens)
                .is_supported(),
            "out-of-vocabulary image metadata must not advertise capability"
        );
        invalid_tokens.image_token_id = invalid_tokens.vision_start_token_id;
        assert!(
            !VisionRuntime::from_model_config(temp.path().to_path_buf(), &invalid_tokens)
                .is_supported(),
            "aliased vision token metadata must not advertise capability"
        );
        let mut runtime = VisionRuntime::from_model_config(temp.path().to_path_buf(), &config);
        assert!(runtime.is_supported());
        let (job_tx, _job_rx) = mpsc::unbounded_channel();
        let client = MetalWorkerClient::with_owner(
            job_tx,
            Arc::new(Semaphore::new(1)),
            runtime.shared_capability(),
            MetalWorkerOwner::unattached_for_test(),
        );
        assert!(client.supports_vision());
        let cancelled = runtime
            .get_or_load(&mut || true)
            .expect("cancellation is not a lazy-load failure");
        assert!(matches!(cancelled, VisionRuntimeLoad::Cancelled));
        assert!(runtime.is_supported());
        assert!(
            client.supports_vision(),
            "cancellation must preserve Pending capability for a later retry"
        );
        let mut never_cancel = || false;
        let first_error = runtime
            .get_or_load(&mut never_cancel)
            .expect_err("junk tensor payload must fail its first lazy load");
        assert!(first_error.contains("vision weights failed to load"));
        assert!(!runtime.is_supported());
        assert!(
            !client.supports_vision(),
            "terminal lazy-load failure must revoke the shared client capability"
        );
        std::fs::remove_file(temp.path().join("visual.bin"))
            .expect("remove source after the first attempt");
        let second_error = runtime
            .get_or_load(&mut never_cancel)
            .expect_err("terminal failure must be returned, not retried");
        assert_eq!(second_error, first_error);
        assert!(
            !VisionRuntime::from_model_config(PathBuf::from("/unused"), &config).is_supported(),
            "config metadata alone must not advertise vision without a supported weight source"
        );
        config.vision_end_token_id = None;
        assert!(
            !VisionRuntime::from_model_config(temp.path().to_path_buf(), &config).is_supported()
        );
    }

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
        let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerJob>();
        let started = Arc::new(AtomicUsize::new(0));
        let ran_tokens = Arc::new(AtomicUsize::new(0));
        let started2 = started.clone();
        let ran2 = ran_tokens.clone();
        let join_handle = std::thread::spawn(move || {
            run_worker_loop(job_rx, fake_generate(1, started2, ran2));
        });
        let owner = test_owner(join_handle, Duration::from_secs(1));
        drop(job_tx);
        assert_eq!(
            owner._inner.wait_for_exit(Duration::from_secs(1)),
            WorkerShutdown::Joined
        );
        assert_eq!(
            owner._inner.wait_for_exit(Duration::ZERO),
            WorkerShutdown::AlreadyStopped,
            "the join handle must be claimed exactly once"
        );
        assert_eq!(started.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn final_client_drop_closes_queue_before_owner_joins() {
        let (job_tx, mut job_rx) = mpsc::unbounded_channel::<WorkerJob>();
        let (queue_closed_tx, queue_closed_rx) = std::sync::mpsc::sync_channel(1);
        let (allow_exit_tx, allow_exit_rx) = std::sync::mpsc::sync_channel(1);
        let join_handle = std::thread::spawn(move || {
            while job_rx.blocking_recv().is_some() {}
            let _ = queue_closed_tx.send(());
            let _ = allow_exit_rx.recv();
        });
        let owner = test_owner(join_handle, Duration::from_secs(1));
        let client = MetalWorkerClient::with_owner(
            job_tx,
            Arc::new(Semaphore::new(1)),
            Arc::new(AtomicBool::new(false)),
            owner.clone(),
        );
        drop(owner);

        let (drop_done_tx, drop_done_rx) = std::sync::mpsc::sync_channel(1);
        let drop_thread = std::thread::spawn(move || {
            drop(client);
            let _ = drop_done_tx.send(());
        });
        queue_closed_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("dropping the last client must close the queue");
        assert!(
            matches!(
                drop_done_rx.try_recv(),
                Err(std::sync::mpsc::TryRecvError::Empty)
            ),
            "last client drop must still be waiting while the worker is live"
        );
        allow_exit_tx
            .send(())
            .expect("the worker must still be waiting for the exit release");
        drop_done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("last client drop must finish after the worker exits");
        drop_thread
            .join()
            .expect("client drop thread must not panic");
    }

    #[test]
    fn final_client_drop_timeout_detaches_instead_of_blocking() {
        let (worker_started_tx, worker_started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_worker_tx, release_worker_rx) = std::sync::mpsc::sync_channel(1);
        let (worker_done_tx, worker_done_rx) = std::sync::mpsc::sync_channel(1);
        let join_handle = std::thread::spawn(move || {
            let _ = worker_started_tx.send(());
            let _ = release_worker_rx.recv();
            let _ = worker_done_tx.send(());
        });
        worker_started_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("the worker must reach its stuck-backend stand-in");
        let owner = test_owner(join_handle, Duration::from_millis(20));
        let (job_tx, _job_rx) = mpsc::unbounded_channel::<WorkerJob>();
        let client = MetalWorkerClient::with_owner(
            job_tx,
            Arc::new(Semaphore::new(1)),
            Arc::new(AtomicBool::new(false)),
            owner.clone(),
        );
        drop(owner);

        let (drop_done_tx, drop_done_rx) = std::sync::mpsc::sync_channel(1);
        let drop_thread = std::thread::spawn(move || {
            drop(client);
            let _ = drop_done_tx.send(());
        });
        let returned_before_watchdog = drop_done_rx
            .recv_timeout(Duration::from_millis(500))
            .is_ok();

        release_worker_tx
            .send(())
            .expect("detached worker must still accept the cleanup release");
        worker_done_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("detached worker must exit after the cleanup release");
        drop_thread
            .join()
            .expect("timed-out client drop thread must not panic");
        assert!(
            returned_before_watchdog,
            "last client Drop must honor its configured deadline instead of joining a stuck worker"
        );
    }

    #[test]
    fn owner_shutdown_reports_worker_panic_after_join() {
        let join_handle = std::thread::spawn(move || {
            panic!("simulated worker panic");
        });
        let owner = test_owner(join_handle, Duration::from_secs(1));

        assert_eq!(
            owner._inner.wait_for_exit(Duration::from_secs(1)),
            WorkerShutdown::Panicked
        );
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
            || -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String> {
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
            || -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String> {
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
