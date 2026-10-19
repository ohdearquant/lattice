//! Thread-affine serving worker host for the serve layer (issue #832, ADR-080
//! cluster C2/C3): the single dedicated thread that owns a `!Send` runtime
//! for the whole process lifetime, replacing the two previously-independent
//! copies of this loop --
//! `lattice.rs`'s `MetalJob`/`MetalHandle` and `lattice_serve.rs`'s
//! `Job`/`spawn_worker`/`run_worker_loop`.
//!
//! # Why this existed twice
//!
//! Both binaries implement the same lifecycle: load the model on a
//! dedicated OS thread (the runtime state can never cross a thread boundary),
//! serve `Job`s FIFO from an unbounded channel, check a per-job
//! disconnect-cancellation signal before paying for any prefill work, reuse
//! the single process-wide [`crate::kv_cache::CrossTurnSlotId::DEFAULT`] cache slot, and
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
//! Every production [`WorkerClient`](crate::serving_worker::WorkerClient) retains a clone of
//! [`WorkerOwner`](crate::serving_worker::WorkerOwner). The client's `Drop` implementation explicitly closes
//! its job sender before automatic field destruction can drop the owner, so
//! the sequence does not depend on field declaration order. The last owner
//! then transfers the worker's join handle to a detached reaper. The owner
//! waits up to two seconds for the reaper's result, so the deadline covers the
//! full join, including thread-local destructors. A backend call or destructor
//! that does not return therefore cannot hang process shutdown indefinitely;
//! on timeout the reaper remains detached and the process remains free to exit.
//!
//! # Testability without a GPU
//!
//! [`run_worker_loop`] -- the FIFO/cancellation/terminal-event state
//! machine -- is generic over an injected `generate` closure, exactly like
//! `lattice_serve.rs`'s pre-existing `run_worker_loop` was. Production
//! supplies callbacks from a model-owned runtime; this module's own tests
//! inject a fake generator instead, so the state machine is covered without
//! a Metal device.
//! `AffineWorker::spawn`'s `loader` failure path is also GPU-free: a loader
//! that returns `Err` before ever constructing a `MetalQwen35State`
//! typechecks and runs with no device involved. The real `spawn` -> real
//! `generate` success path has no equivalent GPU-free test (mirrors
//! precedent: PR #666's `MetalHandle` wiring shipped without a call-site
//! test requiring a real Q4 checkpoint fixture that doesn't exist, relying
//! on `metal_qwen35.rs`'s own exhaustive Device-gated tests for the
//! underlying `generate_streaming_with_prefix_cache_and_cancel` call).

use crate::forward::metal_qwen35::ChatMessage;
#[cfg(any(test, feature = "test-utils"))]
use crate::forward::metal_qwen35::format_chat_template;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
use crate::forward::metal_qwen35::{LoraLayerData, MetalQwen35State};
use crate::generation::{GenerateConfig, GenerateOutput};
use crate::serve::ApiError;
#[cfg(any(test, feature = "test-utils"))]
use crate::serve::lora::AdapterMetadata;
use crate::serve::lora::{
    AdapterControlError, AdapterControlResult, AdapterIndex, LoraSelection, ResidencyLimits,
};
use crate::serve::prepare::PreparationHandle;
#[cfg(any(test, feature = "test-utils"))]
use crate::serve::prompt_adapter::apply_qwen_thinking_mode;
use crate::serving_factory::ServingFactory;
use crate::serving_runtime_contract::WorkerFailure;
pub use crate::serving_runtime_contract::{ContextWindowPolicy, WorkerMetadata};
#[cfg(any(test, feature = "test-utils"))]
use crate::tokenizer::Tokenizer as _;
#[cfg(any(
    test,
    feature = "test-utils",
    all(target_os = "macos", feature = "metal-gpu")
))]
use crate::tokenizer::bpe::BpeTokenizer;
use std::cell::RefCell;
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot, watch};

/// Default cap on outstanding (queued + in-flight) jobs a [`WorkerClient`]
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
pub(crate) enum WorkerShutdown {
    Joined,
    AlreadyStopped,
    TimedOut,
    Panicked,
    ReaperUnavailable,
}

/// One token-stream event from the worker back to a request handler.
/// Replaces `lattice.rs`'s oneshot-reply `MetalJob` contract and
/// `lattice_serve.rs`'s private `Ev` enum with a single shared shape.
#[derive(Debug)]
#[non_exhaustive]
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

/// Worker startup failure: either the `loader` itself returned `Err`
/// (model/tokenizer load failure), the worker thread exited/panicked
/// before ever sending a readiness signal, or the requested admission cap
/// (issue #939) was outside `Semaphore::new`'s valid range.
#[derive(Debug)]
#[non_exhaustive]
pub enum StartupError {
    Load(String),
    ThreadExited,
    /// Both resident limits must be positive before the worker can start.
    InvalidResidencyLimits {
        limits: ResidencyLimits,
    },
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
            StartupError::InvalidResidencyLimits { limits } => write!(
                f,
                "resident adapter limits must be positive (count {}, bytes {})",
                limits.max_adapters, limits.max_bytes
            ),
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
    pub(crate) messages: Vec<ChatMessage>,
    pub(crate) cfg: GenerateConfig,
    pub(crate) lora: Vec<LoraSelection>,
    pub(crate) explicitly_thinking: bool,
    /// Whether the submitting HTTP request is streamed. Generation does not
    /// depend on it; a runtime that records its route per request names the
    /// mode there.
    pub(crate) stream: bool,
    pub(crate) tx: mpsc::UnboundedSender<WorkerEvent>,
    pub(crate) cancel: watch::Receiver<bool>,
    /// Admission slot for this job (issue #932), held from
    /// [`WorkerClient::submit`] until `run_worker_loop` finishes with
    /// this job (whatever the outcome — `Complete`, `Rejected`, `Failed`, or
    /// a dequeue-time `Cancelled`) and drops it, exactly once, via ordinary
    /// struct-field `Drop` — never released early, never released twice,
    /// and never forgotten on any of those paths because nothing in
    /// `run_worker_loop` ever moves it out of `job` or calls
    /// `mem::forget`/`mem::drop` on it directly. The leading underscore
    /// silences "field is never read" (this field's only job is to exist
    /// and be dropped) without needing `#[allow(dead_code)]`.
    pub(crate) _admission_permit: OwnedSemaphorePermit,
}

/// One resident-adapter mutation, applied between whole generation jobs.
pub enum AdapterCommand {
    /// Make weights resident without applying them to generation.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    Load {
        name: String,
        path: String,
        layers: Vec<LoraLayerData>,
        descriptor: Box<lattice_fann::lora::LoraDescriptor>,
    },
    /// Remove a resident id, clearing the applied slot if it uses this id.
    Unload { id: u32 },
}

impl std::fmt::Debug for AdapterCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `LoraLayerData` carries the adapter weights; printing them would
        // dump megabytes into a log line.
        match self {
            #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            Self::Load {
                layers,
                descriptor,
                name,
                path,
            } => f
                .debug_struct("Load")
                .field("layers", &layers.len())
                .field("rank", &descriptor.rank)
                .field("name", name)
                .field("path", path)
                .finish(),
            Self::Unload { id } => f.debug_struct("Unload").field("id", id).finish(),
        }
    }
}

/// An [`AdapterCommand`] in flight toward the worker thread, with the channel
/// its outcome comes back on.
///
/// The permit is a field of this struct and of [`WorkerJob`], and the two come
/// from different semaphores: a control request can never consume a generation
/// admission slot, and a generation can never consume the single control slot,
/// because neither type can be constructed with the other's permit.
pub struct ControlRequest {
    pub(crate) command: AdapterCommand,
    pub(crate) reply: oneshot::Sender<Result<AdapterControlResult, AdapterControlError>>,
    /// Held from [`WorkerClient::submit_adapter_command`] until the
    /// worker has finished applying this command and dropped it. Its only job
    /// is to exist and be dropped, like `WorkerJob::_admission_permit`.
    pub(crate) _control_permit: OwnedSemaphorePermit,
}

impl std::fmt::Debug for ControlRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ControlRequest")
            .field("command", &self.command)
            .finish_non_exhaustive()
    }
}

/// What the worker thread's single channel carries.
///
/// One channel rather than two is what makes the ordering contract in
/// [`run_worker_loop_with_control`]'s doc comment expressible at all: a second
/// channel would need a select, and the worker thread has no runtime to select
/// on -- it is a bare `std::thread` blocking on one receiver.
///
/// Without Metal, adapter commands carry no weights, so a control request is far
/// smaller than a job. Boxing the job would add an allocation to every request
/// to even out a variant that only exists to carry a refusal on those builds.
#[cfg_attr(
    not(all(target_os = "macos", feature = "metal-gpu")),
    expect(clippy::large_enum_variant)
)]
pub enum WorkerMessage {
    Generate(WorkerJob),
    Control(ControlRequest),
}

impl std::fmt::Debug for WorkerMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `WorkerJob` is deliberately not `Debug`: it holds the caller's
        // rendered messages. Name the variant and stop.
        match self {
            Self::Generate(_) => f.write_str("Generate(..)"),
            Self::Control(request) => f.debug_tuple("Control").field(request).finish(),
        }
    }
}

/// Shared owner for the dedicated worker thread.
///
/// Production [`WorkerClient`] values retain an owner clone. Each
/// client's `Drop` explicitly releases its queue sender before automatic
/// field destruction can release that owner clone. The last owner's `Drop`
/// is the sole production shutdown trigger; there is no explicit method that
/// can detach the join handle while a client still keeps the queue open.
#[derive(Debug, Clone)]
pub struct WorkerOwner {
    pub(crate) _inner: Arc<WorkerOwnerInner>,
}

#[derive(Debug)]
pub(crate) struct WorkerOwnerInner {
    join_handle: Mutex<Option<std::thread::JoinHandle<()>>>,
    drop_timeout: Duration,
}

pub(crate) fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl WorkerOwnerInner {
    pub(crate) fn wait_for_exit(&self, timeout: Duration) -> WorkerShutdown {
        let handle = {
            let mut join_handle = lock_unpoisoned(&self.join_handle);
            let Some(handle) = join_handle.take() else {
                return WorkerShutdown::AlreadyStopped;
            };
            handle
        };

        let started = Instant::now();
        let (joined_tx, joined_rx) = std::sync::mpsc::sync_channel(1);
        let reaper = std::thread::Builder::new()
            .name("lattice-metal-worker-reaper".to_string())
            .spawn(move || {
                let _ = joined_tx.send(handle.join());
            });
        let Ok(reaper) = reaper else {
            return WorkerShutdown::ReaperUnavailable;
        };
        // Joining this helper would recreate the TLS-destructor tail that the
        // result channel exists to keep outside the owner's deadline.
        drop(reaper);

        let remaining = timeout.saturating_sub(started.elapsed());
        match joined_rx.recv_timeout(remaining) {
            Ok(Ok(())) => WorkerShutdown::Joined,
            Ok(Err(_)) => WorkerShutdown::Panicked,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => WorkerShutdown::TimedOut,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                WorkerShutdown::ReaperUnavailable
            }
        }
    }
}

impl Drop for WorkerOwnerInner {
    fn drop(&mut self) {
        match self.wait_for_exit(self.drop_timeout) {
            WorkerShutdown::Joined | WorkerShutdown::AlreadyStopped => {}
            WorkerShutdown::TimedOut => {
                let _ = writeln!(
                    std::io::stderr().lock(),
                    "[metal-worker] shutdown timed out after {} ms; detaching worker join reaper",
                    self.drop_timeout.as_millis()
                );
            }
            WorkerShutdown::Panicked => {
                let _ = writeln!(
                    std::io::stderr().lock(),
                    "[metal-worker] worker thread panicked during shutdown"
                );
            }
            WorkerShutdown::ReaperUnavailable => {
                let _ = writeln!(
                    std::io::stderr().lock(),
                    "[metal-worker] worker join reaper unavailable; detaching worker thread"
                );
            }
        }
    }
}

impl WorkerOwner {
    pub(crate) fn from_handle(join_handle: std::thread::JoinHandle<()>) -> Self {
        Self::from_handle_with_timeout(join_handle, WORKER_SHUTDOWN_TIMEOUT)
    }

    pub(crate) fn from_handle_with_timeout(
        join_handle: std::thread::JoinHandle<()>,
        drop_timeout: Duration,
    ) -> Self {
        Self {
            _inner: Arc::new(WorkerOwnerInner {
                join_handle: Mutex::new(Some(join_handle)),
                drop_timeout,
            }),
        }
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fn unattached_for_test() -> Self {
        Self {
            _inner: Arc::new(WorkerOwnerInner {
                join_handle: Mutex::new(None),
                drop_timeout: WORKER_SHUTDOWN_TIMEOUT,
            }),
        }
    }
}

/// Cheaply `Clone` (an `mpsc` sender) handle used to submit generation
/// requests to the worker thread. `Send + Sync` so it lives in a binary's
/// `AppState` the same way the CPU backend's `Arc<Qwen35Model>` does --
/// only the underlying thread-affine runtime inside the worker thread stays
/// confined to that thread.
#[derive(Debug, Clone)]
pub struct WorkerClient {
    jobs: Option<mpsc::UnboundedSender<WorkerMessage>>,
    /// Bounded-admission cap (issue #932): `Semaphore::new(max_pending)`, one
    /// permit per outstanding job (queued + in-flight, i.e. from `submit`
    /// until `run_worker_loop` is done with it). `Arc`-shared with every
    /// clone of this client so the cap is process-wide, not per-clone.
    pub(crate) admission: Arc<Semaphore>,
    /// Single-permit gate on adapter control requests, separate from
    /// `admission` so a control request can never consume a generation slot
    /// and vice versa. One permit, not a queue: a second concurrent control
    /// request is refused outright rather than enqueued, because two
    /// load/unload commands in flight have no meaningful order to a caller
    /// that has not yet seen the first one's result.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    control: Arc<Semaphore>,
    adapters: Arc<RwLock<AdapterIndex>>,
    vision_supported: Arc<AtomicBool>,
    preparation: Option<PreparationHandle>,
    /// Keeps the worker join owner alive for exactly as long as the queue
    /// can accept jobs. Test-only clients without a worker carry an owner
    /// whose join slot is already empty.
    _owner: WorkerOwner,
}

impl WorkerClient {
    pub(crate) fn with_owner(
        jobs: mpsc::UnboundedSender<WorkerMessage>,
        admission: Arc<Semaphore>,
        vision_supported: Arc<AtomicBool>,
        owner: WorkerOwner,
    ) -> Self {
        Self {
            jobs: Some(jobs),
            admission,
            #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            control: Arc::new(Semaphore::new(1)),
            adapters: Arc::new(RwLock::new(AdapterIndex::default())),
            vision_supported,
            preparation: None,
            _owner: owner,
        }
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fn unattached_for_test(
        jobs: mpsc::UnboundedSender<WorkerMessage>,
        admission: Arc<Semaphore>,
    ) -> Self {
        Self::with_owner(
            jobs,
            admission,
            Arc::new(AtomicBool::new(false)),
            WorkerOwner::unattached_for_test(),
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
    /// `WorkerClient::submit` failure mode (worker gone, context
    /// window overflow, generation error) still flows through the
    /// zero-events-on-`rx`/`WorkerEvent::Rejected`/`WorkerEvent::Failed`
    /// contract unchanged.
    pub fn submit(
        &self,
        messages: Vec<ChatMessage>,
        gen_cfg: GenerateConfig,
        cancel: watch::Receiver<bool>,
    ) -> Result<mpsc::UnboundedReceiver<WorkerEvent>, ApiError> {
        self.submit_with_lora(messages, gen_cfg, cancel, Vec::new())
    }

    /// Read confirmed metadata without waiting for the worker queue.
    pub fn adapter_index(&self) -> AdapterIndex {
        self.adapters
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Validate the current resident index before committing an HTTP response.
    pub fn validate_lora(&self, selection: &[LoraSelection]) -> Result<(), ApiError> {
        self.adapters
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .validate(selection)
    }

    /// Submit a request with an ordered adapter mixture; empty means base model.
    pub fn submit_with_lora(
        &self,
        messages: Vec<ChatMessage>,
        gen_cfg: GenerateConfig,
        cancel: watch::Receiver<bool>,
        lora: Vec<LoraSelection>,
    ) -> Result<mpsc::UnboundedReceiver<WorkerEvent>, ApiError> {
        self.submit_with_lora_mode(messages, gen_cfg, cancel, lora, false)
    }

    /// [`Self::submit_with_lora`] for a caller that knows whether the HTTP
    /// response it will build is streamed. The worker hands the flag to the
    /// runtime, which names the mode in its per-request route line. Prompt
    /// rendering follows the effective mode in `gen_cfg`.
    pub fn submit_with_lora_mode(
        &self,
        messages: Vec<ChatMessage>,
        gen_cfg: GenerateConfig,
        cancel: watch::Receiver<bool>,
        lora: Vec<LoraSelection>,
        stream: bool,
    ) -> Result<mpsc::UnboundedReceiver<WorkerEvent>, ApiError> {
        self.submit_with_lora_mode_and_thinking(messages, gen_cfg, cancel, lora, stream, false)
    }

    /// Submit with the legacy explicitness argument.
    ///
    /// The argument remains for source compatibility. The effective thinking
    /// mode in `gen_cfg` controls prompt rendering.
    #[doc(hidden)]
    pub fn submit_with_lora_mode_and_thinking(
        &self,
        messages: Vec<ChatMessage>,
        gen_cfg: GenerateConfig,
        cancel: watch::Receiver<bool>,
        lora: Vec<LoraSelection>,
        stream: bool,
        explicitly_thinking: bool,
    ) -> Result<mpsc::UnboundedReceiver<WorkerEvent>, ApiError> {
        self.validate_lora(&lora)?;
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
            lora,
            explicitly_thinking,
            stream,
            tx,
            cancel,
            _admission_permit: permit,
        };
        // On failure `job` (including `tx` and the admission permit) is
        // simply dropped here, closing `rx` with zero events and freeing
        // the slot immediately -- see the doc comment above.
        if let Some(jobs) = self.jobs.as_ref() {
            let _ = jobs.send(WorkerMessage::Generate(job));
        }
        Ok(rx)
    }

    /// Submit one adapter state change to the worker thread and return the
    /// channel its outcome arrives on.
    ///
    /// The command is queued on the same channel as generation requests and is
    /// therefore applied strictly between whole jobs -- never mid-decode. A
    /// caller that submits this while generations are queued gets a result
    /// only after those generations have finished, which is the point: the
    /// alternative is a weight change landing under a decode that has already
    /// prefilled.
    ///
    /// Returns `Err(ApiError::ServiceUnavailable)` when another adapter
    /// command is already in flight. That is a refusal, not a queue: two
    /// unresolved load/unload commands have no order a caller can reason
    /// about, since neither has reported its result yet.
    ///
    /// Success returns confirmed resident metadata or the unloaded id. Metadata
    /// is published before replying, even if the caller disconnected. A closed receiver leaves
    /// the outcome unknown; inspect the index before retrying.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub fn submit_adapter_command(
        &self,
        command: AdapterCommand,
    ) -> Result<oneshot::Receiver<Result<AdapterControlResult, AdapterControlError>>, ApiError>
    {
        let permit =
            self.control
                .clone()
                .try_acquire_owned()
                .map_err(|_| ApiError::ServiceUnavailable {
                    message: "an adapter command is already in flight on the inference \
                              worker; wait for its result before sending another"
                        .to_string(),
                })?;
        let (reply, rx) = oneshot::channel();
        let request = ControlRequest {
            command,
            reply,
            _control_permit: permit,
        };
        // On failure `request` -- including `reply` and the control permit --
        // is dropped here, closing `rx` and freeing the slot immediately.
        if let Some(jobs) = self.jobs.as_ref() {
            let _ = jobs.send(WorkerMessage::Control(request));
        }
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

    /// Model-bound preparation, present on a successfully loaded worker.
    #[doc(hidden)]
    pub fn preparation(&self) -> Option<&PreparationHandle> {
        self.preparation.as_ref()
    }
}

impl Drop for WorkerClient {
    fn drop(&mut self) {
        drop(self.jobs.take());
    }
}

#[cfg(any(test, feature = "test-utils"))]
fn fake_check_prompt_fits_window(
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

#[cfg(any(test, feature = "test-utils"))]
fn fake_serving_tokenizer(tokenizer: BpeTokenizer, model_max_context: usize) -> BpeTokenizer {
    if tokenizer.max_seq_len() < model_max_context {
        tokenizer.with_max_seq_len(model_max_context)
    } else {
        tokenizer
    }
}

#[cfg(any(test, feature = "test-utils"))]
fn fake_render_text_prompt_within_window(
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
    fake_check_prompt_fits_window(policy, model_max_context, prompt_len, cfg)?;
    Ok((prompt, prompt_len))
}

/// Dequeue -> cancel-check -> generate -> reply, serialized on whatever
/// thread calls this (the dedicated serving worker thread in production; a
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
///
/// Adapter control messages ([`WorkerMessage::Control`]) share that one FIFO
/// queue and occupy a whole position in it. A control message is therefore
/// applied strictly BETWEEN two jobs: after the terminal event of every job
/// dequeued before it, and before any prompt work of every job dequeued after
/// it. It never interleaves with a job's prefill or decode, because the single
/// blocking receive is the only place this loop ever looks at the channel.
/// That is the property the one-channel design buys, and it is not an
/// incidental consequence of the current body -- moving control handling
/// anywhere that can run while `generate` is on the stack breaks it.
///
/// A control message produces no [`WorkerEvent`]. Its result goes back on the
/// request's own reply channel, so the per-job "exactly one terminal event"
/// contract above is unchanged in both directions: control messages add none,
/// and consume none.
///
/// Production no longer calls this form: [`AffineWorker::spawn_with_vision`]
/// owns a model and therefore has a real control handler to inject. What is
/// left is the fake-worker seam and this module's own tests, which is why the
/// two-argument form is gated to the same configurations those are. Widening
/// that gate would mean claiming a production caller that does not exist.
#[cfg(any(test, feature = "test-utils"))]
pub(crate) fn run_worker_loop(
    msg_rx: mpsc::UnboundedReceiver<WorkerMessage>,
    generate: impl FnMut(
        &[ChatMessage],
        &GenerateConfig,
        bool,
        &mut dyn FnMut(&str, u32) -> bool,
        &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure>,
) {
    run_worker_loop_with_control(msg_rx, generate, control_unsupported);
}

/// The error a worker built without adapter support returns for any control
/// message. Written as a named function rather than a closure at each call
/// site so every such worker returns the same string.
#[cfg(any(
    test,
    feature = "test-utils",
    not(all(target_os = "macos", feature = "metal-gpu"))
))]
pub(crate) fn control_unsupported(
    _command: AdapterCommand,
) -> Result<AdapterControlResult, AdapterControlError> {
    Err("this inference worker does not support adapter commands".into())
}

/// [`run_worker_loop`] with an injected adapter-control handler.
///
/// One loop body, not two. The generation path below is byte-identical to what
/// [`run_worker_loop`] ran before control messages existed; the two-argument
/// form is now a call into this one. This module's own header is about two
/// copies of this loop that drifted apart while only comments kept them in
/// sync, so a second copy is the one thing this change must not add.
///
/// `control` runs on the worker thread with exclusive access to whatever state
/// that thread owns, at a point where no generation is in flight. It returns
/// a typed failure so HTTP callers can distinguish capacity from invalid input.
#[cfg(any(test, feature = "test-utils"))]
pub(crate) fn run_worker_loop_with_control(
    msg_rx: mpsc::UnboundedReceiver<WorkerMessage>,
    mut generate: impl FnMut(
        &[ChatMessage],
        &GenerateConfig,
        bool,
        &mut dyn FnMut(&str, u32) -> bool,
        &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure>,
    control: impl FnMut(AdapterCommand) -> Result<AdapterControlResult, AdapterControlError>,
) {
    run_worker_loop_with_lora(
        msg_rx,
        move |messages, cfg, _lora, _stream, explicitly_thinking, on_token, cancel| {
            generate(messages, cfg, explicitly_thinking, on_token, cancel)
        },
        control,
    );
}

pub(crate) fn run_worker_loop_with_lora(
    mut msg_rx: mpsc::UnboundedReceiver<WorkerMessage>,
    mut generate: impl FnMut(
        &[ChatMessage],
        &GenerateConfig,
        &[LoraSelection],
        bool,
        bool,
        &mut dyn FnMut(&str, u32) -> bool,
        &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure>,
    mut control: impl FnMut(AdapterCommand) -> Result<AdapterControlResult, AdapterControlError>,
) {
    while let Some(message) = msg_rx.blocking_recv() {
        let job = match message {
            WorkerMessage::Generate(job) => job,
            WorkerMessage::Control(request) => {
                let ControlRequest {
                    command,
                    reply,
                    _control_permit,
                } = request;
                let outcome = control(command);
                // The caller may have dropped its receiver; the command was
                // still applied, and there is nobody to tell. Dropping
                // `_control_permit` here frees the control slot for the next
                // request, in the same statement sequence either way.
                let _ = reply.send(outcome);
                continue;
            }
        };

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

        match generate(
            &job.messages,
            &job.cfg,
            &job.lora,
            job.stream,
            job.explicitly_thinking,
            &mut on_token,
            &mut should_cancel,
        ) {
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

/// Namespace for [`AffineWorker::spawn`] -- a zero-sized marker type (never
/// constructed) so the shared worker's entry point reads as
/// `AffineWorker::spawn(..)` at every call site, matching the association
/// `lattice.rs`'s prior `MetalHandle::spawn` and `lattice_serve.rs`'s prior
/// `spawn_worker` free function both had with "the serving worker".
pub struct AffineWorker;

pub(crate) fn run_guarded_worker<G>(acquire_guard: impl FnOnce() -> G, work: impl FnOnce()) {
    let _measurement_guard = acquire_guard();
    work();
}

impl AffineWorker {
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
    /// ready, and never gets a `WorkerClient` it could submit jobs to
    /// before that point either.
    ///
    /// `max_pending` (issue #932) is the returned `WorkerClient`'s
    /// outstanding-job admission cap -- see [`WorkerClient::submit`].
    /// Both binaries pass their own `--max-pending`-derived value (default
    /// [`DEFAULT_MAX_PENDING_JOBS`]); this function applies no default of
    /// its own. `residency_limits` bounds explicit adapter loads independently
    /// of request admission; both its fields must be positive.
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    pub fn spawn(
        loader: impl FnOnce() -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String>
        + Send
        + 'static,
        max_pending: usize,
        residency_limits: ResidencyLimits,
    ) -> Result<(WorkerOwner, WorkerClient, WorkerMetadata), StartupError> {
        let (owner, client, metadata, _preparation) = Self::spawn_with_vision(
            ServingFactory::legacy_qwen(loader),
            max_pending,
            residency_limits,
        )?;
        Ok((owner, client, metadata))
    }

    /// Spawn from a one-shot serving model factory. Its state is built on the
    /// dedicated worker thread after the spawn.
    pub fn spawn_with_vision(
        factory: ServingFactory,
        max_pending: usize,
        residency_limits: ResidencyLimits,
    ) -> Result<(WorkerOwner, WorkerClient, WorkerMetadata, PreparationHandle), StartupError> {
        Self::spawn_with_guard(factory, max_pending, residency_limits, || ())
    }

    pub(crate) fn spawn_with_guard<G>(
        factory: ServingFactory,
        max_pending: usize,
        residency_limits: ResidencyLimits,
        acquire_guard: impl FnOnce() -> G + Send + 'static,
    ) -> Result<(WorkerOwner, WorkerClient, WorkerMetadata, PreparationHandle), StartupError> {
        if residency_limits.max_adapters == 0 || residency_limits.max_bytes == 0 {
            return Err(StartupError::InvalidResidencyLimits {
                limits: residency_limits,
            });
        }
        // #939: validate BEFORE `Semaphore::new`, which panics outright for
        // `max_pending > Semaphore::MAX_PERMITS` and would otherwise let
        // `max_pending == 0` silently build a worker that admits nothing.
        if max_pending == 0 || max_pending > Semaphore::MAX_PERMITS {
            return Err(StartupError::InvalidMaxPending { max_pending });
        }
        let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerMessage>();
        let admission = Arc::new(Semaphore::new(max_pending));
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<
            Result<(WorkerMetadata, PreparationHandle, Arc<AtomicBool>), String>,
        >();

        let adapters = Arc::new(RwLock::new(AdapterIndex::default()));
        let worker_index = Arc::clone(&adapters);
        let join_handle = std::thread::spawn(move || {
            run_guarded_worker(acquire_guard, move || {
                match factory.build(worker_index, residency_limits) {
                    Ok((runtime, metadata, preparation)) => {
                        let vision_supported = runtime.vision_supported();
                        let _ = ready_tx.send(Ok((metadata, preparation, vision_supported)));
                        // The non-Send runtime was built after the spawn and never
                        // crosses back. The loop lends it to one callback at a
                        // time, so these mutable borrows cannot overlap.
                        let runtime = RefCell::new(runtime);
                        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
                        let control = |command| runtime.borrow_mut().control(command);
                        #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
                        let control = control_unsupported;
                        run_worker_loop_with_lora(
                            job_rx,
                            |messages,
                             cfg,
                             lora,
                             stream,
                             explicitly_thinking,
                             on_token,
                             should_cancel| {
                                runtime.borrow_mut().generate(
                                    messages,
                                    cfg,
                                    lora,
                                    stream,
                                    explicitly_thinking,
                                    on_token,
                                    should_cancel,
                                )
                            },
                            control,
                        );
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                    }
                }
            });
        });
        let owner = WorkerOwner::from_handle(join_handle);
        match ready_rx.recv() {
            Ok(Ok((meta, preparation, vision_supported))) => {
                let mut client =
                    WorkerClient::with_owner(job_tx, admission, vision_supported, owner.clone());
                client.adapters = adapters;
                client.preparation = Some(preparation.clone());
                Ok((owner, client, meta, preparation))
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
    /// Selection received by this queued job, for HTTP-to-worker contract tests.
    pub fn lora_selection(&self) -> &[LoraSelection] {
        &self.lora
    }

    /// Whether the submitting request was marked streamed, for
    /// HTTP-to-worker contract tests.
    pub fn is_streamed(&self) -> bool {
        self.stream
    }

    /// Reply to this job with one event, exactly as the production worker
    /// loop would via its own `job.tx.send(..)`. Returns `false` once the
    /// submitting caller's event receiver is gone. Test-only: production
    /// code always routes replies through [`run_worker_loop`], never
    /// directly.
    pub fn reply(&self, event: WorkerEvent) -> bool {
        self.tx.send(event).is_ok()
    }
}

/// A [`WorkerClient`] wired to a plain, unattached job receiver, for
/// tests that want to fully control every reply by hand (a fake worker
/// task/thread, or none at all -- see [`WorkerJob::reply`]). Mirrors
/// `lattice_serve.rs`'s pre-existing `test_app_state_with_jobs` helper,
/// generalized so both binaries' test suites build on one shared seam
/// instead of each rolling its own raw `mpsc::unbounded_channel::<Job>()`
/// pair.
#[cfg(any(test, feature = "test-utils"))]
pub fn test_client_and_jobs() -> (WorkerClient, mpsc::UnboundedReceiver<WorkerMessage>) {
    // A large, effectively-unbounded cap: the overwhelming majority of
    // existing callers of this seam predate the #932 admission cap and
    // exercise request validation / routing / cancellation, not admission
    // itself -- they must keep behaving as if the queue were unbounded.
    // Tests that specifically exercise the cap use
    // `test_client_and_jobs_with_cap` instead.
    test_client_and_jobs_with_cap(TEST_EFFECTIVELY_UNBOUNDED_CAP)
}

/// An unattached client with a known metadata snapshot for request routing tests.
#[cfg(any(test, feature = "test-utils"))]
pub fn test_client_and_jobs_with_adapters(
    adapters: Vec<AdapterMetadata>,
) -> (WorkerClient, mpsc::UnboundedReceiver<WorkerMessage>) {
    let (client, jobs) = test_client_and_jobs();
    *client
        .adapters
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = AdapterIndex {
        adapters,
        applied: Vec::new(),
        ..Default::default()
    };
    (client, jobs)
}

/// Same as [`test_client_and_jobs`], with an explicit admission cap (issue
/// #932) instead of the effectively-unbounded default -- for tests that
/// exercise `WorkerClient::submit`'s admission rejection itself.
#[cfg(any(test, feature = "test-utils"))]
pub fn test_client_and_jobs_with_cap(
    max_pending: usize,
) -> (WorkerClient, mpsc::UnboundedReceiver<WorkerMessage>) {
    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    (
        WorkerClient::unattached_for_test(job_tx, Arc::new(Semaphore::new(max_pending))),
        job_rx,
    )
}

/// See [`test_client_and_jobs`]'s doc comment: the cap
/// `test_client_and_jobs`/`spawn_fake` (the two test-utils seams that predate
/// issue #932) use so pre-existing callers keep seeing effectively-unbounded
/// admission unless they opt into the `_with_cap` variant.
#[cfg(any(test, feature = "test-utils"))]
const TEST_EFFECTIVELY_UNBOUNDED_CAP: usize = 1_000_000;

/// A [`WorkerClient`] backed by a REAL background thread running the
/// exact production FIFO/cancellation loop ([`run_worker_loop`]) and the
/// exact production context-window invariant (real
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
) -> WorkerClient {
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
/// `WorkerClient::submit`'s admission rejection at the real-router
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
) -> WorkerClient {
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
) -> WorkerClient {
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
) -> WorkerClient {
    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerMessage>();
    let tokenizer = fake_serving_tokenizer(tokenizer, model_max_context);
    let join_handle = std::thread::spawn(move || {
        run_worker_loop(
            job_rx,
            move |messages, cfg, explicitly_thinking, on_token, should_cancel| {
                let (prompt, _prompt_len) = fake_render_text_prompt_within_window(
                    &tokenizer,
                    messages,
                    context_window_policy,
                    model_max_context,
                    cfg,
                    explicitly_thinking,
                )
                .map_err(WorkerFailure::Rejected)?;
                // The production worker hands generation the rendered prompt and
                // the worker's tokenizer, which re-tokenizes it; report the length
                // that tokenization keeps, so a tokenizer cap below the window is
                // visible here as it is on the real path.
                let prompt_tokens = tokenizer.tokenize(&prompt).real_length;
                generate(messages, cfg, prompt_tokens, on_token, should_cancel)
                    .map_err(WorkerFailure::Failed)
            },
        );
    });
    let owner = WorkerOwner::from_handle(join_handle);
    WorkerClient::with_owner(
        job_tx,
        Arc::new(Semaphore::new(max_pending)),
        Arc::new(AtomicBool::new(vision_supported)),
        owner,
    )
}
