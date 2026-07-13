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
use tokio::sync::{mpsc, watch};

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
    /// Generation failed closed instead of completing (#611: e.g. a grammar
    /// mask that blocks every candidate token). Carries the underlying
    /// error message for server-side logging.
    Failed(String),
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
}

/// Worker startup failure: either the `loader` itself returned `Err`
/// (model/tokenizer load failure), or the worker thread exited/panicked
/// before ever sending a readiness signal.
#[derive(Debug)]
pub enum StartupError {
    Load(String),
    ThreadExited,
}

impl std::fmt::Display for StartupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StartupError::Load(message) => write!(f, "{message}"),
            StartupError::ThreadExited => {
                write!(f, "worker thread exited before loading finished")
            }
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
}

impl MetalWorkerClient {
    /// Submit one generation request; the worker thread processes jobs
    /// strictly FIFO. Returns immediately with the event receiver -- if the
    /// worker thread is no longer running, the returned receiver closes
    /// with zero events (`recv()` resolves to `None` on the first poll).
    /// Callers must treat that the same as an explicit "worker unavailable"
    /// error, mirroring each binary's prior `jobs.send(..).is_err()` check.
    pub fn submit(
        &self,
        messages: Vec<ChatMessage>,
        gen_cfg: GenerateConfig,
        cancel: watch::Receiver<bool>,
    ) -> mpsc::UnboundedReceiver<WorkerEvent> {
        let (tx, rx) = mpsc::unbounded_channel();
        let job = WorkerJob {
            messages,
            cfg: gen_cfg,
            tx,
            cancel,
        };
        // On failure `job` (including `tx`) is simply dropped here, closing
        // `rx` with zero events -- see the doc comment above.
        let _ = self.jobs.send(job);
        rx
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
    pub fn spawn(
        loader: impl FnOnce() -> Result<(MetalQwen35State, BpeTokenizer, WorkerMetadata), String>
        + Send
        + 'static,
    ) -> Result<(MetalWorkerOwner, MetalWorkerClient, WorkerMetadata), StartupError> {
        let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerJob>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<WorkerMetadata, String>>();

        let join_handle = std::thread::spawn(move || match loader() {
            Ok((mut state, tokenizer, meta)) => {
                let _ = ready_tx.send(Ok(meta.clone()));
                run_worker_loop(job_rx, move |messages, cfg, on_token, should_cancel| {
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
                    cached
                        .map(|c| c.output)
                        .map_err(|e| WorkerFailure::Failed(e.to_string()))
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
                MetalWorkerClient { jobs: job_tx },
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

#[cfg(feature = "test-utils")]
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
#[cfg(feature = "test-utils")]
pub fn test_client_and_jobs() -> (MetalWorkerClient, mpsc::UnboundedReceiver<WorkerJob>) {
    let (job_tx, job_rx) = mpsc::unbounded_channel::<WorkerJob>();
    (MetalWorkerClient { jobs: job_tx }, job_rx)
}

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
#[cfg(feature = "test-utils")]
#[allow(clippy::type_complexity)]
pub fn spawn_fake(
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
    MetalWorkerClient { jobs: job_tx }
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
        let job = WorkerJob {
            messages: vec![ChatMessage::user("hi")],
            cfg: GenerateConfig::default(),
            tx,
            cancel: cancel_rx,
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
        let job = WorkerJob {
            messages: vec![ChatMessage::user("hi")],
            cfg: GenerateConfig::default(),
            tx,
            cancel: cancel_rx,
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
        let result = MetalWorker::spawn(|| Err("simulated load failure".to_string()));
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
}
