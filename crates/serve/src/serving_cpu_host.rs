use crate::serve::ApiError;
use crate::serve::contract::ChatRequest;
use crate::serve::prepare::PreparationHandle;
use crate::serving_host::ServedOutput;
use crate::serving_preparation::PreparedCpuChat;
use crate::serving_runtime::{ServingRuntime, WorkerFailure, score_rows_within};
use crate::serving_runtime_contract::{RuntimeInput, TextGenerationEntry};
use crate::serving_worker::{ScoreReply, ScoreRow};
use lattice_inference::generation::{GenerateConfig, GenerateOutput};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

pub(crate) trait CpuRuntimeSource: Send + Sync + 'static {
    fn create_runtime(&self) -> Box<dyn ServingRuntime + '_>;
}

/// Shared CPU resources and provider-owned request preparation.
#[doc(hidden)]
#[derive(Clone)]
pub struct SharedCpuHandle {
    source: Arc<dyn CpuRuntimeSource>,
    preparation: PreparationHandle,
    pools: Option<Arc<CpuPools>>,
}

impl SharedCpuHandle {
    pub(crate) fn new(source: Arc<dyn CpuRuntimeSource>, preparation: PreparationHandle) -> Self {
        Self {
            source,
            preparation,
            pools: None,
        }
    }

    /// Run each request of this handle alone on a rayon pool of `threads`
    /// threads instead of the global pool, so one request's parallel work uses
    /// at most `threads` threads and never shares a pool with another request.
    /// A request takes an idle pool, or a new one when none is idle, and gives
    /// it back before its caller can admit the next request, so the number of
    /// pools is the number of requests the caller runs at a time: `threads`
    /// times that many threads in all. The first pool is started here.
    ///
    /// Nothing here caps the number of pools; the caller's admission does.
    /// Every request that reaches this handle through a [`ServingHandle`]
    /// holds a CPU admission permit (`--max-cpu-requests`) for as long as
    /// it holds a pool, so the pools never outnumber that cap. A caller that
    /// spawns requests on this handle directly must bound them itself, or each
    /// request running at once starts another pool.
    ///
    /// [`ServingHandle`]: crate::serving_host::ServingHandle
    ///
    /// # Errors
    ///
    /// When the first pool's threads cannot be started.
    pub fn with_cpu_threads(
        mut self,
        threads: NonZeroUsize,
    ) -> Result<Self, rayon::ThreadPoolBuildError> {
        let pools = CpuPools {
            threads,
            idle: Mutex::new(Vec::new()),
            built: AtomicUsize::new(0),
        };
        let first = pools.build()?;
        pools.give_back(Arc::new(first));
        self.pools = Some(Arc::new(pools));
        Ok(self)
    }

    /// Prepare the request on the HTTP task with the selected provider.
    pub fn prepare(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
    ) -> Result<PreparedCpuChat, ApiError> {
        self.preparation
            .prepare_cpu(req, model_id, default_max_tokens, max_tokens_cap)
    }

    /// Provider preparation associated with this loaded CPU source.
    pub fn preparation(&self) -> &PreparationHandle {
        &self.preparation
    }

    /// Run one complete CPU request on a blocking task. The output carries the reply's
    /// reasoning text when it has one.
    pub fn spawn_completion(
        &self,
        prompt: String,
        config: GenerateConfig,
    ) -> tokio::task::JoinHandle<Result<ServedOutput, String>> {
        let source = Arc::clone(&self.source);
        let pools = self.pools.clone();
        tokio::task::spawn_blocking(move || {
            run_in(pools.as_ref(), move || {
                let mut runtime = source.create_runtime();
                runtime
                    .execute_with_reasoning(
                        RuntimeInput::PreparedText {
                            prompt: &prompt,
                            entry: TextGenerationEntry::Complete,
                        },
                        &config,
                        false,
                        &mut |_, _| true,
                        &mut |_| true,
                        &mut || false,
                    )
                    .map_err(failure_message)
            })?
        })
    }

    /// Run one streaming CPU request on a blocking task. `on_delta` receives the reply's
    /// content and `on_reasoning` its reasoning, in generated order, both from that task.
    pub fn spawn_streaming(
        &self,
        prompt: String,
        config: GenerateConfig,
        cancel: tokio::sync::watch::Receiver<bool>,
        mut on_delta: impl FnMut(&str) -> bool + Send + 'static,
        mut on_reasoning: impl FnMut(&str) -> bool + Send + 'static,
        finish: impl FnOnce(Result<GenerateOutput, String>) + Send + 'static,
    ) {
        let source = Arc::clone(&self.source);
        let pools = self.pools.clone();
        tokio::task::spawn_blocking(move || {
            let outcome = run_in(pools.as_ref(), move || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut runtime = source.create_runtime();
                    let mut on_token = |delta: &str, _token_id| on_delta(delta);
                    let mut should_cancel = || *cancel.borrow();
                    let result = runtime
                        .execute_with_reasoning(
                            RuntimeInput::PreparedText {
                                prompt: &prompt,
                                entry: TextGenerationEntry::StreamingWithCancel,
                            },
                            &config,
                            true,
                            &mut on_token,
                            &mut on_reasoning,
                            &mut should_cancel,
                        )
                        .map(|served| served.output)
                        .map_err(failure_message);
                    drop(runtime);
                    result
                }))
            });
            let (result, panic_payload) = match outcome {
                Ok(Ok(result)) => (result, None),
                Ok(Err(payload)) => (Err("inference failed".to_owned()), Some(payload)),
                Err(message) => (Err(message), None),
            };
            // The `spawn_blocking` JoinHandle is discarded, so a panic that
            // unwound out of the closure passed to `catch_unwind` would end
            // this task before `finish` ran, and the client would never
            // receive the failure event; `catch_unwind` turns that unwind into
            // the `Err` arm above so `finish` still runs. `finish` stays
            // outside the catch because a panicking token callback may already
            // have sent its terminal event. It runs after the pool is given
            // back, because the caller's admission permit may end inside it.
            finish(result);
            // A custom panic payload can itself panic when dropped.
            drop(panic_payload);
        });
    }

    /// Score option rows on a blocking task, holding `permit` until the rows
    /// are scored or scoring stops. `time_limit` counts from when the task
    /// starts scoring. A panic in the scorer drops the sender, so the receiver
    /// reports a closed channel instead of an answer.
    pub(crate) fn spawn_score(
        &self,
        rows: Vec<ScoreRow>,
        time_limit: Option<std::time::Duration>,
        permit: tokio::sync::OwnedSemaphorePermit,
    ) -> tokio::sync::oneshot::Receiver<ScoreReply> {
        let (reply, receiver) = tokio::sync::oneshot::channel();
        let source = Arc::clone(&self.source);
        let pools = self.pools.clone();
        tokio::task::spawn_blocking(move || {
            let end = run_in(pools.as_ref(), || {
                let mut runtime = source.create_runtime();
                let end =
                    score_rows_within(&mut *runtime, &rows, time_limit, &mut || reply.is_closed());
                drop(runtime);
                end
            });
            // The pool and then the capacity are returned before the reply, so a
            // caller holding the reply can already admit the next request. With
            // no pool to run on, the reply is dropped unsent.
            drop(permit);
            if let Some(outcome) = end.ok().and_then(|end| end.into_reply()) {
                let _ = reply.send(outcome);
            }
        });
        receiver
    }

    /// Build a CPU handle around a tiny Qwen model for binary unit tests.
    #[cfg(feature = "test-utils")]
    pub fn from_qwen_model_for_test(model: lattice_inference::model::qwen35::Qwen35Model) -> Self {
        crate::serving_provider::providers::qwen::cpu::from_model(model)
    }

    /// Replace numerical execution while retaining provider preparation in binary tests.
    #[cfg(feature = "test-utils")]
    pub fn with_qwen_generator_for_test<F>(
        model: Arc<lattice_inference::model::qwen35::Qwen35Model>,
        generate: F,
    ) -> Self
    where
        F: Fn(
                &str,
                &GenerateConfig,
                &mut dyn FnMut(&str) -> bool,
                &mut dyn FnMut() -> bool,
            ) -> Result<GenerateOutput, lattice_inference::error::InferenceError>
            + Send
            + Sync
            + 'static,
    {
        let preparation =
            crate::serving_provider::providers::qwen::cpu::preparation(Arc::clone(&model));
        let source: Arc<dyn CpuRuntimeSource> = Arc::new(TestCpuSource {
            generate: Arc::new(generate),
        });
        Self::new(source, preparation)
    }

    /// As [`Self::with_qwen_generator_for_test`], with a generator that also passes reasoning
    /// pieces to its second sink and returns the reasoning text beside the output.
    #[cfg(feature = "test-utils")]
    pub fn with_reasoning_generator_for_test<F>(
        model: Arc<lattice_inference::model::qwen35::Qwen35Model>,
        generate: F,
    ) -> Self
    where
        F: Fn(
                &str,
                &GenerateConfig,
                &mut dyn FnMut(&str) -> bool,
                &mut dyn FnMut(&str) -> bool,
                &mut dyn FnMut() -> bool,
            )
                -> Result<(GenerateOutput, String), lattice_inference::error::InferenceError>
            + Send
            + Sync
            + 'static,
    {
        let preparation =
            crate::serving_provider::providers::qwen::cpu::preparation(Arc::clone(&model));
        let source: Arc<dyn CpuRuntimeSource> = Arc::new(TestReasoningSource {
            generate: Arc::new(generate),
        });
        Self::new(source, preparation)
    }
}

/// Rayon pools of one width, each running one request at a time.
struct CpuPools {
    threads: NonZeroUsize,
    idle: Mutex<Vec<Arc<rayon::ThreadPool>>>,
    built: AtomicUsize,
}

impl CpuPools {
    fn build(&self) -> Result<rayon::ThreadPool, rayon::ThreadPoolBuildError> {
        let pool = self.built.fetch_add(1, Ordering::Relaxed);
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads.get())
            .thread_name(move |index| format!("lattice-cpu-{pool}-{index}"))
            .build()
    }

    fn take(&self) -> Result<Arc<rayon::ThreadPool>, rayon::ThreadPoolBuildError> {
        let idle = self
            .idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        match idle {
            Some(pool) => Ok(pool),
            None => self.build().map(Arc::new),
        }
    }

    fn give_back(&self, pool: Arc<rayon::ThreadPool>) {
        self.idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(pool);
    }
}

/// A pool taken by one request, given back when the request ends or unwinds.
struct PoolLease<'a> {
    pools: &'a CpuPools,
    pool: Arc<rayon::ThreadPool>,
}

impl Drop for PoolLease<'_> {
    fn drop(&mut self) {
        self.pools.give_back(Arc::clone(&self.pool));
    }
}

/// Run `work` alone on a pool from `pools` when there are pools, with the
/// calling thread's tokio runtime entered on the pool thread; otherwise on the
/// calling thread, whose rayon work uses the global pool. The pool is given
/// back before this returns.
///
/// A request that installed itself on a pool shared with other requests could
/// be picked up by a thread of that pool waiting inside another request's
/// parallel work, and would then run to its end on top of that request.
///
/// Only the pool thread that runs `work` itself is inside the runtime. The
/// other threads of the pool, which run the pieces of `work`'s parallel
/// iterators and joins, are not, so code called from inside parallel work
/// cannot use `Handle::current` or spawn tokio tasks.
fn run_in<R: Send>(
    pools: Option<&Arc<CpuPools>>,
    work: impl FnOnce() -> R + Send,
) -> Result<R, String> {
    let Some(pools) = pools else {
        return Ok(work());
    };
    let pool = pools.take().map_err(|error| {
        eprintln!("CPU thread pool did not start: {error}");
        "inference failed".to_owned()
    })?;
    let lease = PoolLease { pools, pool };
    let runtime = tokio::runtime::Handle::try_current().ok();
    Ok(lease.pool.install(move || {
        let _entered = runtime.as_ref().map(tokio::runtime::Handle::enter);
        work()
    }))
}

fn failure_message(failure: WorkerFailure) -> String {
    match failure {
        WorkerFailure::Failed(message) | WorkerFailure::ConstraintBlocked(message) => message,
        WorkerFailure::Rejected(error) => error.message().to_owned(),
    }
}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SharedCpuHandle>();
};

#[cfg(feature = "test-utils")]
type TestGenerator = dyn Fn(
        &str,
        &GenerateConfig,
        &mut dyn FnMut(&str) -> bool,
        &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, lattice_inference::error::InferenceError>
    + Send
    + Sync;

#[cfg(feature = "test-utils")]
struct TestCpuSource {
    generate: Arc<TestGenerator>,
}

#[cfg(feature = "test-utils")]
impl CpuRuntimeSource for TestCpuSource {
    fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
        Box::new(TestCpuRuntime {
            generate: Arc::clone(&self.generate),
            vision: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }
}

#[cfg(feature = "test-utils")]
struct TestCpuRuntime {
    generate: Arc<TestGenerator>,
    vision: Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(feature = "test-utils")]
impl ServingRuntime for TestCpuRuntime {
    fn execute(
        &mut self,
        input: RuntimeInput<'_>,
        config: &GenerateConfig,
        _http_stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        let RuntimeInput::PreparedText { prompt, .. } = input else {
            return Err(WorkerFailure::Failed(
                "chat messages are not supported by the test CPU runtime".to_owned(),
            ));
        };
        (self.generate)(
            prompt,
            config,
            &mut |delta| on_token(delta, 0),
            should_cancel,
        )
        .map_err(|error| WorkerFailure::Failed(error.to_string()))
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn control(
        &mut self,
        _command: crate::serving_worker::AdapterCommand,
    ) -> Result<crate::serve::lora::AdapterControlResult, crate::serve::lora::AdapterControlError>
    {
        Err(crate::serve::lora::AdapterControlError::InvalidAdapter(
            "test CPU runtime does not accept adapters".to_owned(),
        ))
    }

    fn vision_supported(&self) -> Arc<std::sync::atomic::AtomicBool> {
        Arc::clone(&self.vision)
    }
}

#[cfg(feature = "test-utils")]
type TestReasoningGenerator = dyn Fn(
        &str,
        &GenerateConfig,
        &mut dyn FnMut(&str) -> bool,
        &mut dyn FnMut(&str) -> bool,
        &mut dyn FnMut() -> bool,
    ) -> Result<(GenerateOutput, String), lattice_inference::error::InferenceError>
    + Send
    + Sync;

#[cfg(feature = "test-utils")]
struct TestReasoningSource {
    generate: Arc<TestReasoningGenerator>,
}

#[cfg(feature = "test-utils")]
impl CpuRuntimeSource for TestReasoningSource {
    fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
        Box::new(TestReasoningRuntime {
            generate: Arc::clone(&self.generate),
            vision: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }
}

/// A test runtime whose replies have a reasoning field, so it overrides
/// `execute_with_reasoning` as the Inkling 2 runtime does.
#[cfg(feature = "test-utils")]
struct TestReasoningRuntime {
    generate: Arc<TestReasoningGenerator>,
    vision: Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(feature = "test-utils")]
impl ServingRuntime for TestReasoningRuntime {
    fn execute(
        &mut self,
        input: RuntimeInput<'_>,
        config: &GenerateConfig,
        http_stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        self.execute_with_reasoning(
            input,
            config,
            http_stream,
            on_token,
            &mut |_| true,
            should_cancel,
        )
        .map(|served| served.output)
    }

    fn execute_with_reasoning(
        &mut self,
        input: RuntimeInput<'_>,
        config: &GenerateConfig,
        _http_stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        on_reasoning: &mut dyn FnMut(&str) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<ServedOutput, WorkerFailure> {
        let RuntimeInput::PreparedText { prompt, .. } = input else {
            return Err(WorkerFailure::Failed(
                "chat messages are not supported by the test CPU runtime".to_owned(),
            ));
        };
        (self.generate)(
            prompt,
            config,
            &mut |delta| on_token(delta, 0),
            on_reasoning,
            should_cancel,
        )
        .map(|(output, reasoning)| ServedOutput::new(output, reasoning))
        .map_err(|error| WorkerFailure::Failed(error.to_string()))
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn control(
        &mut self,
        _command: crate::serving_worker::AdapterCommand,
    ) -> Result<crate::serve::lora::AdapterControlResult, crate::serve::lora::AdapterControlError>
    {
        Err(crate::serve::lora::AdapterControlError::InvalidAdapter(
            "test CPU runtime does not accept adapters".to_owned(),
        ))
    }

    fn vision_supported(&self) -> Arc<std::sync::atomic::AtomicBool> {
        Arc::clone(&self.vision)
    }
}

#[cfg(test)]
mod tests;
