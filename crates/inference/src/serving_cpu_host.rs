use crate::generation::{GenerateConfig, GenerateOutput};
use crate::model::serving_runtime::{ServingRuntime, WorkerFailure};
use crate::serve::ApiError;
use crate::serve::contract::ChatRequest;
use crate::serve::prepare::PreparationHandle;
use crate::serving_preparation::PreparedCpuChat;
use crate::serving_runtime_contract::{RuntimeInput, TextGenerationEntry};
use std::sync::Arc;

pub(crate) trait CpuRuntimeSource: Send + Sync + 'static {
    fn create_runtime(&self) -> Box<dyn ServingRuntime + '_>;
}

/// Shared CPU resources and provider-owned request preparation.
#[doc(hidden)]
#[derive(Clone)]
pub struct SharedCpuHandle {
    source: Arc<dyn CpuRuntimeSource>,
    preparation: PreparationHandle,
}

impl SharedCpuHandle {
    pub(crate) fn new(source: Arc<dyn CpuRuntimeSource>, preparation: PreparationHandle) -> Self {
        Self {
            source,
            preparation,
        }
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

    /// Run one complete CPU request on a blocking task.
    pub fn spawn_completion(
        &self,
        prompt: String,
        config: GenerateConfig,
    ) -> tokio::task::JoinHandle<Result<GenerateOutput, String>> {
        let source = Arc::clone(&self.source);
        tokio::task::spawn_blocking(move || {
            let mut runtime = source.create_runtime();
            runtime
                .execute(
                    RuntimeInput::PreparedText {
                        prompt: &prompt,
                        entry: TextGenerationEntry::Complete,
                    },
                    &config,
                    false,
                    &mut |_, _| true,
                    &mut || false,
                )
                .map_err(failure_message)
        })
    }

    /// Run one streaming CPU request on a blocking task.
    pub fn spawn_streaming(
        &self,
        prompt: String,
        config: GenerateConfig,
        cancel: tokio::sync::watch::Receiver<bool>,
        mut on_delta: impl FnMut(&str) -> bool + Send + 'static,
        finish: impl FnOnce(Result<GenerateOutput, String>) + Send + 'static,
    ) {
        let source = Arc::clone(&self.source);
        tokio::task::spawn_blocking(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut runtime = source.create_runtime();
                let mut on_token = |delta: &str, _token_id| on_delta(delta);
                let mut should_cancel = || *cancel.borrow();
                let result = runtime
                    .execute(
                        RuntimeInput::PreparedText {
                            prompt: &prompt,
                            entry: TextGenerationEntry::StreamingWithCancel,
                        },
                        &config,
                        true,
                        &mut on_token,
                        &mut should_cancel,
                    )
                    .map_err(failure_message);
                drop(runtime);
                result
            }));
            let (result, panic_payload) = match outcome {
                Ok(result) => (result, None),
                Err(payload) => (Err("inference failed".to_owned()), Some(payload)),
            };
            // The `spawn_blocking` JoinHandle is discarded, so a panic that
            // unwound out of the closure passed to `catch_unwind` would end
            // this task before `finish` ran, and the client would never
            // receive the failure event; `catch_unwind` turns that unwind into
            // the `Err` arm above so `finish` still runs. `finish` stays
            // outside the catch because a panicking token callback may already
            // have sent its terminal event.
            finish(result);
            // A custom panic payload can itself panic when dropped.
            drop(panic_payload);
        });
    }

    /// Build a CPU handle around a tiny Qwen model for binary unit tests.
    #[cfg(feature = "test-utils")]
    pub fn from_qwen_model_for_test(model: crate::model::qwen35::Qwen35Model) -> Self {
        crate::serving_provider::providers::qwen::cpu::from_model(model)
    }

    /// Replace numerical execution while retaining provider preparation in binary tests.
    #[cfg(feature = "test-utils")]
    pub fn with_qwen_generator_for_test<F>(
        model: Arc<crate::model::qwen35::Qwen35Model>,
        generate: F,
    ) -> Self
    where
        F: Fn(
                &str,
                &GenerateConfig,
                &mut dyn FnMut(&str) -> bool,
                &mut dyn FnMut() -> bool,
            ) -> Result<GenerateOutput, crate::error::InferenceError>
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
    ) -> Result<GenerateOutput, crate::error::InferenceError>
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
        _command: crate::serve::metal_worker::AdapterCommand,
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
mod tests {
    use super::{CpuRuntimeSource, SharedCpuHandle};
    use crate::generation::{GenerateConfig, GenerateOutput};
    use crate::model::serving_runtime::{ServingRuntime, WorkerFailure};
    use crate::serve::prepare::PreparationHandle;
    use crate::serving_runtime_contract::{RuntimeInput, TextGenerationEntry};
    use crate::tokenizer::bpe::BpeTokenizer;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread::ThreadId;
    use std::time::Duration;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Observation {
        entry: TextGenerationEntry,
        cancelled: bool,
    }

    #[derive(Default)]
    struct Records {
        created_on: Mutex<Vec<ThreadId>>,
        observations: Mutex<Vec<Observation>>,
        dropped_on: Mutex<Vec<ThreadId>>,
    }

    struct RecordingSource {
        records: Arc<Records>,
    }

    impl CpuRuntimeSource for RecordingSource {
        fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
            let thread_id = std::thread::current().id();
            self.records
                .created_on
                .lock()
                .expect("record lock")
                .push(thread_id);
            Box::new(RecordingRuntime {
                records: Arc::clone(&self.records),
            })
        }
    }

    struct RecordingRuntime {
        records: Arc<Records>,
    }

    impl ServingRuntime for RecordingRuntime {
        fn execute(
            &mut self,
            input: RuntimeInput<'_>,
            _cfg: &GenerateConfig,
            _http_stream: bool,
            on_token: &mut dyn FnMut(&str, u32) -> bool,
            should_cancel: &mut dyn FnMut() -> bool,
        ) -> Result<GenerateOutput, WorkerFailure> {
            let RuntimeInput::PreparedText { entry, .. } = input else {
                return Err(WorkerFailure::Failed("unexpected test input".to_owned()));
            };
            let cancelled = should_cancel();
            self.records
                .observations
                .lock()
                .expect("record lock")
                .push(Observation { entry, cancelled });
            let _ = on_token("delta", 0);
            Ok(GenerateOutput {
                text: "delta".to_owned(),
                token_ids: Vec::new(),
                prompt_tokens: 0,
                generated_tokens: 1,
                stopped: false,
                stop_reason: None,
                token_logprobs: Vec::new(),
            })
        }

        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        fn control(
            &mut self,
            _command: crate::serve::metal_worker::AdapterCommand,
        ) -> Result<crate::serve::lora::AdapterControlResult, crate::serve::lora::AdapterControlError>
        {
            Err(crate::serve::lora::AdapterControlError::InvalidAdapter(
                "test runtime does not accept adapters".to_owned(),
            ))
        }

        fn vision_supported(&self) -> Arc<AtomicBool> {
            Arc::new(AtomicBool::new(false))
        }
    }

    impl Drop for RecordingRuntime {
        fn drop(&mut self) {
            self.records
                .dropped_on
                .lock()
                .expect("record lock")
                .push(std::thread::current().id());
        }
    }

    struct OverlapGate {
        barrier: Barrier,
        state: Mutex<(bool, usize)>,
        active: AtomicUsize,
        max_active: AtomicUsize,
    }

    impl OverlapGate {
        fn new() -> Self {
            Self {
                barrier: Barrier::new(2),
                state: Mutex::new((false, 0)),
                active: AtomicUsize::new(0),
                max_active: AtomicUsize::new(0),
            }
        }

        fn enter(&self) -> bool {
            let mut state = self.state.lock().expect("overlap state lock");
            if state.0 {
                return false;
            }
            state.1 += 1;
            true
        }

        fn release_after_timeout(&self) -> usize {
            let mut state = self.state.lock().expect("overlap state lock");
            state.0 = true;
            state.1
        }
    }

    struct OverlapSource {
        gate: Arc<OverlapGate>,
    }

    impl CpuRuntimeSource for OverlapSource {
        fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
            Box::new(OverlapRuntime {
                gate: Arc::clone(&self.gate),
                vision: Arc::new(AtomicBool::new(false)),
            })
        }
    }

    struct OverlapRuntime {
        gate: Arc<OverlapGate>,
        vision: Arc<AtomicBool>,
    }

    impl ServingRuntime for OverlapRuntime {
        fn execute(
            &mut self,
            input: RuntimeInput<'_>,
            _cfg: &GenerateConfig,
            _http_stream: bool,
            on_token: &mut dyn FnMut(&str, u32) -> bool,
            _should_cancel: &mut dyn FnMut() -> bool,
        ) -> Result<GenerateOutput, WorkerFailure> {
            let RuntimeInput::PreparedText { .. } = input else {
                return Err(WorkerFailure::Failed("unexpected test input".to_owned()));
            };
            let active = self.gate.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.gate.max_active.fetch_max(active, Ordering::SeqCst);
            if self.gate.enter() {
                let _ = self.gate.barrier.wait();
            }
            self.gate.active.fetch_sub(1, Ordering::SeqCst);
            let _ = on_token("delta", 0);
            Ok(GenerateOutput {
                text: "delta".to_owned(),
                token_ids: Vec::new(),
                prompt_tokens: 0,
                generated_tokens: 1,
                stopped: false,
                stop_reason: None,
                token_logprobs: Vec::new(),
            })
        }

        #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
        fn control(
            &mut self,
            _command: crate::serve::metal_worker::AdapterCommand,
        ) -> Result<crate::serve::lora::AdapterControlResult, crate::serve::lora::AdapterControlError>
        {
            Err(crate::serve::lora::AdapterControlError::InvalidAdapter(
                "test runtime does not accept adapters".to_owned(),
            ))
        }

        fn vision_supported(&self) -> Arc<AtomicBool> {
            Arc::clone(&self.vision)
        }
    }

    fn preparation() -> PreparationHandle {
        let tokenizer = BpeTokenizer::from_vocab_and_merges(
            std::collections::HashMap::from([("a".to_owned(), 0)]),
            Vec::new(),
        )
        .expect("tiny tokenizer constructs");
        PreparationHandle::qwen(Arc::new(tokenizer), 128)
    }

    #[tokio::test]
    async fn completion_and_streaming_keep_distinct_entries_on_blocking_threads() {
        let caller_thread = std::thread::current().id();
        let records = Arc::new(Records::default());
        let host = SharedCpuHandle::new(
            Arc::new(RecordingSource {
                records: Arc::clone(&records),
            }),
            preparation(),
        );

        host.spawn_completion(String::new(), GenerateConfig::default())
            .await
            .expect("completion task joins")
            .expect("completion succeeds");

        let (cancel, cancel_rx) = tokio::sync::watch::channel(true);
        let (finish, finished) = tokio::sync::oneshot::channel();
        host.spawn_streaming(
            String::new(),
            GenerateConfig::default(),
            cancel_rx,
            |_| true,
            move |result| {
                let _ = finish.send(result);
            },
        );
        finished
            .await
            .expect("streaming completion callback runs")
            .expect("streaming succeeds");
        drop(cancel);

        let created_on = records.created_on.lock().expect("record lock").clone();
        let observations = records.observations.lock().expect("record lock").clone();
        let dropped_on = records.dropped_on.lock().expect("record lock").clone();
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].entry, TextGenerationEntry::Complete);
        assert!(!observations[0].cancelled);
        assert_eq!(
            observations[1].entry,
            TextGenerationEntry::StreamingWithCancel
        );
        assert!(observations[1].cancelled);
        assert_eq!(created_on.len(), 2);
        assert_eq!(dropped_on, created_on);
        assert!(
            created_on
                .iter()
                .all(|thread_id| *thread_id != caller_thread)
        );
    }

    #[tokio::test]
    async fn completion_and_streaming_execute_at_the_same_time() {
        let gate = Arc::new(OverlapGate::new());
        let host = SharedCpuHandle::new(
            Arc::new(OverlapSource {
                gate: Arc::clone(&gate),
            }),
            preparation(),
        );
        let completion = host.spawn_completion(String::new(), GenerateConfig::default());
        let (finish, finished) = tokio::sync::oneshot::channel();
        host.spawn_streaming(
            String::new(),
            GenerateConfig::default(),
            tokio::sync::watch::channel(false).1,
            |_| true,
            move |result| {
                let _ = finish.send(result);
            },
        );
        let requests = async {
            let completion = completion
                .await
                .expect("completion task joins")
                .expect("completion succeeds");
            let streaming = finished
                .await
                .expect("streaming callback runs")
                .expect("streaming succeeds");
            (completion, streaming)
        };
        let mut requests = Box::pin(requests);

        let outcome = tokio::time::timeout(Duration::from_secs(2), requests.as_mut()).await;
        let Ok((completion, streaming)) = outcome else {
            if gate.release_after_timeout() == 1 {
                let _ = gate.barrier.wait();
            }
            let _ = tokio::time::timeout(Duration::from_secs(2), requests.as_mut())
                .await
                .expect("requests exit after the bounded overlap probe releases the barrier");
            panic!(
                "completion and streaming did not enter execute concurrently; maximum active runtimes: {}",
                gate.max_active.load(Ordering::SeqCst)
            );
        };

        assert_eq!(completion.text, "delta");
        assert_eq!(streaming.text, "delta");
        assert_eq!(gate.max_active.load(Ordering::SeqCst), 2);
    }

    mod streaming_terminal {
        use super::*;

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Outcome {
            CreatePanic,
            ExecutePanic,
            DeltaPanic,
            DropPanic,
            Success,
            WorkerFailure,
        }

        #[derive(Default)]
        struct TerminalRecords {
            events: Mutex<Vec<(&'static str, ThreadId)>>,
            finished: AtomicUsize,
        }

        impl TerminalRecords {
            fn record(&self, event: &'static str) {
                self.events
                    .lock()
                    .expect("event lock")
                    .push((event, std::thread::current().id()));
            }

            fn assert_events(&self, caller: ThreadId, expected: &[&str]) {
                let events = self.events.lock().expect("event lock");
                assert_eq!(
                    events.iter().map(|(event, _)| *event).collect::<Vec<_>>(),
                    expected
                );
                let blocking_thread = events[0].1;
                assert_ne!(blocking_thread, caller);
                assert!(events.iter().all(|(_, thread)| *thread == blocking_thread));
                assert_eq!(self.finished.load(Ordering::SeqCst), 1);
            }
        }

        struct TerminalSource {
            outcome: Outcome,
            records: Arc<TerminalRecords>,
        }

        impl CpuRuntimeSource for TerminalSource {
            fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
                self.records.record("create");
                if self.outcome == Outcome::CreatePanic {
                    panic!("private creation panic payload");
                }
                Box::new(TerminalRuntime {
                    outcome: self.outcome,
                    records: Arc::clone(&self.records),
                })
            }
        }

        struct TerminalRuntime {
            outcome: Outcome,
            records: Arc<TerminalRecords>,
        }

        impl ServingRuntime for TerminalRuntime {
            fn execute(
                &mut self,
                input: RuntimeInput<'_>,
                config: &GenerateConfig,
                http_stream: bool,
                on_token: &mut dyn FnMut(&str, u32) -> bool,
                should_cancel: &mut dyn FnMut() -> bool,
            ) -> Result<GenerateOutput, WorkerFailure> {
                let RuntimeInput::PreparedText { prompt, entry } = input else {
                    panic!("unexpected runtime input");
                };
                assert_eq!(prompt, "literal prompt");
                assert_eq!(entry, TextGenerationEntry::StreamingWithCancel);
                assert_eq!(config.max_new_tokens, 7);
                assert!(http_stream);
                assert!(!should_cancel());
                self.records.record("execute");
                assert!(on_token("partial", 17));
                match self.outcome {
                    Outcome::ExecutePanic => panic!("private execution panic payload"),
                    Outcome::WorkerFailure => {
                        Err(WorkerFailure::Failed("returned worker failure".to_owned()))
                    }
                    _ => Ok(GenerateOutput {
                        text: "confirmed output".to_owned(),
                        token_ids: vec![17, 19],
                        prompt_tokens: 3,
                        generated_tokens: 2,
                        stopped: false,
                        stop_reason: None,
                        token_logprobs: Vec::new(),
                    }),
                }
            }

            #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
            fn control(
                &mut self,
                _command: crate::serve::metal_worker::AdapterCommand,
            ) -> Result<
                crate::serve::lora::AdapterControlResult,
                crate::serve::lora::AdapterControlError,
            > {
                panic!("CPU terminal probe does not accept adapters");
            }

            fn vision_supported(&self) -> Arc<AtomicBool> {
                Arc::new(AtomicBool::new(false))
            }
        }

        impl Drop for TerminalRuntime {
            fn drop(&mut self) {
                self.records.record("drop");
                if self.outcome == Outcome::DropPanic {
                    panic!("private cleanup panic payload");
                }
            }
        }

        fn host(outcome: Outcome, records: &Arc<TerminalRecords>) -> SharedCpuHandle {
            SharedCpuHandle::new(
                Arc::new(TerminalSource {
                    outcome,
                    records: Arc::clone(records),
                }),
                preparation(),
            )
        }

        fn config() -> GenerateConfig {
            GenerateConfig {
                max_new_tokens: 7,
                ..GenerateConfig::default()
            }
        }

        async fn observe(outcome: Outcome) -> Result<GenerateOutput, String> {
            let caller = std::thread::current().id();
            let records = Arc::new(TerminalRecords::default());
            let host = host(outcome, &records);
            let (cancel, cancel_rx) = tokio::sync::watch::channel(false);
            let (finish, finished) = tokio::sync::oneshot::channel();
            let delta_records = Arc::clone(&records);
            let finish_records = Arc::clone(&records);
            host.spawn_streaming(
                "literal prompt".to_owned(),
                config(),
                cancel_rx,
                move |delta| {
                    assert_eq!(delta, "partial");
                    delta_records.record("delta");
                    assert!(
                        outcome != Outcome::DeltaPanic,
                        "private delta panic payload"
                    );
                    true
                },
                move |result| {
                    finish_records.finished.fetch_add(1, Ordering::SeqCst);
                    finish_records.record("finish");
                    let _ = finish.send(result);
                },
            );
            let result = tokio::time::timeout(Duration::from_secs(5), finished)
                .await
                .expect("terminal callback must meet the bounded deadline")
                .expect("terminal callback must survive a runtime panic");
            drop(cancel);
            let expected: &[&str] = if outcome == Outcome::CreatePanic {
                &["create", "finish"]
            } else {
                &["create", "execute", "delta", "drop", "finish"]
            };
            records.assert_events(caller, expected);
            result
        }

        fn assert_error(result: Result<GenerateOutput, String>, expected: &str) {
            let Err(message) = result else {
                panic!("a failed runtime must not report successful completion");
            };
            assert_eq!(message, expected);
        }

        #[tokio::test]
        async fn streaming_create_runtime_panic_reports_terminal_error_once() {
            assert_error(observe(Outcome::CreatePanic).await, "inference failed");
        }

        #[tokio::test]
        async fn streaming_execute_panic_reports_terminal_error_after_cleanup_once() {
            assert_error(observe(Outcome::ExecutePanic).await, "inference failed");
        }

        #[tokio::test]
        async fn streaming_delta_callback_panic_reports_terminal_error_after_cleanup_once() {
            assert_error(observe(Outcome::DeltaPanic).await, "inference failed");
        }

        #[tokio::test]
        async fn streaming_runtime_drop_panic_reports_terminal_error_once() {
            assert_error(observe(Outcome::DropPanic).await, "inference failed");
        }

        #[tokio::test]
        async fn streaming_success_finishes_once_after_runtime_drop() {
            let output = observe(Outcome::Success).await.expect("runtime succeeds");
            assert_eq!(output.text, "confirmed output");
            assert_eq!(output.token_ids, [17, 19]);
            assert_eq!(output.prompt_tokens, 3);
            assert_eq!(output.generated_tokens, 2);
            assert!(!output.stopped);
            assert!(output.stop_reason.is_none());
            assert!(output.token_logprobs.is_empty());
        }

        #[tokio::test]
        async fn streaming_worker_failure_finishes_once_after_runtime_drop() {
            assert_error(
                observe(Outcome::WorkerFailure).await,
                "returned worker failure",
            );
        }

        struct FinishCapture {
            records: Arc<TerminalRecords>,
            released: Option<tokio::sync::oneshot::Sender<()>>,
        }

        impl Drop for FinishCapture {
            fn drop(&mut self) {
                self.records.record("finish_capture_drop");
                if let Some(released) = self.released.take() {
                    let _ = released.send(());
                }
            }
        }

        #[tokio::test]
        async fn streaming_finish_callback_panic_is_not_retried_and_releases_capture() {
            let caller = std::thread::current().id();
            let records = Arc::new(TerminalRecords::default());
            let host = host(Outcome::Success, &records);
            let (released, release_rx) = tokio::sync::oneshot::channel();
            let capture = FinishCapture {
                records: Arc::clone(&records),
                released: Some(released),
            };
            let delta_records = Arc::clone(&records);
            host.spawn_streaming(
                "literal prompt".to_owned(),
                config(),
                tokio::sync::watch::channel(false).1,
                move |delta| {
                    assert_eq!(delta, "partial");
                    delta_records.record("delta");
                    true
                },
                move |result| {
                    let completion_guard = capture;
                    assert_eq!(result.expect("runtime succeeds").text, "confirmed output");
                    completion_guard
                        .records
                        .finished
                        .fetch_add(1, Ordering::SeqCst);
                    completion_guard.records.record("finish");
                    panic!("private finish panic payload");
                },
            );
            tokio::time::timeout(Duration::from_secs(5), release_rx)
                .await
                .expect("a panicking callback must release its captures")
                .expect("the callback capture must signal its destruction");
            records.assert_events(
                caller,
                &[
                    "create",
                    "execute",
                    "delta",
                    "drop",
                    "finish",
                    "finish_capture_drop",
                ],
            );
        }
    }
}
