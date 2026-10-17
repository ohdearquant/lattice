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
mod tests;
