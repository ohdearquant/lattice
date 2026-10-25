use super::{ServingRuntime, WorkerFailure};
use crate::serve::route::ServedRoute;
use crate::serving_cpu::{qwen_generate_streaming_traced, qwen_generate_traced};
use crate::serving_runtime_contract::{RuntimeInput, TextGenerationEntry};
use lattice_inference::generation::{GenerateConfig, GenerateOutput};
use lattice_inference::model::qwen35::Qwen35Model;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

pub(crate) struct QwenCpuRuntime<'a> {
    model: &'a Qwen35Model,
    vision: Arc<AtomicBool>,
}

impl<'a> QwenCpuRuntime<'a> {
    pub(crate) fn new(model: &'a Qwen35Model) -> Self {
        Self {
            model,
            vision: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl ServingRuntime for QwenCpuRuntime<'_> {
    fn execute(
        &mut self,
        input: RuntimeInput<'_>,
        cfg: &GenerateConfig,
        http_stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        let RuntimeInput::PreparedText { prompt, entry } = input else {
            return Err(WorkerFailure::Failed(
                "chat messages are not supported by the prepared CPU runtime".to_owned(),
            ));
        };
        let (output, evidence) = match entry {
            TextGenerationEntry::Complete => qwen_generate_traced(self.model, prompt, cfg)
                .map_err(|error| WorkerFailure::Failed(error.to_string()))?,
            TextGenerationEntry::StreamingWithCancel => qwen_generate_streaming_traced(
                self.model,
                prompt,
                cfg,
                |delta| on_token(delta, 0),
                should_cancel,
            )
            .map_err(|error| WorkerFailure::Failed(error.to_string()))?,
        };
        eprintln!(
            "{}",
            ServedRoute::QWEN35_CPU.request_marker(http_stream, evidence)
        );
        Ok(output)
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    fn control(
        &mut self,
        _command: crate::serving_worker::AdapterCommand,
    ) -> Result<crate::serve::lora::AdapterControlResult, crate::serve::lora::AdapterControlError>
    {
        Err(crate::serve::lora::AdapterControlError::InvalidAdapter(
            "runtime LoRA adapters are not supported for this model".to_owned(),
        ))
    }

    fn vision_supported(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.vision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving_runtime_contract::TextGenerationEntry;

    #[test]
    fn completion_uses_the_non_cancellable_generation_entry() {
        let model =
            lattice_inference::model::qwen35::test_support::tiny_zero_model_with_context(64);
        let mut runtime = QwenCpuRuntime::new(&model);
        let config = {
            let mut c = GenerateConfig::default();
            c.max_new_tokens = 2;
            c
        };
        let output = runtime
            .execute(
                RuntimeInput::PreparedText {
                    prompt: "hello",
                    entry: TextGenerationEntry::Complete,
                },
                &config,
                false,
                &mut |_, _| true,
                &mut || true,
            )
            .expect("completion succeeds despite a true cancellation predicate");
        assert_ne!(
            output.stop_reason,
            Some(lattice_inference::StopReason::Interrupt)
        );
    }
}
