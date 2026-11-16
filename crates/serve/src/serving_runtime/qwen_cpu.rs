use super::{ServingRuntime, WorkerFailure};
use crate::serve::route::ServedRoute;
use crate::serving_cpu::{qwen_generate_streaming_traced, qwen_generate_traced};
use crate::serving_runtime_contract::{RuntimeInput, TextGenerationEntry};
use crate::serving_worker::{ScoreOutcome, ScoreRow};
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

    fn score_option_rows(&mut self, rows: &[ScoreRow]) -> ScoreOutcome {
        rows.iter()
            .map(|row| {
                self.model
                    .score_option_letters_batched(&row.ids, &row.labels)
                    .map(|scores| scores.logits)
                    .map_err(|error| error.to_string())
            })
            .collect()
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

    #[test]
    fn option_rows_are_scored_as_raw_logits_one_row_each() {
        let model =
            lattice_inference::model::qwen35::test_support::tiny_zero_model_with_context(64);
        let mut runtime = QwenCpuRuntime::new(&model);
        let rows = vec![
            ScoreRow {
                ids: vec![1, 2, 3],
                labels: vec![4, 5, 6],
            },
            ScoreRow {
                ids: vec![6],
                labels: vec![0, 1],
            },
        ];
        let logits = runtime.score_option_rows(&rows).expect("scores");
        // The zero model's logits are all 0.0 while its option softmax is
        // uniform, so this also tells raw logits from probabilities.
        assert_eq!(logits, vec![vec![0.0; 3], vec![0.0; 2]]);
    }

    #[test]
    fn a_row_the_model_refuses_fails_the_request() {
        let model =
            lattice_inference::model::qwen35::test_support::tiny_zero_model_with_context(64);
        let mut runtime = QwenCpuRuntime::new(&model);
        let rows = vec![
            ScoreRow {
                ids: vec![1, 2],
                labels: vec![4, 5],
            },
            ScoreRow {
                ids: Vec::new(),
                labels: vec![4, 5],
            },
        ];
        assert!(runtime.score_option_rows(&rows).is_err());
    }
}
