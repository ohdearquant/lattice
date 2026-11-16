use super::{ServingRuntime, WorkerFailure};
use crate::serve::route::ServedRoute;
use crate::serving_cpu::{qwen_generate_streaming_traced, qwen_generate_traced};
use crate::serving_runtime_contract::{RuntimeInput, TextGenerationEntry};
use crate::serving_worker::{ScoreOutcome, ScoreRow};
use lattice_inference::InferenceError;
use lattice_inference::generation::{GenerateConfig, GenerateOutput};
use lattice_inference::model::qwen35::{Qwen35Model, ScoringPrefix, shared_scoring_prefix_len};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

pub(crate) struct QwenCpuRuntime<'a> {
    model: &'a Qwen35Model,
    vision: Arc<AtomicBool>,
    /// The prefix the current scoring request's rows share, prefilled once.
    /// The CPU host creates a runtime per request, so it is dropped with it.
    prefix: Option<ScoringPrefix>,
}

impl<'a> QwenCpuRuntime<'a> {
    pub(crate) fn new(model: &'a Qwen35Model) -> Self {
        Self {
            model,
            vision: Arc::new(AtomicBool::new(false)),
            prefix: None,
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

    fn begin_option_rows(&mut self, rows: &[ScoreRow]) -> Result<(), String> {
        self.prefix = None;
        let pairs: Vec<(&[u32], &[u32])> = rows
            .iter()
            .map(|row| (row.ids.as_slice(), row.labels.as_slice()))
            .collect();
        let shared = shared_scoring_prefix_len(&pairs);
        let Some(first) = rows.first().filter(|_| rows.len() > 1 && shared > 0) else {
            return Ok(());
        };
        match self.model.prefill_scoring_prefix(&first.ids[..shared]) {
            Ok(prefix) => {
                self.prefix = Some(prefix);
                Ok(())
            }
            // A model the batched prefill does not run scores each row alone.
            Err(InferenceError::UnsupportedModel(_)) => Ok(()),
            Err(error) => Err(error.to_string()),
        }
    }

    fn score_option_rows(&mut self, rows: &[ScoreRow]) -> ScoreOutcome {
        rows.iter()
            .map(|row| {
                let scores = match &self.prefix {
                    Some(prefix) if continues(prefix.ids(), &row.ids) => self
                        .model
                        .score_option_letters_after_prefix(prefix, &row.ids, &row.labels),
                    _ => self
                        .model
                        .score_option_letters_batched(&row.ids, &row.labels),
                };
                scores
                    .map(|scores| scores.logits)
                    .map_err(|error| error.to_string())
            })
            .collect()
    }
}

/// Whether `ids` is `prefix` followed by at least one more token.
fn continues(prefix: &[u32], ids: &[u32]) -> bool {
    ids.len() > prefix.len() && ids.starts_with(prefix)
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

    fn rows(ids: &[&[u32]]) -> Vec<ScoreRow> {
        ids.iter()
            .map(|ids| ScoreRow {
                ids: ids.to_vec(),
                labels: vec![4, 5],
            })
            .collect()
    }

    #[test]
    fn beginning_a_request_prefills_the_prefix_its_rows_share() {
        let model =
            lattice_inference::model::qwen35::test_support::tiny_zero_model_with_context(64);
        let mut runtime = QwenCpuRuntime::new(&model);
        let shared = rows(&[&[1, 2, 3, 4], &[1, 2, 3, 5, 6], &[1, 2, 3]]);
        runtime.begin_option_rows(&shared).expect("begins");
        assert_eq!(
            runtime.prefix.as_ref().map(ScoringPrefix::ids),
            Some(&[1, 2][..])
        );
        let logits = runtime.score_option_rows(&shared).expect("scores");
        assert_eq!(logits, vec![vec![0.0; 2]; 3]);
        // A row the prefix does not reach is scored on its own.
        let other = rows(&[&[3, 2, 1]]);
        assert_eq!(
            runtime.score_option_rows(&other).expect("scores"),
            vec![vec![0.0; 2]]
        );
        // The next request replaces the prefix, or clears it when its rows
        // share none.
        runtime
            .begin_option_rows(&rows(&[&[1, 2], &[3, 4]]))
            .expect("begins");
        assert!(runtime.prefix.is_none());
        runtime
            .begin_option_rows(&rows(&[&[1, 2, 3]]))
            .expect("begins");
        assert!(runtime.prefix.is_none());
    }

    #[test]
    fn a_prefix_the_model_refuses_fails_the_request() {
        let model =
            lattice_inference::model::qwen35::test_support::tiny_zero_model_with_context(64);
        let mut runtime = QwenCpuRuntime::new(&model);
        let out_of_vocab = rows(&[&[1, 500, 3], &[1, 500, 4]]);
        assert!(runtime.begin_option_rows(&out_of_vocab).is_err());
        assert!(runtime.prefix.is_none());
    }

    #[test]
    fn continues_needs_the_prefix_and_one_more_token() {
        assert!(continues(&[1, 2], &[1, 2, 3]));
        assert!(!continues(&[1, 2], &[1, 2]));
        assert!(!continues(&[1, 2], &[1, 3, 3]));
        assert!(!continues(&[1, 2], &[1]));
    }
}
