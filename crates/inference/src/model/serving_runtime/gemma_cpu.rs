//! Gemma 4 E2B text on the CPU, confined to one serving worker.

use super::ServingRuntime;
use crate::forward::metal_qwen35::ChatMessage;
use crate::generation::{GenerateConfig, GenerateOutput};
use crate::serve::ApiError;
use crate::serve::lora::{AdapterControlError, AdapterControlResult, LoraSelection};
use crate::serve::metal_worker::{AdapterCommand, WorkerFailure, cancelled_output};
use crate::serve::prepare::lora_unsupported_backend;
use crate::serve::route::ServedRoute;
use crate::serving_cpu::GemmaCpuServing;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Runs each admitted job on the Gemma CPU session under the shared decoder
/// driver, on the worker thread, one job at a time.
///
/// Gemma 4 text has no adapter, vision or grammar path, so each of those is
/// refused by name before any generation work starts.
pub(crate) struct GemmaCpuRuntime {
    serving: Arc<GemmaCpuServing>,
    vision: Arc<AtomicBool>,
}

impl GemmaCpuRuntime {
    pub(crate) fn new(serving: Arc<GemmaCpuServing>) -> Self {
        Self {
            serving,
            vision: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl ServingRuntime for GemmaCpuRuntime {
    fn generate(
        &mut self,
        messages: &[ChatMessage],
        cfg: &GenerateConfig,
        lora: &[LoraSelection],
        stream: bool,
        on_token: &mut dyn FnMut(&str, u32) -> bool,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<GenerateOutput, WorkerFailure> {
        if !lora.is_empty() {
            return Err(WorkerFailure::Rejected(lora_unsupported_backend()));
        }
        if cfg.grammar.is_some() {
            return Err(WorkerFailure::Rejected(ApiError::BadRequest {
                message: "grammar-constrained output is not supported for this model".to_string(),
                code: "unsupported_feature",
            }));
        }
        let (prompt, _prompt_len) = self
            .serving
            .render_within_window(messages, cfg)
            .map_err(WorkerFailure::Rejected)?;
        if should_cancel() {
            return Ok(cancelled_output());
        }
        let (output, evidence) = self.serving.generate_streaming(
            &prompt,
            cfg,
            |delta| on_token(delta, 0),
            should_cancel,
        )?;
        eprintln!(
            "{}",
            ServedRoute::GEMMA4_CPU.request_marker(stream, evidence)
        );
        Ok(output)
    }

    fn control(
        &mut self,
        _command: AdapterCommand,
    ) -> Result<AdapterControlResult, AdapterControlError> {
        Err(AdapterControlError::InvalidAdapter(
            "runtime LoRA adapters are not supported for this model".to_string(),
        ))
    }

    fn vision_supported(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.vision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::metal_qwen35::ChatMessage;
    use crate::serve::lora::LoraSelection;

    fn runtime() -> GemmaCpuRuntime {
        GemmaCpuRuntime::new(Arc::new(crate::serving_cpu::tiny_zero_serving()))
    }

    fn greedy(max_new_tokens: usize) -> GenerateConfig {
        GenerateConfig {
            max_new_tokens,
            temperature: 0.0,
            repetition_penalty: 1.0,
            stop_token_ids: vec![],
            ..Default::default()
        }
    }

    fn user(text: &str) -> Vec<ChatMessage> {
        vec![ChatMessage::user(text)]
    }

    fn run(
        runtime: &mut GemmaCpuRuntime,
        messages: &[ChatMessage],
        cfg: &GenerateConfig,
        lora: &[LoraSelection],
    ) -> Result<(GenerateOutput, Vec<String>), WorkerFailure> {
        let mut deltas = Vec::new();
        let output = runtime.generate(
            messages,
            cfg,
            lora,
            true,
            &mut |delta, _| {
                deltas.push(delta.to_string());
                true
            },
            &mut || false,
        )?;
        Ok((output, deltas))
    }

    fn rejected_code(result: Result<(GenerateOutput, Vec<String>), WorkerFailure>) -> &'static str {
        match result {
            Err(WorkerFailure::Rejected(ApiError::BadRequest { code, .. })) => code,
            other => panic!("expected a Rejected BadRequest, got {other:?}"),
        }
    }

    #[test]
    fn a_text_job_generates_under_the_shared_driver() {
        let mut runtime = runtime();
        let (output, deltas) = run(&mut runtime, &user("hello"), &greedy(3), &[])
            .expect("a plain text job is generated");
        assert_eq!(output.generated_tokens, 3);
        assert_eq!(deltas.concat(), output.text);
    }

    #[test]
    fn an_adapter_selection_is_refused_by_name_before_generation() {
        let mut runtime = runtime();
        let selection = [LoraSelection { id: 1, scale: 1.0 }];
        assert_eq!(
            rejected_code(run(&mut runtime, &user("hello"), &greedy(3), &selection)),
            "lora_unsupported_backend"
        );
    }

    #[test]
    fn a_grammar_is_refused_by_name_before_generation() {
        let mut runtime = runtime();
        let mut cfg = greedy(3);
        let spec = crate::grammar::GrammarSpec::JsonSchema(serde_json::json!({"type": "object"}));
        cfg.grammar = Some(Arc::new(
            crate::grammar::GrammarEngine::new(&spec, vec![b"{".to_vec(), b"}".to_vec()])
                .expect("a tiny grammar compiles"),
        ));
        assert_eq!(
            rejected_code(run(&mut runtime, &user("hello"), &cfg, &[])),
            "unsupported_feature"
        );
    }

    #[test]
    fn an_image_message_is_refused_by_name() {
        let mut runtime = runtime();
        let messages = vec![ChatMessage::user_with_image("see", vec![1, 2, 3], 0)];
        assert_eq!(
            rejected_code(run(&mut runtime, &messages, &greedy(3), &[])),
            "vision_unsupported"
        );
    }

    #[test]
    fn a_prompt_over_the_window_is_rejected_not_failed() {
        let mut runtime = runtime();
        let window = runtime.serving.max_context();
        let long = "word ".repeat(window + 8);
        assert_eq!(
            rejected_code(run(&mut runtime, &user(&long), &greedy(3), &[])),
            "context_length_exceeded"
        );
    }

    #[test]
    fn a_job_cancelled_before_prefill_does_no_generation() {
        let mut runtime = runtime();
        let output = runtime
            .generate(
                &user("hello"),
                &greedy(3),
                &[],
                false,
                &mut |_, _| true,
                &mut || true,
            )
            .expect("a cancelled job is answered, not failed");
        assert_eq!(output.generated_tokens, 0);
        assert_eq!(output.stop_reason, Some(crate::StopReason::Interrupt));
    }

    #[test]
    fn adapter_commands_are_refused() {
        let mut runtime = runtime();
        let error = runtime
            .control(AdapterCommand::Unload { id: 1 })
            .expect_err("Gemma has no adapters");
        assert!(
            matches!(error, AdapterControlError::InvalidAdapter(_)),
            "{error:?}"
        );
    }

    #[test]
    fn vision_capability_is_off() {
        let runtime = runtime();
        assert!(
            !runtime
                .vision_supported()
                .load(std::sync::atomic::Ordering::Acquire)
        );
    }
}
