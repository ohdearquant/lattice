use crate::serve::ApiError;
use crate::serve::contract::{
    ChatRequest, GenerationDefaults, ServeProfile, ValidatedChatRequest, normalize_request,
};
use crate::serve::prepare::PreparedChatRequest;
use crate::serving_cpu_host::{CpuRuntimeSource, SharedCpuHandle};
use crate::serving_preparation::{PreparationHandle, RequestPreparation};
use lattice_inference::generation::GenerateConfig;
use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::tokenizer::Tokenizer as _;
use lattice_inference::tokenizer::bpe::BpeTokenizer;
use std::sync::Arc;

use super::{build_cfg, lattice_gen_cfg, prepare_chat_request, prepare_standalone_chat_request};
use crate::serving_provider::CheckpointEvidence;
use crate::serving_runtime::{QwenCpuRuntime, ServingRuntime};

struct QwenCpuPreparation {
    model: Arc<Qwen35Model>,
}

impl RequestPreparation for QwenCpuPreparation {
    fn tokenize_len(&self, prompt: &str) -> usize {
        self.model.tokenizer().tokenize(prompt).pre_truncation_len
    }

    fn max_context(&self) -> usize {
        self.model.max_context()
    }

    fn tokenizer(&self) -> Option<&BpeTokenizer> {
        Some(self.model.tokenizer())
    }

    fn prepare_lattice_with_thinking(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
        _vision_supported: bool,
    ) -> Result<(PreparedChatRequest, bool), ApiError> {
        prepare_chat_request(
            req,
            model_id,
            default_max_tokens,
            max_tokens_cap,
            false,
            |prompt| self.tokenize_len(prompt),
            || self.max_context(),
        )
    }

    fn normalize_standalone(
        &self,
        req: &ChatRequest,
        defaults: GenerationDefaults,
        model_id: &str,
        vision_supported: bool,
    ) -> Result<ValidatedChatRequest, ApiError> {
        normalize_request(
            req,
            defaults,
            ServeProfile::lattice_serve(model_id, self.model.max_context())
                .with_vision_support(vision_supported),
        )
    }

    fn standalone_generate_config(&self, req: &ValidatedChatRequest) -> GenerateConfig {
        build_cfg(req)
    }

    fn prepare_standalone(
        &self,
        req: &ChatRequest,
        defaults: GenerationDefaults,
        model_id: &str,
        vision_supported: bool,
    ) -> Result<(ValidatedChatRequest, GenerateConfig), ApiError> {
        prepare_standalone_chat_request(
            req,
            defaults,
            ServeProfile::lattice_serve(model_id, self.model.max_context())
                .with_vision_support(vision_supported),
        )
    }

    fn lattice_generate_config(
        &self,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        seed: Option<u64>,
        stop_strings: Vec<String>,
        reasoning_budget: Option<usize>,
        logprobs: Option<usize>,
        enable_thinking: bool,
    ) -> GenerateConfig {
        lattice_gen_cfg(
            max_tokens,
            temperature,
            top_p,
            seed,
            stop_strings,
            reasoning_budget,
            logprobs,
            enable_thinking,
        )
    }

    fn refuse_standalone_unsupported(&self, _req: &ChatRequest) -> Result<(), ApiError> {
        Ok(())
    }

    fn supports_adapters(&self) -> bool {
        false
    }
}

struct QwenCpuSource {
    model: Arc<Qwen35Model>,
}

impl CpuRuntimeSource for QwenCpuSource {
    fn create_runtime(&self) -> Box<dyn ServingRuntime + '_> {
        Box::new(QwenCpuRuntime::new(&self.model))
    }
}

/// Load for a server, bounding the served window as `--max-context` and
/// [`super::context::DEFAULT_SERVE_MAX_CONTEXT`] do for the Metal worker. The
/// RoPE table is built at the served window, never at the checkpoint's full one.
pub(super) fn load_with_max_context(
    evidence: &CheckpointEvidence,
    requested: Option<usize>,
) -> Result<SharedCpuHandle, String> {
    let mut model = Qwen35Model::from_safetensors_with_max_context(
        &evidence.directory,
        requested.unwrap_or(super::context::DEFAULT_SERVE_MAX_CONTEXT),
    )
    .map_err(|error| format!("failed to load model: {error}"))?;
    let (window, source) = bound_served_window(&mut model, requested);
    eprintln!(
        "[lattice serve] CPU context window {window} ({})",
        source.describe()
    );
    Ok(from_model(model))
}

/// Resolve the served window against the checkpoint's declared window and lower
/// the model to it. A model already loaded at that window is left unchanged.
pub(crate) fn bound_served_window(
    model: &mut Qwen35Model,
    requested: Option<usize>,
) -> (usize, super::context::ServeContextSource) {
    let (window, source) =
        super::context::resolve_serve_context(model.config().max_position_embeddings, requested);
    model.limit_context(window);
    (model.max_context(), source)
}

pub(crate) fn from_model(mut model: Qwen35Model) -> SharedCpuHandle {
    model.ensure_tokenizer_max_seq_len(model.max_context());
    let model = Arc::new(model);
    let preparation = preparation(Arc::clone(&model));
    let source: Arc<dyn CpuRuntimeSource> = Arc::new(QwenCpuSource { model });
    SharedCpuHandle::new(source, preparation)
}

pub(crate) fn preparation(model: Arc<Qwen35Model>) -> PreparationHandle {
    PreparationHandle::new(Arc::new(QwenCpuPreparation { model }))
}

#[cfg(test)]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<QwenCpuSource>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_preparation_preserves_supported_logprobs() {
        let model =
            lattice_inference::model::qwen35::test_support::tiny_zero_model_with_context(4096);
        let host = from_model(model);
        for (logprobs, top_logprobs, expected) in [
            (None, None, None),
            (Some(false), None, None),
            (Some(true), None, Some(0)),
            (Some(true), Some(0), Some(0)),
            (Some(true), Some(2), Some(2)),
            (Some(true), Some(20), Some(20)),
        ] {
            let req: ChatRequest = serde_json::from_value(serde_json::json!({
                "model": "served-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 5,
                "logprobs": logprobs,
                "top_logprobs": top_logprobs,
            }))
            .expect("chat request");
            let prepared = host
                .prepare(&req, "served-model", 64, 4096)
                .expect("CPU preparation must support logprobs");
            assert_eq!(prepared.prepared.logprobs, expected);
            assert_eq!(prepared.config.logprobs, expected);
        }
    }

    #[test]
    fn cpu_source_holds_shared_model_and_preparation_resources() {
        let model =
            lattice_inference::model::qwen35::test_support::tiny_zero_model_with_context(4096);
        assert_eq!(model.max_context(), 4096);
        let host = from_model(model);
        assert!(host.preparation().tokenizer().is_some());
        let _clone = host.clone();
        let base: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": "served-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
        }))
        .expect("default request");
        let default = crate::serve::prepare::prepare_chat_request_with_config(
            &base,
            "served-model",
            64,
            4096,
            false,
            str::len,
            || 4096,
        )
        .expect("default preparation");
        let cpu_default = host
            .prepare(&base, "served-model", 64, 4096)
            .expect("CPU provider default thinking request");
        assert_eq!(cpu_default.prepared.prompt, default.prepared.prompt);
        assert_eq!(
            cpu_default.prepared.prompt,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert!(!cpu_default.config.enable_thinking);
        for enabled in [true, false] {
            let req: ChatRequest = serde_json::from_value(serde_json::json!({
                "model": "served-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 5,
                "chat_template_kwargs": {"enable_thinking": enabled},
            }))
            .expect("explicit request");
            let prepared = crate::serve::prepare::prepare_chat_request_with_config(
                &req,
                "served-model",
                64,
                4096,
                false,
                str::len,
                || 4096,
            )
            .expect("explicit preparation");
            let cpu_prepared = host
                .prepare(&req, "served-model", 64, 4096)
                .expect("CPU provider explicit thinking switch");
            assert_eq!(cpu_prepared.prepared.prompt, prepared.prepared.prompt);
            let expected_prompt = if enabled {
                "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
            } else {
                "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            };
            assert_eq!(cpu_prepared.prepared.prompt, expected_prompt);
            assert_eq!(cpu_prepared.config.enable_thinking, enabled);
            assert_eq!(
                crate::serve::GenerateConfigSnapshot::from(&cpu_prepared.config),
                crate::serve::GenerateConfigSnapshot::from(&prepared.config),
            );
        }

        let budgeted: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": "served-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
            "reasoning_budget": 16,
        }))
        .expect("budgeted request");
        let budgeted = host
            .prepare(&budgeted, "served-model", 64, 4096)
            .expect("CPU provider request budget enables thinking");
        assert_eq!(
            budgeted.prepared.prompt,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert!(budgeted.config.enable_thinking);
        assert_eq!(budgeted.config.reasoning_budget, Some(16));
    }

    #[test]
    fn the_served_window_bounds_what_cpu_preparation_admits() {
        use super::super::context::{DEFAULT_SERVE_MAX_CONTEXT, ServeContextSource};
        let tiny = lattice_inference::model::qwen35::test_support::tiny_zero_model_with_context;

        let mut model = tiny(4096);
        assert_eq!(
            bound_served_window(&mut model, Some(1_000_000)),
            (4096, ServeContextSource::Model)
        );
        let mut model = tiny(4096);
        assert_eq!(
            bound_served_window(&mut model, Some(300)),
            (300, ServeContextSource::Flag)
        );
        assert_eq!(model.max_context(), 300);

        let request = |max_tokens: usize| -> ChatRequest {
            serde_json::from_value(serde_json::json!({
                "model": "served-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": max_tokens,
            }))
            .expect("chat request")
        };
        let over_default = DEFAULT_SERVE_MAX_CONTEXT;
        let unbounded = from_model(tiny(40_000));
        assert!(
            unbounded
                .prepare(&request(over_default), "served-model", 64, 40_000)
                .is_ok(),
            "a 40000-position model admits the request before the bound"
        );

        let mut model = tiny(40_000);
        model.limit_context(DEFAULT_SERVE_MAX_CONTEXT);
        assert_eq!(
            bound_served_window(&mut model, None),
            (DEFAULT_SERVE_MAX_CONTEXT, ServeContextSource::DefaultCap),
            "a model loaded at the served window still names the default cap"
        );

        let mut model = tiny(40_000);
        assert_eq!(
            bound_served_window(&mut model, None),
            (DEFAULT_SERVE_MAX_CONTEXT, ServeContextSource::DefaultCap)
        );
        let bounded = from_model(model);
        assert!(
            bounded
                .prepare(&request(over_default), "served-model", 64, 40_000)
                .is_err(),
            "the default window refuses a prompt plus {over_default} new tokens"
        );
    }
}
