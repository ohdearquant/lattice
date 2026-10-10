use crate::generation::GenerateConfig;
use crate::model::qwen35::Qwen35Model;
use crate::serve::ApiError;
use crate::serve::contract::{
    ChatRequest, GenerationDefaults, ServeProfile, ValidatedChatRequest, normalize_request,
};
use crate::serve::prepare::PreparedChatRequest;
use crate::serving_cpu_host::{CpuRuntimeSource, SharedCpuHandle};
use crate::serving_preparation::{PreparationHandle, RequestPreparation};
use crate::tokenizer::Tokenizer as _;
use crate::tokenizer::bpe::BpeTokenizer;
use std::sync::Arc;

use super::{build_cfg, lattice_gen_cfg, prepare_chat_request};
use crate::model::serving_runtime::{QwenCpuRuntime, ServingRuntime};
use crate::serving_provider::CheckpointEvidence;

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

    fn prepare_lattice(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
        _vision_supported: bool,
    ) -> Result<PreparedChatRequest, ApiError> {
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

    fn lattice_generate_config(
        &self,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        seed: Option<u64>,
        stop_strings: Vec<String>,
        reasoning_budget: Option<usize>,
        logprobs: Option<usize>,
    ) -> GenerateConfig {
        lattice_gen_cfg(
            max_tokens,
            temperature,
            top_p,
            seed,
            stop_strings,
            reasoning_budget,
            logprobs,
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

pub(super) fn load(evidence: &CheckpointEvidence) -> Result<SharedCpuHandle, String> {
    let model = super::load_cpu(&evidence.directory)
        .map_err(|error| format!("failed to load model: {error}"))?;
    Ok(from_model(model))
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
    fn cpu_source_holds_shared_model_and_preparation_resources() {
        let model = crate::model::qwen35::test_support::tiny_zero_model_with_context(64);
        assert_eq!(model.max_context(), 64);
        let host = from_model(model);
        assert!(host.preparation().tokenizer().is_some());
        let _clone = host.clone();
    }
}
