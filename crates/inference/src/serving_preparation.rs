use std::sync::Arc;

use crate::generation::GenerateConfig;
use crate::serve::ApiError;
use crate::serve::contract::{ChatRequest, GenerationDefaults, ValidatedChatRequest};
use crate::serve::prepare::PreparedChatRequest;
use crate::tokenizer::bpe::BpeTokenizer;

/// Prepared CPU request data and the selected provider's generation config.
#[doc(hidden)]
pub struct PreparedCpuChat {
    /// The validated request consumed by the HTTP response path.
    pub prepared: PreparedChatRequest,
    /// The full config produced during provider preparation.
    pub config: GenerateConfig,
}

/// Opaque, model-bound preparation for the serving binaries.
#[doc(hidden)]
#[derive(Clone)]
pub struct PreparationHandle {
    inner: Arc<dyn RequestPreparation>,
}

pub(crate) trait RequestPreparation: Send + Sync {
    fn tokenize_len(&self, prompt: &str) -> usize;

    fn max_context(&self) -> usize;

    fn tokenizer(&self) -> Option<&BpeTokenizer>;

    fn prepare_cpu(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
    ) -> Result<PreparedCpuChat, ApiError> {
        let mut prepared =
            self.prepare_lattice(req, model_id, default_max_tokens, max_tokens_cap, false)?;
        let stop_strings = std::mem::take(&mut prepared.stop_strings);
        let config = self.lattice_generate_config(
            prepared.max_tokens,
            prepared.temperature,
            prepared.top_p,
            prepared.seed,
            stop_strings,
            prepared.reasoning_budget,
            prepared.logprobs,
            prepared.enable_thinking,
        );
        Ok(PreparedCpuChat { prepared, config })
    }

    fn prepare_lattice(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
        vision_supported: bool,
    ) -> Result<PreparedChatRequest, ApiError>;

    fn normalize_standalone(
        &self,
        req: &ChatRequest,
        defaults: GenerationDefaults,
        model_id: &str,
        vision_supported: bool,
    ) -> Result<ValidatedChatRequest, ApiError>;

    fn standalone_generate_config(&self, req: &ValidatedChatRequest) -> GenerateConfig;

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
    ) -> GenerateConfig;

    fn refuse_standalone_unsupported(&self, req: &ChatRequest) -> Result<(), ApiError>;

    fn supports_adapters(&self) -> bool;
}

impl std::fmt::Debug for PreparationHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparationHandle").finish_non_exhaustive()
    }
}

impl PreparationHandle {
    pub(crate) fn new(inner: Arc<dyn RequestPreparation>) -> Self {
        Self { inner }
    }

    /// Tokenize with the same tokenizer used by worker execution.
    pub fn tokenize_len(&self, prompt: &str) -> usize {
        self.inner.tokenize_len(prompt)
    }

    /// The loaded model tokenizer used for token display, when available.
    pub fn tokenizer(&self) -> Option<&BpeTokenizer> {
        self.inner.tokenizer()
    }

    /// Prepare a request for the selected CPU route and retain its full config.
    pub fn prepare_cpu(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
    ) -> Result<PreparedCpuChat, ApiError> {
        self.inner
            .prepare_cpu(req, model_id, default_max_tokens, max_tokens_cap)
    }

    /// Run the CLI's render, tokenize and context check before stop parsing.
    pub fn prepare_lattice(
        &self,
        req: &ChatRequest,
        model_id: &str,
        default_max_tokens: usize,
        max_tokens_cap: usize,
        vision_supported: bool,
    ) -> Result<PreparedChatRequest, ApiError> {
        self.inner.prepare_lattice(
            req,
            model_id,
            default_max_tokens,
            max_tokens_cap,
            vision_supported,
        )
    }

    /// Apply the standalone server's existing normalization profile.
    pub fn normalize_standalone(
        &self,
        req: &ChatRequest,
        defaults: GenerationDefaults,
        model_id: &str,
        vision_supported: bool,
    ) -> Result<ValidatedChatRequest, ApiError> {
        self.inner
            .normalize_standalone(req, defaults, model_id, vision_supported)
    }

    /// Map validated standalone options through the model's prompt adapter.
    pub fn standalone_generate_config(&self, req: &ValidatedChatRequest) -> GenerateConfig {
        self.inner.standalone_generate_config(req)
    }

    /// Map prepared CLI sampling options through the model's prompt adapter.
    #[allow(clippy::too_many_arguments)]
    pub fn lattice_generate_config(
        &self,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        seed: Option<u64>,
        stop_strings: Vec<String>,
        reasoning_budget: Option<usize>,
        logprobs: Option<usize>,
    ) -> GenerateConfig {
        self.lattice_generate_config_with_thinking(
            max_tokens,
            temperature,
            top_p,
            seed,
            stop_strings,
            reasoning_budget,
            logprobs,
            true,
        )
    }

    /// Map prepared CLI options using the request's effective thinking mode.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn lattice_generate_config_with_thinking(
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
        self.inner.lattice_generate_config(
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

    /// Refuse the standalone-server features this model cannot serve.
    ///
    /// Qwen3.5 serves all of them, so it never refuses here. Gemma 4 text on
    /// the CPU has no adapter support and no grammar-constrained decoding:
    /// a request that selects a LoRA adapter is refused with
    /// `lora_unsupported_backend`, the code `lattice serve` answers with on a
    /// backend without adapters, and a `response_format` of type
    /// `json_schema` with `unsupported_feature`. Image content, stop strings,
    /// `logprobs`, a reasoning budget and typed content parts are refused by
    /// [`Self::normalize_standalone`].
    pub fn refuse_standalone_unsupported(&self, req: &ChatRequest) -> Result<(), ApiError> {
        self.inner.refuse_standalone_unsupported(req)
    }

    /// Whether this model accepts runtime LoRA adapters.
    pub fn supports_adapters(&self) -> bool {
        self.inner.supports_adapters()
    }
}
