use std::sync::Arc;

use crate::generation::GenerateConfig;
use crate::serve::ApiError;
use crate::serve::contract::{ChatRequest, GenerationDefaults, ValidatedChatRequest};
use crate::serve::prepare::PreparedChatRequest;

/// Opaque, model-bound preparation for the serving binaries.
#[doc(hidden)]
#[derive(Clone)]
pub struct PreparationHandle {
    inner: Arc<dyn RequestPreparation>,
}

pub(crate) trait RequestPreparation: Send + Sync {
    fn tokenize_len(&self, prompt: &str) -> usize;

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
    #[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
    pub(crate) fn new(inner: Arc<dyn RequestPreparation>) -> Self {
        Self { inner }
    }

    /// Tokenize with the same tokenizer used by worker execution.
    pub fn tokenize_len(&self, prompt: &str) -> usize {
        self.inner.tokenize_len(prompt)
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
        self.inner.lattice_generate_config(
            max_tokens,
            temperature,
            top_p,
            seed,
            stop_strings,
            reasoning_budget,
            logprobs,
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
