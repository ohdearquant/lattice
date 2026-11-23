//! Per-family chat prompt adapters.
//!
//! A prompt adapter owns one model family's chat conventions: rendering
//! normalized chat messages to the prompt text, the family's stop token ids,
//! and the defaults step that turns [`RequestedChatOptions`] into an
//! effective request. Request validation stays family-neutral; a family
//! default is supplied only here.

use crate::serve::ApiError;
use crate::serve::contract::{
    GenerationDefaults, NormalizedChatMessage, RequestedChatOptions, ValidatedChatRequest,
};
use lattice_inference::generation::GenerateConfig;

#[cfg(test)]
pub(crate) use crate::serving_provider::providers::gemma::template_trim;
pub(crate) use crate::serving_provider::providers::qwen::QwenChatDefaults;

/// One model family's chat conventions.
pub(crate) trait PromptAdapter {
    /// The prompt for `messages`, ending with the open generation turn.
    ///
    /// # Errors
    /// A refusal for message text the family's prompt cannot carry.
    fn render(&self, messages: &[NormalizedChatMessage]) -> Result<String, ApiError>;

    /// Render with the effective thinking mode for this request.
    ///
    /// # Errors
    /// As [`Self::render`].
    fn render_with_thinking(
        &self,
        messages: &[NormalizedChatMessage],
        _enable_thinking: bool,
    ) -> Result<String, ApiError> {
        self.render(messages)
    }

    /// Token ids that end the model's turn.
    fn stop_token_ids(&self) -> &[u32];

    /// Effective request values: `options` as the request sent them, with the
    /// family's defaults (and the server's `generation` defaults) applied to
    /// every omitted option, and every control the family cannot serve
    /// refused.
    fn apply_defaults(
        &self,
        generation: GenerationDefaults,
        options: RequestedChatOptions,
    ) -> Result<ValidatedChatRequest, ApiError>;

    /// The `GenerateConfig` for an effective request.
    fn generate_config(&self, req: &ValidatedChatRequest) -> GenerateConfig;
}

/// Gemma 4 E2B text chat adapter loaded from a checkpoint.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct GemmaPromptAdapter {
    /// The id of each template marker, indexed by `GemmaMarker`.
    pub(crate) marker_ids: [u32; 3],
    pub(crate) stop_token_ids: Vec<u32>,
}

#[cfg(test)]
use lattice_inference::model::qwen35_config::QWEN_CHAT_IM_END_TOKEN_ID;
#[cfg(test)]
use lattice_inference::tokenizer::gemma_bpe::GemmaBpeTokenizer;
#[cfg(test)]
mod tests;
