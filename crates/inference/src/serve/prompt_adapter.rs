//! Per-family chat prompt adapters.
//!
//! A prompt adapter owns one model family's chat conventions: rendering
//! normalized chat messages to the prompt text, the family's stop token ids,
//! and the defaults step that turns [`RequestedChatOptions`] into an
//! effective request. Request validation stays family-neutral; a family
//! default is supplied only here.

use crate::generation::GenerateConfig;
use crate::serve::ApiError;
use crate::serve::contract::{
    GenerationDefaults, NormalizedChatMessage, RequestedChatOptions, ValidatedChatRequest,
};

#[cfg(test)]
pub(crate) use crate::serving_provider::providers::gemma::template_trim;
pub(crate) use crate::serving_provider::providers::qwen::QwenChatDefaults;

pub(crate) fn apply_qwen_thinking_mode(prompt: &mut String, enable_thinking: bool) {
    if enable_thinking {
        prompt.push_str("<think>\n");
    } else {
        prompt.push_str("<think>\n\n</think>\n\n");
    }
}

/// One model family's chat conventions.
pub(crate) trait PromptAdapter {
    /// The prompt text for `messages`, ending with the open generation turn.
    fn render(&self, messages: &[NormalizedChatMessage]) -> String;

    /// Render with the effective thinking mode for this request.
    fn render_with_thinking(
        &self,
        messages: &[NormalizedChatMessage],
        _enable_thinking: bool,
    ) -> String {
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
    pub(crate) bos_token: String,
    pub(crate) stop_token_ids: Vec<u32>,
}

#[cfg(test)]
use crate::model::qwen35_config::QWEN_CHAT_IM_END_TOKEN_ID;
#[cfg(test)]
use crate::tokenizer::gemma_bpe::GemmaBpeTokenizer;
#[cfg(test)]
mod tests;
