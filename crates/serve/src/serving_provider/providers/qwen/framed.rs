//! Qwen chat prompts with each template marker as its token.
//!
//! The renderer writes `<|im_start|>`, `<|im_end|>`, `<think>` and `</think>` as framed markers
//! (see [`framed_prompt`](crate::serving_provider::providers::framed_prompt)), so message text
//! can never produce one. A special token of the tokenizer file is never matched in message
//! text. Every other added token of the file is matched in message text as the whole-string
//! encoding of the published template matches it: on the published Qwen3.5 file those are the
//! tool-call, tool-response and fill-in-the-middle spellings, so text a client passes through
//! from an outside source can spell a tool response. If the route ever renders tool calls from
//! structured fields, those spellings join the marker table.

use crate::serve::ApiError;
use crate::serving_provider::providers::framed_prompt::{
    PromptBuilder, PromptMarker, PromptPiece, prompt_pieces,
};
use lattice_inference::serving_support::{ChatTemplateMarker, ChatTemplatePiece};
use lattice_inference::tokenizer::bpe::BpeTokenizer;

impl PromptMarker for ChatTemplateMarker {
    const MARKERS: &'static [Self] = &ChatTemplateMarker::ALL;

    fn spelling(self) -> &'static str {
        ChatTemplateMarker::spelling(self)
    }
}

/// Appends `pieces` to `prompt`.
///
/// # Errors
/// `invalid_messages` when a text piece holds U+FDD0 or U+FDD1, and `vision_unsupported` for an
/// image piece, which only the vision route places.
fn push_pieces(
    prompt: &mut PromptBuilder<ChatTemplateMarker>,
    pieces: &[ChatTemplatePiece],
) -> Result<(), ApiError> {
    for piece in pieces {
        match piece {
            ChatTemplatePiece::Marker(marker) => prompt.marker(*marker),
            ChatTemplatePiece::Text(text) => prompt.text(text)?,
            ChatTemplatePiece::Image => {
                return Err(ApiError::BadRequest {
                    message: "image input requires a vision-capable model".to_owned(),
                    code: "vision_unsupported",
                });
            }
        }
    }
    Ok(())
}

/// Appends the reasoning block the template's generation prompt opens: `<think>\n` when
/// `enable_thinking`, else an empty block.
fn push_generation_reasoning(
    prompt: &mut PromptBuilder<ChatTemplateMarker>,
    enable_thinking: bool,
) -> Result<(), ApiError> {
    prompt.marker(ChatTemplateMarker::ThinkOpen);
    if enable_thinking {
        prompt.text("\n")
    } else {
        prompt.text("\n\n")?;
        prompt.marker(ChatTemplateMarker::ThinkClose);
        prompt.text("\n\n")
    }
}

fn framed_turns<'a>(
    turns: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<PromptBuilder<ChatTemplateMarker>, ApiError> {
    let pieces = lattice_inference::serving_support::format_chat_template_parts_pieces(turns);
    let mut prompt = PromptBuilder::new();
    push_pieces(&mut prompt, &pieces)?;
    Ok(prompt)
}

/// The framed prompt for `(role, content)` turns, ending with the open generation turn.
///
/// # Errors
/// `invalid_messages` when a message's content holds U+FDD0 or U+FDD1.
pub(crate) fn render_framed<'a>(
    turns: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<String, ApiError> {
    Ok(framed_turns(turns)?.finish().into_framed())
}

/// [`render_framed`] followed by the reasoning block `enable_thinking` selects.
///
/// # Errors
/// As [`render_framed`].
pub(crate) fn render_framed_with_thinking<'a>(
    turns: impl IntoIterator<Item = (&'a str, &'a str)>,
    enable_thinking: bool,
) -> Result<String, ApiError> {
    let mut prompt = framed_turns(turns)?;
    push_generation_reasoning(&mut prompt, enable_thinking)?;
    Ok(prompt.finish().into_framed())
}

/// The framed prompts before and after the one image in `pieces`, the second ending with the
/// reasoning block `enable_thinking` selects. `None` when `pieces` hold no image.
///
/// # Errors
/// As [`render_framed`], and `vision_unsupported` for a second image.
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(crate) fn render_framed_around_image(
    pieces: &[ChatTemplatePiece],
    enable_thinking: bool,
) -> Option<Result<(String, String), ApiError>> {
    let image = pieces
        .iter()
        .position(|piece| *piece == ChatTemplatePiece::Image)?;
    let render = || {
        let mut before = PromptBuilder::new();
        push_pieces(&mut before, &pieces[..image])?;
        let mut after = PromptBuilder::new();
        push_pieces(&mut after, &pieces[image + 1..])?;
        push_generation_reasoning(&mut after, enable_thinking)?;
        Ok((before.finish().into_framed(), after.finish().into_framed()))
    };
    Some(render())
}

/// How one tokenizer encodes a framed Qwen prompt.
pub(crate) struct QwenPromptEncoding<'t> {
    tokenizer: &'t BpeTokenizer,
    /// The ids of each marker, indexed by [`ChatTemplateMarker`]: its token's id, or, when the
    /// tokenizer has no such token, the ids of its spelling encoded as text on its own.
    markers: [Vec<u32>; ChatTemplateMarker::ALL.len()],
    /// The added tokens matched in message text: those the file does not mark special, without
    /// the marker spellings.
    text_tokens: Vec<(String, u32)>,
}

impl<'t> QwenPromptEncoding<'t> {
    pub(crate) fn new(tokenizer: &'t BpeTokenizer) -> Self {
        let spellings = ChatTemplateMarker::ALL.map(ChatTemplateMarker::spelling);
        let text_tokens =
            lattice_tokenizer::__internal::non_special_added_tokens(tokenizer, &spellings);
        let markers = ChatTemplateMarker::ALL.map(|marker| {
            match tokenizer.special_token_id(marker.spelling()) {
                Some(id) => vec![id],
                None => lattice_tokenizer::__internal::encode_text_with_added_tokens(
                    tokenizer,
                    marker.spelling(),
                    &text_tokens,
                ),
            }
        });
        Self {
            tokenizer,
            markers,
            text_tokens,
        }
    }

    /// The added tokens matched in message text, longest spelling first.
    #[cfg(test)]
    pub(crate) fn text_tokens(&self) -> &[(String, u32)] {
        &self.text_tokens
    }

    /// The ids of a framed prompt: each marker as its ids, the text between with only
    /// [`Self::text_tokens`] matched. Nothing is added or truncated.
    pub(crate) fn ids(&self, framed: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for piece in prompt_pieces::<ChatTemplateMarker>(framed) {
            match piece {
                PromptPiece::Marker(marker) => {
                    ids.extend_from_slice(&self.markers[marker as usize])
                }
                PromptPiece::Text(text) => {
                    ids.extend(
                        lattice_tokenizer::__internal::encode_text_with_added_tokens(
                            self.tokenizer,
                            text,
                            &self.text_tokens,
                        ),
                    );
                }
            }
        }
        ids
    }
}

/// The ids of a framed Qwen prompt, as [`QwenPromptEncoding::ids`] gives them.
pub(crate) fn framed_prompt_ids(tokenizer: &BpeTokenizer, framed: &str) -> Vec<u32> {
    QwenPromptEncoding::new(tokenizer).ids(framed)
}

#[cfg(test)]
mod tests;
