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
use lattice_inference::serving_support::{
    ChatHistoryReasoning, ChatPromptEncoder, ChatTemplateMarker, ChatTemplatePiece,
};
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
    history: ChatHistoryReasoning,
) -> Result<PromptBuilder<ChatTemplateMarker>, ApiError> {
    let pieces = lattice_inference::serving_support::format_chat_template_parts_pieces_with_history(
        turns, history,
    );
    let mut prompt = PromptBuilder::new();
    push_pieces(&mut prompt, &pieces)?;
    Ok(prompt)
}

/// How a request renders the reasoning block of an earlier assistant turn.
///
/// [`ChatHistoryReasoning::RestoreEmpty`] when `restore_empty_reasoning` is set and the request
/// has reasoning off: the generation prompt then writes the empty block itself, so a reply
/// comes back without it, and writing it back lets the next turn append to the cached reply.
/// The published template's rendering otherwise.
fn history_reasoning(restore_empty_reasoning: bool, enable_thinking: bool) -> ChatHistoryReasoning {
    if restore_empty_reasoning && !enable_thinking {
        ChatHistoryReasoning::RestoreEmpty
    } else {
        ChatHistoryReasoning::Template
    }
}

/// The framed prompt for `(role, content)` turns, ending with the open generation turn.
///
/// # Errors
/// `invalid_messages` when a message's content holds U+FDD0 or U+FDD1.
pub(crate) fn render_framed<'a>(
    turns: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<String, ApiError> {
    Ok(framed_turns(turns, ChatHistoryReasoning::Template)?
        .finish()
        .into_framed())
}

/// [`render_framed`] followed by the reasoning block `enable_thinking` selects.
///
/// # Errors
/// As [`render_framed`].
pub(crate) fn render_framed_with_thinking<'a>(
    turns: impl IntoIterator<Item = (&'a str, &'a str)>,
    enable_thinking: bool,
) -> Result<String, ApiError> {
    render_framed_with_thinking_and_history(turns, enable_thinking, false)
}

/// [`render_framed_with_thinking`], with earlier assistant turns rendered as
/// `restore_empty_reasoning` and `enable_thinking` select (see [`history_reasoning`]). With
/// `restore_empty_reasoning` unset it renders as [`render_framed_with_thinking`] does.
///
/// # Errors
/// As [`render_framed`].
pub(crate) fn render_framed_with_thinking_and_history<'a>(
    turns: impl IntoIterator<Item = (&'a str, &'a str)>,
    enable_thinking: bool,
    restore_empty_reasoning: bool,
) -> Result<String, ApiError> {
    let mut prompt = framed_turns(
        turns,
        history_reasoning(restore_empty_reasoning, enable_thinking),
    )?;
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

/// How one tokenizer encodes a framed Qwen prompt: as [`ChatPromptEncoder`] encodes the pieces
/// the prompt was rendered from.
pub(crate) struct QwenPromptEncoding<'t> {
    encoder: ChatPromptEncoder<'t>,
}

impl<'t> QwenPromptEncoding<'t> {
    pub(crate) fn new(tokenizer: &'t BpeTokenizer) -> Self {
        Self {
            encoder: ChatPromptEncoder::new(tokenizer),
        }
    }

    /// The ids of a framed prompt: each marker as [`ChatPromptEncoder::marker_ids`], the text
    /// between as [`ChatPromptEncoder::text_ids`]. Nothing is added or truncated.
    pub(crate) fn ids(&self, framed: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for piece in prompt_pieces::<ChatTemplateMarker>(framed) {
            match piece {
                PromptPiece::Marker(marker) => {
                    ids.extend_from_slice(self.encoder.marker_ids(marker))
                }
                PromptPiece::Text(text) => ids.extend(self.encoder.text_ids(text)),
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
