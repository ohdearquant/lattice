//! The split of a generated reply into its fields, by token id (ADR-106 section i).
//!
//! The checkpoint's response template names three fields. A message opened by
//! `<|content_thinking|>` is reasoning. One opened by `<|content_text|>`, `<|content_tool_error|>`
//! or `<|content_invoke_tool_text|>` is content, and one opened by `<|content_invoke_tool_json|>` is
//! a tool call. An opener may follow `<|message_model|>` and a name, and every message closes at
//! `<|end_message|>`. The split reads the generated ids one at a time. A token outside a message,
//! the model marker and its name, and every token of a message opened by any other marker are in
//! no field, because none of the template's opening patterns matches them. Inside a message only
//! `<|end_message|>` ends it, so an opener written inside a body belongs to that body.

use lattice_inference_core::InferenceError;
use lattice_tokenizer::__internal::token_ids_for_content;
use lattice_tokenizer::BpeTokenizer;

/// The field a generated token belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReplyField {
    Reasoning,
    Content,
    ToolCall,
}

/// The ids of the markers the split reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReplyMarkers {
    pub(crate) end_message: u32,
    pub(crate) thinking: u32,
    pub(crate) text: u32,
    pub(crate) tool_error: u32,
    pub(crate) invoke_tool_text: u32,
    pub(crate) invoke_tool_json: u32,
}

impl ReplyMarkers {
    /// Resolves each marker to the one id `tokenizer` spells it with.
    ///
    /// # Errors
    ///
    /// [`InferenceError::InvalidInput`] naming a marker the tokenizer lacks or spells with more
    /// than one id.
    pub(crate) fn from_tokenizer(tokenizer: &BpeTokenizer) -> Result<Self, InferenceError> {
        let id = |name: &str| match token_ids_for_content(tokenizer, name).as_slice() {
            [id] => Ok(*id),
            [] => Err(InferenceError::InvalidInput(format!(
                "the tokenizer has no token {name}, which the reply split reads"
            ))),
            ids => Err(InferenceError::InvalidInput(format!(
                "the tokenizer spells {name} with {} ids ({ids:?}); the reply split needs one",
                ids.len()
            ))),
        };
        Ok(Self {
            end_message: id("<|end_message|>")?,
            thinking: id("<|content_thinking|>")?,
            text: id("<|content_text|>")?,
            tool_error: id("<|content_tool_error|>")?,
            invoke_tool_text: id("<|content_invoke_tool_text|>")?,
            invoke_tool_json: id("<|content_invoke_tool_json|>")?,
        })
    }

    fn opens(&self, id: u32) -> Option<ReplyField> {
        if id == self.thinking {
            Some(ReplyField::Reasoning)
        } else if id == self.text || id == self.tool_error || id == self.invoke_tool_text {
            Some(ReplyField::Content)
        } else if id == self.invoke_tool_json {
            Some(ReplyField::ToolCall)
        } else {
            None
        }
    }
}

/// The split's position in the generated ids.
pub(crate) struct ReplySplit {
    markers: ReplyMarkers,
    open: Option<ReplyField>,
}

impl ReplySplit {
    pub(crate) fn new(markers: ReplyMarkers) -> Self {
        Self {
            markers,
            open: None,
        }
    }

    /// The field whose body `id` belongs to, or `None` for a marker and for a token in no field.
    pub(crate) fn push(&mut self, id: u32) -> Option<ReplyField> {
        match self.open {
            Some(_) if id == self.markers.end_message => {
                self.open = None;
                None
            }
            Some(field) => Some(field),
            None => {
                self.open = self.markers.opens(id);
                None
            }
        }
    }
}
