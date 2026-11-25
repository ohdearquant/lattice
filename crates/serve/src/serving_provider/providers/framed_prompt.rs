//! A rendered chat prompt whose markers message text cannot write.
//!
//! [`PromptBuilder`] is the one writer of a [`FramedPrompt`], and its only ways to add are a
//! marker and text. Each marker is framed by the noncharacters U+FDD0 and U+FDD1, and text
//! holding either is refused, so [`prompt_pieces`] reads back exactly the markers the builder
//! wrote. A route tokenizes a marker as its id and text with no added-token matching, so a
//! marker's spelling inside a message stays text.

use crate::serve::ApiError;
use crate::serve::contract::NormalizedChatMessage;
use std::marker::PhantomData;

/// Opens a marker in the rendered prompt. U+FDD0 and U+FDD1 are Unicode noncharacters, which the
/// standard reserves for a program's internal use.
const MARKER_OPEN: char = '\u{FDD0}';
/// Closes a marker in the rendered prompt.
const MARKER_CLOSE: char = '\u{FDD1}';

/// The markers one chat template writes.
pub(crate) trait PromptMarker: Copy + 'static {
    /// Every marker of the template.
    const MARKERS: &'static [Self];

    /// The marker's token as the tokenizer spells it.
    fn spelling(self) -> &'static str;
}

/// A rendered prompt, each marker of `M` framed by U+FDD0 and U+FDD1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FramedPrompt<M> {
    framed: String,
    markers: PhantomData<fn() -> M>,
}

impl<M> FramedPrompt<M> {
    /// The framed prompt.
    pub(crate) fn framed(&self) -> &str {
        &self.framed
    }

    /// The prompt as the published template writes it, each marker as its spelling.
    #[cfg(test)]
    pub(crate) fn template_text(&self) -> String {
        template_text(&self.framed)
    }

    /// The framed prompt, which [`prompt_pieces`] reads back.
    pub(crate) fn into_framed(self) -> String {
        self.framed
    }
}

/// `framed` as the published template writes it, each marker as its spelling.
#[cfg(test)]
pub(crate) fn template_text(framed: &str) -> String {
    framed.chars().filter(|&c| !is_frame(c)).collect()
}

/// The one writer of a [`FramedPrompt`].
pub(crate) struct PromptBuilder<M> {
    framed: String,
    markers: PhantomData<fn() -> M>,
}

impl<M: PromptMarker> PromptBuilder<M> {
    pub(crate) fn new() -> Self {
        Self {
            framed: String::new(),
            markers: PhantomData,
        }
    }

    pub(crate) fn marker(&mut self, marker: M) {
        self.framed.push(MARKER_OPEN);
        self.framed.push_str(marker.spelling());
        self.framed.push(MARKER_CLOSE);
    }

    /// Appends `text`.
    ///
    /// # Errors
    /// `invalid_messages` when `text` holds U+FDD0 or U+FDD1; nothing is appended.
    pub(crate) fn text(&mut self, text: &str) -> Result<(), ApiError> {
        if let Some(c) = text.chars().find(|&c| is_frame(c)) {
            return Err(ApiError::BadRequest {
                message: format!(
                    "the prompt text contains {}, a noncharacter this model's prompt reserves \
                     for its markers",
                    codepoint(c)
                ),
                code: "invalid_messages",
            });
        }
        self.framed.push_str(text);
        Ok(())
    }

    pub(crate) fn finish(self) -> FramedPrompt<M> {
        FramedPrompt {
            framed: self.framed,
            markers: PhantomData,
        }
    }
}

fn is_frame(c: char) -> bool {
    c == MARKER_OPEN || c == MARKER_CLOSE
}

fn codepoint(c: char) -> String {
    format!("U+{:04X}", u32::from(c))
}

/// One piece of a framed prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptPiece<'a, M> {
    Marker(M),
    Text(&'a str),
}

/// The markers and text of a framed prompt, in order. A frame around anything but the spelling
/// of a marker of `M`, and a noncharacter outside a frame, are text; [`PromptBuilder`] writes
/// neither.
pub(crate) fn prompt_pieces<M: PromptMarker>(framed: &str) -> Vec<PromptPiece<'_, M>> {
    let mut pieces = Vec::new();
    let mut text_start = 0;
    let mut pos = 0;
    while let Some(offset) = framed[pos..].find(MARKER_OPEN) {
        let open = pos + offset;
        let name_start = open + MARKER_OPEN.len_utf8();
        let marker = framed[name_start..].find(MARKER_CLOSE).and_then(|len| {
            let name = &framed[name_start..name_start + len];
            M::MARKERS
                .iter()
                .copied()
                .find(|marker| marker.spelling() == name)
                .map(|marker| (marker, name_start + len + MARKER_CLOSE.len_utf8()))
        });
        match marker {
            Some((marker, end)) => {
                if text_start < open {
                    pieces.push(PromptPiece::Text(&framed[text_start..open]));
                }
                pieces.push(PromptPiece::Marker(marker));
                text_start = end;
                pos = end;
            }
            None => pos = name_start,
        }
    }
    if text_start < framed.len() {
        pieces.push(PromptPiece::Text(&framed[text_start..]));
    }
    pieces
}

/// Refuses a message whose content holds U+FDD0 or U+FDD1, naming the message. The prompt
/// builder refuses the same text; this check answers first with the message's position.
pub(crate) fn refuse_reserved_characters(
    messages: &[NormalizedChatMessage],
) -> Result<(), ApiError> {
    for (index, message) in messages.iter().enumerate() {
        if let Some(c) = message.content.chars().find(|&c| is_frame(c)) {
            return Err(ApiError::BadRequest {
                message: format!(
                    "messages[{index}] content contains {}, a noncharacter this model's prompt \
                     reserves for its markers",
                    codepoint(c)
                ),
                code: "invalid_messages",
            });
        }
    }
    Ok(())
}
