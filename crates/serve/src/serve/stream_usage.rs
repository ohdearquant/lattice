//! Streaming usage policy shared by the two HTTP entry points.

use lattice_inference::StopReason;
use serde::Serialize;

use super::contract::ChatStreamOptions;

/// Shared real-router fixtures for the two HTTP entry points.
#[cfg(feature = "test-utils")]
pub mod test_support;

/// Token counts in a chat-completions response.
#[derive(Clone, Copy, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    /// Build counts from the engine's authoritative totals.
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// Controls usage fields and the terminal usage-only chunk.
#[derive(Clone, Copy)]
pub struct StreamingUsage {
    requested: bool,
}

impl StreamingUsage {
    /// Resolve the request's optional streaming usage flag.
    pub fn from_options(options: Option<&ChatStreamOptions>) -> Self {
        Self {
            requested: options
                .and_then(|options| options.include_usage)
                .unwrap_or(false),
        }
    }

    /// Return a null usage field for ordinary chunks when requested.
    pub fn chunk_field(self) -> Option<Option<Usage>> {
        self.requested.then_some(None)
    }

    /// Decide whether a completed generation can emit its final usage chunk.
    pub fn emit_final(self, reason: Option<&StopReason>) -> bool {
        self.requested && !matches!(reason, Some(StopReason::Interrupt))
    }
}

/// The usage-only chunk emitted after a completed stream.
#[derive(Serialize)]
pub struct UsageChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: [(); 0],
    usage: Usage,
}

/// Construct the common usage-only wire chunk.
pub fn usage_chunk<'a>(id: &'a str, created: u64, model: &'a str, usage: Usage) -> UsageChunk<'a> {
    UsageChunk {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [],
        usage,
    }
}
