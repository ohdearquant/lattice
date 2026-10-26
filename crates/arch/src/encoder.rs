//! Text encoder contract (ADR-103 Amendment 21 item 3): a loaded embedding model behind one
//! object-safe trait, shared as `Arc<dyn Encoder>`.
//!
//! **Unstable**: this module is the supported encoder contract. A model family implements
//! [`Encoder`]; a consumer loads it once and shares the handle across threads. The records the
//! trait exchanges have private fields and are built with their constructors, so an
//! implementation outside this crate can return every value the trait asks for.
//!
//! The contract covers execution only. Choosing a role or instruction prefix, request caps,
//! scheduling, caching and remote providers stay with the embedding service above this boundary:
//! [`Encoder::encode_texts`] receives text that is already prepared and embeds it as given.

use crate::error::InferenceError;
use std::num::NonZeroUsize;

/// A loaded text encoder.
///
/// `Send + Sync` because the handle is shared: a consumer holds one `Arc<dyn Encoder>` and calls
/// it from several threads at once. A backend that is not thread-safe needs an owner that
/// serializes access to it; the bounds are never asserted for a type that does not meet them.
pub trait Encoder: Send + Sync {
    /// What this loaded handle accepts, including its effective token limit.
    fn capabilities(&self) -> &EncoderCapabilities;

    /// Length of every vector [`Encoder::encode_texts`] returns. Equals
    /// `self.identity().output_dimension()`.
    fn output_dimension(&self) -> NonZeroUsize;

    /// What determines this handle's vectors: checkpoint, tokenizer, prompt convention, pooling,
    /// normalization, output dimension and execution variant.
    fn identity(&self) -> &EncoderIdentity;

    /// One [`EncodedText`] per input, in input order, each vector exactly
    /// [`Encoder::output_dimension`] long.
    ///
    /// Each input is embedded as given. Input longer than the capability's token limit is
    /// truncated, never refused, and each output's [`TokenUsage`] reports the counts. An error
    /// covers the whole call: an implementation never returns a partial batch. An empty slice
    /// returns an empty vector.
    fn encode_texts(&self, prepared: &[&str]) -> Result<Vec<EncodedText>, InferenceError>;
}

/// What a loaded encoder accepts.
///
/// Text is the only input the [`Encoder`] trait carries; image and document execution use their
/// own extension interfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EncoderCapabilities {
    max_input_tokens: Option<NonZeroUsize>,
}

impl EncoderCapabilities {
    /// A text encoder that truncates each input to at most `max_input_tokens` tokens, special
    /// tokens included. `None` when the handle applies no limit of its own.
    pub const fn text(max_input_tokens: Option<NonZeroUsize>) -> Self {
        Self { max_input_tokens }
    }

    /// The token limit an input is truncated to, special tokens included, if the handle applies
    /// one.
    pub const fn max_input_tokens(&self) -> Option<NonZeroUsize> {
        self.max_input_tokens
    }
}

/// Token counts of one embedded input: the sequence the model embedded, not the padded batch.
///
/// Both counts include the special tokens and any instruction the prepared text carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenUsage {
    before_truncation: usize,
    embedded: usize,
}

impl TokenUsage {
    /// Counts of one input. `None` when `embedded` exceeds `before_truncation`, which no
    /// truncation can produce.
    pub const fn new(before_truncation: usize, embedded: usize) -> Option<Self> {
        if embedded > before_truncation {
            None
        } else {
            Some(Self {
                before_truncation,
                embedded,
            })
        }
    }

    /// Length of the tokenized input before the limit was applied.
    pub const fn before_truncation(&self) -> usize {
        self.before_truncation
    }

    /// Length of the sequence the model embedded.
    pub const fn embedded(&self) -> usize {
        self.embedded
    }

    /// Tokens the limit removed.
    pub const fn truncated(&self) -> usize {
        self.before_truncation - self.embedded
    }
}

/// One input's embedding and its token counts.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedText {
    vector: Vec<f32>,
    usage: TokenUsage,
}

impl EncodedText {
    /// Pairs one input's vector with the counts of the sequence it was computed from.
    pub fn new(vector: Vec<f32>, usage: TokenUsage) -> Self {
        Self { vector, usage }
    }

    /// The embedding.
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }

    /// Token counts of the embedded sequence.
    pub const fn usage(&self) -> TokenUsage {
        self.usage
    }

    /// The embedding, by value.
    pub fn into_vector(self) -> Vec<f32> {
        self.vector
    }
}

/// Where an artifact came from, as the loader established it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Provenance {
    /// An immutable revision or content digest of the artifact.
    Pinned(String),
    /// The loader could not establish one. A directory path or a load time is never put in its
    /// place.
    Unavailable,
}

/// How a sequence's token states become one vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Pooling {
    /// Masked mean over every embedded token.
    Mean,
    /// The state of the first token.
    FirstToken,
    /// The state of the last token.
    LastToken,
}

/// Normalization applied to the pooled vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Normalization {
    /// The vector is returned as pooled.
    None,
    /// The vector is scaled to unit L2 norm.
    L2,
}

/// Everything that determines a loaded encoder's vectors.
///
/// Two handles with equal identities produce the same vector for the same prepared text only
/// when both provenance fields are [`Provenance::Pinned`]; see
/// [`EncoderIdentity::has_pinned_provenance`]. A cache that outlives one loaded handle keys on
/// the identity together with the role or prompt it prepared the text with.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EncoderIdentity {
    checkpoint: Provenance,
    tokenizer: Provenance,
    prompt_convention: String,
    pooling: Pooling,
    normalization: Normalization,
    output_dimension: NonZeroUsize,
    variant: String,
}

impl EncoderIdentity {
    /// An identity with unavailable provenance, the `raw` prompt convention (prepared text is
    /// embedded as given) and an empty execution variant.
    pub fn new(
        pooling: Pooling,
        normalization: Normalization,
        output_dimension: NonZeroUsize,
    ) -> Self {
        Self {
            checkpoint: Provenance::Unavailable,
            tokenizer: Provenance::Unavailable,
            prompt_convention: "raw".to_string(),
            pooling,
            normalization,
            output_dimension,
            variant: String::new(),
        }
    }

    /// Sets the weights' revision or content digest.
    pub fn with_checkpoint(mut self, checkpoint: Provenance) -> Self {
        self.checkpoint = checkpoint;
        self
    }

    /// Sets the tokenizer artifact's revision or content digest, which covers its normalization
    /// and special-token semantics.
    pub fn with_tokenizer(mut self, tokenizer: Provenance) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Sets the versioned name of the template the handle applies to its input before
    /// tokenizing; `raw` when it applies none.
    pub fn with_prompt_convention(mut self, prompt_convention: impl Into<String>) -> Self {
        self.prompt_convention = prompt_convention.into();
        self
    }

    /// Sets the execution and precision variant, for example the backend and weight format,
    /// when it changes the vectors.
    pub fn with_variant(mut self, variant: impl Into<String>) -> Self {
        self.variant = variant.into();
        self
    }

    /// The weights' revision or content digest.
    pub fn checkpoint(&self) -> &Provenance {
        &self.checkpoint
    }

    /// The tokenizer artifact's revision or content digest.
    pub fn tokenizer(&self) -> &Provenance {
        &self.tokenizer
    }

    /// The template the handle applies to its input; `raw` when it applies none.
    pub fn prompt_convention(&self) -> &str {
        &self.prompt_convention
    }

    /// How token states are pooled.
    pub const fn pooling(&self) -> Pooling {
        self.pooling
    }

    /// Normalization of the pooled vector.
    pub const fn normalization(&self) -> Normalization {
        self.normalization
    }

    /// Length of each output vector.
    pub const fn output_dimension(&self) -> NonZeroUsize {
        self.output_dimension
    }

    /// The execution and precision variant.
    pub fn variant(&self) -> &str {
        &self.variant
    }

    /// Whether both the checkpoint and the tokenizer carry a pinned revision or digest. Only
    /// then can an identity stand for the same vectors across loads; with unavailable
    /// provenance it describes one loaded handle.
    pub fn has_pinned_provenance(&self) -> bool {
        matches!(self.checkpoint, Provenance::Pinned(_))
            && matches!(self.tokenizer, Provenance::Pinned(_))
    }
}

#[cfg(test)]
mod tests;
