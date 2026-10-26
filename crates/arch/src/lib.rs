//! The model interface for Lattice inference (ADR-103 Amendment 21): the generation types every
//! decoder path shares ([`generation::GenerateConfig`], [`generation::GenerateOutput`],
//! [`generation::TokenLogprob`], [`generation::TopLogprob`], [`stop_reason::StopReason`]) and
//! [`sampling`]. It also holds the decoder session vocabulary and the shared generation driver
//! (ADR-090), which `lattice-inference` reaches through `__internal`; they are not yet part of the
//! supported API.
//!
//! Until each model family moves to its own crate, this crate also holds the model configurations:
//! the parsed and validated `config.json` of the Qwen3.5 hybrid models, Gemma 4, EmbeddingGemma 2
//! and BitNet b1.58, with the validation bounds those parsers enforce, the Qwen3 embedding model
//! configuration, and the Qwen3.5 checkpoint tensor names a configuration implies.
//!
//! `lattice-inference` re-exports the supported items at their original paths: the generation types
//! at `lattice_inference::{generation, sampling, stop_reason}` and its crate root, the
//! configurations at
//! `lattice_inference::model::{qwen35_config, gemma4_config, embeddinggemma2_config, bitnet_config}`,
//! `QwenConfig` at `lattice_inference::model::qwen::QwenConfig`, and the tensor-name functions
//! at `lattice_inference::model::qwen35`.

mod error {
    pub use lattice_inference_core::InferenceError;
}

pub mod bitnet_config;
mod decode_budget;
mod decoder;
pub mod embeddinggemma2_config;
pub mod gemma4_config;
pub mod generation;
pub mod qwen35_config;
pub mod qwen35_tensors;
pub mod qwen_config;
pub mod sampling;
pub mod stop_reason;
mod stop_strings;

/// Configuration and generation glue used by the `lattice-inference` facade.
///
/// Not part of the supported API. The facade reaches these items by their full path and never
/// re-exports them publicly.
#[doc(hidden)]
pub mod __internal {
    use lattice_inference_core::InferenceError;

    pub use crate::decode_budget::decode_cap;
    pub use crate::generation::{
        DecodePolicy, StepOutcome, StopCheckOutcome, truncate_token_logprobs_to_retained_text,
    };
    pub use crate::sampling::{
        argmax_f32_first_wins, compute_step_logprobs, penalized_logit, sample_full_logits,
        temperature_degenerate, uniform_f32_from_u64, xorshift64_next,
    };
    pub use crate::stop_strings::{StopStringMatcher, earliest_stop_match};

    /// The decoder session vocabulary and the shared driver (ADR-090), which the facade's
    /// concrete sessions implement and run.
    pub mod decoder {
        pub use crate::decoder::{
            AcceptedToken, Cancellation, DecoderSession, ExecutionCapabilities, FinishDisposition,
            GrammarMaskFn, MetadataRequest, PredictionError, PredictionId, PredictionLedger,
            SelectOutcome, SelectionCandidate, SelectionRequest, SpeculativeSession,
            SpeculativeTrace, StepStamp, TokenMetadata, VerifiedRound, driver,
        };
    }

    /// Checked doubling of a config-derived dimension, with the Qwen config error on overflow.
    pub fn checked_double(value: usize, what: &str) -> Result<usize, InferenceError> {
        crate::qwen35_config::checked_double(value, what)
    }
}
