//! Tokenizer implementations and shared tokenizer types.
//!
//! The crate provides WordPiece, SentencePiece, Qwen-style byte-level BPE, and
//! Gemma BPE tokenizers behind the [`Tokenizer`] trait, together with model
//! directory auto-detection and incremental detokenization.

pub mod bpe;
pub mod common;
mod detokenize;
pub mod gemma_bpe;
pub mod sentencepiece;
pub mod wordpiece;

pub use self::bpe::BpeTokenizer;
pub use self::common::{EmbeddingTokenCounts, load_tokenizer, tokenizer_from_json_str};
pub use self::gemma_bpe::{
    GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES, GEMMA4_AUDIO_HOP_LENGTH_SAMPLES,
    GEMMA4_AUDIO_MAX_SOFT_TOKENS, GEMMA4_AUDIO_MS_PER_SOFT_TOKEN, GEMMA4_AUDIO_SAMPLING_RATE_HZ,
    GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE, GemmaBpeTokenizer, audio_marker_expansion_tokens,
    audio_marker_expansion_tokens_from_samples, image_marker_expansion_tokens,
    total_audio_marker_expansion_tokens,
};
pub use self::sentencepiece::SentencePieceTokenizer;
pub use self::wordpiece::{DoubleArrayTrie, TokenizedInput, Tokenizer, WordPieceTokenizer};

/// Engine glue used by the `lattice-inference` facade.
///
/// Not part of the supported tokenizer API. The facade calls these free
/// functions by their full path; they are not re-exported by that facade.
#[doc(hidden)]
pub mod __internal {
    use std::sync::Arc;

    use crate::bpe::BpeTokenizer;
    pub use crate::detokenize::{IncrementalDetokenizer, bytes_to_unicode};

    pub const DEFAULT_BPE_MAX_SEQ_LEN: usize = crate::bpe::DEFAULT_BPE_MAX_SEQ_LEN;

    pub fn tokenize_fragments_with_inserted_ids(
        tokenizer: &BpeTokenizer,
        before: &str,
        inserted_ids: &[u32],
        after: &str,
    ) -> Vec<u32> {
        tokenizer.tokenize_fragments_with_inserted_ids(before, inserted_ids, after)
    }

    pub fn token_ids_for_content(tokenizer: &BpeTokenizer, content: &str) -> Vec<u32> {
        tokenizer.token_ids_for_content(content)
    }

    pub fn newline_boundary_table(tokenizer: &BpeTokenizer) -> Arc<[bool]> {
        tokenizer.newline_boundary_table()
    }

    pub fn max_token_byte_span(tokenizer: &BpeTokenizer) -> usize {
        tokenizer.max_token_byte_span()
    }

    /// Decode token IDs back to text using the BPE tokenizer's reverse lookup.
    pub fn decode_tokens(tokenizer: &BpeTokenizer, ids: &[u32]) -> String {
        crate::detokenize::decode_tokens(tokenizer, ids)
    }

    pub fn incremental_detokenizer_new() -> IncrementalDetokenizer {
        IncrementalDetokenizer::new()
    }

    pub fn incremental_detokenizer_push(
        detokenizer: &mut IncrementalDetokenizer,
        tokenizer: &BpeTokenizer,
        id: u32,
    ) -> String {
        detokenizer.push(tokenizer, id)
    }

    pub fn incremental_detokenizer_push_into(
        detokenizer: &mut IncrementalDetokenizer,
        tokenizer: &BpeTokenizer,
        id: u32,
        out: &mut String,
    ) {
        detokenizer.push_into(tokenizer, id, out);
    }

    pub fn incremental_detokenizer_finish(detokenizer: &mut IncrementalDetokenizer) -> String {
        detokenizer.finish()
    }
}
