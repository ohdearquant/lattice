//! Tokenizer facade over `lattice-tokenizer`.
//!
//! Every public item keeps the path it had before the tokenizer moved into its
//! own crate. The re-exports are named rather than globbed so the facade never
//! widens the supported API.

pub mod bpe {
    #[doc(inline)]
    pub use lattice_tokenizer::bpe::{BpeTokenizer, byte_decode_token, byte_decode_token_bytes};

    #[cfg(test)]
    mod tests;
}

pub mod common {
    #[doc(inline)]
    pub use lattice_tokenizer::common::{
        EmbeddingTokenCounts, JsonValue, ThreadSafeLruCache, TokenizedInput, Tokenizer,
        load_tokenizer, tokenizer_from_json_str,
    };
}

pub mod gemma_bpe {
    #[doc(inline)]
    pub use lattice_tokenizer::gemma_bpe::{
        GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES, GEMMA4_AUDIO_HOP_LENGTH_SAMPLES,
        GEMMA4_AUDIO_MAX_SOFT_TOKENS, GEMMA4_AUDIO_MS_PER_SOFT_TOKEN,
        GEMMA4_AUDIO_SAMPLING_RATE_HZ, GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE, GemmaBpeTokenizer,
        audio_marker_expansion_tokens, audio_marker_expansion_tokens_from_samples,
        image_marker_expansion_tokens, total_audio_marker_expansion_tokens,
    };
}

pub mod sentencepiece {
    #[doc(inline)]
    pub use lattice_tokenizer::sentencepiece::{SentencePieceTokenizer, SentencePieceType};
}

pub mod wordpiece {
    #[doc(inline)]
    pub use lattice_tokenizer::wordpiece::{
        DoubleArrayTrie, TokenizedInput, Tokenizer, WordPieceTokenizer,
    };
}

#[doc(inline)]
pub use lattice_tokenizer::{
    BpeTokenizer, DoubleArrayTrie, EmbeddingTokenCounts, GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES,
    GEMMA4_AUDIO_HOP_LENGTH_SAMPLES, GEMMA4_AUDIO_MAX_SOFT_TOKENS, GEMMA4_AUDIO_MS_PER_SOFT_TOKEN,
    GEMMA4_AUDIO_SAMPLING_RATE_HZ, GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE, GemmaBpeTokenizer,
    SentencePieceTokenizer, TokenizedInput, Tokenizer, WordPieceTokenizer,
    audio_marker_expansion_tokens, audio_marker_expansion_tokens_from_samples,
    image_marker_expansion_tokens, load_tokenizer, tokenizer_from_json_str,
    total_audio_marker_expansion_tokens,
};
