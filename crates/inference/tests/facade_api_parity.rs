use std::collections::HashMap;
use std::sync::Arc;

use lattice_inference::error::InferenceError as FacadeInferenceError;
use lattice_inference::tokenizer::bpe::{
    BpeTokenizer as FacadeBpeTokenizer, byte_decode_token as facade_byte_decode_token,
    byte_decode_token_bytes as facade_byte_decode_token_bytes,
};
use lattice_inference::tokenizer::common::{
    EmbeddingTokenCounts as FacadeEmbeddingTokenCounts, JsonValue as FacadeJsonValue,
    ThreadSafeLruCache as FacadeThreadSafeLruCache, TokenizedInput as FacadeTokenizedInput,
    Tokenizer as FacadeTokenizer, load_tokenizer as facade_load_tokenizer,
    tokenizer_from_json_str as facade_tokenizer_from_json_str,
};
use lattice_inference::tokenizer::gemma_bpe::{
    GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES as FACADE_GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES,
    GEMMA4_AUDIO_HOP_LENGTH_SAMPLES as FACADE_GEMMA4_AUDIO_HOP_LENGTH_SAMPLES,
    GEMMA4_AUDIO_MAX_SOFT_TOKENS as FACADE_GEMMA4_AUDIO_MAX_SOFT_TOKENS,
    GEMMA4_AUDIO_MS_PER_SOFT_TOKEN as FACADE_GEMMA4_AUDIO_MS_PER_SOFT_TOKEN,
    GEMMA4_AUDIO_SAMPLING_RATE_HZ as FACADE_GEMMA4_AUDIO_SAMPLING_RATE_HZ,
    GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE as FACADE_GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE,
    GemmaBpeTokenizer as FacadeGemmaBpeTokenizer,
    audio_marker_expansion_tokens as facade_audio_marker_expansion_tokens,
    audio_marker_expansion_tokens_from_samples as facade_audio_marker_expansion_tokens_from_samples,
    image_marker_expansion_tokens as facade_image_marker_expansion_tokens,
    total_audio_marker_expansion_tokens as facade_total_audio_marker_expansion_tokens,
};
use lattice_inference::tokenizer::sentencepiece::{
    SentencePieceTokenizer as FacadeSentencePieceTokenizer,
    SentencePieceType as FacadeSentencePieceType,
};
use lattice_inference::tokenizer::wordpiece::{
    DoubleArrayTrie as FacadeDoubleArrayTrie, TokenizedInput as FacadeWordPieceTokenizedInput,
    Tokenizer as FacadeWordPieceTokenizerTrait, WordPieceTokenizer as FacadeWordPieceTokenizer,
};
use lattice_inference::{
    BpeTokenizer as RootBpeTokenizer, EmbeddingTokenCounts as RootEmbeddingTokenCounts,
    GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES as ROOT_GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES,
    GEMMA4_AUDIO_HOP_LENGTH_SAMPLES as ROOT_GEMMA4_AUDIO_HOP_LENGTH_SAMPLES,
    GEMMA4_AUDIO_MAX_SOFT_TOKENS as ROOT_GEMMA4_AUDIO_MAX_SOFT_TOKENS,
    GEMMA4_AUDIO_MS_PER_SOFT_TOKEN as ROOT_GEMMA4_AUDIO_MS_PER_SOFT_TOKEN,
    GEMMA4_AUDIO_SAMPLING_RATE_HZ as ROOT_GEMMA4_AUDIO_SAMPLING_RATE_HZ,
    GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE as ROOT_GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE,
    GemmaBpeTokenizer as RootGemmaBpeTokenizer,
    SentencePieceTokenizer as RootSentencePieceTokenizer, TokenizedInput as RootTokenizedInput,
    Tokenizer as RootTokenizer, WordPieceTokenizer as RootWordPieceTokenizer,
    audio_marker_expansion_tokens as root_audio_marker_expansion_tokens,
    audio_marker_expansion_tokens_from_samples as root_audio_marker_expansion_tokens_from_samples,
    image_marker_expansion_tokens as root_image_marker_expansion_tokens,
    load_tokenizer as root_load_tokenizer, tokenizer_from_json_str as root_tokenizer_from_json_str,
    total_audio_marker_expansion_tokens as root_total_audio_marker_expansion_tokens,
};
use lattice_inference_core::InferenceError as CoreInferenceError;
use lattice_tokenizer::BpeTokenizer;
use lattice_tokenizer::bpe::{
    BpeTokenizer as TokenizerBpeTokenizer, byte_decode_token as tokenizer_byte_decode_token,
    byte_decode_token_bytes as tokenizer_byte_decode_token_bytes,
};
use lattice_tokenizer::common::{
    EmbeddingTokenCounts as TokenizerEmbeddingTokenCounts, JsonValue as TokenizerJsonValue,
    ThreadSafeLruCache as TokenizerThreadSafeLruCache, TokenizedInput as TokenizerTokenizedInput,
    Tokenizer as TokenizerTrait, load_tokenizer as tokenizer_load_tokenizer,
    tokenizer_from_json_str,
};
use lattice_tokenizer::common::{JsonValue, ThreadSafeLruCache};
use lattice_tokenizer::gemma_bpe::{
    GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES, GEMMA4_AUDIO_HOP_LENGTH_SAMPLES,
    GEMMA4_AUDIO_MAX_SOFT_TOKENS, GEMMA4_AUDIO_MS_PER_SOFT_TOKEN, GEMMA4_AUDIO_SAMPLING_RATE_HZ,
    GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE, GemmaBpeTokenizer, audio_marker_expansion_tokens,
    audio_marker_expansion_tokens_from_samples, image_marker_expansion_tokens,
    total_audio_marker_expansion_tokens,
};
use lattice_tokenizer::sentencepiece::{SentencePieceTokenizer, SentencePieceType};
use lattice_tokenizer::wordpiece::{DoubleArrayTrie, WordPieceTokenizer};

#[test]
fn inference_error_is_the_same_type_at_both_paths() {
    fn accepts_facade(error: FacadeInferenceError) -> CoreInferenceError {
        error
    }

    fn accepts_core(error: CoreInferenceError) -> FacadeInferenceError {
        error
    }

    let facade = FacadeInferenceError::InvalidInput("facade".to_string());
    let core = accepts_facade(facade);
    let _back: FacadeInferenceError = accepts_core(core);

    let root: lattice_inference::InferenceError =
        FacadeInferenceError::InvalidInput("root".to_string());
    let _: CoreInferenceError = root;
}

#[test]
fn crate_root_exports_are_the_same_items_at_both_paths() {
    type RootLoad = fn(&std::path::Path) -> Result<Box<dyn RootTokenizer>, FacadeInferenceError>;
    type RootFromJson = fn(&str, usize) -> Result<Box<dyn RootTokenizer>, FacadeInferenceError>;

    fn bpe(root: RootBpeTokenizer) -> TokenizerBpeTokenizer {
        root
    }

    fn wordpiece(root: RootWordPieceTokenizer) -> WordPieceTokenizer {
        root
    }

    fn sentencepiece(root: RootSentencePieceTokenizer) -> SentencePieceTokenizer {
        root
    }

    fn gemma(root: RootGemmaBpeTokenizer) -> GemmaBpeTokenizer {
        root
    }

    fn tokenized_input(root: RootTokenizedInput) -> TokenizerTokenizedInput {
        root
    }

    fn counts(root: RootEmbeddingTokenCounts) -> TokenizerEmbeddingTokenCounts {
        root
    }

    let _: fn(RootBpeTokenizer) -> TokenizerBpeTokenizer = bpe;
    let _: fn(RootWordPieceTokenizer) -> WordPieceTokenizer = wordpiece;
    let _: fn(RootSentencePieceTokenizer) -> SentencePieceTokenizer = sentencepiece;
    let _: fn(RootGemmaBpeTokenizer) -> GemmaBpeTokenizer = gemma;
    let _: fn(RootTokenizedInput) -> TokenizerTokenizedInput = tokenized_input;
    let _: fn(RootEmbeddingTokenCounts) -> TokenizerEmbeddingTokenCounts = counts;

    fn accepts_root_tokenizer(_tokenizer: &dyn RootTokenizer) {}
    fn accepts_tokenizer(_tokenizer: &dyn TokenizerTrait) {}

    fn both_traits(tokenizer: &dyn TokenizerTrait) {
        accepts_root_tokenizer(tokenizer);
        accepts_tokenizer(tokenizer);
    }
    let _ = both_traits;

    let _: RootLoad = root_load_tokenizer;
    let _: RootFromJson = root_tokenizer_from_json_str;
    let _: fn(usize) -> usize = root_image_marker_expansion_tokens;
    let _: fn(u32) -> u32 = root_audio_marker_expansion_tokens_from_samples;
    let _: fn(u32) -> u32 = root_audio_marker_expansion_tokens;
    let _: fn(&[u32]) -> usize = root_total_audio_marker_expansion_tokens;

    assert_eq!(
        ROOT_GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE,
        GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE
    );
    assert_eq!(
        ROOT_GEMMA4_AUDIO_MS_PER_SOFT_TOKEN,
        GEMMA4_AUDIO_MS_PER_SOFT_TOKEN
    );
    assert_eq!(
        ROOT_GEMMA4_AUDIO_MAX_SOFT_TOKENS,
        GEMMA4_AUDIO_MAX_SOFT_TOKENS
    );
    assert_eq!(
        ROOT_GEMMA4_AUDIO_SAMPLING_RATE_HZ,
        GEMMA4_AUDIO_SAMPLING_RATE_HZ
    );
    assert_eq!(
        ROOT_GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES,
        GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES
    );
    assert_eq!(
        ROOT_GEMMA4_AUDIO_HOP_LENGTH_SAMPLES,
        GEMMA4_AUDIO_HOP_LENGTH_SAMPLES
    );
}

#[test]
fn tokenizer_types_are_the_same_type_at_both_paths() {
    type FacadeCache = FacadeThreadSafeLruCache<String, Vec<u32>>;
    type TokenizerCache = TokenizerThreadSafeLruCache<String, Vec<u32>>;

    fn facade_to_new(tokenizer: FacadeBpeTokenizer) -> TokenizerBpeTokenizer {
        tokenizer
    }

    fn new_to_facade(tokenizer: TokenizerBpeTokenizer) -> FacadeBpeTokenizer {
        tokenizer
    }

    let _: fn(FacadeBpeTokenizer) -> TokenizerBpeTokenizer = facade_to_new;
    let _: fn(TokenizerBpeTokenizer) -> FacadeBpeTokenizer = new_to_facade;

    fn wordpiece_facade_to_new(tokenizer: FacadeWordPieceTokenizer) -> WordPieceTokenizer {
        tokenizer
    }

    fn wordpiece_new_to_facade(tokenizer: WordPieceTokenizer) -> FacadeWordPieceTokenizer {
        tokenizer
    }

    let _: fn(FacadeWordPieceTokenizer) -> WordPieceTokenizer = wordpiece_facade_to_new;
    let _: fn(WordPieceTokenizer) -> FacadeWordPieceTokenizer = wordpiece_new_to_facade;

    fn sentencepiece_facade_to_new(
        tokenizer: FacadeSentencePieceTokenizer,
    ) -> SentencePieceTokenizer {
        tokenizer
    }

    fn sentencepiece_new_to_facade(
        tokenizer: SentencePieceTokenizer,
    ) -> FacadeSentencePieceTokenizer {
        tokenizer
    }

    let _: fn(FacadeSentencePieceTokenizer) -> SentencePieceTokenizer = sentencepiece_facade_to_new;
    let _: fn(SentencePieceTokenizer) -> FacadeSentencePieceTokenizer = sentencepiece_new_to_facade;

    fn gemma_facade_to_new(tokenizer: FacadeGemmaBpeTokenizer) -> GemmaBpeTokenizer {
        tokenizer
    }

    fn gemma_new_to_facade(tokenizer: GemmaBpeTokenizer) -> FacadeGemmaBpeTokenizer {
        tokenizer
    }

    let _: fn(FacadeGemmaBpeTokenizer) -> GemmaBpeTokenizer = gemma_facade_to_new;
    let _: fn(GemmaBpeTokenizer) -> FacadeGemmaBpeTokenizer = gemma_new_to_facade;

    fn trie_facade_to_new(trie: FacadeDoubleArrayTrie) -> DoubleArrayTrie {
        trie
    }

    fn trie_new_to_facade(trie: DoubleArrayTrie) -> FacadeDoubleArrayTrie {
        trie
    }

    let _: fn(FacadeDoubleArrayTrie) -> DoubleArrayTrie = trie_facade_to_new;
    let _: fn(DoubleArrayTrie) -> FacadeDoubleArrayTrie = trie_new_to_facade;

    fn tokenized_facade_to_new(input: FacadeTokenizedInput) -> TokenizerTokenizedInput {
        input
    }

    fn tokenized_new_to_facade(input: TokenizerTokenizedInput) -> FacadeTokenizedInput {
        input
    }

    let _: fn(FacadeTokenizedInput) -> TokenizerTokenizedInput = tokenized_facade_to_new;
    let _: fn(TokenizerTokenizedInput) -> FacadeTokenizedInput = tokenized_new_to_facade;

    fn wordpiece_tokenized_facade_to_new(
        input: FacadeWordPieceTokenizedInput,
    ) -> TokenizerTokenizedInput {
        input
    }

    let _: fn(FacadeWordPieceTokenizedInput) -> TokenizerTokenizedInput =
        wordpiece_tokenized_facade_to_new;

    fn counts_facade_to_new(counts: FacadeEmbeddingTokenCounts) -> TokenizerEmbeddingTokenCounts {
        counts
    }

    fn counts_new_to_facade(counts: TokenizerEmbeddingTokenCounts) -> FacadeEmbeddingTokenCounts {
        counts
    }

    let _: fn(FacadeEmbeddingTokenCounts) -> TokenizerEmbeddingTokenCounts = counts_facade_to_new;
    let _: fn(TokenizerEmbeddingTokenCounts) -> FacadeEmbeddingTokenCounts = counts_new_to_facade;

    fn json_facade_to_new(value: FacadeJsonValue) -> TokenizerJsonValue {
        value
    }

    fn json_new_to_facade(value: TokenizerJsonValue) -> FacadeJsonValue {
        value
    }

    let _: fn(FacadeJsonValue) -> TokenizerJsonValue = json_facade_to_new;
    let _: fn(TokenizerJsonValue) -> FacadeJsonValue = json_new_to_facade;

    fn cache_facade_to_new(cache: FacadeCache) -> TokenizerCache {
        cache
    }

    fn cache_new_to_facade(cache: TokenizerCache) -> FacadeCache {
        cache
    }

    let _: fn(FacadeCache) -> TokenizerCache = cache_facade_to_new;
    let _: fn(TokenizerCache) -> FacadeCache = cache_new_to_facade;

    fn enum_facade_to_new(value: FacadeSentencePieceType) -> SentencePieceType {
        value
    }

    fn enum_new_to_facade(value: SentencePieceType) -> FacadeSentencePieceType {
        value
    }

    let _: fn(FacadeSentencePieceType) -> SentencePieceType = enum_facade_to_new;
    let _: fn(SentencePieceType) -> FacadeSentencePieceType = enum_new_to_facade;
}

#[test]
fn tokenizer_traits_are_the_same_trait_at_both_paths() {
    fn accepts_facade_tokenizer(_tokenizer: &dyn FacadeTokenizer) {}
    fn accepts_wordpiece_facade_tokenizer(_tokenizer: &dyn FacadeWordPieceTokenizerTrait) {}

    let accepts_new_tokenizer = |tokenizer: &dyn TokenizerTrait| {
        accepts_facade_tokenizer(tokenizer);
        accepts_wordpiece_facade_tokenizer(tokenizer);
    };

    let tokenizer = TokenizerBpeTokenizer::from_vocab_and_merges(
        HashMap::from([("a".to_string(), 0u32)]),
        Vec::new(),
    )
    .expect("synthetic tokenizer must build");
    accept_new_tokenizer_dyn(&tokenizer);
    let _: &dyn FacadeTokenizer = &tokenizer as &dyn FacadeTokenizer;
    let _: &dyn FacadeWordPieceTokenizerTrait = &tokenizer as &dyn FacadeWordPieceTokenizerTrait;
    let _: &dyn TokenizerTrait = &tokenizer;
    let _ = accepts_new_tokenizer;
}

fn accept_new_tokenizer_dyn(tokenizer: &dyn TokenizerTrait) {
    let _: &dyn FacadeTokenizer = tokenizer;
}

#[test]
fn tokenizer_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<BpeTokenizer>();
    assert_send_sync::<WordPieceTokenizer>();
    assert_send_sync::<SentencePieceTokenizer>();
    assert_send_sync::<GemmaBpeTokenizer>();
    assert_send_sync::<TokenizerTokenizedInput>();
    assert_send_sync::<TokenizerEmbeddingTokenCounts>();
    assert_send_sync::<DoubleArrayTrie>();
    assert_send_sync::<JsonValue>();
    assert_send_sync::<ThreadSafeLruCache<String, Vec<u32>>>();
    assert_send_sync::<Arc<[bool]>>();
}

#[test]
fn inference_error_is_std_error() {
    fn assert_std_error<T: std::error::Error + Send + Sync + 'static>() {}

    assert_std_error::<CoreInferenceError>();
}

#[test]
fn functions_and_constants_are_the_same_items_at_both_paths() {
    type FacadeLoad =
        fn(&std::path::Path) -> Result<Box<dyn FacadeTokenizer>, FacadeInferenceError>;
    type TokenizerLoad =
        fn(&std::path::Path) -> Result<Box<dyn TokenizerTrait>, CoreInferenceError>;
    type FacadeFromJson = fn(&str, usize) -> Result<Box<dyn FacadeTokenizer>, FacadeInferenceError>;
    type TokenizerFromJson = fn(&str, usize) -> Result<Box<dyn TokenizerTrait>, CoreInferenceError>;

    let _: fn(&str) -> Vec<u8> = facade_byte_decode_token_bytes;
    let _: fn(&str) -> Vec<u8> = tokenizer_byte_decode_token_bytes;
    let _: fn(&str) -> String = facade_byte_decode_token;
    let _: fn(&str) -> String = tokenizer_byte_decode_token;
    let _: FacadeLoad = facade_load_tokenizer;
    let _: TokenizerLoad = tokenizer_load_tokenizer;
    let _: FacadeFromJson = facade_tokenizer_from_json_str;
    let _: TokenizerFromJson = tokenizer_from_json_str;
    let _: fn(usize) -> usize = facade_image_marker_expansion_tokens;
    let _: fn(usize) -> usize = image_marker_expansion_tokens;
    let _: fn(u32) -> u32 = facade_audio_marker_expansion_tokens_from_samples;
    let _: fn(u32) -> u32 = audio_marker_expansion_tokens_from_samples;
    let _: fn(u32) -> u32 = facade_audio_marker_expansion_tokens;
    let _: fn(u32) -> u32 = audio_marker_expansion_tokens;
    let _: fn(&[u32]) -> usize = facade_total_audio_marker_expansion_tokens;
    let _: fn(&[u32]) -> usize = total_audio_marker_expansion_tokens;

    assert_eq!(
        FACADE_GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE,
        GEMMA4_IMAGE_SOFT_TOKENS_PER_IMAGE
    );
    assert_eq!(
        FACADE_GEMMA4_AUDIO_MS_PER_SOFT_TOKEN,
        GEMMA4_AUDIO_MS_PER_SOFT_TOKEN
    );
    assert_eq!(
        FACADE_GEMMA4_AUDIO_MAX_SOFT_TOKENS,
        GEMMA4_AUDIO_MAX_SOFT_TOKENS
    );
    assert_eq!(
        FACADE_GEMMA4_AUDIO_SAMPLING_RATE_HZ,
        GEMMA4_AUDIO_SAMPLING_RATE_HZ
    );
    assert_eq!(
        FACADE_GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES,
        GEMMA4_AUDIO_FRAME_LENGTH_SAMPLES
    );
    assert_eq!(
        FACADE_GEMMA4_AUDIO_HOP_LENGTH_SAMPLES,
        GEMMA4_AUDIO_HOP_LENGTH_SAMPLES
    );
}
