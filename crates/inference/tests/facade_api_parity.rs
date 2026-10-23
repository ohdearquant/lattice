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

mod grammar_parity {
    use lattice_grammar as extracted;
    use lattice_inference::grammar as facade;
    use std::marker::PhantomData;

    fn same_type<T>(_: PhantomData<T>, _: PhantomData<T>) {}

    fn same_function<F>(_: F, _: F) {}

    #[test]
    fn grammar_types_are_the_same_type_at_both_paths() {
        same_type(
            PhantomData::<facade::engine::MaskProfile>,
            PhantomData::<extracted::engine::MaskProfile>,
        );
        same_type(
            PhantomData::<facade::engine::BuildProfile>,
            PhantomData::<extracted::engine::BuildProfile>,
        );
        same_type(
            PhantomData::<facade::engine::GrammarError>,
            PhantomData::<extracted::engine::GrammarError>,
        );
        same_type(
            PhantomData::<facade::engine::GrammarEngine>,
            PhantomData::<extracted::engine::GrammarEngine>,
        );
        same_type(
            PhantomData::<facade::gbnf::GbnfError>,
            PhantomData::<extracted::gbnf::GbnfError>,
        );
        same_type(
            PhantomData::<facade::json_schema::SchemaError>,
            PhantomData::<extracted::json_schema::SchemaError>,
        );
        same_type(
            PhantomData::<facade::pda::Alt>,
            PhantomData::<extracted::pda::Alt>,
        );
        same_type(
            PhantomData::<facade::pda::Symbol>,
            PhantomData::<extracted::pda::Symbol>,
        );
        same_type(
            PhantomData::<facade::pda::Rule>,
            PhantomData::<extracted::pda::Rule>,
        );
        same_type(
            PhantomData::<facade::pda::CompiledGrammar>,
            PhantomData::<extracted::pda::CompiledGrammar>,
        );
        same_type(
            PhantomData::<facade::pda::StackFrame>,
            PhantomData::<extracted::pda::StackFrame>,
        );
        same_type(
            PhantomData::<facade::pda::GrammarState>,
            PhantomData::<extracted::pda::GrammarState>,
        );
        same_type(
            PhantomData::<facade::pda::StackLimitError>,
            PhantomData::<extracted::pda::StackLimitError>,
        );
        same_type(
            PhantomData::<facade::pda::StepResult>,
            PhantomData::<extracted::pda::StepResult>,
        );
        same_type(
            PhantomData::<facade::pda::SimResult>,
            PhantomData::<extracted::pda::SimResult>,
        );
        same_type(
            PhantomData::<facade::pda::BuilderError>,
            PhantomData::<extracted::pda::BuilderError>,
        );
        same_type(
            PhantomData::<facade::pda::GrammarBuilder>,
            PhantomData::<extracted::pda::GrammarBuilder>,
        );
        same_type(
            PhantomData::<facade::spec::GrammarSpec>,
            PhantomData::<extracted::spec::GrammarSpec>,
        );
        same_type(
            PhantomData::<facade::trie::ByteTrie>,
            PhantomData::<extracted::trie::ByteTrie>,
        );
        same_type(
            PhantomData::<facade::vocab_partition::VocabPartition>,
            PhantomData::<extracted::vocab_partition::VocabPartition>,
        );
        same_type(
            PhantomData::<facade::GrammarEngine>,
            PhantomData::<extracted::GrammarEngine>,
        );
        same_type(
            PhantomData::<facade::GrammarError>,
            PhantomData::<extracted::GrammarError>,
        );
        same_type(
            PhantomData::<facade::GrammarSpec>,
            PhantomData::<extracted::GrammarSpec>,
        );
    }

    #[test]
    fn grammar_functions_are_the_same_items_at_both_paths() {
        same_function(
            facade::engine::enable_mask_profiling,
            extracted::engine::enable_mask_profiling,
        );
        same_function(
            facade::engine::take_mask_profile,
            extracted::engine::take_mask_profile,
        );
        same_function(
            facade::engine::context_recheck_simulated,
            extracted::engine::context_recheck_simulated,
        );
        same_function(
            facade::engine::context_recheck_candidates,
            extracted::engine::context_recheck_candidates,
        );
        same_function(
            facade::engine::last_build_profile,
            extracted::engine::last_build_profile,
        );
        same_function(
            facade::engine::probe_reachable_states,
            extracted::engine::probe_reachable_states,
        );
        same_function(facade::gbnf::parse_gbnf, extracted::gbnf::parse_gbnf);
        same_function(
            facade::json_schema::compile_json_schema,
            extracted::json_schema::compile_json_schema,
        );
        same_function(
            facade::json_schema::compile,
            extracted::json_schema::compile,
        );
        same_function(facade::pda::advance_byte, extracted::pda::advance_byte);
        same_function(facade::pda::simulate_token, extracted::pda::simulate_token);
    }

    #[test]
    fn grammar_constants_are_equal_at_both_paths() {
        assert_eq!(facade::pda::UNCHOSEN_ALT, extracted::pda::UNCHOSEN_ALT);
        assert_eq!(
            facade::vocab_partition::MAX_GRAMMAR_STATES,
            extracted::vocab_partition::MAX_GRAMMAR_STATES
        );
    }

    #[test]
    fn grammar_error_conversions_and_thread_safety_are_preserved() {
        fn send_sync<T: Send + Sync>() {}
        fn error<T: std::error::Error + Send + Sync + 'static>() {}
        fn from<T: From<U>, U>() {}

        send_sync::<facade::GrammarEngine>();
        send_sync::<facade::GrammarSpec>();
        send_sync::<facade::pda::CompiledGrammar>();
        send_sync::<facade::pda::GrammarState>();
        send_sync::<facade::trie::ByteTrie>();
        send_sync::<facade::vocab_partition::VocabPartition>();
        error::<facade::GrammarError>();
        error::<facade::gbnf::GbnfError>();
        error::<facade::json_schema::SchemaError>();
        error::<facade::pda::BuilderError>();
        error::<facade::pda::StackLimitError>();
        from::<facade::GrammarError, extracted::gbnf::GbnfError>();
        from::<facade::GrammarError, extracted::json_schema::SchemaError>();
        from::<facade::GrammarError, extracted::pda::BuilderError>();
        from::<facade::GrammarError, extracted::pda::StackLimitError>();
        from::<facade::gbnf::GbnfError, extracted::pda::BuilderError>();
        from::<facade::json_schema::SchemaError, extracted::pda::BuilderError>();
        from::<lattice_inference::InferenceError, extracted::GrammarError>();
        from::<lattice_inference_core::InferenceError, facade::GrammarError>();
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn grammar_profiling_items_are_the_same_at_both_paths() {
        same_type(
            PhantomData::<facade::engine::BfsDedupProfile>,
            PhantomData::<extracted::engine::BfsDedupProfile>,
        );
        same_function(
            facade::engine::profile_reachable_states,
            extracted::engine::profile_reachable_states,
        );
    }
}
