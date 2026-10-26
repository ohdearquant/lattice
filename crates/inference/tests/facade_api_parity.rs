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

mod weights_parity {
    use lattice_inference::weights as facade;
    use lattice_weights as extracted;
    use std::marker::PhantomData;

    fn same_type<T>(_: PhantomData<T>, _: PhantomData<T>) {}

    fn same_function<F>(_: F, _: F) {}

    #[test]
    fn weights_types_are_the_same_type_at_both_paths() {
        same_type(
            PhantomData::<facade::f16_weights::F16>,
            PhantomData::<extracted::f16_weights::F16>,
        );
        same_type(
            PhantomData::<facade::f16_weights::F16AttentionWeights>,
            PhantomData::<extracted::f16_weights::F16AttentionWeights>,
        );
        same_type(
            PhantomData::<facade::f16_weights::F16CommonLayerWeights>,
            PhantomData::<extracted::f16_weights::F16CommonLayerWeights>,
        );
        same_type(
            PhantomData::<facade::f16_weights::F16FeedForwardWeights>,
            PhantomData::<extracted::f16_weights::F16FeedForwardWeights>,
        );
        same_type(
            PhantomData::<facade::f16_weights::F16FullAttentionLayerWeights>,
            PhantomData::<extracted::f16_weights::F16FullAttentionLayerWeights>,
        );
        same_type(
            PhantomData::<facade::f16_weights::F16GatedDeltaNetWeights>,
            PhantomData::<extracted::f16_weights::F16GatedDeltaNetWeights>,
        );
        same_type(
            PhantomData::<facade::f16_weights::F16ModelWeights>,
            PhantomData::<extracted::f16_weights::F16ModelWeights>,
        );
        same_type(
            PhantomData::<facade::f16_weights::F16MoeLayerWeights>,
            PhantomData::<extracted::f16_weights::F16MoeLayerWeights>,
        );
        same_type(
            PhantomData::<facade::f16_weights::F16MoeRouter>,
            PhantomData::<extracted::f16_weights::F16MoeRouter>,
        );
        same_type(
            PhantomData::<facade::f16_weights::F16RoutedExperts>,
            PhantomData::<extracted::f16_weights::F16RoutedExperts>,
        );
        same_type(
            PhantomData::<facade::f16_weights::F16SharedExpert>,
            PhantomData::<extracted::f16_weights::F16SharedExpert>,
        );
        same_type(
            PhantomData::<facade::f32_weights::BertWeights<'static>>,
            PhantomData::<extracted::f32_weights::BertWeights<'static>>,
        );
        same_type(
            PhantomData::<facade::f32_weights::CrossEncoderWeights>,
            PhantomData::<extracted::f32_weights::CrossEncoderWeights>,
        );
        same_type(
            PhantomData::<facade::f32_weights::QwenLayerWeights<'static>>,
            PhantomData::<extracted::f32_weights::QwenLayerWeights<'static>>,
        );
        same_type(
            PhantomData::<facade::f32_weights::QwenWeights<'static>>,
            PhantomData::<extracted::f32_weights::QwenWeights<'static>>,
        );
        same_type(
            PhantomData::<facade::f32_weights::SafetensorsFile>,
            PhantomData::<extracted::f32_weights::SafetensorsFile>,
        );
        same_type(
            PhantomData::<facade::f32_weights::SafetensorsIndex>,
            PhantomData::<extracted::f32_weights::SafetensorsIndex>,
        );
        same_type(
            PhantomData::<facade::f32_weights::ShardedQwenBacking>,
            PhantomData::<extracted::f32_weights::ShardedQwenBacking>,
        );
        same_type(
            PhantomData::<facade::f32_weights::ShardedSafetensors>,
            PhantomData::<extracted::f32_weights::ShardedSafetensors>,
        );
        same_type(
            PhantomData::<facade::f32_weights::Tensor>,
            PhantomData::<extracted::f32_weights::Tensor>,
        );
        same_type(
            PhantomData::<facade::f32_weights::Tensor1D<'static>>,
            PhantomData::<extracted::f32_weights::Tensor1D<'static>>,
        );
        same_type(
            PhantomData::<facade::f32_weights::Tensor2D<'static>>,
            PhantomData::<extracted::f32_weights::Tensor2D<'static>>,
        );
        same_type(
            PhantomData::<facade::f32_weights::TransformerLayerWeights<'static>>,
            PhantomData::<extracted::f32_weights::TransformerLayerWeights<'static>>,
        );
        same_type(
            PhantomData::<facade::q3_weights::Q3Block>,
            PhantomData::<extracted::q3_weights::Q3Block>,
        );
        same_type(
            PhantomData::<facade::q3_weights::Q3FileHeader>,
            PhantomData::<extracted::q3_weights::Q3FileHeader>,
        );
        same_type(
            PhantomData::<facade::q3_weights::Q3Tensor>,
            PhantomData::<extracted::q3_weights::Q3Tensor>,
        );
        same_type(
            PhantomData::<facade::q4_weights::F16LoadError>,
            PhantomData::<extracted::q4_weights::F16LoadError>,
        );
        same_type(
            PhantomData::<facade::q4_weights::Q4Block>,
            PhantomData::<extracted::q4_weights::Q4Block>,
        );
        same_type(
            PhantomData::<facade::q4_weights::Q4FileHeader>,
            PhantomData::<extracted::q4_weights::Q4FileHeader>,
        );
        same_type(
            PhantomData::<facade::q4_weights::Q4Tensor>,
            PhantomData::<extracted::q4_weights::Q4Tensor>,
        );
        same_type(
            PhantomData::<facade::q8_weights::Q8Matrix>,
            PhantomData::<extracted::q8_weights::Q8Matrix>,
        );
        same_type(
            PhantomData::<facade::SafetensorsFile>,
            PhantomData::<extracted::SafetensorsFile>,
        );
        same_type(
            PhantomData::<facade::SafetensorsIndex>,
            PhantomData::<extracted::SafetensorsIndex>,
        );
        same_type(
            PhantomData::<facade::Tensor>,
            PhantomData::<extracted::Tensor>,
        );
        same_type(
            PhantomData::<facade::CrossEncoderWeights>,
            PhantomData::<extracted::CrossEncoderWeights>,
        );
        same_type(
            PhantomData::<facade::ShardedSafetensors>,
            PhantomData::<extracted::ShardedSafetensors>,
        );
        same_type(
            PhantomData::<facade::Tensor1D<'static>>,
            PhantomData::<extracted::Tensor1D<'static>>,
        );
    }

    #[test]
    fn weights_tensor_source_is_the_same_trait_at_both_paths() {
        fn facade_bound<T: facade::f32_weights::TensorSource + ?Sized>() {}
        fn extracted_bound<T: extracted::f32_weights::TensorSource + ?Sized>() {}
        fn extracted_implies_facade<T: extracted::f32_weights::TensorSource + ?Sized>() {
            facade_bound::<T>();
        }
        fn facade_implies_extracted<T: facade::f32_weights::TensorSource + ?Sized>() {
            extracted_bound::<T>();
        }
        extracted_implies_facade::<extracted::SafetensorsFile>();
        facade_implies_extracted::<facade::SafetensorsFile>();
    }

    #[test]
    fn weights_functions_are_the_same_items_at_both_paths() {
        same_function(
            facade::f16_weights::f16_to_f32_slice,
            extracted::f16_weights::f16_to_f32_slice,
        );
        same_function(
            facade::f16_weights::f32_to_f16_slice,
            extracted::f16_weights::f32_to_f16_slice,
        );
        same_function(
            facade::f32_weights::contained_shard_path,
            extracted::f32_weights::contained_shard_path,
        );
        same_function(
            facade::f32_weights::load_sharded,
            extracted::f32_weights::load_sharded,
        );
        same_function(
            facade::f32_weights::parse_index,
            extracted::f32_weights::parse_index,
        );
        same_function(
            facade::f32_weights::resolve_shard,
            extracted::f32_weights::resolve_shard,
        );
        same_function(
            facade::q3_weights::dequantize_q3_to_f32,
            extracted::q3_weights::dequantize_q3_to_f32,
        );
        same_function(
            facade::q3_weights::dequantize_row_q3_0,
            extracted::q3_weights::dequantize_row_q3_0,
        );
        same_function(
            facade::q3_weights::gemm_q3_reference,
            extracted::q3_weights::gemm_q3_reference,
        );
        same_function(
            facade::q3_weights::gemv_q3_reference,
            extracted::q3_weights::gemv_q3_reference,
        );
        same_function(
            facade::q3_weights::load_q3_file,
            extracted::q3_weights::load_q3_file,
        );
        same_function(
            facade::q3_weights::quantize_bf16_to_q3,
            extracted::q3_weights::quantize_bf16_to_q3,
        );
        same_function(
            facade::q3_weights::quantize_f32_to_q3,
            extracted::q3_weights::quantize_f32_to_q3,
        );
        same_function(
            facade::q3_weights::quantize_f32_to_q3_mode,
            extracted::q3_weights::quantize_f32_to_q3_mode,
        );
        same_function(
            facade::q3_weights::quantize_row_q3_0,
            extracted::q3_weights::quantize_row_q3_0,
        );
        same_function(
            facade::q3_weights::quantize_tensor_q3_0,
            extracted::q3_weights::quantize_tensor_q3_0,
        );
        same_function(
            facade::q3_weights::read_q3_header::<std::fs::File>,
            extracted::q3_weights::read_q3_header::<std::fs::File>,
        );
        same_function(
            facade::q3_weights::save_q3_file,
            extracted::q3_weights::save_q3_file,
        );
        same_function(
            facade::q3_weights::validate_q3_mlp_role,
            extracted::q3_weights::validate_q3_mlp_role,
        );
        same_function(
            facade::q4_weights::dequantize_q4_to_f32,
            extracted::q4_weights::dequantize_q4_to_f32,
        );
        same_function(
            facade::q4_weights::dequantize_row_q4_0,
            extracted::q4_weights::dequantize_row_q4_0,
        );
        same_function(
            facade::q4_weights::load_f16_tensor_file,
            extracted::q4_weights::load_f16_tensor_file,
        );
        same_function(
            facade::q4_weights::load_f16_tensor_file_expecting,
            extracted::q4_weights::load_f16_tensor_file_expecting,
        );
        same_function(
            facade::q4_weights::load_q4_file,
            extracted::q4_weights::load_q4_file,
        );
        same_function(
            facade::q4_weights::quantize_bf16_to_q4,
            extracted::q4_weights::quantize_bf16_to_q4,
        );
        same_function(
            facade::q4_weights::quantize_f32_to_q4,
            extracted::q4_weights::quantize_f32_to_q4,
        );
        same_function(
            facade::q4_weights::quantize_f64_to_q4,
            extracted::q4_weights::quantize_f64_to_q4,
        );
        same_function(
            facade::q4_weights::quantize_f64_to_q4_mode,
            extracted::q4_weights::quantize_f64_to_q4_mode,
        );
        same_function(
            facade::q4_weights::quantize_row_q4_0,
            extracted::q4_weights::quantize_row_q4_0,
        );
        same_function(
            facade::q4_weights::quantize_tensor_q4_0,
            extracted::q4_weights::quantize_tensor_q4_0,
        );
        same_function(
            facade::q4_weights::read_q4_header::<std::fs::File>,
            extracted::q4_weights::read_q4_header::<std::fs::File>,
        );
        same_function(
            facade::q4_weights::save_q4_file,
            extracted::q4_weights::save_q4_file,
        );
        same_function(
            facade::q4_weights::stream_quantize_shard,
            extracted::q4_weights::stream_quantize_shard,
        );
        same_function(
            facade::q8_weights::quantize_matrix,
            extracted::q8_weights::quantize_matrix,
        );
        same_function(facade::load_sharded, extracted::load_sharded);
        same_function(facade::parse_index, extracted::parse_index);
        same_function(
            facade::contained_shard_path,
            extracted::contained_shard_path,
        );
        same_function(facade::resolve_shard, extracted::resolve_shard);
    }

    #[test]
    fn weights_constants_are_equal_at_both_paths() {
        assert_eq!(
            facade::q3_weights::Q3_BLOCK_BYTES,
            extracted::q3_weights::Q3_BLOCK_BYTES
        );
        assert_eq!(
            facade::q4_weights::Q4_BLOCK_BYTES,
            extracted::q4_weights::Q4_BLOCK_BYTES
        );
    }

    #[test]
    fn model_format_items_are_the_same_at_both_paths() {
        use lattice_inference::model_format as facade_format;
        use lattice_weights::model_format as extracted_format;

        let _: fn(facade_format::FormatDetection) -> extracted_format::FormatDetection =
            |value| value;
        let _: fn(extracted_format::FormatDetection) -> facade_format::FormatDetection =
            |value| value;
        let _: fn(facade_format::ModelFamily) -> extracted_format::ModelFamily = |value| value;
        let _: fn(extracted_format::ModelFamily) -> facade_format::ModelFamily = |value| value;
        let _: fn(facade_format::ModelFormat) -> extracted_format::ModelFormat = |value| value;
        let _: fn(extracted_format::ModelFormat) -> facade_format::ModelFormat = |value| value;
        same_function(
            facade_format::detect_family,
            extracted_format::detect_family,
        );
        same_function(
            facade_format::detect_format,
            extracted_format::detect_format,
        );
        same_function(
            facade_format::inspect_format,
            extracted_format::inspect_format,
        );
        same_function(
            facade_format::metal_gpu_required_message,
            extracted_format::metal_gpu_required_message,
        );
        same_function(
            facade_format::unloadable_format_message,
            extracted_format::unloadable_format_message,
        );
        same_function(
            facade_format::unrecognized_format_message,
            extracted_format::unrecognized_format_message,
        );
    }
}

mod quant_parity {
    use lattice_inference::quant as facade;
    use lattice_quant as extracted;
    use std::marker::PhantomData;

    fn same_type<T>(_: PhantomData<T>, _: PhantomData<T>) {}

    fn same_function<F>(_: F, _: F) {}

    #[test]
    fn quant_types_are_the_same_type_at_both_paths() {
        same_type(
            PhantomData::<facade::q4_manifest::ManifestFlavor>,
            PhantomData::<extracted::q4_manifest::ManifestFlavor>,
        );
        same_type(
            PhantomData::<facade::q4_manifest::Q4Manifest>,
            PhantomData::<extracted::q4_manifest::Q4Manifest>,
        );
        same_type(
            PhantomData::<facade::q4_manifest::Q4ManifestEntry>,
            PhantomData::<extracted::q4_manifest::Q4ManifestEntry>,
        );
        same_type(
            PhantomData::<facade::q4_manifest::Q4ManifestError>,
            PhantomData::<extracted::q4_manifest::Q4ManifestError>,
        );
        same_type(
            PhantomData::<facade::quarot::hadamard::BlockHadamard>,
            PhantomData::<extracted::quarot::hadamard::BlockHadamard>,
        );
        same_type(
            PhantomData::<facade::quarot::hadamard::RandomizedHadamard>,
            PhantomData::<extracted::quarot::hadamard::RandomizedHadamard>,
        );
    }

    #[test]
    fn quant_functions_are_the_same_items_at_both_paths() {
        same_function(
            facade::q4_manifest::load_manifest,
            extracted::q4_manifest::load_manifest,
        );
        same_function(
            facade::q4_manifest::parse_manifest,
            extracted::q4_manifest::parse_manifest,
        );
        same_function(
            facade::q4_manifest::read_manifest_bytes_bounded,
            extracted::q4_manifest::read_manifest_bytes_bounded,
        );
        same_function(
            facade::quarot::hadamard::walsh_hadamard_f64_in_place,
            extracted::quarot::hadamard::walsh_hadamard_f64_in_place,
        );
        same_function(
            facade::quarot::hadamard::walsh_hadamard_in_place,
            extracted::quarot::hadamard::walsh_hadamard_in_place,
        );
        same_function(
            facade::quarot::hadamard::walsh_hadamard_orthonormal_f64_in_place,
            extracted::quarot::hadamard::walsh_hadamard_orthonormal_f64_in_place,
        );
        same_function(
            facade::quarot::hadamard::walsh_hadamard_orthonormal_in_place,
            extracted::quarot::hadamard::walsh_hadamard_orthonormal_in_place,
        );
        same_function(
            facade::quarot::rotation::absorb_input_rotation,
            extracted::quarot::rotation::absorb_input_rotation,
        );
        same_function(
            facade::quarot::rotation::absorb_input_rotation_f64,
            extracted::quarot::rotation::absorb_input_rotation_f64,
        );
        same_function(
            facade::quarot::rotation::absorb_output_rotation,
            extracted::quarot::rotation::absorb_output_rotation,
        );
        same_function(
            facade::quarot::rotation::absorb_output_rotation_f64,
            extracted::quarot::rotation::absorb_output_rotation_f64,
        );
    }

    #[test]
    fn quant_constants_are_equal_at_both_paths() {
        assert_eq!(
            facade::q4_manifest::MAX_QUANTIZE_INDEX_LEN,
            extracted::q4_manifest::MAX_QUANTIZE_INDEX_LEN
        );
    }

    #[test]
    fn quarot_conversion_types_are_the_same_type_at_both_paths() {
        same_type(
            PhantomData::<facade::quarot::convert::ConversionOptions>,
            PhantomData::<extracted::quarot::convert::ConversionOptions>,
        );
        same_type(
            PhantomData::<facade::quarot::convert::ConversionReport>,
            PhantomData::<extracted::quarot::convert::ConversionReport>,
        );
        same_type(
            PhantomData::<facade::quarot::convert::PplGateRecord>,
            PhantomData::<extracted::quarot::convert::PplGateRecord>,
        );
        same_type(
            PhantomData::<facade::quarot::convert::PromotionRecord>,
            PhantomData::<extracted::quarot::convert::PromotionRecord>,
        );
        same_type(
            PhantomData::<facade::quarot::convert::PromotionState>,
            PhantomData::<extracted::quarot::convert::PromotionState>,
        );
        same_type(
            PhantomData::<facade::quarot::forward_equivalence::ForwardEquivalenceConfig>,
            PhantomData::<extracted::quarot::forward_equivalence::ForwardEquivalenceConfig>,
        );
        same_type(
            PhantomData::<facade::quarot::forward_equivalence::ForwardEquivalenceReport>,
            PhantomData::<extracted::quarot::forward_equivalence::ForwardEquivalenceReport>,
        );
        same_type(
            PhantomData::<facade::quarot::io::ArtifactVersion>,
            PhantomData::<extracted::quarot::io::ArtifactVersion>,
        );
        same_type(
            PhantomData::<facade::quarot::io::OnlineArtifactDescriptor>,
            PhantomData::<extracted::quarot::io::OnlineArtifactDescriptor>,
        );
        same_type(
            PhantomData::<facade::quarot::io::QuarotTensorReader>,
            PhantomData::<extracted::quarot::io::QuarotTensorReader>,
        );
        same_type(
            PhantomData::<facade::quarot::io::SourceDType>,
            PhantomData::<extracted::quarot::io::SourceDType>,
        );
        same_type(
            PhantomData::<facade::quarot::lora::LoraLayerMut<'static>>,
            PhantomData::<extracted::quarot::lora::LoraLayerMut<'static>>,
        );
        same_type(
            PhantomData::<facade::quarot::lora::LoraRotationApplied>,
            PhantomData::<extracted::quarot::lora::LoraRotationApplied>,
        );
        same_type(
            PhantomData::<facade::quarot::lora::RotationReport>,
            PhantomData::<extracted::quarot::lora::RotationReport>,
        );
        same_type(
            PhantomData::<facade::quarot::pipeline::TensorEntry>,
            PhantomData::<extracted::quarot::pipeline::TensorEntry>,
        );
        same_type(
            PhantomData::<facade::quarot::plan::AbsorptionSide>,
            PhantomData::<extracted::quarot::plan::AbsorptionSide>,
        );
        same_type(
            PhantomData::<facade::quarot::plan::CoverageReport>,
            PhantomData::<extracted::quarot::plan::CoverageReport>,
        );
        same_type(
            PhantomData::<facade::quarot::plan::OnlineRotationSpec>,
            PhantomData::<extracted::quarot::plan::OnlineRotationSpec>,
        );
        same_type(
            PhantomData::<facade::quarot::plan::OnlineTransformSite>,
            PhantomData::<extracted::quarot::plan::OnlineTransformSite>,
        );
        same_type(
            PhantomData::<facade::quarot::plan::RotationId>,
            PhantomData::<extracted::quarot::plan::RotationId>,
        );
        same_type(
            PhantomData::<facade::quarot::plan::RotationPlan>,
            PhantomData::<extracted::quarot::plan::RotationPlan>,
        );
        same_type(
            PhantomData::<facade::quarot::plan::RuleRequirement>,
            PhantomData::<extracted::quarot::plan::RuleRequirement>,
        );
        same_type(
            PhantomData::<facade::quarot::plan::TensorRotation>,
            PhantomData::<extracted::quarot::plan::TensorRotation>,
        );
        same_type(
            PhantomData::<facade::quarot::rmsnorm_fusion::RmsNormFusionTarget>,
            PhantomData::<extracted::quarot::rmsnorm_fusion::RmsNormFusionTarget>,
        );
        same_type(
            PhantomData::<facade::quarot::ArtifactVersion>,
            PhantomData::<extracted::quarot::ArtifactVersion>,
        );
        same_type(
            PhantomData::<facade::quarot::OnlineArtifactDescriptor>,
            PhantomData::<extracted::quarot::OnlineArtifactDescriptor>,
        );
        same_type(
            PhantomData::<facade::quarot::QuarotTensorReader>,
            PhantomData::<extracted::quarot::QuarotTensorReader>,
        );
        same_type(
            PhantomData::<facade::quarot::SourceDType>,
            PhantomData::<extracted::quarot::SourceDType>,
        );
        same_type(
            PhantomData::<facade::quarot::OnlineRotationSpec>,
            PhantomData::<extracted::quarot::OnlineRotationSpec>,
        );
        same_type(
            PhantomData::<facade::quarot::OnlineTransformSite>,
            PhantomData::<extracted::quarot::OnlineTransformSite>,
        );
        same_type(
            PhantomData::<facade::quarot::RotationId>,
            PhantomData::<extracted::quarot::RotationId>,
        );
    }

    #[test]
    fn neutral_rotation_targets_resolve_at_facade_paths() {
        same_type(
            PhantomData::<facade::quarot::plan::RotationTarget>,
            PhantomData::<extracted::quarot::plan::RotationTarget>,
        );
        same_function(
            facade::quarot::plan::RotationTarget::new,
            extracted::quarot::plan::RotationTarget::new,
        );
        same_function(
            facade::quarot::plan::qwen35_rotation_target,
            extracted::quarot::plan::qwen35_rotation_target,
        );
        same_function(
            facade::quarot::plan::qwen35_residual_stream_linear_layers,
            extracted::quarot::plan::qwen35_residual_stream_linear_layers,
        );
        same_function(
            facade::quarot::plan::OnlineRotationSpec::attention_output_r3,
            extracted::quarot::plan::OnlineRotationSpec::attention_output_r3,
        );
        same_function(
            facade::quarot::plan::OnlineRotationSpec::mlp_down_r4,
            extracted::quarot::plan::OnlineRotationSpec::mlp_down_r4,
        );
        same_function(
            facade::quarot::plan::OnlineRotationSpec::validate_target,
            extracted::quarot::plan::OnlineRotationSpec::validate_target,
        );
        same_function(
            facade::quarot::io::OnlineArtifactDescriptor::validate_target,
            extracted::quarot::io::OnlineArtifactDescriptor::validate_target,
        );
    }

    #[test]
    fn quarot_conversion_functions_are_the_same_items_at_both_paths() {
        same_function(
            facade::quarot::convert::convert_quarot_qwen35,
            extracted::quarot::convert::convert_quarot_qwen35,
        );
        same_function(
            facade::quarot::convert::read_promotion_record,
            extracted::quarot::convert::read_promotion_record,
        );
        same_function(
            facade::quarot::convert::record_ppl_gate_result,
            extracted::quarot::convert::record_ppl_gate_result,
        );
        same_function(
            facade::quarot::forward_equivalence::assert_forward_equivalence_qwen35,
            extracted::quarot::forward_equivalence::assert_forward_equivalence_qwen35,
        );
        same_function(
            facade::quarot::lm_head::materialize_lm_head_for_qwen35,
            extracted::quarot::lm_head::materialize_lm_head_for_qwen35,
        );
        same_function(
            facade::quarot::lm_head::qwen35_final_norm_fusion_target,
            extracted::quarot::lm_head::qwen35_final_norm_fusion_target,
        );
        same_function(
            facade::quarot::lm_head::untie_word_embeddings_in_cfg,
            extracted::quarot::lm_head::untie_word_embeddings_in_cfg,
        );
        same_function(
            facade::quarot::lm_head::untie_word_embeddings_in_config_json,
            extracted::quarot::lm_head::untie_word_embeddings_in_config_json,
        );
        same_function(
            facade::quarot::lora::counter_rotate_a,
            extracted::quarot::lora::counter_rotate_a,
        );
        same_function(
            facade::quarot::lora::needs_b_rotation,
            extracted::quarot::lora::needs_b_rotation,
        );
        same_function(
            facade::quarot::lora::needs_counter_rotation,
            extracted::quarot::lora::needs_counter_rotation,
        );
        same_function(
            facade::quarot::lora::rotate_adapter_for_quarot,
            extracted::quarot::lora::rotate_adapter_for_quarot,
        );
        same_function(
            facade::quarot::lora::rotate_b_output_side,
            extracted::quarot::lora::rotate_b_output_side,
        );
        same_function(
            facade::quarot::pipeline::absorb_rotations,
            extracted::quarot::pipeline::absorb_rotations,
        );
        same_function(
            facade::quarot::pipeline::fuse_rmsnorms,
            extracted::quarot::pipeline::fuse_rmsnorms,
        );
        same_function(
            facade::quarot::pipeline::load_tensors_f64,
            extracted::quarot::pipeline::load_tensors_f64,
        );
        same_function(
            facade::quarot::plan::apply_tensor_rotation,
            extracted::quarot::plan::apply_tensor_rotation,
        );
        same_function(
            facade::quarot::plan::apply_tensor_rotation_f64,
            extracted::quarot::plan::apply_tensor_rotation_f64,
        );
        same_function(
            facade::quarot::plan::qwen3_required_tensor_names,
            extracted::quarot::plan::qwen3_required_tensor_names,
        );
        same_function(
            facade::quarot::rmsnorm_fusion::fuse_shifted_rmsnorm_into_next_layer_f64,
            extracted::quarot::rmsnorm_fusion::fuse_shifted_rmsnorm_into_next_layer_f64,
        );
        same_function(
            facade::quarot::rmsnorm_fusion::neutralize_rmsnorm_gamma_f64,
            extracted::quarot::rmsnorm_fusion::neutralize_rmsnorm_gamma_f64,
        );
        same_function(
            facade::quarot::rmsnorm_fusion::qwen35_per_layer_fusion_plan,
            extracted::quarot::rmsnorm_fusion::qwen35_per_layer_fusion_plan,
        );
    }

    #[test]
    fn quarot_conversion_constants_are_equal_at_both_paths() {
        assert_eq!(
            facade::quarot::lm_head::QWEN35_EMBED_TOKENS_NAME,
            extracted::quarot::lm_head::QWEN35_EMBED_TOKENS_NAME
        );
        assert_eq!(
            facade::quarot::lm_head::QWEN35_FINAL_NORM_NAME,
            extracted::quarot::lm_head::QWEN35_FINAL_NORM_NAME
        );
        assert_eq!(
            facade::quarot::lm_head::QWEN35_LM_HEAD_NAME,
            extracted::quarot::lm_head::QWEN35_LM_HEAD_NAME
        );
    }
}

mod model_config_parity {
    use lattice_arch as extracted;
    use lattice_inference::model as facade;
    use std::marker::PhantomData;

    fn same_type<T>(_: PhantomData<T>, _: PhantomData<T>) {}

    #[test]
    fn model_config_types_are_the_same_type_at_both_paths() {
        same_type(
            PhantomData::<facade::qwen35_config::LayerType>,
            PhantomData::<extracted::qwen35_config::LayerType>,
        );
        same_type(
            PhantomData::<facade::qwen35_config::Qwen35Config>,
            PhantomData::<extracted::qwen35_config::Qwen35Config>,
        );
        same_type(
            PhantomData::<facade::qwen35_config::RopeParams>,
            PhantomData::<extracted::qwen35_config::RopeParams>,
        );
        same_type(
            PhantomData::<facade::qwen35_config::ValidatedQwen35Config>,
            PhantomData::<extracted::qwen35_config::ValidatedQwen35Config>,
        );
        same_type(
            PhantomData::<facade::qwen35_config::VisionModelConfig>,
            PhantomData::<extracted::qwen35_config::VisionModelConfig>,
        );
        same_type(
            PhantomData::<facade::gemma4_config::Gemma4Config>,
            PhantomData::<extracted::gemma4_config::Gemma4Config>,
        );
        same_type(
            PhantomData::<facade::gemma4_config::Gemma4LayerType>,
            PhantomData::<extracted::gemma4_config::Gemma4LayerType>,
        );
        same_type(
            PhantomData::<facade::embeddinggemma2_config::EmbeddingGemma2Config>,
            PhantomData::<extracted::embeddinggemma2_config::EmbeddingGemma2Config>,
        );
        same_type(
            PhantomData::<facade::embeddinggemma2_config::EmbeddingGemma2LayerKind>,
            PhantomData::<extracted::embeddinggemma2_config::EmbeddingGemma2LayerKind>,
        );
        same_type(
            PhantomData::<facade::embeddinggemma2_config::EmbeddingGemma2LayerShape>,
            PhantomData::<extracted::embeddinggemma2_config::EmbeddingGemma2LayerShape>,
        );
        same_type(
            PhantomData::<facade::bitnet_config::BitNetConfig>,
            PhantomData::<extracted::bitnet_config::BitNetConfig>,
        );
        same_type(
            PhantomData::<facade::qwen::QwenConfig>,
            PhantomData::<extracted::qwen_config::QwenConfig>,
        );
        same_type(
            PhantomData::<lattice_inference::QwenConfig>,
            PhantomData::<extracted::qwen_config::QwenConfig>,
        );
    }

    fn same_fn<F>(_: F, _: F) {}

    #[test]
    fn model_config_functions_are_the_same_function_at_both_paths() {
        same_fn(
            facade::qwen35::qwen_layer_tensor_prefix,
            extracted::qwen35_tensors::qwen_layer_tensor_prefix,
        );
        same_fn(
            facade::qwen35::qwen_required_tensor_names,
            extracted::qwen35_tensors::qwen_required_tensor_names,
        );
    }

    #[test]
    fn model_config_constants_are_equal_at_both_paths() {
        assert_eq!(
            facade::qwen35_config::MAX_HIDDEN_LAYERS,
            extracted::qwen35_config::MAX_HIDDEN_LAYERS
        );
        assert_eq!(
            facade::qwen35_config::QWEN_CHAT_IM_END_TOKEN_ID,
            extracted::qwen35_config::QWEN_CHAT_IM_END_TOKEN_ID
        );
        assert_eq!(
            facade::qwen35_config::QWEN3_NEWLINE_TOKEN_ID,
            extracted::qwen35_config::QWEN3_NEWLINE_TOKEN_ID
        );
        assert_eq!(
            facade::qwen35_config::QWEN3_NO_THINK_PREFIX,
            extracted::qwen35_config::QWEN3_NO_THINK_PREFIX
        );
        assert_eq!(
            facade::qwen35_config::QWEN3_THINK_CLOSE_TOKEN_ID,
            extracted::qwen35_config::QWEN3_THINK_CLOSE_TOKEN_ID
        );
        assert_eq!(
            facade::qwen35_config::QWEN3_THINK_OPEN_TOKEN_ID,
            extracted::qwen35_config::QWEN3_THINK_OPEN_TOKEN_ID
        );
        assert_eq!(
            facade::gemma4_config::GEMMA4_EXPECTED_DTYPE,
            extracted::gemma4_config::GEMMA4_EXPECTED_DTYPE
        );
    }
}

mod interface_parity {
    use lattice_arch as extracted;
    use std::marker::PhantomData;

    fn same_type<T>(_: PhantomData<T>, _: PhantomData<T>) {}

    #[test]
    fn generation_types_are_the_same_type_at_every_path() {
        use lattice_inference::generation as facade;
        same_type(
            PhantomData::<facade::GenerateConfig>,
            PhantomData::<extracted::generation::GenerateConfig>,
        );
        same_type(
            PhantomData::<facade::GenerateOutput>,
            PhantomData::<extracted::generation::GenerateOutput>,
        );
        same_type(
            PhantomData::<facade::TokenLogprob>,
            PhantomData::<extracted::generation::TokenLogprob>,
        );
        same_type(
            PhantomData::<facade::TopLogprob>,
            PhantomData::<extracted::generation::TopLogprob>,
        );
        same_type(
            PhantomData::<lattice_inference::GenerateConfig>,
            PhantomData::<extracted::generation::GenerateConfig>,
        );
        same_type(
            PhantomData::<lattice_inference::GenerateOutput>,
            PhantomData::<extracted::generation::GenerateOutput>,
        );
        same_type(
            PhantomData::<lattice_inference::TokenLogprob>,
            PhantomData::<extracted::generation::TokenLogprob>,
        );
        same_type(
            PhantomData::<lattice_inference::TopLogprob>,
            PhantomData::<extracted::generation::TopLogprob>,
        );
    }

    #[test]
    fn sampling_types_are_the_same_type_at_both_paths() {
        use lattice_inference::sampling as facade;
        same_type(
            PhantomData::<facade::SamplingConfig>,
            PhantomData::<extracted::sampling::SamplingConfig>,
        );
        same_type(
            PhantomData::<facade::Candidate>,
            PhantomData::<extracted::sampling::Candidate>,
        );
        same_type(
            PhantomData::<facade::CandidateSet>,
            PhantomData::<extracted::sampling::CandidateSet>,
        );
        same_type(
            PhantomData::<facade::Sampler>,
            PhantomData::<extracted::sampling::Sampler>,
        );
    }

    #[test]
    fn stop_reason_is_the_same_type_at_every_path() {
        same_type(
            PhantomData::<lattice_inference::stop_reason::StopReason>,
            PhantomData::<extracted::stop_reason::StopReason>,
        );
        same_type(
            PhantomData::<lattice_inference::StopReason>,
            PhantomData::<extracted::stop_reason::StopReason>,
        );
    }
}

mod compute_parity {
    use lattice_compute as extracted;
    use lattice_inference::rope as facade;
    use std::marker::PhantomData;

    fn same_type<T>(_: PhantomData<T>, _: PhantomData<T>) {}

    fn half_dim_of(table: &extracted::rope::RopeTable) -> usize {
        table.half_dim()
    }

    #[test]
    fn rope_table_is_the_same_type_at_both_paths() {
        same_type(
            PhantomData::<facade::RopeTable>,
            PhantomData::<extracted::rope::RopeTable>,
        );
        let table = facade::RopeTable::new(8, 4, 10_000.0);
        assert_eq!(half_dim_of(&table), 4);
    }

    fn same_function<F>(_: F, _: F) {}

    #[test]
    fn prefix_cache_types_are_the_same_type_at_both_paths() {
        same_type(
            PhantomData::<lattice_inference::kv_cache::AdapterId>,
            PhantomData::<extracted::kv_cache::prefix::AdapterId>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::PrefixKey>,
            PhantomData::<extracted::kv_cache::prefix::PrefixKey>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::SharedPageRef>,
            PhantomData::<extracted::kv_cache::prefix::SharedPageRef>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::PrefixPageCacheConfig>,
            PhantomData::<extracted::kv_cache::prefix::PrefixPageCacheConfig>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::PrefixEntry>,
            PhantomData::<extracted::kv_cache::prefix::PrefixEntry>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::PrefixPageCache>,
            PhantomData::<extracted::kv_cache::prefix::PrefixPageCache>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::CrossTurnSlotId>,
            PhantomData::<extracted::kv_cache::cross_turn::CrossTurnSlotId>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::CrossTurnPrefixMetadata>,
            PhantomData::<extracted::kv_cache::cross_turn::CrossTurnPrefixMetadata>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::KvPrefixHandle>,
            PhantomData::<extracted::kv_cache::cross_turn::KvPrefixHandle>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::CrossTurnPrefixEntry>,
            PhantomData::<extracted::kv_cache::cross_turn::CrossTurnPrefixEntry>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::PrefixReuseMode>,
            PhantomData::<extracted::kv_cache::cross_turn::PrefixReuseMode>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::PrefixRestorePlan>,
            PhantomData::<extracted::kv_cache::cross_turn::PrefixRestorePlan>,
        );
    }

    #[test]
    fn prefix_cache_functions_are_the_same_items_at_both_paths() {
        same_function(
            lattice_inference::kv_cache::longest_common_token_prefix,
            extracted::kv_cache::cross_turn::longest_common_token_prefix,
        );
        same_function(
            lattice_inference::kv_cache::plan_prefix_reuse,
            extracted::kv_cache::cross_turn::plan_prefix_reuse,
        );
        same_function(
            lattice_inference::kv_cache::checkpoint_survives_save,
            extracted::kv_cache::cross_turn::checkpoint_survives_save,
        );
    }

    #[test]
    fn flat_cache_types_are_the_same_type_at_both_paths() {
        same_type(
            PhantomData::<lattice_inference::kv_cache::FlatKVCache>,
            PhantomData::<extracted::kv_cache::flat::FlatKVCache>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::FlatKVCacheConfig>,
            PhantomData::<extracted::kv_cache::flat::FlatKVCacheConfig>,
        );
    }

    /// Migration probe: `FlatKVCacheConfig::new`, the replacement for the deprecated `for_qwen3`,
    /// is reachable at the facade path with the same arguments.
    #[test]
    fn flat_cache_config_new_is_reachable_at_the_facade_path() {
        let new: fn(usize, usize, usize, usize) -> lattice_inference::kv_cache::FlatKVCacheConfig =
            lattice_inference::kv_cache::FlatKVCacheConfig::new;
        let config = new(2, 4, 8, 16);
        assert_eq!(
            (
                config.num_layers,
                config.num_kv_heads,
                config.head_dim,
                config.max_seq_len
            ),
            (2, 4, 8, 16)
        );
    }

    #[test]
    fn paged_cache_types_are_the_same_type_at_both_paths() {
        same_type(
            PhantomData::<lattice_inference::kv_cache::CacheType>,
            PhantomData::<extracted::kv_cache::paged::CacheType>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::EvictionPolicy>,
            PhantomData::<extracted::kv_cache::paged::EvictionPolicy>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::PagePool>,
            PhantomData::<extracted::kv_cache::paged::PagePool>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::PageTable>,
            PhantomData::<extracted::kv_cache::paged::PageTable>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::PagedKVCache>,
            PhantomData::<extracted::kv_cache::paged::PagedKVCache>,
        );
        same_type(
            PhantomData::<lattice_inference::kv_cache::PagedKVCacheConfig>,
            PhantomData::<extracted::kv_cache::paged::PagedKVCacheConfig>,
        );
    }
}

mod lora_hook_parity {
    use lattice_compute::lora_hook as extracted;
    use lattice_inference::lora_hook as facade;
    use std::marker::PhantomData;

    fn same_type<T: ?Sized>(_: PhantomData<T>, _: PhantomData<T>) {}
    fn same_function<F>(_: F, _: F) {}
    fn send_sync<T: Send + Sync + ?Sized>() {}

    #[test]
    fn lora_types_and_trait_are_the_same_at_both_paths() {
        same_type(
            PhantomData::<facade::NoopLoraHook>,
            PhantomData::<extracted::NoopLoraHook>,
        );
        same_type(
            PhantomData::<facade::LoraProjectionShape>,
            PhantomData::<extracted::LoraProjectionShape>,
        );
        same_type(
            PhantomData::<dyn facade::LoraHook>,
            PhantomData::<dyn extracted::LoraHook>,
        );
        same_type(
            PhantomData::<dyn facade::LoraTargetGeometry>,
            PhantomData::<dyn extracted::LoraTargetGeometry>,
        );
        let _: extracted::NoopLoraHook = facade::NoopLoraHook;
        send_sync::<dyn facade::LoraHook>();
        send_sync::<facade::NoopLoraHook>();
    }

    #[test]
    fn lora_functions_are_the_same_items_at_both_paths() {
        same_function(
            facade::gdn_lora_module_is_servable,
            extracted::gdn_lora_module_is_servable,
        );
        let _: fn(
            &lattice_inference::model::qwen35_config::Qwen35Config,
            usize,
            &str,
        ) -> Result<facade::LoraProjectionShape, String> = facade::qwen35_projection_shape;
    }

    #[test]
    fn lora_constants_are_equal_at_both_paths() {
        assert_eq!(facade::GDN_LORA_MODULES, extracted::GDN_LORA_MODULES);
        assert_eq!(
            facade::GDN_LORA_MODULES_SERVABLE,
            extracted::GDN_LORA_MODULES_SERVABLE
        );
    }
}

mod metrics_pruning_parity {
    use lattice_compute as extracted;
    use lattice_inference::{metrics as facade_metrics, pruning as facade_pruning};
    use std::marker::PhantomData;

    fn same_type<T: ?Sized>(_: PhantomData<T>, _: PhantomData<T>) {}
    fn same_function<F>(_: F, _: F) {}

    #[test]
    fn metrics_types_are_the_same_at_both_paths() {
        same_type(
            PhantomData::<facade_metrics::MetricsMode>,
            PhantomData::<extracted::metrics::MetricsMode>,
        );
        same_type(
            PhantomData::<facade_metrics::LayerMetrics>,
            PhantomData::<extracted::metrics::LayerMetrics>,
        );
        same_type(
            PhantomData::<facade_metrics::ForwardMetrics>,
            PhantomData::<extracted::metrics::ForwardMetrics>,
        );
        same_type(
            PhantomData::<facade_metrics::OnlineSoftmaxEntropy>,
            PhantomData::<extracted::metrics::OnlineSoftmaxEntropy>,
        );
        let _: extracted::metrics::OnlineSoftmaxEntropy =
            facade_metrics::OnlineSoftmaxEntropy::new();
    }

    #[test]
    fn metrics_functions_are_the_same_items_at_both_paths() {
        same_function(facade_metrics::l2_norm, extracted::metrics::l2_norm);
    }

    #[test]
    fn pruning_types_are_the_same_at_both_paths() {
        same_type(
            PhantomData::<facade_pruning::BlockInfluence>,
            PhantomData::<extracted::pruning::BlockInfluence>,
        );
        same_type(
            PhantomData::<facade_pruning::BlockInfluenceAccumulator>,
            PhantomData::<extracted::pruning::BlockInfluenceAccumulator>,
        );
    }

    #[test]
    fn pruning_functions_are_the_same_items_at_both_paths() {
        same_function(
            facade_pruning::cosine_similarity,
            extracted::pruning::cosine_similarity,
        );
        same_function(
            facade_pruning::score_from_hidden_states,
            extracted::pruning::score_from_hidden_states,
        );
        same_function(
            facade_pruning::pruning_rank,
            extracted::pruning::pruning_rank,
        );
    }

    #[test]
    fn metrics_mode_variants_resolve_through_the_facade() {
        let mode = facade_metrics::MetricsMode::default();
        assert!(matches!(mode, facade_metrics::MetricsMode::Off));
        let _ = facade_metrics::MetricsMode::HeavyDiagnostics;
    }
}

mod option_scoring_parity {
    use lattice_compute::option_scoring as extracted;
    use lattice_inference::option_scoring as facade;
    use std::marker::PhantomData;

    fn same_type<T: ?Sized>(_: PhantomData<T>, _: PhantomData<T>) {}
    fn same_function<F>(_: F, _: F) {}

    #[test]
    fn option_scoring_types_are_the_same_at_both_paths() {
        same_type(
            PhantomData::<facade::OptionScores>,
            PhantomData::<extracted::OptionScores>,
        );
        same_type(
            PhantomData::<facade::DecisionLabelTable>,
            PhantomData::<extracted::DecisionLabelTable>,
        );
    }

    #[test]
    fn option_scoring_functions_are_the_same_items_at_both_paths() {
        same_function(
            facade::option_scores_from_logits,
            extracted::option_scores_from_logits,
        );
        same_function(
            facade::resolve_option_letters,
            extracted::resolve_option_letters,
        );
        same_function(
            facade::resolve_decision_labels,
            extracted::resolve_decision_labels,
        );
    }

    #[test]
    fn letter_labels_constant_is_equal_at_both_paths() {
        assert_eq!(facade::LETTER_LABELS, extracted::LETTER_LABELS);
    }
}

mod gemma_rope_parity {
    use lattice_compute::gemma_rope as extracted;
    use lattice_inference::model::gemma4_ops as facade;

    fn same_function<F>(_: F, _: F) {}

    #[test]
    fn gemma_rope_functions_are_the_same_items_at_both_paths() {
        same_function(
            facade::gemma4_rope_inv_freq,
            extracted::gemma4_rope_inv_freq,
        );
        same_function(facade::gemma4_rope_cos_sin, extracted::gemma4_rope_cos_sin);
        same_function(
            facade::vision_axial_rope_inv_freq,
            extracted::vision_axial_rope_inv_freq,
        );
    }
}

mod softmax_row_parity {
    use lattice_compute::attention::softmax_row as extracted;
    use lattice_inference::attention::softmax_row as facade;

    fn same_function<F>(_: F, _: F) {}

    #[test]
    fn softmax_row_functions_are_the_same_items_at_both_paths() {
        same_function(facade::row_max_and_any_nan, extracted::row_max_and_any_nan);
        same_function(
            facade::row_fails_closed_pre_exp,
            extracted::row_fails_closed_pre_exp,
        );
        same_function(facade::is_masked_neg_inf, extracted::is_masked_neg_inf);
        same_function(facade::finalize_row, extracted::finalize_row);
    }
}

mod cpu_forward_parity {
    use lattice_compute::forward::cpu as extracted;
    use lattice_inference::forward as forward_root;
    use lattice_inference::forward::cpu as facade;
    use std::marker::PhantomData;

    fn same_type<T>(_: PhantomData<T>, _: PhantomData<T>) {}
    fn same_function<F>(_: F, _: F) {}

    #[test]
    fn cpu_bert_policy_types_are_the_same_at_both_paths() {
        same_type(
            PhantomData::<facade::BertCpuKernelPolicy>,
            PhantomData::<extracted::BertCpuKernelPolicy>,
        );
        same_type(
            PhantomData::<forward_root::BertCpuKernelPolicy>,
            PhantomData::<extracted::BertCpuKernelPolicy>,
        );
        same_type(
            PhantomData::<lattice_inference::BertCpuKernelProfile>,
            PhantomData::<extracted::BertCpuKernelProfile>,
        );
    }

    #[test]
    fn cpu_forward_functions_are_the_same_items_at_both_paths() {
        same_function(facade::add_bias, extracted::add_bias);
        same_function(facade::add_bias_gelu, extracted::add_bias_gelu);
        same_function(facade::gelu, extracted::gelu);
        same_function(facade::sgemm_bt_strided, extracted::sgemm_bt_strided);
        same_function(facade::sgemm_nn_ab, extracted::sgemm_nn_ab);
        same_function(facade::sgemm_nn_strided, extracted::sgemm_nn_strided);
        same_function(facade::elementwise_mul, extracted::elementwise_mul);
        same_function(facade::rms_norm, extracted::rms_norm);
        same_function(facade::silu_inplace, extracted::silu_inplace);
        same_function(facade::matmul, extracted::matmul);
        same_function(facade::matmul_bt, extracted::matmul_bt);
        same_function(facade::matmul_into, extracted::matmul_into);
        same_function(facade::matmul_scalar, extracted::matmul_scalar);
        same_function(facade::layer_norm, extracted::layer_norm);
        same_function(
            facade::residual_add_layer_norm,
            extracted::residual_add_layer_norm,
        );
        same_function(facade::softmax_attention, extracted::softmax_attention);
        same_function(forward_root::matmul_bt, extracted::matmul_bt);
    }

    #[test]
    fn cpu_f16_matmul_is_the_same_item_at_both_paths() {
        same_function(
            lattice_inference::weights::f16_weights::matmul_bt_f16,
            lattice_compute::forward::cpu::f16_matmul::matmul_bt_f16,
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn cpu_neon_matmul_is_the_same_item_at_both_paths() {
        same_function(facade::matmul_neon, extracted::matmul_neon);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn cpu_x86_matmul_is_the_same_item_at_both_paths() {
        same_function(facade::matmul_avx2, extracted::matmul_avx2);
        same_function(facade::matmul_avx512, extracted::matmul_avx512);
    }
}

mod pool_parity {
    use lattice_compute::pool as extracted;
    use lattice_inference::pool as facade;
    use std::marker::PhantomData;

    fn same_type<T>(_: PhantomData<T>, _: PhantomData<T>) {}
    fn same_function<F>(_: F, _: F) {}

    #[test]
    fn pool_types_are_the_same_at_both_paths() {
        same_type(
            PhantomData::<facade::BertPooling>,
            PhantomData::<extracted::BertPooling>,
        );
        same_type(
            PhantomData::<lattice_inference::BertPooling>,
            PhantomData::<extracted::BertPooling>,
        );
    }

    #[test]
    fn pool_functions_are_the_same_items_at_both_paths() {
        same_function(
            facade::default_bert_pooling_for_model_name,
            extracted::default_bert_pooling_for_model_name,
        );
        same_function(facade::mean_pool, extracted::mean_pool);
        same_function(facade::last_token_pool, extracted::last_token_pool);
        same_function(facade::cls_pool, extracted::cls_pool);
        same_function(facade::l2_normalize, extracted::l2_normalize);
    }
}

mod attention_kernel_parity {
    use lattice_compute::attention as extracted;
    use lattice_inference::attention as facade;
    use std::marker::PhantomData;

    fn same_type<T>(_: PhantomData<T>, _: PhantomData<T>) {}
    fn same_function<F>(_: F, _: F) {}

    #[test]
    fn attention_kernel_types_are_the_same_at_both_paths() {
        same_type(
            PhantomData::<facade::gqa::GqaConfig>,
            PhantomData::<extracted::gqa::GqaConfig>,
        );
        same_type(
            PhantomData::<facade::gqa::GqaScratch>,
            PhantomData::<extracted::gqa::GqaScratch>,
        );
        same_type(
            PhantomData::<facade::differential::DiffAttnConfig>,
            PhantomData::<extracted::differential::DiffAttnConfig>,
        );
        same_type(
            PhantomData::<facade::differential::DiffLambdaParams>,
            PhantomData::<extracted::differential::DiffLambdaParams>,
        );
        same_type(
            PhantomData::<facade::differential::DiffAttnScratch>,
            PhantomData::<extracted::differential::DiffAttnScratch>,
        );
        same_type(
            PhantomData::<facade::flash_causal::FlashAttentionConfig>,
            PhantomData::<extracted::flash_causal::FlashAttentionConfig>,
        );
        same_type(
            PhantomData::<facade::native_sparse::NsaConfig>,
            PhantomData::<extracted::native_sparse::NsaConfig>,
        );
        same_type(
            PhantomData::<facade::native_sparse::NsaWeights>,
            PhantomData::<extracted::native_sparse::NsaWeights>,
        );
        same_type(
            PhantomData::<facade::native_sparse::NsaScratch>,
            PhantomData::<extracted::native_sparse::NsaScratch>,
        );
    }

    #[test]
    fn attention_kernel_functions_are_the_same_items_at_both_paths() {
        same_function(
            facade::gqa::apply_gqa_attention,
            extracted::gqa::apply_gqa_attention,
        );
        same_function(
            facade::decode::decode_attention,
            extracted::decode::decode_attention,
        );
        same_function(
            facade::decode::decode_attention_scores,
            extracted::decode::decode_attention_scores,
        );
        same_function(
            facade::differential::apply_differential_attention,
            extracted::differential::apply_differential_attention,
        );
        same_function(
            facade::differential::compute_lambda_full,
            extracted::differential::compute_lambda_full,
        );
        same_function(
            facade::flash_causal::flash_attention_causal,
            extracted::flash_causal::flash_attention_causal,
        );
        same_function(
            facade::flash_causal::flash_attention_causal_with_config,
            extracted::flash_causal::flash_attention_causal_with_config,
        );
        same_function(
            facade::gated::deinterleave_q_gate,
            extracted::gated::deinterleave_q_gate,
        );
        same_function(
            facade::gated::apply_sigmoid_gate,
            extracted::gated::apply_sigmoid_gate,
        );
        same_function(
            facade::gated::apply_sigmoid_gate_fast,
            extracted::gated::apply_sigmoid_gate_fast,
        );
        same_function(
            facade::gated::apply_sigmoid_gate_scalar,
            extracted::gated::apply_sigmoid_gate_scalar,
        );
        same_function(
            facade::gated::apply_sigmoid_gate_simd,
            extracted::gated::apply_sigmoid_gate_simd,
        );
        #[cfg(target_arch = "aarch64")]
        same_function(
            facade::gated::apply_sigmoid_gate_neon,
            extracted::gated::apply_sigmoid_gate_neon,
        );
        #[cfg(target_arch = "x86_64")]
        same_function(
            facade::gated::apply_sigmoid_gate_avx2,
            extracted::gated::apply_sigmoid_gate_avx2,
        );
        same_function(
            facade::native_sparse::apply_native_sparse_attention,
            extracted::native_sparse::apply_native_sparse_attention,
        );
    }

    #[test]
    fn attention_kernel_constants_keep_their_values() {
        assert_eq!(
            facade::flash_causal::DEFAULT_BR,
            extracted::flash_causal::DEFAULT_BR
        );
        assert_eq!(
            facade::flash_causal::DEFAULT_BC,
            extracted::flash_causal::DEFAULT_BC
        );
    }
}

mod gdn_parity {
    use lattice_compute::attention as extracted;
    use lattice_inference::attention as facade;
    use std::marker::PhantomData;

    fn same_type<T>(_: PhantomData<T>, _: PhantomData<T>) {}
    fn same_function<F>(_: F, _: F) {}

    #[test]
    fn gdn_types_are_the_same_at_both_paths() {
        same_type(
            PhantomData::<facade::gdn::GatedDeltaNetWeights>,
            PhantomData::<extracted::gdn::GatedDeltaNetWeights>,
        );
        same_type(
            PhantomData::<facade::gdn::GatedDeltaNetState>,
            PhantomData::<extracted::gdn::GatedDeltaNetState>,
        );
        same_type(
            PhantomData::<facade::gdn::GatedDeltaNetScratch>,
            PhantomData::<extracted::gdn::GatedDeltaNetScratch>,
        );
        same_type(
            PhantomData::<facade::gdn::GdnLayerSnapshot>,
            PhantomData::<extracted::gdn::GdnLayerSnapshot>,
        );
        same_type(
            PhantomData::<facade::gdn::GdnSnapshot>,
            PhantomData::<extracted::gdn::GdnSnapshot>,
        );
        same_type(
            PhantomData::<facade::gdn_fused::GatedDeltaNetFusedScratch>,
            PhantomData::<extracted::gdn_fused::GatedDeltaNetFusedScratch>,
        );
        same_type(
            PhantomData::<facade::gdn::GdnDimensions>,
            PhantomData::<extracted::gdn::GdnDimensions>,
        );
        same_type(
            PhantomData::<facade::gdn::GdnGeometry>,
            PhantomData::<extracted::gdn::GdnGeometry>,
        );
    }

    /// The facade keeps the configuration-taking step signatures; `lattice-compute`'s steps take
    /// a validated geometry instead.
    #[test]
    #[allow(clippy::type_complexity)] // the spelled-out signature is what this test pins
    fn gdn_configuration_steps_keep_their_facade_signatures() {
        use lattice_inference::model::qwen35_config::Qwen35Config;

        let _: fn(
            &[f32],
            &mut facade::gdn::GatedDeltaNetState,
            &facade::gdn::GatedDeltaNetWeights,
            &Qwen35Config,
            &mut facade::gdn::GatedDeltaNetScratch,
            &mut [f32],
        ) = facade::gdn::gated_delta_net_step;
        let _: fn(
            &[f32],
            &mut facade::gdn::GatedDeltaNetState,
            &facade::gdn::GatedDeltaNetWeights,
            &Qwen35Config,
            &mut facade::gdn_fused::GatedDeltaNetFusedScratch,
            &mut [f32],
            &dyn lattice_inference::lora_hook::LoraHook,
            usize,
        ) = facade::gdn_fused::gated_delta_net_step_fused;
        let _: fn(
            &[f32],
            &mut extracted::gdn::GatedDeltaNetState,
            &extracted::gdn::GatedDeltaNetWeights,
            &extracted::gdn::GdnGeometry,
            &mut extracted::gdn::GatedDeltaNetScratch,
            &mut [f32],
        ) = extracted::gdn::gated_delta_net_step;
        let _: fn(
            &[f32],
            &mut extracted::gdn::GatedDeltaNetState,
            &extracted::gdn::GatedDeltaNetWeights,
            &extracted::gdn::GdnGeometry,
            &mut extracted::gdn_fused::GatedDeltaNetFusedScratch,
            &mut [f32],
            &dyn lattice_compute::lora_hook::LoraHook,
            usize,
        ) = extracted::gdn_fused::gated_delta_net_step_fused;
    }

    #[test]
    fn gdn_functions_are_the_same_items_at_both_paths() {
        same_function(facade::gdn::gated_rms_norm, extracted::gdn::gated_rms_norm);
        same_function(
            facade::gdn::l2_normalize_vec,
            extracted::gdn::l2_normalize_vec,
        );
        same_function(facade::gdn::sigmoid, extracted::gdn::sigmoid);
        same_function(facade::gdn::softplus, extracted::gdn::softplus);
        same_function(
            facade::gdn_fused::conv1d_silu_fused,
            extracted::gdn_fused::conv1d_silu_fused,
        );
        same_function(
            facade::gdn_fused::simd_l2_normalize,
            extracted::gdn_fused::simd_l2_normalize,
        );
        same_function(
            facade::gdn_fused::simd_matvec_transpose,
            extracted::gdn_fused::simd_matvec_transpose,
        );
        same_function(
            facade::gdn_fused::simd_decay_and_rank1_update,
            extracted::gdn_fused::simd_decay_and_rank1_update,
        );
        same_function(
            facade::gdn_fused::simd_gated_rms_norm,
            extracted::gdn_fused::simd_gated_rms_norm,
        );
    }
}

mod attention_standard_parity {
    use lattice_compute::attention as extracted;
    use lattice_inference::attention as facade;
    use std::marker::PhantomData;

    fn same_type<T>(_: PhantomData<T>, _: PhantomData<T>) {}
    fn same_function<F>(_: F, _: F) {}

    #[test]
    fn standard_and_tiled_attention_types_are_the_same_at_both_paths() {
        same_type(
            PhantomData::<facade::standard::AttentionBuffers>,
            PhantomData::<extracted::standard::AttentionBuffers>,
        );
        same_type(
            PhantomData::<facade::AttentionBuffers>,
            PhantomData::<extracted::standard::AttentionBuffers>,
        );
        same_type(
            PhantomData::<facade::flash::TiledAttentionConfig>,
            PhantomData::<extracted::flash::TiledAttentionConfig>,
        );
        same_type(
            PhantomData::<facade::flash::TiledAttentionBuffers>,
            PhantomData::<extracted::flash::TiledAttentionBuffers>,
        );
        same_type(
            PhantomData::<facade::AttentionKind>,
            PhantomData::<extracted::AttentionKind>,
        );
        same_type(
            PhantomData::<facade::AttentionTag>,
            PhantomData::<extracted::AttentionTag>,
        );
    }

    #[test]
    fn standard_and_tiled_attention_functions_are_the_same_items_at_both_paths() {
        same_function(
            facade::flash::estimate_materialized_attention_buffer_bytes,
            extracted::flash::estimate_materialized_attention_buffer_bytes,
        );
        same_function(
            facade::flash::tiled_attention_head,
            extracted::flash::tiled_attention_head,
        );
    }

    /// The facade keeps the BERT-layer attention signatures; `lattice-compute`'s kernels take the
    /// layer's projection view instead.
    #[test]
    #[allow(clippy::type_complexity)] // the spelled-out signature is what this test pins
    fn attention_entry_points_keep_their_facade_signatures() {
        use lattice_inference::lora_hook::LoraHook;
        use lattice_inference::weights::TransformerLayerWeights;

        same_type(
            PhantomData::<facade::standard::AttentionProjections<'static>>,
            PhantomData::<extracted::standard::AttentionProjections<'static>>,
        );
        let _: fn(
            &[f32],
            &TransformerLayerWeights<'_>,
            &[u32],
            usize,
            usize,
            usize,
            usize,
            &mut facade::AttentionBuffers,
            &dyn LoraHook,
            usize,
        ) -> Vec<f32> = facade::standard::multi_head_attention;
        same_function(
            facade::multi_head_attention,
            facade::standard::multi_head_attention,
        );
        let _: fn(
            &[f32],
            &extracted::standard::AttentionProjections<'_>,
            &[u32],
            usize,
            usize,
            usize,
            usize,
            &mut extracted::standard::AttentionBuffers,
            &dyn lattice_compute::lora_hook::LoraHook,
            usize,
        ) -> Vec<f32> = extracted::standard::multi_head_attention;
        let _: fn(
            &[f32],
            &TransformerLayerWeights<'_>,
            &[u32],
            usize,
            usize,
            usize,
            usize,
            usize,
            &mut facade::AttentionBuffers,
            &mut facade::flash::TiledAttentionBuffers,
            &facade::flash::TiledAttentionConfig,
        ) -> Vec<f32> = facade::flash::tiled_multi_head_attention;
        let _: fn(
            &[f32],
            &extracted::standard::AttentionProjections<'_>,
            &[u32],
            usize,
            usize,
            usize,
            usize,
            usize,
            &mut extracted::standard::AttentionBuffers,
            &mut extracted::flash::TiledAttentionBuffers,
            &extracted::flash::TiledAttentionConfig,
        ) -> Vec<f32> = extracted::flash::tiled_multi_head_attention;
        #[cfg(feature = "bench-internals")]
        {
            let _: fn(
                &[f32],
                &TransformerLayerWeights<'_>,
                &[f32],
                &[f32],
                &[usize],
                usize,
                usize,
                usize,
                &mut [f32],
                &mut [f32],
                &mut [f32],
                &mut [f32],
                &mut [f32],
                &mut [f32],
                &dyn LoraHook,
                usize,
            ) = facade::standard::bench_support::multi_head_attention_batched;
            same_function(
                facade::bench_support::multi_head_attention_batched,
                facade::standard::bench_support::multi_head_attention_batched,
            );
        }
    }

    #[test]
    fn attention_kind_reports_the_same_tag_through_the_facade() {
        let cfg = facade::gqa::GqaConfig {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 64,
        };
        let kind = facade::AttentionKind::Gqa(cfg);
        assert_eq!(kind.tag(), extracted::AttentionTag::Gqa);
        assert_eq!(kind.name(), "gqa");
        assert!(kind.is_causal());
        assert!(kind.supports_kv_cache());
    }
}

mod model_io_parity {
    use lattice_inference::download as facade;
    use lattice_model_io::download as extracted;

    fn same_function<F>(_: F, _: F) {}

    #[test]
    fn ensure_model_files_is_the_same_function_at_both_paths() {
        same_function(facade::ensure_model_files, extracted::ensure_model_files);
    }
}

#[cfg(feature = "embeddinggemma2")]
mod embeddinggemma2_parity {
    use std::marker::PhantomData;

    fn same_type<T>(_: PhantomData<T>, _: PhantomData<T>) {}

    #[test]
    fn embeddinggemma2_items_keep_the_same_identity() {
        same_type(
            PhantomData::<lattice_inference::model::embeddinggemma2::EmbeddingGemma2Task>,
            PhantomData::<lattice_model_embeddinggemma2::model::EmbeddingGemma2Task>,
        );
        same_type(
            PhantomData::<lattice_inference::model::embeddinggemma2::EmbeddingGemma2Part>,
            PhantomData::<lattice_model_embeddinggemma2::model::EmbeddingGemma2Part>,
        );
        same_type(
            PhantomData::<lattice_inference::model::embeddinggemma2::EmbeddingGemma2PartCounts>,
            PhantomData::<lattice_model_embeddinggemma2::model::EmbeddingGemma2PartCounts>,
        );
        same_type(
            PhantomData::<lattice_inference::model::embeddinggemma2::EmbeddingGemma2PartsEncoding>,
            PhantomData::<lattice_model_embeddinggemma2::model::EmbeddingGemma2PartsEncoding>,
        );
        same_type(
            PhantomData::<lattice_inference::model::embeddinggemma2::EmbeddingGemma2ImageStages>,
            PhantomData::<lattice_model_embeddinggemma2::model::EmbeddingGemma2ImageStages>,
        );
        same_type(
            PhantomData::<lattice_inference::model::embeddinggemma2::EmbeddingGemma2Model>,
            PhantomData::<lattice_model_embeddinggemma2::model::EmbeddingGemma2Model>,
        );
        same_type(
            PhantomData::<lattice_inference::model::EmbeddingGemma2Model>,
            PhantomData::<lattice_model_embeddinggemma2::EmbeddingGemma2Model>,
        );
        same_type(
            PhantomData::<lattice_inference::model::embeddinggemma2_image::EmbeddingGemma2Rgb8Image>,
            PhantomData::<lattice_model_embeddinggemma2::image::EmbeddingGemma2Rgb8Image>,
        );
        same_type(
            PhantomData::<
                lattice_inference::forward::metal_embeddinggemma2::MetalEmbeddingGemma2State,
            >,
            PhantomData::<lattice_model_embeddinggemma2::metal::MetalEmbeddingGemma2State>,
        );
        same_type(
            PhantomData::<lattice_inference::forward::metal_embeddinggemma2::VisionTimings>,
            PhantomData::<lattice_model_embeddinggemma2::metal::VisionTimings>,
        );
        same_type(
            PhantomData::<lattice_inference::encoder::EmbeddingGemma2Encoder>,
            PhantomData::<lattice_model_embeddinggemma2::encoder::EmbeddingGemma2Encoder>,
        );
        let _: fn(&str, &str) -> String =
            lattice_inference::model::embeddinggemma2::format_titled_document;
        let _: fn(&str, &str) -> String =
            lattice_model_embeddinggemma2::model::format_titled_document;
        assert_eq!(
            lattice_inference::model::embeddinggemma2::EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS,
            lattice_model_embeddinggemma2::model::EMBEDDINGGEMMA2_DEFAULT_MAX_TOKENS
        );
        assert!(std::ptr::eq(
            &lattice_inference::family::EMBEDDING_GEMMA2,
            &lattice_model_embeddinggemma2::family::EMBEDDING_GEMMA2
        ));
    }
}

#[test]
fn vision_errors_keep_the_same_identity() {
    let _: fn(
        lattice_inference::vision::MultimodalInput,
    ) -> lattice_model_io::multimodal::MultimodalInput = |input| input;
    let _: fn(lattice_inference::vision::VisionError) -> lattice_model_io::image::VisionError =
        |error| error;
    let _: fn(lattice_model_io::image::VisionError) -> lattice_inference::vision::VisionError =
        |error| error;
}

#[test]
fn gemma_kernels_keep_both_named_paths() {
    type ScaledEmbedding = fn(&[u32], &[f32], usize, &mut [f32]);
    type LogitSoftcap = fn(&mut [f32], f32);
    let _: (ScaledEmbedding, ScaledEmbedding) = (
        lattice_inference::model::gemma4_ops::gemma4_scaled_embedding,
        lattice_compute::gemma_ops::gemma4_scaled_embedding,
    );
    let _: (LogitSoftcap, LogitSoftcap) = (
        lattice_inference::model::gemma4_ops::gemma4_logit_softcap,
        lattice_compute::gemma_ops::gemma4_logit_softcap,
    );
    type RmsNorm = fn(&mut [f32], &[f32], usize, f32);
    type Gelu = fn(&mut [f32]);
    type Geglu =
        fn(&[f32], &[f32], &[f32], &[f32], usize, usize, usize, &mut [f32], &mut [f32], &mut [f32]);
    type QkNorm = fn(&mut [f32], &mut [f32], &mut [f32], &[f32], &[f32], usize, f32);
    type Rope = fn(&mut [f32], &[f32], &[f32], usize, usize, usize);
    type AxialRope = fn(&mut [f32], &[[usize; 2]], &[f32], usize, usize);
    let _: (RmsNorm, RmsNorm) = (
        lattice_inference::model::gemma4_ops::gemma4_rms_norm,
        lattice_compute::gemma_ops::gemma4_rms_norm,
    );
    let _: (Gelu, Gelu) = (
        lattice_inference::model::gemma4_ops::gemma4_gelu_tanh,
        lattice_compute::gemma_ops::gemma4_gelu_tanh,
    );
    let _: (Geglu, Geglu) = (
        lattice_inference::model::gemma4_ops::gemma4_geglu_mlp,
        lattice_compute::gemma_ops::gemma4_geglu_mlp,
    );
    let _: (QkNorm, QkNorm) = (
        lattice_inference::model::gemma4_ops::gemma4_qk_norm_v_unscaled,
        lattice_compute::gemma_ops::gemma4_qk_norm_v_unscaled,
    );
    let _: (Rope, Rope) = (
        lattice_inference::model::gemma4_ops::gemma4_apply_rope,
        lattice_compute::gemma_ops::gemma4_apply_rope,
    );
    let _: (AxialRope, AxialRope) = (
        lattice_inference::model::gemma4_ops::vision_axial_rope_apply,
        lattice_compute::gemma_ops::vision_axial_rope_apply,
    );
}

#[cfg(feature = "gemma4")]
#[test]
fn model_cache_and_preflight_paths_name_family_items() {
    use lattice_inference::model as facade;
    use lattice_model_gemma4 as family;
    use std::marker::PhantomData;
    let _: PhantomData<family::Gemma4Model> = PhantomData::<facade::Gemma4Model>;
    let _: PhantomData<family::Gemma4Model> = PhantomData::<facade::gemma4_model::Gemma4Model>;
    let _: PhantomData<family::gemma4_cache::Gemma4KvCache> =
        PhantomData::<facade::gemma4_cache::Gemma4KvCache>;
    let _: PhantomData<family::gemma4_preflight::TensorInfo> =
        PhantomData::<facade::gemma4_preflight::TensorInfo>;
    let _: PhantomData<family::gemma4_preflight::TensorInventory> =
        PhantomData::<facade::gemma4_preflight::TensorInventory>;
    let _: PhantomData<family::gemma4_preflight::PreflightReport> =
        PhantomData::<facade::gemma4_preflight::PreflightReport>;
    let _: fn(
        &std::path::Path,
    ) -> Result<family::Gemma4Model, lattice_inference_core::InferenceError> =
        facade::Gemma4Model::from_safetensors;
    type Check = fn(
        &lattice_arch::gemma4_config::Gemma4Config,
        &std::collections::HashMap<String, family::gemma4_preflight::TensorInfo>,
    ) -> Result<
        family::gemma4_preflight::PreflightReport,
        lattice_inference_core::InferenceError,
    >;
    let old: Check = facade::gemma4_preflight::preflight_check;
    let new: Check = family::gemma4_preflight::preflight_check;
    assert!(std::ptr::fn_addr_eq(old, new));
}
