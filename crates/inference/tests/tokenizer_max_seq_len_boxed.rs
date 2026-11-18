//! `Tokenizer::with_max_seq_len_boxed` gives a copy that tokenizes exactly as a
//! tokenizer constructed at that limit, for each tokenizer format the native
//! embedding models load.

use lattice_tokenizer::tokenizer_from_json_str;

const LONG_TEXT: &str = "the quick brown fox jumps over the lazy dog while the small \
    cat watches from the warm window and the rain keeps falling on the quiet street";

const FIXTURES: [(&str, &str); 3] = [
    (
        "wordpiece",
        include_str!("fixtures/tokenizers/bge-small-en-v1.5/tokenizer.json"),
    ),
    (
        "unigram",
        include_str!("fixtures/tokenizers/multilingual-e5-small/tokenizer.json"),
    ),
    (
        "bpe",
        include_str!("fixtures/tokenizers/qwen3-embedding-0.6b/tokenizer.json"),
    ),
];

#[test]
fn a_boxed_copy_tokenizes_as_one_built_at_its_limit() {
    for (format, json) in FIXTURES {
        let loaded = tokenizer_from_json_str(json, 512).unwrap();
        let full = loaded.tokenize(LONG_TEXT);
        assert!(
            full.real_length > 12,
            "{format}: the text must outrun the limit, got {} tokens",
            full.real_length
        );

        let limited = loaded
            .with_max_seq_len_boxed(12)
            .unwrap_or_else(|| panic!("{format}: no boxed copy"));
        assert_eq!(limited.max_seq_len(), 12, "{format}");
        let built = tokenizer_from_json_str(json, 12).unwrap();
        let got = limited.tokenize(LONG_TEXT);
        let want = built.tokenize(LONG_TEXT);
        assert_eq!(got.real_length, 12, "{format}");
        assert_eq!(got.input_ids, want.input_ids, "{format}");
        assert_eq!(got.attention_mask, want.attention_mask, "{format}");

        assert_eq!(
            loaded.max_seq_len(),
            512,
            "{format}: the original keeps its limit"
        );
    }
}
