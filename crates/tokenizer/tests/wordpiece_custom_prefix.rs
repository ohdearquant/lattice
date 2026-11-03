use lattice_tokenizer::{Tokenizer, WordPieceTokenizer};

fn ids(json: &str, text: &str) -> Vec<u32> {
    let tokenizer = WordPieceTokenizer::from_tokenizer_json_str(json).unwrap();
    let input = tokenizer.tokenize(text);
    input.input_ids[..input.real_length].to_vec()
}

#[test]
fn custom_continuation_prefix_matches_hf() {
    let json = r#"{"model":{"type":"WordPiece","continuing_subword_prefix":"~","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"~b":6}}}"#;
    assert_eq!(ids(json, "ab"), vec![2, 5, 6, 3]);
}

#[test]
fn empty_prefix_allows_raw_tokens_as_continuations() {
    let json = r#"{"model":{"type":"WordPiece","continuing_subword_prefix":"","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"b":6}}}"#;
    assert_eq!(ids(json, "ab"), vec![2, 5, 6, 3]);
}

#[test]
fn prefix_leading_vocab_token_is_also_a_whole_word() {
    let json = r#"{"model":{"type":"WordPiece","continuing_subword_prefix":"z","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"za":5,"a":6}}}"#;
    assert_eq!(ids(json, "za"), vec![2, 5, 3]);
    assert_eq!(ids(json, "aa"), vec![2, 6, 5, 3]);
}

#[test]
fn absent_prefix_defaults_to_hash_pair() {
    let json = r###"{"model":{"type":"WordPiece","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"##b":6}}}"###;
    assert_eq!(ids(json, "ab"), vec![2, 5, 6, 3]);
}

#[test]
fn custom_prefix_survives_tokenizer_rebuild_and_reports_setting() {
    let json = r#"{"model":{"type":"WordPiece","continuing_subword_prefix":"~","vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"~b":6}}}"#;
    let tokenizer = WordPieceTokenizer::from_tokenizer_json_str(json).unwrap();
    let rebuilt = tokenizer.with_max_seq_len(64);
    for value in [&tokenizer, &rebuilt] {
        let input = value.tokenize("ab");
        assert_eq!(&input.input_ids[..input.real_length], &[2, 5, 6, 3]);
        assert!(value.embedding_settings().unwrap().contains(&(
            "continuing_subword_prefix".into(),
            lattice_tokenizer::common::TokenizerPolicyValue::String("~".into()),
        )));
    }
}

#[test]
fn non_string_continuation_prefix_is_refused() {
    for value in ["null", "true", "1"] {
        let json = format!(
            r#"{{"model":{{"type":"WordPiece","continuing_subword_prefix":{value},"vocab":{{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4}}}}}}"#
        );
        let err = WordPieceTokenizer::from_tokenizer_json_str(&json).unwrap_err();
        assert!(err.to_string().contains("continuing_subword_prefix"));
    }
}
