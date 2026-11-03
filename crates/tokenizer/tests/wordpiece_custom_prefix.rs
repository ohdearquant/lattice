use lattice_tokenizer::WordPieceTokenizer;

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
