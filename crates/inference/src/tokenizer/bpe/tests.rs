use crate::tokenizer::bpe::BpeTokenizer;
use lattice_tokenizer::Tokenizer;
use std::collections::HashMap;

fn bytes_to_unicode() -> Vec<char> {
    lattice_tokenizer::__internal::bytes_to_unicode()
}

#[test]
fn test_grammar_vocab_preserves_split_utf8_bytes() {
    use crate::grammar::{GrammarEngine, GrammarSpec};

    let byte_encoder = bytes_to_unicode();
    let mut vocab = HashMap::new();
    vocab.insert(byte_encoder[0xE5].to_string(), 0);
    vocab.insert(byte_encoder[0xA5].to_string(), 1);
    vocab.insert(byte_encoder[0xBD].to_string(), 2);
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, Vec::new()).unwrap();
    let vocab_bytes = tokenizer.vocab_bytes(3).unwrap();
    assert_eq!(vocab_bytes, vec![vec![0xE5], vec![0xA5], vec![0xBD]]);

    let spec = GrammarSpec::Gbnf("root ::= \"好\"\n".to_string());
    let engine = GrammarEngine::new(&spec, vocab_bytes).unwrap();
    let mut state = engine.initial_state();
    for token_id in 0..3u32 {
        let mut logits = vec![0.0; 3];
        engine
            .mask_logits(&mut state, &mut logits)
            .expect("matching vocab length");
        assert!(
            logits[token_id as usize].is_finite(),
            "split UTF-8 token {token_id} must remain grammar-legal"
        );
        assert!(engine.advance(&mut state, token_id).unwrap());
    }
}

#[test]
fn test_grammar_vocab_masks_rendered_added_tokens_and_controls() {
    use crate::grammar::{GrammarEngine, GrammarSpec};

    let json = r#"{
            "model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]},
            "added_tokens":[
                {"id":99,"content":"<control>","special":true},
                {"id":100,"content":"<extra>","special":false}
            ]
        }"#;
    let tokenizer = BpeTokenizer::from_tokenizer_json_str(json).unwrap();
    let vocab_bytes = tokenizer.vocab_bytes(101).unwrap();
    assert_eq!(vocab_bytes.len(), 101);
    assert!(vocab_bytes[99].is_empty());
    assert_eq!(vocab_bytes[100], b"<extra>");

    let spec = GrammarSpec::Gbnf("root ::= \"a\"\n".to_string());
    let engine = GrammarEngine::new(&spec, vocab_bytes).unwrap();
    let mut state = engine.initial_state();
    let mut logits = vec![0.0; 101];
    engine
        .mask_logits(&mut state, &mut logits)
        .expect("matching vocab length");
    assert_eq!(logits[99], f32::NEG_INFINITY);
    assert_eq!(logits[100], f32::NEG_INFINITY);
}

fn non_ascii_added_tokenizer() -> BpeTokenizer {
    let json = r#"{
            "model":{"type":"BPE","vocab":{"a":0},"merges":[]},
            "added_tokens":[
                {"id":100,"content":"好","special":false},
                {"id":101,"content":"café","special":false}
            ]
        }"#;
    BpeTokenizer::from_tokenizer_json_str(json).unwrap()
}

#[test]
fn grammar_vocab_fail_closes_qwen_reserved_tail() {
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use crate::model::qwen35_config::Qwen35Config;

    let json = r#"{
            "model":{"type":"BPE","vocab":{"a":0},"merges":[]},
            "added_tokens":[
                {"id":248069,"content":"<control>","special":true}
            ]
        }"#;
    let tokenizer = BpeTokenizer::from_tokenizer_json_str(json).unwrap();
    let cfg = Qwen35Config::qwen35_0_8b();
    let vocab_bytes = tokenizer.vocab_bytes(cfg.vocab_size).unwrap();
    let table_len = vocab_bytes.len();
    let spec = GrammarSpec::Gbnf("root ::= \"a\"\n".to_string());
    let engine = GrammarEngine::new(&spec, vocab_bytes).unwrap();
    let mut state = engine.initial_state();
    let mut logits = vec![0.0; cfg.vocab_size];

    engine
        .mask_logits(&mut state, &mut logits)
        .expect("matching vocab length");

    assert!(
        logits[248_070..]
            .iter()
            .all(|logit| *logit == f32::NEG_INFINITY),
        "reserved logits above the tokenizer maximum must be masked"
    );
    assert_eq!(table_len, cfg.vocab_size);
}

#[test]
fn grammar_vocab_uses_literal_non_ascii_added_token_bytes() {
    use crate::grammar::{GrammarEngine, GrammarSpec};

    let tokenizer = non_ascii_added_tokenizer();
    let err = tokenizer.vocab_bytes(101).unwrap_err();
    assert!(err.to_string().contains("token ID 101"));
    let vocab_bytes = tokenizer.vocab_bytes(102).unwrap();
    assert_eq!(vocab_bytes[100], "好".as_bytes());
    assert_eq!(vocab_bytes[101], "café".as_bytes());

    let spec = GrammarSpec::Gbnf("root ::= \"好\" | \"café\"\n".to_string());
    let engine = GrammarEngine::new(&spec, vocab_bytes).unwrap();
    let mut state = engine.initial_state();
    let mut logits = vec![0.0; 102];
    engine
        .mask_logits(&mut state, &mut logits)
        .expect("matching vocab length");
    assert!(logits[100].is_finite());
    assert!(logits[101].is_finite());
}

#[test]
fn test_grammar_vocab_masks_special_added_token_also_in_base_vocab() {
    // Regression (#1722): an `added_tokens` entry marked `special: true`
    // whose id ALSO appears in `model.vocab` used to leak the base-vocab
    // spelling into the grammar byte table and into decode, because
    // nothing removed the id from `id_to_token`. This is the exact shape
    // from the report: id 1 is both `model.vocab["<c>"]` and a special
    // added token.
    use crate::grammar::{GrammarEngine, GrammarSpec};

    let json = r#"{
            "model":{"type":"BPE","vocab":{"a":0,"<c>":1},"merges":[]},
            "added_tokens":[{"id":1,"content":"<c>","special":true}]
        }"#;
    let tokenizer = BpeTokenizer::from_tokenizer_json_str(json).unwrap();

    let vocab_bytes = tokenizer.vocab_bytes(2).unwrap();
    assert!(
        vocab_bytes[1].is_empty(),
        "special id 1 must get no bytes even though it also appears in model.vocab, \
             got {:?}",
        vocab_bytes[1]
    );
    assert_eq!(
        tokenizer.token_bytes_for_id(1),
        None,
        "token_bytes_for_id must return None for a special id, per its own doc"
    );
    assert_eq!(
        tokenizer.token_for_id(1),
        None,
        "token_for_id must swallow a special id exactly like an id absent from vocab"
    );
    // Decode must swallow the special id exactly as it would if id 1 were
    // absent from `model.vocab` entirely: the control token contributes
    // zero bytes, and the surrounding content token still decodes.
    assert_eq!(tokenizer.decode(&[0, 1]), Some("a".to_string()));

    // A grammar that only accepts "a" must therefore mask id 1 out: before
    // the fix, id 1 decoded to b"<c>" and nothing in this grammar's byte
    // table would have masked it on structural grounds unrelated to this
    // bug, so this assertion is the direct behavioral consequence lattice#1722
    // reports (a grammar-illegal control token reachable through decoding).
    let spec = GrammarSpec::Gbnf("root ::= \"a\"\n".to_string());
    let engine = GrammarEngine::new(&spec, vocab_bytes).unwrap();
    let mut state = engine.initial_state();
    let mut logits = vec![0.0; 2];
    engine
        .mask_logits(&mut state, &mut logits)
        .expect("matching vocab length");
    assert!(logits[0].is_finite(), "token \"a\" must stay grammar-legal");
    assert_eq!(
        logits[1],
        f32::NEG_INFINITY,
        "special id 1 must be masked even though its base-vocab spelling is <c>"
    );
}
