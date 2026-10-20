use super::*;

#[test]
fn test_parse_json_round_trip_shapes() {
    let json = parse_json(
        r#"{
                "model": {"type": "WordPiece", "vocab": {"[PAD]": 0, "hello": 1}},
                "added_tokens": [{"id": 42, "content": "<bos>"}],
                "array": [true, false, null, 1.5, "x"]
            }"#,
    )
    .unwrap();

    assert_eq!(
        json_path(&json, &["model", "type"]).and_then(JsonValue::as_str),
        Some("WordPiece")
    );
    assert_eq!(
        json_path(&json, &["model", "vocab", "hello"]).and_then(JsonValue::as_u64),
        Some(1)
    );
    assert_eq!(parse_added_tokens(&json).get("<bos>").copied(), Some(42));
}

#[test]
fn test_parse_rendered_added_tokens_filters_special() {
    let json = parse_json(
        r#"{
                "added_tokens": [
                    {"id": 1, "content": "<|im_start|>", "special": true},
                    {"id": 2, "content": "</think>", "special": false},
                    {"id": 3, "content": "<think>"},
                    {"id": 4, "content": "<tool_call>", "special": false}
                ]
            }"#,
    )
    .unwrap();

    let rendered = parse_rendered_added_tokens(&json);
    // Keyed by id, so the map answers "what does this id render as".
    // special=true is excluded (HF skip_special_tokens=True swallows it).
    assert!(!rendered.values().any(|c| c == "<|im_start|>"));
    // special=false renders verbatim.
    assert_eq!(rendered.get(&2).map(String::as_str), Some("</think>"));
    // absent special field defaults to false → renders.
    assert_eq!(rendered.get(&3).map(String::as_str), Some("<think>"));
    assert_eq!(rendered.get(&4).map(String::as_str), Some("<tool_call>"));
    assert_eq!(rendered.len(), 3);
}

#[test]
fn test_lru_cache_evicts_oldest() {
    let cache = ThreadSafeLruCache::<String, Vec<u32>>::new(2);
    cache.insert("a".to_string(), vec![1]);
    cache.insert("b".to_string(), vec![2]);
    assert_eq!(
        cache.get(&"a".to_string()).map(|v| v.as_slice().to_vec()),
        Some(vec![1])
    );
    cache.insert("c".to_string(), vec![3]);
    assert!(cache.get(&"b".to_string()).is_none());
    assert_eq!(
        cache.get(&"a".to_string()).map(|v| v.as_slice().to_vec()),
        Some(vec![1])
    );
    assert_eq!(
        cache.get(&"c".to_string()).map(|v| v.as_slice().to_vec()),
        Some(vec![3])
    );
}

#[test]
fn test_invert_vocab() {
    let mut vocab = HashMap::new();
    vocab.insert("[PAD]".to_string(), 0);
    vocab.insert("hello".to_string(), 3);
    let inverse = invert_vocab(&vocab).unwrap();
    assert_eq!(inverse[0], "[PAD]");
    assert_eq!(inverse[3], "hello");
}

#[test]
fn test_vocab_id_overflowing_u32_rejected() {
    // A token id past u32::MAX would wrap under `as u32` (2^32 -> 0) and alias another
    // token. json_object_to_vocab must reject it at parse time, not corrupt the vocab.
    let json = parse_json(r#"{"[PAD]": 4294967296}"#).unwrap();
    assert!(
        json_object_to_vocab(&json).is_err(),
        "a vocab id past u32::MAX must be rejected, not wrapped"
    );
}

#[test]
fn test_vocab_id_at_u32_max_accepted() {
    // u32::MAX itself is in range and must be accepted (not rejected) — locks the
    // try_from boundary against an off-by-one that would reject the largest valid id.
    let json = parse_json(r#"{"x": 4294967295}"#).unwrap();
    let vocab = json_object_to_vocab(&json).expect("u32::MAX is a valid id");
    assert_eq!(vocab.get("x"), Some(&u32::MAX));
}

#[test]
fn test_invert_vocab_rejects_disproportionate_sparse_id() {
    // A vocabulary whose largest id vastly exceeds its entry count would force a giant
    // dense-table allocation (a single id of 4_294_967_295 -> ~100 GB) and abort the
    // process at load. invert_vocab caps the table at 16x the entry count and rejects.
    // Here: 1 entry, id 50 -> 51 slots, well past the 16x cap.
    let mut vocab = HashMap::new();
    vocab.insert("a".to_string(), 50u32);
    assert!(
        invert_vocab(&vocab).is_err(),
        "a single sparse id far past the entry count must be rejected"
    );
    // A normal gapped vocab (within the cap) still inverts fine — no regression.
    let mut dense = HashMap::new();
    dense.insert("a".to_string(), 0u32);
    dense.insert("b".to_string(), 3u32);
    assert!(invert_vocab(&dense).is_ok());
}

#[test]
fn test_invert_vocab_cap_boundary() {
    // The cap is `max_id >= len*16`. For a single entry (len 1, cap 16): id 15 -> 16
    // slots is the largest accepted table; id 16 trips the cap. Locks the off-by-one.
    let mut at_limit = HashMap::new();
    at_limit.insert("a".to_string(), 15u32);
    assert!(invert_vocab(&at_limit).is_ok(), "max_id == len*16 - 1 fits");
    let mut over_limit = HashMap::new();
    over_limit.insert("a".to_string(), 16u32);
    assert!(
        invert_vocab(&over_limit).is_err(),
        "max_id == len*16 must trip the cap"
    );
}

#[test]
fn test_added_token_id_overflowing_u32_skipped() {
    // parse_added_tokens is lenient: an id past u32::MAX is skipped, not wrapped into a
    // colliding id. A well-formed sibling entry is still parsed.
    let root = parse_json(
        r#"{"added_tokens":[{"content":"<bad>","id":4294967296},{"content":"<ok>","id":7}]}"#,
    )
    .unwrap();
    let tokens = parse_added_tokens(&root);
    assert_eq!(tokens.get("<ok>"), Some(&7));
    assert!(
        !tokens.contains_key("<bad>"),
        "an added-token id past u32::MAX must be skipped, not wrapped to a colliding id"
    );
}

#[test]
fn test_push_eos_preserving_limit_replaces_tail() {
    let mut ids = vec![10, 11];
    push_eos_preserving_limit(&mut ids, 99, 2);
    assert_eq!(ids, vec![10, 99]);

    push_eos_preserving_limit(&mut ids, 99, 0);
    assert!(ids.is_empty());
}

#[test]
fn test_parse_post_processor_template_special_ids() {
    // Synthetic TemplateProcessing where the template uses [CLS]/[SEP]
    // (e.g. BERT-style) but the model also happens to have <s>/</s> in vocab.
    // The IDs in special_tokens are the authoritative source for prefix/suffix.
    let raw = r#"{
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
                ],
                "special_tokens": {
                    "[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
                    "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
                }
            }
        }"#;
    let root = parse_json(raw).unwrap();
    let pp = parse_post_processor_flags(&root);
    assert!(pp.add_bos, "template starts with SpecialToken → add_bos");
    assert!(pp.add_eos, "template ends with SpecialToken → add_eos");
    assert_eq!(pp.bos_id, Some(101), "[CLS] id authoritative");
    assert_eq!(pp.eos_id, Some(102), "[SEP] id authoritative");
}

#[test]
fn test_parse_post_processor_xlm_roberta_style() {
    // XLM-RoBERTa-style: <s>/</s> in template AND in special_tokens.
    let raw = r#"{
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "<s>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "</s>", "type_id": 0}}
                ],
                "special_tokens": {
                    "<s>": {"id": "<s>", "ids": [0], "tokens": ["<s>"]},
                    "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
                }
            }
        }"#;
    let root = parse_json(raw).unwrap();
    let pp = parse_post_processor_flags(&root);
    assert_eq!(pp.bos_id, Some(0));
    assert_eq!(pp.eos_id, Some(2));
}
