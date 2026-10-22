use super::*;

fn synthetic_bpe() -> BpeTokenizer {
    let mut vocab = HashMap::new();
    vocab.insert("h".to_string(), 0);
    vocab.insert("e".to_string(), 1);
    vocab.insert("l".to_string(), 2);
    vocab.insert("o".to_string(), 3);
    vocab.insert("Ġ".to_string(), 4);
    vocab.insert("w".to_string(), 5);
    vocab.insert("r".to_string(), 6);
    vocab.insert("d".to_string(), 7);
    vocab.insert("he".to_string(), 8);
    vocab.insert("hel".to_string(), 9);
    vocab.insert("hell".to_string(), 10);
    vocab.insert("hello".to_string(), 11);
    vocab.insert("Ġw".to_string(), 12);
    vocab.insert("Ġwo".to_string(), 13);
    vocab.insert("Ġwor".to_string(), 14);
    vocab.insert("Ġworl".to_string(), 15);
    vocab.insert("Ġworld".to_string(), 16);
    vocab.insert("<|endoftext|>".to_string(), 17);

    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
        ("hel".to_string(), "l".to_string()),
        ("hell".to_string(), "o".to_string()),
        ("Ġ".to_string(), "w".to_string()),
        ("Ġw".to_string(), "o".to_string()),
        ("Ġwo".to_string(), "r".to_string()),
        ("Ġwor".to_string(), "l".to_string()),
        ("Ġworl".to_string(), "d".to_string()),
    ];
    BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap()
}

#[test]
fn newline_boundary_table_marks_decoded_newlines() {
    let vocab = HashMap::from([
        ("a".to_string(), 0u32),
        ("Ċ".to_string(), 1),
        (".Ċ".to_string(), 2),
        (".".to_string(), 3),
        ("b".to_string(), 4),
    ]);
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
        .expect("synthetic byte-level vocabulary must load");

    let table = tokenizer.newline_boundary_table();
    assert_eq!(
        &table[..],
        &[false, true, true, false, false],
        "only ids whose decoded text contains a newline may be marked"
    );
    assert!(tokenizer.decode(&[1]).expect("decode").contains('\n'));
    assert_eq!(tokenizer.decode(&[2]).as_deref(), Some(".\n"));
    assert!(!tokenizer.decode(&[3]).expect("decode").contains('\n'));
}

#[test]
fn test_gpt4_regex_pretokenize_matches_hf_bytelevel_use_regex_true() {
    // Golden pieces from HF `tokenizers.pre_tokenizers.ByteLevel(use_regex=True)`
    // (the real, unconditional default for every GPT-2-format tokenizer HF
    // ships — see audit_tokenizer_parity.rs::gpt2_raw_vocab_bytelevel_fallback_parity
    // for the exact command). HF's actual ByteLevel(use_regex=True) regex is
    // the original GPT-2 pattern `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
    // ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`; `gpt4_regex_pretokenize` is a
    // Qwen/tiktoken-flavored superset (its prefix-letters branch can also
    // admit leading punctuation), not a literal reimplementation of that
    // pattern. The cases below are only claimed to match HF ByteLevel/GPT-2
    // behavior for the exact inputs tested, not as a proof the two regexes
    // are equivalent in general.
    assert_eq!(
        gpt4_regex_pretokenize("hello\nworld"),
        vec!["hello", "\n", "world"]
    );
    assert_eq!(gpt4_regex_pretokenize("a'st"), vec!["a", "'s", "t"]);
}

#[test]
fn test_gpt2_raw_vocab_defaults_to_gpt4_regex_pretokenize() {
    // Regression for #330: `from_vocab_and_merges` (what `from_files` and
    // `load_tokenizer`'s file-based-probing fallback use for a raw
    // GPT-2-format vocab.json + merges.txt with no tokenizer.json to read
    // pre_tokenizer metadata from) must default to `Gpt4Regex`
    // pretokenization, not the naive hand-rolled fallback that used to
    // exist — see `test_gpt4_regex_pretokenize_matches_hf_bytelevel_use_regex_true`
    // above for the HF-verified golden pieces this default now produces.
    //
    // This is a merge-boundary discriminator: a rule merging 's'+'t' into
    // "st" can only fire if both chars land in the SAME pretokenize piece
    // (BPE merges never span piece boundaries). Under the correct regex
    // split, "a'st" -> ["a", "'s", "t"]: 's' and 't' are in DIFFERENT
    // pieces ("'s" and "t"), so the merge cannot fire and every char
    // resolves to its own id. Under the buggy naive-fallback split,
    // "a'st" -> ["a", "'st"]: 's' and 't' are adjacent within the same
    // piece, so the merge DOES fire, producing a 3-id sequence instead of
    // 4. This mirrors the real gpt2 vocab divergence verified empirically
    // in audit_tokenizer_parity.rs (pre-fix: [64, 6, 301], HF: [64, 338, 83]).
    let mut vocab = HashMap::new();
    for (s, i) in [("a", 0u32), ("'", 1), ("s", 2), ("t", 3), ("st", 4)] {
        vocab.insert(s.to_string(), i);
    }
    let merges = vec![("s".to_string(), "t".to_string())];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();
    assert_eq!(tokenizer.tokenize_to_ids("a'st"), vec![0, 1, 2, 3]);
}

/// Vocab/merges for the cross-space-merge discriminator used by the
/// `validate_bpe_pretokenizer` fail-closed tests below (#330 finding 1).
/// Byte-level maps a literal space to `Ġ` (U+0120), so
/// `"a b"` byte-encodes to `"aĠb"`. Base single-byte tokens (`a`, `Ġ`,
/// `b`) plus staged merges `Ġ`+`b` -> `Ġb` -> (with `a`) `aĠb` let a
/// SINGLE merged id (2) only be reachable if "a" and " b" land in the
/// SAME pre-tokenize piece — i.e. no regex split at all. `Gpt4Regex`
/// always splits on word boundaries, so "a" and " b" land in different
/// pieces and the cross-space merge can never fire, giving `[0, 1]`
/// (the "a" and "Ġb" ids from separate pieces) instead. This is a
/// direct id-level proxy for the silent-wrong-ids failure mode when
/// explicit `use_regex:false` metadata falls
/// through to the `Gpt4Regex` default unchecked.
fn cross_space_merge_json(pre_tokenizer: &str) -> String {
    format!(
        r#"{{
                "model": {{
                    "type": "BPE",
                    "vocab": {{"a": 0, "Ġb": 1, "aĠb": 2, "Ġ": 3, "b": 4}},
                    "merges": ["Ġ b", "a Ġb"]
                }},
                "pre_tokenizer": {pre_tokenizer}
            }}"#
    )
}

#[test]
fn test_bytelevel_use_regex_false_rejected_without_supporting_split() {
    // #330 finding 1 (major/fail-open): a tokenizer.json
    // that explicitly declares `ByteLevel(use_regex:false)` with no
    // preceding regex `Split` must be REJECTED, not silently tokenized
    // with the `Gpt4Regex` default. HF's ByteLevel `use_regex=false`
    // means "no regex split at all" — falling back to `Gpt4Regex` would
    // silently apply a split the tokenizer author explicitly disabled,
    // producing wrong ids (see `cross_space_merge_json` doc comment).
    let json = cross_space_merge_json(r#"{"type": "ByteLevel", "use_regex": false}"#);
    let err = BpeTokenizer::from_tokenizer_json_str(&json)
        .expect_err("ByteLevel(use_regex:false) without a supporting Split must fail closed");
    let msg = err.to_string();
    assert!(
        msg.contains("unsupported BPE pre_tokenizer"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn test_unknown_pretokenizer_type_rejected() {
    // #330 finding 1: an unrecognized pre_tokenizer type
    // (Metaspace, Whitespace, BertPreTokenizer, ...) on a BPE model must
    // also fail closed rather than silently defaulting to `Gpt4Regex`.
    let json = cross_space_merge_json(r#"{"type": "Metaspace", "replacement": "_"}"#);
    let err = BpeTokenizer::from_tokenizer_json_str(&json)
        .expect_err("unknown pre_tokenizer type must fail closed");
    assert!(
        err.to_string().contains("unsupported BPE pre_tokenizer"),
        "unexpected error message: {err}"
    );
}

#[test]
fn test_bare_bytelevel_use_regex_true_or_absent_accepted() {
    // #330 finding 1: bare `ByteLevel` with `use_regex`
    // absent or `true` is the common gpt2/roberta Hub shape and must
    // keep working exactly as the #330 fix intended — HF's own default
    // for `use_regex` is `true`, which is what `Gpt4Regex` approximates.
    for pre_tokenizer in [
        r#"{"type": "ByteLevel"}"#,
        r#"{"type": "ByteLevel", "use_regex": true}"#,
    ] {
        let json = cross_space_merge_json(pre_tokenizer);
        let tokenizer = BpeTokenizer::from_tokenizer_json_str(&json)
            .unwrap_or_else(|e| panic!("bare ByteLevel must be accepted: {e}"));
        // Gpt4Regex splits "a b" into ["a", " b"]: the cross-space merge
        // cannot fire, so it round-trips to two ids, not the merged one.
        assert_eq!(tokenizer.tokenize_to_ids("a b"), vec![0, 1]);
    }
}

#[test]
fn test_sequence_split_then_bytelevel_use_regex_false_accepted() {
    // #330 finding 1: the real Qwen/Qwen3-Embedding
    // shape is `Sequence[Split{pattern.Regex}, ByteLevel{use_regex:false}]`
    // — a regex Split establishes the correct segmentation, and the
    // trailing `ByteLevel(use_regex:false)` just means "don't split
    // again". This must still be accepted and still dispatch to
    // `Gpt4Regex` (the only implemented mode), matching pre-fix #519
    // behavior for this specific shape.
    let pre_tokenizer = r#"{
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated"},
                {"type": "ByteLevel", "use_regex": false}
            ]
        }"#;
    let json = cross_space_merge_json(pre_tokenizer);
    let tokenizer = BpeTokenizer::from_tokenizer_json_str(&json)
        .unwrap_or_else(|e| panic!("Sequence[Split, ByteLevel(false)] must be accepted: {e}"));
    assert_eq!(tokenizer.tokenize_to_ids("a b"), vec![0, 1]);
}

#[test]
fn test_sequence_split_then_metaspace_rejected() {
    // #330 residual finding: before this fix,
    // `is_supported_bpe_pretokenizer` returned `true` as soon as
    // `has_regex_split` found ANY descendant regex `Split`, so
    // `Sequence[Split{pattern.Regex}, Metaspace]` was wrongly accepted
    // — the unsupported `Metaspace` sibling never got a chance to reject
    // the sequence. A `Split` earlier in the sequence does not make an
    // unrelated, unimplemented sibling honest to run through
    // `Gpt4Regex`: lattice would silently tokenize with GPT-2 byte
    // encoding while HF applies `Metaspace`, producing a completely
    // different token alphabet (wrong ids, not a subtle drift).
    let pre_tokenizer = r#"{
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated"},
                {"type": "Metaspace", "replacement": "_"}
            ]
        }"#;
    let json = cross_space_merge_json(pre_tokenizer);
    let err = BpeTokenizer::from_tokenizer_json_str(&json)
        .expect_err("Sequence[Split, Metaspace] must fail closed even though Split is regex-based");
    assert!(
        err.to_string().contains("unsupported BPE pre_tokenizer"),
        "unexpected error message: {err}"
    );
}

#[test]
fn test_sequence_split_then_whitespace_rejected() {
    // Same class as `test_sequence_split_then_metaspace_rejected` above,
    // with `Whitespace` (also named explicitly in the #330 finding 1
    // reject list) as the unsupported sibling instead of `Metaspace`.
    let pre_tokenizer = r#"{
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated"},
                {"type": "Whitespace"}
            ]
        }"#;
    let json = cross_space_merge_json(pre_tokenizer);
    let err = BpeTokenizer::from_tokenizer_json_str(&json).expect_err(
        "Sequence[Split, Whitespace] must fail closed even though Split is regex-based",
    );
    assert!(
        err.to_string().contains("unsupported BPE pre_tokenizer"),
        "unexpected error message: {err}"
    );
}

#[test]
fn test_bpe_merge_hello_world() {
    let tokenizer = synthetic_bpe();
    let ids = tokenizer.tokenize_to_ids("hello world");
    assert_eq!(ids, vec![11, 16]);
}

#[test]
fn test_empty_content_special_token_does_not_hang() {
    // Regression: a malformed tokenizer.json with a zero-length added-token
    // `content` injects "" into special_tokens. `"".starts_with(_)` is
    // always true, so match_special returns the same `pos` and
    // tokenize_to_ids_into never advances — an infinite loop with unbounded
    // output on the first tokenize() call. The constructor must drop
    // zero-length special tokens. Run in a worker thread so a re-introduced
    // hang fails fast via timeout instead of stalling the test binary.
    let mut vocab = HashMap::new();
    for (s, i) in [("a", 0u32), ("b", 1), ("c", 2)] {
        vocab.insert(s.to_string(), i);
    }
    let mut added = HashMap::new();
    added.insert(String::new(), 0u32);
    let tokenizer = BpeTokenizer::from_vocab_and_merges_with_config(
        vocab,
        Vec::new(),
        added,
        HashMap::new(),
        DEFAULT_BPE_CACHE_CAPACITY,
        DEFAULT_BPE_MAX_SEQ_LEN,
    )
    .expect("construct tokenizer with empty special token");

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(tokenizer.tokenize("abc").real_length);
    });
    match rx.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(len) => assert!(
            len < 1000,
            "empty special token produced runaway output ({len}); zero-length specials must be dropped"
        ),
        Err(_) => {
            panic!("tokenize() hung on a zero-length special token (infinite-loop regression)")
        }
    }
}

#[test]
fn test_bpe_special_token_passthrough() {
    let tokenizer = synthetic_bpe();
    let ids = tokenizer.tokenize_to_ids("hello<|endoftext|>world");
    assert_eq!(ids, vec![11, 17, 5, 3, 6, 2, 7]);
}

#[test]
fn test_bpe_truncation_preserves_eos() {
    let tokenizer = synthetic_bpe().with_add_eos().with_max_seq_len(2);
    let ids = tokenizer.tokenize_to_ids("hello world");
    assert_eq!(ids, vec![11, 17]);
}

#[test]
fn test_bpe_pre_truncation_len_exceeds_real_length_when_truncated() {
    // "hello world" -> [hello(11), Ġworld(16)] untruncated (2 tokens).
    // With max_seq_len=1, the tokenizer truncates to 1 token but must
    // still report the untruncated count via pre_truncation_len -- the
    // context-window admission guard in serve/embeddings.rs compares
    // against this field, not real_length, and this is the collision
    // case where the tokenizer's own cap coincides with the model limit.
    let tokenizer = synthetic_bpe().with_max_seq_len(1);
    let input = tokenizer.tokenize("hello world");
    assert_eq!(input.real_length, 1);
    assert_eq!(
        input.pre_truncation_len, 2,
        "pre_truncation_len must report the untruncated token count"
    );
    assert!(input.pre_truncation_len > input.real_length);
}

#[test]
fn test_bpe_tokenize_batch_pads_to_batch_max() {
    let tokenizer = synthetic_bpe();
    let batch = tokenizer.tokenize_batch(&["hello", "hello world"]);
    assert_eq!(batch[0].input_ids.len(), 2);
    assert_eq!(batch[1].input_ids.len(), 2);
    assert_eq!(batch[0].attention_mask, vec![1, 0]);
    assert_eq!(batch[1].attention_mask, vec![1, 1]);
}

#[test]
fn test_bpe_decode_roundtrip() {
    let tokenizer = synthetic_bpe();
    // Mirrors test_bpe_merge_hello_world: "hello world" → [11, 16].
    let ids = tokenizer.tokenize_to_ids("hello world");
    assert_eq!(ids, vec![11, 16]);
    // Regression: decode must reverse the byte-level encoding (the "Ġ"
    // prefix maps back to a space), not return an empty string — the
    // retired generic detokenize block discarded the text entirely.
    assert_eq!(tokenizer.decode(&ids), Some("hello world".to_string()));
    assert_eq!(tokenizer.decode(&[]), Some(String::new()));
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
fn max_token_byte_span_covers_special_and_added_token_content() {
    let json = r#"{
            "model":{"type":"BPE","vocab":{"a":0},"merges":[]},
            "added_tokens":[
                {"id":1,"content":"<|control|>","special":true},
                {"id":2,"content":"<|rendered extra|>","special":false}
            ]
        }"#;
    let tokenizer = BpeTokenizer::from_tokenizer_json_str(json).unwrap();
    assert_eq!(
        tokenizer.max_token_byte_span(),
        "<|rendered extra|>".len(),
        "the longest special or added token's content must set the span"
    );
    let plain = BpeTokenizer::from_tokenizer_json_str(
        r#"{"model":{"type":"BPE","vocab":{"a":0,"ab":1},"merges":["a b"]}}"#,
    )
    .unwrap();
    assert_eq!(plain.max_token_byte_span(), 2);
}

#[test]
fn decode_and_incremental_detokenize_render_non_ascii_added_tokens() {
    use crate::detokenize::IncrementalDetokenizer;

    let tokenizer = non_ascii_added_tokenizer();
    assert_eq!(
        tokenizer.token_bytes_for_id(100).as_deref(),
        Some("好".as_bytes())
    );
    assert_eq!(
        tokenizer.token_bytes_for_id(101).as_deref(),
        Some("café".as_bytes())
    );
    assert_eq!(tokenizer.decode(&[100, 101]), Some("好café".to_string()));

    let mut detok = IncrementalDetokenizer::new();
    let mut text = String::new();
    for id in [100, 101] {
        text.push_str(&detok.push(&tokenizer, id));
    }
    text.push_str(&detok.finish());
    assert_eq!(text, "好café");
}

#[test]
fn test_decode_renders_nonspecial_added_tokens() {
    // Regression (qwen3.6-27b think-tag bug): added tokens with special=false
    // (`</think>`, `<tool_call>`, FIM markers) live BEYOND the base vocab range,
    // so before the fix `token_for_id` returned None and they decoded to the
    // empty string — silently swallowed from the output stream even though the
    // model sampled them correctly. They must now render as their literal text.
    // special=true markers (im_end-style) must STILL be swallowed.
    let mut vocab = HashMap::new();
    for (s, i) in [("a", 0u32), ("b", 1), ("c", 2)] {
        vocab.insert(s.to_string(), i);
    }
    // rendered_added = the special=false subset (content -> id), ids past base max.
    let mut rendered = HashMap::new();
    rendered.insert(100u32, "</think>".to_string());
    rendered.insert(101u32, "<think>".to_string());
    rendered.insert(102u32, "<tool_call>".to_string());

    let tokenizer = BpeTokenizer::from_vocab_and_merges_with_config(
        vocab,
        Vec::new(),
        HashMap::new(),
        rendered,
        DEFAULT_BPE_CACHE_CAPACITY,
        DEFAULT_BPE_MAX_SEQ_LEN,
    )
    .expect("construct tokenizer with rendered added tokens");

    assert_eq!(tokenizer.special_token_id("</think>"), None);
    assert_eq!(tokenizer.token_ids_for_content("</think>"), vec![100]);

    // special=false added tokens render verbatim (byte-level decode is identity
    // for printable ASCII).
    assert_eq!(tokenizer.decode(&[101]), Some("<think>".to_string()));
    assert_eq!(tokenizer.decode(&[100]), Some("</think>".to_string()));
    assert_eq!(tokenizer.decode(&[102]), Some("<tool_call>".to_string()));
    // Mixed with base content tokens, in stream order.
    assert_eq!(
        tokenizer.decode(&[0, 1, 100, 2]),
        Some("ab</think>c".to_string())
    );
    // An id present in neither base vocab nor the rendered set (e.g. a
    // special=true marker) stays swallowed — token_for_id returns None.
    assert_eq!(tokenizer.decode(&[200]), Some(String::new()));

    // The incremental streaming detokenizer must agree byte-for-byte.
    use crate::detokenize::IncrementalDetokenizer;
    let mut detok = IncrementalDetokenizer::new();
    let mut out = String::new();
    for id in [0u32, 100, 1] {
        out.push_str(&detok.push(&tokenizer, id));
    }
    out.push_str(&detok.finish());
    assert_eq!(out, "a</think>b");
}

/// `token_ids_for_content` must resolve a marker from each of its three
/// tiers. The `vocab` and `added_render` tiers are the load-bearing ones:
/// `special_token_id` alone covers only the `special_tokens` tier, so a
/// marker declared as an ordinary vocab entry or as a `special: false`
/// added token resolves to `None` through that older path -- which is
/// exactly how an active reasoning budget became a silent no-op.
#[test]
fn token_ids_for_content_resolves_special_vocab_and_added_render_tiers() {
    let mut vocab = HashMap::new();
    vocab.insert("a".to_string(), 0u32);
    // Tier 2: a marker declared as an ordinary base-vocabulary entry.
    vocab.insert("</think>".to_string(), 7u32);

    let mut specials = HashMap::new();
    // Tier 1: the classic special-token table entry.
    specials.insert("<|im_end|>".to_string(), 20u32);

    let mut rendered = HashMap::new();
    // Tier 3: a `special: false` added token, absent from both maps above.
    rendered.insert(30u32, "<tool_call>".to_string());

    let tokenizer = BpeTokenizer::from_vocab_and_merges_with_config(
        vocab,
        Vec::new(),
        specials,
        rendered,
        DEFAULT_BPE_CACHE_CAPACITY,
        DEFAULT_BPE_MAX_SEQ_LEN,
    )
    .expect("construct three-tier tokenizer");

    assert_eq!(tokenizer.token_ids_for_content("<|im_end|>"), vec![20]);

    // Tiers 2 and 3 are invisible to `special_token_id`.
    assert_eq!(tokenizer.special_token_id("</think>"), None);
    assert_eq!(tokenizer.token_ids_for_content("</think>"), vec![7]);
    assert_eq!(tokenizer.special_token_id("<tool_call>"), None);
    assert_eq!(tokenizer.token_ids_for_content("<tool_call>"), vec![30]);

    // A spelling present in no tier resolves to None rather than to some
    // near-miss entry.
    assert!(tokenizer.token_ids_for_content("<think>").is_empty());
}

/// One spelling can carry DIFFERENT ids in two tiers, and the single-id
/// lookup necessarily discards all but the first.
///
/// This is not exotic: `vocab` and `rendered_added` are independent
/// constructor inputs with nothing reconciling them, and added-token ids
/// deliberately live beyond the base vocabulary range, so a marker declared
/// in both is guaranteed to have two ids rather than one. The tier test
/// above cannot catch it because every spelling there appears in exactly one
/// tier -- it would pass unchanged against a lookup with no notion of
/// ambiguity at all.
#[test]
fn token_ids_for_content_reports_every_id_sharing_one_spelling() {
    let mut vocab = HashMap::new();
    vocab.insert("a".to_string(), 0u32);
    vocab.insert("</think>".to_string(), 7u32);

    let mut rendered = HashMap::new();
    // The same rendered spelling, at the id an added token would really get.
    rendered.insert(100u32, "</think>".to_string());

    let tokenizer = BpeTokenizer::from_vocab_and_merges_with_config(
        vocab,
        Vec::new(),
        HashMap::new(),
        rendered,
        DEFAULT_BPE_CACHE_CAPACITY,
        DEFAULT_BPE_MAX_SEQ_LEN,
    )
    .expect("construct colliding-spelling tokenizer");

    assert_eq!(tokenizer.token_ids_for_content("</think>"), vec![7, 100]);

    // Tier precedence still decides which id comes first, so a caller that
    // took only the head would silently drop id 100 here. That discard is
    // the reason the resolver refuses this tokenizer rather than resolving
    // it, and the reason no single-id accessor survives on this type.
    assert_eq!(tokenizer.token_ids_for_content("</think>")[0], 7);

    // Unambiguous spellings return exactly one id, so the ambiguity signal
    // is a real discriminator and not something every lookup trips.
    assert_eq!(tokenizer.token_ids_for_content("a"), vec![0]);
    assert!(tokenizer.token_ids_for_content("<think>").is_empty());
}

#[test]
fn test_from_tokenizer_json_renders_real_qwen_think_tags() {
    // End-to-end regression gate for the qwen3.6 think-tag detok bug, driving
    // the REAL public loader (`from_tokenizer_json_str`) with the EXACT added-token
    // ids and special flags from the shipped qwen tokenizer.json (verified identical
    // on both qwen3.5-0.8b and qwen3.6-27b): `<think>`=248068, `</think>`=248069 are
    // special=false and live FAR above the base vocab max (248043). This exercises
    // the full wiring parse_rendered_added_tokens → added_render → append_token_bytes →
    // decode; a token-level parity gate (e2e-parity compares token IDS) is BLIND to
    // this class because the ids matched while the rendered text dropped the tag.
    let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1},
                "merges": []
            },
            "added_tokens": [
                {"id": 248045, "content": "<|im_start|>", "special": true},
                {"id": 248046, "content": "<|im_end|>", "special": true},
                {"id": 248058, "content": "<tool_call>", "special": false},
                {"id": 248068, "content": "<think>", "special": false},
                {"id": 248069, "content": "</think>", "special": false}
            ]
        }"#;
    let tokenizer =
        BpeTokenizer::from_tokenizer_json_str(json).expect("load tokenizer.json fixture");

    // special=false think/tool tags render verbatim (the bug: these were swallowed).
    assert_eq!(tokenizer.decode(&[248069]), Some("</think>".to_string()));
    assert_eq!(tokenizer.decode(&[248068]), Some("<think>".to_string()));
    assert_eq!(tokenizer.decode(&[248058]), Some("<tool_call>".to_string()));
    // special=true chat-control markers stay swallowed (HF skip_special_tokens=True).
    assert_eq!(tokenizer.decode(&[248046]), Some(String::new()));
    assert_eq!(tokenizer.decode(&[248045]), Some(String::new()));
    // A close tag mid-stream between base content tokens renders in order.
    assert_eq!(
        tokenizer.decode(&[0, 248069, 1]),
        Some("a</think>b".to_string())
    );
}

#[test]
fn test_bpe_duplicate_merge_keeps_first_rank() {
    // Regression (#259): a duplicate merge pair must keep its FIRST rank.
    // First-wins: a+b (rank 0) merges before b+c (rank 1) → ["ab","c"].
    // The old last-wins overwrite demoted a+b to rank 2 → ["a","bc"].
    let mut vocab = HashMap::new();
    for (s, i) in [("a", 0u32), ("b", 1), ("c", 2), ("ab", 3), ("bc", 4)] {
        vocab.insert(s.to_string(), i);
    }
    let merges = vec![
        ("a".to_string(), "b".to_string()),
        ("b".to_string(), "c".to_string()),
        ("a".to_string(), "b".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();
    assert_eq!(tokenizer.tokenize_to_ids("abc"), vec![3, 2]);
}

#[test]
fn test_bpe_partial_byte_fallback_rolls_back() {
    // Regression (#259): a merged token absent from vocab whose chars only
    // partially resolve must roll back and emit a single <unk>, not leave
    // the resolved id orphaned. "ab" merges to "ab" (not in vocab); 'a'
    // resolves but 'b' does not → must be [<unk>]=[1], not [id(a),<unk>].
    let mut vocab = HashMap::new();
    vocab.insert("a".to_string(), 0u32);
    vocab.insert("<unk>".to_string(), 1u32);
    let merges = vec![("a".to_string(), "b".to_string())];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();
    assert_eq!(tokenizer.tokenize_to_ids("ab"), vec![1]);
}

#[test]
fn test_non_boolean_special_flag_fails_closed_to_special() {
    // A `"special"` value that is present but not a JSON boolean (here, a
    // string) must fail closed to special rather than reading as `false`
    // via `unwrap_or` (#1722): the id must be masked exactly like an
    // explicit `special: true`, not rendered as literal text.
    let json = r#"{
            "model":{"type":"BPE","vocab":{"a":0},"merges":[]},
            "added_tokens":[{"id":50,"content":"<odd>","special":"true"}]
        }"#;
    let tokenizer = BpeTokenizer::from_tokenizer_json_str(json).unwrap();
    assert_eq!(tokenizer.token_bytes_for_id(50), None);
    assert_eq!(tokenizer.token_for_id(50), None);
    assert_eq!(tokenizer.decode(&[0, 50]), Some("a".to_string()));
}

#[test]
fn test_base_vocab_token_with_unmappable_char_masked_not_rejected() {
    // Regression (#1722): `append_byte_decoded_token_bytes`'s `filter_map`
    // silently drops any character outside the 256-entry GPT-2 byte-level
    // alphabet, so a base-vocab token containing one unmappable character
    // among otherwise-mappable ones used to decode to only the mappable
    // remainder -- which can collide with an unrelated token that spells
    // that same remainder outright. Construction must still succeed (a
    // hard rejection here broke every non-byte-level test/synthetic
    // vocabulary across the crate, e.g. a raw `" "` token id); instead the
    // whole token is masked to no bytes, exactly like a special id.
    //
    // id 0 "a" is fully mappable. id 1 "好a" mixes one unmappable
    // character ('好', U+597D, not one of the 256 GPT-2 byte-level
    // characters) with a mappable one. Under the old partial-decode bug,
    // id 1 would decode to the same single byte as id 0 ('a' -> 97) --
    // exactly the collision this fix closes.
    let mut vocab = HashMap::new();
    vocab.insert("a".to_string(), 0u32);
    vocab.insert("好a".to_string(), 1u32);
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
        .expect("a non-byte-level base-vocab token must not be refused at construction");

    assert_eq!(tokenizer.token_bytes_for_id(0), Some(vec![b'a']));
    assert_eq!(
        tokenizer.token_bytes_for_id(1),
        None,
        "a base-vocab token with any unmappable character must mask to no bytes"
    );

    let vocab_bytes = tokenizer.vocab_bytes(2).unwrap();
    assert!(
        vocab_bytes[1].is_empty(),
        "masked id must get an empty vocab_bytes entry, got {:?}",
        vocab_bytes[1]
    );
    assert_ne!(
        vocab_bytes[0], vocab_bytes[1],
        "the masked id must not collide with the mappable token spelling its surviving bytes"
    );

    assert_eq!(tokenizer.decode(&[1]), Some(String::new()));
    assert_eq!(tokenizer.decode(&[0, 1]), Some("a".to_string()));
}
