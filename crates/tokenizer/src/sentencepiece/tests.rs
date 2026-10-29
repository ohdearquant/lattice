use super::*;

/// Every (end, id) a reference scan over all pieces finds, last id winning for a repeated piece.
fn reference_matches(entries: &[(&str, u32)], bytes: &[u8], start: usize) -> Vec<(usize, u32)> {
    let mut by_end: Vec<(usize, u32)> = Vec::new();
    for &(piece, id) in entries {
        let p = piece.as_bytes();
        if p.is_empty() || !bytes[start..].starts_with(p) {
            continue;
        }
        let end = start + p.len();
        match by_end.iter_mut().find(|(e, _)| *e == end) {
            Some(slot) => slot.1 = id,
            None => by_end.push((end, id)),
        }
    }
    by_end.sort_unstable();
    by_end
}

#[test]
fn byte_trie_matches_reference_scan() {
    let pieces = [
        "",
        "a",
        "ab",
        "abc",
        "abd",
        "b",
        "ab",
        "ba",
        "\u{2581}",
        "\u{2581}a",
        "\u{2581}ab",
        "é",
        "éa",
        "日",
        "日本",
        "日本語",
        "本",
        "語",
        "zz",
        "z",
        "a",
        "\u{ff}",
    ];
    let entries: Vec<(&str, u32)> = pieces
        .iter()
        .enumerate()
        .map(|(i, p)| (*p, i as u32 * 3 + 1))
        .collect();
    let trie = ByteTrie::build(&entries);
    let texts = [
        "abcabdab",
        "\u{2581}ab\u{2581}a",
        "日本語本日",
        "éaé",
        "zzz",
        "x",
        "",
        "ba\u{ff}",
    ];
    let mut checked = 0;
    for text in texts {
        let bytes = text.as_bytes();
        for start in 0..bytes.len() {
            assert_eq!(
                trie.matches(bytes, start),
                reference_matches(&entries, bytes, start),
                "text {text:?} start {start}"
            );
            checked += 1;
        }
    }
    assert!(checked > 40);
    // A repeated piece resolves to its last id: "ab" is entries 2 and 6, "a" is 1 and 20.
    assert_eq!(
        trie.matches(b"ab", 0),
        vec![(1, 20 * 3 + 1), (2, 6 * 3 + 1)]
    );
    assert!(ByteTrie::build(&[]).matches(b"abc", 0).is_empty());
}

fn synthetic_tokenizer() -> SentencePieceTokenizer {
    SentencePieceTokenizer::from_pieces_for_test(
        vec![
            SentencePieceEntry {
                piece: "<unk>".to_string(),
                score: -10.0,
                kind: SentencePieceType::Unknown,
            },
            SentencePieceEntry {
                piece: "▁hello".to_string(),
                score: 5.0,
                kind: SentencePieceType::Normal,
            },
            SentencePieceEntry {
                piece: "▁world".to_string(),
                score: 4.0,
                kind: SentencePieceType::Normal,
            },
            SentencePieceEntry {
                piece: "wor".to_string(),
                score: 1.0,
                kind: SentencePieceType::Normal,
            },
            SentencePieceEntry {
                piece: "ld".to_string(),
                score: 1.0,
                kind: SentencePieceType::Normal,
            },
        ],
        0,
    )
    .unwrap()
}

#[test]
fn test_out_of_range_special_id_rejected_at_construction() {
    // A malformed tokenizer.json / .model can declare unk_id (or bos/eos/pad) past
    // the end of the vocab even when the value fits in u32. Without the construction
    // guard, from_parsed_model would build a tokenizer that emits that out-of-range id
    // from viterbi_encode's unmatched-char fallback, which then indexes the model's
    // embedding table out of bounds. Construction must reject it. Here vocab has 3
    // pieces (ids 0..3) and unk_id = 5.
    let pieces = vec![
        SentencePieceEntry {
            piece: "<unk>".to_string(),
            score: -10.0,
            kind: SentencePieceType::Unknown,
        },
        SentencePieceEntry {
            piece: "▁a".to_string(),
            score: 1.0,
            kind: SentencePieceType::Normal,
        },
        SentencePieceEntry {
            piece: "▁b".to_string(),
            score: 1.0,
            kind: SentencePieceType::Normal,
        },
    ];
    let result = SentencePieceTokenizer::from_pieces_for_test(pieces, 5);
    assert!(
        result.is_err(),
        "an out-of-range unk_id must be rejected at construction, not emitted at tokenize time"
    );
}

#[test]
fn test_out_of_range_unk_id_rejected_from_json() {
    // A malformed file declaring unk_id past the vocab end must be rejected at
    // construction, mirroring the .model path covered by from_pieces_for_test above.
    // Vocab has 2 pieces (ids 0,1) and unk_id = 4.
    let json = r#"{"model":{"type":"Unigram","unk_id":4,"vocab":[["<unk>",0.0],["▁a",-1.0]]}}"#;
    let result = SentencePieceTokenizer::from_tokenizer_json_str(json);
    assert!(
        result.is_err(),
        "an out-of-range unk_id from tokenizer.json must be rejected at construction"
    );
}

#[test]
fn test_overflowing_unk_id_rejected_from_json() {
    let json =
        r#"{"model":{"type":"Unigram","unk_id":4294967296,"vocab":[["<unk>",0.0],["▁a",-1.0]]}}"#;
    let result = SentencePieceTokenizer::from_tokenizer_json_str(json);
    assert!(
        result.is_err(),
        "a SentencePiece special id outside the u32 domain must be rejected"
    );
}

#[test]
fn test_overflowing_special_id_rejected_from_protobuf() {
    fn push_varint(mut value: u64, out: &mut Vec<u8>) {
        while value >= 0x80 {
            out.push((value as u8 & 0x7f) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }

    let mut trainer_spec = Vec::new();
    push_varint(40 << 3, &mut trainer_spec);
    push_varint(u64::from(u32::MAX) + 1, &mut trainer_spec);
    let mut model = ParsedSentencePieceModel::default();
    let result = parse_trainer_spec(&trainer_spec, &mut model);
    assert!(
        result.is_err(),
        "a protobuf SentencePiece special id outside the u32 domain must be rejected"
    );
}

#[test]
fn test_json_added_token_matches_before_normalization() {
    let json = r#"{
            "added_tokens":[{"id":2,"content":"<extra>","special":false}],
            "model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["▁a",-1.0]]}
        }"#;
    let tokenizer = SentencePieceTokenizer::from_tokenizer_json_str(json).unwrap();
    let input = tokenizer.tokenize("<extra>");
    assert_eq!(&input.input_ids[..input.real_length], &[2]);
}

#[test]
fn test_normalize_adds_metaspace() {
    let tokenizer = synthetic_tokenizer();
    assert_eq!(tokenizer.normalize("hello world"), "▁hello▁world");
}

#[test]
fn test_viterbi_prefers_whole_piece() {
    let tokenizer = synthetic_tokenizer();
    let ids = tokenizer.tokenize_to_ids("hello world");
    assert_eq!(ids, vec![1, 2]);
}

#[test]
fn test_sentencepiece_pre_truncation_len_exceeds_real_length_when_truncated() {
    // "hello world" -> [▁hello(1), ▁world(2)] untruncated (2 tokens).
    // With max_seq_len=1, the tokenizer truncates to 1 token but must
    // still report the untruncated count via pre_truncation_len -- the
    // context-window admission guard in serve/embeddings.rs compares
    // against this field, not real_length, and this is the collision
    // case where the tokenizer's own cap coincides with the model limit.
    let tokenizer = synthetic_tokenizer().with_max_seq_len(1);
    let input = tokenizer.tokenize("hello world");
    assert_eq!(input.real_length, 1);
    assert_eq!(
        input.pre_truncation_len, 2,
        "pre_truncation_len must report the untruncated token count"
    );
    assert!(input.pre_truncation_len > input.real_length);
}

#[test]
fn test_sentencepiece_batch_padding() {
    let tokenizer = synthetic_tokenizer();
    let batch = tokenizer.tokenize_batch(&["hello", "hello world"]);
    assert_eq!(batch[0].attention_mask, vec![1, 0]);
    assert_eq!(batch[1].attention_mask, vec![1, 1]);
}

#[test]
fn test_parse_byte_piece() {
    assert_eq!(parse_byte_piece("<0x41>"), Some(0x41));
    assert_eq!(parse_byte_piece("not-a-byte"), None);
}

#[test]
fn test_json_loader_routes_byte_pieces_to_fallback() {
    // A SentencePiece Unigram model loaded from tokenizer.json must recover
    // byte-fallback typing for `<0xNN>` pieces, matching the `.model`
    // protobuf loader.  Regression test for #255: typing was hard-coded to
    // Normal, so byte pieces landed in the trie as literal text and
    // byte_fallback_ids stayed empty — breaking OOV byte fallback in
    // viterbi_encode.
    let json = r#"{"model":{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["▁hello",-1.0],["<0x41>",-5.0],["<0xE2>",-5.0]]}}"#;
    let tok = SentencePieceTokenizer::from_tokenizer_json_str(json).unwrap();
    // `<0xNN>` pieces route to byte_fallback_ids (not the trie).
    assert_eq!(tok.inner.byte_fallback_ids[0x41], Some(2));
    assert_eq!(tok.inner.byte_fallback_ids[0xE2], Some(3));
    // Normal pieces are unaffected by the inference.
    assert_eq!(tok.tokenize_to_ids("hello"), vec![1]);
}

#[test]
fn test_json_loader_respects_explicit_byte_fallback_false() {
    // When the model explicitly declares no byte fallback, a literal
    // `<0xNN>`-shaped piece must stay Normal (trie-resolvable), not be
    // misrouted to byte_fallback_ids by shape inference (#255 edge case).
    let json = r#"{"model":{"type":"Unigram","unk_id":0,"byte_fallback":false,"vocab":[["<unk>",0.0],["<0x41>",-5.0]]}}"#;
    let tok = SentencePieceTokenizer::from_tokenizer_json_str(json).unwrap();
    assert_eq!(tok.inner.byte_fallback_ids[0x41], None);
}

// HF Metaspace semantics: trailing whitespace in input must not produce a
// trailing space-piece token.  These unit tests cover the normalize() fix.

#[test]
fn test_normalize_strips_trailing_metaspace() {
    let tokenizer = synthetic_tokenizer();
    // Trailing whitespace → no trailing ▁ in normalized form.
    assert_eq!(tokenizer.normalize("hello world   "), "▁hello▁world");
    // Single trailing space.
    assert_eq!(tokenizer.normalize("hello "), "▁hello");
}

#[test]
fn test_normalize_leading_whitespace_no_duplicate_prefix() {
    let tokenizer = synthetic_tokenizer();
    // Leading whitespace: dummy_prefix already sets prev_was_space=true, so
    // the leading space is collapsed; no extra ▁ before the first word.
    assert_eq!(tokenizer.normalize("   hello world"), "▁hello▁world");
}

#[test]
fn test_normalize_trailing_and_leading_whitespace() {
    let tokenizer = synthetic_tokenizer();
    // Both leading and trailing whitespace — strip trailing, collapse leading.
    assert_eq!(tokenizer.normalize("   hello world   "), "▁hello▁world");
}
