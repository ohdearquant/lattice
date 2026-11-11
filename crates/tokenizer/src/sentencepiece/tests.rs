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
fn sentencepiece_adjacent_unknown_characters_form_one_unknown() {
    let json = r#"{"model":{"type":"Unigram","unk_id":0,"byte_fallback":false,"vocab":[["<unk>",0.0],["▁a",-1.0],["b",-1.0]]}}"#;
    let tokenizer = SentencePieceTokenizer::from_tokenizer_json_str(json).unwrap();
    assert!(tokenizer.inner.dummy_prefix);
    assert!(!tokenizer.inner.add_bos);
    assert!(!tokenizer.inner.add_eos);
    assert_eq!(tokenizer.normalize("axx"), "▁axx");
    assert_eq!(tokenizer.tokenize_to_ids("axx"), vec![1, 0]);
}

#[test]
fn sentencepiece_known_piece_separates_unknown_runs() {
    let json = r#"{"model":{"type":"Unigram","unk_id":0,"byte_fallback":false,"vocab":[["<unk>",0.0],["▁a",-1.0],["b",-1.0]]}}"#;
    let tokenizer = SentencePieceTokenizer::from_tokenizer_json_str(json).unwrap();
    assert_eq!(tokenizer.tokenize_to_ids("axxbxx"), vec![1, 0, 2, 0]);
}

#[test]
fn sentencepiece_complete_byte_fallback_preserves_each_byte() {
    let json = r#"{"model":{"type":"Unigram","unk_id":0,"byte_fallback":true,"vocab":[["<unk>",0.0],["▁a",-1.0],["<0x78>",-5.0],["<0x79>",-5.0]]}}"#;
    let tokenizer = SentencePieceTokenizer::from_tokenizer_json_str(json).unwrap();
    assert_eq!(tokenizer.tokenize_to_ids("axy"), vec![1, 2, 3]);
}

#[test]
fn sentencepiece_incomplete_byte_fallback_uses_one_unknown_for_the_run() {
    let json = r#"{"model":{"type":"Unigram","unk_id":0,"byte_fallback":true,"vocab":[["<unk>",0.0],["▁a",-1.0],["<0x78>",-5.0]]}}"#;
    let tokenizer = SentencePieceTokenizer::from_tokenizer_json_str(json).unwrap();
    assert_eq!(tokenizer.tokenize_to_ids("axy"), vec![1, 0]);
}

#[test]
fn sentencepiece_selects_unigram_path_before_applying_byte_fallback() {
    let json = r#"{"model":{"type":"Unigram","unk_id":0,"byte_fallback":true,"vocab":[["<unk>",0.0],["▁a",0.0],["xy",-12.0],["<0x78>",0.0],["<0x79>",0.0]]}}"#;
    let tokenizer = SentencePieceTokenizer::from_tokenizer_json_str(json).unwrap();
    assert_eq!(tokenizer.tokenize_to_ids("axy"), vec![1, 2]);
}

#[test]
fn sentencepiece_byte_fallback_preserves_multibyte_utf8_order() {
    let json = r#"{"model":{"type":"Unigram","unk_id":0,"byte_fallback":true,"vocab":[["<unk>",0.0],["▁a",-1.0],["<0xC3>",-5.0],["<0xA9>",-5.0]]}}"#;
    let tokenizer = SentencePieceTokenizer::from_tokenizer_json_str(json).unwrap();
    assert_eq!(tokenizer.tokenize_to_ids("aé"), vec![1, 2, 3]);
}

#[test]
fn sentencepiece_fused_span_is_resolved_as_a_vocabulary_piece_first() {
    let json = r#"{"model":{"type":"Unigram","unk_id":0,"byte_fallback":false,"vocab":[["<unk>",30.0],["xy",30.0]]}}"#;
    let tokenizer = SentencePieceTokenizer::from_tokenizer_json_str(json).unwrap();
    assert_eq!(tokenizer.viterbi_encode("xy"), vec![1]);
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

// A minimal SentencePiece double-array charsmap mapping ASCII A to B.
fn encode_test_base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let value = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        out.push(ALPHABET[((value >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((value >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[((value >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(value & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn test_charsmap(units: &[u32], pool: &[u8]) -> String {
    let mut bytes = ((units.len() * 4) as u32).to_le_bytes().to_vec();
    for unit in units {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    bytes.extend_from_slice(pool);
    encode_test_base64(&bytes)
}

fn test_a_to_b_units(size: usize) -> Vec<u32> {
    let mut units = vec![0; size];
    units[0] = 1 << 10;
    units[64] = u32::from(b'A') | (1 << 8) | (1 << 10);
    units
}

fn test_precompiled_a_to_b() -> String {
    test_charsmap(&test_a_to_b_units(256), b"B\0")
}

fn precompiled_fixture(normalizer: &str) -> String {
    format!(
        r#"{{"model":{{"type":"Unigram","unk_id":0,"byte_fallback":false,"vocab":[["<unk>",0],["▁",-3],["A",-3],["B",-3],["▁A",-1],["▁B",-1]]}},"normalizer":{normalizer}}}"#
    )
}

#[test]
fn sentencepiece_precompiled_map_applies_synthetic_mapping() {
    let map = test_precompiled_a_to_b();
    let normalizer = format!(r#"{{"type":"Precompiled","precompiled_charsmap":"{map}"}}"#);
    let tokenizer =
        SentencePieceTokenizer::from_tokenizer_json_str(&precompiled_fixture(&normalizer)).unwrap();
    assert_eq!(tokenizer.tokenize_to_ids("A"), vec![5]);
    assert_eq!(tokenizer.tokenize_to_ids("B"), vec![5]);
}

#[test]
fn sentencepiece_without_precompiled_map_keeps_existing_tokens() {
    let tokenizer =
        SentencePieceTokenizer::from_tokenizer_json_str(&precompiled_fixture("null")).unwrap();
    assert_eq!(tokenizer.tokenize_to_ids("A"), vec![4]);
    assert_eq!(tokenizer.tokenize_to_ids("B"), vec![5]);
}

#[test]
fn sentencepiece_rejects_malformed_precompiled_maps() {
    for raw in ["not-base64!", "AQ=="] {
        let normalizer = format!(r#"{{"type":"Precompiled","precompiled_charsmap":"{raw}"}}"#);
        let error =
            SentencePieceTokenizer::from_tokenizer_json_str(&precompiled_fixture(&normalizer))
                .unwrap_err();
        assert!(error.to_string().contains("precompiled"), "{error}");
    }
}

#[test]
#[ignore = "requires LATTICE_E5_TOKENIZER_JSON pointing to multilingual-e5-small tokenizer.json"]
fn sentencepiece_e5_precompiled_matches_hf_reference_ids() {
    let path = std::env::var("LATTICE_E5_TOKENIZER_JSON").expect("set E5 tokenizer path");
    let tokenizer =
        SentencePieceTokenizer::from_tokenizer_json(std::path::Path::new(&path)).unwrap();
    for (text, expected) in [
        ("A", vec![0, 62, 2]),
        ("Ａ", vec![0, 62, 2]),
        ("ＡＢＣ hello", vec![0, 47457, 33600, 31, 2]),
        ("ｶﾀｶﾅ", vec![0, 6, 10045, 10793, 10045, 17456, 2]),
        ("①", vec![0, 106, 2]),
        ("ﬁ", vec![0, 809, 2]),
    ] {
        assert_eq!(tokenizer.tokenize_to_ids(text), expected, "{text:?}");
    }
}

#[test]
fn sentencepiece_precompiled_shortest_prefix_replaces_whole_short_grapheme() {
    let map = test_precompiled_a_to_b();
    let normalizer = format!(r#"{{"type":"Precompiled","precompiled_charsmap":"{map}"}}"#);
    let tokenizer =
        SentencePieceTokenizer::from_tokenizer_json_str(&precompiled_fixture(&normalizer)).unwrap();
    assert_eq!(tokenizer.normalize("A\u{301}"), "▁B");
}

#[test]
fn sentencepiece_precompiled_long_grapheme_falls_back_to_scalars() {
    let map = test_precompiled_a_to_b();
    let normalizer = format!(r#"{{"type":"Precompiled","precompiled_charsmap":"{map}"}}"#);
    let tokenizer =
        SentencePieceTokenizer::from_tokenizer_json_str(&precompiled_fixture(&normalizer)).unwrap();
    assert_eq!(
        tokenizer.normalize("A\u{301}\u{301}\u{301}"),
        "▁B\u{301}\u{301}\u{301}"
    );
}

fn assert_invalid_precompiled_map(encoded: &str, reason: &str) {
    let normalizer = format!(r#"{{"type":"Precompiled","precompiled_charsmap":"{encoded}"}}"#);
    let error = SentencePieceTokenizer::from_tokenizer_json_str(&precompiled_fixture(&normalizer))
        .unwrap_err();
    assert!(error.to_string().contains(reason), "{error}");
}

#[test]
fn sentencepiece_precompiled_rejects_unaligned_trie() {
    assert_invalid_precompiled_map(&encode_test_base64(&[3, 0, 0, 0, 0, 0, 0]), "alignment");
}

#[test]
fn sentencepiece_precompiled_rejects_oversized_trie() {
    assert_invalid_precompiled_map(
        &encode_test_base64(&[4, 4, 0, 0, 0, 0, 0, 0]),
        "trie length",
    );
}

#[test]
fn sentencepiece_precompiled_rejects_traversal_out_of_bounds() {
    assert_invalid_precompiled_map(&test_charsmap(&test_a_to_b_units(66), b"B\0"), "traversal");
}

#[test]
fn sentencepiece_precompiled_rejects_pool_offset_out_of_bounds() {
    let mut units = test_a_to_b_units(256);
    units[65] = 10;
    assert_invalid_precompiled_map(&test_charsmap(&units, b"B\0"), "pool offset");
}

#[test]
fn sentencepiece_precompiled_rejects_unterminated_pool_entry() {
    assert_invalid_precompiled_map(&test_charsmap(&test_a_to_b_units(256), b"B"), "terminator");
}

#[test]
fn sentencepiece_precompiled_rejects_invalid_pool_utf8() {
    assert_invalid_precompiled_map(&test_charsmap(&test_a_to_b_units(256), b"\xff\0"), "UTF-8");
}

#[test]
fn sentencepiece_precompiled_applies_sequence_literal_replace() {
    let map = test_precompiled_a_to_b();
    let normalizer = format!(
        r#"{{"type":"Sequence","normalizers":[{{"type":"Precompiled","precompiled_charsmap":"{map}"}},{{"type":"Replace","pattern":{{"String":"B"}},"content":"C"}}]}}"#
    );
    let tokenizer =
        SentencePieceTokenizer::from_tokenizer_json_str(&precompiled_fixture(&normalizer)).unwrap();
    assert_eq!(tokenizer.normalize("A"), "▁C");
}

#[test]
fn sentencepiece_precompiled_applies_e5_regex_replace() {
    let map = test_precompiled_a_to_b();
    let normalizer = format!(
        r#"{{"type":"Sequence","normalizers":[{{"type":"Precompiled","precompiled_charsmap":"{map}"}},{{"type":"Replace","pattern":{{"Regex":" {{2,}}"}},"content":" "}}]}}"#
    );
    let tokenizer =
        SentencePieceTokenizer::from_tokenizer_json_str(&precompiled_fixture(&normalizer)).unwrap();
    assert_eq!(tokenizer.normalize("A  B"), "▁B▁B");
}

#[test]
fn sentencepiece_precompiled_rejects_unsupported_regex_lookaround() {
    let map = test_precompiled_a_to_b();
    let normalizer = format!(
        r#"{{"type":"Sequence","normalizers":[{{"type":"Precompiled","precompiled_charsmap":"{map}"}},{{"type":"Replace","pattern":{{"Regex":"(?=A)"}},"content":"X"}}]}}"#
    );
    let error = SentencePieceTokenizer::from_tokenizer_json_str(&precompiled_fixture(&normalizer))
        .unwrap_err();
    assert!(error.to_string().contains("(?=A)"), "{error}");
}

#[test]
fn sentencepiece_precompiled_replacement_content_is_literal() {
    let map = test_precompiled_a_to_b();
    let normalizer = format!(
        r#"{{"type":"Sequence","normalizers":[{{"type":"Precompiled","precompiled_charsmap":"{map}"}},{{"type":"Replace","pattern":{{"Regex":"B"}},"content":"$1"}}]}}"#
    );
    let tokenizer =
        SentencePieceTokenizer::from_tokenizer_json_str(&precompiled_fixture(&normalizer)).unwrap();
    assert_eq!(tokenizer.normalize("A"), "▁$1");
}

#[test]
fn sentencepiece_precompiled_rejects_unknown_sequence_step() {
    let map = test_precompiled_a_to_b();
    let normalizer = format!(
        r#"{{"type":"Sequence","normalizers":[{{"type":"Precompiled","precompiled_charsmap":"{map}"}},{{"type":"UnknownNormalizer"}}]}}"#
    );
    let error = SentencePieceTokenizer::from_tokenizer_json_str(&precompiled_fixture(&normalizer))
        .unwrap_err();
    assert!(error.to_string().contains("UnknownNormalizer"), "{error}");
}

#[test]
fn sentencepiece_without_precompiled_step_keeps_legacy_normalization() {
    let normalizer = r#"{"type":"Sequence","normalizers":[{"type":"UnknownNormalizer"}]}"#;
    let tokenizer =
        SentencePieceTokenizer::from_tokenizer_json_str(&precompiled_fixture(normalizer)).unwrap();
    assert_eq!(tokenizer.normalize("A"), "▁A");
}

#[test]
fn sentencepiece_precompiled_identity_records_exact_map_and_survives_rebuild() {
    use crate::common::TokenizerPolicyValue as V;
    let map = test_precompiled_a_to_b();
    let normalizer = format!(r#"{{"type":"Precompiled","precompiled_charsmap":"{map}"}}"#);
    let tokenizer =
        SentencePieceTokenizer::from_tokenizer_json_str(&precompiled_fixture(&normalizer)).unwrap();
    for candidate in [tokenizer.clone(), tokenizer.with_max_seq_len(16)] {
        let settings = candidate.embedding_settings().unwrap();
        assert!(settings.contains(&(
            "normalizer.kind".into(),
            V::String("precompiled-sequence-v1".into())
        )));
        assert!(settings.contains(&(
            "normalizer.0.charsmap_sha256".into(),
            V::String("f1dc868811145e0dbf22ac7d7a2c181343b5b0fa89c2d5d80c5530421ea4cd0a".into())
        )));
        assert!(settings.contains(&(
            "program".into(),
            V::String("sentencepiece-single-v2".into())
        )));
    }
    let changed_map = test_charsmap(&test_a_to_b_units(256), b"C\0");
    let normalizer = format!(r#"{{"type":"Precompiled","precompiled_charsmap":"{changed_map}"}}"#);
    let changed =
        SentencePieceTokenizer::from_tokenizer_json_str(&precompiled_fixture(&normalizer)).unwrap();
    assert_ne!(tokenizer.embedding_settings(), changed.embedding_settings());
}
