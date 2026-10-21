use super::*;
use crate::bpe::BpeTokenizer;
use std::collections::HashMap;

#[test]
fn request_buffers_reuse_utf8_delta() {
    let encoded = bytes_to_unicode();
    let chunks: &[&[u8]] = &[b"hi", &[0xE5, 0xA5], &[0xBD], &[0x80], b"ok", &[0xF0, 0x9F]];
    let vocab: HashMap<String, u32> = chunks
        .iter()
        .enumerate()
        .map(|(id, bytes)| {
            (
                bytes.iter().map(|&b| encoded[b as usize]).collect(),
                id as u32,
            )
        })
        .collect();
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, Vec::new()).unwrap();
    let mut detok = IncrementalDetokenizer::new();
    let mut delta = String::with_capacity(128);
    let capacity = delta.capacity();
    let pointer = delta.as_ptr();
    let mut text = String::new();
    for id in 0..chunks.len() as u32 {
        delta.clear();
        detok.push_into(&tokenizer, id, &mut delta);
        assert_eq!(delta.capacity(), capacity);
        assert_eq!(delta.as_ptr(), pointer);
        text.push_str(&delta);
    }
    text.push_str(&detok.finish());
    assert_eq!(
        text,
        decode_tokens(&tokenizer, &(0..chunks.len() as u32).collect::<Vec<_>>())
    );
    assert_eq!(text, "hi好\u{FFFD}ok\u{FFFD}");
}

#[test]
fn incremental_flush_reconstructs_split_codepoints() {
    // 好 = E5 A5 BD, 😀 = F0 9F 98 80 — split across simulated token boundaries
    // so several intermediate buffers end mid-codepoint.
    let chunks: &[&[u8]] = &[&[0xE5, 0xA5], &[0xBD], &[0xF0, 0x9F], &[0x98, 0x80], b"ok"];
    let mut d = IncrementalDetokenizer::new();
    let mut out = String::new();
    for c in chunks {
        d.pending.extend_from_slice(c);
        out.push_str(&d.flush_complete());
    }
    out.push_str(&d.finish());
    assert_eq!(out, "好😀ok");
    assert_eq!(d.retained_byte_len(), 0);
}

#[test]
fn incremental_flush_truncated_mid_codepoint_matches_lossy() {
    // Generation truncated after only the first 2 of 好's 3 bytes: finish()
    // flushes the incomplete tail lossily, matching decode_tokens' behavior.
    let mut d = IncrementalDetokenizer::new();
    let mut out = String::new();
    d.pending.extend_from_slice(&[b'h', b'i', 0xE5, 0xA5]);
    out.push_str(&d.flush_complete());
    out.push_str(&d.finish());
    assert_eq!(out, String::from_utf8_lossy(&[b'h', b'i', 0xE5, 0xA5]));
    assert_eq!(d.retained_byte_len(), 0);
}

#[test]
fn incremental_flush_invalid_byte_does_not_stall_stream() {
    // A lone continuation byte (0x80) is a malformed sequence no later token
    // can complete. flush_complete must emit U+FFFD for it immediately and
    // keep going, not buffer it until finish() (which would stall the stream).
    let mut d = IncrementalDetokenizer::new();
    d.pending.extend_from_slice(&[0x80]);
    let first = d.flush_complete();
    assert_eq!(first, "\u{FFFD}", "invalid byte must flush immediately");
    d.pending.extend_from_slice(b"A");
    let second = d.flush_complete();
    assert_eq!(second, "A", "valid byte after invalid one must flush");
    // Concatenated stream equals from_utf8_lossy over the whole byte buffer.
    assert_eq!(
        format!("{first}{second}"),
        String::from_utf8_lossy(&[0x80, b'A'])
    );
    assert_eq!(d.retained_byte_len(), 0);
}

#[test]
fn incremental_flush_invalid_between_valid_matches_lossy() {
    // Valid · invalid · valid in one buffer: one U+FFFD for the bad run, both
    // valid runs verbatim — byte-for-byte equal to from_utf8_lossy.
    let raw = [b'h', b'i', 0xFF, b'y', b'o'];
    let mut d = IncrementalDetokenizer::new();
    d.pending.extend_from_slice(&raw);
    let out = d.flush_complete();
    assert_eq!(out, String::from_utf8_lossy(&raw));
    assert_eq!(d.retained_byte_len(), 0);
}

#[test]
fn incremental_flush_ascii_is_exact_per_chunk() {
    // Pure-ASCII path: every chunk is whole codepoints, so each flush emits it all.
    let mut d = IncrementalDetokenizer::new();
    let mut out = String::new();
    for c in [b"He".as_slice(), b"llo", b"!"] {
        d.pending.extend_from_slice(c);
        out.push_str(&d.flush_complete());
    }
    out.push_str(&d.finish());
    assert_eq!(out, "Hello!");
    assert_eq!(d.retained_byte_len(), 0);
}

/// Regression for issue #324: retained state must stay bounded by a fixed
/// capacity no matter how many tokens are streamed. Before the fix,
/// `IncrementalDetokenizer` retained the whole generation's decoded bytes
/// (`bytes` + `flushed` cursor), so this loop would grow retained capacity
/// linearly with the token count. If the bound is removed (e.g. reverting to
/// unconditional whole-buffer retention, or dropping the drain/compaction
/// step), `retained_byte_capacity()` grows past `DETOK_RETAINED_BYTE_CAPACITY`
/// and this test fails.
#[test]
fn incremental_detokenizer_retention_bounded_for_long_ascii_stream() {
    let byte_encoder = bytes_to_unicode();
    let mut vocab = HashMap::new();
    vocab.insert(byte_encoder[b'a' as usize].to_string(), 0u32);
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
        .expect("synthetic BPE tokenizer builds");

    let mut detok = IncrementalDetokenizer::new();
    let mut out = String::new();
    for _ in 0..8_192 {
        out.push_str(&detok.push(&tokenizer, 0));
        assert_eq!(detok.retained_byte_len(), 0);
        assert!(
            detok.retained_byte_capacity() <= DETOK_RETAINED_BYTE_CAPACITY,
            "capacity must remain bounded, got {}",
            detok.retained_byte_capacity()
        );
    }
    out.push_str(&detok.finish());
    assert_eq!(out.len(), 8_192);
    assert_eq!(detok.retained_byte_len(), 0);
    assert!(detok.retained_byte_capacity() <= DETOK_RETAINED_BYTE_CAPACITY);
}

/// Companion regression for issue #324: the test above only ever pushes 1-byte ASCII tokens, so `pending`'s
/// `Vec` never grows past its initial `with_capacity(DETOK_RETAINED_BYTE_CAPACITY)`
/// allocation — deleting the `compact_pending_capacity()` call (or its call
/// site) would still pass it, since there is never anything to compact.
/// This test feeds a *single* token whose decoded bytes (12, all valid
/// ASCII) exceed `DETOK_RETAINED_BYTE_CAPACITY` (8) in one `push`, forcing
/// `pending`'s allocation to grow past the bound before `flush_complete`
/// drains it back to empty — then asserts the retained *capacity* still
/// compacts back down to the bound afterward, actually exercising
/// `compact_pending_capacity`'s shrink path.
#[test]
fn incremental_detokenizer_retention_compacts_after_oversized_single_push() {
    let byte_encoder = bytes_to_unicode();
    // One BPE token whose decoded bytes are 12 copies of 'a' — longer than
    // DETOK_RETAINED_BYTE_CAPACITY (8), all valid UTF-8, so every byte
    // flushes in the same `push` call that appended it.
    let oversized_token: String = std::iter::repeat_n(byte_encoder[b'a' as usize], 12).collect();
    let mut vocab = HashMap::new();
    vocab.insert(oversized_token, 0u32);
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
        .expect("synthetic BPE tokenizer builds");

    let mut detok = IncrementalDetokenizer::new();
    let delta = detok.push(&tokenizer, 0);

    assert_eq!(delta, "a".repeat(12));
    assert_eq!(detok.retained_byte_len(), 0);
    assert!(
        detok.retained_byte_capacity() <= DETOK_RETAINED_BYTE_CAPACITY,
        "a single push whose decoded bytes (12) exceed DETOK_RETAINED_BYTE_CAPACITY \
             (8) must still leave the retained allocation compacted back down, got \
             capacity {}",
        detok.retained_byte_capacity()
    );
}

/// UTF-8-boundary property: a CJK/emoji/combining-mark/ZWJ-sequence token
/// stream, fed one BPE-token-sized byte chunk at a time (deliberately
/// splitting multibyte scalars and a multi-codepoint grapheme cluster across
/// token boundaries), must decode byte-identical to `decode_tokens`, the
/// accumulate-everything reference path. Retained bytes must never exceed 3
/// (the longest possible incomplete UTF-8 tail) while streaming. If an
/// over-eager drop clears `pending` on an incomplete tail (`error_len() ==
/// None`) instead of retaining it, or emits U+FFFD for a merely-incomplete
/// sequence, the streamed output diverges from the reference and this test
/// fails.
#[test]
fn incremental_detokenizer_utf8_parity_matches_full_decode_for_split_stream() {
    let chunks: &[&[u8]] = &[
        b"H",
        &[0xE5],
        &[0xA5, 0xBD],
        b" ",
        &[0xF0, 0x9F],
        &[0x98],
        &[0x80],
        b" ",
        b"e",
        &[0xCC],
        &[0x81],
        b" ",
        &[0xF0, 0x9F, 0x91],
        &[0xA9, 0xE2],
        &[0x80, 0x8D],
        &[0xF0, 0x9F, 0x92],
        &[0xBB],
    ];

    let byte_encoder = bytes_to_unicode();
    let mut vocab = HashMap::new();
    let mut ids = Vec::with_capacity(chunks.len());
    let mut next_id = 0u32;
    for chunk in chunks {
        let token_str: String = chunk.iter().map(|&b| byte_encoder[b as usize]).collect();
        // Interning: an already-seen chunk string reuses its existing id
        // instead of colliding as a duplicate vocab key.
        let id = *vocab.entry(token_str).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        ids.push(id);
    }
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
        .expect("synthetic BPE tokenizer builds");

    let reference = decode_tokens(&tokenizer, &ids);
    assert_eq!(
        reference,
        "H\u{597D} \u{1F600} e\u{301} \u{1F469}\u{200D}\u{1F4BB}"
    );

    let mut detok = IncrementalDetokenizer::new();
    let mut streamed = String::new();
    for &id in &ids {
        streamed.push_str(&detok.push(&tokenizer, id));
        assert!(detok.retained_byte_len() <= 3);
    }
    streamed.push_str(&detok.finish());

    assert_eq!(streamed, reference);
    assert_eq!(detok.retained_byte_len(), 0);
    assert!(detok.retained_byte_capacity() <= DETOK_RETAINED_BYTE_CAPACITY);
}
