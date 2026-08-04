//! Qwen3.5 byte-decoder construction, token-byte appending, token decoding, incremental detokenization, and byte-to-Unicode mapping.
use crate::tokenizer::bpe::BpeTokenizer;
use std::collections::HashMap;

/// Build the GPT-2 byte-level reverse map (unicode placeholder char → original byte).
pub(crate) fn build_byte_decoder() -> HashMap<char, u8> {
    let byte_encoder = bytes_to_unicode();
    let mut byte_decoder: HashMap<char, u8> = HashMap::new();
    for (byte_val, &ch) in byte_encoder.iter().enumerate() {
        byte_decoder.insert(ch, byte_val as u8);
    }
    byte_decoder
}

/// Append the raw decoded bytes for a single token id onto `out`.
///
/// Byte-level BPE decoding is context-free per token, so concatenating the
/// per-token byte strings equals decoding the whole id sequence at once.
fn append_token_bytes(
    tokenizer: &BpeTokenizer,
    id: u32,
    byte_decoder: &HashMap<char, u8>,
    out: &mut Vec<u8>,
) {
    tokenizer.append_token_bytes(id, byte_decoder, out);
}

/// Decode token IDs back to text using the BPE tokenizer's reverse lookup.
pub(crate) fn decode_tokens(tokenizer: &BpeTokenizer, ids: &[u32]) -> String {
    let byte_decoder = build_byte_decoder();
    let mut bytes = Vec::new();
    for &id in ids {
        append_token_bytes(tokenizer, id, &byte_decoder, &mut bytes);
    }
    String::from_utf8_lossy(&bytes).to_string()
}

/// Upper bound on the raw bytes an [`IncrementalDetokenizer`] retains between
/// `push` calls. A UTF-8 scalar is at most 4 bytes, so a handful of bytes is
/// always enough to hold the longest possible incomplete trailing codepoint;
/// this bound exists purely to cap allocator churn, not to limit correctness.
const DETOK_RETAINED_BYTE_CAPACITY: usize = 8;

/// Incremental, UTF-8-boundary-safe detokenizer for streaming generation.
///
/// Byte-level BPE frequently splits a single multibyte codepoint (CJK, emoji,
/// accented Latin) across two or more tokens. Decoding the running token list
/// with `from_utf8_lossy` after every token would substitute `U+FFFD` for the
/// incomplete trailing bytes and stream that replacement char to the caller,
/// who cannot retract it. This type instead buffers only the undecided tail —
/// the bytes of a UTF-8 scalar still being assembled across token boundaries —
/// and emits every complete UTF-8 prefix as soon as it is known, so retained
/// state stays bounded regardless of how long the generation runs (issue
/// #324). The full generated text is *not* retained here: callers that need
/// the whole-generation string (e.g. `GenerateOutput.text`) accumulate the
/// `push`/`finish` deltas themselves. The concatenation of every `push` delta
/// plus the final `finish` equals `decode_tokens` over the same ids.
pub(crate) struct IncrementalDetokenizer {
    byte_decoder: HashMap<char, u8>,
    /// Bytes appended since the last `push`, not yet proven to be either a
    /// complete UTF-8 prefix (already emitted) or an unrepairable invalid
    /// sequence (already replaced). Only an in-progress incomplete codepoint
    /// tail should ever accumulate here between calls.
    pending: Vec<u8>,
}

impl IncrementalDetokenizer {
    pub(crate) fn new() -> Self {
        Self {
            byte_decoder: build_byte_decoder(),
            pending: Vec::with_capacity(DETOK_RETAINED_BYTE_CAPACITY),
        }
    }

    /// Feed one token id; return the complete-UTF-8 text it reveals (which may be
    /// empty when the token only contributed continuation bytes of a codepoint
    /// still being assembled).
    pub(crate) fn push(&mut self, tokenizer: &BpeTokenizer, id: u32) -> String {
        append_token_bytes(tokenizer, id, &self.byte_decoder, &mut self.pending);
        self.flush_complete()
    }

    fn flush_complete(&mut self) -> String {
        let mut out = String::new();
        let mut consumed = 0usize;

        loop {
            match std::str::from_utf8(&self.pending[consumed..]) {
                // Entire remaining buffer is valid UTF-8: emit it all.
                Ok(s) => {
                    out.push_str(s);
                    consumed = self.pending.len();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        // [consumed .. consumed + valid] is guaranteed valid by
                        // valid_up_to, so from_utf8_lossy substitutes nothing.
                        let end = consumed + valid;
                        out.push_str(
                            String::from_utf8_lossy(&self.pending[consumed..end]).as_ref(),
                        );
                        consumed = end;
                    }
                    match e.error_len() {
                        // None: the error is an incomplete trailing codepoint,
                        // not a malformed one. A later token may complete it, so
                        // fail closed: keep every remaining byte buffered and
                        // emit no replacement char.
                        None => break,
                        // Some(len): a genuinely invalid sequence of `len` bytes
                        // that no future token can repair. Emit one U+FFFD (matching
                        // from_utf8_lossy's maximal-subpart substitution), skip past
                        // it, and continue — without this the stream would stall on
                        // the bad byte until finish().
                        Some(len) => {
                            out.push('\u{FFFD}');
                            consumed += len;
                        }
                    }
                }
            }
        }

        if consumed > 0 {
            self.pending.drain(..consumed);
        }
        self.compact_pending_capacity();
        out
    }

    /// Flush any trailing incomplete bytes lossily. Call once after the final
    /// token so the concatenated stream matches `decode_tokens` exactly even when
    /// a generation is truncated mid-codepoint.
    pub(crate) fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let tail = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        self.compact_pending_capacity();
        tail
    }

    /// Shrink `pending`'s allocation back to the small retained bound once its
    /// length has fallen back under it, so a rare long invalid-byte run does not
    /// permanently inflate this detokenizer's footprint for the rest of the
    /// generation (see issue #324).
    fn compact_pending_capacity(&mut self) {
        if self.pending.capacity() > DETOK_RETAINED_BYTE_CAPACITY
            && self.pending.len() <= DETOK_RETAINED_BYTE_CAPACITY
        {
            let mut compact = Vec::with_capacity(DETOK_RETAINED_BYTE_CAPACITY);
            compact.extend_from_slice(&self.pending);
            self.pending = compact;
        }
    }

    #[cfg(test)]
    fn retained_byte_len(&self) -> usize {
        self.pending.len()
    }

    #[cfg(test)]
    fn retained_byte_capacity(&self) -> usize {
        self.pending.capacity()
    }
}

/// GPT-2 byte-to-unicode mapping (same as in bpe.rs).
pub fn bytes_to_unicode() -> Vec<char> {
    let mut bs = Vec::new();
    bs.extend(33u16..=126);
    bs.extend(161u16..=172);
    bs.extend(174u16..=255);

    let mut cs = bs.clone();
    let mut n = 0u16;
    for b in 0u16..=255u16 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut table = vec!['\0'; 256];
    for (b, c) in bs.into_iter().zip(cs) {
        table[b as usize] =
            char::from_u32(c as u32).expect("invariant: byte-to-unicode codepoint is valid");
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::bpe::BpeTokenizer;
    use std::collections::HashMap;

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
        let oversized_token: String =
            std::iter::repeat_n(byte_encoder[b'a' as usize], 12).collect();
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
}
