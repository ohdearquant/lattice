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
    if let Some(token_str) = tokenizer.token_for_id(id) {
        for ch in token_str.chars() {
            if let Some(&b) = byte_decoder.get(&ch) {
                out.push(b);
            }
            // Skip characters not in byte_decoder (e.g., special tokens)
        }
    }
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

/// Incremental, UTF-8-boundary-safe detokenizer for streaming generation.
///
/// Byte-level BPE frequently splits a single multibyte codepoint (CJK, emoji,
/// accented Latin) across two or more tokens. Decoding the running token list
/// with `from_utf8_lossy` after every token would substitute `U+FFFD` for the
/// incomplete trailing bytes and stream that replacement char to the caller,
/// who cannot retract it. This type instead buffers raw bytes and emits only
/// the longest *complete* UTF-8 prefix after each token, holding incomplete
/// trailing bytes until a later token completes the codepoint. The concatenation
/// of every `push` delta plus the final `finish` equals `decode_tokens` over the
/// same ids.
pub(crate) struct IncrementalDetokenizer {
    byte_decoder: HashMap<char, u8>,
    bytes: Vec<u8>,
    flushed: usize,
}

impl IncrementalDetokenizer {
    pub(crate) fn new() -> Self {
        Self {
            byte_decoder: build_byte_decoder(),
            bytes: Vec::new(),
            flushed: 0,
        }
    }

    /// Feed one token id; return the complete-UTF-8 text it reveals (which may be
    /// empty when the token only contributed continuation bytes of a codepoint
    /// still being assembled).
    pub(crate) fn push(&mut self, tokenizer: &BpeTokenizer, id: u32) -> String {
        append_token_bytes(tokenizer, id, &self.byte_decoder, &mut self.bytes);
        self.flush_complete()
    }

    fn flush_complete(&mut self) -> String {
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.bytes[self.flushed..]) {
                // Entire remaining buffer is valid UTF-8: emit it all.
                Ok(s) => {
                    out.push_str(s);
                    self.flushed = self.bytes.len();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        // [flushed .. flushed + valid] is guaranteed valid by
                        // valid_up_to, so from_utf8_lossy substitutes nothing.
                        out.push_str(
                            String::from_utf8_lossy(
                                &self.bytes[self.flushed..self.flushed + valid],
                            )
                            .as_ref(),
                        );
                        self.flushed += valid;
                    }
                    match e.error_len() {
                        // None: the error is an incomplete trailing codepoint,
                        // not a malformed one. A later token may complete it, so
                        // keep the tail buffered and emit no replacement char.
                        None => return out,
                        // Some(len): a genuinely invalid sequence of `len` bytes
                        // that no future token can repair. Emit one U+FFFD (matching
                        // from_utf8_lossy's maximal-subpart substitution), skip past
                        // it, and continue — without this the stream would stall on
                        // the bad byte until finish().
                        Some(len) => {
                            out.push('\u{FFFD}');
                            self.flushed += len;
                        }
                    }
                }
            }
        }
    }

    /// Flush any trailing incomplete bytes lossily. Call once after the final
    /// token so the concatenated stream matches `decode_tokens` exactly even when
    /// a generation is truncated mid-codepoint.
    pub(crate) fn finish(&mut self) -> String {
        if self.flushed < self.bytes.len() {
            let tail = String::from_utf8_lossy(&self.bytes[self.flushed..]).into_owned();
            self.flushed = self.bytes.len();
            tail
        } else {
            String::new()
        }
    }

    /// The full decoded text so far (equivalent to `decode_tokens` over every fed id).
    pub(crate) fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).to_string()
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

    #[test]
    fn incremental_flush_reconstructs_split_codepoints() {
        // 好 = E5 A5 BD, 😀 = F0 9F 98 80 — split across simulated token boundaries
        // so several intermediate buffers end mid-codepoint.
        let chunks: &[&[u8]] = &[&[0xE5, 0xA5], &[0xBD], &[0xF0, 0x9F], &[0x98, 0x80], b"ok"];
        let mut d = IncrementalDetokenizer::new();
        let mut out = String::new();
        for c in chunks {
            d.bytes.extend_from_slice(c);
            out.push_str(&d.flush_complete());
        }
        out.push_str(&d.finish());
        assert_eq!(out, "好😀ok");
        assert_eq!(out, d.text());
    }

    #[test]
    fn incremental_flush_truncated_mid_codepoint_matches_lossy() {
        // Generation truncated after only the first 2 of 好's 3 bytes: finish()
        // flushes the incomplete tail lossily, matching decode_tokens' behavior.
        let mut d = IncrementalDetokenizer::new();
        let mut out = String::new();
        d.bytes.extend_from_slice(&[b'h', b'i', 0xE5, 0xA5]);
        out.push_str(&d.flush_complete());
        out.push_str(&d.finish());
        assert_eq!(out, String::from_utf8_lossy(&[b'h', b'i', 0xE5, 0xA5]));
        assert_eq!(out, d.text());
    }

    #[test]
    fn incremental_flush_invalid_byte_does_not_stall_stream() {
        // A lone continuation byte (0x80) is a malformed sequence no later token
        // can complete. flush_complete must emit U+FFFD for it immediately and
        // keep going, not buffer it until finish() (which would stall the stream).
        let mut d = IncrementalDetokenizer::new();
        d.bytes.extend_from_slice(&[0x80]);
        let first = d.flush_complete();
        assert_eq!(first, "\u{FFFD}", "invalid byte must flush immediately");
        d.bytes.extend_from_slice(b"A");
        let second = d.flush_complete();
        assert_eq!(second, "A", "valid byte after invalid one must flush");
        // Concatenated stream equals from_utf8_lossy over the whole byte buffer.
        assert_eq!(
            format!("{first}{second}"),
            String::from_utf8_lossy(&[0x80, b'A'])
        );
    }

    #[test]
    fn incremental_flush_invalid_between_valid_matches_lossy() {
        // Valid · invalid · valid in one buffer: one U+FFFD for the bad run, both
        // valid runs verbatim — byte-for-byte equal to from_utf8_lossy.
        let raw = [b'h', b'i', 0xFF, b'y', b'o'];
        let mut d = IncrementalDetokenizer::new();
        d.bytes.extend_from_slice(&raw);
        let out = d.flush_complete();
        assert_eq!(out, String::from_utf8_lossy(&raw));
        assert_eq!(out, d.text());
    }

    #[test]
    fn incremental_flush_ascii_is_exact_per_chunk() {
        // Pure-ASCII path: every chunk is whole codepoints, so each flush emits it all.
        let mut d = IncrementalDetokenizer::new();
        let mut out = String::new();
        for c in [b"He".as_slice(), b"llo", b"!"] {
            d.bytes.extend_from_slice(c);
            out.push_str(&d.flush_complete());
        }
        out.push_str(&d.finish());
        assert_eq!(out, "Hello!");
    }
}
