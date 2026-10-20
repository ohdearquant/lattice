//! Qwen3.5 byte-decoder construction, token-byte appending, token decoding, incremental detokenization, and byte-to-Unicode mapping.
use crate::bpe::BpeTokenizer;
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
///
/// **Lossy by contract, and silently so.** Byte-level BPE can emit any byte sequence, so
/// the assembled bytes are converted with `String::from_utf8_lossy`: anything not
/// well-formed UTF-8 becomes `U+FFFD`, and the return type carries no signal that any
/// substitution happened.
///
/// That matters under grammar-constrained decoding, because the grammar validates BYTES.
/// A GBNF grammar using `.` or `[^...]` can accept a non-UTF-8 sequence, and the string
/// returned here is then not the sequence the grammar accepted, so it may not re-validate
/// against that same grammar. JSON-schema-constrained output is unaffected: that compiler
/// models well-formed UTF-8 explicitly, so no substitution can occur. The full statement
/// of the seam is in the `grammar::pda` module documentation.
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
pub struct IncrementalDetokenizer {
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
        let mut out = String::new();
        self.flush_complete_into(&mut out);
        out
    }

    /// Append a token's complete UTF-8 prefix to a caller-owned delta buffer.
    pub(crate) fn push_into(&mut self, tokenizer: &BpeTokenizer, id: u32, out: &mut String) {
        append_token_bytes(tokenizer, id, &self.byte_decoder, &mut self.pending);
        self.flush_complete_into(out);
    }

    #[cfg(test)]
    fn flush_complete(&mut self) -> String {
        let mut out = String::new();
        self.flush_complete_into(&mut out);
        out
    }

    fn flush_complete_into(&mut self, out: &mut String) {
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
#[expect(
    clippy::expect_used,
    reason = "the byte-to-unicode table is built from a fixed list of valid scalar values"
)]
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
mod tests;
