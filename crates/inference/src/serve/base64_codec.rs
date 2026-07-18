//! Standard-alphabet (RFC 4648 §4) base64 encode/decode, no external crate: the HTTP servers
//! already clamp the input string to a small fixed size before [`decode_standard`] ever runs
//! (`MAX_CONTENT_PART_BYTES` in `lattice_serve.rs`), so an image-sized hand-rolled codec is
//! simpler than pulling in a dependency for this one call site (PR #1021 review round 5: this
//! codec used to be reimplemented three times independently -- `lattice_serve.rs`'s production
//! decoder, its own `#[cfg(test)]` encoder, and `tests/vision_serve_e2e_test.rs`'s separate
//! encoder -- with no compiler-enforced connection between them).

/// Decodes a standard-alphabet base64 string. Fails closed on every malformed input
/// (non-multiple-of-4 length, invalid alphabet character, misplaced `=` padding) rather than
/// panicking or silently truncating.
pub fn decode_standard(s: &str) -> Result<Vec<u8>, String> {
    fn value(b: u8) -> Result<u32, String> {
        match b {
            b'A'..=b'Z' => Ok((b - b'A') as u32),
            b'a'..=b'z' => Ok((b - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((b - b'0' + 52) as u32),
            b'+' => Ok(62),
            b'/' => Ok(63),
            other => Err(format!("invalid base64 character {:?}", other as char)),
        }
    }
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return Err("empty base64 payload".to_string());
    }
    if !bytes.len().is_multiple_of(4) {
        return Err("base64 length must be a multiple of 4".to_string());
    }
    let n_chunks = bytes.len() / 4;
    let mut out = Vec::with_capacity(n_chunks * 3);
    for (chunk_index, chunk) in bytes.chunks_exact(4).enumerate() {
        let is_last = chunk_index + 1 == n_chunks;
        let pad = chunk.iter().rev().take_while(|&&b| b == b'=').count();
        if pad > 0 && !is_last {
            return Err("'=' padding is only allowed in the final base64 block".to_string());
        }
        if pad > 2 {
            return Err("too much '=' padding in base64 data".to_string());
        }
        if chunk[..4 - pad].contains(&b'=') {
            return Err("'=' padding character in the middle of a base64 block".to_string());
        }
        let mut v = [0u32; 4];
        for (j, &b) in chunk.iter().enumerate() {
            v[j] = if b == b'=' { 0 } else { value(b)? };
        }
        let n = (v[0] << 18) | (v[1] << 12) | (v[2] << 6) | v[3];
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

/// Encodes bytes as standard-alphabet base64 -- the reverse of [`decode_standard`]. Test-only
/// (gated on `cfg(any(test, feature = "test-utils"))`, same convention as
/// `model::qwen35::test_support`): production code never needs to *produce* base64, only decode
/// it from an incoming `data:` URI, but both `lattice_serve.rs`'s own test module and
/// `tests/vision_serve_e2e_test.rs` (a separate compilation unit reached only through the
/// `test-utils` feature) need one to build encoded fixtures.
#[cfg(any(test, feature = "test-utils"))]
pub fn encode_standard(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(n >> 18) as usize & 0x3f] as char);
        out.push(ALPHABET[(n >> 12) as usize & 0x3f] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 0x3f] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 0x3f] as char
        } else {
            '='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_matches_known_vector() {
        // RFC 4648 §10 test vector.
        assert_eq!(decode_standard("Zm9vYmFy").unwrap(), b"foobar");
        assert_eq!(encode_standard(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn roundtrip_arbitrary_bytes() {
        let raw: Vec<u8> = (0..=255u8).collect();
        let encoded = encode_standard(&raw);
        assert_eq!(decode_standard(&encoded).unwrap(), raw);
    }

    #[test]
    fn rejects_malformed_input() {
        assert!(decode_standard("").is_err());
        assert!(decode_standard("abcde").is_err());
        assert!(decode_standard("not-valid-base64!!").is_err());
    }
}
