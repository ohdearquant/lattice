use crate::tokenizer::bpe::BpeTokenizer;
use std::collections::HashMap;

/// Decode token IDs back to text using the BPE tokenizer's reverse lookup.
pub(crate) fn decode_tokens(tokenizer: &BpeTokenizer, ids: &[u32]) -> String {
    let byte_encoder = bytes_to_unicode();
    let mut byte_decoder: HashMap<char, u8> = HashMap::new();
    for (byte_val, &ch) in byte_encoder.iter().enumerate() {
        byte_decoder.insert(ch, byte_val as u8);
    }

    let mut bytes = Vec::new();
    for &id in ids {
        if let Some(token_str) = tokenizer.token_for_id(id) {
            for ch in token_str.chars() {
                if let Some(&b) = byte_decoder.get(&ch) {
                    bytes.push(b);
                }
                // Skip characters not in byte_decoder (e.g., special tokens)
            }
        }
    }

    String::from_utf8_lossy(&bytes).to_string()
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
