//! Shared tokenizer types, utilities, and auto-detection.
//!
//! This module contains the model-facing `TokenizedInput` representation,
//! the object-safe `Tokenizer` trait used by `BertModel`, a tiny JSON parser
//! used for dependency-free `tokenizer.json` loading, a thread-safe LRU cache,
//! and tokenizer auto-detection helpers.

use crate::error::InferenceError;
use crate::tokenizer::WordPieceTokenizer;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::sentencepiece::SentencePieceTokenizer;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::hash::Hash;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// **Stable**: tokenized representation passed into model forward calls; field
/// layout is stable once `BertModel::encode` is stable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedInput {
    /// Token IDs: tokenizer-specific special tokens plus the payload tokens,
    /// padded to the requested sequence length.
    pub input_ids: Vec<u32>,
    /// Attention mask: 1 for real tokens, 0 for padding.
    pub attention_mask: Vec<u32>,
    /// Token type IDs: all zeros for single-sequence encoding.
    pub token_type_ids: Vec<u32>,
    /// Number of real tokens before padding.
    pub real_length: usize,
}

/// **Stable**: object-safe tokenizer trait; concrete impls (`WordPieceTokenizer`,
/// `BpeTokenizer`, `SentencePieceTokenizer`) are boxed behind this.
pub trait Tokenizer: Send + Sync {
    /// **Stable**: tokenize a single text, padded to the configured maximum.
    fn tokenize(&self, text: &str) -> TokenizedInput;

    /// **Stable**: tokenize a batch, padding to batch-local max.
    fn tokenize_batch(&self, texts: &[&str]) -> Vec<TokenizedInput>;

    /// **Stable**: vocabulary size, used for config validation.
    fn vocab_size(&self) -> usize;

    /// **Stable**: configured maximum sequence length.
    fn max_seq_len(&self) -> usize;

    /// Whether this tokenizer supports BERT-style query/document pair encoding.
    ///
    /// Returns `false` by default; only `WordPieceTokenizer` overrides to `true`.
    fn supports_pair_tokenization(&self) -> bool {
        false
    }

    /// Encode a query/document pair as `[CLS] query [SEP] doc [SEP]` with token type IDs.
    ///
    /// `token_type_ids` are 0 for the query segment and 1 for the document segment.
    ///
    /// # Panics
    ///
    /// The default implementation panics.  Callers must check
    /// [`supports_pair_tokenization`](Self::supports_pair_tokenization) before
    /// calling this method.  Use [`try_tokenize_pair`](Self::try_tokenize_pair)
    /// to avoid the panic.
    fn tokenize_pair(&self, query: &str, document: &str) -> TokenizedInput {
        self.try_tokenize_pair(query, document)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Non-panicking variant of [`tokenize_pair`](Self::tokenize_pair).
    ///
    /// Returns `Err(InferenceError::UnsupportedOperation)` when the tokenizer
    /// does not support pair encoding.  Implementors that support pair
    /// encoding should override both this method and `tokenize_pair`.
    fn try_tokenize_pair(
        &self,
        query: &str,
        document: &str,
    ) -> Result<TokenizedInput, InferenceError> {
        let _ = (query, document);
        Err(InferenceError::Tokenizer(
            "tokenize_pair is not implemented for this tokenizer".into(),
        ))
    }

    /// **Stable**: decode token IDs back to text.
    ///
    /// Returns `None` for tokenizers that do not support detokenization —
    /// encoder-only embedding tokenizers (WordPiece/SentencePiece for
    /// BERT/BGE) have no generative decode use. Generative tokenizers
    /// (`BpeTokenizer`) override this with a real implementation.
    fn decode(&self, ids: &[u32]) -> Option<String> {
        let _ = ids;
        None
    }
}

/// Pad an ID sequence and explicit token type IDs to a fixed target length.
pub(crate) fn pad_ids_with_token_types(
    mut ids: Vec<u32>,
    mut token_type_ids: Vec<u32>,
    pad_to: usize,
    pad_id: u32,
) -> TokenizedInput {
    let real_length = ids.len();
    assert_eq!(
        token_type_ids.len(),
        real_length,
        "token_type_ids length must match input_ids before padding"
    );
    let mut attention_mask = vec![1u32; real_length];
    ids.resize(pad_to, pad_id);
    attention_mask.resize(pad_to, 0);
    token_type_ids.resize(pad_to, 0);
    TokenizedInput {
        input_ids: ids,
        attention_mask,
        token_type_ids,
        real_length,
    }
}

/// Pad an ID sequence to a fixed target length.
pub(crate) fn pad_ids(mut ids: Vec<u32>, pad_to: usize, pad_id: u32) -> TokenizedInput {
    let real_length = ids.len();
    let mut attention_mask = vec![1u32; real_length];
    ids.resize(pad_to, pad_id);
    attention_mask.resize(pad_to, 0);
    let token_type_ids = vec![0u32; pad_to];

    TokenizedInput {
        input_ids: ids,
        attention_mask,
        token_type_ids,
        real_length,
    }
}

pub(crate) fn push_eos_preserving_limit(ids: &mut Vec<u32>, eos_id: u32, max_seq_len: usize) {
    if max_seq_len == 0 {
        ids.clear();
        return;
    }

    if ids.len() >= max_seq_len {
        ids.truncate(max_seq_len - 1);
    }
    ids.push(eos_id);
}

/// Best-effort model max-length inference from `config.json` / `tokenizer_config.json`.
pub(crate) fn infer_model_max_seq_len(model_dir: &Path, default_value: usize) -> usize {
    let candidates = [
        model_dir.join("tokenizer_config.json"),
        model_dir.join("config.json"),
    ];

    for path in candidates {
        if !path.exists() {
            continue;
        }

        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };

        let Ok(json) = parse_json(&text) else {
            continue;
        };

        let keys = [
            &["model_max_length"][..],
            &["max_position_embeddings"][..],
            &["n_positions"][..],
            &["max_seq_len"][..],
            &["truncation", "max_length"][..],
        ];

        for path in keys {
            if let Some(value) = json_path(&json, path).and_then(JsonValue::as_u64) {
                if value > 0 {
                    // Cap at 2048 for embedding workloads. Models like Qwen3 advertise
                    // 32K positions but practical embedding inference uses ≤2K tokens.
                    // The GPU activation buffers are pre-allocated for this limit.
                    let capped = (value as usize).min(2048);
                    return capped;
                }
            }
        }
    }

    default_value
}

/// **Stable**: auto-detect and load a tokenizer from a model directory.
///
/// Priority order:
/// 1. `tokenizer.json`
/// 2. `vocab.json` + `merges.txt` or `vocab.txt` + `merges.txt`
/// 3. `vocab.txt`
/// 4. `tokenizer.model`
pub fn load_tokenizer(model_dir: &Path) -> Result<Box<dyn Tokenizer>, InferenceError> {
    let tokenizer_json = model_dir.join("tokenizer.json");
    if tokenizer_json.exists() {
        let text = fs::read_to_string(&tokenizer_json).map_err(|e| {
            InferenceError::Tokenizer(format!("failed to read {}: {e}", tokenizer_json.display()))
        })?;
        let root = parse_json(&text)?;
        let model_type = json_path(&root, &["model", "type"])
            .and_then(JsonValue::as_str)
            .unwrap_or("");
        let max_seq_len = infer_model_max_seq_len(model_dir, 512);

        match model_type {
            "WordPiece" => {
                return Ok(Box::new(
                    WordPieceTokenizer::from_tokenizer_json_str(&text)?
                        .with_max_seq_len(max_seq_len),
                ));
            }
            "BPE" => {
                return Ok(Box::new(
                    BpeTokenizer::from_tokenizer_json_str(&text)?.with_max_seq_len(max_seq_len),
                ));
            }
            "Unigram" | "SentencePieceUnigram" => {
                return Ok(Box::new(
                    SentencePieceTokenizer::from_tokenizer_json_str(&text)?
                        .with_max_seq_len(max_seq_len),
                ));
            }
            other if !other.is_empty() => {
                return Err(InferenceError::UnsupportedModel(format!(
                    "unsupported tokenizer model type {other:?} in {}",
                    tokenizer_json.display()
                )));
            }
            _ => {
                // Fall through to file-based probing when tokenizer.json is missing a model type.
            }
        }
    }

    let max_seq_len = infer_model_max_seq_len(model_dir, 512);

    let vocab_json = model_dir.join("vocab.json");
    let vocab_txt = model_dir.join("vocab.txt");
    let merges_txt = model_dir.join("merges.txt");
    let tokenizer_model = model_dir.join("tokenizer.model");

    if merges_txt.exists() && vocab_json.exists() {
        return Ok(Box::new(
            BpeTokenizer::from_files(&vocab_json, &merges_txt)?.with_max_seq_len(max_seq_len),
        ));
    }

    if merges_txt.exists() && vocab_txt.exists() {
        return Ok(Box::new(
            BpeTokenizer::from_vocab_txt_and_merges(&vocab_txt, &merges_txt)?
                .with_max_seq_len(max_seq_len),
        ));
    }

    if vocab_txt.exists() {
        return Ok(Box::new(
            WordPieceTokenizer::from_file(&vocab_txt)?.with_max_seq_len(max_seq_len),
        ));
    }

    if tokenizer_model.exists() {
        return Ok(Box::new(
            SentencePieceTokenizer::from_model_file(&tokenizer_model)?
                .with_max_seq_len(max_seq_len),
        ));
    }

    Err(InferenceError::ModelNotFound(format!(
        "no supported tokenizer files found in {}",
        model_dir.display()
    )))
}

/// **Unstable**: internal JSON representation for tokenizer metadata parsing;
/// not intended for direct use outside this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    /// **Unstable**: accessor for the internal JSON parser; subject to removal.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(value) => Some(value.as_str()),
            _ => None,
        }
    }

    /// **Unstable**: accessor for the internal JSON parser.
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(values) => Some(values.as_slice()),
            _ => None,
        }
    }

    /// **Unstable**: accessor for the internal JSON parser.
    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            JsonValue::Object(values) => Some(values),
            _ => None,
        }
    }

    /// **Unstable**: accessor for the internal JSON parser.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            JsonValue::Number(value) if *value >= 0.0 && value.fract() == 0.0 => {
                Some(*value as u64)
            }
            _ => None,
        }
    }

    /// **Unstable**: accessor for the internal JSON parser.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            JsonValue::Number(value) if value.fract() == 0.0 => Some(*value as i64),
            _ => None,
        }
    }

    /// **Unstable**: accessor for the internal JSON parser.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(value) => Some(*value),
            _ => None,
        }
    }

    /// **Unstable**: accessor for the internal JSON parser.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// **Unstable**: accessor for the internal JSON parser.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(map) => map.get(key),
            _ => None,
        }
    }
}

pub(crate) fn json_path<'a>(root: &'a JsonValue, path: &[&str]) -> Option<&'a JsonValue> {
    let mut current = root;
    for key in path {
        current = current.get(key)?;
    }
    Some(current)
}

pub(crate) fn parse_json(text: &str) -> Result<JsonValue, InferenceError> {
    let mut parser = JsonParser::new(text);
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    if !parser.is_eof() {
        return Err(InferenceError::Tokenizer(format!(
            "unexpected trailing JSON data at byte {}",
            parser.pos
        )));
    }
    Ok(value)
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            pos: 0,
        }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.pos += 1;
        Some(byte)
    }

    fn skip_whitespace(&mut self) {
        while let Some(byte) = self.peek() {
            if matches!(byte, b' ' | b'\n' | b'\r' | b'\t') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, InferenceError> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'n') => self.parse_literal(b"null", JsonValue::Null),
            Some(b't') => self.parse_literal(b"true", JsonValue::Bool(true)),
            Some(b'f') => self.parse_literal(b"false", JsonValue::Bool(false)),
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(other) => Err(InferenceError::Tokenizer(format!(
                "unexpected JSON byte 0x{other:02x} at offset {}",
                self.pos
            ))),
            None => Err(InferenceError::Tokenizer(
                "unexpected end of JSON input".into(),
            )),
        }
    }

    fn parse_literal(
        &mut self,
        literal: &[u8],
        value: JsonValue,
    ) -> Result<JsonValue, InferenceError> {
        if self.bytes.get(self.pos..self.pos + literal.len()) == Some(literal) {
            self.pos += literal.len();
            Ok(value)
        } else {
            Err(InferenceError::Tokenizer(format!(
                "invalid JSON literal at offset {}",
                self.pos
            )))
        }
    }

    fn parse_string(&mut self) -> Result<String, InferenceError> {
        if self.next() != Some(b'"') {
            return Err(InferenceError::Tokenizer(format!(
                "expected JSON string at offset {}",
                self.pos
            )));
        }

        let mut out = String::new();
        while let Some(byte) = self.next() {
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let escaped = self.next().ok_or_else(|| {
                        InferenceError::Tokenizer("unterminated JSON escape sequence".into())
                    })?;
                    match escaped {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let code = self.parse_u16_hex()?;
                            if (0xD800..=0xDBFF).contains(&code) {
                                if self.next() != Some(b'\\') || self.next() != Some(b'u') {
                                    return Err(InferenceError::Tokenizer(
                                        "invalid JSON surrogate pair".into(),
                                    ));
                                }
                                let low = self.parse_u16_hex()?;
                                if !(0xDC00..=0xDFFF).contains(&low) {
                                    return Err(InferenceError::Tokenizer(
                                        "invalid JSON low surrogate".into(),
                                    ));
                                }
                                let high_ten = (code as u32) - 0xD800;
                                let low_ten = (low as u32) - 0xDC00;
                                let scalar = 0x10000 + ((high_ten << 10) | low_ten);
                                let ch = char::from_u32(scalar).ok_or_else(|| {
                                    InferenceError::Tokenizer("invalid JSON unicode scalar".into())
                                })?;
                                out.push(ch);
                            } else {
                                let ch = char::from_u32(code as u32).ok_or_else(|| {
                                    InferenceError::Tokenizer("invalid JSON unicode scalar".into())
                                })?;
                                out.push(ch);
                            }
                        }
                        other => {
                            return Err(InferenceError::Tokenizer(format!(
                                "invalid JSON escape 0x{other:02x}"
                            )));
                        }
                    }
                }
                byte if byte < 0x20 => {
                    return Err(InferenceError::Tokenizer(
                        "control byte inside JSON string".into(),
                    ));
                }
                _ => {
                    let start = self.pos - 1;
                    let mut end = self.pos;
                    while end < self.bytes.len()
                        && self.bytes[end] != b'"'
                        && self.bytes[end] != b'\\'
                        && self.bytes[end] >= 0x20
                    {
                        end += 1;
                    }
                    out.push_str(std::str::from_utf8(&self.bytes[start..end]).map_err(|e| {
                        InferenceError::Tokenizer(format!("invalid UTF-8 in JSON string: {e}"))
                    })?);
                    self.pos = end;
                }
            }
        }

        Err(InferenceError::Tokenizer("unterminated JSON string".into()))
    }

    fn parse_u16_hex(&mut self) -> Result<u16, InferenceError> {
        let digits = self
            .bytes
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| InferenceError::Tokenizer("truncated JSON unicode escape".into()))?;
        self.pos += 4;
        let raw = std::str::from_utf8(digits)
            .map_err(|e| InferenceError::Tokenizer(format!("invalid JSON unicode escape: {e}")))?;
        u16::from_str_radix(raw, 16)
            .map_err(|e| InferenceError::Tokenizer(format!("invalid JSON unicode escape: {e}")))
    }

    fn parse_number(&mut self) -> Result<JsonValue, InferenceError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }

        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => {
                return Err(InferenceError::Tokenizer(format!(
                    "invalid JSON number at offset {start}"
                )));
            }
        }

        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(InferenceError::Tokenizer(format!(
                    "invalid JSON fractional part at offset {}",
                    self.pos
                )));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(InferenceError::Tokenizer(format!(
                    "invalid JSON exponent at offset {}",
                    self.pos
                )));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }

        let raw = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|e| InferenceError::Tokenizer(format!("invalid JSON number UTF-8: {e}")))?;
        let value = raw.parse::<f64>().map_err(|e| {
            InferenceError::Tokenizer(format!("failed to parse JSON number {raw:?}: {e}"))
        })?;
        Ok(JsonValue::Number(value))
    }

    fn parse_array(&mut self) -> Result<JsonValue, InferenceError> {
        if self.next() != Some(b'[') {
            return Err(InferenceError::Tokenizer(format!(
                "expected JSON array at offset {}",
                self.pos
            )));
        }

        let mut values = Vec::new();
        loop {
            self.skip_whitespace();
            if self.peek() == Some(b']') {
                self.pos += 1;
                return Ok(JsonValue::Array(values));
            }
            values.push(self.parse_value()?);
            self.skip_whitespace();
            match self.next() {
                Some(b',') => continue,
                Some(b']') => return Ok(JsonValue::Array(values)),
                _ => {
                    return Err(InferenceError::Tokenizer(format!(
                        "expected ',' or ']' in JSON array at offset {}",
                        self.pos
                    )));
                }
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, InferenceError> {
        if self.next() != Some(b'{') {
            return Err(InferenceError::Tokenizer(format!(
                "expected JSON object at offset {}",
                self.pos
            )));
        }

        let mut map = BTreeMap::new();
        loop {
            self.skip_whitespace();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                return Ok(JsonValue::Object(map));
            }
            let key = self.parse_string()?;
            self.skip_whitespace();
            if self.next() != Some(b':') {
                return Err(InferenceError::Tokenizer(format!(
                    "expected ':' after JSON object key at offset {}",
                    self.pos
                )));
            }
            let value = self.parse_value()?;
            map.insert(key, value);
            self.skip_whitespace();
            match self.next() {
                Some(b',') => continue,
                Some(b'}') => return Ok(JsonValue::Object(map)),
                _ => {
                    return Err(InferenceError::Tokenizer(format!(
                        "expected ',' or '}}' in JSON object at offset {}",
                        self.pos
                    )));
                }
            }
        }
    }
}

/// Extract a token->id vocabulary map from a JSON object.
pub(crate) fn json_object_to_vocab(
    value: &JsonValue,
) -> Result<HashMap<String, u32>, InferenceError> {
    let object = value
        .as_object()
        .ok_or_else(|| InferenceError::Tokenizer("expected JSON object for vocabulary".into()))?;

    let mut vocab = HashMap::with_capacity(object.len());
    for (token, id_value) in object {
        let id = id_value.as_u64().ok_or_else(|| {
            InferenceError::Tokenizer(format!("invalid non-integer token id for {token:?}"))
        })?;
        // Token ids are stored as u32 everywhere downstream. A JSON value past u32::MAX
        // would silently wrap under `as u32` (e.g. 2^32 -> 0) and alias an unrelated
        // token; reject it instead of corrupting the vocabulary.
        let id = u32::try_from(id).map_err(|_| {
            InferenceError::Tokenizer(format!("token id {id} for {token:?} exceeds u32 range"))
        })?;
        vocab.insert(token.clone(), id);
    }
    Ok(vocab)
}

/// Extract a token->id vocabulary map from a `vocab.txt`-style line list.
pub(crate) fn vocab_txt_to_map(vocab_text: &str) -> HashMap<String, u32> {
    let mut vocab = HashMap::new();
    for (idx, line) in vocab_text.lines().enumerate() {
        let token = line.trim_end_matches('\r').to_string();
        // The line index is the token id. Stop rather than wrap if it would exceed u32
        // range (a vocab.txt past 2^32 lines is malformed); enumerate is monotonic, so no
        // later line could fit either.
        let Ok(id) = u32::try_from(idx) else {
            break;
        };
        vocab.insert(token, id);
    }
    vocab
}

/// Convert a token->id vocabulary map into an ID-indexed token table.
pub(crate) fn invert_vocab(vocab: &HashMap<String, u32>) -> Result<Vec<String>, InferenceError> {
    let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
    // `id_to_token` is a dense table sized to the largest id. Real tokenizer vocabs are
    // dense (ids 0..len), so gaps are small and tolerated. A malformed tokenizer.json with
    // a single sparse huge id (e.g. 4_294_967_295) would otherwise force a ~100 GB
    // allocation here and abort the process at load. Cap the table at 16x the entry count:
    // generous headroom for any real gap, a hard ceiling on the malformed-input blast radius.
    let max_slots = vocab.len().saturating_mul(16);
    if max_id >= max_slots {
        return Err(InferenceError::Tokenizer(format!(
            "vocabulary id {max_id} out of range for {} entries (sparse or malformed)",
            vocab.len()
        )));
    }
    let mut id_to_token = vec![String::new(); max_id + 1];
    for (token, &id) in vocab {
        let slot = &mut id_to_token[id as usize];
        if !slot.is_empty() {
            return Err(InferenceError::Tokenizer(format!(
                "duplicate token id {id} for {token:?}"
            )));
        }
        *slot = token.clone();
    }
    Ok(id_to_token)
}

/// Parse `added_tokens` special-token metadata from `tokenizer.json`.
pub(crate) fn parse_added_tokens(root: &JsonValue) -> HashMap<String, u32> {
    let mut tokens = HashMap::new();
    let Some(array) = root.get("added_tokens").and_then(JsonValue::as_array) else {
        return tokens;
    };

    for item in array {
        let Some(object) = item.as_object() else {
            continue;
        };
        let Some(content) = object.get("content").and_then(JsonValue::as_str) else {
            continue;
        };
        // Skip an id past u32::MAX rather than wrap it under `as u32` into a colliding
        // id (parse_added_tokens is lenient by contract and already skips malformed entries).
        let Some(id) = object
            .get("id")
            .and_then(JsonValue::as_u64)
            .and_then(|v| u32::try_from(v).ok())
        else {
            continue;
        };
        tokens.insert(content.to_string(), id);
    }

    tokens
}

/// Parse the subset of `added_tokens` that should be **rendered as literal text**
/// when decoding, i.e. those whose `"special"` flag is `false` (or absent).
///
/// HF's `decode(skip_special_tokens=True)` — the chat default — skips only the
/// `special=true` set (control/chat/vision markers like `<|im_end|>`), while
/// `special=false` added tokens (`<think>`/`</think>`, `<tool_call>`, FIM markers)
/// are emitted verbatim. The base BPE vocab does not contain any added-token ids,
/// so without this map their decode falls through to nothing and they are silently
/// swallowed from the output stream. Returns `content -> id`.
pub(crate) fn parse_rendered_added_tokens(root: &JsonValue) -> HashMap<String, u32> {
    let mut tokens = HashMap::new();
    let Some(array) = root.get("added_tokens").and_then(JsonValue::as_array) else {
        return tokens;
    };

    for item in array {
        let Some(object) = item.as_object() else {
            continue;
        };
        // Absent "special" defaults to false (matches HF AddedToken), so render it.
        if object
            .get("special")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false)
        {
            continue;
        }
        let Some(content) = object.get("content").and_then(JsonValue::as_str) else {
            continue;
        };
        let Some(id) = object
            .get("id")
            .and_then(JsonValue::as_u64)
            .and_then(|v| u32::try_from(v).ok())
        else {
            continue;
        };
        tokens.insert(content.to_string(), id);
    }

    tokens
}

pub(crate) fn known_special_id(vocab: &HashMap<String, u32>, names: &[&str]) -> Option<u32> {
    names.iter().find_map(|name| vocab.get(*name).copied())
}

pub(crate) struct PostProcessorFlags {
    pub add_bos: bool,
    pub add_eos: bool,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
}

pub(crate) fn parse_post_processor_flags(root: &JsonValue) -> PostProcessorFlags {
    let default = PostProcessorFlags {
        add_bos: false,
        add_eos: false,
        bos_id: None,
        eos_id: None,
    };

    let Some(pp) = root.get("post_processor") else {
        return default;
    };

    let Some(template) = find_template_processing(pp) else {
        return default;
    };

    let Some(single) = template.get("single").and_then(JsonValue::as_array) else {
        return default;
    };

    let special_tokens = template.get("special_tokens");
    let mut add_bos = false;
    let mut add_eos = false;
    let mut bos_id = None;
    let mut eos_id = None;
    let mut seen_sequence = false;

    for item in single {
        if let Some(st) = item.get("SpecialToken") {
            let token_name = st.get("id").and_then(JsonValue::as_str);
            if !seen_sequence {
                add_bos = true;
                if let Some(name) = token_name {
                    bos_id = resolve_template_token_id(special_tokens, name);
                }
            } else {
                add_eos = true;
                if let Some(name) = token_name {
                    eos_id = resolve_template_token_id(special_tokens, name);
                }
            }
        } else if item.get("Sequence").is_some() {
            seen_sequence = true;
        }
    }

    PostProcessorFlags {
        add_bos,
        add_eos,
        bos_id,
        eos_id,
    }
}

fn find_template_processing(pp: &JsonValue) -> Option<&JsonValue> {
    let pp_type = pp.get("type")?.as_str()?;
    match pp_type {
        "TemplateProcessing" => Some(pp),
        "Sequence" => {
            let processors = pp.get("processors")?.as_array()?;
            processors
                .iter()
                .find(|p| p.get("type").and_then(JsonValue::as_str) == Some("TemplateProcessing"))
        }
        _ => None,
    }
}

fn resolve_template_token_id(special_tokens: Option<&JsonValue>, name: &str) -> Option<u32> {
    let ids = special_tokens?.get(name)?.get("ids")?.as_array()?;
    // An id past u32::MAX is treated as absent rather than wrapped into a wrong id.
    ids.first()?.as_u64().and_then(|v| u32::try_from(v).ok())
}

/// **Unstable**: internal LRU cache used by tokenizer implementations;
/// not part of the public embedding API.
///
/// The cache uses a monotonic stamp plus a duplicate-tolerant queue to avoid
/// per-entry linked-list pointers while preserving LRU behavior.
#[derive(Debug)]
pub struct ThreadSafeLruCache<K, V>
where
    K: Eq + Hash + Clone,
{
    capacity: usize,
    state: Mutex<LruState<K, V>>,
}

#[derive(Debug)]
struct LruState<K, V>
where
    K: Eq + Hash + Clone,
{
    clock: u64,
    map: HashMap<K, LruEntry<V>>,
    order: VecDeque<(K, u64)>,
}

#[derive(Debug)]
struct LruEntry<V> {
    stamp: u64,
    value: Arc<V>,
}

impl<K, V> ThreadSafeLruCache<K, V>
where
    K: Eq + Hash + Clone,
{
    /// **Unstable**: construct with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            state: Mutex::new(LruState {
                clock: 0,
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
        }
    }

    /// **Unstable**: returns configured capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// **Unstable**: get a cached value by key.
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        if self.capacity == 0 {
            return None;
        }

        let mut state = self.state.lock().ok()?;
        let value = state.map.get(key)?.value.clone();
        state.clock = state.clock.wrapping_add(1);
        let stamp = state.clock;
        if let Some(entry) = state.map.get_mut(key) {
            entry.stamp = stamp;
        }
        state.order.push_back((key.clone(), stamp));
        Self::evict_if_needed(self.capacity, &mut state);
        Some(value)
    }

    /// **Unstable**: insert a value, returns an `Arc` to it.
    pub fn insert(&self, key: K, value: V) -> Arc<V> {
        let value = Arc::new(value);
        self.insert_arc(key, value.clone());
        value
    }

    /// **Unstable**: insert a pre-boxed `Arc` value.
    pub fn insert_arc(&self, key: K, value: Arc<V>) {
        if self.capacity == 0 {
            return;
        }

        if let Ok(mut state) = self.state.lock() {
            state.clock = state.clock.wrapping_add(1);
            let stamp = state.clock;
            state.map.insert(
                key.clone(),
                LruEntry {
                    stamp,
                    value: value.clone(),
                },
            );
            state.order.push_back((key, stamp));
            Self::evict_if_needed(self.capacity, &mut state);
        }
    }

    fn evict_if_needed(capacity: usize, state: &mut LruState<K, V>) {
        while state.map.len() > capacity {
            let Some((key, stamp)) = state.order.pop_front() else {
                break;
            };
            let should_remove = match state.map.get(&key) {
                Some(entry) => entry.stamp == stamp,
                None => false,
            };
            if should_remove {
                state.map.remove(&key);
            }
        }
    }
}

impl<K, V> Clone for ThreadSafeLruCache<K, V>
where
    K: Eq + Hash + Clone,
{
    fn clone(&self) -> Self {
        Self::new(self.capacity)
    }
}

#[cfg(test)]
mod tests {
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
        // special=true is excluded (HF skip_special_tokens=True swallows it).
        assert_eq!(rendered.get("<|im_start|>"), None);
        // special=false renders verbatim.
        assert_eq!(rendered.get("</think>").copied(), Some(2));
        // absent special field defaults to false → renders.
        assert_eq!(rendered.get("<think>").copied(), Some(3));
        assert_eq!(rendered.get("<tool_call>").copied(), Some(4));
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
}
