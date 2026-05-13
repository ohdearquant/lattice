//! SentencePiece unigram tokenizer.
//!
//! The implementation supports Hugging Face `tokenizer.json` and native
//! `tokenizer.model` protobuf payloads, builds a byte trie over pieces, and
//! performs Viterbi decoding with optional byte fallback.

use crate::error::InferenceError;
use crate::tokenizer::common::{
    JsonValue, ThreadSafeLruCache, TokenizedInput, Tokenizer, json_path, known_special_id, pad_ids,
    parse_json, push_eos_preserving_limit,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tracing::warn;

const DEFAULT_SENTENCEPIECE_CACHE_CAPACITY: usize = 8_192;
const DEFAULT_SENTENCEPIECE_MAX_SEQ_LEN: usize = 4_096;
const UNK_PENALTY: f32 = 10.0;
const META_SPACE: char = '\u{2581}';

/// **Unstable**: SentencePiece piece type; variant set mirrors protobuf enum and may expand.
///
/// SentencePiece piece type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentencePieceType {
    Unknown,
    Normal,
    Control,
    UserDefined,
    Byte,
    Unused,
}

#[derive(Debug, Clone)]
struct SentencePieceEntry {
    piece: String,
    score: f32,
    kind: SentencePieceType,
}

#[derive(Debug, Clone)]
struct ByteTrieNode {
    children: HashMap<u8, usize>,
    terminal: Option<u32>,
}

#[derive(Debug, Clone)]
struct ByteTrie {
    nodes: Vec<ByteTrieNode>,
}

#[derive(Debug, Clone)]
struct BestPathNode {
    score: f32,
    prev: Option<usize>,
    ids: Vec<u32>,
}

#[derive(Debug)]
struct SentencePieceInner {
    pieces: Vec<SentencePieceEntry>,
    id_to_piece: Vec<String>,
    trie: ByteTrie,
    min_score: f32,
    unk_id: u32,
    pad_id: u32,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    max_seq_len: usize,
    dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    byte_fallback_ids: Vec<Option<u32>>,
    cache: ThreadSafeLruCache<String, Vec<u32>>,
}

/// **Unstable**: SentencePiece unigram tokenizer; protobuf parsing and Viterbi internals evolving.
///
/// SentencePiece unigram tokenizer.
#[derive(Debug, Clone)]
pub struct SentencePieceTokenizer {
    inner: Arc<SentencePieceInner>,
}

impl ByteTrie {
    fn new() -> Self {
        Self {
            nodes: vec![ByteTrieNode {
                children: HashMap::new(),
                terminal: None,
            }],
        }
    }

    fn insert(&mut self, piece: &str, id: u32) {
        let mut node_idx = 0usize;
        for &byte in piece.as_bytes() {
            let next_idx = if let Some(&child) = self.nodes[node_idx].children.get(&byte) {
                child
            } else {
                let child = self.nodes.len();
                self.nodes.push(ByteTrieNode {
                    children: HashMap::new(),
                    terminal: None,
                });
                self.nodes[node_idx].children.insert(byte, child);
                child
            };
            node_idx = next_idx;
        }
        self.nodes[node_idx].terminal = Some(id);
    }

    fn matches(&self, bytes: &[u8], start: usize) -> Vec<(usize, u32)> {
        let mut out = Vec::new();
        let mut node_idx = 0usize;
        let mut pos = start;
        while pos < bytes.len() {
            let Some(&child) = self.nodes[node_idx].children.get(&bytes[pos]) else {
                break;
            };
            node_idx = child;
            pos += 1;
            if let Some(id) = self.nodes[node_idx].terminal {
                out.push((pos, id));
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
struct ParsedSentencePieceModel {
    pieces: Vec<SentencePieceEntry>,
    unk_id: Option<u32>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    pad_id: Option<u32>,
    dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
}

impl Default for ParsedSentencePieceModel {
    fn default() -> Self {
        Self {
            pieces: Vec::new(),
            unk_id: None,
            bos_id: None,
            eos_id: None,
            pad_id: None,
            dummy_prefix: true,
            remove_extra_whitespaces: true,
            escape_whitespaces: true,
        }
    }
}

impl SentencePieceTokenizer {
    /// **Unstable**: load from native protobuf; protobuf parsing logic subject to change.
    ///
    /// Load from a native `tokenizer.model` SentencePiece protobuf file.
    pub fn from_model_file(path: &Path) -> Result<Self, InferenceError> {
        let bytes = fs::read(path).map_err(|e| {
            InferenceError::Tokenizer(format!("failed to read {}: {e}", path.display()))
        })?;
        let model = parse_sentencepiece_model(&bytes)?;
        Self::from_parsed_model(
            model,
            DEFAULT_SENTENCEPIECE_CACHE_CAPACITY,
            DEFAULT_SENTENCEPIECE_MAX_SEQ_LEN,
        )
    }

    /// **Unstable**: load from HF tokenizer.json; JSON schema assumptions may change.
    ///
    /// Load from a Hugging Face `tokenizer.json` file.
    pub fn from_tokenizer_json(path: &Path) -> Result<Self, InferenceError> {
        let text = fs::read_to_string(path).map_err(|e| {
            InferenceError::Tokenizer(format!("failed to read {}: {e}", path.display()))
        })?;
        Self::from_tokenizer_json_str(&text)
    }

    /// **Unstable**: load from HF tokenizer.json string; JSON schema assumptions may change.
    ///
    /// Load from a Hugging Face `tokenizer.json` payload.
    pub fn from_tokenizer_json_str(text: &str) -> Result<Self, InferenceError> {
        let root = parse_json(text)?;
        let model_type = json_path(&root, &["model", "type"])
            .and_then(JsonValue::as_str)
            .unwrap_or("");
        if model_type != "Unigram" && model_type != "SentencePieceUnigram" {
            return Err(InferenceError::Tokenizer(format!(
                "expected Unigram tokenizer.json model type, found {model_type:?}"
            )));
        }

        let vocab_entries = json_path(&root, &["model", "vocab"])
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                InferenceError::Tokenizer("tokenizer.json missing model.vocab".into())
            })?;

        let mut pieces = Vec::with_capacity(vocab_entries.len());
        for entry in vocab_entries {
            let values = entry.as_array().ok_or_else(|| {
                InferenceError::Tokenizer("invalid SentencePiece vocab entry".into())
            })?;
            if values.len() < 2 {
                return Err(InferenceError::Tokenizer(
                    "invalid SentencePiece vocab entry length".into(),
                ));
            }
            let piece = values[0].as_str().ok_or_else(|| {
                InferenceError::Tokenizer("invalid SentencePiece piece string".into())
            })?;
            let score = values[1]
                .as_f64()
                .ok_or_else(|| InferenceError::Tokenizer("invalid SentencePiece score".into()))?
                as f32;
            pieces.push(SentencePieceEntry {
                piece: piece.to_string(),
                score,
                kind: SentencePieceType::Normal,
            });
        }

        let mut vocab_map = HashMap::new();
        for (idx, piece) in pieces.iter().enumerate() {
            vocab_map.insert(piece.piece.clone(), idx as u32);
        }

        let unk_id = json_path(&root, &["model", "unk_id"])
            .and_then(JsonValue::as_u64)
            .map(|v| v as u32)
            .or_else(|| known_special_id(&vocab_map, &["<unk>"]));
        let bos_id = known_special_id(&vocab_map, &["<bos>", "<s>"]);
        let eos_id = known_special_id(&vocab_map, &["<eos>", "</s>"]);
        let pad_id = known_special_id(&vocab_map, &["<pad>"]);

        let model = ParsedSentencePieceModel {
            pieces,
            unk_id,
            bos_id,
            eos_id,
            pad_id,
            dummy_prefix: true,
            remove_extra_whitespaces: true,
            escape_whitespaces: true,
        };

        Self::from_parsed_model(
            model,
            DEFAULT_SENTENCEPIECE_CACHE_CAPACITY,
            DEFAULT_SENTENCEPIECE_MAX_SEQ_LEN,
        )
    }

    fn from_parsed_model(
        model: ParsedSentencePieceModel,
        cache_capacity: usize,
        max_seq_len: usize,
    ) -> Result<Self, InferenceError> {
        let unk_id = model.unk_id.ok_or_else(|| {
            InferenceError::Tokenizer("SentencePiece model missing unk_id".into())
        })?;
        let pad_id = model.pad_id.unwrap_or(unk_id);
        let min_score = model
            .pieces
            .iter()
            .map(|piece| piece.score)
            .fold(f32::INFINITY, f32::min);
        let mut trie = ByteTrie::new();
        let mut byte_fallback_ids = vec![None; 256];
        let mut id_to_piece = Vec::with_capacity(model.pieces.len());

        for (idx, piece) in model.pieces.iter().enumerate() {
            let id = idx as u32;
            id_to_piece.push(piece.piece.clone());
            match piece.kind {
                SentencePieceType::Normal | SentencePieceType::UserDefined => {
                    trie.insert(piece.piece.as_str(), id);
                }
                SentencePieceType::Byte => {
                    if let Some(byte) = parse_byte_piece(piece.piece.as_str()) {
                        byte_fallback_ids[byte as usize] = Some(id);
                    }
                }
                SentencePieceType::Unknown
                | SentencePieceType::Control
                | SentencePieceType::Unused => {}
            }
        }

        let inner = SentencePieceInner {
            pieces: model.pieces,
            id_to_piece,
            trie,
            min_score,
            unk_id,
            pad_id,
            bos_id: model.bos_id,
            eos_id: model.eos_id,
            add_bos: false,
            add_eos: false,
            max_seq_len,
            dummy_prefix: model.dummy_prefix,
            remove_extra_whitespaces: model.remove_extra_whitespaces,
            escape_whitespaces: model.escape_whitespaces,
            byte_fallback_ids,
            cache: ThreadSafeLruCache::new(cache_capacity),
        };

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Test and benchmark helper: build directly from pieces.
    #[cfg(test)]
    fn from_pieces_for_test(
        pieces: Vec<SentencePieceEntry>,
        unk_id: u32,
    ) -> Result<Self, InferenceError> {
        Self::from_parsed_model(
            ParsedSentencePieceModel {
                pieces,
                unk_id: Some(unk_id),
                bos_id: None,
                eos_id: None,
                pad_id: Some(unk_id),
                dummy_prefix: true,
                remove_extra_whitespaces: true,
                escape_whitespaces: true,
            },
            DEFAULT_SENTENCEPIECE_CACHE_CAPACITY,
            DEFAULT_SENTENCEPIECE_MAX_SEQ_LEN,
        )
    }

    /// **Unstable**: override maximum sequence length; cloning inner Arc may change.
    ///
    /// Override the configured maximum sequence length.
    pub fn with_max_seq_len(&self, max_seq_len: usize) -> Self {
        let inner = SentencePieceInner {
            pieces: self.inner.pieces.clone(),
            id_to_piece: self.inner.id_to_piece.clone(),
            trie: self.inner.trie.clone(),
            min_score: self.inner.min_score,
            unk_id: self.inner.unk_id,
            pad_id: self.inner.pad_id,
            bos_id: self.inner.bos_id,
            eos_id: self.inner.eos_id,
            add_bos: self.inner.add_bos,
            add_eos: self.inner.add_eos,
            max_seq_len,
            dummy_prefix: self.inner.dummy_prefix,
            remove_extra_whitespaces: self.inner.remove_extra_whitespaces,
            escape_whitespaces: self.inner.escape_whitespaces,
            byte_fallback_ids: self.inner.byte_fallback_ids.clone(),
            cache: self.inner.cache.clone(),
        };
        Self {
            inner: Arc::new(inner),
        }
    }

    /// **Unstable**: vocabulary size accessor; count may shift with special-token handling.
    ///
    /// Access the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.id_to_piece.len()
    }

    /// **Unstable**: maximum sequence length accessor.
    ///
    /// Access the maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.inner.max_seq_len
    }

    /// **Unstable**: reverse lookup for debugging; id_to_piece table layout may change.
    ///
    /// Reverse lookup for debugging.
    pub fn token_for_id(&self, id: u32) -> Option<&str> {
        self.inner.id_to_piece.get(id as usize).map(String::as_str)
    }

    fn normalize(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        let mut out = String::with_capacity(text.len() + 4);
        let mut prev_was_space = false;

        if self.inner.dummy_prefix {
            out.push(META_SPACE);
            prev_was_space = true;
        }

        for ch in text.chars() {
            if ch.is_whitespace() {
                if !self.inner.remove_extra_whitespaces || !prev_was_space {
                    if self.inner.escape_whitespaces {
                        out.push(META_SPACE);
                    } else {
                        out.push(' ');
                    }
                }
                prev_was_space = true;
            } else {
                out.push(ch);
                prev_was_space = false;
            }
        }

        out
    }

    fn add_special_tokens_and_truncate(&self, ids: &mut Vec<u32>) {
        if self.inner.add_bos {
            if let Some(id) = self.inner.bos_id {
                ids.insert(0, id);
            }
        }

        if self.inner.add_eos {
            if let Some(id) = self.inner.eos_id {
                if ids.len() >= self.inner.max_seq_len {
                    warn!(
                        original_len = ids.len().saturating_add(1),
                        max_seq_len = self.inner.max_seq_len,
                        "truncating SentencePiece tokenized input to preserve EOS within max_seq_len"
                    );
                }
                push_eos_preserving_limit(ids, id, self.inner.max_seq_len);
            } else if ids.len() > self.inner.max_seq_len {
                warn!(
                    original_len = ids.len(),
                    max_seq_len = self.inner.max_seq_len,
                    "truncating SentencePiece tokenized input to max_seq_len"
                );
                ids.truncate(self.inner.max_seq_len);
            }
        } else if ids.len() > self.inner.max_seq_len {
            warn!(
                original_len = ids.len(),
                max_seq_len = self.inner.max_seq_len,
                "truncating SentencePiece tokenized input to max_seq_len"
            );
            ids.truncate(self.inner.max_seq_len);
        }
    }

    fn tokenize_to_ids(&self, text: &str) -> Vec<u32> {
        let normalized = self.normalize(text);
        if normalized.is_empty() {
            let mut ids = Vec::new();
            self.add_special_tokens_and_truncate(&mut ids);
            return ids;
        }

        if let Some(cached) = self.inner.cache.get(&normalized) {
            let mut ids = cached.as_ref().clone();
            self.add_special_tokens_and_truncate(&mut ids);
            return ids;
        }

        let mut ids = self.viterbi_encode(&normalized);
        self.inner.cache.insert(normalized, ids.clone());

        self.add_special_tokens_and_truncate(&mut ids);
        ids
    }

    fn viterbi_encode(&self, sentence: &str) -> Vec<u32> {
        let bytes = sentence.as_bytes();
        let size = bytes.len();
        if size == 0 {
            return Vec::new();
        }

        let unk_score = self.inner.min_score - UNK_PENALTY;
        let mut best = vec![
            BestPathNode {
                score: f32::NEG_INFINITY,
                prev: None,
                ids: Vec::new(),
            };
            size + 1
        ];
        best[0].score = 0.0;

        let mut start = 0usize;
        while start < size {
            let ch = sentence[start..]
                .chars()
                .next()
                .expect("invariant: start is inside non-empty UTF-8 sentence");
            let char_len = ch.len_utf8();
            let base_score = best[start].score;
            if !base_score.is_finite() {
                start += char_len;
                continue;
            }

            let mut has_single_piece = false;
            for (end, id) in self.inner.trie.matches(bytes, start) {
                let piece = &self.inner.pieces[id as usize];
                let candidate = base_score + piece.score;
                if candidate > best[end].score {
                    best[end].score = candidate;
                    best[end].prev = Some(start);
                    best[end].ids.clear();
                    best[end].ids.push(id);
                }
                if end == start + char_len {
                    has_single_piece = true;
                }
            }

            if !has_single_piece {
                let end = start + char_len;
                if let Some(byte_ids) = self.byte_fallback_for_char(ch) {
                    let score = byte_ids
                        .iter()
                        .map(|&id| self.inner.pieces[id as usize].score)
                        .fold(base_score, |acc, value| acc + value);
                    if score > best[end].score {
                        best[end].score = score;
                        best[end].prev = Some(start);
                        best[end].ids = byte_ids;
                    }
                } else {
                    let score = base_score + unk_score;
                    if score > best[end].score {
                        best[end].score = score;
                        best[end].prev = Some(start);
                        best[end].ids.clear();
                        best[end].ids.push(self.inner.unk_id);
                    }
                }
            }

            start += char_len;
        }

        let mut results_rev = Vec::new();
        let mut end = size;
        while end > 0 {
            let node = &best[end];
            let Some(prev) = node.prev else {
                results_rev.push(self.inner.unk_id);
                let ch = sentence[..end]
                    .chars()
                    .last()
                    .expect("invariant: end is after a UTF-8 character boundary");
                end = end.saturating_sub(ch.len_utf8());
                continue;
            };
            for &id in node.ids.iter().rev() {
                results_rev.push(id);
            }
            end = prev;
        }
        results_rev.reverse();
        results_rev
    }

    fn byte_fallback_for_char(&self, ch: char) -> Option<Vec<u32>> {
        let mut ids = Vec::new();
        let mut buf = [0u8; 4];
        for &byte in ch.encode_utf8(&mut buf).as_bytes() {
            let id = self.inner.byte_fallback_ids[byte as usize]?;
            ids.push(id);
        }
        Some(ids)
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn tokenize(&self, text: &str) -> TokenizedInput {
        let ids = self.tokenize_to_ids(text);
        pad_ids(ids, self.inner.max_seq_len, self.inner.pad_id)
    }

    fn tokenize_batch(&self, texts: &[&str]) -> Vec<TokenizedInput> {
        if texts.is_empty() {
            return Vec::new();
        }
        let mut max_len = 0usize;
        let mut all = Vec::with_capacity(texts.len());
        for text in texts {
            let ids = self.tokenize_to_ids(text);
            max_len = max_len.max(ids.len());
            all.push(ids);
        }
        all.into_iter()
            .map(|ids| pad_ids(ids, max_len, self.inner.pad_id))
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.inner.id_to_piece.len()
    }

    fn max_seq_len(&self) -> usize {
        self.inner.max_seq_len
    }
}

fn parse_byte_piece(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn parse_sentencepiece_model(bytes: &[u8]) -> Result<ParsedSentencePieceModel, InferenceError> {
    let mut model = ParsedSentencePieceModel::default();
    let mut pos = 0usize;
    while pos < bytes.len() {
        let (field, wire_type) = read_key(bytes, &mut pos)?;
        match (field, wire_type) {
            (1, 2) => {
                let data = read_length_delimited(bytes, &mut pos)?;
                model.pieces.push(parse_piece_message(data)?);
            }
            (2, 2) => {
                let data = read_length_delimited(bytes, &mut pos)?;
                parse_trainer_spec(data, &mut model)?;
            }
            (3, 2) => {
                let data = read_length_delimited(bytes, &mut pos)?;
                parse_normalizer_spec(data, &mut model)?;
            }
            _ => skip_field(bytes, &mut pos, wire_type)?,
        }
    }
    if model.unk_id.is_none() {
        model.unk_id = model
            .pieces
            .iter()
            .position(|piece| piece.kind == SentencePieceType::Unknown)
            .map(|idx| idx as u32)
            .or_else(|| {
                model
                    .pieces
                    .iter()
                    .position(|piece| piece.piece == "<unk>")
                    .map(|idx| idx as u32)
            });
    }
    Ok(model)
}

fn parse_piece_message(bytes: &[u8]) -> Result<SentencePieceEntry, InferenceError> {
    let mut pos = 0usize;
    let mut piece = String::new();
    let mut score = 0.0f32;
    let mut kind = SentencePieceType::Normal;
    while pos < bytes.len() {
        let (field, wire_type) = read_key(bytes, &mut pos)?;
        match (field, wire_type) {
            (1, 2) => {
                let data = read_length_delimited(bytes, &mut pos)?;
                piece = std::str::from_utf8(data)
                    .map_err(|e| {
                        InferenceError::Tokenizer(format!("invalid SentencePiece piece UTF-8: {e}"))
                    })?
                    .to_string();
            }
            (2, 5) => {
                score = read_fixed32(bytes, &mut pos)?;
            }
            (3, 0) => {
                let raw = read_varint(bytes, &mut pos)? as u32;
                kind = match raw {
                    2 => SentencePieceType::Unknown,
                    3 => SentencePieceType::Control,
                    4 => SentencePieceType::UserDefined,
                    5 => SentencePieceType::Unused,
                    6 => SentencePieceType::Byte,
                    _ => SentencePieceType::Normal,
                };
            }
            _ => skip_field(bytes, &mut pos, wire_type)?,
        }
    }
    Ok(SentencePieceEntry { piece, score, kind })
}

fn parse_trainer_spec(
    bytes: &[u8],
    model: &mut ParsedSentencePieceModel,
) -> Result<(), InferenceError> {
    let mut pos = 0usize;
    while pos < bytes.len() {
        let (field, wire_type) = read_key(bytes, &mut pos)?;
        match (field, wire_type) {
            (40, 0) => model.unk_id = Some(read_varint(bytes, &mut pos)? as u32),
            (41, 0) => model.bos_id = Some(read_varint(bytes, &mut pos)? as u32),
            (42, 0) => model.eos_id = Some(read_varint(bytes, &mut pos)? as u32),
            (43, 0) => model.pad_id = Some(read_varint(bytes, &mut pos)? as u32),
            _ => skip_field(bytes, &mut pos, wire_type)?,
        }
    }
    Ok(())
}

fn parse_normalizer_spec(
    bytes: &[u8],
    model: &mut ParsedSentencePieceModel,
) -> Result<(), InferenceError> {
    let mut pos = 0usize;
    while pos < bytes.len() {
        let (field, wire_type) = read_key(bytes, &mut pos)?;
        match (field, wire_type) {
            (3, 0) => model.dummy_prefix = read_varint(bytes, &mut pos)? != 0,
            (4, 0) => model.remove_extra_whitespaces = read_varint(bytes, &mut pos)? != 0,
            (5, 0) => model.escape_whitespaces = read_varint(bytes, &mut pos)? != 0,
            _ => skip_field(bytes, &mut pos, wire_type)?,
        }
    }
    Ok(())
}

fn read_key(bytes: &[u8], pos: &mut usize) -> Result<(u32, u8), InferenceError> {
    let key = read_varint(bytes, pos)? as u32;
    Ok((key >> 3, (key & 0x07) as u8))
}

fn read_varint(bytes: &[u8], pos: &mut usize) -> Result<u64, InferenceError> {
    let mut shift = 0u32;
    let mut value = 0u64;
    loop {
        let byte = *bytes
            .get(*pos)
            .ok_or_else(|| InferenceError::Tokenizer("unexpected end of protobuf input".into()))?;
        *pos += 1;
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
        shift += 7;
        if shift >= 64 {
            return Err(InferenceError::Tokenizer(
                "protobuf varint exceeds 64 bits".into(),
            ));
        }
    }
}

fn read_length_delimited<'a>(bytes: &'a [u8], pos: &mut usize) -> Result<&'a [u8], InferenceError> {
    let len = read_varint(bytes, pos)? as usize;
    let end = pos
        .checked_add(len)
        .ok_or_else(|| InferenceError::Tokenizer("protobuf length overflow".into()))?;
    let slice = bytes.get(*pos..end).ok_or_else(|| {
        InferenceError::Tokenizer("truncated protobuf length-delimited field".into())
    })?;
    *pos = end;
    Ok(slice)
}

fn read_fixed32(bytes: &[u8], pos: &mut usize) -> Result<f32, InferenceError> {
    let slice = bytes
        .get(*pos..*pos + 4)
        .ok_or_else(|| InferenceError::Tokenizer("truncated protobuf fixed32 field".into()))?;
    *pos += 4;
    Ok(f32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn skip_field(bytes: &[u8], pos: &mut usize, wire_type: u8) -> Result<(), InferenceError> {
    match wire_type {
        0 => {
            let _ = read_varint(bytes, pos)?;
        }
        1 => {
            *pos = pos
                .checked_add(8)
                .ok_or_else(|| InferenceError::Tokenizer("protobuf skip overflow".into()))?;
            if *pos > bytes.len() {
                return Err(InferenceError::Tokenizer(
                    "truncated protobuf fixed64 field".into(),
                ));
            }
        }
        2 => {
            let len = read_varint(bytes, pos)? as usize;
            *pos = pos
                .checked_add(len)
                .ok_or_else(|| InferenceError::Tokenizer("protobuf skip overflow".into()))?;
            if *pos > bytes.len() {
                return Err(InferenceError::Tokenizer(
                    "truncated protobuf length-delimited field".into(),
                ));
            }
        }
        5 => {
            *pos = pos
                .checked_add(4)
                .ok_or_else(|| InferenceError::Tokenizer("protobuf skip overflow".into()))?;
            if *pos > bytes.len() {
                return Err(InferenceError::Tokenizer(
                    "truncated protobuf fixed32 field".into(),
                ));
            }
        }
        other => {
            return Err(InferenceError::Tokenizer(format!(
                "unsupported protobuf wire type {other}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
