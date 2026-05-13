//! Byte-level BPE tokenizer.
//!
//! This module implements a dependency-free GPT-2/Qwen-style tokenizer with
//! byte-level pre-tokenization, merge-rank driven BPE, a per-word LRU cache,
//! and `tokenizer.json` / `vocab.json` + `merges.txt` loading.

use crate::error::InferenceError;
use crate::tokenizer::common::{
    JsonValue, ThreadSafeLruCache, TokenizedInput, Tokenizer, invert_vocab, json_object_to_vocab,
    json_path, known_special_id, pad_ids, parse_added_tokens, parse_json,
    push_eos_preserving_limit, vocab_txt_to_map,
};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tracing::warn;

const DEFAULT_BPE_CACHE_CAPACITY: usize = 8_192;
const DEFAULT_BPE_MAX_SEQ_LEN: usize = 4_096;

/// **Unstable**: byte-level BPE tokenizer for Qwen-family models; API evolving.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    inner: Arc<BpeInner>,
}

#[derive(Debug)]
struct BpeInner {
    vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    merges: HashMap<String, HashMap<String, usize>>,
    byte_encoder: Vec<char>,
    special_tokens: HashMap<String, u32>,
    special_tokens_sorted: Vec<String>,
    pad_id: u32,
    unk_id: Option<u32>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    max_seq_len: usize,
    cache: ThreadSafeLruCache<String, Vec<u32>>,
}

#[derive(Debug, Clone)]
struct BpeNode {
    token: String,
    prev: Option<usize>,
    next: Option<usize>,
    alive: bool,
    version: u64,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct Candidate {
    rank: usize,
    left: usize,
    right: usize,
    left_version: u64,
    right_version: u64,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Default)]
struct TokenizeScratch {
    ids: Vec<u32>,
    word_ids: Vec<u32>,
}

impl BpeTokenizer {
    /// **Unstable**: load from vocab.json + merges.txt (GPT-2 format).
    pub fn from_files(vocab_path: &Path, merges_path: &Path) -> Result<Self, InferenceError> {
        let vocab_text = fs::read_to_string(vocab_path).map_err(|e| {
            InferenceError::Tokenizer(format!("failed to read {}: {e}", vocab_path.display()))
        })?;
        let merges_text = fs::read_to_string(merges_path).map_err(|e| {
            InferenceError::Tokenizer(format!("failed to read {}: {e}", merges_path.display()))
        })?;
        let vocab_json = parse_json(&vocab_text)?;
        let vocab = json_object_to_vocab(&vocab_json)?;
        Self::from_vocab_and_merges(vocab, parse_merges_txt(&merges_text))
    }

    /// **Unstable**: load from plain vocab.txt + merges.txt.
    pub fn from_vocab_txt_and_merges(
        vocab_txt_path: &Path,
        merges_path: &Path,
    ) -> Result<Self, InferenceError> {
        let vocab_text = fs::read_to_string(vocab_txt_path).map_err(|e| {
            InferenceError::Tokenizer(format!("failed to read {}: {e}", vocab_txt_path.display()))
        })?;
        let merges_text = fs::read_to_string(merges_path).map_err(|e| {
            InferenceError::Tokenizer(format!("failed to read {}: {e}", merges_path.display()))
        })?;
        Self::from_vocab_and_merges(
            vocab_txt_to_map(&vocab_text),
            parse_merges_txt(&merges_text),
        )
    }

    /// **Unstable**: load from Hugging Face tokenizer.json file.
    pub fn from_tokenizer_json(path: &Path) -> Result<Self, InferenceError> {
        let text = fs::read_to_string(path).map_err(|e| {
            InferenceError::Tokenizer(format!("failed to read {}: {e}", path.display()))
        })?;
        Self::from_tokenizer_json_str(&text)
    }

    /// **Unstable**: load from a Hugging Face tokenizer.json string.
    pub fn from_tokenizer_json_str(text: &str) -> Result<Self, InferenceError> {
        let root = parse_json(text)?;
        let model_type = json_path(&root, &["model", "type"])
            .and_then(JsonValue::as_str)
            .unwrap_or("");
        if model_type != "BPE" {
            return Err(InferenceError::Tokenizer(format!(
                "expected BPE tokenizer.json model type, found {model_type:?}"
            )));
        }

        let vocab =
            json_object_to_vocab(json_path(&root, &["model", "vocab"]).ok_or_else(|| {
                InferenceError::Tokenizer("tokenizer.json missing model.vocab".into())
            })?)?;

        let merges_value = json_path(&root, &["model", "merges"]).ok_or_else(|| {
            InferenceError::Tokenizer("tokenizer.json missing model.merges".into())
        })?;
        let merges = parse_merges_json(merges_value)?;
        let added = parse_added_tokens(&root);

        let mut tokenizer = Self::from_vocab_and_merges_with_config(
            vocab,
            merges,
            added,
            DEFAULT_BPE_CACHE_CAPACITY,
            DEFAULT_BPE_MAX_SEQ_LEN,
        )?;

        if let Some(unk_token) =
            json_path(&root, &["model", "unk_token"]).and_then(JsonValue::as_str)
        {
            tokenizer = tokenizer.with_unk_token(unk_token);
        }

        Ok(tokenizer)
    }

    /// **Unstable**: build from in-memory vocabulary and merge list.
    pub fn from_vocab_and_merges(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
    ) -> Result<Self, InferenceError> {
        Self::from_vocab_and_merges_with_config(
            vocab,
            merges,
            HashMap::new(),
            DEFAULT_BPE_CACHE_CAPACITY,
            DEFAULT_BPE_MAX_SEQ_LEN,
        )
    }

    fn from_vocab_and_merges_with_config(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
        added_tokens: HashMap<String, u32>,
        cache_capacity: usize,
        max_seq_len: usize,
    ) -> Result<Self, InferenceError> {
        let id_to_token = invert_vocab(&vocab)?;
        let mut merge_ranks: HashMap<String, HashMap<String, usize>> = HashMap::new();
        for (rank, (left, right)) in merges.into_iter().enumerate() {
            merge_ranks.entry(left).or_default().insert(right, rank);
        }

        let mut special_tokens = added_tokens;
        for name in [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<pad>",
            "<|pad|>",
            "<bos>",
            "<eos>",
            "<unk>",
        ] {
            if let Some(&id) = vocab.get(name) {
                special_tokens.entry(name.to_string()).or_insert(id);
            }
        }

        let pad_id = known_special_id(&vocab, &["<|pad|>", "<pad>", "[PAD]"])
            .or_else(|| known_special_id(&special_tokens, &["<|pad|>", "<pad>"]))
            .or_else(|| known_special_id(&vocab, &["<|endoftext|>", "</s>"]))
            .unwrap_or(0);
        let unk_id = known_special_id(&vocab, &["<unk>", "[UNK]"]);
        let bos_id = known_special_id(&vocab, &["<bos>", "<s>"]);
        let eos_id = known_special_id(&vocab, &["<eos>", "</s>", "<|endoftext|>"]);

        let mut special_tokens_sorted: Vec<String> = special_tokens.keys().cloned().collect();
        special_tokens_sorted.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));

        let inner = BpeInner {
            vocab,
            id_to_token,
            merges: merge_ranks,
            byte_encoder: bytes_to_unicode(),
            special_tokens,
            special_tokens_sorted,
            pad_id,
            unk_id,
            bos_id,
            eos_id,
            add_bos: false,
            add_eos: false,
            max_seq_len,
            cache: ThreadSafeLruCache::new(cache_capacity),
        };

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// **Unstable**: clone tokenizer with a new max sequence length.
    pub fn with_max_seq_len(&self, max_seq_len: usize) -> Self {
        let inner = BpeInner {
            vocab: self.inner.vocab.clone(),
            id_to_token: self.inner.id_to_token.clone(),
            merges: self.inner.merges.clone(),
            byte_encoder: self.inner.byte_encoder.clone(),
            special_tokens: self.inner.special_tokens.clone(),
            special_tokens_sorted: self.inner.special_tokens_sorted.clone(),
            pad_id: self.inner.pad_id,
            unk_id: self.inner.unk_id,
            bos_id: self.inner.bos_id,
            eos_id: self.inner.eos_id,
            add_bos: self.inner.add_bos,
            add_eos: self.inner.add_eos,
            max_seq_len,
            cache: self.inner.cache.clone(),
        };
        Self {
            inner: Arc::new(inner),
        }
    }

    /// **Unstable**: clone tokenizer with a different unknown token.
    pub fn with_unk_token(&self, token: &str) -> Self {
        let unk_id = self.inner.vocab.get(token).copied();
        let inner = BpeInner {
            vocab: self.inner.vocab.clone(),
            id_to_token: self.inner.id_to_token.clone(),
            merges: self.inner.merges.clone(),
            byte_encoder: self.inner.byte_encoder.clone(),
            special_tokens: self.inner.special_tokens.clone(),
            special_tokens_sorted: self.inner.special_tokens_sorted.clone(),
            pad_id: self.inner.pad_id,
            unk_id,
            bos_id: self.inner.bos_id,
            eos_id: self.inner.eos_id,
            add_bos: self.inner.add_bos,
            add_eos: self.inner.add_eos,
            max_seq_len: self.inner.max_seq_len,
            cache: self.inner.cache.clone(),
        };
        Self {
            inner: Arc::new(inner),
        }
    }

    /// **Unstable**: enable EOS appending for last-token pooling (decoder-only embedding models).
    ///
    /// Required for decoder-only embedding models (e.g., Qwen3-Embedding) that
    /// use last-token pooling: the EOS token (`<|endoftext|>`) provides a
    /// learned summary representation. Without it, last-token pooling grabs
    /// a random content token, producing poorly discriminative embeddings.
    pub fn with_add_eos(self) -> Self {
        let inner = BpeInner {
            vocab: self.inner.vocab.clone(),
            id_to_token: self.inner.id_to_token.clone(),
            merges: self.inner.merges.clone(),
            byte_encoder: self.inner.byte_encoder.clone(),
            special_tokens: self.inner.special_tokens.clone(),
            special_tokens_sorted: self.inner.special_tokens_sorted.clone(),
            pad_id: self.inner.pad_id,
            unk_id: self.inner.unk_id,
            bos_id: self.inner.bos_id,
            eos_id: self.inner.eos_id,
            add_bos: self.inner.add_bos,
            add_eos: true,
            max_seq_len: self.inner.max_seq_len,
            cache: self.inner.cache.clone(),
        };
        Self {
            inner: Arc::new(inner),
        }
    }

    /// **Unstable**: configured maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.inner.max_seq_len
    }

    /// **Unstable**: number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.inner.id_to_token.len()
    }

    /// **Unstable**: reverse lookup; internal id_to_token table layout may change.
    ///
    /// Reverse lookup for debugging.
    pub fn token_for_id(&self, id: u32) -> Option<&str> {
        self.inner.id_to_token.get(id as usize).map(String::as_str)
    }

    /// **Unstable**: look up special token ID by name; set of known special tokens may change.
    ///
    /// Look up special token ID by name (e.g. "<|im_end|>").
    pub fn special_token_id(&self, name: &str) -> Option<u32> {
        self.inner.special_tokens.get(name).copied()
    }

    fn tokenize_to_ids(&self, text: &str) -> Vec<u32> {
        let mut scratch = TokenizeScratch::default();
        self.tokenize_to_ids_into(text, &mut scratch)
    }

    fn tokenize_to_ids_into(&self, text: &str, scratch: &mut TokenizeScratch) -> Vec<u32> {
        scratch.ids.clear();
        if self.inner.add_bos {
            if let Some(bos_id) = self.inner.bos_id {
                scratch.ids.push(bos_id);
            }
        }

        let mut segment_start = 0usize;
        let mut pos = 0usize;
        while pos < text.len() {
            if let Some((special_end, special_id)) = self.match_special(text, pos) {
                if segment_start < pos {
                    self.tokenize_regular_segment_into(&text[segment_start..pos], scratch);
                }
                scratch.ids.push(special_id);
                pos = special_end;
                segment_start = pos;
                continue;
            }

            let ch = text[pos..]
                .chars()
                .next()
                .expect("invariant: pos is inside non-empty UTF-8 text");
            pos += ch.len_utf8();
        }

        if segment_start < text.len() {
            self.tokenize_regular_segment_into(&text[segment_start..], scratch);
        }

        if self.inner.add_eos {
            if let Some(eos_id) = self.inner.eos_id {
                if scratch.ids.len() >= self.inner.max_seq_len {
                    warn!(
                        original_len = scratch.ids.len().saturating_add(1),
                        max_seq_len = self.inner.max_seq_len,
                        "truncating BPE tokenized input to preserve EOS within max_seq_len"
                    );
                }
                push_eos_preserving_limit(&mut scratch.ids, eos_id, self.inner.max_seq_len);
            } else if scratch.ids.len() > self.inner.max_seq_len {
                warn!(
                    original_len = scratch.ids.len(),
                    max_seq_len = self.inner.max_seq_len,
                    "truncating BPE tokenized input to max_seq_len"
                );
                scratch.ids.truncate(self.inner.max_seq_len);
            }
        } else if scratch.ids.len() > self.inner.max_seq_len {
            warn!(
                original_len = scratch.ids.len(),
                max_seq_len = self.inner.max_seq_len,
                "truncating BPE tokenized input to max_seq_len"
            );
            scratch.ids.truncate(self.inner.max_seq_len);
        }

        scratch.ids.clone()
    }

    fn tokenize_regular_segment_into(&self, text: &str, scratch: &mut TokenizeScratch) {
        for piece in byte_level_pretokenize(text) {
            if let Some(cached) = self.inner.cache.get(&piece) {
                scratch.ids.extend(cached.iter().copied());
                continue;
            }

            scratch.word_ids.clear();
            self.encode_piece_to_ids(&piece, &mut scratch.word_ids);
            self.inner.cache.insert(piece, scratch.word_ids.clone());
            scratch.ids.extend(scratch.word_ids.iter().copied());
        }
    }

    fn match_special(&self, text: &str, pos: usize) -> Option<(usize, u32)> {
        let tail = &text[pos..];
        for token in &self.inner.special_tokens_sorted {
            if tail.starts_with(token) {
                if let Some(&id) = self.inner.special_tokens.get(token) {
                    return Some((pos + token.len(), id));
                }
            }
        }
        None
    }

    fn encode_piece_to_ids(&self, piece: &str, out: &mut Vec<u32>) {
        let encoded = self.byte_encode(piece);
        let merged = self.bpe_merge(&encoded);
        for token in merged {
            if let Some(&id) = self.inner.vocab.get(token.as_str()) {
                out.push(id);
                continue;
            }

            let mut recovered = false;
            for ch in token.chars() {
                let one = ch.to_string();
                if let Some(&id) = self.inner.vocab.get(one.as_str()) {
                    out.push(id);
                    recovered = true;
                } else {
                    recovered = false;
                    break;
                }
            }
            if recovered {
                continue;
            }

            if let Some(unk_id) = self.inner.unk_id {
                out.push(unk_id);
            }
        }
    }

    fn byte_encode(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        for &byte in text.as_bytes() {
            out.push(self.inner.byte_encoder[byte as usize]);
        }
        out
    }

    fn merge_rank(&self, left: &str, right: &str) -> Option<usize> {
        self.inner
            .merges
            .get(left)
            .and_then(|inner| inner.get(right))
            .copied()
    }

    fn push_candidate(&self, nodes: &[BpeNode], heap: &mut BinaryHeap<Candidate>, left: usize) {
        let Some(right) = nodes[left].next else {
            return;
        };
        if !nodes[left].alive || !nodes[right].alive {
            return;
        }
        let Some(rank) = self.merge_rank(nodes[left].token.as_str(), nodes[right].token.as_str())
        else {
            return;
        };
        heap.push(Candidate {
            rank,
            left,
            right,
            left_version: nodes[left].version,
            right_version: nodes[right].version,
        });
    }

    fn bpe_merge(&self, encoded: &str) -> Vec<String> {
        let mut nodes: Vec<BpeNode> = encoded
            .chars()
            .enumerate()
            .map(|(idx, ch)| BpeNode {
                token: ch.to_string(),
                prev: idx.checked_sub(1),
                next: None,
                alive: true,
                version: 0,
            })
            .collect();

        if nodes.is_empty() {
            return Vec::new();
        }
        for idx in 0..nodes.len().saturating_sub(1) {
            nodes[idx].next = Some(idx + 1);
        }

        let mut heap = BinaryHeap::new();
        for idx in 0..nodes.len().saturating_sub(1) {
            self.push_candidate(&nodes, &mut heap, idx);
        }

        while let Some(candidate) = heap.pop() {
            if candidate.left >= nodes.len() || candidate.right >= nodes.len() {
                continue;
            }
            let left = &nodes[candidate.left];
            let right = &nodes[candidate.right];
            if !left.alive
                || !right.alive
                || left.next != Some(candidate.right)
                || left.version != candidate.left_version
                || right.version != candidate.right_version
            {
                continue;
            }

            let right_next = nodes[candidate.right].next;
            let right_token = nodes[candidate.right].token.clone();
            nodes[candidate.left].token.push_str(right_token.as_str());
            nodes[candidate.left].version = nodes[candidate.left].version.wrapping_add(1);
            nodes[candidate.left].next = right_next;

            if let Some(next) = right_next {
                nodes[next].prev = Some(candidate.left);
                // NOTE: do NOT bump next.version here — its *token* hasn't changed,
                // only its prev pointer.  Bumping it would invalidate still-valid
                // candidates (next, next.next) already in the priority queue.
            }

            nodes[candidate.right].alive = false;
            nodes[candidate.right].version = nodes[candidate.right].version.wrapping_add(1);

            if let Some(prev) = nodes[candidate.left].prev {
                self.push_candidate(&nodes, &mut heap, prev);
            }
            self.push_candidate(&nodes, &mut heap, candidate.left);
        }

        let mut first = 0usize;
        while first < nodes.len() && !nodes[first].alive {
            first += 1;
        }
        if first >= nodes.len() {
            return Vec::new();
        }

        let mut out = Vec::new();
        let mut current = Some(first);
        while let Some(idx) = current {
            if nodes[idx].alive {
                out.push(nodes[idx].token.clone());
            }
            current = nodes[idx].next;
        }
        out
    }
}

impl Tokenizer for BpeTokenizer {
    fn tokenize(&self, text: &str) -> TokenizedInput {
        let ids = self.tokenize_to_ids(text);
        pad_ids(ids, self.inner.max_seq_len, self.inner.pad_id)
    }

    fn tokenize_batch(&self, texts: &[&str]) -> Vec<TokenizedInput> {
        if texts.is_empty() {
            return Vec::new();
        }
        let mut scratch = TokenizeScratch::default();
        let mut max_len = 0usize;
        let mut all = Vec::with_capacity(texts.len());
        for text in texts {
            let ids = self.tokenize_to_ids_into(text, &mut scratch);
            max_len = max_len.max(ids.len());
            all.push(ids);
        }
        all.into_iter()
            .map(|ids| pad_ids(ids, max_len, self.inner.pad_id))
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.inner.id_to_token.len()
    }

    fn max_seq_len(&self) -> usize {
        self.inner.max_seq_len
    }
}

fn parse_merges_txt(text: &str) -> Vec<(String, String)> {
    let mut merges = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(left) = parts.next() else { continue };
        let Some(right) = parts.next() else { continue };
        merges.push((left.to_string(), right.to_string()));
    }
    merges
}

fn parse_merges_json(value: &JsonValue) -> Result<Vec<(String, String)>, InferenceError> {
    let array = value.as_array().ok_or_else(|| {
        InferenceError::Tokenizer("expected tokenizer.json model.merges array".into())
    })?;
    let mut merges = Vec::with_capacity(array.len());
    for item in array {
        match item {
            JsonValue::String(value) => {
                let mut parts = value.split_whitespace();
                let left = parts.next().ok_or_else(|| {
                    InferenceError::Tokenizer(format!("invalid merge entry {value:?}"))
                })?;
                let right = parts.next().ok_or_else(|| {
                    InferenceError::Tokenizer(format!("invalid merge entry {value:?}"))
                })?;
                merges.push((left.to_string(), right.to_string()));
            }
            JsonValue::Array(items) if items.len() == 2 => {
                let left = items[0].as_str().ok_or_else(|| {
                    InferenceError::Tokenizer("invalid merge left operand".into())
                })?;
                let right = items[1].as_str().ok_or_else(|| {
                    InferenceError::Tokenizer("invalid merge right operand".into())
                })?;
                merges.push((left.to_string(), right.to_string()));
            }
            _ => {
                return Err(InferenceError::Tokenizer(
                    "unsupported tokenizer.json merge entry".into(),
                ));
            }
        }
    }
    Ok(merges)
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SegmentClass {
    Letter,
    Number,
    Other,
}

fn classify_char(ch: char) -> SegmentClass {
    if ch.is_alphabetic() {
        SegmentClass::Letter
    } else if ch.is_numeric() {
        SegmentClass::Number
    } else {
        SegmentClass::Other
    }
}

fn split_whitespace_run(run: &str) -> (&str, &str) {
    if run.is_empty() {
        return ("", "");
    }
    let mut last_start = 0usize;
    for (idx, _) in run.char_indices() {
        last_start = idx;
    }
    (&run[..last_start], &run[last_start..])
}

fn byte_level_pretokenize(text: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut pos = 0usize;

    while pos < text.len() {
        let ws_start = pos;
        while pos < text.len() {
            let ch = text[pos..]
                .chars()
                .next()
                .expect("invariant: pos is inside non-empty UTF-8 text");
            if !ch.is_whitespace() {
                break;
            }
            pos += ch.len_utf8();
        }

        if pos >= text.len() {
            if ws_start < pos {
                pieces.push(text[ws_start..pos].to_string());
            }
            break;
        }

        let ws_run = &text[ws_start..pos];
        let (standalone_ws, attached_ws) = split_whitespace_run(ws_run);
        if !standalone_ws.is_empty() {
            pieces.push(standalone_ws.to_string());
        }

        let mut segment = String::new();
        segment.push_str(attached_ws);

        let current = text[pos..]
            .chars()
            .next()
            .expect("invariant: pos is inside non-empty UTF-8 text");
        let class = if current == '\'' {
            let next_pos = pos + current.len_utf8();
            if next_pos < text.len() {
                classify_char(
                    text[next_pos..]
                        .chars()
                        .next()
                        .expect("invariant: next_pos is inside non-empty UTF-8 text"),
                )
            } else {
                SegmentClass::Other
            }
        } else {
            classify_char(current)
        };

        if current == '\'' && class != SegmentClass::Other {
            segment.push(current);
            pos += current.len_utf8();
        }

        while pos < text.len() {
            let ch = text[pos..]
                .chars()
                .next()
                .expect("invariant: pos is inside non-empty UTF-8 text");
            if ch.is_whitespace() {
                break;
            }
            let ch_class = classify_char(ch);
            if ch_class != class {
                break;
            }
            segment.push(ch);
            pos += ch.len_utf8();
        }

        if segment.is_empty() {
            let ch = text[pos..]
                .chars()
                .next()
                .expect("invariant: pos is inside non-empty UTF-8 text");
            segment.push(ch);
            pos += ch.len_utf8();
            while pos < text.len() {
                let next = text[pos..]
                    .chars()
                    .next()
                    .expect("invariant: pos is inside non-empty UTF-8 text");
                if next.is_whitespace() || classify_char(next) != SegmentClass::Other {
                    break;
                }
                segment.push(next);
                pos += next.len_utf8();
            }
        }

        pieces.push(segment);
    }

    pieces
}

fn bytes_to_unicode() -> Vec<char> {
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

/// **Unstable**: byte-decode helper; encoding table and fallback behavior may change.
///
/// Decode a byte-encoded BPE token string back to UTF-8.
pub fn byte_decode_token(token_str: &str) -> String {
    let table = bytes_to_unicode();
    let mut decoder = std::collections::HashMap::new();
    for (byte_val, &ch) in table.iter().enumerate() {
        decoder.insert(ch, byte_val as u8);
    }
    let mut bytes = Vec::new();
    for ch in token_str.chars() {
        if let Some(&b) = decoder.get(&ch) {
            bytes.push(b);
        }
    }
    String::from_utf8_lossy(&bytes).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_bpe() -> BpeTokenizer {
        let mut vocab = HashMap::new();
        vocab.insert("h".to_string(), 0);
        vocab.insert("e".to_string(), 1);
        vocab.insert("l".to_string(), 2);
        vocab.insert("o".to_string(), 3);
        vocab.insert("Ġ".to_string(), 4);
        vocab.insert("w".to_string(), 5);
        vocab.insert("r".to_string(), 6);
        vocab.insert("d".to_string(), 7);
        vocab.insert("he".to_string(), 8);
        vocab.insert("hel".to_string(), 9);
        vocab.insert("hell".to_string(), 10);
        vocab.insert("hello".to_string(), 11);
        vocab.insert("Ġw".to_string(), 12);
        vocab.insert("Ġwo".to_string(), 13);
        vocab.insert("Ġwor".to_string(), 14);
        vocab.insert("Ġworl".to_string(), 15);
        vocab.insert("Ġworld".to_string(), 16);
        vocab.insert("<|endoftext|>".to_string(), 17);

        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("he".to_string(), "l".to_string()),
            ("hel".to_string(), "l".to_string()),
            ("hell".to_string(), "o".to_string()),
            ("Ġ".to_string(), "w".to_string()),
            ("Ġw".to_string(), "o".to_string()),
            ("Ġwo".to_string(), "r".to_string()),
            ("Ġwor".to_string(), "l".to_string()),
            ("Ġworl".to_string(), "d".to_string()),
        ];
        BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap()
    }

    #[test]
    fn test_byte_pretokenize_preserves_prefix_space() {
        let pieces = byte_level_pretokenize("hello world");
        assert_eq!(pieces, vec!["hello", " world"]);
    }

    #[test]
    fn test_bpe_merge_hello_world() {
        let tokenizer = synthetic_bpe();
        let ids = tokenizer.tokenize_to_ids("hello world");
        assert_eq!(ids, vec![11, 16]);
    }

    #[test]
    fn test_bpe_special_token_passthrough() {
        let tokenizer = synthetic_bpe();
        let ids = tokenizer.tokenize_to_ids("hello<|endoftext|>world");
        assert_eq!(ids, vec![11, 17, 5, 3, 6, 2, 7]);
    }

    #[test]
    fn test_bpe_truncation_preserves_eos() {
        let tokenizer = synthetic_bpe().with_add_eos().with_max_seq_len(2);
        let ids = tokenizer.tokenize_to_ids("hello world");
        assert_eq!(ids, vec![11, 17]);
    }

    #[test]
    fn test_bpe_tokenize_batch_pads_to_batch_max() {
        let tokenizer = synthetic_bpe();
        let batch = tokenizer.tokenize_batch(&["hello", "hello world"]);
        assert_eq!(batch[0].input_ids.len(), 2);
        assert_eq!(batch[1].input_ids.len(), 2);
        assert_eq!(batch[0].attention_mask, vec![1, 0]);
        assert_eq!(batch[1].attention_mask, vec![1, 1]);
    }
}
