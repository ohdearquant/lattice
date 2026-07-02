//! Byte-level BPE tokenizer.
//!
//! This module implements a dependency-free GPT-2/Qwen-style tokenizer with
//! byte-level pre-tokenization, merge-rank driven BPE, a per-word LRU cache,
//! and `tokenizer.json` / `vocab.json` + `merges.txt` loading.

use crate::error::InferenceError;
use crate::tokenizer::common::{
    JsonValue, ThreadSafeLruCache, TokenizedInput, Tokenizer, invert_vocab, json_object_to_vocab,
    json_path, known_special_id, pad_ids, parse_added_tokens, parse_json,
    parse_post_processor_flags, parse_rendered_added_tokens, push_eos_preserving_limit,
    vocab_txt_to_map,
};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tracing::warn;

const DEFAULT_BPE_CACHE_CAPACITY: usize = 8_192;
const DEFAULT_BPE_MAX_SEQ_LEN: usize = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreTokenizeMode {
    Gpt4Regex,
}

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
    /// Decode-side map for added tokens with `special=false` (`<think>`/`</think>`,
    /// `<tool_call>`, FIM markers). These ids are absent from the base BPE `vocab`,
    /// so `token_for_id` falls back to this map to render them as literal text
    /// instead of silently dropping them. `special=true` markers are deliberately
    /// excluded so they stay swallowed (matching `skip_special_tokens=True`).
    added_render: HashMap<u32, String>,
    pad_id: u32,
    unk_id: Option<u32>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    pre_tokenize_mode: PreTokenizeMode,
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
        let rendered_added = parse_rendered_added_tokens(&root);

        let mut tokenizer = Self::from_vocab_and_merges_with_config(
            vocab,
            merges,
            added,
            rendered_added,
            DEFAULT_BPE_CACHE_CAPACITY,
            DEFAULT_BPE_MAX_SEQ_LEN,
        )?;

        if let Some(unk_token) =
            json_path(&root, &["model", "unk_token"]).and_then(JsonValue::as_str)
        {
            tokenizer = tokenizer.with_unk_token(unk_token);
        }

        let pp = parse_post_processor_flags(&root);
        if pp.add_eos {
            tokenizer = tokenizer.with_add_eos();
            if let Some(eos_id) = pp.eos_id {
                tokenizer = tokenizer.with_eos_id(eos_id);
            }
        }

        validate_bpe_pretokenizer(&root)?;
        if detect_gpt4_regex_pretokenizer(&root) {
            tokenizer = tokenizer.with_pre_tokenize_mode(PreTokenizeMode::Gpt4Regex);
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
            HashMap::new(),
            DEFAULT_BPE_CACHE_CAPACITY,
            DEFAULT_BPE_MAX_SEQ_LEN,
        )
    }

    fn from_vocab_and_merges_with_config(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
        added_tokens: HashMap<String, u32>,
        rendered_added: HashMap<String, u32>,
        cache_capacity: usize,
        max_seq_len: usize,
    ) -> Result<Self, InferenceError> {
        let id_to_token = invert_vocab(&vocab)?;
        // Invert the renderable added-token set to id -> content for decode-side
        // lookup. Built once at construction; consulted by `token_for_id` only
        // when the base table misses (added-token ids exceed the base vocab range).
        let added_render: HashMap<u32, String> = rendered_added
            .into_iter()
            .map(|(content, id)| (id, content))
            .collect();
        let mut merge_ranks: HashMap<String, HashMap<String, usize>> = HashMap::new();
        for (rank, (left, right)) in merges.into_iter().enumerate() {
            // First occurrence defines the rank: merges are listed in priority
            // order, so a duplicate pair must not demote it to a later rank.
            merge_ranks
                .entry(left)
                .or_default()
                .entry(right)
                .or_insert(rank);
        }

        let mut special_tokens = added_tokens;
        // A zero-length special token would match at every position
        // (`"".starts_with(_)` is always true), so match_special returns the
        // same `pos` and tokenize() never advances — an infinite loop with
        // unbounded output. A malformed tokenizer.json (an added_token whose
        // "content" is empty) must not be able to wedge the engine: drop
        // zero-length specials at construction.
        special_tokens.retain(|name, _| !name.is_empty());
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
            added_render,
            pad_id,
            unk_id,
            bos_id,
            eos_id,
            add_bos: false,
            add_eos: false,
            // #330: default to the GPT-2 regex pretokenizer, not the naive
            // `byte_level_pretokenize` heuristic. Every real GPT-2-format
            // tokenizer HF ships (raw vocab.json+merges.txt with no
            // tokenizer.json, or a tokenizer.json declaring
            // `"pre_tokenizer": {"type": "ByteLevel"}`) uses the standard
            // GPT-2 regex for word-boundary splitting: it's what the "slow"
            // Python `GPT2Tokenizer` implements directly, and it's what the
            // fast `ByteLevel` pre_tokenizer's `use_regex` flag defaults to
            // (`true`) when absent, which is the case for both `gpt2` and
            // `roberta-base` on the Hub. `from_tokenizer_json_str` below can
            // still detect an explicit regex `Split`/`Sequence` pre_tokenizer
            // and set this mode explicitly, but when there is no
            // pre_tokenizer metadata to inspect at all — the case this
            // default governs — GPT-2 regex is the correct assumption, not
            // the hand-rolled fallback.
            pre_tokenize_mode: PreTokenizeMode::Gpt4Regex,
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
            added_render: self.inner.added_render.clone(),
            pad_id: self.inner.pad_id,
            unk_id: self.inner.unk_id,
            bos_id: self.inner.bos_id,
            eos_id: self.inner.eos_id,
            add_bos: self.inner.add_bos,
            add_eos: self.inner.add_eos,
            pre_tokenize_mode: self.inner.pre_tokenize_mode,
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
            added_render: self.inner.added_render.clone(),
            pad_id: self.inner.pad_id,
            unk_id,
            bos_id: self.inner.bos_id,
            eos_id: self.inner.eos_id,
            add_bos: self.inner.add_bos,
            add_eos: self.inner.add_eos,
            pre_tokenize_mode: self.inner.pre_tokenize_mode,
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
            added_render: self.inner.added_render.clone(),
            pad_id: self.inner.pad_id,
            unk_id: self.inner.unk_id,
            bos_id: self.inner.bos_id,
            eos_id: self.inner.eos_id,
            add_bos: self.inner.add_bos,
            add_eos: true,
            pre_tokenize_mode: self.inner.pre_tokenize_mode,
            max_seq_len: self.inner.max_seq_len,
            cache: self.inner.cache.clone(),
        };
        Self {
            inner: Arc::new(inner),
        }
    }

    fn with_eos_id(self, eos_id: u32) -> Self {
        let inner = BpeInner {
            vocab: self.inner.vocab.clone(),
            id_to_token: self.inner.id_to_token.clone(),
            merges: self.inner.merges.clone(),
            byte_encoder: self.inner.byte_encoder.clone(),
            special_tokens: self.inner.special_tokens.clone(),
            special_tokens_sorted: self.inner.special_tokens_sorted.clone(),
            added_render: self.inner.added_render.clone(),
            pad_id: self.inner.pad_id,
            unk_id: self.inner.unk_id,
            bos_id: self.inner.bos_id,
            eos_id: Some(eos_id),
            add_bos: self.inner.add_bos,
            add_eos: self.inner.add_eos,
            pre_tokenize_mode: self.inner.pre_tokenize_mode,
            max_seq_len: self.inner.max_seq_len,
            cache: self.inner.cache.clone(),
        };
        Self {
            inner: Arc::new(inner),
        }
    }

    fn with_pre_tokenize_mode(self, mode: PreTokenizeMode) -> Self {
        let inner = BpeInner {
            vocab: self.inner.vocab.clone(),
            id_to_token: self.inner.id_to_token.clone(),
            merges: self.inner.merges.clone(),
            byte_encoder: self.inner.byte_encoder.clone(),
            special_tokens: self.inner.special_tokens.clone(),
            special_tokens_sorted: self.inner.special_tokens_sorted.clone(),
            added_render: self.inner.added_render.clone(),
            pad_id: self.inner.pad_id,
            unk_id: self.inner.unk_id,
            bos_id: self.inner.bos_id,
            eos_id: self.inner.eos_id,
            add_bos: self.inner.add_bos,
            add_eos: self.inner.add_eos,
            pre_tokenize_mode: mode,
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
        // Base vocab first; added tokens with `special=false` (think/tool/FIM
        // markers) live beyond the base range, so fall back to `added_render` to
        // render them as text instead of dropping them. `special=true` markers
        // are absent from both maps and stay swallowed.
        self.inner
            .id_to_token
            .get(id as usize)
            .map(String::as_str)
            .or_else(|| self.inner.added_render.get(&id).map(String::as_str))
    }

    /// **Unstable**: look up special token ID by name; set of known special tokens may change.
    ///
    /// Look up special token ID by name (e.g. "<|im_end|>").
    pub fn special_token_id(&self, name: &str) -> Option<u32> {
        self.inner.special_tokens.get(name).copied()
    }

    /// **Unstable**: return the byte representation of every token in the vocabulary.
    ///
    /// `vocab_bytes()[i]` is the UTF-8 byte sequence that token `i` decodes to,
    /// after applying the GPT-2 byte-level encoding reversal.
    ///
    /// Used by [`GrammarEngine::new`](crate::grammar::GrammarEngine::new) to build
    /// the precomputed vocabulary partition for grammar-constrained decoding (ADR-046).
    ///
    /// Cost: O(vocab_size) — called once at engine initialisation.
    pub fn vocab_bytes(&self) -> Vec<Vec<u8>> {
        self.inner
            .id_to_token
            .iter()
            .map(|token_str| {
                // Apply GPT-2 byte-level encoding reversal (same as byte_decode_token).
                let decoded = byte_decode_token(token_str);
                decoded.into_bytes()
            })
            .collect()
    }

    fn tokenize_to_ids(&self, text: &str) -> Vec<u32> {
        let mut scratch = TokenizeScratch::default();
        self.tokenize_to_ids_into(text, &mut scratch)
    }

    fn tokenize_to_ids_into(&self, text: &str, scratch: &mut TokenizeScratch) -> Vec<u32> {
        scratch.ids.clear();
        if self.inner.add_bos
            && let Some(bos_id) = self.inner.bos_id
        {
            scratch.ids.push(bos_id);
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
        let pieces = match self.inner.pre_tokenize_mode {
            PreTokenizeMode::Gpt4Regex => gpt4_regex_pretokenize(text),
        };
        for piece in pieces {
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
            if tail.starts_with(token)
                && let Some(&id) = self.inner.special_tokens.get(token)
            {
                return Some((pos + token.len(), id));
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

            // Byte-fallback: decompose the unmatched token into single
            // characters. Commit only if EVERY character resolves — a partial
            // match must roll back, otherwise the resolved ids are orphaned
            // alongside the <unk> pushed below.
            let fallback_start = out.len();
            let mut recovered = true;
            for ch in token.chars() {
                let one = ch.to_string();
                if let Some(&id) = self.inner.vocab.get(one.as_str()) {
                    out.push(id);
                } else {
                    recovered = false;
                    break;
                }
            }
            if recovered {
                continue;
            }
            out.truncate(fallback_start);

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

    fn decode(&self, ids: &[u32]) -> Option<String> {
        let encoded: String = ids.iter().filter_map(|&id| self.token_for_id(id)).collect();
        Some(byte_decode_token(&encoded))
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

fn detect_gpt4_regex_pretokenizer(root: &JsonValue) -> bool {
    let Some(pt) = root.get("pre_tokenizer") else {
        return false;
    };
    has_regex_split(pt)
}

fn has_regex_split(pt: &JsonValue) -> bool {
    let pt_type = pt.get("type").and_then(JsonValue::as_str).unwrap_or("");
    match pt_type {
        "Split" => pt.get("pattern").and_then(|p| p.get("Regex")).is_some(),
        "Sequence" => pt
            .get("pretokenizers")
            .and_then(JsonValue::as_array)
            .is_some_and(|arr| arr.iter().any(has_regex_split)),
        _ => false,
    }
}

/// Fail closed on explicit `tokenizer.json` `pre_tokenizer` metadata this
/// parser cannot honestly model (#330 codex round-2 finding 1).
///
/// `from_vocab_and_merges_with_config` always defaults to `Gpt4Regex`
/// pre-tokenization, and `detect_gpt4_regex_pretokenizer` above only ever
/// *confirms* that default from a recognized regex `Split`/`Sequence`. That
/// combination means an explicit-but-unrecognized `pre_tokenizer` silently
/// falls through to the `Gpt4Regex` default rather than erroring — including
/// `ByteLevel(use_regex=false)`, whose whole point (per HF's
/// `tokenizers::pre_tokenizers::byte_level::ByteLevel`) is to *skip* regex
/// splitting entirely. Splitting anyway produces silently wrong token ids
/// (a merge that should fire across a former word boundary cannot).
///
/// Accept/reject matrix:
/// - `pre_tokenizer` absent -> accept (the #330 raw-vocab default; no
///   metadata to contradict `Gpt4Regex`).
/// - Top-level recognized regex `Split` -> accept.
/// - `Sequence` -> accept only if EVERY child is individually supported,
///   walked left-to-right: a child is supported if it is itself a
///   recognized regex `Split`, OR a bare `ByteLevel` with `use_regex`
///   absent/`true`, OR a `ByteLevel{use_regex:false}` that is preceded
///   (earlier in the same sequence) by a supported regex `Split` — this is
///   the real Qwen/Qwen3-Embedding shape,
///   `Sequence[Split{pattern.Regex}, ByteLevel{use_regex:false}]`, because
///   the leading `Split` already establishes the correct segmentation and
///   the trailing `ByteLevel(use_regex:false)` is just "don't split again".
///   Any other/unsupported sibling (e.g. `Metaspace`, `Whitespace`) rejects
///   the whole sequence even when another child is a regex `Split` — a
///   `Split` does not honestly dominate later pre-tokenizers (codex round-3,
///   #330 residual finding).
/// - Bare top-level `ByteLevel` with `use_regex` absent or `true` -> accept
///   (HF's default `use_regex=true` is exactly what `Gpt4Regex`
///   approximates; this is the common gpt2/roberta Hub shape).
/// - Bare top-level `ByteLevel` with explicit `use_regex:false` -> reject
///   (no preceding `Split` is possible at the top level to establish
///   segmentation). `Gpt4Regex` would silently apply a split the tokenizer
///   author explicitly disabled.
/// - Any other/unknown `pre_tokenizer` type (`Metaspace`, `Whitespace`,
///   `BertPreTokenizer`, ...) on a BPE model -> reject. This parser has no
///   implementation for it; falling through to `Gpt4Regex` would silently
///   mistokenize.
fn validate_bpe_pretokenizer(root: &JsonValue) -> Result<(), InferenceError> {
    let Some(pt) = root.get("pre_tokenizer") else {
        return Ok(());
    };
    if is_supported_bpe_pretokenizer(pt) {
        return Ok(());
    }
    Err(InferenceError::Tokenizer(format!(
        "tokenizer.json declares an unsupported BPE pre_tokenizer \
         configuration ({}); refusing to silently fall back to the GPT-2 \
         regex default, which could split text the declared pre_tokenizer \
         would not",
        describe_pretokenizer(pt)
    )))
}

/// Whether a single `Split`/`ByteLevel` node (never a `Sequence`) is a
/// recognized regex-establishing split on its own, with no notion of
/// "preceded by". Used as the base case for both the top-level check and
/// the `Sequence` child walk below.
fn is_regex_split_node(pt: &JsonValue) -> bool {
    let pt_type = pt.get("type").and_then(JsonValue::as_str).unwrap_or("");
    pt_type == "Split" && pt.get("pattern").and_then(|p| p.get("Regex")).is_some()
}

/// Whether `pt` is one of the `pre_tokenizer` shapes this parser can
/// honestly honor for a BPE model. See `validate_bpe_pretokenizer` for the
/// full accept/reject matrix and rationale.
///
/// Deliberately does NOT use `has_regex_split`'s `any()`-over-descendants
/// shortcut here: a regex `Split` earlier in a `Sequence` does not make an
/// unrelated, unsupported sibling (e.g. `Metaspace`) honest to run through
/// `Gpt4Regex` (codex round-3, #330 residual finding).
fn is_supported_bpe_pretokenizer(pt: &JsonValue) -> bool {
    if is_regex_split_node(pt) {
        return true;
    }
    let pt_type = pt.get("type").and_then(JsonValue::as_str).unwrap_or("");
    match pt_type {
        "ByteLevel" => pt
            .get("use_regex")
            .and_then(JsonValue::as_bool)
            .unwrap_or(true),
        "Sequence" => pt
            .get("pretokenizers")
            .and_then(JsonValue::as_array)
            .is_some_and(|arr| {
                let mut seen_regex_split = false;
                for child in arr {
                    if is_regex_split_node(child) {
                        seen_regex_split = true;
                        continue;
                    }
                    let child_type = child.get("type").and_then(JsonValue::as_str).unwrap_or("");
                    let child_ok = if child_type == "ByteLevel" {
                        let use_regex = child
                            .get("use_regex")
                            .and_then(JsonValue::as_bool)
                            .unwrap_or(true);
                        use_regex || seen_regex_split
                    } else {
                        false
                    };
                    if !child_ok {
                        return false;
                    }
                }
                true
            }),
        _ => false,
    }
}

fn describe_pretokenizer(pt: &JsonValue) -> String {
    let pt_type = pt
        .get("type")
        .and_then(JsonValue::as_str)
        .unwrap_or("<untyped>");
    if pt_type == "ByteLevel" {
        let use_regex = pt
            .get("use_regex")
            .and_then(JsonValue::as_bool)
            .unwrap_or(true);
        return format!("ByteLevel{{use_regex:{use_regex}}}");
    }
    pt_type.to_string()
}

fn gpt4_regex_pretokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut pieces = Vec::new();
    let mut pos = 0;

    while pos < chars.len() {
        if let Some(end) = try_contraction(&chars, pos) {
            pieces.push(chars[pos..end].iter().collect());
            pos = end;
        } else if let Some(end) = try_prefix_letters(&chars, pos) {
            pieces.push(chars[pos..end].iter().collect());
            pos = end;
        } else if chars[pos].is_numeric() {
            pieces.push(chars[pos].to_string());
            pos += 1;
        } else if let Some(end) = try_punctuation_run(&chars, pos) {
            pieces.push(chars[pos..end].iter().collect());
            pos = end;
        } else if let Some(end) = try_newline_run(&chars, pos) {
            pieces.push(chars[pos..end].iter().collect());
            pos = end;
        } else if let Some(end) = try_trailing_ws(&chars, pos) {
            pieces.push(chars[pos..end].iter().collect());
            pos = end;
        } else if chars[pos].is_whitespace() {
            let start = pos;
            while pos < chars.len() && chars[pos].is_whitespace() {
                pos += 1;
            }
            pieces.push(chars[start..pos].iter().collect());
        } else {
            pieces.push(chars[pos].to_string());
            pos += 1;
        }
    }

    pieces
}

fn eq_ci(a: char, lower: char) -> bool {
    a.to_ascii_lowercase() == lower
}

/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)`
fn try_contraction(chars: &[char], pos: usize) -> Option<usize> {
    if chars.get(pos).copied() != Some('\'') {
        return None;
    }
    let rest = &chars[pos + 1..];
    if rest.len() >= 2 && eq_ci(rest[0], 'l') && eq_ci(rest[1], 'l') {
        return Some(pos + 3);
    }
    if rest.len() >= 2 && eq_ci(rest[0], 'r') && eq_ci(rest[1], 'e') {
        return Some(pos + 3);
    }
    if rest.len() >= 2 && eq_ci(rest[0], 'v') && eq_ci(rest[1], 'e') {
        return Some(pos + 3);
    }
    if !rest.is_empty() {
        let c = rest[0].to_ascii_lowercase();
        if matches!(c, 's' | 't' | 'm' | 'd') {
            return Some(pos + 2);
        }
    }
    None
}

/// `[^\r\n\p{L}\p{N}]?\p{L}+`
fn try_prefix_letters(chars: &[char], pos: usize) -> Option<usize> {
    let mut i = pos;
    if i < chars.len()
        && !chars[i].is_alphabetic()
        && !chars[i].is_numeric()
        && chars[i] != '\r'
        && chars[i] != '\n'
    {
        i += 1;
    }
    let start = i;
    while i < chars.len() && chars[i].is_alphabetic() {
        i += 1;
    }
    if i > start { Some(i) } else { None }
}

/// ` ?[^\s\p{L}\p{N}]+[\r\n]*`
fn try_punctuation_run(chars: &[char], pos: usize) -> Option<usize> {
    let mut i = pos;
    if i < chars.len() && chars[i] == ' ' {
        i += 1;
    }
    let start = i;
    while i < chars.len()
        && !chars[i].is_whitespace()
        && !chars[i].is_alphabetic()
        && !chars[i].is_numeric()
    {
        i += 1;
    }
    if i == start {
        return None;
    }
    while i < chars.len() && (chars[i] == '\r' || chars[i] == '\n') {
        i += 1;
    }
    Some(i)
}

/// `\s*[\r\n]+`
fn try_newline_run(chars: &[char], pos: usize) -> Option<usize> {
    let mut i = pos;
    while i < chars.len() && chars[i].is_whitespace() && chars[i] != '\r' && chars[i] != '\n' {
        i += 1;
    }
    let nl_start = i;
    while i < chars.len() && (chars[i] == '\r' || chars[i] == '\n') {
        i += 1;
    }
    if i > nl_start { Some(i) } else { None }
}

/// `\s+(?!\S)` — whitespace not followed by non-whitespace.
///
/// Implements greedy-with-backtracking semantics: match as many whitespace chars
/// as possible such that the char immediately after is NOT `\S` (non-whitespace).
/// When a whitespace run precedes a non-whitespace char, the longest valid match
/// is the run minus the last char (whose lookahead sees the following whitespace).
fn try_trailing_ws(chars: &[char], pos: usize) -> Option<usize> {
    if !chars.get(pos).is_some_and(|c| c.is_whitespace()) {
        return None;
    }
    let mut i = pos;
    while i < chars.len() && chars[i].is_whitespace() {
        i += 1;
    }
    if i == chars.len() {
        Some(i)
    } else if i > pos + 1 {
        // Whitespace run of ≥2 before a non-whitespace: backtrack one position.
        // The resulting match ends at i-1, whose lookahead char (at i-1) is whitespace.
        Some(i - 1)
    } else {
        // Single space before non-whitespace: (?!\S) fails even after backtracking.
        None
    }
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
    fn test_gpt4_regex_pretokenize_matches_hf_bytelevel_use_regex_true() {
        // Golden pieces from HF `tokenizers.pre_tokenizers.ByteLevel(use_regex=True)`
        // (the real, unconditional default for every GPT-2-format tokenizer HF
        // ships — see audit_tokenizer_parity.rs::gpt2_raw_vocab_bytelevel_fallback_parity
        // for the exact command). HF's actual ByteLevel(use_regex=True) regex is
        // the original GPT-2 pattern `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
        // ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`; `gpt4_regex_pretokenize` is a
        // Qwen/tiktoken-flavored superset (its prefix-letters branch can also
        // admit leading punctuation), not a literal reimplementation of that
        // pattern. The cases below are only claimed to match HF ByteLevel/GPT-2
        // behavior for the exact inputs tested, not as a proof the two regexes
        // are equivalent in general.
        assert_eq!(
            gpt4_regex_pretokenize("hello\nworld"),
            vec!["hello", "\n", "world"]
        );
        assert_eq!(gpt4_regex_pretokenize("a'st"), vec!["a", "'s", "t"]);
    }

    #[test]
    fn test_gpt2_raw_vocab_defaults_to_gpt4_regex_pretokenize() {
        // Regression for #330: `from_vocab_and_merges` (what `from_files` and
        // `load_tokenizer`'s file-based-probing fallback use for a raw
        // GPT-2-format vocab.json + merges.txt with no tokenizer.json to read
        // pre_tokenizer metadata from) must default to `Gpt4Regex`
        // pretokenization, not the naive hand-rolled fallback that used to
        // exist — see `test_gpt4_regex_pretokenize_matches_hf_bytelevel_use_regex_true`
        // above for the HF-verified golden pieces this default now produces.
        //
        // This is a merge-boundary discriminator: a rule merging 's'+'t' into
        // "st" can only fire if both chars land in the SAME pretokenize piece
        // (BPE merges never span piece boundaries). Under the correct regex
        // split, "a'st" -> ["a", "'s", "t"]: 's' and 't' are in DIFFERENT
        // pieces ("'s" and "t"), so the merge cannot fire and every char
        // resolves to its own id. Under the buggy naive-fallback split,
        // "a'st" -> ["a", "'st"]: 's' and 't' are adjacent within the same
        // piece, so the merge DOES fire, producing a 3-id sequence instead of
        // 4. This mirrors the real gpt2 vocab divergence verified empirically
        // in audit_tokenizer_parity.rs (pre-fix: [64, 6, 301], HF: [64, 338, 83]).
        let mut vocab = HashMap::new();
        for (s, i) in [("a", 0u32), ("'", 1), ("s", 2), ("t", 3), ("st", 4)] {
            vocab.insert(s.to_string(), i);
        }
        let merges = vec![("s".to_string(), "t".to_string())];
        let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();
        assert_eq!(tokenizer.tokenize_to_ids("a'st"), vec![0, 1, 2, 3]);
    }

    /// Vocab/merges for the cross-space-merge discriminator used by the
    /// `validate_bpe_pretokenizer` fail-closed tests below (codex round-2,
    /// #330 finding 1). Byte-level maps a literal space to `Ġ` (U+0120), so
    /// `"a b"` byte-encodes to `"aĠb"`. Base single-byte tokens (`a`, `Ġ`,
    /// `b`) plus staged merges `Ġ`+`b` -> `Ġb` -> (with `a`) `aĠb` let a
    /// SINGLE merged id (2) only be reachable if "a" and " b" land in the
    /// SAME pre-tokenize piece — i.e. no regex split at all. `Gpt4Regex`
    /// always splits on word boundaries, so "a" and " b" land in different
    /// pieces and the cross-space merge can never fire, giving `[0, 1]`
    /// (the "a" and "Ġb" ids from separate pieces) instead. This is a
    /// direct id-level proxy for the silent-wrong-ids failure mode codex's
    /// review flagged for explicit `use_regex:false` metadata that fell
    /// through to the `Gpt4Regex` default unchecked.
    fn cross_space_merge_json(pre_tokenizer: &str) -> String {
        format!(
            r#"{{
                "model": {{
                    "type": "BPE",
                    "vocab": {{"a": 0, "Ġb": 1, "aĠb": 2, "Ġ": 3, "b": 4}},
                    "merges": ["Ġ b", "a Ġb"]
                }},
                "pre_tokenizer": {pre_tokenizer}
            }}"#
        )
    }

    #[test]
    fn test_bytelevel_use_regex_false_rejected_without_supporting_split() {
        // codex round-2 (#330 finding 1, major/fail-open): a tokenizer.json
        // that explicitly declares `ByteLevel(use_regex:false)` with no
        // preceding regex `Split` must be REJECTED, not silently tokenized
        // with the `Gpt4Regex` default. HF's ByteLevel `use_regex=false`
        // means "no regex split at all" — falling back to `Gpt4Regex` would
        // silently apply a split the tokenizer author explicitly disabled,
        // producing wrong ids (see `cross_space_merge_json` doc comment).
        let json = cross_space_merge_json(r#"{"type": "ByteLevel", "use_regex": false}"#);
        let err = BpeTokenizer::from_tokenizer_json_str(&json)
            .expect_err("ByteLevel(use_regex:false) without a supporting Split must fail closed");
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported BPE pre_tokenizer"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_unknown_pretokenizer_type_rejected() {
        // codex round-2 (#330 finding 1): an unrecognized pre_tokenizer type
        // (Metaspace, Whitespace, BertPreTokenizer, ...) on a BPE model must
        // also fail closed rather than silently defaulting to `Gpt4Regex`.
        let json = cross_space_merge_json(r#"{"type": "Metaspace", "replacement": "_"}"#);
        let err = BpeTokenizer::from_tokenizer_json_str(&json)
            .expect_err("unknown pre_tokenizer type must fail closed");
        assert!(
            err.to_string().contains("unsupported BPE pre_tokenizer"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn test_bare_bytelevel_use_regex_true_or_absent_accepted() {
        // codex round-2 (#330 finding 1): bare `ByteLevel` with `use_regex`
        // absent or `true` is the common gpt2/roberta Hub shape and must
        // keep working exactly as the #330 fix intended — HF's own default
        // for `use_regex` is `true`, which is what `Gpt4Regex` approximates.
        for pre_tokenizer in [
            r#"{"type": "ByteLevel"}"#,
            r#"{"type": "ByteLevel", "use_regex": true}"#,
        ] {
            let json = cross_space_merge_json(pre_tokenizer);
            let tokenizer = BpeTokenizer::from_tokenizer_json_str(&json)
                .unwrap_or_else(|e| panic!("bare ByteLevel must be accepted: {e}"));
            // Gpt4Regex splits "a b" into ["a", " b"]: the cross-space merge
            // cannot fire, so it round-trips to two ids, not the merged one.
            assert_eq!(tokenizer.tokenize_to_ids("a b"), vec![0, 1]);
        }
    }

    #[test]
    fn test_sequence_split_then_bytelevel_use_regex_false_accepted() {
        // codex round-2 (#330 finding 1): the real Qwen/Qwen3-Embedding
        // shape is `Sequence[Split{pattern.Regex}, ByteLevel{use_regex:false}]`
        // — a regex Split establishes the correct segmentation, and the
        // trailing `ByteLevel(use_regex:false)` just means "don't split
        // again". This must still be accepted and still dispatch to
        // `Gpt4Regex` (the only implemented mode), matching pre-fix #519
        // behavior for this specific shape.
        let pre_tokenizer = r#"{
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated"},
                {"type": "ByteLevel", "use_regex": false}
            ]
        }"#;
        let json = cross_space_merge_json(pre_tokenizer);
        let tokenizer = BpeTokenizer::from_tokenizer_json_str(&json)
            .unwrap_or_else(|e| panic!("Sequence[Split, ByteLevel(false)] must be accepted: {e}"));
        assert_eq!(tokenizer.tokenize_to_ids("a b"), vec![0, 1]);
    }

    #[test]
    fn test_sequence_split_then_metaspace_rejected() {
        // codex round-3 (#330 residual finding): before this fix,
        // `is_supported_bpe_pretokenizer` returned `true` as soon as
        // `has_regex_split` found ANY descendant regex `Split`, so
        // `Sequence[Split{pattern.Regex}, Metaspace]` was wrongly accepted
        // — the unsupported `Metaspace` sibling never got a chance to reject
        // the sequence. A `Split` earlier in the sequence does not make an
        // unrelated, unimplemented sibling honest to run through
        // `Gpt4Regex`: lattice would silently tokenize with GPT-2 byte
        // encoding while HF applies `Metaspace`, producing a completely
        // different token alphabet (wrong ids, not a subtle drift).
        let pre_tokenizer = r#"{
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated"},
                {"type": "Metaspace", "replacement": "_"}
            ]
        }"#;
        let json = cross_space_merge_json(pre_tokenizer);
        let err = BpeTokenizer::from_tokenizer_json_str(&json).expect_err(
            "Sequence[Split, Metaspace] must fail closed even though Split is regex-based",
        );
        assert!(
            err.to_string().contains("unsupported BPE pre_tokenizer"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn test_sequence_split_then_whitespace_rejected() {
        // Same class as `test_sequence_split_then_metaspace_rejected` above,
        // with `Whitespace` (also named explicitly in codex round-2's
        // reject list) as the unsupported sibling instead of `Metaspace`.
        let pre_tokenizer = r#"{
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "\\s+"}, "behavior": "Isolated"},
                {"type": "Whitespace"}
            ]
        }"#;
        let json = cross_space_merge_json(pre_tokenizer);
        let err = BpeTokenizer::from_tokenizer_json_str(&json).expect_err(
            "Sequence[Split, Whitespace] must fail closed even though Split is regex-based",
        );
        assert!(
            err.to_string().contains("unsupported BPE pre_tokenizer"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn test_bpe_merge_hello_world() {
        let tokenizer = synthetic_bpe();
        let ids = tokenizer.tokenize_to_ids("hello world");
        assert_eq!(ids, vec![11, 16]);
    }

    #[test]
    fn test_empty_content_special_token_does_not_hang() {
        // Regression: a malformed tokenizer.json with a zero-length added-token
        // `content` injects "" into special_tokens. `"".starts_with(_)` is
        // always true, so match_special returns the same `pos` and
        // tokenize_to_ids_into never advances — an infinite loop with unbounded
        // output on the first tokenize() call. The constructor must drop
        // zero-length special tokens. Run in a worker thread so a re-introduced
        // hang fails fast via timeout instead of stalling the test binary.
        let mut vocab = HashMap::new();
        for (s, i) in [("a", 0u32), ("b", 1), ("c", 2)] {
            vocab.insert(s.to_string(), i);
        }
        let mut added = HashMap::new();
        added.insert(String::new(), 0u32);
        let tokenizer = BpeTokenizer::from_vocab_and_merges_with_config(
            vocab,
            Vec::new(),
            added,
            HashMap::new(),
            DEFAULT_BPE_CACHE_CAPACITY,
            DEFAULT_BPE_MAX_SEQ_LEN,
        )
        .expect("construct tokenizer with empty special token");

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(tokenizer.tokenize("abc").real_length);
        });
        match rx.recv_timeout(std::time::Duration::from_secs(5)) {
            Ok(len) => assert!(
                len < 1000,
                "empty special token produced runaway output ({len}); zero-length specials must be dropped"
            ),
            Err(_) => {
                panic!("tokenize() hung on a zero-length special token (infinite-loop regression)")
            }
        }
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

    #[test]
    fn test_bpe_decode_roundtrip() {
        let tokenizer = synthetic_bpe();
        // Mirrors test_bpe_merge_hello_world: "hello world" → [11, 16].
        let ids = tokenizer.tokenize_to_ids("hello world");
        assert_eq!(ids, vec![11, 16]);
        // Regression: decode must reverse the byte-level encoding (the "Ġ"
        // prefix maps back to a space), not return an empty string — the
        // prior generate.rs detokenize block discarded the text entirely.
        assert_eq!(tokenizer.decode(&ids), Some("hello world".to_string()));
        assert_eq!(tokenizer.decode(&[]), Some(String::new()));
    }

    #[test]
    fn test_decode_renders_nonspecial_added_tokens() {
        // Regression (qwen3.6-27b think-tag bug): added tokens with special=false
        // (`</think>`, `<tool_call>`, FIM markers) live BEYOND the base vocab range,
        // so before the fix `token_for_id` returned None and they decoded to the
        // empty string — silently swallowed from the output stream even though the
        // model sampled them correctly. They must now render as their literal text.
        // special=true markers (im_end-style) must STILL be swallowed.
        let mut vocab = HashMap::new();
        for (s, i) in [("a", 0u32), ("b", 1), ("c", 2)] {
            vocab.insert(s.to_string(), i);
        }
        // rendered_added = the special=false subset (content -> id), ids past base max.
        let mut rendered = HashMap::new();
        rendered.insert("</think>".to_string(), 100u32);
        rendered.insert("<think>".to_string(), 101u32);
        rendered.insert("<tool_call>".to_string(), 102u32);

        let tokenizer = BpeTokenizer::from_vocab_and_merges_with_config(
            vocab,
            Vec::new(),
            HashMap::new(),
            rendered,
            DEFAULT_BPE_CACHE_CAPACITY,
            DEFAULT_BPE_MAX_SEQ_LEN,
        )
        .expect("construct tokenizer with rendered added tokens");

        // special=false added tokens render verbatim (byte-level decode is identity
        // for printable ASCII).
        assert_eq!(tokenizer.decode(&[101]), Some("<think>".to_string()));
        assert_eq!(tokenizer.decode(&[100]), Some("</think>".to_string()));
        assert_eq!(tokenizer.decode(&[102]), Some("<tool_call>".to_string()));
        // Mixed with base content tokens, in stream order.
        assert_eq!(
            tokenizer.decode(&[0, 1, 100, 2]),
            Some("ab</think>c".to_string())
        );
        // An id present in neither base vocab nor the rendered set (e.g. a
        // special=true marker) stays swallowed — token_for_id returns None.
        assert_eq!(tokenizer.decode(&[200]), Some(String::new()));

        // The incremental streaming detokenizer must agree byte-for-byte.
        use crate::model::qwen35::detokenize::IncrementalDetokenizer;
        let mut detok = IncrementalDetokenizer::new();
        let mut out = String::new();
        for id in [0u32, 100, 1] {
            out.push_str(&detok.push(&tokenizer, id));
        }
        out.push_str(&detok.finish());
        assert_eq!(out, "a</think>b");
    }

    #[test]
    fn test_from_tokenizer_json_renders_real_qwen_think_tags() {
        // End-to-end regression gate for the qwen3.6 think-tag detok bug, driving
        // the REAL public loader (`from_tokenizer_json_str`) with the EXACT added-token
        // ids and special flags from the shipped qwen tokenizer.json (verified identical
        // on both qwen3.5-0.8b and qwen3.6-27b): `<think>`=248068, `</think>`=248069 are
        // special=false and live FAR above the base vocab max (248043). This exercises
        // the full wiring parse_rendered_added_tokens → added_render → token_for_id →
        // decode; a token-level parity gate (e2e-parity compares token IDS) is BLIND to
        // this class because the ids matched while the rendered text dropped the tag.
        let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1},
                "merges": []
            },
            "added_tokens": [
                {"id": 248045, "content": "<|im_start|>", "special": true},
                {"id": 248046, "content": "<|im_end|>", "special": true},
                {"id": 248058, "content": "<tool_call>", "special": false},
                {"id": 248068, "content": "<think>", "special": false},
                {"id": 248069, "content": "</think>", "special": false}
            ]
        }"#;
        let tokenizer =
            BpeTokenizer::from_tokenizer_json_str(json).expect("load tokenizer.json fixture");

        // special=false think/tool tags render verbatim (the bug: these were swallowed).
        assert_eq!(tokenizer.decode(&[248069]), Some("</think>".to_string()));
        assert_eq!(tokenizer.decode(&[248068]), Some("<think>".to_string()));
        assert_eq!(tokenizer.decode(&[248058]), Some("<tool_call>".to_string()));
        // special=true chat-control markers stay swallowed (HF skip_special_tokens=True).
        assert_eq!(tokenizer.decode(&[248046]), Some(String::new()));
        assert_eq!(tokenizer.decode(&[248045]), Some(String::new()));
        // A close tag mid-stream between base content tokens renders in order.
        assert_eq!(
            tokenizer.decode(&[0, 248069, 1]),
            Some("a</think>b".to_string())
        );
    }

    #[test]
    fn test_bpe_duplicate_merge_keeps_first_rank() {
        // Regression (#259): a duplicate merge pair must keep its FIRST rank.
        // First-wins: a+b (rank 0) merges before b+c (rank 1) → ["ab","c"].
        // The old last-wins overwrite demoted a+b to rank 2 → ["a","bc"].
        let mut vocab = HashMap::new();
        for (s, i) in [("a", 0u32), ("b", 1), ("c", 2), ("ab", 3), ("bc", 4)] {
            vocab.insert(s.to_string(), i);
        }
        let merges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
            ("a".to_string(), "b".to_string()),
        ];
        let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();
        assert_eq!(tokenizer.tokenize_to_ids("abc"), vec![3, 2]);
    }

    #[test]
    fn test_bpe_partial_byte_fallback_rolls_back() {
        // Regression (#259): a merged token absent from vocab whose chars only
        // partially resolve must roll back and emit a single <unk>, not leave
        // the resolved id orphaned. "ab" merges to "ab" (not in vocab); 'a'
        // resolves but 'b' does not → must be [<unk>]=[1], not [id(a),<unk>].
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0u32);
        vocab.insert("<unk>".to_string(), 1u32);
        let merges = vec![("a".to_string(), "b".to_string())];
        let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();
        assert_eq!(tokenizer.tokenize_to_ids("ab"), vec![1]);
    }
}
