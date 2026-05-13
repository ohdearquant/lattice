//! WordPiece tokenizer implementation.
//!
//! The primary fast path is a dependency-free double-array trie compiled once at
//! vocabulary load time. Whole-word tokens and `##` continuation tokens are kept
//! in separate tries so runtime tokenization can stay O(text length) while still
//! matching the exact greedy-longest semantics of the legacy implementation.

pub use crate::tokenizer::common::{TokenizedInput, Tokenizer};

use crate::error::InferenceError;
use crate::tokenizer::common::{
    ThreadSafeLruCache, invert_vocab, json_object_to_vocab, json_path, pad_ids,
    pad_ids_with_token_types, parse_json,
};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use tracing::warn;

const DEFAULT_WORDPIECE_CACHE_CAPACITY: usize = 8_192;
const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// **Unstable**: WordPiece tokenizer; double-array trie internals and special-token handling evolving.
///
/// WordPiece tokenizer for BERT-family models.
#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    inner: Arc<WordPieceInner>,
}

#[derive(Debug)]
struct WordPieceInner {
    id_to_token: Vec<String>,
    whole_word_trie: Arc<DoubleArrayTrie>,
    continuation_trie: Arc<DoubleArrayTrie>,
    cls_id: u32,
    sep_id: u32,
    pad_id: u32,
    unk_id: u32,
    mask_id: u32,
    max_seq_len: usize,
    cache: ThreadSafeLruCache<String, Vec<u32>>,
}

/// **Unstable**: double-array trie for O(text-length) WordPiece lookup; DAT construction algorithm may change.
#[derive(Debug, Clone)]
pub struct DoubleArrayTrie {
    /// base[state] + label = next_state
    pub base: Vec<i32>,
    /// check[next_state] == state validates the transition.
    pub check: Vec<i32>,
    /// Accepting value for a state, or -1 when non-accepting.
    pub value: Vec<i32>,
    char_to_label: HashMap<u32, i32>,
    root: usize,
}

#[derive(Debug, Default)]
struct PlainTrieNode {
    children: BTreeMap<i32, usize>,
    value: i32,
}

#[derive(Debug, Default)]
struct PlainTrieBuilder {
    nodes: Vec<PlainTrieNode>,
    char_to_label: HashMap<u32, i32>,
    next_label: i32,
}

#[derive(Debug)]
struct DatCompiler {
    base: Vec<i32>,
    check: Vec<i32>,
    value: Vec<i32>,
    next_search_pos: i32,
    char_to_label: HashMap<u32, i32>,
}

#[derive(Debug, Default)]
struct TokenizeScratch {
    normalized: String,
    words: Vec<String>,
    piece_ids: Vec<u32>,
}

impl PlainTrieBuilder {
    fn new() -> Self {
        Self {
            nodes: vec![PlainTrieNode {
                children: BTreeMap::new(),
                value: -1,
            }],
            char_to_label: HashMap::new(),
            next_label: 1,
        }
    }

    fn label_for_char(&mut self, ch: char) -> i32 {
        let key = ch as u32;
        if let Some(&label) = self.char_to_label.get(&key) {
            return label;
        }
        let label = self.next_label;
        self.next_label += 1;
        self.char_to_label.insert(key, label);
        label
    }

    fn insert(&mut self, token: &str, value: i32) {
        let mut node_idx = 0usize;
        for ch in token.chars() {
            let label = self.label_for_char(ch);
            let next_idx = if let Some(&child_idx) = self.nodes[node_idx].children.get(&label) {
                child_idx
            } else {
                let child_idx = self.nodes.len();
                self.nodes.push(PlainTrieNode {
                    children: BTreeMap::new(),
                    value: -1,
                });
                self.nodes[node_idx].children.insert(label, child_idx);
                child_idx
            };
            node_idx = next_idx;
        }
        self.nodes[node_idx].value = value;
    }

    fn build(self) -> DoubleArrayTrie {
        let mut compiler = DatCompiler::new(self.char_to_label);
        compiler.ensure_state(1);
        compiler.check[1] = 0;
        compiler.value[1] = self.nodes[0].value;
        compiler.place_node(1, &self.nodes, 0);
        compiler.finish()
    }
}

impl DatCompiler {
    fn new(char_to_label: HashMap<u32, i32>) -> Self {
        Self {
            base: vec![0, 0],
            check: vec![-1, 0],
            value: vec![-1, -1],
            next_search_pos: 1,
            char_to_label,
        }
    }

    fn ensure_state(&mut self, state: usize) {
        if state >= self.base.len() {
            let new_len = state + 1;
            self.base.resize(new_len, 0);
            self.check.resize(new_len, -1);
            self.value.resize(new_len, -1);
        }
    }

    fn find_base(&mut self, labels: &[i32]) -> i32 {
        if labels.is_empty() {
            return 0;
        }

        let mut base = self.next_search_pos.max(1);
        'search: loop {
            for &label in labels {
                let target = (base + label) as usize;
                self.ensure_state(target);
                if self.check[target] != -1 {
                    base += 1;
                    continue 'search;
                }
            }
            self.next_search_pos = base;
            return base;
        }
    }

    fn place_node(&mut self, state: usize, plain_nodes: &[PlainTrieNode], plain_state: usize) {
        let node = &plain_nodes[plain_state];
        self.value[state] = node.value;
        if node.children.is_empty() {
            return;
        }

        let labels: Vec<i32> = node.children.keys().copied().collect();
        let base = self.find_base(&labels);
        self.base[state] = base;

        for (&label, &child_plain_state) in &node.children {
            let target = (base + label) as usize;
            self.ensure_state(target);
            self.check[target] = state as i32;
            self.value[target] = plain_nodes[child_plain_state].value;
        }

        for (&label, &child_plain_state) in &node.children {
            let target = (base + label) as usize;
            self.place_node(target, plain_nodes, child_plain_state);
        }
    }

    fn finish(self) -> DoubleArrayTrie {
        DoubleArrayTrie {
            base: self.base,
            check: self.check,
            value: self.value,
            char_to_label: self.char_to_label,
            root: 1,
        }
    }
}

impl DoubleArrayTrie {
    fn longest_match_chars(&self, chars: &[char], start: usize) -> Option<(u32, usize)> {
        let mut state = self.root;
        let mut idx = start;
        let mut last_match = if self.value[state] >= 0 {
            Some((self.value[state] as u32, idx))
        } else {
            None
        };

        while idx < chars.len() {
            let Some(&label) = self.char_to_label.get(&(chars[idx] as u32)) else {
                break;
            };

            let next = self.base[state] + label;
            if next < 0 {
                break;
            }
            let next_state = next as usize;
            if next_state >= self.check.len() || self.check[next_state] != state as i32 {
                break;
            }

            state = next_state;
            idx += 1;
            if self.value[state] >= 0 {
                last_match = Some((self.value[state] as u32, idx));
            }
        }

        last_match
    }
}

impl WordPieceTokenizer {
    /// **Unstable**: load from vocab.txt file; file format assumptions may change.
    ///
    /// Load tokenizer from a vocab.txt file.
    pub fn from_file(path: &Path) -> Result<Self, InferenceError> {
        let vocab_text = fs::read_to_string(path).map_err(|e| {
            InferenceError::Tokenizer(format!("failed to read {}: {e}", path.display()))
        })?;
        Self::from_str(&vocab_text)
    }

    /// **Unstable**: load from in-memory vocab string.
    ///
    /// Load tokenizer from an in-memory vocab string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(vocab_text: &str) -> Result<Self, InferenceError> {
        Self::from_str_with_cache_capacity(vocab_text, DEFAULT_WORDPIECE_CACHE_CAPACITY)
    }

    /// **Unstable**: load from HF tokenizer.json file; JSON schema assumptions may change.
    ///
    /// Load tokenizer from a `tokenizer.json` payload.
    pub fn from_tokenizer_json(path: &Path) -> Result<Self, InferenceError> {
        let text = fs::read_to_string(path).map_err(|e| {
            InferenceError::Tokenizer(format!("failed to read {}: {e}", path.display()))
        })?;
        Self::from_tokenizer_json_str(&text)
    }

    /// **Unstable**: load from HF tokenizer.json string; JSON schema assumptions may change.
    ///
    /// Load tokenizer from a `tokenizer.json` string.
    pub fn from_tokenizer_json_str(text: &str) -> Result<Self, InferenceError> {
        let root = parse_json(text)?;
        let model_type = json_path(&root, &["model", "type"])
            .and_then(|value| value.as_str())
            .unwrap_or("");
        if model_type != "WordPiece" {
            return Err(InferenceError::Tokenizer(format!(
                "expected WordPiece tokenizer.json model type, found {model_type:?}"
            )));
        }
        let vocab =
            json_object_to_vocab(json_path(&root, &["model", "vocab"]).ok_or_else(|| {
                InferenceError::Tokenizer("tokenizer.json missing model.vocab".into())
            })?)?;
        Self::from_vocab_map(vocab, DEFAULT_WORDPIECE_CACHE_CAPACITY, DEFAULT_MAX_SEQ_LEN)
    }

    /// **Unstable**: load from file with explicit cache capacity.
    ///
    /// Load tokenizer with an explicit cache capacity.
    pub fn from_file_with_cache_capacity(
        path: &Path,
        cache_capacity: usize,
    ) -> Result<Self, InferenceError> {
        let vocab_text = fs::read_to_string(path).map_err(|e| {
            InferenceError::Tokenizer(format!("failed to read {}: {e}", path.display()))
        })?;
        Self::from_str_with_cache_capacity(&vocab_text, cache_capacity)
    }

    /// **Unstable**: load from string with explicit cache capacity.
    ///
    /// Load tokenizer with an explicit cache capacity.
    pub fn from_str_with_cache_capacity(
        vocab_text: &str,
        cache_capacity: usize,
    ) -> Result<Self, InferenceError> {
        let mut vocab = HashMap::new();
        for (idx, line) in vocab_text.lines().enumerate() {
            let token = line.trim_end_matches('\r').to_string();
            vocab.insert(token, idx as u32);
        }
        Self::from_vocab_map(vocab, cache_capacity, DEFAULT_MAX_SEQ_LEN)
    }

    fn from_vocab_map(
        vocab: HashMap<String, u32>,
        cache_capacity: usize,
        max_seq_len: usize,
    ) -> Result<Self, InferenceError> {
        let id_to_token = invert_vocab(&vocab)?;

        let cls_id = *vocab
            .get("[CLS]")
            .ok_or_else(|| InferenceError::Tokenizer("missing [CLS] token".into()))?;
        let sep_id = *vocab
            .get("[SEP]")
            .ok_or_else(|| InferenceError::Tokenizer("missing [SEP] token".into()))?;
        let pad_id = *vocab
            .get("[PAD]")
            .ok_or_else(|| InferenceError::Tokenizer("missing [PAD] token".into()))?;
        let unk_id = *vocab
            .get("[UNK]")
            .ok_or_else(|| InferenceError::Tokenizer("missing [UNK] token".into()))?;
        let mask_id = *vocab
            .get("[MASK]")
            .ok_or_else(|| InferenceError::Tokenizer("missing [MASK] token".into()))?;

        let mut whole_builder = PlainTrieBuilder::new();
        let mut continuation_builder = PlainTrieBuilder::new();
        for (token, &id) in &vocab {
            if let Some(stripped) = token.strip_prefix("##") {
                continuation_builder.insert(stripped, id as i32);
            } else {
                whole_builder.insert(token, id as i32);
            }
        }

        let inner = WordPieceInner {
            id_to_token,
            whole_word_trie: Arc::new(whole_builder.build()),
            continuation_trie: Arc::new(continuation_builder.build()),
            cls_id,
            sep_id,
            pad_id,
            unk_id,
            mask_id,
            max_seq_len,
            cache: ThreadSafeLruCache::new(cache_capacity),
        };

        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// **Unstable**: override maximum sequence length; cloning inner Arc may change.
    ///
    /// Override the configured maximum sequence length.
    pub fn with_max_seq_len(&self, max_seq_len: usize) -> Self {
        let inner = WordPieceInner {
            id_to_token: self.inner.id_to_token.clone(),
            whole_word_trie: self.inner.whole_word_trie.clone(),
            continuation_trie: self.inner.continuation_trie.clone(),
            cls_id: self.inner.cls_id,
            sep_id: self.inner.sep_id,
            pad_id: self.inner.pad_id,
            unk_id: self.inner.unk_id,
            mask_id: self.inner.mask_id,
            max_seq_len,
            cache: self.inner.cache.clone(),
        };
        Self {
            inner: Arc::new(inner),
        }
    }

    /// **Unstable**: maximum sequence length accessor.
    ///
    /// Return the configured maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.inner.max_seq_len
    }

    /// **Unstable**: vocabulary size accessor.
    ///
    /// Return the model vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.id_to_token.len()
    }

    /// **Unstable**: tokenize single text; output format mirrors `Tokenizer` trait.
    ///
    /// Tokenize a single text into model-ready input padded to `max_seq_len`.
    pub fn tokenize(&self, text: &str) -> TokenizedInput {
        <Self as Tokenizer>::tokenize(self, text)
    }

    /// **Unstable**: batch tokenize; padding strategy may change.
    ///
    /// Tokenize a batch of texts, padding all examples to a batch-local maximum.
    pub fn tokenize_batch(&self, texts: &[&str]) -> Vec<TokenizedInput> {
        <Self as Tokenizer>::tokenize_batch(self, texts)
    }

    /// **Unstable**: BERT-style pair tokenization for cross-encoder reranking.
    ///
    /// Produces `[CLS] query [SEP] doc [SEP]` with `token_type_ids` 0/1.
    pub fn tokenize_pair(&self, query: &str, document: &str) -> TokenizedInput {
        <Self as Tokenizer>::tokenize_pair(self, query, document)
    }

    fn tokenize_to_ids(&self, text: &str) -> Vec<u32> {
        let mut scratch = TokenizeScratch::default();
        let mut ids = Vec::with_capacity(text.len().saturating_div(2).max(8));
        self.tokenize_to_ids_into(text, &mut scratch, &mut ids);
        ids
    }

    fn tokenize_to_ids_into(&self, text: &str, scratch: &mut TokenizeScratch, out: &mut Vec<u32>) {
        out.clear();
        out.push(self.inner.cls_id);
        pre_tokenize_into(text, &mut scratch.normalized, &mut scratch.words);

        for word in &scratch.words {
            if let Some(cached) = self.inner.cache.get(word) {
                out.extend(cached.iter().copied());
                continue;
            }

            scratch.piece_ids.clear();
            self.wordpiece_tokenize_word_into(word, &mut scratch.piece_ids);
            self.inner
                .cache
                .insert(word.clone(), scratch.piece_ids.clone());
            out.extend(scratch.piece_ids.iter().copied());
        }

        out.push(self.inner.sep_id);

        if out.len() > self.inner.max_seq_len {
            warn!(
                original_len = out.len(),
                max_seq_len = self.inner.max_seq_len,
                "truncating tokenized input to max_seq_len"
            );
            out.truncate(self.inner.max_seq_len);
            if let Some(last) = out.last_mut() {
                *last = self.inner.sep_id;
            }
        }
    }

    /// Tokenize `text` payload without special tokens; push word-piece IDs into `out`.
    fn tokenize_wordpiece_payload_into(
        &self,
        text: &str,
        scratch: &mut TokenizeScratch,
        out: &mut Vec<u32>,
    ) {
        pre_tokenize_into(text, &mut scratch.normalized, &mut scratch.words);
        for word in &scratch.words {
            if let Some(cached) = self.inner.cache.get(word) {
                out.extend(cached.iter().copied());
                continue;
            }
            scratch.piece_ids.clear();
            self.wordpiece_tokenize_word_into(word, &mut scratch.piece_ids);
            self.inner
                .cache
                .insert(word.clone(), scratch.piece_ids.clone());
            out.extend(scratch.piece_ids.iter().copied());
        }
    }

    fn wordpiece_tokenize_word_into(&self, word: &str, out: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }

        let chars: Vec<char> = word.chars().collect();
        let mut start = 0usize;

        while start < chars.len() {
            let trie = if start == 0 {
                &self.inner.whole_word_trie
            } else {
                &self.inner.continuation_trie
            };

            if let Some((id, end)) = trie.longest_match_chars(&chars, start) {
                out.push(id);
                start = end;
            } else {
                out.push(self.inner.unk_id);
                start += 1;
            }
        }
    }

    fn pad_batch(&self, id_batches: Vec<Vec<u32>>, pad_to: usize) -> Vec<TokenizedInput> {
        id_batches
            .into_iter()
            .map(|ids| pad_ids(ids, pad_to, self.inner.pad_id))
            .collect()
    }

    /// **Unstable**: reverse lookup for debugging; id_to_token table layout may change.
    ///
    /// Reverse lookup for debugging.
    pub fn token_for_id(&self, id: u32) -> Option<&str> {
        self.inner.id_to_token.get(id as usize).map(String::as_str)
    }

    /// **Unstable**: [MASK] token id accessor.
    ///
    /// Return the [MASK] token id.
    pub fn mask_id(&self) -> u32 {
        self.inner.mask_id
    }
}

impl Tokenizer for WordPieceTokenizer {
    fn tokenize(&self, text: &str) -> TokenizedInput {
        let ids = self.tokenize_to_ids(text);
        pad_ids(ids, self.inner.max_seq_len, self.inner.pad_id)
    }

    fn tokenize_batch(&self, texts: &[&str]) -> Vec<TokenizedInput> {
        if texts.is_empty() {
            return Vec::new();
        }

        let threads = thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .max(1);

        let id_batches = if texts.len() > 16 && threads > 1 {
            let chunk_count = threads.min(texts.len());
            let chunk_size = texts.len().div_ceil(chunk_count);
            thread::scope(|scope| {
                let mut handles = Vec::new();
                for (chunk_idx, chunk) in texts.chunks(chunk_size).enumerate() {
                    let tokenizer = self;
                    handles.push(scope.spawn(move || {
                        let mut scratch = TokenizeScratch::default();
                        let mut reusable_ids = Vec::new();
                        let mut local_max = 0usize;
                        let mut local = Vec::with_capacity(chunk.len());
                        for text in chunk {
                            tokenizer.tokenize_to_ids_into(text, &mut scratch, &mut reusable_ids);
                            local_max = local_max.max(reusable_ids.len());
                            local.push(reusable_ids.clone());
                        }
                        (chunk_idx, local_max, local)
                    }));
                }

                let mut parts = Vec::with_capacity(handles.len());
                for handle in handles {
                    parts.push(
                        handle
                            .join()
                            .expect("invariant: scoped tokenization worker should not panic"),
                    );
                }
                parts.sort_by_key(|(chunk_idx, _, _)| *chunk_idx);

                let mut max_len = 0usize;
                let mut all = Vec::with_capacity(texts.len());
                for (_, local_max, local) in parts {
                    max_len = max_len.max(local_max);
                    all.extend(local);
                }
                (all, max_len)
            })
        } else {
            let mut scratch = TokenizeScratch::default();
            let mut reusable_ids = Vec::new();
            let mut max_len = 0usize;
            let mut all = Vec::with_capacity(texts.len());
            for text in texts {
                self.tokenize_to_ids_into(text, &mut scratch, &mut reusable_ids);
                max_len = max_len.max(reusable_ids.len());
                all.push(reusable_ids.clone());
            }
            (all, max_len)
        };

        self.pad_batch(id_batches.0, id_batches.1)
    }

    fn vocab_size(&self) -> usize {
        self.inner.id_to_token.len()
    }

    fn max_seq_len(&self) -> usize {
        self.inner.max_seq_len
    }

    fn supports_pair_tokenization(&self) -> bool {
        true
    }

    fn tokenize_pair(&self, query: &str, document: &str) -> TokenizedInput {
        let max_seq_len = self.inner.max_seq_len;
        let mut scratch = TokenizeScratch::default();

        let mut query_ids: Vec<u32> = Vec::new();
        let mut doc_ids: Vec<u32> = Vec::new();

        self.tokenize_wordpiece_payload_into(query, &mut scratch, &mut query_ids);
        self.tokenize_wordpiece_payload_into(document, &mut scratch, &mut doc_ids);

        // 3 slots reserved for [CLS], first [SEP], final [SEP].
        let payload_budget = max_seq_len.saturating_sub(3);
        if query_ids.len() > payload_budget {
            query_ids.truncate(payload_budget);
            doc_ids.clear();
        } else {
            let doc_budget = payload_budget - query_ids.len();
            if doc_ids.len() > doc_budget {
                doc_ids.truncate(doc_budget);
            }
        }

        // Layout: [CLS](0) query...(0) [SEP](0) doc...(1) [SEP](1)
        let mut ids: Vec<u32> = Vec::with_capacity(max_seq_len.max(1));
        let mut token_type_ids: Vec<u32> = Vec::with_capacity(max_seq_len.max(1));

        ids.push(self.inner.cls_id);
        token_type_ids.push(0u32);

        for id in query_ids {
            ids.push(id);
            token_type_ids.push(0u32);
        }

        ids.push(self.inner.sep_id);
        token_type_ids.push(0u32);

        for id in doc_ids {
            ids.push(id);
            token_type_ids.push(1u32);
        }

        ids.push(self.inner.sep_id);
        token_type_ids.push(1u32);

        // Edge case: max_seq_len < 3 means some special tokens do not fit.
        ids.truncate(max_seq_len);
        token_type_ids.truncate(max_seq_len);

        pad_ids_with_token_types(ids, token_type_ids, max_seq_len, self.inner.pad_id)
    }
}

fn is_combining_mark(ch: char) -> bool {
    matches!(
        ch as u32,
        0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F
    )
}

fn fold_diacritic(ch: char) -> &'static str {
    match ch {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' | 'ǎ' | 'ȁ' | 'ȃ' | 'ạ' | 'ả' | 'ấ'
        | 'ầ' | 'ẩ' | 'ẫ' | 'ậ' | 'ắ' | 'ằ' | 'ẳ' | 'ẵ' | 'ặ' => "a",
        'æ' => "ae",
        'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => "c",
        'ď' | 'đ' => "d",
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' | 'ẹ' | 'ẻ' | 'ẽ' | 'ế' | 'ề' | 'ể'
        | 'ễ' | 'ệ' => "e",
        'ƒ' => "f",
        'ĝ' | 'ğ' | 'ġ' | 'ģ' => "g",
        'ĥ' | 'ħ' => "h",
        'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' | 'ǐ' | 'ị' | 'ỉ' => "i",
        'ĵ' => "j",
        'ķ' => "k",
        'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => "l",
        'ñ' | 'ń' | 'ņ' | 'ň' | 'ŋ' => "n",
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' | 'ǒ' | 'ọ' | 'ỏ' | 'ố' | 'ồ' | 'ổ'
        | 'ỗ' | 'ộ' | 'ớ' | 'ờ' | 'ở' | 'ỡ' | 'ợ' => "o",
        'œ' => "oe",
        'ŕ' | 'ŗ' | 'ř' => "r",
        'ś' | 'ŝ' | 'ş' | 'š' | 'ș' => "s",
        'ß' => "ss",
        'ţ' | 'ť' | 'ŧ' | 'ț' => "t",
        'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' | 'ǔ' | 'ụ' | 'ủ' | 'ứ' | 'ừ'
        | 'ử' | 'ữ' | 'ự' => "u",
        'ŵ' => "w",
        'ý' | 'ÿ' | 'ŷ' | 'ỳ' | 'ỵ' | 'ỷ' | 'ỹ' => "y",
        'ź' | 'ż' | 'ž' => "z",
        _ => "",
    }
}

fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

fn is_punctuation(ch: char) -> bool {
    if ch.is_ascii_punctuation() {
        return true;
    }

    let cp = ch as u32;
    matches!(cp, 0x2000..=0x206F | 0x2E00..=0x2E7F)
        || (!ch.is_alphanumeric() && !ch.is_whitespace() && !is_cjk(ch))
}

/// BERT-style basic pre-tokenization.
#[cfg(test)]
fn pre_tokenize(text: &str) -> Vec<String> {
    let mut normalized = String::new();
    let mut words = Vec::new();
    pre_tokenize_into(text, &mut normalized, &mut words);
    words
}

fn pre_tokenize_into(text: &str, normalized: &mut String, out: &mut Vec<String>) {
    out.clear();
    if text.is_empty() {
        normalized.clear();
        return;
    }

    normalized.clear();
    normalized.reserve(text.len().saturating_mul(2));

    for original in text.chars() {
        for lower in original.to_lowercase() {
            let folded = fold_diacritic(lower);
            if folded.is_empty()
                && !is_combining_mark(lower)
                && (!lower.is_control() || lower.is_whitespace())
            {
                push_normalized(lower, normalized);
            } else {
                for ch in folded.chars() {
                    push_normalized(ch, normalized);
                }
            }
        }
    }

    split_normalized_whitespace_into(normalized, out);
}

fn push_normalized(ch: char, normalized: &mut String) {
    if is_combining_mark(ch) {
        return;
    }

    if is_cjk(ch) {
        normalized.push(' ');
        normalized.push(ch);
        normalized.push(' ');
    } else if ch.is_whitespace() {
        normalized.push(' ');
    } else if is_punctuation(ch) {
        normalized.push(' ');
        normalized.push(ch);
        normalized.push(' ');
    } else if !ch.is_control() || ch.is_whitespace() {
        normalized.push(ch);
    }
}

fn split_normalized_whitespace_into(text: &str, out: &mut Vec<String>) {
    out.clear();
    if text.is_empty() {
        return;
    }

    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i] == b' ' {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let start = i;
        match simd_find_next_space(bytes, start) {
            Some(space) => {
                out.push(text[start..space].to_string());
                i = space + 1;
            }
            None => {
                out.push(text[start..].to_string());
                break;
            }
        }
    }
}

fn simd_find_next_space(bytes: &[u8], start: usize) -> Option<usize> {
    if start >= bytes.len() {
        return None;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 is enabled by the runtime feature check. `start < bytes.len()`
            // is checked above, and the kernel bounds every unaligned load by `bytes.len()`.
            return unsafe { find_next_space_avx2(bytes, start) };
        }
        // SAFETY: SSE2 is guaranteed on x86_64. `start < bytes.len()` is checked
        // above, and the kernel bounds every unaligned load by `bytes.len()`.
        return unsafe { find_next_space_sse2(bytes, start) };
    }

    #[cfg(all(target_arch = "aarch64", not(target_arch = "x86_64")))]
    {
        // SAFETY: NEON is part of AArch64. `start < bytes.len()` is checked
        // above, and the kernel bounds every load by `bytes.len()`.
        return unsafe { find_next_space_neon(bytes, start) };
    }

    #[allow(unreachable_code)]
    scalar_find_next_space(bytes, start)
}

fn scalar_find_next_space(bytes: &[u8], start: usize) -> Option<usize> {
    bytes[start..]
        .iter()
        .position(|&byte| byte == b' ')
        .map(|offset| start + offset)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
/// # Safety
///
/// Caller must ensure SSE2 is available and `start < bytes.len()`. The function
/// only performs unaligned 16-byte loads when `i + 16 <= bytes.len()`.
unsafe fn find_next_space_sse2(bytes: &[u8], start: usize) -> Option<usize> {
    use std::arch::x86_64::*;

    let needle = _mm_set1_epi8(b' ' as i8);
    let mut i = start;
    while i + 16 <= bytes.len() {
        let chunk = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);
        let cmp = _mm_cmpeq_epi8(chunk, needle);
        let mask = _mm_movemask_epi8(cmp) as u32;
        if mask != 0 {
            let offset = mask.trailing_zeros() as usize;
            return Some(i + offset);
        }
        i += 16;
    }
    scalar_find_next_space(bytes, i)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// # Safety
///
/// Caller must ensure AVX2 is available and `start < bytes.len()`. The function
/// only performs unaligned 32-byte loads when `i + 32 <= bytes.len()`.
unsafe fn find_next_space_avx2(bytes: &[u8], start: usize) -> Option<usize> {
    use std::arch::x86_64::*;

    let needle = _mm256_set1_epi8(b' ' as i8);
    let mut i = start;
    while i + 32 <= bytes.len() {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(chunk, needle);
        let mask = _mm256_movemask_epi8(cmp) as u32;
        if mask != 0 {
            let offset = mask.trailing_zeros() as usize;
            return Some(i + offset);
        }
        i += 32;
    }
    find_next_space_sse2(bytes, i)
}

#[cfg(target_arch = "aarch64")]
/// # Safety
///
/// Caller must ensure NEON is available and `start < bytes.len()`. The function
/// only performs 16-byte loads when `i + 16 <= bytes.len()`; fallback byte reads
/// stay within that same checked range.
unsafe fn find_next_space_neon(bytes: &[u8], start: usize) -> Option<usize> {
    use std::arch::aarch64::*;

    let needle = vdupq_n_u8(b' ');
    let mut i = start;
    while i + 16 <= bytes.len() {
        let chunk = vld1q_u8(bytes.as_ptr().add(i));
        let cmp = vceqq_u8(chunk, needle);
        if vmaxvq_u8(cmp) != 0 {
            for j in 0..16 {
                if *bytes.get_unchecked(i + j) == b' ' {
                    return Some(i + j);
                }
            }
        }
        i += 16;
    }
    scalar_find_next_space(bytes, i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_spec_vocab() -> String {
        let max_id = 16_515usize;
        let mut tokens = (0..=max_id)
            .map(|i| format!("[unused{}]", i))
            .collect::<Vec<_>>();

        let entries = [
            (0usize, "[PAD]"),
            (100, "[UNK]"),
            (101, "[CLS]"),
            (102, "[SEP]"),
            (103, "[MASK]"),
            (1037, "a"),
            (1996, "the"),
            (2003, "is"),
            (2088, "world"),
            (2829, "brown"),
            (2890, "##re"),
            (4248, "quick"),
            (4419, "fox"),
            (4667, "##ding"),
            (4895, "un"),
            (7592, "hello"),
            (7861, "em"),
            (8270, "##bed"),
            (9932, "ai"),
            (10_880, "##zable"),
            (13_970, "transforming"),
            (16_515, "##cogni"),
        ];

        for (idx, token) in entries {
            tokens[idx] = token.to_string();
        }

        tokens.join("\n")
    }

    #[test]
    fn test_trie_longest_match() {
        let mut builder = PlainTrieBuilder::new();
        builder.insert("em", 1);
        builder.insert("embed", 2);
        builder.insert("embedding", 3);
        let trie = builder.build();
        let chars: Vec<char> = "embedding".chars().collect();
        assert_eq!(trie.longest_match_chars(&chars, 0), Some((3, 9)));
    }

    #[test]
    fn test_tokenize_hello_world() {
        let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
        let input = tokenizer.tokenize("hello world");
        assert_eq!(input.input_ids[0], 101);
        assert_eq!(input.input_ids[1], 7592);
        assert_eq!(input.input_ids[2], 2088);
        assert_eq!(input.input_ids[3], 102);
        assert_eq!(input.real_length, 4);
    }

    #[test]
    fn test_tokenize_subwords() {
        let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
        let input = tokenizer.tokenize("embedding");
        assert_eq!(input.input_ids[1], 7861);
        assert_eq!(input.input_ids[2], 8270);
        assert_eq!(input.input_ids[3], 4667);
        assert_eq!(input.real_length, 5);
    }

    #[test]
    fn test_spec_tokenizer_vectors() {
        let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();

        let cases: &[(&str, &[u32])] = &[
            ("hello world", &[7592, 2088]),
            ("The quick brown fox", &[1996, 4248, 2829, 4419]),
            ("embedding", &[7861, 8270, 4667]),
            ("unrecognizable", &[4895, 2890, 16515, 10880]),
            ("", &[]),
            ("a", &[1037]),
            (
                "AI is transforming the world",
                &[9932, 2003, 13970, 1996, 2088],
            ),
        ];

        for (text, expected) in cases {
            let input = tokenizer.tokenize(text);
            let actual = &input.input_ids[1..input.real_length - 1];
            assert_eq!(actual, *expected, "text={text:?}");
        }
    }

    #[test]
    fn test_pre_tokenize_lowercase_punctuation_and_accents() {
        let tokens = pre_tokenize("He\u{0301}llo, WORLD!");
        assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn test_tokenize_batch_pads_to_batch_max() {
        let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
        let batch = tokenizer.tokenize_batch(&["hello", "hello world"]);
        assert_eq!(batch[0].input_ids.len(), 4);
        assert_eq!(batch[1].input_ids.len(), 4);
        assert_eq!(batch[0].attention_mask, vec![1, 1, 1, 0]);
        assert_eq!(batch[1].attention_mask, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_wordpiece_cache_hits() {
        let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
        let first = tokenizer.tokenize("embedding embedding");
        let second = tokenizer.tokenize("embedding embedding");
        assert_eq!(first, second);
    }

    // --- cross-encoder pair tokenization tests ---

    #[test]
    fn test_tokenize_pair_layout() {
        let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
        let input = tokenizer.tokenize_pair("hello", "world");
        // Expected: [CLS](101) hello(7592) [SEP](102) world(2088) [SEP](102)
        assert_eq!(input.real_length, 5);
        assert_eq!(input.input_ids[0], 101); // [CLS]
        assert_eq!(input.input_ids[1], 7592); // hello
        assert_eq!(input.input_ids[2], 102); // first [SEP]
        assert_eq!(input.input_ids[3], 2088); // world
        assert_eq!(input.input_ids[4], 102); // final [SEP]
    }

    #[test]
    fn test_tokenize_pair_token_type_ids() {
        let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
        let input = tokenizer.tokenize_pair("hello", "world");
        // Segment 0: [CLS], hello, first [SEP] → type 0
        // Segment 1: world, final [SEP] → type 1
        assert_eq!(
            &input.token_type_ids[..input.real_length],
            &[0u32, 0, 0, 1, 1]
        );
        // Padding positions must also be type 0
        for i in input.real_length..input.input_ids.len() {
            assert_eq!(
                input.token_type_ids[i], 0,
                "padding at position {i} must have token_type_id 0"
            );
        }
    }

    #[test]
    fn test_tokenize_pair_doc_truncated_before_query() {
        // max_seq_len=6: payload_budget=3, query "hello world"=2 tokens → doc_budget=1
        // doc "is the world" → 3 tokens, truncated to 1 → [is(2003)]
        // Expected: [CLS(101), hello(7592), world(2088), SEP(102), is(2003), SEP(102)]
        let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab())
            .unwrap()
            .with_max_seq_len(6);
        let input = tokenizer.tokenize_pair("hello world", "is the world");
        assert_eq!(input.real_length, 6);
        assert_eq!(input.input_ids[0], 101); // [CLS]
        assert_eq!(input.input_ids[1], 7592); // hello
        assert_eq!(input.input_ids[2], 2088); // world
        assert_eq!(input.input_ids[3], 102); // first [SEP]
        assert_eq!(input.input_ids[4], 2003); // is (doc, truncated to 1 token)
        assert_eq!(input.input_ids[5], 102); // final [SEP]
        assert_eq!(&input.token_type_ids[..6], &[0u32, 0, 0, 0, 1, 1]);
    }

    #[test]
    fn test_tokenize_pair_query_truncated_drops_doc() {
        // max_seq_len=4: payload_budget=1, query "hello world"=2 > 1
        // → query truncated to [hello(7592)], doc cleared
        // Expected: [CLS(101), hello(7592), SEP(102), SEP(102)]
        let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab())
            .unwrap()
            .with_max_seq_len(4);
        let input = tokenizer.tokenize_pair("hello world", "is");
        assert_eq!(input.real_length, 4);
        assert_eq!(input.input_ids[0], 101); // [CLS]
        assert_eq!(input.input_ids[1], 7592); // hello (only query token kept)
        assert_eq!(input.input_ids[2], 102); // first [SEP]
        assert_eq!(input.input_ids[3], 102); // final [SEP]
        // token_type_ids: CLS=0, hello=0, first SEP=0, final SEP=1
        assert_eq!(&input.token_type_ids[..4], &[0u32, 0, 0, 1]);
    }

    #[test]
    fn test_tokenize_pair_supports_pair_tokenization() {
        let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab()).unwrap();
        assert!(tokenizer.supports_pair_tokenization());
    }

    #[test]
    fn test_tokenize_pair_padding_attention_mask() {
        // With large max_seq_len, positions past real_length must have attention_mask=0
        let tokenizer = WordPieceTokenizer::from_str(&synthetic_spec_vocab())
            .unwrap()
            .with_max_seq_len(16);
        let input = tokenizer.tokenize_pair("hello", "world");
        assert_eq!(input.real_length, 5);
        for i in 0..5 {
            assert_eq!(
                input.attention_mask[i], 1,
                "real token at {i} must be unmasked"
            );
        }
        for i in 5..16 {
            assert_eq!(input.attention_mask[i], 0, "padding at {i} must be masked");
            assert_eq!(input.token_type_ids[i], 0, "padding type at {i} must be 0");
        }
    }
}
