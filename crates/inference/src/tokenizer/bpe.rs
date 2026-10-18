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
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use tracing::warn;

pub(crate) const DEFAULT_BPE_CACHE_CAPACITY: usize = 8_192;
pub(crate) const DEFAULT_BPE_MAX_SEQ_LEN: usize = 4_096;

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
    /// Ids `added_tokens` marks special (`special: true`, or a `"special"`
    /// value that is present but not a JSON boolean, which fails closed to
    /// special -- see `common::parse_rendered_added_tokens`). Computed as
    /// every id in `added_render`'s source map that did *not* make it into
    /// `added_render` itself, i.e. the complement of the render set within
    /// the full added-token set (mirrors `GemmaBpeTokenizer`'s
    /// `special_skip_ids`). These ids must contribute no bytes anywhere --
    /// `append_token_bytes` and `token_for_id` both check this set before
    /// falling back to `added_render` or the base `vocab`, so a control
    /// token declared special can never leak the base-vocab spelling it
    /// happens to collide with (#1722).
    special_skip_ids: HashSet<u32>,
    pad_id: u32,
    unk_id: Option<u32>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    pre_tokenize_mode: PreTokenizeMode,
    max_seq_len: usize,
    cache: ThreadSafeLruCache<String, Vec<u32>>,
    newline_boundary: OnceLock<Arc<[bool]>>,
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

    pub(crate) fn from_vocab_and_merges_with_config(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
        added_tokens: HashMap<String, u32>,
        rendered_added: HashMap<u32, String>,
        cache_capacity: usize,
        max_seq_len: usize,
    ) -> Result<Self, InferenceError> {
        let id_to_token = invert_vocab(&vocab)?;
        // Invert the renderable added-token set to id -> content for decode-side
        // lookup. Built once at construction; consulted by `token_for_id` only
        // when the base table misses (added-token ids exceed the base vocab range).
        let added_render: HashMap<u32, String> = rendered_added;
        // The complement of `added_render` within the full added-token set: ids
        // `added_tokens` declares (any `special` value) that did not make it into
        // the render set are exactly the ids that read as special, including a
        // non-boolean `"special"` value (`parse_rendered_added_tokens` fails that
        // closed to special). Mirrors `GemmaBpeTokenizer`'s `special_skip_ids`.
        let special_skip_ids: HashSet<u32> = added_tokens
            .values()
            .copied()
            .filter(|id| !added_render.contains_key(id))
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
            special_skip_ids,
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
            newline_boundary: OnceLock::new(),
        };

        let tokenizer = Self {
            inner: Arc::new(inner),
        };
        drop(tokenizer.newline_boundary_table());
        Ok(tokenizer)
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
            special_skip_ids: self.inner.special_skip_ids.clone(),
            pad_id: self.inner.pad_id,
            unk_id: self.inner.unk_id,
            bos_id: self.inner.bos_id,
            eos_id: self.inner.eos_id,
            add_bos: self.inner.add_bos,
            add_eos: self.inner.add_eos,
            pre_tokenize_mode: self.inner.pre_tokenize_mode,
            max_seq_len,
            cache: self.inner.cache.clone(),
            newline_boundary: self.inner.newline_boundary.clone(),
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
            special_skip_ids: self.inner.special_skip_ids.clone(),
            pad_id: self.inner.pad_id,
            unk_id,
            bos_id: self.inner.bos_id,
            eos_id: self.inner.eos_id,
            add_bos: self.inner.add_bos,
            add_eos: self.inner.add_eos,
            pre_tokenize_mode: self.inner.pre_tokenize_mode,
            max_seq_len: self.inner.max_seq_len,
            cache: self.inner.cache.clone(),
            newline_boundary: self.inner.newline_boundary.clone(),
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
            special_skip_ids: self.inner.special_skip_ids.clone(),
            pad_id: self.inner.pad_id,
            unk_id: self.inner.unk_id,
            bos_id: self.inner.bos_id,
            eos_id: self.inner.eos_id,
            add_bos: self.inner.add_bos,
            add_eos: true,
            pre_tokenize_mode: self.inner.pre_tokenize_mode,
            max_seq_len: self.inner.max_seq_len,
            cache: self.inner.cache.clone(),
            newline_boundary: self.inner.newline_boundary.clone(),
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
            special_skip_ids: self.inner.special_skip_ids.clone(),
            pad_id: self.inner.pad_id,
            unk_id: self.inner.unk_id,
            bos_id: self.inner.bos_id,
            eos_id: Some(eos_id),
            add_bos: self.inner.add_bos,
            add_eos: self.inner.add_eos,
            pre_tokenize_mode: self.inner.pre_tokenize_mode,
            max_seq_len: self.inner.max_seq_len,
            cache: self.inner.cache.clone(),
            newline_boundary: self.inner.newline_boundary.clone(),
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
            special_skip_ids: self.inner.special_skip_ids.clone(),
            pad_id: self.inner.pad_id,
            unk_id: self.inner.unk_id,
            bos_id: self.inner.bos_id,
            eos_id: self.inner.eos_id,
            add_bos: self.inner.add_bos,
            add_eos: self.inner.add_eos,
            pre_tokenize_mode: mode,
            max_seq_len: self.inner.max_seq_len,
            cache: self.inner.cache.clone(),
            newline_boundary: self.inner.newline_boundary.clone(),
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
        // An id `added_tokens` marks special stays swallowed even when it also
        // exists in the base `vocab` -- checked first so that case can never fall
        // through to the base-vocab spelling below (#1722). Added tokens with
        // `special=false` (think/tool/FIM markers) live beyond the base range, so
        // fall back to `added_render` to render them as text instead of dropping
        // them.
        if self.inner.special_skip_ids.contains(&id) {
            return None;
        }
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

    /// Look up a token ID by its exact rendered spelling (e.g. `"</think>"`,
    /// `"<|im_end|>"`), not by decoded/detokenized text. `content` must match
    /// the literal token string as it appears in the tokenizer's vocabulary or
    /// added-tokens table -- it is never partial text, a text fragment
    /// containing the token plus surrounding characters, or output that has
    /// passed through [`Self::token_for_id`] / decoding.
    ///
    /// Lookup precedence (first match wins):
    /// 1. `special_tokens` -- the fast-path table for tokens marked
    ///    `special: true` in `tokenizer.json` (e.g. `<|im_end|>`).
    /// 2. `vocab` -- the base BPE vocabulary (a `</think>`-style marker can
    ///    live here if the tokenizer declares it as an ordinary vocab entry
    ///    rather than a `special_tokens` entry or an added token).
    /// 3. `added_render` -- added tokens with `special: false` (e.g.
    ///    `<think>`/`</think>`, `<tool_call>`, FIM markers); this tier is an
    ///    O(added_render.len()) linear scan, the only one of the three.
    ///
    /// This is the single resolution path for the `</think>` reasoning-close
    /// marker: [`crate::model::qwen35::resolve_reasoning_close_token`] (the
    /// actual generation-time validation, shared by the CPU and Metal decode
    /// loops) and the Metal cross-turn prefix-cache fingerprint
    /// (`cross_turn_metadata`) both call this directly rather than each
    /// owning a separate cached/uncached copy of the lookup.
    ///
    /// EVERY matching id is returned, not the first, and the tiers make that
    /// distinction load-bearing rather than pedantic: `vocab` and
    /// `rendered_added` are independent constructor inputs with nothing
    /// reconciling them, and added-token ids deliberately live beyond the base
    /// vocabulary range, so one spelling can carry *different* ids in two tiers.
    /// A caller that took only the head would silently discard the rest, and the
    /// model could then emit a discarded alias that renders as the marker while
    /// an id comparison against the head says it did not. Collapsing to a single
    /// id is therefore only sound once a caller has established there is exactly
    /// one, which is why no single-id accessor exists here: the check and the
    /// lookup would be separable, and separable is how they drift.
    pub(crate) fn token_ids_for_content(&self, content: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = Vec::new();
        for id in self
            .inner
            .special_tokens
            .get(content)
            .into_iter()
            .chain(self.inner.vocab.get(content))
            .copied()
            .chain(
                self.inner
                    .added_render
                    .iter()
                    .filter(|(_, token)| token.as_str() == content)
                    .map(|(&id, _)| id),
            )
        {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        ids
    }

    /// **Unstable**: return the byte representation of every model token ID.
    ///
    /// `vocab_bytes(vocab_size)?[i]` is the byte sequence that token `i` decodes
    /// to. Base BPE entries use GPT-2 byte-level reversal; renderable added tokens
    /// use their literal UTF-8 bytes. Unknown and skipped control-token slots are
    /// empty so grammar masking fails closed across the model's full logit space --
    /// including a `special: true` added-token id that also exists in the base
    /// `vocab` (gets no bytes regardless of its base-vocab spelling), and a
    /// base-vocab token containing a character outside the 256-entry GPT-2
    /// byte-level alphabet (gets no bytes rather than a partially-decoded,
    /// possibly-colliding sequence) (#1722).
    ///
    /// Used by [`GrammarEngine::new`](crate::grammar::GrammarEngine::new) to build
    /// the precomputed vocabulary partition for grammar-constrained decoding (ADR-046).
    ///
    /// Cost: O(vocab_size) — called once at engine initialisation.
    pub fn vocab_bytes(&self, vocab_size: usize) -> Result<Vec<Vec<u8>>, InferenceError> {
        let required_vocab_size = self
            .inner
            .special_tokens
            .values()
            .chain(self.inner.added_render.keys())
            .copied()
            .max()
            .map_or(0usize, |id| id as usize + 1);
        let required_vocab_size = self.inner.id_to_token.len().max(required_vocab_size);
        if vocab_size < required_vocab_size {
            return Err(InferenceError::Tokenizer(format!(
                "model vocabulary size {vocab_size} cannot represent tokenizer token ID {}",
                required_vocab_size - 1
            )));
        }

        let byte_decoder = byte_decoder();
        let mut vocab_bytes = Vec::with_capacity(vocab_size);
        for id in 0..vocab_size {
            let mut bytes = Vec::new();
            self.append_token_bytes(id as u32, &byte_decoder, &mut bytes);
            vocab_bytes.push(bytes);
        }
        Ok(vocab_bytes)
    }

    /// **Unstable**: resolve one token ID to the exact bytes it emits.
    ///
    /// Base BPE entries are GPT-2 byte-decoded. Added tokens with `special=false`
    /// are returned as literal UTF-8, while unknown and skipped special IDs return
    /// `None` -- even when the special id also exists in the base `vocab`. A
    /// base-vocab entry containing a character outside the byte-level alphabet
    /// also returns `None` rather than a partial decode (#1722).
    pub fn token_bytes_for_id(&self, id: u32) -> Option<Vec<u8>> {
        let byte_decoder = byte_decoder();
        let mut bytes = Vec::new();
        self.append_token_bytes(id, &byte_decoder, &mut bytes)
            .then_some(bytes)
    }

    /// **Unstable**: per-id newline table for reasoning-close deferral.
    ///
    /// The table is built when the tokenizer is constructed and cached on its
    /// shared internals, so a cloned tokenizer observes the same table and a
    /// loaded tokenizer pays for it at most once. Ids outside the table are
    /// not consulted and therefore are not boundaries.
    pub(crate) fn newline_boundary_table(&self) -> Arc<[bool]> {
        self.inner
            .newline_boundary
            .get_or_init(|| {
                let byte_decoder = byte_decoder();
                let mut len = self.inner.id_to_token.len();
                if let Some(max_added) = self.inner.added_render.keys().max() {
                    let required = (*max_added as usize).saturating_add(1);
                    if required <= len.saturating_mul(16) {
                        len = len.max(required);
                    }
                }
                let mut table = vec![false; len];
                let mut bytes = Vec::new();
                for (id, boundary) in table.iter_mut().enumerate() {
                    bytes.clear();
                    if self.append_token_bytes(id as u32, &byte_decoder, &mut bytes) {
                        *boundary = bytes.contains(&b'\n');
                    }
                }
                Arc::from(table)
            })
            .clone()
    }

    pub(crate) fn append_token_bytes(
        &self,
        id: u32,
        byte_decoder: &HashMap<char, u8>,
        out: &mut Vec<u8>,
    ) -> bool {
        // An id `added_tokens` marks special must contribute no bytes even when
        // it also exists in the base `vocab` -- checked before the base-vocab
        // fallback below so that case can never leak the base-vocab spelling
        // into the grammar byte table or decoded output (#1722). This is the
        // single call site every consumer of token bytes goes through (the
        // grammar vocabulary in `vocab_bytes`, `token_bytes_for_id`, this
        // tokenizer's own `decode`, and the streaming detokenizer), so masking
        // here masks it everywhere uniformly.
        if self.inner.special_skip_ids.contains(&id) {
            return false;
        }
        if let Some(content) = self.inner.added_render.get(&id) {
            out.extend_from_slice(content.as_bytes());
            return true;
        }
        let Some(token_str) = self.inner.id_to_token.get(id as usize) else {
            return false;
        };
        // A base-vocab token containing a character outside the 256-entry
        // GPT-2 byte-level alphabet is masked in full rather than partially
        // decoded: `append_byte_decoded_token_bytes`'s `filter_map` drops only
        // the unmappable characters, which would let this token decode to
        // fewer bytes than its spelling implies and collide with an unrelated
        // token that decodes to that same shortened sequence (#1722). Masking
        // the whole token here -- exactly like a special id above -- closes
        // that gap without refusing tokenizers this loader already accepts:
        // non-byte-level test/synthetic vocabularies (e.g. a raw `" "` token)
        // construct and tokenize as before, they just never contribute bytes
        // to decode/grammar output for that one id.
        if !token_str.chars().all(|ch| byte_decoder.contains_key(&ch)) {
            return false;
        }
        append_byte_decoded_token_bytes(token_str, byte_decoder, out);
        true
    }

    /// Largest number of input-text bytes a single produced token can cover.
    ///
    /// Lets a caller lower-bound a token count from a raw byte length without
    /// tokenizing: no produced token spans more than this, so the token count is at
    /// least `ceil(bytes / span)`. Base vocabulary entries are measured by decoded
    /// byte length; special and added tokens are measured by their content string
    /// length instead, because `match_special` consumes the matched literal text
    /// while the token id itself may decode to fewer bytes than that (a control
    /// token marked special decodes to none at all).
    #[cfg(any(
        test,
        all(
            feature = "serve",
            feature = "f16",
            feature = "router-learning",
            target_os = "macos",
            feature = "metal-gpu"
        )
    ))]
    pub(crate) fn max_token_byte_span(&self) -> usize {
        let byte_decoder = byte_decoder();
        let mut span = 0usize;
        let mut bytes = Vec::new();
        for id in 0..self.inner.id_to_token.len() {
            bytes.clear();
            if self.append_token_bytes(id as u32, &byte_decoder, &mut bytes) {
                span = span.max(bytes.len());
            }
        }
        for content in self
            .inner
            .special_tokens
            .keys()
            .chain(self.inner.added_render.values())
        {
            span = span.max(content.len());
        }
        span
    }

    #[cfg(test)]
    fn tokenize_to_ids(&self, text: &str) -> Vec<u32> {
        let mut scratch = TokenizeScratch::default();
        self.tokenize_to_ids_into(text, &mut scratch).0
    }

    #[cfg(all(feature = "metal-gpu", feature = "serve"))]
    pub(crate) fn tokenize_fragments_with_inserted_ids(
        &self,
        before: &str,
        inserted_ids: &[u32],
        after: &str,
    ) -> Vec<u32> {
        let mut scratch = TokenizeScratch::default();
        scratch.ids.clear();
        if self.inner.add_bos
            && let Some(bos_id) = self.inner.bos_id
        {
            scratch.ids.push(bos_id);
        }
        self.tokenize_text_into(before, &mut scratch);
        scratch.ids.extend_from_slice(inserted_ids);
        self.tokenize_text_into(after, &mut scratch);
        if self.inner.add_eos
            && let Some(eos_id) = self.inner.eos_id
        {
            scratch.ids.push(eos_id);
        }
        scratch.ids
    }

    /// Tokenizes into `scratch.ids`, returning the resulting ids and the
    /// pre-truncation token count (equal to `ids.len()` unless truncation to
    /// `max_seq_len` was applied, in which case it reports the larger,
    /// pre-truncation count -- including the EOS token when `add_eos` adds one).
    fn tokenize_to_ids_into(&self, text: &str, scratch: &mut TokenizeScratch) -> (Vec<u32>, usize) {
        scratch.ids.clear();
        if self.inner.add_bos
            && let Some(bos_id) = self.inner.bos_id
        {
            scratch.ids.push(bos_id);
        }

        self.tokenize_text_into(text, scratch);

        let pre_truncation_len = if self.inner.add_eos && self.inner.eos_id.is_some() {
            scratch.ids.len() + 1
        } else {
            scratch.ids.len()
        };

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

        (scratch.ids.clone(), pre_truncation_len)
    }

    fn tokenize_text_into(&self, text: &str, scratch: &mut TokenizeScratch) {
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

            let Some(ch) = text[pos..].chars().next() else {
                break;
            };
            pos += ch.len_utf8();
        }

        if segment_start < text.len() {
            self.tokenize_regular_segment_into(&text[segment_start..], scratch);
        }
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
        let mut scratch = TokenizeScratch::default();
        let (ids, pre_truncation_len) = self.tokenize_to_ids_into(text, &mut scratch);
        pad_ids(
            ids,
            self.inner.max_seq_len,
            self.inner.pad_id,
            pre_truncation_len,
        )
    }

    fn tokenize_batch(&self, texts: &[&str]) -> Vec<TokenizedInput> {
        if texts.is_empty() {
            return Vec::new();
        }
        let mut scratch = TokenizeScratch::default();
        let mut max_len = 0usize;
        let mut all = Vec::with_capacity(texts.len());
        for text in texts {
            let (ids, pre_truncation_len) = self.tokenize_to_ids_into(text, &mut scratch);
            max_len = max_len.max(ids.len());
            all.push((ids, pre_truncation_len));
        }
        all.into_iter()
            .map(|(ids, pre_truncation_len)| {
                pad_ids(ids, max_len, self.inner.pad_id, pre_truncation_len)
            })
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> Option<String> {
        let byte_decoder = byte_decoder();
        let mut bytes = Vec::new();
        for &id in ids {
            self.append_token_bytes(id, &byte_decoder, &mut bytes);
        }
        Some(String::from_utf8_lossy(&bytes).into_owned())
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

/// Shared with the additive Gemma BPE path (`tokenizer::gemma_bpe`), which
/// parses the same `tokenizer.json` `model.merges` shape but builds its own
/// merge-rank table and pretokenization pipeline; see ADR-082 G17.
pub(crate) fn parse_merges_json(
    value: &JsonValue,
) -> Result<Vec<(String, String)>, InferenceError> {
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

#[expect(
    clippy::expect_used,
    reason = "the byte-to-unicode table is built from a fixed list of valid scalar values"
)]
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
/// Reverse the GPT-2 byte-level encoding of a single BPE token string back to
/// its original raw bytes.
///
/// Unlike [`byte_decode_token`], this returns the exact bytes with no UTF-8
/// re-encoding step, so it is lossless even for a token that represents only
/// part of a multi-byte codepoint (common for byte-level BPE on CJK/emoji
/// text). Used by the `logprobs`/`top_logprobs` response `bytes` field
/// (#585), where callers reconstruct the original output byte-for-byte.
pub fn byte_decode_token_bytes(token_str: &str) -> Vec<u8> {
    let decoder = byte_decoder();
    let mut bytes = Vec::new();
    append_byte_decoded_token_bytes(token_str, &decoder, &mut bytes);
    bytes
}

fn byte_decoder() -> HashMap<char, u8> {
    bytes_to_unicode()
        .into_iter()
        .enumerate()
        .map(|(byte, ch)| (ch, byte as u8))
        .collect()
}

fn append_byte_decoded_token_bytes(
    token_str: &str,
    byte_decoder: &HashMap<char, u8>,
    out: &mut Vec<u8>,
) {
    out.extend(
        token_str
            .chars()
            .filter_map(|ch| byte_decoder.get(&ch).copied()),
    );
}

/// **Unstable**: byte-decode helper; encoding table and fallback behavior may change.
///
/// Decode a byte-encoded BPE token string back to UTF-8.
///
/// Lossy for a token that represents only part of a multi-byte codepoint —
/// see [`byte_decode_token_bytes`] for the byte-exact form.
pub fn byte_decode_token(token_str: &str) -> String {
    String::from_utf8_lossy(&byte_decode_token_bytes(token_str)).to_string()
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
/// parser cannot honestly model (#330 finding 1).
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
///   `Split` does not honestly dominate later pre-tokenizers (#330 residual
///   finding).
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
/// `Gpt4Regex` (#330 residual finding).
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
mod tests;
