//! Gemma 4 BPE tokenizer: literal-space `Split` pre-tokenizer + `▁` (U+2581)
//! metaspace-style normalizer, per the target `tokenizer.json` read for
//! ADR-082 G17.
//!
//! This is an **additive, explicitly-selected** path — `GemmaBpeTokenizer`
//! is never reached through [`crate::tokenizer::common::load_tokenizer`] or
//! [`crate::tokenizer::common::tokenizer_from_json_str`]; callers construct
//! it directly. The existing Qwen-oriented [`crate::tokenizer::BpeTokenizer`]
//! loader's `validate_bpe_pretokenizer` continues to reject Gemma's
//! `tokenizer.json` (a literal-string `Split`, not a regex `Split`/
//! `ByteLevel`), which proves this path is additive rather than a silent
//! widening of the existing loader — see
//! `tests/gemma4_tokenizer_goldens_test.rs`.
//!
//! Pipeline, read directly from the target `tokenizer.json`
//! (`google/gemma-4-E2B-it` @ `9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`):
//!
//! - `normalizer`: `Replace(" " -> "▁")` — a literal replace, with no
//!   dummy-prefix insertion and no leading/trailing-whitespace stripping
//!   (unlike this crate's SentencePiece Unigram path).
//! - `pre_tokenizer`: `Split(pattern=" ", behavior=MergedWithPrevious)` — a
//!   literal-string split. Because the normalizer has already consumed every
//!   space, this never fires in practice: the whole normalized string is one
//!   pre-token, matching the empirically observed behavior (`"a  b   c"` ->
//!   `["a", "▁▁", "b", "▁▁▁", "c"]`, i.e. the BPE merge algorithm — not
//!   pre-tokenization — decides where multi-`▁` runs group).
//! - `model`: `BPE` with `byte_fallback: true`, `fuse_unk: true`, no
//!   `continuing_subword_prefix`/`end_of_word_suffix`. Initial symbols are
//!   per-Unicode-scalar: a character present in `vocab` as its own token
//!   starts as that token; otherwise its UTF-8 bytes each become individual
//!   `<0xXX>` byte-fallback tokens (verified: `\u{10300}` -> four
//!   single-byte tokens). No merge rule ever pairs two byte-fallback tokens
//!   in the target vocab, so once introduced they never re-merge.
//! - `decoder`: `Sequence[Replace("▁" -> " "), ByteFallback, Fuse]`.

use crate::error::InferenceError;
use crate::tokenizer::bpe::parse_merges_json;
use crate::tokenizer::common::{
    JsonValue, TokenizedInput, Tokenizer, invert_vocab, json_object_to_vocab, json_path,
    known_special_id, pad_ids, parse_added_tokens, parse_json, parse_rendered_added_tokens,
};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// U+2581 LOWER ONE EIGHTH BLOCK, used by SentencePiece-descended tokenizers
/// as a metaspace marker for the literal ASCII space.
const METASPACE: char = '\u{2581}';
const DEFAULT_MAX_SEQ_LEN: usize = 4_096;

/// **Unstable**: additive Gemma-family byte-level BPE tokenizer; API evolving.
/// See the module docs for why this is a separate type from
/// [`crate::tokenizer::BpeTokenizer`].
#[derive(Debug, Clone)]
pub struct GemmaBpeTokenizer {
    inner: Arc<GemmaBpeInner>,
}

#[derive(Debug, Clone)]
struct GemmaBpeInner {
    vocab: HashMap<String, u32>,
    id_to_token: Vec<String>,
    merges: HashMap<String, HashMap<String, usize>>,
    special_tokens: HashMap<String, u32>,
    special_tokens_sorted: Vec<String>,
    /// Rendered decode text for added tokens with `special=false`. Empty for
    /// the target checkpoint (all 24 added tokens are `special=true`), kept
    /// for shape parity with the general loading contract.
    added_render: HashMap<u32, String>,
    /// Added-token ids with `special=true`: dropped on decode, matching
    /// `skip_special_tokens=True` (HF `tokenizers` default).
    special_skip_ids: HashSet<u32>,
    pad_id: u32,
    unk_id: Option<u32>,
    max_seq_len: usize,
}

#[derive(Debug, Clone)]
struct MergeNode {
    token: String,
    prev: Option<usize>,
    next: Option<usize>,
    alive: bool,
    version: u64,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct MergeCandidate {
    rank: usize,
    left: usize,
    right: usize,
    left_version: u64,
    right_version: u64,
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
    }
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl GemmaBpeTokenizer {
    /// **Unstable**: load from a Hugging Face `tokenizer.json` file.
    /// Selected explicitly by the caller — never reached via
    /// [`crate::tokenizer::common::load_tokenizer`]'s model-type sniffing.
    pub fn from_tokenizer_json(path: &Path) -> Result<Self, InferenceError> {
        let text = fs::read_to_string(path).map_err(|e| {
            InferenceError::Tokenizer(format!("failed to read {}: {e}", path.display()))
        })?;
        Self::from_tokenizer_json_str(&text)
    }

    /// **Unstable**: load from a Hugging Face `tokenizer.json` string.
    ///
    /// Fails closed if the declared `model.type`, `normalizer`, or
    /// `pre_tokenizer` do not match the Gemma shape this path was built
    /// against (see module docs) — this constructor makes an explicit
    /// structural claim about its input, rather than best-effort adapting.
    pub fn from_tokenizer_json_str(text: &str) -> Result<Self, InferenceError> {
        let root = parse_json(text)?;
        validate_gemma_bpe_shape(&root)?;

        let vocab =
            json_object_to_vocab(json_path(&root, &["model", "vocab"]).ok_or_else(|| {
                InferenceError::Tokenizer("tokenizer.json missing model.vocab".into())
            })?)?;
        let id_to_token = invert_vocab(&vocab)?;

        let merges_value = json_path(&root, &["model", "merges"]).ok_or_else(|| {
            InferenceError::Tokenizer("tokenizer.json missing model.merges".into())
        })?;
        let merges_list = parse_merges_json(merges_value)?;
        let mut merge_ranks: HashMap<String, HashMap<String, usize>> = HashMap::new();
        for (rank, (left, right)) in merges_list.into_iter().enumerate() {
            merge_ranks
                .entry(left)
                .or_default()
                .entry(right)
                .or_insert(rank);
        }

        let mut added = parse_added_tokens(&root);
        added.retain(|name, _| !name.is_empty());
        let rendered_added = parse_rendered_added_tokens(&root);
        let added_render: HashMap<u32, String> = rendered_added
            .iter()
            .map(|(content, &id)| (id, content.clone()))
            .collect();
        let rendered_ids: HashSet<u32> = rendered_added.values().copied().collect();
        let special_skip_ids: HashSet<u32> = added
            .values()
            .copied()
            .filter(|id| !rendered_ids.contains(id))
            .collect();

        let mut special_tokens = added;
        for name in ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"] {
            if let Some(&id) = vocab.get(name) {
                special_tokens.entry(name.to_string()).or_insert(id);
            }
        }
        let mut special_tokens_sorted: Vec<String> = special_tokens.keys().cloned().collect();
        special_tokens_sorted.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));

        let pad_id = known_special_id(&vocab, &["<pad>"]).unwrap_or(0);
        let unk_id = known_special_id(&vocab, &["<unk>"]);

        Ok(Self {
            inner: Arc::new(GemmaBpeInner {
                vocab,
                id_to_token,
                merges: merge_ranks,
                special_tokens,
                special_tokens_sorted,
                added_render,
                special_skip_ids,
                pad_id,
                unk_id,
                max_seq_len: DEFAULT_MAX_SEQ_LEN,
            }),
        })
    }

    /// **Unstable**: override the configured maximum sequence length.
    pub fn with_max_seq_len(self, max_seq_len: usize) -> Self {
        let mut inner = (*self.inner).clone();
        inner.max_seq_len = max_seq_len;
        Self {
            inner: Arc::new(inner),
        }
    }

    fn tokenize_to_ids(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut segment_start = 0usize;
        let mut pos = 0usize;
        while pos < text.len() {
            if let Some((special_end, special_id)) = self.match_special(text, pos) {
                if segment_start < pos {
                    self.encode_segment(&text[segment_start..pos], &mut ids);
                }
                ids.push(special_id);
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
            self.encode_segment(&text[segment_start..], &mut ids);
        }
        if ids.len() > self.inner.max_seq_len {
            ids.truncate(self.inner.max_seq_len);
        }
        ids
    }

    fn match_special(&self, text: &str, pos: usize) -> Option<(usize, u32)> {
        let tail = &text[pos..];
        for token in &self.inner.special_tokens_sorted {
            if tail.starts_with(token.as_str())
                && let Some(&id) = self.inner.special_tokens.get(token)
            {
                return Some((pos + token.len(), id));
            }
        }
        None
    }

    /// Normalize (space -> ▁), split into initial BPE symbols (char, or
    /// per-byte fallback for a char absent from `vocab`), then merge.
    fn encode_segment(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let mut symbols: Vec<String> = Vec::with_capacity(text.len());
        for ch in text.chars() {
            let mapped = if ch == ' ' { METASPACE } else { ch };
            let single = mapped.to_string();
            if self.inner.vocab.contains_key(single.as_str()) {
                symbols.push(single);
                continue;
            }
            let mut buf = [0u8; 4];
            for &byte in mapped.encode_utf8(&mut buf).as_bytes() {
                symbols.push(byte_fallback_token(byte));
            }
        }

        for token in self.bpe_merge(symbols) {
            if let Some(&id) = self.inner.vocab.get(token.as_str()) {
                out.push(id);
            } else if let Some(unk_id) = self.inner.unk_id {
                out.push(unk_id);
            }
        }
    }

    fn merge_rank(&self, left: &str, right: &str) -> Option<usize> {
        self.inner
            .merges
            .get(left)
            .and_then(|inner| inner.get(right))
            .copied()
    }

    fn push_candidate(
        &self,
        nodes: &[MergeNode],
        heap: &mut BinaryHeap<MergeCandidate>,
        left: usize,
    ) {
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
        heap.push(MergeCandidate {
            rank,
            left,
            right,
            left_version: nodes[left].version,
            right_version: nodes[right].version,
        });
    }

    /// Same rank-priority-queue merge algorithm as
    /// [`crate::tokenizer::bpe::BpeTokenizer::bpe_merge`], adapted to merge
    /// pre-built symbol strings (chars or `<0xXX>` byte-fallback tokens)
    /// instead of `chars()` of a byte-remapped word — Gemma's vocab holds
    /// literal UTF-8 pieces, not GPT-2 byte-remapped ones, so a byte
    /// fallback symbol is multiple bytes wide and can't be a `char`.
    fn bpe_merge(&self, symbols: Vec<String>) -> Vec<String> {
        let mut nodes: Vec<MergeNode> = symbols
            .into_iter()
            .enumerate()
            .map(|(idx, token)| MergeNode {
                token,
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

    fn token_bytes_for(&self, id: u32, out: &mut Vec<u8>) {
        if self.inner.special_skip_ids.contains(&id) {
            return;
        }
        if let Some(content) = self.inner.added_render.get(&id) {
            out.extend_from_slice(content.as_bytes());
            return;
        }
        let Some(tok) = self.inner.id_to_token.get(id as usize) else {
            return;
        };
        if tok.is_empty() {
            return;
        }
        if let Some(byte) = parse_byte_fallback_token(tok) {
            out.push(byte);
            return;
        }
        for ch in tok.chars() {
            if ch == METASPACE {
                out.push(b' ');
            } else {
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
}

impl Tokenizer for GemmaBpeTokenizer {
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

    fn decode(&self, ids: &[u32]) -> Option<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            self.token_bytes_for(id, &mut bytes);
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

fn byte_fallback_token(byte: u8) -> String {
    format!("<0x{byte:02X}>")
}

/// Parses a `<0xXX>` byte-fallback token back into its raw byte, or `None`
/// if `tok` is not in that exact shape (2 uppercase hex digits).
fn parse_byte_fallback_token(tok: &str) -> Option<u8> {
    let hex = tok.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// Fails closed if `root` does not match the Gemma tokenizer shape this
/// path was built against (ADR-082 G17): `model.type == "BPE"`,
/// `model.byte_fallback == true`, a `Replace(" " -> "▁")` normalizer, and a
/// literal-string `Split` pre-tokenizer (not a regex `Split`, which is the
/// shape the existing Qwen loader accepts).
fn validate_gemma_bpe_shape(root: &JsonValue) -> Result<(), InferenceError> {
    let model_type = json_path(root, &["model", "type"])
        .and_then(JsonValue::as_str)
        .unwrap_or("");
    if model_type != "BPE" {
        return Err(InferenceError::Tokenizer(format!(
            "GemmaBpeTokenizer expects tokenizer.json model.type == \"BPE\", found {model_type:?}"
        )));
    }

    let byte_fallback = json_path(root, &["model", "byte_fallback"])
        .and_then(JsonValue::as_bool)
        .unwrap_or(false);
    if !byte_fallback {
        return Err(InferenceError::Tokenizer(
            "GemmaBpeTokenizer expects tokenizer.json model.byte_fallback == true".into(),
        ));
    }

    let normalizer = root.get("normalizer");
    let is_gemma_normalizer = normalizer.is_some_and(|n| {
        n.get("type").and_then(JsonValue::as_str) == Some("Replace")
            && json_path(n, &["pattern", "String"]).and_then(JsonValue::as_str) == Some(" ")
            && n.get("content").and_then(JsonValue::as_str) == Some("\u{2581}")
    });
    if !is_gemma_normalizer {
        return Err(InferenceError::Tokenizer(
            "GemmaBpeTokenizer expects a Replace(\" \" -> \"\u{2581}\") normalizer; \
             tokenizer.json declares a different shape, refusing to guess"
                .into(),
        ));
    }

    let pre_tokenizer = root.get("pre_tokenizer");
    let is_gemma_pretokenizer = pre_tokenizer.is_some_and(|pt| {
        pt.get("type").and_then(JsonValue::as_str) == Some("Split")
            && json_path(pt, &["pattern", "String"]).is_some()
    });
    if !is_gemma_pretokenizer {
        return Err(InferenceError::Tokenizer(
            "GemmaBpeTokenizer expects a literal-string Split pre_tokenizer \
             (pattern.String, not pattern.Regex); tokenizer.json declares a \
             different shape, refusing to guess"
                .into(),
        ));
    }

    Ok(())
}
