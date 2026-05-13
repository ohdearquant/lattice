# ADR-005: Tokenizer Architecture

**Status**: Accepted\
**Date**: 2026-05-13\
**Crate**: `lattice-inference`

---

## Context

The crate supports three model families (BERT/BGE, Qwen3, and cross-encoders) that use fundamentally different tokenization algorithms:

- **BERT/BGE**: WordPiece with `##` continuation tokens, `[CLS]`/`[SEP]`/`[PAD]` special tokens, max 512 tokens.
- **Qwen3**: Byte-Pair Encoding (BPE) with a 151,669-token vocabulary and a byte-fallback scheme.
- **SentencePiece** models: Unigram or BPE via the SentencePiece protocol, used by several BERT variants.

Each tokenizer has a different vocabulary format, different inference algorithm, and different pre-tokenization (whitespace splitting, punctuation handling, Unicode normalization).

The `Tokenizer` trait in `src/tokenizer/common.rs`:

```rust
pub trait Tokenizer: Send + Sync {
    fn tokenize(&self, text: &str) -> TokenizedInput;
    fn tokenize_batch(&self, texts: &[&str]) -> Vec<TokenizedInput>;
    fn vocab_size(&self) -> usize;
    fn max_seq_len(&self) -> usize;
    fn supports_pair_tokenization(&self) -> bool;
    fn encode_pair(&self, text_a: &str, text_b: &str) -> TokenizedInput;
}
```

`TokenizedInput`:

```rust
pub struct TokenizedInput {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub token_type_ids: Vec<u32>,  // used by BERT cross-encoders
    pub real_length: usize,
}
```

`WordPieceTokenizer` in `src/tokenizer/wordpiece.rs`:

- Uses `DoubleArrayTrie` (two tries: one for whole-word tokens, one for `##` continuations)
- `char_to_label: HashMap<u32, i32>` maps Unicode code points to trie labels
- `ThreadSafeLruCache` with capacity 8,192 caches tokenized strings
- `DEFAULT_MAX_SEQ_LEN = 512`

`load_tokenizer()` in `src/tokenizer/common.rs` auto-detects which tokenizer to instantiate from the model directory contents (e.g., presence of `tokenizer.json` vs `sentencepiece.bpe.model`).

---

## Decision

Define a **`Tokenizer` trait with three concrete implementations** (`WordPieceTokenizer`, `BpeTokenizer`, `SentencePieceTokenizer`) behind a common boxed interface. A factory function `load_tokenizer()` detects the model type from filesystem artifacts and returns `Box<dyn Tokenizer>`. The trait is `Send + Sync` to support concurrent tokenization in embedding servers.

---

## Key Design Choices

1. **`DoubleArrayTrie` for WordPiece**: WordPiece requires longest-match prefix lookup in the full vocabulary and a separate longest-match lookup in the `##`-continuation vocabulary. A hash map lookup per character would be O(word_length × vocab_size) worst case. The double-array trie reduces this to O(word_length) with excellent cache locality (two contiguous integer arrays).
2. **Per-instance LRU cache (8,192 entries)**: Tokenization of natural-language text has high repetition: common words ("the", "of", "model") appear in nearly every input. Caching at the string level before trie lookup avoids re-traversal for hot tokens. Capacity 8,192 covers the "80/20" vocabulary in most English text workloads.
3. **`trait Tokenizer: Send + Sync`**: Embedding servers call tokenization from request handlers, potentially on multiple threads. The trait bound makes this safe without `Mutex<Box<dyn Tokenizer>>` wrappers.
4. **`token_type_ids` in `TokenizedInput`**: BERT cross-encoders require segment IDs (0 for sentence A, 1 for sentence B) to distinguish the two passages in a pair. Other tokenizers produce all-zero `token_type_ids`. Including the field in the shared struct avoids a separate type.
5. **`load_tokenizer()` factory**: Callers should not need to know which tokenizer a model uses. The factory inspects the model directory and constructs the appropriate implementation. This decouples model loading (`BertModel`, `QwenModel`) from tokenizer selection.

---

## Alternatives Considered

| Alternative                                             | Pros                                      | Cons                                                                               | Why Not                                                                                   |
| ------------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `HuggingFace tokenizers` crate                          | Battle-tested; covers all tokenizer types | Python-origin C extension via FFI; adds non-Rust dependency; binary size           | No external tokenizer runtime; keep pure Rust                                             |
| Hash map vocabulary lookup                              | Simple implementation                     | O(1) per lookup but requires splitting token candidates manually; no prefix search | DoubleArrayTrie enables maximal-munch tokenization natively                               |
| Enum dispatch (`TokenizerKind`) instead of trait object | Monomorphization; no vtable               | Adding a fourth tokenizer requires changing the enum and all match arms            | Open/closed: trait object allows new tokenizers without modifying call sites              |
| No caching                                              | Simpler                                   | Re-traverses trie for every occurrence of common words                             | Trie traversal for "##ing" is ~10 μs; cached hit is ~100 ns; 100× for common tokens       |
| Shared global cache                                     | Single cache for all instances            | Cache invalidation on model reload; contention under concurrent calls              | Per-instance cache avoids cross-model interference and needs no lock on the critical path |

---

## Consequences

**Positive**:

- Model code calls `tokenizer.tokenize(text)` with no knowledge of the algorithm.
- Embedding server can hold `Arc<dyn Tokenizer>` and call from any thread without additional locking.
- LRU cache reduces tokenization latency by ~100× for repeated common tokens.
- New tokenizer algorithms (e.g., Tiktoken) can be added as new `impl Tokenizer` without touching existing code.

**Negative**:

- `Box<dyn Tokenizer>` adds one vtable indirection per call. For embedding workloads where tokenization is <5% of total latency, this is acceptable.
- Maintaining three separate tokenizer implementations for correctness edge cases (Unicode combining characters, byte-fallback in BPE) requires ongoing testing.

**Risks**:

- The `DoubleArrayTrie` construction is not documented publicly (the build process lives outside this crate). If the vocabulary file format changes, the trie must be rebuilt.
- `ThreadSafeLruCache` uses interior mutability. Its implementation must be audited when the cache eviction policy changes.

---

## References

- `src/tokenizer/common.rs` — `Tokenizer` trait, `TokenizedInput`, `load_tokenizer()`
- `src/tokenizer/wordpiece.rs` — `WordPieceTokenizer`, `DoubleArrayTrie`, `ThreadSafeLruCache`
- `src/tokenizer/bpe.rs` — `BpeTokenizer` (Qwen3)
- `src/tokenizer/sentencepiece.rs` — `SentencePieceTokenizer`
- `src/tokenizer/mod.rs` — re-exports
