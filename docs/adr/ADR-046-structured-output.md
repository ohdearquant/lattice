# ADR-046: XGrammar Structured Output Engine

**Status**: Proposed
**Date**: 2026-05-19
**Crate**: lattice-inference

## Context

Agent and tool-call workflows require the model to emit syntactically valid output (JSON schema,
function signatures, regex patterns). Without constrained decoding, the caller must either
post-process malformed output (fragile) or retry the full generation (expensive). Every major
serving framework treats this as a baseline feature: llama.cpp ships GBNF + JSON Schema, Ollama
exposes `format=` with Pydantic/JSON schema, SGLang and vLLM both default to XGrammar as of early
2026, TRT-LLM ships an equivalent implementation.

Lattice currently exposes no grammar-constrained generation interface. Without it, the
`generate()` / `MetalQwen35State::generate_greedy` / `generate_sampled` paths are not usable in
any agent pipeline that requires structured tool responses.

**The XGrammar approach** (MLSys 2025, arxiv:2411.15100) is now the canonical algorithm. Its key
insight is vocabulary partitioning: at grammar initialisation time, every token in the vocabulary
is inspected against the context-free grammar and placed into one of two classes:

- **Context-independent** (~99% of tokens): whether the token is legal depends only on the current
  grammar state, not on the partially accumulated byte sequence within the current token. These
  produce a fully precomputed bitmask per grammar state — one bit per vocabulary position.
- **Context-dependent** (~1% of tokens): legality requires inspecting the runtime pushdown-automaton
  stack (typically tokens that straddle a grammar boundary mid-byte). These are checked at decode
  time via stack inspection.

This partitioning means the per-token overhead is dominated by a bitmask AND against the top-k
logits — measured at under 40 µs on modern hardware. The grammar automaton itself is a byte-level
pushdown automaton (PDA) that handles full context-free grammars, not just regular expressions.

Lattice targets Apple Silicon exclusively. The bitmask apply step fits naturally into the existing
CPU-side logit processing that already happens between the GPU forward pass and `Sampler::sample`.

## Decision

Add a `grammar` module to `lattice-inference` implementing a byte-level pushdown automaton
structured output engine, following the XGrammar vocabulary-partitioning design.

The module is model-agnostic: it operates on the raw logit slice `[vocab_size]` and is independent
of whether the compute backend is CPU, NEON, or Metal GPU.

**Grammar specification accepts two input formats**:

1. **JSON Schema** (primary): the dominant format for agent tool-call schemas; auto-converted to
   the internal grammar representation at initialisation time.
2. **GBNF** (secondary): llama.cpp-compatible format; gives escape hatch for custom grammars that
   are not expressible as JSON Schema.

Both are accepted via a single `GrammarSpec` enum so callers have one entry point.

**Precomputation happens at `GrammarEngine::new` time** (process startup or per-request if the
schema is dynamic). No build-time codegen is required. For static schemas, callers may cache the
`GrammarEngine` across requests.

**Logit masking hooks into `GenerateConfig`**: a new optional field `grammar:
Option<Arc<GrammarEngine>>` is added. The decode loop applies `engine.mask_logits(state, logits)`
immediately before `sampler.sample(logits)` on every step. The `mask_logits` call zeroes
(sets to `f32::NEG_INFINITY`) all disallowed token positions in the logit slice.

## Scope

**v0 (this ADR)**:
- `GrammarSpec`: `JsonSchema(serde_json::Value)` and `Gbnf(String)` variants
- `GrammarEngine`: vocabulary analysis, bitmask precomputation, per-step state machine
- `GrammarState`: runtime PDA stack + current grammar position, cloned per decode sequence
- Integration with `GenerateConfig` in `generate.rs` (CPU path)
- Integration with `Qwen35Config::GenerateConfig` in `metal_qwen35.rs` (Metal path)
- JSON Schema support limited to: `object`, `array`, `string` (with `enum`), `number`,
  `integer`, `boolean`, `null`, `anyOf`/`oneOf`, nested `$ref` resolved within the same document

**Deferred**:
- External `$ref` resolution (URI references) — requires HTTP client, out of scope for inference
- Regex pattern constraints on `string` fields (`"pattern": "..."`)
- Streaming grammar state export for speculative decoding integration
- Separate `lattice-grammar` crate — only warranted if other crates need grammar without pulling
  all of `lattice-inference`
- WASM compilation of the grammar engine

## Architecture

### Core Types

```rust
// grammar/spec.rs
pub enum GrammarSpec {
    JsonSchema(serde_json::Value),
    Gbnf(String),
}

// grammar/engine.rs
pub struct GrammarEngine {
    /// Compiled grammar: states → terminal rules.
    grammar: CompiledGrammar,
    /// vocab_size × n_states bitmask table for context-independent tokens.
    /// Layout: masks[state_id * mask_stride .. state_id * mask_stride + mask_stride]
    /// where mask_stride = vocab_size.roundup_to(64) / 64 (u64 words).
    masks: Vec<u64>,
    mask_stride: usize,
    vocab_size: usize,
    /// Vocabulary byte strings for context-dependent token inspection.
    vocab_bytes: Vec<Vec<u8>>,
}

impl GrammarEngine {
    /// Build from a grammar specification and the model's vocabulary.
    /// `vocab_bytes[i]` is the UTF-8 byte sequence for token i.
    /// Cost: O(vocab_size × grammar_states), runs once at init time.
    pub fn new(spec: &GrammarSpec, vocab_bytes: Vec<Vec<u8>>) -> Result<Self, GrammarError>;

    /// Apply grammar constraints to `logits` in-place.
    /// Sets disallowed token positions to `f32::NEG_INFINITY`.
    /// Cost: O(vocab_size / 64) for the bitmask AND path;
    ///       O(k × stack_depth) for the context-dependent token re-check.
    pub fn mask_logits(&self, state: &mut GrammarState, logits: &mut [f32]);
}

// grammar/state.rs
pub struct GrammarState {
    /// Current grammar state index.
    grammar_pos: usize,
    /// PDA stack for context-free grammar traversal.
    stack: Vec<StackFrame>,
    /// Accumulated bytes in the current token (for context-dependent tokens).
    partial_token_bytes: Vec<u8>,
}
```

### Integration Points

**`generate.rs` (CPU path)**: `GenerateConfig` gains:

```rust
pub grammar: Option<Arc<GrammarEngine>>,
```

In the decode loop, after `forward_with_cache` returns `logits: &[f32]`, the masking call happens
before sampling:

```rust
if let Some(engine) = &config.grammar {
    engine.mask_logits(&mut grammar_state, scratch_logits_mut);
}
let token = sampler.sample(logits);
```

The logits slice is borrowed from `scratch.logits`. The mask call requires a mutable reference to
the same slice, so `forward_with_cache` will need to return `&mut [f32]` or the caller will
copy to a scratch mutable buffer. The cheaper option is to operate on `scratch.logits` directly
before the borrow is passed to `sampler.sample`.

**`metal_qwen35.rs` (Metal path)**: `Qwen35Config::GenerateConfig` gains the same `grammar` field.
After the GPU readback phase returns logits to CPU memory, the identical mask call applies. The
Metal path already copies logits to CPU for sampling — no additional transfer is needed.

**`BpeTokenizer`**: Used at `GrammarEngine::new` time to extract `vocab_bytes`. The tokenizer's
vocabulary iterator will be accessed once; no ongoing coupling.

### Bitmask Precomputation Algorithm

For each grammar state `s` and each token `t`:
1. Obtain the token's byte string from `vocab_bytes[t]`.
2. Simulate advancing the grammar PDA from state `s` by consuming the bytes of `t`.
3. If the simulation succeeds (reaches a valid intermediate or terminal state), set bit `t` in
   `masks[s * mask_stride + t/64]`.
4. If the token straddles a grammar boundary (simulation reaches EOF mid-byte-sequence), mark
   as context-dependent for runtime stack inspection.

The simulation uses a deterministic finite automaton (DFA) compiled from the grammar's regular
prefix — the portion expressible without recursion. Recursive rules (object fields, array items)
are handled by the PDA stack at runtime.

### Mask Apply (Hot Path)

```rust
// Apply bitmask: iterate mask words, set disallowed logits to NEG_INFINITY.
// The bitmask is a separate bitset — it is NOT overlaid on the float buffer.
let mask_base = state.grammar_pos * self.mask_stride;
for word_idx in 0..self.mask_stride {
    let mask_word = self.masks[mask_base + word_idx];
    for bit in 0..64u32 {
        let token_idx = word_idx * 64 + bit as usize;
        if token_idx >= logits.len() { break; }
        if mask_word & (1u64 << bit) == 0 {
            logits[token_idx] = f32::NEG_INFINITY; // disallowed
        }
        // allowed tokens: logits unchanged
    }
}
```

On a vocab of 248,320 tokens (Qwen3), `mask_stride = ceil(248320/64) = 3880` u64 words.
The outer loop is 3880 iterations; the inner bit-scan can be SIMD-vectorized (e.g., `_mm256_movemask_epi8`
+ batch store) but the scalar version is already well under 40 µs.

## Alternatives Considered

**1. Port llama.cpp GBNF engine directly**
llama.cpp's grammar engine (`common/grammar-parser.cpp`) is a regex-level NFA, not a full PDA.
It handles context-free grammars via stack simulation but lacks the vocabulary-partitioning
optimization that keeps XGrammar's overhead under 40 µs. It also requires C FFI, which conflicts
with the pure-Rust constraint.

**2. Integrate Outlines (Python)**
Outlines is the reference Python implementation of the index-based structured generation approach.
MLX-LM uses it. It requires Python runtime and is therefore incompatible with Lattice's pure-Rust,
no-subprocess architecture.

**3. Regular expressions only (no full CFG)**
JSON is not a regular language. Regex-only masking (e.g., `jsonschema-rs` + regex compilation)
fails on recursive structures: nested objects, arrays of arrays. Rejected for correctness reasons.

**4. Post-hoc JSON repair**
Parse the raw output; if invalid, retry. Acceptable for low-stakes pipelines. Unacceptable for
agent tool-calls where the caller blocks on a valid response. Adds 100-500 ms per failure.

**5. Defer to a separate `lattice-grammar` crate**
Appropriate once more than one consumer needs grammar, or if crate size becomes a concern.
For v0, adding to `lattice-inference` avoids a new publish/version coordination cycle. The
module boundary is clean enough to extract later without breaking the public API.

## Risks

**R1: Bitmask table memory**
Qwen3 vocab = 248,320 tokens. A grammar with 500 states produces a table of
`500 × 3880 × 8 bytes ≈ 15.5 MB`. For large, highly recursive grammars (hundreds of states),
this could exceed the memory budget. Mitigation: cap supported grammar states at 256 states for
v0; add a diagnostic at `GrammarEngine::new` that warns when state count is high.

**R2: Context-dependent token handling correctness**
The 1% context-dependent tokens require runtime PDA stack inspection and are the most complex
code path. Mitigation: property-based tests (proptest) with round-trip generation: generate
1,000 random JSON values matching a fixed schema, constrained-decode them, verify the output
is valid JSON matching the schema.

**R3: Metal path logit mutability**
The Metal decode path returns logits via GPU readback into a CPU buffer whose mutability may not
be threaded through cleanly. Mitigation: audit `generate_greedy` and `generate_sampled` in
`metal_qwen35.rs` during implementation; ensure the logit buffer is mutable before sampling.

**R4: Vocabulary extraction from `BpeTokenizer`**
`BpeTokenizer` does not currently expose a vocabulary iterator. Adding one is a minor API
extension. Mitigation: add `pub fn vocab_bytes(&self) -> Vec<Vec<u8>>` as an **Unstable** method
in the same PR.

**R5: Grammar compilation latency**
For complex JSON schemas, `GrammarEngine::new` may take 50-200 ms. Unacceptable if called
per-request. Mitigation: document that `GrammarEngine` must be constructed once and shared via
`Arc<GrammarEngine>` across requests with the same schema.

## References

- XGrammar paper: arxiv:2411.15100 (MLSys 2025) — "XGrammar: Flexible and Efficient Structured
  Generation Engine for Large Language Models"
- llama.cpp GBNF engine: `common/grammar-parser.cpp`, `common/json-schema-to-grammar.cpp`
- vLLM XGrammar integration: `vllm/model_executor/guided_decoding/xgrammar_decoding.py`
- SGLang structured output: `python/sglang/srt/constrained/`
- Outlines: github.com/outlines-dev/outlines — Python reference for index-based constrained decoding
- JSON Schema specification: json-schema.org/draft/2020-12
- Prior lattice ADRs: ADR-011 (sampling strategies), ADR-025 (tokenizer BPE)
