# ADR-009: Model Architecture Support (BERT and Qwen3)

**Status**: Accepted\
**Date**: 2026-05-13\
**Crate**: `lattice-inference`

---

## Context

The crate targets two model families with fundamentally different architectures:

**BERT/BGE (encoder-only)**:

- Bidirectional self-attention (no causal mask)
- Multi-head attention with symmetric Q, K, V head count
- LayerNorm (not RMSNorm): `(x - mean) / sqrt(var + eps) * weight + bias`
- GELU FFN activation
- Mean pooling over all non-padding tokens
- `[CLS]`, `[SEP]`, `[PAD]` special tokens
- Fixed max sequence length: 512 (BERT-base, BGE-small/base/large)

`BertConfig` (from `src/model/bert.rs`):

```rust
pub struct BertConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    // head_dim = hidden_size / num_attention_heads
}
// Variants: BGE-small (6 layers, 384 hidden, 12 heads)
//           BERT-base (12 layers, 768 hidden, 12 heads)
//           BERT-large (24 layers, 1024 hidden, 16 heads)
```

**Qwen3 (decoder-only)**:

- Causal attention (future positions masked)
- Grouped Query Attention: 16 Q heads, 8 KV heads (groups=2 for 0.6B)
- RMSNorm (not LayerNorm): `x / sqrt(mean(x^2) + eps) * weight`; no bias
- SwiGLU FFN: `silu(gate_proj(x)) * up_proj(x)` → `down_proj(...)`
- Last-token pooling (EOS token)
- Rotary positional encoding (θ=1,000,000)
- Matryoshka output dimension truncation

`QwenConfig` (from `src/model/qwen.rs`):

```rust
pub struct QwenConfig {
    pub vocab_size: usize,           // 151,669
    pub hidden_size: usize,          // 1024
    pub num_hidden_layers: usize,    // 28
    pub num_attention_heads: usize,  // 16
    pub num_key_value_heads: usize,  // 8
    pub head_dim: usize,             // 128
    pub intermediate_size: usize,    // 3072
    pub max_position_embeddings: usize, // 32,768
    pub rms_norm_eps: f32,           // 1e-6
    pub rope_theta: f64,             // 1,000,000.0
}
```

**Qwen3.5 (decoder-only + linear attention)**:
A hybrid: standard GQA layers interleaved with `GatedDeltaNet` layers (recurrent linear attention). Defined in `src/model/qwen35_config.rs`.

---

## Decision

Support **two primary model families** (BERT encoder and Qwen3 decoder) as first-class architectures, with Qwen3.5 as an extension of the Qwen3 decoder adding GatedDeltaNet layers. Each family has a distinct config struct, weight struct, and forward pass implementation. No shared forward pass code between BERT and Qwen3.

---

## Key Design Choices

1. **Separate config structs**: `BertConfig` and `QwenConfig` are distinct types with no shared base. BERT's `head_dim` is derived (`hidden_size / num_attention_heads`), Qwen3's is explicit. A unified config struct would require optional fields or enums that complicate the type system with no benefit.
2. **LayerNorm vs RMSNorm are different implementations**: BERT requires LayerNorm (mean subtraction + variance normalization + learned weight+bias). Qwen3 uses RMSNorm (no mean subtraction, no bias, learned scale only). These are computed by different kernels. The choice is architecture-specific and not parameterizable as "use_bias: bool".
3. **Mean pool vs last-token pool**: BERT embedding is the mean of all non-padding token representations. Qwen3 embedding is the last non-padding token (EOS position). These require different pooling implementations in `src/pool.rs`. Using a wrong pooling function silently produces vectors of the right shape but wrong quality.
4. **QKV fusion at load time (Qwen3)**: `QwenLayerWeights` stores `fused_qkv: Vec<f32>` (Q+K+V rows concatenated). For BERT, Q/K/V are separate because `num_attention_heads == num_kv_heads` and the classical BERT implementation keeps them separate. Qwen3's GQA groups make fusion particularly effective (single GEMM produces all queries and fewer key/value vectors).
5. **Matryoshka dimension truncation (Qwen3)**: Qwen3-Embedding supports outputting embeddings of variable dimension (e.g., 256, 512, 1024 instead of the full hidden_size). The model is trained with Matryoshka loss; truncating the output vector to the desired dimension before L2 normalization is sufficient.
6. **Self-referential weight storage (BertModel)**: `BertModel` uses `mem::transmute` to extend slice lifetimes to `'static`. This is safe only because the `Box<SafetensorsFile>` backing store is declared before the weight slices in the struct (RFC 1857 drop order). A comment in `src/model/bert.rs` warns against reordering fields.

---

## Alternatives Considered

| Alternative                               | Pros                                    | Cons                                                                                                          | Why Not                                                                               |
| ----------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Unified `ModelConfig` enum                | Single type for both                    | Every consumer must match on the enum; field access is clunky; no type-level guarantees                       | BERT and Qwen3 configs share no fields; enum adds noise without unification           |
| Candle/burn model definitions             | Rich ecosystem; tested                  | Large dependency; different weight format; tied to those frameworks' abstractions                             | Crate policy: pure Rust, no ML framework dependencies                                 |
| Runtime normalization type selection      | More flexible                           | Branch on hot path per token; conditional compilation is cleaner for known architectures                      | Two architectures are known at compile time; runtime dispatch is unnecessary overhead |
| Shared forward pass with strategy pattern | Less duplication                        | The shared logic is minimal (matmul, activation); differences in attention + normalization + pooling dominate | Premature abstraction; architectures diverge, not converge                            |
| OnnxRuntime backend                       | Hardware portability; optimized kernels | Python runtime requirement; no pure-Rust guarantee; ONNX export limitations                                   | Crate policy: pure Rust, no ONNX                                                      |

---

## Consequences

**Positive**:

- Each architecture's config is self-describing and type-safe.
- Incorrect config combinations (e.g., passing `BertConfig` to `QwenModel::forward`) are caught at compile time.
- BERT and Qwen3 forward passes are independently optimizable without risk of regression in the other path.

**Negative**:

- Two separate forward pass implementations to maintain.
- Adding a new architecture (e.g., LLaMA) requires a new config struct, weight struct, and forward pass — no sharing with existing architectures.
- The `mem::transmute` pattern in `BertModel` is a footgun: future refactors must respect the field ordering invariant.

**Risks**:

- The Qwen3.5 hybrid (GQA + GatedDeltaNet layers) adds a third forward path. If the proportion of GDN layers changes in a future model variant, the layer-type dispatch logic in `Qwen35Model::forward()` must be updated to match.
- Matryoshka truncation relies on the model being trained with Matryoshka loss. A model checkpoint that was not trained with Matryoshka loss but uses `Qwen35Model` will silently produce lower-quality embeddings at truncated dimensions.

---

## References

- `src/model/bert.rs` — `BertConfig`, `BertModel`, field drop order warning, LayerNorm
- `src/model/qwen.rs` — `QwenConfig`, `QwenModel`, `ProfileTimings`, `SafetensorsStorage`
- `src/model/qwen35_config.rs` — `Qwen35Config` hybrid architecture
- `src/pool.rs` — `last_token_pool()`, `l2_normalize()`, mean pool
- `src/forward/` — separate CPU/NEON/Metal paths for each architecture
