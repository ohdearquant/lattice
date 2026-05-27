# ADR-059: Composable Layer Architecture

**Status**: Proposed
**Date**: 2026-05-27
**Crate**: lattice-inference
**Research**: RQ-1 (`workspaces/20260527/01.md`, 1948 lines)
**KG**: ADR-059 `a95ab2d6` | AttentionOp `1ce9672b` | AttentionKind `1f7f0fc3` | LayerOp `f9fb0508` | TransformerLayer `b35c4305` | ModelSpec `3e893c94` | ArchitectureEnv `6e8df9af`

---

## Context

### The composability problem

Lattice has 10 attention mechanisms, each implemented as a standalone module with a different API surface:

```
crates/inference/src/attention/
  decode.rs          — DecodeAttention
  differential.rs    — DifferentialAttention
  flash.rs           — TiledAttentionConfig, cpu_tiled_attention()
  flash_causal.rs    — FlashCausalAttention
  gated.rs           — Gated attention primitives (deinterleave_q_gate, apply_sigmoid_gate)
  gdn.rs             — gated_delta_net_step(), GatedDeltaNetState
  gdn_fused.rs       — GatedDeltaNetFusedAttention
  gqa.rs             — apply_gqa_attention(), GqaScratch
  native_sparse.rs   — NativeSparseAttention
  standard.rs        — multi_head_attention_in_place()
```

The `LayerType` enum in `crates/inference/src/model/qwen35_config.rs:37-42` dispatches between exactly two choices:

```rust
pub enum LayerType {
    LinearAttention,   // GatedDeltaNet
    FullAttention,     // GQA
}
```

The forward loop in `crates/inference/src/forward/` consumes this enum in a `match` statement. Adding a third attention type (NSA, Differential, pruned-layer skip) requires editing:

1. `LayerType` enum definition (`qwen35_config.rs:37`)
2. `compute_layer_types()` function (`qwen35_config.rs:587`)
3. CPU forward loop match arms (`forward/cpu/` -- multiple files)
4. Metal forward loop match arms (`forward/metal_qwen35.rs`)
5. State allocation per-layer (`GatedDeltaNetState` vs KV cache branches)
6. Weight loading branches (different projection names per attention type)

**That is 6+ files and 15+ match arms for a single new attention variant.** This is not composable.

### Why composability matters for lattice specifically

Lattice's unique value proposition is **declarative architecture exploration** on Apple Silicon. An agent-driven research loop needs to express experiments like:

- "Run Qwen3.5-0.8B with [GDN x 18, GQA x 6] vs [GDN x 12, NSA x 6, GQA x 6]"
- "Try QuaRot-Q4 weights on layers 0-17 but keep layers 18-23 in F16"
- "Prune layers [20, 22] via ShortGPT and measure PPL delta"

Today, each experiment requires code changes. The `ModelSpec` DSL proposed here makes each experiment a YAML diff.

### What existing frameworks do

- **Hugging Face**: `config.json` + `AutoModel` dispatch by `model_type`. Configuration loading is separate from weight loading. No per-layer heterogeneity. ([HF docs: Auto Classes](https://huggingface.co/docs/transformers/en/model_doc/auto))
- **GGUF**: Metadata + tensor container with standardized keys (`general.architecture`, `general.quantization_version`). Not a layer-composition DSL; execution remains in the loader/runtime. ([ggml/docs/gguf.md](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md))
- **Burn**: `Module` is a parameter container; `Config` is serializable. Favors type-safe Rust modules over runtime architecture search. ([burn.dev/module](https://burn.dev/books/burn/building-blocks/module.html))
- **Candle**: Minimalist Rust ML with per-model loaders. Ergonomic for fixed architectures, not a search DSL. ([github.com/huggingface/candle](https://github.com/huggingface/candle))
- **TensorRT-LLM**: Graph-compiler model definition API. Closer to a compiler than a research DSL. ([nvidia.github.io/TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html))

None of these support per-layer heterogeneous attention + FFN + quantization + KV cache composition from a single declarative spec.

### Key insight from research

**GDN, NSA, Flash, GQA, MoE, KV cache, and quantization are not independent at runtime.** Flash attention must know KV precision. GDN has recurrent state instead of KV cache. NSA has both KV + sparse state. QuaRot changes the tensor basis and affects LoRA composition. Therefore, lattice should expose composability declaratively, but instantiate **validated, typed runtime objects**.

### Attention variants and their state requirements

| Variant                           | State                                        | Scratch               | Category          |
| --------------------------------- | -------------------------------------------- | --------------------- | ----------------- |
| MHA `10998240`                    | None (scratch `AttentionBuffers`)            | Softmax buffers       | Softmax (encoder) |
| GQA `36e42eb8`                    | KV cache                                     | Softmax + group batch | Softmax           |
| Flash CPU                         | KV cache                                     | Tiled buffers         | Softmax           |
| Flash Causal                      | KV cache                                     | Tiled buffers         | Softmax           |
| GDN `82877a02`                    | Recurrent `[B, heads, state_rank, head_dim]` | Gate/delta buffers    | Linear            |
| GDN Fused                         | Recurrent                                    | Gate/delta + fused    | Linear            |
| Gated GQA                         | KV cache                                     | Gate + GQA scratch    | Softmax           |
| Differential `ba18a56e`           | KV cache                                     | Split Q/K + lambda    | Softmax           |
| NSA (Sparse Attention `44932c9a`) | KV cache + sparse state                      | Sparse selection      | Sparse            |
| Decode                            | KV cache                                     | Minimal               | Softmax           |

The three state categories (KV, Recurrent, Sparse hybrid) are the fundamental reason a single trait with one `State` associated type does not work for heterogeneous composition.

---

## Decision

Adopt a **two-tier architecture** combining enum-dispatched inner kernels (Option B from research) with a `LayerOp` public interface (Option C).

### D1: Category traits + `AttentionKind` enum (inner tier)

Three category-specific traits capture the natural API differences between softmax, linear, and sparse attention:

```rust
/// Softmax-based attention with KV cache (GQA, Flash, Differential, Decode).
/// MHA (BERT-style encoder attention) uses scratch buffers instead of KV cache
/// and implements `SoftmaxAttention` with `KvCacheKind::None`.
pub trait SoftmaxAttention: Send + Sync {
    fn forward_softmax(
        &self,
        input: AttnInput<'_>,
        ctx: &mut ForwardCtx<'_>,
        kv: &mut KvCacheKind,
        scratch: &mut SoftmaxScratch,
        output: AttnOutput<'_>,
    ) -> Result<()>;
}

/// Recurrent linear attention (GDN, GDN-Fused).
/// State is a recurrent matrix, not a KV cache.
pub trait LinearAttention: Send + Sync {
    fn forward_recurrent(
        &self,
        input: AttnInput<'_>,
        ctx: &mut ForwardCtx<'_>,
        state: &mut RecurrentStateKind,
        scratch: &mut LinearAttnScratch,
        output: AttnOutput<'_>,
    ) -> Result<()>;
}

/// Hybrid sparse attention (NSA).
/// State is KV cache + sparse selection state.
pub trait SparseAttention: Send + Sync {
    fn forward_sparse(
        &self,
        input: AttnInput<'_>,
        ctx: &mut ForwardCtx<'_>,
        kv: &mut KvCacheKind,
        sparse: &mut SparseState,
        scratch: &mut SparseScratch,
        output: AttnOutput<'_>,
    ) -> Result<()>;
}
```

The unifying `AttentionOp` trait erases category differences for the layer-level dispatch:

```rust
pub trait AttentionOp: Send + Sync {
    fn tag(&self) -> AttentionTag;
    fn alloc_state(&self, shape: &AttnShape, cache: &KvCacheSpec) -> Result<AttentionStateKind>;
    fn alloc_scratch(&self, shape: &AttnShape) -> Result<AttentionScratchKind>;
    fn forward(
        &self,
        input: AttnInput<'_>,
        ctx: &mut ForwardCtx<'_>,
        state: &mut AttentionStateKind,
        scratch: &mut AttentionScratchKind,
        output: AttnOutput<'_>,
    ) -> Result<()>;
}
```

State and scratch are enums that carry the correct variant per attention category:

```rust
/// Per-layer attention state. Allocated once at model construction.
pub enum AttentionStateKind {
    /// No persistent state (MHA with recomputed attention).
    None,
    /// KV cache for softmax-based attention (GQA, Flash, Differential, Decode).
    Kv(KvCacheKind),
    /// Recurrent state matrix for linear attention (GDN).
    /// Shape: [batch, num_heads, state_rank, head_dim].
    Recurrent(RecurrentStateKind),
    /// Hybrid: KV cache for global context + sparse selection state.
    /// Used by NSA which maintains both a compressed KV summary and
    /// per-step block selection indices.
    Sparse {
        kv: KvCacheKind,
        sparse: SparseState,
    },
}

pub enum AttentionScratchKind {
    Softmax(SoftmaxScratch),
    Linear(LinearAttnScratch),
    Sparse(SparseScratch),
}
```

The `AttentionKind` enum dispatches via hand-written match arms (or `enum_dispatch` `2d536fb5` if macro diagnostics prove acceptable):

```rust
#[derive(Debug)]
pub enum AttentionKind {
    Mha(MhaAttention),
    Gqa(GqaAttention),
    FlashCpu(FlashCpuAttention),
    FlashCausal(FlashCausalAttention),
    Gdn(GatedDeltaNetAttention),
    GdnFused(GatedDeltaNetFusedAttention),
    /// GQA with sigmoid gate (Qwen3.5-style). Uses gated.rs primitives
    /// (deinterleave_q_gate, apply_sigmoid_gate) composed with GQA attention.
    /// Not a standalone kernel — wraps GQA with pre/post gate application.
    GatedGqa(GatedGqaAttention),
    Differential(DifferentialAttention),
    Nsa(NativeSparseAttention),
    Decode(DecodeAttention),
}

impl AttentionOp for AttentionKind {
    fn tag(&self) -> AttentionTag {
        match self {
            Self::Mha(_) => AttentionTag::Mha,
            Self::Gqa(_) => AttentionTag::Gqa,
            Self::FlashCpu(_) => AttentionTag::FlashCpu,
            Self::FlashCausal(_) => AttentionTag::FlashCausal,
            Self::Gdn(_) => AttentionTag::Gdn,
            Self::GdnFused(_) => AttentionTag::GdnFused,
            Self::GatedGqa(_) => AttentionTag::GatedGqa,
            Self::Differential(_) => AttentionTag::Differential,
            Self::Nsa(_) => AttentionTag::Nsa,
            Self::Decode(_) => AttentionTag::Decode,
        }
    }

    fn alloc_state(
        &self,
        shape: &AttnShape,
        cache: &KvCacheSpec,
    ) -> Result<AttentionStateKind> {
        match self {
            // Softmax-family with KV cache
            Self::Gqa(a) => Ok(AttentionStateKind::Kv(a.alloc_kv(shape, cache)?)),
            Self::FlashCpu(a) => Ok(AttentionStateKind::Kv(a.alloc_kv(shape, cache)?)),
            Self::FlashCausal(a) => Ok(AttentionStateKind::Kv(a.alloc_kv(shape, cache)?)),
            Self::Differential(a) => Ok(AttentionStateKind::Kv(a.alloc_kv(shape, cache)?)),
            Self::Decode(a) => Ok(AttentionStateKind::Kv(a.alloc_kv(shape, cache)?)),
            // MHA (BERT-style encoder): scratch buffers, no KV cache
            Self::Mha(_) => Ok(AttentionStateKind::None),
            // Linear-family: recurrent state, NO KV cache
            Self::Gdn(a) => Ok(AttentionStateKind::Recurrent(a.alloc_recurrent(shape)?)),
            Self::GdnFused(a) => Ok(AttentionStateKind::Recurrent(a.alloc_recurrent(shape)?)),
            // Sparse-family: KV + sparse selection state
            Self::Nsa(a) => {
                let kv = a.alloc_kv(shape, cache)?;
                let sparse = a.alloc_sparse_state(shape)?;
                Ok(AttentionStateKind::Sparse { kv, sparse })
            }
        }
    }

    fn forward(
        &self,
        input: AttnInput<'_>,
        ctx: &mut ForwardCtx<'_>,
        state: &mut AttentionStateKind,
        scratch: &mut AttentionScratchKind,
        output: AttnOutput<'_>,
    ) -> Result<()> {
        match (self, state, scratch) {
            // Softmax dispatch: type-checked (state, scratch) triple
            (Self::Gqa(a), AttentionStateKind::Kv(kv), AttentionScratchKind::Softmax(s)) => {
                a.forward_softmax(input, ctx, kv, s, output)
            }
            (Self::FlashCausal(a), AttentionStateKind::Kv(kv), AttentionScratchKind::Softmax(s)) => {
                a.forward_softmax(input, ctx, kv, s, output)
            }
            // Linear dispatch
            (Self::Gdn(a), AttentionStateKind::Recurrent(st), AttentionScratchKind::Linear(s)) => {
                a.forward_recurrent(input, ctx, st, s, output)
            }
            (Self::GdnFused(a), AttentionStateKind::Recurrent(st), AttentionScratchKind::Linear(s)) => {
                a.forward_recurrent(input, ctx, st, s, output)
            }
            // Sparse dispatch
            (Self::Nsa(a), AttentionStateKind::Sparse { kv, sparse }, AttentionScratchKind::Sparse(s)) => {
                a.forward_sparse(input, ctx, kv, sparse, s, output)
            }
            // ... remaining softmax variants omitted for brevity
            _ => Err(LatticeError::StateMismatch {
                layer: ctx.layer_idx,
                attention: self.tag(),
            }),
        }
    }
}
```

**Why this works for GDN's recurrent state vs GQA's KV cache**: GDN's `GatedDeltaNetState` holds a persistent `recurrent: TensorBuf` of shape `[B, num_heads, state_rank, head_dim]` that accumulates across tokens via the delta rule update `S = alpha * (S - beta * k^T @ (S @ k)) + beta * k^T @ v`. This is fundamentally different from GQA's append-only KV cache. The `AttentionStateKind` enum makes this type-safe at the boundary: `alloc_state()` returns `Recurrent(...)` for GDN and `Kv(...)` for GQA. A misconfigured layer (GDN with KV cache spec) is caught at construction, not at runtime.

**Why this works for NSA's hybrid state**: NSA (ADR-042, `44932c9a`) maintains both a KV cache (for the sliding-window branch) and sparse state (compression MLP outputs, block selection indices). The `Sparse { kv, sparse }` variant carries both. The `forward_sparse` trait method receives both references.

**How QuaRot interacts**: QuaRot (`e754741e`) changes the tensor basis via Hadamard rotation absorbed into linear projections (ADR-044). This does not change the attention algorithm; it changes the `LinearKind` wrapping each projection. The `QuantPlan` passed to attention kernel constructors specifies which weight format each projection uses. QuaRot-Q4 projections use `LinearKind::QuaRotQ4(QuaRotQ4Linear)`.

### D2: `LayerOp` trait + `TransformerLayer` struct (outer tier)

```rust
pub struct LayerInput<'a> {
    pub hidden: TensorView<'a>,    // [B, T, H]
    pub mask: Option<MaskView<'a>>,
}

pub struct LayerOutput<'a> {
    pub hidden: TensorMut<'a>,     // [B, T, H]
}

/// Per-layer mutable runtime state. Allocated once per layer at model construction.
pub struct LayerRuntime {
    pub attention_state: AttentionStateKind,
    pub attention_scratch: AttentionScratchKind,
    pub ffn_scratch: FfnScratchKind,
}

/// Public trait for a single transformer block. A model is Vec<Box<dyn LayerOp>>.
pub trait LayerOp: Send + Sync {
    fn layer_idx(&self) -> usize;
    fn spec(&self) -> &LayerSpec;
    fn alloc_runtime(&self, shape: &AttnShape, cache: &KvCacheSpec) -> Result<LayerRuntime>;
    fn forward(
        &self,
        input: LayerInput<'_>,
        ctx: &mut ForwardCtx<'_>,
        runtime: &mut LayerRuntime,
        output: LayerOutput<'_>,
    ) -> Result<()>;
}

/// Concrete implementation. Each field is an enum — one dispatch per component per layer.
pub struct TransformerLayer {
    idx: usize,
    spec: LayerSpec,
    ln1: NormKind,
    attention: AttentionKind,
    ln2: NormKind,
    ffn: FfnKind,
    residual: ResidualPolicy,
}

impl LayerOp for TransformerLayer {
    fn layer_idx(&self) -> usize { self.idx }
    fn spec(&self) -> &LayerSpec { &self.spec }

    fn alloc_runtime(
        &self,
        shape: &AttnShape,
        cache: &KvCacheSpec,
    ) -> Result<LayerRuntime> {
        Ok(LayerRuntime {
            attention_state: self.attention.alloc_state(shape, cache)?,
            attention_scratch: self.attention.alloc_scratch(shape)?,
            ffn_scratch: self.ffn.alloc_scratch(shape)?,
        })
    }

    fn forward(
        &self,
        input: LayerInput<'_>,
        ctx: &mut ForwardCtx<'_>,
        runtime: &mut LayerRuntime,
        output: LayerOutput<'_>,
    ) -> Result<()> {
        // Pre-norm: LN1 -> attention
        let x_norm = self.ln1.forward_to_scratch(input.hidden)?;
        let attn_out = output.hidden.reborrow();

        self.attention.forward(
            AttnInput {
                hidden: x_norm.as_view(),
                mask: input.mask,
            },
            ctx,
            &mut runtime.attention_state,
            &mut runtime.attention_scratch,
            AttnOutput { hidden: attn_out },
        )?;

        // Residual connection
        self.residual.apply(input.hidden, output.hidden.reborrow())?;

        // Pre-norm: LN2 -> FFN
        let y_norm = self.ln2.forward_to_scratch(output.hidden.as_view())?;
        self.ffn.forward(
            y_norm.as_view(),
            &mut runtime.ffn_scratch,
            output.hidden,
        )?;

        Ok(())
    }
}
```

**How LoRA hooks compose**: LoRA adapters (ADR-057, `0648de39`) wrap individual `LinearKind` projections. A LoRA-wrapped linear is `LinearKind::LoraWrapped { base: Box<LinearKind>, lora_a: TensorBuf, lora_b: TensorBuf, alpha: f32 }`. The `TransformerLayer` does not know about LoRA; it sees a `LinearKind` that happens to add a low-rank correction during `forward()`. For QuaRot + LoRA composition (ADR-045), the LoRA matrices are counter-rotated at load time: `A <- A * R^T`, `B <- R * B`.

**How Metal dispatch differs from CPU dispatch**: Metal uses fused kernels (`fused_attention`, `fused_qk_norm_rope` in `forward/metal.rs`) that combine multiple sub-operations into single GPU dispatches. The `TransformerLayer` is the CPU path. The Metal path constructs a parallel `MetalTransformerLayer` that holds `MetalAttentionKind` (GPU buffer handles instead of CPU tensors) but implements the same `LayerOp` trait. The `AttentionKind` enum's `tag()` method determines which MSL kernel to compile at model init.

### D3: `ForwardCtx` with metrics hooks

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    Prefill { seq_len: usize },
    Decode { token_pos: usize },
}

#[derive(Clone, Copy, Debug)]
pub struct AttnShape {
    pub batch: usize,
    pub q_len: usize,
    pub kv_len: usize,
    pub hidden: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

/// Per-layer metrics for architecture search diagnostics.
/// **The authoritative `LayerMetrics` schema is defined in ADR-061.**
/// This struct is an illustrative subset; implementers should use ADR-061's
/// full definition (including mode, block_influence, sparsity, kv_page_mass).
pub struct LayerMetrics {
    pub mode: MetricsMode,
    pub latency_ns: u64,
    pub input_norm: f32,
    pub output_norm: f32,
    pub update_ratio: f32,
    pub block_influence: f32,
    pub entropy: Option<f32>,
    pub sparsity: Option<f32>,
}

pub struct ForwardCtx<'a> {
    pub layer_idx: usize,
    pub step: StepKind,
    pub shape: AttnShape,
    pub position_ids: &'a [u32],
    pub causal: bool,
    pub rotary: Option<&'a RotaryTables>,
    /// Optional metrics collector for architecture search.
    /// When Some, attention/FFN kernels record diagnostics.
    pub metrics: Option<&'a mut LayerMetrics>,
}
```

The `metrics` field connects to ADR-061's architecture search loop. When `None` (production inference), zero overhead. When `Some`, kernels record attention entropy, activation RMS, and timing -- the gradient-free quality proxies that Training-Free NAS (`4d6a5c98`) and Zero-Cost NAS Proxies (`7e5b0314`) use to rank architecture candidates cheaply.

### D4: `FfnKind` and `LinearKind` enums

Same enum-dispatch pattern as `AttentionKind`:

```rust
#[derive(Debug)]
pub enum FfnKind {
    SwiGlu(SwiGluFfn),
    Moe(MoeFfn),
    PrunedMoe(PrunedMoeFfn),
}

impl FfnKind {
    pub fn forward(
        &self,
        input: TensorView<'_>,
        scratch: &mut FfnScratchKind,
        output: TensorMut<'_>,
    ) -> Result<()> {
        match (self, scratch) {
            (Self::SwiGlu(f), FfnScratchKind::SwiGlu(s)) => f.forward(input, s, output),
            (Self::Moe(f), FfnScratchKind::Moe(s)) => f.forward(input, s, output),
            (Self::PrunedMoe(f), FfnScratchKind::Moe(s)) => f.forward(input, s, output),
            _ => Err(LatticeError::ScratchMismatch),
        }
    }
}

#[derive(Debug)]
pub enum LinearKind {
    F32(F32Linear),
    F16(F16Linear),
    Q8(Q8Linear),
    Q4(Q4Linear),
    QuaRotQ4(QuaRotQ4Linear),
    LoraWrapped {
        base: Box<LinearKind>,
        lora_a: TensorBuf,
        lora_b: TensorBuf,
        alpha: f32,
        rank: usize,
    },
}

impl LinearKind {
    #[inline]
    pub fn forward(&self, x: TensorView<'_>, y: TensorMut<'_>) -> Result<()> {
        match self {
            Self::F32(op) => op.forward(x, y),
            Self::F16(op) => op.forward(x, y),
            Self::Q8(op) => op.forward(x, y),
            Self::Q4(op) => op.forward(x, y),
            Self::QuaRotQ4(op) => op.forward(x, y),
            Self::LoraWrapped { base, lora_a, lora_b, alpha, rank } => {
                // y = base(x) + (alpha/rank) * (x @ A^T) @ B^T
                base.forward(x, y.reborrow())?;
                apply_lora_correction(x, lora_a, lora_b, *alpha, *rank, y)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum NormKind {
    RmsNorm { weight: TensorBuf, eps: f32 },
    LayerNorm { weight: TensorBuf, bias: TensorBuf, eps: f32 },
}

#[derive(Debug, Clone, Copy)]
pub enum ResidualPolicy {
    /// Standard: output += input (pre-norm residual).
    Add,
    /// Identity skip: output = input. For ShortGPT-style layer removal.
    Skip,
}
```

### D5: `ModelSpec` YAML DSL

**Schema version**: `lattice.arch/v1`

```yaml
schema_version: lattice.arch/v1
name: qwen35_0_8b_hybrid_search
base_model:
  family: qwen35
  parameter_count: 0.8b
  weight_path: ./weights/qwen35-0.8b.safetensors
  tokenizer_path: ./tokenizer.json

dimensions:
  num_layers: 24
  hidden_size: 1024
  intermediate_size: 3584
  num_q_heads: 8
  num_kv_heads: 2
  head_dim: 256
  vocab_size: 248320
  rope:
    theta: 10000000.0
    partial_rotary_factor: 0.25
    scaling: none

defaults:
  norm: rms_norm
  attention:
    kind: gqa
    kernel: standard
  ffn:
    kind: swiglu
  quant:
    weights: f16
    activations: f16
    residual: f16
  kv_cache:
    kind: paged
    dtype: f16
    block_size: 16
  sparsity:
    kind: dense

layers:
  - repeat: 6
    sequence:
      - attention: { kind: gdn_fused, state_rank: 64 }
        kv_cache: { kind: none }
      - attention: { kind: gdn_fused, state_rank: 64 }
        kv_cache: { kind: none }
      - attention: { kind: gdn, state_rank: 64 }
        kv_cache: { kind: none }
      - attention: { kind: gqa, kernel: flash_causal }
        kv_cache: { kind: paged, dtype: f16, block_size: 16 }

overrides:
  - layers: [0, 1, 2]
    quant: { weights: f16 }
  - layers: "12"
    attention:
      kind: nsa
      compression: { block_size: 32 }
      selection: { top_k: 64 }
      sliding_window: { window: 512 }
  - layers: "18..23"
    ffn: { kind: moe, num_experts: 8, top_k: 2, router_dtype: f16 }

search:
  mutable:
    attention:
      allowed: [gqa, flash_causal, gdn, gdn_fused, nsa, differential, gated_attention]
    quant.weights:
      allowed: [f16, int8_symmetric, int4_grouped, quarot_q4]
    kv_cache.dtype:
      allowed: [f16, int8_symmetric]
    ffn.kind:
      allowed: [swiglu, moe]
  constraints:
    max_memory_mb: 4096
    max_ppl_delta: 0.20
    target: apple_m2_cpu
```

**Serde Rust representation** (key types):

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub schema_version: String,          // "lattice.arch/v1"
    pub name: String,
    pub base_model: BaseModelSpec,
    pub dimensions: ModelDims,
    pub defaults: LayerDefaults,
    pub layers: Vec<LayerPattern>,
    #[serde(default)]
    pub overrides: Vec<LayerOverride>,
    #[serde(default)]
    pub search: Option<SearchSpec>,
}

/// Layers support repeat/sequence patterns for compact specs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LayerPattern {
    Single(LayerSpec),
    Repeat { repeat: usize, sequence: Vec<LayerSpecPatch> },
}

/// Attention spec is tagged by kind. Each variant carries kind-specific fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AttentionSpec {
    Mha(MhaSpec),
    Gqa(GqaSpec),
    FlashCpu(FlashSpec),
    FlashCausal(FlashSpec),
    Gdn(GdnSpec),
    GdnFused(GdnSpec),
    GatedAttention(GatedAttentionSpec),
    Differential(DifferentialAttentionSpec),
    Nsa(NsaSpec),
    Decode(DecodeSpec),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FfnSpec {
    SwiGlu { intermediate_size: Option<usize> },
    Moe { num_experts: usize, top_k: usize, router_dtype: RuntimeDType },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KvCacheSpec {
    None,
    Flat { dtype: KvDType },
    Paged { dtype: KvDType, block_size: usize },
}
```

**Validation rules** (mandatory before weight loading):

```rust
pub fn validate_model_spec(spec: &ModelSpec) -> Result<NormalizedModelSpec> {
    let layers = expand_layers(&spec.defaults, &spec.layers, &spec.overrides)?;

    // Layer count must match dimensions
    if layers.len() != spec.dimensions.num_layers {
        return Err(LatticeError::InvalidConfig(format!(
            "expected {} layers, got {}",
            spec.dimensions.num_layers, layers.len()
        )));
    }

    for layer in &layers {
        // GDN uses recurrent state -- KV cache must be None
        validate_kv_cache_compat(layer)?;
        // Flash-causal has no int4 KV-cache kernel
        validate_quant_attention_compat(layer)?;
        // Head dims must match model dimensions
        validate_attention_dims(&spec.dimensions, layer)?;
        // MoE expert count must be > top_k
        validate_ffn_compat(&spec.dimensions, layer)?;
    }

    Ok(NormalizedModelSpec {
        name: spec.name.clone(),
        dims: spec.dimensions.clone(),
        layers,
        config_hash: stable_hash(spec)?,
    })
}
```

Key validation: GDN with a KV cache spec is rejected. Softmax attention without a KV cache spec is rejected. Flash-causal with int4 KV cache is rejected (no kernel exists).

**Example configs**:

**Config 1: Standard Qwen3.5-0.8B** (reproduces current behavior):

```yaml
schema_version: lattice.arch/v1
name: qwen35_0_8b_standard
base_model: { family: qwen35, parameter_count: 0.8b }
dimensions: { num_layers: 24, hidden_size: 1024, head_dim: 256 }
defaults:
  attention: { kind: gdn }
  kv_cache: { kind: none }
layers:
  - repeat: 6
    sequence:
      - {} # GDN (default)
      - {}
      - {}
      - attention: { kind: gqa }
        kv_cache: { kind: paged, dtype: f16, block_size: 16 }
```

**Config 2: Hybrid GDN+GQA+NSA experiment**:

```yaml
schema_version: lattice.arch/v1
name: qwen35_0_8b_nsa_experiment
base_model: { family: qwen35, parameter_count: 0.8b }
dimensions: { num_layers: 24, hidden_size: 1024, head_dim: 256 }
layers:
  - repeat: 4
    sequence:
      - attention: { kind: gdn_fused, state_rank: 64 }
        kv_cache: { kind: none }
      - attention: { kind: gdn, state_rank: 64 }
        kv_cache: { kind: none }
      - attention: { kind: nsa, compression: { block_size: 32 }, selection: { top_k: 64 } }
        kv_cache: { kind: paged, dtype: f16, block_size: 16 }
      - attention: { kind: gqa, kernel: flash_causal }
        kv_cache: { kind: paged, dtype: f16, block_size: 16 }
healing:
  method: lora
  layers: [2, 6, 10, 14, 18, 22] # NSA layers need adaptation
  rank: 8
  alpha: 16
```

**Config 3: Pruned model with QuaRot-Q4**:

```yaml
schema_version: lattice.arch/v1
name: qwen35_0_8b_pruned_quarot
base_model: { family: qwen35, parameter_count: 0.8b }
dimensions: { num_layers: 24, hidden_size: 1024, head_dim: 256 }
defaults:
  quant: { weights: quarot_q4, group_size: 128, rotation: { seed: 0xC0FFEE } }
overrides:
  - layers: [20, 22]
    residual: skip # ShortGPT-style removal
  - layers: [0, 1, 2]
    quant: { weights: f16 } # Keep early layers full-precision
```

### D6: Architecture search environment (Future Work)

**Note:** D6 is deferred to a dedicated architecture-search ADR. ADR-059 defines the layer
representation and DSL; search environments consume `NormalizedModelSpec` but belong in
ADR-061's experiment runner or a separate ADR. The hooks below are preserved as design intent,
not as v0.3 scope.

```rust
pub trait ArchitectureEnv {
    fn reset(&mut self, seed: u64) -> Result<ArchObservation>;
    fn step(&mut self, action: ArchAction) -> Result<(ArchObservation, Reward, Done)>;
    fn evaluate(&mut self, spec: &NormalizedModelSpec) -> Result<EvalReport>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ArchAction {
    SetAttention { layer: usize, attention: AttentionSpec },
    SetWeightQuant { layer: usize, tensor: TensorSelector, quant: QuantSpec },
    SetKvCache { layer: usize, cache: KvCacheSpec },
    SetFfn { layer: usize, ffn: FfnSpec },
    AttachLora { layer: usize, rank: usize, alpha: f32, targets: Vec<String> },
    NoOp,
}
```

Search space for 24 layers with 10 attention x 4 quantization = 40 choices per layer is 40^24 ~ 2.8 x 10^38. The `search` section constrains this with allowed sets and hardware constraints. Search algorithms (staged plan):

| Algorithm                          | Fit for lattice | When to use                                       |
| ---------------------------------- | --------------- | ------------------------------------------------- |
| Random search + constraints        | Strong baseline | DSL validation, early exploration                 |
| Evolutionary search                | Very good       | Discrete architecture strings, Pareto fronts      |
| Bayesian optimization / TPE        | Very good       | Expensive evaluations, structured features        |
| PPO `e59014dd` on discrete actions | Later           | Needs many evaluations; better with cheap proxies |
| LLM-agent search                   | Orchestration   | Emit configs/actions, not source patches          |

Prior work: NAS-RL (`7504a737`, Zoph & Le 2017, arxiv:1611.01578), HAT (`b8d7b2ff`, Wang et al. 2020, arxiv:2005.14187), DynaBERT (`aa3ffd4a`, Hou et al. 2020, arxiv:2004.04037), KVTuner (`6bf7f34d`, Liu et al. 2025, arxiv:2502.04420), Zero-Cost NAS Proxies (`7e5b0314`, Abdelfattah et al. 2021, arxiv:2101.08134).

---

## Alternatives Considered

### Option A: Single `AttentionKernel` trait with associated types

```rust
pub trait AttentionKernel: Send + Sync + 'static {
    type Config: Clone + Send + Sync + 'static;
    type State: AttentionState;
    type Scratch: AttentionScratch;
    // ...
}
```

| Property             | Assessment                                                       |
| -------------------- | ---------------------------------------------------------------- |
| Type safety          | Excellent -- GDN cannot accidentally receive KV cache state      |
| Runtime composition  | **Poor** -- heterogeneous storage requires `dyn Any` downcasting |
| Hot-loop performance | Excellent with monomorphized generics                            |
| Code size            | Grows via attention x quant x FFN generics                       |

**Rejected because**: `Vec<Box<dyn AttentionKernel>>` requires type erasure of the associated `State` and `Scratch` types. The only way to get them back is `downcast_ref::<GqaState>()`, which reintroduces unsafe code and loses the type safety that motivated associated types in the first place. For a closed set of 10 implementations, the cure is worse than the disease.

**Kept as**: Internal kernel authoring API for unit tests and specialized AOT builds where the model architecture is fixed at compile time.

### Option B (selected): Category traits + `AttentionKind` enum

See Decision section above. This is the selected approach.

| Property             | Assessment                                                          |
| -------------------- | ------------------------------------------------------------------- |
| Type safety          | Good -- mismatches caught at construction or with one checked match |
| Runtime composition  | Excellent for DSL-driven models                                     |
| Hot-loop performance | Very good -- one predictable branch per layer                       |
| Extensibility        | Good for lattice's closed set of 10 attention variants              |

**Why enum dispatch wins for lattice's specific set**: The `enum_dispatch` crate's benchmarks report large speedups over `dyn Trait` for tiny trait methods because enum dispatch avoids heap indirection, vtable lookups, and enables compiler inlining ([docs.rs/enum_dispatch](https://docs.rs/enum_dispatch/latest/enum_dispatch/)). For lattice, dispatch happens once per layer per token (24 calls/token). Even if an indirect call costs tens of nanoseconds, total dispatch is sub-microsecond -- dominated by projection matmuls. But vtable dispatch **prevents inlining across the call boundary**, which matters when the compiler could otherwise fuse the dispatch with the subsequent GEMM setup. Enum dispatch preserves this optimization path.

The Rust Book explicitly describes the runtime lookup and missed inlining tradeoff for trait objects ([doc.rust-lang.org/book/ch18-02-trait-objects](https://doc.rust-lang.org/book/ch18-02-trait-objects.html)).

**Dispatch policy** (from research):

```
Allowed:
  dispatch once per layer
  dispatch once per major kernel choice
  dispatch once per quantized linear call

Forbidden:
  dispatch inside matmul inner loops
  dispatch per quant group
  dispatch per attention tile element
  dispatch per token inside a recurrent scan kernel
```

### Option C: `LayerOp` trait at transformer-block level

See Decision section. Used as the **public runtime interface** (combined with Option B internally).

### Pure `dyn Trait` with vtable dispatch everywhere

**Rejected.** Dispatch per tile/head/token in the hot loop is measurable. The enum set is closed (10 attention variants, 3 FFN variants, 5 linear variants). Open-ended plugins are not a requirement.

### Compile-time generics only

**Rejected.** `Block<GdnFused, SwiGlu, Q8>` produces excellent code but prevents heterogeneous layer stacks -- you cannot mix GDN and GQA layers in the same `Vec` without existential types. Not compatible with a YAML-driven DSL.

**Kept as**: Optional AOT "frozen" build path where `type L0 = Block<GdnFused, SwiGlu, Q8>` produces zero-dispatch code for production deployment of a validated architecture.

### Keep current ad-hoc dispatch

**Rejected.** Adding a third attention type requires 6+ file edits and 15+ match arms. The `LayerType` enum supports exactly 2 variants. This is the problem the ADR solves.

---

## Consequences

### Positive

- **Adding a new attention variant**: Implement the appropriate category trait (`SoftmaxAttention`, `LinearAttention`, or `SparseAttention`), add a variant to `AttentionKind` and `AttentionTag`, add one match arm per dispatch method. No forward loop changes.
- **Architecture experiments**: A YAML diff, not a code change. Agent-driven search emits config mutations, not source patches.
- **Performance**: Zero-cost at steady state. Enum dispatch is a single branch per layer -- same cost as the current `match layer_type`. The `#[inline]` on `LinearKind::forward()` allows the compiler to see through the dispatch when the variant is known.
- **Pruning**: `ResidualPolicy::Skip` enables ShortGPT-style layer removal without tensor surgery (ADR-060). The pruned config sets `residual: skip` on removed layers.
- **Mixed quantization**: Per-layer `QuantPlanSpec` enables "F16 for early layers, QuaRot-Q4 for middle layers, Q8 for final layers" -- a capability KVTuner (`6bf7f34d`) showed is important for quality.

### Negative

- **10-variant enum boilerplate**: Each new variant adds match arms to `tag()`, `alloc_state()`, `alloc_scratch()`, `forward()`. With 10 variants, this is ~40 match arms per method. `enum_dispatch` reduces this to a `#[enum_dispatch]` annotation.
- **State enum downcasting**: The `forward()` triple-match `(self, state, scratch)` can fail at runtime if a misconfigured layer pairs GDN with KV state. This is caught at construction time by the validator, not at forward time, but the runtime error path must exist.
- **YAML complexity**: The DSL has repeat patterns, overrides, search sections -- a learning curve for users who just want to run a model.

### Risks

- **Over-abstraction before validation.** Mitigated by phased implementation: D1-D2 first (refactor existing code), D3-D6 incrementally. Each phase must ship with bench-compare evidence (ADR-058) that the refactor does not regress throughput.
- **GDN state leak between sequences.** `RecurrentStateKind` must be reset between unrelated sequences. Failing to call `reset()` between users in a multi-tenant server leaks recurrent state -- wrong outputs with no error. The `TransformerLayer::alloc_runtime()` initializes state to zero; the serving layer must call `alloc_runtime()` or explicit reset per sequence.
- **Metal kernel recompilation.** MSL kernel constants (`__FA_HEAD_DIM__`, `__FA_GQA_GROUPS__`) are injected at model init. A `ModelSpec` change that alters head dimensions requires recompiling the Metal library. The `ModelBuilder` must detect this and recompile, not silently reuse a stale kernel.

---

## Migration Plan

The `LayerType` enum (`qwen35_config.rs:37-42`) is superseded by `AttentionKind`. Migration preserves all existing behavior:

| Phase | Scope                                                                                               | Files Changed                                                                          | PRs |
| ----- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | --- |
| P1    | `AttentionOp` trait + `AttentionKind` enum wrapping existing 10 modules                             | `attention/mod.rs`, new `attention/attention_op.rs`, new `attention/attention_kind.rs` | 1   |
| P2    | `LayerOp` trait + `TransformerLayer` struct; refactor Qwen3.5 CPU forward to use `TransformerLayer` | `forward/cpu/`, new `layer/mod.rs`, new `layer/transformer_layer.rs`                   | 1   |
| P3    | `FfnKind` + `LinearKind` + `NormKind` + `ResidualPolicy` enums                                      | New `layer/ffn_kind.rs`, `layer/linear_kind.rs`, `layer/norm_kind.rs`                  | 1   |
| P4    | `ModelSpec` YAML parser + validator + builder                                                       | New `arch/spec.rs`, `arch/normalize.rs`, `arch/validate.rs`, `arch/build.rs`           | 1-2 |
| P5    | `ResidualPolicy::Skip` for pruned layers (links to ADR-060)                                         | `layer/transformer_layer.rs`                                                           | 1   |
| P6    | `ArchitectureEnv` + search database (links to ADR-061) — **Future Work**                            | New `search/env.rs`, `search/reward.rs`, `search/db.rs`                                | 2-3 |

**v0.3 scope**: P1 through P5. P6 is deferred to a dedicated architecture-search ADR.

**Dependency order**: P1 -> P2 -> P3 -> P4 -> P5.

**Key invariant**: After each phase, `make bench-compare` must show no throughput regression on Qwen3.5-0.8B decode. The refactor adds abstraction layers; it must not add overhead.

**Quality gates per phase** (beyond bench-compare throughput):

- P1-P2: PPL/NLL delta < 0.001 vs pre-refactor on WikiText-2 (refactor must be semantically identical)
- P3-P4: Token agreement test: identical logits for 100 random prompts before/after
- P5: Paired PPL comparison for pruned vs unpruned (links to ADR-060 quality gates)

---

## Crate Layout

```
crates/inference/src/
  arch/
    mod.rs
    spec.rs              # Serde ModelSpec, AttentionSpec, FfnSpec, etc.
    normalize.rs         # expand_layers(): repeat/override resolution
    validate.rs          # compatibility matrix checks
    build.rs             # ModelBuilder: spec -> Vec<TransformerLayer>
    search_space.rs      # SearchSpec, ArchAction grammar

  layer/
    mod.rs
    layer_op.rs          # LayerOp trait, LayerInput/Output/Runtime
    transformer_layer.rs # TransformerLayer struct
    ffn_kind.rs          # FfnKind enum
    linear_kind.rs       # LinearKind enum
    norm_kind.rs         # NormKind enum
    residual.rs          # ResidualPolicy enum

  attention/
    mod.rs
    attention_op.rs      # AttentionOp, SoftmaxAttention, LinearAttention, SparseAttention
    attention_kind.rs    # AttentionKind enum, AttentionStateKind, AttentionScratchKind
    gqa.rs               # (existing, unchanged)
    gdn.rs               # (existing, unchanged)
    gdn_fused.rs         # (existing, unchanged)
    native_sparse.rs     # (existing, unchanged)
    flash.rs             # (existing, unchanged)
    ...

  search/
    env.rs               # ArchitectureEnv trait
    action.rs            # ArchAction enum
    reward.rs            # multi-objective reward function
    mutators.rs          # LocalSearchMutator, etc.
    pareto.rs            # Pareto frontier maintenance
    db.rs                # SQLite result store

  quant/
    ...                  # (existing, augmented with QuantSpec/OfflineQuantizer)

  cache/
    kv_cache_kind.rs     # KvCacheKind enum (existing, promoted)
    ...
```

---

## References

- ADR-010: Attention Mechanisms (`docs/adr/ADR-010-attention-mechanisms.md`) -- current 4-variant design
- ADR-042: Native Sparse Attention (`44932c9a`) -- NSA three-branch architecture
- ADR-044: QuaRot Rotated Quantization (`e754741e`) -- Hadamard rotation + Q4
- ADR-045: QuaRot + LoRA Composition -- counter-rotation for LoRA adapters
- ADR-057: LoRA Lifecycle (`0648de39`) -- adapter loading/merging
- ADR-058: Regression Gate -- bench-compare CI requirement
- `enum_dispatch` crate -- [docs.rs/enum_dispatch](https://docs.rs/enum_dispatch/latest/enum_dispatch/)
- Rust Book Ch. 18.2 -- trait objects and dynamic dispatch -- [doc.rust-lang.org](https://doc.rust-lang.org/book/ch18-02-trait-objects.html)
- Ainslie et al. 2023 -- GQA: Training Generalized Multi-Query Transformer Models -- EMNLP 2023
- Yang et al. 2024 -- Gated Delta Networks -- [arxiv:2412.06464](https://arxiv.org/abs/2412.06464)
- Yuan et al. 2025 -- Native Sparse Attention -- ACL 2025
- Ashkboos et al. 2024 -- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs -- NeurIPS 2024 -- [arxiv:2404.00456](https://arxiv.org/abs/2404.00456)
- Hu et al. 2021 -- LoRA: Low-Rank Adaptation -- [arxiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Zoph & Le 2017 -- Neural Architecture Search with RL -- ICLR 2017 -- [arxiv:1611.01578](https://arxiv.org/abs/1611.01578)
- Schulman et al. 2017 -- PPO -- [arxiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Abdelfattah et al. 2021 -- Zero-Cost Proxies for Lightweight NAS -- [arxiv:2101.08134](https://arxiv.org/abs/2101.08134)
- Wang et al. 2020 -- HAT: Hardware-Aware Transformers -- ACL 2020 -- [arxiv:2005.14187](https://arxiv.org/abs/2005.14187)
- Hou et al. 2020 -- DynaBERT: Dynamic BERT with Adaptive Width and Depth -- NeurIPS 2020 -- [arxiv:2004.04037](https://arxiv.org/abs/2004.04037)
- Liu et al. 2025 -- KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache -- [arxiv:2502.04420](https://arxiv.org/abs/2502.04420)
- vLLM Quantization docs -- [docs.vllm.ai/en/latest/features/quantization/](https://docs.vllm.ai/en/latest/features/quantization/)
- TensorRT-LLM Quantization docs -- [nvidia.github.io/TensorRT-LLM/features/quantization.html](https://nvidia.github.io/TensorRT-LLM/features/quantization.html)
- HF Transformers Quantization -- [huggingface.co/docs/transformers/en/main_classes/quantization](https://huggingface.co/docs/transformers/en/main_classes/quantization)
