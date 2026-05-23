# ADR-057: LoRA Full-Lifecycle Consumer API

**Status**: Draft
**Date**: 2026-05-23
**Crate**: lattice-tune, lattice-inference

## Context

khive's brain pack (ADR-032, ADR-042) consumes lattice as its inference and fine-tuning engine. Five gaps exist between what lattice exposes and what a downstream LoRA consumer needs for a complete adapter lifecycle:

1. **No LoraHook in cross-encoder inference.** `BertModel::forward` takes no hook argument. `CrossEncoderModel::score_batch` delegates to `bert.forward_tokenized` which has no injection point. Only `QwenModel` supports hook injection today (via `set_lora`). A consumer that swaps profile-resolved adapters per rerank call has no way to inject them into the BERT forward pass. (Issue #59)

2. **No public SafeTensors save path.** `load_peft_safetensors` exists in `crates/tune/src/lora/safetensors.rs` and handles PEFT + MLX formats. The inverse — a public `save_lora_safetensors` — is test-only (`write_test_peft_safetensors`, line 529, `#[cfg(test)]`). Consumers wanting portable adapter export (audit trails, deployment transfer, PEFT-tooling interop) must each reimplement the codec. (Issue #60)

3. **No online gradient step for LoRA weights.** `apply_lora` in `crates/tune/src/lora/apply.rs` is inference-only: it reads A/B matrices but never mutates them. `train_step_cpu` in `crates/tune/src/train/jit.rs:368` is explicitly marked "Placeholder CPU training step" and returns synthetic loss without touching weights. ADR-033 §Consequences/Negative confirms this gap. Consumers needing per-event online learning (reinforcement from user signals) must roll their own SGD against `LoraAdapter`'s public fields. (Issue #61)

4. **Stringly-typed module names in LoraAdapter.** `HashMap<(usize, String), LoraLayer>` keys are open-extensible (ADR-008 §Decision rationale) but typo-silent-no-op: `"q_porj"` compiles fine, applies nothing, and no error is raised. The tradeoff between open extensibility (lattice's concern) and compile-time safety (consumer's concern) has no canonical resolution. (Issue #62)

5. **No consumer docs for inference-hook.** The `inference-hook` feature on `lattice-tune` works — `impl LoraHook for LoraAdapter` exists behind `#[cfg(feature = "inference-hook")]` — but no ADR, README, or doc comment shows the downstream `Cargo.toml` pattern. Consumers infer it from `cfg` attributes. (Issue #63)

### Consumer Constraints (from khive ADR-032, ADR-042)

- **Rerank latency**: ≤200ms CPU / ≤50ms GPU for 32 (query, candidate) pairs, batching internal to lattice.
- **Online learning**: per-event SGD (or Adam) over LoRA A/B matrices, deterministic given identical starting weights + signal.
- **Adapter portability**: PEFT-compatible SafeTensors round-trip (what `load` can recover, `save` must produce).
- **Registry dispatch**: khive resolves adapters by `RegisteredModel.id: Uuid` (lattice ADR-029); the dispatch layer lives in khive, not lattice.

## Decision

### D1: LoraHook injection in BertModel and CrossEncoderModel

Add a `&dyn LoraHook` parameter to the BERT forward path. The hook is called after each encoder-layer linear projection, using BERT-convention module names (not Qwen-convention).

**Structural prerequisite**: today, BERT's Q/K/V and output projections are computed inside the opaque `multi_head_attention_in_place()` function (`crates/inference/src/attention/standard.rs`). This function takes layer weights and buffers but has no LoraHook parameter. To inject hooks at per-projection granularity, `multi_head_attention_in_place` must be extended with a `lora: &dyn LoraHook` parameter and `layer_idx: usize` so the hook can be called after each `matmul_bt`. The existing no-hook call sites pass `&NoopLoraHook` — zero behavioral change for non-LoRA paths.

The FFN projections (`intermediate.dense`, `output.dense`) are already individual `matmul_bt` calls in `BertModel::forward` (`bert.rs:363`, `bert.rs:381`) and can accept hook calls directly without refactoring.

**BERT module names for LoraHook** (matching actual BERT safetensors weight keys, NOT Qwen/LLaMA convention):

| Hook module name | BERT weight key | Struct field |
|---|---|---|
| `"query"` | `encoder.layer.{i}.attention.self.query.weight` | `query_weight` |
| `"key"` | `encoder.layer.{i}.attention.self.key.weight` | `key_weight` |
| `"value"` | `encoder.layer.{i}.attention.self.value.weight` | `value_weight` |
| `"attn_output"` | `encoder.layer.{i}.attention.output.dense.weight` | `attn_output_weight` |
| `"ffn_intermediate"` | `encoder.layer.{i}.intermediate.dense.weight` | `ffn_intermediate_weight` |
| `"ffn_output"` | `encoder.layer.{i}.output.dense.weight` | `ffn_output_weight` |

These differ from the Qwen path (`q_proj`, `k_proj`, etc.). PEFT adapters trained on BERT checkpoints use BERT-convention names; adapters trained on Qwen use Qwen-convention names. The `LoraHook` trait already dispatches by string — no conflict as long as consumers load the right adapter for the right model.

Two new methods on `CrossEncoderModel`, plus the internal `multi_head_attention_in_place` refactor:

```rust
// CrossEncoderModel — new methods, existing score/score_batch unchanged
pub fn score_with_hook(
    &self,
    query: &str,
    document: &str,
    lora: &dyn LoraHook,
) -> f32;

pub fn score_batch_with_hook(
    &self,
    query: &str,
    documents: &[&str],
    lora: &dyn LoraHook,
) -> Vec<f32>;
```

`BertModel::forward_tokenized` is `pub(crate)` today. The hook-aware variant stays `pub(crate)` — downstream consumers access hooks through `CrossEncoderModel::score_with_hook` (which is `pub`), not through the internal BERT forward method directly.

The existing `forward_tokenized` and `score`/`score_batch` remain unchanged (they internally pass `&NoopLoraHook` from `lattice_inference::lora_hook::NoopLoraHook`). This is additive, not breaking.

**Implementation note**: when D1 lands, the `LoraHook` trait doc comment (`crates/inference/src/lora_hook.rs:19-21`) and `LoraAdapter` module doc (`crates/tune/src/lora/mod.rs:12-15`) must be updated to include the BERT module names alongside the existing Qwen/GDN names.

### D2: Public SafeTensors save

Implement a public save function in `crates/tune/src/lora/safetensors.rs`, using the existing test fixture (`write_test_peft_safetensors`, line 529) as a reference for the safetensors serialization API. The test helper writes hardcoded synthetic data for two layers; the public function must iterate an arbitrary `&LoraAdapter`'s layers, infer block names from module names, and write proper metadata:

```rust
pub fn save_peft_safetensors(
    adapter: &LoraAdapter,
    path: &Path,
) -> Result<(), TuneError>;
```

Semantics:
- Writes PEFT-format keys: `base_model.model.model.layers.{i}.{block}.{module}.lora_{A|B}.weight`
- Stores `rank`, `alpha`, `target_modules` in SafeTensors metadata fields for lossless round-trip
- `block` inferred from module name: attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_*`, `out_proj`) → `self_attn`; MLP projections (`gate_proj`, `up_proj`, `down_proj`) → `mlp`
- All tensors written as F32 (matching the in-memory representation)
- Symmetric with `load_peft_safetensors`: `save(load(file)) ≈ file` up to metadata ordering
- **Scope**: LLaMA-family PEFT key format only (`base_model.model.model.layers.{i}.{block}.{module}`). BERT PEFT keys use a different hierarchy (`encoder.layer.{i}.attention.self.{module}`) which neither `save_peft_safetensors` nor the existing `parse_peft_key` / `load_peft_safetensors` handle. BERT PEFT format support is a follow-up tracked alongside the D1 implementation

The `LoraAdapter` method facade:

```rust
impl LoraAdapter {
    #[cfg(feature = "safetensors")]
    pub fn save_safetensors(&self, path: &Path) -> crate::error::Result<()> {
        safetensors::save_peft_safetensors(self, path)
    }
}
```

### D3: Online adapt_step

New module `crates/tune/src/lora/online.rs` (not extending `apply.rs`, which is scoped to immutable inference):

```rust
pub struct AdaptStepResult {
    pub loss: f32,
    pub grad_norm: f32,
}

pub fn adapt_step(
    adapter: &mut LoraAdapter,
    layer_idx: usize,
    module: &str,
    input: &[f32],
    target_delta: &[f32],
    learning_rate: f32,
) -> Result<AdaptStepResult, TuneError>;
```

Semantics:
- One SGD step for a single `(layer_idx, module)` LoRA layer
- `target_delta`: desired change to the base projection output (the error signal)
- Computes `loss = ||current_delta - target_delta||²` where `current_delta = scale * B @ (A @ x)`
- Gradient expressions (pinned to prevent implementation drift):
  - Let `residual = scale * B @ (A @ x) - target_delta`
  - `dL/dB = 2 * scale * outer(residual, A @ x)` — outer product, shape `(d_out, rank)`
  - `dL/dA = 2 * scale * outer(B^T @ residual, x)` — backprop through B, then outer product with input, shape `(rank, d_in)`
  - SGD update: `B -= lr * dL/dB`, `A -= lr * dL/dA`
- Returns `Err(TuneError::InvalidInput(...))` if the adapter has no layer for `(layer_idx, module)`, or if `input.len() != d_in` / `target_delta.len() != d_out` (dimension mismatch)
- Deterministic: identical inputs + starting weights → identical outputs
- No async, no IO — same purity contract as `apply_lora`
- Per-parameter Adam state is a follow-up (this ADR scopes to vanilla SGD)

This fills the gap flagged in ADR-033 §Consequences/Negative and replaces the `train_step_cpu` placeholder for the single-event case.

### D4: ModuleName — keep String keys, add validation

Decision: **keep `HashMap<(usize, String), LoraLayer>` keys.** The ADR-008 rationale (open extensibility for arbitrary architectures) is correct — lattice cannot enumerate all future module names at compile time, and different model families use different names (Qwen GDN: `in_proj_qkv`, `in_proj_z`, etc. vs standard: `q_proj`, `k_proj`, etc.).

Add a validation method instead:

```rust
impl LoraAdapter {
    pub fn validate_modules(&self, known: &[&str]) -> Vec<(usize, String)> {
        self.layers.keys()
            .filter(|(_, m)| !known.iter().any(|k| k == m))
            .cloned()
            .collect()
    }
}
```

Consumers call `validate_modules(&["query", "key", "value", ...])` (BERT) or `validate_modules(&["q_proj", "k_proj", ...])` (Qwen) at load time and handle unknowns (warn, error, or ignore) per their policy. khive maintains its own `ModuleName` enum on its side for serde/compile-time safety and maps through `validate_modules` at the lattice boundary.

This preserves lattice's open extensibility while giving consumers an opt-in validation hook.

Tests: `validate_modules` with all-known modules returns empty vec; with a typo (`"q_porj"`) returns that entry; with an empty adapter returns empty vec.

### D5: Consumer documentation for inference-hook

Add a `## Downstream Usage` section to ADR-031 §Consequences and a doc comment on the `inference-hook` feature in `crates/tune/Cargo.toml`:

```toml
[features]
inference-hook = ["lattice-inference"]  # Enable LoraHook impl for LoraAdapter — see ADR-031
```

Include in ADR-031 (or a new `USAGE.md` for `lattice-tune`):

```toml
# Cargo.toml for downstream consumers
[dependencies]
lattice-tune = { version = "0.2", features = ["inference-hook"] }
lattice-inference = "0.2"  # Required for LoraHook trait
```

```rust
use lattice_tune::lora::LoraAdapter;
use lattice_inference::lora_hook::LoraHook;  // not re-exported at crate root

let adapter = LoraAdapter::from_safetensors(path)?;

// Qwen (decoder) — set_lora stores the hook on the model
let hook: Box<dyn LoraHook> = Box::new(adapter.clone());
qwen_model.set_lora(hook);

// BERT/cross-encoder — pass hook per call (D1, this ADR)
let scores = cross_encoder.score_batch_with_hook(query, &docs, &adapter);
```

### Alternatives Considered

| Alternative | Pros | Cons | Why Not |
|---|---|---|---|
| Generic `CrossEncoderModel<H: LoraHook>` | Zero-cost monomorphization | Breaks all downstream type signatures; every consumer must parameterize on H | New method is additive and cheaper |
| `LoraAdapter::set_optimizer(Adam)` in D3 | Full optimizer support from day one | Adds per-parameter momentum/variance state (~2x adapter memory); Adam is not needed for single-event updates | SGD first, Adam as follow-up when consumers demonstrate need |
| Typed `ModuleName` enum in lattice-tune (D4) | Compile-time typo detection | Closed enum → new architectures require lattice release; violates ADR-008 design intent | Validation method gives opt-in safety without closing extensibility |
| `serde` for SafeTensors metadata round-trip (D2) | Familiar Rust pattern | SafeTensors metadata is `HashMap<String, String>`, not structured — serde adds a parse layer | Direct string key/value is simpler and matches the safetensors spec |

## Consequences

### Positive

- LoRA adapters can be injected into both decoder (Qwen) and encoder (BERT/cross-encoder) inference paths — consumers need one adapter type for all model architectures
- Adapter save/load is symmetric: PEFT tooling can consume lattice-exported adapters and vice versa
- Online learning is possible without forking `LoraAdapter` internals — consumers call `adapt_step` instead of reaching into `.layers[key].a` directly
- Module name validation catches typos at load time without restricting future architectures
- `inference-hook` has documented consumer path — new downstream crates don't need to reverse-engineer cfg attributes

### Negative

- `multi_head_attention_in_place` gains a `&dyn LoraHook` + `layer_idx` parameter — all existing call sites must be updated to pass `&NoopLoraHook` (mechanical but wide-reaching across BERT forward paths)
- `adapt_step` is vanilla SGD — consumers expecting Adam/AdamW must wait for a follow-up or wrap with their own optimizer state
- `save_peft_safetensors` produces F32-only output — no f16/bf16 quantized save (acceptable: adapters are small, ~1-10 MB)

### Implementation Order

| Phase | Scope | Issues | Effort |
|---|---|---|---|
| 1 | D5: Consumer docs for inference-hook | #63 | ~30 min |
| 2 | D2: SafeTensors save | #60 | ~2 hours |
| 3 | D4: validate_modules | #62 | ~1 hour |
| 4 | D1: LoraHook in BertModel + CrossEncoderModel (includes `multi_head_attention_in_place` refactor) | #59 | ~6 hours |
| 5 | D3: Online adapt_step | #61 | ~4 hours |

Phases 1-3 are independent and can run in parallel. Phase 4 requires reading the BERT forward path. Phase 5 is independent of phase 4.

## References

- Issue #59: Stable rerank API with LoraHook injection
- Issue #60: SafeTensors save for LoraAdapter
- Issue #61: adapt_step for online single-event gradient steps
- Issue #62: Typed ModuleName enum discussion
- Issue #63: Downstream consumer Cargo.toml example
- ADR-008: LoRA Injection via Trait Hook (`ADR-008-lora-injection.md`; trait shape, String key rationale)
- ADR-031: LoRA adapter management (load/apply)
- ADR-033: JIT adaptation (placeholder gap)
- `crates/inference/src/lora_hook.rs` — LoraHook trait
- `crates/inference/src/model/cross_encoder.rs` — CrossEncoderModel (no hook today)
- `crates/inference/src/model/bert.rs` — BertModel (no hook today)
- `crates/tune/src/lora/mod.rs` — LoraAdapter, LoraConfig, inference-hook impl
- `crates/tune/src/lora/apply.rs` — apply_lora (immutable inference)
- `crates/tune/src/lora/safetensors.rs` — load_peft_safetensors, test-only writer
