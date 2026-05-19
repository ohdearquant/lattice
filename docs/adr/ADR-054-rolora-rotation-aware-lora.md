# ADR-054: Rotation-Aware LoRA Training (RoLoRA Integration)

**Status**: Proposed
**Date**: 2026-05-19
**Crate**: `lattice-tune` (training) + `lattice-inference` (serving)
**Depends on**: ADR-044 (QuaRot), ADR-045 (QuaRot+LoRA serving composition), ADR-031 (LoRA management)

---

## Context

ADR-045 solved the *serving-side* coupling between QuaRot Q4 and LoRA adapters: adapters trained
on the unrotated basis are corrected at load time via counter-rotation (`A ← A·R^T` for input-side
projections, `B ← R·B` for output-side). This is exact and zero-cost at inference time.

The remaining problem is *training-time*. Standard LoRA fine-tuning trains adapters against the
original (un-rotated) weight basis. When the base model has had its residual stream rotated by R
(the Hadamard transform from ADR-044), the quantization noise environment changes dramatically:

- **At Q8**: quantization noise amplitude is ~2:1 relative to a typical LoRA delta. Adapters
  trained on unrotated weights, counter-rotated at load time (ADR-045), function acceptably.
- **At Q4**: quantization noise amplitude can *exceed* the adapter delta. The adapter signal is
  in the wrong basis relative to the quantization error structure, and the counter-rotation at
  load time corrects the basis mismatch but does not improve the signal-to-noise ratio at training
  time. The adapter learns to compensate for the unrotated model's error surface, not the rotated
  model's error surface.

**RoLoRA** (EMNLP 2024, arxiv:2407.08044) directly addresses this. It applies the same Hadamard
rotation used for quantization *before* fine-tuning, training LoRA adapters natively in the rotated
weight basis. Adapters trained this way require no post-hoc correction; they are already in the
basis the served model operates in. Reported gain: +29.5 percentage points absolute on W4A4
commonsense reasoning versus standard LoRA.

The QuaRot rotation seed in Lattice is deterministic. The same seed produces the same R matrix.
This means training-time and serving-time rotation are trivially synchronized — the basis
alignment problem reduces to a metadata handshake.

### What exists in Lattice today

| Component | Location | Status |
|-----------|----------|--------|
| `LoraAdapter` / `LoraLayer` | `crates/tune/src/lora/mod.rs` | Active |
| `LoraConfig` | `crates/tune/src/lora/mod.rs` | Active |
| `RandomizedHadamard` | `crates/inference/src/quant/quarot/` | Active — deterministic from seed |
| QuaRot rotation plan | `crates/inference/src/quant/quarot/plan.rs` | Active — per-projection side rules |
| QuaRot seed in artifact | `config.json` of `quantize_quarot` output | **NOT YET PERSISTED** (see ADR-045 §Prerequisites) |
| Serving-side counter-rotation | `crates/inference/src/…` | Active (ADR-045, PRs #35-#37) |

---

## Decision

### The gauge alignment problem

In the QuaRot residual stream, the model operates with rotated hidden states `h_rot = R·h`. A
LoRA adapter `(B, A)` trained on the *unrotated* model learns to correct the unrotated error
surface. Serving via counter-rotation (ADR-045) fixes the basis mismatch exactly, but the adapter
has been optimized against the wrong noise distribution.

A RoLoRA adapter `(B, A)` is trained from the start with the base weights already in the rotated
basis: `W_rot = W·R^T` (input-side) or `W_rot = R·W` (output-side). The LoRA delta is then
trained directly on `W_rot`, so at serving time no counter-rotation is needed — the adapter is
already basis-correct.

For the serving-side equation (input-side projection as example):

```
Standard LoRA + ADR-045 counter-rotation:
  y = W_rot·h_rot + s·B·(A·R^T)·h_rot = W·h + s·B·A·h  ✓ (basis corrected post-hoc)

RoLoRA (adapter trained in rotated basis):
  y = W_rot·h_rot + s·B·A·h_rot                         ✓ (no correction needed)
```

Both produce correct output. The difference is in what loss landscape the adapter was trained
against, which determines quality at low bit-widths.

### Training-time changes (lattice-tune)

Introduce a `rotation_aware` flag in `LoraConfig`:

```rust
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f32,
    pub target_modules: Vec<String>,
    /// When Some(seed), apply the QuaRot Hadamard rotation to the frozen base
    /// weights before LoRA initialization. Adapters trained with this flag are
    /// native to the rotated basis and require no counter-rotation at serving time.
    pub quarot_seed: Option<u64>,
}
```

When `quarot_seed` is `Some(seed)`, the training pipeline:

1. Reconstructs `R = RandomizedHadamard::new(seed, hidden_dim)` using the same primitive from
   `lattice-inference` (no new Hadamard implementation — reuse via a `lattice-inference` dev
   dependency in `lattice-tune`, which already exists in `tune → inference` direction).
2. Applies rotation absorption to the frozen base weight snapshots before LoRA init:
   - Input-side projections: `W_frozen ← W_frozen·R^T`
   - Output-side projections: `W_frozen ← R·W_frozen`
3. Initializes LoRA matrices `A` (random Gaussian) and `B` (zeros) against the rotated frozen
   weights, following standard LoRA initialization.
4. Trains. The LoRA delta `s·B·A` is learned in the rotated basis throughout.
5. Saves the adapter safetensors with an `"rotation_aware": true` metadata key and the
   `"quarot_seed": <seed>` value in the adapter config.

The frozen base weights themselves are not written to disk — only the adapter `A` and `B` matrices
are exported. The rotation is applied transiently during forward/backward.

### Serving-time changes (lattice-inference)

The `load_lora_adapter` API (ADR-045) accepts `quarot_seed: Option<u64>`. Extend the loading
logic to read the adapter metadata:

```rust
pub fn load_lora_adapter(
    &mut self,
    layers: Vec<LoraLayerData>,
    scale: f32,
    quarot_seed: Option<u64>,  // None = legacy path; Some = apply counter-rotation
) -> Result<(), InferenceError>;
```

When loading an adapter whose metadata contains `"rotation_aware": true`:

- **The adapter is already in the rotated basis.** Upload A and B directly to Metal buffers
  without applying WHT transforms. Pass `quarot_seed: None` at the API callsite (or detect from
  metadata automatically once seed persistence ships, per §Prerequisites).

When loading a legacy adapter (no rotation_aware metadata) against a QuaRot Q4 base:

- Apply counter-rotation as specified in ADR-045. This path is unchanged.

**Verification at load time**: if the served model is a QuaRot artifact and the adapter metadata
is absent or `"rotation_aware": false`, emit a `warn!` log identifying the adapter as legacy and
confirming counter-rotation is being applied. This surfaces the distinction without breaking
existing workflows.

### Metadata contract

The QuaRot converter (`quantize_quarot`) MUST persist rotation metadata (ADR-045 §Prerequisites
is still open). This ADR depends on that field existing. The required `config.json` addition:

```json
{
  "quarot": {
    "kind": "randomized_hadamard",
    "seed": 12648430,
    "hidden_dim": 1024,
    "plan": "qwen35_residual_stream_linear_layers"
  }
}
```

The adapter safetensors `config.json` MUST contain:

```json
{
  "rotation_aware": true,
  "quarot_seed": 12648430
}
```

At load time: if the model artifact has a `quarot.seed` and the adapter has
`rotation_aware: true` with a matching seed, skip counter-rotation. Seed mismatch between model
artifact and adapter metadata is an error (fail-closed).

---

## Scope

**In scope for v1**:

- `LoraConfig.quarot_seed` field in `lattice-tune`
- Transient rotation application to frozen weights during training
- Adapter metadata (`rotation_aware`, `quarot_seed`) written to exported safetensors
- Serving-side metadata detection and conditional counter-rotation bypass in `lattice-inference`
- Backward compatibility: legacy adapters (no metadata) continue to use ADR-045 counter-rotation
  path unchanged

**Out of scope (deferred)**:

- QuAILoRA initialization (quantization-error-minimizing LoRA init, arxiv:2410.14713) — separate
  concern, compatible with RoLoRA, defer to a future ADR
- QA-LoRA 4-bit adapter merging — deferred, no current use case
- Automatic seed detection from model artifact (requires ADR-045 §Prerequisites to ship first;
  until then, callers supply seed explicitly)
- Multi-adapter / MoLoRA routing in the rotated basis (v2, KG entity `W2-Barycenter-Adapter-Blending`)

---

## Architecture

```
TRAINING (lattice-tune)
┌─────────────────────────────────────────────────────────────────────┐
│  LoraConfig { rank, alpha, targets, quarot_seed: Some(seed) }       │
│                                                                     │
│  1. Load frozen BF16/F32 base weights                               │
│  2. Reconstruct R = RandomizedHadamard(seed, hidden_dim) via        │
│     lattice-inference primitive (no new implementation)             │
│  3. Apply rotation to frozen weight snapshots (transient, in-mem):  │
│     Input-side:  W_frozen ← W_frozen · R^T                         │
│     Output-side: W_frozen ← R · W_frozen                           │
│  4. Init A (Gaussian), B (zero) against rotated W_frozen           │
│  5. Train LoRA on rotated basis                                     │
│  6. Export: adapter A/B safetensors +                               │
│     config { rotation_aware: true, quarot_seed: seed }              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (adapter .safetensors)
SERVING (lattice-inference)
┌─────────────────────────────────────────────────────────────────────┐
│  load_lora_adapter()                                                │
│                                                                     │
│  Read adapter metadata:                                             │
│    rotation_aware: true  → upload A, B directly (no WHT)           │
│    rotation_aware: false │                                          │
│    or absent             → apply counter-rotation (ADR-045 path)   │
│                            warn! log for visibility                 │
│                                                                     │
│  Seed mismatch (model quarot.seed ≠ adapter quarot_seed) → Error   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
METAL FORWARD (per token, unchanged from ADR-045)
┌─────────────────────────────────────────────────────────────────────┐
│  base_out = Q4_GEMV(W_rot, h_rot)          [existing]               │
│  lora_out = scale * B @ (A @ h_rot)        [existing kernel]        │
│  out = base_out + lora_out                                          │
│  (A and B are already basis-correct regardless of training path)    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **ADR-045 counter-rotation only (current)** | No training changes needed; works with existing PEFT adapters | Adapter trained against wrong noise distribution at Q4; SNR gap grows at lower bit-widths | Keep as legacy path; insufficient for new Q4 adapter training |
| **RoLoRA (this ADR)** | Adapter trained in correct basis; +29.5pp reported on W4A4; zero serving-side overhead for rotation-aware adapters; seed already deterministic | Requires rotation-aware training infra; not compatible with generic PEFT export unless seed metadata added | **Accept** |
| **QuAILoRA initialization** (arxiv:2410.14713) | Minimizes quantization error at LoRA init; composable with RoLoRA | Orthogonal concern (init strategy, not basis alignment); separate ADR | Defer to ADR-055; composable with this decision |
| **Runtime rotation at each training step** | Avoids modifying frozen weights | Per-step `d×d` WHT overhead on all target projections during training; unnecessary given seed is fixed | Reject — one-time absorption is equivalent and free |
| **Separate RoLoRA training crate** | Clean separation | Violates `Π_AEP` (FindExisting > Create); `LoraConfig` already has the right extension point | Reject |

---

## Risks

1. **ADR-045 seed persistence prerequisite is still open.** Until `quantize_quarot` writes the
   `quarot.seed` to `config.json`, the serving-side auto-detection cannot verify seed alignment.
   Mitigation: callers supply `quarot_seed` explicitly (same as current ADR-045 behavior). The
   seed mismatch error fires immediately if the wrong seed is passed. This ADR does not ship
   until the seed persistence ticket is resolved.

2. **Frozen weight rotation doubles peak memory during training.** Rotating all frozen weights
   in-memory requires holding both the original and rotated snapshots transiently (or recomputing
   from the original on the fly). Mitigation: apply rotation per-layer and discard immediately;
   never hold the full rotated model in memory simultaneously. Cost: `O(max_layer_size)` working
   buffer, not `O(total_params)`.

3. **Legacy PEFT adapter ecosystem.** Adapters exported from Hugging Face PEFT are trained on
   unrotated weights and have no rotation metadata. These use the ADR-045 counter-rotation path
   unchanged. Users fine-tuning from Lattice's own training pipeline benefit from RoLoRA; users
   loading third-party adapters do not, but are no worse than before.

4. **Seed confidentiality.** If the QuaRot seed is treated as proprietary (e.g., leaked seed
   would allow reconstruction of the rotation applied to the weights), storing it in the adapter
   metadata exposes it. At current scale this is not a concern; flag for review before any
   commercial model distribution.

---

## References

- RoLoRA: Zhu et al., EMNLP 2024 — "RoLoRA: Fine-tuning Rotated Outlier-Free LLMs for Effective
  Weight-Activation Quantization", arxiv:2407.08044
- QuAILoRA: arxiv:2410.14713 — quantization-error-aware LoRA initialization (deferred to ADR-055)
- ADR-044: QuaRot Hadamard-rotated 4-bit quantization (rotation math, `RandomizedHadamard`)
- ADR-045: QuaRot + LoRA composition at inference time (serving-side counter-rotation)
- ADR-043: LoRA serving verification (correctness framework, `LoraHook` trait)
- ADR-031: LoRA management (adapter lifecycle)
- KG: `LoRA SNR vs Base Quantization Noise`, `QuaRot (Ashkboos 2024)` (86ec6a4f),
  `LoRA-Low-Rank-Adaptation` (e916fb8b), `W2-Barycenter-Adapter-Blending` (e6a40019)
- Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models", arxiv:2106.09685
- Ashkboos et al. 2024, "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", arxiv:2404.00456

---

## Knowledge Graph

- New entity on implementation PR: `RoLoRA` (kind: concept, type: technique, status: proposed)
  - `extends` → `LoRA-Low-Rank-Adaptation`
  - `composed_with` → `QuaRot (Ashkboos 2024)`
  - `introduced_by` → `Zhu et al. EMNLP 2024`
  - `competes_with` → `QuAILoRA` (alternative quantization-aware LoRA init approach)
