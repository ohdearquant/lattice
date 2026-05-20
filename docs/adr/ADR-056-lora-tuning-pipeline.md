---
status: Proposed
date: 2026-05-19
supersedes: []
depends_on: [ADR-008, ADR-027]
related: [ADR-043, ADR-045, ADR-054]
---

# ADR-056: LoRA Tuning Pipeline

**Status**: Proposed\
**Date**: 2026-05-19\
**Crate**: `lattice-tune`

---

## 1. Context

Lattice has a complete LoRA _serving_ stack. ADR-008 defines `LoraHook` as a
`Send + Sync` trait in `lattice-inference`; `platform/tune` implements it so that
`LoraAdapter::apply` calls the forward-only scalar kernel
(`crates/tune/src/lora/apply.rs:23`). ADR-031 covers three-layer adapter
representation (`LoraConfig`, `LoraLayer`, `LoraAdapter`) and PEFT/MLX safetensors
import. ADR-043 verifies hook call sites across all twelve Qwen3.5 projection
families. ADR-045 handles QuaRot+LoRA composition at serving time. ADR-054
(Proposed) describes RoLoRA training but is not yet implemented and explicitly
depends on an end-to-end training base that does not exist.

`lattice-tune` also has finetuning _scaffolding_. ADR-027 declares the crate's
three capabilities: LoRA adapter loading, knowledge distillation, and model
registry. ADR-032 defines `TrainingCallback` lifecycle hooks. ADR-033 defines JIT
adaptation including a `LowRank` strategy. ADR-034 defines a dataset pipeline for
intent classification. All of this scaffolding is in place, but the critical
training operations are placeholders:

- `TrainingLoop::train_batch` simulates loss with synthetic values and does not
  compute real forward or backward passes
  (`crates/tune/src/train/loop/mod.rs:205`, `crates/tune/src/train/loop/mod.rs:214`).
- `GpuTrainer::backprop_cpu` uploads placeholder gradients; the AdamW GPU
  dispatch submits an empty command encoder with a TODO comment
  (`crates/tune/src/train/gpu/mod.rs:321`, `crates/tune/src/train/gpu/optimizers.rs:141`).
- `JitAdapter` with `LowRank { rank }` freezes layers but does not allocate or
  train LoRA A/B matrices
  (`crates/tune/src/train/jit.rs:418`, `docs/adr/ADR-033-jit-adaptation.md:106`).
- `LoraConfig` has only `rank`, `alpha`, and `target_modules`; no trainable
  parameter buffers, gradient buffers, or optimizer state exist in the current
  `LoraAdapter` (`crates/tune/src/lora/mod.rs:43`).
- `Checkpoint::new` initializes `weights` and `optimizer_state` as empty byte
  slices; no save or load function is defined
  (`crates/tune/src/train/loop/checkpoint.rs:119`,
  `crates/tune/src/train/loop/checkpoint.rs:120`).
- `lora/safetensors.rs` exposes only `load_peft_safetensors`; no public LoRA
  adapter export function exists
  (`crates/tune/src/lora/safetensors.rs:213`).
- The current `Dataset` struct is in-memory and holds embedding/intent-label
  examples, not tokenized LLM batches; no `Dataset` trait exists
  (`crates/tune/src/data/dataset.rs:164`).

No Rust framework currently provides a drop-in PEFT-equivalent LoRA trainer
aligned with lattice's adapter representation and serving hook contract. Candle
delegates LoRA to an external `candle-lora` crate. Burn provides a full training
framework but no official LoRA APIs in the reviewed documentation. The missing
math is bounded: a token dataset trait, manual LoRA gradient equations for A and
B, a CPU AdamW update, checkpoint persistence, eval hooks, and a safetensors
export path.

---

## 2. Decision

We will implement an end-to-end LoRA supervised fine-tuning pipeline inside
`lattice-tune`, building on existing scaffolding rather than adopting an external
training framework. The pipeline is single-GPU-first; correctness on a CPU
reference path is the v1 gate. GPU acceleration is a post-v1 optimization that
reuses the existing `wgpu` scaffolding in `crates/tune/src/train/gpu/`.

The pipeline adds four bounded modules to `lattice-tune`: `train/lora/` (config,
dataset trait, gradient accumulation, AdamW, train loop), `train/lora/checkpoint`
(save/load and resume), `train/lora/eval` (loss/perplexity and generation smoke
hooks), and `lora/export` (PEFT-compatible safetensors adapter export plus sidecar
`adapter_config.json`). Base model weights are held read-only in a
`FrozenBaseWeights` wrapper; only LoRA A and B matrices receive gradient buffers
and optimizer state. The AdamW optimizer receives only `&mut LoraTrainableParams`
and never touches base weight arrays.

The training output is a `LoraAdapter` value that roundtrips through the existing
import path (`load_peft_safetensors`) and attaches to `Qwen35Model` via
`set_lora(Box<dyn LoraHook>)`. The end-to-end correctness gate is Phase 4: a
fixed-seed smoke run on a tiny model and synthetic dataset where loss decreases
and the exported adapter loads cleanly in `lattice-inference`.

---

## 3. Scope

**In scope**:

- `LoraDataset` trait for tokenized batches (`input_ids`, `attention_mask`,
  `labels`, `loss_mask`, sequence lengths), with `JsonlSftDataset` as the default
  implementation over `{prompt, response}` JSONL records.
- Manual LoRA gradient equations for A and B at each target projection, with
  column-vector and row-batch conventions documented in this ADR.
- Base weight freezing: no gradient buffers or optimizer slots for base model
  weights; only `LoraTrainableParams` (A and B matrices) are trainable.
- CPU AdamW reference optimizer (`AdamWState` with per-layer m/v moments for A
  and B, step counter, clip-by-norm, and serializable state).
- `LoraTrainLoop` lifecycle: init, train step, evaluate, save/load checkpoint,
  and finalize to `LoraAdapter`.
- `LoraCheckpoint`: adapter weights, optimizer state, scheduler state,
  step/epoch, RNG seed, and metric history, with save and load implementations.
- `EvalHook` trait with default `LossPerplexityEvalHook` and
  `GenerationSmokeEvalHook`.
- `safetensors_export(adapter: &LoraAdapter, path: &Path) -> anyhow::Result<()>`
  producing PEFT-compatible adapter weights.
- `peft_adapter_config_export` producing `adapter_config.json` alongside the
  safetensors file.
- Roundtrip verification: exported adapter loads via `load_peft_safetensors`,
  attaches via `Qwen35Model::set_lora`, and produces a non-zero generation delta.
- `LoraTrainingConfig` struct adding base model path, tokenizer path, dataset
  path, rank, alpha, target modules, dropout, adapter init, export flags, and
  PEFT key style to existing training controls.

**Out of scope**:

- Distributed training (single-GPU and CPU reference only in v1).
- Full HuggingFace PEFT API compatibility beyond adapter-weights-plus-config
  roundtrip.
- RoLoRA training (deferred; ADR-054 explicitly depends on this ADR's training
  base and adds `quarot_seed: Option<u64>` to `LoraConfig`).
- MoE LoRA (out of scope per ADR-054).
- Gradient checkpointing optimization (deferred to v2; v1 controls memory via
  batch size and accumulation steps).
- Automatic mixed-precision training (F32 weights only in v1).
- Multi-adapter serving (serving concern; out of scope here).

---

## 4. Design

### Training Configuration

`LoraTrainingConfig` extends the existing `TrainingConfig` defaults with
LoRA-specific fields. The existing config already covers epochs, batch size,
optimizer, LR schedule, regularization, early stopping, validation split, seeds,
checkpoint interval, log interval, accumulation, mixed precision, and memory
budget (`crates/tune/src/train/config/mod.rs:28`). ADR-056 adds a LoRA-specific
struct rather than overloading the classifier config.

```rust
pub struct LoraTrainingConfig {
    // Required paths; no defaults.
    pub base_model_path: std::path::PathBuf,
    pub tokenizer_path: std::path::PathBuf,
    pub dataset_path: std::path::PathBuf,
    pub output_dir: std::path::PathBuf,
    pub adapter_name: String,

    // LoRA adapter shape.
    pub rank: usize,                        // default: 16
    pub alpha: f32,                         // default: 32.0; scale = alpha / rank
    pub target_modules: Vec<String>,        // default: ["q_proj", "v_proj"]
    pub dropout: f32,                       // default: 0.0
    pub init: LoraInit,                     // default: KaimingAZeroB

    // Batching.
    pub max_seq_len: usize,                 // default: 2048
    pub batch_size: usize,                  // default: 1
    pub gradient_accumulation_steps: usize, // default: 8
    pub shuffle: bool,                      // default: true
    pub seed: u64,                          // default: 42
    pub completion_only_loss: bool,         // default: true

    // Optimizer.
    pub learning_rate: f32,                 // default: 2.0e-4
    pub adam_beta1: f32,                    // default: 0.9
    pub adam_beta2: f32,                    // default: 0.999
    pub adam_epsilon: f32,                  // default: 1.0e-8
    pub weight_decay: f32,                  // default: 0.0
    pub max_grad_norm: Option<f32>,         // default: Some(1.0)
    pub lr_schedule: LrScheduleConfig,      // default: CosineWarmup { warmup_ratio: 0.03 }

    // Loop control.
    pub num_epochs: usize,                  // default: 1
    pub max_steps: Option<usize>,           // default: None
    pub eval_steps: usize,                  // default: 100
    pub save_steps: usize,                  // default: 100
    pub log_steps: usize,                   // default: 10
    pub early_stopping: Option<EarlyStoppingConfig>,

    // Runtime.
    pub device: TrainDevice,               // default: CpuReference
    pub precision: TrainPrecision,         // default: F32

    // Export.
    pub export_safetensors: bool,          // default: true
    pub export_adapter_config_json: bool,  // default: true
    pub peft_key_style: PeftKeyStyle,      // default: LatticeQwen35
}

pub enum LoraInit {
    KaimingAZeroB,
    GaussianAZeroB,
}

pub enum TrainDevice {
    CpuReference,
    GpuWgpu,
}

pub enum TrainPrecision {
    F32,
    F16ActivationsF32Master,
}

pub enum PeftKeyStyle {
    LatticeQwen35,
    PeftLlamaLike,
}
```

### Dataset Trait

The existing `Dataset` struct holds embedding/intent-label examples
(`crates/tune/src/data/dataset.rs:164`). ADR-056 adds a separate trait for
tokenized LLM batches. The trait avoids assuming an in-memory layout so future
streaming or file-backed implementations are drop-in replacements.

```rust
pub struct TokenBatch {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u8>>,
    pub labels: Vec<Vec<i64>>,           // -100 means ignored; no loss contribution.
    pub loss_mask: Option<Vec<Vec<u8>>>, // completion-only or custom masking.
    pub seq_lens: Vec<usize>,
}

pub trait LoraDataset: Send {
    fn next_batch(&mut self, batch_size: usize) -> anyhow::Result<Option<TokenBatch>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn reset(&mut self);
    fn shuffle(&mut self, seed: u64);
    fn stats(&self) -> DatasetStats;
    fn split(
        &self,
        validation_fraction: f32,
        seed: u64,
    ) -> anyhow::Result<(Box<dyn LoraDataset>, Box<dyn LoraDataset>)>;
}
```

`JsonlSftDataset` is the default implementation: reads `{prompt, response}` or
`{messages}` JSONL records, tokenizes into next-token-prediction labels, and
applies completion-only loss masking when `completion_only_loss` is set.

### AdamW Optimizer and Trainable Parameters

Base weights and LoRA parameters are separated at the type level. The optimizer
receives only `&mut LoraTrainableParams`; there is no code path through which it
can access or mutate base model arrays.

```rust
pub struct LoraTrainableParams {
    pub layers: Vec<TrainableLoraLayer>,
}

pub struct TrainableLoraLayer {
    pub layer_idx: usize,
    pub module: String,
    pub a: Vec<f32>,      // shape: [rank × in_dim]; Kaiming init
    pub b: Vec<f32>,      // shape: [out_dim × rank]; zero init
    pub grad_a: Vec<f32>, // same shape as a; zeroed after each optimizer step
    pub grad_b: Vec<f32>, // same shape as b; zeroed after each optimizer step
    pub in_dim: usize,
    pub out_dim: usize,
    pub rank: usize,
    pub scale: f32, // alpha / rank
}

pub struct AdamW {
    pub config: AdamWConfig,
    pub state: AdamWState,
}

impl AdamW {
    pub fn new(config: AdamWConfig, params: &LoraTrainableParams) -> Self;
    pub fn step(&mut self, params: &mut LoraTrainableParams) -> anyhow::Result<()>;
    pub fn zero_grad(&mut self, params: &mut LoraTrainableParams);
    pub fn state_dict(&self) -> AdamWState;
    pub fn load_state_dict(&mut self, state: AdamWState) -> anyhow::Result<()>;
}

pub struct AdamWState {
    pub step: u64,
    pub m_a: Vec<Vec<f32>>, // first moment for A per layer
    pub v_a: Vec<Vec<f32>>, // second moment for A per layer
    pub m_b: Vec<Vec<f32>>, // first moment for B per layer
    pub v_b: Vec<Vec<f32>>, // second moment for B per layer
}
```

### TrainLoop Lifecycle

```rust
pub struct LoraTrainLoop<M, D>
where
    M: LoraTrainableModel,
    D: LoraDataset,
{
    pub config: LoraTrainingConfig,
    pub model: M,
    pub train_dataset: D,
    pub eval_dataset: Option<D>,
    pub adapter: LoraTrainableParams,
    pub optimizer: AdamW,
    pub eval_hooks: Vec<Box<dyn EvalHook<M>>>,
    pub state: LoraTrainState,
}

impl<M, D> LoraTrainLoop<M, D>
where
    M: LoraTrainableModel,
    D: LoraDataset,
{
    pub fn init(config: LoraTrainingConfig, model: M, dataset: D) -> anyhow::Result<Self>;
    pub fn train(&mut self) -> anyhow::Result<LoraTrainReport>;
    pub fn train_step(&mut self, batch: TokenBatch) -> anyhow::Result<TrainStepOutput>;
    pub fn evaluate(&mut self) -> anyhow::Result<EvalReport>;
    pub fn save_checkpoint(&self, path: &std::path::Path) -> anyhow::Result<()>;
    pub fn load_checkpoint(&mut self, path: &std::path::Path) -> anyhow::Result<()>;
    pub fn finalize(&mut self) -> anyhow::Result<LoraAdapter>;
}

pub trait LoraTrainableModel: Send {
    fn forward_with_lora(
        &mut self,
        batch: &TokenBatch,
        adapter: &LoraTrainableParams,
        capture: ActivationCapture,
    ) -> anyhow::Result<ModelOutput>;

    fn backward_lora(
        &mut self,
        loss_grad: &LossGradient,
        adapter: &mut LoraTrainableParams,
    ) -> anyhow::Result<()>;
}
```

The `train` outer loop:

```text
init()
for epoch in 0..config.num_epochs:
    train_dataset.shuffle(config.seed + epoch)
    while let Some(batch) = train_dataset.next_batch(batch_size)?:
        train_step(batch)
        if global_step % eval_steps == 0: evaluate()
        if global_step % save_steps == 0: save_checkpoint(checkpoint_dir)
finalize() -> LoraAdapter -> safetensors_export(output_dir)
```

### Gradient Flow and Base Weight Freezing

The forward pass at each target projection computes:

```text
y = W x + scale * B (A x)
z = A x              (cached for backward)
```

Backward — column-vector convention per projection site:

```text
g      = dL/dy
dL/dB  = scale * g z^T
dL/dA  = scale * B^T g x^T
```

Backward — row-batch convention (B = batch, T = seq, H = hidden, R = rank):

```text
Y      = X W^T + scale * (X A^T) B^T
G      = dL/dY
dL/dB  = scale * G^T (X A^T)     — shape (out_dim × rank) matching B
dL/dA  = scale * (G B)^T X       — shape (rank × in_dim) matching A
```

Accumulated `grad_a` and `grad_b` in `TrainableLoraLayer` are clipped by L2 norm
before the AdamW step if `max_grad_norm` is set.

Freezing is enforced structurally. Base model weights are held in
`FrozenBaseWeights<'a>` carrying a read-only reference:

```rust
pub struct FrozenBaseWeights<'a> {
    pub tensors: &'a BaseModelWeights,
}
```

`AdamW::step` accepts only `&mut LoraTrainableParams`, so no base weight array is
ever passed to the update function. The train loop may propagate upstream
gradients through base weight paths to compute activations required for `grad_a`
and `grad_b`, but it allocates no `grad_*` fields for base layers and passes no
base tensors to `AdamW`.

### Checkpoint

```rust
pub struct LoraCheckpoint {
    pub version: u32,
    pub adapter_name: String,
    pub adapter_weights: LoraAdapterState,
    pub optimizer_state: AdamWState,
    pub scheduler_state: LrSchedulerState,
    pub step: u64,
    pub epoch: usize,
    pub loss_history: Vec<MetricPoint>,
    pub eval_history: Vec<EvalReport>,
    pub config: LoraTrainingConfig,
    pub rng_seed: u64,
}

impl LoraCheckpoint {
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()>;
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self>;
}
```

This replaces the existing empty `Checkpoint` bytes
(`crates/tune/src/train/loop/checkpoint.rs:119`) with a real LoRA-specific layout
covering adapter A/B weights, AdamW moment tensors, scheduler step, global step,
epoch, RNG seed, and metric history. Saves are triggered at `save_steps` intervals
and on `finalize()`.

### EvalHook Trait

```rust
pub trait EvalHook<M: LoraTrainableModel>: Send {
    fn name(&self) -> &str;
    fn on_eval_start(&mut self, state: &LoraTrainState) -> anyhow::Result<()>;
    fn compute_metric(
        &mut self,
        model: &mut M,
        adapter: &mut LoraTrainableParams,
        batch: &TokenBatch,
    ) -> anyhow::Result<MetricValue>;
    fn on_eval_end(&mut self, report: &EvalReport) -> anyhow::Result<()>;
}

pub struct EvalReport {
    pub step: u64,
    pub metrics: Vec<MetricValue>,
}

pub struct MetricValue {
    pub name: String,
    pub value: f64,
    pub higher_is_better: bool,
}
```

Default implementations: `LossPerplexityEvalHook` (cross-entropy over eval
batches, perplexity = `exp(loss)`) and `GenerationSmokeEvalHook` (verifies that
attaching the adapter via `LoraHook` changes logits versus `NoopLoraHook` for
the same input).

### Safetensors Export

```rust
pub fn safetensors_export(
    adapter: &LoraAdapter,
    path: &std::path::Path,
) -> anyhow::Result<()>;

pub fn peft_adapter_config_export(
    adapter: &LoraAdapter,
    config: &LoraTrainingConfig,
    path: &std::path::Path,
) -> anyhow::Result<()>;
```

Export writes only A and B matrices for each `(layer_idx, module)` pair; no base
weights are included. Tensor keys follow `LatticeQwen35` or `PeftLlamaLike` style
as configured in `LoraTrainingConfig`. The `adapter_config.json` sidecar includes
`peft_type`, `r`, `lora_alpha`, `lora_dropout`, `target_modules`,
`base_model_name_or_path`, `bias`, and `fan_in_fan_out`.

The roundtrip gate: `safetensors_export` → `load_peft_safetensors` →
`Qwen35Model::set_lora(Box<dyn LoraHook>)` → non-zero generation delta. This
closes the import-only gap at `crates/tune/src/lora/safetensors.rs:213`.

### Supporting Types

```rust
pub struct AdamWConfig {
    pub lr: f32,           // default: 2.0e-4
    pub beta1: f32,        // default: 0.9
    pub beta2: f32,        // default: 0.999
    pub weight_decay: f32, // default: 0.0
    pub epsilon: f32,      // default: 1.0e-8
    pub max_grad_norm: Option<f32>,
}

pub struct ActivationCapture {
    pub layer_idx: usize,
    pub module: String,
    pub x: Vec<f32>, // input activation cached for backward
}

pub struct LossGradient {
    pub layer_idx: usize,
    pub module: String,
    pub grad: Vec<f32>, // dL/dy at LoRA output
}

pub struct BaseModelWeights {
    // Read-only projection tensors keyed by (layer_idx, module).
    pub tensors: std::collections::HashMap<(usize, String), Vec<f32>>,
}

pub struct ModelOutput {
    pub logits: Vec<f32>,
    pub loss: Option<f32>,
}

pub struct TrainStepOutput {
    pub loss: f32,
    pub grad_norm: f32,
    pub tokens: usize,
    pub step: u64,
}

pub struct LoraTrainState {
    pub epoch: usize,
    pub global_step: u64,
    pub best_eval_loss: f32,
    pub tokens_seen: u64,
    pub current_lr: f32,
}

pub struct LoraTrainReport {
    pub total_steps: u64,
    pub total_epochs: usize,
    pub final_loss: f32,
    pub best_eval_loss: f32,
    pub loss_history: Vec<MetricPoint>,
    pub duration_secs: f64,
}

pub struct DatasetStats {
    pub examples: usize,
    pub tokens: usize,
    pub max_seq_len: usize,
    pub average_seq_len: f32,
}

pub struct LrSchedulerState {
    pub current_lr: f32,
    pub warmup_steps_remaining: usize,
    pub decay_factor: f32,
    pub step: u64,
}

pub struct MetricPoint {
    pub step: u64,
    pub name: String,
    pub value: f64,
    pub epoch: usize,
}

pub struct LoraAdapterState {
    pub rank: usize,
    pub alpha: f32,
    pub target_modules: Vec<String>,
    pub layers: Vec<LoraLayerState>,
}

pub struct LoraLayerState {
    pub layer_idx: usize,
    pub module: String,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub in_dim: usize,
    pub out_dim: usize,
}

pub struct EarlyStoppingConfig {
    pub metric: String,  // default: "eval_loss"
    pub patience: usize, // default: 3
    pub min_delta: f32,  // default: 0.0
}

pub enum LrScheduleConfig {
    Constant,
    CosineAnnealing { min_lr: f32, total_steps: usize },
    LinearWarmup { warmup_steps: usize },
    WarmupCosine { warmup_steps: usize, min_lr: f32, total_steps: usize },
}
```

---

## 5. Implementation Plan

**Phase 1 — CPU correctness path (1 week)**

Add `LoraTrainingConfig` and supporting enums in
`crates/tune/src/train/lora/config.rs`. Implement `LoraDataset` trait and
`JsonlSftDataset` in `crates/tune/src/train/lora/dataset.rs`. Implement
`LoraTrainableParams`, `TrainableLoraLayer`, `AdamW`, and `AdamWState` in
`crates/tune/src/train/lora/optimizer.rs`. Implement `LoraTrainLoop::init`,
`train_step`, and the epoch loop in `crates/tune/src/train/lora/loop.rs`.

Gate: `gradient_flows_only_through_lora_params` and
`adamw_step_decreases_loss_on_synthetic_target` pass.

**Phase 2 — Checkpoint and eval (1 week)**

Implement `LoraCheckpoint::save` and `LoraCheckpoint::load` in
`crates/tune/src/train/lora/checkpoint.rs`. Implement `EvalHook`,
`LossPerplexityEvalHook`, and `GenerationSmokeEvalHook` in
`crates/tune/src/train/lora/eval.rs`. Wire checkpoint saves and eval calls into
`LoraTrainLoop::train`.

Gate: `checkpoint_roundtrip_preserves_optimizer_state` passes.

**Phase 3 — Safetensors export and inference roundtrip (1 week)**

Implement `safetensors_export` and `peft_adapter_config_export` in
`crates/tune/src/lora/export.rs`. Wire `LoraTrainLoop::finalize` to call
`safetensors_export`.

Gate: `safetensors_export_roundtrip_matches_in_memory` and
`exported_adapter_loads_in_lattice_inference` pass.

**Phase 4 — End-to-end smoke test (0.5 week)**

Write `test_end_to_end_smoke_loss_decreases` using a two-layer toy model,
synthetic JSONL dataset, fixed seed 42, and 50 steps. Assert final loss < 50% of
initial loss and that the exported adapter attaches to `Qwen35Model` without
error. This test becomes a required CI target for `lattice-tune`.

---

## 6. Test Matrix

| Test name                                        | Assertion                                                                                                                                      |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_next_batch_returns_correct_shape`       | `TokenBatch` fields have dimensions `(batch_size, seq_len)` for `input_ids`, `attention_mask`, `labels`, and `loss_mask`.                      |
| `adamw_step_decreases_loss_on_synthetic_target`  | Loss after 10 AdamW steps on a fixed-seed quadratic synthetic target is strictly less than the initial loss.                                   |
| `gradient_flows_only_through_lora_params`        | After `train_step`, `grad_a` and `grad_b` in `LoraTrainableParams` are non-zero; `FrozenBaseWeights` tensors are bitwise unchanged.            |
| `base_weights_unchanged_after_train_step`        | Byte-for-byte comparison of all base weight tensors before and after 100 train steps shows no difference.                                      |
| `checkpoint_roundtrip_preserves_optimizer_state` | Save `LoraCheckpoint`, load into a fresh `AdamW`, verify `state.step`, `m_a`, `v_a`, `m_b`, and `v_b` are element-wise equal.                  |
| `safetensors_export_roundtrip_matches_in_memory` | `safetensors_export` then `load_peft_safetensors` produces a `LoraAdapter` whose A and B tensors are bitwise identical to the trained adapter. |
| `exported_adapter_loads_in_lattice_inference`    | Export adapter, load via `load_peft_safetensors`, attach to `Qwen35Model::set_lora`, run forward: output differs from `NoopLoraHook` baseline. |
| `end_to_end_smoke_loss_decreases`                | Fixed seed 42, 50 steps, synthetic JSONL, two-layer toy model: `final_loss < 0.5 * initial_loss`.                                              |
| `dataset_reset_reproduces_same_first_batch`      | After iterating to exhaustion and calling `reset()`, the first batch is identical to the original first batch (no-shuffle deterministic mode). |
| `adamw_zero_grad_clears_accumulation`            | After accumulating non-zero gradients across two `train_step` calls, `zero_grad` sets every `grad_a` and `grad_b` element to exactly zero.     |
| `eval_hook_on_step_end_receives_correct_metrics` | `EvalHook::on_eval_end` is called with an `EvalReport` whose `step` matches `global_step` and loss value matches the batch CE loss.            |
| `peft_adapter_config_export_valid_json`          | `peft_adapter_config_export` writes valid JSON containing `r`, `lora_alpha`, and `target_modules` matching `LoraConfig` fields.                |

---

## 7. Alternatives

| Alternative                     | Pros                                                                                                                        | Cons                                                                                                                                                                                      | Rejection rationale                                                                                                                                                                                      |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Wrap `candle-train`             | Rust-native; close to the HuggingFace ecosystem; safetensors and tokenizers integration available out of the box.           | External `candle-lora` handles adapter semantics separately; lattice must still own key naming, checkpoint policy, and roundtrip; adds a dependency absent from `crates/tune/Cargo.toml`. | Not a drop-in solution for the missing pipeline math; trading one bounded implementation effort for a larger dependency-alignment effort with no certainty of matching the existing `LoraHook` contract. |
| Integrate burn-rs               | Strongest Rust training-framework surface: autodiff, AdamW with moment state, checkpoint directory semantics, GPU backends. | Requires full framework adoption; reviewed docs showed no official Burn LoRA APIs; high architectural divergence from existing lattice hooks, adapter structs, and config conventions.    | Too much divergence for a pipeline whose missing math is bounded and testable; CPU-reference AdamW and manual gradients avoid framework lock-in risk without sacrificing correctness.                    |
| Python HF Transformers via PyO3 | Richest LoRA/SFT ecosystem; PEFT config, `SFTTrainer` lifecycle, and adapter save semantics are mature and well-tested.     | Python runtime dependency; does not exercise the `LoraHook` bridge until after export; non-Rust training path weakens the train-to-serve roundtrip contract.                              | Directly conflicts with the goal of owning lattice's LoRA tuning infrastructure; the roundtrip gate requires that training produce adapters the existing serving stack can load without a Python step.   |

---

## 8. Risks and Mitigation

**Manual gradient equations are wrong for some projection shape or batch
convention** → Add synthetic finite-difference tests for A and B gradients per
target projection before closing Phase 1. `gradient_flows_only_through_lora_params`
and `base_weights_unchanged_after_train_step` are required Phase 1 gates.

**Loss explodes or produces NaN within the first few steps** → Default
`max_grad_norm = Some(1.0)` with full configurability (QLoRA defaults to 0.3).
The `train_step` implementation checks for NaN in loss and gradient norm before
the AdamW update and aborts with a diagnostic error instead of writing a corrupt
checkpoint.

**Memory is higher than expected despite LoRA parameter efficiency** → Activation
gradients required to compute `grad_a` and `grad_b` can dominate memory at long
sequence lengths, more so than the LoRA parameter count. V1 defaults to
`batch_size = 1` and `gradient_accumulation_steps = 8`. Gradient checkpointing
(recompute activations on backward) is deferred to v2 and must be added before
`max_seq_len` is increased beyond 2048.

**Export-load divergence between safetensors key naming conventions** → The
export must produce keys that pass the same validation rules applied by
`load_peft_safetensors`: dtype, shape, finite values, orphaned A/B pairs, and
rank (`crates/tune/src/lora/safetensors.rs:213`). The `PeftKeyStyle` enum allows
switching between the lattice-native format and a PEFT/Llama-style format.
`safetensors_export_roundtrip_matches_in_memory` is a required v1 gate.

**ADR-054 (RoLoRA) is blocked until this ADR lands** → RoLoRA training adds
`quarot_seed: Option<u64>` to `LoraConfig` and builds on `LoraTrainingConfig` and
`LoraTrainLoop`. The ADR-056 API surface must remain stable after v1; breaking
changes require a new ADR.

---

## References

- `crates/tune/src/lora/mod.rs:43` — `LoraConfig` current fields
- `crates/tune/src/lora/apply.rs:23` — forward-only `apply_lora` scalar kernel
- `crates/tune/src/lora/safetensors.rs:213` — `load_peft_safetensors` (import; no export)
- `crates/tune/src/train/loop/mod.rs:205` — placeholder `train_batch` with simulated loss
- `crates/tune/src/train/loop/checkpoint.rs:119` — `Checkpoint::new` empty weights bytes
- `crates/tune/src/train/gpu/mod.rs:321` — placeholder backprop upload
- `crates/tune/src/train/gpu/optimizers.rs:141` — incomplete AdamW GPU dispatch
- `crates/tune/src/train/jit.rs:418` — `LowRank` freeze without A/B matrix training
- `crates/tune/src/data/dataset.rs:164` — concrete embedding/intent dataset struct
- `crates/inference/src/lora_hook.rs:9` — `LoraHook` trait definition
- `crates/inference/src/model/qwen35/model.rs:63` — `Qwen35Model::set_lora`
- ADR-008 — LoRA Injection via Trait Hook
- ADR-027 — Fine-Tuning Pipeline
- ADR-031 — LoRA Adapter Management
- ADR-032 — Training Callbacks
- ADR-033 — JIT Adaptation
- ADR-034 — Dataset Pipeline
- ADR-043 — LoRA Serving Verification
- ADR-045 — QuaRot + LoRA Composition
- ADR-054 — Rotation-Aware LoRA Training (RoLoRA; depends on this ADR)
- Hu et al. 2021 — "LoRA: Low-Rank Adaptation of Large Language Models" — https://arxiv.org/abs/2106.09685
- Dettmers et al. 2023 — "QLoRA: Efficient Finetuning of Quantized LLMs" — https://arxiv.org/abs/2305.14314
