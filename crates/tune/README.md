# lattice-tune

Knowledge distillation and training infrastructure for lattice neural models.

## Features

- **Knowledge Distillation**: Teacher (LLM) -> soft labels -> student (tiny NN) pipeline
- **Multi-Provider Teachers**: Claude, OpenAI, Gemini, local models (Ollama)
- **Model Registry**: Semver versioning with lineage tracking
- **Endpoint Security**: TLS enforcement, domain whitelisting, checksum verification

## Current Implementation Status

`DistillationPipeline` and the CPU `TrainingLoop` currently provide orchestration
placeholders, not end-to-end training. `DistillationPipeline::label_single` does
not contact a teacher API and fails closed with `TuneError::TeacherApi`. Fixed
labels are available only through the non-default `simulated-teacher` feature and
explicitly named simulation methods. `TrainingLoop::train` drives batching, callbacks, metrics, early
stopping, and in-memory checkpoints, but its loss and accuracy are simulated and
it does not update model weights. Do not treat either result as evidence of live
teacher labeling or trained parameters.

## Pipeline Overview

```text
Raw Data -> Teacher (LLM) -> Soft Labels -> Dataset -> Training -> Model -> Registry
                                                          |
                                                     Deployment
```

## Quick Start

```rust
use lattice_tune::data::{TrainingExample, IntentLabels, Dataset, DatasetConfig};

// Create training examples
let examples = vec![
    TrainingExample::new(
        vec![vec![0.1, 0.2, 0.3]],  // context embeddings
        vec![0.4, 0.5, 0.6],        // message embedding
        IntentLabels::continuation(0.8),
    ),
];

// Create a dataset
let dataset = Dataset::from_examples(examples);
let stats = dataset.stats();
println!("Dataset has {} examples", stats.num_examples);
```

## Teacher Configuration

### Providers

| Provider | Model Examples                    | Notes             |
| -------- | --------------------------------- | ----------------- |
| `Claude` | claude-sonnet-4, claude-3-5-haiku | Anthropic API     |
| `OpenAI` | gpt-4-turbo-preview               | OpenAI API        |
| `Gemini` | gemini-pro                        | Google AI         |
| `Local`  | llama, mistral                    | Ollama-compatible |

### TeacherConfig

```rust
use lattice_tune::distill::{TeacherConfig, TeacherProvider};

// Pre-configured providers
let claude = TeacherConfig::claude_sonnet();
let gpt = TeacherConfig::gpt4();
let gemini = TeacherConfig::gemini_pro();
let local = TeacherConfig::local("llama2", "http://localhost:11434");

// Custom configuration with builder
let config = TeacherConfig::builder()
    .provider(TeacherProvider::OpenAI)
    .model_id("gpt-4")
    .temperature(0.3)
    .max_tokens(1024)
    .timeout_ms(30000)
    .max_retries(3)
    .build();

// Validate before use
config.validate()?;
```

### Endpoint Security

```rust
use lattice_tune::distill::EndpointSecurity;

// Default secure settings (TLS required, known domains only)
let security = EndpointSecurity::default_secure();

// For local models (no TLS required)
let local_security = EndpointSecurity::for_local();

// Custom security with certificate pinning
let custom = EndpointSecurity::default_secure()
    .with_cert_fingerprint("a94a8fe5ccb19ba61c4c0873d391e987982fbbd3...")
    .with_model_checksum("sha256:abc123...");

// Verify endpoint before use
security.verify_endpoint("https://api.anthropic.com/v1")?;
```

## Training Examples

### TrainingExample

```rust
use lattice_tune::data::{TrainingExample, IntentLabels, ExampleMetadata};

// Create with soft labels
let example = TrainingExample::new(
    vec![vec![0.1, 0.2, 0.3]; 3],  // 3 context message embeddings
    vec![0.4, 0.5, 0.6],           // current message embedding
    IntentLabels::explicit_query(0.9),
);

// Access fields
println!("ID: {}", example.id);
println!("Context messages: {}", example.context_embeddings.len());
```

### IntentLabels

Soft labels for 6 intent classes.

```rust
use lattice_tune::data::IntentLabels;

// Single-class dominant labels
let continuation = IntentLabels::continuation(0.8);
let topic_shift = IntentLabels::topic_shift(0.7);
let explicit_query = IntentLabels::explicit_query(0.9);
let person_lookup = IntentLabels::person_lookup(0.85);
let health_check = IntentLabels::health_check(0.75);
let task_status = IntentLabels::task_status(0.8);

// Access probabilities
println!("Continuation: {}", continuation.continuation);
println!("Topic shift: {}", continuation.topic_shift);
```

## Dataset Management

### Creating a Dataset

```rust
use lattice_tune::data::{Dataset, DatasetConfig, TrainingExample, IntentLabels};

let examples: Vec<TrainingExample> = (0..100)
    .map(|i| TrainingExample::new(
        vec![vec![0.1, 0.2, 0.3]; 3],
        vec![0.4, 0.5, 0.6],
        IntentLabels::continuation(0.8),
    ))
    .collect();

let mut dataset = Dataset::from_examples(examples);

// Configure batching
let config = DatasetConfig::with_batch_size(32)
    .shuffle(true)
    .seed(42);

dataset.set_config(config)?;

// Split explicitly: 80% training, 20% validation
let (train, validation) = dataset.split(0.8)?;

let stats = dataset.stats();
println!("Examples: {}", stats.num_examples);
println!("Embedding dim: {}", stats.embedding_dim);
```

### Validation Split

```rust
use lattice_tune::data::Dataset;

let dataset = Dataset::from_examples(examples);
let (train, validation) = dataset.split(0.8)?; // 20% for validation
```

## Distillation Pipeline

This example shows the default fail-closed behavior. The pipeline validates the
teacher configuration, but it does not fabricate labels when live transport is
unavailable.

```rust
use lattice_tune::distill::{TeacherConfig, DistillationPipeline, RawExample};

// Configure teacher
let teacher = TeacherConfig::claude_sonnet();

// Create pipeline
let mut pipeline = DistillationPipeline::with_teacher(teacher)?;

// Create raw example (text, not embeddings)
let raw = RawExample::new(
    vec!["Hello".to_string(), "How are you?".to_string()],
    "What's the weather like?",
);

let error = pipeline.label_single(&raw).unwrap_err();
assert!(error.to_string().contains("not configured"));

// Check statistics
let stats = pipeline.stats();
println!("Successful: {}", stats.successful); // 0
println!("Failed: {}", stats.failed);
```

Tests and examples that deliberately need fixed output can enable the
`simulated-teacher` feature and call `label_single_simulated` or
`label_batch_simulated`. These methods remain separate from the default live
labeling methods even when the feature is enabled.

## Training

### TrainingConfig

```rust
use lattice_tune::train::{TrainingConfig, OptimizerConfig, LRSchedule, EarlyStopping};

let config = TrainingConfig::default()
    .epochs(100)
    .batch_size(32)
    .learning_rate(0.001)
    .optimizer(OptimizerConfig::adam(0.001))
    .lr_schedule(LRSchedule::CosineAnnealing { min_lr: 1e-6, t_max: 100 })
    .early_stopping(EarlyStopping::val_loss(10));  // Stop if no improvement for 10 epochs

config.validate()?;
```

### Learning Rate Schedules

| Schedule                                      | Description                                  |
| --------------------------------------------- | -------------------------------------------- |
| `Constant`                                    | Fixed learning rate                          |
| `StepDecay { step_size, gamma }`              | Reduce by `gamma` every `step_size` epochs   |
| `CosineAnnealing { min_lr, t_max }`           | Smooth decay to `min_lr` over `t_max` epochs |
| `OneCycle { max_lr, pct_start, total_steps }` | Warmup + annealing                           |
| `LinearWarmup { warmup_steps }`               | Linear warmup phase                          |

### Training Loop

The CPU loop below reports simulated metrics and does not forward through or
update a neural network.

```rust
use lattice_tune::train::{TrainingConfig, TrainingLoop};
use lattice_tune::data::Dataset;

let config = TrainingConfig::quick();  // Fast training preset
let mut trainer = TrainingLoop::new(config)?;

let metrics = trainer.train(&mut dataset)?;
println!("Simulated final loss: {:.4}", metrics.final_train_loss);
println!("Epochs completed: {}", metrics.epochs_completed);
```

### Checkpointing

`Checkpoint` is an in-memory record. The training loop sends periodic records to
`TrainingCallback::on_checkpoint`; it does not write `checkpoint_dir`, and
`Checkpoint` has no file `save`, `load`, or `into_network` methods. Callers own
any serialization and persistence. Resuming restores only loop epoch, global
step, and metrics—not model weights or optimizer state.

```rust
let checkpoint = trainer.checkpoint();
trainer.resume_from(&checkpoint);
```

## Model Registry

### RegisteredModel

```rust
use lattice_tune::registry::{ModelRegistry, RegisteredModel, ModelMetadata};

// Create in-memory registry
let registry = ModelRegistry::in_memory();

// Create model with metadata
let metadata = ModelMetadata::classifier(768, 6, 10000)
    .dataset("conversation_intents", 10000);

let model = RegisteredModel::new("intent_classifier", "1.0.0")
    .with_metadata(metadata)
    .with_description("Intent classification model");

// Register with weights
let weights = vec![0u8; 1000];
let id = registry.register(model, &weights)?;

// Retrieve the model
if let Some(loaded) = registry.get("intent_classifier", "1.0.0") {
    println!("Loaded: {}", loaded.full_name());
}
```

### Model Status Lifecycle

This is the conventional lifecycle, not an enforced transition graph.
`update_status` can assign any status, and `promote_to_production` does not
require a validated or staged target. Callers must enforce deployment gates.

```text
Pending -> Validated -> Staged -> Production
                                    |
                                    v
                              Archived/Deprecated
```

```rust
use lattice_tune::registry::ModelStatus;

// Status transitions
let status = ModelStatus::Pending;    // Just registered
let status = ModelStatus::Validated;  // Tests passed
let status = ModelStatus::Staged;     // Ready for deployment
let status = ModelStatus::Production; // Live in production
let status = ModelStatus::Archived;   // Superseded by newer version
```

### Versioning & Lineage

```rust
use lattice_tune::registry::{RegisteredModel, ModelMetadata};

// Base model
let base = RegisteredModel::new("intent_classifier", "1.0.0");
let base_id = registry.register(base, &weights)?;

// Fine-tuned model with parent reference
let metadata = ModelMetadata::classifier(768, 6, 15000);

let finetuned = RegisteredModel::new("intent_classifier", "1.1.0")
    .with_metadata(metadata)
    .with_parent(base_id);  // Store the caller-supplied parent reference
```

Registration does not validate `parent_id` or append the fine-tuned model's ID
to the parent's `children` field. Applications that need reverse lineage must
maintain both sides themselves.

## Feature Flags

| Feature             | Default | Description                                                                                                               |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------- |
| `std`               | Yes     | Standard library support                                                                                                  |
| `sqlite`            | Yes     | SQLite-backed model registry storage (implies `serde`)                                                                    |
| `serde`             | No      | Serialization (propagates to lattice-fann)                                                                                |
| `gpu`               | No      | GPU-accelerated forward/backward passes and validation[^gpu-limitation]                                                   |
| `gpu-tests`         | No      | GPU tests requiring hardware                                                                                              |
| `safetensors`       | No      | PEFT-compatible LoRA adapter serialization                                                                                |
| `inference-hook`    | No      | `impl LoraHook for LoraAdapter` (pulls in `lattice-inference`)                                                            |
| `train-backward`    | No      | Backward/gradient training surface (pulls in `lattice-inference`)                                                         |
| `mixture`           | No      | Online adapter-selector refit from preference feedback (experimental, bench-gated; pulls in `lattice-fann/online-router`) |
| `simulated-teacher` | No      | Fixed labels for tests and examples; never enables the default labeling path                                              |

[^gpu-limitation]: **Current limitation**: `GpuTrainer::train_batch` returns
    `Err` for every optimizer choice (Adam, AdamW, SGD-momentum, plain SGD,
    RMSprop) — the GPU-shader optimizer dispatch has no buffer bindings wired to
    the network's weight/gradient buffers, and the CPU-side plain-SGD arm has
    neither real gradient plumbing nor a mutable weight write-back path. Forward
    pass and loss computation work correctly; only the weight-update step is
    unimplemented. See [issue #797](https://github.com/ohdearquant/lattice/issues/797).
    This note will be removed once that wiring lands.

```toml
[dependencies]
lattice-tune = { version = "0.12" }                           # Default
lattice-tune = { version = "0.12", features = ["serde"] }     # With serialization
lattice-tune = { version = "0.12", features = ["gpu"] }       # GPU training
lattice-tune = { version = "0.12", features = ["safetensors"] } # LoRA adapter I/O
```

### Injecting an adapter into `lattice-inference`

Enable `inference-hook` to get `impl lattice_inference::lora_hook::LoraHook for
LoraAdapter`, so a trained adapter can be handed to a running inference engine
as a `Box<dyn LoraHook>`:

```toml
[dependencies]
lattice-tune = { version = "0.12", features = ["inference-hook", "safetensors"] }
lattice-inference = "0.12" # provides the LoraHook trait
```

```rust,ignore
use std::path::Path;
use lattice_tune::lora::LoraAdapter;
use lattice_inference::lora_hook::LoraHook; // not re-exported at the crate root

let adapter = LoraAdapter::from_safetensors(Path::new("adapter.safetensors"))?;
let hook: Box<dyn LoraHook> = Box::new(adapter);
// the engine calls hook.apply(layer_idx, module, x, output) on each projection
```

## Dependencies

This crate depends on `lattice-fann` for neural network primitives.

```toml
[dependencies]
lattice-tune = { version = "0.12" }
# lattice-fann is automatically included as a transitive dependency
```

## API Reference

### DistillationPipeline

```rust
impl DistillationPipeline {
    pub fn with_teacher(config: TeacherConfig) -> Result<Self>;
    pub fn label_single(&mut self, example: &RawExample) -> Result<LabelingResult>;
    pub fn label_batch(&mut self, examples: &[RawExample]) -> Vec<LabelingResult>;
    #[cfg(feature = "simulated-teacher")]
    pub fn label_single_simulated(&mut self, example: &RawExample) -> Result<LabelingResult>;
    #[cfg(feature = "simulated-teacher")]
    pub fn label_batch_simulated(&mut self, examples: &[RawExample]) -> Vec<LabelingResult>;
    pub fn stats(&self) -> &DistillationStats;
}
```

### TrainingLoop

```rust
impl TrainingLoop {
    pub fn new(config: TrainingConfig) -> Result<Self>;
    pub fn train(&mut self, dataset: &mut Dataset) -> Result<TrainingMetrics>;
    pub fn checkpoint(&self) -> Checkpoint;
    pub fn resume_from(&mut self, checkpoint: &Checkpoint);
}
```

### ModelRegistry

```rust
impl ModelRegistry {
    pub fn in_memory() -> Self;
    pub fn register(&self, model: RegisteredModel, weights: &[u8]) -> Result<Uuid>;
    pub fn get(&self, name: &str, version: &str) -> Option<RegisteredModel>;
    pub fn promote_to_production(&self, id: &Uuid) -> Result<()>;
    pub fn get_latest(&self, name: &str) -> Option<RegisteredModel>;
}
```

## Crate reference

The sections below are generated by `scripts/crate-readme.py` from the source tree and are refreshed after pushes to `main`. Text between the `generated` markers is overwritten; edit the source or the script instead.

<!-- generated:facts:begin -->

### Facts

- Version: `0.15.0`
- Minimum supported Rust version: `1.93`
- Edition: `2024`
- License: `MIT OR Apache-2.0`
- crates.io: <https://crates.io/crates/lattice-tune>
- API documentation: <https://docs.rs/lattice-tune>

<!-- generated:facts:end -->

<!-- generated:layout:begin -->

### Source layout

The tracked files under `src/`; each Rust file shows the first sentence of its `//!` module documentation.

<details>
<summary><code>src/</code>: 84 files</summary>

- `bin/`
  - `generate_lora.rs`: Qwen3.5 generation with LoRA adapter — proves PEFT/MLX-trained adapters work in Rust inference.
  - `train_grad.rs`
  - `train_grad_full.rs`
  - `train_grad_layer23/`
    - `main.rs`
- `data/`
  - `dataset.rs`: In-memory dataset filtering, batching, splitting, and statistics.
  - `dataset/`
    - `tests.rs`
  - `example.rs`: Training examples, soft intent labels, and traceability metadata.
  - `mod.rs`: Data contracts and in-memory batching for training.
- `distill/`
  - `mod.rs`: Knowledge-distillation interfaces for producing soft intent labels.
  - `pipeline/`
    - `distill.rs`: Orchestration from raw prompts to labeled examples.
    - `mod.rs`: Distillation pipeline for labeling training data
    - `tests.rs`
    - `types.rs`: Inputs, outputs, and accounting for the distillation pipeline.
  - `teacher/`
    - `config.rs`: Teacher selection, request policy, and endpoint-security configuration.
    - `mod.rs`: Teacher model configuration
    - `security.rs`: Endpoint policy for teacher-model connections.
    - `tests.rs`
- `error.rs`: Error vocabulary shared by lattice-tune subsystems.
- `lib.rs`: Training infrastructure for Lattice neural models.
- `lora/`
  - `apply.rs`: Runtime LoRA application: out += scale * (B @ (A @ x)).
  - `apply/`
    - `tests.rs`
  - `blend.rs`: Exact CPU blending of adapters into one higher-rank adapter.
  - `blend/`
    - `tests.rs`
  - `loader.rs`: Fail-closed manifest-driven loading for LoRA adapters.
  - `loader/`
    - `tests.rs`
  - `manifest.rs`: Versioned LoRA manifest schema and bounded JSON I/O.
  - `manifest/`
    - `tests.rs`
  - `mod.rs`: LoRA adapters, lightweight training helpers, and inference integration.
  - `online.rs`: One-event SGD refitting for a LoRA layer.
  - `online/`
    - `tests.rs`
  - `optimizer.rs`: Adam/AdamW state and exact single-layer LoRA gradient computation.
  - `optimizer/`
    - `tests.rs`
  - `safetensors.rs`: PEFT/MLX safetensors import and PEFT-compatible export for LoRA adapters.
  - `safetensors/`
    - `tests.rs`
  - `tests.rs`
  - `train.rs`: CPU micro-LoRA training over a bounded Qwen3.5 layer suffix.
  - `train/`
    - `tests.rs`
  - `train_core.rs`: Private backward tape for micro-LoRA training.
  - `train_core/`
    - `agreement_tests.rs`
    - `ffn_compute_range_tests.rs`
    - `ffn_seq_forward_tests.rs`
    - `gdn_lora_tests.rs`
    - `lm_head_logits_tests.rs`
    - `train_ctx_tests.rs`
- `registry/`
  - `live_model.rs`: Lock-free snapshots and atomic replacement for a live model record.
  - `mod.rs`: Versioned model records, artifact storage, and deployment helpers.
  - `model.rs`: Versioned registry records, metadata, status, and lineage fields.
  - `model/`
    - `tests.rs`
  - `rollback.rs`: Bounded in-memory audit records for model rollbacks.
  - `rollback/`
    - `tests.rs`
  - `safetensors_io.rs`: Safetensors serialization for flat `f32` weight vectors.
  - `safetensors_io/`
    - `tests.rs`
  - `shadow.rs`: Candidate-versus-production comparison for deployment decisions.
  - `shadow/`
    - `tests.rs`
  - `storage/`
    - `backends.rs`: In-memory, filesystem, and optional SQLite artifact storage backends.
    - `mod.rs`: Model storage backend
    - `query.rs`: Model query builder.
    - `registry.rs`: In-process versioned model index with pluggable artifact storage.
    - `tests.rs`
- `train/`
  - `config/`
    - `early_stopping.rs`: Early stopping configuration
    - `lr_schedule.rs`: Learning-rate schedule definitions and evaluation.
    - `mod.rs`: Training configuration, validation, and memory estimation.
    - `optimizer.rs`: Optimizer configuration
    - `regularization.rs`: Regularization settings and gradient-norm clipping.
  - `gpu/`
    - `builder.rs`: GPU trainer builder
    - `mod.rs`: GPU forward, loss, validation, and training-step infrastructure.
    - `optimizers.rs`: GPU optimizer dispatch.
    - `state.rs`: GPU buffer state management
    - `tests.rs`
    - `uniforms.rs`: GPU shader uniform structures
  - `jit.rs`: Just-in-time adaptation configuration, control flow, and layer-selection helpers.
  - `jit/`
    - `tests.rs`
  - `loop/`
    - `callbacks.rs`: Training callbacks
    - `checkpoint.rs`: In-memory training checkpoint representation.
    - `metrics.rs`: Per-epoch and aggregate training metrics.
    - `mod.rs`: Training-loop orchestration, callbacks, metrics, and checkpoints.
    - `state.rs`: Training state
  - `mod.rs`: Training orchestration: the training loop, optimizer configuration, checkpointing, early stopping, and GPU-accelerated training behind the `gpu` feature.
- `train_support.rs`: Support shared by the gradient-training executables and the layer-23 golden bridge test.
- `train_support/`
  - `full_driver.rs`: Shared exact multi-layer LoRA training driver.
  - `full_driver/`
    - `adapter_load_tests.rs`
    - `gdn_lora_ctor_tests.rs`
    - `run_bounds_tests.rs`
  - `tests.rs`

</details>

Targets declared by the manifest, including the ones Cargo discovers by directory:

- lib (1): `lattice_tune`
- bin (4): `generate_lora` (requires `safetensors`, `inference-hook`), `train_grad_full` (requires `train-backward`), `train_grad_layer23` (requires `train-backward`), `train_grad` (requires `train-backward`)
- example (1): `prompt_router` (requires `mixture`, `inference-hook`, `serde`)
- test (8): `bench_backward_737` (requires `train-backward`), `bert_cross_encoder_over_complete_lora`, `gdn_lora_wiring` (requires `train-backward`), `layer23_golden_bridge` (requires `train-backward`), `lora_apply_no_alloc`, `proptest_softmax`, `router_loop_closure` (requires `mixture`, `inference-hook`, `serde`), `training_pipeline`

<!-- generated:layout:end -->

<!-- generated:stats:begin -->

### Statistics

Rust lines by kind (code, comment, blank; doc comments count as comment), test functions (`#[test]` and `#[<runtime>::test]` attributes), items declared `pub` (not `pub(crate)`; reachability through private modules is not resolved) and occurrences of the `unsafe` keyword outside comments and strings.

| Area        | Files | Code  | Comment | Blank | Tests | `pub` items | `unsafe` |
| ----------- | ----- | ----- | ------- | ----- | ----- | ----------- | -------- |
| `src/`      | 84    | 19686 | 3628    | 2731  | 472   | 545         | 0        |
| `tests/`    | 8     | 1790  | 339     | 278   | 71    | 0           | 9        |
| `examples/` | 1     | 244   | 4       | 10    | 0     | 0           | 0        |
| total       | 93    | 21720 | 3971    | 3019  | 543   | 545         | 9        |

Other tracked files (the README excluded):

| Extension      | Files | Size      |
| -------------- | ----- | --------- |
| `.jsonl`       | 1     | 61 B      |
| `.md`          | 18    | 198.3 KiB |
| `.safetensors` | 1     | 26.6 KiB  |
| `.toml`        | 1     | 3.5 KiB   |
| `.txt`         | 1     | 695 B     |

<!-- generated:stats:end -->

<!-- generated:docs:begin -->

### In-crate documentation

<details>
<summary>18 documents</summary>

- [Changelog](CHANGELOG.md) (`CHANGELOG.md`)
- [docs/ADR-001-teacher-providers.md](docs/ADR-001-teacher-providers.md) (`docs/ADR-001-teacher-providers.md`)
- [docs/ADR-002-model-registry.md](docs/ADR-002-model-registry.md) (`docs/ADR-002-model-registry.md`)
- [lattice-tune documentation](docs/INDEX.md) (`docs/INDEX.md`)
- [ADR-001: Fine-Tuning Pipeline](docs/adr/ADR-001-finetuning-pipeline.md) (`docs/adr/ADR-001-finetuning-pipeline.md`)
- [ADR-003: Knowledge Distillation Pipeline](docs/adr/ADR-003-knowledge-distillation.md) (`docs/adr/ADR-003-knowledge-distillation.md`)
- [ADR-004: LoRA Adapter Management](docs/adr/ADR-004-lora-adapter-management.md) (`docs/adr/ADR-004-lora-adapter-management.md`)
- [ADR-005: Training Callbacks](docs/adr/ADR-005-training-callbacks.md) (`docs/adr/ADR-005-training-callbacks.md`)
- [ADR-006: JIT Adaptation](docs/adr/ADR-006-jit-adaptation.md) (`docs/adr/ADR-006-jit-adaptation.md`)
- [ADR-007: Dataset Pipeline](docs/adr/ADR-007-dataset-pipeline.md) (`docs/adr/ADR-007-dataset-pipeline.md`)
- [Training data](docs/data.md) (`docs/data.md`)
- [lattice-tune design](docs/design.md) (`docs/design.md`)
- [Distillation](docs/distill.md) (`docs/distill.md`)
- [LoRA Training Core](docs/lora-core.md) (`docs/lora-core.md`)
- [LoRA Safetensors and Governed Loading](docs/lora-io.md) (`docs/lora-io.md`)
- [Model and adapter registry](docs/registry.md) (`docs/registry.md`)
- [Training pipeline](docs/train.md) (`docs/train.md`)
- [Prompt embedding routing experiment](examples/prompt_router.md) (`examples/prompt_router.md`)

</details>

<!-- generated:docs:end -->

<!-- generated:features:begin -->

### Features

Default set: `sqlite`, `std`. Each description is the first sentence of the comment above the feature in `Cargo.toml`.

- `gpu`: GPU-accelerated training via lattice-fann Enables `lattice-fann/gpu`, `dep:wgpu`, `dep:bytemuck`.
- `gpu-tests`: Enable GPU tests that require actual GPU hardware (disabled in CI by default) Enables `gpu`.
- `inference-hook`: Enable `impl LoraHook for LoraAdapter` — lets lattice-tune adapters be passed directly to inference model hooks. Enables `dep:lattice-inference`.
- `mixture`: Online adapter-selector refit from preference feedback (experimental, bench-gated). Enables `lattice-fann/online-router`, `lattice-inference?/mixture`.
- `safetensors`: Safe checkpoint serialization (recommended for production) Enables `dep:safetensors`.
- `serde`: Enables `dep:serde`, `dep:serde_json`, `lattice-fann/serde`.
- `simulated-teacher`: Fixed labels for tests and examples. Enables nothing.
- `sqlite` (default): SQLite-backed model registry storage Enables `dep:rusqlite`, `serde`.
- `std` (default): Enables nothing.
- `train-backward`: Enables `dep:lattice-inference`, `lattice-inference/train-backward`, `lattice-inference/f16`, `safetensors`, `serde`.

<!-- generated:features:end -->

<!-- generated:dependencies:begin -->

### Workspace dependencies

- Depends on: `lattice-fann`, `lattice-inference` (optional, enabled by `inference-hook`, `train-backward`; features `f16`, `qwen35`)
- Used by: no other crate in this workspace

<!-- generated:dependencies:end -->

<!-- generated:adrs:begin -->

### Architecture decisions

ADRs under `docs/adr/` that name `lattice-tune` or `crates/tune/`:

<details>
<summary>31 ADRs</summary>

- [ADR-020: Neural Network Primitives](../../docs/adr/ADR-020-neural-net-primitives.md)
- [ADR-023: Activation Functions](../../docs/adr/ADR-023-activation-functions.md)
- [ADR-024: Network Builder Pattern](../../docs/adr/ADR-024-network-builder.md)
- [ADR-026: Training Loop](../../docs/adr/ADR-026-training-loop.md)
- [ADR-027: Fine-Tuning Pipeline](../../docs/adr/ADR-027-finetuning-pipeline.md)
- [ADR-028: Multi-Provider Teacher Strategy](../../docs/adr/ADR-028-teacher-providers.md)
- [ADR-029: Model Registry with Lineage](../../docs/adr/ADR-029-model-registry.md)
- [ADR-030: Knowledge Distillation Pipeline](../../docs/adr/ADR-030-knowledge-distillation.md)
- [ADR-031: LoRA Adapter Management](../../docs/adr/ADR-031-lora-adapter-management.md)
- [ADR-032: Training Callbacks](../../docs/adr/ADR-032-training-callbacks.md)
- [ADR-033: JIT Adaptation](../../docs/adr/ADR-033-jit-adaptation.md)
- [ADR-034: Dataset Pipeline](../../docs/adr/ADR-034-dataset-pipeline.md)
- [ADR-043: LoRA Serving Verification and Qwen3.5-0.8B Support](../../docs/adr/ADR-043-lora-serving-verification.md)
- [ADR-045: QuaRot + LoRA Composition at Inference Time](../../docs/adr/ADR-045-quarot-lora-composition.md)
- [ADR-047: Paged KV Cache with Prefix Reuse](../../docs/adr/ADR-047-paged-kv-cache.md)
- [ADR-054: Rotation-Aware LoRA Training (RoLoRA Integration)](../../docs/adr/ADR-054-rolora-rotation-aware-lora.md)
- [ADR-056: LoRA Tuning Pipeline](../../docs/adr/ADR-056-lora-tuning-pipeline.md)
- [ADR-057: LoRA Full-Lifecycle Consumer API](../../docs/adr/ADR-057-lora-consumer-api.md)
- [ADR-060: Pruning Toolbox — Calibration Infrastructure, Structured Pruning, and Analysis Scorers](../../docs/adr/ADR-060-pruning-toolbox.md)
- [ADR-061: Inference Metrics & Experiment Runner Infrastructure](../../docs/adr/ADR-061-inference-metrics-infrastructure.md)
- [ADR-064: CI Gate Taxonomy and Promotion Policy](../../docs/adr/ADR-064-ci-gate-taxonomy.md)
- [ADR-077: Pruning + Distillation Priority (Metal + CPU)](../../docs/adr/ADR-077-pruning-distill-priority.md)
- [ADR-079: Adaptive Micro-LoRA Brain — the Composed Train → Govern → Compose → Route → Consume Loop](../../docs/adr/ADR-079-adaptive-micro-lora-brain.md)
- [ADR-083: Backward SIMD Dispatch Contract](../../docs/adr/ADR-083-backward-simd-dispatch.md)
- [ADR-087: bench-compare gate calibration, coverage, and admissible structural proof](../../docs/adr/ADR-087-bench-compare-gate-calibration-and-coverage.md)
- [ADR-091: Learned mixture weights over a resident LoRA pool](../../docs/adr/ADR-091-learned-mixture-weights.md)
- [ADR-094: Where the routing code lives, and what feeds it](../../docs/adr/ADR-094-where-the-routing-code-lives.md)
- [ADR-095: The learning surface — feedback endpoint, router state, and what the API reports](../../docs/adr/ADR-095-the-learning-surface.md)
- [ADR-103: Splitting lattice-inference into focused crates behind a facade](../../docs/adr/ADR-103-inference-crate-decomposition.md)
- [ADR-ALIGNMENT-AUDIT: ADR Set Classification as of 2026-06-30](../../docs/adr/ADR-ALIGNMENT-AUDIT.md)
- [Architecture Decision Records](../../docs/adr/INDEX.md)

</details>

<!-- generated:adrs:end -->

<!-- generated:benchmarks:begin -->

### Benchmarks

No CI benchmark data for this crate: `bench-update.yml` records only `lattice-embed` `simd`, `lattice-inference` `elementwise_cpu_bench`.

<!-- generated:benchmarks:end -->
