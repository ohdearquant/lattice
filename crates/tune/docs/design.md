# lattice-tune design

`lattice-tune` is the training side of Lattice: a knowledge-distillation pipeline that
turns teacher-model (Claude, GPT, Gemini) responses into a compact student network, plus
the training loop, LoRA adapters, and a model registry that tracks what got trained and
why.

## Pipeline

```text
Raw Data → Teacher (LLM) → Soft Labels → Dataset → Training → Model → Registry
                                                        ↓
                                                   Deployment
```

- **`data`** — `TrainingExample`, `Dataset`, and `Batch`: the labeled-example format and
  batching used throughout the rest of the crate.
- **`distill`** — pipelines a teacher model (Claude/GPT/Gemini) over raw text into soft
  labels, producing `TrainingExample`s.
- **`train`** — the training loop, optimizer, LR schedule, early stopping, checkpointing,
  and (behind the `gpu` feature) GPU-accelerated forward/backward.
- **`registry`** — versions and stores trained models with training-config and metric
  provenance, plus shadow-deployment and rollback support.

## Design principles

1. **Data-first** — a well-defined training-example format with full traceability from
   raw input through to the trained model.
2. **Modular** — distillation, training, and registry are separate concerns that compose
   through the shared `data` types.
3. **Extensible** — teacher providers (Claude, GPT, Gemini) are pluggable via
   `TeacherConfig`/`TeacherProvider`.
4. **Traceable** — every registered model carries its version, training config, and
   metrics.

## Distillation walkthrough

```ignore
use lattice_tune::distill::{TeacherConfig, DistillationPipeline, RawExample};

// Configure teacher model
let teacher = TeacherConfig::claude_sonnet();

// Create distillation pipeline
let mut pipeline = DistillationPipeline::with_teacher(teacher)?;

// Create raw examples (text, not embeddings)
let raw = RawExample::new(
    vec!["Hello".to_string(), "How are you?".to_string()],
    "What's the weather like?",
);

// Label with teacher
let result = pipeline.label_single(&raw)?;
println!("Labeled with confidence: {}", result.confidence);
```

## Training walkthrough

```ignore
use lattice_tune::train::{TrainingConfig, TrainingLoop};
use lattice_tune::data::Dataset;

// Configure training
let config = TrainingConfig::default()
    .epochs(100)
    .batch_size(32)
    .learning_rate(0.001);

// Train
let mut trainer = TrainingLoop::new(config)?;
let metrics = trainer.train(&mut dataset)?;

println!("Final loss: {:.4}", metrics.final_train_loss);
```

### GPU training

With the `gpu` feature enabled, forward/backward passes and validation run on the GPU:

```ignore
use lattice_tune::train::{GpuTrainer, GpuTrainerBuilder, TrainingConfig};
use lattice_fann::Activation;

// Build GPU trainer
let mut trainer = GpuTrainerBuilder::new(768, 6)
    .hidden(64, Activation::ReLU)
    .hidden(32, Activation::ReLU)
    .config(TrainingConfig::default())
    .build()?;

// Train batches
for batch in dataset.batches() {
    let loss = trainer.train_batch(&batch)?;
}
```

`GpuTrainer::train_batch` currently returns `Err` for every optimizer choice — the
GPU weight-update gap documented on `lattice_tune::train`'s module docs
(<https://github.com/ohdearquant/lattice/issues/797>) — while `GpuTrainer::validate`
(forward-only) works today.

## Registry walkthrough

```rust
use lattice_tune::registry::{ModelRegistry, RegisteredModel, ModelMetadata};

// Create a registry
let registry = ModelRegistry::in_memory();

// Register a model
let metadata = ModelMetadata::classifier(768, 6, 10000);
let model = RegisteredModel::new("intent_classifier", "1.0.0")
    .with_metadata(metadata)
    .with_description("Intent classification model");

let weights = vec![0u8; 1000]; // Model weights
let id = registry.register(model, &weights).unwrap();

// Retrieve the model
let loaded = registry.get("intent_classifier", "1.0.0").unwrap();
println!("Loaded: {}", loaded.full_name());
```

## Consuming the `inference-hook` bridge

To inject a trained `LoraAdapter` into a running `lattice-inference` forward pass, enable
the `inference-hook` feature. It pulls in `lattice-inference` and implements
`lattice_inference::lora_hook::LoraHook` for `LoraAdapter`, so an adapter can be handed to
the engine as a `Box<dyn LoraHook>`:

```toml
[dependencies]
lattice-tune = { version = "0.4.2", features = ["inference-hook"] }
lattice-inference = "0.4.2" # provides the LoraHook trait
```

```ignore
use lattice_tune::lora::LoraAdapter;
use lattice_inference::lora_hook::LoraHook;

let adapter: LoraAdapter = /* load via LoraAdapter::load_peft_safetensors(...) */;
let hook: Box<dyn LoraHook> = Box::new(adapter);
// hand `hook` to the inference engine; on each projection it calls
// hook.apply(layer_idx, module, x, output)
```

The trait path is `lattice_inference::lora_hook::LoraHook` (it is not re-exported at the
inference crate root). Without the `inference-hook` feature, `LoraAdapter` is a pure
training type and carries no dependency on `lattice-inference`.

## Checkpoint serialization format

`Checkpoint` serializes to JSON when the `serde` feature is enabled:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "epoch": 10,
  "global_step": 5000,
  "metrics": {
    "final_train_loss": 0.05,
    "final_val_loss": 0.06,
    "epochs_completed": 10,
    "total_steps": 5000,
    "best_epoch": 8,
    "best_val_loss": 0.055
  },
  "created_at": "2024-01-15T10:30:00Z",
  "weights": "<base64-encoded bytes>",
  "optimizer_state": "<base64-encoded bytes>"
}
```

| Field             | Type            | Description                                      |
| ----------------- | ---------------- | ------------------------------------------------ |
| `id`              | UUID v4          | Unique identifier for this checkpoint            |
| `epoch`           | usize            | Training epoch when created (0-indexed)          |
| `global_step`     | usize            | Total batches processed                          |
| `metrics`         | TrainingMetrics  | Training statistics at checkpoint time           |
| `created_at`      | ISO 8601         | UTC timestamp of creation                        |
| `weights`         | bytes            | Serialized model parameters (byte layout below)  |
| `optimizer_state` | bytes            | Serialized optimizer momentum/state (below)      |

The byte-level layout of `weights` and `optimizer_state` is the load-bearing part of this
format and stays on `Checkpoint` itself in `src/train/loop/checkpoint.rs`, so a consumer
parsing the raw bytes has it without a docs/ round-trip:

- `weights`: little-endian `f32` values, concatenated layer-by-layer (each layer's weight
  matrix in row-major order, then its biases; input-to-output order across layers). For a
  network with layers `[4->8, 8->2]`: `[layer0_weights: 32 floats][layer0_biases: 8
  floats][layer1_weights: 16 floats][layer1_biases: 2 floats]`.
- `optimizer_state`: SGD-with-momentum stores one velocity vector per parameter; Adam
  stores first (`m`) and second (`v`) moment vectors per parameter.

### Loading a checkpoint

```ignore
use lattice_tune::Checkpoint;

// Load from JSON file
let json = std::fs::read_to_string("checkpoint_epoch_10.json")?;
let checkpoint: Checkpoint = serde_json::from_str(&json)?;

// Restore model weights
model.load_weights(&checkpoint.weights);
optimizer.load_state(&checkpoint.optimizer_state);
```

### Naming convention

Recommended file naming: `checkpoint_epoch_{epoch:04d}_step_{step:08d}.json` — for
example, `checkpoint_epoch_0010_step_00005000.json`.

## Feature flags

- `std` (default): standard library support.
- `serde`: serialization support for all types, including `Checkpoint`.
- `gpu`: GPU-accelerated training via `GpuTrainer`.
- `inference-hook`: implements `lattice_inference::lora_hook::LoraHook` for `LoraAdapter`,
  so a trained adapter can be handed straight to `lattice-inference` as a `Box<dyn LoraHook>`.
