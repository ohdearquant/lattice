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

| Feature          | Default | Description                                                             |
| ---------------- | ------- | ----------------------------------------------------------------------- |
| `std`            | Yes     | Standard library support                                                |
| `serde`          | No      | Serialization (propagates to lattice-fann)                              |
| `gpu`            | No      | GPU-accelerated forward/backward passes and validation[^gpu-limitation] |
| `gpu-tests`      | No      | GPU tests requiring hardware                                            |
| `safetensors`    | No      | PEFT-compatible LoRA adapter serialization                              |
| `inference-hook` | No      | `impl LoraHook for LoraAdapter` (pulls in `lattice-inference`)          |
| `train-backward` | No      | Backward/gradient training surface (pulls in `lattice-inference`)       |

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
lattice-tune = { version = "0.4.2" }                           # Default
lattice-tune = { version = "0.4.2", features = ["serde"] }     # With serialization
lattice-tune = { version = "0.4.2", features = ["gpu"] }       # GPU training
lattice-tune = { version = "0.4.2", features = ["safetensors"] } # LoRA adapter I/O
```

### Injecting an adapter into `lattice-inference`

Enable `inference-hook` to get `impl lattice_inference::lora_hook::LoraHook for
LoraAdapter`, so a trained adapter can be handed to a running inference engine
as a `Box<dyn LoraHook>`:

```toml
[dependencies]
lattice-tune = { version = "0.4.2", features = ["inference-hook", "safetensors"] }
lattice-inference = "0.4.2" # provides the LoraHook trait
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
lattice-tune = { version = "0.4.2" }
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
