# lattice-tune

Knowledge distillation and training infrastructure for lattice neural models.

## Features

- **Knowledge Distillation**: Teacher (LLM) -> soft labels -> student (tiny NN) pipeline
- **Multi-Provider Teachers**: Claude, OpenAI, Gemini, local models (Ollama)
- **Model Registry**: Semver versioning with lineage tracking
- **Endpoint Security**: TLS enforcement, domain whitelisting, checksum verification

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
    .seed(42)
    .validation_split(0.2);

dataset.set_config(config)?;

let stats = dataset.stats();
println!("Examples: {}", stats.num_examples);
println!("Embedding dim: {}", stats.embedding_dim);
```

### Validation Split

```rust
use lattice_tune::data::{Dataset, DatasetConfig};

let config = DatasetConfig::with_batch_size(32)
    .validation_split(0.2);  // 20% for validation

// Dataset automatically handles train/validation split
```

## Distillation Pipeline

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

// Label with teacher
let result = pipeline.label_single(&raw)?;
println!("Labeled with confidence: {}", result.confidence);

// Check statistics
let stats = pipeline.stats();
println!("Successful: {}", stats.successful);
println!("Failed: {}", stats.failed);
```

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

```rust
use lattice_tune::train::{TrainingConfig, TrainingLoop};
use lattice_tune::data::Dataset;

let config = TrainingConfig::quick();  // Fast training preset
let mut trainer = TrainingLoop::new(config)?;

let metrics = trainer.train(&mut dataset)?;
println!("Final loss: {:.4}", metrics.final_train_loss);
println!("Epochs completed: {}", metrics.epochs_completed);
```

### Checkpointing

```rust
use lattice_tune::train::Checkpoint;

// Save checkpoint
let checkpoint = Checkpoint::new(&network, &training_state);
checkpoint.save("checkpoints/epoch_50.ckpt")?;

// Load checkpoint
let loaded = Checkpoint::load("checkpoints/epoch_50.ckpt")?;
let network = loaded.into_network();
```

## Model Registry

### RegisteredModel

```rust
use lattice_tune::registry::{ModelRegistry, RegisteredModel, ModelMetadata};

// Create in-memory registry
let mut registry = ModelRegistry::in_memory();

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
let loaded = registry.get("intent_classifier", "1.0.0")?;
println!("Loaded: {}", loaded.full_name());
```

### Model Status Lifecycle

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
    .with_parent(base_id);  // Track lineage
```

## Feature Flags

| Feature       | Default | Description                                |
| ------------- | ------- | ------------------------------------------ |
| `std`         | Yes     | Standard library support                   |
| `serde`       | No      | Serialization (propagates to lattice-fann) |
| `gpu`         | No      | GPU-accelerated training                   |
| `gpu-tests`   | No      | GPU tests requiring hardware               |
| `safetensors` | No      | Safe checkpoint serialization              |

```toml
[dependencies]
lattice-tune = { version = "0.1" }                           # Default
lattice-tune = { version = "0.1", features = ["serde"] }     # With serialization
lattice-tune = { version = "0.1", features = ["gpu"] }       # GPU training
lattice-tune = { version = "0.1", features = ["safetensors"] } # Safe checkpoints
```

## Dependencies

This crate depends on `lattice-fann` for neural network primitives.

```toml
[dependencies]
lattice-tune = { version = "0.1" }
# lattice-fann is automatically included as a transitive dependency
```

## API Reference

### DistillationPipeline

```rust
impl DistillationPipeline {
    pub fn with_teacher(config: TeacherConfig) -> Result<Self>;
    pub fn label_single(&mut self, example: &RawExample) -> Result<LabelingResult>;
    pub fn label_batch(&mut self, examples: &[RawExample]) -> Result<Vec<LabelingResult>>;
    pub fn stats(&self) -> DistillationStats;
}
```

### TrainingLoop

```rust
impl TrainingLoop {
    pub fn new(config: TrainingConfig) -> Result<Self>;
    pub fn train(&mut self, dataset: &mut Dataset) -> Result<TrainingMetrics>;
    pub fn train_epoch(&mut self, dataset: &mut Dataset) -> Result<EpochMetrics>;
}
```

### ModelRegistry

```rust
impl ModelRegistry {
    pub fn in_memory() -> Self;
    pub fn register(&mut self, model: RegisteredModel, weights: &[u8]) -> Result<Uuid>;
    pub fn get(&self, name: &str, version: &str) -> Result<RegisteredModel>;
    pub fn promote(&mut self, id: Uuid, status: ModelStatus) -> Result<()>;
    pub fn find_latest(&self, name: &str) -> Option<&RegisteredModel>;
}
```
