# lattice-fann

Fast neural network primitives for sub-5ms CPU inference.

## Features

- **Zero-Alloc Inference**: Pre-allocated buffers eliminate runtime allocations during
  forward pass
- **Fluent Builder API**: Chain `.input()`, `.hidden()`, `.output()` for intuitive
  network construction
- **Numeric Stability**: Built-in NaN/Inf guards with configurable handling strategies
- **Optional GPU**: wgpu-based acceleration (Metal/Vulkan/DX12)

## Quick Start

```rust
use lattice_fann::{Network, NetworkBuilder, Activation};

// Build a simple classifier: 4 inputs -> 8 hidden -> 3 outputs
let mut network = NetworkBuilder::new()
    .input(4)
    .hidden(8, Activation::ReLU)
    .output(3, Activation::Softmax)
    .build()?;

// Run inference
let input = [1.0, 2.0, 3.0, 4.0];
let output = network.forward(&input)?;

// Output is a probability distribution (sums to 1.0)
assert_eq!(output.len(), 3);
```

## Network Construction

### Builder API

```rust
use lattice_fann::{NetworkBuilder, Activation};

let network = NetworkBuilder::new()
    .input(784)                          // MNIST input size
    .hidden(128, Activation::ReLU)       // Hidden layer 1
    .hidden(64, Activation::ReLU)        // Hidden layer 2
    .output(10, Activation::Softmax)     // 10 classes
    .build()?;

println!("Architecture: {}", network.architecture());
// Output: "784 -> ReLU(128) -> ReLU(64) -> Softmax(10)"

println!("Total parameters: {}", network.total_params());
```

### Reproducible Initialization

Use seeded initialization for deterministic networks.

```rust
use lattice_fann::{NetworkBuilder, Activation};

let network1 = NetworkBuilder::new()
    .input(4)
    .hidden(8, Activation::ReLU)
    .output(2, Activation::Softmax)
    .build_with_seed(42)?;

let network2 = NetworkBuilder::new()
    .input(4)
    .hidden(8, Activation::ReLU)
    .output(2, Activation::Softmax)
    .build_with_seed(42)?;

// Same seed produces identical weights
assert_eq!(
    network1.layer(0).unwrap().weights(),
    network2.layer(0).unwrap().weights()
);
```

## Activations

| Activation     | Formula            | Use Case                |
| -------------- | ------------------ | ----------------------- |
| `Linear`       | y = x              | Output regression       |
| `Sigmoid`      | y = 1/(1+e^-x)     | Binary classification   |
| `Tanh`         | y = tanh(x)        | Bounded output          |
| `ReLU`         | y = max(0, x)      | Hidden layers (default) |
| `LeakyReLU(a)` | y = max(ax, x)     | Prevent dead neurons    |
| `Softmax`      | y = e^x / sum(e^x) | Multi-class output      |

## Training

### TrainingConfig

```rust
use lattice_fann::{TrainingConfig, GradientGuardStrategy};

let config = TrainingConfig::new()
    .learning_rate(0.01)
    .momentum(0.9)
    .weight_decay(0.0001)
    .max_epochs(1000)
    .target_error(0.001)
    .batch_size(32)
    .shuffle(true)
    .gradient_guard(GradientGuardStrategy::Sanitize);
```

### BackpropTrainer

```rust
use lattice_fann::{NetworkBuilder, Activation, BackpropTrainer, TrainingConfig, Trainer};

let mut network = NetworkBuilder::new()
    .input(2)
    .hidden(4, Activation::Tanh)
    .output(1, Activation::Tanh)
    .build()?;

// XOR training data
let inputs = vec![
    vec![0.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 0.0],
    vec![1.0, 1.0],
];
let targets = vec![
    vec![0.0],
    vec![1.0],
    vec![1.0],
    vec![0.0],
];

let mut trainer = BackpropTrainer::new();
let config = TrainingConfig::new()
    .learning_rate(0.5)
    .max_epochs(1000);

let result = trainer.train(&mut network, &inputs, &targets, &config)?;
println!("Final error: {:.6}", result.final_error);
```

### Gradient Guards

| Strategy    | Behavior                 |
| ----------- | ------------------------ |
| `Error`     | Return error on NaN/Inf  |
| `Sanitize`  | Replace with safe values |
| `SkipBatch` | Skip problematic batches |

## Inference

### Forward Pass

```rust
use lattice_fann::{NetworkBuilder, Activation};

let mut network = NetworkBuilder::new()
    .input(4)
    .hidden(8, Activation::ReLU)
    .output(2, Activation::Softmax)
    .build()?;

let input = [1.0, 2.0, 3.0, 4.0];
let output = network.forward(&input)?;

// Access intermediate activations (for debugging)
let hidden_activations = network.activations(0).unwrap();
println!("Hidden layer output: {:?}", hidden_activations);
```

### Async API

Unified async interface for CPU/GPU switching.

```rust
use lattice_fann::{NetworkBuilder, Activation};

let mut network = NetworkBuilder::new()
    .input(4)
    .output(2, Activation::Softmax)
    .build()?;

let input = vec![1.0, 2.0, 3.0, 4.0];
let output = network.forward_async(&input).await?;
```

### Parallel Batch Inference

Requires the `parallel` feature to compile.

```rust
// This example requires: lattice-fann = { version = "0.1", features = ["parallel"] }
use lattice_fann::{NetworkBuilder, Activation};

let network = NetworkBuilder::new()
    .input(4)
    .hidden(8, Activation::ReLU)
    .output(2, Activation::Softmax)
    .build()?;

let inputs = vec![
    vec![1.0, 2.0, 3.0, 4.0],
    vec![5.0, 6.0, 7.0, 8.0],
    vec![9.0, 10.0, 11.0, 12.0],
];

let outputs = network.forward_batch(&inputs)?;
```

## Serialization

Compact binary format with magic number "FANN".

```rust
use lattice_fann::{Network, NetworkBuilder, Activation};

let network = NetworkBuilder::new()
    .input(4)
    .hidden(8, Activation::ReLU)
    .output(2, Activation::Softmax)
    .build()?;

// Save to bytes
let bytes = network.to_bytes();

// Load from bytes
let restored = Network::from_bytes(&bytes)?;

assert_eq!(network.num_inputs(), restored.num_inputs());
assert_eq!(network.num_outputs(), restored.num_outputs());
assert_eq!(network.total_params(), restored.total_params());
```

Format specification:

- Magic: 4 bytes "FANN"
- Version: u32 little-endian (1)
- Num layers: u32 little-endian
- Per layer: dimensions, activation type, weights, biases

## Feature Flags

| Feature     | Default | Description                        |
| ----------- | ------- | ---------------------------------- |
| `std`       | Yes     | Standard library support           |
| `simd`      | Yes     | SIMD optimizations for matrix ops  |
| `parallel`  | No      | Parallel batch inference via rayon |
| `serde`     | No      | Serialization support              |
| `gpu`       | No      | GPU acceleration via wgpu          |
| `gpu-tests` | No      | GPU tests requiring hardware       |

```toml
[dependencies]
lattice-fann = { version = "0.1" }                         # Default (std + simd)
lattice-fann = { version = "0.1", features = ["parallel"] } # Parallel inference
lattice-fann = { version = "0.1", features = ["gpu"] }      # GPU acceleration
```

## Performance

Average inference time <5ms for networks up to 128->256->128->64->10.

| Network Size     | Parameters | Inference Time |
| ---------------- | ---------- | -------------- |
| 4->8->2          | 50         | ~10us          |
| 128->256->64->10 | 50K        | ~500us         |
| 784->128->64->10 | 109K       | ~1ms           |

## GPU Acceleration

Enable the `gpu` feature for wgpu-based GPU acceleration.

```rust
#[cfg(feature = "gpu")]
use lattice_fann::gpu::{GpuContext, GpuNetwork};

#[cfg(feature = "gpu")]
async fn gpu_inference() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = GpuContext::new().await?;
    let network = GpuNetwork::from_cpu(&ctx, cpu_network)?;

    let output = network.forward(&input).await?;
    Ok(())
}
```

Supported backends:

- Metal (macOS)
- Vulkan (Linux, Windows)
- DX12 (Windows)

## API Reference

### Network

```rust
impl Network {
    pub fn forward(&mut self, input: &[f32]) -> FannResult<&[f32]>;
    pub async fn forward_async(&mut self, input: &[f32]) -> FannResult<Vec<f32>>;
    #[cfg(feature = "parallel")]
    pub fn forward_batch(&self, inputs: &[Vec<f32>]) -> FannResult<Vec<Vec<f32>>>;
    pub fn num_inputs(&self) -> usize;
    pub fn num_outputs(&self) -> usize;
    pub fn num_layers(&self) -> usize;
    pub fn total_params(&self) -> usize;
    pub fn layer(&self, index: usize) -> Option<&Layer>;
    pub fn to_bytes(&self) -> Vec<u8>;
    pub fn from_bytes(bytes: &[u8]) -> FannResult<Self>;
}
```

### Trainer Trait

```rust
pub trait Trainer {
    fn train(
        &mut self,
        network: &mut Network,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        config: &TrainingConfig,
    ) -> FannResult<TrainingResult>;
}
```
