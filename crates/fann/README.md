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

`Network::forward_async` is an async-signature wrapper around the synchronous CPU `forward`
(`self.forward(input).map(<[f32]>::to_vec)`, `crates/fann/src/network/mod.rs`) — it does not
itself dispatch to GPU or perform any async work; it exists so callers on an async runtime can
call it without wrapping `forward` themselves. See "GPU Acceleration" below for the actual
GPU-backed path.

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
// This example requires: lattice-fann = { version = "0.12", features = ["parallel"] }
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
lattice-fann = { version = "0.12" }                         # Default (std + simd)
lattice-fann = { version = "0.12", features = ["parallel"] } # Parallel inference
lattice-fann = { version = "0.12", features = ["gpu"] }      # GPU acceleration
```

## Performance

Average inference time <5ms for networks up to 128->256->128->64->10.

| Network Size     | Parameters | Inference Time |
| ---------------- | ---------- | -------------- |
| 4->8->2          | 50         | ~10us          |
| 128->256->64->10 | 50K        | ~500us         |
| 784->128->64->10 | 109K       | ~1ms           |

No committed benchmark target currently produces these numbers
(`crates/fann/examples/router_online.rs` measures online-learning convergence, not raw forward-pass
latency) — treat this table as illustrative rather than reproducible until a forward-pass bench lands.

## GPU Acceleration

Enable the `gpu` feature for wgpu-based GPU acceleration.

```rust
#[cfg(feature = "gpu")]
use lattice_fann::{
    Activation, NetworkBuilder,
    gpu::{GpuContext, GpuNetwork},
};
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
async fn gpu_inference() -> Result<(), Box<dyn std::error::Error>> {
    let cpu_network = NetworkBuilder::new()
        .input(4)
        .hidden(8, Activation::ReLU)
        .output(2, Activation::Softmax)
        .build()?;
    let ctx = Arc::new(GpuContext::new().await?);
    let mut network = GpuNetwork::new(ctx, cpu_network)?;
    let input = [0.1, 0.2, 0.3, 0.4];

    let output = network.forward(&input).await?;
    println!("{output:?}");
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

## Crate reference

The sections below are generated by `scripts/crate-readme.py` from the source tree and are refreshed after pushes to `main`. Text between the `generated` markers is overwritten; edit the source or the script instead.

<!-- generated:facts:begin -->

### Facts

- Version: `0.15.0`
- Minimum supported Rust version: `1.93`
- Edition: `2024`
- License: `MIT OR Apache-2.0`
- crates.io: <https://crates.io/crates/lattice-fann>
- API documentation: <https://docs.rs/lattice-fann>

<!-- generated:facts:end -->

<!-- generated:layout:begin -->

### Source layout

The tracked files under `src/`; each Rust file shows the first sentence of its `//!` module documentation.

<details>
<summary><code>src/</code>: 35 files</summary>

- `activation.rs`: Activation functions for dense networks.
- `activation/`
  - `tests.rs`
- `error.rs`: Error types and allocation-shape validation for `lattice-fann`.
- `error/`
  - `tests.rs`
- `gpu/`
  - `buffer.rs`: Three-tier GPU buffer reuse pool with lifecycle-based eviction and pressure control.
  - `circuit_breaker.rs`: Memory-pressure levels and a fail-closed circuit breaker for GPU allocations.
  - `context.rs`: GPU device context: adapter selection, queue ownership, pooled memory, and pipelines.
  - `error.rs`: Error taxonomy and recovery classification for GPU operations.
  - `mod.rs`: GPU compute backend: contexts, pooled memory, WGSL pipelines, and dense inference.
  - `network.rs`: GPU dense-network inference with size-based CPU fallback.
  - `shader_manager.rs`: WGSL compute-pipeline selection, warmup, and cache accounting.
  - `shaders.rs`: WGSL compute shaders for neural network operations
- `layer.rs`: Dense feedforward-layer storage and inference.
- `layer/`
  - `tests.rs`
- `lib.rs`: `lattice-fann` provides small dense neural networks for CPU-first inference.
- `lora.rs`: LoRA adapter descriptor and blend-validation primitives, re-exported from `lattice_lora` at the paths this crate published before they moved.
- `network/`
  - `builder.rs`: Fluent construction of dense feedforward networks.
  - `builder/`
    - `tests.rs`
  - `mod.rs`: Feedforward-network execution and construction.
  - `serde_validation_tests.rs`
  - `serialization.rs`: Compact binary network serialization.
  - `serialization/`
    - `tests.rs`
  - `tests.rs`
- `tests.rs`
- `training/`
  - `backprop.rs`: Backpropagation trainer implementation
  - `backprop/`
    - `tests.rs`
  - `ewc.rs`: Elastic Weight Consolidation (EWC++) diagonal-Fisher forgetting guard.
  - `ewc/`
    - `tests.rs`
  - `gradient.rs`: Gradient utilities for training
  - `mod.rs`: Training algorithms and their shared configuration.
  - `rloo.rs`: REINFORCE with Leave-One-Out baseline (RLOO) policy-gradient trainer.
  - `rloo/`
    - `tests.rs`
    - `tests/`
      - `batch_refit.rs`: Batch refit over recorded explored actions (ADR-096 decision 3).
  - `router_update.rs`: Online refit of an adapter-selection gate from preference feedback.
  - `router_update/`
    - `tests.rs`

</details>

Targets declared by the manifest, including the ones Cargo discovers by directory:

- lib (1): `lattice_fann`
- example (2): `fann_xor`, `router_online` (requires `online-router`)
- test (3): `cpu_gpu_parity`, `proptest_network`, `shader_validation`

<!-- generated:layout:end -->

<!-- generated:stats:begin -->

### Statistics

Rust lines by kind (code, comment, blank; doc comments count as comment), test functions (`#[test]` and `#[<runtime>::test]` attributes), items declared `pub` (not `pub(crate)`; reachability through private modules is not resolved) and occurrences of the `unsafe` keyword outside comments and strings.

| Area        | Files | Code  | Comment | Blank | Tests | `pub` items | `unsafe` |
| ----------- | ----- | ----- | ------- | ----- | ----- | ----------- | -------- |
| `src/`      | 35    | 8706  | 2556    | 1197  | 199   | 248         | 15       |
| `tests/`    | 3     | 826   | 102     | 219   | 31    | 0           | 0        |
| `examples/` | 2     | 556   | 168     | 75    | 0     | 0           | 0        |
| total       | 40    | 10088 | 2826    | 1491  | 230   | 248         | 15       |

Other tracked files (the README excluded):

| Extension | Files | Size      |
| --------- | ----- | --------- |
| `.md`     | 7     | 108.8 KiB |
| `.toml`   | 1     | 1.8 KiB   |

<!-- generated:stats:end -->

<!-- generated:docs:begin -->

### In-crate documentation

- [Changelog](CHANGELOG.md) (`CHANGELOG.md`)
- [lattice-fann documentation](docs/INDEX.md) (`docs/INDEX.md`)
- [lattice-fann design](docs/design.md) (`docs/design.md`)
- [GPU backend](docs/gpu.md) (`docs/gpu.md`)
- [LoRA Router Refit](docs/lora-router.md) (`docs/lora-router.md`)
- [Network core reference](docs/network.md) (`docs/network.md`)
- [Training](docs/training.md) (`docs/training.md`)

<!-- generated:docs:end -->

<!-- generated:features:begin -->

### Features

Default set: `simd`, `std`. Each description is the first sentence of the comment above the feature in `Cargo.toml`.

- `gpu`: GPU acceleration via wgpu (Metal on macOS, Vulkan on Linux, DX12 on Windows) Enables `dep:wgpu`, `dep:pollster`, `dep:bytemuck`, `dep:futures`, `std`.
- `gpu-tests`: Enable GPU tests that require actual GPU hardware (disabled in CI by default) Enables `gpu`.
- `online-router`: Policy-gradient router trainer and EWC forgetting guard — off by default. Enables nothing.
- `parallel`: Enables `dep:rayon`.
- `serde`: Enables `dep:serde`, `dep:serde_json`.
- `simd` (default): Enables nothing.
- `std` (default): Enables nothing.

<!-- generated:features:end -->

<!-- generated:dependencies:begin -->

### Workspace dependencies

- Depends on: `lattice-lora`
- Used by: `lattice-inference` (optional), `lattice-router` (optional), `lattice-serve` (optional), `lattice-tune`

<!-- generated:dependencies:end -->

<!-- generated:adrs:begin -->

### Architecture decisions

ADRs under `docs/adr/` that name `lattice-fann` or `crates/fann/`:

<details>
<summary>25 ADRs</summary>

- [ADR-020: Neural Network Primitives](../../docs/adr/ADR-020-neural-net-primitives.md)
- [ADR-021: Zero-Allocation Inference](../../docs/adr/ADR-021-zero-alloc.md)
- [ADR-022: Gradient Guard Strategy](../../docs/adr/ADR-022-gradient-guards.md)
- [ADR-023: Activation Functions](../../docs/adr/ADR-023-activation-functions.md)
- [ADR-024: Network Builder Pattern](../../docs/adr/ADR-024-network-builder.md)
- [ADR-025: GPU Backend](../../docs/adr/ADR-025-gpu-backend.md)
- [ADR-026: Training Loop](../../docs/adr/ADR-026-training-loop.md)
- [ADR-027: Fine-Tuning Pipeline](../../docs/adr/ADR-027-finetuning-pipeline.md)
- [ADR-030: Knowledge Distillation Pipeline](../../docs/adr/ADR-030-knowledge-distillation.md)
- [ADR-032: Training Callbacks](../../docs/adr/ADR-032-training-callbacks.md)
- [ADR-033: JIT Adaptation](../../docs/adr/ADR-033-jit-adaptation.md)
- [ADR-034: Dataset Pipeline](../../docs/adr/ADR-034-dataset-pipeline.md)
- [ADR-064: CI Gate Taxonomy and Promotion Policy](../../docs/adr/ADR-064-ci-gate-taxonomy.md)
- [ADR-079: Adaptive Micro-LoRA Brain — the Composed Train → Govern → Compose → Route → Consume Loop](../../docs/adr/ADR-079-adaptive-micro-lora-brain.md)
- [ADR-080: Consolidation of Duplicated Numeric-Contract Helpers (Softmax, HTTP Serving, Decode Policy, GEMM Validation)](../../docs/adr/ADR-080-consolidation-duplicated-contracts.md)
- [ADR-087 Amendment 1 — open pull request population snapshot](../../docs/adr/ADR-087-amendment-1-population-snapshot.md)
- [ADR-087: bench-compare gate calibration, coverage, and admissible structural proof](../../docs/adr/ADR-087-bench-compare-gate-calibration-and-coverage.md)
- [ADR-091: Learned mixture weights over a resident LoRA pool](../../docs/adr/ADR-091-learned-mixture-weights.md)
- [ADR-094: Where the routing code lives, and what feeds it](../../docs/adr/ADR-094-where-the-routing-code-lives.md)
- [ADR-095: The learning surface — feedback endpoint, router state, and what the API reports](../../docs/adr/ADR-095-the-learning-surface.md)
- [ADR-096: Producing a router artifact — schema lineage, bootstrap, the learned action, and admission](../../docs/adr/ADR-096-producing-a-router-artifact.md)
- [ADR-103: Splitting lattice-inference into focused crates behind a facade](../../docs/adr/ADR-103-inference-crate-decomposition.md)
- [ADR-105: lattice-lora, One Home for Adapter Format and Execution Contracts](../../docs/adr/ADR-105-lattice-lora.md)
- [ADR-ALIGNMENT-AUDIT: ADR Set Classification as of 2026-06-30](../../docs/adr/ADR-ALIGNMENT-AUDIT.md)
- [Architecture Decision Records](../../docs/adr/INDEX.md)

</details>

<!-- generated:adrs:end -->

<!-- generated:benchmarks:begin -->

### Benchmarks

No CI benchmark data for this crate: `bench-update.yml` records only `lattice-embed` `simd`, `lattice-inference` `elementwise_cpu_bench`.

<!-- generated:benchmarks:end -->
