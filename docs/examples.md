# Lattice Examples

This document provides runnable examples for the three main crates in the lattice workspace:
`lattice-embed`, `lattice-fann`, and `lattice-transport`.

All examples live in their respective crate's `examples/` directory and can be run with:

```
cargo run -p lattice-embed --example basic_embed
cargo run -p lattice-embed --example similarity
cargo run -p lattice-fann --example fann_xor
cargo run -p lattice-transport --example sinkhorn_ot
```

---

## lattice-embed

### Generating a single embedding

`NativeEmbeddingService` downloads model weights on first use and caches them under
`~/.lattice/cache/`. Subsequent calls within the same process are fast because the model
is loaded once per process lifetime via `OnceLock`.

```rust
use lattice_embed::{EmbeddingModel, EmbeddingService, NativeEmbeddingService};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let service = NativeEmbeddingService::default();

    let embedding = service
        .embed_one(
            "The quick brown fox jumps over the lazy dog",
            EmbeddingModel::default(), // BgeSmallEnV15, 384 dims
        )
        .await?;

    println!("Dimension: {}", embedding.len()); // 384
    Ok(())
}
```

The `EmbeddingModel::default()` is `BgeSmallEnV15` (384 dimensions). The available models are:

| Variant               | Dimensions | Notes                  |
| --------------------- | ---------- | ---------------------- |
| `BgeSmallEnV15`       | 384        | Default, fast          |
| `BgeBaseEnV15`        | 768        | Balanced               |
| `BgeLargeEnV15`       | 1024       | Highest quality        |
| `MultilingualE5Small` | 384        | Multilingual           |
| `MultilingualE5Base`  | 768        | Multilingual           |
| `AllMiniLmL6V2`       | 384        | Sentence-transformers  |
| `Qwen3Embedding0_6B`  | 1024       | Decoder-only, GPU      |
| `Qwen3Embedding4B`    | 2560       | Decoder-only, GPU, MRL |

### Batch embeddings

Pass a `Vec<String>` to `embed()`. The service enforces a maximum batch size of 1000 and a
maximum text length of 32 768 characters; both limits exist to prevent OOM.

```rust
use lattice_embed::{EmbeddingModel, EmbeddingService, NativeEmbeddingService};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let service = NativeEmbeddingService::default();

    let texts = vec![
        "Rust is a systems programming language".to_string(),
        "Python is great for data science".to_string(),
        "WebAssembly runs near-natively in browsers".to_string(),
    ];

    let embeddings = service
        .embed(&texts, EmbeddingModel::BgeSmallEnV15)
        .await?;

    assert_eq!(embeddings.len(), 3);
    for (text, emb) in texts.iter().zip(&embeddings) {
        println!("{}: {} dims", text, emb.len());
    }
    Ok(())
}
```

### Choosing a model and checking its properties

`EmbeddingModel` has `const` methods you can call without the service:

```rust
use lattice_embed::EmbeddingModel;

let model: EmbeddingModel = "bge-base".parse().unwrap(); // flexible FromStr
assert_eq!(model, EmbeddingModel::BgeBaseEnV15);
assert_eq!(model.dimensions(), 768);
assert_eq!(model.max_input_tokens(), 512);
assert!(model.is_local());
assert_eq!(model.model_id(), "BAAI/bge-base-en-v1.5");

// E5 models need an explicit prefix for asymmetric retrieval
let e5 = EmbeddingModel::MultilingualE5Small;
assert_eq!(e5.query_instruction(), Some("query: "));
```

For E5 models the query text should be wrapped with the instruction prefix before embedding,
while documents can be embedded as-is:

```rust
use lattice_embed::{EmbeddingModel, EmbeddingService, NativeEmbeddingService};

async fn embed_for_retrieval(
    service: &NativeEmbeddingService,
    query: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let model = EmbeddingModel::MultilingualE5Small;
    let prefixed = match model.query_instruction() {
        Some(prefix) => format!("{prefix}{query}"),
        None => query.to_string(),
    };
    Ok(service.embed_one(&prefixed, model).await?)
}
```

### Cosine similarity and nearest-neighbor search

`lattice_embed::utils` exposes SIMD-accelerated vector operations. On x86_64 the dispatch
order is AVX-512F > AVX2+FMA > scalar; on aarch64 it uses NEON.

```rust
use lattice_embed::utils::{cosine_similarity, normalize, batch_cosine_similarity};

fn nearest_neighbor<'a>(query: &[f32], documents: &'a [Vec<f32>]) -> (usize, f32) {
    documents
        .iter()
        .enumerate()
        .map(|(i, doc)| (i, cosine_similarity(query, doc)))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
}

// Batch variant — more efficient than a loop
fn all_similarities(query: &[f32], docs: &[Vec<f32>]) -> Vec<f32> {
    let pairs: Vec<(&[f32], &[f32])> = docs.iter().map(|d| (query, d.as_slice())).collect();
    batch_cosine_similarity(&pairs)
}
```

For normalized vectors, `dot_product` is equivalent to `cosine_similarity` but skips the
norm computation, so it is faster:

```rust
use lattice_embed::utils::{dot_product, normalize};

let mut a = vec![1.0_f32, 2.0, 3.0];
let mut b = vec![4.0_f32, 5.0, 6.0];
normalize(&mut a);
normalize(&mut b);

// After normalization, dot == cosine
let dot = dot_product(&a, &b);
let cosine = lattice_embed::utils::cosine_similarity(&a, &b);
assert!((dot - cosine).abs() < 1e-5);
```

The full set of distance utilities:

```rust
use lattice_embed::utils::{
    cosine_similarity, dot_product, euclidean_distance,
    normalize, batch_cosine_similarity, batch_dot_product,
};

let a = vec![3.0_f32, 4.0];
let b = vec![0.0_f32, 0.0];

println!("euclidean: {}", euclidean_distance(&a, &b)); // 5.0
```

### SIMD configuration inspection

```rust
use lattice_embed::{SimdConfig, simd_config};

let cfg: SimdConfig = simd_config(); // detected once per process, cached
println!("AVX-512: {}", cfg.avx512f_enabled);
println!("AVX2:    {}", cfg.avx2_enabled);
println!("NEON:    {}", cfg.neon_enabled);
println!("Any SIMD available: {}", cfg.simd_available());
```

### EmbeddingKey and VectorDType

`EmbeddingKey` is used to uniquely identify an embedding space (model + revision +
dims + metric + dtype + norm). It generates a stable `canonical_bytes()` for deduplication
and cache keying.

```rust
use lattice_embed::types::{DistanceMetric, EmbeddingKey, VectorDType, VectorNorm};

let key = EmbeddingKey::new(
    "bge-small-en-v1.5",
    "v1.5",
    384,
    DistanceMetric::Cosine,
    VectorDType::F32,
    VectorNorm::Unit,
);

// Stable bytes for hashing — same inputs always produce the same bytes
let bytes = key.canonical_bytes();
println!("Key bytes len: {}", bytes.len());
println!("dtype size: {} bytes/element", VectorDType::F32.size_bytes()); // 4
```

---

## lattice-fann

`lattice-fann` provides a fast feedforward neural network for tiny models that need
sub-millisecond CPU inference. Pre-allocated buffers mean no heap allocation during the
forward pass.

### Building and running a classifier

```rust
use lattice_fann::{Activation, Network, NetworkBuilder};

// Build: 4 inputs -> 8 hidden (ReLU) -> 3 outputs (Softmax)
let mut network = NetworkBuilder::new()
    .input(4)
    .hidden(8, Activation::ReLU)
    .output(3, Activation::Softmax)
    .build()
    .unwrap();

println!("Inputs:  {}", network.num_inputs());   // 4
println!("Outputs: {}", network.num_outputs());  // 3
println!("Params:  {}", network.total_params()); // 4*8+8 + 8*3+3 = 67
println!("Arch:    {}", network.architecture()); // "4 -> ReLU(8) -> Softmax(3)"

let input = [1.0_f32, 2.0, 3.0, 4.0];
let output = network.forward(&input).unwrap();

// Softmax output sums to 1.0
let sum: f32 = output.iter().sum();
assert!((sum - 1.0).abs() < 1e-5);
```

The `dense()` shorthand adds a ReLU hidden layer:

```rust
use lattice_fann::{Activation, NetworkBuilder};

let network = NetworkBuilder::new()
    .input(784)
    .dense(128)   // hidden + ReLU
    .dense(64)    // hidden + ReLU
    .output(10, Activation::Softmax)
    .build()
    .unwrap();

assert_eq!(network.num_inputs(), 784);
assert_eq!(network.num_outputs(), 10);
```

### Reproducible initialization with a seed

```rust
use lattice_fann::{Activation, NetworkBuilder};

let net1 = NetworkBuilder::new()
    .input(4)
    .hidden(8, Activation::Tanh)
    .output(2, Activation::Sigmoid)
    .build_with_seed(42)
    .unwrap();

let net2 = NetworkBuilder::new()
    .input(4)
    .hidden(8, Activation::Tanh)
    .output(2, Activation::Sigmoid)
    .build_with_seed(42)
    .unwrap();

// Same seed => identical weights
assert_eq!(
    net1.layer(0).unwrap().weights(),
    net2.layer(0).unwrap().weights()
);
```

### Training a small XOR network

`BackpropTrainer` uses SGD with momentum. `TrainingConfig` has builder methods for all
hyperparameters.

```rust
use lattice_fann::{
    Activation, BackpropTrainer, NetworkBuilder, Trainer, TrainingConfig,
};

let mut network = NetworkBuilder::new()
    .input(2)
    .hidden(4, Activation::Tanh)
    .output(1, Activation::Tanh)
    .build_with_seed(1)
    .unwrap();

// XOR truth table
let inputs: Vec<Vec<f32>> = vec![
    vec![0.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 0.0],
    vec![1.0, 1.0],
];
let targets: Vec<Vec<f32>> = vec![
    vec![-1.0], // Tanh outputs in [-1, 1]; use -1/+1 for XOR
    vec![1.0],
    vec![1.0],
    vec![-1.0],
];

let config = TrainingConfig::new()
    .learning_rate(0.5)
    .momentum(0.9)
    .max_epochs(2000)
    .target_error(0.01)
    .batch_size(4)
    .seed(42);

let mut trainer = BackpropTrainer::new();
let result = trainer.train(&mut network, &inputs, &targets, &config).unwrap();

println!("Converged: {}", result.converged);
println!("Final error: {:.4}", result.final_error);
println!("Epochs: {}", result.epochs_trained);
```

### Accessing intermediate activations

After `forward()`, each layer's output buffer is accessible via `activations()`:

```rust
use lattice_fann::{Activation, NetworkBuilder};

let mut network = NetworkBuilder::new()
    .input(2)
    .hidden(4, Activation::ReLU)
    .output(2, Activation::Linear)
    .build()
    .unwrap();

network.forward(&[1.0, 2.0]).unwrap();

let hidden = network.activations(0).unwrap(); // layer 0 output
println!("Hidden activations: {:?}", hidden); // len 4
```

### Available activations

```rust
use lattice_fann::Activation;

// All available variants
let _linear  = Activation::Linear;
let _sigmoid = Activation::Sigmoid;
let _tanh    = Activation::Tanh;
let _relu    = Activation::ReLU;
let _leaky   = Activation::LeakyReLU(0.01); // alpha parameter
let _softmax = Activation::Softmax;

// Convenience predicates
assert!(Activation::Sigmoid.is_bounded());
assert!(!Activation::ReLU.is_bounded());
assert!(Activation::Softmax.is_softmax());

// Single-value forward (element-wise, not applicable to Softmax)
let y = Activation::ReLU.forward(-0.5);
assert_eq!(y, 0.0);
```

---

## lattice-transport

`lattice-transport` implements entropy-regularized Optimal Transport (the Sinkhorn
algorithm) in log-domain for numerical stability. It is designed for quantifying
distribution-level similarity between sets of embeddings.

The crate is self-contained — no BLAS, no external math library.

### Basic balanced Sinkhorn solve

```rust
use lattice_transport::{
    DenseCostMatrix, SinkhornConfig, SinkhornSolver, SinkhornWorkspace, uniform_weights,
};

// 2-point problem: source and target each have two atoms
// Cost matrix: moving within the same point is free, moving between points costs 1
let cost = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
let source = uniform_weights(2); // [0.5, 0.5]
let target = uniform_weights(2); // [0.5, 0.5]

let solver = SinkhornSolver::default();
let mut workspace = SinkhornWorkspace::new(2, 2);
let result = solver.solve(&cost, &source, &target, &mut workspace).unwrap();

println!("Converged: {}", result.converged);
println!("Transport cost: {:.4}", result.transport_cost);
println!("Regularized cost: {:.4}", result.regularized_cost);
```

### Sinkhorn configuration

`SinkhornConfig` controls the trade-off between precision and speed:

```rust
use lattice_transport::{
    DenseCostMatrix, SinkhornConfig, SinkhornSolver, SinkhornWorkspace, uniform_weights,
};

let config = SinkhornConfig {
    epsilon: 0.05,              // regularization strength
    max_iterations: 500,
    convergence_threshold: 1e-5,
    check_convergence_every: 10,
    min_marginal: 1e-12,
    error_on_non_convergence: false,
};
let solver = SinkhornSolver { config };

let cost = DenseCostMatrix::new(3, 3, vec![
    0.0, 1.0, 4.0,
    1.0, 0.0, 1.0,
    4.0, 1.0, 0.0,
]);
let source = uniform_weights(3);
let target = uniform_weights(3);
let mut ws = SinkhornWorkspace::new(3, 3);

let result = solver.solve(&cost, &source, &target, &mut ws).unwrap();
println!("OT cost (3x3): {:.4}", result.transport_cost);
```

### Debiased Sinkhorn divergence

The raw Sinkhorn cost is non-zero even when the two distributions are identical, because
the entropy term introduces a positive self-cost. `sinkhorn_divergence` removes this bias:

```rust
use lattice_transport::{
    DenseCostMatrix, SinkhornConfig, SinkhornSolver, SinkhornWorkspace,
    sinkhorn_divergence, uniform_weights,
};

let solver = SinkhornSolver::default();
let cost_xy = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
let cost_xx = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
let cost_yy = DenseCostMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 0.0]);

let source = uniform_weights(2);
let target = uniform_weights(2);

let mut ws_xy = SinkhornWorkspace::new(2, 2);
let mut ws_xx = SinkhornWorkspace::new(2, 2);
let mut ws_yy = SinkhornWorkspace::new(2, 2);

let divergence = sinkhorn_divergence(
    &solver,
    &cost_xy, &cost_xx, &cost_yy,
    &source, &target,
    &mut ws_xy, &mut ws_xx, &mut ws_yy,
).unwrap();

println!("Sinkhorn divergence: {:.6}", divergence.value);
// For identical distributions the debiased divergence approaches 0
```

### Non-uniform marginals

The solver requires strictly positive weights (FP-024: exact zeros are rejected to prevent
phantom-mass injection). Use `normalize_weights()` with a small floor to handle
near-degenerate distributions:

```rust
use lattice_transport::{
    DenseCostMatrix, SinkhornSolver, SinkhornWorkspace,
    normalize_weights, uniform_weights,
};

// 999 units at atom 0, 1 unit at atom 1
let raw = vec![999.0_f32, 1.0];
let marginal = normalize_weights(&raw, 1e-12).unwrap();
// marginal sums to 1.0, all entries are positive
assert!((marginal.iter().sum::<f32>() - 1.0).abs() < 1e-5);
```

### Detecting embedding drift

`detect_drift_records` is the high-level API for comparing two snapshots of an embedding
collection:

```rust
use lattice_transport::{
    DriftConfig, DriftMetricKind, DriftWeighting, EmbeddingRecord, detect_drift_records,
};

let old_embeddings: Vec<Vec<f32>> = (0..10)
    .map(|i| vec![i as f32, (i * 2) as f32])
    .collect();

let new_embeddings: Vec<Vec<f32>> = (0..10)
    .map(|i| vec![i as f32 + 0.1, (i * 2) as f32 + 0.1])
    .collect();

let old_records: Vec<EmbeddingRecord<usize>> = old_embeddings
    .iter()
    .enumerate()
    .map(|(i, emb)| EmbeddingRecord::uniform(i, emb))
    .collect();

let new_records: Vec<EmbeddingRecord<usize>> = new_embeddings
    .iter()
    .enumerate()
    .map(|(i, emb)| EmbeddingRecord::uniform(i, emb))
    .collect();

let config = DriftConfig {
    metric: DriftMetricKind::SquaredEuclidean,
    weighting: DriftWeighting::Uniform,
    ..DriftConfig::default()
};

let report = detect_drift_records(&old_records, &new_records, &config).unwrap();
println!("Drift magnitude: {:.4}", report.summary.drift_magnitude);
println!("Converged: {}", report.summary.converged);
```
