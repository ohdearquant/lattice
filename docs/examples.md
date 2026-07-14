# Lattice Examples

This document provides runnable examples for the main crates in the lattice workspace:
`lattice-embed`, `lattice-fann`, `lattice-transport`, `lattice-inference`, and the LoRA
fine-tuning workflow in `lattice-tune`.

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

---

## lattice-tune

`lattice-tune` provides LoRA fine-tuning for the Qwen3.5 generation model, using the
`train_grad_full` binary (`crates/tune/src/bin/train_grad_full.rs`) for training and the
`generate_lora` binary (`crates/tune/src/bin/generate_lora.rs`) to run generation with the
trained adapter. Both are existing binaries in the crate — there is no separate training API to
call directly.

### Fine-tuning a Qwen3.5 LoRA adapter

`train_grad_full` trains LoRA parameters over a range of Qwen3.5 layers (default: layers 19-23)
using exact reverse-mode gradients, and can save the result as a PEFT-compatible safetensors
adapter.

Training data is JSON Lines under a data directory (default `data/lora-train`):

- `train.jsonl` is required. Each line is a JSON object with `prompt` and `completion` string
  fields; both must be non-empty.
- `valid.jsonl` is optional. If present, it is used for held-out evaluation during training; if
  absent or empty, held-out evaluation is skipped.

```jsonl
{"prompt":"Write a Rust function that returns true for primes.","completion":"\nfn is_prime(n: u64) -> bool { ... }"}
```

```sh
cargo run --release -p lattice-tune --features train-backward --bin train_grad_full -- \
  --model-dir ~/.lattice/models/qwen3.5-0.8b \
  --data-dir data/lora-train \
  --steps 25 \
  --lr 1e-3 \
  --rank 8 \
  --alpha 16 \
  --seq-len 64 \
  --max-train 3 \
  --save adapter.safetensors
```

The `train-backward` feature pulls in `lattice-inference`'s backward-pass support along with the
`safetensors` feature needed by `--save`. The saved adapter targets the `q_proj` and `v_proj`
projections of the trained layers and is written in PEFT safetensors format.

### Running generation with the trained adapter

`generate_lora` loads a Qwen3.5 model and an optional LoRA adapter, then runs ordinary
autoregressive generation:

```sh
cargo run --release -p lattice-tune --features "safetensors,inference-hook" --bin generate_lora -- \
  --model-dir ~/.lattice/models/qwen3.5-0.8b \
  --lora adapter.safetensors \
  --prompt "Write a Rust function that checks if a number is prime" \
  --max-tokens 64
```

Internally, `generate_lora` loads the adapter with `LoraAdapter::from_safetensors`, validates it
against the loaded model's `model.config()`, attaches it with `model.set_lora(Box::new(adapter))`,
and prints `LoRA: ACTIVE` once an adapter is attached, before generating. Omitting `--lora` runs
the base model unmodified.

---

## lattice-inference

The examples below cover the shipped inference-time LoRA, speculative-decoding, and sampling
surfaces. Metal examples require macOS and the `f16,metal-gpu` features.

### Serving a trained LoRA adapter with `chat_metal`

The adapter produced by the training example above is PEFT-compatible, so `chat_metal` can load it
directly. Run the same deterministic prompt once against the base model and once with the adapter:

```sh
cargo build --release -p lattice-inference --features f16,metal-gpu --bin chat_metal

./target/release/chat_metal \
  --model ~/.lattice/models/qwen3.5-0.8b \
  --prompt "Write a Rust function that checks if a number is prime" \
  --max-tokens 64 --temperature 0 --top-k 1 --top-p 1 \
  --repetition-penalty 1

./target/release/chat_metal \
  --model ~/.lattice/models/qwen3.5-0.8b \
  --lora adapter.safetensors \
  --prompt "Write a Rust function that checks if a number is prime" \
  --max-tokens 64 --temperature 0 --top-k 1 --top-p 1 \
  --repetition-penalty 1
```

The second process prints `Loading LoRA adapter` and runs with the adapter installed. Greedy
settings make a before/after text comparison reproducible; whether the text changes depends on
what the adapter learned.

`chat_metal --lora` loads one adapter when the process starts. It does not expose an interactive
command or JSON field that replaces the adapter inside an already-running `chat_metal` process.
Library users can replace the one Metal adapter slot with `load_lora_adapter` and
`unload_lora_adapter`; both operations invalidate the retained cross-turn prefix cache before the
next request.

### Weighted LoRA mixtures

For CPU inference, `blend_lora_adapters` accepts a slice of `(adapter, mixture_weight)` pairs. It
folds each adapter's own `alpha / rank` scale and the supplied mixture weight into one rank-sum
adapter, which can be installed through the ordinary `LoraHook` slot:

```rust,no_run
use std::path::Path;

use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::model::qwen35_config::GenerateConfig;
use lattice_tune::lora::{LoraAdapter, blend_lora_adapters};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = Path::new("/path/to/qwen3.5-0.8b");
    let mut model = Qwen35Model::from_safetensors(model_dir)?;

    let domain = LoraAdapter::from_safetensors(Path::new("domain.safetensors"))?;
    let style = LoraAdapter::from_safetensors(Path::new("style.safetensors"))?;
    domain.validate_against(model.config())?;
    style.validate_against(model.config())?;

    // The request shape is &[(&LoraAdapter, f32)]: 70% domain + 30% style.
    let mixed = blend_lora_adapters(&[(&domain, 0.7), (&style, 0.3)])?;
    model.set_lora(Box::new(mixed));

    let config = GenerateConfig {
        max_new_tokens: 64,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        ..GenerateConfig::default()
    };
    println!("{}", model.generate("Explain ownership in Rust", &config)?.text);
    Ok(())
}
```

This requires `lattice-tune` features `safetensors,inference-hook`. The Metal equivalent is
`MetalQwen35State::generate_with_lora_mixture`, whose request shape is
`&[(&[LoraLayerData], effective_weight)]`. For that lower-level API,
`effective_weight = mixture_weight * alpha / rank`; unlike `blend_lora_adapters`, it cannot infer
`alpha` because `LoraLayerData` does not carry it. The blend is loaded into Metal's single adapter
slot for the request and is always unloaded afterward. There is no LoRA-mixture field on either
HTTP server or on `chat_metal`.

The shipped synthetic benchmark is a runnable example of this Metal request shape:

```sh
cargo run --release -p lattice-inference --features f16,metal-gpu \
  --bin bench_lora_mixture
```

Set `LATTICE_MODEL_DIR` to a Q4 model directory to request its optional GPU-decode half. The
current GPU half has a known Qwen3.5 hybrid-layer bug documented in
[`cli-tools.md`](cli-tools.md#common-failure-modes-including-one-confirmed-bug-not-fixed-here);
the CPU blend measurements still exercise the weighted rank-sum operation.

#### Router refits are caller-managed

`AdapterRouter` is behind lattice-inference's `mixture` feature. It ranks the available adapters
once per request and returns the selected top-k with fixed, uniform `1/k` weights. Online refitting
is a separate lattice-tune `mixture` API: `update_router` consumes `FeedbackEvent` values plus a
`ReplayBuffer`, `DiagonalFisher`, and `RouterUpdateConfig`, and returns a full replacement gate in
`RouterDelta::network_bytes`.

Nothing in Lattice automatically collects feedback, triggers a refit, persists the returned bytes,
or reloads them into a live `AdapterRouter`. The host application must perform that loop by loading
the returned bytes with `Network::from_bytes()` and constructing the next router. The available
knobs are the fields of `RouterUpdateConfig`; there are no CLI or HTTP router-refit knobs. In v1,
Fisher null-space projection is active, while the `ewc_lambda` anchor-penalty path is intentionally
inactive.

### Native MTP and self-speculative decoding

The live native-MTP path is Metal/Q4-only. `MetalQwen35State::from_q4_dir` must find a model config
with MTP layers and the complete MTP tensor set; the BF16 constructor does not load MTP weights.
If `LATTICE_MTP` is set but a required MTP file is missing, loading warns and ordinary decoding is
used.

For a command-line A/B on the same prompt, keep every MTP gate condition identical between the two
runs:

```sh
./target/release/chat_metal \
  --model /path/to/qwen3.5-0.8b-q4 \
  --prompt "Explain the Rust ownership model" \
  --max-tokens 128 --temperature 0 --top-k 1 --top-p 1 \
  --repetition-penalty 1

LATTICE_MTP=1 LATTICE_MTP_VERBOSE=1 \
  ./target/release/chat_metal \
  --model /path/to/qwen3.5-0.8b-q4 \
  --prompt "Explain the Rust ownership model" \
  --max-tokens 128 --temperature 0 --top-k 1 --top-p 1 \
  --repetition-penalty 1
```

Each run prints overall token throughput; the MTP run also prints rounds, accepted extra tokens,
fallbacks, and time spent drafting, verifying, and rolling back. There is no guaranteed speedup:
acceptance depends on the checkpoint and workload, and verification overhead can make MTP slower.
For a Criterion comparison that explicitly sets `enable_mtp: Some(false)` and `Some(true)`, use the
existing benchmark:

```sh
LATTICE_MODEL_DIR=/path/to/qwen3.5-0.8b-q4 \
LATTICE_TOKENIZER_DIR=/path/to/qwen3.5-0.8b \
  cargo bench -p lattice-inference --features metal-gpu,f16 -- mtp_decode
```

The live MTP branch is used only when all of these conditions hold:

- MTP weights loaded successfully and `enable_mtp` is `Some(true)`, or it is `None` and
  `LATTICE_MTP` is set.
- Greedy decoding is selected (`temperature <= 0` and `top_k <= 1`).
- Repetition penalty is exactly `1.0`, and compact GPU top-k is not enabled.
- Grammar, string stops, reasoning budget, and log-probability capture are all disabled.

Otherwise generation deliberately uses the ordinary path. The live Metal implementation drafts
one MTP token per round and verifies it against the target. In greedy mode, accepted and rejected
rounds are token-for-token equivalent to ordinary greedy decoding, including first-wins tied-logit
behavior. Stop tokens are verified before they can terminate generation and are excluded from the
returned text/token IDs, matching ordinary decoding; a wrong draft-EOS is rejected instead of
truncating output.

`LATTICE_SELF_SPEC=1` is a separate greedy-only mechanism that needs a Qwen3.5 hybrid model with
active GDN layers but does not need an MTP head. Set the variable before constructing the Metal
state (for a CLI, before starting the process) so its checkpoint pool is allocated. It drafts up to
four tokens with GDN-only forwards and verifies the longest matching prefix with the full model.
Its accepted, rejected, and fallback EOS cases use the same rule: the target-confirmed stop token
ends generation but is not included in output. This path also has no promised throughput gain; use
`LATTICE_SELF_SPEC_VERBOSE=1` to inspect its counters.

### Sampling-parameter cookbook for `lattice_serve`

`lattice_serve` exposes each sampler control twice: a process flag sets the server default, and the
corresponding JSON field overrides that default for one request.

```sh
cargo run --release -p lattice-inference --features f16,metal-gpu,serve \
  --bin lattice_serve -- \
  --model /path/to/qwen3.5-0.8b-q4 \
  --temperature 0.7 --top-k 50 --top-p 0.9 --repetition-penalty 1.1
```

| Control            | Server default | Shipped behavior                                                                                                                                                                                                         | CLI flag / HTTP field                         |
| ------------------ | -------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------- |
| Temperature        |          `0.7` | `<= 0` is greedy; `1.0` leaves logits unscaled; values between 0 and 1 sharpen the distribution; values above 1 flatten it. No upper bound is enforced by `lattice_serve`.                                               | `--temperature` / `temperature`               |
| Top-k              |           `50` | Keeps the k highest-logit tokens. `1` is greedy and `0` disables this filter. Values at least as large as the vocabulary also leave it unfiltered.                                                                       | `--top-k` / `top_k`                           |
| Top-p              |          `0.9` | Samples from the smallest high-probability prefix whose cumulative mass reaches p. The sampler clamps finite input to `[0, 1]`; `1` disables this filter and `0` keeps only the highest-probability token.               | `--top-p` / `top_p`                           |
| Repetition penalty |          `1.1` | Applies once to every token already present in the prompt or generated history. `1` disables it; values above 1 discourage repeats; values between 0 and 1 boost repeats. Non-positive values are treated as no penalty. | `--repetition-penalty` / `repetition_penalty` |

The JSON numbers accepted by the server are passed into `GenerateConfig` without a separate range
validation step, so the table describes the sampler's actual normalization rather than an HTTP
validation promise. `top_k` is an unsigned integer. For normal use, keep temperature non-negative,
top-p in `[0, 1]`, and repetition penalty at least `1`.

#### Deterministic evaluation

Temperature zero or top-k one takes the deterministic argmax path. Disable repetition adjustment
when exact base-model greedy output is the goal:

```sh
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Summarize Rust ownership in one sentence."}],
    "max_tokens": 64,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1,
    "repetition_penalty": 1,
    "seed": 42
  }'
```

#### Balanced sampling

The shipped defaults combine moderate temperature with top-k and nucleus filtering. A fixed seed
makes a particular sampled run reproducible:

```sh
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Invent three names for a Rust database."}],
    "max_tokens": 96,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "seed": 42
  }'
```

#### Repetition-loop mitigation

For a long response that starts repeating phrases, increase the penalty modestly and narrow the
nucleus. Excessive penalties can damage coherence, so tune from the default rather than jumping to
a very large value:

```sh
curl http://127.0.0.1:11435/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Write a detailed deployment checklist."}],
    "max_tokens": 512,
    "temperature": 0.6,
    "top_k": 40,
    "top_p": 0.85,
    "repetition_penalty": 1.2,
    "seed": 42
  }'
```

`top_k` and `repetition_penalty` are Lattice extensions to the OpenAI-shaped request. A request
field wins over its corresponding process flag; omitted fields inherit the process defaults shown
above.
