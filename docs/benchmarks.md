# Benchmarks

Lattice uses [Criterion](https://github.com/bheisler/criterion.rs) for micro-benchmarks and
standalone example binaries for end-to-end profiling. No benchmark numbers are recorded in
this document — hardware varies. Run the benchmarks on your target machine.

## Running the Benchmarks

### Embedding SIMD operations (no model download required)

```sh
cargo bench -p lattice-embed
```

Runs the benchmarks in `crates/embed/benches/`:

- `embeddings.rs` — cosine similarity, euclidean distance, L2 normalization across 384/768/1024-d
  vectors at batch sizes 1, 10, 100, 1000
- `simd_bench.rs` / `simd.rs` — raw SIMD kernel throughput (dot product, cosine, normalize)
- `simsimd_comparison.rs` — comparison between SIMD tiers (scalar, AVX2, NEON)

Results land in `target/criterion/`.

### Inference kernel benchmarks

```sh
cargo bench -p lattice-inference
```

Benchmarks in `crates/inference/benches/`:

- `inference_bench.rs` — matmul, attention, layer norm, GELU, softmax at BERT-small dimensions
- `attention_bench.rs` — multi-head attention at varying sequence lengths
- `compute_attention_bench.rs` — raw attention kernel throughput
- `decode_attn_bench.rs` — decoder attention path (Qwen3)
- `kv_cache_layout_bench.rs` — KV cache access patterns (flat vs. paged)
- `tokenizer_bench.rs` — tokenizer throughput for WordPiece/BPE/SentencePiece
- `e2e_bench.rs` — end-to-end embedding latency (requires downloaded models, marked `#[ignore]`)
- `topk_readback.rs` — top-k sampling readback from GPU

### Optimal transport benchmarks

```sh
cargo bench -p lattice-transport
```

Benchmarks in `crates/transport/benches/round0_transport.rs`:

- Sinkhorn convergence at varying problem sizes (n=10 to n=500)
- Log-domain vs. direct Sinkhorn stability comparison
- Barycenter computation throughput
- Unbalanced transport (KL-relaxed) vs. balanced

### End-to-end example benchmarks

These require model weights to be already cached:

```sh
# Full embedding pipeline: tokenize → forward → pool → normalize
cargo run -p lattice-inference --example bench_embedding --release

# Metal GPU path on Apple Silicon
cargo run -p lattice-inference --example bench_metal --release --features metal-gpu

# Concurrent embedding throughput
cargo run -p lattice-inference --example bench_concurrent --release
```

## What the Benchmarks Cover

**SIMD distance operations** (`embed` benchmarks): The primary claim is that cosine similarity
and L2 normalization dispatch correctly to AVX2/NEON at runtime. The benchmarks verify this
by comparing throughput across vector dimensions and batch sizes.

**Transformer forward pass** (`inference` benchmarks): Each sub-component is benchmarked in
isolation — matmul, layer norm, attention — so regressions in individual kernels are visible
without running a full model.

**Attention scaling**: Sequence length strongly affects attention cost. The attention benchmarks
cover 32, 64, 128, 256, and 512 token sequences with BERT-small head dimensions.

**KV cache layout**: The `kv_cache_layout_bench.rs` benchmark compares flat (contiguous)
and paged KV cache memory layouts. This matters for batched inference where sequence lengths
differ.

**Tokenizer throughput**: Tokenization is often overlooked as a bottleneck. The tokenizer
benchmark measures tokens/second for all three tokenizer implementations.

**Optimal transport**: Sinkhorn iteration count to convergence and wall-clock time at
realistic problem sizes (n=50 to n=200 source/target points is typical for drift detection).

## SIMD Dispatch

Lattice uses runtime CPU feature detection. No build flags are needed for AVX2 or NEON —
the dispatcher in `crates/embed/src/simd/tier.rs` selects the fastest available path.

AVX-512 requires the `avx512` feature flag and a nightly compiler:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench -p lattice-embed --features avx512 +nightly
```

## Memory Footprint Reference

These are rough figures for planning purposes, not performance guarantees:

- BGE-small (f32): ~130 MB weight file, ~140 MB RSS at runtime (mmap + overhead)
- BGE-base (f32): ~435 MB weight file
- BGE-large (f32): ~1.3 GB weight file
- Qwen3-0.6B (f32): ~2.4 GB weight file (sharded checkpoint supported)
- Qwen3-4B (f32): ~16 GB weight file (sharded checkpoint required)

Use the `f16` or `Q8` feature flags in `lattice-inference` to halve or quarter the weight
footprint at the cost of some numerical precision.

## Profiling a Forward Pass

`QwenModel` exposes per-component profiling via `encode_with_profile()`:

```rust
let (embedding, timings) = model.encode_with_profile("your text")?;
// timings.tokenize_us, timings.embed_lookup_us, timings.layers[i], etc.
```

`BertModel` does not currently expose per-layer timings. Use `bench_profile.rs` for
whole-pass measurement.
