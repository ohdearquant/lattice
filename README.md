# Lattice

Pure Rust inference engine for transformer models on Apple Silicon, with a native macOS app.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/lattice-embed.svg)](https://crates.io/crates/lattice-embed)
[![CI](https://github.com/ohdearquant/lattice/actions/workflows/ci.yml/badge.svg)](https://github.com/ohdearquant/lattice/actions)

**[Quick start](#quick-start-cli)** · **[Lattice Studio](#lattice-studio-macos-app)** · **[Benchmarks](#benchmarks)** · **[Roadmap](#roadmap)**

No ONNX. No Python. No CUDA. No external ML runtime. Lattice implements the full compute graph
in Rust: weight loading, tokenization, forward pass, vector operations, quantization, and LoRA
training. Hand-written Metal shaders accelerate inference on Apple Silicon. SIMD kernels (AVX2
on x86, NEON on ARM) handle the CPU path.

---

## What is Lattice

Lattice is two things in one repo.

**A Rust inference library.** Five published crates covering embeddings, generation, quantization,
LoRA fine-tuning, and optimal transport. Use `lattice-embed` as a library dependency, or run
the `lattice` binary for interactive chat and an OpenAI-compatible HTTP server.

**Lattice Studio: a native macOS app.** A SwiftUI instrument panel that drives the Rust engine
via CLI subprocesses. Train LoRA adapters with a live loss oscilloscope, quantize models with
a before/after comparison, hot-swap adapters in chat with zero reload, and manage your model
library from a single window. To build and install it, follow the step-by-step guide in
[`apps/macos/INSTALL.md`](apps/macos/INSTALL.md).

---

## Capabilities

|                        |                                                                                                                                                                                                                                                                                                                      |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pure Rust compute      | Hand-written SIMD kernels (AVX2/NEON). No C++, no ONNX, no CUDA.                                                                                                                                                                                                                                                     |
| Metal GPU backend      | Native Apple Silicon acceleration via Metal MSL shaders. WGPU fallback for cross-platform.                                                                                                                                                                                                                           |
| Generation models      | Qwen3.5-0.8B / 2B via `lattice chat`/`serve`. Qwen3.6-27B via `lattice chat`/`serve` from a native Q4 checkpoint (requires the Metal GPU build, `--features "f16 metal-gpu"`); safetensors 27B is loader-level only. Qwen3.6-35B-A3B (MoE): config + weight loader support. Hybrid GatedDeltaNet + GQA architecture. |
| Embedding models       | 9 models: BGE, E5, MiniLM, Qwen3-Embedding families. Auto-download for 7 BERT-family variants.                                                                                                                                                                                                                       |
| Three tokenizers       | WordPiece, SentencePiece, BPE. No Hugging Face tokenizers C extension.                                                                                                                                                                                                                                               |
| Quantization           | Q8, Q4, and QuaRot (rotation-based 4-bit). No other engine runs Q4 + LoRA hot-swap on Qwen3.5.                                                                                                                                                                                                                       |
| LoRA                   | Inference hook, hot-swap with no reload, PEFT safetensors format, training via `lattice-tune`.                                                                                                                                                                                                                       |
| HTTP API               | OpenAI-compatible `/v1/chat/completions` via `lattice serve`.                                                                                                                                                                                                                                                        |
| Safetensors native     | Memory-mapped weight loading. Single-file and sharded checkpoints.                                                                                                                                                                                                                                                   |
| KV cache               | Incremental decoding with key-value caching.                                                                                                                                                                                                                                                                         |
| Speculative decoding   | Draft-model acceleration on the CPU path.                                                                                                                                                                                                                                                                            |
| Grammar decoding       | Constrained output via a pushdown automaton. OpenAI string-level stop sequences.                                                                                                                                                                                                                                     |
| MRL support            | Matryoshka truncation for Qwen3-Embedding models (output dimension >= 32).                                                                                                                                                                                                                                           |
| LRU cache              | `CachedEmbeddingService` with sharded in-memory cache and hit/miss stats.                                                                                                                                                                                                                                            |
| Knowledge distillation | Train small models from Claude/GPT/Gemini teacher soft labels via `lattice-tune`.                                                                                                                                                                                                                                    |
| Optimal transport      | Sinkhorn-Knopp solver for embedding drift detection via `lattice-transport`.                                                                                                                                                                                                                                         |

---

## Benchmarks

Measured on Apple M2 Max, Qwen3.5-0.8B. Greedy decoding, median of 5 runs.

These are **decode-only** rates: steady-state token generation speed with prompt prefill and
one-time model load excluded. Per-token decode latency is fit as `t = intercept + slope * ctx`
across several context lengths (the "slope method") and reported as `1000 / t` at each. `Context`
is the number of tokens already in the KV cache when the measured token is decoded. The wall-clock
speed you observe in an interactive chat is lower than these numbers, because it also includes
prefill of the whole prompt (see "Why throughput falls with context" below).

| Context | Lattice (Q8, f16 head) | Ollama (Q8_0) | MLX (Q8 g64, AMX) | Lattice vs Ollama |
| ------- | ---------------------- | ------------- | ----------------- | ----------------- |
| 64 tok  | **187 tok/s**          | 93            | 265               | 2.0x              |
| 128 tok | **171 tok/s**          | 92            | 263               | 1.9x              |
| 256 tok | **146 tok/s**          | 88            | 260               | 1.6x              |

### Why throughput falls with context

Decode slows as context grows (187 → 171 → 146 tok/s above) because every generated token attends
over the entire KV cache:

- The grouped-query attention (GQA) layers do work proportional to context length on each token: the
  full key/value cache is streamed through memory every step. Decode is memory-bandwidth bound, so
  per-token latency rises linearly with context (`t ≈ intercept + slope * ctx`) and throughput is its
  reciprocal.
- The linear-attention (gated-delta) layers are constant-time in context and do not contribute to the
  slope.

In an interactive chat this compounds with prompt prefill. The serve path currently re-prefills the
whole conversation on every turn ([#462](https://github.com/ohdearquant/lattice/issues/462)), a
separate and larger cost than the decode slope, and the main reason a long chat feels progressively
slower.

MLX uses Apple's private MPS/AMX matrix engines. Lattice uses the public Metal compute API,
the same tier as Ollama. MLX decodes faster than Lattice at raw throughput. Lattice's edge is
portability (pure Rust, zero Python, zero framework) plus capabilities neither Ollama nor MLX
provide for this model family:

| Capability                          | Lattice | MLX | Ollama |
| ----------------------------------- | ------- | --- | ------ |
| QuaRot 4-bit (rotation-based quant) | yes     | no  | no     |
| Q4 + LoRA hot-swap (no reload)      | yes     | no  | no     |
| Pure Rust, zero Python or framework | yes     | no  | no     |

PPL benchmark (wikitext-2, from `docs/bench_results/perplexity.tsv`): Lattice q4 19.27, q4-QuaRot 19.95 (2048 tokens); MLX q8 15.82, q4 18.18 (2041 tokens). Reproduce:
`./scripts/bench_context_scaling.sh`

---

## Crates

| Crate                                    | Description                                                                                                                                                           | crates.io                                                                                                |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [`lattice-embed`](crates/embed/)         | Embedding service: `EmbeddingService` trait, `NativeEmbeddingService`, `CachedEmbeddingService`, SIMD cosine/dot/euclidean, backfill                                  | [![](https://img.shields.io/crates/v/lattice-embed.svg)](https://crates.io/crates/lattice-embed)         |
| [`lattice-inference`](crates/inference/) | Transformer kernel: safetensors loading, BERT/BGE/Qwen3 forward pass, three tokenizers, Metal/WGPU backends, LoRA hooks, KV cache, speculative decoding, quantization | [![](https://img.shields.io/crates/v/lattice-inference.svg)](https://crates.io/crates/lattice-inference) |
| [`lattice-fann`](crates/fann/)           | Fast neural network primitives: `NetworkBuilder`, pre-allocated layers, zero-alloc forward pass, backprop trainer                                                     | [![](https://img.shields.io/crates/v/lattice-fann.svg)](https://crates.io/crates/lattice-fann)           |
| [`lattice-tune`](crates/tune/)           | Training: knowledge distillation pipeline, dataset management, LoRA adapter management, model registry                                                                | [![](https://img.shields.io/crates/v/lattice-tune.svg)](https://crates.io/crates/lattice-tune)           |
| [`lattice-transport`](crates/transport/) | Optimal transport: Sinkhorn-Knopp (balanced + unbalanced), Wasserstein barycenters, embedding drift detection                                                         | [![](https://img.shields.io/crates/v/lattice-transport.svg)](https://crates.io/crates/lattice-transport) |

Two leaf crates (`fann`, `transport`) have zero intra-workspace dependencies. `inference` is
standalone by default and pulls in `fann` only when the optional `mixture` feature is enabled.
All three can be used on their own.

---

## Quick Start: Embeddings

```toml
[dependencies]
lattice-embed = "0.4"
tokio = { version = "1", features = ["full"] }
```

```rust
use lattice_embed::{EmbeddingService, EmbeddingModel, NativeEmbeddingService};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let service = NativeEmbeddingService::default();

    // Single embedding (BGE-small-en-v1.5, 384 dimensions)
    let embedding = service
        .embed_one("The quick brown fox jumps over the lazy dog", EmbeddingModel::default())
        .await?;

    println!("Dimensions: {}", embedding.len()); // 384

    // Batch
    let texts = vec![
        "First document".to_string(),
        "Second document".to_string(),
    ];
    let embeddings = service.embed(&texts, EmbeddingModel::BgeSmallEnV15).await?;

    // SIMD-accelerated similarity
    let similarity = lattice_embed::utils::cosine_similarity(&embeddings[0], &embeddings[1]);
    println!("Similarity: {:.4}", similarity);

    Ok(())
}
```

Model weights are downloaded from HuggingFace on first use and cached at `~/.lattice/models`
(or `$LATTICE_MODEL_CACHE`).

### GPU acceleration (macOS)

```toml
lattice-embed = { version = "0.4", features = ["metal-gpu"] }
```

### Cross-platform GPU

```toml
lattice-embed = { version = "0.4", features = ["wgpu-gpu"] }
```

---

## Quick Start: CLI

### Install

Three ways to get `lattice`, in order of convenience:

**1. `cargo install` (from [crates.io](https://crates.io/crates/lattice-inference)):**

```bash
# CPU build (Linux/macOS). f16 is required to load the BF16/F16 safetensors
# that HuggingFace checkpoints ship in.
cargo install lattice-inference --bin lattice --features f16

# With Metal GPU (macOS only)
cargo install lattice-inference --bin lattice --features f16,metal-gpu
```

This installs `lattice` to `~/.cargo/bin/lattice`. Requires Rust 1.93+
(`rustup update` if `cargo install` complains about the `rust-version`).

**2. Prebuilt release binaries:**

[GitHub releases](https://github.com/ohdearquant/lattice/releases) ship
`lattice-<version>-<target>.tar.gz` for `aarch64-apple-darwin` (macOS,
Metal-enabled), `x86_64-unknown-linux-gnu`, and `aarch64-unknown-linux-gnu`
(both CPU-only), plus a matching `.sha256` file. Check the release's Assets
list first: releases published before this workflow landed have no prebuilt
binaries (use `cargo install` for those versions).

```bash
VERSION=<version>   # replace with a release whose Assets list includes lattice-* tarballs
TARGET=aarch64-apple-darwin   # or x86_64-unknown-linux-gnu / aarch64-unknown-linux-gnu

curl -LO "https://github.com/ohdearquant/lattice/releases/download/v${VERSION}/lattice-${VERSION}-${TARGET}.tar.gz"
curl -LO "https://github.com/ohdearquant/lattice/releases/download/v${VERSION}/lattice-${VERSION}-${TARGET}.tar.gz.sha256"

# Verify before extracting
shasum -a 256 -c "lattice-${VERSION}-${TARGET}.tar.gz.sha256"

tar -xzf "lattice-${VERSION}-${TARGET}.tar.gz"
./lattice-${VERSION}-${TARGET}/lattice chat --model ~/.lattice/models/qwen3.5-0.8b
```

No Homebrew tap yet — tracked as a follow-up once release binaries have shipped
for a few versions ([#633](https://github.com/ohdearquant/lattice/issues/633)).

**3. Build from source** (requires Rust 1.93+ and, for Metal, macOS 14+):

```bash
git clone https://github.com/ohdearquant/lattice
cd lattice

# CLI binary (chat + serve). The f16 feature is required to load the
# BF16/F16 safetensors that HuggingFace checkpoints ship in.
cargo build --release -p lattice-inference --bin lattice --features f16

# Interactive chat
./target/release/lattice chat --model ~/.lattice/models/qwen3.5-0.8b

# OpenAI-compatible HTTP server
./target/release/lattice serve --model ~/.lattice/models/qwen3.5-0.8b --port 8080
```

With Metal GPU (macOS only):

```bash
cargo build --release -p lattice-inference --bin lattice --features metal-gpu,f16
```

Beyond the unified `lattice` binary, several standalone tools live in
`crates/inference/src/bin/` for quantization, direct Metal-GPU chat, perplexity scoring, and
LoRA-mixture benchmarking — see [`docs/cli-tools.md`](docs/cli-tools.md) for verified command
sequences and flag references.

### Raspberry Pi / ARM Linux (aarch64)

Lattice builds and runs on 64-bit ARM Linux with no GPU: the CPU path dispatches
hand-written NEON kernels at runtime. Linux aarch64 is part of CI (`ubuntu-24.04-arm`),
and the same steps apply to a Raspberry Pi running a 64-bit OS.

Requirements and tips:

- **64-bit OS required.** Raspberry Pi OS (64-bit) or Ubuntu Server for Pi. Check with
  `uname -m` — it must say `aarch64`, not `armv7l`.
- **Rust via rustup** (1.93+): `curl https://sh.rustup.rs -sSf | sh`, then build with
  `--features f16` exactly as above — without it, BF16/F16 checkpoints fail at load
  time. Leave `metal-gpu` off — it is macOS-only.
- **Memory: 8 GB strongly recommended for chat models.** The CPU path converts bf16
  weights to f32 in RAM, so Qwen3.5-0.8B occupies roughly 3.2 GB steady-state plus KV
  cache and activations — and the load-time peak is higher still, because the mapped
  bf16 source and converted f32 tensors coexist during conversion. On a 4 GB board this
  will swap-thrash or be OOM-killed. Embedding models are much smaller (bge-small is
  ~130 MB) and run comfortably on 4 GB.
- **Get a chat model** the same way as on macOS — download from HuggingFace and point
  `--model` at the directory:

  ```bash
  pip install -U "huggingface_hub[cli]"
  huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir ~/.lattice/models/qwen3.5-0.8b
  ./target/release/lattice chat --model ~/.lattice/models/qwen3.5-0.8b
  ```

- **Compile time:** a release build takes a while on a Pi (expect tens of minutes on a
  Pi 5). Cross-compiling from a faster machine with target `aarch64-unknown-linux-gnu`
  works too.
- **Set expectations:** this is a CPU-only, memory-bandwidth-bound path. Decode speed is
  far below the Apple Silicon numbers in the benchmark table above — treat a Pi as a
  functional target for experimenting with a fully self-contained Rust inference stack,
  not a fast one. We haven't published Pi throughput numbers yet; benchmark reports from
  real boards are welcome in the issues.

### HTTP API

`lattice serve` exposes an OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-0.8b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

---

## Lattice Studio (macOS App)

Lattice Studio is a native SwiftUI app for macOS 14+. It wraps the Rust engine in an
instrument-panel interface: live loss curves, before/after quantization comparisons,
LoRA hot-swap in chat, and a model library manager.

> New to the app? [`apps/macos/INSTALL.md`](apps/macos/INSTALL.md) is a step-by-step
> install, get-a-model, and first-chat guide. The reference below covers building the
> bundle and what each screen does.

### Build and package

```bash
# Requires: Xcode (Swift 6.3+), Rust toolchain
./apps/macos/scripts/package-app.sh
```

This builds the Swift frontend, compiles all Rust engine binaries in release mode, and
assembles a self-contained `LatticeStudio.app` bundle at `apps/macos/dist/`. A `.dmg`
and `.zip` are also produced. The packaged app needs no Rust toolchain on the recipient
machine.

```bash
# Skip Swift rebuild (use existing build output)
./apps/macos/scripts/package-app.sh --skip-build

# Skip Cargo rebuild
./apps/macos/scripts/package-app.sh --skip-cargo
```

### Install

Drag `LatticeStudio.app` from the `.dmg` to `/Applications`.

The app is ad-hoc signed. On first launch, right-click and choose "Open" to bypass
Gatekeeper, then click "Open" in the dialog. macOS remembers the exception for subsequent
launches. Alternatively:

```bash
xattr -dr com.apple.quarantine /Applications/LatticeStudio.app
```

### What is in Lattice Studio

**Models (cmd-1).** A table of all local models and adapters under `~/.lattice/models`, with
file manifests, config details (18 GatedDeltaNet + 6 GQA layers called out explicitly), and
Download/Verify/Reveal in Finder actions.

**Train (cmd-2).** LoRA fine-tuning with a live loss oscilloscope. A 56-point hero loss numeral
ticks digit-by-digit as the run progresses. Scrub the loss curve to freeze all metric readouts
(lr, grad-norm, tok/s, ETA) to any step. Results stream as line-delimited JSON from the engine.

**Quantize (cmd-3).** Q4 or QuaRot quantization. A side-by-side comparison shows size,
bits, and estimated PPL before and after. True-scale bars animate to show the compression
ratio. A gate pill states the result: PASS, WARN, or FAIL.

**Chat (cmd-4).** Side-by-side base vs. base+adapter columns streaming the same prompt in
parallel. The adapter selector is a console fader: slide it and the engine swaps the adapter
with no reload. A "0 ms reload" stamp confirms it. Each column shows live tok/s and time-to-first-token.

**Data (cmd-5).** Paste raw text or load files. Preview the derived `{prompt, completion}` pairs
as a JSONL table, validate token counts (using the same tokenizer the engine uses), and export
straight into the Train dataset field.

**Runs (cmd-6).** Archive of every training and quantization run. Select a row to reopen its
exact config and view the frozen loss curve.

The command bar (cmd-K) runs everything from a single keyboard shortcut:
`train qwen3.5 r8` or `quantize quarot` parse into argument chips and fire a run.

---

## Supported Models

### Embedding models

| Variant                             | HuggingFace ID                                                | Dims  | Max tokens | Auto-download  |
| ----------------------------------- | ------------------------------------------------------------- | ----- | ---------- | -------------- |
| `BgeSmallEnV15`                     | `BAAI/bge-small-en-v1.5`                                      | 384   | 512        | yes            |
| `BgeBaseEnV15`                      | `BAAI/bge-base-en-v1.5`                                       | 768   | 512        | yes            |
| `BgeLargeEnV15`                     | `BAAI/bge-large-en-v1.5`                                      | 1024  | 512        | yes            |
| `MultilingualE5Small`               | `intfloat/multilingual-e5-small`                              | 384   | 512        | yes            |
| `MultilingualE5Base`                | `intfloat/multilingual-e5-base`                               | 768   | 512        | yes            |
| `AllMiniLmL6V2`                     | `sentence-transformers/all-MiniLM-L6-v2`                      | 384   | 256        | yes            |
| `ParaphraseMultilingualMiniLmL12V2` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384   | 128        | yes            |
| `Qwen3Embedding0_6B`                | `Qwen/Qwen3-Embedding-0.6B`                                   | 1024  | 8192       | local dir only |
| `Qwen3Embedding4B`                  | `Qwen/Qwen3-Embedding-4B`                                     | 2560* | 8192       | local dir only |

*Qwen3-Embedding-4B and 0.6B support MRL truncation to any dimension >= 32.
BGE v1.5 uses CLS pooling. E5 and MiniLM use mean pooling.
E5 `embed_passage()` applies the `"passage: "` prefix automatically.

### Generation models (local files required)

| Config preset                  | Description                                                                                                      |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `Qwen35Config::qwen35_0_8b`    | 24 layers, 1024 hidden, 1 MTP layer. Base decode shipped; MTP experimental.                                      |
| `Qwen35Config::qwen35_2b`      | 24 layers, 2048 hidden, dense FFN, tied embeddings.                                                              |
| `Qwen35Config::qwen36_35b_a3b` | 40 layers, MoE 256 experts top-8. Config and weight loader supported.                                            |
| `Qwen35Config::qwen36_27b`     | 64 layers, 5120 hidden, dense FFN. Runs in `lattice chat`/`serve` from a native Q4 checkpoint (Metal GPU build). |

The Qwen3.5 architecture uses a hybrid of 18 GatedDeltaNet layers and 6 GQA attention layers.
Lattice is the only open-source engine that correctly runs this hybrid recurrence at Q4 on Apple Silicon.

---

## Model Selection (Embeddings)

```rust
use lattice_embed::EmbeddingModel;

// Fast English retrieval
let model = EmbeddingModel::BgeSmallEnV15;   // 384-dim, fastest, auto-download

// Balanced quality/speed
let model = EmbeddingModel::BgeBaseEnV15;    // 768-dim, auto-download

// Best quality, English
let model = EmbeddingModel::BgeLargeEnV15;   // 1024-dim, auto-download

// Multilingual retrieval
let model = EmbeddingModel::MultilingualE5Base;  // 768-dim, 100+ languages

// Long context + multilingual (local files required)
let model = EmbeddingModel::Qwen3Embedding0_6B;  // 1024-dim, 8K context

// MRL: variable output dimension
use lattice_embed::ModelConfig;
let config = ModelConfig::try_new(EmbeddingModel::Qwen3Embedding4B, Some(512))?;
```

---

## Vector Operations

`lattice-embed` exposes SIMD-accelerated vector utilities as a stable public API:

```rust
use lattice_embed::utils;

// Runtime dispatch: AVX2 on x86_64, NEON on aarch64, scalar fallback elsewhere
let sim = utils::cosine_similarity(&a, &b);
let dot = utils::dot_product(&a, &b);
let dist = utils::euclidean_distance(&a, &b);

utils::normalize(&mut vector);  // in-place L2 normalization

// Batch operations
let sims = utils::batch_cosine_similarity(&pairs);
```

Measured performance on normalized vectors (internal benchmarks, subject to hardware):

| Operation                    | Scalar   | SIMD    |
| ---------------------------- | -------- | ------- |
| cosine similarity (384-dim)  | ~650 ns  | ~90 ns  |
| cosine similarity (768-dim)  | ~1300 ns | ~180 ns |
| cosine similarity (1024-dim) | ~1700 ns | ~240 ns |

---

## Feature Flags

### lattice-embed

| Feature     | Default | Description                                 |
| ----------- | ------- | ------------------------------------------- |
| `native`    | yes     | Pure Rust inference via `lattice-inference` |
| `metal-gpu` | no      | Metal GPU acceleration (macOS)              |
| `avx512`    | no      | AVX-512 SIMD kernels (requires nightly)     |

### lattice-inference

| Feature     | Default | Description                                            |
| ----------- | ------- | ------------------------------------------------------ |
| `f16`       | no      | Half-precision weights                                 |
| `metal-gpu` | no      | Metal compute backend                                  |
| `wgpu-gpu`  | no      | WGPU cross-platform GPU backend                        |
| `download`  | yes     | HuggingFace weight download with checksum verification |
| `backfill`  | no      | Re-embedding coordinator (requires `rusqlite`)         |

---

## Architecture

```
Application
    |
    v
lattice-embed          (public API: embedding service, SIMD distance ops, LRU cache)
    |
    v
lattice-inference      (transformer kernel: BERT/Qwen3 forward pass, tokenizers, weights)
    |
    +---> CPU (primary)      Metal (macOS)     WGPU (fallback)
          AVX2/NEON kernels   Apple Silicon      Vulkan/DX12


lattice-fann           (standalone: tiny network primitives, <5ms CPU inference)
lattice-transport      (standalone: optimal transport math, Wasserstein distances)
lattice-tune           (depends on fann + inference: LoRA, distillation, model registry)
```

---

## Running Benchmarks

### Embedding throughput

```bash
cargo bench --package lattice-embed
```

### Metal GPU decode (macOS only, requires model weights)

```bash
cargo bench -p lattice-inference --features metal-gpu,f16 -- metal_decode
```

### Context scaling (Qwen3.5-0.8B vs Ollama vs MLX)

```bash
./scripts/bench_context_scaling.sh
```

Performance depends on hardware, model size, batch size, and sequence length. Run benchmarks
on your target hardware for representative numbers.

---

## Roadmap

Three capability lanes are next for the engine:

- **Vision tensors**: image input through the pure-Rust graph, from patch embedding through a
  vision encoder and projector into the language model.
- **Audio tensors**: on-device speech, starting with a mel-spectrogram frontend and audio
  encoder for speech input; speech output follows.
- **Gemma 4 model family**: broaden the engine beyond Qwen with a second model family.

Tracking issues with first-milestone slices: see the
[issue tracker](https://github.com/ohdearquant/lattice/issues).

---

## Documentation

- [Architecture](docs/architecture.md): crate dependency graph, design decisions, stability tiers
- [Models](docs/models.md): full model support matrix, attention variants, inference features
- [Getting started](docs/getting-started.md): step-by-step setup guide
- [Examples](docs/examples.md): code samples for common tasks
- [CLI tool walkthroughs](docs/cli-tools.md): verified command sequences for `quantize_q4`,
  `chat_metal`'s full flag surface, `ppl_metal`, and `bench_lora_mixture`
- [Capability matrix](docs/capability-matrix.md): `lattice` CLI vs `lattice_serve` endpoint/field parity
- ADR directory: `docs/adr/`

---

## Publishing

Publish order follows the internal dependency DAG (dependencies before dependents): `fann` and
`transport` (leaves), wait 30s, then `inference` (depends on `fann`), wait 30s, then `embed`
and `tune`.

```bash
make publish
```

---

## License

Apache-2.0. See [LICENSE](LICENSE).

Built by [Ocean (HaiyangLi)](https://github.com/ohdearquant). Powers
[khive](https://khive.ai), a cognitive infrastructure for AI agents.
