# Using lattice-inference Directly

## Scope

`lattice-inference` is **Experimental** and primarily intended for contributors working on the
transformer kernel itself, or for low-level integrations that need direct control over model
loading, weight formats, or the forward pass. Application code that just needs embeddings should
use `lattice-embed` instead — see [`docs/architecture.md`](architecture.md#when-to-use-which-crate)
for the crate selection guide.

This document covers the two supported entry points into `lattice-inference` as a standalone
library: the Qwen3 embedding path (`QwenModel`) and the Qwen3.5 generation path (`Qwen35Model`).

## Local Checkpoint Layout

Both model types load from a local directory containing HuggingFace-style safetensors weights:

- Weights: either a single `model.safetensors` file, or a sharded checkpoint described by
  `model.safetensors.index.json`. The loader checks for `model.safetensors` first and falls back
  to the index file.
- Tokenizer: `tokenizer.json` (a HuggingFace `tokenizers` file) is required.
- Config: `config.json` is read when present; if absent, the loader falls back to a compiled-in
  default configuration for the model's size variant.

## Qwen3 Embedding Example

The Qwen3 embedding path uses `QwenModel`, re-exported from the crate root:

```rust
use lattice_inference::QwenModel;
use std::path::Path;

let model = QwenModel::from_directory(Path::new("/path/to/qwen3-embedding-0.6b"))?;
let embedding: Vec<f32> = model.encode("The quick brown fox jumps over the lazy dog")?;
```

This path is exercised by `crates/inference/examples/bench_embedding.rs`, which can be run with:

```sh
cargo run --release -p lattice-inference --example bench_embedding --features f16
```

(`bench_embedding` is registered in `Cargo.toml` as a Cargo `[[example]]` target and requires the
`f16` feature — run it with `--example`, not `--bin`.)

## Qwen3.5 Generation Example

The Qwen3.5 generation path uses `Qwen35Model`, available under the `model` module
(`lattice_inference::model::qwen35::Qwen35Model` — it is not re-exported at the crate root):

```rust
use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::model::qwen35_config::GenerateConfig;
use std::path::Path;

let model = Qwen35Model::from_safetensors(Path::new("/path/to/qwen3.5-0.8b"))?;
let gen_cfg = GenerateConfig::default();
let output = model.generate("Write a haiku about Rust.", &gen_cfg)?;
```

The same path backs the `qwen35_generate` example binary:

```sh
cargo run --release -p lattice-inference --bin qwen35_generate -- \
  --model-dir PATH --prompt "Hello" --max-tokens 64 --seed 42
```

For the full forward-pass trace of this path — from safetensors loading through tokenization,
prefill, and the per-token decode loop — see the "Forward Pass Pipeline" section of
[`docs/architecture.md`](architecture.md#forward-pass-pipeline), with a deeper function-by-function
walkthrough in [`docs/forward-pass.md`](forward-pass.md).

## Tokenizer Note

Both model types tokenize through the shared `Tokenizer` trait:

```rust
fn tokenize(&self, text: &str) -> TokenizedInput;
```

The verified trait method is `tokenize`, not `encode`. `encode` is a method on `QwenModel`
specifically (the embedding path) and combines tokenization with the forward pass and pooling;
it is not part of the `Tokenizer` trait and does not apply to the Qwen3.5 generation path.

## Sampling Note

The generic sampling helpers live in the `sampling` module. The Qwen3.5 generation path uses its
own internal sampling helper (temperature/greedy fallback, repetition penalty, top-k, top-p); that
helper is a private implementation detail of `model::qwen35` and is not part of the public API.

## Feature Flags

- `f16` — required to build the `bench_embedding` example target and to use f16 weight storage.
- `train-backward` — enables backward-pass support for training and LoRA workflows (see
  [`docs/examples.md`](examples.md) for the `lattice-tune` training walkthrough). It is not needed
  for ordinary inference.
