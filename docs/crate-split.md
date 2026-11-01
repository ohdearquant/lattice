# The `lattice-inference` crate split and API compatibility

The workspace split inference functionality into smaller crates. The first split crates, `lattice-inference-core` and `lattice-tokenizer`, shipped in 0.14.0. The other crates listed below first shipped in 0.15.0. This page maps old `lattice_inference::...` paths to the crates that define their items.

## Summary

- Most old facade paths continue to compile through compatibility re-exports until 1.0. These paths are experimental and are not a stable API. New code should import from the owning crate.
- `lattice-embed` depends on `lattice-inference-core`; its `lattice-inference` dependency is for development tests. Embedding applications can depend on `lattice-embed`.
- HTTP serving and the `lattice` and `lattice_serve` binaries belong to `lattice-serve`. They are not re-exported by `lattice-inference`.
- The `serve` feature of `lattice-inference`, on by default in 0.14.0, no longer exists. A manifest that enables it fails to resolve; depend on `lattice-serve` instead.

## Which crate to depend on

| You want | Depend on |
| --- | --- |
| Embeddings | `lattice-embed` |
| Existing inference facade imports | `lattice-inference` |
| A specific area | Its owning crate in the table below |
| The HTTP server or CLI binaries | `lattice-serve` |

Workspace crates share version 0.15.0. When depending on both the facade and an owning crate, use the same version.

## What moved where

The paths in the first column are facade paths that remain re-exported by `lattice-inference`. Model type paths are shown where the model module exposes them.

| Old `lattice_inference::...` path | Defined in | First release |
| --- | --- | --- |
| `error::InferenceError` | `lattice-inference-core` | 0.14.0 |
| `tokenizer::{bpe, common, gemma_bpe, sentencepiece, wordpiece}` and tokenizer root re-exports | `lattice-tokenizer` | 0.14.0 |
| `grammar` | `lattice-grammar` | 0.15.0 |
| `weights::{f32_weights, q3_weights, q4_weights, q8_weights}`, `model_format` | `lattice-weights` | 0.15.0 |
| `quant::q4_manifest`, `quant::quarot::*` | `lattice-quant` | 0.15.0 |
| `model::{qwen_config, qwen35_config, gemma4_config, embeddinggemma2_config, bitnet_config}` | `lattice-arch` | 0.15.0 |
| `generation::{GenerateConfig, GenerateOutput}`, `sampling`, `stop_reason` | `lattice-arch` | 0.15.0 |
| `attention` shared APIs, `kv_cache::*`, `batch::*`, `pool::*` | `lattice-compute` | 0.15.0 |
| `model::qwen::QwenModel` | `lattice-model-qwen3-embedding` | 0.15.0 |
| `model::qwen35::Qwen35Model` | `lattice-model-qwen35` | 0.15.0 |
| `model::bert::BertModel`, `model::cross_encoder::CrossEncoderModel` | `lattice-model-bert` | 0.15.0 |
| `model::bert::BertConfig` | `lattice-arch` | 0.15.0 |
| `model::gemma4_model::Gemma4Model` | `lattice-model-gemma4` | 0.15.0 |
| `model::embeddinggemma2::EmbeddingGemma2Model` | `lattice-model-embeddinggemma2` | 0.15.0 |
| `model::ernie45::*` | `lattice-model-ernie45` | 0.15.0 |
| `download::*` | `lattice-model-io` | 0.15.0 |
| `forecast::*` | `lattice-forecast` | 0.15.0 |
| `router_*` modules and feature-gated `mixture` | `lattice-router` | 0.15.0 |
| `backward::*` | `lattice-backward` | 0.15.0 |

Configuration modules are in `lattice-arch`. There is no `lattice-model-config` crate. `lattice-metal-runtime` also shipped in 0.15.0 and provides shared Metal helpers, without an old public path in `lattice-inference`.

## Serving moved to `lattice-serve`

| Former interface | Current interface |
| --- | --- |
| `lattice_inference::serve::...` | `lattice_serve::serve::...` |
| `lattice-inference` feature `serve` | No such feature; depend on `lattice-serve` |
| Binary `lattice` | `lattice-serve` binary `lattice` |
| Metal daemon `lattice_serve` | `lattice-serve` binary `lattice_serve`, requiring `f16` and `metal-gpu` |

`lattice-serve` declares the `f16` and `metal-gpu` features. The `lattice` binary is declared without required features. The standalone `lattice_serve` binary requires both features.

## How compatibility is checked

- `crates/inference/tests/facade_api_parity.rs` imports selected items through facade and owning-crate paths and compares their identities.
- `scripts/public-api-listing.py` reads rustdoc JSON and emits a public item listing, resolving re-exports across workspace crates.

`lattice-inference` is experimental. Its compatibility re-exports are removed at 1.0 and are not a stable API.
