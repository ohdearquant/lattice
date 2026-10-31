# The `lattice-inference` crate split and API compatibility

From 0.14 on, code that used to live in `lattice-inference` is moving into smaller crates, one area at a time, as
recorded in [ADR-103](adr/ADR-103-inference-crate-decomposition.md). This page says what moved, where it went, and what
a caller has to change. For almost everyone the answer is nothing.

## Summary

- **Imports from `lattice_inference` keep compiling.** Every public path that existed before a move still exists after
  it and names the same item. The facade re-exports each moved item by name; it never copies one, so a type reached
  through `lattice_inference::...` and through the new crate is one type, and values pass between the two paths
  freely.
- **`lattice-embed` is not split.** Nothing moves out of it, and its use of `lattice-inference` goes through the same
  facade paths. Applications that only need embeddings should keep depending on `lattice-embed`.
- **One exception: the HTTP servers.** The serving stack and the `lattice` and `lattice_serve` binaries moved to the
  `lattice-serve` crate and are not re-exported by `lattice-inference`. See [Serving](#serving-moved-to-lattice-serve).

## Which crate to depend on

| You want                                                     | Depend on                  |
| ------------------------------------------------------------ | -------------------------- |
| Text embeddings in an application                            | `lattice-embed`            |
| Models, generation, tokenizers, weights through stable paths | `lattice-inference`        |
| One area only, with a smaller dependency tree                | the focused crate below    |
| The OpenAI-compatible HTTP server or the `lattice` CLI       | `lattice-serve` (binaries) |

Depending on a focused crate directly is supported and gives the same items. All lattice crates share one version
number and are released together; when you use both `lattice-inference` and a focused crate, give them the same
version.

## What moved where

Each row lists the facade paths that keep working and the crate that now defines the items behind them.

| Facade path (unchanged)                                                                                                                                       | Defined in               | First release |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------------- |
| `lattice_inference::InferenceError`, `lattice_inference::error::InferenceError`                                                                               | `lattice-inference-core` | 0.14.0        |
| `lattice_inference::tokenizer::{bpe, common, gemma_bpe, sentencepiece, wordpiece}` and the tokenizer re-exports at the crate root                             | `lattice-tokenizer`      | 0.14.0        |
| `lattice_inference::grammar` (`GrammarEngine` and the JSON Schema, GBNF and pushdown-automaton items)                                                         | `lattice-grammar`        | 0.15.0        |
| `lattice_inference::weights` (f32, f16, Q3, Q4 and Q8 weight types) and `lattice_inference::model_format`                                                     | `lattice-weights`        | 0.15.0        |
| `lattice_inference::quant::q4_manifest`, `lattice_inference::quant::quarot::{hadamard, rotation}`                                                             | `lattice-quant`          | 0.15.0        |
| `lattice_inference::model::{qwen35_config, gemma4_config, embeddinggemma2_config, bitnet_config}`                                                             | `lattice-model-config`   | 0.15.0        |
| `lattice_inference::quant::quarot::{convert, forward_equivalence, io, lm_head, lora, pipeline, plan, rmsnorm_fusion}` and the `quant::quarot` root re-exports | `lattice-quant`          | 0.15.0        |

`lattice-metal-runtime` (Metal buffer and pipeline helpers) was also split out in 0.15.0. It had no public path in
`lattice-inference`, so there is nothing to keep compatible.

The model types themselves (`QwenModel`, `Qwen35Model`, `BertModel`, `CrossEncoderModel`, the Gemma 4 family), the
generation API (`GenerateConfig`, `GenerateOutput`, sampling, stop reasons), the KV caches, attention and the forward
pass are still defined in `lattice-inference`.

## Serving moved to `lattice-serve`

Serving is an application on top of the inference engine, so it now lives above it in its own crate. This is the one
place where a path is retired instead of re-exported.

| Before (0.14)                                                                                           | From 0.15                                                                             |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `lattice_inference::serve::...`                                                                         | `lattice_serve::serve::...`                                                           |
| `lattice_inference::serving_{cpu, cpu_host, embeddings, factory, host, provider, worker}::...` (hidden) | `lattice_serve::serving_<same name>::...`                                             |
| `lattice-inference` feature `serve` (on by default)                                                     | removed; depend on `lattice-serve`                                                    |
| `cargo install lattice-inference --bin lattice --features f16` (add `metal-gpu` on macOS)               | `cargo install lattice-serve --bin lattice --features f16` (add `metal-gpu` on macOS) |
| the Metal `lattice_serve` daemon (`--features f16,metal-gpu`)                                           | `cargo install lattice-serve --bin lattice_serve --features f16,metal-gpu`            |

Module names inside `lattice-serve` match the old ones, so a migration is a crate-name change in `use` lines. The HTTP
API itself is unchanged; see [serve-http-api.md](serve-http-api.md).

## How compatibility is checked

- `crates/inference/tests/facade_api_parity.rs` imports each moved item through both its facade path and its new crate
  and asserts they are the same type, function or value. A move that dropped or copied an item fails this test.
- Each move compares the facade's public item listing (`scripts/public-api-listing.py`, built from rustdoc JSON) before
  and after the change. The listing must be identical, apart from items a change adds deliberately.
- `cargo semver-checks` does not follow a re-export of an item defined in another crate, so for `lattice-inference` it
  reports moved items as missing even though every one of those paths still compiles. The two checks above are the
  evidence for this crate; the release notes say so for each release.

## What is not covered

- `lattice-inference` is still an experimental-tier crate. The promise here is about paths: an item you could name
  before a move you can still name after it. It is not a promise that signatures never change between minor versions;
  any such change is listed under "Breaking changes" in the [release notes](releases/).
- Items marked `#[doc(hidden)]` and modules named `__internal` are not supported API, in the facade or in the focused
  crates.
