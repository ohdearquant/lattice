# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). All lattice crates share one
workspace version; the workspace release notes are in
[`docs/releases/`](https://github.com/ohdearquant/lattice/tree/main/docs/releases).

## [Unreleased]

## [0.12.1] - 2026-10-14

Documentation and package metadata only; no code changes.

### Documentation

- crates.io keywords and categories; docs.rs builds the documentation with `f16`, `x86-gemm`, `mixture`, `train-backward` and `wgpu-gpu` (#1936).
- The crate documentation builds without rustdoc warnings under the default features and the docs.rs feature set (#1937).
- The README and crate docs no longer point at a stability document that does not exist (#1936).
- The README's supported architectures table lists EmbeddingGemma 2, Gemma 4 E2B and PaddleOCR-VL / ERNIE-4.5 (#1938).

## [0.12.0] - 2026-10-13

### Breaking

- `router_state::resolve_startup` takes `Option<&RouterPin>` instead of `Option<u64>`; build the pin with `RouterPin::parse` (#1916).
- `RouterArtifact` has a new public field, `provenance: RouterProvenance`; use `RouterProvenance::Legacy` for an artifact without recorded lineage (#1916).
- `RouterArtifact` and `ResolvedRouter` no longer implement `Eq`; they keep `PartialEq` (#1916).
- `RouterArtifactError::UnknownFormat.known` is a `String` instead of a `u32` (#1916).
- `ROUTER_ARTIFACT_FORMAT` is 5; format 4 artifacts still load (#1916).

### Added

- `x86-gemm` feature: on Linux x86_64, multi-row `matmul_bt` products run through the `gemm` crate on the calling thread (#1934).
- EmbeddingGemma 2 inputs that mix text and image parts, with the vision tower in f32 on the CPU (#1906).
- `Qwen35Model::prefill_readout`, unstable: one prefill that returns logits at chosen positions over chosen `lm_head` rows (#1915).
- `serve::systemone` request parsing, prompt layout and answer assembly for `decider` checkpoints, without an HTTP route yet (#1914).
- Router artifact format 5 with lineage and admission policy, a committed head per state directory, and `--router-pin <version>[:<hash-prefix>]` (#1916).
- `lattice router init` and `training_state` in `GET /v1/lora` (#1921).
- `RouterPolicy` and `AdapterRouter::route_explored` (`mixture` feature) (#1928).
- `lattice serve --max-cpu-generations <N>`, default 2; a request over the cap gets 503 `server_busy` (#1927).

### Changed

- The SentencePiece tokenizer stores its piece trie in flat arrays and shares it across `with_max_seq_len`; the loaded paraphrase-multilingual-MiniLM-L12-v2 model's heap drops from 148.2 MiB to 59.2 MiB (#1933).
- On x86_64, a single-row `matmul_bt` product with `n * k >= 524,288` splits its output columns across the rayon pool (#1931).

### Fixed

- Metal Qwen3.5 Q4 chat requests with `logprobs` get 400 `unsupported_feature` instead of 500 (#1901).
- `POST /v1/lora/load` and `/v1/lora/unload` validate the request before checking adapter support (#1899).
- A Qwen `config.json` without `text_config` is read from its top level instead of falling back to the MoE preset (#1913).

## 0.1.0 to 0.11.1

These releases are described in the workspace release notes, `docs/releases/v<version>.md`.
The entries below were recorded in this file under the Unreleased heading until 2026-08-11
and shipped in releases up to and including 0.10.0.

### Changed

- **Behavior change (#831):** `lattice serve`'s `/v1/chat/completions` now actually applies a
  request's `reasoning_budget` field to generation, instead of silently accepting and ignoring
  it. A malformed `reasoning_budget` (e.g. a string or object) is now rejected with
  `invalid_request_body`, matching `lattice_serve`'s pre-existing behavior for the same field.
  Clients that were previously sending `reasoning_budget` to `lattice serve` expecting no effect
  will now see it applied.
- **Behavior change (#831):** `lattice_serve`'s `/v1/chat/completions` now accepts a `stop`
  field (string or array of up to 4 non-empty strings) and applies it, instead of rejecting the
  request outright with `unsupported_feature` ("stop is not supported by this server"). Aligns
  with `lattice serve`'s pre-existing `stop` support.
- **Behavior change (#831):** both `lattice serve` and `lattice_serve` now reject a request
  whose `prompt + max_tokens + reasoning_budget` would leave no room for the generation-turn
  delimiter token: the shared context-window check now requires
  `prompt + max_new_tokens + reasoning_budget + 1 <= max_context` (previously `lattice serve`'s
  own HTTP preflight accepted `prompt + max_new_tokens == max_context` exactly, one token
  looser than this).
