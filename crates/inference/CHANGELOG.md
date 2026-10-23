# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). All lattice crates share one
workspace version; the workspace release notes are in
[`docs/releases/`](https://github.com/ohdearquant/lattice/tree/main/docs/releases).

## [Unreleased]

### Changed

- **Behavior change:** Qwen chat requests now turn thinking off when `chat_template_kwargs.enable_thinking` is omitted. A positive request `reasoning_budget` enables thinking; a server-configured `generation.reasoning_budget` alone does not. Set `enable_thinking: true` to keep the previous default.
- Reasoning-budgeted requests now close the thinking block at the first line boundary after the budget, or at most 32 tokens later, instead of at the budget itself. The decode cap and answer budget are unchanged.

## [0.13.0] - 2026-10-18

### Breaking

- `router_init::bootstrap_policy` takes the tau and the exploration scale, and `BootstrapSpec` gains `tau` and `exploration_scale` (#1958).
- `ResolvedRouter::report` takes the applied policy; `RouterReport` gains `applied_policy` and no longer derives `Copy` (#1958).
- `ServingRouter::new` and `ServedRouter::new` take `learning: bool` (#1958). `ServingRouter::route` and `ServedRouter::route` take the context vector by value and return `RoutedRequest` (#1949), and take the completion id (#1958).

### Added

- Reverso time-series forecasting on the CPU (nano, small and full), the `forecast` module and the `lattice forecast` subcommand (#1962); Chronos-2 quantile forecasting (#1983) and Toto 2.0 forecasting (#1984) on the CPU.
- Opt-in Metal path for the EmbeddingGemma 2 vision tower (macOS, `metal-gpu`) (#1940), on the simdgroup-matrix kernels `gemm_f32_sg` and `eg2_attention_sg_hd64` (Apple7 family or later) (#1943), with each layer's attention block in one command buffer (#1944).
- Model-format detection from safetensors headers and `config.json`: `ModelFormat` gains `MlxQuantized`, `Gguf` and `Refused`, and `inspect_format` returns the reason for a refusal (#1957).
- `lattice chat -p/--prompt <text>`: generate once, write the reply to stdout and exit (#1905).
- Routed-completion episodes (`router_episode`) and `POST /v1/router/feedback` (#1949).
- `lattice serve --router-learn`, off by default, with exploration draws seeded from the completion id (`mixture::seed_from_completion_id`); `lattice router init --tau` and `--exploration-scale` (#1958).
- `router-learning` feature, not in the default set, for the bounded router trainer (#1964), and bounded candidate construction behind it (`router_refit`), fitting diagnostics only (#1974).
- Router admission over a registered held-out population (ADR-096 decision 5): admission evidence only, with nothing published or activated (#1975).
- Serialized, create-only router publication with synced staging outside the
  selectable namespace. Bootstrap commits its durable HEAD before returning.
  A failure after HEAD replacement or legacy artifact publication reports an
  uncertain commit for startup recovery, without authorizing live activation.
- Historical format-5 receipts preserve the exact previous HEAD before an
  advance. Explicit format-5 pins require current HEAD or such a receipt (#1970).

### Changed

- Q4 ingress admits a block with a negative scale; a zero or non-finite scale or bias is still refused (#1954).
- `WordPieceTokenizer::tokenize_batch` runs batches of more than 16 texts on the current rayon pool (#1929).
- Gemma 4 CPU prefill projects to vocabulary logits only for the last prompt token (#1965).
- Qwen CPU requests reserve KV cache and attention scratch for their token budget before prefill, and reuse one buffer for text deltas (#1966).
- Standalone Q8 generation keeps its macOS tile buffers across a request (#1982).
- Model downloads resolve at a pinned commit per repository instead of `main` (#1961).
- `lattice` subcommand `--help` output ends with example invocations and each command's platform and checkpoint requirements (#2027).
- **Behavior change:** LoRA adapter load and unload (`/v1/lora/load`, `/v1/lora/unload`) on both
  `lattice serve` and `lattice_serve` now answer 413 `request_body_too_large` for a request body
  over the 1 MiB limit, instead of 400.
- **Behavior change:** under `lattice serve`, a LoRA load whose body cannot be read for any other
  reason now answers 400 `invalid_request` (previously `invalid_request_body`), the same code
  `lattice_serve` already used (#1985).
- Format-5 pins require the canonical counter plus 16 hexadecimal characters,
  or the full 64-character hash. Historical artifacts without commit proof
  refuse with their identity and fresh-directory bootstrap migration guidance.
- Headless format-5 and mixed directories refuse startup selection. Pure
  format-4 directories retain highest-counter selection; malformed or unknown
  lower manifests now also refuse because they cannot establish legacy format.
- Legacy publication refuses existing gate or manifest paths across both
  formats, including orphan gates. Bootstrap refuses all prior state remnants.
- Router readers use a shared persistent lock when available. Older read-only
  directories remain readable without creating a lock; cooperating publishers
  create the lock before publication and never replace or remove its inode.
- Directory initialization synchronizes the target and newly created ancestry
  through its first existing parent, without opening unrelated higher ancestors (#1970).

### Fixed

- Decoder sessions always reach `finish` after a cancelled or failed prefill, and `finish(Reusable)` invalidates the live prediction on the Qwen and Gemma CPU sessions (#1902).
- EmbeddingGemma 2 Metal shared-buffer copies refuse a count past the allocation in every build profile (#1942).
- Measurement lock files have their times refreshed on acquire, so macOS no longer deletes a held lock file (#1903).
- Downloaded models create directories `0755` and files `0644`, so a group-writable umask no longer makes them unloadable (#1951).
- A Qwen `config.json` with a negative nested `rope_parameters.rope_theta` is refused (#1993).
- A worker panic during a CPU streaming completion ends the stream with its error event and `[DONE]` (#1990).
- Batch admission at full capacity keeps FIFO order instead of treating a queued request as admitted (#1991).
- `rustls` 0.23.46 in `Cargo.lock` for RUSTSEC-2026-0346 to 0349 (#1950).

### Deprecated

- `write_artifact` accepts only format 4; use `write_legacy_artifact` or
  `router_init::init_directory` for a fresh format-5 bootstrap.
- `commit_head` refuses caller-supplied publication authority; use
  `router_init::init_directory` for a fresh bootstrap. Learned admission has no
  production publication entry point yet.
- `load_latest` now delegates to `load_selected`, honoring committed HEAD and
  the format rules above. Use `read_artifact` for raw artifact inspection (#1970).

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
