# ADR-043: LoRA Serving Verification and Qwen3.5-0.8B Support

**Status**: Accepted\
**Date**: 2026-05-14\
**Crate**: `lattice-inference`, `lattice-tune`\
**Extends**: ADR-008 (LoRA Injection via Trait Hook)

---

## Context

ADR-008 established the `LoraHook` trait in `lattice-inference` and the `LoraAdapter` implementation in `lattice-tune`. The full serving pipeline — train in Python/PEFT, export safetensors, load in Rust, apply during inference — was wired across 12 projection sites per layer (4 GQA + 5 GDN + 3 MLP) but had **zero CI test coverage** for the inference-side hook invocation. The `tune` crate tested its own loader and `apply_lora()` math, but nothing verified that the forward pass actually calls `hook.apply()` at every adapted projection, with correct buffer shapes, or that an adapter changes the model's output.

Separately, the only supported generative model was Qwen3.5-2B (~5GB). Qwen3.5-0.8B (~1.6GB, same hybrid GDN/GQA architecture scaled down) was released but had no config preset, making it unusable without manual configuration.

---

## Decision

### 1. Internal spy/delta hook tests in `lattice-inference`

Add LoRA serving integration tests **inside** the `inference` crate (`src/model/qwen35/tests.rs`), not as an external integration test in `tune`.

**Why internal**: `Qwen35Model` has no public constructor — it can only be built via `from_safetensors()` or the test helper `build_model()` that constructs synthetic random weights. An external test in `tune` would need to either make the constructor public (leaking internals) or download real weights (non-hermetic CI). Internal tests can use the same `build_model()` pattern already established by `batch_prefill.rs` tests.

**Why not use `LoraAdapter` from `tune`**: The dependency arrow is `tune → inference`. Using `LoraAdapter` in an `inference` test would create a circular dependency. Instead, the tests implement the `LoraHook` trait directly with two minimal structs:

- **`SpyHook`**: records every `apply()` call as `(layer_idx, module, x_len, output_len)` into an `Arc<Mutex<CallLog>>`. Asserts the set of `(layer, module)` pairs matches the config-derived expected set, and that buffer dimensions satisfy the trait contract.
- **`DeltaHook`**: adds a fixed delta to output at one specific `(layer, module)` target. Used to prove: (a) an adapter changes logits vs. a noop baseline, (b) two noop runs are bit-identical, (c) a non-existent target key is a no-op.

### 2. Qwen3.5-0.8B config preset and fixture

Add `qwen35_0_8b()` to `Qwen35Config` alongside `qwen35_2b()`. Include the real HuggingFace `config.json` as a test fixture (`tests/fixtures/qwen35_0_8b_config.json`) with a parse test asserting `from_config_json` produces the correct config.

Notable 0.8B specifics vs. 2B:
- `mtp_num_hidden_layers: 1` (2B has 0) — MTP weights present in safetensors but ignored by the loader since the main loop only iterates `num_hidden_layers`
- `rope_theta` and `partial_rotary_factor` are nested under `rope_parameters` in the config.json; `from_config_json` still produces correct values because the container-level `#[serde(default)]` backfills from `Qwen35Config::default()` which happens to match

### 3. Default model changed to 0.8B

Both `qwen35_generate` and `generate_lora` binaries now default to `qwen3.5-0.8b` instead of `qwen3.5-2b`, and accept `--model-dir` (absolute path) or `--model` (name resolved under `$LATTICE_MODEL_CACHE` or `~/.lattice/models/`).

---

## Key Design Choices

1. **Spy hook over mock framework**: Two 15-line structs vs. a mock dependency. The call log is the test's primary assertion target — a mock framework would add indirection without value.

2. **`type CallLog` alias**: Required to satisfy clippy's `type_complexity` lint on `Arc<Mutex<Vec<(usize, String, usize, usize)>>>`.

3. **Fixture is the real config.json**: Downloaded verbatim from HuggingFace, not hand-written. This catches real-world serde surprises (nested `rope_parameters`, `mamba_ssm_dtype` field, vision config alongside text config).

4. **Batch prefill LoRA gap documented, not fixed**: `batch_prefill.rs` has no `lora.apply()` calls. The live `generate()` path is unaffected (it prefills via `forward_step` which does apply LoRA), but the batched path's own docs suggest it as a "one-line swap" — that swap would silently drop adapters. A doc comment flags this rather than wiring the hook through, since the batch path is not used in production and the wiring is non-trivial (the batched kernel fuses projections differently).

---

## Alternatives Considered

| Alternative | Pros | Cons | Why Not |
|---|---|---|---|
| E2e test with real weights in CI | Tests the actual model | ~1.6GB download in CI; non-hermetic; slow | CI should be fast and hermetic; e2e verified manually |
| External integration test in `tests/` | Standard Rust test location | `Qwen35Model` has no public constructor; would need `tune` dep (circular) | Dependency direction forbids it |
| Test only in `tune` crate | Where `LoraAdapter` lives | Only tests `tune`'s math, not that `inference` actually calls the hook | Misses the integration gap that was the actual risk |
| `#[cfg(test)]` public constructor | Enables external tests | Leaks test-only API surface; `#[cfg(test)]` on a lib item is fragile | Internal tests avoid the need entirely |

---

## Consequences

**Positive**:
- CI now catches regressions in LoRA hook invocation (new projection sites forgotten, wrong buffer sizes, broken module name strings).
- 0.8B model is usable out of the box — 3x smaller download, same architecture, adequate for development and adapter testing.
- The spy/delta pattern is reusable for future hook traits.

**Negative**:
- The `serde(default)` coincidence for `rope_theta`/`partial_rotary_factor` is fragile — if `Qwen35Config::default()` changes to a different model's values, the 0.8B fixture parse test will catch it, but silently loading wrong rope parameters in production (without the fixture test) would produce degraded output, not an error.
- The synthetic-weight tests do not verify numerical correctness of LoRA + real model weights (only that the pipeline runs and the hook fires). Real-weight verification is manual.

---

## References

- ADR-008 — LoRA Injection via Trait Hook (the `LoraHook` trait design)
- `src/model/qwen35/tests.rs` — spy/delta hook tests
- `src/model/qwen35_config.rs` — `qwen35_0_8b()` preset
- `tests/fixtures/qwen35_0_8b_config.json` — real 0.8B config fixture
- `src/forward/batch_prefill.rs` — LoRA gap documentation
- Hu et al. 2021 — "LoRA: Low-Rank Adaptation of Large Language Models" — https://arxiv.org/abs/2106.09685
