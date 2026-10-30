> lattice lora is better
>
> — the maintainer, choosing a dedicated crate over spreading LoRA across the existing ones
>
> Scope accepted with that answer: the descriptor and its validation leave `lattice-fann`, the hook trait and CPU apply leave `lattice-compute`, the adapter format and PEFT loading come here; `lattice-tune` keeps its training and `lattice-serve` its HTTP and residency, and Metal apply moves with the engine.

# ADR-105: lattice-lora, One Home for Adapter Format and Execution Contracts

**Status**: Proposed
**Date**: 2026-10-28
**Crates**: lattice-lora (new), lattice-fann, lattice-compute, lattice-tune, lattice-model-qwen35, lattice-inference
**Depends on**: ADR-008, ADR-031, ADR-045, ADR-057, ADR-091, ADR-102, ADR-103

## Context

LoRA code lives in six places today:

- `lattice-fann::lora`: `LoraDescriptor`, target-module validation, scale and blend-cap checks.
- `lattice-compute::lora_hook`: the `LoraHook` trait, projection shapes, the target-geometry trait, row application and `NoopLoraHook`.
- `lattice-tune::lora`: the adapter type, CPU apply, blend, the safetensors reader and writer, the manifest and governed admission, and online adaptation.
- `lattice-model-qwen35`: a second safetensors reader (`lora_file.rs`) and the host-side blend for the Metal path.
- `lattice-quant`: QuaRot adapter rotation.
- Model crates: per-model geometry and the injection points.

The two safetensors readers disagree on the following:

|                                   | `lattice-model-qwen35` reader                                                           | `lattice-tune` reader         |
| --------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------- |
| Key acceptance                    | `base_model.model.model.layers.` and `model.layers.`, blocks `self_attn` and `mlp` only | any key containing `.layers.` |
| Unknown module under `mlp`        | skipped                                                                                 | accepted                      |
| Duplicate normalized keys         | last one wins                                                                           | last one wins                 |
| Non-numeric `alpha`               | falls back to rank                                                                      | refused                       |
| Non-finite `alpha` (`NaN`, `inf`) | accepted                                                                                | refused                       |
| Rank-0 factor pair                | loads                                                                                   | loads                         |
| Header read                       | unbounded                                                                               | bounded                       |

The Qwen reader matches only the `base_model.model.model.layers.` and `model.layers.` prefixes, so it misses a text decoder stored under `language_model` (Qwen3.5 checkpoints name their layers `model.language_model.layers.<i>`); the tune reader finds `.layers.` anywhere in the key and accepts that form. The Qwen reader also does not accept the `linear_attn` block where Qwen3.5 keeps its GatedDeltaNet projections.

Neither reader takes `lora_alpha` from a PEFT `adapter_config.json`, and PEFT writes alpha there rather than into the safetensors header. The consequence was measured on Gemma 4 E2B with PEFT 0.21.2, on a branch that adds CPU LoRA attachment to that model (not yet on main), using a rank-1 adapter with `lora_alpha=0.5` on one projection:

- The raw PEFT file loads at alpha = rank = 1, twice PEFT's scale.
- With the header set to `alpha=0.5`, the same tensors match the HF+PEFT logit delta (cosine 1.000000, max difference 1.2e-4).

## Decision

### D1: one crate, bounded scope

`lattice-lora` (`crates/lora`) owns:

- the descriptor and its checks;
- the shared host factor records;
- the scalar CPU apply;
- host blend arithmetic;
- the hook and row contract;
- PEFT/MLX format reading and writing;
- the mechanics of governed admission.

Every other LoRA concern stays with its current owner:

- model geometry and injection stay in the model crates;
- Metal buffers, upload, encoding and apply stay in `lattice-model-qwen35`, until a second Metal model consumes them (ADR-102 Amendment 1, item 4);
- QuaRot rotation stays in `lattice-quant`;
- routing, residency and HTTP stay in `lattice-router` and `lattice-serve`;
- training, experiments, online adaptation and the registry stay in `lattice-tune`.

`lattice-lora` depends on no model crate and on none of `lattice-compute`, `lattice-fann`, `lattice-tune`, `lattice-serve`, `lattice-router` or `lattice-arch`. With no features enabled it has no dependency at all: today's `LoraDescriptor` derives only `Debug`, `Clone` and `PartialEq` (`crates/fann/src/lora.rs:28`). Optional dependencies are feature-gated:

- `safetensors`: format reading and writing. Reading a PEFT `adapter_config.json` sidecar also needs JSON and PEFT's `target_modules` patterns, so this feature enables `serde_json` and `regex`.
- `serde`: the manifest types, with `serde_json` for their JSON form.
- Admission mechanics need both features, as tune's `loader` does today (`all(feature = "safetensors", feature = "serde")`).

The admission integrity check takes the expected digest and a digest of the bytes from its caller. `lattice-lora` does not depend on a hash crate, and tune keeps `sha2`.

### D2: fann depends on lora; lora is the leaf

`LoraDescriptor` and its checks move to `lattice-lora`. `lattice-fann` re-exports them at their published paths, so `lattice_fann::LoraDescriptor` and `lattice_fann::lora::*` keep compiling and name the same types.

`lattice-fann` therefore stops being a leaf. Its only new edge is to `lattice-lora`, which itself has none, so no cycle is possible and fann's build stays small. In the same PR:

- the crate-ownership notes list `lora`, `transport`, `simd` and `inference-core` as the unconditional leaves, with `fann` depending only on `lora`;
- `scripts/publish.sh` publishes `lattice-lora` in the leaf tier and `lattice-fann` after the first indexing wait, because `cargo publish --dry-run` cannot package a crate whose path dependency is not yet on the registry;
- `tests/test_publish_script.py` and the release checklists follow the new order.

### D3: the hook trait moves after the 0.16.0 removal of its Qwen-typed validator

`LoraHook::validate_against(&Qwen35Config)` is the only item in `lattice-compute::lora_hook` that names a `lattice-arch` type (`crates/compute/src/lora_hook.rs:9`). `validate_against_bert` takes three `usize` values and needs nothing from arch. Both are deprecated since 0.15.0, and ADR-103 Amendment 21 item 4 removes them in 0.16.0.

The items in that module that do not name `LoraHook` move in 0.15.x, with re-exports from `lattice-compute::lora_hook` and `lattice-inference::lora_hook`:

- `LoraProjectionShape`;
- the `LoraTargetGeometry` trait;
- `GDN_LORA_MODULES`, `GDN_LORA_MODULES_SERVABLE` and `gdn_lora_module_is_servable`.

`apply_lora_rows` takes a `&dyn LoraHook`, so it moves with the trait. `LoraHook`, `NoopLoraHook` and `apply_lora_rows` move to `lattice-lora` in the 0.16.0 cycle, in the same release as the validator removal. `lattice-compute` and `lattice-inference` then re-export exactly `lattice_lora::LoraHook`, so the trait keeps one identity. Moving it earlier would need `lattice-lora` to depend on `lattice-arch` for a method that is being deleted. `LoraHook::validate_geometry` names `LoraTargetGeometry`, which is already in `lattice-lora` by then.

The model crates keep their geometry implementations and their validate-before-attach order. A failed install leaves the previous hook installed.

### D4: tune keeps its facade types and the training lane

`lattice-tune` keeps the only definitions of `LoraConfig`, `LoraAdapter`, `LoraManifest` and `LoadedAdapter`. Their inherent methods return `TuneError`, and moving the types would force either an upward dependency or a signature change. Each keeps its public fields and its methods, and delegates to free functions in `lattice-lora` over the shared descriptor and factor records:

- construction;
- application;
- blend;
- format I/O;
- manifest parsing.

`LoraLayer`, `AdapterGovernance`, `AdapterStatus`, `ManifestEntry`, `AdapterId` and `RunningRevisions` move to `lattice-lora` where their signatures allow, and tune re-exports them. `load_adapters_from_manifest` stays in tune. It returns `Vec<LoadedAdapter>` and takes a Qwen config under `inference-hook`, so it becomes a wrapper over a `lattice-lora` admission helper that works on manifest entries and a `&dyn LoraTargetGeometry` and returns factor records. `LoraAdapter::layers_mut` stays private, so `online::adapt_step` is unchanged. No type is defined twice.

### D5: one parser, strict by default

One parser reads PEFT and MLX safetensors adapters. It:

- matches complete factor keys (`.lora_A[.weight]`, `.lora_B[.weight]`, and the lowercase MLX forms `.lora_a[.weight]`, `.lora_b[.weight]`) under an explicit namespace table, never "any key containing `.layers.`". The table covers the text decoder of both checkpoint layouts, `…model.layers.<i>` and `…model.language_model.layers.<i>`, for the PEFT and MLX prefixes, and the blocks `self_attn`, `linear_attn` and `mlp`. Each entry is pinned by a fixture taken from a real exported adapter before the hardening PR merges;
- skips a module name it does not recognise inside a recognised block, and lists it in the load report (tune: `LoraAdapter::skipped_factor_keys`; the Metal loader prints it). The Qwen reader skips these today, and two `chat_metal` tests depend on that;
- refuses tensors outside the selected text namespace, such as vision and audio towers;
- refuses duplicate keys after normalization, including identical bytes;
- refuses a pair with mixed dialects, and any unknown suffix;
- checks pair completeness, shapes, rank agreement and finiteness before any factor is exposed. A rank-0 pair is not a parse error. Descriptor validation stays with the caller. Neither reader refuses rank 0 today (tune's descriptor check covers only alpha and scale finiteness), and both keep that behaviour until a separate decision;
- reads the header under an explicit size cap and checks the decoded size before allocating.

Alpha is resolved as follows:

- For a path-based import, a bounded adjacent `adapter_config.json` is read when present. A missing file is different from an unreadable or malformed one, and only a missing file permits fallback.
- PEFT `lora_alpha` from that file is authoritative. If the header also carries `alpha`, the two must be equal, or the load fails and names both sources. `r` must equal the tensor rank. `target_modules` must cover every target present, with PEFT's matching: a list entry matches a module path equal to it or ending in `.` plus it, and a string is a pattern that must match the whole path. `exclude_modules`, when present, must match no target present.
- `use_rslora=true`, a non-empty `rank_pattern` or `alpha_pattern`, and DoRA are refused as unsupported, as are trained biases, `modules_to_save`, trainable tokens, `"all-linear"` and a pattern the reader cannot evaluate. They are never approximated.
- MLX config semantics are not assumed. An MLX file uses header alpha, an explicitly supplied alpha, or the rank fallback.
- When no source declares alpha, alpha = rank, and the load report records that the rank default was used (tune: `LoraAdapter::alpha_rank_default`; the Metal loader prints the alpha source).

Governed admission reads only the bytes it hashes. It never reads a sidecar file. A raw PEFT bundle is converted offline, writing the resolved alpha into the header, and the converted file is what gets approved and hashed.

Per caller, these are behavior changes:

| Behavior         | `lattice-model-qwen35`  | `lattice-tune`          |
| ---------------- | ----------------------- | ----------------------- |
| Key matching     | tightened               | tightened               |
| Duplicate keys   | refused (was last-wins) | refused (was last-wins) |
| Sidecar alpha    | read                    | read                    |
| Non-finite alpha | refused (was accepted)  | unchanged               |
| Header size      | capped                  | unchanged               |
| `use_rslora`     | refused                 | refused                 |

They ship in their own PRs, with release notes, after the mechanical moves.

### D6: blend scale is resolved once

One blend kernel computes, for each logical projection:

- `A* = vconcat(A_e)`;
- `B* = hconcat(c_e · s_e · B_e)`;

where `c_e` is the mixture coefficient and `s_e` the adapter's validated scale. The result is applied at unit scale.

The new API carries this in its types:

- a raw adapter view holds its source scale;
- a resolved contribution is built once from a raw view and a mixture weight;
- the kernel accepts only resolved contributions;
- its result exposes no conversion back to a raw view.

The existing wrappers keep their contracts. Tune's blend folds `config.scale()` exactly once and has no placeholder pre-filter: a positive-rank layer with empty `a` and `b` is refused by the buffer-length check, and a rank-0 layer contributes nothing. The Qwen blend passes every layer to the shared planner, which does not refuse rank 0; Qwen refuses rank 0 later, when it loads the blended adapter. The Qwen Metal blend takes weights that are already effective, through a narrowly named legacy bridge. Routing keeps every choice about coefficient normalization and floors (ADR-091).

### D7: errors

New APIs return `LoraError`, whose variants cover I/O, format, validation or conflict, and resource or unsupported input. Existing entry points keep their published error types (`TuneError`, `InferenceError`, `String`, boxed errors) and map at the boundary, keeping current error categories and serve's HTTP status codes.

## Order of work

Each step is one PR that compiles alone. Every PR in an engine crate carries a bench-compare disposition. The rows marked A/B change build inputs that measured targets compile, including `router_online` in `lattice-fann` and `bench_lora_mixture` in `lattice-inference`.

| Step        | Change                                                                                                                                                                                                                                                                                                  | Disposition                            |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| 1           | Create `lattice-lora`; move the descriptor and checks from fann; fann re-exports; publish order, its test and the release checklists                                                                                                                                                                    | A/B (`router_online`, mixture targets) |
| 2           | Move the host factor records, scalar apply, projection shape, target-geometry trait and GDN module constants; compute, tune and Qwen re-export or delegate; serve and router import from `lattice-lora`; update `crates/inference/tests/data/pipeline_boundary_baseline.txt` for every moved definition | A/B                                    |
| 3           | Move tune's codec into neutral functions, keeping current acceptance behavior                                                                                                                                                                                                                           | A/B on reachable load paths            |
| 4           | Parser hardening (D5 keys, duplicates, dialects, caps)                                                                                                                                                                                                                                                  | A/B and fixture matrix                 |
| 5           | PEFT sidecar alpha and the unsupported-mode refusals, with an HF+PEFT differential re-run (on Qwen3.5, or on Gemma 4 once its CPU LoRA attachment is on main)                                                                                                                                           | A/B and differential                   |
| 6           | Qwen reader cutover: remove `lora_file.rs`'s parser                                                                                                                                                                                                                                                     | A/B                                    |
| 7           | Shared host blend kernel and legacy wrappers; `pipeline_boundary_baseline.txt` updated if `blend_lora_layer_data` moves                                                                                                                                                                                 | A/B with a direct sum-of-deltas oracle |
| 8           | Typed blend API (D6)                                                                                                                                                                                                                                                                                    | A/B or a proven structural waiver      |
| 9           | Manifest and admission mechanics                                                                                                                                                                                                                                                                        | per reachability                       |
| 10 (0.16.0) | Move `LoraHook`, `NoopLoraHook` and `apply_lora_rows` after the validator removal                                                                                                                                                                                                                       | A/B                                    |

## Consequences

- One parser and one scale resolver replace two of each. The measured PEFT alpha defect is fixed in step 5.
- `lattice-fann` gains one internal dependency, on a crate with none, and stops being a leaf. The crate-ownership notes and the publish order change: `lattice-lora` publishes with the leaves, and `lattice-fann` after the first indexing wait.
- Inputs that the old parsers accepted ambiguously are now refused, and the release notes list them.
- `lattice-tune`'s public types and training code are unchanged apart from delegation.
- The hook trait's move waits for 0.16.0, so until then `lattice-compute` still defines `LoraHook` and `NoopLoraHook`. The geometry types and GDN constants move in 0.15.x; `apply_lora_rows` moves with the trait.

### Supersedes, narrowly

- ADR-008, ADR-031 and ADR-057: the canonical owner of the descriptor, format and math changes. Model injection and geometry stay where those ADRs put them. ADR-031's rank fallback no longer stands in for PEFT sidecar alpha.
- ADR-103 Amendments 8 and 10, in two releases. From 0.15.x, compute forwards the projection shapes, the target-geometry trait and the GDN constants from `lattice-lora` instead of defining them. From 0.16.0, it also forwards `LoraHook`, `NoopLoraHook` and `apply_lora_rows`.
- ADR-103 Amendment 21, item 4: unchanged. This ADR depends on its 0.16.0 removal.
- ADR-045 and ADR-091: portable admission and blend arithmetic move. QuaRot rotation, routing policy and GPU application stay.

## References

- ADR-008, ADR-031, ADR-045, ADR-057, ADR-091, ADR-102 (Amendment 1), ADR-103 (Amendments 8, 10, 21)
- `crates/fann/src/lora.rs`, `crates/compute/src/lora_hook.rs`, `crates/tune/src/lora/`, `crates/models/qwen35/src/lora_file.rs`, `crates/quant/src/quarot/lora.rs`
