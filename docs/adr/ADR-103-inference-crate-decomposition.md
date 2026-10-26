# ADR-103: Splitting lattice-inference into focused crates behind a facade

**Status**: Accepted (2026-10-19)\
**Date**: 2026-10-19\
**Origin**: roadmap item "Inference crate decomposition" (docs/ROADMAP.md), 2026-10-18\
**Crate**: lattice-inference and the crates extracted from it; workspace manifests, publish script, CI\
**Related**: ADR-102 (in-crate reorganisation and the 3000-line file cap; complementary, see "Relation to ADR-102")

## Amendment 1 (2026-10-19)

A review of the accepted text found two contradictions that the first extraction would hit, and several statements
that did not match the repository. Changed:

1. D4: "Public" no longer applies to members of a type the facade re-exports, because making one public adds it to
   the facade and breaks D6.2's identical listing. Such members take "Internal", and the example moved there. Items
   already public through the facade keep their path through a public export of the new crate, never `__internal`.
2. D6.2 now states how an extraction change clears the required semver job: it merges only while `main` carries an
   unreleased minor version, where the job runs zero checks and says so, and its own listing instruments are the
   surface gate. The Consequences line that said the job "stops seeing" moved items is corrected.
3. D5's reason for removing consumer-feature gates, D6.3 (the CI rustdoc job does not block, so the change runs the
   command itself), D6.4 and D7 (the rules they point at are in `CLAUDE.md`, not a contribution guide), D6.5 (bench
   and binary lists that name a target's crate), D8.3 (name ownership re-checked before the first publish), the
   Context counting method, the Context statement that `bytes_to_unicode` is public (its facade re-export is
   `cfg(test)` only), and the compile-scope consequence.

## Amendment 2 (2026-10-22)

The grammar extraction (lattice-grammar) met a case D5 did not cover: a consumer-feature gate on a public item that
the facade re-exports. Changed:

1. D5: such a gate is kept, as a feature of the new crate with the same name and meaning, forwarded by the facade's
   feature of that name. Removing it would compile the item unconditionally and add it to the facade's default public
   listing, which breaks D6.2's identical listing. First case: lattice-grammar `bench-internals` (`BfsDedupProfile`,
   `profile_reachable_states`, and a test module gated `all(test, feature = "bench-internals")`, now run by its own
   Linux CI step). The removal rule still applies to gates on items that are not on the facade's public listing.

## Amendment 3 (2026-10-24)

The first case of D9 step 3 moves the shared Metal runtime module (`forward/metal_runtime.rs`) into its own crate.
Recorded:

1. D8.1: the crate is `lattice-metal-runtime`. It depends on `lattice-inference-core` for `InferenceError` and, on
   macOS under its `metal-gpu` feature, on `metal`, `objc` and `bytemuck`; its whole library compiles only on macOS
   with that feature. The facade depends on it only under its own `metal-gpu`, which forwards the feature. It publishes
   in the second tier, beside `lattice-tokenizer`, `lattice-grammar` and `lattice-weights`.
2. D4: the module's items take disposition 2 (Public). They are the crate's whole job, and the facade imports them
   `pub(crate)`, so its public listing is unchanged. One item needed more than wider visibility. `scalar` copies the
   bytes of its argument into a Metal argument table. While it was crate-private, every caller was in view; a public
   safe function cannot rely on that. Its bound narrows from `Copy` to `bytemuck::NoUninit`, which excludes types with
   padding or other uninitialized bytes, and a compile-time assertion limits the type to 1 to 4095 bytes, inside the
   4 KB that Metal's `setBytes` accepts. This is the move's one signature change, and D6.6 admits it because
   disposition 2 requires it. The two model parameter structs passed to `scalar` gain the derive.
3. The GPU test lock stays in `lattice-inference` (`measurement::gpu_test_lock`). The measurement harness admits
   `lattice-inference` targets only and the lock-coverage contract lists entry points inside `lattice-inference`, so
   moving the lock would be a harness change, which D6.5 keeps out of an extraction. The module's device test
   therefore moves to `lattice-inference`'s integration tests (`tests/metal_runtime_buffers.rs`), takes the same lock
   there, and is listed as a guarded entry point in the lock-coverage contract. The new crate keeps the module's pure
   unit test and has no device test of its own.

## Amendment 4 (2026-10-24)

The maintainer, 2026-10-24: "maybe we should finish a bit more splitting first, I really want the serve to be out
first, that's not needed in ruvector neither". Read against D9, that moves step 4 (serving) ahead of step 3's remaining
per-model work. Serving is an
application, not a library layer: it calls nearly every model, so it cannot sit below the facade without a cycle. This
amendment records how it leaves. Evidence: a check-only probe at f6a9b643 (serving modules, `model::serving_runtime` and
both binaries moved into a new crate) compiled the facade with no errors under its default features minus `serve`, and
under `std` alone; every error was in the new crate or in tests importing it. The probe did not compile Metal and ran no
tests, so the ledgers below also carry crossings found by source reading at f6a9b643, and item 7 makes the compile and
test arms the proof.

The work lands as two pull requests in order: the harness preparation of item 4, then the extraction.

1. **D7 and D3.1, application crate.** `lattice-serve` is an application crate above the facade. It may depend on
   `lattice-inference` and `lattice-weights`. It is the only such crate. The `cargo metadata` check names it and fails if
   any workspace crate depends on it through a normal, build or dev dependency, and in particular if `lattice-inference`
   or an extracted crate does (a test-only edge included). Every rule D7 and D3.1 set for extracted library crates still
   applies to them; this exception covers `lattice-serve` alone.

2. **D1 and D6.2, what is retired.** Three populations, each listed:
   - Public paths retired from the facade and available from `lattice-serve`: `serve`, and the seven public
     `#[doc(hidden)]` roots `serving_cpu`, `serving_cpu_host`, `serving_embeddings`, `serving_factory`, `serving_host`,
     `serving_provider`, `serving_worker`. Hidden does not mean uncallable, so these count as retired paths.
   - Physical moves with no external path to retire: the private `serving_preparation`, the crate-private
     `serving_runtime_contract` and `model::serving_runtime`.
   - Feature-conditioned public items that change: `router_admission::{ReferenceAdapter, score_metal_reference}` (exported
     under router-learning, serve, f16, macOS and metal-gpu) stay in the facade under `serving-support` in place of
     `serve`, and `score_metal_reference` takes the embedder through an inference-owned trait instead of serving's
     concrete types (item 3). That signature change is listed in the release notes with the path map.
   - Additions, all hidden and all outside the default listing: the `serving-support` ledger of item 3, the
     `router_admission::ReferenceEmbedder` trait with its methods, and `tiny_zero_model_with_vocab` under `test-utils`.
     Nothing else is added under any feature.
     The facade's `serve` feature and its nine optional dependencies are removed. The `lattice` and `lattice_serve`
     binaries move; installation becomes `cargo install lattice-serve`, with `--features f16,metal-gpu` for the Metal
     binary as its old required-features demanded. This is the one D1 exception in this ADR and it ships in an unreleased
     minor version. D6.2's test naming each moved item by its old and new path cannot exist for retired paths without an
     inference to serve edge, so for the retired populations it is replaced by migration compile probes in `lattice-serve`
     naming each item at its new path. The identity requirement stands for every shared type that does not move.
     Maintained descriptions and the rustdoc link to `serve` (lib.rs:123) are updated.

3. **D3, D4 and D5, the seams.** Each crossing is resolved as stated; nothing else may be widened.
   - **`serving-support`.** One named facade module, `#[doc(hidden)]`, behind an off-by-default feature of the same name
     that `lattice-serve` enables. It is a D4 exception: D4 places such wrappers in the extracted crate, and here they sit
     in the facade because the extracted crate is the caller. It is an unsupported, version-coupled production interface:
     free functions and minimal support records only, no exported inherent methods, no public engine fields, no setters,
     no test-utils requirement on any production item. `lattice-serve` depends on `lattice-inference` with an exact
     version requirement (`=x.y.z`), and any change to this module releases both crates together. Closed item ledger
     (each a free wrapper over the existing crate-private item):
     - traced generation: Qwen3.5 and Gemma CPU `generate_with_trace`, `generate_streaming_with_trace`,
       `generate_streaming_via_driver`, and Metal `generate_streaming_with_prefix_cache_with_trace`; the trace is read
       through an observation record carrying `opened` and `consumed`, built inside the facade from `DriverTrace`;
     - engine configuration observation for the Metal worker (the values `qwen_metal.rs` reads from
       `state.engine.config`, returned by value);
     - chat formatting: `format_chat_template_parts`, and the role rendering plus the three ChatML delimiters used by
       the Qwen vision provider (`push_chat_turn_open`, `push_chat_turn_close`, `push_chat_generation_open`);
     - `resolve_stop_token_ids`;
     - cancellation-aware vision: `load_qwen35_vision_weights_with_cancel`, `qwen35_vit_forward_metal_with_cancel` with
       its output (hidden states plus the `metal_dispatches` and `gemm_calls` evidence fields, whose cfg moves from
       `serve` to `serving-support`), and `qwen35_merger_forward_with_cancel`;
     - router: `refit_snapshot`, `effective_reasoning_budget`, and `refit_source_from_serving`, a free function over
       `RefitSource::from_serving`, which stays an inherent method of its owner, regated from `serve` to
       `serving-support`.
       The ledger is closed: an item not on it stays crate-private, and growing it is an amendment.
   - **Router ownership.** `RefitSource` construction and its invariants stay in inference. The Metal reference scorer
     stays in inference too, because it builds `ReferenceEvidence` and reads private Metal and candidate state. Its five
     serving imports (`EmbeddingModel`, `QWEN35_F16_DECODER_LOADER`, `EmbedderIdentity`, `PromptSource`,
     `trained_pooling`) are replaced by one inference-owned trait, `router_admission::ReferenceEmbedder`, under exactly
     the cfg `score_metal_reference` carries after this amendment (router-learning, serving-support, f16, macOS,
     metal-gpu), `#[doc(hidden)]`, and counted in the `serving-support` population (item 2). Its contract:
     - `identity(&self) -> &str`, `loader_format(&self) -> &str`, `prompt_source(&self) -> &str` and
       `dimensions(&self) -> usize`, the four values the scorer compares against the registered representation, and
       `max_context(&self) -> usize`, the bound its byte check reads before any tokenizing;
     - an associated `Pooling` type, `parse_pooling(&self, name: &str) -> Result<Self::Pooling, String>`, and
       `embed(&self, text: &str, pooling: &Self::Pooling) -> Result<Vec<f32>, String>`.
       `lattice-serve` implements it on a serve-local adapter that borrows the embedding model together with the
       already-resolved `EmbedderIdentity` the scorer receives today, returns `QWEN35_F16_DECODER_LOADER` and
       `PromptSource::SERVED.as_str()`, parses through `trained_pooling` (still the one parser, still accepting exactly
       `mean_visual` and `last_token`), and embeds through the existing model path. No identity is derived from the model,
       and no second pooling parser exists. Inside the scorer the order and the error classes do not change: the byte and
       context bound, then the four representation comparisons, then `parse_pooling`, all before payload validation and
       router construction. Errors from `parse_pooling` map to `ReferenceError::Invalid`; errors from `embed` map to
       `ReferenceError::Model`. Existing error mappings outside this trait boundary are unchanged. The
       finite-value and dimension checks on each embedding stay where they are. `score_metal_reference` takes the adapter in
       place of the model and identity arguments, which is the signature change item 2 lists.
       The cfg change is transitive. Regating `RefitSource::from_serving` and the scorer from `serve` to
       `serving-support` also regates every crate-private item they call that is gated on `serve` today, for example
       `AdapterRouter::gate_bytes` (`mixture.rs`), which stays crate-private. Those call sites get the replacement gate and
       no export. Unit tests in `router_refit` and `router_admission` that assert owner-private state stay in inference,
       with their serving-router setup replaced by equivalent owner-local setup (a test embedder for the scorer); the
       coverage that genuinely exercises the serving router moves to `lattice-serve` as integration tests over public and
       `serving-support` items. No field is widened, no foreign inherent impl is relocated, and inference gains no
       dependency on serve, dev included.
   - **Inside `lattice-serve`, no facade change.** The bounded config reader comes from `lattice-weights` directly.
     `GenerateConfig` (`#[non_exhaustive]`, all fields public) is built from `Default` and then assigned the same fields
     at every construction site. The one exhaustive `ChatRole` match gains a wildcard arm that rejects with the existing
     `invalid_role` code, after the existing image rejection; coercing an unknown role or panicking is not allowed. This
     is the move's one behaviour addition, and no current variant reaches it.
   - **Test helper.** `tiny_zero_model_with_vocab` is exported through `test-utils`, like the Qwen test support.
   - **Features (D5).** Facade defaults become `std` and `download`; `serving-support` is off by default. `lattice-serve`
     depends on `lattice-inference` with `default-features = false` and `features = ["std", "serving-support"]`, so
     every other facade feature reaches it only through its own feature table. It has default `download` (an
     application, so D5's no-defaults rule for extracted library crates does not apply) and these features:
     `download = ["lattice-inference/download"]`; `f16`, `test-utils` and `bench-internals`, each forwarding the facade
     feature of the same name; `metal-gpu = ["lattice-inference/metal-gpu", "dep:lattice-fann"]`;
     `mixture = ["lattice-inference/mixture", "dep:lattice-fann"]`; and
     `router-learning = ["mixture", "lattice-inference/router-learning"]`. The local
     `mixture` exists because moved code gates on it in its own crate (the `ServingRouter` constructor, `lattice router
     init`, and the episode's `exploration` field, which inference requires whenever its `mixture` is on); forwarding
     `router-learning` alone would leave the application's `mixture` off while inference's is on, which does not
     compile. `lattice-fann` is a direct normal dependency of `lattice-serve` (the moved routing code names
     `lattice_fann::Network` under `mixture`, the moved residency module, gated on `metal-gpu`, names
     `lattice_fann::lora::LoraDescriptor`), optional and enabled by `mixture` and by `metal-gpu`, matching the facade's
     own two routes to it. No dev dependency supplies a production requirement.

4. **D6.5, harness, as a separate prior change.** The measurement inventory's discovery grammar
   (`scripts/bench-measurements.toml`) and its checker (`tests/test_bench_measurement_supervision.py`, which asserts
   discovered paths equal classified paths) name only inference's examples and binaries, and the GPU lock-coverage
   contract (`crates/inference/tests/metal_measurement_lock_contract.rs` with its helpers) scans from inference's own
   manifest. A third, `pipeline_boundary_contract.rs`, takes its production frontend roots from inference's own tree
   (`src/serve/mod.rs`, `src/serving_worker.rs`, the provider files and every `src/bin` target), all of which move. The
   first pull request teaches these three about the `crates/serve` package boundary while changing no classification,
   enforcement, exemption, ratchet entry or GPU-handoff admission set, and proves it with the existing suites plus a
   negative control under `crates/serve` for each. `env_var_contract.rs` is different: its source census already
   covers every `crates/*/src` and will discover `crates/serve` unchanged, and that workspace-wide population, its
   presence rules and its positive controls stay as they are. Only its documentation reader is inference-only (it
   reads `crates/inference/CONFIG.md`). In the extraction, `crates/serve` gets its own `CONFIG.md` for the `LATTICE_*`
   variables its code reads, and the reader checks the union of the two tables against the workspace-wide census: each
   variable is documented in exactly one table (the table of the crate that reads it, or inference's when both do), a
   name in both tables is refused as a duplicate, and the missing and stale checks keep running over the whole
   workspace. A negative control proves a variable read only under `crates/serve` and documented nowhere is refused. The extraction then moves the two serving examples
   (`bench_serve_http`, `bench_serve_prepare`), their target declarations, classifications and lock-contract positions
   together, with the same reviewed population and the same worker-thread lock evidence for `bench_serve_prepare`. The
   GPU test lock stays in `lattice-inference`. Neither example nor binary is admitted to GPU handoff today and the move
   admits nothing.

5. **D8.** `lattice-serve` publishes in its own tier after `lattice-inference`, in the same release whenever
   `serving-support` changed. The name returned 404 on crates.io at 2026-10-24T21:16:11Z beside a 200 control read for
   serde (receipt kept with the probe); D8.3's owner check runs again before the first upload.

6. **D9.** Step 4 runs before the remaining step 3 work, at the maintainer's direction. Expected cost: the traced
   generation seam is cut now and again when the ADR-090 driver work changes it; `serving-support` is where that second
   cut lands, deliberately, as an amendment to its ledger.

7. **Acceptance additions to D6.** D6.1, D6.3 and D6.4 apply unchanged; these cells are added, each with executed tests
   and a test-name union, not only checks:
   1. isolated inference: default, `std` only, and the router-learning, test-utils and bench-internals suites without
      `serving-support`, then the same with it; required-feature skips accounted for by target;
   2. isolated `lattice-serve`: default and `--no-default-features` (inference resolved without `download`), lib and all
      targets, tests under the forwarded CPU feature union, and three router profiles: neither `mixture` nor
      router-learning, `mixture` alone, and router-learning;
   3. macOS on the Metal host: both crates with `f16,metal-gpu` under the same three router profiles plus test-utils
      and bench-internals, building test targets and running the existing device and route suites under the GPU lock.
      Test names are the inventory, not the proof: the vision provider and prefix-cache worker paths, and
      `score_metal_reference` itself, are shown executed. The scorer cell adds a call of `score_metal_reference` through
      the production serve adapter on a real device, and focused contract tests (a recording test embedder is enough for
      these) for each of the identity, loader, prompt-source and width mismatches and for an invalid pooling name, each
      asserting `ReferenceError::Invalid` with no embedding performed;
   4. consumers: `lattice-embed` normal and dev profiles, `lattice-tune` with inference-hook and train-backward, then a
      joint build including `lattice-serve`, with resolved inference feature sets inspected for `serving-support`;
   5. the D6.2 listing diff under default features equals the first population of item 2 and adds nothing; under
      `serving-support`, test-utils and the macOS router features the delta equals exactly item 2's other populations and
      its listed additions; and the bench measurement disposition covers
      the changed build inputs (the manifest and feature changes are build inputs, so no structural waiver applies).

## Amendment 5 (2026-10-25)

The model configurations move below the facade, so that code which needs a model's configuration and nothing else no
longer has to live in `lattice-inference`. The QuaRot conversion modules, for example, read `Qwen35Config` and cannot
join `lattice-quant` while that type is defined in the facade. This is the first per-model step of D9 step 3, and it
moves the configuration parsers only. Recorded:

1. **D8.1.** The crate is `lattice-model-config`. It holds the four configuration modules `qwen35_config`,
   `gemma4_config`, `embeddinggemma2_config` and `bitnet_config`, with their unit tests. It depends on
   `lattice-inference-core` for `InferenceError` and on `lattice-weights` for the bounded `config.json` reader, and it
   publishes in the third tier beside `lattice-quant`. The model code, the loaders, `QwenConfig` (defined inside
   `model/qwen.rs` with the Qwen3 embedding model) and the Qwen3.5 tensor-name functions stay in the facade for later
   changes.
2. **D4.** Crate-private items used from outside the moving modules:
   - Public: `qwen35_config::{MAX_VISION_TENSOR_BYTES, compute_layer_types}`, `gemma4_config::resolve_stop_token_ids`,
     and `embeddinggemma2_config::{ImagePartsConfig, image_parts_config}` with the three fields of `ImagePartsConfig`.
     Each is part of parsing or validating a configuration. The facade imports them `pub(crate)` at their old paths, so
     its public listing is unchanged.
   - Internal: `qwen35_config::checked_double`, which reports an overflow as a Qwen configuration error. Its definition
     stays crate-private, and `__internal::checked_double` forwards to it for the two facade callers.
   - Every other crate-private item of the four modules is used only inside them and stays crate-private.
3. **D6.6.** Not everything in `model::qwen35_config` is configuration. The four deprecated aliases of the generation
   types (`GenerateConfig`, `GenerateOutput`, `TokenLogprob`, `TopLogprob`) name types that stay in the facade, and the
   reasoning-budget helpers `REASONING_CLOSE_GRACE_TOKENS`, `force_close_think` and `decode_cap` are generation code.
   They stay in the facade's `model::qwen35_config` module, unchanged and at the same paths, together with their 15
   unit tests, which keep their names. The rest of the module's tests move with it.
4. **Fixtures.** `tests/fixtures/qwen36_config.json` is read only by the moved tests and moves to
   `crates/model-config/tests/fixtures/`. `qwen35_0_8b_config.json` and `gemma4/e2b_config.json` are also read by tests
   that stay in `lattice-inference`, so they stay where they are, and the moved tests read them through a path relative
   to the workspace, as `lattice-embed`'s tests already read the first of them.

## Amendment 6 (2026-10-25)

The rest of the configuration surface that Amendment 5 left in the facade moves to `lattice-model-config`. QuaRot reads
`QwenConfig` and the Qwen3.5 tensor-name functions as well as `Qwen35Config`, so they have to sit below the facade
before the QuaRot modules can leave it. Recorded:

1. **D8.1.** `QwenConfig` with its `impl` moves out of `model/qwen.rs` into the crate's `qwen_config` module, and
   `qwen_layer_tensor_prefix` and `qwen_required_tensor_names` move out of `model/qwen35/loading.rs` into a
   `qwen35_tensors` module. The facade re-exports them at `model::qwen::QwenConfig`, `model::QwenConfig`, the crate root
   `QwenConfig` and `model::qwen35::{qwen_layer_tensor_prefix, qwen_required_tensor_names}`. The `config.json` parser for
   `QwenConfig` stays in `model/qwen.rs` with the model loader that calls it, and so does `ModelInferenceConfig`, which
   reads `inference_config.json` and derives cache revisions from the checkpoint's files.
2. **D4.** No crate-private item crosses the boundary. Everything moved was already public.
3. **D6.1.** The unit tests of the moved items stay in the facade and reach them through the re-exports, so no test is
   renamed or moved. `facade_api_parity` adds a compile-time check that each re-exported function is the same function
   item as its definition.

## Amendment 7 (2026-10-25)

With the configurations below the facade (Amendments 5 and 6), the QuaRot conversion joins the primitives already in
`lattice-quant`. This completes D9 step 2 for quantization. Recorded:

1. **D8.1.** `quant::quarot::{convert, forward_equivalence, io, lm_head, lora, pipeline, plan, rmsnorm_fusion}` move to
   `lattice-quant` under `quarot::`, with their tests, and so does the test-only `r3_reference` module. The crate gains
   two normal dependencies: `lattice-model-config`, for the Qwen3.5 configuration, `QwenConfig` and the tensor-name
   functions, and `memmap2`, because the source-tensor reader holds a memory map. Maps are still constructed only in
   `lattice_weights::mmap_trust`; holding one is allowed, and the trust-boundary contract already scans this crate.
   `lattice-quant` now depends on `lattice-model-config`, so the publish order puts `lattice-model-config` in its own
   tier before `lattice-quant`, which changes Amendment 5's "third tier beside `lattice-quant`".
2. **D1.** The facade keeps every path. Each moved module becomes a facade module of named re-exports, and
   `quant::quarot` keeps its root re-exports of `io::{ArtifactVersion, OnlineArtifactDescriptor, QuarotTensorReader,
   SourceDType}` and `plan::{OnlineRotationSpec, OnlineTransformSite, RotationId}`.
3. **D4.** One crate-private item crosses the boundary: `convert::read_quarot_seed_from_index`, read by the Metal
   weight loader and its tests. It is not part of any public API, so its definition stays crate-private and
   `__internal::read_quarot_seed_from_index` forwards to it; the facade imports it `pub(crate)` at its old path under
   its caller's gate, `all(target_os = "macos", feature = "metal-gpu")`. The definition was gated on
   `any(test, feature = "metal-gpu")` for the conversion tests that also called it; those tests moved with it and
   `lattice-quant` has no `metal-gpu` feature, so the gate on the definition goes. Every other crate-private item of the
   moved modules is used only inside them.
4. **D6.1.** The moved unit tests keep their names under the new crate's paths. `facade_api_parity` adds same-item
   checks for every moved public type, function and constant, and for the root re-exports.

## Amendment 8 (2026-10-25)

The host-side computation of the decoder forward pass gets its own crate, `lattice-compute`, so that the CPU kernels,
attention, pooling, KV caches and sampling can leave the facade without each one becoming a crate of its own. It is
filled by small moves, each its own change, and the first move is the rotary position embedding tables. Recorded:

1. **D9.** Step 3 widens from "the shared Metal runtime and per-model definitions" to the shared compute, host and
   Metal, and per-model definitions. Host compute is shared by every model's forward pass in the same way the Metal
   runtime is. The ADR-102 condition on step 3 concerns the Metal files and does not apply to host modules that are
   already under the 3000-line cap; `rope.rs` is 86 lines. Step 4 (serving and the binaries, lattice#2227) is not
   blocked by this widening: the first move lands after it.
2. **D8.1.** The crate is `lattice-compute`. Its first content, `rope`, has no dependencies, internal or external, so the
   crate is a leaf and publishes in the first tier beside `lattice-fann`, `lattice-transport` and
   `lattice-inference-core`. `lattice-inference` gains it as a normal dependency. Later moves that add dependencies
   re-derive its tier in their own change. The name was checked free on crates.io in the same pass as a control read of
   a crate known to exist; D8.3's re-check before the first upload still applies.
3. **D1.** The facade keeps `lattice_inference::rope::RopeTable`, as a named re-export of `lattice_compute::rope::RopeTable`.
4. **D4.** No crate-private item crosses the boundary. `rope.rs` imports nothing from the rest of `lattice-inference`,
   and every item it defines was already public.
5. **D6.1.** `rope/tests.rs` moves unchanged with its five tests, which keep their names under the new crate's path.
   `facade_api_parity` adds a check that the facade's `RopeTable` and the extracted one are the same type.
6. **D6.4.** The RoPE tables are timed by `inference_perf`'s decode RoPE group under `bench-internals`, and the change
   adds a crate and a dependency, so it carries a measured A/B and does not claim the structural waiver.
7. **Later moves.** Each later move into lattice-compute is its own PR with its own measured A/B, or a structural
   waiver that names why no default bench target reaches it, and records its D4 and D6 listings in its own amendment. The
   alternative of a separate crate for host primitives is reconsidered when a consumer needs only those primitives and
   the measured cost of the rest of the crate's dependencies is significant to it.

## Amendment 9 (2026-10-25)

KV storage and prefix reuse join `lattice-compute`, following the RoPE tables. The facade keeps its existing paths;
the implementation and its unit tests move together. Recorded:

1. **D8.1.** `kv_cache::{cross_turn, flat, paged, prefix}` move into `lattice-compute`. The crate gains normal
   dependencies on `lattice-inference-core`, `half`, `indexmap` and `rustc-hash`. It therefore moves from publish tier 1
   to tier 2, after `lattice-inference-core`, beside tokenizer, grammar, weights and the Metal runtime. The publish script
   and its dependency-order test record that order.
2. **D1.** The facade's four `kv_cache` modules become named re-exports, preserving their 20 public types and three
   public functions, including the existing `kv_cache` root re-exports. Callers keep the same types at both paths.
3. **D4.3.** One crate-private crossing needs a bridge:
   `lattice_compute::__internal::paged_cache_try_floats_per_page`. It delegates to
   `PagedKVCacheConfig::try_floats_per_page`, which stays crate-private in compute. The facade's existing
   `batch::worker::PagedKVCacheConfigExt::try_floats_per_page_pub` calls the bridge. Widening the inherent method would
   widen the re-exported type's facade API, so the checked geometry stays with its owner behind the hidden bridge;
   the facade does not re-export the bridge. `try_kv_dim` has no cross-crate caller and stays crate-private.
4. **D5.** Compute keeps an empty default feature set; these modules and dependencies are unconditional. The move
   adds no caller feature or target gate.
5. **D6.1.** The four test modules move with 20, 41, 30 and six test functions respectively, retaining all 97 names.
   The flat-cache tests use a local first-index-wins `oracle_top1` instead of reaching upward into speculative
   generation; the other three test files are unchanged. `facade_api_parity` adds
   `prefix_cache_types_are_the_same_type_at_both_paths`, `prefix_cache_functions_are_the_same_items_at_both_paths`,
   `flat_cache_types_are_the_same_type_at_both_paths` and `paged_cache_types_are_the_same_type_at_both_paths`.
6. **D6.4.** The added dependencies and changed build inputs require measured A/B acceptance. This amendment records
   the extraction and its validation obligations; it does not assert a measurement result or a structural waiver.

## Amendment 10 (2026-10-25)

The LoRA hook contract and its row application helper join `lattice-compute`, so shared forward computation can use
them without depending on the inference facade. Recorded:

1. **D8.1.** `lora_hook` and its tests move into `lattice-compute`. The crate gains a normal dependency on
   `lattice-model-config` for `Qwen35Config`, alongside its existing core dependency. Compute therefore moves from
   publish tier 2 to tier 4, after model-config and beside quant; the script publishes quant then compute within that
   tier. This order adds no compute-to-quant dependency. The publish-order test and dependency diagrams follow it.
2. **D1.** `lattice_inference::lora_hook` keeps named public re-exports of `LoraHook`, `NoopLoraHook`,
   `LoraProjectionShape`, `GDN_LORA_MODULES`, `GDN_LORA_MODULES_SERVABLE`, `gdn_lora_module_is_servable` and
   `qwen35_projection_shape`. The types and trait are the same items at both paths.
3. **D4.2.** The formerly crate-private free function `apply_lora_rows` becomes the documented public
   `lattice_compute::lora_hook::apply_lora_rows`: applying an adapter to flattened token rows is a coherent compute
   operation. The facade imports it with `pub(crate) use`, preserving its private facade visibility. No new
   `__internal` bridge is added. The helper's row geometry, activity check and row order stay unchanged.
4. **D5.** Compute keeps an empty default feature set. `lora_hook` and the model-config dependency are unconditional;
   the move adds no caller feature or target gate.
5. **D6.1.** `lora_hook/tests.rs` moves unchanged with all nine test names. The facade adds the unit test
   `row_helper_is_the_same_function_item`. `facade_api_parity` adds
   `lora_types_and_trait_are_the_same_at_both_paths`, `lora_functions_are_the_same_items_at_both_paths` and
   `lora_constants_are_equal_at_both_paths`, covering type and trait identity, function-item identity and constant
   equality without exposing the row helper publicly through the facade.
6. **D6.4.** The added dependency and changed build inputs require measured A/B acceptance. This amendment records
   the extraction and its validation obligations; it does not assert a measurement result or a structural waiver.

## Amendment 11 (2026-10-25)

The numerical diagnostics (`metrics`, `pruning`) and option-letter scoring (`option_scoring`) join `lattice-compute`,
so forward and evaluation code can use them without depending on the inference facade. Recorded:

1. **D8.1.** `metrics`, `pruning` and `option_scoring` move into `lattice-compute` with their tests. `metrics` and
   `pruning` reach only the core crate. `option_scoring` names the `Tokenizer` trait, so compute gains a normal
   dependency on `lattice-tokenizer`. The tokenizer publishes in tier 2, before compute's tier 4, so compute's tier,
   the publish script and the publish-order test are unchanged.
2. **D1.** `lattice_inference::metrics` keeps named public re-exports of `MetricsMode`, `LayerMetrics`,
   `ForwardMetrics`, `OnlineSoftmaxEntropy` and `l2_norm`; `lattice_inference::pruning` of `cosine_similarity`,
   `BlockInfluence`, `BlockInfluenceAccumulator`, `score_from_hidden_states` and `pruning_rank`;
   `lattice_inference::option_scoring` of `OptionScores`, `option_scores_from_logits`, `resolve_option_letters`,
   `resolve_decision_labels`, `LETTER_LABELS` and `DecisionLabelTable`. The items are the same at both paths.
   `MetricsMode` keeps its `#[non_exhaustive]` annotation; no match on it exists outside compute.
3. **D4.2.** The formerly crate-private free function `check_letter_ids` becomes the documented public
   `lattice_compute::option_scoring::check_letter_ids`: validating letter ids against a vocabulary is part of option
   scoring. The facade imports it with `pub(crate) use`, preserving its private facade visibility, and its callers in
   the CPU and Metal scorers keep their crate path. No new `__internal` bridge is added. `metrics` and `pruning` have
   no crate-private items.
4. **D5.** Compute keeps an empty default feature set. The three modules and the tokenizer dependency are
   unconditional; the move adds no caller feature or target gate.
5. **D6.1.** `metrics/tests.rs` (28 tests), `pruning/tests.rs` (25) and `option_scoring/tests.rs` (20) move with all
   names unchanged; only the error and tokenizer import paths change. `facade_api_parity` adds
   `metrics_types_are_the_same_at_both_paths`, `metrics_functions_are_the_same_items_at_both_paths`,
   `pruning_types_are_the_same_at_both_paths`, `pruning_functions_are_the_same_items_at_both_paths`,
   `metrics_mode_variants_resolve_through_the_facade`, `option_scoring_types_are_the_same_at_both_paths`,
   `option_scoring_functions_are_the_same_items_at_both_paths` and `letter_labels_constant_is_equal_at_both_paths`.
6. **D6.4.** The added dependency and changed build inputs require measured A/B acceptance. `metrics_bench` and
   `pruning_bench` stay in `lattice-inference` and reach the moved code through the facade; option scoring has no
   declared bench target. This amendment records the extraction and its validation obligations; it does not assert a
   measurement result or a structural waiver.

## Amendment 12 (2026-10-25)

The Gemma 4 rotary embedding tables join `lattice-compute`, so the Gemma, EmbeddingGemma and ERNIE forward paths can
build them without depending on the inference facade. Recorded:

1. **D8.1.** `gemma4_rope_inv_freq`, `gemma4_rope_cos_sin` and `vision_axial_rope_inv_freq` move from
   `lattice_inference::model::gemma4_ops` into a new `lattice_compute::gemma_rope` module. Their bodies use only the
   standard library, so compute gains no dependency and its publish tier is unchanged. Applying the rotation
   (`gemma4_apply_rope`, `vision_axial_rope_apply`) and the other Gemma 4 operations stay in the facade.
2. **D1.** `lattice_inference::model::gemma4_ops` keeps named public re-exports of the three functions; they are the
   same function items at both paths, and every existing caller keeps its path.
3. **D3.** The one documentation link that pointed back into the facade (`gemma4_apply_rope`) becomes plain
   qualified code text, so compute does not reference `lattice-inference`.
4. **D4 and D5.** The three functions were already public and take no facade-owned type: no crate-private crossing,
   no `__internal` bridge, no feature or target gate. Compute keeps an empty default feature set.
5. **D6.1.** No test moves: the fifteen `gemma4_ops` tests, which check the tables against reference goldens, keep
   running in `lattice-inference` through the re-exported paths. `facade_api_parity` adds
   `gemma_rope_functions_are_the_same_items_at_both_paths`.
6. **D6.4.** `bench_gemma4_driver_pair` reaches `gemma4_rope_cos_sin` on every decode step, and the crate boundary is
   a changed build input, so a paired before/after of that target is required before merge. This amendment records
   the extraction and its validation obligation; it does not assert a measurement result or a structural waiver.

## Amendment 13 (2026-10-25)

The fail-closed softmax row contract joins `lattice-compute`, so the CPU kernels that move next can apply it without
depending on the inference facade. Recorded:

1. **D8.1.** `row_max_and_any_nan`, `row_fails_closed_pre_exp`, `is_masked_neg_inf` and `finalize_row` move from
   `lattice_inference::attention::softmax_row` into a new `lattice_compute::attention::softmax_row` module. They use
   only the standard library, so compute gains no dependency and its publish tier is unchanged. The attention kernels
   that call them (GQA, decode, flash, differential, native sparse) stay in the facade.
2. **D1.** `lattice_inference::attention::softmax_row` keeps named public re-exports of the four functions; they are
   the same function items at both paths, and every existing caller keeps its path.
3. **D3.** The module documentation linked the GQA and decode kernels that stay in the facade; those two links become
   plain qualified code text, so compute does not reference `lattice-inference`.
4. **D4 and D5.** The four functions were already public and keep their `#[inline]` attributes: no crate-private
   crossing, no `__internal` bridge, no feature or target gate. Compute keeps an empty default feature set.
5. **D6.1.** The module's twelve tests move with it, names unchanged. `facade_api_parity` adds
   `softmax_row_functions_are_the_same_items_at_both_paths`.
6. **D6.4.** `elementwise_cpu_bench` reaches the contract through `softmax_attention`, and the crate boundary is a
   changed build input, so a measured A/B of that target is required before merge. This amendment records the
   extraction and its validation obligation; it does not assert a measurement result or a structural waiver.

## Amendment 14 (2026-10-25)

The CPU forward kernels join `lattice-compute`, so the host-side matmul, GEMM, normalization, activation and softmax
code sits below the facade. Recorded:

1. **D8.1.** The `forward::cpu` kernels (activation, the architecture kernels, BERT kernel policy, BLAS-style GEMM,
   elementwise, F16 matmul, GEMM validation, matmul, norm, SIMD detection, softmax and the tiled variants) move from
   `lattice_inference::forward::cpu` into `lattice_compute::forward::cpu` with their tests. The Q8 matrix kernel and
   its tests stay in the facade. Compute gains direct dependencies on `lattice-weights` (the F16 storage type used by
   the F16 matmul) and `rayon`, and the optional `gemm` dependency moves from inference to compute. Weights already
   publishes before compute, so compute's tier, the publish script and the publish-order test are unchanged.
2. **D1.** `lattice_inference::forward::cpu` keeps named public re-exports of every function and type it exported
   before, including the architecture-gated `matmul_neon` (aarch64) and `matmul_avx2`/`matmul_avx512` (x86_64).
   `lattice_inference::forward` replaces its glob re-export of the CPU module with the same named list,
   `lattice_inference::{BertCpuKernelPolicy, BertCpuKernelProfile}` keep their hidden crate-root paths, and
   `lattice_inference::weights::f16_weights::matmul_bt_f16` keeps its path. The items are the same at every path.
3. **D3.** The BERT policy documentation linked the BERT model that stays in the facade; that link becomes plain text.
4. **D4.** `simd_config`, `SimdConfig`, `validate_gemm_nn`, `validate_gemm_bt` and `validate_ternary_matvec_args` were
   crate-private and become public in `lattice_compute::forward::cpu`, because facade code that stays (attention, the
   ternary kernel, the GPU GEMM paths, vision and backward) calls them. The facade imports the four functions with `pub(crate) use`, preserving their
   facade visibility, and does not export `SimdConfig`. The `f16_matmul` module becomes public in compute so the
   facade can keep the `matmul_bt_f16` path. No `__internal` bridge is added. Each of the five is documented as a
   public item of compute, and `SimdConfig` is `#[non_exhaustive]`, so detecting a further instruction set adds a field
   without a breaking change. `validate_gemm_strided_shape`, the sixth crate-private item of these modules, has no
   caller outside them and stays crate-private.
5. **D5.** Compute keeps an empty default feature set and gains `x86-gemm = ["dep:gemm"]`; inference's `x86-gemm`
   forwards to it, so every caller keeps the same feature name and the same off-by-default behaviour.
6. **D6.1.** The moved source and test files carry 135 tests, names unchanged. Integration tests that use the kernels
   through the facade (`bert_cpu_kernel_policy_api`, `x86_gemm_allocations` and others) stay in inference.
   `facade_api_parity` adds `cpu_forward_parity`, covering the policy types, every re-exported function, the F16
   kernel and the architecture-gated kernels.
7. **D6.4.** `elementwise_cpu_bench` runs these kernels directly, and the workspace builds without link-time
   optimization, so a facade caller now reaches them across a crate boundary where only `#[inline]` functions inline.
   A measured A/B of that target is required before merge. This amendment records the extraction and its validation
   obligation; it does not assert a measurement result or a structural waiver.

## Amendment 15 (2026-10-25)

BERT-family pooling joins `lattice-compute`, next to the SIMD detection it dispatches on. Recorded:

1. **D8.1.** `BertPooling`, `default_bert_pooling_for_model_name`, `mean_pool`, `last_token_pool`, `cls_pool` and
   `l2_normalize`, with their NEON and AVX2 paths, move from `lattice_inference::pool` into a new
   `lattice_compute::pool` module, byte-identical. Its only crate-internal dependency is `simd_config`, already in
   compute after Amendment 14, so compute gains no dependency and its publish tier is unchanged. The BERT, cross-encoder
   and Qwen embedding models that call pooling stay in the facade.
2. **D1.** `lattice_inference::pool` keeps named public re-exports of the six items, and the crate-root
   `lattice_inference::BertPooling` stays the same type. Every existing caller keeps its path.
3. **D4 and D5.** The six items were already public: no crate-private crossing, no `__internal` bridge, no feature or
   target gate. Compute keeps an empty default feature set.
4. **D6.1.** The twelve pooling tests move with the module, names unchanged. `facade_api_parity` adds `pool_parity`,
   covering the type at both facade paths and the five functions.
5. **Non-exhaustive match.** `BertPooling` is `#[non_exhaustive]`, so `BertModel::pool` in the facade now needs a
   wildcard arm; it applies mean pooling, the type's default, and the two existing variants behave as before. Because
   the wildcard would also accept a variant added later, compute adds
   `every_pooling_variant_is_known_to_the_bert_model`, an exhaustive match that stops compiling when a variant is
   added and names `BertModel::pool` as the second place to handle it.
6. **D6.4.** The `mean_pool_l2_normalize` group of `inference_bench` runs `mean_pool` and `l2_normalize` directly, and
   the crate boundary is a changed build input, so a measured A/B of that group is required before merge. This
   amendment records the extraction and its validation obligation; it does not assert a measurement result or a
   structural waiver.

## Amendment 16 (2026-10-25)

Three command-line tools move from the facade into the application crate. Recorded:

1. **D7, application crate.** `quantize_q4`, `quantize_quarot` and `moe_admission_sim` move from
   `crates/inference/src/bin/` to `crates/serve/src/bin/`, byte-identical. They call only the facade's public API, so
   no item changes visibility and there is no D4 crossing. `lattice-serve` already depends on `lattice-inference` and
   `lattice-weights` under Amendment 4 item 1; it gains no dependency, no feature and no lockfile change. The
   facade's library source is unchanged. Its manifest loses the two explicit `[[bin]]` tables, and `lattice-serve`
   declares all three binaries explicitly. None of the three has a feature-conditional path. Under the application
   crate's default features they link the facade with `std`, `download` and `image`, the facade's own defaults, plus
   `serving-support`.
2. **D6.1, tests and fixtures.** `quantize_q4_reader_refactor` and its two golden fixtures move to
   `crates/serve/tests/`, test names and fixture bytes unchanged; the fixtures load relative to the test file.
3. **D6.5, CI and tooling reach.** Every invocation that named the binaries on the facade now names `lattice-serve`: the
   parity workflow's Q4 and QuaRot build steps, the CI contract test anchors for those steps, the app build and
   packaging scripts, and the tool documentation. The Metal lock contract and the measurement inventory list the three
   files at their new paths; all three stay classified as non-measurement surfaces.
4. **D6.4.** None of the three binaries is a bench target, and no declared bench or measurement target of the facade
   reaches their code. The facade's library source, dependencies, features, profiles, bench tables and lockfile are
   identical between base and head; the only manifest change removes `[[bin]]` tables, which Cargo does not build for
   a bench target. This is a structural disposition: the move is unmeasured, and a later change to the tools'
   behaviour takes its own measurement where a target reaches it.

## Amendment 17 (2026-10-26)

The CPU attention kernels that build only on the CPU primitives join `lattice-compute`. Recorded:

1. **D8.1.** `gqa`, `decode`, `differential`, `flash_causal`, `gated` and `native_sparse` move from
   `lattice_inference::attention` into `lattice_compute::attention`, byte-identical, with their tests. Their only
   crate-internal dependencies are the CPU primitives (Amendment 14) and the softmax row rules (Amendment 13), both
   already in compute at the same paths, so compute gains no dependency and its publish tier is unchanged. Standard
   attention, the flash projection wrappers, `AttentionBuffers`, the GDN modules and `AttentionKind`/`AttentionTag`
   stay in the facade for later moves.
2. **D1.** Each of the six facade modules keeps named public re-exports of every public item it had, including
   `apply_sigmoid_gate_neon` on aarch64 and `apply_sigmoid_gate_avx2` on x86_64 under the same target gates.
   `AttentionKind::Gqa` carries the same `GqaConfig` type. Every existing caller keeps its path.
3. **D4.** Every moved item was already public, so no crate-private item crosses and there is no `__internal` bridge.
   The facade's crate-private re-export of `simd_config` had only `gated` and `decode` as users and is removed.
4. **D5.** No feature, default-feature or target gate changes.
5. **D6.1.** The 90 test functions of the six modules (some target-gated) move with them, names unchanged.
   `facade_api_parity` adds `attention_kernel_parity`, covering the nine types, fifteen functions (the two
   target-gated ones under their gates) and the two tile constants at both paths.
6. **D6.4.** Declared targets that reach the moved code: `attn_opt_bench`, `differential_attention_bench`,
   `elementwise_bench`, `gated_attention_bench`, `native_sparse_attention_bench`, `inference_perf`, and
   `decode_attn_bench` under `f16`; the Qwen CPU decode path calls `decode_attention` and `apply_gqa_attention`. The
   crate boundary is a changed build input, so a measured A/B of those targets is required before merge. This
   amendment records the extraction and its validation obligation; it does not assert a measurement result or a
   structural waiver.

## Amendment 18 (2026-10-26)

The GatedDeltaNet recurrent kernels join `lattice-compute`, beside the CPU attention kernels. Recorded:

1. **D8.1.** `gdn` and `gdn_fused` move from `lattice_inference::attention` into `lattice_compute::attention` with
   their tests. The only source change is the import of `Qwen35Config`, which now names
   `lattice_model_config::qwen35_config` directly instead of the facade's `model::qwen35_config` re-export of the same
   type. Their other crate-internal dependencies are the CPU primitives and the LoRA hook contract (Amendment 10), both
   already in compute, and compute already depends on `lattice-model-config`, so it gains no runtime dependency and its
   publish tier is unchanged. `gdn_backward`, gated on `train-backward` and calling the facade's LoRA VJP, stays in the
   facade for its own move.
2. **D1.** `lattice_inference::attention::gdn` and `::gdn_fused` keep named public re-exports of every public item:
   the weight and state types, the two scratch types, the two snapshot aliases and the eleven functions. The Qwen3.5
   CPU, Q8, F16, batch prefill, Metal and speculative paths and `lattice-tune` keep their paths.
3. **D4 and D5.** Every moved item was already public: no crate-private crossing, no `__internal` bridge, no feature
   or target gate.
4. **D6.1.** The 41 test functions and the property tests of the two modules move with them, names unchanged; the
   property tests need `proptest`, so compute gains it as a dev-dependency (a lockfile change to compute's own entry
   only, no new package). `facade_api_parity` adds `gdn_parity`, covering the six types and eleven functions at both
   paths.
5. **D6.4.** The Qwen3.5 CPU forward steps (f32, F16 and Q8) call `gated_delta_net_step_fused` or its SIMD helpers once
   per linear-attention layer per token, and these are `#[inline]`. `gated_rms_norm` is not, and the Q8 NEON forward
   step and the batch prefill path call it across the new boundary; `gated_delta_net_step` is not either, and only tests
   and examples call it. `inference_perf` reaches the code through its `q8_neon_forward` group under `bench-internals`
   and its `qwen_cpu_driver_allocations` group under `bench-internals` and `test-utils` with a Qwen3.5 checkpoint, and
   `backward_stage0` uses the weight type under `train-backward`. The boundary is a changed build input on a hot path,
   so a measured A/B is required before merge. This amendment records the extraction and its validation obligation; it
   does not assert a measurement result or a structural waiver.

## Amendment 19 (2026-10-26)

Standard and tiled attention and the attention variant taxonomy join `lattice-compute`. Recorded:

1. **D8.1.** `standard` (with `AttentionBuffers`) and `flash` (the tiled configuration, scratch and projection
   wrappers) move from `lattice_inference::attention` into `lattice_compute::attention` with their tests, as do
   `AttentionKind` and `AttentionTag` with their tests. The only source changes are imports (the layer weights now name
   `lattice_weights::f32_weights`, and `flash` names `AttentionBuffers` at `attention::standard`), two comments that
   named facade paths, and the visibility of the shape check described in item 3. Their crate-internal dependencies are
   the CPU primitives, the LoRA hook contract and the grouped-query configuration, all already in compute, so compute
   gains no dependency and its publish tier is unchanged. `gdn_backward` stays in the facade.
2. **D1.** `lattice_inference::attention::standard` and `::flash` keep named public re-exports of every public item,
   and `lattice_inference::attention` keeps `AttentionBuffers`, `multi_head_attention`, `AttentionKind` and
   `AttentionTag` by name in place of the former glob over `standard`. Under `bench-internals`,
   `attention::standard::bench_support` and `attention::bench_support` keep their paths as re-exports of the moved
   module, which was already public through the facade and so stays a public compute item (D4, last paragraph).
3. **D4.** Three crate-private items are called from the facade and take disposition 3, as `#[inline]` forwarding
   functions in `lattice_compute::__internal` that the facade calls by full path: `multi_head_attention_in_place` and
   `multi_head_attention_batched`, the BERT encoder's single-sequence and packed-batch entry points, and
   `assert_standard_no_overflow`, which becomes crate-visible in compute for its wrapper. Disposition 2 does not fit
   the two entry points: they take the facade BERT model's load-time fused Q/K/V layout and raw scratch slices, glue
   whose shape follows that model rather than a stable compute contract. The test-only padded batched reference, used
   only by the BERT packed-batch parity test, moves to the facade (`attention/standard/padded_reference.rs`, compiled
   under `cfg(test)`) and calls the shape check by that full path. `total_scratch_len` and
   `tiled_multi_head_attention_in_place` stay crate-private in compute with their only callers, the moved tests.
4. **D5.** Amendment 2 applies to `bench_support`, which is gated on the consumer feature `bench-internals`: compute
   gains a `bench-internals` feature of the same meaning, forwarded by the facade's `bench-internals`. No other
   feature, default-feature or target gate changes. The CI step that lints the facade under `bench-internals` also
   lints compute under its feature, so the moved module keeps its clippy coverage.
5. **D6.1.** The 64 test functions of the three moved test modules move with them, names unchanged. `facade_api_parity`
   adds `attention_standard_parity`, covering the five types and four functions at both paths, the `attention::`
   convenience paths of `AttentionBuffers` and `multi_head_attention`, and the `AttentionKind` methods through the
   facade.
6. **D6.4.** `inference_bench` times `multi_head_attention` with `AttentionBuffers` under default features and the
   packed-batch path under `bench-internals`; `e2e_bench` runs the BERT encoder, which now calls the two attention
   entry points across the boundary, given a BGE checkpoint; `attention_dispatch_bench` matches on `AttentionKind`; and
   the CPU batch prefill builds a `TiledAttentionConfig` once per scratch. The boundary and the feature table are
   changed build inputs on the BERT hot path, so a measured A/B of those targets is required before merge. This
   amendment records the extraction and its validation obligation; it does not assert a measurement result or a
   structural waiver.

## Amendment 20 (2026-10-26)

The model-file layer joins a new crate, `lattice-model-io`: what every model uses to find its files and read engine
switches, and nothing that belongs to one model. Recorded:

1. **D8.1.** The crate is `lattice-model-io`. `download` (canonical model names, checksum verification and the
   cache-miss download), `default_cache_dir` and the by-value switch reader `env_switch_enabled` move out of
   `lattice_inference` with their tests, as does the checkpoint-directory resolver from `test_support`. The only
   source changes are imports (the error type is named at `lattice_inference_core`), the visibility changes in item 3,
   an `#[inline]` hint on `env_switch_enabled`, and comments that named facade paths. Its only internal dependency is
   `lattice-inference-core`, so it publishes in the second tier beside the tokenizer. The name was checked free on
   crates.io in the same pass as a control read of `serde`.
2. **D1.** `lattice_inference::download::ensure_model_files` keeps its path as a named re-export, and
   `facade_api_parity` adds `model_io_parity`, which checks that both paths name one function. `default_cache_dir` and
   `env_switch_enabled` stay crate-private in the facade through a `pub(crate) use`, so every `crate::` caller is
   unchanged.
3. **D4.** `default_cache_dir` and `env_switch_enabled` were crate-private and are called from the facade, so they take
   disposition 2 and become documented public items of the new crate: a path rule and a by-value switch rule that
   `CONFIG.md` already documents. `switch_value_enabled` stays private to the new crate beside its only callers.
   `require_checkpoint_dir` becomes a `#[doc(hidden)]` item of a `test_support` module compiled only for the crate's
   own tests and under its `test-utils` feature; the facade keeps its crate-local `test_support::require_checkpoint_dir`
   path as a re-export for its checkpoint-gated unit tests, and the facade's source-policy tests over its own tree stay
   in the facade.
4. **D5.** The facade's `download` feature forwards to the new crate's `download`, which owns the optional HTTP client.
   The new crate enables `download` by default, as the facade does, and the facade depends on it with default features
   off, so the facade's no-default build still carries no HTTP client. No other feature, default-feature or target
   gate changes.
5. **D6.1.** The 19 test functions of the moved code move with it, names unchanged: 12 download tests, 4
   checkpoint-resolver tests and 3 switch tests. The environment-variable contract moves its `LATTICE_OFFLINE` anchor
   to the new crate and adds the new crate as a reader of `LATTICE_MODEL_CACHE`, and the trust-boundary scan lists the
   new crate among the sources it must reach.
6. **D6.4.** `env_switch_enabled` is called three times per Metal decode step (`forward/metal_qwen35/inner/step.rs`),
   once per generation (`model/qwen35/generation.rs`) and at load (`model/qwen.rs`), now across a crate boundary; the
   BERT and Qwen `from_pretrained` paths, which `e2e_bench` times given a checkpoint, call `default_cache_dir` and
   `ensure_model_files`. The boundary and the feature table are changed build inputs on the Metal decode path, so a
   measured A/B of a Metal decode target is required before merge. This amendment records the extraction and its
   validation obligation; it does not assert a measurement result or a structural waiver.

## Amendment 21 (2026-10-26)

The maintainer's design rule for model code, recorded here as the target the remaining steps of this decomposition
serve: split by layer; code that exists for one model family lives only in that family's crate under `crates/models/`,
and the central crate `lattice-arch` holds the model interface, the shared generation machinery and the registry. The
facade, `lattice-serve` and `lattice-embed` reach models through that interface. The end state is that no crate below
the family crates names a model; items 2 and 6 say how and in which release that is reached. Recorded:

1. **D8.1.** `lattice-model-config` is renamed `lattice-arch`. Neither name has been published; the name is re-checked
   on crates.io immediately before its first upload, as this ADR's publish rule requires. Each family gets one crate,
   `crates/models/<family>/`, published as `lattice-model-<family>` (`bert`, `qwen3`, `qwen35`, `gemma4`,
   `embeddinggemma2`, `ernie45`, `paddleocr-vl`), with the same re-check. A family crate holds that family's
   configuration (from item 6 on), weight-name mapping, weight bundles and loaders, layer composition, inherent impls,
   generation policy (chat tokens, stop strings, decode caps) and the dispatch of kernels only it uses. Family crates
   depend on lower crates and `lattice-arch` only, never on each other; code two families share moves into
   `lattice-arch` or a lower crate. Two named exemptions from "names no model": tokenizer serialization dialects, which
   are formats selected by the caller, and `lattice-model-io`'s download catalog (model names, pinned revisions,
   checksums and tokenizer-file choice), which stays as data until family descriptors supply resolved artifact
   descriptors; pinned checksums and offline behaviour are preserved through that change.
2. **What the lower crates hold today, by item.** Before the first code step an ownership and API ledger is committed
   beside this ADR: every item below with its destination, its disposition (moves with its family, stays as a neutral
   kernel contract, or is replaced by a neutral input) and its public-path consequence. The ledger is reviewed before
   any code moves, and each later step cites its rows.
   - `lattice-compute`: the GatedDeltaNet kernels and `GatedDeltaNetState::new` take `&Qwen35Config` (`attention::gdn`,
     `attention::gdn_fused`); standard, batched and tiled attention take `TransformerLayerWeights`, the per-layer BERT
     bundle; `LoraHook::validate_against` and `validate_against_bert` validate against model types, and the
     Metal-servable GDN module list is backend policy; `default_bert_pooling_for_model_name` chooses pooling from a
     model name; the BERT CPU policy types are BERT metadata. The pooling math and the RoPE tables that Gemma 4,
     EmbeddingGemma2 and ERNIE 4.5 share stay.
   - `lattice-weights`: `TransformerLayerWeights`, `BertWeights`, `CrossEncoderWeights`, `QwenLayerWeights`,
     `QwenWeights` and `ShardedQwenBacking`; the loaders `SafetensorsFile::{load_bert_weights,
     load_cross_encoder_weights, load_qwen_weights}` and `ShardedSafetensors::load_qwen_weights_owned`, which are
     inherent methods of storage types that stay; the F16 hybrid model assembly (`F16ModelWeights` with its attention,
     feed-forward and mixture-of-experts aggregates), where a GDN, GQA or MoE operand record may instead be classified
     as a neutral kernel contract; and `ModelFamily`/`detect_family`. Loaders reach storage through its public tensor
     read methods; no field is widened to make a move compile, and backing lifetimes are preserved.
   - `lattice-quant`: the Qwen recipes of QuaRot (`convert`, the Qwen naming in `io`, `lm_head`, the plan builders in
     `plan`, the configuration plumbing of `rmsnorm_fusion`, `forward_equivalence`, the `lora` and `pipeline`
     orchestration that consumes them, and the hidden seed reader in `__internal`) are split from the neutral plan and
     reader records, the Hadamard and rotation math, the R3 reference, `q4_manifest` and the quantized format types,
     item by item rather than file by file. Qwen3's required tensor list goes to Qwen3, not Qwen3.5.
3. **The interface.** New orchestration paths dispatch through this interface; compatibility exports may still name
   concrete types. `lattice-arch` sits where `lattice-model-config` sits today, above `lattice-weights` and below
   `lattice-compute`, and gains, moved from the facade:
   - ADR-090's `DecoderSession`, its vocabulary (`GenerateConfig`, `TopLogprob`, the stop and decode policy types,
     `StopReason`), sampling and the shared generation driver, which depend on `lattice-grammar` and on nothing in
     `lattice-compute`. Family generation policy that sits in the shared generation code today (the Qwen chat end
     token, stop strings, decode cap, think closing) becomes data each family supplies. Promoting the trait makes it a
     public API, so its supporting types, construction, errors and lifecycle are closed in the same step; prediction
     ids stay opaque and session state stays session-local; sessions carry no `Send` or `Sync` requirement.
   - a registry type and a per-family descriptor (name, decoder or encoder, the `model_type` strings it claims,
     constructors). `lattice-arch` cannot list the families, so each family crate exports its descriptor and each
     consumer builds its table from the families it links. The family and backend are chosen once, when a session or
     encoder is constructed; the session's coarse methods are called per token, as ADR-090 already does, and no
     per-kernel dispatch is added. A name whose family is not compiled in is an error at resolution. The legacy
     `detect_family` keeps its current fallback (an unrecognized configuration resolves to Qwen3.5) on its own path.
     Registration stays in the facade until each constructor moves with its family.
   - an encoder handle that binds one loaded local model and reports its capabilities, its effective output dimension
     and identity (checkpoint revision and the tokenizer, prompt and pooling semantics, not load time), token counts and
     truncation. `lattice-embed` keeps the service layer on top of it: `EmbeddingService`, `EmbeddingModel` and its
     methods, `EmbeddingRole`, the `EmbedError` categories, `MAX_TEXT_BYTES` and `DEFAULT_MAX_BATCH_SIZE`, caching,
     scheduling and remote providers, and the native and cached service types under its `native` feature. This is not
     a second service trait. Reusing one loaded model across separately constructed services needs a named cache
     owner and is not promised by this amendment. Image and document inputs are separate capabilities, not a
     requirement on every encoder.
4. **D1, D4 and the release split.** Free functions and moved owning types keep their public paths by named
   re-export (D1). Items whose published signature names a model type, or whose name and contract belong to one family
   while their owner stays in a lower crate, cannot keep an identical signature once that type leaves the lower crate,
   and D4's internal bridges are never re-exported, so no facade wrapper preserves them:
   `GatedDeltaNetState::new`, `LoraHook::validate_against` and `validate_against_bert`, the four `SafetensorsFile` and
   `ShardedSafetensors` loaders, and the `quarot` plan and descriptor methods whose owners stay in `lattice-quant`.
   0.15.0 adds the neutral replacements beside them, moves every other caller in the workspace to the replacements,
   and marks them deprecated; none of these items changes signature in 0.15.0. 0.16.0 removes them (item 6). That
   removal is this ADR's second D1 exception: it is listed in the release notes with the path map, and D6.2's
   identical-path test is replaced for those items by migration compile probes at the new paths. `lattice-tune`'s
   `LoraAdapter` implements `LoraHook` and overrides both validators; it moves to the replacement in the change that
   adds it. The new validation method defaults to accepting, as the two existing ones do, and through 0.15.0 the code
   that installs a hook calls the deprecated validator as well as the new one, so an implementor that overrides only
   the old method keeps its check; validation still rejects before any state changes.
5. **D5.** A consumer selects families by depending on their crates. The facade takes each family crate as an optional
   dependency behind a feature named for the family and forwards it, and its default set keeps every family path that
   is unconditional today, so the facade's default public surface stays; manifests and cfg change. Families form a
   finite additive feature graph: the cross-encoder is part of `bert`, `paddleocr-vl` implies `ernie45`, and support
   that several families use stays outside any family gate. If a facade `--no-default-features` build loses a model
   path, that is listed as an intentional change. `lattice-arch` states its own minimal feature set.
6. **D9.** Order: (1) the ledger of item 2; (2) the rename and the interface of item 3, with the family configurations
   still in `lattice-arch`, which its crate documentation states; (3) the neutral inputs and deprecations of item 4;
   (4) BERT and the cross-encoder; (5) Qwen3 on CPU and EmbeddingGemma2, after which `lattice-embed` depends on
   `lattice-arch`, `lattice-compute` and those family crates instead of the facade; (6) Qwen3.5 including its Metal
   forward, after ADR-102 has moved the shared Metal engine out of `forward/metal_qwen35`, then Gemma 4, ERNIE 4.5 and
   PaddleOCR-VL, each family's Metal code behind its `metal-gpu` feature; (7) in 0.16.0, the deprecated items of item 4
   are removed, and the configurations, weight bundles and model policy leave `lattice-arch` and the lower crates for
   their families. Until (7), the lower crates still name models through the deprecated items; that is the transition,
   not the target.
7. **Acceptance.** `lattice-arch` depends on no family crate and not on `lattice-compute`. After each family step,
   `cargo tree -i <family crate> --workspace -e normal` lists only the facade, `lattice-embed`, `lattice-serve` and
   `lattice-tune` through the facade, and no family crate depends on another; the graph is read for each feature arm the
   step touches, test dependencies included, and a test-only edge from a lower crate to a family crate fails the step.
   After step 7, a source census of the lower crates for model types and names, run with a known-present control in the
   same pass, finds only the item 1 exemptions. Feature coverage: a build with no family, each family alone with
   defaults off, all families, and the implication pairs; a disabled family's registry entry is absent or returns the
   defined error.
8. **D6.1 and D6.4.** Tests move with their code, and each step compares test-name lists before and after. Kernel
   tests in lower crates that build a family configuration get neutral fixtures or move to the family. Steps 4, 5 and 6
   move code that bench targets execute and each takes a measured A/B on the targets that reach it; step 6 takes a Metal
   decode and prefill A/B. This amendment records the target and its obligations; it does not assert a measurement
   result.

## Direction

The project maintainer's direction, 2026-10-18, quoted as given:

> add to roadmap, inference crate decomposition, will do those in a harmless fashion in high level public API level,
> inference crate itself might be the aggragator interface for acceessing stuff. right now inference crate is huge

and, on the tokenizer as the first extraction:

> if you think that's reasonable enough of a scope, sure., we probably want some kind of standard around these,

This ADR is that standard. "Harmless at the public API level" is D1 and D6, "the aggregator interface" is D1 and D7,
"tokenizer first" is D9.

## Decision summary

`lattice-inference` is split into focused crates along module boundaries that already exist. `lattice-inference`
remains the published entry point: a facade that depends on the extracted crates and re-exports their items at the
paths they have today, so every public `lattice_inference::...` path, type and feature keeps working. `InferenceError`
moves to one small base crate so that extracted crates and the facade share one error type. Every extraction is a
zero-behaviour move with a fixed readiness check and a fixed acceptance list. The first extraction is the tokenizer,
together with the base crate.

## Context

Measured at `main` b58ce287 with `wc -l` over the `.rs` files under each crate's `src/` (shader sources excluded):

| Crate               | Source lines (src/, binaries included) |
| ------------------- | -------------------------------------: |
| `lattice-inference` |                                353,172 |
| `lattice-tune`      |                                 25,698 |
| `lattice-embed`     |                                 22,554 |
| `lattice-fann`      |                                 13,516 |
| `lattice-transport` |                                  5,723 |

`lattice-inference` is about 84% of the workspace's source lines. Its largest top-level modules are `forward`
(89,894 lines), `model` (57,355), `bin` (29,771), `weights` (22,509), `serve` (21,709), `attention` (16,348), `quant`
(15,691), `grammar` (13,595), `vision` (11,033), `decoder` (9,282) and `tokenizer` (8,243). Every change to any of them
recompiles the whole crate, review ownership is crate-wide, and a downstream user who needs one tokenizer compiles a
Metal inference engine.

Facts about the first candidate, the tokenizer module (`src/tokenizer/`), read at the same commit:

- Outside its own files its non-test code uses one thing from the rest of the crate:
  `crate::error::InferenceError`. Its tests additionally use `crate::grammar` and `crate::model::qwen35_config`, and
  two doc comments link into the rest of the crate (`bpe.rs`, links to `crate::model::qwen35` and `crate::grammar`).
- Crate-private items used from other modules include `BpeTokenizer::newline_boundary_table` (called from
  `decoder/qwen_metal.rs` and `model/qwen35/generation.rs`), `token_ids_for_content`, `max_token_byte_span`,
  `tokenize_fragments_with_inserted_ids`, `DEFAULT_BPE_MAX_SEQ_LEN` and `IncrementalDetokenizer`.
  `detokenize::bytes_to_unicode` is re-exported at `lattice_inference::model::qwen35` only under `cfg(test)`, for the
  crate's own unit tests, so it is not part of the public API.
- Three of those items are compiled only under feature combinations that belong to consumers:
  `max_token_byte_span` (`serve`, `f16`, `router-learning`, `metal-gpu` on macOS, or `test`),
  `tokenize_fragments_with_inserted_ids` (`metal-gpu` and `serve`), and `IncrementalDetokenizer::push` (`metal-gpu` on
  macOS, or `test`). CPU test suites that run without `metal-gpu` call `push` and `max_token_byte_span`.
- `InferenceError` (`src/error.rs`) depends on nothing in the crate and is `#[non_exhaustive]`. Two trait impls on it
  live in other modules: `From<GrammarError>` (`grammar/engine.rs`) and `From<PredictionError>` (`decoder.rs`).

Two tool facts shape the acceptance rules. Both were measured with `cargo-semver-checks` 0.50.0 on a minimal workspace
that moves a struct, a function and an error enum out of a crate and re-exports them from it:

- `cargo semver-checks` reports every moved item as removed (`struct_missing`, `function_missing`, `enum_missing`)
  whether the facade re-exports them by name with `#[doc(inline)]` or by glob. Controls in the same run behaved
  correctly: identical code passed, and a really removed function failed. Re-exports from another crate are not
  followed, so the facade check can neither pass on an extraction nor see moved items afterwards.
- The repository's semver job checks default features on Linux only, and under 0.x a workspace minor bump makes it
  skip every check (this is already disclosed by the job itself).

## Decisions

### D1. The facade keeps every public path

`lattice-inference` depends on each extracted crate and keeps each module at its current path as a small shim that
re-exports named items:

```rust
pub mod tokenizer {
    #[doc(inline)]
    pub use lattice_tokenizer::{bpe, common, BpeTokenizer, Tokenizer, TokenizedInput /* ... the full current list */};
}
```

The list is exactly today's public surface of that module, and the crate-root re-exports in `lib.rs` keep their
current names. Re-exports, never copies, so a type reached through the old path and the new crate is the same type.
The facade uses neither a whole-crate alias (`pub use lattice_tokenizer as tokenizer;`) nor a glob
(`pub use lattice_tokenizer::*;`): both forward everything the new crate makes public, including the internal items
D4 keeps out of the supported API, and a glob can collide with another glob.

### D2. `InferenceError` lives in one base crate

The base crate is `lattice-inference-core` (`lattice-core` is taken on crates.io by an unrelated crate). It is created
in the first extraction and holds `InferenceError`, re-exported at `lattice_inference::error::InferenceError` and at
the crate root. Extracted crates return that same type, so no public signature changes. The tokenizer crate does not
own it, because the next candidates (weight and quantization formats, grammar) also return it and have no reason to
depend on a tokenizer.

Rules for the base crate:

1. No dependency on any other lattice crate, no Metal, no default features.
2. `InferenceError` variants carry only standard-library or primitive payloads, never a type from an extracted crate
   (that would create a cycle). A crate with a richer error defines its own error type and implements
   `From<ItsError> for InferenceError`, as `grammar` already does by converting to a string.
3. A new variant is a change to the base crate first, released with the crate that needs it.
4. Any type other than `InferenceError` enters the base crate only in the change that makes a second extracted crate
   need it.

### D3. When a module is ready to move

A module is extractable when all of these hold, each checked by a search with a known-positive control in the same
run:

1. **Outbound paths.** Its non-test sources, doc comments included, reach outside it only into the standard library,
   external crates, the base crate and crates already extracted. Doc links to code that stays behind become plain code
   spans in the move. Self-paths such as `crate::tokenizer::...` are rewritten to the new crate root.
2. **Inbound impls.** No code that stays behind holds an inherent `impl` of a moving type (it stops compiling once
   the type lives in another crate), and every trait impl on a moving type that stays behind still satisfies the
   orphan rule. Inherent impls split across modules exist in the crate today (`impl Qwen35Model` in
   `forward/batch_prefill.rs`, `impl PreparationHandle` in three files), so this check is not hypothetical for later
   steps.
3. **Field access.** No code that stays behind reads or writes a private or crate-private field of a moving type.

Tests that exercise the module together with code that stays behind (the tokenizer's grammar tests, for example) stay
in `lattice-inference` as integration tests, unchanged except for their paths.

### D4. Crate-private items that cross the boundary

Every `pub(crate)` item of the moving module that is used from outside it is listed in the extraction change with one
of three dispositions, in this order of preference:

1. **Stays crate-private** in the new crate, when its outside callers can move with it or use public API instead.
2. **Public**: a documented public item of the new crate, when it is a coherent part of that crate's job and making
   it public does not add to the facade's public surface. A member of a type the facade re-exports (an inherent
   method, an associated item, a field) fails that test: members travel with their type, so making one public adds
   it at the facade path and breaks the identical listing D6.2 requires. In an extraction change such a member takes
   disposition 3; making it public is a separate API change after the move.
3. **Internal**: a free function or extension trait inside `#[doc(hidden)] pub mod __internal` of the new crate,
   called by the facade's code by its full path and never re-exported by the facade. This is for engine-specific glue,
   and for members of re-exported types that other modules still call (for example
   `BpeTokenizer::token_ids_for_content` becomes a free function in `__internal` that calls the crate-private method).

An internal item is never an inherent method of a type the facade re-exports: methods travel with their type, so a
hidden inherent method would be callable through the facade path. No item is widened without a listed disposition.

An item of the moving module that is already public through the facade today, under any path, keeps that path. The
new crate exports it publicly (disposition 2, which adds nothing to the facade because the path already exists) and the
facade re-exports it from there. It is never routed through `__internal`, which the facade does not re-export. A
re-export compiled only under `cfg(test)` is not public API and is not covered by this rule.

### D5. Features

A feature that gates code inside a moved module because the code needs an optional dependency becomes a feature of the
new crate with the same name and meaning, and the facade's feature of that name forwards to it
(`serve = [..., "lattice-tokenizer/serve"]`).

A `cfg` that exists only because the item's callers are compiled under some combination of consumer features (the
three tokenizer items listed under Context) is removed in the move, and the item compiles unconditionally. Keeping such
a gate would require the new crate to declare consumer features it does not use, only so the facade can forward them.
Where the gate also has a `cfg(test)` alternative, it cannot be kept at all: `cfg(test)` is false in a dependency when
the facade's tests run, so a kept gate would break the CPU test suites that call these items today.

A consumer-feature gate on a public item that the facade re-exports is kept, as a feature of the new crate with the
same name and meaning, forwarded by the facade's feature of that name (Amendment 2). Removing it would compile the item
unconditionally and add it to the facade's default public listing, which breaks D6.2's identical listing. First case:
lattice-grammar `bench-internals`. The removal rule above still applies to gates on items that are not on the facade's
public listing.

New crates add no default features of their own; the facade's defaults are unchanged.

### D6. Acceptance of every extraction change

1. **Same tests.** The set of `(crate, test name)` pairs listed by `cargo test --lib --tests -- --list` for
   `lattice-inference` before the move equals the union over `lattice-inference` and the new crate after it, apart from
   the crate prefix of moved tests. Test targets declared `test = false` or `harness = false` do not answer `--list`, and are accounted for by name.
   The counts are stated in the change, and every CI feature-matrix job passes.
2. **Same public surface.** Because `cargo semver-checks` does not follow re-exports from another crate (Context),
   the repository's semver job reports the moved items as missing on an extraction change. The change states this and
   carries two instruments instead:
   - an integration test in `lattice-inference` that names every moved public item through both its facade path and
     its new crate path and proves they are the same item (for types, by assigning one to the other; for traits and
     functions, by use), together with the trait bounds callers rely on (`Send`, `Sync`, `std::error::Error`, `From`
     conversions);
   - a listing of the facade's public items that resolves re-exports across the workspace's rustdoc JSON, identical
     before and after the move.

   Clearing the semver job. The job is part of the required gate, and against a published release of the same
   version it fails on an extraction change, because it reports the moved items as missing. An extraction change
   therefore merges only while `main`'s workspace version is an unreleased minor above the latest published release,
   that is, after the release-bump change for the release that will publish the new crates and before that release is
   published. In that state the job runs zero checks under 0.x versioning, and its disclosure step reports that
   (Context). The two instruments above are then that change's public-surface gate, and their outputs are pasted in the
   change. An extraction change never edits the semver job, its package list or its lint levels, and never bumps the
   workspace version itself.
3. **Clippy and docs.** `cargo clippy` with `-D warnings` on the CI feature sets, and the CI rustdoc command with its
   broken-link lints over the whole workspace, so links in the new crate are checked too. The CI rustdoc job does not
   block merges, so the change runs that command itself and states that it passed.
4. **Measurement.** The bench-compare disposition follows the reachability rules in `CLAUDE.md` ("Keep the Machine
   Quiet During A/B Runs"). A moved module that a declared measurement target reaches gets a before/after comparison of that target. A
   move also changes build inputs (manifests, crate boundaries). Under `CLAUDE.md` the structural waiver applies only
   when build inputs are identical, so reachability cannot waive a build-input change: a declared target that the
   changed build inputs can affect is measured before and after.
5. **CI reach.** Workflow path filters that select work by source path (for example the end-to-end parity gate,
   which watches `crates/inference/src/**` and `crates/embed/src/**`) are extended to the new crate's paths in the same
   change, so moved code keeps triggering the gates it triggered before. The same holds for checked-in lists that
   name the crate a measurement target lives in: GPU-handoff admission reads the `targets` and `bins` arrays in
   `[package.metadata.gpu-bench-handoff]` in `crates/inference/Cargo.toml` and its `[[bench]]` and `[[bin]]` declarations.
   For a binary without a `[[bin]]` table, when `autobins` is not false it reads
   `crates/inference/src/bin/<name>.rs`. `scripts/lib/bench_admission.py` keeps required-feature tables that must name the
   same sets (admission refuses a revision where they differ), and `scripts/bench-measurements.toml` classifies
   binaries by path. A change that moves a bench target or a binary out of `crates/inference` updates all three in
   the same change. The measurement harness admits targets of `lattice-inference` only, so moving a GPU-driving
   target into another crate also needs a harness change, which is itself a build input; that move waits for its own
   decision rather than riding an extraction.
6. **Pure move.** No logic, signature or name changes in the same change beyond what D3 to D5 require. Renames and API
   redesigns are separate changes after the move lands.

### D7. Dependency direction

Extracted crates may depend on the base crate and on each other, in one direction only. None depends on
`lattice-inference`. `lattice-embed` and `lattice-tune` keep depending on the facade; they may later depend on a
narrower crate in their own changes. A CI check over `cargo metadata` fails if any extracted crate depends on
`lattice-inference` or if the extracted crates form a cycle. The crate-ownership and publishing sections of the
`CLAUDE.md` ("Crate Ownership", "Publishing") are updated in the same change that adds a crate.

### D8. Naming, versioning and publishing

1. New crates are named `lattice-<area>`, share the workspace version, licence, lints and repository metadata, and
   carry a README and `package.metadata.docs.rs` features. Each name is checked free on crates.io in the change that
   introduces it.
2. The publish script's tiers are re-derived from path dependencies in that change. With the base crate and the
   tokenizer the order becomes: `lattice-fann`, `lattice-transport`, `lattice-inference-core`; then
   `lattice-tokenizer`; then `lattice-inference`; then `lattice-embed`, `lattice-tune`, with an index wait between
   tiers.
3. A crate that has never been published cannot be resolved from the registry, so a registry-backed dry run of its
   dependents fails until it exists. The first release with a new crate states how its publish was rehearsed.
   Because a dry run also passes when someone else already owns the name, that release re-checks on crates.io,
   immediately before the first upload, that each new name is still unowned or owned by the account that publishes
   the existing lattice crates. The check reads `https://crates.io/api/v1/crates/<name>` and its `owners` endpoint
   with an identifying `User-Agent`, in the same pass as a control read of a crate that is known to exist. Only a
   404 for the name, with the control read succeeding, counts as unowned. A refused, unreadable or timed-out answer,
   a failed control read, or an owner other than that account stops the release before anything is uploaded.
4. A new crate joins the semver job's package list in the first change after its first release, when a baseline
   exists. From then on the extracted crate is checked directly, which is where its API is now defined.

### D9. Order

1. The base crate and the tokenizer: the smallest real case, and it exercises D1 to D8 end to end.
2. Grammar, and weight and quantization formats.
3. The shared Metal runtime and per-model definitions, after ADR-102's in-crate split has brought the Metal files
   under the 3000-line cap, so that code is not cut twice.
4. Serving and the binaries last, since they depend on nearly everything.

Each step is its own change and lands only after the previous one is released or explicitly batched into one release.

## Relation to ADR-102

ADR-102 reorganises code inside `lattice-inference` and states that it introduces no new crate. That remains true of
ADR-102's own changes. This ADR is the separate decision to introduce crates, and it reuses ADR-102's acceptance
discipline (pure moves, measured targets, file cap). Where both apply to the Metal code, ADR-102's in-crate split comes
first.

## Alternatives considered

- **Keep one crate and rely on modules and the file cap.** Reviewability improves, but compile scope, dependency
  weight for narrow users, and crate-level ownership do not. Rejected as the end state; it is the interim state.
- **Split without a facade** (users import the new crates directly). Breaks every existing import and forces
  downstream changes for no behavioural gain. Rejected.
- **Per-crate error types with `From` conversions into `InferenceError`.** Changes the return type of every moved
  public function, which is a breaking change at the facade. Rejected in favour of D2.
- **The tokenizer crate owns `InferenceError`.** Saves one crate now and forces weight, quantization and grammar
  crates to depend on a tokenizer later, or a second move of the error type. Rejected in favour of D2.
- **One large split in a single change.** Unreviewable and not bisectable. Rejected in favour of D9's order.

## Consequences

- Downstream users see no change unless they opt into a narrower crate.
- The workspace gains crates to publish and version; the publish script and release notes cover them.
- Review scope for tokenizer, grammar and format work shrinks to the crate that owns it, and so does compile scope
  for a consumer that depends on the narrow crate directly. Inside the workspace an edit to an extracted crate still
  rebuilds the facade and everything that depends on it, because the facade depends on every extracted crate (D1).
- The semver job on `lattice-inference` reports moved items as missing and cannot check them through the facade
  (Context); extraction changes clear it by the rule in D6.2. Those items are checked on their
  own crates after first release (D8.4), and the facade's surface by the D6.2 instruments.
- Internal glue modules (D4.3) are a cost that should fall over time as consumers move.
