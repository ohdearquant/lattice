Verdict: REQUEST CHANGES
Findings: 3 High, 2 Medium, 0 Low/Nit

## Summary

ADR-059 through ADR-063 are directionally coherent as a research stack, but the current set is not yet implementation-ready. The main defects are stale type names from before ADR-059, conflicting ownership between pruning calibration and metrics infrastructure, an unmarked reopening of ADR-047's prefix-cache decision, and an overlarge combined roadmap.

## Findings

### [High] ADR-060 still uses superseded `LayerType` instead of ADR-059's attention taxonomy

Evidence: `docs/adr/ADR-059-composable-layer-architecture.md:887` says "`LayerType` ... is superseded by `AttentionKind`"; `docs/adr/ADR-059-composable-layer-architecture.md:193` through `docs/adr/ADR-059-composable-layer-architecture.md:219` define/use `AttentionKind` variants and `AttentionTag` outputs. ADR-060 still defines `LayerStats.layer_type: LayerType` at `docs/adr/ADR-060-pruning-toolbox.md:57`, accepts `layer_type: LayerType` in `CalibrationObserver` at `docs/adr/ADR-060-pruning-toolbox.md:115`, copies `s.layer_type` into candidates at `docs/adr/ADR-060-pruning-toolbox.md:225`, and cites `LayerType enum` as current code context at `docs/adr/ADR-060-pruning-toolbox.md:641`.

Why this matters: This directly violates the requested coherence point (2). A pruning implementer following ADR-060 would rebuild the old two-value `GatedDeltaNet | FullAttention` dispatch surface while ADR-059's whole point is to support the broader `AttentionKind`/`AttentionTag` set.

Suggested fix: Replace ADR-060's `LayerType` fields with `AttentionTag` for scoring/reporting, or `AttentionKind` only where runtime dispatch is actually required. Also add an explicit `AttentionTag` enum definition to ADR-059, since ADR-059 uses the type but only shows its variants indirectly through `tag()`.

### [High] `LayerMetrics` has two incompatible schemas

Evidence: ADR-059 defines `LayerMetrics` with `attention_entropy`, `activation_rms`, `outlier_ratio`, `residual_cosine_sim`, and `latency_us` at `docs/adr/ADR-059-composable-layer-architecture.md:414` through `docs/adr/ADR-059-composable-layer-architecture.md:432`. ADR-061 redefines `LayerMetrics` with `mode`, `latency_ns`, `input_norm`, `output_norm`, `update_ratio`, `block_influence`, `entropy`, `sparsity`, `kv_page_mass`, and `pattern_label` at `docs/adr/ADR-061-inference-metrics-infrastructure.md:111` through `docs/adr/ADR-061-inference-metrics-infrastructure.md:137`.

Why this matters: `ForwardCtx` itself is mostly consistent across the two ADRs, but `pub metrics: Option<&mut LayerMetrics>` points to different concrete structs depending on which ADR is read. That makes the API contract ambiguous and will cause drift in code, tests, and examples.

Suggested fix: Make ADR-061 the authoritative `LayerMetrics` schema and update ADR-059 to reference it instead of defining a separate struct, or downgrade ADR-059's snippet to an illustrative placeholder. Normalize timing units (`latency_ns` vs `latency_us`) in the same edit.

### [High] Pruning and metrics dependency order is contradictory

Evidence: ADR-060 says "Every pruning method requires: calibration data -> forward pass -> activation capture -> importance scoring. Build this once" at `docs/adr/ADR-060-pruning-toolbox.md:48` through `docs/adr/ADR-060-pruning-toolbox.md:50`, then schedules `CalibrationObserver` and `LayerStats` as P0 at `docs/adr/ADR-060-pruning-toolbox.md:581`. ADR-061 declares `Depends on: ADR-059 (ModelSpec), ADR-060 (Structured Pruning)` at `docs/adr/ADR-061-inference-metrics-infrastructure.md:7`, but also says missing metrics "block" structured pruning at `docs/adr/ADR-061-inference-metrics-infrastructure.md:25` and that "CheapOnline mode IS the pruning scoring signal. No separate calibration pass needed" at `docs/adr/ADR-061-inference-metrics-infrastructure.md:413` through `docs/adr/ADR-061-inference-metrics-infrastructure.md:415`.

Why this matters: The phase graph has no clear owner for the forward hooks and scoring signal. Either ADR-060 can ship its calibration loop before ADR-061, or ADR-061 supplies the metrics substrate that ADR-060 needs, but the current text claims both.

Suggested fix: Choose one ownership model. If ADR-061 owns metrics, change ADR-060 to depend on ADR-061 Phase 1 and remove/rename ADR-060 P0 as pruning-specific consumers. If ADR-060 owns calibration, keep ADR-061 depending on ADR-060 but remove the "no separate calibration pass needed" claim and say ADR-061 consumes `LayerStats`.

### [Medium] ADR-063 reopens ADR-047 radix-prefix caching without marking supersession or scheduling it

Evidence: ADR-047 accepts a hash-map `PrefixPageCache` at `docs/adr/ADR-047-paged-kv-cache.md:37` through `docs/adr/ADR-047-paged-kv-cache.md:50`, explicitly excludes full radix tree work at `docs/adr/ADR-047-paged-kv-cache.md:106` through `docs/adr/ADR-047-paged-kv-cache.md:110`, and defers radix tree implementation at `docs/adr/ADR-047-paged-kv-cache.md:176`. ADR-063 exposes `--prefix-cache none | radix` at `docs/adr/ADR-063-serving-architecture.md:148`, then says `PagedKVCache(page_size = 256)` uses a radix tree at `docs/adr/ADR-063-serving-architecture.md:744` through `docs/adr/ADR-063-serving-architecture.md:750`. Its implementation plan at `docs/adr/ADR-063-serving-architecture.md:958` through `docs/adr/ADR-063-serving-architecture.md:968` has no phase for building that radix cache.

Why this matters: This is not necessarily a bad design choice because ADR-047 deferred radix trees to the server path, and ADR-063 is that server path. The defect is that ADR-063 depends on ADR-047 instead of saying it extends or supersedes ADR-047's prefix-cache choice, and it describes a v1 feature with no phase or gate.

Suggested fix: Either make ADR-063 v1 use ADR-047's `PrefixPageCache` and defer radix to v2, or add an explicit "Supersedes ADR-047 prefix-cache implementation choice for serving" note plus a dedicated implementation phase and acceptance criteria.

### [Medium] The combined roadmap is too large for a 2-person, 6-month delivery target

Evidence: ADR-059 has P1-P6 and estimates 7-9 PRs at `docs/adr/ADR-059-composable-layer-architecture.md:889` through `docs/adr/ADR-059-composable-layer-architecture.md:898`. ADR-060 has P0-P13 at `docs/adr/ADR-060-pruning-toolbox.md:579` through `docs/adr/ADR-060-pruning-toolbox.md:594`. ADR-061 adds five phases at `docs/adr/ADR-061-inference-metrics-infrastructure.md:458` through `docs/adr/ADR-061-inference-metrics-infrastructure.md:488`. ADR-062 adds Phase 0 through Phase 6 at `docs/adr/ADR-062-metal-fa2-prefill.md:368` through `docs/adr/ADR-062-metal-fa2-prefill.md:410`. ADR-063 adds P1-P8 at `docs/adr/ADR-063-serving-architecture.md:956` through `docs/adr/ADR-063-serving-architecture.md:968`.

Why this matters: The combined plan is roughly 40 phases across architecture refactoring, pruning, metrics, Metal shader extraction, FA2 prefill, KV quantization, radix caching, and HTTP serving. That exceeds what a 2-person team can credibly ship in 6 months unless many phases are explicitly research backlog rather than delivery scope.

Suggested fix: Split the ADR set into a 6-month MVP dependency chain and a research backlog. Mark the MVP phases that must ship together, move experimental phases such as SliceGPT, int4/SRFT/RateQuant, radix cache, and agent-loop automation behind a later milestone, and add rough effort estimates for every ADR, not only ADR-063.

## What I Checked

- Type consistency for `ForwardCtx`, `LayerMetrics`, `AttentionKind`, and `AttentionTag` across ADR-059 through ADR-063.
- Whether ADR-060 still references superseded `LayerType`.
- Phase numbering and dependency ordering across the five ADR roadmaps.
- ADR-062/ADR-063 references to ADR-047 and ADR-048, including whether supersession is accounted for.
- Combined phase count against the requested 2-person, 6-month realism check.

## False Positives Ruled Out

- ADR-062's f16/int8/int4 KV work does account for ADR-047's f32 prefix-cache baseline: ADR-047 defers quantized KV to a separate ADR, and ADR-062 is explicitly that separate ADR.
- ADR-063's continuous batching usage is compatible with ADR-048 Phase 1: it implements `FifoScheduler`, chunked prefill, and single-resource GPU scheduling rather than assuming full disaggregated prefill/decode.
- ADR-059's P5/P6 dependency note is not itself circular if read narrowly: `ResidualPolicy::Skip` can land before ADR-060 consumes it, and `ArchitectureEnv` can depend on ADR-061 after `ModelSpec` exists.

## What I Did Not Check

- I did not validate the ADR claims against Rust implementation files beyond the line references already embedded in the ADRs.
- I did not run `cargo test`, benchmarks, or Metal validation because this was a documentation coherence review.
- I did not verify external paper claims or performance numbers.

## Recommended Next Steps

1. Fix the stale `LayerType` references and define a single authoritative attention taxonomy.
2. Make `LayerMetrics` authoritative in exactly one ADR, then update `ForwardCtx` snippets to reference that schema.
3. Resolve pruning-vs-metrics ownership before implementation planning continues.
4. Decide whether ADR-063 supersedes ADR-047 for radix prefix caching or defers radix cache out of v1.
5. Cut the combined roadmap into an MVP milestone and a later research backlog.

Re-review guidance: a narrow re-review is useful after the taxonomy, metrics schema, and dependency graph edits land.

Domain utility: SKIPPED — the requested lore `suggest`/`compose` tools were not available in this session, so this review used the local ADR/spec-alignment rubric and direct file inspection.
