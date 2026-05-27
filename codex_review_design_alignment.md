Verdict: REQUEST CHANGES
Findings: 0 Critical, 4 High, 4 Medium, 0 Low/Nit

Scope: design-goal alignment review for ADR-059 through ADR-063 against `AGENTS.md` and `CLAUDE.md`, plus staleness review for ADR-010, ADR-044, ADR-047, and ADR-048.

## Summary

The new ADRs mostly preserve the core product constraints: no CUDA implementation, no ONNX/PyTorch/TensorFlow runtime, CPU/Metal focus, and Rust-native tooling. ADR-062 explicitly rejects a CUDA-faithful FA2 port, and ADR-063 chooses Rust `hf-hub` rather than Python CLI tooling.

The blocking issue is contract drift. ADR-059 through ADR-063 introduce new taxonomies, cache formats, scheduler ownership rules, and quality claims without amending the accepted ADRs that currently define those contracts. Treat the new ADRs as a useful research roadmap, but do not accept them as stable design records until the stale ADRs and missing gates below are fixed.

## Findings

### [High] ADR-010 Must Be Amended Or Marked Superseded By ADR-059

Evidence: `docs/adr/ADR-010-attention-mechanisms.md:11` says the crate supports "three distinct attention algorithms", while `docs/adr/ADR-010-attention-mechanisms.md:58` decides to implement "four attention variants". ADR-059 says "Lattice has 10 attention mechanisms" at `docs/adr/ADR-059-composable-layer-architecture.md:15` and lists those 10 modules at `docs/adr/ADR-059-composable-layer-architecture.md:17`. ADR-059 also references ADR-010 as the "current 4-variant design" at `docs/adr/ADR-059-composable-layer-architecture.md:956`.

Why this matters: ADR-010 is still Accepted and still rejects sparse attention as out of scope at `docs/adr/ADR-010-attention-mechanisms.md:82`, but ADR-059 relies on Native Sparse Attention, Differential Attention, Gated Attention, Decode Attention, and fused GDN variants as present architectural units. A reader can no longer tell whether ADR-010 is the normative attention taxonomy or a historical baseline.

Suggested fix: Add a short amendment to ADR-010: either update it to the 10-module taxonomy and point dispatch/composition authority to ADR-059, or mark the dispatch/taxonomy portion superseded by ADR-059 while preserving ADR-010 as the historical primitive-level decision.

### [High] ADR-047's F32 Page Contract Conflicts With ADR-062's F16 KV Cache Plan

Evidence: ADR-047 defines `SharedPageRef` as wrapping `Arc<[f32]>` at `docs/adr/ADR-047-paged-kv-cache.md:46`, uses `Vec<Arc<[f32]>>` in `PrefixEntry` at `docs/adr/ADR-047-paged-kv-cache.md:71`, and says Metal reads the same `Arc<[f32]>` allocation at `docs/adr/ADR-047-paged-kv-cache.md:164`. It explicitly defers fp16 prefix pages because they break the identical-layout invariant at `docs/adr/ADR-047-paged-kv-cache.md:180`. ADR-062 now says the existing cache is f32-only at `docs/adr/ADR-062-metal-fa2-prefill.md:48`, and proposes `PagePool` as `Vec<u16>` and `SharedPageRef` as `Arc<[u16]>` for f16 pages at `docs/adr/ADR-062-metal-fa2-prefill.md:105` and `docs/adr/ADR-062-metal-fa2-prefill.md:109`.

Why this matters: This is not just an implementation detail. It changes the cache's storage invariant, prefix copy behavior, Metal buffer interpretation, and any code assuming page element type equals f32. ADR-062 is the "quantized-KV ADR" ADR-047 deferred to, but ADR-047 still reads as the authoritative accepted design.

Suggested fix: Amend ADR-047 with a "superseded by ADR-062 for KV element format" note. Define a format-parameterized page contract (`KvFormat`, element width, scale metadata, f32 reference path) and state which parts of ADR-047 remain unchanged: adapter namespacing, LRU ownership, restore/promote semantics, and 256-token multi-sequence pages unless ADR-062 changes them.

### [Medium] ADR-048 And ADR-063 Are Mostly Compatible, But Phase-2 Worker Ownership Is Now Stale

Evidence: ADR-048 Phase 1 chooses a single Metal command queue and interleaved prefill/decode at `docs/adr/ADR-048-continuous-batching.md:66`, and defines `FifoScheduler` at `docs/adr/ADR-048-continuous-batching.md:185`. ADR-063 says its scheduler implements that `FifoScheduler` at `docs/adr/ADR-063-serving-architecture.md:553` and keeps one Metal command buffer per scheduling tick at `docs/adr/ADR-063-serving-architecture.md:557`. That part is evolution, not contradiction. The stale part is ADR-048 Phase 2: it proposes a separate Tokio task owning a dedicated Metal command buffer encoder at `docs/adr/ADR-048-continuous-batching.md:70`, while ADR-063 says GPU work must run on a dedicated OS thread, not on the tokio runtime, at `docs/adr/ADR-063-serving-architecture.md:265` and `docs/adr/ADR-063-serving-architecture.md:267`.

Why this matters: The current documents agree on FIFO chunked prefill for v1, but disagree on the future disaggregation ownership model. If both remain Accepted/Proposed without an explicit relationship, implementers may build the wrong async boundary.

Suggested fix: Add an ADR-048 amendment that ADR-063 supersedes the "Tokio task owning GPU work" wording. Keep the FifoScheduler and chunked-prefill contract, but restate Phase 2 as an OS-thread or actor boundary with tokio used only for request/channel I/O.

### [Medium] ADR-044 Should Point To ADR-060's OrthogonalBasis Refactor

Evidence: ADR-044 v0 says Hadamard QuaRot covers power-of-two decoder dimensions only and leaves non-power-of-two support to a future v1 choice at `docs/adr/ADR-044-quarot-rotated-quantization.md:83`. ADR-060 says SliceGPT and QuaRot share code through an `OrthogonalBasis` trait at `docs/adr/ADR-060-pruning-toolbox.md:250`, defines that trait at `docs/adr/ADR-060-pruning-toolbox.md:316`, and adds `BlockHadamard`, `PcaCalibration`, and `DenseOrthogonal` basis kinds at `docs/adr/ADR-060-pruning-toolbox.md:340`. ADR-060's implementation plan also explicitly refactors from `quant::quarot` into `quant/orthogonal_basis.rs` at `docs/adr/ADR-060-pruning-toolbox.md:590`.

Why this matters: This is a reasonable evolution, not a contradiction. But ADR-044 remains the accepted QuaRot design and still presents the generic basis question as future/open. ADR-060 now partially resolves the abstraction shape and imposes a composition order: SliceGPT first, QuaRot second at `docs/adr/ADR-060-pruning-toolbox.md:357`.

Suggested fix: Add an ADR-044 amendment noting that ADR-060 generalizes the rotation abstraction into `OrthogonalBasis` while preserving ADR-044's v0 shipped path. Also state that BlockHadamard support in ADR-060 does not automatically make non-power-of-two QuaRot production-ready until ADR-044's v1 quality gates are updated.

### [High] ADR-062 Silently Introduces A New Workspace Crate Despite Claiming Scope Is `lattice-inference`

Evidence: ADR-062's header says `Crate: lattice-inference` at `docs/adr/ADR-062-metal-fa2-prefill.md:5`, but its shader extraction target is `crates/lattice-metal/` with `build.rs` and `src/` at `docs/adr/ADR-062-metal-fa2-prefill.md:63`. The workspace currently lists only five members at `Cargo.toml:3`: `crates/inference`, `crates/embed`, `crates/fann`, `crates/tune`, and `crates/transport`. Local guidance says "Do not create new crates without explicit approval" at `CLAUDE.md:87`, and internal path dependencies must include a version at `AGENTS.md:31`.

Why this matters: A new `lattice-metal` crate changes the workspace topology, dependency direction, publishing order, feature gating, and crate boundary for a CPU-first engine. The ADR may be the right place to grant that approval, but as written it does not acknowledge that it is granting an exception or define whether `lattice-inference` depends downward on `lattice-metal` or whether shaders remain internal to inference.

Suggested fix: Either keep the shader tree inside `crates/inference` or explicitly change ADR-062's crate field to include a new `lattice-metal` crate. If a new crate is intentional, add dependency-direction, feature-gating, versioned path-dependency, and publish-order consequences.

### [High] Quality Gates Are Uneven Across The New ADR Suite

Evidence: The project guidance requires measurement for performance-sensitive changes at `AGENTS.md:10` and `CLAUDE.md:95`, and it explicitly calls out differential/PPL validation as a way to catch structural inference bugs at `CLAUDE.md:33` and `CLAUDE.md:46`. ADR-060 has a mandatory PPL gate at `docs/adr/ADR-060-pruning-toolbox.md:469`. ADR-061 defines paired comparison reports with PPL, latency, confidence intervals, and permutation tests at `docs/adr/ADR-061-inference-metrics-infrastructure.md:411`. But ADR-059's migration invariant only requires `make bench-compare` throughput at `docs/adr/ADR-059-composable-layer-architecture.md:900`. ADR-062 Phase 5 says "Qwen PPL and LongBench/RULER/NIAH within threshold" without naming the thresholds at `docs/adr/ADR-062-metal-fa2-prefill.md:404`. ADR-063's benchmark suite defines latency/throughput metrics at `docs/adr/ADR-063-serving-architecture.md:822` but includes an embedding workload at `docs/adr/ADR-063-serving-architecture.md:808` without embedding quality/cosine-similarity gates.

Why this matters: These ADRs change architecture selection, KV precision, quantization, and serving behavior. Throughput-only gates can bless a faster but wrong model. The missing thresholds are especially risky for f16/int8/int4 KV cache, architecture search, and API-compatible embeddings.

Suggested fix: Add explicit quality gates per ADR phase: paired PPL/NLL delta, token agreement where applicable, max-logit or distribution-distance checks for kernel refactors, embedding cosine similarity/top-k retrieval stability for embedding endpoints, and named LongBench/RULER/NIAH thresholds for ADR-062 Phase 5. Keep `make bench-compare` as a performance gate, not the only correctness gate.

### [Medium] The ADR Set Is Over-Ambitious Unless Split Into MVP And Research Tracks

Evidence: AGENTS describes `lattice-inference` as about 57K LOC at `AGENTS.md:95`. ADR-059 proposes a YAML `ModelSpec`, heterogeneous attention/FFN/quant composition, and architecture search environment at `docs/adr/ADR-059-composable-layer-architecture.md:515` and `docs/adr/ADR-059-composable-layer-architecture.md:752`. ADR-062 spans shader extraction, f16 KV, FA2 prefill, prefix cache wiring, int8 KV, int4 WHT, SRFT, and RateQuant across six phases at `docs/adr/ADR-062-metal-fa2-prefill.md:368`. ADR-063 proposes CLI, model registry, HTTP server, OpenAI and Anthropic APIs, continuous batching, prefix cache, and cross-framework benchmarks with an 8-phase implementation plan at `docs/adr/ADR-063-serving-architecture.md:956`.

Why this matters: The direction is aligned with Lattice's goals, but accepting all of this as one design wave would convert a compact pure-Rust inference engine into a serving platform, experiment framework, pruning suite, shader subproject, and architecture-search system at once. That is feasible only if the status model distinguishes "accepted MVP contract" from "research roadmap".

Suggested fix: Split each large ADR into an MVP accepted section and explicitly Future/Experimental follow-ons. For example: ADR-059 D1-D2 first, D5-D6 future; ADR-062 Phase 0-2 first, Phase 4-6 future; ADR-063 P1-P5 first, P6-P8 after quality/perf gates. Use ADR-061's experiment runner as the evidence spine before promoting research phases.

### [Medium] ADR Index And Local Guidance Are Stale For ADR-059 Through ADR-063

Evidence: `AGENTS.md:245` says there are "58 ADRs in `docs/adr/INDEX.md`". The index's last section is "workspace / CI (ADR-058)" at `docs/adr/INDEX.md:87`, with only ADR-058 listed at `docs/adr/INDEX.md:91`. ADR-059 through ADR-063 exist as files but are not indexed.

Why this matters: ADR numbering and status are part of the governance contract. The new ADRs are Proposed, so they should be visible in the index with status, crate, and dependency links. Otherwise reviewers and implementers may miss that accepted ADRs have proposed successors.

Suggested fix: Add ADR-059 through ADR-063 to `docs/adr/INDEX.md` under the appropriate crate/workspace sections, with `Proposed` status. Update `AGENTS.md` after they are intentionally part of the repo, or change AGENTS to avoid hard-coding the count.

## Direct Answers To The Requested Checks

1. ADR-010 should be updated or marked superseded for attention taxonomy/dispatch. The old 4-variant design conflicts with ADR-059's 10-variant architecture.
2. ADR-047 needs an amendment. ADR-062 is the quantized-KV ADR that ADR-047 deferred, and it changes the `Arc<[f32]>` page invariant.
3. ADR-048 vs ADR-063 is mostly evolution, not contradiction, for Phase 1 FIFO/chunked prefill. The future "Tokio task owns GPU work" language in ADR-048 should be amended because ADR-063 moves GPU work to a dedicated OS thread.
4. ADR-044 does not need to be rewritten, but it needs an amendment pointing to ADR-060's `OrthogonalBasis` refactor and preserving ADR-044's v0 QuaRot invariants.
5. The ADRs are over-ambitious if treated as one acceptance batch. They stay aligned only if split into staged MVP contracts and research follow-ons. They do maintain the no-CUDA/no-ONNX constraint; no new external ML runtime is proposed.
6. Measured quality benchmarks are missing or underspecified in ADR-059, ADR-062 Phase 5, and ADR-063 embeddings/serving. ADR-060 and ADR-061 are the strongest because they define mandatory PPL and paired-comparison machinery.

## What I Checked

- `AGENTS.md` project goals: pure Rust, CPU-first/GPU-optional, no CUDA, no ONNX/PyTorch/TensorFlow, crate structure, deps, tests, ADR rules.
- `CLAUDE.md` process rules: measure-first, differential/PPL validation, no new crates without approval, no CUDA or ONNX dependencies.
- ADR-059 through ADR-063 for design alignment, scope, quality gates, dependency implications, and cross-ADR references.
- ADR-010, ADR-044, ADR-047, and ADR-048 for staleness against the new ADRs.
- Targeted code searches for current attention modules, current `LayerType`, and current f32 KV/prefix-cache storage.

## False Positives Ruled Out

- No CUDA implementation is being added. ADR-062 explicitly rejects a CUDA-faithful FA2 port, and ADR-063 mentions CUDA only in competitor context.
- No ONNX/PyTorch/TensorFlow runtime is proposed. ADR-063's added dependencies are web/CLI/template/download crates, not ML runtimes.
- ADR-048 and ADR-063 do not contradict on v1 FIFO scheduling; the conflict is limited to future worker ownership wording.

## What I Did Not Check

- I did not run benchmarks, PPL evals, or LongBench/RULER/NIAH. This was a static ADR alignment review.
- I did not verify external market/funding claims or paper claims. Those are outside this local design-alignment pass.
- I did not review every referenced research workspace file under `workspaces/20260527/`.

## Recommended Next Steps

1. Patch ADR-010, ADR-047, ADR-048, and ADR-044 with explicit amendment/supersession notes.
2. Add ADR-059 through ADR-063 to `docs/adr/INDEX.md` with `Proposed` status and dependency links.
3. Split ADR-059, ADR-062, and ADR-063 into MVP acceptance gates plus Future/Experimental sections before accepting them.
4. Add quality thresholds where they are currently vague: PPL/NLL deltas, token agreement, logit-distance checks, embedding cosine similarity, and named LongBench/RULER/NIAH thresholds.
5. Decide whether ADR-062 is explicitly approving a new `lattice-metal` crate; if yes, document the workspace/dependency/publish consequences.

Re-review guidance: another pass is useful after the amendment notes and quality gates are added. A narrow re-review should be enough if the fixes are textual and do not change the proposed architecture.

Domain utility: SKIPPED - no `mcp__lore__suggest` / `mcp__lore__compose` tools were available in this session; review used the local spec-alignment skill and repository artifacts.
