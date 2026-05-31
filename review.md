# PR #128 Review — ADR-059 to ADR-063 + `gap_inventory_20260527.md`

Verdict: **REQUEST CHANGES**

## 1) Coherency of "composable architecture exploration platform" across all 5 ADRs
The pitch is largely coherent: ADR-059 introduces a typed layer graph and `ModelSpec` DSL, ADR-060 maps pruning/search actions onto that graph, ADR-061 adds `LayerMetrics`/experiment infrastructure to score candidates, ADR-062 targets memory/perf substrate, and ADR-063 turns that into a product loop. The cross-dependency chain is explicit in PR body and within each ADR (`depends on` entries and phase plans).

Strengths that hold across ADRs:
- [ADR-059]( /Users/lion/projects/khive/lattice/docs/adr/ADR-059-composable-layer-architecture.md:98) sets a shared `LayerOp`/`AttentionKind` runtime model.
- [ADR-060]( /Users/lion/projects/khive/lattice/docs/adr/ADR-060-pruning-toolbox.md:46) and [ADR-061]( /Users/lion/projects/khive/lattice/docs/adr/ADR-061-inference-metrics-infrastructure.md:7) explicitly build on that model.
- [Gap inventory]( /Users/lion/projects/khive/lattice/docs/gap_inventory_20260527.md:177) already frames those capabilities as the current strategic wedge.

Weakening factors:
- ADR-010 is still `Accepted` with a 4-attention-variant model baseline, while ADR-059 pivots to 10 variants and a search DSL, so this platform story is not yet internally synchronized.
- Several assumptions are high-risk and deferred (LoRA recovery, full sparse kernels, quantized-KV serving), which means the immediate architecture-research loop is not fully end-to-end in v1.

## 2) Critical path to 60-second seed-round demo
ADR-063 defines the target path as `lattice pull -> lattice serve -> curl` and calls it a 60-second demo (critical path) at [line 942]( /Users/lion/projects/khive/lattice/docs/adr/ADR-063-serving-architecture.md:942).

This is directionally clear but under-specified for execution reliability:
- There is no explicit latency budget by sub-step (first-time HF pull vs cached model, model size assumptions, tokenizer download, CPU vs Metal cold-start, etc.).
- The full path depends on ADR-062 and ADR-047/048 integration work that is out-of-scope in parts (`D6` wiring and multi-model server behavior in ADR-063 are deferred milestones), so “under 60s” can fail before demo-readiness unless those are explicitly pre-implemented.
- The CLI/serve split itself is significant: ADR-063 estimates [8 PRs/effort blocks]( /Users/lion/projects/khive/lattice/docs/adr/ADR-063-serving-architecture.md:956) with no explicit dependency compression for parallelization.

## 3) Competitive moat vs llama.cpp / vLLM / SGLang / Burn / Ollama
The moat argument is strongest where it is specific to lattice assets:
- Pure-Rust implementation and Apple/Metal-first focus ([ADR-063]( /Users/lion/projects/khive/lattice/docs/adr/ADR-063-serving-architecture.md:74) and [ADR-044]( /Users/lion/projects/khive/lattice/docs/adr/ADR-044-quarot-rotated-quantization.md:116)).
- Architecture exploration workflow (declarative model spec + pruning/search + metrics) is a real differentiator if shipped as operational tooling.

Weakening factors:
- The docs repeatedly compare capabilities but do not yet establish defensible exclusivity for serving-level UX. Competitors already offer comparable API coverage plus broad ecosystem breadth.
- The pitch risks over-indexing on “only this platform” claims at the feature level while deferring several hard differentiators (training loop, LoRA runtime compatibility, quantized-KV in server mode).

## 4) Research platform vs production engine tension
The tension is identified and partially addressed.
- [Gap inventory]( /Users/lion/projects/khive/lattice/docs/gap_inventory_20260527.md:167) explicitly calls out missing training-as-exploration.
- ADR-060 and ADR-061 are research-first (search + metrics + pruning).
- ADR-063 is production-first (server, APIs, CLI, benchmarks).

The tension is still meaningful:
- ADR-044 marks runtime LoRA incompatibility with QuaRot artifacts, yet ADR-060 and ADR-063 assume LoRA-adjacent flows as normal user/serving scenarios.
- ADR-062’s quantization path is substantial and touches serving-critical cache/code-paths, while production API maturity is still v1-limited.
- Without a clear “research-in-loop” gating policy (what can be exposed, what is sandboxed), the architecture may accumulate incompatible feature branches.

## 5) Realism of 40-phase plan across 5 ADRs in 3 months
The package contains `40` named phases across ADR-059–063 (`P1-P6`, `P0-P13`, `P1-P5`, `P0-P6`, `P1-P8`).

Major realism issue:
- This is an extremely aggressive integration sequence for one team, with multiple high-complexity kernels (Metal FA2 + quantized KV), large API surface (axum, cancellation, streaming), and experimental research components.

At minimum, this plan needs a resource/capacity model or explicit phase truncation for seed-round.

## 6) Investor questions not currently answered in ADR set
Open questions likely raised immediately in diligence:
- What are exact acceptance thresholds for each milestone (latency/TTFT/TPS/quality) and the go/no-go rule per phase?
- What is the model-card support matrix for seed demo (which weights, exact quant formats, and hardware floor)?
- What are operational guarantees (authN/authZ, key rotation, tenant isolation, cancellation semantics, backpressure policy under queue overrun)?
- What is the plan for reproducibility and auditability in claims (seeded corpora, hardware SKU, confidence intervals, significance thresholds)?
- What is the support posture when a user wants LoRA on QuaRot-converted models (explicitly unsupported in v0)?

## 7) Do ADR-010, ADR-044, ADR-047, ADR-048 need updating?

Yes — all four need at least a follow-up note or status update:
- [ADR-010]( /Users/lion/projects/khive/lattice/docs/adr/ADR-010-attention-mechanisms.md:56) currently documents the accepted 4-variant model; ADR-059 now defines 10 variants and DSL composition. This is a direct scope mismatch.
- [ADR-044]( /Users/lion/projects/khive/lattice/docs/adr/ADR-044-quarot-rotated-quantization.md:125) explicitly blocks runtime LoRA on QuaRot assets; ADR-060 and ADR-063 depend on LoRA-adjacent workflows and should reference this as a product constraint.
- [ADR-047]( /Users/lion/projects/khive/lattice/docs/adr/ADR-047-paged-kv-cache.md:110) excludes quantized KV; ADR-062 makes quantized KV a core phase before serving quality wins. The interface and invariants should be reconciled.
- [ADR-048]( /Users/lion/projects/khive/lattice/docs/adr/ADR-048-continuous-batching.md:76) is still production-scheduler centric and explicitly out-of-scope for serving/CPU path details; ADR-063 can reference it, but should not reuse assumptions that clash with its threaded scheduler + API lifecycle contract.

## Findings

1. [Major] The 60-second demo claim is not yet operationally grounded.
   - Evidence: seed-demo assertion at [ADR-063:942]( /Users/lion/projects/khive/lattice/docs/adr/ADR-063-serving-architecture.md:942) without explicit preconditions or cold-start timing assumptions.
   - Impact: investor-facing claim can overpromise; brittle under unmodeled conditions.

2. [Major] 40-phase plan is not realistic as a 3-month MVP target without explicit staffing/parallelization controls.
   - Evidence: combined phase load is large and explicit in each ADR plan sections (`P0-P13`, `P0-P6`, `P1-P5`, `P1-P8`) [ADR-060:579]( /Users/lion/projects/khive/lattice/docs/adr/ADR-060-pruning-toolbox.md:579), [ADR-062:370]( /Users/lion/projects/khive/lattice/docs/adr/ADR-062-metal-fa2-prefill.md:370), [ADR-063:956]( /Users/lion/projects/khive/lattice/docs/adr/ADR-063-serving-architecture.md:956), [ADR-061:458]( /Users/lion/projects/khive/lattice/docs/adr/ADR-061-inference-metrics-infrastructure.md:458).
   - Impact: schedule drift and missed quality gates are likely if executed as one stream.

3. [Major] ADR-010 remains a stale baseline vs ADR-059 scope.
   - Evidence: ADR-010 decision stays at 4 attention variants at [ADR-010:58]( /Users/lion/projects/khive/lattice/docs/adr/ADR-010-attention-mechanisms.md:58) while ADR-059 states 10 variants + DSL-driven composition.
   - Impact: engineers and reviewers can follow conflicting design assumptions.

4. [Major] Research-platform assumptions rely on capabilities explicitly deferred or excluded in foundational ADRs.
   - Evidence: ADR-044 marks runtime QuaRot LoRA incompatibility [ADR-044:125]( /Users/lion/projects/khive/lattice/docs/adr/ADR-044-quarot-rotated-quantization.md:125), while ADR-060/063 propose LoRA-adjacent flows and serving assumptions.
   - Impact: claims of “research loop” quality and deployability are conditional but not surfaced clearly in ADR-063 dependency story.

5. [Major] 60s demo and moat claims should be tightened against competitor baselines.
   - Evidence: projected performance in ADR-062 is directional (“Expect 2-5x prefill speedup” at [ADR-062:418]( /Users/lion/projects/khive/lattice/docs/adr/ADR-062-metal-fa2-prefill.md:418)) and no in-ADR acceptance envelope is attached to these targets.
   - Impact: strategic narrative can outpace substantiated engineering deliverables.

6. [Medium] Metrics instrumentation plan is strong but does not include a concrete fail-safe on cross-layer invalid configs.
   - Evidence: ADR-059 explicitly warns about runtime mismatches in `AttentionOp` dispatch [ADR-059:874]( /Users/lion/projects/khive/lattice/docs/adr/ADR-059-composable-layer-architecture.md:873) and ADR-061 depends on these signals for pruning loops.
   - Impact: a malformed spec can pass config-level construction and still destabilize optimization loops.

7. [Medium] Product boundary includes meaningful omissions at v1 that should be called out in investor deck language.
   - Evidence: v1 defers several “obvious” endpoints and capabilities [ADR-063:930]( /Users/lion/projects/khive/lattice/docs/adr/ADR-063-serving-architecture.md:930) and [ADR-063:933]( /Users/lion/projects/khive/lattice/docs/adr/ADR-063-serving-architecture.md:933).
   - Impact: pitch phrasing should avoid “serving-complete” framing until v2 items are explicitly out of scope.

## Suggested remediation sequence

1. Add a v1 alignment ADR that reconciles ADR-010/044/047/048 baseline behavior with ADR-059–063 assumptions and constraints.
2. Add quantitative acceptance criteria and explicit preconditions for all demo and performance claims.
3. Add a 12-week calendarized rollout with owners, dependencies, and cancellation criteria for 40 phases.
4. Add explicit investor-ready architecture-invariance guarantees (quantization, LoRA compatibility, API compatibility levels, benchmark methodology).
