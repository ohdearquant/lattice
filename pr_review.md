# PR #128 Review (`ADR-059`…`ADR-063` + `docs/gap_inventory_20260527.md`)

## Verdict
**REQUEST CHANGES**

The ADR set is directionally coherent, but the strategy has execution and investor-readiness gaps that block a defensible seed-stage story in one shot. Several major risks are acknowledged yet not resolved in the cross-ADR critical-path graph.

## Executive answers

1. **Composability pitch coherence across 5 ADRs:** mostly coherent, but not fully closed. ADR-059 establishes architecture search and typed state categories, ADR-060/061 depend on that, ADR-047/062/063 reuse the same primitives. Coherence is strongest on capability, weakest on production integration semantics.
2. **Critical path to 60-second demo:** exists, but under-scoped/under-sequenced. `lattice pull -> lattice serve -> curl` is the declared pitch, but ADR-063 also introduces many adjacent features before hard lock on what is truly required for seed demo.
3. **Competitive moat:** mixed. The technical thesis (Apple-Silicon-native + Rust + research-composable + QuaRot/Speculative + low-friction serving) is credible, but moat is currently mostly latent and not yet evidenced with cross-framework benchmarks.
4. **Research platform vs production engine tension:** present and real. State safety (especially GDN recurrent state), quantization/LoRA compatibility, and serving lifecycle are still separate threads; unified failure modes are not fully resolved.
5. **40-phase scope realism (3 months):** not realistic without aggressive parallelization and strict phase pruning.
6. **Investor questions not yet addressed:** unit economics, reliability/operational guarantees, model coverage roadmap beyond M-series, and explicit go/no-go criteria for claiming seed demo performance.
7. **Need updates to ADR-010/044/047/048:** yes, at least to prevent stale architecture assumptions.

## Findings

1. **Major — 60-second seed demo still coupled to non-critical work and has ambiguous acceptance criteria**
   - `docs/adr/ADR-063-serving-architecture.md:942-950` defines seed demo as `pull -> serve -> curl`, while phase map includes API expansion and benchmark scaffolding before/around that path (`:96-105`, `:962-969`).
   - `docs/adr/ADR-063-serving-architecture.md:834-848` repeatedly labels benchmark rows as *targets*, not achieved.
   - Impact: the pitch reads aspirational but lacks a strict minimal-milestone lock for the 60-second claim.
   - Fix: add a single “DEMO-READY” gate that includes exactly the minimal runnable feature set, hard metrics, owner, and owners for blocking risks.

2. **Major — Strong research/production split is not fully reconciled for stateful components**
   - ADR-059 risk explicitly flags GDN state leaks in multi-user settings (`docs/adr/ADR-059-composable-layer-architecture.md:880-881`) and ADR-048 similarly tracks this in batching context (`docs/adr/ADR-048-continuous-batching.md:104-123`, `:190-205`, `:266-287`).
   - ADR-063 assumes serving as product surface but does not encode how research-mode layer mutability and multi-tenant request lifecycle will safely interact with these state rules.
   - Impact: architectural composability could outperform in experiments but fail in production if request isolation/state reset contracts differ across CPU/Metal and search/runtime modes.
   - Fix: define one cross-document “State Contract” section (sequence init/reset, preemption, cancellation, rollback) and make it required before ADR-063 Phase 3 completion.

3. **Major — LoRA/QuaRot incompatibility is a core product risk and not consistently integrated into roadmap dependencies**
   - ADR-044: Runtime LoRA injection is incompatible with QuaRot-converted models (`docs/adr/ADR-044-quarot-rotated-quantization.md:143`).
   - ADR-060 reuses QuaRot for SliceGPT composition and still schedules LoRA recovery as P13 (`docs/adr/ADR-060-pruning-toolbox.md:575`, `:593-594`), while ADR-063’s v1 wedge includes serving APIs as core pitch (`docs/adr/ADR-063-serving-architecture.md:948-950`).
   - Impact: production claims can overpromote composability without clarifying adapter lifecycle limitations.
   - Fix: add explicit compatibility matrix (serve + prune + quant + LoRA) and a “no-op”/hard-fail behavior for unsupported combinations.

4. **Major — Performance moat claims are under-validated for investor-facing positioning**
   - ADR-063 lists competitors and states a strategic wedge (`docs/adr/ADR-063-serving-architecture.md:34-44`, `:65-72`), but gap inventory flags cross-framework comparison as currently unsubstantiated (`docs/gap_inventory_20260527.md:54`, `:117`, `:173`).
   - Impact: moat is conceptual, not evidence-backed, especially against established servers (llama.cpp/Ollama/vLLM).
   - Fix: require published baseline scripts and non-optimized run instructions before investor-facing claims.

5. **Major — Scope density indicates plan risk: ~40+ explicit phases across 5 ADRs in one quarter**
   - Phases in ADR-059 (`:891-896`), ADR-060 (`:581-594`), ADR-061 (`:460-488`), ADR-062 (`:56-63`, `:95-108`, `:254-267`, `:332-406`), ADR-063 (`:962-967`) total roughly 40+ phases.
   - Impact: too many coupled changes for a single 3-month lane unless staffing and parallel ownership are explicitly declared.
   - Fix: collapse into a two-track plan: **Seed Path** (serving+metrics+one optimization) and **Research Path** (prune/search/experiments), each with kill criteria.

6. **Minor-to-Major — ADR-047 and ADR-062 overlap without versioned ownership of prefix-cache integration status**
   - ADR-047 defines PrefixPageCache API and assumptions (`docs/adr/ADR-047-paged-kv-cache.md:40-49`, `:90-94`).
   - ADR-062 states prefix cache integration steps in paging/prefill context and calls out additional constraints (`docs/adr/ADR-062-metal-fa2-prefill.md:258-267`, `:364`).
   - Impact: risk of duplicated work or inconsistent ownership between research and engine tracks.
   - Fix: mark ADR-047 as historical design baseline and update ADR-062 as its implementation status owner; remove parallel ownership ambiguity.

## Update recommendations for legacy ADRs

- `docs/adr/ADR-010-attention-mechanisms.md`: add a deprecation/addendum that its 4-variant hard-coded flow is superseded by ADR-059; keep as historical implementation reference, not the current architecture strategy.
- `docs/adr/ADR-044-quarot-rotated-quantization.md`: cross-reference ADR-060/063 compatibility matrix and production constraints from the new exploration agenda so LoRA incompatibility is not an implicit, v0-only caveat.
- `docs/adr/ADR-047-paged-kv-cache.md`: add explicit status linkage that prefix-cache wiring in engine serve path moved into ADR-062 and ADR-063 milestones.
- `docs/adr/ADR-048-continuous-batching.md`: add update that HTTP/API serving milestones are no longer “deferred,” now in ADR-063, and clarify which batching semantics are unchanged.

## Why the strategy is strengthened

- Strong coherence between ADR-059/060/061 that turns architecture research from source edits into validated experiments.
- Explicit, phased performance path in ADR-062 maps well to Apple-Silicon bottlenecks (prefill, FA2, KV quantization).
- ADR-063 gives a practical seed product boundary with concrete API surfaces.

## Why the strategy is weakened

- Moat vs established servers is mostly narrative until cross-framework benchmark evidence is in place.
- Production-state contracts (state reset, LoRA compatibility, scheduler semantics) trail research innovation and could become integration blockers.
- 3-month time horizon is exposed by phase volume and dependency coupling.

## Requested follow-up before merge

1. Add one-page Critical Path Plan with explicit “done” criteria and owners for 60-second demo.
2. Add cross-ADR state/compatibility matrix (GDN + LoRA + QuaRot + quant + multi-tenant).
3. Reduce ADR-063 Phase 0–3 deliverable scope to a strict minimal slice.
4. Add benchmark evidence block (scripts + artifact links) before claiming competitive superiority.
5. Update ADR-010/044/047/048 with explicit status addenda (not just references).

Domain utility: **MEDIUM** — this review added cross-document consistency checks and execution-risk analysis that materially reduces strategic ambiguity, but no implementation validation was performed.
