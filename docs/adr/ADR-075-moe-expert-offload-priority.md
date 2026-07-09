# ADR-075: MoE Expert-Offload Priority (Metal)

**Status**: Proposed
**Date**: 2026-07-09
**Crate**: lattice-inference

> **Evidence basis**: The Decision below is grounded in this engine's own shipped source (the
> **Measured / source-verified reality** table). This is the fifth sequencing ADR in the bundle set
> (siblings ADR-071 GDN-state, ADR-072 weight-quant, ADR-073 KV-quant, ADR-074 MTP). Its verdict is
> **NEEDS-EXPERIMENT, reframed**: the ADR-075 question as commissioned — *rank the open MoE
> expert-offload levers* — presupposes a working, measured, resident MoE serving path whose next
> optimization is offload. **That precondition is false on the current head.** The single MoE preset
> is source-verified as not loadable on the 32–128 GB M-series/M-Max machines today (loadability on
> the 192–512 GB Ultra-class desktops is unproven pending measurement), its Metal dispatch has zero
> test/bench/measurement, and it runs only at f16. This ADR also **records a source-derived
> correction to a merged ADR**: ADR-053's headline "~15 GiB resident expert weights" figure is off by
> 4× — the real resident footprint is **60 GiB at f16** — because the Q4 basis it assumed was never
> implemented for MoE. A parallel external prior-art survey is folded as **[prior, unvalidated on our
> hardware]**; it is repo-blind and scoped to a different model family (DeepSeek-V4-Flash), and its
> arithmetic is unranked until re-derived on this engine's real numbers.

## Context

MoE (mixture-of-experts) expert-offload keeps a sparse model's expert weights out of scarce fast
memory and streams the cold ones from CPU RAM or disk on demand, so a model larger than device memory
can serve. This ADR was commissioned to rank the open offload levers (#682 the literal lever, #686
the epic, with FP4 #683 / FP8 #684 as adjacent footprint-shrinking alternatives). Ranking them turns
out to be **not yet a well-posed question**, for reasons that are themselves the finding.

Three facts shape everything below:

1. **Nothing measured on this engine touches MoE.** Lattice ships exactly one MoE preset
   (`qwen36_35b_a3b`: 256 routed experts, top-8, 1 shared, 40 layers, hidden 2048, moe_intermediate
   512). The large model it actually *serves and has measured* (18 GB Q4, ~4 tok/s decode) is
   `qwen36_27b`, which is **dense** (`num_experts: None`). So every tok/s, PPL, and parity number this
   engine has ever produced for a large model is for a non-MoE model. The MoE preset is even the
   struct `Default`, yet no CLI surface selects it.
2. **The Metal MoE dispatch is real and wired, but unmeasured and f16-only.** ADR-053's per-expert
   Metal dispatch (merged PR #57) compiles into the live per-layer forward loop (`encode_moe_ffn` has
   exactly one production call site and appears nowhere in a test). It has **zero committed bench,
   zero e2e/CI coverage, and no measured tok/s of any kind** — despite ADR-053's own R1 asking for
   post-implementation measurement, which never landed in the seven weeks since. And it runs **only at
   f16**: Q4/QuaRot, Q8-CPU, and Q8-NEON *all three* explicitly refuse MoE configs.
3. **The resident footprint is 60 GiB (f16), not 15 GiB, and the loader can't reach it anyway.**
   Because only f16 runs, routed-expert weights are 60 GiB resident — 4× ADR-053's "~15 GiB @ Q4"
   figure (§ amendment below). Worse, the engine constructor takes an already-fully-loaded
   `&ModelWeights`, so the CPU loader must materialize the entire model as owned f32 (~120 GiB) before
   any Metal buffer is built. The 32–128 GB M-series/M-Max machines (laptops, mini, base Studio)
   cannot absorb a 120 GiB f32 transient plus a growing 60 GiB f16 GPU allocation; only the 192–512 GB
   Ultra-class Studio desktops could hold that peak, and whether they do is unproven until Step 0
   measures the real peak RSS + Metal residency. **This is a loader-sequencing ceiling, prior to and
   independent of "should experts be offloaded."** As with weight/KV quant
   (ADR-072/073), Apple GPUs have no low-precision matrix unit, so any offload/quant of experts is a
   storage/bandwidth play, not a compute one — but here the prior question is whether the *resident*
   path works at all.

## Measured / source-verified reality (this engine)

Each row is tagged **source-read** (a structural fact read from merged code on `origin/main @
c9ae21a9`), **unit-test / gate-pinned** (a value asserted by a committed test), or **runtime-measured**
(a number from running this hardware), with its most durable public pointer. No runtime MoE number
exists — that absence is itself a load-bearing finding.

| # | Finding | How known | Pointer |
|---|---------|-----------|---------|
| E1 | **Exactly one MoE preset**: `qwen36_35b_a3b` — 256 routed experts, top-8, 1 shared, 40 layers, hidden 2048, moe_intermediate 512. Test pins `num_experts=256`, `per_tok=8`, `moe_intermediate=512`. | source-read + unit-test-pinned | `model/qwen35_config.rs:260-299`; test `:960-962` |
| E2 | **The MoE config is the struct `Default`** (`Qwen35Config::default() → qwen36_35b_a3b()`), yet no CLI/loader default selects it — the dense defaults are path-specific: Q4 chat/serve fall back to `qwen36_27b`, safetensors loading falls back to `qwen35_2b`, some benches to `qwen35_0_8b`. Reachable only by constructing the preset in Rust. | source-read | `qwen35_config.rs:154-158`; `bin/lattice.rs:1061` (safetensors→2b), `:1076` (Q4→27b); `model/qwen35/model.rs:37-42` (from_safetensors→2b) |
| E3 | **The served/measured large model, `qwen36_27b`, is dense** (`num_experts: None`) — a separate preset. It is the one labeled shipped (18 GB Q4, ~11 s load, ~4 tok/s M-series 32 GB); the MoE preset is labeled "partial: loader supported, not a polished serving path." | source-read + doc | `qwen35_config.rs:302-349`; `docs/models.md:66-67` |
| E4 | **Metal MoE dispatch (ADR-053) is implemented and wired into the live forward loop**: the `FeedForwardWeights::Moe` arm builds `MoeMetalBuffers` and calls `encode_moe_ffn` from the main encode path. Merged by PR #57. | source-read + gh-verified | `forward/metal_qwen35.rs:2968-3060` (build), `:9844` (only call site), `:10412-10705` (impl); `gh pr view 57` MERGED 2026-05-20 |
| E5 | **The Metal MoE path has zero runtime test/bench coverage.** `encode_moe_ffn` appears exactly twice in the file (definition + the single call site), never under `#[cfg(test)]`; PR #57's "8 new tests" are all CPU router-logic units. No `docs/bench_results/` MoE entry, no MoE bench bin, no e2e-parity MoE model. | source-read (absence) + gh-verified | grep: 2 occurrences of `encode_moe_ffn`; `e2e-parity.yml` names only `Qwen3.5-0.8B`; `docs/bench_results/` = 0 MoE hits |
| E6 | **Expert weights convert CPU-f32 → f16 at Metal buffer construction** (`make_buffer_f16` on `m.experts.gate_up_proj`). Not Q4, not Q8. | source-read | `forward/metal_qwen35.rs:2990-3016` |
| E7 | **All three lower-precision paths refuse MoE** — a triple-independent, consistently-enforced absence: Q4/QuaRot (`if cfg.is_moe()` → "MoE configs are deferred to v1"), Q8-CPU (`UnsupportedModel` for MoE checkpoints), Q8-NEON (`FeedForwardWeights::Moe(_)` → "Q8 NEON packing is dense-only"). So **f16 is the only precision the MoE path can run at today.** | source-read | `quant/quarot/convert.rs:367-369` (+test `:1282-1297`); `forward/cpu_q8.rs:44`; `forward/neon_forward.rs:244-248` |
| E8 | **The engine constructor takes a fully-loaded `&ModelWeights`** — `MetalQwen35Engine::new(weights: &ModelWeights, ...)`; the struct owns a single `Vec` of all 40 layers with `RoutedExperts.{gate_up,down}_proj: Vec<f32>`. The full f32 model must be CPU-resident before any Metal buffer is built. (API-shape confirmed; whether `from_safetensors` streams/drops internally was not ruled out — see the experiment's Step 0.) | source-read (API shape) | `forward/metal_qwen35.rs:2713-2714`; `model/qwen35/weights.rs:44-48,104-110` |
| E9 | **No offload/paging/streaming/mmap/LRU machinery of any kind for experts.** Repo-wide grep for `PagedMoe`/`ExpertPager`/`ExpertCache`/`ExpertLoader`/`PrefetchScheduler`/`expert.*offload|stream|page|mmap` = 0 hits. ADR-053's own "Deferred to v2" list names "expert weight swapping / LRU eviction" and "pre-gated routing prefetch" as not done. | source-read (absence) | grep (0 hits); `docs/adr/ADR-053-moe-metal-dispatch.md:133-141` |
| E10 | **No batched/prefill MoE path.** Six independent call sites (three in `batch_prefill.rs`, three in `metal_qwen35.rs`) return `UnsupportedModel("MoE ... not yet implemented")` for M>1; MoE runs serial decode-only (one token at a time). | source-read | `forward/batch_prefill.rs:346-350,379-383,954-956`; `forward/metal_qwen35.rs:4313-4316,6210-6213,7136-7139` |
| E11 | **The load-time memory guard is per-layer with no cross-layer accumulator**: each layer's routed-expert buffer is rejected if it exceeds `0.85 × recommendedMaxWorkingSetSize`, but the check never accumulates a running total across the 40 layers already allocated. Invisible at small scale; a latent gap for the 40-layer case. | source-read | `forward/metal_qwen35.rs:2976-2988` (no `total`/`cumulative` accumulator repo-wide) |
| E12 | **Prior MoE robustness work has landed**: MoE-config `.expect()` panic → structured error (#209 CLOSED), Q8-on-MoE panic → `UnsupportedModel` (#385 CLOSED), router NaN fail-closed (unit-tested). The CPU `moe_ffn_step` router is well-tested; only the *Metal dispatch* and *loadability* are unmeasured. | gh-verified + unit-test-pinned | `gh issue view 209/385` CLOSED; `model/qwen35/moe.rs:54-89,247-745` |

**Derived (not measured) — memory-capacity arithmetic** from E1/E6/E7. Per-expert element count
(gate_up + down) = `2·512·2048 + 2048·512 = 3,145,728` elements:

| Precision | Bytes/expert | Per layer (256 experts) | Total (40 layers) | Status on this head |
|-----------|-------------:|------------------------:|------------------:|---------------------|
| f32 (CPU-resident, current loader, E8) | 12.0 MiB | 3.0 GiB | **120 GiB** | happens today (loader materializes before any GPU buffer) |
| f16 (current Metal GPU buffers, E6) | 6.0 MiB | 1.5 GiB | **60 GiB** | happens today, *after* the 120 GiB f32 load |
| Q4 (ADR-053's quoted figure) | 1.5 MiB | 384 MiB | ~15 GiB | **never implemented** — Q4/QuaRot refuses MoE (E7) |

Routed-expert weight is ~93–97% of model bytes at either precision, so it is the correct offload
target *once a working resident path exists* — but at **f16 (60 GiB)**, not the Q4 (15 GiB) ADR-053
assumed.

## Amendment to ADR-053 (source-derived correction, recorded at priority level)

ADR-053 §"Expert weight memory" states, verbatim: *"At Q4 with the current config dimensions ... Total
routed expert weights across 40 layers: ~384 MiB × 40 ≈ **15 GiB**"* (`ADR-053:48-51`), and its R2
risk analysis reasons from *"~15 GiB for routed expert weights ... borderline on 32 GB"*
(`ADR-053:252-254`). **This figure is superseded.** The Q4 expert format it assumes was never
implemented — Q4/QuaRot, Q8-CPU, and Q8-NEON all refuse MoE at three independent enforcement points
(E7) — so the only precision the MoE path runs at is f16, making the resident routed-expert footprint
**60 GiB, 4× the ADR-053 figure**. Causal mechanism: ADR-044 deferred per-expert QuaRot rotation to
"v1," the deferral was enforced (not just documented) in the converter, and no subsequent MoE
quantizer landed. Consequence: ADR-053-R2's "borderline on 32 GB" conclusion is wrong by 4× — with a
120 GiB f32 load peak (E8) preceding a 60 GiB f16 GPU allocation, the model is **not loadable on the
32–128 GB M-series/M-Max machines** ADR-053-R2 targeted, not merely borderline; only the 192–512 GB
Ultra-class Studio desktops could hold that peak, and that is unproven until Step 0 measures it. This amendment lands here (not by
editing ADR-053, which is an immutable record); ADR-053's dispatch-count and routing design are
unaffected — only its resident-memory arithmetic and the 32 GB feasibility claim are corrected.

## Prior / unvalidated on our hardware

Folded from the external survey (`fleet_atlas_lat_moeoffload_001` harvest) as **data**. The survey is
an internal research artifact not part of this repo, so its repo-blindness is stated below as
**repository-verifiable facts** — a reviewer can grep this tree to confirm each cited path does not
exist; the packet-internal quotes are corroborating attribution only, not the load-bearing proof. It
is **repo-blind and scoped to a different model family**, decisively:

- **Repo-blind (repository-verifiable)**: the survey's proposed `patch.diff` and experiment commands
  target `crates/engine/src/moe/*`, a `.gguf` loader, and binaries like `moe_trace` / `cache_sim` —
  **none of which exist in this tree**: there is no `crates/engine` crate (the crates are
  `embed`/`fann`/`inference`/`transport`/`tune`), no GGUF loader anywhere in the source, and no such
  binaries under `crates/inference/src/bin/`. The real MoE code is
  `crates/inference/src/model/qwen35/moe.rs` and `forward/metal_qwen35.rs`. (The packet's own
  `MANIFEST.md` corroborates this — conceding its diff is "approximate ... anchored to plausible
  modules" and its `QUESTIONS.md` opening by asking for the repo SHA and module paths — but the reject
  rests on the grep-confirmable absences above, not the unpublished packet text.)
- **Wrong model family**: its primary target throughout is DeepSeek-V4-Flash-DSpark (284B/13B-active,
  FP4 experts, ~12 MiB/expert) — a model lattice has **no config, loader, or plan for**. Its
  "moderate model" comparison uses a **1.5 MiB/expert** figure that matches lattice's *aspirational
  Q4* size, not its *actual f16* 6.0 MiB (4× larger). Its "90–95% hit-rate workable" conclusion must
  be re-derived at the real bundle size (its own formula `h ≥ 1 − budget·bw/(invocations·bundle)`
  pushes h toward the 98–99% regime when bundle bytes are 4× larger) — so the number is **unranked,
  not refuted**.
- **One convergent, useful conclusion**: the survey independently reaches *"the first prototype should
  be an in-memory MoE implementation on a model that already fits, with exact routing parity and
  throughput measurements, before any disk streaming is added"* — which matches this engine's
  source-read verdict exactly (E4/E5: the resident dispatch exists but is unmeasured/untested). Its
  `mmap`-skepticism ("GPU-visible page faults are the wrong latency shape for decode") is reasonable
  general guidance, not a finding about this codebase (which has no mmap expert path at all, E9). Its
  bibliography (llama.cpp disk-paging RFC, Mixtral-offloading, Fiddler, KTransformers, ZeRO-Infinity)
  is a legitimate literature survey — useful background for whoever eventually designs #682, not a
  lattice-specific plan.

## Decision

**Top-level verdict: NEEDS-EXPERIMENT, reframed — gated, not ranked.** Ranking expert-offload against
FP4 #683 / FP8 #684 / hybrid-attn #685 is not yet well-posed: offload's entire value proposition
("stream from disk instead of holding resident") cannot be evaluated without first knowing whether the
*resident* path works and what it costs — which nobody has measured (E4/E5). This is **not** the
"REFUTED — wrong architecture" honest-negative of the KV-quant/MTP recons (MoE *does* exist and *is*
wired). It is "the lever is real, but the measurement floor beneath it does not exist yet."

### Rankable now (no experiment)

- **Nothing in the offload family is rankable.** #682 (the literal offload lever), #686 (the epic),
  #683/#684 (footprint-shrinking substitutes) all sit above an unmeasured, currently-unloadable
  resident MoE path. Ranking them on the survey's borrowed numbers (§ prior) would be ranking on
  literature for a different model family.
- **The ADR-053 memory correction IS rankable now** (§ amendment) — it is source-derived, not gated,
  and it reframes the whole area: the resident footprint is 60 GiB f16 and the model is not loadable
  today, so the *first* work is a loadable, measured resident path, not offload.

### NEEDS-EXPERIMENT — the minimal experiment (the gate), in priority order

**Step 0 — loader-sequencing confirmation (prerequisite, a load-path question not an offload one).**
Confirm whether `MetalQwen35Engine::new` genuinely requires the full f32 `ModelWeights` resident
before any Metal conversion (E8 traces the constructor's API shape; a streaming/drop-per-layer
optimization *inside* `from_safetensors` was not ruled out by source-read alone). Instrument an actual
load attempt of the MoE preset. If confirmed, this is a loader-sequencing ceiling independent of
offload; fixing it (stream layer-by-layer, drop CPU f32 after each layer's Metal buffer is built)
lowers the peak-memory floor before any offload code is written.

**Step 1 — the actual minimal experiment (synthetic small-scale MoE, no offload code).** Drive a
synthetic small MoE config (e.g. 4 layers, 16 experts, top-2, hidden 256 — values the test fixtures at
`moe.rs:437-507` already construct) through the *real* `MetalQwen35Engine` (not the CPU-router-only
path already tested, E12) to get the **first-ever**: (a) Metal MoE dispatch **correctness** — CPU
`moe_ffn_step` output vs `encode_moe_ffn` output on identical synthetic weights, tolerance per the
e2e-parity gate's bar; (b) Metal MoE dispatch **latency** — extend the `bench-compare` / criterion
harness with a `moe_dispatch` group; (c) confirmation the per-layer 0.85× guard behaves at small scale
(and flag E11's no-accumulation gap for the 40-layer case in the PR). Rigor bar per house convention:
n≥3, report CV, GPU flock (`/tmp/lion-metal-gpu-test.lock`) held for the duration, **commit the
artifact** to `docs/bench_results/` (zero MoE entries today, E5) — not a chat-log number.

**Step 2 — re-derive the offload arithmetic (only after Step 1 produces a real number).** Plug
lattice's actual **6.0 MiB/expert f16** bundle size (§ derived) and Step 1's measured dispatch latency
into the survey's hit-rate-vs-bundle formula (§ prior) to get a real (not DeepSeek-V4-borrowed)
required-hit-rate target. *Then* #682's LRU/mmap design becomes decidable.

**Do not** propose a full V4-Flash-scale (284B) experiment: lattice has no config/loader/plan for that
family (§ prior), and #686 itself scopes it as "not near-term, groundwork only."

**Adjacent levers, same gate.** FP4 #683 / FP8 #684 shrink the resident footprint (a substitute or
complement to offload) and are gated on the same measurement floor — a real MoE tok/s number is the
prerequisite for ranking *any* MoE memory lever. StorageModePrivate #179 is an adjacent Metal
memory-management lever (current MoE buffers are `StorageModeShared`) relevant to a future residency
redesign, not MoE-specific.

**Forward instruction (mirrors ADR-074's S-row rider).** When Step 1's bench lands, its measured MoE
dispatch tok/s and peak-memory numbers amend both this ADR's derived table and ADR-053's corrected
resident-memory figure with runtime-measured rows — measured floor, not silence.

## Consequences

- **Positive.** The ADR leads with a source-derived correction of ADR-053's central memory figure
  (4× under, with the causal mechanism), refuses to rank offload on a measurement floor that does not
  exist, and names the exact minimal experiment (a synthetic-MoE dispatch bench) that would produce
  the first real MoE number on this engine. It keeps the offload family correctly gated and the
  survey's arithmetic unranked rather than importing DeepSeek-V4 numbers.
- **Cost.** Step 1 needs a synthetic-MoE dispatch bench harness built (none exists — E5). Step 0 needs
  a load-path instrumentation of the MoE preset. Both are ≤1–2 session efforts and prerequisite to
  *any* MoE work, not just offload.
- **Risk.** If Step 1 reveals `encode_moe_ffn` is incorrect (never verified against the CPU path,
  E5) or the loader ceiling (E8) is unfixable without a residency redesign, the MoE serving path needs
  rework before offload is even a question. That is a legitimate outcome the cheap experiment is
  designed to surface early, not a failure — and far cheaper to learn from a 4-layer synthetic than
  from a failed 60 GiB load.

## Follow-ups

- #682 (MoE expert offload — the literal lever) → **gated** behind Step 0 + Step 1; do not start LRU/mmap design before a real resident MoE number exists.
- #686 (epic: run MoE that exceeds RAM, V4-Flash class) → **roadmap-gated**, its own "not near-term, groundwork only" scope; needs a DeepSeek-family loader that does not exist.
- #683 (FP4) / #684 (FP8) → **adjacent, same gate** — footprint-shrinking substitutes for offload; also blocked on the resident-MoE measurement floor. FP4 is additionally the path that could bring MoE experts below f16 (the absence E7 documents).
- #179 (StorageModePrivate) → adjacent Metal memory-management lever for a future residency redesign.
- **ADR-053** → resident-memory figure amended here (60 GiB f16, not 15 GiB Q4); its per-layer memory guard has no cross-layer accumulator (E11, latent 40-layer gap); its R1 post-implementation measurement is still outstanding (E5) — Step 1 discharges it.
- **New issue candidate** (not filed by this ADR): the loader-sequencing ceiling (E8, 120 GiB f32 peak before Metal construction) is a distinct correctness/feasibility bug from offload; Step 0 confirms whether it warrants its own tracking issue.
