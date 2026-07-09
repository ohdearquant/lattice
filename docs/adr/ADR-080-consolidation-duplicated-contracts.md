# ADR-080: Consolidation of Duplicated Numeric-Contract Helpers (Softmax, HTTP Serving, Decode Policy, GEMM Validation)

**Status**: Proposed
**Kind**: Aspirational
**Date**: 2026-07-09
**Crate**: lattice-inference (`crates/inference/src/attention/`, `src/forward/`, `src/bin/`,
`src/model/qwen35/`)
**Research**: Internal duplication audit, run 2026-07-09 (audited at `13c8de8a3`; adversarially verified at the audited commit; re-verified at `origin/main @ 0699e60cc`)
**Issues**: #739, #740, #741 (attention-softmax); #744, #745, #746 (http-serve); related
non-cluster audit items tracked in #764-#777 (per-family checklists and standalone items)
**Depends on**: ADR-058 (CPU performance regression CI), ADR-064 (CI gate taxonomy), ADR-066 (e2e parity gate)

## Context

A structural duplication sweep over `crates/inference` (and the sibling `crates/tune`,
`crates/fann` for cross-checking) built a per-family map of every site implementing the same
nominal numeric operation, then an adversarial verification pass read each candidate divergence against the live source
and re-verified the survivors against a later commit. The sweep covers ten operation families;
this ADR addresses the four families where the duplication itself — not a single bug — is the
root cause of recurring, independently-discovered defects: attention softmax, HTTP request/response
handling across the two OpenAI-compatible server binaries, autoregressive decode-policy
application, and GEMM argument validation.

### Why these four, not all ten

The other six families (`kv-cache`, `lora-apply`, `norm-elementwise`, `rope-position`,
`tokenizers`, `weight-loading`) also contain duplicated sites, but in each of those the audit's own
conclusion is either "no canonical site is warranted — the duplication is backend/lifecycle
separation, not drift" (e.g. KV-cache position/geometry across CPU/Metal/paged storage; GDN
snapshot copies across CPU clone / Metal slot blit / cross-turn persistence), or the fix is already
scoped as a single-site defect with its own issue (e.g. #736, #735 seeds noted directly in the
weight-loading map). The four families below are the ones where **the same nominal contract is
implemented three to fourteen times, each copy has independently drifted from a fail-closed or
validated baseline, and at least one drift has already reached production as a filed, independently verified
defect** — the signature of a missing shared contract, not of intentional backend specialization.

### Evidence summary (from the duplication maps)

| cluster | nominal operations mapped | site counts (largest → smallest) | held findings (post-verification) |
|---|---|---|---|
| attention-softmax | 6 | 6, 6, 4, 4, 3, 3 | ATTENTIONSOFTMAX-P2-002, -005, -006, -007 (+ filed #739, #740, #741) |
| http-serve | 6 | 3, 3, 3, 3, 2, 2 | HTTPSERVE-P2-005 (+ filed #744, #745, #746) |
| sampling-decode | 6 | 10, 6, 4, 4, 4, 1 | SAMPLINGDECODE-P2-002, -003 |
| matmul-gemm | 10 | 14, 7, 6, 5, 5, 5, 4, 4, 3, 1 | MATMULGEMM-P2-001, -002, -003, -004, -005 |

All 12 held findings and the six filed issues above were confirmed against live source at the
audited commit and re-checked against a commit 51 revisions later; none was a stale artifact of
the original audit commit.

### Provenance

The duplication sweep and the first adversarial verification pass ran at `13c8de8a3` — a local
checkout later found to be 51 commits behind `origin/main`. Every finding whose cited file changed
in that 51-commit window was then re-verified at `origin/main @ 0699e60cc`; findings in unchanged
files transfer directly. Three audit findings (outside this ADR's held set) were fixed by
concurrent work in that window (#656, #657) and were not filed; none of the 12 held findings cited
in this ADR was among them. Six of the findings driving this ADR are already filed as standalone issues
(#739-#741, #744-#746); the 12 held findings are intentionally NOT filed as issues — they are
resolved by this ADR's clusters and tracked in the post-ADR implementation checklists.

## Decision

**Consolidate within existing crates. Do not create a new crate.** Extract one small, checked,
module-level helper per cluster inside `lattice-inference` (the crate that owns every duplicated
site in all four clusters). Keep every backend-specific kernel — CPU scalar, CPU SIMD (NEON/AVX2),
Metal, WGPU — as a separate implementation. The scalar/CPU implementation in each cluster is named
the **numeric reference** ("formula oracle"): it is the specification other backends are checked
against in parity tests, not itself replaced by a cross-backend abstraction.

A new crate is justified only when a helper's canonical dependency-free logic is needed by two or
more *crates* that do not already share a dependency edge. None of the four clusters meets that
bar: attention-softmax, sampling-decode, and matmul-gemm are entirely intra-`lattice-inference`;
http-serve spans two binaries (`lattice.rs`, `lattice_serve.rs`) that already live in the same
crate. Splitting a crate to share ~50-150 lines of validation logic between sibling modules of the
same crate would invert the project's find-and-modify-before-create principle, add a publish-order
dependency for no cross-crate consumer, and contradict the "pure-Rust, minimal-dependency-surface"
posture in `AGENTS.md`.

### C1: Softmax fail-closed row contract (attention-softmax)

**Evidence.** Six nominal softmax operations recur across 4-6 sites each: materialized masked CPU
attention (`forward/cpu/softmax.rs`, `attention/standard.rs`), online causal softmax
(`attention/decode.rs`, `attention/gqa.rs`, `generate.rs`, `attention/flash_causal.rs`), Qwen3.5
cached-decode/batched-prefill softmax (`model/qwen35/forward.rs`, `forward/cpu_f16.rs`,
`forward/cpu_q8.rs`, `forward/neon_forward.rs`, `forward/batch_prefill.rs`,
`forward/metal_qwen35.rs`), Qwen3 secondary full-prefill softmax (`model/qwen.rs`), standalone
attention-variant softmax (`attention/differential.rs`, `attention/native_sparse.rs`,
`vision/vit.rs`), and GPU softmax (`forward/metal.rs`, `forward/gpu/inner/shaders.rs`).

**Canonical row contract**, taken from the two sites the map identifies as authoritative
(`attention/gqa.rs` and `attention/decode.rs`): exact-exp arithmetic for finite deltas, a **zeroed
row** when the row's accumulated denominator is non-positive or the row contains a NaN or +inf
score (fail-closed), and no floor-clamping of structural `-inf` mask values into finite mass.

**Divergences the map found against that contract**: `fast_exp` in `forward/cpu/softmax.rs` clamps
`-inf` to a finite floor (structural masks get positive weight instead of exact zero); the same
file zeros only the offending lane on a NaN score instead of the whole row; both Flash-causal
implementations (`attention/flash_causal.rs`) discard an individual NaN key and normalize the
remaining keys instead of failing the row closed; the two Qwen3 secondary loops
(`model/qwen.rs:1126`, `:1449`) leave NaN in the output instead of zeroing; `differential.rs`,
`native_sparse.rs`, and `vision/vit.rs` all propagate NaN for a NaN/+inf/all-invalid row where GQA
and decode zero it; the live Metal kernels (`forward/metal.rs`) already zero on a non-positive
denominator, but an unselected legacy Metal kernel with only a finite sentinel remains in the
source tree.

**Extraction plan**: add one allocation-free, `#[inline]` row-finalizer helper in
`crates/inference/src/attention/` (co-located with the two already-canonical implementations) that
takes row scores and returns the fail-closed normalized row per the contract above. Every CPU dtype
variant (f32/f16/Q8/NEON), the Flash-causal tiled and fallback paths, the two Qwen3 secondary loops,
and the three standalone-attention-variant sites call it instead of reimplementing row reduction.
Metal and WGPU kernels stay separate GPU code but gain an explicit non-finite-row contract test
that checks the *kernel output* against the same reference. The unselected legacy Metal
score/softmax kernel is deleted (dead code, not a live divergence to fix).

**Resolves**: ATTENTIONSOFTMAX-P2-002 (exact zero for -inf-masked CPU MHA positions),
-005 (Flash-causal NaN-key fail-close), -006 (differential/native-sparse hardening),
-007 (Qwen3 secondary softmax duplicate removal); plus filed #739 (Qwen3.5 f32/F16/batched-prefill
fail-close), #740 (CPU MHA NaN-lane drop), #741 (ViT softmax non-finite guard) — all of which land
on the same shared helper once it exists, rather than being patched independently three more times.

### C2: Dual-HTTP-server shared serving module (http-serve)

**Evidence.** `lattice.rs` (the unified CPU/Metal/Q4 server) and `lattice_serve.rs` (the
app-facing daemon) independently implement: router construction and endpoint exposure, OpenAI
chat-request normalization and token-budget limits, SSE disconnect cancellation, and completion
termination (`finish_reason`) serialization — 2-3 sites per operation, split across the two
binaries. #656 already unified one slice of this (fail-closed roles/content-parts,
`max_completion_tokens` field, config-derived context limits landed via commit `e65b88789`,
confirmed by re-verification at `0699e60cc`).

**Divergences confirmed live** (re-verified at `origin/main @ 0699e60cc`): `lattice.rs` documents
in its own source comment that it has no disconnect-cancellation and keeps generating after a
client disconnects; `lattice_serve.rs`'s `CancelOnDrop`/`should_cancel` machinery genuinely covers
prefill and decode and is unaffected by #656. `lattice_serve.rs`'s `spawn_worker` structurally
discards the engine's `stopped`/`stop_reason` fields (`.map(|out| (out.prompt_tokens,
out.completion_tokens))`), so both its SSE and JSON responses discard the ENGINE-reported stop cause (stop-string and EOS terminations both serialize as `"stop"`; only a locally computed length branch survives), while `lattice.rs` carries the engine's actual stop state
through. The two binaries also expose different router surfaces (only `lattice_serve.rs`'s
`/v1/models` is actually installed and reachable from the macOS app).

**Extraction plan**: move router construction, request normalization/validation, and response
termination serialization into one shared serving module inside `crates/inference/src/bin/`
(or a private `serve` sub-module of the crate, not a new crate — both binaries already live in
`lattice-inference`), and have each binary retain only its backend-specific generation wiring
(CPU/Metal dispatch for `lattice.rs`, the daemon/queue model for `lattice_serve.rs`). Disconnect
cancellation is added as a CPU-generation-aware API so `lattice.rs` can adopt the same
cancellation contract `lattice_serve.rs` already has, rather than each binary growing its own
cancellation state machine.

**Resolves**: HTTPSERVE-P2-005 (unify endpoint and error contracts); plus filed #744 (disconnect
cancellation), #745 (max_tokens=0 fail-closed, scope narrowed during audit triage), #746
(`finish_reason` hardcoding) — all three are instances of the same "two independently-maintained
request/response paths" root cause this cluster targets.

### C3: Backend-neutral decode policy (sampling-decode)

**Evidence.** The Qwen autoregressive decode-policy loop recurs across 10 sites:
`model/qwen35/generation.rs` (both canonical CPU entry points), `forward/cpu_f16.rs`,
`forward/cpu_q8.rs`, `forward/neon_forward.rs`, `forward/batch_prefill.rs`, and four
`forward/metal_qwen35.rs` entry points (direct, streaming, MTP, multimodal-adjacent). #657 already
fixed `stop_strings` propagation on four Metal decode loops (landed 2026-07-04, confirmed in
re-verification at `0699e60cc`), narrowing but not closing the family.

**Divergence.** The two canonical CPU generation entry points apply the full `GenerateConfig`
policy — `stop_strings` and reasoning-budget — and fail closed on unsupported grammar. The
F16/Q8/NEON/batch-prefill CPU siblings and the direct/multimodal Metal siblings each silently omit
one or both controls instead of either applying them or rejecting the config as unsupported.
Quantized and batch paths reject unsupported grammar and logprobs (an existing fail-closed
precedent) but have no equivalent guard for unsupported `stop_strings`/`reasoning_budget` — the gap
is inconsistent application, not a missing capability everywhere.

**Extraction plan**: factor the backend-neutral parts of decode policy (stop-string matching state,
reasoning-budget accounting, logprobs formatting) into a small policy struct in
`crates/inference/src/model/qwen35/generation.rs` that wraps a backend-specific per-step forward
callback. Every CPU dtype variant and every Metal entry point either drives its decode loop through
that wrapper, or — where a field genuinely cannot be supported on that backend today (e.g. an MTP
fast path) — explicitly rejects the unsupported `GenerateConfig` field at that entry point instead
of silently ignoring it. This is the same "apply or fail closed" contract the audit found already
in place for grammar/logprobs; the extraction generalizes it to stop_strings/reasoning_budget.

**Resolves**: SAMPLINGDECODE-P2-002 (canonical first-wins/repetition-aware greedy selection shared
by MTP and self-speculative paths), -003 (fail-closed generation controls on the alternate CPU
decode loops that currently silently omit stop_strings/reasoning_budget).

### C4: One checked GEMM argument validator (matmul-gemm)

**Evidence.** Ten nominal GEMM-family operations recur across the CPU/GPU boundary: dense f32
row-major GEMM at 14 sites (`forward/cpu/matmul.rs`, `blas.rs`, `tiled.rs`, `tiled_neon.rs`,
`tiled_avx2.rs`, `arch_kernels.rs`, `gpu_gemm.rs`, `gpu/inner/shaders.rs`, `metal_gemm.rs`,
`metal.rs` ×2, `metal_qwen35.rs`, `vision/vit.rs`); scaled SGEMM at 3 sites (`forward/cpu/blas.rs`
×2, `attention/gdn_fused.rs`); mixed f16/BF16 projection at 5 sites; Q8 projection GEMV/GEMM at 7
sites; asymmetric Q4 Metal projection at 5 sites; BitNet ternary matvec at 4 sites; GDN
`Sᵀ @ vector` and decay-update at 6 sites; training/autodiff dense projection and transpose-VJP at
5 sites; fann's dense-layer `y = Wx + b` at 4 sites; embed-crate's single delegating call site.

**Canonical validation contract**, from `forward/cpu/matmul.rs`: overflow-first argument checking
(reject before compute, not after), release-active (not `debug_assert!`-gated), and an explicit
documented allow-list for oversized scratch-buffer prefixes (a caller may pass a buffer longer than
the logical operand without that being treated as a shape error).

**Divergence.** The canonical CPU boundary is the only site in the family with all three
properties. Standalone WGPU and Metal wrapper functions perform no argument validation at all
(`metal_gemm.rs`). The generic scaled-SGEMM helper (`forward/cpu/blas.rs`) was never upgraded to
the release-active contract; on a short-slice input its non-macOS scalar copy fails through Rust's
own bounds checks while its macOS-`cfg`-gated Accelerate-FFI copy crosses into unsafe code with
dangling/undersized pointers instead of being rejected beforehand. The BitNet ternary matvec's
release scalar path silently truncates a too-short input slice (its own safe wrapper independently
re-derives the same check the unsafe NEON kernel needs, rather than sharing one validator). The
training SwiGLU forward silently truncates a short activation slice via `zip` where the canonical
`matmul_bt` and the materialized GQA training forward both reject the same malformed shape.
SIMD/BLAS/GPU reduction-order differences within f32 tolerance are explicitly **not** part of this
divergence — they are intentional and stay.

**Extraction plan**: add one checked GEMM-argument validator (overflow-first, release-active,
oversized-scratch-prefix-aware, matching the documented `forward/cpu/matmul.rs` contract) as a
module-level function in `crates/inference/src/forward/cpu/`, and call it from every safe entry
point across CPU, standalone WGPU, and standalone Metal wrapper functions before dispatch. Backend
kernels (SIMD tile variants, Metal shaders, WGPU shaders) stay separate; only the pre-dispatch
argument check is shared. The BitNet ternary matvec gets its own `validate_ternary_matvec_args`
(a distinct contract — packed ternary layout, not row-major f32) called from every safe entry point
ahead of the unsafe NEON kernel. Undispatched rectangular Metal GEMM kernel variants — confirmed by
the map to be redundant, non-selected implementation surface rather than a live output divergence —
are deleted or explicitly relabeled test/benchmark-only so they stop appearing as apparent
duplication in future sweeps.

**Resolves**: MATMULGEMM-P2-001 (validate scaled SGEMM before the macOS FFI call), -002 (validate
standalone WGPU/Metal GEMM slices before dispatch), -003 (promote BitNet scalar GEMV checks to
release-active validation shared with the safe wrapper), -004 (reject short activations in the
training SwiGLU projection), -005 (unify oversized-input handling across f32/f16/Q8 GDN
projections).

## What we are NOT doing

- **No new crate.** Every extracted helper lands inside `lattice-inference`, the crate that already
  owns 100% of the duplicated sites in all four clusters. A new crate would add a publish-order
  dependency edge for zero cross-crate consumers.
- **No cross-crate kernel unification.** `lattice-fann`'s dense-layer GEMM and
  `lattice-inference`'s attention/projection GEMMs stay independent implementations; only
  `lattice-inference`'s own internal sites are consolidated in C4.
- **No numeric behavior changes beyond the documented fail-closed fixes.** SIMD-vs-scalar-vs-Metal
  reduction-order differences (f32 reassociation) are not touched by this ADR — they remain
  documented tolerance, consistent with the project's existing parity-testing posture
  (`e2e-parity.yml`, ADR-058). Where two backends already agree on output for valid input (e.g. Q8
  dispatch geometry, dense-GEMM row-major convention), no behavior changes; only the missing
  argument-validation and fail-closed-row guards are added.
- **No backend removal.** CPU scalar, SIMD, Metal, and WGPU kernels for every cluster are kept as
  separate implementations behind the shared validated/checked entry point. The scalar
  implementation in each cluster becomes the named numeric reference for parity tests; it does not
  replace the optimized kernels.
- **No unification of the two HTTP binaries into one.** `lattice.rs` and `lattice_serve.rs` remain
  separate binaries with separate backend-dispatch responsibility (C2); only the shared
  request/response/router logic moves into one module both binaries consume.

## Sequencing & risk

Each cluster ships as its own PR series, in the order C1 → C2 → C3 → C4 (softmax first: it has the
most already-filed, independently verified issues and the smallest blast radius; GEMM last: it touches the
widest span of call sites and the training/autodiff paths).

1. **Bench-compare is required** on any PR touching `crates/inference/src/forward/` or
   `crates/inference/src/attention/` per ADR-058 / the repository's `make bench-compare` process —
   this applies to C1, C3, and C4, and to the CPU-generation-cancellation change in C2. Paste
   before/after numbers in every such PR description; a shared helper that changes hot-path
   instruction count is exactly the kind of change this gate exists to catch.
2. **Every extracted guard needs a mutation-sensitive regression test**: for each fail-closed fix
   (NaN row zeroing, oversized-slice rejection, disconnect cancellation, `finish_reason` fidelity),
   the test must fail when the guard is reverted (reverse-apply the diff, `touch` the source file
   to force a rebuild, confirm red, then restore and confirm green) before the PR is considered
   complete. A regression test that still passes with the fix reverted is not evidence of anything.
3. **Sibling-invocation-path grep is the per-cluster completion gate.** Before closing out each
   cluster's PR series, grep the touched files for any remaining site that constructs the same
   operation independently (a second subprocess/router builder, a second inline softmax loop, a
   second matvec bounds check) — this class of bug (a fix landing on one path while a copy-paste
   sibling in the same file goes unguarded) has recurred five times in this repository's history
   and is exactly the failure mode this ADR exists to close off structurally.
4. **Risk**: the http-serve and sampling-decode clusters touch production-facing serving behavior
   (`finish_reason`, disconnect handling, stop-string matching) that external OpenAI-API clients may
   depend on for exact wire compatibility — the e2e-parity CI gate (ADR-066/e2e-parity.yml) and the
   existing local-helper unit tests on `lattice.rs`'s request normalization are the primary
   regression backstop; no new CI surface is proposed by this ADR.
5. **Risk**: the matmul-gemm cluster's argument validator sits on the hottest code paths in the
   engine (every GEMM/GEMV call in prefill and decode). The bench-compare requirement in item 1
   above is the mitigation; if a measurable regression appears, the validator must be proven
   branch-predictable/inlined away for the common valid-shape case rather than the fail-closed
   check being dropped.

## Alternatives considered

1. **Leave each duplicated site as-is and file/fix each held finding independently.**
   Rejected. Six of the twelve held findings driving this ADR are already independently-filed
   issues (#739-#741, #744-#746) that are each an instance of the same missing shared contract;
   fixing them one at a time guarantees the next new call site repeats the same drift, which is
   exactly what happened across the four to fourteen sites already mapped per cluster.

2. **Extract a new `lattice-kernels-core` (or similar) crate for all four clusters.**
   Rejected. None of the four clusters has a cross-crate consumer today — all duplicated sites in
   all four clusters live inside `lattice-inference`. A new crate would add a publish-order
   dependency, a version-bump surface, and packaging overhead (per the repo's `make publish`
   dependency-DAG process) with no crate outside `lattice-inference` to justify it.

3. **Unify backend kernels themselves (e.g. one softmax/GEMM implementation for CPU and Metal).**
   Rejected. The audit explicitly found that CPU/SIMD/Metal/WGPU reduction-order and
   precision-staging differences (e.g. f32 vs half-tile staging in Q4 Metal kernels) are
   intentional performance/precision trade-offs, not defects. Only the argument-validation and
   fail-closed-row *contracts* are duplicated and drifting; the arithmetic kernels themselves are
   correctly backend-specific and are kept separate.

4. **Do nothing beyond the individually-filed issues, and let periodic audits re-discover the same
   duplication shape.** Rejected. The duplication maps show the same nominal operation re-appearing
   with fresh divergences at every new call site added since the family was last touched (e.g. the
   Q4/Metal GEMV family gained a new asymmetric variant since the last review); without a shared
   checked entry point, each new caller is a new opportunity for the same drift.

## Consequences

### Positive

- Each of the four clusters collapses N independently-drifting copies of a fail-closed or validated
  contract into one checked helper plus N backend-specific kernels that consume it, closing the
  root cause behind six already-filed issues and seven additional held findings in one motion per
  cluster rather than one motion per site.
- The scalar/CPU reference implementation in each cluster becomes an explicit, named numeric oracle
  usable by future parity tests, rather than an implicit convention scattered across the crate.
- Future new call sites (new dtype variants, new Metal entry points, new HTTP endpoints) get the
  fail-closed/validated contract by construction instead of by manual re-implementation.

### Negative

- Four PR series touch some of the hottest paths in the engine (softmax, GEMM, decode loop); each
  requires bench-compare evidence and mutation-sensitive tests before merge, which is real
  incremental cost over a narrower single-issue fix.
- The http-serve and sampling-decode extractions touch production request/response wire behavior;
  a mis-scoped refactor risks a wire-compatibility regression that the e2e-parity gate may not
  fully cover (it checks greedy token agreement, not full HTTP response shape).
- This ADR does not itself fix the six other duplication families the sweep mapped
  (`kv-cache`, `lora-apply`, `norm-elementwise`, `rope-position`, `tokenizers`, `weight-loading`);
  those remain tracked as individual issues where the audit found a genuine defect, without a
  cluster-level consolidation motion, because the audit's own conclusion in each of those six is
  that the duplication is intentional lifecycle/backend separation rather than contract drift.

### Risks

- See "Sequencing & risk" above for the concrete per-cluster mitigations (bench-compare,
  mutation-sensitive tests, sibling-invocation-path grep, e2e-parity as the wire-compatibility
  backstop).

## Implementation plan

### Phase 1: C1 — softmax fail-closed row helper

Extract the row finalizer in `crates/inference/src/attention/`; wire it into every CPU dtype
variant, Flash-causal, and the three standalone-attention-variant sites; add non-finite-row
contract tests for the live Metal kernel; delete the unselected legacy Metal softmax kernel. Closes
#739, #740, #741 and the four held ATTENTIONSOFTMAX findings in this ADR.

### Phase 2: C2 — shared serving module

Extract router construction, request normalization, and termination serialization into one module
consumed by both `lattice.rs` and `lattice_serve.rs`; add CPU disconnect-cancellation as an explicit
generation API. Closes #744, #745, #746 and HTTPSERVE-P2-005.

### Phase 3: C3 — backend-neutral decode policy

Factor stop-string/reasoning-budget/logprobs policy into a wrapper around backend-specific forward
callbacks; add explicit fail-closed rejection for any `GenerateConfig` field a given backend cannot
honor. Closes SAMPLINGDECODE-P2-002 and -003.

### Phase 4: C4 — GEMM argument validator

Add the checked GEMM validator and the BitNet ternary validator; wire every safe CPU/WGPU/Metal
wrapper entry point through them; delete or relabel undispatched rectangular Metal kernels; fix the
training SwiGLU short-slice truncation. Closes MATMULGEMM-P2-001 through -005.

### Phase 5: Founder sign-off and merge sequencing

Each phase lands as its own PR series with bench-compare evidence and mutation-sensitive tests
before merge, per "Sequencing & risk" above. No phase blocks another; they can proceed in parallel
worktrees once this ADR is signed off, but land in the order given to front-load the
lowest-risk/highest-already-confirmed cluster first.

## References

- Internal duplication audit, run 2026-07-09 (audited at `13c8de8a3`; verified and re-verified at `0699e60cc`) — the four cluster evidence tables above
- `crates/inference/src/attention/gqa.rs`, `crates/inference/src/attention/decode.rs`
- `crates/inference/src/bin/lattice.rs`, `crates/inference/src/bin/lattice_serve.rs`
- `crates/inference/src/model/qwen35/generation.rs`
- `crates/inference/src/forward/cpu/matmul.rs`, `crates/inference/src/forward/cpu/blas.rs`
- `docs/adr/ADR-058-cpu-perf-regression-ci.md`
- `docs/adr/ADR-064-ci-gate-taxonomy.md`
