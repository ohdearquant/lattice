# ADR-065: Feature Promotion Gates — Merge Measured Primitives, Not Research Ideas

**Status**: Proposed
**Date**: 2026-06-30
**Scope**: lattice workspace — all feature branches, research issues, and experiment PRs
**Depends on**: ADR-064 (CI Gate Taxonomy and Promotion Policy)

---

## Context

Lattice ships a research-composable inference kernel. The cost of this composability is an ever-growing backlog of ideas that are technically plausible but not yet measured: learnable test-time state updates, latent GDN reasoning, state interpolation, adaptive compute routing, and more. Without a shared gate taxonomy, the bar for "ready to merge" is implicit and inconsistent — some PRs land at the research-note stage, others stall indefinitely because the bar is unclear.

ADR-064 classifies *existing* CI gates. This ADR defines the *decision rule* for any new feature or research idea from inception to default-on: a five-level gate ladder where each level has an explicit measurable threshold and a standard issue template that every research question must answer before merge.

The core tension is: features are cheap to add and expensive to maintain. Every default-on feature is a permanent regression surface, a forever-tested configuration, and a mental model that every future developer must carry. The gate ladder is designed to make that cost explicit before it is paid.

---

## Decision

### Principle

**Lattice merges measured primitives, not research ideas.**

A research idea may be correct. It may even be elegant. It does not merge until it has produced a measurement that could falsify it, passed a kill threshold that would have killed it if it were wrong, and lived behind a flag long enough to accumulate evidence. The gate ladder encodes this as five levels; every feature must pass its entry gate before merge.

---

### The Five-Gate Ladder

#### G0 — Research Note

A research note is a markdown issue, a Jupyter/Pluto notebook, or a script under `experiments/`. It does **not** touch any file in `crates/` or `.github/`. Its only merge bar is that it is readable and cites the evidence it is based on.

G0 artifacts: GitHub issue, `experiments/<topic>/`, literature reference.

#### G1 — Measurement Primitive

A measurement primitive adds visibility *without changing model output*. It is telemetry, a bench harness, a state dump, or a JSONL trace, always behind a Cargo feature flag or a runtime `LATTICE_*` env var.

Merge bar for G1:
- Adds no observable change to generated token sequences when the flag is off.
- `< 2%` decode-throughput overhead when the flag is on; `0%` when off (confirmed by `bench_decode_ab` on the primary target workload).
- Stable JSONL schema that captures at minimum: `model`, `tokenizer`, `quant`, `commit`, `hardware`, `prompt_hash`, and the metric being observed.
- No change to any existing public API surface.

G1 artifacts: Cargo feature or env-gated telemetry path, bench JSONL trace, schema doc.

#### G2 — Experiment Branch

An experiment branch implements the feature idea behind a named feature gate. It does not change any default behavior.

Merge bar for G2:
- Exactly one target benchmark (the primary metric this feature is supposed to improve).
- Exactly one baseline measurement on the same hardware and model checkpoint (committed as `experiments/<name>/baseline.jsonl`).
- Exactly one kill threshold: a value of the primary metric at or below which the branch is abandoned (committed as `experiments/<name>/kill_threshold.md`).
- Machine-readable metrics output (JSONL conforming to the G1 schema).
- Reproducible from a single command: `cargo bench --features <flag> -- <bench_name>` or equivalent.

G2 artifacts: feature-gated implementation, `experiments/<name>/baseline.jsonl`, `experiments/<name>/kill_threshold.md`, CI bench integration.

#### G3 — Default-Off Feature

A default-off feature has beaten its baseline on the primary metric, survived its kill threshold, and produced no regression on secondary metrics. It ships with the feature gate off in `Cargo.toml` defaults and in all CI configurations unless explicitly activated.

Merge bar for G3 (all thresholds must be met simultaneously):

| Metric class | Threshold |
|---|---|
| Speed | ≥ 15% wall-clock improvement on the primary decode or prefill benchmark |
| Memory | ≥ 20% peak memory reduction (if memory is the primary claim) |
| Quality | Statistically significant improvement at matched token counts (p < 0.05, Welch t-test or bootstrap CI) |
| Safety | Lower false-negative rate at matched false-positive rate on the safety eval suite |
| Telemetry overhead | < 2% decode throughput impact when on; exactly 0% when off |
| Training overhead | No hot-path cost increase unless < 5% additional decode overhead |

A G3 feature may relax any threshold that is not its primary claim (e.g., a speed feature need not show memory improvement), but it must not *worsen* any metric beyond measurement noise (< 1 σ of the baseline distribution).

G3 artifacts: feature-gated implementation, full benchmark results against G2 baseline, CI integration with feature enabled, updated `docs/adr/` entry or new ADR.

#### G4 — Default-On

Default-on is rare and reserved for primitives that cannot be wrong by construction. Acceptable G4 categories:

- **Pure telemetry-off paths**: a code path that is only reached when a telemetry feature is explicitly disabled; removing it cannot change model behavior.
- **Deterministic artifacts**: changes that produce bit-identical output under all inputs (e.g., a correctness fix, a deterministic hash, a canonical serialization format).
- **Exact snapshot/restore**: checkpoint/restore logic that is verified to round-trip by property test.
- **Safe scheduling or cache management**: changes to KV page allocation or scheduler bookkeeping that preserve all existing invariants and are covered by the existing property-test suite.
- **No-quality-loss kernels**: a kernel replacement verified to be numerically identical to the replaced kernel by the existing forward-equivalence test suite.

G4 merge bar: the category must be justified in the PR description by reference to one of the five categories above, and the relevant test must pass in CI.

---

### The Eight-Question Issue Template

Every GitHub issue that proposes a new feature or research direction must answer all eight questions before any code is written. Unanswered questions are not acceptable; `N/A` requires justification.

```markdown
## Feature Proposal Checklist

1. **User-visible outcome**: What user-observable behavior improves if this works?
   <!-- e.g., "Time-to-first-token for long-context requests drops by ≥ 15%" -->

2. **Prior art**: What existing implementation, paper, or ADR already covers this?
   <!-- If none, explain why this is genuinely novel. -->

3. **Cheapest falsifying experiment**: What is the smallest experiment that would
   prove this idea wrong?
   <!-- e.g., "Run bench_decode_ab with flag on vs off on a 2k-token prompt" -->

4. **Promote threshold**: At what measured value does this become a G3 candidate?
   <!-- Must be a concrete number, e.g., "> 15% wall-clock on bench_decode_ab" -->

5. **Kill threshold**: At what measured value is this idea abandoned?
   <!-- Must be a concrete number, e.g., "< 5% improvement after 3 experiment PRs" -->

6. **Code location before proof**: Where does the implementation live while it is
   still at G0/G1/G2?
   <!-- e.g., "experiments/mtp-adaptive-draft/ or behind feature = 'exp-adaptive'" -->

7. **Default-on guard**: What flag or Cargo feature prevents this from becoming
   default-on before it reaches G4?
   <!-- e.g., "LATTICE_ADAPTIVE_DRAFT=1 env var; default false" -->

8. **Gate level**: What gate level (G0–G4) does this PR target?
   <!-- G0 = research note only; G1 = telemetry; G2 = experiment; G3 = default-off; G4 = default-on -->
```

---

### The Adaptive-Feature Gating-Vector Pattern

Any feature whose job is to "decide how hard to think" or "how much to trust this token" should be built as a gating vector that maps cheap per-token signals to a calibrated error probability and then to an action. This is the standard shape for adaptive compute features in lattice; new features in this space should follow it rather than inventing ad hoc routing logic.

#### Cheap per-token signals (inputs)

| Signal | Source | Cost |
|---|---|---|
| Token entropy | softmax of top-k logits | one pass over logits |
| Top-k logit mass | sum of top-k softmax | same pass |
| Logit margin (top1 − top2) | already computed for sampling | free |
| Sequence average log-prob | running sum, O(1) update | O(1) per token |
| MTP accept rate | `MtpVerifier` output | G1 telemetry flag |
| MTP head disagreement | KL between draft and target distributions | G1 telemetry flag |
| GDN state-delta cosine | cosine similarity of consecutive `GatedDeltaNetState` snapshots | G1 telemetry flag |
| State-norm growth | L2 norm of delta between snapshots | G1 telemetry flag |
| Gate entropy | entropy of GDN gating vector | G1 telemetry flag |
| Context length | token position / KV fill fraction | free |
| Domain flags | per-request metadata (e.g., code, math, safety-sensitive) | O(1) |

#### Calibrated error probability (intermediate)

The signals above are mapped to a single scalar `p_err ∈ [0, 1]` representing the estimated probability that the current token or segment is in the tail of the model's competence distribution. Calibration is required: raw confidence scores are systematically over- or under-confident.

**Isotonic regression / UCCI-style margin calibration** is the recommended prior art. Isotonic regression fits a monotone mapping from the raw score (e.g., logit margin) to empirical error rate on a held-out calibration corpus without assuming a parametric form. Conformal / UCCI-style margin calibration extends this to sequence-level coverage guarantees. Isotonic regression is available directly as a scikit-learn primitive (`sklearn.isotonic`); the conformal variants are standard, separately-published methods. Both families have been validated on LLM confidence calibration tasks. Any G2 or G3 gating-vector feature must include a calibration curve on the primary eval corpus as part of its benchmark output.

#### Action set (outputs)

| Action | Trigger condition | Cost |
|---|---|---|
| `allow-direct` | `p_err < θ_low` | baseline decode |
| `short-think` | `θ_low ≤ p_err < θ_mid` | reasoning budget × short multiplier |
| `long-think` | `θ_mid ≤ p_err < θ_high` | reasoning budget × long multiplier |
| `multi-sample` | `p_err ≥ θ_high` and budget allows | N independent samples, majority vote |
| `escalate-larger-local-model` | `p_err ≥ θ_high` and larger model available | switch to resident larger model |
| `quarantine-or-defer` | safety flag set or `p_err = 1.0` | return structured error |

Thresholds `θ_low`, `θ_mid`, `θ_high` are calibrated per domain and per model checkpoint. They are not compile-time constants; they are loaded from a config file at model startup and are themselves G1-gated until calibrated on production data.

---

### Closing Note: Research Roadmap and High-Risk Hypotheses

The open research-roadmap epics — adaptive compute routing, MTP acceptance-rate-driven early exit, GDN state-guided speculative drafting, and domain-aware threshold tuning — are the first consumers of this ADR. Each of those epics should open with the eight-question template and proceed through the gate ladder in order.

Three hypotheses carry elevated risk because they require state telemetry that does not yet exist:

- **Learnable test-time state updates**: updating model parameters or adapter weights at inference time based on the current sequence. Risk: silent distribution shift, unbounded memory, and safety-surface expansion.
- **Latent GDN reasoning**: using the GDN state vector as an implicit chain-of-thought medium without an explicit reasoning trace. Risk: unobservable intermediate state makes safety evaluation impossible.
- **State interpolation**: interpolating between two GDN state vectors to blend two conversation contexts. Risk: emergent behavior at the interpolation boundary has no principled bound.

All three **stay at G0 or G1** until the G1 state-telemetry primitive (GDN state-delta cosine, state-norm growth, gate entropy) exists in production and has accumulated sufficient calibration data to make the kill thresholds in the eight-question template meaningful. Merging any of these hypotheses before state telemetry exists would mean shipping a G3 feature with an untestable kill threshold — a direct violation of this ADR's principle.

---

## Consequences

**Positive**:
- Any contributor opening a new research issue knows exactly what questions to answer and what gate to target.
- The kill threshold is committed before the feature is built, removing retrospective ambiguity about when to abandon a direction.
- G1 measurement primitives accumulate calibration data that makes all future gating-vector features cheaper to evaluate.

**Negative**:
- The gate ladder adds friction for genuinely obvious improvements that would have passed G3 trivially. Mitigation: G3 merge bar can be satisfied in a single PR if the experiment results are included.
- Calibration curves require a held-out corpus, which must be maintained as the model checkpoint evolves.

---

## References

- ADR-064: CI Gate Taxonomy and Promotion Policy
- ADR-006: Speculative Decoding (`MtpVerifier`, GDN rollback — G1 telemetry consumers)
- ADR-052: GDN State Management (`GatedDeltaNetState::snapshot`/`restore_from` — state telemetry substrate)
- ADR-061: Inference Metrics Infrastructure — the G1 substrate for metric capture
- Isotonic regression calibration: Zadrozny & Elkan 2002, "Transforming classifier scores into accurate multiclass probability estimates"
- Conformal calibration (comparison method, distinct from isotonic regression): Angelopoulos & Bates 2023, "Conformal Risk Control", arxiv:2208.02814
- `crates/inference/src/attention/gdn.rs` — `GatedDeltaNetState::snapshot`/`restore_from`
- `crates/inference/src/speculative.rs` — `MtpVerifier`, accept-rate tracking
- `crates/inference/src/metrics.rs` — `LayerMetrics`, `ForwardMetrics`
