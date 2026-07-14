# ADR-055: Online Distribution Drift Detection via Sinkhorn Divergence

**Status**: Accepted
**Date**: 2026-05-19
**Crate**: `lattice-transport` + `lattice-inference`

---

## Context

Lattice serves LoRA adapters and MoE router weights that are trained on a fixed data distribution.
As a model deployment ages, the input distribution drifts — vocabulary shift, domain change,
user population change — and the adapter (or router) that was optimal at training time becomes
stale. No mechanism currently exists to detect this staleness at serving time.

`lattice-transport` already implements the full Sinkhorn-Knopp solver (ADR-035), log-domain
stability (ADR-036), cost matrices (ADR-037), Wasserstein barycenters (ADR-038), and the
Sinkhorn divergence with correct debiasing (ADR-039). That stack is the OT primitive layer.

The KG contains three relevant entities:

- **`Online Sinkhorn Drift Estimator`**: O(W²) sliding-window algorithm for distribution shift
  detection with a minimum window of 128 samples.
- **`SinkhornAdapterStalenessDetector`**: Applies the drift estimator to detect when LoRA
  adapters need retraining.
- **`Adapter Refresh Threshold`**: Renewal theory formula for computing optimal refresh
  intervals given drift rate and excess NLL cost.

A structural insight emerges: the Sinkhorn drift signal that triggers adapter retraining applies
equally to MoE router staleness. The Qwen3.5 MoE router (see ADR-040) was trained on a token
distribution. When input token distributions drift, the routing assignments drift with them —
the experts that receive tokens shift, some experts become under-utilized, load imbalance
emerges, and effective capacity decreases. The `Adapter Refresh Threshold` renewal theory result
applies symmetrically to both mechanisms. This ADR formalizes the dual-use design.

---

## Decision

Add an `OnlineDriftDetector` to `lattice-transport` that computes Sinkhorn divergence over
a sliding window of hidden-state samples. Expose an `OnlineDriftSignal` per-sample status enum that
`lattice-inference` subscribes to for triggering adapter and router staleness checks.

**Existing drift API**: `crates/transport/src/drift.rs` already provides `DriftConfig`,
`DriftReport`, and `detect_drift_records` for batch embedding drift analysis. The new online
detector is a distinct mechanism — streaming, sample-by-sample, with a sliding window — and
uses non-conflicting names (`OnlineDriftDetector`, `OnlineDriftConfig`, `OnlineDriftSignal`)
to coexist with the existing batch API.

The boundary is explicit:

- `lattice-transport` owns: the sliding window, divergence computation, threshold comparison,
  and signal emission. It is distribution-agnostic — it does not know what the samples represent.
- `lattice-inference` owns: sampling from the inference pipeline (which layer, how often),
  reacting to `OnlineDriftSignal` (which component to refresh, how to refresh it).

**Crate dependency note**: `lattice-inference` does not currently depend on `lattice-transport`.
Rather than adding that edge (which would break the leaf-crate invariant), the integration uses
a callback boundary: `lattice-inference` defines a `DriftSampler` trait, and the application
binary (`bin/chat_metal` or the serving layer) wires the concrete `OnlineDriftDetector` from
`lattice-transport` into the `DriftSampler` impl. Neither crate imports the other.

---

## Scope

New files:

- `crates/transport/src/online_drift.rs` — `OnlineDriftDetector`, `OnlineDriftConfig`, `OnlineDriftSignal`

Modified files:

- `crates/transport/src/lib.rs` — re-export `online_drift` module (existing `drift` module unchanged)
- `crates/inference/src/monitor.rs` — `DriftSampler` trait with `on_adapter_stale()` and
  `on_router_stale()` callbacks; no import of `lattice-transport`

No new crates. No Metal kernel changes. No changes to serving hot path.

---

## Architecture

### Layer 0: `lattice-transport` — drift primitive

```rust
pub struct OnlineDriftConfig {
    pub window_size: usize,     // W; minimum 128. O(W²) per divergence call.
    pub check_interval: usize,  // compute divergence every N new samples (amortizes cost)
    pub threshold: f32,         // S(ref, current) > threshold → return Drift
    pub epsilon: f32,           // Sinkhorn regularization (inherited from SinkhornConfig)
}

pub enum OnlineDriftSignal {
    Warming { samples_seen: usize, window_size: usize },
    Skipped { samples_seen: usize, next_check_in: usize },
    Stable { divergence: f32, window_pos: usize },
    Drift { divergence: f32, window_pos: usize },
}

pub struct OnlineDriftDetector {
    config: OnlineDriftConfig,
    reference_window: VecDeque<Vec<f32>>,   // first W samples; frozen as reference
    current_window: VecDeque<Vec<f32>>,     // sliding window of last W samples
    samples_seen: usize,
    // Three independently sized, reusable workspaces; divergence solves reset dual variables
    workspace_xy: SinkhornWorkspace,
    workspace_xx: SinkhornWorkspace,
    workspace_yy: SinkhornWorkspace,
}

impl OnlineDriftDetector {
    pub fn observe(&mut self, sample: Vec<f32>)
        -> Result<OnlineDriftSignal, SinkhornError>;
    pub fn reset_reference(&mut self);  // call after adapter/router refresh
}
```

`observe()` appends to `current_window` and evicts the oldest entry if full. It returns
`Warming` while the windows fill and `Skipped` between checks. Every `check_interval`
samples, it calls `point_set_sinkhorn_divergence` (ADR-039) between `reference_window` and
`current_window`, returning `Stable` or `Drift` according to the threshold. Solver failures
are returned as `Err(SinkhornError)`.

The reference window is frozen at construction from the first W samples. After an adapter
refresh, `reset_reference()` promotes `current_window` to `reference_window`, restarting
the baseline from the updated distribution.

### Layer 1: `lattice-inference` — sampling and reaction

```rust
// lattice-inference owns only the trait — no lattice-transport import.
// The application binary wires the concrete OnlineDriftDetector from transport.

/// Callback trait for inference-side drift reaction. Implemented by the application
/// binary, not by lattice-inference itself.
pub trait DriftSampler: Send + Sync {
    /// Push a hidden-state sample. Uses &self with interior mutability
    /// (e.g., Mutex<OnlineDriftDetector>) so the sampler can be stored
    /// in Arc<dyn DriftSampler> without exclusive access.
    fn push_sample(&self, hidden_state: &[f32]);
    fn on_adapter_stale(&self, divergence: f32);
    fn on_router_stale(&self, divergence: f32);
}
```

The monitor hooks into `lattice-inference`'s forward pass via an optional `Arc<dyn DriftSampler>`
on `GenerateConfig`. If `None` (the default), zero overhead — no sampling, no allocation.
When present, every `sample_every_n_tokens` tokens, the monitor receives the hidden states at
`sample_layer` and pushes a mean-pooled sample vector (dim = hidden_size) to both detectors.

Mean pooling collapses a variable-length sequence of vectors (one per token in the current
step) to a single fixed-size sample, making the window storage O(W × hidden_dim) regardless
of sequence length.

### Dual-use signal path

```
Forward pass (every N tokens)
  → sample hidden states at layer L
  → mean pool → Vec<f32> of dim hidden_size
  → DriftSampler::push_sample(hidden_state)
  → if drift detected: DriftSampler::on_adapter_stale / on_router_stale
```

Both detectors share the same sample stream because both adapter staleness and router staleness
are caused by the same root: input distribution drift. A single sample population feeds two
independent sliding windows with potentially different configs (the router may tolerate more
drift before refresh is warranted).

### Threshold calibration

The `Adapter Refresh Threshold` uses renewal theory:

```
refresh_interval* = sqrt(2 * C_refresh / (δ * rate_of_divergence))
```

where `C_refresh` is the cost of retraining/swapping, `δ` is the excess NLL from stale adapter
use, and `rate_of_divergence` is empirically estimated from the drift signal time series.

In practice, `threshold` in `OnlineDriftConfig` is a Sinkhorn divergence value calibrated from a
validation set: compute `S(training_dist, held_out_dist)` at known excess NLL levels and pick
the divergence that corresponds to acceptable degradation. The `divergence` value carried by the
`Stable` and `Drift` variants lets the reactor apply graduated responses (warn vs. force-refresh
vs. fallback to base model).

---

## Alternatives Considered

| Alternative                                                           | Pros                             | Cons                                                                                        | Why Not                                                                   |
| --------------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| KL divergence on token ID distribution                                | O(n log n), no OT                | Requires binning; undefined when support differs; no geometry                               | Continuous hidden states don't bin naturally                              |
| MMD (Maximum Mean Discrepancy)                                        | Closed form kernel test          | Kernel choice matters; no geometry; harder to threshold                                     | Less interpretable than OT divergence; no transport interpretation        |
| Exponential moving average of cosine similarity                       | Zero OT cost, O(1) per step      | Not a proper divergence; `cos(a, a) = 1` always, no `S(a, b) = 0` guarantee                 | Cannot distinguish in-distribution drift from magnitude shift             |
| Separate detectors per crate (transport-side only, no inference hook) | Keeps transport fully standalone | Forces inference callers to wire sampling manually; drift is only useful with reaction      | Incomplete without the reaction path; adds boilerplate at every callsite  |
| Monitor every token (no `sample_every_n_tokens`)                      | Maximally sensitive              | O(W²) per token is prohibitive at 128 window; 128×128×3 solves at 1 ns/op = 50 µs per token | Sample-and-amortize: every 16 tokens, 50 µs total amortized to 3 µs/token |

---

## Risks

**R1: O(W²) cost at large windows.** At W=128, three Sinkhorn solves over 128×128 matrices.
Each solve ≈ 50 iterations × 128×128 × 1 ns/op ≈ 0.8 ms; total ≈ 2.4 ms per divergence call.
At `check_interval=256` tokens, amortized cost is 2.4 ms / 256 ≈ 9 µs/token — below serving
latency noise floor on M-series silicon. W > 512 requires explicit approval and profiling.

**R2: Reference window quality.** The reference window is built from the first W samples
seen after construction (or after `reset_reference()`). If the first W samples are not
representative — cold start, single user, atypical prompt — the reference is biased and
divergence will fire spuriously. Mitigation: `reset_reference()` should be called after a
warm-up phase of at least 2W samples; document this in `OnlineDriftConfig`.

**R3: Mean pooling loses distributional shape.** Mean pooling is an order-1 statistic.
Two very different distributions with the same mean pool to the same sample. For the first
deployment this is acceptable — mean shift is the primary signal of distribution drift.
Higher-order statistics (covariance pooling) are a future extension if mean-pool sensitivity
proves insufficient.

**R4: Crate boundary violation.** If `lattice-transport` were to import anything from
`lattice-inference` (for example, to know what layer to sample from), the dependency would
invert. The design prevents this by making `OnlineDriftDetector` fully sample-agnostic.
`lattice-inference` owns all decisions about what to sample; `lattice-transport` only
receives `Vec<f32>` values. Enforcement: `crates/transport/Cargo.toml` must never contain
`lattice-inference` as a dependency.

---

## References

- ADR-035: Sinkhorn-Knopp Solver — sliding window builds on this primitive
- ADR-036: Log-Domain Stability — numerical robustness inherited
- ADR-039: Sinkhorn Divergence — `point_set_sinkhorn_divergence`, three reusable workspaces
- ADR-040: Gated Attention (MoE router) — router staleness context
- ADR-043: LoRA Serving Verification — adapter serving context
- Genevay, Peyré, Cuturi, "Learning Generative Models with Sinkhorn Divergences", AISTATS 2018
- Feydy et al., "Interpolating between Optimal Transport and MMD using Sinkhorn Divergences",
  AISTATS 2019 — positive semi-definiteness proof used to justify divergence thresholding

## Implementation status (2026-06-24)

`OnlineDriftDetector` is fully implemented at `crates/transport/src/online_drift.rs:109` and
re-exported from `crates/transport/src/lib.rs:119`. The `DriftSampler` bridge trait described
in §"Layer 1: `lattice-inference`" was not built. Searching `crates/inference/src/` for
`DriftSampler` returns no results. The inference forward pass has no `Arc<dyn DriftSampler>`
hook. The end-to-end drift sampling path (inference → transport → reaction) described in this
ADR's architecture exists only in `lattice-transport`; the inference-side callback boundary
and application-level wiring remain unimplemented.
