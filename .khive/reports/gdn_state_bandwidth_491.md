# GDN-State Bandwidth-Share Decision Gate — Issue #491

Branch: `bench/gdn-state-bandwidth-491` (off `36d9bf68c`)
Scope: G1, telemetry-only, zero production-path behavior change.

## Verdict: **KILL**

GDN recurrent-state traffic is a **single-digit fraction** of per-token decode
memory bandwidth (weights + KV + state) at every tested context length and
precision on Qwen3.5-0.8B Metal decode:

| Precision | 512 | 2K | 8K |
|---|---:|---:|---:|
| Q4  | 7.81% | 7.53% | 6.60% |
| f16 | 2.61% | 2.57% | 2.46% |

This does not reach the PROMOTE threshold (low-double-digit percent or higher
at any relevant context). **Deprioritize the state-compression research
family** — #494 (int4/int8/fp8 state precision), #498 (low-rank/SVD state
compression) — for this target, and treat #501 as a weight/KV/state
accounting issue only (not a state-compression justification). **Redirect
optimization effort to weight-bandwidth levers #420/#421/#346/#423**, since
weights are the dominant byte component in every cell (76.94%–96.99%).

Decision confidence: **HIGH**. The numerator is measured from live Metal GPU
runs on both quantization formats; the denominator components (weights, KV)
are deterministic source-derived analytical byte models, not literature
guesses.

## Share Table (state / weights / KV %)

Evidence tags: `[M]` = measured (live Metal run), `[A]` = analytical
(source-derived formula), `[C]` = computed from M+A.

### Q4

| Context | Weights bytes | KV bytes | GDN-state bytes | Total bytes | Weight % | KV % | **State %** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512   | `[A]` 470,747,136 | `[A]` 6,291,456   | `[M]` 40,402,944 | `[C]` 517,441,536 | 90.98% | 1.22%  | **7.81%** |
| 2,048 | `[A]` 470,747,136 | `[A]` 25,165,824  | `[M]` 40,402,944 | `[C]` 536,315,904 | 87.77% | 4.69%  | **7.53%** |
| 8,192 | `[A]` 470,747,136 | `[A]` 100,663,296 | `[M]` 40,402,944 | `[C]` 611,813,376 | 76.94% | 16.45% | **6.60%** |

### f16

| Context | Weights bytes | KV bytes | GDN-state bytes | Total bytes | Weight % | KV % | **State %** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512   | `[A]` 1,503,791,104 | `[A]` 6,291,456   | `[M]` 40,402,944 | `[C]` 1,550,485,504 | 96.99% | 0.41% | **2.61%** |
| 2,048 | `[A]` 1,503,791,104 | `[A]` 25,165,824  | `[M]` 40,402,944 | `[C]` 1,569,359,872 | 95.82% | 1.60% | **2.57%** |
| 8,192 | `[A]` 1,503,791,104 | `[A]` 100,663,296 | `[M]` 40,402,944 | `[C]` 1,644,857,344 | 91.42% | 6.12% | **2.46%** |

Formula: `GDN-state share = state / (weights + KV + state)`.

## Method — Measured vs. Analytical per Component

| Component | Status | Value | How obtained |
|---|---|---|---|
| GDN recurrent-state bytes/step | **MEASURED** | `40,402,944 B/step` (20,201,472 B read + 20,201,472 B write) | Live Metal GPU run (Apple M2 Max) of `bench_gdn_state` against both `~/.lattice/models/qwen3.5-0.8b` (f16) and `~/.lattice/models/qwen3.5-0.8b-q4` (Q4) checkpoints, reading the #513 `GdnStateTrafficReport` counters (feature `gdn-state-counters`). Byte-identical for Q4 and f16, and byte-identical across every sampled decode position (pos 19–28) → `context_invariant=true`. State buffers (conv cache + S matrix) are a fixed function of model config (18 active GDN layers × 1,122,304 B/layer), not of sequence position, so this value is reused unchanged at 512/2K/8K rather than re-measured at each length. |
| Weight bytes/step (all 24 layers) | **ANALYTICAL** | Q4: `470,747,136 B/step`; f16: `1,503,791,104 B/step` | Source-derived from per-layer weight-matrix sizes in the model config and quantization format (Q4 ≈ 0.5 B/weight + scales; f16 = 2 B/weight), summed across all layers. No production-path counters added — kept analytical per G1 scope. |
| KV bytes/step | **ANALYTICAL** | 512: `6,291,456 B`; 2K: `25,165,824 B`; 8K: `100,663,296 B` | `kv_bytes_per_token` (`crates/inference/src/model/qwen35_config.rs:557`) × cached context length; grows linearly with context, unlike weight and state bytes. |
| Share % per cell | **COMPUTED** | See table above | `state / (weights + KV + state)`, combining the fixed measured numerator with the two analytical, context-dependent denominators. |

No production-path instrumentation was added. The only change is additive
`println!` telemetry inside the existing `#[cfg(all(target_os = "macos",
feature = "metal-gpu", feature = "gdn-state-counters"))]` gate in
`crates/inference/examples/bench_gdn_state.rs`, reading fields already
exposed by the #513 `GdnStateTrafficReport`/`GdnStateTrafficCounters` API. No
new feature flags, no new counters, no changes to `metal_qwen35.rs` or
`qwen35_config.rs`.

### Bench extension

`crates/inference/examples/bench_gdn_state.rs`:
- Each `GDN_STATE_STEP` line now also reports `state_bytes_moved` (decode
  read + write bytes for that step).
- A new summary line `GDN_STATE_BYTES_PER_STEP avg=... min=... max=...
  context_invariant=<bool>` reports the average/min/max across sampled steps
  and whether the figure held constant (context-invariance check).

Raw output (tail, both f16 and Q4 runs identical):

```
GDN_STATE_SHAPE active_layers=18 allocated_layers=18 conv_bytes_per_layer=73728 s_bytes_per_layer=1048576 per_layer_bytes=1122304 active_state_bytes=20201472 allocated_state_bytes=20201472
GDN_STATE_STEP step=1  pos=19 decode_read_bytes=20201472 decode_write_bytes=20201472 state_bytes_moved=40402944 ... ok=true
...
GDN_STATE_STEP step=10 pos=28 decode_read_bytes=20201472 decode_write_bytes=20201472 state_bytes_moved=40402944 ... ok=true
GDN_STATE_TOTAL steps=10 decode_read_bytes=202014720 decode_write_bytes=202014720 ...
GDN_STATE_CROSSCHECK expected_per_step=20201472 profiler_group=gdn_mixer formula=active_gdn_layers*per_layer_state_bytes
GDN_STATE_BYTES_PER_STEP avg=40402944 min=40402944 max=40402944 context_invariant=true
```

## Gates — Actual Output (run 2026-07-01, this session)

### 1. `cargo fmt --check`

```
$ cargo fmt --check
EXIT=0
```
Clean, no output, no diff.

### 2. `cargo clippy -p lattice-inference --features "f16,metal-gpu,gdn-state-counters" --lib --examples -- -D warnings`

```
$ cargo clippy -p lattice-inference --features "f16,metal-gpu,gdn-state-counters" --lib --examples -- -D warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.68s
EXIT=0
```
Clean, zero warnings.

### 3. `cargo clippy -p lattice-inference -- -D warnings` (default features)

```
$ cargo clippy -p lattice-inference -- -D warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.17s
EXIT=0
```
Clean, zero warnings — confirms the default (non-`gdn-state-counters`) build
path is untouched by this change.

### 4. Bench build (release)

```
$ cargo build --release --example bench_gdn_state -p lattice-inference --features "f16,metal-gpu,gdn-state-counters"
    Finished `release` profile [optimized] target(s) in 0.36s
EXIT=0
```
Builds clean.

All four gates pass with real, reproduced output in this environment. No gate
failures encountered; no fixes were required.

## Interpretation

Weights dominate byte traffic in every cell tested (76.94%–96.99%). GDN
recurrent-state traffic is real and measured (not zero), but it is not a
first-order term in whole-model decode bandwidth for Qwen3.5-0.8B on Metal —
at best (Q4, 512 ctx) eliminating it entirely would only reclaim 7.81% of
total bytes moved per step, and less at longer context or in f16. KV becomes
the second-largest component at long context (16.45% at Q4/8K) but still
trails weights by a wide margin.

## Recommendation Routing

1. **KILL** state-compression as a priority path for this target — deprioritize #494 and #498.
2. **Redirect** near-term optimization effort to weight-bandwidth levers #420/#421/#346/#423 (weights are 76.94%–96.99% of bytes moved).
3. Treat KV compression/serving policy as the secondary long-context lever (KV is the only non-weight, non-state component that grows with context).
4. Keep the `gdn-state-counters` bench as telemetry/regression coverage — cheap guardrail if GDN shape, dtype, or serving mode changes later.
5. Only revisit state precision (#494) if the target changes to a state-dominant setting (isolated GDN kernel, different architecture, batched serving where state memory limits concurrency).

Full source-traced analysis: see team artifacts `analysis.md`, `recommendations.md`, and `verdict.md` (analyst-2) and `measured_state.md`/`implementation_notes.md` (implementer) from this play.

## Provenance

- Measured numerator: live Metal GPU (Apple M2 Max) runs against both
  `qwen3.5-0.8b` (f16) and `qwen3.5-0.8b-q4` (Q4) checkpoints, extending
  `crates/inference/examples/bench_gdn_state.rs`.
- Analytical denominator: source-derived weight-byte and `kv_bytes_per_token`
  formulas from `crates/inference/src/model/qwen35_config.rs`, no new
  production-path counters added (G1 scope).
- No fabricated numbers. Every value above is labeled measured, analytical,
  or computed from the two.
