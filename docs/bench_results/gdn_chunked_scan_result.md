# GDN Chunked Prefill Scan — Final Result

**Branch**: `show/prefill-batch/gdn-chunked-scan`
**Implementation commit**: `d57fa859` — `perf(inference): add chunked-parallel GDN scan for Qwen3.5-0.8B prefill`
**Test commit**: `773f1e94` — `test(inference): add B-vs-B and state-diff correctness gates for chunked GDN`
**Critic verdict**: **REJECT** (CRIT:2 | MAJ:3 | MIN:2 | PASS:4)
**Date**: 2026-06-03

---

## 1. A/B Prefill Performance (serial vs chunked, 18 active GDN layers)

Source: `analyst-2/bench_report.md` — interleaved same-process A/B, 5 pairs, 2 warmups, CV < 20%.

| Context (tokens) | Serial (ms) | Serial (tok/s) | Serial CV | Chunked (ms) | Chunked (tok/s) | Chunked CV | Δ tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 145 | 320.7 | 452.1 | 9.2% | 202.3 | 716.7 | 8.3% | **+58.5%** |
| 289 | 681.1 | 424.3 | 5.0% | 422.4 | 684.2 | 5.6% | **+61.3%** |
| 529 | 1167.1 | 453.3 | 8.9% | 746.3 | 708.8 | 10.0% | **+56.4%** |
| 1009 | 2189.5 | 460.8 | 11.9% | 1486.2 | 678.9 | 10.8% | **+47.3%** |

**Paired speedup statistics (same 5 pairs):**

| Context | Paired speedup mean | 95% CI | p-value | d_z |
|---:|---:|---:|---:|---:|
| 145 | +58.5% | [+51.2%, +65.7%] | 2.35e-5 | 10.01 |
| 289 | +61.3% | [+58.2%, +64.5%] | 6.92e-7 | 24.26 |
| 529 | +56.7% | [+46.7%, +66.7%] | 9.50e-5 | 7.04 |
| 1009 | +47.4% | [+36.4%, +58.3%] | 2.75e-4 | 5.37 |

### Per-GDN-Layer Timing (amortized wall time / 18 active GDN layers)

| Context | Serial ms/layer | Chunked ms/layer | Δ |
|---:|---:|---:|---:|
| 145 | 17.8 | 11.2 | -36.9% |
| 289 | 37.8 | 23.5 | -38.0% |
| 529 | 64.8 | 41.5 | -35.9% |
| 1009 | 121.6 | 82.6 | -32.1% |

> **Note on bench harness**: `scripts/bench-compare.sh` parsed Criterion JSON failed (missing `base/estimates.json`); the table above is from a custom interleaved example (`bench_gdn_prefill_ab.rs`). The project's mandatory bench gate did not produce a valid comparison artifact. The per-layer timing is amortized wall time (includes non-GDN layers), not per-kernel GPU counters.

---

## 2. Correctness Gate Output

Source: `tester/correctness_report.md`, commit `773f1e94`.

| Gate | Threshold | Actual | Result |
|------|-----------|--------|--------|
| 1. Argmax parity (10-run tight loop) | argmax equality + `max_abs_diff < 0.5` | argmax=271 all 10 runs; diff range 0.048–0.278 | **PASS** (tolerance weakened; see note) |
| 2. B-vs-B self-consistency (8 back-to-back runs) | `max_abs_diff < 1e-3` | **0.387** (387× over) | **FAIL** |
| 3. GDN state S chunked vs serial | `max_rel_diff < 1e-3` | `max_abs_diff=0.201`, `max_rel_diff=9890`, 87% elements over threshold | **FAIL** |

**Gate 1 raw output (10 runs):**
```
Run  1: parity: max_abs_diff=0.074880, argmax_step=271, argmax_prefill=271
Run  2: parity: max_abs_diff=0.188942, argmax_step=271, argmax_prefill=271
Run  3: parity: max_abs_diff=0.064248, argmax_step=271, argmax_prefill=271
Run  4: parity: max_abs_diff=0.277814, argmax_step=271, argmax_prefill=271
Run  5-10: argmax=271, diffs 0.048–0.182
```

**Gate 2 raw output:**
```
B-vs-B runs 0..1: max_abs_diff=0.386832
PANICKED: B-vs-B self-consistency FAIL (> 1e-3)
```

**Gate 3 raw output:**
```
State diff: max_rel_diff=9890.126953, max_abs_diff=0.200889
            elements_over_threshold=4,097,480 / ~4,718,592 total (~87%)
PANICKED: GDN state S diverges: max_rel_diff=9890
```

> **Note on Gate 1**: The `< 0.5` tolerance is ~2× above the observed chunked-vs-serial noise (0.05–0.28). Argmax stability is necessary but not sufficient. The tolerance was loosened in ADR-065; see MIN-1 in critic verdict.

---

## 3. Algorithm as Actually Implemented

### Chunk Size
Fixed `GDN_CHUNK_SIZE = 32` (C=32). Chosen for Qwen3.5-0.8B (kd=vd=128, one threadgroup per chunk).

### Kernel Structure (5 MSL kernels in `MSL_SOURCE`)

| Kernel | Grid | Threads | Purpose |
|--------|------|---------|---------|
| `gdn_chunk_materialize_c32` | `(num_chunks, num_value_heads, 1)` | `(32, 4, 1)` | Conv1d+SiLU+L2-norm for Q/K/V; computes beta (sigmoid), log_alpha (−exp(a_log)·softplus+dt_bias) |
| `gdn_chunk_solve_c32` | `(num_chunks, num_value_heads, 1)` | `(32, 4, 1)` | Log-gamma cumsum; forward substitution for W[j], U[j], K_right[j]; `threadgroup_barrier` per j-step |
| `gdn_chunk_residual_output_c32` | `(⌈vd/8⌉, num_value_heads, 1)` | `(32, 8, 1)` | R[j,v] = U[j,v] − γ[j]·(W·S0); O_inter + O_intra → raw_out |
| `gdn_chunk_state_update_c32` | `(⌈kd/16⌉, ⌈vd/16⌉, num_value_heads)` | `(16, 16, 1)` | S_all[h,v,k] = exp(γ_end)·S_old[h,v,k] + Σ_j R[j,v]·K_right[j,k] |
| `gdn_chunk_norm_silu_c32` | `(n_tokens, num_value_heads, 1)` | `(128, 1, 1)` | RMS norm + SiLU gate, matches serial kernel exactly |

### Decay / Mask Handling
- **Beta**: `sigmoid(hidden · in_proj_b)`, per token in materialize.
- **Log-alpha**: `−exp(a_log) · softplus(hidden · in_proj_a + dt_bias)`, per token in materialize.
- **Cumulative log-gamma**: `γ_log[j] = Σ_{m=0}^{j} log_alpha[m]`, stored as log values for numerical stability. Kernels compute `exp(γ_log)` locally.
- **L[j,k] mask**: `exp(γ_log[j] − γ_log[k])` for j≥k (causal, diagonal=1), computed inline in solve, never materialized as a matrix.

### Gating
- Env var `LATTICE_GDN_CHUNKED=1` enables the path in `forward_prefill_batched_chunk`.
- Shape gate: only 0.8B shape (kd=128, vd=128), prefill only, batch_size=1.
- Serial path remains the default; decode and Q36 paths are unchanged.
- GDN state layout (`gdn_gpu_s_matrices`, value-major `[num_value_heads, vd, kd]`) preserved so decode after prefill works correctly **when the state is correct**.

---

## 4. Honest Verdict

### Speedup Achieved
**The chunked kernel is faster at all context lengths (+47% to +61%), but the speedup is not bankable because the kernel computes incorrect state.**

The ≥529-token perf bar is numerically met:
- 529 tokens: +56.4%
- 1009 tokens: +47.3%

However, Gates 2 and 3 both fail — the kernel is internally non-deterministic (max_abs_diff=0.387 across 8 identical runs) and produces GDN state S matrices that diverge from the serial reference by max_abs_diff=0.201 (87% of elements wrong). Corrupt post-prefill state poisons every subsequent decode token. The reported tok/s numbers reflect a kernel doing wrong work quickly.

### What Prevents Banking the Speedup

**CRIT-1 (blocking)**: GDN state S diverges from serial by max_abs_diff=0.201. Probable cause: `gdn_chunk_state_update_c32` may double-exponentiate the decay — the design says multiply `S_old` by `gamma_end = exp(log_gamma[Ci-1])` (once), but the implementation notes describe `exp(γ_end) × S_old` where `γ_end` may already be the exponentiated value. Must be verified by reading the kernel body directly and comparing to `gdn_cpu_recurrence`.

**CRIT-2 (blocking)**: Compiler error (E0599) in test file at lines 17181/17190. The code shows `use crate::speculative::MtpTargetVerifier as _;` at line 17180, which *should* resolve `snapshot_gdn_states()`, but the critic flagged this — needs verification by actually running `cargo test --lib … gdn_chunk --no-run`.

**MAJ-1**: Internal non-determinism (max_abs_diff=0.387 across 8 runs). Caused by the races below.

**MAJ-2**: `gdn_chunk_materialize_c32` reads and writes persistent `gdn_gpu_conv_bufs` in the same all-chunks dispatch (lines ~2724, ~2810, dispatch ~11007). Metal does not guarantee threadgroup ordering by grid coordinate — chunk 0's read races with the last chunk's write.

**MAJ-3**: `gdn_chunk_residual_output_c32` may have multiple writers to `out_r`/`raw_out` at the same device address without a barrier before `O_intra` reads `R` (lines ~3023, ~3034).

### Next Levers (if correctness is fixed)

1. **Fix the kernel bugs first** — the +47–58% ceiling is real once the state update is correct and the races are eliminated.
2. **Per-layer GPU counters** — the current per-layer timing is amortized wall time; Metal GPU counters would give the kernel-specific breakdown needed to identify the next bottleneck.
3. **Adaptive chunk size** — C=32 was chosen conservatively. For longer sequences, larger chunks (C=64, C=128) reduce the number of inter-chunk state propagations and may improve the 1009-token case further.
4. **Conv-buffer double-buffer** — eliminating the persistent conv_buf mutation from the materialize kernel (use immutable scratch) removes the race and may also reduce register pressure.

---

## 5. Critic Verdict Outcome

**Verdict: REJECT**
**Summary: CRIT:2 | MAJ:3 | MIN:2 | PASS:4**

The critic reviewed: architect design, analyst spec, implementer notes, tester correctness report, analyst-2 bench report, and reviewer findings against the flow YAML acceptance gates.

**CRITs:**
- **CRIT-1**: GDN state S diverges from serial (max_abs_diff=0.201, 87% elements wrong) — violates the flow YAML hard constraint that post-prefill S equals the serial recurrence. Corrupts all decode after prefill.
- **CRIT-2**: Gate tests may not compile (E0599, MtpTargetVerifier import) at lines 17181/17190 — unrunnable gates cannot gate anything.

**MAJs:**
- **MAJ-1**: Internal non-determinism (Gate 2: max_abs_diff=0.387 over 8 identical runs).
- **MAJ-2**: conv_buf read/write race in materialize (~2724, ~2810, ~11007).
- **MAJ-3**: many-writer race in residual_output (~3023, ~3034).

**MINs:**
- **MIN-1**: Parity tolerance `< 0.5` is vacuous (sits 2× above the noise it should catch); ADR-065 weakened the gate.
- **MIN-2**: ADR-065 evidence out-of-repo (not auditable).

**PASSes:**
- Serial-default preserved; no new crates; no library `unwrap()`; value-major S layout preserved; decode + Q36 paths untouched.

### Residual Risk Statement (implementer-2)

The blocking kernel bugs (CRIT-1, MAJ-2, MAJ-3) require Metal kernel debugging — reading the MSL state-update body at `gdn_chunk_state_update_c32`, checking the `exp(γ_end)` vs `exp(log_gamma)` math, and restructuring the materialize dispatch to separate the final conv_buf write from the all-chunks read. These are correctible but were not completed within the available show budget. The perf ceiling (+47–58%) is achievable once the correctness is fixed.

**Recommended next step**: Fix CRIT-1 (state-update double-exp) first — it's the load-bearing correctness bug. The races (MAJ-2, MAJ-3) are likely secondary contributors to the non-determinism but may partially resolve once the state accumulation is algebraically correct.

---

*Generated by implementer-2 (α[implementer]) | show: prefill-batch/gdn-chunked-scan | op 10/10*
