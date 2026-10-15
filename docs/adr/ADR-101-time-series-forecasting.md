# ADR-101: Time-series forecasting models in lattice-inference

**Status**: Proposed (2026-10-15)\
**Date**: 2026-10-15\
**Crate**: lattice-inference (new module `forecast`); consumers: the `lattice` CLI, an evaluation harness

## Decision summary

Lattice gains native time-series forecasting, built as shared infrastructure first and models second. Models are ported in
this order: **Reverso, then Chronos-2, then the Toto-2.0 family**. The deliverable of each port is the shared layer it
forces into existence (context preparation, rollout, forecast types, evaluation), not only the model. After the three
ports, the same layer hosts a learned combination of several local forecasters, measured on GIFT-Eval, whose best
published aggregate today is 0.419 CRPS (lower is better). Nothing in this ADR reorders or drops a model from that
sequence.

## Context

### What lattice has today (source-read at `main` e8b40087)

| #  | Fact                                                                                                                                                                                           | Pointer                                                                 |
| -- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| L1 | No forecasting code and no FFT anywhere in the workspace.                                                                                                                                      | `git grep -i -n "fn .*fft\|rustfft\|realfft" -- crates` returns nothing |
| L2 | A delta-rule recurrence exists, but bound to the Qwen3.5 GatedDeltaNet layer: its step reads `Qwen35Config` and that layer's fused weight layout (`in_proj_qkv`, `in_proj_z`, gating `alpha`). | `attention/gdn.rs` `gated_delta_net_step`                               |
| L3 | f32 safetensors loading with exact shape reads, a CPU `matmul_bt` / `matmul_into` dispatch (Accelerate on macOS, SIMD elsewhere) and an in-place `layer_norm`.                                 | `weights/f32_weights.rs`, `forward/cpu/{matmul,norm}.rs`                |
| L4 | `serde_json` is already a dependency of lattice-inference.                                                                                                                                     | `crates/inference/Cargo.toml`                                           |

### The model landscape (GIFT-Eval, recomputed from the public per-configuration results)

GIFT-Eval scores 97 dataset, frequency and horizon configurations. The aggregate is, per configuration, a model's MASE and
CRPS divided by Seasonal Naive's, then the geometric mean over configurations. Recomputing it from the published raw
files reproduces a published paper value to four decimals (t0-alpha: MASE 0.7240, CRPS 0.4941), so the figures below
are the leaderboard's own. Snapshot: results as of 2026-10-09.

| Model                 | Params      | License        | CRPS               | Shape                                                                                               |
| --------------------- | ----------- | -------------- | ------------------ | --------------------------------------------------------------------------------------------------- |
| Best agentic ensemble | n/a         | n/a            | 0.419              | Several models combined per series                                                                  |
| TimesFM-3             | n/a         | Non-commercial | 0.456              | Decoder-only patched transformer (excluded: license)                                                |
| Toto-2.0 2.5B / 313M  | 2.5B / 313M | Apache-2.0     | 0.476 / 0.481      | Patched pre-norm transformer, alternating time and variate attention, one code path from 4M to 2.5B |
| Chronos-2             | 120M        | Apache-2.0     | 0.485              | T5-style encoder, patch 16, context 8192, 21 quantiles, group attention for covariates              |
| Toto-2.0 22M          | 22M         | Apache-2.0     | 0.496              |                                                                                                     |
| Reverso               | 2.6M        | MIT            | 0.610 (MASE 0.711) | Gated long convolution alternating with delta-rule linear attention; point forecasts                |

The spread among the best 28 single models is about 7 % (0.456 to 0.490). Every entry above 0.456 combines models.

### Why this order

- **Reverso first.** It is small (2.6M parameters), MIT-licensed, and exercises the parts no other model needs but the
  infrastructure must own: a long FFT convolution, a delta-rule recurrence (nearest to existing code, L2), an
  autoregressive rollout, and the evaluation's context preparation (missing values, padding, striding, sign-flip
  averaging). A full end-to-end port costs days, not weeks, and a published score checks it.
- **Chronos-2 second.** A transformer encoder in the BERT size class, which lattice already runs; it adds patching,
  quantile heads, and multi-series group attention, so the forecast type gains quantiles and the input gains
  covariates.
- **Toto-2.0 third.** One architecture across a 4M to 2.5B size ladder; it adds variate attention and makes model size a
  configuration choice rather than a port.

## Decision

### D1. One module, no new crate

`lattice_inference::forecast`, always compiled, no new dependency and no cargo feature. Forecasting needs the crate's
weight loading and CPU kernels (L3), so a separate crate would depend on lattice-inference anyway; a crate split is
revisited only if forecasting grows dependencies the inference crate should not carry.

### D2. Shared context preparation

`forecast::prep` owns missing-value imputation (linear interpolation between observed points with constant extension,
else forward then backward fill, else zero), left padding with the first value, striding, and linear resampling. The
arithmetic follows `numpy.interp` in f64 because the published evaluations prepare contexts with it; a model reproduces
its published score only when it sees the same context. Each model's predictor composes these steps in its reference
order.

### D3. FFT in-house

A radix-2 complex FFT in f64, crate-private, with precomputed kernel spectra. Reverso needs one fixed power-of-two length
per model; f64 keeps the convolution's own rounding below the f32 values around it, and the cost at these lengths is
negligible next to the projections. A general-length or SIMD FFT is added only when a model needs it, with a bench.

### D4. Weights are converted outside lattice

Upstream checkpoints are PyTorch pickles. Lattice reads only safetensors: conversion happens in a separate step that
loads with `weights_only=True`, strips training-only buffers, and writes f32 tensors under the reference state-dict
names. Loading requires every tensor the configuration needs, at its exact shape, and refuses any tensor left over. A
leftover tensor means the configuration and the checkpoint disagree. The upstream torch-native Reverso port drops the
output position embedding that the full checkpoint was trained with and cannot load that checkpoint strictly, so this
refusal is the guard that matters most.

### D5. Model configuration from the reference file

Each model reads its reference configuration file (`args.json` for Reverso) and refuses configurations it does not
compute exactly (for Reverso: non-power-of-two windows, unknown block types, even gating kernels, a head width that
differs from the window). Unsupported means refused, never approximated.

### D6. Precision

f32 compute, matching the reference's f32 path. Where the reference rounds to bfloat16 explicitly inside the model
(Reverso's gated convolution), lattice rounds at the same points, round-to-nearest-even. Published scores that were
produced under mixed-precision autocast are matched by score, not bit for bit.

### D7. Forecast output

Point forecasts in this first step. Quantile forecasts arrive with Chronos-2 (21 quantiles) as a `Forecast` type
carrying a point forecast and named quantile levels; Reverso's point forecast is reported to quantile metrics as a
degenerate distribution, as its reference evaluation does.

### D8. Verification

- **Goldens.** Each port is tested against reference outputs generated from the pinned upstream source and pinned
  checkpoints: every layer's output, one forward call, an autoregressive rollout, the rollout of the negated series, and
  the context preparation on missing-value, short and constant inputs. Thresholds are registered before the first run.
  For Reverso: model outputs within 1e-3 of `max |reference| + 1`, each layer within 1e-2 of its largest magnitude,
  context preparation bit-identical.
- **Score.** Each port reproduces its published GIFT-Eval aggregate within a stated tolerance before it is called done.
  The harness is split: a Python step reads the benchmark data and scores forecasts with the benchmark's own metric
  code; lattice produces every forecast through the CLI. No model code runs in Python.
- **Mutation.** Each architectural feature the goldens cover (state weaving, output positions, the bfloat16 casts) has a
  control that removes it and must fail.

### D9. CLI

`lattice forecast --model <dir>` reads one series per JSON line (`target` with `null` for missing values, `horizon`,
optional `downsample` and `flip`) and writes one forecast per line. It is the harness's only entry point into lattice.

### D10. Combination after the three ports

With three model families running locally, a learned per-series combination (selection or weighting by series
features) is built on the same module and scored on GIFT-Eval. The aim is a combination whose aggregate CRPS, on the
same protocol, is lower than 0.419, the best published aggregate today. It trains on the benchmark's training split only; a
result is reported with the split and the protocol, and no leaderboard submission is made without an explicit
decision to publish.

## Alternatives considered

- **A separate `lattice-forecast` crate.** Cleaner boundary, but it would depend on lattice-inference for loading and
  kernels, and new crates need explicit approval. Deferred (D1).
- **`rustfft` / `realfft`.** Mature and pure Rust, but a dependency for one power-of-two length; reconsidered if a model
  needs arbitrary lengths or FFT becomes hot.
- **Reuse `gated_delta_net_step`.** Same recurrence with the decay fixed at one, but bound to the Qwen3.5 layer's
  config and fused weights (L2). Reverso's layer has separate projections, separate short convolutions and a per-head
  RMSNorm. A shared kernel is extracted once both paths have benches.
- **Chronos-2 or Toto-2.0 first.** Both score better. Starting with Reverso builds the convolution, recurrence, rollout and
  preparation layers against a small model with a published score, before the larger transformers arrive.
- **Running the reference model in Python for evaluation.** Fast to set up, but it measures the reference, not lattice.

## Consequences

- New public surface: `forecast::{prep, Reverso, ReversoConfig, PredictOptions}`. Marked unstable until the second model
  lands and the shared types settle.
- Benchmarks: the module is new code that no existing bench target reaches; the first performance change to it carries
  its own bench.
- Evaluation cost: a GIFT-Eval pass is a long CPU job (thousands of series, several rollout calls each), run on build
  hardware rather than a laptop.

## Plan

1. Reverso module, goldens, tests (this ADR's first implementation PR).
2. CLI and the GIFT-Eval harness; reproduce Reverso's published MASE and CRPS.
3. Chronos-2: encoder, patching, quantile head, `Forecast` type with quantiles; goldens; score.
4. Toto-2.0: size ladder on one code path; goldens; score.
5. Learned combination (D10).
