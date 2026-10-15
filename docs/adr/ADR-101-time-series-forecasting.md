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
are the leaderboard's own. Source: the per-model raw result files of the GIFT-Eval leaderboard (Hugging Face Space
`Salesforce/GIFT-Eval`), snapshot as of 2026-10-09.

| Model                 | Params      | License        | CRPS               | Shape                                                                                               |
| --------------------- | ----------- | -------------- | ------------------ | --------------------------------------------------------------------------------------------------- |
| Best agentic ensemble | n/a         | n/a            | 0.419              | Several models combined per series (EXAONE-Forecast-Agent)                                          |
| TimesFM-3             | n/a         | Non-commercial | 0.456              | Decoder-only patched transformer (excluded: license)                                                |
| Toto-2.0 2.5B / 313M  | 2.5B / 313M | Apache-2.0     | 0.476 / 0.481      | Patched pre-norm transformer, alternating time and variate attention, one code path from 4M to 2.5B |
| Chronos-2             | 120M        | Apache-2.0     | 0.485              | T5-style encoder, patch 16, context 8192, 21 quantiles, group attention for covariates              |
| Toto-2.0 22M          | 22M         | Apache-2.0     | 0.496              |                                                                                                     |
| Reverso               | 2.6M        | MIT            | 0.610 (MASE 0.711) | Gated long convolution alternating with delta-rule linear attention; point forecasts                |

The spread among the best 28 single models is about 7 % (0.456 to 0.490). Every entry above 0.456 combines models.

### Why this order

- **Reverso first.** It is small (2.6M parameters), MIT-licensed, and exercises the parts no other model needs but the
  infrastructure must own: a long FFT convolution, a delta-rule recurrence (nearest to existing code, L2), an
  autoregressive rollout, and the evaluation's prediction path (missing values, padding, striding, and sign-flip
  averaging over a second rollout). A full end-to-end port costs days, not weeks, and a published score checks it.
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

Sign-flip averaging belongs to each model's predictor (for Reverso, `Reverso::predict`), not to `prep`, because it needs
a second rollout: with the flag set, the predictor also rolls out the negated context and returns
`0.5 * (f(x) - f(-x))` elementwise, before any resampling, so the averaged forecast keeps the sign of the input.

### D3. FFT in-house

A radix-2 complex FFT in f64, crate-private, with precomputed kernel spectra. Reverso needs one fixed power-of-two length
per model; f64 keeps the convolution's own rounding below the f32 values around it, and the cost at these lengths is
negligible next to the projections. A general-length or SIMD FFT is added only when a model needs it, with a bench.

### D4. Weights are converted outside lattice

Upstream checkpoints are PyTorch pickles. Lattice reads only safetensors (ADR-003): conversion happens in a separate step
that checks each checkpoint's sha256 against its pin (D8), loads it with `weights_only=True` under a pinned torch
version, drops the upstream FFT-convolution library's `flashfftconv.*` entries (the reference model loads strictly
without them), and writes f32 tensors under the reference state-dict names. The golden generator loads checkpoints the same way. Loading requires every tensor the configuration needs, at its exact shape, and refuses any tensor left over. A
leftover tensor means the configuration and the checkpoint disagree. The upstream torch-native Reverso port drops the
output position embedding that the full checkpoint was trained with and cannot load that checkpoint strictly, so this
refusal is the guard that matters most.

### D5. Model configuration from the reference file

Each model reads its reference configuration file (`args.json` for Reverso) and refuses configurations it does not
compute exactly (for Reverso: non-power-of-two windows, unknown block types, even gating kernels, a head width that
differs from the window). Unsupported means refused, never approximated.

### D6. Precision

f32 compute, matching the reference's f32 path, with two named f64 exceptions: context preparation (D2) and the FFT
convolution (D3), whose results are rounded to f32 before the next layer. The goldens are not evidence for the f64
choice in D3 over an f32 FFT; it is a design choice, not a measured requirement. Where the reference rounds to bfloat16
explicitly inside the model (Reverso's gated convolution), lattice rounds at the same points, round-to-nearest-even.
Published scores that were produced under mixed-precision autocast are matched by score, not bit for bit.

### D7. Forecast output

Point forecasts in this first step. Quantile forecasts arrive with Chronos-2 (21 quantiles) as a `Forecast` type
carrying a point forecast and named quantile levels; Reverso's point forecast is reported to quantile metrics as a
degenerate distribution, as its reference evaluation does.

### D8. Verification

- **Pins.** Each port names its upstream source commit and the sha256 of every checkpoint it converts. Reverso:
  `shinfxh/reverso` at `35f724d3ccf26527f17bc774e897ff0c07d0761a`, checkpoints
  `checkpoints/<name>/checkpoint.pth` with sha256 `91e4244e786ca26ffaaddd452775bd5d57bf7bd257493f6849af2059468e1a8d`
  (`reverso`, 2.6M), `a475a728a7d6a625bead27b283abd4e7746d3a7099086e5a8cf6a23bb647502b` (`reverso_small`) and
  `7ad57992b268deb8f1d4c1cff23db6383b1924d7880ba8edf28002ec4613f389` (`reverso_nano`). The benchmark data and metric
  code are pinned the same way: GIFT-Eval code at commit `577a65cfa3acb812405c00d989230bf59f7f8a3d` with its lockfile,
  and the `Salesforce/GiftEval` dataset at revision `30841734ac5cfddbd0c3bad6d09d2b6b32becbb0`.
- **Goldens.** Each port is tested against reference outputs generated from the pinned source and checkpoints: every
  layer's output, one forward call, an autoregressive rollout, the rollout of the negated series, the sign-flip
  averaged prediction, and the context preparation on missing-value, short and constant inputs. Thresholds are
  registered before the first run. For Reverso, with `ref` the reference tensor and `got` lattice's: model outputs
  satisfy `max |got - ref| / (max |ref| + 1) <= 1e-3`; each layer satisfies `max |got - ref| / max |ref| <= 1e-2`;
  context preparation is identical in f32 bits.
- **Score.** Each port reproduces its published GIFT-Eval aggregate MASE and CRPS within a tolerance registered before
  its first scored run. For Reverso: each aggregate within 2 % (relative) of the published 0.711 and 0.610, and the mean
  over the 97 configurations of `|log(lattice MASE / published MASE)|` below 0.03. The published run used bfloat16
  autocast on a GPU and lattice computes f32 (D6), so equality is not expected; the score checks the whole harness end
  to end, and the goldens, not the score, establish that the right model was ported. The harness is split: a Python
  step reads the benchmark data and scores forecasts with the benchmark's own metric code; lattice produces every
  forecast through the CLI. No model code runs in Python.
- **Mutation.** Each architectural feature the goldens cover has a control that removes it and must fail. For Reverso:
  state weaving (in each attention block that is neither the first nor the last block of the stack, the final
  position's input is added to the first position's before the block runs), the output position table, and the
  bfloat16 casts.

### D9. CLI

`lattice forecast --model <dir>` reads one series per JSON line (`id`, `target` with `null` for missing values, `horizon`,
optional `downsample` and `flip`, the sign-flip averaging of D2) and writes one line per input, `{"id", "forecast"}`, with
`forecast` the point forecast of length `horizon`. The scorer turns each point forecast into the degenerate distribution
of D7. It is the harness's only entry point into lattice.

### D10. Combination after the three ports

With three model families running locally, a learned per-series combination (selection or weighting by series
features, the same shape of decision as ADR-096's learned router) is built on the same module and scored on GIFT-Eval. The aim is a combination whose aggregate CRPS, on the
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
