#!/usr/bin/env python3
"""ADR-064 Phase-0: robust linear fit of decode throughput vs context length.

Reads SLOPEFIT lines from stdin (produced by bench_decode_slopefit binary):
  SLOPEFIT ctx=<N> tokens=<T> warmup_ms=0.0 measure_ms=<M> rep=<R>

Derives per-token cost:
  per_tok_ms = measure_ms / tokens

Fits  per_tok_ms = slope * ctx + intercept  via Theil-Sen (rank-based, outlier
resistant) on the raw per-repeat pairs, then bootstraps 95% CI on slope and
intercept via 2000 resamples from the same raw pairs.  Point estimate and
bootstrap are computed on identical data so the point estimate is always inside
its bootstrap percentile interval by construction.

Emits JSON to stdout matching the ADR-064 schema:
  {slope, slope_ci95, intercept, intercept_ci95, R2,
   tok_s_at_ctx_measured, tok_s_at_ctx_extrapolated,
   dispatches_per_token, command_buffers_per_token}

tok_s_at_ctx_measured contains only context points actually measured.
tok_s_at_ctx_extrapolated contains fit-derived values at standard reporting
  contexts that were NOT in the measurement grid.

dispatches_per_token and command_buffers_per_token are null — the engine does
not expose those counts via the public API (honest-nil per ADR-064 §null policy).
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Parse SLOPEFIT lines from stdin
# ---------------------------------------------------------------------------

LINE_RE = re.compile(
    r"SLOPEFIT ctx=(\d+) tokens=(\d+) warmup_ms=[\d.]+ measure_ms=([\d.]+) rep=\d+"
)

raw: dict[int, list[float]] = defaultdict(list)

for line in sys.stdin:
    m = LINE_RE.search(line)
    if m:
        ctx = int(m.group(1))
        tokens = int(m.group(2))
        measure_ms = float(m.group(3))
        if tokens > 0:
            raw[ctx].append(measure_ms / tokens)

if len(raw) < 2:
    print(json.dumps({"error": "need at least 2 context points", "n": len(raw)}))
    sys.exit(1)

ctx_vals: list[int] = sorted(raw.keys())

# ---------------------------------------------------------------------------
# Flatten all raw (ctx, per_tok_ms) pairs — point estimate and bootstrap both
# operate on this same dataset so the point estimate is inside the CI by
# construction (it is the Theil-Sen estimate of the population from which
# bootstrap samples are drawn).
# ---------------------------------------------------------------------------

all_pairs: list[tuple[float, float]] = []
for ctx in ctx_vals:
    for ptm in raw[ctx]:
        all_pairs.append((float(ctx), ptm))

all_x = np.array([p[0] for p in all_pairs])
all_y = np.array([p[1] for p in all_pairs])


# ---------------------------------------------------------------------------
# Theil-Sen estimator (robust to ~29% outliers)
# ---------------------------------------------------------------------------

def theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) via Theil-Sen median-of-slopes."""
    slopes = []
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if dx != 0:
                slopes.append((y[j] - y[i]) / dx)
    if slopes:
        slope = float(np.median(slopes))
    else:
        slope = float((y[-1] - y[0]) / (x[-1] - x[0]))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


# Point estimate on all raw pairs.
slope, intercept = theil_sen(all_x, all_y)


# ---------------------------------------------------------------------------
# Bootstrap 95% CI (2000 resamples) — same raw pairs as point estimate.
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
N_BOOT = 2000
boot_slopes: list[float] = []
boot_intercepts: list[float] = []

n_pairs = len(all_pairs)
for _ in range(N_BOOT):
    idx = rng.integers(0, n_pairs, size=n_pairs)
    bx = all_x[idx]
    by = all_y[idx]
    # Need at least 2 distinct ctx values in the bootstrap sample.
    if len(np.unique(bx)) < 2:
        continue
    bs, bi = theil_sen(bx, by)
    boot_slopes.append(bs)
    boot_intercepts.append(bi)

if len(boot_slopes) < 100:
    slope_ci95: list[float | None] = [None, None]
    intercept_ci95: list[float | None] = [None, None]
else:
    slope_ci95 = [
        float(np.percentile(boot_slopes, 2.5)),
        float(np.percentile(boot_slopes, 97.5)),
    ]
    intercept_ci95 = [
        float(np.percentile(boot_intercepts, 2.5)),
        float(np.percentile(boot_intercepts, 97.5)),
    ]


# ---------------------------------------------------------------------------
# R²  (against the Theil-Sen fit, computed on the per-ctx trimmed means so
# the summary statistic is over the same aggregation the report describes)
# ---------------------------------------------------------------------------

trimmed_ptm: list[float] = []
for ctx in ctx_vals:
    pts = sorted(raw[ctx])
    trim = max(1, int(len(pts) * 0.20))
    trimmed = pts[trim : len(pts) - trim] if len(pts) > 2 * trim else pts
    trimmed_ptm.append(float(np.mean(trimmed)))

xs_summary = np.array(ctx_vals, dtype=float)
ys_summary = np.array(trimmed_ptm, dtype=float)

y_hat = slope * xs_summary + intercept
ss_res = float(np.sum((ys_summary - y_hat) ** 2))
ss_tot = float(np.sum((ys_summary - np.mean(ys_summary)) ** 2))
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


# ---------------------------------------------------------------------------
# tok/s: separate measured vs extrapolated
# ---------------------------------------------------------------------------

measured_ctx_set = set(ctx_vals)
# Standard reporting contexts across typical agentic workloads.
report_ctxs = [64, 256, 512, 1024, 2048, 4096, 8192, 16384]

tok_s_at_ctx_measured: dict[str, float | None] = {}
tok_s_at_ctx_extrapolated: dict[str, float | None] = {}

for c in report_ctxs:
    ptm = slope * c + intercept
    tps = round(1000.0 / ptm, 2) if ptm > 0 else None
    if c in measured_ctx_set:
        tok_s_at_ctx_measured[str(c)] = tps
    else:
        tok_s_at_ctx_extrapolated[str(c)] = tps

# Also include any measured contexts that aren't in the standard list.
for c in ctx_vals:
    if str(c) not in tok_s_at_ctx_measured:
        ptm = slope * c + intercept
        tok_s_at_ctx_measured[str(c)] = round(1000.0 / ptm, 2) if ptm > 0 else None


# ---------------------------------------------------------------------------
# Emit JSON
# ---------------------------------------------------------------------------

result = {
    "slope": round(slope, 8),
    "slope_ci95": slope_ci95,
    "intercept": round(intercept, 4),
    "intercept_ci95": intercept_ci95,
    "R2": round(r2, 6),
    "tok_s_at_ctx_measured": tok_s_at_ctx_measured,
    "tok_s_at_ctx_extrapolated": tok_s_at_ctx_extrapolated,
    # Honest-nil: not accessible via public MetalQwen35State API.
    "dispatches_per_token": None,
    "command_buffers_per_token": None,
    # Diagnostics (not in ADR spec — informational only).
    "_n_ctx_points": len(ctx_vals),
    "_n_raw_repeats": len(all_pairs),
    "_n_boot_samples": len(boot_slopes),
    "_ctx_grid": ctx_vals,
    "_per_tok_ms_trimmed": [round(v, 4) for v in trimmed_ptm],
}

print(json.dumps(result, indent=2))
