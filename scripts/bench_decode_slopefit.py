#!/usr/bin/env python3
"""ADR-064 GPU/decode gate: robust linear fit of decode throughput vs context length.

Reads stdout lines produced by the `bench_decode_slopefit` binary:
  SLOPEFIT ctx=<N> tokens=<T> warmup_ms=0.0 measure_ms=<M> rep=<R>
  SLOPEFIT_META kv_cache_len=<N> warmup=<N> measure=<N> repeats=<N>   (once)
  SLOPEFIT_META ctx=<N> actual_prompt_tokens=<N>                      (per ctx)

Fits  per_tok_ms = intercept_ms + slope_ms_per_ctx_tok * ctx  via Theil-Sen
(rank-based, outlier resistant) on the raw per-repeat pairs, then bootstraps a
1-sided 95% CI (5th/95th percentile bounds) via 2000 resamples from the same
raw pairs.

Mandatory TBV self-checks run BEFORE any success JSON is emitted (see
docs/adr/ADR-064 harness spec). Any failure prints a JSON error object and
exits nonzero -- this is what catches the KV-cache-cap / EOS self-corruption
class of bug where the harness silently reports tokens=1.

Emits the ADR-064 perf-baselines-compatible JSON schema (`benches` map keyed
by `<category>/<metric>[/<ctx>]`) to stdout, and to --out if given.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SLOPEFIT_RE = re.compile(
    r"SLOPEFIT ctx=(\d+) tokens=(\d+) warmup_ms=[\d.]+ measure_ms=([\d.]+) rep=(\d+)"
)
META_RUN_RE = re.compile(
    r"SLOPEFIT_META kv_cache_len=(\d+) warmup=(\d+) measure=(\d+) repeats=(\d+)"
)
META_CTX_RE = re.compile(r"SLOPEFIT_META ctx=(\d+) actual_prompt_tokens=(\d+)")

REPORT_CTXS = [64, 256, 512, 1024, 2048, 4096, 8192, 16384]
N_BOOT = 2000
CI_LOW_PCT = 5.0
CI_HIGH_PCT = 95.0


def fail(msg: str, **extra: object) -> None:
    print(json.dumps({"error": msg, **extra}, indent=2))
    sys.exit(1)


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


def bootstrap_one_sided_ci(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> tuple[float | None, float | None]:
    """1-sided 95% CI bounds (5th/95th percentile) for the Theil-Sen slope."""
    boot_slopes: list[float] = []
    n_pairs = len(x)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        bx, by = x[idx], y[idx]
        if len(np.unique(bx)) < 2:
            continue
        bs, _ = theil_sen(bx, by)
        boot_slopes.append(bs)
    if len(boot_slopes) < 100:
        return None, None
    return (
        float(np.percentile(boot_slopes, CI_LOW_PCT)),
        float(np.percentile(boot_slopes, CI_HIGH_PCT)),
    )


def bench_entry(
    value: float | None,
    ci_low: float | None,
    ci_high: float | None,
    unit: str,
    higher_is_better: bool,
) -> dict:
    return {
        "value": value,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "unit": unit,
        "higher_is_better": higher_is_better,
    }


def null_bench(unit: str, higher_is_better: bool) -> dict:
    return bench_entry(None, None, None, unit, higher_is_better)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    is_full = os.environ.get("SLOPEFIT_FULL", "") == "1"

    raw: dict[int, list[float]] = defaultdict(list)
    tokens_by_ctx: dict[int, list[int]] = defaultdict(list)
    meta_ctx: dict[int, int] = {}
    meta_run: dict[str, int] | None = None

    for line in sys.stdin:
        m = SLOPEFIT_RE.search(line)
        if m:
            ctx, tokens, measure_ms = int(m.group(1)), int(m.group(2)), float(m.group(3))
            tokens_by_ctx[ctx].append(tokens)
            if tokens > 0:
                raw[ctx].append(measure_ms / tokens)
            continue
        m = META_RUN_RE.search(line)
        if m:
            meta_run = {
                "kv_cache_len": int(m.group(1)),
                "warmup": int(m.group(2)),
                "measure": int(m.group(3)),
                "repeats": int(m.group(4)),
            }
            continue
        m = META_CTX_RE.search(line)
        if m:
            meta_ctx[int(m.group(1))] = int(m.group(2))
            continue

    # --- TBV self-checks (ADR-064 harness spec) -----------------------------
    if meta_run is None:
        fail("missing SLOPEFIT_META run line -- binary did not emit expected metadata")

    measure_target = meta_run["measure"]
    kv_cache_len = meta_run["kv_cache_len"]
    warmup_n = meta_run["warmup"]

    for ctx, tok_list in tokens_by_ctx.items():
        for tokens in tok_list:
            if tokens <= 1:
                fail(
                    "tokens<=1 self-corruption detected (EOS/KV-cache-cap class bug)",
                    ctx=ctx,
                    tokens=tokens,
                )
            if tokens < 0.95 * measure_target:
                fail(
                    "measured tokens far below requested measure length",
                    ctx=ctx,
                    tokens=tokens,
                    measure_target=measure_target,
                )

    for ctx, actual_prompt_tokens in meta_ctx.items():
        if actual_prompt_tokens + max(warmup_n, measure_target) + 16 > kv_cache_len:
            fail(
                "KV-cache-cap corruption: prompt + decode horizon exceeds cache_len",
                ctx=ctx,
                actual_prompt_tokens=actual_prompt_tokens,
                kv_cache_len=kv_cache_len,
            )

    if len(raw) < 3:
        fail("fewer than 3 context points measured", n_ctx=len(raw))

    if is_full:
        for ctx, ptm_list in raw.items():
            if len(ptm_list) < 3:
                fail(
                    "fewer than 3 repeats per context in production mode",
                    ctx=ctx,
                    n_repeats=len(ptm_list),
                )

    ctx_vals = sorted(raw.keys())
    all_pairs = [(float(ctx), ptm) for ctx in ctx_vals for ptm in raw[ctx]]
    all_x = np.array([p[0] for p in all_pairs])
    all_y = np.array([p[1] for p in all_pairs])

    slope, intercept = theil_sen(all_x, all_y)

    rng = np.random.default_rng(42)
    # Slope and intercept CIs computed on the same joint bootstrap resample so
    # combined (ctx-scaled) bounds stay internally consistent.
    boot_slopes: list[float] = []
    boot_intercepts: list[float] = []
    n_pairs = len(all_pairs)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        bx, by = all_x[idx], all_y[idx]
        if len(np.unique(bx)) < 2:
            continue
        bs, bi = theil_sen(bx, by)
        boot_slopes.append(bs)
        boot_intercepts.append(bi)

    if len(boot_slopes) < 100:
        slope_lo = slope_hi = intercept_lo = intercept_hi = None
    else:
        slope_lo = float(np.percentile(boot_slopes, CI_LOW_PCT))
        slope_hi = float(np.percentile(boot_slopes, CI_HIGH_PCT))
        intercept_lo = float(np.percentile(boot_intercepts, CI_LOW_PCT))
        intercept_hi = float(np.percentile(boot_intercepts, CI_HIGH_PCT))

    if slope <= 0 or intercept <= 0:
        fail("fitted slope/intercept non-positive -- physically insane fit", slope=slope, intercept=intercept)

    # --- Per-context tok/s (measured: empirical bootstrap; else: fit-derived) -
    benches: dict[str, dict] = {}

    for c in REPORT_CTXS:
        if c in raw:
            samples = np.array(raw[c])
            point_ms = float(np.mean(samples))
            boot_means = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, len(samples), size=len(samples))
                boot_means.append(float(np.mean(samples[idx])))
            ms_lo = float(np.percentile(boot_means, CI_LOW_PCT))
            ms_hi = float(np.percentile(boot_means, CI_HIGH_PCT))
        else:
            if slope_lo is None:
                continue
            point_ms = slope * c + intercept
            ms_lo = slope_lo * c + intercept_lo
            ms_hi = slope_hi * c + intercept_hi

        tok_s_val = round(1000.0 / point_ms, 4) if point_ms > 0 else None
        tok_s_lo = round(1000.0 / ms_hi, 4) if ms_hi and ms_hi > 0 else None
        tok_s_hi = round(1000.0 / ms_lo, 4) if ms_lo and ms_lo > 0 else None
        if tok_s_val is not None and not (1.0 <= tok_s_val <= 1000.0):
            fail("derived tok/s outside sane [1,1000] range", ctx=c, tok_s=tok_s_val)

        benches[f"decode/tok_s/{c}"] = bench_entry(
            tok_s_val, tok_s_lo, tok_s_hi, "tok_s", True
        )

    benches["decode/slope_ms_per_ctx_tok"] = bench_entry(
        round(slope, 8), slope_lo, slope_hi, "ms/ctx_tok", False
    )
    benches["decode/intercept_ms"] = bench_entry(
        round(intercept, 4), intercept_lo, intercept_hi, "ms", False
    )

    # --- Honest-nil rows: producers not yet landed --------------------------
    benches["decode/ttft_ms/4096"] = null_bench("ms", False)
    benches["decode/ttft_ms/16384"] = null_bench("ms", False)
    benches["decode/dispatches_per_token"] = null_bench("count", False)
    benches["decode/command_buffers_per_token"] = null_bench("count", False)
    benches["quality/ppl_delta/f16"] = null_bench("ppl_delta", False)
    benches["quality/ppl_delta/bf16"] = null_bench("ppl_delta", False)
    benches["quality/ppl_delta/q4_kv"] = null_bench("ppl_delta", False)
    benches["quality/greedy_agreement"] = null_bench("ratio", True)
    benches["quality/topk_exact"] = null_bench("ratio", True)
    benches["contention/loss_pp/w4"] = null_bench("pp", False)
    benches["contention/loss_frac/w10"] = null_bench("fraction", False)
    # This harness's own TBV self-checks (tokens sanity, KV-cache-cap, fit
    # sanity) all passed by construction if execution reached this point.
    benches["runtime/kv_layout_assertion"] = bench_entry(1.0, 1.0, 1.0, "bool01", True)

    repo_root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    run_id = str(uuid.uuid4())
    total_calls = sum(len(v) for v in tokens_by_ctx.values())

    result = {
        "commit": commit,
        "date": datetime.now(timezone.utc).isoformat(),
        "arch": "m2max-metal",
        "benches": benches,
        "kernel_metrics": [
            {
                "run_id": run_id,
                "op": "decode_total",
                "layer": None,
                "calls": total_calls,
                "gpu_ns_total": 0,
                "modeled_flops_total": None,
                "modeled_bytes_total": None,
                "achieved_gbps": None,
                "achieved_gflops": None,
                "bandwidth_utilization": None,
            }
        ],
        # Honest-nil: no per-kernel dispatch tracing is exposed by the public
        # MetalQwen35State API, so there are no raw events to report.
        "raw_kernel_events_jsonl": [],
        "repro": {
            "model_dir": os.environ.get(
                "LATTICE_MODEL_DIR", str(Path.home() / ".lattice/models/qwen3.5-0.8b")
            ),
            "timing_mode": "CommandBufferAggregate",
            "contexts": ctx_vals,
            "warmup": warmup_n,
            "measure": measure_target,
            "repeats": meta_run["repeats"],
            "tbv_self_check": "passed",
        },
    }

    out_json = json.dumps(result, indent=2)
    print(out_json)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json + "\n")


if __name__ == "__main__":
    main()
