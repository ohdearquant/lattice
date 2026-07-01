#!/usr/bin/env python3
"""Apply the ADR-064 GPU/decode perf-regression gate table to current vs baseline JSON.

Follows the ADR-058 lower-bound-of-CI method: a row FAILs only when the
*conservative* end of the confidence interval already confirms a regression,
so measurement noise cannot flake the gate. See docs/adr/ADR-058-cpu-perf-regression-ci.md
section D3 and issue #167's gate table.

Both --current and --baseline take the harness JSON shape produced by
scripts/bench_decode_slopefit.sh (root keys: commit, date, arch, benches, ...).
Each entry under "benches" has {value, ci95_low, ci95_high, unit, higher_is_better}.
A row is INCOMPLETE (not evaluated) when a metric it depends on is missing/null in
either file -- this is an honest-nil state, not a pass.

Usage:
  adr064-gpu-decode-gate.py --current CURRENT.json --baseline BASELINE.json [--out REPORT.md]

Exit codes:
  0 - all evaluated rows PASS, nothing INCOMPLETE
  1 - at least one row FAIL (confirmed regression)
  2 - no FAIL, but at least one row INCOMPLETE (missing data)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RowResult:
    name: str
    verdict: str  # "PASS" | "FAIL" | "INCOMPLETE"
    reasons: list[str] = field(default_factory=list)


def _bench(benches: dict[str, Any], key: str) -> dict[str, Any] | None:
    entry = benches.get(key)
    if entry is None:
        return None
    if (
        entry.get("value") is None
        or entry.get("ci95_low") is None
        or entry.get("ci95_high") is None
    ):
        return None
    return entry


def _lb_lower_is_better(current: dict[str, Any], baseline: dict[str, Any]) -> float:
    """Fractional regression lower bound for a lower-is-better metric (e.g. time, ms)."""
    return (current["ci95_low"] - baseline["ci95_high"]) / baseline["ci95_high"]


def _lb_higher_is_better(current: dict[str, Any], baseline: dict[str, Any]) -> float:
    """Fractional regression lower bound for a higher-is-better metric (e.g. tok/s)."""
    return (baseline["ci95_low"] - current["ci95_high"]) / baseline["ci95_low"]


def _decode_tok_s_keys(benches: dict[str, Any]) -> list[str]:
    return sorted(k for k in benches if k.startswith("decode/tok_s/"))


def eval_decode_row(current: dict[str, Any], baseline: dict[str, Any]) -> RowResult:
    name = "decode_tok_s_slope_intercept"
    reasons: list[str] = []

    slope_c = _bench(current, "decode/slope_ms_per_ctx_tok")
    slope_b = _bench(baseline, "decode/slope_ms_per_ctx_tok")
    intercept_c = _bench(current, "decode/intercept_ms")
    intercept_b = _bench(baseline, "decode/intercept_ms")
    if slope_c is None or slope_b is None or intercept_c is None or intercept_b is None:
        return RowResult(
            name, "INCOMPLETE", ["decode/slope_ms_per_ctx_tok or decode/intercept_ms missing"]
        )

    slope_lb = _lb_lower_is_better(slope_c, slope_b)
    if slope_lb > 0.05:
        reasons.append(f"decode/slope_ms_per_ctx_tok lb={slope_lb:.4f} > 0.05")

    intercept_lb = _lb_lower_is_better(intercept_c, intercept_b)
    if intercept_lb > 0.07:
        reasons.append(f"decode/intercept_ms lb={intercept_lb:.4f} > 0.07")

    current_ctx_keys = set(_decode_tok_s_keys(current))
    baseline_ctx_keys = set(_decode_tok_s_keys(baseline))
    for key in sorted(current_ctx_keys & baseline_ctx_keys):
        c = _bench(current, key)
        b = _bench(baseline, key)
        if c is None or b is None:
            continue
        lb = _lb_higher_is_better(c, b)
        if lb > 0.07:
            reasons.append(f"{key} lb={lb:.4f} > 0.07")

    return RowResult(name, "FAIL" if reasons else "PASS", reasons)


def eval_ttft_dispatch_row(current: dict[str, Any], baseline: dict[str, Any]) -> RowResult:
    name = "ttft_dispatch"
    required = [
        "decode/ttft_ms/4096",
        "decode/ttft_ms/16384",
        "decode/dispatches_per_token",
        "decode/command_buffers_per_token",
    ]
    entries_c = {k: _bench(current, k) for k in required}
    entries_b = {k: _bench(baseline, k) for k in required}
    if any(entries_c[k] is None or entries_b[k] is None for k in required):
        missing = [k for k in required if entries_c[k] is None or entries_b[k] is None]
        return RowResult(name, "INCOMPLETE", [f"missing: {', '.join(missing)}"])

    reasons: list[str] = []

    for key, threshold in (("decode/ttft_ms/4096", 0.10), ("decode/ttft_ms/16384", 0.10)):
        lb = _lb_lower_is_better(entries_c[key], entries_b[key])
        if lb > threshold:
            reasons.append(f"{key} lb={lb:.4f} > {threshold}")

    disp_c, disp_b = (
        entries_c["decode/dispatches_per_token"],
        entries_b["decode/dispatches_per_token"],
    )
    disp_lb = _lb_lower_is_better(disp_c, disp_b)
    if disp_lb > 0.05:
        reasons.append(f"decode/dispatches_per_token lb={disp_lb:.4f} > 0.05")
    disp_abs = disp_c["ci95_low"] - disp_b["ci95_high"]
    if disp_abs > 10:
        reasons.append(f"decode/dispatches_per_token abs_delta={disp_abs:.4f} > 10")

    cmdbuf_c = entries_c["decode/command_buffers_per_token"]
    if cmdbuf_c["ci95_low"] > 2:
        reasons.append(f"decode/command_buffers_per_token ci95_low={cmdbuf_c['ci95_low']:.4f} > 2")

    return RowResult(name, "FAIL" if reasons else "PASS", reasons)


def eval_quality_row(current: dict[str, Any], baseline: dict[str, Any]) -> RowResult:
    name = "quality"
    required = [
        "quality/ppl_delta/f16",
        "quality/ppl_delta/bf16",
        "quality/ppl_delta/q4_kv",
        "quality/greedy_agreement",
        "quality/topk_exact",
    ]
    entries_c = {k: _bench(current, k) for k in required}
    if any(v is None for v in entries_c.values()):
        missing = [k for k, v in entries_c.items() if v is None]
        return RowResult(name, "INCOMPLETE", [f"missing: {', '.join(missing)}"])

    reasons: list[str] = []
    thresholds = {
        "quality/ppl_delta/f16": ("gt", 0.005),
        "quality/ppl_delta/bf16": ("gt", 0.05),
        "quality/ppl_delta/q4_kv": ("gt", 0.30),
        "quality/greedy_agreement": ("lt", 1.0),
        "quality/topk_exact": ("lt", 1.0),
    }
    for key, (op, threshold) in thresholds.items():
        low = entries_c[key]["ci95_low"]
        if op == "gt" and low > threshold:
            reasons.append(f"{key} ci95_low={low:.4f} > {threshold}")
        if op == "lt" and low < threshold:
            reasons.append(f"{key} ci95_low={low:.4f} < {threshold}")

    return RowResult(name, "FAIL" if reasons else "PASS", reasons)


def eval_contention_row(current: dict[str, Any], baseline: dict[str, Any]) -> RowResult:
    name = "contention_layout"
    required = ["contention/loss_pp/w4", "contention/loss_frac/w10", "runtime/kv_layout_assertion"]
    entries_c = {k: _bench(current, k) for k in required}
    entries_b = {k: _bench(baseline, k) for k in required}
    if any(entries_c[k] is None or entries_b[k] is None for k in required):
        missing = [k for k in required if entries_c[k] is None or entries_b[k] is None]
        return RowResult(name, "INCOMPLETE", [f"missing: {', '.join(missing)}"])

    reasons: list[str] = []

    w4_c, w4_b = entries_c["contention/loss_pp/w4"], entries_b["contention/loss_pp/w4"]
    w4_abs = w4_c["ci95_low"] - w4_b["ci95_high"]
    if w4_abs > 3.0:
        reasons.append(f"contention/loss_pp/w4 abs_delta={w4_abs:.4f} > 3.0")

    w10_c, w10_b = entries_c["contention/loss_frac/w10"], entries_b["contention/loss_frac/w10"]
    if w10_c["ci95_low"] > 0.10:
        reasons.append(f"contention/loss_frac/w10 ci95_low={w10_c['ci95_low']:.4f} > 0.10")
    w10_lb = _lb_lower_is_better(w10_c, w10_b)
    if w10_lb > 0.05:
        reasons.append(f"contention/loss_frac/w10 lb={w10_lb:.4f} > 0.05")

    kv_c = entries_c["runtime/kv_layout_assertion"]
    if kv_c["ci95_low"] < 1.0:
        reasons.append(f"runtime/kv_layout_assertion ci95_low={kv_c['ci95_low']:.4f} < 1.0")

    return RowResult(name, "FAIL" if reasons else "PASS", reasons)


ROW_EVALUATORS = [
    eval_decode_row,
    eval_ttft_dispatch_row,
    eval_quality_row,
    eval_contention_row,
]


def evaluate(current: dict[str, Any], baseline: dict[str, Any]) -> list[RowResult]:
    if current.get("arch") != baseline.get("arch"):
        raise ValueError(
            f"arch mismatch: current={current.get('arch')!r} baseline={baseline.get('arch')!r}"
        )
    current_benches = current.get("benches", {})
    baseline_benches = baseline.get("benches", {})
    return [fn(current_benches, baseline_benches) for fn in ROW_EVALUATORS]


def render_report(results: list[RowResult]) -> str:
    lines = ["# ADR-064 GPU/decode gate report", ""]
    lines.append("| Row | Verdict | Reasons |")
    lines.append("| --- | --- | --- |")
    for r in results:
        reasons = "; ".join(r.reasons) if r.reasons else "-"
        lines.append(f"| {r.name} | {r.verdict} | {reasons} |")
    return "\n".join(lines) + "\n"


def overall_exit_code(results: list[RowResult]) -> int:
    if any(r.verdict == "FAIL" for r in results):
        return 1
    if any(r.verdict == "INCOMPLETE" for r in results):
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    current = json.loads(args.current.read_text())
    baseline = json.loads(args.baseline.read_text())

    try:
        results = evaluate(current, baseline)
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    report = render_report(results)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    sys.stdout.write(report)

    return overall_exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
