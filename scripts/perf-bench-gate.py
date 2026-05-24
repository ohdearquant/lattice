#!/usr/bin/env python3
"""Parse Criterion change reports and apply the ADR-058 regression gate.

For every Criterion bench under target/criterion/, read change/estimates.json
(produced when running with --baseline <name>). Apply the rule:

  CI-lower of change in (-inf, +3%]    : pass silently
  CI-lower of change in (+3%, +7%]     : warn (PR-comment only, no fail)
  CI-lower of change in (+7%, +inf)    : FAIL
  Point estimate < -3% AND CI-upper<0% : celebrate

Usage:
  perf-bench-gate.py <criterion_root> <arch_label> [--out report.md]

Exit codes:
  0 — pass (no FAILs)
  1 — at least one FAIL (regression > 7% confirmed by 95% CI)
  2 — parse error / bad input
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Thresholds — ADR-058 §D3. Edit here; the workflow imports nothing else.
WARN_PCT = 3.0   # CI-lower above this => warning
FAIL_PCT = 7.0   # CI-lower above this => FAIL
CELEBRATE_PCT = -3.0  # point estimate below this AND CI-upper<0 => celebrate


@dataclass
class BenchResult:
    name: str            # e.g. "rms_norm/4096"
    point: float         # change.estimates.mean.point_estimate (fraction, not %)
    ci_low: float
    ci_high: float
    new_ns: float        # new median time, nanoseconds
    old_ns: float        # baseline median time, nanoseconds

    @property
    def point_pct(self) -> float: return self.point * 100.0
    @property
    def ci_low_pct(self) -> float: return self.ci_low * 100.0
    @property
    def ci_high_pct(self) -> float: return self.ci_high * 100.0

    def verdict(self) -> str:
        if self.ci_low_pct > FAIL_PCT:
            return "FAIL"
        if self.ci_low_pct > WARN_PCT:
            return "WARN"
        if self.point_pct < CELEBRATE_PCT and self.ci_high_pct < 0:
            return "WIN"
        return "PASS"


def find_change_files(root: Path) -> list[Path]:
    """Find every change/estimates.json under root (Criterion's per-bench output)."""
    return sorted(root.rglob("change/estimates.json"))


def parse_bench(change_file: Path, root: Path) -> BenchResult | None:
    """Parse one change/estimates.json + sibling new/estimates.json + base/estimates.json.

    Returns None if files are malformed (bench skipped, not failed).
    """
    bench_dir = change_file.parent.parent  # .../<bench>/<test>/
    rel = bench_dir.relative_to(root)
    name = str(rel)

    try:
        change = json.loads(change_file.read_text())
        mean = change["mean"]
        point = mean["point_estimate"]
        ci_low = mean["confidence_interval"]["lower_bound"]
        ci_high = mean["confidence_interval"]["upper_bound"]

        new_path = bench_dir / "new" / "estimates.json"
        new_ns = json.loads(new_path.read_text())["mean"]["point_estimate"]

        base_path = bench_dir / "base" / "estimates.json"
        old_ns = json.loads(base_path.read_text())["mean"]["point_estimate"]
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"warn: skipping {name}: {e}", file=sys.stderr)
        return None

    return BenchResult(name=name, point=point, ci_low=ci_low, ci_high=ci_high,
                       new_ns=new_ns, old_ns=old_ns)


def render_report(results: list[BenchResult], arch: str) -> str:
    fails = [r for r in results if r.verdict() == "FAIL"]
    warns = [r for r in results if r.verdict() == "WARN"]
    wins = [r for r in results if r.verdict() == "WIN"]

    lines = [f"### `{arch}` — perf regression report\n"]
    if fails:
        lines.append(f"**❌ {len(fails)} FAIL** (regression >{FAIL_PCT}% confirmed by 95% CI)")
    if warns:
        lines.append(f"**⚠ {len(warns)} WARN** (regression {WARN_PCT}-{FAIL_PCT}% confirmed)")
    if wins:
        lines.append(f"**🚀 {len(wins)} confirmed improvement**")
    if not (fails or warns or wins):
        lines.append(f"✅ All {len(results)} benches within noise band (±{WARN_PCT}%)")
    lines.append("")

    if fails or warns or wins:
        lines.append("| Bench | Δ point | 95% CI | new ns | base ns | verdict |")
        lines.append("|---|---:|---|---:|---:|---|")
        for r in sorted(fails + warns + wins, key=lambda r: -r.ci_low_pct):
            icon = {"FAIL": "❌", "WARN": "⚠", "WIN": "🚀"}[r.verdict()]
            lines.append(
                f"| `{r.name}` | {r.point_pct:+.2f}% | [{r.ci_low_pct:+.2f}%, {r.ci_high_pct:+.2f}%] "
                f"| {r.new_ns:.1f} | {r.old_ns:.1f} | {icon} {r.verdict()} |"
            )
        lines.append("")

    lines.append(
        f"<details><summary>All {len(results)} measurements</summary>\n\n"
        "| Bench | Δ point | CI-lower | CI-upper |\n|---|---:|---:|---:|"
    )
    for r in sorted(results, key=lambda r: r.name):
        lines.append(
            f"| `{r.name}` | {r.point_pct:+.2f}% | {r.ci_low_pct:+.2f}% | {r.ci_high_pct:+.2f}% |"
        )
    lines.append("\n</details>\n")
    lines.append(
        f"_Rule: CI-lower of change ≤{WARN_PCT}% passes silently; "
        f"({WARN_PCT}%, {FAIL_PCT}%] warns; >{FAIL_PCT}% fails. Override via PR label `bench-allow-regression`._\n"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("criterion_root", type=Path, help="Path to target/criterion (or per-bench root)")
    ap.add_argument("arch", help="Arch label for the report header (e.g. aarch64-linux)")
    ap.add_argument("--out", type=Path, help="Write markdown report to this path")
    args = ap.parse_args()

    if not args.criterion_root.exists():
        print(f"error: {args.criterion_root} does not exist", file=sys.stderr)
        return 2

    change_files = find_change_files(args.criterion_root)
    if not change_files:
        print(f"warn: no change/estimates.json under {args.criterion_root}; baseline missing?",
              file=sys.stderr)
        # Treat missing baseline as pass — first run on a bench has no comparison.
        if args.out:
            args.out.write_text(f"### `{args.arch}` — no baseline to compare\n\n"
                                f"No `change/estimates.json` found; this is expected on the "
                                f"first run for a bench. Future runs will gate against this run.\n")
        return 0

    results = []
    for cf in change_files:
        r = parse_bench(cf, args.criterion_root)
        if r is not None:
            results.append(r)

    if not results:
        print("error: change files found but all failed to parse", file=sys.stderr)
        return 2

    report = render_report(results, args.arch)
    print(report)
    if args.out:
        args.out.write_text(report)

    fails = sum(1 for r in results if r.verdict() == "FAIL")
    return 1 if fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
