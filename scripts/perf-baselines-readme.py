#!/usr/bin/env python3
"""Regenerate perf-baselines/README.md from history/ snapshots.

Run on the orphan `perf-baselines` branch after `bench-update.yml` adds a new
snapshot. Reads every `history/<timestamp>_<sha>_<arch>.json`, produces a
trend-rich README that renders natively on GitHub.

Sparklines use Unicode block characters (▁▂▃▄▅▆▇█), 8-step quantized.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SPARK_CHARS = "▁▂▃▄▅▆▇█"
HISTORY_DEPTH_SPARK = 20
HISTORY_DEPTH_DRILL = 30
HEADLINE_WINDOW_DAYS = 30
GITHUB_REPO_ENV = "GITHUB_REPOSITORY"  # owner/repo from GH Actions


@dataclass
class Snapshot:
    commit: str
    date: str           # ISO 8601
    arch: str
    benches: dict[str, float]   # bench_name -> ns (point estimate)


def load_history(history_dir: Path) -> list[Snapshot]:
    snaps: list[Snapshot] = []
    for f in sorted(history_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            snaps.append(Snapshot(
                commit=d["commit"],
                date=d["date"],
                arch=d["arch"],
                benches={k: v["ns"] for k, v in d["benches"].items()},
            ))
        except (KeyError, json.JSONDecodeError) as e:
            print(f"warn: skipping malformed {f.name}: {e}", file=sys.stderr)
    return snaps


def sparkline(values: list[float]) -> str:
    """Lower is faster, so flip: max becomes ▁ (best), min becomes █ (worst)."""
    if not values or all(v == values[0] for v in values):
        return SPARK_CHARS[0] * len(values)
    vmin, vmax = min(values), max(values)
    span = vmax - vmin
    out = []
    for v in values:
        # invert: faster (smaller) = lower block
        norm = (v - vmin) / span
        idx = int(norm * (len(SPARK_CHARS) - 1) + 0.5)
        idx = max(0, min(len(SPARK_CHARS) - 1, idx))
        out.append(SPARK_CHARS[idx])
    return "".join(out)


def format_ns(ns: float) -> str:
    if ns >= 1e6: return f"{ns / 1e6:.2f}ms"
    if ns >= 1e3: return f"{ns / 1e3:.2f}µs"
    return f"{ns:.1f}ns"


def commit_link(sha: str) -> str:
    repo = os.environ.get(GITHUB_REPO_ENV, "")
    short = sha[:7]
    if repo:
        return f"[`{short}`](https://github.com/{repo}/commit/{sha})"
    return f"`{short}`"


def render_arch_section(arch: str, snaps: list[Snapshot]) -> str:
    """Section 1+2 for a single arch."""
    arch_snaps = [s for s in snaps if s.arch == arch]
    arch_snaps.sort(key=lambda s: s.date)
    if not arch_snaps:
        return f"## `{arch}`\n\n_No history yet._\n"

    latest = arch_snaps[-1]
    out = [f"## `{arch}`\n"]
    out.append(f"Last update: **{latest.date}**, commit {commit_link(latest.commit)}\n")

    # Per-bench: best, latest, delta from best, sparkline of last N
    bench_names = sorted({n for s in arch_snaps for n in s.benches})
    out.append("| Bench | Trend (last 20) | Best | Latest | Δ from best |")
    out.append("|---|---|---:|---:|---:|")
    for bn in bench_names:
        series = [s.benches[bn] for s in arch_snaps if bn in s.benches]
        if not series: continue
        best = min(series)
        latest_val = series[-1]
        delta = (latest_val - best) / best * 100.0
        spark = sparkline(series[-HISTORY_DEPTH_SPARK:])
        delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
        out.append(f"| `{bn}` | {spark} | {format_ns(best)} | {format_ns(latest_val)} | {delta_str} |")
    out.append("")

    # Headline regressions/improvements over last N days
    if len(arch_snaps) >= 2:
        worst_reg = None  # (pct, bench, snap, prev_val)
        best_imp = None
        for i in range(1, len(arch_snaps)):
            for bn in arch_snaps[i].benches:
                if bn not in arch_snaps[i - 1].benches: continue
                cur = arch_snaps[i].benches[bn]
                prev = arch_snaps[i - 1].benches[bn]
                if prev == 0: continue
                delta = (cur - prev) / prev * 100.0
                if worst_reg is None or delta > worst_reg[0]:
                    worst_reg = (delta, bn, arch_snaps[i], prev)
                if best_imp is None or delta < best_imp[0]:
                    best_imp = (delta, bn, arch_snaps[i], prev)
        out.append("**Headlines:**")
        if worst_reg and worst_reg[0] > 0:
            d, bn, s, _ = worst_reg
            out.append(f"- Worst step-regression: **+{d:.1f}%** on `{bn}` at commit {commit_link(s.commit)} ({s.date})")
        if best_imp and best_imp[0] < 0:
            d, bn, s, _ = best_imp
            out.append(f"- Best step-improvement: **{d:.1f}%** on `{bn}` at commit {commit_link(s.commit)} ({s.date})")
        out.append("")

    return "\n".join(out)


def render_drilldown(arch: str, snaps: list[Snapshot]) -> str:
    arch_snaps = sorted([s for s in snaps if s.arch == arch], key=lambda s: s.date)
    if not arch_snaps: return ""
    bench_names = sorted({n for s in arch_snaps for n in s.benches})
    out = [f"### `{arch}` per-bench history"]
    for bn in bench_names:
        out.append(f"<details><summary>{bn}</summary>\n")
        out.append("| Commit | Date | ns | Δ vs prev |\n|---|---|---:|---:|")
        history = [(s.commit, s.date, s.benches[bn]) for s in arch_snaps if bn in s.benches]
        history = history[-HISTORY_DEPTH_DRILL:]
        prev = None
        for commit, date, ns in history:
            if prev is None or prev == 0:
                delta = "—"
            else:
                d = (ns - prev) / prev * 100.0
                delta = f"+{d:.2f}%" if d >= 0 else f"{d:.2f}%"
            out.append(f"| {commit_link(commit)} | {date} | {format_ns(ns)} | {delta} |")
            prev = ns
        out.append("\n</details>\n")
    return "\n".join(out)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    history_dir = root / "history"
    if not history_dir.exists():
        print(f"error: {history_dir} does not exist", file=sys.stderr)
        return 1

    snaps = load_history(history_dir)
    if not snaps:
        print(f"warn: no snapshots in {history_dir}", file=sys.stderr)

    archs = sorted({s.arch for s in snaps})

    parts = [
        "# Lattice CPU Perf Baselines",
        "",
        "_Auto-generated by `scripts/perf-baselines-readme.py` from `history/*.json`._",
        "_See [ADR-058](../docs/adr/ADR-058-cpu-perf-regression-ci.md) for the scoring rules._",
        "",
        "Lower numbers are faster. Trend sparkline reads left → right (oldest → newest); "
        "block height encodes time relative to the row's span (`▁` = fastest, `█` = slowest).",
        "",
    ]
    for arch in archs:
        parts.append(render_arch_section(arch, snaps))
    parts.append("---\n")
    for arch in archs:
        parts.append(render_drilldown(arch, snaps))

    out = "\n".join(parts).rstrip() + "\n"
    (root / "README.md").write_text(out)
    print(f"wrote {root / 'README.md'} ({len(out)} bytes, {len(snaps)} snapshots, {len(archs)} archs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
