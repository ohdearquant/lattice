#!/usr/bin/env python3
"""Convert Criterion's per-bench output into a single history snapshot JSON.

Reads target/criterion/<bench>/<test>/new/estimates.json for every bench,
emits a flat snapshot suitable for perf-baselines/history/.

Usage:
  perf-bench-snapshot.py <criterion_root> <arch> <commit_sha> [--out path]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("criterion_root", type=Path)
    ap.add_argument("arch", help="e.g. aarch64-linux")
    ap.add_argument("commit", help="git SHA")
    ap.add_argument("--out", type=Path, help="output path (default: stdout)")
    args = ap.parse_args()

    if not args.criterion_root.exists():
        print(f"error: {args.criterion_root} missing", file=sys.stderr)
        return 1

    benches: dict[str, dict] = {}
    for est_file in sorted(args.criterion_root.rglob("new/estimates.json")):
        bench_dir = est_file.parent.parent
        rel = bench_dir.relative_to(args.criterion_root)
        name = str(rel)
        try:
            est = json.loads(est_file.read_text())
            mean = est["mean"]
            ci = mean["confidence_interval"]
            benches[name] = {
                "ns": mean["point_estimate"],
                "ns_ci_low": ci["lower_bound"],
                "ns_ci_high": ci["upper_bound"],
            }
        except (KeyError, json.JSONDecodeError) as e:
            print(f"warn: skipping {name}: {e}", file=sys.stderr)

    snap = {
        "commit": args.commit,
        "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "arch": args.arch,
        "benches": benches,
    }
    text = json.dumps(snap, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out} ({len(benches)} benches)")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
