#!/usr/bin/env python3
"""Summarize one arm's in-phase load samples and decide certifiability (lattice#1515).

    phase-load-report.py --arm head1 --in .../phase-load/head1.jsonl --floor 70

Reads the perf-phase-sample/v1 JSONL that phase-load-sampler.py wrote during
one measured arm and prints a single summary line: sample count, machine idle
min/mean, foreign-load max/mean, self-load mean, and the top foreign process
at the foreign-max sample.

Exit codes:
  0 - ok, foreign load stayed at or below the ceiling (100 - floor) throughout
  1 - LOUD, some sample's foreign load exceeded the ceiling
  2 - the sampler produced no usable samples (dead instrument) -- this is
      never certifiable, in any mode: a sampler that measured nothing is not
      weaker evidence of a quiet arm, it is no evidence at all.

The caller (bench-compare-impl.sh) decides what exit 1 means for the current
run mode: refuse outright, or (under PERF_POSTMERGE_STATUS_DIR) record and
continue, mirroring quiet_gate's treatment of a below-floor boundary sample.
Exit 2 is never downgraded by the caller.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--in", dest="input", required=True)
    ap.add_argument("--floor", type=float, default=70.0)
    args = ap.parse_args()

    path = Path(args.input)
    records = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print(
            f"[phase-load] {args.arm}: PROBE FAILED (zero usable samples) - "
            "refusing to measure"
        )
        return 2

    idle = [r["idle_pct"] for r in records]
    foreign = [r["foreign_pct"] for r in records]
    self_load = [r["self_pct"] for r in records]
    top_record = max(records, key=lambda r: r["foreign_pct"])

    ceiling = 100.0 - args.floor
    loud = max(foreign) > ceiling
    verdict = "LOUD" if loud else "ok"
    print(
        f"[phase-load] {args.arm}: samples={len(records)} "
        f"idle min/mean={min(idle):.1f}%/{sum(idle) / len(idle):.1f}% "
        f"foreign max/mean={max(foreign):.1f}%/{sum(foreign) / len(foreign):.1f}% "
        f"self mean={sum(self_load) / len(self_load):.1f}% "
        f"(ceiling {ceiling:.1f}%) {verdict} | "
        f"top foreign at max: {top_record['top_foreign']} "
        f"{top_record['top_foreign_pct']:.1f}% @ {top_record['captured_utc']} "
        f"| per-process=ps-pcpu-decaying-avg"
    )
    return 1 if loud else 0


if __name__ == "__main__":
    sys.exit(main())
