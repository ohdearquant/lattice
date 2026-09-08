#!/usr/bin/env python3
"""Apply the pre-registered W1 decision rule to a set of trainer logs.

The rule is written here so it is applied mechanically to whatever the logs say, rather than
narrated after the numbers are read. It encodes exactly what the definition and the
pre-registration already fixed, and it has no way to express "close enough":

  PASS      the real arm's held-out NLL drop is >= --pass-threshold (default 0.15 relative)
            AND the permuted arm does not reach it.
  KILL      the permuted arm reaches --pass-threshold. The metric cannot separate learning the
            TASK from learning the output FORMAT on this data, and no threshold is rewritten to
            rescue a pass; a format-controlled metric gets designed instead.
  FAIL      the real arm does not reach the threshold.

The MARGIN between the arms is printed beside the verdict because it is the quantity that survives
even when both arms move: a permuted arm that drops 25% while the real arm drops 41% still leaves
16 points that permuted labels cannot explain. The margin is reported, never silently substituted
for the registered rule.

Usage:
    uv run python scripts/microlora/score_w1.py --real A.log --permuted B.log
    uv run python scripts/microlora/score_w1.py --summary OUT/SUMMARY.tsv
"""

from __future__ import annotations

import argparse
import re
import statistics as st
import sys
from pathlib import Path


def read_log(path: Path) -> dict:
    """Pull the held-out NLL trajectory out of one trainer log."""
    tr, ho, steps = [], [], []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = re.search(r"step\s+(\d+)\s+train NLL:\s*([0-9.]+)\s+held-out NLL:\s*([0-9.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            tr.append(float(m.group(2)))
            ho.append(float(m.group(3)))
    if len(ho) < 2:
        return {"path": str(path), "ok": False,
                "why": f"found {len(ho)} held-out NLL points; a drop needs a baseline and a final"}
    return {"path": str(path), "ok": True, "steps": steps,
            "train_base": tr[0], "train_final": tr[-1],
            "ho_base": ho[0], "ho_final": ho[-1],
            "ho_drop": (ho[0] - ho[-1]) / ho[0]}


def verdict(real: list[dict], perm: list[dict], thr: float) -> tuple[str, str]:
    if not real:
        return "REFUSED", "no real-label arm was supplied; there is nothing to decide"
    if not perm:
        return "REFUSED", ("no permuted-label arm was supplied. The control is not optional: on "
                          "this task the base model cannot emit the output format at all, so a "
                          "real-arm drop alone cannot be read")
    r = st.mean(d["ho_drop"] for d in real)
    p = st.mean(d["ho_drop"] for d in perm)
    if p >= thr:
        return "KILL", (f"the permuted arm reached {p:.1%}, at or above the {thr:.0%} bar. Training "
                        f"on scrambled labels produces a passing score, so this metric is measuring "
                        f"format acquisition and cannot decide W1 on this task")
    if r < thr:
        return "FAIL", f"the real arm reached {r:.1%}, below the {thr:.0%} bar"
    return "PASS", (f"real {r:.1%} clears {thr:.0%} while permuted {p:.1%} does not; "
                    f"margin {r - p:+.1%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", type=Path, action="append", default=[])
    ap.add_argument("--permuted", type=Path, action="append", default=[])
    ap.add_argument("--summary", type=Path, help="a run's SUMMARY.tsv; arms are read from its arm column")
    ap.add_argument("--pass-threshold", type=float, default=0.15)
    args = ap.parse_args()

    real_paths, perm_paths = list(args.real), list(args.permuted)
    if args.summary:
        for ln in args.summary.read_text().splitlines()[1:]:
            if not ln.strip():
                continue
            cells = ln.split("\t")
            arm, log = cells[1], Path(cells[-1])
            (real_paths if arm == "real" else perm_paths).append(log)

    real, perm, broken = [], [], []
    for paths, bucket in ((real_paths, real), (perm_paths, perm)):
        for p in paths:
            d = read_log(p)
            (bucket if d["ok"] else broken).append(d)

    print(f"{'arm':9} {'log':44} {'ho_base':>8} {'ho_final':>9} {'drop':>8}")
    for label, bucket in (("real", real), ("permuted", perm)):
        for d in bucket:
            print(f"{label:9} {Path(d['path']).name[:43]:44} {d['ho_base']:8.4f} "
                  f"{d['ho_final']:9.4f} {d['ho_drop']:7.1%}")
    for d in broken:
        print(f"{'UNUSABLE':9} {Path(d['path']).name[:43]:44} {d['why']}")

    if broken:
        print(f"\nREFUSED: {len(broken)} log(s) could not be read. A verdict computed over the "
              f"readable subset would silently answer a different question than the one asked.",
              file=sys.stderr)
        return 2

    v, why = verdict(real, perm, args.pass_threshold)
    print(f"\nVERDICT: {v}\n  {why}")
    print(f"  n = {len(real)} real arm(s), {len(perm)} permuted arm(s), threshold {args.pass_threshold:.0%}")
    # The exit code carries the VERDICT, not merely "the scorer ran". A verdict-bearing tool whose
    # rc is constant is a trap: a caller that checks rc reads KILL as success, which is the exact
    # reading this whole experiment exists to prevent. The codes are distinct so that no caller can
    # collapse them by accident, and they are documented here rather than only in a commit message.
    #   0 PASS     the real arm clears the bar and the permuted arm does not
    #   2 REFUSED  a required arm is missing or a log was unreadable; no verdict was computed
    #   3 FAIL     the real arm did not clear the bar
    #   4 KILL     the permuted arm also cleared it, so the metric is uninformative for this task
    return {"PASS": 0, "REFUSED": 2, "FAIL": 3, "KILL": 4}[v]


if __name__ == "__main__":
    sys.exit(main())
