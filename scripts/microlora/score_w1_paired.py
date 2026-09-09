#!/usr/bin/env python3
"""Score W1 by the PAIRED metric (M2): held-out NLL of the real arm minus the permuted arm.

This rule was fixed BEFORE its first number existed: it was written while the permuted arm's log
still held zero held-out lines, and that was checked immediately before and after writing it. The
ordering is the whole point of the rule, so it is recorded here rather than left to memory.

WHY THE DIFFERENCE AND NOT THE DROP. On a patch-emitting task the absolute held-out drop is
contaminated by format acquisition: a model that learns only the diff grammar improves on the real
held-out split no matter which completion it trained against. The permuted arm sees identical
prompts and identical patch syntax and differs ONLY in which completion is paired with which prompt,
so everything learnable from format is available to both arms and cancels in the difference. What
survives is whether the completion was predictable FROM ITS OWN PROMPT. The same cancellation covers
target-string memorisation, because permuting labels preserves the completion multiset.

THE CONFOUND THIS CANNOT REMOVE, printed beside every result rather than footnoted. A large negative
difference arises two ways: the real arm LEARNING, or the permuted arm DEGRADING as it fits noise.
The difference alone cannot separate them, so each arm's absolute change from the shared base is
printed next to it, and if the permuted arm ends above its own base the verdict line says the
difference is partly damage avoided rather than skill gained.

THE BAR. Per-seed differences must be negative in EVERY seed. With three seeds that is a sign test
at p = 0.125 one-sided, which is a pilot's bar, not a publication's, and the output says so. A
single seed is not a test at all and is reported as a direction only.

Exit: 0 PASS, 2 REFUSED (missing arm, unreadable log, or mismatched baselines), 3 FAIL.

Usage:
    uv run python scripts/microlora/score_w1_paired.py --real R1.log [R2.log ...] --permuted P1.log [...]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

STEP = re.compile(r"step\s+(\d+)\s+train NLL:\s*([0-9.]+)\s+held-out NLL:\s*([0-9.]+)")


def read(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "ok": False, "why": "file does not exist"}
    steps = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STEP.search(line)
        if m:
            steps[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    if len(steps) < 2:
        return {"path": str(path), "ok": False,
                "why": f"found {len(steps)} step line(s); a trajectory needs at least a base and a final"}
    return {"path": str(path), "ok": True, "steps": steps,
            "base": steps[min(steps)][1], "final": steps[max(steps)][1],
            "first_step": min(steps), "last_step": max(steps)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", nargs="+", required=True, type=Path)
    ap.add_argument("--permuted", nargs="+", required=True, type=Path)
    args = ap.parse_args()

    real = [read(p) for p in args.real]
    perm = [read(p) for p in args.permuted]
    broken = [d for d in real + perm if not d["ok"]]
    for d in broken:
        print(f"UNUSABLE  {Path(d['path']).name}: {d['why']}")
    if broken:
        print(f"\nREFUSED: {len(broken)} log(s) unreadable. A paired metric computed over the readable "
              f"subset would silently pair arms that are not partners.", file=sys.stderr)
        return 2
    if len(real) != len(perm):
        print(f"\nREFUSED: {len(real)} real arm(s) against {len(perm)} permuted arm(s). The metric is "
              f"PAIRED; unequal counts mean the pairing is being invented here rather than measured.",
              file=sys.stderr)
        return 2

    print(f"{'seed#':5} {'base':>8} {'real fin':>9} {'perm fin':>9} {'real d':>8} {'perm d':>8} {'REAL-PERM':>10}")
    diffs, notes = [], []
    for i, (r, p) in enumerate(zip(real, perm), 1):
        if abs(r["base"] - p["base"]) > 1e-4:
            print(f"\nREFUSED: pair {i} does not share a baseline (real {r['base']:.4f} vs permuted "
                  f"{p['base']:.4f}). The arms must score the same untouched held-out split with the "
                  f"same initial model, or the difference is not a difference.", file=sys.stderr)
            return 2
        if r["last_step"] != p["last_step"]:
            print(f"\nREFUSED: pair {i} compares step {r['last_step']} against {p['last_step']}. The "
                  f"metric is defined at a MATCHED step index.", file=sys.stderr)
            return 2
        d = r["final"] - p["final"]
        diffs.append(d)
        rd = (r["final"] - r["base"]) / r["base"]
        pd = (p["final"] - p["base"]) / p["base"]
        print(f"{i:5} {r['base']:8.4f} {r['final']:9.4f} {p['final']:9.4f} {rd:7.1%} {pd:7.1%} {d:+10.4f}")
        if p["final"] > p["base"]:
            notes.append(f"pair {i}: the permuted arm ended ABOVE its base ({p['final']:.4f} > "
                         f"{p['base']:.4f}), so its share of the difference is damage, not absence of skill")

    n = len(diffs)
    neg = sum(1 for d in diffs if d < 0)
    print(f"\nper-seed differences: {['%+.4f' % d for d in diffs]}")
    print(f"negative in {neg} of {n}")
    for s in notes:
        print(f"NOTE: {s}")

    if neg == n and n >= 3:
        print(f"\nVERDICT: PASS\n  the real arm is better in every one of {n} seeds. Sign test p = "
              f"{0.5 ** n:.3f} one-sided. This is a pilot's bar: enough to justify more spend, not "
              f"enough to publish.")
        return 0
    if n < 3:
        print(f"\nVERDICT: DIRECTION ONLY\n  {n} pair(s) is not a test. The real arm is "
              f"{'better' if neg == n else 'not uniformly better'}; the registered bar needs three "
              f"seeds negative and this cannot meet it either way.")
        return 0 if neg == n else 3
    print(f"\nVERDICT: FAIL\n  the real arm is better in only {neg} of {n} seeds. Per the registered "
          f"kill point this is not a call for a third metric: it says the task is not being learned "
          f"from content at this scale and step budget, and the task or the budget changes, not the "
          f"instrument.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
