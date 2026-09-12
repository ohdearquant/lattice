#!/usr/bin/env python3
"""Check a train/held-out JSONL pair for the leakage that would invalidate a held-out NLL claim.

It reports three distinct things and refuses to collapse them, because they have different
consequences and only the first is a broken split by default:

  PROMPT OVERLAP      a held-out prompt that also appears in train. This is real leakage: the model
                      was trained on the exact input it is being scored on, and any held-out number
                      is void.
  COMPLETION OVERLAP  a held-out TARGET string that also appears as a training target, under a
                      DIFFERENT prompt. This is not a broken split. On repair-style tasks the same
                      fix legitimately applies to many documents, so identical targets are expected.
                      It matters for a different reason: for those rows a model that memorised the
                      output string can score well without performing the mapping, which inflates an
                      absolute NLL-drop metric. It does NOT bias a paired real-versus-permuted
                      comparison, because a label permutation preserves the completion multiset, so
                      both arms get the same memorisation opportunity and it cancels.
  LENGTH IMBALANCE    the two splits' prompt-length distributions. A held-out split much shorter or
                      longer than train is not exchangeable with it, so the held-out number answers
                      a slightly different question than "how well does this generalise".

With --verb-partition, also report distinct completion calls and their crossing set.
Double-quoted string literals are blanked before extracting pack.verb calls and bare kg verbs.
Any crossing breaks the whole-verb partition; train against itself must cross on every verb.
This lexical check does not inspect operations embedded inside string literals.

Every count is printed with its denominator, and a positive control (train against itself) runs in
the same invocation so an empty overlap cannot be a broken comparison silently reading as clean.

Exit: 0 clean, 2 unreadable input/control failure, 3 PROMPT overlap,
4 VERB crossing with --verb-partition (takes precedence over prompt overlap).

Usage:
    uv run python scripts/microlora/check_split_integrity.py --dir DIR [--train train.jsonl] [--valid valid.jsonl] [--verb-partition]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path


def load(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def completion_verbs(rows: list[dict]) -> set[str]:
    verbs = set()
    for row in rows:
        blanked = re.sub(r'"(?:\\.|[^"\\])*"', '""', row["completion"])
        verbs.update(re.findall(r"\b([a-z_]+\.[a-z_]+)\s*\(", blanked))
        verbs.update(re.findall(r"(?<![\w.])([a-z_]+)\s*\(", blanked))
    return verbs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--valid", default="valid.jsonl")
    ap.add_argument("--verb-partition", action="store_true", help="fail on completion verbs shared by both splits")
    args = ap.parse_args()

    tp, vp = args.dir / args.train, args.dir / args.valid
    for p in (tp, vp):
        if not p.exists():
            print(f"REFUSED: {p} does not exist; an absence check over a missing file is not a clean result",
                  file=sys.stderr)
            return 2
    tr, va = load(tp), load(vp)
    if not tr or not va:
        print(f"REFUSED: empty split ({len(tr)} train, {len(va)} valid)", file=sys.stderr)
        return 2

    tr_prompts = {h(r["prompt"]) for r in tr}
    tr_comps = collections.Counter(h(r["completion"]) for r in tr)

    # POSITIVE CONTROL, same invocation: train's own prompts must all be found in train's prompt set.
    control = sum(1 for r in tr if h(r["prompt"]) in tr_prompts)
    if control != len(tr):
        print(f"REFUSED: the control failed ({control}/{len(tr)}); the comparison itself is broken, "
              f"so any zero overlap below would be meaningless", file=sys.stderr)
        return 2

    p_over = [r for r in va if h(r["prompt"]) in tr_prompts]
    c_over = [r for r in va if h(r["completion"]) in tr_comps]

    print(f"train {len(tr)} rows ({len(tr_comps)} distinct completions, "
          f"{sum(1 for c in tr_comps.values() if c == 1)} appearing once)")
    print(f"valid {len(va)} rows")
    print(f"  control (train prompts found in train): {control}/{len(tr)}  [must be all]")
    print(f"  PROMPT overlap:     {len(p_over)}/{len(va)}")
    print(f"  COMPLETION overlap: {len(c_over)}/{len(va)}"
          f"  ({len({h(r['completion']) for r in c_over})} distinct strings)")
    for r in sorted({h(x["completion"]): x for x in c_over}.values(), key=lambda x: -len(x["completion"])):
        first = next((ln for ln in r["completion"].splitlines() if ln.strip()), "(blank)")
        print(f"      x{tr_comps[h(r['completion'])]:>3} in train | {len(r['completion']):>5}B | {first[:58]}")

    lt = [len(r["prompt"]) for r in tr]
    lv = [len(r["prompt"]) for r in va]
    print(f"  prompt length median: train {statistics.median(lt):.0f}B, valid {statistics.median(lv):.0f}B "
          f"(ratio {statistics.median(lv)/statistics.median(lt):.2f})")

    if args.verb_partition:
        train_verbs, valid_verbs = completion_verbs(tr), completion_verbs(va)
        verb_control = train_verbs & completion_verbs(tr)
        crossing = train_verbs & valid_verbs
        print(f"  VERB count: train {len(train_verbs)} distinct/{len(tr)} rows, "
              f"valid {len(valid_verbs)} distinct/{len(va)} rows")
        print(f"  control (train verbs found in train): {len(verb_control)}/{len(train_verbs)}  [must be all]")
        print(f"  VERB crossing: {len(crossing)}/{len(train_verbs)} train, "
              f"{len(crossing)}/{len(valid_verbs)} valid: {', '.join(sorted(crossing)) or 'none'}")
        if verb_control != train_verbs:
            print("REFUSED: the verb control failed; the comparison itself is broken", file=sys.stderr)
            return 2
        if crossing:
            print("FAIL: completion verbs appear in both splits: " + ", ".join(sorted(crossing)), file=sys.stderr)
            return 4

    print()
    if p_over:
        print(f"FAIL: {len(p_over)} held-out prompt(s) appear in train. Held-out numbers from this "
              f"split are void, not merely optimistic.", file=sys.stderr)
        return 3
    print("PASS on the condition that voids a run: no held-out prompt appears in train.")
    if c_over:
        print(f"NOTE, not a failure: {len(c_over)}/{len(va)} held-out rows have a target string that also "
              f"appears in train under a different prompt. Expected on repair tasks where one fix serves "
              f"many documents. It inflates an ABSOLUTE NLL-drop metric for those rows and cancels in a "
              f"paired real-vs-permuted comparison, since permuting labels preserves the completion multiset.")
    if not 0.75 <= statistics.median(lv) / statistics.median(lt) <= 1.33:
        print(f"NOTE: the splits' prompt lengths differ by more than a third at the median, so held-out is "
              f"not exchangeable with train. With n={len(va)} this may be sampling noise; it is reported as "
              f"an observed imbalance, not a demonstrated bias.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
