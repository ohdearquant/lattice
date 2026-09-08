#!/usr/bin/env python3
"""Score a patch-emitting repair adapter against a held-out pair set.

The adapter under test reads a defective front-matter block and emits a unified-diff patch. This
scorer answers three separate questions and never collapses them into one number:

  1. APPLIES   - does the emitted patch apply to the input at all? A patch that does not apply is
                 a failure of the most basic kind, and it is reported before anything else.
  2. CLEAN     - does the repaired document still trip the defect detector? Measured
                 DIFFERENTIALLY: the detector runs on both the prediction and the reference, with
                 the SAME relative path, and only arms present in the prediction but absent from
                 the reference count against it. That way the score cannot be moved by arms that
                 depend on where a file sits, which these extracted pairs cannot answer.
  3. EXACT     - is the repaired block byte-identical to the reference?

CLEAN is the headline: a repair can be correct without being the reference's spelling. EXACT is
reported beside it because a model that reaches CLEAN by deleting the offending lines is not doing
the task, and the two numbers diverging is how that shows up.

CONTROLS RUN IN THE SAME INVOCATION, and the scorer refuses to print a score unless all behave:

  - ORACLE   (predict the reference patch)  must reach 100% on all three. If it does not, the
             scorer itself is broken and no adapter number from this run means anything.
  - IDENTITY (predict an empty patch)       must reach ~0% CLEAN. Every input is defective by
             construction, so an instrument that calls "change nothing" clean cannot detect
             anything. A floor is enforced rather than assumed.
  - VANDAL   (keep only the reference patch's DELETIONS, add nothing) is the degenerate strategy:
             delete whatever the detector complains about. It is not a pass/fail control but a
             FLOOR, and it is here because it is not hypothetical. Measured 2026-09-08 on a 45-row
             held-out split of real (not injected) defects: VANDAL reaches CLEAN on 13 of 45
             (0.289) with EXACT 0.0, and it takes EVERY row of the id-versus-fields arm, 12 of 12,
             because deleting the disagreeing fields removes the disagreement. So CLEAN ALONE IS
             GAMEABLE AT ROUGHLY 30% ON THIS DATA, and on a split of INJECTED defects it was worse
             still: 0.50 CLEAN and 0.30 EXACT, since deleting an injected duplicate key IS the
             reference fix. That is a second, independent reason not to headline injected data. An
             adapter whose CLEAN rate is not clearly above the VANDAL number printed beside it has
             not demonstrated repair, and the scorer says so in `score_notes` rather than
             leaving the reader to notice.

The defect detector is supplied with `--linter`, a Python file exposing
`lint_text(text: str, rel: str) -> list[tuple[arm, detail]]`. It is a parameter and not a constant
so this script carries no dependency on any particular corpus.

Usage:
    uv run python scripts/microlora/score_patch_repair.py \
        --pairs-dir DIR --members DIR/test_real.members.jsonl \
        --linter /path/to/lint_atoms.py [--predictions preds.jsonl] [--json out.json]

`--predictions` is JSONL with {"file", "completion"}; omit it to run the controls alone, which is
the right thing to do before any adapter exists.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pair_jsonl import apply_patch, make_patch, split_pair  # noqa: E402


def load_linter(path: Path):
    spec = importlib.util.spec_from_file_location("_linter_under_test", path)
    if spec is None or spec.loader is None:
        sys.exit(f"cannot load linter from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_linter_under_test"] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "lint_text", None)
    if fn is None:
        sys.exit(f"{path} exposes no lint_text(text, rel); it cannot serve as the detector")
    return fn


def deletions_only(patch: str) -> str:
    """The degenerate repair: keep every hunk's deletions, drop every addition.

    Source consumption is unchanged (only `-` lines advance the cursor in `apply_patch`), so the
    result still applies cleanly; a hunk left with no deletions is dropped entirely.
    """
    out: list[str] = []
    hdr: str | None = None
    body: list[str] = []
    for ln in patch.splitlines():
        if ln.startswith("@@"):
            if hdr is not None and body:
                out.append(hdr)
                out.extend(body)
            hdr, body = ln, []
        elif ln.startswith("-"):
            body.append(ln)
    if hdr is not None and body:
        out.append(hdr)
        out.extend(body)
    return "\n".join(out) + "\n" if out else ""


def score_one(broken_block, fixed_block, body, rel, patch, lint_text):
    """Return (applies, clean, exact, new_arms)."""
    try:
        got = apply_patch(broken_block, patch)
    except ValueError:
        return False, False, False, []
    ref_arms = {a for a, _ in lint_text(fixed_block + body, rel)}
    got_arms = {a for a, _ in lint_text(got + body, rel)}
    new_arms = sorted(got_arms - ref_arms)
    return True, not new_arms, got == fixed_block, new_arms


def run(rows, preds, lint_text):
    n = len(rows)
    applies = clean = exact = 0
    arm_fails: collections.Counter = collections.Counter()
    def _zeros() -> list[int]:
        return [0, 0, 0, 0]

    by_origin: dict[str, list[int]] = collections.defaultdict(_zeros)
    for r in rows:
        patch = preds.get(r["file"])
        if patch is None:
            continue
        a, c, e, new = score_one(r["broken"], r["fixed"], r["body"], r["rel"], patch, lint_text)
        applies += a
        clean += c
        exact += e
        for arm in new:
            arm_fails[arm] += 1
        slot = by_origin[r["origin"]]
        slot[0] += 1
        slot[1] += a
        slot[2] += c
        slot[3] += e
    return {
        "n": n,
        "scored": sum(v[0] for v in by_origin.values()),
        "applies": applies,
        "clean": clean,
        "exact": exact,
        "applies_rate": round(applies / n, 4) if n else None,
        "clean_rate": round(clean / n, 4) if n else None,
        "exact_rate": round(exact / n, 4) if n else None,
        "new_arms": dict(arm_fails),
        "by_origin": {k: {"n": v[0], "applies": v[1], "clean": v[2], "exact": v[3]} for k, v in by_origin.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-dir", required=True, type=Path)
    ap.add_argument("--members", required=True, type=Path, help="the split's .members.jsonl")
    ap.add_argument("--linter", required=True, type=Path, help="python file exposing lint_text(text, rel)")
    ap.add_argument("--predictions", type=Path, help='JSONL of {"file", "completion"}')
    ap.add_argument("--json", type=Path)
    ap.add_argument("--identity-clean-ceiling", type=float, default=0.10,
                    help="the identity control must score at or below this CLEAN rate")
    args = ap.parse_args()

    lint_text = load_linter(args.linter)

    rows = []
    for ln in args.members.read_text().splitlines():
        if not ln.strip():
            continue
        m = json.loads(ln)
        broken_doc = (args.pairs_dir / "broken" / m["file"]).read_text(encoding="utf-8")
        fixed_doc = (args.pairs_dir / "fixed" / m["file"]).read_text(encoding="utf-8")
        bf, ff = split_pair(broken_doc, fixed_doc)
        rows.append({
            "file": m["file"], "origin": m["origin"], "arm": m["arm"],
            "broken": bf, "fixed": ff, "body": fixed_doc[len(ff):],
            # The detector's path-derived arms cannot be answered from an extracted pair, so a
            # stable stand-in is used for BOTH sides; the differential comparison cancels it.
            "rel": m["file"],
        })
    if not rows:
        sys.exit("members file yielded no rows")

    truth = {r["file"]: make_patch(r["broken"], r["fixed"]) for r in rows}
    oracle = run(rows, truth, lint_text)
    identity = run(rows, {r["file"]: "" for r in rows}, lint_text)
    vandal = run(rows, {f: deletions_only(p) for f, p in truth.items()}, lint_text)

    report = {"n": len(rows),
              "controls": {"oracle": oracle, "identity": identity, "vandal": vandal}}

    problems = []
    if not (oracle["applies_rate"] == oracle["clean_rate"] == oracle["exact_rate"] == 1.0):
        problems.append(f"ORACLE control did not reach 1.0 on all three: {oracle['applies_rate']}/"
                        f"{oracle['clean_rate']}/{oracle['exact_rate']} — the scorer is broken, "
                        f"no adapter number from this run is usable")
    if identity["clean_rate"] > args.identity_clean_ceiling:
        problems.append(f"IDENTITY control scored CLEAN {identity['clean_rate']} > ceiling "
                        f"{args.identity_clean_ceiling} — the detector cannot see these defects, "
                        f"so a high adapter CLEAN rate would mean nothing")

    if args.predictions:
        preds = {}
        for ln in args.predictions.read_text().splitlines():
            if ln.strip():
                p = json.loads(ln)
                preds[p["file"]] = p["completion"]
        report["adapter"] = run(rows, preds, lint_text)
        missing = len(rows) - report["adapter"]["scored"]
        if missing:
            report["adapter"]["missing_predictions"] = missing
        notes = []
        if report["adapter"]["clean_rate"] is not None:
            margin = report["adapter"]["clean_rate"] - vandal["clean_rate"]
            notes.append(f"CLEAN {report['adapter']['clean_rate']} vs VANDAL floor "
                         f"{vandal['clean_rate']} (margin {margin:+.4f})")
            if margin <= 0:
                notes.append("AT OR BELOW the delete-everything floor: this is not evidence of repair")
        if report["adapter"]["exact_rate"] is not None and report["adapter"]["clean_rate"]:
            notes.append(f"EXACT {report['adapter']['exact_rate']} — a CLEAN rate far above EXACT "
                         f"means repairs that satisfy the detector without matching the reference; "
                         f"read the two together, never CLEAN alone")
        report["score_notes"] = notes

    report["controls_ok"] = not problems
    report["problems"] = problems
    text = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    if problems:
        print("\n".join("REFUSED: " + p for p in problems), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
