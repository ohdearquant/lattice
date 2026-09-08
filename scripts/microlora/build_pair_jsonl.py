#!/usr/bin/env python3
"""Turn a broken/fixed document-pair set into trainer JSONL for a micro-LoRA repair adapter.

Input layout (produced by an external generator; the data itself never lives in this repo):

    <pairs-dir>/broken/<name>     the defective document
    <pairs-dir>/fixed/<name>      the repaired document
    <pairs-dir>/manifest.jsonl    one row per pair: {"file", "arm", "origin": "real"|"synthetic", ...}

Output: {"prompt", "completion"} rows in the shape `crates/tune/src/train_support.rs::load_jsonl`
reads. The completion is a PATCH (unified-diff hunks over the front-matter block, no file headers),
not the whole repaired block. Measured on a 501-pair set with the Qwen3.5 tokenizer: a whole-block
target keeps 93/251 real pairs under 512 tokens and 134/251 under 768; the patch target keeps
154/251 and 225/251. Every emitted patch is applied back to the broken block in this script and must
reproduce the fixed block byte for byte, so the target format is proven lossless before any row
is written, and a row whose patch does not round-trip is refused rather than emitted.

Splits are stratified by (origin, arm) with a fixed seed. Real and synthetic rows are written to
SEPARATE held-out files: a blended accuracy over both would be dominated by whichever origin is
larger, and accuracy on synthetic rows measures the generator's injection fingerprint, not repair.

Usage:
    uv run python scripts/microlora/build_pair_jsonl.py --pairs-dir DIR --out DIR \
        [--seed 20260908] [--seq-budget 768] [--synthetic-holdout-per-arm 5]
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import random
import sys
from pathlib import Path

INSTRUCTION = (
    "Repair the YAML front matter so the id, its derived fields, and the required keys agree. "
    "Answer with a unified-diff patch over the front matter only.\n\n"
)


def front_matter(text: str) -> str:
    """The leading `---` block including its closing fence, or the whole text if there is none."""
    if text.startswith("---"):
        j = text.find("\n---", 3)
        if j > 0:
            return text[: j + 4].rstrip("\n")
    return text.rstrip("\n")


def split_pair(broken: str, fixed: str) -> tuple[str, str]:
    """Front-matter blocks of both sides.

    The fixed side always has a well-formed block. The broken side may not (a missing or
    malformed closing fence is one of the defect arms), in which case `front_matter` would
    return the whole document and the patch would carry the entire body as deletions. The body
    is identical on both sides by construction, so the broken block is cut where the fixed
    side's body begins; that cut is derived from shared text, never from the answer.
    """
    ff = front_matter(fixed)
    body = fixed[len(ff):]
    if body.strip() and broken.endswith(body):
        bf = broken[: len(broken) - len(body)].rstrip("\n")
    else:
        bf = front_matter(broken)
    return bf, ff


def make_patch(before: str, after: str) -> str:
    a, b = before.splitlines(), after.splitlines()
    lines = [
        ln
        for ln in difflib.unified_diff(a, b, lineterm="", n=0)
        if not ln.startswith(("---", "+++"))
    ]
    return "\n".join(lines) + "\n"


def apply_patch(before: str, patch: str) -> str:
    """Apply a zero-context unified diff produced by make_patch. Raises on any mismatch."""
    src = before.splitlines()
    out: list[str] = []
    pos = 0  # index into src of the next unconsumed line
    lines = patch.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.startswith("@@"):
            raise ValueError(f"expected hunk header, got {ln!r}")
        # @@ -a[,n] +b[,m] @@
        old = ln.split()[1]
        old_start = int(old[1:].split(",")[0])
        old_len = int(old[1:].split(",")[1]) if "," in old else 1
        # a zero-length old range is written as the line BEFORE the insertion point
        anchor = old_start - 1 if old_len else old_start
        if anchor < pos:
            raise ValueError("hunks out of order")
        out.extend(src[pos:anchor])
        pos = anchor
        i += 1
        while i < len(lines) and not lines[i].startswith("@@"):
            body = lines[i]
            if body.startswith("-"):
                if pos >= len(src) or src[pos] != body[1:]:
                    raise ValueError(f"context mismatch at source line {pos + 1}")
                pos += 1
            elif body.startswith("+"):
                out.append(body[1:])
            else:
                raise ValueError(f"unexpected patch line {body!r}")
            i += 1
    out.extend(src[pos:])
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--seq-budget", type=int, default=768, help="rows whose prompt+completion exceed this many tokens are dropped and counted; needs the tokenizer")
    ap.add_argument("--tokenizer", type=Path, default=Path.home() / ".lattice/models/qwen3.5-0.8b/tokenizer.json")
    ap.add_argument("--synthetic-holdout-per-arm", type=int, default=5)
    ap.add_argument("--real-valid-frac", type=float, default=0.15)
    ap.add_argument("--real-test-frac", type=float, default=0.15)
    args = ap.parse_args()

    manifest = [json.loads(ln) for ln in (args.pairs_dir / "manifest.jsonl").read_text().splitlines() if ln.strip()]
    if not manifest:
        sys.exit("manifest.jsonl is empty")

    try:
        from tokenizers import Tokenizer  # type: ignore

        tok = Tokenizer.from_file(str(args.tokenizer))
    except Exception as e:  # noqa: BLE001
        sys.exit(f"tokenizer unavailable ({e!r}); the seq budget cannot be enforced, refusing to emit")

    rows: list[dict] = []
    refused = collections.Counter()
    for r in manifest:
        broken = (args.pairs_dir / "broken" / r["file"]).read_text(encoding="utf-8", errors="strict")
        fixed = (args.pairs_dir / "fixed" / r["file"]).read_text(encoding="utf-8", errors="strict")
        bf, ff = split_pair(broken, fixed)
        if bf == ff:
            refused["identical_front_matter"] += 1
            continue
        patch = make_patch(bf, ff)
        try:
            rt = apply_patch(bf, patch)
        except ValueError as e:
            refused[f"patch_apply_error:{e}"] += 1
            continue
        if rt != ff:
            refused["patch_not_lossless"] += 1
            continue
        prompt = INSTRUCTION + bf
        n = len(tok.encode(prompt).ids) + len(tok.encode(patch).ids)
        if n > args.seq_budget:
            refused["over_seq_budget"] += 1
            continue
        rows.append({"prompt": prompt, "completion": patch, "arm": r["arm"], "origin": r["origin"], "file": r["file"], "tokens": n})

    # Stratified split.
    rng = random.Random(args.seed)
    by_key: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_key[(row["origin"], row["arm"])].append(row)
    split: dict[str, list[dict]] = collections.defaultdict(list)
    for (origin, _arm), group in sorted(by_key.items()):
        rng.shuffle(group)
        if origin == "real":
            n_test = max(1, round(len(group) * args.real_test_frac))
            n_valid = max(1, round(len(group) * args.real_valid_frac))
            split["test_real"] += group[:n_test]
            split["valid_real"] += group[n_test : n_test + n_valid]
            split["train"] += group[n_test + n_valid :]
        else:
            k = min(args.synthetic_holdout_per_arm, len(group))
            split["test_synthetic"] += group[:k]
            split["train"] += group[k:]

    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"seed": args.seed, "seq_budget": args.seq_budget, "refused": dict(refused), "splits": {}}
    for name, group in split.items():
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in group:
                fh.write(json.dumps({"prompt": row["prompt"], "completion": row["completion"]}, ensure_ascii=False) + "\n")
        with (args.out / f"{name}.members.jsonl").open("w", encoding="utf-8") as fh:
            for row in group:
                fh.write(json.dumps({k: row[k] for k in ("file", "arm", "origin", "tokens")}) + "\n")
        summary["splits"][name] = {
            "rows": len(group),
            "by_arm": dict(collections.Counter(r["arm"] for r in group)),
            "by_origin": dict(collections.Counter(r["origin"] for r in group)),
        }
    (args.out / "SPLIT_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
