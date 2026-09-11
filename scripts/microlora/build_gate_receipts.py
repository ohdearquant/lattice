#!/usr/bin/env python3
"""Build a (gate log -> structured receipt) dataset from real gate logs.

The logs themselves are local build/CI artifacts and never enter the repo; this script
is the reproducible path from them to the dataset, and it records what it read.

WHY THE DISTRIBUTION IS REPORTED AND NOT JUST THE COUNT. The receipt's `format` field is
derived, so a corpus of logs that are not cargo output yields rows whose answer is
`unknown`. That is the CORRECT answer, and it is also a training hazard: a set dominated
by one answer teaches that answer. The split is printed so the skew is visible before
anyone trains on it, rather than discovered afterwards as a model that always abstains.

TRUNCATION IS DECLARED. Files above --max-bytes are read to the cap and the row records
`truncated: true`, because a receipt derived from a partial log can miss a later
`test result:` line, and a row that hides that is indistinguishable from a complete one.

Usage:
    uv run python scripts/microlora/build_gate_receipts.py --logs-root DIR --out DIR \
        [--max-bytes 2000000] [--self-test]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gate_log_receipt as G  # noqa: E402

EXTS = {".log", ".out", ".txt", ".nohup"}


def rows_for(root: pathlib.Path, max_bytes: int):
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix not in EXTS:
            continue
        try:
            raw = f.read_bytes()
        except OSError:
            continue
        truncated = len(raw) > max_bytes
        text = raw[:max_bytes].decode("utf-8", errors="replace")
        receipt = G.parse_cargo_test_log(text)
        receipt["log"] = G.value({"path": str(f), "bytes": len(raw),
                                  "sha256": G.hashlib.sha256(raw).hexdigest()[:16],
                                  "truncated": truncated})
        receipt["exit_code"] = G.rc_from_log(text)
        receipt["command"] = G.unknown("a gate log does not record the invocation that produced it; pass --command")
        receipt["source_hash"] = G.unknown("a gate log does not record the ref it was produced from; pass --source-hash")
        receipt["executed_generated_command"] = G.value(False)
        yield f, receipt, truncated


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-root", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--max-bytes", type=int, default=2_000_000)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if not a.logs_root or not a.out:
        ap.error("--logs-root and --out are required")

    a.out.mkdir(parents=True, exist_ok=True)
    dist = collections.Counter()
    malformed, truncs, n = [], 0, 0
    with (a.out / "receipts.jsonl").open("w", encoding="utf-8") as fh:
        for f, receipt, truncated in rows_for(a.logs_root, a.max_bytes):
            bad = [k for k, v in receipt.items() if not (("value" in v) ^ ("unknown" in v))]
            if bad:
                malformed.append((str(f), bad))
            dist["cargo" if G.is_known(receipt["format"]) else "unknown-format"] += 1
            if G.is_known(receipt["outcome"]):
                dist[f"outcome={receipt['outcome']['value']}"] += 1
            truncs += int(truncated)
            n += 1
            fh.write(json.dumps({"log_path": str(f), "receipt": receipt}, sort_keys=True) + "\n")

    summary = {"rows": n, "distribution": dict(dist), "truncated_rows": truncs,
               "malformed_rows": len(malformed), "malformed": malformed[:20],
               "max_bytes": a.max_bytes, "logs_root": str(a.logs_root)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    # A malformed row means a field was neither value nor unknown, which is the one
    # invariant this dataset exists to uphold. Refuse rather than ship it.
    return 3 if malformed else 0


def _self_test() -> int:
    import tempfile
    cases = []
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        (root / "a.log").write_text("   Compiling x v0.1.0\ntest result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.0s\n")
        (root / "b.log").write_text("watching CI, ALL GREEN\n")
        (root / "c.log").write_text("")
        (root / "skip.md").write_text("test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.0s\n")
        got = {f.name: (r, t) for f, r, t in rows_for(root, 1_000_000)}
        cases.append(("only log-shaped extensions are read; a .md quoting a result is NOT a gate log",
                      set(got) == {"a.log", "b.log", "c.log"}))
        cases.append(("a real cargo log reports format", got["a.log"][0]["format"]["value"] == "cargo-test"))
        cases.append(("a CI-watcher log does NOT claim cargo format", not G.is_known(got["b.log"][0]["format"])))
        cases.append(("an empty log does NOT claim cargo format", not G.is_known(got["c.log"][0]["format"])))
        cases.append(("every field is two-shaped on every row",
                      all(("value" in v) ^ ("unknown" in v) for _, (r, _) in got.items() for v in r.values())))
        big = root / "big.log"
        big.write_text("x" * 50 + "\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.0s\n")
        small = {f.name: (r, t) for f, r, t in rows_for(root, 10)}
        cases.append(("truncation is DECLARED on the row, not silent",
                      small["big.log"][1] is True and small["big.log"][0]["log"]["value"]["truncated"] is True))
        cases.append(("a truncated read does not invent counts",
                      not G.is_known(small["big.log"][0]["passed"])))
    bad = [n for n, ok in cases if not ok]
    for n, ok in cases:
        print(f"  {'ok  ' if ok else 'FAIL'}  {n}")
    if bad:
        print(f"SELF-TEST FAILED: {len(bad)} case(s)", file=sys.stderr)
        return 3
    print(f"self-test OK  {len(cases)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
