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

# A log-shaped EXTENSION is not a log, and these two exclusions were found by looking at
# rows whose content spanned the train/held-out boundary rather than by inspecting the
# filter.
#
# `fixtures/` -- ten rows, six of them classified `cargo`, and they are THIS EXTRACTOR'S OWN
# test fixtures (`ci-log-ansi-zero.txt`, `real-arm-a-healthy.txt`). Training on the fixtures
# that define the correct answer is self-contamination, and it is invisible in any summary
# that reports only class counts. Also caught here: `merges.txt`, a checked-out tokenizer
# vocabulary sitting in a workspace, which is not captured output at all.
#
# MIN_BYTES -- 45 rows are empty or near-empty captures (an `stderr.log` with nothing in it).
# They are honestly captured output and `unknown` is the honest answer for them, so this is
# a stated population choice rather than a correctness fix: degenerate rows carry no signal,
# they are byte-identical to each other, and they were over half of all duplicate content.
EXCLUDE_PATH_PARTS = {"fixtures"}
MIN_BYTES = 9


def rows_for(root: pathlib.Path, max_bytes: int):
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix not in EXTS:
            continue
        if EXCLUDE_PATH_PARTS & set(f.parts):
            continue
        try:
            raw = f.read_bytes()
        except OSError:
            continue
        if len(raw) < MIN_BYTES:
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
        if truncated:
            # `outcome` is an ALL-quantified claim over the log's binaries -- "ok" means
            # every one of them said ok. Truncation removes binaries from view, so the
            # quantifier can no longer be established and the field must not carry a
            # value. Measured on a real 4.6MB CI log whose run CONCLUSION was failure:
            # the first 2MB contained 260 passing binaries and no failing one, so the
            # truncated read said "ok" about a run that failed. The counts stay, because
            # they are sums over the `test result:` lines actually read, which is what
            # they are in every log; only the universal claim is withdrawn.
            receipt["outcome"] = G.unknown(
                f"log truncated at {max_bytes} of {len(raw)} bytes; binaries after the cut "
                "are unread, so 'every binary was ok' cannot be established")
        yield f, receipt, truncated


# THE POPULATION, NAMED, because the count invites a false reading. This corpus is
# *captured stdout/stderr of commands this seat ran, stored under the logs root with a
# log-shaped extension*. It is NOT "gate logs": measured on 1110 files, 780 are not cargo
# output at all, and there is no path-based provenance partition to recover one -- 174
# directories, 32 of them mixed, and the purest gate-producer directory still holds 4
# non-cargo files. So the population is the sweep, and the two things that were wrong were
# the NAME and the BALANCE. This function fixes the balance; the name is fixed above.
def dedupe_by_content(rows):
    """Drop rows whose bytes duplicate an earlier row, keeping the first path in sort order.

    Found by measurement, not by review: before this ran, 10 content hashes spanned the
    train/held-out boundary and 14 of 219 held-out rows were byte-identical to a training
    row. A held-out score over those rows is a memorisation test wearing a generalisation
    label, and nothing in a class-count summary shows it. Path-disjointness -- which the
    partition does guarantee -- is not content-disjointness.
    """
    seen, kept = set(), []
    for r in sorted(rows, key=lambda r: r["log_path"]):
        h = r["receipt"]["log"]["value"]["sha256"]
        if h in seen:
            continue
        seen.add(h)
        kept.append(r)
    return kept, len(rows) - len(kept)


def partition(rows, holdout_frac: float):
    """Split by a stable hash of the log path, so the partition does not depend on read order.

    Deliberately NOT random-by-index: re-running after a new log lands would reshuffle an
    index-based split and silently move rows across the boundary, which is the leakage
    `check_split_integrity.py` exists to catch. Hashing the path pins each file to one
    side for its lifetime.
    """
    train, held = [], []
    for r in rows:
        h = int(G.hashlib.sha256(r["log_path"].encode("utf-8")).hexdigest()[:8], 16)
        (held if (h % 10_000) < holdout_frac * 10_000 else train).append(r)
    return train, held


def klass(row) -> str:
    return "cargo" if G.is_known(row["receipt"]["format"]) else "unknown"


def downsample(rows, seed: int):
    """Equalise the two format classes by discarding majority rows, with a fixed seed.

    Returns (kept, discarded_count). The discard is RETURNED rather than merely performed
    because a balancing step that does not report what it threw away has changed the
    population without saying so, which is the defect this function exists to fix.
    """
    import random
    by = collections.defaultdict(list)
    for r in rows:
        by[klass(r)].append(r)
    if not by["cargo"] or not by["unknown"]:
        return rows, 0
    n = min(len(by["cargo"]), len(by["unknown"]))
    rng = random.Random(seed)
    kept = []
    for c in ("cargo", "unknown"):
        pool = sorted(by[c], key=lambda r: r["log_path"])
        kept.extend(pool if len(pool) == n else rng.sample(pool, n))
    return sorted(kept, key=lambda r: r["log_path"]), len(rows) - len(kept)


def split_report(train, held, discarded: int, pre_train: int) -> dict:
    """Counts with denominators, plus the number any headline accuracy must beat.

    ALWAYS-ABSTAIN BASELINE. On a set that is 70% `unknown`, a model that answers `unknown`
    to everything scores 70%. Reporting accuracy without that number beside it makes doing
    nothing look like learning, so the baseline is emitted as a field of the split itself
    rather than left for a scorer to remember.
    """
    def dist(rows):
        d = collections.Counter(klass(r) for r in rows)
        return {"n": len(rows), "cargo": d["cargo"], "unknown": d["unknown"]}
    e = dist(held)
    base = max(e["cargo"], e["unknown"]) / e["n"] if e["n"] else None
    return {
        "population": ("captured stdout/stderr of commands this seat ran, under the logs root "
                       "with a log-shaped extension -- NOT a set of gate logs"),
        "train": {**dist(train), "rows_before_balance": pre_train, "discarded_by_balance": discarded},
        "holdout": {**e, "prevalence_preserved": True,
                    "always_abstain_accuracy": base,
                    "note": ("the held-out split keeps the natural mix because that is what a "
                             "caller hands the model; any accuracy below always_abstain_accuracy "
                             "is worse than answering 'unknown' to everything")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-root", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--max-bytes", type=int, default=2_000_000)
    ap.add_argument("--split-out", type=pathlib.Path,
                    help="also write train.jsonl/holdout.jsonl and a split report here")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance", choices=("none", "downsample"), default="downsample",
                    help="balance the TRAIN split only; the held-out split keeps natural prevalence")
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

    if a.split_out:
        rows = [json.loads(ln) for ln in (a.out / "receipts.jsonl").read_text(encoding="utf-8").splitlines() if ln]
        rows, dup_dropped = dedupe_by_content(rows)
        train, held = partition(rows, a.holdout_frac)
        pre = len(train)
        if a.balance == "downsample":
            train, discarded = downsample(train, a.seed)
        else:
            discarded = 0
        a.split_out.mkdir(parents=True, exist_ok=True)
        for name, part in (("train.jsonl", train), ("holdout.jsonl", held)):
            with (a.split_out / name).open("w", encoding="utf-8") as fh:
                for r in part:
                    fh.write(json.dumps(r, sort_keys=True) + "\n")
        rep = split_report(train, held, discarded, pre)
        rep["duplicate_content_rows_dropped"] = dup_dropped
        (a.split_out / "split_report.json").write_text(json.dumps(rep, indent=2, sort_keys=True))
        print(json.dumps({"split": rep}, indent=2, sort_keys=True))

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
        (root / "fixtures").mkdir()
        (root / "fixtures" / "golden.log").write_text("   Compiling x v0.1.0\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.0s\n")
        (root / "skip.md").write_text("test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.0s\n")
        got = {f.name: (r, t) for f, r, t in rows_for(root, 1_000_000)}
        cases.append(("only log-shaped extensions are read; a .md quoting a result is NOT a gate log",
                      set(got) == {"a.log", "b.log"}))
        cases.append(("a file under fixtures/ is excluded even though it IS cargo output",
                      "golden.log" not in got))
        cases.append(("a real cargo log reports format", got["a.log"][0]["format"]["value"] == "cargo-test"))
        cases.append(("a CI-watcher log does NOT claim cargo format", not G.is_known(got["b.log"][0]["format"])))
        # c.log is empty. It used to enter the dataset as a degenerate `unknown` row; under
        # MIN_BYTES it is excluded at the builder. The extractor's own answer for empty input
        # is unchanged and is covered by gate_log_receipt.py's self-test -- this case asserts
        # the BUILDER's exclusion, which is a different claim and is the one that moved.
        cases.append(("a sub-MIN_BYTES capture is excluded, not carried as a degenerate row",
                      "c.log" not in got))
        cases.append(("a non-empty NON-cargo capture is still kept and still answers unknown",
                      "b.log" in got and not G.is_known(got["b.log"][0]["format"])))
        cases.append(("every field is two-shaped on every row",
                      all(("value" in v) ^ ("unknown" in v) for _, (r, _) in got.items() for v in r.values())))
        big = root / "big.log"
        big.write_text("x" * 50 + "\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.0s\n")
        small = {f.name: (r, t) for f, r, t in rows_for(root, 10)}
        cases.append(("truncation is DECLARED on the row, not silent",
                      small["big.log"][1] is True and small["big.log"][0]["log"]["value"]["truncated"] is True))
        cases.append(("a truncated read does not invent counts",
                      not G.is_known(small["big.log"][0]["passed"])))
        okbig = root / "okbig.log"
        okbig.write_text("   Compiling x v0.1.0\n"
                         "test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.0s\n"
                         + "filler\n" * 200)
        cut = {f.name: (r, t) for f, r, t in rows_for(root, 120)}
        cases.append(("a truncated log does NOT claim outcome ok, because 'all ok' needs all",
                      cut["okbig.log"][1] is True and not G.is_known(cut["okbig.log"][0]["outcome"])))
        whole = {f.name: (r, t) for f, r, t in rows_for(root, 10_000_000)}
        cases.append(("...and the same log read WHOLE does report its outcome",
                      whole["okbig.log"][0]["outcome"]["value"] == "ok"))
    # --- split machinery -------------------------------------------------------
    # A synthetic corpus with a DELIBERATE skew (18 unknown : 6 cargo), because a balanced
    # fixture would let a no-op downsample pass every case below.
    mk = lambda i, cargo: {"log_path": f"/x/{i:03d}.log",
                           "receipt": {"format": (G.value("cargo-test") if cargo else G.unknown("no marker"))}}
    corpus = [mk(i, i % 4 == 0) for i in range(24)]
    dd = [mk(0, True), mk(1, False), mk(2, True)]
    for r in dd:
        r["receipt"]["log"] = G.value({"sha256": "AA" if r["log_path"].endswith("000.log") else "BB"})
    dd[2]["receipt"]["log"] = G.value({"sha256": "AA"})
    kept, dropped = dedupe_by_content(dd)
    cases.append(("duplicate content is dropped, keeping the first path", dropped == 1 and len(kept) == 2))
    cases.append(("...and the SURVIVING row is the first path, not an arbitrary one",
                  [r["log_path"] for r in kept] == ["/x/000.log", "/x/001.log"]))
    uniq = [mk(9, True), mk(8, False)]
    for i, r in enumerate(uniq):
        r["receipt"]["log"] = G.value({"sha256": f"U{i}"})
    cases.append(("dedupe drops nothing when every row is distinct", dedupe_by_content(uniq)[1] == 0))
    tr, hd = partition(corpus, 0.25)
    cases.append(("train and held-out share no row",
                  not ({r["log_path"] for r in tr} & {r["log_path"] for r in hd})))
    cases.append(("the partition covers the corpus", len(tr) + len(hd) == len(corpus)))
    raw_hd = collections.Counter(klass(r) for r in hd)
    bal, disc = downsample(tr, 0)
    cases.append(("balancing equalises the TRAIN classes",
                  collections.Counter(klass(r) for r in bal)["cargo"]
                  == collections.Counter(klass(r) for r in bal)["unknown"]))
    cases.append(("the discard count is the rows actually dropped", disc == len(tr) - len(bal)))
    cases.append(("balancing DID drop something on a skewed corpus, so the case above is live",
                  disc > 0))
    rep = split_report(bal, hd, disc, len(tr))
    cases.append(("the held-out split is untouched by balancing",
                  (rep["holdout"]["cargo"], rep["holdout"]["unknown"]) == (raw_hd["cargo"], raw_hd["unknown"])))
    base = rep["holdout"]["always_abstain_accuracy"]
    cases.append(("the always-abstain baseline is the held-out majority rate",
                  base is None or abs(base - max(raw_hd.values()) / sum(raw_hd.values())) < 1e-9))
    # The docstring's whole claim: adding a file must not move an existing file's side.
    grown = corpus + [mk(900 + i, i % 2 == 0) for i in range(7)]
    tr2, hd2 = partition(grown, 0.25)
    before = {r["log_path"]: "h" for r in hd} | {r["log_path"]: "t" for r in tr}
    after = {r["log_path"]: "h" for r in hd2} | {r["log_path"]: "t" for r in tr2}
    cases.append(("adding rows moves no existing row across the boundary",
                  all(after[k] == v for k, v in before.items())))
    cases.append(("...and the growth fixture really did add rows", len(grown) > len(corpus)))

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
