#!/usr/bin/env python3
"""Construct seeded, within-verb memory/task adapter splits from a pinned capture."""

import argparse
import collections
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

SEED = 20260912


def calls(text):
    """Extract qualified calls outside double-quoted literals."""
    blank = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    return re.findall(r"\b([a-z_]+\.[a-z_]+)\s*\(", blank)


def load(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def write(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def build(source, schema, out):
    raw = source.read_bytes()
    if not hashlib.sha256(raw).hexdigest().startswith("6ff089c0e380213e"):
        raise ValueError("source hash differs")
    rows = load(source)
    if len(rows) != 2806 or any("\n" in r["completion"] for r in rows):
        raise ValueError("expected 2806 single-line completions")
    files = sorted(schema.glob("*.json"))
    if len(files) != 100:
        raise ValueError("expected 100 schema JSON files")
    manifest = {
        "files": [
            {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in files
        ]
    }
    digest = hashlib.sha256()
    for item in manifest["files"]:
        digest.update(f"{item['name']}:{item['sha256']}\n".encode())
    if digest.hexdigest()[:16] != "625b8392f752d63d":
        raise ValueError("schema fingerprint differs")
    out.mkdir(parents=True, exist_ok=True)
    (out / "schemas").mkdir(exist_ok=True)
    for path in files:
        shutil.copyfile(path, out / "schemas" / path.name)
    shutil.copyfile(source, out / "train.jsonl")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rng = random.Random(SEED)
    seen, held, router, counts = set(), [], [], {}
    for label, (family, expected) in enumerate((("memory", 257), ("gtd", 204))):
        family_rows = [
            r
            for r in rows
            if {v.split(".")[0] for v in calls(r["completion"])} == {family}
        ]
        if len(family_rows) != expected:
            raise ValueError(f"{family}: expected {expected}, found {len(family_rows)}")
        unique = []
        for row in family_rows:
            key = row["prompt"].lower()
            if key not in seen:
                unique.append(row)
                seen.add(key)
        groups = collections.defaultdict(list)
        for row in unique:
            verbs = set(calls(row["completion"]))
            if len(verbs) != 1:
                raise ValueError("stratification requires one distinct verb per row")
            groups[next(iter(verbs))].append(row)
        expected_verbs = (
            {"memory.recall", "memory.feedback", "memory.prune"}
            if label == 0
            else {"gtd.tasks", "gtd.transition", "gtd.next"}
        )
        if set(groups) != expected_verbs:
            raise ValueError("family verb set differs")
        allocation = {v: max(1, 40 * len(g) // len(unique)) for v, g in groups.items()}
        while sum(allocation.values()) < 40:
            v = max(
                sorted(groups),
                key=lambda v: 40 * len(groups[v]) / len(unique) - allocation[v],
            )
            allocation[v] += 1
        while sum(allocation.values()) > 40:
            v = max(
                (v for v in sorted(groups) if allocation[v] > 1),
                key=lambda v: allocation[v] - 40 * len(groups[v]) / len(unique),
            )
            allocation[v] -= 1
        train, test = [], []
        for verb in sorted(groups):
            group = groups[verb]
            rng.shuffle(group)
            test.extend(group[: allocation[verb]])
            train.extend(group[allocation[verb] :])
        rng.shuffle(train)
        rng.shuffle(test)
        if len(test) != 40 or len(train) < 48:
            raise ValueError("insufficient rows")
        directory = out / f"adapter-{family}"
        directory.mkdir(exist_ok=True)
        write(directory / "train.jsonl", train[:48])
        write(directory / "valid.jsonl", test[:8])
        write(directory / "heldout.jsonl", test)
        write(out / f"adapter-train-{family}.jsonl", train)
        write(out / f"heldout-{family}.jsonl", test)
        offset = len(held)
        held.extend(
            dict(row, idx=offset + i, family=family) for i, row in enumerate(test)
        )
        for split, selected in (("train", train[:48]), ("test", test)):
            if len(selected) < 30:
                raise ValueError("router requires 30 rows per domain and split")
            router.extend(
                {"prompt": r["prompt"], "label": label, "split": split}
                for r in selected
            )
        counts[family] = {
            "raw": expected,
            "deduped": len(unique),
            "dropped": expected - len(unique),
            "heldout_by_verb": allocation,
            "remaining": len(train),
        }
    if collections.Counter(r["family"] for r in held) != {"memory": 40, "gtd": 40}:
        raise ValueError("held-out labels must balance 40/40")
    write(out / "heldout.jsonl", held)
    write(out / "router.jsonl", router)
    (out / "construction.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--schema-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("w6-input"))
    args = parser.parse_args()
    build(args.source, args.schema_dir, args.out)
