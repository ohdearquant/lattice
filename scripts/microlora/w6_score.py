#!/usr/bin/env python3
"""Validate controls before scoring deterministic adapter-output selections."""

import argparse
import collections
import json
import random
import re
import subprocess
import sys
from pathlib import Path

from score_dsl_args import EXIT_FAIL, EXIT_PASS, EXIT_REFUSED, Schemas, score_row
from w6_build_sets import calls, load, write


def validate(rows, args, name):
    pairs = args.out / f"{name}-pairs.jsonl"
    validated = args.out / f"{name}-validated.jsonl"
    write(pairs, rows)
    result = subprocess.run(
        [str(args.validator.resolve())],
        input="".join(json.dumps({"completion": r["completion"]}) + "\n" for r in rows),
        text=True,
        capture_output=True,
        check=False,
    )
    validated.write_text(result.stdout)
    (args.out / f"{name}-validator-stderr.log").write_text(result.stderr)
    parsed = load(validated)
    if len(parsed) != len(rows) or [r["line"] for r in parsed] != list(
        range(1, len(rows) + 1)
    ):
        raise ValueError("validator row identity mismatch")
    if any(r.get("parser_source_sha256") != args.parser_sha for r in parsed):
        raise ValueError("parser source changed")
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("score_dsl_args.py")),
        "--validated",
        str(validated),
        "--pairs",
        str(pairs),
        "--schema-dir",
        str(args.schema_dir),
        "--manifest",
        str(args.manifest),
    ]
    scored = subprocess.run(cmd, text=True, capture_output=True, check=False)
    (args.out / f"{name}-scorer.log").write_text(
        json.dumps(cmd)
        + "\n"
        + scored.stdout
        + scored.stderr
        + f"\nexit_code={scored.returncode}\n"
    )
    scores = []
    for row, parsed_row in zip(rows, parsed):
        if parsed_row["ok"] and not isinstance(
            parsed_row.get("canonical_request"), str
        ):
            raise ValueError("accepted row missing canonical request")
        score = score_row(parsed_row, args.schemas, row["prompt"])
        verbs = calls(row["completion"])
        family_ok = bool(verbs) and all(v.split(".")[0] == row["family"] for v in verbs)
        scores.append(
            dict(
                idx=row["idx"],
                parser_ok=parsed_row["ok"],
                canonical=parsed_row.get("canonical_request"),
                family_ok=family_ok,
                callable_on_family=bool(
                    parsed_row["ok"] and score["verdict"] == "PASS" and family_ok
                ),
                **score,
            )
        )
    counts = collections.Counter(row["verdict"] for row in scores)
    expected_exit = (
        EXIT_FAIL
        if counts["FAIL"]
        else EXIT_REFUSED
        if counts["REFUSED"]
        else EXIT_PASS
    )
    cli_counts = {
        name: int(count)
        for name, count in re.findall(
            r"^\s*(PASS|FAIL|REFUSED)\s+(\d+)\s", scored.stdout, re.MULTILINE
        )
    }
    if scored.returncode != expected_exit or cli_counts != {
        name: counts[name] for name in ("PASS", "FAIL", "REFUSED")
    }:
        raise ValueError("scorer CLI result differs from per-row scoring")
    write(args.out / f"{name}-scores.jsonl", scores)
    return scores


def controls(gold, args):
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("score_dsl_args.py")),
        "--self-test",
        "--schema-dir",
        str(args.schema_dir),
        "--manifest",
        str(args.manifest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    (args.out / "scorer-self-test.log").write_text(result.stdout + result.stderr)
    c0 = (
        result.returncode == 0
        and re.search(r"capture=625b8392f752d63d\s+verbs=98", result.stdout) is not None
        and "['FAIL', 'PASS', 'REFUSED']" in result.stdout
    )
    table = [
        {
            "control": "C0",
            "ok": c0,
            "expected": "three verdicts; capture=625b8392f752d63d verbs=98",
            "actual": result.stdout.splitlines()[-1:],
        }
    ]
    if not c0:
        (args.out / "controls.json").write_text(json.dumps(table, indent=2) + "\n")
        raise ValueError("K4: C0 failed")
    canonical_gold = None
    for name in ("C1", "C2", "C3", "C4"):
        rows = [dict(r) for r in gold]
        for row in rows:
            if name == "C2":
                blank = re.sub(
                    r'"(?:\\.|[^"\\])*"', lambda m: " " * len(m[0]), row["completion"]
                )
                match = re.search(r"\(\s*([a-z_]+)\s*=", blank)
                if not match:
                    raise ValueError("K4: gold has no first parameter to mutate")
                key = match[1]
                new = {
                    "query": "q",
                    "limit": "max",
                    "id": "ident",
                    "status": "state",
                }.get(key, key + "_x")
                row["completion"] = (
                    row["completion"][: match.start(1)]
                    + new
                    + row["completion"][match.end(1) :]
                )
            elif name == "C3":
                row["family"] = "gtd" if row["family"] == "memory" else "memory"
            elif name == "C4":
                row["completion"] = "Here is the request: " + row["completion"]
        scores = validate(rows, args, name)
        if name == "C1":
            canonical_gold = [r["canonical"] for r in scores]
            count = sum(r["callable_on_family"] for r in scores)
        elif name == "C2":
            count = sum(r["verdict"] != "PASS" for r in scores)
        elif name == "C3":
            count = sum(not r["family_ok"] for r in scores)
        else:
            count = sum(not r["parser_ok"] for r in scores)
        table.append(
            {"control": name, "expected": 80, "actual": count, "ok": count == 80}
        )
    (args.out / "controls.json").write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps(table, indent=2))
    if not all(r["ok"] for r in table):
        raise ValueError("K4: SCORER DEFECT")
    return canonical_gold


def score_generations(gold, canonical_gold, args):
    outputs = {}
    for arm in ("base", "M", "G"):
        rows = load(args.generations / f"gen-{arm}.jsonl")
        if len(rows) != 80:
            raise ValueError(f"expected 80 generations: {arm}")
        for row, target in zip(rows, gold):
            if any(row[k] != target[k] for k in ("idx", "family", "prompt")):
                raise ValueError("generation identity mismatch")
        duplicates = load(args.generations / f"duplicates-{arm}.jsonl")
        if [r["idx"] for r in duplicates] != [0, 1, 40, 41]:
            raise ValueError("K5: missing duplicate processes")
        for row in duplicates:
            if any(
                row[k] != rows[row["idx"]][k] for k in ("family", "prompt", "output")
            ):
                raise ValueError("K5: duplicate outputs differ")
        outputs[arm] = rows
    decisions = re.findall(
        r"^decision idx=(\d+) label=(\d+) predicted=(\d+)$",
        (args.generations / "route-decisions.log").read_text(),
        re.MULTILINE,
    )
    if len(decisions) != 80:
        raise ValueError("missing router decisions")
    predicted = []
    for i, (idx, label, pred) in enumerate(decisions):
        if (
            int(idx) != i
            or int(label) != (gold[i]["family"] == "gtd")
            or pred not in ("0", "1")
        ):
            raise ValueError("router identity mismatch")
        predicted.append(int(pred))
    rng = random.Random(20260912)
    selections = {
        "base": ["base"] * 80,
        "single-M": ["M"] * 80,
        "single-G": ["G"] * 80,
        "routed": ["M" if p == 0 else "G" for p in predicted],
        "oracle": ["M" if r["family"] == "memory" else "G" for r in gold],
        "random": [rng.choice(("M", "G")) for _ in gold],
    }
    summary = {}
    for arm, selection in selections.items():
        rows = []
        for i, source in enumerate(selection):
            row = dict(outputs[source][i])
            row.update(
                completion=row["output"].split("\n", 1)[0].strip(),
                newline_emitted="\n" in row["output"],
                selected=source,
            )
            rows.append(row)
        scores = validate(rows, args, arm)
        result = {}
        for family in ("overall", "memory", "gtd"):
            indices = [
                i
                for i, row in enumerate(gold)
                if family == "overall" or row["family"] == family
            ]
            result[family] = {
                "n": len(indices),
                "callable": sum(scores[i]["callable_on_family"] for i in indices),
                "exact": sum(
                    scores[i]["parser_ok"]
                    and scores[i]["canonical"] == canonical_gold[i]
                    for i in indices
                ),
                "newline": sum(rows[i]["newline_emitted"] for i in indices),
            }
        summary[arm] = result
    summary["routing_correct"] = sum(
        p == (r["family"] == "gtd") for p, r in zip(predicted, gold)
    )
    base = summary["base"]["overall"]["callable"]
    single = max(
        summary[arm]["overall"]["callable"] for arm in ("single-M", "single-G")
    )
    decision = {"routing_failure": summary["routing_correct"] < 72, "nll": {}}
    for family, suffix in (("memory", "M"), ("gtd", "G")):
        log = args.generations / f"train-{suffix}.log"
        if log.exists():
            points = re.findall(
                r"step\s+(0|96)\s+train NLL: ([0-9.]+)\s+held-out NLL: ([0-9.]+)",
                log.read_text(),
            )
            if len(points) != 2 or [p[0] for p in points] != ["0", "96"]:
                raise ValueError("missing step-0/final NLL read-out")
            before, after = float(points[0][2]), float(points[1][2])
            if before <= 0:
                raise ValueError("invalid initial NLL")
            decision["nll"][family] = {
                "initial_train": float(points[0][1]),
                "final_train": float(points[1][1]),
                "initial_valid": before,
                "final_valid": after,
                "drop_fraction": 1 - after / before,
            }
    if base >= 72:
        decision["verdict"] = "KILLED(K1): CEILING"
    elif single <= base:
        decision["verdict"] = "KILLED(K2): ADAPTERS NOT USEFUL"
    elif len(decision["nll"]) != 2:
        raise ValueError("both trainer NLL read-outs required")
    elif any(v["drop_fraction"] < 0.5 for v in decision["nll"].values()):
        decision["verdict"] = "KILLED(K3): UNDER-TRAINED; decision rule not applied"
    elif decision["routing_failure"]:
        decision["verdict"] = "KILLED(K6): ROUTING failure; oracle retained"
    else:
        failures = []
        if summary["routed"]["overall"]["callable"] < single - 4:
            failures.append("routed below best single adapter minus five points")
        for family in ("memory", "gtd"):
            if (
                summary["routed"][family]["callable"]
                <= summary["base"][family]["callable"]
            ):
                failures.append(f"routed does not beat base on {family}")
        decision["verdict"] = (
            "NOT USEFUL: " + "; ".join(failures) if failures else "USEFUL"
        )
    (args.out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    (args.out / "response-scores.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "| arm | callable overall | memory | gtd | exact overall | memory | gtd |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in selections:
        cells = []
        for metric in ("callable", "exact"):
            for family in ("overall", "memory", "gtd"):
                cell = summary[arm][family]
                cells.append(
                    f"{cell[metric]}/{cell['n']} ({100 * cell[metric] / cell['n']:.1f}%)"
                )
        lines.append("| " + arm + " | " + " | ".join(cells) + " |")
    (args.out / "response-table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("w6-input"))
    parser.add_argument("--out", type=Path, default=Path(".khive/leg"))
    parser.add_argument(
        "--validator", type=Path, default=Path(".khive/leg/khive-dsl-validator")
    )
    parser.add_argument("--generations", type=Path, default=Path(".khive/leg/w6-out"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.schema_dir, args.manifest = (
        args.input / "schemas",
        args.input / "manifest.json",
    )
    args.schemas = Schemas(args.schema_dir, args.manifest)
    if args.schemas.drift:
        raise ValueError("schema capture drift")
    result = subprocess.run(
        [str(args.validator.resolve()), "--self-test"],
        capture_output=True,
        text=True,
        check=True,
    )
    (args.out / "validator-self-test.json").write_text(result.stdout)
    receipt = json.loads(result.stdout)
    if receipt.get("ok") is not True or not receipt.get("parser_source_sha256"):
        raise ValueError("validator self-test failed")
    args.parser_sha = receipt["parser_source_sha256"]
    gold = load(args.input / "heldout.jsonl")
    if (
        len(gold) != 80
        or [r["idx"] for r in gold] != list(range(80))
        or collections.Counter(r["family"] for r in gold) != {"memory": 40, "gtd": 40}
    ):
        raise ValueError("expected 80 uniquely indexed, balanced gold rows")
    canonical_gold = controls(gold, args)
    if not args.self_test:
        score_generations(gold, canonical_gold, args)


if __name__ == "__main__":
    main()
