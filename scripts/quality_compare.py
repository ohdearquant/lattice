#!/usr/bin/env python3
"""Compare named greedy-generation dumps without loading a model.

Schema v1 is an object containing `schema_version: 1`, a nonempty `name`,
`config` (a JSON object with `mode: "greedy"`, positive `max_new_tokens`, and
nonempty `stop_policy`), and a nonempty `cases` list. Each case has a unique
nonempty `id`, a nonempty `prompt`, nonempty `prompt_ids`, `generated_ids`,
`stop_reason` (length, eos, or stop), and `complete: true`. Token IDs are
nonnegative integers. Stop tokens are excluded from `generated_ids` and a
`length` result must contain exactly max_new_tokens IDs. Other config keys
are permitted but must match exactly, including their JSON types.

Outputs are report-only: prefix rates include only cases with enough tokens
on both sides, and never imply issue acceptance. Real producers do not yet
emit this schema; logits, KL and model provenance are deferred.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class DumpError(ValueError):
    """A dump cannot be compared safely."""


def _fail(message: str) -> None:
    raise DumpError(message)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field}: expected a nonempty string")
    return value


def _ids(value: Any, field: str, *, allow_empty: bool) -> list[int]:
    if not isinstance(value, list) or (not allow_empty and not value):
        _fail(f"{field}: expected {'a' if allow_empty else 'a nonempty'} token ID list")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in value):
        _fail(f"{field}: token IDs must be nonnegative integers")
    return value


def validate_dump(raw: Any, label: str) -> dict[str, Any]:
    if (not isinstance(raw, dict) or isinstance(raw.get("schema_version"), bool)
            or not isinstance(raw.get("schema_version"), int) or raw.get("schema_version") != 1):
        _fail(f"{label}: expected schema_version 1 object")
    name = _text(raw.get("name"), f"{label}.name")
    config = raw.get("config")
    if not isinstance(config, dict) or config.get("mode") != "greedy":
        _fail(f"{label}.config: expected greedy config")
    maximum = config.get("max_new_tokens")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        _fail(f"{label}.config.max_new_tokens: expected positive integer")
    _text(config.get("stop_policy"), f"{label}.config.stop_policy")
    try:
        config_identity = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise DumpError(f"{label}.config: invalid JSON: {error}") from error
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        _fail(f"{label}.cases: expected nonempty list")
    by_id: dict[str, Any] = {}
    for index, case in enumerate(cases):
        field = f"{label}.cases[{index}]"
        if not isinstance(case, dict):
            _fail(f"{field}: expected object")
        if "logits" in case:
            _fail(f"{field}: logits are outside schema v1")
        case_id = _text(case.get("id"), f"{field}.id")
        if case_id in by_id:
            _fail(f"{label}: duplicate case ID {case_id!r}")
        prompt = _text(case.get("prompt"), f"{field}.prompt")
        prompt_ids = _ids(case.get("prompt_ids"), f"{field}.prompt_ids", allow_empty=False)
        generated = _ids(case.get("generated_ids"), f"{field}.generated_ids", allow_empty=True)
        if len(generated) > maximum:
            _fail(f"{field}: generated length exceeds max_new_tokens")
        stop = case.get("stop_reason")
        if stop not in ("length", "eos", "stop"):
            _fail(f"{field}.stop_reason: expected length, eos, or stop")
        if stop == "length" and len(generated) != maximum:
            _fail(f"{field}: length stop requires max_new_tokens IDs")
        if case.get("complete") is not True:
            _fail(f"{field}: producer completion marker missing")
        by_id[case_id] = {"prompt": prompt, "prompt_ids": prompt_ids,
                          "generated_ids": generated, "stop_reason": stop}
    return {"name": name, "config_identity": config_identity, "cases": by_id}


def compare(reference_raw: Any, candidate_raw: Any) -> dict[str, Any]:
    reference = validate_dump(reference_raw, "reference")
    candidate = validate_dump(candidate_raw, "candidate")
    if reference["config_identity"] != candidate["config_identity"]:
        _fail("generation config or stop policy mismatch")
    left, right = reference["cases"], candidate["cases"]
    if left.keys() != right.keys():
        _fail("case ID sets differ")
    cases = []
    for case_id in sorted(left):
        a, b = left[case_id], right[case_id]
        if a["prompt"] != b["prompt"] or a["prompt_ids"] != b["prompt_ids"]:
            _fail(f"prompt identity mismatch for {case_id!r}")
        x, y = a["generated_ids"], b["generated_ids"]
        prefix = 0
        while prefix < min(len(x), len(y)) and x[prefix] == y[prefix]:
            prefix += 1
        if prefix < min(len(x), len(y)):
            divergence, kind = prefix, "token"
        elif len(x) != len(y):
            divergence, kind = prefix, "length"
        elif a["stop_reason"] != b["stop_reason"]:
            divergence, kind = prefix, "termination"
        else:
            divergence, kind = None, "identical"
        cases.append({"id": case_id, "prompt": a["prompt"], "exact_prefix_length": prefix,
                      "first_divergence": divergence, "divergence_kind": kind,
                      "reference_length": len(x), "candidate_length": len(y),
                      "reference_stop_reason": a["stop_reason"], "candidate_stop_reason": b["stop_reason"]})
    histogram: dict[str, int] = {}
    for case in cases:
        key = "identical" if case["first_divergence"] is None else str(case["first_divergence"])
        histogram[key] = histogram.get(key, 0) + 1
    summary = {}
    for n in (10, 50):
        eligible = [case for case in cases if min(case["reference_length"], case["candidate_length"]) >= n]
        matching = sum(case["exact_prefix_length"] >= n for case in eligible)
        summary[str(n)] = {"matched": matching, "eligible": len(eligible),
                           "insufficient": len(cases) - len(eligible),
                           "rate": matching / len(eligible) if eligible else None}
    return {"reference": reference["name"], "candidate": candidate["name"], "cases": cases,
            "first_divergence_histogram": histogram, "prefix_matches": summary}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    _fail(f"non-finite JSON constant {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    try:
        with args.reference.open(encoding="utf-8") as handle:
            reference = json.load(handle, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        with args.candidate.open(encoding="utf-8") as handle:
            candidate = json.load(handle, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        result = compare(reference, candidate)
    except (OSError, UnicodeError, json.JSONDecodeError, DumpError) as error:
        print(f"quality comparison refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
