#!/usr/bin/env python3
"""Score generated khive DSL for ARGUMENT validity, beyond parser acceptance.

WHY THIS EXISTS. `dsl_validator.rs` runs the real parser: syntax, AST round-trip, canonical
reparse. It emits `executed: false` and it is right to. But it cannot know what a verb's
arguments are called, so `memory.recall(q="...")` validates perfectly and then fails at
runtime, because the parameter is `query`. Parser acceptance is a weaker property than
callability, and an adapter trained against the weaker one learns to emit plausible-looking
commands that do not run.

WHAT IT DOES NOT DO. It never executes a generated command, and it never contacts a server.
It reads the captured verb schemas and compares names. That is the whole mechanism.

THE ONE DESIGN RULE THAT MATTERS, and it decides whether this thing is worth having:
ABSENCE IS NOT PERMISSION. A verb with no captured schema is REFUSED, never passed. The
capture distinguishes the two cases and that is what makes the rule enforceable: all 98
captured verbs carry an explicit `params` list, and the 8 parameterless ones carry an
explicit EMPTY list. So "takes nothing" is a positive fact in the data, while "not captured"
is a missing file. A scorer that treated a missing schema as "no constraints" would pass
every hallucinated verb in the corpus and report a flattering number.

Verdicts: PASS, FAIL, REFUSED. All three are reachable; `--self-test` proves it rather than
asserting it, because a scorer that can only emit one verdict is not measuring anything.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# No baked-in paths. Every tracked script beside this one takes its inputs as explicit
# flags, and a default pointing into a dated local workspace resolves on exactly one
# machine while reading, in a public tree, as though it were a repository location.
# Making them required also removes the shape of the bug this file already had once:
# a self-test that read a module default instead of the flag it was handed reported
# "OK verbs=98" while pointed at an empty directory.

EXIT_PASS, EXIT_USAGE, EXIT_REFUSED, EXIT_FAIL = 0, 2, 3, 4


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _find_params(obj):
    """The captured schemas nest `params` at a depth that is not contractual, so search."""
    if isinstance(obj, dict):
        if isinstance(obj.get("params"), list):
            return obj["params"]
        for value in obj.values():
            found = _find_params(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_params(value)
            if found is not None:
                return found
    return None


class Schemas:
    """The captured verb schemas, with their identity.

    The fingerprint is over the manifest's (name, sha256) pairs, not over a timestamp: a date
    says when a capture was taken, a fingerprint says WHICH capture a score came from, and only
    the second one survives the capture being retaken.
    """

    def __init__(self, schema_dir: Path, manifest_path: Path | None):
        self.dir = schema_dir
        self.verbs: dict[str, dict] = {}
        self.fingerprint = "unverified"
        self.drift: list[str] = []

        for path in sorted(schema_dir.glob("*.json")):
            if path.name.startswith("_"):
                continue
            params = _find_params(json.loads(path.read_text()))
            if params is None:
                # No params KEY at all is a broken capture, not a parameterless verb.
                self.drift.append(f"{path.name}: no params key")
                continue
            self.verbs[path.stem] = {
                "known": {p["name"] for p in params if "name" in p},
                "required": {p["name"] for p in params if p.get("required") and "name" in p},
                "types": {p["name"]: p.get("type") for p in params if "name" in p},
            }

        if manifest_path and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            pairs = []
            for entry in manifest.get("files", []):
                pairs.append((entry["name"], entry["sha256"]))
                actual = schema_dir / entry["name"]
                if not actual.exists():
                    self.drift.append(f"{entry['name']}: missing")
                elif _sha256(actual) != entry["sha256"]:
                    self.drift.append(f"{entry['name']}: hash mismatch")
            digest = hashlib.sha256()
            for name, sha in sorted(pairs):
                digest.update(f"{name}:{sha}\n".encode())
            self.fingerprint = digest.hexdigest()[:16]



# The capture's type vocabulary is NOT normalized: 15 distinct strings for 415 params, with
# `bool`/`boolean`, `array of string`/`array<string>`, `array of object`/`array<object>` and
# `number`/`float` each naming one thing two ways. A checker keying on raw spellings would
# silently mis-handle whichever spelling it did not anticipate, so everything is normalized first.
# Anything NOT in this map is reported `unchecked` and counted -- never passed, never failed.
_TYPE_NORMAL = {
    "string": "string", "uuid": "string",
    "integer": "integer",
    "number": "number", "float": "number",
    "boolean": "boolean", "bool": "boolean",
    "object": "object",
    "array": "array", "array of string": "array", "array<string>": "array",
    "array of uuid": "array", "array of object": "array", "array<object>": "array",
    # deliberately absent: "string | array<string>" -- a union this checker will not guess at.
}


def check_arg_type(wire_arg, declared: str | None) -> str:
    """ok | mismatch | unchecked. `unchecked` is a first-class outcome, not a quiet pass."""
    norm = _TYPE_NORMAL.get((declared or "").strip())
    if norm is None:
        return "unchecked"
    if not isinstance(wire_arg, dict):
        return "unchecked"
    kind = wire_arg.get("kind")
    if kind == "prev_ref":
        # A $prev reference's runtime type is unknowable without executing, and this never executes.
        return "unchecked"
    if kind == "array":
        return "ok" if norm == "array" else "mismatch"
    if kind == "object":
        return "ok" if norm == "object" else "mismatch"
    if kind != "value":
        return "unchecked"
    value = wire_arg.get("value")
    if isinstance(value, bool):
        actual = "boolean"          # before int: bool is an int subclass in Python
    elif isinstance(value, int):
        actual = "integer"
    elif isinstance(value, float):
        actual = "number"
    elif isinstance(value, str):
        actual = "string"
    elif isinstance(value, list):
        actual = "array"
    elif isinstance(value, dict):
        actual = "object"
    else:
        return "unchecked"
    if norm == "number" and actual == "integer":
        return "ok"                 # an integer is an acceptable number
    return "ok" if actual == norm else "mismatch"



# --- axis 3: groundedness -------------------------------------------------------------
# A literal can be well-named and well-typed and still be invented. session.export(
# id="00000000-0000-0000-0000-000000000000") passes both earlier axes and names nothing.
# Type checking is structurally blind to this, so it needs its own axis.
#
# The decision boundary is measured, not assumed. Over the 917-row reference test split,
# all 1837 string literals in reference completions are grounded in their own prompt:
# 79.0% verbatim, 12.5% as base64 of prompt text (blob.put payloads), 8.5% as nested DSL
# whose inner literals are grounded (schedule.schedule action strings). Zero residual.
# A zero reference-violation rate is what makes "ungrounded" decidable rather than a
# threshold someone has to pick.
#
# HONESTY BOUND ON THAT NUMBER: this dataset is synthetic (dsl-final-audit.json records
# 4201 rows, real_rows=1). A 100% groundedness rate is partly a property of the generator
# that produced prompt/completion pairs from shared material, not a discovered law of the
# task. It licenses "an ungrounded literal is a hallucination ON THIS DATASET" and does
# not license a claim about held-out human-written prompts.

_B64 = re.compile(r"^[A-Za-z0-9+/]{16,}={0,2}$")
_INNER_ESCAPED = re.compile(r'\\"([^"\\]*)\\"')
_INNER_PLAIN = re.compile(r'"([^"]*)"')
_OPISH = re.compile(r"^[a-z_][a-z0-9_.]*\(.*\)$", re.S)
_SCHEMA_ISH = re.compile(r"[a-z][a-z0-9_.:-]*|[\d:T+\-]+")


def ground_value(value: str, prompt: str) -> str:
    """verbatim | base64 | nested-dsl | UNGROUNDED."""
    if not value:
        return "verbatim"
    if value in prompt:
        return "verbatim"
    if _B64.match(value):
        try:
            if base64.b64decode(value, validate=True).decode("utf-8") in prompt:
                return "base64"
        except (binascii.Error, UnicodeDecodeError):
            pass
    # Nested DSL arrives in TWO representations and a rule written for one is inert on the
    # other. In raw completion text the inner quotes are JSON-escaped (\\"); in a parsed arg
    # value they are plain ("). The first version of this matched only the escaped form, so
    # it silently found nothing on parsed input and reported 87 of 917 REFERENCE rows as
    # ungrounded -- gold data, wrong by construction. Both forms are matched here, and only
    # for values that are actually shaped like an op, so an ordinary string that happens to
    # contain quotes is not granted a recovery path it has not earned.
    inner = [g for g in _INNER_ESCAPED.findall(value) if g]
    if not inner and _OPISH.match(value):
        inner = [g for g in _INNER_PLAIN.findall(value) if g]
    if inner and all(g in prompt or _SCHEMA_ISH.fullmatch(g) for g in inner):
        return "nested-dsl"
    return "UNGROUNDED"


def _wire_strings(wire_arg):
    """Every string literal an arg carries, unwrapping the validator's wire shape."""
    if isinstance(wire_arg, str):
        yield wire_arg
    elif isinstance(wire_arg, dict):
        if wire_arg.get("kind") == "prev_ref":
            return              # $prev.id refers to a sibling op, never to the prompt
        v = wire_arg.get("value", wire_arg)
        if v is not wire_arg:
            yield from _wire_strings(v)
    elif isinstance(wire_arg, list):
        for item in wire_arg:
            yield from _wire_strings(item)


def score_groundedness(op: dict, prompt: str) -> dict:
    hits, ungrounded = {}, []
    for name, wire_arg in (op.get("args") or {}).items():
        for value in _wire_strings(wire_arg):
            verdict = ground_value(value, prompt)
            hits[verdict] = hits.get(verdict, 0) + 1
            if verdict == "UNGROUNDED":
                ungrounded.append({"arg": name, "value": value[:120]})
    return {"recovery": hits, "ungrounded": ungrounded}

def score_op(op: dict, schemas: Schemas, prompt: str | None = None) -> dict:
    verb = op.get("tool")
    args = set((op.get("args") or {}).keys())
    entry = schemas.verbs.get(verb)
    if entry is None:
        return {
            "verb": verb,
            "verdict": "REFUSED",
            "reason": f"verb {verb!r} is not in the capture, so its arguments cannot be judged; "
            "absence of a schema is not permission",
        }
    unknown = sorted(args - entry["known"])
    missing = sorted(entry["required"] - args)

    mismatched, unchecked = [], []
    for name, wire_arg in (op.get("args") or {}).items():
        if name in unknown:
            continue                # already failing on the name; the type is moot
        outcome = check_arg_type(wire_arg, entry["types"].get(name))
        if outcome == "mismatch":
            mismatched.append(name)
        elif outcome == "unchecked":
            unchecked.append(name)

    # Groundedness is only evaluated when a prompt was supplied. With no prompt the axis
    # reports not_evaluated and never contributes a PASS: a check that did not run is not
    # a check that passed, and reporting it as absent is the only way a reader can tell.
    if prompt is None:
        ground = {"status": "not_evaluated"}
        ungrounded = []
    else:
        ground = score_groundedness(op, prompt)
        ground["status"] = "evaluated"
        ungrounded = ground["ungrounded"]

    if unknown or missing or mismatched or ungrounded:
        return {
            "verb": verb,
            "verdict": "FAIL",
            "unknown_args": unknown,
            "missing_required": missing,
            "type_mismatches": sorted(mismatched),
            "unchecked_args": sorted(unchecked),
            "groundedness": ground,
        }
    return {
        "verb": verb,
        "verdict": "PASS",
        "arg_count": len(args),
        "unchecked_args": sorted(unchecked),
        "groundedness": ground,
    }


def score_row(row: dict, schemas: Schemas, prompt: str | None = None) -> dict:
    if not row.get("ok"):
        return {"verdict": "FAIL", "reason": "parser rejected the completion", "ops": []}
    ops = [score_op(op, schemas, prompt) for op in row.get("ops", [])]
    if not ops:
        return {"verdict": "REFUSED", "reason": "no ops in a parser-accepted row", "ops": []}
    verdicts = {o["verdict"] for o in ops}
    # FAIL dominates REFUSED: a row with a known-bad argument is bad whatever else it contains.
    overall = "FAIL" if "FAIL" in verdicts else ("REFUSED" if "REFUSED" in verdicts else "PASS")
    return {"verdict": overall, "ops": ops}


def _self_test(schema_dir: Path, manifest: Path) -> int:
    """Prove all three verdicts are reachable. A one-verdict scorer measures nothing.

    Takes the schema dir as an ARGUMENT rather than reading the module default. The first
    version hardcoded the default, so `--self-test --schema-dir <empty>` printed
    "self-test OK ... verbs=98" and exited 0 while pointed at an empty directory -- an
    inert control that reported success, which is the worst failure shape available.
    """
    schemas = Schemas(schema_dir, manifest)
    if not schemas.verbs:
        print("SELF-TEST REFUSED: no schemas loaded", file=sys.stderr)
        return EXIT_REFUSED

    cases = [
        ("correct arg name", {"ok": True, "ops": [{"tool": "memory.recall", "args": {"query": "x"}}]}, "PASS"),
        ("the q-vs-query bug", {"ok": True, "ops": [{"tool": "memory.recall", "args": {"q": "x"}}]}, "FAIL"),
        ("explicit-empty verb, no args", {"ok": True, "ops": [{"tool": "stats", "args": {}}]}, "PASS"),
        ("explicit-empty verb, given an arg", {"ok": True, "ops": [{"tool": "stats", "args": {"limit": 1}}]}, "FAIL"),
        ("hallucinated verb", {"ok": True, "ops": [{"tool": "memory.teleport", "args": {"x": 1}}]}, "REFUSED"),
        ("missing required arg", {"ok": True, "ops": [{"tool": "memory.recall", "args": {"limit": 5}}]}, "FAIL"),
        ("parser already rejected", {"ok": False}, "FAIL"),
        ("FAIL dominates REFUSED", {"ok": True, "ops": [
            {"tool": "memory.recall", "args": {"q": "x"}},
            {"tool": "memory.teleport", "args": {}}]}, "FAIL"),
        # Wire-shaped cases: the validator emits {"kind": ..., "value": ...}, and only these
        # exercise type checking. The plain-args cases above leave types `unchecked` by design,
        # so without these the type path would be entirely untested while the suite read green.
        ("wire shape, correct type", {"ok": True, "ops": [
            {"tool": "memory.recall", "args": {"query": {"kind": "value", "value": "x"}}}]}, "PASS"),
        ("wire shape, integer given a string", {"ok": True, "ops": [
            {"tool": "memory.recall", "args": {
                "query": {"kind": "value", "value": "x"},
                "limit": {"kind": "value", "value": "banana"}}}]}, "FAIL"),
        ("prev_ref is unchecked, not failed", {"ok": True, "ops": [
            {"tool": "memory.recall", "args": {
                "query": {"kind": "prev_ref", "path": "$prev.id"}}}]}, "PASS"),
        # Groundedness with NO prompt: must not fail, and must not silently claim to have
        # checked. The verdict here proves only that the axis stays out of the way.
        ("no prompt, axis not evaluated", {"ok": True, "ops": [
            {"tool": "memory.recall", "args": {
                "query": {"kind": "value", "value": "anything at all"}}}]}, "PASS"),
    ]
    bad = 0
    for label, row, want in cases:
        got = score_row(row, schemas)["verdict"]
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:34s} want {want:8s} got {got}")

    # The groundedness axis needs its own block because it takes a second input. The pair
    # below is DISCRIMINATING by construction: identical op, identical schema, one literal
    # changed, opposite verdicts. A fixture where both arms pass would test nothing -- the
    # earlier version of this file's W1 control had exactly that defect.
    real_id = "36d8a874-a02a-5cee-b47b-99e5fd60654e"
    fake_id = "00000000-0000-0000-0000-000000000000"
    prompt = f"Write the request ops for this task: Export stored session {real_id} as markdown."
    pair = [
        ("grounded id + prompt", real_id, "PASS"),
        ("hallucinated id + same prompt", fake_id, "FAIL"),
    ]
    for label, ident, want in pair:
        row = {"ok": True, "ops": [{"tool": "session.export", "args": {
            "id": {"kind": "value", "value": ident},
            "format": {"kind": "value", "value": "markdown"}}}]}
        got = score_row(row, schemas, prompt)["verdict"]
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:34s} want {want:8s} got {got}")

    # And the mutation control on the axis itself: the SAME hallucinated row must PASS when
    # no prompt is supplied, which proves the FAIL above came from groundedness and not
    # from some other axis reacting to the changed literal.
    row = {"ok": True, "ops": [{"tool": "session.export", "args": {
        "id": {"kind": "value", "value": fake_id},
        "format": {"kind": "value", "value": "markdown"}}}]}
    got = score_row(row, schemas)["verdict"]
    ok = got == "PASS"
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  {'same row, axis off -> PASS':34s} want PASS     got {got}")

    # Regression fixture for the representation bug: nested DSL in PARSED form (plain inner
    # quotes). Without the parsed branch this row reads UNGROUNDED and fails, which is how
    # 87 gold rows failed. The second arm keeps it honest -- an inner literal absent from the
    # prompt must still fail, or the branch would be granting blanket amnesty to any op-ish
    # value rather than actually checking the inner literals.
    nested_prompt = ('At 2027-02-01T09:00:00Z, dispatch a call that schedules a reminder for '
                     '2027-03-16T09:00:00Z with content "Check pagination regression tests."')
    for label, content, want in [
        ("nested DSL, parsed form", "Check pagination regression tests.", "PASS"),
        ("nested DSL, invented inner literal", "Delete the production database.", "FAIL"),
    ]:
        action = f'schedule.remind(content="{content}",at="2027-03-16T09:00:00Z")'
        row = {"ok": True, "ops": [{"tool": "schedule.schedule", "args": {
            "action": {"kind": "value", "value": action},
            "at": {"kind": "value", "value": "2027-02-01T09:00:00Z"}}}]}
        got = score_row(row, schemas, nested_prompt)["verdict"]
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {label:34s} want {want:8s} got {got}")

    reached = {score_row(r, schemas)["verdict"] for _, r, _ in cases}
    print(f"\nverdicts reached: {sorted(reached)}")
    if reached != {"PASS", "FAIL", "REFUSED"}:
        print("SELF-TEST FAILED: not all three verdicts are reachable", file=sys.stderr)
        return EXIT_FAIL
    if bad:
        print(f"SELF-TEST FAILED: {bad} case(s)", file=sys.stderr)
        return EXIT_FAIL
    print(f"self-test OK  capture={schemas.fingerprint}  verbs={len(schemas.verbs)}")
    if schemas.drift:
        print(f"WARNING capture drift: {schemas.drift}", file=sys.stderr)
        return EXIT_REFUSED
    return EXIT_PASS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validated", type=Path, help="JSONL emitted by dsl_validator")
    ap.add_argument("--schema-dir", type=Path, required=True,
                    help="directory of captured verb schemas")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="capture manifest; its (name, sha256) pairs fingerprint the capture")
    ap.add_argument("--pairs", type=Path,
                    help="prompt/completion JSONL; enables the groundedness axis. Joined on the\n"
                         "validator's 1-indexed `line` field, never on position.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _self_test(args.schema_dir, args.manifest)
    if not args.validated:
        ap.error("--validated or --self-test is required")

    schemas = Schemas(args.schema_dir, args.manifest)
    if not schemas.verbs:
        print("REFUSED: no schemas loaded; scoring nothing", file=sys.stderr)
        return EXIT_REFUSED
    if schemas.drift:
        print(f"REFUSED: capture drift, scores would not be attributable: {schemas.drift}", file=sys.stderr)
        return EXIT_REFUSED

    prompts: dict[int, str] = {}
    if args.pairs:
        for i, raw in enumerate(args.pairs.read_text().splitlines(), start=1):
            if raw.strip():
                prompts[i] = json.loads(raw)["prompt"]
        if not prompts:
            print("REFUSED: --pairs supplied but empty; an empty join is not a clean score",
                  file=sys.stderr)
            return EXIT_REFUSED

    counts = {"PASS": 0, "FAIL": 0, "REFUSED": 0}
    ground_tally: dict[str, int] = {}
    unjoined = 0
    for raw in args.validated.read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        prompt = None
        if prompts:
            lineno = row.get("line")
            prompt = prompts.get(lineno)
            if prompt is None:
                # A validated row whose line number is not in the pairs file means the two
                # inputs do not describe the same run. Scoring it with the axis silently off
                # would report a PASS that no groundedness check produced.
                unjoined += 1
        scored = score_row(row, schemas, prompt)
        counts[scored["verdict"]] += 1
        for op in scored.get("ops", []):
            g = op.get("groundedness") or {}
            for k, n in (g.get("recovery") or {}).items():
                ground_tally[k] = ground_tally.get(k, 0) + n

    if unjoined:
        print(f"REFUSED: {unjoined} validated row(s) had no matching line in --pairs; "
              "the two inputs are not the same run", file=sys.stderr)
        return EXIT_REFUSED

    total = sum(counts.values())
    if total == 0:
        print("REFUSED: input contained no rows; an empty input is not a clean score", file=sys.stderr)
        return EXIT_REFUSED

    print(f"capture={schemas.fingerprint}  verbs={len(schemas.verbs)}  n={total}")
    for verdict in ("PASS", "FAIL", "REFUSED"):
        print(f"  {verdict:8s} {counts[verdict]:6d}  {counts[verdict]/total:6.1%}")
    if prompts:
        gtot = sum(ground_tally.values())
        print(f"\ngroundedness: evaluated, {gtot} string literal(s)")
        for k in sorted(ground_tally, key=lambda k: -ground_tally[k]):
            print(f"  {k:12s} {ground_tally[k]:6d}  {ground_tally[k]/gtot:6.1%}" if gtot else k)
    else:
        print("\ngroundedness: NOT EVALUATED (no --pairs) — literals were not checked "
              "against any prompt")
    print("\nexecuted: false — no generated command was run")
    if counts["FAIL"]:
        return EXIT_FAIL
    return EXIT_REFUSED if counts["REFUSED"] else EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
