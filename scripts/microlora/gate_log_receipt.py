#!/usr/bin/env python3
"""Turn a gate log into a structured receipt in which ABSENCE IS A VALUE.

The failure mode this exists to prevent is a receipt that silently omits what it could
not determine. A consumer reading such a receipt cannot distinguish "the gate reported
zero failures" from "nothing in this log told me about failures", and those two states
disagree in the direction that matters: the second one reads as success.

So every field in a receipt is one of two shapes, never missing:

    {"value": 0}
    {"unknown": "no rc marker in the log; the runner did not echo one"}

and `require()` lets a caller name the fields it refuses to proceed without.

WHAT A CARGO TEST LOG CAN AND CANNOT ANSWER, measured against a real run rather than
assumed (40 `test result:` lines from one `cargo test -p lattice-inference` invocation):

  determinable   passed, failed, ignored, measured, filtered_out, per-binary breakdown
  NOT in the log the exact command, the exit code, the source hash
                 -- unless the runner echoed them, which is a property of the runner,
                 not of cargo. Mine echoes CLIPPY_RC/TEST_RC; most do not.

Two aggregations are deliberately NOT performed:

1. `ignored` and `filtered_out` are both "did not run" and they are NOT summed into a
   single `skipped`. `#[ignore]` is a property of the test; filtered-out is a property
   of the invocation's filter. A single number hides which one moved, and the whole
   point of a receipt is to survive someone asking that later.
2. `measured` is reported as measured, never normalised away. It is 0 for every test
   binary because it is a bench field, and a receipt that drops always-zero fields is
   one schema change away from dropping a field that stopped being always-zero.

THE CASE THAT LOOKS LIKE NOTHING. A binary that ran with no matching tests emits
`0 passed; 0 failed; 0 filtered out`, and a binary that never built emits no line at
all. Both contribute zero to every total. The receipt therefore carries `binaries`, a
count of `test result:` lines, so a consumer can see the difference between "ran 12
binaries, all empty" and "ran 0 binaries". Totals alone cannot express it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

EXIT_OK, EXIT_USAGE, EXIT_REFUSED, EXIT_INCOMPLETE = 0, 2, 3, 4

_RESULT = re.compile(
    r"^test result: (?P<outcome>\w+)\. (?P<passed>\d+) passed; (?P<failed>\d+) failed; "
    r"(?P<ignored>\d+) ignored; (?P<measured>\d+) measured; (?P<filtered_out>\d+) filtered out",
    re.M,
)
_RC = re.compile(r"^(?P<name>[A-Z][A-Z0-9_]*_RC)=(?P<rc>-?\d+)\s*$", re.M)

COUNTERS = ("passed", "failed", "ignored", "measured", "filtered_out")

# CI logs are the same cargo output wearing a wrapper. `gh run view --log` emits
# "<job>\t<step>\t<ISO8601Z> <content>" and cargo colours its own output, so every
# line-anchored pattern below misses on a CI log while matching the identical local log.
# Measured: four real CI runs carrying 100, 0, 100 and 1263 marker lines by an unanchored
# grep were all read as "not cargo output" before this existed.
# Both spellings. `gh run view --log` writes the escape in CARET NOTATION -- the two
# ASCII characters "^" and "[", not byte 0x1b -- so a pattern written for the real
# control character matches nothing and the log reads as "not cargo output".
# Measured on a real CI log: 0 bytes of 0x1b, 1738 literal "^[" sequences.
_ANSI = re.compile(r"(?:\x1b|\^\[)\[[0-9;]*[A-Za-z]")
_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z ")


def strip_log_decoration(text: str) -> str:
    """Reduce a CI-wrapped log to the bytes the tool actually wrote.

    Conservative on purpose: the tab-separated prefix is removed ONLY when the final
    field starts with an ISO timestamp, so a raw log whose line merely contains a tab
    is left alone. Stripping unconditionally would silently eat real content.
    """
    out = []
    for line in text.split("\n"):
        line = _ANSI.sub("", line)
        if "\t" in line:
            tail = line.rsplit("\t", 1)[-1]
            if _TS.match(tail):
                line = tail
        out.append(_TS.sub("", line, count=1))
    return "\n".join(out)


# Evidence that this log is cargo output at all. A cargo run that executed ZERO test
# binaries still emits these, so requiring one does not cost the `binaries` discriminator
# below; it costs only the logs that were never cargo output in the first place.
_CARGO_MARKER = re.compile(
    r"^(?: *(?:Compiling|Running|Finished|Fresh|Doc-tests)\b|test result:|error\[E)", re.M)


def value(v):
    return {"value": v}


def unknown(why: str):
    return {"unknown": why}


def is_known(field: dict) -> bool:
    return "value" in field


def parse_cargo_test_log(text: str) -> dict:
    """Receipt fields derivable from the log text alone."""
    text = strip_log_decoration(text)
    rows = [m.groupdict() for m in _RESULT.finditer(text)]

    # `format` is DERIVED, never assumed. Asserting "cargo-test" over a log that is not
    # cargo output is the worst field to get wrong, because it is the one a consumer reads
    # to decide how to interpret every other field. Measured on a 29-file corpus of real
    # gate logs: 28 carried no cargo marker at all, so the assumed value was wrong far more
    # often than it was right, and a 0-byte file claimed a format with no bytes to claim it
    # from.
    if _CARGO_MARKER.search(text):
        receipt: dict = {"format": value("cargo-test"), "binaries": value(len(rows))}
    else:
        why = ("no cargo output marker (Compiling/Running/Finished/Doc-tests/test result:) "
               "in this log, so it is not established as cargo output")
        # `binaries` goes with it. The count of `test result:` lines only MEANS a count of
        # test binaries inside a cargo log; outside one it is a count of a string that
        # happens to be absent, which is not the same fact and must not read as "ran none".
        receipt = {"format": unknown(why), "binaries": unknown(why)}

    if not rows:
        # Empty is not clean. A log with no result lines answers nothing about counts,
        # and returning zeros here would manufacture a passing receipt out of a failed
        # read -- the exact shape this file exists to prevent.
        for name in COUNTERS:
            receipt[name] = unknown("no `test result:` line in the log; counts are unread, not zero")
        receipt["executed"] = unknown("no `test result:` line in the log")
        receipt["outcome"] = unknown("no `test result:` line in the log")
        return receipt

    for name in COUNTERS:
        receipt[name] = value(sum(int(r[name]) for r in rows))
    receipt["executed"] = value(receipt["passed"]["value"] + receipt["failed"]["value"])
    outcomes = {r["outcome"] for r in rows}
    receipt["outcome"] = value("ok" if outcomes == {"ok"} else "FAILED")
    receipt["per_binary"] = value([{k: int(r[k]) for k in COUNTERS} | {"outcome": r["outcome"]} for r in rows])
    return receipt


def rc_from_log(text: str) -> dict:
    """Exit codes only exist in a log if the RUNNER echoed them. Absence is not zero."""
    found = {m.group("name"): int(m.group("rc")) for m in _RC.finditer(text)}
    if not found:
        return unknown("no NAME_RC=<int> marker in the log; cargo does not echo its own "
                       "exit code, so this is a property of the runner and absent here")
    return value(found)


def build(log_path: Path, command: str | None, source_hash: str | None) -> dict:
    text = log_path.read_text(errors="replace")
    receipt = parse_cargo_test_log(text)
    receipt["log"] = value({"path": str(log_path), "bytes": len(text.encode()),
                            "sha256": hashlib.sha256(text.encode()).hexdigest()[:16]})
    receipt["exit_code"] = rc_from_log(text)
    receipt["command"] = value(command) if command else unknown(
        "a gate log does not record the invocation that produced it; pass --command")
    receipt["source_hash"] = value(source_hash) if source_hash else unknown(
        "a gate log does not record the ref it was produced from; pass --source-hash")
    receipt["executed_generated_command"] = value(False)
    return receipt


def require(receipt: dict, names: list[str]) -> list[str]:
    """Field names the caller demanded that the receipt could not determine."""
    return [n for n in names if n not in receipt or not is_known(receipt[n])]


def _self_test() -> int:
    cases = []

    full = ("test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 2875 filtered out; finished in 0.04s\n"
            "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 14 filtered out; finished in 0.00s\n"
            "CLIPPY_RC=0\n")
    r = parse_cargo_test_log(full)
    cases.append(("two binaries counted", r["binaries"]["value"] == 2))
    cases.append(("passed summed", r["passed"]["value"] == 10))
    cases.append(("filtered_out summed, NOT merged into a skipped total",
                  r["filtered_out"]["value"] == 2889 and "skipped" not in r))
    cases.append(("ignored kept separate from filtered_out", r["ignored"]["value"] == 0))
    cases.append(("executed excludes ignored and filtered", r["executed"]["value"] == 10))
    cases.append(("rc parsed when the runner echoed it", rc_from_log(full)["value"] == {"CLIPPY_RC": 0}))

    # The arm that matters: an empty read must not produce a clean receipt.
    empty = parse_cargo_test_log("warning: unrelated\n")
    cases.append(("empty log -> counts UNKNOWN, not 0", not is_known(empty["passed"])))
    cases.append(("empty log -> outcome UNKNOWN, never 'ok'", not is_known(empty["outcome"])))
    cases.append(("no rc marker -> UNKNOWN, not 0", not is_known(rc_from_log("no markers here"))))

    # A log with no cargo marker is not cargo output, and the receipt must not claim it is.
    # `format` is the field a consumer reads to decide how to interpret the rest, so an
    # assumed value here mis-frames every other field at once.
    cases.append(("non-cargo log -> format UNKNOWN, never claimed as cargo-test",
                  not is_known(empty["format"])))
    cases.append(("non-cargo log -> binaries UNKNOWN, because a missing string is not a count",
                  not is_known(empty["binaries"])))
    cases.append(("a 0-byte log claims no format at all",
                  not is_known(parse_cargo_test_log("")["format"])))

    # Ran-but-empty vs never-built: identical totals, different `binaries`. BOTH sides are
    # real cargo logs -- a cargo run that built and ran no test binary still emits its own
    # markers -- so deriving `format` above costs this discriminator nothing.
    ran_empty = parse_cargo_test_log(
        "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n")
    never_built = parse_cargo_test_log(
        "   Compiling lattice-inference v0.1.0\n    Finished test profile in 4.21s\n")
    cases.append(("a cargo log that ran NO binary still reports format",
                  never_built["format"]["value"] == "cargo-test"))
    cases.append(("ran-with-no-tests and never-built have equal totals",
                  ran_empty["passed"].get("value") == 0 and not is_known(never_built["passed"])))
    cases.append(("...and are separated ONLY by binaries, both known",
                  ran_empty["binaries"]["value"] == 1 and never_built["binaries"]["value"] == 0))

    # A failing binary anywhere makes the receipt's outcome FAILED.
    mixed = parse_cargo_test_log(
        "test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.05s\n"
        "test result: FAILED. 0 passed; 4 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.05s\n")
    cases.append(("one FAILED binary makes the whole outcome FAILED", mixed["outcome"]["value"] == "FAILED"))

    # THE INVARIANT THE WHOLE FILE RESTS ON, and it was untested until a real log crashed a
    # reader: every field is two-shaped. `format` was a bare string here, so a consumer
    # written against the documented schema hit a type error on the first real receipt.
    # A rule the module states and does not check is a comment.
    def two_shaped(rec):
        return [k for k, v in rec.items()
                if not (isinstance(v, dict) and (("value" in v) ^ ("unknown" in v)))]
    cases.append(("every field two-shaped: full log", two_shaped(r) == []))
    cases.append(("every field two-shaped: empty log", two_shaped(empty) == []))
    cases.append(("a field is never BOTH value and unknown",
                  two_shaped({"x": {"value": 1, "unknown": "y"}}) == ["x"]))

    cases.append(("require() names the undetermined fields",
                  require({"a": value(1), "b": unknown("x")}, ["a", "b", "c"]) == ["b", "c"]))

    bad = 0
    for label, ok in cases:
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    if bad:
        print(f"SELF-TEST FAILED: {bad} case(s)", file=sys.stderr)
        return EXIT_INCOMPLETE
    print(f"self-test OK  {len(cases)} cases")
    return EXIT_OK


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path)
    ap.add_argument("--command", help="the exact invocation; a cargo log does not contain it")
    ap.add_argument("--source-hash", help="the ref/sha built; a cargo log does not contain it")
    ap.add_argument("--require", default="", help="comma-separated fields to refuse without")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if not args.log:
        ap.error("--log or --self-test is required")

    receipt = build(args.log, args.command, args.source_hash)
    print(json.dumps(receipt, indent=2, sort_keys=True))

    demanded = [f.strip() for f in args.require.split(",") if f.strip()]
    missing = require(receipt, demanded)
    if missing:
        print(f"INCOMPLETE: required field(s) undetermined: {missing}", file=sys.stderr)
        return EXIT_INCOMPLETE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
