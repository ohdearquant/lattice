#!/usr/bin/env python3
"""Disclose when cargo-semver-checks executed zero checks.

Under 0.x semver, a workspace minor bump (0.7.1 -> 0.8.0) reads to
cargo-semver-checks as an assumed-major change, so every check is skipped
and the tool still prints its normal "no semver update required" success
line. A CI gate that only looks at cargo-semver-checks' exit code cannot
tell that pass-with-zero-checks apart from a real pass, so it reports
green while providing no coverage at all.

This script is a pure observer: it never re-implements or overrides the
gate's verdict. It re-parses a captured run of the same check command and
reports, via $GITHUB_STEP_SUMMARY, whether any checks actually executed.

  Usage:
    semver-checks-disclose.py <captured-output-file> [--summary-out PATH]
    semver-checks-disclose.py --selftest

  Exit codes:
    0 — parsed successfully (silent when checks executed, loud when zero)
    1 — could not parse: no "Checked" line found in the captured output.
        This is a broken instrument, not a finding of zero checks, and it
        is reported with its own distinct message so it can never be
        mistaken for either the healthy or the zero-checks case.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# " Checked [ 0.023s] 196 checks: 196 pass, 57 skip"
# The whitespace inside the brackets varies run to run; match on the
# " checks:" literal the task names as the anchor, not on column position.
_CHECKED_RE = re.compile(
    r"Checked\s*\[[^\]]*\]\s*(\d+)\s*checks:\s*(\d+)\s*pass,\s*(\d+)\s*skip"
)

# "Checking lattice-embed v0.7.1 -> v0.8.0 (major change)"
_CHECKING_RE = re.compile(
    r"^Checking\s+(\S+)\s+v(\S+)\s*->\s*v(\S+)", re.MULTILINE
)


@dataclass
class Transition:
    package: str
    version_from: str
    version_to: str


@dataclass
class ParseResult:
    parse_ok: bool
    total_checks: int = 0
    total_pass: int = 0
    total_skip: int = 0
    transitions: list[Transition] = field(default_factory=list)


def parse_semver_checks_output(text: str) -> ParseResult:
    """Sum the executed-check count across every 'Checked' line.

    An input with no 'Checked' line at all is a broken instrument (the
    output does not look like cargo-semver-checks output), not evidence of
    zero checks — callers must treat parse_ok=False differently from
    total_checks == 0.
    """
    checked_matches = _CHECKED_RE.findall(text)
    if not checked_matches:
        return ParseResult(parse_ok=False)

    total_checks = sum(int(m[0]) for m in checked_matches)
    total_pass = sum(int(m[1]) for m in checked_matches)
    total_skip = sum(int(m[2]) for m in checked_matches)
    transitions = [
        Transition(package=pkg, version_from=frm, version_to=to)
        for pkg, frm, to in _CHECKING_RE.findall(text)
    ]
    return ParseResult(
        parse_ok=True,
        total_checks=total_checks,
        total_pass=total_pass,
        total_skip=total_skip,
        transitions=transitions,
    )


def _describe_transitions(transitions: list[Transition]) -> str:
    if not transitions:
        return "version transition unavailable (no 'Checking <pkg> v.. -> v..' line found)"
    distinct = {(t.version_from, t.version_to) for t in transitions}
    if len(distinct) == 1:
        frm, to = next(iter(distinct))
        return f"{frm} -> {to}"
    return ", ".join(f"{t.package} {t.version_from} -> {t.version_to}" for t in transitions)


def compose_summary(result: ParseResult) -> str | None:
    """Return the $GITHUB_STEP_SUMMARY text, or None to stay silent.

    Silence is the healthy-case default: when checks actually executed,
    this function must return None so the step writes nothing at all.
    """
    if not result.parse_ok:
        return (
            "**SEMVER DISCLOSURE: could not read check output** — no "
            "`Checked [...] N checks: ...` line was found in the captured "
            "cargo-semver-checks run. This is NOT a report of zero checks; "
            "the disclosure instrument itself failed to parse the output "
            "and cannot say anything about coverage this run.\n"
        )

    if result.total_checks > 0:
        return None

    transition = _describe_transitions(result.transitions)
    return (
        "**SEMVER: 0 checks executed** "
        f"({transition}; a 0.x minor bump reads as an assumed-major change, "
        f"so cargo-semver-checks skipped all {result.total_skip} checks). "
        "A green result here is NOT coverage. It becomes coverage again "
        "once the bumped version is published and becomes the crates.io "
        "baseline.\n"
    )


def _run_selftest() -> int:
    healthy = "\n".join(
        f"Checking lattice-{name} v0.7.1 -> v0.7.1 (no change; assume minor)\n"
        " Checked [ 0.023s] 196 checks: 196 pass, 57 skip\n"
        " Summary no semver update required"
        for name in ("fann", "transport", "inference", "embed", "tune")
    )
    zero = "\n".join(
        f"Checking lattice-{name} v0.7.1 -> v0.8.0 (major change)\n"
        " Checked [ 0.000s] 0 checks: 0 pass, 253 skip\n"
        " Summary no semver update required"
        for name in ("fann", "transport", "inference", "embed", "tune")
    )
    unparseable = "error: could not locate baseline rustdoc JSON\n"

    checks = [
        ("healthy", healthy, None),
        ("zero", zero, "**SEMVER: 0 checks executed**"),
        ("unparseable", unparseable, "**SEMVER DISCLOSURE: could not read check output**"),
    ]
    all_pass = True
    for name, text, expect_prefix in checks:
        summary = compose_summary(parse_semver_checks_output(text))
        if expect_prefix is None:
            ok = summary is None
        else:
            ok = summary is not None and summary.startswith(expect_prefix)
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
        all_pass = all_pass and ok
    return 0 if all_pass else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "captured_output",
        nargs="?",
        help="Path to a file holding the captured cargo-semver-checks output",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Path to append the disclosure line to (default: $GITHUB_STEP_SUMMARY)",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run the three built-in fixtures and exit",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return _run_selftest()

    if not args.captured_output:
        parser.error("captured_output is required unless --selftest is given")

    text = Path(args.captured_output).read_text()
    result = parse_semver_checks_output(text)
    summary = compose_summary(result)

    if summary is None:
        return 0

    summary_out = args.summary_out
    if summary_out is None:
        import os

        summary_out = os.environ.get("GITHUB_STEP_SUMMARY")

    if summary_out:
        with open(summary_out, "a", encoding="utf-8") as fh:
            fh.write(summary)
    else:
        print(summary, file=sys.stderr)

    return 0 if result.parse_ok else 1


if __name__ == "__main__":
    sys.exit(main())
