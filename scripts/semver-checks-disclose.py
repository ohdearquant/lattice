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

# "    Checking lattice-embed v0.7.1 -> v0.8.0 (major change)"
# Real cargo-semver-checks output indents this line (verified against a real
# captured run with `od -c`); anchoring on column 0 silently matches nothing.
_CHECKING_RE = re.compile(
    r"^[ \t]*Checking\s+(\S+)\s+v(\S+)\s*->\s*v(\S+)", re.MULTILINE
)

# Real CI output is colourized (verified against a real captured GitHub
# Actions log): "ESC[1mESC[32m    CheckingESC[0m lattice-embed ...", with
# escape codes both before "Checking" and between "Checking" and the package
# name. Neither the whitespace class in _CHECKING_RE nor a column-0 anchor
# can see through that. The capture step also sets CARGO_TERM_COLOR=never,
# but this strip runs regardless — the tool can colour for reasons other
# than that one variable, and a parser that only works when a upstream env
# var was set correctly is not a parser that can be trusted on its own.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


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
    text = _strip_ansi(text)
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


def could_not_read_summary(detail: str) -> str:
    """The disclosure line for 'this instrument could not read its input'.

    Used both when a captured file parses with no 'Checked' line (broken
    capture) and when the file could not even be opened (missing/unreadable
    capture step). Both are instrument failures, never a zero-checks finding
    — the wording stays unmistakably distinct from the zero-checks line so a
    broken instrument can never read as either healthy silence or a real
    disclosure.
    """
    return (
        "**SEMVER DISCLOSURE: could not read check output** — "
        f"{detail} This is NOT a report of zero checks; the disclosure "
        "instrument itself failed and cannot say anything about coverage "
        "this run.\n"
    )


def _observed_scope_clause(observed_package: str | None) -> str:
    """The scope-tripwire sentence: name what was actually checked.

    The capture re-run checks one crate, not the workspace's full set.
    That is sound only because every workspace crate shares one
    `[workspace.package] version` — a bump voids the gate for all of them
    at once through that single shared value, so one crate's real
    `Checked N checks` line answers the question for the rest. If the
    workspace ever splits into per-crate versions, that inference goes
    silently false, and this sentence is the only thing in the disclosure
    that says the instrument's basis has expired — it must stay in the
    disclosure text itself, not a code comment nobody reading the alert
    will see.
    """
    if not observed_package:
        return ""
    return (
        f" Observed `{observed_package}` directly; the other workspace "
        "crates are inferred to be in the same state because they share "
        "one `[workspace.package] version`."
    )


def compose_summary(
    result: ParseResult, observed_package: str | None = None
) -> str | None:
    """Return the $GITHUB_STEP_SUMMARY text, or None to stay silent.

    Silence is the healthy-case default: when checks actually executed,
    this function must return None so the step writes nothing at all.
    """
    if not result.parse_ok:
        return could_not_read_summary(
            "no `Checked [...] N checks: ...` line was found in the "
            "captured cargo-semver-checks run."
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
        f"baseline.{_observed_scope_clause(observed_package)}\n"
    )


def as_workflow_warning(summary: str) -> str:
    """Collapse a disclosure into a single-line `::warning::` workflow command.

    $GITHUB_STEP_SUMMARY is not a readable carrier in practice: on a real run,
    `check_runs[].output.summary` for this job came back null via the GitHub
    API, and the step summary appeared nowhere in the job log either. The
    annotations channel (`::warning::`, surfaced through
    repos/{owner}/{repo}/check-runs/{id}/annotations) DID return content on
    the same run, so this is the channel that actually makes the disclosure
    readable — by a person scanning a green check list, or by an automated
    reader. Workflow commands must be a single line; GitHub also truncates
    and reformats embedded newlines unpredictably, so collapse to whitespace
    up front rather than relying on that behavior.
    """
    return "::warning::" + " ".join(summary.split())


def _run_selftest() -> int:
    # Indentation here matches a real captured `cargo semver-checks
    # check-release` run byte-for-byte (verified with `od -c`): both the
    # "Checking" and "Checked" lines carry leading whitespace, "Checking" by
    # 4 spaces and "Checked" by 1. A fixture built from paraphrased prose
    # instead of the real tool output can drift from that indentation and
    # still pass against a regex with the same wrong assumption baked in.
    healthy = "\n".join(
        f"    Checking lattice-{name} v0.7.1 -> v0.7.1 (no change; assume minor)\n"
        " Checked [   0.023s] 196 checks: 196 pass, 57 skip\n"
        " Summary no semver update required"
        for name in ("fann", "transport", "inference", "embed", "tune")
    )
    zero = "\n".join(
        f"    Checking lattice-{name} v0.7.1 -> v0.8.0 (major change)\n"
        " Checked [   0.000s] 0 checks: 0 pass, 253 skip\n"
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
    parser.add_argument(
        "--observed-package",
        default=None,
        help=(
            "Name of the single crate the capture re-run actually checked "
            "(e.g. lattice-transport). Included in the zero-checks "
            "disclosure so it names its own scope instead of implying "
            "workspace-wide coverage."
        ),
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return _run_selftest()

    if not args.captured_output:
        parser.error("captured_output is required unless --selftest is given")

    # A missing or unreadable capture file must take the SAME could-not-read
    # path as a captured-but-unparseable one, not raise before compose_summary
    # is reached. In the workflow this step carries continue-on-error: true,
    # so an uncaught exception here would make the job go green and
    # completely silent — the one outcome this instrument must never produce.
    try:
        text = Path(args.captured_output).read_text()
    except OSError as exc:
        summary = could_not_read_summary(
            f"reading '{args.captured_output}' raised {exc.__class__.__name__}: {exc}."
        )
        parse_ok = False
    else:
        result = parse_semver_checks_output(text)
        summary = compose_summary(result, observed_package=args.observed_package)
        parse_ok = result.parse_ok

    if summary is None:
        return 0

    # Both channels, always together: the step summary for a human reading
    # the job page, and a `::warning::` workflow command (printed to stdout,
    # which is how GitHub Actions recognizes workflow commands) so the
    # disclosure also lands in the log and the annotations API.
    print(as_workflow_warning(summary))

    summary_out = args.summary_out
    if summary_out is None:
        import os

        summary_out = os.environ.get("GITHUB_STEP_SUMMARY")

    if summary_out:
        with open(summary_out, "a", encoding="utf-8") as fh:
            fh.write(summary)
    else:
        print(summary, file=sys.stderr)

    return 0 if parse_ok else 1


if __name__ == "__main__":
    sys.exit(main())
