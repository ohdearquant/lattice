#!/usr/bin/env python3
"""Contract tests for the semver-checks zero-check disclosure instrument."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "semver-checks-disclose.py"

_CRATES = ("fann", "transport", "inference", "embed", "tune")


def _healthy_fixture() -> str:
    return "\n".join(
        f"Checking lattice-{name} v0.7.1 -> v0.7.1 (no change; assume minor)\n"
        " Checked [ 0.023s] 196 checks: 196 pass, 57 skip\n"
        " Summary no semver update required"
        for name in _CRATES
    )


def _zero_fixture() -> str:
    return "\n".join(
        f"Checking lattice-{name} v0.7.1 -> v0.8.0 (major change)\n"
        " Checked [ 0.000s] 0 checks: 0 pass, 253 skip\n"
        " Summary no semver update required"
        for name in _CRATES
    )


def _unparseable_fixture() -> str:
    return "error: could not locate baseline rustdoc JSON for lattice-fann\n"


def _run(text: str, root: Path) -> tuple[subprocess.CompletedProcess, Path]:
    captured = root / "captured.txt"
    captured.write_text(text)
    summary = root / "summary.md"
    result = subprocess.run(
        ["python3", str(SCRIPT), str(captured), "--summary-out", str(summary)],
        text=True,
        capture_output=True,
    )
    return result, summary


class SemverChecksDiscloseTests(unittest.TestCase):
    def test_selftest(self) -> None:
        result = subprocess.run(
            ["python3", str(SCRIPT), "--selftest"],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS: healthy", result.stdout)
        self.assertIn("PASS: zero", result.stdout)
        self.assertIn("PASS: unparseable", result.stdout)

    def test_healthy_run_is_silent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, summary = _run(_healthy_fixture(), root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(
                summary.exists() and summary.read_text(),
                "healthy run must write nothing to the summary",
            )

    def test_zero_checks_writes_loud_disclosure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, summary = _run(_zero_fixture(), root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(summary.exists())
            text = summary.read_text()
            self.assertIn("SEMVER: 0 checks executed", text)
            self.assertIn("0.7.1 -> 0.8.0", text)
            # 253 skip appears on every one of the five Checked lines above;
            # the disclosure must report the SUMMED skip count (5 * 253),
            # not one crate's line, or it would understate what was skipped.
            self.assertIn(str(253 * len(_CRATES)), text)

    def test_unparseable_output_is_neither_silent_nor_a_zero_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, summary = _run(_unparseable_fixture(), root)
            # Non-zero: a broken instrument must not exit clean, but must
            # not be able to fail the enclosing job either (that's the
            # workflow step's job via continue-on-error, not this script's).
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue(summary.exists())
            text = summary.read_text()
            self.assertIn("could not read check output", text)
            self.assertNotIn("0 checks executed", text)

    def test_partial_execution_stays_silent(self) -> None:
        # Sanity: any nonzero executed-check total is healthy, regardless of
        # how many others were skipped alongside it.
        text = (
            "Checking lattice-fann v0.7.1 -> v0.8.0 (major change)\n"
            " Checked [ 0.010s] 3 checks: 3 pass, 250 skip\n"
            " Summary no semver update required"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, summary = _run(text, root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(summary.exists() and summary.read_text())


if __name__ == "__main__":
    unittest.main()
