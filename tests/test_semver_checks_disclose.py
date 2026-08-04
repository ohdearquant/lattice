#!/usr/bin/env python3
"""Contract tests for the semver-checks zero-check disclosure instrument."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "semver-checks-disclose.py"
FIXTURES = REPO / "tests" / "fixtures" / "semver_checks"
REAL_HEALTHY = FIXTURES / "real-arm-a-healthy.txt"
REAL_ZERO = FIXTURES / "real-arm-b-zero.txt"

_CRATES = ("fann", "transport", "inference", "embed", "tune")


def _healthy_fixture() -> str:
    # Indentation matches a real captured `cargo semver-checks check-release`
    # run byte-for-byte (see the fixture files under tests/fixtures/
    # semver_checks/, verified with `od -c`): "Checking" is indented 4
    # spaces, "Checked" 1. A fixture built at column 0 shares the same wrong
    # assumption as a regex anchored at column 0 and cannot catch it.
    return "\n".join(
        f"    Checking lattice-{name} v0.7.1 -> v0.7.1 (no change; assume minor)\n"
        " Checked [   0.023s] 196 checks: 196 pass, 57 skip\n"
        " Summary no semver update required"
        for name in _CRATES
    )


def _zero_fixture() -> str:
    return "\n".join(
        f"    Checking lattice-{name} v0.7.1 -> v0.8.0 (major change)\n"
        " Checked [   0.000s] 0 checks: 0 pass, 253 skip\n"
        " Summary no semver update required"
        for name in _CRATES
    )


def _unparseable_fixture() -> str:
    return "error: could not locate baseline rustdoc JSON for lattice-fann\n"


def _run(text: str, root: Path) -> tuple[subprocess.CompletedProcess, Path]:
    captured = root / "captured.txt"
    captured.write_text(text)
    return _run_path(captured, root)


def _run_path(captured: Path, root: Path) -> tuple[subprocess.CompletedProcess, Path]:
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
            "    Checking lattice-fann v0.7.1 -> v0.8.0 (major change)\n"
            " Checked [   0.010s] 3 checks: 3 pass, 250 skip\n"
            " Summary no semver update required"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, summary = _run(text, root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(summary.exists() and summary.read_text())

    def test_real_healthy_capture_is_silent(self) -> None:
        # A real captured 0.7.1 -> 0.7.1 no-change run (196 checks executed
        # per crate). Guards the indentation regression directly: this file
        # is untouched tool output, not a hand-authored fixture string.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, summary = _run_path(REAL_HEALTHY, root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(
                summary.exists() and summary.read_text(),
                "a real run that executed checks must stay silent",
            )

    def test_real_zero_capture_discloses_the_version_transition(self) -> None:
        # A real captured 0.7.1 -> 0.8.0 assumed-major run (0 checks executed
        # per crate). The version transition must be recovered from the
        # real, indented "Checking" lines, not fall back to "unavailable".
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, summary = _run_path(REAL_ZERO, root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(summary.exists())
            text = summary.read_text()
            self.assertIn("SEMVER: 0 checks executed", text)
            self.assertIn("0.7.1 -> 0.8.0", text)
            self.assertNotIn("version transition unavailable", text)

    def test_missing_capture_file_is_could_not_read_not_silent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "does-not-exist.txt"
            result, summary = _run_path(missing, root)
            # Never a clean silent exit: continue-on-error on the workflow
            # step already makes the job green regardless of this script's
            # exit code, so a missing capture must not ALSO make the summary
            # silent — that combination is exactly the failure this
            # instrument exists to prevent.
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue(summary.exists())
            text = summary.read_text()
            self.assertIn("could not read check output", text)
            self.assertNotIn("0 checks executed", text)

    def test_empty_capture_file_is_could_not_read_not_a_healthy_silent_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "empty.txt"
            empty.write_text("")
            result, summary = _run_path(empty, root)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertTrue(
                summary.exists() and summary.read_text(),
                "zero bytes contains no 'Checked' line and must not read as "
                "a healthy silent run",
            )
            text = summary.read_text()
            self.assertIn("could not read check output", text)
            self.assertNotIn("0 checks executed", text)


if __name__ == "__main__":
    unittest.main()
