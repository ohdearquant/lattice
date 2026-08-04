#!/usr/bin/env python3
"""Contract tests for the semver-checks zero-check disclosure instrument."""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "semver-checks-disclose.py"
CI_WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
FIXTURES = REPO / "tests" / "fixtures" / "semver_checks"
REAL_HEALTHY = FIXTURES / "real-arm-a-healthy.txt"
REAL_ZERO = FIXTURES / "real-arm-b-zero.txt"
REAL_ANSI_ZERO = FIXTURES / "ci-log-ansi-zero.txt"

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


def _run(
    text: str, root: Path, *, observed_package: str | None = None
) -> tuple[subprocess.CompletedProcess, Path]:
    captured = root / "captured.txt"
    captured.write_text(text)
    return _run_path(captured, root, observed_package=observed_package)


def _run_path(
    captured: Path, root: Path, *, observed_package: str | None = None
) -> tuple[subprocess.CompletedProcess, Path]:
    summary = root / "summary.md"
    argv = ["python3", str(SCRIPT), str(captured), "--summary-out", str(summary)]
    if observed_package:
        argv += ["--observed-package", observed_package]
    result = subprocess.run(argv, text=True, capture_output=True)
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

    def test_zero_disclosure_names_its_own_observed_scope(self) -> None:
        # ci.yml's capture re-run now checks one crate (lattice-transport),
        # not all five, because every workspace crate shares one
        # `[workspace.package] version` — a bump voids the gate for all of
        # them at once through that shared value. The disclosure text is the
        # only place that inference is written down, so it must name both
        # the crate actually observed and the inference itself in its own
        # words; a future edit that drops this clause would make the
        # disclosure imply workspace-wide coverage it never measured.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, summary = _run(
                _zero_fixture(), root, observed_package="lattice-transport"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(summary.exists())
            text = summary.read_text()
            self.assertIn("lattice-transport", text)
            self.assertIn("inferred", text)
            self.assertIn("[workspace.package] version", text)

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

    def test_real_ansi_colored_capture_still_discloses_the_version_transition(
        self,
    ) -> None:
        # Real GitHub Actions log output is colourized: escape codes sit both
        # before "Checking" and between "Checking" and the package name
        # (verified against a real captured CI log). A parser that only
        # tolerates whitespace, not escape codes, silently drops the version
        # transition here instead of raising an error.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, summary = _run_path(REAL_ANSI_ZERO, root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(summary.exists())
            text = summary.read_text()
            self.assertIn("SEMVER: 0 checks executed", text)
            self.assertIn("0.7.1 -> 0.8.0", text)
            self.assertNotIn("version transition unavailable", text)
            # The captured escape bytes must not leak into the disclosure —
            # a "clean" extraction that just happens to still contain \x1b
            # would corrupt the step summary and the workflow annotation.
            self.assertNotIn("\x1b", text)

    def test_zero_disclosure_also_emits_a_workflow_warning(self) -> None:
        # $GITHUB_STEP_SUMMARY is not a readable carrier in practice (measured
        # on a real run: check_runs[].output.summary came back null via the
        # GitHub API, and the step summary appeared nowhere in the job log).
        # The annotations channel, driven by `::warning::` on stdout, DID
        # return content on that same run — so the disclosure must land there
        # too, not only in the file this script also writes.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, _summary = _run(_zero_fixture(), root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("::warning::", result.stdout)
            self.assertIn("SEMVER: 0 checks executed", result.stdout)
            self.assertIn("0.7.1 -> 0.8.0", result.stdout)
            # A workflow command must be one line: GitHub does not parse a
            # `::warning::` spanning multiple lines as a single annotation.
            warning_lines = [
                line for line in result.stdout.splitlines() if "::warning::" in line
            ]
            self.assertEqual(len(warning_lines), 1)

    def test_could_not_read_also_emits_a_workflow_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, _summary = _run(_unparseable_fixture(), root)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("::warning::", result.stdout)
            self.assertIn("could not read check output", result.stdout)

    def test_healthy_run_emits_no_workflow_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, _summary = _run(_healthy_fixture(), root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("::warning::", result.stdout)

    def test_capture_step_is_scoped_to_one_crate(self) -> None:
        # The gate step above (untouched by this scoping change) still checks
        # all five packages; only the observer's own re-run narrows to one.
        # A version bump voids the gate for every crate at once through the
        # single shared `[workspace.package] version`, so one crate's real
        # `Checked N checks` line answers the workspace-wide question.
        with open(CI_WORKFLOW) as fh:
            workflow = yaml.safe_load(fh)
        steps = workflow["jobs"]["semver-checks"]["steps"]
        capture_steps = [
            step
            for step in steps
            if step.get("name") == "Re-run semver-checks to capture executed-check counts"
        ]
        self.assertEqual(len(capture_steps), 1)
        run = capture_steps[0]["run"]
        self.assertEqual(run.count("-p lattice-"), 1, run)
        self.assertIn("-p lattice-transport", run)

    def test_disclosure_step_passes_the_observed_package_it_checked(self) -> None:
        with open(CI_WORKFLOW) as fh:
            workflow = yaml.safe_load(fh)
        steps = workflow["jobs"]["semver-checks"]["steps"]
        disclose_steps = [
            step for step in steps if step.get("name") == "Disclose zero-check semver-checks runs"
        ]
        self.assertEqual(len(disclose_steps), 1)
        self.assertIn(
            "--observed-package lattice-transport", disclose_steps[0]["run"]
        )

    def test_capture_step_disables_ansi_color(self) -> None:
        # Prevention, not the guarantee (the parser's own ANSI strip is the
        # guarantee, covered by test_real_ansi_colored_capture_still_
        # discloses_the_version_transition above) — but real CI output is
        # colourized by default (workflow-level CARGO_TERM_COLOR: always at
        # the top of this file), so the capture step must override that
        # locally rather than relying on the tool's own auto-detection.
        with open(CI_WORKFLOW) as fh:
            workflow = yaml.safe_load(fh)
        steps = workflow["jobs"]["semver-checks"]["steps"]
        capture_steps = [
            step
            for step in steps
            if step.get("name") == "Re-run semver-checks to capture executed-check counts"
        ]
        self.assertEqual(
            len(capture_steps), 1, "expected exactly one capture step in ci.yml"
        )
        self.assertEqual(
            capture_steps[0].get("env", {}).get("CARGO_TERM_COLOR"), "never"
        )

    def test_disclosure_steps_use_bare_python3_matching_repo_convention(self) -> None:
        # Every existing Python step in ci.yml is bare `python3` (e.g.
        # decode-harness-unit-tests below invokes `python3 tests/...`); this
        # workflow never uses `uv` anywhere else. `uv run python3 ...` is
        # command-not-found (exit 127) on a runner that never installed uv —
        # exactly the failure a real run of this job measured. Assert the
        # convention directly rather than only through the script-cannot-run
        # simulation below, since a sandbox that happens to have uv installed
        # locally can mask this defect by finding a real interpreter anyway.
        with open(CI_WORKFLOW) as fh:
            workflow = yaml.safe_load(fh)
        steps = workflow["jobs"]["semver-checks"]["steps"]
        for step in steps:
            run = step.get("run", "")
            if "semver-checks-disclose.py" not in run:
                continue
            self.assertNotIn(
                "uv run", run, f"step {step.get('name')!r} must use bare python3"
            )
            self.assertIn("python3 scripts/semver-checks-disclose.py", run)

    def test_disclosure_step_still_warns_when_the_script_cannot_run(self) -> None:
        # Extracted from the live workflow rather than restated here: a copy
        # of the shell logic proves only that the copy behaves, and the two
        # drift silently. This grades the actual step ci.yml will run.
        #
        # continue-on-error on this step hides a nonzero step conclusion from
        # the job, so a step that dies invisibly (command-not-found, a
        # missing script, a bad interpreter) must announce that itself. This
        # simulates exactly that: `python3` on PATH always exits 127, standing
        # in for "the interpreter or script never ran at all" — the failure
        # measured on a real run, where the job stayed green with no
        # annotation at all.
        with open(CI_WORKFLOW) as fh:
            workflow = yaml.safe_load(fh)
        steps = workflow["jobs"]["semver-checks"]["steps"]
        disclose_steps = [
            step for step in steps if step.get("name") == "Disclose zero-check semver-checks runs"
        ]
        self.assertEqual(
            len(disclose_steps), 1, "expected exactly one disclosure step in ci.yml"
        )
        script = disclose_steps[0]["run"]
        self.assertIn("::warning::", script)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python3 = fake_bin / "python3"
            fake_python3.write_text("#!/bin/sh\nexit 127\n")
            fake_python3.chmod(fake_python3.stat().st_mode | stat.S_IEXEC)

            step_script = root / "step.sh"
            step_script.write_text(script)

            env = dict(os.environ)
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            env["RUNNER_TEMP"] = str(root)
            result = subprocess.run(
                ["bash", str(step_script)],
                text=True,
                capture_output=True,
                env=env,
                cwd=REPO,
            )
            self.assertIn(
                "::warning::",
                result.stdout,
                f"a script that never ran must still self-report; stdout={result.stdout!r} stderr={result.stderr!r}",
            )
            self.assertIn("did not run to completion", result.stdout)


if __name__ == "__main__":
    unittest.main()
