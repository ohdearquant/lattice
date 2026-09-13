"""Contract tests for the post-merge performance workflow trigger."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "perf-postmerge-gate.yml"
_BENCH_IMPL = _ROOT / "scripts" / "lib" / "bench-compare-impl.sh"
_BENCH_BINARY_INPUTS = {
    "crates/inference/src/forward/cpu/**",
    "crates/inference/src/attention/**",
    "crates/inference/benches/elementwise_cpu_bench.rs",
    "crates/inference/Cargo.toml",
    "crates/embed/src/simd/**",
    "crates/embed/benches/simd.rs",
    "crates/embed/Cargo.toml",
    "Cargo.lock",
    "Cargo.toml",
    ".cargo/**",
}
_PATH_ENTRY = re.compile(r"^      - '([^']+)'$")
_ACTIVE_PUSH = "  push:"
_PAUSED_PUSH = "#   push:"
_PHASE_WARNING = "::warning title=In-phase foreign CPU load::"


def _classify_script() -> str:
    step = _WORKFLOW.read_text().split(
        "      - name: Classify the gate outcome\n", 1
    )[1].split("\n      - name: ", 1)[0]
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


def _classify(statuses: dict, rc: str):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name, status in statuses.items():
            (root / name).write_text(
                status if isinstance(status, str) else json.dumps(status)
            )
        output = root / "github-output"
        run = subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _classify_script()],
            env={**os.environ, "AB_RC": rc, "STATUS_DIR": str(root),
                 "GITHUB_OUTPUT": str(output)},
            capture_output=True, text=True, timeout=10,
        )
        return run, output.read_text() if output.exists() else ""


def _status(rc: int = 0, loud_arms=()) -> dict:
    return {
        "schema": "perf-bench-gate-status/v1",
        "verdict": {0: "pass", 1: "regression", 2: "error", 3: "not_measurable"}[rc],
        "exit_code": rc,
        "reason": "no complete comparisons" if rc == 2 else "order-bias envelope too large",
        "ambient": {"floor_pct": 70.0, "samples": {"before": 100.0, "between": 100.0, "after": 100.0}},
        "phase_load_verdict": "LOUD" if loud_arms else "ok",
        "phase_load": [
            {"arm": arm, "verdict": "LOUD" if arm in loud_arms else "ok",
             "foreign_max_pct": 45.0 if arm in loud_arms else 0.0, "floor_pct": 70.0}
            for arm in ("base1", "head1", "head2", "base2")
        ],
    }


def _uncomment(line: str) -> str:
    if line.startswith("# "):
        return line[2:]
    if line == "#":
        return ""
    return line


def _trigger_state() -> str:
    """Return 'active' or 'paused', failing if the file is in neither or both.

    The lane can be paused by commenting out its push trigger, leaving
    workflow_dispatch as the only entry point. In that state the paths list is
    retained verbatim as comments so restoring the trigger is a revert rather
    than a re-derivation of coverage, and the path contract below keeps pinning
    the retained list. A file carrying both forms, or neither, is ambiguous
    about which one governs, so it is rejected rather than guessed at.
    """
    lines = _WORKFLOW.read_text(encoding="utf-8").splitlines()
    active = _ACTIVE_PUSH in lines
    paused = _PAUSED_PUSH in lines
    if active and paused:
        raise AssertionError("workflow carries both a live and a commented push trigger")
    if not active and not paused:
        raise AssertionError("post-merge workflow push paths block is missing")
    return "active" if active else "paused"


def _push_paths() -> set[str]:
    raw = _WORKFLOW.read_text(encoding="utf-8").splitlines()
    if _trigger_state() == "active":
        try:
            push_start = raw.index(_ACTIVE_PUSH)
            block_end = raw.index("  workflow_dispatch:", push_start)
            lines = raw
        except ValueError as error:
            raise AssertionError("post-merge workflow push paths block is missing") from error
    else:
        try:
            start = raw.index(_PAUSED_PUSH)
            # The commented block is terminated by the live `on:` that follows it.
            block_end_raw = raw.index("on:", start)
        except ValueError as error:
            raise AssertionError("commented push paths block is malformed") from error
        lines = [_uncomment(line) for line in raw[start:block_end_raw]]
        push_start = 0
        block_end = len(lines)

    try:
        paths_start = lines.index("    paths:", push_start, block_end)
    except ValueError as error:
        raise AssertionError("post-merge workflow push paths block is missing") from error

    paths: set[str] = set()
    for line in lines[paths_start + 1 : block_end]:
        if line.lstrip().startswith("- "):
            match = _PATH_ENTRY.fullmatch(line)
            if match is None:
                raise AssertionError(f"unrecognized push path entry: {line!r}")
            path = match.group(1)
            if path in paths:
                raise AssertionError(f"duplicate push path entry: {path}")
            paths.add(path)
    return paths


class PerfPostmergeWorkflowTests(unittest.TestCase):
    def test_loud_status_produces_annotation_without_changing_pass(self) -> None:
        run, output = _classify({"a-quiet.json": _status(),
                                 "b-loud.json": _status(loud_arms=("head1", "base2"))}, "0")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(output, "effective_rc=0\n")
        self.assertEqual(run.stdout.splitlines(), [
            _PHASE_WARNING + "b-loud.json: arm=head1; foreign_max_pct=45.0",
            _PHASE_WARNING + "b-loud.json: arm=base2; foreign_max_pct=45.0",
        ])

    def test_quiet_and_legacy_statuses_produce_no_phase_annotation(self) -> None:
        legacy = _status()
        del legacy["phase_load"]
        del legacy["phase_load_verdict"]
        for statuses in ({}, {"quiet.json": _status()}, {"legacy.json": legacy}):
            with self.subTest(statuses=statuses):
                run, output = _classify(statuses, "0")
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(output, "effective_rc=0\n")
                self.assertEqual(run.stdout, "")
                self.assertEqual(run.stderr, "")

    def test_phase_annotation_preserves_existing_failure_paths(self) -> None:
        cases = [
            ("1", _status(1), 1),
            ("1", _status(2), 2),
            ("2", _status(2), 2),
            ("3", _status(3), 3),
        ]
        ambient_failure = _status(3)
        ambient_failure["ambient"]["samples"]["between"] = 12.0
        cases.append(("3", ambient_failure, 3))
        for rc, quiet, expected_rc in cases:
            with self.subTest(rc=rc, status=quiet):
                loud = {**quiet, "phase_load_verdict": "LOUD",
                        "phase_load": _status(loud_arms=("head1",))["phase_load"]}
                control, control_output = _classify({"target.json": quiet}, rc)
                run, output = _classify({"target.json": loud}, rc)
                self.assertEqual(control.returncode, expected_rc, control.stderr)
                self.assertEqual(run.returncode, control.returncode, run.stderr)
                self.assertEqual(output, control_output)
                self.assertEqual(run.stderr, control.stderr)
                self.assertIn(_PHASE_WARNING + "target.json: arm=head1; foreign_max_pct=45.0\n", run.stdout)
                remaining = "".join(line for line in run.stdout.splitlines(keepends=True)
                                    if not line.startswith(_PHASE_WARNING))
                self.assertEqual(remaining, control.stdout)

    def test_annotation_reader_tolerates_unusable_optional_evidence(self) -> None:
        statuses = {
            "a-malformed.json": "{broken}",
            "b-not-object.json": [],
            "c-missing-arms.json": {"phase_load_verdict": "LOUD"},
            "d-bad-arms.json": {"phase_load_verdict": "LOUD", "phase_load": None},
            "e-bad-entry.json": {"phase_load_verdict": "LOUD", "phase_load": [None, {}]},
            "z-valid.json": _status(loud_arms=("head1",)),
        }
        for rc in ("0", "2"):
            with self.subTest(rc=rc):
                run, output = _classify(statuses, rc)
                self.assertEqual(run.returncode, int(rc), run.stderr)
                self.assertEqual(output, f"effective_rc={rc}\n")
                self.assertEqual(run.stderr, "")
                self.assertEqual(run.stdout, _PHASE_WARNING +
                                 "z-valid.json: arm=head1; foreign_max_pct=45.0\n")
        run, output = _classify({}, "2")
        self.assertEqual(run.returncode, 2)
        self.assertEqual(output, "effective_rc=2\n")
        self.assertEqual(run.stdout, "")

    def test_phase_annotation_escapes_workflow_command_data(self) -> None:
        run, output = _classify({"bad%\r\n::error::.json": _status(loud_arms=("head1",))}, "0")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(output, "effective_rc=0\n")
        self.assertEqual(run.stdout, _PHASE_WARNING +
                         "bad%25%0D%0A::error::.json: arm=head1; foreign_max_pct=45.0\n")

    def test_push_filter_contains_only_bench_binary_inputs(self) -> None:
        self.assertEqual(_push_paths(), _BENCH_BINARY_INPUTS)

    def test_trigger_state_is_unambiguous(self) -> None:
        self.assertIn(_trigger_state(), {"active", "paused"})

    def test_paused_lane_has_no_automatic_trigger(self) -> None:
        if _trigger_state() != "paused":
            self.skipTest("lane is active; this contract governs the paused state")
        workflow = _WORKFLOW.read_text(encoding="utf-8").splitlines()
        on_index = workflow.index("on:")
        block: list[str] = []
        for line in workflow[on_index + 1 :]:
            # The `on:` block ends at the next top-level key.
            if re.fullmatch(r"\w[\w-]*:.*", line):
                break
            block.append(line)
        trigger_keys = [line for line in block if re.fullmatch(r"  \w+:", line)]
        self.assertEqual(trigger_keys, ["  workflow_dispatch:"])
        self.assertIn("PAUSED", workflow[0])

    def test_automated_lane_wires_ambient_status_contract(self) -> None:
        workflow = _WORKFLOW.read_text()
        bench_impl = _BENCH_IMPL.read_text()
        self.assertIn(
            "PERF_POSTMERGE_STATUS_DIR: ${{ runner.temp }}/perf-postmerge-status",
            workflow,
        )
        self.assertIn("perf-postmerge-status-${{ matrix.arch }}", workflow)
        self.assertIn('if [ "$AB_RC" = "3" ]; then', workflow)
        self.assertIn("::warning title=Performance run not measurable::", workflow)
        self.assertIn('--ambient-samples "$AMBIENT_SAMPLES_FILE"', bench_impl)
        self.assertIn("--status-out", bench_impl)
        self.assertIn('if [ -n "${PERF_POSTMERGE_STATUS_DIR:-}" ]', bench_impl)

    def test_not_measurable_does_not_advance_progression_base(self) -> None:
        workflow = _WORKFLOW.read_text()
        record_step = workflow.split(
            "- name: Record the successfully measured head", 1
        )[1].split("- name: Classify the gate outcome", 1)[0]
        self.assertIn("steps.ab.outputs.rc == '0'", record_step)
        self.assertNotIn("success()", record_step)
        self.assertIn("git push origin", record_step)

    def test_not_measurable_fails_closed_at_workflow_consumer(self) -> None:
        workflow = _WORKFLOW.read_text()
        classify_step = workflow.split(
            "- name: Classify the gate outcome", 1
        )[1].split("- name: Record the outcome honestly", 1)[0]
        not_measurable_branch = classify_step.split(
            'if [ "$AB_RC" = "3" ]; then', 1
        )[1].split("fi", 1)[0]
        self.assertIn("exit 3", not_measurable_branch)
        self.assertNotIn("exit 0", not_measurable_branch)


if __name__ == "__main__":
    unittest.main()
