"""Contract tests for the post-merge performance workflow trigger."""

from __future__ import annotations

import re
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


if __name__ == "__main__":
    unittest.main()
