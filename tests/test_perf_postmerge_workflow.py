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


def _push_paths() -> set[str]:
    lines = _WORKFLOW.read_text(encoding="utf-8").splitlines()
    try:
        push_start = lines.index("  push:")
        dispatch_start = lines.index("  workflow_dispatch:", push_start)
        paths_start = lines.index("    paths:", push_start, dispatch_start)
    except ValueError as error:
        raise AssertionError("post-merge workflow push paths block is missing") from error

    paths: set[str] = set()
    for line in lines[paths_start + 1 : dispatch_start]:
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
