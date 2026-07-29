"""Contract tests for the post-merge performance workflow trigger."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "perf-postmerge-gate.yml"
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


if __name__ == "__main__":
    unittest.main()
