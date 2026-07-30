#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "scripts" / "perf-bench-gate.py"
SPEC = importlib.util.spec_from_file_location("perf_bench_gate", GATE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GATE
SPEC.loader.exec_module(GATE)


def result(ci_low_pct: float) -> object:
    return GATE.BenchResult(
        name="layer_norm/896",
        point=ci_low_pct / 100.0 + 0.005,
        ci_low=ci_low_pct / 100.0,
        ci_high=ci_low_pct / 100.0 + 0.01,
        new_ns=104.0,
        old_ns=100.0,
    )


class PerfBenchGateResolutionTests(unittest.TestCase):
    def test_quick_warn_band_is_unattributable_but_reported(self) -> None:
        report = GATE.render_report(
            [result(4.0)], "test-arch", resolution="quick"
        )

        self.assertIn("1 UNATTRIBUTABLE", report)
        self.assertIn("layer_norm/896", report)
        self.assertIn("+4.00%", report)
        self.assertIn("UNATTRIBUTABLE (quick resolution)", report)
        self.assertNotIn("1 WARN", report)

    def test_full_warn_band_remains_warn(self) -> None:
        report = GATE.render_report(
            [result(4.0)], "test-arch", resolution="full"
        )

        self.assertIn("1 WARN", report)
        self.assertIn("layer_norm/896", report)
        self.assertNotIn("UNATTRIBUTABLE", report)

    def test_above_fail_threshold_fails_at_both_resolutions(self) -> None:
        for resolution in ("quick", "full"):
            with self.subTest(resolution=resolution):
                report = GATE.render_report(
                    [result(8.0)], "test-arch", resolution=resolution
                )
                self.assertIn("1 FAIL", report)
                self.assertIn("❌ FAIL", report)
                self.assertNotIn("UNATTRIBUTABLE", report)

    def test_clean_row_passes_at_both_resolutions(self) -> None:
        for resolution in ("quick", "full"):
            with self.subTest(resolution=resolution):
                report = GATE.render_report(
                    [result(2.0)], "test-arch", resolution=resolution
                )
                self.assertIn("All 1 gated benches within noise band", report)
                self.assertNotIn("WARN", report)
                self.assertNotIn("FAIL", report)

    def test_empty_results_refuse_with_and_without_optimization(self) -> None:
        program = f"""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("perf_bench_gate", {str(GATE_PATH)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.render_report([], "test-arch", resolution="quick")
"""
        for flags in ([], ["-O"]):
            with self.subTest(optimized=bool(flags)):
                completed = subprocess.run(
                    [sys.executable, *flags, "-c", program],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(
                    "ValueError: classifier requires at least one measured row",
                    completed.stderr,
                )


if __name__ == "__main__":
    unittest.main()
