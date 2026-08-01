#!/usr/bin/env python3
"""Contract tests for perf-bench-gate's automated ambient status."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "perf-bench-gate.py"
SCHEMA = "perf-ambient-sample/v1"
PHASES = ("before", "between", "after")


def _criterion_root(
    parent: Path,
    name: str,
    ci_low: float,
    *,
    baseline_name: str = "compare-base",
) -> Path:
    root = parent / name / "criterion"
    bench = root / "group" / "bench"
    for artifact in (baseline_name, "new", "change"):
        (bench / artifact).mkdir(parents=True)
    (bench / baseline_name / "estimates.json").write_text(
        '{"mean":{"point_estimate":100.0}}\n'
    )
    (bench / "new" / "estimates.json").write_text(
        '{"mean":{"point_estimate":110.0}}\n'
    )
    (bench / "change" / "estimates.json").write_text(json.dumps({
        "mean": {
            "point_estimate": ci_low + 0.01,
            "confidence_interval": {
                "lower_bound": ci_low,
                "upper_bound": ci_low + 0.02,
            },
        },
    }))
    return root


def _samples(path: Path, values: dict[str, float], extras=()) -> None:
    records = [
        {"schema": SCHEMA, "phase": phase, "idle_pct": idle}
        for phase, idle in values.items()
    ]
    records.extend(extras)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _run(root: Path, samples: Path, status: Path, target: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "python3",
            str(GATE),
            str(root),
            f"fixture/{target}",
            "--target",
            target,
            "--require-measurements",
            "--ambient-samples",
            str(samples),
            "--status-out",
            str(status),
        ],
        text=True,
        capture_output=True,
    )


class PerfBenchGateStatusTests(unittest.TestCase):
    def test_single_abba_block_cannot_detect_sign_changing_order_effects(
        self,
    ) -> None:
        spec = importlib.util.spec_from_file_location("perf_bench_gate", GATE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        forward = module.BenchResult(
            "group/bench", 0.10, 0.10, 0.10, 110.0, 100.0
        )
        reverse = module.BenchResult(
            "group/bench", -0.10, -0.10, -0.10, 90.0, 100.0
        )

        result = module.order_balance_pair(forward, reverse)

        self.assertAlmostEqual(result.point_pct, 10.554160, places=6)
        self.assertAlmostEqual(result.order_bias_bound_pct, 0.503782, places=6)
        self.assertEqual(result.verdict(), "FAIL")

    def test_extra_reverse_comparison_is_an_input_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            criterion = _criterion_root(root, "forward", 0.095)
            control = _criterion_root(
                root,
                "reverse",
                0.095,
                baseline_name="compare-head",
            )
            extra = control / "group" / "unexpected" / "change"
            extra.mkdir(parents=True)
            (extra / "estimates.json").write_text(
                '{"mean":{"point_estimate":0.0,'
                '"confidence_interval":{"lower_bound":0.0,'
                '"upper_bound":0.0}}}\n'
            )
            samples = root / "ambient.jsonl"
            _samples(samples, {phase: 95.0 for phase in PHASES})
            status = root / "status.json"
            target = "lattice-inference:elementwise_cpu_bench"

            result = subprocess.run(
                [
                    "python3",
                    str(GATE),
                    str(criterion),
                    "fixture/extra-order-control",
                    "--target",
                    target,
                    "--require-measurements",
                    "--require-order-balance",
                    "--order-control-root",
                    str(control),
                    "--ambient-samples",
                    str(samples),
                    "--status-out",
                    str(status),
                ],
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            payload = json.loads(status.read_text())
            self.assertEqual(payload["verdict"], "error")
            self.assertIn("order-control comparison set differs", payload["reason"])
            self.assertIn("group/unexpected", payload["reason"])

    def test_malformed_reverse_numbers_are_input_errors_with_status(self) -> None:
        cases = (
            ("string point", ("point_estimate",), "corrupt"),
            ("boolean CI", ("confidence_interval", "lower_bound"), True),
        )
        for label, path, invalid_value in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                criterion = _criterion_root(root, "forward", 0.095)
                control = _criterion_root(
                    root,
                    "reverse",
                    0.095,
                    baseline_name="compare-head",
                )
                change_path = control / "group" / "bench" / "change" / "estimates.json"
                change = json.loads(change_path.read_text())
                destination = change["mean"]
                for key in path[:-1]:
                    destination = destination[key]
                destination[path[-1]] = invalid_value
                change_path.write_text(json.dumps(change))

                samples = root / "ambient.jsonl"
                _samples(samples, {phase: 95.0 for phase in PHASES})
                status = root / "status.json"
                target = "lattice-inference:elementwise_cpu_bench"
                result = subprocess.run(
                    [
                        "python3",
                        str(GATE),
                        str(criterion),
                        "fixture/malformed-order-control",
                        "--target",
                        target,
                        "--require-measurements",
                        "--require-order-balance",
                        "--order-control-root",
                        str(control),
                        "--ambient-samples",
                        str(samples),
                        "--status-out",
                        str(status),
                    ],
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                payload = json.loads(status.read_text())
                self.assertEqual(payload["verdict"], "error")
                self.assertEqual(payload["exit_code"], 2)
                self.assertIn("invalid order-control evidence", payload["reason"])

    def test_required_order_balance_without_control_writes_error_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            criterion = _criterion_root(root, "forward", 0.095)
            samples = root / "ambient.jsonl"
            _samples(samples, {phase: 95.0 for phase in PHASES})
            status = root / "status.json"
            target = "lattice-inference:elementwise_cpu_bench"
            result = subprocess.run(
                [
                    "python3",
                    str(GATE),
                    str(criterion),
                    "fixture/missing-order-control",
                    "--target",
                    target,
                    "--require-measurements",
                    "--require-order-balance",
                    "--ambient-samples",
                    str(samples),
                    "--status-out",
                    str(status),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            payload = json.loads(status.read_text())
            self.assertEqual(payload["verdict"], "error")
            self.assertEqual(payload["exit_code"], 2)
            self.assertIn("needs --order-control-root", payload["reason"])

    def test_gate_sized_order_bias_is_not_measurable_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            criterion = _criterion_root(root, "forward", 0.095)
            control = _criterion_root(
                root,
                "reverse",
                0.095,
                baseline_name="compare-head",
            )
            samples = root / "ambient.jsonl"
            _samples(samples, {phase: 95.0 for phase in PHASES})
            status = root / "status.json"
            target = "lattice-inference:elementwise_cpu_bench"
            result = subprocess.run(
                [
                    "python3",
                    str(GATE),
                    str(criterion),
                    "fixture/order-bias",
                    "--target",
                    target,
                    "--require-measurements",
                    "--require-order-balance",
                    "--order-control-root",
                    str(control),
                    "--ambient-samples",
                    str(samples),
                    "--status-out",
                    str(status),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 3, result.stderr)
            self.assertIn("**⏸ NOT MEASURABLE**", result.stdout)
            self.assertNotIn("✅ All 1 gated benches", result.stdout)
            payload = json.loads(status.read_text())
            self.assertEqual(payload["verdict"], "not_measurable")
            self.assertEqual(payload["exit_code"], 3)
            self.assertIn("order-bias bound above", payload["reason"])
            self.assertEqual(payload["measurement_count"], 1)
            self.assertEqual(payload["ambient"]["assessment"], "valid")

    def test_completeness_error_outranks_not_measurable_for_informational_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            criterion = root / "empty" / "criterion"
            criterion.mkdir(parents=True)
            samples = root / "ambient.jsonl"
            _samples(samples, {"before": 95.0, "between": 12.0, "after": 94.0})
            status = root / "status.json"
            target = "lattice-embed:simd"
            result = subprocess.run(
                [
                    "python3",
                    str(GATE),
                    str(criterion),
                    "fixture/informational",
                    "--target",
                    target,
                    "--informational-target",
                    target,
                    "--require-measurements",
                    "--ambient-samples",
                    str(samples),
                    "--status-out",
                    str(status),
                ],
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 2, result.stderr)
            payload = json.loads(status.read_text())
            self.assertEqual(payload["verdict"], "error")
            self.assertEqual(payload["exit_code"], 2)
            self.assertEqual(payload["ambient"]["samples"]["between"], 12.0)

    def test_below_floor_refuses_both_target_verdicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "ambient.jsonl"
            _samples(samples, {"before": 95.0, "between": 12.0, "after": 94.0})

            for target, ci_low in (
                ("lattice-inference:elementwise_cpu_bench", 0.0),
                ("lattice-embed:simd", 0.20),
            ):
                with self.subTest(target=target):
                    status = root / f"{target.split(':')[0]}.json"
                    result = _run(
                        _criterion_root(root, target.split(":")[0], ci_low),
                        samples,
                        status,
                        target,
                    )
                    self.assertEqual(result.returncode, 3, result.stderr)
                    payload = json.loads(status.read_text())
                    self.assertEqual(payload["verdict"], "not_measurable")
                    self.assertEqual(
                        payload["ambient"]["assessment"], "not_measurable"
                    )
                    self.assertEqual(payload["ambient"]["samples"]["between"], 12.0)
                    self.assertNotIn("measurement_count", payload)

    def test_missing_and_duplicated_voting_phases_refuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            criterion = _criterion_root(root, "pass", 0.0)
            cases = {
                "missing": [
                    {"schema": SCHEMA, "phase": "before", "idle_pct": 95.0},
                    {"schema": SCHEMA, "phase": "after", "idle_pct": 95.0},
                ],
                "duplicate": [
                    {"schema": SCHEMA, "phase": phase, "idle_pct": 95.0}
                    for phase in (*PHASES, "before")
                ],
            }
            for name, records in cases.items():
                with self.subTest(name=name):
                    samples = root / f"{name}.jsonl"
                    samples.write_text(
                        "".join(json.dumps(record) + "\n" for record in records)
                    )
                    status = root / f"{name}.json"
                    result = _run(
                        criterion, samples, status, "lattice-inference:fixture"
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    payload = json.loads(status.read_text())
                    self.assertEqual(payload["verdict"], "error")
                    self.assertEqual(payload["ambient"]["assessment"], "invalid")

    def test_invalid_ambient_sample_streams_are_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            criterion = _criterion_root(root, "pass", 0.0)
            cases = {
                "malformed": "{not-json}\n",
                "wrong-schema": "".join(
                    json.dumps(
                        {
                            "schema": "perf-ambient-sample/v2",
                            "phase": phase,
                            "idle_pct": 95.0,
                        }
                    )
                    + "\n"
                    for phase in PHASES
                ),
                "empty": "",
            }
            for name, contents in cases.items():
                with self.subTest(name=name):
                    samples = root / f"{name}.jsonl"
                    samples.write_text(contents)
                    status = root / f"{name}.json"
                    result = _run(
                        criterion, samples, status, "lattice-inference:fixture"
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    payload = json.loads(status.read_text())
                    self.assertEqual(payload["verdict"], "error")
                    self.assertEqual(payload["exit_code"], 2)
                    self.assertEqual(payload["ambient"]["assessment"], "invalid")

    def test_missing_ambient_sample_file_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status = root / "status.json"
            result = _run(
                _criterion_root(root, "pass", 0.0),
                root / "missing.jsonl",
                status,
                "lattice-inference:fixture",
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            payload = json.loads(status.read_text())
            self.assertEqual(payload["verdict"], "error")
            self.assertEqual(payload["exit_code"], 2)
            self.assertEqual(payload["ambient"]["assessment"], "invalid")

    def test_non_voting_phase_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "ambient.jsonl"
            _samples(
                samples,
                {phase: 95.0 for phase in PHASES},
                [{"schema": SCHEMA, "phase": "build", "idle_pct": 0.0}],
            )
            status = root / "status.json"
            result = _run(
                _criterion_root(root, "pass", 0.0),
                samples,
                status,
                "lattice-inference:fixture",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(status.read_text())
            self.assertEqual(payload["verdict"], "pass")
            self.assertEqual(payload["ambient"]["assessment"], "valid")


if __name__ == "__main__":
    unittest.main()
