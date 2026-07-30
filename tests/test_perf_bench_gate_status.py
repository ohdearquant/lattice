#!/usr/bin/env python3
"""Contract tests for perf-bench-gate's automated ambient status."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "perf-bench-gate.py"
SCHEMA = "perf-ambient-sample/v1"
PHASES = ("before", "between", "after")


def _criterion_root(parent: Path, name: str, ci_low: float) -> Path:
    root = parent / name / "criterion"
    bench = root / "group" / "bench"
    for artifact in ("compare-base", "new", "change"):
        (bench / artifact).mkdir(parents=True)
    (bench / "compare-base" / "estimates.json").write_text(
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
                    self.assertEqual(result.returncode, 3, result.stderr)
                    self.assertEqual(
                        json.loads(status.read_text())["verdict"],
                        "not_measurable",
                    )

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
            self.assertEqual(json.loads(status.read_text())["verdict"], "pass")


if __name__ == "__main__":
    unittest.main()
