#!/usr/bin/env python3
"""Regression tests for the macOS benchmark machine-state governor."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "perf_governor.py"


def load_governor():
    spec = importlib.util.spec_from_file_location("perf_governor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ThermalReaders(unittest.TestCase):
    def setUp(self):
        self.pg = load_governor()

    def test_shared_probe_pressure_is_not_rewritten_as_nominal(self):
        with mock.patch.object(
            self.pg._MACHINE_STATE_PROBE,
            "read_macos_thermal",
            return_value={
                "status": "measured",
                "source": "pmset",
                "state": "throttled",
                "cpu_speed_limit_percent": 80,
            },
        ):
            state = self.pg._read_thermal()
        self.assertFalse(state["nominal"])
        self.assertEqual(state["speed_limit"], 80)
        self.assertEqual(state["source"], "pmset")

    def test_unavailable_shared_probe_fails_closed(self):
        """Missing thermal evidence cannot be rewritten as nominal.

        Mutation-sensitive: restoring the old ``assume nominal`` branch makes
        this assertion read true.
        """
        with mock.patch.object(
            self.pg._MACHINE_STATE_PROBE,
            "read_macos_thermal",
            return_value={
                "status": "unavailable",
                "reason": "pmset and ProcessInfo unavailable",
            },
        ):
            state = self.pg._read_thermal()
        self.assertFalse(state["nominal"])
        self.assertEqual(state["state"], "unavailable")
        self.assertIn("ProcessInfo unavailable", state["source"])


class Checkpoint(unittest.TestCase):
    def setUp(self):
        self.pg = load_governor()

    def governor(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        gov = self.pg.PerfGovernor(
            cooldown_s=0,
            afk_threshold_s=30,
            sentinel_path=Path(temp.name) / "stop",
        )
        gov._ac_reader = lambda: True
        gov._thermal_reader = lambda: {
            "speed_limit": None,
            "nominal": True,
            "state": "nominal",
            "source": "fixture",
        }
        gov._idle_reader = lambda: 31.25
        return gov

    @staticmethod
    def record():
        return {
            "schema": "lattice-machine-state-v1",
            "label": "fixture",
            "captured_at_utc": "2026-07-29T00:00:00Z",
            "power": {
                "status": "measured",
                "source": "fixture",
                "state": "ac",
            },
            "thermal": {
                "status": "measured",
                "source": "fixture",
                "state": "nominal",
            },
            "idle": {
                "status": "measured",
                "source": "fixture",
                "seconds": 31.25,
            },
        }

    def test_preflight_uses_the_exact_snapshot_that_is_reported(self):
        """A passing line and the decision must describe one hardware read."""
        gov = self.governor()
        reads = 0

        def thermal():
            nonlocal reads
            reads += 1
            return {
                "speed_limit": None,
                "nominal": reads == 1,
                "state": "nominal" if reads == 1 else "critical",
                "source": "fixture",
            }

        gov._thermal_reader = thermal
        state = gov.status()
        gov.preflight(state)
        self.assertEqual(reads, 1)

    def test_checkpoint_emits_machine_readings_and_passes(self):
        gov = self.governor()
        stdout = io.StringIO()
        record = self.record()
        record["label"] = "before base"
        with mock.patch.object(
            self.pg._MACHINE_STATE_PROBE,
            "collect_record",
            return_value=record,
        ) as collect:
            with contextlib.redirect_stdout(stdout):
                with contextlib.redirect_stderr(io.StringIO()):
                    rc = self.pg._cmd_checkpoint(gov, "before base")
        self.assertEqual(rc, 0)
        emitted = json.loads(stdout.getvalue())
        self.assertEqual(emitted["label"], "before base")
        self.assertEqual(emitted["power"]["state"], "ac")
        self.assertEqual(emitted["thermal"]["state"], "nominal")
        self.assertEqual(emitted["idle"]["seconds"], 31.25)
        self.assertEqual(emitted["gate"]["status"], "passed")
        self.assertEqual(emitted["gate"]["afk_threshold_seconds"], 30)
        collect.assert_called_once_with("before base", self.pg.sys.platform)

    def test_checkpoint_reports_then_refuses_non_nominal_state(self):
        """The state record cannot be emitted as a warning with exit zero."""
        gov = self.governor()
        record = self.record()
        record["thermal"]["state"] = "serious"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            self.pg._MACHINE_STATE_PROBE,
            "collect_record",
            return_value=record,
        ):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = self.pg._cmd_checkpoint(gov, "between phases")
        self.assertEqual(rc, 2)
        emitted = json.loads(stdout.getvalue())
        self.assertEqual(emitted["thermal"]["state"], "serious")
        self.assertEqual(emitted["gate"]["status"], "blocked")
        self.assertIn("CHECKPOINT BLOCKED: THERMAL", stderr.getvalue())

    def test_checkpoint_refuses_battery_unavailable_and_active_states(self):
        mutations = (
            ("battery", ("power", "state"), "battery", "AC-GATE"),
            (
                "power unavailable",
                ("power", "status"),
                "unavailable",
                "AC-GATE",
            ),
            (
                "thermal unavailable",
                ("thermal", "status"),
                "unavailable",
                "THERMAL",
            ),
            ("machine active", ("idle", "seconds"), 2.0, "AFK-ONLY"),
        )
        for name, path, value, reason in mutations:
            with self.subTest(name=name):
                gov = self.governor()
                record = self.record()
                record[path[0]][path[1]] = value
                stdout = io.StringIO()
                stderr = io.StringIO()
                with mock.patch.object(
                    self.pg._MACHINE_STATE_PROBE,
                    "collect_record",
                    return_value=record,
                ):
                    with contextlib.redirect_stdout(
                        stdout
                    ), contextlib.redirect_stderr(stderr):
                        rc = self.pg._cmd_checkpoint(gov, "fixture")
                self.assertEqual(rc, 2)
                self.assertEqual(
                    json.loads(stdout.getvalue())["gate"]["status"],
                    "blocked",
                )
                self.assertIn(reason, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
