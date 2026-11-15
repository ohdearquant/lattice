#!/usr/bin/env python3
"""Versioned checkpoint evidence and explicitly required ambient phases."""

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest import mock

if __package__:
    from . import bench_compare_support as support
    from . import test_perf_bench_gate_status as status_fixture
else:
    import bench_compare_support as support
    import test_perf_bench_gate_status as status_fixture


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LEGACY_LABELS = ("before first arm", "between order strata", "after final arm")
LABELS = (
    "before first arm", "after base1 settling", "after head1 settling",
    "between order strata", "after head2 settling", "after base2 settling",
    "after final arm",
)
PHASES = (
    "before", "settled-base1", "settled-head1", "between", "settled-head2",
    "settled-base2", "after",
)
LEGACY_HEADER = """schema=lattice-bench-provenance-v2
started_utc=2026-07-29T12:00:00Z
finished_utc=2026-07-29T12:00:40Z
host_id=hostname-sha256:0123456789abcdef
os=fixture-os
base_ref=fixture-base
base_sha=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
head_ref=fixture-head
head_sha=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
head_mode=detached-worktree
base_rustc=rustc fixture-base
head_rustc=rustc fixture-head
base_cargo=cargo fixture-base
head_cargo=cargo fixture-head
base_criterion=0.5.1
head_criterion=0.5.1
criterion_mode=quick
baseline_name=compare-base
targets=lattice-inference:fixture
inference_features=<none>
embed_features=<none>
filters=inference='<all>' embed='<all>'
enforcement=fail-on-regression
lock=bench-window: acquired
lock=Metal GPU: acquired
"""


def checkpoints(labels):
    lines = []
    for i, label in enumerate(labels):
        lines.append(f"ambient=[quiet] {label}: idle 99.0% (floor 70.0%) ok | top: none")
        record = {
            "schema": "lattice-machine-state-v1", "label": label,
            "captured_at_utc": f"2026-07-29T12:00:{10 + i:02d}Z",
            "power": {"status": "unavailable", "reason": "fixture"},
            "thermal": {"status": "unavailable", "reason": "fixture"},
            "idle": {"status": "unavailable", "reason": "fixture"},
        }
        lines.append("machine_state=" + json.dumps(record, separators=(",", ":")))
    return "\n".join(lines) + "\n"


LEGACY_V2 = LEGACY_HEADER + checkpoints(LEGACY_LABELS)
V3 = (LEGACY_HEADER.replace("schema=lattice-bench-provenance-v2\n",
                            "schema=lattice-bench-provenance-v3\nsettle_seconds=5\n")
      + checkpoints(LABELS))


class SettledCheckpointContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = load_module("settling_contract_gate", support.GATE)

    def load_record(self, text):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "provenance.txt"
            path.write_text(text)
            result = self.gate.load_run_provenance(path)
            self.assertEqual(path.read_text(), text)
            return result

    def test_valid_seven_checkpoint_record_is_supported(self):
        error = None
        record = None
        try:
            record = self.load_record(V3)
        except ValueError as failure:
            error = str(failure)
        self.assertIsNone(error, f"valid seven-checkpoint provenance was rejected: {error}")
        self.assertEqual(record.fields["settle_seconds"], "5")
        self.assertEqual(len(record.ambient_samples), 7)
        self.assertEqual([row["label"] for row in record.machine_states], list(LABELS))

    def test_unchanged_v2_record_and_v1_compatibility(self):
        record = self.load_record(LEGACY_V2)
        self.assertEqual(record.schema, "lattice-bench-provenance-v2")
        self.assertEqual(len(record.ambient_samples), 3)
        self.assertEqual([row["label"] for row in record.machine_states], list(LEGACY_LABELS))
        self.assertNotIn("settle_seconds", record.fields)
        legacy_v1 = LEGACY_V2.replace("lattice-bench-provenance-v2", "lattice-bench-provenance-v1").replace("embed_features=<none>\n", "")
        self.assertEqual(self.load_record(legacy_v1).fields["embed_features"], "<none>")
        for old in (LEGACY_V2, legacy_v1):
            with self.assertRaises(ValueError):
                self.load_record(old + "settle_seconds=5\n")

    def test_v3_refuses_missing_duplicate_reordered_and_invalid_checkpoints(self):
        lines = V3.splitlines()
        ambient = next(line for line in lines if line.startswith("ambient=[quiet] after base1"))
        machine = next(line for line in lines if line.startswith("machine_state=") and '"label":"after base1 settling"' in line)
        variants = [
            V3.replace(ambient + "\n", ""),
            V3 + ambient + "\n",
            V3.replace(machine + "\n", ""),
            V3 + machine + "\n",
            V3.replace("after base1 settling", "unexpected arm"),
            V3.replace("after base1 settling", "TEMP").replace("after head1 settling", "after base1 settling").replace("TEMP", "after head1 settling"),
            V3.replace("settle_seconds=5", "settle_seconds=0"),
            V3.replace("settle_seconds=5", "settle_seconds=0.5"),
            V3.replace("settle_seconds=5\n", ""),
            V3.replace("idle 99.0%", "idle 10.0%", 1),
            V3.replace("ok | top", "BELOW FLOOR | top", 1),
        ]
        for text in variants:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    self.load_record(text)
        loud = V3.replace("idle 99.0% (floor 70.0%) ok", "idle 10.0% (floor 70.0%) BELOW FLOOR", 1)
        self.assertEqual(len(self.load_record(loud).ambient_samples), 7)

    def run_status(self, records, *, required=True, provenance=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            criterion = status_fixture._criterion_root(root, "target", -0.02)
            samples = root / "ambient.jsonl"
            samples.write_text("".join(json.dumps(row) + "\n" for row in records))
            status = root / "status.json"
            options = ["--require-settled-arms"] if required else []
            if provenance is not None:
                path = root / "provenance.txt"
                path.write_text(provenance)
                options.extend(("--provenance-file", str(path), "--require-provenance"))
            with mock.patch.dict(os.environ, {"BENCH_IDLE_FLOOR": "70"}):
                result = status_fixture._run(
                    criterion, samples, status, "lattice-inference:fixture",
                    options,
                )
            return result, json.loads(status.read_text())

    def records(self):
        return [{"schema": "perf-ambient-sample/v1", "phase": phase, "idle_pct": 100.0}
                for phase in PHASES]

    def test_required_arm_idle_votes_in_real_status_gate(self):
        result, payload = self.run_status(self.records())
        self.assertEqual(result.returncode, 0, result.stderr)
        healthy = payload
        for phase in PHASES:
            rows = self.records()
            next(row for row in rows if row["phase"] == phase)["idle_pct"] = 10.0
            with self.subTest(phase=phase):
                result, payload = self.run_status(rows)
                self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
                self.assertEqual(payload["ambient"]["assessment"], "not_measurable")
                self.assertEqual(payload["ambient"]["samples"][phase], 10.0)
        self.assertEqual(set(healthy["ambient"]["samples"]), set(PHASES))

    def test_required_phase_stream_rejects_missing_duplicate_unknown_and_bad_numbers(self):
        variants = []
        for i in range(len(PHASES)):
            rows = self.records()
            variants.extend((rows[:i] + rows[i + 1:], rows + [rows[i]]))
        variants.append(self.records() + [{"schema": "perf-ambient-sample/v1", "phase": "other", "idle_pct": 100.0}])
        for value in (True, "100", None, float("nan"), float("inf"), -1.0, 101.0):
            rows = self.records()
            rows[1]["idle_pct"] = value
            variants.append(rows)
        for rows in variants:
            with self.subTest(rows=rows):
                result, payload = self.run_status(rows)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(payload["ambient"]["assessment"], "invalid")

    def test_legacy_three_phase_gate_still_ignores_nonvoting_rows(self):
        rows = self.records()
        rows[1]["idle_pct"] = 0.0
        result, payload = self.run_status(rows, required=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(payload["ambient"]["samples"], {"before": 100.0, "between": 100.0, "after": 100.0})

    def test_provenance_version_and_explicit_required_contract_must_agree(self):
        for required, text in ((True, LEGACY_V2), (False, V3)):
            with self.subTest(required=required):
                result, _ = self.run_status(self.records(), required=required, provenance=text)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("must be used together", result.stderr)

    def test_quiet_probe_emits_each_supported_phase(self):
        probe = load_module("settling_quiet_probe", support.LIB / "quiet-probe.py")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "samples.jsonl"
            for phase in PHASES:
                with mock.patch.object(sys, "argv", ["quiet-probe", "--label", phase,
                                                     "--floor", "70", "--phase", phase,
                                                     "--jsonl-out", str(output)]), \
                        mock.patch.object(probe, "idle_percent", return_value=100.0), \
                        mock.patch.object(probe, "top_consumers", return_value="fixture"):
                    self.assertEqual(probe.main(), 0)
            self.assertEqual([json.loads(line)["phase"] for line in output.read_text().splitlines()], list(PHASES))

    def test_rounded_equality_preserves_raw_below_floor_status(self):
        probe = load_module("settling_boundary_probe", support.LIB / "quiet-probe.py")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "samples.jsonl"
            displayed = io.StringIO()
            with mock.patch.object(sys, "argv", ["quiet-probe", "--label", "after base1 settling",
                                                 "--floor", "70", "--phase", "settled-base1",
                                                 "--jsonl-out", str(output)]), \
                    mock.patch.object(probe, "idle_percent", return_value=69.96), \
                    mock.patch.object(probe, "top_consumers", return_value="fixture"), \
                    redirect_stdout(displayed):
                probe_status = probe.main()
            raw = json.loads(output.read_text())
        self.assertEqual(probe_status, 1)
        self.assertEqual(raw["idle_pct"], 69.96)
        self.assertIn("idle 70.0% (floor 70.0%) BELOW FLOOR", displayed.getvalue())
        text = V3.replace(
            "ambient=[quiet] after base1 settling: idle 99.0% (floor 70.0%) ok | top: none",
            "ambient=" + displayed.getvalue().strip(),
        )
        record = self.load_record(text)
        self.assertIn(displayed.getvalue().strip(), record.ambient_samples)
        self.assertEqual(len(self.load_record(text.replace("BELOW FLOOR", "ok")).ambient_samples), 7)
        rows = self.records()
        rows[1] = raw
        result, payload = self.run_status(rows)
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertEqual(payload["ambient"]["assessment"], "not_measurable")
        self.assertEqual(payload["ambient"]["samples"]["settled-base1"], 69.96)

    def test_journal_write_failure_refuses_certification(self):
        mktemp = r'''#!/usr/bin/env bash
case "${1:-}" in
  *bench-checkpoints.XXXXXX)
    mkdir -p "$STUB_JOURNAL_DIRECTORY"
    printf '%s\n' "$STUB_JOURNAL_DIRECTORY"
    ;;
  *) exec /usr/bin/mktemp "$@" ;;
esac
'''
        with tempfile.TemporaryDirectory() as temporary:
            result = support.run_compare(
                [], stub_cargo=support.STALE_CHANGE_CARGO, stub_mktemp=mktemp,
                extra_env={"STUB_JOURNAL_DIRECTORY": str(Path(temporary) / "directory")},
            )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("cannot retain machine-state checkpoint", result.stderr)
        self.assertNotIn("=== Run conditions ===", result.stdout)


if __name__ == "__main__":
    support.FailOnEmptyTestProgram(verbosity=2)
