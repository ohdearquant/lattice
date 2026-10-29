#!/usr/bin/env python3
"""Stream validation and early failures, without running numerical fitting."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]


class _UnavailableNumpy(types.ModuleType):
    def __getattr__(self, name):
        raise AssertionError(f"NumPy operation {name!r} is outside stream validation")


def _load_slopefit():
    spec = importlib.util.spec_from_file_location(
        "slopefit_record_validation", REPO / "scripts/bench_decode_slopefit.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"numpy": _UnavailableNumpy("numpy")}):
        spec.loader.exec_module(module)
    return module


def _run_record(repeats=2):
    return f"SLOPEFIT_META kv_cache_len=65536 warmup=4 measure=16 repeats={repeats}"


def _context_record(ctx):
    return f"SLOPEFIT_META ctx={ctx} actual_prompt_tokens={ctx + 6}"


def _measurement(ctx, rep, tokens=16, measure_ms=64.0):
    return (
        f"SLOPEFIT ctx={ctx} tokens={tokens} warmup_ms=0.0 "
        f"measure_ms={measure_ms} rep={rep}"
    )


def _complete_records(repeats=2):
    records = [_run_record(repeats)]
    for ctx in (64, 256, 512):
        records.append(_context_record(ctx))
        records.extend(_measurement(ctx, rep) for rep in range(repeats))
    return records


def _interleaved_records():
    return [
        "[slopefit] loading model",
        _measurement(256, 1, tokens=16, measure_ms=88.5),
        _measurement(64, 0, tokens=8, measure_ms=24.0),
        _context_record(512),
        _measurement(512, 0, tokens=32, measure_ms=224.0),
        _context_record(64),
        _measurement(64, 1, tokens=16, measure_ms=64.0),
        "[slopefit] ctx=64 actual_prompt_tokens=9999",
        _measurement(256, 0, tokens=32, measure_ms=208.0),
        _context_record(256),
        _measurement(512, 1, tokens=16, measure_ms=120.0),
        _run_record(),
        "bench-supervision: measured-phase receipt available",
    ]


class SlopefitRecordParsingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.slopefit = _load_slopefit()

    def test_duplicate_repeat_in_complete_stream_is_rejected(self):
        records = _complete_records()
        records.append(_measurement(64, 0))
        with self.assertRaisesRegex(ValueError, r"^duplicate SLOPEFIT repeat: ctx=64 rep=0$"):
            self.slopefit.parse_stream(records)

    def test_duplicate_repeat_with_changed_payload_is_rejected(self):
        records = _complete_records()
        records.append(_measurement(64, 0, tokens=32, measure_ms=96.0))
        with self.assertRaisesRegex(ValueError, r"^duplicate SLOPEFIT repeat: ctx=64 rep=0$"):
            self.slopefit.parse_stream(records)

    def test_duplicate_run_metadata_is_rejected(self):
        for duplicate in (_run_record(), _run_record(repeats=3)):
            with self.subTest(duplicate=duplicate):
                with self.assertRaisesRegex(ValueError, r"^duplicate SLOPEFIT_META run record$"):
                    self.slopefit.parse_stream([*_complete_records(), duplicate])

    def test_duplicate_context_metadata_is_rejected(self):
        for duplicate in (
            _context_record(64),
            "SLOPEFIT_META ctx=64 actual_prompt_tokens=9999",
        ):
            with self.subTest(duplicate=duplicate):
                with self.assertRaisesRegex(
                    ValueError, r"^duplicate SLOPEFIT_META context record: ctx=64$"
                ):
                    self.slopefit.parse_stream([*_complete_records(), duplicate])

    def test_repeat_ids_are_scoped_to_each_context(self):
        raw, tokens, metadata, run = self.slopefit.parse_stream(_complete_records())
        self.assertEqual(dict(raw), {64: [4.0, 4.0], 256: [4.0, 4.0], 512: [4.0, 4.0]})
        self.assertEqual(dict(tokens), {64: [16, 16], 256: [16, 16], 512: [16, 16]})
        self.assertEqual(metadata, {64: 70, 256: 262, 512: 518})
        self.assertEqual(run["repeats"], 2)

    def test_missing_context_metadata_is_rejected(self):
        records = _complete_records()
        records.remove(_context_record(64))
        with self.assertRaisesRegex(ValueError, r"^missing SLOPEFIT_META context record: ctx=64$"):
            self.slopefit.parse_stream(records)

    def test_context_metadata_without_measurements_is_rejected(self):
        records = _complete_records()
        records.remove(_measurement(64, 0))
        records.remove(_measurement(64, 1))
        with self.assertRaisesRegex(ValueError, r"^missing SLOPEFIT measurements: ctx=64$"):
            self.slopefit.parse_stream(records)

    def test_missing_middle_repeat_is_rejected(self):
        records = _complete_records(repeats=3)
        records.remove(_measurement(64, 1))
        with self.assertRaisesRegex(
            ValueError, r"^incomplete SLOPEFIT repeats: ctx=64 expected=3 observed=2$"
        ):
            self.slopefit.parse_stream(records)

    def test_repeat_ids_cannot_start_at_one(self):
        records = _complete_records()
        records.remove(_measurement(64, 0))
        records.append(_measurement(64, 2))
        with self.assertRaisesRegex(
            ValueError, r"^SLOPEFIT repeat outside declared range: ctx=64 rep=2 repeats=2$"
        ):
            self.slopefit.parse_stream(records)

    def test_extra_out_of_range_repeat_is_rejected(self):
        records = [*_complete_records(), _measurement(64, 7)]
        with self.assertRaisesRegex(
            ValueError, r"^SLOPEFIT repeat outside declared range: ctx=64 rep=7 repeats=2$"
        ):
            self.slopefit.parse_stream(records)

    def test_zero_declared_repeats_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"^SLOPEFIT_META repeats must be positive$"):
            self.slopefit.parse_stream([_run_record(repeats=0)])

    def test_valid_interleaving_preserves_order_shape_and_numeric_values(self):
        result = self.slopefit.parse_stream(iter(_interleaved_records()))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        raw, tokens, metadata, run = result
        for values in (raw, tokens):
            self.assertIsInstance(values, defaultdict)
            self.assertIs(values.default_factory, list)
        self.assertEqual(list(raw.items()), [(256, [5.53125, 6.5]), (64, [3.0, 4.0]), (512, [7.0, 7.5])])
        self.assertEqual(list(tokens.items()), [(256, [16, 32]), (64, [8, 16]), (512, [32, 16])])
        self.assertEqual(list(metadata.items()), [(512, 518), (64, 70), (256, 262)])
        self.assertEqual(run, {"kv_cache_len": 65536, "warmup": 4, "measure": 16, "repeats": 2})

    def test_noise_only_stream_retains_empty_parser_shape(self):
        result = self.slopefit.parse_stream([
            "[slopefit] loading model",
            "[slopefit] ctx=64 actual_prompt_tokens=9999",
            "[slopefit] ctx=64 tokens=1 warmup_ms=0.0 measure_ms=1.0 rep=0",
            "bench-supervision: measured-phase receipt available",
        ])
        self.assertEqual(result, ({}, {}, {}, None))
        self.assertIs(result[0].default_factory, list)
        self.assertIs(result[1].default_factory, list)

    def test_empty_stream_retains_empty_parser_shape(self):
        self.assertEqual(self.slopefit.parse_stream([]), ({}, {}, {}, None))

    def test_missing_run_metadata_remains_visible_to_caller(self):
        records = _complete_records()[1:]
        raw, tokens, metadata, run = self.slopefit.parse_stream(records)
        self.assertEqual(set(raw), {64, 256, 512})
        self.assertEqual(set(tokens), {64, 256, 512})
        self.assertEqual(set(metadata), {64, 256, 512})
        self.assertIsNone(run)


class SlopefitMainFailureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.slopefit = _load_slopefit()

    def _assert_failure(self, records, expected, output):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["bench_decode_slopefit.py", "--out", str(output)]),
            mock.patch.object(sys, "stdin", io.StringIO("\n".join(records) + "\n")),
            mock.patch.dict(os.environ, {"SLOPEFIT_FULL": "0"}),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as caught,
        ):
            self.slopefit.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertEqual(json.loads(stdout.getvalue()), expected)
        self.assertEqual(stderr.getvalue(), "")

    def test_parser_failure_emits_only_error_json_and_creates_no_output(self):
        records = [*_complete_records(), _measurement(64, 0)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new" / "result.json"
            self._assert_failure(records, {"error": "duplicate SLOPEFIT repeat: ctx=64 rep=0"}, output)
            self.assertFalse(output.exists())
            self.assertFalse(output.parent.exists())

    def test_missing_run_metadata_emits_error_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            self._assert_failure(_complete_records()[1:], {
                "error": "missing SLOPEFIT_META run line -- binary did not emit expected metadata"
            }, output)
            self.assertFalse(output.exists())

    def test_empty_or_noise_only_stream_fails_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            for records in ([], ["[slopefit] loading model", "supervisor notice"]):
                with self.subTest(records=records):
                    self._assert_failure(records, {
                        "error": "missing SLOPEFIT_META run line -- binary did not emit expected metadata"
                    }, output)
                    self.assertFalse(output.exists())

    def test_run_metadata_without_contexts_emits_error_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            self._assert_failure([_run_record()], {
                "error": "fewer than 3 context points measured", "n_ctx": 0
            }, output)
            self.assertFalse(output.exists())

    def test_parser_failure_preserves_existing_output(self):
        records = _complete_records()
        records.remove(_context_record(64))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            original = b"existing report\n"
            output.write_bytes(original)
            self._assert_failure(records, {
                "error": "missing SLOPEFIT_META context record: ctx=64"
            }, output)
            self.assertEqual(output.read_bytes(), original)


class _FailOnEmptyTestProgram(unittest.TestProgram):
    def runTests(self) -> None:
        if self.test.countTestCases() == 0:
            raise SystemExit("no tests collected")
        super().runTests()


if __name__ == "__main__":
    _FailOnEmptyTestProgram()
