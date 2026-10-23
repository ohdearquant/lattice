"""Unit tests for scripts/adr064-gpu-decode-gate.py.

Run with: python3 -m pytest tests/test_adr064_gpu_decode_gate.py -v
(or plain `python3 -m unittest tests/test_adr064_gpu_decode_gate.py` -- no
pytest-only features are used).
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "adr064-gpu-decode-gate.py"
_SPEC = importlib.util.spec_from_file_location("adr064_gpu_decode_gate", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
gate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gate
_SPEC.loader.exec_module(gate)


def _bench(
    value: float, ci95_low: float, ci95_high: float, higher_is_better: bool, unit: str = "unit"
) -> dict:
    return {
        "value": value,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "unit": unit,
        "higher_is_better": higher_is_better,
    }


def _base_benches() -> dict:
    """A clean baseline: every ADR-064 row's inputs, all comfortably passing."""
    return {
        "decode/tok_s/64": _bench(100.0, 98.0, 102.0, True, "tok_s"),
        "decode/tok_s/4096": _bench(80.0, 78.0, 82.0, True, "tok_s"),
        "decode/slope_ms_per_ctx_tok": _bench(0.010, 0.0095, 0.0105, False, "ms/ctx_tok"),
        "decode/intercept_ms": _bench(5.0, 4.8, 5.2, False, "ms"),
        "decode/ttft_ms/4096": _bench(50.0, 48.0, 52.0, False, "ms"),
        "decode/ttft_ms/16384": _bench(180.0, 175.0, 185.0, False, "ms"),
        "decode/dispatches_per_token": _bench(20.0, 19.5, 20.5, False, "count"),
        "decode/command_buffers_per_token": _bench(1.0, 0.9, 1.1, False, "count"),
        "quality/ppl_delta/f16": _bench(0.001, 0.0005, 0.0015, False, "ppl_delta"),
        "quality/ppl_delta/bf16": _bench(0.01, 0.005, 0.015, False, "ppl_delta"),
        "quality/ppl_delta/q4_kv": _bench(0.10, 0.05, 0.15, False, "ppl_delta"),
        "quality/greedy_agreement": _bench(1.0, 1.0, 1.0, True, "ratio"),
        "quality/topk_exact": _bench(1.0, 1.0, 1.0, True, "ratio"),
        "contention/loss_pp/w4": _bench(1.0, 0.5, 1.5, False, "pp"),
        "contention/loss_frac/w10": _bench(0.02, 0.01, 0.03, False, "fraction"),
        "runtime/kv_layout_assertion": _bench(1.0, 1.0, 1.0, True, "bool01"),
    }


def _doc(benches: dict) -> dict:
    return {
        "commit": "deadbeef",
        "date": "2026-07-01T00:00:00Z",
        "arch": "m2max-metal",
        "benches": benches,
    }


class NoChangeFixtureTest(unittest.TestCase):
    def test_identical_docs_via_evaluate(self):
        baseline = _doc(_base_benches())
        current = _doc(_base_benches())
        results = gate.evaluate(current, baseline)
        for r in results:
            self.assertEqual(r.verdict, "PASS", f"{r.name}: {r.reasons}")
        self.assertEqual(gate.overall_exit_code(results), 0)

    def test_mild_noise_within_ci_passes(self):
        baseline = _doc(_base_benches())
        current_benches = _base_benches()
        # Point estimate wobbles but CI lower bound (of the confirmed-regression
        # direction) stays inside tolerance -- must not fail.
        current_benches["decode/slope_ms_per_ctx_tok"] = _bench(
            0.0102, 0.0096, 0.0108, False, "ms/ctx_tok"
        )
        current = _doc(current_benches)
        results = gate.evaluate(current, baseline)
        for r in results:
            self.assertEqual(r.verdict, "PASS", f"{r.name}: {r.reasons}")


class DecodeRowRegressionTest(unittest.TestCase):
    def test_slope_regression_trips_only_decode_row(self):
        baseline = _doc(_base_benches())
        current_benches = _base_benches()
        # slope regresses by >5% at the CI lower bound (lower-is-better).
        current_benches["decode/slope_ms_per_ctx_tok"] = _bench(
            0.0114, 0.0112, 0.0116, False, "ms/ctx_tok"
        )
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        self.assertEqual(results["decode_tok_s_slope_intercept"].verdict, "FAIL")
        for name, r in results.items():
            if name != "decode_tok_s_slope_intercept":
                self.assertEqual(r.verdict, "PASS", f"{name}: {r.reasons}")

    def test_tok_s_regression_trips_decode_row(self):
        baseline = _doc(_base_benches())
        current_benches = _base_benches()
        # tok/s drops >7% at the CI lower bound (higher-is-better).
        current_benches["decode/tok_s/64"] = _bench(85.0, 83.0, 87.0, True, "tok_s")
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        self.assertEqual(results["decode_tok_s_slope_intercept"].verdict, "FAIL")

    def test_intercept_regression_trips_decode_row(self):
        baseline = _doc(_base_benches())
        current_benches = _base_benches()
        current_benches["decode/intercept_ms"] = _bench(5.65, 5.65, 5.75, False, "ms")
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        self.assertEqual(results["decode_tok_s_slope_intercept"].verdict, "FAIL")


class TtftDispatchRowRegressionTest(unittest.TestCase):
    def test_ttft_regression_trips_only_ttft_row(self):
        baseline = _doc(_base_benches())
        current_benches = _base_benches()
        current_benches["decode/ttft_ms/4096"] = _bench(58.0, 57.5, 58.5, False, "ms")
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        self.assertEqual(results["ttft_dispatch"].verdict, "FAIL")
        for name, r in results.items():
            if name != "ttft_dispatch":
                self.assertEqual(r.verdict, "PASS", f"{name}: {r.reasons}")

    def test_dispatches_absolute_delta_below_threshold_passes_row(self):
        baseline_benches = _base_benches()
        baseline_benches["decode/dispatches_per_token"] = _bench(
            250.0, 249.0, 250.0, False, "count"
        )
        baseline = _doc(baseline_benches)
        current_benches = _base_benches()
        # disp_abs=9.9, disp_lb=3.96% -- both comfortably under threshold.
        current_benches["decode/dispatches_per_token"] = _bench(259.9, 259.9, 260.0, False, "count")
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        self.assertEqual(
            results["ttft_dispatch"].verdict,
            "PASS",
            f"dispatch abs below-threshold: expected PASS below +10, got "
            f"{results['ttft_dispatch'].verdict}: {results['ttft_dispatch'].reasons}",
        )

    def test_dispatches_absolute_delta_above_threshold_trips_abs_only(self):
        baseline_benches = _base_benches()
        baseline_benches["decode/dispatches_per_token"] = _bench(
            250.0, 249.0, 250.0, False, "count"
        )
        baseline = _doc(baseline_benches)
        current_benches = _base_benches()
        # disp_abs=10.1, disp_lb=4.04% -- only the absolute sub-predicate fires.
        current_benches["decode/dispatches_per_token"] = _bench(260.1, 260.1, 260.2, False, "count")
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        row = results["ttft_dispatch"]
        self.assertEqual(
            row.verdict,
            "FAIL",
            f"dispatch abs above-threshold: expected abs_delta-only FAIL above +10, "
            f"got {row.verdict}: {row.reasons}",
        )
        self.assertTrue(
            any("abs_delta" in reason for reason in row.reasons),
            f"dispatch abs above-threshold: expected abs_delta-only FAIL above +10, "
            f"got {row.verdict}: {row.reasons}",
        )
        self.assertFalse(
            any("lb=" in reason for reason in row.reasons),
            f"dispatch abs above-threshold: expected abs_delta-only FAIL above +10, "
            f"got {row.verdict}: {row.reasons}",
        )

    def test_dispatches_absolute_delta_boundary_trips_fail_closed(self):
        baseline_benches = _base_benches()
        baseline_benches["decode/dispatches_per_token"] = _bench(
            250.0, 249.0, 250.0, False, "count"
        )
        baseline = _doc(baseline_benches)
        current_benches = _base_benches()
        # disp_abs=10.0 exactly at the +10 boundary, disp_lb=4.00% -- fail-closed ruling.
        current_benches["decode/dispatches_per_token"] = _bench(260.0, 260.0, 260.1, False, "count")
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        row = results["ttft_dispatch"]
        self.assertEqual(
            row.verdict,
            "FAIL",
            f"dispatch abs boundary: expected fail-closed exact +10 FAIL, "
            f"got {row.verdict}: {row.reasons}",
        )
        self.assertTrue(
            any("abs_delta=10.0000 >= 10" in reason for reason in row.reasons),
            f"dispatch abs boundary: expected fail-closed exact +10 FAIL, "
            f"got {row.verdict}: {row.reasons}",
        )
        self.assertFalse(
            any("lb=" in reason for reason in row.reasons),
            f"dispatch abs boundary: expected fail-closed exact +10 FAIL, "
            f"got {row.verdict}: {row.reasons}",
        )

    def test_command_buffers_ceiling_trips_row(self):
        baseline = _doc(_base_benches())
        current_benches = _base_benches()
        current_benches["decode/command_buffers_per_token"] = _bench(2.5, 2.1, 2.9, False, "count")
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        self.assertEqual(results["ttft_dispatch"].verdict, "FAIL")


class QualityRowRegressionTest(unittest.TestCase):
    def test_ppl_delta_f16_trips_only_quality_row(self):
        baseline = _doc(_base_benches())
        current_benches = _base_benches()
        current_benches["quality/ppl_delta/f16"] = _bench(0.01, 0.006, 0.014, False, "ppl_delta")
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        self.assertEqual(results["quality"].verdict, "FAIL")
        for name, r in results.items():
            if name != "quality":
                self.assertEqual(r.verdict, "PASS", f"{name}: {r.reasons}")

    def test_greedy_agreement_below_one_trips_row(self):
        baseline = _doc(_base_benches())
        current_benches = _base_benches()
        current_benches["quality/greedy_agreement"] = _bench(0.98, 0.97, 0.99, True, "ratio")
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        self.assertEqual(results["quality"].verdict, "FAIL")

    def test_topk_exact_below_one_trips_row(self):
        baseline = _doc(_base_benches())
        current_benches = _base_benches()
        current_benches["quality/topk_exact"] = _bench(0.99, 0.98, 1.0, True, "ratio")
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        self.assertEqual(results["quality"].verdict, "FAIL")


class ContentionLayoutRowRegressionTest(unittest.TestCase):
    def test_kv_layout_assertion_failure_trips_only_contention_row(self):
        baseline = _doc(_base_benches())
        current_benches = _base_benches()
        current_benches["runtime/kv_layout_assertion"] = _bench(0.0, 0.0, 0.0, True, "bool01")
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        self.assertEqual(results["contention_layout"].verdict, "FAIL")
        for name, r in results.items():
            if name != "contention_layout":
                self.assertEqual(r.verdict, "PASS", f"{name}: {r.reasons}")

    def test_loss_pp_w4_absolute_delta_trips_row(self):
        baseline = _doc(_base_benches())
        current_benches = _base_benches()
        current_benches["contention/loss_pp/w4"] = _bench(5.0, 4.6, 5.4, False, "pp")
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        self.assertEqual(results["contention_layout"].verdict, "FAIL")

    def test_loss_frac_w10_relative_regression_trips_row(self):
        baseline = _doc(_base_benches())
        current_benches = _base_benches()
        current_benches["contention/loss_frac/w10"] = _bench(0.033, 0.032, 0.034, False, "fraction")
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        self.assertEqual(results["contention_layout"].verdict, "FAIL")


class MissingDataIncompleteTest(unittest.TestCase):
    def test_missing_ttft_row_is_incomplete_not_pass(self):
        baseline = _doc(_base_benches())
        current_benches = _base_benches()
        current_benches["decode/ttft_ms/4096"] = {
            "value": None,
            "ci95_low": None,
            "ci95_high": None,
            "unit": "ms",
            "higher_is_better": False,
        }
        current = _doc(current_benches)
        results = {r.name: r for r in gate.evaluate(current, baseline)}
        self.assertEqual(results["ttft_dispatch"].verdict, "INCOMPLETE")
        # Other rows remain fully evaluated and passing.
        self.assertEqual(results["decode_tok_s_slope_intercept"].verdict, "PASS")
        self.assertEqual(gate.overall_exit_code(list(results.values())), 2)

    def test_missing_row_does_not_mask_a_real_fail_elsewhere(self):
        baseline = _doc(_base_benches())
        current_benches = _base_benches()
        current_benches["decode/ttft_ms/4096"] = {
            "value": None,
            "ci95_low": None,
            "ci95_high": None,
            "unit": "ms",
            "higher_is_better": False,
        }
        current_benches["decode/slope_ms_per_ctx_tok"] = _bench(
            0.0114, 0.0112, 0.0116, False, "ms/ctx_tok"
        )
        current = _doc(current_benches)
        results = list(gate.evaluate(current, baseline))
        self.assertEqual(gate.overall_exit_code(results), 1)


class ArchMismatchTest(unittest.TestCase):
    def test_mismatched_arch_raises(self):
        baseline = _doc(_base_benches())
        current = _doc(_base_benches())
        current["arch"] = "x86_64-linux"
        with self.assertRaises(ValueError):
            gate.evaluate(current, baseline)


class ReportRenderingTest(unittest.TestCase):
    def test_render_report_contains_all_rows(self):
        baseline = _doc(_base_benches())
        current = _doc(_base_benches())
        results = gate.evaluate(current, baseline)
        report = gate.render_report(results)
        for r in results:
            self.assertIn(r.name, report)
            self.assertIn(r.verdict, report)


class NumericMetricValidationTest(unittest.TestCase):
    FIELDS = ("value", "ci95_low", "ci95_high")
    W10 = "contention/loss_frac/w10"
    THROUGHPUT = "decode/tok_s/64"
    FINITE_REPORT = (
        "# ADR-064 GPU/decode gate report\n\n"
        "| Row | Verdict | Reasons |\n"
        "| --- | --- | --- |\n"
        "| decode_tok_s_slope_intercept | PASS | - |\n"
        "| ttft_dispatch | PASS | - |\n"
        "| quality | PASS | - |\n"
        "| contention_layout | PASS | - |\n"
    )

    def _documents(self):
        return {"current": _doc(_base_benches()), "baseline": _doc(_base_benches())}

    def _error_message(self, key, field):
        return f"{key}.{field} must be a finite number in the supported range"

    def _assert_field_rejected(self, key, invalid_values):
        for document in ("current", "baseline"):
            for field in self.FIELDS:
                for label, value in invalid_values:
                    with self.subTest(document=document, field=field, value=label):
                        docs = self._documents()
                        docs[document]["benches"][key][field] = value
                        message = self._error_message(key, field)
                        with self.assertRaisesRegex(ValueError, "^" + re.escape(message) + "$"):
                            gate.evaluate(docs["current"], docs["baseline"])

    def test_nonfinite_fields_rejected_in_both_documents(self):
        self._assert_field_rejected(
            self.W10,
            [("nan", float("nan")), ("positive infinity", float("inf")), ("negative infinity", -float("inf"))],
        )

    def test_boolean_and_string_fields_rejected_in_both_documents(self):
        self._assert_field_rejected(
            self.W10, [("true", True), ("false", False), ("numeric string", "0.02")]
        )

    def test_out_of_range_integer_fields_rejected_in_both_documents(self):
        self._assert_field_rejected(
            self.W10, [("positive huge integer", 10**400), ("negative huge integer", -(10**400))]
        )

    def test_null_or_missing_field_does_not_mask_an_invalid_present_field(self):
        for document in ("current", "baseline"):
            for invalid_field in self.FIELDS:
                for absent_field in self.FIELDS:
                    if invalid_field == absent_field:
                        continue
                    for absent_kind in ("null", "missing"):
                        with self.subTest(
                            document=document, invalid=invalid_field, absent=absent_field, kind=absent_kind
                        ):
                            docs = self._documents()
                            entry = docs[document]["benches"][self.W10]
                            entry[invalid_field] = float("nan")
                            if absent_kind == "null":
                                entry[absent_field] = None
                            else:
                                del entry[absent_field]
                            message = self._error_message(self.W10, invalid_field)
                            with self.assertRaisesRegex(ValueError, "^" + re.escape(message) + "$"):
                                gate.evaluate(docs["current"], docs["baseline"])

    def test_optional_throughput_rejects_invalid_fields_in_both_documents(self):
        self._assert_field_rejected(
            self.THROUGHPUT,
            [
                ("nan", float("nan")),
                ("positive infinity", float("inf")),
                ("negative infinity", -float("inf")),
                ("boolean", True),
                ("numeric string", "100"),
                ("huge integer", 10**400),
            ],
        )

    def test_nan_contention_metric_cannot_pass(self):
        current = _doc(_base_benches())
        baseline = _doc(_base_benches())
        current["benches"]["contention/loss_frac/w10"] = _bench(
            float("nan"), float("nan"), float("nan"), False, "fraction"
        )
        with self.assertRaisesRegex(ValueError, r"^contention/loss_frac/w10\.value must be a finite number in the supported range$"):
            gate.evaluate(current, baseline)

    def test_nan_optional_throughput_metric_cannot_be_skipped(self):
        current = _doc(_base_benches())
        baseline = _doc(_base_benches())
        current["benches"]["decode/tok_s/64"] = _bench(
            float("nan"), float("nan"), float("nan"), True, "tok_s"
        )
        with self.assertRaisesRegex(ValueError, r"^decode/tok_s/64\.value must be a finite number in the supported range$"):
            gate.evaluate(current, baseline)

    def test_finite_fixture_preserves_exact_results_and_report(self):
        docs = self._documents()
        results = gate.evaluate(docs["current"], docs["baseline"])
        self.assertEqual(
            [(row.name, row.verdict, row.reasons) for row in results],
            [
                ("decode_tok_s_slope_intercept", "PASS", []),
                ("ttft_dispatch", "PASS", []),
                ("quality", "PASS", []),
                ("contention_layout", "PASS", []),
            ],
        )
        self.assertEqual(gate.overall_exit_code(results), 0)
        self.assertEqual(gate.render_report(results), self.FINITE_REPORT)

    def test_integer_and_negative_finite_fields_remain_accepted(self):
        docs = self._documents()
        for document in docs.values():
            document["benches"]["runtime/kv_layout_assertion"] = _bench(1, 1, 1, True, "bool01")
            document["benches"]["contention/loss_pp/w4"] = _bench(-2, -3.0, -1.0, False, "pp")
        results = gate.evaluate(docs["current"], docs["baseline"])
        self.assertEqual(gate.overall_exit_code(results), 0)
        self.assertEqual(gate.render_report(results), self.FINITE_REPORT)

    def test_missing_or_null_required_metrics_remain_incomplete(self):
        cases = ["missing entry", "null entry", "all null"]
        cases += [f"{kind} {field}" for kind in ("missing", "null") for field in self.FIELDS]
        for document in ("current", "baseline"):
            for case in cases:
                with self.subTest(document=document, case=case):
                    docs = self._documents()
                    benches = docs[document]["benches"]
                    if case == "missing entry":
                        del benches[self.W10]
                    elif case == "null entry":
                        benches[self.W10] = None
                    elif case == "all null":
                        benches[self.W10] = _bench(None, None, None, False, "fraction")
                    else:
                        kind, field = case.split()
                        if kind == "missing":
                            del benches[self.W10][field]
                        else:
                            benches[self.W10][field] = None
                    results = gate.evaluate(docs["current"], docs["baseline"])
                    self.assertEqual(
                        [(row.name, row.verdict, row.reasons) for row in results],
                        [
                            ("decode_tok_s_slope_intercept", "PASS", []),
                            ("ttft_dispatch", "PASS", []),
                            ("quality", "PASS", []),
                            ("contention_layout", "INCOMPLETE", ["missing: contention/loss_frac/w10"]),
                        ],
                    )
                    self.assertEqual(gate.overall_exit_code(results), 2)

    def test_cli_invalid_fields_preserve_existing_or_absent_report(self):
        for document in ("current", "baseline"):
            for field in self.FIELDS:
                for report_exists in (False, True):
                    with self.subTest(document=document, field=field, report_exists=report_exists):
                        docs = self._documents()
                        docs[document]["benches"][self.W10][field] = float("nan")
                        with tempfile.TemporaryDirectory() as temporary:
                            root = Path(temporary)
                            current_path = root / "current.json"
                            baseline_path = root / "baseline.json"
                            report_path = root / "report.md"
                            current_path.write_text(json.dumps(docs["current"]))
                            baseline_path.write_text(json.dumps(docs["baseline"]))
                            previous = b"previous report must remain unchanged\n"
                            if report_exists:
                                report_path.write_bytes(previous)
                            with (
                                mock.patch.object(sys, "stdout", new_callable=io.StringIO) as stdout,
                                mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
                            ):
                                status = gate.main(
                                    ["--current", str(current_path), "--baseline", str(baseline_path), "--out", str(report_path)]
                                )
                            self.assertEqual(status, 2)
                            self.assertEqual(stdout.getvalue(), "")
                            self.assertEqual(
                                stderr.getvalue(), f"error: {self._error_message(self.W10, field)}\n"
                            )
                            if report_exists:
                                self.assertEqual(report_path.read_bytes(), previous)
                            else:
                                self.assertFalse(report_path.exists())

    def test_cli_finite_inputs_publish_exact_report(self):
        docs = self._documents()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current_path = root / "current.json"
            baseline_path = root / "baseline.json"
            report_path = root / "reports" / "gate.md"
            current_path.write_text(json.dumps(docs["current"]))
            baseline_path.write_text(json.dumps(docs["baseline"]))
            with (
                mock.patch.object(sys, "stdout", new_callable=io.StringIO) as stdout,
                mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
            ):
                status = gate.main(
                    ["--current", str(current_path), "--baseline", str(baseline_path), "--out", str(report_path)]
                )
            self.assertEqual(status, 0)
            self.assertEqual(stdout.getvalue(), self.FINITE_REPORT)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(report_path.read_text(), self.FINITE_REPORT)


if __name__ == "__main__":
    unittest.main()
