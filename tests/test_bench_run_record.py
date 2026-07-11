"""Unit tests for the schema-v2 run-record contract added to
scripts/bench_decode_harness.py (bench-overhaul PR 1, "canonical contract
and policy, in days").

Deterministic fixtures only -- no engine binary, GPU, or heavy lane
required. Loaded by file path (matching test_bench_decode_harness.py's
convention: scripts/ is not a package).

Covers the PR's acceptance row: fixtures prove missing cell / missing
path proof / missing lock receipt / non-finite metric / low valid-n /
post-run threshold change / wrong SHA all fail closed, a v1 observation
still imports/validates unchanged, and the north-star ranking query works
over the shipped `CellRecord` shape without further schema change.

Run with: python3 -m pytest tests/test_bench_run_record.py -v
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_decode_harness.py"
_SPEC = importlib.util.spec_from_file_location("bench_decode_harness", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
harness = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = harness
_SPEC.loader.exec_module(harness)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _provenance(**overrides) -> harness.ProvenanceRecord:
    d = {
        "repo_sha": "a" * 40,
        "candidate_sha": "a" * 40,
        "base_sha": "b" * 40,
        "dirty": False,
        "profile_name": "cpu_smoke_v2",
        "profile_version": 1,
        "profile_sha": "c" * 64,
        "policy_version": 1,
        "policy_sha": "d" * 64,
        "script_sha": "e" * 40,
        "hardware_fingerprint": "Darwin-arm64-M2Max-32GB",
        "collected_at": "2026-07-11T00:00:00+00:00",
        "workflow_run_id": None,
    }
    d.update(overrides)
    return harness.parse_provenance(d)


def _aggregate(**overrides) -> harness.CellAggregate:
    d = {
        "point_estimate": 0.02,
        "ci_low": -0.01,
        "ci_high": 0.05,
        "corrected_lower_bound": 0.01,
        "n_valid": 7,
        "n_invalid": 0,
        "measured_cv": 0.01,
        "required_n": 7,
    }
    d.update(overrides)
    return harness.parse_cell_aggregate(d)


def _phase_events(n_tokens: int = 2) -> tuple[harness.PhaseEvent, ...]:
    """A valid, canonically-ordered load->prefill->decode phase-event
    sequence -- the default every `_cell_record` fixture carries unless a
    test is specifically exercising `_validate_phase_sequence`."""
    events = [
        harness.parse_phase_event({"name": "load_start", "monotonic_ns": 0}),
        harness.parse_phase_event({"name": "backend_ready", "monotonic_ns": 10}),
        harness.parse_phase_event({"name": "prefill_start", "monotonic_ns": 20}),
        harness.parse_phase_event({"name": "prefill_end", "monotonic_ns": 30}),
    ]
    for i in range(n_tokens):
        events.append(
            harness.parse_phase_event({"name": "token_available", "monotonic_ns": 40 + i * 5, "token_index": i + 1})
        )
    return tuple(events)


def _resource_sample(**overrides) -> harness.ResourceSample:
    d = {"monotonic_ns": 5, "phys_footprint_bytes": 1_000_000}
    d.update(overrides)
    return harness.parse_resource_sample(d)


def _cell_record(**overrides) -> harness.CellRecord:
    return harness.CellRecord(
        cell_id=overrides.pop("cell_id", "decode:qwen3.5-small:f16:cpu:1024"),
        metric_family=overrides.pop("metric_family", "decode"),
        metric_name=overrides.pop("metric_name", "decode_tok_s"),
        path=overrides.pop("path", "decode"),
        model_tier=overrides.pop("model_tier", "qwen3.5-small"),
        quant_tier=overrides.pop("quant_tier", "f16"),
        device=overrides.pop("device", "cpu"),
        context_point=overrides.pop("context_point", 1024),
        aggregate=overrides.pop("aggregate", _aggregate()),
        verdict=overrides.pop("verdict", "PASS"),
        unsupported_reason=overrides.pop("unsupported_reason", None),
        path_proof=overrides.pop("path_proof", ()),
        lock_receipts=overrides.pop("lock_receipts", ()),
        phase_events=overrides.pop("phase_events", _phase_events()),
        resource_samples=overrides.pop("resource_samples", ()),
        provenance=overrides.pop("provenance", _provenance()),
        order_balance=overrides.pop("order_balance", (4, 3)),
        raw_artifact_digest=overrides.pop("raw_artifact_digest", "f" * 64),
    )


def _metal_lock(**overrides) -> harness.LockReceipt:
    d = {
        "lock_name": "metal-gpu",
        "acquired_at": "2026-07-11T00:00:00+00:00",
        "released_at": "2026-07-11T00:05:00+00:00",
        "held_continuously": True,
        "wait_seconds": 0.5,
    }
    d.update(overrides)
    return harness.parse_lock_receipt(d)


def _heavy_lane_lock(**overrides) -> harness.LockReceipt:
    d = {
        "lock_name": "heavy-lane",
        "acquired_at": "2026-07-11T00:00:00+00:00",
        "released_at": "2026-07-11T00:05:00+00:00",
        "held_continuously": True,
        "wait_seconds": 0.2,
    }
    d.update(overrides)
    return harness.parse_lock_receipt(d)


def _metal_locks() -> tuple[harness.LockReceipt, ...]:
    """Both continuously-held receipts a device=metal cell needs: the
    outer heavy-lane receipt AND the metal-gpu receipt."""
    return (_heavy_lane_lock(), _metal_lock())


# --------------------------------------------------------------------------
# Phase events / resource samples / lock receipts / provenance parsing
# --------------------------------------------------------------------------


class PhaseEventParseTest(unittest.TestCase):
    def test_valid_load_start(self):
        ev = harness.parse_phase_event({"name": "load_start", "monotonic_ns": 0})
        self.assertEqual(ev.name, "load_start")

    def test_token_available_requires_token_index(self):
        with self.assertRaisesRegex(harness.RunRecordValidationError, "token_index"):
            harness.parse_phase_event({"name": "token_available", "monotonic_ns": 100})

    def test_non_token_event_rejects_token_index(self):
        with self.assertRaisesRegex(harness.RunRecordValidationError, "token_index"):
            harness.parse_phase_event({"name": "load_start", "monotonic_ns": 0, "token_index": 1})

    def test_unknown_name_rejected(self):
        with self.assertRaisesRegex(harness.RunRecordValidationError, "name"):
            harness.parse_phase_event({"name": "bogus", "monotonic_ns": 0})

    def test_negative_monotonic_ns_rejected(self):
        with self.assertRaisesRegex(harness.RunRecordValidationError, "monotonic_ns"):
            harness.parse_phase_event({"name": "load_start", "monotonic_ns": -1})


class ResourceSampleParseTest(unittest.TestCase):
    def test_valid_sample(self):
        s = harness.parse_resource_sample({"monotonic_ns": 10, "phys_footprint_bytes": 1000})
        self.assertIsNone(s.phase)

    def test_invalid_phase_rejected(self):
        with self.assertRaisesRegex(harness.RunRecordValidationError, "phase"):
            harness.parse_resource_sample({"monotonic_ns": 10, "phys_footprint_bytes": 1000, "phase": "bogus"})


class LockReceiptParseTest(unittest.TestCase):
    def test_valid_receipt(self):
        lr = _metal_lock()
        self.assertTrue(lr.held_continuously)

    def test_empty_lock_name_rejected(self):
        with self.assertRaisesRegex(harness.RunRecordValidationError, "lock_name"):
            harness.parse_lock_receipt(
                {
                    "lock_name": "",
                    "acquired_at": "x",
                    "released_at": None,
                    "held_continuously": True,
                    "wait_seconds": 0,
                }
            )


class ProvenanceParseTest(unittest.TestCase):
    def test_valid_provenance(self):
        p = _provenance()
        self.assertEqual(p.policy_version, 1)

    def test_missing_field_rejected(self):
        with self.assertRaisesRegex(harness.RunRecordValidationError, "missing required field"):
            harness.parse_provenance({"repo_sha": "a"})


# --------------------------------------------------------------------------
# Fail-closed acceptance fixtures (the PR's explicit acceptance row)
# --------------------------------------------------------------------------


class MissingPathProofTest(unittest.TestCase):
    def test_metal_cell_without_path_proof_fails_closed(self):
        record = _cell_record(
            device="metal", path_proof=(), lock_receipts=_metal_locks(), resource_samples=(_resource_sample(),)
        )
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*path_proof"):
            harness.validate_run_record(record)

    def test_metal_cell_with_path_proof_passes(self):
        record = _cell_record(
            device="metal",
            path_proof=("attention_dispatch_counter>0",),
            lock_receipts=_metal_locks(),
            resource_samples=(_resource_sample(),),
        )
        harness.validate_run_record(record)  # must not raise

    def test_cpu_cell_needs_no_path_proof(self):
        record = _cell_record(device="cpu", path_proof=())
        harness.validate_run_record(record)  # must not raise


class MissingLockReceiptTest(unittest.TestCase):
    def test_metal_cell_without_lock_receipt_fails_closed(self):
        record = _cell_record(device="metal", path_proof=("proof",), lock_receipts=())
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*lock_receipt"):
            harness.validate_run_record(record)

    def test_metal_cell_with_released_not_held_continuously_fails_closed(self):
        bad_lock = _metal_lock(held_continuously=False)
        record = _cell_record(device="metal", path_proof=("proof",), lock_receipts=(_heavy_lane_lock(), bad_lock))
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*lock_receipt"):
            harness.validate_run_record(record)

    def test_metal_cell_with_wrong_lock_name_fails_closed(self):
        wrong_lock = _metal_lock(lock_name="something-else")
        record = _cell_record(device="metal", path_proof=("proof",), lock_receipts=(_heavy_lane_lock(), wrong_lock))
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*lock_receipt"):
            harness.validate_run_record(record)

    def test_metal_cell_with_only_metal_gpu_lock_fails_closed(self):
        """A Metal record with ONLY the metal-gpu
        receipt (no heavy-lane) must reject -- both locks are required."""
        record = _cell_record(
            device="metal", path_proof=("proof",), lock_receipts=(_metal_lock(),), resource_samples=(_resource_sample(),)
        )
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*heavy-lane"):
            harness.validate_run_record(record)

    def test_metal_cell_with_only_heavy_lane_lock_fails_closed(self):
        """A Metal record with ONLY the heavy-lane
        receipt (no metal-gpu) must reject -- both locks are required."""
        record = _cell_record(
            device="metal", path_proof=("proof",), lock_receipts=(_heavy_lane_lock(),), resource_samples=(_resource_sample(),)
        )
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*metal-gpu"):
            harness.validate_run_record(record)

    def test_metal_cell_without_resource_samples_fails_closed(self):
        """A non-unsupported Metal record with both
        locks but empty resource_samples must reject -- the
        contention/memory-footprint proof is missing."""
        record = _cell_record(
            device="metal", path_proof=("proof",), lock_receipts=_metal_locks(), resource_samples=()
        )
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*resource_samples"):
            harness.validate_run_record(record)

    def test_metal_cell_with_both_locks_and_samples_passes(self):
        record = _cell_record(
            device="metal", path_proof=("proof",), lock_receipts=_metal_locks(), resource_samples=(_resource_sample(),)
        )
        harness.validate_run_record(record)  # must not raise


class PhaseSequenceTest(unittest.TestCase):
    """Every non-unsupported record must carry a
    validated, correctly-ordered phase-event sequence."""

    def test_empty_phase_events_fails_closed(self):
        record = _cell_record(phase_events=())
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*phase_events"):
            harness.validate_run_record(record)

    def test_missing_required_phase_fails_closed(self):
        events = tuple(ev for ev in _phase_events() if ev.name != "prefill_end")
        record = _cell_record(phase_events=events)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*missing required phase"):
            harness.validate_run_record(record)

    def test_no_token_available_fails_closed(self):
        events = tuple(ev for ev in _phase_events() if ev.name != "token_available")
        record = _cell_record(phase_events=events)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*token_available"):
            harness.validate_run_record(record)

    def test_out_of_order_phases_fails_closed(self):
        events = (
            harness.parse_phase_event({"name": "prefill_start", "monotonic_ns": 0}),
            harness.parse_phase_event({"name": "load_start", "monotonic_ns": 10}),
            harness.parse_phase_event({"name": "backend_ready", "monotonic_ns": 20}),
            harness.parse_phase_event({"name": "prefill_end", "monotonic_ns": 30}),
            harness.parse_phase_event({"name": "token_available", "monotonic_ns": 40, "token_index": 1}),
        )
        record = _cell_record(phase_events=events)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*out of order"):
            harness.validate_run_record(record)

    def test_monotonic_ns_backwards_fails_closed(self):
        events = (
            harness.parse_phase_event({"name": "load_start", "monotonic_ns": 100}),
            harness.parse_phase_event({"name": "backend_ready", "monotonic_ns": 10}),
            harness.parse_phase_event({"name": "prefill_start", "monotonic_ns": 20}),
            harness.parse_phase_event({"name": "prefill_end", "monotonic_ns": 30}),
            harness.parse_phase_event({"name": "token_available", "monotonic_ns": 40, "token_index": 1}),
        )
        record = _cell_record(phase_events=events)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*backwards"):
            harness.validate_run_record(record)

    def test_non_increasing_token_index_fails_closed(self):
        events = _phase_events()[:4] + (
            harness.parse_phase_event({"name": "token_available", "monotonic_ns": 40, "token_index": 2}),
            harness.parse_phase_event({"name": "token_available", "monotonic_ns": 45, "token_index": 2}),
        )
        record = _cell_record(phase_events=events)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*token_index"):
            harness.validate_run_record(record)

    def test_unsupported_cell_exempt_from_phase_sequence(self):
        agg = _aggregate(measured_cv=None, required_n=None, n_valid=0)
        record = _cell_record(phase_events=(), aggregate=agg, verdict="unsupported", unsupported_reason="no checkpoint yet")
        harness.validate_run_record(record)  # must not raise

    def test_valid_sequence_passes(self):
        record = _cell_record(phase_events=_phase_events())
        harness.validate_run_record(record)  # must not raise


class NonFiniteMetricTest(unittest.TestCase):
    def test_nan_point_estimate_rejected_at_parse(self):
        with self.assertRaisesRegex(harness.RunRecordValidationError, "not finite"):
            harness.parse_cell_aggregate(
                {
                    "point_estimate": float("nan"),
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "n_valid": 7,
                    "n_invalid": 0,
                }
            )

    def test_inf_ci_high_rejected_at_parse(self):
        with self.assertRaisesRegex(harness.RunRecordValidationError, "not finite"):
            harness.parse_cell_aggregate(
                {
                    "point_estimate": 0.0,
                    "ci_low": 0.0,
                    "ci_high": float("inf"),
                    "n_valid": 7,
                    "n_invalid": 0,
                }
            )

    def test_non_finite_caught_at_validate_run_record_too(self):
        # bypass parse_cell_aggregate to prove validate_run_record itself
        # also fails closed on a hand-built non-finite aggregate
        agg = harness.CellAggregate(
            point_estimate=float("nan"), ci_low=0.0, ci_high=0.0,
            corrected_lower_bound=None, n_valid=7, n_invalid=0, measured_cv=0.01, required_n=7,
        )
        record = _cell_record(aggregate=agg)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*not finite"):
            harness.validate_run_record(record)


class LowValidNTest(unittest.TestCase):
    def test_below_required_n_fails_closed(self):
        agg = _aggregate(n_valid=5, required_n=7, measured_cv=0.01)
        record = _cell_record(aggregate=agg)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*n too small"):
            harness.validate_run_record(record)

    def test_at_required_n_passes(self):
        agg = _aggregate(n_valid=7, required_n=7, measured_cv=0.01)
        record = _cell_record(aggregate=agg)
        harness.validate_run_record(record)  # must not raise

    def test_missing_measured_cv_fails_closed(self):
        """Correction 1: a cell with no measured_cv on record can never be
        validated -- required_n cannot be trusted without it."""
        agg = _aggregate(measured_cv=None, required_n=None)
        record = _cell_record(aggregate=agg)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*measured_cv"):
            harness.validate_run_record(record)

    def test_unsupported_cell_exempt_from_cv_n_checks(self):
        agg = _aggregate(measured_cv=None, required_n=None, n_valid=0)
        record = _cell_record(aggregate=agg, verdict="unsupported", unsupported_reason="no checkpoint yet")
        harness.validate_run_record(record)  # must not raise


class SubmitterControlledRequiredNTest(unittest.TestCase):
    """required_n must be RE-DERIVED from the registered
    per-metric policy and the measured CV -- never trusted as optional,
    submitter-supplied record metadata."""

    def test_cv_three_percent_with_required_n_none_fails_closed(self):
        # decode/decode_tok_s is class A; CV 3% -> policy requires n=25,
        # not the n=7 band. required_n=None can never match.
        agg = _aggregate(n_valid=0, required_n=None, measured_cv=0.03)
        record = _cell_record(aggregate=agg)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*required_n"):
            harness.validate_run_record(record)

    def test_forged_required_n_seven_at_cv_three_percent_fails_closed(self):
        # a submitter claims required_n=7 (matching a low, easy n_valid)
        # at CV 3%, where the registered policy actually derives n=25.
        agg = _aggregate(n_valid=7, required_n=7, measured_cv=0.03)
        record = _cell_record(aggregate=agg)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*required_n"):
            harness.validate_run_record(record)

    def test_embed_batch_p95_class_b_with_seven_pairs_fails_closed(self):
        # embed.batch_p95 is registered class B (min nine pairs at low
        # CV); a family-name heuristic that treated it as class A
        # silently accepted seven (the concrete regression this guards).
        agg = _aggregate(n_valid=7, required_n=7, measured_cv=0.01)
        record = _cell_record(metric_family="embed", metric_name="batch_p95", aggregate=agg)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*required_n"):
            harness.validate_run_record(record)

    def test_embed_batch_p95_class_b_with_nine_pairs_passes(self):
        agg = _aggregate(n_valid=9, required_n=9, measured_cv=0.01)
        record = _cell_record(metric_family="embed", metric_name="batch_p95", aggregate=agg)
        harness.validate_run_record(record)  # must not raise

    def test_decode_at_cv_three_percent_with_correct_required_n_passes(self):
        agg = _aggregate(n_valid=25, required_n=25, measured_cv=0.03)
        record = _cell_record(aggregate=agg)
        harness.validate_run_record(record)  # must not raise

    def test_class_c_metric_exempt_from_required_n_derivation(self):
        # memory.model_load_time is registered class C (informational
        # only, never gates a required cell) -- no required_n derivation
        # applies, so a null required_n on a low n_valid does not reject.
        agg = _aggregate(n_valid=1, required_n=None, measured_cv=0.20)
        record = _cell_record(metric_family="memory", metric_name="model_load_time", aggregate=agg)
        harness.validate_run_record(record)  # must not raise

    def test_unregistered_metric_fails_closed(self):
        agg = _aggregate(n_valid=7, required_n=7, measured_cv=0.01)
        record = _cell_record(metric_family="decode", metric_name="not_a_registered_metric", aggregate=agg)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL"):
            harness.validate_run_record(record)


class PostRunThresholdChangeTest(unittest.TestCase):
    def test_policy_sha_mismatch_fails_closed(self):
        record = _cell_record(provenance=_provenance(policy_sha="d" * 64))
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*policy_sha"):
            harness.validate_run_record(record, current_policy_sha="different" + "0" * 56)

    def test_matching_policy_sha_passes(self):
        record = _cell_record(provenance=_provenance(policy_sha="d" * 64))
        harness.validate_run_record(record, current_policy_sha="d" * 64)  # must not raise

    def test_no_current_policy_sha_given_skips_check(self):
        record = _cell_record(provenance=_provenance(policy_sha="d" * 64))
        harness.validate_run_record(record)  # must not raise (no current_policy_sha supplied)


class WrongShaTest(unittest.TestCase):
    def test_candidate_sha_mismatch_fails_closed(self):
        record = _cell_record(provenance=_provenance(candidate_sha="a" * 40))
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL.*SHA mismatch"):
            harness.validate_run_record(record, expected_repo_sha="b" * 40)

    def test_matching_sha_passes(self):
        record = _cell_record(provenance=_provenance(candidate_sha="a" * 40))
        harness.validate_run_record(record, expected_repo_sha="a" * 40)  # must not raise


class MissingCellRegistryTest(unittest.TestCase):
    def test_missing_expected_cell_fails_closed(self):
        groups = [
            harness.ExpectedCellGroup(
                name="decode_cpu_mini",
                path="decode",
                metric_family="decode",
                device="cpu",
                model_tiers=("qwen3.5-small",),
                quant_tiers=("f16",),
                context_points=(512, 1024),
                required_in=("hosted_pr_smoke",),
                anchor_quant_tiers=("f16",),
                anchor_context_points=(512, 1024),
                anchor_required_in=("hosted_pr_smoke",),
            )
        ]
        present = [_cell_record(cell_id="decode:qwen3.5-small:f16:cpu:512", context_point=512)]
        with self.assertRaisesRegex(harness.RunRecordValidationError, "INFRA-FAIL: expected cell"):
            harness.validate_registry_coverage(present, groups, "hosted_pr_smoke")

    def test_unsupported_cell_counts_as_present(self):
        groups = [
            harness.ExpectedCellGroup(
                name="decode_cpu_mini",
                path="decode",
                metric_family="decode",
                device="cpu",
                model_tiers=("qwen3.5-small",),
                quant_tiers=("f16",),
                context_points=(512,),
                required_in=("hosted_pr_smoke",),
                anchor_quant_tiers=("f16",),
                anchor_context_points=(512,),
                anchor_required_in=("hosted_pr_smoke",),
            )
        ]
        present = [
            _cell_record(
                cell_id="decode:qwen3.5-small:f16:cpu:512",
                context_point=512,
                verdict="unsupported",
                unsupported_reason="checkpoint unavailable on this runner",
            )
        ]
        harness.validate_registry_coverage(present, groups, "hosted_pr_smoke")  # must not raise

    def test_full_coverage_passes(self):
        groups = [
            harness.ExpectedCellGroup(
                name="decode_cpu_mini",
                path="decode",
                metric_family="decode",
                device="cpu",
                model_tiers=("qwen3.5-small",),
                quant_tiers=("f16",),
                context_points=(512, 1024, 2048),
                required_in=("hosted_pr_smoke",),
                anchor_quant_tiers=("f16",),
                anchor_context_points=(512, 1024, 2048),
                anchor_required_in=("hosted_pr_smoke",),
            )
        ]
        present = [
            _cell_record(cell_id="decode:qwen3.5-small:f16:cpu:512", context_point=512),
            _cell_record(cell_id="decode:qwen3.5-small:f16:cpu:1024", context_point=1024),
            _cell_record(cell_id="decode:qwen3.5-small:f16:cpu:2048", context_point=2048),
        ]
        harness.validate_registry_coverage(present, groups, "hosted_pr_smoke")  # must not raise

    def test_shipped_registry_loads_and_expands_without_error(self):
        groups = harness.load_expected_cells()
        self.assertGreaterEqual(len(groups), 4)
        ids = harness.expected_cell_ids_for_cadence(groups, "hosted_pr_smoke")
        self.assertIn("decode:qwen3.5-small:f16:cpu:1024", ids)
        self.assertNotIn("decode:qwen3.5-small:q8:cpu:1024", ids)  # q8 is not an anchor quant

    def test_reserved_cadence_never_matches_a_live_cadence(self):
        groups = harness.load_expected_cells()
        reserved = [g for g in groups if g.name == "streaming_moe_reserved"][0]
        expanded = harness.expand_cell_group(reserved)
        for cadences in expanded.values():
            self.assertEqual(cadences, ("reserved",))


# --------------------------------------------------------------------------
# Correction 3: promotion record honesty
# --------------------------------------------------------------------------


class PromotionRecordTest(unittest.TestCase):
    def _promotion(self, **overrides):
        d = {
            "cell_id": "decode:qwen3.5-small:f16:cpu:1024",
            "policy_version": 1,
            "null_sessions": 20,
            "failures": 0,
            "cp_bound_95": 0.14,  # rounded UP from the true ~0.1391 bound -- honest, not tighter
            "mainline_sessions": 10,
            "invalid_pair_replacements": 0,
        }
        d.update(overrides)
        return harness.parse_promotion_record(d)

    def test_honest_bound_passes(self):
        record = self._promotion()
        harness.validate_promotion_record(record)  # must not raise

    def test_asserting_tighter_than_true_bound_fails_closed(self):
        # claims <1% from only 20 sessions/0 failures -- the true bound is ~13.91%
        record = self._promotion(cp_bound_95=0.009)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "tighter than"):
            harness.validate_promotion_record(record)

    def test_below_twenty_sessions_fails_closed(self):
        record = self._promotion(null_sessions=10, failures=0, cp_bound_95=0.5)
        with self.assertRaisesRegex(harness.RunRecordValidationError, ">= 20"):
            harness.validate_promotion_record(record)

    def test_upfront_298_sessions_not_required(self):
        """Correction 3: promotion is gated on >=20 sessions + an honest
        bound, NOT on reaching the ~298 sessions that would independently
        prove <1% at 95% confidence from zero failures."""
        record = self._promotion(null_sessions=20, failures=0, cp_bound_95=0.14)
        harness.validate_promotion_record(record)  # must not raise despite far fewer than 298 sessions

    def test_bound_tightens_as_sessions_accrue(self):
        early = self._promotion(null_sessions=20, failures=0, cp_bound_95=0.14)
        later = self._promotion(null_sessions=300, failures=0, cp_bound_95=0.01)
        harness.validate_promotion_record(early)
        harness.validate_promotion_record(later)
        self.assertGreater(early.cp_bound_95, later.cp_bound_95)

    def test_failures_exceeding_sessions_rejected_at_parse(self):
        with self.assertRaisesRegex(harness.RunRecordValidationError, "cannot exceed"):
            harness.parse_promotion_record(
                {
                    "cell_id": "x",
                    "policy_version": 1,
                    "null_sessions": 5,
                    "failures": 6,
                    "cp_bound_95": 0.5,
                    "mainline_sessions": 0,
                    "invalid_pair_replacements": 0,
                }
            )

    def test_missing_invalid_pair_replacements_rejected_at_parse(self):
        with self.assertRaisesRegex(harness.RunRecordValidationError, "missing required field"):
            harness.parse_promotion_record(
                {
                    "cell_id": "x",
                    "policy_version": 1,
                    "null_sessions": 20,
                    "failures": 0,
                    "cp_bound_95": 0.14,
                    "mainline_sessions": 10,
                }
            )


class PromotionMainlineSessionsTest(unittest.TestCase):
    """An earlier revision accepted a 20-null,
    0-mainline-session promotion because `mainline_sessions` was never
    examined. The policy registers `min_mainline_sessions = 10`."""

    def _promotion(self, **overrides):
        d = {
            "cell_id": "decode:qwen3.5-small:f16:cpu:1024",
            "policy_version": 1,
            "null_sessions": 20,
            "failures": 0,
            "cp_bound_95": 0.14,
            "mainline_sessions": 10,
            "invalid_pair_replacements": 0,
        }
        d.update(overrides)
        return harness.parse_promotion_record(d)

    def test_zero_mainline_sessions_fails_closed(self):
        record = self._promotion(mainline_sessions=0)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "mainline sessions"):
            harness.validate_promotion_record(record)

    def test_nine_mainline_sessions_fails_closed(self):
        record = self._promotion(mainline_sessions=9)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "mainline sessions"):
            harness.validate_promotion_record(record)

    def test_ten_mainline_sessions_passes(self):
        record = self._promotion(mainline_sessions=10)
        harness.validate_promotion_record(record)  # must not raise


class PromotionReplacementCapTest(unittest.TestCase):
    """The registered replacement cap
    (`max_invalid_pair_replacements = 3`) must be enforced fail-closed,
    and a policy that omits the cap must reject rather than silently
    default."""

    def _promotion(self, **overrides):
        d = {
            "cell_id": "decode:qwen3.5-small:f16:cpu:1024",
            "policy_version": 1,
            "null_sessions": 20,
            "failures": 0,
            "cp_bound_95": 0.14,
            "mainline_sessions": 10,
            "invalid_pair_replacements": 0,
        }
        d.update(overrides)
        return harness.parse_promotion_record(d)

    def test_within_cap_passes(self):
        record = self._promotion(invalid_pair_replacements=3)
        harness.validate_promotion_record(record)  # must not raise

    def test_exceeding_cap_fails_closed(self):
        record = self._promotion(invalid_pair_replacements=4)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "invalid_pair_replacements"):
            harness.validate_promotion_record(record)

    def test_missing_cap_in_policy_fails_closed(self):
        doc = harness.bench_gate_math.load_policy()
        broken_promotion = {k: v for k, v in doc["promotion"].items() if k != "max_invalid_pair_replacements"}
        broken_policy = {**doc, "promotion": broken_promotion}
        record = self._promotion(invalid_pair_replacements=0)
        with self.assertRaisesRegex(harness.RunRecordValidationError, "max_invalid_pair_replacements"):
            harness.validate_promotion_record(record, policy=broken_policy)


# --------------------------------------------------------------------------
# North-star ranking query
# --------------------------------------------------------------------------


class RankCellsByGapTest(unittest.TestCase):
    def test_ranks_by_descending_absolute_gap(self):
        records = [
            _cell_record(cell_id="decode:a:f16:cpu:512", aggregate=_aggregate(point_estimate=0.02)),
            _cell_record(cell_id="decode:a:f16:cpu:1024", aggregate=_aggregate(point_estimate=-0.15)),
            _cell_record(cell_id="decode:a:f16:cpu:2048", aggregate=_aggregate(point_estimate=0.08)),
        ]
        ranked = harness.rank_cells_by_gap(records)
        self.assertEqual([r.cell_id for r in ranked], [
            "decode:a:f16:cpu:1024",  # |−0.15| largest
            "decode:a:f16:cpu:2048",  # |0.08|
            "decode:a:f16:cpu:512",   # |0.02| smallest
        ])

    def test_top_n_limits_output(self):
        records = [
            _cell_record(cell_id="c1", aggregate=_aggregate(point_estimate=0.5)),
            _cell_record(cell_id="c2", aggregate=_aggregate(point_estimate=0.3)),
            _cell_record(cell_id="c3", aggregate=_aggregate(point_estimate=0.1)),
        ]
        ranked = harness.rank_cells_by_gap(records, top_n=2)
        self.assertEqual([r.cell_id for r in ranked], ["c1", "c2"])

    def test_unsupported_cells_excluded(self):
        records = [
            _cell_record(cell_id="c1", aggregate=_aggregate(point_estimate=0.5)),
            _cell_record(
                cell_id="c2",
                verdict="unsupported",
                unsupported_reason="no checkpoint",
                aggregate=_aggregate(point_estimate=999.0, measured_cv=None, required_n=None, n_valid=0),
            ),
        ]
        ranked = harness.rank_cells_by_gap(records)
        self.assertEqual([r.cell_id for r in ranked], ["c1"])

    def test_ties_broken_by_cell_id(self):
        records = [
            _cell_record(cell_id="zzz", aggregate=_aggregate(point_estimate=0.1)),
            _cell_record(cell_id="aaa", aggregate=_aggregate(point_estimate=-0.1)),
        ]
        ranked = harness.rank_cells_by_gap(records)
        self.assertEqual([r.cell_id for r in ranked], ["aaa", "zzz"])

    def test_ranking_is_groupable_by_metric_context_quant_without_new_fields(self):
        """The north-star requirement: cell records must make 'largest
        stable gaps by metric/context/quant' queryable WITHOUT a later
        schema change. Prove the shape supports a grouped-then-ranked
        query using only fields CellRecord already carries."""
        records = [
            _cell_record(
                cell_id="decode:a:f16:cpu:512", metric_name="decode_tok_s", quant_tier="f16",
                context_point=512, aggregate=_aggregate(point_estimate=0.09),
            ),
            _cell_record(
                cell_id="decode:a:q4:cpu:512", metric_name="decode_tok_s", quant_tier="q4",
                context_point=512, aggregate=_aggregate(point_estimate=0.02),
            ),
            _cell_record(
                cell_id="decode:a:f16:cpu:1024", metric_name="decode_tok_s", quant_tier="f16",
                context_point=1024, aggregate=_aggregate(point_estimate=0.15),
            ),
            _cell_record(
                cell_id="embed:a:f16:cpu:512", metric_family="embed", metric_name="texts_s",
                path="embed_encode", quant_tier="f16", context_point=512,
                aggregate=_aggregate(point_estimate=0.30),
            ),
        ]
        # group by (metric_name, quant_tier) -- a query shape a downstream
        # dashboard/mining tool would run -- then rank within each group.
        groups: dict[tuple[str, str], list[harness.CellRecord]] = {}
        for r in records:
            groups.setdefault((r.metric_name, r.quant_tier), []).append(r)
        self.assertEqual(len(groups), 3)  # (decode_tok_s,f16) (decode_tok_s,q4) (texts_s,f16)
        top_decode_f16 = harness.rank_cells_by_gap(groups[("decode_tok_s", "f16")], top_n=1)
        self.assertEqual(top_decode_f16[0].cell_id, "decode:a:f16:cpu:1024")
        overall_top = harness.rank_cells_by_gap(records, top_n=1)
        self.assertEqual(overall_top[0].cell_id, "embed:a:f16:cpu:512")


# --------------------------------------------------------------------------
# v1 import/readability regression (v1 must remain readable, unchanged)
# --------------------------------------------------------------------------


class V1StillReadableTest(unittest.TestCase):
    def test_v1_observation_still_validates(self):
        row = {
            "schema_version": harness.SCHEMA_VERSION,
            "git_sha": "deadbeefcafebabe",
            "profile": "legacy_v1_profile",
            "engine": "lattice",
            "engine_version": "1.0.0",
            "model": "qwen3.5-0.8b",
            "quantization": "q8",
            "prompt_hash": "abc123",
            "requested_prompt_tokens": 12,
            "actual_prompt_tokens": 12,
            "requested_completion_tokens": 32,
            "actual_completion_tokens": 32,
            "warmup": False,
            "run_index": 1,
            "order_index": 0,
            "elapsed_ns": 1_000_000,
            "engine_native_ns": 900_000,
            "hardware_id": "Darwin-arm64-testhost",
            "timestamp": "2026-07-10T00:00:00+00:00",
        }
        harness.validate_observation(row)  # must not raise -- v1 contract unchanged by v2 additions

    def test_v1_schema_version_constant_unchanged(self):
        # v2 additions must be additive: v1's SCHEMA_VERSION stays 1, a
        # separate RUN_RECORD_SCHEMA_VERSION governs the new record kinds.
        self.assertEqual(harness.SCHEMA_VERSION, 1)
        self.assertEqual(harness.RUN_RECORD_SCHEMA_VERSION, 2)


if __name__ == "__main__":
    unittest.main()
