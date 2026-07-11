"""Unit tests for scripts/bench_decode_harness.py (issue #813).

Deterministic fake-adapter tests and raw-schema validation only -- no engine
binary is required to run this file, matching the harness-core landing gate.

Run with: python3 -m unittest tests/test_bench_decode_harness.py -v
(or `python3 -m pytest tests/test_bench_decode_harness.py -v` -- no
pytest-only features are used).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
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


def _valid_observation(**overrides) -> dict:
    row = {
        "schema_version": harness.SCHEMA_VERSION,
        "git_sha": "deadbeefcafebabe",
        "profile": "unit_test",
        "engine": "fake",
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
    row.update(overrides)
    return row


@dataclass
class _FakeClock:
    """Deterministic monotonic clock: each call returns start + step*calls."""

    step_ns: int = 1_000_000
    start_ns: int = 0
    _calls: int = 0

    def __call__(self) -> int:
        value = self.start_ns + self.step_ns * self._calls
        self._calls += 1
        return value


class _FakeAdapter:
    """Deterministic fake engine adapter: fixed tok/s, exact requested token counts."""

    def __init__(self, engine_version: str = "fake-1.0", native_ns_per_token: int | None = None):
        self.engine_version = engine_version
        self.native_ns_per_token = native_ns_per_token
        self.calls: list[tuple[int, bool]] = []

    def run(self, *, prompt: str, n_tokens: int, warmup: bool) -> "harness.AdapterRunResult":
        self.calls.append((n_tokens, warmup))
        native_ns = self.native_ns_per_token * n_tokens if self.native_ns_per_token else None
        return harness.AdapterRunResult(
            actual_completion_tokens=n_tokens,
            actual_prompt_tokens=len(prompt.split()),
            native_ns=native_ns,
            engine_version=self.engine_version,
        )


def _profile(**overrides) -> "harness.ProfileConfig":
    raw = {
        "description": "unit test profile",
        "windows": [32, 256],
        "warmup_repeats": 0,
        "measured_repeats": 3,
        "engines": ["fake"],
        "prompt": "the quick brown fox",
        "model": "qwen3.5-0.8b",
        "quantization": "q8",
    }
    raw.update(overrides)
    return harness._parse_profile("unit_test", raw)


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------


class ValidateObservationTest(unittest.TestCase):
    def test_valid_observation_passes(self):
        harness.validate_observation(_valid_observation())  # must not raise

    def test_nullable_fields_accept_none(self):
        harness.validate_observation(
            _valid_observation(
                requested_prompt_tokens=None, actual_prompt_tokens=None, engine_native_ns=None
            )
        )

    def test_missing_field_rejected(self):
        row = _valid_observation()
        del row["engine"]
        with self.assertRaisesRegex(harness.ObservationValidationError, "missing required field"):
            harness.validate_observation(row)

    def test_unexpected_field_rejected(self):
        row = _valid_observation(extra_field="nope")
        with self.assertRaisesRegex(harness.ObservationValidationError, "unexpected field"):
            harness.validate_observation(row)

    def test_wrong_schema_version_rejected(self):
        row = _valid_observation(schema_version=999)
        with self.assertRaisesRegex(harness.ObservationValidationError, "schema_version"):
            harness.validate_observation(row)

    def test_bool_rejected_where_int_expected(self):
        row = _valid_observation(run_index=True)
        with self.assertRaisesRegex(harness.ObservationValidationError, "run_index"):
            harness.validate_observation(row)

    def test_int_rejected_where_bool_expected(self):
        row = _valid_observation(warmup=1)
        with self.assertRaisesRegex(harness.ObservationValidationError, "warmup"):
            harness.validate_observation(row)

    def test_negative_elapsed_ns_rejected(self):
        row = _valid_observation(elapsed_ns=-1)
        with self.assertRaisesRegex(harness.ObservationValidationError, "elapsed_ns"):
            harness.validate_observation(row)

    def test_run_index_must_be_at_least_one(self):
        row = _valid_observation(run_index=0)
        with self.assertRaisesRegex(harness.ObservationValidationError, "run_index"):
            harness.validate_observation(row)

    def test_empty_string_field_rejected(self):
        row = _valid_observation(engine="")
        with self.assertRaisesRegex(harness.ObservationValidationError, "engine"):
            harness.validate_observation(row)

    def test_non_iso_timestamp_rejected(self):
        row = _valid_observation(timestamp="not-a-timestamp")
        with self.assertRaisesRegex(harness.ObservationValidationError, "timestamp"):
            harness.validate_observation(row)

    def test_zulu_timestamp_accepted(self):
        harness.validate_observation(_valid_observation(timestamp="2026-07-10T00:00:00Z"))


class ValidateJsonlTest(unittest.TestCase):
    def test_valid_file_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.jsonl"
            rows = [_valid_observation(run_index=i + 1) for i in range(3)]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            parsed = harness.validate_jsonl(path)
            self.assertEqual(len(parsed), 3)

    def test_blank_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.jsonl"
            path.write_text(f"\n{json.dumps(_valid_observation())}\n\n", encoding="utf-8")
            parsed = harness.validate_jsonl(path)
            self.assertEqual(len(parsed), 1)

    def test_malformed_json_reports_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.jsonl"
            path.write_text(f"{json.dumps(_valid_observation())}\nnot json\n", encoding="utf-8")
            with self.assertRaisesRegex(harness.ObservationValidationError, ":2:"):
                harness.validate_jsonl(path)

    def test_invalid_row_reports_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.jsonl"
            bad = _valid_observation()
            del bad["engine"]
            path.write_text(f"{json.dumps(_valid_observation())}\n{json.dumps(bad)}\n", encoding="utf-8")
            with self.assertRaisesRegex(harness.ObservationValidationError, ":2:"):
                harness.validate_jsonl(path)


# --------------------------------------------------------------------------
# Profile configuration validation
# --------------------------------------------------------------------------


class ProfileConfigTest(unittest.TestCase):
    def test_valid_profile_parses(self):
        profile = _profile()
        self.assertEqual(profile.windows, (32, 256))
        self.assertEqual(profile.aggregation, "median")

    def test_single_window_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "at least 2"):
            _profile(windows=[32])

    def test_non_increasing_windows_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "strictly increasing"):
            _profile(windows=[256, 32])

    def test_duplicate_windows_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "strictly increasing"):
            _profile(windows=[32, 32])

    def test_negative_warmup_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "warmup_repeats"):
            _profile(warmup_repeats=-1)

    def test_zero_measured_repeats_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "measured_repeats"):
            _profile(measured_repeats=0)

    def test_empty_engines_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "engines"):
            _profile(engines=[])

    def test_duplicate_engines_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "duplicates"):
            _profile(engines=["fake", "fake"])

    def test_unknown_aggregation_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "aggregation"):
            _profile(aggregation="fastest")

    def test_trim_with_median_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "trim"):
            _profile(aggregation="median", trim=1)

    def test_trim_too_large_for_repeats_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "trim"):
            _profile(aggregation="trimmed_mean", trim=2, measured_repeats=3)

    def test_missing_required_key_rejected(self):
        raw = {
            "windows": [32, 256],
            "warmup_repeats": 0,
            "measured_repeats": 3,
            "engines": ["fake"],
            "prompt": "x",
            "model": "m",
            # quantization omitted
        }
        with self.assertRaisesRegex(harness.ProfileConfigError, "quantization"):
            harness._parse_profile("bad", raw)


class LoadProfilesFileTest(unittest.TestCase):
    def test_scaffold_file_loads_with_no_profiles(self):
        schema_version, profiles = harness.load_profiles_file(harness.DEFAULT_PROFILES_FILE)
        self.assertEqual(schema_version, harness.SCHEMA_VERSION)
        self.assertEqual(profiles, {})

    def test_wrong_schema_version_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.toml"
            path.write_text("schema_version = 999\n", encoding="utf-8")
            with self.assertRaisesRegex(harness.ProfileConfigError, "schema_version"):
                harness.load_profiles_file(path)

    def test_named_profile_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.toml"
            path.write_text(
                """
schema_version = 1

[profiles.smoke]
windows = [32, 256]
warmup_repeats = 1
measured_repeats = 5
engines = ["fake"]
prompt = "hello world"
model = "qwen3.5-0.8b"
quantization = "q8"
""",
                encoding="utf-8",
            )
            _, profiles = harness.load_profiles_file(path)
            self.assertIn("smoke", profiles)
            self.assertEqual(profiles["smoke"].windows, (32, 256))
            self.assertEqual(profiles["smoke"].warmup_repeats, 1)


# --------------------------------------------------------------------------
# run_profile (fake-adapter, deterministic)
# --------------------------------------------------------------------------


class RunProfileTest(unittest.TestCase):
    def test_produces_expected_observation_count(self):
        profile = _profile(warmup_repeats=2, measured_repeats=3, windows=[32, 256])
        adapter = _FakeAdapter()
        result = harness.run_profile(
            profile,
            {"fake": adapter},
            clock=_FakeClock(),
            git_sha_value="deadbeef",
            hardware_id_value="test-host",
        )
        # 1 engine * 2 windows * (2 warmup + 3 measured) = 10
        self.assertEqual(len(result.observations), 10)
        self.assertEqual(result.missing_engines, ())

    def test_warmup_flag_and_run_index_reset_per_group(self):
        profile = _profile(warmup_repeats=2, measured_repeats=3, windows=[32, 256])
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter()}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        window_32 = [o for o in result.observations if o.requested_completion_tokens == 32]
        warmups = [o for o in window_32 if o.warmup]
        measured = [o for o in window_32 if not o.warmup]
        self.assertEqual([o.run_index for o in warmups], [1, 2])
        self.assertEqual([o.run_index for o in measured], [1, 2, 3])

    def test_order_index_strictly_increasing(self):
        profile = _profile(warmup_repeats=1, measured_repeats=2, windows=[32, 128, 256])
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter()}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        order_indices = [o.order_index for o in result.observations]
        self.assertEqual(order_indices, list(range(len(order_indices))))

    def test_baseline_window_executed_once_not_per_comparison(self):
        # Mirrors bench_context_scaling.sh's shape: N1 is a shared baseline
        # run once, not re-executed for every comparison window.
        profile = _profile(warmup_repeats=0, measured_repeats=4, windows=[8, 64, 128, 256])
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter()}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        baseline_runs = [o for o in result.observations if o.requested_completion_tokens == 8]
        self.assertEqual(len(baseline_runs), 4)

    def test_actual_and_requested_token_counts_recorded_independently(self):
        class _DriftingAdapter:
            engine_version = "drift-1.0"

            def run(self, *, prompt, n_tokens, warmup):
                return harness.AdapterRunResult(
                    actual_completion_tokens=n_tokens - 1,  # engine stopped one token early
                    actual_prompt_tokens=7,
                    engine_version=self.engine_version,
                )

        profile = _profile(warmup_repeats=0, measured_repeats=1, windows=[32, 256])
        result = harness.run_profile(
            profile, {"fake": _DriftingAdapter()}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        for obs in result.observations:
            self.assertEqual(obs.actual_completion_tokens, obs.requested_completion_tokens - 1)
            self.assertEqual(obs.actual_prompt_tokens, 7)

    def test_every_observation_is_schema_valid(self):
        profile = _profile(warmup_repeats=1, measured_repeats=2, windows=[32, 256])
        result = harness.run_profile(
            profile,
            {"fake": _FakeAdapter(native_ns_per_token=30_000)},
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )
        for obs in result.observations:
            harness.validate_observation(obs.to_dict())  # must not raise

    def test_missing_engine_fails_closed_by_default(self):
        profile = _profile(engines=["fake", "ghost"])
        with self.assertRaises(harness.MissingEngineError):
            harness.run_profile(profile, {"fake": _FakeAdapter()}, clock=_FakeClock())

    def test_missing_engine_allowed_with_flag(self):
        profile = _profile(engines=["fake", "ghost"])
        result = harness.run_profile(
            profile,
            {"fake": _FakeAdapter()},
            allow_missing_engine=True,
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )
        self.assertEqual(result.missing_engines, ("ghost",))
        self.assertTrue(all(o.engine == "fake" for o in result.observations))

    def test_missing_engine_never_fabricates_observations(self):
        profile = _profile(engines=["ghost"])
        result = harness.run_profile(
            profile, {}, allow_missing_engine=True, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        self.assertEqual(result.observations, ())
        self.assertEqual(result.missing_engines, ("ghost",))


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


class AggregateTest(unittest.TestCase):
    def test_median_slope_matches_hand_computed_value(self):
        # Fixed 1ms-per-call harness clock overhead is negligible next to the
        # adapter's own synthetic timing, so drive elapsed_ns directly via a
        # clock that advances by a known amount per adapter call.
        profile = _profile(warmup_repeats=0, measured_repeats=3, windows=[32, 256], aggregation="median")
        # 10 tok/s target: dt = (256-32)/10 = 22.4s. Each window's 3 measured
        # runs get identical elapsed_ns so the median is exact.
        # window 32: elapsed = 3.2s/call (32 tok @ 10 tok/s)
        # window 256: elapsed = 25.6s/call (256 tok @ 10 tok/s)
        steps = [3.2e9] * 3 + [25.6e9] * 3
        clock = _StepClock(steps)
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter()}, clock=clock, git_sha_value="x", hardware_id_value="h"
        )
        slopes = harness.aggregate(result)
        self.assertEqual(len(slopes), 1)
        self.assertAlmostEqual(slopes[0].slope_tok_per_s, 10.0, places=6)
        self.assertIsNone(slopes[0].slope_ci95)

    def test_trimmed_mean_slope_has_ci(self):
        profile = _profile(
            warmup_repeats=0, measured_repeats=5, windows=[32, 256], aggregation="trimmed_mean", trim=1
        )
        steps = [3.0e9, 3.2e9, 3.2e9, 3.2e9, 3.4e9, 25.0e9, 25.6e9, 25.6e9, 25.6e9, 26.0e9]
        clock = _StepClock(steps)
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter()}, clock=clock, git_sha_value="x", hardware_id_value="h"
        )
        slopes = harness.aggregate(result)
        self.assertEqual(len(slopes), 1)
        self.assertIsNotNone(slopes[0].slope_ci95)
        self.assertGreater(slopes[0].slope_tok_per_s, 0)

    def test_multi_window_produces_one_slope_per_comparison_window(self):
        profile = _profile(warmup_repeats=0, measured_repeats=2, windows=[8, 64, 128])
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter()}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        slopes = harness.aggregate(result)
        self.assertEqual({s.window for s in slopes}, {64, 128})
        self.assertTrue(all(s.baseline_window == 8 for s in slopes))

    def test_native_throughput_uses_actual_completion_tokens(self):
        profile = _profile(warmup_repeats=0, measured_repeats=1, windows=[32, 256])
        result = harness.run_profile(
            profile,
            {"fake": _FakeAdapter(native_ns_per_token=1_000_000)},  # 1000 tok/s native
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )
        native = harness.native_throughput(result, "fake")
        self.assertIsNotNone(native)
        self.assertAlmostEqual(native, 1000.0, places=3)

    def test_native_throughput_none_when_unreported(self):
        profile = _profile(warmup_repeats=0, measured_repeats=1, windows=[32, 256])
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter(native_ns_per_token=None)}, clock=_FakeClock(),
            git_sha_value="x", hardware_id_value="h",
        )
        self.assertIsNone(harness.native_throughput(result, "fake"))


@dataclass
class _StepClock:
    """Deterministic clock driven by an explicit list of per-call durations (ns)."""

    durations_ns: list[float]
    _t: float = 0.0
    _idx: int = 0

    def __call__(self) -> int:
        # First call in a pair returns current t, second call advances by the
        # next scheduled duration then returns the new t.
        if self._idx % 2 == 1:
            self._t += self.durations_ns[self._idx // 2]
        value = int(self._t)
        self._idx += 1
        return value


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


class RenderReportTest(unittest.TestCase):
    def test_report_mentions_profile_and_missing_engines(self):
        profile = _profile(engines=["fake", "ghost"])
        result = harness.run_profile(
            profile,
            {"fake": _FakeAdapter()},
            allow_missing_engine=True,
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )
        slopes = harness.aggregate(result)
        report = harness.render_report(result, slopes)
        self.assertIn("unit_test", report)
        self.assertIn("ghost", report)
        self.assertIn("fake", report)

    def test_report_handles_no_measured_data(self):
        profile = _profile(engines=["ghost"])
        result = harness.run_profile(profile, {}, allow_missing_engine=True, clock=_FakeClock())
        report = harness.render_report(result, harness.aggregate(result))
        self.assertIn("no measured data", report)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


class CliTest(unittest.TestCase):
    def test_validate_subcommand_reports_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.jsonl"
            path.write_text(json.dumps(_valid_observation()) + "\n", encoding="utf-8")
            rc = harness.main(["validate", str(path)])
            self.assertEqual(rc, 0)

    def test_validate_subcommand_fails_closed_on_bad_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.jsonl"
            path.write_text("not json\n", encoding="utf-8")
            rc = harness.main(["validate", str(path)])
            self.assertEqual(rc, 1)

    def test_run_subcommand_unknown_profile_fails_closed(self):
        rc = harness.main(["run", "--profile", "does-not-exist"])
        self.assertEqual(rc, 1)

    def test_run_subcommand_missing_engine_fails_closed_without_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.toml"
            path.write_text(
                """
schema_version = 1

[profiles.smoke]
windows = [32, 256]
warmup_repeats = 0
measured_repeats = 1
engines = ["nonexistent-engine"]
prompt = "hello"
model = "m"
quantization = "q8"
""",
                encoding="utf-8",
            )
            rc = harness.main(["run", "--profile", "smoke", "--profiles-file", str(path)])
            self.assertEqual(rc, 1)

    def test_run_subcommand_missing_engine_allowed_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.toml"
            path.write_text(
                """
schema_version = 1

[profiles.smoke]
windows = [32, 256]
warmup_repeats = 0
measured_repeats = 1
engines = ["nonexistent-engine"]
prompt = "hello"
model = "m"
quantization = "q8"
""",
                encoding="utf-8",
            )
            rc = harness.main(
                ["run", "--profile", "smoke", "--profiles-file", str(path), "--allow-missing-engine"]
            )
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
