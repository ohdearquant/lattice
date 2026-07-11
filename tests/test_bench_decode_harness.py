"""Unit tests for scripts/bench_decode_harness.py (issue #813).

Deterministic fake-adapter tests and raw-schema validation only -- no engine
binary is required to run this file, matching the harness-core landing gate.

Run with: python3 tests/test_bench_decode_harness.py -v
(or `python3 -m pytest tests/test_bench_decode_harness.py -v` -- no
pytest-only features are used).
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
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
    """Deterministic fake engine adapter: fixed tok/s, exact requested token counts.

    `component_ns`, when given, is returned verbatim as
    `AdapterRunResult.component_ns` on every call -- for tests exercising a
    `MeasuredCall` with `len(yields_windows) > 1` (e.g. Ollama's
    one-call-yields-two-windows shape), where every call in the test only
    ever requests the one multi-yield `n_tokens` budget so a single static
    map is sufficient.
    """

    def __init__(
        self,
        engine_version: str = "fake-1.0",
        native_ns_per_token: int | None = None,
        component_ns: dict[int, int] | None = None,
    ):
        self.engine_version = engine_version
        self.native_ns_per_token = native_ns_per_token
        self.component_ns = component_ns
        self.calls: list[tuple[int, bool, str, str]] = []

    def run(
        self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str
    ) -> "harness.AdapterRunResult":
        self.calls.append((n_tokens, warmup, model, quantization))
        native_ns = self.native_ns_per_token * n_tokens if self.native_ns_per_token else None
        return harness.AdapterRunResult(
            actual_completion_tokens=n_tokens,
            actual_prompt_tokens=len(prompt.split()),
            native_ns=native_ns,
            engine_version=self.engine_version,
            component_ns=self.component_ns,
        )


def _engine(
    name: str = "fake",
    warmup_repeats: int = 0,
    warmup_tokens: int | None = None,
    warmup_prompt: str | None = None,
    model: str = "qwen3.5-0.8b",
    quantization: str = "q8",
    measured_order: str | None = None,
    measured_calls: list[dict] | None = None,
) -> dict:
    d = {"name": name, "warmup_repeats": warmup_repeats, "model": model, "quantization": quantization}
    if warmup_tokens is not None:
        d["warmup_tokens"] = warmup_tokens
    if warmup_prompt is not None:
        d["warmup_prompt"] = warmup_prompt
    if measured_order is not None:
        d["measured_order"] = measured_order
    if measured_calls is not None:
        d["measured_calls"] = measured_calls
    return d


def _profile(**overrides) -> "harness.ProfileConfig":
    raw = {
        "description": "unit test profile",
        "windows": [32, 256],
        "measured_repeats": 3,
        "engines": [_engine()],
        "prompt": "the quick brown fox",
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
        self.assertEqual(profile.engines, ("fake",))
        self.assertEqual(len(profile.engine_groups), 1)
        self.assertEqual(profile.engine_groups[0].warmup_repeats, 0)
        self.assertIsNone(profile.engine_groups[0].warmup_tokens)
        self.assertIsNone(profile.engine_groups[0].warmup_prompt)
        self.assertEqual(profile.engine_groups[0].model, "qwen3.5-0.8b")
        self.assertEqual(profile.engine_groups[0].quantization, "q8")

    def test_single_window_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "at least 2"):
            _profile(windows=[32])

    def test_non_increasing_windows_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "strictly increasing"):
            _profile(windows=[256, 32])

    def test_duplicate_windows_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "strictly increasing"):
            _profile(windows=[32, 32])

    def test_zero_measured_repeats_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "measured_repeats"):
            _profile(measured_repeats=0)

    def test_empty_engines_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "engines"):
            _profile(engines=[])

    def test_engine_entry_must_be_a_table(self):
        # The old flat `engines = ["fake"]` name-list shape is no longer
        # representable -- every engine now needs its own warmup/model/
        # quantization, so a bare string entry is rejected.
        with self.assertRaisesRegex(harness.ProfileConfigError, "must be a table"):
            _profile(engines=["fake"])

    def test_duplicate_engine_names_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "duplicate"):
            _profile(engines=[_engine("fake"), _engine("fake")])

    def test_empty_engine_name_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "name must be non-empty"):
            _profile(engines=[_engine(name="")])

    def test_negative_engine_warmup_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "warmup_repeats"):
            _profile(engines=[_engine(warmup_repeats=-1)])

    # -- Warmup is an explicit pre-window batch (round-2 blocker fix):
    # warmup_tokens is the completion-token budget for that batch, and is
    # only meaningful together with a positive warmup_repeats. --

    def test_warmup_tokens_required_when_warmup_repeats_positive(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "warmup_tokens.*required"):
            _profile(engines=[_engine(warmup_repeats=1)])

    def test_warmup_tokens_rejected_when_warmup_repeats_zero(self):
        raw_engine = {"name": "fake", "warmup_repeats": 0, "warmup_tokens": 8, "model": "m", "quantization": "q8"}
        with self.assertRaisesRegex(harness.ProfileConfigError, "warmup_tokens.*only meaningful"):
            _profile(engines=[raw_engine])

    def test_warmup_tokens_must_be_positive(self):
        raw_engine = {"name": "fake", "warmup_repeats": 1, "warmup_tokens": 0, "model": "m", "quantization": "q8"}
        with self.assertRaisesRegex(harness.ProfileConfigError, "warmup_tokens.*positive"):
            _profile(engines=[raw_engine])

    def test_warmup_tokens_wrong_type_rejected(self):
        raw_engine = {
            "name": "fake",
            "warmup_repeats": 1,
            "warmup_tokens": "eight",
            "model": "m",
            "quantization": "q8",
        }
        with self.assertRaisesRegex(harness.ProfileConfigError, "warmup_tokens"):
            _profile(engines=[raw_engine])

    def test_warmup_prompt_optional_at_zero_warmup(self):
        # warmup_prompt is simply unused when warmup_repeats == 0, so it is
        # not rejected the way warmup_tokens is -- there is nothing
        # ambiguous about specifying it (it would only matter if warmups
        # were ever scheduled).
        profile = _profile(engines=[_engine(warmup_repeats=0, warmup_prompt="unused prompt")])
        self.assertEqual(profile.engine_groups[0].warmup_prompt, "unused prompt")

    def test_warmup_prompt_must_be_non_empty_if_provided(self):
        raw_engine = {
            "name": "fake",
            "warmup_repeats": 1,
            "warmup_tokens": 8,
            "warmup_prompt": "",
            "model": "m",
            "quantization": "q8",
        }
        with self.assertRaisesRegex(harness.ProfileConfigError, "warmup_prompt"):
            _profile(engines=[raw_engine])

    def test_warmup_prompt_wrong_type_rejected(self):
        raw_engine = {
            "name": "fake",
            "warmup_repeats": 1,
            "warmup_tokens": 8,
            "warmup_prompt": 123,
            "model": "m",
            "quantization": "q8",
        }
        with self.assertRaisesRegex(harness.ProfileConfigError, "warmup_prompt"):
            _profile(engines=[raw_engine])

    def test_empty_engine_model_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "model must be non-empty"):
            _profile(engines=[_engine(model="")])

    def test_empty_engine_quantization_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "quantization must be non-empty"):
            _profile(engines=[_engine(quantization="")])

    # -- Round-3 blocker fix: measured_order / measured_calls, the
    # per-engine measured-schedule contract that windows/measured_repeats
    # alone could not represent (bench_compare_1k.py). --

    def test_measured_order_defaults_to_window_major(self):
        profile = _profile(engines=[_engine()])
        self.assertEqual(profile.engine_groups[0].measured_order, "window_major")

    def test_measured_order_accepts_repeat_major(self):
        profile = _profile(engines=[_engine(measured_order="repeat_major")])
        self.assertEqual(profile.engine_groups[0].measured_order, "repeat_major")

    def test_measured_order_rejects_unknown_value(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "measured_order"):
            _profile(engines=[_engine(measured_order="engine_major")])

    def test_measured_order_wrong_type_rejected(self):
        raw_engine = {
            "name": "fake",
            "warmup_repeats": 0,
            "model": "m",
            "quantization": "q8",
            "measured_order": 1,
        }
        with self.assertRaisesRegex(harness.ProfileConfigError, "measured_order"):
            _profile(engines=[raw_engine])

    def test_measured_calls_defaults_to_none(self):
        profile = _profile(engines=[_engine()])
        self.assertIsNone(profile.engine_groups[0].measured_calls)

    def test_measured_calls_parsed_into_tuple_of_measured_call(self):
        profile = _profile(
            engines=[_engine(measured_calls=[{"n_tokens": 100, "yields_windows": [1, 100]}])]
        )
        calls = profile.engine_groups[0].measured_calls
        self.assertEqual(calls, (harness.MeasuredCall(n_tokens=100, yields_windows=(1, 100)),))

    def test_measured_calls_empty_list_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "measured_calls.*non-empty"):
            _profile(engines=[_engine(measured_calls=[])])

    def test_measured_calls_entry_not_a_table_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "measured_calls\\[0\\].*table"):
            _profile(engines=[_engine(measured_calls=["not-a-table"])])

    def test_measured_calls_unexpected_key_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "unexpected key.*prompt_override"):
            _profile(
                engines=[
                    _engine(
                        measured_calls=[
                            {"n_tokens": 100, "yields_windows": [100], "prompt_override": "x"}
                        ]
                    )
                ]
            )

    def test_measured_calls_missing_n_tokens_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "missing required key.*n_tokens"):
            _profile(engines=[_engine(measured_calls=[{"yields_windows": [100]}])])

    def test_measured_calls_missing_yields_windows_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "missing required key.*yields_windows"):
            _profile(engines=[_engine(measured_calls=[{"n_tokens": 100}])])

    def test_measured_calls_n_tokens_must_be_positive(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "n_tokens.*positive"):
            _profile(engines=[_engine(measured_calls=[{"n_tokens": 0, "yields_windows": [1]}])])

    def test_measured_calls_n_tokens_wrong_type_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "n_tokens"):
            _profile(engines=[_engine(measured_calls=[{"n_tokens": "100", "yields_windows": [100]}])])

    def test_measured_calls_n_tokens_bool_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "n_tokens"):
            _profile(engines=[_engine(measured_calls=[{"n_tokens": True, "yields_windows": [1]}])])

    def test_measured_calls_yields_windows_must_be_non_empty(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "yields_windows.*non-empty"):
            _profile(engines=[_engine(measured_calls=[{"n_tokens": 100, "yields_windows": []}])])

    def test_measured_calls_yields_windows_entry_must_be_positive(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "yields_windows.*positive"):
            _profile(engines=[_engine(measured_calls=[{"n_tokens": 100, "yields_windows": [0]}])])

    def test_measured_calls_yields_windows_entry_bool_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "yields_windows.*positive"):
            _profile(engines=[_engine(measured_calls=[{"n_tokens": 100, "yields_windows": [True]}])])

    def test_measured_calls_yields_windows_duplicate_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "yields_windows.*duplicate"):
            _profile(engines=[_engine(measured_calls=[{"n_tokens": 100, "yields_windows": [1, 1]}])])

    def test_measured_calls_multiple_entries_parsed_in_order(self):
        profile = _profile(
            engines=[
                _engine(
                    measured_order="repeat_major",
                    measured_calls=[
                        {"n_tokens": 1, "yields_windows": [1]},
                        {"n_tokens": 100, "yields_windows": [100]},
                    ],
                )
            ]
        )
        calls = profile.engine_groups[0].measured_calls
        self.assertEqual(
            calls,
            (
                harness.MeasuredCall(n_tokens=1, yields_windows=(1,)),
                harness.MeasuredCall(n_tokens=100, yields_windows=(100,)),
            ),
        )

    def test_unexpected_engine_key_rejected(self):
        bad_engine = {"name": "fake", "warmup_repeats": 0, "model": "m", "quantization": "q8", "warmups": 1}
        with self.assertRaisesRegex(harness.ProfileConfigError, "unexpected key"):
            _profile(engines=[bad_engine])

    def test_missing_required_engine_key_rejected(self):
        raw = {
            "windows": [32, 256],
            "measured_repeats": 3,
            "engines": [{"name": "fake", "warmup_repeats": 0, "model": "m"}],  # quantization omitted
            "prompt": "x",
        }
        with self.assertRaisesRegex(harness.ProfileConfigError, "quantization"):
            harness._parse_profile("bad", raw)

    def test_missing_required_top_level_key_rejected(self):
        raw = {
            "windows": [32, 256],
            "measured_repeats": 3,
            "engines": [_engine()],
            # prompt omitted
        }
        with self.assertRaisesRegex(harness.ProfileConfigError, "prompt"):
            harness._parse_profile("bad", raw)

    def test_unknown_aggregation_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "aggregation"):
            _profile(aggregation="fastest")

    def test_trim_with_median_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "trim"):
            _profile(aggregation="median", trim=1)

    def test_trim_too_large_for_repeats_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "trim"):
            _profile(aggregation="trimmed_mean", trim=2, measured_repeats=3)

    def test_description_wrong_type_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, "description"):
            _profile(description=123)

    # -- Fail-closed on unknown/misspelled profile keys (round-1 major: a
    # typo in an optional methodology key must never silently fall back to
    # a default instead of being rejected). --

    def test_misspelled_aggregation_key_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, r"unexpected key.*aggregtion"):
            _profile(aggregtion="trimmed_mean")

    def test_misspelled_trim_key_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, r"unexpected key.*trmi"):
            _profile(trmi=1)

    def test_misspelled_measured_repeats_key_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, r"unexpected key.*measured_repeat\b"):
            _profile(measured_repeat=3)

    def test_misspelled_requested_prompt_tokens_key_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, r"unexpected key.*requested_prompt_tokenz"):
            _profile(requested_prompt_tokenz=10)

    def test_misspelled_description_key_rejected(self):
        with self.assertRaisesRegex(harness.ProfileConfigError, r"unexpected key.*description2"):
            _profile(description2="typo")


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

    def test_unexpected_top_level_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.toml"
            path.write_text(
                "schema_version = 1\nextra_top_level_key = true\n\n[profiles]\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(harness.ProfileConfigError, "unexpected top-level key"):
                harness.load_profiles_file(path)

    def test_named_profile_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.toml"
            path.write_text(
                """
schema_version = 1

[profiles.smoke]
windows = [32, 256]
measured_repeats = 5
prompt = "hello world"

[[profiles.smoke.engines]]
name = "fake"
warmup_repeats = 1
warmup_tokens = 8
model = "qwen3.5-0.8b"
quantization = "q8"
""",
                encoding="utf-8",
            )
            _, profiles = harness.load_profiles_file(path)
            self.assertIn("smoke", profiles)
            self.assertEqual(profiles["smoke"].windows, (32, 256))
            self.assertEqual(profiles["smoke"].engines, ("fake",))
            self.assertEqual(profiles["smoke"].engine_groups[0].warmup_repeats, 1)
            self.assertEqual(profiles["smoke"].engine_groups[0].warmup_tokens, 8)

    def test_heterogeneous_q4_q8_profile_round_trips(self):
        # Reproduces bench_q4_apples.sh's shape in one profile: Lattice and
        # MLX both run Q4, Ollama runs Q8 as a reference only, and only MLX
        # gets a warmup (bench_apples_to_apples.sh's pattern).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.toml"
            path.write_text(
                """
schema_version = 1

[profiles.q4_apples]
windows = [32, 256]
measured_repeats = 5
prompt = "hello world"

[[profiles.q4_apples.engines]]
name = "lattice"
warmup_repeats = 0
model = "qwen3.5-0.8b-q4"
quantization = "q4"

[[profiles.q4_apples.engines]]
name = "mlx"
warmup_repeats = 1
warmup_tokens = 8
model = "qwen3.5-0.8b"
quantization = "q4"

[[profiles.q4_apples.engines]]
name = "ollama"
warmup_repeats = 0
model = "qwen3.5:0.8b"
quantization = "q8"
""",
                encoding="utf-8",
            )
            _, profiles = harness.load_profiles_file(path)
            profile = profiles["q4_apples"]
            self.assertEqual(profile.engines, ("lattice", "mlx", "ollama"))
            groups = {g.name: g for g in profile.engine_groups}
            self.assertEqual(groups["lattice"].warmup_repeats, 0)
            self.assertEqual(groups["lattice"].quantization, "q4")
            self.assertEqual(groups["mlx"].warmup_repeats, 1)
            self.assertEqual(groups["mlx"].warmup_tokens, 8)
            self.assertEqual(groups["mlx"].quantization, "q4")
            self.assertEqual(groups["ollama"].warmup_repeats, 0)
            self.assertEqual(groups["ollama"].quantization, "q8")


# --------------------------------------------------------------------------
# run_profile (fake-adapter, deterministic)
# --------------------------------------------------------------------------


class RunProfileTest(unittest.TestCase):
    def test_produces_expected_observation_count(self):
        profile = _profile(
            engines=[_engine(warmup_repeats=2, warmup_tokens=8)], measured_repeats=3, windows=[32, 256]
        )
        adapter = _FakeAdapter()
        result = harness.run_profile(
            profile,
            {"fake": adapter},
            clock=_FakeClock(),
            git_sha_value="deadbeef",
            hardware_id_value="test-host",
        )
        # 1 engine * (2 warmups, once, NOT per window) + 1 engine * 2 windows * 3 measured = 2 + 6 = 8
        self.assertEqual(len(result.observations), 8)
        self.assertEqual(result.missing_engines, ())

    def test_warmup_executes_once_before_all_measured_windows(self):
        # The round-2 blocker fix: warmup is a pre-window batch at its own
        # token budget, not looped once per measured window.
        profile = _profile(
            engines=[_engine(warmup_repeats=2, warmup_tokens=8)], measured_repeats=3, windows=[32, 256]
        )
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter()}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        warmups = [o for o in result.observations if o.warmup]
        measured = [o for o in result.observations if not o.warmup]

        # Exactly warmup_repeats warmup rows total (not per window), all at
        # warmup_tokens, never at a measured window's token count.
        self.assertEqual(len(warmups), 2)
        self.assertEqual([o.requested_completion_tokens for o in warmups], [8, 8])
        self.assertEqual([o.run_index for o in warmups], [1, 2])

        # Every warmup observation's order_index precedes every measured
        # observation's -- the warmup batch runs first, in full, before any
        # measured window.
        self.assertLess(max(o.order_index for o in warmups), min(o.order_index for o in measured))

        # Measured run_index still resets per window (unaffected by the
        # warmup-schedule fix).
        measured_32 = [o for o in measured if o.requested_completion_tokens == 32]
        measured_256 = [o for o in measured if o.requested_completion_tokens == 256]
        self.assertEqual([o.run_index for o in measured_32], [1, 2, 3])
        self.assertEqual([o.run_index for o in measured_256], [1, 2, 3])

    def test_order_index_strictly_increasing(self):
        profile = _profile(
            engines=[_engine(warmup_repeats=1, warmup_tokens=8)], measured_repeats=2, windows=[32, 128, 256]
        )
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter()}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        order_indices = [o.order_index for o in result.observations]
        self.assertEqual(order_indices, list(range(len(order_indices))))

    def test_baseline_window_executed_once_not_per_comparison(self):
        # Mirrors bench_context_scaling.sh's shape: N1 is a shared baseline
        # run once, not re-executed for every comparison window.
        profile = _profile(measured_repeats=4, windows=[8, 64, 128, 256])
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter()}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        baseline_runs = [o for o in result.observations if o.requested_completion_tokens == 8]
        self.assertEqual(len(baseline_runs), 4)

    def test_actual_and_requested_token_counts_recorded_independently(self):
        class _DriftingAdapter:
            engine_version = "drift-1.0"

            def run(self, *, prompt, n_tokens, warmup, model, quantization):
                return harness.AdapterRunResult(
                    actual_completion_tokens=n_tokens - 1,  # engine stopped one token early
                    actual_prompt_tokens=7,
                    engine_version=self.engine_version,
                )

        profile = _profile(measured_repeats=1, windows=[32, 256])
        result = harness.run_profile(
            profile, {"fake": _DriftingAdapter()}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        for obs in result.observations:
            self.assertEqual(obs.actual_completion_tokens, obs.requested_completion_tokens - 1)
            self.assertEqual(obs.actual_prompt_tokens, 7)

    def test_every_observation_is_schema_valid(self):
        profile = _profile(
            engines=[_engine(warmup_repeats=1, warmup_tokens=8)], measured_repeats=2, windows=[32, 256]
        )
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
        profile = _profile(engines=[_engine("fake"), _engine("ghost")])
        with self.assertRaises(harness.MissingEngineError):
            harness.run_profile(profile, {"fake": _FakeAdapter()}, clock=_FakeClock())

    def test_missing_engine_allowed_with_flag(self):
        profile = _profile(engines=[_engine("fake"), _engine("ghost")])
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
        profile = _profile(engines=[_engine("ghost")])
        result = harness.run_profile(
            profile, {}, allow_missing_engine=True, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        self.assertEqual(result.observations, ())
        self.assertEqual(result.missing_engines, ("ghost",))

    def test_adapter_default_identity_matches_requested(self):
        profile = _profile(
            engines=[_engine("fake", model="qwen3.5-0.8b-q4", quantization="q4")],
            measured_repeats=1,
            windows=[32, 256],
        )
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter()}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        for obs in result.observations:
            self.assertEqual(obs.model, "qwen3.5-0.8b-q4")
            self.assertEqual(obs.quantization, "q4")

    def test_adapter_reported_actual_identity_overrides_requested(self):
        class _DriftingIdentityAdapter:
            engine_version = "drift-1.0"

            def run(self, *, prompt, n_tokens, warmup, model, quantization):
                return harness.AdapterRunResult(
                    actual_completion_tokens=n_tokens,
                    actual_prompt_tokens=4,
                    engine_version=self.engine_version,
                    actual_model="qwen3.5-0.8b-fallback",
                    actual_quantization="q8",
                )

        profile = _profile(
            engines=[_engine("fake", model="qwen3.5-0.8b-q4", quantization="q4")],
            measured_repeats=1,
            windows=[32, 256],
        )
        result = harness.run_profile(
            profile,
            {"fake": _DriftingIdentityAdapter()},
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )
        for obs in result.observations:
            # The profile requested Q4, but the adapter reports it actually
            # invoked a fallback Q8 artifact -- the raw observation must
            # record what actually ran, never the requested identity.
            self.assertEqual(obs.model, "qwen3.5-0.8b-fallback")
            self.assertEqual(obs.quantization, "q8")

    def test_heterogeneous_engine_schedule_exact_call_sequence_and_identities(self):
        """The blocker fix: reproduce bench_q4_apples.sh (Lattice Q4, MLX
        Q4, Ollama Q8-reference) with bench_apples_to_apples.sh's single
        pre-loop 8-token MLX-only warmup, all inside one profile. MLX's
        warmup is scheduled once, at its OWN 8-token budget -- distinct
        from both measured windows (32, 256) -- entirely before its
        measured runs, exactly reproducing the legacy script's call order.

        Engine order is Lattice -> Ollama -> MLX, matching
        `bench_q4_apples.sh`'s (and `bench_apples_to_apples.sh`'s,
        `bench_apples_precise.sh`'s) actual script section order -- NOT
        Lattice -> MLX -> Ollama (a round-3 review finding: an earlier
        version of this test asserted the wrong global order while
        claiming to reproduce the legacy schedule).
        """
        profile = harness._parse_profile(
            "q4_apples",
            {
                "windows": [32, 256],
                "measured_repeats": 2,
                "prompt": "the quick brown fox",
                "engines": [
                    _engine("lattice", warmup_repeats=0, model="qwen3.5-0.8b-q4", quantization="q4"),
                    _engine("ollama", warmup_repeats=0, model="qwen3.5:0.8b", quantization="q8"),
                    _engine("mlx", warmup_repeats=1, warmup_tokens=8, model="qwen3.5-0.8b", quantization="q4"),
                ],
            },
        )
        lattice = _FakeAdapter()
        ollama = _FakeAdapter()
        mlx = _FakeAdapter()
        result = harness.run_profile(
            profile,
            {"lattice": lattice, "ollama": ollama, "mlx": mlx},
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )

        # Exact call sequence per engine. lattice/ollama: 0 warmup + 2
        # measured per window = 4 calls. mlx: ONE 8-token warmup call,
        # then 2 measured per window = 5 calls -- exactly
        # bench_apples_to_apples.sh's single pre-loop MLX warmup, not a
        # warmup repeated at every measured window's token count.
        self.assertEqual(
            lattice.calls,
            [
                (32, False, "qwen3.5-0.8b-q4", "q4"),
                (32, False, "qwen3.5-0.8b-q4", "q4"),
                (256, False, "qwen3.5-0.8b-q4", "q4"),
                (256, False, "qwen3.5-0.8b-q4", "q4"),
            ],
        )
        self.assertEqual(
            ollama.calls,
            [
                (32, False, "qwen3.5:0.8b", "q8"),
                (32, False, "qwen3.5:0.8b", "q8"),
                (256, False, "qwen3.5:0.8b", "q8"),
                (256, False, "qwen3.5:0.8b", "q8"),
            ],
        )
        self.assertEqual(
            mlx.calls,
            [
                (8, True, "qwen3.5-0.8b", "q4"),
                (32, False, "qwen3.5-0.8b", "q4"),
                (32, False, "qwen3.5-0.8b", "q4"),
                (256, False, "qwen3.5-0.8b", "q4"),
                (256, False, "qwen3.5-0.8b", "q4"),
            ],
        )

        # Engine run order in the raw observation stream matches the
        # profile's engines order exactly: all of lattice's rows, then all
        # of ollama's, then all of mlx's -- the legacy Lattice -> Ollama ->
        # MLX global order.
        self.assertEqual(
            [o.engine for o in result.observations],
            ["lattice"] * 4 + ["ollama"] * 4 + ["mlx"] * 5,
        )

        # Per-row identity: each observation records ITS OWN engine's
        # requested model/quantization, never another engine's or a
        # profile-wide value (the blocker this schema fixes).
        for obs in result.observations:
            if obs.engine == "lattice":
                self.assertEqual((obs.model, obs.quantization), ("qwen3.5-0.8b-q4", "q4"))
            elif obs.engine == "mlx":
                self.assertEqual((obs.model, obs.quantization), ("qwen3.5-0.8b", "q4"))
            elif obs.engine == "ollama":
                self.assertEqual((obs.model, obs.quantization), ("qwen3.5:0.8b", "q8"))
            else:  # pragma: no cover -- defensive, should be unreachable
                self.fail(f"unexpected engine {obs.engine!r}")

    def test_precise_style_warmup_twice_before_both_measured_windows(self):
        """Reproduces bench_apples_precise.sh: every engine warms twice at
        N2=512, once before both measured windows (N1=64, N2=512) --
        never once per window (which would be 4 warmups: 2x64, 2x512)."""
        profile = harness._parse_profile(
            "precise",
            {
                "windows": [64, 512],
                "measured_repeats": 13,
                "prompt": "the quick brown fox",
                "engines": [_engine("fake", warmup_repeats=2, warmup_tokens=512)],
            },
        )
        adapter = _FakeAdapter()
        harness.run_profile(
            profile, {"fake": adapter}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        expected = (
            [(512, True, "qwen3.5-0.8b", "q8")] * 2
            + [(64, False, "qwen3.5-0.8b", "q8")] * 13
            + [(512, False, "qwen3.5-0.8b", "q8")] * 13
        )
        self.assertEqual(adapter.calls, expected)

    def test_context_scaling_style_single_warmup_before_all_comparison_windows(self):
        """Reproduces bench_context_scaling.sh: MLX warms once at 4 tokens
        before the N1=8 baseline and every N2 comparison window
        ({64, 128, 256}) -- one warmup total, not one per window."""
        profile = harness._parse_profile(
            "context_scaling",
            {
                "windows": [8, 64, 128, 256],
                "measured_repeats": 5,
                "prompt": "the quick brown fox",
                "engines": [_engine("mlx", warmup_repeats=1, warmup_tokens=4)],
            },
        )
        adapter = _FakeAdapter()
        result = harness.run_profile(
            profile, {"mlx": adapter}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        expected = (
            [(4, True, "qwen3.5-0.8b", "q8")]
            + [(8, False, "qwen3.5-0.8b", "q8")] * 5
            + [(64, False, "qwen3.5-0.8b", "q8")] * 5
            + [(128, False, "qwen3.5-0.8b", "q8")] * 5
            + [(256, False, "qwen3.5-0.8b", "q8")] * 5
        )
        self.assertEqual(adapter.calls, expected)
        warmups = [o for o in result.observations if o.warmup]
        self.assertEqual(len(warmups), 1)
        self.assertEqual(warmups[0].requested_completion_tokens, 4)

    def test_default_warmup_prompt_uses_profile_prompt(self):
        profile = _profile(
            engines=[_engine(warmup_repeats=1, warmup_tokens=8)],
            measured_repeats=1,
            windows=[32, 256],
            prompt="the measured prompt",
        )
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter()}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        warmup_obs = [o for o in result.observations if o.warmup][0]
        measured_obs = [o for o in result.observations if not o.warmup][0]
        self.assertEqual(warmup_obs.prompt_hash, measured_obs.prompt_hash)
        self.assertEqual(warmup_obs.prompt_hash, harness.prompt_hash("the measured prompt"))

    def test_custom_warmup_prompt_overrides_profile_prompt(self):
        """Reproduces bench_compare_1k.py's distinct per-engine warmup
        prompt, separate from the measured prompt."""
        profile = _profile(
            engines=[_engine(warmup_repeats=1, warmup_tokens=4, warmup_prompt="a distinct warmup prompt")],
            measured_repeats=1,
            windows=[32, 256],
            prompt="the measured prompt",
        )
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter()}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        warmup_obs = [o for o in result.observations if o.warmup][0]
        measured_obs = [o for o in result.observations if not o.warmup][0]
        self.assertEqual(warmup_obs.prompt_hash, harness.prompt_hash("a distinct warmup prompt"))
        self.assertEqual(measured_obs.prompt_hash, harness.prompt_hash("the measured prompt"))
        self.assertNotEqual(warmup_obs.prompt_hash, measured_obs.prompt_hash)

    # -- Round-3 blocker fix: the MEASURED-side schedule contract
    # (`measured_calls` / `measured_order` / `AdapterRunResult.
    # component_ns`), added because windows/measured_repeats alone cannot
    # represent bench_compare_1k.py's three distinct measured shapes. --

    def test_measured_order_repeat_major_alternates_calls_per_repeat(self):
        """MLX's bench_compare_1k.py shape: an explicit two-call plan
        (1-token, 100-token) replayed repeat-major -- (1, 100) alternating
        every repeat -- NOT window-major (all 1s then all 100s)."""
        profile = harness._parse_profile(
            "agentic_mlx_only",
            {
                "windows": [1, 100],
                "measured_repeats": 3,
                "prompt": "the padded ctx prompt",
                "engines": [
                    _engine(
                        "fake",
                        measured_order="repeat_major",
                        measured_calls=[
                            {"n_tokens": 1, "yields_windows": [1]},
                            {"n_tokens": 100, "yields_windows": [100]},
                        ],
                    )
                ],
            },
        )
        adapter = _FakeAdapter()
        harness.run_profile(
            profile, {"fake": adapter}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        expected = [(1, False, "qwen3.5-0.8b", "q8"), (100, False, "qwen3.5-0.8b", "q8")] * 3
        self.assertEqual(adapter.calls, expected)

    def test_measured_calls_default_matches_explicit_window_major_equivalent(self):
        """Sanity: an engine with no measured_calls (implicit default,
        derived from profile.windows) produces the identical call sequence
        as the same shape written out explicitly with measured_order=
        'window_major' (the default)."""
        implicit = harness._parse_profile(
            "implicit", {"windows": [32, 256], "measured_repeats": 2, "prompt": "x", "engines": [_engine()]}
        )
        explicit = harness._parse_profile(
            "explicit",
            {
                "windows": [32, 256],
                "measured_repeats": 2,
                "prompt": "x",
                "engines": [
                    _engine(
                        measured_calls=[
                            {"n_tokens": 32, "yields_windows": [32]},
                            {"n_tokens": 256, "yields_windows": [256]},
                        ]
                    )
                ],
            },
        )
        a1, a2 = _FakeAdapter(), _FakeAdapter()
        harness.run_profile(implicit, {"fake": a1}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h")
        harness.run_profile(explicit, {"fake": a2}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h")
        self.assertEqual(a1.calls, a2.calls)

    def test_measured_call_multi_yield_uses_component_ns_for_elapsed_ns(self):
        """Ollama's bench_compare_1k.py shape: ONE 100-token call per
        repeat yields BOTH a window=1 and a window=100 observation, with
        elapsed_ns for each coming from the adapter-reported component --
        never from the harness's own single wall-clock span for the whole
        call (which would conflate the two components)."""
        profile = harness._parse_profile(
            "agentic_ollama_only",
            {
                "windows": [1, 100],
                "measured_repeats": 2,
                "prompt": "the padded ctx prompt",
                "engines": [_engine("fake", measured_calls=[{"n_tokens": 100, "yields_windows": [1, 100]}])],
            },
        )
        adapter = _FakeAdapter(component_ns={1: 40_000_000, 100: 400_000_000})
        result = harness.run_profile(
            profile, {"fake": adapter}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        # Exactly ONE physical call per repeat -- the harness never
        # re-invokes the engine to synthesize the second window.
        self.assertEqual(adapter.calls, [(100, False, "qwen3.5-0.8b", "q8")] * 2)

        measured = [o for o in result.observations if not o.warmup]
        self.assertEqual([o.requested_completion_tokens for o in measured], [1, 100, 1, 100])
        self.assertEqual([o.elapsed_ns for o in measured], [40_000_000, 400_000_000, 40_000_000, 400_000_000])
        # Both windows from the SAME physical call share the same
        # run_index (they are one repeat's two derived readings) and the
        # SAME actual_completion_tokens (the real underlying call
        # requested/generated 100 tokens, regardless of which window a
        # given derived observation represents).
        self.assertEqual([o.run_index for o in measured], [1, 1, 2, 2])
        self.assertTrue(all(o.actual_completion_tokens == 100 for o in measured))

    def test_measured_call_multi_yield_missing_component_ns_raises(self):
        profile = harness._parse_profile(
            "agentic_ollama_only",
            {
                "windows": [1, 100],
                "measured_repeats": 1,
                "prompt": "x",
                "engines": [_engine("fake", measured_calls=[{"n_tokens": 100, "yields_windows": [1, 100]}])],
            },
        )
        adapter = _FakeAdapter()  # component_ns=None
        with self.assertRaisesRegex(harness.AdapterContractError, "component_ns is None"):
            harness.run_profile(
                profile, {"fake": adapter}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
            )

    def test_measured_call_multi_yield_missing_window_entry_raises(self):
        profile = harness._parse_profile(
            "agentic_ollama_only",
            {
                "windows": [1, 100],
                "measured_repeats": 1,
                "prompt": "x",
                "engines": [_engine("fake", measured_calls=[{"n_tokens": 100, "yields_windows": [1, 100]}])],
            },
        )
        adapter = _FakeAdapter(component_ns={100: 400_000_000})  # window=1 missing
        with self.assertRaisesRegex(harness.AdapterContractError, r"missing entries for window\(s\) \[1\]"):
            harness.run_profile(
                profile, {"fake": adapter}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
            )

    def test_agentic_style_global_flattened_sequence_lattice_ollama_mlx(self):
        """The round-3 blocker fix, end to end: reproduce
        `bench_compare_1k.py`'s complete Lattice -> Ollama -> MLX stream in
        one profile. Lattice is window-major (all 1-token then all
        100-token calls, the default shape). Ollama warms once at 4
        tokens with its own 'hi' prompt, then issues ONE 100-token call
        per repeat yielding both a window=1 (TTFT) and window=100 (total)
        observation via component_ns. MLX warms once at 4 tokens with its
        own 'BASE' prompt, then alternates (1, 100) every repeat
        (repeat-major)."""
        profile = harness._parse_profile(
            "agentic",
            {
                "windows": [1, 100],
                "measured_repeats": 2,
                "prompt": "the padded ctx prompt",
                "engines": [
                    _engine("lattice", model="qwen3.5-0.8b", quantization="q8"),
                    _engine(
                        "ollama",
                        warmup_repeats=1,
                        warmup_tokens=4,
                        warmup_prompt="hi",
                        model="qwen3.5:0.8b",
                        quantization="q8",
                        measured_calls=[{"n_tokens": 100, "yields_windows": [1, 100]}],
                    ),
                    _engine(
                        "mlx",
                        warmup_repeats=1,
                        warmup_tokens=4,
                        warmup_prompt="BASE",
                        model="qwen3.5-0.8b",
                        quantization="q8",
                        measured_order="repeat_major",
                        measured_calls=[
                            {"n_tokens": 1, "yields_windows": [1]},
                            {"n_tokens": 100, "yields_windows": [100]},
                        ],
                    ),
                ],
            },
        )
        lattice = _FakeAdapter()
        ollama = _FakeAdapter(component_ns={1: 40_000_000, 100: 400_000_000})
        mlx = _FakeAdapter()
        result = harness.run_profile(
            profile,
            {"lattice": lattice, "ollama": ollama, "mlx": mlx},
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )

        # Lattice: no warmup, all 1-token calls then all 100-token calls.
        self.assertEqual(
            lattice.calls,
            [
                (1, False, "qwen3.5-0.8b", "q8"),
                (1, False, "qwen3.5-0.8b", "q8"),
                (100, False, "qwen3.5-0.8b", "q8"),
                (100, False, "qwen3.5-0.8b", "q8"),
            ],
        )
        # Ollama: one 4-token 'hi' warmup, then exactly TWO physical
        # 100-token calls (one per repeat) -- never four.
        self.assertEqual(
            ollama.calls,
            [
                (4, True, "qwen3.5:0.8b", "q8"),
                (100, False, "qwen3.5:0.8b", "q8"),
                (100, False, "qwen3.5:0.8b", "q8"),
            ],
        )
        # MLX: one 4-token 'BASE' warmup, then (1, 100) alternating per repeat.
        self.assertEqual(
            mlx.calls,
            [
                (4, True, "qwen3.5-0.8b", "q8"),
                (1, False, "qwen3.5-0.8b", "q8"),
                (100, False, "qwen3.5-0.8b", "q8"),
                (1, False, "qwen3.5-0.8b", "q8"),
                (100, False, "qwen3.5-0.8b", "q8"),
            ],
        )

        # Global observation stream: all of lattice's rows (4, no warmup),
        # then all of ollama's (1 warmup + 4 measured -- two calls, two
        # yielded windows each), then all of mlx's (1 warmup + 4 measured).
        self.assertEqual(
            [o.engine for o in result.observations],
            ["lattice"] * 4 + ["ollama"] * 5 + ["mlx"] * 5,
        )
        # Ollama's measured rows: window=1 (TTFT-equivalent, from
        # component_ns) immediately followed by window=100 (total), twice.
        ollama_measured = [o for o in result.observations if o.engine == "ollama" and not o.warmup]
        self.assertEqual([o.requested_completion_tokens for o in ollama_measured], [1, 100, 1, 100])
        self.assertEqual(
            [o.elapsed_ns for o in ollama_measured], [40_000_000, 400_000_000, 40_000_000, 400_000_000]
        )
        # MLX's measured rows alternate (1, 100) per repeat -- repeat-major.
        mlx_measured = [o for o in result.observations if o.engine == "mlx" and not o.warmup]
        self.assertEqual([o.requested_completion_tokens for o in mlx_measured], [1, 100, 1, 100])

    def test_precise_style_global_flattened_sequence_all_three_engines(self):
        """`bench_apples_precise.sh` reproduced with its actual three
        engines (Lattice, Ollama, MLX), each warming identically -- twice
        at N2=512, once before both measured windows -- confirming the
        warmup mechanism holds across every engine in the real script, not
        just a single stand-in adapter."""
        profile = harness._parse_profile(
            "precise",
            {
                "windows": [64, 512],
                "measured_repeats": 3,
                "prompt": "the quick brown fox",
                "engines": [
                    _engine("lattice", warmup_repeats=2, warmup_tokens=512, model="qwen3.5-0.8b", quantization="q8"),
                    _engine("ollama", warmup_repeats=2, warmup_tokens=512, model="qwen3.5:0.8b", quantization="q8"),
                    _engine("mlx", warmup_repeats=2, warmup_tokens=512, model="qwen3.5-0.8b", quantization="q8"),
                ],
            },
        )
        lattice, ollama, mlx = _FakeAdapter(), _FakeAdapter(), _FakeAdapter()
        result = harness.run_profile(
            profile,
            {"lattice": lattice, "ollama": ollama, "mlx": mlx},
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )
        expected_per_engine = (
            [(512, True, "qwen3.5-0.8b", "q8")] * 2
            + [(64, False, "qwen3.5-0.8b", "q8")] * 3
            + [(512, False, "qwen3.5-0.8b", "q8")] * 3
        )
        expected_ollama = (
            [(512, True, "qwen3.5:0.8b", "q8")] * 2
            + [(64, False, "qwen3.5:0.8b", "q8")] * 3
            + [(512, False, "qwen3.5:0.8b", "q8")] * 3
        )
        self.assertEqual(lattice.calls, expected_per_engine)
        self.assertEqual(ollama.calls, expected_ollama)
        self.assertEqual(mlx.calls, expected_per_engine)
        self.assertEqual(
            [o.engine for o in result.observations],
            ["lattice"] * 8 + ["ollama"] * 8 + ["mlx"] * 8,
        )

    def test_context_scaling_style_global_flattened_sequence_all_three_engines(self):
        """`bench_context_scaling.sh` reproduced with its actual three
        engines: only MLX warms (once, at 4 tokens, before the N1=8
        baseline and every N2 comparison window) -- Lattice and Ollama
        have no separate warmup step in this script."""
        profile = harness._parse_profile(
            "context_scaling",
            {
                "windows": [8, 64, 128, 256],
                "measured_repeats": 2,
                "prompt": "the quick brown fox",
                "engines": [
                    _engine("lattice", warmup_repeats=0, model="qwen3.5-0.8b", quantization="q8"),
                    _engine("ollama", warmup_repeats=0, model="qwen3.5:0.8b", quantization="q8"),
                    _engine("mlx", warmup_repeats=1, warmup_tokens=4, model="qwen3.5-0.8b", quantization="q8"),
                ],
            },
        )
        lattice, ollama, mlx = _FakeAdapter(), _FakeAdapter(), _FakeAdapter()
        result = harness.run_profile(
            profile,
            {"lattice": lattice, "ollama": ollama, "mlx": mlx},
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )
        no_warmup_expected = (
            [(8, False, "qwen3.5-0.8b", "q8")] * 2
            + [(64, False, "qwen3.5-0.8b", "q8")] * 2
            + [(128, False, "qwen3.5-0.8b", "q8")] * 2
            + [(256, False, "qwen3.5-0.8b", "q8")] * 2
        )
        self.assertEqual(lattice.calls, no_warmup_expected)
        self.assertEqual(
            ollama.calls,
            [(c[0], c[1], "qwen3.5:0.8b", "q8") for c in no_warmup_expected],
        )
        self.assertEqual(
            mlx.calls,
            [(4, True, "qwen3.5-0.8b", "q8")] + no_warmup_expected,
        )
        self.assertEqual(
            [o.engine for o in result.observations],
            ["lattice"] * 8 + ["ollama"] * 8 + ["mlx"] * 9,
        )


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


class AggregateTest(unittest.TestCase):
    def test_median_slope_matches_hand_computed_value(self):
        # Fixed 1ms-per-call harness clock overhead is negligible next to the
        # adapter's own synthetic timing, so drive elapsed_ns directly via a
        # clock that advances by a known amount per adapter call.
        profile = _profile(measured_repeats=3, windows=[32, 256], aggregation="median")
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
        self.assertIsNone(slopes[0].slope_ci95_legacy)

    def test_trimmed_mean_stat_exact_with_asymmetric_outliers(self):
        # Direct unit test of the trimming primitive: a low and a high
        # outlier that trimming must remove for the trimmed mean to differ
        # measurably from the plain mean. Mutation `trimmed = ordered`
        # (dropping the trim entirely) changes this result from 3.2 to
        # 4.52 -- verified locally by reverting the trim slice and
        # confirming this test fails, then restoring it.
        values = [3.0, 3.1, 3.2, 3.3, 10.0]
        trimmed_subset = [3.1, 3.2, 3.3]
        expected_mean = statistics.mean(trimmed_subset)
        expected_ci_legacy = 1.96 * statistics.stdev(trimmed_subset) / math.sqrt(len(trimmed_subset))

        mean, ci_legacy = harness._trimmed_mean_stat(values, trim=1)

        self.assertAlmostEqual(mean, expected_mean, places=10)
        self.assertAlmostEqual(ci_legacy, expected_ci_legacy, places=10)
        # The untrimmed mean is far enough away that a trim-removal
        # mutation cannot accidentally still pass these assertions.
        self.assertNotAlmostEqual(mean, statistics.mean(values), places=1)

    def test_trimmed_mean_slope_exact_with_asymmetric_outliers(self):
        profile = _profile(measured_repeats=5, windows=[32, 256], aggregation="trimmed_mean", trim=1)
        # Both windows carry one high outlier each so the plain mean and
        # the trimmed mean diverge; the trimmed subset is known exactly, so
        # the expected slope and legacy CI are computable by hand from it.
        baseline_steps = [3.0e9, 3.1e9, 3.2e9, 3.3e9, 10.0e9]
        window_steps = [25.0e9, 25.1e9, 25.2e9, 25.3e9, 40.0e9]
        clock = _StepClock(baseline_steps + window_steps)
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter()}, clock=clock, git_sha_value="x", hardware_id_value="h"
        )
        slopes = harness.aggregate(result)
        self.assertEqual(len(slopes), 1)

        trimmed_baseline = [3.1, 3.2, 3.3]
        trimmed_window = [25.1, 25.2, 25.3]
        expected_baseline_mean = statistics.mean(trimmed_baseline)
        expected_window_mean = statistics.mean(trimmed_window)
        expected_baseline_ci = 1.96 * statistics.stdev(trimmed_baseline) / math.sqrt(len(trimmed_baseline))
        expected_window_ci = 1.96 * statistics.stdev(trimmed_window) / math.sqrt(len(trimmed_window))
        expected_slope = (256 - 32) / (expected_window_mean - expected_baseline_mean)
        expected_ci_legacy = expected_slope * math.sqrt(
            (expected_baseline_ci / expected_baseline_mean) ** 2
            + (expected_window_ci / expected_window_mean) ** 2
        )

        self.assertAlmostEqual(slopes[0].slope_tok_per_s, expected_slope, places=6)
        self.assertIsNotNone(slopes[0].slope_ci95_legacy)
        self.assertAlmostEqual(slopes[0].slope_ci95_legacy, expected_ci_legacy, places=6)

    def test_multi_window_produces_one_slope_per_comparison_window(self):
        profile = _profile(measured_repeats=2, windows=[8, 64, 128])
        result = harness.run_profile(
            profile, {"fake": _FakeAdapter()}, clock=_FakeClock(), git_sha_value="x", hardware_id_value="h"
        )
        slopes = harness.aggregate(result)
        self.assertEqual({s.window for s in slopes}, {64, 128})
        self.assertTrue(all(s.baseline_window == 8 for s in slopes))

    def test_native_throughput_uses_actual_completion_tokens(self):
        profile = _profile(measured_repeats=1, windows=[32, 256])
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
        profile = _profile(measured_repeats=1, windows=[32, 256])
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
        profile = _profile(engines=[_engine("fake"), _engine("ghost")])
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
        profile = _profile(engines=[_engine("ghost")])
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
measured_repeats = 1
prompt = "hello"

[[profiles.smoke.engines]]
name = "nonexistent-engine"
warmup_repeats = 0
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
measured_repeats = 1
prompt = "hello"

[[profiles.smoke.engines]]
name = "nonexistent-engine"
warmup_repeats = 0
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
