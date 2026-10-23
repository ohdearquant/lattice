"""Deterministic tests for scripts/bench_decode_adapters_context_scaling.py
(issue #813 step 2: migrate bench_context_scaling.sh, keeping its chart
renderer).

Run with: python3 tests/test_bench_decode_adapters_context_scaling.py -v
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"

sys.path.insert(0, str(_SCRIPTS))


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harness = _load_module("bench_decode_harness", "bench_decode_harness.py")
adapters = _load_module("bench_decode_adapters_context_scaling", "bench_decode_adapters_context_scaling.py")
chart = adapters.chart

DEFAULT_PROFILES_FILE = _SCRIPTS / "bench_decode_profiles.toml"


@dataclass
class _FakeClock:
    step_ns: int = 1_000_000
    start_ns: int = 0
    _calls: int = 0

    def __call__(self) -> int:
        value = self.start_ns + self.step_ns * self._calls
        self._calls += 1
        return value


class _FakeAdapter:
    def __init__(self):
        self.calls: list[tuple[int, bool, str, str]] = []

    def run(self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str):
        self.calls.append((n_tokens, warmup, model, quantization))
        return harness.AdapterRunResult(
            actual_completion_tokens=n_tokens, native_ns=n_tokens * 1000, engine_version="fake-1.0"
        )


class _SequencedNativeAdapter:
    """Returns a pre-scripted `native_ns` per call, in call order, ignoring
    the wall clock entirely -- lets a test drive exact elapsed-seconds
    values for `compute_context_scaling_aggregate` without choreographing a
    `_StepClock`."""

    def __init__(self, native_ns_sequence: list[int]):
        self._durations = iter(native_ns_sequence)

    def run(self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str):
        return harness.AdapterRunResult(
            actual_completion_tokens=n_tokens,
            native_ns=next(self._durations),
            engine_version="fake-native",
        )


class ProfileParameterTranscriptionTest(unittest.TestCase):
    """bench_context_scaling.sh: N1=8 baseline, default CONTEXTS=(64 128
    256), RUNS=5, only mlx warms (4 tokens, once, before the whole loop)."""

    @classmethod
    def setUpClass(cls):
        _, cls.profiles = harness.load_profiles_file(DEFAULT_PROFILES_FILE)

    def test_profile_defined(self):
        self.assertIn("context_scaling", self.profiles)

    def test_windows_match_legacy_n1_and_default_contexts(self):
        p = self.profiles["context_scaling"]
        self.assertEqual(p.windows, (8, 64, 128, 256))
        self.assertEqual(p.measured_repeats, 5)

    def test_only_mlx_has_a_warmup(self):
        p = self.profiles["context_scaling"]
        groups = {g.name: g for g in p.engine_groups}
        self.assertEqual(groups["lattice"].warmup_repeats, 0)
        self.assertEqual(groups["ollama"].warmup_repeats, 0)
        self.assertEqual(groups["mlx"].warmup_repeats, 1)
        self.assertEqual(groups["mlx"].warmup_tokens, 4)

    def test_all_engines_q8(self):
        p = self.profiles["context_scaling"]
        for group in p.engine_groups:
            self.assertEqual(group.quantization, "q8", group.name)


class CallScheduleContractTest(unittest.TestCase):
    """Proves the default (no measured_calls override) window-major replay
    of profile.windows reproduces "measure N1=8 once, then measure each
    context length in turn, reusing the N1 baseline for every slope" --
    exactly bench_context_scaling.sh's structure, with no measured_calls
    override needed (see bench_decode_profiles.toml's comment)."""

    @classmethod
    def setUpClass(cls):
        _, cls.profiles = harness.load_profiles_file(DEFAULT_PROFILES_FILE)

    def test_call_sequence_baseline_then_each_context_in_order(self):
        profile = self.profiles["context_scaling"]
        lattice = _FakeAdapter()
        harness.run_profile(
            profile,
            {"lattice": lattice},
            allow_missing_engine=True,
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )
        self.assertEqual(
            lattice.calls,
            [(8, False, "qwen3.5-0.8b", "q8")] * 5
            + [(64, False, "qwen3.5-0.8b", "q8")] * 5
            + [(128, False, "qwen3.5-0.8b", "q8")] * 5
            + [(256, False, "qwen3.5-0.8b", "q8")] * 5,
        )

    def test_every_later_window_slopes_against_the_same_n1_baseline(self):
        profile = self.profiles["context_scaling"]
        result = harness.run_profile(
            profile, {"lattice": _FakeAdapter()}, allow_missing_engine=True, clock=_FakeClock()
        )
        slopes = harness.aggregate(result)
        self.assertEqual({s.baseline_window for s in slopes}, {8})
        self.assertEqual({s.window for s in slopes}, {64, 128, 256})

    def test_produces_schema_valid_observations(self):
        profile = self.profiles["context_scaling"]
        result = harness.run_profile(
            profile,
            {"lattice": _FakeAdapter(), "ollama": _FakeAdapter(), "mlx": _FakeAdapter()},
            clock=_FakeClock(),
        )
        for obs in result.observations:
            harness.validate_observation(obs.to_dict())


class LegacyTsvRenderingTest(unittest.TestCase):
    """render_legacy_tsv must produce the exact
    engine/context_tokens/slope_tok_s/... header
    scripts/bench_context_scaling_chart.py parses positionally."""

    def test_header_and_columns(self):
        _, profiles = harness.load_profiles_file(DEFAULT_PROFILES_FILE)
        profile = profiles["context_scaling"]
        result = harness.run_profile(
            profile,
            {"lattice": _FakeAdapter(), "ollama": _FakeAdapter(), "mlx": _FakeAdapter()},
            clock=_FakeClock(),
        )
        slopes, medians = adapters.compute_context_scaling_aggregate(result)
        tsv = adapters.render_legacy_tsv(result, slopes, medians)
        lines = [line for line in tsv.strip("\n").split("\n") if not line.startswith("#")]
        self.assertEqual(lines[0], "engine\tcontext_tokens\tslope_tok_s\tt1_ms\tt2_ms\truns\tscope")
        self.assertEqual(len(lines), 1 + len(slopes))
        for line in lines[1:]:
            parts = line.split("\t")
            self.assertEqual(len(parts), 7)
            int(parts[1])  # context_tokens parses as int
            float(parts[2])  # slope_tok_s parses as float
            if parts[0] == "ollama":
                self.assertEqual(parts[6], harness.OLLAMA_SCOPE)
            else:
                self.assertEqual(parts[6], "")


class ContextsAndRunsOverrideTest(unittest.TestCase):
    """Proves the wrapper's --contexts/--runs flags (forwarded from the
    legacy CONTEXTS/RUNS env vars by scripts/bench_context_scaling.sh)
    rebuild an equivalent ProfileConfig without mutating the on-disk
    profile."""

    def test_contexts_override_replaces_windows_keeping_n1(self):
        import dataclasses

        _, profiles = harness.load_profiles_file(DEFAULT_PROFILES_FILE)
        profile = profiles["context_scaling"]
        overridden = dataclasses.replace(profile, windows=(8, 512, 1024))
        self.assertEqual(overridden.windows, (8, 512, 1024))
        self.assertEqual(overridden.measured_repeats, profile.measured_repeats)

    def test_runs_override_replaces_measured_repeats_only(self):
        import dataclasses

        _, profiles = harness.load_profiles_file(DEFAULT_PROFILES_FILE)
        profile = profiles["context_scaling"]
        overridden = dataclasses.replace(profile, measured_repeats=10)
        self.assertEqual(overridden.measured_repeats, 10)
        self.assertEqual(overridden.windows, profile.windows)


class ParseContextsTest(unittest.TestCase):
    """issue #813 codex round-1 finding 2a: --contexts must preserve caller
    order (never sort) and reject a duplicated baseline/internal
    duplicate."""

    def test_preserves_caller_order_not_sorted(self):
        self.assertEqual(adapters.parse_contexts("256,64,128", 8), (256, 64, 128))

    def test_skips_blank_entries_and_whitespace(self):
        self.assertEqual(adapters.parse_contexts(" 64, ,128 ", 8), (64, 128))

    def test_rejects_internal_duplicate(self):
        with self.assertRaises(adapters.ContextsConfigError):
            adapters.parse_contexts("64,128,64", 8)

    def test_rejects_context_equal_to_baseline(self):
        with self.assertRaises(adapters.ContextsConfigError):
            adapters.parse_contexts("8,64", 8)

    def test_rejects_non_positive_entry(self):
        with self.assertRaises(adapters.ContextsConfigError):
            adapters.parse_contexts("0,64", 8)

    def test_rejects_non_integer_entry(self):
        with self.assertRaises(adapters.ContextsConfigError):
            adapters.parse_contexts("64,abc", 8)

    def test_rejects_empty(self):
        with self.assertRaises(adapters.ContextsConfigError):
            adapters.parse_contexts("", 8)


class LegacyOrderedMedianTest(unittest.TestCase):
    """issue #813 codex round-1 finding 2b: the exact per-engine even-n
    tie-break `bench_context_scaling.sh`'s lattice_median/ollama_median
    (lower middle) and MLX heredoc (upper middle) implement."""

    def test_odd_n_matches_true_median_for_every_engine(self):
        values = [5.0, 1.0, 3.0, 2.0, 4.0]
        for engine in ("lattice", "ollama", "mlx"):
            self.assertEqual(adapters._legacy_ordered_median(values, engine), 3.0)

    def test_even_n_lattice_and_ollama_take_lower_middle(self):
        # sorted: [1,2,3,4,5,6] -- lower middle (1-indexed position 3) = 3.0
        values = [6.0, 2.0, 4.0, 1.0, 5.0, 3.0]
        self.assertEqual(adapters._legacy_ordered_median(values, "lattice"), 3.0)
        self.assertEqual(adapters._legacy_ordered_median(values, "ollama"), 3.0)

    def test_even_n_mlx_takes_upper_middle(self):
        values = [6.0, 2.0, 4.0, 1.0, 5.0, 3.0]
        self.assertEqual(adapters._legacy_ordered_median(values, "mlx"), 4.0)

    def test_empty_values_rejected(self):
        with self.assertRaises(ValueError):
            adapters._legacy_ordered_median([], "lattice")


class ContextScalingEvenRunsLegacyMedianTest(unittest.TestCase):
    """issue #813 codex round-1 finding 2b/2c, end to end: with an even
    RUNS override, lattice takes the LOWER middle sample and mlx takes the
    UPPER middle -- `statistics.median()` would instead average them -- and
    the TSV's t1_ms/t2_ms must come from that SAME aggregate, never an
    independently recomputed mean."""

    @staticmethod
    def _profile(measured_repeats: int, windows: list[int]):
        raw = {
            "description": "test",
            "windows": windows,
            "measured_repeats": measured_repeats,
            "engines": [
                {"name": "lattice", "warmup_repeats": 0, "model": "m", "quantization": "q8"},
                {"name": "mlx", "warmup_repeats": 0, "model": "m", "quantization": "q8"},
            ],
            "prompt": "test prompt",
        }
        return harness._parse_profile("unit_test_ctx", raw)

    def test_even_runs_lower_vs_upper_middle_and_tsv_consistency(self):
        profile = self._profile(measured_repeats=6, windows=[8, 64])
        # sorted ms: [10,20,30,40,50,60] -- lower middle=30, upper middle=40
        baseline_ms = [50, 10, 60, 20, 40, 30]
        # sorted ms: [100,...,600] -- lower middle=300, upper middle=400
        window_ms = [100, 200, 300, 400, 500, 600]
        sequence = [int(v * 1e6) for v in baseline_ms + window_ms]

        result = harness.run_profile(
            profile,
            {
                "lattice": _SequencedNativeAdapter(list(sequence)),
                "mlx": _SequencedNativeAdapter(list(sequence)),
            },
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )
        slopes, medians = adapters.compute_context_scaling_aggregate(result)

        self.assertAlmostEqual(medians[("lattice", 8)], 0.030, places=9)
        self.assertAlmostEqual(medians[("lattice", 64)], 0.300, places=9)
        self.assertAlmostEqual(medians[("mlx", 8)], 0.040, places=9)
        self.assertAlmostEqual(medians[("mlx", 64)], 0.400, places=9)
        # statistics.median() would average the two middle values (35ms /
        # 350ms for BOTH engines) -- assert we are not doing that.
        self.assertNotAlmostEqual(medians[("lattice", 8)], 0.035, places=6)
        self.assertNotAlmostEqual(medians[("mlx", 8)], 0.035, places=6)

        lattice_slope = next(s for s in slopes if s.engine == "lattice")
        mlx_slope = next(s for s in slopes if s.engine == "mlx")
        self.assertAlmostEqual(lattice_slope.slope_tok_per_s, 56 / 0.27, places=6)
        self.assertAlmostEqual(mlx_slope.slope_tok_per_s, 56 / 0.36, places=6)

        tsv = adapters.render_legacy_tsv(result, slopes, medians)
        for line in [line for line in tsv.strip("\n").split("\n") if not line.startswith("#")][1:]:
            engine, _ctx, _slope, t1_ms, t2_ms, _runs, _scope = line.split("\t")
            if engine == "lattice":
                self.assertAlmostEqual(float(t1_ms), 30.0, places=3)
                self.assertAlmostEqual(float(t2_ms), 300.0, places=3)
            else:
                self.assertAlmostEqual(float(t1_ms), 40.0, places=3)
                self.assertAlmostEqual(float(t2_ms), 400.0, places=3)


class ContextScalingSkewedFiveSampleTsvTest(unittest.TestCase):
    """issue #813 codex round-1 finding 2c, at the DEFAULT odd RUNS=5: the
    legacy script's t1_ms/t2_ms were always medians, never means -- a
    single high outlier in an otherwise tight five-sample set must not
    move the rendered value."""

    def test_tsv_uses_median_not_mean_when_skewed(self):
        raw = {
            "description": "test",
            "windows": [8, 64],
            "measured_repeats": 5,
            "engines": [
                {"name": "lattice", "warmup_repeats": 0, "model": "m", "quantization": "q8"}
            ],
            "prompt": "test prompt",
        }
        profile = harness._parse_profile("unit_test_skew", raw)
        baseline_ms = [10, 130, 15, 25, 20]  # median=20, mean=40
        window_ms = [200, 210, 205, 195, 800]  # median=205, mean=322
        adapter = _SequencedNativeAdapter([int(v * 1e6) for v in baseline_ms + window_ms])
        result = harness.run_profile(
            profile,
            {"lattice": adapter},
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )
        slopes, medians = adapters.compute_context_scaling_aggregate(result)
        tsv = adapters.render_legacy_tsv(result, slopes, medians)
        line = [line for line in tsv.strip("\n").split("\n") if not line.startswith("#")][1]
        _engine, _ctx, _slope, t1_ms, t2_ms, _runs, _scope = line.split("\t")
        self.assertAlmostEqual(float(t1_ms), 20.0, places=3)
        self.assertAlmostEqual(float(t2_ms), 205.0, places=3)
        self.assertNotAlmostEqual(float(t1_ms), 40.0, places=1)
        self.assertNotAlmostEqual(float(t2_ms), 322.0, places=1)


def _base_numeric_tsv(run_result, slopes, window_medians):
    """Frozen render_legacy_tsv body from d94daf81530b42cac7871166aaa1c8859b4d7c6d."""
    lines = ["engine\tcontext_tokens\tslope_tok_s\tt1_ms\tt2_ms\truns\tscope"]
    baseline_window = run_result.profile.windows[0]
    for s in slopes:
        t1_ms = window_medians[(s.engine, baseline_window)] * 1000
        t2_ms = window_medians[(s.engine, s.window)] * 1000
        scope = harness.OLLAMA_SCOPE if s.engine == "ollama" else ""
        lines.append(
            f"{s.engine}\t{s.window}\t{s.slope_tok_per_s:.1f}\t{t1_ms:.3f}\t{t2_ms:.3f}\t{s.window_n}\t{scope}"
        )
    return "\n".join(lines) + "\n"


class ProvenanceRenderingTest(unittest.TestCase):
    def setUp(self):
        _, profiles = harness.load_profiles_file(DEFAULT_PROFILES_FILE)
        self.result = harness.run_profile(
            profiles["context_scaling"],
            {"lattice": _FakeAdapter(), "ollama": _FakeAdapter(), "mlx": _FakeAdapter()},
            clock=_FakeClock(),
            git_sha_value="a" * 40,
            hardware_id_value="Linux-x86_64-recorded-host",
            timestamp_fn=lambda: "2001-02-03T04:05:06+00:00",
        )
        self.slopes, self.medians = adapters.compute_context_scaling_aggregate(self.result)

    def render(self, result=None):
        return adapters.render_legacy_tsv(self.result if result is None else result, self.slopes, self.medians)

    def test_numeric_output_matches_frozen_base_formatter_byte_for_byte(self):
        rendered = self.render()
        numeric = "".join(line for line in rendered.splitlines(keepends=True) if not line.startswith("#"))
        baseline = _base_numeric_tsv(self.result, self.slopes, self.medians)
        self.assertEqual(numeric.encode(), baseline.encode())
        self.assertEqual(rendered.splitlines()[0], baseline.splitlines()[0])
        self.assertEqual(len(numeric.splitlines()), 1 + len(self.slopes))
        self.assertTrue(all(len(line.split("\t")) == 7 for line in numeric.splitlines()))

    def test_metadata_records_original_measurement_timestamp(self):
        _, provenance = chart.parse_tsv(self.render())
        self.assertEqual(provenance["schema_version"], 1)
        self.assertEqual(provenance["observation_schema_versions"], [harness.SCHEMA_VERSION])
        self.assertEqual(provenance["harness"], "bench_decode_harness.py")
        self.assertEqual(provenance["git_shas"], ["a" * 40])
        self.assertEqual(provenance["hardware_ids"], ["Linux-x86_64-recorded-host"])
        self.assertEqual(provenance["observation_timestamps"], ["2001-02-03T04:05:06+00:00"])

    def test_all_distinct_recorded_identities_and_timestamps_are_retained(self):
        observations = list(self.result.observations)
        observations[-1] = dataclasses.replace(
            observations[-1], git_sha="b" * 40, hardware_id="Darwin-arm64-other-host",
            timestamp="2001-02-04T04:05:06+00:00",
        )
        mixed = dataclasses.replace(self.result, observations=tuple(observations))
        _, provenance = chart.parse_tsv(self.render(mixed))
        self.assertEqual(provenance["git_shas"], ["a" * 40, "b" * 40])
        self.assertEqual(provenance["hardware_ids"], ["Darwin-arm64-other-host", "Linux-x86_64-recorded-host"])
        self.assertEqual(provenance["observation_timestamps"], [
            "2001-02-03T04:05:06+00:00", "2001-02-04T04:05:06+00:00",
        ])

    def test_metadata_escapes_control_characters_and_round_trips(self):
        hardware = 'host\twith\n"quotes" and $math$'
        observations = tuple(dataclasses.replace(obs, hardware_id=hardware) for obs in self.result.observations)
        rendered = self.render(dataclasses.replace(self.result, observations=observations))
        comments = [line for line in rendered.splitlines() if line.startswith("#")]
        self.assertEqual(len(comments), 1)
        self.assertNotIn("\t", comments[0])
        self.assertIn("\\t", comments[0])
        self.assertIn("\\n", comments[0])
        _, provenance = chart.parse_tsv(rendered)
        self.assertEqual(provenance["hardware_ids"], [hardware])

    def test_new_reports_refuse_missing_or_invalid_recorded_provenance(self):
        with self.assertRaises(ValueError):
            self.render(dataclasses.replace(self.result, observations=()))
        for field, value in (("timestamp", ""), ("timestamp", "not-a-date"),
                             ("timestamp", "2001-02-03T04:05:06"), ("hardware_id", ""), ("git_sha", "")):
            with self.subTest(field=field, value=value):
                observations = tuple(dataclasses.replace(obs, **{field: value}) for obs in self.result.observations)
                with self.assertRaises(ValueError):
                    self.render(dataclasses.replace(self.result, observations=observations))


class ChartProvenanceTest(unittest.TestCase):
    HEADER = "engine\tcontext_tokens\tslope_tok_s\tt1_ms\tt2_ms\truns\tscope\n"
    ROW = "lattice\t64\t123.4\t1.000\t2.000\t5\t\n"

    def setUp(self):
        self.metadata = {
            "schema_version": 1,
            "harness": "bench_decode_harness.py",
            "observation_schema_versions": [1],
            "git_shas": ["a" * 40, "b" * 40],
            "hardware_ids": ["Linux-x86_64-recorded-host", "host\twith\n$math$"],
            "observation_timestamps": ["2001-02-03T04:05:06+00:00", "2001-02-04T04:05:06+00:00"],
        }

    def tsv(self, comments):
        return self.HEADER + "\n".join(comments) + "\n" + self.ROW

    def test_saved_metadata_is_used_by_chart_only(self):
        text = self.tsv([chart.provenance_comment(self.metadata)])
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "old.tsv", Path(tmp) / "chart.png"
            source.write_text(text)
            with mock.patch.object(chart, "render_chart") as render:
                self.assertEqual(chart.main([str(source), str(output)]), 0)
            render.assert_called_once_with({"lattice": {64: 123.4}}, self.metadata, output)

    def test_legacy_six_column_and_historical_comment_files_remain_readable(self):
        legacy = "engine\tcontext_tokens\tslope_tok_s\tt1_ms\tt2_ms\truns\n"
        legacy += "# PROVENANCE (2026-07-20): historical freeform prose\n"
        legacy += "# comment\twith\ttabs\n"
        legacy += "lattice\t64\t123.4\t1.000\t2.000\t5\n"
        engines, provenance = chart.parse_tsv(legacy)
        self.assertEqual(engines, {"lattice": {64: 123.4}})
        self.assertIsNone(provenance)
        self.assertIn("timestamps, hardware and harness revision are unknown", chart.provenance_caption(provenance))
        committed = (_REPO_ROOT / "docs/bench_results/context_scaling.tsv").read_text()
        engines, provenance = chart.parse_tsv(committed)
        self.assertEqual(set(engines), {"lattice", "ollama", "mlx"})
        self.assertIsNone(provenance)

    def test_malformed_or_unsupported_metadata_is_not_legacy_unknown(self):
        invalid = [
            {}, [], {**self.metadata, "schema_version": 2}, {**self.metadata, "schema_version": True},
            {**self.metadata, "extra": "field"}, {**self.metadata, "harness": "other.py"},
            {**self.metadata, "hardware_ids": []}, {**self.metadata, "hardware_ids": ["same", "same"]},
            {**self.metadata, "git_shas": [""]}, {**self.metadata, "git_shas": "a" * 40},
            {**self.metadata, "observation_schema_versions": [True]},
            {**self.metadata, "observation_timestamps": ["bad"]},
        ]
        for value in invalid:
            with self.subTest(value=value):
                text = self.tsv([chart.PROVENANCE_PREFIX + json.dumps(value)])
                with self.assertRaises(ValueError):
                    chart.parse_tsv(text)
        for comment in (chart.PROVENANCE_PREFIX + "{", chart.PROVENANCE_TAG + " missing colon"):
            with self.subTest(comment=comment), self.assertRaises(ValueError):
                chart.parse_tsv(self.tsv([comment]))

    def test_duplicate_and_conflicting_metadata_records_and_fields_are_rejected(self):
        first = chart.provenance_comment(self.metadata)
        second = chart.provenance_comment({**self.metadata, "hardware_ids": ["different"]})
        for other in (first, second):
            with self.subTest(other=other), self.assertRaises(ValueError):
                chart.parse_tsv(self.tsv([first, other]))
        duplicate_key = json.dumps(self.metadata).replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
        with self.assertRaisesRegex(ValueError, "duplicate provenance field"):
            chart.parse_tsv(self.tsv([chart.PROVENANCE_PREFIX + duplicate_key]))

    def test_bad_metadata_exits_nonzero_before_rendering_or_writing_chart(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "bad.tsv", Path(tmp) / "chart.png"
            source.write_text(self.tsv([chart.PROVENANCE_PREFIX + "{}"]))
            with mock.patch.object(chart, "render_chart") as render:
                self.assertEqual(chart.main([str(source), str(output)]), 1)
            render.assert_not_called()
            self.assertFalse(output.exists())

    def test_caption_timestamp_range_uses_instants_and_retains_original_offset_text(self):
        metadata = {**self.metadata, "observation_timestamps": [
            "2000-12-31T23:30:00+00:00", "2001-01-01T01:00:00+02:00",
        ]}
        caption = chart.provenance_caption(metadata)
        self.assertIn("2001-01-01T01:00:00+02:00 to 2000-12-31T23:30:00+00:00", caption)

    def test_actual_chart_attaches_literal_recorded_or_unknown_caption(self):
        for provenance in (self.metadata, None):
            with self.subTest(provenance=provenance):
                plt, fig, ax = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
                plt.subplots.return_value = (fig, ax)
                ax.bar.return_value = []
                mpl, numpy = types.ModuleType("matplotlib"), types.ModuleType("numpy")
                mpl.pyplot = plt
                numpy.arange = mock.Mock(return_value=mock.MagicMock())
                with mock.patch.dict(sys.modules, {"matplotlib": mpl, "matplotlib.pyplot": plt, "numpy": numpy}):
                    chart.render_chart({"lattice": {64: 123.4}}, provenance, Path("chart.png"))
                title = ax.set_title.call_args.args[0]
                self.assertNotIn("Apple M2 Max", title)
                fig.text.assert_called_once()
                caption = fig.text.call_args.args[2]
                self.assertFalse(fig.text.call_args.kwargs["parse_math"])
                self.assertFalse(fig.text.call_args.kwargs["usetex"])
                if provenance is None:
                    self.assertIn("timestamps, hardware and harness revision are unknown", caption)
                    self.assertNotIn("Apple M2 Max", caption)
                else:
                    for identity in provenance["hardware_ids"] + provenance["git_shas"]:
                        self.assertIn(json.dumps(identity, ensure_ascii=True), caption)
                    for timestamp in provenance["observation_timestamps"]:
                        self.assertIn(timestamp, caption)
                    self.assertIn("bench_decode_harness.py", caption)
                    self.assertNotIn("host\twith\n", caption)
                plt.savefig.assert_called_once_with(Path("chart.png"), dpi=150, bbox_inches="tight")


class OllamaResponseParsingTest(unittest.TestCase):
    def test_valid_response(self):
        result = adapters.ollama_response_to_result(
            {"eval_count": 64, "eval_duration": 500_000_000, "total_duration": 800_000_000}
        )
        self.assertEqual(result.actual_completion_tokens, 64)
        self.assertEqual(result.native_ns, 500_000_000)

    def test_error_body_raises(self):
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result({"error": "boom"})

    def test_required_fields_keep_type_presence_and_range_validation(self):
        valid = {"eval_count": 64, "eval_duration": 500_000_000, "total_duration": 800_000_000}
        for field in valid:
            with self.subTest(field=field, missing=True):
                data = {name: value for name, value in valid.items() if name != field}
                with self.assertRaisesRegex(adapters.OllamaResponseError, field):
                    adapters.ollama_response_to_result(data)
            for invalid in (True, False, 1.5, "1", None, -1):
                with self.subTest(field=field, value=invalid):
                    with self.assertRaisesRegex(adapters.OllamaResponseError, field):
                        adapters.ollama_response_to_result({**valid, field: invalid})
        with self.assertRaisesRegex(adapters.OllamaResponseError, "total_duration.*positive"):
            adapters.ollama_response_to_result({**valid, "total_duration": 0})
        result = adapters.ollama_response_to_result({**valid, "eval_count": 0})
        self.assertEqual(result.actual_completion_tokens, 0)

    def test_zero_eval_duration_is_rejected(self):
        with self.assertRaisesRegex(adapters.OllamaResponseError, "eval_duration.*positive"):
            adapters.ollama_response_to_result(
                {"eval_count": 64, "eval_duration": 0, "total_duration": 800_000_000}
            )


class OllamaContextScalingDurationTest(unittest.TestCase):
    def _run_profile(self, load_durations, prompt_durations):
        _, profiles = harness.load_profiles_file(DEFAULT_PROFILES_FILE)
        profile = dataclasses.replace(profiles["context_scaling"], measured_repeats=1)
        calls = []

        def respond(request, *, timeout):
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
            self.assertEqual(request.get_method(), "POST")
            self.assertEqual(timeout, 600)
            payload = json.loads(request.data)
            calls.append(payload)
            index = len(calls) - 1
            tokens = payload["options"]["num_predict"]
            decode_duration = tokens * 15_625_000
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.read.return_value = json.dumps({
                "eval_count": tokens,
                "eval_duration": decode_duration,
                "total_duration": decode_duration + load_durations[index] + prompt_durations[index],
                "load_duration": load_durations[index],
                "prompt_eval_duration": prompt_durations[index],
            }).encode("utf-8")
            return response

        with mock.patch.object(adapters.urllib.request, "urlopen", side_effect=respond) as urlopen:
            result = harness.run_profile(
                profile,
                {"ollama": adapters.OllamaAdapter("http://127.0.0.1:11434")},
                allow_missing_engine=True,
                clock=_FakeClock(),
                git_sha_value="fixture-revision",
                hardware_id_value="fixture-host",
                timestamp_fn=lambda: "2001-02-03T04:05:06+00:00",
            )
        self.assertEqual(urlopen.call_count, 4)
        self.assertEqual(calls, [{
            "model": "qwen3.5:0.8b", "prompt": profile.prompt, "stream": False,
            "options": {"num_predict": tokens, "temperature": 0, "seed": 42},
        } for tokens in (8, 64, 128, 256)])
        self.assertEqual([obs.engine_native_ns for obs in result.observations], [125_000_000, 1_000_000_000, 2_000_000_000, 4_000_000_000])
        slopes, medians = adapters.compute_context_scaling_aggregate(result)
        return result, slopes, medians, adapters.render_legacy_tsv(result, slopes, medians)

    def test_cold_single_repeat_uses_decode_duration_through_tsv(self):
        result, slopes, medians, tsv = self._run_profile(
            [10_000_000_000, 0, 0, 0], [7_000_000_000, 1_000_000_000, 2_000_000_000, 3_000_000_000],
        )
        self.assertEqual([obs.requested_completion_tokens for obs in result.observations], [8, 64, 128, 256])
        self.assertEqual([obs.actual_completion_tokens for obs in result.observations], [8, 64, 128, 256])
        self.assertEqual([obs.warmup for obs in result.observations], [False] * 4)
        self.assertEqual([obs.run_index for obs in result.observations], [1] * 4)
        self.assertEqual(medians, {("ollama", 8): 0.125, ("ollama", 64): 1.0, ("ollama", 128): 2.0, ("ollama", 256): 4.0})
        self.assertEqual([(s.baseline_window, s.window, s.slope_tok_per_s, s.baseline_n, s.window_n) for s in slopes],
                         [(8, 64, 64.0, 1, 1), (8, 128, 64.0, 1, 1), (8, 256, 64.0, 1, 1)])
        _, provenance = chart.parse_tsv(tsv)
        self.assertEqual(provenance["git_shas"], ["fixture-revision"])
        self.assertEqual(provenance["hardware_ids"], ["fixture-host"])
        self.assertEqual(provenance["observation_timestamps"], ["2001-02-03T04:05:06+00:00"])
        numeric = "".join(line for line in tsv.splitlines(keepends=True) if not line.startswith("#"))
        self.assertEqual(numeric, "engine\tcontext_tokens\tslope_tok_s\tt1_ms\tt2_ms\truns\tscope\n"
                         f"ollama\t64\t64.0\t125.000\t1000.000\t1\t{harness.OLLAMA_SCOPE}\n"
                         f"ollama\t128\t64.0\t125.000\t2000.000\t1\t{harness.OLLAMA_SCOPE}\n"
                         f"ollama\t256\t64.0\t125.000\t4000.000\t1\t{harness.OLLAMA_SCOPE}\n")

    def test_decode_tsv_is_invariant_to_load_and_prompt_overhead(self):
        _, first_slopes, first_medians, first_tsv = self._run_profile(
            [10_000_000_000, 0, 0, 0], [7_000_000_000, 1_000_000_000, 2_000_000_000, 3_000_000_000],
        )
        _, second_slopes, second_medians, second_tsv = self._run_profile(
            [0, 1_234_567_890_123, 99_999_999_999, 500], [50_000_000_000, 0, 123, 9_876_543_210],
        )
        self.assertEqual(first_medians, second_medians)
        self.assertEqual(first_slopes, second_slopes)
        self.assertEqual(first_tsv, second_tsv)


class LatticeResultParsingTest(unittest.TestCase):
    def test_parses_result_line(self):
        self.assertEqual(
            adapters.parse_lattice_result_line("RESULT n_req=8 completion=8 total_ms=50.0"),
            (8, 8, 50.0),
        )

    def test_extract_single_result_rejects_wrong_window(self):
        with self.assertRaises(adapters.LatticeResultError):
            adapters.extract_single_result("RESULT n_req=8 completion=8 total_ms=1.0", n_tokens=64)


class WrapperExitStatusTest(unittest.TestCase):
    """A zero exit from the wrapper has to mean something was measured.

    `--allow-missing-engine` exists for the partial-availability case (one
    framework not installed on this box); it must not also let a run where
    every engine was absent write a report, name a chart path and exit zero.
    """

    def _run_main(self, registry: dict, tmp: Path) -> int:
        def render_chart(command, **_kwargs):
            Path(command[3]).write_bytes(b"synthetic chart")
            return subprocess.CompletedProcess(command, 0)

        saved = dict(harness.ADAPTER_REGISTRY)
        harness.ADAPTER_REGISTRY.clear()
        harness.ADAPTER_REGISTRY.update(registry)
        try:
            with (
                mock.patch.object(adapters, "register_available_adapters"),
                mock.patch.object(adapters, "REPO_ROOT", tmp),
                mock.patch.object(adapters, "OUT_DIR", tmp),
                mock.patch.object(adapters, "DATA_TSV", tmp / "data.tsv"),
                mock.patch.object(adapters, "CHART_PNG", tmp / "chart.png"),
                mock.patch.object(adapters, "subprocess") as fake_subprocess,
            ):
                fake_subprocess.run.side_effect = render_chart
                return adapters.main(["--allow-missing-engine"])
        finally:
            harness.ADAPTER_REGISTRY.clear()
            harness.ADAPTER_REGISTRY.update(saved)

    def test_all_engines_missing_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._run_main({}, Path(tmp)), 1)

    def test_one_live_engine_with_missing_peers_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._run_main({"lattice": _FakeAdapter()}, Path(tmp)), 0)


class ChartPublicationTest(unittest.TestCase):
    OLD_TSV = b"previous canonical TSV\n"
    OLD_PNG = b"previous canonical PNG\n"
    NEW_TSV = "engine\tcontext_tokens\tslope_tok_s\tt1_ms\tt2_ms\truns\tscope\nlattice\t64\t100.0\t80.000\t640.000\t1\t\n"
    NEW_PNG = b"synthetic staged PNG\n"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out_dir = self.root / "published"
        self.out_dir.mkdir()
        self.data_tsv = self.out_dir / "data.tsv"
        self.chart_png = self.out_dir / "chart.png"
        self.data_tsv.write_bytes(self.OLD_TSV)
        self.chart_png.write_bytes(self.OLD_PNG)

    def _run_main(self, *, returncode: int, create_png: bool):
        def render_chart(command, *, cwd, check):
            self.assertEqual(command[:2], [sys.executable, str(adapters.CHART_SCRIPT)])
            self.assertEqual(cwd, self.root)
            self.assertFalse(check)
            self.staged_tsv = Path(command[2])
            self.staged_png = Path(command[3])
            self.stage_dir = self.staged_tsv.parent
            self.assertEqual(self.stage_dir.parent, self.root / ".cache")
            self.assertEqual(self.staged_png.parent, self.stage_dir)
            self.assertNotEqual(self.staged_tsv, self.data_tsv)
            self.assertNotEqual(self.staged_png, self.chart_png)
            self.assertEqual(self.staged_tsv.read_bytes(), self.NEW_TSV.encode())
            if create_png:
                self.staged_png.write_bytes(self.NEW_PNG)
            return subprocess.CompletedProcess(command, returncode)

        with (
            mock.patch.object(harness, "coordinator_is_supervised", return_value=False),
            mock.patch.object(adapters, "register_available_adapters"),
            mock.patch.object(adapters, "REPO_ROOT", self.root),
            mock.patch.object(adapters, "OUT_DIR", self.out_dir),
            mock.patch.object(adapters, "DATA_TSV", self.data_tsv),
            mock.patch.object(adapters, "CHART_PNG", self.chart_png),
            mock.patch.object(harness, "run_profile", return_value=mock.sentinel.result) as run_profile,
            mock.patch.object(adapters, "compute_context_scaling_aggregate", return_value=([], {})),
            mock.patch.object(harness, "render_report", return_value="synthetic report"),
            mock.patch.object(adapters, "render_legacy_tsv", return_value=self.NEW_TSV),
            mock.patch.object(adapters, "subprocess") as fake_subprocess,
            mock.patch.object(sys, "stdout", new_callable=io.StringIO) as stdout,
            mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
        ):
            fake_subprocess.run.side_effect = render_chart
            try:
                status = adapters.main(["--allow-missing-engine"])
            except FileNotFoundError:
                self.fail("wrapper raised FileNotFoundError instead of returning a publication status")
            run_profile.assert_called_once()
            fake_subprocess.run.assert_called_once()
            return status, stdout.getvalue(), stderr.getvalue()

    def _assert_failure_preserved_outputs(self, stdout: str, stderr: str):
        self.assertEqual(self.data_tsv.read_bytes(), self.OLD_TSV)
        self.assertEqual(self.chart_png.read_bytes(), self.OLD_PNG)
        self.assertTrue(self.stage_dir.is_dir())
        self.assertEqual(self.staged_tsv.read_bytes(), self.NEW_TSV.encode())
        self.assertEqual((self.stage_dir / "report.txt").read_text(), "synthetic report\n")
        self.assertIn(f"staged data retained at {self.stage_dir}", stderr)
        self.assertNotIn("Raw data (chart-compatible TSV):", stdout)
        self.assertNotIn("Chart:", stdout)

    def test_nonzero_renderer_preserves_outputs_even_with_staged_png(self):
        status, stdout, stderr = self._run_main(returncode=23, create_png=True)
        self.assertEqual(status, 23, "renderer failure status must propagate")
        self._assert_failure_preserved_outputs(stdout, stderr)
        self.assertEqual(self.staged_png.read_bytes(), self.NEW_PNG)
        self.assertIn("FAIL: chart generation exited with status 23;", stderr)

    def test_success_without_png_refuses_publication_and_retains_stage(self):
        status, stdout, stderr = self._run_main(returncode=0, create_png=False)
        self.assertEqual(status, 1, "missing PNG must return failure")
        self._assert_failure_preserved_outputs(stdout, stderr)
        self.assertFalse(self.staged_png.exists())
        self.assertIn("FAIL: chart generation produced no PNG;", stderr)

    def test_success_publishes_exact_outputs_and_removes_stage(self):
        status, stdout, stderr = self._run_main(returncode=0, create_png=True)
        self.assertEqual(status, 0)
        self.assertEqual(self.data_tsv.read_bytes(), self.NEW_TSV.encode())
        self.assertEqual(self.chart_png.read_bytes(), self.NEW_PNG)
        self.assertFalse(self.stage_dir.exists())
        self.assertEqual(stderr, "")
        self.assertIn(f"Raw data (chart-compatible TSV): {self.data_tsv}\n", stdout)
        self.assertIn(f"Chart: {self.chart_png}\n", stdout)


if __name__ == "__main__":
    unittest.main()
