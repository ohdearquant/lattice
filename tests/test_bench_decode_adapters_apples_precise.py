"""Deterministic tests for scripts/bench_decode_adapters_apples_precise.py
(issue #813 step 2: migrate bench_apples_precise.sh onto the shared
decode-bench harness).

No engine binary or network access is required: real adapters are
exercised only through their pure parsing helpers
(`parse_lattice_result_line`, `ollama_response_to_result`); the profile ->
call-schedule contract is checked against the harness's own deterministic
fake adapters, matching
tests/test_bench_decode_adapters_apples_to_apples.py's convention.

Run with: python3 tests/test_bench_decode_adapters_apples_precise.py -v
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

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
adapters = _load_module(
    "bench_decode_adapters_apples_precise", "bench_decode_adapters_apples_precise.py"
)

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


class ProfileParameterTranscriptionTest(unittest.TestCase):
    """bench_apples_precise.sh: N1=64, N2=512, TOTAL_RUNS=15, WARMUP=2 (so
    13 measured), trimmed mean dropping top/bottom 2, every engine warms
    identically (unlike apples_to_apples's mlx-only warmup)."""

    @classmethod
    def setUpClass(cls):
        _, cls.profiles = harness.load_profiles_file(DEFAULT_PROFILES_FILE)

    def test_profile_defined(self):
        self.assertIn("apples_precise", self.profiles)

    def test_windows_and_repeats_match_legacy_n1_n2_and_measured_count(self):
        p = self.profiles["apples_precise"]
        self.assertEqual(p.windows, (64, 512))
        # TOTAL_RUNS=15, WARMUP=2 -> RUNS = TOTAL_RUNS - WARMUP = 13 measured.
        self.assertEqual(p.measured_repeats, 13)

    def test_aggregation_is_trimmed_mean_with_trim_2(self):
        p = self.profiles["apples_precise"]
        self.assertEqual(p.aggregation, "trimmed_mean")
        self.assertEqual(p.trim, 2)

    def test_prompt_matches_legacy_prompt_variable(self):
        p = self.profiles["apples_precise"]
        self.assertEqual(
            p.prompt,
            "The quick brown fox jumps over the lazy dog. Once upon a time in a land "
            "far away, there lived a",
        )

    def test_engines_and_order(self):
        p = self.profiles["apples_precise"]
        self.assertEqual(p.engines, ("lattice", "ollama", "mlx"))

    def test_every_engine_warms_twice_at_n2(self):
        p = self.profiles["apples_precise"]
        for group in p.engine_groups:
            self.assertEqual(group.warmup_repeats, 2, group.name)
            self.assertEqual(group.warmup_tokens, 512, group.name)

    def test_quantization_is_q8_for_every_engine(self):
        p = self.profiles["apples_precise"]
        for group in p.engine_groups:
            self.assertEqual(group.quantization, "q8", group.name)


class CallScheduleContractTest(unittest.TestCase):
    """Proves the produced call schedule matches bench_apples_precise.sh's
    loop structure: every engine gets 2 warmup calls at N2=512, then N1 x13,
    then N2 x13."""

    @classmethod
    def setUpClass(cls):
        _, cls.profiles = harness.load_profiles_file(DEFAULT_PROFILES_FILE)

    def test_call_sequence_every_engine_warms_before_measuring(self):
        profile = self.profiles["apples_precise"]
        lattice, ollama, mlx = _FakeAdapter(), _FakeAdapter(), _FakeAdapter()
        harness.run_profile(
            profile,
            {"lattice": lattice, "ollama": ollama, "mlx": mlx},
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )
        expected_lattice = (
            [(512, True, "qwen3.5-0.8b", "q8")] * 2
            + [(64, False, "qwen3.5-0.8b", "q8")] * 13
            + [(512, False, "qwen3.5-0.8b", "q8")] * 13
        )
        self.assertEqual(lattice.calls, expected_lattice)
        expected_ollama = (
            [(512, True, "qwen3.5:0.8b", "q8")] * 2
            + [(64, False, "qwen3.5:0.8b", "q8")] * 13
            + [(512, False, "qwen3.5:0.8b", "q8")] * 13
        )
        self.assertEqual(ollama.calls, expected_ollama)
        expected_mlx = (
            [(512, True, "qwen3.5-0.8b", "q8")] * 2
            + [(64, False, "qwen3.5-0.8b", "q8")] * 13
            + [(512, False, "qwen3.5-0.8b", "q8")] * 13
        )
        self.assertEqual(mlx.calls, expected_mlx)

    def test_produces_schema_valid_observations(self):
        profile = self.profiles["apples_precise"]
        result = harness.run_profile(
            profile,
            {"lattice": _FakeAdapter(), "ollama": _FakeAdapter(), "mlx": _FakeAdapter()},
            clock=_FakeClock(),
        )
        for obs in result.observations:
            harness.validate_observation(obs.to_dict())

    def test_trimmed_mean_aggregation_actually_drops_the_extremes(self):
        """Constructs 13 synthetic measured observations per window with a
        deliberate outlier at each extreme, and proves `aggregate()`'s
        trimmed-mean (trim=2) differs from the plain mean over the same
        data -- a mutation reverting `trim` to 0 (or the profile default)
        would make this test fail, since the plain-mean slope would then
        differ from the asserted trimmed-mean slope."""
        profile = self.profiles["apples_precise"]
        # 13 values with two low outliers and two high outliers; trimmed
        # mean (drop 2 lowest, 2 highest) of the remaining 9 differs from
        # the mean of all 13.
        baseline_secs = [0.05, 0.05, 0.30, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.31, 0.60, 0.60]
        window_secs = [0.80, 0.80, 2.0, 2.01, 2.01, 2.01, 2.01, 2.01, 2.01, 2.01, 2.01, 4.0, 4.0]
        self.assertEqual(len(baseline_secs), profile.measured_repeats)
        self.assertEqual(len(window_secs), profile.measured_repeats)

        def _obs(engine: str, window: int, elapsed_s: float, run_index: int) -> harness.Observation:
            return harness.Observation(
                schema_version=harness.SCHEMA_VERSION,
                git_sha="x", profile=profile.name, engine=engine, engine_version="fake",
                model="qwen3.5-0.8b", quantization="q8", prompt_hash="abc",
                requested_prompt_tokens=None, actual_prompt_tokens=None,
                requested_completion_tokens=window, actual_completion_tokens=window,
                warmup=False, run_index=run_index, order_index=run_index,
                elapsed_ns=round(elapsed_s * 1e9), engine_native_ns=None,
                hardware_id="h", timestamp="2026-01-01T00:00:00+00:00",
            )

        observations = tuple(
            _obs("lattice", 64, s, i + 1) for i, s in enumerate(baseline_secs)
        ) + tuple(
            _obs("lattice", 512, s, i + 1) for i, s in enumerate(window_secs)
        )
        run_result = harness.HarnessRunResult(profile=profile, observations=observations, missing_engines=())
        slopes = harness.aggregate(run_result)
        lattice_slope = next(s for s in slopes if s.engine == "lattice")

        trimmed_baseline_mean = sum(sorted(baseline_secs)[2:-2]) / 9
        trimmed_window_mean = sum(sorted(window_secs)[2:-2]) / 9
        expected_slope = (512 - 64) / (trimmed_window_mean - trimmed_baseline_mean)
        plain_baseline_mean = sum(baseline_secs) / 13
        plain_window_mean = sum(window_secs) / 13
        plain_slope = (512 - 64) / (plain_window_mean - plain_baseline_mean)

        self.assertNotAlmostEqual(expected_slope, plain_slope, places=2)
        self.assertAlmostEqual(lattice_slope.slope_tok_per_s, expected_slope, places=6)
        self.assertIsNotNone(lattice_slope.slope_ci95_legacy)

    def test_missing_engine_falls_back_to_allow_missing_engine(self):
        profile = self.profiles["apples_precise"]
        result = harness.run_profile(
            profile,
            {"mlx": _FakeAdapter()},
            allow_missing_engine=True,
            clock=_FakeClock(),
        )
        self.assertEqual(set(result.missing_engines), {"lattice", "ollama"})
        self.assertTrue(all(o.engine == "mlx" for o in result.observations))


class OllamaResponseParsingTest(unittest.TestCase):
    def test_valid_response(self):
        result = adapters.ollama_response_to_result(
            {"eval_count": 512, "eval_duration": 3_000_000_000, "total_duration": 4_000_000_000}
        )
        self.assertEqual(result.actual_completion_tokens, 512)
        self.assertEqual(result.native_ns, 4_000_000_000)

    def test_error_body_raises(self):
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result({"error": "model not found"})

    def test_missing_field_raises(self):
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result({"eval_count": 1, "eval_duration": 1})

    def test_float_field_rejected(self):
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result(
                {"eval_count": 512, "eval_duration": 3.5, "total_duration": 4_000_000_000}
            )


class LatticeResultParsingTest(unittest.TestCase):
    def test_parses_result_line(self):
        parsed = adapters.parse_lattice_result_line("RESULT n_req=64 completion=64 total_ms=210.5")
        self.assertEqual(parsed, (64, 64, 210.5))

    def test_non_result_line_returns_none(self):
        self.assertIsNone(adapters.parse_lattice_result_line("[bench] loading model"))

    def test_extract_single_result_rejects_wrong_window(self):
        with self.assertRaises(adapters.LatticeResultError):
            adapters.extract_single_result("RESULT n_req=64 completion=64 total_ms=1.0", n_tokens=512)

    def test_extract_single_result_rejects_zero_lines(self):
        with self.assertRaises(adapters.LatticeResultError):
            adapters.extract_single_result("no result here", n_tokens=64)

    def test_extract_single_result_rejects_multiple_lines(self):
        stdout = "RESULT n_req=64 completion=64 total_ms=1.0\nRESULT n_req=64 completion=64 total_ms=2.0\n"
        with self.assertRaises(adapters.LatticeResultError):
            adapters.extract_single_result(stdout, n_tokens=64)


if __name__ == "__main__":
    unittest.main()
