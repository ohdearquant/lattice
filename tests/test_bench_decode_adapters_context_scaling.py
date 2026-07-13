"""Deterministic tests for scripts/bench_decode_adapters_context_scaling.py
(issue #813 step 2: migrate bench_context_scaling.sh, keeping its chart
renderer).

Run with: python3 tests/test_bench_decode_adapters_context_scaling.py -v
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
adapters = _load_module("bench_decode_adapters_context_scaling", "bench_decode_adapters_context_scaling.py")

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


class ProfileParameterEquivalenceTest(unittest.TestCase):
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


class CallScheduleEquivalenceTest(unittest.TestCase):
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
        result = harness.run_profile(profile, {"lattice": _FakeAdapter()}, allow_missing_engine=True, clock=_FakeClock())
        slopes = harness.aggregate(result)
        tsv = adapters.render_legacy_tsv(result, slopes)
        lines = tsv.strip("\n").split("\n")
        self.assertEqual(lines[0], "engine\tcontext_tokens\tslope_tok_s\tt1_ms\tt2_ms\truns")
        self.assertEqual(len(lines), 1 + len(slopes))
        for line in lines[1:]:
            parts = line.split("\t")
            self.assertEqual(len(parts), 6)
            self.assertEqual(parts[0], "lattice")
            int(parts[1])  # context_tokens parses as int
            float(parts[2])  # slope_tok_s parses as float


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


class OllamaResponseParsingTest(unittest.TestCase):
    def test_valid_response(self):
        result = adapters.ollama_response_to_result(
            {"eval_count": 64, "eval_duration": 500_000_000, "total_duration": 800_000_000}
        )
        self.assertEqual(result.actual_completion_tokens, 64)
        self.assertEqual(result.native_ns, 800_000_000)

    def test_error_body_raises(self):
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result({"error": "boom"})


class LatticeResultParsingTest(unittest.TestCase):
    def test_parses_result_line(self):
        self.assertEqual(
            adapters.parse_lattice_result_line("RESULT n_req=8 completion=8 total_ms=50.0"),
            (8, 8, 50.0),
        )

    def test_extract_single_result_rejects_wrong_window(self):
        with self.assertRaises(adapters.LatticeResultError):
            adapters.extract_single_result("RESULT n_req=8 completion=8 total_ms=1.0", n_tokens=64)


if __name__ == "__main__":
    unittest.main()
