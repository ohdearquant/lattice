"""Deterministic tests for scripts/bench_decode_adapters_apples_to_apples.py
(issue #813 step 2: migrate bench_apples_to_apples.sh onto the shared
decode-bench harness).

No engine binary or network access is required to run this file: real
adapters are exercised only through their pure parsing helpers
(`parse_lattice_result_line`, `ollama_response_to_result`); the profile ->
call-schedule equivalence is checked with the harness's own deterministic
`_FakeAdapter`, matching `tests/test_bench_decode_harness.py`'s convention
and the issue #813 gate: "Deterministic fake-adapter tests and raw-schema
validation pass without any engine binary present."

Run with: python3 tests/test_bench_decode_adapters_apples_to_apples.py -v
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
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
adapters = _load_module(
    "bench_decode_adapters_apples_to_apples", "bench_decode_adapters_apples_to_apples.py"
)

DEFAULT_PROFILES_FILE = _SCRIPTS / "bench_decode_profiles.toml"


# --------------------------------------------------------------------------
# Fakes (mirrors tests/test_bench_decode_harness.py's _FakeAdapter/_FakeClock)
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Profile parameter equivalence (the issue #813 gate: byte-for-byte
# equivalent window/repeats/warmup parameters vs the legacy script)
# --------------------------------------------------------------------------


class ProfileParameterEquivalenceTest(unittest.TestCase):
    """bench_apples_to_apples.sh: N1=32, N2=256, RUNS=5, no lattice/ollama
    warmup, mlx warms up once at 8 tokens -- see the script's own
    N1/N2/RUNS constants and its single pre-loop `generate(..., max_tokens=8,
    ...)` warmup call before the N1/N2 loop."""

    @classmethod
    def setUpClass(cls):
        _, cls.profiles = harness.load_profiles_file(DEFAULT_PROFILES_FILE)

    def test_both_profiles_defined(self):
        self.assertIn("apples_to_apples_q8", self.profiles)
        self.assertIn("apples_to_apples_q4", self.profiles)

    def test_q8_windows_and_repeats_match_legacy_n1_n2_runs(self):
        p = self.profiles["apples_to_apples_q8"]
        self.assertEqual(p.windows, (32, 256))
        self.assertEqual(p.measured_repeats, 5)
        self.assertEqual(p.aggregation, "median")

    def test_q4_windows_and_repeats_match_legacy_n1_n2_runs(self):
        p = self.profiles["apples_to_apples_q4"]
        self.assertEqual(p.windows, (32, 256))
        self.assertEqual(p.measured_repeats, 5)

    def test_q8_prompt_matches_legacy_prompt_variable(self):
        p = self.profiles["apples_to_apples_q8"]
        self.assertEqual(
            p.prompt,
            "The quick brown fox jumps over the lazy dog. Once upon a time in a land "
            "far away, there lived a",
        )

    def test_q8_engines_and_order_match_legacy_section_order(self):
        p = self.profiles["apples_to_apples_q8"]
        self.assertEqual(p.engines, ("lattice", "ollama", "mlx"))

    def test_q4_engines_omit_ollama_no_q4_variant(self):
        p = self.profiles["apples_to_apples_q4"]
        self.assertEqual(p.engines, ("lattice", "mlx"))

    def test_q8_only_mlx_has_a_warmup(self):
        p = self.profiles["apples_to_apples_q8"]
        groups = {g.name: g for g in p.engine_groups}
        self.assertEqual(groups["lattice"].warmup_repeats, 0)
        self.assertEqual(groups["ollama"].warmup_repeats, 0)
        self.assertEqual(groups["mlx"].warmup_repeats, 1)
        self.assertEqual(groups["mlx"].warmup_tokens, 8)

    def test_q4_only_mlx_has_a_warmup(self):
        p = self.profiles["apples_to_apples_q4"]
        groups = {g.name: g for g in p.engine_groups}
        self.assertEqual(groups["lattice"].warmup_repeats, 0)
        self.assertEqual(groups["mlx"].warmup_repeats, 1)
        self.assertEqual(groups["mlx"].warmup_tokens, 8)

    def test_q8_quantization_tiers(self):
        p = self.profiles["apples_to_apples_q8"]
        groups = {g.name: g for g in p.engine_groups}
        self.assertEqual(groups["lattice"].quantization, "q8")
        self.assertEqual(groups["ollama"].quantization, "q8")
        self.assertEqual(groups["mlx"].quantization, "q8")

    def test_q4_quantization_tiers_lattice_quarot_mlx_bits4(self):
        p = self.profiles["apples_to_apples_q4"]
        groups = {g.name: g for g in p.engine_groups}
        self.assertEqual(groups["lattice"].model, "qwen3.5-0.8b-q4-quarot")
        self.assertEqual(groups["lattice"].quantization, "q4")
        self.assertEqual(groups["mlx"].quantization, "q4")
        # MLX loads from the Q8 safetensors dir and quantizes on the fly to
        # 4 bits -- same source model as the Q8 tier, matching the legacy
        # script's bench_mlx() which always loads $Q8_DIR regardless of tier.
        self.assertEqual(groups["mlx"].model, "qwen3.5-0.8b")


# --------------------------------------------------------------------------
# Call-schedule equivalence via fake adapters (no engine binary needed)
# --------------------------------------------------------------------------


class CallScheduleEquivalenceTest(unittest.TestCase):
    """Runs the REAL shipped profiles through harness.run_profile with fake
    adapters, proving the produced call schedule matches
    bench_apples_to_apples.sh's loop structure exactly: lattice/ollama get
    N1 x5 then N2 x5 with no warmup; mlx gets one 8-token warmup, then N1 x5
    then N2 x5."""

    @classmethod
    def setUpClass(cls):
        _, cls.profiles = harness.load_profiles_file(DEFAULT_PROFILES_FILE)

    def test_q8_tier_call_sequence(self):
        profile = self.profiles["apples_to_apples_q8"]
        lattice, ollama, mlx = _FakeAdapter(), _FakeAdapter(), _FakeAdapter()
        harness.run_profile(
            profile,
            {"lattice": lattice, "ollama": ollama, "mlx": mlx},
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )
        self.assertEqual(
            lattice.calls,
            [(32, False, "qwen3.5-0.8b", "q8")] * 5 + [(256, False, "qwen3.5-0.8b", "q8")] * 5,
        )
        self.assertEqual(
            ollama.calls,
            [(32, False, "qwen3.5:0.8b", "q8")] * 5 + [(256, False, "qwen3.5:0.8b", "q8")] * 5,
        )
        self.assertEqual(
            mlx.calls,
            [(8, True, "qwen3.5-0.8b", "q8")]
            + [(32, False, "qwen3.5-0.8b", "q8")] * 5
            + [(256, False, "qwen3.5-0.8b", "q8")] * 5,
        )

    def test_q4_tier_call_sequence_no_ollama(self):
        profile = self.profiles["apples_to_apples_q4"]
        lattice, mlx = _FakeAdapter(), _FakeAdapter()
        result = harness.run_profile(
            profile,
            {"lattice": lattice, "mlx": mlx},
            clock=_FakeClock(),
            git_sha_value="x",
            hardware_id_value="h",
        )
        self.assertEqual(
            lattice.calls,
            [(32, False, "qwen3.5-0.8b-q4-quarot", "q4")] * 5
            + [(256, False, "qwen3.5-0.8b-q4-quarot", "q4")] * 5,
        )
        self.assertEqual(
            mlx.calls,
            [(8, True, "qwen3.5-0.8b", "q4")]
            + [(32, False, "qwen3.5-0.8b", "q4")] * 5
            + [(256, False, "qwen3.5-0.8b", "q4")] * 5,
        )
        self.assertNotIn("ollama", [o.engine for o in result.observations])

    def test_q8_produces_schema_valid_observations(self):
        profile = self.profiles["apples_to_apples_q8"]
        result = harness.run_profile(
            profile,
            {"lattice": _FakeAdapter(), "ollama": _FakeAdapter(), "mlx": _FakeAdapter()},
            clock=_FakeClock(),
        )
        for obs in result.observations:
            harness.validate_observation(obs.to_dict())

    def test_missing_engine_binary_falls_back_to_allow_missing_engine(self):
        # Reproduces the wrapper script's --allow-missing-engine usage: an
        # engine with no adapter registered (e.g. lattice not built) is
        # skipped, not fatal, matching the legacy script's per-engine
        # graceful skip.
        profile = self.profiles["apples_to_apples_q8"]
        result = harness.run_profile(
            profile,
            {"mlx": _FakeAdapter()},
            allow_missing_engine=True,
            clock=_FakeClock(),
        )
        self.assertEqual(set(result.missing_engines), {"lattice", "ollama"})
        self.assertTrue(all(o.engine == "mlx" for o in result.observations))


# --------------------------------------------------------------------------
# Pure parsing helpers (real-adapter plumbing, no process/network needed)
# --------------------------------------------------------------------------


class LatticeResultParsingTest(unittest.TestCase):
    def test_parses_result_line(self):
        line = "RESULT n_req=32 completion=32 total_ms=812.345"
        parsed = adapters.parse_lattice_result_line(line)
        self.assertEqual(parsed, (32, 32, 812.345))

    def test_parses_result_line_with_differing_actual_completion(self):
        # completion can differ from n_req (e.g. early stop) -- both are
        # recorded independently, never assumed equal.
        line = "RESULT n_req=256 completion=201 total_ms=1900.0"
        parsed = adapters.parse_lattice_result_line(line)
        self.assertEqual(parsed, (256, 201, 1900.0))

    def test_ignores_non_result_lines(self):
        self.assertIsNone(adapters.parse_lattice_result_line("[bench] loading /some/dir (Q4)"))
        self.assertIsNone(adapters.parse_lattice_result_line("[bench] prompt_tokens=42"))
        self.assertIsNone(adapters.parse_lattice_result_line(""))

    def test_uses_last_result_line_on_multiple(self):
        stdout = (
            "RESULT n_req=32 completion=32 total_ms=100.0\n"
            "RESULT n_req=32 completion=32 total_ms=200.0\n"
        )
        parsed = None
        for line in stdout.splitlines():
            result = adapters.parse_lattice_result_line(line)
            if result is not None:
                parsed = result
        self.assertEqual(parsed, (32, 32, 200.0))


class OllamaResponseParsingTest(unittest.TestCase):
    def test_translates_eval_count_and_duration_to_native_ns(self):
        data = {"eval_count": 256, "eval_duration": 2_000_000_000, "total_duration": 2_500_000_000}
        result = adapters.ollama_response_to_result(data, requested_tokens=256)
        self.assertEqual(result.actual_completion_tokens, 256)
        self.assertEqual(result.native_ns, 2_000_000_000)

    def test_falls_back_to_requested_tokens_when_eval_count_absent(self):
        result = adapters.ollama_response_to_result({}, requested_tokens=32)
        self.assertEqual(result.actual_completion_tokens, 32)
        self.assertIsNone(result.native_ns)

    def test_zero_eval_duration_yields_no_native_ns(self):
        data = {"eval_count": 10, "eval_duration": 0}
        result = adapters.ollama_response_to_result(data, requested_tokens=10)
        self.assertIsNone(result.native_ns)


class AvailabilityCheckTest(unittest.TestCase):
    def test_lattice_available_false_when_binary_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "bench_decode_ab"
            self.assertFalse(adapters.lattice_available(missing))

    def test_lattice_available_true_when_binary_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "bench_decode_ab"
            bin_path.write_text("#!/bin/sh\n")
            bin_path.chmod(0o755)
            self.assertTrue(adapters.lattice_available(bin_path))

    def test_ollama_available_false_when_binary_not_on_path(self):
        with mock.patch.object(adapters.shutil, "which", return_value=None):
            self.assertFalse(adapters.ollama_available(model_tag="qwen3.5:0.8b"))

    def test_mlx_available_reflects_import_success(self):
        # mlx_lm is a declared project dependency (pyproject.toml); this
        # documents the expectation rather than mocking import machinery.
        self.assertTrue(adapters.mlx_available())


if __name__ == "__main__":
    unittest.main()
