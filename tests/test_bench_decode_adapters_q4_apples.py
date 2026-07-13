"""Deterministic tests for scripts/bench_decode_adapters_q4_apples.py
(issue #813 step 2: migrate bench_q4_apples.sh onto the shared
decode-bench harness).

Run with: python3 tests/test_bench_decode_adapters_q4_apples.py -v
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
adapters = _load_module("bench_decode_adapters_q4_apples", "bench_decode_adapters_q4_apples.py")

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
    """bench_q4_apples.sh: N1=32, N2=256, RUNS=5, no lattice/ollama warmup,
    mlx warms once at 8 tokens (same call shape as apples_to_apples_q4, but
    this is that script's own separate reimplementation, migrated on its
    own profile per the issue's one-script-per-PR plan)."""

    @classmethod
    def setUpClass(cls):
        _, cls.profiles = harness.load_profiles_file(DEFAULT_PROFILES_FILE)

    def test_profile_defined(self):
        self.assertIn("q4_apples", self.profiles)

    def test_windows_and_repeats_match_legacy_n1_n2_runs(self):
        p = self.profiles["q4_apples"]
        self.assertEqual(p.windows, (32, 256))
        self.assertEqual(p.measured_repeats, 5)
        self.assertEqual(p.aggregation, "median")

    def test_engines_and_order(self):
        p = self.profiles["q4_apples"]
        self.assertEqual(p.engines, ("lattice", "ollama", "mlx"))

    def test_only_mlx_has_a_warmup(self):
        p = self.profiles["q4_apples"]
        groups = {g.name: g for g in p.engine_groups}
        self.assertEqual(groups["lattice"].warmup_repeats, 0)
        self.assertEqual(groups["ollama"].warmup_repeats, 0)
        self.assertEqual(groups["mlx"].warmup_repeats, 1)
        self.assertEqual(groups["mlx"].warmup_tokens, 8)

    def test_quantization_tiers_lattice_q4_ollama_q8_reference_mlx_q4(self):
        p = self.profiles["q4_apples"]
        groups = {g.name: g for g in p.engine_groups}
        self.assertEqual(groups["lattice"].model, "qwen3.5-0.8b-q4")
        self.assertEqual(groups["lattice"].quantization, "q4")
        self.assertEqual(groups["ollama"].quantization, "q8")
        self.assertEqual(groups["mlx"].quantization, "q4")
        self.assertEqual(groups["mlx"].model, "qwen3.5-0.8b")


class CallScheduleEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.profiles = harness.load_profiles_file(DEFAULT_PROFILES_FILE)

    def test_call_sequence(self):
        profile = self.profiles["q4_apples"]
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
            [(32, False, "qwen3.5-0.8b-q4", "q4")] * 5 + [(256, False, "qwen3.5-0.8b-q4", "q4")] * 5,
        )
        self.assertEqual(
            ollama.calls,
            [(32, False, "qwen3.5:0.8b", "q8")] * 5 + [(256, False, "qwen3.5:0.8b", "q8")] * 5,
        )
        self.assertEqual(
            mlx.calls,
            [(8, True, "qwen3.5-0.8b", "q4")]
            + [(32, False, "qwen3.5-0.8b", "q4")] * 5
            + [(256, False, "qwen3.5-0.8b", "q4")] * 5,
        )

    def test_produces_schema_valid_observations(self):
        profile = self.profiles["q4_apples"]
        result = harness.run_profile(
            profile,
            {"lattice": _FakeAdapter(), "ollama": _FakeAdapter(), "mlx": _FakeAdapter()},
            clock=_FakeClock(),
        )
        for obs in result.observations:
            harness.validate_observation(obs.to_dict())

    def test_missing_engine_falls_back_to_allow_missing_engine(self):
        profile = self.profiles["q4_apples"]
        result = harness.run_profile(
            profile,
            {"mlx": _FakeAdapter()},
            allow_missing_engine=True,
            clock=_FakeClock(),
        )
        self.assertEqual(set(result.missing_engines), {"lattice", "ollama"})


class LatticeAdapterEnvVarCorrectionTest(unittest.TestCase):
    """Proves the DISCLOSED correction (see bench_decode_profiles.toml's
    [profiles.q4_apples] comment): the adapter must set LATTICE_MODEL_DIR
    (the name bench_decode_ab.rs actually reads), not the legacy script's
    unread BENCH_Q4_DIR/BENCH_TOKENIZER_DIR names."""

    def test_lattice_adapter_sets_the_real_env_var_names(self):
        import subprocess
        from unittest import mock

        captured = {}

        def fake_run(cmd, env, capture_output, text, timeout, check):
            captured.update(env)
            return subprocess.CompletedProcess(
                cmd, 0, stdout="RESULT n_req=32 completion=32 total_ms=100.0\n", stderr=""
            )

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "weights.q4").write_bytes(b"stub")
            adapter = adapters.LatticeAdapter(
                bin_path=Path("/bin/true"), model_dir=model_dir, tokenizer_dir=model_dir
            )
            with mock.patch.object(subprocess, "run", side_effect=fake_run):
                adapter.run(
                    prompt="hi", n_tokens=32, warmup=False, model="qwen3.5-0.8b-q4", quantization="q4"
                )

        self.assertIn("LATTICE_MODEL_DIR", captured)
        self.assertIn("LATTICE_TOKENIZER_DIR", captured)
        self.assertNotIn("BENCH_Q4_DIR", captured)
        self.assertNotIn("BENCH_TOKENIZER_DIR", captured)


class LatticeAdapterModelIdentityTest(unittest.TestCase):
    """The adapter must refuse to run when the resolved dir is not what
    `detect_format` would load as Q4 (safetensors markers take precedence in
    the binary, so their presence means a mislabeled full-precision run --
    the historical bench_q4_apples.sh failure mode)."""

    def _adapter_for(self, model_dir):
        return adapters.LatticeAdapter(
            bin_path=Path("/bin/true"), model_dir=model_dir, tokenizer_dir=model_dir
        )

    def test_refuses_safetensors_shaped_dir_even_with_q4_files(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "weights.q4").write_bytes(b"stub")
            (model_dir / "model.safetensors").write_bytes(b"stub")
            with self.assertRaises(adapters.LatticeUnavailableError) as ctx:
                self._adapter_for(model_dir).run(
                    prompt="hi", n_tokens=32, warmup=False, model="m", quantization="q4"
                )
            self.assertIn("safetensors", str(ctx.exception))

    def test_refuses_dir_without_q4_files(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_bytes(b"{}")
            with self.assertRaises(adapters.LatticeUnavailableError) as ctx:
                self._adapter_for(model_dir).run(
                    prompt="hi", n_tokens=32, warmup=False, model="m", quantization="q4"
                )
            self.assertIn("no .q4 files", str(ctx.exception))


class LatticeResultParsingTest(unittest.TestCase):
    def test_parses_result_line(self):
        self.assertEqual(
            adapters.parse_lattice_result_line("RESULT n_req=256 completion=256 total_ms=999.9"),
            (256, 256, 999.9),
        )

    def test_extract_single_result_rejects_wrong_window(self):
        with self.assertRaises(adapters.LatticeResultError):
            adapters.extract_single_result("RESULT n_req=32 completion=32 total_ms=1.0", n_tokens=256)


class OllamaResponseParsingTest(unittest.TestCase):
    def test_valid_response(self):
        result = adapters.ollama_response_to_result(
            {"eval_count": 256, "eval_duration": 1_000_000_000, "total_duration": 2_000_000_000}
        )
        self.assertEqual(result.actual_completion_tokens, 256)
        self.assertEqual(result.native_ns, 2_000_000_000)

    def test_error_body_raises(self):
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result({"error": "boom"})


if __name__ == "__main__":
    unittest.main()
