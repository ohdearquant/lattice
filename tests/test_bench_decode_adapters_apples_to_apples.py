"""Deterministic tests for scripts/bench_decode_adapters_apples_to_apples.py
(issue #813 step 2: migrate bench_apples_to_apples.sh onto the shared
decode-bench harness).

No engine binary or network access is required to run this file: real
adapters are exercised only through their pure parsing helpers
(`parse_lattice_result_line`, `ollama_response_to_result`); the profile ->
call-schedule contract is checked against the harness's own deterministic
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
# Profile parameter transcription (the issue #813 gate: byte-for-byte
# equivalent window/repeats/warmup parameters vs the legacy script)
# --------------------------------------------------------------------------


class ProfileParameterTranscriptionTest(unittest.TestCase):
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
# Call-schedule contract via fake adapters (no engine binary needed)
# --------------------------------------------------------------------------


class CallScheduleContractTest(unittest.TestCase):
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

    def test_rejects_prefixed_noise(self):
        # A RESULT-bearing SUBSTRING is no longer good enough: the legacy
        # awk filter (`/^RESULT/`, matches anywhere) would have accepted
        # this; the stricter BENCH_RUNS=1 contract requires a full-line
        # match and must not scavenge a match out of a noisy line.
        line = "[stale-cache] RESULT n_req=32 completion=32 total_ms=812.345"
        self.assertIsNone(adapters.parse_lattice_result_line(line))

    def test_rejects_suffixed_noise(self):
        line = "RESULT n_req=32 completion=32 total_ms=812.345 (cached)"
        self.assertIsNone(adapters.parse_lattice_result_line(line))


class ExtractSingleResultTest(unittest.TestCase):
    """`extract_single_result` is the BENCH_RUNS=1 contract: exactly one
    full-line RESULT record, whose n_req matches what was actually
    requested, with finite/non-negative fields -- anything else raises
    `LatticeResultError` rather than silently accepting a stale, mismatched,
    or duplicated measurement.
    """

    def test_happy_path(self):
        stdout = "[bench] loading model\nRESULT n_req=32 completion=32 total_ms=812.345\n"
        result = adapters.extract_single_result(stdout, n_tokens=32)
        self.assertEqual(result, (32, 32, 812.345))

    def test_rejects_no_result_lines(self):
        with self.assertRaises(adapters.LatticeResultError):
            adapters.extract_single_result("[bench] loading model\n[bench] done\n", n_tokens=32)

    def test_rejects_duplicate_result_lines(self):
        # BENCH_RUNS=1 means exactly one RESULT line is expected; two
        # (e.g. a stale binary that ignored BENCH_RUNS, or leaked output
        # from a prior invocation) must fail loud, never silently resolve
        # to "the last one" the way the pre-fix scan did.
        stdout = (
            "RESULT n_req=32 completion=32 total_ms=100.0\n"
            "RESULT n_req=32 completion=32 total_ms=200.0\n"
        )
        with self.assertRaises(adapters.LatticeResultError):
            adapters.extract_single_result(stdout, n_tokens=32)

    def test_rejects_wrong_n_req(self):
        # A stale/wrong binary emitting a RESULT line for a DIFFERENT
        # window than the one just requested must not silently become an
        # observation labeled as the requested window.
        stdout = "RESULT n_req=256 completion=256 total_ms=1900.0\n"
        with self.assertRaises(adapters.LatticeResultError):
            adapters.extract_single_result(stdout, n_tokens=32)

    def test_rejects_prefixed_noise_end_to_end(self):
        stdout = "[stale-cache] RESULT n_req=32 completion=32 total_ms=812.345\n"
        with self.assertRaises(adapters.LatticeResultError):
            adapters.extract_single_result(stdout, n_tokens=32)


class OllamaResponseParsingTest(unittest.TestCase):
    """`ollama_response_to_result` mirrors the legacy script's PRIMARY slope
    field exactly (`total_duration`, not the decode-only `eval_duration`),
    and requires+type-checks the response shape rather than substituting
    defaults for missing/invalid fields -- a structurally-valid-JSON error
    body must crash the call loud, never become a fake measurement.
    """

    def test_translates_total_duration_to_native_ns(self):
        data = {"eval_count": 256, "eval_duration": 2_000_000_000, "total_duration": 2_500_000_000}
        result = adapters.ollama_response_to_result(data)
        self.assertEqual(result.actual_completion_tokens, 256)
        # native_ns is total_duration (the legacy script's own primary slope
        # field), NOT eval_duration (that was the legacy script's separate
        # decode-only reference column).
        self.assertEqual(result.native_ns, 2_500_000_000)

    def test_error_body_raises(self):
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result({"error": "model not found"})

    def test_empty_dict_raises(self):
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result({})

    def test_missing_eval_count_raises(self):
        data = {"eval_duration": 1, "total_duration": 1}
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result(data)

    def test_missing_eval_duration_raises(self):
        data = {"eval_count": 32, "total_duration": 1}
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result(data)

    def test_missing_total_duration_raises(self):
        data = {"eval_count": 32, "eval_duration": 1}
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result(data)

    def test_non_numeric_field_type_raises(self):
        data = {"eval_count": "32", "eval_duration": 1, "total_duration": 1}
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result(data)

    def test_bool_field_type_raises(self):
        # bool is an int subclass in Python -- must not sneak past the
        # int-only check.
        data = {"eval_count": True, "eval_duration": 1, "total_duration": 1}
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result(data)

    def test_fractional_field_values_raise(self):
        # The API documents eval_count/eval_duration/total_duration as
        # integers; a float (even a "clean" one like 1.0) must be rejected
        # outright rather than silently truncated via int(31.9) == 31 --
        # that truncation is exactly the fake-observation class this
        # function exists to close, reached through a technically-numeric
        # value instead of a missing one.
        data = {"eval_count": 31.9, "eval_duration": 1.0, "total_duration": 2.9}
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result(data)

    def test_fractional_eval_count_alone_raises(self):
        data = {"eval_count": 31.9, "eval_duration": 1, "total_duration": 1}
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result(data)

    def test_fractional_eval_duration_alone_raises(self):
        # eval_count is validated first, so the all-fractional case above
        # never reaches eval_duration; this isolates it (a "clean" 1.0
        # would survive int() conversion without truncation, so it is the
        # sharpest float to reject on type alone).
        data = {"eval_count": 32, "eval_duration": 1.0, "total_duration": 1}
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result(data)

    def test_fractional_total_duration_alone_raises(self):
        data = {"eval_count": 32, "eval_duration": 1, "total_duration": 2.9}
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result(data)

    def test_negative_eval_count_raises(self):
        data = {"eval_count": -1, "eval_duration": 1, "total_duration": 1}
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result(data)

    def test_zero_total_duration_raises(self):
        data = {"eval_count": 10, "eval_duration": 1, "total_duration": 0}
        with self.assertRaises(adapters.OllamaResponseError):
            adapters.ollama_response_to_result(data)

    def test_eval_count_zero_is_trusted_not_substituted(self):
        # A real, validly-typed zero must be reported as-is -- never
        # silently replaced with the requested token count (that would be
        # exactly the kind of fabrication this hardening pass closes).
        data = {"eval_count": 0, "eval_duration": 1, "total_duration": 1}
        result = adapters.ollama_response_to_result(data)
        self.assertEqual(result.actual_completion_tokens, 0)


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

    @unittest.skipUnless(
        importlib.util.find_spec("mlx_lm") is not None,
        "asserts the import-success case, so mlx_lm must be installed "
        "(macOS-only dependency; skipped on Linux CI)",
    )
    def test_mlx_available_reflects_import_success(self):
        # mlx_lm is a declared project dependency (pyproject.toml); this
        # documents the expectation rather than mocking import machinery.
        self.assertTrue(adapters.mlx_available())


# --------------------------------------------------------------------------
# Model-aware --allow-missing-engine: binary present + model dir absent must
# reproduce the legacy per-engine graceful skip, not raise mid-run.
# --------------------------------------------------------------------------


class LatticeModelAwareAvailabilityTest(unittest.TestCase):
    """Real-adapter coverage: a present, executable binary with a missing
    model directory must (a) make the real `LatticeAdapter.run()` raise
    cleanly (defense in depth, unchanged), and (b) make the REGISTRATION
    decision skip lattice entirely for the profile that needs that model,
    so `--allow-missing-engine` actually engages instead of a
    `LatticeUnavailableError` aborting the tier mid-run.
    """

    @classmethod
    def setUpClass(cls):
        _, cls.profiles = harness.load_profiles_file(DEFAULT_PROFILES_FILE)

    def test_lattice_adapter_run_raises_when_model_dir_missing(self):
        # The literal "binary present, model dir absent" scenario, exercised
        # against the REAL LatticeAdapter (not a fake).
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "bench_decode_ab"
            bin_path.write_text("#!/bin/sh\necho should-not-run\n")
            bin_path.chmod(0o755)
            adapter = adapters.LatticeAdapter(bin_path=bin_path, tokenizer_dir=Path(tmp))
            missing_dir = Path(tmp) / "does-not-exist"
            with mock.patch.object(adapters, "_LATTICE_MODEL_DIRS", {"qwen3.5-0.8b": missing_dir}):
                with self.assertRaises(adapters.LatticeUnavailableError):
                    adapter.run(
                        prompt="p", n_tokens=32, warmup=False, model="qwen3.5-0.8b", quantization="q8"
                    )

    def test_registration_status_skips_when_profile_model_dir_missing(self):
        profile = self.profiles["apples_to_apples_q8"]
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "bench_decode_ab"
            bin_path.write_text("#!/bin/sh\n")
            bin_path.chmod(0o755)
            missing_dir = Path(tmp) / "does-not-exist"
            with mock.patch.object(adapters, "_LATTICE_MODEL_DIRS", {"qwen3.5-0.8b": missing_dir}):
                should_register, missing = adapters.lattice_registration_status(profile, bin_path=bin_path)
        self.assertFalse(should_register)
        self.assertEqual(missing, (missing_dir,))

    def test_registration_status_registers_when_binary_and_model_dir_present(self):
        profile = self.profiles["apples_to_apples_q8"]
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "bench_decode_ab"
            bin_path.write_text("#!/bin/sh\n")
            bin_path.chmod(0o755)
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            with mock.patch.object(adapters, "_LATTICE_MODEL_DIRS", {"qwen3.5-0.8b": model_dir}):
                should_register, missing = adapters.lattice_registration_status(profile, bin_path=bin_path)
        self.assertTrue(should_register)
        self.assertEqual(missing, ())

    def test_registration_status_false_when_binary_missing_even_if_model_present(self):
        profile = self.profiles["apples_to_apples_q8"]
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "bench_decode_ab"  # never created
            model_dir = Path(tmp) / "model"
            model_dir.mkdir()
            with mock.patch.object(adapters, "_LATTICE_MODEL_DIRS", {"qwen3.5-0.8b": model_dir}):
                should_register, missing = adapters.lattice_registration_status(profile, bin_path=bin_path)
        self.assertFalse(should_register)
        self.assertEqual(missing, ())  # binary check fails before model dirs are even inspected

    def test_registration_status_falls_back_to_binary_only_when_profile_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "bench_decode_ab"
            bin_path.write_text("#!/bin/sh\n")
            bin_path.chmod(0o755)
            should_register, missing = adapters.lattice_registration_status(None, bin_path=bin_path)
        self.assertTrue(should_register)
        self.assertEqual(missing, ())

    def test_end_to_end_missing_model_dir_yields_graceful_skip_via_harness(self):
        # Closes the loop: when registration correctly declines to register
        # lattice, harness.run_profile's own --allow-missing-engine path
        # (not a mid-run exception) is what handles it, with other engines
        # still producing observations.
        profile = self.profiles["apples_to_apples_q8"]
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = Path(tmp) / "bench_decode_ab"
            bin_path.write_text("#!/bin/sh\n")
            bin_path.chmod(0o755)
            missing_dir = Path(tmp) / "does-not-exist"
            with mock.patch.object(adapters, "_LATTICE_MODEL_DIRS", {"qwen3.5-0.8b": missing_dir}):
                should_register, _missing = adapters.lattice_registration_status(profile, bin_path=bin_path)
        self.assertFalse(should_register)
        result = harness.run_profile(
            profile,
            {"ollama": _FakeAdapter(), "mlx": _FakeAdapter()},  # lattice deliberately NOT registered
            allow_missing_engine=True,
            clock=_FakeClock(),
        )
        self.assertIn("lattice", result.missing_engines)
        self.assertTrue(any(o.engine in ("ollama", "mlx") for o in result.observations))


class RegisterAvailableAdaptersEndToEndTest(unittest.TestCase):
    """End-to-end coverage for the ACTUAL registration entry points --
    `_peek_requested_profile` and `register_available_adapters` -- not just
    `lattice_registration_status` called directly. A bug in the argv-peeking
    path itself (the `--profile name` / `--profile=name` / `--profiles-file`
    / omitted-`--profile` forms) would not fail any of
    `LatticeModelAwareAvailabilityTest`'s tests, since those call
    `lattice_registration_status` directly with an already-resolved profile
    object -- this class drives the real argv shape
    `bench_decode_adapters_apples_to_apples.py run --profile ... [...]`
    produces through the module-global entry point instead.
    """

    def _write_solo_lattice_profile(self, tmp: Path) -> Path:
        # A minimal single-engine profile: only "lattice" is declared, so
        # the registration decision under test is isolated from ollama/mlx.
        profiles_file = tmp / "profiles.toml"
        profiles_file.write_text(
            "schema_version = 1\n"
            "\n"
            "[profiles.solo_lattice]\n"
            "windows = [8, 16]\n"
            "measured_repeats = 1\n"
            'prompt = "hello"\n'
            'aggregation = "median"\n'
            "\n"
            "[[profiles.solo_lattice.engines]]\n"
            'name = "lattice"\n'
            "warmup_repeats = 0\n"
            'model = "test-model"\n'
            'quantization = "q8"\n'
        )
        return profiles_file

    def _make_binary(self, tmp: Path) -> Path:
        bin_path = tmp / "bench_decode_ab"
        bin_path.write_text("#!/bin/sh\n")
        bin_path.chmod(0o755)
        return bin_path

    def setUp(self):
        # register_available_adapters() mutates the module-global registry
        # -- isolate this test class from it and from every other test.
        self._registry_backup = dict(harness.ADAPTER_REGISTRY)
        harness.ADAPTER_REGISTRY.clear()
        self.addCleanup(self._restore_registry)
        # Real ollama/mlx probing does real subprocess/import work that is
        # unrelated to what this class covers (the lattice argv-peek path)
        # -- stub both out so this stays fast and hermetic, matching this
        # file's own no-binary/no-network design (see module docstring).
        patcher_ollama = mock.patch.object(adapters, "ollama_available", return_value=False)
        patcher_mlx = mock.patch.object(adapters, "mlx_available", return_value=False)
        patcher_ollama.start()
        patcher_mlx.start()
        self.addCleanup(patcher_ollama.stop)
        self.addCleanup(patcher_mlx.stop)

    def _restore_registry(self):
        harness.ADAPTER_REGISTRY.clear()
        harness.ADAPTER_REGISTRY.update(self._registry_backup)

    def test_skips_lattice_when_selected_profile_model_dir_missing(self):
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            profiles_file = self._write_solo_lattice_profile(tmp)
            bin_path = self._make_binary(tmp)
            missing_dir = tmp / "does-not-exist"
            with (
                mock.patch.object(adapters, "LAT_BIN", bin_path),
                mock.patch.object(adapters, "_LATTICE_MODEL_DIRS", {"test-model": missing_dir}),
            ):
                adapters.register_available_adapters(
                    ["run", "--profile", "solo_lattice", "--profiles-file", str(profiles_file)]
                )
        self.assertNotIn("lattice", harness.ADAPTER_REGISTRY)

    def test_registers_lattice_when_selected_profile_model_dir_present(self):
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            profiles_file = self._write_solo_lattice_profile(tmp)
            bin_path = self._make_binary(tmp)
            model_dir = tmp / "model"
            model_dir.mkdir()
            with (
                mock.patch.object(adapters, "LAT_BIN", bin_path),
                mock.patch.object(adapters, "_LATTICE_MODEL_DIRS", {"test-model": model_dir}),
            ):
                adapters.register_available_adapters(
                    ["run", "--profile", "solo_lattice", "--profiles-file", str(profiles_file)]
                )
        self.assertIn("lattice", harness.ADAPTER_REGISTRY)

    def test_peek_handles_profile_equals_name_form(self):
        # `--profile=solo_lattice` (single-token `=` form), not the
        # two-token `--profile solo_lattice` form the other tests use.
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            profiles_file = self._write_solo_lattice_profile(tmp)
            bin_path = self._make_binary(tmp)
            missing_dir = tmp / "does-not-exist"
            with (
                mock.patch.object(adapters, "LAT_BIN", bin_path),
                mock.patch.object(adapters, "_LATTICE_MODEL_DIRS", {"test-model": missing_dir}),
            ):
                adapters.register_available_adapters(
                    ["run", "--profile=solo_lattice", "--profiles-file", str(profiles_file)]
                )
        self.assertNotIn("lattice", harness.ADAPTER_REGISTRY)

    def test_peek_handles_profiles_file_flag_before_profile_flag(self):
        # Argument order independence: --profiles-file appears BEFORE
        # --profile on the command line (argparse doesn't care, but the
        # hand-rolled peek must not assume a fixed order either).
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            profiles_file = self._write_solo_lattice_profile(tmp)
            bin_path = self._make_binary(tmp)
            model_dir = tmp / "model"
            model_dir.mkdir()
            with (
                mock.patch.object(adapters, "LAT_BIN", bin_path),
                mock.patch.object(adapters, "_LATTICE_MODEL_DIRS", {"test-model": model_dir}),
            ):
                adapters.register_available_adapters(
                    ["run", "--profiles-file", str(profiles_file), "--profile", "solo_lattice"]
                )
        self.assertIn("lattice", harness.ADAPTER_REGISTRY)

    def test_peek_falls_back_to_binary_only_when_profile_omitted(self):
        # No --profile at all: _peek_requested_profile can't resolve a
        # profile (harness.main() itself would later fail closed on the
        # missing required argument), so registration falls back to
        # binary-only availability -- the pre-fix behavior -- rather than
        # silently never registering lattice for an unrecognized invocation.
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            profiles_file = self._write_solo_lattice_profile(tmp)
            bin_path = self._make_binary(tmp)
            missing_dir = tmp / "does-not-exist"
            with (
                mock.patch.object(adapters, "LAT_BIN", bin_path),
                mock.patch.object(adapters, "_LATTICE_MODEL_DIRS", {"test-model": missing_dir}),
            ):
                adapters.register_available_adapters(["run", "--profiles-file", str(profiles_file)])
        self.assertIn("lattice", harness.ADAPTER_REGISTRY)


@unittest.skipUnless(
    importlib.util.find_spec("mlx_lm") is not None,
    "mock.patch resolves 'mlx_lm.load' by import, so mlx_lm must be installed "
    "(macOS-only dependency; skipped on Linux CI)",
)
class MlxFallbackProvenanceTest(unittest.TestCase):
    """A missing/corrupt local model dir makes `MlxAdapter` fall back to the
    Hub artifact -- the raw observation must record that via `actual_model`,
    never silently label it as the requested local artifact.
    """

    def test_fallback_load_reports_hub_identifier_via_actual_model(self):
        fake_model, fake_tok, fake_sampler = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
        fake_model.parameters.return_value = mock.MagicMock()

        def fake_load(path):
            if path == "Qwen/Qwen3.5-0.8B":
                return fake_model, fake_tok
            raise OSError(f"no such local model: {path}")

        adapter = adapters.MlxAdapter(source_model_dir=Path("/nonexistent/local/model"))
        with (
            mock.patch("mlx_lm.load", side_effect=fake_load),
            mock.patch("mlx.nn.quantize"),
            mock.patch("mlx.core.eval"),
            mock.patch("mlx_lm.sample_utils.make_sampler", return_value=fake_sampler),
            mock.patch("mlx_lm.generate", return_value="irrelevant generated text"),
        ):
            result = adapter.run(
                prompt="p", n_tokens=32, warmup=False, model="qwen3.5-0.8b", quantization="q8"
            )
        self.assertEqual(result.actual_model, "Qwen/Qwen3.5-0.8B")
        self.assertEqual(result.actual_completion_tokens, 32)

    def test_local_load_success_reports_no_fallback(self):
        fake_model, fake_tok, fake_sampler = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
        fake_model.parameters.return_value = mock.MagicMock()

        adapter = adapters.MlxAdapter(source_model_dir=Path("/some/local/model"))
        with (
            mock.patch("mlx_lm.load", return_value=(fake_model, fake_tok)),
            mock.patch("mlx.nn.quantize"),
            mock.patch("mlx.core.eval"),
            mock.patch("mlx_lm.sample_utils.make_sampler", return_value=fake_sampler),
            mock.patch("mlx_lm.generate", return_value="irrelevant generated text"),
        ):
            result = adapter.run(
                prompt="p", n_tokens=32, warmup=False, model="qwen3.5-0.8b", quantization="q8"
            )
        self.assertIsNone(result.actual_model)


if __name__ == "__main__":
    unittest.main()
