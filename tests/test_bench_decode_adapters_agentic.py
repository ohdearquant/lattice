"""Deterministic tests for the agentic decode-benchmark profile migration."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
sys.path.insert(0, str(_SCRIPTS))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


harness = _load("bench_decode_harness", "bench_decode_harness.py")
agentic = _load("bench_decode_adapters_agentic", "bench_decode_adapters_agentic.py")


@dataclass
class _Clock:
    value: int = 0

    def __call__(self) -> int:
        self.value += 1_000
        return self.value


class _Fake:
    def __init__(self, component: bool = False):
        self.component = component
        self.calls: list[tuple[int, bool, str]] = []

    def run(self, *, prompt: str, n_tokens: int, warmup: bool, model: str, quantization: str):
        self.calls.append((n_tokens, warmup, prompt))
        components = {1: 10_000_000, 100: 210_000_000} if self.component and not warmup else None
        return harness.AdapterRunResult(
            actual_completion_tokens=n_tokens,
            actual_prompt_tokens=1007,
            component_ns=components,
            engine_version="fake",
        )


class AgenticProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, profiles = harness.load_profiles_file(agentic.PROFILES_FILE)
        cls.profile = agentic.configure_profile(profiles["agentic"], ctx=2000, runs=3, padded_prompt="PAD")

    def test_runtime_parameters_and_prompts(self):
        self.assertEqual(self.profile.windows, (1, 100))
        self.assertEqual(self.profile.measured_repeats, 3)
        self.assertEqual(self.profile.requested_prompt_tokens, 2000)
        groups = {group.name: group for group in self.profile.engine_groups}
        self.assertEqual(self.profile.prompt, "PAD")
        self.assertEqual(groups["ollama"].measured_prompt, agentic.make_ollama_prompt(2000))
        self.assertEqual(groups["mlx"].measured_order, "repeat_major")
        self.assertEqual(groups["ollama"].measured_calls[0].yields_windows, (1, 100))

    def test_exact_legacy_call_order(self):
        lattice, ollama, mlx = _Fake(), _Fake(component=True), _Fake()
        result = harness.run_profile(
            self.profile,
            {"lattice": lattice, "ollama": ollama, "mlx": mlx},
            clock=_Clock(),
            git_sha_value="abc",
            hardware_id_value="host",
        )
        self.assertEqual([c[:2] for c in lattice.calls], [(1, False)] * 3 + [(100, False)] * 3)
        self.assertEqual([c[:2] for c in ollama.calls], [(4, True)] + [(100, False)] * 3)
        self.assertEqual(
            [c[:2] for c in mlx.calls],
            [(4, True), (1, False), (100, False), (1, False), (100, False), (1, False), (100, False)],
        )
        slopes = harness.aggregate(result)
        ollama_slope = next(s for s in slopes if s.engine == "ollama")
        self.assertAlmostEqual(ollama_slope.slope_tok_per_s, 495.0)


class ParsingTest(unittest.TestCase):
    def test_lattice_result_and_prompt_count(self):
        parsed = agentic.parse_lattice_output(
            "RESULT n_req=100 completion=100 total_ms=250.5\n",
            "[bench] prompt_tokens=1007\n",
            n_tokens=100,
        )
        self.assertEqual(parsed.actual_completion_tokens, 100)
        self.assertEqual(parsed.actual_prompt_tokens, 1007)
        self.assertEqual(parsed.native_ns, 250_500_000)

    def test_prompt_marker_must_be_a_whole_line(self):
        """An embedded marker must not be mined for benchmark evidence.

        The emitter writes `[bench] prompt_tokens=N` alone on a line. An
        unanchored search accepts it inside a longer diagnostic and reports
        whatever digits follow `=` as the measured prompt length, so this
        fails if the match is loosened back to `search` over the whole blob.
        """
        with self.assertRaises(agentic.AdapterOutputError):
            agentic.parse_lattice_output(
                "RESULT n_req=100 completion=100 total_ms=250.5\n",
                "note: retrying after [bench] prompt_tokens=9999 was rejected\n",
                n_tokens=100,
            )

    def test_prompt_marker_must_appear_exactly_once(self):
        """Two markers mean two candidate answers, so neither may be picked.

        Fails if the parser reverts to taking the first match, which is what
        an unanchored search does and what makes a stale line from an earlier
        attempt indistinguishable from this run's diagnostic.
        """
        with self.assertRaises(agentic.AdapterOutputError):
            agentic.parse_lattice_output(
                "RESULT n_req=100 completion=100 total_ms=250.5\n",
                "[bench] prompt_tokens=1007\n[bench] prompt_tokens=2014\n",
                n_tokens=100,
            )

    def test_ollama_components_are_ttft_and_total(self):
        result = agentic.ollama_response_to_result(
            {
                "load_duration": 2_000_000,
                "prompt_eval_duration": 8_000_000,
                "eval_duration": 200_000_000,
                "eval_count": 100,
                "prompt_eval_count": 1012,
            }
        )
        self.assertEqual(result.component_ns, {1: 10_000_000, 100: 210_000_000})
        self.assertIsNone(result.native_ns)


class CanonicalCliTest(unittest.TestCase):
    def test_harness_dispatches_agentic_runtime_flags(self):
        with mock.patch.object(agentic, "main", return_value=0) as delegated:
            status = harness.main(
                [
                    "run",
                    "--profile",
                    "agentic",
                    "--ctx",
                    "2048",
                    "--runs",
                    "3",
                    "--allow-missing-engine",
                ]
            )
        self.assertEqual(status, 0)
        delegated.assert_called_once_with(
            ["--ctx", "2048", "--runs", "3", "--allow-missing-engine"]
        )

    def test_mlx_prompt_load_failure_also_skips_lattice_fail_closed(self):
        with (
            mock.patch.object(agentic, "_mlx_available", return_value=True),
            mock.patch.object(agentic, "_ollama_available", return_value=False),
            mock.patch.object(agentic.MlxAdapter, "padded_prompt", side_effect=RuntimeError("bad tokenizer")),
        ):
            registered, missing, prompt = agentic.register_available_adapters(1000)
        self.assertNotIn("mlx", registered)
        self.assertNotIn("lattice", registered)
        self.assertIn("tokenizer load failed", missing["mlx"])
        self.assertIn("tokenizer-padded prompt", missing["lattice"])
        self.assertTrue(prompt.startswith(agentic.BASE))


class ExitStatusMeansMeasuredTest(unittest.TestCase):
    """A zero exit must mean a measurement happened, not that reporting finished.

    `--allow-missing-engine` exists so a sweep can proceed when one framework is
    not installed. It must not also let a run where EVERY engine was absent
    report success: the status notices print either way, and a caller reading
    only the exit code cannot distinguish a full comparison from a table of
    unavailable rows.
    """

    def test_all_engines_unavailable_is_a_failure(self):
        unavailable = [
            {"engine": "mlx", "source": "unavailable", "unavailable_reason": "not installed"},
            {"engine": "lattice", "source": "unavailable", "unavailable_reason": "not built"},
        ]
        with mock.patch.object(agentic, "run_context", return_value=unavailable):
            status = agentic.main(["--ctx", "1024", "--runs", "1", "--allow-missing-engine"])
        self.assertEqual(status, 1, "a run that measured nothing must not exit 0")

    def test_one_live_row_among_unavailable_ones_still_succeeds(self):
        """The tolerance the flag was added for must survive the stricter check."""
        mixed = [
            {"engine": "mlx", "source": "unavailable", "unavailable_reason": "not installed"},
            {"engine": "lattice", "source": "live", "tok_s": 157.0},
        ]
        with mock.patch.object(agentic, "run_context", return_value=mixed):
            status = agentic.main(["--ctx", "1024", "--runs", "1", "--allow-missing-engine"])
        self.assertEqual(status, 0)


class PartialSweepIsPersistedTest(unittest.TestCase):
    """A sweep that dies partway still wrote measurements; they must survive.

    Engine availability differs between contexts -- the per-context
    `padded_prompt(ctx)` call drops both mlx and lattice when it fails -- so a
    long context can measure nothing where a short one measured fine. The run
    must fail, because it did not complete, and it must still persist what it
    measured, under a name that cannot be mistaken for a complete sweep.
    """

    def _sweep(self, tmp, side_effect):
        with mock.patch.object(agentic, "OUT_DIR", tmp), \
             mock.patch.object(agentic, "run_context", side_effect=side_effect):
            return agentic.main(["--runs", "1", "--sweep", "--allow-missing-engine"])

    def test_rows_measured_before_the_failure_are_written(self):
        import json
        import tempfile

        live = [{"engine": "lattice", "source": "live", "tok_s": 157.0}]
        # First context measures, the next one loses every engine.
        effects = [live, harness.MissingEngineError("no measured observation")]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            status = self._sweep(tmp, effects)
            self.assertEqual(status, 1, "an incomplete sweep must not exit 0")
            # Checked before the presence assertion below so that each way of
            # breaking this reports its own cause: writing the partial rows to
            # the canonical name fails here, and discarding them fails there.
            # Ordered the other way, the first assertion shadows the second and
            # both mutations produce the same misleading message.
            self.assertFalse(
                (tmp / "agentic_sweep.json").exists(),
                "a partial sweep must not occupy the canonical artifact name",
            )
            partial = tmp / "agentic_sweep.partial.json"
            self.assertTrue(partial.is_file(), "measured rows were discarded on failure")
            self.assertEqual(json.loads(partial.read_text()), live)

    def test_a_complete_sweep_uses_the_canonical_name(self):
        import tempfile

        live = [{"engine": "lattice", "source": "live", "tok_s": 157.0}]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            status = self._sweep(tmp, lambda *a, **k: list(live))
            self.assertEqual(status, 0)
            self.assertTrue((tmp / "agentic_sweep.json").is_file())
            self.assertFalse((tmp / "agentic_sweep.partial.json").exists())


if __name__ == "__main__":
    unittest.main()
