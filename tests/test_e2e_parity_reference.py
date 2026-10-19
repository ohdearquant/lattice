import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import ExitStack, contextmanager
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts/e2e_parity_check.py"
ROOT = SCRIPT_PATH.parent.parent
CANDIDATE_SELECTION_PATH = (
    ROOT
    / "crates/inference/tests/fixtures/e2e_parity_reference_v1/"
    "candidate_selection.json"
)
RUNBOOK_PATH = ROOT / "docs/e2e-parity-frozen-reference.md"
SPEC = importlib.util.spec_from_file_location("e2e_parity_check", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PARITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARITY)


def output(tokens, margins):
    return {
        "generated_ids": tokens,
        "logit_margins": margins,
        "text": "",
        "elapsed_s": 1.0,
        "tok_per_sec": len(tokens),
    }


class RegenerationDeterminismTest(unittest.TestCase):
    def test_accepts_matching_thread_outputs_and_finds_global_minimum(self):
        one = {
            "first": output([1, 2], [4.0, 2.0]),
            "second": output([3, 4], [3.0, 1.5]),
        }
        four = {
            "first": output([1, 2], [3.5, 2.5]),
            "second": output([3, 4], [2.5, 1.75]),
        }

        result = PARITY.validate_regeneration_outputs(
            {1: one, 4: four}, ["first", "second"], 1.0
        )

        self.assertEqual(result["global_minimum_margin"], 1.5)
        self.assertEqual(result["global_minimum_prompt"], "second")
        self.assertEqual(result["global_minimum_position"], 1)

    def test_names_prompt_and_position_on_token_disagreement(self):
        one = {"prompt": output([10, 20, 30], [3.0, 2.0, 1.0])}
        four = {"prompt": output([10, 21, 30], [3.0, 2.0, 1.0])}

        with self.assertRaisesRegex(
            RuntimeError, "prompt 'prompt'.*token position 1.*20.*21"
        ):
            PARITY.validate_regeneration_outputs(
                {1: one, 4: four}, ["prompt"], 0.5
            )

    def test_rejects_non_finite_or_below_floor_margin(self):
        for margin, message in (
            (math.nan, "non-finite"),
            (math.inf, "non-finite"),
            (0.25, "below refusal floor"),
        ):
            with self.subTest(margin=margin):
                runs = {
                    1: {"prompt": output([10], [margin])},
                    4: {"prompt": output([10], [1.0])},
                }
                with self.assertRaisesRegex(RuntimeError, message):
                    PARITY.validate_regeneration_outputs(
                        runs, ["prompt"], 0.5
                    )


class CandidateSelectionEvidenceTest(unittest.TestCase):
    def test_every_candidate_has_thread_measurements_and_selected_winners(self):
        selection = json.loads(CANDIDATE_SELECTION_PATH.read_text())
        measurements = selection["measurements"]

        self.assertEqual(
            selection["provenance_limit"],
            (
                "The preserved measurement outputs did not record model revision, "
                "package versions, or measurement date."
            ),
        )
        candidates = {
            candidate["id"]
            for pool in selection["pools"]
            for candidate in pool["candidates"]
        }
        self.assertEqual(set(measurements), candidates)
        for candidate_id, measurement in measurements.items():
            with self.subTest(candidate=candidate_id):
                self.assertEqual(set(measurement["minimum_margin"]), {"1", "4"})
                self.assertTrue(measurement["generated_ids_agree_across_threads"])
                for minima in measurement["minimum_margin"].values():
                    self.assertEqual(set(minima), {"4_tokens", "15_tokens"})

        self.assertEqual(
            {
                pool["id"]: pool["selected_candidate"]
                for pool in selection["pools"]
            },
            {
                "short-general-prose": "short-01",
                "long-prefill-python": "long-01",
            },
        )

    def test_regeneration_help_and_workflow_link_to_the_runbook(self):
        runbook = "docs/e2e-parity-frozen-reference.md"
        self.assertTrue(RUNBOOK_PATH.is_file())
        self.assertIn(runbook, SCRIPT_PATH.read_text())
        workflow = (ROOT / ".github/workflows/e2e-parity.yml").read_text()
        self.assertIn(runbook, workflow)


def gen(tokens, text="", tok_per_sec=1.0):
    return {
        "generated_ids": tokens,
        "logit_margins": [1.0] * len(tokens),
        "text": text,
        "elapsed_s": 1.0,
        "tok_per_sec": tok_per_sec,
    }


class CompareGateNegativeControlTest(unittest.TestCase):
    """The gate must still FAIL.

    A frozen reference removes the live HF run, so nothing external would
    notice if compare() silently stopped discriminating. These are the
    durable controls: each one fails if window_match/pass is forced true.
    """

    WINDOW = 3

    def test_first_token_mismatch_fails(self):
        verdict = PARITY.compare(
            "p", gen([10, 20, 30, 40]), gen([11, 20, 30, 40]), self.WINDOW
        )
        self.assertEqual(verdict["first_mismatch"], 0)
        self.assertFalse(verdict["window_match"])
        self.assertFalse(verdict["pass"])

    def test_mismatch_inside_window_fails(self):
        verdict = PARITY.compare(
            "p", gen([10, 20, 30, 40]), gen([10, 20, 31, 40]), self.WINDOW
        )
        self.assertEqual(verdict["first_mismatch"], 2)
        self.assertFalse(verdict["pass"])

    def test_matching_prefix_passes(self):
        """Positive control: without this, a gate stuck at FAIL looks correct."""
        verdict = PARITY.compare(
            "p", gen([10, 20, 30, 40]), gen([10, 20, 30, 40]), self.WINDOW
        )
        self.assertIsNone(verdict["first_mismatch"])
        self.assertTrue(verdict["pass"])
        self.assertEqual(verdict["agree_rate"], 1.0)

    def test_mismatch_at_window_boundary_passes(self):
        """Divergence at index == window is outside the gated prefix."""
        verdict = PARITY.compare(
            "p", gen([10, 20, 30, 40]), gen([10, 20, 30, 41]), self.WINDOW
        )
        self.assertEqual(verdict["first_mismatch"], 3)
        self.assertTrue(verdict["pass"])

    def test_empty_generation_fails_closed(self):
        """An instrument that compared NOTHING must not report a pass.

        min_len == 0 makes first_mismatch 0 and agree_rate 0; the verdict has
        to be a refusal, never a vacuous green.
        """
        verdict = PARITY.compare("p", gen([]), gen([]), self.WINDOW)
        self.assertFalse(verdict["pass"])
        self.assertEqual(verdict["total_compared"], 0)
        self.assertEqual(verdict["agree_rate"], 0)

    def test_lattice_truncated_before_window_fails(self):
        """A short lattice generation cannot satisfy a longer match window."""
        verdict = PARITY.compare("p", gen([10, 20, 30, 40]), gen([10, 20]), self.WINDOW)
        self.assertFalse(verdict["pass"])
        self.assertEqual(verdict["total_compared"], 2)


class FrozenReferenceLoaderRefusalTest(unittest.TestCase):
    """The loader must refuse missing / empty / malformed fixtures."""

    def _load_with(self, path):
        with unittest.mock.patch.object(PARITY, "REFERENCE_PATH", path):
            return PARITY.load_frozen_reference(64)

    def test_missing_fixture_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "missing, unreadable, or malformed"):
                self._load_with(Path(d) / "does_not_exist.json")

    def test_empty_fixture_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty.json"
            p.write_text("")
            with self.assertRaisesRegex(RuntimeError, "missing, unreadable, or malformed"):
                self._load_with(p)

    def test_non_object_root_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "list.json"
            p.write_text("[]")
            with self.assertRaisesRegex(RuntimeError, "root must be a JSON object"):
                self._load_with(p)

    def test_empty_object_refuses_at_the_schema_guard(self):
        """An empty object fails the SCHEMA guard.

        Named for what it actually proves. It says nothing about the
        provenance check, which sits behind the schema guard and is covered
        separately below.
        """
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "obj.json"
            p.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "malformed or do not match this gate"):
                self._load_with(p)

    def _schema_valid_fixture(self, versions):
        """A fixture that clears every schema check, so only provenance can fail."""
        return {
            "schema_version": 1,
            "package_versions": versions,
            "model": {
                "repo_id": PARITY.MODEL_REPO,
                "revision": PARITY.MODEL_REVISION,
            },
            "generation": {
                "max_new_tokens": max(PARITY.REFERENCE_TOKEN_COUNTS),
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            },
            "prompts": [],
        }

    FIXTURE_INSTALLED_VERSION = "9.9.9+fixture"

    def _patched_installed(self):
        """Own the installed side of the comparison under test.

        Both provenance tests below read the real environment, so they ran only
        where torch and transformers happen to be installed and ERRORED
        everywhere else, this repo's own no-engine job included. Their subject
        is the recorded-vs-installed comparison inside the loader, not which
        packages the host has, so the installed side belongs to the test.
        """
        return unittest.mock.patch.object(
            PARITY,
            "installed_reference_versions",
            lambda: {
                pkg: self.FIXTURE_INSTALLED_VERSION
                for pkg in PARITY.REFERENCE_PACKAGES
            },
        )

    def test_wrong_package_version_refuses_on_provenance_alone(self):
        """Isolates the provenance check.

        The fixture is schema-valid in every other respect, so the only thing
        that can reject it is the recorded-vs-installed version comparison.
        A test that only ever feeds malformed input cannot show that
        provenance is enforced at all.
        """
        bogus = {pkg: "0.0.0+not-installed" for pkg in PARITY.REFERENCE_PACKAGES}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "prov.json"
            p.write_text(json.dumps(self._schema_valid_fixture(bogus)))
            with self._patched_installed():
                with self.assertRaisesRegex(
                    RuntimeError, "package version mismatch for"
                ):
                    self._load_with(p)

    def test_missing_package_entry_refuses_on_provenance_alone(self):
        """Same isolation, for an incomplete rather than wrong provenance set."""
        partial = {
            pkg: self.FIXTURE_INSTALLED_VERSION
            for pkg in PARITY.REFERENCE_PACKAGES[:-1]
        }
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "partial.json"
            p.write_text(json.dumps(self._schema_valid_fixture(partial)))
            with self._patched_installed():
                with self.assertRaisesRegex(RuntimeError, "must name exactly"):
                    self._load_with(p)

    def test_frozen_loader_failure_never_falls_back_to_live_hf(self):
        with tempfile.TemporaryDirectory() as d:
            model_dir = Path(d) / "model"
            model_dir.mkdir()
            lattice_bin = Path(d) / "lattice"
            lattice_bin.touch()
            with (
                unittest.mock.patch.object(PARITY, "MODEL_DIR", str(model_dir)),
                unittest.mock.patch.object(PARITY, "LATTICE_BIN", str(lattice_bin)),
                unittest.mock.patch.object(PARITY, "_LATTICE_BIN_EXPLICIT", True),
                unittest.mock.patch.object(
                    PARITY, "REFERENCE_PATH", Path(d) / "missing.json"
                ),
                unittest.mock.patch.object(PARITY, "run_hf_reference") as run_hf,
                unittest.mock.patch.object(sys, "argv", [str(SCRIPT_PATH)]),
                unittest.mock.patch.dict(
                    os.environ, {"GITHUB_EVENT_NAME": "pull_request"}
                ),
            ):
                self.assertEqual(PARITY.main(), 2)
                run_hf.assert_not_called()


def verdict_row(passed, xfail_issue=None):
    row = {
        "prompt": "p",
        "pass": passed,
        "match_window": 3,
        "agree_rate": 1.0 if passed else 0.0,
        "total_agree": 3 if passed else 0,
        "total_compared": 3,
        "first_mismatch": None if passed else 0,
        "hf_tok_s": 1.0,
        "lat_tok_s": 1.0,
        "hf_text": "",
        "lat_text": "",
    }
    if xfail_issue is not None:
        row["xfail_issue"] = xfail_issue
    return row


class EmptyGatingSetTest(unittest.TestCase):
    """A waiver is not coverage, and a table of waivers is not a pass.

    The expected-divergence map is meant to grow one tracked issue at a time.
    Nothing noticed when it covered the whole prompt table: every remaining
    branch reported PASS and the process exited 0, so the gate announced parity
    over a population of zero.
    """

    def test_all_waived_refuses_instead_of_passing(self):
        rows = [verdict_row(False, "#535") for _ in range(4)]

        report = PARITY.render_report(rows)

        self.assertIn("NO GATING PROMPT", report)
        self.assertNotIn("PASS", report)

    def test_a_waived_prompt_that_passed_is_still_not_gating(self):
        """Excluded in BOTH directions, which is where the old count leaked.

        `len(results) - len(known)` counted only the waived prompts that
        DIVERGED, so a waived prompt that happened to match was reported as a
        gating prompt that passed.
        """
        rows = [verdict_row(True), verdict_row(True), verdict_row(True, "#535")]

        self.assertEqual(len(PARITY.gating_results(rows)), 2)

    def test_empty_gating_reason_claims_only_the_cases_it_owns(self):
        self.assertIsNone(
            PARITY.empty_gating_reason([verdict_row(True), verdict_row(False, "#535")])
        )
        self.assertIsNotNone(PARITY.empty_gating_reason([]))
        self.assertIsNotNone(
            PARITY.empty_gating_reason([verdict_row(False, "#535")])
        )


class RegenerationPublicationTest(unittest.TestCase):
    SENTINEL = b'{"previous": "reference"}\n'

    @contextmanager
    def _regeneration(
        self, *, prompts=None, mismatches=(), waived=(), existing=True, budget=15,
        backend="cpu"
    ):
        if prompts is None:
            prompts = [("first", 3), ("second", 3)]
        outputs = {prompt: gen(list(range(budget))) for prompt, _ in prompts}
        summary = PARITY.validate_regeneration_outputs(
            {1: outputs, 4: outputs},
            [prompt for prompt, _ in prompts],
            PARITY.REFERENCE_LOGIT_MARGIN_FLOOR,
        )
        versions = {package: "9.9.9+fixture" for package in PARITY.REFERENCE_PACKAGES}

        def lattice_output(prompt, max_tokens):
            self.assertEqual(max_tokens, budget)
            tokens = list(outputs[prompt]["generated_ids"])
            if prompt in mismatches:
                tokens[0] += 100
            return gen(tokens)

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            binary = root / "lattice"
            binary.touch()
            reference = root / "reference.json"
            if existing:
                reference.write_bytes(self.SENTINEL)
            report = root / "report.md"
            stdout, stderr = io.StringIO(), io.StringIO()
            for name, value in {
                "MODEL_DIR": str(model),
                "LATTICE_BIN": str(binary),
                "_LATTICE_BIN_EXPLICIT": True,
                "REFERENCE_PATH": reference,
                "REPORT_PATH": str(report),
                "MAX_TOKENS": budget,
                "PROMPTS": prompts,
                "METAL_EXPECTED_DIVERGENCE": {prompt: "#535" for prompt in waived},
            }.items():
                stack.enter_context(unittest.mock.patch.object(PARITY, name, value))
            workers = stack.enter_context(
                unittest.mock.patch.object(
                    PARITY, "run_regeneration_workers", return_value=(outputs, summary)
                )
            )
            lattice = stack.enter_context(
                unittest.mock.patch.object(
                    PARITY, "run_lattice_metal" if backend == "metal" else "run_lattice",
                    side_effect=lattice_output
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    PARITY, "installed_reference_versions", return_value=versions
                )
            )
            stack.enter_context(
                unittest.mock.patch.object(
                    sys, "argv", [str(SCRIPT_PATH), "--backend", backend, "--regenerate"]
                )
            )
            stack.enter_context(unittest.mock.patch.object(sys, "stdout", stdout))
            stack.enter_context(unittest.mock.patch.object(sys, "stderr", stderr))
            yield reference, report, stdout, stderr, workers, lattice

    def test_failed_parity_preserves_existing_reference(self):
        with self._regeneration(mismatches=("first",)) as state:
            reference, report, stdout, stderr, workers, lattice = state
            self.assertEqual(PARITY.main(), 1)
            self.assertEqual(reference.read_bytes(), self.SENTINEL)
            self.assertEqual(stdout.getvalue(), report.read_text() + "\n")
            self.assertIn("FAIL: 1/2 prompts failed parity gate", stderr.getvalue())
            self.assertNotIn("wrote frozen reference", stderr.getvalue())
            workers.assert_called_once_with(15)
            self.assertEqual(lattice.call_count, 2)

    def test_failed_parity_does_not_create_reference(self):
        with self._regeneration(mismatches=("first",), existing=False) as state:
            reference, _, _, stderr, _, _ = state
            self.assertEqual(PARITY.main(), 1)
            self.assertFalse(reference.exists())
            self.assertFalse(reference.with_suffix(".json.tmp").exists())
            self.assertNotIn("wrote frozen reference", stderr.getvalue())

    def test_pass_publishes_loadable_reference(self):
        with self._regeneration() as state:
            reference, _, _, stderr, _, _ = state
            self.assertEqual(PARITY.main(), 0)
            self.assertNotEqual(reference.read_bytes(), self.SENTINEL)
            for count in PARITY.REFERENCE_TOKEN_COUNTS:
                loaded = PARITY.load_frozen_reference(count)
                self.assertEqual(set(loaded), {"first", "second"})
                for output in loaded.values():
                    self.assertEqual(output["generated_ids"], list(range(count)))
            self.assertFalse(reference.with_suffix(".json.tmp").exists())
            self.assertIn("wrote frozen reference", stderr.getvalue())
            self.assertIn("PASS: all 2 gating prompts passed", stderr.getvalue())

    def test_empty_prompt_population_preserves_reference(self):
        with self._regeneration(prompts=[]) as state:
            reference, _, _, stderr, workers, lattice = state
            self.assertEqual(PARITY.main(), 2)
            self.assertEqual(reference.read_bytes(), self.SENTINEL)
            self.assertIn("NO GATING PROMPT", stderr.getvalue())
            self.assertNotIn("wrote frozen reference", stderr.getvalue())
            workers.assert_called_once_with(15)
            lattice.assert_not_called()

    def test_all_waived_population_preserves_reference(self):
        for mismatches in ((), ("first", "second")):
            with self.subTest(mismatches=mismatches):
                with self._regeneration(
                    mismatches=mismatches, waived=("first", "second"), backend="metal"
                ) as state:
                    reference, _, _, stderr, _, lattice = state
                    self.assertEqual(PARITY.main(), 2)
                    self.assertEqual(reference.read_bytes(), self.SENTINEL)
                    self.assertEqual(lattice.call_count, 2)
                    self.assertIn("NO GATING PROMPT", stderr.getvalue())
                    self.assertNotIn("wrote frozen reference", stderr.getvalue())

    def test_mixed_waived_population_still_publishes(self):
        with self._regeneration(
            mismatches=("second",), waived=("second",), backend="metal"
        ) as state:
            _, _, _, stderr, _, _ = state
            self.assertEqual(PARITY.main(), 0)
            loaded = PARITY.load_frozen_reference(15)
            self.assertEqual(loaded["second"]["generated_ids"], list(range(15)))
            self.assertIn("PASS: all 1 gating prompts passed", stderr.getvalue())
            self.assertIn("1 known divergence excluded", stderr.getvalue())

    def test_report_write_failure_preserves_reference(self):
        with self._regeneration() as state:
            reference, report, _, stderr, _, _ = state
            report.mkdir()
            self.assertEqual(PARITY.main(), 2)
            self.assertEqual(reference.read_bytes(), self.SENTINEL)
            self.assertIn("error: could not write report", stderr.getvalue())
            self.assertNotIn("wrote frozen reference", stderr.getvalue())

    def test_reference_write_failure_remains_setup_error(self):
        with self._regeneration() as state:
            reference, report, _, stderr, _, _ = state
            reference.with_suffix(".json.tmp").mkdir()
            self.assertEqual(PARITY.main(), 2)
            self.assertEqual(reference.read_bytes(), self.SENTINEL)
            self.assertTrue(report.is_file())
            self.assertIn("error: could not write frozen reference", stderr.getvalue())
            self.assertNotIn("wrote frozen reference", stderr.getvalue())
            self.assertNotIn("PASS: all", stderr.getvalue())

    def test_short_regeneration_budget_remains_setup_error(self):
        with self._regeneration(budget=4, mismatches=("first",)) as state:
            reference, _, _, stderr, workers, lattice = state
            self.assertEqual(PARITY.main(), 2)
            self.assertEqual(reference.read_bytes(), self.SENTINEL)
            self.assertIn("--regenerate requires E2E_MAX_TOKENS >= 15", stderr.getvalue())
            workers.assert_not_called()
            lattice.assert_not_called()


class SuiteIsWholeWhenRunAsAScriptTest(unittest.TestCase):
    """This module used to execute 5 of its 18 tests and print OK.

    A second `if __name__ == "__main__": unittest.main()` sat in the middle of
    the file, so running it the way the workflow runs its siblings stopped
    there: the classes defined below that line were never defined, and the
    truncated run exited 0. Measured 5 as a script against 18 under
    `python -m unittest` before removing it.

    An AST scan of all 26 test modules found no second instance, with the
    pre-fix copy of this file used as the positive control for the scan.

    This guard cannot catch the defect in the mode that has it: a stray
    mid-file guard truncates the run before this class is defined, so as a
    script it would pass by not existing. That is why the workflow invokes
    this module as `python3 -m unittest`, where `__name__` is never
    `__main__` and no guard can truncate anything. The guard below is the
    backstop for anyone running the file directly.
    """

    def test_exactly_one_main_guard_and_it_is_last(self):
        import ast

        tree = ast.parse(Path(__file__).read_text())
        guards = [
            index
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.If) and "__main__" in ast.dump(node.test)
        ]

        self.assertEqual(len(guards), 1)
        self.assertEqual(guards[0], len(tree.body) - 1)


if __name__ == "__main__":
    unittest.main()
