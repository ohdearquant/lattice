#!/usr/bin/env python3
"""Synthetic unit and CLI controls for quality_compare; no model required."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import quality_compare as qc

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures" / "quality_compare"


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.reference = load_fixture("reference.json")
        self.candidate = load_fixture("candidate.json")

    def test_divergences_and_explicit_denominators(self):
        result = qc.compare(self.reference, self.candidate)
        self.assertEqual([row["id"] for row in result["cases"]], ["a", "b", "c"])
        self.assertEqual(result["cases"][0]["first_divergence"], 9)
        self.assertEqual(result["cases"][0]["exact_prefix_length"], 9)
        self.assertEqual(result["first_divergence_histogram"], {"9": 1, "identical": 2})
        self.assertEqual(result["prefix_matches"]["10"], {"matched": 1, "eligible": 2, "insufficient": 1, "rate": 0.5})
        self.assertEqual(result["prefix_matches"]["50"], {"matched": 0, "eligible": 1, "insufficient": 2, "rate": 0.0})

    def test_first_mismatch_boundaries(self):
        for position in (0, 9, 10, 49):
            with self.subTest(position=position):
                candidate = copy.deepcopy(self.reference)
                candidate["cases"][0]["generated_ids"][position] = 999
                case = qc.compare(self.reference, candidate)["cases"][0]
                self.assertEqual(case["first_divergence"], position)
                self.assertEqual(case["exact_prefix_length"], position)
                self.assertEqual(case["divergence_kind"], "token")

    def test_mismatch_after_fifty_preserves_fifty_token_prefix(self):
        reference = copy.deepcopy(self.reference)
        reference["config"]["max_new_tokens"] = 60
        reference["cases"] = [reference["cases"][0]]
        reference["cases"][0]["generated_ids"] = list(range(60))
        candidate = copy.deepcopy(reference)
        candidate["cases"][0]["generated_ids"][50] = 999
        result = qc.compare(reference, candidate)
        self.assertEqual(result["cases"][0]["first_divergence"], 50)
        self.assertEqual(result["cases"][0]["exact_prefix_length"], 50)
        self.assertEqual(result["prefix_matches"]["50"]["matched"], 1)

    def test_length_and_termination_are_distinct(self):
        candidate = copy.deepcopy(self.reference)
        candidate["cases"][1]["generated_ids"] = list(range(9))
        result = qc.compare(self.reference, candidate)
        self.assertEqual((result["cases"][1]["first_divergence"], result["cases"][1]["divergence_kind"]), (9, "length"))
        self.assertEqual(result["prefix_matches"]["10"]["insufficient"], 2)
        candidate = copy.deepcopy(self.reference)
        candidate["cases"][1]["stop_reason"] = "stop"
        case = qc.compare(self.reference, candidate)["cases"][1]
        self.assertEqual((case["first_divergence"], case["divergence_kind"]), (10, "termination"))

    def test_no_eligible_cases_report_no_rate(self):
        reference = copy.deepcopy(self.reference)
        reference["cases"] = [reference["cases"][2]]
        result = qc.compare(reference, copy.deepcopy(reference))
        self.assertEqual(result["prefix_matches"]["10"], {"matched": 0, "eligible": 0, "insufficient": 1, "rate": None})

    def test_reordered_cases_are_matched_by_id(self):
        candidate = copy.deepcopy(self.reference)
        candidate["cases"].reverse()
        result = qc.compare(self.reference, candidate)
        self.assertEqual([row["id"] for row in result["cases"]], ["a", "b", "c"])
        self.assertTrue(all(row["divergence_kind"] == "identical" for row in result["cases"]))

    def test_invalid_or_mismatched_inputs_are_refused(self):
        mutations = {
            "schema": lambda d: d.update(schema_version=True),
            "schema float": lambda d: d.update(schema_version=1.0),
            "name": lambda d: d.update(name=""),
            "empty cases": lambda d: d.update(cases=[]),
            "cases not list": lambda d: d.update(cases={}),
            "case not object": lambda d: d["cases"].__setitem__(0, 1),
            "missing case ID": lambda d: d["cases"][0].pop("id"),
            "duplicate ID": lambda d: d["cases"].append(copy.deepcopy(d["cases"][0])),
            "missing ID": lambda d: d["cases"].pop(),
            "extra ID": lambda d: d["cases"][0].update(id="new"),
            "prompt": lambda d: d["cases"][0].update(prompt="changed"),
            "empty prompt": lambda d: d["cases"][0].update(prompt=""),
            "prompt IDs": lambda d: d["cases"][0].update(prompt_ids=[9]),
            "empty prompt IDs": lambda d: d["cases"][0].update(prompt_ids=[]),
            "config": lambda d: d["config"].update(repetition_penalty=1.1),
            "config not object": lambda d: d.update(config=[]),
            "non-greedy": lambda d: d["config"].update(mode="sample"),
            "config JSON type": lambda d: d["config"].update(repetition_penalty=True),
            "stop policy": lambda d: d["config"].update(stop_policy="include-stop-token"),
            "missing stop policy": lambda d: d["config"].pop("stop_policy"),
            "max": lambda d: d["config"].update(max_new_tokens=True),
            "zero max": lambda d: d["config"].update(max_new_tokens=0),
            "nonfinite config": lambda d: d["config"].update(extra=float("nan")),
            "bad token": lambda d: d["cases"][0]["generated_ids"].__setitem__(0, True),
            "token list missing": lambda d: d["cases"][0].pop("generated_ids"),
            "negative token": lambda d: d["cases"][0]["generated_ids"].__setitem__(0, -1),
            "overlong": lambda d: d["cases"][0]["generated_ids"].append(1),
            "truncated length": lambda d: d["cases"][0]["generated_ids"].pop(),
            "bad reason": lambda d: d["cases"][1].update(stop_reason="unknown"),
            "missing completion": lambda d: d["cases"][1].pop("complete"),
            "unsupported logits": lambda d: d["cases"][1].update(logits=[]),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                candidate = copy.deepcopy(self.reference)
                mutate(candidate)
                with self.assertRaises(qc.DumpError):
                    qc.compare(self.reference, candidate)

    def test_cli_fixture_and_malformed_dump_refuse(self):
        command = [sys.executable, str(ROOT / "quality_compare.py"),
                   str(FIXTURES / "reference.json"), str(FIXTURES / "candidate.json")]
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["prefix_matches"]["10"]["eligible"], 2)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            for text in ('{"schema_version": 1, "schema_version": 1}', '{"value": NaN}', '{'):
                with self.subTest(text=text):
                    bad.write_text(text, encoding="utf-8")
                    failed = subprocess.run([*command[:2], str(FIXTURES / "reference.json"), str(bad)],
                                            capture_output=True, text=True, check=False)
                    self.assertEqual(failed.returncode, 2)
                    self.assertEqual(failed.stdout, "")
                    self.assertIn("refused", failed.stderr)
            bad.write_bytes(b"\xff")
            invalid_utf8 = subprocess.run([*command[:2], str(FIXTURES / "reference.json"), str(bad)],
                                          capture_output=True, text=True, check=False)
            self.assertEqual(invalid_utf8.returncode, 2)
            missing = subprocess.run([*command[:2], str(FIXTURES / "reference.json"), str(bad.parent / "missing.json")],
                                     capture_output=True, text=True, check=False)
            self.assertEqual(missing.returncode, 2)


if __name__ == "__main__":
    unittest.main()
