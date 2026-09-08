#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Stdlib tests; --validator and --schemas enable the real-parser integration legs."""

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import synth_khive_dsl as synth

VALIDATOR = None
SCHEMAS = None


def schema(verb, pack="kg", params=()):
    return {
        "verb": verb,
        "pack": pack,
        "params": [
            {"name": name, "type": typ, "required": required}
            for name, typ, required in params
        ],
    }


def fixture_schemas():
    return {
        "memory.recall": schema(
            "memory.recall",
            "memory",
            [("query", "string", True), ("limit", "integer", False)],
        ),
        "comm.thread": schema("comm.thread", "comm", [("id", "string", True)]),
        "comm.mark_read": schema(
            "comm.mark_read", "comm", [("ids", "array of string", True)]
        ),
        "schedule.remind": schema(
            "schedule.remind",
            "schedule",
            [("content", "string", True), ("at", "string", True)],
        ),
        "schedule.agenda": schema("schedule.agenda", "schedule"),
        "knowledge.delete_atoms": schema(
            "knowledge.delete_atoms", "knowledge", [("ids", "array<string>", True)]
        ),
        "get": schema("get", params=[("id", "uuid", True)]),
        "link": schema(
            "link",
            params=[
                ("source_id", "uuid", False),
                ("target_id", "uuid", False),
                ("relation", "string", False),
                ("weight", "number", False),
                ("metadata", "object", False),
            ],
        ),
    }


def parser_record(completion, ops, mode="single", line=1):
    return {
        "ok": True,
        "parser": synth.PARSER,
        "parser_source_sha256": "a" * 64,
        "executed": False,
        "line": line,
        "completion": completion,
        "completion_parsed_unchanged": True,
        "mode": mode,
        "ops": ops,
        "ast_json_roundtrip_equal": True,
        "parser_roundtrip_equal": True,
    }


class SplitAndSchemaTests(unittest.TestCase):
    def test_stratified_assignment_is_stable_and_schedule_is_entirely_test(self):
        schemas = {f"pack.v{i}": schema(f"pack.v{i}", "pack") for i in range(20)}
        schemas.update(
            {f"schedule.v{i}": schema(f"schedule.v{i}", "schedule") for i in range(4)}
        )
        splits = synth.split_verbs(schemas)
        self.assertEqual(
            Counter(splits[v] for v in schemas if v.startswith("pack.")),
            {"train": 16, "valid": 2, "test": 2},
        )
        self.assertTrue(
            all(splits[v] == "test" for v in schemas if v.startswith("schedule."))
        )
        self.assertEqual(
            splits, synth.split_verbs(dict(reversed(list(schemas.items()))))
        )

    def test_composition_leakage_is_rejected_even_when_first_verb_is_train(self):
        item = synth.Example(
            "Do both", "[a(),b()]", ["a", "b"], "p", "train", "parallel", "parallel", []
        )
        with self.assertRaisesRegex(synth.CurationError, "crosses"):
            synth.add_split_guard(item, {"a": "train", "b": "test"})
        item.verbs = ["a", "a"]
        synth.add_split_guard(item, {"a": "train"})

    def test_dedup_and_three_completion_cap(self):
        pool = [
            synth.Example(
                f"Ask {i}", "get()", ["get"], "kg", "train", "intent", "single", []
            )
            for i in range(10)
        ]
        selected, drops = synth.select_candidates(
            {"kg": [pool[0], *pool]}, {"get": "train"}, per_pack=10
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(drops["completion_frequency_cap"], 7)
        self.assertEqual(drops["exact_duplicate"], 1)

    def test_schema_names_required_fields_types_and_null_are_checked(self):
        schemas = fixture_schemas()
        for args in (
            {"q": synth.tagged("cache")},
            {"query": synth.tagged(None)},
            {"query": synth.tagged("cache"), "limit": synth.tagged(True)},
            {},
        ):
            with self.subTest(args=args), self.assertRaises(synth.CurationError):
                synth.validate_ast(
                    {
                        "mode": "single",
                        "ops": [{"tool": "memory.recall", "args": args}],
                    },
                    schemas,
                )
        synth.validate_ast(
            {
                "mode": "single",
                "ops": [synth.op_ast("memory.recall", {"query": "cache"})],
            },
            schemas,
        )
        with self.assertRaisesRegex(synth.CurationError, "internal"):
            synth.validate_ast(
                {"mode": "single", "ops": [synth.op_ast("exec.run", {})]}, schemas
            )

    def test_live_type_spellings(self):
        self.assertTrue(synth.value_matches(["a"], "array<string>"))
        self.assertTrue(synth.value_matches([{"name": "a"}], "array of object"))
        self.assertTrue(synth.value_matches(["draft"], "string | array<string>"))
        self.assertFalse(synth.value_matches([1], "array of string"))
        self.assertFalse(synth.value_matches(True, "float"))
        self.assertFalse(synth.value_matches(float("inf"), "number"))

    def test_missing_schema_capture_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "_verbs.json").write_text(
                synth.compact({"total": 1, "verbs": [{"verb": "get", "pack": "kg"}]})
            )
            with self.assertRaisesRegex(synth.CurationError, "complete schema"):
                synth.read_schemas(root)

    def test_output_refuses_tracked_and_nonignored_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text(".khive/\n")
            (root / "data").mkdir()
            (root / "data" / "tracked").write_text("source")
            subprocess.run(["git", "-C", str(root), "add", "data/tracked"], check=True)
            for out in (root / "data", root / "new-data"):
                with self.assertRaisesRegex(synth.CurationError, "tracked"):
                    synth.safe_output(out)
            self.assertEqual(
                synth.safe_output(root / ".khive" / "data"),
                (root / ".khive" / "data").resolve(),
            )

    def test_unverified_prev_ref_is_rejected(self):
        schemas = fixture_schemas()
        parsed = {
            "mode": "chain",
            "ops": [
                synth.op_ast("get", {"id": synth.fixture(0)["id"]}),
                synth.op_ast("get", {"id": synth.Prev("imaginary_id")}),
            ],
        }
        with self.assertRaisesRegex(synth.CurationError, "Unverified previous"):
            synth.validate_ast(parsed, schemas)

    def test_trap_correction_is_unique_and_keeps_values(self):
        schemas = fixture_schemas()
        cases = [
            ("memory.recall", {"q": "cache"}, "query="),
            ("comm.thread", {"thread_id": "12345678"}, "id="),
            ("comm.mark_read", {"slugs": ["12345678"]}, "ids="),
            (
                "schedule.remind",
                {"content": "check", "due": "2027-01-01T00:00:00Z"},
                "at=",
            ),
            ("link", {"properties": {"basis": "review"}, "weight": 0.5}, "metadata="),
        ]
        for verb, args, expected in cases:
            with self.subTest(verb=verb):
                fixed = synth.corrected_ast(
                    {"mode": "single", "ops": [synth.op_ast(verb, args)]}, schemas
                )
                self.assertIn(expected, fixed)
        ambiguous = {
            "mode": "single",
            "ops": [synth.op_ast("memory.recall", {"q": "cache", "query": "other"})],
        }
        self.assertIsNone(synth.corrected_ast(ambiguous, schemas))


class ValidatorProtocolTests(unittest.TestCase):
    def setUp(self):
        self.validator = synth.Validator(sys.executable)
        self.completion = 'memory.recall(query="cache")'
        self.record = parser_record(
            self.completion, [synth.op_ast("memory.recall", {"query": "cache"})]
        )

    def invoke(self, records, code=0, completions=None):
        output = "".join(synth.compact(record) + "\n" for record in records)
        response = subprocess.CompletedProcess([], code, output, "")
        with patch.object(synth.subprocess, "run", return_value=response) as run:
            result = self.validator.parse(completions or [self.completion])
            self.assertEqual(
                run.call_args.args[0], [str(Path(sys.executable).resolve())]
            )
            self.assertNotIn("shell", run.call_args.kwargs)
            return result

    def test_missing_validator(self):
        with self.assertRaisesRegex(synth.CurationError, "missing"):
            synth.Validator("/definitely-absent/khive-validator")

    def test_no_output_fails_even_with_exit_zero(self):
        with self.assertRaisesRegex(synth.CurationError, "incomplete"):
            self.invoke([])

    def test_missing_marker_changed_completion_or_execution_fails(self):
        for changed in (
            {"parser": "fake"},
            {"executed": True},
            {"completion": self.completion + " "},
            {"completion_parsed_unchanged": False},
            {"parser_roundtrip_equal": False},
            {"parser_source_sha256": None},
            {"line": 2},
        ):
            with self.subTest(changed=changed), self.assertRaises(synth.CurationError):
                self.invoke([{**self.record, **changed}])

    def test_misordered_output_fails(self):
        with self.assertRaises(synth.CurationError):
            self.invoke(
                [{**self.record, "line": 2}, self.record],
                completions=[self.completion] * 2,
            )

    def test_wrong_ast_cannot_certify_a_synthetic_row(self):
        schemas = fixture_schemas()
        example = synth.Example(
            "Recall cache",
            self.completion,
            ["memory.recall"],
            "memory",
            "train",
            "test",
            "single",
            [synth.op_ast("memory.recall", {"query": "cache"})],
        )
        malicious = {
            **self.record,
            "ops": [synth.op_ast("memory.recall", {"query": "different"})],
        }
        with (
            patch.object(self.validator, "parse", return_value=[malicious]),
            self.assertRaisesRegex(synth.CurationError, "differs"),
        ):
            synth.validate_examples(
                [example], self.validator, schemas, {"memory.recall": "train"}
            )

    def test_malformed_real_json_fails_without_adding_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "real.jsonl"
            path.write_text('{"ops":"get()","ok":"yes"}\n')
            with self.assertRaisesRegex(synth.CurationError, "Malformed real row"):
                synth.merge_real(path, [], self.validator, fixture_schemas(), {})


class RealParserTests(unittest.TestCase):
    def setUp(self):
        if not VALIDATOR:
            self.skipTest("Pass --validator to exercise the real parser")
        self.validator = synth.Validator(VALIDATOR)
        self.validator.probe()

    def test_unicode_quotes_and_literal_prev_roundtrip(self):
        text = 'Snow 雪, café, "quoted" text\nC:\\notes and literal $prev.id'
        completion = synth.call("memory.recall", {"query": text})
        result = self.validator.parse([completion])[0]
        self.assertEqual(
            result["ops"], [synth.op_ast("memory.recall", {"query": text})]
        )
        self.assertEqual(result["completion"], completion)

    def test_leading_reserved_literal_prefixes_and_nested_values(self):
        values = [
            "$prev",
            "$prev.id",
            "$prev[0].id",
            "$prev.not valid",
            r"\\$prev.id",
            r"C:\notes",
            "$previous",
            ["$prev.id", {"$prev.key": "$prev[0]"}],
            {"literal": "$prev.id", "nested": ["$prev", r"\\$prev.id"]},
        ]
        completions = [
            synth.call("create", {"properties": {"value": value}}) for value in values
        ]
        results = self.validator.parse(completions)
        for value, result in zip(values, results, strict=True):
            with self.subTest(value=value):
                self.assertEqual(
                    result["ops"],
                    [synth.op_ast("create", {"properties": {"value": value}})],
                )

    def test_unrepresentable_single_backslash_literal_is_rejected(self):
        for value in (r"\$prev.id", [r"\$prev"], {"nested": r"\$prev[0]"}):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(synth.CurationError, "cannot be preserved"),
            ):
                synth.render_value(value)

    def test_real_merge_skips_unknown_ambiguous_and_leaking_rows(self):
        schemas = fixture_schemas()
        splits = synth.split_verbs(schemas)
        uid = synth.fixture(0)["id"]
        rows = [
            {"ops": 'memory.recall(q="cache")', "ok": False, "error": "unknown q"},
            {"ops": 'memory.recall(query="storage")', "ok": True, "error": None},
            {"ops": 'memory.recall(query="index")', "ok": False, "error": "timeout"},
            {"ops": 'exec.run(cmd="unsafe")', "ok": True},
            {"ops": 'memory.recall(q="cache",query="other")', "ok": False},
            {
                "ops": 'memory.recall(query="bad")',
                "ok": False,
                "corrected_ops": 'schedule.remind(content="check",at="2027-01-01T00:00:00Z")',
            },
            {"ops": f'[get(id="{uid}"),schedule.agenda()]', "ok": True},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "real.jsonl"
            path.write_text("".join(synth.compact(row) + "\n" for row in rows))
            accepted, skips = synth.merge_real(
                path, [], self.validator, schemas, splits
            )
        self.assertEqual(len(accepted), 2)
        self.assertEqual(sum(skips.values()), 5)
        self.assertEqual(accepted[0].completion, 'memory.recall(query="cache")')
        synth.validate_examples(accepted, self.validator, schemas, splits)

    def test_complete_capture_deterministic_generation_and_write_readback(self):
        if not SCHEMAS:
            self.skipTest("Pass --schemas to test complete generation")
        schemas, hashes = synth.read_schemas(SCHEMAS)
        splits = synth.split_verbs(schemas)
        pools, drops = synth.candidates(schemas, splits)
        first, selection_drops = synth.select_candidates(pools, splits)
        second, _ = synth.select_candidates(pools, splits)
        self.assertEqual([e.row() for e in first], [e.row() for e in second])
        self.assertTrue(3000 <= len(first) <= 6000)
        self.assertTrue(all(synth.bounded(e) for e in first))
        self.assertLessEqual(max(Counter(e.completion for e in first).values()), 3)
        self.assertTrue(
            any(e.mode == "chain" and "$prev." in e.completion for e in first)
        )
        self.assertTrue(any(len(set(e.verbs)) > 1 for e in first))
        for example in first:
            synth.add_split_guard(example, splits)
        synth.validate_examples(first, self.validator, schemas, splits)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "dataset"
            counts = synth.write_dataset(
                out,
                first,
                self.validator,
                schemas,
                hashes,
                splits,
                drops + selection_drops,
                {},
                None,
            )
            for split in synth.SPLITS:
                rows = [
                    json.loads(line)
                    for line in (out / (split + ".jsonl")).read_text().splitlines()
                ]
                sidecars = [
                    json.loads(line)
                    for line in (out / (split + ".provenance.jsonl"))
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(len(rows), counts[split])
                self.assertEqual(len(rows), len(sidecars))
                self.assertTrue(
                    all(all(splits[v] == split for v in p["verbs"]) for p in sidecars)
                )
            report = (out / "CURATION.md").read_text()
            self.assertIn("NOT tokenizer validation", report)
            self.assertIn(f"Rows: {len(first)}", report)

    def test_failed_readback_preserves_previous_dataset(self):
        schemas = fixture_schemas()
        splits = {"memory.recall": "train"}
        completion = 'memory.recall(query="cache")'
        example = synth.Example(
            "Recall cache",
            completion,
            ["memory.recall"],
            "memory",
            "train",
            "fixture",
            "single",
            [synth.op_ast("memory.recall", {"query": "cache"})],
        )
        synth.validate_examples([example], self.validator, schemas, splits)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "dataset"
            out.mkdir()
            (out / "CURATION.md").write_text("previous evidence")
            with (
                patch.object(
                    self.validator,
                    "parse",
                    side_effect=synth.CurationError("readback refused"),
                ),
                self.assertRaisesRegex(synth.CurationError, "readback refused"),
            ):
                synth.write_dataset(
                    out, [example], self.validator, schemas, {}, splits, {}, {}, None
                )
            self.assertEqual((out / "CURATION.md").read_text(), "previous evidence")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--validator")
    parser.add_argument("--schemas")
    options, rest = parser.parse_known_args()
    VALIDATOR, SCHEMAS = options.validator, options.schemas
    unittest.main(argv=[sys.argv[0], *rest])
