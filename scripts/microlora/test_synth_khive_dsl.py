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
        self.assertEqual(
            {v for v, split in splits.items() if split == "valid"},
            {"pack.v8", "pack.v4"},
        )

    def test_composition_leakage_is_rejected_even_when_first_verb_is_train(self):
        item = synth.Example(
            "Do both", "[a(),b()]", ["a", "b"], "p", "train", "parallel", "parallel", []
        )
        with self.assertRaisesRegex(synth.CurationError, "crosses"):
            synth.add_split_guard(item, {"a": "train", "b": "test"})
        item.verbs = ["a", "a"]
        synth.add_split_guard(item, {"a": "train"})
        item.verbs = []
        with self.assertRaisesRegex(synth.CurationError, "Unknown verb"):
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
        selected, drops = synth.select_candidates(
            {"kg": pool}, {"get": "train"}, per_pack=2
        )
        self.assertEqual(selected, pool[:2])
        self.assertEqual(dict(drops), {})

    def test_schema_names_required_fields_types_and_null_are_checked(self):
        schemas = fixture_schemas()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "_verbs.json").write_text(
                synth.compact({"total": 1, "verbs": [{"verb": "get", "pack": "kg"}]})
            )
            capture = root / "get.json"
            capture.write_text(synth.compact(schemas["get"]))
            self.assertEqual(synth.read_schemas(root)[0], {"get": schemas["get"]})
            for changed in ({"required": "yes"}, {"type": 1}):
                invalid = {
                    **schemas["get"],
                    "params": [{**schemas["get"]["params"][0], **changed}],
                }
                capture.write_text(synth.compact(invalid))
                with (
                    self.subTest(changed=changed),
                    self.assertRaisesRegex(
                        synth.CurationError, "Malformed parameter schema"
                    ),
                ):
                    synth.read_schemas(root)
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
        self.assertTrue(synth.value_matches(None, "string|null"))
        self.assertTrue(synth.value_matches("a" * 40, "string|null"))
        self.assertFalse(synth.value_matches(1, "string|null"))
        self.assertTrue(synth.value_matches([{"a": 1}], "object or array of object"))
        self.assertTrue(synth.value_matches({"a": 1}, "object or array of object"))
        self.assertFalse(synth.value_matches("x", "object or array of object"))
        self.assertTrue(synth.value_matches({"k": [1]}, "JSON value"))
        self.assertFalse(synth.value_matches([1], "array of string"))
        self.assertFalse(synth.value_matches(True, "float"))
        self.assertFalse(synth.value_matches(float("inf"), "number"))
        self.assertTrue(
            synth.value_matches("12345678-1234-5678-9234-567812345678", "uuid")
        )
        self.assertTrue(synth.value_matches("1234abcd", "uuid"))
        self.assertFalse(synth.value_matches("not-a-uuid", "uuid"))
        self.assertFalse(synth.value_matches(12345678, "uuid"))
        self.assertTrue(synth.value_matches([], "array"))
        self.assertFalse(synth.value_matches({}, "array"))
        for typename in ("bool", "boolean"):
            self.assertTrue(synth.value_matches(False, typename))
            self.assertFalse(synth.value_matches(0, typename))
        self.assertTrue(synth.value_matches(1, "integer"))
        self.assertFalse(synth.value_matches(1.5, "integer"))
        with self.assertRaisesRegex(synth.CurationError, "Unimplemented schema type"):
            synth.value_matches("value", "unrecognized")

    def test_registry_refusal_names_every_untemplated_verb(self):
        schemas = fixture_schemas()
        self.assertEqual(synth.untemplated_verbs(schemas), [])
        # Names outside every excluded family, so the refusal is about missing templates.
        schemas["ledger.post"] = schema(
            "ledger.post", "ledger", [("entry", "string", True)]
        )
        schemas["ledger.void"] = schema("ledger.void", "ledger", [("id", "uuid", True)])
        self.assertNotIn("ledger.post", synth.EXCLUSIONS)
        with self.assertRaisesRegex(
            synth.CurationError, r"2 registered verb\(s\): ledger\.post, ledger\.void"
        ):
            synth.candidates(schemas, synth.split_verbs(schemas))

    def test_missing_schema_capture_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "_verbs.json").write_text(
                synth.compact({"total": 1, "verbs": [{"verb": "get", "pack": "kg"}]})
            )
            with self.assertRaisesRegex(synth.CurationError, "complete schema"):
                synth.read_schemas(root)
            (root / "get.json").write_text(synth.compact(fixture_schemas()["get"]))
            for total, entries in (
                (2, [{"verb": "get", "pack": "kg"}]),
                (2, [{"verb": "get", "pack": "kg"}] * 2),
            ):
                (root / "_verbs.json").write_text(
                    synth.compact({"total": total, "verbs": entries})
                )
                with (
                    self.subTest(total=total, entries=entries),
                    self.assertRaisesRegex(
                        synth.CurationError, "Registry count or duplicate"
                    ),
                ):
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
            out = root / ".khive" / "data"
            out.mkdir(parents=True)
            (out / "unrelated.txt").write_text("keep")
            with self.assertRaisesRegex(synth.CurationError, "unrelated files"):
                synth.safe_output(out)
            self.assertEqual((out / "unrelated.txt").read_text(), "keep")

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
        schemas["comm.probe"] = schema(
            "comm.probe", "comm", [("since_us", "integer", False)]
        )
        parsed["ops"] = [
            synth.op_ast("comm.probe", {}),
            synth.op_ast("comm.probe", {"since_us": synth.Prev("cursor_us")}),
        ]
        synth.validate_ast(parsed, schemas)
        for path in ("id", "cursor", "cursor_us.id", "results"):
            parsed["ops"][1] = synth.op_ast(
                "comm.probe", {"since_us": synth.Prev(path)}
            )
            with (
                self.subTest(path=path),
                self.assertRaisesRegex(synth.CurationError, "Unverified previous"),
            ):
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
                aliases = {
                    "q": "query",
                    "thread_id": "id",
                    "slugs": "ids",
                    "due": "at",
                    "properties": "metadata",
                }
                self.assertEqual(
                    fixed,
                    synth.call(verb, {aliases.get(k, k): v for k, v in args.items()}),
                )
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

    def invoke(self, records, code=0, completions=None, allow_invalid=False):
        output = "".join(synth.compact(record) + "\n" for record in records)
        response = subprocess.CompletedProcess([], code, output, "")
        with patch.object(synth.subprocess, "run", return_value=response) as run:
            result = self.validator.parse(
                completions or [self.completion], allow_invalid
            )
            self.assertEqual(
                run.call_args.args[0], [str(Path(sys.executable).resolve())]
            )
            self.assertNotIn("shell", run.call_args.kwargs)
            return result

    def test_missing_validator(self):
        with self.assertRaisesRegex(synth.CurationError, "missing"):
            synth.Validator("/definitely-absent/khive-validator")
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "validator"
            binary.write_text("not executable")
            binary.chmod(0o600)
            with self.assertRaisesRegex(synth.CurationError, "not executable"):
                synth.Validator(binary)

    def test_no_output_fails_even_with_exit_zero(self):
        with self.assertRaisesRegex(synth.CurationError, "incomplete"):
            self.invoke([])
        with self.assertRaisesRegex(synth.CurationError, "Validator exit 2:"):
            self.invoke([self.record], code=2)

    def test_missing_marker_changed_completion_or_execution_fails(self):
        for changed in (
            {"parser": "fake"},
            {"executed": True},
            {"completion": self.completion + " "},
            {"completion_parsed_unchanged": False},
            {"parser_roundtrip_equal": False},
            {"ast_json_roundtrip_equal": False},
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
        failure = {**self.record, "ok": False, "error": "invalid syntax"}
        self.assertEqual(self.invoke([failure], code=1, allow_invalid=True), [failure])
        for error in (None, 7, {}, []):
            with (
                self.subTest(error=error),
                self.assertRaisesRegex(
                    synth.CurationError, "Invalid validator failure record"
                ),
            ):
                self.invoke([{**failure, "error": error}], code=1, allow_invalid=True)

    def test_wrong_ast_cannot_certify_a_synthetic_row(self):
        schemas = fixture_schemas()
        generated = synth.make_single(
            "memory.recall", 0, schemas, {"memory.recall": "train"}
        )
        self.assertEqual(
            generated.prompt,
            'Find up to 3 memories about "tokenizer regression tests".\nReturn only the request ops string.',
        )
        self.assertEqual(
            generated.completion,
            'memory.recall(query="tokenizer regression tests",limit=3)',
        )
        self.assertEqual(
            generated.ops,
            [
                synth.op_ast(
                    "memory.recall", {"query": "tokenizer regression tests", "limit": 3}
                )
            ],
        )
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

    def test_real_merge_skips_rows_carrying_addresses_or_credentials_whole(self):
        schemas = fixture_schemas()
        splits = synth.split_verbs(schemas)
        rows = [
            {"ops": 'memory.recall(query="mail bob.smith@example.org")', "ok": True},
            {
                "ops": 'memory.recall(q="cache")',
                "ok": False,
                "error": "refused: token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ab",
                "corrected_ops": 'memory.recall(query="cache")',
            },
            {"ops": 'memory.recall(query="storage")', "ok": True},
        ]

        def parse(completions, allow_invalid=False):
            records = []
            for line, completion in enumerate(completions, 1):
                query = completion.split('="', 1)[1].rsplit('"', 1)[0]
                key = "q" if completion.startswith("memory.recall(q=") else "query"
                records.append(
                    parser_record(
                        completion,
                        [synth.op_ast("memory.recall", {key: query})],
                        line=line,
                    )
                )
            return records

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "real.jsonl"
            path.write_text("".join(synth.compact(row) + "\n" for row in rows))
            with patch.object(self.validator, "parse", side_effect=parse):
                accepted, skips = synth.merge_real(
                    path, [], self.validator, schemas, splits
                )
                self.assertEqual(
                    [example.completion for example in accepted],
                    ['memory.recall(query="storage")'],
                )
                self.assertEqual(
                    dict(skips), {"screened_email": 1, "screened_secret_pattern": 1}
                )
                # Mutation arm: with the screen disabled the same fixture emits all three.
                with patch.object(synth, "sensitive_reason", return_value=None):
                    accepted, skips = synth.merge_real(
                        path, [], self.validator, schemas, splits
                    )
                self.assertEqual(len(accepted), 3)
                self.assertNotIn("screened_email", skips)
                leaked = [
                    e for e in accepted if "bob.smith@" in e.prompt + e.completion
                ]
                self.assertEqual(len(leaked), 1)

    def test_malformed_real_json_fails_without_adding_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "real.jsonl"
            path.write_text('{"ops":"get()","ok":"yes"}\n')
            with self.assertRaisesRegex(synth.CurationError, "Malformed real row"):
                synth.merge_real(path, [], self.validator, fixture_schemas(), {})

    def test_literal_rejection_without_external_validator(self):
        RealParserTests.test_unrepresentable_single_backslash_literal_is_rejected(self)

    def test_dataset_rollback_without_external_validator(self):
        def parse(completions):
            return [
                parser_record(
                    completion,
                    [synth.op_ast("memory.recall", {"query": "cache"})],
                    line=line,
                )
                for line, completion in enumerate(completions, 1)
            ]

        with patch.object(self.validator, "parse", side_effect=parse):
            RealParserTests.test_failed_readback_preserves_previous_dataset(self)

    def test_real_merge_filtering_without_external_validator(self):
        token = "ghp_" + "A" * 36
        operations = [
            ("memory.recall", {"q": "cache"}),
            ("memory.recall", {"q": token}),
            ("memory.recall", {"q": "cache", "query": "other"}),
            ("exec.run", {"cmd": "unsafe"}),
            ("schedule.remind", {"content": "check", "at": "2027-01-01T00:00:00Z"}),
            *[
                ("memory.recall", {"query": query})
                for query in ("cache", "storage", "index", "bad")
            ],
        ]
        records = {
            synth.call(verb, args): parser_record(
                synth.call(verb, args), [synth.op_ast(verb, args)]
            )
            for verb, args in operations
        }
        completion = f'[get(id="{synth.fixture(0)["id"]}"),schedule.agenda()]'
        records[completion] = parser_record(
            completion,
            [
                synth.op_ast("get", {"id": synth.fixture(0)["id"]}),
                synth.op_ast("schedule.agenda", {}),
            ],
            mode="parallel",
        )

        def parse(completions, allow_invalid=False):
            return [
                {**records[completion], "line": line}
                for line, completion in enumerate(completions, 1)
            ]

        with patch.object(self.validator, "parse", side_effect=parse):
            RealParserTests.test_real_merge_skips_unknown_ambiguous_and_partition_crossing_rows(
                self
            )


class DatasetReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.out = self.root / "dataset"
        self.schemas = fixture_schemas()
        self.splits = synth.split_verbs(self.schemas)
        self.validator = synth.Validator(sys.executable)
        self.example = synth.Example(
            "Recall cache", 'memory.recall(query="cache")', ["memory.recall"],
            "memory", "train", "fixture", "single",
            [synth.op_ast("memory.recall", {"query": "cache"})],
        )

        def parse(completions):
            return [
                parser_record(completion, self.example.ops, line=line)
                for line, completion in enumerate(completions, 1)
            ]

        parser = patch.object(self.validator, "parse", side_effect=parse)
        parser.start()
        self.addCleanup(parser.stop)
        self.write()
        self.previous = self.files(self.out)
        self.example.prompt = "Recall stored cache"

    def write(self):
        synth.write_dataset(
            self.out, [self.example], self.validator, self.schemas, {},
            self.splits, {}, {}, None,
        )

    def files(self, directory):
        return {path.name: path.read_bytes() for path in directory.iterdir()}

    def test_failed_replacement_and_restore_preserves_backup(self):
        rename = Path.rename
        backups = []

        def fail_replacement_and_restore(source, destination):
            if source == self.out:
                backups.append(Path(destination))
                return rename(source, destination)
            raise OSError("replacement or restore refused")

        with patch.object(Path, "rename", new=fail_replacement_and_restore):
            with self.assertRaises((OSError, synth.CurationError)) as raised:
                self.write()
        self.assertEqual(len(backups), 1)
        backup = backups[0]
        self.assertTrue(backup.is_dir(), "Previous dataset was deleted")
        self.assertEqual(self.files(backup), self.previous)
        self.assertEqual(backup.parent, self.root)
        self.assertIsInstance(raised.exception, synth.CurationError)
        self.assertIn(str(backup), str(raised.exception))
        self.assertEqual(set(self.root.iterdir()), {backup})

    def test_successful_replacement_removes_backup(self):
        self.write()
        self.assertNotEqual(self.files(self.out), self.previous)
        self.assertIn(b"Recall stored cache", (self.out / "train.jsonl").read_bytes())
        self.assertEqual(set(self.root.iterdir()), {self.out})

    def test_failed_replacement_restores_previous_dataset(self):
        rename = Path.rename

        def fail_replacement(source, destination):
            if source.name == "dataset" and source != self.out:
                raise OSError("replacement refused")
            return rename(source, destination)

        with patch.object(Path, "rename", new=fail_replacement):
            with self.assertRaisesRegex(OSError, "replacement refused"):
                self.write()
        self.assertEqual(self.files(self.out), self.previous)
        self.assertEqual(set(self.root.iterdir()), {self.out})


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
        self.assertEqual(result["canonical_format"], "json_request")
        self.assertEqual(
            json.loads(result["canonical_request"]),
            {"tool": "memory.recall", "args": {"query": text}},
        )

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
                for marker in (
                    "ok",
                    "completion_parsed_unchanged",
                    "ast_json_roundtrip_equal",
                    "parser_roundtrip_equal",
                ):
                    self.assertIs(result[marker], True)
                self.assertIs(result["executed"], False)
                self.assertEqual(result["parser"], synth.PARSER)
                self.assertEqual(
                    result["parser_source_sha256"], self.validator.source_hash
                )
                self.assertEqual(result["line"], values.index(value) + 1)
                self.assertEqual(
                    result["completion"],
                    synth.call("create", {"properties": {"value": value}}),
                )

    def test_unrepresentable_single_backslash_literal_is_rejected(self):
        for value in (
            r"\$prev.id",
            [r"\$prev"],
            {"nested": r"\$prev[0]"},
            r"\$prev[1]",
            {"nested": [r"\$prev[1].id"]},
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(synth.CurationError, "cannot be preserved"),
            ):
                synth.render_value(value)

    def test_real_merge_skips_unknown_ambiguous_and_partition_crossing_rows(self):
        schemas = fixture_schemas()
        splits = synth.split_verbs(schemas)
        uid = synth.fixture(0)["id"]
        token = "ghp_" + "A" * 36
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
            {"ops": 'memory.recall(q="cache")', "ok": False, "error": token},
            {
                "ops": synth.call("memory.recall", {"q": token}),
                "ok": False,
                "corrected_ops": 'memory.recall(query="cache")',
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "real.jsonl"
            path.write_text("".join(synth.compact(row) + "\n" for row in rows))
            accepted, skips = synth.merge_real(
                path, [], self.validator, schemas, splits
            )
        self.assertEqual(len(accepted), 2)
        self.assertEqual(sum(skips.values()), 7)
        self.assertEqual(skips["screened_secret_pattern"], 2)
        self.assertTrue(
            all(token not in item.prompt + item.completion for item in accepted)
        )
        self.assertEqual(accepted[0].completion, 'memory.recall(query="cache")')
        synth.validate_examples(accepted, self.validator, schemas, splits)

    def test_complete_capture_is_covered_and_every_template_validates(self):
        if not SCHEMAS:
            self.skipTest("Pass --schemas to check the capture against the templates")
        schemas, _ = synth.read_schemas(SCHEMAS)
        splits = synth.split_verbs(schemas)
        self.assertEqual(synth.untemplated_verbs(schemas), [])
        failures = {}
        templated = 0
        for verb in sorted(schemas):
            if verb in synth.EXCLUSIONS:
                continue
            templated += 1
            for index in (0, 1, 383):
                item = synth.make_single(verb, index, schemas, splits)
                try:
                    synth.validate_ast(
                        {"ok": True, "mode": item.mode, "ops": item.ops}, schemas
                    )
                except synth.CurationError as error:
                    failures[verb] = str(error)
        self.assertEqual(failures, {})
        print(
            f"capture: {len(schemas)} registered verbs, {templated} templated, "
            f"{len(schemas) - templated} excluded, 0 template/schema mismatches"
        )

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
        splits = synth.split_verbs(schemas)
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
            rename = Path.rename
            calls = []

            def fail_stage(source, destination):
                calls.append((source.name, Path(destination).name))
                if source.name == "dataset" and source != out.resolve():
                    raise OSError("dataset replacement refused")
                return rename(source, destination)

            with (
                patch.object(Path, "rename", new=fail_stage),
                self.assertRaisesRegex(OSError, "dataset replacement refused"),
            ):
                synth.write_dataset(
                    out, [example], self.validator, schemas, {}, splits, {}, {}, None
                )
            backup_name = calls[0][1]
            self.assertTrue(backup_name.startswith("dataset.previous-"))
            self.assertEqual(
                calls,
                [
                    ("dataset", backup_name),
                    ("dataset", "dataset"),
                    (backup_name, "dataset"),
                ],
            )
            self.assertTrue(out.is_dir(), "Previous dataset directory was not restored")
            self.assertEqual((out / "CURATION.md").read_text(), "previous evidence")
            self.assertEqual({p.name for p in out.iterdir()}, {"CURATION.md"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--validator")
    parser.add_argument("--schemas")
    options, rest = parser.parse_known_args()
    VALIDATOR, SCHEMAS = options.validator, options.schemas
    unittest.main(argv=[sys.argv[0], *rest])
