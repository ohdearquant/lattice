#!/usr/bin/env python3
"""Acceptance tests for scripts/public-api-listing.py."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "public-api-listing.py"
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "public_api_listing"
GENERATOR = FIXTURE_ROOT / "generate.py"
CRATE = "listing_baseline"
CORE_CRATE = "listing_core"


def listing(doc_dir: Path, crate: str = CRATE) -> list[str]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--crate", crate, "--doc-dir", str(doc_dir)],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"listing failed for {crate} in {doc_dir} (exit {result.returncode}): {result.stderr}"
        )
    return result.stdout.splitlines()


class PublicApiListingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="lattice-api-listing-")
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.docs = Path(cls._tmp.name) / "docs"
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(Path(cls._tmp.name) / "target")
        env["PYTHONPYCACHEPREFIX"] = str(Path(cls._tmp.name) / "pycache")
        result = subprocess.run(
            [sys.executable, str(GENERATOR), "--output-dir", str(cls.docs)],
            cwd=REPO,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise AssertionError(
                "fixture rustdoc JSON generation failed: " + result.stdout + result.stderr
            )
        cls.baseline = listing(cls.docs / "baseline")
        cls.split = listing(cls.docs / "split")
        cls.glob = listing(cls.docs / "glob")
        cls.missing = listing(cls.docs / "missing")
        cls.added = listing(cls.docs / "added")

    def test_baseline_shape_covers_every_required_kind(self) -> None:
        expected = {
            "assoc_const listing_baseline::BpeTokenizer::MAX_SEQUENCE",
            "constant listing_baseline::DEFAULT_MAX_SEQUENCE",
            "enum listing_baseline::TokenizerError",
            "field listing_baseline::BpeTokenizer::merges",
            "field listing_baseline::TokenPair::0",
            "function listing_baseline::default_tokenizer",
            "function listing_baseline::engine::run",
            "hidden listing_baseline::internal_padding",
            "macro listing_baseline::tokenizer_name",
            "method listing_baseline::BpeTokenizer::new",
            "method listing_baseline::BpeTokenizer::vocabulary_size",
            "module listing_baseline::engine",
            "static listing_baseline::DEFAULT_TOKENIZER_NAME",
            "static listing_baseline::TOKENIZER_NAME",
            "struct listing_baseline::BpeTokenizer",
            "struct listing_baseline::TokenPair",
            "trait listing_baseline::Tokenizer",
            "trait_method listing_baseline::BpeTokenizer::default",
            "trait_method listing_baseline::BpeTokenizer::encode",
            "trait_method listing_baseline::Tokenizer::decode",
            "trait_method listing_baseline::Tokenizer::encode",
            "trait_method listing_baseline::TokenizerError::fmt",
            "type_alias listing_baseline::DefaultTokenizer",
            "variant listing_baseline::TokenizerError::EmptyInput",
            "variant listing_baseline::TokenizerError::TooLong",
        }
        self.assertEqual(set(self.baseline), expected)

    def test_named_inline_reexports_are_identical(self) -> None:
        self.assertEqual(self.split, self.baseline)

    def test_glob_reexports_are_identical(self) -> None:
        self.assertEqual(self.glob, self.baseline)

    def test_missing_reexport_differs_by_exactly_that_item(self) -> None:
        self.assertEqual(
            set(self.baseline) - set(self.missing),
            {"function listing_baseline::default_tokenizer"},
        )
        self.assertEqual(set(self.missing) - set(self.baseline), set())

    def test_added_method_differs_by_exactly_that_method(self) -> None:
        self.assertEqual(
            set(self.added) - set(self.baseline),
            {"method listing_baseline::BpeTokenizer::is_empty"},
        )
        self.assertEqual(set(self.baseline) - set(self.added), set())

    def test_missing_dependency_json_fails_and_names_the_crate(self) -> None:
        incomplete = Path(self._tmp.name) / "incomplete"
        incomplete.mkdir()
        shutil.copy(self.docs / "split" / f"{CRATE}.json", incomplete / f"{CRATE}.json")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--crate", CRATE, "--doc-dir", str(incomplete)],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"missing rustdoc JSON for crate {CORE_CRATE}", result.stderr)

    def test_hyphenated_package_name_resolves_underscored_json(self) -> None:
        renamed = Path(self._tmp.name) / "renamed"
        renamed.mkdir()
        source = self.docs / "baseline" / f"{CRATE}.json"
        document = json.loads(source.read_text(encoding="utf-8"))
        document["crate_version"] = document.get("crate_version", "0.1.0")
        renamed_path = renamed / f"{CRATE}.json"
        renamed_path.write_text(json.dumps(document), encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--crate",
                "listing-baseline",
                "--doc-dir",
                str(renamed),
            ],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), self.baseline)
        renamed_path.unlink()
        underscored = renamed / f"{CRATE.replace('-', '_')}.json"
        underscored.write_text(json.dumps(document), encoding="utf-8")
        resolved = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--crate",
                "listing-baseline",
                "--doc-dir",
                str(renamed),
            ],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(resolved.returncode, 0, resolved.stderr)
        self.assertEqual(resolved.stdout.splitlines(), self.baseline)

    def test_unsupported_format_version_fails_and_names_the_version(self) -> None:
        unsupported = Path(self._tmp.name) / "unsupported"
        unsupported.mkdir()
        document = json.loads((self.docs / "baseline" / f"{CRATE}.json").read_text(encoding="utf-8"))
        document["format_version"] = 999
        (unsupported / f"{CRATE}.json").write_text(json.dumps(document), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--crate", CRATE, "--doc-dir", str(unsupported)],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported rustdoc JSON format_version 999", result.stderr)

    def test_module_reexport_is_followed(self) -> None:
        self.assertIn("module listing_baseline::engine", self.split)
        self.assertIn("function listing_baseline::engine::run", self.split)

    def test_empty_listing_fails_closed(self) -> None:
        empty = Path(self._tmp.name) / "empty"
        empty.mkdir()
        document = json.loads((self.docs / "baseline" / f"{CRATE}.json").read_text(encoding="utf-8"))
        document["index"][str(document["root"])]["inner"]["module"]["items"] = []
        (empty / f"{CRATE}.json").write_text(json.dumps(document), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--crate", CRATE, "--doc-dir", str(empty)],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("public listing is empty", result.stderr)

    def test_missing_root_crate_fails_and_names_the_crate(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--crate",
                CRATE,
                "--doc-dir",
                str(Path(self._tmp.name) / "absent"),
            ],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"missing rustdoc JSON for crate {CRATE}", result.stderr)

    def test_malformed_json_fails_closed(self) -> None:
        malformed = Path(self._tmp.name) / "malformed"
        malformed.mkdir()
        (malformed / f"{CRATE}.json").write_text("{not json", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--crate",
                CRATE,
                "--doc-dir",
                str(malformed),
            ],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot read rustdoc JSON", result.stderr)

    def test_unresolved_reexport_fails_closed(self) -> None:
        broken = Path(self._tmp.name) / "unresolved"
        broken.mkdir()
        document = json.loads((self.docs / "split" / f"{CRATE}.json").read_text(encoding="utf-8"))
        root_id = document["root"]
        use_id = next(
            item_id
            for item_id in document["index"][str(root_id)]["inner"]["module"]["items"]
            if "use" in document["index"][str(item_id)]["inner"]
        )
        document["index"][str(use_id)]["inner"]["use"]["id"] = 2_000_000
        (broken / f"{CRATE}.json").write_text(json.dumps(document), encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--crate",
                CRATE,
                "--doc-dir",
                str(broken),
            ],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("targets unknown item", result.stderr)

    def test_unknown_item_kind_fails_closed(self) -> None:
        unknown = Path(self._tmp.name) / "unknown"
        unknown.mkdir()
        document = json.loads((self.docs / "baseline" / f"{CRATE}.json").read_text(encoding="utf-8"))
        root_id = document["root"]
        first = document["index"][str(root_id)]["inner"]["module"]["items"][0]
        document["index"][str(first)]["inner"] = {"future_kind": {}}
        (unknown / f"{CRATE}.json").write_text(json.dumps(document), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--crate", CRATE, "--doc-dir", str(unknown)],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown item kind", result.stderr)

    def test_hidden_items_are_listed_separately(self) -> None:
        self.assertIn("hidden listing_baseline::internal_padding", self.baseline)
        self.assertNotIn("function listing_baseline::internal_padding", self.baseline)


if __name__ == "__main__":
    unittest.main()
