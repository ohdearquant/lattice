#!/usr/bin/env python3
"""Contract tests for the ADR-102 D4 file-size gate.

Each gate case runs a real change through `evaluate` in a throwaway repository
and asserts the specific failure text, not just a nonzero exit. A gate that
refuses for an unrelated reason, or that lumps every violation behind one
message, would pass an exit-code-only test and fail these.

"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "file_size_gate.py"
MANIFEST = "file-size-manifest.toml"

SPEC = importlib.util.spec_from_file_location("file_size_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

EXIT_OK = 0
EXIT_VIOLATION = 1
EXIT_REFUSAL = 2

GIT_ENV_OVERRIDES = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
}
IDENTITY = ("-c", "user.name=fx", "-c", "user.email=fx@example.invalid")


def block(lines: int, *, crlf: bool = False, tail: bytes = b"") -> bytes:
    unit = b"y\r\n" if crlf else b"y\n"
    return unit * lines + tail


class Scratch:
    """A throwaway git repository, one commit per `commit()` call."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q", "-b", "main")

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        env = dict(gate._child_env())
        env.update(GIT_ENV_OVERRIDES)
        proc = subprocess.run(
            ["git", "-C", str(self.root), "-c", "core.hooksPath=", *IDENTITY, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        if check and proc.returncode != 0:
            raise AssertionError(
                "scratch git %s failed: %s" % (" ".join(args), proc.stderr.decode())
            )
        return proc

    def put(self, relative: str, payload: object) -> None:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, bytes):
            target.write_bytes(payload)
        else:
            target.write_text(str(payload), encoding="utf-8")

    def drop(self, relative: str) -> None:
        (self.root / relative).unlink()

    def shift(self, source: str, target: str) -> None:
        destination = self.root / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        (self.root / source).rename(destination)

    def commit(self, message: str = "fixture") -> str:
        self._git("add", "-A")
        self._git("commit", "-q", "--allow-empty", "-m", message)
        return self._git("rev-parse", "HEAD").stdout.decode().strip()


def manifest(ratchets=(), data=()) -> str:
    lines: list[str] = []
    for item in ratchets:
        path, ceiling = item[0], item[1]
        retired_by = item[2] if len(item) > 2 else None
        renamed_from = item[3] if len(item) > 3 else None
        lines.append("[[ratchet]]")
        lines.append('path = "%s"' % path)
        lines.append("ceiling = %d" % ceiling)
        if renamed_from:
            lines.append('renamed_from = "%s"' % renamed_from)
        if retired_by:
            lines.append('retired_by = "%s"' % retired_by)
        lines.append("")
    for path in data:
        lines.append("[[data]]")
        lines.append('path = "%s"' % path)
        lines.append("")
    return "\n".join(lines) + "\n"


SOURCE = "crates/demo/src/heavy.rs"
ASSET = "assets/tokenizer.json"


class GateCase(unittest.TestCase):
    """Base that hands each test its own disposable repository."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="file-size-gate-test-")
        self.addCleanup(self._tmp.cleanup)
        self.box = Scratch(Path(self._tmp.name) / "repo")

    def verdict(self, base: str, head: str):
        return gate.evaluate(self.box.root, base, head)

    def assert_fails_with(self, verdict, needle: str) -> None:
        self.assertEqual(
            verdict.exit_code,
            EXIT_VIOLATION,
            "expected a violation, got exit %d with %s" % (verdict.exit_code, verdict.failures),
        )
        joined = "\n".join(verdict.failures)
        self.assertIn(needle, joined)
        for failure in verdict.failures:
            self.assertTrue(failure.startswith("FAIL "), failure)

    def assert_clean(self, verdict) -> None:
        self.assertEqual(
            verdict.exit_code,
            EXIT_OK,
            "expected clean, got exit %d with %s" % (verdict.exit_code, verdict.failures),
        )
        self.assertEqual(verdict.failures, [])


class PureHelpers(unittest.TestCase):
    def test_line_count_counts_lf_and_final_unterminated_line(self) -> None:
        self.assertEqual(gate.line_count(b""), 0)
        self.assertEqual(gate.line_count(b"a\n"), 1)
        self.assertEqual(gate.line_count(b"a"), 1)
        self.assertEqual(gate.line_count(b"a\nb"), 2)
        self.assertEqual(gate.line_count(b"a\r\n"), 1)

    def test_line_count_at_the_boundary(self) -> None:
        self.assertEqual(gate.line_count(block(3000)), 3000)
        self.assertEqual(gate.line_count(block(2999, tail=b"last")), 3000)
        self.assertEqual(gate.line_count(block(3000, tail=b"last")), 3001)

    def test_is_binary_probes_only_the_first_window(self) -> None:
        self.assertFalse(gate.is_binary(b"text"))
        self.assertTrue(gate.is_binary(b"\x00"))
        self.assertTrue(gate.is_binary(b"x" * 8189 + b"\x00"))
        self.assertFalse(gate.is_binary(b"x" * 8192 + b"\x00"))

    def test_is_maintained_by_extension_and_workflow_path(self) -> None:
        for path in (
            "a.rs",
            "deep/dir/b.metal",
            "scripts/c.py",
            "scripts/d.sh",
            "scripts/e.bash",
            "docs/f.md",
            ".github/workflows/ci.yml",
            ".github/workflows/ci.yaml",
        ):
            self.assertTrue(gate.is_maintained(path), path)
        for path in (
            "Cargo.lock",
            "assets/tokenizer.json",
            ".github/workflows/nested/ci.yml",
            "docs/notes.txt",
            "Makefile",
            "notes.MD",
        ):
            self.assertFalse(gate.is_maintained(path), path)

    def test_parse_manifest_reads_both_tables(self) -> None:
        parsed = gate.parse_manifest(
            manifest(ratchets=((SOURCE, 4000, "ADR-102 M8"),), data=("Cargo.lock",)),
            "fixture",
        )
        self.assertEqual(parsed.ratchets[SOURCE]["ceiling"], 4000)
        self.assertEqual(parsed.ratchets[SOURCE]["retired_by"], "ADR-102 M8")
        self.assertEqual(set(parsed.data), {"Cargo.lock"})

    def test_parse_manifest_refuses_malformed_input(self) -> None:
        for text in (
            "[[ratchet]\npath = 'x'\n",
            "[[ratchet]]\npath = 'x'\nceiling = 'many'\n",
            "[[ratchet]]\npath = 'x'\nceiling = 1\nunexpected = 2\n",
            "[[data]]\npath = 'a'\n\n[[ratchet]]\npath = 'a'\nceiling = 1\n",
            "[[data]]\npath = 'a'\n\n[[data]]\npath = 'a'\n",
            "top = 1\n",
            "[[ratchet]]\nceiling = 10\n",
            "[[ratchet]]\npath = 'a'\nceiling = 1\nretired_by = 7\n",
        ):
            with self.assertRaises(gate.Refusal, msg=repr(text)):
                gate.parse_manifest(text, "fixture")

    def test_parse_manifest_accepts_an_empty_document(self) -> None:
        parsed = gate.parse_manifest("", "fixture")
        self.assertEqual(parsed.ratchets, {})
        self.assertEqual(parsed.data, {})

    def test_parse_manifest_refuses_unknown_top_level_key(self) -> None:
        with self.assertRaises(gate.Refusal):
            gate.parse_manifest("[[ratchet]]\npath = 'a'\nceiling = 1\n\n[other]\nk = 1\n", "f")


class Population(GateCase):
    def test_population_line_reports_the_scan(self) -> None:
        self.box.put(SOURCE, block(10))
        self.box.put("README.md", "# docs\n")
        base = self.box.commit("base")
        verdict = self.verdict(base, base)
        self.assertIn("population: 2 tracked text files (2 maintained)", verdict.header)
        self.assertIn("base %s, head %s" % (base, base), verdict.header)

    def test_empty_population_refuses(self) -> None:
        head = self.box.commit("empty")
        verdict = self.verdict(head, head)
        self.assertEqual(verdict.exit_code, EXIT_REFUSAL)
        self.assertIn("REFUSAL", "\n".join(verdict.failures))

    def test_binary_only_population_refuses(self) -> None:
        self.box.put("assets/model.bin", b"\x00" + b"x\n" * 5000)
        head = self.box.commit("head")
        verdict = self.verdict(head, head)
        self.assertEqual(verdict.exit_code, EXIT_REFUSAL)

    def test_base_that_is_not_a_commit_refuses(self) -> None:
        self.box.put(SOURCE, block(10))
        head = self.box.commit("head")
        verdict = gate.evaluate(self.box.root, "1" * 40, head)
        self.assertEqual(verdict.exit_code, EXIT_REFUSAL)
        self.assertIn("base revision is not a commit", "\n".join(verdict.failures))
        # A refusal still states the population: a refusal that printed none
        # would be indistinguishable from a gate that never scanned anything.
        self.assertTrue(verdict.header.startswith("population: 1 tracked text files"))

    def test_head_that_is_not_a_commit_refuses(self) -> None:
        verdict = gate.evaluate(self.box.root, gate.ZERO_SHA, "2" * 40)
        self.assertEqual(verdict.exit_code, EXIT_REFUSAL)
        self.assertIn("head revision is not a commit", "\n".join(verdict.failures))

    def test_all_zero_base_skips_the_manifest_diff_rules(self) -> None:
        self.box.put(SOURCE, block(4000))
        self.box.put(MANIFEST, manifest(ratchets=((SOURCE, 4000),)))
        head = self.box.commit("head")
        self.assert_clean(gate.evaluate(self.box.root, gate.ZERO_SHA, head))

    def test_dirty_worktree_does_not_change_the_verdict(self) -> None:
        self.box.put(SOURCE, block(4000))
        self.box.put(MANIFEST, manifest(ratchets=((SOURCE, 4000),)))
        head = self.box.commit("head")
        # A working-tree copy that would fail if it were read: the file at the
        # revision under test is fine, and the gate must not see this one.
        self.box.put(SOURCE, block(2))
        self.box.put(MANIFEST, "not toml at all")
        self.assert_clean(self.verdict(head, head))


class NewFiles(GateCase):
    def test_unlisted_maintained_file_over_cap_fails(self) -> None:
        base = self.box.commit("base")
        self.box.put(SOURCE, block(3001))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "exceeds the 3000-line cap")

    def test_unlisted_maintained_file_at_cap_passes(self) -> None:
        base = self.box.commit("base")
        self.box.put(SOURCE, block(3000))
        head = self.box.commit("head")
        self.assert_clean(self.verdict(base, head))

    def test_unlisted_maintained_file_at_cap_without_final_newline_passes(self) -> None:
        base = self.box.commit("base")
        self.box.put(SOURCE, block(2999, tail=b"last"))
        head = self.box.commit("head")
        self.assert_clean(self.verdict(base, head))

    def test_crlf_lines_count_once_each(self) -> None:
        base = self.box.commit("base")
        self.box.put(SOURCE, block(3001, crlf=True))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "3001 lines")

    def test_crlf_at_cap_passes(self) -> None:
        base = self.box.commit("base")
        self.box.put(SOURCE, block(3000, crlf=True))
        head = self.box.commit("head")
        self.assert_clean(self.verdict(base, head))

    def test_unterminated_final_line_is_counted(self) -> None:
        base = self.box.commit("base")
        self.box.put(SOURCE, block(3000, tail=b"last"))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "3001 lines")

    def test_binary_file_of_any_size_is_ignored(self) -> None:
        self.box.put("README.md", "# fixture\n")
        base = self.box.commit("base")
        self.box.put("assets/model.bin", b"\x00" + b"x\n" * 9000)
        head = self.box.commit("head")
        self.assert_clean(self.verdict(base, head))

    def test_workflow_yaml_is_maintained(self) -> None:
        base = self.box.commit("base")
        self.box.put(".github/workflows/extra.yml", block(3001))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "exceeds the 3000-line cap")

    def test_workflow_yaml_in_a_subdirectory_is_not_maintained(self) -> None:
        base = self.box.commit("base")
        self.box.put(".github/workflows/nested/extra.yml", block(3001))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "with no data entry")


class Ratchet(GateCase):
    def seed(self, lines: int, ceiling: int):
        self.box.put(SOURCE, block(lines))
        self.box.put(MANIFEST, manifest(ratchets=((SOURCE, ceiling),)))
        return self.box.commit("base")

    def test_growth_above_ceiling_fails(self) -> None:
        base = self.seed(4000, 4000)
        self.box.put(SOURCE, block(4001))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "exceeds the effective ceiling 4000")

    def test_shrink_with_lowered_ceiling_passes(self) -> None:
        base = self.seed(4000, 4000)
        self.box.put(SOURCE, block(3500))
        self.box.put(MANIFEST, manifest(ratchets=((SOURCE, 3500),)))
        head = self.box.commit("head")
        self.assert_clean(self.verdict(base, head))

    def test_shrink_without_lowering_fails(self) -> None:
        base = self.seed(4000, 4000)
        self.box.put(SOURCE, block(3500))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "lower the ceiling to 3500")

    def test_shrink_then_regrow_fails(self) -> None:
        base = self.seed(1000, 1000)
        self.box.put(SOURCE, block(4000))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "exceeds the effective ceiling 1000")

    def test_forged_ceiling_fails(self) -> None:
        base = self.seed(4000, 4000)
        self.box.put(SOURCE, block(4001))
        self.box.put(MANIFEST, manifest(ratchets=((SOURCE, 9000),)))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "ceiling raised from 4000 to 9000")

    def test_new_ratchet_entry_fails(self) -> None:
        self.box.put(SOURCE, block(3100))
        self.box.put(MANIFEST, manifest())
        base = self.box.commit("base")
        self.box.put(MANIFEST, manifest(ratchets=((SOURCE, 3100),)))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "new ratchet entry")

    def test_drop_to_cap_with_entry_removed_passes(self) -> None:
        base = self.seed(4000, 4000)
        self.box.put(SOURCE, block(3000))
        self.box.put(MANIFEST, manifest())
        head = self.box.commit("head")
        self.assert_clean(self.verdict(base, head))

    def test_drop_to_cap_with_entry_kept_fails(self) -> None:
        base = self.seed(4000, 4000)
        self.box.put(SOURCE, block(3000))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "remove this entry")

    def test_entry_removed_while_still_over_cap_fails(self) -> None:
        base = self.seed(4000, 4000)
        self.box.put(MANIFEST, manifest())
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "entry removed while the file is still 4000")

    def test_entry_removed_after_a_rename_is_not_a_removal(self) -> None:
        base = self.seed(4000, 4000)
        self.box.shift(SOURCE, "crates/demo/src/renamed.rs")
        self.box.put(
            MANIFEST,
            manifest(ratchets=(("crates/demo/src/renamed.rs", 4000, None, SOURCE),)),
        )
        head = self.box.commit("head")
        self.assert_clean(self.verdict(base, head))

    def test_entry_re_added_after_below_cap_fails(self) -> None:
        self.box.put(SOURCE, block(2999))
        self.box.put(MANIFEST, manifest())
        base = self.box.commit("base")
        self.box.put(SOURCE, block(3001))
        self.box.put(MANIFEST, manifest(ratchets=((SOURCE, 3001),)))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "cannot return")

    def test_ratchet_entry_for_a_missing_file_fails(self) -> None:
        self.box.put(SOURCE, block(4000))
        self.box.put(MANIFEST, manifest(ratchets=((SOURCE, 4000),)))
        base = self.box.commit("base")
        self.box.put("README.md", "# fixture\n")
        self.box.drop(SOURCE)
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "names no tracked text file at head")

    def test_ratchet_entry_on_a_non_maintained_path_fails(self) -> None:
        self.box.put(ASSET, block(4000))
        self.box.put(MANIFEST, manifest(ratchets=((ASSET, 4000),)))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(head, head), "use a data entry")


class Renames(GateCase):
    def seed(self):
        self.box.put(SOURCE, block(4000))
        self.box.put(MANIFEST, manifest(ratchets=((SOURCE, 4000),)))
        return self.box.commit("base")

    def test_rename_with_field_passes(self) -> None:
        base = self.seed()
        self.box.shift(SOURCE, "crates/demo/src/renamed.rs")
        self.box.put(
            MANIFEST,
            manifest(ratchets=(("crates/demo/src/renamed.rs", 4000, None, SOURCE),)),
        )
        head = self.box.commit("head")
        self.assert_clean(self.verdict(base, head))

    def test_rename_without_field_fails(self) -> None:
        base = self.seed()
        self.box.shift(SOURCE, "crates/demo/src/renamed.rs")
        self.box.put(MANIFEST, manifest(ratchets=(("crates/demo/src/renamed.rs", 4000),)))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "rename target")

    def test_rename_with_field_keeps_ceiling_discipline(self) -> None:
        base = self.seed()
        self.box.shift(SOURCE, "crates/demo/src/renamed.rs")
        self.box.put("crates/demo/src/renamed.rs", block(4200))
        self.box.put(
            MANIFEST,
            manifest(ratchets=(("crates/demo/src/renamed.rs", 4200, None, SOURCE),)),
        )
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "exceeds the effective ceiling 4000")

    def test_rename_that_drops_the_entry_still_fails(self) -> None:
        # "A rename carries its entry explicitly": dropping it while the file
        # is still over the cap must fail, even though the removal itself is
        # excused by the rename.
        base = self.seed()
        self.box.shift(SOURCE, "crates/demo/src/renamed.rs")
        self.box.put(MANIFEST, manifest())
        head = self.box.commit("head")
        self.assert_fails_with(
            self.verdict(base, head), "exceeds the 3000-line cap with no ratchet entry"
        )

    def test_stale_renamed_from_fails(self) -> None:
        base = self.seed()
        self.box.put(
            MANIFEST,
            manifest(ratchets=((SOURCE, 4000, None, "crates/demo/src/gone.rs"),)),
        )
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "renamed_from is stale")

    def test_renamed_from_naming_no_base_entry_fails(self) -> None:
        base = self.seed()
        self.box.put("crates/demo/src/moved.rs", block(4000))
        self.box.put(
            MANIFEST,
            manifest(ratchets=(("crates/demo/src/moved.rs", 4000, None, "crates/demo/src/other.rs"),)),
        )
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "renamed_from is stale")


class DataEntries(GateCase):
    def test_non_maintained_file_over_cap_without_entry_fails(self) -> None:
        base = self.box.commit("base")
        self.box.put(ASSET, block(3500))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "not a maintained file and with no data entry")

    def test_non_maintained_file_with_entry_passes(self) -> None:
        base = self.box.commit("base")
        self.box.put(ASSET, block(3500))
        self.box.put(MANIFEST, manifest(data=(ASSET,)))
        head = self.box.commit("head")
        self.assert_clean(self.verdict(base, head))

    def test_data_entry_on_maintained_path_fails(self) -> None:
        base = self.box.commit("base")
        self.box.put(SOURCE, block(3500))
        self.box.put(MANIFEST, manifest(data=(SOURCE,)))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "data entry on a maintained path")

    def test_data_entry_glob_fails(self) -> None:
        base = self.box.commit("base")
        self.box.put(ASSET, block(3500))
        self.box.put(MANIFEST, manifest(data=("assets/*.json",)))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "glob")

    def test_data_entry_directory_fails(self) -> None:
        base = self.box.commit("base")
        self.box.put(ASSET, block(3500))
        self.box.put(MANIFEST, manifest(data=("assets/",)))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "names a directory")

    def test_data_entry_with_no_such_file_fails(self) -> None:
        base = self.box.commit("base")
        self.box.put(ASSET, block(3500))
        self.box.put(MANIFEST, manifest(data=("assets/absent.json",)))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "no such tracked text file")

    def test_stale_data_entry_on_a_shrunk_file_fails(self) -> None:
        self.box.put(ASSET, block(3500))
        self.box.put(MANIFEST, manifest(data=(ASSET,)))
        base = self.box.commit("base")
        self.box.put(ASSET, block(2000))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "under the 3000-line cap")

    def test_stale_data_entry_at_exactly_the_cap_fails(self) -> None:
        self.box.put(ASSET, block(3500))
        self.box.put(MANIFEST, manifest(data=(ASSET,)))
        base = self.box.commit("base")
        self.box.put(ASSET, block(3000))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "under the 3000-line cap")

    def test_data_entry_for_a_now_binary_file_fails(self) -> None:
        self.box.put(ASSET, block(3500))
        self.box.put(MANIFEST, manifest(data=(ASSET,)))
        base = self.box.commit("base")
        self.box.put(ASSET, b"\x00" + b"x\n" * 4000)
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "the file is binary")


class RetiredBy(GateCase):
    def test_qwen35_entry_without_retired_by_fails(self) -> None:
        path = "crates/inference/src/forward/metal_qwen35.rs"
        self.box.put(path, block(4000))
        self.box.put(MANIFEST, manifest(ratchets=((path, 4000, "ADR-102 M8"),)))
        base = self.box.commit("base")
        self.box.put(MANIFEST, manifest(ratchets=((path, 4000),)))
        head = self.box.commit("head")
        self.assert_fails_with(self.verdict(base, head), "retired_by is required")

    def test_qwen35_entry_with_retired_by_passes(self) -> None:
        path = "crates/inference/src/forward/metal_qwen35.rs"
        self.box.put(path, block(4000))
        self.box.put(MANIFEST, manifest(ratchets=((path, 4000, "ADR-102 M8"),)))
        head = self.box.commit("head")
        self.assert_clean(self.verdict(head, head))

    def test_retired_by_is_optional_elsewhere(self) -> None:
        self.box.put(SOURCE, block(4000))
        self.box.put(MANIFEST, manifest(ratchets=((SOURCE, 4000),)))
        head = self.box.commit("head")
        self.assert_clean(self.verdict(head, head))

    def test_needs_retired_by_matches_the_shipped_gate(self) -> None:
        self.assertTrue(gate.needs_retired_by("crates/inference/src/forward/metal_qwen35.rs"))
        self.assertFalse(gate.needs_retired_by("crates/inference/src/forward/metal_qwen35/inner/x.rs"))
        self.assertFalse(gate.needs_retired_by("crates/inference/src/forward/metal_qwen35.rs.bak"))


class Refusals(GateCase):
    def test_unreadable_head_manifest_refuses(self) -> None:
        self.box.put(SOURCE, block(10))
        base = self.box.commit("base")
        self.box.put(MANIFEST, "[[ratchet]\npath = 3\n")
        head = self.box.commit("head")
        verdict = self.verdict(base, head)
        self.assertEqual(verdict.exit_code, EXIT_REFUSAL)
        self.assertIn("REFUSAL", "\n".join(verdict.failures))

    def test_unreadable_base_manifest_refuses(self) -> None:
        self.box.put(SOURCE, block(10))
        self.box.put(MANIFEST, "this is not toml")
        base = self.box.commit("base")
        self.box.put(MANIFEST, manifest())
        head = self.box.commit("head")
        verdict = self.verdict(base, head)
        self.assertEqual(verdict.exit_code, EXIT_REFUSAL)

    def test_manifest_absent_at_base_is_not_a_refusal(self) -> None:
        self.box.put(SOURCE, block(4000))
        base = self.box.commit("base")
        self.box.put(MANIFEST, manifest(ratchets=((SOURCE, 4000),)))
        head = self.box.commit("head")
        self.assert_clean(self.verdict(base, head))


class ShippedManifest(unittest.TestCase):
    """The manifest committed to the repository, read as the gate reads it."""

    def test_manifest_is_present_and_parses(self) -> None:
        text = (REPO / MANIFEST).read_text(encoding="utf-8")
        parsed = gate.parse_manifest(text, MANIFEST)
        self.assertTrue(parsed.ratchets, "the shipped manifest lists no ratchet entries")

    def test_manifest_entries_are_sorted(self) -> None:
        parsed = gate.parse_manifest((REPO / MANIFEST).read_text(encoding="utf-8"), MANIFEST)
        self.assertEqual(list(parsed.ratchets), sorted(parsed.ratchets))
        self.assertEqual(list(parsed.data), sorted(parsed.data))

    def test_manifest_matches_the_working_tree(self) -> None:
        parsed = gate.parse_manifest((REPO / MANIFEST).read_text(encoding="utf-8"), MANIFEST)
        tracked = set(
            subprocess.run(
                ["git", "-C", str(REPO), "ls-files"],
                stdout=subprocess.PIPE,
                check=True,
            )
            .stdout.decode()
            .splitlines()
        )
        for path in list(parsed.ratchets) + list(parsed.data):
            self.assertIn(path, tracked, "%s is listed but not tracked" % path)

    def test_manifest_is_valid_toml(self) -> None:
        tomllib.loads((REPO / MANIFEST).read_text(encoding="utf-8"))


class CommandLine(unittest.TestCase):
    def test_missing_revisions_is_a_usage_error(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=REPO,
        )
        self.assertEqual(proc.returncode, 2)

    def test_real_check_runs_against_a_known_revision(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--base",
                "HEAD",
                "--head",
                "HEAD",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=REPO,
        )
        self.assertIn(proc.returncode, (EXIT_OK, EXIT_VIOLATION), proc.stdout.decode())
        self.assertTrue(proc.stdout.decode().startswith("population: "))


if __name__ == "__main__":
    unittest.main()
