#!/usr/bin/env python3
"""Contract tests for scripts/crate-readme.py.

The fixture is a throwaway git repository holding one small crate, an ADR, a
`bench-update.yml` and a `perf-baselines` branch, with a hand-written
`cargo metadata` document so the tests need no Rust toolchain. Each case
asserts the generated text or the refusal message, not just the exit code.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "crate-readme.py"

SPEC = importlib.util.spec_from_file_location("crate_readme", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
readme = importlib.util.module_from_spec(SPEC)
# dataclasses resolve string annotations through sys.modules[cls.__module__].
sys.modules[SPEC.name] = readme
SPEC.loader.exec_module(readme)

GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "fx",
    "GIT_AUTHOR_EMAIL": "fx@example.invalid",
    "GIT_COMMITTER_NAME": "fx",
    "GIT_COMMITTER_EMAIL": "fx@example.invalid",
}

LIB_RS = '''\
//! The demo crate: it exists for tests, e.g. `this_one`. A second sentence.
//!
//! A later paragraph.

/// Documented.
pub fn visible() -> &'static str {
    "unsafe // not a comment"
}

pub(crate) fn hidden() {}

pub mod a;

/* block /* nested */ still comment */
#[cfg(test)]
mod tests {
    #[test]
    fn one() {
        let quote = '"';
        let raw = r#"pub fn fake() { unsafe {} }"#;
        assert!(!raw.is_empty() && quote == '"');
    }

    #[tokio::test(flavor = "current_thread")]
    async fn two() {
        unsafe { core::hint::unreachable_unchecked() };
    }
}
'''

B_RS = """\
//! Inner module for [`crate::visible`], see [the docs](https://example.invalid).
pub struct B;
"""

TEST_RS = """\
#[test]
fn integration() {}
"""

CARGO_TOML = """\
[package]
name = "lattice-demo"
version = "0.1.0"

[features]
default = ["fast"]
# Faster paths. They cost a dependency.
fast = [
    "dep:lattice-leaf",
]
# Undocumented on purpose? No: documented.
slow = []
plain = []
"""

BENCH_WORKFLOW = """\
jobs:
  bench:
    steps:
      - run: |
          cargo bench -p lattice-demo --bench speed -- --save-baseline base
"""

BENCH_RS = """\
fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("demo_group");
    group.bench_function("inner", |b| b.iter(|| 1));
}
"""


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True, env=GIT_ENV
    ).stdout


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        files = {
            "crates/demo/Cargo.toml": CARGO_TOML,
            "crates/demo/src/lib.rs": LIB_RS,
            "crates/demo/src/a.rs": "pub mod b;\n",
            "crates/demo/src/a/b.rs": B_RS,
            "crates/demo/src/kernel.metal": "// shader\n",
            "crates/demo/tests/t.rs": TEST_RS,
            "crates/demo/benches/speed.rs": BENCH_RS,
            "crates/demo/docs/guide.md": "# Demo [guide]\n\ntext\n",
            "crates/leaf/Cargo.toml": '[package]\nname = "lattice-leaf"\nversion = "0.1.0"\n',
            "crates/leaf/src/lib.rs": "pub fn leaf() {}\n",
            "crates/leaf/README.md": "# lattice-leaf\n\nHand-written prose.\n",
            "docs/adr/ADR-001-demo.md": "# ADR-001: Demo\n\nThis concerns `lattice-demo`.\n",
            "docs/adr/ADR-002-other.md": "# ADR-002: Other\n\nOnly lattice-demo-extra here.\n",
            ".github/workflows/bench-update.yml": BENCH_WORKFLOW,
        }
        for path, text in files.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        (root / "crates/demo/untracked.rs").write_text("pub fn untracked() {}\n", encoding="utf-8")
        git(root, "init", "-q", "-b", "main")
        git(root, "add", *files)
        git(root, "commit", "-q", "-m", "fixture")
        self.metadata = root / "metadata.json"
        self.metadata.write_text(json.dumps(self.metadata_document()), encoding="utf-8")

    def metadata_document(self) -> dict:
        root = str(self.root)

        def target(name: str, kind: str, path: str, required: list[str] | None = None) -> dict:
            entry = {"name": name, "kind": [kind], "src_path": f"{root}/{path}"}
            if required:
                entry["required-features"] = required
            return entry

        demo = {
            "id": "demo",
            "name": "lattice-demo",
            "version": "0.1.0",
            "edition": "2024",
            "license": "MIT",
            "rust_version": "1.93",
            "description": "Demo crate.",
            "publish": None,
            "manifest_path": f"{root}/crates/demo/Cargo.toml",
            "features": {"default": ["fast"], "fast": ["dep:lattice-leaf"], "slow": [], "plain": []},
            "dependencies": [
                {"name": "lattice-leaf", "kind": None, "optional": True, "uses_default_features": True,
                 "features": [], "target": None, "path": f"{root}/crates/leaf"},
                {"name": "serde", "kind": None, "optional": False, "uses_default_features": True,
                 "features": [], "target": None},
            ],
            "targets": [
                target("lattice_demo", "lib", "crates/demo/src/lib.rs"),
                target("t", "test", "crates/demo/tests/t.rs"),
                target("speed", "bench", "crates/demo/benches/speed.rs", ["fast"]),
            ],
        }
        leaf = {
            "id": "leaf",
            "name": "lattice-leaf",
            "version": "0.1.0",
            "edition": "2024",
            "license": "MIT",
            "rust_version": None,
            "description": "Leaf crate.",
            "publish": [],
            "manifest_path": f"{root}/crates/leaf/Cargo.toml",
            "features": {},
            "dependencies": [],
            "targets": [target("lattice_leaf", "lib", "crates/leaf/src/lib.rs")],
        }
        return {"packages": [demo, leaf], "workspace_members": ["demo", "leaf"]}

    def baselines(self, benches: dict[str, float]) -> None:
        """Commit one x86_64 snapshot to a `perf-baselines` branch, leaving main checked out."""
        snapshot = {
            "arch": "x86_64-linux",
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "date": "2026-10-01T00:00:00+00:00",
            "benches": {key: {"ns": ns, "ns_ci_low": ns, "ns_ci_high": ns} for key, ns in benches.items()},
        }
        older = dict(snapshot, benches={"demo_group/inner": {"ns": 999.0}}, date="2026-09-01T00:00:00+00:00")
        with tempfile.TemporaryDirectory() as scratch:
            work = Path(scratch) / "wt"
            git(self.root, "worktree", "add", "-q", "--detach", str(work))
            git(work, "checkout", "-q", "--orphan", "perf-baselines")
            git(work, "rm", "-q", "-r", "--cached", ".")
            history = work / "history"
            history.mkdir()
            (history / "20260901T000000_0123456_x86_64-linux.json").write_text(json.dumps(older))
            (history / "20261001T000000_0123456_x86_64-linux.json").write_text(json.dumps(snapshot))
            git(work, "add", "history")
            git(work, "commit", "-q", "-m", "baselines")
            git(self.root, "worktree", "remove", "--force", str(work))

    def run(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = readme.main(
                ["--root", str(self.root), "--metadata", str(self.metadata), "--baselines-ref", "perf-baselines", *args]
            )
        return code, out.getvalue(), err.getvalue()

    def read(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8")


class FixtureCase(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory()
        self.fx = Fixture(Path(self.scratch.name).resolve())

    def tearDown(self) -> None:
        self.scratch.cleanup()


class ScannerTests(unittest.TestCase):
    def test_counts_ignore_strings_and_comments(self) -> None:
        counts = readme.count_rust(LIB_RS)
        self.assertEqual(counts.tests, 2)
        # `pub fn visible` and `pub mod a`; `pub(crate)` and the raw string's text do not count.
        self.assertEqual(counts.pub_items, 2)
        # The one real block; the string, the raw string and the comment hold the others.
        self.assertEqual(counts.unsafe, 1)

    def test_line_kinds(self) -> None:
        kinds = [kind for kind, _ in readme.scan_rust(LIB_RS)]
        self.assertEqual(kinds.count("blank"), 5)
        self.assertEqual(kinds[:3], ["comment", "comment", "comment"])
        self.assertEqual(kinds[13], "comment", "a nested block comment is one comment line")
        self.assertEqual(kinds.count("comment"), 5)
        self.assertEqual(len(kinds), len(LIB_RS.splitlines()))

    def test_multiline_string_lines_are_code(self) -> None:
        kinds = [kind for kind, _ in readme.scan_rust('let s = "a\n// inside\nb";\n')]
        self.assertEqual(kinds, ["code", "code", "code"])

    def test_raw_string_runs_to_its_own_terminator(self) -> None:
        # Read as an ordinary string, the inner `"#` would end it early and expose `unsafe`.
        _, code = readme.scan_rust('let r = r##"a "# unsafe"##; let s = "\\\\"; // x\n')[0]
        self.assertEqual(code, 'let r = ""; let s = ""; ')

    def test_lifetime_is_not_a_char_literal(self) -> None:
        _, code = readme.scan_rust("fn f<'a>(x: &'a str) -> char { 'x' } // unsafe\n")[0]
        self.assertIn("<'a>", code)
        self.assertNotIn("unsafe", code)

    def test_first_sentence(self) -> None:
        self.assertEqual(
            readme.first_sentence(LIB_RS), "The demo crate: it exists for tests, e.g. `this_one`."
        )
        self.assertEqual(
            readme.first_sentence(B_RS), "Inner module for `crate::visible`, see the docs."
        )
        self.assertEqual(readme.first_sentence("#![allow(x)]\n//! # Title\n//! Body.\n"), "Title")
        self.assertEqual(readme.first_sentence("use x;\n//! late\n"), "")


class MarkerTests(unittest.TestCase):
    def refuse(self, text: str, message: str) -> None:
        with self.assertRaisesRegex(readme.InputError, message):
            readme.parse_blocks(text, "R.md")

    def test_unbalanced_nested_duplicate_and_unknown_refuse(self) -> None:
        begin = "<!-- generated:{}:begin -->"
        end = "<!-- generated:{}:end -->"
        self.refuse(begin.format("stats"), "begins but never ends")
        self.refuse(end.format("stats"), "ends without a matching begin")
        self.refuse(begin.format("stats") + begin.format("docs"), "begins inside")
        self.refuse(
            begin.format("stats") + end.format("stats") + begin.format("stats") + end.format("stats"),
            "appears twice",
        )
        self.refuse(begin.format("bogus") + end.format("bogus"), "unknown generated section")

    def test_splice_keeps_prose_and_appends_missing(self) -> None:
        text = "# T\n\nprose\n\n<!-- generated:stats:begin -->\nold\n<!-- generated:stats:end -->\n\ntail\n"
        out = readme.splice(text, "R.md", {"stats": "NEW", "docs": "DOCS"})
        self.assertTrue(out.startswith("# T\n\nprose\n\n<!-- generated:stats:begin -->\n\nNEW\n\n"))
        self.assertIn("<!-- generated:stats:end -->\n\ntail\n", out)
        self.assertTrue(out.endswith("<!-- generated:docs:begin -->\n\nDOCS\n\n<!-- generated:docs:end -->\n"))
        self.assertNotIn("old", out)
        self.assertNotIn("## Crate reference", out, "the intro is added only to a README without blocks")


class GeneratorTests(FixtureCase):
    def test_write_then_check_is_clean_and_prose_survives(self) -> None:
        self.fx.baselines({"demo_group/inner": 1500.0, "unowned/x": 1.0})
        code, out, err = self.fx.run("--check")
        self.assertEqual(code, readme.EXIT_STALE, out + err)
        self.assertIn("stale crates/demo/README.md", out)
        code, _, err = self.fx.run("--write")
        self.assertEqual(code, readme.EXIT_OK, err)
        self.assertIn("no bench target source names group 'unowned'", err)
        first = self.fx.read("crates/demo/README.md")
        code, out, _ = self.fx.run("--check")
        self.assertEqual(code, readme.EXIT_OK, out)
        self.assertIn("0 stale", out)
        self.assertEqual(self.fx.read("crates/demo/README.md"), first)
        self.assertTrue(self.fx.read("crates/leaf/README.md").startswith("# lattice-leaf\n\nHand-written prose.\n\n"))

    def test_new_readme_layout_stats_and_sections(self) -> None:
        self.fx.baselines({"demo_group/inner": 1500.0})
        self.assertEqual(self.fx.run("--write")[0], readme.EXIT_OK)
        text = self.fx.read("crates/demo/README.md")
        self.assertTrue(text.startswith("# lattice-demo\n\nDemo crate.\n\n## Crate reference\n"))
        self.assertIn("- `a/`\n  - `b.rs`: Inner module for `crate::visible`, see the docs.\n", text)
        self.assertIn("- `a.rs`\n", text)
        self.assertIn("- `kernel.metal`\n", text)
        self.assertNotIn("untracked", text, "the population is the git index")
        self.assertIn("- bench (1): `speed` (requires `fast`)\n", text)
        self.assertIn("| `src/`     | 3     | 20   | 6       | 5     | 2     | 4           | 1        |", text)
        self.assertIn("| total      | 5     | 26   | 6       | 5     | 3     | 4           | 1        |", text)
        self.assertIn("| `.metal`  | 1     | 10 B  |", text)
        self.assertIn("- [Demo \\[guide\\]](docs/guide.md) (`docs/guide.md`)", text)
        self.assertIn("- `fast` (default): Faster paths. Enables `dep:lattice-leaf`.", text)
        self.assertIn("- `plain`: Enables nothing.", text)
        self.assertIn("- `slow`: Undocumented on purpose? Enables nothing.", text)
        self.assertIn("- Depends on: `lattice-leaf` (optional, enabled by `fast`)", text)
        self.assertIn("- [ADR-001: Demo](../../docs/adr/ADR-001-demo.md)", text)
        self.assertNotIn("ADR-002", text, "lattice-demo-extra is a different name")
        self.assertIn("| `demo_group/inner` | 1.50 µs        |", text)
        self.assertIn("commit `0123456789ab`, recorded 2026-10-01T00:00:00+00:00", text)
        self.assertNotIn("999", text, "only the newest snapshot per architecture is read")
        self.assertIn("- crates.io: <https://crates.io/crates/lattice-demo>", text)

    def test_crate_without_bench_data_says_so(self) -> None:
        self.fx.baselines({"demo_group/inner": 1500.0})
        self.fx.run("--write")
        leaf = self.fx.read("crates/leaf/README.md")
        self.assertIn(
            "No CI benchmark data for this crate: `bench-update.yml` records only `lattice-demo` `speed`.",
            leaf,
        )
        self.assertIn("- Used by: `lattice-demo` (optional)", leaf)
        self.assertIn("- Not published to crates.io (`publish = false`)", leaf)

    def test_snapshot_without_the_crates_groups(self) -> None:
        self.fx.baselines({"elsewhere/x": 1.0})
        self.fx.run("--write")
        self.assertIn(
            "The newest snapshots hold no entry for this crate's benchmark groups.",
            self.fx.read("crates/demo/README.md"),
        )

    def test_missing_baselines_ref_refuses_and_writes_nothing(self) -> None:
        code, _, err = self.fx.run("--write")
        self.assertEqual(code, readme.EXIT_ERROR)
        self.assertIn("perf-baselines", err)
        self.assertFalse((self.fx.root / "crates/demo/README.md").exists())

    def test_no_benchmarks_leaves_that_section_out(self) -> None:
        self.assertEqual(self.fx.run("--write", "--no-benchmarks")[0], readme.EXIT_OK)
        text = self.fx.read("crates/demo/README.md")
        self.assertNotIn("generated:benchmarks", text)
        self.assertIn("generated:stats:end", text)

    def test_bench_workflow_without_invocations_refuses(self) -> None:
        self.fx.baselines({"demo_group/inner": 1.0})
        (self.fx.root / ".github/workflows/bench-update.yml").write_text("jobs: {}\n", encoding="utf-8")
        code, _, err = self.fx.run("--check")
        self.assertEqual(code, readme.EXIT_ERROR)
        self.assertIn("no `cargo bench -p <package> --bench <target>` line", err)

    def test_malformed_markers_refuse(self) -> None:
        path = self.fx.root / "crates/leaf/README.md"
        path.write_text("# x\n<!-- generated:stats:begin -->\n", encoding="utf-8")
        code, _, err = self.fx.run("--write", "--no-benchmarks")
        self.assertEqual(code, readme.EXIT_ERROR)
        self.assertIn("crates/leaf/README.md: 'stats' begins but never ends", err)
        self.assertEqual(path.read_text(encoding="utf-8"), "# x\n<!-- generated:stats:begin -->\n")

    def test_crate_under_models_gets_a_readme(self) -> None:
        files = {
            "crates/models/deep/Cargo.toml": '[package]\nname = "lattice-model-deep"\nversion = "0.1.0"\n',
            "crates/models/deep/src/lib.rs": "//! Deep model.\npub fn deep() {}\n",
        }
        for path, text in files.items():
            target = self.fx.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        git(self.fx.root, "add", *files)
        git(self.fx.root, "commit", "-q", "-m", "nested crate")
        document = self.fx.metadata_document()
        document["packages"].append({
            "id": "deep",
            "name": "lattice-model-deep",
            "version": "0.1.0",
            "edition": "2024",
            "license": "MIT",
            "rust_version": None,
            "description": "Deep model crate.",
            "publish": None,
            "manifest_path": f"{self.fx.root}/crates/models/deep/Cargo.toml",
            "features": {},
            "dependencies": [],
            "targets": [{"name": "lattice_model_deep", "kind": ["lib"],
                         "src_path": f"{self.fx.root}/crates/models/deep/src/lib.rs"}],
        })
        document["workspace_members"].append("deep")
        self.fx.metadata.write_text(json.dumps(document), encoding="utf-8")
        code, out, _ = self.fx.run("--write", "--no-benchmarks", "--crate", "lattice-model-deep")
        self.assertEqual(code, readme.EXIT_OK)
        self.assertIn("wrote crates/models/deep/README.md", out)
        text = self.fx.read("crates/models/deep/README.md")
        self.assertIn("Deep model crate.", text)
        self.assertIn("`lib.rs`", text)
        self.assertIn("Deep model.", text)

    def test_crate_filter(self) -> None:
        code, out, _ = self.fx.run("--write", "--no-benchmarks", "--crate", "lattice-leaf")
        self.assertEqual(code, readme.EXIT_OK)
        self.assertIn("1 READMEs", out)
        self.assertFalse((self.fx.root / "crates/demo/README.md").exists())
        self.assertEqual(self.fx.run("--check", "--no-benchmarks", "--crate", "nope")[0], readme.EXIT_ERROR)


if __name__ == "__main__":
    unittest.main()
