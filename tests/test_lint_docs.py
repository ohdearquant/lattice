"""Regression tests for scripts/lint-docs.sh."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lint-docs.sh"
_GIT = ("git", "-c", "core.hooksPath=/dev/null")


class LintDocsTests(unittest.TestCase):
    def test_selftest_does_not_mutate_callers_relative_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            # Test helpers invoking real Git must disable repository hooks.
            subprocess.run([*_GIT, "init", "-q"], cwd=repo, check=True)
            for name in ("README.md", "keep.txt"):
                (repo / name).write_text(f"{name}\n", encoding="utf-8")
            subprocess.run([*_GIT, "add", "."], cwd=repo, check=True)

            before = self._index_count(repo)
            env = os.environ.copy()
            env["TMPDIR"] = str(repo)
            env["GIT_INDEX_FILE"] = "../../.git/index"
            subprocess.run([str(_SCRIPT), "--selftest"], cwd=repo, env=env, check=True)

            self.assertEqual(self._index_count(repo), before)

    @staticmethod
    def _index_count(repo: Path) -> int:
        result = subprocess.run(
            [*_GIT, "ls-files"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return len(result.stdout.splitlines())
