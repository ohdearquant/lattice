"""Tests for the fail-closed CI changed-file range selector."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci-changed-files.sh"
_ROOT = _SCRIPT.parent.parent
_ZERO_SHA = "0" * 40
_REQUIRED_WORKFLOWS = (
    ".github/workflows/app-binaries.yml",
    ".github/workflows/cargo-audit.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/e2e-parity.yml",
)


class ChangedFilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tempdir.name)
        self._git("init", "-q", "-b", "main")
        self._git("config", "user.name", "CI Test")
        self._git("config", "user.email", "ci-test@example.invalid")

    def tearDown(self) -> None:
        self._tempdir.cleanup()

    def _git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _commit(self, path: str, contents: str) -> str:
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")
        self._git("add", path)
        self._git("commit", "-q", "-m", f"write {path}")
        return self._git("rev-parse", "HEAD")

    def _run(
        self,
        event: str,
        base_sha: str,
        head_sha: str,
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            GITHUB_EVENT_NAME=event,
            CI_BASE_SHA=base_sha,
            CI_HEAD_SHA=head_sha,
        )
        return subprocess.run(
            [str(_SCRIPT)],
            cwd=self.repo,
            env=env,
            check=check,
            capture_output=True,
            text=True,
        )

    def test_merge_group_reports_every_change_in_the_event_range(self) -> None:
        base = self._commit("README.md", "base\n")
        self._commit("docs/queue.md", "docs\n")
        head = self._commit("crates/inference/src/queue.rs", "code\n")

        result = self._run("merge_group", base, head)

        self.assertEqual(
            result.stdout.splitlines(),
            ["crates/inference/src/queue.rs", "docs/queue.md"],
        )

    def test_pull_request_uses_the_event_base(self) -> None:
        base = self._commit("README.md", "base\n")
        head = self._commit("crates/embed/src/change.rs", "code\n")

        result = self._run("pull_request", base, head)

        self.assertEqual(result.stdout.splitlines(), ["crates/embed/src/change.rs"])

    def test_multi_commit_push_is_not_reduced_to_the_last_commit(self) -> None:
        base = self._commit("README.md", "base\n")
        self._commit("crates/fann/src/change.rs", "code\n")
        head = self._commit("docs/followup.md", "docs\n")

        result = self._run("push", base, head)

        self.assertEqual(
            result.stdout.splitlines(),
            ["crates/fann/src/change.rs", "docs/followup.md"],
        )

    def test_rename_reports_the_relevant_source_and_destination(self) -> None:
        base = self._commit("crates/embed/src/moved.rs", "code\n")
        (self.repo / "docs").mkdir()
        self._git("mv", "crates/embed/src/moved.rs", "docs/moved.md")
        self._git("commit", "-q", "-m", "move code into docs")
        head = self._git("rev-parse", "HEAD")

        result = self._run("merge_group", base, head)

        self.assertEqual(
            result.stdout.splitlines(),
            ["crates/embed/src/moved.rs", "docs/moved.md"],
        )

    def test_unicode_path_remains_classifiable(self) -> None:
        base = self._commit("README.md", "base\n")
        head = self._commit("crates/inference/src/café.rs", "code\n")

        result = self._run("merge_group", base, head)

        self.assertEqual(
            result.stdout.splitlines(), ["crates/inference/src/café.rs"]
        )

    def test_control_character_path_fails_closed(self) -> None:
        base = self._commit("README.md", "base\n")
        head = self._commit("crates/inference/src/line\nbreak.rs", "code\n")

        result = self._run("merge_group", base, head, check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires Git quoting", result.stderr)

    def test_first_push_reports_root_commit_files(self) -> None:
        head = self._commit("Cargo.toml", "[workspace]\n")

        result = self._run("push", _ZERO_SHA, head)

        self.assertEqual(result.stdout.splitlines(), ["Cargo.toml"])

    def test_first_push_with_history_reports_the_entire_tree(self) -> None:
        self._commit("crates/inference/src/change.rs", "code\n")
        head = self._commit("docs/followup.md", "docs\n")

        result = self._run("push", _ZERO_SHA, head)

        self.assertEqual(
            result.stdout.splitlines(),
            ["crates/inference/src/change.rs", "docs/followup.md"],
        )

    def test_checkout_head_mismatch_fails_closed(self) -> None:
        base = self._commit("README.md", "base\n")
        expected_head = self._commit("docs/change.md", "change\n")
        self._git("checkout", "-q", base)

        result = self._run("merge_group", base, expected_head, check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match event head", result.stderr)

    def test_non_ancestor_base_fails_closed(self) -> None:
        base = self._commit("README.md", "base\n")
        self._git("checkout", "-q", "--orphan", "other")
        head = self._commit("other.txt", "unrelated\n")

        result = self._run("merge_group", base, head, check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is not an ancestor", result.stderr)

    def test_non_hexadecimal_revision_fails_before_git_parsing(self) -> None:
        head = self._commit("README.md", "base\n")

        result = self._run("merge_group", "HEAD^{tree}", head, check=False)

        self.assertEqual(result.returncode, 2)
        self.assertIn("must be hexadecimal commit IDs", result.stderr)

    def test_schedule_requires_explicit_workflow_policy(self) -> None:
        head = self._commit("README.md", "base\n")

        result = self._run("schedule", head, head, check=False)

        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported change-detection event", result.stderr)


class MergeQueueWorkflowTests(unittest.TestCase):
    def test_every_required_workflow_listens_for_merge_group_checks(self) -> None:
        trigger = (
            "  merge_group:\n"
            "    branches: [main]\n"
            "    types: [checks_requested]\n"
        )

        for relative_path in _REQUIRED_WORKFLOWS:
            with self.subTest(workflow=relative_path):
                contents = (_ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn(trigger, contents)


if __name__ == "__main__":
    unittest.main()
