#!/usr/bin/env python3
"""Regression tests for post-merge benchmark progression state."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "perf-postmerge-base.py"
WORKFLOW = REPO / ".github" / "workflows" / "perf-postmerge-gate.yml"
STATE_X86 = "perf-postmerge-measured-x86_64-linux"
GIT = ("git", "-c", "core.hooksPath=/dev/null")


class Repository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.remote = root.parent / "remote.git"
        subprocess.run([*GIT, "init", "-q", "--bare", str(self.remote)], check=True)
        subprocess.run([*GIT, "init", "-q", "-b", "main", str(root)], check=True)
        self.git("remote", "add", "origin", str(self.remote))
        self.env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        }
        self.commits = [self.commit(str(index)) for index in range(3)]
        self.git("push", "-q", "-u", "origin", "main")

    def git(
        self,
        *args: str,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*GIT, "-C", str(self.root), *args],
            check=True,
            capture_output=True,
            text=True,
            input=input_text,
            env=getattr(self, "env", os.environ),
        )

    def commit(self, value: str) -> str:
        (self.root / "value.txt").write_text(value, encoding="utf-8")
        self.git("add", "value.txt")
        self.git("commit", "-qm", value)
        return self.git("rev-parse", "HEAD").stdout.strip()

    def push_state(self, commit: str, branch: str = STATE_X86) -> None:
        self.git("push", "-q", "origin", f"{commit}:refs/heads/{branch}")

    def resolve(
        self,
        *,
        event_name: str = "push",
        event_before: str = "",
        base_ref: str = "",
        head_ref: str = "",
        state_branch: str = STATE_X86,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
        output = self.root / "github-output.txt"
        output.unlink(missing_ok=True)
        result = subprocess.run(
            [
                "python3",
                str(SCRIPT),
                "--repo",
                str(self.root),
                "--event-name",
                event_name,
                "--event-before",
                event_before,
                "--base-ref",
                base_ref,
                "--head-ref",
                head_ref,
                "--state-branch",
                state_branch,
                "--github-output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        values = {}
        if output.exists():
            values = dict(
                line.split("=", 1)
                for line in output.read_text(encoding="utf-8").splitlines()
            )
        return result, values


class LastMeasuredBaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Repository(Path(self.tmp.name) / "repo")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_recorded_commit_spans_every_skipped_merge(self) -> None:
        first, second, head = self.repo.commits
        self.repo.push_state(first)

        result, values = self.repo.resolve(event_before=second)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(values["base"], first)
        self.assertEqual(values["head"], head)
        self.assertEqual(values["base_source"], "recorded")
        self.assertEqual(values["span_count"], "2")
        self.assertEqual(values["state_before"], first)

    def test_missing_record_falls_back_loudly_to_event_parent(self) -> None:
        _, parent, head = self.repo.commits

        result, values = self.repo.resolve(
            event_before=parent,
            state_branch="perf-postmerge-measured-aarch64-linux",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(values["base"], parent)
        self.assertEqual(values["head"], head)
        self.assertEqual(values["base_source"], "event-parent-fallback")
        self.assertEqual(values["span_count"], "1")
        self.assertEqual(values["state_before"], "missing")

    def test_dispatch_inputs_ignore_progression_and_do_not_query_it(self) -> None:
        first, _, head = self.repo.commits
        self.repo.push_state(self.repo.commits[1])

        result, values = self.repo.resolve(
            event_name="workflow_dispatch",
            base_ref=first,
            head_ref=head,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(values["base"], first)
        self.assertEqual(values["head"], head)
        self.assertEqual(values["base_source"], "dispatch-input")
        self.assertEqual(values["state_before"], "not-queried")

    def test_non_ancestor_record_fails_closed(self) -> None:
        tree = self.repo.git("mktree", input_text="").stdout.strip()
        orphan = self.repo.git("commit-tree", tree, "-m", "orphan").stdout.strip()
        self.repo.push_state(orphan)

        result, values = self.repo.resolve(event_before=self.repo.commits[1])

        self.assertEqual(values, {})
        self.assertEqual(result.returncode, 2)
        self.assertIn("is not an ancestor", result.stderr)


class WorkflowContractTests(unittest.TestCase):
    def test_successful_push_advances_only_its_architecture_record(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("contents: write", workflow)
        self.assertIn("perf-postmerge-measured-${{ matrix.arch }}", workflow)
        self.assertIn("python3 scripts/perf-postmerge-base.py", workflow)
        self.assertIn("if: ${{ success() && github.event_name == 'push' }}", workflow)
        self.assertIn('git push origin "$HEAD:refs/heads/$STATE_BRANCH"', workflow)
        self.assertNotIn("'scripts/perf-postmerge-base.py'", workflow)


if __name__ == "__main__":
    unittest.main()
