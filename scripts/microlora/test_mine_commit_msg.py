"""Fixtures for commit-message curation, including real local Git histories."""

import json
import os
import subprocess
import tempfile
import unittest
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import mine_commit_msg as miner


class MinerTests(unittest.TestCase):
    def row(
        self,
        *,
        repo="lattice",
        sha="a",
        day="2026-07-01",
        prompt="prompt",
        completion=" message",
        diff="diff",
    ):
        authored = datetime.fromisoformat(day).replace(tzinfo=UTC)
        return miner.Candidate(
            repo, sha, authored, miner.time_split(authored), prompt, completion, diff
        )

    def test_author_time_boundaries_include_offsets(self):
        for timestamp, expected in (
            ("2026-07-14T23:59:59+00:00", "train"),
            ("2026-07-15T00:00:00+00:00", "valid"),
            ("2026-08-14T23:59:59+00:00", "valid"),
            ("2026-08-15T00:00:00+00:00", "test"),
            ("2026-07-14T20:00:00-04:00", "valid"),
        ):
            with self.subTest(timestamp=timestamp):
                self.assertEqual(
                    miner.time_split(datetime.fromisoformat(timestamp)), expected
                )

    def test_trailers_pr_number_and_body_budget(self):
        cleaned = miner.clean_completion(
            "fix(core): preserve cache keys (#123)",
            "Keep keys stable.\n\nCo-Authored-By: Fixture <fixture@example.org>\n continuation\nSigned-off-by: Fixture <fixture@example.org>\nClaude-Session: id",
        )
        self.assertEqual(
            cleaned, " fix(core): preserve cache keys\n\nKeep keys stable."
        )
        self.assertEqual(
            miner.clean_completion("feat!: support new wire format", "x" * 200),
            " feat!: support new wire format",
        )
        self.assertIsNone(miner.clean_completion("fix: short", ""))

    def test_filter_both_rename_paths_binary_and_unsafe_headers(self):
        diff_lines = [
            "diff --git a/Cargo.lock b/src/clean.rs",
            "@@ -1 +1 @@",
            "-old",
            "+new",
            "diff --git a/src/clean.rs b/vendor/clean.rs",
            "@@ -1 +1 @@",
            "-old",
            "+new",
            'diff --git "a/file name.rs" "b/file name.rs"',
            "@@ -1 +1 @@",
            "+bad",
            "diff --git a/image.bin b/image.bin",
            "Binary files a/image.bin and b/image.bin differ",
            "diff --git a/src/old.rs b/src/new.rs",
            "similarity index 80%",
            "rename from src/old.rs",
            "rename to src/new.rs",
            "--- a/src/old.rs",
            "+++ b/src/new.rs",
            "@@ -1 +1 @@",
            " unchanged",
            "-before",
            "+after",
        ]
        diff = "\n".join(diff_lines)
        counts = Counter()
        reduced = miner.reduce_diff(diff, counts)
        self.assertEqual(counts["files_excluded_path"], 2)
        self.assertEqual(counts["files_unsafe_header"], 1)
        self.assertEqual(counts["files_binary"], 1)
        self.assertIn("diff --git a/src/old.rs b/src/new.rs", reduced)
        self.assertIn("+after", reduced)
        self.assertNotIn(" unchanged", reduced)
        self.assertNotIn("+bad", reduced)
        for path in (
            "a/../secret.rs",
            ".khive/private.md",
            "src/generated/code.rs",
            "src/data.jsonl",
            "web/package-lock.json",
            "web/npm-shrinkwrap.json",
            "web/pnpm-lock.yaml",
            "web/bun.lockb",
            "web/file.min.js",
            ".env.local",
        ):
            self.assertTrue(miner.excluded_path(path), path)

    def test_truncation_reserves_exact_marker_and_never_splits_line(self):
        lines = ["diff --git a/a.rs b/a.rs", "@@ -1 +1 @@", "+first"] + [
            "+" + "x" * 30
        ] * 120
        result = miner.truncate_lines(lines, 110)
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 110)
        kept = result.splitlines()[:-1]
        self.assertEqual(kept, lines[: len(kept)])
        self.assertEqual(
            result.splitlines()[-1], f"[... {len(lines) - len(kept)} more lines]"
        )
        self.assertIsNone(miner.truncate_lines(lines, 10))

    def test_dedup_keeps_earliest_and_global_boilerplate_removes_all(self):
        counts = {"lattice": Counter()}
        early = self.row(sha="early")
        later = self.row(sha="later", day="2026-08-20")
        frequent = [
            self.row(
                sha=str(i), prompt=f"prompt {i}", completion=" same", diff=f"diff {i}"
            )
            for i in range(4)
        ]
        kept = miner.deduplicate([later, *frequent, early], counts)
        self.assertEqual(kept, [early])
        self.assertEqual(counts["lattice"]["rows_exact_duplicate"], 1)
        self.assertEqual(counts["lattice"]["rows_boilerplate"], 4)

    def test_repo_token_cannot_hide_cross_split_diff_leakage(self):
        counts = {"lattice": Counter(), "khive": Counter()}
        rows = [
            self.row(prompt="repo: lattice\npatch", completion=" first"),
            self.row(
                repo="khive",
                day="2026-08-20",
                prompt="repo: khive\npatch",
                completion=" second",
            ),
        ]
        self.assertEqual(miner.deduplicate(rows, counts), [])
        self.assertEqual(counts["lattice"]["rows_cross_split_diff"], 1)
        self.assertEqual(counts["khive"]["rows_cross_split_diff"], 1)

    def test_exact_tokenizer_rejections_and_unknown_hash_fail_closed(self):
        row = self.row()
        digest = miner.pair_digest(row.prompt, row.completion)
        counts = {"lattice": Counter()}
        self.assertEqual(miner.deduplicate([row], counts, {digest}), [])
        self.assertEqual(counts["lattice"]["rows_tokenizer_rejected"], 1)
        with self.assertRaises(miner.CurationError):
            miner.deduplicate([row], {"lattice": Counter()}, {"0" * 64})

    def test_sensitive_screen_does_not_confuse_hunks_or_decorators_with_email(self):
        self.assertEqual(miner.sensitive_reason("+name: person@example.org"), "email")
        self.assertEqual(
            miner.sensitive_reason('+api_key = "abcdefghijklmno"'), "secret_pattern"
        )
        self.assertEqual(
            miner.sensitive_reason("-----BEGIN RSA PRIVATE KEY-----"), "secret_pattern"
        )
        self.assertIsNone(
            miner.sensitive_reason("@@ -1 +1 @@\n+@decorator\n+import @scope/package")
        )
        long_line = "@@ -1 +1 @@\n+" + "x" * 20_000
        self.assertIsNone(miner.sensitive_reason(long_line))
        self.assertEqual(
            miner.sensitive_reason(long_line + " person@example.org"), "email"
        )

    def test_readback_fails_for_invalid_rows_and_count(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            for row in (
                {"prompt": "", "completion": " ok"},
                {"prompt": 7, "completion": " ok"},
                {"prompt": "p", "completion": " ok", "extra": 1},
            ):
                path.write_text(json.dumps(row) + "\n")
                with self.assertRaises(miner.CurationError):
                    miner.validate_jsonl(path, 1)
            path.write_text(json.dumps({"prompt": "p", "completion": " ok"}) + "\n")
            with self.assertRaises(miner.CurationError):
                miner.validate_jsonl(path, 2)


class GitFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        self.run_git("init", "-b", "main")
        self.run_git("config", "user.name", "Fixture")
        self.run_git("config", "user.email", "fixture@example.org")
        self.run_git(
            "remote", "add", "origin", "https://github.com/ohdearquant/lattice.git"
        )

    def run_git(self, *args, date=None, author=None):
        env = os.environ.copy()
        if date:
            env.update(
                GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE="2026-09-01T00:00:00+00:00"
            )
        if author:
            env["GIT_AUTHOR_NAME"] = author
        result = subprocess.run(
            ["git", "-C", str(self.repo), *args],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def commit(self, name, content, subject, date, author=None):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self.run_git("add", "--", name)
        self.run_git("commit", "-m", subject, date=date, author=author)

    def test_mining_pins_main_excludes_branch_bot_lock_and_sensitive_tail(self):
        self.commit(
            "src/train.rs",
            "let train = 1;\n",
            "feat: introduce training fixture",
            "2026-07-14T23:59:59+00:00",
        )
        self.commit(
            "src/valid.rs",
            "let valid = 2;\n",
            "test: introduce validation fixture",
            "2026-07-15T00:00:00+00:00",
        )
        self.commit(
            "src/test.rs",
            "let test = 3;\n",
            "fix: introduce testing fixture",
            "2026-08-15T00:00:00+00:00",
        )
        self.commit(
            "Cargo.lock",
            "generated = 1\n",
            "chore: refresh generated lockfile",
            "2026-08-16T00:00:00+00:00",
        )
        self.commit(
            "src/bot.rs",
            "let bot = 4;\n",
            "chore: introduce automated fixture",
            "2026-08-17T00:00:00+00:00",
            "dependabot[bot]",
        )
        self.commit(
            "src/secret.rs",
            "let item = 1;\n" * 200 + 'api_key = "abcdefghijklmno"\n',
            "test: add suspicious tail fixture",
            "2026-08-18T00:00:00+00:00",
        )
        source = miner.pin_source(self.repo)
        self.run_git("checkout", "-b", "other")
        self.commit(
            "src/branch.rs",
            "let branch = 5;\n",
            "feat: add unrelated branch fixture",
            "2026-08-19T00:00:00+00:00",
        )
        (self.repo / "src/train.rs").write_text("uncommitted text must never be read\n")
        counts = Counter()
        rows = miner.mine(source, counts)
        self.assertEqual([row.split for row in rows], ["test", "valid", "train"])
        self.assertTrue(all(row.prompt.startswith("repo: lattice\n") for row in rows))
        self.assertTrue(all("uncommitted" not in row.prompt for row in rows))
        self.assertEqual(counts["commits_seen"], 6)
        self.assertEqual(counts["commits_bot"], 1)
        self.assertEqual(counts["commits_secret_pattern"], 1)
        self.assertEqual(counts["files_excluded_path"], 1)
        out = Path(self.temp.name) / "out"
        totals = {"lattice": counts}
        kept = miner.deduplicate(rows, totals)
        miner.write_outputs(out, [source], kept, totals)
        self.assertIn("| TOTAL | 1 | 1 | 1 |", (out / "CURATION.md").read_text())
        self.assertIn(source.sha, (out / "CURATION.md").read_text())
        self.assertEqual(
            miner.validate_jsonl(out / "train.jsonl", 1)[0]["completion"],
            " feat: introduce training fixture",
        )

    def test_unknown_origin_and_missing_main_fail_closed(self):
        with self.assertRaises(miner.CurationError):
            miner.pin_source(self.repo)
        self.run_git(
            "remote", "set-url", "origin", "https://github.com/example/private.git"
        )
        with self.assertRaises(miner.CurationError):
            miner.pin_source(self.repo)

    def test_pinned_reader_ignores_mutable_attributes_and_git_environment(self):
        self.commit(
            "code.rs",
            "pub fn fixture() {}\n",
            "feat: introduce a public fixture",
            "2026-07-01T00:00:00+00:00",
        )
        source = miner.pin_source(self.repo)
        baseline = miner.mine(source, Counter())
        self.assertEqual(len(baseline), 1)
        (self.repo / ".gitattributes").write_text("*.rs binary\n")
        (self.repo / ".git/info/attributes").write_text("*.rs binary\n")
        attrs = Path(self.temp.name) / "global-attributes"
        attrs.write_text("*.rs binary\n")
        self.run_git("config", "core.attributesFile", str(attrs))
        self.run_git("add", ".gitattributes")
        with patch.dict(
            os.environ,
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.attributesFile",
                "GIT_CONFIG_VALUE_0": str(attrs),
                "GIT_WORK_TREE": str(self.repo),
                "GIT_INDEX_FILE": str(Path(self.temp.name) / "foreign-index"),
            },
        ):
            self.assertEqual(miner.mine(source, Counter()), baseline)


if __name__ == "__main__":
    unittest.main()
