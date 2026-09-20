"""Hermetic shell-dispatch contracts; fake Deno does not validate Markdown."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lint-docs.sh"
_TOOLS = "git dirname wc tr mktemp mkdir cat chmod cmp rm xargs".split()
_MARKDOWN = ["README.md", "docs/deep/nested.md", "docs/space name.md"]
_CHECKS = [
    "check-capability-matrix.sh:--selftest", "check-capability-matrix.sh:",
    "lint-absolute-paths.sh:--selftest", "lint-absolute-paths.sh:",
    "lint-source-markers.sh:--selftest", "lint-source-markers.sh:",
]


class LintDocsModeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lint-docs-modes-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for tool in _TOOLS:
            executable = shutil.which(tool)
            self.assertIsNotNone(executable, f"required fixture tool: {tool}")
            (self.bin / tool).symlink_to(executable)
        self.env = {
            "PATH": str(self.bin), "HOME": str(self.root),
            "TMPDIR": str(self.root), "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "DENO_ARGS": str(self.root / "deno-args"),
            "CHECK_LOG": str(self.root / "checks"),
        }
        self.repo = self.root / "repo"
        scripts = self.repo / "scripts"
        scripts.mkdir(parents=True)
        self.script = scripts / "lint-docs.sh"
        self.script.write_bytes(_SCRIPT.read_bytes())
        self.script.chmod(0o755)
        for name in ("check-capability-matrix.sh", "lint-absolute-paths.sh",
                     "lint-source-markers.sh"):
            check = scripts / name
            check.write_text(
                f'#!/bin/sh\nprintf "{name}:%s\\n" "${{1:-}}" >>"$CHECK_LOG"\n'
            )
            check.chmod(0o755)
        for name in _MARKDOWN:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Fixture\n")
        for args in (("init", "-q"), ("add", "--", *_MARKDOWN)):
            subprocess.run([str(self.bin / "git"), *args], cwd=self.repo,
                           env=self.env, check=True, capture_output=True)
        (self.repo / "untracked.md").write_text("# Untracked\n")

    def invoke(self, *args, strict=None, deno=False, fmt_status=0):
        for name in ("deno-args", "checks"):
            (self.root / name).unlink(missing_ok=True)
        if deno:
            executable = self.bin / "deno"
            executable.write_text(
                '#!/bin/sh\nprintf "%s\\0" "$@" >>"$DENO_ARGS"\n'
                'printf "\\n" >>"$DENO_ARGS"\necho "fake-deno:$1"\n'
                'if [ "$1" = fmt ]; then exit "$DENO_FMT_STATUS"; fi\n'
            )
            executable.chmod(0o755)
        else:
            (self.bin / "deno").unlink(missing_ok=True)
        env = dict(self.env, DENO_FMT_STATUS=str(fmt_status))
        if strict is not None:
            env["LATTICE_REQUIRE_DENO"] = strict
        result = subprocess.run(["/bin/sh", str(self.script), *args],
                                cwd=self.repo, env=env, capture_output=True,
                                text=True, timeout=20)
        capture = self.root / "deno-args"
        calls = ([row.decode().rstrip("\0").split("\0")
                  for row in capture.read_bytes().splitlines()]
                 if capture.exists() else [])
        checks = self.root / "checks"
        return result, calls, checks.read_text().splitlines() if checks.exists() else []

    def test_missing_deno_default_reports_skip_and_runs_other_checks(self):
        for strict in (None, "0"):
            with self.subTest(strict=strict):
                result, calls, checks = self.invoke(strict=strict)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines()[-1],
                                 "=== Doc Lint Passed (Markdown checks SKIPPED: deno not found) ===")
                self.assertEqual(calls, [])
                self.assertEqual(checks, _CHECKS)

    def test_missing_deno_strict_refuses_without_success(self):
        result, calls, checks = self.invoke(strict="1")
        self.assertEqual(result.returncode, 127, result.stdout + result.stderr)
        self.assertIn("LATTICE_REQUIRE_DENO=1 requires deno", result.stderr)
        self.assertNotIn("Doc Lint Passed", result.stdout + result.stderr)
        self.assertEqual(calls, [])
        self.assertEqual(checks, [])

    def test_missing_deno_required_modes_keep_their_errors(self):
        for mode, action in (("--format", "format"), ("--markdown-only", "lint")):
            for strict in (None, "1"):
                with self.subTest(mode=mode, strict=strict):
                    result, calls, checks = self.invoke(mode, strict=strict)
                    self.assertEqual(result.returncode, 127)
                    self.assertIn(f"deno not found; cannot {action} Markdown", result.stderr)
                    self.assertNotIn("Doc Lint Passed", result.stdout + result.stderr)
                    self.assertEqual(calls, [])
                    self.assertEqual(checks, [])

    def test_present_deno_checks_exact_tracked_paths_and_reports_full_run(self):
        for strict in (None, "1"):
            with self.subTest(strict=strict):
                result, calls, checks = self.invoke(strict=strict, deno=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(calls, [["fmt", "--check", *_MARKDOWN], ["lint", *_MARKDOWN]])
                self.assertIn("fake-deno:fmt", result.stdout)
                self.assertIn("recursive tracked-Markdown selftest OK", result.stdout)
                self.assertEqual(checks, _CHECKS)
                self.assertEqual(result.stdout.splitlines()[-1], "=== Doc Lint Passed ===")
                self.assertNotIn("SKIPPED", result.stdout)

    def test_formatter_failure_prevents_success_and_later_checks(self):
        for strict in (None, "1"):
            with self.subTest(strict=strict):
                result, calls, checks = self.invoke(strict=strict, deno=True, fmt_status=37)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls, [["fmt", "--check", *_MARKDOWN]])
                self.assertNotIn("Doc Lint Passed", result.stdout + result.stderr)
                self.assertNotIn("recursive tracked-Markdown selftest OK", result.stdout)
                self.assertEqual(checks, [])


if __name__ == "__main__":
    unittest.main()
