#!/usr/bin/env python3
"""Argument contract tests for scripts/publish.sh.

The script runs against stub `cargo`, `sleep` and package-size-check.sh
executables that record their argv, so no test reaches the registry.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "publish.sh"

_RECORDER = """#!/bin/sh
printf '%s\\n' "$(basename "$0") $*" >> "$CALL_LOG"
"""


def _write_exec(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class PublishScriptArgs(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        scripts = root / "scripts"
        scripts.mkdir()
        shutil.copy2(SCRIPT, scripts / "publish.sh")
        _write_exec(scripts / "package-size-check.sh", _RECORDER)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        _write_exec(bin_dir / "cargo", _RECORDER)
        _write_exec(bin_dir / "sleep", _RECORDER)
        self.script = scripts / "publish.sh"
        self.call_log = root / "calls.log"
        self.env = dict(os.environ)
        self.env["PATH"] = f"{bin_dir}{os.pathsep}{self.env.get('PATH', '')}"
        self.env["CALL_LOG"] = str(self.call_log)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, *args: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        proc = subprocess.run(
            ["sh", str(self.script), *args],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        calls = self.call_log.read_text().splitlines() if self.call_log.exists() else []
        return proc, calls

    def _publishes(self, calls: list[str]) -> list[str]:
        return [c for c in calls if c.startswith("cargo publish ")]

    def test_no_argument_publishes_every_tier(self) -> None:
        proc, calls = self._run()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        publishes = self._publishes(calls)
        self.assertEqual(len(publishes), 11, calls)
        self.assertFalse(any("--dry-run" in c for c in publishes), publishes)
        self.assertEqual(
            [c for c in calls if c.startswith(("cargo ", "sleep "))],
            [
                "cargo publish -p lattice-fann",
                "cargo publish -p lattice-transport",
                "cargo publish -p lattice-inference-core",
                "sleep 30",
                "cargo publish -p lattice-tokenizer",
                "sleep 30",
                "cargo publish -p lattice-grammar",
                "cargo publish -p lattice-weights",
                "cargo publish -p lattice-metal-runtime",
                "sleep 30",
                "cargo publish -p lattice-quant",
                "sleep 30",
                "cargo publish -p lattice-inference",
                "sleep 30",
                "cargo publish -p lattice-embed",
                "cargo publish -p lattice-tune",
            ],
        )

    def test_dry_run_stops_after_the_leaf_tier(self) -> None:
        proc, calls = self._run("--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        publishes = self._publishes(calls)
        self.assertEqual(len(publishes), 3, calls)
        self.assertTrue(all(c.endswith(" --dry-run") for c in publishes), publishes)
        self.assertEqual(
            publishes,
            [
                "cargo publish -p lattice-fann --dry-run",
                "cargo publish -p lattice-transport --dry-run",
                "cargo publish -p lattice-inference-core --dry-run",
            ],
        )
        self.assertFalse(any(c.startswith("sleep ") for c in calls), calls)

    def test_misspelled_flag_exits_before_any_command(self) -> None:
        for arg in ("--dryrun", "-n", "--dry-run=1", "dry-run", " --dry-run"):
            with self.subTest(arg=arg):
                if self.call_log.exists():
                    self.call_log.unlink()
                proc, calls = self._run(arg)
                self.assertEqual(proc.returncode, 2, proc.stdout)
                self.assertIn("usage:", proc.stderr)
                self.assertEqual(calls, [])

    def test_extra_argument_exits_before_any_command(self) -> None:
        proc, calls = self._run("--dry-run", "extra")
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertIn("usage:", proc.stderr)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
