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


class _PublishScriptFixture(unittest.TestCase):
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


class PublishScriptArgs(_PublishScriptFixture):
    def test_no_argument_publishes_every_tier(self) -> None:
        proc, calls = self._run()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        publishes = self._publishes(calls)
        self.assertEqual(len(publishes), 26, calls)
        self.assertFalse(any("--dry-run" in c for c in publishes), publishes)
        self.assertEqual(
            [c for c in calls if c.startswith(("cargo ", "sleep "))],
            [
                "cargo publish -p lattice-lora",
                "cargo publish -p lattice-transport",
                "cargo publish -p lattice-simd",
                "cargo publish -p lattice-inference-core",
                "sleep 30",
                "cargo publish -p lattice-fann",
                "cargo publish -p lattice-tokenizer",
                "sleep 30",
                "cargo publish -p lattice-grammar",
                "cargo publish -p lattice-weights",
                "cargo publish -p lattice-metal-runtime",
                "cargo publish -p lattice-model-io",
                "sleep 30",
                "cargo publish -p lattice-arch",
                "sleep 30",
                "cargo publish -p lattice-quant",
                "cargo publish -p lattice-compute",
                "sleep 30",
                "cargo publish -p lattice-forecast",
                "cargo publish -p lattice-model-embeddinggemma2",
                "cargo publish -p lattice-model-bert",
                "cargo publish -p lattice-model-qwen3-embedding",
                "cargo publish -p lattice-model-gemma4",
                "cargo publish -p lattice-model-ernie45",
                "cargo publish -p lattice-model-qwen35",
                "sleep 30",
                "cargo publish -p lattice-router",
                "cargo publish -p lattice-backward",
                "sleep 30",
                "cargo publish -p lattice-inference",
                "sleep 30",
                "cargo publish -p lattice-embed",
                "cargo publish -p lattice-tune",
                "cargo publish -p lattice-serve",
            ],
        )

    def test_dry_run_stops_after_the_leaf_tier(self) -> None:
        proc, calls = self._run("--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        publishes = self._publishes(calls)
        self.assertEqual(len(publishes), 4, calls)
        self.assertTrue(all(c.endswith(" --dry-run") for c in publishes), publishes)
        self.assertEqual(
            publishes,
            [
                "cargo publish -p lattice-lora --dry-run",
                "cargo publish -p lattice-transport --dry-run",
                "cargo publish -p lattice-simd --dry-run",
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


_FAILING_CARGO = """#!/bin/sh
printf '%s\\n' "$(basename "$0") $*" >> "$CALL_LOG"
[ "$*" = "publish -p $STUB_CRATE" ] || exit 0
case "$STUB_MODE" in
    429-once)
        [ -e "$STUB_STATE" ] && exit 0
        : > "$STUB_STATE"
        echo "error: failed to publish to registry at https://crates.io"
        echo "Caused by:"
        echo "  the remote server responded with an error (status 429 Too Many Requests): You have published too many new crates in a short period of time."
        exit 101
        ;;
    429-always)
        echo "  the remote server responded with an error (status 429 Too Many Requests): You have published too many new crates in a short period of time."
        exit 101
        ;;
    exists)
        echo "error: crate $STUB_CRATE@0.15.0 already exists on crates.io index"
        exit 101
        ;;
    exists-other)
        echo "error: crate lattice-other@0.15.0 already exists on crates.io index"
        exit 101
        ;;
    *)
        echo "error: failed to verify package tarball"
        exit 101
        ;;
esac
"""


class PublishScriptRetries(_PublishScriptFixture):
    """A failing `cargo publish` for one crate: rate limits wait and retry, an
    already-published version is skipped, anything else stops the script."""

    def _fail(self, crate: str, mode: str, **extra: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        bin_dir = Path(self.env["PATH"].split(os.pathsep)[0])
        _write_exec(bin_dir / "cargo", _FAILING_CARGO)
        self.env.update(STUB_CRATE=crate, STUB_MODE=mode, STUB_STATE=str(self.call_log) + ".state", **extra)
        return self._run()

    def _sleeps(self, calls: list[str]) -> list[str]:
        return [c for c in calls if c.startswith("sleep ") and c != "sleep 30"]

    def test_rate_limited_crate_waits_and_retries(self) -> None:
        proc, calls = self._fail("lattice-quant", "429-once")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        publishes = self._publishes(calls)
        self.assertEqual(len(publishes), 27, calls)
        self.assertEqual(publishes.count("cargo publish -p lattice-quant"), 2, publishes)
        self.assertEqual(self._sleeps(calls), ["sleep 610"], calls)
        i = calls.index("sleep 610")
        self.assertEqual(calls[i - 1], "cargo publish -p lattice-quant")
        self.assertEqual(calls[i + 1], "cargo publish -p lattice-quant")
        self.assertEqual(publishes[-1], "cargo publish -p lattice-serve")
        self.assertIn("rate limited by the registry (attempt 1 of 12)", proc.stdout)

    def test_already_published_crate_is_skipped(self) -> None:
        proc, calls = self._fail("lattice-lora", "exists")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        publishes = self._publishes(calls)
        self.assertEqual(len(publishes), 26, calls)
        self.assertEqual(self._sleeps(calls), [], calls)
        self.assertIn("lattice-lora: this version is already on the registry; skipping", proc.stdout)

    def test_other_failure_stops_the_script(self) -> None:
        proc, calls = self._fail("lattice-arch", "other")
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        publishes = self._publishes(calls)
        self.assertEqual(publishes[-1], "cargo publish -p lattice-arch", publishes)
        self.assertEqual(publishes.count("cargo publish -p lattice-arch"), 1)
        self.assertEqual(self._sleeps(calls), [], calls)
        self.assertIn("lattice-arch: cargo publish failed", proc.stderr)

    def test_rate_limit_retries_are_bounded(self) -> None:
        proc, calls = self._fail("lattice-quant", "429-always", PUBLISH_RETRY_MAX="3")
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        publishes = self._publishes(calls)
        self.assertEqual(publishes.count("cargo publish -p lattice-quant"), 3, publishes)
        self.assertEqual(publishes[-1], "cargo publish -p lattice-quant", publishes)
        self.assertEqual(self._sleeps(calls), ["sleep 610", "sleep 610"], calls)

    def test_already_exists_for_another_crate_is_not_a_skip(self) -> None:
        proc, calls = self._fail("lattice-lora", "exists-other")
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertEqual(self._publishes(calls), ["cargo publish -p lattice-lora"], calls)


if __name__ == "__main__":
    unittest.main()
