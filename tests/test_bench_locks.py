#!/usr/bin/env python3
"""Regression tests for the bench-compare locking and ambient-load gates.

Two properties are pinned here, and both are about refusing rather than about
measuring.

The lock status file is PROOF, not a claim. scripts/bench-compare.sh runs the
measurement body under scripts/lib/bench-locks.py, which records its own PID
after taking both machine-wide locks. The body requires that PID to be one of
its own ancestors. Process ancestry comes from the OS, so a stale file, a copied
file, or a hand-written one cannot satisfy it -- which is what keeps the body
from being run directly and silently measuring without isolation. A check that
merely asserted the file exists would be a claim supplied by the thing being
checked.

The ambient-load gate REFUSES. A lock excludes peers on this machine; it says
nothing about how busy the machine is. A warning printed on a bench report is
read by nobody at the moment it matters, which is weeks later when someone
quotes the number.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "bench-compare.sh"
LIB = REPO / "scripts" / "lib"

STUB_CARGO = """#!/usr/bin/env bash
exit 0
"""


class _Sandbox:
    """A throwaway repo holding the shipping scripts, with locks redirected."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()

    def __enter__(self):
        tmp = self._tmp.name
        self.root = Path(tmp) / "repo"
        (self.root / "scripts").mkdir(parents=True)
        shutil.copy2(SCRIPT, self.root / "scripts" / SCRIPT.name)
        shutil.copytree(LIB, self.root / "scripts" / "lib")

        env_git = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                   "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run(["git", "init", "-q", "-b", "main", str(self.root)], check=True)
        for i in range(2):
            (self.root / f"f{i}.txt").write_text(str(i))
            subprocess.run(["git", "-C", str(self.root), "add", "-A"], check=True)
            subprocess.run(["git", "-C", str(self.root), "commit", "-qm", f"c{i}"],
                           check=True, env=env_git)

        locks = self.root / "scripts" / "lib" / "bench-locks.py"
        src = locks.read_text()
        for const in ("BENCH_WINDOW", "GPU_LOCK", "PENDING_DIR"):
            src = re.sub(rf'^{const} = "[^"]*"$', f'{const} = "{tmp}/{const.lower()}"',
                         src, flags=re.M)
        locks.write_text(src)

        bindir = Path(tmp) / "bin"
        bindir.mkdir()
        cargo = bindir / "cargo"
        cargo.write_text(STUB_CARGO)
        cargo.chmod(0o755)
        self.env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"}
        return self

    def __exit__(self, *exc):
        self._tmp.cleanup()
        return False

    def run(self, argv, **env):
        return subprocess.run(
            ["bash", *argv, "HEAD~1", "HEAD"],
            capture_output=True, text=True, env={**self.env, **env}, timeout=300)

    @property
    def entry(self):
        return str(self.root / "scripts" / SCRIPT.name)

    @property
    def impl(self):
        return str(self.root / "scripts" / "lib" / "bench-compare-impl.sh")

    @property
    def status(self):
        return self.root / ".cache" / "bench-locks-status.txt"


class LockProof(unittest.TestCase):
    def test_body_invoked_directly_refuses(self):
        """No status file at all means no proof of isolation.

        Mutation-sensitive: delete the verify_locks call from the body and this
        run proceeds to bench without either machine-wide lock held.
        """
        with _Sandbox() as sb:
            r = sb.run([sb.impl], BENCH_IDLE_FLOOR="0")
            self.assertEqual(r.returncode, 2, f"stderr:\n{r.stderr}")
            self.assertIn("no lock status", r.stderr)

    def test_status_file_naming_a_non_ancestor_is_refused(self):
        """A status file is not evidence unless its PID is really an ancestor.

        Mutation-sensitive: replace the ancestry walk with a file-exists check
        and this hand-written file satisfies it, which is the whole failure mode
        the walk exists to remove. PID 1 is chosen because it always exists and
        is never the parent chain of a test subprocess.
        """
        with _Sandbox() as sb:
            sb.status.parent.mkdir(parents=True, exist_ok=True)
            sb.status.write_text("supervisor_pid=1\nlock=fabricated\n")
            r = sb.run([sb.impl], BENCH_IDLE_FLOOR="0")
            self.assertEqual(r.returncode, 2, f"stderr:\n{r.stderr}")
            self.assertIn("not an ancestor", r.stderr)

    def test_supervised_run_reaches_the_measurement(self):
        """Through the entry point the proof holds and the body runs.

        Without this the two refusals above would also pass if the body refused
        unconditionally, which would be a broken script and a green suite.
        """
        with _Sandbox() as sb:
            r = sb.run([sb.entry], BENCH_IDLE_FLOOR="0")
            self.assertIn("Run conditions", r.stdout, f"stdout:\n{r.stdout}")
            self.assertIn("bench-window", r.stdout)


class AmbientLoadGate(unittest.TestCase):
    def test_below_floor_refuses_rather_than_warns(self):
        """An impossible floor stands in for a busy machine.

        No machine reports more than 100% idle, so a floor of 101 is failed by
        every environment this can run in, including an idle CI runner.

        Mutation-sensitive: turn the gate into a warning, or drop the exit-code
        check around the probe, and this exits 0 with the numbers printed.
        """
        with _Sandbox() as sb:
            r = sb.run([sb.entry], BENCH_IDLE_FLOOR="101")
            self.assertEqual(r.returncode, 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}")
            self.assertIn("was not quiet", r.stderr)

    def test_probe_reports_measured_idle_and_consumers(self):
        """The report must carry the conditions, not just a verdict."""
        out = subprocess.run(
            ["python3", str(LIB / "quiet-probe.py"), "--label", "unit", "--floor", "0"],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertRegex(out.stdout, r"\[quiet\] unit: idle [\d.]+% \(floor 0\.0%\)")
        self.assertIn("top:", out.stdout)


if __name__ == "__main__":
    unittest.main()
