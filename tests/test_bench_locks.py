#!/usr/bin/env python3
"""Regression tests for the bench-compare locking and ambient-load gates.

Two properties are pinned here, and both are about refusing rather than about
measuring.

The body refuses to measure unless a live supervisor is above it.
scripts/bench-compare.sh runs the measurement body under
scripts/lib/bench-locks.py, which records its own PID after taking both
machine-wide locks; the body requires that PID to be one of its own ancestors.

Stated exactly, because the tempting claim is one word wider than the truth: the
file supplies the PID and the OS supplies the chain, so the check establishes a
RELATION. It refuses a status file left over from a finished run, one copied
from a different run, and accidental direct invocation -- the ways this actually
gets run without isolation. It does not stop a caller who deliberately records
an ancestor's PID. Closing that needs the lock descriptor rather than a PID, and
arrives with the nested-acquirer work.

A check that merely asserted the file exists would refuse none of the above.

The ambient-load gate REFUSES. A lock excludes peers on this machine; it says
nothing about how busy the machine is. A warning printed on a bench report is
read by nobody at the moment it matters, which is weeks later when someone
quotes the number.
"""
import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import time
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


class LockPrecondition(unittest.TestCase):
    def test_body_invoked_directly_refuses(self):
        """No status file at all means no evidence of isolation.

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

    def test_deliberately_recorded_ancestor_pid_is_accepted(self):
        """The boundary of the guard, pinned as a fact rather than left in prose.

        A caller who records a PID that really is one of its ancestors -- its own
        shell, here -- passes the check with no lock held. That is the limit of
        what a PID can establish: the file supplies the number, the OS confirms
        only the relation.

        This is a characterization test, not a wish. It exists so the comment
        describing that limit cannot drift away from the code: strengthen the
        guard to close this and the test fails, which is the signal to update
        every place the limit is described.
        """
        with _Sandbox() as sb:
            sb.status.parent.mkdir(parents=True, exist_ok=True)
            script = (
                f'echo "supervisor_pid=$$" > {sb.status}\n'
                f'echo "lock=fabricated, nothing is held" >> {sb.status}\n'
                # The recording shell must stay alive as the parent. Two ways to
                # lose it, both of which make the body inherit that PID as its
                # OWN and get refused (the walk starts at PPID, so self never
                # matches): an explicit `exec`, and bash's implicit exec of the
                # LAST simple command in a -c script. The trailing statement
                # below defeats the second. An interactive operator typing the
                # command hits neither, which is the case being characterized.
                f'bash {sb.impl} HEAD~1 HEAD\n'
                'rc=$?\n'
                'exit "$rc"\n'
            )
            r = subprocess.run(
                ["bash", "-c", script], capture_output=True, text=True,
                env={**sb.env, "BENCH_IDLE_FLOOR": "0"}, timeout=300)
            self.assertNotIn("not an ancestor", r.stderr)
            self.assertIn("Run conditions", r.stdout, f"stderr:\n{r.stderr}")

    def test_supervised_run_reaches_the_measurement(self):
        """Through the entry point the check passes and the body runs.

        Without this the two refusals above would also pass if the body refused
        unconditionally, which would be a broken script and a green suite.
        """
        with _Sandbox() as sb:
            r = sb.run([sb.entry], BENCH_IDLE_FLOOR="0")
            self.assertIn("Run conditions", r.stdout, f"stdout:\n{r.stdout}")
            self.assertIn("bench-window", r.stdout)


class ContentionDiagnostics(unittest.TestCase):
    def test_holder_report_never_includes_command_line_arguments(self):
        """Waiting on a lock must not print other processes' arguments.

        The contention message goes to stderr, which on this repository's
        workflows lands in publicly readable job logs. Arguments carry tokens,
        keys and connection strings, so a diagnostic that prints the full
        command line discloses them.

        Mutation-sensitive: change the executable-name lookup back to
        `ps -o command=` and the marker below appears in the output.
        """
        if shutil.which("lsof") is None:
            self.skipTest("lsof unavailable; the diagnostic returns nothing")

        spec = importlib.util.spec_from_file_location(
            "bench_locks", str(LIB / "bench-locks.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        marker = "PRETEND-CREDENTIAL-do-not-log"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "held.lock")
            open(path, "w").close()
            holder = subprocess.Popen(
                ["python3", "-c",
                 "import sys,time; f=open(sys.argv[1]); time.sleep(20)",
                 path, marker])
            try:
                for _ in range(40):
                    found = mod._openers(path)
                    if any(pid == holder.pid for pid, _ in found):
                        break
                    time.sleep(0.25)
                else:
                    self.skipTest("holder never appeared in lsof output")
                rendered = mod._describe_contention(path)
                self.assertIn(str(holder.pid), rendered)
                self.assertNotIn(marker, rendered)
                self.assertNotIn(path, rendered)
            finally:
                holder.kill()
                holder.wait()


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
