#!/usr/bin/env python3
"""Regression tests for scripts/lib/bench-python.sh's PYTHON_BIN override.

bench_require_python3 used to overwrite an inherited PYTHON_BIN unconditionally
with a fresh PATH search (lattice#1551): a caller naming an interpreter
explicitly -- typically an absolute path reachable when a normal PATH search
is not, e.g. a non-interactive carrier whose PATH omits a uv-managed
interpreter -- had no way to make that choice stick, and got a PATH-resolved
interpreter it never asked for instead.

These tests drive the real scripts/lib/bench-python.sh function directly (by
sourcing it in a bash subprocess and calling bench_require_python3), not a
rewritten copy, and never invoke bench-compare.sh, bench-command.sh, or any
bench/cargo measurement.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH_PYTHON_SH = REPO / "scripts" / "lib" / "bench-python.sh"


def _run_resolver(
    env: dict[str, str], caller: str = "test-caller"
) -> subprocess.CompletedProcess[str]:
    script = f'source "{BENCH_PYTHON_SH}"\nbench_require_python3 "{caller}"\n'
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


class ExplicitPythonBinOverride(unittest.TestCase):
    """An inherited PYTHON_BIN wins over the PATH search (lattice#1551)."""

    def setUp(self):
        self.assertGreaterEqual(
            sys.version_info[:2],
            (3, 11),
            "the interpreter running this test must itself satisfy the "
            "harness's floor to stand in as a valid PYTHON_BIN",
        )

    def test_valid_python_bin_wins_over_a_restricted_path(self):
        """(a) A caller-supplied interpreter is used even when PATH is
        restricted to a directory that cannot itself satisfy the floor (real
        macOS /usr/bin/python3 is 3.9): the resolver's own stdout -- what a
        caller records as the resolved interpreter -- names the supplied
        path exactly, not a PATH-search result."""
        env = {"PATH": "/usr/bin:/bin", "PYTHON_BIN": sys.executable}
        result = _run_resolver(env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), sys.executable)

    def test_python_bin_below_the_floor_is_refused_not_silently_replaced(self):
        """(b) A stub interpreter that reports a pre-3.11 version is refused,
        and the refusal names both the floor and the version found. PATH
        still names a real >=3.11 interpreter (the one running this test)
        ahead of the stub, so a regression that fell back to searching PATH
        on a floor failure would succeed quietly instead of refusing --
        exactly the failure mode this pins."""
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / "fake-python"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                "case \"$1\" in\n"
                "  --version) echo 'Python 3.6.0'; exit 0 ;;\n"
                "  *) exit 1 ;;\n"
                "esac\n"
            )
            stub.chmod(0o755)
            bindir = Path(sys.executable).parent
            env = {"PATH": f"{bindir}:/usr/bin:/bin", "PYTHON_BIN": str(stub)}
            result = _run_resolver(env)
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertEqual("", result.stdout)
        self.assertIn("3.11", result.stderr)
        self.assertIn("3.6.0", result.stderr)
        self.assertIn(str(stub), result.stderr)

    def test_python_bin_naming_a_nonexecutable_path_is_refused(self):
        """A PYTHON_BIN pointing at something that cannot even be invoked is
        refused by name, never silently replaced by a PATH search."""
        with tempfile.TemporaryDirectory() as tmp:
            not_executable = Path(tmp) / "not-a-python"
            not_executable.write_text("not a script")
            env = {"PATH": "/usr/bin:/bin", "PYTHON_BIN": str(not_executable)}
            result = _run_resolver(env)
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertEqual("", result.stdout)
        self.assertIn(str(not_executable), result.stderr)
        self.assertIn("not an executable file", result.stderr)

    def test_unset_python_bin_resolves_by_path_search_exactly_as_before(self):
        """(c) With PYTHON_BIN unset, a qualifying interpreter reachable on
        PATH under one of the fixed candidate names is still found and
        returned, unchanged from before this override existed.

        The fixture must own the FIRST candidate bench_resolve_python3 tries
        (python3.13), not a later one. An Ubuntu CI runner (setup-python 3.11
        plus the distro's own /usr/bin/python3.12) satisfies the floor at the
        python3.12 candidate -- checked before python3.11 -- so a fixture
        naming only a python3.11 link is never reached there: the loop
        returns /usr/bin/python3.12 first and this test fails on that runner
        while passing on a machine whose /usr/bin lacks a qualifying
        interpreter (e.g. a stock macOS with only a pre-floor /usr/bin/python3).
        Owning the first-tried name removes the dependency on what /usr/bin
        happens to contain, on any platform."""
        with tempfile.TemporaryDirectory() as tmp:
            bindir = Path(tmp)
            link = bindir / "python3.13"
            link.symlink_to(sys.executable)
            env = {"PATH": f"{bindir}:/usr/bin:/bin"}
            result = _run_resolver(env)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(str(link), result.stdout.strip())

    def test_unset_python_bin_with_no_qualifying_interpreter_refuses_as_before(
        self,
    ):
        """(c) With PYTHON_BIN unset and nothing on PATH satisfying the
        floor, the original PATH-search refusal is unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            bindir = Path(tmp)
            for name in ("python3.13", "python3.12", "python3.11", "python3"):
                stub = bindir / name
                stub.write_text("#!/usr/bin/env bash\nexit 1\n")
                stub.chmod(0o755)
            env = {"PATH": f"{bindir}:/usr/bin:/bin"}
            result = _run_resolver(env)
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("no Python >= 3.11 found on PATH", result.stderr)


if __name__ == "__main__":
    unittest.main()
