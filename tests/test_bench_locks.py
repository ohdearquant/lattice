#!/usr/bin/env python3
"""Regression tests for bench-compare isolation and execution provenance.

The locking and ambient-load properties are about refusing rather than about
measuring.

The body refuses to measure unless it holds inherited lock descriptors that
prove -- by fstat identity against the recorded lock paths, plus a non-blocking
flock re-acquire and a probe open that must fail to lock -- that both
machine-wide locks are actually held. scripts/bench-compare.sh runs the
measurement body under scripts/lib/bench-locks.py --pass-lock-fds, which
records its own PID and both lock dispositions and hands the body the two
acquired descriptors.

A PID recorded in the lock status and found in this process's ancestry is only
a RELATION, never a proof: the file supplies the PID and the OS supplies the
chain, so a caller willing to record an ancestor's own PID -- its own shell's
included -- used to get through with neither lock held. That was fixed by
requiring the descriptor proof unconditionally: a status file with no inherited
LATTICE_BENCH_LOCK_FDS is refused outright, with no ancestry fallback.

A check that merely asserted the file exists would refuse none of the above.

The ambient-load gate REFUSES. A lock excludes peers on this machine; it says
nothing about how busy the machine is. A warning printed on a bench report is
read by nobody at the moment it matters, which is weeks later when someone
quotes the number.

The execution-provenance checks pin which source tree supplies the head arm and
which invoking-checkout gate grades it. Those facts differ when the positional
head ref changes, so both the early log and the pasted run-conditions block must
carry them.
"""
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "bench-compare.sh"
LIB = REPO / "scripts" / "lib"
GATE = REPO / "scripts" / "perf-bench-gate.py"
STATE_PROBE = LIB / "machine-state-probe.py"
HOST_ID = LIB / "bench-host-id.py"

STUB_CARGO = """#!/usr/bin/env bash
exit 0
"""

# Test helpers invoking real Git must disable repository hooks.
GIT = ("git", "-c", "core.hooksPath=/dev/null")

STUB_GOVERNOR = """#!/usr/bin/env python3
import json
import os
import sys
from datetime import UTC, datetime

label = sys.argv[sys.argv.index("--label") + 1]
calls = os.environ.get("MACHINE_STATE_CALLS_FILE")
if calls:
    with open(calls, "a") as handle:
        handle.write(label + "\\n")
rc = int(os.environ.get("STUB_GOVERNOR_RC", "0"))
print(json.dumps({
    "schema": "lattice-machine-state-v1",
    "label": label,
    "captured_at_utc": datetime.now(UTC).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z"),
    "power": {"status": "measured", "source": "fixture", "state": "ac"},
    "thermal": {
        "status": "measured",
        "source": "fixture",
        "state": "nominal" if rc == 0 else "serious",
    },
    "idle": {
        "status": "measured",
        "source": "fixture",
        "seconds": 30.0,
    },
    "gate": {
        "status": "passed" if rc == 0 else "blocked",
        "cooldown_seconds": 30.0,
        "afk_threshold_seconds": 30.0,
        **({"kill_switch": "clear"} if rc == 0 else {"reason": "fixture block"}),
    },
}, separators=(",", ":"), sort_keys=True))
raise SystemExit(rc)
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
        shutil.copy2(GATE, self.root / "scripts" / GATE.name)
        shutil.copytree(LIB, self.root / "scripts" / "lib")
        shutil.copy2(REPO / ".gitignore", self.root / ".gitignore")
        governor = self.root / "scripts" / "perf_governor.py"
        governor.write_text(STUB_GOVERNOR)
        governor.chmod(0o755)
        quiet_probe = self.root / "scripts" / "lib" / "quiet-probe.py"
        quiet_probe.write_text(
            "#!/usr/bin/env python3\n"
            "import argparse\n"
            "import os\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--label', required=True)\n"
            "p.add_argument('--floor', type=float, "
            "default=float(os.environ.get('BENCH_IDLE_FLOOR', '70')))\n"
            "a = p.parse_args()\n"
            "ok = a.floor <= 100.0\n"
            "print(f'[quiet] {a.label}: idle 100.0% (floor {a.floor:.1f}%) '"
            "      f'{\"ok\" if ok else \"BELOW FLOOR\"} | top: fixture 0.0%')\n"
            "raise SystemExit(0 if ok else 1)\n"
        )
        machine_probe = self.root / "scripts" / "lib" / "machine-state-probe.py"
        machine_probe.write_text(
            "#!/usr/bin/env python3\n"
            "import datetime, json, sys\n"
            "label = sys.argv[sys.argv.index('--label') + 1]\n"
            "print(json.dumps({'schema':'lattice-machine-state-v1','label':label,"
            "'captured_at_utc':datetime.datetime.now(datetime.UTC)"
            ".strftime('%Y-%m-%dT%H:%M:%SZ'),"
            "'power':{'status':'unavailable','reason':'fixture'},"
            "'thermal':{'status':'unavailable','reason':'fixture'},"
            "'idle':{'status':'unavailable','reason':'fixture'}},"
            "separators=(',', ':'), sort_keys=True))\n"
        )

        env_git = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                   "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run([*GIT, "init", "-q", "-b", "main", str(self.root)], check=True)
        (self.root / "Cargo.lock").write_text(
            'version = 4\n\n'
            '[[package]]\n'
            'name = "criterion"\n'
            'version = "0.5.1"\n'
        )
        subprocess.run(
            [*GIT, "-C", str(self.root), "add", "-f", "Cargo.lock"],
            check=True,
        )
        for i in range(2):
            (self.root / f"f{i}.txt").write_text(str(i))
            subprocess.run([*GIT, "-C", str(self.root), "add", "-A"], check=True)
            subprocess.run([*GIT, "-C", str(self.root), "commit", "-qm", f"c{i}"],
                           check=True, env=env_git)

        locks = self.root / "scripts" / "lib" / "bench-locks.py"
        src = locks.read_text()
        for const in ("BENCH_WINDOW", "GPU_LOCK", "PENDING_DIR"):
            src = re.sub(rf'^{const} = "[^"]*"$', f'{const} = "{tmp}/{const.lower()}"',
                         src, flags=re.M)
        locks.write_text(src)
        subprocess.run(
            [*GIT, "-C", str(self.root), "add", "scripts/lib/bench-locks.py"],
            check=True,
        )
        subprocess.run(
            [*GIT, "-C", str(self.root), "commit", "-qm", "fixture lock paths"],
            check=True,
            env=env_git,
        )

        self.bindir = Path(tmp) / "bin"
        self.bindir.mkdir()
        cargo = self.bindir / "cargo"
        cargo.write_text(STUB_CARGO)
        cargo.chmod(0o755)
        self.machine_calls = Path(tmp) / "machine-state-calls.txt"
        self.env = {
            **os.environ,
            "PATH": f"{self.bindir}:{os.environ['PATH']}",
            "LATTICE_BENCH_HOST_ID_FILE": f"{tmp}/bench-host-id",
            "MACHINE_STATE_CALLS_FILE": str(self.machine_calls),
        }
        return self

    def __exit__(self, *exc):
        self._tmp.cleanup()
        return False

    def run(self, argv, *, base_ref="HEAD~1", head_ref="HEAD", **env):
        return subprocess.run(
            ["bash", *argv, base_ref, head_ref],
            capture_output=True, text=True, env={**self.env, **env}, timeout=300)

    def force_platform(self, name):
        uname = self.bindir / "uname"
        uname.write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$#" -eq 0 ] || [ "${1:-}" = "-s" ]; then\n'
            f"  echo {name}\n"
            "else\n"
            '  exec /usr/bin/uname "$@"\n'
            "fi\n"
        )
        uname.chmod(0o755)

    def machine_state_labels(self):
        if not self.machine_calls.exists():
            return []
        return self.machine_calls.read_text().splitlines()

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

    def test_status_file_without_inherited_lock_fds_is_refused(self):
        """A status file without descriptor capabilities is only text.

        Mutation-sensitive: removing the descriptor precondition lets this
        caller-controlled receipt reach the measurement body without either
        machine-wide lock.
        """
        with _Sandbox() as sb:
            sb.status.parent.mkdir(parents=True, exist_ok=True)
            sb.status.write_text("supervisor_pid=1\nlock=fabricated\n")
            r = sb.run([sb.impl], BENCH_IDLE_FLOOR="0")
            self.assertEqual(r.returncode, 2, f"stderr:\n{r.stderr}")
            self.assertIn("LATTICE_BENCH_LOCK_FDS", r.stderr)
            self.assertNotIn("Run conditions", r.stdout)

    def test_deliberately_recorded_ancestor_pid_alone_is_refused(self):
        """A caller-supplied ancestor PID is a relation, never a proof.

        A caller who records a PID that really is one of its ancestors -- its
        own shell, here -- used to pass the check with no lock held, which is
        exactly the stale-environment bypass: an exported status made an ordinary
        invocation failure read as a supervised run. The fix requires
        the inherited LATTICE_BENCH_LOCK_FDS descriptor proof unconditionally,
        so this receipt-only invocation must now be refused before the body
        ever prints its run-conditions banner.

        Mutation-sensitive: reintroduce the ancestry-walk fallback and this run
        reaches "Run conditions" again with neither lock held.
        """
        with _Sandbox() as sb:
            sb.status.parent.mkdir(parents=True, exist_ok=True)
            script = (
                f'echo "supervisor_pid=$$" > {sb.status}\n'
                f'echo "lock=fabricated, nothing is held" >> {sb.status}\n'
                f'bash {sb.impl} HEAD~1 HEAD\n'
                'rc=$?\n'
                'exit "$rc"\n'
            )
            r = subprocess.run(
                ["bash", "-c", script], capture_output=True, text=True,
                env={**sb.env, "BENCH_IDLE_FLOOR": "0"}, timeout=300)
            self.assertEqual(r.returncode, 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}")
            self.assertIn("LATTICE_BENCH_LOCK_FDS", r.stderr)
            self.assertNotIn("Run conditions", r.stdout)

    def test_supervised_run_reaches_the_measurement(self):
        """Through the entry point the check passes and the body runs.

        Without this the two refusals above would also pass if the body refused
        unconditionally, which would be a broken script and a green suite.
        """
        with _Sandbox() as sb:
            r = sb.run([sb.entry], BENCH_IDLE_FLOOR="0")
            self.assertIn("Run conditions", r.stdout, f"stdout:\n{r.stdout}")
            self.assertIn("bench-window", r.stdout)


class HeadModeReporting(unittest.TestCase):
    """The log and pasted run-conditions block identify the resolved head arm."""

    def assert_provenance_in_header_and_report(self, stdout, head_line, gate_line):
        header, separator, report = stdout.partition("=== Run conditions ===")
        self.assertTrue(separator, stdout)
        for section in (header, report):
            self.assertIn(head_line, section)
            self.assertIn(gate_line, section)

    def test_head_ref_uses_and_reports_the_invoking_checkout(self):
        """Mutation-sensitive: removing either provenance emission leaves one
        half of this header/report assertion without the in-place mode."""
        with _Sandbox() as sb:
            r = sb.run([sb.entry], BENCH_IDLE_FLOOR="0")
            self.assertEqual(r.returncode, 0, f"stderr:\n{r.stderr}")
            self.assert_provenance_in_header_and_report(
                r.stdout,
                "  head arm: detached snapshot worktree",
                "  gate: scripts/perf-bench-gate.py from the invoking checkout",
            )
            self.assertNotIn("head arm: in-place", r.stdout)

    def test_explicit_head_ref_uses_and_reports_a_detached_worktree(self):
        """Mutation-sensitive: collapsing the mode selection to in-place makes
        the expected worktree provenance disappear from both report locations."""
        with _Sandbox() as sb:
            r = sb.run(
                [sb.entry],
                base_ref="HEAD~1",
                head_ref="HEAD~1",
                BENCH_IDLE_FLOOR="0",
            )
            self.assertEqual(r.returncode, 0, f"stderr:\n{r.stderr}")
            self.assert_provenance_in_header_and_report(
                r.stdout,
                "  head arm: detached worktree",
                "  gate: scripts/perf-bench-gate.py from the invoking checkout",
            )
            self.assertNotIn("head arm: in-place", r.stdout)

    def test_in_place_head_refuses_uncommitted_source(self):
        with _Sandbox() as sb:
            (sb.root / "f1.txt").write_text("dirty")
            r = sb.run([sb.entry], BENCH_IDLE_FLOOR="0")
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("not commit-clean", r.stderr)

    def test_detached_head_ignores_dirty_invoking_worktree(self):
        with _Sandbox() as sb:
            (sb.root / "f1.txt").write_text("dirty")
            r = sb.run(
                [sb.entry],
                base_ref="HEAD~1",
                head_ref="HEAD~1",
                BENCH_IDLE_FLOOR="0",
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("head arm: detached worktree", r.stdout)

    def test_in_place_head_refuses_source_changed_during_measurement(self):
        with _Sandbox() as sb, tempfile.TemporaryDirectory() as shim:
            cargo = Path(shim) / "cargo"
            cargo.write_text(
                "#!/usr/bin/env bash\n"
                f'if [[ "$PWD" == *"/.cache/bench-compare-head" ]]; then\n'
                f'  printf "%s\\n" changed > "{sb.root}/f1.txt"\n'
                "fi\n"
                "exit 0\n"
            )
            cargo.chmod(0o755)
            r = sb.run(
                [sb.entry],
                BENCH_IDLE_FLOOR="0",
                PATH=f"{shim}:{sb.env['PATH']}",
            )
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("not commit-clean", r.stderr)

    def test_restored_source_race_cannot_change_snapshot_measurement(self):
        with _Sandbox() as sb, tempfile.TemporaryDirectory() as shim:
            cargo = Path(shim) / "cargo"
            observed = Path(shim) / "observed"
            cargo.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f'if [[ "$PWD" == *"/.cache/bench-compare-head" ]] '
                '&& [ "${1:-}" != "--version" ]; then\n'
                f'  printf "%s:%s\\n" "$PWD" "$(cat f1.txt)" >> "{observed}"\n'
                f'  original="$(cat "{sb.root}/f1.txt")"\n'
                f'  printf "%s" mutated-during-run > "{sb.root}/f1.txt"\n'
                f'  printf "%s" "$original" > "{sb.root}/f1.txt"\n'
                "fi\n"
                "exit 0\n"
            )
            cargo.chmod(0o755)
            r = sb.run(
                [sb.entry],
                BENCH_IDLE_FLOOR="0",
                PATH=f"{shim}:{sb.env['PATH']}",
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual((sb.root / "f1.txt").read_text(), "1")
            lines = observed.read_text().splitlines()
            self.assertTrue(lines, r.stdout)
            self.assertTrue(
                all(line.endswith(":1") for line in lines),
                "\n".join(lines),
            )

    def test_in_place_head_refuses_commit_changed_during_measurement(self):
        with _Sandbox() as sb, tempfile.TemporaryDirectory() as shim:
            cargo = Path(shim) / "cargo"
            cargo.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f'if [[ "$PWD" == *"/.cache/bench-compare-head" ]] '
                '&& [ "${1:-}" != "--version" ] '
                f'&& [ "$(cat "{sb.root}/f1.txt")" != "committed-during-run" ]; then\n'
                f'  printf "%s\\n" committed-during-run > "{sb.root}/f1.txt"\n'
                f'  cd "{sb.root}"\n'
                "  git -c core.hooksPath=/dev/null -c user.name=t -c user.email=t "
                "add f1.txt\n"
                "  git -c core.hooksPath=/dev/null -c user.name=t -c user.email=t "
                "commit -qm committed-during-run\n"
                "fi\n"
                "exit 0\n"
            )
            cargo.chmod(0o755)
            r = sb.run(
                [sb.entry],
                BENCH_IDLE_FLOOR="0",
                PATH=f"{shim}:{sb.env['PATH']}",
            )
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("HEAD commit changed during the run", r.stderr)


class RunProvenanceHandoff(unittest.TestCase):
    def test_supervised_run_writes_complete_three_phase_handoff(self):
        """The Markdown gate receives auditable machine and phase conditions."""
        with _Sandbox() as sb:
            sb.force_platform("Darwin")
            r = sb.run([sb.entry], BENCH_IDLE_FLOOR="0")
            self.assertEqual(r.returncode, 0, f"stderr:\n{r.stderr}")
            provenance = sb.root / ".cache" / "bench-run-provenance.txt"
            self.assertTrue(provenance.is_file(), r.stdout)
            lines = provenance.read_text().splitlines()

            for prefix in (
                "schema=lattice-bench-provenance-v1",
                "started_utc=",
                "finished_utc=",
                "host_id=local-random:",
                "os=",
                "base_ref=HEAD~1",
                "base_sha=",
                "head_ref=HEAD",
                "head_sha=",
                "head_mode=detached-worktree",
                "base_rustc=",
                "head_rustc=",
                "base_cargo=",
                "head_cargo=",
                "base_criterion=",
                "head_criterion=",
                "criterion_mode=quick",
                "baseline_name=compare-base",
                "targets=lattice-inference:elementwise_cpu_bench, lattice-embed:simd",
                "inference_features=<none>",
                "filters=inference='<all>' embed='<all>'",
                "enforcement=report-only",
                "lock=",
            ):
                self.assertTrue(
                    any(line.startswith(prefix) for line in lines),
                    f"missing provenance prefix {prefix!r}:\n"
                    + "\n".join(lines),
                )
            self.assertEqual(
                sum(line.startswith("ambient=") for line in lines),
                3,
                "\n".join(lines),
            )
            self.assertEqual(
                sum(line.startswith("machine_state=") for line in lines),
                3,
                "\n".join(lines),
            )
            states = [
                json.loads(line.removeprefix("machine_state="))
                for line in lines
                if line.startswith("machine_state=")
            ]
            self.assertEqual(
                [state["label"] for state in states],
                ["before base", "between phases", "after head"],
            )
            self.assertTrue(
                all(
                    state["power"]["status"] in ("measured", "unavailable")
                    and state["thermal"]["status"] in ("measured", "unavailable")
                    and state["idle"]["status"] in ("measured", "unavailable")
                    for state in states
                )
            )
            self.assertNotIn(os.uname().nodename, provenance.read_text())
            head_mode = next(
                line for line in lines if line.startswith("head_mode=")
            )
            self.assertNotIn(" at ", head_mode)
            self.assertIn("<summary>Run provenance</summary>", r.stdout)
            self.assertIn("host_id=local-random:", r.stdout)
            self.assertIn("HID idle 30.0s via fixture", r.stdout)
            self.assertIn(
                "gate passed (cooldown 30.0s, AFK floor 30.0s, "
                "kill-switch clear)",
                r.stdout,
            )
            self.assertNotIn("unsuitable as benchmark evidence", r.stdout)


class HostIdentifier(unittest.TestCase):
    def test_random_identifier_is_stable_and_contains_no_hostname(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "host-id"
            env = {**os.environ, "LATTICE_BENCH_HOST_ID_FILE": str(path)}
            first = subprocess.run(
                ["python3", str(HOST_ID)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.strip()
            second = subprocess.run(
                ["python3", str(HOST_ID)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            ).stdout.strip()
            self.assertRegex(first, r"^local-random:[0-9a-f]{32}$")
            self.assertEqual(first, second)
            self.assertNotIn(os.uname().nodename, first)

    def test_malformed_persisted_identifier_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "host-id"
            path.write_text("not-a-valid-id\n")
            result = subprocess.run(
                ["python3", str(HOST_ID)],
                capture_output=True,
                text=True,
                env={**os.environ, "LATTICE_BENCH_HOST_ID_FILE": str(path)},
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("does not contain one 32-hex", result.stderr)


class MachineStateParsers(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "machine_state_probe", str(STATE_PROBE)
        )
        self.probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.probe)

    def test_power_source_is_measured_only_when_pmset_names_it(self):
        ac = self.probe.parse_macos_power("Now drawing from 'AC Power'\n")
        battery = self.probe.parse_macos_power("Now drawing from 'Battery Power'\n")
        missing = self.probe.parse_macos_power("Battery information unavailable\n")
        self.assertEqual(ac["state"], "ac")
        self.assertEqual(battery["state"], "battery")
        self.assertEqual(missing["status"], "unavailable")

    def test_thermal_error_text_is_not_reported_as_nominal(self):
        result = self.probe.parse_macos_thermal(
            "Error: Failed to get thermal warning level with error code 0xe00002bc\n"
        )
        self.assertEqual(result["status"], "unavailable")
        self.assertNotEqual(result.get("state"), "nominal")

    def test_cpu_speed_limit_distinguishes_nominal_and_throttled(self):
        nominal = self.probe.parse_macos_thermal("CPU_Speed_Limit = 100\n")
        throttled = self.probe.parse_macos_thermal("CPU_Speed_Limit = 75\n")
        self.assertEqual(nominal["state"], "nominal")
        self.assertEqual(throttled["state"], "throttled")
        self.assertEqual(throttled["cpu_speed_limit_percent"], 75)

    def test_explicit_no_warning_pair_is_nominal(self):
        result = self.probe.parse_macos_thermal(
            "Note: No thermal warning level has been recorded\n"
            "Note: No performance warning level has been recorded\n"
        )
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["state"], "nominal")

    def test_current_pmset_error_falls_back_to_process_info(self):
        pmset_error = (
            "Error:Failed to get thermal warning level with error code 0xe00002bc\n"
            "Error: Failed to get performance warning level with error code 0xe00002bc\n"
            "Error: No CPU power status with error code 0xe00002bc\n"
        )
        with mock.patch.object(
            self.probe, "run_pmset", return_value=(0, pmset_error)
        ):
            with mock.patch.object(
                self.probe,
                "read_process_info_thermal",
                return_value={
                    "status": "measured",
                    "source": "ProcessInfo.thermalState",
                    "state": "nominal",
                },
            ):
                state = self.probe.read_macos_thermal()
        self.assertEqual(state["status"], "measured")
        self.assertEqual(state["state"], "nominal")
        self.assertEqual(state["source"], "ProcessInfo.thermalState")
        self.assertIn("fallback_reason", state)

    def test_two_unavailable_thermal_sources_remain_unavailable(self):
        with mock.patch.object(
            self.probe, "run_pmset", return_value=(1, "")
        ):
            with mock.patch.object(
                self.probe,
                "read_process_info_thermal",
                return_value=self.probe.unavailable("Foundation unavailable"),
            ):
                state = self.probe.read_macos_thermal()
        self.assertEqual(state["status"], "unavailable")
        self.assertNotEqual(state.get("state"), "nominal")

    def test_process_info_only_accepts_nominal_enum_as_nominal(self):
        self.assertEqual(
            self.probe.thermal_state_from_raw(0)["state"], "nominal"
        )
        for raw, state in ((1, "fair"), (2, "serious"), (3, "critical")):
            with self.subTest(raw=raw):
                self.assertEqual(
                    self.probe.thermal_state_from_raw(raw)["state"], state
                )
        self.assertEqual(
            self.probe.thermal_state_from_raw(4)["status"], "unavailable"
        )

    def test_hid_idle_parser_is_fail_closed(self):
        measured = self.probe.parse_macos_idle(
            '    | |   "HIDIdleTime" = 31250000000\n'
        )
        missing = self.probe.parse_macos_idle("IOHIDSystem unavailable\n")
        self.assertEqual(measured["seconds"], 31.25)
        self.assertEqual(missing["status"], "unavailable")


class MachineStateGate(unittest.TestCase):
    def test_macos_checkpoints_gate_all_three_measurement_boundaries(self):
        with _Sandbox() as sb:
            sb.force_platform("Darwin")
            r = sb.run([sb.entry], BENCH_IDLE_FLOOR="0")
            self.assertEqual(r.returncode, 0, f"stderr:\n{r.stderr}")
            self.assertEqual(
                sb.machine_state_labels(),
                ["before base", "between phases", "after head"],
            )
            _, separator, report = r.stdout.partition("=== Run conditions ===")
            self.assertTrue(separator, r.stdout)
            provenance = sb.root / ".cache" / "bench-run-provenance.txt"
            states = [
                json.loads(line.removeprefix("machine_state="))
                for line in provenance.read_text().splitlines()
                if line.startswith("machine_state=")
            ]
            self.assertEqual(
                [state["label"] for state in states],
                ["before base", "between phases", "after head"],
            )
            self.assertTrue(
                all(
                    state["gate"]["status"] == "passed"
                    and state["gate"]["cooldown_seconds"] == 30.0
                    and state["gate"]["afk_threshold_seconds"] == 30.0
                    for state in states
                )
            )
            self.assertIn('"gate":', report)

    def test_checkpoint_failure_refuses_before_measurement(self):
        with _Sandbox() as sb:
            sb.force_platform("Darwin")
            r = sb.run(
                [sb.entry, "--fail-on-regression"],
                BENCH_IDLE_FLOOR="0",
                STUB_GOVERNOR_RC="2",
            )
            self.assertEqual(r.returncode, 2, f"stdout:\n{r.stdout}")
            self.assertEqual(sb.machine_state_labels(), ["before base"])
            self.assertIn("machine-state checkpoint 'before base' failed", r.stderr)
            self.assertNotIn("Building + benching BASE", r.stdout)

    def test_blocked_macos_checkpoint_reports_or_refuses_by_enforcement_mode(self):
        with _Sandbox() as sb:
            sb.force_platform("Darwin")
            report_only = sb.run(
                [sb.entry],
                BENCH_IDLE_FLOOR="0",
                STUB_GOVERNOR_RC="2",
            )
            self.assertEqual(
                report_only.returncode,
                0,
                f"stdout:\n{report_only.stdout}\nstderr:\n{report_only.stderr}",
            )
            self.assertIn("Building + benching BASE", report_only.stdout)
            self.assertIn("=== Run conditions ===", report_only.stdout)
            self.assertIn("gate blocked (fixture block)", report_only.stdout)
            self.assertIn(
                "unsuitable as benchmark evidence",
                report_only.stdout,
            )
            provenance = sb.root / ".cache" / "bench-run-provenance.txt"
            states = [
                json.loads(line.removeprefix("machine_state="))
                for line in provenance.read_text().splitlines()
                if line.startswith("machine_state=")
            ]
            self.assertEqual(len(states), 3)
            self.assertTrue(
                all(state["gate"]["status"] == "blocked" for state in states)
            )

        with _Sandbox() as sb:
            sb.force_platform("Darwin")
            enforcing = sb.run(
                [sb.entry, "--fail-on-regression"],
                BENCH_IDLE_FLOOR="0",
                STUB_GOVERNOR_RC="2",
            )
            self.assertEqual(enforcing.returncode, 2, enforcing.stdout)
            self.assertEqual(sb.machine_state_labels(), ["before base"])
            self.assertIn(
                "machine-state checkpoint 'before base' failed",
                enforcing.stderr,
            )
            self.assertNotIn("Building + benching BASE", enforcing.stdout)


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


class IdleParsers(unittest.TestCase):
    """Both platform parsers, against fixtures rather than the live machine.

    Only one of these two branches runs on any given host, so a test that reads
    the real machine leaves the other parser permanently unexercised. CI is
    Linux-only, which means the macOS branch -- the one every local bench on
    this project actually takes -- would otherwise be covered nowhere.
    """

    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "quiet_probe", str(LIB / "quiet-probe.py"))
        self.qp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.qp)

    def test_macos_idle_is_the_field_named_idle_not_the_first_percentage(self):
        """Mutation-sensitive: anchor the regex on position instead of the word
        `idle` and this returns 12.5, the busiest field, as the idle figure."""
        sample = (
            "Processes: 700 total\n"
            "CPU usage: 12.50% user, 6.25% sys, 81.25% idle\n"
        )
        self.assertAlmostEqual(self.qp.parse_top_idle(sample), 81.25)

    def test_macos_idle_comes_from_the_last_sample_not_the_first(self):
        """Mutation-sensitive: take hits[0] and this returns 99.0, top's
        since-boot average, which reads quiet on a machine that is busy now."""
        sample = (
            "CPU usage: 0.50% user, 0.50% sys, 99.00% idle\n"
            "CPU usage: 70.00% user, 10.00% sys, 20.00% idle\n"
        )
        self.assertAlmostEqual(self.qp.parse_top_idle(sample), 20.0)

    def test_macos_unparseable_output_raises_rather_than_defaulting(self):
        for junk in ("", "top: command produced nothing", "CPU usage: n/a"):
            with self.assertRaises(RuntimeError):
                self.qp.parse_top_idle(junk)

    def test_linux_idle_is_the_delta_share_not_the_absolute(self):
        """Half the jiffies in the interval went to idle, so 50%, even though
        both samples are dominated by since-boot idle.

        Mutation-sensitive: compute from the second sample alone and this
        returns about 91%, because a long-idle machine's totals swamp the
        interval that is actually being measured.
        """
        line0 = "cpu 1000 0 1000 10000 0 0 0 0"
        line1 = "cpu 1050 0 1050 10100 0 0 0 0"
        self.assertAlmostEqual(self.qp.linux_idle_pct(line0, line1), 50.0)

    def test_linux_counts_iowait_as_idle(self):
        """Pinned deliberately: the CPU is available to the bench during iowait.

        Mutation-sensitive in the other direction too -- drop the iowait term
        and this reads 50% instead of 100%.
        """
        line0 = "cpu 1000 0 1000 10000 500 0 0 0"
        line1 = "cpu 1000 0 1000 10050 550 0 0 0"
        self.assertAlmostEqual(self.qp.linux_idle_pct(line0, line1), 100.0)

    def test_linux_non_advancing_counters_raise(self):
        """A repeated sample is not evidence of an idle machine."""
        line = "cpu 1000 0 1000 10000 0 0 0 0"
        with self.assertRaises(RuntimeError):
            self.qp.linux_idle_pct(line, line)


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
        spec = importlib.util.spec_from_file_location(
            "quiet_probe_ambient", str(LIB / "quiet-probe.py")
        )
        qp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qp)
        argv = ["quiet-probe.py", "--label", "unit", "--floor", "0"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(qp, "idle_percent", return_value=99.5), \
                mock.patch.object(
                    qp, "top_consumers", return_value="fixture 0.0%"
                ), mock.patch(
                    "sys.stdout", new_callable=io.StringIO
                ) as stdout:
            self.assertEqual(qp.main(), 0)
        self.assertRegex(
            stdout.getvalue(), r"\[quiet\] unit: idle 99\.5% \(floor 0\.0%\)"
        )
        self.assertIn("top: fixture 0.0%", stdout.getvalue())

    def test_probe_appends_versioned_ambient_sample(self):
        spec = importlib.util.spec_from_file_location(
            "quiet_probe_jsonl", str(LIB / "quiet-probe.py")
        )
        qp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qp)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "ambient.jsonl"
            argv = [
                "quiet-probe.py",
                "--label",
                "between phases",
                "--phase",
                "between",
                "--jsonl-out",
                str(output),
                "--floor",
                "0",
            ]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(qp, "idle_percent", return_value=87.25), \
                    mock.patch.object(qp, "top_consumers", return_value="fixture"):
                self.assertEqual(qp.main(), 0)
            self.assertEqual(
                json.loads(output.read_text()),
                {
                    "schema": "perf-ambient-sample/v1",
                    "phase": "between",
                    "idle_pct": 87.25,
                },
            )


if __name__ == "__main__":
    unittest.main()
