#!/usr/bin/env python3
"""Tests for enforced local measurement supervision: locks, idle checks and the GPU handoff."""

from __future__ import annotations

import fcntl
import importlib.util
import io
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent


class _SupervisorSandbox:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()

    def __enter__(self):
        self.root = Path(self.tmp.name) / "repo"
        lib = self.root / "scripts" / "lib"
        lib.mkdir(parents=True)
        for name in (
            "bench_supervision.py",
            "bench-locks.py",
            "quiet-probe.py",
            "bench-python.sh",
            "ensure-noindex-marker.sh",
        ):
            shutil.copy2(REPO / "scripts" / "lib" / name, lib / name)
        shutil.copy2(
            REPO / "scripts" / "lib" / "bench-supervision.sh",
            lib / "bench-supervision.sh",
        )

        self.bench_lock = Path(self.tmp.name) / "bench-window.lock"
        self.gpu_lock = Path(self.tmp.name) / "metal-gpu.lock"
        self.pending = Path(self.tmp.name) / "pending"
        lock_source = (lib / "bench-locks.py").read_text()
        replacements = {
            "BENCH_WINDOW": str(self.bench_lock),
            "GPU_LOCK": str(self.gpu_lock),
            "PENDING_DIR": str(self.pending),
        }
        for name, value in replacements.items():
            lock_source = re.sub(
                rf'^{name} = "[^"]*"$',
                f'{name} = "{value}"',
                lock_source,
                flags=re.M,
            )
        (lib / "bench-locks.py").write_text(lock_source)
        self.helper = lib / "bench_supervision.py"
        self.target_env = mock.patch.dict(
            os.environ, {"CARGO_TARGET_DIR": str(self.root / "target")})
        self.target_env.start()
        os.environ.pop("CARGO_BUILD_TARGET_DIR", None)
        return self

    def __exit__(self, *exc):
        self.target_env.stop()
        self.tmp.cleanup()
        return False

    def run(
        self,
        command: list[str],
        *,
        entrypoint: bool = False,
        **env: str,
    ) -> subprocess.CompletedProcess[str]:
        entrypoint_arg = ["--entrypoint"] if entrypoint else []
        return subprocess.run(
            [
                sys.executable,
                str(self.helper),
                "run",
                "--label",
                "fixture",
                *entrypoint_arg,
                "--",
                *command,
            ],
            capture_output=True,
            text=True,
            env={**os.environ, **env},
            timeout=30,
        )


class InheritedLockFileRefresh(unittest.TestCase):
    CHILD_SOURCE = '''
import errno
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location("refresh_supervisor", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
fds = tuple(int(value) for value in os.environ[module.FDS_ENV].split(","))
fail_index = int(sys.argv[2])
marker = Path(sys.argv[3])
real_utime = os.utime
calls = []

def refresh(fd, times):
    calls.append(fd)
    if fail_index >= 0 and fd == fds[fail_index]:
        raise PermissionError(errno.EPERM, "fixture timestamp refresh refused")
    return real_utime(fd, times)

failure = None
verified = None
try:
    with mock.patch.object(module.os, "utime", side_effect=refresh):
        receipt, held, paths = module.verify_supervision()
        verified = [str(receipt), list(held), list(map(str, paths))]
except module.SupervisionError as exc:
    failure = {
        "message": str(exc),
        "cause_type": type(exc.__cause__).__name__,
        "errno": getattr(exc.__cause__, "errno", None),
    }
else:
    marker.write_text("admitted")

# A parent copy would hide an accidental close in this process.
identities = [[os.fstat(fd).st_dev, os.fstat(fd).st_ino] for fd in fds]
print(json.dumps({
    "failure": failure,
    "verified": verified,
    "calls": calls,
    "identities": identities,
    "inheritable": [os.get_inheritable(fd) for fd in fds],
}))
raise SystemExit(2 if failure is not None else 0)
'''

    def _assert_retained_locks(self, paths):
        for path in paths:
            probe = os.open(path, os.O_RDWR)
            try:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(probe)

    def _run_refresh_fixture(self, *, held: bool, fail_index: int = -1):
        with _SupervisorSandbox() as sb:
            paths = (sb.bench_lock, sb.gpu_lock)
            sentinels = (b"bench-window sentinel\n", b"metal-gpu sentinel\n")
            for path, sentinel in zip(paths, sentinels, strict=True):
                path.write_bytes(sentinel)
            status = Path(sb.tmp.name) / "refresh.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): fixture\n"
                f"lock=Metal GPU ({sb.gpu_lock}): fixture\n"
            )
            child = Path(sb.tmp.name) / "refresh-child.py"
            child.write_text(self.CHILD_SOURCE)
            marker = Path(sb.tmp.name) / "admitted"
            inherited = []
            try:
                # Do not call sb.run or bench-locks.acquire: either could mask
                # a missing refresh in verify_supervision.
                for path in paths:
                    fd = os.open(path, os.O_RDWR)
                    inherited.append(fd)
                    if held:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                old = time.time_ns() - 4 * 24 * 60 * 60 * 1_000_000_000
                identities = []
                for fd in inherited:
                    # ctime cannot be backdated; this fixture ages only atime/mtime.
                    os.utime(fd, ns=(old, old))
                    before = os.fstat(fd)
                    self.assertEqual(before.st_atime_ns, old)
                    self.assertEqual(before.st_mtime_ns, old)
                    identities.append([before.st_dev, before.st_ino])
                started = time.time_ns()
                result = subprocess.run(
                    [sys.executable, str(child), str(sb.helper),
                     str(fail_index), str(marker)],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_LOCK_FDS": ",".join(map(str, inherited)),
                    },
                    pass_fds=tuple(inherited),
                    timeout=30,
                )
                finished = time.time_ns()
                self.assertEqual(
                    result.returncode, 2 if fail_index >= 0 else 0,
                    result.stderr + result.stdout,
                )
                payload = json.loads(result.stdout)
                self.assertEqual(payload["identities"], identities)
                for fd, path, identity in zip(inherited, paths, identities, strict=True):
                    # Inspect timestamps before sentinel reads can change atime.
                    for current in (os.fstat(fd), path.stat()):
                        self.assertEqual([current.st_dev, current.st_ino], identity)
                        self.assertEqual(current.st_size, len(sentinels[paths.index(path)]))
                        if fail_index < 0:
                            for stamp in (current.st_atime_ns, current.st_mtime_ns,
                                          current.st_ctime_ns):
                                self.assertGreaterEqual(stamp, started - 2_000_000_000)
                                self.assertLessEqual(stamp, finished + 2_000_000_000)
                self.assertEqual(
                    payload["calls"],
                    inherited[:fail_index + 1] if fail_index >= 0 else inherited,
                )
                if fail_index >= 0:
                    self.assertFalse(marker.exists())
                    self.assertIsNone(payload["verified"])
                    self.assertEqual(payload["failure"]["cause_type"], "PermissionError")
                    self.assertEqual(payload["failure"]["errno"], 1)
                    self.assertIn("fixture timestamp refresh refused",
                                  payload["failure"]["message"])
                    self.assertIn(str(paths[fail_index]), payload["failure"]["message"])
                else:
                    self.assertEqual(marker.read_text(), "admitted")
                    self.assertIsNone(payload["failure"])
                    self.assertEqual(payload["verified"],
                                     [str(status), inherited, list(map(str, paths))])
                    self.assertEqual(payload["inheritable"], [False, False])
                self._assert_retained_locks(paths)
                for path, sentinel in zip(paths, sentinels, strict=True):
                    self.assertEqual(path.read_bytes(), sentinel)
            finally:
                for fd in inherited:
                    os.close(fd)

    def test_idle_inherited_descriptors_refresh_times(self):
        self._run_refresh_fixture(held=False)

    def test_held_inherited_descriptors_refresh_times(self):
        self._run_refresh_fixture(held=True)

    def test_first_refresh_error_preserves_borrowed_descriptors(self):
        self._run_refresh_fixture(held=True, fail_index=0)

    def test_second_refresh_error_preserves_borrowed_descriptors(self):
        self._run_refresh_fixture(held=True, fail_index=1)


def _run_bench_ci_fixture(*, fail_first: bool) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    with _SupervisorSandbox() as sb:
        entrypoint = sb.root / "scripts" / "bench-ci.sh"
        shutil.copy2(REPO / "scripts" / "bench-ci.sh", entrypoint)
        quiet_probe = sb.root / "scripts" / "lib" / "quiet-probe.py"
        quiet_probe.write_text("raise SystemExit(0)\n")

        bindir = Path(sb.tmp.name) / "bin"
        bindir.mkdir()
        calls = Path(sb.tmp.name) / "cargo-calls"
        cargo = bindir / "cargo"
        cargo.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "if [[ \" $* \" == *\" lattice-inference \"* ]]; then\n"
            f"  printf '%s\\n' inference >> {str(calls)!r}\n"
            "  if [[ \"${FIXTURE_FAIL_FIRST:-0}\" == 1 ]]; then\n"
            "    printf '%s\\n' 'fixture first target failed' >&2\n"
            "    exit 7\n"
            "  fi\n"
            "else\n"
            f"  printf '%s\\n' embed >> {str(calls)!r}\n"
            "fi\n"
        )
        cargo.chmod(0o755)
        env = {
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "BENCH_IDLE_FLOOR": "0",
            "FIXTURE_FAIL_FIRST": "1" if fail_first else "0",
        }
        for name in (
            "LATTICE_BENCH_LOCK_STATUS",
            "LATTICE_BENCH_LOCK_FDS",
            "LATTICE_BENCH_SUPERVISOR_FD",
        ):
            env.pop(name, None)
        result = subprocess.run(
            ["bash", str(entrypoint)],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        recorded = calls.read_text().splitlines() if calls.exists() else []
        return result, recorded


def _run_wasm_fixture(*, prerequisites: bool) -> subprocess.CompletedProcess[str]:
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("node is unavailable")

    with _SupervisorSandbox() as sb:
        entrypoint = sb.root / "scripts" / "bench_wasm_simd.mjs"
        shutil.copy2(REPO / "scripts" / "bench_wasm_simd.mjs", entrypoint)
        quiet_probe = sb.root / "scripts" / "lib" / "quiet-probe.py"
        quiet_probe.write_text("raise SystemExit(0)\n")

        bindir = Path(sb.tmp.name) / "bin"
        bindir.mkdir()
        (bindir / "python3").symlink_to(sys.executable)
        for name in ("mkdir", "rm"):
            executable = shutil.which(name)
            assert executable is not None
            (bindir / name).symlink_to(executable)
        if prerequisites:
            for name, body in {
                "cargo": (
                    "#!/bin/sh\n"
                    "if [ \"${1:-}\" = --version ]; then echo 'cargo fixture'; fi\n"
                    "exit 0\n"
                ),
                "rustup": (
                    "#!/bin/sh\n"
                    "echo wasm32-unknown-unknown\n"
                ),
                "rustc": (
                    "#!/bin/sh\n"
                    "echo 'rustc fixture'\n"
                ),
            }.items():
                executable = bindir / name
                executable.write_text(body)
                executable.chmod(0o755)

            module = (
                "export function simdDotProduct() { return 0; }\n"
                "export function simdSquaredEuclideanDistance() { return 0; }\n"
                "export function simdCosineSimilarity() { return 0; }\n"
                "export function simdNormalize() {}\n"
            )
            bindgen = bindir / "wasm-bindgen"
            bindgen.write_text(
                f"#!{sys.executable}\n"
                "import sys\n"
                "from pathlib import Path\n"
                "if '--version' in sys.argv:\n"
                "    print('wasm-bindgen fixture')\n"
                "    raise SystemExit(0)\n"
                "out = Path(sys.argv[sys.argv.index('--out-dir') + 1])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "(out / 'package.json').write_text('{\"type\":\"module\"}\\n')\n"
                f"(out / 'lattice_embed.js').write_text({module!r})\n"
            )
            bindgen.chmod(0o755)

        env = {
            **os.environ,
            "PATH": str(bindir),
            "BENCH_IDLE_FLOOR": "0",
            "LATTICE_BENCH_WASM_SIMD_ENFORCE": "",
        }
        for name in (
            "LATTICE_BENCH_LOCK_STATUS",
            "LATTICE_BENCH_LOCK_FDS",
            "LATTICE_BENCH_SUPERVISOR_FD",
        ):
            env.pop(name, None)
        return subprocess.run(
            [
                sys.executable,
                str(sb.helper),
                "run",
                "--label",
                "fixture",
                "--quiet",
                "--entrypoint",
                "--",
                node,
                str(entrypoint),
                "--dims",
                "1",
                "--reps",
                "1",
                "--warmup",
                "0",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )


class CargoTargetProtection(unittest.TestCase):
    def load_supervisor(self, sb):
        spec = importlib.util.spec_from_file_location("target_supervision", sb.helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def run_fixture(self, sb, module, env, child_check, *, resolver=None):
        real_run = subprocess.run
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if command[0] == "/bin/bash":
                return real_run(command, **kwargs)
            self.assertEqual(command, ["fixture-child"])
            child_check(kwargs["env"])
            return subprocess.CompletedProcess(command, 0)

        output = io.StringIO()
        with mock.patch.dict(os.environ, {module.STATUS_ENV: "fixture-receipt", **env}, clear=True), \
                mock.patch.object(module, "verify_supervision", return_value=(Path("receipt"), (10, 11), (sb.bench_lock, sb.gpu_lock))), \
                mock.patch.object(module, "_sample_path_identities"), \
                mock.patch.object(module.subprocess, "run", fake_run), \
                mock.patch.object(sys, "stdout", output):
            if resolver is None:
                rc = module.run_supervised("fixture", ["fixture-child"], quiet=False)
            else:
                with mock.patch.object(module, "_cargo_target_directory", resolver):
                    rc = module.run_supervised("fixture", ["fixture-child"], quiet=False)
        return rc, calls, output.getvalue()

    def test_out_of_tree_env_target_is_marked_before_child(self):
        with _SupervisorSandbox() as sb:
            module = self.load_supervisor(sb)
            target = Path(sb.tmp.name) / "external-target"

            def check_child(env):
                marker = target / ".metadata_never_index"
                self.assertTrue(marker.is_file())
                self.assertFalse(marker.is_symlink())
                record = json.loads(env[module.TARGET_PROTECTION_ENV])
                self.assertEqual(record, {"status": "marked", "directory": str(target.resolve())})

            rc, calls, output = self.run_fixture(
                sb, module, {"CARGO_TARGET_DIR": str(target)}, check_child)
            self.assertEqual(rc, 0)
            self.assertEqual(sum(call[-1] == str(target.resolve()) for call in calls), 1)
            self.assertIn("Run conditions: cargo target protection", output)

    def test_unresolvable_target_is_reported_before_child(self):
        with _SupervisorSandbox() as sb:
            module = self.load_supervisor(sb)

            def unavailable(command):
                raise module.SupervisionError("fixture metadata unavailable")

            def check_child(env):
                record = json.loads(env[module.TARGET_PROTECTION_ENV])
                self.assertEqual(record["status"], "unresolved")
                self.assertIn("fixture metadata unavailable", record["reason"])

            rc, calls, output = self.run_fixture(
                sb, module, {}, check_child, resolver=unavailable)
            self.assertEqual(rc, 0)
            self.assertIn('"status": "unresolved"', output)
            self.assertIn("fixture metadata unavailable", output)
            self.assertEqual(sum(call[0] == "/bin/bash" for call in calls), 2)

    def test_in_repo_target_is_not_marked_twice(self):
        with _SupervisorSandbox() as sb:
            module = self.load_supervisor(sb)
            target = sb.root / "target"
            rc, calls, _ = self.run_fixture(
                sb, module, {}, lambda env: self.assertEqual(
                    json.loads(env[module.TARGET_PROTECTION_ENV])["status"], "marked"),
                resolver=lambda command: target.resolve())
            self.assertEqual(rc, 0)
            self.assertEqual(sum(call[-1] == str(target.resolve()) for call in calls), 1)
            self.assertEqual(sum(call[0] == "/bin/bash" for call in calls), 2)

    def test_unresolvable_repo_tree_refuses_before_target_query(self):
        with _SupervisorSandbox() as sb:
            module = self.load_supervisor(sb)
            with mock.patch.object(Path, "resolve", side_effect=RuntimeError("fixture symlink loop")):
                rc, calls, _ = self.run_fixture(
                    sb, module, {}, lambda env: self.fail("child launched"),
                    resolver=lambda command: self.fail("target query ran"))
            self.assertEqual(rc, module.REFUSAL_EXIT)
            self.assertEqual(calls, [])

    def test_unmarkable_external_target_refuses_child(self):
        with _SupervisorSandbox() as sb:
            module = self.load_supervisor(sb)
            target = Path(sb.tmp.name) / "external-target"
            (target / ".metadata_never_index" / "occupied").mkdir(parents=True)
            rc, calls, _ = self.run_fixture(
                sb, module, {"CARGO_TARGET_DIR": str(target)},
                lambda env: self.fail("unprotected child launched"))
            self.assertEqual(rc, module.REFUSAL_EXIT)
            self.assertNotIn(["fixture-child"], calls)

    def test_metadata_target_is_resolved_without_running_cargo(self):
        with _SupervisorSandbox() as sb:
            module = self.load_supervisor(sb)
            target = Path(sb.tmp.name) / "configured-target"
            response = subprocess.CompletedProcess(
                ["cargo", "metadata"], 0,
                stdout=json.dumps({"target_directory": str(target)}), stderr="")
            with mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch.object(module.subprocess, "run", return_value=response) as run:
                resolved = module._cargo_target_directory(
                    ["cargo", "+1.94.1", "bench", "--manifest-path", "other/Cargo.toml",
                     "--config", 'build.target-dir="configured-target"'])
            self.assertEqual(resolved, target.resolve())
            argv = run.call_args.args[0]
            self.assertEqual(argv[:3], ["cargo", "+1.94.1", "metadata"])
            self.assertIn("--no-deps", argv)
            self.assertIn("--manifest-path", argv)
            self.assertIn("--config", argv)
            self.assertEqual(run.call_args.kwargs["timeout"], 30)

    def test_failed_metadata_reports_resolution_error(self):
        with _SupervisorSandbox() as sb:
            module = self.load_supervisor(sb)
            response = subprocess.CompletedProcess(
                ["cargo", "metadata"], 1, stdout="", stderr="fixture resolution failed")
            with mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch.object(module.subprocess, "run", return_value=response):
                with self.assertRaisesRegex(module.SupervisionError, "fixture resolution failed"):
                    module._cargo_target_directory(["cargo", "bench"])

    def test_direct_target_dir_overrides_env_and_stops_at_separator(self):
        with _SupervisorSandbox() as sb:
            module = self.load_supervisor(sb)
            target = Path(sb.tmp.name) / "command-target"
            with mock.patch.dict(os.environ, {"CARGO_TARGET_DIR": "wrong-target"}, clear=True), \
                    mock.patch.object(module.subprocess, "run", side_effect=AssertionError("cargo invoked")):
                resolved = module._cargo_target_directory(
                    ["cargo", "bench", "--target-dir", str(target), "--", "--target-dir=ignored"])
            self.assertEqual(resolved, target.resolve())

    def test_config_override_with_target_env_uses_metadata(self):
        with _SupervisorSandbox() as sb:
            module = self.load_supervisor(sb)
            target = Path(sb.tmp.name) / "configured-target"
            response = subprocess.CompletedProcess(
                ["cargo", "metadata"], 0,
                stdout=json.dumps({"target_directory": str(target)}), stderr="")
            with mock.patch.dict(os.environ, {"CARGO_TARGET_DIR": "env-target"}, clear=True), \
                    mock.patch.object(module.subprocess, "run", return_value=response) as run:
                resolved = module._cargo_target_directory(
                    ["cargo", "bench", "--config", 'build.target-dir="configured-target"'])
            self.assertTrue(run.called)
            self.assertEqual(resolved, target.resolve())

    def test_both_target_env_spellings_use_metadata(self):
        with _SupervisorSandbox() as sb:
            module = self.load_supervisor(sb)
            target = Path(sb.tmp.name) / "metadata-target"
            response = subprocess.CompletedProcess(
                ["cargo", "metadata"], 0,
                stdout=json.dumps({"target_directory": str(target)}), stderr="")
            with mock.patch.dict(os.environ, {"CARGO_TARGET_DIR": "first", "CARGO_BUILD_TARGET_DIR": "second"}, clear=True), \
                    mock.patch.object(module.subprocess, "run", return_value=response) as run:
                resolved = module._cargo_target_directory(["cargo", "bench"])
            self.assertTrue(run.called)
            self.assertEqual(resolved, target.resolve())


class RuntimeContract(unittest.TestCase):
    def test_bench_command_protects_build_trees_before_each_command(self):
        for durable in (False, True):
            with self.subTest(durable=durable), _SupervisorSandbox() as sb:
                wrapper = sb.root / "scripts/bench-command.sh"
                shutil.copy2(REPO / "scripts/bench-command.sh", wrapper)
                (sb.helper.parent / "quiet-probe.py").write_text("print('fixture quiet')\n")
                markers = [sb.root / tree / ".metadata_never_index" for tree in ("target", ".cache")]
                observed = Path(sb.tmp.name) / "observed"
                code = (
                    "from pathlib import Path; "
                    f"markers = {list(map(str, markers))!r}; "
                    "assert all(Path(p).is_file() and not Path(p).is_symlink() for p in markers); "
                    f"Path({str(observed)!r}).touch()"
                )
                argv = ["bash", str(wrapper), "--label", "fixture"]
                if durable:
                    argv.append("--durable")
                argv += ["--", sys.executable, "-c", code]
                env = dict(os.environ)
                for name in ("LATTICE_BENCH_LOCK_STATUS", "LATTICE_BENCH_LOCK_FDS", "LATTICE_BENCH_SUPERVISOR_FD"):
                    env.pop(name, None)
                for _ in range(2):
                    result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertTrue(observed.is_file())
                    observed.unlink()
                    for marker in markers:
                        shutil.rmtree(marker.parent)

    def test_bench_command_refuses_unprotected_build_trees(self):
        for tree in ("target", ".cache"):
            with self.subTest(tree=tree), _SupervisorSandbox() as sb:
                wrapper = sb.root / "scripts/bench-command.sh"
                shutil.copy2(REPO / "scripts/bench-command.sh", wrapper)
                blocker = sb.root / tree / ".metadata_never_index" / "occupied"
                blocker.mkdir(parents=True)
                observed = Path(sb.tmp.name) / "observed"
                env = dict(os.environ)
                for name in ("LATTICE_BENCH_LOCK_STATUS", "LATTICE_BENCH_LOCK_FDS", "LATTICE_BENCH_SUPERVISOR_FD"):
                    env.pop(name, None)
                result = subprocess.run(
                    ["bash", str(wrapper), "--label", "fixture", "--", sys.executable,
                     "-c", f"from pathlib import Path; Path({str(observed)!r}).touch()"],
                    env=env, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("[noindex] FATAL", result.stderr)
                self.assertFalse(observed.exists())

    def test_first_failed_bench_ci_target_refuses_before_later_targets(self):
        result, calls = _run_bench_ci_fixture(fail_first=True)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(calls, ["inference"])
        self.assertIn("fixture first target failed", result.stderr)
        self.assertIn("NOT MEASURABLE: lattice-inference benchmark failed", result.stderr)

    def test_bench_ci_healthy_targets_complete(self):
        result, calls = _run_bench_ci_fixture(fail_first=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, ["inference", "embed"])

    def test_bench_command_forwards_entrypoint_to_self_supervising_python_target(self):
        """bench-command.sh must pass --entrypoint to bench_supervision.py.

        Without it, a wrapped command that is itself a self-supervising
        Python entry point that calls ensure_python_entrypoint sees LATTICE_BENCH_LOCK_STATUS and
        concludes a supervisor is already present, then refuses because the
        liveness pipe it looks for (LATTICE_BENCH_SUPERVISOR_FD) was never
        created for it.
        """
        with _SupervisorSandbox() as sb:
            wrapper = sb.root / "scripts" / "bench-command.sh"
            shutil.copy2(REPO / "scripts" / "bench-command.sh", wrapper)
            marker = Path(sb.tmp.name) / "self-supervising-target-ran"
            target = sb.root / "scripts" / "self_supervising_target.py"
            target.write_text(
                "import sys\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import ensure_python_entrypoint\n"
                "ensure_python_entrypoint('fixture-target')\n"
                f"open({str(marker)!r}, 'w').write('ran')\n"
            )
            env = {**os.environ}
            for name in (
                "LATTICE_BENCH_LOCK_STATUS",
                "LATTICE_BENCH_LOCK_FDS",
                "LATTICE_BENCH_SUPERVISOR_FD",
            ):
                env.pop(name, None)
            result = subprocess.run(
                ["bash", str(wrapper), "--label", "fixture", "--", sys.executable, str(target)],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("LATTICE_BENCH_SUPERVISOR_FD is not set", result.stderr)
            self.assertTrue(marker.exists())

    def test_recipe_outcome_one_is_not_reclassified_as_supervision_failure(self):
        with _SupervisorSandbox() as sb:
            entrypoint = sb.root / "scripts" / "outcome_entrypoint.sh"
            entrypoint.write_text(
                "#!/usr/bin/env bash\n"
                "set -uo pipefail\n"
                f"source {str(sb.root / 'scripts/lib/bench-supervision.sh')!r}\n"
                "measurement() { return 1; }\n"
                'bench_supervise_entry "fixture" ordinary measurement "$@"\n'
            )
            entrypoint.chmod(0o755)
            result = sb.run([str(entrypoint)], entrypoint=True)

        self.assertEqual(result.returncode, 1, result.stderr)

    def test_successful_parent_with_live_descendant_refuses_before_unlock(self):
        with _SupervisorSandbox() as sb:
            pid_file = Path(sb.tmp.name) / "grandchild-pid"
            grandchild = "import time; time.sleep(30)"
            parent = (
                "import subprocess, sys; from pathlib import Path; "
                f"child=subprocess.Popen([sys.executable, '-c', {grandchild!r}], "
                "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
                "stderr=subprocess.DEVNULL); "
                f"Path({str(pid_file)!r}).write_text(str(child.pid))"
            )
            result = sb.run([sys.executable, "-c", parent])
            child_pid = int(pid_file.read_text())
            try:
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("live process-group descendants", result.stderr)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
                for path in (sb.bench_lock, sb.gpu_lock):
                    fd = os.open(path, os.O_RDWR)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
                        os.close(fd)
            finally:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_successful_parent_without_descendants_completes(self):
        with _SupervisorSandbox() as sb:
            result = sb.run([sys.executable, "-c", "raise SystemExit(0)"])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_wasm_prerequisite_is_not_measurable(self):
        result = _run_wasm_fixture(prerequisites=False)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("NOT MEASURABLE", result.stderr)
        self.assertIn("cargo not found on PATH", result.stderr)

    def test_wasm_fixture_with_prerequisites_completes(self):
        result = _run_wasm_fixture(prerequisites=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("| dot_product | 1 |", result.stdout)

    def test_command_waits_for_each_machine_wide_lock(self):
        """Mutation-sensitive: dropping either acquire lets its subtest run early."""

        for which in ("bench", "gpu"):
            with self.subTest(lock=which), _SupervisorSandbox() as sb:
                held_path = sb.bench_lock if which == "bench" else sb.gpu_lock
                held_path.touch()
                marker = Path(sb.tmp.name) / f"{which}.ran"
                with held_path.open("r+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    proc = subprocess.Popen(
                        [
                            sys.executable,
                            str(sb.helper),
                            "run",
                            "--label",
                            "fixture",
                            "--",
                            sys.executable,
                            "-c",
                            f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    time.sleep(0.35)
                    self.assertIsNone(proc.poll())
                    self.assertFalse(marker.exists())
                    fcntl.flock(lock, fcntl.LOCK_UN)
                stdout, stderr = proc.communicate(timeout=30)
                self.assertEqual(proc.returncode, 0, f"{stdout}\n{stderr}")
                self.assertEqual(marker.read_text(), "ran")

    def test_durable_run_refuses_before_command_on_busy_machine(self):
        """Mutation-sensitive: warning instead of refusal creates the marker."""

        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "should-not-run"
            result = subprocess.run(
                [
                    sys.executable,
                    str(sb.helper),
                    "run",
                    "--label",
                    "fixture",
                    "--quiet",
                    "--",
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "BENCH_IDLE_FLOOR": "101"},
                timeout=30,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(marker.exists())
            self.assertIn("refusing to measure", result.stderr)

    def test_unsupervised_command_runs_with_both_lock_receipts(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "receipt"
            code = (
                "import os; from pathlib import Path; "
                "p=Path(os.environ['LATTICE_BENCH_LOCK_STATUS']); "
                f"Path({str(marker)!r}).write_text(p.read_text())"
            )
            result = sb.run([sys.executable, "-c", code])
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = marker.read_text()
            self.assertIn("bench-window", receipt)
            self.assertIn("Metal GPU", receipt)

    def test_arbitrary_command_does_not_inherit_lock_capabilities(self):
        """A raw Cargo command must not leak lock fds into build daemons."""

        with _SupervisorSandbox() as sb:
            code = (
                "import os,sys; "
                "sys.exit(1 if 'LATTICE_BENCH_LOCK_FDS' in os.environ else 0)"
            )
            result = sb.run([sys.executable, "-c", code])
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_entrypoint_cannot_unlock_supervisor_locks_during_measurement(self):
        """LOCK_UN in the child must not reach the supervisor's open files."""

        with _SupervisorSandbox() as sb:
            ready = Path(sb.tmp.name) / "unlock-attempted"
            release = Path(sb.tmp.name) / "release"
            marker = Path(sb.tmp.name) / "measurement-ran"
            entrypoint = sb.root / "scripts" / "unlock_entrypoint.py"
            entrypoint.write_text(
                "import fcntl, os, time\n"
                "from pathlib import Path\n"
                "for fd in range(3, 256):\n"
                "    try:\n"
                "        os.fstat(fd)\n"
                "        fcntl.flock(fd, fcntl.LOCK_UN)\n"
                "    except (OSError, OverflowError):\n"
                "        pass\n"
                f"Path({str(ready)!r}).write_text('attempted')\n"
                f"release = Path({str(release)!r})\n"
                "while not release.exists():\n"
                "    time.sleep(0.01)\n"
                f"Path({str(marker)!r}).write_text('ran')\n"
            )
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(sb.helper),
                    "run",
                    "--label",
                    "fixture",
                    "--entrypoint",
                    "--",
                    sys.executable,
                    str(entrypoint),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and proc.poll() is None:
                    if time.monotonic() >= deadline:
                        self.fail("entrypoint did not attempt LOCK_UN")
                    time.sleep(0.01)
                self.assertIsNone(proc.poll())
                for path in (sb.bench_lock, sb.gpu_lock):
                    fd = os.open(path, os.O_RDWR)
                    try:
                        with self.assertRaises(OSError):
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        os.close(fd)
            finally:
                release.touch()
                stdout, stderr = proc.communicate(timeout=30)
            self.assertEqual(proc.returncode, 0, f"{stdout}\n{stderr}")
            self.assertEqual(marker.read_text(), "ran")

    def test_sigkill_of_supervisor_reaps_measurement_before_unlock(self):
        """An orphaned measurement must not outlive the held lock window."""

        with _SupervisorSandbox() as sb:
            ready = Path(sb.tmp.name) / "measurement-ready"
            release = Path(sb.tmp.name) / "release"
            entrypoint = sb.root / "scripts" / "sigkill_entrypoint.py"
            entrypoint.write_text(
                "import os, sys, time\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import sample_measurement_handoff\n"
                "_, pipe_fd, _ = sample_measurement_handoff()\n"
                "os.close(pipe_fd)\n"
                f"Path({str(ready)!r}).write_text(f'{{os.getppid()}},{{os.getpid()}}')\n"
                f"release = Path({str(release)!r})\n"
                "while not release.exists():\n"
                "    time.sleep(0.01)\n"
            )

            def lock_states() -> list[str]:
                states = []
                for path in (sb.bench_lock, sb.gpu_lock):
                    fd = os.open(path, os.O_RDWR)
                    try:
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except OSError:
                            states.append("blocked")
                        else:
                            states.append("acquired")
                            fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
                        os.close(fd)
                return states

            outer = subprocess.Popen(
                [
                    sys.executable,
                    str(sb.helper),
                    "run",
                    "--label",
                    "fixture",
                    "--entrypoint",
                    "--",
                    sys.executable,
                    str(entrypoint),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            measurement_pid = None
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and outer.poll() is None:
                    if time.monotonic() >= deadline:
                        self.fail("measurement did not reach the handoff barrier")
                    time.sleep(0.01)
                self.assertIsNone(outer.poll())
                supervisor_pid, measurement_pid = map(
                    int, ready.read_text().split(",")
                )
                self.assertEqual(lock_states(), ["blocked", "blocked"])

                os.kill(supervisor_pid, signal.SIGKILL)
                outer.wait(timeout=30)

                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        os.kill(measurement_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.01)
                else:
                    self.fail(
                        "measurement remained alive after the lock owner returned"
                    )
                self.assertEqual(lock_states(), ["acquired", "acquired"])
            finally:
                release.touch()
                if measurement_pid is not None:
                    try:
                        os.kill(measurement_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if outer.poll() is None:
                    outer.kill()
                    outer.wait()

    def test_substitute_and_restore_is_outside_cooperative_contract(self):
        """A green result deliberately pins behavior outside the contract."""

        with _SupervisorSandbox() as sb:
            substituted = Path(sb.tmp.name) / "substituted"
            restore = Path(sb.tmp.name) / "restore"
            marker = Path(sb.tmp.name) / "restored"
            entrypoint = sb.root / "scripts" / "substitute_paths.py"
            entrypoint.write_text(
                "import os, time\n"
                "from pathlib import Path\n"
                f"paths = tuple(Path(p) for p in ({str(sb.bench_lock)!r}, {str(sb.gpu_lock)!r}))\n"
                "for path in paths:\n"
                "    path.rename(path.with_name(path.name + '.held'))\n"
                "    path.touch()\n"
                f"Path({str(substituted)!r}).write_text('ready')\n"
                f"restore = Path({str(restore)!r})\n"
                "while not restore.exists():\n"
                "    time.sleep(0.01)\n"
                "for path in paths:\n"
                "    path.unlink()\n"
                "    path.with_name(path.name + '.held').rename(path)\n"
                f"Path({str(marker)!r}).write_text('restored')\n"
            )
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(sb.helper),
                    "run",
                    "--label",
                    "fixture",
                    "--",
                    sys.executable,
                    str(entrypoint),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 10
                while not substituted.exists() and proc.poll() is None:
                    if time.monotonic() >= deadline:
                        self.fail("entrypoint did not substitute lock paths")
                    time.sleep(0.01)
                self.assertIsNone(proc.poll())
                for path in (sb.bench_lock, sb.gpu_lock):
                    fd = os.open(path, os.O_RDWR)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        os.close(fd)
            finally:
                restore.touch()
                stdout, stderr = proc.communicate(timeout=30)
            outside_contract = (
                "success deliberately records behavior outside the cooperative contract"
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"{outside_contract}:\n{stdout}\n{stderr}",
            )
            self.assertEqual(marker.read_text(), "restored", outside_contract)

    def test_forged_live_pipe_handoff_is_accepted_but_outside_cooperative_contract(
        self,
    ):
        """Acceptance records the deliberate same-user bypass limitation."""

        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            status = Path(sb.tmp.name) / "forged-live-pipe.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): fabricated\n"
                f"lock=Metal GPU ({sb.gpu_lock}): fabricated\n"
            )
            handoff_accepted = Path(sb.tmp.name) / "handoff-accepted"
            release = Path(sb.tmp.name) / "release-temporary-holders"
            marker = Path(sb.tmp.name) / "forged-handoff-result"
            entrypoint = sb.root / "scripts" / "forged_handoff.py"
            entrypoint.write_text(
                "import fcntl, os, sys, time\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import ensure_python_entrypoint\n"
                "ensure_python_entrypoint('fixture')\n"
                f"Path({str(handoff_accepted)!r}).write_text('accepted')\n"
                f"release = Path({str(release)!r})\n"
                "while not release.exists():\n"
                "    time.sleep(0.01)\n"
                "states = []\n"
                f"for path in ({str(sb.bench_lock)!r}, {str(sb.gpu_lock)!r}):\n"
                "    fd = os.open(path, os.O_RDWR)\n"
                "    try:\n"
                "        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                "    except OSError:\n"
                "        states.append('blocked')\n"
                "    else:\n"
                "        states.append('acquired')\n"
                "    finally:\n"
                "        os.close(fd)\n"
                f"Path({str(marker)!r}).write_text('accepted:' + ','.join(states))\n"
            )
            holders = tuple(
                os.open(path, os.O_RDWR) for path in (sb.bench_lock, sb.gpu_lock)
            )
            read_fd, write_fd = os.pipe()
            env = {
                **os.environ,
                "LATTICE_BENCH_LOCK_STATUS": str(status),
                "LATTICE_BENCH_SUPERVISOR_FD": str(read_fd),
            }
            env.pop("LATTICE_BENCH_LOCK_FDS", None)
            proc = None
            try:
                for fd in holders:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                proc = subprocess.Popen(
                    [sys.executable, str(entrypoint)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    pass_fds=(read_fd,),
                )
                deadline = time.monotonic() + 10
                while not handoff_accepted.exists() and proc.poll() is None:
                    if time.monotonic() >= deadline:
                        self.fail("forged handoff was not sampled")
                    time.sleep(0.01)
                self.assertIsNone(proc.poll())
                for fd in holders:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                release.touch()
                stdout, stderr = proc.communicate(timeout=30)
            finally:
                release.touch()
                for fd in (*holders, read_fd, write_fd):
                    os.close(fd)
                if proc is not None and proc.poll() is None:
                    proc.kill()
                    proc.communicate()
            outside_contract = (
                "success deliberately records behavior outside the cooperative contract"
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"{outside_contract}:\n{stdout}\n{stderr}",
            )
            self.assertEqual(
                marker.read_text(),
                "accepted:acquired,acquired",
                outside_contract,
            )

    def test_oversized_descriptor_uses_normal_refusal_diagnostic(self):
        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            status = Path(sb.tmp.name) / "oversized.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): fabricated\n"
                f"lock=Metal GPU ({sb.gpu_lock}): fabricated\n"
            )
            result = subprocess.run(
                [sys.executable, str(sb.helper), "verify"],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "LATTICE_BENCH_LOCK_STATUS": str(status),
                    "LATTICE_BENCH_LOCK_FDS": f"{2**100},{2**101}",
                },
                timeout=30,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("cannot be matched", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_python_entrypoint_hides_capability_names_during_work(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "python-entrypoint"
            entrypoint = sb.root / "scripts" / "entrypoint.py"
            entrypoint.write_text(
                "import fcntl, os, sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import ensure_python_entrypoint\n"
                "ensure_python_entrypoint('fixture')\n"
                "states = []\n"
                f"for path in ({str(sb.bench_lock)!r}, {str(sb.gpu_lock)!r}):\n"
                "    fd = os.open(path, os.O_RDWR)\n"
                "    try:\n"
                "        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                "    except OSError:\n"
                "        states.append('blocked')\n"
                "    else:\n"
                "        states.append('acquired')\n"
                "    finally:\n"
                "        os.close(fd)\n"
                f"Path({str(marker)!r}).write_text("
                "('present' if ('LATTICE_BENCH_LOCK_FDS' in os.environ or "
                "'LATTICE_BENCH_SUPERVISOR_FD' in os.environ) "
                "else 'hidden') + ':' + ','.join(states))\n"
            )
            result = subprocess.run(
                [sys.executable, str(entrypoint)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(), "hidden:blocked,blocked")

    def test_python_entrypoint_refuses_pipe_without_open_writer(self):
        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            status = Path(sb.tmp.name) / "closed-pipe.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): acquired\n"
                f"lock=Metal GPU ({sb.gpu_lock}): acquired\n"
            )
            marker = Path(sb.tmp.name) / "must-not-run"
            entrypoint = sb.root / "scripts" / "closed_pipe.py"
            entrypoint.write_text(
                "import sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import ensure_python_entrypoint\n"
                "ensure_python_entrypoint('fixture')\n"
                f"Path({str(marker)!r}).write_text('ran')\n"
            )
            witness_fd, writer_fd = os.pipe()
            os.close(writer_fd)
            try:
                result = subprocess.run(
                    [sys.executable, str(entrypoint)],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_SUPERVISOR_FD": str(witness_fd),
                    },
                    pass_fds=(witness_fd,),
                    timeout=30,
                )
            finally:
                os.close(witness_fd)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("no open writer", result.stderr)
            self.assertFalse(marker.exists())

    def test_replaced_canonical_paths_are_refused_before_measurement(self):
        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            status = Path(sb.tmp.name) / "path-replacement.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): acquired\n"
                f"lock=Metal GPU ({sb.gpu_lock}): acquired\n"
            )
            marker = Path(sb.tmp.name) / "replacement-measurement-ran"
            driver = sb.root / "scripts" / "replace_paths.py"
            driver.write_text(
                "import sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "import bench_supervision as supervision\n"
                f"paths = ({str(sb.bench_lock)!r}, {str(sb.gpu_lock)!r})\n"
                "original_flock = supervision.fcntl.flock\n"
                "replaced = False\n"
                "def replace_then_flock(fd, operation):\n"
                "    global replaced\n"
                "    if not replaced:\n"
                "        replaced = True\n"
                "        for raw in paths:\n"
                "            path = Path(raw)\n"
                "            path.rename(path.with_name(path.name + '.held'))\n"
                "            path.touch()\n"
                "    return original_flock(fd, operation)\n"
                "supervision.fcntl.flock = replace_then_flock\n"
                "raise SystemExit(supervision.main([\n"
                "    'run', '--label', 'fixture', '--',\n"
                f"    sys.executable, '-c', \"from pathlib import Path; Path({str(marker)!r}).write_text('ran')\",\n"
                "]))\n"
            )
            inherited = tuple(
                os.open(path, os.O_RDWR) for path in (sb.bench_lock, sb.gpu_lock)
            )
            try:
                for fd in inherited:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(
                    [sys.executable, str(driver)],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_LOCK_FDS": ",".join(map(str, inherited)),
                    },
                    pass_fds=inherited,
                    timeout=30,
                )
            finally:
                for fd in inherited:
                    os.close(fd)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("changed while acquiring", result.stderr)
            self.assertFalse(marker.exists())

    def test_replaced_canonical_paths_during_measurement_are_refused_afterward(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "replacement-during-measurement"
            code = (
                "from pathlib import Path; "
                f"paths=({str(sb.bench_lock)!r}, {str(sb.gpu_lock)!r}); "
                "[(lambda path: (path.rename(path.with_name(path.name + '.held')), "
                "path.touch()))(Path(raw)) for raw in paths]; "
                f"Path({str(marker)!r}).write_text('ran')"
            )
            result = sb.run([sys.executable, "-c", code])
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("changed after measurement", result.stderr)
            self.assertEqual(marker.read_text(), "ran")

    def test_durable_entrypoint_refuses_lock_only_outer_supervisor(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "must-not-run"
            entrypoint = sb.root / "scripts" / "durable_entrypoint.py"
            entrypoint.write_text(
                "import sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import ensure_python_entrypoint\n"
                "ensure_python_entrypoint('fixture', quiet=True)\n"
                f"Path({str(marker)!r}).write_text('ran')\n"
            )
            result = sb.run(
                [sys.executable, str(entrypoint)],
                entrypoint=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(marker.exists())
            self.assertIn("lock-only", result.stderr)

    def test_shell_measurement_children_do_not_inherit_supervisor_witness(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "shell-entrypoint"
            entrypoint = sb.root / "scripts" / "entrypoint.sh"
            entrypoint.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"source {str(sb.root / 'scripts/lib/bench-supervision.sh')!r}\n"
                'inherited_fd="${LATTICE_BENCH_SUPERVISOR_FD:-}"\n'
                "measurement() {\n"
                '  [[ -z "${LATTICE_BENCH_SUPERVISOR_FD:-}" ]]\n'
                "  if python3 -c \"import os; os.fstat($inherited_fd)\" 2>/dev/null; then\n"
                "      state=inherited\n"
                "    else\n"
                "      state=closed\n"
                "    fi\n"
                f"  printf '%s' \"$state\" > {str(marker)!r}\n"
                "}\n"
                'bench_supervise_entry "fixture" ordinary measurement "$@"\n'
            )
            entrypoint.chmod(0o755)
            result = subprocess.run(
                ["bash", str(entrypoint)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(), "closed")

    def test_shell_handoff_keeps_witness_for_nested_python_guard(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "nested-python-entrypoint"
            cargo_marker = Path(sb.tmp.name) / "cargo-capabilities"
            python_entrypoint = sb.root / "scripts" / "nested.py"
            python_entrypoint.write_text(
                "import os, sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import ensure_python_entrypoint\n"
                "ensure_python_entrypoint('fixture')\n"
                f"Path({str(marker)!r}).write_text("
                "'present' if 'LATTICE_BENCH_SUPERVISOR_FD' in os.environ else 'hidden')\n"
            )
            shell_entrypoint = sb.root / "scripts" / "nested.sh"
            shell_entrypoint.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"source {str(sb.root / 'scripts/lib/bench-supervision.sh')!r}\n"
                'bench_supervise_entry "fixture" handoff - "$@"\n'
                "(\n"
                "  bench_close_supervisor_witness\n"
                f"  printf '%s' \"${{LATTICE_BENCH_SUPERVISOR_FD:-closed}}\" > {str(cargo_marker)!r}\n"
                ")\n"
                f"exec {sys.executable!r} {str(python_entrypoint)!r}\n"
            )
            shell_entrypoint.chmod(0o755)
            result = subprocess.run(
                ["bash", str(shell_entrypoint)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(cargo_marker.read_text(), "closed")
            self.assertEqual(marker.read_text(), "hidden")

    def test_actual_e2e_entrypoint_hands_off_to_build_and_parity_check(self):
        with _SupervisorSandbox() as sb:
            entrypoint = sb.root / "scripts" / "e2e-parity-local.sh"
            shutil.copy2(REPO / "scripts" / "e2e-parity-local.sh", entrypoint)
            cargo_args = Path(sb.tmp.name) / "cargo-args"
            parity_marker = Path(sb.tmp.name) / "parity-ran"
            bindir = Path(sb.tmp.name) / "bin"
            bindir.mkdir()
            cargo = bindir / "cargo"
            cargo.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                '[[ -z "${LATTICE_BENCH_SUPERVISOR_FD:-}" ]]\n'
                f"printf '%s' \"$*\" > {str(cargo_args)!r}\n"
            )
            cargo.chmod(0o755)
            parity = sb.root / "scripts" / "e2e_parity_check.py"
            parity.write_text(
                "import sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import ensure_python_entrypoint\n"
                "ensure_python_entrypoint('e2e-parity-local')\n"
                f"Path({str(parity_marker)!r}).write_text('ran')\n"
            )
            env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"}
            for name in (
                "LATTICE_BENCH_LOCK_STATUS",
                "LATTICE_BENCH_LOCK_FDS",
                "LATTICE_BENCH_SUPERVISOR_FD",
            ):
                env.pop(name, None)
            result = subprocess.run(
                ["bash", str(entrypoint)],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                cargo_args.read_text(),
                "build --release --bin qwen35_generate -p lattice-inference "
                "--features f16",
            )
            self.assertEqual(parity_marker.read_text(), "ran")

    def test_durable_shell_refuses_lock_only_outer_without_errexit(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "shell-must-not-run"
            entrypoint = sb.root / "scripts" / "durable_entrypoint.sh"
            entrypoint.write_text(
                "#!/usr/bin/env bash\n"
                "set -uo pipefail\n"
                f"source {str(sb.root / 'scripts/lib/bench-supervision.sh')!r}\n"
                f"measurement() {{ printf ran > {str(marker)!r}; }}\n"
                'bench_supervise_entry "fixture" durable measurement "$@"\n'
            )
            entrypoint.chmod(0o755)
            result = sb.run(
                [str(entrypoint)],
                entrypoint=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(marker.exists())
            self.assertIn("lock-only", result.stderr)

    @unittest.skipUnless(shutil.which("node"), "node is unavailable")
    def test_node_child_can_forward_pipe_to_handoff_sample(self):
        """Mutation-sensitive: ordinary spawn closes the handoff pipe."""

        with _SupervisorSandbox() as sb:
            code = """
const {spawnSync} = require('node:child_process');
const fd = Number(process.env.LATTICE_BENCH_SUPERVISOR_FD);
const stdio = ['ignore', 'pipe', 'pipe'];
while (stdio.length <= fd) stdio.push('ignore');
stdio[fd] = fd;
const child = spawnSync(
  'python3',
  [process.argv[1], 'verify'],
  {env: process.env, encoding: 'utf8', stdio},
);
process.stdout.write(child.stdout ?? '');
process.stderr.write(child.stderr ?? '');
process.exit(child.status ?? 2);
"""
            result = sb.run(
                ["node", "-e", code, str(sb.helper)],
                entrypoint=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_idle_inherited_fds_are_acquired_by_measuring_process(self):
        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            status = Path(sb.tmp.name) / "idle.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): fabricated\n"
                f"lock=Metal GPU ({sb.gpu_lock}): fabricated\n"
            )
            inherited = tuple(
                os.open(path, os.O_RDWR) for path in (sb.bench_lock, sb.gpu_lock)
            )
            try:
                result = subprocess.run(
                    [sys.executable, str(sb.helper), "verify"],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_LOCK_FDS": ",".join(map(str, inherited)),
                    },
                    pass_fds=inherited,
                    timeout=30,
                )
            finally:
                for fd in inherited:
                    os.close(fd)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_locked_noncanonical_paths_are_refused(self):
        with _SupervisorSandbox() as sb:
            fake_paths = (
                Path(sb.tmp.name) / "fake-window.lock",
                Path(sb.tmp.name) / "fake-gpu.lock",
            )
            inherited = tuple(
                os.open(path, os.O_RDWR | os.O_CREAT) for path in fake_paths
            )
            status = Path(sb.tmp.name) / "fake-path.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({fake_paths[0]}): fabricated\n"
                f"lock=Metal GPU ({fake_paths[1]}): fabricated\n"
            )
            try:
                for fd in inherited:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(
                    [sys.executable, str(sb.helper), "verify"],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_LOCK_FDS": ",".join(map(str, inherited)),
                    },
                    pass_fds=inherited,
                    timeout=30,
                )
            finally:
                for fd in inherited:
                    os.close(fd)
            self.assertEqual(result.returncode, 2)
            self.assertIn(f"expected {sb.bench_lock}", result.stderr)

    def test_swapped_canonical_lock_order_is_refused(self):
        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            inherited = tuple(
                os.open(path, os.O_RDWR) for path in (sb.gpu_lock, sb.bench_lock)
            )
            status = Path(sb.tmp.name) / "swapped.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.gpu_lock}): fabricated\n"
                f"lock=Metal GPU ({sb.bench_lock}): fabricated\n"
            )
            try:
                for fd in inherited:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(
                    [sys.executable, str(sb.helper), "verify"],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_LOCK_FDS": ",".join(map(str, inherited)),
                    },
                    pass_fds=inherited,
                    timeout=30,
                )
            finally:
                for fd in inherited:
                    os.close(fd)
            self.assertEqual(result.returncode, 2)
            self.assertIn(f"expected {sb.bench_lock}", result.stderr)

    def test_duplicate_lock_inode_is_refused(self):
        with _SupervisorSandbox() as sb:
            sb.bench_lock.touch()
            os.link(sb.bench_lock, sb.gpu_lock)
            fd = os.open(sb.bench_lock, os.O_RDWR)
            twin = os.dup(fd)
            status = Path(sb.tmp.name) / "duplicate-inode.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): fabricated\n"
                f"lock=Metal GPU ({sb.gpu_lock}): fabricated\n"
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(
                    [sys.executable, str(sb.helper), "verify"],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_LOCK_FDS": f"{fd},{twin}",
                    },
                    pass_fds=(fd, twin),
                    timeout=30,
                )
            finally:
                os.close(twin)
                os.close(fd)
            self.assertEqual(result.returncode, 2)
            self.assertIn("distinct inodes", result.stderr)

    def test_self_held_fds_do_not_replace_supervisor_witness(self):
        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            status = Path(sb.tmp.name) / "self-held.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): acquired\n"
                f"lock=Metal GPU ({sb.gpu_lock}): acquired\n"
            )
            marker = Path(sb.tmp.name) / "must-not-run"
            code = (
                "import fcntl, os, sys; from pathlib import Path; "
                f"sys.path.insert(0, {str(sb.helper.parent)!r}); "
                f"paths=({str(sb.bench_lock)!r}, {str(sb.gpu_lock)!r}); "
                "fds=tuple(os.open(path, os.O_RDWR) for path in paths); "
                "[fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB) for fd in fds]; "
                "os.environ['LATTICE_BENCH_LOCK_FDS']=','.join(map(str, fds)); "
                "from bench_supervision import ensure_python_entrypoint; "
                "ensure_python_entrypoint('fixture'); "
                f"Path({str(marker)!r}).write_text('ran')"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "LATTICE_BENCH_LOCK_STATUS": str(status),
                },
                timeout=30,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("LATTICE_BENCH_SUPERVISOR_FD is not set", result.stderr)
            self.assertFalse(marker.exists())

    def test_forged_ancestor_receipt_without_fds_is_refused_before_command(self):
        with _SupervisorSandbox() as sb:
            status = Path(sb.tmp.name) / "forged-ancestor.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): fabricated\n"
                f"lock=Metal GPU ({sb.gpu_lock}): fabricated\n"
            )
            marker = Path(sb.tmp.name) / "unsupervised-command-ran"
            result = subprocess.run(
                [
                    sys.executable,
                    str(sb.helper),
                    "run",
                    "--label",
                    "forged-ancestor",
                    "--",
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
                ],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "LATTICE_BENCH_LOCK_STATUS": str(status),
                },
                timeout=30,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("LATTICE_BENCH_LOCK_FDS is not set", result.stderr)
            self.assertFalse(marker.exists())

    def test_unlocked_inherited_fds_do_not_borrow_another_holders_contention(self):
        """Mutation-sensitive: probing only by path accepts the wrong holder."""

        with _SupervisorSandbox() as sb:
            for path in (sb.bench_lock, sb.gpu_lock):
                path.touch()
            status = Path(sb.tmp.name) / "wrong-holder.status"
            status.write_text(
                f"supervisor_pid={os.getpid()}\n"
                f"lock=bench-window ({sb.bench_lock}): fabricated\n"
                f"lock=Metal GPU ({sb.gpu_lock}): fabricated\n"
            )
            inherited = tuple(
                os.open(path, os.O_RDWR) for path in (sb.bench_lock, sb.gpu_lock)
            )
            holders = tuple(
                os.open(path, os.O_RDWR) for path in (sb.bench_lock, sb.gpu_lock)
            )
            try:
                for fd in holders:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(
                    [sys.executable, str(sb.helper), "verify"],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "LATTICE_BENCH_LOCK_STATUS": str(status),
                        "LATTICE_BENCH_LOCK_FDS": ",".join(map(str, inherited)),
                    },
                    pass_fds=inherited,
                    timeout=30,
                )
            finally:
                for fd in (*inherited, *holders):
                    os.close(fd)
            self.assertEqual(result.returncode, 2)
            self.assertIn("could not acquire canonical lock", result.stderr)


class BenchCompareDirectInvocationRefusal(unittest.TestCase):
    """The bench-compare body requires a descriptor-free supervisor handoff.

    The entry point gives inherited lock capabilities only to the dedicated
    Python supervisor. The shell receives its non-lock handoff pipe. A caller
    that writes a status receipt and invokes the body directly has neither, so
    verify_locks refuses before touching cargo.
    """

    def _build_repo(self, tmp: str) -> Path:
        root = Path(tmp) / "repo"
        lib = root / "scripts" / "lib"
        lib.mkdir(parents=True)
        shutil.copy2(
            REPO / "scripts" / "bench-compare.sh",
            root / "scripts" / "bench-compare.sh",
        )
        (root / "scripts" / "bench-compare.sh").chmod(0o755)
        shutil.copy2(
            REPO / "scripts" / "lib" / "bench-compare-impl.sh",
            lib / "bench-compare-impl.sh",
        )
        (lib / "bench-compare-impl.sh").chmod(0o755)
        shutil.copy2(
            REPO / "scripts" / "lib" / "bench_supervision.py",
            lib / "bench_supervision.py",
        )
        shutil.copy2(
            REPO / "scripts" / "lib" / "bench-python.sh",
            lib / "bench-python.sh",
        )
        self.bench_lock = Path(tmp) / "bench-window.lock"
        self.gpu_lock = Path(tmp) / "metal-gpu.lock"
        self.pending = Path(tmp) / "bench-window-pending"
        lock_source = (REPO / "scripts" / "lib" / "bench-locks.py").read_text()
        for name, value in (
            ("BENCH_WINDOW", self.bench_lock),
            ("GPU_LOCK", self.gpu_lock),
            ("PENDING_DIR", self.pending),
        ):
            lock_source = re.sub(
                rf'^{name} = "[^"]*"$',
                f'{name} = "{value}"',
                lock_source,
                flags=re.M,
            )
        (lib / "bench-locks.py").write_text(lock_source)
        (lib / "quiet-probe.py").write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "label = sys.argv[sys.argv.index('--label') + 1]\n"
            "print(f'[quiet] {label}: idle 100.0% (floor 0.0%) ok | top: fixture 0.0%')\n"
        )
        (lib / "machine-state-probe.py").write_text(
            "#!/usr/bin/env python3\n"
            "import datetime, json, sys\n"
            "label = sys.argv[sys.argv.index('--label') + 1]\n"
            "print(json.dumps({'schema': 'lattice-machine-state-v1', 'label': label,"
            "'captured_at_utc': datetime.datetime.now(datetime.UTC)"
            ".strftime('%Y-%m-%dT%H:%M:%SZ'),"
            "'power': {'status': 'unavailable', 'reason': 'fixture'},"
            "'thermal': {'status': 'unavailable', 'reason': 'fixture'},"
            "'idle': {'status': 'unavailable', 'reason': 'fixture'}}))\n"
        )
        (root / "scripts" / "perf_governor.py").write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "from datetime import UTC, datetime\n"
            "label = sys.argv[sys.argv.index('--label') + 1]\n"
            "print(json.dumps({'schema': 'lattice-machine-state-v1', 'label': label,"
            "'captured_at_utc': datetime.now(UTC).replace(microsecond=0)"
            ".isoformat().replace('+00:00', 'Z'),"
            "'power': {'status': 'measured', 'source': 'fixture', 'state': 'ac'},"
            "'thermal': {'status': 'measured', 'source': 'fixture', 'state': 'nominal'},"
            "'idle': {'status': 'measured', 'source': 'fixture', 'seconds': 30.0},"
            "'gate': {'status': 'passed', 'cooldown_seconds': 30.0,"
            "'afk_threshold_seconds': 30.0, 'kill_switch': 'clear'}}))\n"
        )
        (root / "scripts" / "lib" / "ensure-noindex-marker.sh").write_text(
            "#!/usr/bin/env bash\nexit 0\n"
        )
        (root / "scripts" / "lib" / "ensure-noindex-marker.sh").chmod(0o755)

        shutil.copy2(REPO / ".gitignore", root / ".gitignore")
        env_git = {
            **os.environ,
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        }
        git = ("git", "-c", "core.hooksPath=/dev/null")
        subprocess.run([*git, "init", "-q", "-b", "main", str(root)], check=True)
        for i in range(2):
            (root / f"f{i}.txt").write_text(str(i))
            subprocess.run([*git, "-C", str(root), "add", "-A"], check=True)
            subprocess.run(
                [*git, "-C", str(root), "commit", "-qm", f"c{i}"],
                check=True, env=env_git,
            )

        bindir = Path(tmp) / "bin"
        bindir.mkdir()
        cargo_marker = Path(tmp) / "cargo-was-invoked"
        cargo = bindir / "cargo"
        cargo.write_text(
            "#!/usr/bin/env bash\n"
            f"printf ran > {str(cargo_marker)!r}\n"
            "if [[ \"$*\" == *--version* ]]; then printf '%s\\n' 'cargo 1.94.1 (fixture)'; fi\n"
            "exit 0\n"
        )
        cargo.chmod(0o755)
        self.cargo_marker = cargo_marker
        self.bindir = bindir
        return root

    def _write_forged_status(self, root: Path, own_pid: int) -> None:
        cache = root / ".cache"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "bench-locks-status.txt").write_text(
            f"supervisor_pid={own_pid}\n"
            f"lock=bench-window ({self.bench_lock}): fabricated\n"
            f"lock=Metal GPU ({self.gpu_lock}): fabricated\n"
        )

    def test_preflight_helper_cannot_observe_or_release_lock_capabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._build_repo(tmp)
            marker = Path(tmp) / "date-descriptor-probe"
            date = self.bindir / "date"
            date.write_text(
                "#!/usr/bin/env python3\n"
                "import fcntl, os\n"
                "from pathlib import Path\n"
                f"paths = ({str(self.bench_lock)!r}, {str(self.gpu_lock)!r})\n"
                "lock_ids = {\n"
                "    (os.stat(path).st_dev, os.stat(path).st_ino) for path in paths\n"
                "}\n"
                "observed = []\n"
                "for fd in range(3, 256):\n"
                "    try:\n"
                "        fd_stat = os.fstat(fd)\n"
                "    except OSError:\n"
                "        continue\n"
                "    if (fd_stat.st_dev, fd_stat.st_ino) in lock_ids:\n"
                "        observed.append(fd)\n"
                "        fcntl.flock(fd, fcntl.LOCK_UN)\n"
                "states = []\n"
                "for path in paths:\n"
                "    fd = os.open(path, os.O_RDWR)\n"
                "    try:\n"
                "        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                "    except OSError:\n"
                "        states.append('blocked')\n"
                "    else:\n"
                "        states.append('acquired')\n"
                "    finally:\n"
                "        os.close(fd)\n"
                f"Path({str(marker)!r}).write_text(\n"
                "    ('observed' if observed else 'hidden')\n"
                "    + ':' + ','.join(states)\n"
                ")\n"
                "print('2026-07-31T00:00:00Z')\n"
                "raise SystemExit(42)\n"
            )
            date.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{self.bindir}:{os.environ['PATH']}",
                "BENCH_HOST_ID": "fixture",
            }
            for name in (
                "LATTICE_BENCH_LOCK_STATUS",
                "LATTICE_BENCH_LOCK_FDS",
                "LATTICE_BENCH_SUPERVISOR_FD",
            ):
                env.pop(name, None)
            result = subprocess.run(
                [
                    "bash",
                    str(root / "scripts" / "bench-compare.sh"),
                    "HEAD~1",
                    "HEAD",
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            observed = marker.read_text()

        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertEqual(
            observed,
            "hidden:blocked,blocked",
            f"preflight helper inherited usable lock capabilities:\n{result.stderr}",
        )

    def _invoke(self, root: Path) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PATH": f"{self.bindir}:{os.environ['PATH']}",
            "BENCH_HOST_ID": "fixture",
            "BENCH_IDLE_FLOOR": "0",
        }
        env.pop("LATTICE_BENCH_LOCK_FDS", None)
        return subprocess.run(
            ["bash", str(root / "scripts" / "lib" / "bench-compare-impl.sh"),
             "HEAD~1", "HEAD"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

    def test_receipt_only_invocation_is_refused_before_any_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._build_repo(tmp)
            # The receipt contains a parseable caller-controlled PID, but no
            # descriptor-free handoff from the dedicated supervisor.
            self._write_forged_status(root, os.getpid())
            result = self._invoke(root)

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("LATTICE_BENCH_SUPERVISOR_FD", result.stderr)
        self.assertIn("refusing to measure", result.stderr)
        self.assertFalse(self.cargo_marker.exists())
        # Refused before the run-conditions banner, i.e. before base/head
        # resolution and worktree setup ever started.
        self.assertNotIn("=== bench-compare:", result.stdout)


class ImplBodyInheritsResolvedPythonInterpreter(unittest.TestCase):
    """bench-compare.sh resolves an interpreter meeting the harness's 3.11
    floor and exports it as PYTHON_BIN; bench-compare-impl.sh must consume
    that inherited variable rather than a bare `python3` lookup, or a stock
    macOS PATH (old /usr/bin/python3 ahead of a newer interpreter) makes the
    body die on the floor check instead of using the interpreter its own
    entry point already found (lattice#1333).

    This drives the real scripts/lib/bench-compare-impl.sh, not a rewritten
    copy: only scripts/lib/bench-host-id.py is needed alongside it, since the
    very first thing the body does (before touching git, locks, or cargo) is
    resolve a run host id through a Python interpreter.
    """

    def _build_fixture(self, tmp: str) -> Path:
        root = Path(tmp) / "repo"
        lib = root / "scripts" / "lib"
        lib.mkdir(parents=True)
        shutil.copy2(
            REPO / "scripts" / "lib" / "bench-compare-impl.sh",
            lib / "bench-compare-impl.sh",
        )
        (lib / "bench-compare-impl.sh").chmod(0o755)
        shutil.copy2(
            REPO / "scripts" / "lib" / "bench-host-id.py",
            lib / "bench-host-id.py",
        )
        return root

    def _run(self, tmp: str, root: Path) -> subprocess.CompletedProcess[str]:
        bindir = Path(tmp) / "bin"
        bindir.mkdir()
        stub_python3 = bindir / "python3"
        stub_python3.write_text(
            "#!/usr/bin/env bash\n"
            "echo 'STUB_PYTHON3_INVOKED: Python 3.9.6 (fixture, pre-floor)' >&2\n"
            "exit 17\n"
        )
        stub_python3.chmod(0o755)

        self.assertGreaterEqual(
            sys.version_info[:2],
            (3, 11),
            "the interpreter running this test must itself satisfy the "
            "harness's floor to stand in as the resolved PYTHON_BIN",
        )
        env = {
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "PYTHON_BIN": sys.executable,
            "LATTICE_BENCH_HOST_ID_FILE": f"{tmp}/bench-host-id",
        }
        for name in (
            "LATTICE_BENCH_LOCK_STATUS",
            "LATTICE_BENCH_LOCK_FDS",
            "LATTICE_BENCH_SUPERVISOR_FD",
        ):
            env.pop(name, None)
        return subprocess.run(
            ["bash", str(root / "scripts" / "lib" / "bench-compare-impl.sh")],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

    def test_impl_body_proceeds_past_host_id_handoff_instead_of_dying_on_stub(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._build_fixture(tmp)
            result = self._run(tmp, root)

        # The stub is never reached: the body used the inherited PYTHON_BIN
        # for the host-id resolution, so it proceeds to the next real
        # checkpoint (the supervisor lock-status refusal) instead of dying
        # inside the stub at exit 17.
        self.assertNotIn("STUB_PYTHON3_INVOKED", result.stderr, result.stderr)
        self.assertNotEqual(17, result.returncode, result.stderr)
        self.assertEqual(2, result.returncode, result.stderr)
        self.assertIn("no lock status at", result.stderr)
        self.assertIn(
            "Run scripts/bench-compare.sh, not this file directly", result.stderr
        )

    def test_impl_body_direct_invocation_resolves_its_own_interpreter_when_unset(
        self,
    ):
        """A direct invocation (no entry point, so PYTHON_BIN never arrived)
        must still end up with a qualifying interpreter, resolved by the
        body itself via scripts/lib/bench-python.sh -- not with bash's own
        "unbound variable" error (lattice#1333's original failure mode) and
        not with a silent fall back to an unqualified `python3`.

        PATH here is a fixture-only directory that shadows every name
        bench_resolve_python3 tries (python3.13, python3.12, python3.11,
        python3) with a stub that always fails the version-floor probe, so
        bench_require_python3 has no qualifying candidate regardless of
        what real interpreters the host running this test happens to ship.
        A bare /bin:/usr/bin PATH is not a portable way to get that: it
        relies on the host's system python3 being pre-3.11, which is true
        of macOS but not of Ubuntu 24.04 CI images, whose /usr/bin/python3
        is 3.12 and would satisfy the floor -- letting resolution succeed
        and fall through to the next real checkpoint (the lock-status
        guard) instead of exercising this refusal.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = self._build_fixture(tmp)
            shutil.copy2(
                REPO / "scripts" / "lib" / "bench-python.sh",
                root / "scripts" / "lib" / "bench-python.sh",
            )
            bindir = Path(tmp) / "bin"
            bindir.mkdir()
            for name in ("python3.13", "python3.12", "python3.11", "python3"):
                stub = bindir / name
                stub.write_text("#!/usr/bin/env bash\nexit 1\n")
                stub.chmod(0o755)
            env = {
                "PATH": f"{bindir}:/usr/bin:/bin",
                "LATTICE_BENCH_HOST_ID_FILE": f"{tmp}/bench-host-id",
            }
            for name in (
                "LATTICE_BENCH_LOCK_STATUS",
                "LATTICE_BENCH_LOCK_FDS",
                "LATTICE_BENCH_SUPERVISOR_FD",
                "PYTHON_BIN",
            ):
                env.pop(name, None)
            result = subprocess.run(
                ["bash", str(root / "scripts" / "lib" / "bench-compare-impl.sh")],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )

        self.assertEqual(2, result.returncode, result.stderr)
        self.assertNotIn("unbound variable", result.stderr, result.stderr)
        self.assertIn(
            "no Python >= 3.11 found on PATH", result.stderr, result.stderr
        )


class GpuHandoffProtocol(unittest.TestCase):
    """Exercise the shipping service with CPU-only canonical-lock fixtures."""

    @classmethod
    def setUpClass(cls):
        import runpy

        cls.module = runpy.run_path(str(REPO / "scripts" / "lib" / "bench_handoff.py"))

    def _run_fixture(
        self, *, target_mode="ready", requested_entry="command", quiet_ok=True,
        disconnect=False, replay=False, disconnect_after_ready=False,
    ):
        import json

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            gpu_path = root / "gpu.lock"
            result_path = root / "output"
            work_marker = root / "measured"
            descendant_pid = root / "descendant.pid"
            request_path = root / "request.json"
            helper_trace_path = root / "helper-trace.jsonl"
            refusal_state_path = root / "refusal-state"
            service_trace = []
            target_code = """
import fcntl, json, os, pathlib, socket, subprocess, sys, time
assert os.environ["LATTICE_GPU_HANDOFF_FD"] == "0"
gpu = pathlib.Path(os.environ["FIXTURE_GPU"])
assert os.fstat(0).st_ino == gpu.stat().st_ino
with gpu.open("r+") as independent:
    try:
        fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        raise AssertionError("supervisor GPU ownership was absent")
fcntl.flock(0, fcntl.LOCK_EX | fcntl.LOCK_NB)
if os.environ["FIXTURE_TARGET_MODE"] == "missing":
    print("time: 1 ns")
    raise SystemExit(0)
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect(os.environ["LATTICE_GPU_HANDOFF_CONTROL"])
message = dict(protocol=1, token=os.environ["LATTICE_GPU_HANDOFF_TOKEN"],
               pid=os.getpid(), exe=os.path.realpath(sys.executable))
if os.environ["FIXTURE_TARGET_MODE"] == "wrong-pid":
    message["pid"] += 1
client.sendall(json.dumps(message).encode() + b"\\n")
reply = client.makefile("rb").readline()
assert json.loads(reply)["status"] == "ready", reply
pathlib.Path(os.environ["FIXTURE_WORK"]).write_text("measured")
if os.environ["FIXTURE_TARGET_MODE"] == "silent":
    time.sleep(10)
if os.environ["FIXTURE_TARGET_MODE"] == "inherited-output":
    descendant = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    pathlib.Path(os.environ["FIXTURE_DESCENDANT_PID"]).write_text(str(descendant.pid))
print("time: 1 ns; measured after ACK")
"""
            helper_code = """
import base64, json, os, pathlib, socket, sys, time
def diagnostic(event, **fields):
    if os.environ["FIXTURE_TARGET_MODE"] == "inherited-output":
        with pathlib.Path(os.environ["FIXTURE_HELPER_TRACE"]).open("a") as trace:
            trace.write(json.dumps(dict(event=event, when=time.monotonic(), **fields)) + "\\n")
gpu = pathlib.Path(os.environ["FIXTURE_GPU"]).stat()
for fd in range(256):
    try:
        opened = os.fstat(fd)
    except OSError:
        continue
    assert (opened.st_dev, opened.st_ino) != (gpu.st_dev, gpu.st_ino), fd
request = json.loads(pathlib.Path(os.environ["FIXTURE_REQUEST"]).read_text())
request.update(protocol=1, token=os.environ["LATTICE_GPU_HANDOFF_BROKER_TOKEN"])
status = 0
for attempt in range(2 if os.environ["FIXTURE_REPLAY"] == "1" else 1):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(os.environ["LATTICE_GPU_HANDOFF_BROKER"])
    client.sendall(json.dumps(request).encode() + b"\\n")
    if os.environ["FIXTURE_DISCONNECT"] == "1":
        client.close()
        raise SystemExit(0)
    if os.environ["FIXTURE_DISCONNECT_AFTER_READY"] == "1":
        deadline = time.monotonic() + 3
        while not pathlib.Path(os.environ["FIXTURE_WORK"]).exists():
            if time.monotonic() >= deadline:
                raise AssertionError("target did not enter its measured phase")
            time.sleep(0.01)
        client.close()
        time.sleep(10)
        raise SystemExit(0)
    with client.makefile("rb") as stream:
        for line in stream:
            frame = json.loads(line)
            diagnostic("helper_frame_read", kind="output" if "output" in frame else "status")
            if "output" in frame:
                payload = base64.b64decode(frame["output"])
                if b"inherited output still open" in payload:
                    state = pathlib.Path(os.environ["FIXTURE_REFUSAL_STATE"])
                    schedule = os.environ["FIXTURE_CONSUMER_SCHEDULE"]
                    state.write_text("received")
                    diagnostic("helper_refusal_received", schedule=schedule)
                    if schedule in {"delay-250ms", "never-finish"}:
                        state.write_text("waiting-before-persist")
                        diagnostic("helper_wait_before_persist", schedule=schedule)
                        if schedule == "never-finish":
                            while True:
                                time.sleep(10)
                        time.sleep(0.25)
                        state.write_text("released-before-persist")
                        diagnostic("helper_delay_released")
                    elif schedule != "immediate":
                        raise AssertionError("unknown fixture consumer schedule")
                with pathlib.Path(os.environ["FIXTURE_OUTPUT"]).open("ab") as output:
                    output.write(base64.b64decode(frame["output"]))
                diagnostic("helper_output_saved", text=base64.b64decode(frame["output"]).decode())
            if "status" in frame:
                status = frame["status"]
                diagnostic("helper_status_read", status=status)
                break
    client.close()
raise SystemExit(status)
"""
            entry = {
                "id": "command", "revision": "a" * 40,
                "package": "lattice-inference", "target": "metal_decode_bench",
                "features": "f16,metal-gpu", "feature_set": ["f16", "metal-gpu"],
                "cwd": str(root), "run_cwd": str(root),
                "criterion_home": str(root / "criterion"),
                "argv": ["-c", target_code],
            }
            plan = {
                "version": 1, "mode": "command", "repo": str(root),
                "entries": [entry], "environment": {},
                "command": [sys.executable, "-c", helper_code],
            }
            request_path.write_text(json.dumps({"entry": requested_entry}))
            env = {
                **os.environ,
                "FIXTURE_GPU": str(gpu_path), "FIXTURE_REQUEST": str(request_path),
                "FIXTURE_OUTPUT": str(result_path), "FIXTURE_WORK": str(work_marker),
                "FIXTURE_TARGET_MODE": target_mode,
                "FIXTURE_HELPER_TRACE": str(helper_trace_path),
                "FIXTURE_REFUSAL_STATE": str(refusal_state_path),
                "FIXTURE_CONSUMER_SCHEDULE": os.environ.get("DRAIN_CONSUMER_SCHEDULE", "immediate"),
                "FIXTURE_DISCONNECT": str(int(disconnect)),
                "FIXTURE_DISCONNECT_AFTER_READY": str(int(disconnect_after_ready)),
                "FIXTURE_DESCENDANT_PID": str(descendant_pid),
                "FIXTURE_REPLAY": str(int(replay)),
            }
            quiet_calls = []
            def quiet(label, mode):
                self.assertFalse(work_marker.exists(), "work began before quiet certification")
                quiet_calls.append((label, mode))
                return quiet_ok, "inside measured-phase quiet sample\n"

            class DiagnosticService(self.module["HandoffService"]):
                def _launch(self, client, entry, request):
                    try:
                        return super()._launch(client, entry, request)
                    except Exception as exc:
                        service_trace.append({
                            "event": "launch_exception", "when": time.monotonic(),
                            "type": type(exc).__name__, "repr": repr(exc),
                            "message": str(exc), "client_timeout": client.gettimeout(),
                        })
                        raise

                def _output(self, client, output):
                    if target_mode != "inherited-output":
                        return super()._output(client, output)
                    service_trace.append({
                        "event": "service_output_start", "when": time.monotonic(),
                        "text": output.decode(errors="replace"),
                        "client_timeout": client.gettimeout(), "failure": self.failure,
                    })
                    try:
                        result = super()._output(client, output)
                    except Exception as exc:
                        service_trace.append({
                            "event": "service_output_exception", "when": time.monotonic(),
                            "type": type(exc).__name__, "repr": repr(exc),
                        })
                        raise
                    service_trace.append({
                        "event": "service_output_sent", "when": time.monotonic(),
                    })
                    return result

                def _serve(self):
                    try:
                        return super()._serve()
                    finally:
                        service_trace.append({
                            "event": "service_thread_finished", "when": time.monotonic(),
                            "failure": self.failure,
                        })

            with gpu_path.open("w+") as gpu:
                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                service = DiagnosticService(
                    plan, gpu.fileno(), lambda: None, quiet,
                    validate_request=lambda _entry, _request: Path(sys.executable),
                    ready_timeout=3,
                )
                self.assertLess(len(os.fsencode(service.broker_path)), 104)
                service_globals = service.run.__globals__
                original_terminate = service_globals["_terminate"]
                def trace_terminate(proc):
                    role = "none" if proc is None else (
                        "outer_helper" if proc.args == plan["command"] else "selected_target"
                    )
                    service_trace.append({
                        "event": "terminate_enter", "when": time.monotonic(),
                        "role": role, "pid": None if proc is None else proc.pid,
                        "returncode": None if proc is None else proc.poll(),
                        "failure": service.failure,
                        "refusal_state": refusal_state_path.read_text() if refusal_state_path.exists() else None,
                    })
                    try:
                        return original_terminate(proc)
                    finally:
                        service_trace.append({
                            "event": "terminate_return", "when": time.monotonic(),
                            "role": role, "pid": None if proc is None else proc.pid,
                            "returncode": None if proc is None else proc.poll(),
                        })
                started = time.monotonic()
                try:
                    with mock.patch.dict(service_globals, {"_terminate": trace_terminate}):
                        status = service.run(plan["command"], env, ())
                    service_trace.append({"event": "run_returned", "when": time.monotonic(), "status": status})
                finally:
                    if descendant_pid.exists():
                        try:
                            os.kill(int(descendant_pid.read_text()), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                elapsed = time.monotonic() - started
                with gpu_path.open("r+") as independent:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
                receipt = service.receipt_path.read_text()
            if target_mode == "inherited-output":
                helper_trace = [
                    json.loads(line) for line in helper_trace_path.read_text().splitlines()
                ] if helper_trace_path.exists() else []
            return {
                "status": status,
                "failure": service.failure,
                "service_trace": service_trace,
                "helper_trace": helper_trace if target_mode == "inherited-output" else [],
                "refusal_state": refusal_state_path.read_text() if refusal_state_path.exists() else None,
                "worked": work_marker.exists(),
                "quiet_calls": quiet_calls,
                "receipt": receipt,
                "output": result_path.read_text() if result_path.exists() else "",
                "elapsed": elapsed,
            }

    def test_selected_target_proves_lock_then_quiet_then_ack(self):
        result = self._run_fixture()
        self.assertEqual(0, result["status"])
        self.assertTrue(result["worked"])
        self.assertEqual(1, len(result["quiet_calls"]))
        self.assertIn("measured after ACK", result["output"])
        self.assertIn('"status": "ready"', result["receipt"])
        self.assertIn("inside measured-phase quiet", result["receipt"])

    def test_successful_process_without_ready_cannot_certify(self):
        result = self._run_fixture(target_mode="missing")
        self.assertEqual(2, result["status"])
        self.assertFalse(result["worked"])
        self.assertFalse(result["quiet_calls"])

    def test_wrong_pid_refuses_before_quiet_or_work(self):
        result = self._run_fixture(target_mode="wrong-pid")
        self.assertEqual(2, result["status"])
        self.assertFalse(result["worked"])
        self.assertFalse(result["quiet_calls"])

    def test_busy_measured_phase_refuses_before_work(self):
        result = self._run_fixture(quiet_ok=False)
        self.assertEqual(2, result["status"])
        self.assertFalse(result["worked"])
        self.assertIn('"status": "refused"', result["receipt"])

    def test_unknown_invocation_never_launches_target(self):
        result = self._run_fixture(requested_entry="head1")
        self.assertEqual(2, result["status"])
        self.assertFalse(result["worked"])
        self.assertFalse(result["quiet_calls"])

    def test_replay_cannot_make_completed_plan_succeed(self):
        result = self._run_fixture(replay=True)
        self.assertEqual(2, result["status"])
        self.assertEqual(1, len(result["quiet_calls"]))

    def test_disconnected_helper_cannot_certify(self):
        result = self._run_fixture(disconnect=True)
        self.assertEqual(2, result["status"])

    def test_exited_target_with_inherited_stdout_refuses_within_drain_bound(self):
        result = self._run_fixture(target_mode="inherited-output")
        self.assertEqual(2, result["status"])
        self.assertTrue(result["worked"])
        self.assertIn("inherited output still open", result["output"])
        self.assertLess(result["elapsed"], 5)

    def test_delayed_refusal_consumer_persists_reason_before_cleanup(self):
        with mock.patch.dict(os.environ, {"DRAIN_CONSUMER_SCHEDULE": "delay-250ms"}):
            result = self._run_fixture(target_mode="inherited-output")
        self.assertEqual(2, result["status"])
        self.assertTrue(result["worked"])
        self.assertIn("inherited output still open", result["output"])
        self.assertLess(result["elapsed"], 5)
        events = result["helper_trace"]
        names = [event["event"] for event in events]
        self.assertIn("helper_refusal_received", names)
        self.assertIn("helper_wait_before_persist", names)
        self.assertIn("helper_delay_released", names)
        refusal_saved = [event for event in events
                         if event["event"] == "helper_output_saved"
                         and "inherited output still open" in event["text"]]
        self.assertEqual(1, len(refusal_saved))
        received = next(event for event in events if event["event"] == "helper_refusal_received")
        waiting = next(event for event in events if event["event"] == "helper_wait_before_persist")
        released = next(event for event in events if event["event"] == "helper_delay_released")
        self.assertEqual("delay-250ms", received["schedule"])
        self.assertLess(received["when"], waiting["when"])
        self.assertLess(waiting["when"], released["when"])
        self.assertLess(released["when"], refusal_saved[0]["when"])
        self.assertEqual("released-before-persist", result["refusal_state"])
        status_read = [event for event in events if event["event"] == "helper_status_read"]
        self.assertEqual(1, len(status_read))
        self.assertEqual(2, status_read[0]["status"])
        helper_end = [event for event in result["service_trace"]
                      if event["event"] == "terminate_return" and event["role"] == "outer_helper"]
        self.assertEqual(1, len(helper_end))
        self.assertEqual(2, helper_end[0]["returncode"])

    def test_nonfinishing_refusal_consumer_reports_completion_expiry(self):
        import contextlib
        import io

        supervisor_stderr = io.StringIO()
        with contextlib.redirect_stderr(supervisor_stderr):
            with mock.patch.dict(os.environ, {"DRAIN_CONSUMER_SCHEDULE": "never-finish"}):
                result = self._run_fixture(target_mode="inherited-output")
        self.assertIn(
            "bench-supervision: refusal-completion deadline expired after 1.000s; forcing helper cleanup",
            supervisor_stderr.getvalue(),
        )
        self.assertIn(
            "selected benchmark exited with inherited output still open; refusing to certify",
            supervisor_stderr.getvalue(),
        )
        self.assertEqual(2, result["status"])
        self.assertTrue(result["worked"])
        self.assertEqual(
            "selected benchmark exited with inherited output still open", result["failure"]
        )
        self.assertLess(result["elapsed"], 5)
        self.assertEqual("waiting-before-persist", result["refusal_state"])
        helper_events = result["helper_trace"]
        received = [event for event in helper_events if event["event"] == "helper_refusal_received"]
        waiting = [event for event in helper_events if event["event"] == "helper_wait_before_persist"]
        self.assertEqual(1, len(received))
        self.assertEqual("never-finish", received[0]["schedule"])
        self.assertEqual(1, len(waiting))
        self.assertLess(received[0]["when"], waiting[0]["when"])
        self.assertFalse(any(event["event"] == "helper_delay_released" for event in helper_events))
        self.assertFalse(any(event["event"] == "helper_output_saved"
                             and "inherited output still open" in event["text"]
                             for event in helper_events))
        helper_end = [event for event in result["service_trace"]
                      if event["event"] == "terminate_return" and event["role"] == "outer_helper"]
        self.assertEqual(1, len(helper_end))
        self.assertEqual(-signal.SIGTERM, helper_end[0]["returncode"])
        self.assertTrue(any(event["event"] == "service_thread_finished"
                            for event in result["service_trace"]))

    def test_disconnected_live_helper_stops_a_silent_benchmark(self):
        result = self._run_fixture(target_mode="silent", disconnect_after_ready=True)
        self.assertEqual(2, result["status"])
        self.assertTrue(result["worked"])
        self.assertLess(result["elapsed"], 5)

    def test_receive_deadline_is_not_extended_by_partial_frame_progress(self):
        import socket
        import threading

        sender, receiver = socket.socketpair()
        release = threading.Event()
        def trickle():
            finish = time.monotonic() + 0.5
            try:
                while not release.wait(0.01) and time.monotonic() < finish:
                    sender.sendall(b" ")
            except OSError:
                pass
            finally:
                sender.close()
        worker = threading.Thread(target=trickle)
        worker.start()
        started = time.monotonic()
        try:
            with self.assertRaises((self.module["HandoffError"], TimeoutError)):
                self.module["_receive"](receiver, 4096, started + 0.05)
            self.assertLess(time.monotonic() - started, 0.4)
        finally:
            release.set()
            sender.close()
            receiver.close()
            worker.join(timeout=0.5)

    def test_frozen_plan_rejects_changes_and_partial_signals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            plan = {
                "version": 1, "mode": "command", "repo": str(root),
                "entries": [{"id": "command"}], "environment": {},
                "command": ["typed-helper"],
            }
            path, digest = self.module["write_frozen_plan"](root, plan)
            names = {
                self.module["PLAN_ENV"]: str(path),
                self.module["PLAN_DIGEST_ENV"]: digest,
            }
            with mock.patch.dict(os.environ, names, clear=True):
                self.assertEqual(
                    plan,
                    self.module["read_frozen_plan"](root, "command", ["typed-helper"]),
                )
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaisesRegex(self.module["HandoffError"], "changed"):
                    self.module["read_frozen_plan"](root)
            with mock.patch.dict(os.environ, {self.module["PLAN_ENV"]: str(path)}, clear=True):
                with self.assertRaisesRegex(self.module["HandoffError"], "missing"):
                    self.module["read_frozen_plan"](root)


class GpuLockSupervisorMarker(unittest.TestCase):
    """#1643: a launch route that retains GPU_LOCK without handing it to the
    child must name the pid actually holding it, so
    lattice_metal_runtime::measurement::gpu_test_lock_for_path's non-handoff
    branch can refuse fast instead of waiting out its own 30-minute timeout
    against a lock this process is holding.
    """

    def test_bench_supervision_plain_route_marks_child_with_supervisor_pid(self):
        """The pid named must be bench_supervision.py's own respawned pid,
        not bench-locks.py's: bench-locks.py hands the descriptors
        themselves off (LATTICE_BENCH_LOCK_FDS) to that respawned process,
        which is the one actually holding GPU_LOCK when `command` starts."""
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "receipt"
            code = (
                "import os\n"
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text("
                "os.environ.get('LATTICE_GPU_LOCK_SUPERVISOR_PID', '') "
                "+ ':' + str(os.getppid()))\n"
            )
            result = sb.run([sys.executable, "-c", code])
            self.assertEqual(result.returncode, 0, result.stderr)
            recorded_marker, _, ppid = marker.read_text().partition(":")
            self.assertTrue(recorded_marker.isdigit(), recorded_marker)
            self.assertEqual(recorded_marker, ppid)

    def test_bench_locks_plain_route_marks_child_with_its_own_pid(self):
        """bench-locks.py's non-`--pass-lock-fds` route retains both locks
        itself, so a child that self-locks must be told this process's pid
        rather than inherit an environment silent about the holder."""
        with _SupervisorSandbox() as sb:
            lib = sb.helper.parent
            status = Path(sb.tmp.name) / "direct-status.txt"
            marker = Path(sb.tmp.name) / "direct-receipt"
            code = (
                "import os\n"
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text("
                "os.environ.get('LATTICE_GPU_LOCK_SUPERVISOR_PID', '') "
                "+ ':' + str(os.getppid()))\n"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(lib / "bench-locks.py"),
                    "--label",
                    "fixture",
                    "--status-file",
                    str(status),
                    "--",
                    sys.executable,
                    "-c",
                    code,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            recorded_marker, _, ppid = marker.read_text().partition(":")
            self.assertTrue(recorded_marker.isdigit(), recorded_marker)
            self.assertEqual(recorded_marker, ppid)

    def test_bench_locks_pass_lock_fds_route_does_not_mark_its_immediate_child(self):
        """--pass-lock-fds hands the descriptors themselves to its immediate
        child (LATTICE_BENCH_LOCK_FDS) — a real handoff — so that child must
        not also see the supervisor marker."""
        with _SupervisorSandbox() as sb:
            lib = sb.helper.parent
            status = Path(sb.tmp.name) / "handoff-status.txt"
            marker = Path(sb.tmp.name) / "handoff-receipt"
            code = (
                "import os\n"
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text("
                "'present' if 'LATTICE_GPU_LOCK_SUPERVISOR_PID' in os.environ "
                "else 'absent')\n"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(lib / "bench-locks.py"),
                    "--label",
                    "fixture",
                    "--status-file",
                    str(status),
                    "--pass-lock-fds",
                    "--",
                    sys.executable,
                    "-c",
                    code,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(), "absent")


class _FailOnEmptyTestProgram(unittest.TestProgram):
    def runTests(self) -> None:
        if self.test.countTestCases() == 0:
            raise SystemExit("no tests collected")
        super().runTests()


if __name__ == "__main__":
    _FailOnEmptyTestProgram()
