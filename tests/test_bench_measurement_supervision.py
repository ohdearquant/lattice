#!/usr/bin/env python3
"""Contract tests for the local measurement-path inventory and supervisor."""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "scripts" / "bench-measurements.toml"
SPECIAL_ENTRYPOINTS = {
    "scripts/compare_logits.py",
    "scripts/e2e-parity-local.sh",
    "scripts/e2e_parity_check.py",
    "scripts/fake_quant_pilot.py",
    "scripts/perf_governor.py",
}
INTERNAL_OR_TEST = {
    "scripts/ensure-noindex-marker-selftest.sh",
    "scripts/lib/bench-compare-impl.sh",
}
MEASUREMENT_SIGNAL = re.compile(
    r"perf_counter(?:_ns)?\(|hrtime\.bigint\(|cargo bench|tok_per_sec|"
    r"elapsed_ns|elapsed_s|total_ms|tokens/s|tok/s|PPL:"
)


def manifest_entries() -> dict[str, dict[str, str]]:
    data = tomllib.loads(MANIFEST.read_text())
    return {entry["path"]: entry for entry in data["entry"]}


def excluded_measurement_surfaces() -> set[str]:
    data = tomllib.loads(MANIFEST.read_text())
    return {
        path
        for surface in data["excluded_surface"]
        for path in surface["paths"]
    }


def discovered_declared_rust_inventory_paths() -> set[str]:
    paths = {
        str(path.relative_to(REPO))
        for path in (REPO / "crates").glob("*/benches/*.rs")
    }
    paths.update(
        str(path.relative_to(REPO))
        for path in (REPO / "crates/inference/examples").glob("bench*.rs")
    )
    paths.update(
        str(path.relative_to(REPO))
        for path in (REPO / "crates/inference/src/bin").glob("*.rs")
        if path.stem.startswith("bench_")
        or path.stem in {"eval_perplexity", "gramperf_profile", "ppl_metal"}
    )
    paths.add("README.md")
    return paths


class InventoryContract(unittest.TestCase):
    def test_canonical_lock_paths_are_pinned(self):
        source = (REPO / "scripts" / "lib" / "bench-locks.py").read_text()
        self.assertRegex(
            source,
            r'(?m)^BENCH_WINDOW = "/tmp/lion-bench-window\.lock"$',
        )
        self.assertRegex(
            source,
            r'(?m)^GPU_LOCK = "/tmp/lion-metal-gpu-test\.lock"$',
        )

    def test_manifest_schema_is_explicit_and_nonduplicated(self):
        data = tomllib.loads(MANIFEST.read_text())
        self.assertEqual(data["schema"], 1)
        entries = data["entry"]
        paths = [entry["path"] for entry in entries]
        self.assertEqual(len(paths), len(set(paths)))
        for entry in entries:
            with self.subTest(path=entry["path"]):
                self.assertIn(
                    entry["role"],
                    {"measurement", "consumer", "policy-check", "supervisor"},
                )
                self.assertIn(
                    entry["supervision"],
                    {
                        "none",
                        "both-locks",
                        "both-locks+quiet",
                        "both-locks+quiet-baseline",
                        "both-locks+three-phase-quiet",
                    },
                )
                self.assertTrue((REPO / entry["path"]).is_file())
                if entry["role"] == "measurement":
                    self.assertNotEqual(entry["supervision"], "none")

    def test_declared_rust_inventory_grammar_is_exact_and_fail_closed(self):
        """A Rust path matching the declared grammar must be classified."""

        data = tomllib.loads(MANIFEST.read_text())
        contract = data["contract"]
        self.assertEqual(contract["caller_trust"], "cooperative")
        self.assertEqual(
            contract["handoff_check"],
            "instantaneous silent-pipe open-writer and lock-contention "
            "diagnostics; not authenticated ownership, continuous lock-lifetime "
            "proof, or deliberate same-user bypass resistance",
        )
        self.assertEqual(
            contract["rust_inventory_grammar"],
            "crates/*/benches/*.rs; crates/inference/examples/bench*.rs; "
            "crates/inference/src/bin/bench_*.rs plus eval_perplexity.rs, "
            "gramperf_profile.rs, and ppl_metal.rs; README.md",
        )
        self.assertEqual(
            contract["rust_inventory_limitation"],
            "does not discover other Rust examples, binaries, or tests",
        )
        confirmed_outside = {
            "crates/inference/examples/profile_metal_decode.rs",
            "crates/inference/examples/profile_metal.rs",
            "crates/inference/examples/decode_profile.rs",
            "crates/inference/examples/layer_sweep.rs",
            "crates/tune/tests/bench_backward_737.rs",
        }
        self.assertEqual(
            set(contract["confirmed_outside_rust_inventory"]), confirmed_outside
        )
        self.assertTrue(
            confirmed_outside.isdisjoint(discovered_declared_rust_inventory_paths())
        )
        for path in confirmed_outside:
            self.assertTrue((REPO / path).is_file(), path)
        surfaces = data["excluded_surface"]
        self.assertTrue(surfaces)
        for surface in surfaces:
            with self.subTest(family=surface["family"]):
                self.assertRegex(surface["tracking_issue"], r"^#[1-9][0-9]*$")
                self.assertTrue(surface["reason"])
                self.assertTrue(surface["paths"])
                for path in surface["paths"]:
                    self.assertTrue((REPO / path).is_file(), path)
        excluded = excluded_measurement_surfaces()
        self.assertEqual(len(excluded), sum(len(s["paths"]) for s in surfaces))
        self.assertEqual(excluded, discovered_declared_rust_inventory_paths())

    def test_every_benchmark_named_script_is_classified(self):
        """A new bench script cannot appear without an explicit classification."""

        discovered = {
            str(path.relative_to(REPO))
            for path in (REPO / "scripts").iterdir()
            if path.is_file()
            and path.name.startswith("bench")
            and path.suffix in {".py", ".sh", ".mjs"}
        }
        discovered.update(SPECIAL_ENTRYPOINTS)
        discovered.update(
            str(path.relative_to(REPO))
            for path in (REPO / "scripts").rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".sh", ".mjs"}
            and MEASUREMENT_SIGNAL.search(path.read_text())
        )
        discovered.difference_update(INTERNAL_OR_TEST)
        self.assertEqual(set(manifest_entries()), discovered)

    def test_every_measurement_entry_has_a_live_guard(self):
        """Mutation-sensitive: deleting any entry-point guard fails this scan."""

        for path, entry in manifest_entries().items():
            if entry["role"] != "measurement":
                continue
            source = (REPO / path).read_text()
            with self.subTest(path=path):
                if path == "scripts/bench-compare.sh":
                    self.assertIn("bench-locks.py", source)
                elif path.endswith(".py"):
                    self.assertIn("ensure_python_entrypoint(", source)
                elif path.endswith(".mjs"):
                    self.assertIn("bench_supervision.py", source)
                    self.assertIn("'verify'", source)
                else:
                    self.assertIn("bench_supervise_entry", source)

    def test_make_delegates_whole_durable_recipes(self):
        """The lock must cover the recipe, not one command inside Make quoting."""

        makefile = (REPO / "Makefile").read_text()
        for target, script in (
            ("bench-ci", "./scripts/bench-ci.sh"),
            ("bench-gate", "./scripts/bench-gate.sh"),
        ):
            match = re.search(
                rf"(?m)^{re.escape(target)}:\n(?P<body>(?:\t.*\n)+)",
                makefile,
            )
            self.assertIsNotNone(match, target)
            body = [
                line.strip()
                for line in match.group("body").splitlines()
                if line.strip()
            ]
            self.assertEqual(body, [script])

    def test_make_has_no_raw_cargo_bench_recipe(self):
        """A new Make measurement must enter through a supervised script."""

        recipes = [
            line.strip()
            for line in (REPO / "Makefile").read_text().splitlines()
            if line.startswith("\t")
        ]
        self.assertFalse(
            [line for line in recipes if re.search(r"\bcargo\s+bench\b", line)]
        )

    def test_durable_multi_target_recipes_probe_between_targets(self):
        """Mutation-sensitive: the outer before/after probes are not a midpoint."""

        for path in ("scripts/bench-ci.sh", "scripts/bench-gate.sh"):
            source = (REPO / path).read_text()
            with self.subTest(path=path):
                self.assertIn('bench_supervise_entry "', source)
                self.assertIn(" durable ", source)
                self.assertGreaterEqual(source.count("bench_quiet_checkpoint"), 2)
                self.assertIn("between targets", source)
                final_probe = source.rfind("bench_quiet_checkpoint")
                final_measurement = source.rfind("cargo bench")
                self.assertGreater(final_probe, final_measurement)
                if path == "scripts/bench-gate.sh":
                    self.assertLess(
                        final_probe,
                        source.index("python3 scripts/perf-bench-gate.py"),
                    )

    def test_fake_quant_does_not_nest_its_old_gpu_only_lock(self):
        """The outer both-lock supervisor and an inner flock would deadlock."""

        source = (REPO / "scripts" / "fake_quant_pilot.py").read_text()
        self.assertNotIn("fcntl.flock", source)
        self.assertNotIn("GPU_LOCK_PATH", source)

    def test_node_measurement_forwards_pipe_to_handoff_sample(self):
        """Node closes extra fds unless the handoff sample maps the pipe."""

        source = (REPO / "scripts" / "bench_wasm_simd.mjs").read_text()
        self.assertIn("LATTICE_BENCH_SUPERVISOR_FD", source)
        self.assertNotIn("LATTICE_BENCH_LOCK_FDS", source)
        self.assertIn("'--require-quiet'", source)
        self.assertIn("stdio: supervisionStdio()", source)
        self.assertIn("closeSync(SUPERVISOR_FD)", source)
        self.assertIn("delete process.env.LATTICE_BENCH_SUPERVISOR_FD", source)
        self.assertGreaterEqual(source.count("[SUPERVISION, 'verify'"), 2)
        self.assertNotIn("verify-retained", source)

    def test_possessing_entrypoints_recheck_canonical_paths(self):
        for path in (
            "scripts/lib/bench-supervision.sh",
            "scripts/lib/bench-compare-impl.sh",
        ):
            with self.subTest(path=path):
                source = (REPO / path).read_text()
                self.assertGreaterEqual(
                    len(re.findall(r'python3 "[^"]+" verify', source)), 2
                )
                self.assertNotIn("verify-retained", source)


class _SupervisorSandbox:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()

    def __enter__(self):
        self.root = Path(self.tmp.name) / "repo"
        lib = self.root / "scripts" / "lib"
        lib.mkdir(parents=True)
        for name in ("bench_supervision.py", "bench-locks.py", "quiet-probe.py"):
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
        return self

    def __exit__(self, *exc):
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


class RuntimeContract(unittest.TestCase):
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

    def test_substitute_and_restore_is_outside_cooperative_contract(self):
        """Endpoint identity samples do not certify hostile pathname continuity."""

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
            self.assertEqual(proc.returncode, 0, f"{stdout}\n{stderr}")
            self.assertEqual(marker.read_text(), "restored")

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
            self.assertEqual(proc.returncode, 0, f"{stdout}\n{stderr}")
            self.assertEqual(marker.read_text(), "accepted:acquired,acquired")

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
        """Mutation-sensitive: ordinary spawn closes the liveness witness."""

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
    """scripts/lib/bench-compare-impl.sh verify_locks must require inherited
    lock descriptors, not just an ancestor-PID relation in a caller-supplied
    status file.

    Mutation-sensitive: this reproduces the exact receipt-only exploit —
    a caller invokes the body directly after writing a status file whose
    supervisor_pid is its own PID (trivially "an ancestor" of the child bash
    process it then spawns), with no LATTICE_BENCH_LOCK_FDS. Neither lock is
    held. With the fix, verify_locks refuses (exit 2) before touching cargo.
    Reverting the fix (restoring the ancestry-only check) must make the same
    invocation exit 0, proving the bypass was real.
    """

    def _build_repo(self, tmp: str) -> Path:
        root = Path(tmp) / "repo"
        lib = root / "scripts" / "lib"
        lib.mkdir(parents=True)
        shutil.copy2(
            REPO / "scripts" / "lib" / "bench-compare-impl.sh",
            lib / "bench-compare-impl.sh",
        )
        (lib / "bench-compare-impl.sh").chmod(0o755)
        shutil.copy2(
            REPO / "scripts" / "lib" / "bench_supervision.py",
            lib / "bench_supervision.py",
        )
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
            "lock=bench-window (/tmp/fake-bench-window.lock): fabricated\n"
            "lock=Metal GPU (/tmp/fake-metal-gpu.lock): fabricated\n"
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
            # own_pid is this test process's PID -- the forged supervisor_pid
            # a caller would write for itself. It IS an ancestor of the bash
            # child spawned below (its direct parent), which is exactly why
            # the ancestry check alone used to accept it.
            self._write_forged_status(root, os.getpid())
            result = self._invoke(root)

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("LATTICE_BENCH_LOCK_FDS", result.stderr)
        self.assertIn("refusing to measure", result.stderr)
        self.assertFalse(self.cargo_marker.exists())
        # Refused before the run-conditions banner, i.e. before base/head
        # resolution and worktree setup ever started.
        self.assertNotIn("=== bench-compare:", result.stdout)


if __name__ == "__main__":
    unittest.main()
