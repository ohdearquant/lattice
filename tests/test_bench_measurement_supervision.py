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


class InventoryContract(unittest.TestCase):
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

    def test_node_measurement_forwards_lock_capabilities_to_verifier(self):
        """Node closes extra fds unless its verifier stdio maps them explicitly."""

        source = (REPO / "scripts" / "bench_wasm_simd.mjs").read_text()
        self.assertIn("LATTICE_BENCH_LOCK_FDS", source)
        self.assertIn("'--require-quiet'", source)
        self.assertIn("stdio: supervisionStdio()", source)
        self.assertIn("closeSync(fd)", source)
        self.assertIn("delete process.env.LATTICE_BENCH_LOCK_FDS", source)


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

    def test_python_entrypoint_retires_capabilities_before_work(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "python-entrypoint"
            entrypoint = sb.root / "scripts" / "entrypoint.py"
            entrypoint.write_text(
                "import os, sys\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0, {str(sb.helper.parent)!r})\n"
                "from bench_supervision import ensure_python_entrypoint\n"
                "ensure_python_entrypoint('fixture')\n"
                f"Path({str(marker)!r}).write_text("
                "'present' if 'LATTICE_BENCH_LOCK_FDS' in os.environ else 'retired')\n"
            )
            result = subprocess.run(
                [sys.executable, str(entrypoint)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(), "retired")

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

    def test_shell_entrypoint_retires_capabilities_before_work(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "shell-entrypoint"
            entrypoint = sb.root / "scripts" / "entrypoint.sh"
            entrypoint.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"source {str(sb.root / 'scripts/lib/bench-supervision.sh')!r}\n"
                'bench_supervise_entry "fixture" ordinary "$@"\n'
                '[[ -z "${LATTICE_BENCH_LOCK_FDS:-}" ]]\n'
                f"printf retired > {str(marker)!r}\n"
            )
            entrypoint.chmod(0o755)
            result = subprocess.run(
                ["bash", str(entrypoint)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(marker.read_text(), "retired")

    def test_durable_shell_refuses_lock_only_outer_without_errexit(self):
        with _SupervisorSandbox() as sb:
            marker = Path(sb.tmp.name) / "shell-must-not-run"
            entrypoint = sb.root / "scripts" / "durable_entrypoint.sh"
            entrypoint.write_text(
                "#!/usr/bin/env bash\n"
                "set -uo pipefail\n"
                f"source {str(sb.root / 'scripts/lib/bench-supervision.sh')!r}\n"
                'bench_supervise_entry "fixture" durable "$@"\n'
                f"printf ran > {str(marker)!r}\n"
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
    def test_node_child_can_forward_lock_capabilities_to_verifier(self):
        """Mutation-sensitive: ordinary spawn closes the inherited lock fds."""

        with _SupervisorSandbox() as sb:
            code = """
const {spawnSync} = require('node:child_process');
const fds = process.env.LATTICE_BENCH_LOCK_FDS.split(',').map(Number);
const stdio = ['ignore', 'pipe', 'pipe'];
for (const fd of fds) {
  while (stdio.length <= fd) stdio.push('ignore');
  stdio[fd] = fd;
}
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

    def test_forged_nonancestor_receipt_is_refused(self):
        with _SupervisorSandbox() as sb:
            status = Path(sb.tmp.name) / "forged.status"
            status.write_text(
                "supervisor_pid=1\n"
                "lock=bench-window (/tmp/fake): fabricated\n"
                "lock=Metal GPU (/tmp/fake): fabricated\n"
            )
            result = subprocess.run(
                [sys.executable, str(sb.helper), "verify"],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "LATTICE_BENCH_LOCK_STATUS": str(status),
                },
                timeout=30,
            )
            self.assertEqual(result.returncode, 2)
            self.assertRegex(result.stderr, r"not an ancestor|could not inspect")

    def test_unlocked_inherited_fds_do_not_borrow_another_holders_proof(self):
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
            self.assertIn("does not carry the lock", result.stderr)


if __name__ == "__main__":
    unittest.main()
