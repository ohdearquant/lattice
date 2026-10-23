#!/usr/bin/env python3
"""Regression tests for the GPU-handoff launches of the shipping benchmark scripts.

These classes run the real scripts the way an operator does: `bench-compare.sh` and
`bench-command.sh` started through the system bash and reaching the supervisor with their
arguments intact, the `--gpu-handoff` command route for declared benchmark and binary
targets, and the slopefit, quality and logit scripts that launch their binaries through the
handoff. Each case builds a disposable repository with stub cargo, uv and Python modules, so
no model, GPU, network or benchmark is used.

They were split out of tests/test_bench_compare_measurement.py, which keeps the
measurement-integrity guard, the in-phase load report and GPU benchmark admission.
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "scripts" / "lib"

# Disable hooks and auto-gc so disposable Git fixtures have no automatic GC writer.
GIT = ("git", "-c", "core.hooksPath=/dev/null", "-c", "gc.auto=0")


class SystemBashEntrypoints(unittest.TestCase):
    def run_wrapper(self, name, args):
        import json

        with tempfile.TemporaryDirectory(prefix="bench-system-bash-") as temporary:
            root = Path(temporary)
            lib = root / "scripts/lib"
            lib.mkdir(parents=True)
            shutil.copy2(REPO / "scripts" / name, root / "scripts" / name)
            shutil.copy2(LIB / "bench-python.sh", lib / "bench-python.sh")
            # Capture the wrapper boundary without entering a measurement window.
            (lib / "bench_supervision.py").write_text(
                "import json,sys\nprint(json.dumps(sys.argv[1:]))\n"
            )
            bindir = root / "bin"
            bindir.mkdir()
            (bindir / "python3.13").symlink_to(sys.executable)
            result = subprocess.run(
                ["/bin/bash", str(root / "scripts" / name), *args],
                env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"},
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            forwarded = json.loads(result.stdout)
            return forwarded, str(root / "scripts/lib/bench-compare-impl.sh")

    def test_compare_without_arguments_reaches_supervisor(self):
        forwarded, implementation = self.run_wrapper("bench-compare.sh", [])
        self.assertEqual(forwarded, ["run", "--label", "bench-compare", "--entrypoint", "--", implementation])

    def test_compare_handoff_without_refs_reaches_supervisor(self):
        forwarded, implementation = self.run_wrapper("bench-compare.sh", ["--gpu-handoff"])
        self.assertEqual(forwarded, ["run", "--label", "bench-compare", "--entrypoint",
                                     "--gpu-handoff", "compare", "--", implementation])

    def test_compare_handoff_preserves_argument_boundaries(self):
        args = ["--full", "--", "base ref", ""]
        forwarded, implementation = self.run_wrapper("bench-compare.sh", ["--gpu-handoff", *args])
        self.assertEqual(forwarded, ["run", "--label", "bench-compare", "--entrypoint",
                                     "--gpu-handoff", "compare", "--", implementation, *args])

    def test_ordinary_command_reaches_supervisor_with_exact_arguments(self):
        # --entrypoint must always be forwarded: it is what lets a wrapped
        # command that is itself a self-supervising Python entry point (one
        # that calls ensure_python_entrypoint) receive the liveness pipe it
        # looks for instead of refusing with "LATTICE_BENCH_SUPERVISOR_FD is
        # not set". An ordinary command never reads that pipe, so the flag
        # is unconditional here, matching bench-compare.sh and
        # bench_supervise_entry, which already pass it unconditionally too.
        command = ["printf", "%s", "two words", ""]
        for durable in (False, True):
            with self.subTest(durable=durable):
                flags = ["--durable"] if durable else []
                forwarded, _ = self.run_wrapper("bench-command.sh", ["--label", "fixture label", *flags, "--", *command])
                quiet = ["--quiet"] if durable else []
                self.assertEqual(forwarded, ["run", "--label", "fixture label", *quiet, "--entrypoint", "--", *command])

    def test_command_handoff_reaches_supervisor_with_exact_arguments(self):
        command = ["cargo", "bench", "--", "two words", ""]
        for durable in (False, True):
            with self.subTest(durable=durable):
                flags = ["--durable"] if durable else []
                forwarded, _ = self.run_wrapper("bench-command.sh", ["--gpu-handoff", "--label", "fixture label", *flags, "--", *command])
                quiet = ["--quiet"] if durable else []
                self.assertEqual(forwarded, ["run", "--label", "fixture label", *quiet, "--entrypoint",
                                             "--gpu-handoff", "command", "--", *command])


# The stand-in for `uv run ...`: it records its stdin, its arguments and whether both machine
# locks are held by someone else while it runs, then prints the rows/diagnostics the test asks for.
UV_STUB = r'''
import fcntl,json,os,pathlib,sys
def held(path):
    with open(path, "r+") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False
events = os.environ.get("FIXTURE_EVENTS")
if events:
    open(events, "a").write("mlx-start\n")
pathlib.Path(os.environ["FIXTURE_UV_STDIN"]).write_bytes(sys.stdin.buffer.read())
pathlib.Path(os.environ["FIXTURE_UV_ARGS"]).write_text(" ".join(sys.argv[1:]) + "\n")
pathlib.Path(os.environ["FIXTURE_UV_RECORD"]).write_text(json.dumps(dict(
    gpu_lock_held=held(os.environ["FIXTURE_GPU"]), window_held=held(os.environ["FIXTURE_WINDOW"]),
    supervisor_marker="LATTICE_GPU_LOCK_SUPERVISOR_PID" in os.environ,
    handoff_env=sorted(name for name in os.environ if name.startswith("LATTICE_GPU_HANDOFF_")),
    lock_fds="LATTICE_BENCH_LOCK_FDS" in os.environ)))
sys.stdout.write(os.environ.get("FIXTURE_UV_OUT", "").replace("\\t", "\t").replace("\\n", "\n"))
sys.stderr.write(os.environ.get("FIXTURE_UV_ERR", ""))
if events:
    open(events, "a").write("mlx-end\n")
sys.exit(int(os.environ["FIXTURE_UV_RC"]))
'''


class GpuHandoffShippingCommand(unittest.TestCase):
    BINS = ["bench_decode_ab", "bench_decode_slopefit", "bench_logit_dump", "eval_perplexity"]

    def run_fixture(self, *, eligibility="valid", artifact_valid=True, kind="bench", bin_args=("--q4-dir", "model"),
                    bin_name="eval_perplexity", second_acquisition=False, launch=None, commit_files=None,
                    extra_env=None, gitignore=None, stdout_text=None, stderr_text=None, logits=False,
                    features=("default", "metal-gpu", "std"), uv_rc=0, read_files=()):
        """`launch(root, temp)` returns (argv, cwd) for a shipping script that runs the
        fixture binary through the real bench-command.sh instead of calling it directly;
        each launch of the fixture binary is then recorded, not asserted against bin_args."""
        import json
        with tempfile.TemporaryDirectory(prefix="admission-command-") as temporary:
            temp = Path(temporary).resolve()
            root = temp / "repo"
            lib = root / "scripts/lib"
            lib.mkdir(parents=True)
            for name in ("bench_admission.py", "bench_handoff.py", "bench_supervision.py",
                         "ensure-noindex-marker.sh",
                         "bench-locks.py", "bench-python.sh", "quiet-probe.py"):
                shutil.copy2(LIB / name, lib / name)
            shutil.copy2(REPO / "scripts/bench-command.sh", root / "scripts/bench-command.sh")
            # The fixture substitutes the host/Metal workload, never admission or RPC validation.
            source = (lib / "bench_admission.py").read_text()
            self.assertIn('if sys.platform != "darwin":', source)
            (lib / "bench_admission.py").write_text(source.replace('if sys.platform != "darwin":', 'if False:', 1))
            lock_names = {"BENCH_WINDOW": temp / "window.lock", "GPU_LOCK": temp / "gpu.lock", "PENDING_DIR": temp / "pending"}
            source = (lib / "bench-locks.py").read_text()
            for name, path in lock_names.items():
                source, count = re.subn(rf'^{name} = "[^"]*"$', f'{name} = "{path}"', source, flags=re.M)
                self.assertEqual(count, 1)
            (lib / "bench-locks.py").write_text(source)
            (lib / "quiet-probe.py").write_text(
                "import os,pathlib,sys\n"
                "assert not pathlib.Path(os.environ['FIXTURE_WORK']).exists()\n"
                "pathlib.Path(os.environ['FIXTURE_QUIET']).write_text('inside guard')\n"
                "open(os.environ['FIXTURE_QUIET'] + '.log', 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
                "if os.environ.get('FIXTURE_QUIET_FAIL') and os.environ['FIXTURE_QUIET_FAIL'] in ' '.join(sys.argv[1:]):\n"
                "    print('fixture: machine not quiet'); sys.exit(1)\n"
                "print('fixture inside-guard CPU idle sample')\n")
            (root / ".gitignore").write_text(".cache/\ntarget/\n__pycache__/\n" if gitignore is None else gitignore)
            (root / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/inference"]\n[workspace.package]\nversion = "0.0.1"\n')
            (root / "Cargo.lock").write_text("version = 4\n")
            package = root / "crates/inference"
            (package / "benches").mkdir(parents=True)
            targets = ["cross_turn_prefix_cache_bench", "decode_attn_bench", "lm_head_bench",
                       "metal_decode_bench", "mtp_decode", "topk_readback"]
            policy = "" if eligibility == "absent" else (
                '[package.metadata.gpu-bench-handoff]\nversion = ' + ('1' if eligibility == "valid" else '2')
                + '\ntargets = ' + json.dumps(targets) + '\nbins = ' + json.dumps(self.BINS) + '\n')
            manifest = '[package]\nname = "lattice-inference"\nversion.workspace = true\n' + policy
            manifest += '[features]\ndefault = ["std"]\nstd = []\nmetal-gpu = []\nf16 = []\nbench-internals = []\n'
            for target in targets:
                manifest += f'[[bench]]\nname = "{target}"\nharness = false\n'
                (package / f"benches/{target}.rs").write_text("fn main() {}\n")
            (package / "Cargo.toml").write_text(manifest)
            (package / "src/bin").mkdir(parents=True)
            for name in self.BINS:
                (package / f"src/bin/{name}.rs").write_text("fn main() {}\n")
            (package / "fixture-model.txt").write_text("package-relative model")
            (root / "fixture-model.txt").write_text("repo-relative model")
            for relative, content in (commit_files or {}).items():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, Path):
                    shutil.copy2(content, root / relative)
                else:
                    (root / relative).write_text(content)
            env_git = {**os.environ, "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
                       "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
            subprocess.run([*GIT, "init", "-q", str(root)], check=True, env=env_git)
            subprocess.run([*GIT, "-C", str(root), "add", "."], check=True, env=env_git)
            subprocess.run([*GIT, "-C", str(root), "commit", "-qm", "fixture"], check=True, env=env_git)
            executable_source = temp / "target-fixture.py"
            executable_source.write_text(f"#!{sys.executable}\n" + r'''
import fcntl,json,os,pathlib,socket,sys
binary = os.environ["FIXTURE_KIND"] == "bin"
repo = pathlib.Path(os.environ["FIXTURE_REPO"])
package = repo / "crates/inference"
# cargo bench runs from the package directory; cargo run keeps the caller's cwd.
expected_cwd = repo if binary else package
assert pathlib.Path.cwd() == expected_cwd, (pathlib.Path.cwd(), expected_cwd)
assert pathlib.Path(os.environ["LATTICE_MODEL_DIR"]).read_text() == ("repo-relative model" if binary else "package-relative model")
assert os.environ.get("FIXTURE_LAUNCHES") or sys.argv[1:] == (os.environ["FIXTURE_ARGS"].split() if binary else ["--bench", "lookup", "--quick"]), sys.argv
assert ("CRITERION_HOME" in os.environ) != binary
assert "LATTICE_GPU_HANDOFF_BROKER_TOKEN" not in os.environ
assert "LATTICE_BENCH_LOCK_FDS" not in os.environ
gpu = pathlib.Path(os.environ["FIXTURE_GPU"])
held, canonical = os.fstat(0), gpu.stat()
assert (held.st_dev, held.st_ino) == (canonical.st_dev, canonical.st_ino)
with gpu.open("r+") as probe:
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pass
    else:
        raise AssertionError("GPU lock was not held")
fcntl.flock(0, fcntl.LOCK_EX | fcntl.LOCK_NB)
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
    connection.connect(os.environ["LATTICE_GPU_HANDOFF_CONTROL"])
    token = os.environ["LATTICE_GPU_HANDOFF_TOKEN"]
    connection.sendall(json.dumps(dict(protocol=1, token=token, pid=os.getpid(), exe=str(pathlib.Path(__file__).resolve()))).encode() + b"\n")
    with connection.makefile("rb") as stream:
        response = json.loads(stream.readline())
    assert response == dict(protocol=1, token=token, status="ready"), response
home = None if binary else pathlib.Path(os.environ["CRITERION_HOME"])
assert home is None or home == package / "relative-evidence", home
assert pathlib.Path(os.environ["FIXTURE_QUIET"]).exists()
record = dict(cwd=str(pathlib.Path.cwd()), criterion_home=None if home is None else str(home))
if os.environ["FIXTURE_SECOND"] == "1":
    # A second gpu_test_lock() in one process repeats the handshake on the same control path.
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as again:
            again.connect(os.environ["LATTICE_GPU_HANDOFF_CONTROL"])
        record["second"] = "connected"
    except OSError as error:
        record["second"] = type(error).__name__
if os.environ.get("FIXTURE_EVENTS"):
    open(os.environ["FIXTURE_EVENTS"], "a").write("lattice-start\n")
if os.environ.get("FIXTURE_LAUNCHES"):
    with open(os.environ["FIXTURE_LAUNCHES"], "a") as launches:
        launches.write(json.dumps(dict(argv=sys.argv[1:], cwd=str(pathlib.Path.cwd()),
            model=os.environ.get("LATTICE_MODEL_DIR"), tokenizer=os.environ.get("LATTICE_TOKENIZER_DIR"),
            out=os.environ.get("LATTICE_LOGIT_OUT"))) + "\n")
else:
    pathlib.Path(os.environ["FIXTURE_WORK"]).write_text(json.dumps(record))
if os.environ.get("FIXTURE_LOGITS") == "1":
    assert os.path.isabs(os.environ["LATTICE_LOGIT_OUT"]), os.environ["LATTICE_LOGIT_OUT"]
    pathlib.Path(os.environ["LATTICE_LOGIT_OUT"]).write_bytes(bytes(32))
if os.environ.get("FIXTURE_STDERR"):
    print(os.environ["FIXTURE_STDERR"], file=sys.stderr, flush=True)
print(os.environ.get("FIXTURE_STDOUT") or ("RESULT fixture binary" if binary else "lookup time: [1.0 ns 1.1 ns 1.2 ns]"), flush=True)
if os.environ.get("FIXTURE_EVENTS"):
    open(os.environ["FIXTURE_EVENTS"], "a").write("lattice-end\n")
''')
            bindir = temp / "bin"
            bindir.mkdir()
            for version in ("python3.13", "python3.12", "python3.11", "python3"):
                (bindir / version).symlink_to(sys.executable)
            cargo = bindir / "cargo"
            cargo.write_text(f"#!{sys.executable}\n" + r'''
import json,os,pathlib,shutil,sys
binary = os.environ["FIXTURE_KIND"] == "bin"
assert sys.argv[1] == ("build" if binary else "bench"), sys.argv
assert "--locked" in sys.argv and "--message-format=json" in sys.argv
assert ("--no-run" in sys.argv) != binary and ("--bin" in sys.argv) == binary and ("--release" in sys.argv) == binary
assert not any(name.startswith("LATTICE_GPU_HANDOFF_") for name in os.environ)
assert "LATTICE_BENCH_LOCK_FDS" not in os.environ
assert "LATTICE_BENCH_SUPERVISOR_FD" not in os.environ
gpu = pathlib.Path(os.environ["FIXTURE_GPU"]).stat()
for fd in range(256):
    try:
        candidate = os.fstat(fd)
    except OSError:
        continue
    assert (candidate.st_dev, candidate.st_ino) != (gpu.st_dev, gpu.st_ino), fd
with open(os.environ["FIXTURE_CARGO"], "a") as built:
    built.write(json.dumps(sys.argv[1:]) + "\n")
root = pathlib.Path.cwd()
target = sys.argv[sys.argv.index("--bin" if binary else "--bench") + 1]
target_dir = pathlib.Path(sys.argv[sys.argv.index("--target-dir") + 1])
exe = target_dir / ("release" if binary else "release/deps") / target
exe.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(os.environ["FIXTURE_SOURCE"], exe)
exe.chmod(0o755)
source = root / "crates/inference" / ("src/bin" if binary else "benches") / (target + ".rs")
if os.environ["FIXTURE_ARTIFACT_VALID"] != "1":
    source = root / "different.rs"
print(json.dumps(dict(reason="compiler-artifact", package_id="path+" + (root / "crates/inference").as_uri() + "#lattice-inference@0.0.1",
    target=dict(name=target,kind=["bin" if binary else "bench"],src_path=str(source)), features=json.loads(os.environ["FIXTURE_FEATURES"]), executable=str(exe))))
''')
            cargo.chmod(0o755)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("LATTICE_GPU_HANDOFF_") and key not in (
                       "LATTICE_BENCH_LOCK_STATUS", "LATTICE_BENCH_LOCK_FDS", "LATTICE_BENCH_SUPERVISOR_FD",
                       "CARGO_BUILD_TARGET", "CARGO_BUILD_TARGET_DIR", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS")}
            env.update({"PATH": f"{bindir}:{env['PATH']}", "PYTHONDONTWRITEBYTECODE": "1",
                        "CARGO_TARGET_DIR": str(root / "target"),
                        "LATTICE_MODEL_DIR": "fixture-model.txt", "FIXTURE_KIND": kind, "FIXTURE_ARGS": " ".join(bin_args), "FIXTURE_SECOND": str(int(second_acquisition)),
                        "FIXTURE_REPO": str(root), "FIXTURE_GPU": str(lock_names["GPU_LOCK"]),
                        "FIXTURE_WORK": str(temp / "worked.json"), "FIXTURE_QUIET": str(temp / "quiet"),
                        "FIXTURE_CARGO": str(temp / "cargo-ran"), "FIXTURE_SOURCE": str(executable_source),
                        "FIXTURE_ARTIFACT_VALID": str(int(artifact_valid)), "FIXTURE_FEATURES": json.dumps(sorted(features))})
            if launch is not None:
                env.update({"FIXTURE_LAUNCHES": str(temp / "launches.jsonl"), "FIXTURE_LOGITS": str(int(logits)),
                            "FIXTURE_UV_STDIN": str(temp / "uv-stdin"), "FIXTURE_UV_ARGS": str(temp / "uv-args"),
                            "FIXTURE_UV_RC": str(uv_rc), "FIXTURE_EVENTS": str(temp / "events.log"),
                            "FIXTURE_UV_RECORD": str(temp / "uv-record.json"),
                            "FIXTURE_WINDOW": str(lock_names["BENCH_WINDOW"])})
                if stdout_text is not None:
                    env["FIXTURE_STDOUT"] = stdout_text
                if stderr_text is not None:
                    env["FIXTURE_STDERR"] = stderr_text
                uv = bindir / "uv"
                uv.write_text(f"#!{sys.executable}\n" + UV_STUB)
                uv.chmod(0o755)
            env.update({key: value.replace("{TEMP}", str(temp)) for key, value in (extra_env or {}).items()})
            if kind == "bench":
                env["CRITERION_HOME"] = "relative-evidence"
                command = ["cargo", "bench", "--locked", "-p", "lattice-inference", "--bench", "topk_readback",
                           "--features", "metal-gpu", "--", "lookup", "--quick"]
            else:
                command = ["cargo", "run", "--locked", "--release", "-p", "lattice-inference", "--bin",
                           bin_name, "--features", "metal-gpu", "--", *bin_args]
            argv, cwd = (["/bin/bash", str(root / "scripts/bench-command.sh"), "--gpu-handoff", "--label", "fixture", "--",
                *command], root) if launch is None else launch(root, temp)
            result = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True, timeout=60)
            launches_file, quiet_log = temp / "launches.jsonl", temp / "quiet.log"
            return result, {
                "root": root, "temp": temp,
                "launches": [json.loads(line) for line in launches_file.read_text().splitlines()] if launches_file.exists() else [],
                "quiet_log": quiet_log.read_text().splitlines() if quiet_log.exists() else [],
                "cargo_calls": [json.loads(line) for line in (temp / "cargo-ran").read_text().splitlines()] if (temp / "cargo-ran").exists() else [],
                "uv_stdin": (temp / "uv-stdin").read_text() if (temp / "uv-stdin").exists() else None,
                "uv_args": (temp / "uv-args").read_text().strip() if (temp / "uv-args").exists() else None,
                "uv_record": json.loads((temp / "uv-record.json").read_text()) if (temp / "uv-record.json").exists() else None,
                "events": (temp / "events.log").read_text().splitlines() if (temp / "events.log").exists() else [],
                "mlx_loads": [json.loads(line) for line in (temp / "events.log.loads").read_text().splitlines()]
                if (temp / "events.log.loads").exists() else [],
                "files": {name: (root / name).read_text() if (root / name).exists() else None for name in read_files},
                "cargo": (temp / "cargo-ran").exists(), "worked": (temp / "worked.json").exists(),
                "quiet": (temp / "quiet").exists(),
                "locks": [lock_names[name].exists() for name in ("BENCH_WINDOW", "GPU_LOCK")],
                "work": json.loads((temp / "worked.json").read_text()) if (temp / "worked.json").exists() else None,
                "run_cwd": str(root if kind == "bin" else package),
                "criterion_home": None if kind == "bin" else str(package / "relative-evidence"),
            }

    def test_shipping_handoff_preserves_package_cwd_and_relative_paths(self):
        result, evidence = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(evidence["cargo"])
        self.assertTrue(evidence["quiet"])
        self.assertEqual(evidence["work"], {"cwd": evidence["run_cwd"], "criterion_home": evidence["criterion_home"]})
        self.assertIn("lookup time:", result.stdout)

    def test_historical_or_invalid_eligibility_refuses_before_locks_or_cargo(self):
        for eligibility in ("absent", "invalid"):
            with self.subTest(eligibility=eligibility):
                result, evidence = self.run_fixture(eligibility=eligibility)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("no supported GPU handoff declaration", result.stderr)
                self.assertFalse(evidence["cargo"])
                self.assertFalse(evidence["worked"])
                self.assertFalse(evidence["quiet"])
                self.assertEqual(evidence["locks"], [False, False])

    def test_wrong_cargo_artifact_refuses_before_target_or_quiet(self):
        result, evidence = self.run_fixture(artifact_valid=False)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Cargo artifact does not match", result.stderr)
        self.assertTrue(evidence["cargo"])
        self.assertFalse(evidence["worked"])
        self.assertFalse(evidence["quiet"])

    def test_declared_binary_runs_under_the_handoff_from_the_invocation_directory(self):
        result, evidence = self.run_fixture(kind="bin")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(evidence["cargo"])
        self.assertTrue(evidence["quiet"])
        self.assertEqual(evidence["work"], {"cwd": evidence["run_cwd"], "criterion_home": None})
        self.assertIn("RESULT fixture binary", result.stdout)

    def test_binary_from_a_revision_without_a_bins_declaration_refuses_before_locks_or_cargo(self):
        for eligibility in ("absent", "invalid"):
            with self.subTest(eligibility=eligibility):
                result, evidence = self.run_fixture(kind="bin", eligibility=eligibility)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("no supported GPU handoff declaration for bins", result.stderr)
                self.assertFalse(evidence["cargo"])
                self.assertFalse(evidence["worked"])
                self.assertFalse(evidence["quiet"])
                self.assertEqual(evidence["locks"], [False, False])

    def test_second_lock_acquisition_in_one_process_finds_the_control_channel_closed(self):
        # The premise of the eval_perplexity restriction: the supervisor closes the
        # control channel after the first acknowledgement.
        result, evidence = self.run_fixture(kind="bin", bin_name="bench_decode_slopefit", bin_args=(), second_acquisition=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(evidence["work"]["second"], ("ConnectionRefusedError", "FileNotFoundError"))

    def test_dual_q4_binary_invocation_refuses_before_locks_or_cargo(self):
        result, evidence = self.run_fixture(kind="bin", bin_args=("--q4-dir", "a", "--quarot-q4-dir", "b"))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("one GPU lock acquisition per process", result.stderr)
        self.assertFalse(evidence["cargo"])
        self.assertFalse(evidence["worked"])
        self.assertEqual(evidence["locks"], [False, False])

    def test_wrong_binary_artifact_refuses_before_target_or_quiet(self):
        result, evidence = self.run_fixture(kind="bin", artifact_valid=False)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Cargo artifact does not match", result.stderr)
        self.assertTrue(evidence["cargo"])
        self.assertFalse(evidence["worked"])
        self.assertFalse(evidence["quiet"])

    SLOPEFIT_STDOUT = "\n".join([
        "SLOPEFIT_META kv_cache_len=1024 warmup=8 measure=32 repeats=1",
        "SLOPEFIT_META ctx=64 actual_prompt_tokens=70",
        "SLOPEFIT ctx=64 tokens=32 warmup_ms=0.0 measure_ms=96.000 rep=0",
    ])
    SLOPEFIT_STDERR = "\n".join([
        "[slopefit] loading /models/qwen (Q4)",
        "[slopefit] grid=[64] warmup=8 measure=32 repeats=1",
        "[slopefit] kv_cache_len=1024 (deepest_prompt=70 for max_ctx=64 + decode_horizon 32)",
        "[slopefit] ctx=64 actual_prompt_tokens=70",
    ])
    SLOPEFIT_FILES = ("scripts/bench_decode_slopefit.py", "scripts/lib/ensure-noindex-marker.sh")

    def run_script(self, relative, args=(), *, files=(), cwd=None, extra_files=None, **options):
        """Run a shipping script, copied into the fixture repository, from `cwd` (under the repo root)."""
        committed = {name: REPO / name for name in (relative, *files)}
        committed.update(extra_files or {})

        def launch(root, temp):
            workdir = root / cwd if cwd else root
            workdir.mkdir(parents=True, exist_ok=True)
            return ["/bin/bash", str(root / relative), *args], workdir

        return self.run_fixture(kind="bin", launch=launch, commit_files=committed, **options)

    def run_slopefit(self, args=(), **options):
        options.setdefault("stdout_text", self.SLOPEFIT_STDOUT)
        options.setdefault("stderr_text", self.SLOPEFIT_STDERR)
        return self.run_script(
            "scripts/bench_decode_slopefit.sh", args, files=self.SLOPEFIT_FILES,
            bin_name="bench_decode_slopefit", bin_args=(), features=("default", "f16", "metal-gpu", "std"), **options)

    def test_slopefit_script_runs_its_binary_through_the_handoff_and_feeds_the_parser(self):
        result, evidence = self.run_slopefit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        root = evidence["root"]
        # One handoff launch from the repository root, building the release binary with its features.
        self.assertEqual([(item["argv"], item["cwd"]) for item in evidence["launches"]], [([], str(root))])
        self.assertEqual(len(evidence["cargo_calls"]), 1)
        call = evidence["cargo_calls"][0]
        self.assertEqual(call[:1], ["build"])
        self.assertIn("--release", call)
        self.assertEqual(call[call.index("--bin") + 1], "bench_decode_slopefit")
        self.assertEqual(call[call.index("--features") + 1], "f16,metal-gpu")
        # The script was durable before, so the before/after ambient-idle checks survive,
        # and the measured-phase certification runs inside the handoff.
        self.assertEqual(evidence["quiet_log"], [
            "--label decode-slopefit: before",
            "--label command:bench_decode_slopefit: measured guard",
            "--label decode-slopefit: after",
        ])
        self.assertEqual(evidence["uv_args"].split()[:3], ["run", "--project", str(root)])
        # The post-processor reads the binary's stdout and, merged in by the handoff, its stderr and
        # the supervisor's own notices; the parser takes the same records out of either stream.
        stream = evidence["uv_stdin"]
        for line in (*self.SLOPEFIT_STDOUT.splitlines(), *self.SLOPEFIT_STDERR.splitlines(),
                     "fixture inside-guard CPU idle sample"):
            self.assertIn(line + "\n", stream)
        parser = _load_script_with_numpy_stub("slopefit_for_stream_test", "bench_decode_slopefit.py")
        clean = parser.parse_stream(self.SLOPEFIT_STDOUT.splitlines())
        self.assertEqual(parser.parse_stream(stream.splitlines()), clean)
        self.assertEqual(clean[3], {"kv_cache_len": 1024, "warmup": 8, "measure": 32, "repeats": 1})
        self.assertEqual(dict(clean[0]), {64: [3.0]})

    def test_slopefit_script_resolves_caller_relative_paths_before_the_handoff(self):
        # The handoff runs from the repository root; a path the caller wrote against its own
        # directory must still mean the same file.
        result, evidence = self.run_slopefit(
            ["--out", "out/slope.json"], cwd="sub", extra_env={"LATTICE_MODEL_DIR": "../fixture-model.txt"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        root = evidence["root"]
        self.assertEqual(evidence["launches"][0]["cwd"], str(root))
        self.assertEqual(evidence["launches"][0]["model"], f"{root}/sub/../fixture-model.txt")
        self.assertTrue(evidence["uv_args"].endswith(f"--out {root}/sub/out/slope.json"), evidence["uv_args"])

    def test_slopefit_script_refuses_before_locks_or_build_when_the_handoff_refuses(self):
        result, evidence = self.run_slopefit(eligibility="absent", uv_rc=1)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("no supported GPU handoff declaration for bins", result.stderr)
        self.assertEqual((evidence["launches"], evidence["cargo_calls"], evidence["locks"]), ([], [], [False, False]))

    def test_quality_script_runs_each_lattice_tier_as_its_own_handoff_launch(self):
        with tempfile.TemporaryDirectory(prefix="quality-models-") as models:
            dirs = {name: Path(models).resolve() / name for name in ("q4", "quarot", "tokenizer")}
            for directory in dirs.values():
                directory.mkdir()
            result, evidence = self.run_script(
                "scripts/bench_quality.sh", bin_name="eval_perplexity", bin_args=(), cwd="sub",
                # The real ignore file: the staged result is untracked scratch inside the worktree, and
                # the handoff admits only a commit-clean one.
                gitignore=(REPO / ".gitignore").read_text(),
                extra_files={"docs/bench_results/wiki.test.raw": "corpus\n", "docs/bench_results/perplexity.tsv": "canonical\n"},
                extra_env={"Q4_DIR": str(dirs["q4"]), "QUAROT_DIR": str(dirs["quarot"]), "TOK_DIR": str(dirs["tokenizer"]),
                           "SKIP_MLX": "1", "BENCH_MACHINE": "fixture"},
                stdout_text="PPL:                16.589166", read_files=("docs/bench_results/perplexity.tsv",))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            root = evidence["root"]
            self.assertEqual([item["cwd"] for item in evidence["launches"]], [str(root)] * 2)
            first, second = (item["argv"] for item in evidence["launches"])
            # Each tier takes the GPU lock once: exactly one Metal mode per process.
            modes = ("--metal-model-dir", "--q4-dir", "--quarot-q4-dir")
            self.assertEqual([[arg for arg in argv if arg in modes] for argv in (first, second)],
                             [["--q4-dir"], ["--quarot-q4-dir"]])
            self.assertEqual(first[first.index("--q4-dir") + 1], str(dirs["q4"]))
            self.assertEqual(second[second.index("--quarot-q4-dir") + 1], str(dirs["quarot"]))
            self.assertEqual(len(evidence["cargo_calls"]), 2)
            self.assertEqual(sorted(evidence["quiet_log"]), sorted(
                [f"--label {label}" for label in ["quality-perplexity: before", "quality-perplexity: after"] * 2
                 + ["command:eval_perplexity: measured guard"] * 2]))
            published = evidence["files"]["docs/bench_results/perplexity.tsv"]
            self.assertIn("lattice\tq4\t16.589166\t2048\n", published)
            self.assertIn("lattice\tq4-quarot\t16.589166\t2048\n", published)
            # SKIP_MLX=1 skips the cross-check: no MLX launch of any kind.
            self.assertNotIn("mlx-start", evidence["events"])

    def test_quality_script_resolves_caller_relative_model_directories(self):
        # Written against the caller's directory, one level below the repository root, and
        # naming directories next to it; the handoff runs from the root.
        def launch(root, temp):
            for name in ("q4", "quarot", "tokenizer"):
                (temp / "models" / name).mkdir(parents=True)
            workdir = root / "sub"
            workdir.mkdir()
            return ["/bin/bash", str(root / "scripts/bench_quality.sh")], workdir

        result, evidence = self.run_fixture(
            kind="bin", launch=launch, bin_name="eval_perplexity", bin_args=(),
            gitignore=(REPO / ".gitignore").read_text(),
            commit_files={"scripts/bench_quality.sh": REPO / "scripts/bench_quality.sh",
                          "docs/bench_results/wiki.test.raw": "corpus\n", "docs/bench_results/perplexity.tsv": "canonical\n"},
            extra_env={"Q4_DIR": "../../models/q4", "QUAROT_DIR": "../../models/quarot", "TOK_DIR": "../../models/tokenizer",
                       "SKIP_MLX": "1"},
            stdout_text="PPL:                16.589166")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        root = evidence["root"]
        first, second = (item["argv"] for item in evidence["launches"])
        self.assertEqual(first[first.index("--q4-dir") + 1], f"{root}/sub/../../models/q4")
        self.assertEqual(second[second.index("--quarot-q4-dir") + 1], f"{root}/sub/../../models/quarot")
        self.assertEqual(second[second.index("--tokenizer-dir") + 1], f"{root}/sub/../../models/tokenizer")

    def test_logit_dump_helper_runs_the_binary_through_the_handoff_with_absolute_paths(self):
        def launch(root, temp):
            stubs = temp / "pystub"
            stubs.mkdir()
            (stubs / "numpy.py").write_text(NUMPY_STUB)
            workdir = root / "sub"
            (workdir / "out").mkdir(parents=True)
            code = ("import sys; sys.path[:0] = [%r, %r]; import compare_logits as c; from pathlib import Path; "
                    "array = c.run_lattice_logit_dump([5, 6, 7], Path('../fixture-model.txt'), 'out/logits.bin'); "
                    "print('SHAPE', array.shape)" % (str(root / "scripts"), str(stubs)))
            return [sys.executable, "-c", code], workdir

        result, evidence = self.run_fixture(
            kind="bin", launch=launch, commit_files={"scripts/compare_logits.py": REPO / "scripts/compare_logits.py"},
            bin_name="bench_logit_dump", bin_args=(), features=("default", "f16", "metal-gpu", "std"), logits=True,
            stdout_text="VOCAB=4\nNPOS=2\nOUT=elsewhere", stderr_text="[bench_logit_dump] loading model")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SHAPE (2, 4)", result.stdout)
        root = evidence["root"]
        launch_record = evidence["launches"][0]
        self.assertEqual((launch_record["argv"], launch_record["cwd"]), ([], str(root)))
        self.assertEqual(launch_record["model"], f"{root}/fixture-model.txt")
        self.assertEqual(launch_record["out"], f"{root}/sub/out/logits.bin")
        call = evidence["cargo_calls"][0]
        self.assertEqual(call[call.index("--features") + 1], "metal-gpu,f16")
        self.assertIn("--release", call)
        # Lock-only before, as it was: no before/after probes, the measured-phase one only.
        self.assertEqual(evidence["quiet_log"], ["--label command:bench_logit_dump: measured guard"])

    MLX_ROWS = "mlx\tq8\t15.8218\t2041\nmlx\tq4\t18.1839\t2041\n"

    def run_quality_with_mlx(self, *, uv_rc=0, uv_out=MLX_ROWS, env=None):
        """bench_quality.sh with the MLX cross-check enabled, the cross-check stubbed as `uv`."""
        with tempfile.TemporaryDirectory(prefix="quality-models-") as models:
            dirs = {name: Path(models).resolve() / name for name in ("q4", "quarot", "tokenizer")}
            for directory in dirs.values():
                directory.mkdir()
            result, evidence = self.run_script(
                "scripts/bench_quality.sh", bin_name="eval_perplexity", bin_args=(),
                gitignore=(REPO / ".gitignore").read_text(),
                extra_files={"docs/bench_results/wiki.test.raw": "corpus\n", "docs/bench_results/perplexity.tsv": "canonical\n"},
                extra_env={"Q4_DIR": str(dirs["q4"]), "QUAROT_DIR": str(dirs["quarot"]), "TOK_DIR": str(dirs["tokenizer"]),
                           "BENCH_MACHINE": "fixture", "MLX_LOG": str(Path(models).resolve() / "mlx.log"),
                           "FIXTURE_UV_OUT": uv_out, "FIXTURE_UV_ERR": "  q8: PPL = 15.8218\n", **(env or {})},
                stdout_text="PPL:                16.589166", uv_rc=uv_rc, read_files=("docs/bench_results/perplexity.tsv",))
            evidence["dirs"] = dirs
            evidence["mlx_log"] = (Path(models).resolve() / "mlx.log").read_text()
            return result, evidence

    def test_quality_script_runs_the_mlx_cross_check_as_a_durable_plain_launch_under_both_locks(self):
        result, evidence = self.run_quality_with_mlx()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # The cross-check runs after the two lattice tiers and never overlaps either of them.
        self.assertEqual(evidence["events"], ["lattice-start", "lattice-end"] * 2 + ["mlx-start", "mlx-end"])
        # Both machine locks were held by the supervisor around the MLX process, which was not
        # handed any lock or handoff capability.
        self.assertEqual(evidence["uv_record"], {
            "gpu_lock_held": True, "window_held": True, "supervisor_marker": True,
            "handoff_env": [], "lock_fds": False})
        # Plain route: the supervisor ran no admission, so no handoff build happened for it,
        # and the durable before/after ambient-idle checks surround it like the lattice tiers.
        self.assertEqual(len(evidence["cargo_calls"]), 2)
        self.assertEqual(sorted(evidence["quiet_log"]), sorted(
            [f"--label {label}" for label in ["quality-perplexity: before", "quality-perplexity: after"] * 2
             + ["command:eval_perplexity: measured guard"] * 2 + ["quality-mlx: before", "quality-mlx: after"]]))
        # The program still arrives on the cross-check's stdin through the supervisor, with the same arguments.
        dirs = evidence["dirs"]
        root = evidence["root"]
        self.assertEqual(
            evidence["uv_args"],
            f"run --quiet --with mlx-lm python3 - {dirs['tokenizer']} {root}/docs/bench_results/wiki.test.raw 512 256 2048")
        for text in ("from mlx_lm import load", "ppl_at_bits(8, \"q8\")", "ppl_at_bits(4, \"q4\")"):
            self.assertIn(text, evidence["uv_stdin"])
        # Output contract: only the exact MLX rows reach the data file (the supervisor's probe
        # lines share that stdout), and stderr goes to the log.
        published = evidence["files"]["docs/bench_results/perplexity.tsv"]
        for row in ("lattice\tq4\t16.589166\t2048\n", "lattice\tq4-quarot\t16.589166\t2048\n",
                    "mlx\tq8\t15.8218\t2041\n", "mlx\tq4\t18.1839\t2041\n"):
            self.assertIn(row, published)
        self.assertNotIn("fixture inside-guard", published)
        self.assertIn("q8: PPL = 15.8218", evidence["mlx_log"])

    def test_quality_script_treats_a_failing_mlx_cross_check_as_fatal_and_publishes_nothing(self):
        # It was fatal before the handoff migration and stays so: exit 1 after the lattice tiers,
        # with the canonical file untouched, whether or not the failed process had printed rows.
        for uv_out in ("", self.MLX_ROWS):
            with self.subTest(rows_printed=bool(uv_out)):
                result, evidence = self.run_quality_with_mlx(uv_rc=3, uv_out=uv_out)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("MLX cross-check failed (exit 3", result.stderr)
                self.assertEqual(evidence["files"]["docs/bench_results/perplexity.tsv"], "canonical\n")
                self.assertEqual(evidence["events"], ["lattice-start", "lattice-end"] * 2 + ["mlx-start", "mlx-end"])

    def test_quality_script_refuses_an_mlx_cross_check_that_produced_no_rows(self):
        result, evidence = self.run_quality_with_mlx(uv_out="")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("did not produce exactly one q8 row", result.stderr)
        self.assertEqual(evidence["files"]["docs/bench_results/perplexity.tsv"], "canonical\n")

    def test_quality_script_mlx_cross_check_is_gated_by_the_durable_ambient_idle_check(self):
        # A noisy machine before the cross-check: the supervisor refuses (exit 2) and the MLX
        # process never starts; the script fails and publishes nothing.
        result, evidence = self.run_quality_with_mlx(env={"FIXTURE_QUIET_FAIL": "quality-mlx: before"})
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("MLX cross-check failed (exit 2", result.stderr)
        self.assertNotIn("mlx-start", evidence["events"])
        self.assertEqual(evidence["files"]["docs/bench_results/perplexity.tsv"], "canonical\n")

    MLX_ENV = {"LATTICE_LOGIT_TMP": ".cache/lattice_logits.bin"}

    def run_logit_script(self, *, mlx_mode="", n_tokens=2):
        """compare_logits.py main() end to end: real bench-command.sh and handoff, stub numpy/MLX/cargo."""
        def launch(root, temp):
            stubs = temp / "pystub"
            (stubs / "mlx").mkdir(parents=True)
            (stubs / "numpy.py").write_text(NUMPY_STUB)
            (stubs / "mlx_lm.py").write_text(MLX_LM_STUB)
            (stubs / "mlx/__init__.py").write_text("")
            (stubs / "mlx/core.py").write_text(MLX_CORE_STUB)
            (stubs / "transformers.py").write_text(TRANSFORMERS_STUB)
            code = "\n".join([
                "import os, sys",
                "print('PARENT_PID', os.getpid())",
                "sys.argv = ['compare_logits.py', '--n-tokens=%d']" % n_tokens,
                "sys.path[:0] = [%r, %r]" % (str(root / "scripts"), str(stubs)),
                "import compare_logits as c",
                "collect = c.collect_mlx_via_child",
                "def wrapped():",
                "    ids, logits = collect()",
                "    print('CHILD', len(ids), logits.shape)",
                "    return ids, logits",
                "c.collect_mlx_via_child = wrapped",
                "c.analyze = lambda lat, mlx, ids: print('ANALYZE', lat.shape, mlx.shape, ids, mlx.flat[:4], mlx.flat[4:8])",
                "c.main()",
            ])
            return [sys.executable, "-c", code], root

        return self.run_fixture(
            kind="bin", launch=launch, bin_name="bench_logit_dump", bin_args=(), features=("default", "f16", "metal-gpu", "std"),
            logits=True, stdout_text="VOCAB=4\nNPOS=2\nOUT=elsewhere",
            commit_files={"scripts/compare_logits.py": REPO / "scripts/compare_logits.py",
                          "docs/bench_results/wiki.test.raw": "corpus\n"},
            extra_env={**self.MLX_ENV, "FIXTURE_MLX_MODE": mlx_mode, "PYTHONPATH": "{TEMP}/pystub"})

    def test_logit_script_runs_all_mlx_work_in_one_child_under_both_locks_before_the_lattice_run(self):
        result, evidence = self.run_logit_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # Tokenizer load, then prefill load (the original two loads), one process exit, and only then
        # the Lattice binary. Nothing of the MLX work happened in the parent or overlapped the Lattice run.
        self.assertEqual(evidence["events"], ["mlx-load", "mlx-load", "mlx-exit", "lattice-start", "lattice-end"])
        loads = evidence["mlx_loads"]
        self.assertEqual(len(loads), 2)
        self.assertEqual({item["pid"] for item in loads}.__len__(), 1)
        # The MLX work is a different process from the one that runs main().
        self.assertNotIn(f"PARENT_PID {loads[0]['pid']}\n", result.stdout)
        self.assertRegex(result.stdout, r"PARENT_PID \d+\n")
        for record in loads:
            self.assertEqual(
                {key: record[key] for key in ("gpu_lock_held", "window_held", "supervisor_marker", "handoff_env", "lock_fds")},
                {"gpu_lock_held": True, "window_held": True, "supervisor_marker": True, "handoff_env": [], "lock_fds": False})
        # Lock-only, as before the migration: no before/after probes for the MLX child.
        self.assertEqual(evidence["quiet_log"], ["--label command:bench_logit_dump: measured guard"])
        # The child's token ids and logits reach the analysis (which runs after both passes, outside the locks).
        self.assertIn("CHILD 2 (2, 4)\n", result.stdout)
        self.assertIn("ANALYZE (2, 4) (2, 4) [101, 102] [0.0, 1.0, 2.0, 3.0] [10.0, 11.0, 12.0, 13.0]", result.stdout)
        # The child's progress lines are not captured.
        self.assertIn("[step 1] tokenizing corpus with MLX tokenizer", result.stdout)
        self.assertEqual(len(evidence["launches"]), 1)

    def test_logit_script_keeps_the_transformers_tokenizer_fallback_inside_the_child(self):
        result, evidence = self.run_logit_script(mlx_mode="first-load-fails")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("mlx_lm tokenizer failed", result.stdout)
        self.assertIn("ANALYZE (2, 4) (2, 4) [201, 202]", result.stdout)
        self.assertEqual(evidence["events"], ["mlx-load", "mlx-load", "mlx-exit", "lattice-start", "lattice-end"])

    def test_logit_script_stops_before_the_lattice_run_when_the_mlx_child_fails(self):
        for mode in ("all-loads-fail", "prefill-fails"):
            with self.subTest(mode=mode):
                result, evidence = self.run_logit_script(mlx_mode=mode)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("MLX child exited 1", result.stderr)
                self.assertNotIn("lattice-start", evidence["events"])
                self.assertEqual((evidence["launches"], evidence["cargo_calls"]), ([], []))
                self.assertNotIn("ANALYZE", result.stdout)


NUMPY_STUB = """
import struct

float32 = "<f4"


class _Array:
    def __init__(self, flat, shape):
        self.flat, self.shape = flat, shape

    def reshape(self, *shape):
        assert len(self.flat) == shape[0] * shape[1], (len(self.flat), shape)
        return _Array(self.flat, shape)

    def copy(self):
        return self

    def astype(self, dtype, **options):
        return self

    def tobytes(self):
        return struct.pack("<%df" % len(self.flat), *self.flat)

    def __getitem__(self, rows):
        width = self.shape[1]
        return _Array(self.flat[rows.start or 0:][: (rows.stop - (rows.start or 0)) * width], (rows.stop - (rows.start or 0), width))


def asarray(rows, dtype=None):
    return _Array([float(value) for row in rows for value in row], (len(rows), len(rows[0])))


def frombuffer(raw, dtype=None):
    count = len(raw) // 4
    return _Array(list(struct.unpack("<%df" % count, raw)), (count,))
"""


MLX_LM_STUB = r"""
import atexit, fcntl, json, os, pathlib

_events = os.environ["FIXTURE_EVENTS"]
_mode = os.environ.get("FIXTURE_MLX_MODE", "")
_marker = pathlib.Path(_events + ".first-load-done")
atexit.register(lambda: open(_events, "a").write("mlx-exit\n"))


def _held(path):
    with open(path, "r+") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [101 + i for i in range(64)]


class _Logits:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        return self

    def tolist(self):
        return [[float(10 * i + j) for j in range(4)] for i in range(self.rows)]


class _Model:
    def eval(self):
        pass

    def __call__(self, ids):
        if _mode == "prefill-fails":
            raise RuntimeError("injected prefill failure")
        return _Logits(len(ids.rows[0]))


def load(path):
    open(_events, "a").write("mlx-load\n")
    with open(_events + ".loads", "a") as sink:
        sink.write(json.dumps(dict(
            pid=os.getpid(), gpu_lock_held=_held(os.environ["FIXTURE_GPU"]), window_held=_held(os.environ["FIXTURE_WINDOW"]),
            supervisor_marker="LATTICE_GPU_LOCK_SUPERVISOR_PID" in os.environ,
            handoff_env=sorted(name for name in os.environ if name.startswith("LATTICE_GPU_HANDOFF_")),
            lock_fds="LATTICE_BENCH_LOCK_FDS" in os.environ)) + "\n")
    if _mode == "all-loads-fail" or (_mode == "first-load-fails" and not _marker.exists()):
        _marker.write_text("1")
        raise RuntimeError("injected mlx load failure")
    return _Model(), _Tok()
"""

MLX_CORE_STUB = """
class array:
    def __init__(self, rows):
        self.rows = rows


def eval(value):
    pass
"""

TRANSFORMERS_STUB = """
import os


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [201 + i for i in range(64)]


class AutoTokenizer:
    @staticmethod
    def from_pretrained(path, **options):
        assert options == {"trust_remote_code": False, "local_files_only": True}, options
        if os.environ.get("FIXTURE_MLX_MODE") == "all-loads-fail":
            raise RuntimeError("injected transformers failure")
        return _Tok()
"""


def _load_script_with_numpy_stub(name, filename):
    """Import a script whose numpy-dependent bodies the test does not run."""
    import types
    from unittest import mock

    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"numpy": types.ModuleType("numpy")}):
        spec.loader.exec_module(module)
    return module


class GpuHandoffScriptStreams(unittest.TestCase):
    """The handoff merges the binary's standard error into the stream its callers parse."""

    def test_slopefit_parser_counts_only_tagged_records(self):
        parser = _load_script_with_numpy_stub("slopefit_stream", "bench_decode_slopefit.py")
        records = [
            "SLOPEFIT_META kv_cache_len=1024 warmup=8 measure=32 repeats=2",
            "SLOPEFIT_META ctx=64 actual_prompt_tokens=70",
            "SLOPEFIT ctx=64 tokens=32 warmup_ms=0.0 measure_ms=96.000 rep=0",
            "SLOPEFIT ctx=64 tokens=32 warmup_ms=0.0 measure_ms=64.000 rep=1",
        ]
        noise = [
            "fixture inside-guard CPU idle sample",
            "[slopefit] loading /models/qwen (Q4)",
            "[slopefit] grid=[64] warmup=8 measure=32 repeats=2",
            "[slopefit] kv_cache_len=1024 (deepest_prompt=70 for max_ctx=64 + decode_horizon 32)",
            "[slopefit] ctx=64 actual_prompt_tokens=70",
            # Progress text that quotes record fields without being a record.
            "[slopefit] ctx=64 actual_prompt_tokens=9999",
            "[slopefit] ctx=64 tokens=1 warmup_ms=0.0 measure_ms=1.000 rep=0",
            "bench-supervision: measured-phase receipt: /repo/.cache/receipt.jsonl",
        ]
        clean = parser.parse_stream(records)
        merged = [noise[0], noise[1], records[0], noise[2], noise[3], noise[4], records[1], noise[5],
                  records[2], noise[6], records[3], noise[7]]
        self.assertEqual(parser.parse_stream(merged), clean)
        self.assertEqual(clean[2], {64: 70})
        self.assertEqual(dict(clean[0]), {64: [3.0, 2.0]})
        # A stream that lost its records yields no run metadata, which the post-processor refuses.
        self.assertIsNone(parser.parse_stream(noise)[3])

    def test_logit_dump_header_counts_only_exact_records(self):
        module = _load_script_with_numpy_stub("compare_logits_stream", "compare_logits.py")
        output = "\n".join([
            "fixture inside-guard CPU idle sample",
            "[bench_logit_dump] loading /models/qwen",
            "VOCAB=4",
            "[bench_logit_dump] writing 2x4 f32",
            "NPOS=2",
            "OUT=/tmp/logits.bin",
            "[bench_logit_dump] VOCAB=999",
            "[bench_logit_dump] NPOS=999",
        ])
        self.assertEqual(module.parse_dump_header(output), (4, 2))
        self.assertEqual(module.parse_dump_header("[bench_logit_dump] VOCAB=4\n[x] NPOS=2"), (None, None))

    def test_logit_dump_helper_refuses_the_retired_prebuilt_binary_override(self):
        module = _load_script_with_numpy_stub("compare_logits_bindir", "compare_logits.py")
        with self.assertRaisesRegex(SystemExit, "LATTICE_BIN_DIR is not supported"):
            module.reject_prebuilt_binary_dir({"LATTICE_BIN_DIR": "/prebuilt"})
        module.reject_prebuilt_binary_dir({})
        module.reject_prebuilt_binary_dir({"LATTICE_BIN_DIR": ""})

    def test_logit_script_rejects_the_override_before_any_work(self):
        """The refusal has to precede the MLX pass, not surface after it."""
        import ast

        source = (REPO / "scripts/compare_logits.py").read_text()
        main = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        first = main.body[0]
        self.assertIsInstance(first, ast.Expr)
        self.assertEqual(ast.unparse(first.value), "reject_prebuilt_binary_dir(os.environ)")


class _FailOnEmptyTestProgram(unittest.TestProgram):
    def runTests(self) -> None:
        if self.test.countTestCases() == 0:
            raise SystemExit("no tests collected")
        super().runTests()


if __name__ == "__main__":
    _FailOnEmptyTestProgram()
