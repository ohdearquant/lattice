#!/usr/bin/env python3
"""Revision-bound admission and descriptor-free builds for selected Metal benches."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import tomllib
from typing import Mapping

PROTOCOL = 1
PACKAGE = "lattice-inference"
TARGETS = {
    "metal_decode_bench": {"metal-gpu", "f16"},
    "cross_turn_prefix_cache_bench": {"metal-gpu", "f16"},
    "mtp_decode": {"metal-gpu", "f16"},
    "decode_attn_bench": {"metal-gpu", "f16"},
    "lm_head_bench": {"metal-gpu", "f16", "bench-internals"},
    "topk_readback": {"metal-gpu"},
}
BROKER_ENV = "LATTICE_GPU_HANDOFF_BROKER"
BROKER_TOKEN_ENV = "LATTICE_GPU_HANDOFF_BROKER_TOKEN"
MAX_FRAME = 2 * 1024 * 1024


class AdmissionError(RuntimeError):
    """A selected invocation cannot be safely admitted."""


def _git(repo: Path, *args: str) -> str:
    env = _cargo_environment()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(name, None)
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, env=env,
    )
    if result.returncode:
        raise AdmissionError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _revision(repo: Path, ref: str) -> str:
    sha = _git(repo, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise AdmissionError(f"{ref!r} did not resolve to one immutable commit")
    return sha


def _manifest(repo: Path, sha: str, path: str) -> dict:
    try:
        return tomllib.loads(_git(repo, "show", f"{sha}:{path}"))
    except (tomllib.TOMLDecodeError, UnicodeError) as exc:
        raise AdmissionError(f"unreadable {path} at {sha}: {exc}") from exc


def _clean_revision(cwd: Path, sha: str) -> None:
    if _revision(cwd, "HEAD") != sha:
        raise AdmissionError(f"measured worktree {cwd} is not admitted revision {sha}")
    if _git(cwd, "status", "--porcelain=v1", "--untracked-files=normal"):
        raise AdmissionError(f"measured worktree {cwd} is not commit-clean")


def feature_closure(manifest: dict, raw: str) -> list[str]:
    """Resolve package-local default and requested features from one revision."""
    declarations = manifest.get("features", {})
    if not isinstance(declarations, dict) or not all(
        isinstance(name, str) and isinstance(edges, list)
        and all(isinstance(edge, str) for edge in edges)
        for name, edges in declarations.items()
    ):
        raise AdmissionError("invalid feature declarations")
    suppressed = {
        edge[4:] for edges in declarations.values() for edge in edges
        if edge.startswith("dep:")
    }
    dependencies = manifest.get("dependencies", {})
    if not isinstance(dependencies, dict):
        raise AdmissionError("invalid dependency feature declarations")
    implicit = {
        name for name, dep in dependencies.items()
        if isinstance(dep, dict) and dep.get("optional") is True and name not in suppressed
    }
    requested = set(filter(None, re.split(r"[\s,]+", raw)))
    if "default" in declarations:
        requested.add("default")
    result: set[str] = set()
    pending = list(requested)
    while pending:
        name = pending.pop()
        if name in result:
            continue
        if name not in declarations and name not in implicit:
            raise AdmissionError(f"unknown or unsupported package feature {name!r}")
        result.add(name)
        for edge in declarations.get(name, []):
            if edge.startswith("dep:"):
                continue
            if "/" in edge:
                dependency = edge.split("/", 1)[0]
                if not dependency.endswith("?") and dependency in implicit:
                    pending.append(dependency)
                continue
            pending.append(edge)
    return sorted(result)


def _host(env: Mapping[str, str]) -> str:
    if sys.platform != "darwin":
        raise AdmissionError("GPU handoff requires a macOS measurement host")
    # A custom build target can select a different cfg from the admitted host.
    for name in ("CARGO_BUILD_TARGET", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"):
        if env.get(name):
            raise AdmissionError(f"explicit GPU handoff does not support {name}")
    return "darwin"


def _entry(
    repo: Path, sha: str, target: str, features: str, *, entry_id: str,
    cwd: Path, criterion_home: Path, criterion_args: list[str], platform: str,
) -> dict:
    if target not in TARGETS:
        raise AdmissionError(f"{entry_id}: unsupported self-locking target {target!r}")
    manifest = _manifest(repo, sha, "crates/inference/Cargo.toml")
    package = manifest.get("package", {})
    if not isinstance(package, dict) or not isinstance(package.get("metadata", {}), dict):
        raise AdmissionError(f"{entry_id} {sha}: invalid package metadata")
    policy = package.get("metadata", {}).get("gpu-bench-handoff", {})
    declared = policy.get("targets", []) if isinstance(policy, dict) else []
    if (not isinstance(policy, dict) or type(policy.get("version")) is not int
            or policy.get("version") != PROTOCOL or not isinstance(declared, list)
            or not all(isinstance(name, str) for name in declared)
            or sorted(declared) != sorted(TARGETS)):
        raise AdmissionError(f"{entry_id} {sha}: revision has no supported GPU handoff declaration")
    declarations = manifest.get("bench", [])
    if not isinstance(declarations, list) or not all(isinstance(item, dict) for item in declarations):
        raise AdmissionError(f"{entry_id} {sha}: invalid benchmark declarations")
    benches = [item for item in declarations if item.get("name") == target]
    if len(benches) != 1 or benches[0].get("harness") is not False:
        raise AdmissionError(f"{entry_id} {sha}: {target} is not one declared Criterion target")
    source_path = benches[0].get("path", f"benches/{target}.rs")
    if source_path != f"benches/{target}.rs":
        raise AdmissionError(f"{entry_id} {sha}: unsupported benchmark source {source_path!r}")
    source_path = f"crates/inference/{source_path}"
    _git(repo, "show", f"{sha}:{source_path}")
    feature_set = feature_closure(manifest, features)
    cargo_required = benches[0].get("required-features", [])
    if not isinstance(cargo_required, list) or not all(isinstance(name, str) for name in cargo_required):
        raise AdmissionError(f"{entry_id} {sha}: invalid target required-features")
    required = TARGETS[target] | set(cargo_required)
    missing = required - set(feature_set)
    if missing:
        raise AdmissionError(f"{entry_id} {sha}: {target} lacks features {sorted(missing)}")
    root_manifest = _manifest(repo, sha, "Cargo.toml")
    version = package.get("version")
    if isinstance(version, dict) and version.get("workspace") is True:
        version = root_manifest.get("workspace", {}).get("package", {}).get("version")
    if not isinstance(version, str) or package.get("name") != PACKAGE:
        raise AdmissionError(f"{entry_id} {sha}: invalid package identity")
    # The helper owns this output path, independent of caller Cargo settings.
    target_dir = repo / ".cache" / "bench-gpu-handoff-build" / sha / target
    return {
        "id": entry_id, "revision": sha, "package": PACKAGE, "target": target,
        "features": features, "feature_set": feature_set, "source_path": source_path,
        "platform": platform, "cwd": str(cwd), "run_cwd": str(cwd / "crates/inference"),
        "criterion_home": str(criterion_home),
        "argv": ["--bench", *criterion_args], "target_dir": str(target_dir),
        "package_version": version,
    }


def _criterion_args(args: list[str]) -> list[str]:
    if any(not isinstance(arg, str) or "\x00" in arg for arg in args):
        raise AdmissionError("invalid Criterion argument")
    return args


def plan_compare(repo: Path, args: list[str], env: Mapping[str, str]) -> dict:
    """Freeze both revisions and all four inference invocations before locking."""
    repo = Path(repo).resolve()
    platform = _host(env)
    flags: list[str] = []
    rest = list(args)
    quick = True
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if flag == "--":
            flags.append(flag)
            break
        if flag not in ("--full", "--fail-on-regression"):
            raise AdmissionError(f"unsupported comparison flag {flag!r}")
        flags.append(flag)
        if flag == "--full":
            quick = False
    if len(rest) > 2 or ("--" not in flags and any(ref.startswith("-") for ref in rest)):
        raise AdmissionError("expected at most BASE and HEAD after comparison flags")
    base_ref = rest[0] if rest else "origin/main"
    head_ref = rest[1] if len(rest) > 1 else "HEAD"
    base, head = _revision(repo, base_ref), _revision(repo, head_ref)
    if head_ref == "HEAD":
        _clean_revision(repo, head)
    target = env.get("BENCHES_INFERENCE") or "elementwise_cpu_bench"
    features = env.get("CARGO_FEATURES_INFERENCE") or ""
    # Only inference is self-locking here; ordinary embed remains supervised.
    embed_target = env.get("BENCHES_EMBED") or "simd"
    if embed_target not in ("simd", "embeddings", "simd_bench", "simd_opt_bench", "simsimd_comparison"):
        raise AdmissionError(f"unclassified embed target {embed_target!r} in GPU handoff comparison")
    root = repo / ".cache" / "bench-compare-criterion"
    group = env.get("BENCH_GROUPS_INFERENCE") or ""
    prefix = [group] if group else []
    suffix = ["--noplot", *(["--quick"] if quick else [])]
    locations = [
        ("base1", base, "bench-compare-base", root / "base" / "inference" / target / "criterion", "--save-baseline", "compare-base"),
        ("head1", head, "bench-compare-head", root / "head" / "inference" / target / "criterion", "--baseline", "compare-base"),
        ("head2", head, "bench-compare-head", root / "order-control" / "head" / "inference" / target / "criterion", "--save-baseline", "compare-head"),
        ("base2", base, "bench-compare-base", root / "order-control" / "base" / "inference" / target / "criterion", "--baseline", "compare-head"),
    ]
    entries = [
        _entry(repo, sha, target, features, entry_id=entry_id, cwd=repo / ".cache" / directory,
               criterion_home=home, criterion_args=[*prefix, operation, baseline, *suffix],
               platform=platform)
        for entry_id, sha, directory, home, operation, baseline in locations
    ]
    return {
        "version": PROTOCOL, "mode": "compare", "repo": str(repo), "entries": entries,
        "environment": {"LATTICE_GPU_HANDOFF_BASE_SHA": base, "LATTICE_GPU_HANDOFF_HEAD_SHA": head},
        "command": [str(repo / "scripts/lib/bench-compare-impl.sh"), *args],
    }


def command_criterion_home(repo: Path, env: Mapping[str, str]) -> Path:
    """Preserve Criterion's output lookup from Cargo's benchmark working directory."""
    run_cwd = repo / "crates/inference"
    if "CRITERION_HOME" in env:
        home = Path(env["CRITERION_HOME"])
    elif "CARGO_TARGET_DIR" in env:
        home = Path(env["CARGO_TARGET_DIR"]) / "criterion"
    else:
        metadata = subprocess.run(
            ["cargo", "metadata", "--locked", "--offline", "--no-deps", "--format-version=1"],
            cwd=repo, env=_cargo_environment(env), capture_output=True, text=True,
        )
        if metadata.returncode:
            raise AdmissionError(f"cannot resolve Criterion output directory: {metadata.stderr.strip()}")
        try:
            target_directory = json.loads(metadata.stdout)["target_directory"]
            if not isinstance(target_directory, str) or not Path(target_directory).is_absolute():
                raise ValueError("target_directory must be absolute")
            home = Path(target_directory) / "criterion"
        except (ValueError, KeyError, TypeError) as exc:
            raise AdmissionError(f"invalid Cargo target-directory metadata: {exc}") from exc
    return (home if home.is_absolute() else run_cwd / home).resolve()


def plan_command(repo: Path, command: list[str], env: Mapping[str, str]) -> dict:
    """Admit one explicit Cargo bench command; unknown command grammar refuses."""
    repo = Path(repo).resolve()
    platform = _host(env)
    if Path.cwd().resolve() != repo:
        raise AdmissionError("explicit bench-command must run from its repository root")
    if command[:2] != ["cargo", "bench"]:
        raise AdmissionError("GPU handoff accepts only cargo bench with one explicit target")
    rest = list(command[2:])
    values: dict[str, str] = {}
    locked = False
    criterion_args: list[str] = []
    while rest:
        flag = rest.pop(0)
        if flag == "--":
            criterion_args = _criterion_args(rest)
            break
        if flag == "--locked" and not locked:
            locked = True
            continue
        key = {"-p": "package", "--package": "package", "--bench": "target", "--features": "features"}.get(flag)
        if key is None or key in values or not rest:
            raise AdmissionError(f"unsupported or repeated cargo bench argument {flag!r}")
        values[key] = rest.pop(0)
    if not locked or values.get("package") != PACKAGE or "target" not in values:
        raise AdmissionError("require cargo bench --locked -p lattice-inference --bench TARGET")
    revision = _revision(repo, "HEAD")
    _clean_revision(repo, revision)
    home = command_criterion_home(repo, env)
    features = values.get("features", "")
    entry = _entry(repo, revision, values["target"], features, entry_id="command", cwd=repo,
                   criterion_home=home.resolve(), criterion_args=criterion_args, platform=platform)
    return {
        "version": PROTOCOL, "mode": "command", "repo": str(repo), "entries": [entry],
        "environment": {},
        "command": [sys.executable, str(repo / "scripts/lib/bench_admission.py"), "measure",
                    "--entry", "command", "--revision", revision, "--target", entry["target"],
                    "--features", features, "--", *criterion_args],
    }


def validate_artifact(entry: dict, artifact: dict, cwd: Path) -> Path:
    """Validate Cargo's selected bench record against the admitted source/configuration."""
    target = artifact.get("target", {})
    source = cwd / entry["source_path"]
    if (
        artifact.get("reason") != "compiler-artifact"
        or target.get("name") != entry["target"]
        or target.get("kind") != ["bench"]
        or not isinstance(target.get("src_path"), str)
        or Path(target["src_path"]).resolve() != source.resolve()
        or sorted(artifact.get("features", [])) != entry["feature_set"]
    ):
        raise AdmissionError("Cargo artifact does not match admitted bench source/features")
    package_id = artifact.get("package_id", "")
    package_uri = (cwd / "crates/inference").resolve().as_uri()
    expected_id = f"path+{package_uri}#{PACKAGE}@{entry['package_version']}"
    if package_id != expected_id:
        raise AdmissionError(f"Cargo package identity differs from {expected_id}")
    executable = artifact.get("executable")
    if not isinstance(executable, str):
        raise AdmissionError("Cargo emitted no executable for admitted target")
    executable_path = Path(executable).resolve()
    approved = Path(entry["target_dir"]).resolve()
    if not executable_path.is_relative_to(approved):
        raise AdmissionError("Cargo executable is outside the admitted build directory")
    try:
        info = executable_path.stat()
    except OSError as exc:
        raise AdmissionError(f"cannot inspect Cargo executable: {exc}") from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(executable_path, os.X_OK):
        raise AdmissionError("Cargo artifact is not a regular executable")
    return executable_path


def validate_measurement_request(entry: dict, request: dict) -> Path:
    """Independently validate one broker request before the supervisor launches it."""
    expected = {
        "entry": entry["id"], "cwd": entry["cwd"], "revision": entry["revision"],
        "run_cwd": entry["run_cwd"],
        "target": entry["target"], "features": entry["features"],
        "criterion_home": entry["criterion_home"], "argv": entry["argv"],
    }
    for field, value in expected.items():
        if request.get(field) != value:
            raise AdmissionError(f"measurement request changed admitted {field}")
    cwd = Path(entry["cwd"])
    _clean_revision(cwd, entry["revision"])
    artifact = request.get("artifact")
    if not isinstance(artifact, dict):
        raise AdmissionError("measurement request lacks its Cargo artifact record")
    executable = validate_artifact(entry, artifact, cwd)
    if request.get("executable") != str(executable):
        raise AdmissionError("measurement request changed selected Cargo executable")
    return executable


def _cargo_environment(env: Mapping[str, str] | None = None) -> dict[str, str]:
    return {
        key: value for key, value in (os.environ if env is None else env).items()
        if not key.startswith("LATTICE_GPU_HANDOFF_")
        and key not in ("LATTICE_BENCH_LOCK_FDS", "LATTICE_BENCH_SUPERVISOR_FD")
    }


def measure(args: argparse.Namespace) -> int:
    from bench_handoff import read_frozen_plan

    cwd = Path.cwd().resolve()
    plan = read_frozen_plan(Path(__file__).resolve().parents[2])
    matches = [entry for entry in plan["entries"] if entry["id"] == args.entry]
    if len(matches) != 1:
        raise AdmissionError("measurement entry is absent from the frozen plan")
    entry = matches[0]
    criterion_args = args.criterion[1:] if args.criterion[:1] == ["--"] else args.criterion
    if (str(cwd), args.revision, args.target, args.features, ["--bench", *criterion_args]) != (
        entry["cwd"], entry["revision"], entry["target"], entry["features"], entry["argv"]
    ):
        raise AdmissionError("measurement helper arguments differ from the frozen plan")
    home = Path(os.environ.get("CRITERION_HOME", entry["criterion_home"]))
    if not home.is_absolute():
        home = Path(entry["run_cwd"]) / home
    if home.resolve() != Path(entry["criterion_home"]).resolve():
        raise AdmissionError("measurement evidence directory differs from the frozen plan")
    _clean_revision(cwd, entry["revision"])
    command = ["cargo", "bench", "--locked", "-p", entry["package"], "--bench", entry["target"],
               "--no-run", "--message-format=json", "--target-dir", entry["target_dir"]]
    if entry["features"]:
        command.extend(["--features", entry["features"]])
    build = subprocess.run(command, cwd=cwd, env=_cargo_environment(), capture_output=True, text=True)
    if build.stderr:
        print(build.stderr, file=sys.stderr, end="")
    if build.returncode:
        return build.returncode
    artifacts = []
    for line in build.stdout.splitlines():
        try:
            artifact = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdmissionError("Cargo emitted an invalid JSON build record") from exc
        if artifact.get("reason") == "compiler-artifact" and artifact.get("executable"):
            target = artifact.get("target", {})
            if target.get("name") == entry["target"] and target.get("kind") == ["bench"]:
                artifacts.append(artifact)
    if len(artifacts) != 1:
        raise AdmissionError("Cargo did not emit exactly one selected benchmark executable")
    executable = validate_artifact(entry, artifacts[0], cwd)
    request = {
        "protocol": PROTOCOL, "token": os.environ.get(BROKER_TOKEN_ENV), "entry": entry["id"],
        "cwd": str(cwd), "run_cwd": entry["run_cwd"],
        "revision": entry["revision"], "target": entry["target"],
        "features": entry["features"], "executable": str(executable),
        "criterion_home": entry["criterion_home"], "argv": entry["argv"], "artifact": artifacts[0],
    }
    broker = os.environ.get(BROKER_ENV)
    if not broker or not request["token"]:
        raise AdmissionError("measurement helper has no complete broker capability")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(broker)
        client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
        with client.makefile("rb") as stream:
            while True:
                line = stream.readline(MAX_FRAME + 1)
                if not line or len(line) > MAX_FRAME or not line.endswith(b"\n"):
                    raise AdmissionError("broker ended without a valid completion frame")
                frame = json.loads(line)
                if set(frame) == {"output"}:
                    chunk = base64.b64decode(frame["output"], validate=True)
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                elif set(frame) == {"status"} and type(frame["status"]) is int:
                    status = frame["status"]
                    return status if 0 <= status <= 255 else 2
                else:
                    raise AdmissionError("broker returned an unsupported output frame")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("measure")
    run.add_argument("--entry", required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--target", required=True)
    run.add_argument("--features", required=True)
    run.add_argument("criterion", nargs=argparse.REMAINDER)
    try:
        return measure(parser.parse_args())
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"bench-admission: {exc}; refusing to measure", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
