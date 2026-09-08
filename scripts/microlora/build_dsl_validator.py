#!/usr/bin/env python3
"""Build the adjacent Rust validator against a local khive checkout, offline."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint(runtime_repo):
    paths = [runtime_repo / "crates/Cargo.toml"]
    for crate in ("khive-request", "khive-types"):
        root = runtime_repo / "crates" / crate
        paths.append(root / "Cargo.toml")
        paths.extend(sorted((root / "src").rglob("*.rs")))
    hashes = {
        str(path.relative_to(runtime_repo)): digest(path) for path in sorted(paths)
    }
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), hashes


def optional_git(runtime_repo, arguments):
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=runtime_repo,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def is_within(path, directory):
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def build(args):
    runtime_repo = args.runtime_repo.expanduser().resolve(strict=True)
    source = Path(__file__).resolve().with_name("dsl_validator.rs")
    validator_source = source.read_bytes()
    source_sha, source_files = source_fingerprint(runtime_repo)
    output = args.out.expanduser().resolve()
    target = args.target_dir.expanduser().resolve() if args.target_dir else None
    for path in [output, *([target] if target else [])]:
        if is_within(path, runtime_repo):
            raise ValueError("--out and --target-dir must be outside --runtime-repo")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = output.with_name(output.name + ".receipt.json")
    lock_path = output.with_name(output.name + ".Cargo.lock")
    log_path = output.with_name(output.name + ".build.log")
    cargo = ["cargo"] + ([f"+{args.toolchain}"] if args.toolchain else [])
    rustc = ["rustc"] + ([f"+{args.toolchain}"] if args.toolchain else [])
    manifest = "\n".join(
        [
            "[package]",
            'name = "khive-dsl-validator"',
            'version = "0.1.0"',
            'edition = "2021"',
            "publish = false",
            "",
            "[workspace]",
            "",
            "[dependencies]",
            "khive-request = { path = "
            + json.dumps(str(runtime_repo / "crates/khive-request"), ensure_ascii=False)
            + " }",
            'serde = { version = "1.0", features = ["derive"] }',
            'serde_json = "1.0"',
            "",
            "[profile.dev]",
            "debug = 0",
            "incremental = false",
            "",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="khive-dsl-validator-") as temporary:
        project = Path(temporary)
        (project / "src").mkdir()
        (project / "src/main.rs").write_bytes(validator_source)
        (project / "Cargo.toml").write_text(manifest)
        if lock_path.exists():
            shutil.copyfile(lock_path, project / "Cargo.lock")
        build_target = target if target else project / "target"
        build_target.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "CARGO_TARGET_DIR": str(build_target),
            "KHIVE_PARSER_SOURCE_SHA": source_sha,
        }
        command = [
            *cargo,
            "build",
            "--offline",
            "--manifest-path",
            str(project / "Cargo.toml"),
        ]
        if (project / "Cargo.lock").exists():
            command.append("--locked")
        with log_path.open("w") as log:
            subprocess.run(
                command,
                cwd=project,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if source_fingerprint(runtime_repo) != (source_sha, source_files):
            raise RuntimeError(
                "Parser sources changed during compilation; rebuild against a stable checkout"
            )
        if source.read_bytes() != validator_source:
            raise RuntimeError("Validator source changed during compilation")
        executable = (
            build_target
            / "debug"
            / ("khive-dsl-validator.exe" if os.name == "nt" else "khive-dsl-validator")
        )
        self_test_run = subprocess.run(
            [str(executable), "--self-test"],
            cwd=project,
            text=True,
            capture_output=True,
            check=True,
        )
        self_test = json.loads(self_test_run.stdout)
        if (
            self_test.get("ok") is not True
            or self_test.get("parser") != "khive_request::parse_request"
            or self_test.get("parser_source_sha256") != source_sha
            or self_test.get("executed") is not False
        ):
            raise RuntimeError(
                "Built validator failed its parser identity/self-test check"
            )
        rejection = subprocess.run(
            [str(executable)],
            cwd=project,
            input=json.dumps({"completion": "stats("}) + "\n",
            text=True,
            capture_output=True,
            check=False,
        )
        rejected_row = json.loads(rejection.stdout)
        if (
            rejection.returncode != 1
            or rejected_row.get("ok") is not False
            or rejected_row.get("parser") != "khive_request::parse_request"
        ):
            raise RuntimeError("Built validator did not reject the malformed DSL probe")
        shutil.copy2(executable, output)
        shutil.copyfile(project / "Cargo.lock", lock_path)
        receipt = {
            "parser": "khive_request::parse_request",
            "parser_source_sha256": source_sha,
            "source_files": source_files,
            "runtime_repo": str(runtime_repo),
            "runtime_git_head": optional_git(runtime_repo, ["rev-parse", "HEAD"]),
            "runtime_source_git_status": optional_git(
                runtime_repo,
                [
                    "status",
                    "--porcelain",
                    "--",
                    "crates/khive-request",
                    "crates/khive-types",
                    "crates/Cargo.toml",
                ],
            ),
            "validator_source_sha256": hashlib.sha256(validator_source).hexdigest(),
            "builder_source_sha256": digest(Path(__file__).resolve()),
            "binary": str(output),
            "binary_sha256": digest(output),
            "lockfile": str(lock_path),
            "lockfile_sha256": digest(lock_path),
            "manifest": manifest,
            "build_command": command,
            "build_environment": {
                key: env[key] for key in ("CARGO_TARGET_DIR", "KHIVE_PARSER_SOURCE_SHA")
            },
            "target_directory_retained": target is not None,
            "rustc": subprocess.check_output(
                [*rustc, "--version"], cwd=project, text=True
            ).strip(),
            "cargo": subprocess.check_output(
                [*cargo, "--version"], cwd=project, text=True
            ).strip(),
            "self_test": self_test,
            "malformed_probe_exit": rejection.returncode,
            "offline": True,
            "executes_dsl": False,
        }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "binary": str(output),
                "receipt": str(receipt_path),
                "parser_source_sha256": source_sha,
                "binary_sha256": receipt["binary_sha256"],
                "ok": True,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-repo",
        required=True,
        type=Path,
        help="Local khive repository containing crates/khive-request",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Executable output path, outside the runtime repository",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        help="Dedicated build cache; omitted means a temporary cache removed after building",
    )
    parser.add_argument(
        "--toolchain", help="Optional installed Rust toolchain, for example 1.94.1"
    )
    args = parser.parse_args()
    try:
        build(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"build_dsl_validator: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
