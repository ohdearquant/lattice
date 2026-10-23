#!/usr/bin/env python3
"""Check ADR-096's router-learning boundary in Cargo-reported library artifacts.

Run the enabled build first as a positive control, then mixture and defaults.
Cargo may reuse its cache: only artifacts reported by that invocation are read.
No inference or training executable is launched.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path


VARIANTS = ("router-learning", "mixture", "default")
CRATES = {"lattice_inference": "inference", "lattice_fann": "fann"}
TRAINER_METHOD = "step_batch_with_explored_actions"
ONLINE_MODULES = ("rloo", "router_update", "ewc")


class ProbeError(Exception):
    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


def run_command(command: list[str], root: Path) -> str:
    try:
        result = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=False
        )
    except (OSError, UnicodeError) as error:
        raise ProbeError(f"cannot run {shlex.join(command)}: {error}", 2) from error
    if result.returncode:
        status = result.returncode
        exit_code = status if status > 0 else 128 - status
        raise ProbeError(
            f"command failed (exit {status}): {shlex.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            exit_code,
        )
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    return result.stdout


def validate_artifact(
    record: dict, crate: str, root: Path, target_dir: Path, variant: str
) -> Path:
    target = record["target"]
    crate_dir = root / "crates" / CRATES[crate]
    for value, expected in (
        (record.get("manifest_path"), crate_dir / "Cargo.toml"),
        (target.get("src_path"), crate_dir / "src" / "lib.rs"),
    ):
        if not isinstance(value, str) or Path(value).resolve() != expected.resolve():
            raise ProbeError(f"{variant}: {crate} has an unexpected source or manifest")
    if (
        target.get("kind") != ["lib"]
        or target.get("crate_types") != ["lib"]
        or record.get("executable") is not None
        or not isinstance(record.get("profile"), dict)
        or record["profile"].get("test") is not False
    ):
        raise ProbeError(f"{variant}: {crate} is not a non-test library artifact")

    features = record.get("features")
    if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
        raise ProbeError(f"{variant}: {crate} has invalid Cargo feature evidence")
    expected_features = (
        {"router-learning": variant == "router-learning", "mixture": variant != "default"}
        if crate == "lattice_inference"
        else {"online-router": variant == "router-learning"}
    )
    for feature, enabled in expected_features.items():
        if (feature in features) != enabled:
            raise ProbeError(f"{variant}: {crate} has unexpected feature {feature}")

    filenames = record.get("filenames")
    if not isinstance(filenames, list) or not all(isinstance(item, str) for item in filenames):
        raise ProbeError(f"{variant}: {crate} has invalid Cargo filenames")
    libraries = [Path(item) for item in filenames if Path(item).suffix == ".rlib"]
    if len(libraries) != 1:
        raise ProbeError(f"{variant}: expected exactly one {crate} .rlib, found {len(libraries)}")
    library = libraries[0].resolve()
    if not library.is_relative_to(target_dir.resolve()):
        raise ProbeError(f"{variant}: {crate} artifact is outside the requested target directory")
    try:
        if not stat.S_ISREG(library.stat().st_mode):
            raise ProbeError(f"{variant}: {library} is not a regular file")
        with library.open("rb") as stream:
            if stream.read(8) != b"!<arch>\n":
                raise ProbeError(f"{variant}: {library} is not an rlib archive")
    except OSError as error:
        raise ProbeError(f"{variant}: cannot read {library}: {error}") from error
    return library


def collect_artifacts(
    output: str, root: Path, target_dir: Path, variant: str
) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    finished = False
    for number, line in enumerate(output.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            raise ProbeError(f"{variant}: invalid Cargo JSON at line {number}: {error}") from error
        if not isinstance(record, dict) or not isinstance(record.get("reason"), str):
            raise ProbeError(f"{variant}: invalid Cargo message at line {number}")
        if finished:
            raise ProbeError(f"{variant}: Cargo message after build-finished")
        reason = record["reason"]
        if reason == "build-finished":
            if record.get("success") is not True:
                raise ProbeError(f"{variant}: Cargo build-finished did not report success")
            finished = True
        elif reason == "compiler-artifact":
            target = record.get("target")
            if not isinstance(target, dict) or not isinstance(target.get("name"), str):
                raise ProbeError(f"{variant}: malformed Cargo artifact at line {number}")
            crate = target["name"]
            if crate in CRATES:
                if crate in artifacts:
                    raise ProbeError(f"{variant}: duplicate Cargo artifact for {crate}")
                artifacts[crate] = validate_artifact(record, crate, root, target_dir, variant)
    if not finished:
        raise ProbeError(f"{variant}: Cargo build-finished success is missing")
    required = {"lattice_inference"}
    if variant != "default":
        required.add("lattice_fann")
    missing = required - artifacts.keys()
    if missing:
        raise ProbeError(f"{variant}: missing Cargo library artifacts: {', '.join(sorted(missing))}")
    return artifacts


def defined_symbols(output: str, library: Path) -> set[str]:
    """Read portable nm -P output without treating U/w/v references as definitions."""
    defined: set[str] = set()
    for number, line in enumerate(output.splitlines(), 1):
        line = line.strip()
        if not line or line.endswith(":"):
            continue
        fields = line.split()
        if (
            not 2 <= len(fields) <= 4
            or re.fullmatch(r"[A-Za-z?]", fields[1]) is None
            or any(re.fullmatch(r"[0-9A-Fa-f]+", value) is None for value in fields[2:])
        ):
            raise ProbeError(f"{library}: unrecognized nm -P output at line {number}: {line}")
        if fields[1] == "?":
            raise ProbeError(f"{library}: nm could not classify symbol at line {number}: {line}")
        if fields[1] not in {"U", "w", "v"}:
            if len(fields) < 3:
                raise ProbeError(f"{library}: defined nm symbol has no address: {line}")
            defined.add(fields[0])
    if not defined:
        raise ProbeError(f"{library}: nm reported no defined symbols")
    return defined


def is_online_symbol(symbol: str) -> bool:
    # Both Rust mangling formats retain these identifiers; no demangler is needed.
    return (
        "lattice_fann" in symbol
        and "training" in symbol
        and any(module in symbol for module in ONLINE_MODULES)
    )


def check(root: Path, target_dir: Path, cargo: str = "cargo", nm: str = "nm") -> None:
    for variant in VARIANTS:
        command = [
            cargo, "build", "--locked", "--lib", "-p", "lattice-inference",
            "--manifest-path", str(root / "Cargo.toml"),
            "--message-format=json", "--target-dir", str(target_dir),
        ]
        if variant != "default":
            command.extend(["--features", variant])
        print(f"Checking {variant}: {shlex.join(command)}", flush=True)
        artifacts = collect_artifacts(run_command(command, root), root, target_dir, variant)
        found: dict[str, set[str]] = {}
        for crate, library in sorted(artifacts.items()):
            symbols = defined_symbols(run_command([nm, "-P", str(library)], root), library)
            found[crate] = {symbol for symbol in symbols if is_online_symbol(symbol)}
            print(f"  {crate}: {library} ({len(symbols)} defined symbols)", flush=True)
        if variant == "router-learning":
            control = {
                symbol for symbol in found["lattice_fann"]
                if "RlooTrainer" in symbol and TRAINER_METHOD in symbol
            }
            if not control:
                raise ProbeError(f"{variant}: positive control missing defined RlooTrainer::{TRAINER_METHOD}")
            print(f"PASS {variant}: defined RlooTrainer::{TRAINER_METHOD}", flush=True)
        else:
            leaked = sorted(set().union(*found.values()))
            if leaked:
                raise ProbeError(f"{variant}: online-router definitions present:\n" + "\n".join(leaked))
            print(f"PASS {variant}: no defined online-router symbols", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo", default="cargo", help="Cargo executable")
    parser.add_argument("--nm", default="nm", help="nm executable supporting POSIX -P output")
    parser.add_argument("--target-dir", type=Path, help="Cargo output directory for all three builds")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    target_dir = (args.target_dir or root / "target" / "router-learning-artifacts").resolve()
    try:
        check(root, target_dir, args.cargo, args.nm)
    except ProbeError as error:
        print(f"router-learning artifact check: {error}", file=sys.stderr)
        return error.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
