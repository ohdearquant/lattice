#!/usr/bin/env python3
"""Generate rustdoc JSON for the public API listing fixtures."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

FIXTURE_CRATES = {
    "baseline": ["listing_baseline"],
    "split": ["listing_core", "listing_baseline"],
    "glob": ["listing_core", "listing_baseline"],
    "missing": ["listing_core", "listing_baseline"],
    "added": ["listing_core", "listing_baseline"],
}


def package_manifest(workspace: Path, crate: str) -> Path:
    manifests = sorted(workspace.glob("*/Cargo.toml"))
    matches = [
        path
        for path in manifests
        if any(
            line.strip() == f'name = "{crate}"'
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve manifest for {crate} under {workspace}: {matches}")
    return matches[0]


def generate(fixture_root: Path, output_dir: Path) -> None:
    for fixture, crates in FIXTURE_CRATES.items():
        workspace = fixture_root / fixture
        fixture_output = output_dir / fixture
        fixture_output.mkdir(parents=True, exist_ok=True)
        for crate in crates:
            manifest = package_manifest(workspace, crate) if fixture != "baseline" else workspace / "Cargo.toml"
            command = [
                "cargo",
                "rustdoc",
                "--manifest-path",
                str(manifest),
                "--",
                "-Z",
                "unstable-options",
                "--output-format",
                "json",
                "--document-hidden-items",
            ]
            environment = os.environ.copy()
            environment["RUSTC_BOOTSTRAP"] = "1"
            result = subprocess.run(command, capture_output=True, text=True, env=environment)
            if result.returncode != 0:
                sys.stderr.write(result.stdout)
                sys.stderr.write(result.stderr)
                raise RuntimeError(f"cargo rustdoc failed for {crate}")
            target_dir = Path(environment.get("CARGO_TARGET_DIR", "target"))
            json_path = target_dir / "doc" / f"{crate}.json"
            if not json_path.is_file():
                raise RuntimeError(f"rustdoc did not produce {json_path}")
            destination = fixture_output / f"{crate}.json"
            destination.write_bytes(json_path.read_bytes())
            print(f"{destination}: {destination.stat().st_size} bytes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fixture_root = Path(__file__).resolve().parent
    generate(fixture_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
