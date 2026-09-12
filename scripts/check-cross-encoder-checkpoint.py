#!/usr/bin/env python3
"""Fetch a pinned checkpoint unchanged, then invoke the Rust loading example."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request


def fetch_checkpoint(config, destination):
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, expected in config["files"].items():
        path = destination / name
        if not path.exists():
            url = (
                f"https://huggingface.co/{config['model_id']}/resolve/"
                f"{config['revision']}/{name}"
            )
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read(expected["size"] + 1)
            verify_file(name, data, expected)
            with path.open("xb") as output:
                output.write(data)
        data = path.read_bytes()
        verify_file(name, data, expected)
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def verify_file(name, data, expected):
    if len(data) != expected["size"]:
        raise ValueError(f"Unexpected size for {name}: {len(data)}")
    if "sha256" in expected:
        actual = hashlib.sha256(data).hexdigest()
        wanted = expected["sha256"]
    else:
        blob = f"blob {len(data)}\0".encode() + data
        actual = hashlib.sha1(blob).hexdigest()
        wanted = expected["git_blob_sha1"]
    if actual != wanted:
        raise ValueError(f"Checksum mismatch for {name}: {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--cargo", default="cargo")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    config = json.loads((root / "scripts/cross-encoder-checkpoint.json").read_text())
    hashes = fetch_checkpoint(config, args.model_dir)
    print(json.dumps({"model_id": config["model_id"], "revision": config["revision"],
                      "sha256": hashes}, sort_keys=True), flush=True)
    if args.fetch_only:
        return 0
    return subprocess.run(
        [args.cargo, "run", "--locked", "-p", "lattice-inference", "--example",
         "load_cross_encoder", "--", str(args.model_dir.resolve())],
        cwd=root,
        check=False,
    ).returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"CHECKPOINT_INPUT_ERROR: {error}", file=sys.stderr)
        sys.exit(2)
