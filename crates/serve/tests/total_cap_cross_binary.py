#!/usr/bin/env python3
"""Compare the two binaries' actual HTTP streaming fixtures on a macOS host."""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
MARKER = "TOTAL_CAP_PARITY "
TEST = "cross_binary_total_cap_streaming_fixture"


def run(binary):
    command = [
        "cargo", "test", "--locked", "-p", "lattice-serve", "--bin", binary,
        "--features", "f16,metal-gpu,test-utils", TEST, "--", "--nocapture",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{binary} test failed ({result.returncode})")
    records = [json.loads(line.split(MARKER, 1)[1]) for line in result.stdout.splitlines() if MARKER in line]
    by_case = {record["case"]: record["result"] for record in records}
    if len(records) != 4 or set(by_case) != {"below", "at", "stop_at", "over_after_delta"}:
        raise RuntimeError(f"{binary} did not emit all four unique parity records: {records}")
    return by_case


def main():
    if sys.platform != "darwin":
        raise RuntimeError("both binary fixtures require macOS and metal-gpu")
    unified = run("lattice")
    standalone = run("lattice_serve")
    if unified != standalone:
        raise RuntimeError(f"streaming mismatch: lattice={unified}, lattice_serve={standalone}")
    print(f"Both routers agree: {json.dumps(unified, sort_keys=True)}")


if __name__ == "__main__":
    main()
