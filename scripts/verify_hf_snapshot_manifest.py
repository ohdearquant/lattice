#!/usr/bin/env python3
"""Verify a restored HuggingFace snapshot directory against a committed hash
manifest. Fails closed: a missing manifest, a missing file, a size mismatch,
or a digest mismatch is a hard failure (non-zero exit) — absence is never
treated as a pass.

The manifest (scripts/hf_manifests/<name>.json) is captured OFFLINE from the
HuggingFace model-info API's per-file digests at a pinned commit revision —
an INDEPENDENT trusted source — and committed to the repo. This script never
regenerates the manifest from the artifact it is checking: hashing a
downloaded snapshot and comparing it against a manifest derived from that
same download proves nothing (it always "passes", even for a fully
substituted snapshot).

LFS-tracked files carry the LFS sha256 digest in the manifest. Small non-LFS
files carry the git blob sha1 (sha1(b"blob " + str(len(data)).encode() +
b"\\0" + data)) — this is exactly what the HF API's `blobId` field reports
for a repo file, and it is reproducible locally with no network access.

Usage:
    python3 scripts/verify_hf_snapshot_manifest.py \\
        --snapshot-dir ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<sha> \\
        --manifest scripts/hf_manifests/qwen3.5-0.8b.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fail(msg: str) -> None:
    print(f"::error::[hf-manifest-verify] {msg}", file=sys.stderr)
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        fail(
            f"manifest not found at {manifest_path} — a missing manifest is "
            "a hard failure, never treated as a pass"
        )

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        fail(f"manifest at {manifest_path} is not valid JSON: {e}")

    files = manifest.get("files")
    if not files:
        fail(
            f"manifest at {manifest_path} has no 'files' entries — refusing "
            "to pass on an empty manifest"
        )

    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    if not snapshot_dir.is_dir():
        fail(f"snapshot dir not found at {snapshot_dir}")

    mismatches = []
    for rel_name, expect in sorted(files.items()):
        fpath = snapshot_dir / rel_name
        if not fpath.is_file():
            mismatches.append(f"{rel_name}: MISSING (expected at {fpath})")
            continue

        data = fpath.read_bytes()
        actual_size = len(data)
        if actual_size != expect["size"]:
            mismatches.append(
                f"{rel_name}: size mismatch (expected {expect['size']}, got {actual_size})"
            )
            continue

        if "sha256" in expect:
            algo, expected, actual = "sha256", expect["sha256"], sha256_hex(data)
        else:
            algo, expected, actual = (
                "git_blob_sha1",
                expect["git_blob_sha1"],
                git_blob_sha1(data),
            )

        if actual != expected:
            mismatches.append(
                f"{rel_name}: {algo} mismatch (expected {expected}, got {actual})"
            )

    if mismatches:
        fail(
            f"{len(mismatches)} file(s) failed manifest verification against "
            f"{manifest_path} (revision {manifest.get('revision', '?')}):\n  "
            + "\n  ".join(mismatches)
        )

    print(
        f"PASS: {len(files)} file(s) verified against manifest {manifest_path} "
        f"(revision {manifest.get('revision', '?')})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
