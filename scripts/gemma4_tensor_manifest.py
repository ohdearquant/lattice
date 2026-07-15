#!/usr/bin/env python3
"""Extract the Gemma 4 E2B tensor manifest from the checkpoint's
`model.safetensors` header via HTTP Range requests — zero weight bytes
fetched, ever.

Safetensors layout: bytes 0-7 are a little-endian u64 giving the JSON
header length N; bytes 8..8+N are the header itself (tensor name -> dtype/
shape/data_offsets). This script fetches exactly those two ranges
(~264 KB total for this checkpoint) and never touches the ~10.25 GB of
weight data that follows.

Pinned checkpoint (ADR-082): `google/gemma-4-E2B-it` @ revision
`9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`. The revision is baked into the
URL (the `/resolve/<commit>/` form) so the pin is real, not advisory.

Usage:
    # Verify (default): fetch the header, diff against the committed
    # fixture, fail closed on any drift. This is the CI / Stage-0 gate.
    uv run python3 scripts/gemma4_tensor_manifest.py

    # Regenerate the committed fixture (deliberate, reviewable, never run
    # by CI):
    uv run python3 scripts/gemma4_tensor_manifest.py --write-fixture
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = "google/gemma-4-E2B-it"
REVISION = "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf"
SAFETENSORS_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors"

# Hard cap on total bytes ever fetched by this script. The header for this
# checkpoint is ~264 KB; 1 MB leaves headroom without coming close to the
# ~10.25 GB weight payload.
MAX_FETCH_BYTES = 1_000_000

DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "crates"
    / "inference"
    / "tests"
    / "fixtures"
    / "gemma4"
    / "e2b_tensor_manifest.json"
)

# Ground truth confirmed by header extraction on 2026-07-14 (see G16 in
# ADR-082). If a re-fetch disagrees, that is drift — investigate, do not
# adjust these numbers to make a script pass.
EXPECTED_TOTAL = 2011
EXPECTED_BUCKETS = {
    "model.audio_tower": 751,
    "model.vision_tower": 658,
    "model.language_model": 600,
    "model.embed_audio": 1,
    "model.embed_vision": 1,
}


def _http_get_range(url: str, start: int, end_inclusive: int) -> bytes:
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end_inclusive}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (pinned https HF URL)
        if resp.status not in (200, 206):
            raise RuntimeError(
                f"unexpected HTTP status {resp.status} fetching {url} "
                f"range {start}-{end_inclusive}"
            )
        return resp.read()


def fetch_header(url: str = SAFETENSORS_URL) -> tuple[dict[str, Any], bytes, int]:
    """Fetch the safetensors header via two Range requests.

    Returns (header_dict, raw_header_bytes, total_bytes_fetched). Fails
    closed (raises) before ever fetching more than MAX_FETCH_BYTES.
    """
    len_bytes = _http_get_range(url, 0, 7)
    if len(len_bytes) != 8:
        raise RuntimeError(
            f"expected 8 bytes for the safetensors header-length prefix, got {len(len_bytes)}"
        )
    header_len = struct.unpack("<Q", len_bytes)[0]
    if 8 + header_len > MAX_FETCH_BYTES:
        raise RuntimeError(
            f"declared header length {header_len} bytes would push total fetch past "
            f"the {MAX_FETCH_BYTES}-byte cap — refusing to fetch further"
        )

    header_bytes = _http_get_range(url, 8, 8 + header_len - 1)
    total_fetched = len(len_bytes) + len(header_bytes)
    if total_fetched > MAX_FETCH_BYTES:
        raise RuntimeError(
            f"fetched {total_fetched} bytes, exceeding the {MAX_FETCH_BYTES}-byte cap"
        )
    if len(header_bytes) != header_len:
        raise RuntimeError(
            f"declared header length {header_len} but received {len(header_bytes)} bytes"
        )

    header = json.loads(header_bytes.decode("utf-8"))
    return header, header_bytes, total_fetched


def bucket_of(tensor_name: str) -> str:
    parts = tensor_name.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return tensor_name


def build_manifest(
    header: dict[str, Any], header_bytes: bytes, total_fetched: int
) -> dict[str, Any]:
    tensors: dict[str, dict[str, Any]] = {}
    buckets: dict[str, int] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        tensors[name] = {"dtype": meta["dtype"], "shape": meta["shape"]}
        b = bucket_of(name)
        buckets[b] = buckets.get(b, 0) + 1

    return {
        "metadata": {
            "source_repo": REPO,
            "revision": REVISION,
            "source_url": SAFETENSORS_URL,
            "header_length_bytes": len(header_bytes),
            "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
            "total_bytes_fetched": total_fetched,
            "extraction_date": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        },
        "bucket_counts": buckets,
        "total_tensors": len(tensors),
        "tensors": tensors,
    }


def diff_against_fixture(
    manifest: dict[str, Any], fixture: dict[str, Any]
) -> list[str]:
    """Full name/shape/dtype/bucket diff between a freshly fetched manifest
    and the committed fixture. Any non-empty return means drift."""
    errors: list[str] = []

    if manifest["total_tensors"] != fixture["total_tensors"]:
        errors.append(
            "total tensor count drift: fetched="
            f"{manifest['total_tensors']} fixture={fixture['total_tensors']}"
        )

    fetched_tensors = manifest["tensors"]
    fixture_tensors = fixture["tensors"]
    fetched_names = set(fetched_tensors)
    fixture_names = set(fixture_tensors)

    for missing in sorted(fixture_names - fetched_names):
        errors.append(f"tensor missing from fetched header: {missing}")
    for extra in sorted(fetched_names - fixture_names):
        errors.append(f"unexpected tensor in fetched header: {extra}")

    for name in sorted(fetched_names & fixture_names):
        f_meta = fetched_tensors[name]
        x_meta = fixture_tensors[name]
        if f_meta["dtype"] != x_meta["dtype"] or f_meta["shape"] != x_meta["shape"]:
            errors.append(
                f"shape/dtype drift for {name}: fetched={f_meta} fixture={x_meta}"
            )

    for bucket, expected_count in fixture.get("bucket_counts", {}).items():
        actual_count = manifest["bucket_counts"].get(bucket, 0)
        if actual_count != expected_count:
            errors.append(
                f"bucket {bucket} count drift: fetched={actual_count} fixture={expected_count}"
            )

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-fixture",
        action="store_true",
        help="(Re)generate the committed fixture from a live fetch instead of "
        "diffing against it. Run this deliberately and commit the review-able "
        "diff; never run it from CI.",
    )
    parser.add_argument(
        "--fixture-path",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help=f"default: {DEFAULT_FIXTURE_PATH}",
    )
    args = parser.parse_args(argv)

    print(
        f"Fetching safetensors header from {SAFETENSORS_URL} "
        "(two Range requests, zero weight bytes)...",
        file=sys.stderr,
    )
    header, header_bytes, total_fetched = fetch_header()
    manifest = build_manifest(header, header_bytes, total_fetched)

    print(
        f"Fetched {total_fetched} bytes total (cap: {MAX_FETCH_BYTES}).",
        file=sys.stderr,
    )
    print(f"Total tensors: {manifest['total_tensors']}", file=sys.stderr)
    for bucket, count in sorted(manifest["bucket_counts"].items()):
        print(f"  {bucket}: {count}", file=sys.stderr)

    if manifest["total_tensors"] != EXPECTED_TOTAL:
        print(
            f"FAIL: total tensor count {manifest['total_tensors']} != expected {EXPECTED_TOTAL}",
            file=sys.stderr,
        )
        return 1
    for bucket, expected_count in EXPECTED_BUCKETS.items():
        actual = manifest["bucket_counts"].get(bucket, 0)
        if actual != expected_count:
            print(
                f"FAIL: bucket {bucket} count {actual} != expected {expected_count}",
                file=sys.stderr,
            )
            return 1

    if args.write_fixture:
        args.fixture_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.fixture_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote fixture to {args.fixture_path}", file=sys.stderr)
        return 0

    if not args.fixture_path.exists():
        print(
            f"FAIL: committed fixture not found at {args.fixture_path} — "
            "run with --write-fixture first",
            file=sys.stderr,
        )
        return 1
    with open(args.fixture_path) as f:
        fixture = json.load(f)

    errors = diff_against_fixture(manifest, fixture)
    if errors:
        print(
            f"FAIL: {len(errors)} drift(s) between fetched header and committed fixture:",
            file=sys.stderr,
        )
        for e in errors[:50]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more", file=sys.stderr)
        return 1

    print("OK: fetched header matches committed fixture exactly.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
