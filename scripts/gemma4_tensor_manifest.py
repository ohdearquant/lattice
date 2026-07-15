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

    # Offline self-test (bounded-range-read + drift-mutation cases). No
    # network access; safe to run in CI.
    uv run python3 scripts/gemma4_tensor_manifest.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import struct
import sys
import threading
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


def _http_get_range(
    url: str, start: int, end_inclusive: int, remaining_budget: int
) -> tuple[bytes, int]:
    """Fetch an inclusive byte range under a strict, fail-closed contract.

    The server must answer `206 Partial Content` with a `Content-Range` that
    matches the requested range exactly, and any declared `Content-Length`
    must equal the requested range length. The body is read with an explicit
    size cap of `requested_length + 1` so a streaming body larger than
    declared is rejected — the extra byte is only ever used to detect
    overrun, never retained. `remaining_budget` is checked *before* issuing
    the request, against `requested_length + 1` (not just `requested_length`)
    so the overrun-probe byte itself is reserved in the budget — the read
    call never asks the socket for more than what's left of
    `MAX_FETCH_BYTES`, even when detecting an overlong body. This means both
    range reads in `fetch_header` are jointly bounded by `MAX_FETCH_BYTES`,
    not just individually, and an otherwise-maximum-sized header is rejected
    conservatively rather than letting the probe byte push the true fetch
    past the cap.

    Returns (data, new_remaining_budget).
    """
    expected_len = end_inclusive - start + 1
    read_cap = expected_len + 1
    if read_cap > remaining_budget:
        raise RuntimeError(
            f"requested range {start}-{end_inclusive} ({expected_len} bytes, "
            f"{read_cap} bytes including the overrun probe) exceeds the "
            f"remaining fetch budget ({remaining_budget} bytes)"
        )
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end_inclusive}"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (pinned https HF URL)
        if resp.status != 206:
            raise RuntimeError(
                f"expected HTTP 206 Partial Content fetching {url} range "
                f"{start}-{end_inclusive}, got {resp.status}"
            )
        content_range = resp.headers.get("Content-Range", "")
        expected_prefix = f"bytes {start}-{end_inclusive}/"
        if not content_range.startswith(expected_prefix):
            raise RuntimeError(
                f"Content-Range {content_range!r} does not match requested range "
                f"{expected_prefix}*"
            )
        content_length = resp.headers.get("Content-Length")
        if content_length is not None and int(content_length) != expected_len:
            raise RuntimeError(
                f"declared Content-Length {content_length} != expected {expected_len} "
                f"for range {start}-{end_inclusive}"
            )
        # Read at most one byte past what was requested: if that succeeds,
        # the body is longer than declared and must be rejected before any
        # of it is retained.
        data = resp.read(read_cap)
        if len(data) > expected_len:
            raise RuntimeError(
                f"response body for range {start}-{end_inclusive} exceeded the "
                f"requested {expected_len} bytes (got {len(data)} bytes) — refusing "
                "to retain an overlong body"
            )
    return data, remaining_budget - len(data)


def fetch_header(url: str = SAFETENSORS_URL) -> tuple[dict[str, Any], bytes, int]:
    """Fetch the safetensors header via two Range requests.

    Returns (header_dict, raw_header_bytes, total_bytes_fetched). Fails
    closed (raises) before ever fetching more than MAX_FETCH_BYTES; both
    range reads are checked against a shared, shrinking budget.
    """
    remaining = MAX_FETCH_BYTES
    len_bytes, remaining = _http_get_range(url, 0, 7, remaining)
    if len(len_bytes) != 8:
        raise RuntimeError(
            f"expected 8 bytes for the safetensors header-length prefix, got {len(len_bytes)}"
        )
    header_len = struct.unpack("<Q", len_bytes)[0]
    # Reserve the second request's one-byte overrun probe in this check too
    # (see `_http_get_range`): a header exactly `MAX_FETCH_BYTES - 8` bytes
    # long would otherwise pass this guard and then have its overrun-probe
    # read pull one byte past the global cap before the per-request budget
    # check below catches it. Reject it here instead, conservatively.
    if 8 + header_len + 1 > MAX_FETCH_BYTES:
        raise RuntimeError(
            f"declared header length {header_len} bytes would push total fetch past "
            f"the {MAX_FETCH_BYTES}-byte cap (including the overrun probe byte) — "
            "refusing to fetch further"
        )

    header_bytes, remaining = _http_get_range(url, 8, 8 + header_len - 1, remaining)
    total_fetched = MAX_FETCH_BYTES - remaining
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


# Metadata fields that must be byte-identical between a freshly fetched
# manifest and the committed fixture. `extraction_date` is deliberately
# excluded — it is a generation timestamp, not provenance.
IMMUTABLE_METADATA_FIELDS = (
    "source_repo",
    "revision",
    "source_url",
    "header_length_bytes",
    "header_sha256",
    "total_bytes_fetched",
)


def _is_plain_int(v: Any) -> bool:
    """True for a genuine JSON number deserialized as `int`, false for a
    JSON boolean (Python `bool` is an `int` subclass, so `True == 1`)."""
    return isinstance(v, int) and not isinstance(v, bool)


def diff_against_fixture(
    manifest: dict[str, Any], fixture: dict[str, Any]
) -> list[str]:
    """Full name/shape/dtype/bucket/metadata diff between a freshly fetched
    manifest and the committed fixture. Any non-empty return means drift."""
    errors: list[str] = []

    if manifest["total_tensors"] != fixture["total_tensors"]:
        errors.append(
            "total tensor count drift: fetched="
            f"{manifest['total_tensors']} fixture={fixture['total_tensors']}"
        )

    fetched_meta = manifest.get("metadata", {})
    fixture_meta = fixture.get("metadata", {})
    for field in IMMUTABLE_METADATA_FIELDS:
        f_val = fetched_meta.get(field)
        x_val = fixture_meta.get(field)
        if f_val != x_val:
            errors.append(f"metadata.{field} drift: fetched={f_val!r} fixture={x_val!r}")

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

    # Exact map equality: a bucket dropped from either side, an extra bucket
    # on either side, or a wrong count are all drift — not just the buckets
    # the fixture happens to still enumerate. Values must also be *plain*
    # ints: `bool` is a subclass of `int` in Python, so `True == 1` would
    # otherwise let a JSON `true` silently pass as a count of 1.
    fetched_buckets = manifest.get("bucket_counts", {})
    fixture_buckets = fixture.get("bucket_counts", {})
    for bucket in sorted(set(fetched_buckets) | set(fixture_buckets)):
        f_count = fetched_buckets.get(bucket)
        x_count = fixture_buckets.get(bucket)
        if not _is_plain_int(f_count) or not _is_plain_int(x_count):
            errors.append(
                f"bucket {bucket} count is not a plain int (fetched={f_count!r} "
                f"fixture={x_count!r}) — JSON booleans/floats are not valid "
                "tensor counts"
            )
            continue
        if f_count != x_count:
            errors.append(
                f"bucket {bucket} count drift: fetched={f_count} fixture={x_count}"
            )

    return errors


# ---------------------------------------------------------------------------
# Offline self-test: --self-test. No network access — a local HTTP fixture
# stands in for Hugging Face, and drift-diff mutation cases exercise
# diff_against_fixture() directly.
# ---------------------------------------------------------------------------


def _parse_range_header(range_header: str) -> tuple[int, int]:
    # "bytes=START-END"
    spec = range_header.split("=", 1)[1]
    start_s, end_s = spec.split("-", 1)
    return int(start_s), int(end_s)


class _SelfTestHandler(http.server.BaseHTTPRequestHandler):
    """Serves one canned response per instance, keyed by `scenario`."""

    scenario = "ok"
    ok_payload = b""

    def log_message(self, format_str: str, *args: Any) -> None:  # noqa: A002
        pass  # silence request logging in test output

    def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
        if self.scenario == "200-multi-mb-body":
            body = b"\x00" * (2 * 1024 * 1024)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.scenario == "206-wrong-content-range":
            body = b"12345678"
            self.send_response(206)
            self.send_header("Content-Range", "bytes 500-507/999999999")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.scenario == "206-overlong-body":
            start, end = _parse_range_header(self.headers.get("Range", "bytes=0-7"))
            body = b"\xff" * (2 * 1024 * 1024)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/999999999")
            self.end_headers()
            self.wfile.write(body)
        elif self.scenario == "ok":
            start, end = _parse_range_header(self.headers.get("Range", "bytes=0-7"))
            body = self.ok_payload[start : end + 1]
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{end}/{len(self.ok_payload)}"
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:  # pragma: no cover - defensive
            self.send_response(500)
            self.end_headers()


class _QuietHTTPServer(http.server.HTTPServer):
    """The self-test client deliberately stops reading part-way through
    several scenarios below (that's the point — proving bounded reads), which
    raises BrokenPipeError server-side. Expected, not a self-test failure."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass


def _start_scenario_server(
    scenario: str, ok_payload: bytes = b""
) -> http.server.HTTPServer:
    handler_cls = type(
        "_ScenarioHandler",
        (_SelfTestHandler,),
        {"scenario": scenario, "ok_payload": ok_payload},
    )
    server = _QuietHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _stop_scenario_server(server: http.server.HTTPServer) -> None:
    server.shutdown()
    server.server_close()


def _run_bounded_read_selftest() -> list[str]:
    failures: list[str] = []

    # (a) 200 with a multi-MB body must be rejected without ever reading it.
    server = _start_scenario_server("200-multi-mb-body")
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        try:
            _http_get_range(url, 0, 7, MAX_FETCH_BYTES)
        except RuntimeError:
            pass
        else:
            failures.append("200-multi-mb-body: expected RuntimeError, got success")
    finally:
        _stop_scenario_server(server)

    # (b) 206 with a Content-Range that doesn't match the request.
    server = _start_scenario_server("206-wrong-content-range")
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        try:
            _http_get_range(url, 0, 7, MAX_FETCH_BYTES)
        except RuntimeError:
            pass
        else:
            failures.append(
                "206-wrong-content-range: expected RuntimeError, got success"
            )
    finally:
        _stop_scenario_server(server)

    # (c) 206 with an overlong streaming body: must fail while reading at
    # most requested+1 bytes (asserted via the exception's reported count).
    server = _start_scenario_server("206-overlong-body")
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        try:
            data, _ = _http_get_range(url, 0, 7, MAX_FETCH_BYTES)
        except RuntimeError as e:
            if "got 9 bytes" not in str(e):
                failures.append(
                    f"206-overlong-body: expected overrun message citing 9 bytes "
                    f"(requested 8 + 1), got: {e}"
                )
        else:
            failures.append(
                f"206-overlong-body: expected RuntimeError, got {len(data)} bytes"
            )
    finally:
        _stop_scenario_server(server)

    # (d) Exact-budget boundary: a request whose declared length leaves no
    # room for the reserved overrun-probe byte (expected_len == budget) must
    # be rejected before any request is issued at all — the read call
    # itself must never be allowed to ask for more than the remaining
    # budget, even when the requested length alone would fit.
    budget = 1000
    try:
        _http_get_range("http://127.0.0.1:1/", 8, 8 + budget - 1, budget)
    except RuntimeError as e:
        if "exceeds the remaining fetch budget" not in str(e):
            failures.append(
                f"exact-budget-no-probe-room: expected budget-exceeded message, got: {e}"
            )
    else:
        failures.append("exact-budget-no-probe-room: expected RuntimeError, got success")

    # (e) Same boundary, one byte smaller (expected_len == budget - 1, the
    # max allowed once the probe byte is reserved): an overlong second-range
    # response — reproducing the header-sized (not just 8-byte) request in
    # `fetch_header` — must still be rejected, and the reserved budget check
    # means the read call can pull at most `budget` bytes total, never past
    # the cap.
    server = _start_scenario_server("206-overlong-body")
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        try:
            data, _ = _http_get_range(url, 8, 8 + (budget - 1) - 1, budget)
        except RuntimeError:
            pass
        else:
            failures.append(
                f"exact-budget-overlong-second-range: expected RuntimeError, got {len(data)} bytes"
            )
    finally:
        _stop_scenario_server(server)

    # (f) fetch_header end to end: a header whose declared length is exactly
    # `MAX_FETCH_BYTES - 8` (the previously-buggy "otherwise maximum-sized
    # header" that left zero budget for the second request's overrun probe)
    # must be rejected before the second Range request is ever issued.
    huge_header_len = MAX_FETCH_BYTES - 8
    len_bytes_payload = struct.pack("<Q", huge_header_len)
    server = _start_scenario_server("ok", ok_payload=len_bytes_payload)
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        try:
            fetch_header(url)
        except RuntimeError:
            pass
        else:
            failures.append("max-sized-header: expected RuntimeError, got success")
    finally:
        _stop_scenario_server(server)

    # Valid two-range success case must still work end to end.
    header_obj = {
        "model.dummy.weight": {
            "dtype": "BF16",
            "shape": [1, 1],
            "data_offsets": [0, 2],
        }
    }
    header_bytes = json.dumps(header_obj).encode("utf-8")
    ok_payload = struct.pack("<Q", len(header_bytes)) + header_bytes
    server = _start_scenario_server("ok", ok_payload=ok_payload)
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        try:
            header, _raw, total_fetched = fetch_header(url)
            if "model.dummy.weight" not in header:
                failures.append("ok: fetched header missing expected tensor entry")
            if total_fetched > MAX_FETCH_BYTES:
                failures.append(f"ok: total_fetched {total_fetched} exceeds cap")
            if total_fetched != len(ok_payload):
                failures.append(
                    f"ok: total_fetched {total_fetched} != payload size {len(ok_payload)}"
                )
        except Exception as e:  # noqa: BLE001 - self-test must report, not raise
            failures.append(f"ok: expected success, got {e!r}")
    finally:
        _stop_scenario_server(server)

    return failures


def _run_drift_mutation_selftest() -> list[str]:
    failures: list[str] = []

    base_metadata = {
        "source_repo": REPO,
        "revision": REVISION,
        "source_url": SAFETENSORS_URL,
        "header_length_bytes": 263952,
        "header_sha256": "12740d6fe7a66b316040fa4d77471a8e1809498a71992b3364a6d5417d10662e",
        "total_bytes_fetched": 263960,
        "extraction_date": "2026-07-15T00:40:25Z",
    }
    base_tensors = {
        "model.language_model.layers.0.self_attn.k_proj.weight": {
            "dtype": "BF16",
            "shape": [256, 1536],
        },
        "model.audio_tower.subsample_conv_projection.layer0.conv.weight": {
            "dtype": "BF16",
            "shape": [128, 1, 3, 3],
        },
    }
    base_buckets = {"model.language_model": 1, "model.audio_tower": 1}
    base_manifest = {
        "metadata": dict(base_metadata),
        "bucket_counts": dict(base_buckets),
        "total_tensors": 2,
        "tensors": {k: dict(v) for k, v in base_tensors.items()},
    }
    base_fixture = json.loads(json.dumps(base_manifest))

    if diff_against_fixture(base_manifest, base_fixture):
        failures.append(
            "mutation-sanity: identical manifest/fixture reported spurious drift"
        )

    cases = [
        (
            "missing-bucket",
            lambda f: f["bucket_counts"].pop("model.audio_tower"),
            "bucket model.audio_tower",
        ),
        (
            "wrong-bucket-count",
            lambda f: f["bucket_counts"].__setitem__("model.language_model", 999),
            "bucket model.language_model",
        ),
        (
            "changed-revision",
            lambda f: f["metadata"].__setitem__("revision", "f" * 40),
            "metadata.revision",
        ),
        (
            "changed-header-sha",
            lambda f: f["metadata"].__setitem__("header_sha256", "b" * 64),
            "metadata.header_sha256",
        ),
        (
            "bucket-count-json-bool-not-int",
            lambda f: f["bucket_counts"].__setitem__("model.audio_tower", True),
            "bucket model.audio_tower",
        ),
    ]
    for label, mutator, expected_substring in cases:
        mutated_fixture = json.loads(json.dumps(base_fixture))
        mutator(mutated_fixture)
        errors = diff_against_fixture(base_manifest, mutated_fixture)
        if not errors:
            failures.append(f"{label}: expected drift error, got none")
        elif not any(expected_substring in e for e in errors):
            failures.append(
                f"{label}: no error mentioned {expected_substring!r}: {errors}"
            )

    return failures


def run_selftest() -> int:
    failures = _run_bounded_read_selftest() + _run_drift_mutation_selftest()
    if failures:
        print(f"FAIL: {len(failures)} self-test failure(s):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("OK: all offline self-tests passed.", file=sys.stderr)
    return 0


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
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the offline self-tests (bounded-range-read HTTP fixtures + "
        "drift-mutation cases) and exit. No network access; safe for CI.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return run_selftest()

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
