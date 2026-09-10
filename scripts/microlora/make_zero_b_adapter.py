#!/usr/bin/env python3
"""Build the zero-B control adapter: same file, every LoRA B matrix set to zero.

WHY THIS EXISTS. Loading an adapter and generating text tells you almost nothing on its own. If the
output matches the base model you cannot say whether the hook is correctly installed and the adapter
is simply weak, or whether the adapter was loaded and then silently ignored. Those two have opposite
consequences and identical symptoms. A zero-B adapter separates them: because a LoRA contribution is
`scale * B @ A @ x`, an adapter whose B is all zeros contributes EXACTLY zero at every adapted
projection, so generation under it MUST be byte-identical to generation with no adapter at all. Run
it as the middle arm of three (base / zero-B / trained) and the pair of comparisons is decisive.

WHY IT EDITS BYTES RATHER THAN TENSORS. The B ranges are overwritten with 0x00 in the raw data
section and nothing else in the file is touched. Every IEEE-754 float format the loader accepts —
f32, f16, bf16 — encodes +0.0 as all-zero bits, so this is correct without the script ever needing
to know the dtype, and it makes the "A tensors are untouched" claim provable by byte comparison
rather than by numeric tolerance. The header is copied verbatim, so shapes, dtypes, offsets and any
governance metadata survive exactly.

WHICH KEYS COUNT AS B. Mirrored from the loader at `crates/tune/src/lora/safetensors.rs:36-52`, which
is the authority: strip a trailing `.weight` if present, then the key is a B matrix when what remains
ends in `.lora_B` (PEFT) or `.lora_b` (MLX). Guessing this rule instead of copying it is how a script
like this quietly zeroes nothing.

Refuses rather than guesses: no B tensors found, a B range outside the data section, or any A tensor
that failed to survive byte-identical.

Usage:
    uv run python scripts/microlora/make_zero_b_adapter.py --in trained.safetensors --out zero_b.safetensors
    uv run python scripts/microlora/make_zero_b_adapter.py --self-test
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

HEADER_LEN_BYTES = 8


def is_b_key(key: str) -> bool:
    """True when `key` names a LoRA B matrix, by the loader's own rule."""
    k = key[: -len(".weight")] if key.endswith(".weight") else key
    return k.endswith(".lora_B") or k.endswith(".lora_b")


def is_a_key(key: str) -> bool:
    k = key[: -len(".weight")] if key.endswith(".weight") else key
    return k.endswith(".lora_A") or k.endswith(".lora_a")


def parse(buf: bytes) -> tuple[dict, int]:
    """Return (header dict, offset where the data section begins)."""
    if len(buf) < HEADER_LEN_BYTES:
        raise ValueError("file is shorter than the 8-byte header length prefix")
    (n,) = struct.unpack("<Q", buf[:HEADER_LEN_BYTES])
    end = HEADER_LEN_BYTES + n
    if end > len(buf):
        raise ValueError(f"header claims {n} bytes but the file holds {len(buf) - HEADER_LEN_BYTES}")
    return json.loads(buf[HEADER_LEN_BYTES:end].decode("utf-8")), end


def zero_b(buf: bytes) -> tuple[bytes, list[str], list[str]]:
    """Return (new bytes, B keys zeroed, A keys seen). Raises on anything unexpected."""
    header, data_start = parse(buf)
    out = bytearray(buf)
    b_keys, a_keys = [], []
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        if is_a_key(key):
            a_keys.append(key)
        if not is_b_key(key):
            continue
        lo, hi = meta["data_offsets"]
        s, e = data_start + lo, data_start + hi
        if not (data_start <= s <= e <= len(buf)):
            raise ValueError(f"tensor {key!r} has offsets [{lo}, {hi}] outside the data section")
        out[s:e] = b"\x00" * (e - s)
        b_keys.append(key)
    if not b_keys:
        raise ValueError(
            "no .lora_B / .lora_b tensor found, so this would emit an unmodified copy that reads "
            "as a passing control while testing nothing"
        )
    return bytes(out), sorted(b_keys), sorted(a_keys)


def verify(src: bytes, dst: bytes, b_keys: list[str]) -> None:
    """Assert the output is the input with exactly the B ranges zeroed, and nothing else."""
    if len(src) != len(dst):
        raise ValueError(f"length changed: {len(src)} -> {len(dst)}")
    h_src, start_src = parse(src)
    h_dst, start_dst = parse(dst)
    if h_src != h_dst or start_src != start_dst:
        raise ValueError("header changed; shapes, dtypes or metadata were not preserved")
    zeroed = set()
    for key in b_keys:
        lo, hi = h_dst[key]["data_offsets"]
        s, e = start_dst + lo, start_dst + hi
        if any(dst[s:e]):
            raise ValueError(f"{key} is not all zero in the output")
        zeroed.update(range(s, e))
    for key, meta in h_src.items():
        if key == "__metadata__" or key in b_keys:
            continue
        lo, hi = meta["data_offsets"]
        s, e = start_src + lo, start_src + hi
        if src[s:e] != dst[s:e]:
            raise ValueError(f"{key} was modified but only B tensors may change")
    # Bytes outside every tensor range must also be untouched.
    for i in range(start_src, len(src)):
        if i not in zeroed and src[i] != dst[i]:
            raise ValueError(f"byte {i} changed outside any B tensor range")


def _synth(with_b: bool = True) -> bytes:
    """A minimal well-formed safetensors file, for the self-test."""
    import array

    a = array.array("f", [1.5, -2.5, 3.0, 4.25])
    b = array.array("f", [9.0, -8.0])
    other = array.array("f", [7.5])
    blobs, header, off = [], {}, 0
    entries = [("model.layers.0.self_attn.q_proj.lora_A.weight", a, [2, 2])]
    if with_b:
        entries.append(("model.layers.0.self_attn.q_proj.lora_B.weight", b, [1, 2]))
    entries.append(("model.layers.0.mlp.something_else", other, [1]))
    for name, arr, shape in entries:
        raw = arr.tobytes()
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    header["__metadata__"] = {"format": "pt"}
    hj = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(hj)) + hj + b"".join(blobs)


def self_test() -> int:
    ok = True

    src = _synth(with_b=True)
    dst, b_keys, a_keys = zero_b(src)
    verify(src, dst, b_keys)
    assert len(b_keys) == 1 and len(a_keys) == 1, (b_keys, a_keys)
    assert src != dst, "the control did not change the file, so it cannot be a control"
    print(f"  ok  round trip: zeroed {b_keys}, preserved {a_keys}")

    # MUST-FAIL arm 1: a file with no B tensor is refused, not silently copied.
    try:
        zero_b(_synth(with_b=False))
        print("  FAIL  a B-less file was accepted; it would emit a control that tests nothing")
        ok = False
    except ValueError as e:
        print(f"  ok  refuses a B-less file: {e}")

    # MUST-FAIL arm 2: the verifier detects a B range left non-zero. Built by mutating the OUTPUT,
    # so the failure comes from the verifier and not from a second copy of the zeroing code.
    hdr, start = parse(dst)
    lo, _ = hdr[b_keys[0]]["data_offsets"]
    tampered = bytearray(dst)
    tampered[start + lo] = 0x01
    try:
        verify(src, bytes(tampered), b_keys)
        print("  FAIL  the verifier passed a B tensor that was not zero")
        ok = False
    except ValueError as e:
        print(f"  ok  verifier catches a non-zero B: {e}")

    # MUST-FAIL arm 3: the verifier detects a modified A tensor.
    lo_a, _ = hdr[a_keys[0]]["data_offsets"]
    tampered2 = bytearray(dst)
    tampered2[start + lo_a] ^= 0xFF
    try:
        verify(src, bytes(tampered2), b_keys)
        print("  FAIL  the verifier passed a modified A tensor")
        ok = False
    except ValueError as e:
        print(f"  ok  verifier catches a modified A: {e}")

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path)
    ap.add_argument("--out", dest="dst", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.src or not args.dst:
        ap.error("--in and --out are both required unless --self-test is given")

    raw = args.src.read_bytes()
    out, b_keys, a_keys = zero_b(raw)
    verify(raw, out, b_keys)
    args.dst.write_bytes(out)
    print(f"zeroed {len(b_keys)} B tensors, preserved {len(a_keys)} A tensors -> {args.dst}")
    for k in b_keys:
        print(f"  B  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
