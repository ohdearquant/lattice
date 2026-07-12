#!/usr/bin/env python3
"""Fake-quant PILOT harness for the projection-selective Q4 ablation.

Pre-registered protocol: METHODOLOGY ff140ca8, amended by
.khive/workspaces/20260712/proj-selective-quant/PILOT_AMENDMENT.md.
Recon: .khive/workspaces/20260712/proj-selective-quant/PLAN.md.

Mechanism (amendment, verbatim): "quantize each weight tensor with the
candidate scheme, dequantize back to f16, evaluate through the existing f16
path. Justified because the production `from_q4_dir` path already eagerly
dequants to f16-resident buffers, so fake-quant reproduces the numerics the
real container would produce without container/kernel changes."

This script does NOT reimplement the Rust Q4 quantizer as a fresh design —
it replicates crates/inference/src/weights/q4_weights.rs's
`quantize_block_with_mode_len` arithmetic exactly (same block chunking over
the flattened tensor, same min/max-derived affine scale/bias, same
abs_max/7 symmetric scale with bias=-8*scale, same f32::round-half-away-
from-zero nibble rounding order, same f16 round-to-nearest-even scale/bias
storage) — generalized only by making the group size a parameter (32 or
64), since no group-64 code path exists in the Rust source (PLAN.md §1).
The `self-test` subcommand is a MANDATORY gate: it re-derives one tensor's
Q4 blocks in this Python arithmetic and diffs the result byte-for-byte
against the real `quantize_q4`-produced `.q4` file already on disk, both at
the per-block scale/bias level and the fully dequantized-value level. No
arm may run before self-test passes (protocol requirement: "define the
exact quantizer bit-for-bit, not a reimplementation").

Usage:
    uv run python3 scripts/fake_quant_pilot.py self-test
    uv run python3 scripts/fake_quant_pilot.py quantize --arm A --output-dir /private/tmp/fq_pilot/arm_A
    uv run python3 scripts/fake_quant_pilot.py quantize --arm E --output-dir /private/tmp/fq_pilot/arm_E
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

# ---------------------------------------------------------------------------
# Defaults (this machine's local checkpoint layout)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = Path.home() / ".lattice/models/qwen3.5-0.8b"
DEFAULT_Q4_DIR = Path.home() / ".lattice/models/qwen3.5-0.8b-q4"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Arms per the registered protocol + pilot amendment (binding, do not alter
# the semantics here without a corresponding amendment to PILOT_AMENDMENT.md).
#   default = (mode, group_size) applied to every quantized tensor
#   overrides = {predicate_name: (mode, group_size)} — arm E only
ARM_DEFAULT_SCHEME = {
    "A": ("affine", 32),
    "B": ("affine", 64),
    "C": ("symmetric", 32),
    "D": ("symmetric", 64),
    "E": ("affine", 32),  # down_proj + attention/embeddings/lm_head/etc.
}
ARM_E_GATE_UP_SCHEME = ("affine", 64)


# ---------------------------------------------------------------------------
# f16 / bf16 <-> f32 conversion (numpy IEEE-754 round-to-nearest-even casts;
# verified bit-exact against the Rust half_bits module by self-test below)
# ---------------------------------------------------------------------------


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Widen bf16 bit patterns (uint16) to f32 — exact zero-extend."""
    widened = bits.astype(np.uint32) << 16
    return widened.view(np.float32)


def f16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return bits.view(np.float16).astype(np.float32)


def f32_to_f16_roundtrip(x: np.ndarray) -> np.ndarray:
    """f32 -> f16 (round-to-nearest-even) -> f32, as the Rust storage path does."""
    return x.astype(np.float16).astype(np.float32)


def f32_to_f16_bits(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).view(np.uint16)


def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    """Match Rust's `f32::round` (ties away from zero, NOT numpy's default
    round-half-to-even). x and the return value are float32.

    NOT implemented as `floor(x + 0.5)` — that classic trick has a real
    precision bug: for x just below a tie boundary (e.g. x=0.49999997),
    the addition `x + 0.5` itself can round UP to the next representable
    f32 (1.0), so `floor` then returns 1 instead of the correct 0. This
    was caught empirically by this script's own self-test (2 mismatched
    elements out of 254,279,680 in `embed_tokens.weight`, both exactly
    this failure mode). `np.round` is a correctly-rounded nearest-integer
    primitive (no intermediate-precision pitfall); only its tie-breaking
    rule (round-half-to-even) differs from Rust's (round-half-away-from-
    zero), so only exact ties need a fixup.
    """
    x = np.asarray(x, dtype=np.float32)
    nearest = np.round(x).astype(np.float32)  # correctly rounded; ties-to-even
    trunc = np.trunc(x)
    frac_abs = np.abs(x - trunc)
    is_tie = frac_abs == np.float32(0.5)
    away_from_zero = (trunc + np.sign(x)).astype(np.float32)
    return np.where(is_tie, away_from_zero, nearest).astype(np.float32)


# ---------------------------------------------------------------------------
# The quantizer: bit-for-bit port of quantize_block_with_mode_len
# (q4_weights.rs:159-217) generalized over group_size.
# ---------------------------------------------------------------------------


def fake_quant_dequant(
    flat: np.ndarray, group_size: int, symmetric: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize+dequantize a flattened f32 tensor in `group_size`-element
    blocks (contiguous over the FLATTENED array — NOT per matrix row; this
    matches `quantize_f32_to_q4`'s `data.chunks(group_size)` over the whole
    tensor, the function `quantize_q4.rs`'s single call site actually uses).

    Returns (dequantized_flat_f32, scale_f16_as_f32_per_block, bias_f16_as_f32_per_block).
    """
    flat = flat.astype(np.float32, copy=False)
    n = flat.shape[0]
    n_blocks = -(-n // group_size)
    pad = n_blocks * group_size - n
    padded = np.concatenate([flat, np.zeros(pad, dtype=np.float32)]) if pad else flat
    blocks = padded.reshape(n_blocks, group_size)

    if symmetric:
        # Rust: abs_max over the FULL 32-wide vals array (padding included);
        # zero padding never raises abs_max, so this is safe unconditionally.
        abs_max = np.max(np.abs(blocks), axis=1).astype(np.float32)
        scale = np.where(abs_max == 0, np.float32(1.0), abs_max / np.float32(7.0)).astype(
            np.float32
        )
        bias = (np.float32(-8.0) * scale).astype(np.float32)
    else:
        # Rust: min/max over `real = &vals[..valid_len]` — the last (possibly
        # padded) block excludes zero padding from its min/max.
        min_full = blocks.min(axis=1).astype(np.float32)
        max_full = blocks.max(axis=1).astype(np.float32)
        if pad:
            last_valid = blocks[-1, : group_size - pad]
            min_full[-1] = last_valid.min()
            max_full[-1] = last_valid.max()
        rng = (max_full - min_full).astype(np.float32)
        scale = np.where(rng == 0, np.float32(1.0), rng / np.float32(15.0)).astype(np.float32)
        bias = min_full

    inv_scale = (np.float32(1.0) / scale).astype(np.float32)

    if symmetric:
        q = round_half_away_from_zero(blocks * inv_scale[:, None]) + np.float32(8.0)
    else:
        q = round_half_away_from_zero((blocks - bias[:, None]) * inv_scale[:, None])
    q = np.clip(q, 0.0, 15.0).astype(np.float32)

    scale_f16 = f32_to_f16_roundtrip(scale)
    bias_f16 = f32_to_f16_roundtrip(bias)

    dequant_blocks = (q * scale_f16[:, None] + bias_f16[:, None]).astype(np.float32)
    dequant = dequant_blocks.reshape(-1)[:n]
    return dequant, scale_f16, bias_f16


# ---------------------------------------------------------------------------
# Source safetensors reading (raw header parse — bf16 has no numpy dtype)
# ---------------------------------------------------------------------------


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def load_tensor_f32(path: Path, header: dict, data_start: int, name: str) -> np.ndarray:
    """Read one tensor from a single-file safetensors, widened to f32."""
    meta = header[name]
    dtype = meta["dtype"]
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + start)
        raw = f.read(end - start)
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype=np.uint16)
        return bf16_bits_to_f32(bits).copy()
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).copy()
    raise ValueError(f"unsupported source dtype {dtype} for tensor {name}")


# ---------------------------------------------------------------------------
# should_quantize ground truth — sourced from the REAL quantize_q4 output's
# quantize_index.json (produced by the actual Rust binary on this exact
# checkpoint) rather than re-deriving quantize_q4.rs::should_quantize by
# hand. This guarantees arm A-fake's target tensor set is IDENTICAL to the
# Stage-0 real-container baseline's set, which is required for the
# A-fake-vs-Stage-0 agreement check to mean anything.
# ---------------------------------------------------------------------------


def load_quantized_tensor_set(q4_dir: Path) -> set[str]:
    idx = json.load(open(q4_dir / "quantize_index.json"))
    return {e["name"] for e in idx if e["quantized"]}


def tensor_scheme_for_arm(name: str, arm: str) -> tuple[str, int]:
    if arm != "E":
        return ARM_DEFAULT_SCHEME[arm]
    if name.endswith("gate_proj.weight") or name.endswith("up_proj.weight"):
        return ARM_E_GATE_UP_SCHEME
    # down_proj (explicit per the amendment) + everything else the real
    # pipeline quantizes (attention/linear_attn projections, embed_tokens,
    # visual, mtp mlp) stays at the control scheme.
    return ARM_DEFAULT_SCHEME["E"]


# ---------------------------------------------------------------------------
# Self-test (MANDATORY gate)
# ---------------------------------------------------------------------------


def cmd_self_test(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    q4_dir = Path(args.q4_dir)
    st_path = model_dir / "model.safetensors"
    header, data_start = read_safetensors_header(st_path)

    idx = json.load(open(q4_dir / "quantize_index.json"))
    by_name = {e["name"]: e for e in idx}
    entry = by_name.get(args.tensor)
    if entry is None or not entry["quantized"]:
        print(f"FAIL: {args.tensor} is not a quantized tensor in {q4_dir}")
        return 1

    q4_path = q4_dir / entry["file"]
    with open(q4_path, "rb") as f:
        data = f.read()
    magic = data[0:4]
    if magic != b"KHQ4":
        print(f"FAIL: {q4_path} bad magic {magic!r}")
        return 1
    version = struct.unpack("<I", data[4:8])[0]
    ndim = struct.unpack("<I", data[8:12])[0]
    off = 12
    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack("<Q", data[off : off + 8])[0])
        off += 8
    original_len = struct.unpack("<Q", data[off : off + 8])[0]
    off += 8
    block_bytes = data[off:]
    n_blocks = len(block_bytes) // 20
    if n_blocks * 20 != len(block_bytes):
        print(f"FAIL: {q4_path} block payload not a multiple of 20 bytes")
        return 1

    # Parse real Rust-produced blocks: scale u16, bias u16, packed[16].
    blk = np.frombuffer(block_bytes, dtype=np.uint8).reshape(n_blocks, 20)
    real_scale_bits = blk[:, 0:2].copy().view(np.uint16).reshape(-1)
    real_bias_bits = blk[:, 2:4].copy().view(np.uint16).reshape(-1)
    real_scale_f32 = f16_bits_to_f32(real_scale_bits)
    real_bias_f32 = f16_bits_to_f32(real_bias_bits)
    packed = blk[:, 4:20]
    lo = (packed & 0x0F).astype(np.float32)
    hi = (packed >> 4).astype(np.float32)
    # sequential-pairs layout: weight[2b] = lo, weight[2b+1] = hi
    interleaved = np.empty((n_blocks, 32), dtype=np.float32)
    interleaved[:, 0::2] = lo
    interleaved[:, 1::2] = hi
    real_dequant_blocks = interleaved * real_scale_f32[:, None] + real_bias_f32[:, None]
    real_dequant = real_dequant_blocks.reshape(-1)[:original_len]

    print(f"=== fake_quant_pilot self-test: {args.tensor} ===")
    print(f"  real .q4 file:   {q4_path}")
    print(f"  version={version} shape={shape} original_len={original_len} n_blocks={n_blocks}")

    # Python arm-A path (g32 affine) on the same source tensor.
    src_f32 = load_tensor_f32(st_path, header, data_start, args.tensor)
    assert src_f32.shape[0] == original_len, (
        f"source numel {src_f32.shape[0]} != q4 original_len {original_len}"
    )
    py_dequant, py_scale_f16, py_bias_f16 = fake_quant_dequant(
        src_f32, group_size=32, symmetric=False
    )
    py_n_blocks = py_scale_f16.shape[0]

    ok = True
    if py_n_blocks != n_blocks:
        print(f"  FAIL: block count mismatch: python={py_n_blocks} rust={n_blocks}")
        ok = False
    else:
        py_scale_bits = f32_to_f16_bits(py_scale_f16)
        py_bias_bits = f32_to_f16_bits(py_bias_f16)
        scale_match = np.array_equal(py_scale_bits, real_scale_bits)
        bias_match = np.array_equal(py_bias_bits, real_bias_bits)
        print(f"  per-block scale bits exact match: {scale_match} ({n_blocks} blocks)")
        print(f"  per-block bias  bits exact match: {bias_match} ({n_blocks} blocks)")
        if not scale_match:
            bad = np.where(py_scale_bits != real_scale_bits)[0][:5]
            print(f"    first mismatches (block idx): {bad.tolist()}")
            ok = False
        if not bias_match:
            bad = np.where(py_bias_bits != real_bias_bits)[0][:5]
            print(f"    first mismatches (block idx): {bad.tolist()}")
            ok = False

        dequant_exact = np.array_equal(py_dequant, real_dequant)
        max_abs_diff = float(np.max(np.abs(py_dequant - real_dequant))) if not dequant_exact else 0.0
        print(f"  dequantized value exact match: {dequant_exact} (max_abs_diff={max_abs_diff:.3e})")
        if not dequant_exact:
            ok = False

    if ok:
        print("SELF-TEST: PASS — python g32-affine quantizer is bit-exact vs the real Rust .q4 output.")
        return 0
    print("SELF-TEST: FAIL — python quantizer diverges from the real Rust quantizer. DO NOT run arms.")
    return 1


# ---------------------------------------------------------------------------
# quantize: produce one arm's derived f16 checkpoint dir
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def cmd_quantize(args: argparse.Namespace) -> int:
    arm = args.arm
    model_dir = Path(args.model_dir)
    q4_dir = Path(args.q4_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    st_path = model_dir / "model.safetensors"
    header, data_start = read_safetensors_header(st_path)
    tensor_names = sorted(k for k in header if k != "__metadata__")

    quantized_set = load_quantized_tensor_set(q4_dir) if arm != "full_precision" else set()

    scheme_map: dict[str, str] = {}
    out_tensors: dict[str, np.ndarray] = {}

    t0 = time.time()
    n_quantized = 0
    for name in tensor_names:
        src = load_tensor_f32(st_path, header, data_start, name)
        if arm != "full_precision" and name in quantized_set:
            mode, gs = tensor_scheme_for_arm(name, arm)
            dequant, _, _ = fake_quant_dequant(
                src, group_size=gs, symmetric=(mode == "symmetric")
            )
            scheme_map[name] = f"{mode}-g{gs}"
            n_quantized += 1
            arr_f32 = dequant
        else:
            scheme_map[name] = "f16-passthrough"
            arr_f32 = src
        out_tensors[name] = arr_f32.reshape(header[name]["shape"]).astype(np.float16)
    elapsed = time.time() - t0

    out_st_path = output_dir / "model.safetensors"
    save_file(out_tensors, str(out_st_path))

    # config.json + tokenizer.json: config copied (small), tokenizer symlinked
    # (12MB, identical across every arm — no reason to duplicate 5x).
    shutil.copy2(model_dir / "config.json", output_dir / "config.json")
    tok_link = output_dir / "tokenizer.json"
    if tok_link.exists() or tok_link.is_symlink():
        tok_link.unlink()
    tok_link.symlink_to(model_dir / "tokenizer.json")

    corpus_path = REPO_ROOT / "docs/bench_results/wiki.test.raw"
    manifest = {
        "arm": arm,
        "source_model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "quantize_index_source": str(q4_dir / "quantize_index.json"),
        "n_tensors_total": len(tensor_names),
        "n_tensors_quantized": n_quantized,
        "scheme_map": scheme_map,
        "checkpoint_sha256": sha256_file(out_st_path),
        "corpus_file": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path) if corpus_path.exists() else None,
        "commit_sha": git_commit_sha(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quantize_wall_seconds": elapsed,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"[{arm}] wrote {out_st_path} ({out_st_path.stat().st_size / 1e9:.2f} GB), "
        f"{n_quantized}/{len(tensor_names)} tensors fake-quantized, {elapsed:.1f}s"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("self-test", help="MANDATORY bit-exactness gate vs the real Rust quantizer")
    st.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    st.add_argument("--q4-dir", default=str(DEFAULT_Q4_DIR))
    st.add_argument(
        "--tensor",
        default="model.language_model.layers.0.mlp.down_proj.weight",
        help="tensor name to self-test (must be quantized=true in quantize_index.json)",
    )
    st.set_defaults(func=cmd_self_test)

    q = sub.add_parser("quantize", help="produce one arm's derived fake-quant f16 checkpoint dir")
    q.add_argument("--arm", required=True, choices=["A", "B", "C", "D", "E", "full_precision"])
    q.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    q.add_argument("--q4-dir", default=str(DEFAULT_Q4_DIR))
    q.add_argument("--output-dir", required=True)
    q.set_defaults(func=cmd_quantize)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
