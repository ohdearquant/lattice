#!/usr/bin/env python3
"""Fake-quant PILOT harness for the projection-selective Q4 ablation.

This is a pre-registered pilot: kill/keep bars were fixed before any arm
ran, and dense-model results are directional, not confirmatory, for MoE
targets (routing sensitivity is MoE-specific and stays unmeasured here).

Mechanism: quantize each weight tensor with the candidate scheme,
dequantize back to f16, evaluate through the existing f16 path. Justified
because the production `from_q4_dir` path already eagerly dequants to
f16-resident buffers, so fake-quant reproduces the numerics the real
container would produce without container/kernel changes.

This script does NOT reimplement the Rust Q4 quantizer as a fresh design —
it replicates crates/inference/src/weights/q4_weights.rs's
`quantize_block_with_mode_len` arithmetic exactly (same block chunking over
the flattened tensor, same min/max-derived affine scale/bias, same
abs_max/7 symmetric scale with bias=-8*scale, same f32::round-half-away-
from-zero nibble rounding order, same f16 round-to-nearest-even scale/bias
storage) — generalized only by making the group size a parameter (32 or
64), since no group-64 code path exists in the Rust source.

Invariants enforced by this script:

- The `self-test` subcommand is a MANDATORY gate: it re-derives Q4 blocks
  in this Python arithmetic and diffs the result byte-for-byte (scale,
  bias, and the full 16-byte packed-nibble payload) against the real
  `quantize_q4`-produced `.q4` file already on disk. `--all` sweeps every
  tensor marked `quantized: true` in `quantize_index.json`, reproducing
  the full-checkpoint coverage claim from the CLI.
- `quantize` and `run-arm` both run the full self-test sweep before
  writing any arm checkpoint and refuse to proceed on failure; the sweep
  result is recorded in the arm's manifest.
- Non-finite (NaN/Inf) source elements are rejected before padding or
  reduction, matching the Rust quantizer's fail-closed behavior; `--self-
  check` proves the guard is wired.
- `run-arm` is the complete write -> eval -> delete loop: it writes the
  arm checkpoint, evaluates perplexity under the machine-wide Metal GPU
  advisory lock, records the parsed result in the manifest, then deletes
  the arm's weights (keeping only the manifest and console log).

Usage:
    uv run python3 scripts/fake_quant_pilot.py self-test --all
    uv run python3 scripts/fake_quant_pilot.py self-check
    uv run python3 scripts/fake_quant_pilot.py quantize --arm A \\
        --output-dir /private/tmp/fq_pilot/arm_A
    uv run python3 scripts/fake_quant_pilot.py run-arm --arm E
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import stat as stat_mod
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from safetensors.numpy import save as safetensors_save

# ---------------------------------------------------------------------------
# Defaults (this machine's local checkpoint layout)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = Path.home() / ".lattice/models/qwen3.5-0.8b"
DEFAULT_Q4_DIR = Path.home() / ".lattice/models/qwen3.5-0.8b-q4"
REPO_ROOT = Path(__file__).resolve().parent.parent

# The real Rust quantizer (crates/inference/src/bin/quantize_q4.rs) only
# ever produces group-32 affine blocks; that is what self-test diffs
# against. Group-64 and symmetric are Python-only candidate schemes with no
# on-disk Rust counterpart to bit-compare against.
GROUP_SIZE_REAL = 32

# Dedicated scratch root this script owns for `run-arm` output. Weights
# written here are deleted after a successful eval; only manifest.json and
# the console log persist.
SCRATCH_ROOT = Path("/private/tmp/fq_pilot")

# Machine-wide Metal GPU advisory lock (fleet convention). Any process
# driving the GPU for measurements serializes through this flock.
GPU_LOCK_PATH = "/tmp/lion-metal-gpu-test.lock"
GPU_LOCK_TIMEOUT_S = 30 * 60

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
    flat: np.ndarray, group_size: int, symmetric: bool, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quantize+dequantize a flattened f32 tensor in `group_size`-element
    blocks (contiguous over the FLATTENED array — NOT per matrix row; this
    matches `quantize_f32_to_q4`'s `data.chunks(group_size)` over the whole
    tensor, the function `quantize_q4.rs`'s single call site actually uses).

    Rejects any non-finite element before padding/reduction, mirroring
    `quantize_block_with_mode_len`'s finite check (q4_weights.rs:170-177):
    a NaN silently leaves abs_max/min_val/max_val unchanged there too, so
    Rust rejects up front rather than propagate a wrong-but-no-error block.

    Returns (dequantized_flat_f32, scale_f16_as_f32_per_block,
    bias_f16_as_f32_per_block, quantized_nibble_codes[n_blocks, group_size]
    as uint8 in 0..15).
    """
    flat = flat.astype(np.float32, copy=False)
    non_finite = np.flatnonzero(~np.isfinite(flat))
    if non_finite.size:
        bad_idx = int(non_finite[0])
        raise ValueError(
            f"tensor {name!r} element {bad_idx} contains a non-finite value "
            f"({flat[bad_idx]!r}); source weights must be finite"
        )

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
    return dequant, scale_f16, bias_f16, q.astype(np.uint8)


def pack_q4_nibbles(q: np.ndarray) -> np.ndarray:
    """Pack per-block quantized nibble codes (n_blocks, group_size) uint8
    0..15 into the on-disk byte layout used by `quantize_block_with_mode_len`
    (q4_weights.rs:187-192): `packed[b] = (q1 << 4) | q0`, where
    `weight[2b] = q0` (low nibble) and `weight[2b+1] = q1` (high nibble).
    """
    q = q.astype(np.uint8)
    q0 = q[:, 0::2]
    q1 = q[:, 1::2]
    return ((q1 << 4) | (q0 & 0x0F)).astype(np.uint8)


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


def load_quantize_index(q4_dir: Path) -> list[dict]:
    with open(q4_dir / "quantize_index.json") as f:
        return json.load(f)


def load_quantized_tensor_set(q4_dir: Path) -> set[str]:
    return {e["name"] for e in load_quantize_index(q4_dir) if e["quantized"]}


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


def read_real_q4_blocks(q4_path: Path) -> dict:
    """Parse a Rust-produced `.q4` file: header (magic/version/shape/
    original_len) plus the 20-byte-per-block payload (2B scale, 2B bias,
    16B packed nibbles)."""
    with open(q4_path, "rb") as f:
        data = f.read()
    magic = data[0:4]
    if magic != b"KHQ4":
        raise ValueError(f"{q4_path}: bad magic {magic!r}")
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
        raise ValueError(f"{q4_path}: block payload not a multiple of 20 bytes")
    blk = np.frombuffer(block_bytes, dtype=np.uint8).reshape(n_blocks, 20)
    scale_bits = blk[:, 0:2].copy().view(np.uint16).reshape(-1)
    bias_bits = blk[:, 2:4].copy().view(np.uint16).reshape(-1)
    packed = blk[:, 4:20].copy()
    return {
        "version": version,
        "shape": shape,
        "original_len": original_len,
        "n_blocks": n_blocks,
        "scale_bits": scale_bits,
        "bias_bits": bias_bits,
        "packed": packed,
    }


def verify_tensor_bit_exact(
    tensor: str,
    st_path: Path,
    header: dict,
    data_start: int,
    q4_dir: Path,
    entry: dict,
) -> tuple[bool, str]:
    """Diff the Python g32-affine port against the real Rust `.q4` output
    for one tensor: per-block scale bits, bias bits, the complete 16-byte
    packed-nibble payload (all 20 bytes/block together), and the fully
    dequantized values recomputed independently from the on-disk bytes."""
    q4_path = q4_dir / entry["file"]
    real = read_real_q4_blocks(q4_path)

    src_f32 = load_tensor_f32(st_path, header, data_start, tensor)
    if src_f32.shape[0] != real["original_len"]:
        return False, (
            f"{tensor}: FAIL source numel {src_f32.shape[0]} != "
            f"q4 original_len {real['original_len']}"
        )

    py_dequant, py_scale_f16, py_bias_f16, py_q = fake_quant_dequant(
        src_f32, group_size=GROUP_SIZE_REAL, symmetric=False, name=tensor
    )
    if py_scale_f16.shape[0] != real["n_blocks"]:
        return False, (
            f"{tensor}: FAIL block count mismatch: "
            f"python={py_scale_f16.shape[0]} rust={real['n_blocks']}"
        )

    py_scale_bits = f32_to_f16_bits(py_scale_f16)
    py_bias_bits = f32_to_f16_bits(py_bias_f16)
    py_packed = pack_q4_nibbles(py_q)

    scale_match = np.array_equal(py_scale_bits, real["scale_bits"])
    bias_match = np.array_equal(py_bias_bits, real["bias_bits"])
    packed_match = np.array_equal(py_packed, real["packed"])

    # Independent end-to-end check: dequantize straight from the REAL
    # on-disk bytes and compare to the Python dequant, rather than only
    # comparing scale/bias/packed equality pairwise.
    lo = (real["packed"] & 0x0F).astype(np.float32)
    hi = (real["packed"] >> 4).astype(np.float32)
    interleaved = np.empty((real["n_blocks"], GROUP_SIZE_REAL), dtype=np.float32)
    interleaved[:, 0::2] = lo
    interleaved[:, 1::2] = hi
    real_scale_f32 = f16_bits_to_f32(real["scale_bits"])
    real_bias_f32 = f16_bits_to_f32(real["bias_bits"])
    real_dequant = (interleaved * real_scale_f32[:, None] + real_bias_f32[:, None]).reshape(-1)[
        : real["original_len"]
    ]
    dequant_match = np.array_equal(py_dequant, real_dequant)

    ok = scale_match and bias_match and packed_match and dequant_match
    if ok:
        return True, f"{tensor}: PASS ({real['n_blocks']} blocks, byte-exact)"
    return False, (
        f"{tensor}: FAIL scale={scale_match} bias={bias_match} "
        f"packed={packed_match} dequant={dequant_match}"
    )


def run_self_test_sweep(model_dir: Path, q4_dir: Path, tensor: str | None = None) -> dict:
    """Run the bit-exactness self-test. `tensor=None` sweeps every tensor
    marked `quantized: true` in `quantize_index.json` (the full
    reproducibility mode this script's docstring claims); `tensor=<name>`
    checks just that one tensor (fast smoke-check)."""
    st_path = model_dir / "model.safetensors"
    header, data_start = read_safetensors_header(st_path)
    idx = load_quantize_index(q4_dir)
    by_name = {e["name"]: e for e in idx}
    targets = [e["name"] for e in idx if e["quantized"]] if tensor is None else [tensor]

    n_pass = 0
    failures: list[str] = []
    for name in targets:
        entry = by_name.get(name)
        if entry is None or not entry["quantized"]:
            print(f"  {name}: FAIL — not a quantized tensor in {q4_dir}")
            failures.append(name)
            continue
        ok, detail = verify_tensor_bit_exact(name, st_path, header, data_start, q4_dir, entry)
        print(f"  {detail}")
        if ok:
            n_pass += 1
        else:
            failures.append(name)

    return {
        "ok": len(failures) == 0,
        "mode": "all" if tensor is None else "single",
        "n_tensors_checked": len(targets),
        "n_pass": n_pass,
        "n_fail": len(failures),
        "failures": failures,
    }


def cmd_self_test(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    q4_dir = Path(args.q4_dir)
    scope = "all quantized tensors" if args.all else args.tensor
    print(f"=== fake_quant_pilot self-test: {scope} ===")
    result = run_self_test_sweep(model_dir, q4_dir, tensor=None if args.all else args.tensor)
    print(f"{result['n_pass']}/{result['n_tensors_checked']} tensors bit-exact")
    if result["ok"]:
        print(
            "SELF-TEST: PASS — python g32-affine quantizer is bit-exact vs the "
            "real Rust .q4 output."
        )
        return 0
    print(f"SELF-TEST: FAIL — {result['n_fail']} tensor(s) diverge: {result['failures'][:10]}")
    print("DO NOT run arms.")
    return 1


def require_self_test_pass(model_dir: Path, q4_dir: Path) -> dict:
    """Mandatory gate for `quantize` and `run-arm`: run the full self-test
    sweep over every quantized tensor and refuse to proceed if any tensor
    diverges from the real Rust `.q4` output. Returns the sweep result,
    which callers record into the arm manifest."""
    print("=== mandatory self-test gate: sweeping every quantized tensor ===")
    result = run_self_test_sweep(model_dir, q4_dir, tensor=None)
    print(f"self-test gate: {result['n_pass']}/{result['n_tensors_checked']} tensors bit-exact")
    if not result["ok"]:
        raise SystemExit(
            f"SELF-TEST GATE FAILED — {result['n_fail']} tensor(s) diverge from the real "
            f"Rust .q4 output: {result['failures'][:10]}. Refusing to write an arm checkpoint."
        )
    return result


def resolve_scratch_output_dir(
    arm: str, output_dir_arg: str | None, scratch_root: Path | None = None
) -> Path:
    """Resolve run-arm's output directory, confined to the scratch root.

    run-arm deletes `model.safetensors` from its output directory after a
    successful eval, so the directory must live inside the scratch tree this
    script owns: an arbitrary --output-dir (or a symlink escaping the tree)
    would let the post-eval cleanup unlink a file anywhere on disk."""
    scratch_root = scratch_root if scratch_root is not None else SCRATCH_ROOT
    root = scratch_root.resolve()
    candidate = Path(output_dir_arg) if output_dir_arg else scratch_root / f"arm_{arm}"
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise SystemExit(
            f"run-arm --output-dir must resolve inside {root} (got {resolved}); "
            "the post-eval cleanup deletes model.safetensors from this directory"
        )
    return resolved


def _require_secure_dir(p: Path) -> None:
    """Require `p` to be a real directory (not a symlink), owned by the
    current user, and not group/world-writable — the preconditions for
    trusting later writes and deletions under it. lstat (no follow) so a
    symlink planted at `p` is detected rather than traversed."""
    st = os.lstat(p)
    if stat_mod.S_ISLNK(st.st_mode):
        raise SystemExit(f"scratch dir {p} is a symlink; refusing to use it")
    if not stat_mod.S_ISDIR(st.st_mode):
        raise SystemExit(f"scratch dir {p} is not a directory; refusing to use it")
    if st.st_uid != os.geteuid():
        raise SystemExit(f"scratch dir {p} is not owned by the current user; refusing to use it")
    if st.st_mode & 0o022:
        raise SystemExit(f"scratch dir {p} is group/world-writable; refusing to use it")


def provision_scratch_dir(
    arm: str, output_dir_arg: str | None, scratch_root: Path | None = None
) -> tuple[Path, int]:
    """Resolve, create, and validate the arm's output directory, returning it
    with an O_NOFOLLOW directory handle.

    The pathname containment check alone is a time-of-check/time-of-use
    hazard: between the check and the later write/eval/delete sequence, a
    component under a writable scratch root could be swapped for a symlink
    pointing outside the tree. Provisioning therefore (1) validates the root
    and the arm directory via lstat — real dir, current-user-owned, not
    group/world-writable, (2) creates the arm directory BEFORE any
    long-running step, and (3) opens and returns an O_NOFOLLOW|O_DIRECTORY
    handle so cleanup can unlink via dir_fd and writers can revalidate the
    handle still names the checked directory."""
    scratch_root = scratch_root if scratch_root is not None else SCRATCH_ROOT
    resolved = resolve_scratch_output_dir(arm, output_dir_arg, scratch_root)
    os.makedirs(scratch_root, mode=0o700, exist_ok=True)
    _require_secure_dir(scratch_root)
    os.makedirs(resolved, mode=0o700, exist_ok=True)
    _require_secure_dir(resolved)
    dfd = os.open(str(resolved), os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_DIRECTORY", 0))
    return resolved, dfd


def _revalidate_dir_handle(dfd: int, path: Path) -> None:
    """Abort if `path` no longer names the directory `dfd` was opened on
    (i.e. it was swapped since provisioning)."""
    held = os.fstat(dfd)
    now = os.stat(path)
    if (held.st_dev, held.st_ino) != (now.st_dev, now.st_ino):
        raise SystemExit(
            f"scratch dir {path} changed identity since provisioning; aborting before writing"
        )


def cmd_self_check(_args: argparse.Namespace) -> int:
    """Fail-closed guard checks: minimal, self-contained assertions (no
    pytest dependency) proving (1) `fake_quant_dequant` rejects NaN/Inf
    input with an error naming the tensor and element index, and (2)
    `resolve_scratch_output_dir` rejects directories outside SCRATCH_ROOT."""
    flat = np.zeros(32, dtype=np.float32)
    flat[5] = float("nan")
    try:
        fake_quant_dequant(flat, group_size=32, symmetric=False, name="self_check_tensor")
    except ValueError as e:
        msg = str(e)
        if "self_check_tensor" in msg and "element 5" in msg:
            print(f"SELF-CHECK: PASS — non-finite input raised with tensor+index detail: {e}")
        else:
            print(f"SELF-CHECK: FAIL — raised but message missing tensor/index detail: {e}")
            return 1
    else:
        print(
            "SELF-CHECK: FAIL — non-finite input did not raise; the fail-closed guard is not wired."
        )
        return 1

    inside = resolve_scratch_output_dir("A", None)
    if inside != (SCRATCH_ROOT / "arm_A").resolve():
        print(f"SELF-CHECK: FAIL — default output dir resolved unexpectedly: {inside}")
        return 1
    for outside in ("/tmp/fq-outside-scratch", str(SCRATCH_ROOT) + "-sibling", "/etc"):
        try:
            resolve_scratch_output_dir("A", outside)
        except SystemExit:
            continue
        print(f"SELF-CHECK: FAIL — output dir outside scratch root was accepted: {outside}")
        return 1
    print(f"SELF-CHECK: PASS — output-dir boundary confined to {SCRATCH_ROOT}.")

    # Provisioning guards: the pathname check alone is a TOCTOU hazard, so
    # provision_scratch_dir must reject symlinked roots/arm dirs and hand back
    # a directory handle whose identity revalidation detects a swap.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        outside_dir = base / "outside"
        outside_dir.mkdir()

        # root itself a symlink to elsewhere -> rejected
        sym_root = base / "sym_root"
        sym_root.symlink_to(outside_dir)
        try:
            provision_scratch_dir("A", None, scratch_root=sym_root)
        except SystemExit:
            pass
        else:
            print("SELF-CHECK: FAIL — symlinked scratch root was accepted.")
            return 1

        # arm dir pre-planted as a symlink escaping the root -> rejected
        real_root = base / "real_root"
        real_root.mkdir(mode=0o700)
        (real_root / "arm_A").symlink_to(outside_dir)
        try:
            provision_scratch_dir("A", None, scratch_root=real_root)
        except SystemExit:
            pass
        else:
            print("SELF-CHECK: FAIL — symlinked arm dir was accepted.")
            return 1

        # happy path: handle held, unlink-via-dfd works, swap is detected
        (real_root / "arm_A").unlink()
        arm_dir, dfd = provision_scratch_dir("A", None, scratch_root=real_root)
        (arm_dir / "model.safetensors").write_bytes(b"x")
        _revalidate_dir_handle(dfd, arm_dir)
        os.unlink("model.safetensors", dir_fd=dfd)
        if (arm_dir / "model.safetensors").exists():
            print("SELF-CHECK: FAIL — dir_fd unlink did not remove the file.")
            return 1
        # simulate the race: swap the checked directory, expect revalidation to abort
        swapped = base / "swapped"
        swapped.mkdir(mode=0o700)
        arm_dir.rmdir()
        swapped.rename(arm_dir)
        try:
            _revalidate_dir_handle(dfd, arm_dir)
        except SystemExit:
            pass
        else:
            print("SELF-CHECK: FAIL — directory swap was not detected by revalidation.")
            os.close(dfd)
            return 1
        os.close(dfd)

        # fd-anchored write immunity: swap the checked path immediately after
        # revalidation (the worst-case race window) and prove the write still
        # lands in the held directory inode, never through the swapped path.
        (real_root / "arm_A").rmdir()
        arm_dir, dfd = provision_scratch_dir("A", None, scratch_root=real_root)
        _revalidate_dir_handle(dfd, arm_dir)
        moved = base / "arm_A_moved"
        arm_dir.rename(moved)  # swap: path now free...
        (base / "outside_target").mkdir(mode=0o700)
        arm_dir.symlink_to(base / "outside_target")  # ...and points outside
        _write_bytes_in_dir(dfd, "probe.bin", b"inside")
        os.close(dfd)
        if not (moved / "probe.bin").exists():
            print("SELF-CHECK: FAIL — fd-anchored write did not land in the held directory.")
            return 1
        if (base / "outside_target" / "probe.bin").exists():
            print("SELF-CHECK: FAIL — fd-anchored write escaped through the swapped path.")
            return 1
    print(
        "SELF-CHECK: PASS — scratch provisioning rejects symlinks, detects dir swaps, "
        "and fd-anchored writes are swap-immune."
    )
    return 0


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


def _write_bytes_in_dir(dfd: int, name: str, data: bytes) -> None:
    """Write `data` to `name` relative to the held directory handle. Anchoring
    every mutation to `dfd` (openat semantics) makes the write immune to the
    checked directory's PATH being swapped after validation — the handle pins
    the directory inode itself. O_NOFOLLOW refuses a symlink planted at the
    final component."""
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600, dir_fd=dfd)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def write_arm_checkpoint(
    arm: str,
    model_dir: Path,
    q4_dir: Path,
    output_dir: Path,
    dfd: int,
    self_test_result: dict,
) -> dict:
    """Write one arm's fake-quantized f16 checkpoint + manifest.json into
    `output_dir`, with every filesystem mutation anchored to the held
    directory handle `dfd` — never bare pathnames. The tensor-build phase is
    long, and a path-based write after it would reopen the swap window that
    provisioning closed; openat-relative writes are pinned to the directory
    inode regardless of what the path names by then. Callers must have
    already run `require_self_test_pass` and pass its result through so it
    lands in the manifest."""
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
            dequant, _, _, _ = fake_quant_dequant(
                src, group_size=gs, symmetric=(mode == "symmetric"), name=name
            )
            scheme_map[name] = f"{mode}-g{gs}"
            n_quantized += 1
            arr_f32 = dequant
        else:
            scheme_map[name] = "f16-passthrough"
            arr_f32 = src
        out_tensors[name] = arr_f32.reshape(header[name]["shape"]).astype(np.float16)
    elapsed = time.time() - t0

    st_blob = safetensors_save(out_tensors)
    del out_tensors
    checkpoint_sha256 = hashlib.sha256(st_blob).hexdigest()
    _write_bytes_in_dir(dfd, "model.safetensors", st_blob)
    st_size = len(st_blob)
    del st_blob

    # config.json + tokenizer.json: config copied (small), tokenizer symlinked
    # (12MB, identical across every arm — no reason to duplicate 5x).
    _write_bytes_in_dir(dfd, "config.json", (model_dir / "config.json").read_bytes())
    with contextlib.suppress(FileNotFoundError):
        os.unlink("tokenizer.json", dir_fd=dfd)
    os.symlink(str(model_dir / "tokenizer.json"), "tokenizer.json", dir_fd=dfd)

    corpus_path = REPO_ROOT / "docs/bench_results/wiki.test.raw"
    manifest = {
        "arm": arm,
        "source_model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "quantize_index_source": str(q4_dir / "quantize_index.json"),
        "n_tensors_total": len(tensor_names),
        "n_tensors_quantized": n_quantized,
        "scheme_map": scheme_map,
        "checkpoint_sha256": checkpoint_sha256,
        "corpus_file": str(corpus_path),
        "corpus_sha256": sha256_file(corpus_path) if corpus_path.exists() else None,
        "commit_sha": git_commit_sha(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quantize_wall_seconds": elapsed,
        "self_test": {
            "ok": self_test_result["ok"],
            "mode": self_test_result["mode"],
            "n_tensors_checked": self_test_result["n_tensors_checked"],
            "n_pass": self_test_result["n_pass"],
        },
    }
    _write_bytes_in_dir(dfd, "manifest.json", json.dumps(manifest, indent=2).encode())

    print(
        f"[{arm}] wrote {output_dir / 'model.safetensors'} ({st_size / 1e9:.2f} GB), "
        f"{n_quantized}/{len(tensor_names)} tensors fake-quantized, {elapsed:.1f}s"
    )
    return manifest


def cmd_quantize(args: argparse.Namespace) -> int:
    arm = args.arm
    model_dir = Path(args.model_dir)
    q4_dir = Path(args.q4_dir)
    output_dir, dfd = provision_scratch_dir(arm, args.output_dir)

    self_test_result = require_self_test_pass(model_dir, q4_dir)
    _revalidate_dir_handle(dfd, output_dir)
    write_arm_checkpoint(arm, model_dir, q4_dir, output_dir, dfd, self_test_result)
    os.close(dfd)
    return 0


# ---------------------------------------------------------------------------
# run-arm: the complete write -> eval -> delete loop
# ---------------------------------------------------------------------------


def run_eval_under_gpu_lock(cmd: list[str], cwd: Path) -> dict:
    """Run `cmd` while holding the machine-wide Metal GPU advisory flock
    (fleet convention: /tmp/lion-metal-gpu-test.lock — any process driving
    the GPU for measurements serializes through it; concurrent GPU work
    corrupts both timing and numerics). Blocks up to 30 minutes, then fails
    loud rather than hanging. Parses the evaluator's stdout for the
    `@@lattice {"ev":"perplexity",...}` structured event line and returns
    it parsed."""
    deadline = time.time() + GPU_LOCK_TIMEOUT_S
    with open(GPU_LOCK_PATH, "w") as lock_fh:
        while True:
            try:
                fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() > deadline:
                    raise SystemExit(
                        f"timed out waiting for {GPU_LOCK_PATH}; "
                        f"check holder with: lsof {GPU_LOCK_PATH}"
                    ) from None
                time.sleep(5)

        try:
            proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)

    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"evaluator failed with exit code {proc.returncode}")

    for line in proc.stdout.splitlines():
        if line.startswith("@@lattice "):
            event = json.loads(line[len("@@lattice ") :])
            if event.get("ev") == "perplexity":
                return event
    raise SystemExit("evaluator produced no @@lattice perplexity event; cannot record PPL")


def cmd_run_arm(args: argparse.Namespace) -> int:
    arm = args.arm
    model_dir = Path(args.model_dir)
    q4_dir = Path(args.q4_dir)
    output_dir, dfd = provision_scratch_dir(arm, args.output_dir)

    self_test_result = require_self_test_pass(model_dir, q4_dir)
    _revalidate_dir_handle(dfd, output_dir)
    manifest = write_arm_checkpoint(arm, model_dir, q4_dir, output_dir, dfd, self_test_result)

    # The evaluator receives output_dir as a PATH (read-only use), so verify
    # the path still names the provisioned directory right before launch.
    _revalidate_dir_handle(dfd, output_dir)

    eval_cmd = [
        "cargo",
        "run",
        "--release",
        "--features",
        "f16",
        "--bin",
        "eval_perplexity",
        "--",
        "--model-dir",
        str(output_dir),
        "--corpus-file",
        str(REPO_ROOT / "docs/bench_results/wiki.test.raw"),
        "--window",
        str(args.window),
        "--stride",
        str(args.stride),
        "--max-tokens",
        str(args.max_tokens),
        "--json",
        "--label",
        arm,
    ]
    print(f"=== running evaluator under GPU lock: {' '.join(eval_cmd)} ===")
    ppl_event = run_eval_under_gpu_lock(eval_cmd, cwd=REPO_ROOT)

    manifest["evaluator_invocation"] = " ".join(eval_cmd)
    manifest["perplexity"] = ppl_event
    _write_bytes_in_dir(dfd, "manifest.json", json.dumps(manifest, indent=2).encode())

    try:
        os.unlink("model.safetensors", dir_fd=dfd)
    except FileNotFoundError:
        pass
    else:
        print(
            f"[{arm}] deleted {output_dir / 'model.safetensors'} after successful eval "
            "(manifest + log retained)"
        )
    os.close(dfd)

    print(
        f"[{arm}] run-arm complete: ppl={ppl_event.get('ppl')} nll={ppl_event.get('nll')} "
        f"tokens={ppl_event.get('tokens')} windows={ppl_event.get('windows')}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("self-test", help="MANDATORY bit-exactness gate vs the real Rust quantizer")
    st.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    st.add_argument("--q4-dir", default=str(DEFAULT_Q4_DIR))
    st.add_argument(
        "--tensor",
        default="model.language_model.layers.0.mlp.down_proj.weight",
        help="tensor name to self-test (must be quantized=true in quantize_index.json)",
    )
    st.add_argument(
        "--all",
        action="store_true",
        help="sweep every tensor marked quantized=true in quantize_index.json",
    )
    st.set_defaults(func=cmd_self_test)

    sc = sub.add_parser("self-check", help="prove the non-finite (NaN/Inf) input guard is wired")
    sc.set_defaults(func=cmd_self_check)

    q = sub.add_parser("quantize", help="produce one arm's derived fake-quant f16 checkpoint dir")
    q.add_argument("--arm", required=True, choices=["A", "B", "C", "D", "E", "full_precision"])
    q.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    q.add_argument("--q4-dir", default=str(DEFAULT_Q4_DIR))
    q.add_argument("--output-dir", required=True)
    q.set_defaults(func=cmd_quantize)

    ra = sub.add_parser(
        "run-arm",
        help="self-test gate, write the arm checkpoint, evaluate PPL under the GPU lock, "
        "record the result, then delete the arm's weights",
    )
    ra.add_argument("--arm", required=True, choices=["A", "B", "C", "D", "E", "full_precision"])
    ra.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ra.add_argument("--q4-dir", default=str(DEFAULT_Q4_DIR))
    ra.add_argument(
        "--output-dir",
        default=None,
        help=f"defaults to {SCRATCH_ROOT}/arm_<ARM>; must resolve inside {SCRATCH_ROOT} "
        "(post-eval cleanup deletes model.safetensors from this directory)",
    )
    ra.add_argument("--window", type=int, default=512)
    ra.add_argument("--stride", type=int, default=256)
    ra.add_argument("--max-tokens", type=int, default=2048)
    ra.set_defaults(func=cmd_run_arm)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
