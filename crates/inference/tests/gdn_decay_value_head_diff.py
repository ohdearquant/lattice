# /// script
# dependencies = ["numpy"]
# ///
"""Spec oracle for Qwen3.6-27B GDN decay value-head granularity.

Run from the repository root with:

    uv run crates/inference/tests/gdn_decay_value_head_diff.py

The oracle reads the real layer-0 GDN tensors from the local qwen3.6-27B q4
checkpoint and verifies that decay gates are computed per value head. The
known-bad key-head collapse reuses rows 0..15 three times each for the 48 value
heads.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


DEFAULT_MODEL_DIR = Path.home() / ".lattice/models/qwen3.6-27b-q4"
LAYER_PREFIX = "model_language_model_layers_0_linear_attn"
NUM_VALUE_HEADS = 48
NUM_KEY_HEADS = 16
HEAD_RATIO = NUM_VALUE_HEADS // NUM_KEY_HEADS
HIDDEN_SIZE = 5120
DISTINCT_TOL = 1.0e-7
DELTA_TOL = 1.0e-6


def _read_exact_header(raw: bytes, expected_magic: bytes) -> tuple[tuple[int, ...], int]:
    if len(raw) < 20:
        raise ValueError("file too small for tensor header")
    magic = raw[:4]
    if magic != expected_magic:
        raise ValueError(f"invalid magic: expected {expected_magic!r}, got {magic!r}")

    version, ndim = struct.unpack_from("<II", raw, 4)
    expected_version = 1 if expected_magic == b"KHF1" else 2
    if version != expected_version:
        raise ValueError(
            f"unsupported {expected_magic.decode()} version {version}; "
            f"expected {expected_version}"
        )

    offset = 12
    dims_end = offset + 8 * ndim
    if dims_end + 8 > len(raw):
        raise ValueError("truncated tensor header")
    shape = struct.unpack_from("<" + "Q" * ndim, raw, offset)
    offset = dims_end
    numel = struct.unpack_from("<Q", raw, offset)[0]
    offset += 8

    shape_product = int(np.prod(shape, dtype=np.uint64))
    if shape_product != numel:
        raise ValueError(f"shape product {shape_product} != header numel {numel}")
    return tuple(int(dim) for dim in shape), offset


def read_khf1_f16(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    shape, offset = _read_exact_header(raw, b"KHF1")
    numel = int(np.prod(shape, dtype=np.uint64))
    payload_bytes = numel * 2
    if offset + payload_bytes != len(raw):
        raise ValueError(
            f"{path.name}: expected {payload_bytes} payload bytes, "
            f"found {len(raw) - offset}"
        )
    return np.frombuffer(raw, dtype="<f2", count=numel, offset=offset).astype(
        np.float32
    ).reshape(shape)


def read_khq4(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    shape, offset = _read_exact_header(raw, b"KHQ4")
    numel = int(np.prod(shape, dtype=np.uint64))
    block_count = (numel + 31) // 32
    payload_bytes = block_count * 20
    if offset + payload_bytes != len(raw):
        raise ValueError(
            f"{path.name}: expected {payload_bytes} payload bytes, "
            f"found {len(raw) - offset}"
        )

    out = np.empty(block_count * 32, dtype=np.float32)
    cursor = offset
    dst = 0
    for _ in range(block_count):
        scale = np.frombuffer(raw, dtype="<f2", count=1, offset=cursor)[0].astype(
            np.float32
        )
        bias = np.frombuffer(raw, dtype="<f2", count=1, offset=cursor + 2)[0].astype(
            np.float32
        )
        packed = np.frombuffer(raw, dtype=np.uint8, count=16, offset=cursor + 4)
        vals = np.empty(32, dtype=np.float32)
        vals[0::2] = (packed & 0x0F).astype(np.float32) * scale + bias
        vals[1::2] = (packed >> 4).astype(np.float32) * scale + bias
        out[dst : dst + 32] = vals
        cursor += 20
        dst += 32

    return out[:numel].reshape(shape)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(x.astype(np.float32), np.float32(0.0)).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (np.float32(1.0) / (np.float32(1.0) + np.exp(-x))).astype(np.float32)


def assert_shape(name: str, array: np.ndarray, expected: tuple[int, ...]) -> None:
    if array.shape != expected:
        raise AssertionError(f"{name} shape {array.shape} != expected {expected}")


def pairwise_min_delta(values: np.ndarray) -> float:
    sorted_values = np.sort(values.astype(np.float64))
    return float(np.min(np.diff(sorted_values)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3.6-27B layer-0 GDN decay value-head oracle"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing qwen3.6-27B q4 tensor files",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser()
    a_log = read_khf1_f16(model_dir / f"{LAYER_PREFIX}_A_log.f16")
    dt_bias = read_khf1_f16(model_dir / f"{LAYER_PREFIX}_dt_bias.f16")
    in_proj_a = read_khq4(model_dir / f"{LAYER_PREFIX}_in_proj_a_weight.q4")
    in_proj_b = read_khq4(model_dir / f"{LAYER_PREFIX}_in_proj_b_weight.q4")

    assert_shape("A_log", a_log, (NUM_VALUE_HEADS,))
    assert_shape("dt_bias", dt_bias, (NUM_VALUE_HEADS,))
    assert_shape("in_proj_a", in_proj_a, (NUM_VALUE_HEADS, HIDDEN_SIZE))
    assert_shape("in_proj_b", in_proj_b, (NUM_VALUE_HEADS, HIDDEN_SIZE))

    hidden = np.linspace(-1.0, 1.0, HIDDEN_SIZE, dtype=np.float32)
    alpha = in_proj_a @ hidden
    beta = sigmoid(in_proj_b @ hidden)

    gates = np.exp(-np.exp(a_log) * softplus(alpha + dt_bias)).astype(np.float32)
    wrong_collapsed = np.empty_like(gates)
    for h in range(NUM_VALUE_HEADS):
        kh = h // HEAD_RATIO
        wrong_collapsed[h] = np.exp(
            -np.exp(a_log[kh]) * softplus(alpha[kh] + dt_bias[kh])
        ).astype(np.float32)

    rounded_unique = int(np.unique(np.round(gates.astype(np.float64), 8)).size)
    min_gate_delta = pairwise_min_delta(gates)
    if rounded_unique != NUM_VALUE_HEADS or min_gate_delta <= DISTINCT_TOL:
        raise AssertionError(
            f"expected {NUM_VALUE_HEADS} distinct value-head gates; "
            f"rounded_unique={rounded_unique}, min_pairwise_delta={min_gate_delta:.9g}"
        )

    group_min_deltas: list[float] = []
    for kh in range(NUM_KEY_HEADS):
        group = gates[kh * HEAD_RATIO : (kh + 1) * HEAD_RATIO]
        group_min_delta = pairwise_min_delta(group)
        group_min_deltas.append(group_min_delta)
        if group_min_delta <= DISTINCT_TOL:
            raise AssertionError(
                f"value heads sharing key head {kh} collapsed: {group.tolist()}"
            )

    max_delta = float(np.max(np.abs(gates - wrong_collapsed)))
    wrong_changed = int(np.count_nonzero(np.abs(gates - wrong_collapsed) > DELTA_TOL))
    wrong_unique = int(np.unique(np.round(wrong_collapsed.astype(np.float64), 8)).size)

    print("Qwen3.6-27B layer-0 GDN decay oracle")
    print(f"model_dir: {model_dir}")
    print(
        f"shapes: A_log={a_log.shape}, dt_bias={dt_bias.shape}, "
        f"in_proj_a={in_proj_a.shape}, in_proj_b={in_proj_b.shape}"
    )
    print("hidden: linspace(-1.0, 1.0, 5120, dtype=float32)")
    print(f"alpha_range: [{float(alpha.min()):.9g}, {float(alpha.max()):.9g}]")
    print(f"beta_range:  [{float(beta.min()):.9g}, {float(beta.max()):.9g}]")
    print(f"gate_range:  [{float(gates.min()):.9g}, {float(gates.max()):.9g}]")
    print(f"distinct_value_head_gates: {rounded_unique}/48")
    print(f"min_pairwise_gate_delta: {min_gate_delta:.9g}")
    print(f"min_shared_key_head_group_delta: {min(group_min_deltas):.9g}")
    print(f"wrong_collapsed_unique_gates: {wrong_unique}/48")
    print(f"wrong_collapsed_changed_heads(|delta|>{DELTA_TOL:g}): {wrong_changed}/48")
    print(f"max|delta g| vs wrong per-key-head collapse: {max_delta:.9g}")
    print("assertion: PASS - GDN decay gates are per-value-head, not 16 repeated x3")


if __name__ == "__main__":
    main()
