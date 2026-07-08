#!/usr/bin/env python3
"""Generate NumPy golden fixtures for the chunkwise Gated-DeltaNet (GDN)
prefill oracle parity test.

This script is self-contained: it embeds its own copy of the reference
sequential and chunkwise GDN recurrences (pure NumPy, f32 throughout) so the
fixtures can be regenerated from this repository alone. It writes, per case,
a JSON file containing both the generated inputs and the NumPy-computed
outputs of both recurrences, so the Rust parity test can load identical
inputs and compare against frozen NumPy outputs without reproducing any RNG
state.

Usage:
    uv run python3 crates/inference/tests/fixtures/gdn_chunk/generate.py

Requirements:
    - numpy

Run once to (re)generate the fixtures, then commit the JSON output alongside
this script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Reference recurrence (verbatim copy — keep in lockstep with the validated
# NumPy reference this fixture generator was derived from).
# ---------------------------------------------------------------------------


def _as_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def l2_normalize_rows(x: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Row L2 normalization: x / sqrt(sum(x^2) + eps)."""
    x = _as_f32(x).copy()
    denom = np.sqrt(np.sum(x * x, axis=1, keepdims=True, dtype=np.float32) + np.float32(eps))
    return (x / denom).astype(np.float32)


def solve_unit_lower(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve A X = B for a unit-lower-triangular A.

    A is C x C with diagonal entries equal to 1. B is C x D. Written as a
    reference solve (forward substitution), not a fast kernel.
    """
    a = _as_f32(a)
    x = _as_f32(b).copy()
    c = a.shape[0]
    for i in range(c):
        if i:
            # x[i] -= A[i, :i] @ x[:i]
            x[i] = x[i] - (a[i, :i].astype(np.float32) @ x[:i].astype(np.float32)).astype(np.float32)
    return x.astype(np.float32)


def sequential_gdn(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    beta: np.ndarray,
    alpha: np.ndarray,
    h0: np.ndarray,
    *,
    apply_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sequential gated delta recurrence in value-major H = S^T layout.

    Shapes:
        q, k:   [T, d_k]
        v:      [T, d_v]
        beta:   [T]
        alpha:  [T]  decay gate g_t
        h0:     [d_v, d_k]

    Recurrence:
        H_t = alpha_t H_{t-1} + beta_t (v_t - alpha_t H_{t-1} k_t) k_t^T
        o_t = H_t q_t / sqrt(d_k)
    """
    q = _as_f32(q)
    k = _as_f32(k)
    v = _as_f32(v)
    beta = _as_f32(beta).reshape(-1)
    alpha = _as_f32(alpha).reshape(-1)
    h = _as_f32(h0).copy()
    t, dk = q.shape
    dv = v.shape[1]
    out = np.empty((t, dv), dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(np.float32(dk))) if apply_scale else np.float32(1.0)

    for i in range(t):
        h_decayed = (alpha[i] * h).astype(np.float32)
        kv = (h_decayed @ k[i].astype(np.float32)).astype(np.float32)
        r = (beta[i] * (v[i] - kv)).astype(np.float32)
        h = (h_decayed + np.outer(r, k[i]).astype(np.float32)).astype(np.float32)
        out[i] = (h @ q[i].astype(np.float32) * scale).astype(np.float32)
    return out, h


def chunkwise_gdn(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    beta: np.ndarray,
    alpha: np.ndarray,
    h0: np.ndarray,
    *,
    chunk_size: int = 128,
    apply_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact chunkwise WY/DPLR formulation for scalar-gated DeltaNet.

    For one chunk of C tokens, let
        gamma_log_j = sum_{r=0}^j ln(alpha_r)   (log-space cumulative decay)
        G       = K K^T
        A       = I + tril(diag(beta) G, -1)
        T       = A^{-1} diag(beta)
        W       = T K
        U       = Gamma T Gamma^{-1} V
        R       = U - Gamma W H0^T
        LQK     = tril(Gamma (Q K^T) Gamma^{-1}, 0)

    Then
        O_chunk = Gamma (Q H0^T) + LQK R
        H_next  = gamma_end H0 + R^T (gamma_end Gamma^{-1} K)

    Every `Gamma`/`Gamma^{-1}` factor above is, algebraically, the ratio
    `gamma_i / gamma_j` (i>=j) or a bare `gamma_i`, all bounded in `[0, 1]`.
    Rather than materializing `gamma = cumprod(alpha)` and `gamma_inv =
    1/gamma` in LINEAR space (which underflows to `0.0` / `inf` within a
    single chunk on strongly-decaying real GDN heads, poisoning the chunk
    with `0*inf = NaN`), every factor here is computed as `exp` of a
    difference of `gamma_log` — a pure numerical-stability reformulation
    that keeps the Rust port (`gdn_chunk_ref::chunkwise_gdn`) and this
    generator byte-consistent; the values are identical to the linear-space
    ratios up to f32 rounding.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_size not in (64, 128):
        raise ValueError("this reference intentionally gates to chunk_size in {64, 128}")

    q = _as_f32(q)
    k = _as_f32(k)
    v = _as_f32(v)
    beta = _as_f32(beta).reshape(-1)
    alpha = _as_f32(alpha).reshape(-1)
    h = _as_f32(h0).copy()

    t_total, dk = q.shape
    dv = v.shape[1]
    out = np.empty((t_total, dv), dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(np.float32(dk))) if apply_scale else np.float32(1.0)

    for start in range(0, t_total, chunk_size):
        end = min(start + chunk_size, t_total)
        q_c = q[start:end]
        k_c = k[start:end]
        v_c = v[start:end]
        b_c = beta[start:end]
        a_c = alpha[start:end]
        c = end - start

        # gamma_log[i] = sum_{r<=i} ln(alpha_c[r]); alpha_c is always in
        # (0, 1] by construction so ln is finite. The tiny-positive clamp
        # only guards a degenerate exact-zero input (keeps ln finite, near
        # the kernel's own -88 log-alpha floor, instead of -inf).
        tiny = np.float32(np.finfo(np.float32).tiny)
        ln_a_c = np.log(np.maximum(a_c, tiny)).astype(np.float32)
        gamma_log = np.cumsum(ln_a_c, dtype=np.float32).astype(np.float32)
        gamma_log_end = gamma_log[-1]
        gamma = np.exp(gamma_log, dtype=np.float32).astype(np.float32)

        # 1. Intra-chunk key Gram and unit-lower triangular solve.
        g = (k_c @ k_c.T).astype(np.float32)  # [C, C]
        a_plain = (np.eye(c, dtype=np.float32)
                   + np.tril((b_c[:, None] * g).astype(np.float32), k=-1)).astype(np.float32)
        rhs_k = (b_c[:, None] * k_c).astype(np.float32)  # diag(beta) K
        rhs_v = (b_c[:, None] * v_c).astype(np.float32)  # diag(beta) V

        w = solve_unit_lower(a_plain, rhs_k)  # W = T K, [C, d_k]

        # U = Gamma T Gamma^{-1} V, built as a directly gamma-scaled
        # unit-lower system (l_jk = exp(gamma_log_j - gamma_log_k) =
        # gamma_j/gamma_k, computed without ever forming 1/gamma). The
        # upper-triangle entries of this full [C,C] difference matrix (i<j)
        # can legitimately overflow `exp` on a strongly-decaying head — they
        # are discarded by `np.tril`'s literal-zero masking below (not a
        # multiply-by-zero), so an `inf`/`nan` there never reaches the
        # result; suppress the resulting benign RuntimeWarning locally.
        with np.errstate(over="ignore", invalid="ignore"):
            decay_ratio = np.exp(
                gamma_log[:, None] - gamma_log[None, :], dtype=np.float32
            ).astype(np.float32)
            a_gamma_full = ((b_c[:, None] * g) * decay_ratio).astype(np.float32)
        a_gamma = (np.eye(c, dtype=np.float32)
                   + np.tril(a_gamma_full, k=-1)).astype(np.float32)
        u = solve_unit_lower(a_gamma, rhs_v)  # [C, d_v]

        # 2. Decayed incoming-state contribution and pseudo-value residual.
        w_h0 = (w @ h.T).astype(np.float32)  # [C, d_v]
        r = (u - gamma[:, None] * w_h0).astype(np.float32)  # [C, d_v]

        # 3. Within-chunk output: Gamma QH0^T + tril(Gamma QK^T Gamma^-1) R.
        q_h0 = (q_c @ h.T).astype(np.float32)  # [C, d_v]
        qk = (q_c @ k_c.T).astype(np.float32)  # [C, C]
        with np.errstate(over="ignore", invalid="ignore"):
            lqk_full = (qk * decay_ratio).astype(np.float32)
        lqk = np.tril(lqk_full, k=0)
        out[start:end] = ((gamma[:, None] * q_h0 + lqk @ r) * scale).astype(np.float32)

        # 4. Cross-chunk state carry.
        k_right_scale = np.exp(gamma_log_end - gamma_log, dtype=np.float32).astype(np.float32)
        k_right = (k_right_scale[:, None] * k_c).astype(np.float32)  # [C, d_k]
        gamma_end = np.exp(gamma_log_end, dtype=np.float32)
        h = (gamma_end * h + r.T @ k_right).astype(np.float32)  # [d_v, d_k]

    return out, h


# ---------------------------------------------------------------------------
# Fixture generation.
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).resolve().parent

# (seed, length, dk, dv, chunk_size, alpha_range, tag) — lengths chosen to
# exercise both an even chunk split and an uneven tail chunk. `tag`, when
# non-empty, is appended to the output filename (`..._{tag}.json`).
CASES = [
    (7, 191, 128, 128, 64, (0.82, 0.999), ""),   # 3 chunks: 64, 64, 63
    (11, 384, 128, 128, 128, (0.82, 0.999), ""),  # 3 chunks: 128, 128, 128
    (23, 130, 128, 128, 64, (0.82, 0.999), ""),  # uneven tail: 64, 64, 2
    # gdn175 S1 fix: strongly-decaying head (alpha as low as 1e-4) so
    # cumprod(alpha) underflows to 0.0 in LINEAR space within one C=64
    # chunk on the pre-fix oracle; the log-space reformulation must still
    # reproduce sequential_gdn to <= 1e-5 here (regression fixture for the
    # oracle gamma-underflow NaN this case was added to catch).
    (31, 160, 128, 128, 64, (1.0e-4, 0.4), "strongdecay"),
]

TOL = 1.0e-5


def _round_scalar(x: float, sig: int = 9) -> float:
    return float(f"{x:.{sig}g}")


def _round_nested(obj):
    if isinstance(obj, list):
        return [_round_nested(v) for v in obj]
    return _round_scalar(float(obj))


def to_json_array(arr: np.ndarray) -> list:
    return _round_nested(arr.astype(np.float64).tolist())


def generate_case(
    seed: int,
    length: int,
    dk: int,
    dv: int,
    chunk_size: int,
    alpha_range: Tuple[float, float] = (0.82, 0.999),
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    q = l2_normalize_rows(rng.normal(size=(length, dk)).astype(np.float32))
    k = l2_normalize_rows(rng.normal(size=(length, dk)).astype(np.float32))
    # Keep magnitudes modest so f32 associativity, not ill-conditioning, dominates.
    v = (0.15 * rng.normal(size=(length, dv))).astype(np.float32)
    beta = rng.uniform(0.02, 0.98, size=(length,)).astype(np.float32)
    alpha = rng.uniform(alpha_range[0], alpha_range[1], size=(length,)).astype(np.float32)
    h0 = (0.03 * rng.normal(size=(dv, dk))).astype(np.float32)

    out_seq, h_seq = sequential_gdn(q, k, v, beta, alpha, h0)
    out_chk, h_chk = chunkwise_gdn(q, k, v, beta, alpha, h0, chunk_size=chunk_size)

    max_abs_out = float(np.abs(out_seq - out_chk).max(initial=0.0))
    max_abs_state = float(np.abs(h_seq - h_chk).max(initial=0.0))
    if max_abs_out > TOL or max_abs_state > TOL:
        raise AssertionError(
            f"seed={seed}: chunkwise vs sequential exceeded tolerance {TOL}: "
            f"max_abs_out={max_abs_out}, max_abs_state={max_abs_state}"
        )

    return {
        "seed": seed,
        "length": length,
        "dk": dk,
        "dv": dv,
        "chunk_size": chunk_size,
        "q": to_json_array(q),
        "k": to_json_array(k),
        "v": to_json_array(v),
        "beta": to_json_array(beta),
        "alpha": to_json_array(alpha),
        "h0": to_json_array(h0),
        "out_seq": to_json_array(out_seq),
        "h_seq": to_json_array(h_seq),
        "out_chk": to_json_array(out_chk),
        "h_chk": to_json_array(h_chk),
    }


def main() -> None:
    for seed, length, dk, dv, chunk_size, alpha_range, tag in CASES:
        case = generate_case(seed, length, dk, dv, chunk_size, alpha_range=alpha_range)
        suffix = f"_{tag}" if tag else ""
        out_path = FIXTURE_DIR / f"case_seed{seed}_len{length}_chunk{chunk_size}{suffix}.json"
        with out_path.open("w") as f:
            json.dump(case, f)
        print(f"wrote {out_path} (seed={seed}, length={length}, chunk_size={chunk_size})")


if __name__ == "__main__":
    main()
