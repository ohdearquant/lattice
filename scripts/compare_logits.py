#!/usr/bin/env python3
"""
compare_logits.py — Side-by-side logit divergence: Lattice (F16 Metal) vs MLX.

Loads Qwen3.5-0.8B in both engines, runs prefill on the first 32 tokens of
WikiText-2, and prints per-position diagnostics.

Usage:
    PYTHONPATH=<mlx-site-packages> python3.11 scripts/compare_logits.py [--n-tokens N]

Env:
    LATTICE_MODEL_DIR     model dir (default ~/.lattice/models/qwen3.5-0.8b)
    LATTICE_BIN_DIR       dir containing bench_logit_dump (default ./target/release)
    LATTICE_LOGIT_TMP     temp file for binary logit dump (default /tmp/lattice_logits.bin)
    CORPUS_FILE           wiki corpus path (default docs/bench_results/wiki.test.raw)
    MLX_MODEL_PATH        local MLX model path (default same as LATTICE_MODEL_DIR)
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()

MODEL_DIR = Path(os.environ.get("LATTICE_MODEL_DIR", HOME / ".lattice/models/qwen3.5-0.8b"))
BIN_DIR = Path(os.environ.get("LATTICE_BIN_DIR", REPO_ROOT / "target/release"))
LOGIT_TMP = os.environ.get("LATTICE_LOGIT_TMP", "/tmp/lattice_logits.bin")
CORPUS_FILE = Path(os.environ.get("CORPUS_FILE", REPO_ROOT / "docs/bench_results/wiki.test.raw"))
MLX_MODEL_PATH = os.environ.get("MLX_MODEL_PATH", str(MODEL_DIR))

N_TOKENS = 32  # number of prefill positions to compare
for arg in sys.argv[1:]:
    if arg.startswith("--n-tokens="):
        N_TOKENS = int(arg.split("=", 1)[1])
    elif arg == "--n-tokens" and sys.argv.index(arg) + 1 < len(sys.argv):
        N_TOKENS = int(sys.argv[sys.argv.index(arg) + 1])

# ---------------------------------------------------------------------------
# Step 1: Tokenize corpus with Lattice tokenizer (via BPE JSON)
# ---------------------------------------------------------------------------

def tokenize_corpus_bpe(corpus_path: Path, model_dir: Path, n: int) -> list[int]:
    """Tokenize text using the Qwen3.5 tokenizer JSON (tiktoken-compatible BPE).
    We use mlx_lm's tokenizer here — it loads the same tokenizer.json file.
    """
    from transformers import AutoTokenizer  # type: ignore
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    text = corpus_path.read_text(encoding="utf-8")[:8192]  # first ~8K chars
    ids = tok.encode(text, add_special_tokens=False)
    return ids[:n]


def tokenize_via_mlx(corpus_path: Path, model_dir: Path, n: int) -> list[int]:
    """Tokenize using mlx_lm's load (which bundles the tokenizer)."""
    from mlx_lm import load  # type: ignore
    _, tok = load(str(model_dir))
    text = corpus_path.read_text(encoding="utf-8")[:8192]
    ids = tok.encode(text, add_special_tokens=False)
    return ids[:n]


# ---------------------------------------------------------------------------
# Step 2: Lattice logit dump (via bench_logit_dump subprocess)
# ---------------------------------------------------------------------------

def run_lattice_logit_dump(token_ids: list[int], model_dir: Path, bin_dir: Path, out_path: str) -> np.ndarray:
    """Run bench_logit_dump, return float32 array [n_pos, vocab]."""
    binary = bin_dir / "bench_logit_dump"
    if not binary.exists():
        raise FileNotFoundError(f"bench_logit_dump not found at {binary}")

    env = os.environ.copy()
    env["LATTICE_MODEL_DIR"] = str(model_dir)
    env["LATTICE_LOGIT_OUT"] = out_path
    env["LATTICE_TOKENS"] = " ".join(str(t) for t in token_ids)

    print(f"[lattice] running {binary} ({len(token_ids)} tokens)...")
    result = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        print(result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"bench_logit_dump exited {result.returncode}")

    # Parse VOCAB=N and NPOS=N from stdout
    vocab, npos = None, None
    for line in result.stdout.splitlines():
        if line.startswith("VOCAB="):
            vocab = int(line.split("=", 1)[1])
        elif line.startswith("NPOS="):
            npos = int(line.split("=", 1)[1])

    print(result.stderr[-500:], file=sys.stderr)

    if vocab is None or npos is None:
        raise RuntimeError(f"bench_logit_dump stdout missing VOCAB/NPOS: {result.stdout}")

    raw = Path(out_path).read_bytes()
    arr = np.frombuffer(raw, dtype="<f4").reshape(npos, vocab).copy()
    print(f"[lattice] loaded logits: shape={arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Step 3: MLX logit collection
# ---------------------------------------------------------------------------

def run_mlx_logits(token_ids: list[int], model_path: str) -> np.ndarray:
    """Run MLX prefill and collect logits at every position.

    mlx_lm.models return logits of shape [1, seq_len, vocab] from __call__.
    """
    import mlx.core as mx  # type: ignore
    from mlx_lm import load  # type: ignore

    print(f"[mlx] loading model from {model_path}...")
    model, tokenizer = load(model_path)
    model.eval()

    input_ids = mx.array([token_ids])  # [1, n]

    print(f"[mlx] running prefill ({len(token_ids)} tokens)...")
    logits = model(input_ids)  # [1, n, vocab]
    mx.eval(logits)

    # MLX arrays may need explicit conversion via tolist() or __array__
    logits_np = np.asarray(logits[0].tolist(), dtype=np.float32)  # [n, vocab]
    arr = logits_np
    print(f"[mlx] loaded logits: shape={arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Step 4: Per-position analysis
# ---------------------------------------------------------------------------

def top5_jaccard(a_logits: np.ndarray, b_logits: np.ndarray) -> float:
    a5 = set(np.argsort(a_logits)[-5:].tolist())
    b5 = set(np.argsort(b_logits)[-5:].tolist())
    if not a5 and not b5:
        return 1.0
    return len(a5 & b5) / len(a5 | b5)


def kl_div(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(p || q) where p=mlx (reference), q=lattice."""
    p = stable_softmax(p_logits)
    q = stable_softmax(q_logits)
    # Clip to avoid log(0)
    q = np.clip(q, 1e-10, None)
    p = np.clip(p, 1e-10, None)
    return float(np.sum(p * np.log(p / q)))


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def analyze(lat: np.ndarray, mlx: np.ndarray, token_ids: list[int]) -> None:
    n = min(lat.shape[0], mlx.shape[0], len(token_ids))
    vocab = lat.shape[1]

    header = (
        f"{'pos':>4}  {'tok_id':>8}  {'argmax_lat':>12}  {'argmax_mlx':>12}  "
        f"{'match':>5}  {'top5_jac':>9}  {'kl_div':>8}  "
        f"{'lat_max':>8}  {'mlx_max':>8}"
    )
    print()
    print(header)
    print("-" * len(header))

    first_div = None
    kl_sum = 0.0
    jac_sum = 0.0

    for i in range(n):
        lat_v = lat[i]
        mlx_v = mlx[i]

        am_lat = int(np.argmax(lat_v))
        am_mlx = int(np.argmax(mlx_v))
        match = am_lat == am_mlx
        jac = top5_jaccard(lat_v, mlx_v)
        kl = kl_div(mlx_v, lat_v)
        kl_sum += kl
        jac_sum += jac

        if first_div is None and not match:
            first_div = i

        print(
            f"{i:>4}  {token_ids[i]:>8}  {am_lat:>12}  {am_mlx:>12}  "
            f"{'Y' if match else 'N':>5}  {jac:>9.4f}  {kl:>8.4f}  "
            f"{lat_v.max():>8.3f}  {mlx_v.max():>8.3f}"
        )

    print()
    print("=" * 60)
    print(f"  first_argmax_divergence : pos={first_div if first_div is not None else 'none (all match)'}")
    print(f"  mean_kl_div             : {kl_sum / n:.6f}")
    print(f"  mean_top5_jaccard       : {jac_sum / n:.4f}")
    print(f"  vocab_size              : {vocab}")
    print(f"  positions_compared      : {n}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[setup] model dir  : {MODEL_DIR}")
    print(f"[setup] corpus     : {CORPUS_FILE}")
    print(f"[setup] n_tokens   : {N_TOKENS}")

    # Tokenize
    print("\n[step 1] tokenizing corpus with MLX tokenizer...")
    try:
        token_ids = tokenize_via_mlx(CORPUS_FILE, MLX_MODEL_PATH, N_TOKENS)
    except Exception as e:
        print(f"  mlx_lm tokenizer failed ({e}), trying transformers...")
        try:
            token_ids = tokenize_corpus_bpe(CORPUS_FILE, MODEL_DIR, N_TOKENS)
        except Exception as e2:
            # Fallback: read raw token IDs from a pre-computed source
            raise RuntimeError(f"Tokenization failed: {e}, {e2}") from e2

    print(f"  token_ids: {token_ids[:10]}... ({len(token_ids)} tokens)")

    # MLX logits
    print("\n[step 2] collecting MLX logits...")
    mlx_logits = run_mlx_logits(token_ids, MLX_MODEL_PATH)

    # Lattice logits
    print("\n[step 3] collecting Lattice logits (F16 Metal)...")
    lat_logits = run_lattice_logit_dump(token_ids, MODEL_DIR, BIN_DIR, LOGIT_TMP)

    # Align sequence lengths
    n = min(lat_logits.shape[0], mlx_logits.shape[0], len(token_ids))
    if lat_logits.shape[0] != mlx_logits.shape[0]:
        print(f"[warn] shape mismatch: lattice={lat_logits.shape[0]}, mlx={mlx_logits.shape[0]} — comparing first {n}")

    # Analysis table
    print("\n[step 4] per-position analysis")
    analyze(lat_logits[:n], mlx_logits[:n], token_ids[:n])


if __name__ == "__main__":
    main()
