#!/usr/bin/env python3
"""
compare_logits.py — Side-by-side logit divergence: Lattice (F16 Metal) vs MLX.

Loads Qwen3.5-0.8B in both engines, runs prefill on the first 32 tokens of
WikiText-2, and prints per-position diagnostics.

bench_logit_dump takes the Metal GPU lock itself, so the Lattice side runs as a
GPU-handoff run (scripts/bench-command.sh builds the binary and launches it under
the machine locks). The MLX side also drives the GPU but never takes that lock, so
all of it (tokenizer load, token ids, prefill logits) runs in one child of this
script, launched as a plain bench-command.sh run that holds both machine locks
around it. The two runs are sequential: the MLX child finishes before the Lattice
run starts. Only argument handling, the analysis and output parsing run outside
the locks. The handoff admits a commit-clean checkout only and runs from the
repository root; it builds the binary itself, so a prebuilt binary directory is
not accepted.

Usage:
    PYTHONPATH=<mlx-site-packages> python3.11 scripts/compare_logits.py [--n-tokens N]

Env:
    LATTICE_MODEL_DIR     model dir (default ~/.lattice/models/qwen3.5-0.8b)
    LATTICE_LOGIT_TMP     temp file for binary logit dump (default /tmp/lattice_logits.bin)
    CORPUS_FILE           wiki corpus path (default docs/bench_results/wiki.test.raw)
    MLX_MODEL_PATH        local MLX model path (default same as LATTICE_MODEL_DIR)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()

MODEL_DIR = Path(os.environ.get("LATTICE_MODEL_DIR", HOME / ".lattice/models/qwen3.5-0.8b"))
LOGIT_TMP = os.environ.get("LATTICE_LOGIT_TMP", "/tmp/lattice_logits.bin")
CORPUS_FILE = Path(os.environ.get("CORPUS_FILE", REPO_ROOT / "docs/bench_results/wiki.test.raw"))
MLX_MODEL_PATH = os.environ.get("MLX_MODEL_PATH", str(MODEL_DIR))

N_TOKENS = 32  # number of prefill positions to compare
MLX_CHILD_DIR: str | None = None  # set only in the MLX child: where it writes its results
for arg in sys.argv[1:]:
    if arg.startswith("--n-tokens="):
        N_TOKENS = int(arg.split("=", 1)[1])
    elif arg == "--n-tokens" and sys.argv.index(arg) + 1 < len(sys.argv):
        N_TOKENS = int(sys.argv[sys.argv.index(arg) + 1])
    elif arg == "--mlx-child" and sys.argv.index(arg) + 1 < len(sys.argv):
        MLX_CHILD_DIR = sys.argv[sys.argv.index(arg) + 1]

# ---------------------------------------------------------------------------
# Step 1: Tokenize corpus with Lattice tokenizer (via BPE JSON)
# ---------------------------------------------------------------------------

def tokenize_corpus_bpe(corpus_path: Path, model_dir: Path, n: int) -> list[int]:
    """Tokenize text using the Qwen3.5 tokenizer JSON (tiktoken-compatible BPE).
    We use mlx_lm's tokenizer here — it loads the same tokenizer.json file.
    """
    from transformers import AutoTokenizer  # type: ignore
    # Qwen3.5's tokenizer is a stock Qwen2Tokenizer/tokenizer.json — no custom
    # tokenization class ships with the snapshot, so no remote code is needed
    # to load it, and running arbitrary code from a caller-supplied directory
    # is not a trade this script should make on their behalf.
    tok = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=False, local_files_only=True
    )
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

# The handoff builds bench_logit_dump itself, so the arguments below are the whole
# launch: the binary comes from this cargo invocation, never from a prebuilt path.
HANDOFF_ARGV = [
    "--gpu-handoff", "--label", "logit-divergence", "--",
    "cargo", "run", "--locked", "--release", "-p", "lattice-inference",
    "--bin", "bench_logit_dump", "--features", "metal-gpu,f16",
]

VOCAB_LINE = re.compile(r"VOCAB=(\d+)")
NPOS_LINE = re.compile(r"NPOS=(\d+)")


def parse_dump_header(output: str) -> tuple[int | None, int | None]:
    """Read VOCAB=N and NPOS=N from the handoff output.

    The handoff merges the binary's standard error into the stream, so progress
    lines share it; only a line that is exactly a header record counts.
    """
    vocab, npos = None, None
    for line in output.splitlines():
        record = line.rstrip("\r")
        if (match := VOCAB_LINE.fullmatch(record)) is not None:
            vocab = int(match.group(1))
        elif (match := NPOS_LINE.fullmatch(record)) is not None:
            npos = int(match.group(1))
    return vocab, npos


def reject_prebuilt_binary_dir(env: Mapping[str, str]) -> None:
    """Refuse the retired LATTICE_BIN_DIR override instead of silently ignoring it."""
    if env.get("LATTICE_BIN_DIR"):
        raise SystemExit(
            "LATTICE_BIN_DIR is not supported: the GPU handoff builds and launches "
            "bench_logit_dump itself and cannot run a prebuilt binary; unset it"
        )


def run_lattice_logit_dump(token_ids: list[int], model_dir: Path, out_path: str) -> np.ndarray:
    """Run bench_logit_dump under the GPU handoff, return float32 array [n_pos, vocab]."""
    # The handoff runs from the repository root, so a relative path the caller
    # gave is made absolute against the caller's directory first.
    out_file = os.path.abspath(out_path)
    env = os.environ.copy()
    env["LATTICE_MODEL_DIR"] = os.path.abspath(model_dir)
    env["LATTICE_LOGIT_OUT"] = out_file
    env["LATTICE_TOKENS"] = " ".join(str(t) for t in token_ids)
    if env.get("LATTICE_TOKENIZER_DIR"):
        env["LATTICE_TOKENIZER_DIR"] = os.path.abspath(env["LATTICE_TOKENIZER_DIR"])

    print(f"[lattice] running bench_logit_dump under the GPU handoff ({len(token_ids)} tokens)...")
    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "bench-command.sh"), *HANDOFF_ARGV],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(result.stdout[-2000:], file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"bench_logit_dump exited {result.returncode}")

    vocab, npos = parse_dump_header(result.stdout)

    print(result.stdout[-500:], file=sys.stderr)

    if vocab is None or npos is None:
        raise RuntimeError(f"bench_logit_dump output missing VOCAB/NPOS: {result.stdout}")

    raw = Path(out_file).read_bytes()
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


# MLX child: every MLX pass of this script runs in one child process, launched as
# a plain (no --gpu-handoff) bench-command.sh run. MLX does not call
# gpu_test_lock(), so the supervisor keeps both machine locks held around the
# child. The child is this script in an explicit mode, started with the parent's
# interpreter and environment, and hands its results back through files.
MLX_LAUNCH_ARGV = [
    "--label", "logit-mlx", "--",
    sys.executable, str(Path(__file__).resolve()), "--mlx-child",
]
MLX_TOKENS_FILE = "mlx_tokens.json"
MLX_LOGITS_FILE = "mlx_logits.bin"


def run_mlx_child(out_dir: str) -> None:
    """Child mode: tokenize, run the MLX prefill, write both results into out_dir."""
    print("\n[step 1] tokenizing corpus with MLX tokenizer...")
    try:
        token_ids = tokenize_via_mlx(CORPUS_FILE, MLX_MODEL_PATH, N_TOKENS)
    except Exception as e:
        print(f"  mlx_lm tokenizer failed ({e}), trying transformers...")
        try:
            token_ids = tokenize_corpus_bpe(CORPUS_FILE, MODEL_DIR, N_TOKENS)
        except Exception as e2:
            raise RuntimeError(f"Tokenization failed: {e}, {e2}") from e2
    print(f"  token_ids: {token_ids[:10]}... ({len(token_ids)} tokens)")

    print("\n[step 2] collecting MLX logits...")
    logits = run_mlx_logits(token_ids, MLX_MODEL_PATH)

    out = Path(out_dir)
    (out / MLX_LOGITS_FILE).write_bytes(logits.astype("<f4").tobytes())
    # Written last and renamed into place, so the parent finds it only when the logits are complete.
    pending = out / (MLX_TOKENS_FILE + ".part")
    pending.write_text(json.dumps({"token_ids": token_ids, "n_pos": int(logits.shape[0]), "vocab": int(logits.shape[1])}))
    pending.replace(out / MLX_TOKENS_FILE)


def collect_mlx_via_child() -> tuple[list[int], np.ndarray]:
    """Run the MLX child under the machine locks and read back its token ids and logits."""
    work = tempfile.mkdtemp(prefix="compare-logits-mlx-")
    try:
        # Neither stream is captured: the child's progress lines reach the caller's console.
        result = subprocess.run(
            [str(REPO_ROOT / "scripts" / "bench-command.sh"), *MLX_LAUNCH_ARGV, work,
             f"--n-tokens={N_TOKENS}"],
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"MLX child exited {result.returncode}")
        meta_path = Path(work) / MLX_TOKENS_FILE
        if not meta_path.is_file():
            raise RuntimeError(f"MLX child exited 0 but wrote no {MLX_TOKENS_FILE}")
        meta = json.loads(meta_path.read_text())
        raw = (Path(work) / MLX_LOGITS_FILE).read_bytes()
        token_ids = [int(t) for t in meta["token_ids"]]
        n_pos, vocab = int(meta["n_pos"]), int(meta["vocab"])
        if len(raw) != n_pos * vocab * 4:
            raise RuntimeError(f"MLX logits file is {len(raw)} bytes, expected {n_pos * vocab * 4}")
        arr = np.frombuffer(raw, dtype="<f4").reshape(n_pos, vocab).copy()
        print(f"[mlx] child returned {len(token_ids)} token ids and logits shape={arr.shape}")
        return token_ids, arr
    finally:
        shutil.rmtree(work, ignore_errors=True)


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
    reject_prebuilt_binary_dir(os.environ)
    if MLX_CHILD_DIR is not None:
        run_mlx_child(MLX_CHILD_DIR)
        return
    print(f"[setup] model dir  : {MODEL_DIR}")
    print(f"[setup] corpus     : {CORPUS_FILE}")
    print(f"[setup] n_tokens   : {N_TOKENS}")

    # Steps 1 and 2: tokenizer, token ids and MLX logits, all in the MLX child.
    # It finishes before the Lattice run starts; the two never overlap.
    token_ids, mlx_logits = collect_mlx_via_child()

    # Lattice logits
    print("\n[step 3] collecting Lattice logits (F16 Metal)...")
    lat_logits = run_lattice_logit_dump(token_ids, MODEL_DIR, LOGIT_TMP)

    # Align sequence lengths
    n = min(lat_logits.shape[0], mlx_logits.shape[0], len(token_ids))
    if lat_logits.shape[0] != mlx_logits.shape[0]:
        print(f"[warn] shape mismatch: lattice={lat_logits.shape[0]}, mlx={mlx_logits.shape[0]} — comparing first {n}")

    # Analysis table
    print("\n[step 4] per-position analysis")
    analyze(lat_logits[:n], mlx_logits[:n], token_ids[:n])


if __name__ == "__main__":
    main()
