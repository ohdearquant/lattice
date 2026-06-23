#!/usr/bin/env python3
"""Agentic-workload speed comparison: N-token context / 100-token response.

Three-way comparison: lattice (pure-Rust Metal) vs ollama (llama.cpp Metal)
vs MLX (mlx_lm, Apple MLX).

Methodology:
  TTFT   = prefill latency (wall time to generate 1 token)
  Total  = wall time for prefill + RESP-token decode
  Decode = Total - TTFT  ->  decode_tok_s = RESP / (decode_ms / 1000)

  NOTE ON METHODOLOGY DIFFERENCES:
  Lattice and MLX use exact tokenizer-padded prompts (iterative BASE repetition
  until the tokenizer confirms the target token count is reached). Ollama uses
  a character-count heuristic that overshoots — the reported context for ollama
  will be higher than the target (e.g. 1450 vs 1000). Additionally, ollama
  (llama.cpp Metal) appears to prefix-cache repeated filler text, so its
  reported TTFT and `prefill_tok_s` are NOT representative of fresh-token
  prefill throughput: they reflect cache lookup + overhead, not actual
  quadratic attention over new tokens. The raw ollama API fields
  (`load_ms`, `prompt_eval_ms`, `eval_ms`, `prompt_eval_count`) are stored
  in the JSON for diagnosability. The `prefill_tok_s` column is labelled
  `prefill_tok_s*` (asterisked) for ollama in table output.

  The lattice-vs-MLX decode comparison is apples-to-apples: same tokenizer,
  same prompt token count, same RESP budget, same methodology. Precision
  differs (lattice Q8 safetensors, MLX bf16), so lattice's decode gap is
  conservative: MLX runs ~2x heavier weights and is still faster, meaning
  the real gap is larger than the raw tok/s ratio implies.

Usage:
  uv run python3 scripts/bench_compare_1k.py              # default: ctx=1000
  uv run python3 scripts/bench_compare_1k.py --ctx 2000   # single context
  uv run python3 scripts/bench_compare_1k.py --sweep      # ctx in {1000,2000,4000}
  uv run python3 scripts/bench_compare_1k.py --ctx 1000 --runs 3

  make bench-agentic         # sweep all three contexts (equiv: --sweep)
  make bench-agentic-quick   # ctx=1000 only, 3 runs (fast sanity check)

Output:
  docs/bench_results/agentic_{ctx}tok.json   — machine-readable per context
  docs/bench_results/agentic_1k_compare.json — kept for backward compatibility
  Markdown table printed to stdout (copy-paste into PR description)

Lattice prereq: binary at target/release/bench_decode_ab, built with
  cargo build --release --bin bench_decode_ab -p lattice-inference --features "f16,metal-gpu"

MLX prereq: uv run python3 -c "import mlx_lm"
  Falls back to external JSON (docs/bench_results/agentic_workload.json)
  if mlx_lm is not importable — a prominent NOTICE is printed in that case.

Ollama prereq: ollama serve running + model present (qwen3.5:0.8b by default).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO = Path(__file__).parent.parent
BIN = REPO / "target" / "release" / "bench_decode_ab"
MODEL_DIR = Path.home() / ".lattice" / "models" / "qwen3.5-0.8b"
MLX_MODEL = "Qwen/Qwen3.5-0.8B"
OLLAMA_MODEL = "qwen3.5:0.8b"
RESP = 100
DEFAULT_RUNS = 5

# Filler text repeated to reach target context depth.  Same base as the Rust
# binary so the token counts are comparable when both are active.
BASE = (
    "The quick brown fox jumps over the lazy dog. "
    "Once upon a time in a land far away, there lived a wise old owl "
    "who knew many secrets. Every morning the sun rose over the "
    "mountains and cast long shadows across the quiet valley. "
)

SWEEP_CTXS = [1000, 2000, 4000]

# ---------------------------------------------------------------------------
# Lattice
# ---------------------------------------------------------------------------


def _lattice_one_run(n_tokens: int, ctx: int, runs: int) -> tuple[float, Optional[int]]:
    """Run bench_decode_ab, return (median_total_ms, actual_prompt_tokens)."""
    env = os.environ.copy()
    env.update(
        BENCH_N=str(n_tokens),
        BENCH_RUNS=str(runs),
        BENCH_PROMPT_TOKENS=str(ctx),
        LATTICE_MODEL_DIR=str(MODEL_DIR),
    )
    out = subprocess.run(
        [str(BIN)], env=env, capture_output=True, text=True, timeout=600
    )
    times: list[float] = []
    actual_prompt: Optional[int] = None
    for line in (out.stdout + out.stderr).splitlines():
        if "prompt_tokens=" in line:
            try:
                actual_prompt = int(line.split("prompt_tokens=")[1].split()[0])
            except (IndexError, ValueError):
                pass
        if line.startswith("RESULT"):
            for part in line.split():
                if part.startswith("total_ms="):
                    try:
                        times.append(float(part.split("=")[1]))
                    except ValueError:
                        pass
    if not times:
        raise RuntimeError(
            f"bench_decode_ab produced no RESULT lines.\n"
            f"stdout: {out.stdout}\nstderr: {out.stderr}"
        )
    return median(times), actual_prompt


def bench_lattice(ctx: int, runs: int) -> dict:
    print(f"\n--- Lattice (Q8 safetensors, pure-Rust Metal) @ ctx~{ctx} ---")
    if not BIN.exists():
        msg = (
            f"binary not found: {BIN}\n"
            "Build with: cargo build --release --bin bench_decode_ab "
            "-p lattice-inference --features \"f16,metal-gpu\""
        )
        print(f"  NOTICE: lattice unavailable — {msg}")
        return _unavailable("lattice", ctx, msg)

    ttft_ms, pt = _lattice_one_run(1, ctx, runs)
    total_ms, _ = _lattice_one_run(RESP, ctx, runs)
    actual_ctx = pt if pt is not None else ctx
    decode_ms = total_ms - ttft_ms
    return _mk("lattice", actual_ctx, ttft_ms, total_ms, decode_ms, live=True, runs=runs)


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


def _ollama_available() -> bool:
    try:
        import requests  # noqa: PLC0415
        requests.get("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _detect_ollama_quant(model: str) -> str:
    """Return quantization label from ollama API, e.g. 'Q8_0'."""
    try:
        import requests  # noqa: PLC0415
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        for m in r.json().get("models", []):
            if m["name"] == model:
                return m.get("details", {}).get("quantization_level", "")
    except Exception:
        pass
    return ""


def bench_ollama(ctx: int, runs: int, model: str = OLLAMA_MODEL) -> dict:
    quant = _detect_ollama_quant(model) or "gguf"
    print(f"\n--- Ollama ({quant}, llama.cpp Metal) @ ctx~{ctx} ---")
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        msg = "requests library not installed"
        print(f"  NOTICE: ollama unavailable — {msg}")
        return _unavailable("ollama", ctx, msg)

    if not _ollama_available():
        msg = "ollama serve not running on localhost:11434"
        print(f"  NOTICE: ollama unavailable — {msg}")
        return _unavailable("ollama", ctx, msg)

    url = "http://localhost:11434/api/generate"
    prompt = _make_prompt(ctx)

    # Warmup — evicts stale KV state, ensures model is loaded.
    try:
        requests.post(
            url,
            json={"model": model, "prompt": "hi", "stream": False,
                  "options": {"num_predict": 4}},
            timeout=120,
        )
    except Exception as exc:
        msg = f"ollama warmup failed: {exc}"
        print(f"  NOTICE: ollama unavailable — {msg}")
        return _unavailable("ollama", ctx, msg)

    ttfts: list[float] = []
    totals: list[float] = []
    load_mss: list[float] = []
    prompt_eval_mss: list[float] = []
    eval_mss: list[float] = []
    pcount: Optional[int] = None

    for i in range(runs):
        try:
            r = requests.post(
                url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": RESP, "temperature": 0, "seed": 42},
                },
                timeout=600,
            ).json()
        except Exception as exc:
            print(f"  run {i+1} failed: {exc}")
            continue

        # Durations in nanoseconds from ollama API.
        prompt_eval_ms = r.get("prompt_eval_duration", 0) / 1e6
        eval_ms = r.get("eval_duration", 0) / 1e6
        load_ms = r.get("load_duration", 0) / 1e6
        ttfts.append(load_ms + prompt_eval_ms)
        totals.append(load_ms + prompt_eval_ms + eval_ms)
        load_mss.append(load_ms)
        prompt_eval_mss.append(prompt_eval_ms)
        eval_mss.append(eval_ms)
        if pcount is None:
            pcount = r.get("prompt_eval_count")

    if not ttfts:
        msg = "all ollama runs failed"
        print(f"  NOTICE: ollama unavailable — {msg}")
        return _unavailable("ollama", ctx, msg)

    actual_ctx = pcount if pcount is not None else ctx
    ttft_ms = median(ttfts)
    total_ms = median(totals)
    row = _mk("ollama", actual_ctx, ttft_ms, total_ms, total_ms - ttft_ms,
              live=True, runs=runs)
    # Store raw ollama API timing breakdown for diagnosability.
    # NOTE: prompt_eval_ms here reflects llama.cpp prefix-cache lookup, NOT
    # fresh-token prefill — see module docstring for caveats on prefill_tok_s.
    row["ollama_breakdown"] = {
        "load_ms": round(median(load_mss), 1),
        "prompt_eval_ms": round(median(prompt_eval_mss), 1),
        "eval_ms": round(median(eval_mss), 1),
        "prompt_eval_count": pcount,
    }
    return row


# ---------------------------------------------------------------------------
# MLX
# ---------------------------------------------------------------------------


def _build_padded_prompt(target_tokens: int, tokenizer=None) -> str:
    """Return a string that tokenizes to approximately target_tokens tokens.

    If a tokenizer is provided, iteratively adds BASE repetitions until the
    token count meets or exceeds target_tokens (same logic as the Rust binary).
    Falls back to a ~4-chars/token character heuristic otherwise.
    """
    if tokenizer is not None:
        prompt = ""
        while len(tokenizer.encode(prompt)) < target_tokens:
            prompt += BASE
        return prompt
    # Heuristic fallback: English prose ~0.75 words/token, BASE is ~35 words.
    # So target_tokens * (1/0.75) / 35 BASE repetitions.
    reps = max(1, (target_tokens * 4) // (len(BASE.split()) * 3) + 1)
    return BASE * reps


def bench_mlx(ctx: int, runs: int, model: str = MLX_MODEL) -> dict:
    print(f"\n--- MLX (mlx_lm, Apple MLX) @ ctx~{ctx} ---")
    try:
        import mlx_lm  # noqa: PLC0415
        import mlx.core as mx  # noqa: PLC0415
    except ImportError as exc:
        # Fall back to external JSON if present.
        msg = f"mlx_lm not importable: {exc}"
        fallback = _mlx_from_json(ctx)
        if fallback is not None:
            print(
                f"\n  *** NOTICE: MLX numbers loaded from external JSON "
                f"(docs/bench_results/agentic_workload.json), NOT measured live.\n"
                f"  Reason: {msg}\n"
                f"  To get live numbers: uv add mlx-lm && uv run python3 scripts/bench_compare_1k.py\n"
            )
            fallback["source"] = "external_json"
            return fallback
        print(f"  NOTICE: mlx unavailable — {msg}")
        return _unavailable("mlx", ctx, msg)

    # Load model once (reused across runs).
    print(f"  Loading {model}...")
    t_load = time.perf_counter()
    try:
        mlx_model, tokenizer = mlx_lm.load(model)
    except Exception as exc:
        msg = f"mlx_lm.load({model!r}) failed: {exc}"
        print(f"  NOTICE: mlx unavailable — {msg}")
        return _unavailable("mlx", ctx, msg)
    print(f"  Loaded in {time.perf_counter() - t_load:.2f}s")

    # Build prompt padded to target context using the actual tokenizer.
    prompt = _build_padded_prompt(ctx, tokenizer=tokenizer)

    # Count actual prompt tokens.
    encoded = tokenizer.encode(prompt)
    actual_ctx = len(encoded) if hasattr(encoded, "__len__") else ctx

    # Warmup.
    mlx_lm.generate(
        mlx_model, tokenizer, prompt=BASE, max_tokens=4, verbose=False
    )
    mx.eval(mx.array([0]))  # flush Metal command queue

    ttfts: list[float] = []
    totals: list[float] = []

    for i in range(runs):
        # TTFT: generate exactly 1 token from the full padded prompt.
        t0 = time.perf_counter()
        mlx_lm.generate(
            mlx_model, tokenizer, prompt=prompt, max_tokens=1, verbose=False
        )
        mx.eval(mx.array([0]))
        ttft_ms = (time.perf_counter() - t0) * 1000

        # Total: generate RESP tokens from the same prompt.
        t0 = time.perf_counter()
        mlx_lm.generate(
            mlx_model, tokenizer, prompt=prompt, max_tokens=RESP, verbose=False
        )
        mx.eval(mx.array([0]))
        total_ms = (time.perf_counter() - t0) * 1000

        ttfts.append(ttft_ms)
        totals.append(total_ms)
        print(f"  run {i+1}/{runs}: TTFT {ttft_ms:.0f}ms, total {total_ms:.0f}ms")

    ttft_ms = median(ttfts)
    total_ms = median(totals)
    return _mk("mlx", actual_ctx, ttft_ms, total_ms, total_ms - ttft_ms, live=True, runs=runs)


def _mlx_from_json(ctx: int) -> Optional[dict]:
    """Load the closest MLX row from the legacy agentic_workload.json."""
    candidates = [
        REPO / "docs" / "bench_results" / "agentic_workload.json",
        REPO / "docs" / "bench_results" / "agentic_1k_compare.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        for row in data:
            if row.get("engine") == "mlx" and abs(row.get("context", 0) - ctx) <= 100:
                return dict(row)
    return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_prompt(n_tokens_approx: int) -> str:
    """Repeat BASE to approximate a token count (via character-count heuristic)."""
    reps = max(1, n_tokens_approx * 4 // (len(BASE.split()) * 3))
    return BASE * reps


def _mk(engine: str, ctx, ttft_ms: float, total_ms: float, decode_ms: float,
        live: bool = True, runs: int = DEFAULT_RUNS) -> dict:
    decode_tok_s = RESP / (decode_ms / 1000) if decode_ms > 0 else 0
    prefill_tok_s = ctx / (ttft_ms / 1000) if ttft_ms > 0 else 0
    row = {
        "engine": engine,
        "context": ctx,
        "response": RESP,
        "runs": runs,
        "ttft_ms": round(ttft_ms, 1),
        "decode_ms": round(decode_ms, 1),
        "total_ms": round(total_ms, 1),
        "prefill_tok_s": round(prefill_tok_s),
        "decode_tok_s": round(decode_tok_s),
        "source": "live" if live else "unavailable",
    }
    print(
        f"  ctx={ctx} tok | TTFT {ttft_ms:.0f}ms ({prefill_tok_s:.0f} tok/s)"
        f" | decode {decode_ms:.0f}ms ({decode_tok_s:.0f} tok/s) | total {total_ms:.0f}ms"
    )
    return row


def _unavailable(engine: str, ctx: int, reason: str, runs: int = 0) -> dict:
    return {
        "engine": engine,
        "context": ctx,
        "response": RESP,
        "runs": runs,
        "ttft_ms": None,
        "decode_ms": None,
        "total_ms": None,
        "prefill_tok_s": None,
        "decode_tok_s": None,
        "source": "unavailable",
        "unavailable_reason": reason,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_COL = {
    "engine": 12,
    "context": 7,
    "ttft_ms": 11,
    "decode_ms": 13,
    "total_ms": 12,
    "prefill_tok_s": 13,
    "decode_tok_s": 12,
    "source": 10,
}


def _fmt_val(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.0f}"
    return str(v)


def print_table(rows: list[dict], ctx: int) -> str:
    """Print and return a markdown table for the given context depth.

    Ollama's prefill_tok_s is asterisked (*) because it reflects
    prefix-cache lookup rather than fresh-token prefill throughput.
    Precision note is included below the table.
    """
    header = (
        f"\n## Agentic Workload: ~{ctx}-token context, {RESP}-token response\n\n"
        "| Engine | Precision | Ctx (tok) | TTFT (ms) | Decode (ms) | Total (ms) "
        "| Prefill t/s | Decode t/s | Runs | Source |\n"
        "|--------|-----------|-----------|-----------|-------------|-----------|"
        "-------------|------------|------|--------|\n"
    )
    _PRECISION = {
        "lattice": "Q8 sf",
        "ollama": "Q8_0",
        "mlx": "bf16",
    }
    lines = [header]
    for r in rows:
        engine = r["engine"]
        precision = _PRECISION.get(engine, "?")
        # Mark ollama prefill as a cache artifact, not fresh-prefill throughput.
        prefill_val = _fmt_val(r["prefill_tok_s"])
        if engine == "ollama" and r["prefill_tok_s"] is not None:
            prefill_val = prefill_val + "*"
        line = (
            f"| {engine:<10} "
            f"| {precision:<9} "
            f"| {_fmt_val(r['context']):>9} "
            f"| {_fmt_val(r['ttft_ms']):>9} "
            f"| {_fmt_val(r['decode_ms']):>11} "
            f"| {_fmt_val(r['total_ms']):>9} "
            f"| {prefill_val:>11} "
            f"| {_fmt_val(r['decode_tok_s']):>10} "
            f"| {r.get('runs', '?'):>4} "
            f"| {r.get('source','?'):<6} |\n"
        )
        lines.append(line)
    footnote = (
        "\n*ollama prefill tok/s reflects llama.cpp prefix-cache lookup (repeated filler text), "
        "NOT fresh-token prefill throughput. See module docstring.\n"
        "Precision note: MLX runs bf16 (~2x memory bandwidth vs lattice Q8). "
        "The lattice decode gap is conservative — bandwidth-adjusted, the gap is larger.\n"
    )
    lines.append(footnote)
    table = "".join(lines)
    print(table)
    return table


# ---------------------------------------------------------------------------
# Per-context run
# ---------------------------------------------------------------------------


def run_one_ctx(ctx: int, runs: int) -> tuple[list[dict], str]:
    rows: list[dict] = []

    rows.append(bench_lattice(ctx, runs))
    rows.append(bench_ollama(ctx, runs))
    rows.append(bench_mlx(ctx, runs))

    table = print_table(rows, ctx)

    out_dir = REPO / "docs" / "bench_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"agentic_{ctx}tok.json"
    out_file.write_text(json.dumps(rows, indent=2))
    print(f"Raw JSON: {out_file}")

    # Backward-compatibility alias for ctx≈1000.
    if abs(ctx - 1000) <= 50:
        compat = out_dir / "agentic_1k_compare.json"
        compat.write_text(json.dumps(rows, indent=2))

    return rows, table


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic-workload benchmark: lattice vs ollama vs MLX"
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=1000,
        help="Context depth in tokens (default: 1000).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep contexts {1000, 2000, 4000} and save one JSON per context.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Repetitions per data point (default: {DEFAULT_RUNS}; median reported).",
    )
    args = parser.parse_args()

    ctxs = SWEEP_CTXS if args.sweep else [args.ctx]
    all_rows: list[dict] = []

    for ctx in ctxs:
        rows, _ = run_one_ctx(ctx, args.runs)
        all_rows.extend(rows)

    if args.sweep:
        sweep_file = REPO / "docs" / "bench_results" / "agentic_sweep.json"
        sweep_file.write_text(json.dumps(all_rows, indent=2))
        print(f"\nSweep JSON: {sweep_file}")

    # Summary notice for any frameworks that ran from external JSON or were unavailable.
    notices = [
        r for r in all_rows
        if r.get("source") not in ("live",)
    ]
    if notices:
        print("\n*** FRAMEWORK STATUS NOTICES ***")
        for r in notices:
            engine = r["engine"]
            src = r.get("source", "?")
            reason = r.get("unavailable_reason", "")
            if src == "external_json":
                print(
                    f"  {engine}: numbers loaded from external JSON (not measured live this run)\n"
                    f"           To measure live: uv add mlx-lm"
                )
            elif src == "unavailable":
                print(f"  {engine}: UNAVAILABLE — {reason}")


if __name__ == "__main__":
    main()
