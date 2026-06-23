#!/usr/bin/env python3
"""Agentic-workload speed comparison at 1000-token context / 100-token response.

Lattice  — real Metal e2e path via bench_decode_ab (prompt padded to ~1000 tok).
Ollama   — /api/generate with the same workload, timings from the API.
MLX      — measured separately by bench_agentic_workload.py (loaded from JSON).

Methodology (identical across engines):
  TTFT   = prefill latency (generate 1 token)
  Total  = prefill + 100-token decode
  Decode = Total - TTFT  ->  decode_tok_s = 100 / (decode_ms/1000)
"""
import json
import os
import subprocess
import time
from pathlib import Path
from statistics import median

import requests

REPO = Path(__file__).parent.parent
BIN = REPO / "target" / "release" / "bench_decode_ab"
MODEL_DIR = Path.home() / ".lattice" / "models" / "qwen3.5-0.8b"
CTX = 1000
RESP = 100
RUNS = 5

BASE = (
    "The quick brown fox jumps over the lazy dog. "
    "Once upon a time in a land far away, there lived a wise old owl "
    "who knew many secrets. Every morning the sun rose over the "
    "mountains and cast long shadows across the quiet valley. "
)


def lattice_total_ms(n_tokens):
    env = os.environ.copy()
    env.update(
        BENCH_N=str(n_tokens),
        BENCH_RUNS=str(RUNS),
        BENCH_PROMPT_TOKENS=str(CTX),
        LATTICE_MODEL_DIR=str(MODEL_DIR),
    )
    out = subprocess.run(
        [str(BIN)], env=env, capture_output=True, text=True, timeout=300
    )
    times = []
    actual_prompt = None
    for line in (out.stdout + out.stderr).splitlines():
        if "prompt_tokens=" in line:
            actual_prompt = int(line.split("prompt_tokens=")[1].split()[0])
        if line.startswith("RESULT"):
            for p in line.split():
                if p.startswith("total_ms="):
                    times.append(float(p.split("=")[1]))
    if not times:
        raise RuntimeError(f"no lattice results:\n{out.stdout}\n{out.stderr}")
    return median(times), actual_prompt


def bench_lattice():
    print("\n─── Lattice (Q8 safetensors, pure-Rust Metal) ───")
    ttft, pt = lattice_total_ms(1)
    total, _ = lattice_total_ms(RESP)
    decode_ms = total - ttft
    return mk("lattice", pt, ttft, total, decode_ms)


def make_prompt(n_tokens_approx):
    # ~0.75 words/token heuristic; ollama will report the true prompt count.
    reps = max(1, n_tokens_approx * 4 // (len(BASE.split()) * 3))
    return BASE * reps


def bench_ollama(model="qwen3.5:0.8b"):
    print("\n─── Ollama (Q4_K_M, llama.cpp Metal) ───")
    url = "http://localhost:11434/api/generate"
    prompt = make_prompt(CTX)
    # warmup
    requests.post(url, json={"model": model, "prompt": "hi", "stream": False,
                             "options": {"num_predict": 4}}, timeout=120)
    ttfts, totals, pcount = [], [], None
    for _ in range(RUNS):
        r = requests.post(url, json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"num_predict": RESP, "temperature": 0, "seed": 42},
        }, timeout=300).json()
        prompt_eval = r.get("prompt_eval_duration", 0) / 1e6  # ns -> ms
        eval_dur = r.get("eval_duration", 0) / 1e6
        load = r.get("load_duration", 0) / 1e6
        ttfts.append(load + prompt_eval)
        totals.append(load + prompt_eval + eval_dur)
        pcount = r.get("prompt_eval_count", pcount)
    ttft, total = median(ttfts), median(totals)
    return mk("ollama", pcount, ttft, total, total - ttft)


def mk(engine, ctx, ttft, total, decode_ms):
    decode_tok_s = RESP / (decode_ms / 1000) if decode_ms > 0 else 0
    prefill_tok_s = ctx / (ttft / 1000) if ttft > 0 else 0
    row = {
        "engine": engine, "context": ctx, "response": RESP,
        "ttft_ms": round(ttft, 1), "decode_ms": round(decode_ms, 1),
        "total_ms": round(total, 1),
        "prefill_tok_s": round(prefill_tok_s), "decode_tok_s": round(decode_tok_s),
    }
    print(f"    context: {ctx} tok | TTFT {ttft:.0f}ms ({prefill_tok_s:.0f} tok/s) "
          f"| decode {decode_ms:.0f}ms ({decode_tok_s:.0f} tok/s) | total {total:.0f}ms")
    return row


def main():
    rows = [bench_lattice(), bench_ollama()]

    # Pull MLX @ ctx≈1000 from the prior measurement.
    mlx_json = REPO / "docs" / "bench_results" / "agentic_workload.json"
    if mlx_json.exists():
        for r in json.loads(mlx_json.read_text()):
            if r["engine"] == "mlx" and abs(r["context"] - CTX) <= 50:
                rows.append(r)
                break

    print("\n═══ Agentic Workload: 1000-token context, 100-token response ═══\n")
    h = f"{'Engine':<10}{'Ctx':>6}{'TTFT(ms)':>10}{'Decode(ms)':>12}{'Total(ms)':>11}{'Prefill t/s':>13}{'Decode t/s':>12}"
    print(h)
    print("-" * len(h))
    for r in rows:
        print(f"{r['engine']:<10}{r['context']:>6}{r['ttft_ms']:>10.0f}"
              f"{r['decode_ms']:>12.0f}{r['total_ms']:>11.0f}"
              f"{r['prefill_tok_s']:>13.0f}{r['decode_tok_s']:>12.0f}")

    out = REPO / "docs" / "bench_results" / "agentic_1k_compare.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nRaw: {out}")


if __name__ == "__main__":
    main()
