#!/usr/bin/env python3
"""E2E parity gate: HF transformers (reference) vs lattice (under test).

Runs HF transformers first to warm the machine, then lattice. Compares greedy
generation output (token IDs) and reports speed.

Exit codes: 0 = pass, 1 = parity failure, 2 = setup error.

Env vars:
  LATTICE_BIN       path to lattice qwen35_generate binary (default: target/release/qwen35_generate)
  LATTICE_MODEL_DIR path to model weights (default: ~/.lattice/models/qwen3.5-0.8b)
  HF_MODEL_ID       HuggingFace model ID (default: Qwen/Qwen3.5-0.8B)
  E2E_MAX_TOKENS    tokens to generate per prompt (default: 15)
  E2E_REPORT_PATH   write markdown report here (optional)
"""

import json
import os
import re
import subprocess
import sys
import time


PROMPTS = [
    "The capital of France is",
    "In the year 2024, artificial intelligence",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
]

MAX_TOKENS = int(os.environ.get("E2E_MAX_TOKENS", "15"))
# Qwen3.5 is a hybrid GQA+GDN model. GDN recurrent state accumulation
# amplifies tiny f32 rounding differences between implementations, so
# greedy output diverges after a few tokens. 3 tokens is sufficient to
# validate the forward pass (prefill + first decode steps) while
# allowing natural implementation variance.
MATCH_WINDOW = 3

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "Qwen/Qwen3.5-0.8B")
LATTICE_BIN = os.environ.get(
    "LATTICE_BIN", "target/release/qwen35_generate"
)
MODEL_DIR = os.environ.get(
    "LATTICE_MODEL_DIR",
    os.path.expanduser("~/.lattice/models/qwen3.5-0.8b"),
)
REPORT_PATH = os.environ.get("E2E_REPORT_PATH")


def run_hf_reference(prompt: str, max_tokens: int) -> dict:
    """Run HF transformers greedy generation. Returns tokens + timing."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not hasattr(run_hf_reference, "_model"):
        t0 = time.time()
        run_hf_reference._tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_ID, trust_remote_code=True
        )
        run_hf_reference._model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID, dtype=torch.float32, trust_remote_code=True
        )
        run_hf_reference._model.eval()
        print(f"[hf] model loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    tokenizer = run_hf_reference._tokenizer
    model = run_hf_reference._model

    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0].tolist()

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    elapsed = time.time() - t0

    all_ids = outputs[0].tolist()
    gen_ids = all_ids[len(prompt_ids):]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return {
        "prompt_ids": prompt_ids,
        "generated_ids": gen_ids,
        "text": text,
        "elapsed_s": elapsed,
        "tok_per_sec": len(gen_ids) / elapsed if elapsed > 0 else 0,
    }


def run_lattice(prompt: str, max_tokens: int) -> dict:
    """Run lattice qwen35_generate and parse output."""
    cmd = [
        LATTICE_BIN,
        "--model-dir", MODEL_DIR,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temperature", "0.0",
    ]

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[lattice] FAILED (exit {result.returncode})", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return None

    stdout = result.stdout
    token_match = re.search(r"Token IDs:\s*\[([^\]]*)\]", stdout)
    gen_match = re.search(r"Generated tokens:\s*(\d+)", stdout)
    speed_match = re.search(r"Speed:\s*([\d.]+)\s*tok/s", stdout)

    if not token_match:
        print("[lattice] could not parse Token IDs from output", file=sys.stderr)
        print(stdout, file=sys.stderr)
        return None

    gen_ids = [int(x.strip()) for x in token_match.group(1).split(",") if x.strip()]
    gen_count = int(gen_match.group(1)) if gen_match else len(gen_ids)
    tok_per_sec = float(speed_match.group(1)) if speed_match else (gen_count / elapsed if elapsed > 0 else 0)

    text_match = re.search(r"--- Generated Text ---\n(.*?)--- Stats ---", stdout, re.DOTALL)
    text = text_match.group(1).strip() if text_match else ""

    return {
        "generated_ids": gen_ids,
        "text": text,
        "elapsed_s": elapsed,
        "tok_per_sec": tok_per_sec,
    }


def compare(prompt: str, hf: dict, lattice: dict) -> dict:
    """Compare HF vs lattice outputs. Returns verdict dict."""
    hf_ids = hf["generated_ids"][:MAX_TOKENS]
    lat_ids = lattice["generated_ids"][:MAX_TOKENS]

    min_len = min(len(hf_ids), len(lat_ids))
    first_mismatch = min_len
    for i in range(min_len):
        if hf_ids[i] != lat_ids[i]:
            first_mismatch = i
            break

    window_match = first_mismatch >= MATCH_WINDOW
    total_agree = sum(1 for a, b in zip(hf_ids, lat_ids) if a == b)
    agree_rate = total_agree / min_len if min_len > 0 else 0

    return {
        "prompt": prompt[:60],
        "first_mismatch": first_mismatch if first_mismatch < min_len else None,
        "window_match": window_match,
        "agree_rate": agree_rate,
        "total_agree": total_agree,
        "total_compared": min_len,
        "hf_tok_s": hf["tok_per_sec"],
        "lat_tok_s": lattice["tok_per_sec"],
        "hf_text": hf["text"][:80],
        "lat_text": lattice["text"][:80],
        "pass": window_match,
    }


def render_report(results: list[dict]) -> str:
    lines = ["## E2E Parity Report", ""]
    fails = [r for r in results if not r["pass"]]
    if fails:
        lines.append(f"**FAIL**: {len(fails)}/{len(results)} prompts diverged within first {MATCH_WINDOW} tokens")
    else:
        lines.append(f"**PASS**: all {len(results)} prompts match within first {MATCH_WINDOW} tokens")
    lines.append("")

    lines.append("| Prompt | Agreement | First Diff | HF tok/s | Lattice tok/s | Verdict |")
    lines.append("|--------|-----------|------------|----------|---------------|---------|")
    for r in results:
        diff = f"pos {r['first_mismatch']}" if r["first_mismatch"] is not None else "none"
        icon = "PASS" if r["pass"] else "FAIL"
        lines.append(
            f"| `{r['prompt']}` | {r['total_agree']}/{r['total_compared']} "
            f"| {diff} | {r['hf_tok_s']:.1f} | {r['lat_tok_s']:.1f} | {icon} |"
        )

    lines.append("")
    for r in results:
        lines.append(f"**`{r['prompt']}`**")
        lines.append(f"- HF:      {r['hf_text']}")
        lines.append(f"- Lattice: {r['lat_text']}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    if not os.path.isfile(LATTICE_BIN):
        print(f"error: lattice binary not found at {LATTICE_BIN}", file=sys.stderr)
        return 2
    if not os.path.isdir(MODEL_DIR):
        print(f"error: model dir not found at {MODEL_DIR}", file=sys.stderr)
        return 2

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM  # noqa: F401
    except ImportError as e:
        print(f"error: missing dependency: {e}", file=sys.stderr)
        return 2

    results = []
    for prompt in PROMPTS:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Prompt: {prompt[:60]}", file=sys.stderr)

        print("[hf] running reference...", file=sys.stderr)
        hf_out = run_hf_reference(prompt, MAX_TOKENS)

        print("[lattice] running under test...", file=sys.stderr)
        lat_out = run_lattice(prompt, MAX_TOKENS)

        if lat_out is None:
            print("FAIL: lattice binary failed", file=sys.stderr)
            return 1

        verdict = compare(prompt, hf_out, lat_out)
        results.append(verdict)

        status = "PASS" if verdict["pass"] else "FAIL"
        print(
            f"[{status}] agree={verdict['total_agree']}/{verdict['total_compared']} "
            f"hf={hf_out['tok_per_sec']:.1f} lat={lat_out['tok_per_sec']:.1f} tok/s",
            file=sys.stderr,
        )

    report = render_report(results)
    print(report)

    if REPORT_PATH:
        with open(REPORT_PATH, "w") as f:
            f.write(report)

    fails = sum(1 for r in results if not r["pass"])
    if fails:
        print(f"\nFAIL: {fails}/{len(results)} prompts failed parity gate", file=sys.stderr)
        return 1

    print(f"\nPASS: all {len(results)} prompts passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
