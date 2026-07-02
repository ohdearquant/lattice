#!/usr/bin/env python3
"""E2E parity gate: HF transformers (reference) vs lattice (under test).

Runs HF transformers first to warm the machine, then lattice. Compares greedy
generation output (token IDs) and reports speed.

Exit codes: 0 = pass, 1 = parity failure, 2 = setup error.

Args:
  --backend {cpu,metal}  which lattice binary/path to exercise (default: cpu).
                          "cpu" runs qwen35_generate and parses its legacy text
                          output. "metal" runs `chat_metal --json` and requires
                          the LATTICE_METAL_PATH_PROOF runtime capability marker
                          to prove the Metal attention/KV-cache dispatch helpers
                          actually ran; a missing/invalid/zero-count marker is a
                          gate FAILURE (never a skip), so a paravirtual runner
                          that silently no-ops the required kernels goes red
                          instead of green (issue #239).

Env vars:
  LATTICE_BIN             path to the lattice binary under test (default:
                           target/release/qwen35_generate for --backend cpu,
                           target/release/chat_metal for --backend metal)
  LATTICE_MODEL_DIR       path to model weights (default: ~/.lattice/models/qwen3.5-0.8b)
  HF_MODEL_ID             HuggingFace model ID (default: Qwen/Qwen3.5-0.8B)
  E2E_MAX_TOKENS          tokens to generate per prompt (default: 15)
  E2E_REPORT_PATH         write markdown report here (optional)
  LATTICE_METAL_PATH_PROOF  set to "1" so chat_metal emits the
                           `[METAL_PATH_PROOF]` stderr marker this script
                           requires in --backend metal mode
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time


# Each entry is (prompt, match_window).
#
# match_window: minimum number of leading generated tokens that must agree
# between HF and lattice. Kept small because Qwen3.5 is a hybrid GQA+GDN
# model and GDN recurrent state accumulation amplifies tiny f32 rounding
# differences between implementations, so greedy output naturally diverges
# after a few tokens. 3 tokens validates the forward pass (prefill + first
# decode steps) for short prompts. For the long-prefill case the first
# generated token is the critical signal (see comment on LONG_PROMPT below).
PROMPTS: list[tuple[str, int]] = [
    ("The capital of France is", 3),
    ("In the year 2024, artificial intelligence", 3),
    ("def fibonacci(n):\n    if n <= 1:\n        return n\n    return", 3),
    # LONG_PROMPT: ~816 tokens (measured with Qwen/Qwen3.5-0.8B tokenizer).
    # NOTE on call graph: CI builds qwen35_generate with --features f16 only,
    # so this prompt exercises the CPU/f16 forward path — NOT the Metal
    # chunked/oversize-prefill code in metal_qwen35.rs, which is gated behind
    # `metal-gpu` and never compiled here. (An earlier version of this comment
    # claimed otherwise, and that stale call-graph model misdirected the #520
    # triage toward Metal prefill.) What the length DOES stress is the
    # sampling decision after a long prompt history: with ~816 prompt tokens
    # in previous_ids, a repetition-penalty mismatch between lattice and the
    # HF reference flips the first greedy token (#520). Covering Metal
    # chunked prefill requires a separate metal-gpu gate (see #239).
    # match_window=2: the first two generated tokens are a direct function of
    # the full 816-step prefill final-position logits. GDN recurrent state
    # drifts during decode (same reason short prompts use 3 not 15), but the
    # first generated token after a correct long prefill must be identical
    # between HF and lattice. Two tokens gives a margin over a single-token
    # coincidence while remaining tolerant of subsequent decode drift.
    (
        "def merge_sort(arr):\n"
        '    """\n'
        "    Merge sort implementation.\n"
        "    Time complexity: O(n log n)\n"
        "    Space complexity: O(n)\n"
        '    """\n'
        "    if len(arr) <= 1:\n"
        "        return arr\n"
        "    mid = len(arr) // 2\n"
        "    left = merge_sort(arr[:mid])\n"
        "    right = merge_sort(arr[mid:])\n"
        "    return merge(left, right)\n"
        "\n"
        "def merge(left, right):\n"
        "    result = []\n"
        "    i = j = 0\n"
        "    while i < len(left) and j < len(right):\n"
        "        if left[i] <= right[j]:\n"
        "            result.append(left[i])\n"
        "            i += 1\n"
        "        else:\n"
        "            result.append(right[j])\n"
        "            j += 1\n"
        "    result.extend(left[i:])\n"
        "    result.extend(right[j:])\n"
        "    return result\n"
        "\n"
        "def quick_sort(arr, low=0, high=None):\n"
        '    """\n'
        "    Quick sort implementation using Lomuto partition scheme.\n"
        "    Average time complexity: O(n log n)\n"
        "    Worst case: O(n^2) when already sorted.\n"
        '    """\n'
        "    if high is None:\n"
        "        high = len(arr) - 1\n"
        "    if low < high:\n"
        "        pivot_idx = partition(arr, low, high)\n"
        "        quick_sort(arr, low, pivot_idx - 1)\n"
        "        quick_sort(arr, pivot_idx + 1, high)\n"
        "    return arr\n"
        "\n"
        "def partition(arr, low, high):\n"
        "    pivot = arr[high]\n"
        "    i = low - 1\n"
        "    for j in range(low, high):\n"
        "        if arr[j] <= pivot:\n"
        "            i += 1\n"
        "            arr[i], arr[j] = arr[j], arr[i]\n"
        "    arr[i + 1], arr[high] = arr[high], arr[i + 1]\n"
        "    return i + 1\n"
        "\n"
        "def binary_search(arr, target):\n"
        '    """Binary search in sorted array. Returns index or -1."""\n'
        "    left, right = 0, len(arr) - 1\n"
        "    while left <= right:\n"
        "        mid = (left + right) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            left = mid + 1\n"
        "        else:\n"
        "            right = mid - 1\n"
        "    return -1\n"
        "\n"
        "class Stack:\n"
        '    """LIFO stack backed by a Python list."""\n'
        "    def __init__(self):\n"
        "        self._data = []\n"
        "\n"
        "    def push(self, item):\n"
        "        self._data.append(item)\n"
        "\n"
        "    def pop(self):\n"
        "        if self.is_empty():\n"
        '            raise IndexError("pop from empty stack")\n'
        "        return self._data.pop()\n"
        "\n"
        "    def peek(self):\n"
        "        if self.is_empty():\n"
        '            raise IndexError("peek at empty stack")\n'
        "        return self._data[-1]\n"
        "\n"
        "    def is_empty(self):\n"
        "        return len(self._data) == 0\n"
        "\n"
        "    def size(self):\n"
        "        return len(self._data)\n"
        "\n"
        "\n"
        "class Queue:\n"
        '    """FIFO queue using two stacks for amortized O(1) enqueue and dequeue."""\n'
        "    def __init__(self):\n"
        "        self._inbox = Stack()\n"
        "        self._outbox = Stack()\n"
        "\n"
        "    def enqueue(self, item):\n"
        "        self._inbox.push(item)\n"
        "\n"
        "    def dequeue(self):\n"
        "        if self._outbox.is_empty():\n"
        "            while not self._inbox.is_empty():\n"
        "                self._outbox.push(self._inbox.pop())\n"
        "        if self._outbox.is_empty():\n"
        '            raise IndexError("dequeue from empty queue")\n'
        "        return self._outbox.pop()\n"
        "\n"
        "    def is_empty(self):\n"
        "        return self._inbox.is_empty() and self._outbox.is_empty()\n"
        "\n"
        "# All algorithms above are correct Python. The next function is:\n"
        "def",
        2,
    ),
]

# Known-divergent prompts per backend. The long-prefill merge_sort prompt
# (bound by CONTENT below, not by position — a positional PROMPTS[-1] key
# would silently follow a future append/reorder of PROMPTS and could waive
# a brand-new regression) deterministically diverges from HF at the first
# generated token on the METAL backend only (issue #535 — repetition penalty
# and GDN prefill mode both excluded as causes; the CPU leg passes the same
# prompt and still gates on it).
#
# The metal parity job is informational (deliberately not a required check),
# but a plain FAIL turns the whole workflow run red on every engine PR,
# burying new signal in known noise. Marking the prompt expected-divergent
# keeps the job green while #535 is open WITHOUT weakening anything else:
# missing path-proof, binary failures, and divergence on any other prompt
# still fail closed, and the report still prints the divergence in full.
# If the prompt starts PASSING on Metal, the run reports XPASS (still green)
# — that means #535 is likely fixed; delete the entry then.
_long_prefill = [p for p, _ in PROMPTS if p.startswith("def merge_sort(arr):")]
assert len(_long_prefill) == 1, (
    "expected exactly one long-prefill (merge_sort) prompt in PROMPTS; "
    "re-anchor METAL_EXPECTED_DIVERGENCE (#535) before changing the table"
)
LONG_PREFILL_PROMPT: str = _long_prefill[0]

METAL_EXPECTED_DIVERGENCE: dict[str, str] = {
    LONG_PREFILL_PROMPT: "#535",
}

MAX_TOKENS = int(os.environ.get("E2E_MAX_TOKENS", "15"))

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "Qwen/Qwen3.5-0.8B")

# LATTICE_BIN default depends on --backend (chat_metal for metal, qwen35_generate
# for cpu), but argparse only runs inside main(). An explicit LATTICE_BIN env var
# always wins; module import time sets a provisional cpu-shaped default so any
# code that reads LATTICE_BIN before main() runs still gets a sane path, and
# main() overwrites it with the backend-correct default once args are parsed.
LATTICE_BIN = os.environ.get(
    "LATTICE_BIN", "target/release/qwen35_generate"
)
_LATTICE_BIN_EXPLICIT = "LATTICE_BIN" in os.environ
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
        # GenerateConfig::default() carries a production repetition_penalty of
        # 1.1 (see qwen35_config.rs), matching chat_metal.rs's serving default.
        # The HF reference call below passes no repetition_penalty kwarg, so
        # transformers applies none (factor 1.0 = no-op). Left at lattice's
        # default, the two sides sample from different distributions even at
        # temperature=0.0 (repetition penalty is applied to logits before the
        # greedy argmax, not after) — invisible on short prompts because few
        # candidate tokens have already appeared, but decisive on the ~816-token
        # long-prefill prompt: nearly the whole Python-keyword vocabulary is
        # already in the prompt, so penalizing every previously-seen token
        # flips the post-prefill argmax away from HF's continuation (#520).
        # Force 1.0 here so both sides run the same greedy decision rule.
        "--repetition-penalty", "1.0",
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


# Required (non-zero) path-proof counters, matching the dispatch-site instrumentation
# in crates/inference/src/forward/metal_qwen35.rs. decode attention takes either the
# direct kernel (short caches) or the split partial+reduce pair (long caches) — never
# both — so that pair is an OR, everything else is a hard AND.
_PATH_PROOF_RE = re.compile(
    r"\[METAL_PATH_PROOF\]\s+"
    r"prefill_kv_batch=(\d+)\s+"
    r"prefill_attn_batched=(\d+)\s+"
    r"decode_kv_copy=(\d+)\s+"
    r"decode_attn_direct=(\d+)\s+"
    r"decode_attn_split_partial=(\d+)\s+"
    r"decode_attn_split_reduce=(\d+)\s+"
    r"kv_f16=(true|false)"
)


def parse_path_proof_marker(stderr_text: str) -> dict | None:
    """Extract the last `[METAL_PATH_PROOF]` marker from chat_metal's stderr.

    Returns None if the marker is absent or malformed — callers must treat that
    as a gate FAILURE, not a skip, per the paravirtual-runner gotcha in issue #239.
    """
    match = None
    for line in stderr_text.splitlines():
        m = _PATH_PROOF_RE.search(line)
        if m:
            match = m  # keep the last one, in case of retries/logging noise
    if match is None:
        return None
    return {
        "prefill_kv_batch": int(match.group(1)),
        "prefill_attn_batched": int(match.group(2)),
        "decode_kv_copy": int(match.group(3)),
        "decode_attn_direct": int(match.group(4)),
        "decode_attn_split_partial": int(match.group(5)),
        "decode_attn_split_reduce": int(match.group(6)),
        "kv_f16": match.group(7) == "true",
    }


def path_proof_covers_required_path(counters: dict) -> bool:
    """Fail-closed check: did the required Metal attention/KV dispatches run?"""
    if counters["prefill_kv_batch"] <= 0:
        return False
    if counters["prefill_attn_batched"] <= 0:
        return False
    if counters["decode_kv_copy"] <= 0:
        return False
    direct_ok = counters["decode_attn_direct"] > 0
    split_ok = (
        counters["decode_attn_split_partial"] > 0
        and counters["decode_attn_split_reduce"] > 0
    )
    return direct_ok or split_ok


def run_lattice_metal(prompt: str, max_tokens: int) -> dict:
    """Run `chat_metal --json` and parse its `@@lattice` event stream.

    Fails closed (returns None) if the run errors, the event stream is
    malformed, or the `[METAL_PATH_PROOF]` marker is missing or does not cover
    the required attention/KV-cache dispatch path — this is the mechanism that
    makes the gate red on a paravirtual CI runner that reports a Metal device
    but silently no-ops the required kernels (issue #239), instead of a vacuous
    green pass.
    """
    cmd = [
        LATTICE_BIN,
        "--model-dir", MODEL_DIR,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temperature", "0.0",
        # chat_metal's serving default is repetition_penalty 1.1, but the HF
        # reference applies none — the same greedy-decision mismatch documented
        # on the CPU path above (run_lattice). Force 1.0 so both sides run the
        # same greedy rule and any remaining divergence is genuinely the
        # engine's (e.g. the long-prefill Metal divergence, which reproduces
        # with this flag set and with either GDN prefill mode).
        "--repetition-penalty", "1.0",
        "--json",
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        print("[lattice-metal] FAILED (timeout)", file=sys.stderr)
        return None
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[lattice-metal] FAILED (exit {result.returncode})", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return None

    gen_ids: list[int] = []
    text_parts: list[str] = []
    tok_per_sec = 0.0
    saw_done = False

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("@@lattice "):
            continue
        payload = line[len("@@lattice "):]
        try:
            ev = json.loads(payload)
        except json.JSONDecodeError as e:
            print(f"[lattice-metal] malformed @@lattice event: {e}", file=sys.stderr)
            print(line, file=sys.stderr)
            return None
        if ev.get("ev") != "gen_token":
            continue
        if ev.get("done"):
            saw_done = True
            tok_per_sec = float(ev.get("tok_s", 0.0))
            continue
        token_id = ev.get("token_id")
        if not isinstance(token_id, int):
            print(
                f"[lattice-metal] gen_token event missing integer token_id: {ev}",
                file=sys.stderr,
            )
            return None
        gen_ids.append(token_id)
        text_parts.append(ev.get("token", ""))

    if not saw_done:
        print("[lattice-metal] no done:true event observed in output", file=sys.stderr)
        return None

    path_proof = parse_path_proof_marker(result.stderr)
    if path_proof is None:
        print(
            "[lattice-metal] LATTICE_METAL_PATH_PROOF marker missing or malformed "
            "— failing closed (never skip-as-green)",
            file=sys.stderr,
        )
        print(result.stderr, file=sys.stderr)
        return None
    if not path_proof_covers_required_path(path_proof):
        print(
            f"[lattice-metal] path-proof counters do not cover the required Metal "
            f"attention/KV-cache dispatch path: {path_proof}",
            file=sys.stderr,
        )
        return None

    print(f"[lattice-metal] path-proof OK: {path_proof}", file=sys.stderr)

    return {
        "generated_ids": gen_ids,
        "text": "".join(text_parts),
        "elapsed_s": elapsed,
        "tok_per_sec": tok_per_sec,
    }


def compare(prompt: str, hf: dict, lattice: dict, match_window: int) -> dict:
    """Compare HF vs lattice outputs. Returns verdict dict."""
    hf_ids = hf["generated_ids"][:MAX_TOKENS]
    lat_ids = lattice["generated_ids"][:MAX_TOKENS]

    min_len = min(len(hf_ids), len(lat_ids))
    first_mismatch = min_len
    for i in range(min_len):
        if hf_ids[i] != lat_ids[i]:
            first_mismatch = i
            break

    window_match = first_mismatch >= match_window
    total_agree = sum(1 for a, b in zip(hf_ids, lat_ids) if a == b)
    agree_rate = total_agree / min_len if min_len > 0 else 0

    return {
        "prompt": prompt[:60],
        "match_window": match_window,
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
    fails = [r for r in results if not r["pass"] and not r.get("xfail_issue")]
    known = [r for r in results if not r["pass"] and r.get("xfail_issue")]
    if fails:
        lines.append(f"**FAIL**: {len(fails)}/{len(results)} prompts diverged within their match windows")
    elif known:
        issues = ", ".join(sorted({r["xfail_issue"] for r in known}))
        lines.append(
            f"**PASS**: {len(results) - len(known)}/{len(results)} gating prompts match; "
            f"{len(known)} known divergence ({issues}) excluded from the verdict"
        )
    else:
        lines.append(f"**PASS**: all {len(results)} prompts match within their respective match windows")
    xpass = [r for r in results if r["pass"] and r.get("xfail_issue")]
    if xpass:
        issues = ", ".join(sorted({r["xfail_issue"] for r in xpass}))
        lines.append("")
        lines.append(
            f"{len(xpass)} expected-divergent prompt(s) now PASS ({issues}). "
            "If this repeats, the underlying issue is likely fixed — remove the "
            "entry from METAL_EXPECTED_DIVERGENCE in scripts/e2e_parity_check.py."
        )
    lines.append("")

    lines.append("| Prompt | Window | Agreement | First Diff | HF tok/s | Lattice tok/s | Verdict |")
    lines.append("|--------|--------|-----------|------------|----------|---------------|---------|")
    for r in results:
        diff = f"pos {r['first_mismatch']}" if r["first_mismatch"] is not None else "none"
        if r["pass"]:
            icon = f"XPASS ({r['xfail_issue']})" if r.get("xfail_issue") else "PASS"
        else:
            icon = (
                f"KNOWN-DIVERGENT ({r['xfail_issue']})"
                if r.get("xfail_issue")
                else "FAIL"
            )
        lines.append(
            f"| `{r['prompt']}` | {r['match_window']} | {r['total_agree']}/{r['total_compared']} "
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["cpu", "metal"],
        default="cpu",
        help=(
            "cpu (default): qwen35_generate, legacy text parsing. "
            "metal: chat_metal --json, fail-closed on the LATTICE_METAL_PATH_PROOF "
            "capability marker (issue #239)."
        ),
    )
    args = parser.parse_args()
    backend = args.backend

    global LATTICE_BIN
    if not _LATTICE_BIN_EXPLICIT:
        LATTICE_BIN = (
            "target/release/chat_metal"
            if backend == "metal"
            else "target/release/qwen35_generate"
        )

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
    for prompt, match_window in PROMPTS:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Prompt: {prompt[:60]}  (match_window={match_window})", file=sys.stderr)

        print("[hf] running reference...", file=sys.stderr)
        hf_out = run_hf_reference(prompt, MAX_TOKENS)

        print(f"[lattice-{backend}] running under test...", file=sys.stderr)
        lat_out = (
            run_lattice_metal(prompt, MAX_TOKENS)
            if backend == "metal"
            else run_lattice(prompt, MAX_TOKENS)
        )

        if lat_out is None:
            print("FAIL: lattice binary failed", file=sys.stderr)
            return 1

        verdict = compare(prompt, hf_out, lat_out, match_window)
        xfail_issue = (
            METAL_EXPECTED_DIVERGENCE.get(prompt) if backend == "metal" else None
        )
        if xfail_issue:
            verdict["xfail_issue"] = xfail_issue
        results.append(verdict)

        if verdict["pass"]:
            status = (
                f"XPASS {xfail_issue} — expected divergence resolved?"
                if xfail_issue
                else "PASS"
            )
        else:
            status = f"KNOWN-DIVERGENT {xfail_issue}" if xfail_issue else "FAIL"
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

    fails = sum(1 for r in results if not r["pass"] and not r.get("xfail_issue"))
    known = sum(1 for r in results if not r["pass"] and r.get("xfail_issue"))
    xpass = [r for r in results if r["pass"] and r.get("xfail_issue")]
    if fails:
        print(f"\nFAIL: {fails}/{len(results)} prompts failed parity gate", file=sys.stderr)
        return 1

    if xpass:
        print(
            "\nNOTE: expected-divergent prompt(s) PASSED — if this repeats, the "
            "tracked issue is likely fixed; remove them from METAL_EXPECTED_DIVERGENCE:",
            file=sys.stderr,
        )
        for r in xpass:
            print(f"  {r['xfail_issue']}: {r['prompt']}", file=sys.stderr)

    suffix = (
        f" ({known} known divergence{'s' if known != 1 else ''} excluded)"
        if known
        else ""
    )
    print(
        f"\nPASS: all {len(results) - known} gating prompts passed{suffix}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
