#!/usr/bin/env python3
"""E2E parity gate: frozen/live HF reference vs lattice (under test).

PR and push runs load a frozen HF reference. Scheduled and manually dispatched
runs execute HF transformers live. Compares greedy generation output (token IDs)
and reports speed.

Exit codes: 0 = pass, 1 = parity failure, 2 = setup error.

A lattice binary that exits non-zero, emits unparseable output, or never
finishes is a lattice failure and exits 1 on both backends. It is not a setup
error: reporting it as one tells whoever reads the run that their environment
is broken when what actually broke is the thing under test. 2 is reserved for
everything that stops this script from producing a comparison at all, which
includes anything wrong with the reference side, since a bad reference is
never evidence about lattice.

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
  --regenerate            run HF live and rewrite the frozen reference fixture.

Env vars:
  LATTICE_BIN             path to the lattice binary under test (default:
                           target/release/qwen35_generate for --backend cpu,
                           target/release/chat_metal for --backend metal)
  LATTICE_MODEL_DIR       path to model weights (default: ~/.lattice/models/qwen3.5-0.8b);
                          also the source of the HF reference load (local files only,
                          no Hub fetch)
  E2E_MAX_TOKENS          tokens to generate per prompt (default: 15)
  E2E_REPORT_PATH         write markdown report here (optional)
  LATTICE_METAL_PATH_PROOF  set to "1" so chat_metal emits the
                           `[METAL_PATH_PROOF]` stderr marker this script
                           requires in --backend metal mode
"""

import argparse
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path


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
# The metal parity job is a required RATCHET gate (promoted per ADR-066 F2,
# issue #239) precisely BECAUSE this waiver is here: what it certifies is
# Metal path-proof, binary health, and token parity on every prompt except
# this #535-waived one — not full Metal parity. Its CI display name says so.
# Marking the prompt expected-divergent keeps the job green while #535 is
# open WITHOUT weakening anything else:
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

# compare() reads this at call time, so it stays module scope. The parse must not
# raise here: an exception at import ends the process on a traceback, and a
# traceback exits 1, which this script reserves for lattice disagreeing with the
# reference. A typo in the workflow is a setup error. Hold the raw value and let
# main()'s preflight reject it as one.
_MAX_TOKENS_RAW = os.environ.get("E2E_MAX_TOKENS", "15")
try:
    MAX_TOKENS: int | None = int(_MAX_TOKENS_RAW)
except ValueError:
    MAX_TOKENS = None

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
REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "crates/inference/tests/fixtures/e2e_parity_reference_v1/reference.json"
)
REFERENCE_PACKAGES = ("torch", "transformers", "tokenizers", "huggingface_hub")
REFERENCE_TOKEN_COUNTS = (4, 15)
REFERENCE_THREAD_COUNTS = (1, 4)
# The 2026-07-28 pinned 1-thread/4-thread refresh observed fragile minima of
# 0.0150260925 and 0.0319595337; the remaining per-prompt minima start at
# 0.110658646. Refuse margins below 0.1 so the fragile positions cannot be
# admitted into a refreshed reference.
REFERENCE_LOGIT_MARGIN_FLOOR = 0.1
MODEL_REPO = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def installed_reference_versions() -> dict[str, str]:
    """Read the installed reference package versions."""
    versions = {}
    for package in REFERENCE_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as e:
            raise RuntimeError(
                f"reference package {package!r} is not installed"
            ) from e
    return versions


def validate_regeneration_outputs(
    runs: dict[int, dict[str, dict]],
    prompts: list[str],
    margin_floor: float,
) -> dict:
    """Validate token stability and logit margins across reference workers."""
    first_thread, second_thread = REFERENCE_THREAD_COUNTS
    summary = {
        "global_minimum_margin": math.inf,
        "global_minimum_prompt": None,
        "global_minimum_position": None,
        "global_minimum_thread_count": None,
    }
    for prompt in prompts:
        first = runs[first_thread][prompt]
        second = runs[second_thread][prompt]
        first_ids = first["generated_ids"]
        second_ids = second["generated_ids"]
        if first_ids != second_ids:
            common_length = min(len(first_ids), len(second_ids))
            position = next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(first_ids, second_ids)
                    )
                    if left != right
                ),
                common_length,
            )
            first_token = (
                first_ids[position] if position < len(first_ids) else "<end>"
            )
            second_token = (
                second_ids[position] if position < len(second_ids) else "<end>"
            )
            raise RuntimeError(
                f"reference disagreement for prompt {prompt!r} at token position "
                f"{position}: {first_thread} thread(s) produced "
                f"{first_token}, {second_thread} thread(s) produced "
                f"{second_token}"
            )
        for thread_count, output in (
            (first_thread, first),
            (second_thread, second),
        ):
            margins = output.get("logit_margins")
            if not isinstance(margins, list) or len(margins) != len(first_ids):
                raise RuntimeError(
                    f"invalid logit margins for prompt {prompt!r} with "
                    f"{thread_count} thread(s)"
                )
            for position, margin in enumerate(margins):
                if not isinstance(margin, (int, float)) or not math.isfinite(margin):
                    raise RuntimeError(
                        f"non-finite logit margin for prompt {prompt!r} at token "
                        f"position {position} with {thread_count} thread(s)"
                    )
                if margin < margin_floor:
                    raise RuntimeError(
                        f"logit margin {margin:.9g} below refusal floor "
                        f"{margin_floor:.9g} for prompt {prompt!r} at token "
                        f"position {position} with {thread_count} thread(s)"
                    )
                if margin < summary["global_minimum_margin"]:
                    summary.update(
                        {
                            "global_minimum_margin": margin,
                            "global_minimum_prompt": prompt,
                            "global_minimum_position": position,
                            "global_minimum_thread_count": thread_count,
                        }
                    )
    return summary


def load_frozen_reference(max_tokens: int) -> dict[str, dict]:
    """Load and validate the frozen HF reference, keyed by prompt content."""
    try:
        with REFERENCE_PATH.open() as f:
            fixture = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"frozen reference {REFERENCE_PATH} is missing, unreadable, or malformed: {e}"
        ) from e

    if not isinstance(fixture, dict):
        raise RuntimeError("frozen reference root must be a JSON object")
    versions = fixture.get("package_versions")
    model = fixture.get("model")
    generation = fixture.get("generation")
    determinism = fixture.get("determinism")
    entries = fixture.get("prompts")
    schema_version = fixture.get("schema_version")
    if (
        schema_version not in (1, 2)
        or not isinstance(versions, dict)
        or model != {"repo_id": MODEL_REPO, "revision": MODEL_REVISION}
        or not isinstance(generation, dict)
        or generation.get("max_new_tokens") != max(REFERENCE_TOKEN_COUNTS)
        or generation.get("do_sample") is not False
        or generation.get("temperature") is not None
        or generation.get("top_p") is not None
        or generation.get("top_k") is not None
        or (
            schema_version == 2
            and (
                not isinstance(determinism, dict)
                or determinism.get("thread_counts")
                != list(REFERENCE_THREAD_COUNTS)
                or determinism.get("logit_margin_floor")
                != REFERENCE_LOGIT_MARGIN_FLOOR
            )
        )
        or not isinstance(entries, list)
    ):
        raise RuntimeError(
            "frozen reference schema, model provenance, or generation settings "
            "are malformed or do not match this gate"
        )

    installed = installed_reference_versions()
    if set(versions) != set(REFERENCE_PACKAGES):
        raise RuntimeError(
            "frozen reference package_versions must name exactly "
            f"{', '.join(REFERENCE_PACKAGES)}"
        )
    for package in REFERENCE_PACKAGES:
        recorded = versions.get(package)
        if recorded != installed[package]:
            raise RuntimeError(
                f"frozen reference package version mismatch for {package}: "
                f"recorded {recorded!r}, installed {installed[package]!r}"
            )

    by_prompt = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError(f"frozen reference prompt entry {index} is not an object")
        prompt = entry.get("prompt")
        match_window = entry.get("match_window")
        references = entry.get("references")
        if (
            not isinstance(prompt, str)
            or isinstance(match_window, bool)
            or not isinstance(match_window, int)
            or not isinstance(references, list)
        ):
            raise RuntimeError(f"frozen reference prompt entry {index} is malformed")
        if prompt in by_prompt:
            raise RuntimeError(
                f"frozen reference contains duplicate entry for prompt {prompt[:40]!r}"
            )
        matching_references = [
            reference
            for reference in references
            if isinstance(reference, dict)
            and reference.get("token_count") == max_tokens
        ]
        if len(matching_references) != 1:
            raise RuntimeError(
                f"frozen reference has {len(matching_references)} entries with token "
                f"count {max_tokens} for prompt {prompt[:40]!r}; expected exactly one"
            )
        reference = matching_references[0]
        generated_ids = reference.get("generated_ids")
        logit_margins = reference.get("logit_margins")
        if (
            not isinstance(generated_ids, list)
            or any(
                isinstance(token, bool) or not isinstance(token, int)
                for token in generated_ids
            )
            or len(generated_ids) != max_tokens
            or (
                schema_version == 2
                and (
                    not isinstance(logit_margins, list)
                    or len(logit_margins) != max_tokens
                    or any(
                        isinstance(margin, bool)
                        or not isinstance(margin, (int, float))
                        or not math.isfinite(margin)
                        or margin < REFERENCE_LOGIT_MARGIN_FLOOR
                        for margin in logit_margins
                    )
                )
            )
        ):
            raise RuntimeError(
                f"frozen reference token count for prompt {prompt[:40]!r} is "
                f"{reference.get('token_count')}, but "
                f"{len(generated_ids) if isinstance(generated_ids, list) else 'an invalid number of'} "
                "token ids are recorded"
            )
        by_prompt[prompt] = {
            "generated_ids": generated_ids,
            "text": f"frozen token ids: {generated_ids}",
            "tok_per_sec": None,
        }

    for prompt, match_window in PROMPTS:
        entry = by_prompt.get(prompt)
        if entry is None:
            raise RuntimeError(
                f"frozen reference has no entry for prompt {prompt[:40]!r}"
            )
        fixture_entry = next(item for item in entries if item.get("prompt") == prompt)
        if fixture_entry["match_window"] != match_window:
            raise RuntimeError(
                f"frozen reference match window for prompt {prompt[:40]!r} is "
                f"{fixture_entry['match_window']}, expected {match_window}"
            )
    if len(by_prompt) != len(PROMPTS):
        raise RuntimeError(
            "frozen reference contains prompts that are not present in PROMPTS"
        )
    return by_prompt


def run_regeneration_workers(max_tokens: int) -> tuple[dict[str, dict], dict]:
    """Generate and compare references in isolated thread-count workers."""
    runs = {}
    for thread_count in REFERENCE_THREAD_COUNTS:
        command = [
            "nice",
            "-n",
            "10",
            sys.executable,
            str(Path(__file__).resolve()),
            "--hf-reference-worker",
            "--hf-threads",
            str(thread_count),
        ]
        environment = os.environ.copy()
        environment["E2E_MAX_TOKENS"] = str(max_tokens)
        print(
            f"[hf] starting isolated {thread_count}-thread reference worker",
            file=sys.stderr,
        )
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        if completed.returncode != 0:
            raise RuntimeError(
                f"{thread_count}-thread reference worker exited "
                f"{completed.returncode}"
            )
        try:
            runs[thread_count] = json.loads(completed.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"{thread_count}-thread reference worker returned invalid JSON: {e}"
            ) from e

    prompts = [prompt for prompt, _ in PROMPTS]
    for prompt in prompts:
        for token_count in REFERENCE_TOKEN_COUNTS:
            minima = {
                thread_count: min(
                    runs[thread_count][prompt]["logit_margins"][:token_count]
                )
                for thread_count in REFERENCE_THREAD_COUNTS
            }
            print(
                f"[hf] prompt {prompt[:40]!r}, {token_count} tokens; minimum "
                f"margins 1-thread={minima[1]:.9g}, "
                f"4-thread={minima[4]:.9g}",
                file=sys.stderr,
            )
    summary = validate_regeneration_outputs(
        runs, prompts, REFERENCE_LOGIT_MARGIN_FLOOR
    )
    print(
        "[hf] global minimum margin "
        f"{summary['global_minimum_margin']:.9g} at token position "
        f"{summary['global_minimum_position']} with "
        f"{summary['global_minimum_thread_count']} thread(s) for prompt "
        f"{summary['global_minimum_prompt'][:40]!r}",
        file=sys.stderr,
    )
    return runs[REFERENCE_THREAD_COUNTS[0]], summary


def write_frozen_reference(
    outputs: dict[str, dict], max_tokens: int, determinism: dict
) -> None:
    """Write live HF outputs as the frozen reference fixture."""
    if max_tokens < max(REFERENCE_TOKEN_COUNTS):
        raise RuntimeError(
            f"--regenerate requires E2E_MAX_TOKENS >= {max(REFERENCE_TOKEN_COUNTS)}"
        )
    fixture = {
        "schema_version": 2,
        "package_versions": installed_reference_versions(),
        "model": {"repo_id": MODEL_REPO, "revision": MODEL_REVISION},
        "generation": {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        },
        "determinism": {
            "thread_counts": list(REFERENCE_THREAD_COUNTS),
            "logit_margin_floor": REFERENCE_LOGIT_MARGIN_FLOOR,
            **determinism,
        },
        "prompts": [
            {
                "prompt": prompt,
                "match_window": match_window,
                "references": [
                    {
                        "token_count": token_count,
                        "generated_ids": outputs[prompt]["generated_ids"][:token_count],
                        "logit_margins": outputs[prompt]["logit_margins"][:token_count],
                    }
                    for token_count in REFERENCE_TOKEN_COUNTS
                ],
            }
            for prompt, match_window in PROMPTS
        ],
    }
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = REFERENCE_PATH.with_suffix(".json.tmp")
    with temporary_path.open("w") as f:
        json.dump(fixture, f, indent=2)
        f.write("\n")
    temporary_path.replace(REFERENCE_PATH)


def run_hf_reference(
    prompt: str, max_tokens: int, *, collect_scores: bool = False
) -> dict:
    """Run HF transformers greedy generation. Returns tokens + timing."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not hasattr(run_hf_reference, "_model"):
        if not os.path.isdir(MODEL_DIR):
            raise RuntimeError(
                f"reference model snapshot not found at {MODEL_DIR}; "
                "provision it before running this script (no network fallback)"
            )
        # `use_safetensors=True` below is the enforcement at the load call:
        # trust_remote_code=False blocks remote *code*, but a `.bin` weight file
        # is a pickle and stays an arbitrary-execution vector at load time
        # regardless of that flag. Refusing a snapshot that carries one at all
        # happens in the preflight, where it can exit as a setup error instead
        # of being mistaken for a parity failure.
        t0 = time.time()
        run_hf_reference._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, trust_remote_code=False, local_files_only=True
        )
        run_hf_reference._model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            dtype=torch.float32,
            trust_remote_code=False,
            local_files_only=True,
            use_safetensors=True,
        )
        run_hf_reference._model.eval()
        print(f"[hf] model loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    tokenizer = run_hf_reference._tokenizer
    model = run_hf_reference._model

    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0].tolist()

    t0 = time.time()
    with torch.no_grad():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            output_scores=collect_scores,
            return_dict_in_generate=collect_scores,
        )
    elapsed = time.time() - t0

    sequences = generation.sequences if collect_scores else generation
    all_ids = sequences[0].tolist()
    gen_ids = all_ids[len(prompt_ids):]
    logit_margins = []
    if collect_scores:
        for scores in generation.scores:
            top_two = torch.topk(scores[0], k=2).values
            logit_margins.append(float((top_two[0] - top_two[1]).item()))
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return {
        "prompt_ids": prompt_ids,
        "generated_ids": gen_ids,
        "logit_margins": logit_margins,
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
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        # Mirrors run_lattice_metal: a lattice binary that never finishes is a
        # lattice failure, in the same category as one that exits non-zero, and
        # both reach main() as a None result. Letting the exception escape here
        # would classify it as a setup error, which tells whoever reads the run
        # that their environment is broken when what actually broke is the thing
        # under test.
        print("[lattice] FAILED (timeout)", file=sys.stderr)
        return None
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

    # Parser boundary. A binary that exits 0 and then prints a field this cannot
    # read has failed to produce a usable result, which is the same category as
    # exiting non-zero, so it leaves as None and the run reports a lattice
    # failure. Narrowed to the conversions malformed output actually raises: a
    # launch-side OSError must still reach the entry point as the setup failure
    # it is. The regexes are not sufficient protection here, since `[\d.]+`
    # matches `1.2.3` and the token list is unconstrained.
    try:
        gen_ids = [int(x.strip()) for x in token_match.group(1).split(",") if x.strip()]
        gen_count = int(gen_match.group(1)) if gen_match else len(gen_ids)
        tok_per_sec = float(speed_match.group(1)) if speed_match else (gen_count / elapsed if elapsed > 0 else 0)
    except (ValueError, TypeError) as e:
        print(f"[lattice] unreadable output field: {e}", file=sys.stderr)
        print(stdout, file=sys.stderr)
        return None

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

        # Check the event's shape once, here, rather than defending each field
        # where it is read. json.loads returns whatever JSON value the binary
        # emitted, so `@@lattice []` is well-formed JSON and `ev.get` raises on
        # it; typing every field afterwards means one more escape route per
        # field added later. Everything below this point may assume a dict with
        # correctly typed values, and a binary that emits anything else has
        # failed to produce a usable result, which leaves as None like every
        # other lattice failure.
        if not isinstance(ev, dict):
            print(
                f"[lattice-metal] @@lattice event is not a JSON object: {ev!r}",
                file=sys.stderr,
            )
            return None
        if ev.get("ev") != "gen_token":
            continue

        # bool is a subclass of int in Python, so `isinstance(True, int)` is
        # true; the token_id check below rejects a boolean explicitly rather
        # than silently accepting `true` as token 1.
        bad_field = None
        if "token_id" in ev and (
            isinstance(ev["token_id"], bool) or not isinstance(ev["token_id"], int)
        ):
            bad_field = "token_id must be an integer"
        elif "token" in ev and not isinstance(ev["token"], str):
            bad_field = "token must be a string"
        elif "tok_s" in ev and (
            isinstance(ev["tok_s"], bool)
            or not isinstance(ev["tok_s"], (int, float))
        ):
            bad_field = "tok_s must be a number"
        if bad_field is not None:
            print(
                f"[lattice-metal] gen_token event is ill-typed ({bad_field}): {ev}",
                file=sys.stderr,
            )
            print(line, file=sys.stderr)
            return None

        if ev.get("done"):
            saw_done = True
            tok_per_sec = float(ev.get("tok_s", 0.0))
            continue
        if "token_id" not in ev:
            print(
                f"[lattice-metal] gen_token event missing token_id: {ev}",
                file=sys.stderr,
            )
            return None
        gen_ids.append(ev["token_id"])
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
        hf_speed = (
            f"{r['hf_tok_s']:.1f}" if r["hf_tok_s"] is not None else "frozen"
        )
        lines.append(
            f"| `{r['prompt']}` | {r['match_window']} | {r['total_agree']}/{r['total_compared']} "
            f"| {diff} | {hf_speed} | {r['lat_tok_s']:.1f} | {icon} |"
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
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help=(
            "run HF live and rewrite the frozen reference fixture; follow "
            "docs/e2e-parity-frozen-reference.md"
        ),
    )
    parser.add_argument(
        "--hf-reference-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--hf-threads",
        type=int,
        choices=REFERENCE_THREAD_COUNTS,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    backend = args.backend
    live_reference = args.regenerate or os.environ.get("GITHUB_EVENT_NAME") in {
        "schedule",
        "workflow_dispatch",
    }

    global LATTICE_BIN
    if not _LATTICE_BIN_EXPLICIT:
        LATTICE_BIN = (
            "target/release/chat_metal"
            if backend == "metal"
            else "target/release/qwen35_generate"
        )

    # Config first, since it needs no filesystem and its diagnosis is exact. A
    # non-positive budget is rejected rather than run: zero tokens compares two
    # empty sequences, which reports as a parity failure and sends a workflow
    # typo to whoever is on the hook for a lattice regression.
    if MAX_TOKENS is None or MAX_TOKENS < 1:
        print(
            f"error: E2E_MAX_TOKENS must be a positive integer, got "
            f"{_MAX_TOKENS_RAW!r}",
            file=sys.stderr,
        )
        return 2

    if not os.path.isdir(MODEL_DIR):
        print(f"error: model dir not found at {MODEL_DIR}", file=sys.stderr)
        return 2

    # A reference snapshot carrying pickle weights is a provisioning problem, so
    # it belongs here with the other setup validation and exits 2. Raising it
    # from the loader instead would surface as exit 1, which this script defines
    # as a parity failure, and would file a poisoned reference against lattice.
    # rglob, not glob: transformers resolves sharded and subfoldered weights, so
    # a nested `.bin` is loadable and a non-recursive check would miss it. CI's
    # provisioned snapshot is flat and the exact-set verifier would reject a
    # subdirectory before this runs, but this script also runs by hand against
    # directories nothing verified, and that is where the nested case lives.
    pickle_weights = sorted(Path(MODEL_DIR).rglob("*.bin"))
    if pickle_weights:
        print(
            f"error: model dir {MODEL_DIR} contains pickle weight file(s) "
            f"{[str(p.relative_to(MODEL_DIR)) for p in pickle_weights]}; "
            "only safetensors weights are "
            "trusted here (a .bin is a pickle and an arbitrary-execution vector "
            "at load time even with trust_remote_code=False)",
            file=sys.stderr,
        )
        return 2

    if args.hf_reference_worker:
        if args.hf_threads is None:
            print("error: --hf-reference-worker requires --hf-threads", file=sys.stderr)
            return 2
        try:
            import torch

            torch.set_num_threads(args.hf_threads)
            outputs = {
                prompt: run_hf_reference(
                    prompt, MAX_TOKENS, collect_scores=True
                )
                for prompt, _ in PROMPTS
            }
        except Exception as e:  # noqa: BLE001 - worker reports setup uniformly
            print(
                f"error: reference worker failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            return 2
        json.dump(outputs, sys.stdout)
        return 0

    if not os.path.isfile(LATTICE_BIN):
        print(f"error: lattice binary not found at {LATTICE_BIN}", file=sys.stderr)
        return 2

    try:
        regeneration_summary = None
        if args.regenerate:
            regenerated_outputs, regeneration_summary = run_regeneration_workers(
                MAX_TOKENS
            )
        elif live_reference:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM  # noqa: F401
        else:
            frozen_outputs = load_frozen_reference(MAX_TOKENS)
    except (ImportError, RuntimeError) as e:
        print(f"error: reference setup failed: {e}", file=sys.stderr)
        return 2

    results = []
    live_outputs = {}
    for prompt, match_window in PROMPTS:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Prompt: {prompt[:60]}  (match_window={match_window})", file=sys.stderr)

        print(
            "[hf] running live reference..."
            if live_reference
            else "[hf] loading frozen reference...",
            file=sys.stderr,
        )
        # Anything that goes wrong producing the reference is a problem with the
        # reference, not evidence about lattice. Letting it propagate would end
        # the process on a traceback, and a traceback exits 1, which this script
        # defines as a parity failure. That mislabels every reference-side fault
        # (a snapshot that will not load, a transformers version that refuses
        # it, a directory that disappeared after the preflight) as a lattice
        # regression, which is the one conclusion this gate must never reach by
        # accident.
        try:
            hf_out = (
                regenerated_outputs[prompt]
                if args.regenerate
                else (
                    run_hf_reference(prompt, MAX_TOKENS)
                    if live_reference
                    else frozen_outputs[prompt]
                )
            )
        except Exception as e:  # noqa: BLE001 - deliberately broad, see above
            print(
                f"error: reference generation failed for prompt {prompt[:40]!r}: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            return 2
        if live_reference:
            live_outputs[prompt] = hf_out

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
        hf_speed = (
            f"{hf_out['tok_per_sec']:.1f}"
            if hf_out["tok_per_sec"] is not None
            else "frozen"
        )
        print(
            f"[{status}] agree={verdict['total_agree']}/{verdict['total_compared']} "
            f"hf={hf_speed} lat={lat_out['tok_per_sec']:.1f} tok/s",
            file=sys.stderr,
        )

    if args.regenerate:
        try:
            write_frozen_reference(
                live_outputs, MAX_TOKENS, regeneration_summary
            )
        except (OSError, RuntimeError) as e:
            print(f"error: could not write frozen reference: {e}", file=sys.stderr)
            return 2
        print(f"[hf] wrote frozen reference to {REFERENCE_PATH}", file=sys.stderr)

    report = render_report(results)
    print(report)

    if REPORT_PATH:
        # A run whose report cannot be written is a broken run, not a verdict
        # about lattice. Letting the OSError propagate would exit 1 and label an
        # unwritable output path a parity failure, including on a run where every
        # prompt matched.
        try:
            with open(REPORT_PATH, "w") as f:
                f.write(report)
        except OSError as e:
            print(
                f"error: could not write report to {REPORT_PATH}: {e}",
                file=sys.stderr,
            )
            return 2

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
    # Exit 1 means one thing here: lattice and the reference were both produced
    # and disagreed. Anything else that stops this script is a problem with the
    # harness or its inputs, and an uncaught exception would otherwise inherit
    # the interpreter's exit code of 1 and be read as a lattice regression. The
    # traceback still prints, so nothing is hidden by classifying it as setup.
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        print(
            "error: parity harness aborted; this is a harness or setup failure, "
            "not a lattice parity result",
            file=sys.stderr,
        )
        sys.exit(2)
