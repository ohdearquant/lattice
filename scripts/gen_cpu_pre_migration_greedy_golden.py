#!/usr/bin/env python3
"""Regenerate the pre-migration CPU greedy golden (ADR-090 rollout row R03).

The golden freezes what `Qwen35Model::generate` produced on the CPU path BEFORE
the shared decode driver existed, so the migration has something to be compared
against that is not itself. Regenerating it therefore destroys the only reason
it exists, and this script is deliberate and manual: CI never runs it, and it
refuses to overwrite without `--update-golden`.

Regenerate only when the inputs legitimately change (a different checkpoint, a
different prompt, a different pinned sampling config) -- never to make a failing
gate pass. A golden is self-ratifying by construction: whatever it captures
becomes the definition of correct, so a divergence is a question about the code,
not about the fixture.

    python3 scripts/gen_cpu_pre_migration_greedy_golden.py \
        --model-dir /abs/path/to/qwen3.5-0.8b --update-golden

Build the producer first (the `f16` feature is required -- the checkpoint is
bf16, and CI builds this binary the same way):

    cargo build --release -p lattice-inference --bin qwen35_generate --features f16
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN = (
    REPO_ROOT
    / "crates/inference/tests/fixtures/cpu_pre_migration_greedy_v1"
    / "qwen35_0_8b_cpu_greedy_tokens.json"
)
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "qwen35_generate"

# Pinned generation config, registered before the original capture. Temperature
# alone does not pin greedy: `apply_repetition_penalty` runs BEFORE the
# degenerate-temperature check routes to argmax, and GenerateConfig::default()
# carries a serving repetition_penalty of 1.1. See
# crates/inference/src/model/qwen35/sampling.rs, and scripts/e2e_parity_check.py,
# which pins both flags for the same reason.
PINNED = {"temperature": "0.0", "repetition_penalty": "1.0", "max_new_tokens": 16}

TOKEN_IDS_RE = re.compile(r"^Token IDs: \[(.*)\]$", re.MULTILINE)
PROMPT_TOKENS_RE = re.compile(r"^Prompt tokens:\s+(\d+)$", re.MULTILINE)


def run_case(
    binary: Path, model_dir: Path, prompt: str, reasoning_budget: int | None = None
) -> tuple[list[int], int]:
    # A budgeted case is a different code path, not a different prompt: the
    # override that replaces the sampled id with `</think>` is unreachable while
    # the budget is unset, so a golden captured without this flag freezes the
    # decode loop with that branch permanently dark.
    budget_args = [] if reasoning_budget is None else ["--reasoning-budget", str(reasoning_budget)]
    proc = subprocess.run(
        [
            str(binary),
            "--model-dir", str(model_dir),
            "--prompt", prompt,
            "--max-tokens", str(PINNED["max_new_tokens"]),
            "--temperature", PINNED["temperature"],
            "--repetition-penalty", PINNED["repetition_penalty"],
            *budget_args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.exit(f"FAILED: {binary.name} exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}")

    ids_match = TOKEN_IDS_RE.search(proc.stdout)
    prompt_match = PROMPT_TOKENS_RE.search(proc.stdout)
    if not ids_match or not prompt_match:
        # A parse miss is a read failure, not an empty result: the producer's
        # output format changed, and silently writing a golden without the ids
        # would be worse than stopping.
        sys.exit(f"FAILED: could not parse producer output:\n{proc.stdout}")
    ids = [int(tok) for tok in ids_match.group(1).split(",") if tok.strip()]
    return ids, int(prompt_match.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True, help="absolute path to the Qwen3.5-0.8B checkpoint")
    ap.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    ap.add_argument("--update-golden", action="store_true", help="required to write; without it this is a dry run")
    args = ap.parse_args()

    if not args.model_dir.is_absolute():
        return sys.exit(f"FAILED: --model-dir must be absolute, got {args.model_dir}")
    if not args.model_dir.exists():
        return sys.exit(f"FAILED: checkpoint {args.model_dir} does not exist")
    if not args.binary.exists():
        return sys.exit(
            f"FAILED: producer not found at {args.binary}; build it with\n"
            "  cargo build --release -p lattice-inference --bin qwen35_generate --features f16"
        )
    if not GOLDEN.exists():
        return sys.exit(f"FAILED: existing golden {GOLDEN} not found; this script updates, it does not bootstrap")

    doc = json.loads(GOLDEN.read_text())
    if doc["max_new_tokens"] != PINNED["max_new_tokens"]:
        return sys.exit(
            f"FAILED: golden pins max_new_tokens={doc['max_new_tokens']} but this script pins "
            f"{PINNED['max_new_tokens']}. Reconcile deliberately; do not let the script win by default."
        )

    changed = []
    for case in doc["cases"]:
        ids, prompt_tokens = run_case(
            args.binary, args.model_dir, case["prompt"], case.get("reasoning_budget")
        )
        if ids != case["expected_generated_ids"] or prompt_tokens != case["prompt_tokens"]:
            changed.append(case["name"])
            print(f"{case['name']}: CHANGED")
            print(f"  committed: {case['expected_generated_ids']} (prompt_tokens={case['prompt_tokens']})")
            print(f"  measured:  {ids} (prompt_tokens={prompt_tokens})")
        else:
            print(f"{case['name']}: unchanged")
        case["expected_generated_ids"] = ids
        case["prompt_tokens"] = prompt_tokens

    if not changed:
        print("\nNo case changed. The committed golden already matches this checkpoint and build.")
        return 0

    if not args.update_golden:
        print(f"\nDRY RUN: {len(changed)} case(s) would change ({', '.join(changed)}). Pass --update-golden to write.")
        return 1

    GOLDEN.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"\nWROTE {GOLDEN} ({len(changed)} case(s) changed: {', '.join(changed)})")
    print("Review the diff before committing: a changed golden is a behaviour change until proven otherwise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
