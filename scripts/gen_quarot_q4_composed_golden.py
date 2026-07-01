#!/usr/bin/env python3
"""Regenerate the QuaRot+Q4 composed-path golden fixture (issue #320).

Orchestrates the full composed pipeline against a real Qwen3.5-0.8B model
and writes a frozen lattice-self token golden consumed by
`crates/inference/tests/quarot_q4_composed_golden.rs`:

    real safetensors
      -> quantize_quarot (rotation + fusion + Q4_0 quantization)
      -> dump_quarot_q4_golden (from_q4_dir + greedy generate)
      -> committed JSON golden

This script only BUILDS the Rust binaries and pipes their output; it does
not reimplement any numerics. The golden is a deliberately frozen artifact:
this script refuses to overwrite an existing golden unless --update-golden
is passed explicitly.

Usage:
    python3 scripts/gen_quarot_q4_composed_golden.py \\
        --model-dir ~/.lattice/models/qwen3.5-0.8b \\
        --q4-dir target/quarot-q4-golden/qwen3.5-0.8b-quarot-q4 \\
        --seed 0xCAFE_BABE_DEAD_BEEF \\
        --output crates/inference/tests/fixtures/quarot_q4_composed_v1/\
qwen35_0_8b_quarot_q4_greedy_tokens.json \\
        --update-golden

Requirements:
    - macOS with Metal (the composed path is Metal-only).
    - Real Qwen3.5-0.8B safetensors at --model-dir.
    - cargo toolchain available on PATH.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_MODEL_DIR = "~/.lattice/models/qwen3.5-0.8b"
DEFAULT_SEED = "0xCAFE_BABE_DEAD_BEEF"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "crates"
    / "inference"
    / "tests"
    / "fixtures"
    / "quarot_q4_composed_v1"
    / "qwen35_0_8b_quarot_q4_greedy_tokens.json"
)


def run(cmd: list[str], **kwargs) -> None:
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR, help="Real Qwen3.5-0.8B safetensors dir"
    )
    p.add_argument(
        "--model-id", default=DEFAULT_MODEL_ID, help="HF model ID recorded for provenance"
    )
    p.add_argument(
        "--q4-dir",
        default="target/quarot-q4-golden/qwen3.5-0.8b-quarot-q4",
        help="Ephemeral output dir for the QuaRot Q4 artifact",
    )
    p.add_argument(
        "--seed", default=DEFAULT_SEED, help="QuaRot rotation seed (decimal or 0x... hex)"
    )
    p.add_argument(
        "--num-probe-tokens",
        type=int,
        default=4,
        help="quantize_quarot forward-equivalence probe count",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=8, help="Greedy tokens generated per prompt"
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Committed golden JSON path")
    p.add_argument(
        "--update-golden",
        action="store_true",
        help="Required to overwrite an existing golden. Without this flag the "
        "script refuses to run if --output already exists, so the frozen "
        "golden can never be clobbered by an accidental re-run.",
    )
    p.add_argument(
        "--skip-convert",
        action="store_true",
        help="Reuse an existing --q4-dir instead of rerunning quantize_quarot "
        "(useful when iterating on the dumper only).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_path = args.output.resolve()
    if output_path.exists() and not args.update_golden:
        print(
            f"ERROR: {output_path} already exists. The QuaRot+Q4 composed golden "
            "is a deliberately frozen artifact — pass --update-golden to "
            "intentionally regenerate and overwrite it (this is a reviewable "
            "source change, not routine CI behavior).",
            file=sys.stderr,
        )
        sys.exit(1)

    model_dir = Path(args.model_dir).expanduser()
    q4_dir = Path(args.q4_dir)
    if not q4_dir.is_absolute():
        q4_dir = REPO_ROOT / q4_dir

    if not args.skip_convert:
        run(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "lattice-inference",
                "--bin",
                "quantize_quarot",
                "--features",
                "f16",
            ]
        )
        if q4_dir.exists():
            import shutil

            shutil.rmtree(q4_dir)
        run(
            [
                "target/release/quantize_quarot",
                "--model-dir",
                str(model_dir),
                "--output-dir",
                str(q4_dir),
                "--seed",
                args.seed,
                "--num-probe-tokens",
                str(args.num_probe_tokens),
            ]
        )
    else:
        print(
            f"[gen_quarot_q4_composed_golden] --skip-convert set; reusing {q4_dir}", file=sys.stderr
        )

    run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "lattice-inference",
            "--bin",
            "dump_quarot_q4_golden",
            "--features",
            "f16,metal-gpu",
        ]
    )

    dump_result = subprocess.run(
        [
            "target/release/dump_quarot_q4_golden",
            "--q4-dir",
            str(q4_dir),
            "--tokenizer-dir",
            str(model_dir),
            "--model-id",
            args.model_id,
            "--model-dir-default",
            args.model_dir,
            "--quarot-seed",
            args.seed,
            "--converter",
            "target/release/quantize_quarot",
            "--max-new-tokens",
            str(args.max_new_tokens),
        ],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print(dump_result.stderr, file=sys.stderr)

    golden = json.loads(dump_result.stdout)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"[gen_quarot_q4_composed_golden] wrote {output_path}", file=sys.stderr)

    print(
        "\nNext: verify the regenerated golden with the enforced gate:\n\n"
        "  LATTICE_Q4_COMPOSED_GATE_ENFORCE=1 \\\n"
        f"  LATTICE_MODEL_DIR={args.model_dir} \\\n"
        f"  LATTICE_QUAROT_Q4_DIR={q4_dir} \\\n"
        "  cargo test --release -p lattice-inference \\\n"
        "    --test quarot_q4_composed_golden \\\n"
        '    --features "f16,metal-gpu" \\\n'
        "    -- --nocapture\n",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
