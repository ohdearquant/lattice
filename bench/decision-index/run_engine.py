"""Run the kit CLI with this workspace's in-process engine wrappers importable under `python -I`.

-I drops PYTHONPATH and the script directory from sys.path, so the engines directory is added explicitly.
Usage: python -I run_engine.py run --engine intern_decision_engine:InternDecisionEngine --option model_dir=... ...

The MPS allocator is capped at the device's recommended working set (high watermark 1.0, low 0.8) unless
the caller sets these variables, so a row that needs more memory than the host has fails as an ordinary error record
instead of driving the host to critical memory pressure. Allocation policy only; no numerics change.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")
print("run_engine: PYTORCH_MPS_HIGH_WATERMARK_RATIO=%s PYTORCH_MPS_LOW_WATERMARK_RATIO=%s" % (
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"], os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"]), file=sys.stderr, flush=True)
sys.path.insert(0, str(Path(__file__).resolve().parent / "engines"))
from decision_index.cli import main  # noqa: E402

main(sys.argv[1:])
