#!/bin/bash
# decider2b_serve.sh <model_dir> <python> <port>: the decider-2b repo's own server (decider/serve.py) on MPS, bound to
# 127.0.0.1. The author's device auto-selection picks mps here anyway; pinned so a missing MPS fails instead of falling to
# cpu. fp16 on MPS is the author's choice (serve.py:392); compile and fp8 are CUDA-only and stay off (serve.py:391).
MD=$1; PY=$2; PORT=$3
export DECIDER_MODEL="$MD" DECIDER_DEVICE=mps DECIDER_COMPILE=0 DECIDER_FP8=0
exec "$PY" -I -c 'import sys; sys.path.insert(0, sys.argv[1]); import uvicorn; uvicorn.run("decider.serve:app", host="127.0.0.1", port=int(sys.argv[2]), log_level="warning")' "$MD" "$PORT"
