#!/bin/bash
# onejev_serve.sh <model_dir> <python> <port>
# OneJev 0.8B: the author's own server (qev.cli `serve` -> qev.server.create_app + uvicorn),
# 127.0.0.1:<port>, local model directory only.
#
# --device mps is required, not a numerics choice: the author's default is "cuda:0" (qev/cli.py:164), which cannot
# run on the run host. The author documents CUDA/ROCm/CPU targets only (README "Option A: PyTorch"; docker/start.sh
# profiles torch|rocm|vulkan|gguf|cpu) and has no Apple-silicon path at all, so this is the smallest possible
# device plumbing; whether MPS actually runs this hybrid Qwen3.5 runtime is unverified.
# No --dtype is passed, so the author's default bfloat16 (qev/cli.py:165) and head_dtype float32 (cli.py:166) stand.
# If MPS rejects bf16 kernels, --dtype float32 is the documented fallback (not applied here).
#
# --model <local dir>: qev.cli._resolve_model returns a name that exists as a path unchanged (cli.py:19-20) and
# local_model_dir returns it (mm_engine.py:36-40), so nothing is fetched. HF_HUB_OFFLINE=1 makes any accidental
# call to the Hub fail instead of downloading.
#
# The 0.8B release is the multimodal checkpoint (config.json has vision_config, qev/cli.py:118), so the server
# builds MMDecisionEngine; requests without `media` are rendered exactly like text requests (qev/media.py:13-18).
set -euo pipefail
MD=${1:?usage: onejev_serve.sh <model_dir> <python> <port>}
PY=${2:?usage: onejev_serve.sh <model_dir> <python> <port>}
PORT=${3:?usage: onejev_serve.sh <model_dir> <python> <port>}
export HF_HUB_OFFLINE=1
exec "$PY" -I -m qev.cli serve \
  --model "$MD" \
  --device mps \
  --host 127.0.0.1 \
  --port "$PORT"
