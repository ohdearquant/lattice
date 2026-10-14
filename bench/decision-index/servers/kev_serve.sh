#!/bin/bash
# kev_serve.sh <model_dir> <python> <port>
# Kev 0.8B r15: the author's own server (python -m kev.serve -> kev.serve:app + uvicorn), 127.0.0.1:<port>,
# local checkpoint directory only.
#
# --run <model_dir> points at the local checkpoint (kev/serve.py:331-337); is_hub_id() is False for a path and
# head.pt exists there, so no Hub lookup happens. HF_HUB_OFFLINE=1 makes any accidental call fail. The LoRA base
# (Qwen/Qwen3.5-0.8B-Base) is loaded from the base model id in head.pt's meta, however (kev/checkpoint.py:238 ->
# load_tokenizer(meta.base) and DecisionModel(meta.base)), so the base must already be in the HF cache; offline
# then resolves it to the cached snapshot. There is no flag that redirects the base to a local path.
#
# --device is left at the author's default "auto" on purpose: on Apple silicon with mlx-lm installed, a hybrid
# Qwen3.5 checkpoint runs on MLX and the server does NOT probe torch MPS (serve.py:315-326, _serving_device);
# that is the author's measured Apple-silicon path (kev/mlx_model.py, tests/test_mlx.py). Passing --device mps
# still allows MLX (requested in ("auto","mps")), but --device auto keeps the exact published default.
#
# No numerics/calibration flag is added: the checkpoint's own temperature (head.pt, kev/checkpoint.py:241) and
# the server's serving defaults (backend auto, attn=sdpa on mps, dtype bf16 off-CPU, kev/serve.py:340-347) are the
# author's defaults. KEV_* env vars (KEV_DTYPE, KEV_TEMPERATURE, KEV_BACKEND, ...) stay unset for the same reason.
set -euo pipefail
MD=${1:?usage: kev_serve.sh <model_dir> <python> <port>}
PY=${2:?usage: kev_serve.sh <model_dir> <python> <port>}
PORT=${3:?usage: kev_serve.sh <model_dir> <python> <port>}
export HF_HUB_OFFLINE=1
exec "$PY" -I -m kev.serve \
  --run "$MD" \
  --host 127.0.0.1 \
  --port "$PORT"
