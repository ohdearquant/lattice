#!/bin/bash
# jev_style_serve.sh <model_dir> <python> <port>
# Jev-Style 0.8B Decision v3: the author's own server (jev_style/cli.py `serve` -> jev_style/server.py
# `create_app` + uvicorn), 127.0.0.1:<port>, local model directory only.
#
# --backend torch is required on the run host, not a numerics choice: the release snapshot at <model_dir> is the
# torch repo (jev_style_decision.py, pinned revisions["torch"] in jev_style/models.py:97). The author's `auto`
# default resolves to MLX on Apple silicon whenever mlx+mlx-lm are installed (models.py:143-153) and would then
# import jev_style_decision_mlx, which only the separate -MLX repo ships (models.py:79) -> FileNotFoundError.
# torch is also the path the release's own parity numbers were measured against (release_config.json g5_parity).
#
# --device mps pins what the runtime would auto-select on this host (jev_style_decision.py:620-622: cuda > mps >
# cpu) so a missing MPS fails loudly instead of silently serving on CPU. --dtype float32 is the author's default
# and the reference path (jev_style_decision.py:609).
#
# HF_HUB_OFFLINE=1 enforces "local files only": with --model-dir the runtime never calls the Hub anyway
# (models.py:273-276 skips download), so this only turns a future accidental fetch into a startup failure.
set -euo pipefail
MD=${1:?usage: jev_style_serve.sh <model_dir> <python> <port>}
PY=${2:?usage: jev_style_serve.sh <model_dir> <python> <port>}
PORT=${3:?usage: jev_style_serve.sh <model_dir> <python> <port>}
export HF_HUB_OFFLINE=1
exec "$PY" -I -m jev_style serve \
  --release "${JEV_RELEASE:-0.8b-v3}" \
  --model-dir "$MD" \
  --backend torch \
  --device mps \
  --dtype float32 \
  --host 127.0.0.1 \
  --port "$PORT"
