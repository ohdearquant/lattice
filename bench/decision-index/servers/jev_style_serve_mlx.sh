#!/bin/bash
# jev_style_serve_mlx.sh <model_dir> <python> <port>
# Jev-Style 2B v3 on the author's MLX bf16 build: the author's own server (jev_style/cli.py `serve`), 127.0.0.1:<port>,
# local model directory only. The torch path at the author's default float32 does not fit a 16 GiB host (measured:
# memory pressure critical at load, 8.75 GB RSS); bfloat16 on torch is CUDA-only in the author's runtime. The MLX bf16
# build is the format the card recommends for Apple silicon and gates against FP32 (99.7% same top-1, 43/43 long).
# HF_HUB_OFFLINE=1: with --model-dir the runtime never calls the Hub; this turns an accidental fetch into a failure.
set -euo pipefail
MD=${1:?usage: jev_style_serve_mlx.sh <model_dir> <python> <port>}
PY=${2:?usage: jev_style_serve_mlx.sh <model_dir> <python> <port>}
PORT=${3:?usage: jev_style_serve_mlx.sh <model_dir> <python> <port>}
export HF_HUB_OFFLINE=1
exec "$PY" -I -m jev_style serve \
  --release "${JEV_RELEASE:-2b-v3}" \
  --model-dir "$MD" \
  --backend mlx \
  --precision bf16 \
  --host 127.0.0.1 \
  --port "$PORT"
