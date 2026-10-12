#!/bin/bash
# decider08_serve.sh <model_dir> <python> <port>: decider-0.8b's own server through the MPS device shim
# (serve_mps.py beside this script; device plumbing only).
set -euo pipefail
MD=${1:?usage: decider08_serve.sh <model_dir> <python> <port>}
PY=${2:?usage: decider08_serve.sh <model_dir> <python> <port>}
PORT=${3:?usage: decider08_serve.sh <model_dir> <python> <port>}
export DECIDER_COMPILE=0 DECIDER_FP8=0
exec "$PY" -I "$(cd "$(dirname "$0")" && pwd)/serve_mps.py" "$MD" "$PORT"
