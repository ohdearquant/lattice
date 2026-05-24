#!/usr/bin/env bash
# Precise apples-to-apples decode benchmark — reduced noise edition.
#
# Differences from bench_apples_to_apples.sh:
#   - 15 runs instead of 5, with 2 warmup runs excluded
#   - N1=64, N2=512 for longer decode windows (more signal, less prefill noise)
#   - Trimmed mean: drops highest and lowest 2 values
#   - Reports 95% CI from the trimmed set
#   - Warns if concurrent GPU processes detected
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LAT_BIN="$REPO/target/release/bench_decode_ab"
MODEL_DIR="$HOME/.lattice/models/qwen3.5-0.8b"
N1=64
N2=512
TOTAL_RUNS=15
WARMUP=2
PROMPT="The quick brown fox jumps over the lazy dog. Once upon a time in a land far away, there lived a"
OUT="$REPO/target/bench_results"
mkdir -p "$OUT"
DATA="$OUT/precise_raw.tsv"
: > "$DATA"

# Warn about concurrent GPU load
GPU_PROCS=$(ps aux | grep -E "li play|mlx|ollama" | grep -v grep | wc -l | tr -d ' ')
if [ "$GPU_PROCS" -gt 0 ]; then
  echo "⚠  $GPU_PROCS concurrent GPU processes detected — results may have elevated noise"
fi

echo "=== Precise decode bench | Qwen3.5-0.8B | N1=$N1 N2=$N2 runs=$TOTAL_RUNS (warmup=$WARMUP) ==="

# ---- Lattice ----
if [[ -x "$LAT_BIN" ]]; then
  # Warmup
  for _ in $(seq 1 $WARMUP); do
    BENCH_N=$N2 BENCH_RUNS=1 LATTICE_MODEL_DIR="$MODEL_DIR" "$LAT_BIN" 2>/dev/null >/dev/null
  done
  RUNS=$((TOTAL_RUNS - WARMUP))
  for N in $N1 $N2; do
    BENCH_N=$N BENCH_RUNS=$RUNS LATTICE_MODEL_DIR="$MODEL_DIR" "$LAT_BIN" 2>/dev/null \
    | awk -v N=$N '/^RESULT/{
        for(i=1;i<=NF;i++){split($i,kv,"="); v[kv[1]]=kv[2]}
        printf "lattice\t%s\t%d\t%.6f\t\n", N, ++r, v["total_ms"]/1000.0
      }' >> "$DATA"
  done
  echo "  lattice: done ($RUNS measured runs)"
else
  echo "  lattice: BIN MISSING ($LAT_BIN)"
fi

# ---- Ollama ----
if command -v ollama >/dev/null 2>&1; then
  ollama list 2>/dev/null | grep -q "qwen3.5:0.8b" || ollama pull qwen3.5:0.8b >/dev/null 2>&1
  curl -s localhost:11434/api/tags >/dev/null 2>&1 || { ollama serve >/dev/null 2>&1 & sleep 3; }
  # Warmup
  for _ in $(seq 1 $WARMUP); do
    curl -s localhost:11434/api/generate -d "{\"model\":\"qwen3.5:0.8b\",\"prompt\":$(python3 -c "import json;print(json.dumps('$PROMPT'))"),\"stream\":false,\"options\":{\"num_predict\":$N2,\"temperature\":0}}" >/dev/null
  done
  RUNS=$((TOTAL_RUNS - WARMUP))
  for N in $N1 $N2; do
    for run in $(seq 1 $RUNS); do
      R=$(curl -s localhost:11434/api/generate -d "{\"model\":\"qwen3.5:0.8b\",\"prompt\":$(python3 -c "import json;print(json.dumps('$PROMPT'))"),\"stream\":false,\"options\":{\"num_predict\":$N,\"temperature\":0}}")
      echo "$R" | python3 -c "
import sys,json
d=json.load(sys.stdin)
tot=d.get('total_duration',0)/1e9
ec=d.get('eval_count',0); ed=d.get('eval_duration',1)/1e9
print(f'ollama\t$N\t$run\t{tot:.6f}\t{ec/ed:.3f}')" >> "$DATA"
    done
  done
  echo "  ollama: done ($RUNS measured runs)"
else
  echo "  ollama: not installed — skipped"
fi

# ---- MLX ----
uv run --quiet --with mlx-lm python3 - "$MODEL_DIR" "$N1" "$N2" "$TOTAL_RUNS" "$WARMUP" "$PROMPT" <<'PY' >> "$DATA" 2>/dev/null
import sys, time
mdir, n1, n2 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
total_runs, warmup = int(sys.argv[4]), int(sys.argv[5])
prompt = sys.argv[6]
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
try:
    model, tok = load(mdir)
except Exception:
    model, tok = load("Qwen/Qwen3.5-0.8B")
nn.quantize(model, bits=8, group_size=64); mx.eval(model.parameters())
samp = make_sampler(temp=0.0)
# Warmup
for _ in range(warmup):
    generate(model, tok, prompt=prompt, max_tokens=n2, sampler=samp, verbose=False)
runs = total_runs - warmup
for N in (n1, n2):
    for run in range(1, runs+1):
        t0=time.time()
        generate(model, tok, prompt=prompt, max_tokens=N, sampler=samp, verbose=False)
        dt=time.time()-t0
        print(f"mlx\t{N}\t{run}\t{dt:.6f}\t")
PY
echo "  mlx: done"

# ---- Analysis: trimmed mean + CI ----
python3 - "$DATA" "$N1" "$N2" <<'PY'
import sys, statistics as st, collections, math
data, N1, N2 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows=collections.defaultdict(lambda: collections.defaultdict(list))
nat=collections.defaultdict(list)
for ln in open(data):
    p=ln.rstrip("\n").split("\t")
    if len(p)<4 or not p[3]: continue
    eng,N,_,tot=p[0],int(p[1]),p[2],float(p[3])
    rows[eng][N].append(tot)
    if len(p)>=5 and p[4]: nat[eng].append(float(p[4]))

def trimmed_stats(vals, trim=2):
    """Trimmed mean + 95% CI from the middle values."""
    s = sorted(vals)
    if len(s) <= 2*trim:
        return st.mean(s), 0, s
    trimmed = s[trim:-trim]
    m = st.mean(trimmed)
    if len(trimmed) > 1:
        se = st.stdev(trimmed) / math.sqrt(len(trimmed))
        ci95 = 1.96 * se
    else:
        ci95 = 0
    return m, ci95, trimmed

print(f"\n{'engine':<10}{'slope tok/s':>13}{'±95% CI':>10}{'spread%':>10}{'native':>10}   (trimmed mean, {len(rows.get('lattice',{}).get(N2,[]))} runs)")
print("-"*64)
res={}
for eng in ("lattice","mlx","ollama"):
    if N1 not in rows[eng] or N2 not in rows[eng]:
        print(f"{eng:<10}{'(no data)':>13}"); continue
    t1_mean, t1_ci, t1_vals = trimmed_stats(rows[eng][N1])
    t2_mean, t2_ci, t2_vals = trimmed_stats(rows[eng][N2])
    dt = t2_mean - t1_mean
    slope = (N2-N1)/dt if dt > 0 else float('nan')
    # Propagate CI
    slope_ci = slope * math.sqrt((t1_ci/t1_mean)**2 + (t2_ci/t2_mean)**2) if t1_mean > 0 and t2_mean > 0 else 0
    spread = (max(t2_vals)-min(t2_vals))/t2_mean*100 if t2_vals else 0
    native = st.median(nat[eng]) if nat[eng] else float('nan')
    res[eng] = (slope, slope_ci)
    ns = f"{native:10.1f}" if native==native else f"{'—':>10}"
    print(f"{eng:<10}{slope:13.1f}{slope_ci:10.1f}{spread:9.1f}%{ns}")
print("-"*64)
if "lattice" in res and "mlx" in res:
    ls, lci = res["lattice"]; ms, mci = res["mlx"]
    gap = (1 - ls/ms)*100 if ms > 0 else float('nan')
    print(f"\nLattice vs MLX: {gap:.1f}% slower (±{lci:.1f}/{mci:.1f} tok/s CI)")
if "lattice" in res and "ollama" in res:
    ls, lci = res["lattice"]; os_, oci = res["ollama"]
    adv = (ls/os_ - 1)*100 if os_ > 0 else float('nan')
    print(f"Lattice vs Ollama: {adv:.1f}% faster")
print(f"\nRaw: {data}")
PY
