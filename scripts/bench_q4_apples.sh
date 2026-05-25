#!/usr/bin/env bash
# Quant-MATCHED apples-to-apple decode benchmark — slope (marginal) method, Q4.
#
# Companion to bench_apples_to_apples.sh (which does f16 Lattice vs Q8 MLX/Ollama).
# This one is the honest quant-matched comparison Ocean asked for:
#
#   Lattice Q4 (plain per-row, ~/.lattice/models/qwen3.5-0.8b-q4)
#   vs MLX Q4  (nn.quantize bits=4 group_size=64, same source weights)
#   Ollama     = Q8_0 REFERENCE ONLY — no Q4 tag exists for qwen3.5:0.8b
#               (verified via `ollama show`), so it is NOT a Q4 competitor.
#
# Methodology identical for every engine — the slope cancels prefill, model
# load, and fixed per-call overhead at a fixed prompt:
#
#   decode_tok_per_s = (N2 - N1) / (T_total(N2) - T_total(N1))
#
# Cross-check: each engine's slope vs its own native decode rate where it
# reports one (ollama eval_duration). Lattice Q4 output was eyeballed for
# coherence before trusting any number (see target/q4_coherence.log).
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LAT_BIN="$REPO/target/release/bench_decode_ab"
Q4_DIR="$HOME/.lattice/models/qwen3.5-0.8b-q4"     # plain Q4, tied embeddings
TOK_DIR="$HOME/.lattice/models/qwen3.5-0.8b"        # tokenizer.json + f16 src
N1=32
N2=256
RUNS=5
PROMPT="The quick brown fox jumps over the lazy dog. Once upon a time in a land far away, there lived a"
OUT="$REPO/target/bench_results"
mkdir -p "$OUT"
DATA="$OUT/q4_a2a_raw.tsv"      # engine<TAB>N<TAB>run<TAB>total_s<TAB>native_tok_s
: > "$DATA"

echo "=== Q4 apples-to-apple decode bench | Qwen3.5-0.8B | N1=$N1 N2=$N2 runs=$RUNS ==="
echo "    Lattice=Q4 plain | MLX=Q4 g64 | Ollama=Q8_0 (reference, no Q4 tag)"

# ---- Lattice (real Metal e2e Q4 path, same loader as chat_metal Q4 branch) ----
if [[ -x "$LAT_BIN" ]]; then
  for N in $N1 $N2; do
    BENCH_N=$N BENCH_RUNS=$RUNS BENCH_Q4_DIR="$Q4_DIR" BENCH_TOKENIZER_DIR="$TOK_DIR" \
      "$LAT_BIN" 2>/dev/null \
    | awk -v N=$N '/^RESULT/{
        for(i=1;i<=NF;i++){split($i,kv,"="); v[kv[1]]=kv[2]}
        printf "lattice\t%s\t%d\t%.6f\t\n", N, ++r, v["total_ms"]/1000.0
      }' >> "$DATA"
  done
  echo "  lattice Q4: done"
else
  echo "  lattice: BIN MISSING ($LAT_BIN) — build first"
fi

# ---- Ollama (Q8_0 reference — no Q4 tag for qwen3.5:0.8b) ----
if command -v ollama >/dev/null 2>&1; then
  ollama list 2>/dev/null | grep -q "qwen3.5:0.8b" || ollama pull qwen3.5:0.8b >/dev/null 2>&1
  curl -s localhost:11434/api/tags >/dev/null 2>&1 || { ollama serve >/dev/null 2>&1 & sleep 3; }
  for N in $N1 $N2; do
    for run in $(seq 1 $RUNS); do
      R=$(curl -s localhost:11434/api/generate -d "{\"model\":\"qwen3.5:0.8b\",\"prompt\":$(python3 -c "import json,sys;print(json.dumps('$PROMPT'))"),\"stream\":false,\"options\":{\"num_predict\":$N,\"temperature\":0}}")
      echo "$R" | python3 -c "
import sys,json
d=json.load(sys.stdin)
tot=d.get('total_duration',0)/1e9
ec=d.get('eval_count',0); ed=d.get('eval_duration',1)/1e9
print(f'ollama\t$N\t$run\t{tot:.6f}\t{ec/ed:.3f}')" >> "$DATA"
    done
  done
  echo "  ollama Q8 (ref): done"
else
  echo "  ollama: not installed — skipped"
fi

# ---- MLX (Apple framework, Q4 g64 — quant-matched to Lattice Q4) ----
uv run --quiet --with mlx-lm python3 - "$TOK_DIR" "$N1" "$N2" "$RUNS" "$PROMPT" <<'PY' >> "$DATA" 2>/dev/null
import sys, time
mdir, n1, n2, runs, prompt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
try:
    model, tok = load(mdir)            # same local f16 weights as Lattice Q4 source
except Exception:
    model, tok = load("Qwen/Qwen3.5-0.8B")
nn.quantize(model, bits=4, group_size=64); mx.eval(model.parameters())
samp = make_sampler(temp=0.0)
generate(model, tok, prompt=prompt, max_tokens=8, sampler=samp, verbose=False)  # warmup
for N in (n1, n2):
    for run in range(1, runs+1):
        t0=time.time()
        generate(model, tok, prompt=prompt, max_tokens=N, sampler=samp, verbose=False)
        dt=time.time()-t0
        print(f"mlx\t{N}\t{run}\t{dt:.6f}\t")
PY
echo "  mlx Q4: done"

# ---- Slope + report (identical math to bench_apples_to_apples.sh) ----
python3 - "$DATA" "$N1" "$N2" <<'PY'
import sys, statistics as st, collections
data, N1, N2 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows=collections.defaultdict(lambda: collections.defaultdict(list))
nat=collections.defaultdict(list)
for ln in open(data):
    p=ln.rstrip("\n").split("\t")
    if len(p)<4 or not p[3]: continue
    eng,N,_,tot=p[0],int(p[1]),p[2],float(p[3])
    rows[eng][N].append(tot)
    if len(p)>=5 and p[4]: nat[eng].append(float(p[4]))
labels={"lattice":"Lattice Q4","mlx":"MLX Q4","ollama":"Ollama Q8*"}
print(f"\n{'engine':<12}{'slope tok/s':>13}{'native tok/s':>15}{'naive N2/T2':>14}   (median of runs)")
print("-"*68)
res={}
for eng in ("lattice","mlx","ollama"):
    if N1 not in rows[eng] or N2 not in rows[eng]:
        print(f"{labels.get(eng,eng):<12}{'(no data)':>13}"); continue
    t1,t2=st.median(rows[eng][N1]),st.median(rows[eng][N2])
    slope=(N2-N1)/(t2-t1) if t2>t1 else float('nan')
    naive=N2/t2
    native=st.median(nat[eng]) if nat[eng] else float('nan')
    res[eng]=slope
    ns=f"{native:13.1f}" if native==native else f"{'—':>13}"
    print(f"{labels.get(eng,eng):<12}{slope:13.1f}{ns:>15}{naive:14.1f}")
print("-"*68)
print("* Ollama is Q8_0 (no Q4 tag for qwen3.5:0.8b) — reference, NOT a Q4 competitor.")
if "lattice" in res and "mlx" in res and res["mlx"]>0:
    d=(res["lattice"]/res["mlx"]-1)*100
    verb="FASTER" if d>0 else "SLOWER"
    print(f"\nQuant-matched (slope, Q4): Lattice is {abs(d):.1f}% {verb} than MLX")
if "lattice" in res and "ollama" in res and res["ollama"]>0:
    d=(res["lattice"]/res["ollama"]-1)*100
    print(f"Lattice Q4 vs Ollama Q8 (cross-quant ref): {abs(d):.1f}% {'FASTER' if d>0 else 'SLOWER'}")
print(f"\nRaw: {data}")
PY
