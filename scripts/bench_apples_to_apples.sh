#!/usr/bin/env bash
# Apples-to-apple decode-throughput benchmark — slope (marginal) method.
#
# Measures three engines IDENTICALLY at two quantization tiers:
#   decode_tok_per_s = (N2 - N1) / (T_total(N2) - T_total(N1))
#
# Prefill, model load, and per-call overhead are constant for a fixed prompt
# so they cancel in the difference — leaving the true marginal decode rate,
# methodology-identical across engines.
#
# Q8 tier (lattice default for <2B params, ollama default, mlx explicit):
#   - lattice: F16 safetensors → auto-quantized to Q8_0 on Metal upload
#   - ollama:  qwen3.5:0.8b (registry default is Q8_0)
#   - mlx:     nn.quantize(bits=8, group_size=64)
#
# Q4 tier (lattice's product differentiator — QuaRot-rotated 4-bit):
#   - lattice: .q4 files in qwen3.5-0.8b-q4-quarot dir
#   - mlx:     nn.quantize(bits=4, group_size=64)
#   - ollama:  SKIPPED — no Q4 variant for qwen3.5:0.8b in registry
#
# MLX uses Apple's private MPS/MPSGraph (AMX matrix engines) — strictly a
# different category than public-Metal-compute engines (lattice, ollama).
# Reported for reference, not as the headline comparison.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LAT_BIN="$REPO/target/release/bench_decode_ab"
Q8_DIR="$HOME/.lattice/models/qwen3.5-0.8b"
Q4_DIR="$HOME/.lattice/models/qwen3.5-0.8b-q4-quarot"
N1=32
N2=256
RUNS=5
PROMPT="The quick brown fox jumps over the lazy dog. Once upon a time in a land far away, there lived a"
OUT="$REPO/docs/bench_results"
mkdir -p "$OUT"
DATA="$OUT/a2a_raw.tsv"   # tier<TAB>engine<TAB>N<TAB>run<TAB>total_s<TAB>native_tok_s
: > "$DATA"

echo "=== Apples-to-apple decode bench | Qwen3.5-0.8B | N1=$N1 N2=$N2 runs=$RUNS ==="
echo "  Output: $DATA"
echo ""

bench_lattice() {
  local tier="$1" model_dir="$2" tokenizer_dir="$3" quant_env="$4"
  if [[ ! -x "$LAT_BIN" ]]; then echo "  lattice/$tier: BIN MISSING — build first"; return; fi
  if [[ ! -d "$model_dir" ]]; then echo "  lattice/$tier: MODEL MISSING ($model_dir)"; return; fi
  for N in $N1 $N2; do
    env BENCH_N=$N BENCH_RUNS=$RUNS LATTICE_MODEL_DIR="$model_dir" \
        LATTICE_TOKENIZER_DIR="$tokenizer_dir" $quant_env "$LAT_BIN" 2>/dev/null \
    | awk -v tier="$tier" -v N=$N '/^RESULT/{
        for(i=1;i<=NF;i++){split($i,kv,"="); v[kv[1]]=kv[2]}
        printf "%s\tlattice\t%s\t%d\t%.6f\t\n", tier, N, ++r, v["total_ms"]/1000.0
      }' >> "$DATA"
  done
  echo "  lattice/$tier: done"
}

bench_ollama() {
  local tier="$1" model_tag="$2"
  if ! command -v ollama >/dev/null 2>&1; then echo "  ollama/$tier: not installed"; return; fi
  ollama list 2>/dev/null | grep -q "$model_tag" || ollama pull "$model_tag" >/dev/null 2>&1 || {
    echo "  ollama/$tier: pull failed for $model_tag — skipping"; return; }
  curl -s localhost:11434/api/tags >/dev/null 2>&1 || { ollama serve >/dev/null 2>&1 & sleep 3; }
  for N in $N1 $N2; do
    for run in $(seq 1 $RUNS); do
      R=$(curl -s localhost:11434/api/generate -d "{\"model\":\"$model_tag\",\"prompt\":$(python3 -c "import json,sys;print(json.dumps('$PROMPT'))"),\"stream\":false,\"options\":{\"num_predict\":$N,\"temperature\":0}}")
      echo "$R" | python3 -c "
import sys,json
d=json.load(sys.stdin)
tot=d.get('total_duration',0)/1e9
ec=d.get('eval_count',0); ed=d.get('eval_duration',1)/1e9
print(f'$tier\tollama\t$N\t$run\t{tot:.6f}\t{ec/ed:.3f}')" >> "$DATA"
    done
  done
  echo "  ollama/$tier: done"
}

bench_mlx() {
  local tier="$1" bits="$2"
  uv run --quiet --with mlx-lm python3 - "$Q8_DIR" "$N1" "$N2" "$RUNS" "$PROMPT" "$tier" "$bits" <<'PY' >> "$DATA" 2>/dev/null
import sys, time
mdir, n1, n2, runs, prompt, tier, bits = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], int(sys.argv[7])
import mlx.core as mx, mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
try:    model, tok = load(mdir)
except: model, tok = load("Qwen/Qwen3.5-0.8B")
nn.quantize(model, bits=bits, group_size=64); mx.eval(model.parameters())
samp = make_sampler(temp=0.0)
generate(model, tok, prompt=prompt, max_tokens=8, sampler=samp, verbose=False)  # warmup
for N in (n1, n2):
    for run in range(1, runs+1):
        t0 = time.time()
        generate(model, tok, prompt=prompt, max_tokens=N, sampler=samp, verbose=False)
        dt = time.time() - t0
        print(f"{tier}\tmlx\t{N}\t{run}\t{dt:.6f}\t")
PY
  echo "  mlx/$tier: done"
}

echo "─── Q8 tier (apples — public Metal compute API) ───"
bench_lattice "q8" "$Q8_DIR" "$Q8_DIR" ""
bench_ollama  "q8" "qwen3.5:0.8b"
echo "─── Q8 reference (MLX uses private MPS/AMX — different category) ───"
bench_mlx     "q8" "8"
echo ""
echo "─── Q4 tier (lattice differentiator — QuaRot rotation) ───"
bench_lattice "q4" "$Q4_DIR" "$Q8_DIR" "LATTICE_QUANT_FORMAT=Q4"
echo "─── Q4 reference (MLX Q4 g64) ───"
bench_mlx     "q4" "4"
echo ""

# ---- Slope + report ----
python3 - "$DATA" "$N1" "$N2" <<'PY'
import sys, statistics as st, collections
data, N1, N2 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
# rows[tier][engine][N] = list[total_s]
rows=collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
nat =collections.defaultdict(lambda: collections.defaultdict(list))
for ln in open(data):
    p=ln.rstrip("\n").split("\t")
    if len(p)<5 or not p[4]: continue
    tier,eng,N,_,tot = p[0],p[1],int(p[2]),p[3],float(p[4])
    rows[tier][eng][N].append(tot)
    if len(p)>=6 and p[5]: nat[tier][eng].append(float(p[5]))

def slope(t1,t2): return (N2-N1)/(t2-t1) if t2>t1 else float('nan')

def report_tier(tier, engines, header):
    if not any(N1 in rows[tier][e] and N2 in rows[tier][e] for e in engines): return None
    print(f"\n{header}")
    print(f"  {'engine':<10}{'slope tok/s':>13}{'native tok/s':>15}{'naive N2/T2':>14}")
    print("  " + "-"*52)
    out={}
    for eng in engines:
        if N1 not in rows[tier][eng] or N2 not in rows[tier][eng]:
            print(f"  {eng:<10}{'(no data)':>13}"); continue
        t1,t2 = st.median(rows[tier][eng][N1]), st.median(rows[tier][eng][N2])
        s = slope(t1,t2); naive = N2/t2
        native = st.median(nat[tier][eng]) if nat[tier][eng] else float('nan')
        out[eng] = s
        ns = f"{native:13.1f}" if native==native else f"{'—':>13}"
        print(f"  {eng:<10}{s:13.1f}{ns:>15}{naive:14.1f}")
    return out

q8 = report_tier("q8", ["lattice","ollama","mlx"], "═══ Q8 tier ═══")
q4 = report_tier("q4", ["lattice","mlx"],          "═══ Q4 tier (lattice product differentiator) ═══")

print("\n═══ Apples-to-apple verdict (public Metal API only) ═══")
if q8 and "lattice" in q8 and "ollama" in q8 and q8["ollama"]>0:
    d = (q8["lattice"]/q8["ollama"]-1)*100
    print(f"  Q8: Lattice is {abs(d):.1f}% {'FASTER' if d>0 else 'SLOWER'} than Ollama")
if q4 and "lattice" in q4:
    print(f"  Q4: Lattice {q4['lattice']:.1f} tok/s (no public-API Q4 peer — Ollama lacks 0.8b Q4)")

print("\n═══ Reference (different category — AMX private API) ═══")
if q8 and "lattice" in q8 and "mlx" in q8 and q8["mlx"]>0:
    d = (q8["lattice"]/q8["mlx"]-1)*100
    print(f"  Q8 vs MLX: Lattice is {abs(d):.1f}% {'FASTER' if d>0 else 'SLOWER'} (MLX uses AMX)")
if q4 and "lattice" in q4 and "mlx" in q4 and q4["mlx"]>0:
    d = (q4["lattice"]/q4["mlx"]-1)*100
    print(f"  Q4 vs MLX: Lattice is {abs(d):.1f}% {'FASTER' if d>0 else 'SLOWER'} (MLX uses AMX)")

print(f"\nRaw data: {data}")
PY
