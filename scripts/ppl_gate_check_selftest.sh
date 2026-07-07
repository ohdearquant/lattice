#!/usr/bin/env bash
# Self-test for scripts/ppl_gate_check.py control flow (issue #616).
#
# The gate is a fail-closed CI guard; this proves each branch's exit code in an
# isolated sandbox (a fake REPO tree + a stub eval_perplexity), touching neither
# the real repo nor the GPU. Run locally or in the cheap CI lane:
#   bash scripts/ppl_gate_check_selftest.sh
#
# Guards, in order: RECORD-on-null, ENFORCE pass, ENFORCE fail, and every
# fail-closed path (binary error even in RECORD, missing corpus/dir, unparsed
# output, env-record cannot mask an armed golden, require-armed rejects a null
# golden on the required leg, non-finite golden/tolerance).
set -uo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)/scripts/ppl_gate_check.py"
SB="$(mktemp -d)/repo"
mkdir -p "$SB/scripts" "$SB/target/release" \
         "$SB/crates/inference/tests/fixtures/ppl_gate_v1" \
         "$SB/docs/bench_results" "$SB/q4dir" "$SB/tokdir"
cp "$SRC" "$SB/scripts/ppl_gate_check.py"
echo "some corpus text" > "$SB/docs/bench_results/wiki.test.raw"
touch "$SB/q4dir/config.json" "$SB/tokdir/tokenizer.json"

golden() {  # $1 = ppl (number or the literal null), $2 = tolerance (default 0.05)
  cat > "$SB/crates/inference/tests/fixtures/ppl_gate_v1/golden.json" <<EOF
{ "model":"m","tier":"q4","corpus":"docs/bench_results/wiki.test.raw",
  "window":512,"stride":256,"max_tokens":2048,"ppl":$1,"tolerance":${2:-0.05} }
EOF
}

stub() {  # $1 = emitted ppl, $2 = exit code
  cat > "$SB/target/release/eval_perplexity" <<EOF
#!/usr/bin/env bash
echo '@@lattice {"ev":"perplexity","label":"q4","ppl":$1,"nll":2.8,"tokens":2047,"windows":7,"ms":100}'
exit $2
EOF
  chmod +x "$SB/target/release/eval_perplexity"
}

run() {  # extra env as "KEY=VAL" args; captures combined output to $OUT, returns rc
  OUT="$(cd "$SB" && env LATTICE_PPL_Q4_DIR="$SB/q4dir" \
        LATTICE_PPL_TOKENIZER_DIR="$SB/tokdir" \
        PPL_GATE_REPORT="$SB/report.md" "$@" \
        python3 scripts/ppl_gate_check.py 2>&1)"
  return $?
}

pass=0; fail=0
check() {  # $1=desc $2=expected_exit $3=actual_exit $4=must_contain
  if [ "$2" = "$3" ] && grep -qF "$4" <<<"$OUT"; then
    echo "  PASS: $1 (exit $3)"; pass=$((pass+1))
  else
    echo "  FAIL: $1 — expected exit $2 got $3; wanted substring: $4"
    echo "        output: $(tr '\n' '|' <<<"$OUT" | tail -c 300)"
    fail=$((fail+1))
  fi
}

echo "=== ppl_gate_check.py control-flow self-test ==="

golden null;   stub 16.60 0; run;                          check "RECORD/null golden -> exit0 UNARMED"        0 $? "GATE UNARMED"
golden 16.60;  stub 16.61 0; run;                          check "ENFORCE within tol -> exit0 PASS"           0 $? "verdict: **PASS**"
golden 16.60;  stub 17.00 0; run;                          check "ENFORCE out of tol -> exit1 FAIL"           1 $? "Q4 PPL regressed"
golden null;   stub 16.60 3; run;                          check "ERROR binary rc=3 in RECORD -> exit1"       1 $? "eval_perplexity exited 3"
golden 16.60;  stub 16.61 0; run LATTICE_PPL_GATE_RECORD=1; check "env-record cannot mask armed golden (pass)" 0 $? "verdict: **PASS**"
golden 16.60;  stub 17.00 0; run LATTICE_PPL_GATE_RECORD=1; check "env-record cannot mask armed regression"    1 $? "Q4 PPL regressed"
golden null;   stub 16.60 0; run LATTICE_PPL_GATE_REQUIRE_ARMED=1; check "require-armed + null golden -> exit1" 1 $? "required gate and must be armed"
golden 16.60;  stub 16.61 0; run LATTICE_PPL_GATE_REQUIRE_ARMED=1; check "require-armed + armed golden enforces" 0 $? "verdict: **PASS**"
golden 16.60 1e999; stub 17.00 0; run;                     check "non-finite tolerance rejected -> exit1"     1 $? "finite positive"
golden -1 0.05; stub 16.60 0; run;                         check "non-positive golden ppl rejected -> exit1"  1 $? "finite positive"

# provisioning / parse failures must all fail closed
golden 16.60; stub 16.61 0; rm -f "$SB/docs/bench_results/wiki.test.raw"; run
check "corpus missing -> exit1" 1 $? "corpus not found"
echo "some corpus text" > "$SB/docs/bench_results/wiki.test.raw"

golden 16.60; stub 16.61 0
OUT="$(cd "$SB" && env LATTICE_PPL_Q4_DIR="$SB/nonexistent" \
      LATTICE_PPL_TOKENIZER_DIR="$SB/tokdir" PPL_GATE_REPORT="$SB/report.md" \
      python3 scripts/ppl_gate_check.py 2>&1)"
check "q4 dir missing -> exit1" 1 $? "not a directory"

# path resolution: $VAR / ~ in the env value must be expanded, not taken literally
golden null; stub 16.60 0
OUT="$(cd "$SB" && env LATTICE_PPL_Q4_DIR="$SB/q4dir" REALTOK="$SB/tokdir" \
      LATTICE_PPL_TOKENIZER_DIR='${REALTOK}' PPL_GATE_REPORT="$SB/report.md" \
      python3 scripts/ppl_gate_check.py 2>&1)"
check "env-var in path is expanded (not literal) -> exit0" 0 $? "measured PPL"

golden 16.60; stub 16.61 0
printf '#!/usr/bin/env bash\necho "no structured event"\nexit 0\n' > "$SB/target/release/eval_perplexity"
chmod +x "$SB/target/release/eval_perplexity"; run
check "no event line -> exit1" 1 $? "no @@lattice perplexity event"

echo ""
echo "=== $pass passed, $fail failed ==="
rm -rf "$(dirname "$SB")"
[ "$fail" = 0 ]
