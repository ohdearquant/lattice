#!/usr/bin/env bash
# Fail-closed control-flow self-test for bench_quality.sh. It runs the real
# script in a disposable repository against stub lattice/MLX scorers; no model,
# GPU, network, cargo build, or benchmark is used.
set -uo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)/scripts/bench_quality.sh"
SB_ROOT="$(mktemp -d)"
SB="$SB_ROOT/repo"
trap 'chmod -R u+w "$SB_ROOT" 2>/dev/null; rm -rf "$SB_ROOT"' EXIT

mkdir -p "$SB/scripts" "$SB/docs/bench_results" "$SB/target/release" \
  "$SB/q4" "$SB/quarot" "$SB/tokenizer" "$SB/fake-bin"
cp "$SRC" "$SB/scripts/bench_quality.sh"

CANONICAL="$SB/docs/bench_results/perplexity.tsv"
EXPECTED="$SB/expected.tsv"
CORPUS="$SB/docs/bench_results/wiki.test.raw"
printf "committed canonical sentinel\n" > "$EXPECTED"

cat > "$SB/target/release/eval_perplexity" <<'EOF'
#!/usr/bin/env bash
case "${EVAL_MODE:-ok}" in
  fail)
    echo "injected lattice scorer failure" >&2
    exit 9
    ;;
  no-ppl)
    echo "scoring completed without a PPL record"
    exit 0
    ;;
esac
if [[ " $* " == *" --quarot-q4-dir "* ]]; then
  echo "PPL:                19.007144"
else
  echo "PPL:                16.589166"
fi
EOF
chmod +x "$SB/target/release/eval_perplexity"

cat > "$SB/fake-bin/uv" <<'EOF'
#!/usr/bin/env bash
case "${MLX_MODE:-ok}" in
  fail)
    echo "injected MLX scorer failure" >&2
    exit 7
    ;;
  partial)
    printf "mlx\tq8\t15.8218\t2041\n"
    exit 0
    ;;
esac
printf "mlx\tq8\t15.8218\t2041\n"
printf "mlx\tq4\t18.1839\t2041\n"
EOF
chmod +x "$SB/fake-bin/uv"

run_bench() {
  OUT="$(
    cd "$SB" && env \
      PATH="$SB/fake-bin:$PATH" \
      Q4_DIR="$SB/q4" \
      QUAROT_DIR="$SB/quarot" \
      TOK_DIR="$SB/tokenizer" \
      BENCH_MACHINE="bench-quality-selftest" \
      "$@" \
      bash scripts/bench_quality.sh 2>&1
  )"
  return $?
}

pass=0
fail=0
check_failure_preserves() {
  desc="$1"
  expected_rc="$2"
  actual_rc="$3"
  needle="$4"
  if [[ "$actual_rc" -eq "$expected_rc" ]] \
    && grep -qF "$needle" <<<"$OUT" \
    && cmp -s "$EXPECTED" "$CANONICAL" \
    && [[ -z "$(find "$(dirname "$CANONICAL")" -maxdepth 1 -name '.perplexity.tsv.*' -print -quit)" ]]; then
    echo "  PASS: $desc"
    pass=$((pass + 1))
  else
    echo "  FAIL: $desc — expected exit $expected_rc, preserved canonical, and '$needle'" >&2
    echo "        output: $(tr '\n' '|' <<<"$OUT" | tail -c 500)" >&2
    fail=$((fail + 1))
  fi
}

echo "=== bench_quality.sh atomic-publication self-test ==="

cp "$EXPECTED" "$CANONICAL"
rm -f "$CORPUS"
run_bench
check_failure_preserves "missing corpus preserves canonical" 1 "$?" "wiki.test.raw not found"

echo "some corpus text" > "$CORPUS"
cp "$EXPECTED" "$CANONICAL"
mv "$SB/q4" "$SB/q4.missing"
run_bench
check_failure_preserves "missing model preserves canonical" 1 "$?" "Q4 model directory not found"
mv "$SB/q4.missing" "$SB/q4"

cp "$EXPECTED" "$CANONICAL"
mv "$SB/target/release/eval_perplexity" "$SB/target/release/eval_perplexity.missing"
run_bench
check_failure_preserves "missing binary preserves canonical" 1 "$?" "eval_perplexity is not executable"
mv "$SB/target/release/eval_perplexity.missing" "$SB/target/release/eval_perplexity"

cp "$EXPECTED" "$CANONICAL"
run_bench EVAL_MODE=fail
check_failure_preserves "lattice scorer failure preserves canonical" 1 "$?" "lattice/q4 (exit 9)"

cp "$EXPECTED" "$CANONICAL"
run_bench EVAL_MODE=no-ppl
check_failure_preserves "unparseable lattice output preserves canonical" 1 "$?" "no parseable PPL"

cp "$EXPECTED" "$CANONICAL"
run_bench MLX_MODE=fail
check_failure_preserves "MLX scorer failure preserves canonical" 1 "$?" "MLX cross-check failed (exit 7"

cp "$EXPECTED" "$CANONICAL"
run_bench MLX_MODE=partial
check_failure_preserves "incomplete MLX output preserves canonical" 1 "$?" "exactly one q4 row"

cp "$EXPECTED" "$CANONICAL"
run_bench
SUCCESS_RC=$?
if [[ "$SUCCESS_RC" -eq 0 ]] \
  && grep -qF $'lattice\tq4\t16.589166\t2048' "$CANONICAL" \
  && grep -qF $'lattice\tq4-quarot\t19.007144\t2048' "$CANONICAL" \
  && grep -qF $'mlx\tq8\t15.8218\t2041' "$CANONICAL" \
  && grep -qF $'mlx\tq4\t18.1839\t2041' "$CANONICAL" \
  && grep -qF "machine=bench-quality-selftest" "$CANONICAL" \
  && grep -qE 'harness=bench_quality\.sh@[0-9a-f]{12}' "$CANONICAL" \
  && ! cmp -s "$EXPECTED" "$CANONICAL"; then
  echo "  PASS: complete run atomically publishes all rows with provenance"
  pass=$((pass + 1))
else
  echo "  FAIL: complete run did not publish the expected rows and provenance (exit $SUCCESS_RC)" >&2
  echo "        output: $(tr '\n' '|' <<<"$OUT" | tail -c 500)" >&2
  fail=$((fail + 1))
fi

echo ""
echo "=== $pass passed, $fail failed ==="
[[ "$fail" -eq 0 ]]
