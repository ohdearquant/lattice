#!/usr/bin/env bash
# Fail-closed control-flow self-test for bench_quality.sh. It runs the real
# script in a disposable repository against stub lattice/MLX scorers; no model,
# GPU, network, cargo build, or benchmark is used.
set -uo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)/scripts/bench_quality.sh"
SRC_ROOT="${SRC%/scripts/bench_quality.sh}"
SB_ROOT="$(mktemp -d)"
SB="$SB_ROOT/repo"
trap 'chmod -R u+w "$SB_ROOT" 2>/dev/null; rm -rf "$SB_ROOT"' EXIT

mkdir -p "$SB/scripts/lib" "$SB/docs/bench_results" "$SB/target/release" \
  "$SB/q4" "$SB/quarot" "$SB/tokenizer" "$SB/fake-bin" "$SB/tmp"
cp "$SRC" "$SB/scripts/bench_quality.sh"
if ! cp \
  "$SRC_ROOT/scripts/lib/bench-supervision.sh" \
  "$SRC_ROOT/scripts/lib/bench-python.sh" \
  "$SRC_ROOT/scripts/lib/bench_supervision.py" \
  "$SRC_ROOT/scripts/lib/bench-locks.py" \
  "$SB/scripts/lib/"; then
  echo "failed to copy supervision helpers into fixture" >&2
  exit 1
fi

if ! cat > "$SB/scripts/lib/quiet-probe.py" <<'PY'
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

if len(sys.argv) != 3 or sys.argv[1] != "--label" or sys.argv[2] not in {
    "quality-perplexity: before",
    "quality-perplexity: after",
}:
    raise SystemExit(f"unexpected quiet-check arguments: {sys.argv[1:]}")
with Path(os.environ["BENCH_SELFTEST_QUIET_LOG"]).open("a", encoding="utf-8") as log:
    log.write(sys.argv[2] + "\n")
PY
then
  echo "failed to create fixture quiet-check shim" >&2
  exit 1
fi

if ! python3 - \
  "$SB/scripts/lib/bench-locks.py" \
  "$SB/bench-window.lock" \
  "$SB/metal-gpu.lock" \
  "$SB/bench-window-pending" <<'PY'
import re
import runpy
import sys
from pathlib import Path

path = Path(sys.argv[1])
names = ("BENCH_WINDOW", "GPU_LOCK", "PENDING_DIR")
values = sys.argv[2:]
source = path.read_text()
for name, value in zip(names, values):
    source, count = re.subn(
        rf'(?m)^{name} = "[^"]*"$',
        f"{name} = {value!r}",
        source,
        count=1,
    )
    if count != 1:
        raise SystemExit(f"expected one {name} assignment, found {count}")
path.write_text(source)
loaded = runpy.run_path(str(path))
if [loaded[name] for name in names] != values:
    raise SystemExit("fixture supervision paths are not disposable")
PY
then
  echo "failed to isolate supervision paths inside fixture" >&2
  exit 1
fi

CANONICAL="$SB/docs/bench_results/perplexity.tsv"
EXPECTED="$SB/expected.tsv"
CORPUS="$SB/docs/bench_results/wiki.test.raw"
MKTEMP_LOG="$SB/mktemp.log"
PUBLISH_LOG="$SB/publish.log"
QUIET_LOG="$SB/quiet.log"
REAL_MKTEMP="$(command -v mktemp)"
REAL_MV="$(command -v mv)"
printf "committed canonical sentinel\n" > "$EXPECTED"
: > "$QUIET_LOG"

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
  sink-fail)
    result="$(awk -F '\t' 'NR == 1 { print $2 }' "${MKTEMP_LOG:?}")"
    [[ -f "$result" ]] && chmod 0444 "$result"
    ;;
  sink-fail-late)
    if [[ " $* " == *" --quarot-q4-dir "* ]]; then
      result="$(awk -F '\t' 'NR == 1 { print $2 }' "${MKTEMP_LOG:?}")"
      if ! grep -qF $'lattice\tq4\t16.589166\t2048' "$result"; then
        echo "late sink-failure injection did not observe the q4 row" >&2
        exit 98
      fi
      chmod 0444 "$result"
    fi
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
  duplicate)
    printf "mlx\tq8\t15.8218\t2041\n"
    printf "mlx\tq4\t18.1839\t2041\n"
    printf "mlx\tq4\t18.1839\t2041\n"
    exit 0
    ;;
esac
printf "mlx\tq8\t15.8218\t2041\n"
printf "mlx\tq4\t18.1839\t2041\n"
EOF
chmod +x "$SB/fake-bin/uv"

cat > "$SB/fake-bin/mktemp" <<'EOF'
#!/usr/bin/env bash
result="$("${REAL_MKTEMP:?}" "$@")"
rc=$?
if [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi
if ! printf "%s\t%s\n" "${1-}" "$result" >> "${MKTEMP_LOG:?}"; then
  exit 97
fi
printf "%s\n" "$result"
EOF
chmod +x "$SB/fake-bin/mktemp"

cat > "$SB/fake-bin/mv" <<'EOF'
#!/usr/bin/env bash
source_exists=0
if [[ "$#" -eq 3 ]] && [[ "$1" == "-f" ]] && [[ -f "$2" ]]; then
  source_exists=1
fi
if ! printf "%s\t%s\t%s\t%s\n" "${1-}" "${2-}" "${3-}" "$source_exists" >> "${PUBLISH_LOG:?}"; then
  exit 97
fi
exec "${REAL_MV:?}" "$@"
EOF
chmod +x "$SB/fake-bin/mv"

run_bench() {
  : > "$MKTEMP_LOG"
  : > "$PUBLISH_LOG"
  OUT="$(
    cd "$SB" && env \
      -u LATTICE_BENCH_LOCK_STATUS \
      -u LATTICE_BENCH_LOCK_FDS \
      -u LATTICE_BENCH_SUPERVISOR_FD \
      -u LATTICE_BENCH_QUIET \
      PATH="$SB/fake-bin:$PATH" \
      TMPDIR="$SB/tmp" \
      Q4_DIR="$SB/q4" \
      QUAROT_DIR="$SB/quarot" \
      TOK_DIR="$SB/tokenizer" \
      BENCH_MACHINE="bench-quality-selftest" \
      BENCH_SELFTEST_QUIET_LOG="$QUIET_LOG" \
      MLX_LOG="$SB/mlx_ppl.log" \
      MKTEMP_LOG="$MKTEMP_LOG" \
      PUBLISH_LOG="$PUBLISH_LOG" \
      REAL_MKTEMP="$REAL_MKTEMP" \
      REAL_MV="$REAL_MV" \
      "$@" \
      bash scripts/bench_quality.sh 2>&1
  )"
  return $?
}

pass=0
fail=0
failure_preserved() {
  expected_rc="$1"
  actual_rc="$2"
  needle="$3"
  [[ "$actual_rc" -eq "$expected_rc" ]] \
    && grep -qF "$needle" <<<"$OUT" \
    && cmp -s "$EXPECTED" "$CANONICAL" \
    && [[ -z "$(find "$(dirname "$CANONICAL")" -maxdepth 1 -name '.perplexity.tsv.*' -print -quit)" ]]
}

check_failure_preserves() {
  desc="$1"
  expected_rc="$2"
  actual_rc="$3"
  needle="$4"
  if failure_preserved "$expected_rc" "$actual_rc" "$needle"; then
    echo "  PASS: $desc"
    pass=$((pass + 1))
  else
    echo "  FAIL: $desc — expected exit $expected_rc, preserved canonical, and '$needle'" >&2
    echo "        output: $(tr '\n' '|' <<<"$OUT" | tail -c 500)" >&2
    fail=$((fail + 1))
  fi
}

check_group_part() {
  desc="$1"
  expected_rc="$2"
  actual_rc="$3"
  needle="$4"
  if failure_preserved "$expected_rc" "$actual_rc" "$needle"; then
    echo "    OK: $desc"
  else
    echo "    FAIL: $desc" >&2
    echo "          output: $(tr '\n' '|' <<<"$OUT" | tail -c 500)" >&2
    GROUP_OK=0
  fi
}

echo "=== bench_quality.sh atomic-publication self-test ==="

cp "$EXPECTED" "$CANONICAL"
rm -f "$CORPUS"
run_bench
RC=$?
GROUP_OK=1
check_group_part "missing corpus preserves canonical" 1 "$RC" "wiki.test.raw not found"

echo "some corpus text" > "$CORPUS"
cp "$EXPECTED" "$CANONICAL"
mv "$SB/q4" "$SB/q4.missing"
run_bench
RC=$?
check_group_part "missing model preserves canonical" 1 "$RC" "Q4 model directory not found"
mv "$SB/q4.missing" "$SB/q4"

cp "$EXPECTED" "$CANONICAL"
mv "$SB/target/release/eval_perplexity" "$SB/target/release/eval_perplexity.missing"
run_bench
RC=$?
check_group_part "missing binary preserves canonical" 1 "$RC" "eval_perplexity is not executable"
mv "$SB/target/release/eval_perplexity.missing" "$SB/target/release/eval_perplexity"

if [[ "$GROUP_OK" -eq 1 ]]; then
  echo "  PASS: preflight failures preserve canonical"
  pass=$((pass + 1))
else
  echo "  FAIL: preflight failures did not all preserve canonical" >&2
  fail=$((fail + 1))
fi

cp "$EXPECTED" "$CANONICAL"
run_bench EVAL_MODE=fail
RC=$?
check_failure_preserves "lattice scorer failure preserves canonical" 1 "$RC" "lattice/q4 (exit 9)"

cp "$EXPECTED" "$CANONICAL"
run_bench EVAL_MODE=no-ppl
RC=$?
check_failure_preserves "unparseable lattice output preserves canonical" 1 "$RC" "no parseable PPL"

cp "$EXPECTED" "$CANONICAL"
run_bench MLX_MODE=fail
RC=$?
check_failure_preserves "MLX scorer failure preserves canonical" 1 "$RC" "MLX cross-check failed (exit 7"

cp "$EXPECTED" "$CANONICAL"
run_bench MLX_MODE=partial
RC=$?
GROUP_OK=1
check_group_part "incomplete MLX output preserves canonical" 1 "$RC" "exactly one q4 row"

cp "$EXPECTED" "$CANONICAL"
run_bench MLX_MODE=duplicate
RC=$?
check_group_part "duplicate MLX output preserves canonical" 1 "$RC" "exactly one q4 row"

if [[ "$GROUP_OK" -eq 1 ]]; then
  echo "  PASS: invalid MLX cardinality preserves canonical"
  pass=$((pass + 1))
else
  echo "  FAIL: invalid MLX cardinality did not preserve canonical" >&2
  fail=$((fail + 1))
fi

cp "$EXPECTED" "$CANONICAL"
run_bench MAX_TOKENS=$'2048\textra'
RC=$?
check_failure_preserves "invalid staged schema preserves canonical" 1 "$RC" "staged results failed schema/cardinality validation"

cp "$EXPECTED" "$CANONICAL"
run_bench EVAL_MODE=sink-fail
RC=$?
check_failure_preserves "staged output write failure preserves canonical" 1 "$RC" "failed to write lattice/q4 result"
chmod u+w "$CANONICAL" 2>/dev/null || true

cp "$EXPECTED" "$CANONICAL"
run_bench EVAL_MODE=sink-fail-late
RC=$?
check_failure_preserves "late staged output write failure preserves canonical" 1 "$RC" "failed to write lattice/q4-quarot result"
chmod u+w "$CANONICAL" 2>/dev/null || true

cp "$EXPECTED" "$CANONICAL"
run_bench
SUCCESS_RC=$?
CONTENT_OK=0
if [[ "$SUCCESS_RC" -eq 0 ]] \
  && grep -qF $'lattice\tq4\t16.589166\t2048' "$CANONICAL" \
  && grep -qF $'lattice\tq4-quarot\t19.007144\t2048' "$CANONICAL" \
  && grep -qF $'mlx\tq8\t15.8218\t2041' "$CANONICAL" \
  && grep -qF $'mlx\tq4\t18.1839\t2041' "$CANONICAL" \
  && grep -qF "machine=bench-quality-selftest" "$CANONICAL" \
  && grep -qE 'harness=bench_quality\.sh@[0-9a-f]{12}' "$CANONICAL" \
  && ! cmp -s "$EXPECTED" "$CANONICAL"; then
  CONTENT_OK=1
  echo "    OK: complete run publishes all rows with provenance"
fi

EXPECTED_TEMPLATE="$(dirname "$CANONICAL")/.perplexity.tsv.XXXXXX"
FIRST_TEMPLATE="$(awk -F '\t' 'NR == 1 { print $1 }' "$MKTEMP_LOG")"
TEMPLATE_OK=0
if [[ "$FIRST_TEMPLATE" == "$EXPECTED_TEMPLATE" ]]; then
  TEMPLATE_OK=1
  echo "    OK: result staging uses the destination-directory mktemp template"
fi

PUBLISH_COUNT="$(wc -l < "$PUBLISH_LOG" | tr -d '[:space:]')"
PUBLISH_FLAG="$(awk -F '\t' 'NR == 1 { print $1 }' "$PUBLISH_LOG")"
PUBLISH_SOURCE="$(awk -F '\t' 'NR == 1 { print $2 }' "$PUBLISH_LOG")"
PUBLISH_DEST="$(awk -F '\t' 'NR == 1 { print $3 }' "$PUBLISH_LOG")"
PUBLISH_SOURCE_EXISTS="$(awk -F '\t' 'NR == 1 { print $4 }' "$PUBLISH_LOG")"
PUBLISH_OK=0
if [[ "$PUBLISH_COUNT" -eq 1 ]] \
  && [[ "$PUBLISH_FLAG" == "-f" ]] \
  && [[ "$PUBLISH_SOURCE" == "$(dirname "$CANONICAL")/.perplexity.tsv."* ]] \
  && [[ "$PUBLISH_DEST" == "$CANONICAL" ]] \
  && [[ "$PUBLISH_SOURCE_EXISTS" == "1" ]]; then
  PUBLISH_OK=1
fi

cp "$EXPECTED" "$CANONICAL"
run_bench SKIP_MLX=1
SKIP_MLX_RC=$?
SKIP_MLX_ROWS="$(awk -F '\t' '!/^#/ { count++ } END { print count + 0 }' "$CANONICAL")"
SKIP_MLX_OK=0
if [[ "$SKIP_MLX_RC" -eq 0 ]] \
  && [[ "$SKIP_MLX_ROWS" -eq 2 ]] \
  && grep -qF $'lattice\tq4\t16.589166\t2048' "$CANONICAL" \
  && grep -qF $'lattice\tq4-quarot\t19.007144\t2048' "$CANONICAL" \
  && ! grep -q $'^mlx\t' "$CANONICAL"; then
  SKIP_MLX_OK=1
  echo "    OK: SKIP_MLX publishes exactly the two lattice rows"
fi

QUIET_OK=0
if grep -qFx "quality-perplexity: before" "$QUIET_LOG" \
  && grep -qFx "quality-perplexity: after" "$QUIET_LOG"; then
  QUIET_OK=1
  echo "    OK: durable supervision invokes both quiet checkpoints"
fi

if [[ "$CONTENT_OK" -eq 1 ]] \
  && [[ "$TEMPLATE_OK" -eq 1 ]] \
  && [[ "$PUBLISH_OK" -eq 1 ]] \
  && [[ "$SKIP_MLX_OK" -eq 1 ]] \
  && [[ "$QUIET_OK" -eq 1 ]]; then
  echo "  PASS: complete run renames same-directory staged result onto canonical"
  pass=$((pass + 1))
else
  echo "  FAIL: complete run renames same-directory staged result onto canonical" >&2
  echo "        content=$CONTENT_OK template=$TEMPLATE_OK publish=$PUBLISH_OK skip_mlx=$SKIP_MLX_OK quiet=$QUIET_OK exit=$SUCCESS_RC" >&2
  echo "        output: $(tr '\n' '|' <<<"$OUT" | tail -c 500)" >&2
  fail=$((fail + 1))
fi

echo ""
echo "=== $pass passed, $fail failed ==="
[[ "$pass" -eq 9 && "$fail" -eq 0 ]]
