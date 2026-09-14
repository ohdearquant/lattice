#!/usr/bin/env bash
# Prove that scripts/lib/package-size-bracket.py can fail, and fails for the
# right reasons. A size gate that only ever prints a comfortable number is
# decoration: the two states worth guarding are an archive over the limit and a
# measurement that cannot be taken at all, and both must be visible in the exit
# code, not only in the text.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/bench-python.sh
. "${REPO_ROOT}/scripts/lib/bench-python.sh"
PYTHON_BIN="$(bench_require_python3 "package-size-check-selftest.sh")" || exit 1
BRACKET="${REPO_ROOT}/scripts/lib/package-size-bracket.py"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

failures=0

arm() {
    # arm <name> <expected-rc> <expected-substring> <list-file>
    local name="$1" want_rc="$2" want_text="$3" list="$4"
    local out rc=0
    out="$("$PYTHON_BIN" "$BRACKET" fixture-crate "$list" 2>&1)" || rc=$?
    if [ "$rc" -ne "$want_rc" ]; then
        echo "package-size-check-selftest: ${name}: expected rc ${want_rc}, got ${rc}" >&2
        echo "  output: ${out}" >&2
        failures=$((failures + 1))
        return
    fi
    case "$out" in
        *"$want_text"*) echo "package-size-check-selftest: ${name} OK (rc ${rc})" ;;
        *)
            echo "package-size-check-selftest: ${name}: output did not mention '${want_text}'" >&2
            echo "  output: ${out}" >&2
            failures=$((failures + 1))
            ;;
    esac
}

printf 'Cargo.toml\n' > Cargo.toml.list
printf '[package]\nname = "fixture-crate"\n' > Cargo.toml

# 1. An empty list is an instrument failure, never a crate of size zero.
: > empty.list
arm "empty list refuses" 1 "empty file list" empty.list

# 2. A listed file that is not on disk is the wrong-directory signature. Summing
#    what remains would report a small crate, so the only sound answer is to stop.
printf 'Cargo.toml\nsrc/lib.rs\n' > absent.list
arm "absent file refuses" 1 "not on disk" absent.list

# 3. A small crate passes and says so.
arm "small crate reports ok" 0 "[ok]" Cargo.toml.list

# 4. An archive past the limit is caught. The payload is incompressible on
#    purpose: a fixture of zeros would gzip to nothing and the arm would pass
#    while proving the opposite of what it claims to prove.
"$PYTHON_BIN" -c 'import os; open("payload.bin","wb").write(os.urandom(11 * 1024 * 1024))'
printf 'Cargo.toml\npayload.bin\n' > over.list
arm "over-limit crate refuses" 1 "OVER LIMIT" over.list

if [ "$failures" -ne 0 ]; then
    echo "package-size-check-selftest: ${failures} arm(s) failed" >&2
    exit 1
fi
echo "package-size-check-selftest: all arms passed"
