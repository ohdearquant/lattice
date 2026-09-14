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

# 5. A symlink must be measured at its target's size. Cargo dereferences when it
#    packages, so a link header of zero bytes undercounts the archive, and the
#    gate then passes a crate the registry rejects. Measured on a crate cargo
#    archived at 10,490,174 bytes from one 5 MiB file reached through a link:
#    before the fix this bracket reported 5,245,520 bytes and called it ok.
mkdir -p linkdir
"$PYTHON_BIN" -c 'import os; open("linkdir/real.bin","wb").write(os.urandom(11 * 1024 * 1024))'
ln -s real.bin linkdir/link.bin
printf 'linkdir/link.bin\n' > symlink.list
arm "symlink counted at target size" 1 "OVER LIMIT" symlink.list

# 6-7. The wrapper itself, not just the bracket: crate discovery, the cargo
#      invocation, and what it does when cargo refuses. A stub cargo stands in
#      for the real one so this needs no Rust toolchain and packages nothing.
STUB_DIR="${WORK}/stub"
mkdir -p "$STUB_DIR" "${WORK}/stubcrate"
printf '[package]\nname = "stub-crate"\n' > "${WORK}/stubcrate/Cargo.toml"
printf 'fn main() {}\n' > "${WORK}/stubcrate/main.rs"

write_stub_cargo() {
    # $1: exit status for `cargo package --list`
    cat > "${STUB_DIR}/cargo" <<STUB
#!/bin/sh
case "\$1" in
  metadata)
    printf '{"packages":[{"name":"stub-crate","publish":null,"manifest_path":"%s/Cargo.toml"}]}' "${WORK}/stubcrate"
    ;;
  package)
    if [ "$1" -ne 0 ]; then
      echo "error: stub cargo refuses this tree" >&2
      exit $1
    fi
    printf 'Cargo.toml\nmain.rs\n'
    ;;
  *) exit 64 ;;
esac
STUB
    chmod +x "${STUB_DIR}/cargo"
}

wrapper_arm() {
    # wrapper_arm <name> <expected-rc> <expected-substring>
    local name="$1" want_rc="$2" want_text="$3" out rc=0
    out="$(PATH="${STUB_DIR}:${PATH}" "${REPO_ROOT}/scripts/package-size-check.sh" 2>&1)" || rc=$?
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

write_stub_cargo 0
wrapper_arm "wrapper measures a discovered crate" 0 "stub-crate"
write_stub_cargo 101
wrapper_arm "wrapper propagates a cargo refusal" 1 "stub cargo refuses this tree"

if [ "$failures" -ne 0 ]; then
    echo "package-size-check-selftest: ${failures} arm(s) failed" >&2
    exit 1
fi
echo "package-size-check-selftest: all arms passed"
