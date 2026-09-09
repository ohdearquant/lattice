#!/bin/sh
# Every tracked .rs file must be reached by a formatting gate.
#
# `cargo fmt --all` formats WORKSPACE MEMBERS. Measured on this repo: appending a
# deliberately misformatted fn to crates/inference/src/lib.rs makes
# `cargo fmt --all -- --check` exit 1 and name the file, while the identical
# mutation in npm/lattice-embed-native/src/lib.rs exits 0 and names nothing --
# that package is not in the members list. So "we run cargo fmt --all" and "every
# Rust file we ship is formatted" are different claims, and the gap between them
# is silent: nothing tells the gate the file exists.
#
# This closes it by enumeration. Every tracked *.rs outside every workspace member
# must be listed in scripts/rust-fmt-coverage.allow with a reason, and every listed
# file is then run through `rustfmt --check` here. The allowlist is a gate, not an
# exemption -- an entry buys the file a rustfmt run, it does not buy it a pass.
#
# Usage: scripts/lint-rust-fmt-coverage.sh [--selftest]

set -eu

ALLOW_REL="scripts/rust-fmt-coverage.allow"

# --- decision logic, kept free of I/O so the selftest can drive it directly ---

# uncovered_files <members-newline-list> <files-newline-list>
# Prints the files that lie under no member directory. Path-boundary safe: the
# member `crates/inference` does not cover `crates/inference-extra/src/x.rs`.
uncovered_files() {
    _members=$1
    _files=$2
    printf '%s\n' "$_files" | while IFS= read -r f; do
        [ -n "$f" ] || continue
        _hit=no
        for d in $_members; do
            case "$f" in
                "$d"/*) _hit=yes; break ;;
            esac
        done
        [ "$_hit" = no ] && printf '%s\n' "$f"
    done
    return 0
}

# A row is: <path> <edition> <reason...>. The edition is stated per row rather
# than derived from the nearest Cargo.toml, because for these files the nearest
# manifest is not the one they are built under: scripts/microlora/dsl_validator.rs
# sits under a 2024-edition workspace and is compiled by its generator in an
# ephemeral 2021-edition crate. Checking it at 2024 would reorder its imports away
# from the only edition it ever compiles in.
ALLOW_EDITIONS="2015 2018 2021 2024"

# allow_paths <allowlist-text>: paths from an allowlist, comments and blanks out.
allow_paths() {
    printf '%s\n' "$1" | sed 's/#.*//' | awk 'NF { print $1 }'
}

# allow_edition <allowlist-text> <path>
allow_edition() {
    printf '%s\n' "$1" | sed 's/#.*//' | awk -v p="$2" '$1 == p { print $2; exit }'
}

# allow_bad_rows <allowlist-text>: rows missing an edition, a reason, or naming an
# edition rustfmt would not take.
allow_bad_rows() {
    printf '%s\n' "$1" | sed 's/#.*//' | awk -v eds="$ALLOW_EDITIONS" '
        NF > 0 {
            if (NF < 3) { print $1; next }
            ok = 0
            n = split(eds, e, " ")
            for (i = 1; i <= n; i++) if ($2 == e[i]) ok = 1
            if (!ok) print $1
        }'
}

# allow_dupes <allowlist-text>
allow_dupes() {
    allow_paths "$1" | sort | uniq -d
}

# --- selftest ---------------------------------------------------------------

selftest() {
    _fail=0
    arm() { # arm <name> <got> <want>
        if [ "$2" = "$3" ]; then
            echo "  PASS  $1"
        else
            echo "  FAIL  $1"
            echo "        got  [$2]"
            echo "        want [$3]"
            _fail=1
        fi
    }

    M='crates/inference
crates/embed'

    arm "a file under a member is covered" \
        "$(uncovered_files "$M" 'crates/inference/src/lib.rs')" ""
    arm "a file under no member is uncovered" \
        "$(uncovered_files "$M" 'scripts/x/y.rs')" "scripts/x/y.rs"
    arm "a member name is not a substring match" \
        "$(uncovered_files "$M" 'crates/inference-extra/src/x.rs')" \
        "crates/inference-extra/src/x.rs"
    arm "the member directory itself is not a file under it" \
        "$(uncovered_files "$M" 'crates/inference')" "crates/inference"
    arm "mixed input separates cleanly" \
        "$(uncovered_files "$M" 'crates/embed/src/a.rs
npm/p/src/b.rs
crates/inference/src/c.rs')" "npm/p/src/b.rs"

    A='# a comment line
scripts/x.rs   2021  not a workspace member, built out of tree

npm/p/src/b.rs 2024  shipped via npm, package is outside the workspace'
    arm "allowlist skips comments and blanks" \
        "$(allow_paths "$A")" "scripts/x.rs
npm/p/src/b.rs"
    arm "the row states the edition" \
        "$(allow_edition "$A" 'npm/p/src/b.rs')" "2024"
    arm "an unlisted path has no edition" \
        "$(allow_edition "$A" 'nope.rs')" ""
    arm "a row with only a path is rejected" \
        "$(allow_bad_rows 'scripts/x.rs')" "scripts/x.rs"
    arm "a row with an edition but no reason is rejected" \
        "$(allow_bad_rows 'scripts/x.rs 2021')" "scripts/x.rs"
    arm "a complete row is not rejected" \
        "$(allow_bad_rows 'scripts/x.rs 2021 because reasons')" ""
    arm "an edition rustfmt would not take is rejected" \
        "$(allow_bad_rows 'scripts/x.rs 2020 because reasons')" "scripts/x.rs"
    arm "a trailing comment is not a reason" \
        "$(allow_bad_rows 'scripts/x.rs 2021 # because reasons')" "scripts/x.rs"
    arm "duplicate entries are caught" \
        "$(allow_dupes 'a.rs 2021 one
a.rs 2021 two')" "a.rs"
    arm "distinct entries are not duplicates" \
        "$(allow_dupes 'a.rs 2021 one
b.rs 2021 two')" ""

    # rustfmt really rejects a misformatted file, and really passes a clean one:
    # without this pair the allowlist could be gating with a dead instrument.
    _t=$(mktemp -d)
    printf 'fn main() {\n    let x = 1;\n    println!("{x}");\n}\n' > "$_t/clean.rs"
    printf 'fn  main( ){let x=1;println!("{x}");}\n' > "$_t/dirty.rs"
    rustfmt --check --edition 2021 "$_t/clean.rs" >/dev/null 2>&1 && _c=0 || _c=$?
    rustfmt --check --edition 2021 "$_t/dirty.rs" >/dev/null 2>&1 && _d=0 || _d=$?
    rm -rf "$_t"
    arm "rustfmt passes a clean file" "$_c" "0"
    arm "rustfmt rejects a misformatted file (instrument is alive)" \
        "$([ "$_d" -ne 0 ] && echo nonzero || echo zero)" "nonzero"

    # The edition column changes the verdict, so it is not decoration: the 2024
    # style edition sorts `Value` ahead of `json`, the 2021 one does not.
    _t2=$(mktemp -d)
    printf 'use serde_json::{json, Value};\n\nfn main() {\n    let _ = (json!(1), Value::Null);\n}\n' > "$_t2/ed.rs"
    rustfmt --check --edition 2021 "$_t2/ed.rs" >/dev/null 2>&1 && _e21=0 || _e21=$?
    rustfmt --check --edition 2024 "$_t2/ed.rs" >/dev/null 2>&1 && _e24=0 || _e24=$?
    rm -rf "$_t2"
    arm "the same file is clean under edition 2021" "$_e21" "0"
    arm "and not clean under edition 2024 (the column decides)" \
        "$([ "$_e24" -ne 0 ] && echo nonzero || echo zero)" "nonzero"

    if [ "$_fail" -eq 0 ]; then
        echo "lint-rust-fmt-coverage: selftest OK"
        return 0
    fi
    echo "lint-rust-fmt-coverage: SELFTEST FAILED"
    return 1
}

# --- main -------------------------------------------------------------------

if [ "${1:-}" = "--selftest" ]; then
    selftest
    exit $?
fi

cd "$(git rev-parse --show-toplevel)"

command -v cargo >/dev/null 2>&1 || { echo "lint-rust-fmt-coverage: cargo is required" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "lint-rust-fmt-coverage: jq is required" >&2; exit 1; }
command -v rustfmt >/dev/null 2>&1 || { echo "lint-rust-fmt-coverage: rustfmt is required (run 'make setup')" >&2; exit 1; }

MEMBERS=$(cargo metadata --no-deps --offline --format-version 1 \
    | jq -r '.packages[].manifest_path' \
    | sed "s|^$PWD/||; s|/Cargo\.toml$||" \
    | sort -u)
[ -n "$MEMBERS" ] || { echo "lint-rust-fmt-coverage: cargo metadata named no workspace member" >&2; exit 1; }

FILES=$(git ls-files -- '*.rs')
[ -n "$FILES" ] || { echo "lint-rust-fmt-coverage: no tracked .rs files found; refusing to pass on an empty enumeration" >&2; exit 1; }

# A path with whitespace would split silently in the loops above. None exists
# today; refuse rather than mis-enumerate if one ever lands.
if printf '%s\n' "$FILES" | grep -q '[[:space:]]'; then
    echo "lint-rust-fmt-coverage: a tracked .rs path contains whitespace; this check cannot enumerate it safely" >&2
    printf '%s\n' "$FILES" | grep '[[:space:]]' >&2
    exit 1
fi

UNCOVERED=$(uncovered_files "$MEMBERS" "$FILES")

ALLOW_TEXT=""
[ -f "$ALLOW_REL" ] && ALLOW_TEXT=$(cat "$ALLOW_REL")

status=0

BAD=$(allow_bad_rows "$ALLOW_TEXT")
if [ -n "$BAD" ]; then
    echo "$ALLOW_REL: these rows are not '<path> <edition> <reason>' with a known edition:"
    printf '  %s\n' $BAD
    status=1
fi

DUPES=$(allow_dupes "$ALLOW_TEXT")
if [ -n "$DUPES" ]; then
    echo "$ALLOW_REL: duplicate entries:"
    printf '  %s\n' $DUPES
    status=1
fi

ALLOWED=$(allow_paths "$ALLOW_TEXT")

for f in $UNCOVERED; do
    listed=no
    for a in $ALLOWED; do
        [ "$f" = "$a" ] && listed=yes && break
    done
    if [ "$listed" = no ]; then
        echo "$f is a tracked Rust file that no workspace member owns, so 'cargo fmt --all' never sees it."
        echo "  Put it under a workspace member, or add a '<path> <edition> <reason>' row to $ALLOW_REL."
        status=1
    fi
done

for a in $ALLOWED; do
    if [ ! -f "$a" ]; then
        echo "$ALLOW_REL: $a is listed but does not exist; remove the row."
        status=1
        continue
    fi
    still=no
    for f in $UNCOVERED; do
        [ "$f" = "$a" ] && still=yes && break
    done
    if [ "$still" = no ]; then
        echo "$ALLOW_REL: $a is now owned by a workspace member and is covered by 'cargo fmt --all'; remove the row."
        status=1
        continue
    fi
    ed=$(allow_edition "$ALLOW_TEXT" "$a")
    if ! rustfmt --check --edition "$ed" "$a"; then
        echo "$a is not rustfmt-clean (edition $ed). Run: rustfmt --edition $ed $a"
        status=1
    fi
done

n_allowed=$(printf '%s\n' "$ALLOWED" | awk 'NF' | wc -l | tr -d ' ')
n_files=$(printf '%s\n' "$FILES" | wc -l | tr -d ' ')
if [ "$status" -eq 0 ]; then
    echo "lint-rust-fmt-coverage: $n_files tracked .rs files, $n_allowed outside the workspace and rustfmt-checked here"
fi
exit "$status"
