#!/bin/sh
# ADR-095 decision 4, item 3: the OTHER direction.
#
# The per-binary presence tests prove every entry in `LORA_ROUTES` is
# registered. They cannot see a route registered in a binary and never added
# to the list -- nothing iterates it. So the set of `/v1/lora*` paths each
# binary REGISTERS must equal the other's and equal the list.
#
# WHY REGISTRATIONS AND NOT EVERY `/v1/lora` LITERAL. The decision's wording is
# "path literals", and taken at face value that is wrong here: both binaries'
# test modules contain `/v1/lora*` literals in request fixtures, including a
# deliberately unregistered path used as a must-not-match control. A lint over
# every literal would flag that control as drift, and the obvious repair --
# delete the control -- would remove the only thing proving the presence test
# can fail. So this reads `.route("...")` occurrences: the registration set is
# the set the decision is actually about.
#
# FAIL CLOSED ON AN EMPTY SET. Three empty sets are equal, so a broken
# extractor reports a clean tree while checking nothing. Each set is asserted
# non-empty before any comparison, and the selftest runs a known-positive
# through the same extractor the real run uses.
#
# Exit codes:
#   0 = the three sets agree
#   1 = drift (printed)
#   2 = lint failure (a file missing, an extractor producing nothing) -- never
#       a silent pass
#
# Usage:
#   scripts/lint-lora-routes.sh
#   scripts/lint-lora-routes.sh --selftest

set -u

LIST_FILE=crates/inference/src/serve/lora.rs
BIN_A=crates/inference/src/bin/lattice/serve.rs
BIN_B=crates/inference/src/bin/lattice_serve.rs

# Paths registered via `.route("/v1/lora...")` in a binary's source.
extract_registered() {
    grep -o '\.route("/v1/lora[^"]*"' "$1" 2>/dev/null |
        sed 's/^\.route("//; s/"$//' | sort -u
}

# Paths in the LORA_ROUTES const. Bounded to the const's own body so an
# unrelated `/v1/lora` literal elsewhere in the file cannot join the set.
extract_list() {
    sed -n '/^pub const LORA_ROUTES/,/^];/p' "$1" 2>/dev/null |
        grep -o '("/v1/lora[^"]*"' | sed 's/^("//; s/"$//' | sort -u
}

run_lint() {
    list_file=$1
    bin_a=$2
    bin_b=$3
    rc=0

    for f in "$list_file" "$bin_a" "$bin_b"; do
        if [ ! -f "$f" ]; then
            echo "lint-lora-routes: $f does not exist" >&2
            return 2
        fi
    done

    list=$(extract_list "$list_file")
    a=$(extract_registered "$bin_a")
    b=$(extract_registered "$bin_b")

    for pair in "list:$list" "$bin_a:$a" "$bin_b:$b"; do
        name=${pair%%:*}
        value=${pair#*:}
        if [ -z "$value" ]; then
            echo "lint-lora-routes: extracted NO routes from $name; refusing to compare empty sets" >&2
            return 2
        fi
    done

    # `<(...)` is a bashism; this script declares /bin/sh, so the comparison
    # uses real files rather than quietly requiring bash.
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/lattice-lora-cmp.XXXXXX") || {
        echo "lint-lora-routes: could not create a tempdir for the comparison" >&2
        return 2
    }
    printf '%s\n' "$list" > "$tmp/list"
    printf '%s\n' "$a" > "$tmp/a"
    printf '%s\n' "$b" > "$tmp/b"
    if [ "$a" != "$list" ]; then
        echo "lint-lora-routes: $bin_a registers a different set than LORA_ROUTES:"
        diff "$tmp/list" "$tmp/a" | sed 's/^/  /'
        rc=1
    fi
    if [ "$b" != "$list" ]; then
        echo "lint-lora-routes: $bin_b registers a different set than LORA_ROUTES:"
        diff "$tmp/list" "$tmp/b" | sed 's/^/  /'
        rc=1
    fi
    rm -rf "$tmp"
    [ "$rc" -eq 0 ] && echo "lint-lora-routes: clean ($(echo "$list" | wc -l | tr -d ' ') routes, both binaries agree)"
    return "$rc"
}

selftest() {
    sandbox=$(mktemp -d "${TMPDIR:-/tmp}/lattice-lint-lora.XXXXXX") || {
        echo "lint-lora-routes selftest: could not create a sandbox" >&2
        return 2
    }
    mkdir -p "$sandbox"

    cat > "$sandbox/list.rs" <<'EOF'
pub const LORA_ROUTES: &[(&str, &[&str])] = &[
    ("/v1/lora", &["GET"]),
    ("/v1/lora/load", &["POST"]),
];
EOF
    # A decoy literal outside the const: the extractor must not pick it up.
    echo 'let decoy = "/v1/lora/not-in-the-const";' >> "$sandbox/list.rs"

    cat > "$sandbox/good_a.rs" <<'EOF'
    .route("/v1/lora", get(lora_list))
    .route("/v1/lora/load", post(lora_load))
    let fixture = "/v1/lora/definitely-not-a-route";
EOF
    cp "$sandbox/good_a.rs" "$sandbox/good_b.rs"

    # MUST-MATCH CONTROL, first: the extractor finds a route it is supposed to
    # find. Every negative result below is meaningless without this passing.
    found=$(extract_registered "$sandbox/good_a.rs")
    case "$found" in
        */v1/lora/load*) : ;;
        *)
            echo "lint-lora-routes selftest: extractor found no known route; every other arm is void" >&2
            rm -rf "$sandbox"
            return 2 ;;
    esac
    if echo "$found" | grep -q 'definitely-not-a-route'; then
        echo "lint-lora-routes selftest: extractor picked up a non-registration literal" >&2
        rm -rf "$sandbox"
        return 2
    fi
    if extract_list "$sandbox/list.rs" | grep -q 'not-in-the-const'; then
        echo "lint-lora-routes selftest: list extractor reached outside the const body" >&2
        rm -rf "$sandbox"
        return 2
    fi
    echo "  PASS  the extractors find registrations and ignore neighbouring literals"

    run_lint "$sandbox/list.rs" "$sandbox/good_a.rs" "$sandbox/good_b.rs" >/dev/null
    [ $? -eq 0 ] || { echo "lint-lora-routes selftest: an agreeing trio was not clean" >&2; rm -rf "$sandbox"; return 2; }
    echo "  PASS  an agreeing trio is clean"

    # A route registered in one binary and absent from the list: the exact
    # direction the presence tests cannot see.
    cp "$sandbox/good_a.rs" "$sandbox/extra.rs"
    echo '    .route("/v1/lora/feedback", post(lora_feedback))' >> "$sandbox/extra.rs"
    run_lint "$sandbox/list.rs" "$sandbox/extra.rs" "$sandbox/good_b.rs" >/dev/null 2>&1
    [ $? -eq 1 ] || { echo "lint-lora-routes selftest: an unlisted route was not flagged" >&2; rm -rf "$sandbox"; return 2; }
    echo "  PASS  a route registered but not listed is flagged"

    # One binary missing a listed route: the presence test catches this too,
    # and the lint must agree rather than contradict it.
    printf '    .route("/v1/lora", get(lora_list))\n' > "$sandbox/short.rs"
    run_lint "$sandbox/list.rs" "$sandbox/short.rs" "$sandbox/good_b.rs" >/dev/null 2>&1
    [ $? -eq 1 ] || { echo "lint-lora-routes selftest: a missing route was not flagged" >&2; rm -rf "$sandbox"; return 2; }
    echo "  PASS  a binary missing a listed route is flagged"

    # Fail-closed arm: an extractor that yields nothing must refuse, not pass.
    : > "$sandbox/empty.rs"
    run_lint "$sandbox/list.rs" "$sandbox/empty.rs" "$sandbox/good_b.rs" >/dev/null 2>&1
    [ $? -eq 2 ] || { echo "lint-lora-routes selftest: an empty extraction did not fail closed" >&2; rm -rf "$sandbox"; return 2; }
    echo "  PASS  an empty extraction refuses instead of comparing empty sets"

    run_lint "$sandbox/list.rs" "$sandbox/good_a.rs" "$sandbox/nonexistent.rs" >/dev/null 2>&1
    [ $? -eq 2 ] || { echo "lint-lora-routes selftest: a missing file did not fail closed" >&2; rm -rf "$sandbox"; return 2; }
    echo "  PASS  a missing source file refuses"

    rm -rf "$sandbox"
    echo "lint-lora-routes selftest: all checks passed"
    return 0
}

if [ "${1:-}" = "--selftest" ]; then
    selftest
    exit $?
fi

run_lint "$LIST_FILE" "$BIN_A" "$BIN_B"
exit $?
