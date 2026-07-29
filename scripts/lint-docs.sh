#!/bin/sh
set -e

unset GIT_INDEX_FILE GIT_DIR GIT_WORK_TREE

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
script_path="$script_dir/lint-docs.sh"
mode=${1:-}

run_discovery_selftest() {
    index_count_before=$(git ls-files | wc -l | tr -d '[:space:]')
    sandbox=$(mktemp -d "${TMPDIR:-/tmp}/lattice-lint-docs.XXXXXX")
    case "$sandbox" in
        "${TMPDIR:-/tmp}"/lattice-lint-docs.*) ;;
        *)
            echo "lint-docs selftest: unexpected sandbox path: $sandbox" >&2
            exit 1
            ;;
    esac
    trap 'rm -rf "$sandbox"' 0 1 2 3 15

    repo="$sandbox/repo"
    capture="$sandbox/capture"
    expected="$sandbox/expected"
    mkdir -p "$repo/bin" "$repo/docs/adr" "$repo/crates/inference/docs/deep"
    printf '%s\n' '# root' >"$repo/README.md"
    printf '%s\n' '# depth one' >"$repo/docs/one.md"
    printf '%s\n' '# depth two' >"$repo/docs/adr/two.md"
    printf '%s\n' '# depth four' >"$repo/crates/inference/docs/deep/four.md"
    printf '%s\n' '# spaced' >"$repo/docs/space name.md"
    printf '%s\n' '# untracked' >"$repo/untracked.md"
    git -C "$repo" init -q
    git -C "$repo" add README.md docs crates

    cat >"$repo/bin/deno" <<'EOF'
#!/bin/sh
set -e

mode=${1:-}
shift
case "$mode" in
    fmt)
        if [ "${1:-}" != "--check" ]; then
            echo "lint-docs selftest: formatter missing --check" >&2
            exit 1
        fi
        shift
        ;;
    lint) ;;
    *)
        echo "lint-docs selftest: unexpected deno mode: $mode" >&2
        exit 1
        ;;
esac
if [ "$#" -eq 0 ]; then
    echo "lint-docs selftest: deno received no Markdown paths" >&2
    exit 1
fi

output="${LINT_DOCS_CAPTURE:?}.${mode}"
: >"$output"
for path in "$@"; do
    printf '%s\0' "$path" >>"$output"
done
EOF
    chmod +x "$repo/bin/deno"

    (
        cd "$repo"
        PATH="$repo/bin:$PATH" LINT_DOCS_CAPTURE="$capture" \
            "$script_path" --markdown-only
    )
    git -C "$repo" ls-files -z -- '*.md' >"$expected"

    if ! cmp -s "$expected" "$capture.fmt"; then
        echo "lint-docs selftest: formatter did not receive the exact tracked Markdown set" >&2
        exit 1
    fi
    if ! cmp -s "$expected" "$capture.lint"; then
        echo "lint-docs selftest: linter did not receive the exact tracked Markdown set" >&2
        exit 1
    fi
    index_count_after=$(git ls-files | wc -l | tr -d '[:space:]')
    if [ "$index_count_after" -ne "$index_count_before" ]; then
        echo "lint-docs selftest: real index entry count changed: $index_count_before -> $index_count_after" >&2
        exit 1
    fi
    echo "lint-docs: recursive tracked-Markdown selftest OK"
}

case "$mode" in
    --selftest)
        run_discovery_selftest
        exit 0
        ;;
    "" | --format | --markdown-only) ;;
    *)
        echo "usage: $0 [--format|--markdown-only|--selftest]" >&2
        exit 64
        ;;
esac

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
markdown_list=$(mktemp "${TMPDIR:-/tmp}/lattice-markdown-files.XXXXXX")
trap 'rm -f "$markdown_list"' 0 1 2 3 15
git ls-files -z -- '*.md' >"$markdown_list"
markdown_count=$(tr -cd '\000' <"$markdown_list" | wc -c | tr -d '[:space:]')
if [ "$markdown_count" -eq 0 ]; then
    echo "lint-docs: tracked Markdown discovery returned zero files" >&2
    exit 1
fi

if [ "$mode" = "--format" ]; then
    echo "=== Formatting $markdown_count tracked Markdown files (deno) ==="
    xargs -0 deno fmt <"$markdown_list"
    exit 0
fi

echo "=== Doc Linting $markdown_count tracked Markdown files (deno) ==="
xargs -0 deno fmt --check <"$markdown_list"
xargs -0 deno lint <"$markdown_list" 2>/dev/null || true

if [ "$mode" = "--markdown-only" ]; then
    exit 0
fi

echo "=== Recursive Markdown Discovery Self-Test (#1148) ==="
"$script_path" --selftest

echo "=== Capability Matrix Fixture Check (#654) ==="
"$script_dir/check-capability-matrix.sh" --selftest
"$script_dir/check-capability-matrix.sh"

echo "=== Absolute Developer Path Check (#1102) ==="
"$script_dir/lint-absolute-paths.sh" --selftest
"$script_dir/lint-absolute-paths.sh"

echo "=== Doc Lint Passed ==="
