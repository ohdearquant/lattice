#!/bin/sh
set -e

script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
script_path="$script_dir/lint-docs.sh"
mode=${1:-}

clear_local_git_environment() {
    git_local_env_names=$(git rev-parse --local-env-vars)
    if [ -n "$git_local_env_names" ]; then
        # Git emits only variable names; word splitting supplies them to unset.
        unset $git_local_env_names
    fi
}

run_discovery_selftest() (
    clear_local_git_environment

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
    echo "lint-docs: recursive tracked-Markdown selftest OK"
)

run_index_isolation_selftest() (
    clear_local_git_environment

    sandbox=$(mktemp -d "${TMPDIR:-/tmp}/lattice-lint-docs-index.XXXXXX")
    case "$sandbox" in
        "${TMPDIR:-/tmp}"/lattice-lint-docs-index.*) ;;
        *)
            echo "lint-docs index selftest: unexpected sandbox path: $sandbox" >&2
            exit 1
            ;;
    esac
    trap 'rm -rf "$sandbox"' 0 1 2 3 15

    repo="$sandbox/repo"
    before_index="$sandbox/before-index"
    after_index="$sandbox/after-index"
    tracked_files="$sandbox/tracked-files"
    committed_files="$sandbox/committed-files"
    hook_marker="$sandbox/hook-ran"
    global_config="$sandbox/global-config"
    mkdir -p "$repo/sentinel"
    printf '%s\n' '# outer repository' >"$repo/README.md"
    printf '%s\n' 'keep one' >"$repo/sentinel/one.txt"
    printf '%s\n' 'keep two' >"$repo/sentinel/two.md"
    git -C "$repo" init -q
    git -C "$repo" add README.md sentinel
    git -C "$repo" ls-files --stage -z >"$before_index"
    git config --file "$global_config" core.hooksPath /dev/null

    cat >"$repo/.git/hooks/pre-commit" <<'EOF'
#!/bin/sh
set -e
LATTICE_LINT_DOCS_INDEX_PROBE=1 \
    "${LATTICE_LINT_DOCS_SELFTEST_SCRIPT:?}" --selftest
: >"${LATTICE_LINT_DOCS_HOOK_MARKER:?}"
EOF
    chmod +x "$repo/.git/hooks/pre-commit"
    (
        cd "$repo"
        LATTICE_LINT_DOCS_SELFTEST_SCRIPT="$script_path" \
            LATTICE_LINT_DOCS_HOOK_MARKER="$hook_marker" \
            GIT_CONFIG_GLOBAL="$global_config" \
            GIT_INDEX_FILE="$repo/.git/index" \
            git -c core.hooksPath="$repo/.git/hooks" \
            -c user.name=selftest -c user.email=selftest@example.invalid \
            commit -qm "lint-docs hook index isolation probe"
    )

    if [ ! -f "$hook_marker" ]; then
        echo "lint-docs index selftest: synthetic pre-commit hook did not complete" >&2
        exit 1
    fi
    git -C "$repo" ls-files --stage -z >"$after_index"
    if ! cmp -s "$before_index" "$after_index"; then
        echo "lint-docs index selftest: staged entries changed under hook context" >&2
        exit 1
    fi
    git -C "$repo" ls-files -z >"$tracked_files"
    git -C "$repo" ls-tree -r --name-only -z HEAD >"$committed_files"
    if ! cmp -s "$tracked_files" "$committed_files"; then
        echo "lint-docs index selftest: committed files differ from the index" >&2
        exit 1
    fi
    echo "lint-docs: hook-index isolation selftest OK"
)

case "$mode" in
    --selftest)
        if [ "${LATTICE_LINT_DOCS_INDEX_PROBE:-}" = "1" ]; then
            run_discovery_selftest
        else
            run_index_isolation_selftest
        fi
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

echo "=== Doc Lint Passed ==="
