#!/bin/sh
set -eu

case "${GITHUB_EVENT_NAME:-}" in
    pull_request | merge_group | push) ;;
    *)
        echo "unsupported change-detection event: ${GITHUB_EVENT_NAME:-<unset>}" >&2
        exit 2
        ;;
esac

base_sha=${CI_BASE_SHA:-}
head_sha=${CI_HEAD_SHA:-}

if [ -z "$base_sha" ] || [ -z "$head_sha" ]; then
    echo "CI_BASE_SHA and CI_HEAD_SHA are required" >&2
    exit 2
fi

case "$base_sha$head_sha" in
    *[!0-9a-fA-F]*)
        echo "CI_BASE_SHA and CI_HEAD_SHA must be hexadecimal commit IDs" >&2
        exit 2
        ;;
esac

actual_head=$(git rev-parse HEAD)
if [ "$actual_head" != "$head_sha" ]; then
    echo "checked-out HEAD $actual_head does not match event head $head_sha" >&2
    exit 1
fi

zero_sha=0000000000000000000000000000000000000000
if [ "$GITHUB_EVENT_NAME" = "push" ] && [ "$base_sha" = "$zero_sha" ]; then
    changed=$(git -c core.quotePath=false ls-tree -r --name-only "$head_sha")
else
    if ! git cat-file -e "${base_sha}^{commit}" 2>/dev/null; then
        echo "event base $base_sha is not available as a commit" >&2
        exit 1
    fi

    if ! git cat-file -e "${head_sha}^{commit}" 2>/dev/null; then
        echo "event head $head_sha is not available as a commit" >&2
        exit 1
    fi

    if ! git merge-base --is-ancestor "$base_sha" "$head_sha"; then
        echo "event base $base_sha is not an ancestor of event head $head_sha" >&2
        exit 1
    fi

    changed=$(git -c core.quotePath=false diff --name-only --no-renames "${base_sha}..${head_sha}")
fi

case "$changed" in
    \"* | *'
"'*)
        echo "changed path requires Git quoting; refusing ambiguous classification" >&2
        exit 1
        ;;
esac

printf '%s\n' "$changed"
