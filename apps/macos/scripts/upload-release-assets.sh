#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <release-tag> <owner/repo> <expected-tag-sha> <artifact-dir>" >&2
    exit 2
fi

if [[ "${GITHUB_EVENT_NAME:-}" != "workflow_dispatch" ]]; then
    echo "ERROR: release assets may only upload from a release workflow dispatch" >&2
    exit 1
fi

TAG="$1"
REPOSITORY="$2"
EXPECTED_SHA="$3"
ARTIFACT_DIR="$4"

if [[ ! "$TAG" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
    echo "ERROR: release tag must use vMAJOR.MINOR.PATCH form" >&2
    exit 1
fi
VERSION="${BASH_REMATCH[1]}"
if [[ ! "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: expected tag SHA must be a lowercase 40-character commit SHA" >&2
    exit 1
fi
if [[ ! "$REPOSITORY" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
    echo "ERROR: repository must use owner/name form" >&2
    exit 1
fi

PAYLOAD_NAMES=(
    "lattice-$VERSION-aarch64-apple-darwin.tar.gz"
    "lattice-$VERSION-x86_64-unknown-linux-gnu.tar.gz"
    "lattice-$VERSION-aarch64-unknown-linux-gnu.tar.gz"
    "Lattice.dmg"
    "Lattice.zip"
)
EXPECTED_NAMES=()
for NAME in "${PAYLOAD_NAMES[@]}"; do
    EXPECTED_NAMES+=("$NAME" "$NAME.sha256")
done

verify_directory() {
    local directory="$1"
    local actual_count=0
    local found
    local expected
    local name
    local checksum
    local digest
    local checksum_name
    local extra

    for expected in "${EXPECTED_NAMES[@]}"; do
        if [[ ! -s "$directory/$expected" ]]; then
            echo "ERROR: release asset is missing or empty: $directory/$expected" >&2
            return 1
        fi
    done

    while IFS= read -r found; do
        name="${found##*/}"
        actual_count=$((actual_count + 1))
        local matched=false
        for expected in "${EXPECTED_NAMES[@]}"; do
            if [[ "$name" == "$expected" ]]; then
                matched=true
                break
            fi
        done
        if [[ "$matched" != true ]]; then
            echo "ERROR: unexpected release asset: $name" >&2
            return 1
        fi
    done < <(find "$directory" -maxdepth 1 -type f -print)
    if [[ "$actual_count" -ne "${#EXPECTED_NAMES[@]}" ]]; then
        echo "ERROR: release inventory has $actual_count files; expected ${#EXPECTED_NAMES[@]}" >&2
        return 1
    fi

    for name in "${PAYLOAD_NAMES[@]}"; do
        checksum="$directory/$name.sha256"
        digest=""
        checksum_name=""
        extra=""
        read -r digest checksum_name extra < "$checksum" || true
        if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]] ||
            [[ "$checksum_name" != "$name" ]] || [[ -n "$extra" ]]; then
            echo "ERROR: malformed checksum inventory entry: $checksum" >&2
            return 1
        fi
        if ! (cd "$directory" && shasum -a 256 -c "$name.sha256" >/dev/null); then
            echo "ERROR: checksum verification failed for $name" >&2
            return 1
        fi
    done
}

upload_directory() {
    local directory="$1"
    local upload_args=()
    local name
    for name in "${EXPECTED_NAMES[@]}"; do
        upload_args+=("$directory/$name")
    done
    gh release upload "$TAG" \
        "${upload_args[@]}" \
        --repo "$REPOSITORY" \
        --clobber
}

download_and_verify() {
    local directory="$1"
    mkdir -p "$directory"
    if ! gh release download "$TAG" \
        --repo "$REPOSITORY" \
        --dir "$directory"; then
        echo "ERROR: could not download the complete release asset set" >&2
        return 1
    fi
    verify_directory "$directory"
}

verify_directory "$ARTIFACT_DIR"

REF_INFO="$(
    gh api "repos/$REPOSITORY/git/ref/tags/$TAG" \
        --jq '.object.type + " " + .object.sha'
)"
read -r REF_TYPE REF_SHA <<< "$REF_INFO"
TAG_DEPTH=0
while [[ "$REF_TYPE" == "tag" && "$TAG_DEPTH" -lt 8 ]]; do
    REF_INFO="$(
        gh api "repos/$REPOSITORY/git/tags/$REF_SHA" \
            --jq '.object.type + " " + .object.sha'
    )"
    read -r REF_TYPE REF_SHA <<< "$REF_INFO"
    TAG_DEPTH=$((TAG_DEPTH + 1))
done
if [[ "$REF_TYPE" != "commit" || "$REF_SHA" != "$EXPECTED_SHA" ]]; then
    echo "ERROR: release tag $TAG resolves to $REF_TYPE $REF_SHA, expected commit $EXPECTED_SHA" >&2
    exit 1
fi

IS_DRAFT="$(
    gh release view "$TAG" \
        --repo "$REPOSITORY" \
        --json isDraft \
        --jq .isDraft
)"

if [[ "$IS_DRAFT" == "false" ]]; then
    echo "ERROR: published release $TAG is immutable; create a new draft with a new tag and version" >&2
    exit 1
fi
if [[ "$IS_DRAFT" != "true" ]]; then
    echo "ERROR: unexpected release draft state: $IS_DRAFT" >&2
    exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

if upload_directory "$ARTIFACT_DIR"; then
    :
else
    UPLOAD_STATUS=$?
    echo "ERROR: draft release upload failed; the release remains unpublished" >&2
    exit "$UPLOAD_STATUS"
fi
if ! download_and_verify "$WORK_DIR/draft-verify"; then
    echo "ERROR: draft release inventory verification failed; the release remains unpublished" >&2
    exit 1
fi
gh release edit "$TAG" \
    --repo "$REPOSITORY" \
    --draft=false
