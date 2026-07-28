#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <release-tag> <owner/repo> [artifact-dir]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="$1"
REPOSITORY="$2"
ARTIFACT_DIR="${3:-$SCRIPT_DIR/../dist}"

if [[ "${GITHUB_EVENT_NAME:-}" != "release" ]] &&
    [[ "${GITHUB_EVENT_NAME:-}" != "workflow_dispatch" ]]; then
    echo "ERROR: macOS release assets may only upload from a release workflow" >&2
    exit 1
fi

VERSION="$("$SCRIPT_DIR/package-app.sh" --print-version)"
EXPECTED_TAG="v$VERSION"
if [[ "$TAG" != "$EXPECTED_TAG" ]]; then
    echo "ERROR: release tag $TAG does not match workspace version $EXPECTED_TAG" >&2
    exit 1
fi

UPLOAD_ARGS=()
for NAME in Lattice.dmg Lattice.zip; do
    ASSET="$ARTIFACT_DIR/$NAME"
    if [[ ! -s "$ASSET" ]]; then
        echo "ERROR: release asset is missing or empty: $ASSET" >&2
        exit 1
    fi
    CHECKSUM="$ASSET.sha256"
    shasum -a 256 "$ASSET" > "$CHECKSUM"
    UPLOAD_ARGS+=("$ASSET" "$CHECKSUM")
done

gh release upload "$TAG" \
    "${UPLOAD_ARGS[@]}" \
    --repo "$REPOSITORY" \
    --clobber
