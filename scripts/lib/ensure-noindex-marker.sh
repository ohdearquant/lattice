#!/usr/bin/env bash
# ensure-noindex-marker.sh — install the Spotlight exclusion marker in a directory.
#
#   scripts/lib/ensure-noindex-marker.sh <dir>
#
# Creates <dir>/.metadata_never_index so macOS does not index the directory.
# Benchmark entry points call this before creating or building their measurement
# trees. Indexing the resulting filesystem churn can overlap a timing phase, and
# a base-then-head A/B turns asymmetric overlap into an apparent code delta.
#
# FAIL-CLOSED BY DESIGN. This guards measurement integrity, so it must never
# report success without the marker actually being in place. A silently absent
# marker is worse than no protection at all, because the A/B still runs and its
# numbers still look trustworthy. Every failure path exits non-zero.
set -euo pipefail

DIR="${1:?usage: ensure-noindex-marker.sh <dir>}"
MARKER="$DIR/.metadata_never_index"

mkdir -p "$DIR"

# A symlink (or any non-regular entry) here defeats the protection silently:
# `test -e` is FALSE for a dangling link, so an existence check falls through to
# a redirect, and the redirect then follows the link to a target that may not be
# creatable. Replace anything that is not a plain file.
if [ -L "$MARKER" ] || { [ -e "$MARKER" ] && [ ! -f "$MARKER" ]; }; then
  rm -f "$MARKER"
fi

# Idempotent: an existing regular marker is left untouched, not re-truncated.
if [ ! -f "$MARKER" ] && ! : > "$MARKER"; then
  echo "[noindex] FATAL: cannot create $MARKER" >&2
  echo "[noindex] Without it Spotlight indexes this directory. Its build churn can" >&2
  echo "[noindex] land asymmetrically across timing phases and read as a" >&2
  echo "[noindex] code delta. Refusing to continue rather than measure unprotected." >&2
  exit 1
fi

# Post-condition: prove the marker is really there before reporting success.
[ -f "$MARKER" ] || { echo "[noindex] FATAL: $MARKER missing after creation" >&2; exit 1; }
