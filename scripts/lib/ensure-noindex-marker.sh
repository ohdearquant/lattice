#!/usr/bin/env bash
# ensure-noindex-marker.sh — install the Spotlight exclusion marker in a directory.
#
#   scripts/lib/ensure-noindex-marker.sh <dir>
#
# Creates <dir>/.metadata_never_index so macOS does not index the directory.
# bench-compare.sh calls this before creating its base/head worktrees: indexing
# full repository checkouts produces filesystem churn that lands asymmetrically
# across the base and head measurement phases, and a base-then-head A/B turns
# that asymmetry into an apparent code delta.
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
  echo "[noindex] Without it Spotlight indexes this directory. For bench worktrees that" >&2
  echo "[noindex] churn lands asymmetrically across the base/head phases and reads as a" >&2
  echo "[noindex] code delta. Refusing to continue rather than measure unprotected." >&2
  exit 1
fi

# Post-condition: prove the marker is really there before reporting success.
[ -f "$MARKER" ] || { echo "[noindex] FATAL: $MARKER missing after creation" >&2; exit 1; }
