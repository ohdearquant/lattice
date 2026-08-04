#!/usr/bin/env bash
# ensure-noindex-marker.sh — install the Spotlight exclusion marker in a directory.
#
#   scripts/lib/ensure-noindex-marker.sh <dir>
#
# Creates <dir>/.metadata_never_index so macOS does not index the directory.
# Benchmark entry points call this before creating or building their measurement
# trees. Indexing the resulting filesystem churn can overlap a timing phase, and
# an asymmetric overlap can become an apparent code delta (or make the
# order-balanced run honestly refuse as not measurable).
#
# FAIL-CLOSED BY DESIGN. This guards measurement integrity, so it must never
# report success without the marker actually being in place. A silently absent
# marker is worse than no protection at all, because the A/B still runs and its
# numbers still look trustworthy. Every failure path exits 2: this is an
# input/instrumentation-error failure (it fires before any worktree or
# benchmark exists), never a regression verdict. Exit 1 in the caller chain
# (bench-compare-impl.sh -> bench-locks.py -> perf-postmerge-gate.yml) is
# reserved for the gate's own confirmed-regression status; a caller running
# this guard under `set -e` propagates our exit code verbatim, so 1 here would
# be misread downstream as "confirmed regression" rather than "could not set
# up measurement protection".
set -euo pipefail

if [ "$#" -lt 1 ] || [ -z "${1:-}" ]; then
  echo "[noindex] FATAL: usage: ensure-noindex-marker.sh <dir>" >&2 || :
  exit 2
fi
DIR="$1"
MARKER="$DIR/.metadata_never_index"

if ! mkdir -p "$DIR" 2>/dev/null; then
  echo "[noindex] FATAL: cannot create $DIR (unwritable parent, or a" >&2 || :
  echo "[noindex] non-directory already occupies that path). Refusing to" >&2 || :
  echo "[noindex] continue rather than measure unprotected." >&2 || :
  exit 2
fi

# A symlink (or any non-regular entry) here defeats the protection silently:
# `test -e` is FALSE for a dangling link, so an existence check falls through to
# a redirect, and the redirect then follows the link to a target that may not be
# creatable. Replace anything that is not a plain file.
if [ -L "$MARKER" ] || { [ -e "$MARKER" ] && [ ! -f "$MARKER" ]; }; then
  # `rm -f` on a non-empty directory fails (and, un-normalized, would exit 1
  # under set -e below -- the same code the gate uses for a confirmed
  # regression). Convert it to our own exit 2 explicitly rather than let a
  # coreutils exit status leak through the contract.
  if ! rm -f "$MARKER" 2>/dev/null; then
    echo "[noindex] FATAL: $MARKER exists and is not a plain file (and could" >&2 || :
    echo "[noindex] not be removed, e.g. a non-empty directory). Refusing to" >&2 || :
    echo "[noindex] continue rather than measure unprotected." >&2 || :
    exit 2
  fi
fi

# Idempotent: an existing regular marker is left untouched, not re-truncated.
if [ ! -f "$MARKER" ] && ! : > "$MARKER"; then
  echo "[noindex] FATAL: cannot create $MARKER" >&2 || :
  echo "[noindex] Without it Spotlight indexes this directory. Its build churn can" >&2 || :
  echo "[noindex] land asymmetrically across timing phases and read as a" >&2 || :
  echo "[noindex] code delta. Refusing to continue rather than measure unprotected." >&2 || :
  exit 2
fi

# Post-condition: prove the marker is really there before reporting success.
[ -f "$MARKER" ] || { echo "[noindex] FATAL: $MARKER missing after creation" >&2 || :; exit 2; }
