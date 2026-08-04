#!/usr/bin/env bash
# Resolve a Python interpreter that satisfies the bench harness's version
# floor (scripts/lib/bench_supervision.py, scripts/perf-bench-gate.py,
# scripts/lib/machine-state-probe.py: 3.11+), instead of trusting whichever
# `python3` happens to be first on PATH. macOS ships /usr/bin/python3 at
# 3.9, and PATH order putting it ahead of a newer interpreter is a normal,
# not exotic, caller environment.
#
# Prints the resolved interpreter's path on stdout and returns 0, or prints
# nothing and returns 1 if no candidate on PATH satisfies the floor.
bench_resolve_python3() {
    local candidate resolved
    for candidate in python3.13 python3.12 python3.11 python3; do
        resolved="$(command -v "$candidate" 2>/dev/null)" || continue
        if "$resolved" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
            printf '%s\n' "$resolved"
            return 0
        fi
    done
    return 1
}

# Resolves via bench_resolve_python3 or exits 1 with a message naming the
# requirement, what was found, and how to get a suitable interpreter.
# Usage: PYTHON_BIN="$(bench_require_python3 "$0")"
bench_require_python3() {
    local caller="${1:-$0}" resolved
    if resolved="$(bench_resolve_python3)"; then
        printf '%s\n' "$resolved"
        return 0
    fi
    {
        echo "$caller: no Python >= 3.11 found on PATH (tried python3.13, python3.12, python3.11, python3)."
        echo "$caller: found on PATH: $(command -v python3 2>/dev/null || echo 'none') ($(command -v python3 >/dev/null 2>&1 && "$(command -v python3)" --version 2>&1 || echo 'n/a'))"
        echo "$caller: install a newer interpreter, e.g. \`brew install python@3.12\`, and ensure it's reachable."
    } >&2
    return 1
}
