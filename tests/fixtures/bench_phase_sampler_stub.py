"""Shared STUB_PHASE_SAMPLER text for tests that drive bench-compare-impl.sh
end to end (lattice#1515 Amendment 2).

Loaded via importlib.util.spec_from_file_location (the same pattern these
test files already use for perf-bench-gate.py) rather than a package import,
so it works the same way from any test file regardless of how it is invoked
(`python3 tests/test_x.py` vs a test runner). Canonical source for
STUB_PHASE_SAMPLER; test_bench_locks.py and test_bench_compare_measurement.py
both load it from here instead of each keeping their own copy.

It must write `<out>.ready` right after installing its signal handlers:
scripts/lib/bench-compare-impl.sh's phase_sampler_start polls for that marker
before returning, and a stub that never creates it trips the "did not start"
refusal meant for a genuinely dead sampler process.
"""

STUB_PHASE_SAMPLER = """#!/usr/bin/env python3
import json
import pathlib
import signal
import sys
import time


def _handle(signum, frame):
    raise SystemExit(0)


signal.signal(signal.SIGTERM, _handle)
signal.signal(signal.SIGINT, _handle)

arm = sys.argv[sys.argv.index("--arm") + 1]
out = sys.argv[sys.argv.index("--out") + 1]
pathlib.Path(out + ".ready").touch()
with open(out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({
        "schema": "perf-phase-sample/v1",
        "arm": arm,
        "captured_utc": "2026-01-01T00:00:00Z",
        "idle_pct": 100.0,
        "foreign_pct": 0.0,
        "self_pct": 5.0,
        "top_foreign": "none",
        "top_foreign_pct": 0.0,
    }) + "\\n")
    fh.flush()
    while True:
        time.sleep(0.05)
"""
