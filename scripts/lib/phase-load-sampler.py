#!/usr/bin/env python3
"""Sample ambient load DURING one measured bench-compare arm (lattice#1515).

    phase-load-sampler.py --arm head1 --self-pid 4821 \
        --out .cache/bench-compare-criterion/phase-load/head1.jsonl --interval 5

Runs in the foreground until SIGTERM/SIGINT, appending one JSON line per
sample to --out. bench-compare-impl.sh starts this as a background process
immediately before an arm's first `run_bench` call and stops it (SIGTERM,
then wait) immediately after the arm's last one.

WHY THIS EXISTS. scripts/lib/quiet-probe.py samples at three BOUNDARIES:
before the first arm, between the two measurement-order strata, and after the
last arm. A load that starts and ends between two boundaries is invisible to
all three by construction -- lattice#1515 has two measured instances, a 23s
compile and a 42s burst that ended 7s before a boundary sample, neither of
which any boundary probe saw. This sampler closes that gap by reading load
throughout the arm, not just at its edges.

SELF VS FOREIGN. The bench itself is load: one core busy on a 2-core CI
runner is 50% of `100 - idle`, so a raw idle floor applied mid-arm would
refuse every honest run. Each sample therefore also walks the process tree
rooted at the impl script's own PID (passed as --self-pid) and splits total
CPU% between that tree (self) and everything else (foreign). Only foreign
load is judged against the refusal ceiling; self is recorded for context.

PER-PROCESS CPU FIGURE. This samples `ps -Ao pid,ppid,pcpu,comm` (Darwin) /
`ps -eo pid,ppid,pcpu,comm` (Linux) once per cadence tick. `ps`'s pcpu field
is a DECAYING AVERAGE over the process's recent lifetime, not an interval-
exact figure for the sampling window -- unlike quiet-probe.py's idle-percent,
which is computed from two point-in-time reads bracketing a short sleep. A
short-lived foreign process that spikes and exits between two sampler ticks
can therefore be under- or over-counted relative to what it actually drew
during this exact interval. This is the FALLBACK figure the constraint
allows: an interval-exact per-process reading (`top -l 2 -stats
pid,ppid,cpu` on macOS, two `/proc/<pid>/stat` reads on Linux) would be more
precise but is not implemented here. The artifact this script produces
carries the decaying-average `ps` figure described above, not an
interval-exact one.

PARSER SHARING. Idle-percent parsing (parse_top_idle / parse_proc_stat /
linux_idle_pct below) is a PINNED COPY of quiet-probe.py's own parsers, not
an `importlib` import of that module: quiet-probe.py is invoked only as a
script elsewhere in this repo, and several existing test fixtures stub it
with module-level (unguarded) argparse calls that run any time the file is
exec'd -- importing it here would execute those stubs' CLI parsing against
THIS process's argv the moment this module loaded, in every test that
substitutes its own quiet-probe.py without a callers-only prompt. Copying is
the documented fallback (see CLAUDE.md's bench harness section): the PARSER
SHARING test in tests/test_bench_compare_measurement.py pins that this copy
and quiet-probe.py's original agree on the same top(1)/`/proc/stat`
transcript, so the two cannot silently drift apart.

READINESS AND GUARANTEED FIRST/LAST SAMPLE (lattice#1515 Amendment 2). On
Linux CI the caller's arm subshell can finish (killing this process via its
EXIT trap) before the interpreter has reached `signal.signal(...)` -- the
default SIGTERM action then terminates the process silently with zero
samples written, which reads identically to a dead instrument. To close
that: the SIGTERM/SIGINT handlers are installed as the very first action in
`main()`, before argparse even runs, and once the output file is open this
process touches `<out>.ready` so the caller (`phase_sampler_start` in
bench-compare-impl.sh) can poll for that file and know a signal will now be
handled rather than kill the process outright. The sampling loop then takes
its first sample unconditionally (do-while shape, not `while not stop:`
first), so even an arm shorter than one cadence interval yields at least one
row, and takes one closing sample on stop unless the previous sample
completed less than half an interval ago (avoiding a near-duplicate at the
very end of an ordinary-length arm).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_proc_stat(line: str) -> tuple[int, int]:
    """(total, idle) jiffies from a /proc/stat aggregate `cpu` line.

    Pinned copy of quiet-probe.py's parse_proc_stat -- see PARSER SHARING
    above for why this is a copy, not an import.
    """
    fields = [int(x) for x in line.split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    return sum(fields), idle


def linux_idle_pct(line0: str, line1: str) -> float:
    """Idle share between two /proc/stat samples. Pinned copy, see above."""
    total0, idle0 = parse_proc_stat(line0)
    total1, idle1 = parse_proc_stat(line1)
    dt = total1 - total0
    if dt <= 0:
        raise RuntimeError("/proc/stat did not advance")
    return 100.0 * (idle1 - idle0) / dt


def parse_top_idle(out: str) -> float:
    """Idle percentage from top's LAST 'CPU usage' line.

    Pinned copy of quiet-probe.py's parse_top_idle -- see PARSER SHARING
    above for why this is a copy, not an import.
    """
    import re

    hits = re.findall(r"CPU usage:.*?([\d.]+)%\s+idle", out)
    if not hits:
        raise RuntimeError("could not parse 'CPU usage' from top")
    return float(hits[-1])


def _idle_linux() -> float:
    def first_line() -> str:
        with open("/proc/stat") as fh:
            return fh.readline()

    line0 = first_line()
    time.sleep(1.0)
    return linux_idle_pct(line0, first_line())


def _idle_macos() -> float:
    out = subprocess.run(
        ["top", "-l", "2", "-n", "0", "-s", "1"],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    return parse_top_idle(out)


def idle_percent() -> float:
    return _idle_linux() if sys.platform.startswith("linux") else _idle_macos()


def logical_cpu_count() -> int:
    count = os.cpu_count()
    if not count or count < 1:
        raise RuntimeError("could not determine logical CPU count")
    return count


def ps_snapshot() -> dict[int, tuple[int, float, str]]:
    """pid -> (ppid, pcpu, comm) from one `ps` read."""
    if sys.platform.startswith("linux"):
        argv = ["ps", "-eo", "pid,ppid,pcpu,comm"]
    else:
        argv = ["ps", "-Ao", "pid,ppid,pcpu,comm"]
    out = subprocess.run(
        argv, capture_output=True, text=True, timeout=15, check=True
    ).stdout
    rows: dict[int, tuple[int, float, str]] = {}
    for line in out.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid, ppid, pcpu = int(parts[0]), int(parts[1]), float(parts[2])
        except ValueError:
            continue
        rows[pid] = (ppid, pcpu, parts[3])
    return rows


def self_tree(rows: dict[int, tuple[int, float, str]], root_pid: int) -> set[int]:
    """Every pid descending from root_pid (inclusive), by repeated ppid closure."""
    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _pcpu, _comm) in rows.items():
            if pid in tree or ppid not in tree:
                continue
            tree.add(pid)
            changed = True
    return tree


def sample_once(self_pid: int, cores: int) -> dict:
    idle = idle_percent()
    rows = ps_snapshot()
    tree = self_tree(rows, self_pid)
    foreign_total = 0.0
    self_total = 0.0
    top_name, top_pct = "none", 0.0
    for pid, (_ppid, pcpu, comm) in rows.items():
        if pid in tree:
            self_total += pcpu
        else:
            foreign_total += pcpu
            if pcpu > top_pct:
                top_name, top_pct = os.path.basename(comm), pcpu
    return {
        "idle_pct": idle,
        "foreign_pct": foreign_total / cores,
        "self_pct": self_total / cores,
        "top_foreign": top_name,
        "top_foreign_pct": top_pct,
    }


def main() -> int:
    # Installed FIRST, before argparse: a SIGTERM landing before this line
    # runs would otherwise terminate the process via the default action
    # (lattice#1515 Amendment 2) with zero samples written -- indistinguishable
    # from a dead instrument to the caller.
    stop = False

    def _handle(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--self-pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    cores = logical_cpu_count()

    def write_sample(handle) -> float | None:
        """Take and write one sample; return its wall-clock time, or None on failure."""
        try:
            record = sample_once(args.self_pid, cores)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
            sys.stderr.write(f"[phase-load] {args.arm}: sample failed: {exc}\n")
            return None
        record.update(
            {
                "schema": "perf-phase-sample/v1",
                "arm": args.arm,
                "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        handle.flush()
        return time.monotonic()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        # The readiness marker is created only once the output file is open and
        # the handlers above are installed, so the caller polling for it knows
        # a SIGTERM from this point on will be handled, not silently kill us.
        ready_path = out_path.with_name(out_path.name + ".ready")
        ready_path.touch()

        # Do-while shape: every arm, however short, gets >=1 sample -- `while
        # not stop:` first would yield zero samples for an arm that finishes
        # (and signals us) before the loop body ever runs.
        last_sample_at = write_sample(handle)
        while not stop:
            # Sleep in short slices so SIGTERM lands within ~0.1s rather than
            # waiting out a full multi-second cadence.
            ticks = max(int(args.interval * 10), 1)
            for _ in range(ticks):
                if stop:
                    break
                time.sleep(0.1)
            if stop:
                break
            last_sample_at = write_sample(handle)
        # Closing-edge sample: skip only if the loop's own last sample is
        # still fresh (within half an interval), so an ordinary-length arm
        # doesn't get a near-duplicate final row.
        if last_sample_at is None or (
            time.monotonic() - last_sample_at >= args.interval / 2
        ):
            write_sample(handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
