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
rooted at --self-pid and splits total CPU% between that tree (self) and
everything else (foreign). The report judges both foreign load and an
implied foreign load, max(0, 100 - idle - self), against the refusal
ceiling; self is otherwise recorded for context. --self-pid is ordinarily the impl
script's own PID, but under --gpu-handoff (lattice#1732) the admitted
benchmark executable is launched by the cooperative supervisor as a SIBLING of
the impl script, not a descendant of it, so bench-compare-impl.sh passes its
own PARENT PID instead in that case -- the supervisor that launched both.

PER-PROCESS CPU FIGURE, which differs by platform; every row names its
source in `cpu_source`.

Linux (`proc-stat-interval`): the same one-second window that yields the idle
share also brackets two reads of every `/proc/<pid>/stat`. A process's share
is its utime+stime delta over that window divided by the aggregate `/proc/stat`
delta, so self, foreign and idle are shares of one window in one unit, and
`100 - idle - self` (the report's implied figure) has nothing left in it that
belongs to the bench. The process totals include every thread. This is what a
benchmark that fans out needs: before it, `ps` pcpu was the source here too,
and on Linux that is the process's average over its whole life, so a few
seconds of a rayon matmul spread over 16 cores moved the bench's self figure
by about one point while the idle reading dropped to about a third. The
difference was booked as foreign load and the arm refused, at the same sample
of every inference_bench run on a 16-core host. A process that exits inside
the window loses that window's share (its time moves to the parent's cutime
only when reaped), and a process with no first read is counted only when its
start time falls inside the window, so a pid reused mid-window is not charged
its predecessor's history.

macOS (`ps-pcpu-decaying-avg`): `ps -Ao pid,ppid,pcpu,comm` once per tick.
pcpu there is a DECAYING AVERAGE over the process's recent lifetime, not an
interval-exact figure, so a short burst is under- or over-counted relative to
what it drew in this interval. An interval-exact reading (`top -l 2 -stats
pid,ppid,cpu`) is not implemented, and rows from this path carry the
decaying-average figure.

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


def parse_pid_stat(data: str) -> tuple[int, int, int, str]:
    """(ppid, utime+stime ticks, start ticks, comm) from one /proc/<pid>/stat.

    comm sits in parentheses and may itself hold spaces or parentheses, so the
    numeric fields are read after the LAST ')'. Field n (1-based, proc(5)) is
    rest[n - 3]: ppid is field 4, utime 14, stime 15, starttime 22.
    """
    open_paren = data.index("(")
    close_paren = data.rindex(")")
    comm = data[open_paren + 1:close_paren]
    rest = data[close_paren + 2:].split()
    return int(rest[1]), int(rest[11]) + int(rest[12]), int(rest[19]), comm


def proc_snapshot() -> dict[int, tuple[int, int, int, str]]:
    """pid -> parse_pid_stat(...) for every process this /proc lists."""
    rows: dict[int, tuple[int, int, int, str]] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(f"/proc/{entry.name}/stat") as fh:
                rows[int(entry.name)] = parse_pid_stat(fh.read())
        except (OSError, ValueError, IndexError):
            continue
    return rows


def interval_rows(
    line0: str,
    line1: str,
    snap0: dict[int, tuple[int, int, int, str]],
    snap1: dict[int, tuple[int, int, int, str]],
    window_start_ticks: int,
) -> tuple[float, dict[int, tuple[int, float, str]]]:
    """(idle %, pid -> (ppid, % of the machine over the window, comm)).

    window_start_ticks is the uptime, in clock ticks, at the first read: a pid
    absent from snap0, or present with a different start time, is a process
    born in the window and is charged its whole time only if it started at or
    after that instant; one that started earlier had no first read and is
    charged nothing rather than its lifetime.
    """
    total0, idle0 = parse_proc_stat(line0)
    total1, idle1 = parse_proc_stat(line1)
    dt = total1 - total0
    if dt <= 0:
        raise RuntimeError("/proc/stat did not advance")
    rows: dict[int, tuple[int, float, str]] = {}
    for pid, (ppid, cpu1, start1, comm) in snap1.items():
        before = snap0.get(pid)
        if before is not None and before[2] == start1:
            used = max(cpu1 - before[1], 0)
        elif start1 >= window_start_ticks:
            used = cpu1
        else:
            used = 0
        rows[pid] = (ppid, 100.0 * used / dt, comm)
    return 100.0 * (idle1 - idle0) / dt, rows


def _linux_interval_sample() -> tuple[float, dict[int, tuple[int, float, str]]]:
    def first_line() -> str:
        with open("/proc/stat") as fh:
            return fh.readline()

    with open("/proc/uptime") as fh:
        window_start = int(float(fh.read().split()[0]) * os.sysconf("SC_CLK_TCK"))
    line0 = first_line()
    snap0 = proc_snapshot()
    time.sleep(1.0)
    line1 = first_line()
    snap1 = proc_snapshot()
    return interval_rows(line0, line1, snap0, snap1, window_start)


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


def sample_once(self_pid: int, cores: int, linux: bool | None = None) -> dict:
    """One row. Shares are of the whole machine; top_foreign_pct is of one core."""
    if linux is None:
        linux = sys.platform.startswith("linux")
    if linux:
        idle, rows = _linux_interval_sample()
        scale, source = 1.0, "proc-stat-interval"
    else:
        idle, rows = idle_percent(), ps_snapshot()
        scale, source = 1.0 / cores, "ps-pcpu-decaying-avg"
    tree = self_tree(rows, self_pid)
    foreign_total = 0.0
    self_total = 0.0
    top_name, top_pct = "none", 0.0
    for pid, (_ppid, pct, comm) in rows.items():
        share = pct * scale
        if pid in tree:
            self_total += share
        else:
            foreign_total += share
            if share * cores > top_pct:
                top_name, top_pct = os.path.basename(comm), share * cores
    return {
        "idle_pct": idle,
        "foreign_pct": foreign_total,
        "self_pct": self_total,
        "top_foreign": top_name,
        "top_foreign_pct": top_pct,
        "cpu_source": source,
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
