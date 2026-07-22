#!/usr/bin/env python3
"""Sample ambient CPU load and refuse to measure on a busy machine.

    quiet-probe.py --label "before base"          # exits 1 below the floor
    quiet-probe.py --label "after head" --floor 60

Prints one line naming the measured idle percentage, the floor it was judged
against, and the largest consumers at the moment of sampling.

WHY THIS IS A GATE AND NOT A WARNING. A lock excludes peers on this machine; it
says nothing about ambient load. Both are needed and they are not the same
check: a bench window has been held, uncontended, while the box drew a
double-digit percentage of its CPU from unrelated desktop work, and two A/B
runs against the SAME base binary disagreed by tens of percent under exactly
that condition. A warning printed on a bench report is read by nobody at the
moment it matters, which is weeks later when someone cites the number.

WHY THE SAMPLES GO INTO THE REPORT. A number that does not record the
conditions that produced it is indistinguishable from one produced under good
conditions, and the reader will assume the good case because they cannot
reconstruct it. The same rule that makes a truncated artifact disclose its own
truncation applies to a measurement taken on a machine that may not have been
quiet.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

DEFAULT_FLOOR = 70.0


# The parsing is separated from the sampling on purpose. A parser that can only
# be reached by reading the live machine can only be tested against whatever the
# machine happens to be doing, which is a test that passes on a correct parser
# and on several wrong ones -- including a regex that captures the busy field
# instead of the idle one, whose error is invisible on an idle box.


def parse_proc_stat(line: str) -> tuple[int, int]:
    """(total, idle) jiffies from a /proc/stat aggregate `cpu` line.

    iowait counts as idle: the CPU is available for the bench during it. That
    is a deliberate choice, pinned by test rather than left to be rediscovered.
    """
    fields = [int(x) for x in line.split()[1:]]
    # user nice system idle iowait irq softirq steal ...
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    return sum(fields), idle


def linux_idle_pct(line0: str, line1: str) -> float:
    """Idle share between two /proc/stat samples."""
    total0, idle0 = parse_proc_stat(line0)
    total1, idle1 = parse_proc_stat(line1)
    dt = total1 - total0
    if dt <= 0:
        raise RuntimeError("/proc/stat did not advance")
    return 100.0 * (idle1 - idle0) / dt


def parse_top_idle(out: str) -> float:
    """Idle percentage from top's LAST 'CPU usage' line.

    The last one, because `top -l 2` reports its first sample as an average
    since boot, which on a machine that has been idle for hours reads as quiet
    no matter what is running right now.

    The percentage immediately preceding the word `idle`, because the same line
    carries user and sys percentages first; anchoring on position rather than on
    the word reports the busiest field as if it were the idlest.
    """
    hits = re.findall(r"CPU usage:.*?([\d.]+)%\s+idle", out)
    if not hits:
        raise RuntimeError("could not parse 'CPU usage' from top")
    return float(hits[-1])


def _idle_linux() -> float:
    """Idle share over a short interval, from two /proc/stat reads."""

    def first_line() -> str:
        with open("/proc/stat") as fh:
            return fh.readline()

    line0 = first_line()
    time.sleep(1.0)
    return linux_idle_pct(line0, first_line())


def _idle_macos() -> float:
    """Idle share from top's second sample (the first is since boot)."""
    out = subprocess.run(
        ["top", "-l", "2", "-n", "0", "-s", "1"],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    return parse_top_idle(out)


def idle_percent() -> float:
    return _idle_linux() if sys.platform.startswith("linux") else _idle_macos()


def top_consumers(n: int = 4) -> str:
    if sys.platform.startswith("linux"):
        argv = ["ps", "-eo", "pcpu,comm", "--sort=-pcpu"]
    else:
        argv = ["ps", "-Ao", "pcpu,comm", "-r"]
    try:
        lines = subprocess.run(
            argv, capture_output=True, text=True, timeout=15
        ).stdout.splitlines()[1 : n + 1]
    except (subprocess.SubprocessError, OSError):
        return "unavailable"
    parts = []
    for line in lines:
        bits = line.split(None, 1)
        if len(bits) == 2:
            parts.append(f"{os.path.basename(bits[1])} {bits[0]}%")
    return ", ".join(parts) or "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument(
        "--floor",
        type=float,
        default=float(os.environ.get("BENCH_IDLE_FLOOR", DEFAULT_FLOOR)),
        help=f"minimum acceptable idle percentage (default {DEFAULT_FLOOR})",
    )
    args = ap.parse_args()

    try:
        idle = idle_percent()
    except Exception as exc:  # noqa: BLE001 - the reason is reported, not swallowed
        # Fail closed. An unreadable probe is not evidence of a quiet machine,
        # and treating it as one is how an absence-based guard passes on the
        # exact input it exists to reject.
        print(f"[quiet] {args.label}: PROBE FAILED ({exc}) - refusing to measure")
        return 1

    consumers = top_consumers()
    verdict = "ok" if idle >= args.floor else "BELOW FLOOR"
    print(
        f"[quiet] {args.label}: idle {idle:.1f}% (floor {args.floor:.1f}%) "
        f"{verdict} | top: {consumers}"
    )
    return 0 if idle >= args.floor else 1


if __name__ == "__main__":
    sys.exit(main())
