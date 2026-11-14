#!/usr/bin/env python3
"""Wait a configured post-build interval before quiet admission and sampling."""

import argparse
import math
import os
import re
import sys
import time
from collections.abc import Mapping


def positive_seconds(raw: str) -> int:
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise ValueError("settling seconds must be a positive integer")
    seconds = int(raw)
    if seconds <= 0 or not math.isfinite(float(seconds)):
        raise ValueError("settling seconds must be a finite positive integer")
    return seconds


def resolve_seconds(interval: str, environment: Mapping[str, str]) -> int:
    raw = (environment["BENCH_SETTLE_SECONDS"]
           if "BENCH_SETTLE_SECONDS" in environment else interval)
    return positive_seconds(raw)


def wait_seconds(seconds: int) -> None:
    deadline = time.monotonic() + seconds
    if not math.isfinite(deadline):
        raise ValueError("settling deadline is not finite")
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        try:
            time.sleep(remaining)
        except InterruptedError:
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--resolve", action="store_true")
    mode.add_argument("--seconds")
    parser.add_argument("--interval", default="5")
    parser.add_argument("--arm", choices=("base1", "head1", "head2", "base2"))
    args = parser.parse_args()
    try:
        if args.resolve:
            print(resolve_seconds(args.interval, os.environ))
        else:
            if args.arm is None:
                parser.error("--seconds requires --arm")
            wait_seconds(positive_seconds(args.seconds))
    except (ValueError, OverflowError, OSError) as error:
        print(f"bench-settle: {error}; refusing to measure", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
