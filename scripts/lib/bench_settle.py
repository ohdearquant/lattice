#!/usr/bin/env python3
"""Wait a configured post-build interval before quiet admission and sampling."""

import argparse
import math
import re
import sys
import time


def positive_seconds(raw: str) -> int:
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise ValueError("settling seconds must be a positive integer")
    seconds = int(raw)
    if seconds <= 0 or not math.isfinite(float(seconds)):
        raise ValueError("settling seconds must be a finite positive integer")
    return seconds


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
    parser.add_argument("--seconds", required=True)
    parser.add_argument("--arm", required=True, choices=("base1", "head1", "head2", "base2"))
    args = parser.parse_args()
    try:
        wait_seconds(positive_seconds(args.seconds))
    except (ValueError, OverflowError, OSError) as error:
        print(f"bench-settle: {error}; refusing to measure", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
