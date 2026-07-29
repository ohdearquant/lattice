#!/usr/bin/env python3
"""Record power and thermal state without turning unavailable sensors into nominal."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime

SCHEMA = "lattice-machine-state-v1"


def unavailable(reason: str) -> dict[str, str]:
    return {"status": "unavailable", "reason": reason}


def parse_macos_power(output: str) -> dict[str, str]:
    """Parse `pmset -g batt` without guessing when its source is absent."""
    match = re.search(r"Now drawing from '([^']+)'", output)
    if match is None:
        return unavailable("pmset did not report a power source")
    source = match.group(1).strip()
    lowered = source.lower()
    if "ac power" in lowered:
        state = "ac"
    elif "battery power" in lowered:
        state = "battery"
    else:
        return unavailable(f"unrecognized pmset power source: {source}")
    return {"status": "measured", "source": "pmset", "state": state}


def parse_macos_thermal(output: str) -> dict[str, str | int]:
    """Parse `pmset -g therm`; diagnostic text is unavailable, never nominal."""
    lowered = output.lower()
    if "error:" in lowered or "failed to" in lowered:
        return unavailable("pmset thermal query returned an error")

    speed = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", output)
    if speed is not None:
        limit = int(speed.group(1))
        if not 0 <= limit <= 100:
            return unavailable("pmset reported an invalid CPU speed limit")
        return {
            "status": "measured",
            "source": "pmset",
            "state": "nominal" if limit == 100 else "throttled",
            "cpu_speed_limit_percent": limit,
        }

    no_thermal_warning = "no thermal warning level has been recorded" in lowered
    no_performance_warning = "no performance warning level has been recorded" in lowered
    if no_thermal_warning and no_performance_warning:
        return {"status": "measured", "source": "pmset", "state": "nominal"}
    return unavailable("pmset thermal output was not recognized")


def run_pmset(mode: str) -> tuple[int, str]:
    try:
        result = subprocess.run(
            ["pmset", "-g", mode],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return 1, str(error)
    return result.returncode, result.stdout + result.stderr


def collect_state(platform: str) -> tuple[dict[str, object], dict[str, object]]:
    if platform != "darwin":
        reason = f"unsupported platform: {platform}"
        return unavailable(reason), unavailable(reason)

    power_rc, power_output = run_pmset("batt")
    power = (
        parse_macos_power(power_output)
        if power_rc == 0
        else unavailable(f"pmset power query failed with exit {power_rc}")
    )
    thermal_rc, thermal_output = run_pmset("therm")
    thermal = (
        parse_macos_thermal(thermal_output)
        if thermal_rc == 0
        else unavailable(f"pmset thermal query failed with exit {thermal_rc}")
    )
    return power, thermal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    power, thermal = collect_state(sys.platform)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "label": args.label,
                "captured_at_utc": datetime.now(UTC)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "power": power,
                "thermal": thermal,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
