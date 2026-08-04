#!/usr/bin/env python3
"""Record power, thermal, and HID-idle state without guessing unavailable sensors."""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import subprocess
import sys

_RUNNING_PYTHON = ".".join(str(part) for part in sys.version_info[:3])
_PYTHON_REQUIREMENT_ERROR = (
    f"{sys.argv[0]} requires Python 3.11 or newer; "
    f"running Python {_RUNNING_PYTHON} at {sys.executable}"
)
try:
    from datetime import UTC, datetime
except ImportError:
    raise SystemExit(_PYTHON_REQUIREMENT_ERROR) from None
if sys.version_info[:2] < (3, 11):
    raise SystemExit(_PYTHON_REQUIREMENT_ERROR)

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


def thermal_state_from_raw(raw_state: int) -> dict[str, str]:
    """Normalize Foundation's ProcessInfo thermal-state enum."""
    states = {
        0: "nominal",
        1: "fair",
        2: "serious",
        3: "critical",
    }
    if raw_state not in states:
        return unavailable(f"unknown ProcessInfo thermal state: {raw_state}")
    return {
        "status": "measured",
        "source": "ProcessInfo.thermalState",
        "state": states[raw_state],
    }


def read_process_info_thermal() -> dict[str, str]:
    """Read the supported macOS thermal-pressure API without PyObjC."""
    if sys.platform != "darwin":
        return unavailable("ProcessInfo.thermalState is only available on macOS")
    try:
        ctypes.CDLL(
            "/System/Library/Frameworks/Foundation.framework/Foundation"
        )
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")

        objc_get_class = objc.objc_getClass
        objc_get_class.argtypes = [ctypes.c_char_p]
        objc_get_class.restype = ctypes.c_void_p

        sel_register_name = objc.sel_registerName
        sel_register_name.argtypes = [ctypes.c_char_p]
        sel_register_name.restype = ctypes.c_void_p

        objc_msg_send = objc.objc_msgSend
        objc_msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        objc_msg_send.restype = ctypes.c_void_p

        process_info = objc_msg_send(
            objc_get_class(b"NSProcessInfo"),
            sel_register_name(b"processInfo"),
        )
        if not process_info:
            return unavailable("NSProcessInfo.processInfo returned nil")

        objc_msg_send.restype = ctypes.c_long
        raw_state = int(
            objc_msg_send(process_info, sel_register_name(b"thermalState"))
        )
        return thermal_state_from_raw(raw_state)
    except (AttributeError, OSError, TypeError, ValueError) as error:
        return unavailable(f"ProcessInfo.thermalState failed: {error}")


def parse_macos_idle(output: str) -> dict[str, str | float]:
    """Parse IOHIDSystem idle nanoseconds without treating absence as idle."""
    match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', output)
    if match is None:
        return unavailable("ioreg did not report HIDIdleTime")
    return {
        "status": "measured",
        "source": "IOHIDSystem.HIDIdleTime",
        "seconds": int(match.group(1)) / 1_000_000_000,
    }


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


def run_ioreg() -> tuple[int, str]:
    try:
        result = subprocess.run(
            ["ioreg", "-c", "IOHIDSystem"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return 1, str(error)
    return result.returncode, result.stdout + result.stderr


def read_macos_power() -> dict[str, object]:
    power_rc, power_output = run_pmset("batt")
    if power_rc != 0:
        return unavailable(f"pmset power query failed with exit {power_rc}")
    return parse_macos_power(power_output)


def read_macos_thermal() -> dict[str, object]:
    thermal_rc, thermal_output = run_pmset("therm")
    if thermal_rc == 0:
        pmset_state = parse_macos_thermal(thermal_output)
    else:
        pmset_state = unavailable(
            f"pmset thermal query failed with exit {thermal_rc}"
        )
    if pmset_state["status"] == "measured":
        return pmset_state

    process_info = read_process_info_thermal()
    if process_info["status"] == "measured":
        process_info["fallback_reason"] = pmset_state["reason"]
        return process_info
    return unavailable(
        f"{pmset_state['reason']}; {process_info['reason']}"
    )


def read_macos_idle() -> dict[str, object]:
    idle_rc, idle_output = run_ioreg()
    if idle_rc != 0:
        return unavailable(f"ioreg idle query failed with exit {idle_rc}")
    return parse_macos_idle(idle_output)


def collect_state(
    platform: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if platform != "darwin":
        reason = f"unsupported platform: {platform}"
        return unavailable(reason), unavailable(reason), unavailable(reason)

    return read_macos_power(), read_macos_thermal(), read_macos_idle()


def collect_record(label: str, platform: str) -> dict[str, object]:
    power, thermal, idle = collect_state(platform)
    return {
        "schema": SCHEMA,
        "label": label,
        "captured_at_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "power": power,
        "thermal": thermal,
        "idle": idle,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    print(
        json.dumps(
            collect_record(args.label, sys.platform),
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
