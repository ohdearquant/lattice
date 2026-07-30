#!/usr/bin/env python3
"""Return a checkout-local benchmark host identifier without exposing a hostname."""

from __future__ import annotations

import os
import re
import secrets
import sys
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}")


def read_token(path: Path) -> str:
    token = path.read_text(encoding="utf-8").strip()
    if TOKEN_PATTERN.fullmatch(token) is None:
        raise ValueError(f"{path} does not contain one 32-hex host identifier")
    return token


def load_or_create(path: Path) -> str:
    try:
        return read_token(path)
    except FileNotFoundError:
        pass

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return read_token(path)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(token + "\n")
    return token


def main() -> int:
    configured_path = os.environ.get("LATTICE_BENCH_HOST_ID_FILE")
    path = (
        Path(configured_path).expanduser()
        if configured_path
        else Path(__file__).resolve().parents[2] / ".cache" / "bench-host-id"
    )
    try:
        token = load_or_create(path)
    except (OSError, ValueError) as error:
        print(f"bench-host-id: {error}", file=sys.stderr)
        return 2
    print(f"local-random:{token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
