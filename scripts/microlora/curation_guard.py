#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Guards shared by both curation lanes: the sensitive-content screen and the output-path check.

One definition each, imported by the commit miner and the DSL generator, so the two lanes cannot
drift apart on what counts as a secret or where generated data may land.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


class GuardError(ValueError):
    """A fail-closed guard refusal."""


# An address with a dotted domain, or a bare `user@host` form (internal addresses have no TLD).
# Both forms require a local part carrying an alphanumeric, so `@decorator`, `import @scope/package`
# and `+@app.route` in a diff are not addresses.
EMAIL = re.compile(
    r"(?<![A-Z0-9.!#$%&'*+/=?^_`{|}~-])"
    r"(?=[.!#$%&'*+/=?^_`{|}~-]*[A-Z0-9])"
    r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9](?:[A-Z0-9.-]*[A-Z0-9])?\.[A-Z]{2,}"
    r"|(?<![A-Z0-9.!#$%&'*+/=?^_`{|}~-])[A-Z0-9][A-Z0-9._%+-]*@(?:localhost|[A-Z0-9][A-Z0-9-]*)(?![A-Z0-9.\-/])",
    re.IGNORECASE,
)

# Credential shapes. The assignment arm accepts quoted or bare token-like values but not
# expressions: `api_key = os.environ["API_KEY"]` reads a key, it does not carry one.
SECRET = re.compile(
    r"-----BEGIN (?:[A-Z ]*PRIVATE KEY|OPENSSH PRIVATE KEY)-----"
    r"|\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b"
    r"|\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"
    r"|\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}"
    r"|\bxox[abpr]-[A-Za-z0-9-]{10,}"
    r"|\bAIza[0-9A-Za-z_-]{35}"
    r"|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+"
    r"|[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@"
    r"|(?i:\b(?:password|passwd|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret"
    r"|secret[_-]?key|aws_secret_access_key|aws_session_token)\s*[:=]\s*[\"']?[A-Za-z0-9+/=_.-]{8,}[\"']?(?=[\s,;)}\]]|\Z))"
)


def sensitive_reason(text: str) -> str | None:
    if EMAIL.search(text):
        return "email"
    if SECRET.search(text):
        return "secret_pattern"
    if any(ord(c) < 32 and c not in "\n\t" for c in text) or "\x7f" in text:
        return "control_character"
    return None


def safe_output(path, allowed: frozenset[str] | set[str]):
    """Inside a repository, data must be ignored and entirely untracked.

    `allowed` names the files a lane writes; an existing output directory holding anything
    else is refused rather than replaced.
    """
    path = Path(path).resolve()
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise GuardError("Output ancestor is not a directory")
    probe = subprocess.run(
        ["git", "-C", str(ancestor), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0:
        root = Path(probe.stdout.strip()).resolve()
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", str(path)],
            capture_output=True,
            check=False,
        )
        ignored = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q", "--", str(path)],
            capture_output=True,
            check=False,
        )
        if tracked.returncode != 0 or tracked.stdout or ignored.returncode != 0:
            raise GuardError(
                "Refusing output in a tracked or non-ignored repository location"
            )
    elif probe.returncode != 128:
        raise GuardError("Could not establish output repository status")
    if path.exists() and not path.is_dir():
        raise GuardError("Output path is not a directory")
    if path.exists():
        unknown = {p.name for p in path.iterdir()} - set(allowed)
        if unknown:
            raise GuardError(
                "Refusing to replace an output directory containing unrelated files"
            )
    return path
