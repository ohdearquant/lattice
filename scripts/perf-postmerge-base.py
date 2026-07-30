#!/usr/bin/env python3
"""Resolve post-merge A/B endpoints from the last successfully measured commit."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ZERO_SHA = "0" * 40
SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


class ResolutionError(RuntimeError):
    """The requested A/B range cannot be established safely."""


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise ResolutionError(f"git {' '.join(args)} failed: {detail}")
    return result


def resolve_commit(repo: Path, ref: str) -> str:
    result = git(repo, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
    sha = result.stdout.strip()
    if SHA_PATTERN.fullmatch(sha) is None:
        raise ResolutionError(f"{ref!r} did not resolve to one full commit SHA")
    return sha


def is_ancestor(repo: Path, base: str, head: str) -> bool:
    result = git(repo, "merge-base", "--is-ancestor", base, head, check=False)
    if result.returncode not in (0, 1):
        detail = result.stderr.strip() or "no diagnostic"
        raise ResolutionError(f"cannot test ancestry for {base} -> {head}: {detail}")
    return result.returncode == 0


def remote_record(repo: Path, branch: str) -> str | None:
    if git(repo, "check-ref-format", "--branch", branch, check=False).returncode != 0:
        raise ResolutionError(f"invalid progression branch name: {branch!r}")

    full_ref = f"refs/heads/{branch}"
    query = git(
        repo,
        "ls-remote",
        "--exit-code",
        "--heads",
        "origin",
        full_ref,
        check=False,
    )
    if query.returncode == 2:
        return None
    if query.returncode != 0:
        detail = query.stderr.strip() or "no diagnostic"
        raise ResolutionError(f"cannot query progression branch {branch!r}: {detail}")

    lines = query.stdout.splitlines()
    if len(lines) != 1:
        raise ResolutionError(
            f"progression branch {branch!r} returned {len(lines)} remote records"
        )
    fields = lines[0].split()
    if len(fields) != 2 or fields[1] != full_ref or SHA_PATTERN.fullmatch(fields[0]) is None:
        raise ResolutionError(f"progression branch {branch!r} returned a malformed record")

    tracking_ref = f"refs/remotes/origin/{branch}"
    git(repo, "fetch", "--no-tags", "origin", f"{full_ref}:{tracking_ref}")
    recorded = resolve_commit(repo, tracking_ref)
    if recorded != fields[0]:
        raise ResolutionError(
            f"progression branch {branch!r} changed while its record was fetched"
        )
    return recorded


def event_parent(repo: Path, head: str, before: str, suffix: str) -> tuple[str, str]:
    if before and before != ZERO_SHA:
        try:
            return resolve_commit(repo, before), f"event-parent-{suffix}"
        except ResolutionError:
            pass
    return resolve_commit(repo, f"{head}^"), f"head-parent-{suffix}"


def resolve(args: argparse.Namespace) -> dict[str, str]:
    repo = args.repo.resolve()
    head_ref = args.head_ref or "HEAD"
    head = resolve_commit(repo, head_ref)
    state_before = "not-queried"

    if args.base_ref:
        base = resolve_commit(repo, args.base_ref)
        source = "dispatch-input"
    elif args.event_name == "push":
        recorded = remote_record(repo, args.state_branch)
        state_before = recorded or "missing"
        if recorded is None:
            base, source = event_parent(repo, head, args.event_before, "fallback")
        elif recorded == head:
            base, source = event_parent(repo, head, args.event_before, "rerun")
        elif is_ancestor(repo, recorded, head):
            base = recorded
            source = "recorded"
        else:
            raise ResolutionError(
                f"recorded commit {recorded} is not an ancestor of head {head}"
            )
    else:
        base, source = event_parent(repo, head, args.event_before, "dispatch")

    if base == head:
        raise ResolutionError(f"base and head resolve to the same commit ({head})")
    if not is_ancestor(repo, base, head):
        raise ResolutionError(f"base {base} is not an ancestor of head {head}")

    span_result = git(repo, "rev-list", "--count", f"{base}..{head}")
    span = span_result.stdout.strip()
    if not span.isdigit() or int(span) < 1:
        raise ResolutionError(f"commit span for {base}..{head} is not positive: {span!r}")

    return {
        "base": base,
        "head": head,
        "base_source": source,
        "span_count": span,
        "state_branch": args.state_branch,
        "state_before": state_before,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--event-name", choices=("push", "workflow_dispatch"), required=True)
    parser.add_argument("--event-before", default="")
    parser.add_argument("--base-ref", default="")
    parser.add_argument("--head-ref", default="")
    parser.add_argument("--state-branch", required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()

    try:
        outputs = resolve(args)
        with args.github_output.open("a", encoding="utf-8") as output:
            for key, value in outputs.items():
                output.write(f"{key}={value}\n")
    except (OSError, ResolutionError) as error:
        print(f"perf-postmerge-base: {error}", file=sys.stderr)
        return 2

    print(
        "A/B endpoints: "
        f"{outputs['base']} -> {outputs['head']} "
        f"(source={outputs['base_source']}, commits={outputs['span_count']}, "
        f"state={outputs['state_before']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
