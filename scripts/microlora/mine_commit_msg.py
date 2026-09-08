#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Mine bounded, time-split commit-message pairs from approved public Git histories."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

DEFAULT_REPOS = (
    Path.home() / "projects/khive/lattice",
    Path.home() / "projects/khive/khive-oss",
    Path.home() / "projects/lionagi",
)
PUBLIC_REPOS = {"lattice", "khive", "lionagi"}
SPLITS = ("train", "valid", "test")
TRAIN_END = datetime(2026, 7, 15, tzinfo=UTC)
VALID_END = datetime(2026, 8, 15, tzinfo=UTC)
PROMPT_CAP = 1200
COMPLETION_CAP = 300
SUBJECT = re.compile(
    r"^(feat|fix|perf|docs|refactor|test|chore|ci|bench|build)(\([^)]*\))?!?: .{8,}$"
)
TRAILER = re.compile(r"^(Co-Authored-By|Claude-Session|Signed-off-by):", re.IGNORECASE)
EMAIL = re.compile(
    r"(?<![A-Z0-9.!#$%&'*+/=?^_`{|}~-])"
    r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Z0-9](?:[A-Z0-9.-]*[A-Z0-9])?\.[A-Z]{2,}",
    re.IGNORECASE,
)
SECRET = re.compile(
    r"-----BEGIN (?:[A-Z ]*PRIVATE KEY|OPENSSH PRIVATE KEY)-----"
    r"|\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b"
    r"|\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"
    r"|(?i:\b(?:password|api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*[\"'][^\"'\s]{8,}[\"'])"
)
SAFE_PATH = re.compile(r"[A-Za-z0-9_./@+~-]+")
DIFF_HEADER = re.compile(r"diff --git a/([A-Za-z0-9_./@+~-]+) b/([A-Za-z0-9_./@+~-]+)")
BLOCKED_DIRS = {
    "target",
    "node_modules",
    "vendor",
    "vendored",
    "generated",
    "dist",
    "build",
    ".khive",
    ".claude",
    ".codex",
    ".git",
    "__pycache__",
}
BLOCKED_FILES = (
    "*.lock",
    "package-lock.json",
    "npm-shrinkwrap.json",
    "pnpm-lock.yaml",
    "bun.lockb",
    "*.svg",
    "*.png",
    "*.jsonl",
    "*.min.*",
    ".env*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
)


class CurationError(Exception):
    """Input or output could not be safely curated."""


class NonUtf8Error(CurationError):
    """A Git record cannot be represented as a UTF-8 training example."""


@dataclass(frozen=True)
class Source:
    path: Path
    name: str
    sha: str
    origin: str


@dataclass(frozen=True)
class Candidate:
    repo: str
    sha: str
    authored: datetime
    split: str
    prompt: str
    completion: str
    diff: str


def git(repo: Path, *args: str) -> str:
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_ATTR_NOSYSTEM="1",
        GIT_OPTIONAL_LOCKS="0",
    )
    result = subprocess.run(
        [
            "git",
            "--no-pager",
            "--no-replace-objects",
            "-c",
            "core.quotePath=true",
            "-c",
            f"core.attributesFile={os.devnull}",
            "-C",
            str(repo),
            *args,
        ],
        capture_output=True,
        check=False,
        env=env,
    )
    if result.returncode:
        # Git errors can quote credentials embedded in a remote URL or file content.
        raise CurationError(
            f"git {args[0]} failed in {repo} (exit {result.returncode})"
        )
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NonUtf8Error(f"non-UTF-8 Git output from {repo}") from exc


def pin_source(path: Path) -> Source:
    path = path.resolve(strict=True)
    origin = git(path, "config", "--get", "remote.origin.url").strip()
    match = re.fullmatch(
        r"(?:https://github\.com/|git@github\.com:)ohdearquant/([A-Za-z0-9_-]+?)(?:\.git)?",
        origin,
    )
    if not match or match[1] not in PUBLIC_REPOS:
        raise CurationError(
            f"{path}: origin is not one of the three approved public repositories"
        )
    if git(path, "rev-parse", "--is-shallow-repository").strip() != "false":
        raise CurationError(
            f"{path}: shallow history cannot establish the requested time splits"
        )
    sha = git(path, "rev-parse", "--verify", "refs/heads/main^{commit}").strip()
    return Source(path, match[1], sha, origin)


def apply_main_pins(sources: list[Source], entries: list[str]) -> list[Source]:
    by_name = {source.name: source for source in sources}
    pins = {}
    for entry in entries:
        match = re.fullmatch(r"([a-z0-9_-]+)=([0-9a-f]{40}|[0-9a-f]{64})", entry)
        if not match or match[1] not in by_name or match[1] in pins:
            raise CurationError(
                "main pins require distinct selected repository names and full lowercase commit hashes"
            )
        pins[match[1]] = match[2]
    pinned = []
    for source in sources:
        if source.name not in pins:
            pinned.append(source)
            continue
        sha = pins[source.name]
        if (
            git(source.path, "rev-parse", "--verify", f"{sha}^{{commit}}").strip()
            != sha
        ):
            raise CurationError(
                f"{source.name}: pin does not identify a full commit hash"
            )
        try:
            git(source.path, "merge-base", "--is-ancestor", sha, source.sha)
        except CurationError as exc:
            raise CurationError(
                f"{source.name}: requested pin is not a verified ancestor of local main"
            ) from exc
        pinned.append(Source(source.path, source.name, sha, source.origin))
    return pinned


def time_split(authored: datetime) -> str:
    if authored.tzinfo is None:
        raise CurationError("author date has no timezone")
    return (
        "train" if authored < TRAIN_END else "valid" if authored < VALID_END else "test"
    )


def sensitive_reason(text: str) -> str | None:
    if EMAIL.search(text):
        return "email"
    if SECRET.search(text):
        return "secret_pattern"
    if any(ord(c) < 32 and c not in "\n\t" for c in text) or "\x7f" in text:
        return "control_character"
    return None


def excluded_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (
        not SAFE_PATH.fullmatch(path)
        or path.startswith("/")
        or any(p in {".", ".."} for p in path.split("/"))
        or bool(BLOCKED_DIRS.intersection(parts))
        or any(fnmatch.fnmatch(parts[-1].lower(), pattern) for pattern in BLOCKED_FILES)
    )


def clean_completion(subject: str, body: str) -> str | None:
    subject = re.sub(r"\s*\(#\d+\)", "", subject).strip()
    if not SUBJECT.fullmatch(subject):
        return None
    kept = []
    in_trailer = False
    for line in body.splitlines():
        if TRAILER.match(line):
            in_trailer = True
        elif in_trailer and line[:1].isspace():
            continue
        else:
            in_trailer = False
            kept.append(line)
    body = "\n".join(kept).strip()
    completion = " " + subject
    if body and len(body) < 200:
        completion += "\n\n" + body
    # Omitting a body to satisfy the combined cap is explicit in CURATION.md.
    if len(completion) > COMPLETION_CAP:
        completion = " " + subject
    return completion if len(completion) <= COMPLETION_CAP else None


def reduce_diff(diff: str, counts: Counter) -> list[str]:
    blocks: list[list[str]] = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            blocks.append([line])
        elif blocks:
            blocks[-1].append(line)
        elif line:
            raise CurationError("patch contained text before its first diff header")
    reduced = []
    for block in blocks:
        match = DIFF_HEADER.fullmatch(block[0])
        if not match:
            counts["files_unsafe_header"] += 1
            continue
        if excluded_path(match[1]) or excluded_path(match[2]):
            counts["files_excluded_path"] += 1
            continue
        if any(
            line.startswith(("Binary files ", "GIT binary patch")) for line in block
        ):
            counts["files_binary"] += 1
            continue
        lines = [
            line for line in block if line.startswith(("diff --git ", "@@", "+", "-"))
        ]
        if not any(
            line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
            for line in lines
        ):
            counts["files_no_text_change"] += 1
            continue
        reduced.extend(lines)
    return reduced


def truncate_lines(lines: list[str], cap: int) -> str | None:
    lengths = [0]
    first_change = None
    for index, line in enumerate(lines):
        lengths.append(lengths[-1] + len(line) + (1 if index else 0))
        if (
            first_change is None
            and line.startswith(("+", "-"))
            and not line.startswith(("+++", "---"))
        ):
            first_change = index
    if first_change is None:
        return None
    for size in range(len(lines), 0, -1):
        omitted = len(lines) - size
        marker = f"\n[... {omitted} more lines]" if omitted else ""
        if lengths[size] + len(marker) <= cap and first_change < size:
            return "\n".join(lines[:size]) + marker
    return None


def mine(source: Source, counts: Counter) -> list[Candidate]:
    objects = Path(git(source.path, "rev-parse", "--git-path", "objects").strip())
    if not objects.is_absolute():
        objects = source.path / objects
    objects = objects.resolve(strict=True)
    if "\n" in str(objects):
        raise CurationError("object directory cannot contain a newline")
    object_format = git(source.path, "rev-parse", "--show-object-format").strip()
    with tempfile.TemporaryDirectory(prefix="commit-msg-reader-") as temp:
        reader = Path(temp)
        git(
            reader,
            "-c",
            "init.templateDir=",
            "init",
            "--bare",
            f"--object-format={object_format}",
            ".",
        )
        # A separate Git dir excludes source info/attributes, index, config, and worktree.
        (reader / "objects/info/alternates").write_text(
            str(objects) + "\n", encoding="utf-8"
        )
        return mine_snapshot(
            Source(reader, source.name, source.sha, source.origin), counts
        )


def mine_snapshot(source: Source, counts: Counter) -> list[Candidate]:
    log = git(
        source.path,
        "log",
        "--first-parent",
        "--no-merges",
        "-z",
        "--format=%H%x00%aI%x00%an%x00%s%x00%b",
        source.sha,
        "--",
    )
    fields = log.split("\0")
    if fields.pop() != "" or len(fields) % 5:
        raise CurationError(f"{source.path}: malformed Git log field framing")
    candidates = []
    for offset in range(0, len(fields), 5):
        sha, author_date, author, subject, body = fields[offset : offset + 5]
        counts["commits_seen"] += 1
        if author in {"dependabot[bot]", "github-actions[bot]"}:
            counts["commits_bot"] += 1
            continue
        completion = clean_completion(subject, body)
        if completion is None:
            counts["commits_subject_or_completion_budget"] += 1
            continue
        reason = sensitive_reason(completion)
        if reason:
            counts[f"commits_{reason}"] += 1
            continue
        try:
            diff = git(
                source.path,
                f"--attr-source={source.sha}",
                "show",
                "--format=",
                "--unified=1",
                "--no-color",
                "--no-ext-diff",
                "--no-textconv",
                "--src-prefix=a/",
                "--dst-prefix=b/",
                "--find-renames",
                sha,
                "--",
            )
        except NonUtf8Error:
            counts["commits_non_utf8_diff"] += 1
            continue
        lines = reduce_diff(diff, counts)
        # Scan every retained line before truncation so a hidden tail cannot escape screening.
        reason = sensitive_reason("\n".join(lines))
        if reason:
            counts[f"commits_{reason}"] += 1
            continue
        prefix = f"repo: {source.name}\nWrite a conventional commit message for this diff.\n\n"
        suffix = "\n\nCommit message:"
        reduced = truncate_lines(lines, PROMPT_CAP - len(prefix) - len(suffix))
        if reduced is None:
            counts["commits_empty_or_diff_budget"] += 1
            continue
        authored = datetime.fromisoformat(author_date).astimezone(UTC)
        if len(reduced.splitlines()) < len(lines) or "\n[... " in reduced:
            counts["candidates_truncated"] += 1
        candidates.append(
            Candidate(
                source.name,
                sha,
                authored,
                time_split(authored),
                prefix + reduced + suffix,
                completion,
                reduced,
            )
        )
        counts["candidates"] += 1
    return candidates


def pair_digest(prompt: str, completion: str) -> str:
    encoded = json.dumps(
        {"prompt": prompt, "completion": completion},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deduplicate(
    candidates: list[Candidate],
    counts: dict[str, Counter],
    rejections: set[str] | None = None,
) -> list[Candidate]:
    unique = []
    seen = set()
    for row in sorted(candidates, key=lambda c: (c.authored, c.repo, c.sha)):
        key = (row.prompt, row.completion)
        if key in seen:
            counts[row.repo]["rows_exact_duplicate"] += 1
        else:
            seen.add(key)
            unique.append(row)
    frequencies = Counter(row.completion for row in unique)
    diff_splits: dict[str, set[str]] = defaultdict(set)
    for row in unique:
        diff_splits[row.diff].add(row.split)
    kept = []
    used_rejections = set()
    for row in unique:
        if frequencies[row.completion] > 3:
            counts[row.repo]["rows_boilerplate"] += 1
        elif len(diff_splits[row.diff]) > 1:
            # Repo conditioning must not hide the same input patch in multiple splits.
            counts[row.repo]["rows_cross_split_diff"] += 1
        elif rejections and pair_digest(row.prompt, row.completion) in rejections:
            used_rejections.add(pair_digest(row.prompt, row.completion))
            counts[row.repo]["rows_tokenizer_rejected"] += 1
        else:
            kept.append(row)
            counts[row.repo][f"written_{row.split}"] += 1
    if rejections and used_rejections != rejections:
        raise CurationError("rejection manifest has unknown or unused pair hashes")
    return kept


def validate_jsonl(path: Path, expected: int) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"prompt", "completion"}:
                raise CurationError(f"{path}:{number}: unexpected JSONL shape")
            if any(
                not isinstance(value, str) or not value.strip()
                for value in row.values()
            ):
                raise CurationError(f"{path}:{number}: empty or non-string field")
            if (
                len(row["prompt"]) > PROMPT_CAP
                or len(row["completion"]) > COMPLETION_CAP
            ):
                raise CurationError(f"{path}:{number}: character budget exceeded")
            if sensitive_reason(row["prompt"] + row["completion"]):
                raise CurationError(f"{path}:{number}: sensitive-content screen failed")
            rows.append(row)
    if len(rows) != expected:
        raise CurationError(f"{path}: wrote {expected} rows but read {len(rows)}")
    return rows


def histogram(rows: list[Candidate]) -> list[str]:
    lines = [
        "| Split | Percentile bin | Rows | Combined characters min..max | Estimated tokens min..max |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for split in SPLITS:
        lengths = sorted(
            len(row.prompt) + len(row.completion) for row in rows if row.split == split
        )
        for decile in range(10):
            bucket = lengths[
                len(lengths) * decile // 10 : len(lengths) * (decile + 1) // 10
            ]
            chars = f"{min(bucket)}..{max(bucket)}" if bucket else "n/a"
            tokens = (
                f"{math.ceil(min(bucket) / 3.5)}..{math.ceil(max(bucket) / 3.5)}"
                if bucket
                else "n/a"
            )
            lines.append(
                f"| {split} | {decile * 10}–{(decile + 1) * 10}% | {len(bucket)} | {chars} | {tokens} |"
            )
    return lines


def write_outputs(
    out: Path,
    sources: list[Source],
    rows: list[Candidate],
    counts: dict[str, Counter],
    rejection_note: str | None = None,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    report = [
        "# Commit-message curation",
        "",
        "Sources are local clones of the three approved public repositories. Origin allowlisting is a local identity check, not a fresh remote visibility check. No fetch or source checkout changes are performed.",
        "",
        "| Repository | Local path | Pinned main snapshot | Origin |",
        "| --- | --- | --- | --- |",
    ]
    report.extend(
        f"| {s.name} | `{s.path}` | `{s.sha}` | `{s.origin}` |" for s in sources
    )
    report.extend(
        [
            "",
            "## Recipe",
            "",
            "Walk each pinned main with `git log --first-parent --no-merges` in a temporary bare reader sharing only the source object store. Replay with `--pin-main name=full-sha` for every repository in the source table; explicit pins must identify commits reachable from the current local main. Render attributes from the pinned main tree; source worktree/index/info attributes and source/global/system config cannot affect the reader. Splits use author timestamps normalized to UTC: train before 2026-07-15T00:00:00Z; valid from then until 2026-08-15T00:00:00Z; test thereafter. Committer date is not used.",
            "",
            "Keep conventional subjects with at least eight description characters; remove PR-number suffixes and Co-Authored-By, Claude-Session, Signed-off-by trailer lines plus their continuations. Exclude the two named bots. Include a nonempty body only below 200 characters and when the completion fits 300 characters; otherwise keep the subject alone.",
            "",
            "Drop commits with non-UTF-8 diffs, binary files, and both sides of renames touching excluded paths. Excluded directory components: "
            + ", ".join(f"`{p}`" for p in sorted(BLOCKED_DIRS))
            + ". Excluded filename globs: "
            + ", ".join(f"`{p}`" for p in BLOCKED_FILES)
            + ". Reject quoted, whitespace-containing, non-ASCII, traversal, or otherwise unsupported diff headers rather than guessing their paths. Retain diff/hunk headers and +/- lines, discard context and metadata; truncate only on line boundaries, reserving room for the exact omitted-line marker. Require a retained changed line.",
            "",
            "Reject email addresses, known credential patterns, and unsafe control characters in completions or any retained diff line before truncation. The screen is conservative, not proof that arbitrary secrets or previously published private material cannot occur; no working-tree or untracked content is read.",
            "",
            "Exact (prompt, completion) dedup keeps the earliest author timestamp (then repo and SHA). After dedup, drop completions occurring more than three times globally. Also drop every remaining occurrence of an identical reduced diff present in multiple splits, ignoring repo conditioning. Near-duplicate patches, recurring completions below the boilerplate threshold, and base-model pretraining contamination are not ruled out.",
            "",
            "Tokenizer status: UNVERIFIED. This stdlib-only miner does not invoke the Qwen tokenizer or trainer. Prompt <= 1200 characters, completion <= 300; estimated tokens = ceil(combined characters / 3.5), never an actual tokenizer count. Skip count at --seq-len 512 remains UNVERIFIED pending the separate inference-tokenizer check.",
            "",
            rejection_note or "No external tokenizer rejection manifest supplied.",
            "",
            "## Counts",
            "",
            "| Repository | Train | Valid | Test |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for source in sources:
        report.append(
            f"| {source.name} | "
            + " | ".join(
                str(counts[source.name][f"written_{split}"]) for split in SPLITS
            )
            + " |"
        )
    report.append(
        "| TOTAL | "
        + " | ".join(str(sum(row.split == split for row in rows)) for split in SPLITS)
        + " |"
    )
    report.extend(
        [
            "",
            "## Filter counters",
            "",
            "File counters count file blocks; commit/row counters count examples. Truncation is informational and overlaps candidate counts.",
            "",
            "| Counter | " + " | ".join(s.name for s in sources) + " |",
            "| --- | " + " | ".join("---:" for _ in sources) + " |",
        ]
    )
    for key in sorted(set().union(*(set(c) for c in counts.values()))):
        report.append(
            f"| {key} | " + " | ".join(str(counts[s.name][key]) for s in sources) + " |"
        )
    report.extend(
        [
            "",
            "## Length distribution by decile",
            "",
            *histogram(rows),
            "",
            "## Output validation",
            "",
        ]
    )
    at_counts = Counter()
    for split in SPLITS:
        selected = [row for row in rows if row.split == split]
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(
                    json.dumps(
                        {"prompt": row.prompt, "completion": row.completion},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        reread = validate_jsonl(path, len(selected))
        if reread != [
            {"prompt": row.prompt, "completion": row.completion} for row in selected
        ]:
            raise CurationError(f"{path}: readback differs from curated rows")
        for row in reread:
            for text in row.values():
                for line in text.splitlines():
                    if "@" in line:
                        at_counts[
                            "hunk_headers"
                            if line.startswith("@@")
                            else "other_non_email_lines"
                        ] += 1
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        report.append(
            f"- `{path.name}`: read back {len(reread)} valid rows; SHA-256 `{digest}`."
        )
        print(f"{split}: {len(reread)}")
    report.extend(
        [
            "",
            f"Literal @ audit: {at_counts['hunk_headers']} hunk-header lines; {at_counts['other_non_email_lines']} other lines (code decorators, package names, or mentions may contain @). Email-regex matches: 0. Other @ occurrences require inspection; this classification does not assert each is harmless.",
            "",
        ]
    )
    if sum(row.split == "train" for row in rows) < 800:
        report.extend(
            [
                "## Insufficient training population",
                "",
                "Fewer than 800 train rows survived. No filter was relaxed. Proposed next decision: inspect source/time and rejection counts before considering additional approved public history or a dated time-boundary amendment; keep evaluation separation and secret/email screening intact.",
                "",
            ]
        )
    rng = random.Random(0)
    report.extend(
        [
            "## Examples",
            "",
            "Deterministic random sample, seed 0, at most three rows per split. JSON is indented to prevent source content from changing Markdown structure.",
            "",
        ]
    )
    for split in SPLITS:
        selected = [row for row in rows if row.split == split]
        for row in rng.sample(selected, min(3, len(selected))):
            report.extend(
                [
                    f"### {split}: {row.repo} {row.sha}",
                    "",
                    f"Author timestamp: {row.authored.isoformat()}",
                    "",
                ]
            )
            report.extend(
                "    " + line
                for line in json.dumps(
                    {"prompt": row.prompt, "completion": row.completion},
                    ensure_ascii=False,
                    indent=2,
                ).splitlines()
            )
            report.append("")
    (out / "CURATION.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        action="append",
        help="Repeat for approved public local clones; otherwise use the configured default checkout paths.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--pin-main",
        action="append",
        default=[],
        metavar="REPO=SHA",
        help="Replay a full main commit snapshot for a selected repository; repeat per repository.",
    )
    parser.add_argument(
        "--reject-pairs",
        type=Path,
        help="JSON array of exact pair SHA-256 hashes rejected by a separate tokenizer check.",
    )
    args = parser.parse_args(argv)
    try:
        sources = [
            pin_source(path) for path in (args.repo or [Path(p) for p in DEFAULT_REPOS])
        ]
        if len({s.name for s in sources}) != len(sources):
            raise CurationError("provide each public repository at most once")
        sources = apply_main_pins(sources, args.pin_main)
        counts: dict[str, Counter] = {s.name: Counter() for s in sources}
        rejections: set[str] = set()
        rejection_note = None
        if args.reject_pairs:
            raw = args.reject_pairs.read_bytes()
            manifest = json.loads(raw)
            if (
                not isinstance(manifest, list)
                or any(
                    not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item)
                    for item in manifest
                )
                or len(set(manifest)) != len(manifest)
            ):
                raise CurationError(
                    "rejection manifest must be an array of unique lowercase SHA-256 hashes"
                )
            rejections = set(manifest)
            rejection_note = (
                f"External tokenizer rejection manifest: `{args.reject_pairs.resolve()}`, "
                f"SHA-256 `{hashlib.sha256(raw).hexdigest()}`, {len(rejections)} exact pairs. "
                "Pair digest is SHA-256 of UTF-8 JSON with keys sorted, ensure_ascii=false, "
                "compact comma/colon separators, and exactly prompt/completion fields. "
                "Unknown or unused rejection hashes fail closed. This filter records external "
                "decisions; it does not replace rechecking the final outputs with the actual tokenizer."
            )
        candidates = []
        for source in sources:
            print(
                f"Mining {source.name}: pinned main {source.sha}",
                file=sys.stderr,
                flush=True,
            )
            candidates.extend(mine(source, counts[source.name]))
        rows = deduplicate(candidates, counts, rejections)
        write_outputs(args.out, sources, rows, counts, rejection_note)
        print(f"Curation: {args.out / 'CURATION.md'}")
        return 0
    except (CurationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
