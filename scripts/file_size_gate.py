#!/usr/bin/env python3
"""ADR-102 D4 file-size gate: a line cap, a ratchet manifest, a data list.

A tracked text file may not exceed a physical-line cap unless the manifest
records it. The manifest is a ratchet: a recorded ceiling may fall but never
rise, and an entry leaves the manifest as soon as its file fits under the cap.
The gate never rewrites the manifest, so a change that adds or raises an entry
is refused rather than normalised into a passing state.

Every input is read at a revision (`git ls-tree`, `git cat-file`, `git diff`),
never from the working tree, so a dirty checkout cannot change a verdict. The
population is derived from the tree rather than from the manifest, and a gate
that finds no files refuses (exit 2) instead of reporting a clean tree it never
scanned.

Exit status:
  0 = clean
  1 = violations, each printed as `FAIL <path>: <reason>`
  2 = refusal (unreadable manifest, base not a commit, empty population)

Usage:
  file_size_gate.py --base <sha> --head <sha>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple, Optional

EXIT_OK = 0
EXIT_VIOLATION = 1
EXIT_REFUSAL = 2

LINE_CAP = 3000
MANIFEST_PATH = "file-size-manifest.toml"
ZERO_SHA = "0" * 40
BINARY_PROBE_BYTES = 8192

MAINTAINED_SUFFIXES = frozenset({".rs", ".metal", ".py", ".sh", ".bash", ".md"})
WORKFLOW_PREFIX = ".github/workflows/"
# Entries naming the Qwen3.5 Metal kernel must say which phase retires them, so
# the ratchet cannot outlive the reorganisation that created it.
RETIRED_BY_PATTERN = "crates/inference/src/forward/metal_qwen35*.rs"

# Inherited from a caller these variables override `-C` and would aim every git
# call at the wrong repository, so they are stripped from every child process.
GIT_SCRUBBED_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR")

RATCHET_KEYS = frozenset({"path", "ceiling", "retired_by", "renamed_from"})
DATA_KEYS = frozenset({"path"})
GLOB_CHARS = frozenset("*?[]")


class Refusal(Exception):
    """The gate cannot render a verdict at all."""


class Verdict(NamedTuple):
    exit_code: int
    header: str
    failures: list[str]


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in GIT_SCRUBBED_VARS:
        env.pop(name, None)
    return env


def _git_argv(repo: Path, args: list[str]) -> list[str]:
    # An empty core.hooksPath keeps every call out of the caller's hook chain,
    # so running the gate does not pull in a repository-wide pre-commit build.
    return ["git", "-C", str(repo), "--no-replace-objects", "-c", "core.hooksPath=", *args]


def _git(repo: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        _git_argv(repo, args),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_child_env(),
    )
    if check and proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise Refusal("git %s failed: %s" % (" ".join(args), detail or "no output"))
    return proc


def rev_is_commit(repo: Path, rev: str) -> bool:
    proc = _git(repo, ["rev-parse", "--verify", "--quiet", rev + "^{commit}"], check=False)
    return proc.returncode == 0


def is_binary(data: bytes) -> bool:
    """A NUL byte in the first 8192 bytes, which is how git decides the same."""
    return b"\0" in data[:BINARY_PROBE_BYTES]


def line_count(data: bytes) -> int:
    """Physical lines: one per LF byte, plus a final unterminated line."""
    count = data.count(b"\n")
    if data and not data.endswith(b"\n"):
        count += 1
    return count


def is_maintained(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    dot = name.rfind(".")
    if dot > 0 and name[dot:] in MAINTAINED_SUFFIXES:
        return True
    if path.startswith(WORKFLOW_PREFIX):
        rest = path[len(WORKFLOW_PREFIX) :]
        if "/" not in rest and (rest.endswith(".yml") or rest.endswith(".yaml")):
            return True
    return False


def needs_retired_by(path: str) -> bool:
    return Path(path).match(RETIRED_BY_PATTERN)


def _ls_tree(repo: Path, rev: str) -> list[tuple[str, str, int, str]]:
    raw = _git(repo, ["ls-tree", "-r", "-l", "-z", "--full-tree", rev, "--"]).stdout
    entries: list[tuple[str, str, int, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, sep, raw_path = record.partition(b"\t")
        if not sep:
            continue
        fields = meta.decode("ascii").split()
        if len(fields) != 4:
            continue
        mode, kind, oid, size = fields
        entries.append((kind, oid, int(size), raw_path.decode("utf-8", "surrogateescape")))
    return entries


def _stream_blobs(repo: Path, names: list[str]) -> Iterator[Optional[bytes]]:
    """Yield each object's bytes in input order, or None when git reports missing."""
    if not names:
        return
    with tempfile.TemporaryFile() as feed:
        for name in names:
            feed.write(name.encode("utf-8", "surrogateescape") + b"\n")
        feed.seek(0)
        proc = subprocess.Popen(
            _git_argv(repo, ["cat-file", "--batch"]),
            stdin=feed,
            stdout=subprocess.PIPE,
            env=_child_env(),
        )
        try:
            for _ in names:
                header = proc.stdout.readline()
                if not header:
                    raise Refusal("git cat-file --batch ended before every object was read")
                parts = header.rstrip(b"\n").split(b" ")
                if parts[-1] == b"missing":
                    yield None
                    continue
                try:
                    size = int(parts[-1])
                except ValueError as exc:
                    raise Refusal("git cat-file --batch header was not understood: %r" % header)
                data = proc.stdout.read(size)
                if len(data) != size:
                    raise Refusal("git cat-file --batch truncated an object")
                proc.stdout.read(1)
                yield data
        finally:
            proc.stdout.close()
            proc.wait()


def _blobs_for(repo: Path, rev: str, paths: list[str]) -> dict[str, Optional[bytes]]:
    """Read the named paths out of a revision, without touching the working tree."""
    wanted = {path: True for path in paths}
    by_path: dict[str, str] = {}
    for kind, oid, _size, path in _ls_tree(repo, rev):
        if kind == "blob" and path in wanted:
            by_path[path] = oid
    ordered = [path for path in paths if path in by_path]
    data = _stream_blobs(repo, [by_path[path] for path in ordered])
    return dict(zip(ordered, data))


def _counts_for(repo: Path, rev: str, paths: list[str]) -> dict[str, tuple[int, bool]]:
    blobs = _blobs_for(repo, rev, paths)
    counts: dict[str, tuple[int, bool]] = {}
    for path, data in blobs.items():
        if data is None:
            continue
        counts[path] = (line_count(data), is_binary(data))
    return counts


def _scan_tree(
    repo: Path, rev: str
) -> tuple[dict[str, tuple[int, bool]], int, int, frozenset[str]]:
    """Every tracked text file at `rev` with its line count, plus population sizes.

    The set of binary paths is returned as well, because a manifest entry naming
    a binary file is a different finding ("this file is ignored") from an entry
    naming a path that is simply absent.
    """
    entries = [entry for entry in _ls_tree(repo, rev) if entry[0] == "blob"]
    data = _stream_blobs(repo, [oid for _kind, oid, _size, _path in entries])
    files: dict[str, tuple[int, bool]] = {}
    binary_paths: set[str] = set()
    text_total = 0
    maintained_total = 0
    for (_kind, _oid, _size, path), blob in zip(entries, data):
        if blob is None:
            continue
        if is_binary(blob):
            binary_paths.add(path)
            continue
        text_total += 1
        if is_maintained(path):
            maintained_total += 1
        files[path] = (line_count(blob), False)
    return files, text_total, maintained_total, frozenset(binary_paths)


def _require_str(entry: dict, key: str, where: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise Refusal("%s: %s must be a non-empty string" % (where, key))
    return value


def _require_int(entry: dict, key: str, where: str) -> int:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise Refusal("%s: %s must be an integer" % (where, key))
    return value


class Manifest(NamedTuple):
    ratchets: dict[str, dict]
    data: dict[str, dict]

    @staticmethod
    def empty() -> "Manifest":
        return Manifest({}, {})


def parse_manifest(text: str, where: str) -> Manifest:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise Refusal("unreadable manifest at %s: %s" % (where, exc))
    unknown_top = sorted(set(document) - {"ratchet", "data"})
    if unknown_top:
        raise Refusal("unreadable manifest at %s: unknown keys %s" % (where, unknown_top))
    ratchets: dict[str, dict] = {}
    for index, entry in enumerate(document.get("ratchet", [])):
        label = "%s ratchet entry %d" % (where, index)
        if not isinstance(entry, dict):
            raise Refusal("unreadable manifest at %s: not a table" % label)
        unknown = sorted(set(entry) - RATCHET_KEYS)
        if unknown:
            raise Refusal("unreadable manifest at %s: unknown keys %s" % (label, unknown))
        path = _require_str(entry, "path", label)
        _require_int(entry, "ceiling", label)
        if "retired_by" in entry:
            _require_str(entry, "retired_by", label)
        if "renamed_from" in entry:
            _require_str(entry, "renamed_from", label)
        if path in ratchets:
            raise Refusal("unreadable manifest at %s: duplicate ratchet path %s" % (where, path))
        ratchets[path] = entry
    data_entries: dict[str, dict] = {}
    for index, entry in enumerate(document.get("data", [])):
        label = "%s data entry %d" % (where, index)
        if not isinstance(entry, dict):
            raise Refusal("unreadable manifest at %s: not a table" % label)
        unknown = sorted(set(entry) - DATA_KEYS)
        if unknown:
            raise Refusal("unreadable manifest at %s: unknown keys %s" % (label, unknown))
        path = _require_str(entry, "path", label)
        if path in data_entries:
            raise Refusal("unreadable manifest at %s: duplicate data path %s" % (where, path))
        data_entries[path] = entry
    for path in sorted(set(ratchets) & set(data_entries)):
        raise Refusal("unreadable manifest at %s: %s is both ratchet and data" % (where, path))
    return Manifest(ratchets, data_entries)


def _data_shape_error(path: str) -> Optional[str]:
    if any(char in GLOB_CHARS for char in path):
        return 'data entry is a glob, not an exact path'
    if path.endswith("/"):
        return 'data entry ends in "/", which names a directory'
    if path.startswith("/") or ".." in path.split("/"):
        return "data entry is not a repository-relative path"
    return None


def _renames(repo: Path, base: str, head: str) -> dict[str, str]:
    """Rename target -> source, as git's own rename detection reports it."""
    raw = _git(repo, ["diff", "-M", "--name-status", "-z", base, head, "--"]).stdout
    fields = [field for field in raw.split(b"\0") if field]
    renames: dict[str, str] = {}
    index = 0
    while index < len(fields):
        status = fields[index].decode("utf-8", "surrogateescape")
        if status.startswith("R"):
            source = fields[index + 1].decode("utf-8", "surrogateescape")
            target = fields[index + 2].decode("utf-8", "surrogateescape")
            renames[target] = source
            index += 3
            continue
        index += 2
    return renames


def _check_ratchet(
    entry: dict,
    path: str,
    head_count: int,
    base_count: Optional[int],
    diff_rules: bool,
    base_ratchets: dict[str, dict],
    renames: dict[str, str],
    failures: list[str],
) -> None:
    ceiling = entry["ceiling"]
    renamed_from = entry.get("renamed_from")
    if needs_retired_by(path) and not entry.get("retired_by"):
        failures.append("FAIL %s: retired_by is required for this path" % path)

    if head_count <= LINE_CAP:
        failures.append(
            "FAIL %s: file is %d lines, at or under the %d-line cap; remove this entry "
            "from the manifest" % (path, head_count, LINE_CAP)
        )
        return

    # The effective ceiling is the tighter of the recorded ceiling and the count
    # the file carried at the base revision. That is what makes a forged ceiling
    # useless: a raised ceiling still has to clear the base count.
    effective = ceiling if base_count is None else min(ceiling, base_count)
    if head_count > effective:
        failures.append(
            "FAIL %s: %d lines exceeds the effective ceiling %d (recorded %d, base %s)"
            % (path, head_count, effective, ceiling, base_count)
        )
    elif head_count < ceiling:
        failures.append(
            "FAIL %s: file is %d lines; lower the ceiling to %d" % (path, head_count, head_count)
        )

    if not diff_rules:
        return

    if renamed_from is not None:
        if renames.get(path) != renamed_from:
            failures.append(
                "FAIL %s: renamed_from is stale: git reports no rename from %s to %s"
                % (path, renamed_from, path)
            )
        elif renamed_from not in base_ratchets:
            failures.append(
                "FAIL %s: renamed_from names no ratchet entry in the base manifest" % path
            )
        return

    detected = renames.get(path)
    if detected is not None:
        failures.append(
            'FAIL %s: this path is a rename target; add renamed_from = "%s"' % (path, detected)
        )
    if path in base_ratchets:
        if ceiling > base_ratchets[path]["ceiling"]:
            failures.append(
                "FAIL %s: ceiling raised from %d to %d in this change"
                % (path, base_ratchets[path]["ceiling"], ceiling)
            )
        return
    if base_count is not None and base_count <= LINE_CAP:
        failures.append(
            "FAIL %s: cannot return: the file was %d lines at base, at or under the "
            "%d-line cap" % (path, base_count, LINE_CAP)
        )
    else:
        failures.append("FAIL %s: new ratchet entry: no entry for this path at base" % path)


def _check_data_entry(
    path: str,
    files: dict[str, tuple[int, bool]],
    binary_paths: frozenset[str],
    all_paths: frozenset[str],
    failures: list[str],
) -> None:
    shape_error = _data_shape_error(path)
    if shape_error is not None:
        failures.append("FAIL %s: %s" % (path, shape_error))
        return
    if is_maintained(path):
        failures.append("FAIL %s: data entry on a maintained path is refused" % path)
        return
    if path in binary_paths:
        failures.append("FAIL %s: data entry is stale: the file is binary, so it is ignored" % path)
        return
    if path not in files:
        if any(other.startswith(path + "/") for other in all_paths):
            failures.append('FAIL %s: data entry names a directory' % path)
            return
        failures.append("FAIL %s: data entry is stale: no such tracked text file at head" % path)
        return
    count, _binary = files[path]
    if count <= LINE_CAP:
        failures.append(
            "FAIL %s: data entry is stale: the file is %d lines, at or under the %d-line cap"
            % (path, count, LINE_CAP)
        )


def evaluate(
    repo: Path,
    base: str,
    head: str,
    manifest_path: str = MANIFEST_PATH,
) -> Verdict:
    """Run the gate over a change and report its verdict, never the working tree."""
    if not rev_is_commit(repo, head):
        return Verdict(EXIT_REFUSAL, "", ["REFUSAL: head revision is not a commit: %s" % head])

    files, text_total, maintained_total, binary_paths = _scan_tree(repo, head)
    # The population line is assembled before any base check so a refusal still
    # reports what was scanned; a refusal that prints no population would read
    # like a gate that never looked.
    header = "population: %d tracked text files (%d maintained), base %s, head %s" % (
        text_total,
        maintained_total,
        base,
        head,
    )
    if text_total == 0:
        return Verdict(EXIT_REFUSAL, header, ["REFUSAL: population is empty at head %s" % head])

    base_available = base != ZERO_SHA
    if base_available and not rev_is_commit(repo, base):
        return Verdict(EXIT_REFUSAL, header, ["REFUSAL: base revision is not a commit: %s" % base])

    head_manifest_blob = _blobs_for(repo, head, [manifest_path]).get(manifest_path)
    try:
        if head_manifest_blob is None:
            head_manifest = Manifest.empty()
        else:
            head_manifest = parse_manifest(
                head_manifest_blob.decode("utf-8"), "%s:%s" % (head, manifest_path)
            )
    except Refusal as exc:
        return Verdict(EXIT_REFUSAL, header, ["REFUSAL: %s" % exc])

    base_manifest = Manifest.empty()
    diff_rules = False
    if base_available:
        base_blob = _blobs_for(repo, base, [manifest_path]).get(manifest_path)
        if base_blob is not None:
            try:
                base_manifest = parse_manifest(
                    base_blob.decode("utf-8"), "%s:%s" % (base, manifest_path)
                )
            except Refusal as exc:
                return Verdict(EXIT_REFUSAL, header, ["REFUSAL: %s" % exc])
            # The manifest diff rules need a baseline to compare against. A base
            # revision that carries no manifest is the change that seeds one, so
            # there is nothing to compare and every entry would read as new; the
            # per-entry ceiling checks below still run against the base counts.
            diff_rules = True

    base_counts: dict[str, tuple[int, bool]] = {}
    if base_available:
        wanted = sorted(
            set(head_manifest.ratchets)
            | {entry.get("renamed_from", "") for entry in head_manifest.ratchets.values()}
            | set(base_manifest.ratchets)
        )
        wanted = [path for path in wanted if path]
        base_counts = _counts_for(repo, base, wanted)

    renames: dict[str, str] = {}
    if diff_rules:
        renames = _renames(repo, base, head)

    failures: list[str] = []
    all_paths = frozenset(files)

    for path in sorted(head_manifest.ratchets):
        entry = head_manifest.ratchets[path]
        if not is_maintained(path):
            failures.append(
                "FAIL %s: ratchet entry on a non-maintained path; use a data entry" % path
            )
            continue
        if path in binary_paths:
            failures.append("FAIL %s: ratchet entry names a binary file, which is ignored" % path)
            continue
        if path not in files:
            failures.append("FAIL %s: ratchet entry names no tracked text file at head" % path)
            continue
        count, _binary = files[path]
        renamed_from = entry.get("renamed_from")
        lookup = renamed_from if isinstance(renamed_from, str) and renamed_from else path
        base_count = base_counts.get(lookup, (None, False))[0]
        _check_ratchet(
            entry,
            path,
            count,
            base_count,
            diff_rules,
            base_manifest.ratchets,
            renames,
            failures,
        )

    for path in sorted(head_manifest.data):
        _check_data_entry(path, files, binary_paths, all_paths, failures)

    listed_data = set(head_manifest.data)
    listed_ratchets = set(head_manifest.ratchets)
    for path in sorted(files):
        count, binary = files[path]
        if binary:
            continue
        if is_maintained(path):
            if path not in listed_ratchets and count > LINE_CAP:
                failures.append(
                    "FAIL %s: %d lines exceeds the %d-line cap with no ratchet entry"
                    % (path, count, LINE_CAP)
                )
            continue
        if path not in listed_data and count > LINE_CAP:
            failures.append(
                "FAIL %s: %d lines, not a maintained file and with no data entry"
                % (path, count)
            )

    if diff_rules:
        for path in sorted(base_manifest.ratchets):
            if path in head_manifest.ratchets:
                continue
            if any(source == path for source in renames.values()):
                continue
            if path not in files:
                continue
            count, _binary = files[path]
            if count > LINE_CAP:
                failures.append(
                    "FAIL %s: entry removed while the file is still %d lines at head"
                    % (path, count)
                )

    exit_code = EXIT_VIOLATION if failures else EXIT_OK
    return Verdict(exit_code, header, failures)


def run_check(repo: Path, base: str, head: str) -> int:
    verdict = evaluate(repo, base, head)
    if verdict.header:
        print(verdict.header)
    for failure in verdict.failures:
        print(failure)
    return verdict.exit_code


def repository_root() -> Path:
    proc = _git(Path.cwd(), ["rev-parse", "--show-toplevel"], check=False)
    if proc.returncode == 0:
        return Path(proc.stdout.decode("utf-8", "replace").strip())
    return Path.cwd()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ADR-102 D4 file-size gate")
    parser.add_argument("--base", help="base commit, or the all-zero sha for no base")
    parser.add_argument("--head", help="head commit")
    args = parser.parse_args(argv)

    if not args.base or not args.head:
        parser.error("--base and --head are required")
    try:
        return run_check(repository_root(), args.base, args.head)
    except Refusal as exc:
        print("REFUSAL: %s" % exc)
        return EXIT_REFUSAL


if __name__ == "__main__":
    sys.exit(main())
