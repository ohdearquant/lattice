#!/usr/bin/env python3
"""Bracket one crate's packaged archive against the crates.io upload limit.

Reads the file list `cargo package --list` produces and reports the gzipped tar
size that list implies. Builds nothing and needs no registry, which is the whole
point: cargo cannot package a crate whose internal path dependencies are not yet
live on the registry, so at release time the only crate it can self-measure is
the leaf tier.

Invoked with the crate directory as the working directory, because
`cargo package --list` prints crate-relative paths.
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
import tarfile

LIMIT = 10 * 1024 * 1024

# The three entries cargo synthesizes at package time. They appear in --list
# output and are never on disk; every other listed path must exist, or the
# measurement is being taken against the wrong tree.
GENERATED = frozenset({".cargo_vcs_info.json", "Cargo.lock", "Cargo.toml.orig"})

# Cargo's own default. Level 9 shaves a few hundred KB off the estimate, which
# flatters a crate that is close to the limit in the one direction that matters.
COMPRESS_LEVEL = 6

VCS_INFO = b'{"git":{"sha1":"0000000000000000000000000000000000000000"},"path_in_vcs":""}'


class Refusal(Exception):
    """The inputs cannot support a size claim."""


class _ByteCounter:
    """Sink that measures the gzip stream without holding it."""

    def __init__(self) -> None:
        self.total = 0

    def write(self, data: bytes) -> int:
        self.total += len(data)
        return len(data)

    def flush(self) -> None:
        pass


def bracket(crate: str, listed: list[str], lockfile: str | None) -> tuple[int, int]:
    """Return (compressed size, number of listed entries)."""
    if not listed:
        raise Refusal(f"{crate}: empty file list, refusing to report a size")

    on_disk = [path for path in listed if path not in GENERATED]
    # os.path.exists follows symlinks, so a listed but broken link is reported as
    # absent and refused below rather than measured as an empty file.
    absent = [path for path in on_disk if not os.path.exists(path)]
    if absent:
        raise Refusal(
            f"{crate}: {len(absent)} listed files are not on disk "
            f"(first: {absent[0]}); refusing to report a size"
        )

    counter = _ByteCounter()
    prefix = f"{crate}-0.0.0"
    # Streamed rather than buffered: the uncompressed tar is several times the
    # size of the archive being bounded, and holding all of it to measure a
    # number would make memory scale with the fixtures rather than the answer.
    #
    # dereference=True is load-bearing. Cargo follows a symlink and packages its
    # target's contents, while tarfile's default writes a link header of zero
    # bytes. Measured on a crate cargo archived at 10,490,174 bytes, over the
    # limit from one 5 MiB file reached through a link: the buffered default
    # reported 5,245,520 bytes and passed it. A size gate that undercounts is
    # worse than no gate, because it answers the question it was installed to
    # answer and answers it wrong.
    with gzip.GzipFile(
        fileobj=counter, mode="wb", compresslevel=COMPRESS_LEVEL, mtime=0
    ) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", dereference=True) as tar:
            for path in on_disk:
                tar.add(path, arcname=f"{prefix}/{path}")
            # Bracket the synthesized entries with real content rather than a
            # guess. Cargo.toml.orig is this crate's manifest, and the packaged
            # Cargo.lock is a pruned form of the workspace lockfile, so the
            # workspace copy bounds it from above.
            sources = [("Cargo.toml.orig", os.path.join(os.getcwd(), "Cargo.toml"))]
            if lockfile:
                sources.append(("Cargo.lock", lockfile))
            for arcname, source in sources:
                if os.path.exists(source):
                    tar.add(source, arcname=f"{prefix}/{arcname}")
            info = tarfile.TarInfo(name=f"{prefix}/.cargo_vcs_info.json")
            info.size = len(VCS_INFO)
            tar.addfile(info, io.BytesIO(VCS_INFO))

    return counter.total, len(listed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crate")
    parser.add_argument("list_file", help="output of `cargo package --list`")
    parser.add_argument("--lockfile", default=None, help="workspace Cargo.lock")
    parser.add_argument(
        "--warn-fraction",
        type=float,
        default=float(os.environ.get("PKG_SIZE_WARN_FRACTION", "0.75")),
    )
    args = parser.parse_args(argv)

    with open(args.list_file, encoding="utf-8") as handle:
        listed = [line.strip() for line in handle if line.strip()]

    try:
        size, count = bracket(args.crate, listed, args.lockfile)
    except Refusal as refusal:
        print(f"package-size-check: {refusal}", file=sys.stderr)
        return 1

    fraction = size / LIMIT
    if size > LIMIT:
        verdict, rc = "OVER LIMIT", 1
    elif fraction >= args.warn_fraction:
        verdict, rc = "near limit", 0
    else:
        verdict, rc = "ok", 0

    print(
        f"{args.crate:22s} {size:>10,} B  {fraction * 100:5.1f}% of {LIMIT:,}  "
        f"headroom {LIMIT - size:>10,} B  {count:>4d} files  [{verdict}]"
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
