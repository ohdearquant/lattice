#!/usr/bin/env python3
"""Retain the four absolute Criterion means independently of gate verdicts."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import uuid


PHASES = ("base1", "head1", "head2", "base2")
BASELINES = ("compare-base", "new", "compare-head", "new")


class CollectionError(Exception):
    def __init__(self, path, message, detail=None):
        super().__init__(message)
        self.path = str(path)
        self.detail = detail


def finite_number(value, path, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollectionError(path, f"{field} must be a finite JSON number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise CollectionError(path, f"{field} must be a finite JSON number")


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def read_mean(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeError, ValueError, RecursionError) as error:
        raise CollectionError(path, "cannot read Criterion estimates", str(error)) from error
    mean = data.get("mean") if isinstance(data, dict) else None
    if not isinstance(mean, dict):
        raise CollectionError(path, "missing mean object")
    finite_number(mean.get("point_estimate"), path, "mean.point_estimate")
    interval = mean.get("confidence_interval")
    if not isinstance(interval, dict):
        raise CollectionError(path, "missing mean.confidence_interval")
    for field in ("lower_bound", "upper_bound"):
        finite_number(interval.get(field), path, f"mean.confidence_interval.{field}")
    if interval["lower_bound"] > interval["upper_bound"]:
        raise CollectionError(path, "mean.confidence_interval.lower_bound exceeds upper_bound")
    # Criterion computes these fields independently; its point can lie outside its CI.
    try:
        json.dumps(mean, allow_nan=False)
    except (ValueError, RecursionError) as error:
        raise CollectionError(path, "non-finite mean field", str(error)) from error
    return mean


def collect_phase(root, baseline):
    rows = {}

    def walk_error(error):
        raise error

    try:
        for directory, _, files in os.walk(root, onerror=walk_error):
            parent = Path(directory)
            if parent.name != baseline or "estimates.json" not in files:
                continue
            name = parent.parent.relative_to(root).as_posix()
            if name == ".":
                raise CollectionError(parent, "missing benchmark identity")
            rows[name] = read_mean(parent / "estimates.json")
    except OSError as error:
        raise CollectionError(error.filename or root, "cannot scan Criterion phase", str(error)) from error
    if not rows:
        raise CollectionError(root, f"no estimates for baseline {baseline}")
    return rows


def collect(args):
    provenance_path = Path(args.provenance)
    try:
        raw = provenance_path.read_bytes()
        spec = importlib.util.spec_from_file_location(
            "phase_estimates_gate", Path(__file__).resolve().parent.parent / "perf-bench-gate.py"
        )
        gate = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = gate
        spec.loader.exec_module(gate)
        provenance = gate.load_run_provenance(provenance_path)
        if provenance_path.read_bytes() != raw:
            raise ValueError("provenance changed while collecting")
        provenance_text = raw.decode("utf-8")
    except (OSError, UnicodeError, ValueError) as error:
        raise CollectionError(provenance_path, "invalid run provenance", str(error)) from error

    fields = provenance.fields
    expected = {
        "base_ref": args.base_ref, "head_ref": args.head_ref,
        "base_sha": args.base_revision, "head_sha": args.head_revision,
        "criterion_mode": args.resolution, "baseline_name": "compare-base",
        "targets": ", ".join(f"{target[0]}:{target[1]}" for target in args.target),
    }
    for target in args.target:
        crate, _, features, _ = target[:4]
        expected[f"{crate.removeprefix('lattice-')}_features"] = features or "<none>"
    expected["filters"] = " ".join(
        f"{target[0].removeprefix('lattice-')}='{target[3] or '<all>'}'"
        for target in args.target
    )
    for key, value in expected.items():
        if fields.get(key) != value:
            raise CollectionError(provenance_path, f"provenance does not match {key}")

    targets = []
    for crate, target, features, group_filter, *roots in args.target:
        target_id = f"{crate}:{target}"
        phases = [collect_phase(Path(root), baseline) for root, baseline in zip(roots, BASELINES)]
        population = set(phases[0])
        if any(set(phase) != population for phase in phases[1:]):
            raise CollectionError(roots[0], f"phase populations differ for {target_id}")
        targets.append({
            "id": target_id, "crate": crate, "target": target,
            "features": features, "filter": group_filter,
            "benchmarks": [{
                "id": name,
                "phases": {phase: {"mean": rows[name]} for phase, rows in zip(PHASES, phases)},
            } for name in sorted(population)],
        })
    return {
        "schema": "lattice-bench-phase-estimates-v1",
        "base_ref": args.base_ref, "head_ref": args.head_ref,
        "base_revision": args.base_revision, "head_revision": args.head_revision,
        "resolution": args.resolution,
        "provenance_sha256": hashlib.sha256(raw).hexdigest(),
        "provenance_text": provenance_text,
        "targets": sorted(targets, key=lambda target: target["id"]),
    }


def publish(path, data):
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory link publishes complete bytes without replacing a prior run.
        os.link(temporary, path)
    except OSError as error:
        raise CollectionError(error.filename or path, "cannot publish phase estimates", str(error)) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("provenance", "output-dir", "base-ref", "head-ref", "base-revision", "head-revision"):
        parser.add_argument(f"--{field}", required=True)
    parser.add_argument("--resolution", choices=("quick", "full"), required=True)
    parser.add_argument("--target", nargs=8, action="append", required=True,
                        metavar=("CRATE", "TARGET", "FEATURES", "FILTER", "BASE1", "HEAD1", "HEAD2", "BASE2"))
    args = parser.parse_args()
    run_id = str(uuid.uuid4())
    path = Path(args.output_dir).absolute() / f"{run_id}.json"
    try:
        artifact = {"run_id": run_id, **collect(args)}
        data = (json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        publish(path, data)
    except CollectionError as error:
        record = {
            "schema": "lattice-bench-phase-estimates-error-v1", "run_id": run_id,
            "artifact_path": str(path), "path": error.path, "message": str(error),
        }
        if error.detail is not None:
            record["detail"] = error.detail
        print(json.dumps(record, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"schema": "lattice-bench-phase-estimates-published-v1",
                      "path": str(path), "sha256": hashlib.sha256(data).hexdigest()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
