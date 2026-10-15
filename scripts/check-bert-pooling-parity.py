#!/usr/bin/env python3
"""Run named BERT pooling controls and compare fresh native/HTTP evidence."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ["hello", "hello world", "quiet blue river"]
DIMENSIONS = 384
MODELS = [("bge-small-en-v1.5", "cls"), ("all-minilm-l6-v2", "mean")]
FIXTURE_TESTS = [
    'bert_pooling_fixture::tests::checkpoint_is_deterministic_and_loads_all_twenty_one_tensors',
    'bert_pooling_fixture::tests::checkpoint_makes_pooling_and_input_order_observable',
    'bert_pooling_fixture::tests::checkpoint_refuses_existing_destination_without_replacement',
    'bert_pooling_fixture::tests::checkpoint_failed_write_removes_only_owned_files',
]
NATIVE_TESTS = [
    'service::native::pooling_tests::native_pooling_bge_matches_explicit_cls_for_single_and_batch',
    'service::native::pooling_tests::native_pooling_minilm_matches_explicit_mean_for_single_and_batch',
    'service::native::pooling_tests::native_pooling_acquisition_failure_preserves_model_name_and_error',
    'service::native::pooling_tests::native_pooling_artifact_writer_refuses_to_overwrite',
]
ROUTE_TESTS = [
    'imp::tests::bert_pooling_tests::embeddings_default_pooling_matches_library_for_single_and_batch',
    'imp::tests::bert_pooling_tests::embeddings_explicit_pooling_override_matches_library',
    'imp::tests::bert_pooling_tests::embedding_startup_preserves_load_and_override_errors',
    'imp::tests::bert_pooling_tests::pooling_artifact_rejects_invalid_identity_schema_and_vectors',
    'imp::tests::bert_pooling_tests::pooling_comparison_rejects_nonfinite_wrong_shape_and_wrong_values',
]
EXPORT_TEST = "service::native::pooling_tests::native_pooling_exports_bound_cross_runtime_artifact"
IMPORT_TEST = "imp::tests::bert_pooling_tests::embeddings_pooling_matches_bound_native_artifact"
CONTEXT_TEST = "imp::tests::embeddings_item_over_context_window_400"
SOURCE_PATHS = [
    "crates/embed/Cargo.toml",
    "crates/embed/src/model.rs",
    "crates/embed/src/service/native.rs",
    "crates/embed/src/service/native_pooling_tests.rs",
    "crates/inference/src/lib.rs",
    "crates/inference/src/bert_pooling_fixture.rs",
    "crates/inference/src/bin/lattice_serve.rs",
    "crates/inference/src/bin/lattice_serve/bert_pooling_tests.rs",
    "crates/inference/src/model/bert.rs",
    "crates/inference/src/pool.rs",
    "scripts/check-bert-pooling-parity.py",
    "Cargo.lock",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_snapshot():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"], cwd=ROOT)
    hashes = {name: digest((ROOT / name).read_bytes()) for name in SOURCE_PATHS}
    identity = digest(head + b"\0" + diff + b"\0" + json.dumps(hashes, sort_keys=True).encode())
    return {"head": head.decode(), "source_id": identity, "source_hashes": hashes}


def validate_log(log, expected):
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected test list must be nonempty and unique")
    names = re.findall(r"^test (?!result:)(\S+) \.\.\. ok$", log, re.MULTILINE)
    results = re.findall(r"^test (?!result:).+ \.\.\. .+$", log, re.MULTILINE)
    if sorted(names) != sorted(expected) or len(results) != len(expected):
        raise ValueError(f"test population mismatch: expected {expected!r}, passed {names!r}")
    starts = re.findall(r"^running (\d+) tests?$", log, re.MULTILINE)
    summaries = re.findall(
        r"^test result: ok\. (\d+) passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out; finished in .+$",
        log,
        re.MULTILINE,
    )
    if starts != [str(len(expected))] or summaries != [str(len(expected))]:
        raise ValueError("expected exactly one nonempty harness and zero failed/ignored tests")
    if re.search(r"^test result: FAILED|\.\.\. (?:FAILED|ignored)|^error(?:\[|:)", log, re.MULTILINE):
        raise ValueError("failed, ignored or compile-error output is not accepted")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def exact_keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"expected object keys {keys!r}")


def validate_vector(value):
    if not isinstance(value, list) or len(value) != DIMENSIONS:
        raise ValueError("wrong vector dimensions")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("nonfinite or nonnumeric vector")
    if sum(x * x for x in value) < 0.5:
        raise ValueError("zero or degenerate embedding")


def validate_artifact(path, run_id, source_id):
    artifact = json.loads(path.read_text(), object_pairs_hook=unique_object)
    exact_keys(artifact, ["schema", "run_id", "source_id", "checkpoint_hashes", "dimensions", "inputs", "models"])
    if type(artifact["schema"]) is not int or artifact["schema"] != 1:
        raise ValueError("wrong artifact schema")
    if artifact["run_id"] != run_id or artifact["source_id"] != source_id:
        raise ValueError("stale run or source identity")
    if type(artifact["dimensions"]) is not int or artifact["dimensions"] != DIMENSIONS:
        raise ValueError("wrong artifact dimensions")
    if artifact["inputs"] != INPUTS:
        raise ValueError("different input corpus")
    hashes = artifact["checkpoint_hashes"]
    exact_keys(hashes, ["config.json", "model.safetensors", "vocab.txt"])
    if any(not isinstance(x, str) or not re.fullmatch(r"[0-9a-f]{64}", x) for x in hashes.values()):
        raise ValueError("invalid checkpoint hashes")
    models = artifact["models"]
    if not isinstance(models, list) or len(models) != len(MODELS):
        raise ValueError("expected exactly two model cases")
    for model, (name, pooling) in zip(models, MODELS):
        exact_keys(model, ["name", "pooling", "single", "batch"])
        if (model["name"], model["pooling"]) != (name, pooling):
            raise ValueError("missing, reordered or incorrect pooling case")
        validate_vector(model["single"])
        batch = model["batch"]
        if not isinstance(batch, list) or len(batch) != len(INPUTS):
            raise ValueError("incorrect batch population")
        for vector in batch:
            validate_vector(vector)
        if any(abs(a - b) > 1e-5 + 1e-5 * abs(b) for a, b in zip(model["single"], batch[0])):
            raise ValueError("single/batch disagreement")
        if any(max(abs(a - b) for a, b in zip(batch[i], batch[j])) < 1e-3
               for i in range(len(batch)) for j in range(i)):
            raise ValueError("batch rows do not discriminate order")
    for index in range(len(INPUTS)):
        if math.dist(models[0]["batch"][index], models[1]["batch"][index]) < 1e-3:
            raise ValueError("CLS and Mean do not discriminate pooling")
    return artifact


def command_log(command, label, expected, output, env):
    path = output / (label + ".log")
    with path.open("x") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    content = path.read_text(errors="replace")
    print(content, end="" if content.endswith("\n") else "\n", flush=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    validate_log(content, expected)
    return {"command": command, "passed": expected, "log_sha256": digest(path.read_bytes())}


def test_command(package, target, features, selector, ignored=False, skip=None):
    command = ["cargo", "test", "--locked", "-p", package, *target]
    if features:
        command += ["--features", features]
    command += [selector, "--", "--show-output", "--test-threads=1"]
    if ignored:
        command += ["--ignored", "--exact"]
    if skip:
        command += ["--skip", skip]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-only", action="store_true", help="run portable fixture/native controls only")
    parser.add_argument("--output-dir", type=Path, help="new absolute directory outside this checkout")
    args = parser.parse_args()
    if not args.native_only and sys.platform != "darwin":
        parser.error("actual standalone HTTP route gate requires macOS; use --native-only for portable controls")
    if args.output_dir:
        output = args.output_dir
        if not output.is_absolute() or output.resolve().is_relative_to(ROOT):
            parser.error("output directory must be absolute and outside the checkout")
        output.mkdir(parents=False, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="lattice-bert-pooling-"))
    before = source_snapshot()
    run_id = uuid.uuid4().hex
    artifact = output / "native.json"
    env = dict(os.environ)
    env.update({"CARGO_TERM_COLOR": "never", "LATTICE_BERT_POOLING_RUN_ID": run_id,
                "LATTICE_BERT_POOLING_SOURCE_ID": before["source_id"],
                "LATTICE_BERT_POOLING_ARTIFACT": str(artifact)})
    receipts = []
    fixture = test_command("lattice-inference", ["--lib"], "test-utils", "bert_pooling_fixture::tests::")
    receipts.append(command_log(fixture, "fixture", FIXTURE_TESTS, output, env))
    native = test_command("lattice-embed", ["--lib"], "", "service::native::pooling_tests::", skip=EXPORT_TEST)
    receipts.append(command_log(native, "native", NATIVE_TESTS, output, env))
    if not args.native_only:
        features = "f16,metal-gpu,serve,test-utils"
        route = test_command("lattice-inference", ["--bin", "lattice_serve"], features,
                             "imp::tests::bert_pooling_tests::", skip=IMPORT_TEST)
        receipts.append(command_log(route, "route", ROUTE_TESTS, output, env))
        context = test_command("lattice-inference", ["--bin", "lattice_serve"], features, CONTEXT_TEST)
        context.append("--exact")
        receipts.append(command_log(context, "context", [CONTEXT_TEST], output, env))
        export = test_command("lattice-embed", ["--lib"], "", EXPORT_TEST, ignored=True)
        receipts.append(command_log(export, "native-export", [EXPORT_TEST], output, env))
        validate_artifact(artifact, run_id, before["source_id"])
        artifact_hash = digest(artifact.read_bytes())
        route_import = test_command("lattice-inference", ["--bin", "lattice_serve"], features, IMPORT_TEST, ignored=True)
        receipts.append(command_log(route_import, "route-import", [IMPORT_TEST], output, env))
        if digest(artifact.read_bytes()) != artifact_hash:
            raise ValueError("route consumer modified the native artifact")
    else:
        artifact_hash = None
    if source_snapshot() != before:
        raise ValueError("source changed during gate")
    receipt = dict(before, run_id=run_id, native_only=args.native_only, commands=receipts, artifact_sha256=artifact_hash)
    (output / "PASS.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"BERT_POOLING_GATE_PASS {output}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode if error.returncode > 0 else 1)
    except (OSError, ValueError) as error:
        print(f"BERT_POOLING_GATE_FAILED: {error}", file=sys.stderr)
        sys.exit(1)
