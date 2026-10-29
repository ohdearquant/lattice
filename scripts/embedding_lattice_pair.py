#!/usr/bin/env python3
"""Compare local BGE reference records with a long-lived direct Lattice BERT producer."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import embedding_st_adapter as ref


class Producer:
    def __init__(self, executable, directory):
        self.process = subprocess.Popen([str(executable), str(directory)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, encoding="utf-8")

    def encode(self, texts):
        self.process.stdin.write(json.dumps(texts, ensure_ascii=False, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline(4_000_001)
        if not line or len(line) > 4_000_000 or not line.endswith("\n"):
            raise ref.Refusal("invalid producer response")
        return ref._json(line)

    def finish(self):
        self.process.stdin.close()
        extra = self.process.stdout.read(1)
        status = self.process.wait()
        if extra or status != 0:
            raise ref.Refusal("producer failed or returned extra output")

    def abort(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait()


def vector(value):
    if not isinstance(value, list) or len(value) != ref.DIMENSION:
        raise ref.Refusal("embedding dimension mismatch")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ref.Refusal("invalid embedding value")
    norm = math.sqrt(sum(x * x for x in value))
    if abs(norm - 1.0) > 0.001:
        raise ref.Refusal("embedding is not L2 normalized")
    return value


def reference_records(path, cases, hashes, version):
    text = ref._read_bounded(path, 16_000_000)
    records = [ref._json(line) for line in text.splitlines()]
    if len(records) != len(cases):
        raise ref.Refusal("reference case count mismatch")
    keys = {"schema_version", "case_id", "text_sha256", "engine", "engine_version", "model_id",
            "model_revision", "files_sha256", "device", "pooling", "normalization", "prompt",
            "dimension", "embedding"}
    for case, record in zip(cases, records):
        expected = {"schema_version": 1, "case_id": case["id"],
                    "text_sha256": hashlib.sha256(case["text"].encode()).hexdigest(),
                    "engine": "sentence-transformers", "engine_version": version,
                    "model_id": ref.MODEL_ID, "model_revision": ref.REVISION, "files_sha256": hashes,
                    "device": "cpu", "pooling": "cls", "normalization": "l2", "prompt": "",
                    "dimension": ref.DIMENSION}
        if (not isinstance(record, dict) or set(record) != keys
                or type(record["schema_version"]) is not int or type(record["dimension"]) is not int
                or any(record[k] != v for k, v in expected.items())):
            raise ref.Refusal("reference identity or case mismatch")
        vector(record["embedding"])
    return records


def compare(cases, hashes, references, backend, producer, batch_size, lattice_revision):
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ref.Refusal("batch size must be 1 through 32")
    if not isinstance(lattice_revision, str) or len(lattice_revision) != 40 or any(c not in "0123456789abcdef" for c in lattice_revision):
        raise ref.Refusal("expected full lowercase Lattice commit id")
    output = []
    for start in range(0, len(cases), batch_size):
        group = cases[start:start + batch_size]
        texts = [case["text"] for case in group]
        expected_ids = []
        for text in texts:
            tokens = backend.model.tokenizer(text, add_special_tokens=True, truncation=False)
            ids, mask = tokens["input_ids"], tokens["attention_mask"]
            if (not isinstance(ids, list) or not 0 < len(ids) <= ref.MAX_TOKENS
                    or any(type(i) is not int or i < 0 for i in ids) or mask != [1] * len(ids)
                    or tokens.get("token_type_ids", [0] * len(ids)) != [0] * len(ids)):
                raise ref.Refusal("reference tokenizer refused or truncated")
            expected_ids.append(ids)
        result = producer.encode(texts)
        if (not isinstance(result, dict) or set(result) != {"ids", "vectors"}
                or not isinstance(result["ids"], list)
                or any(not isinstance(ids, list) or any(type(i) is not int for i in ids) for ids in result["ids"])
                or result["ids"] != expected_ids):
            raise ref.Refusal("tokenization preflight mismatch")
        vectors = result["vectors"]
        if not isinstance(vectors, list) or len(vectors) != len(group):
            raise ref.Refusal("producer vector count mismatch")
        for offset, (case, values) in enumerate(zip(group, vectors)):
            values = vector(values)
            reference = vector(references[start + offset]["embedding"])
            dot = sum(a * b for a, b in zip(values, reference))
            norm_a = math.sqrt(sum(a * a for a in values))
            norm_b = math.sqrt(sum(b * b for b in reference))
            cosine = dot / (norm_a * norm_b)
            maximum = max(abs(a - b) for a, b in zip(values, reference))
            if cosine < 0.99999 or maximum > 0.005:
                raise ref.Refusal(f"embedding parity failed for {case['id']}: cosine={cosine}, max_abs={maximum}")
            output.append({"schema_version": 1, "case_id": case["id"],
                           "text_sha256": hashlib.sha256(case["text"].encode()).hexdigest(),
                           "engine": "lattice-bert-direct", "engine_version": lattice_revision, "model_id": ref.MODEL_ID,
                           "model_revision": ref.REVISION, "files_sha256": hashes, "device": "cpu",
                           "pooling": "cls", "normalization": "l2", "prompt": "",
                           "dimension": ref.DIMENSION, "embedding": values,
                           "token_preflight": "retokenized separately; not captured during encode",
                           "token_count": len(expected_ids[offset]), "reference_cosine": cosine,
                           "reference_max_abs": maximum})
    return output


def run(directory, manifest, cases_path, reference_path, executable, output_path, lattice_revision, batch_size=8,
        backend_factory=ref.Backend, producer_factory=Producer):
    cases = ref.read_cases(cases_path)
    hashes = ref.verify_snapshot(directory, manifest)
    backend = backend_factory(directory)
    references = reference_records(reference_path, cases, hashes, backend.version)
    producer = producer_factory(executable, directory)
    try:
        records = compare(cases, hashes, references, backend, producer, batch_size, lattice_revision)
        producer.finish()
    except BaseException:
        producer.abort()
        raise
    if ref.verify_snapshot(directory, manifest) != hashes:
        raise ref.Refusal("model files changed during run")
    output_path = Path(output_path)
    fd, temporary = tempfile.mkstemp(prefix=".embedding-pair-", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        os.link(temporary, output_path)
    finally:
        os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model_dir", "manifest", "input_jsonl", "reference_jsonl", "producer", "output_jsonl"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lattice-revision", required=True, help="full source commit used to build producer (caller assertion)")
    args = parser.parse_args()
    try:
        run(args.model_dir, args.manifest, args.input_jsonl, args.reference_jsonl,
            args.producer, args.output_jsonl, args.lattice_revision, args.batch_size)
    except (ref.Refusal, OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        print(f"embedding pair refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
