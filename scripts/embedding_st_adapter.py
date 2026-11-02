#!/usr/bin/env python3
"""Produce offline CPU embedding JSONL for a locally provisioned BGE-small snapshot.

The manifest records caller-supplied SHA-256 digests; it is not an independent
attestation of their origin. No model files or dependencies are downloaded.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

MODEL_ID = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DIMENSION = 384
MAX_TOKENS = 512
FILES = (
    "config.json", "config_sentence_transformers.json", "model.safetensors",
    "modules.json", "sentence_bert_config.json", "1_Pooling/config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
)


class Refusal(ValueError):
    """An input or model did not satisfy the adapter contract."""


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise Refusal(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _json(text):
    def reject_constant(value):
        raise Refusal(f"non-finite JSON value: {value}")
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=reject_constant)
    except (ValueError, TypeError) as exc:
        raise Refusal(f"invalid JSON: {exc}") from exc


def _read_bounded(path, limit):
    with open(path, "rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise Refusal(f"file too large: {path}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Refusal(f"invalid UTF-8: {path}") from exc


def read_cases(path):
    text = _read_bounded(path, 1_048_576)
    result, seen = [], set()
    for line in text.splitlines():
        case = _json(line)
        if not isinstance(case, dict) or set(case) != {"id", "text"}:
            raise Refusal("each case must contain exactly id and text")
        case_id, prompt = case["id"], case["text"]
        if not isinstance(case_id, str) or not case_id or len(case_id) > 256:
            raise Refusal("invalid case id")
        if case_id in seen:
            raise Refusal("duplicate case id")
        if not isinstance(prompt, str) or not prompt or len(prompt.encode("utf-8")) > 65536:
            raise Refusal("invalid case text")
        seen.add(case_id)
        result.append(case)
        if len(result) > 1024:
            raise Refusal("too many cases")
    if not result:
        raise Refusal("no cases")
    return result


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_snapshot(directory, manifest_path):
    manifest = _json(_read_bounded(manifest_path, 65536))
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "model_id", "revision", "files"}:
        raise Refusal("invalid model manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise Refusal("unsupported manifest version")
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION:
        raise Refusal("model identity mismatch")
    expected = manifest["files"]
    if not isinstance(expected, dict) or set(expected) != set(FILES):
        raise Refusal("model manifest file set mismatch")
    hashes = {}
    for name in FILES:
        digest = expected[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise Refusal(f"invalid digest: {name}")
        actual = _digest(directory / name)
        if actual != digest:
            raise Refusal(f"digest mismatch: {name}")
        hashes[name] = actual
    modules = _json(_read_bounded(directory / "modules.json", 65536))
    if not isinstance(modules, list) or [m.get("type") if isinstance(m, dict) else None for m in modules] != [
        "sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
    ] or [m.get("path") for m in modules] != ["", "1_Pooling", "2_Normalize"]:
        raise Refusal("unexpected sentence-transformers modules")
    pooling = _json(_read_bounded(directory / "1_Pooling/config.json", 65536))
    if not isinstance(pooling, dict) or pooling.get("word_embedding_dimension") != DIMENSION or pooling.get("pooling_mode_cls_token") is not True or any(pooling.get(k, False) is not False for k in ("pooling_mode_mean_tokens", "pooling_mode_max_tokens", "pooling_mode_mean_sqrt_len_tokens", "pooling_mode_weightedmean_tokens", "pooling_mode_lasttoken")):
        raise Refusal("expected CLS pooling")
    config = _json(_read_bounded(directory / "sentence_bert_config.json", 65536))
    if not isinstance(config, dict) or config.get("max_seq_length") != MAX_TOKENS or config.get("do_lower_case") is not True:
        raise Refusal("unexpected tokenizer configuration")
    return hashes


class Backend:
    def __init__(self, directory):
        try:
            import sentence_transformers
        except ImportError as exc:
            raise Refusal("sentence-transformers dependency is unavailable") from exc
        self.version = sentence_transformers.__version__
        self.model = sentence_transformers.SentenceTransformer(
            str(directory), device="cpu", local_files_only=True, trust_remote_code=False,
            model_kwargs={"use_safetensors": True},
        )
        modules = list(self.model._modules.values())
        if ([type(m).__name__ for m in modules] != ["Transformer", "Pooling", "Normalize"]
                or str(self.model.device) != "cpu"
                or self.model.max_seq_length != MAX_TOKENS
                or self.model.get_sentence_embedding_dimension() != DIMENSION):
            raise Refusal("loaded model does not match CPU BGE-small contract")
        pooling = modules[1]
        if not pooling.pooling_mode_cls_token or any(getattr(pooling, name, False) for name in (
            "pooling_mode_mean_tokens", "pooling_mode_max_tokens", "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_weightedmean_tokens", "pooling_mode_lasttoken",
        )):
            raise Refusal("loaded model is not CLS pooled")

    def encode(self, texts):
        for text in texts:
            encoded = self.model.tokenizer(text, add_special_tokens=True, truncation=False)
            if len(encoded["input_ids"]) > MAX_TOKENS:
                raise Refusal("input exceeds the model token limit")
        return self.model.encode(texts, prompt="", device="cpu", batch_size=8,
                                 precision="float32", normalize_embeddings=False,
                                 convert_to_numpy=True, show_progress_bar=False)


def produce(cases, hashes, backend):
    vectors = backend.encode([case["text"] for case in cases])
    if len(vectors) != len(cases):
        raise Refusal("embedding count mismatch")
    records = []
    for case, vector in zip(cases, vectors):
        if len(vector) != DIMENSION:
            raise Refusal("embedding dimension mismatch")
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise Refusal("invalid embedding value") from exc
        if not all(math.isfinite(value) for value in values):
            raise Refusal("non-finite embedding")
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isfinite(norm) or abs(norm - 1.0) > 0.001:
            raise Refusal("embedding is not L2 normalized")
        records.append({"schema_version": 1, "case_id": case["id"],
                        "text_sha256": hashlib.sha256(case["text"].encode("utf-8")).hexdigest(),
                        "engine": "sentence-transformers", "engine_version": backend.version,
                        "model_id": MODEL_ID, "model_revision": REVISION, "files_sha256": hashes,
                        "device": "cpu", "pooling": "cls", "normalization": "l2",
                        "prompt": "", "dimension": DIMENSION, "embedding": values})
    return records


def run(directory, manifest, input_path, output_path, backend_factory=Backend):
    cases = read_cases(input_path)
    hashes = verify_snapshot(directory, manifest)
    records = produce(cases, hashes, backend_factory(directory))
    output_path = Path(output_path)
    fd, temporary = tempfile.mkstemp(prefix=".embedding-", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        os.link(temporary, output_path)
    finally:
        os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    args = parser.parse_args()
    try:
        run(args.model_dir, args.manifest, args.input_jsonl, args.output_jsonl)
    except (Refusal, OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        print(f"embedding adapter refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
