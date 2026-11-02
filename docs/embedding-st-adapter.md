# Offline sentence-transformers embedding adapter

This optional correctness-data adapter reads a locally provisioned
`BAAI/bge-small-en-v1.5` snapshot at revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. It loads sentence-transformers
lazily, uses CPU, CLS pooling and the model's Normalize module, and refuses
inputs longer than 512 tokens. It does not download a model, measure timing,
or produce Lattice embeddings.

Install a compatible sentence-transformers environment separately and obtain the
exact snapshot independently. Create a UTF-8 JSON manifest with keys
`schema_version` (1), `model_id`, `revision`, and `files`. The `files` object
must map each path named by `FILES` in `embedding_st_adapter.py` to its lowercase
SHA-256 digest. These hashes identify the local files and detect later changes;
they do not independently authenticate a caller-supplied manifest or prove
where the snapshot came from. Verify its origin separately before comparing
results. Keep the snapshot unchanged while the adapter runs.

Input is UTF-8 JSONL with exactly `id` and `text` string fields per line and
unique nonempty IDs. For example:

```json
{ "id": "case-1", "text": "A short sentence." }
```

Run `python3 scripts/embedding_st_adapter.py MODEL_DIR MANIFEST INPUT_JSONL OUTPUT_JSONL`.
The destination must not already exist. Every output row identifies the case,
input text hash, local model file hashes, model revision, runtime version,
CPU/CLS/L2 settings and a 384-element embedding. No raw input text or timing is
written. Outputs are published only after every case passes validation.

The schema is specific to embedding correctness and is not the decode-benchmark
schema. Actual tokenization and numerical parity with Lattice require a paired
producer and host validation; this adapter alone does not establish either.
