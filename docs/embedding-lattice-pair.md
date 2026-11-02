# Offline BGE embedding pairing

This correctness-only stage compares the sentence-transformers records from
[the reference adapter](embedding-st-adapter.md) with CPU `BertModel` loaded
once from the same local BGE-small snapshot. It uses CLS pooling, L2-normalized
384-dimensional output and no prompt. It does not download, time, or compare
HTTP or embedding-service paths.

Build `embedding_pair_producer` as an example of `lattice-model-bert` on the
host, generate the reference JSONL using the existing adapter, then run:

```sh
python3 scripts/embedding_lattice_pair.py MODEL_DIR MANIFEST CASES.jsonl REFERENCE.jsonl PRODUCER OUTPUT.jsonl --lattice-revision FULL_GIT_SHA --batch-size 8
```

`PRODUCER` is the path to the built example; `FULL_GIT_SHA` is a caller assertion
about its source, not binary attestation. No output file may already exist. All
records, model-file digests, model configuration, cases, token preflights and
embedding comparisons must pass before publication. Each row contains the
Lattice vector and cosine and maximum absolute difference against its reference;
initial limits are cosine at least 0.99999 and maximum absolute difference at most
0.005. The output holds no input text or timing.

For each batch the producer loads its own tokenizer and reports token IDs from a
separate preflight using the same single or batch tokenizer API that the later
BERT call uses. The Python side independently retokenizes with the currently
loaded sentence-transformers tokenizer, with special tokens and truncation off.
Matching preflights are **not** captured tokens from either actual encode call.
Run singleton and mixed-length batches on the real model before claiming parity.
The reference adapter and pairing runner should use the same pinned Python
environment. File hashes and the revision label are caller-supplied evidence,
not proof of upstream origin; keep the snapshot immutable during both runs,
and verify its provenance and any additional framework-consumed files separately.
Hashes are checked again before output publication, which detects changed final
contents but does not prevent changes and restoration while a model is loading.
The Lattice SHA is also a caller assertion. No real-model or performance result
is included in this stage.
