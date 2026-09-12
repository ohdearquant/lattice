# Prompt embedding routing experiment

This two-stage experiment measures whether prompt embeddings separate two adapter
domains. It does not measure response quality or prove either adapter is useful.
No dependency on `lattice-embed` is added to the training or inference crates.

Prepare JSONL records containing `prompt` (string), `label` (0 or 1), and `split`
(`train` or `test`). Use independently held-out prompts, at least 30 records per
class per split, and equal class counts in the test split. Assign splits before
embedding; keep related sources in the same split and audit near duplicates.
The examples reject case-insensitive exact duplicates across the entire dataset.
Do not include completions or provenance in the embedded prompt.

```sh
cargo run --release -p lattice-embed --example prompt_router_vectors -- prompts.jsonl vectors.json
cargo run --release -p lattice-tune --features mixture,inference-hook,serde --example prompt_router -- vectors.json
```

The first stage uses `NativeEmbeddingService::embed_query`, BGE-small-en-v1.5,
384 dimensions, no MRL truncation. It exports embeddings and per-call wall times.
Its first call includes lazy model initialization; subsequent calls use the
loaded model. There is no embedding cache wrapper. Model files must already be
available when offline mode is enabled.

The second stage uses a 384 → 16 tanh → 2 softmax FANN classifier, initialized
with seed 42, trained for 100 epochs with seed 42, learning rate 0.03, batch 16,
and the remaining `TrainingConfig` defaults. The shuffled arm permutes labels
independently within each split with fixed seeds, preserving class counts and
using identical embeddings and training settings. Real accuracy must exceed the
held-out majority-class rate. Shuffled accuracy must lie within three binomial
standard errors of 0.5; this interval requires a balanced held-out split. A small
corpus or one split/seed is limited evidence, even when these checks pass.

The independent feedback arm requires linear output logits for RLOO, retaining
the same hidden shape and initialization seed. It routes a held-out vector,
records 16 positive events for the initially unselected adapter, calls
`update_router`, and routes the identical vector and pool again. It requires a
changed selection, 16 consumed events, replay accuracy 1.0, and an unchanged
no-update counterfactual. This is a controlled preference reversal, not a replay
of measured adapter response quality.

An optional final argument selects `real`, `shuffled`, `balance`, `loop`, or `cost`
for isolated checks. Cost output is one warm embedding call and one warm gate
forward, plus the cold embedding call. These are observational timings, without
prefill comparison or statistical performance guarantees. Follow the repository's
machine-lock policy when running measurements.

Report real and shuffled accuracy together with class balance, source provenance,
seed policy, model/dimension, and a lexical baseline when domain phrasing might
explain separation. Preserve failures: a failed real arm is a valid negative
result, while a successful shuffled classifier invalidates the interpretation.
