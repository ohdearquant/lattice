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

## Recorded measurements

These local corpora are not committed. All use the classifier and feedback settings
above, with 160 training and 50 test prompts per label. Bag-of-words (BoW) controls
use train-only vocabulary and logistic weights on normalized token counts.

Corpus A pairs public-history commit-message prompts containing wrapped diffs with
curated request prompts, preserving source split boundaries. Its four-word opening
census found 48 request openings and one commit opening across 210 commit rows.
The `repo:` prefix rule scores 420/420 across both splits: this corpus is degenerate.
Train-only BoW scores 100/100; real / shuffled accuracy is 100/100 / 48/100, with
majority floor .50. No embedding-value claim follows from this template-driven ceiling.

Corpus B pairs memory requests with task-management requests, assigning whole verbs
to disjoint train/test splits: remember is held out for memory, complete for tasks.
Family-qualified verb tokens and bare family names were stripped to remove explicit
label leakage; other vocabulary remains. The unstripped run was confounded by label
tokens and is not reported here. The stripped census found 17 / 29 four-word openings
by label, with 4 shared openings covering 306/420 rows. The train-only opening-majority
rule scores 47/100; train-only BoW scores 50/100. Real / shuffled accuracy is
88/100 / 52/100, with majority floor .50.

Corpus C pairs brain requests with session requests using the same stripping protocol
to remove explicit family labels. Whole verbs are held out between train and test:
register adapter and unbind for brain, export for session. Its census found 31 / 22
four-word openings, with 2 shared openings covering 131 / 101 rows. The train-only
opening-majority rule scores 64/100; train-only BoW scores 76/100. Effective held-out
templates are 6 brain and 7 session normalized skeletons, masking quoted literals,
UUIDs, numbers and hashes while retaining export formats; these are operational counts,
not proof of independence. Real / shuffled accuracy is 99/100 / 47/100, floor 50/100.
All BoW misses are 24 of the 25 held-out register adapter rows. Their content words
(register, adapter, weights, hash, revision, base, supplied, against) never occur in
training prompts; a separate logistic model on embeddings places 24 of the 25 correctly.

All corpora reversed selection to the initially unselected adapter after 16 feedback
events, with replay accuracy 1.0 and an unchanged no-update counterfactual. Both
opening-majority rules resolve ties and unseen openings to label zero.

Limits: generated phrasing and residual vocabulary, one split and one seed per corpus.
Test rows are template-generated, so row count overstates the evidence, including for
A and B; effective test size is the template count, not the row count. Opening counts
are not template counts. Single-call timings are observational, without a prefill
comparison. No response quality was measured, and these results support no routing decision.
