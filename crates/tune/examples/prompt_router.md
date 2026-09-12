# Prompt embedding routing experiment

The separation arms measure whether prompt embeddings separate two adapter
domains. Separation alone does not establish response quality or adapter usefulness.
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

An optional final argument selects `real`, `shuffled`, `balance`, `loop`, `cost`,
or `decisions` for isolated checks. The `decisions` arm uses the same real-label
fit and seed 42, then emits `decision idx=<i> label=<l> predicted=<p>` for every
test row in input order, followed by the accuracy line. It is explicitly selected
and is excluded from the default `all` arm. Cost output is one warm embedding call and one warm gate
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

### Response quality under routing

The fixed construction uses only the training split of a captured request corpus
(SHA-256 prefix `6ff089c0e380213e`, 2,806 single-line completions). Its separate
validation/test splits hold out entire verbs and answer a different generalization
question. Single-family extraction ignores calls inside double-quoted literals:
257 memory rows and 204 task rows, with zero case-insensitive prompt duplicates
dropped. With seed 20260912, proportional largest-remainder allocation reserves
40 prompts per family, at least one per verb. Verb groups are shuffled in lexical
order, followed by each family's remaining and held-out rows. The held-out verb
counts are memory recall/feedback/prune 24/14/2 and tasks/transition/next 16/16/8.

Each adapter uses the first 48 shuffled training rows. The first eight held-out
rows supply the trainer's NLL read-out without weight updates and remain among
the 40 generation-scored prompts. Exact prompt overlap is zero; completion overlap
under different prompts is 4/40 memory and 5/40 tasks, limiting independence.
The Qwen3.5-0.8B trainer flags are `--seq-len 256 --steps 96 --max-train 48
--max-valid 8 --first-layer 19 --rank 8 --alpha 16 --lr 1e-3 --log-every 96
--seed 4277009102`. The generator uses each verbatim prompt with
`--max-tokens 64 --temperature 0 --seed 1`.

The router uses all 96 adapter-training prompts and the same 80 held-out prompts,
unstripped, with memory label 0 and task label 1. Both splits have at least 30
rows per label, and held-out labels balance 40/40. The `real`, `shuffled`, `cost`,
and `decisions` arms use BGE-small-en-v1.5 vectors. Routed, oracle, and seed-20260912
coin selections reuse the two adapter outputs only after the four duplicate
prompts per generation arm match byte-for-byte in separate processes.

Callable-on-family requires the first generated line to pass the real request
parser and the captured argument scorer, with every qualified verb belonging to
the prompt's family. Exact match compares canonical requests. All 100 schema JSON
files, including underscore files, contribute to capture `625b8392f752d63d`
(98 verbs). The parser source SHA-256 is
`62992e89258640e2c76752961eda1af42bb0cc097fb713341182b3e9d2d8a915`.
The control command is `uv run --no-project python3 scripts/microlora/w6_score.py
--self-test`, after preparing the input directory and validator. All controls were
completed before generation. Family acceptance requires a nonempty set of qualified
calls outside double-quoted literals, all belonging to the prompt's family.

| Control                            | Expected                                                            | Observed                                                        |
| ---------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------- |
| C0 scorer self-test                | PASS, FAIL, REFUSED reachable; capture `625b8392f752d63d`, 98 verbs | Matched                                                         |
| C1 gold callable-on-family         | 80/80                                                               | 80/80                                                           |
| C2 wrong first parameter NOT PASS  | 80/80                                                               | 80/80; all parse successfully and fail argument-name validation |
| C3 wrong family rejected           | 80/80                                                               | 80/80                                                           |
| C4 prefixed prose parser rejection | 80/80                                                               | 80/80                                                           |

Base has no adapter. Single-M applies the memory adapter to every prompt;
single-G applies the task adapter to every prompt. Routed follows the gate,
oracle follows the true family, and random uses Python `Random(20260912).choice`
over M/G in held-out input order.

| Arm      | Callable overall | Callable memory | Callable gtd  | Exact overall | Exact memory | Exact gtd     |
| -------- | ---------------- | --------------- | ------------- | ------------- | ------------ | ------------- |
| base     | 0/80 (0.0%)      | 0/40 (0.0%)     | 0/40 (0.0%)   | 0/80 (0.0%)   | 0/40 (0.0%)  | 0/40 (0.0%)   |
| single-M | 8/80 (10.0%)     | 8/40 (20.0%)    | 0/40 (0.0%)   | 8/80 (10.0%)  | 8/40 (20.0%) | 0/40 (0.0%)   |
| single-G | 20/80 (25.0%)    | 0/40 (0.0%)     | 20/40 (50.0%) | 17/80 (21.2%) | 0/40 (0.0%)  | 17/40 (42.5%) |
| routed   | 28/80 (35.0%)    | 8/40 (20.0%)    | 20/40 (50.0%) | 25/80 (31.2%) | 8/40 (20.0%) | 17/40 (42.5%) |
| oracle   | 28/80 (35.0%)    | 8/40 (20.0%)    | 20/40 (50.0%) | 25/80 (31.2%) | 8/40 (20.0%) | 17/40 (42.5%) |
| random   | 12/80 (15.0%)    | 5/40 (12.5%)    | 7/40 (17.5%)  | 11/80 (13.8%) | 5/40 (12.5%) | 6/40 (15.0%)  |

Routing accuracy is 80/80; shuffled-label accuracy is 47/80, within its
predefined chance interval. All 12 duplicate outputs match byte-for-byte.
Newline presence is retained per output in the scoring artifacts.
Under the fixed rule, routing is **USEFUL**: it meets the overall best-single
minus-five-points bound and strictly exceeds base on each family.

Memory train NLL: 1.7006 → 0.0077; held-out NLL: 1.2385 → 0.0025.
Task train NLL: 1.4251 → 0.0036; held-out NLL: 1.7972 → 0.0057.
Both clear the fixed 50% held-out NLL-reduction threshold.

Timings were recorded on an Apple-silicon desktop. The shared bench window makes
these timings informational, never a benchmark verdict. Recorded phases used a
direct Python interpreter entry; the packaged shell wrapper now invokes the same
standard-library driver through `uv run --no-project python3`. The trainer's `done`
duration covers the step loop and its scoring, not the full phase; each conditions
receipt records the full phase elapsed time. Cache and scoring passes are shown
separately. Each timing below belongs to its named phase's start/end conditions.

train-M, with its conditions below:

```text
1182 completion positions across 48 samples in 83.3s (10 threads)
held-out: 255 completion positions across 8 valid samples in 17.5s (10 threads)
baseline scoring: train pass 105.6s, held-out pass 22.3s
step loop: 96 steps in 318.0s (3.31s/step), in-loop scoring 127.6s over 1 point(s), epilogue re-scoring 0.0s
=== done: train 1.7006→0.0077 (-1.6929)  |  held-out 1.2385→0.0025 (-1.2360)  in 445.6s ===
```

train-G, with its conditions below:

```text
1688 completion positions across 48 samples in 110.9s (10 threads)
held-out: 232 completion positions across 8 valid samples in 16.9s (10 threads)
baseline scoring: train pass 146.9s, held-out pass 20.3s
step loop: 96 steps in 437.0s (4.55s/step), in-loop scoring 164.0s over 1 point(s), epilogue re-scoring 0.0s
=== done: train 1.4251→0.0036 (-1.4215)  |  held-out 1.7972→0.0057 (-1.7914)  in 601.1s ===
```

Route cost, with the route conditions below:

```text
warm_embedding_seconds=0.00464475 single_gate_forward_seconds=0.000000458 cold_embedding_seconds=0.302235958
```

```json
{
  "phase": "train-M",
  "start": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.68 1.78 1.85 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  },
  "window": "shared; informational timing only",
  "ok": true,
  "elapsed_seconds": 680.5020301659999,
  "end": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.91 2.57 2.85 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  }
}
```

```json
{
  "phase": "train-G",
  "start": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.55 2.34 2.75 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  },
  "window": "shared; informational timing only",
  "ok": true,
  "elapsed_seconds": 906.7692175840001,
  "end": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.62 1.99 2.64 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  }
}
```

```json
{
  "phase": "gen-base",
  "start": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 2.74 2.52 2.74 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  },
  "window": "shared; informational timing only",
  "ok": true,
  "elapsed_seconds": 333.334467208,
  "end": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.45 1.83 2.33 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  }
}
```

```json
{
  "phase": "gen-M",
  "start": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.21 1.73 2.28 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  },
  "window": "shared; informational timing only",
  "ok": true,
  "elapsed_seconds": 306.585580333,
  "end": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.74 1.70 2.08 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  }
}
```

```json
{
  "phase": "gen-G",
  "start": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.46 1.62 2.04 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  },
  "window": "shared; informational timing only",
  "ok": true,
  "elapsed_seconds": 321.186848292,
  "end": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.87 1.70 1.93 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  }
}
```

```json
{
  "phase": "route",
  "start": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.25 1.54 1.84 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  },
  "window": "shared; informational timing only",
  "ok": true,
  "elapsed_seconds": 1.751243125,
  "end": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.25 1.54 1.84 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  }
}
```
