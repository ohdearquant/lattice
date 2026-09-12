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

Run 2 uses the corrected loader from #1546: prompt and completion tokenize
separately, and the model EOS token is appended. The split, training flags,
and scoring rule remain fixed.

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

The control command is `uv run --no-project python3 scripts/microlora/w6_score.py --self-test --input INPUT_DIR --out SCORE_DIR --validator VALIDATOR --generations OUTPUT_DIR`. Controls completed before generation. First-line extraction is `output.split("\n", 1)[0].strip()` and records whether any newline was emitted. Family acceptance requires a nonempty set of qualified calls outside double-quoted literals, all in the prompt's family. C2's 80 mutated completions all parse and are rejected by argument validation.

Base has no adapter. Single-M applies the memory adapter to all prompts; single-G applies the task adapter to all prompts. Routed follows the gate, oracle follows the true family, and random uses Python `Random(20260912).choice` over M/G in held-out order. Selection arms reuse deterministic M/G outputs.

| Control                            | Expected                                                            | Observed                                            |
| ---------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------- |
| C0 scorer self-test                | PASS, FAIL, REFUSED reachable; capture `625b8392f752d63d`, 98 verbs | Matched                                             |
| C1 gold callable-on-family         | 80/80                                                               | 80/80                                               |
| C2 wrong first parameter NOT PASS  | 80/80                                                               | 80/80; all parse, all fail argument-name validation |
| C3 wrong family rejected           | 80/80                                                               | 80/80                                               |
| C4 prefixed prose parser rejection | 80/80                                                               | 80/80                                               |

| arm      | callable overall | memory        | gtd            | exact overall | memory        | gtd           |
| -------- | ---------------- | ------------- | -------------- | ------------- | ------------- | ------------- |
| base     | 0/80 (0.0%)      | 0/40 (0.0%)   | 0/40 (0.0%)    | 0/80 (0.0%)   | 0/40 (0.0%)   | 0/40 (0.0%)   |
| single-M | 39/80 (48.8%)    | 39/40 (97.5%) | 0/40 (0.0%)    | 39/80 (48.8%) | 39/40 (97.5%) | 0/40 (0.0%)   |
| single-G | 40/80 (50.0%)    | 0/40 (0.0%)   | 40/40 (100.0%) | 38/80 (47.5%) | 0/40 (0.0%)   | 38/40 (95.0%) |
| routed   | 79/80 (98.8%)    | 39/40 (97.5%) | 40/40 (100.0%) | 77/80 (96.2%) | 39/40 (97.5%) | 38/40 (95.0%) |
| oracle   | 79/80 (98.8%)    | 39/40 (97.5%) | 40/40 (100.0%) | 77/80 (96.2%) | 39/40 (97.5%) | 38/40 (95.0%) |
| random   | 42/80 (52.5%)    | 26/40 (65.0%) | 16/40 (40.0%)  | 41/80 (51.2%) | 26/40 (65.0%) | 15/40 (37.5%) |

Routing accuracy is 80/80; shuffled-label accuracy is 47/80, within its predefined chance interval. All 12 primary-arm duplicate outputs match byte-for-byte.

memory: train NLL 2.2866 → 0.0076; held-out NLL 1.8271 → 0.0034 (99.814% reduction).
gtd: train NLL 1.7828 → 0.0032; held-out NLL 2.2718 → 0.0028 (99.877% reduction).

Under the fixed rule, routing is **USEFUL**: routed meets the best-single-adapter minus-five-points bound overall and strictly exceeds base on each family.

S1 is supplementary and outside the decision rule, registered before the original adapter outputs existed. It uses the base model with `<think>\n\n</think>\n\n` appended to each verbatim prompt, with the same decode flags. Four separate-process duplicates also match.

| Supplementary arm | First-line parser accepted | First-line callable | First-non-blank-line parser accepted | First-non-blank-line callable | Reopened `<think>` |
| ----------------- | -------------------------- | ------------------- | ------------------------------------ | ----------------------------- | ------------------ |
| S1                | 0/80                       | 0/80                | 0/80                                 | 0/80                          | 0/80               |

**Why base reads 0.** In run 1, the raw model opened `<think>` on 70/80 prompts and a code fence on 9/80; its first line was blank on 80/80. With the think block closed (S1), its outputs include prose, SQL, HTTP, and unrelated code, and still parse on 0/80 under both line rules. These controls support attributing the floor to the model not knowing the request DSL, beyond the leading blank line and open thinking block.

**Run 1 (pre-fix).** First-line callable results were routed 28/80 (35.0%), single-M 8/80 (10.0%), single-G 20/80 (25.0%), base 0/80 (0.0%). The old loader merged the prompt/completion boundary on 52/80 held-out rows; all 31 leading-newline memory-adapter outputs across the 80 prompts belonged to merged rows. It also omitted EOS, and memory-adapter outputs on their own family contained glued repeats on 16/40 prompts (issue #1545, fixed by #1546).

Timings were recorded on an Apple-silicon desktop and are informational, never a benchmark verdict: the relay holds the bench window shared. Every timing below carries its producing phase's start/end conditions. The trainer's `done` duration covers the step loop and scoring; the conditions receipt records full phase time. The driver uses `uv run --no-project python3`.

train-M:

```text
1259 completion positions across 48 samples in 84.4s (10 threads)
held-out: 269 completion positions across 8 valid samples in 17.8s (10 threads)
baseline scoring: train pass 109.4s, held-out pass 22.7s
step loop: 96 steps in 328.9s (3.43s/step), in-loop scoring 132.1s over 1 point(s), epilogue re-scoring 0.0s
=== done: train 2.2866→0.0076 (-2.2790)  |  held-out 1.8271→0.0034 (-1.8237)  in 461.0s ===
```

```json
{
  "phase": "train-M",
  "start": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.10 1.17 1.31 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  },
  "window": "shared; informational timing only",
  "ok": true,
  "elapsed_seconds": 699.4037639590097,
  "end": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.66 2.00 2.04 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  }
}
```

train-G:

```text
1762 completion positions across 48 samples in 113.1s (10 threads)
held-out: 246 completion positions across 8 valid samples in 18.6s (10 threads)
baseline scoring: train pass 152.4s, held-out pass 20.9s
step loop: 96 steps in 446.0s (4.65s/step), in-loop scoring 169.1s over 1 point(s), epilogue re-scoring 0.0s
=== done: train 1.7828→0.0032 (-1.7796)  |  held-out 2.2718→0.0028 (-2.2690)  in 615.1s ===
```

```json
{
  "phase": "train-G",
  "start": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 3.74 2.56 2.24 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  },
  "window": "shared; informational timing only",
  "ok": true,
  "elapsed_seconds": 930.964566583978,
  "end": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.78 2.01 2.35 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  }
}
```

gen-base:

```json
{
  "phase": "gen-base",
  "start": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 2.09 2.15 2.38 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  },
  "window": "shared; informational timing only",
  "ok": true,
  "elapsed_seconds": 338.5525895419996,
  "end": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.90 1.92 2.18 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  }
}
```

gen-M:

```json
{
  "phase": "gen-M",
  "start": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.62 1.85 2.15 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  },
  "window": "shared; informational timing only",
  "ok": true,
  "elapsed_seconds": 210.76446954201674,
  "end": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.75 1.79 2.06 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  }
}
```

gen-G:

```json
{
  "phase": "gen-G",
  "start": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.66 1.77 2.04 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  },
  "window": "shared; informational timing only",
  "ok": true,
  "elapsed_seconds": 203.1446277089999,
  "end": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.86 1.74 1.96 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  }
}
```

route:

```text
warm_embedding_seconds=0.004635875 single_gate_forward_seconds=0.000000416 cold_embedding_seconds=0.303303792
```

```json
{
  "phase": "route",
  "start": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.76 1.72 1.95 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  },
  "window": "shared; informational timing only",
  "ok": true,
  "elapsed_seconds": 1.7448832500376739,
  "end": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.76 1.72 1.95 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  }
}
```

gen-S1:

```json
{
  "phase": "gen-S1",
  "start": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.56 1.68 1.92 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  },
  "window": "shared; informational timing only",
  "ok": true,
  "elapsed_seconds": 332.17996287497226,
  "end": {
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.loadavg": "{ 1.74 1.66 1.82 }",
    "hw.model": "Mac16,10",
    "hw.memsize": "17179869184",
    "machdep.cpu.brand_string": "Apple M4"
  }
}
```
