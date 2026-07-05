# CLI Tool Walkthroughs

Several binaries in `lattice-inference` ship with no walkthrough beyond `--help` (or, for the
ones with no `--help` at all, beyond reading the source). This document works through
`quantize_q4`, `chat_metal`'s full flag surface, `ppl_metal`, and `bench_lora_mixture`: what each
one is for, what it needs on disk, and a verified command sequence with real output.

Every command below was actually run against a local `qwen3.5-0.8b` checkpoint (and its Q4
variants) in this repository: the flags, the failure messages, and the output shapes are copied
from that terminal, not reconstructed from memory. Metal-GPU-touching runs went through the
machine-wide `/tmp/lion-metal-gpu-test.lock` advisory flock (see `CLAUDE.md`) so they never
overlapped another process's GPU work.

**A note on the numbers below.** Every `tok/s`, latency, timing, and perplexity figure in this
document is a sample measurement from a single local run on one Apple Silicon machine, not an
official or reproducible benchmark result. Treat them as "this ran and produced output of this
shape," not as a performance claim. For real throughput/perplexity comparisons, use the
project's own benchmark tooling: `make bench-compare` (see the root `CLAUDE.md`) and
`docs/bench_results/perplexity.tsv`.

For the deeper Q4/QuaRot quality story (why quantize, what perplexity delta to expect, the
QuaRot rotation), see [`docs/q4-quantization.md`](q4-quantization.md). This document stays
scoped to "how do I invoke the tool."

## Conventions used below

The commands in this document use three portable variables instead of one author's actual home
directory paths. Set them once per shell session before running anything:

```bash
MODEL_DIR="${LATTICE_MODEL_CACHE:-$HOME/.lattice/models}/qwen3.5-0.8b"
SCRATCH_DIR="${TMPDIR:-/tmp}/lattice-cli-tools"
Q4_DIR="$SCRATCH_DIR/qwen3.5-0.8b-q4"
mkdir -p "$SCRATCH_DIR"
```

`$MODEL_DIR` should point at a local `qwen3.5-0.8b` BF16/F16 checkpoint directory (weights,
`config.json`, `tokenizer.json`); `$SCRATCH_DIR`/`$Q4_DIR` are scratch locations for
`quantize_q4` output and any sample corpus file, safe to delete when you're done. In pasted
command output below, `<model-dir>` and `<q4-dir>` stand in for whatever `$MODEL_DIR` and
`$Q4_DIR` resolved to on the machine the command was actually run on.

## Which binary do I want?

| Binary               | Purpose                                                         | Platform            |
| -------------------- | --------------------------------------------------------------- | ------------------- |
| `quantize_q4`        | Convert a BF16/F16 safetensors checkpoint to Q4_0               | any                 |
| `chat_metal`         | Interactive chat / JSON one-shot / JSON serve loop on Metal GPU | macOS + `metal-gpu` |
| `ppl_metal`          | Perplexity scoring of a corpus on Metal GPU                     | macOS + `metal-gpu` |
| `bench_lora_mixture` | CPU blend-latency + GPU decode benchmark for LoRA mixtures      | macOS + `metal-gpu` |

`quantize_quarot`, `eval_perplexity`, `qwen35_generate`, and the unified `lattice` binary are
covered elsewhere (`docs/q4-quantization.md`, `docs/inference-usage.md`, `README.md`).

---

## `quantize_q4`

### What it does

Streams a sharded BF16/F16/F32 safetensors checkpoint tensor-by-tensor and writes one file per
tensor into an output directory: large weight matrices (`*_proj.weight`, MLP projections,
`embed_tokens`, `lm_head`) become Q4_0 `.q4` files, everything else (norms, biases, small
scalars) is kept at F16 as a minimal `.f16` file. It also writes a `quantize_index.json`
manifest recording, per tensor, its output filename, whether it was quantized, its original
shape, and element count. This is the layout `MetalQwen35State::from_q4_dir` expects.

### When you'd use it

Producing a smaller, faster-to-load 4-bit checkpoint for the Metal decode path, without the
QuaRot Hadamard rotation (`quantize_quarot` is the rotated variant: lower quantization error, at
the cost of an extra preprocessing step; see `docs/q4-quantization.md`).

### Prerequisites

- A local safetensors checkpoint directory (`model.safetensors` or a sharded
  `model.safetensors.index.json`).
- Build with no special features: `cargo build --release -p lattice-inference --bin quantize_q4`.
- An output directory: always a scratch location, never overwrite a model directory in place.

### Verified command sequence

```bash
cargo build --release -p lattice-inference --bin quantize_q4
```

No arguments prints usage and exits 1 (there is no `--help` flag, but any missing/unknown
argument routes through the same usage message):

```
$ ./target/release/quantize_q4
--model-dir is required
Usage: quantize_q4 --model-dir <DIR> --output-dir <DIR> [--dry-run]

  --model-dir   directory containing model.safetensors[.index.json]
  --output-dir  directory to write .q4 and index files
  --dry-run     read tensors but skip writing output
```

Real run against `qwen3.5-0.8b`, output into a scratch directory:

```bash
./target/release/quantize_q4 \
  --model-dir "$MODEL_DIR" \
  --output-dir "$Q4_DIR"
```

```
=== quantize_q4: SafeTensors → Q4_0 ===
Model dir:  <model-dir>
Output dir: <q4-dir>
Tensors:    488
  [1/488] Q4_0  model.language_model.embed_tokens.weight  shape=[248320, 1024]  969.2MB→303.0MB  0.61s
  ...
  [488/488] F16   mtp.pre_fc_norm_hidden.weight  shape=[1024]  0.0MB  dtype=BF16  0.000s
Index written: <q4-dir>/quantize_index.json

=== Summary ===
Tensors processed: 488
  Quantized (Q4_0): 271
  Kept (F16):       217
Input size:   1.63 GB
Output size:  0.51 GB
Ratio:        3.20x  (31.3%)
Total time:   17.0s
```

`--dry-run` runs the same read/classify pass but writes nothing (useful for a fast sanity check
of tensor classification before committing to a full write).

### Common failure modes

- **No `tokenizer.json` or `config.json` in the output directory.** `quantize_q4` only writes
  weight files and the index: it does **not** copy the tokenizer or model config from the
  source directory. A downstream loader (`chat_metal`, `lattice_serve`, `eval_perplexity
  --q4-dir`) pointed straight at a fresh `quantize_q4` output directory will fail:

  ```
  chat_metal: failed to load tokenizer from <out-dir>: Tokenizer error: failed to read
  <out-dir>/tokenizer.json: No such file or directory (os error 2)
  ```

  Fix: either copy `tokenizer.json` (and `config.json`) from the source model directory into the
  output directory, or point the consuming binary's `--tokenizer-dir`/`--tokenizer` flag back at
  the source directory (see `chat_metal` below; not every binary has this override, see the
  `bench_lora_mixture` section).

- **Missing `config.json` silently loads the wrong architecture preset.** With no `config.json`
  in the output directory, `chat_metal` falls back to a compiled-in preset, and it picks
  `qwen36_27b`, not the 0.8b config, regardless of what was actually quantized:

  ```
  [chat_metal] No config.json; using qwen36_27b preset
  [chat_metal] Loading Q4 model...
  chat_metal: failed to initialize Metal from Q4 dir: from_q4_dir: tie_word_embeddings=false but
  lm_head.q4 is missing at <out-dir>/lm_head_weight.q4; the runtime requires the materialized
  lm_head matrix (ADR-044 step 3c) and must not fall back to embed_tokens, which would yield
  wrong logits and a misleading perplexity report
  ```

  This is a real correctness trap, not just an inconvenience: a `qwen3.5-0.8b` Q4 output loaded
  under the `qwen36_27b` preset has the wrong `tie_word_embeddings` assumption baked in, and the
  failure it produces (a missing-file error) is a lucky fail-closed outcome, not a guarantee.
  Always copy `config.json` alongside a fresh `quantize_q4` output.

  Fix: copy `config.json` from the source directory before loading:

  ```bash
  cp "$MODEL_DIR/config.json" "$Q4_DIR/"
  ```

- **Never point `--output-dir` at an existing model directory.** There is no overwrite guard;
  it will happily interleave `.q4`/`.f16` files with whatever is already there.

---

## `chat_metal`

### What it does

Interactive Qwen3.5 chat on the Metal GPU path, with three modes: an interactive REPL (default,
no flags), a one-shot JSON mode (`--json --prompt "..."`, used by Lattice Studio for one process
per message), and a persistent JSON serve mode (`--json --serve`, one process kept warm, fed
newline-delimited JSON requests on stdin). All three share the same `@@lattice gen_token` event
format and the same underlying `MetalQwen35State` generation path.

### When you'd use it

Manually exercising the Metal decode path: checking a Q4 conversion actually produces sane
text, benchmarking real tok/s at the CLI, testing a LoRA adapter, or as the subprocess Lattice
Studio drives for its own chat UI.

### Prerequisites

- macOS 14+, Metal GPU.
- Build: `cargo build --release -p lattice-inference --bin chat_metal --features f16,metal-gpu`.
- A model directory: either a BF16/F16 safetensors checkpoint (`model.safetensors[.index.json]`,
  `config.json`, `tokenizer.json`) or a Q4 directory (`*.q4`/`*.f16` files + `quantize_index.json`,
  ideally also `config.json` and `tokenizer.json`; see the `quantize_q4` failure modes above for
  what happens when they're missing). `chat_metal` auto-detects Q4 vs BF16 by checking for
  `model.safetensors`/`.index.json` first, then falling back to "any `.q4` file present."
- GPU-lock discipline: acquire `/tmp/lion-metal-gpu-test.lock` before running (see top of this
  document).

### Full flag surface (enumerated from `crates/inference/src/bin/chat_metal.rs`, not memory)

| Flag                                          | Default                                                      | Notes                                                                                                                                 |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `--model <PATH-OR-NAME>`                      | `LATTICE_MODEL_DIR` env, else `~/.lattice/models/qwen3.5-2b` | Absolute/`~`-prefixed path used as-is; a bare name (one path component) resolves against `LATTICE_MODEL_CACHE` or `~/.lattice/models` |
| `--model-dir <DIR>`                           | none                                                         | Takes precedence over `--model` if both given                                                                                         |
| `--tokenizer-dir <DIR>` / `--tokenizer <DIR>` | model dir                                                    | Also settable via `LATTICE_TOKENIZER_DIR` env; needed when the model dir has no `tokenizer.json` of its own                           |
| `--prompt <TEXT>`                             | none                                                         | Required for one-shot `--json` mode; ignored in REPL/`--serve` mode                                                                   |
| `--max-tokens <N>`                            | 512                                                          |                                                                                                                                       |
| `--seed <N>`                                  | none (random)                                                |                                                                                                                                       |
| `--temperature <F>`                           | 0.7                                                          |                                                                                                                                       |
| `--top-k <N>`                                 | 50                                                           |                                                                                                                                       |
| `--top-p <F>`                                 | 0.9                                                          |                                                                                                                                       |
| `--repetition-penalty <F>`                    | 1.1                                                          |                                                                                                                                       |
| `--reasoning-budget <N>`                      | none                                                         | Only applied when `N > 0`                                                                                                             |
| `--lora <PATH>`                               | none                                                         | PEFT- or MLX-format `.safetensors` adapter; relative paths resolve against CWD                                                        |
| `--json`                                      | off                                                          | Emits `@@lattice gen_token`/`done`/`ready`/`error` NDJSON events instead of REPL text                                                 |
| `--serve`                                     | off                                                          | Requires `--json`; reads NDJSON requests from stdin instead of `--prompt`, keeps the model warm                                       |

There is no `--grammar`/structured-output flag on this binary today: `GenerateConfig.grammar` is
hardcoded to `None` in both the CLI and serve-request paths (see ADR-068 for the grammar
wire-contract work in progress).

### Verified command sequence

Basic one-shot JSON mode against the BF16 checkpoint:

```bash
./target/release/chat_metal --model "$MODEL_DIR" \
  --prompt "Hi, tell me a fact about the ocean in one sentence." --max-tokens 24 --json
```

```
[chat_metal] Loading bf16 model from <model-dir>...
[chat_metal] Model loaded in 7.9s
[chat_metal] Initializing Metal GPU...
[chat_metal] Metal ready in 4.6s
@@lattice {"ev":"gen_token","token":"\n\n","token_id":271,"done":false}
@@lattice {"ev":"gen_token","token":"<think>","token_id":248068,"done":false}
...
@@lattice {"ev":"gen_token","token":"","done":true,"tok_s":56.7,"ttft_ms":115.4,"prompt_tokens":13,"gen_tokens":24,"total_ms":423}
[chat_metal] GPU Metal bf16: 13 prompt + 24 gen in 423ms = 56.7 tok/s
```

Note the `model_format` the binary prints (`bf16` here) always reflects which loader path was
actually taken; a `q4` tag in this same event stream format means the Q4 loader ran instead, not
the BF16 one.

Against a Q4 directory that lacks its own `tokenizer.json` (`--tokenizer-dir` pointed back at
the source checkpoint; this is the fix for the `quantize_q4` "no tokenizer.json" failure mode
above, applied live):

```bash
./target/release/chat_metal \
  --model "$Q4_DIR" \
  --tokenizer-dir "$MODEL_DIR" \
  --prompt "Hi, tell me a fact about the ocean in one sentence." --max-tokens 24 --json
```

```
[chat_metal] Detected Q4 model directory: <q4-dir>
[chat_metal] Loading Q4 model...
[chat_metal] Q4 model loaded in 3.9s
@@lattice {"ev":"gen_token","token":"\n\n","token_id":271,"done":false}
...
@@lattice {"ev":"gen_token","token":"","done":true,"tok_s":21.9,"ttft_ms":487.4,"prompt_tokens":13,"gen_tokens":24,"total_ms":1098}
[chat_metal] GPU Metal q4: 13 prompt + 24 gen in 1098ms = 21.9 tok/s
```

This confirms `--tokenizer-dir` is the correct override, and also shows the `model_format` tag
flipping from `bf16` (previous run) to `q4` (this run) for the same underlying prompt.

JSON serve mode: model loads once, then one NDJSON request line is read from stdin per turn:

```bash
printf '{"prompt":"Say hi in one word.","max_tokens":6}\n' \
  | ./target/release/chat_metal --model "$MODEL_DIR" --json --serve
```

```
[chat_metal] Loading bf16 model from <model-dir>...
[chat_metal] Model loaded in 11.0s
[chat_metal] Initializing Metal GPU...
[chat_metal] Metal ready in 11.1s
@@lattice {"ev":"ready"}
@@lattice {"ev":"gen_token","token":"\n\n","token_id":271,"done":false}
@@lattice {"ev":"gen_token","token":"<think>","token_id":248068,"done":false}
@@lattice {"ev":"gen_token","token":"\n\n","token_id":271,"done":false}
@@lattice {"ev":"gen_token","token":"</think>","token_id":248069,"done":false}
@@lattice {"ev":"gen_token","token":"\n\n","token_id":271,"done":false}
@@lattice {"ev":"gen_token","token":"Hello","token_id":9419,"done":false}
@@lattice {"ev":"gen_token","token":"","done":true,"tok_s":22.1,"ttft_ms":106.8,"prompt_tokens":6,"gen_tokens":6,"total_ms":272}
```

The `ready` event fires once the model is fully loaded and resident, before any request is read,
so the app can show an honest "loaded" state without sending a generation request first. The
process exits cleanly on stdin EOF or a broken output pipe.

Interactive REPL mode (no `--json`), one line piped in over stdin:

```bash
printf 'Say hi in one word.\n\n' \
  | ./target/release/chat_metal --model "$Q4_DIR" \
      --tokenizer-dir "$MODEL_DIR" --max-tokens 16
```

```
[chat_metal] Detected Q4 model directory: <q4-dir>
[chat_metal] Loading Q4 model...
[chat_metal] Q4 model loaded in 3.9s

=== GPU Metal q4 - Qwen3.5 Chat ===
Type your message. Empty line or Ctrl-D to quit.

> <think>

</think>

Hello!
[30 prompt + 6 gen in 230.6ms = 26.0 tok/s | GPU Metal q4]
>
```

An empty line (or Ctrl-D) ends the session, matching the docstring.

### Common failure modes

- Running without macOS + `metal-gpu` prints `chat_metal requires macOS + metal-gpu feature.`
  and exits 1: this binary has no CPU fallback path.
- `--json` one-shot mode without `--prompt` and without `--serve` returns
  `--json mode requires --prompt (or --serve for persistent serve mode)`.
- `--lora` was not run-verified in this pass: no local LoRA adapter `.safetensors` file was
  available in this environment. The flag and its PEFT/MLX-format auto-detection are
  source-derived from `load_lora_safetensors` (`crates/inference/src/bin/chat_metal.rs`), not
  run-verified here.
- Two concurrent `chat_metal --serve` processes hitting the same GPU without the
  `/tmp/lion-metal-gpu-test.lock` discipline will corrupt each other's timing and, per the
  fleet's own #628/#629 history, can inflate top-k logit margins under contention. Always take
  the lock for anything more than a single quick manual check.

---

## `ppl_metal`

### What it does

A thin, env-var-configured perplexity scorer on the Metal GPU path (Q8 activations + F16
`lm_head`, per its own log line). Tokenizes a corpus file, scores a sliding window (fixed at
`window=512`, `stride=256`), and prints `PPL`, `NLL`, token count, and elapsed time. There is no
CLI flag parsing at all: every input is an environment variable, and there is no `--help`.

### When you'd use it

A quick perplexity spot-check of a model or corpus on the Metal path without wiring up the
fuller `eval_perplexity` binary (which has proper flags including `--q4-dir`), useful for a
fast one-off number during a debugging session.

### Prerequisites

- macOS + Metal GPU. Build: `cargo build --release -p lattice-inference --bin ppl_metal --features f16,metal-gpu`.
- A BF16/F16 checkpoint directory with `tokenizer.json` (this binary always loads via
  `Qwen35Model::from_safetensors`, so it does not accept a Q4 directory: there is no Q4 branch
  in its ~45-line source).
- A plain-text corpus file (no default corpus ships in the repo, so you must supply one).

### Environment variables (enumerated from `crates/inference/src/bin/ppl_metal.rs`)

| Variable            | Default                          |
| ------------------- | -------------------------------- |
| `LATTICE_MODEL_DIR` | `~/.lattice/models/qwen3.5-0.8b` |
| `PPL_TOKENS`        | 2048                             |
| `CORPUS`            | `/tmp/wikitext2_test.txt`        |

### Verified command sequence

No corpus ships in the repo, so a short local text file was created for this run
(a few paragraphs of plain prose, ~330 words) at `$SCRATCH_DIR/sample_corpus.txt`:

```bash
LATTICE_MODEL_DIR="$MODEL_DIR" \
PPL_TOKENS=256 \
CORPUS="$SCRATCH_DIR/sample_corpus.txt" \
  ./target/release/ppl_metal
```

```
[ppl_metal] loading <model-dir>
[ppl_metal] scoring 256 tokens (Metal GPU, Q8 + f16 lm_head)
PPL:     10.9164
NLL:     2.390270
Tokens:  255
Time:    3.5s
```

(`Tokens: 255` vs the requested 256: the corpus only tokenized to 255 tokens after tokenization,
so the request is silently clamped to what's available; see the `.min(n_tokens)` call in the
source. As noted above, the `10.9164` figure is a one-machine sample measurement, not an
official perplexity number for this checkpoint.)

### Common failure modes

- `CORPUS` pointing at a missing file panics with `read corpus`: there is no graceful error
  path (every fallible step in `main` uses `.expect(...)`, not `?`).
- Pointing `LATTICE_MODEL_DIR` at a Q4 directory fails at `Qwen35Model::from_safetensors`, since
  this binary has no Q4 loading path; use `eval_perplexity --q4-dir` for Q4 perplexity instead
  (see `docs/q4-quantization.md`).
- No `tokenizer.json` in the model directory panics at the `BpeTokenizer::from_tokenizer_json`
  `.unwrap()` call.

---

## `bench_lora_mixture`

### What it does

Two benchmarks in one binary. First, a CPU-only synthetic benchmark of the latency to blend `k`
LoRA adapters of rank `r` into one combined rank-`k*r` adapter (`k ∈ {1,4,8}`, `r ∈ {1,2}`),
printed as `BLEND_BENCH` lines: this always runs, no model needed. Second, if
`LATTICE_MODEL_DIR` is set, an end-to-end GPU decode benchmark that actually loads a Q4 model,
applies each synthetic blended adapter, and measures decode tok/s, printed as `DECODE_BENCH`
lines.

### When you'd use it

Estimating the CPU overhead of the LoRA mixture path (PR #443's weighted CPU pre-blend) versus
the single-adapter path, and optionally getting a real decode-tok/s comparison across mixture
sizes on real hardware.

### Prerequisites

- macOS + Metal GPU for the `DECODE_BENCH` half (the binary refuses to build/run at all off
  macOS or without `metal-gpu`; there is no CPU-only build). Build:
  `cargo build --release -p lattice-inference --bin bench_lora_mixture --features f16,metal-gpu`.
- For the `DECODE_BENCH` half: a Q4 model directory that has **both** `config.json` and
  `tokenizer.json` alongside the `.q4`/`.f16` weight files (the tokenizer path is hardcoded to
  `<LATTICE_MODEL_DIR>/tokenizer.json`; there is no `--tokenizer-dir` override on this binary,
  unlike `chat_metal`). A bare `quantize_q4` output directory does not have either file by
  default (see the `quantize_q4` section above), so copy them in first.

### Environment variables

| Variable            | Default | Notes                                                |
| ------------------- | ------- | ---------------------------------------------------- |
| `LATTICE_MODEL_DIR` | unset   | Q4 model dir; enables the `DECODE_BENCH` half if set |
| `BENCH_WARMUP`      | 5       | warmup iterations for the blend bench                |
| `BENCH_ITERS`       | 20      | measured iterations for the blend bench              |
| `BENCH_NEW_TOKENS`  | 32      | tokens generated per `DECODE_BENCH` measurement      |

### Verified command sequence

Blend-only run (no `LATTICE_MODEL_DIR`):

```bash
BENCH_ITERS=3 BENCH_WARMUP=1 ./target/release/bench_lora_mixture
```

```
BLEND_BENCH r=1 k=1 layers=56 blend_us=344.0
BLEND_BENCH r=1 k=4 layers=56 blend_us=1266.3
BLEND_BENCH r=1 k=8 layers=56 blend_us=2992.3
BLEND_BENCH r=2 k=1 layers=56 blend_us=501.0
BLEND_BENCH r=2 k=4 layers=56 blend_us=10080.7
BLEND_BENCH r=2 k=8 layers=56 blend_us=28584.3
[bench_lora_mixture] LATTICE_MODEL_DIR not set; skipping GPU decode bench. Set it to a valid Qwen3.5-0.8b Q4 dir to enable decode tok/s measurements.
```

(`layers=56` = the source's hardcoded `NUM_LAYERS=28` × 2 modules (`q_proj`+`v_proj`) for this
half of the bench; see the failure-mode note below on why 28 is stale for `qwen3.5-0.8b`, which
actually has 24 hidden layers.)

With `LATTICE_MODEL_DIR` pointed at a Q4 dir that has `tokenizer.json` alongside it:

```bash
BENCH_ITERS=3 BENCH_WARMUP=1 BENCH_NEW_TOKENS=8 \
LATTICE_MODEL_DIR="$Q4_DIR" \
  ./target/release/bench_lora_mixture
```

```
BLEND_BENCH r=1 k=1 layers=56 blend_us=347.3
...
[bench_lora_mixture] loading model from <q4-dir>
[load-timer] Total: 3.155s
bench_lora_mixture failed: Inference error: module 'q_proj' is a full-attention projection but layer 21 is GDN
```

### Common failure modes, including one confirmed bug not fixed here

- **`GDN layer` crash on the `DECODE_BENCH` half: confirmed bug, filed as
  [ohdearquant/lattice#637](https://github.com/ohdearquant/lattice/issues/637).** Qwen3.5-0.8b
  is a hybrid architecture: `config.json`'s `layer_types` shows only every 4th layer
  (`full_attention_interval=4`) is `full_attention` (layers 3, 7, 11, 15, 19, 23 of 24 layers
  total, not the 28 the source comment assumes); the rest are `linear_attention` (GDN) and have
  no `q_proj`. `run_gpu_decode_bench` unconditionally builds a synthetic `q_proj` LoRA delta for
  every layer index `0..num_hidden_layers`, so it deterministically crashes the first time it
  reaches a GDN layer index (layer 21 in the run above); this reproduces every time against any
  Qwen3.5-family Q4 checkpoint, not just this one. **Do not attempt to fix this in a docs PR**;
  see the linked issue for the real fix (derive full-attention layer indices from
  `cfg.layer_types` instead of hardcoding `q_proj` on every layer). The `BLEND_BENCH` half is
  unaffected (it's pure CPU synthetic data, no real model), but its `layers=56` figure is also
  not representative of qwen3.5-0.8b's real 24-layer count.
- Missing `tokenizer.json` next to `LATTICE_MODEL_DIR` prints
  `[bench_lora_mixture] tokenizer.json not found; skipping GPU bench` and returns `Ok(())`. The
  blend-only numbers still print, just no `DECODE_BENCH` lines: this is a soft skip, not a crash.
- Running off macOS or without `--features metal-gpu,f16` prints
  `bench_lora_mixture requires macOS + --features metal-gpu,f16.` and exits 1 for the whole
  binary. There is no CPU-only mode, unlike `chat_metal`'s REPL-vs-Metal split.

---

## GPU-lock discipline used for this document

Every Metal-GPU-touching command above (`chat_metal`, `ppl_metal`, `bench_lora_mixture`'s
`DECODE_BENCH` half) was run wrapped in an exclusive advisory flock on
`/tmp/lion-metal-gpu-test.lock`, mirroring `gpu_test_lock()` in
`crates/inference/src/forward/metal_qwen35.rs`, so these runs never overlapped concurrent GPU
work from another process on the machine. macOS ships no `flock(1)`; any equivalent
`fcntl.flock`/`flock(2)`-based wrapper that takes the same exclusive lock on that path before
running the command satisfies the same discipline.
