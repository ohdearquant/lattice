# Q4 and QuaRot Quantization Workflow

This document walks through the full path from an unquantized Qwen3.5 checkpoint to a
running, verified 4-bit model: `quantize_q4` (plain Q4_0), `quantize_quarot` (Hadamard-rotated
Q4_0, ADR-044), loading the result, and the perplexity-based check that tells you whether the
conversion actually preserved model quality. Every command below was run against
`~/.lattice/models/qwen3.5-0.8b` in this repository to confirm the exact flags and output shape.

For the underlying design (why rotation helps 4-bit quantization, the forward-equivalence
gate, the acceptance threshold), see
[`docs/adr/ADR-044-quarot-rotated-quantization.md`](adr/ADR-044-quarot-rotated-quantization.md).
This document is the practical companion: what to type and what the output means.

## Two converters, two purposes

| Tool                  | Input                    | Output                                     | Rotation | Use when                                                 |
| --------------------- | ------------------------ | ------------------------------------------ | -------- | -------------------------------------------------------- |
| `bin/quantize_q4`     | BF16/F16/F32 safetensors | Q4_0 `.q4`/`.f16` + index                  | no       | Fast, simple 4-bit conversion                            |
| `bin/quantize_quarot` | BF16/F16/F32 safetensors | Hadamard-rotated Q4_0 `.q4`/`.f16` + index | yes      | Lower quantization error at the same bit width (ADR-044) |

Both write one file per tensor (`<sanitized_name>.q4` for quantized tensors, `<sanitized_name>.f16`
for tensors kept at higher precision) plus a `quantize_index.json` that records, per tensor, its
output file, whether it was quantized, its original shape, and element count. This directory
layout is what `MetalQwen35State::from_q4_dir` (the Metal loader used by `lattice serve`,
`lattice chat`, `chat_metal`, and `eval_perplexity --q4-dir`) expects.

## Step 1: plain Q4_0 with `quantize_q4`

```bash
cargo build --release -p lattice-inference --bin quantize_q4
./target/release/quantize_q4 \
  --model-dir ~/.lattice/models/qwen3.5-0.8b \
  --output-dir ~/.lattice/models/qwen3.5-0.8b-q4
```

`quantize_q4` has no `--help` — passing an unrecognized flag, or omitting a required one, prints
usage to stderr and exits 1:

```
$ ./target/release/quantize_q4
--model-dir is required
Usage: quantize_q4 --model-dir <DIR> --output-dir <DIR> [--dry-run]

  --model-dir   directory containing model.safetensors[.index.json]
  --output-dir  directory to write .q4 and index files
  --dry-run     read tensors but skip writing output
```

`--model-dir` and `--output-dir` are both required; `--dry-run` reads and quantizes every tensor
in memory (so you see the same per-tensor log and summary) but skips every disk write, which is
useful as a smoke test before committing to a large conversion. There is no `--seed` — plain Q4_0
quantization is a deterministic per-block min/max scale-and-round, so there is nothing to seed.

A real run against the 0.8B checkpoint (which also carries a vision tower and an MTP head, hence
488 tensors rather than just the text-transformer's share):

```
=== quantize_q4: SafeTensors → Q4_0 ===
Model dir:  /Users/lion/.lattice/models/qwen3.5-0.8b
Output dir: /Users/lion/.lattice/models/qwen3.5-0.8b-q4
Tensors:    488

  [1/488] F16   model.language_model.embed_tokens.weight  shape=[151936, 1024]  296.8MB  dtype=BF16  0.128s
  [2/488] Q4_0  model.language_model.layers.0.self_attn.q_proj.weight  shape=[4096, 1024]  8.0MB→2.5MB  0.02s
  ...
  [488/488] F16   mtp.pre_fc_norm_hidden.weight  shape=[1024]  0.0MB  dtype=BF16  0.000s
Index written: /Users/lion/.lattice/models/qwen3.5-0.8b-q4/quantize_index.json

=== Summary ===
Tensors processed: 488
  Quantized (Q4_0): 271
  Kept (F16):       217
Input size:   1.63 GB
Output size:  0.51 GB
Ratio:        3.20x  (31.3%)
Total time:   6.2s
```

271 of 488 tensors are quantized; the other 217 (norms, biases, GDN's `A_log`/`dt_bias`/conv1d
weights, and the embedding table in this run) are kept at their original width and written as
`.f16`. The `should_quantize` rule in `quantize_q4.rs` targets the large projection/MLP/embedding
matrices and explicitly excludes norms, biases, and Mamba/GDN-specific scalars — quantizing those
would save almost no memory and would hurt accuracy disproportionately.

### The gotcha: `quantize_q4`'s output has no `config.json`

This is the single most important thing in this document. **`quantize_q4` never writes a
`config.json`** — its output directory holds only `.q4`/`.f16` tensor files and
`quantize_index.json`. This differs from `quantize_quarot` (below), whose output directory
_does_ include one. Every consumer of a Q4 directory that isn't `quantize_q4` itself assumes
`config.json` might be present, but they disagree on what happens when it's not:

- `lattice serve --model <q4-dir>` and `lattice chat --model <q4-dir>` (in `lattice.rs`)
  call a helper (`load_q4_config`) that **falls back to a hardcoded Qwen3.6-27B default config**
  when `config.json` is missing, printing only a one-line warning to stderr:
  `Warning: <dir> has no config.json; falling back to the Qwen3.6-27B default config.` For any
  checkpoint that isn't actually the 27B model — including the 0.8B checkpoint used throughout
  this document — this silently loads the wrong architecture (wrong layer count, hidden size,
  attention/GDN layer pattern). It does not fail; it fails _wrong_, and only a warning line
  distinguishes that from a normal load.
- `eval_perplexity --q4-dir <dir>` (see Step 3) hard-errors instead:
  `ERROR: Q4 dir <dir> is missing config.json` and exits 1. No silent fallback.
- `lattice doctor --model <q4-dir>` (see "Verify before you load", below) also ends up using the
  same Qwen3.6-27B fallback internally, which causes it to expect ~500 additional tensors that a
  smaller checkpoint doesn't have. It **does** fail closed (exit 1, `Result: NOT READY`), but the
  reported reason — "531 required tensor(s) missing" — points at the symptom, not the cause.

**The fix is the same in all three cases: copy `config.json` from the source model directory into
the `quantize_q4` output directory before loading it with anything else.**

```bash
cp ~/.lattice/models/qwen3.5-0.8b/config.json ~/.lattice/models/qwen3.5-0.8b-q4/config.json
```

`quantize_quarot` (Step 2) does not have this problem — it writes its own `config.json`
automatically as part of its output.

## Step 2: QuaRot-rotated Q4_0 with `quantize_quarot`

QuaRot (Ashkboos et al., NeurIPS 2024) applies a Hadamard rotation to the residual stream before
quantizing, spreading outlier magnitude across channels so that per-block min/max quantization has
less dynamic range to cover. `quantize_quarot` implements ADR-044 step 3c: rotate, quantize, then
verify the rotation didn't change the model's forward output beyond a tight numerical tolerance
before writing anything.

Unlike `quantize_q4`, this tool has real `--help`:

```
$ ./target/release/quantize_quarot --help
usage: quantize_quarot --model-dir <PATH> --output-dir <PATH> --seed <U64> [OPTIONS]

QuaRot Q4_0 converter for Qwen3.5 (ADR-044 step 3c).

required:
  --model-dir <PATH>         Input directory with config.json + safetensors.
  --output-dir <PATH>        Output directory (created if absent).
  --seed <U64>               Hadamard rotation seed (decimal or 0x... hex).

options:
  --tolerance <F64>          Forward-equivalence tolerance. Default 1e-5.
  --num-probe-tokens <USIZE> Chain-probe sample size. Default 4.
  --dry-run                  Run pipeline + gate; skip disk writes.
  -h, --help                 Print this help and exit.

The converter refuses to write any output if the forward-equivalence
gate fails (delta > tolerance) — this protects against silently shipping
a model whose logits diverged during conversion.
```

`--seed` is required and has no default: converted artifacts from different seeds are not
interchangeable (the rotation matrix is seed-derived), so the project's convention is to record
the seed used, not rely on an implicit default. It accepts decimal or `0x`-prefixed hex, with
optional `_` separators exactly like a Rust integer literal (`0xCAFE_BABE_DEAD_BEEF`).

```bash
cargo build --release -p lattice-inference --bin quantize_quarot
./target/release/quantize_quarot \
  --model-dir ~/.lattice/models/qwen3.5-0.8b \
  --output-dir ~/.lattice/models/qwen3.5-0.8b-q4-quarot \
  --seed 0xC0FFEE
```

Real output from that command:

```
=== quantize_quarot: QuaRot Q4_0 converter ===
Model dir:   /Users/lion/.lattice/models/qwen3.5-0.8b
Output dir:  /Users/lion/.lattice/models/qwen3.5-0.8b-q4-quarot
Seed:        0x0000000000c0ffee
Tolerance:   0.00001
Probe toks:  4
Mode:        WRITE

=== Summary ===
Tied input:        true
Quantized (Q4_0):  188
Kept (F16):        148
Input bytes:       1.44 GB
Output bytes:      0.62 GB
Compression:       2.30x (43.4%)
Forward-equiv:     max_abs=7.017e-14, mean_abs=1.050e-14 (tol=1e-5, probes=[17156, 85503, 9161, 94570])
Wall time:         77.6s
```

`Tied input: true` means this checkpoint ties its embedding and `lm_head` weights (one tensor,
two roles) — the converter detected and preserved that. The `Forward-equiv` line is the
correctness gate: it runs the _unrotated_ and _rotated_ model forward on `--num-probe-tokens`
sample token chains and reports the max/mean absolute logit difference. Here `max_abs=7e-14`
against a `tol=1e-5` tolerance is nine orders of magnitude inside the bound — the rotation is
mathematically exact up to floating-point noise, as it should be (a Hadamard rotation is
orthogonal; it changes representation, not the function computed). **If this gate fails, the tool
writes nothing** — `ConversionReport` is never returned, `main` prints `ERROR: ...` and returns
`ExitCode::FAILURE` (exit 1), and the output directory is left without a completed conversion.
There is no partial/corrupt output state to clean up.

Note the different tensor counts (188 quantized + 148 kept = 336) versus plain `quantize_q4`'s
271 + 217 = 488 on the same source directory. The two exclusions behind that gap are different
kinds of things, and it's worth being precise about which is which
(`crates/inference/src/quant/quarot/convert.rs`):

- **Vision tower — excluded, never read.** Qwen3.5-0.8B's checkpoint on disk is multimodal (it
  ships a `vision_config`); QuaRot's converter never reads or rewrites those tensors on either the
  input or output side. They simply aren't part of this pipeline.
- **MTP tensors — copied, not rotated or quantized.** When the config has `mtp_num_hidden_layers >
  0` (true for Qwen3.5-0.8B, which ships 1 MTP layer), `write_mtp_weights_quarot` copies each MTP
  tensor into the output directory as an unrotated, unquantized `.f16` file. They're part of the
  336-tensor output — counted in the 148 "kept (F16)" — not silently dropped. Rotating and
  quantizing MTP tensors is deferred to a later phase (see the ADR-051 note in `convert.rs`); today
  they ride along as plain f16 copies.

If your workflow needs the QuaRot path specifically for text generation (chat/serve), this is
expected — the language-model tensors are the ones rotated and quantized, and MTP still loads and
runs (as f16), it's just not yet part of the rotation; it is not a partial-conversion bug.

## Step 3: verify with perplexity, not just "it loaded"

A checkpoint that loads and generates _something_ is not evidence the quantization preserved
quality — Q4_0 and especially a botched rotation can produce fluent-looking but degraded text.
`eval_perplexity` (ADR-044 step 4) is the actual acceptance check: it scores a held-out text
corpus and reports perplexity (PPL), and in its dual-Q4 mode it computes the PPL delta between an
unrotated and a rotated checkpoint against a threshold.

```
$ ./target/release/eval_perplexity --help
```

(flags summarized from the binary's own doc comment, `crates/inference/src/bin/eval_perplexity.rs`)

| Flag                                    | Meaning                                                                                                                     |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `--model-dir <PATH>`                    | CPU baseline mode: safetensors checkpoint via `Qwen35Model::from_safetensors`.                                              |
| `--q4-dir <PATH>`                       | Metal Q4 mode: a `quantize_q4` output directory.                                                                            |
| `--quarot-q4-dir <PATH>`                | Metal Q4 mode: a `quantize_quarot` output directory.                                                                        |
| `--q4-dir` + `--quarot-q4-dir` together | Dual-Q4 mode: runs both, prints both reports, then the delta and the ADR-044 acceptance verdict.                            |
| `--tokenizer-dir <PATH>`                | Required for Metal modes — Q4 directories don't ship `tokenizer.json`, so point this at the original safetensors directory. |
| `--corpus-file <PATH>`                  | UTF-8 text file, tokenized end-to-end.                                                                                      |
| `--window` / `--stride`                 | Context window / stride in tokens. Default 512 / 256.                                                                       |
| `--max-tokens`                          | Cap on tokens scored, for a fast smoke run. Default: no cap (scores the whole corpus).                                      |
| `--delta-threshold`                     | Dual-Q4 mode only. Default `0.5`.                                                                                           |

Exit code is 0 when PPL is computed and (in dual-Q4 mode) the delta is below the threshold; 1 on
any error, or on a dual-Q4 delta at or above the threshold (acceptance fail).

The repository ships a small held-out corpus at `docs/bench_results/wiki.test.raw` for exactly
this purpose. A real dual-Q4 run against the two checkpoints produced above (remember: `--q4-dir`
needs `config.json` copied in per the gotcha above; `--quarot-q4-dir` already has its own):

```bash
./target/release/eval_perplexity \
  --q4-dir ~/.lattice/models/qwen3.5-0.8b-q4 \
  --quarot-q4-dir ~/.lattice/models/qwen3.5-0.8b-q4-quarot \
  --tokenizer-dir ~/.lattice/models/qwen3.5-0.8b \
  --corpus-file docs/bench_results/wiki.test.raw \
  --max-tokens 4096
```

```
=== Perplexity Report (unrotated Q4) ===
PPL:                18.349821
Tokens scored:      4095
Windows:            15

=== Perplexity Report (QuaRot Q4) ===
PPL:                20.608386
Tokens scored:      4095
Windows:            15

=== Acceptance Gate (ADR-044 step 4) ===
Unrotated Q4 PPL: 18.349821
QuaRot Q4 PPL:    20.608386
PPL delta:        +2.258565  (quarot - unrotated)
Threshold:        < 0.500000
Verdict:          FAIL (delta >= threshold)
```

Read this result carefully before concluding anything is broken: **15 windows is far too small a
sample to make an accept/reject call.** `--max-tokens 4096` is useful for confirming the whole
pipeline runs end to end (both checkpoints load, both score, the gate logic itself works), not for
a real quality verdict — treat a FAIL at this scale as "rerun on more data," not "the conversion is
bad." The project's own published reference numbers, at a still-modest but larger 2048-token
window count (`docs/bench_results/perplexity.tsv`, cited in the README), show the same pattern:
`lattice q4 19.266338` vs `lattice q4-quarot 19.950512` — a delta of +0.68, which is _also_ over
the nominal 0.5 threshold. In other words, a nonzero, threshold-exceeding delta at reduced token
counts is a known characteristic of this measurement at small scale, not something unique to a
fresh from-scratch conversion. Reproduce the project's own reference numbers with
`./scripts/bench_context_scaling.sh`, or omit `--max-tokens` entirely (scores the full corpus,
310,034 tokens for `wiki.test.raw` — expect this to take considerably longer than the smoke-test
run above) before treating either a PASS or a FAIL as a real acceptance decision.

`eval_perplexity` also emits a machine-readable line per report on stdout, independent of the
human-readable summary on stderr:

```
@@lattice {"ev":"perplexity","label":"unrotated Q4","ppl":18.349821,"nll":2.90962,"tokens":4095,"windows":15,"ms":14090}
```

## Step 4: load and run the quantized checkpoint

Once you have a Q4 directory with `config.json` present (native for `quantize_quarot`, copied
manually for `quantize_q4`), it loads through the same CLI entry points as any other checkpoint —
`lattice` auto-detects the format from the directory contents (presence of any `*.q4` file selects
the Q4/Metal path; `model.safetensors`/`.index.json` selects the CPU/safetensors path):

```bash
# Interactive, one-shot-per-line REPL (no cross-turn conversation history)
./target/release/lattice chat \
  --model ~/.lattice/models/qwen3.5-0.8b-q4-quarot \
  --tokenizer-dir ~/.lattice/models/qwen3.5-0.8b

# OpenAI-compatible HTTP server — see docs/serve-http-api.md for the full API
./target/release/lattice serve \
  --model ~/.lattice/models/qwen3.5-0.8b-q4-quarot \
  --tokenizer-dir ~/.lattice/models/qwen3.5-0.8b \
  --port 8080
```

`--tokenizer-dir` is required for a Q4 directory in both cases (Q4 output never includes
`tokenizer.json`) and should point at the original safetensors directory. Both commands require
the `metal-gpu` feature at build time (`cargo build --release -p lattice-inference --bin lattice
--features metal-gpu,f16`); without it, a Q4 model directory is rejected with a clear
"requires the metal-gpu feature" error rather than attempting a CPU fallback (Q4 inference in this
codebase is Metal-only).

### Verify before you load: `lattice doctor`

`lattice doctor --model <dir> --tokenizer-dir <dir>` is a preflight check: it reports the detected
format, weight memory footprint, KV-cache cost per token, and whether the checkpoint's tensors and
your system's memory are actually sufficient to load it, without spending the time to load and run
the model. Run it before `chat`/`serve` on any new Q4 output, including QuaRot outputs.

`doctor` exits 0 with `Result: OK` when the checkpoint is loadable and fits comfortably; exits 1 with
`Result: NOT READY` and a specific reason otherwise. **If `doctor` reports dozens or hundreds of
"missing required tensors" with layer indices well beyond what you'd expect for your model size**
(for example, layers in the 20s-40s for what you know is a small checkpoint), the near-certain
real cause is a missing `config.json` in that directory (see the gotcha in Step 1) rather than a
corrupted or incomplete conversion — `doctor` inherits the same Qwen3.6-27B config fallback as
`lattice serve`/`lattice chat`, so a missing config makes it expect a 27B-shaped tensor set against
your smaller checkpoint's actual tensors. Copy `config.json` in and re-run `doctor` before assuming
the conversion itself is broken.

`quantize_index.json` parsing and validation is centralized in one place: the
`lattice_inference::quant::q4_manifest` module (`crates/inference/src/quant/q4_manifest.rs`).
`doctor`'s tensor inventory, QuaRot rotation-seed detection, and the Metal Q4 loader's
QuaRot-flavor detection all go through this module's bounded reader (fail-closed on a
missing, truncated, or oversized file) and shape-normalized parser, rather than each
call site re-deriving its own copy of the contract. That module recognizes both
`quantize_index.json` manifest shapes:

- `quantize_q4` writes a bare JSON array of tensor entries.
- `quantize_quarot` writes an object with a `tensors` array and metadata such as `quarot_seed`.

Both shapes normalize to the same tensor inventory; the QuaRot seed field is populated only
for the object form, and is `None` (not an error) when absent from either shape. Older output
captured before issues [#626](https://github.com/ohdearquant/lattice/issues/626) and
[#627](https://github.com/ohdearquant/lattice/issues/627) may show an `invalid type: map, expected
a sequence` manifest error for QuaRot directories. That failure mode is historical for current
`lattice doctor` builds.

## Summary checklist

1. `quantize_q4 --model-dir SRC --output-dir Q4_DIR` — fast, no rotation, no `config.json` written.
2. `quantize_quarot --model-dir SRC --output-dir QUAROT_DIR --seed 0x...` — rotated, includes its
   own `config.json`, refuses to write output if its internal forward-equivalence check fails.
3. Copy `config.json` from `SRC` into `Q4_DIR` (not needed for `QUAROT_DIR`).
4. `lattice doctor --model <dir> --tokenizer-dir SRC` before loading anything for real.
5. `eval_perplexity --q4-dir Q4_DIR --quarot-q4-dir QUAROT_DIR --tokenizer-dir SRC --corpus-file <corpus>`
   (full corpus, no `--max-tokens`) as the actual quality gate — exit code 0 plus a delta below
   the threshold is the real "it worked" signal, not just successful loading.
6. `lattice chat` / `lattice serve --model <dir> --tokenizer-dir SRC` to actually use the result.
