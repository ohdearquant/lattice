# ADR-089: PaddleOCR-VL in lattice: CPU reference forward per slice, each gated by HF goldens, before any accelerated path

**Status**: Proposed
**Date**: 2026-09-02
**Crate**: lattice-inference (`model/ernie45.rs`, `model/paddleocr_vl.rs`, `vision/paddleocr_vit.rs`, `vision/paddleocr_preprocess.rs`, tokenizer goldens)

## Context

PaddleOCR-VL-1.6 is a document-reading vision-language model: a NaViT-style vision encoder, a 2x2 spatial
merge projector, and an ERNIE-4.5 decoder (about 0.9B parameters) with sectioned three-dimensional M-RoPE.
It reads tables and mixed Chinese and Latin text from photographs of printed documents, which is the workload
lattice is adding it for. Nothing in the crate shared its conventions: the vision position table is resampled
with half-pixel bilinear interpolation rather than the linspace convention the Qwen3.5 encoder uses, the
block MLP uses tanh-GELU while the projector uses exact GELU with a different epsilon, the decoder's M-RoPE
sections follow the checkpoint's own layout, and the tokenizer splits digits into single tokens.

Every one of these is a place where a port can be plausibly wrong and still produce text. A convention
mismatch on Qwen3.5 (RoPE pairing) once cost days while being attributed to precision drift; the
differential-test rule in `CLAUDE.md` came from that. This model has more such conventions than any previous
port, and a Metal path written against a wrong CPU reference would carry the error into every device.

## Decision

1. The model lands as four slices, each a CPU f32 reference forward with its own golden-gated test, in
   dependency order: tokenizer (#1455), text decoder (#1457), vision encoder and projector (#1458), and the
   end-to-end forward with preprocessing, prompt template, embedding splice, and greedy decode (#1463).
2. Goldens are captured from the pinned checkpoint's own HF modeling source over synthetic inputs, stored as
   per-checkpoint activation summaries so a divergence localizes to a block, and compared with a stated
   tolerance and a reported worst diff. The end-to-end gate asserts the preprocessing grid, prompt ids, rope
   positions, projector rows, spliced embeddings, the greedy choice at every prompt position, and the greedy
   token sequence exactly.
3. Every gate is fail-closed where the checkpoint exists: the job `PaddleOCR-VL ERNIE-4.5 decoder gate
   (ARM Linux)` in `.github/workflows/e2e-parity.yml` provisions the pinned checkpoint and runs the decoder,
   vision, and end-to-end goldens with `LATTICE_POCR_GATE_ENFORCE=1`, under which a missing checkpoint or a
   disabled `f16` feature panics. In every other job the test prints `SKIP <test>: checkpoint not found` and
   returns; that run is degraded, not passing, and no PR may cite it as gate evidence. A slice is not merged
   until its gate runs in that job.
4. Each slice's PR records mutation arms: a deliberate convention flip (interpolation mode, RoPE section
   order, projector row order, GELU variant) run against the gate, with the outcome stated even when the
   fixture does not discriminate it. An undiscriminated arm is written down as a known gap, not dropped.
5. No accelerated path (Metal, quantized weights, KV cache, batched prefill) is written for this family until
   the CPU reference is merged; each later path is compared against the CPU reference on the same goldens, and
   the reference stays in the tree as the oracle.
6. Bench-compare dispositions remain structural only while a search of every declared measurement target
   proves none executes a changed path, with the population and residual risk stated in the PR body. The
   first text-only Metal prefill implementation exposes an explicit backend without adding a measurement
   target. This revises the earlier proposal to add a benchmark in the same change: correctness and its
   controls are established first, while initialization cost, latency, throughput and scaling remain
   unmeasured. Performance optimization or automatic backend selection requires a measured target and
   recorded comparisons; parity is never evidence of speed.

### Explicit Metal text prefill

`forward::metal_ernie45::MetalErnie45State::new(&config, &weights, max_seq_len)` uploads an
`Ernie45Weights` decoder into persistent f32 buffers. The implementation requires macOS and `metal-gpu`;
the same API returns an availability error on unsupported builds. The separate `f16` feature is needed to
load the shipped BF16 checkpoint, not to execute already-loaded f32 weights.

`prefill(&mut self, ids, logits)` recomputes one complete text sequence. The caller supplies exactly
`ids.len() * config.vocab_size` f32 output elements, laid out as contiguous token-major rows. Positions begin
at zero on every call, attention is causal, and no KV cache survives between calls. Head width is 128;
query and KV projection widths remain independent of the hidden-state width. The CPU configuration
validator must accept the section layout before it is reduced to text-only 1-D stride-half RoPE.

Construction validates the complete layer count, every weight shape and value, indexing capacity and
actual device/pipeline limits. Prefill checks sequence capacity, token IDs and exact output size before
encoding. All layers share one queue, pipeline collection, RoPE tables and scratch set. Embedding lookup,
the complete layer loop, final RMSNorm and the untied language head execute on Metal. The caller's output
changes only after command completion and a finite-value check. GPU failure returns an error.

The full-model correctness test compares every logit with the retained CPU forward on all four committed
decoder cases and a separate 17-token cross-tile case. It applies the committed HF logit assertions directly
to Metal output and checks the CPU activation summaries. Its predeclared componentwise bound is
`abs(metal - cpu) <= 0.002 + 0.002 * abs(cpu)`, with finite values and exact per-position greedy choices.
Negating only the final decoder layer's down projection must make the unchanged comparator reject the
output; restoring its original bits must recover parity. A separate head-width rejection test must fail
when that guard is removed. Test-only dispatch counters verify the full embedding/layer/head path, and
enforcement turns missing features, device or checkpoint into failures instead of successful skips.

### Explicit Metal cache entries

The embedding-driven entries preserve the production decoder's input contract without changing its
backend selection. `kv_prefill(embeds, positions, cache, logits)` accepts contiguous `[sequence, hidden]`
embeddings and one `[T,H,W]` position triple per token. `kv_decode_step` accepts one embedding and its
explicit position triple. Both return only the final vocabulary row. Positions are independent of cache
length; token order determines causality. The existing token-ID prefill retains its all-token output.

`new_kv_cache(capacity)` allocates separate device-side f32 K and V buffers with layout
`[layers, capacity, kv_dim]`. K is post-RoPE, V is unrotated, and only rows below `len` are live. The cache
belongs to its exact decoder state, including its weights and device, rather than merely matching its
geometry. `clear()` resets the live length and retains storage. Decode appends at the old length, attends
over that row and its prefix, and publishes the new length and caller output only after successful GPU
completion and finite-output validation. A failed execution may leave an unpublished row; a retry
replaces it. Invalid inputs must preserve the live prefix and output. Capacity is fixed at construction.

Two separate acceptance invariants apply. Cached K/V rows must equal independent Metal full-forward
rows bitwise in every layer. Every cached output must also meet the existing componentwise CPU bound
and agree on argmax. Validation crosses the attention tile boundary over multiple cached steps, compares
the same growing prefixes, and checks prefix preservation. A deliberate finite V-row perturbation must
make the unchanged parity assertion fail, and restoring the original row and rebuilding must recover
parity. Public guard tests require removal-sensitive controls with byte-verified restoration. These
entries do not select a backend in `generate_greedy`, execute vision, or establish a performance claim.

## Consequences

- Correctness is established before speed: the end-to-end CPU forward takes minutes per image in a release
  build with no KV cache, which is acceptable for a reference and unusable for a product path. That is the
  point of the ordering, and the cost is that the accelerated path arrives later.
- The goldens fixture couples the tests to one checkpoint revision; a checkpoint update regenerates the
  goldens through the same capture script and re-runs every gate.
- Shared helpers widened to `pub(crate)` for reuse (multi-head attention, exact GELU, in-place RoPE) now have
  two callers; a change to them must keep both families' goldens green.
- The original projector summaries do not discriminate the tanh-GELU substitution. The additional
  `first_row_max_abs` golden scans all 1024 channels of the first projected row and compares its maximum
  magnitude with a separate absolute tolerance of `1.25e-4`. The original summary tolerances remain
  unchanged. This is a first-row statistic, not a full-output elementwise comparison. The second linear
  is a learned weighted dot product; measurement shows amplification of this mutation, not averaging.
- `scripts/gen_paddleocr_vision_goldens.py` regenerates the vision fixture from the pinned model and
  hash-verified reference source. Its runtime pins and output manifest make the capture reproducible;
  removing only the added projector fields reproduces the original fixture bytes. A changed old field
  fails generation rather than silently replacing the oracle.
- The checkpoint job runs decoder, vision and end-to-end goldens with `--release --features f16` and
  `LATTICE_POCR_GATE_ENFORCE=1`. It preserves Cargo's failure status before checking the execution markers
  and fixture counts, so a new assertion failure cannot be hidden by the earlier summary lines.

## Evidence

- Tokenizer: 17 corpus cases against HF `tokenizers` (digit splitting, bbox tokens, CJK text, byte fallback,
  chat prompt); 8 tests pass; the Gemma goldens sharing the engine still pass (23).
- Decoder: four cases, greedy argmax agrees with HF at every position; tolerance 2e-3 absolute and relative on
  the golden fields with about six times headroom; three mutations each redden the gate and the restored file
  is byte-identical.
- Vision encoder and projector: three synthetic grids (4x4, 6x10, 12x8); worst diffs 4.5e-5, 3.7e-5, 2.6e-4
  on the original sampled checkpoints against `1e-3 + 1e-3 * abs(reference)`; interpolation and RoPE-phase
  mutations fail at the first compared checkpoint. The GELU mutation passes those original checks.
- Projector calibration compares unchanged Rust, projector-only tanh-GELU substitution and the pinned HF
  reference on CPU f32. For the first-row maximum magnitude, mutation signal is `abs(swapped - unchanged)`.
  A conservative noise bound is the maximum unchanged/HF error over every channel of that same first row:

  | grid | first-row maximum signal | full-first-row noise bound | ratio |
  | ---- | -----------------------: | -------------------------: | ----: |
  | 4x4  |               1.25408e-3 |                 3.95775e-5 | 31.69 |
  | 6x10 |               4.32491e-4 |                 2.09808e-5 | 20.61 |
  | 12x8 |               6.79016e-4 |                 2.12193e-5 | 32.00 |

  Even the smallest signal divided by the largest noise bound across cases is 10.93. The common
  `1.25e-4` absolute tolerance leaves 3.16 times the observed noise bound and the smallest mutated/HF
  residual is 3.29 times that tolerance. These measurements are from one CPU host; byte-identical
  unchanged/restored outputs establish repeatability there, not a universal floating-point error bound.
  A tight full-tensor comparator was rejected because its 12x8 case separates maximum signal from
  maximum noise by only 5.48. Argmax indices do not change under the substitution. The first-row maximum
  extends the existing first-row sample without selecting individual channels by their measured noise.
- End to end: a rendered table image, 24 greedy tokens from HF matched exactly; decoder goldens 5 s, vision
  goldens 161 s, end-to-end 527 s in a release build with `--features f16` and the enforce flag set; two
  mutations (M-RoPE row swap, projector row reversal) each redden their gate.
- The producing commands are in each PR body and in the tests' module docs.

## Alternatives considered

- Port the Metal path first and validate against HF at the end: rejected; a wrong convention would be found
  after the kernels were written, and the diagnosis would have no CPU oracle to bisect against.
- Validate only the final tokens: rejected; a token match on one image does not localize a divergence and
  cannot tell a wrong-but-lucky block from a right one.
- Reuse the Qwen3.5 vision encoder with configuration flags: rejected; the position-embedding, RoPE, norm, and
  activation conventions differ in ways that flags would hide rather than document.
