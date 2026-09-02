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
3. Every gate is fail-closed: it skips with a printed line when the checkpoint is absent and panics under
   `LATTICE_POCR_GATE_ENFORCE=1`, so a CI run without the checkpoint cannot read as a pass.
4. Each slice's PR records mutation arms: a deliberate convention flip (interpolation mode, RoPE section
   order, projector row order, GELU variant) run against the gate, with the outcome stated even when the
   fixture does not discriminate it. An undiscriminated arm is written down as a known gap, not dropped.
5. No accelerated path (Metal, quantized weights, KV cache, batched prefill) is written for this family until
   the CPU reference is merged; each later path is compared against the CPU reference on the same goldens, and
   the reference stays in the tree as the oracle.
6. Bench-compare dispositions for these slices are structural while no declared bench target reaches the
   model, with the population searched and the residual risk stated in the PR body; the first Metal slice adds
   a bench target and moves the family onto measured dispositions.

## Consequences

- Correctness is established before speed: the end-to-end CPU forward takes minutes per image in a release
  build with no KV cache, which is acceptable for a reference and unusable for a product path. That is the
  point of the ordering, and the cost is that the accelerated path arrives later.
- The goldens fixture couples the tests to one checkpoint revision; a checkpoint update regenerates the
  goldens through the same capture script and re-runs every gate.
- Shared helpers widened to `pub(crate)` for reuse (multi-head attention, exact GELU, in-place RoPE) now have
  two callers; a change to them must keep both families' goldens green.
- One known undiscriminated arm exists: the projector GELU variant swap passes the vision goldens at the
  current tolerance because the 4608-wide second linear averages the difference down; the end-to-end token
  gate is the check that catches it.

## Evidence

- Tokenizer: 17 corpus cases against HF `tokenizers` (digit splitting, bbox tokens, CJK text, byte fallback,
  chat prompt); 8 tests pass; the Gemma goldens sharing the engine still pass (23).
- Decoder: four cases, greedy argmax agrees with HF at every position; tolerance 2e-3 absolute and relative on
  the golden fields with about six times headroom; three mutations each redden the gate and the restored file
  is byte-identical.
- Vision encoder and projector: three synthetic grids (4x4, 6x10, 12x8); worst diffs 4.5e-5, 3.7e-5, 2.6e-4
  against a 1e-3 tolerance; interpolation and RoPE-phase mutations fail at the first compared checkpoint; the
  GELU mutation passes (recorded above).
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
