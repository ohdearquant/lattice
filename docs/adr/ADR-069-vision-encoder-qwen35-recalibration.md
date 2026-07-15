# ADR-069: Vision Encoder Recalibration for Qwen3.5-0.8B

**Status**: Proposed
**Date**: 2026-07-05
**Crate**: lattice-inference
**Revises**: ADR-049 (Qwen-VL vision encoder path) — recalibrates the ViT geometry, weight-key
schema, and memory tier to the actual shipping checkpoint, and overturns ADR-049 R3 (the decoder
RoPE assumption for image tokens).

## Context

ADR-049 (Accepted 2026-05-19) built the vision scaffold in `crates/inference/src/vision/`
(`preprocess.rs`, `vit.rs`, `merger.rs`, `mod.rs`, `multimodal.rs`) plus a tested
`generate_multimodal` entry point at `metal_qwen35.rs:11318`. That scaffold was written against a
hypothetical 7B Qwen3-VL and is **inert**: no `model.visual.*` weights are loaded, the config parser
drops every vision field, and the decoder discards the supplied patch embeddings
(`metal_qwen35.rs:11492-11498` steps token id 0 at each visual position and drops the patch
embedding, marked "v0 limitation / v1 Metal injection point"). There is real scaffolding and zero
live image capability; the gap is entirely engine-side.

Meanwhile the checkpoint the engine actually runs — `Qwen3.5-0.8B`, arch class
`Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5` — is a **native unified vision-language
model** in both its fp16 and lattice-q4 forms. This is verified two ways against the checkpoint's
public config and weight index:

- `config.json`: `vision_config` present (`depth 12`, `hidden_size 768`, `num_heads 12`,
  `patch_size 16`, `spatial_merge_size 2`, `out_hidden_size 1024`, `temporal_patch_size 2`,
  `num_position_embeddings 2304`, `in_channels 3`), plus `image_token_id 248056`,
  `video_token_id 248057`, `vision_start_token_id 248053`, `vision_end_token_id 248054`.
- `model.safetensors.index.json`: 153 real `model.visual.*` tensors — 144 ViT-block tensors
  (12 blocks × 12 tensors) + 6 merger + 2 `patch_embed.proj` + 1 `pos_embed`. The text decoder is
  under `model.language_model.*`. The lattice-q4 build **retains** all 153 quantized visual files;
  the 27B-q4 dump dropped them, which is why 27B is text-only at rest.

The 0.8B checkpoint is Apache-2.0, roughly 1.6 GB fp16 and 0.5 GB q4, and is the smallest
vision-capable member of the family; no larger checkpoint is required for this plan. Building image
input on the 0.8B checkpoint therefore needs no new model architecture, only engine work.

Because the ViT geometry, weight-key schema, memory tier, and the decoder RoPE decision all differ
from ADR-049's assumptions, this is a material specification change rather than a code detail, and it
is recorded here so the corrected contract is signed off before dependent implementation lands.

## What ADR-049 assumed vs. the actual 0.8B checkpoint

| ADR-049 assumption (7B Qwen3-VL) | Actual Qwen3.5-0.8B checkpoint |
|---|---|
| ViT depth 27, d_model 1152, 16 heads | ViT depth 12, hidden 768, 12 heads, out_hidden 1024 |
| Weight keys `vision_model.*` | Weight keys `model.visual.*`; text decoder `model.language_model.*` |
| ~15 GB total, targets 32 GB tier | ~1.6 GB fp16 / ~0.5 GB q4 whole model; runs on the 16 GB base tier and below |
| Image tokens **bypass** decoder RoPE (R3) | Decoder applies genuine 3-axis interleaved M-RoPE to image-token positions (R3 overturned — see Decision §2) |
| ViT Metal path is a deferred "v1" item; v0 is Metal-for-ViT-only, decoder unchanged | Every stage (ViT, merger, visual-embedding injection, decode) executes on Metal GPU |
| DeepStack visual indexes possible | `deepstack_visual_indexes: []` — not used by this checkpoint |

## Decision

### 1. Recalibrate to the 0.8B geometry and weight schema

Adopt the real `vision_config` (depth 12, hidden 768, 12 heads, patch 16, spatial_merge_size 2,
out_hidden 1024, temporal_patch_size 2) as the v0 target. Load the 153 `model.visual.*` tensors from
the safetensors checkpoint in both fp16 and q4 forms. The `out_hidden_size` (1024) equals the text
decoder `hidden_size` (1024), so the merger projects directly into the decoder embedding space with
no dimension mismatch. Parse the previously-dropped vision fields in `qwen35_config.rs` (see Scope
S1); text-only checkpoints without a `vision_config` continue to load unchanged.

### 2. Decoder-side 3-axis M-RoPE for image tokens (overturns ADR-049 R3)

ADR-049 R3 assumed image tokens bypass RoPE in the decoder, with position information applied only
inside the ViT. **This is wrong for Qwen3.5.** The decoder applies genuine multi-axis
(temporal / height / width) M-RoPE to image-token positions. Evidence, from a primary-source read of
the reference implementation (`transformers==5.12.1`, `models/qwen3_5/modeling_qwen3_5.py`):

- `apply_interleaved_mrope` (line 169) splices per-axis frequencies in an **interleaved** layout
  `[T,H,W,T,H,W,...]` (`length = mrope_section[dim] * 3`, line 181), preserving frequency
  continuity.
- `mrope_section` defaults to `[11, 11, 10]` (line 113) and is consumed when building the rotary
  frequencies (line 162: `freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)`).
- `get_vision_position_ids` (line 1251) and `get_rope_index` (line 1309) compute the 3-axis
  position ids from the image grid (`grid_thw`) across the processor boundary.

The blast radius is bounded by two facts:

- **Only the 6 GQA (`full_attention`) layers consume position embeddings.** In the layer dispatch
  (lines 761–797), `full_attention` layers pass `position_embeddings=position_embeddings`, while the
  18 `linear_attention` (GatedDeltaNet) layers do **not** touch them. The checkpoint's
  `text_config.layer_types` places the `full_attention` layers at indices
  **[3, 7, 11, 15, 19, 23]** (stride 4, offset 3) out of 24. So the M-RoPE work is confined to those
  6 layers.
- **Text-only prompts are unaffected.** `mrope_section` sums to 32, which equals the half-rotary
  dimension (`head_dim 256 × partial_rotary_factor 0.25 = 64`, half = 32). For a purely textual
  sequence the three position axes are identical (all equal to the 1-D token position), so the
  interleaved M-RoPE collapses exactly to the existing 1-D stride-half partial RoPE
  (`forward.rs:417-422`). No text-path behavior changes.

Consequence for the Rust implementation: the engine needs (a) a `get_rope_index`-equivalent that
maintains 3-axis (t, h, w) position bookkeeping for image-token spans given the image grid, and
(b) an interleaved per-axis frequency splice in RoPE-table construction, applied only in the 6 GQA
layers. This cannot be shimmed by reusing 1-D positions for image tokens; it is genuine new scope,
resolved here so the ADR body is complete rather than deferred.

### 3. Every stage on Metal GPU

ViT forward, merger/projection, visual-embedding injection into the residual stream, and the M-RoPE
frequency construction all execute on the Metal GPU (subject to the repository's GPU serialization
lock for any measurement run). This replaces ADR-049's "CPU-side ViT forward pass (Metal GPU path is
a v1 item)" wording. Sequential GPU use (encode image, read back, release ViT buffers, then decode)
remains the memory-pressure mitigation; at ~1.6 GB total on this checkpoint, peak pressure is not a
constraint on the 16 GB tier.

### 4. Studio surface: image content-part on the Chat composer

Image input is wired into Studio's Chat surface for hands-on testing. The composer gains an
image-attach affordance (attach / drag-drop a PNG or JPEG, shown as a thumbnail chip in the turn);
on send, the bytes route through the vision path so the reply is grounded in the image. The serve
chat path already carries `ChatMessage`; its content model is extended to carry an optional image
part that mirrors the OpenAI `image_url` content-part shape the API surface currently rejects. This
preserves the single unified rendering path and the single serve surface. A bundled sample image
plus a one-click "describe this" affordance gives a zero-setup smoke test. The exact widget and the
content-part schema are pinned in this ADR's Scope S6 rather than decided during implementation.

## Scope

S0 (the M-RoPE differential question) is **resolved** and its answer is Decision §2 above; it
produced no engine code. The remaining stages are independently verifiable; S1 and S2 are CPU-side
config and weight-loading preparation, and the runtime compute stages S3-S5 execute on Metal GPU:

- **S1 — Config parsing.** Parse `vision_config`, the four image/vision token ids, and
  `text_config.rope_parameters.{mrope_section, mrope_interleaved}` in `qwen35_config.rs` (stop
  dropping them). Gate: unit test round-trips the 0.8B config fields.
- **S2 — Weight loading.** Load the 153 `model.visual.*` tensors (fp16 and q4) into `VisionWeights`.
  Gate: loads `qwen3.5-0.8b` and `qwen3.5-0.8b-q4` without error; tensor-count and shape assertions.
- **S3 — ViT forward (split by Amendment 1 into S3a CPU reference + S3b Metal port).**
  S3a: a CPU-reference forward over the real depth-12 / hidden-768 weights. Gate: cosine > 0.999
  vs the HF pre-merger hidden states on the fixed golden image, enforced in required CI (the gate
  must fail loud, never skip-pass, when the checkpoint is absent on an enforcing runner).
  S3b: the Metal port of the same forward. Gate: matches the S3a CPU reference at cosine > 0.999
  on the same golden image, under the GPU lock.
- **S4 — Merger + image-token expansion.** Run `merge_and_project` on real weights; expand
  `image_token_id` placeholder spans in the input pipeline to the merged `visual_tokens` count.
  Gate: token-stream shape matches the HF processor for the same image + prompt.
- **S5 — Metal visual-embedding injection + decoder M-RoPE.** Replace the
  `forward_step(0, …)`-and-discard stub (`metal_qwen35.rs:11492-11498`) with residual-stream
  injection of merged patch embeddings at image positions, and add the 3-axis M-RoPE from
  Decision §2 to the 6 GQA layers. Gate: end-to-end greedy first-N token parity vs HF Qwen3.5-0.8B
  on a fixed image + prompt (the e2e-parity discipline).
- **S6 — Chat/serve + Studio wiring.** Extend `ChatMessage` content with the optional image part;
  wire the Metal worker to call vision encode + `generate_multimodal`; add the Studio composer
  affordance and bundled-image smoke test. Gate: an image can be attached in Studio and produce a
  grounded answer, live, on Metal.

Verification spine throughout: differential-test-first for any cross-framework question, the GPU
serialization lock for all Metal measurement runs, mutation-sensitive parity tests, and no
performance claim without a measurement from the same session.

**Deferred** (unchanged from ADR-049 unless noted): dynamic-resolution tiling, multi-image inputs,
video frame input, alternative ViT backbones, ViT-output caching across decode steps, and vision
LoRA. DeepStack is not applicable to this checkpoint (`deepstack_visual_indexes: []`).

## Changes to ADR-049

ADR-049 remains the historical record of the vision scaffold. This ADR revises the following
load-bearing points; on acceptance, ADR-049's Status line gains a pointer
"(vision geometry + R3 RoPE assumption revised by ADR-069)":

1. ViT geometry: depth 27 / d_model 1152 → depth 12 / hidden 768 / out_hidden 1024.
2. Weight-key schema: `vision_model.*` → `model.visual.*` (decoder `model.language_model.*`).
3. Memory tier: 32 GB target → 16 GB base tier and below (~1.6 GB fp16 whole model).
4. R3 (positional encoding): image tokens do **not** bypass decoder RoPE; the decoder applies
   3-axis interleaved M-RoPE to image-token positions in the 6 GQA layers (Decision §2).
5. Metal scope: ViT and all downstream stages run on Metal GPU (not a deferred v1 item).

## Risks

- **R1 — ViT Metal kernel correctness.** Same as ADR-049 R1: validate ViT output against the HF
  reference at cosine > 0.999 on a fixed image before merging S3.
- **R2 — M-RoPE position-id correctness.** The 3-axis `get_rope_index` bookkeeping is the highest
  new-scope risk. Mitigation: differential-test-first against HF `get_rope_index` on a fixed
  image + prompt (compare per-token position ids and the resulting logits at image positions) before
  writing the Metal frequency splice; e2e greedy token parity at S5 is the acceptance gate.
- **R3 — GQA-only application.** Applying M-RoPE to the wrong layer set silently corrupts output.
  Mitigation: assert at construction that position embeddings are threaded only into the 6
  `full_attention` layers and never into the 18 `linear_attention` layers; a mutation test that
  wires M-RoPE into a GDN layer must fail parity.
- **R4 — Text-path regression.** The M-RoPE path must collapse to the current 1-D RoPE for text-only
  sequences. Mitigation: a parity test asserting bit-for-bit (or < 1e-5) equality of text-only
  logits before and after the M-RoPE change; the axes-collapse argument in Decision §2 is the reason
  this must hold.
- **R5 — Weight-key drift.** Enumerate the actual `model.visual.*` prefixes; emit a clear error on
  unrecognized vision keys.
- **R6 — `image` crate decode performance.** Unchanged from ADR-049 R5: ~5–20 ms CPU decode per
  image, acceptable for interactive use, documented.

## References

- ADR-049 (vision encoder scaffold — revised here), ADR-007 (RoPE), ADR-009 (model architectures),
  ADR-010 / ADR-059 (attention taxonomy), ADR-003 (safetensors loading), ADR-063 (serving
  architecture).
- Reference implementation: `transformers` `models/qwen3_5/modeling_qwen3_5.py`
  (`apply_interleaved_mrope`, `get_vision_position_ids`, `get_rope_index`).
- Checkpoint: `Qwen3.5-0.8B` `config.json` (`vision_config`, `text_config.rope_parameters`) and
  `model.safetensors.index.json` (153 `model.visual.*` tensors).
- Lattice code: `crates/inference/src/vision/`, `metal_qwen35.rs:11318` (`generate_multimodal`),
  `metal_qwen35.rs:11492-11498` (injection point), `qwen35_config.rs` (config parser),
  `forward.rs:417-422` (1-D partial RoPE).

## Amendment 1 — S3 split into S3a (CPU reference) + S3b (Metal port) (2026-07-15)

The original Scope labeled S3 "Metal ViT forward" with a single cosine gate under the GPU lock.
Implementation review established that the stage bundles two independently verifiable deliverables
with different failure modes, and that landing them as one unreviewable unit is worse than staging
them:

- **S3a — CPU-reference ViT forward.** The convention-bearing work (temporal-patch fold order,
  block-major spatial-merge ordering, bilinear position-embedding interpolation, rotate-half
  2-axis vision RoPE, biased LayerNorm, GELU-tanh) lands as a plain-Rust forward gated at
  cosine > 0.999 against the committed HF pre-merger golden (from the S-goldens fixtures).
  The gate is a required CI job that provisions the pinned checkpoint and runs with enforcement
  armed — a runner without the checkpoint fails loud rather than skip-passing. This is the
  numerical contract every later stage (and the Metal port) verifies against.
- **S3b — Metal port.** The same forward on Metal, gated against the S3a CPU reference at
  cosine > 0.999 on the same golden image, under the machine GPU lock (`gpu_test_lock()`).
  No Metal-side convention decisions are permitted — divergence from the CPU reference is a bug
  by definition.

R1's "before merging S3" reading maps to: the cosine-vs-HF criterion binds S3a; S3b's criterion is
parity with the verified CPU reference. S4-S6 stage definitions are unchanged; S5's injection work
composes with S3b (Metal) for the end-to-end path but may develop against S3a outputs.
