# ADR-100: Loading foreign quantized checkpoints without a foreign runtime

**Status**: Proposed (2026-10-14)\
**Date**: 2026-10-14\
**Crate**: lattice-inference (format adapters); consumers: `lattice` CLI, serving binaries, the macOS app's model pull

## Context

Most quantized models people download for Apple silicon are published in one of two layouts that lattice
cannot load today: MLX quantized safetensors (the `mlx-community` organization) and GGUF. The macOS app's
model pull is meant to offer what the engine can load, so the engine needs (1) a way to load those
checkpoints and (2) a fast, honest answer to "can this repository be loaded" before anything is downloaded.

Hard constraints: no foreign runtime (no MLX, llama.cpp, ONNX Runtime, PyTorch), no `unwrap()` in library
code, leaf crates stay leaf, and no new crate without approval. Everything below is a reader plus a repack
into formats lattice already executes.

### What lattice has today (source-read at `main` 3ec4a327)

| #   | Fact                                                                                                                                                                                                                                                                                                                                                                                                                                  | Pointer                                                                                                                                                  |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| L1  | Format detection returns `Safetensors` whenever `model.safetensors` or its index exists. An MLX quantized repository is such a directory, so it is routed to the floating-point loader, which then meets `U32` weights.                                                                                                                                                                                                               | `model_format.rs` `detect_format`                                                                                                                        |
| L2  | The native Q4 block is 32 weights in 20 bytes: f16 scale, f16 bias, 16 bytes of nibbles. Dequantization is `w = nibble * scale + bias`. Byte `b` holds element `2b` in its low nibble and `2b+1` in its high nibble.                                                                                                                                                                                                                  | `weights/q4_weights.rs` module header (line 23) and `Q4Block`                                                                                            |
| L3  | Every Q4 consumer is affine and independent of the scale's sign: the Metal kernels `gemv_q4_decode`, `gemm_q4`, `gemm_q4_tiled` and `lm_head_block_topk_q4`, and the five CPU dequantization sites. No consumer reads the scale's sign or magnitude except through `nibble * scale + bias`.                                                                                                                                           | `forward/shaders/qwen35.metal`, `forward/shaders/gemm_q4_tiled.metal`, `forward/metal_qwen35.rs`, `forward/moe_expert_cache.rs`, `weights/q4_weights.rs` |
| L4  | Q4 ingress refuses a zero, negative or non-finite block scale and a non-finite bias. An f16 subnormal scale is positive and is accepted.                                                                                                                                                                                                                                                                                              | `weights/ingress.rs` (`NativeQ4Block` arm); `q4_weights.rs` `validate_q4_block_metadata`                                                                 |
| L5  | The Metal Q8 path consumes 34-byte Q8_0 blocks: f16 scale plus 32 signed bytes.                                                                                                                                                                                                                                                                                                                                                       | `qwen35.metal` (Q8_0 block comment); ADR-072 W2                                                                                                          |
| L6  | A Q3 kernel exists: 16 bytes per 32 weights, plane-split, `w = q * scale + bias`.                                                                                                                                                                                                                                                                                                                                                     | `qwen35.metal` Q3 block                                                                                                                                  |
| L7  | Native Q4 conversion quantizes projections, MLP weights, embeddings and `lm_head`.                                                                                                                                                                                                                                                                                                                                                    | `bin/quantize_q4.rs` `should_quantize`                                                                                                                   |
| L8  | The Q4 directory loader reads every projection as a `.q4` file; `.f16` files are read only for small tensors. A projection has no per-tensor floating-point fallback.                                                                                                                                                                                                                                                                 | `forward/metal_qwen35.rs` `from_q4_dir` (`q4_tensor_path` call sites)                                                                                    |
| L9  | The Q4 loader enforces the tied head both ways: `tie_word_embeddings=false` requires `lm_head.weight.q4`, and `true` refuses one if present. The quantized `embed_tokens` serves as the head when tied.                                                                                                                                                                                                                               | `metal_qwen35.rs` `from_q4_dir` (logits-head contract)                                                                                                   |
| L10 | A Gemma 4 checkpoint in the Q4 format has no serving route and is refused before any loader runs. Gemma 4 is served from floating-point safetensors.                                                                                                                                                                                                                                                                                  | `bin/lattice/main.rs` (serve path, `serving_provider::select`)                                                                                           |
| L11 | The Qwen3.5 vision tower is loaded separately from the decoder, from the same directory. With `quantize_index.json` present, every listed `model.visual.*` tensor is read from a `.q4` file (dequantized to f32 at load) or an `.f16` file, and an inventory that differs from `vision_config` (missing, unexpected, duplicate or misshapen) refuses. The Metal serving worker loads it lazily when the checkpoint advertises vision. | `vision/checkpoint.rs` `load_qwen35_vision_weights`; `serve/metal_worker.rs` (vision state)                                                              |

### What the foreign formats are (external survey, upstream source pinned by commit)

The layouts below come from a survey that read upstream source at pinned commits (mlx `36969c09`, mlx-lm
`ba8a08fd`, llama.cpp `a0e6dc73`, ggml `3612c0af`, candle `12bb5d1f`) and built 17 byte-level fixtures. A
spot check decoded 9 of the 17 fixtures with decoders written independently from the same pinned source and got
exact equality: Q4_0, Q4_K, and MLX affine at 1, 2, 3, 4, 5, 6 and 8 bits; a wrong-rule decode of the 3- and 6-bit
fixtures mismatched as it should. Q8_0, Q5_K, Q6_K, the MLX floating-point modes and the container fixture are
unverified, and the fixtures are constructed, not upstream test outputs. Treat this section as claims to re-check
in the first implementation PR.

- **MLX affine.** For a linear module `p` with logical weight `[O, K]`: `p.weight` is `U32 [O, K*bits/32]`,
  `p.scales` and `p.biases` are `[O, K/group]` in the original floating dtype, and `w = scale * q + bias`.
  `p.biases` (the quantization intercept) is distinct from `p.bias` (a linear layer's output bias). Codes form
  one continuous least-significant-bit-first stream along the last dimension, read from little-endian 32-bit
  words, with no padding at word boundaries. Bits 1, 2, 3, 4, 5, 6, 8; groups 32, 64, 128. **Scales can be
  negative**: the quantizer picks its range edge by magnitude and can negate the scale. Embeddings are quantized
  the same way and a tied output head uses the quantized embedding. Mixture-of-experts layers store rank-3
  `[E, O, K]` switch tensors. Tensors of one module may sit in different shards.
- **MLX configuration.** A top-level `quantization` object holds global `bits`, `group_size` and an optional
  `mode` (absent means affine), plus per-module dictionaries keyed by module path that are not merged with the
  global values (a per-module `false` leaves the module unquantized). Mixed recipes serialize as per-module
  entries. `quantization_config` without `quantization` is a different, foreign branch (AWQ, GPTQ,
  compressed-tensors, FP8, and others), and it can also sit in `text_config.quantization_config` or in a
  separate `hf_quant_config.json`. Some MLX conversions carry both keys. Non-affine modes (MXFP4, NVFP4, MXFP8)
  carry `U8` scales and no biases. The native safetensors reader also accepts `F8_E4M3` and `F8_E8M0` dtypes.
- **Observed defaults.** The survey read seven configurations: `Qwen3-0.6B-4bit`, `Qwen3.5-4B-MLX-4bit`,
  `Qwen3.5-9B-MLX-4bit`, `gemma-3-1b-it-qat-4bit`, `gemma-4-31b-it-4bit`, `all-MiniLM-L6-v2-4bit` and
  `Qwen3-Embedding-0.6B-4bit-DWQ`. The spot check read an eighth, `gemma-4-e4b-it-4bit`, which carries both
  `quantization` and `quantization_config`. All were 4 bits, group 64, affine, several by omitting `mode`.
  `Qwen3-4B-4bit` remains unread. The Qwen3.5 and Gemma 4 cards name
  MLX-VLM conversion paths; their tensor naming was not audited.
- **GGUF blocks.** Q4_0 is 18 bytes per 32 weights (f16 `d`, then 16 bytes where byte `j` holds elements `j`
  and `j+16`), `w = d * (q - 8)`. Q8_0 is 34 bytes (f16 `d`, 32 signed bytes). Q4_K is 144 bytes per 256
  weights in eight 32-weight sub-blocks with six-bit scales and minimums, `w = d*sc*q - dmin*m`. Within each
  64-weight segment, the low nibbles of 32 bytes hold the first sub-block and the high nibbles the second. Q5_K
  and Q6_K add high-bit planes (Q6_K has signed 8-bit sub-scales). GGUF lists dimensions innermost first.
  Candle's GGUF reader accepts lossy UTF-8, overwrites duplicate keys and does not validate a zero alignment; a
  reader here must not copy those choices.

## Decision

Phase 1 ships MLX affine 4-bit for Qwen3.5 through the Q4 path. Gemma 4 MLX checkpoints are detected and refused
with a message that the Gemma 4 Q4 route is pending (L10). GGUF is phase 2 and ONNX-format weights phase 3, each
behind the same detection and probe surface. Each fork below records the options, the recommendation, and what
would change it.

### D1: detection names the format, and a foreign quantized directory is never read as floating point

Add `ModelFormat` variants (the enum is `#[non_exhaustive]`) decided before any tensor I/O. The header set is
every shard `model.safetensors.index.json` lists when the index exists, and `model.safetensors` otherwise. No
file escapes inspection: with an index, a `.safetensors` file the index does not list refuses; without one,
`model.safetensors` must be the only `.safetensors` file in the directory, the rule the vision loader already
applies to an unindexed directory (`vision/checkpoint.rs`). A `.safetensors` symbolic link that resolves to a
shard the index lists counts as listed; a link that resolves anywhere else, and a plain copy of a listed shard
under another name, refuse (`a_symlink_to_a_listed_shard_is_not_an_unlisted_file`).

`config.json` is read once, with the size bound the loaders use. A `config.json` that is missing, unreadable,
unparsable, oversized or not a JSON object declares nothing, and detection proceeds on the files and
`hf_quant_config.json`: a foreign declaration with no native `quantization` object still refuses under rule 2, a
quantized checkpoint is still refused by its dtypes under rule 4, and a floating-point checkpoint stays
`Safetensors` for its loader to report the broken `config.json` (`an_unreadable_config_declares_nothing`). A
`quantization` value that is present but not an object refuses (`a_non_object_quantization_refuses`).

The rules apply in order, and the first that matches decides:

1. `Gguf`: a `.gguf` file and no safetensors file. A directory holding both refuses as ambiguous.
2. A foreign quantization declaration with no native one refuses, naming what was found: a
   `quantization_config` at the top level, in `text_config`, or in `hf_quant_config.json`, with no `quantization`
   object at the top level or in `text_config`. A native-shaped `quantization_config` alone is included, until a
   compatibility policy for it is written.
3. `MlxQuantized`: a `quantization` object at the top level of `config.json` or in `text_config`, whatever its
   mode. When both `quantization` and `quantization_config` are present, `quantization` governs. Module claims
   are evaluated only inside this rule, from the header set:
   - an affine module is a `U32` `<p>.weight` with `<p>.scales` and `<p>.biases` of one floating dtype (F16, BF16
     or F32);
   - a non-affine module is a `U32` `<p>.weight` with `<p>.scales` of dtype `U8`, `F8_E8M0` or `F8_E4M3`, no
     `<p>.biases`, and an optional `F32` `<p>.global_scale`. Non-affine modules are claimed here and refused by
     mode in D6, so a per-module MXFP4 override is reported as MXFP4, not as an unknown dtype;
   - a partial or mixed group refuses: `<p>.scales` or `<p>.biases` without a `U32` `<p>.weight`, a `U32`
     `<p>.weight` with floating `<p>.scales` and no `<p>.biases`, `<p>.biases` beside non-floating scales, scales
     and biases of different dtypes, and a `<p>.global_scale` on an affine module;
   - after claiming, any tensor of dtype `U32`, `U8`, `I8`, `I32`, `F8_E4M3` or `F8_E8M0` that no module claims
     refuses, and so do a `<p>.global_scale` that no non-affine module claims and a `quantization` object with
     no claimed module.
4. With no `quantization` object and no foreign declaration, any tensor of dtype `U32`, `U8`, `I8`, `I32`,
   `F8_E4M3` or `F8_E8M0`, and any tensor named `*.global_scale`, refuses, whether or not its neighbours look
   like an MLX module. Shape inference never
   stands in for a missing declaration.
5. Otherwise `Safetensors`, as today.

None of the refusals falls through to `Safetensors`.

Rejected: inferring MLX from the repository name or the `format=mlx` safetensors metadata alone. Names lie,
and the metadata does not say which modules are quantized.

### D2: one repack library; a converter first, a direct loader second, native MLX kernels only on evidence

The core is one library function per source layout that produces lattice's existing `Q4Tensor` from the foreign
tensors. Where it runs:

- **(a) Converter** (recommended first): `lattice convert --from mlx <dir> --out <dir>` writes a native `.q4`
  directory plus a conversion report. The existing Q4 loader, kernels and quality gates apply, and the output is
  memory-mapped like any native `.q4` file. Peak disk use is the source plus the output; the converter checks free
  space for the predicted output size before it starts. The converter never deletes the source.
- **Source removal** is a separate, explicit action in the app, offered only when the conversion report shows a
  verified conversion: every written `.q4` and `.f16` tensor re-read (the `.q4` files through Q4 ingress
  validation) and compared against its source under M4's first check; `config.json` and every tokenizer and
  generation file the loaders read either copied byte for byte or, for `config.json`, written by D7's rules and
  re-parsed; `quantize_index.json` re-read and matched one to one against the written files; the vision loader's
  inventory check passed when a tower was kept; and no tensor dropped (D7). The conversion is exact only for an f16
  source. With bf16 or f32
  metadata it carries D5's casts, so the offer states the D5 numbers instead of calling the conversion lossless. A
  probe result alone never qualifies, since the probe reads no payloads.
- **(b) Direct loader** (second): read the MLX directory and repack at load, no second copy on disk. Same library
  function; ships once (a) has passed the parity checks below. It gives up memory-mapping the weights, since
  repacked blocks live in owned buffers.
- **(c) Native MLX-layout Metal kernels** (group 64, separate scale and bias arrays): deferred until M6 shows a
  decode-throughput gap worth a second Q4 kernel family; see D3.

What would change this: M5 exceeding its registered threshold moves (b) ahead of (a).

### D3: group 64 and 128 are split into 32-weight blocks; the cost is stated, not hidden

A group of 64 becomes two `Q4Block`s with the same scale and bias. At 4 bits the nibble bytes copy across
unchanged (L2 matches MLX's packing byte for byte), so the codes are exact. Cost: 5.0 bits per weight in
lattice's layout against 4.5 in the MLX file at group 64 (f16 scale and bias), about 11% more weight bytes and,
for a bandwidth-bound decode, up to that much more decode time. Kernel work (D2 c) is justified only if M6 shows
that gap.

### D4: scale sign and zero

- **Negative scale: admitted at Q4 ingress.** The ingress rule changes from refusing `scale <= 0` to refusing
  `scale == 0`, a non-finite scale or a non-finite bias. With f16 metadata the repack is then bitwise exact for
  every nonzero scale: codes, scale and bias are copied. L3 shows every current consumer computes
  `nibble * scale + bias` with no dependence on the sign. The guard is a negative-scale fixture run through each
  consumer named in L3 (the four Metal kernels and the five CPU sites), each with a must-fail control, landed with
  the ingress change. The `Q4Block` documentation states the widened contract: the scale may be negative, a
  block's minimum and maximum weight are `bias` and `bias + 15*scale` in either order, and a consumer that needs
  a range, an ordering or a magnitude (clamping, a lookup table, a dynamic range estimate) takes it from those two
  values, never from the sign of the scale. A future consumer passes the same fixture before it lands.
- **Zero scale.** Every weight in the group equals its bias. Repack with all codes set to 0 and scale 1.0, which
  reproduces `w = bias` exactly and passes ingress. The source codes carry no information when the scale is zero.
- Rejected: re-expressing a negative scale as `q' = 15 - q`, `scale' = -scale`, `bias' = bias + 15*scale`. It keeps
  the old ingress rule but is not exact. The rounding of `bias'` is up to half an f16 ulp of `bias'` itself, and
  `bias'` can sit in a higher binade than every weight of a 32-weight block (a split group need not contain code
  15). For example, with the f16 values `bias = 14` and `scale = -2050/1024`, `bias'` is `-16414/1024` and rounds
  to `-16416/1024`, so a block whose weights are near `-14/1024` takes an error of `2^-9` on each weight. When weights cluster away from zero the error approaches a full quantization
  step. It can also overflow f16 where every weight is finite.

### D5: scale and bias dtypes

MLX stores scales and biases in the model's dtype (f16, bf16 or f32); the safetensors header is the authority, not
`config.json`. Repack casts to f16. For an f16 source the cast is the identity. For other sources each group is
judged from its metadata alone, with `ds = f16(s) - s`, `db = f16(b) - b`, and `u = ulp_f16(f16(b))`. The cases
are taken in order:

- A non-finite source scale or bias, or an f16 result that is not finite: refuse the tensor.
- A source scale of exactly zero: the D4 zero rule, with `|db|` reported in units of `u`.
- A nonzero scale that underflows to zero: the D4 zero rule applies if `f16(b + 15*s) == f16(b)`, evaluated exactly
  (as the test that `b + 15*s` lies inside the rounding interval of `f16(b)`, with round-to-nearest-even at its
  ends). Otherwise refuse the tensor. Because `b + q*s` is monotone in `q` and rounding is monotone, the endpoint
  test decides all sixteen codes. A distance bound is not used, because the rounding interval is asymmetric at a
  power of two: with `b = 2^-13 - 0.20*2^-24` and `s = -0.03*2^-24` in f32, `15*|s| + |db|` is below half an ulp of
  `f16(b) = 2^-13`, yet `b + 15*s` rounds down to `2^-13 - 2^-24`.
- Every other group: the worst-case reconstruction error over the sixteen codes is `15*|ds| + |db|`. The two terms
  are reported separately in their own units, because they scale with different things: `15*|ds|/|s|` in
  quantization steps, and `|db|/u` in f16 ulps of the intercept. Dividing `|db|` by `|s|` would report a group whose
  weights cluster far from zero as many steps wrong when each weight moved by half an ulp. `|db|/u` is at most 1/2
  by construction, so it is reported, never thresholded.
- Counted for every group, whichever case above it took: biases that underflow to zero (error at most `2^-25`), and
  scales that land in the f16 subnormal range.

Phase 1 refuses on a non-finite value, an unrecoverable underflow, and the scale rule registered here: a group
whose `15*|ds|` exceeds a quarter of `|s|` refuses its tensor. For a normal f16 scale `15*|ds|/|s|` is at most
`15*2^-11`, below 0.008 steps, so the rule can fire only on a scale that lands deep in the subnormal range, where
the cast loses a visible part of the step. M2 measures how often it fires; it does not move the threshold.

### D6: widths, modes and per-module resolution

- **Per-module resolution**, following mlx-lm at the pinned commit:
  1. No `<p>.scales` in the header set: the module is unquantized. An explicit entry for it other than `false`
     refuses.
  2. Entry `true`: the global parameters, with the mode's defaults for any field the global object omits or sets
     to `null`.
  3. Entry is a dictionary: its own fields, and the mode's defaults (not the global values) for omitted or `null`
     fields. The defaults are affine 4/64, MXFP4 4/32, NVFP4 4/16 and MXFP8 8/32.
  4. No entry but `<p>.scales` present: inferred from the saved shapes as mlx-lm's `infer_quant_config` does,
     `bits = 32*P/K` and `group = K/S`, where `P` is the saved packed width, `S` the saved scale width, and `K` the
     module's logical input width. The mode follows from the scale dtype together with the inferred widths: a
     floating scale dtype (F16, BF16, F32) is affine. `U8`, `F8_E8M0` and `F8_E4M3` scales all count as byte scales,
     since mlx's safetensors reader loads the two F8 dtypes as `uint8` and `infer_quant_config` tests only for
     `uint8`. Byte scales are NVFP4 at (4, 16), MXFP4 at (4, 32) and MXFP8 at (8, 32), and any other combination
     refuses.

  `K` and `O` never come from the packed tensor, since `P` and `S` alone fix only the product `bits*group`. They come
  from the family's architecture shape table, computed from `config.json` (hidden size, intermediate size, head
  counts and dimensions, vocabulary size, expert count). A module the table does not cover refuses in the
  converter. The probe, which also runs on families phase 1 refuses, reports such a module's recipe as unverified
  instead of guessing.

  Every resolved module, in every case, must then satisfy all of these, or it refuses:
  - `32*P == K*bits` and `S*group == K`, as integer products, so no division can truncate a remainder;
  - `<p>.weight`, `<p>.scales` and (for affine) `<p>.biases` share every leading dimension, and the leading
    dimensions equal the table's `[O]` or, on a switch layer, `[E, O]`;
  - `(mode, bits, group)` is on mlx's list: affine with bits in {1, 2, 3, 4, 5, 6, 8} and group in {32, 64, 128},
    NVFP4 (4, 16), MXFP4 (4, 32), MXFP8 (8, 32);
  - the mode agrees with the module's D1 claim: affine needs the affine triad, and a non-affine mode needs byte
    scales and no biases. A declared mode that contradicts the tensors refuses, since the shape checks alone cannot
    tell affine 4/32 from MXFP4.

  The `quantization` object is validated as a whole as well, whether or not any module takes case 2: its global
  mode must be one mlx names, and its global `(mode, bits, group)`, after defaults, must be on the list. When the
  object sits both at the top level and in `text_config`, the two must be equal. Phase 1 adds one rule of its own:
  a case-4 module whose inferred recipe differs from the global recipe refuses. mlx writes a module that departs
  from the global recipe as an explicit entry, so an inferred departure means the file and its declaration
  disagree.
- **Modules lattice reads as `.q4`** (L7, L8) must be affine 4-bit in the source, and they repack to `Q4Block`
  (exact up to D5). If such a module is floating-point in the source, or quantized at another width (1, 2, 3, 5, 6
  or 8 bits), phase 1 refuses and names the modules. Lattice does not requantize a floating-point module, since the
  result would no longer be the model the file describes, and it does not dequantize a quantized one.
  - The remedy is loader work: a per-tensor `.f16` or Q8_0 route for projections in the Q4 directory loader (L5,
    L8). That route is a phase-1 prerequisite if M3 finds such overrides in the target Qwen3.5 checkpoints, and
    later work otherwise.
- **Modules lattice reads as `.f16`** (norms and other small tensors) take the source floating-point values cast
  to f16, with the cast reported under D5. A linear layer's output bias `<p>.bias` is written the same way, split
  along the expert dimension on a switch layer. `<p>.scales` and `<p>.biases` are consumed into the module's
  `.q4` file.
- **Non-affine modes and foreign methods.** MXFP4, NVFP4, MXFP8 and every foreign `quantization_config` method
  refuse with a message naming the mode.
- **Gemma 4** checkpoints are detected and refused; the message says the Gemma 4 Q4 route is pending.

### D7: names, tied heads and multimodal towers, mapped per family from real headers

Phase 1 accepts Qwen3.5 checkpoints in the MLX-LM and MLX-VLM naming. Other families, including Qwen3 and the
`qwen3` embedding models, are detected as `MlxQuantized` and refused by family name. The mapping table is built
from the safetensors headers of the target repositories (M3), not from source reading alone.

- **Unknown names refuse.** A tensor name the mapping does not know refuses the conversion with the name.
- **Tied head.** The source flag is `text_config.tie_word_embeddings` and the top-level `tie_word_embeddings`; when
  both are present and disagree, the conversion refuses. Rules for one family do not carry to another: in mlx-lm
  at the pinned commit, Qwen3's flag decides whether a head exists and a tied load removes a present
  `lm_head.weight`, while Gemma3's sanitizer treats an absent head as tied. The survey read neither the Qwen3.5
  model definition nor MLX-VLM, so phase 1 knows neither Qwen3.5's head rule nor its default for the flag. Until
  M3 has read both the target headers and the upstream Qwen3.5 definition (mlx-lm and MLX-VLM, at pinned commits),
  the flag must be present explicitly, and the converter requires it and the tensors to agree:
  - flag `true` and no `lm_head.weight`: tied;
  - flag `false` and `lm_head.weight` present: untied;
  - flag absent: refuses, since the family default is unknown;
  - flag `false` and no `lm_head.weight`: refuses, since the head is missing;
  - flag `true` and `lm_head.weight` present: refuses, until M3 shows which head the upstream definition uses.

  A rule that infers tying from an absent head, or a family default for the flag, is adopted only when the
  family's upstream definition is read to say so. The converter writes `lm_head.weight.q4` if and only if the
  result is untied, and writes the flag at the top level, and in `text_config` whenever the output keeps one (L9).
- **The output directory** is a native Q4 directory: the converter writes the same `quantize_index.json` manifest a
  native Q4 conversion writes, listing every tensor it wrote, and `config.json` by these rules in every conversion,
  whether or not a tower was dropped. `quantization` and `quantization_config`, at the top level and in
  `text_config`, are written from the resolved recipe of the modules actually written and keyed by the output
  names, so no source-prefixed key (such as `language_model.`) survives and no entry names a module that was not
  written.
- **Multimodal towers.** The Q4 directory loader for the Qwen3.5 decoder reads no vision tensor; the vision tower
  has its own loader that reads the same directory (L11), and its `.q4` dequantization is one of the CPU sites L3
  already covers. So the Qwen3.5 vision tower can be served from a converted directory:
  - Its linear and embedding weights (including `pos_embed`) follow D6 like the decoder's, with one difference: a
    vision module stored floating-point is written as `.f16`, since this loader already takes `.f16` for any
    visual tensor. Such modules are expected, since mlx quantizes a module only when its input width divides the
    group size. A vision module quantized at a width other than 4 bits, or in a non-affine mode, refuses the
    conversion.
  - Layout is part of the mapping, not only the name. The loader requires the patch embedding as a rank-5
    `[O, C, T, H, W]` tensor, while mlx stores convolution weights channels-last. M3 records the source rank and
    axis order of every vision tensor, and the mapping table carries the transpose for each one that differs. A
    vision tensor whose layout M3 has not mapped refuses.
  - The source prefix (MLX-VLM's naming) is mapped to `model.visual.*` by the M3 table, every visual tensor is
    listed in `quantize_index.json`, and `vision_config` is kept. The loader's exact-inventory check against
    `vision_config` is the guard that the mapping is complete, and D2's verification runs it.
  - The direct loader (D2 b) serves text only until it has its own route to these tensors: the vision loader
    reads a converted directory or floating-point safetensors under lattice's names, and an MLX directory is
    neither. Until then the probe reports vision as dropped for the direct loader.
  - The audio tower, and any tower of another family, is not served and is dropped as below.

  A dropped tower changes the output `config.json` further:
  - `vision_config` and `audio_config` are removed as applicable;
  - when the source is a multimodal wrapper and the result is text-only, the `text_config` hyperparameters are
    promoted to the top level, and `model_type` and `architectures` are set to the text model's values from the
    family table;
  - the model is recorded as text-only.

  The report lists every dropped prefix, the probe reports served and dropped modalities before any download, and
  a conversion that dropped anything never qualifies for source removal (D2). Keeping the source when a tower was
  dropped costs disk: that is the price of not deleting data the converter did not carry.
- **Mixture-of-experts.** Rank-3 `[E, O, K]` switch tensors are split along the expert dimension into lattice's
  per-expert layout. A switch layer M3 has not mapped refuses.

### D8: GGUF (phase 2) is an independent reader

No dependency on candle or llama.cpp. The reader:

- requires strict UTF-8 and refuses duplicate keys and tensor names;
- validates alignment (nonzero, power of two), resolves the data base as the tensor-info end aligned up, and
  checks every offset with checked arithmetic;
- reverses GGUF's innermost-first dimensions and requires the innermost dimension to be a multiple of the block
  size;
- accepts little-endian only until a big-endian file is in hand.

Block mapping:

- Q4_0 maps to `Q4Block` with `scale = d` and `bias = -8*d`, nibbles reordered from GGUF's `j, j+16` pairing to
  sequential pairs.
  - A negative `d` is admitted (D4).
  - A zero `d` takes the D4 zero rule.
  - A non-finite `d`, or a `-8*d` that overflows f16, refuses the tensor.
- Q4_K maps to one `Q4Block` per 32-weight sub-block, with `scale = d*sc` and `bias = -dmin*m`, each computed in
  f32 and rounded once to f16.
  - The six-bit `sc` and `m` are unpacked from the 12-byte scale array by llama.cpp's `get_scale_min_k4` rule.
  - Each 64-weight segment is de-interleaved: the low nibbles feed sub-block `2k`, the high nibbles `2k+1`, and
    each is repacked into sequential pairs.
  - A non-finite `d` or `dmin` refuses the tensor, even where `sc` or `m` is zero (a product with a non-finite
    factor is not zero). Then a zero `d*sc` takes the D4 zero rule, and a non-finite or overflowing scale or bias
    refuses the tensor.
- Q8_0 maps byte for byte onto the Metal Q8_0 layout (L5) once the Q4 directory loader has a Q8_0 projection
  route. A non-finite `d` refuses the tensor; a zero `d` is copied as it is, since every weight of the block is then 0.
- F32, F16 and BF16 tensors (norms, biases, and any unquantized weight) follow D6's rules for floating-point
  modules: an `.f16`-read tensor is cast under D5's report, and a `.q4`-read module stored floating-point refuses
  until the projection route exists.
- Q5_K and Q6_K have no exact mapping onto a lattice block: Q6_K's 16-weight sub-scales do not fit a 32-weight
  Q8_0 block. Common `Q4_K_M` and `Q4_K_S` files carry them in some tensors. Phase 2 therefore needs either native
  kernels for them, or a measured requantization to Q8_0 through the Q8_0 route. M7 decides which, from the
  type census of real files. Until then a file containing them refuses.
- llama.cpp tensor names (`blk.N.attn_q.weight` and so on) and its architecture metadata map to lattice's
  configuration per family, under the same rules as D7.

### D9: ONNX-format weights (phase 3) through a hand-written initializer reader

Phase 3 reads weights out of `.onnx` files, including external-data files, with a minimal protobuf reader written
in the crate: initializers, their dtypes and shapes, and external-data locations, nothing else. The repository's
rules forbid an ONNX runtime and ONNX dependencies (`docs/CONTRIBUTING.md`, `AGENTS.md`, `CLAUDE.md`). A reader that
adds no dependency and executes no graph is a format adapter under the same rule as D1 to D8, not a runtime, so it
is in scope. Graph execution is not: the network is lattice's own forward pass, and the ONNX graph is read only to
map initializer names, under D7's refuse-unknown rule. Phase 3's block mapping is designed when phase 2 ships.

### D10: one loadability probe serves the CLI, the servers and the app

`lattice doctor` gains a mode that takes a repository's file list, `config.json`, `model.safetensors.index.json`
and `hf_quant_config.json` when present, and the safetensors headers (or a GGUF header), with no weight payloads.
It answers:

- accepted or refused, with the reason;
- the resolved recipe per module;
- the modalities served and dropped;
- the predicted resident size and output size.

Detection (D1), per-module resolution (D6) and name mapping (D7) are the same code the converter runs, so the
probe and the converter decide alike on everything the probe can see. The probe reads
no payload, so every acceptance is provisional on the converter's payload checks: non-finite or overflowing
values in any tensor it casts or repacks (D4, D5, D6, and D8's derived GGUF scales and biases), and, for a non-f16
source, D5's underflow and subnormal-scale rules. The probe says so in every acceptance. The app filters its pull
list with it.

## Measurement plan (registered before the dependent decisions are final)

M6 runs on the Metal host; everything else runs on the build host; nothing runs on a developer laptop. Header-only measurements (M1 to M3, and M7's census) use HTTP range
requests; M4, M5 and the rest of M7 download whole files. Each measurement names the tensors it reads.

- **M1** (informs D4), over the `scales` tensors of `Qwen3.5-4B-MLX-4bit` and `gemma-4-e4b-it-4bit`: the share of
  blocks with negative scale and with zero scale, per tensor and in total. D4 no longer depends on the numbers;
  they say how often the ingress change and the zero rule fire on real files.
- **M2** (decides D5 thresholds), over the `scales` and `biases` of every repository in M3's header inventory that
  stores them as bf16 or f32: D5's numbers per tensor. These are overflow, scale underflow and its recoverability,
  bias underflow, the subnormal count, the two error terms in their own units, and how many groups D5's
  quarter-step scale rule refuses. If no target stores non-f16 metadata, M2 adds one survey repository that does,
  found by header, and says so. M2 informs, it does not set the threshold: D5's rule is fixed in advance, and a
  target that it refuses is reported as a phase-1 refusal with the tensor named, not loosened to admit it.
- **M3** (decides D6's loader prerequisite and D7), over the headers and `config.json` of the M1 repositories,
  `Qwen3.5-9B-MLX-4bit` and one Qwen3.5 MoE MLX repository:
  - the complete tensor name, dtype and shape inventory, diffed against the names lattice's loaders consume;
  - every module that lattice reads as `.q4` but the file stores floating-point or at another width;
  - the scale and bias dtypes;
  - how `tie_word_embeddings` and `lm_head.weight` are serialized, and the head rule of the upstream Qwen3.5
    definitions in mlx-lm and MLX-VLM, read at pinned commits;
  - the multimodal prefixes present, their mapping to the names lattice's loaders read, and the dtype, width,
    rank and axis order of every vision tensor.
- **M4** (acceptance for D2 a), on one converted Qwen3.5 checkpoint:
  1. Written data against the source. Codes bitwise for every block with `f16(s) != 0`, whatever the metadata
     dtype. Scale and bias bitwise for an f16 source, and equal to `f16(s)` and `f16(b)` for any other. Blocks
     with `f16(s) == 0` (an exact-zero source scale, or a nonzero scale D5 found recoverable) checked as all
     codes 0, scale 1.0 and bias `f16(b)`. Every `.f16` tensor bitwise equal to `f16(source)`.
     `quantize_index.json` lists exactly the written files, and the vision loader loads the converted directory
     with its inventory check passing.
  2. Lattice's dequantization against MLX's `dequantize` of the original tensors, with the output dtype passed
     explicitly (f32) so bf16 metadata is not compared at bf16. Lattice computes `s*q + b` in f32 (L2). M4 records
     the precision MLX's own computation uses; the bound below assumes f32, and if MLX computes in a narrower type
     the comparison is made instead against an f64 reference computed from the source tensors, and M4 says so. The
     difference is measured in units of `U = ulp_f32(max(|s*q|, |b|))`: an operand-relative unit, because an
     output-relative one is meaningless where `s*q` and `b` cancel, and an f32 one, because the operands of a finite
     f16 block can exceed the f16 range (`s = 8192`, `q = 15`). Acceptance, in absolute terms: `|w_lattice - w_mlx|
     <= U` for an f16 source, and `<= 15*|ds| + |db| + U` otherwise.
  3. End to end, greedy tokens and perplexity of the converted checkpoint in lattice against the same checkpoint
     run in MLX, on the same prompts and corpus. This is a new reference arm; the existing parity harness compares
     against Hugging Face transformers. Acceptance is the first three greedy tokens equal on every prompt and a
     perplexity difference within 0.05. Repack is exact up to D5, so a larger gap points at a structural defect
     (naming, tying, layout), not rounding.
- **M5** (D2 ordering): converter wall time and peak memory on `Qwen3.5-4B-MLX-4bit`, then the largest phase-1
  model as a check that nothing scales worse than linearly. If conversion of the 4B model exceeds 60 s or 2 GiB of
  peak memory on the build host, (b) moves ahead of (a). The same run times a prototype of (b)'s repack at load,
  both per layer and whole-model, against a native `.q4` load of the same model; that comparison decides Q1.
- **M6** (decides D2 c): a prototype Metal GEMV reading the MLX group-64 layout (separate `U32` codes, f16 scales
  and biases) against `gemv_q4_decode`, at Qwen3.5-4B projection shapes on the Metal host, under the bench-window
  discipline. A median speedup of 7% or more for the prototype triggers D2 c. MLX's own decode throughput on the
  same host and checkpoint is recorded as context, not as an arm. Comparing a converted checkpoint with a native Q4
  conversion would measure nothing: both are 5.0 bits per weight on the same kernel.
- **M7** (phase 2, decides D8), on the Q4_0, Q4_K_S, Q4_K_M and Q8_0 files of `unsloth/Qwen3.5-4B-GGUF`:
  - the per-tensor `ggml_type` census;
  - the rates of zero, negative and overflowing scales and biases under the Q4_0 and Q4_K mappings, with each
    rounding error;
  - for Q5_K and Q6_K tensors, the error of a requantization to Q8_0. Registered rule, per file that carries them
    (Q4_K_S, Q4_K_M): the two arms differ only in those tensors, with every Q4_K tensor repacked identically in
    both. Requantization is adopted if the arm with requantized tensors has perplexity within 0.05 of the arm that
    dequantizes them natively (a CPU reference written from the format), and its first three greedy tokens agree
    on every prompt; otherwise native kernels.
  - acceptance for phase 2: each of the four converted files passes M4's checks against a CPU reference
    dequantization of the original file, with the end-to-end gate of M4 check 3 (first three greedy tokens equal,
    perplexity within 0.05).

## Acceptance criteria for phase 1

1. Detection (D1) tests:
   - `detect_format` returns `MlxQuantized` for the M1 repositories and for `Qwen3-Embedding-0.6B-4bit-DWQ`.
   - It never returns `Safetensors` for a directory holding any dtype D1 names, including a complete affine triad
     with no `quantization` object in `config.json`.
   - It reads single-file and sharded layouts, and refuses an unlisted shard, an orphan `scales` tensor, a weight
     with scales and no biases, an affine module with a `global_scale`, an orphan `global_scale`, an unindexed
     directory with a second `.safetensors` file, and a directory holding both `.gguf` and `.safetensors` files.
   - It claims a non-affine module whose scales are `F8_E8M0` and reports it by mode, through both a declared
     entry and case-4 inference.
   - It refuses a foreign `quantization_config` (top level, in `text_config`, and in `hf_quant_config.json`) and an
     unclaimed `F8_E4M3` tensor.
2. Converter and probe (D6, D7, D10) refusal tests:
   - an MXFP4 module;
   - a non-4-bit width and a floating-point module on a `.q4`-read path;
   - an unsupported family (`Qwen3-Embedding-0.6B-4bit-DWQ`, Gemma 4);
   - an unmapped tensor name;
   - case-4 inference that fails divisibility, leading-dimension equality or the mode list, and a module the
     shape table does not cover;
   - a declared mode that contradicts the tensors (affine 4/32 declared over byte scales), a global object off
     mlx's list, top-level and `text_config` objects that differ, and a case-4 recipe that departs from the
     global one;
   - an absent flag, top-level and `text_config` flags that disagree, an untied flag with no `lm_head.weight`, and
     a tied flag with one present;
   - for every conversion, kept tower or dropped: the output `config.json` holds no source-prefixed key and no key
     for an unwritten module in either alias, and `quantize_index.json` lists exactly the written files; for a
     dropped tower, the text model's hyperparameters sit at the top level;
   - a vision tensor whose source layout has no mapping.
3. Repack unit tests:
   - the survey's 4-bit affine fixtures reproduce as `Q4Block`s, each with a must-fail control (wrong bit order,
     wrong nibble order);
   - a zero-scale group;
   - a negative-scale group copied bitwise;
   - the per-module resolution cases of D6;
   - D5's cases on bf16 and f32 metadata: an exact-zero scale, an underflowing scale admitted and one refused
     (including the power-of-two counterexample in D5, which a distance bound admits), a group whose bias
     underflows still counted for its scale terms, the quarter-step scale rule firing on a deep subnormal scale,
     and the two error terms reported in their own units.
4. The negative-scale fixture passes through every consumer named in L3, with its must-fail control.
5. M4 passes on one Qwen3.5 checkpoint with its vision tower kept.
6. The probe and the converter agree on every M1 and M3 repository and on each refusal fixture.

## Known gaps

These are defects in the decisions above, found after the text was settled. Each is stated with what was checked. The implementing change fixes every one, and each is an acceptance item for that change, alongside the criteria above. Where a gap changes a rule, the rule given here governs over the text above.

- **K1. The vision inventory check reads no shapes on a Q4 directory.** `validate_qwen35_vision_weight_inventory` with `quantize_index.json` present checks that each `model.visual.*` name appears once and that its file opens. It reads no shape, so a misshapen visual tensor passes D2's source-removal gate and fails only when the server loads the tower. Fix: D2 verification and M4's first check load a kept tower through the full vision loader (`load_qwen35_vision_weights`), which checks every shape.
- **K2. M4's first check does not account for the mapping.** "Every `.f16` tensor bitwise equal to `f16(source)`" fails on a tensor D7 transposes and on a switch-layer bias D6 splits by expert, and comparing before the mapping would accept an untransposed tensor. Fix: compare against the source after D7's layout transpose and D6's expert split.
- **K3. A transpose may move a quantized tensor's packed axis.** D7 lets the mapping carry a transpose for any vision tensor. A transpose that moves the packed last axis of a quantized tensor scatters each 32-weight group across groups with different scales, which cannot be repacked without requantizing, and D6 forbids that. Fix: on a quantized tensor the mapping may permute only the other axes, and anything else refuses. mlx quantizes only modules that expose `to_quantized`, so the convolutional patch embedding is stored floating point and this is not known to occur.
- **K4. D6 has no case for a quantized module whose entry is `false`.** With `<p>.scales` present and `quantization["<p>"]` set to `false`, or to anything that is neither `true` nor an object, none of the four cases applies. A misspelled key such as `group` for `group_size` reads as omitted and takes the default. Fix: that combination refuses, and an unknown key in the global object or a per-module object refuses.
- **K5. The rule refusing an inferred recipe that differs from the global one is withdrawn.** mlx-lm infers recipes from saved shapes precisely for module paths absent from the per-module map of mixed-bit checkpoints (mlx-lm `9d1e356e`, its #1482), so an inferred departure does not mean the file and its declaration disagree. The rule would also refuse an inferred MXFP4 module in an affine checkpoint as a departure before D6 could refuse it by mode, contrary to acceptance criterion 1. Fix: an inferred recipe stands. It is judged by mode and width first, then against D6's list.
- **K6. The probe and the converter do not decide alike on everything the probe can see.** For a family phase 1 accepts, a module the shape table does not cover must refuse in the probe as it does in the converter; reporting it as unverified is only for families already refused by name. The header checks cover quantized modules only, so a misshapen floating tensor or a missing required tensor appears only after conversion. Fix: both check every floating tensor's shape after the mapping, and the complete mapped inventory (decoder tensors and every `vision_config` tensor), from headers.
- **K7. The converter's own output would be refused by D1.** D1's ordered rules never name the native Q4 format (today `detect_format` returns Q4 only when no safetensors file exists), and D7 writes `quantization` into the output `config.json`. Rule 3 then matches a converted directory that has no quantized safetensors module and refuses it. Fix: native Q4 detection comes before the configuration rules, and the converter writes neither `quantization` nor `quantization_config` into its output, since `quantize_index.json` is the native manifest. This also removes the unconditional write into `text_config` for an output that has none.
- **K8. Detection order and index validation.** The header-set rule is stated before rule 1, so a directory holding only a `.gguf` file has no defined header set. A directory with more than one `.gguf` file is not covered, and an index's `weight_map` is not required to be non-empty or to match the shard headers. Fix: Q4 and GGUF detection run before the header set is resolved. More than one `.gguf` file refuses. An index's `weight_map` must be non-empty and match the tensors in the shards it lists.
- **K9. D5's scale rule is unscoped and not monotonic.** Read as written, the quarter-step rule applies to a scale that underflows to zero (`15*|ds| = 15*|s|`) and refuses every such group. It also refuses a subnormal scale such as `s = 1.25 * 2^-24` beside `b = 1.0`, whose sixteen values all round to `f16(b)`, while admitting a smaller scale that underflows. Fix: the rule applies only where `f16(s) != 0`, and any group with `f16(b + 15*s) == f16(b)` takes the D4 zero rule.
- **K10. M4's second check needs a wider unit.** `U = ulp_f32(max(|s*q|, |b|))` is half of `ulp_f32(w)` when `w = s*q + b` crosses a power of two above both operands. Same-sign random samples break the stated bound by up to 1.5 `U` (91,504 of 2,000,000). Fix: `U' = ulp_f32(max(|s*q|, |b|, |w|))`, with acceptance `|w_lattice - w_mlx| <= 15*|ds| + |db| + 1.5*U'`. That is the analytic worst case when MLX rounds `s*q` before the add: one rounding of the sum on each side, plus MLX's rounding of the product (Lattice's product of an f16 scale and a 4-bit code is exact in f32). The measured maximum over 8,000,000 samples was `1.0*U'`, and M4 reports it.
- **K11. M7's phase-2 acceptance cannot hold as written.** D8's derived `Q4_K` scales `f16(f32(d) * sc)` and biases `f16(-f32(dmin) * m)` are not exact in f16, so the f16-source bound fails. A `Q5_K` or `Q6_K` tensor requantized to `Q8_0` cannot match the original's dequantization. Fix: acceptance per type. `Q4_0` is an exact reorder with `bias = f16(-8*d)`. `Q4_K` takes the non-f16 bound with `ds` and `db` from the derived-scale rounding. `Q8_0` blocks compare bitwise. A requantized tensor compares against the requantizer's own output, with quality gated end to end by M7.

## Consequences

- Qwen3.5 MLX 4-bit checkpoints run on lattice's existing kernels and gates in phase 1, at 5.0 bits per weight
  instead of the file's 4.5, and with a one-time conversion step. Both costs are measured (M5, M6) before anyone
  builds kernels to remove them.
- Q4 ingress admits finite nonzero negative scales. Every current consumer is sign-agnostic, and a fixture through
  each one guards that property. A future consumer must pass the same fixture.
- Gemma 4, other families, other widths, floating-point overrides on `.q4`-read modules and non-affine modes are
  refused with a reason in phase 1. For Gemma 4 the reason says its Q4 serving route is pending; an
  original-precision checkpoint is not a substitute for a large model on a laptop. Nothing in phase 1 silently
  dequantizes or requantizes, and nothing deletes a user's download.
- The probe makes "can lattice load this" a product answer instead of a download-and-see, including which
  modalities survive.

## Decided scope boundaries

- ONNX-format weights are in scope as phase 3 (D9).
- A Gemma 4 Q4 serving route is its own lane, under a serving decision record opened after phase 1 ships. Until
  then the phase-1 refusal stands, and its message says the Q4 route is pending.

## Open questions

- Q1: whether the direct loader (D2 b) should repack lazily per layer or everything at load; decided by M5's
  load-time comparison.
