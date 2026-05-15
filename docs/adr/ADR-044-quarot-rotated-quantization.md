# ADR-044: QuaRot — Hadamard-Rotated 4-bit Quantization

**Status**: Accepted
**Date**: 2026-05-15
**Crate**: lattice-inference

## Context

Existing 4-bit quantization in `lattice-inference` (`weights::q4_weights`) is naïve per-block symmetric round-to-nearest (RTN). RTN works for weight-only Q4 on well-behaved layers, but two failure modes are well-documented in the literature and present in Qwen3 weights:

1. **Outlier channels**: a small fraction of hidden-state channels carry weights with magnitudes 10-100× the median. Uniform per-block scaling either clips them (information loss in the outlier) or wastes dynamic range across the rest of the block (information loss everywhere else).
2. **Activation distribution skew**: extending Q4 to activations or KV cache fails — activations have heavier tails than weights, so the same uniform scaling is even more lossy.

QuaRot (Ashkboos et al., NeurIPS 2024, [arxiv:2404.00456]) proposes applying random Hadamard rotations to hidden-state spaces _before_ quantization. Rotations are orthogonal and computationally invariant — they can be absorbed into adjacent linear layers, so the model output is mathematically identical to the un-rotated model. But the rotated activations have **uniform-magnitude channels** (no outliers) because Hadamard projection spreads outlier energy across all dimensions.

Result on LLaMA-2-70B (paper): 4-bit weights+activations+KV with at most 0.47 WikiText-2 perplexity loss; 99% zero-shot retention. Lossless 6/8-bit.

The reference QuaRot implementation ([spcl/QuaRot](https://github.com/spcl/QuaRot)) is Python/CUDA. We did not find a public pure-Rust QuaRot implementation in a brief search; some Rust-adjacent quantization work exists (e.g., [konjoai/squish](https://github.com/konjoai/squish)'s `squish_quant_rs`) but is not QuaRot. The "first pure-Rust QuaRot" framing is plausible but unverified — claim only after a more thorough survey if it ever matters for external communication.

## Decision

Implement QuaRot v0 in `lattice-inference` as a new module `quant::quarot` (not a new crate — per CLAUDE.md the inference crate already holds weight-format and quantization code).

### Scope — v0 (multi-PR)

v0 is the offline rotation + Q4-weight-only path. Ships in 4 sequential PRs to keep each diff small and reviewable:

| Step | PR                             | Scope                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | PR #15 (merged)                | Walsh-Hadamard + RandomizedHadamard primitives (f32 + f64), unit-tested. No model code touched.                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| 2    | PR #18 (merged)                | Rotation absorption math: `absorb_input_rotation`, `absorb_output_rotation` (f32 + f64) for row-major weight matrices. Synthetic-linear-layer equivalence tests validate the identity `(W · R^T) · (R · x) = W · x` and the two-layer round-trip pattern. No SafeTensors I/O.                                                                                                                                                                                                                                                                    |
| 3a   | PR #19 (merged)                | `RotationPlan` data structure: which weight tensors absorb the residual-stream rotation, on which side, for Qwen3.5 hybrid (GQA + GDN + dense MLP + embed/lm_head). Includes `RuleRequirement::{Required, Optional}`, `validate_coverage()` (suffix-presence sanity check, NOT a correctness gate), and the full prose contract for what step 3c must additionally implement (RMSNorm fusion, config flip, lm_head materialization). See module doc in `quant::quarot::plan`.                                                                    |
| 3b   | PR #20 (merged)                | Streaming SafeTensors reader [`quant::quarot::io::QuarotTensorReader`] with single-file + sharded auto-detect, hand-rolled F32/F16/BF16→f64 decode (no `f16` feature dependency), no conversion cache — each `read_tensor_f64` allocates fresh. Promoted `qwen_required_tensor_names(cfg)` out of `#[cfg(test)]` at `model/qwen35/mod.rs:43-44` so the conversion binary can consume it for per-layer coverage checks.                                                                                                                           |
| 3c   | series in progress (5 sub-PRs) | `quantize_quarot` binary tying 3a + 3b + the rotation math from PR #18. Owns the step-3c contract spelled out under §"Step 3c contract" below (rotation absorption in f64, RMSNorm fusion, lm_head materialization, forward-equivalence refuse-on-fail, Q4 bridge decision). Sub-PRs: **3c-1 this PR (#25)** Q4 bridge entry points; 3c-2 (next) rotation + RMSNorm fusion pipeline; 3c-3 lm_head materialization + `tie_word_embeddings` config flip; 3c-4 forward-equivalence harness + refuse-on-fail; 3c-5 `bin/quantize_quarot` CLI wiring. |
| 4    | next                           | Bench: rotated-Q4 vs unrotated-Q4 perplexity delta on Qwen3.5-0.8B against WikiText-2 calibration. Acceptance: delta < 0.5 PPL. The number is re-measured on our pipeline; do not echo the paper's numbers as ours. Qwen3-0.6B is a step-4 target once the Qwen3 plan / required-name helper ship (see model-coverage table).                                                                                                                                                                                                                    |

**Step 3c contract.** The `quantize_quarot` binary owns ALL of:

- Rotation absorption in f64 (consume `RotationPlan` from step 3a, apply via the math from step 3b's reader feeding the math from PR #18).
- RMSNorm `(1 + gamma)` fusion at `input_layernorm` / `post_attention_layernorm` / `final_norm`. NOT at GDN's `linear_attn.norm` (see Risks).
- Tied-embedding untying when present.
- `tie_word_embeddings=false` config flip in the output.
- `lm_head` materialization with fused `(1 + g_final)` + rotation.
- Forward-equivalence assertion on a small batch: `‖rotated_forward − original_forward‖ < 1e-5`. Refuse-on-fail (no partial conversion artifacts written).
- Q4 bridge — **decision: path (b)**, landed in step 3c-1. Sibling entry points `quantize_f32_to_q4` (`q4_weights.rs:356`) and `quantize_f64_to_q4` (`q4_weights.rs:384`) sit alongside the existing `quantize_bf16_to_q4` (`q4_weights.rs:318`). Both entry points quantize at **f32 precision** — matching §Risks above ("keep rotation math in f64 [...] quantize in f32, store scales in f16 as before"). The f64 wrapper exists to avoid the BF16 round-trip in `quantize_bf16_to_q4` and to skip an intermediate `Vec<f32>` allocation at the call site; it is NOT a true f64 quantizer (`abs_max`, the scale reciprocal, and the per-nibble `round` all execute in f32). Values within ~½ ULP of an f32 representation may quantize to a different nibble than a hypothetical f64 reference would. Acceptable per §Risks — what matters is that we no longer truncate the rotation pass's output to 7 mantissa bits before quantizing. Measurement (`weights::q4_weights::tests::quantize_f32_to_q4_lower_error_than_bf16_path_on_high_precision_input`, 2048 synthetic f32 values uniform on `[-1, 1]`): path (b) mean abs err `0.034279`, max `0.071060`; path (a) f32→BF16→Q4 mean `0.034312`, max `0.071323`. Path (b) is strictly equal-or-better on max and strictly less on mean across the suite. Delta is small on uniform-rotated inputs but grows for inputs with concentrated absmax (a few outlier values whose BF16 rounding shifts the block scale for all 32 elements).
- Output: `.q4` file consumable by the existing inference path with no runtime changes (subject to the LoRA caveat in Risks).

**Out of v0 entirely (deferred to v1)**:

- INT4 activation quantization (online during forward pass)
- INT4 KV cache quantization
- INT4 attention scores
- Per-layer rotation seeds (v0 uses one global Hadamard for the residual stream + one per attention head dim)
- Calibration-data tuning (v0 uses random Hadamard only — the QuaRot paper shows this is competitive without learned rotations)

**Model coverage in v0**:

| Model                                                                 | hidden | GQA head_dim | linear-attn head_dim | v0 supported?                                                                                                                                                                                                                                     |
| --------------------------------------------------------------------- | -----: | -----------: | -------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3-0.6B (decoder) [^qwen3-0.6b]                                    |   1024 |          128 |                  n/a | ⏸ **pending** — dims are power-of-2 (Hadamard-friendly), but no Qwen3 `RotationPlan` or `QwenConfig`-typed required-name helper exists yet. Tracked in [issue #21](https://github.com/ohdearquant/lattice/issues/21). Becomes ✅ once those land. |
| Qwen3.5-0.8B (hybrid GDN+GQA) [^qwen35-0.8b]                          |   1024 |          256 |     128 (linear K/V) | ✅ all power of 2                                                                                                                                                                                                                                 |
| Qwen3-Embedding-4B (decoder, served by `QwenModel`) [^qwen3-embed-4b] |   2560 |          128 |                  n/a | ❌ **explicitly deferred** — `2560 = 2^9 · 5`, hidden not a power of 2                                                                                                                                                                            |
| BERT-base (encoder) [^bert-base]                                      |    768 |           64 |                  n/a | ❌ **explicitly deferred** — `768 = 2^8 · 3`, hidden not a power of 2                                                                                                                                                                             |

[^qwen3-0.6b]: HuggingFace config: <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config.json> (`hidden_size: 1024`, `head_dim: 128`). The Qwen3-Embedding-0.6B base shares the decoder backbone with Qwen3-0.6B used as a generation reference here. Note: as of step 3b the dims being Hadamard-friendly is necessary but not sufficient — see [issue #21](https://github.com/ohdearquant/lattice/issues/21) for the missing Qwen3-side plan and required-name helper.

[^qwen35-0.8b]: HuggingFace config: <https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json> (`hidden_size: 1024`, GQA `head_dim: 256`, linear-attention `linear_key_head_dim: 128` and `linear_value_head_dim: 128`).

[^qwen3-embed-4b]: HuggingFace config: <https://huggingface.co/Qwen/Qwen3-Embedding-4B/blob/main/config.json> (`hidden_size: 2560`, `head_dim: 128`).

[^bert-base]: HuggingFace config: <https://huggingface.co/google-bert/bert-base-uncased/blob/main/config.json> (`hidden_size: 768`, `num_attention_heads: 12` → `head_dim: 64`).

For non-power-of-2 hidden dims (4B, BERT), v1 cannot use naive zero-pad-then-project — `π ∘ H_N ∘ P` (project up to power of 2, apply Hadamard, drop padded coords) is **not** orthogonal on the original space and breaks the QuaRot invariant. v1 needs one of: (a) the QuaRot paper's online block-diagonal approach for non-power-of-2 dims (rotates within sub-blocks that are individually power of 2; see Ashkboos et al. 2024 §4 "Online Hadamard transformations" [^quarot-paper]), (b) Paley-construction Hadamard matrices that exist for some specific non-power-of-2 dims, or (c) Givens / Householder products that are orthogonal but not structured. Which path v1 takes is an open question.

[^quarot-paper]: Ashkboos et al., _QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs_, NeurIPS 2024. NeurIPS PDF: <https://papers.nips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf>; arxiv:2404.00456: <https://arxiv.org/abs/2404.00456>.

### Key Design Choices

**Hadamard size = power of 2 only.** v0 covers decoder models whose hidden and head dims are all powers of 2. Currently shipped: Qwen3.5-0.8B (hybrid GDN+GQA). Qwen3-0.6B's dims also qualify but the Qwen3 plan / required-name helper land separately under [issue #21](https://github.com/ohdearquant/lattice/issues/21). Models with non-power-of-2 hidden dims (Qwen3-Embedding-4B at 2560, BERT-base at 768) are **explicitly deferred** to v1; naive zero-pad + project-back is not orthogonal and so cannot be used. v1 will pick from block-diagonal partition (QuaRot paper's approach), Paley-construction Hadamards, or Givens products.

**Offline rotation, not on-the-fly.** v0 saves the rotated+quantized weights as a new `.q4` file. The forward pass loads it like any other Q4 model — no runtime rotation cost. Disk during conversion is the mmap'd BF16 input plus the smaller `.q4` output — no full-precision rotated intermediate. Trade is worth it because rotation is a one-time cost.

**Rotation absorption pattern.** For a linear layer `y = W · x` (bias-free):

- Pre-rotation: `y = W · R^T · R · x = (W · R^T) · (R · x)`.
- The previous layer's output gets `R · ·` absorbed (output-side); this layer's `W` becomes `W · R^T` (input-side). Both absorptions are free (one matmul during conversion, no runtime cost).

For QKV projections in attention, the rotation `R` is shared across Q, K, V (preserves dot-product). The output projection absorbs `R` on its output side to undo it.

The full per-tensor recipe for Qwen3.5 hybrid (GQA + GDN + dense MLP + embed/lm_head) — including the `[2 * q_dim, hidden]` Q + gate-z fused shape, GDN's `linear_attn.in_proj_*` / `out_proj` pattern, and the embedding storage `[vocab_size, hidden]` requiring input-side absorption — is in `quant::quarot::plan` (PR #19). The plan is **rotation-rule data only**; the conversion binary in step 3c additionally implements the non-rotation mutations listed in Risks below.

**Q4 block format unchanged.** The rotated weights still go through the same Q4_0 per-block scheme as `weights::q4_weights` — QuaRot's job is purely to redistribute outliers before quantization, not to change the quantized format. Forward pass is identical.

### Alternatives Considered

| Alternative                                               | Pros                                          | Cons                                                                                                   | Why Not                                             |
| --------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| New crate `lattice-quant`                                 | Clean boundary                                | CLAUDE.md says no new crates without approval; quantization code already lives in `inference::weights` | Adds workspace overhead with no abstraction benefit |
| Implement SpinQuant or learned rotations instead          | Better paper results                          | Requires training data + optimization loop                                                             | Out of scope for a pure-inference v0                |
| Skip rotation, just improve Q4 (e.g., per-channel scales) | Smaller change                                | Doesn't address outlier channels structurally                                                          | Doesn't reach paper-claimed accuracy at 4-bit       |
| GPTQ-style Hessian-based weight quantization              | Existing literature, possibly better accuracy | Requires calibration data + per-layer Hessian compute (~minutes per layer at scale)                    | v0 wants random Hadamard only — no calibration      |
| Activation quantization in v0                             | Bigger speedup                                | Requires fused INT4 GEMM kernels; ours don't exist yet                                                 | Defer to v1 with proper Metal/NEON INT4 kernels     |

## Consequences

These are consequences of accepting v0 as the design (across all 4 PRs), not of the primitives-only PR that lands first.

### Positive

- Once v0 is fully landed (step 4 PR): 4-bit weights targeting paper-claimed PPL delta < 0.5 vs F32 on Qwen3.5-0.8B (and Qwen3-0.6B once [issue #21](https://github.com/ohdearquant/lattice/issues/21) lands). The target number is the paper's; it must be re-measured on our pipeline before being claimed.
- Foundation for v1 INT4 activation+KV — the rotated activation distributions are quantizable; the un-rotated ones aren't.
- Pure-Rust QuaRot implementation. The reference [spcl/QuaRot](https://github.com/spcl/QuaRot) is Python/CUDA; some Rust-adjacent quantization work exists (e.g., `konjoai/squish`'s `squish_quant_rs`) but isn't a QuaRot implementation. So "first pure-Rust QuaRot" is plausible but not certain — verify before claiming externally.
- Knowledge-graph node `QuaRot` moves from `researched` → `implemented` when step 4 lands.
- Composes with future SpinQuant / learned rotations (same absorption infrastructure).

### Negative

- Hadamard transform requires power-of-2 dimensions; non-power-of-2 hidden sizes (4B, BERT-base) deferred to v1 with no naive padding option.
- One-time conversion cost (~minutes for Qwen3-0.6B; ~hour for 8B-class models). Disk during conversion is original BF16 (mmap) + smaller `.q4` output — no full-precision rotated intermediate (step 3c's pipeline does read → absorb → quantize → write in one pass).
- Public API surface lands incrementally across PRs #15, #18, #19 before any consumer exists in the codebase — locks in transform-order, precision, and naming before downstream code stresses the contract. Mitigation: matches existing `lattice-inference` convention (every module is `pub`), and the crate-level STABILITY doc already flags it `Experimental` with churn expected. If a downstream PR needs to change the primitive API, that change ships in the same PR.
- QuaRot-converted `.q4` files are NOT drop-in compatible with the runtime LoRA injection path — see Risks. v0 ships without runtime adapter support on QuaRot bases; adapter compatibility is a v1 follow-up.

### Risks

- **Numerical precision**: Hadamard transforms in f32 are stable, but interaction with f16 storage of scales (in `Q4Block::scale: u16`) may compound rounding. Mitigation: keep rotation math in f64 (PR #15 ships both f32 and f64 primitives), quantize in f32, store scales in f16 as before. Measure perplexity delta vs full-precision rotation in step 4.

- **Forgotten absorptions**: A rotation that doesn't have its inverse absorbed somewhere will change model output. Mitigation: assert computational equivalence on a small batch (identity tolerance < 1e-5) before saving rotated weights. Refuse-on-fail.

- **Residual stream vs head-dim mismatch**: Different rotations for hidden vs head-dim spaces. Need careful tracking. Mitigation: explicit `RotationPlan` type that records which dimension each rotation applies to + an end-to-end test asserting equivalence. v0 plan uses only the residual-stream rotation; per-head-dim rotations deferred to v1.

- **Shifted RMSNorm `(1 + gamma)` does not commute with Hadamard rotation.** Qwen3.5 applies shifted RMSNorm at `norm.rs:16` for `input_layernorm` (forward.rs:41), `post_attention_layernorm` (forward.rs:66), and `final_norm` (forward.rs:81). A diagonal scale `D = diag(1 + g)` does not commute with a dense Hadamard `R`. Mitigation (step 3c): fuse `(1 + g)` into the immediately-following linear layer as a column multiply (`W[i, j] *= (1 + g[j])`), then zero out the runtime `*_layernorm.weight` (set `gamma = 0` so the shifted formula returns 1). The normalize-only step IS rotation-invariant (`||R · x|| = ||x||`).

- **GDN's `linear_attn.norm` is structurally different and must NOT be fused.** Verified at `gdn.rs:425, 440` and `gdn_fused.rs:127`: GDN's gated-RMSNorm multiplies plain `gamma[i]`, NOT `(1 + gamma[i])`. More importantly, this norm runs INSIDE the GDN block between linear-attention compute and `out_proj` — the residual rotation enters at `in_proj_*` (input-side absorbed) and exits at `out_proj` (output-side absorbed) and does NOT cross this internal norm. A converter that tries to fuse the GDN norm will produce wrong logits.

- **Tied embeddings + final-norm fusion are mutually exclusive without surgery.** Qwen3.5 defaults to `tie_word_embeddings=true` (`qwen35_config.rs:177`); the loader at `loading.rs:266` only loads `lm_head.weight` when this flag is false. The tied tensor cannot simultaneously serve as the embedding (rows = `R · E[i]`) and as the lm_head after final-norm fusion (rows = `R · diag(1 + g_final) · E[i]`) — these require different matrices unless `g_final == 0`. Mitigation (step 3c): the converter MUST (a) materialize an untied `lm_head` (copy `embed_tokens` pre-rotation when input is tied), (b) fuse `(1 + g_final)` into the new `lm_head` as a column multiply, (c) input-side-rotate the result, (d) zero the runtime `final_norm.weight`, AND (e) flip `tie_word_embeddings=false` in the output config. Without (e), the runtime falls back to `embed_tokens` via `logits_weight()` at `weights.rs:51` and the fused `lm_head` is never consulted — silent wrong logits. Spelled out in `quant::quarot::plan` module doc §Tied embeddings.

- **`is_complete()` on `validate_coverage` is NOT a correctness gate.** The rotation-rule coverage check only verifies suffix-presence + non-ambiguity. The conversion binary's full-validity check must additionally verify per-layer coverage (against `qwen_required_tensor_names(cfg)`, promoted out of `#[cfg(test)]` at `model/qwen35/mod.rs:43-44` in step 3b), RMSNorm fusion applied to input/post/final norms, GDN internal norm NOT fused, `tie_word_embeddings=false` flipped, fused `lm_head` materialized, forward-equivalence delta below threshold.

- **Runtime LoRA injection is incompatible with QuaRot-converted models.** The forward path at `forward.rs:249, 467` and `gdn_fused.rs:385` adds LoRA delta to base matmul output using the same activation. Rotated base produces output in a different basis than an un-rotated adapter delta — sum is invalid. v0 marks QuaRot models as LoRA-runtime-incompatible. Mitigation: the saved `.q4` artifact must include rotation-seed metadata so a future LoRA-aware path can (a) rotate adapter weights on load using the stored seed, or (b) refuse-on-compose at adapter-load time when the base is QuaRot-converted. Neither is in v0.

- **Conversion-binary public-API surface (step 3c).** The `RotationPlan` exposed in PR #19 is rotation-rule data; the broader conversion contract (RMSNorm fusion, config flip, lm_head materialization) is NOT in the plan structure. Step 3c will introduce a `ConversionPlan` (or similar) wrapping the rotation plan plus the required mutations; only that combined structure is the correctness gate. Documented in `plan.rs` §Known gaps to prevent cargo-culting `RotationPlan::is_complete()` as a "ready to ship" check.

## References

- Paper: Ashkboos et al., _QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs_, NeurIPS 2024, [arxiv:2404.00456](https://arxiv.org/abs/2404.00456). NeurIPS PDF: <https://papers.nips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf>.
- Related:
  - SpinQuant — Liu et al., _SpinQuant: LLM Quantization with Learned Rotations_, 2024, [arxiv:2405.16406](https://arxiv.org/abs/2405.16406) (learned rotations, contrasted with QuaRot's random Hadamard).
  - Hadamard transforms in ML — Yu et al., _Orthogonal Random Features_, NeurIPS 2016 / _Structured Adaptive and Random Spinners_, AISTATS 2017, [arxiv:1610.06209](https://arxiv.org/abs/1610.06209) (cited by QuaRot for the spread-outlier-energy property of Hadamard projections).
- HuggingFace model configs referenced in the coverage table: Qwen3-Embedding-0.6B <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config.json>, Qwen3.5-0.8B <https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json>, Qwen3-Embedding-4B <https://huggingface.co/Qwen/Qwen3-Embedding-4B/blob/main/config.json>, BERT-base <https://huggingface.co/google-bert/bert-base-uncased/blob/main/config.json>.
- SafeTensors format reference (used by `quant::quarot::io`): official Rust crate source <https://docs.rs/safetensors/latest/src/safetensors/tensor.rs.html>; HF format docs <https://huggingface.co/docs/safetensors/index>.
- KG entities: `QuaRot (Ashkboos 2024)` (86ec6a4f), `Rotation-Based Quantization` (d534ca51), `QuantizationRankFlipCondition` (224ef1a7), `HAWQ`, `BRECQ`, `AWQ`, `SmoothQuant`, `HQQ`
- KG ontology: `.khive/taxonomy.md`
- Existing code: `crates/inference/src/weights/q4_weights.rs`, `crates/inference/src/bin/quantize_q4.rs`
