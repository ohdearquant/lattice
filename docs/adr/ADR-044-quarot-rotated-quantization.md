# ADR-044: QuaRot — Hadamard-Rotated 4-bit Quantization

**Status**: Accepted
**Date**: 2026-05-15
**Crate**: lattice-inference

## Context

Existing 4-bit quantization in `lattice-inference` (`weights::q4_weights`) is naïve per-block symmetric round-to-nearest (RTN). RTN works for weight-only Q4 on well-behaved layers, but two failure modes are well-documented in the literature and present in Qwen3 weights:

1. **Outlier channels**: a small fraction of hidden-state channels carry weights with magnitudes 10-100× the median. Uniform per-block scaling either clips them (information loss in the outlier) or wastes dynamic range across the rest of the block (information loss everywhere else).
2. **Activation distribution skew**: extending Q4 to activations or KV cache fails — activations have heavier tails than weights, so the same uniform scaling is even more lossy.

QuaRot (Ashkboos et al., NeurIPS 2024, [arxiv:2404.00456]) proposes applying random Hadamard rotations to hidden-state spaces *before* quantization. Rotations are orthogonal and computationally invariant — they can be absorbed into adjacent linear layers, so the model output is mathematically identical to the un-rotated model. But the rotated activations have **uniform-magnitude channels** (no outliers) because Hadamard projection spreads outlier energy across all dimensions.

Result on LLaMA-2-70B (paper): 4-bit weights+activations+KV with at most 0.47 WikiText-2 perplexity loss; 99% zero-shot retention. Lossless 6/8-bit.

The reference QuaRot implementation ([spcl/QuaRot](https://github.com/spcl/QuaRot)) is Python/CUDA. We did not find a public pure-Rust QuaRot implementation in a brief search; some Rust-adjacent quantization work exists (e.g., [konjoai/squish](https://github.com/konjoai/squish)'s `squish_quant_rs`) but is not QuaRot. The "first pure-Rust QuaRot" framing is plausible but unverified — claim only after a more thorough survey if it ever matters for external communication.

## Decision

Implement QuaRot v0 in `lattice-inference` as a new module `quant::quarot` (not a new crate — per CLAUDE.md the inference crate already holds weight-format and quantization code).

### Scope — v0 (multi-PR)

v0 is the offline rotation + Q4-weight-only path. Ships in 4 sequential PRs to keep each diff small and reviewable:

| Step | PR | Scope |
|---|---|---|
| 1 | this PR | Walsh-Hadamard + RandomizedHadamard primitives, unit-tested. No model code touched. |
| 2 | next | Rotation absorption: read SafeTensors → compute `W·Q^T` for each linear layer affected by a planned rotation → save rotated weights as new SafeTensors. Includes equivalence assertion (`||rotated_forward − original_forward|| < 1e-5` on a small batch) before the rotated weights are written. |
| 3 | next | Wire rotated SafeTensors through existing `weights::q4_weights::quantize_bf16_to_q4`. Binary `quantize_quarot` modeled on `quantize_q4` taking seed + model path. |
| 4 | next | Bench: rotated-Q4 vs unrotated-Q4 perplexity delta on Qwen3-0.6B / Qwen3.5-0.8B against WikiText-2 calibration. Acceptance: delta < 0.5 PPL. |

**Out of v0 entirely (deferred to v1)**:
- INT4 activation quantization (online during forward pass)
- INT4 KV cache quantization
- INT4 attention scores
- Per-layer rotation seeds (v0 uses one global Hadamard for the residual stream + one per attention head dim)
- Calibration-data tuning (v0 uses random Hadamard only — the QuaRot paper shows this is competitive without learned rotations)

**Model coverage in v0**:

| Model | hidden | GQA head_dim | linear-attn head_dim | v0 supported? |
|---|---:|---:|---:|---|
| Qwen3-0.6B (decoder) | 1024 | 128 | n/a | ✅ both power of 2 |
| Qwen3.5-0.8B (hybrid GDN+GQA) | 1024 | 256 | 128 (linear K/V) | ✅ all power of 2 |
| Qwen3-Embedding-4B (decoder, served by `QwenModel`) | 2560 | 128 | n/a | ❌ **explicitly deferred** — `2560 = 2^9 · 5`, hidden not a power of 2 |
| BERT-base (encoder) | 768 | 64 | n/a | ❌ **explicitly deferred** — `768 = 2^8 · 3`, hidden not a power of 2 |

For non-power-of-2 hidden dims (4B, BERT), v1 cannot use naive zero-pad-then-project — `π ∘ H_N ∘ P` (project up to power of 2, apply Hadamard, drop padded coords) is **not** orthogonal on the original space and breaks the QuaRot invariant. v1 needs one of: (a) the QuaRot paper's online block-diagonal approach (rotates within sub-blocks that are individually power of 2), (b) Paley-construction Hadamard matrices that exist for some specific non-power-of-2 dims, or (c) Givens / Householder products that are orthogonal but not structured. Which path v1 takes is an open question.

### Key Design Choices

**Hadamard size = power of 2 only.** v0 covers decoder models whose hidden and head dims are all powers of 2 (Qwen3-0.6B, Qwen3.5-0.8B — see model-coverage table above). Models with non-power-of-2 hidden dims (Qwen3-Embedding-4B at 2560, BERT-base at 768) are **explicitly deferred** to v1; naive zero-pad + project-back is not orthogonal and so cannot be used. v1 will pick from block-diagonal partition (QuaRot paper's approach), Paley-construction Hadamards, or Givens products.

**Offline rotation, not on-the-fly.** v0 saves the rotated+quantized weights as a new `.q4` file. The forward pass loads it like any other Q4 model — no runtime rotation cost. Tradeoff: 2× disk usage during conversion. Trade is worth it because rotation is a one-time cost.

**Rotation absorption pattern.** For a linear layer `y = Wx`:
- Pre-rotation: `y = W @ Q^T @ Q @ x = (W @ Q^T) @ (Q @ x)`
- The previous layer's output gets `Q @ ·` absorbed; this layer's `W` becomes `W @ Q^T`.
- Both absorptions are free (one matmul during conversion, no runtime cost).

For QKV projections in attention, the rotation `Q` is shared across Q, K, V heads (preserves dot-product). The output projection absorbs `Q^T` to undo it.

**Q4 block format unchanged.** The rotated weights still go through the same Q4_0 per-block scheme as `weights::q4_weights` — QuaRot's job is purely to redistribute outliers before quantization, not to change the quantized format. Forward pass is identical.

### Alternatives Considered

| Alternative | Pros | Cons | Why Not |
|---|---|---|---|
| New crate `lattice-quant` | Clean boundary | CLAUDE.md says no new crates without approval; quantization code already lives in `inference::weights` | Adds workspace overhead with no abstraction benefit |
| Implement SpinQuant or learned rotations instead | Better paper results | Requires training data + optimization loop | Out of scope for a pure-inference v0 |
| Skip rotation, just improve Q4 (e.g., per-channel scales) | Smaller change | Doesn't address outlier channels structurally | Doesn't reach paper-claimed accuracy at 4-bit |
| GPTQ-style Hessian-based weight quantization | Existing literature, possibly better accuracy | Requires calibration data + per-layer Hessian compute (~minutes per layer at scale) | v0 wants random Hadamard only — no calibration |
| Activation quantization in v0 | Bigger speedup | Requires fused INT4 GEMM kernels; ours don't exist yet | Defer to v1 with proper Metal/NEON INT4 kernels |

## Consequences

These are consequences of accepting v0 as the design (across all 4 PRs), not of the primitives-only PR that lands first.

### Positive

- Once v0 is fully landed (step 4 PR): 4-bit weights targeting paper-claimed PPL delta < 0.5 vs F32 on Qwen3-0.6B / Qwen3.5-0.8B. The target number is the paper's; it must be re-measured on our pipeline before being claimed.
- Foundation for v1 INT4 activation+KV — the rotated activation distributions are quantizable; the un-rotated ones aren't.
- Pure-Rust QuaRot implementation. The reference [spcl/QuaRot](https://github.com/spcl/QuaRot) is Python/CUDA; some Rust-adjacent quantization work exists (e.g., `konjoai/squish`'s `squish_quant_rs`) but isn't a QuaRot implementation. So "first pure-Rust QuaRot" is plausible but not certain — verify before claiming externally.
- Knowledge-graph node `QuaRot` moves from `researched` → `implemented` when step 4 lands.
- Composes with future SpinQuant / learned rotations (same absorption infrastructure).

### Negative

- Hadamard transform requires power-of-2 dimensions; non-power-of-2 hidden sizes (4B, BERT-base) deferred to v1 with no naive padding option.
- 2× disk during conversion (original + rotated).
- One-time conversion cost (~minutes for Qwen3-0.6B; ~hour for 8B-class models).
- Public API surface (`hadamard::{walsh_hadamard_in_place, RandomizedHadamard}`) lands before any consumer exists in the codebase — locks in transform-order, precision, and naming before downstream code stresses the contract. Mitigation: this matches the existing lattice-inference convention of keeping module APIs `pub` for cross-module use, and the crate-level STABILITY doc already flags the crate as `Experimental` with churn expected. If the step-2 PR (rotation absorption) needs to change the primitive API, that change ships in the same PR.

### Risks

- **Numerical precision**: Hadamard transforms in f32 are stable, but interaction with f16 storage of scales (in `Q4Block::scale: u16`) may compound rounding. Mitigation: keep rotation math in f64, quantize in f32, store scales in f16 as before — measure perplexity delta vs full-precision rotation.
- **Forgotten absorptions**: A rotation that doesn't have its inverse `Q^T` absorbed somewhere will change model output. Mitigation: assert computational equivalence on a small batch (identity tolerance < 1e-5) before saving rotated weights. Refuse-on-fail.
- **Residual stream vs head-dim mismatch**: Different rotations for hidden vs head-dim spaces. Need careful tracking. Mitigation: explicit `RotationPlan` type that records which dimension each rotation applies to + an end-to-end test asserting equivalence.

## References

- Paper: Ashkboos et al., *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*, NeurIPS 2024, [arxiv:2404.00456](https://arxiv.org/abs/2404.00456)
- Related: SpinQuant (Liu 2024), Hadamard transforms in ML (Yu et al. 2017)
- KG entities: `QuaRot (Ashkboos 2024)` (86ec6a4f), `Rotation-Based Quantization` (d534ca51), `QuantizationRankFlipCondition` (224ef1a7), `HAWQ`, `BRECQ`, `AWQ`, `SmoothQuant`, `HQQ`
- KG ontology: `.khive/taxonomy.md`
- Existing code: `crates/inference/src/weights/q4_weights.rs`, `crates/inference/src/bin/quantize_q4.rs`
