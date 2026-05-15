# ADR-044: QuaRot — Hadamard-Rotated 4-bit Quantization

**Status**: Proposed
**Date**: 2026-05-15
**Crate**: lattice-inference

## Context

Existing 4-bit quantization in `lattice-inference` (`weights::q4_weights`) is naïve per-block symmetric round-to-nearest (RTN). RTN works for weight-only Q4 on well-behaved layers, but two failure modes are well-documented in the literature and present in Qwen3 weights:

1. **Outlier channels**: a small fraction of hidden-state channels carry weights with magnitudes 10-100× the median. Uniform per-block scaling either clips them (information loss in the outlier) or wastes dynamic range across the rest of the block (information loss everywhere else).
2. **Activation distribution skew**: extending Q4 to activations or KV cache fails — activations have heavier tails than weights, so the same uniform scaling is even more lossy.

QuaRot (Ashkboos et al., NeurIPS 2024, [arxiv:2404.00456]) proposes applying random Hadamard rotations to hidden-state spaces *before* quantization. Rotations are orthogonal and computationally invariant — they can be absorbed into adjacent linear layers, so the model output is mathematically identical to the un-rotated model. But the rotated activations have **uniform-magnitude channels** (no outliers) because Hadamard projection spreads outlier energy across all dimensions.

Result on LLaMA-2-70B (paper): 4-bit weights+activations+KV with at most 0.47 WikiText-2 perplexity loss; 99% zero-shot retention. Lossless 6/8-bit.

No public pure-Rust QuaRot implementation exists.

## Decision

Implement QuaRot v0 in `lattice-inference` as a new module `quant::quarot` (not a new crate — per CLAUDE.md the inference crate already holds weight-format and quantization code).

### Scope — v0 (this PR)

**In**:
- Walsh-Hadamard transform primitive (fast in-place, `n` must be a power of 2)
- Random Hadamard matrix generator (signed permutation × structured Hadamard, seedable)
- Rotation absorption into linear-layer weight matrices (offline, save rotated weights as new safetensors)
- Integration with existing `weights::q4_weights::quantize_bf16_to_q4` — apply rotation first, then quantize
- Binary: `quantize_quarot` modeled on `quantize_q4`, takes rotation seed + model path
- Bench: rotated-Q4 vs unrotated-Q4 perplexity delta on Qwen3-0.6B WikiText-2 calibration

**Out (deferred to v1)**:
- INT4 activation quantization (online during forward pass)
- INT4 KV cache quantization
- INT4 attention scores
- Per-layer rotation seeds (v0 uses one global Hadamard for the residual stream + one per attention head dim)
- Calibration-data tuning (v0 uses random Hadamard only — the QuaRot paper shows this is competitive without learned rotations)

### Key Design Choices

**Hadamard size = power of 2 only.** Qwen3-0.6B has `hidden=1024`, `head_dim=128` — both are powers of 2, so structured Walsh-Hadamard works without padding. For non-power-of-2 hidden sizes we'd need randomized Hadamard via Hadamard-of-bigger-power-of-2 with masking; deferred.

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

### Positive

- 4-bit weights with paper-claimed near-lossless accuracy (PPL delta < 0.5 vs F32 target)
- Foundation for v1 INT4 activation+KV — the rotated activation distributions are quantizable, which the un-rotated ones aren't
- Demonstrates pure-Rust QuaRot ahead of any other inference engine (no public Rust impl exists as of 2026-05-15)
- Knowledge-graph node `QuaRot` moves from `researched` → `implemented`
- Composes with future SpinQuant / learned rotations (same absorption infrastructure)

### Negative

- Hadamard transform requires power-of-2 dimensions; non-power-of-2 hidden sizes need padding or skip
- 2× disk during conversion (original + rotated)
- One-time conversion cost (~minutes for Qwen3-0.6B; ~hour for 8B-class models)

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
