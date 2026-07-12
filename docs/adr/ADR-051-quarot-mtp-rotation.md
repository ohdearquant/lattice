# ADR-051: QuaRot-MTP Rotation Reconciliation

**Status**: Accepted
**Date**: 2026-05-19
**Crate**: `lattice-inference`

---

## Context

ADR-044 ships QuaRot Q4 quantization for Qwen3.5. It applies a global random Hadamard rotation R
(seed-deterministic, orthonormal, d×d) to the residual stream. All linear layers in the base model
absorb R: input-side weights become W·R^T, output-side weights have R pre-applied, so every
activation in the forward pass lives in the rotated basis R·h rather than the original h.

ADR-006 ships MTP speculative decoding. The MTP module (one transformer layer with MoE FFN) drafts
an extra token using two inputs from the base model:

- `embed(x_t)` — the embedding of the just-generated token (looked up from `embed_tokens`)
- `h_{t-1}^{pre}` — the pre-final-norm hidden state captured from the previous base-model step

Both inputs arrive in the rotated basis when the base model is QuaRot-quantized:

```
embed(x_t)      -> R · embed_original(x_t)    [embed_tokens rows are rotated]
h_{t-1}^{pre}   -> R · h_{t-1}^{pre,original} [residual stream is rotated throughout]
```

MTP's weights (`pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `input_layernorm`,
`post_attention_layernorm`, `fc`, `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`, `norm`) were trained in the original unrotated basis. This creates a basis mismatch.

### Why RMSNorm Gamma Does Not Commute With R

RMSNorm with learned scale gamma computes:

```
RMSNorm(x) = (x / RMS(x)) * gamma   where gamma is per-element
```

For a rotated input R·x:

```
RMSNorm(R·x) = (R·x / RMS(R·x)) * gamma
             = (R·x / RMS(x)) * gamma     [RMS is rotation-invariant: ||R·x|| = ||x||]
```

This is NOT equal to R · RMSNorm(x):

```
R · RMSNorm(x) = R · ((x / RMS(x)) * gamma)
               = (R·x / RMS(x)) * R[gamma as diagonal]
```

The per-element gamma scale distributes as diag(gamma); diag(gamma) does not commute with a dense
orthogonal R. Therefore:

```
RMSNorm(R·x) = diag(gamma) · (R·x) / RMS(x)   [gamma applied in rotated coordinates]
R · RMSNorm(x) = R · diag(gamma) · x / RMS(x) [gamma applied in original coordinates]
```

These differ by (diag(gamma)·R - R·diag(gamma)) which is non-zero whenever gamma is not uniform.

### Measured Magnitude

PR #42 captured rotated vs unrotated hidden states at the MTP boundary and measured:

```
|R·h - h|^2 / |h|^2 ≈ 2
```

A ratio of 2 means the rotated and unrotated hidden states are fully decorrelated (same L2 norm,
completely different directions). Draft token acceptance drops to approximately 1/|V|, i.e., random
chance over vocabulary. PR #42 responded by safety-skipping MTP emission when a QuaRot base is
detected, leaving MTP disabled for all QuaRot models.

### Shared lm_head Dependency

The MTP module shares `lm_head_weight` with the base model. Under QuaRot, `lm_head` was
reparameterized during conversion (ADR-044 step 3c): the fused form stores
`W_lm · diag(1 + g_final) · R^T` (final-norm scale absorbed, input-rotation absorbed). This means
`lm_head` expects its input to be in the rotated basis. If MTP produces an unrotated hidden state
h_mtp and passes it directly to the shared `lm_head`, the logit computation becomes:

```
(W_lm · diag(1+g) · R^T) · h_mtp   [wrong: lm_head expects R·h_mtp, not h_mtp]
```

Any fix that produces unrotated MTP hidden states must re-rotate before the shared `lm_head` call.

---

## Decision

Implement **Strategy A: counter-rotation at MTP boundaries** (two FWHT calls per draft step).

Apply R^T to MTP inputs before the MTP forward pass and apply R to MTP outputs before the shared
`lm_head` matmul. All MTP internal weights operate in the original unrotated basis unchanged — no
weight modifications, no offline preprocessing, no new conversion artifacts.

The two-FWHT approach is chosen over reparameterization (Strategy B) as Phase 1 because it:

- Has a single, auditable insertion point per boundary
- Requires zero changes to the quantize_quarot binary or stored artifacts
- Is fully reversible (R^T · R = I) with no accumulated error beyond floating-point round-trip
- Validates correctness by comparing MTP draft logits against a reference CPU forward

Strategy B (offline reparameterization of all 15 MTP weight tensors) is designated Phase 2 and
deferred until Phase 1 acceptance rates are confirmed on hardware.

---

## Scope

**In scope (this ADR)**:

- Counter-rotation shim in `mtp_forward_one()` at `src/forward/metal_qwen35.rs`
- Detection of QuaRot base (rotation seed metadata read from quantize index)
- Rotation seed propagation into `MetalQwen35Engine` for runtime use
- Re-rotation of MTP output hidden state before shared `lm_head` matmul
- CPU validation path: compare draft logits with and without shim against reference
- `quantize_quarot` binary: store rotation seed in `quantize_index.json` (currently omitted)

**Out of scope**:

- Strategy B offline reparameterization of MTP weights
- INT4 activation quantization of MTP inputs/outputs (ADR-044 v1 deferred)
- N-gram speculator is unaffected (no hidden states)
- Models other than Qwen3.5 (only model with MTP weights in v0)

---

## Architecture

### Rotation Propagation

The QuaRot seed is currently used only inside `quantize_quarot` and is not stored in the output
artifact. Phase 1 requires it at runtime. The fix:

1. `quantize_quarot` writes `"quarot_seed": <u64>` into `quantize_index.json`.
2. The runtime loader (`loading.rs`) reads `quarot_seed` as `Option<u64>` when loading a QuaRot
   index; absent key = unrotated model (no counter-rotation needed).
3. `MetalQwen35Engine` gains `quarot_seed: Option<u64>`. Non-None means all activations are in
   the rotated basis and MTP must counter-rotate.

### Counter-Rotation Calls in `mtp_forward_one`

The insertion points within the existing Phase 1 / Phase 2 structure of `mtp_forward_one`:

```
Phase 1 (CPU):
  1. embed lookup -> normed_embed  (currently in rotated basis when QuaRot)
  2. pre_fc_norm_embedding RMSNorm on normed_embed
  3. normed_hidden = last_pre_final_hidden.clone()  (in rotated basis)
  4. pre_fc_norm_hidden RMSNorm on normed_hidden
  5. concat [normed_embed || normed_hidden] -> fused buffer (2*hidden)

  --- NEW: if quarot_seed.is_some() ---
  Before step 2: apply R^T to normed_embed  [counter-rotate embedding]
  Before step 4: apply R^T to normed_hidden [counter-rotate hidden state]
  --- END NEW ---

Phase 2 (GPU):
  fc: fused -> hidden (linear projection)
  ... (all MTP attention + MLP in unrotated basis, unchanged) ...
  final hidden state produced: h_mtp in unrotated basis

  --- NEW: if quarot_seed.is_some() ---
  After final GPU computation, before shared lm_head:
  apply R to h_mtp [re-rotate for lm_head]
  --- END NEW ---

lm_head matmul: (W_lm · diag(1+g) · R^T) · (R · h_mtp) = W_lm · diag(1+g) · h_mtp  [correct]
```

### FWHT Call

`walsh_hadamard_orthonormal_in_place(data: &mut [f32])` from `quant::quarot::hadamard` implements
the normalized FWHT. For an orthonormal Hadamard H_d (H_d · H_d^T = I), applying H_d twice returns
the identity: H_d is self-inverse up to the normalization factor 1/sqrt(d), which the orthonormal
variant absorbs. Therefore:

```
H_d · H_d · x = x    [exact, no separate R^T needed]
```

The code convention (see `quant::quarot::hadamard.rs:120-127`) defines `R = H · diag(s)` where
`s_i ∈ {+1,-1}` are the random sign flips from the seed. `apply()` performs sign flip then WHT.
Therefore `R^T = diag(s) · H` (since `diag(s)^T = diag(s)` and `H^T = H`), and `apply_inverse()`
performs WHT then sign flip.

The counter-rotation procedure for a vector x in the rotated basis:

```rust
// Recover original: x_original = R^T · x_rotated
// R = H · diag(s)  =>  R^T = diag(s) · H
// This is RandomizedHadamard::apply_inverse()
hadamard.apply_inverse(&mut x)?;
// Internally: walsh_hadamard_orthonormal_in_place(&mut x) then apply_sign_flip(&mut x, &signs)
```

The re-rotation (original -> rotated) uses the forward direction:

```rust
// x_rotated = R · x_original = H · diag(s) · x_original
// This is RandomizedHadamard::apply()
hadamard.apply(&mut x)?;
// Internally: apply_sign_flip(&mut x, &signs) then walsh_hadamard_orthonormal_in_place(&mut x)
```

Sign flip vectors are deterministic from the seed via the same `RandomizedHadamard` generator used
in `quantize_quarot`. The engine holds a `RandomizedHadamard` instance derived once at load time.

**Verification**: any implementation must pass a round-trip test: `apply(apply_inverse(x)) == x` and
`apply_inverse(apply(x)) == x` for a fixed seed, plus an MTP draft-logit equivalence test comparing
counter-rotated MTP logits against the unquantized reference model.

### Complexity

Per draft token: 2 × FWHT(d) where d = hidden_size.
For Qwen3.5-7B (d = 3584): O(d log d) = O(3584 × 11.8) ≈ 42K multiply-adds per counter-rotation,
two calls = ~84K ops. Target forward pass is memory-bandwidth-bound at ~100M ops; this is <0.1%
overhead.

### quantize_quarot Binary Change

`quantize_index.json` gains one field written unconditionally when `--seed` is provided:

```json
{ "quarot_seed": 13258600446175248384 }
```

The runtime loader treats an absent `quarot_seed` key as `None` (unrotated model). Existing
unrotated `.q4` artifacts remain loadable without conversion.

---

## Alternatives Considered

| Strategy                           | Mechanism                                                     | Runtime cost          | Weight changes               | lm_head correctness                       |
| ---------------------------------- | ------------------------------------------------------------- | --------------------- | ---------------------------- | ----------------------------------------- |
| **A: Counter-rotate (chosen)**     | R^T before MTP, R after; 2×FWHT/step                          | ~84K ops/draft step   | None                         | R cancels: (W·R^T)·(R·h) = W·h            |
| **B: Offline reparameterize**      | W'=W·R for each of 15 MTP tensors, gamma'=R·gamma for 7 norms | Zero                  | 15 tensors rewritten offline | lm_head receives rotated output naturally |
| **C: Rotation-aware MTP training** | Fine-tune MTP to handle rotated inputs                        | Zero (after training) | Full retraining required     | Training learns invariance                |
| **D: Separate lm_head for MTP**    | MTP uses its own unshared lm_head (unrotated)                 | Zero                  | New 1.2GB tensor per model   | No shared lm_head conflict                |

**Why not B (reparameterize)**: Correct but requires non-trivial per-layer-type math to
reparameterize fc (2d×d), q/k/v/o projections (d×d each), gate/up/down projections (FFN), and 7
norm gamma vectors. Each has a different absorption formula. One wrong formula silently produces
bad logits with no crash. Phase 2 will implement B after Phase 1 validates the expected acceptance
rates, using Phase 1 as the oracle for correctness verification.

**Why not C (training)**: Out of scope. Requires a training pipeline that Lattice does not have.

**Why not D (separate lm_head)**: Doubles the output embedding storage (+1.2GB for Qwen3.5-7B) and
diverges MTP draft logits from the base model vocabulary distribution.

---

## Risks

**Sign vector reconstruction.** The counter-rotation uses sign vectors derived from `quarot_seed`
via `RandomizedHadamard`. If the runtime reconstructs a different sign sequence than the converter
used, the rotation will be wrong but not obviously so (logits will look plausible, not garbage). The
seed must be stored exactly as a `u64` in `quantize_index.json` and reconstructed via the identical
code path. Mitigation: add a round-trip test asserting that `R^T · (R · v) == v` for a fixed seed
before any runtime use of the sign vectors.

**lm_head basis assumption.** The re-rotation step (apply R to h_mtp) is predicated on the stored
`lm_head` weight having form `W_lm · diag(1+g) · R^T`. If a future conversion path changes this
(e.g., drops the final-norm fusion), the re-rotation would produce wrong logits. Mitigation: the
QuaRot forward-equivalence gate (ADR-044 step 3c-4) must be extended to cover the MTP path once
MTP tensors are no longer safety-skipped.

**MTP tensors still safety-skipped in quantize_quarot.** The converter currently skips the 15 MTP
tensors to avoid writing Q4-quantized weights in the wrong basis. Phase 1 (counter-rotation) fixes
the runtime basis but does NOT change this: MTP weights are still stored as f16 (unquantized). The
15 MTP tensors are loaded as f16 and used at f32 precision. This is correct for Phase 1 but means
MTP draft steps do not benefit from Q4 weight compression. Phase 2 (reparameterize) will rotate and
quantize the MTP tensors.

**Fused activation precision.** Counter-rotation inserts `walsh_hadamard_orthonormal_in_place` on
the CPU-side f32 vectors before they are written into the Metal-allocated `fused` buffer. This does
not change the dtype or buffer layout; no GPU kernel changes are required for Phase 1.

---

## References

- ADR-044: QuaRot rotated quantization (rotation absorption, RMSNorm fusion, forward-equivalence gate)
- ADR-006: Speculative decoding (MtpVerifier, `mtp_forward_one`, cache rollback)
- `src/forward/metal_qwen35.rs:5015` — `mtp_forward_one` Phase 1 (CPU) and Phase 2 (GPU) structure
- `src/quant/quarot/hadamard.rs` — `walsh_hadamard_orthonormal_in_place` (f32, self-inverse)
- `src/quant/quarot/plan.rs` — `RotationPlan`, `RandomizedHadamard`
- Ashkboos et al., _QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs_, NeurIPS 2024, arxiv:2404.00456
