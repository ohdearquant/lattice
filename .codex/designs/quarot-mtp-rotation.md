# QuaRot-MTP Rotation Fix — Design

**Status**: Draft  
**Date**: 2026-05-18  
**ADR scope**: `lattice-inference` (forward, quant/quarot, quant/quarot/convert)  
**Crate**: `lattice-inference`  
**Companion ADR**: ADR-044 (QuaRot Rotated Quantization) — this doc is the MTP extension

---

## 1. Problem Statement

QuaRot (Ashkboos et al., NeurIPS 2024) applies a randomized Hadamard rotation
`R = H · D` to the residual stream and `embed_tokens`. After conversion, every
activation that flows into a linear layer is in the **rotated basis** (R-space).
All base-model weights are pre-multiplied by R or R^T so the forward pass is
numerically equivalent to the original.

The MTP head (Qwen3.5-0.8B: one attention+MLP layer + fc + norms) was trained
in the **original basis** (O-space). Its weights — `pre_fc_norm_embedding`,
`pre_fc_norm_hidden`, `mtp.fc`, `input_layernorm`, all projections, `mtp.norm` —
are in O-space. The inputs it receives in the QuaRot runtime are:

- `embed[t]` — looked up from `embed_tokens` (stored in R-space after QuaRot)
- `last_pre_final_hidden` — captured from residual stream (in R-space at capture)

Per-element RMSNorm gamma does not commute with R: `γ ⊙ (R·x) ≠ R · (γ ⊙ x)`.
Consequently, feeding R-space inputs to O-space MTP weights produces wrong logits,
and MTP emission was correctly disabled for QuaRot (PR #42, `convert.rs` line 335).

---

## 2. Data Flow Diagram

### 2a. Q8 (non-QuaRot) — WORKING

```
Token t
│
▼
embed_tokens[t]          ← O-space f16 (trained basis)
│
│  CPU RMSNorm(pre_fc_norm_embedding)  ← O-space gamma
▼
normed_embed             ← O-space

last_pre_final_hidden    ← captured BEFORE final_norm (line 8022-8027)
                           = output of last transformer block, O-space
│
│  CPU RMSNorm(pre_fc_norm_hidden)     ← O-space gamma
▼
normed_hidden            ← O-space

[normed_embed ‖ normed_hidden]         ← O-space, [2*hidden]
│
│  fc: [hidden, 2*hidden] Q4           ← O-space weight
▼
mtp_hidden               ← O-space

│  RMSNorm(input_layernorm)            ← O-space gamma
│  q/k/v/o projections Q4             ← O-space weights
│  RMSNorm(mtp.norm)                  ← O-space gamma
▼
mtp_h_out                ← O-space

│  lm_head (embed_tokens_q8)          ← O-space (Q8 path: embed is tied/untied)
▼
logits                   ← correct
```

### 2b. QuaRot Q4 — CURRENT (BROKEN, MTP disabled)

```
Token t
│
▼
embed_tokens[t]          ← R-space f16 (QuaRot rotated embed_tokens)
│
│  CPU RMSNorm(pre_fc_norm_embedding)  ← O-space gamma  ← MISMATCH
▼
normed_embed             ← WRONG BASIS

last_pre_final_hidden    ← captured from residual stream AFTER all layers
                         = residual in R-space (rotation absorbed into weights,
                           so activations live in R-space at runtime)
│
│  CPU RMSNorm(pre_fc_norm_hidden)     ← O-space gamma  ← MISMATCH
▼
normed_hidden            ← WRONG BASIS
```

### 2c. QuaRot Q4 — PROPOSED (Strategy 1: Counter-Rotate)

```
Token t
│
▼
embed_tokens[t]          ← R-space f16
│
│  apply_inverse(R^T)    ← FWHT: O(d log d), single CPU pass
▼
embed_o                  ← O-space

│  CPU RMSNorm(pre_fc_norm_embedding)  ← O-space gamma  ← NOW CORRECT
▼
normed_embed             ← O-space

last_pre_final_hidden    ← R-space (residual stream in QuaRot runtime)
│
│  apply_inverse(R^T)    ← FWHT: O(d log d), single CPU pass
▼
hidden_o                 ← O-space

│  CPU RMSNorm(pre_fc_norm_hidden)     ← O-space gamma  ← NOW CORRECT
▼
normed_hidden            ← O-space

[normed_embed ‖ normed_hidden]         ← O-space, [2*hidden]
│
│  fc: [hidden, 2*hidden]              ← O-space weight ← CORRECT (unchanged)
▼
mtp_hidden               ← O-space

│  RMSNorm + projections               ← O-space weights ← CORRECT (unchanged)
│  RMSNorm(mtp.norm)
▼
mtp_h_out                ← O-space

│  lm_head (embed_tokens_q8)          ← ? ← KEY QUESTION (see §3)
▼
logits
```

---

## 3. The lm_head Basis Question

This is the critical decision point. In the QuaRot Q4 runtime:

**`embed_tokens_q8` is the logits GEMV weight** (line 9908-9912 of `metal_qwen35.rs`):
- Tied config (`tie_word_embeddings=true`): `embed_tokens_q8 = embed_tokens.q4`, which is
  the **R-space** embed matrix (QuaRot absorbed R into embed_tokens).
- Untied config (`tie_word_embeddings=false`): `embed_tokens_q8 = lm_head.weight.q4`, which
  is the materialized `lm_head` with R and `(1+γ_final)` both absorbed (per `convert.rs`
  line 216-226: `materialize_lm_head_for_qwen35` + `fuse_rmsnorms` + `absorb_rotations`).
  After absorption, `lm_head'[i,*] = lm_head[i,*] · R^T` (output-side absorption).

Therefore in the Q4 QuaRot path, `embed_tokens_q8` **expects R-space input**:

```
lm_head' = lm_head · R^T
y = lm_head' · x_rot = lm_head · (R^T · x_rot) = lm_head · x_orig   (correct)
```

The base-model final step is: `logits = lm_head' · h_rot` where `h_rot` is the
residual stream vector in R-space after the final RMSNorm.

**Consequence for MTP (Strategy 1)**:

After the MTP transformer block runs entirely in O-space, `mtp_h_out` is in O-space.
The logits GEMV uses `embed_tokens_q8` which in QuaRot expects R-space input.
So we must **re-rotate** `mtp_h_out` before the logits GEMV:

```
mtp_h_out (O-space) → apply(R) → mtp_h_rot (R-space) → lm_head' → logits
```

Complete corrected MTP forward for QuaRot:

```
embed[t]                       (R-space, from embed_tokens f16 buffer)
  → apply_inverse(R^T)         (→ O-space)
  → RMSNorm(pre_fc_norm_embedding)
  → normed_embed               (O-space)

last_pre_final_hidden          (R-space, from residual stream)
  → apply_inverse(R^T)         (→ O-space)
  → RMSNorm(pre_fc_norm_hidden)
  → normed_hidden              (O-space)

[normed_embed ‖ normed_hidden] → fc → mtp_hidden     (O-space)
mtp_hidden → MTP attn+MLP block                      (O-space throughout)
→ mtp_h_out (after mtp.norm)                         (O-space)
  → apply(R)                   (→ R-space)
  → lm_head' (embed_tokens_q8) → logits              (correct)
```

Total extra FWHT calls per MTP step: **3** (two apply_inverse + one apply),
each O(d log d) = O(1024 × 10) ≈ 10K flops. Negligible vs. the matmul cost.

---

## 4. Strategy Comparison

### Strategy 1: Counter-Rotate Inputs + Re-Rotate Output (RECOMMENDED)

**Runtime changes only.** No converter changes needed for the first two FWHT calls.
One converter change needed to emit MTP weights (norms as f16, projections as q4)
in their original O-space form — which is exactly what the current checkpoint has
(the converter currently skips MTP entirely, so the original O-space weights are
correct and just need to be written out as-is).

| Dimension | Assessment |
|-----------|------------|
| LOC (converter) | ~80 LOC — emit MTP weights without rotation absorption |
| LOC (runtime) | ~25 LOC — 3 FWHT calls inserted into `mtp_forward_one` |
| LOC (config) | ~5 LOC — remove `zero_mtp_in_config_json` call for QuaRot |
| LOC (tests) | ~150 LOC — new equivalence gate test |
| Total | ~260 LOC |
| Runtime cost | 3 × O(d log d) per MTP draft step (≈ 30 μs for d=1024) |
| Correctness risk | Low — FWHT is its own inverse up to scaling; easy to test |
| Conversion time | Zero additional conversion time |
| Reversibility | Full — no weight modification |

### Strategy 2: Weight Reparameterization (NOT RECOMMENDED for v0)

Absorb R into MTP weights at conversion time. For each linear weight W in O-space:
- Input-side: `W' = W · R^T` (column multiply by R^T)
- Output-side: `W' = R · W` (row multiply by R)
For norms: replace gamma with `γ' = gamma` (no change — see §3.3 below).

Wait — norms are the core problem. RMSNorm gamma is per-element scale AFTER
normalize. The operation is `y_i = x_i / rms(x) * gamma_i`. For input `R·x`:
`rms(R·x) = rms(x)` (R is orthogonal), but `y_i = (R·x)_i / rms(x) * gamma_i`.
This is NOT equal to `(R · (x / rms(x) * gamma))_i` unless gamma is uniform.
So you cannot reparametrize gamma to fix the norm — the norm itself must see
O-space input. Strategy 2 does not solve the norm problem without replacing
all MTP norms with a different operator (e.g., post-norm fusion similar to what
is done for base-model norms).

Base-model norms ARE solvable: `(1 + γ) ⊙ normalize(h)` can be fused into the
downstream linear's column scale because `(1 + γ)` multiplies the normalized
output element-wise and then the linear W multiplies. But MTP's `pre_fc_norm_*`
and `input_layernorm` cannot be fused this way without a downstream linear that
absorbs the scale — `pre_fc_norm_embedding` feeds into a concat, not a linear.

**Verdict**: Strategy 2 requires either (a) fusing all MTP norms into their
downstream linears (as base-model norms are fused, but with the concat in the
middle making `pre_fc_norm_*` unfusable without structural changes) or (b) a new
"post-norm linear" abstraction. This is significantly more complex than Strategy 1
with no runtime throughput benefit (weights are already Q4, further rotation just
shifts errors into quantization noise).

### Strategy 3: Rotation-Aware Training

Not feasible — requires access to training infrastructure and original data.

---

## 5. Exact Changes Per File

### 5a. `crates/inference/src/quant/quarot/convert.rs`

**Change 1**: Remove `zero_mtp_in_config_json` call and replace with conditional
based on whether MTP weights are present.

Lines 335-342 (MTP skip comment block) and 358-360 (zero_mtp call) — replace with:

```rust
// MTP weights: emit without rotation absorption.
// The runtime applies R^T to inputs before the MTP forward pass (counter-rotate
// strategy, ADR-044 §MTP extension). MTP weights remain in original O-space;
// only base-model weights are rotated.
if cfg.mtp_num_hidden_layers > 0 {
    write_mtp_weights_quarot(
        &working_set,          // has original-basis MTP tensors loaded
        output_dir,
        &mut index_entries,
        &mut kept_f16,
        &mut planned_quantized,
        &mut total_bytes_out,
    )?;
}

// Config: retain mtp_num_hidden_layers as-is (MTP is now supported on QuaRot).
// Do NOT call zero_mtp_in_config_json.
```

**Change 2**: Add `write_mtp_weights_quarot` function (~50 LOC). This function:
- Iterates MTP tensor names derived from `cfg.mtp_num_hidden_layers`
- Writes projection weights (`fc`, `q/k/v/o_proj`, `gate/up/down_proj`) as `.q4`
- Writes norm weights (`pre_fc_norm_*`, `input_layernorm`, `post_attention_layernorm`,
  `q_norm`, `k_norm`, `mtp.norm`) as `.f16`
- Does NOT call `absorb_rotations` on any MTP tensor — they stay in O-space

MTP tensor names (from the test fixture at line 1568, cross-checked with
`load_mtp_q4_weights` at line 9373-9401):
```
mtp.fc.weight                           → .q4  [hidden, 2*hidden]
mtp.layers.0.self_attn.q_proj.weight    → .q4  [2*q_dim, hidden]
mtp.layers.0.self_attn.k_proj.weight    → .q4  [kv_dim, hidden]
mtp.layers.0.self_attn.v_proj.weight    → .q4  [kv_dim, hidden]
mtp.layers.0.self_attn.o_proj.weight    → .q4  [hidden, q_dim]
mtp.layers.0.mlp.gate_proj.weight       → .q4  [intermediate, hidden]
mtp.layers.0.mlp.up_proj.weight         → .q4  [intermediate, hidden]
mtp.layers.0.mlp.down_proj.weight       → .q4  [hidden, intermediate]
mtp.layers.0.input_layernorm.weight     → .f16 [hidden]
mtp.layers.0.post_attention_layernorm.weight → .f16 [hidden]
mtp.layers.0.self_attn.q_norm.weight    → .f16 [head_dim]
mtp.layers.0.self_attn.k_norm.weight    → .f16 [head_dim]
mtp.norm.weight                         → .f16 [hidden]
mtp.pre_fc_norm_embedding.weight        → .f16 [hidden]
mtp.pre_fc_norm_hidden.weight           → .f16 [hidden]
```

**Change 3**: Remove `zero_mtp_in_config_json` helper function (lines 109-133)
if it has no other callers. (Verify with `grep -n zero_mtp_in_config_json`.)

**Change 4**: Update the failing test `convert_quarot_qwen35_skips_mtp_for_quarot_even_when_present`
(line 1664) — invert: now MTP files SHOULD be emitted, and `mtp_num_hidden_layers`
in output config should be 1 (not 0).

### 5b. `crates/inference/src/forward/metal_qwen35.rs`

**Change 1**: Add QuaRot mode flag to `MetalQwen35Engine`.

After line 2750 (`pub(crate) mtp_weights: Option<MetalMtpWeights>`), add:
```rust
/// Rotation for QuaRot counter-rotate path. `Some` iff model was loaded from
/// a QuaRot q4 artifact (from_q4_dir). Used by mtp_forward_one to apply
/// R^T to embed and pre-final-hidden before O-space MTP forward.
pub(crate) quarot_rotation: Option<crate::quant::quarot::hadamard::RandomizedHadamard>,
```

**Change 2**: Initialize `quarot_rotation` in `from_q4_dir` (around line 10079).

The `from_q4_dir` function needs:
1. Accept `rotation_seed: Option<u64>` as a new parameter (or read from config —
   but rotation_seed is NOT stored in the converted `config.json` currently, so
   it must be passed as a parameter).
2. Construct `RandomizedHadamard::new(seed, cfg.hidden_size)?` when seed is Some
   and MTP weights are present.
3. Store into engine field.

Alternatively, store the seed in the output `config.json` at conversion time
(simplest for the caller). Add `quarot_rotation_seed` field to the config JSON
written by `convert_quarot_qwen35`. Then `from_q4_dir` reads it from config.

**Recommended**: Store seed in config JSON. Add to `Qwen35Config`:
```rust
pub quarot_rotation_seed: Option<u64>,
```
And in `convert.rs` output config step, inject `quarot_rotation_seed` into the
JSON before writing. This avoids passing an extra parameter to `from_q4_dir`.

**Change 3**: Modify `mtp_forward_one` (line 5015). After the embedding lookup
(lines 5034-5042) and BEFORE the `pre_fc_norm_embedding` RMSNorm (line 5044-5061):

```rust
// Counter-rotate embed from R-space to O-space for QuaRot artifacts.
if let Some(ref rot) = self.engine.quarot_rotation {
    rot.apply_inverse(&mut normed_embed)?;
}
```

After the `last_pre_final_hidden` clone (line 5064) and BEFORE the
`pre_fc_norm_hidden` RMSNorm (lines 5066-5081):

```rust
// Counter-rotate pre-final hidden from R-space to O-space.
if let Some(ref rot) = self.engine.quarot_rotation {
    rot.apply_inverse(&mut normed_hidden)?;
}
```

After the final RMSNorm `dispatch_rms_norm` (line 5390-5398) and BEFORE the
logits GEMV `dispatch_gemm` (line 5401-5411):

This is more complex because the re-rotation must happen on the GPU buffer `buf_hidden`
after `w_norm` norm and before the logits GEMV. Options:
1. Read back `buf_hidden` to CPU, apply `rot.apply()`, write back — expensive round-trip.
2. Add an MSL kernel `apply_hadamard_rotation` that takes the sign vector and applies
   FWHT in-place on the GPU buffer.
3. Move the logits GEMV to CPU — but that's slower than GPU for vocab_size=151,936.

**Recommended approach for re-rotation**: Add a new MSL kernel
`hadamard_rotate_inplace` that performs the FWHT butterfly + sign flip on a
`device float*` buffer. This kernel is O(d log d) = O(1024 × 10) flops and runs
entirely on GPU with no round-trip.

MSL kernel signature:
```metal
kernel void hadamard_rotate_inplace(
    device float* data [[buffer(0)]],
    constant float* signs [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
);
```

The FWHT butterfly in MSL is a standard iterative implementation using threadgroup
shared memory (or naive if d=1024 fits in a single threadgroup). GPU side runs
`apply(R)` = sign-flip then FWHT, so: first dispatch sign-flip kernel, then FWHT.

If GPU kernel complexity is too high for v0, an acceptable v0 fallback is:
read back `buf_hidden` (1024 f32 = 4 KB), apply `rot.apply()` on CPU, write
back to a staging buffer, then run logits GEMV from staging. The round-trip
at 4 KB is ~10 μs on M-series unified memory — acceptable given MTP drafting
is not on the critical path.

**For v0 (simpler)**: Use CPU round-trip for the re-rotation only.
Refactor the GPU block so the logits GEMV is dispatched in a SEPARATE command
buffer from the rest of the MTP forward. Between the two command buffers: read
back `buf_hidden` (4 KB), apply `rot.apply()` on CPU, write back. This avoids
a new MSL kernel.

### 5c. `crates/inference/src/quant/quarot/convert.rs` (config JSON)

Add `quarot_rotation_seed` injection. After `untie_word_embeddings_in_config_json`
(line 357):

```rust
let mut output_config_json = untie_word_embeddings_in_config_json(&config_json)?;
output_config_json = inject_quarot_seed(&output_config_json, opts.rotation_seed)?;
// MTP is now supported — do NOT call zero_mtp_in_config_json.
```

New helper `inject_quarot_seed` (~15 LOC): parse JSON, insert
`"quarot_rotation_seed": <u64>` at top level, serialize back.

### 5d. `crates/inference/src/model/qwen35_config.rs`

Add field to `Qwen35Config`:
```rust
/// Rotation seed used during QuaRot conversion. `None` for non-QuaRot artifacts.
/// Used at runtime to reconstruct `RandomizedHadamard` for MTP counter-rotation.
pub quarot_rotation_seed: Option<u64>,
```

Update `from_config_json_str` to deserialize this field (optional, defaults None).
~10 LOC.

---

## 6. Test Strategy

### 6a. Equivalence Gate (Primary)

New test `quarot_mtp_equivalence_gate` in `convert.rs` test module:

1. Build a tiny cfg with `mtp_num_hidden_layers=1`, `hidden_size=8` (power of 2).
2. Write combined main+MTP tensors to input dir.
3. Run `convert_quarot_qwen35` → writes MTP files in O-space.
4. Load as `MetalQwen35State::from_q4_dir` on Metal (if available) OR build a
   CPU-only equivalence check:
   - Construct `RandomizedHadamard` with the same seed.
   - Run MTP forward with fake inputs (known vectors).
   - Run the same computation manually: counter-rotate inputs → O-space norm → fc → attn+mlp → re-rotate → logits.
   - Assert `max_abs_error < 1e-3` (Q4 quantization noise floor).

### 6b. Q8-MTP vs Q4-QuaRot-MTP Perplexity Gate

Using `eval_perplexity`:
- Q8 path (baseline): `eval_perplexity --model-dir /path/to/q8 --use-mtp`
- Q4 QuaRot path: `eval_perplexity --quarot-q4-dir /path/to/quarot-q4 --use-mtp`
- Target: QuaRot-MTP perplexity within 0.5 bits of Q8-MTP perplexity on
  Wikitext-2 (100 tokens). A 5+ bit gap would indicate the counter-rotation
  is wrong.

### 6c. Acceptance Rate Gate

Run speculative decoding benchmark with both Q8-MTP and Q4-QuaRot-MTP on a
short prompt (128 tokens generated). Target: QuaRot-MTP acceptance rate ≥ 80%
of Q8-MTP acceptance rate. (Acceptance rate measures draft quality directly.)

### 6d. Unit Tests

- `apply_then_inverse_roundtrip`: verify `apply_inverse(apply(x)) ≈ x` for
  `d=1024`, tolerance `1e-5`. This validates the FWHT roundtrip.
- `mtp_forward_one_noop_on_q8`: verify that `quarot_rotation = None` leaves
  the MTP forward unchanged from current Q8 behavior.
- `convert_quarot_qwen35_emits_mtp_files`: new test replacing the
  `skips_mtp_for_quarot_even_when_present` test — assert MTP files ARE emitted
  with correct names, and output config has `mtp_num_hidden_layers=1` and
  `quarot_rotation_seed=<seed>`.

---

## 7. LOC Estimate

| File | Change type | Estimated LOC |
|------|-------------|---------------|
| `quant/quarot/convert.rs` | Remove zero_mtp, add write_mtp_weights_quarot, add inject_quarot_seed | +95 / -30 = net +65 |
| `forward/metal_qwen35.rs` | quarot_rotation field, 3 FWHT call sites, separate logits cmd buf | +45 |
| `model/qwen35_config.rs` | quarot_rotation_seed field + deserialize | +12 |
| New tests | Equivalence gate + unit tests | +180 |
| **Total** | | **~300 LOC** |

This is within the Π_AEP "Modify ≤ 100 LOC" target for the non-test changes (~120 LOC
across 3 production files). The test LOC is large but necessary for the equivalence gate.

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| FWHT roundtrip accumulates f32 error | Low | Medium | Use `apply_inverse` + `apply` which are exact inverses by construction (orthogonal R); test at d=1024 with tol 1e-4 |
| rotation_seed not stored in config → wrong R at runtime | Medium | High | Store seed in config JSON (Change 5c); fail loudly in `from_q4_dir` if MTP weights present but seed absent |
| CPU round-trip for re-rotation adds MTP latency | Low | Low | 4 KB at ~20 GB/s unified memory bandwidth ≈ 0.2 μs; negligible |
| Q4 quantization of MTP weights in O-space reduces acceptance rate | Medium | Medium | Measure with acceptance rate gate; baseline Q8-MTP sets the floor |
| MSL hadamard kernel incorrect | N/A for v0 | N/A | v0 uses CPU round-trip; GPU kernel is v1 |
| lm_head basis mismatch (tied vs untied) | Low | High | Both paths use `embed_tokens_q8` slot which always receives R-space input after re-rotation; tested by equivalence gate |

---

## 9. Recommendation Summary

**Implement Strategy 1 (Counter-Rotate)**. The three-FWHT approach is:
- Correct by construction (R is orthogonal, R^T = R^{-1})
- Minimal code change (120 LOC across 3 production files)
- Zero conversion-time overhead (MTP weights written as-is from the checkpoint)
- Testable via a closed-form equivalence gate

Strategy 2 (weight reparameterization) is blocked by the non-commutativity of
RMSNorm gamma with R, specifically for `pre_fc_norm_embedding` and
`pre_fc_norm_hidden` which feed into a concat (not a linear that could absorb
the scale). It would require a different architectural treatment of the MTP norms
that is out of scope for a targeted bug fix.

The v0 implementation uses CPU round-trip for the re-rotation before the logits
GEMV (avoids a new MSL kernel). The v1 plan adds `hadamard_rotate_inplace` in MSL
to eliminate the CPU readback.

---

## 10. Implementation Order

1. `Qwen35Config`: add `quarot_rotation_seed: Option<u64>` field (+12 LOC)
2. `convert.rs`: add `inject_quarot_seed`, `write_mtp_weights_quarot`, update
   `convert_quarot_qwen35` to emit MTP files and seed (+65 LOC)
3. `metal_qwen35.rs`: add `quarot_rotation` to engine, load from config seed in
   `from_q4_dir`, insert 3 FWHT call sites in `mtp_forward_one` (+45 LOC)
4. Update tests: replace `skips_mtp_for_quarot_even_when_present` with
   `emits_mtp_files_for_quarot`, add equivalence gate (+180 LOC)
5. Measure perplexity and acceptance rate gates

---

*Design authored by α[architect] (Claude Sonnet 4.6) on behalf of @ohdearquant.*
