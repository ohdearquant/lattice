//! [`RotationPlan`]: which rotation is absorbed into which tensor, on which
//! side. This module owns the architecture-specific recipes (currently
//! Qwen3.5 hybrid) — the math primitives in [`super::rotation`] are
//! architecture-agnostic, this module says "in Qwen3.5, the attention
//! `q_proj` takes the residual-stream rotation on its input side."
//!
//! Step 3 of [ADR-044](../../../../../docs/adr/ADR-044-quarot-rotated-quantization.md)
//! glues this plan to SafeTensors I/O and the Q4 bridge.
//!
//! ## Scope of this slice
//!
//! Plan covers the **linear-layer tensors that read or write the residual
//! stream** in Qwen3.5 hybrid (both GQA full-attention layers and
//! GatedDeltaNet linear-attention layers):
//!
//! | Tensor | Reads from residual | Writes to residual | Storage shape | Absorption | Why |
//! |---|---|---|---|---|---|
//! | `self_attn.q_proj.weight` | ✓ | — | `[q_dim, hidden]` | input-side | hidden = `cols` |
//! | `self_attn.k_proj.weight` | ✓ | — | `[kv_dim, hidden]` | input-side | hidden = `cols` |
//! | `self_attn.v_proj.weight` | ✓ | — | `[kv_dim, hidden]` | input-side | hidden = `cols` |
//! | `self_attn.o_proj.weight` | — | ✓ | `[hidden, q_dim]` | output-side | hidden = `rows` |
//! | `mlp.gate_proj.weight` | ✓ | — | `[intermediate, hidden]` | input-side | hidden = `cols` |
//! | `mlp.up_proj.weight` | ✓ | — | `[intermediate, hidden]` | input-side | hidden = `cols` |
//! | `mlp.down_proj.weight` | — | ✓ | `[hidden, intermediate]` | output-side | hidden = `rows` |
//! | `linear_attn.in_proj_qkv.weight` | ✓ | — | `[qkv_dim, hidden]` | input-side | hidden = `cols` |
//! | `linear_attn.in_proj_z.weight` | ✓ | — | `[output_dim, hidden]` | input-side | hidden = `cols` |
//! | `linear_attn.in_proj_b.weight` | ✓ | — | `[num_heads, hidden]` | input-side | hidden = `cols` |
//! | `linear_attn.in_proj_a.weight` | ✓ | — | `[num_heads, hidden]` | input-side | hidden = `cols` |
//! | `linear_attn.out_proj.weight` | — | ✓ | `[hidden, output_dim]` | output-side | hidden = `rows` |
//! | `lm_head.weight` (untied) | ✓ | — | `[vocab_size, hidden]` | input-side | hidden = `cols` (each row is a linear functional on the hidden state) |
//! | `embed_tokens.weight` | — | ✓ | `[vocab_size, hidden]` | **input-side** | despite being a "writer", the storage shape `[vocab_size, hidden]` means each row IS an embedding vector of dim hidden; rotating those rows produces a rotated output, which corresponds to **`W ← W · R^T`** (input-side absorption on the hidden dimension). Output-side absorption would try to match R against `vocab_size` rows and fail dimensionally. |
//!
//! Tied embeddings (Qwen3.5 default): when `tie_word_embeddings=true` the
//! logits weight is `embed_tokens` itself (`logits_weight()` in
//! `weights.rs:51`). Input-side absorption on `embed_tokens` gives the
//! right answer for both uses: as embedding it produces `R · embedding`
//! (rotated output written to residual); as tied lm_head it consumes
//! `R · hidden_state` (rotated input read from residual) and produces
//! `W · hidden_original = logits_original` — both correct from a single
//! absorption.
//!
//! ## Known gaps (BLOCKERS for real-model conversion — step 3c/3d must address)
//!
//! - **RMSNorm `(1 + gamma)` scaling does not commute with Hadamard
//!   rotation in general.** Qwen3.5 applies RMSNorm with shifted scale
//!   `(1 + gamma)` (norm.rs:16) at `input_layernorm`,
//!   `post_attention_layernorm`, `final_norm`, and inside GatedDeltaNet's
//!   `linear_attn.norm`. Linear-weight absorption alone does NOT preserve
//!   the model output. The QuaRot paper fuses `(1 + gamma)` into the
//!   immediately-following linear layer (so each `g_i` becomes part of
//!   the next `W`'s rows); after fusion, the normalize-only step is
//!   rotation-invariant. This fusion is NOT in step 3a — step 3c or 3d
//!   must implement it before any real-model conversion can be claimed
//!   correct.
//! - **No coverage validation against the loader's expected tensor list.**
//!   `for_tensor` matches by suffix, so a stray `mtp.*` or future-named
//!   tensor in the SafeTensors file (ignored by the runtime loader per
//!   ADR-043 §Out-of-scope) could be rotated without affecting the
//!   model. Step 3b's SafeTensors reader must call `validate_coverage`
//!   to assert: every plan rule matches ≥ 1 tensor, every loaded tensor
//!   in the residual-touching set matches exactly one plan rule.
//!
//! ## Deferred (correctly v1)
//!
//! - Per-head-dim rotations on QKV head spaces (improves activation
//!   quantization; not needed for weight-only Q4)
//! - MoE expert weights (DeepSeekMoE-style routed experts in Qwen3.5 MoE
//!   layers — same absorption pattern but applied per-expert slice;
//!   tensor names `mlp.experts.gate_up_proj`, `mlp.experts.down_proj`)

use crate::error::InferenceError;
use crate::quant::quarot::hadamard::RandomizedHadamard;
use crate::quant::quarot::rotation::{
    absorb_input_rotation, absorb_input_rotation_f64, absorb_output_rotation,
    absorb_output_rotation_f64,
};

/// Which side of a linear layer's weight matrix gets the rotation absorbed.
///
/// `InputSide`: `W ← W · R^T`. Used when the layer reads from a residual
/// stream that gets pre-rotated by `R` upstream.
///
/// `OutputSide`: `W ← R · W`. Used when the layer writes to a residual
/// stream that should be pre-rotated by `R` for downstream consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsorptionSide {
    InputSide,
    OutputSide,
}

/// Plan for a single weight tensor: which rotation goes on which side.
///
/// A tensor with no rotation should not appear in the plan at all —
/// `RotationPlan::for_tensor` returns `None` for unplanned tensors so the
/// caller can decide whether to pass it through unchanged or warn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorRotation {
    pub side: AbsorptionSide,
    pub rotation_id: RotationId,
}

/// Stable identifier for a planned rotation. The actual [`RandomizedHadamard`]
/// is constructed once from `(seed, dim)` when the plan is materialized for
/// execution — the plan itself does not own rotations so it stays cheap to
/// clone and serialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RotationId {
    /// `R_res` of dimension `hidden_size`.
    ResidualStream,
}

/// Plan binding tensor-name patterns to [`TensorRotation`] entries.
///
/// Patterns are matched as suffixes of the tensor's SafeTensors name —
/// e.g., a pattern `"self_attn.q_proj.weight"` matches
/// `"model.layers.0.self_attn.q_proj.weight"` and
/// `"model.layers.23.self_attn.q_proj.weight"`. This is intentionally simple;
/// Qwen3.5's naming is regular enough that suffix-matching covers it
/// without a full glob engine.
#[derive(Debug, Clone)]
pub struct RotationPlan {
    rules: Vec<(String, TensorRotation)>,
}

impl RotationPlan {
    /// Plan rules for **the linear-layer subset** of Qwen3.5 residual-stream
    /// rotation. Covers GQA + GDN + dense MLP + embed/lm_head.
    ///
    /// **NOT a complete v0 conversion plan.** Two known blockers (see module doc
    /// §Known gaps): (1) RMSNorm `(1+gamma)` does not commute with Hadamard,
    /// must be fused into the following linear layer before the conversion
    /// is correctness-preserving; (2) no coverage validation against the
    /// loader's tensor list. Step 3c/3d will close both before a real
    /// `quantize_quarot` binary runs.
    pub fn qwen35_residual_stream_linear_layers() -> Self {
        let r_in = TensorRotation {
            side: AbsorptionSide::InputSide,
            rotation_id: RotationId::ResidualStream,
        };
        let r_out = TensorRotation {
            side: AbsorptionSide::OutputSide,
            rotation_id: RotationId::ResidualStream,
        };
        Self {
            rules: vec![
                // GQA full-attention layers
                ("self_attn.q_proj.weight".into(), r_in),
                ("self_attn.k_proj.weight".into(), r_in),
                ("self_attn.v_proj.weight".into(), r_in),
                ("self_attn.o_proj.weight".into(), r_out),
                // GDN linear-attention layers — same absorption pattern as GQA
                ("linear_attn.in_proj_qkv.weight".into(), r_in),
                ("linear_attn.in_proj_z.weight".into(), r_in),
                ("linear_attn.in_proj_b.weight".into(), r_in),
                ("linear_attn.in_proj_a.weight".into(), r_in),
                ("linear_attn.out_proj.weight".into(), r_out),
                // Dense MLP layers
                ("mlp.gate_proj.weight".into(), r_in),
                ("mlp.up_proj.weight".into(), r_in),
                ("mlp.down_proj.weight".into(), r_out),
                // Embedding + LM head (each row is a hidden-dim functional;
                // input-side absorption applies R to the hidden dim regardless
                // of whether the tensor is used as embedding or lm_head)
                ("embed_tokens.weight".into(), r_in),
                ("lm_head.weight".into(), r_in),
            ],
        }
    }

    /// Look up the rotation for a tensor by its SafeTensors name.
    ///
    /// Suffix-match against the rule patterns. Step 3b's SafeTensors reader
    /// MUST cross-check this against the loader's expected tensor list via
    /// [`Self::validate_coverage`] before rotating anything — `for_tensor`
    /// alone is too permissive (it would match a future-added or unused
    /// tensor with a colliding suffix).
    pub fn for_tensor(&self, name: &str) -> Option<TensorRotation> {
        self.rules
            .iter()
            .find(|(pat, _)| name.ends_with(pat))
            .map(|(_, r)| *r)
    }

    /// Number of pattern rules in the plan. Useful for dimensional sanity
    /// checks but not for runtime dispatch.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Cross-check the plan against an actual SafeTensors tensor list.
    ///
    /// Returns a [`CoverageReport`] listing:
    /// - which plan rule patterns matched zero tensors (likely typo or
    ///   wrong model architecture)
    /// - which loaded tensors matched a plan rule (the rotation targets)
    ///
    /// Step 3b's binary should refuse-on-fail when any rule has zero matches —
    /// that signals the plan was built for a different architecture than
    /// the loaded SafeTensors. Step 3c's correctness pass should also assert
    /// that every residual-touching tensor in the loader's expected list IS
    /// in the plan; that check requires the loader's list, which lives in
    /// `model/qwen35/loading.rs` and is not imported here to keep this module
    /// architecture-agnostic at the dispatch layer.
    pub fn validate_coverage<'a, I>(&self, tensor_names: I) -> CoverageReport
    where
        I: IntoIterator<Item = &'a str>,
    {
        let names: Vec<&str> = tensor_names.into_iter().collect();
        let mut matched_tensors: Vec<String> = Vec::new();
        let mut unmatched_rules: Vec<String> = Vec::new();

        for (pat, _) in &self.rules {
            let any = names.iter().any(|n| n.ends_with(pat));
            if !any {
                unmatched_rules.push(pat.clone());
            }
        }
        for name in &names {
            if self.for_tensor(name).is_some() {
                matched_tensors.push((*name).to_string());
            }
        }
        CoverageReport {
            matched_tensors,
            unmatched_rules,
        }
    }
}

/// Result of [`RotationPlan::validate_coverage`]. `unmatched_rules` being
/// non-empty indicates a plan-vs-model mismatch; the conversion binary
/// should refuse to proceed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageReport {
    pub matched_tensors: Vec<String>,
    pub unmatched_rules: Vec<String>,
}

impl CoverageReport {
    /// `true` iff every plan rule matched at least one tensor name.
    pub fn is_complete(&self) -> bool {
        self.unmatched_rules.is_empty()
    }
}

/// Apply a planned rotation to a single weight tensor (f32 in place).
///
/// Returns `Ok(true)` if the tensor matched a plan rule and was rotated,
/// `Ok(false)` if the tensor was not in the plan (caller decides whether
/// to pass through or warn), or an error on shape / dimension mismatch.
pub fn apply_tensor_rotation(
    name: &str,
    weight: &mut [f32],
    rows: usize,
    cols: usize,
    plan: &RotationPlan,
    residual_rotation: &RandomizedHadamard,
) -> Result<bool, InferenceError> {
    let Some(tr) = plan.for_tensor(name) else {
        return Ok(false);
    };
    let rotation = match tr.rotation_id {
        RotationId::ResidualStream => residual_rotation,
    };
    match tr.side {
        AbsorptionSide::InputSide => absorb_input_rotation(weight, rows, cols, rotation)?,
        AbsorptionSide::OutputSide => absorb_output_rotation(weight, rows, cols, rotation)?,
    }
    Ok(true)
}

/// `f64` variant of [`apply_tensor_rotation`]. Step 3 will use this when
/// running the absorption pass in f64 precision per ADR-044 §Risks.
pub fn apply_tensor_rotation_f64(
    name: &str,
    weight: &mut [f64],
    rows: usize,
    cols: usize,
    plan: &RotationPlan,
    residual_rotation: &RandomizedHadamard,
) -> Result<bool, InferenceError> {
    let Some(tr) = plan.for_tensor(name) else {
        return Ok(false);
    };
    let rotation = match tr.rotation_id {
        RotationId::ResidualStream => residual_rotation,
    };
    match tr.side {
        AbsorptionSide::InputSide => absorb_input_rotation_f64(weight, rows, cols, rotation)?,
        AbsorptionSide::OutputSide => absorb_output_rotation_f64(weight, rows, cols, rotation)?,
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen35_plan_misses_moe_expert_weights() {
        // Documents the v1 deferral — MoE expert tensors are NOT in the v0 plan.
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let moe_names = [
            "model.layers.0.mlp.experts.gate_up_proj.weight",
            "model.layers.0.mlp.experts.down_proj.weight",
        ];
        for name in moe_names {
            assert!(
                plan.for_tensor(name).is_none(),
                "MoE expert tensor {name} should not match v0 plan"
            );
        }
    }

    #[test]
    fn apply_tensor_rotation_skips_unplanned() {
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let hidden = 64;
        let r = RandomizedHadamard::new(7, hidden).unwrap();
        let mut weight = vec![1.0_f32; hidden * hidden];
        let weight_copy = weight.clone();

        let rotated = apply_tensor_rotation(
            "model.layers.0.linear_attn.in_proj.weight",
            &mut weight,
            hidden,
            hidden,
            &plan,
            &r,
        )
        .unwrap();
        assert!(!rotated, "unplanned tensor should report not rotated");
        assert_eq!(weight, weight_copy, "unplanned tensor must not be mutated");
    }

    #[test]
    fn apply_tensor_rotation_mlp_layer_pair_rotates_output() {
        // Dense MLP layer pair: gate_proj (input-side) + down_proj (output-side).
        // Rotation R is on the residual stream. After absorption:
        //   gate_proj' · (R · x) = gate_proj · x   (intermediate unchanged)
        //   down_proj' · intermediate = R · (down_proj · intermediate) = R · y_original
        // So the rotated pipeline output equals R · y_original, NOT y_original —
        // because down_proj writes BACK to the rotated residual stream.
        let hidden = 64;
        let intermediate = 128;
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let r = RandomizedHadamard::new(0xC0FFEE, hidden).unwrap();

        let mut state = 1_u64;
        let mut rand = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as u32 as f32 / u32::MAX as f32) - 0.5
        };
        let gate_proj: Vec<f32> = (0..intermediate * hidden).map(|_| rand()).collect();
        let down_proj: Vec<f32> = (0..hidden * intermediate).map(|_| rand()).collect();
        let x: Vec<f32> = (0..hidden).map(|_| rand()).collect();

        let matvec = |w: &[f32], r: usize, c: usize, v: &[f32]| -> Vec<f32> {
            (0..r)
                .map(|i| (0..c).map(|j| w[i * c + j] * v[j]).sum())
                .collect()
        };
        let intermediate_out = matvec(&gate_proj, intermediate, hidden, &x);
        let y_original = matvec(&down_proj, hidden, intermediate, &intermediate_out);
        let mut y_expected = y_original.clone();
        r.apply(&mut y_expected).unwrap();

        let mut gate_proj_abs = gate_proj.clone();
        let mut down_proj_abs = down_proj.clone();
        assert!(
            apply_tensor_rotation(
                "model.layers.0.mlp.gate_proj.weight",
                &mut gate_proj_abs,
                intermediate,
                hidden,
                &plan,
                &r,
            )
            .unwrap()
        );
        assert!(
            apply_tensor_rotation(
                "model.layers.0.mlp.down_proj.weight",
                &mut down_proj_abs,
                hidden,
                intermediate,
                &plan,
                &r,
            )
            .unwrap()
        );

        let mut x_rotated = x.clone();
        r.apply(&mut x_rotated).unwrap();
        let intermediate_rot = matvec(&gate_proj_abs, intermediate, hidden, &x_rotated);
        let y_rotated = matvec(&down_proj_abs, hidden, intermediate, &intermediate_rot);

        let max_abs_diff = y_expected
            .iter()
            .zip(y_rotated.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_diff < 1e-3,
            "MLP pair output should equal R · y_original: max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn f64_apply_matches_f32_apply_pattern() {
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let hidden = 64;
        let r = RandomizedHadamard::new(11, hidden).unwrap();
        let mut weight_f32: Vec<f32> = (0..hidden * hidden)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let mut weight_f64: Vec<f64> = weight_f32.iter().map(|&v| f64::from(v)).collect();

        let r1 = apply_tensor_rotation(
            "model.layers.0.self_attn.q_proj.weight",
            &mut weight_f32,
            hidden,
            hidden,
            &plan,
            &r,
        )
        .unwrap();
        let r2 = apply_tensor_rotation_f64(
            "model.layers.0.self_attn.q_proj.weight",
            &mut weight_f64,
            hidden,
            hidden,
            &plan,
            &r,
        )
        .unwrap();
        assert!(r1 && r2);

        for (i, (a, b)) in weight_f32.iter().zip(weight_f64.iter()).enumerate() {
            let delta = (f64::from(*a) - b).abs();
            assert!(
                delta < 1e-5,
                "tensor[{i}]: f32={a} vs f64={b}, delta={delta}"
            );
        }
    }
}
