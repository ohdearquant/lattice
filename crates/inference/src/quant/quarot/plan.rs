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
//! | Tensor | Reads from residual | Writes to residual | Storage shape (Qwen3.5) | Absorption | Why |
//! |---|---|---|---|---|---|
//! | `self_attn.q_proj.weight` | ✓ | — | **`[2*q_dim, hidden]`** (Qwen3.5 fuses Q + gate_z into one matrix per `weights.rs:8`; plain Qwen3 is `[q_dim, hidden]`) | input-side | hidden = `cols`; absorption applies row-wise so the gate-z half rotates correctly alongside the Q half |
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
//! | `lm_head.weight` (untied only) | ✓ | — | `[vocab_size, hidden]` | input-side | hidden = `cols`; **optional rule** — Qwen3.5 ties embeddings (`tie_word_embeddings=true` at `qwen35_config.rs:177`) so `lm_head.weight` is absent from the SafeTensors file in the default Qwen3.5 builds. `validate_coverage` does not treat its absence as a coverage failure. |
//! | `embed_tokens.weight` | — | ✓ | `[vocab_size, hidden]` | **input-side** | despite being a "writer", the storage shape `[vocab_size, hidden]` means each row IS an embedding vector of dim hidden; rotating those rows produces a rotated output, which corresponds to **`W ← W · R^T`** (input-side absorption on the hidden dimension). Output-side absorption would try to match R against `vocab_size` rows and fail dimensionally. |
//!
//! ## Tied embeddings + final-RMSNorm: only one correct path
//!
//! Step 3c/3d MUST implement the QuaRot fusion of the final-norm scale
//! into an **untied** `lm_head`, AND set the runtime `final_norm.weight`
//! (the `(1 + g_final)` Qwen3.5 shifted-scale at `norm.rs:16`) to the
//! neutral value (gamma = 0, so `D = I`).
//!
//! ### Why no "keep final-RMSNorm online" alternative
//!
//! Qwen3.5's forward at `forward.rs:81` is norm-then-linear with shifted
//! RMSNorm: `n = D · normalize(h)` then `logits = W_lm · n`, where
//! `D = diag(1 + g_final)`. After QuaRot, the residual is in the rotated
//! basis (`h_rot = R · h`). Using normalize's rotation-invariance
//! (`||R · x|| = ||x||` so `normalize(R · h) = R · normalize(h)`), the
//! online-runtime computes:
//!
//! ```text
//!   n_rot       = D · normalize(h_rot) = D · R · normalize(h)
//!   logits_run  = lm_head' · n_rot     = lm_head' · D · R · normalize(h)
//! ```
//!
//! For `logits_run = logits_original = W_lm · D · normalize(h)`, we need
//! `lm_head' · D · R = W_lm · D`, i.e., `lm_head' = W_lm · D · R^T · D^{-1}`.
//! That is **not** a clean input-side rotation of `W_lm`: it depends on
//! `D` non-commutatively (`D` is diagonal, `R` is dense Hadamard, they
//! do not commute). Any "rotate lm_head input-side and leave final norm
//! alone" recipe is wrong — for both tied and untied embeddings.
//!
//! ### The only correct fusion path
//!
//! Offline (step 3c performs ALL of these):
//! 1. **Untie embeddings**: materialize `lm_head` as a separate tensor.
//!    When the input model has `tie_word_embeddings=true`, the converter
//!    must copy `embed_tokens` into `lm_head` BEFORE rotation, so the
//!    two roles can be transformed independently.
//! 2. **Flip the output config** to `tie_word_embeddings=false`. The
//!    runtime loader at `loading.rs:266` only loads `lm_head.weight`
//!    when `cfg.tie_word_embeddings` is false; without this flip, the
//!    runtime falls back to `embed_tokens` via `logits_weight()` at
//!    `weights.rs:51` and the fused `lm_head` is never consulted —
//!    silently producing wrong logits. **This is a config-mutation
//!    requirement, NOT just a weight transform.**
//! 3. **Fuse final-norm scale into the new lm_head**:
//!    `lm_head_fused := W_lm · diag(1 + g_final)` (column multiply on
//!    `W_lm` storage, per the Known gaps section).
//! 4. **Rotate**: apply input-side absorption to `lm_head_fused`:
//!    `lm_head_final := lm_head_fused · R^T = W_lm · D · R^T`.
//! 5. **Zero out the runtime final-norm scale**: in the saved model,
//!    set `final_norm.weight` such that the shifted formula
//!    `(1 + g)` evaluates to `1` (i.e., `g = 0`).
//! 6. The plain `embed_tokens` continues to absorb just `R^T`
//!    (input-side, as in this plan).
//!
//! Runtime then computes:
//! `lm_head_final · normalize(h_rot) = W_lm · D · R^T · R · normalize(h)
//! = W_lm · D · normalize(h) = logits_original`. Correct.
//!
//! `lm_head.weight` is Optional in THIS plan because the input
//! SafeTensors may not contain it pre-conversion (tied case). The
//! converter at step 3c materializes it. After conversion, the saved
//! model is always-untied (step 2 above), so `validate_coverage` on
//! the OUTPUT SafeTensors will see `lm_head.weight` present and the
//! rule satisfied.
//!
//! ## Known gaps — BLOCKERS for real-model conversion that this plan DOES NOT capture
//!
//! **This plan is rotation-rule data only.** `validate_coverage().is_complete()`
//! verifies suffix-presence of the rotation targets — nothing more. A
//! converter that rotates every planned tensor and stops there will
//! **produce wrong logits**. Step 3c's binary owns a separate
//! `ConversionPlan` (TBD) that bundles this rotation plan with all the
//! mutations below; only `ConversionPlan::validate` should be treated
//! as the full-correctness gate.
//!
//! Required mutations NOT captured in this plan:
//!
//! - **Shifted RMSNorm `(1 + gamma)` fusion + neutralization.** Qwen3.5
//!   applies RMSNorm with shifted scale `(1 + gamma)` at `norm.rs:16` for
//!   `input_layernorm` (forward.rs:41), `post_attention_layernorm`
//!   (forward.rs:66), and `final_norm` (forward.rs:81). A diagonal
//!   scale does not commute with Hadamard rotation. The QuaRot paper
//!   fuses `(1 + gamma)` into the immediately-following linear layer
//!   as a column multiply: `W ← W · diag(1 + g)` (storage:
//!   `W[i, j] *= (1 + g[j])`), then sets the runtime `*_layernorm.weight`
//!   to the neutral value (`gamma = 0`, so the shifted formula returns
//!   `1`). The converter must do this fusion for input_layernorm,
//!   post_attention_layernorm, AND final_norm (the final_norm + lm_head
//!   case is spelled out separately above).
//! - **GDN `linear_attn.norm` is different — internal, plain `gamma`.**
//!   GDN's gated-RMSNorm at `gdn.rs:425, 440` and `gdn_fused.rs:127`
//!   multiplies plain `gamma[i]`, NOT `(1 + gamma[i])`. More
//!   importantly, this norm runs INSIDE the GDN block, between the
//!   linear-attention compute and `out_proj`. The residual-stream
//!   rotation enters at `in_proj_*` (input-side absorbed) and exits at
//!   `out_proj` (output-side absorbed); it does not cross
//!   `linear_attn.norm`. So the plain-gamma GDN norm does NOT need
//!   QuaRot fusion — it operates on internal GDN state in a basis
//!   unaffected by residual rotation. Step 3c must NOT attempt to fuse
//!   this norm.
//! - **Tied-embedding untying + `tie_word_embeddings=false` config flip**
//!   (described in detail in the §Tied embeddings section above).
//!
//! Required loader-side helper not yet exposed:
//!
//! - **`qwen_required_tensor_names(cfg)` is `#[cfg(test)]`-only** at
//!   `model/qwen35/mod.rs:44`. Step 3b's conversion binary cannot call
//!   it as written. Step 3b must promote this re-export out of the
//!   test cfg before the per-layer coverage check can run.
//!
//! Out-of-scope (silently-broken) integrations:
//!
//! - **Runtime LoRA injection is incompatible with QuaRot-converted
//!   models.** The forward path applies LoRA deltas after the base
//!   matmul using the same activation, e.g., `forward.rs:249, 467` and
//!   `gdn_fused.rs:385`. The rotated base projection produces output in
//!   a different basis than an un-rotated LoRA adapter delta, so summing
//!   them is invalid. v0 marks QuaRot models as LoRA-runtime-incompatible.
//!   A future LoRA-aware path would either (a) rotate adapter weights
//!   on load using the converter's seed (requires the seed in the
//!   `.q4` artifact metadata), or (b) refuse to compose at adapter-load
//!   time when the base is QuaRot-converted.
//!
//! ## Deferred (correctly v1)
//!
//! - Per-head-dim rotations on QKV head spaces (improves activation
//!   quantization; not needed for weight-only Q4)
//! - MoE expert weights (DeepSeekMoE-style routed experts in Qwen3.5 MoE
//!   layers — same absorption pattern but applied per-expert slice;
//!   tensor names `mlp.experts.gate_up_proj`, `mlp.experts.down_proj`)
//! - Runtime LoRA compatibility (see above — currently incompatible)

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

/// Whether a rule must match at least one tensor to count as complete coverage.
///
/// `Required` rules are model-architecture invariants — `q_proj`, `o_proj`,
/// `embed_tokens`, etc. If a Required rule has zero matches, the plan is
/// being applied to the wrong architecture and the converter must refuse.
///
/// `Optional` rules cover tensors that may or may not be present depending
/// on config — `lm_head.weight` is the canonical case, absent when
/// `tie_word_embeddings=true` (Qwen3.5 default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleRequirement {
    Required,
    Optional,
}

#[derive(Debug, Clone)]
struct Rule {
    pattern: String,
    rotation: TensorRotation,
    requirement: RuleRequirement,
}

/// Plan binding tensor-name patterns to [`TensorRotation`] entries.
///
/// Patterns are matched as suffixes of the tensor's SafeTensors name —
/// e.g., a pattern `"self_attn.q_proj.weight"` matches
/// `"model.layers.0.self_attn.q_proj.weight"` and
/// `"model.layers.23.self_attn.q_proj.weight"`. This is intentionally
/// simple; Qwen3.5's naming is regular enough that suffix-matching
/// covers it without a full glob engine.
///
/// Construct one of the architecture-specific plans (currently only
/// [`Self::qwen35_residual_stream_linear_layers`]) and look up
/// per-tensor absorption via [`Self::for_tensor`]. For coverage sanity
/// checks against a SafeTensors file's tensor list see
/// [`Self::validate_coverage`] — but note that method's contract is
/// **suffix-presence only**, not per-layer coverage; the conversion
/// binary in step 3b/3c will need a stricter check against the loader's
/// config-derived expected tensor list.
#[derive(Debug, Clone)]
pub struct RotationPlan {
    rules: Vec<Rule>,
}

impl RotationPlan {
    /// Plan rules for **the linear-layer subset** of Qwen3.5 residual-stream
    /// rotation. Covers GQA + GDN + dense MLP + embed/lm_head.
    ///
    /// **NOT a complete v0 conversion plan.** Rotation rules only — see
    /// module doc §Known gaps for the full list of conversion-binary
    /// responsibilities not captured here: shifted RMSNorm `(1+gamma)`
    /// fusion + neutralization (input/post/final), `tie_word_embeddings=false`
    /// config flip, fused `lm_head` materialization, per-layer coverage
    /// against `qwen_required_tensor_names` (currently `#[cfg(test)]`-gated),
    /// and LoRA-runtime incompatibility marker. Step 3c/3d's
    /// `quantize_quarot` binary owns all of these.
    pub fn qwen35_residual_stream_linear_layers() -> Self {
        let r_in = TensorRotation {
            side: AbsorptionSide::InputSide,
            rotation_id: RotationId::ResidualStream,
        };
        let r_out = TensorRotation {
            side: AbsorptionSide::OutputSide,
            rotation_id: RotationId::ResidualStream,
        };
        let req = |pat: &str, rot: TensorRotation| Rule {
            pattern: pat.into(),
            rotation: rot,
            requirement: RuleRequirement::Required,
        };
        let opt = |pat: &str, rot: TensorRotation| Rule {
            pattern: pat.into(),
            rotation: rot,
            requirement: RuleRequirement::Optional,
        };
        Self {
            rules: vec![
                // GQA full-attention layers
                req("self_attn.q_proj.weight", r_in),
                req("self_attn.k_proj.weight", r_in),
                req("self_attn.v_proj.weight", r_in),
                req("self_attn.o_proj.weight", r_out),
                // GDN linear-attention layers — same absorption pattern as GQA
                req("linear_attn.in_proj_qkv.weight", r_in),
                req("linear_attn.in_proj_z.weight", r_in),
                req("linear_attn.in_proj_b.weight", r_in),
                req("linear_attn.in_proj_a.weight", r_in),
                req("linear_attn.out_proj.weight", r_out),
                // Dense MLP layers
                req("mlp.gate_proj.weight", r_in),
                req("mlp.up_proj.weight", r_in),
                req("mlp.down_proj.weight", r_out),
                // Embedding — always required
                req("embed_tokens.weight", r_in),
                // lm_head — optional in the INPUT SafeTensors. Absent when
                // tie_word_embeddings=true (Qwen3.5 default per
                // qwen35_config.rs:177); step 3c's converter materializes
                // it from embed_tokens AND flips tie_word_embeddings=false
                // in the output config so the runtime loads it. See module
                // §Tied embeddings for why the tied-fallback path is NOT
                // correctness-preserving.
                opt("lm_head.weight", r_in),
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
            .find(|rule| name.ends_with(&rule.pattern))
            .map(|rule| rule.rotation)
    }

    /// Number of pattern rules in the plan. Useful for dimensional sanity
    /// checks but not for runtime dispatch.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Cross-check the plan against an actual SafeTensors tensor list at
    /// the **suffix-presence** level.
    ///
    /// This validates that every Required rule pattern matches at least one
    /// tensor name in the input — it does NOT validate per-layer coverage
    /// (a model with 24 layers but only 1 layer's tensors present will
    /// pass this check), and it does NOT validate the non-rotation
    /// mutations required for a correct conversion (RMSNorm fusion,
    /// `tie_word_embeddings` flip, etc. — see module §Known gaps).
    ///
    /// Per-layer coverage requires the loader's config-derived expected
    /// name list, which lives at `model/qwen35/loading.rs:12` as
    /// `qwen_required_tensor_names(cfg)`. Step 3b must promote that
    /// re-export out of `#[cfg(test)]` at `model/qwen35/mod.rs:44`
    /// before the conversion binary can call it.
    ///
    /// Returns a [`CoverageReport`] with five lists:
    /// - `matched_tensors`: loaded tensors that matched some plan rule
    /// - `unplanned_tensors`: loaded tensors that matched NO plan rule
    ///   (caller decides — likely passes them through unchanged, but a
    ///   residual-touching tensor here means the plan is incomplete)
    /// - `unmatched_required_rules`: Required rules that matched zero
    ///   tensors. Non-empty → wrong architecture; converter must refuse.
    /// - `unmatched_optional_rules`: Optional rules that matched zero
    ///   tensors. Expected when the corresponding tensor is config-absent
    ///   (e.g., `lm_head.weight` under tied embeddings).
    /// - `ambiguous_tensors`: loaded tensors that matched ≥ 2 plan rules.
    ///   Should always be empty for the Qwen3.5 plan because the patterns
    ///   are disjoint; a non-empty list indicates a pattern collision bug.
    ///
    /// `is_complete()` returns true iff `unmatched_required_rules` and
    /// `ambiguous_tensors` are both empty (optional rules with zero
    /// matches are allowed; unplanned tensors are caller's call).
    ///
    /// **`is_complete()` is NOT a correctness gate.** It only validates
    /// the rotation rules' suffix presence. The conversion binary's
    /// full-validity check must additionally verify: RMSNorm fusion
    /// applied to input_layernorm/post_attention_layernorm/final_norm,
    /// `tie_word_embeddings` flipped to false, fused `lm_head`
    /// materialized, per-layer coverage against
    /// `qwen_required_tensor_names`, and forward-equivalence delta
    /// below threshold. See module §Known gaps.
    pub fn validate_coverage<'a, I>(&self, tensor_names: I) -> CoverageReport
    where
        I: IntoIterator<Item = &'a str>,
    {
        let names: Vec<&str> = tensor_names.into_iter().collect();
        let mut matched_tensors: Vec<String> = Vec::new();
        let mut unplanned_tensors: Vec<String> = Vec::new();
        let mut ambiguous_tensors: Vec<String> = Vec::new();
        let mut unmatched_required_rules: Vec<String> = Vec::new();
        let mut unmatched_optional_rules: Vec<String> = Vec::new();

        for rule in &self.rules {
            let any = names.iter().any(|n| n.ends_with(&rule.pattern));
            if !any {
                match rule.requirement {
                    RuleRequirement::Required => {
                        unmatched_required_rules.push(rule.pattern.clone())
                    }
                    RuleRequirement::Optional => {
                        unmatched_optional_rules.push(rule.pattern.clone())
                    }
                }
            }
        }
        for name in &names {
            let match_count = self
                .rules
                .iter()
                .filter(|rule| name.ends_with(&rule.pattern))
                .count();
            match match_count {
                0 => unplanned_tensors.push((*name).to_string()),
                1 => matched_tensors.push((*name).to_string()),
                _ => ambiguous_tensors.push((*name).to_string()),
            }
        }
        CoverageReport {
            matched_tensors,
            unplanned_tensors,
            ambiguous_tensors,
            unmatched_required_rules,
            unmatched_optional_rules,
        }
    }
}

/// Result of [`RotationPlan::validate_coverage`]. See that method's docs for
/// the exact semantics of each field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageReport {
    pub matched_tensors: Vec<String>,
    pub unplanned_tensors: Vec<String>,
    pub ambiguous_tensors: Vec<String>,
    pub unmatched_required_rules: Vec<String>,
    pub unmatched_optional_rules: Vec<String>,
}

impl CoverageReport {
    /// `true` iff every Required rule matched at least one tensor AND no
    /// tensor matched multiple rules (i.e., no architecture mismatch and
    /// no pattern-collision bug). Optional rules being unmatched is fine;
    /// unplanned tensors are not flagged here because their semantics are
    /// caller-specific (the SafeTensors file may contain unused tensors
    /// per ADR-043 §Out-of-scope).
    pub fn is_complete(&self) -> bool {
        self.unmatched_required_rules.is_empty() && self.ambiguous_tensors.is_empty()
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
    fn qwen35_plan_covers_residual_stream_tensors() {
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        assert_eq!(plan.rule_count(), 14);
        let cases = [
            // GQA full-attention
            (
                "model.layers.0.self_attn.q_proj.weight",
                AbsorptionSide::InputSide,
            ),
            (
                "model.layers.5.self_attn.k_proj.weight",
                AbsorptionSide::InputSide,
            ),
            (
                "model.layers.5.self_attn.v_proj.weight",
                AbsorptionSide::InputSide,
            ),
            (
                "model.layers.5.self_attn.o_proj.weight",
                AbsorptionSide::OutputSide,
            ),
            // GDN linear-attention
            (
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                AbsorptionSide::InputSide,
            ),
            (
                "model.layers.0.linear_attn.in_proj_z.weight",
                AbsorptionSide::InputSide,
            ),
            (
                "model.layers.0.linear_attn.in_proj_b.weight",
                AbsorptionSide::InputSide,
            ),
            (
                "model.layers.0.linear_attn.in_proj_a.weight",
                AbsorptionSide::InputSide,
            ),
            (
                "model.layers.0.linear_attn.out_proj.weight",
                AbsorptionSide::OutputSide,
            ),
            // MLP
            (
                "model.layers.23.mlp.gate_proj.weight",
                AbsorptionSide::InputSide,
            ),
            (
                "model.layers.23.mlp.up_proj.weight",
                AbsorptionSide::InputSide,
            ),
            (
                "model.layers.23.mlp.down_proj.weight",
                AbsorptionSide::OutputSide,
            ),
            // Embed + lm_head (both input-side)
            (
                "model.language_model.embed_tokens.weight",
                AbsorptionSide::InputSide,
            ),
            ("lm_head.weight", AbsorptionSide::InputSide),
        ];
        for (name, expected_side) in cases {
            let tr = plan
                .for_tensor(name)
                .unwrap_or_else(|| panic!("plan missed tensor {name}"));
            assert_eq!(tr.side, expected_side, "wrong side for {name}");
            assert_eq!(tr.rotation_id, RotationId::ResidualStream);
        }
    }

    #[test]
    fn qwen35_plan_misses_rmsnorm_and_scalar_weights() {
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let skipped = [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
            "model.layers.0.linear_attn.norm.weight",
            "model.layers.0.linear_attn.A_log",
            "model.layers.0.linear_attn.dt_bias",
            "model.layers.0.linear_attn.conv1d.weight",
        ];
        for name in skipped {
            assert!(
                plan.for_tensor(name).is_none(),
                "non-linear-layer tensor {name} should not match plan"
            );
        }
    }

    /// Required-rule list that step 3b's converter MUST find in any Qwen3.5
    /// hybrid SafeTensors file. Suffixes match the loader's expectations at
    /// `crates/inference/src/model/qwen35/loading.rs:16, 33`.
    fn qwen35_required_residual_tensor_suffixes() -> &'static [&'static str] {
        &[
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_a.weight",
            "linear_attn.out_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "embed_tokens.weight",
        ]
    }

    fn synthetic_qwen35_tensor_names(include_lm_head: bool) -> Vec<String> {
        let mut names: Vec<String> = qwen35_required_residual_tensor_suffixes()
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if s.contains("embed_tokens") {
                    format!("model.language_model.{s}")
                } else {
                    format!("model.layers.{i}.{s}")
                }
            })
            .collect();
        if include_lm_head {
            names.push("lm_head.weight".to_string());
        }
        names
    }

    #[test]
    fn validate_coverage_tied_embeddings_is_complete_without_lm_head() {
        // Qwen3.5 default: tie_word_embeddings=true, so lm_head.weight is
        // ABSENT from the SafeTensors file. The plan's lm_head rule is
        // Optional — its absence must NOT fail is_complete().
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let names = synthetic_qwen35_tensor_names(false);
        let report = plan.validate_coverage(names.iter().map(String::as_str));
        assert!(
            report.is_complete(),
            "tied-embeddings tensor list should yield complete coverage, got: {report:?}"
        );
        assert_eq!(
            report.unmatched_required_rules.len(),
            0,
            "no required rule should be missing: {:?}",
            report.unmatched_required_rules
        );
        assert_eq!(
            report.unmatched_optional_rules,
            vec!["lm_head.weight".to_string()],
            "lm_head should appear in unmatched_optional_rules"
        );
        assert_eq!(report.matched_tensors.len(), 13);
        assert!(report.ambiguous_tensors.is_empty());
    }

    #[test]
    fn validate_coverage_untied_embeddings_matches_lm_head() {
        // tie_word_embeddings=false case: lm_head.weight is loaded.
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let names = synthetic_qwen35_tensor_names(true);
        let report = plan.validate_coverage(names.iter().map(String::as_str));
        assert!(
            report.is_complete(),
            "untied case should be complete: {report:?}"
        );
        assert!(report.unmatched_optional_rules.is_empty());
        assert_eq!(report.matched_tensors.len(), 14);
    }

    #[test]
    fn validate_coverage_flags_unmatched_required_rule() {
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        // Drop o_proj — a Required rule
        let names: Vec<String> = synthetic_qwen35_tensor_names(false)
            .into_iter()
            .filter(|n| !n.ends_with("self_attn.o_proj.weight"))
            .collect();
        let report = plan.validate_coverage(names.iter().map(String::as_str));
        assert!(
            !report.is_complete(),
            "missing required rule should fail completeness"
        );
        assert_eq!(
            report.unmatched_required_rules,
            vec!["self_attn.o_proj.weight".to_string()]
        );
    }

    #[test]
    fn validate_coverage_lists_unplanned_tensors() {
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        let mut names = synthetic_qwen35_tensor_names(false);
        names.push("model.layers.0.input_layernorm.weight".to_string());
        names.push("model.mtp.head.weight".to_string());
        let report = plan.validate_coverage(names.iter().map(String::as_str));
        // input_layernorm + mtp are unplanned, but the required rules still all match
        assert!(
            report.is_complete(),
            "unplanned tensors should not break required-rule completeness"
        );
        assert_eq!(report.unplanned_tensors.len(), 2);
        assert!(
            report
                .unplanned_tensors
                .iter()
                .any(|n| n.ends_with("input_layernorm.weight"))
        );
    }

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
