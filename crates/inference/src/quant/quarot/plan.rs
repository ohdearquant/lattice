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
//! Loader-side helpers available to the converter:
//!
//! - [`crate::model::qwen35::qwen_required_tensor_names`] returns the
//!   per-config expected tensor list (per-layer, per-architecture
//!   variant). Step 3b promoted this out of `#[cfg(test)]`. Step 3c
//!   uses it to validate that every required tensor in the input
//!   safetensors is reachable before quantization begins.
//! - [`crate::quant::quarot::io::QuarotTensorReader`] (step 3b) provides
//!   the streaming f64 read path with single-file + sharded auto-detect.
//!
//! ## LoRA Composition (ADR-045, implemented)
//!
//! Runtime LoRA injection on QuaRot-converted models is supported through
//! `MetalQwen35State::load_lora_adapter(..., quarot_seed: Some(seed))`.
//! The loader counter-rotates adapter weights using the same seed that
//! produced the rotated base, so deltas land in the correct basis.
//! The caller must supply the seed explicitly (artifact metadata TBD).
//!
//! ## Deferred (correctly v1)
//!
//! - Per-head-dim rotations on QKV head spaces (improves activation
//!   quantization; not needed for weight-only Q4)
//! - MoE expert weights (DeepSeekMoE-style routed experts in Qwen3.5 MoE
//!   layers — same absorption pattern but applied per-expert slice;
//!   tensor names `mlp.experts.gate_up_proj`, `mlp.experts.down_proj`)

use std::collections::HashSet;

use crate::error::InferenceError;
use crate::model::qwen::QwenConfig;
use crate::model::qwen35_config::Qwen35Config;
use crate::quant::quarot::hadamard::{
    MAX_BLOCK_HADAMARD_BLOCKS, MAX_BLOCK_HADAMARD_LEN, RandomizedHadamard,
};
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
///
/// `AttentionOutputR3` and `MlpDownR4` are **contract-only** as of issue
/// #703 PR1: [`apply_tensor_rotation`] refuses them rather than silently
/// treating them as a no-op, because no plan in this PR references them and
/// no online (runtime) rotation is wired yet — see [`OnlineRotationSpec`]
/// for the artifact-level metadata these identifiers carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum RotationId {
    /// `R_res` of dimension `hidden_size`.
    ResidualStream,
    /// `R3`: full-attention gated context, immediately before `o_proj`.
    /// Scope is full-attention layers only (GDN excluded) — see
    /// [`OnlineRotationSpec::r3_full_attention`].
    AttentionOutputR3,
    /// `R4`: post-`SiLU × up` MLP activation, immediately before
    /// `down_proj`. Applies to every layer's dense MLP (both full-attention
    /// and GDN layers carry a dense MLP in Qwen3.5 hybrid) — see
    /// [`OnlineRotationSpec::r4_dense_mlp`].
    MlpDownR4,
}

impl RotationId {
    /// The physical activation site this rotation ID attaches to when used
    /// as an *online* (runtime-applied) rotation, or `None` if this ID has
    /// no online site (`ResidualStream`/R1 is purely an offline weight
    /// fusion — see [`apply_tensor_rotation`]'s refusal of the other two IDs
    /// there, which is the mirror-image rule: R3/R4 never go through offline
    /// absorption, R1 never goes through an online site).
    ///
    /// This is the single authoritative `RotationId` → [`OnlineTransformSite`]
    /// mapping. Anything that needs to know where an online rotation sits,
    /// or which tensor absorbs it (via
    /// [`OnlineTransformSite::weight_tensor_suffix`]), must go through this
    /// method rather than re-deriving the association.
    pub fn online_transform_site(self) -> Option<OnlineTransformSite> {
        match self {
            RotationId::ResidualStream => None,
            RotationId::AttentionOutputR3 => Some(OnlineTransformSite::AttentionOutputPreOProj),
            RotationId::MlpDownR4 => Some(OnlineTransformSite::MlpPreDownProj),
        }
    }
}

/// Which physical activation site an online (runtime-applied) rotation
/// attaches to. Distinct from [`AbsorptionSide`], which describes which side
/// of the *weight matrix* absorbs the offline counter-rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OnlineTransformSite {
    /// R3: full-attention gated context, immediately before `o_proj`.
    AttentionOutputPreOProj,
    /// R4: post-`SiLU × up` MLP activation, immediately before `down_proj`.
    MlpPreDownProj,
}

impl OnlineTransformSite {
    /// Suffix (matching [`RotationPlan::for_tensor`]'s suffix convention) of
    /// the weight tensor whose input side absorbs this site's online
    /// rotation. The authoritative link between "where the runtime rotation
    /// sits" and "which stored tensor counter-rotates it" — callers that
    /// need to know which tensors an online recipe affects (e.g.
    /// [`super::io::OnlineArtifactDescriptor::validate`]) must go through
    /// this method rather than re-deriving the tensor name independently.
    pub fn weight_tensor_suffix(self) -> &'static str {
        match self {
            OnlineTransformSite::AttentionOutputPreOProj => "self_attn.o_proj.weight",
            OnlineTransformSite::MlpPreDownProj => "mlp.down_proj.weight",
        }
    }
}

/// Artifact-level record of a single online (runtime) rotation: which
/// physical site it attaches to, which side of the consuming weight
/// absorbs its transpose (per `rotation.rs`'s documented `y = W R^T (R x)`
/// algebra), its seed(s)/block size, and the explicit layer scope.
///
/// **v0 scope (issue #703 PR1): validated at load, not yet executed.** The
/// load path deserializes this struct (via `io.rs`'s
/// `OnlineArtifactDescriptor`) and calls [`Self::validate`] on every spec it
/// carries — a structurally malformed R3/R4 recipe is already rejected
/// before an artifact can be used. What is still absent is the forward-path
/// consumer: no runtime code applies the recipe this struct describes to an
/// activation yet, so a *structurally valid* `V1Online` descriptor is
/// currently rejected at load too (see `read_quarot_seed_from_index`'s
/// fail-closed contract), pending the later PR that wires the runtime side.
/// The orientation recorded here (`side: InputSide` for both R3 and R4) is
/// the winning orientation proven by the one-layer reference test in
/// `r3_reference.rs` — see that module's doc comment for the full
/// enumeration result.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OnlineRotationSpec {
    pub id: RotationId,
    /// Which side of the consuming weight (`o_proj` for R3, `down_proj`
    /// for R4) absorbs the counter-rotation. Both R3 and R4 use
    /// `InputSide`: the *runtime* activation is rotated by `R`, and the
    /// weight's input side absorbs `R^T` offline — matching
    /// `rotation.rs`'s `y = W · R^T · (R · x)` identity exactly, because
    /// in both cases the transform sits on an activation that is the
    /// weight's *input*, not its output onto the residual stream.
    pub side: AbsorptionSide,
    pub seed: u64,
    /// Power-of-two block size dividing the transformed axis.
    ///
    /// For R3 this divides `num_attention_heads` — QuaRot's online
    /// "Hadamard heads" factor (paper Eq. 9: `H_num_heads ⊗ I_head_dim`)
    /// mixes values ACROSS heads at each fixed within-head channel, so the
    /// axis being block-Hadamard-transformed is the head axis, not
    /// `head_dim`. `block_size == num_attention_heads` reproduces the
    /// paper's single dense `H_num_heads` exactly; a smaller power-of-two
    /// divisor partitions the heads into independently-rotated groups —
    /// the same `BlockHadamard` fallback pattern already used for
    /// non-power-of-two axes elsewhere in this module (see
    /// [`super::hadamard::BlockHadamard`]'s doc), applied here so
    /// `num_attention_heads` values that are not themselves a power of two
    /// (e.g. Qwen3.6-27B's 24) still admit a valid R3 recipe instead of
    /// requiring a non-Hadamard randomized-orthogonal fallback.
    ///
    /// For R4 this divides `intermediate_size`.
    pub block_size: usize,
    /// Explicit layer indices this rotation applies to. `None` means every
    /// layer that carries the target tensor. R3 is always `Some` and lists
    /// only full-attention layer indices (GDN layers are excluded — the
    /// design doc's §B "Hybrid Qwen3.5" scoping rule: GDN attention is not
    /// paper-standard softmax attention and R3's derivation does not cover
    /// it). R4 is always `None` because every layer (full-attention and
    /// GDN alike) carries a dense MLP in Qwen3.5 hybrid.
    ///
    /// v1 contract: when `Some`, the indices must be strictly sorted
    /// ascending with no duplicates — this is the artifact's sole canonical
    /// representation, so no runtime consumer ever has to decide whether a
    /// repeated index means "apply twice" or "de-dupe". See
    /// [`Self::validate`].
    pub layer_scope: Option<Vec<usize>>,
}

/// Reject `(dim, block_size)` pairs [`super::hadamard::BlockHadamard::new`]
/// refuses to construct: `dim == 0`, `dim` exceeding
/// [`MAX_BLOCK_HADAMARD_LEN`], or a block count (`dim / block_size`)
/// exceeding [`MAX_BLOCK_HADAMARD_BLOCKS`]. This is the single shared
/// source of truth between the two gates that certify an R3/R4 recipe
/// against a config: the convenience constructors
/// ([`OnlineRotationSpec::r3_full_attention`]/[`OnlineRotationSpec::r4_dense_mlp`])
/// and [`OnlineRotationSpec::validate`]'s `cfg`-present branch (which also
/// certifies hand-built specs that bypass the constructors, e.g. specs
/// deserialized from an artifact). Without this shared check, a descriptor
/// naming a `(dim, block_size)` pair `BlockHadamard` refuses to construct
/// (e.g. Qwen3.6-27B's `intermediate_size = 17408` with `block_size = 4`,
/// giving 4,352 blocks over the block-count cap; or `dim = 33_554_432`
/// with `block_size = 8192`, giving exactly 4,096 blocks — under the
/// block-count cap, but `dim` itself is over [`MAX_BLOCK_HADAMARD_LEN`])
/// could pass both certification gates and only fail once a later PR
/// tries to materialize the rotation at runtime.
///
/// Caller must already know `block_size` evenly divides `dim` (this
/// function does not re-check divisibility) so `num_blocks = dim /
/// block_size` is exact.
fn check_block_hadamard_num_blocks_cap(
    axis_name: &str,
    dim: usize,
    block_size: usize,
) -> Result<(), InferenceError> {
    if dim == 0 {
        return Err(InferenceError::Inference(format!(
            "OnlineRotationSpec: {axis_name} must be non-zero — \
             BlockHadamard::new refuses a zero-length rotation axis"
        )));
    }
    if dim > MAX_BLOCK_HADAMARD_LEN {
        return Err(InferenceError::Inference(format!(
            "OnlineRotationSpec: {axis_name} {dim} exceeds the \
             MAX_BLOCK_HADAMARD_LEN cap of {MAX_BLOCK_HADAMARD_LEN} — this \
             recipe cannot be materialized by BlockHadamard::new, so it is \
             refused here rather than certified as a valid artifact"
        )));
    }
    let num_blocks = dim / block_size;
    if num_blocks > MAX_BLOCK_HADAMARD_BLOCKS {
        return Err(InferenceError::Inference(format!(
            "OnlineRotationSpec: {axis_name} {dim} with block_size \
             {block_size} needs {num_blocks} BlockHadamard blocks, exceeding \
             the MAX_BLOCK_HADAMARD_BLOCKS cap of \
             {MAX_BLOCK_HADAMARD_BLOCKS} — this recipe cannot be \
             materialized by BlockHadamard::new, so it is refused here \
             rather than certified as a valid artifact"
        )));
    }
    Ok(())
}

impl OnlineRotationSpec {
    /// Upper bound on `layer_scope` length. A real R3 recipe scopes to a
    /// config's full-attention layer count (tens, not thousands); this
    /// ceiling sits far above any legitimate spec while keeping
    /// [`Self::validate`]'s per-layer membership check bounded regardless
    /// of how large a caller-supplied `layer_scope` is. Mirrors
    /// [`super::io::OnlineArtifactDescriptor::MAX_VALIDATED_ENTRIES`].
    const MAX_LAYER_SCOPE_ENTRIES: usize = 4096;

    /// Build the R3 spec for Qwen3.5 hybrid: scope is exactly the
    /// config-declared full-attention layer indices (GDN excluded).
    ///
    /// `block_size` divides `num_attention_heads` — see the `block_size`
    /// field doc on [`OnlineRotationSpec`] for why the axis is heads, not
    /// `head_dim` (QuaRot Eq. 9's cross-head `H_num_heads ⊗ I_head_dim`
    /// online factor, confirmed against arXiv:2404.00456 Stage 1c: "insert
    /// a block into the forward pass that computes `Z ← Z(H_nh ⊗ I)` [...]
    /// denoted Hadamard heads").
    ///
    /// Refuses if `block_size` is zero, not a power of two, does not
    /// divide `cfg.num_attention_heads`, or if the config has zero
    /// full-attention layers (would silently produce a no-op rotation with
    /// an empty scope — that is a config/architecture mismatch, not a valid
    /// R3 artifact).
    pub fn r3_full_attention(
        cfg: &Qwen35Config,
        seed: u64,
        block_size: usize,
    ) -> Result<Self, InferenceError> {
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(InferenceError::Inference(format!(
                "OnlineRotationSpec::r3_full_attention requires a power-of-two \
                 block_size, got {block_size}"
            )));
        }
        if !cfg.num_attention_heads.is_multiple_of(block_size) {
            return Err(InferenceError::Inference(format!(
                "OnlineRotationSpec::r3_full_attention: block_size {block_size} \
                 does not divide num_attention_heads {} — R3's online factor \
                 is QuaRot Eq. 9's cross-head Hadamard (H_num_heads ⊗ \
                 I_head_dim), so block_size divides the head axis, not \
                 head_dim",
                cfg.num_attention_heads
            )));
        }
        check_block_hadamard_num_blocks_cap(
            "num_attention_heads",
            cfg.num_attention_heads,
            block_size,
        )?;
        let layers: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|&i| cfg.is_full_attention(i))
            .collect();
        if layers.is_empty() {
            return Err(InferenceError::Inference(
                "OnlineRotationSpec::r3_full_attention: config resolved zero \
                 full-attention layers — refusing an empty-scope R3 artifact"
                    .to_string(),
            ));
        }
        Ok(Self {
            id: RotationId::AttentionOutputR3,
            side: AbsorptionSide::InputSide,
            seed,
            block_size,
            layer_scope: Some(layers),
        })
    }

    /// Build the R4 spec for Qwen3.5 hybrid dense MLP: scope is every
    /// layer (`layer_scope: None`).
    ///
    /// Refuses if `block_size` is zero, not a power of two, does not
    /// divide `cfg.intermediate_size`, `cfg` is a MoE configuration (this
    /// constructor's `mlp.down_proj.weight` target is the DENSE tensor
    /// name — a MoE config's loader requires `mlp.experts.down_proj`
    /// instead, so building an R4 spec against a MoE config would certify a
    /// fabricated dense per-layer name; MoE R4 targets are not yet
    /// modeled), or `cfg.intermediate_size / block_size` exceeds
    /// `MAX_BLOCK_HADAMARD_BLOCKS` (a recipe [`super::hadamard::BlockHadamard::new`]
    /// cannot construct — see `check_block_hadamard_num_blocks_cap`).
    pub fn r4_dense_mlp(
        cfg: &Qwen35Config,
        seed: u64,
        block_size: usize,
    ) -> Result<Self, InferenceError> {
        if cfg.is_moe() {
            return Err(InferenceError::Inference(
                "OnlineRotationSpec::r4_dense_mlp: this config is a MoE \
                 configuration (loader requires mlp.experts.down_proj, not \
                 the dense mlp.down_proj.weight this constructor targets) — \
                 MoE R4 targets are not yet modeled"
                    .to_string(),
            ));
        }
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(InferenceError::Inference(format!(
                "OnlineRotationSpec::r4_dense_mlp requires a power-of-two \
                 block_size, got {block_size}"
            )));
        }
        if !cfg.intermediate_size.is_multiple_of(block_size) {
            return Err(InferenceError::Inference(format!(
                "OnlineRotationSpec::r4_dense_mlp: block_size {block_size} \
                 does not divide intermediate_size {}",
                cfg.intermediate_size
            )));
        }
        check_block_hadamard_num_blocks_cap(
            "intermediate_size",
            cfg.intermediate_size,
            block_size,
        )?;
        Ok(Self {
            id: RotationId::MlpDownR4,
            side: AbsorptionSide::InputSide,
            seed,
            block_size,
            layer_scope: None,
        })
    }

    /// Validate this spec's own internal invariants — independent of any
    /// artifact-level tensor-declaration coverage (see
    /// [`super::io::OnlineArtifactDescriptor::validate`] for that separate
    /// check). Every public field on `OnlineRotationSpec` is directly
    /// constructible: the convenience
    /// constructors [`Self::r3_full_attention`]/[`Self::r4_dense_mlp`]
    /// enforce these rules when building a spec, but nothing previously
    /// stopped a hand-built spec (e.g. deserialized from a corrupted index,
    /// or assembled by a future caller that skips the constructors) from
    /// violating them. `OnlineArtifactDescriptor::validate` now calls this
    /// on every spec it carries, so no descriptor with a malformed spec can
    /// pass validation regardless of how the spec was constructed.
    ///
    /// Checks, independent of `cfg`:
    /// - `id` must be [`RotationId::AttentionOutputR3`] or
    ///   [`RotationId::MlpDownR4`] — [`RotationId::ResidualStream`] is a
    ///   purely offline rotation (see [`RotationId::online_transform_site`]'s
    ///   doc) and must never appear as an *online* spec.
    /// - `side` must be [`AbsorptionSide::InputSide`] for both R3 and R4 —
    ///   the only orientation proven correct by the one-layer reference
    ///   test (see the `side` field doc).
    /// - `block_size` must be nonzero and a power of two.
    /// - R3 requires `layer_scope` to be `Some` and non-empty (an R3 recipe
    ///   with no full-attention layers is a contract violation, not a valid
    ///   degenerate case).
    /// - R3's `layer_scope` must be strictly sorted ascending with no
    ///   duplicate indices (canonical-form contract, cfg-independent) — a
    ///   scope like `[3,3,7,11,15,19,23]` is rejected even though a
    ///   set-equality check alone would silently accept it after collecting
    ///   into a set.
    /// - R4 requires `layer_scope` to be exactly `None` (every layer,
    ///   full-attention and GDN alike, carries a dense MLP — see the
    ///   `layer_scope` field doc). A hand-built R4 spec that restricts
    ///   `layer_scope` to specific layers is `None`-wildcard semantics
    ///   violated in the *other* direction and must also be rejected.
    ///
    /// Additional checks when `cfg` is supplied:
    /// - R3's `block_size` must divide `cfg.num_attention_heads`, and
    ///   `layer_scope` must be set-equal to `cfg`'s full-attention layers —
    ///   every scoped index must be a valid full-attention layer
    ///   (`cfg.is_full_attention`), AND every one of `cfg`'s full-attention
    ///   layers must appear in `layer_scope` (a scope naming a strict
    ///   subset, e.g. `[3]` when the
    ///   config's full-attention layers are `[3,7,11,15,19,23]`, previously
    ///   passed membership-only checking).
    /// - R4's `block_size` must divide `cfg.intermediate_size`.
    ///
    /// `cfg: None` is accepted because `OnlineArtifactDescriptor::validate`
    /// does not always have a config available (issue #703 PR1's v0 scope
    /// is contract-only) — the cfg-independent checks above are exactly the
    /// ones that catch the review's reported malformed spec (`OutputSide`,
    /// `block_size=3`, `layer_scope=None` for an R3 id) without needing one.
    pub fn validate(&self, cfg: Option<&Qwen35Config>) -> Result<(), InferenceError> {
        if self.side != AbsorptionSide::InputSide {
            return Err(InferenceError::Inference(format!(
                "OnlineRotationSpec::validate: {:?} requires side=InputSide \
                 (the only orientation proven correct by the R3/R4 reference \
                 test), got {:?}",
                self.id, self.side
            )));
        }
        if self.block_size == 0 || !self.block_size.is_power_of_two() {
            return Err(InferenceError::Inference(format!(
                "OnlineRotationSpec::validate: {:?} requires a power-of-two \
                 block_size, got {}",
                self.id, self.block_size
            )));
        }
        match self.id {
            RotationId::ResidualStream => {
                return Err(InferenceError::Inference(
                    "OnlineRotationSpec::validate: RotationId::ResidualStream \
                     is a purely offline rotation and must never appear as an \
                     online OnlineRotationSpec"
                        .to_string(),
                ));
            }
            RotationId::AttentionOutputR3 => {
                let layers = self.layer_scope.as_ref().ok_or_else(|| {
                    InferenceError::Inference(
                        "OnlineRotationSpec::validate: AttentionOutputR3 (R3) \
                         requires an explicit non-empty layer_scope (full-\
                         attention layers only) — layer_scope=None is invalid \
                         for R3"
                            .to_string(),
                    )
                })?;
                if layers.is_empty() {
                    return Err(InferenceError::Inference(
                        "OnlineRotationSpec::validate: AttentionOutputR3 (R3) \
                         layer_scope must not be empty"
                            .to_string(),
                    ));
                }
                if layers.len() > Self::MAX_LAYER_SCOPE_ENTRIES {
                    return Err(InferenceError::Inference(format!(
                        "OnlineRotationSpec::validate: R3 layer_scope declares {} \
                         entries, exceeding the maximum of {} — a spec this large \
                         is rejected before the per-layer membership check to keep \
                         validation cost bounded regardless of input",
                        layers.len(),
                        Self::MAX_LAYER_SCOPE_ENTRIES
                    )));
                }
                // Canonical-form contract (v1, cfg-independent): layer_scope
                // must be strictly sorted ascending with no duplicates. A
                // duplicate index (e.g. `[3,3,7,...]`) would silently
                // collapse to a smaller set under any downstream
                // set-equality check, and an unsorted or repeated scope
                // gives the future runtime no canonical answer for whether
                // to de-duplicate, apply twice, or pick one — so both are
                // rejected here rather than left ambiguous.
                for pair in layers.windows(2) {
                    if pair[0] >= pair[1] {
                        return Err(InferenceError::Inference(format!(
                            "OnlineRotationSpec::validate: R3 layer_scope {layers:?} \
                             must be strictly sorted ascending with no \
                             duplicates — found {} at or after {}",
                            pair[1], pair[0]
                        )));
                    }
                }
                if let Some(cfg) = cfg {
                    if !cfg.num_attention_heads.is_multiple_of(self.block_size) {
                        return Err(InferenceError::Inference(format!(
                            "OnlineRotationSpec::validate: R3 block_size {} \
                             does not divide cfg.num_attention_heads {}",
                            self.block_size, cfg.num_attention_heads
                        )));
                    }
                    check_block_hadamard_num_blocks_cap(
                        "num_attention_heads",
                        cfg.num_attention_heads,
                        self.block_size,
                    )?;
                    for &idx in layers {
                        if idx >= cfg.num_hidden_layers || !cfg.is_full_attention(idx) {
                            return Err(InferenceError::Inference(format!(
                                "OnlineRotationSpec::validate: R3 layer_scope \
                                 includes layer {idx}, which is not a valid \
                                 full-attention layer for this config"
                            )));
                        }
                    }
                    // Membership alone is not enough: R3's contract (and the
                    // r3_full_attention constructor) requires ALL of the
                    // config's full-attention layers, not a strict subset —
                    // a scope naming only some of them would leave the
                    // omitted layers' weights un-counter-rotated at runtime.
                    // Require set equality between `layers` and the config's
                    // full-attention layers; the membership loop above
                    // already rejects the "extra/non-member" direction, so
                    // this only needs to check the "missing" direction.
                    // `layers` is bounded by MAX_LAYER_SCOPE_ENTRIES above, so
                    // build a set once and probe it — O(n+m) instead of the
                    // O(n*m) a `Vec::contains` scan per config layer would
                    // cost.
                    let scoped: HashSet<usize> = layers.iter().copied().collect();
                    let required_full_attention_layers: Vec<usize> = (0..cfg.num_hidden_layers)
                        .filter(|&idx| cfg.is_full_attention(idx))
                        .collect();
                    let missing_layers: Vec<usize> = required_full_attention_layers
                        .iter()
                        .copied()
                        .filter(|idx| !scoped.contains(idx))
                        .collect();
                    if !missing_layers.is_empty() {
                        return Err(InferenceError::Inference(format!(
                            "OnlineRotationSpec::validate: R3 layer_scope {layers:?} \
                             does not cover all of this config's full-attention \
                             layers {required_full_attention_layers:?} — missing \
                             layer(s) {missing_layers:?}"
                        )));
                    }
                }
            }
            RotationId::MlpDownR4 => {
                if self.layer_scope.is_some() {
                    return Err(InferenceError::Inference(
                        "OnlineRotationSpec::validate: MlpDownR4 (R4) \
                         layer_scope must be None — every layer carries a \
                         dense MLP, so R4 is never restricted to a subset of \
                         layers"
                            .to_string(),
                    ));
                }
                if let Some(cfg) = cfg {
                    // R4's contract hard-codes the DENSE
                    // `mlp.down_proj.weight` target (see
                    // `OnlineTransformSite::weight_tensor_suffix`), but a MoE
                    // config's loader requires `mlp.experts.down_proj`
                    // instead (`is_moe()` — `num_experts`/
                    // `num_experts_per_tok`/`moe_intermediate_size`/
                    // `shared_expert_intermediate_size` set — see
                    // `Qwen35Config::is_moe`). `Qwen35Config::qwen36_35b_a3b()`
                    // is MoE, so without this check `r4_dense_mlp(&cfg, ...)`
                    // and `validate(Some(&cfg))` would both succeed while
                    // certifying a fabricated dense per-layer name that
                    // never appears in a real MoE checkpoint. MoE R4 targets
                    // are not yet modeled by this plan; refuse rather than
                    // validate against the wrong tensor.
                    if cfg.is_moe() {
                        return Err(InferenceError::Inference(
                            "OnlineRotationSpec::validate: MlpDownR4 (R4) \
                             targets the dense mlp.down_proj.weight tensor, \
                             but this config is a MoE configuration (loader \
                             requires mlp.experts.down_proj instead) — MoE \
                             R4 targets are not yet modeled by this plan"
                                .to_string(),
                        ));
                    }
                    if !cfg.intermediate_size.is_multiple_of(self.block_size) {
                        return Err(InferenceError::Inference(format!(
                            "OnlineRotationSpec::validate: R4 block_size {} \
                             does not divide cfg.intermediate_size {}",
                            self.block_size, cfg.intermediate_size
                        )));
                    }
                    // A structurally valid, divisibility-passing recipe can
                    // still name a block count BlockHadamard::new refuses to
                    // construct (e.g. Qwen3.6-27B's intermediate_size=17408
                    // with block_size=4 gives 4,352 blocks, over the
                    // MAX_BLOCK_HADAMARD_BLOCKS cap of 4,096) — this gate
                    // must independently reject that case for hand-built
                    // specs that bypass `r4_dense_mlp`, not just rely on the
                    // constructor's own check.
                    check_block_hadamard_num_blocks_cap(
                        "intermediate_size",
                        cfg.intermediate_size,
                        self.block_size,
                    )?;
                }
            }
        }
        Ok(())
    }
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
    /// against `qwen_required_tensor_names` (exposed in step 3b via
    /// [`crate::model::qwen35::qwen_required_tensor_names`]), and
    /// LoRA-runtime incompatibility marker. Step 3c/3d's `quantize_quarot`
    /// binary owns all of these.
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

    /// Plan rules for **the linear-layer subset** of plain Qwen3 residual-stream
    /// rotation. Covers GQA attention + dense MLP + embed/lm_head.
    ///
    /// Plain Qwen3 has no GDN (GatedDeltaNet) linear-attention layers — use
    /// [`Self::qwen35_residual_stream_linear_layers`] for Qwen3.5 hybrid.
    ///
    /// Key shape difference vs Qwen3.5: `self_attn.q_proj.weight` is
    /// `[q_dim, hidden]` (plain Qwen3) not the fused `[2*q_dim, hidden]`
    /// (Qwen3.5). The absorption rule is identical (input-side) — the plan
    /// only needs the suffix to match correctly.
    ///
    /// Use [`qwen3_required_tensor_names`] to cross-check that every required
    /// tensor is present before quantization begins (step 3c of ADR-044).
    pub fn qwen3_residual_stream_linear_layers() -> Self {
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
                // GQA attention layers — no GDN in plain Qwen3
                req("self_attn.q_proj.weight", r_in),
                req("self_attn.k_proj.weight", r_in),
                req("self_attn.v_proj.weight", r_in),
                req("self_attn.o_proj.weight", r_out),
                // Dense MLP layers
                req("mlp.gate_proj.weight", r_in),
                req("mlp.up_proj.weight", r_in),
                req("mlp.down_proj.weight", r_out),
                // Embedding — always required
                req("embed_tokens.weight", r_in),
                // lm_head — optional in INPUT SafeTensors; materialised by
                // step 3c when tie_word_embeddings=true
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

    /// Look up the absorption side for a LoRA module name (e.g., `"q_proj"`).
    ///
    /// Returns `Some(InputSide)` or `Some(OutputSide)` if any plan rule's
    /// pattern ends with `"{module}.weight"`, or `None` if the module is not
    /// in the plan (e.g., not a residual-stream projection).
    ///
    /// Used by the adapter rotation logic (ADR-045) to decide:
    /// - `InputSide` → counter-rotate A: `A_cr = A · R^T`
    /// - `OutputSide` → rotate B: `B_rot = R · B`
    pub fn absorption_for_module(&self, module: &str) -> Option<AbsorptionSide> {
        let suffix = format!("{module}.weight");
        self.rules
            .iter()
            .find(|rule| rule.pattern.ends_with(&suffix))
            .map(|rule| rule.rotation.side)
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
    /// name list at [`crate::model::qwen35::qwen_required_tensor_names`].
    /// Step 3b promoted that re-export out of `#[cfg(test)]` so the
    /// conversion binary can call it; step 3c performs the actual
    /// per-layer cross-check.
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
        RotationId::AttentionOutputR3 | RotationId::MlpDownR4 => {
            return Err(InferenceError::Inference(
                "apply_tensor_rotation: R3/R4 rotation IDs are contract-only in \
                 this PR (issue #703 PR1) — no plan should reference them yet, \
                 and no online rotation is wired into offline absorption"
                    .to_string(),
            ));
        }
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
        RotationId::AttentionOutputR3 | RotationId::MlpDownR4 => {
            return Err(InferenceError::Inference(
                "apply_tensor_rotation_f64: R3/R4 rotation IDs are contract-only \
                 in this PR (issue #703 PR1) — no plan should reference them \
                 yet, and no online rotation is wired into offline absorption"
                    .to_string(),
            ));
        }
    };
    match tr.side {
        AbsorptionSide::InputSide => absorb_input_rotation_f64(weight, rows, cols, rotation)?,
        AbsorptionSide::OutputSide => absorb_output_rotation_f64(weight, rows, cols, rotation)?,
    }
    Ok(true)
}

/// Required tensor names for a plain Qwen3 model with the given `cfg`.
///
/// Analogous to [`crate::model::qwen35::qwen_required_tensor_names`] for
/// Qwen3.5. Plain Qwen3 uses `layers.{i}.*` names directly — no
/// `model.language_model.*` prefix — and has no GDN `linear_attn.*` tensors.
///
/// Step 3c of ADR-044 calls this to cross-check that every required tensor
/// is reachable in the SafeTensors file before quantization begins.
pub fn qwen3_required_tensor_names(cfg: &QwenConfig) -> Vec<String> {
    let mut names: Vec<String> = vec!["embed_tokens.weight".to_string(), "norm.weight".to_string()];
    for i in 0..cfg.num_hidden_layers {
        let p = format!("layers.{i}");
        names.extend([
            format!("{p}.self_attn.q_proj.weight"),
            format!("{p}.self_attn.k_proj.weight"),
            format!("{p}.self_attn.v_proj.weight"),
            format!("{p}.self_attn.o_proj.weight"),
            format!("{p}.self_attn.q_norm.weight"),
            format!("{p}.self_attn.k_norm.weight"),
            format!("{p}.input_layernorm.weight"),
            format!("{p}.mlp.gate_proj.weight"),
            format!("{p}.mlp.up_proj.weight"),
            format!("{p}.mlp.down_proj.weight"),
            format!("{p}.post_attention_layernorm.weight"),
        ]);
    }
    names
}

#[cfg(test)]
mod tests;
