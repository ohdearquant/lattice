//! Materialize `lm_head.weight` for tied-embedding Qwen3.5 configs, construct
//! the `final_norm → lm_head` fusion target so the pipeline's existing
//! [`crate::quant::quarot::pipeline::fuse_rmsnorms`] can fold `(1 + g_final)`
//! into the materialized matrix, and flip `tie_word_embeddings` to `false`
//! in the output config so the runtime loader actually consults the new
//! `lm_head.weight`.
//!
//! Step 3c-3 of ADR-044 (see `docs/adr/ADR-044-quarot-rotated-quantization.md`).
//!
//! ## Why every step here is required
//!
//! Qwen3.5 ships with `tie_word_embeddings=true` by default. At runtime the
//! loader leaves `lm_head: None` and the forward path falls back to
//! `embed_tokens` via `logits_weight()` (`model/qwen35/weights.rs`). After
//! QuaRot, this tied fallback is **incorrect** because:
//!
//! - `embed_tokens` and `lm_head` need DIFFERENT transformations under
//!   QuaRot. `embed_tokens` only absorbs the residual-stream rotation
//!   `R` on its input side (so the embedding lookup outputs are in the
//!   rotated basis). `lm_head` must additionally absorb the shifted
//!   final-RMSNorm scale `(1 + g_final)` BEFORE absorbing `R`, because
//!   `D = diag(1 + g_final)` does not commute with the Hadamard rotation
//!   (see `quant/quarot/plan.rs` §"Tied embeddings: only one correct
//!   path"). Sharing one matrix between the embedding lookup and the
//!   output projection breaks one of the two transforms.
//! - The output `.q4` file therefore stores `lm_head.weight` separately
//!   with both transforms applied, and the runtime config must say
//!   `tie_word_embeddings=false` so the loader actually loads it instead
//!   of routing through `embed_tokens`.
//!
//! ## Where this fits in the pipeline
//!
//! ```text
//!   1. read f64 tensors        ← pipeline::load_tensors_f64
//!   2. materialize lm_head     ← THIS MODULE (only when cfg.tie_word_embeddings)
//!   3. fuse_rmsnorms with both the per-layer plan AND
//!      `qwen35_final_norm_fusion_target` appended      ← pipeline::fuse_rmsnorms
//!   4. absorb_rotations        ← pipeline::absorb_rotations (covers lm_head
//!                                via the existing `opt("lm_head.weight", r_in)`
//!                                rule in `RotationPlan`)
//!   5. quantize (caller)       ← weights::q4_weights
//!   6. flip output config      ← `untie_word_embeddings_in_config_json` on the
//!                                output config.json (the in-memory
//!                                `untie_word_embeddings_in_cfg` alone is NOT
//!                                sufficient — the HF config has two tie
//!                                flags and the parser gives the top-level
//!                                value precedence)
//! ```
//!
//! Step 2 must happen BEFORE step 3 (the fusion target references
//! `lm_head.weight`), and step 6's config flip must happen before the
//! converter writes the output `config.json` — otherwise the runtime
//! falls back to `embed_tokens` and silently produces wrong logits.

use std::collections::HashMap;

use crate::error::InferenceError;
use crate::model::qwen35_config::Qwen35Config;
use crate::quant::quarot::pipeline::TensorEntry;
use crate::quant::quarot::rmsnorm_fusion::RmsNormFusionTarget;

/// SafeTensors name of Qwen3.5's input embedding matrix.
pub const QWEN35_EMBED_TOKENS_NAME: &str = "model.language_model.embed_tokens.weight";

/// SafeTensors name the runtime loader expects for the output projection
/// matrix when `tie_word_embeddings=false`.
pub const QWEN35_LM_HEAD_NAME: &str = "lm_head.weight";

/// SafeTensors name of Qwen3.5's final pre-`lm_head` RMSNorm tensor.
pub const QWEN35_FINAL_NORM_NAME: &str = "model.language_model.norm.weight";

/// If `cfg.tie_word_embeddings`, clone `embed_tokens.weight` into a new
/// `lm_head.weight` entry in the working set so the rotation/fusion
/// pipeline can transform it independently of the embedding lookup.
///
/// Untied configs (`tie_word_embeddings=false`) require `lm_head.weight`
/// to already be present in the working set, with shape
/// `[cfg.vocab_size, cfg.hidden_size]`. This is the converter's boundary
/// shape check — `qwen_required_tensor_names` only checks presence,
/// and `pipeline::fuse_rmsnorms` only checks the column count against
/// `final_norm`, so a wrong row count would otherwise slip past.
///
/// MUST be called BEFORE [`crate::quant::quarot::pipeline::fuse_rmsnorms`]
/// runs on the fusion plan that includes the final-norm target, and
/// before [`crate::quant::quarot::pipeline::absorb_rotations`].
///
/// # Errors
///
/// - Tied config and `embed_tokens.weight` is missing from the working set.
/// - Tied config and `lm_head.weight` is already in the working set (caller
///   has somehow loaded both, which is inconsistent with the tied flag).
/// - Untied config and `lm_head.weight` is missing from the working set
///   (the loader did not request it, or the source SafeTensors is malformed).
/// - `embed_tokens.weight` (tied) or `lm_head.weight` (untied) shape is not
///   `[cfg.vocab_size, cfg.hidden_size]`, or its `data.len()` disagrees
///   with the shape product.
pub fn materialize_lm_head_for_qwen35(
    tensors: &mut HashMap<String, TensorEntry>,
    cfg: &Qwen35Config,
) -> Result<(), InferenceError> {
    let expected_shape = vec![cfg.vocab_size, cfg.hidden_size];
    let expected_len = cfg.vocab_size.checked_mul(cfg.hidden_size).ok_or_else(|| {
        InferenceError::Inference(format!(
            "materialize_lm_head_for_qwen35: vocab_size*hidden_size overflow \
             (vocab_size={}, hidden_size={})",
            cfg.vocab_size, cfg.hidden_size
        ))
    })?;

    if cfg.tie_word_embeddings {
        if tensors.contains_key(QWEN35_LM_HEAD_NAME) {
            return Err(InferenceError::Inference(format!(
                "materialize_lm_head_for_qwen35: `{QWEN35_LM_HEAD_NAME}` already in working set \
                 but config says tie_word_embeddings=true; refusing to overwrite. Caller bug."
            )));
        }
        let embed = tensors.get(QWEN35_EMBED_TOKENS_NAME).ok_or_else(|| {
            InferenceError::Inference(format!(
                "materialize_lm_head_for_qwen35: tied config requires `{QWEN35_EMBED_TOKENS_NAME}` \
                 in the working set to clone into `{QWEN35_LM_HEAD_NAME}`"
            ))
        })?;
        if embed.shape != expected_shape {
            return Err(InferenceError::Inference(format!(
                "materialize_lm_head_for_qwen35: `{QWEN35_EMBED_TOKENS_NAME}` shape {:?} \
                 != expected [vocab_size={}, hidden_size={}]",
                embed.shape, cfg.vocab_size, cfg.hidden_size
            )));
        }
        if embed.data.len() != expected_len {
            return Err(InferenceError::Inference(format!(
                "materialize_lm_head_for_qwen35: `{QWEN35_EMBED_TOKENS_NAME}` data.len()={} \
                 != vocab_size*hidden_size {expected_len}",
                embed.data.len()
            )));
        }
        let materialized = TensorEntry {
            name: QWEN35_LM_HEAD_NAME.to_string(),
            shape: embed.shape.clone(),
            data: embed.data.clone(),
        };
        tensors.insert(QWEN35_LM_HEAD_NAME.to_string(), materialized);
        Ok(())
    } else {
        let lm = tensors.get(QWEN35_LM_HEAD_NAME).ok_or_else(|| {
            InferenceError::Inference(format!(
                "materialize_lm_head_for_qwen35: untied config requires `{QWEN35_LM_HEAD_NAME}` \
                 to be already present in the working set (loaded from SafeTensors); not found"
            ))
        })?;
        if lm.shape != expected_shape {
            return Err(InferenceError::Inference(format!(
                "materialize_lm_head_for_qwen35: `{QWEN35_LM_HEAD_NAME}` shape {:?} \
                 != expected [vocab_size={}, hidden_size={}]",
                lm.shape, cfg.vocab_size, cfg.hidden_size
            )));
        }
        if lm.data.len() != expected_len {
            return Err(InferenceError::Inference(format!(
                "materialize_lm_head_for_qwen35: `{QWEN35_LM_HEAD_NAME}` data.len()={} \
                 != vocab_size*hidden_size {expected_len}",
                lm.data.len()
            )));
        }
        Ok(())
    }
}

/// Return the [`RmsNormFusionTarget`] that folds `(1 + g_final)` into
/// the materialized `lm_head.weight` as a column multiply.
///
/// Callers append this to the per-layer fusion plan from
/// [`crate::quant::quarot::rmsnorm_fusion::qwen35_per_layer_fusion_plan`]
/// AFTER [`materialize_lm_head_for_qwen35`] has populated the working set.
pub fn qwen35_final_norm_fusion_target() -> RmsNormFusionTarget {
    RmsNormFusionTarget {
        norm_tensor: QWEN35_FINAL_NORM_NAME.to_string(),
        downstream_weights: vec![QWEN35_LM_HEAD_NAME.to_string()],
    }
}

/// Set `cfg.tie_word_embeddings = false`. Idempotent; safe to call on
/// already-untied configs.
///
/// In-memory only. **The converter binary must ALSO mutate the output
/// `config.json` on disk** via [`untie_word_embeddings_in_config_json`] —
/// the on-disk HF config has two tie flags (nested + top-level), and the
/// runtime loader reloads from JSON, so an in-memory flip alone leaves
/// the output incoherent.
pub fn untie_word_embeddings_in_cfg(cfg: &mut Qwen35Config) {
    cfg.tie_word_embeddings = false;
}

/// Mutate a raw HF `config.json` string so that on reload by
/// [`crate::model::qwen35_config::Qwen35Config::from_config_json_str`] the
/// parsed `cfg.tie_word_embeddings` evaluates to `false`. Returns the new
/// JSON.
///
/// HF Qwen3.5 `config.json` carries `tie_word_embeddings` in two
/// locations: nested under `text_config` and at the top level. The
/// lattice parser gives the **top-level** value precedence (see
/// `Qwen35Config::from_config_json_str`), so flipping only the nested
/// field would still parse back to `tie_word_embeddings=true` and bypass
/// the materialized `lm_head.weight` at runtime — producing silently
/// wrong logits. This helper:
///
/// - Sets the top-level `tie_word_embeddings` to `false` (insert if absent).
/// - If `text_config.tie_word_embeddings` is present, updates it to `false`.
///   (Absent → leaves it absent; the parser falls back to top-level.)
///
/// MUST be called on the output config before the converter writes it
/// to disk; the in-memory [`untie_word_embeddings_in_cfg`] is not
/// sufficient because the runtime reloads from JSON.
///
/// # Errors
///
/// Returns `InferenceError::Inference` if the JSON is malformed or the
/// top-level value is not a JSON object.
pub fn untie_word_embeddings_in_config_json(json: &str) -> Result<String, InferenceError> {
    let mut value: serde_json::Value = serde_json::from_str(json).map_err(|e| {
        InferenceError::Inference(format!(
            "untie_word_embeddings_in_config_json: invalid JSON: {e}"
        ))
    })?;
    let obj = value.as_object_mut().ok_or_else(|| {
        InferenceError::Inference(
            "untie_word_embeddings_in_config_json: top-level JSON must be an object".to_string(),
        )
    })?;
    if let Some(text_config) = obj.get_mut("text_config")
        && let Some(text_obj) = text_config.as_object_mut()
        && text_obj.contains_key("tie_word_embeddings")
    {
        text_obj.insert(
            "tie_word_embeddings".to_string(),
            serde_json::Value::Bool(false),
        );
    }
    obj.insert(
        "tie_word_embeddings".to_string(),
        serde_json::Value::Bool(false),
    );
    serde_json::to_string_pretty(&value).map_err(|e| {
        InferenceError::Inference(format!(
            "untie_word_embeddings_in_config_json: serialize failed: {e}"
        ))
    })
}

#[cfg(test)]
mod tests;
