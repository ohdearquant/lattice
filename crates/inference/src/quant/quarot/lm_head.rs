//! Materialize `lm_head.weight` for tied-embedding Qwen3.5 configs, construct
//! the `final_norm → lm_head` fusion target so the pipeline's existing
//! [`fuse_rmsnorms`] can fold `(1 + g_final)` into the materialized matrix,
//! and flip `tie_word_embeddings` to `false` in the output config so the
//! runtime loader actually consults the new `lm_head.weight`.
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
//!   6. flip output config      ← `untie_word_embeddings_in_cfg` BEFORE serialization
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
/// to already be present in the working set — the function validates this
/// and returns an error otherwise.
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
/// - `embed_tokens.weight` shape is not `[cfg.vocab_size, cfg.hidden_size]`.
pub fn materialize_lm_head_for_qwen35(
    tensors: &mut HashMap<String, TensorEntry>,
    cfg: &Qwen35Config,
) -> Result<(), InferenceError> {
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
        let expected_shape = vec![cfg.vocab_size, cfg.hidden_size];
        if embed.shape != expected_shape {
            return Err(InferenceError::Inference(format!(
                "materialize_lm_head_for_qwen35: `{QWEN35_EMBED_TOKENS_NAME}` shape {:?} \
                 != expected [vocab_size={}, hidden_size={}]",
                embed.shape, cfg.vocab_size, cfg.hidden_size
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
        if !tensors.contains_key(QWEN35_LM_HEAD_NAME) {
            return Err(InferenceError::Inference(format!(
                "materialize_lm_head_for_qwen35: untied config requires `{QWEN35_LM_HEAD_NAME}` \
                 to be already present in the working set (loaded from SafeTensors); not found"
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
/// MUST run before the converter serializes the output `config.json`,
/// otherwise the runtime loader falls back to `embed_tokens` via
/// `logits_weight()` and ignores the materialized `lm_head.weight` —
/// producing silently wrong logits since the two matrices now carry
/// different QuaRot transforms.
pub fn untie_word_embeddings_in_cfg(cfg: &mut Qwen35Config) {
    cfg.tie_word_embeddings = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert_tensor(
        tensors: &mut HashMap<String, TensorEntry>,
        name: &str,
        shape: Vec<usize>,
        data: Vec<f64>,
    ) {
        tensors.insert(
            name.to_string(),
            TensorEntry {
                name: name.to_string(),
                shape,
                data,
            },
        );
    }

    /// Test vocab — small enough to keep `[vocab_size, hidden_size]` tensors
    /// tractable per test (the qwen35_0_8b preset's 248_320×1024 ≈ 2 GB
    /// would be wasteful here).
    const TEST_VOCAB: usize = 64;

    fn tied_qwen35_test_cfg() -> Qwen35Config {
        let mut cfg = Qwen35Config::qwen35_0_8b();
        assert!(cfg.tie_word_embeddings, "qwen35_0_8b preset must be tied");
        cfg.vocab_size = TEST_VOCAB;
        cfg
    }

    fn untied_qwen35_test_cfg() -> Qwen35Config {
        let mut cfg = tied_qwen35_test_cfg();
        cfg.tie_word_embeddings = false;
        cfg
    }

    fn synthetic_f64(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (state >> 11) as u32;
                (bits as f64 / u32::MAX as f64) - 0.5
            })
            .collect()
    }

    #[test]
    fn tied_config_materializes_lm_head_as_clone_of_embed_tokens() {
        let cfg = tied_qwen35_test_cfg();
        let mut tensors = HashMap::new();
        let embed_data = synthetic_f64(cfg.vocab_size * cfg.hidden_size, 1);
        insert_tensor(
            &mut tensors,
            QWEN35_EMBED_TOKENS_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            embed_data.clone(),
        );
        assert!(!tensors.contains_key(QWEN35_LM_HEAD_NAME));

        materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap();

        let lm = tensors
            .get(QWEN35_LM_HEAD_NAME)
            .expect("lm_head materialized");
        assert_eq!(lm.shape, vec![cfg.vocab_size, cfg.hidden_size]);
        assert_eq!(lm.data, embed_data);
        assert_eq!(lm.name, QWEN35_LM_HEAD_NAME);
        // embed_tokens unchanged.
        assert_eq!(tensors[QWEN35_EMBED_TOKENS_NAME].data, embed_data);
    }

    #[test]
    fn untied_config_with_lm_head_present_is_noop() {
        let cfg = untied_qwen35_test_cfg();
        let mut tensors = HashMap::new();
        let embed_data = synthetic_f64(cfg.vocab_size * cfg.hidden_size, 2);
        let lm_data = synthetic_f64(cfg.vocab_size * cfg.hidden_size, 3);
        insert_tensor(
            &mut tensors,
            QWEN35_EMBED_TOKENS_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            embed_data.clone(),
        );
        insert_tensor(
            &mut tensors,
            QWEN35_LM_HEAD_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            lm_data.clone(),
        );

        materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap();

        // No-op: lm_head data preserved (NOT overwritten with embed_tokens).
        assert_eq!(tensors[QWEN35_LM_HEAD_NAME].data, lm_data);
        assert_eq!(tensors[QWEN35_EMBED_TOKENS_NAME].data, embed_data);
    }

    #[test]
    fn tied_config_with_missing_embed_tokens_errors() {
        let cfg = tied_qwen35_test_cfg();
        let mut tensors = HashMap::new();
        let err = materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(QWEN35_EMBED_TOKENS_NAME),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("tied config"), "unexpected error: {msg}");
    }

    #[test]
    fn tied_config_with_lm_head_already_present_errors() {
        // Defensive check: tied flag + pre-existing lm_head is inconsistent.
        let cfg = tied_qwen35_test_cfg();
        let mut tensors = HashMap::new();
        insert_tensor(
            &mut tensors,
            QWEN35_EMBED_TOKENS_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            vec![0.0; cfg.vocab_size * cfg.hidden_size],
        );
        insert_tensor(
            &mut tensors,
            QWEN35_LM_HEAD_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            vec![0.0; cfg.vocab_size * cfg.hidden_size],
        );

        let err = materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("already in working set"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn untied_config_with_missing_lm_head_errors() {
        let cfg = untied_qwen35_test_cfg();
        let mut tensors = HashMap::new();
        insert_tensor(
            &mut tensors,
            QWEN35_EMBED_TOKENS_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            vec![0.0; cfg.vocab_size * cfg.hidden_size],
        );
        let err = materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("untied config"), "unexpected error: {msg}");
        assert!(msg.contains(QWEN35_LM_HEAD_NAME), "unexpected error: {msg}");
    }

    #[test]
    fn tied_config_with_embed_tokens_shape_mismatch_errors() {
        let cfg = tied_qwen35_test_cfg();
        let mut tensors = HashMap::new();
        // Wrong vocab_size dimension.
        insert_tensor(
            &mut tensors,
            QWEN35_EMBED_TOKENS_NAME,
            vec![cfg.vocab_size + 1, cfg.hidden_size],
            vec![0.0; (cfg.vocab_size + 1) * cfg.hidden_size],
        );
        let err = materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("shape"), "unexpected error: {msg}");
        assert!(
            msg.contains(QWEN35_EMBED_TOKENS_NAME),
            "unexpected error: {msg}"
        );
        // Did NOT insert lm_head on failure.
        assert!(!tensors.contains_key(QWEN35_LM_HEAD_NAME));
    }

    #[test]
    fn final_norm_fusion_target_uses_canonical_names() {
        let tgt = qwen35_final_norm_fusion_target();
        assert_eq!(tgt.norm_tensor, QWEN35_FINAL_NORM_NAME);
        assert_eq!(
            tgt.downstream_weights,
            vec![QWEN35_LM_HEAD_NAME.to_string()]
        );
    }

    #[test]
    fn final_norm_fusion_target_norm_name_matches_loader() {
        // Round-trip: the final_norm name we emit must match what the loader
        // actually requests for this config.
        let cfg = tied_qwen35_test_cfg();
        let required = crate::model::qwen35::qwen_required_tensor_names(&cfg);
        let tgt = qwen35_final_norm_fusion_target();
        assert!(
            required.contains(&tgt.norm_tensor),
            "final_norm tensor `{}` not in qwen_required_tensor_names",
            tgt.norm_tensor
        );
    }

    #[test]
    fn untie_word_embeddings_flips_true_to_false() {
        let mut cfg = tied_qwen35_test_cfg();
        assert!(cfg.tie_word_embeddings);
        untie_word_embeddings_in_cfg(&mut cfg);
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn untie_word_embeddings_is_idempotent_on_untied() {
        let mut cfg = untied_qwen35_test_cfg();
        assert!(!cfg.tie_word_embeddings);
        untie_word_embeddings_in_cfg(&mut cfg);
        assert!(!cfg.tie_word_embeddings);
    }

    /// End-to-end: after materialization + final-norm fusion + rotation
    /// absorption, the materialized lm_head must carry BOTH transforms
    /// while embed_tokens carries only the rotation. The two matrices must
    /// not be byte-identical (they were before fusion + absorption ran).
    #[test]
    fn materialized_lm_head_diverges_from_embed_tokens_after_pipeline() {
        use crate::quant::quarot::hadamard::RandomizedHadamard;
        use crate::quant::quarot::pipeline::{absorb_rotations, fuse_rmsnorms};
        use crate::quant::quarot::plan::RotationPlan;

        let cfg = tied_qwen35_test_cfg();
        let vocab = cfg.vocab_size;
        let hidden = cfg.hidden_size;
        let mut tensors = HashMap::new();
        insert_tensor(
            &mut tensors,
            QWEN35_EMBED_TOKENS_NAME,
            vec![vocab, hidden],
            synthetic_f64(vocab * hidden, 7),
        );
        insert_tensor(
            &mut tensors,
            QWEN35_FINAL_NORM_NAME,
            vec![hidden],
            synthetic_f64(hidden, 8),
        );

        materialize_lm_head_for_qwen35(&mut tensors, &cfg).unwrap();
        // Pre-pipeline: clone of embed_tokens.
        assert_eq!(
            tensors[QWEN35_LM_HEAD_NAME].data,
            tensors[QWEN35_EMBED_TOKENS_NAME].data
        );

        let final_norm_target = qwen35_final_norm_fusion_target();
        fuse_rmsnorms(&mut tensors, std::slice::from_ref(&final_norm_target)).unwrap();

        let rotation = RandomizedHadamard::new(0xCAFE_BABE, hidden).unwrap();
        let plan = RotationPlan::qwen35_residual_stream_linear_layers();
        absorb_rotations(&mut tensors, &plan, &rotation).unwrap();

        // Post-pipeline: lm_head must differ from embed_tokens because the
        // final-norm scale was folded into lm_head before rotation.
        assert_ne!(
            tensors[QWEN35_LM_HEAD_NAME].data, tensors[QWEN35_EMBED_TOKENS_NAME].data,
            "lm_head must carry the (1 + g_final) factor that embed_tokens does not"
        );
    }
}
