//! LM-head materialization and untying for QuaRot conversion, re-exported from `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.

#[doc(inline)]
pub use lattice_quant::quarot::lm_head::{
    QWEN35_EMBED_TOKENS_NAME, QWEN35_FINAL_NORM_NAME, QWEN35_LM_HEAD_NAME,
    materialize_lm_head_for_qwen35, qwen35_final_norm_fusion_target, untie_word_embeddings_in_cfg,
    untie_word_embeddings_in_config_json,
};
