//! LM-head materialization and untying for QuaRot conversion, re-exported from `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-quant` so that code written
//! against the pre-split `lattice-inference` keeps compiling. These low-level re-exports are not a
//! stable API and will be removed at 1.0; import from `lattice-quant` directly.

#[doc(inline)]
pub use lattice_quant::quarot::lm_head::{
    QWEN35_EMBED_TOKENS_NAME, QWEN35_FINAL_NORM_NAME, QWEN35_LM_HEAD_NAME,
    materialize_lm_head_for_qwen35, qwen35_final_norm_fusion_target, untie_word_embeddings_in_cfg,
    untie_word_embeddings_in_config_json,
};
