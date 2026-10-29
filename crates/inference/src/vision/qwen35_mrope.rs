//! Qwen3.5 family compatibility exports.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-qwen35` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-qwen35` directly.

#[doc(inline)]
pub use lattice_model_qwen35::vision::qwen35_mrope::MRopePositions;

#[doc(inline)]
pub use lattice_model_qwen35::vision::qwen35_mrope::MRopeTables;

#[doc(inline)]
pub use lattice_model_qwen35::vision::qwen35_mrope::build_position_ids;

#[doc(inline)]
pub use lattice_model_qwen35::vision::qwen35_mrope::build_cos_sin;

#[doc(inline)]
pub use lattice_model_qwen35::vision::qwen35_mrope::decode_position;
