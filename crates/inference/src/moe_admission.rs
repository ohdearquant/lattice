//! Qwen3.5 family compatibility exports.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-qwen35` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-qwen35` directly.

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::TraceRecord;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::TraceReadError;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::read_trace;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::group_by_layer;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::AccessOutcome;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::AdmissionPolicy;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::LruPolicy;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::ArcPolicy;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::FreqAdmissionPolicy;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::PolicySpec;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::LayerStats;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::simulate_layer;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::SimConfig;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::LayerRow;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::PolicyReport;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::run_simulation;

#[doc(inline)]
pub use lattice_model_qwen35::moe_admission::format_table;
