//! QuaRot conversion of a Qwen3.5 checkpoint into a rotated Q4 checkpoint, and its promotion records, re-exported from
//! `lattice-quant`.
//!
//! The re-exports are named rather than globbed so the facade never widens the supported API.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub(crate) use lattice_quant::__internal::read_quarot_seed_from_index;
#[doc(inline)]
pub use lattice_quant::quarot::convert::{
    ConversionOptions, ConversionReport, PplGateRecord, PromotionRecord, PromotionState,
    convert_quarot_qwen35, read_promotion_record, record_ppl_gate_result,
};
