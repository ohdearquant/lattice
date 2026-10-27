//! Gemma4 checkpoint preflight, re-exported from its family crate.

#[doc(inline)]
pub use lattice_model_gemma4::gemma4_preflight::{
    PreflightReport, TensorInfo, TensorInventory, deserialize_tensor_inventory_no_duplicates,
    preflight_check,
};
