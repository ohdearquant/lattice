//! Rotary position embedding tables, re-exported from `lattice-compute`.
//!
//! The re-export is named rather than globbed so the facade never widens the supported API.

#[doc(inline)]
pub use lattice_compute::rope::RopeTable;
