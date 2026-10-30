//! Rotary position embedding tables, re-exported from `lattice-compute`.
//!
//! The re-export is named rather than globbed so the facade never widens the supported API.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-compute` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-compute` directly.

#[doc(inline)]
pub use lattice_compute::rope::RopeTable;
