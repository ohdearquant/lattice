//! ERNIE4.5 Metal state and cache, owned by the ERNIE4.5 family.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-ernie45` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-ernie45` directly.

#[doc(inline)]
pub use lattice_model_ernie45::forward::metal_ernie45::{MetalErnie45KvCache, MetalErnie45State};
