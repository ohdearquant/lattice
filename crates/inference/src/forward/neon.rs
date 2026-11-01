//! Native Q8_0 NEON dotprod matvec kernel, re-exported from `lattice-model-qwen35`.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-qwen35` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-qwen35` directly.

#[doc(inline)]
pub use lattice_model_qwen35::forward::neon::{
    matmul_q8_neon, matmul_q8_neon_into, matvec_q8_scalar, pack_weights_q8, quantize_vec_q8,
    quantize_vec_q8_into,
};
