//! Native Q8_0 NEON dotprod matvec kernel, re-exported from `lattice-model-qwen35`.

#[doc(inline)]
pub use lattice_model_qwen35::forward::neon::{
    matmul_q8_neon, matmul_q8_neon_into, matvec_q8_scalar, pack_weights_q8, quantize_vec_q8,
    quantize_vec_q8_into,
};
