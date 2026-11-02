//! Native Q8_0 CPU kernels, re-exported from `lattice-compute`.
//!
//! **Compatibility path.** These low-level APIs are unstable and may change;
//! new code should import them from `lattice-compute`.

#[doc(inline)]
pub use lattice_compute::forward::cpu::q8_neon::{
    matmul_q8_neon, matmul_q8_neon_into, matvec_q8_scalar, pack_weights_q8, quantize_vec_q8,
    quantize_vec_q8_into,
};
