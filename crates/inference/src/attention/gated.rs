//! Sigmoid output gating for gated attention, re-exported from `lattice-compute`.
//!
//! Named exports keep every existing path; the implementation lives in
//! `lattice_compute::attention::gated`.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-compute` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-compute` directly.

#[cfg(target_arch = "x86_64")]
#[doc(inline)]
pub use lattice_compute::attention::gated::apply_sigmoid_gate_avx2;
#[cfg(target_arch = "aarch64")]
#[doc(inline)]
pub use lattice_compute::attention::gated::apply_sigmoid_gate_neon;
#[doc(inline)]
pub use lattice_compute::attention::gated::{
    apply_sigmoid_gate, apply_sigmoid_gate_fast, apply_sigmoid_gate_scalar,
    apply_sigmoid_gate_simd, deinterleave_q_gate,
};
