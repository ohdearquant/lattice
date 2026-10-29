//! **Compatibility path.** Re-exported from `lattice-compute` so that code written against the pre-split `lattice-inference`
//! keeps compiling. Not a stable API; removed at 1.0. Import from `lattice-compute` directly.

#[cfg(target_arch = "aarch64")]
pub use lattice_compute::forward::bitnet_kernel::matvec_ternary_neon;
pub use lattice_compute::forward::bitnet_kernel::{
    matmul_ternary, matvec_ternary_scalar, pack_ternary, packed_row_bytes, quantize_activation,
    unpack_weight,
};
