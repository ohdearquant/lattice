//! **Compatibility path.** Re-exported from `lattice-compute` so that code written against the pre-split `lattice-inference`
//! keeps compiling. Not a stable API; removed at 1.0. Import from `lattice-compute` directly.

pub use lattice_compute::forward::gpu_gemm::{gpu_matmul, gpu_matmul_bt};
