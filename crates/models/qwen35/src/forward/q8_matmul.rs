//! Q8 matrix multiplication, re-exported from `lattice-compute`.

pub use lattice_compute::forward::cpu::q8_matmul::matmul_bt_q8;
pub(crate) use lattice_compute::forward::cpu::q8_matmul::{
    Q8MatmulScratch, matmul_bt_q8_with_scratch,
};
