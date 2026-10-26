//! Shared attention semantics and the CPU attention kernels that build only on the CPU primitives.

pub mod decode;
pub mod differential;
pub mod flash_causal;
pub mod gated;
pub mod gqa;
pub mod native_sparse;
pub mod softmax_row;
