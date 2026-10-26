//! Shared attention semantics, the CPU attention kernels that build only on the CPU primitives, and the
//! GatedDeltaNet recurrent kernels.

pub mod decode;
pub mod differential;
pub mod flash_causal;
pub mod gated;
pub mod gdn;
pub mod gdn_fused;
pub mod gqa;
pub mod native_sparse;
pub mod softmax_row;
