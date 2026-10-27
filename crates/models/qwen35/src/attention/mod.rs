//! GatedDeltaNet steps that take a Qwen3.5 configuration.
//!
//! The kernels live in `lattice_compute::attention`; these modules build their validated
//! geometry from a [`Qwen35Config`](lattice_arch::qwen35_config::Qwen35Config).
pub mod gdn;
pub mod gdn_fused;
