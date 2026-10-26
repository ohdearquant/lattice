//! Qwen3.5 CPU and Metal model, generation and vision composition.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

pub mod decoder;
pub mod family;
pub mod forward;
mod generation;
pub mod model;
mod sampling;
#[cfg(test)]
mod signed_scale_fixture;
pub mod speculative;
mod stop_reason;
#[cfg(test)]
mod stop_token_contract;
pub mod vision;

#[doc(inline)]
pub use lattice_arch::qwen35_config::VisionModelConfig;

#[cfg(feature = "serving-support")]
pub mod serving_support;

#[cfg(feature = "serving-support")]
#[doc(hidden)]
#[path = "internal.rs"]
pub mod __internal;

/// Repository checkpoint fixtures for unsupported test utilities.
#[cfg(any(test, feature = "test-utils"))]
pub mod qwen35_checkpoint_fixture;

#[cfg(any(test, feature = "test-utils"))]
mod reference_scoring_fixture;

/// Unsupported model-specific fixtures for version-coupled tests.
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub mod test_support;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub mod lora_file;

/// Qwen3.5 expert-cache trace admission simulation.
pub mod moe_admission;
