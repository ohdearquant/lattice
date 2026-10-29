//! Qwen3.5 CPU and Metal model, generation and vision composition.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

pub mod attention;
pub mod decoder;
pub mod family;
pub mod forward;
/// Experimental, scoped CPU GDN state-health observations.
#[cfg(feature = "gdn-state-observer")]
#[doc(hidden)]
pub mod gdn_state_observer;
mod generation;
pub mod lora_hook;
pub mod model;
mod sampling;
/// Negative-scale Q4 fixture shared by every Q4 consumer's sign test.
///
/// Every block has a negative scale, so a consumer that drops the sign, clamps the
/// scale at zero or otherwise reads the scale's sign computes different weights. Each
/// consumer test runs the fixture, the same blocks with `|scale|`, and the same blocks
/// with a zero scale: the first must match the f64 reference, and the other two, which
/// are exactly what a sign-dropping or zero-clamping consumer would compute on the
/// fixture, must miss it by more than ten times the tolerance.
#[cfg(test)]
mod signed_scale_fixture;
pub mod speculative;
mod stop_reason;
/// Cross-path sweep (#613): every CPU-family `generate*` entry point agrees on
/// the stop-token contract (excluded from `token_ids`/`text`). The Metal-family
/// entry points are covered in `forward::metal_qwen35`'s own test module; see
/// this module's doc comment for the full manifest and rationale.
#[cfg(test)]
mod stop_token_contract;
pub mod vision;
pub mod weights;

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

#[cfg(any(
    all(test, target_os = "macos", feature = "metal-gpu"),
    feature = "test-utils"
))]
mod reference_scoring_fixture;

/// Unsupported model-specific fixtures for version-coupled tests.
#[cfg(feature = "test-utils")]
#[doc(hidden)]
pub mod test_support;

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub mod lora_file;

/// Qwen3.5 expert-cache trace admission simulation.
pub mod moe_admission;
