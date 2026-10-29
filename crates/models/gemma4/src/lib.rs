//! Gemma4 CPU checkpoint assembly, cache and decoder sessions.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

/// Donor-slot cache for Gemma4 CPU execution.
pub mod gemma4_cache;
mod gemma4_loading;
/// Gemma4 text generation model.
pub mod gemma4_model;
/// Checkpoint tensor-inventory validation.
pub mod gemma4_preflight;
mod gemma4_weights;
mod gemma_cpu;

pub use gemma4_model::Gemma4Model;

/// Version-coupled generation glue used by the inference facade.
#[cfg(feature = "serving-support")]
#[doc(hidden)]
pub mod __internal {
    use crate::Gemma4Model;
    use lattice_arch::decoder::driver::DriverTrace;
    use lattice_arch::generation::{GenerateConfig, GenerateOutput};
    use lattice_inference_core::InferenceError;

    /// Keeps the existing driver observation available without exposing an inherent method.
    pub fn generate_traced(
        model: &Gemma4Model,
        prompt_ids: &[u32],
        gen_cfg: &GenerateConfig,
    ) -> Result<(GenerateOutput, DriverTrace), InferenceError> {
        model.generate_with_trace(prompt_ids, gen_cfg)
    }

    /// Keeps streaming driver observations at the family owner.
    pub fn generate_streaming_traced<F, C>(
        model: &Gemma4Model,
        prompt_ids: &[u32],
        gen_cfg: &GenerateConfig,
        on_token: F,
        should_cancel: C,
    ) -> Result<(GenerateOutput, DriverTrace), InferenceError>
    where
        F: FnMut(&str) -> bool,
        C: FnMut() -> bool,
    {
        model.generate_streaming_via_driver(prompt_ids, gen_cfg, on_token, should_cancel)
    }
}
