//! ERNIE-4.5 f32 text prefill on Metal with an independent CPU reference.
//!
//! This explicit backend requires macOS and `metal-gpu`. Loading the shipped
//! BF16 checkpoint additionally requires `f16`; already-loaded f32 weights do
//! not. Calls process one complete sequence with positions starting at zero and
//! produce logits for every input position. No KV cache or multimodal positions
//! are retained between calls.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod state;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub use state::MetalErnie45State;

/// ERNIE-4.5 Metal text-prefill state; unavailable without macOS and `metal-gpu`.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub struct MetalErnie45State;

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
impl MetalErnie45State {
    /// Create persistent resources for a bounded text sequence.
    ///
    /// # Errors
    /// Returns an availability error on this build. The supported implementation
    /// validates model geometry, tensor contents, capacity and device limits.
    pub fn new(
        _config: &crate::model::ernie45::Ernie45Config,
        _weights: &crate::model::ernie45::Ernie45Weights,
        _max_seq_len: usize,
    ) -> Result<Self, crate::InferenceError> {
        Err(crate::InferenceError::Inference(
            "ERNIE-4.5 Metal prefill requires macOS and the metal-gpu feature".into(),
        ))
    }

    /// Recompute a complete text sequence into row-major `[ids.len(), vocab_size]` logits.
    ///
    /// The output slice must have exactly that shape. No retained KV cache is used.
    ///
    /// # Errors
    /// Returns an availability error without changing the caller's output.
    pub fn prefill(
        &mut self,
        _ids: &[u32],
        _logits: &mut [f32],
    ) -> Result<(), crate::InferenceError> {
        Err(crate::InferenceError::Inference(
            "ERNIE-4.5 Metal prefill requires macOS and the metal-gpu feature".into(),
        ))
    }
}

#[cfg(test)]
mod tests;
