//! ERNIE-4.5 f32 prefill and cached decode on Metal.
//!
//! This explicit backend requires macOS and `metal-gpu`. Loading the shipped
//! BF16 checkpoint additionally requires `f16`; already-loaded f32 weights do
//! not. The token-ID entry recomputes a complete sequence from position zero;
//! cached entries accept embeddings and explicit sectioned RoPE positions.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod state;
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub use state::MetalErnie45State;

/// Device-resident f32 post-RoPE keys and unrotated values for one decoder state.
///
/// Each store has layout `[layers, capacity, kv_dim]`. Only rows below `len`
/// are live. Create with [`MetalErnie45State::new_kv_cache`]; a cache belongs to
/// that exact state, so equal geometry cannot permit reuse with different weights.
pub struct MetalErnie45KvCache {
    len: usize,
    capacity: usize,
    layers: usize,
    kv_dim: usize,
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    storage: state::CacheStorage,
}

impl MetalErnie45KvCache {
    /// Number of live token rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether prefill is required before decoding.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum token count without reallocating.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of decoder layers.
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// Number of key or value elements per token and layer.
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Forget the sequence while retaining its allocation.
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

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

    /// Allocate a cache bound to this decoder state.
    ///
    /// # Errors
    /// Returns an availability error on this build.
    pub fn new_kv_cache(
        &self,
        _capacity: usize,
    ) -> Result<MetalErnie45KvCache, crate::InferenceError> {
        Err(crate::InferenceError::Inference(
            "ERNIE-4.5 Metal cache requires macOS and the metal-gpu feature".into(),
        ))
    }

    /// Prefill embeddings with explicit positions into an empty cache.
    ///
    /// # Errors
    /// Returns an availability error without changing the cache or logits.
    pub fn kv_prefill(
        &mut self,
        _embeds: &[f32],
        _positions: &[[u32; 3]],
        _cache: &mut MetalErnie45KvCache,
        _logits: &mut [f32],
    ) -> Result<(), crate::InferenceError> {
        Err(crate::InferenceError::Inference(
            "ERNIE-4.5 Metal cache requires macOS and the metal-gpu feature".into(),
        ))
    }

    /// Append one embedding row at its explicit position and return its logits.
    ///
    /// # Errors
    /// Returns an availability error without changing the cache or logits.
    pub fn kv_decode_step(
        &mut self,
        _embeds: &[f32],
        _position: [u32; 3],
        _cache: &mut MetalErnie45KvCache,
        _logits: &mut [f32],
    ) -> Result<(), crate::InferenceError> {
        Err(crate::InferenceError::Inference(
            "ERNIE-4.5 Metal cache requires macOS and the metal-gpu feature".into(),
        ))
    }
}

#[cfg(test)]
mod tests;
