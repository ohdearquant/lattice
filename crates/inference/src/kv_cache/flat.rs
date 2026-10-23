//! Flat (contiguous) KV cache for autoregressive decoding.
//!
//! Each transformer layer gets its own K and V buffer, pre-allocated to
//! `max_seq_len * num_kv_heads * head_dim`. Tokens are appended sequentially
//! during decode steps.
//!
//! Layout per layer: `[max_seq_len, kv_dim]` where `kv_dim = num_kv_heads * head_dim`.
//! Only `[0..seq_len, kv_dim]` contains valid data.
//!
//! Storage format: f16 (half-precision) to halve memory footprint vs f32.
//! Writes convert f32→f16; reads dequantize f16→f32 via `read_k_into`/`read_v_into`.

use half::f16;

use crate::error::InferenceError;

/// **Unstable**: flat KV cache configuration; fields may change as the
/// generation infrastructure evolves.
#[derive(Debug, Clone)]
pub struct FlatKVCacheConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV heads (GQA key-value heads, not query heads).
    pub num_kv_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Maximum sequence length the cache can hold.
    pub max_seq_len: usize,
}

impl FlatKVCacheConfig {
    /// **Unstable**: derived KV dimension.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Total f16 elements per layer buffer.
    #[inline]
    fn layer_capacity(&self) -> usize {
        self.max_seq_len * self.kv_dim()
    }

    /// **Unstable**: total memory footprint in bytes (f16 = 2 bytes per element).
    pub fn total_bytes(&self) -> usize {
        2 * self.num_layers * self.layer_capacity() * std::mem::size_of::<f16>()
    }

    /// **Unstable**: convenience constructor for Qwen3 models.
    pub fn for_qwen3(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
        }
    }
}

/// **Unstable**: flat KV cache; internal generation infrastructure,
/// not consumed outside this crate.
///
/// Pre-allocates all memory upfront. Append is O(1) per token per layer.
/// Best for single-sequence inference where the max context length is known.
///
/// Storage is f16 (half-precision) to halve memory relative to f32.
/// Callers supply f32 on write; reads dequantize via `read_k_into`/`read_v_into`.
#[derive(Debug)]
pub struct FlatKVCache {
    /// Per-layer K cache, each `[max_seq_len * kv_dim]` f16 elements.
    k: Vec<Vec<f16>>,
    /// Per-layer V cache, each `[max_seq_len * kv_dim]` f16 elements.
    v: Vec<Vec<f16>>,
    /// Current number of tokens cached (same across all layers).
    seq_len: usize,
    /// Config.
    config: FlatKVCacheConfig,
    /// Tracks which layers have been written at the current `seq_len` position.
    /// `advance()` requires this to be all-true before bumping `seq_len`.
    /// `advance_by()` skips this check (bulk prefill writes per-layer then advances).
    /// Reset to all-false on every successful `advance()`, `advance_by()`,
    /// `reset()`, `reset_fast()`, and `truncate_to()`.
    layers_written: Vec<bool>,
}

impl FlatKVCache {
    /// **Unstable**: construct with zero-initialized buffers.
    ///
    /// Dimensions are not validated against overflow; production callers must
    /// guard with `check_alloc_capacity` or equivalent before calling this.
    /// Prefer [`try_new`][`FlatKVCache::try_new`] when constructing from
    /// attacker-controlled or unvalidated config values.
    pub fn new(config: FlatKVCacheConfig) -> Self {
        let cap = config.layer_capacity();
        let num_layers = config.num_layers;
        let k = (0..num_layers).map(|_| vec![f16::ZERO; cap]).collect();
        let v = (0..num_layers).map(|_| vec![f16::ZERO; cap]).collect();
        Self {
            k,
            v,
            seq_len: 0,
            layers_written: vec![false; num_layers],
            config,
        }
    }

    /// **Unstable**: fallible constructor that validates all dimension products
    /// with checked arithmetic before allocating.
    ///
    /// Returns `Err(InferenceError::InvalidInput)` when any intermediate product
    /// (`num_kv_heads * head_dim`, `max_seq_len * kv_dim`, or the total element
    /// count) would overflow `usize`. Callers that supply attacker-influenced or
    /// user-supplied dimensions should prefer this over [`new`][`FlatKVCache::new`].
    ///
    /// # Errors
    /// Returns `InferenceError::InvalidInput` if any dimension product overflows `usize`.
    pub fn try_new(config: FlatKVCacheConfig) -> Result<Self, InferenceError> {
        // Validate kv_dim before any allocation so that Vec::with_capacity
        // never receives a silently-wrapped (attacker-influenced) capacity.
        let kv_dim = config
            .num_kv_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "num_kv_heads ({}) * head_dim ({}) overflows usize",
                    config.num_kv_heads, config.head_dim
                ))
            })?;
        let layer_cap = config.max_seq_len.checked_mul(kv_dim).ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "max_seq_len ({}) * kv_dim ({kv_dim}) overflows usize",
                config.max_seq_len
            ))
        })?;
        // Total element count across both K and V sides: 2 * num_layers * layer_cap.
        let _total = 2usize
            .checked_mul(config.num_layers)
            .and_then(|n| n.checked_mul(layer_cap))
            .ok_or_else(|| {
                InferenceError::InvalidInput(format!(
                    "2 * num_layers ({}) * layer_capacity ({layer_cap}) overflows usize",
                    config.num_layers
                ))
            })?;

        let num_layers = config.num_layers;
        let k: Vec<Vec<f16>> = (0..num_layers)
            .map(|_| vec![f16::ZERO; layer_cap])
            .collect();
        let v: Vec<Vec<f16>> = (0..num_layers)
            .map(|_| vec![f16::ZERO; layer_cap])
            .collect();
        Ok(Self {
            k,
            v,
            seq_len: 0,
            layers_written: vec![false; num_layers],
            config,
        })
    }

    /// **Unstable**: current cached sequence length.
    #[inline]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// **Unstable**: maximum tokens this cache can hold.
    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.config.max_seq_len
    }

    /// **Unstable**: number of transformer layers tracked.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// **Unstable**: KV dimension per token.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.config.kv_dim()
    }

    /// **Unstable**: returns true when the cache has reached max capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.seq_len >= self.config.max_seq_len
    }

    /// **Unstable**: append K/V for a single token.
    ///
    /// `k_token` and `v_token` must each have length `kv_dim`.
    /// Returns the position index where the token was stored.
    /// Converts f32→f16 on write.
    ///
    /// # Errors
    /// Returns `InferenceError::InvalidInput` if the cache is full, the layer index
    /// is out of bounds, or either slice length does not equal `kv_dim`.
    pub fn append_kv(
        &mut self,
        layer: usize,
        k_token: &[f32],
        v_token: &[f32],
    ) -> Result<usize, InferenceError> {
        let kv_dim = self.config.kv_dim();
        if k_token.len() != kv_dim {
            return Err(InferenceError::InvalidInput(
                "k_token length must equal kv_dim".to_string(),
            ));
        }
        if v_token.len() != kv_dim {
            return Err(InferenceError::InvalidInput(
                "v_token length must equal kv_dim".to_string(),
            ));
        }
        if layer >= self.config.num_layers {
            return Err(InferenceError::InvalidInput(
                "layer index out of bounds".to_string(),
            ));
        }
        if self.seq_len >= self.config.max_seq_len {
            return Err(InferenceError::InvalidInput(format!(
                "KV cache is full (seq_len={}, max={})",
                self.seq_len, self.config.max_seq_len
            )));
        }

        let offset = self.seq_len * kv_dim;
        for (i, &val) in k_token.iter().enumerate() {
            self.k[layer][offset + i] = f16::from_f32(val);
        }
        for (i, &val) in v_token.iter().enumerate() {
            self.v[layer][offset + i] = f16::from_f32(val);
        }

        // Mark this layer as written for the current token position.
        // `advance()` uses this to enforce the layer-completeness contract.
        self.layers_written[layer] = true;

        Ok(self.seq_len)
    }

    /// **Unstable**: advance position counter after all layers are appended.
    ///
    /// Enforces the layer-completeness contract: every layer must have been
    /// written (via [`append_kv`][`FlatKVCache::append_kv`]) at the current
    /// `seq_len` position before the position counter is advanced. This prevents
    /// a caller that only writes a subset of layers from publishing a position
    /// where the remaining layers contain stale (zero-initialized) data.
    ///
    /// For bulk prefill that writes layers independently, use
    /// [`prefill_layer`][`FlatKVCache::prefill_layer`] followed by
    /// [`advance_by`][`FlatKVCache::advance_by`], which skips this check.
    ///
    /// # Errors
    /// Returns `InferenceError::InvalidInput` if the cache is already full or
    /// if not all layers have been written at the current token position.
    pub fn advance(&mut self) -> Result<(), InferenceError> {
        if self.seq_len >= self.config.max_seq_len {
            return Err(InferenceError::InvalidInput(
                "cannot advance: cache is full".to_string(),
            ));
        }
        // Enforce the documented caller contract: all layers must be written
        // before the position is published. Collect the missing layer indices
        // for a diagnostic message rather than failing with a bare "incomplete".
        let missing: Vec<usize> = self
            .layers_written
            .iter()
            .enumerate()
            .filter(|(_, w)| !*w)
            .map(|(i, _)| i)
            .collect();
        if !missing.is_empty() {
            return Err(InferenceError::InvalidInput(format!(
                "advance() requires all {} layers written at position {}; \
                 layers {missing:?} have not been written",
                self.config.num_layers, self.seq_len,
            )));
        }
        self.seq_len += 1;
        self.layers_written.fill(false);
        Ok(())
    }

    /// **Unstable**: append K/V for all layers in one call and advance.
    ///
    /// # Errors
    /// Returns `InferenceError::InvalidInput` if `k_all` or `v_all` length does not
    /// equal the number of layers, or if any per-layer `append_kv` or `advance` fails.
    pub fn append_kv_all_layers(
        &mut self,
        k_all: &[&[f32]],
        v_all: &[&[f32]],
    ) -> Result<(), InferenceError> {
        if k_all.len() != self.config.num_layers {
            return Err(InferenceError::InvalidInput(format!(
                "k_all length {} does not match num_layers {}",
                k_all.len(),
                self.config.num_layers
            )));
        }
        if v_all.len() != self.config.num_layers {
            return Err(InferenceError::InvalidInput(format!(
                "v_all length {} does not match num_layers {}",
                v_all.len(),
                self.config.num_layers
            )));
        }
        for layer in 0..self.config.num_layers {
            self.append_kv(layer, k_all[layer], v_all[layer])?;
        }
        self.advance()?;
        Ok(())
    }

    /// **Unstable**: batch-append tokens for one layer during prefill.
    /// Converts f32→f16 on write.
    ///
    /// # Errors
    /// Returns `InferenceError::InvalidInput` if slice lengths are wrong, the layer
    /// index is out of bounds, or the tokens would overflow the cache capacity.
    pub fn prefill_layer(
        &mut self,
        layer: usize,
        k_tokens: &[f32],
        v_tokens: &[f32],
        num_tokens: usize,
    ) -> Result<(), InferenceError> {
        let kv_dim = self.config.kv_dim();
        // Use checked arithmetic so a malformed `num_tokens` returns a clean
        // error rather than overflow-panicking in debug builds (matches `advance_by`).
        let total = num_tokens.checked_mul(kv_dim).ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "prefill_layer: num_tokens ({num_tokens}) * kv_dim ({kv_dim}) overflows usize"
            ))
        })?;
        if k_tokens.len() != total {
            return Err(InferenceError::InvalidInput(format!(
                "k_tokens length {} does not equal num_tokens * kv_dim ({total})",
                k_tokens.len(),
            )));
        }
        if v_tokens.len() != total {
            return Err(InferenceError::InvalidInput(format!(
                "v_tokens length {} does not equal num_tokens * kv_dim ({total})",
                v_tokens.len(),
            )));
        }
        if layer >= self.config.num_layers {
            return Err(InferenceError::InvalidInput(
                "layer index out of bounds".to_string(),
            ));
        }
        let new_len = self.seq_len.checked_add(num_tokens).ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "prefill_layer: seq_len ({}) + num_tokens ({num_tokens}) overflows usize",
                self.seq_len
            ))
        })?;
        if new_len > self.config.max_seq_len {
            return Err(InferenceError::InvalidInput(
                "prefill would exceed max_seq_len".to_string(),
            ));
        }

        let offset = self.seq_len * kv_dim;
        for (i, &val) in k_tokens[..total].iter().enumerate() {
            self.k[layer][offset + i] = f16::from_f32(val);
        }
        for (i, &val) in v_tokens[..total].iter().enumerate() {
            self.v[layer][offset + i] = f16::from_f32(val);
        }
        Ok(())
    }

    /// **Unstable**: advance position counter by n after prefilling all layers.
    ///
    /// Skips the per-layer completeness check performed by
    /// [`advance`][`FlatKVCache::advance`]; intended for use after
    /// [`prefill_layer`][`FlatKVCache::prefill_layer`], which writes multiple
    /// tokens per layer rather than one token per `advance` call.
    ///
    /// # Errors
    /// Returns `InferenceError::InvalidInput` if `seq_len + n` would overflow
    /// `usize` or exceed `max_seq_len`.
    pub fn advance_by(&mut self, n: usize) -> Result<(), InferenceError> {
        let new_len = self.seq_len.checked_add(n).ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "advance_by({n}) would overflow usize (seq_len={})",
                self.seq_len
            ))
        })?;
        if new_len > self.config.max_seq_len {
            return Err(InferenceError::InvalidInput(
                "advance_by would exceed max_seq_len".to_string(),
            ));
        }
        self.seq_len = new_len;
        // Reset the per-layer write tracking so the next decode step starts clean.
        self.layers_written.fill(false);
        Ok(())
    }

    /// **Unstable**: roll back the cache to a shorter sequence without deallocating buffers.
    ///
    /// Does not zero K/V buffers — later forwards overwrite from `seq_len` onward.
    pub fn truncate_to(&mut self, seq_len: usize) {
        assert!(
            seq_len <= self.seq_len,
            "truncate_to cannot grow cache: requested {}, current {}",
            seq_len,
            self.seq_len
        );
        self.seq_len = seq_len;
        // The rewound position has no pending writes yet; reset tracking so the
        // next append_kv → advance cycle starts from a clean state.
        self.layers_written.fill(false);
    }

    /// **Unstable**: dequantize K slice for a layer into `buf` up to current seq_len.
    ///
    /// `buf` must have length >= `seq_len * kv_dim`. Converts f16→f32.
    pub fn read_k_into(&self, layer: usize, buf: &mut [f32]) {
        assert!(layer < self.config.num_layers, "layer index out of bounds");
        let end = self.seq_len * self.config.kv_dim();
        assert!(
            buf.len() >= end,
            "output buffer too small: need {end}, got {}",
            buf.len()
        );
        for (i, &h) in self.k[layer][..end].iter().enumerate() {
            buf[i] = h.to_f32();
        }
    }

    /// **Unstable**: dequantize V slice for a layer into `buf` up to current seq_len.
    ///
    /// `buf` must have length >= `seq_len * kv_dim`. Converts f16→f32.
    pub fn read_v_into(&self, layer: usize, buf: &mut [f32]) {
        assert!(layer < self.config.num_layers, "layer index out of bounds");
        let end = self.seq_len * self.config.kv_dim();
        assert!(
            buf.len() >= end,
            "output buffer too small: need {end}, got {}",
            buf.len()
        );
        for (i, &h) in self.v[layer][..end].iter().enumerate() {
            buf[i] = h.to_f32();
        }
    }

    /// **Unstable**: read K slice for a layer up to current seq_len (f16).
    ///
    /// Use `read_k_into` for f32 dequantized access.
    #[inline]
    pub fn get_k_f16(&self, layer: usize) -> &[f16] {
        assert!(layer < self.config.num_layers, "layer index out of bounds");
        let end = self.seq_len * self.config.kv_dim();
        &self.k[layer][..end]
    }

    /// **Unstable**: read V slice for a layer up to current seq_len (f16).
    ///
    /// Use `read_v_into` for f32 dequantized access.
    #[inline]
    pub fn get_v_f16(&self, layer: usize) -> &[f16] {
        assert!(layer < self.config.num_layers, "layer index out of bounds");
        let end = self.seq_len * self.config.kv_dim();
        &self.v[layer][..end]
    }

    /// **Unstable**: read K slice for a layer up to current seq_len, dequantizing to f32.
    ///
    /// Allocates a Vec; prefer `read_k_into` when a pre-allocated buffer is available.
    #[inline]
    pub fn get_k(&self, layer: usize) -> Vec<f32> {
        assert!(layer < self.config.num_layers, "layer index out of bounds");
        let end = self.seq_len * self.config.kv_dim();
        self.k[layer][..end].iter().map(|h| h.to_f32()).collect()
    }

    /// **Unstable**: read V slice for a layer up to current seq_len, dequantizing to f32.
    ///
    /// Allocates a Vec; prefer `read_v_into` when a pre-allocated buffer is available.
    #[inline]
    pub fn get_v(&self, layer: usize) -> Vec<f32> {
        assert!(layer < self.config.num_layers, "layer index out of bounds");
        let end = self.seq_len * self.config.kv_dim();
        self.v[layer][..end].iter().map(|h| h.to_f32()).collect()
    }

    /// **Unstable**: mutable K slice for in-place f16 operations (e.g. RoPE applied in f16).
    #[inline]
    pub fn get_k_mut(&mut self, layer: usize) -> &mut [f16] {
        assert!(layer < self.config.num_layers, "layer index out of bounds");
        let end = self.seq_len * self.config.kv_dim();
        &mut self.k[layer][..end]
    }

    /// **Unstable**: mutable V slice for in-place f16 operations.
    #[inline]
    pub fn get_v_mut(&mut self, layer: usize) -> &mut [f16] {
        assert!(layer < self.config.num_layers, "layer index out of bounds");
        let end = self.seq_len * self.config.kv_dim();
        &mut self.v[layer][..end]
    }

    /// **Unstable**: full K buffer including unwritten positions (f16); used by prefill.
    #[inline]
    pub fn k_buffer_mut(&mut self, layer: usize) -> &mut [f16] {
        assert!(layer < self.config.num_layers, "layer index out of bounds");
        &mut self.k[layer]
    }

    /// **Unstable**: immutable full K buffer (f16).
    #[inline]
    pub fn k_buffer(&self, layer: usize) -> &[f16] {
        assert!(layer < self.config.num_layers, "layer index out of bounds");
        &self.k[layer]
    }

    /// **Unstable**: full V buffer including unwritten positions (f16).
    #[inline]
    pub fn v_buffer_mut(&mut self, layer: usize) -> &mut [f16] {
        assert!(layer < self.config.num_layers, "layer index out of bounds");
        &mut self.v[layer]
    }

    /// **Unstable**: immutable full V buffer (f16).
    #[inline]
    pub fn v_buffer(&self, layer: usize) -> &[f16] {
        assert!(layer < self.config.num_layers, "layer index out of bounds");
        &self.v[layer]
    }

    /// **Unstable**: reset and zero all KV data; does not deallocate.
    pub fn reset(&mut self) {
        self.seq_len = 0;
        self.layers_written.fill(false);
        // Zero out for safety (prevent stale data leaking).
        for layer in 0..self.config.num_layers {
            self.k[layer].fill(f16::ZERO);
            self.v[layer].fill(f16::ZERO);
        }
    }

    /// **Unstable**: fast reset; stale data remains in buffers.
    pub fn reset_fast(&mut self) {
        self.seq_len = 0;
        self.layers_written.fill(false);
    }

    /// **Unstable**: total memory usage of this cache in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.config.total_bytes()
    }
}

#[cfg(test)]
mod tests;
