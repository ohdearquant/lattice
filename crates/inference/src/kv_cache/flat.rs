//! Flat (contiguous) KV cache for autoregressive decoding.
//!
//! Each transformer layer gets its own K and V buffer, pre-allocated to
//! `max_seq_len * num_kv_heads * head_dim`. Tokens are appended sequentially
//! during decode steps.
//!
//! Layout per layer: `[max_seq_len, kv_dim]` where `kv_dim = num_kv_heads * head_dim`.
//! Only `[0..seq_len, kv_dim]` contains valid data.

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

    /// Total f32 elements per layer buffer.
    #[inline]
    fn layer_capacity(&self) -> usize {
        self.max_seq_len * self.kv_dim()
    }

    /// **Unstable**: total memory footprint in bytes.
    pub fn total_bytes(&self) -> usize {
        2 * self.num_layers * self.layer_capacity() * std::mem::size_of::<f32>()
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
#[derive(Debug)]
pub struct FlatKVCache {
    /// Per-layer K cache, each `[max_seq_len * kv_dim]`.
    k: Vec<Vec<f32>>,
    /// Per-layer V cache, each `[max_seq_len * kv_dim]`.
    v: Vec<Vec<f32>>,
    /// Current number of tokens cached (same across all layers).
    seq_len: usize,
    /// Config.
    config: FlatKVCacheConfig,
}

impl FlatKVCache {
    /// **Unstable**: construct with zero-initialized buffers.
    pub fn new(config: FlatKVCacheConfig) -> Self {
        let cap = config.layer_capacity();
        let k = (0..config.num_layers).map(|_| vec![0.0f32; cap]).collect();
        let v = (0..config.num_layers).map(|_| vec![0.0f32; cap]).collect();
        Self {
            k,
            v,
            seq_len: 0,
            config,
        }
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

    /// **Unstable**: append K/V for a single token; panics on overflow.
    ///
    /// `k_token` and `v_token` must each have length `kv_dim`.
    /// Returns the position index where the token was stored.
    ///
    /// # Panics
    /// Panics if the cache is full or if slice lengths are wrong.
    pub fn append_kv(&mut self, layer: usize, k_token: &[f32], v_token: &[f32]) -> usize {
        let kv_dim = self.config.kv_dim();
        assert_eq!(k_token.len(), kv_dim, "k_token length must equal kv_dim");
        assert_eq!(v_token.len(), kv_dim, "v_token length must equal kv_dim");
        assert!(layer < self.config.num_layers, "layer index out of bounds");
        assert!(
            self.seq_len < self.config.max_seq_len,
            "KV cache is full (seq_len={}, max={})",
            self.seq_len,
            self.config.max_seq_len
        );

        let offset = self.seq_len * kv_dim;
        self.k[layer][offset..offset + kv_dim].copy_from_slice(k_token);
        self.v[layer][offset..offset + kv_dim].copy_from_slice(v_token);

        self.seq_len
    }

    /// **Unstable**: advance position counter after all layers are appended.
    pub fn advance(&mut self) {
        assert!(
            self.seq_len < self.config.max_seq_len,
            "cannot advance: cache is full"
        );
        self.seq_len += 1;
    }

    /// **Unstable**: append K/V for all layers in one call and advance.
    pub fn append_kv_all_layers(&mut self, k_all: &[&[f32]], v_all: &[&[f32]]) {
        assert_eq!(k_all.len(), self.config.num_layers);
        assert_eq!(v_all.len(), self.config.num_layers);
        for layer in 0..self.config.num_layers {
            self.append_kv(layer, k_all[layer], v_all[layer]);
        }
        self.advance();
    }

    /// **Unstable**: batch-append tokens for one layer during prefill.
    pub fn prefill_layer(
        &mut self,
        layer: usize,
        k_tokens: &[f32],
        v_tokens: &[f32],
        num_tokens: usize,
    ) {
        let kv_dim = self.config.kv_dim();
        assert_eq!(k_tokens.len(), num_tokens * kv_dim);
        assert_eq!(v_tokens.len(), num_tokens * kv_dim);
        assert!(layer < self.config.num_layers);
        assert!(
            self.seq_len + num_tokens <= self.config.max_seq_len,
            "prefill would exceed max_seq_len"
        );

        let offset = self.seq_len * kv_dim;
        let total = num_tokens * kv_dim;
        self.k[layer][offset..offset + total].copy_from_slice(k_tokens);
        self.v[layer][offset..offset + total].copy_from_slice(v_tokens);
    }

    /// **Unstable**: advance position counter by n after prefilling all layers.
    pub fn advance_by(&mut self, n: usize) {
        assert!(
            self.seq_len + n <= self.config.max_seq_len,
            "advance_by would exceed max_seq_len"
        );
        self.seq_len += n;
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
    }

    /// **Unstable**: read K slice for a layer up to current seq_len.
    #[inline]
    pub fn get_k(&self, layer: usize) -> &[f32] {
        debug_assert!(layer < self.config.num_layers);
        let end = self.seq_len * self.config.kv_dim();
        &self.k[layer][..end]
    }

    /// **Unstable**: read V slice for a layer up to current seq_len.
    #[inline]
    pub fn get_v(&self, layer: usize) -> &[f32] {
        debug_assert!(layer < self.config.num_layers);
        let end = self.seq_len * self.config.kv_dim();
        &self.v[layer][..end]
    }

    /// **Unstable**: mutable K slice for in-place operations (e.g. RoPE).
    #[inline]
    pub fn get_k_mut(&mut self, layer: usize) -> &mut [f32] {
        debug_assert!(layer < self.config.num_layers);
        let end = self.seq_len * self.config.kv_dim();
        &mut self.k[layer][..end]
    }

    /// **Unstable**: mutable V slice for in-place operations.
    #[inline]
    pub fn get_v_mut(&mut self, layer: usize) -> &mut [f32] {
        debug_assert!(layer < self.config.num_layers);
        let end = self.seq_len * self.config.kv_dim();
        &mut self.v[layer][..end]
    }

    /// **Unstable**: full K buffer including unwritten positions; used by prefill.
    #[inline]
    pub fn k_buffer_mut(&mut self, layer: usize) -> &mut [f32] {
        debug_assert!(layer < self.config.num_layers);
        &mut self.k[layer]
    }

    /// **Unstable**: immutable full K buffer.
    #[inline]
    pub fn k_buffer(&self, layer: usize) -> &[f32] {
        debug_assert!(layer < self.config.num_layers);
        &self.k[layer]
    }

    /// **Unstable**: full V buffer including unwritten positions.
    #[inline]
    pub fn v_buffer_mut(&mut self, layer: usize) -> &mut [f32] {
        debug_assert!(layer < self.config.num_layers);
        &mut self.v[layer]
    }

    /// **Unstable**: immutable full V buffer.
    #[inline]
    pub fn v_buffer(&self, layer: usize) -> &[f32] {
        debug_assert!(layer < self.config.num_layers);
        &self.v[layer]
    }

    /// **Unstable**: reset and zero all KV data; does not deallocate.
    pub fn reset(&mut self) {
        self.seq_len = 0;
        // Zero out for safety (prevent stale data leaking).
        for layer in 0..self.config.num_layers {
            self.k[layer].fill(0.0);
            self.v[layer].fill(0.0);
        }
    }

    /// **Unstable**: fast reset; stale data remains in buffers.
    pub fn reset_fast(&mut self) {
        self.seq_len = 0;
    }

    /// **Unstable**: total memory usage of this cache in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.config.total_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(num_layers: usize, max_seq_len: usize) -> FlatKVCacheConfig {
        FlatKVCacheConfig {
            num_layers,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len,
        }
    }

    #[test]
    fn append_get_roundtrip() {
        let config = make_config(2, 64);
        let kv_dim = config.kv_dim(); // 8 * 128 = 1024
        let mut cache = FlatKVCache::new(config);

        // Append one token to layer 0.
        let k: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.01).collect();
        let v: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.02 + 1.0).collect();
        cache.append_kv(0, &k, &v);

        // Also append to layer 1.
        let k1: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.03).collect();
        let v1: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.04 + 2.0).collect();
        cache.append_kv(1, &k1, &v1);

        cache.advance();
        assert_eq!(cache.seq_len(), 1);

        // Verify roundtrip.
        let got_k = cache.get_k(0);
        assert_eq!(got_k.len(), kv_dim);
        assert_eq!(got_k, &k[..]);

        let got_v = cache.get_v(0);
        assert_eq!(got_v, &v[..]);

        let got_k1 = cache.get_k(1);
        assert_eq!(got_k1, &k1[..]);

        let got_v1 = cache.get_v(1);
        assert_eq!(got_v1, &v1[..]);
    }

    #[test]
    fn per_layer_isolation() {
        let config = make_config(4, 32);
        let kv_dim = config.kv_dim();
        let mut cache = FlatKVCache::new(config);

        // Append different data to each layer.
        for layer in 0..4 {
            let marker = (layer + 1) as f32;
            let k = vec![marker; kv_dim];
            let v = vec![marker + 0.5; kv_dim];
            cache.append_kv(layer, &k, &v);
        }
        cache.advance();

        // Each layer should have its own data.
        for layer in 0..4 {
            let marker = (layer + 1) as f32;
            let got_k = cache.get_k(layer);
            assert!(
                got_k.iter().all(|&x| x == marker),
                "layer {layer} K mismatch"
            );
            let got_v = cache.get_v(layer);
            assert!(
                got_v.iter().all(|&x| x == marker + 0.5),
                "layer {layer} V mismatch"
            );
        }
    }

    #[test]
    fn reset_clears_state() {
        let config = make_config(2, 16);
        let kv_dim = config.kv_dim();
        let mut cache = FlatKVCache::new(config);

        let k = vec![1.0; kv_dim];
        let v = vec![2.0; kv_dim];
        cache.append_kv(0, &k, &v);
        cache.append_kv(1, &k, &v);
        cache.advance();
        assert_eq!(cache.seq_len(), 1);

        cache.reset();
        assert_eq!(cache.seq_len(), 0);
        // After reset, get_k returns empty slice.
        assert_eq!(cache.get_k(0).len(), 0);
        assert_eq!(cache.get_v(1).len(), 0);
    }

    #[test]
    fn capacity_limit() {
        let config = make_config(1, 4);
        let kv_dim = config.kv_dim();
        let mut cache = FlatKVCache::new(config);

        let k = vec![1.0; kv_dim];
        let v = vec![2.0; kv_dim];

        // Fill to capacity.
        for _ in 0..4 {
            cache.append_kv(0, &k, &v);
            cache.advance();
        }
        assert!(cache.is_full());
        assert_eq!(cache.seq_len(), 4);
    }

    #[test]
    #[should_panic(expected = "KV cache is full")]
    fn append_beyond_capacity_panics() {
        let config = make_config(1, 2);
        let kv_dim = config.kv_dim();
        let mut cache = FlatKVCache::new(config);

        let k = vec![1.0; kv_dim];
        let v = vec![2.0; kv_dim];
        cache.append_kv(0, &k, &v);
        cache.advance();
        cache.append_kv(0, &k, &v);
        cache.advance();
        // This should panic.
        cache.append_kv(0, &k, &v);
    }

    #[test]
    fn prefill_then_decode() {
        let config = make_config(2, 64);
        let kv_dim = config.kv_dim();
        let mut cache = FlatKVCache::new(config);

        let prompt_len = 10;

        // Prefill: 10 tokens, each layer.
        for layer in 0..2 {
            let marker = (layer + 1) as f32;
            let k_prefill: Vec<f32> = (0..prompt_len * kv_dim)
                .map(|i| marker + (i as f32) * 0.001)
                .collect();
            let v_prefill: Vec<f32> = (0..prompt_len * kv_dim)
                .map(|i| marker + 0.5 + (i as f32) * 0.001)
                .collect();
            cache.prefill_layer(layer, &k_prefill, &v_prefill, prompt_len);
        }
        cache.advance_by(prompt_len);
        assert_eq!(cache.seq_len(), prompt_len);

        // Decode: append 5 more tokens one at a time.
        for step in 0..5 {
            for layer in 0..2 {
                let marker = 100.0 + step as f32;
                let k = vec![marker; kv_dim];
                let v = vec![marker + 0.5; kv_dim];
                cache.append_kv(layer, &k, &v);
            }
            cache.advance();
        }
        assert_eq!(cache.seq_len(), 15);

        // Verify decode tokens are at correct positions.
        let k0 = cache.get_k(0);
        assert_eq!(k0.len(), 15 * kv_dim);

        // Check the first decode token (at position 10).
        let decode_start = 10 * kv_dim;
        assert_eq!(k0[decode_start], 100.0);

        // Check last decode token (at position 14).
        let last_start = 14 * kv_dim;
        assert_eq!(k0[last_start], 104.0);
    }

    #[test]
    fn memory_bytes_calculation() {
        // Qwen3-0.6B: 28 layers, 8 KV heads, head_dim=128, max 4096
        // kv_dim = 8 * 128 = 1024
        // Per side: 28 * 4096 * 1024 * 4 bytes
        // Total: 2 * that = 939,524,096 bytes ~ 0.875 GB
        let config = FlatKVCacheConfig::for_qwen3(28, 8, 128, 4096);
        let kv_dim = 8 * 128; // 1024
        let expected = 2 * 28 * 4096 * kv_dim * 4;
        assert_eq!(config.total_bytes(), expected);
        let mb = config.total_bytes() as f64 / (1024.0 * 1024.0);
        assert!((mb - 896.0).abs() < 1.0, "expected ~896 MB, got {mb:.1} MB");
    }

    #[test]
    fn truncate_to_rewinds_seq_len_without_deallocating() {
        let config = FlatKVCacheConfig {
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            max_seq_len: 8,
        };
        let kv_dim = config.kv_dim();
        let mut cache = FlatKVCache::new(config);

        cache.advance_by(5);
        let k_len_before = cache.k_buffer(0).len();
        let v_len_before = cache.v_buffer(0).len();

        cache.truncate_to(3);
        assert_eq!(cache.seq_len(), 3);
        // Buffer lengths must be unchanged (no deallocation)
        assert_eq!(cache.k_buffer(0).len(), k_len_before);
        assert_eq!(cache.v_buffer(0).len(), v_len_before);
        // Valid slice shrinks accordingly
        assert_eq!(cache.get_k(0).len(), 3 * kv_dim);
    }

    #[test]
    fn reset_fast_preserves_buffer() {
        let config = make_config(1, 8);
        let kv_dim = config.kv_dim();
        let mut cache = FlatKVCache::new(config);

        let k = vec![42.0; kv_dim];
        let v = vec![43.0; kv_dim];
        cache.append_kv(0, &k, &v);
        cache.advance();

        cache.reset_fast();
        assert_eq!(cache.seq_len(), 0);
        // Stale data is still there in the raw buffer (this is expected).
        assert_eq!(cache.k[0][0], 42.0);
    }

    #[test]
    fn get_k_mut_allows_modification() {
        let config = make_config(1, 8);
        let kv_dim = config.kv_dim();
        let mut cache = FlatKVCache::new(config);

        let k = vec![1.0; kv_dim];
        let v = vec![2.0; kv_dim];
        cache.append_kv(0, &k, &v);
        cache.advance();

        // Modify K in-place (e.g., for RoPE).
        let k_mut = cache.get_k_mut(0);
        k_mut[0] = 99.0;

        assert_eq!(cache.get_k(0)[0], 99.0);
    }
}
