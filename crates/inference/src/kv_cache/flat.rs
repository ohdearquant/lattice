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
}

impl FlatKVCache {
    /// **Unstable**: construct with zero-initialized buffers.
    pub fn new(config: FlatKVCacheConfig) -> Self {
        let cap = config.layer_capacity();
        let k = (0..config.num_layers)
            .map(|_| vec![f16::ZERO; cap])
            .collect();
        let v = (0..config.num_layers)
            .map(|_| vec![f16::ZERO; cap])
            .collect();
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
    /// Converts f32→f16 on write.
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
        for (i, &val) in k_token.iter().enumerate() {
            self.k[layer][offset + i] = f16::from_f32(val);
        }
        for (i, &val) in v_token.iter().enumerate() {
            self.v[layer][offset + i] = f16::from_f32(val);
        }

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
    /// Converts f32→f16 on write.
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
        for (i, &val) in k_tokens[..total].iter().enumerate() {
            self.k[layer][offset + i] = f16::from_f32(val);
        }
        for (i, &val) in v_tokens[..total].iter().enumerate() {
            self.v[layer][offset + i] = f16::from_f32(val);
        }
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
        // Zero out for safety (prevent stale data leaking).
        for layer in 0..self.config.num_layers {
            self.k[layer].fill(f16::ZERO);
            self.v[layer].fill(f16::ZERO);
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

    /// f16 tolerance: ~3.3 decimal digits relative precision (eps ≈ 9.77e-4).
    /// We use max(absolute_tol, relative_tol * max(|a|, |b|)) to handle both
    /// near-zero and large values correctly.
    fn approx_eq(a: f32, b: f32) -> bool {
        let abs_tol = 1e-3_f32;
        let rel_tol = 2e-3_f32; // 2× f16 epsilon for rounding slack
        let scale = a.abs().max(b.abs()).max(1.0);
        (a - b).abs() <= abs_tol.max(rel_tol * scale)
    }

    #[test]
    fn append_get_roundtrip() {
        let config = make_config(2, 64);
        let kv_dim = config.kv_dim(); // 8 * 128 = 1024
        let mut cache = FlatKVCache::new(config);

        // Append one token to layer 0.
        // Use small values well within f16 range to avoid precision loss.
        let k: Vec<f32> = (0..kv_dim).map(|i| (i as f32) * 0.01).collect();
        let v: Vec<f32> = (0..kv_dim).map(|i| (i as f32) * 0.02 + 1.0).collect();
        cache.append_kv(0, &k, &v);

        // Also append to layer 1.
        let k1: Vec<f32> = (0..kv_dim).map(|i| (i as f32) * 0.03).collect();
        let v1: Vec<f32> = (0..kv_dim).map(|i| (i as f32) * 0.04 + 2.0).collect();
        cache.append_kv(1, &k1, &v1);

        cache.advance();
        assert_eq!(cache.seq_len(), 1);

        // Verify roundtrip with f16 tolerance.
        let got_k = cache.get_k(0);
        assert_eq!(got_k.len(), kv_dim);
        for (i, (&orig, &got)) in k.iter().zip(got_k.iter()).enumerate() {
            assert!(
                approx_eq(orig, got),
                "k[{i}]: expected {orig}, got {got}, diff {}",
                (orig - got).abs()
            );
        }

        let got_v = cache.get_v(0);
        for (i, (&orig, &got)) in v.iter().zip(got_v.iter()).enumerate() {
            assert!(approx_eq(orig, got), "v[{i}]: expected {orig}, got {got}");
        }

        let got_k1 = cache.get_k(1);
        for (i, (&orig, &got)) in k1.iter().zip(got_k1.iter()).enumerate() {
            assert!(approx_eq(orig, got), "k1[{i}]: expected {orig}, got {got}");
        }

        let got_v1 = cache.get_v(1);
        for (i, (&orig, &got)) in v1.iter().zip(got_v1.iter()).enumerate() {
            assert!(approx_eq(orig, got), "v1[{i}]: expected {orig}, got {got}");
        }
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

        // Each layer should have its own data (with f16 tolerance).
        for layer in 0..4 {
            let marker = (layer + 1) as f32;
            let got_k = cache.get_k(layer);
            assert!(
                got_k.iter().all(|&x| approx_eq(x, marker)),
                "layer {layer} K mismatch"
            );
            let got_v = cache.get_v(layer);
            assert!(
                got_v.iter().all(|&x| approx_eq(x, marker + 0.5)),
                "layer {layer} V mismatch"
            );
        }
    }

    #[test]
    fn reset_clears_state() {
        let config = make_config(2, 16);
        let kv_dim = config.kv_dim();
        let mut cache = FlatKVCache::new(config);

        let k = vec![1.0f32; kv_dim];
        let v = vec![2.0f32; kv_dim];
        cache.append_kv(0, &k, &v);
        cache.append_kv(1, &k, &v);
        cache.advance();
        assert_eq!(cache.seq_len(), 1);

        cache.reset();
        assert_eq!(cache.seq_len(), 0);
        // After reset, get_k/get_v return empty Vec.
        assert_eq!(cache.get_k(0).len(), 0);
        assert_eq!(cache.get_v(1).len(), 0);
    }

    #[test]
    fn capacity_limit() {
        let config = make_config(1, 4);
        let kv_dim = config.kv_dim();
        let mut cache = FlatKVCache::new(config);

        let k = vec![1.0f32; kv_dim];
        let v = vec![2.0f32; kv_dim];

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

        let k = vec![1.0f32; kv_dim];
        let v = vec![2.0f32; kv_dim];
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

        // Check the first decode token (at position 10), f16 roundtrip of 100.0 is exact.
        let decode_start = 10 * kv_dim;
        assert!(
            approx_eq(k0[decode_start], 100.0),
            "expected ~100.0, got {}",
            k0[decode_start]
        );

        // Check last decode token (at position 14).
        let last_start = 14 * kv_dim;
        assert!(
            approx_eq(k0[last_start], 104.0),
            "expected ~104.0, got {}",
            k0[last_start]
        );
    }

    #[test]
    fn memory_bytes_calculation() {
        // Qwen3-0.6B: 28 layers, 8 KV heads, head_dim=128, max 4096
        // kv_dim = 8 * 128 = 1024
        // Per side: 28 * 4096 * 1024 * 2 bytes (f16)
        // Total: 2 * that = 469,762,048 bytes ~ 448 MB (was 896 MB with f32)
        let config = FlatKVCacheConfig::for_qwen3(28, 8, 128, 4096);
        let kv_dim = 8 * 128; // 1024
        let expected = 2 * 28 * 4096 * kv_dim * 2; // 2 bytes per f16
        assert_eq!(config.total_bytes(), expected);
        let mb = config.total_bytes() as f64 / (1024.0 * 1024.0);
        assert!((mb - 448.0).abs() < 1.0, "expected ~448 MB, got {mb:.1} MB");
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
        // Valid slice (f16) shrinks accordingly
        assert_eq!(cache.get_k_f16(0).len(), 3 * kv_dim);
    }

    #[test]
    fn reset_fast_preserves_buffer() {
        let config = make_config(1, 8);
        let kv_dim = config.kv_dim();
        let mut cache = FlatKVCache::new(config);

        let k = vec![42.0f32; kv_dim];
        let v = vec![43.0f32; kv_dim];
        cache.append_kv(0, &k, &v);
        cache.advance();

        cache.reset_fast();
        assert_eq!(cache.seq_len(), 0);
        // Stale data is still there in the raw f16 buffer (this is expected).
        assert!(approx_eq(cache.k[0][0].to_f32(), 42.0));
    }

    #[test]
    fn get_k_mut_allows_modification() {
        let config = make_config(1, 8);
        let kv_dim = config.kv_dim();
        let mut cache = FlatKVCache::new(config);

        let k = vec![1.0f32; kv_dim];
        let v = vec![2.0f32; kv_dim];
        cache.append_kv(0, &k, &v);
        cache.advance();

        // Modify K in-place via f16 mutation.
        let k_mut = cache.get_k_mut(0);
        k_mut[0] = f16::from_f32(99.0);

        let got_k = cache.get_k(0);
        assert!(approx_eq(got_k[0], 99.0));
    }

    #[test]
    fn read_k_into_dequantizes_correctly() {
        let config = make_config(1, 8);
        let kv_dim = config.kv_dim();
        let mut cache = FlatKVCache::new(config);

        let k: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.5).collect();
        let v = vec![0.0f32; kv_dim];
        cache.append_kv(0, &k, &v);
        cache.advance();

        let mut buf = vec![0.0f32; kv_dim];
        cache.read_k_into(0, &mut buf);

        for (i, (&orig, &got)) in k.iter().zip(buf.iter()).enumerate() {
            assert!(
                approx_eq(orig, got),
                "read_k_into[{i}]: expected {orig}, got {got}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "layer index out of bounds")]
    fn read_k_into_out_of_range_layer_panics_clearly() {
        // Regression for #244: read accessors guarded `layer` with `debug_assert!`,
        // which compiled out in release and degraded to a bare `index out of bounds`
        // panic. Now a runtime `assert!` gives the same descriptive message in both
        // profiles, matching the write path (`append_kv` / `prefill_layer`).
        let config = make_config(2, 8);
        let kv_dim = config.kv_dim();
        let cache = FlatKVCache::new(config);
        let mut buf = vec![0.0f32; kv_dim];
        cache.read_k_into(2, &mut buf); // valid layers are 0..2
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn read_k_into_undersized_buffer_panics_clearly() {
        // Regression for #244: the `buf.len() >= end` precondition was a
        // `debug_assert!` and so unchecked in release. It is now a runtime
        // `assert!` with a descriptive message.
        let config = make_config(1, 8);
        let kv_dim = config.kv_dim();
        let mut cache = FlatKVCache::new(config);
        let k = vec![1.0f32; kv_dim];
        let v = vec![0.0f32; kv_dim];
        cache.append_kv(0, &k, &v);
        cache.advance();

        let mut too_small = vec![0.0f32; kv_dim - 1];
        cache.read_k_into(0, &mut too_small);
    }

    #[test]
    fn f16_storage_halves_memory() {
        // Verify the storage byte count matches f16 (not f32).
        let config = FlatKVCacheConfig {
            num_layers: 1,
            num_kv_heads: 2,
            head_dim: 4,
            max_seq_len: 16,
        };
        let kv_dim = 2 * 4; // 8
        // f16: 2 * 1 * 16 * 8 * 2 = 512 bytes
        let expected_f16 = 2 * 16 * kv_dim * std::mem::size_of::<f16>();
        assert_eq!(config.total_bytes(), expected_f16);
        // Would have been 1024 with f32
        let would_be_f32 = 2 * 16 * kv_dim * std::mem::size_of::<f32>();
        assert_eq!(config.total_bytes() * 2, would_be_f32);
    }

    // -----------------------------------------------------------------------
    // Boundary / special-value conversion tests (f32→f16→f32 contract).
    //
    // These tests assert the *chosen contract* for each boundary rather than
    // relying on roundtrip tolerance alone.  `f16::from_f32` follows IEEE-754:
    //
    //  * NaN input  → f16 NaN  → f32 NaN  (is_nan() propagates)
    //  * ±∞ input   → f16 ±∞  → f32 ±∞  (exact preservation)
    //  * +0.0 / -0.0 → f16 ±0 → f32 ±0  (sign bit preserved)
    //  * f16::MAX (65504.0) roundtrips exactly within f16 representable range
    //  * f32 values above f16 range (e.g. 1e38) overflow → f16::INFINITY
    //  * f32 subnormals flush to ±0 in f16 (magnitude below f16 MIN_POSITIVE)
    // -----------------------------------------------------------------------

    /// Helper: store a single f32 value in layer 0, advance, read back via get_k.
    fn roundtrip_single(val: f32) -> f32 {
        let config = FlatKVCacheConfig {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 1,
            max_seq_len: 4,
        };
        let mut cache = FlatKVCache::new(config);
        cache.append_kv(0, &[val], &[0.0f32]);
        cache.advance();
        cache.get_k(0)[0]
    }

    #[test]
    fn boundary_nan_propagates() {
        // Contract: NaN input → NaN stored and read back.
        let out = roundtrip_single(f32::NAN);
        assert!(out.is_nan(), "expected NaN, got {out}");
    }

    #[test]
    fn boundary_positive_infinity() {
        // Contract: +∞ input → +∞ stored and read back.
        let out = roundtrip_single(f32::INFINITY);
        assert!(
            out.is_infinite() && out.is_sign_positive(),
            "expected +∞, got {out}"
        );
    }

    #[test]
    fn boundary_negative_infinity() {
        // Contract: -∞ input → -∞ stored and read back.
        let out = roundtrip_single(f32::NEG_INFINITY);
        assert!(
            out.is_infinite() && out.is_sign_negative(),
            "expected -∞, got {out}"
        );
    }

    #[test]
    fn boundary_positive_zero() {
        // Contract: +0.0 roundtrips as zero (sign not guaranteed by f16 spec but
        // practically preserved; assert at minimum the value is zero).
        let out = roundtrip_single(0.0f32);
        assert_eq!(out, 0.0f32, "expected 0.0, got {out}");
    }

    #[test]
    fn boundary_negative_zero() {
        // Contract: -0.0 roundtrips as zero (sign preserved in IEEE-754 f16).
        let out = roundtrip_single(-0.0f32);
        assert_eq!(out, 0.0f32, "expected -0.0 (==0.0), got {out}");
        assert!(
            out.is_sign_negative(),
            "expected sign bit to be negative for -0.0"
        );
    }

    #[test]
    fn boundary_f16_max_exact() {
        // f16::MAX = 65504.0; this value is exactly representable in f16,
        // so the roundtrip must be exact.
        let f16_max = f16::MAX.to_f32(); // 65504.0
        let out = roundtrip_single(f16_max);
        assert_eq!(out, f16_max, "f16::MAX roundtrip must be exact, got {out}");
    }

    #[test]
    fn boundary_overflow_to_infinity() {
        // f32 values above f16 range (> 65504) overflow to ±∞ in f16.
        // Contract: store overflows to +∞ and is read back as +∞.
        let out = roundtrip_single(1e38_f32);
        assert!(
            out.is_infinite() && out.is_sign_positive(),
            "expected +∞ (f32→f16 overflow), got {out}"
        );
    }

    #[test]
    fn boundary_f32_subnormal_flushes() {
        // A real f32 subnormal (nonzero, below f32::MIN_POSITIVE) is too small
        // for f16 and flushes to ±0. Use from_bits(1) = smallest positive f32 subnormal.
        let tiny = f32::from_bits(1); // ~1.4e-45, a real subnormal
        assert!(
            tiny > 0.0 && tiny < f32::MIN_POSITIVE,
            "should be a real f32 subnormal"
        );
        let out = roundtrip_single(tiny);
        assert_eq!(
            out, 0.0f32,
            "f32 subnormal below f16 range should flush to 0, got {out}"
        );
    }

    #[test]
    fn boundary_f16_min_positive_subnormal_survives() {
        // The smallest nonzero f16 value should survive the roundtrip.
        let smallest_f16 = half::f16::MIN_POSITIVE_SUBNORMAL.to_f32();
        assert!(
            smallest_f16 > 0.0,
            "f16 MIN_POSITIVE_SUBNORMAL should be positive"
        );
        let out = roundtrip_single(smallest_f16);
        assert_eq!(
            out, smallest_f16,
            "f16 MIN_POSITIVE_SUBNORMAL should survive roundtrip, got {out}"
        );
    }

    // -- Exact bit-pattern overflow and RNE tests (Defect 3 fix) ---------

    #[test]
    fn boundary_requested_overflow_inputs_map_to_infinity() {
        // 65536.0 = 2^16 is above f16::MAX (65504.0) and must overflow to ±inf.
        // These are the *exact* boundary values requested in the hardening pass.
        assert_eq!(
            f16::from_f32(65536.0_f32).to_bits(),
            f16::INFINITY.to_bits(),
            "65536.0 must overflow to +inf in f16"
        );
        assert_eq!(
            f16::from_f32(-65536.0_f32).to_bits(),
            f16::NEG_INFINITY.to_bits(),
            "-65536.0 must overflow to -inf in f16"
        );
    }

    #[test]
    fn boundary_round_to_nearest_even_ties() {
        // Verify IEEE-754 round-to-nearest-even at the exact halfway points.
        //
        // Case 1: 2^-25 is below the smallest f16 subnormal (2^-24); rounds to 0.
        assert_eq!(
            f16::from_f32(2.0_f32.powi(-25)).to_bits(),
            0x0000,
            "2^-25 must round to 0 (below subnormal range)"
        );
        // Case 2: 1.5 * 2^-24 is the midpoint between subnormal bit=1 and bit=2.
        // RNE selects even mantissa → bit=2 → 0x0002.
        assert_eq!(
            f16::from_f32(1.5 * 2.0_f32.powi(-24)).to_bits(),
            0x0002,
            "1.5*2^-24 must round to subnormal bit=2 (RNE)"
        );
        // Case 3: 1.0 + 2^-11 is the midpoint between 0x3C00 (1.0) and 0x3C01.
        // 0x3C00 has even mantissa LSB; RNE rounds to 0x3C00.
        assert_eq!(
            f16::from_f32(1.0_f32 + 2.0_f32.powi(-11)).to_bits(),
            0x3c00,
            "1.0 + 2^-11 must round to 1.0 = 0x3C00 (RNE, even mantissa wins)"
        );
        // Case 4: 1.0 + 3*2^-11 is the midpoint between 0x3C01 and 0x3C02.
        // 0x3C02 has even mantissa LSB; RNE rounds to 0x3C02.
        assert_eq!(
            f16::from_f32(1.0_f32 + 3.0 * 2.0_f32.powi(-11)).to_bits(),
            0x3c02,
            "1.0 + 3*2^-11 must round to 0x3C02 (RNE, even mantissa wins)"
        );
    }

    // -- Allocation-contract test (Defect 2 fix, Option B) ---------------

    #[test]
    fn constructed_cache_eagerly_initializes_configured_len() {
        // Verifies that FlatKVCache::new immediately materializes all buffers
        // at max_seq_len capacity (eager-allocation contract; issue #12 resolved by
        // capping max_seq_len via GenerateConfig::kv_cache_capacity in generate()).
        let config = FlatKVCacheConfig::for_qwen3(2, 2, 4, 16);
        let cache = FlatKVCache::new(config);
        let kv_dim = cache.kv_dim(); // 2 * 4 = 8
        let max_seq = cache.max_seq_len(); // 16
        // num_layers * (K-side + V-side) * max_seq * kv_dim
        // = 2 * 2 * 16 * 8 = 512
        let expected_elems = cache.num_layers() * 2 * max_seq * kv_dim;
        let actual_elems: usize = (0..cache.num_layers())
            .map(|layer| cache.k_buffer(layer).len() + cache.v_buffer(layer).len())
            .sum();
        assert_eq!(
            actual_elems, expected_elems,
            "FlatKVCache eagerly allocates max_seq_len * kv_dim per layer per side"
        );
    }

    /// Quality measurement: f32→f16→f32 roundtrip error over representative KV value ranges.
    ///
    /// Reports max absolute error, max relative error, and confirms they are within
    /// the bounds implied by IEEE-754 binary16 (f16 epsilon = 2^-10 ≈ 9.77e-4).
    #[test]
    fn quality_measurement_f16_roundtrip_error() {
        let n = 200_000usize;

        // Ranges representing typical KV values in transformer models
        let ranges: &[(&str, f32, f32)] = &[
            ("tiny [-0.1, 0.1]", -0.1, 0.1),
            ("small [-1.0, 1.0]", -1.0, 1.0),
            ("typical [-5.0, 5.0]", -5.0, 5.0),
            ("large [-10.0, 10.0]", -10.0, 10.0),
        ];

        // f16::MIN_POSITIVE is the smallest *normal* f16 value (~6.1e-5).
        // Values below this may flush to a subnormal or zero — relative error near zero
        // is by-design large (subnormal flush is expected behavior). We check only
        // normal f16 values for the relative-error invariant.
        let f16_min_normal = f16::MIN_POSITIVE.to_f32();

        for (label, lo, hi) in ranges {
            let mut max_abs = 0.0f32;
            let mut max_rel_normal = 0.0f32;
            for i in 0..n {
                let t = i as f32 / (n - 1) as f32;
                let v = lo + t * (hi - lo);
                let v_h = f16::from_f32(v).to_f32();
                let abs_err = (v - v_h).abs();
                if abs_err > max_abs {
                    max_abs = abs_err;
                }
                // Relative error only meaningful for normal f16 inputs
                if v.abs() >= f16_min_normal {
                    let rel_err = abs_err / v.abs();
                    if rel_err > max_rel_normal {
                        max_rel_normal = rel_err;
                    }
                }
            }
            // Max relative error for normal inputs must be <= 0.5 * f16::EPSILON
            let half_eps = f16::EPSILON.to_f32() / 2.0;
            assert!(
                max_rel_normal <= half_eps + 1e-7,
                "{label}: max relative error (normal inputs) {max_rel_normal:.2e} exceeds 0.5*ε ({half_eps:.2e})"
            );
        }

        // Confirm the specific "< 0.1% relative error" claim for [-10, 10]
        let mut max_rel_kv = 0.0f32;
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            let v = -10.0 + t * 20.0_f32;
            let v_h = f16::from_f32(v).to_f32();
            let rel_err = if v.abs() > 1e-10 {
                (v - v_h).abs() / v.abs()
            } else {
                (v - v_h).abs()
            };
            if rel_err > max_rel_kv {
                max_rel_kv = rel_err;
            }
        }
        // Measured relative error must be < 0.1% (0.001)
        assert!(
            max_rel_kv < 0.001,
            "max relative error for KV in [-10,10] is {max_rel_kv:.4e}, expected < 0.001"
        );
    }

    // -----------------------------------------------------------------------
    // Tensor oracle: f32-KV reference vs f16-KV-via-FlatKVCache logit diff.
    //
    // Implements the mandatory fallback harness from quality_measurement_design.md:
    //   - Builds deterministic Q/K/V tensors (xorshift32 PRNG, no model weights).
    //   - Runs scaled dot-product attention with f32 KV directly.
    //   - Stores same K/V in FlatKVCache (quantizes to f16), dequantizes via
    //     the scratch-loop used by generate.rs, runs same attention.
    //   - Projects both outputs through a deterministic W_out to produce logits.
    //   - Asserts: logit_max_abs_diff < 0.02, top1_match_rate >= 0.95,
    //              nan_count == 0, synthetic_nll_delta_abs < 0.01.
    //
    // This is NOT a PPL measurement. It is the CI tensor oracle only.
    // -----------------------------------------------------------------------

    /// Deterministic xorshift32 PRNG (Marsaglia 2003).
    fn xorshift32(state: &mut u32) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        x
    }

    /// Sample a f32 uniformly in [lo, hi] using xorshift32.
    fn rand_f32(state: &mut u32, lo: f32, hi: f32) -> f32 {
        let bits = xorshift32(state);
        let t = (bits as f32) / (u32::MAX as f32);
        lo + t * (hi - lo)
    }

    /// Scaled dot-product attention (q_seq_len=1 decode, GQA).
    ///
    /// Q:      [num_heads * head_dim]            (single query token)
    /// K, V:   [kv_seq_len * num_kv_heads * head_dim]
    /// output: [num_heads * head_dim]
    ///
    /// Groups = num_heads / num_kv_heads; each KV head is shared across `groups` Q heads.
    fn sdpa_decode(
        output: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        kv_seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        let groups = num_heads / num_kv_heads;
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_heads * head_dim;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        output[..q_dim].fill(0.0);
        let mut scores = vec![0.0f32; kv_seq_len];

        for h in 0..num_heads {
            let kv_h = h / groups;
            let q_off = h * head_dim;

            // Phase 1: QK^T
            for ki in 0..kv_seq_len {
                let k_off = ki * kv_dim + kv_h * head_dim;
                let dot: f32 = (0..head_dim).map(|d| q[q_off + d] * k[k_off + d]).sum();
                scores[ki] = dot * scale;
            }

            // Phase 2: stable softmax
            let max_s = scores[..kv_seq_len]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores[..kv_seq_len]
                .iter_mut()
                .map(|s| {
                    *s = (*s - max_s).exp();
                    *s
                })
                .sum();
            if sum > 0.0 {
                scores[..kv_seq_len].iter_mut().for_each(|s| *s /= sum);
            }

            // Phase 3: weighted V sum
            let out_off = h * head_dim;
            for ki in 0..kv_seq_len {
                let v_off = ki * kv_dim + kv_h * head_dim;
                let w = scores[ki];
                for d in 0..head_dim {
                    output[out_off + d] += w * v[v_off + d];
                }
            }
        }
    }

    /// The mandatory tensor oracle: compare f32-KV reference vs f16-KV-via-FlatKVCache.
    ///
    /// Produces actual measured logit_max_abs_diff and top1_match_rate.
    #[test]
    fn f16_kv_tensor_oracle_logit_diff() {
        // Oracle parameters (matching design doc).
        const NUM_HEADS: usize = 4;
        const NUM_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 16;
        const VOCAB: usize = 257;
        const Q_DIM: usize = NUM_HEADS * HEAD_DIM; // 64
        const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 32

        // KV context lengths and value ranges per design doc.
        let seq_lens: &[usize] = &[1, 8, 64, 256];
        let kv_ranges: &[(&str, f32, f32)] = &[
            ("tiny", -0.1, 0.1),
            ("typical", -5.0, 5.0),
            ("outlier", -10.0, 10.0),
        ];

        // Deterministic W_out: [VOCAB, Q_DIM], scaled to [-0.05, 0.05].
        let mut w_seed: u32 = 0xDEAD_BEEF;
        let w_out: Vec<f32> = (0..VOCAB * Q_DIM)
            .map(|_| rand_f32(&mut w_seed, -0.05, 0.05))
            .collect();

        let mut global_max_logit_diff = 0.0f32;
        let mut top1_match_count = 0usize;
        let mut total_cases = 0usize;
        let mut nan_count = 0usize;
        let mut max_synth_nll_delta = 0.0f32;

        for &seq_len in seq_lens {
            for &(range_name, lo, hi) in kv_ranges {
                let mut seed: u32 = 0x1234_5678u32
                    .wrapping_add(seq_len as u32)
                    .wrapping_mul(0x9E37_79B9)
                    .wrapping_add(range_name.len() as u32);

                // Build deterministic Q (seq_len=1 decode query).
                let q: Vec<f32> = (0..Q_DIM).map(|_| rand_f32(&mut seed, -1.0, 1.0)).collect();

                // Build deterministic K_f32, V_f32.
                let k_f32: Vec<f32> = (0..seq_len * KV_DIM)
                    .map(|_| rand_f32(&mut seed, lo, hi))
                    .collect();
                let v_f32: Vec<f32> = (0..seq_len * KV_DIM)
                    .map(|_| rand_f32(&mut seed, lo, hi))
                    .collect();

                // ---- Reference path: f32 KV directly ----
                let mut out_f32 = vec![0.0f32; Q_DIM];
                sdpa_decode(
                    &mut out_f32,
                    &q,
                    &k_f32,
                    &v_f32,
                    seq_len,
                    NUM_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                );

                // ---- f16 path: store in FlatKVCache, dequantize via k_buffer/v_buffer ----
                let cfg = FlatKVCacheConfig {
                    num_layers: 1,
                    num_kv_heads: NUM_KV_HEADS,
                    head_dim: HEAD_DIM,
                    max_seq_len: seq_len,
                };
                let mut cache = FlatKVCache::new(cfg);

                // Write K/V row by row (matching generate.rs prefill pattern).
                {
                    let k_layer = cache.k_buffer_mut(0);
                    for (i, &val) in k_f32.iter().enumerate() {
                        k_layer[i] = f16::from_f32(val);
                    }
                    let v_layer = cache.v_buffer_mut(0);
                    for (i, &val) in v_f32.iter().enumerate() {
                        v_layer[i] = f16::from_f32(val);
                    }
                }
                cache.advance_by(seq_len);

                // Dequantize via the same scratch-loop pattern as generate.rs:463-466.
                let k_end = seq_len * KV_DIM;
                let mut k_dequant = vec![0.0f32; k_end];
                let mut v_dequant = vec![0.0f32; k_end];
                for (i, &h) in cache.k_buffer(0)[..k_end].iter().enumerate() {
                    k_dequant[i] = h.to_f32();
                }
                for (i, &h) in cache.v_buffer(0)[..k_end].iter().enumerate() {
                    v_dequant[i] = h.to_f32();
                }

                let mut out_f16 = vec![0.0f32; Q_DIM];
                sdpa_decode(
                    &mut out_f16,
                    &q,
                    &k_dequant,
                    &v_dequant,
                    seq_len,
                    NUM_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                );

                // ---- Project both outputs to logits via W_out ----
                let mut logits_f32 = vec![0.0f32; VOCAB];
                let mut logits_f16 = vec![0.0f32; VOCAB];
                for v_idx in 0..VOCAB {
                    let row_off = v_idx * Q_DIM;
                    logits_f32[v_idx] = (0..Q_DIM)
                        .map(|d| out_f32[d] * w_out[row_off + d])
                        .sum::<f32>();
                    logits_f16[v_idx] = (0..Q_DIM)
                        .map(|d| out_f16[d] * w_out[row_off + d])
                        .sum::<f32>();
                }

                // ---- Measure diff ----
                let case_max_diff = logits_f32
                    .iter()
                    .zip(logits_f16.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);

                let nans = logits_f16.iter().filter(|&&x| x.is_nan()).count();
                nan_count += nans;

                let top1_f32 = logits_f32
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                let top1_f16 = logits_f16
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();

                // Synthetic NLL delta on deterministic target token.
                let target = (seq_len * 37 + 11) % VOCAB;
                let nll_f32 = -softmax_log_prob(&logits_f32, target);
                let nll_f16 = -softmax_log_prob(&logits_f16, target);
                let nll_delta = (nll_f16 - nll_f32).abs();

                if case_max_diff > global_max_logit_diff {
                    global_max_logit_diff = case_max_diff;
                }
                if top1_f32 == top1_f16 {
                    top1_match_count += 1;
                }
                if nll_delta > max_synth_nll_delta {
                    max_synth_nll_delta = nll_delta;
                }
                total_cases += 1;

                eprintln!(
                    "  oracle seq={:3} range={:<8} logit_max_diff={:.2e}  top1={}  nll_delta={:.2e}",
                    seq_len,
                    range_name,
                    case_max_diff,
                    if top1_f32 == top1_f16 {
                        "MATCH"
                    } else {
                        "DIFF"
                    },
                    nll_delta
                );
            }
        }

        let top1_rate = top1_match_count as f32 / total_cases as f32;
        eprintln!(
            "\n=== Tensor Oracle Summary ===\n  logit_max_abs_diff = {global_max_logit_diff:.4e}  (gate: < 0.02)\n  top1_match_rate    = {top1_rate:.4}    (gate: >= 0.95)\n  nan_count          = {nan_count}\n  max_synth_nll_delta= {max_synth_nll_delta:.4e}  (gate: < 0.01)"
        );

        assert_eq!(nan_count, 0, "f16 KV dequant introduced NaN in logits");
        assert!(
            global_max_logit_diff < 0.02,
            "logit_max_abs_diff {global_max_logit_diff:.4e} >= 0.02 gate"
        );
        assert!(
            top1_rate >= 0.95,
            "top1_match_rate {top1_rate:.4} < 0.95 gate"
        );
        assert!(
            max_synth_nll_delta < 0.01,
            "max synthetic NLL delta {max_synth_nll_delta:.4e} >= 0.01"
        );
    }

    /// Compute log softmax probability for target token (for synthetic NLL).
    fn softmax_log_prob(logits: &[f32], target: usize) -> f32 {
        let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
        let log_sum = sum.ln();
        (logits[target] - max_l) - log_sum
    }
}
