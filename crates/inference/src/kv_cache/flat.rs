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
        debug_assert!(layer < self.config.num_layers);
        let end = self.seq_len * self.config.kv_dim();
        debug_assert!(buf.len() >= end);
        for (i, &h) in self.k[layer][..end].iter().enumerate() {
            buf[i] = h.to_f32();
        }
    }

    /// **Unstable**: dequantize V slice for a layer into `buf` up to current seq_len.
    ///
    /// `buf` must have length >= `seq_len * kv_dim`. Converts f16→f32.
    pub fn read_v_into(&self, layer: usize, buf: &mut [f32]) {
        debug_assert!(layer < self.config.num_layers);
        let end = self.seq_len * self.config.kv_dim();
        debug_assert!(buf.len() >= end);
        for (i, &h) in self.v[layer][..end].iter().enumerate() {
            buf[i] = h.to_f32();
        }
    }

    /// **Unstable**: read K slice for a layer up to current seq_len (f16).
    ///
    /// Use `read_k_into` for f32 dequantized access.
    #[inline]
    pub fn get_k_f16(&self, layer: usize) -> &[f16] {
        debug_assert!(layer < self.config.num_layers);
        let end = self.seq_len * self.config.kv_dim();
        &self.k[layer][..end]
    }

    /// **Unstable**: read V slice for a layer up to current seq_len (f16).
    ///
    /// Use `read_v_into` for f32 dequantized access.
    #[inline]
    pub fn get_v_f16(&self, layer: usize) -> &[f16] {
        debug_assert!(layer < self.config.num_layers);
        let end = self.seq_len * self.config.kv_dim();
        &self.v[layer][..end]
    }

    /// **Unstable**: read K slice for a layer up to current seq_len, dequantizing to f32.
    ///
    /// Allocates a Vec; prefer `read_k_into` when a pre-allocated buffer is available.
    #[inline]
    pub fn get_k(&self, layer: usize) -> Vec<f32> {
        debug_assert!(layer < self.config.num_layers);
        let end = self.seq_len * self.config.kv_dim();
        self.k[layer][..end].iter().map(|h| h.to_f32()).collect()
    }

    /// **Unstable**: read V slice for a layer up to current seq_len, dequantizing to f32.
    ///
    /// Allocates a Vec; prefer `read_v_into` when a pre-allocated buffer is available.
    #[inline]
    pub fn get_v(&self, layer: usize) -> Vec<f32> {
        debug_assert!(layer < self.config.num_layers);
        let end = self.seq_len * self.config.kv_dim();
        self.v[layer][..end].iter().map(|h| h.to_f32()).collect()
    }

    /// **Unstable**: mutable K slice for in-place f16 operations (e.g. RoPE applied in f16).
    #[inline]
    pub fn get_k_mut(&mut self, layer: usize) -> &mut [f16] {
        debug_assert!(layer < self.config.num_layers);
        let end = self.seq_len * self.config.kv_dim();
        &mut self.k[layer][..end]
    }

    /// **Unstable**: mutable V slice for in-place f16 operations.
    #[inline]
    pub fn get_v_mut(&mut self, layer: usize) -> &mut [f16] {
        debug_assert!(layer < self.config.num_layers);
        let end = self.seq_len * self.config.kv_dim();
        &mut self.v[layer][..end]
    }

    /// **Unstable**: full K buffer including unwritten positions (f16); used by prefill.
    #[inline]
    pub fn k_buffer_mut(&mut self, layer: usize) -> &mut [f16] {
        debug_assert!(layer < self.config.num_layers);
        &mut self.k[layer]
    }

    /// **Unstable**: immutable full K buffer (f16).
    #[inline]
    pub fn k_buffer(&self, layer: usize) -> &[f16] {
        debug_assert!(layer < self.config.num_layers);
        &self.k[layer]
    }

    /// **Unstable**: full V buffer including unwritten positions (f16).
    #[inline]
    pub fn v_buffer_mut(&mut self, layer: usize) -> &mut [f16] {
        debug_assert!(layer < self.config.num_layers);
        &mut self.v[layer]
    }

    /// **Unstable**: immutable full V buffer (f16).
    #[inline]
    pub fn v_buffer(&self, layer: usize) -> &[f16] {
        debug_assert!(layer < self.config.num_layers);
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
        let expected_f16 = 2 * 1 * 16 * kv_dim * std::mem::size_of::<f16>();
        assert_eq!(config.total_bytes(), expected_f16);
        // Would have been 1024 with f32
        let would_be_f32 = 2 * 1 * 16 * kv_dim * std::mem::size_of::<f32>();
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
        // f32 subnormals with magnitude below f16 MIN_POSITIVE (~6e-8) flush to
        // ±0 when converted to f16.  Contract: result is zero (positive or negative).
        let tiny = f32::MIN_POSITIVE * 1e-10; // well below f16 representable range
        let out = roundtrip_single(tiny);
        assert_eq!(
            out, 0.0f32,
            "f32 subnormal below f16 range should flush to 0, got {out}"
        );
    }
}
