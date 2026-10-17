use super::*;

fn make_prefix_cache(capacity: usize) -> Arc<Mutex<PrefixPageCache>> {
    Arc::new(Mutex::new(PrefixPageCache::new(PrefixPageCacheConfig {
        capacity,
        prefix_page_size: 4,
        num_layers: 2,
        num_kv_heads: 2,
        head_dim: 4,
    })))
}

#[test]
fn test_prefix_cache_miss_fallthrough() {
    let config = make_config(4);
    let kv_dim = config.kv_dim();
    let prefix_cache = make_prefix_cache(4);
    let mut cache = PagedKVCache::with_prefix_cache(config, Some(prefix_cache));

    let restored = cache
        .restore_prefix(AdapterId::BASE, &[1, 2, 3])
        .expect("restore miss should not fail");
    assert_eq!(restored, None);

    let k = vec![1.0; kv_dim];
    let v = vec![2.0; kv_dim];
    for layer in 0..2 {
        cache.append_kv_layer(layer, &k, &v);
    }
    cache.advance();

    assert_eq!(cache.seq_len(), 1);
    assert_eq!(cache.num_pages(), 1);
}

#[test]
fn restore_prefix_rejects_append_before_advance() {
    let config = make_config(4);
    let kv_dim = config.kv_dim();
    let mut cache = PagedKVCache::new(config);
    cache.append_kv_layer(0, &vec![1.0; kv_dim], &vec![2.0; kv_dim]);

    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.num_pages(), 1);
    let err = cache
        .restore_prefix(AdapterId::BASE, &[1, 2, 3])
        .expect_err("partially initialized cache must be rejected");
    assert!(
        matches!(err, InferenceError::PrefixCache(_)),
        "expected PrefixCache error, got {err:?}"
    );
}

#[test]
fn test_restore_prefix_hit_fast_forwards_seq_len() {
    let config = make_config(4);
    let kv_dim = config.kv_dim();
    let prefix_cache = make_prefix_cache(4);
    let tokens: [u32; 3] = [1, 2, 3];

    let mut source =
        PagedKVCache::with_prefix_cache(config.clone(), Some(Arc::clone(&prefix_cache)));
    for step in 0..tokens.len() {
        for layer in 0..2 {
            let marker = (step * 10 + layer) as f32;
            let k = vec![marker; kv_dim];
            let v = vec![marker + 0.5; kv_dim];
            source.append_kv_layer(layer, &k, &v);
        }
        source.advance();
    }
    assert_eq!(
        source
            .promote_to_prefix(AdapterId::BASE, &tokens)
            .expect("promotion should succeed"),
        Some(1)
    );

    let mut restored = PagedKVCache::with_prefix_cache(config, Some(prefix_cache));
    assert_eq!(
        restored
            .restore_prefix(AdapterId::BASE, &tokens)
            .expect("restore should succeed"),
        Some(tokens.len())
    );
    assert_eq!(restored.seq_len(), tokens.len());

    let mut k_buf = vec![0.0f32; tokens.len() * kv_dim];
    restored.gather_k(0, &mut k_buf);
    assert_eq!(k_buf[0], 0.0);
    assert_eq!(k_buf[kv_dim], 10.0);
    assert_eq!(k_buf[2 * kv_dim], 20.0);
}

#[test]
fn test_restore_prefix_with_different_page_sizes() {
    // page_size=8 (live), prefix_page_size=2 — forces copy_token_between_page_layouts
    // to run with different src/dst strides: the production case where prefix pages
    // are finer-grained than live pages.
    let config = PagedKVCacheConfig {
        page_size: 8,
        max_pages: 4,
        num_layers: 2,
        num_kv_heads: 2,
        head_dim: 4,
        eviction: EvictionPolicy::None,
    };
    let kv_dim = config.kv_dim(); // 8
    let prefix_cache = Arc::new(Mutex::new(PrefixPageCache::new(PrefixPageCacheConfig {
        capacity: 4,
        prefix_page_size: 2, // Different from page_size=8.
        num_layers: 2,
        num_kv_heads: 2,
        head_dim: 4,
    })));
    let tokens: [u32; 5] = [10, 20, 30, 40, 50];

    // Append 5 tokens with deterministic K/V values keyed by step and layer.
    let mut source =
        PagedKVCache::with_prefix_cache(config.clone(), Some(Arc::clone(&prefix_cache)));
    for (step, _) in tokens.iter().enumerate() {
        for layer in 0..2 {
            let k_val = (step * 100 + layer * 10) as f32;
            let k = vec![k_val; kv_dim];
            let v = vec![k_val + 0.5; kv_dim];
            source.append_kv_layer(layer, &k, &v);
        }
        source.advance();
    }
    assert_eq!(source.seq_len(), 5);

    // 5 tokens / prefix_page_size=2 → ceil(5/2) = 3 prefix pages.
    let page_count = source
        .promote_to_prefix(AdapterId::BASE, &tokens)
        .expect("promote should succeed")
        .expect("promote should insert pages");
    assert_eq!(page_count, 3);

    // Restore into fresh cache with page_size=8 (all 5 tokens in 1 live page).
    let mut restored = PagedKVCache::with_prefix_cache(config.clone(), Some(prefix_cache));
    let prefix_len = restored
        .restore_prefix(AdapterId::BASE, &tokens)
        .expect("restore should succeed")
        .expect("restore should hit");
    assert_eq!(prefix_len, 5);
    assert_eq!(restored.seq_len(), 5);

    // Verify K and V for all tokens, all layers — exact float equality since these
    // are copies, not computations.
    for layer in 0..2 {
        let mut k_buf = vec![0.0f32; tokens.len() * kv_dim];
        let mut v_buf = vec![0.0f32; tokens.len() * kv_dim];
        restored.gather_k(layer, &mut k_buf);
        restored.gather_v(layer, &mut v_buf);

        for (step, _) in tokens.iter().enumerate() {
            let k_expected = (step * 100 + layer * 10) as f32;
            let v_expected = k_expected + 0.5;
            for i in 0..kv_dim {
                assert_eq!(
                    k_buf[step * kv_dim + i],
                    k_expected,
                    "K mismatch at step={step}, layer={layer}, i={i}"
                );
                assert_eq!(
                    v_buf[step * kv_dim + i],
                    v_expected,
                    "V mismatch at step={step}, layer={layer}, i={i}"
                );
            }
        }
    }
}

fn make_config(max_pages: usize) -> PagedKVCacheConfig {
    PagedKVCacheConfig {
        page_size: 4, // Small pages for testing.
        max_pages,
        num_layers: 2,
        num_kv_heads: 2,
        head_dim: 4,
        eviction: EvictionPolicy::None,
    }
}

#[test]
fn paged_append_gather_roundtrip() {
    let config = make_config(4);
    let kv_dim = config.kv_dim(); // 2 * 4 = 8
    let mut cache = PagedKVCache::new(config);
    assert_eq!(cache.cache_type(), CacheType::F32);

    // Append 3 tokens.
    for step in 0..3u32 {
        for layer in 0..2 {
            let marker = (step * 10 + layer as u32) as f32;
            let k = vec![marker; kv_dim];
            let v = vec![marker + 0.5; kv_dim];
            cache.append_kv_layer(layer, &k, &v);
        }
        cache.advance();
    }
    assert_eq!(cache.seq_len(), 3);

    // Gather and verify.
    let mut k_buf = vec![0.0f32; 3 * kv_dim];
    let mut v_buf = vec![0.0f32; 3 * kv_dim];

    cache.gather_k(0, &mut k_buf);
    cache.gather_v(0, &mut v_buf);

    // Token 0, layer 0: marker = 0.0
    assert_eq!(k_buf[0], 0.0);
    assert_eq!(v_buf[0], 0.5);
    // Token 1, layer 0: marker = 10.0
    assert_eq!(k_buf[kv_dim], 10.0);
    assert_eq!(v_buf[kv_dim], 10.5);
    // Token 2, layer 0: marker = 20.0
    assert_eq!(k_buf[2 * kv_dim], 20.0);
    assert_eq!(v_buf[2 * kv_dim], 20.5);

    // Check layer 1.
    cache.gather_k(1, &mut k_buf);
    cache.gather_v(1, &mut v_buf);
    assert_eq!(k_buf[0], 1.0); // marker = 0*10 + 1 = 1
    assert_eq!(v_buf[0], 1.5);
}

#[test]
fn paged_q8_append_gather_roundtrip() {
    let config = make_config(4);
    let kv_dim = config.kv_dim();
    let values_per_page = config.num_layers * 2 * config.page_size * kv_dim;
    let scales_per_page = config.num_layers * 2 * config.page_size;
    let expected_bytes_per_page = values_per_page + scales_per_page * size_of::<f32>();
    let expected_total_bytes = config.max_pages * expected_bytes_per_page;
    assert!(expected_total_bytes < config.total_bytes());
    let mut cache = PagedKVCache::try_with_cache_type(config, CacheType::Q8)
        .expect("valid Q8 config must succeed");
    let mut expected_k = vec![Vec::new(); 2];
    let mut expected_v = vec![Vec::new(); 2];

    for step in 0..6 {
        for layer in 0..2 {
            let k: Vec<f32> = (0..kv_dim)
                .map(|i| {
                    let magnitude =
                        0.37 * (step + 1) as f32 + 0.11 * (layer + 1) as f32 + 0.073 * i as f32;
                    if (step + layer + i) % 2 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    }
                })
                .collect();
            let v: Vec<f32> = k.iter().map(|value| value * -0.61 + 0.19).collect();
            expected_k[layer].extend_from_slice(&k);
            expected_v[layer].extend_from_slice(&v);
            cache.append_kv_layer(layer, &k, &v);
        }
        cache.advance();
    }

    assert_eq!(cache.cache_type(), CacheType::Q8);
    assert_eq!(cache.total_memory_bytes(), expected_total_bytes);
    assert_eq!(cache.num_pages(), 2);
    assert_eq!(cache.used_memory_bytes(), 2 * expected_bytes_per_page);

    let mut observed_quantization_error = false;
    for layer in 0..2 {
        let mut gathered_k = vec![0.0; 6 * kv_dim];
        let mut gathered_v = vec![0.0; 6 * kv_dim];
        cache.gather_k(layer, &mut gathered_k);
        cache.gather_v(layer, &mut gathered_v);

        for (expected, actual) in [
            (&expected_k[layer], &gathered_k),
            (&expected_v[layer], &gathered_v),
        ] {
            for (expected_token, actual_token) in expected
                .chunks_exact(kv_dim)
                .zip(actual.chunks_exact(kv_dim))
            {
                let abs_max = expected_token
                    .iter()
                    .fold(0.0f32, |max, value| max.max(value.abs()));
                let tolerance = abs_max / 254.0 + f32::EPSILON * abs_max;
                for (&expected_value, &actual_value) in expected_token.iter().zip(actual_token) {
                    let error = (expected_value - actual_value).abs();
                    assert!(
                        error <= tolerance,
                        "Q8 round-trip error {error} exceeds {tolerance}"
                    );
                    observed_quantization_error |= error > f32::EPSILON;
                }
            }
        }
    }
    assert!(observed_quantization_error);
}

#[test]
fn paged_q8_preserves_subnormal_vectors() {
    // `abs_max / 127.0` underflows to zero below roughly 1.8e-43, which
    // made the store divide by zero and read the whole vector back as zero.
    // These are valid finite inputs the Q8 contract accepts, so a nonzero
    // input must come back nonzero.
    let config = make_config(2);
    let kv_dim = config.kv_dim();
    let mut cache = PagedKVCache::try_with_cache_type(config, CacheType::Q8)
        .expect("valid Q8 config must succeed");

    let smallest = f32::from_bits(1);
    let mut k = vec![0.0f32; kv_dim];
    k[0] = smallest;
    k[1] = -smallest;
    let v: Vec<f32> = k.iter().map(|value| value * 4.0).collect();
    assert!(k.iter().all(|value| value.is_finite()));
    assert!(k[0] > 0.0 && k[1] < 0.0);

    cache.append_kv_layer(0, &k, &v);
    cache.append_kv_layer(1, &k, &v);
    cache.advance();

    let mut gathered_k = vec![0.0; kv_dim];
    let mut gathered_v = vec![0.0; kv_dim];
    cache.gather_k(0, &mut gathered_k);
    cache.gather_v(0, &mut gathered_v);

    assert!(
        gathered_k[0] > 0.0,
        "subnormal K vanished: {:?}",
        &gathered_k[..2]
    );
    assert!(
        gathered_k[1] < 0.0,
        "subnormal K sign lost: {:?}",
        &gathered_k[..2]
    );
    assert!(
        gathered_v[0] > 0.0,
        "subnormal V vanished: {:?}",
        &gathered_v[..2]
    );
    assert!(gathered_k.iter().all(|value| value.is_finite()));
    // The fallback scale is abs_max itself, so the surviving codes are
    // {-1, 0, 1} and the error stays within half a step.
    for (expected, actual) in k.iter().zip(&gathered_k) {
        assert!((expected - actual).abs() <= smallest);
    }
}

#[test]
fn paged_q8_try_rejects_overflowing_page_bytes() {
    // Zero pages makes both backing allocations empty, so neither
    // allocation check constrains the per-page geometry, while
    // `bytes_per_page` still evaluates it. The fallible constructor
    // promises InvalidInput rather than overflowing arithmetic, and the
    // F32 path already delivers that through `try_total_bytes`.
    let config = PagedKVCacheConfig {
        page_size: 1,
        max_pages: 0,
        num_layers: 1,
        num_kv_heads: (usize::MAX - 1) / 2,
        head_dim: 1,
        eviction: EvictionPolicy::None,
    };
    assert_eq!(config.try_floats_per_page().expect("fits"), usize::MAX - 1);

    let err = PagedKVCache::try_with_cache_type(config.clone(), CacheType::Q8)
        .expect_err("overflowing per-page byte count must be rejected");
    assert!(
        matches!(err, InferenceError::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
    // The F32 branch is the behaviour being matched, not a second bug.
    assert!(matches!(
        PagedKVCache::try_with_cache_type(config, CacheType::F32),
        Err(InferenceError::InvalidInput(_))
    ));
}

#[test]
#[should_panic(expected = "Q8 KV input must be finite")]
fn paged_q8_rejects_non_finite_input() {
    let config = make_config(2);
    let kv_dim = config.kv_dim();
    let mut cache = PagedKVCache::try_with_cache_type(config, CacheType::Q8)
        .expect("valid Q8 config must succeed");
    let mut k = vec![0.5f32; kv_dim];
    k[kv_dim - 1] = f32::NAN;
    cache.append_kv_layer(0, &k, &vec![0.5; kv_dim]);
}

#[test]
fn paged_page_allocation() {
    let config = make_config(8);
    let kv_dim = config.kv_dim();
    let mut cache = PagedKVCache::new(config);

    // page_size=4, so first page covers tokens 0-3.
    assert_eq!(cache.num_pages(), 0);
    assert_eq!(cache.free_pages(), 8);

    // Append 1 token -> allocates 1 page.
    let k = vec![1.0; kv_dim];
    let v = vec![2.0; kv_dim];
    for layer in 0..2 {
        cache.append_kv_layer(layer, &k, &v);
    }
    cache.advance();
    assert_eq!(cache.num_pages(), 1);
    assert_eq!(cache.free_pages(), 7);

    // Append 3 more -> still 1 page.
    for _ in 0..3 {
        for layer in 0..2 {
            cache.append_kv_layer(layer, &k, &v);
        }
        cache.advance();
    }
    assert_eq!(cache.num_pages(), 1);
    assert_eq!(cache.seq_len(), 4);

    // Append 1 more -> needs page 2.
    for layer in 0..2 {
        cache.append_kv_layer(layer, &k, &v);
    }
    cache.advance();
    assert_eq!(cache.num_pages(), 2);
    assert_eq!(cache.free_pages(), 6);
}

#[test]
fn paged_reset() {
    let config = make_config(4);
    let kv_dim = config.kv_dim();
    let mut cache = PagedKVCache::new(config);

    let k = vec![1.0; kv_dim];
    let v = vec![2.0; kv_dim];
    for _ in 0..6 {
        for layer in 0..2 {
            cache.append_kv_layer(layer, &k, &v);
        }
        cache.advance();
    }
    assert!(cache.num_pages() > 0);

    cache.reset();
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.num_pages(), 0);
    assert_eq!(cache.free_pages(), 4);
}

#[test]
fn paged_cross_page_boundary() {
    // Verify data integrity when tokens span multiple pages.
    let config = make_config(4);
    let kv_dim = config.kv_dim(); // 8
    let mut cache = PagedKVCache::new(config);

    // Append 6 tokens (spans 2 pages at page_size=4).
    for step in 0..6u32 {
        for layer in 0..2 {
            let k: Vec<f32> = (0..kv_dim)
                .map(|i| step as f32 * 100.0 + i as f32)
                .collect();
            let v: Vec<f32> = (0..kv_dim)
                .map(|i| step as f32 * 100.0 + i as f32 + 0.5)
                .collect();
            cache.append_kv_layer(layer, &k, &v);
        }
        cache.advance();
    }
    assert_eq!(cache.num_pages(), 2);

    // Verify all 6 tokens K and V, layer 0.
    let mut k_buf = vec![0.0f32; 6 * kv_dim];
    let mut v_buf = vec![0.0f32; 6 * kv_dim];
    cache.gather_k(0, &mut k_buf);
    cache.gather_v(0, &mut v_buf);
    for step in 0..6u32 {
        for i in 0..kv_dim {
            let k_expected = step as f32 * 100.0 + i as f32;
            let v_expected = k_expected + 0.5;
            let k_got = k_buf[step as usize * kv_dim + i];
            let v_got = v_buf[step as usize * kv_dim + i];
            assert!(
                (k_got - k_expected).abs() < 1e-6,
                "K step={step}, i={i}: expected {k_expected}, got {k_got}"
            );
            assert!(
                (v_got - v_expected).abs() < 1e-6,
                "V step={step}, i={i}: expected {v_expected}, got {v_got}"
            );
        }
    }
}

#[test]
fn paged_non_multiple_gather_kv() {
    // seq_len = page_size * 2 + 3: exercises partial final page for both K and V.
    let config = make_config(8);
    let kv_dim = config.kv_dim(); // 8
    let page_size = config.page_size; // 4
    let seq_len = page_size * 2 + 3; // 11
    let mut cache = PagedKVCache::new(config);

    for step in 0..seq_len as u32 {
        for layer in 0..2 {
            let k: Vec<f32> = (0..kv_dim)
                .map(|i| step as f32 * 10.0 + layer as f32 + i as f32 * 0.1)
                .collect();
            let v: Vec<f32> = k.iter().map(|&x| x + 0.5).collect();
            cache.append_kv_layer(layer, &k, &v);
        }
        cache.advance();
    }
    assert_eq!(cache.seq_len(), seq_len);

    let mut k_buf = vec![0.0f32; seq_len * kv_dim];
    let mut v_buf = vec![0.0f32; seq_len * kv_dim];
    cache.gather_k(0, &mut k_buf);
    cache.gather_v(0, &mut v_buf);

    for step in 0..seq_len as u32 {
        for i in 0..kv_dim {
            let k_expected = step as f32 * 10.0 + 0.0 + i as f32 * 0.1;
            let v_expected = k_expected + 0.5;
            let k_got = k_buf[step as usize * kv_dim + i];
            let v_got = v_buf[step as usize * kv_dim + i];
            assert!(
                (k_got - k_expected).abs() < 1e-5,
                "K step={step}, i={i}: expected {k_expected}, got {k_got}"
            );
            assert!(
                (v_got - v_expected).abs() < 1e-5,
                "V step={step}, i={i}: expected {v_expected}, got {v_got}"
            );
        }
    }
}

#[test]
fn paged_memory_accounting() {
    let config = PagedKVCacheConfig {
        page_size: 256,
        max_pages: 16,
        num_layers: 28,
        num_kv_heads: 8,
        head_dim: 128,
        eviction: EvictionPolicy::None,
    };

    // Each page: 28 layers * 2 * 256 tokens * 1024 kv_dim * 4 bytes
    let expected_per_page = 28 * 2 * 256 * 1024 * 4;
    assert_eq!(config.bytes_per_page(), expected_per_page);

    let cache = PagedKVCache::new(config);
    assert_eq!(cache.total_memory_bytes(), 16 * expected_per_page);
    assert_eq!(cache.used_memory_bytes(), 0);
}

#[test]
#[should_panic(expected = "PagePool exhausted")]
fn paged_no_eviction_panics_on_exhaustion() {
    let config = make_config(1); // Only 1 page, page_size=4.
    let kv_dim = config.kv_dim();
    let mut cache = PagedKVCache::new(config);

    let k = vec![1.0; kv_dim];
    let v = vec![2.0; kv_dim];

    // Fill the single page (4 tokens).
    for _ in 0..4 {
        for layer in 0..2 {
            cache.append_kv_layer(layer, &k, &v);
        }
        cache.advance();
    }

    // 5th token needs a new page -> should panic.
    for layer in 0..2 {
        cache.append_kv_layer(layer, &k, &v);
    }
}

#[test]
#[should_panic(expected = "does not yet support sequences beyond max_tokens")]
fn paged_lru_eviction_fails_closed_beyond_max_tokens() {
    // Regression for #337: the growth loop evicted a page and re-pushed
    // it (net-zero num_pages), without re-touching it in lru_order, so
    // appending past max_tokens (max_pages * page_size) silently drained
    // lru_order and panicked on an unrelated internal invariant ("LRU
    // order empty but pool exhausted") a few tokens later instead of
    // failing closed on the actual precondition.
    let mut config = make_config(2); // 2 pages * page_size 4 = 8 max_tokens.
    config.eviction = EvictionPolicy::Lru;
    let kv_dim = config.kv_dim();
    let mut cache = PagedKVCache::new(config);

    let k = vec![1.0; kv_dim];
    let v = vec![2.0; kv_dim];

    // Fill exactly max_tokens (8) tokens across the 2 pages.
    for _ in 0..8 {
        for layer in 0..2 {
            cache.append_kv_layer(layer, &k, &v);
        }
        cache.advance();
    }

    // 9th token exceeds max_tokens; Lru cannot yet slide the window.
    for layer in 0..2 {
        cache.append_kv_layer(layer, &k, &v);
    }
}

#[test]
fn paged_lru_overflow_leaves_live_pages_unchanged() {
    let mut config = make_config(2); // 2 pages * page_size 4 = 8 max_tokens.
    config.eviction = EvictionPolicy::Lru;
    let kv_dim = config.kv_dim();
    let max_tokens = config.max_tokens();
    let mut cache = PagedKVCache::new(config);

    for step in 0..max_tokens {
        for layer in 0..2 {
            let k = vec![step as f32; kv_dim];
            let v = vec![1000.0 + step as f32; kv_dim];
            cache.append_kv_layer(layer, &k, &v);
        }
        cache.advance();
    }

    let overflow = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for layer in 0..2 {
            let k = vec![999.0; kv_dim];
            let v = vec![1999.0; kv_dim];
            cache.append_kv_layer(layer, &k, &v);
        }
    }));
    assert!(overflow.is_err(), "LRU overflow should fail closed");

    assert_eq!(cache.seq_len(), max_tokens);
    assert_eq!(cache.num_pages(), 2);

    let mut k_buf = vec![0.0f32; max_tokens * kv_dim];
    let mut v_buf = vec![0.0f32; max_tokens * kv_dim];
    cache.gather_k(0, &mut k_buf);
    cache.gather_v(0, &mut v_buf);

    for step in 0..max_tokens {
        for i in 0..kv_dim {
            assert_eq!(k_buf[step * kv_dim + i], step as f32);
            assert_eq!(v_buf[step * kv_dim + i], 1000.0 + step as f32);
        }
    }
}

#[test]
#[should_panic(expected = "page_size must be non-zero")]
fn paged_zero_page_size_panics_at_construction() {
    // Regression for #244: page_size == 0 is the divisor in PageTable::resolve,
    // so it must be rejected at construction with a clear message instead of
    // surfacing later as a cryptic divide-by-zero in the hot path.
    let mut config = make_config(4);
    config.page_size = 0;
    let _ = PagedKVCache::new(config);
}

#[test]
#[should_panic(expected = "page_size must be non-zero")]
fn page_table_zero_page_size_panics() {
    // Regression for #244: `PageTable::new` is the authoritative chokepoint
    // (`SequenceManager::add` reaches it directly, bypassing the
    // `PagedKVCache` config guard), so the guard must live here too.
    let _ = PageTable::new(0);
}

#[test]
fn page_pool_alloc_free_cycle() {
    let mut pool = PagePool::new(4, 16);
    assert_eq!(pool.free_count(), 4);

    let p0 = pool.alloc().unwrap();
    let _p1 = pool.alloc().unwrap();
    assert_eq!(pool.free_count(), 2);
    assert_eq!(pool.allocated_count(), 2);

    pool.free(p0);
    assert_eq!(pool.free_count(), 3);

    // Re-allocate should give back p0 (LIFO).
    let p2 = pool.alloc().unwrap();
    assert_eq!(p2, p0);
}

#[test]
fn page_table_resolve() {
    let mut table = PageTable::new(4);
    table.push_page(10); // logical 0 -> physical 10
    table.push_page(5); // logical 1 -> physical 5
    table.set_seq_len(6);

    // Token 0 -> page 10, offset 0
    assert_eq!(table.resolve(0), (10, 0));
    // Token 3 -> page 10, offset 3
    assert_eq!(table.resolve(3), (10, 3));
    // Token 4 -> page 5, offset 0
    assert_eq!(table.resolve(4), (5, 0));
    // Token 5 -> page 5, offset 1
    assert_eq!(table.resolve(5), (5, 1));
}

// --- Overflow hardening tests (#460) ---

#[test]
fn paged_try_new_overflow_kv_dim_returns_invalid_input() {
    let config = PagedKVCacheConfig {
        page_size: 1,
        max_pages: 1,
        num_layers: 1,
        num_kv_heads: usize::MAX,
        head_dim: 2,
        eviction: EvictionPolicy::None,
    };
    let r = PagedKVCache::try_new(config);
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "expected InvalidInput on kv_dim overflow, got {r:?}"
    );
}

#[test]
fn paged_try_new_overflow_floats_per_page_returns_invalid_input() {
    let config = PagedKVCacheConfig {
        page_size: 8,
        max_pages: 1,
        num_layers: usize::MAX / 16 + 2,
        num_kv_heads: 1,
        head_dim: 1,
        eviction: EvictionPolicy::None,
    };
    let r = PagedKVCache::try_new(config);
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "expected InvalidInput on floats_per_page overflow, got {r:?}"
    );
}

#[test]
fn paged_try_new_overflow_total_bytes_returns_invalid_input() {
    let config = PagedKVCacheConfig {
        page_size: 1,
        max_pages: usize::MAX / 8 + 2,
        num_layers: 1,
        num_kv_heads: 1,
        head_dim: 1,
        eviction: EvictionPolicy::None,
    };
    let r = PagedKVCache::try_new(config);
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "expected InvalidInput on total_bytes overflow, got {r:?}"
    );
}

#[test]
fn page_pool_try_new_overflow_capacity_returns_invalid_input() {
    let r = PagePool::try_new(usize::MAX, 2);
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "expected InvalidInput on page-pool capacity overflow, got {r:?}"
    );
}

#[test]
fn page_pool_try_new_overflow_bytes_returns_invalid_input() {
    // Element count (1 * usize::MAX/2) fits usize, but the f32 byte layout
    // (len * 4) overflows. The fallible constructor must fail closed rather
    // than panic with "capacity overflow" inside `vec!`. This is the gap
    // `BatchWorker::try_new` inherits via `PagePool::try_new`.
    let r = PagePool::try_new(1, usize::MAX / 2);
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "expected InvalidInput on page-pool byte overflow, got {r:?}"
    );
}

#[test]
fn paged_try_new_valid_config_succeeds() {
    let config = make_config(2);
    let cache = PagedKVCache::try_new(config).expect("valid config must succeed");
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.free_pages(), 2);
}

// --- isize::MAX boundary tests (Fix 1 / Fix 3) ---

#[test]
fn page_pool_try_new_isize_max_byte_bound_returns_invalid_input() {
    // Element count fits usize (1 * (isize::MAX/4 + 1)), but the byte size
    // (floats * 4) exceeds isize::MAX. The guard added in Fix 1 must catch
    // this; without it the vec! macro panics with "capacity overflow".
    let floats_per_page = (isize::MAX as usize / std::mem::size_of::<f32>()) + 1;
    let r = PagePool::try_new(1, floats_per_page);
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "expected InvalidInput when byte size exceeds isize::MAX, got {r:?}"
    );
}

#[test]
fn page_pool_try_new_free_list_capacity_bound_returns_invalid_input() {
    // `floats_per_page = 0` makes `len = max_pages * 0 = 0`, so the `data`
    // guard above (byte_len = 0) passes trivially regardless of how huge
    // `max_pages` is. This test pins the *sibling* `free_list` guard,
    // which must independently catch a `max_pages` whose `usize`-element
    // (8 bytes each) byte layout exceeds `isize::MAX` even when the
    // `data` allocation is empty.
    let max_pages = (isize::MAX as usize / std::mem::size_of::<usize>()) + 1;
    let r = PagePool::try_new(max_pages, 0);
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "expected InvalidInput when free_list byte size exceeds isize::MAX, got {r:?}"
    );
}

#[test]
fn try_with_prefix_cache_oversized_prefix_geometry_returns_invalid_input() {
    // Construct a prefix cache whose prefix_page_size produces a per-page
    // float count such that floats * 4 > isize::MAX. The Fix 3 construction
    // guard must catch this before any allocation is attempted.
    let kv_dim = 1usize; // num_kv_heads=1, head_dim=1 → kv_dim=1
    // num_layers=1, 2, prefix_page_size=X, kv_dim=1 → floats = 2*X
    // We want 2*X > isize::MAX/4, so X > isize::MAX/8.
    let bad_prefix_page_size = (isize::MAX as usize / (2 * std::mem::size_of::<f32>())) + 1;
    let prefix_cache = Arc::new(Mutex::new(PrefixPageCache::new(PrefixPageCacheConfig {
        capacity: 1,
        prefix_page_size: bad_prefix_page_size,
        num_layers: 1,
        num_kv_heads: 1,
        head_dim: kv_dim,
    })));
    let config = PagedKVCacheConfig {
        page_size: 4,
        max_pages: 1,
        num_layers: 1,
        num_kv_heads: 1,
        head_dim: kv_dim,
        eviction: EvictionPolicy::None,
    };
    let r = PagedKVCache::try_with_prefix_cache(config, Some(prefix_cache));
    assert!(
        matches!(r, Err(InferenceError::InvalidInput(_))),
        "expected InvalidInput on oversized prefix geometry, got {r:?}"
    );
}
