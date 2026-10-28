use super::*;

fn make_config(capacity: usize) -> PrefixPageCacheConfig {
    PrefixPageCacheConfig {
        capacity,
        prefix_page_size: 4,
        num_layers: 2,
        num_kv_heads: 2,
        head_dim: 4,
    }
}

fn make_page(config: PrefixPageCacheConfig, marker: f32) -> SharedPageRef {
    SharedPageRef::from_vec(vec![marker; config.floats_per_prefix_page()])
}

fn make_pages(config: PrefixPageCacheConfig, prefix_len: usize, marker: f32) -> Vec<SharedPageRef> {
    let count = PrefixEntry::pages_for_tokens(prefix_len, config.prefix_page_size);
    (0..count)
        .map(|idx| make_page(config, marker + idx as f32))
        .collect()
}

#[test]
fn test_empty_cache_lookup() {
    let config = make_config(4);
    let mut cache = PrefixPageCache::new(config);
    let key = PrefixKey::from_token_ids(AdapterId::BASE, &[1, 2, 3]);

    assert!(cache.lookup(&key).is_none());
    assert!(cache.is_empty());
}

#[test]
fn test_prefix_lookup_hit() {
    let config = make_config(4);
    let mut cache = PrefixPageCache::new(config);
    let key = PrefixKey::from_token_ids(AdapterId::BASE, &[1, 2, 3, 4]);
    let pages = make_pages(config, 4, 7.0);

    cache.insert(key, 4, pages);
    let entry = cache.lookup(&key).expect("prefix should hit");

    assert_eq!(entry.prefix_len, 4);
    assert_eq!(entry.pages.len(), 1);
    assert_eq!(entry.pages[0].as_slice()[0], 7.0);
    assert!(entry.last_used > 0);
}

#[test]
fn test_prefix_cache_miss() {
    let config = make_config(4);
    let mut cache = PrefixPageCache::new(config);
    let key = PrefixKey::from_token_ids(AdapterId::BASE, &[99, 100]);

    assert!(cache.lookup(&key).is_none());
}

#[test]
fn test_evict_lru_reclaims_unreferenced() {
    let config = make_config(2);
    let mut cache = PrefixPageCache::new(config);
    let key_a = PrefixKey::from_token_ids(AdapterId::BASE, &[1, 2, 3, 4]);
    let key_b = PrefixKey::from_token_ids(AdapterId::BASE, &[5, 6, 7, 8]);

    cache.insert(key_a, 4, make_pages(config, 4, 1.0));
    cache.insert(key_b, 4, make_pages(config, 4, 2.0));

    let _ = cache
        .lookup(&key_a)
        .expect("key_a should hit and become MRU");
    let pages_freed = cache.evict_lru();

    assert_eq!(pages_freed, 1);
    assert!(cache.contains_key(&key_a));
    assert!(!cache.contains_key(&key_b));
}

#[test]
fn test_adapter_keying_separates_entries() {
    let config = make_config(4);
    let mut cache = PrefixPageCache::new(config);
    let tokens = [1, 2, 3, 4];
    let base_key = PrefixKey::from_token_ids(AdapterId::BASE, &tokens);
    let adapter_key = PrefixKey::from_token_ids(AdapterId::new(42), &tokens);

    cache.insert(base_key, 4, make_pages(config, 4, 1.0));

    assert!(cache.lookup(&base_key).is_some());
    assert!(cache.lookup(&adapter_key).is_none());
}

#[test]
fn test_insert_at_capacity() {
    let config = make_config(1);
    let mut cache = PrefixPageCache::new(config);
    let key_a = PrefixKey::from_token_ids(AdapterId::BASE, &[1, 2, 3, 4]);
    let key_b = PrefixKey::from_token_ids(AdapterId::BASE, &[9, 8, 7, 6]);

    cache.insert(key_a, 4, make_pages(config, 4, 1.0));
    cache.insert(key_b, 4, make_pages(config, 4, 2.0));

    assert_eq!(cache.len(), 1);
    assert!(!cache.contains_key(&key_a));
    assert!(cache.contains_key(&key_b));
}

impl PrefixPageCache {
    pub(in crate::kv_cache) fn rekey_candidate_for_test(
        &mut self,
        original: PrefixKey,
        selected: PrefixKey,
    ) {
        let stored = self
            .entries
            .shift_remove(&original)
            .expect("original entry");
        assert!(self.entries.insert(selected, stored).is_none());
    }
}

#[test]
fn verified_identity_mismatch_preserves_cache_clock_and_order() {
    let config = make_config(4);
    let mut cache = PrefixPageCache::new(config);
    let tokens_a = [1, 2, 3, 4];
    let tokens_b = [4, 3, 2, 1];
    let key_a = PrefixKey::from_token_ids(AdapterId::BASE, &tokens_a);
    let key_b = PrefixKey::from_token_ids(AdapterId::BASE, &tokens_b);
    cache.insert_for_tokens(AdapterId::BASE, &tokens_a, make_pages(config, 4, 7.0));
    cache.insert_for_tokens(AdapterId::new(42), &tokens_a, make_pages(config, 4, 8.0));
    // Force candidate selection without asserting a natural FxHash collision.
    cache.rekey_candidate_for_test(key_a, key_b);
    let selected = cache
        .lookup(&key_b)
        .expect("ordinary lookup selects A pages");
    assert_eq!(selected.prefix_len, tokens_b.len());
    assert_eq!(selected.pages[0].as_slice(), vec![7.0; 128]);
    let stored = &cache.entries[&key_b];
    assert_eq!(stored.token_ids.as_deref(), Some(tokens_a.as_slice()));
    let token_pointer = stored.token_ids.as_ref().expect("verified tokens").as_ptr();
    let page_pointer = stored.entry.pages[0].as_slice().as_ptr();
    let clock = cache.clock;
    let order: Vec<_> = cache.entries.keys().copied().collect();
    let last_used = stored.entry.last_used;

    let error = cache
        .lookup_for_tokens(&key_b, &tokens_b)
        .expect_err("different ordered tokens must not qualify");

    assert!(matches!(error, InferenceError::PrefixCache(ref message)
        if message == "prefix token identity mismatch"));
    assert_eq!(cache.clock, clock);
    assert_eq!(cache.entries.keys().copied().collect::<Vec<_>>(), order);
    let stored = &cache.entries[&key_b];
    assert_eq!(stored.entry.last_used, last_used);
    assert_eq!(stored.token_ids.as_deref(), Some(tokens_a.as_slice()));
    assert_eq!(
        stored.token_ids.as_ref().expect("verified tokens").as_ptr(),
        token_pointer
    );
    assert_eq!(stored.entry.pages[0].as_slice().as_ptr(), page_pointer);
    assert_eq!(stored.entry.pages[0].as_slice(), vec![7.0; 128]);
}

#[test]
fn legacy_entry_checks_length_before_returning_an_unverified_miss() {
    let config = make_config(2);
    let mut cache = PrefixPageCache::new(config);
    let tokens = [1, 2, 3, 4];
    let key = PrefixKey::from_token_ids(AdapterId::BASE, &tokens);
    cache.insert(key, 8, make_pages(config, 8, 3.0));
    let clock = cache.clock;
    let order: Vec<_> = cache.entries.keys().copied().collect();

    let error = cache
        .lookup_for_tokens(&key, &tokens)
        .expect_err("length refusal");

    assert!(matches!(error, InferenceError::PrefixCache(ref message)
        if message == "prefix hash collision or invalid entry length: key length 4, entry length 8"));
    assert_eq!(cache.clock, clock);
    assert_eq!(cache.entries.keys().copied().collect::<Vec<_>>(), order);
    cache.insert(key, 4, make_pages(config, 4, 5.0));
    let clock = cache.clock;
    assert!(
        cache
            .lookup_for_tokens(&key, &tokens)
            .expect("safe miss")
            .is_none()
    );
    assert_eq!(cache.clock, clock);
    assert_eq!(
        cache
            .lookup(&key)
            .expect("metadata lookup is preserved")
            .pages[0]
            .as_slice(),
        vec![5.0; 128]
    );
}

#[test]
fn token_free_insert_invalidates_same_key_verified_identity() {
    let config = make_config(2);
    let mut cache = PrefixPageCache::new(config);
    let tokens = [1, 2, 3, 4];
    let key = PrefixKey::from_token_ids(AdapterId::BASE, &tokens);
    cache.insert_for_tokens(AdapterId::BASE, &tokens, make_pages(config, 4, 7.0));

    let replaced = cache
        .insert(key, 4, make_pages(config, 4, 9.0))
        .expect("same key replaced");

    assert_eq!(replaced.pages[0].as_slice(), vec![7.0; 128]);
    let clock = cache.clock;
    assert!(
        cache
            .lookup_for_tokens(&key, &tokens)
            .expect("unverified miss")
            .is_none()
    );
    assert_eq!(cache.clock, clock);
    assert!(cache.entries[&key].token_ids.is_none());
    assert_eq!(
        cache.lookup(&key).expect("replacement retained").pages[0].as_slice(),
        vec![9.0; 128]
    );
}

#[test]
fn token_free_insert_entry_invalidates_same_key_verified_identity() {
    let config = make_config(2);
    let mut cache = PrefixPageCache::new(config);
    let tokens = [1, 2, 3, 4];
    let key = PrefixKey::from_token_ids(AdapterId::BASE, &tokens);
    cache.insert_for_tokens(AdapterId::BASE, &tokens, make_pages(config, 4, 7.0));
    let entry = PrefixEntry {
        prefix_len: 4,
        prefix_page_size: 4,
        pages: make_pages(config, 4, 11.0),
        last_used: 999,
    };

    let replaced = cache.insert_entry(key, entry).expect("same key replaced");

    assert_eq!(replaced.pages[0].as_slice(), vec![7.0; 128]);
    let clock = cache.clock;
    assert!(
        cache
            .lookup_for_tokens(&key, &tokens)
            .expect("unverified miss")
            .is_none()
    );
    assert_eq!(cache.clock, clock);
    assert!(cache.entries[&key].token_ids.is_none());
    let replacement = cache.lookup(&key).expect("replacement retained");
    assert_eq!(replacement.pages[0].as_slice(), vec![11.0; 128]);
    assert_ne!(replacement.last_used, 999);
}

#[test]
fn verified_identity_does_not_survive_clear_or_eviction() {
    for clear in [true, false] {
        let config = make_config(1);
        let mut cache = PrefixPageCache::new(config);
        let tokens = [1, 2, 3, 4];
        let key = PrefixKey::from_token_ids(AdapterId::BASE, &tokens);
        cache.insert_for_tokens(AdapterId::BASE, &tokens, make_pages(config, 4, 7.0));
        let external = cache.lookup(&key).expect("external page reference");
        let old_page = Arc::downgrade(&external.pages[0].page);
        assert_eq!(external.pages[0].strong_count(), 2);

        if clear {
            cache.clear();
            assert_eq!(cache.clock, 0);
        } else {
            assert_eq!(cache.evict_lru(), 0, "external clone keeps page alive");
        }

        assert!(cache.entries.is_empty());
        assert_eq!(external.pages[0].strong_count(), 1);
        cache.insert(key, 4, make_pages(config, 4, 5.0));
        assert!(
            cache
                .lookup_for_tokens(&key, &tokens)
                .expect("no stale proof")
                .is_none()
        );
        drop(external);
        assert!(old_page.upgrade().is_none());
    }
}

#[test]
fn prefix_cache_debug_keeps_verified_token_ids_private() {
    let config = make_config(2);
    let mut cache = PrefixPageCache::new(config);
    let tokens: [u32; 4] = [1_234_567_890, 2_345_678_901, 3_456_789_012, 4_000_000_001];
    cache.insert_for_tokens(AdapterId::BASE, &tokens, make_pages(config, 4, 7.0));

    let rendered = format!("{cache:?}");

    assert!(!rendered.contains(&format!("{tokens:?}")));
    assert!(!rendered.contains("token_ids"));
    assert!(rendered.contains("PrefixEntry"));
    assert!(rendered.contains("prefix_len: 4"));
}

#[test]
fn lookup_refresh_preserves_other_entries_lru_order() {
    let config = make_config(3);
    let mut cache = PrefixPageCache::new(config);
    let [a, b, c, d] = [1_u32, 5, 9, 13].map(|first| {
        PrefixKey::from_token_ids(AdapterId::BASE, &[first, first + 1, first + 2, first + 3])
    });
    for (key, marker) in [(a, 1.0), (b, 2.0), (c, 3.0)] {
        assert!(
            cache
                .insert(key, 4, make_pages(config, 4, marker))
                .is_none()
        );
    }

    assert!(cache.lookup(&a).is_some());
    assert!(cache.insert(d, 4, make_pages(config, 4, 4.0)).is_none());

    assert_eq!(cache.len(), 3);
    assert!(cache.contains_key(&a));
    assert!(!cache.contains_key(&b));
    assert!(cache.contains_key(&c));
    assert!(cache.contains_key(&d));
    assert_eq!(cache.evict_lru(), 1);
    assert!(!cache.contains_key(&c));
    assert!(cache.contains_key(&a));
    assert!(cache.contains_key(&d));
}

#[test]
fn replacement_refresh_preserves_other_entries_lru_order() {
    let config = make_config(3);
    let mut cache = PrefixPageCache::new(config);
    let [a, b, c, d] = [1_u32, 5, 9, 13].map(|first| {
        PrefixKey::from_token_ids(AdapterId::BASE, &[first, first + 1, first + 2, first + 3])
    });
    for (key, marker) in [(a, 1.0), (b, 2.0), (c, 3.0)] {
        assert!(
            cache
                .insert(key, 4, make_pages(config, 4, marker))
                .is_none()
        );
    }

    let replacement = PrefixEntry::new(4, config.prefix_page_size, make_pages(config, 4, 9.0), 0);
    let replaced = cache
        .insert_entry(a, replacement)
        .expect("same key replaced");
    assert_eq!(replaced.pages[0].as_slice()[0], 1.0);
    assert!(cache.insert(d, 4, make_pages(config, 4, 4.0)).is_none());

    assert_eq!(cache.len(), 3);
    assert!(cache.contains_key(&a));
    assert!(!cache.contains_key(&b));
    assert!(cache.contains_key(&c));
    assert!(cache.contains_key(&d));
    assert_eq!(cache.evict_lru(), 1);
    assert!(!cache.contains_key(&c));
    assert!(cache.contains_key(&a));
    assert!(cache.contains_key(&d));
    let retained = cache.lookup(&a).expect("replacement remains cached");
    assert_eq!(retained.pages[0].as_slice()[0], 9.0);
}
