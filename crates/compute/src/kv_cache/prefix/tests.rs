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
