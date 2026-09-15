use super::super::{MetalErnie45KvCache, MetalErnie45State};
use crate::InferenceError;
use crate::measurement::gpu_test_lock;
use crate::model::ernie45::{Ernie45Config, Ernie45LayerWeights, Ernie45Weights};

const STATE_CAPACITY: usize = 4;
const VOCAB: usize = 8;
const SENTINEL: f32 = 1234567.0;
type DispatchCounts = (u32, u32, u32, u32, u32, u32, u32);
const DISPATCHES: DispatchCounts = (8, 3, 2, 1, 2, 0, 1);
const NO_DISPATCHES: DispatchCounts = (0, 0, 0, 0, 0, 0, 0);

fn fixture() -> (Ernie45Config, Ernie45Weights) {
    let mut cfg = super::config(128);
    cfg.num_hidden_layers = 1;
    cfg.vocab_size = VOCAB;
    let h = cfg.hidden_size;
    let q = cfg.num_attention_heads * cfg.head_dim;
    let kv = cfg.num_key_value_heads * cfg.head_dim;
    let i = cfg.intermediate_size;
    let weights = Ernie45Weights {
        embed_tokens: vec![0.25; VOCAB * h],
        layers: vec![Ernie45LayerWeights {
            q_proj: vec![0.001; q * h],
            k_proj: vec![0.002; kv * h],
            v_proj: vec![0.003; kv * h],
            o_proj: vec![0.004; h * q],
            gate_proj: vec![0.001; i * h],
            up_proj: vec![0.002; i * h],
            down_proj: vec![0.003; h * i],
            input_layernorm: vec![1.0; h],
            post_attention_layernorm: vec![1.0; h],
        }],
        final_norm: vec![1.0; h],
        lm_head: vec![0.001; VOCAB * h],
    };
    (cfg, weights)
}

fn embedding() -> Vec<f32> {
    (0..1024).map(|i| 0.25 + (i % 11) as f32 * 0.02).collect()
}

fn fill_cache(state: &mut MetalErnie45State, cache: &mut MetalErnie45KvCache) {
    assert!(cache.is_empty());
    let row = embedding();
    let mut logits = vec![f32::NAN; VOCAB];
    state
        .kv_prefill(&row, &[[7; 3]], cache, &mut logits)
        .expect("synthetic Metal prefill executes");
    assert_eq!(state.last_dispatch_counts(), DISPATCHES);
    assert_eq!(cache.len(), 1);
    assert!(logits.iter().all(|value| value.is_finite()));
    state
        .kv_decode_step(&row, [13; 3], cache, &mut logits)
        .expect("synthetic Metal decode executes");
    assert_eq!(state.last_dispatch_counts(), DISPATCHES);
    assert_eq!(cache.len(), 2);
    assert!(logits.iter().all(|value| value.is_finite()));
    let (k, v) = cache.layer_rows_for_test(0);
    assert!(k.iter().any(|&value| value != 0.0));
    assert!(v.iter().any(|&value| value != 0.0));
}

fn primed_cache(state: &mut MetalErnie45State, capacity: usize) -> MetalErnie45KvCache {
    let mut cache = state.new_kv_cache(capacity).expect("valid cache allocates");
    fill_cache(state, &mut cache);
    cache
}

#[derive(Debug, PartialEq)]
struct Snapshot {
    len: usize,
    capacity: usize,
    layers: usize,
    kv_dim: usize,
    k: Vec<u32>,
    v: Vec<u32>,
}

fn snapshot(cache: &MetalErnie45KvCache) -> Snapshot {
    let mut k = Vec::new();
    let mut v = Vec::new();
    for layer in 0..cache.layers() {
        let (keys, values) = cache.layer_rows_for_test(layer);
        k.extend(keys.into_iter().map(f32::to_bits));
        v.extend(values.into_iter().map(f32::to_bits));
    }
    Snapshot {
        len: cache.len(),
        capacity: cache.capacity(),
        layers: cache.layers(),
        kv_dim: cache.kv_dim(),
        k,
        v,
    }
}

fn assert_invalid<T>(result: Result<T, InferenceError>, reason: &str, context: &str) {
    match result {
        Err(InferenceError::InvalidInput(actual)) => {
            assert_eq!(actual, format!("ernie45 Metal: {reason}"), "{context}");
        }
        Err(other) => panic!("{context}: expected InvalidInput({reason}), got {other:?}"),
        Ok(_) => panic!("{context}: expected InvalidInput({reason}), got success"),
    }
}

fn rejected_prefill(
    state: &mut MetalErnie45State,
    cache: &mut MetalErnie45KvCache,
    embeds: &[f32],
    positions: &[[u32; 3]],
    output_len: usize,
    reason: &str,
    context: &str,
) {
    let before = snapshot(cache);
    let mut logits = vec![SENTINEL; output_len];
    assert_invalid(
        state.kv_prefill(embeds, positions, cache, &mut logits),
        reason,
        context,
    );
    assert!(
        logits
            .iter()
            .all(|&value| value.to_bits() == SENTINEL.to_bits()),
        "{context}: output changed"
    );
    assert_eq!(snapshot(cache), before, "{context}: cache changed");
    assert_eq!(state.last_dispatch_counts(), NO_DISPATCHES, "{context}");
}

fn rejected_decode(
    state: &mut MetalErnie45State,
    cache: &mut MetalErnie45KvCache,
    embeds: &[f32],
    output_len: usize,
    reason: &str,
    context: &str,
) {
    rejected_decode_at(state, cache, embeds, [19; 3], output_len, reason, context);
}

fn rejected_decode_at(
    state: &mut MetalErnie45State,
    cache: &mut MetalErnie45KvCache,
    embeds: &[f32],
    position: [u32; 3],
    output_len: usize,
    reason: &str,
    context: &str,
) {
    let before = snapshot(cache);
    let mut logits = vec![SENTINEL; output_len];
    assert_invalid(
        state.kv_decode_step(embeds, position, cache, &mut logits),
        reason,
        context,
    );
    assert!(
        logits
            .iter()
            .all(|&value| value.to_bits() == SENTINEL.to_bits()),
        "{context}: output changed"
    );
    assert_eq!(snapshot(cache), before, "{context}: cache changed");
    assert_eq!(state.last_dispatch_counts(), NO_DISPATCHES, "{context}");
}

fn report_success(marker: &str, dispatches: DispatchCounts) {
    assert_eq!(dispatches, DISPATCHES);
    eprintln!("{marker} executed=true prefill_and_decode=true dispatches={dispatches:?}");
}

#[test]
fn metal_ernie45_guard_cache_capacity() {
    let _gpu_guard = gpu_test_lock();
    if metal::Device::system_default().is_none() {
        assert!(!super::enforce(), "Metal device required under enforcement");
        eprintln!("SKIP metal_ernie45 cache capacity: Metal device missing");
        return;
    }
    let (cfg, weights) = fixture();
    let mut state = MetalErnie45State::new(&cfg, &weights, STATE_CAPACITY)
        .expect("synthetic Metal state constructs");
    let cache = primed_cache(&mut state, STATE_CAPACITY);
    let dispatches = state.last_dispatch_counts();
    let before = snapshot(&cache);
    // A disabled capacity guard can only allocate one extra row before this assertion fails.
    for capacity in [STATE_CAPACITY + 1, 0] {
        assert_invalid(
            state.new_kv_cache(capacity),
            "cache capacity is outside the state capacity",
            "cache capacity refusal",
        );
        assert_eq!(
            snapshot(&cache),
            before,
            "allocation refusal changed live cache"
        );
    }
    report_success("[METAL_ERNIE45_GUARD_CACHE_CAPACITY]", dispatches);
}

#[test]
fn metal_ernie45_guard_cache_owner() {
    let _gpu_guard = gpu_test_lock();
    if metal::Device::system_default().is_none() {
        assert!(!super::enforce(), "Metal device required under enforcement");
        eprintln!("SKIP metal_ernie45 cache owner: Metal device missing");
        return;
    }
    let (cfg, mut weights) = fixture();
    let mut state = MetalErnie45State::new(&cfg, &weights, STATE_CAPACITY)
        .expect("synthetic Metal state constructs");
    let own_cache = primed_cache(&mut state, STATE_CAPACITY);
    let dispatches = state.last_dispatch_counts();
    let own_before = snapshot(&own_cache);
    weights.lm_head[0] += 0.125;
    let mut other_state = MetalErnie45State::new(&cfg, &weights, STATE_CAPACITY)
        .expect("same geometry with different weights constructs");
    let mut foreign_cache = primed_cache(&mut other_state, STATE_CAPACITY);
    assert_eq!(foreign_cache.layers(), own_cache.layers());
    assert_eq!(foreign_cache.kv_dim(), own_cache.kv_dim());
    assert_eq!(foreign_cache.capacity(), own_cache.capacity());
    let reason = "cache shape or owning decoder does not match";
    let context = "foreign cache ownership refusal";
    // The short embedding still refuses before submission if ownership validation disappears.
    rejected_decode(&mut state, &mut foreign_cache, &[], VOCAB, reason, context);
    rejected_decode(
        &mut state,
        &mut foreign_cache,
        &embedding(),
        VOCAB,
        reason,
        context,
    );
    foreign_cache.clear();
    rejected_prefill(
        &mut state,
        &mut foreign_cache,
        &[],
        &[[7; 3]],
        VOCAB,
        reason,
        context,
    );
    rejected_prefill(
        &mut state,
        &mut foreign_cache,
        &embedding(),
        &[[7; 3]],
        VOCAB,
        reason,
        context,
    );
    assert_eq!(snapshot(&own_cache), own_before);
    report_success("[METAL_ERNIE45_GUARD_CACHE_OWNER]", dispatches);
}

#[test]
fn metal_ernie45_guard_prefill_nonempty() {
    let _gpu_guard = gpu_test_lock();
    if metal::Device::system_default().is_none() {
        assert!(!super::enforce(), "Metal device required under enforcement");
        eprintln!("SKIP metal_ernie45 nonempty prefill: Metal device missing");
        return;
    }
    let (cfg, weights) = fixture();
    let mut state = MetalErnie45State::new(&cfg, &weights, STATE_CAPACITY)
        .expect("synthetic Metal state constructs");
    let mut cache = primed_cache(&mut state, STATE_CAPACITY);
    let dispatches = state.last_dispatch_counts();
    let row = embedding();
    for embeds in [&[][..], row.as_slice()] {
        rejected_prefill(
            &mut state,
            &mut cache,
            embeds,
            &[[7; 3]],
            VOCAB,
            "kv prefill requires an empty cache",
            "nonempty prefill cache refusal",
        );
    }
    report_success("[METAL_ERNIE45_GUARD_PREFILL_NONEMPTY]", dispatches);
}

#[test]
fn metal_ernie45_guard_prefill_sequence() {
    let _gpu_guard = gpu_test_lock();
    if metal::Device::system_default().is_none() {
        assert!(!super::enforce(), "Metal device required under enforcement");
        eprintln!("SKIP metal_ernie45 prefill sequence: Metal device missing");
        return;
    }
    let (cfg, weights) = fixture();
    let mut state = MetalErnie45State::new(&cfg, &weights, STATE_CAPACITY)
        .expect("synthetic Metal state constructs");
    let mut cache = primed_cache(&mut state, STATE_CAPACITY - 1);
    let dispatches = state.last_dispatch_counts();
    cache.clear();
    let reason = "kv prefill sequence is outside cache capacity";
    let context = "prefill sequence capacity refusal";
    rejected_prefill(&mut state, &mut cache, &[], &[], VOCAB, reason, context);
    let oversized_positions = [[7; 3]; STATE_CAPACITY];
    rejected_prefill(
        &mut state,
        &mut cache,
        &[],
        &oversized_positions,
        VOCAB,
        reason,
        context,
    );
    rejected_prefill(
        &mut state,
        &mut cache,
        &embedding().repeat(STATE_CAPACITY),
        &oversized_positions,
        VOCAB,
        reason,
        context,
    );
    report_success("[METAL_ERNIE45_GUARD_PREFILL_SEQUENCE]", dispatches);
}

#[test]
fn metal_ernie45_guard_decode_empty() {
    let _gpu_guard = gpu_test_lock();
    if metal::Device::system_default().is_none() {
        assert!(!super::enforce(), "Metal device required under enforcement");
        eprintln!("SKIP metal_ernie45 empty decode: Metal device missing");
        return;
    }
    let (cfg, weights) = fixture();
    let mut state = MetalErnie45State::new(&cfg, &weights, STATE_CAPACITY)
        .expect("synthetic Metal state constructs");
    let mut cache = primed_cache(&mut state, STATE_CAPACITY);
    let dispatches = state.last_dispatch_counts();
    cache.clear();
    let row = embedding();
    for embeds in [&[][..], row.as_slice()] {
        rejected_decode(
            &mut state,
            &mut cache,
            embeds,
            VOCAB,
            "kv decode cache is empty; run kv_prefill first",
            "empty decode cache refusal",
        );
    }
    report_success("[METAL_ERNIE45_GUARD_DECODE_EMPTY]", dispatches);
}

#[test]
fn metal_ernie45_guard_decode_full() {
    let _gpu_guard = gpu_test_lock();
    if metal::Device::system_default().is_none() {
        assert!(!super::enforce(), "Metal device required under enforcement");
        eprintln!("SKIP metal_ernie45 full decode: Metal device missing");
        return;
    }
    let (cfg, weights) = fixture();
    let mut state = MetalErnie45State::new(&cfg, &weights, STATE_CAPACITY)
        .expect("synthetic Metal state constructs");
    let mut cache = primed_cache(&mut state, 2);
    let dispatches = state.last_dispatch_counts();
    let row = embedding();
    // The first refusal must fail on its reason before a removed full-cache guard can append.
    for embeds in [&[][..], row.as_slice()] {
        rejected_decode(
            &mut state,
            &mut cache,
            embeds,
            VOCAB,
            "kv decode cache is full",
            "full decode cache refusal",
        );
    }
    report_success("[METAL_ERNIE45_GUARD_DECODE_FULL]", dispatches);
}

#[test]
fn metal_ernie45_guard_embedding_shape() {
    let _gpu_guard = gpu_test_lock();
    if metal::Device::system_default().is_none() {
        assert!(!super::enforce(), "Metal device required under enforcement");
        eprintln!("SKIP metal_ernie45 embedding shape: Metal device missing");
        return;
    }
    let (cfg, weights) = fixture();
    let mut state = MetalErnie45State::new(&cfg, &weights, STATE_CAPACITY)
        .expect("synthetic Metal state constructs");
    let mut cache = primed_cache(&mut state, STATE_CAPACITY);
    let dispatches = state.last_dispatch_counts();
    cache.clear();
    let reason = "embeds must have the exact [seq_len,hidden_size] shape";
    let context = "embedding shape refusal";
    // An independently short output catches a missing shape guard before the host copy.
    rejected_prefill(&mut state, &mut cache, &[], &[[7; 3]], 0, reason, context);
    let row = embedding();
    let mut long = row.clone();
    long.push(0.25);
    for embeds in [&row[..row.len() - 1], long.as_slice()] {
        rejected_prefill(
            &mut state,
            &mut cache,
            embeds,
            &[[7; 3]],
            VOCAB,
            reason,
            context,
        );
    }
    fill_cache(&mut state, &mut cache);
    rejected_decode(&mut state, &mut cache, &[], 0, reason, context);
    for embeds in [&row[..row.len() - 1], long.as_slice()] {
        rejected_decode(&mut state, &mut cache, embeds, VOCAB, reason, context);
    }
    report_success("[METAL_ERNIE45_GUARD_EMBEDDING_SHAPE]", dispatches);
}

#[test]
fn metal_ernie45_guard_output_shape() {
    let _gpu_guard = gpu_test_lock();
    if metal::Device::system_default().is_none() {
        assert!(!super::enforce(), "Metal device required under enforcement");
        eprintln!("SKIP metal_ernie45 output shape: Metal device missing");
        return;
    }
    let (cfg, weights) = fixture();
    let mut state = MetalErnie45State::new(&cfg, &weights, STATE_CAPACITY)
        .expect("synthetic Metal state constructs");
    let mut cache = primed_cache(&mut state, STATE_CAPACITY);
    let dispatches = state.last_dispatch_counts();
    cache.clear();
    let reason = "cached logits must have exactly vocab_size values";
    let context = "output shape refusal";
    let row = embedding();
    let mut nonfinite = row.clone();
    nonfinite[0] = f32::NAN;
    // Non-finite input catches a removed output guard before any GPU submission.
    rejected_prefill(
        &mut state,
        &mut cache,
        &nonfinite,
        &[[7; 3]],
        0,
        reason,
        context,
    );
    for output_len in [VOCAB - 1, VOCAB + 1] {
        rejected_prefill(
            &mut state,
            &mut cache,
            &row,
            &[[7; 3]],
            output_len,
            reason,
            context,
        );
    }
    fill_cache(&mut state, &mut cache);
    rejected_decode(&mut state, &mut cache, &nonfinite, 0, reason, context);
    for output_len in [VOCAB - 1, VOCAB + 1] {
        rejected_decode(&mut state, &mut cache, &row, output_len, reason, context);
    }
    report_success("[METAL_ERNIE45_GUARD_OUTPUT_SHAPE]", dispatches);
}

#[test]
fn metal_ernie45_guard_finite_embeddings() {
    let _gpu_guard = gpu_test_lock();
    if metal::Device::system_default().is_none() {
        assert!(!super::enforce(), "Metal device required under enforcement");
        eprintln!("SKIP metal_ernie45 finite embeddings: Metal device missing");
        return;
    }
    let (mut cfg, weights) = fixture();
    cfg.rope_theta = 1e-30;
    let mut state = MetalErnie45State::new(&cfg, &weights, STATE_CAPACITY)
        .expect("synthetic Metal state constructs");
    let mut cache = primed_cache(&mut state, STATE_CAPACITY);
    let dispatches = state.last_dispatch_counts();
    cache.clear();
    let reason = "embeds contain a non-finite value";
    let context = "non-finite embedding refusal";
    let mut nonfinite = embedding();
    nonfinite[0] = f32::NAN;
    // The independently overflowing angle refuses before submission if input validation disappears.
    rejected_prefill(
        &mut state,
        &mut cache,
        &nonfinite,
        &[[u32::MAX; 3]],
        VOCAB,
        reason,
        context,
    );
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut row = embedding();
        row[0] = value;
        rejected_prefill(
            &mut state,
            &mut cache,
            &row,
            &[[7; 3]],
            VOCAB,
            reason,
            context,
        );
    }
    fill_cache(&mut state, &mut cache);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut row = embedding();
        row[0] = value;
        rejected_decode(&mut state, &mut cache, &row, VOCAB, reason, context);
    }
    report_success("[METAL_ERNIE45_GUARD_FINITE_EMBEDDINGS]", dispatches);
}

#[test]
fn metal_ernie45_guard_finite_positions() {
    let _gpu_guard = gpu_test_lock();
    if metal::Device::system_default().is_none() {
        assert!(!super::enforce(), "Metal device required under enforcement");
        eprintln!("SKIP metal_ernie45 finite positions: Metal device missing");
        return;
    }
    let (mut cfg, weights) = fixture();
    cfg.rope_theta = 1e-30;
    let mut state = MetalErnie45State::new(&cfg, &weights, STATE_CAPACITY)
        .expect("small positive theta passes construction for bounded initial positions");
    let mut cache = primed_cache(&mut state, STATE_CAPACITY);
    let dispatches = state.last_dispatch_counts();
    cache.clear();
    let row = embedding();
    let reason = "position produces non-finite RoPE tables";
    let context = "non-finite position refusal";
    // This accepted theta keeps positions 0..19 finite, while u32::MAX overflows high lanes.
    rejected_prefill(
        &mut state,
        &mut cache,
        &row,
        &[[u32::MAX; 3]],
        VOCAB,
        reason,
        context,
    );
    fill_cache(&mut state, &mut cache);
    rejected_decode_at(
        &mut state,
        &mut cache,
        &row,
        [u32::MAX; 3],
        VOCAB,
        reason,
        context,
    );
    report_success("[METAL_ERNIE45_GUARD_FINITE_POSITIONS]", dispatches);
}
