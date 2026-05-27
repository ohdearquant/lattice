//! Criterion benchmarks for FlatKVCache f16 storage (ADR-062 Phase 1).
//!
//! Measures:
//! - `append_kv` throughput (f32→f16 conversion cost per token per layer)
//! - `read_k_into` throughput (f16→f32 dequantization cost)
//! - Memory footprint for Qwen3-0.6B config (target: ~448 MB vs 896 MB f32)

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_inference::kv_cache::{FlatKVCache, FlatKVCacheConfig};

/// Qwen3-0.6B: 28 layers, 8 KV heads, head_dim=128, max_seq=4096.
/// kv_dim = 8 * 128 = 1024 elements per token per layer.
fn qwen3_06b_config(max_seq_len: usize) -> FlatKVCacheConfig {
    FlatKVCacheConfig::for_qwen3(28, 8, 128, max_seq_len)
}

/// Small config for per-token benchmarks (1 layer, short seq).
fn bench_config() -> FlatKVCacheConfig {
    FlatKVCacheConfig {
        num_layers: 1,
        num_kv_heads: 8,
        head_dim: 128,
        max_seq_len: 512,
    }
}

fn bench_append_kv(c: &mut Criterion) {
    let config = bench_config();
    let kv_dim = config.kv_dim();

    let k_token: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.001).collect();
    let v_token: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.002 + 1.0).collect();

    let mut group = c.benchmark_group("kv_cache_f16/append_kv");
    // Throughput in bytes of f32 input converted per iteration.
    group.throughput(Throughput::Bytes(
        (kv_dim * 2 * std::mem::size_of::<f32>()) as u64,
    ));

    // Pre-allocate the cache outside the measured loop; each iteration resets it
    // so that only `append_kv` (the f32→f16 conversion) is measured, not allocation.
    group.bench_function("f32_to_f16_one_token", |b| {
        b.iter_batched(
            || FlatKVCache::new(config.clone()),
            |mut cache| {
                cache.append_kv(0, black_box(&k_token), black_box(&v_token));
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_read_k_into(c: &mut Criterion) {
    let config = bench_config();
    let kv_dim = config.kv_dim();
    let seq_len = 256;

    // Pre-fill the cache with seq_len tokens.
    let mut cache = FlatKVCache::new(config.clone());
    let k: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.001).collect();
    let v: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.002).collect();
    for _ in 0..seq_len {
        cache.append_kv(0, &k, &v);
        cache.advance();
    }

    let mut buf = vec![0.0f32; seq_len * kv_dim];

    let mut group = c.benchmark_group("kv_cache_f16/read_k_into");
    // Throughput in bytes of f32 output produced.
    group.throughput(Throughput::Bytes(
        (seq_len * kv_dim * std::mem::size_of::<f32>()) as u64,
    ));

    group.bench_function("f16_to_f32_256_tokens", |b| {
        b.iter(|| {
            cache.read_k_into(0, black_box(&mut buf));
            black_box(&buf);
        });
    });

    group.finish();
}

fn bench_memory_footprint(c: &mut Criterion) {
    // This group is a comparison metric, not a time benchmark.
    // `total_bytes()` is a pure arithmetic formula with no allocation; the
    // Criterion measurement shows the cost of the formula itself (sub-nanosecond),
    // while the assertions below verify the ADR-062 memory reduction requirement.
    let config_f16 = qwen3_06b_config(4096);
    let bytes_f16 = config_f16.total_bytes();
    let mb_f16 = bytes_f16 as f64 / (1024.0 * 1024.0);

    // The equivalent f32 footprint would be 2× larger (f32 = 4 bytes, f16 = 2 bytes).
    let bytes_f32 = bytes_f16 * 2;
    let mb_f32 = bytes_f32 as f64 / (1024.0 * 1024.0);

    // Verify the expected values match (ADR-062 requirement: ~448 MB for Qwen3-0.6B).
    assert!(
        (mb_f16 - 448.0).abs() < 1.0,
        "f16 KV cache should be ~448 MB for Qwen3-0.6B, got {mb_f16:.1} MB"
    );
    assert!(
        (mb_f32 - 896.0).abs() < 1.0,
        "f32 baseline should be ~896 MB for Qwen3-0.6B, got {mb_f32:.1} MB"
    );

    let mut group = c.benchmark_group("kv_cache_f16/memory");
    group.bench_function("qwen3_06b_total_bytes", |b| {
        b.iter(|| {
            let cfg = qwen3_06b_config(black_box(4096));
            black_box(cfg.total_bytes())
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_append_kv,
    bench_read_k_into,
    bench_memory_footprint
);
criterion_main!(benches);
