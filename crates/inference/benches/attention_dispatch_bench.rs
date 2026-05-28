//! Benchmark: `AttentionKind` enum dispatch overhead (ADR-059).
//!
//! Measures the cost of a `match` on `AttentionKind` vs a direct function call.
//! The workload inside each branch is a trivial `black_box(0.0f32)` return so
//! that the measurement isolates dispatch cost rather than attention math.
//!
//! Acceptance threshold: < 2.5 ns per dispatch (ADR-059 gate).

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use lattice_inference::attention::{AttentionKind, gqa::GqaConfig};

// ---------------------------------------------------------------------------
// Mock workload — purely measures dispatch, not computation
// ---------------------------------------------------------------------------

/// Simulate the outcome of selecting an attention kernel.
/// Returns a raw sentinel f32 per variant — the caller black_boxes the input
/// and output so per-arm opaque barriers don't inflate the dispatch measurement.
#[inline(never)]
fn dispatch_kind(kind: &AttentionKind) -> f32 {
    match kind {
        AttentionKind::Mha => 0.0f32,
        AttentionKind::Gqa(_) => 1.0f32,
        AttentionKind::Flash => 2.0f32,
        AttentionKind::FlashCausal => 3.0f32,
        AttentionKind::Gdn => 4.0f32,
        AttentionKind::GdnFused => 5.0f32,
        AttentionKind::GatedGqa => 6.0f32,
        AttentionKind::Differential => 7.0f32,
        AttentionKind::NativeSparse => 8.0f32,
        AttentionKind::Decode => 9.0f32,
    }
}

/// Baseline: direct call with no dispatch at all.
#[inline(never)]
fn direct_call(x: f32) -> f32 {
    black_box(x)
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_dispatch_vs_direct(c: &mut Criterion) {
    let cfg = GqaConfig {
        num_heads: 16,
        num_kv_heads: 8,
        head_dim: 64,
    };

    // All 10 variants — cycle through to prevent branch-predictor speculation
    // from hiding true dispatch cost.
    let kinds: Vec<AttentionKind> = vec![
        AttentionKind::Mha,
        AttentionKind::Gqa(cfg),
        AttentionKind::Flash,
        AttentionKind::FlashCausal,
        AttentionKind::Gdn,
        AttentionKind::GdnFused,
        AttentionKind::GatedGqa,
        AttentionKind::Differential,
        AttentionKind::NativeSparse,
        AttentionKind::Decode,
    ];

    let mut group = c.benchmark_group("attention_dispatch");

    // --- enum match dispatch ---
    group.bench_function("enum_match_all_variants", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let kind = &kinds[i % kinds.len()];
            i = i.wrapping_add(1);
            // Black-box the input so the compiler cannot specialize dispatch on
            // a known-constant enum variant. Black-box the output to prevent
            // dead-code elimination of the result.
            let k = black_box(kind);
            let result = dispatch_kind(k);
            black_box(result);
        });
    });

    // --- direct call baseline ---
    group.bench_function("direct_call_baseline", |b| {
        let mut i = 0u32;
        b.iter(|| {
            let x = (i % 10) as f32;
            i = i.wrapping_add(1);
            black_box(direct_call(x))
        });
    });

    // --- name() accessor overhead ---
    group.bench_function("name_accessor_all_variants", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let kind = &kinds[i % kinds.len()];
            i = i.wrapping_add(1);
            black_box(kind.name())
        });
    });

    // --- is_causal() overhead ---
    group.bench_function("is_causal_all_variants", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let kind = &kinds[i % kinds.len()];
            i = i.wrapping_add(1);
            black_box(kind.is_causal())
        });
    });

    // --- supports_kv_cache() overhead ---
    group.bench_function("supports_kv_cache_all_variants", |b| {
        let mut i = 0usize;
        b.iter(|| {
            let kind = &kinds[i % kinds.len()];
            i = i.wrapping_add(1);
            black_box(kind.supports_kv_cache())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_dispatch_vs_direct);
criterion_main!(benches);
