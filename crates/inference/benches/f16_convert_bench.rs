//! Criterion micro-bench for the f16->f32 row conversion used in the embedding
//! lookup hot path (`forward_step_inner`, `forward_step_gdn_only`, prefill).
//!
//! Measures at N=1024 (the Qwen3.5-0.8B hidden size):
//! - `scalar`: the crate's production scalar decoder
//!   (`lattice_inference::weights::bench_support::f16_bits_to_f32`, which
//!   `weights/mod.rs` re-exports behind `bench-internals` from the single
//!   always-compiled `weights::half_bits` module every load/quant call site
//!   delegates to — lattice#799). This bench previously carried its own
//!   copied decoder rather than exercising that migrated path, so its
//!   numbers said nothing about the consolidation. Requires `--features bench-internals`.
//!
//! A NEON fast path (`vcvt_f32_f16`, 4 lanes/instruction) was removed in #568: it
//! required the nightly-only `stdarch_neon_f16` feature and broke stable-toolchain
//! builds. `convert_f16_row` in `metal_qwen35.rs` is scalar-only for the same reason.
//!
//! Run: `cargo bench -p lattice-inference --bench f16_convert_bench --features bench-internals`
//!
//! ADR-058: every perf PR must include before/after bench output.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_inference::weights::bench_support::f16_bits_to_f32;
use std::time::Duration;

/// Convert n f16 values (as u16 bits) at `src` into f32 values at `dst`,
/// calling the production decoder for every element.
///
/// # Safety
/// `src` must point to at least `n` initialized u16 values; `dst` to at least `n` writable f32.
unsafe fn convert_f16_row_scalar(src: *const u16, dst: *mut f32, n: usize) {
    for i in 0..n {
        *dst.add(i) = f16_bits_to_f32(*src.add(i));
    }
}

// --------------------------------------------------------------------------
// Bench harness
// --------------------------------------------------------------------------

fn make_f16_inputs(n: usize) -> Vec<u16> {
    // Deterministic LCG — representative mix of normal, subnormal, and zero values.
    let mut state = 0x1234_5678u32;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            // Map to f16 normal range (avoid all-ones exponent for simplicity).
            (state as u16) & 0x7bff
        })
        .collect()
}

fn bench_f16_convert(c: &mut Criterion) {
    const N: usize = 1024;
    let src = make_f16_inputs(N);
    let mut dst = vec![0.0f32; N];

    let mut group = c.benchmark_group("f16_convert_row");
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(N as u64));

    group.bench_with_input(BenchmarkId::new("scalar", N), &N, |b, &_n| {
        b.iter(|| {
            // SAFETY: src and dst are N-element buffers; loop is inbounds.
            unsafe {
                convert_f16_row_scalar(black_box(src.as_ptr()), black_box(dst.as_mut_ptr()), N);
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_f16_convert);
criterion_main!(benches);
