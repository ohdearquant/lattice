//! Criterion micro-bench for the f16->f32 row conversion used in the embedding
//! lookup hot path (`forward_step_inner`, `forward_step_gdn_only`, prefill).
//!
//! Measures at N=1024 (the Qwen3.5-0.8B hidden size):
//! - `scalar`: hand-written IEEE-754 bit-manipulation loop (~10 ops/element)
//!
//! A NEON fast path (`vcvt_f32_f16`, 4 lanes/instruction) was removed in #568: it
//! required the nightly-only `stdarch_neon_f16` feature and broke stable-toolchain
//! builds. `convert_f16_row` in `metal_qwen35.rs` is scalar-only for the same reason.
//!
//! Run: `cargo bench -p lattice-inference --bench f16_convert_bench`
//!
//! ADR-058: every perf PR must include before/after bench output.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::Duration;

// --------------------------------------------------------------------------
// Scalar reference (mirrors the hand-written f16_to_f32 in metal_qwen35.rs)
// --------------------------------------------------------------------------

#[inline(always)]
fn f16_to_f32_scalar(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 0x1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;

    let f32_bits = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, _) => {
            let mut mant = frac;
            let mut e = -14i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x03ff;
            (sign << 31) | (((e + 127) as u32) << 23) | (mant << 13)
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, _) => (sign << 31) | 0x7f80_0000 | (frac << 13),
        _ => (sign << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (frac << 13),
    };
    f32::from_bits(f32_bits)
}

/// Convert n f16 values (as u16 bits) at `src` into f32 values at `dst`.
///
/// # Safety
/// `src` must point to at least `n` initialized u16 values; `dst` to at least `n` writable f32.
unsafe fn convert_f16_row_scalar(src: *const u16, dst: *mut f32, n: usize) {
    for i in 0..n {
        *dst.add(i) = f16_to_f32_scalar(*src.add(i));
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
