//! Benchmarks for gated attention primitives.
//!
//! Measures:
//! - `deinterleave_q_gate` throughput at Qwen3.5 and Qwen3-Next head shapes.
//! - `apply_sigmoid_gate` scalar vs SIMD throughput.
//! - Isolated gate application cost vs hypothetical non-gated baseline (zero-cost memcopy).
//!
//! Run with:
//!   cargo bench --bench gated_attention_bench
//!
//! Results are printed to stdout. Use `--nocapture` to see output when running
//! under `cargo test --bench`.

use lattice_inference::attention::gated::{
    apply_sigmoid_gate, apply_sigmoid_gate_scalar, deinterleave_q_gate,
};
use std::hint::black_box;
use std::time::Instant;

// ===================================================================
// Utilities
// ===================================================================

#[derive(Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        let unit = (self.next_u32() as f32) / (u32::MAX as f32);
        unit - 0.5
    }
}

fn random_vec(len: usize, rng: &mut Lcg) -> Vec<f32> {
    (0..len).map(|_| rng.next_f32()).collect()
}

fn iterations_for_size(n_floats: usize) -> usize {
    let bytes = n_floats * 4;
    match bytes {
        0..=4_096 => 10_000,
        4_097..=65_536 => 2_000,
        _ => 500,
    }
}

/// Run `f` for `iters` iterations and return average wall-clock time in microseconds.
fn average_us<F>(iters: usize, mut f: F) -> f64
where
    F: FnMut(),
{
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64
}

fn throughput_gb_s(bytes: usize, us: f64) -> f64 {
    (bytes as f64) / (us * 1e-6) / 1e9
}

// ===================================================================
// Model configurations
// ===================================================================

struct GatedAttnCase {
    name: &'static str,
    num_heads: usize,
    head_dim: usize,
}

const MODEL_CASES: &[GatedAttnCase] = &[
    GatedAttnCase {
        name: "Qwen3.5-0.6B",
        num_heads: 16,
        head_dim: 64,
    },
    GatedAttnCase {
        name: "Qwen3.5-1.5B",
        num_heads: 16,
        head_dim: 128,
    },
    GatedAttnCase {
        name: "Qwen3.5-7B",
        num_heads: 32,
        head_dim: 128,
    },
    GatedAttnCase {
        name: "Qwen3-Next",
        num_heads: 16,
        head_dim: 256,
    },
];

// ===================================================================
// Benchmark: deinterleave_q_gate
// ===================================================================

fn bench_deinterleave() {
    println!(
        "\n=== deinterleave_q_gate ===\n{:<18} {:>8} {:>10} {:>12} {:>12}",
        "model", "q_dim", "iters", "avg_us", "GB/s"
    );
    println!(
        "{:<18} {:>8} {:>10} {:>12} {:>12}",
        "------------------", "--------", "----------", "------------", "------------",
    );

    for case in MODEL_CASES {
        let q_dim = case.num_heads * case.head_dim;
        let packed_len = 2 * q_dim;
        let mut rng = Lcg::new((case.num_heads as u64) ^ ((case.head_dim as u64) << 16));
        let q_and_gate = random_vec(packed_len, &mut rng);
        let mut q_buf = vec![0.0f32; q_dim];
        let mut gate_buf = vec![0.0f32; q_dim];

        // Warm-up
        for _ in 0..100 {
            deinterleave_q_gate(
                black_box(&q_and_gate),
                black_box(&mut q_buf),
                black_box(&mut gate_buf),
                case.num_heads,
                case.head_dim,
            );
        }

        let iters = iterations_for_size(packed_len);
        let us = average_us(iters, || {
            deinterleave_q_gate(
                black_box(&q_and_gate),
                &mut q_buf,
                &mut gate_buf,
                case.num_heads,
                case.head_dim,
            );
            black_box(&q_buf);
            black_box(&gate_buf);
        });

        // Reads 2*q_dim floats, writes 2*q_dim floats.
        let bytes = 4 * q_dim * 4;
        let gb_s = throughput_gb_s(bytes, us);

        println!(
            "{:<18} {:>8} {:>10} {:>12.3} {:>12.2}",
            case.name, q_dim, iters, us, gb_s
        );
    }
}

// ===================================================================
// Benchmark: apply_sigmoid_gate — scalar vs SIMD
// ===================================================================

fn bench_sigmoid_gate() {
    println!(
        "\n=== apply_sigmoid_gate (scalar vs SIMD) ===\n{:<18} {:>8} {:>14} {:>14} {:>10}",
        "model", "q_dim", "scalar_us", "simd_us", "speedup"
    );
    println!(
        "{:<18} {:>8} {:>14} {:>14} {:>10}",
        "------------------", "--------", "--------------", "--------------", "----------",
    );

    for case in MODEL_CASES {
        let q_dim = case.num_heads * case.head_dim;
        let mut rng = Lcg::new(0xDEAD_BEEF ^ (q_dim as u64));
        let gate = random_vec(q_dim, &mut rng);
        let ctx_base = random_vec(q_dim, &mut rng);

        let iters = iterations_for_size(q_dim);

        // Scalar
        let us_scalar = {
            let mut ctx = ctx_base.clone();
            // Warm-up
            for _ in 0..50 {
                ctx.copy_from_slice(&ctx_base);
                apply_sigmoid_gate_scalar(black_box(&mut ctx), black_box(&gate));
            }
            average_us(iters, || {
                ctx.copy_from_slice(&ctx_base);
                apply_sigmoid_gate_scalar(black_box(&mut ctx), black_box(&gate));
                black_box(&ctx);
            })
        };

        // SIMD (dispatched)
        let us_simd = {
            let mut ctx = ctx_base.clone();
            // Warm-up
            for _ in 0..50 {
                ctx.copy_from_slice(&ctx_base);
                apply_sigmoid_gate(black_box(&mut ctx), black_box(&gate));
            }
            average_us(iters, || {
                ctx.copy_from_slice(&ctx_base);
                apply_sigmoid_gate(black_box(&mut ctx), black_box(&gate));
                black_box(&ctx);
            })
        };

        let speedup = us_scalar / us_simd;
        println!(
            "{:<18} {:>8} {:>14.3} {:>14.3} {:>10.2}x",
            case.name, q_dim, us_scalar, us_simd, speedup
        );
    }
}

// ===================================================================
// Benchmark: gated vs non-gated decode step overhead
// ===================================================================

/// Measure the overhead of the sigmoid gate step in isolation compared to a
/// hypothetical non-gated baseline (a plain `copy_from_slice`).
///
/// This gives the marginal compute cost that gating adds to the decode step.
fn bench_gate_overhead() {
    println!(
        "\n=== gate application overhead vs non-gated baseline ===\n{:<18} {:>8} {:>14} {:>14} {:>14}",
        "model", "q_dim", "copy_us", "gate_us", "overhead_us"
    );
    println!(
        "{:<18} {:>8} {:>14} {:>14} {:>14}",
        "------------------", "--------", "--------------", "--------------", "--------------",
    );

    for case in MODEL_CASES {
        let q_dim = case.num_heads * case.head_dim;
        let mut rng = Lcg::new(0xC0DE ^ (q_dim as u64));
        let gate = random_vec(q_dim, &mut rng);
        let ctx_base = random_vec(q_dim, &mut rng);
        let mut dst = vec![0.0f32; q_dim];

        let iters = iterations_for_size(q_dim);

        // Baseline: plain copy (what a non-gated model would do).
        let us_copy = average_us(iters, || {
            dst.copy_from_slice(black_box(&ctx_base));
            black_box(&dst);
        });

        // Gated path.
        let us_gate = {
            let mut ctx = ctx_base.clone();
            average_us(iters, || {
                ctx.copy_from_slice(&ctx_base);
                apply_sigmoid_gate(black_box(&mut ctx), black_box(&gate));
                black_box(&ctx);
            })
        };

        let overhead = (us_gate - us_copy).max(0.0);
        println!(
            "{:<18} {:>8} {:>14.3} {:>14.3} {:>14.3}",
            case.name, q_dim, us_copy, us_gate, overhead
        );
    }
}

// ===================================================================
// Entry point
// ===================================================================

fn run_benchmarks() {
    println!("Gated Attention Benchmark");
    println!("=========================");
    bench_deinterleave();
    bench_sigmoid_gate();
    bench_gate_overhead();
    println!();
}

pub fn main() {
    run_benchmarks();
}
