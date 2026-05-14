//! Benchmarks for differential attention (`apply_differential_attention`).
//!
//! Measures:
//! - `apply_differential_attention` throughput at representative sequence lengths.
//! - Overhead of the differential mechanism vs a plain GQA-style single-softmax baseline
//!   (the cost of the second softmax + subtract + sub-RMSNorm).
//!
//! Run with:
//!   cargo bench --bench differential_attention_bench
//!
//! Results are printed to stdout.

use lattice_inference::attention::differential::{
    DiffAttnConfig, DiffAttnScratch, DiffLambdaParams, apply_differential_attention,
};
use lattice_inference::attention::gqa::{GqaConfig, GqaScratch, apply_gqa_attention};
use std::hint::black_box;
use std::time::Instant;

// ===================================================================
// Utilities
// ===================================================================

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
        (self.next_u32() as f32) / (u32::MAX as f32) - 0.5
    }
}

fn random_vec(len: usize, rng: &mut Lcg) -> Vec<f32> {
    (0..len).map(|_| rng.next_f32()).collect()
}

fn iterations_for_seq(seq_len: usize) -> usize {
    match seq_len {
        0..=64 => 2_000,
        65..=256 => 500,
        _ => 100,
    }
}

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

// ===================================================================
// Benchmark shapes
// ===================================================================

/// (name, num_heads, num_kv_heads, head_dim, layer_depth)
const MODEL_CASES: &[(&str, usize, usize, usize, usize)] = &[
    // Small/toy
    ("diff-2h-mha-hd8", 2, 2, 8, 1),
    // Medium — analogous to a 4-head GQA config
    ("diff-4h-gqa-2kv-hd16", 4, 2, 16, 3),
    // Larger, comparable to small transformer layers
    ("diff-8h-gqa-2kv-hd32", 8, 2, 32, 6),
];

const SEQ_LENS: &[usize] = &[64, 256, 512];

// ===================================================================
// Benchmark: apply_differential_attention throughput
// ===================================================================

fn bench_diff_attn() {
    println!(
        "\n=== apply_differential_attention ===\n{:<26} {:>8} {:>12} {:>12}",
        "model", "seq_len", "avg_us", "tok/ms"
    );
    println!(
        "{:<26} {:>8} {:>12} {:>12}",
        "--------------------------", "--------", "------------", "------------",
    );

    for &(name, num_heads, num_kv_heads, head_dim, layer_depth) in MODEL_CASES {
        let cfg = DiffAttnConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            layer_depth,
        };

        for &seq_len in SEQ_LENS {
            let mut rng = Lcg::new((seq_len as u64) ^ ((head_dim as u64) << 16));

            let q = random_vec(seq_len * cfg.q_dim(), &mut rng);
            let k = random_vec(seq_len * cfg.k_dim(), &mut rng);
            let v = random_vec(seq_len * cfg.v_dim(), &mut rng);
            let params = DiffLambdaParams {
                lambda_q1: random_vec(head_dim, &mut rng)
                    .into_iter()
                    .map(|x| x * 0.1)
                    .collect(),
                lambda_k1: random_vec(head_dim, &mut rng)
                    .into_iter()
                    .map(|x| x * 0.1)
                    .collect(),
                lambda_q2: random_vec(head_dim, &mut rng)
                    .into_iter()
                    .map(|x| x * 0.1)
                    .collect(),
                lambda_k2: random_vec(head_dim, &mut rng)
                    .into_iter()
                    .map(|x| x * 0.1)
                    .collect(),
            };
            let subln_w = vec![1.0_f32; 2 * head_dim];
            let mut out = vec![0.0_f32; seq_len * cfg.out_dim()];
            let mut scratch = DiffAttnScratch::default();

            // Warm-up
            for _ in 0..20 {
                apply_differential_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    black_box(&params),
                    black_box(&subln_w),
                    1e-6,
                    black_box(&mut out),
                    seq_len,
                    &cfg,
                    &mut scratch,
                );
            }

            let iters = iterations_for_seq(seq_len);
            let us = average_us(iters, || {
                apply_differential_attention(
                    black_box(&q),
                    black_box(&k),
                    black_box(&v),
                    black_box(&params),
                    black_box(&subln_w),
                    1e-6,
                    black_box(&mut out),
                    seq_len,
                    &cfg,
                    &mut scratch,
                );
                black_box(&out);
            });

            let tok_per_ms = seq_len as f64 / us * 1_000.0;
            println!("{name:<26} {seq_len:>8} {us:>12.3} {tok_per_ms:>12.1}");
        }
    }
}

// ===================================================================
// Benchmark: differential vs GQA baseline overhead
//
// GQA attention (apply_gqa_attention) does one softmax per head-pair.
// Differential attention does two softmax passes per head-pair + subtract + subln.
// This section shows the marginal cost of the differential mechanism.
//
// NOTE: GQA and differential have different buffer layouts (GQA uses the
// standard embed_dim = num_heads*head_dim; differential uses 2*num_heads
// packed heads). We compare them at the same "effective" model size:
// a GQA config with (2*num_heads, num_kv_heads, head_dim) covering the same
// parameter count as the differential config's Q/K projections.
// ===================================================================

fn bench_overhead_vs_gqa() {
    println!(
        "\n=== overhead: differential vs GQA-baseline (same Q/K projection size) ===\n{:<26} {:>8} {:>14} {:>14} {:>10}",
        "model", "seq_len", "gqa_us", "diff_us", "overhead"
    );
    println!(
        "{:<26} {:>8} {:>14} {:>14} {:>10}",
        "--------------------------", "--------", "--------------", "--------------", "----------",
    );

    // Only benchmark at seq_len=256 to keep output concise
    const BENCH_SEQ: usize = 256;

    for &(name, num_heads, num_kv_heads, head_dim, layer_depth) in MODEL_CASES {
        let diff_cfg = DiffAttnConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            layer_depth,
        };

        // GQA with the same packed head count (2*num_heads Q heads, same KV heads)
        // and same head_dim. This matches the same Q/K projection sizes.
        let gqa_cfg = GqaConfig {
            num_heads: 2 * num_heads,       // packed heads
            num_kv_heads: 2 * num_kv_heads, // packed KV heads
            head_dim,
        };

        let seq_len = BENCH_SEQ;
        let mut rng = Lcg::new((seq_len as u64) ^ 0xDEAD_BEEF);

        // Differential inputs
        let dq = random_vec(seq_len * diff_cfg.q_dim(), &mut rng);
        let dk = random_vec(seq_len * diff_cfg.k_dim(), &mut rng);
        let dv = random_vec(seq_len * diff_cfg.v_dim(), &mut rng);
        let params = DiffLambdaParams {
            lambda_q1: vec![0.0; head_dim],
            lambda_k1: vec![0.0; head_dim],
            lambda_q2: vec![0.0; head_dim],
            lambda_k2: vec![0.0; head_dim],
        };
        let subln_w = vec![1.0_f32; 2 * head_dim];
        let mut dout = vec![0.0_f32; seq_len * diff_cfg.out_dim()];
        let mut dscratch = DiffAttnScratch::default();

        // GQA inputs (same total Q/K/V bytes as differential)
        let gq = random_vec(seq_len * gqa_cfg.q_dim(), &mut rng);
        let gk = random_vec(seq_len * gqa_cfg.kv_dim(), &mut rng);
        let gv = random_vec(seq_len * gqa_cfg.kv_dim(), &mut rng);
        let mut gout = vec![0.0_f32; seq_len * gqa_cfg.q_dim()];
        let mut gscratch = GqaScratch::default();

        // Warm-up both
        for _ in 0..20 {
            apply_differential_attention(
                black_box(&dq),
                black_box(&dk),
                black_box(&dv),
                black_box(&params),
                black_box(&subln_w),
                1e-6,
                black_box(&mut dout),
                seq_len,
                &diff_cfg,
                &mut dscratch,
            );
            apply_gqa_attention(
                black_box(&gq),
                black_box(&gk),
                black_box(&gv),
                black_box(&mut gout),
                seq_len,
                gqa_cfg,
                &mut gscratch,
            );
        }

        let iters = iterations_for_seq(seq_len);

        let us_gqa = average_us(iters, || {
            apply_gqa_attention(
                black_box(&gq),
                black_box(&gk),
                black_box(&gv),
                black_box(&mut gout),
                seq_len,
                gqa_cfg,
                &mut gscratch,
            );
            black_box(&gout);
        });

        let us_diff = average_us(iters, || {
            apply_differential_attention(
                black_box(&dq),
                black_box(&dk),
                black_box(&dv),
                black_box(&params),
                black_box(&subln_w),
                1e-6,
                black_box(&mut dout),
                seq_len,
                &diff_cfg,
                &mut dscratch,
            );
            black_box(&dout);
        });

        let overhead = us_diff / us_gqa;
        println!("{name:<26} {seq_len:>8} {us_gqa:>14.3} {us_diff:>14.3} {overhead:>9.2}x");
    }
}

// ===================================================================
// Entry point
// ===================================================================

fn main() {
    println!("Differential Attention Benchmark");
    println!("=================================");
    bench_diff_attn();
    bench_overhead_vs_gqa();
    println!();
}
