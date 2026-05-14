//! Benchmarks for differential attention (`apply_differential_attention`).
//!
//! Measures:
//! - `apply_differential_attention` throughput at representative sequence lengths.
//! - Cost of the differential mechanism, isolated against a single-softmax baseline
//!   that has the *same loop topology* (per-KV-head / per-pair) as the differential
//!   kernel. The difference is exactly: second Q/K extract + second GEMM + second
//!   softmax + subtract + sub-RMSNorm. This controls for kernel topology, which a
//!   comparison against the batched `apply_gqa_attention` would not.
//!
//! Run with:
//!   cargo bench --bench differential_attention_bench
//!
//! Results are printed to stdout.

use lattice_inference::attention::differential::{
    DiffAttnConfig, DiffAttnScratch, DiffLambdaParams, apply_differential_attention,
};
use lattice_inference::forward::cpu::matmul_bt;
use std::hint::black_box;
use std::time::Instant;

const MASK_VALUE: f32 = -10_000.0_f32;

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
// Single-softmax baseline — SAME loop topology as the differential kernel
//
// Mirrors apply_differential_attention's per-KV-head / per-pair structure
// exactly, but computes only ONE softmax map: no second Q/K extract, no
// second GEMM, no second softmax, no subtract, no sub-RMSNorm. The gap
// between this and the differential kernel is the mechanism cost, with
// loop topology held constant. (A comparison against the *batched*
// apply_gqa_attention would conflate topology with mechanism.)
// ===================================================================

#[derive(Default)]
struct SingleSoftmaxScratch {
    scores: Vec<f32>,
    v_head_t: Vec<f32>,
    q_packed: Vec<f32>,
    k_packed: Vec<f32>,
    context: Vec<f32>,
}

impl SingleSoftmaxScratch {
    fn reserve_for(&mut self, seq_len: usize, cfg: &DiffAttnConfig) {
        let v_head_dim = 2 * cfg.head_dim;
        self.scores.resize(seq_len * seq_len, 0.0);
        self.v_head_t.resize(v_head_dim * seq_len, 0.0);
        self.q_packed.resize(seq_len * cfg.head_dim, 0.0);
        self.k_packed.resize(seq_len * cfg.head_dim, 0.0);
        self.context.resize(seq_len * v_head_dim, 0.0);
    }
}

/// Single-softmax attention with the same per-KV-head / per-pair loop topology
/// as `apply_differential_attention`, minus the differential mechanism.
fn single_softmax_baseline(
    q_buf: &[f32],
    k_buf: &[f32],
    v_buf: &[f32],
    attn_out: &mut [f32],
    seq_len: usize,
    cfg: &DiffAttnConfig,
    scratch: &mut SingleSoftmaxScratch,
) {
    if seq_len == 0 {
        return;
    }
    scratch.reserve_for(seq_len, cfg);

    let head_dim = cfg.head_dim;
    let v_head_dim = 2 * head_dim;
    let n_rep = cfg.n_rep();
    let scale = (head_dim as f32).powf(-0.5);
    let out_scale = 1.0 - cfg.lambda_init();

    let q_row_stride = cfg.q_dim();
    let k_row_stride = cfg.k_dim();
    let v_row_stride = cfg.v_dim();
    let out_row_stride = cfg.out_dim();

    for kv_h in 0..cfg.num_kv_heads {
        // Extract + transpose V for this KV head.
        let v_head_offset = kv_h * v_head_dim;
        for pos in 0..seq_len {
            let src_off = pos * v_row_stride + v_head_offset;
            for d in 0..v_head_dim {
                scratch.v_head_t[d * seq_len + pos] = v_buf[src_off + d];
            }
        }

        let pair_start = kv_h * n_rep;
        for pair_h in pair_start..pair_start + n_rep {
            // Extract the FIRST packed Q head and FIRST packed K head only.
            let q_ph = 2 * pair_h;
            let k_ph = 2 * kv_h;
            for pos in 0..seq_len {
                let qs = pos * q_row_stride + q_ph * head_dim;
                scratch.q_packed[pos * head_dim..pos * head_dim + head_dim]
                    .copy_from_slice(&q_buf[qs..qs + head_dim]);
                let ks = pos * k_row_stride + k_ph * head_dim;
                scratch.k_packed[pos * head_dim..pos * head_dim + head_dim]
                    .copy_from_slice(&k_buf[ks..ks + head_dim]);
            }

            // One GEMM, scale + causal mask, one softmax.
            matmul_bt(
                &scratch.q_packed[..seq_len * head_dim],
                &scratch.k_packed[..seq_len * head_dim],
                &mut scratch.scores[..seq_len * seq_len],
                seq_len,
                head_dim,
                seq_len,
            );
            for qi in 0..seq_len {
                let row = &mut scratch.scores[qi * seq_len..(qi + 1) * seq_len];
                for (ki, v) in row.iter_mut().enumerate() {
                    if ki <= qi {
                        *v *= scale;
                    } else {
                        *v = MASK_VALUE;
                    }
                }
                let valid = qi + 1;
                let max_val = row[..valid]
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                for v in &mut row[..valid] {
                    *v = (*v - max_val).exp();
                    sum += *v;
                }
                if sum > 0.0 {
                    let inv = 1.0 / sum;
                    for v in &mut row[..valid] {
                        *v *= inv;
                    }
                }
                row[valid..].fill(0.0);
            }

            // One GEMM with V, scale, write out.
            matmul_bt(
                &scratch.scores[..seq_len * seq_len],
                &scratch.v_head_t[..v_head_dim * seq_len],
                &mut scratch.context[..seq_len * v_head_dim],
                seq_len,
                seq_len,
                v_head_dim,
            );
            for pos in 0..seq_len {
                let src = pos * v_head_dim;
                let dst = pos * out_row_stride + pair_h * v_head_dim;
                for d in 0..v_head_dim {
                    attn_out[dst + d] = scratch.context[src + d] * out_scale;
                }
            }
        }
    }
}

// ===================================================================
// Benchmark: differential mechanism overhead (topology-controlled)
// ===================================================================

fn bench_mechanism_overhead() {
    println!(
        "\n=== differential mechanism overhead (vs same-topology single-softmax baseline) ===\n{:<26} {:>8} {:>14} {:>14} {:>10}",
        "model", "seq_len", "baseline_us", "diff_us", "overhead"
    );
    println!(
        "{:<26} {:>8} {:>14} {:>14} {:>10}",
        "--------------------------", "--------", "--------------", "--------------", "----------",
    );

    const BENCH_SEQ: usize = 256;

    for &(name, num_heads, num_kv_heads, head_dim, layer_depth) in MODEL_CASES {
        let cfg = DiffAttnConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            layer_depth,
        };

        let seq_len = BENCH_SEQ;
        let mut rng = Lcg::new((seq_len as u64) ^ 0xDEAD_BEEF);

        // Shared inputs — both kernels read the same Q/K/V buffers.
        let q = random_vec(seq_len * cfg.q_dim(), &mut rng);
        let k = random_vec(seq_len * cfg.k_dim(), &mut rng);
        let v = random_vec(seq_len * cfg.v_dim(), &mut rng);
        let params = DiffLambdaParams {
            lambda_q1: vec![0.0; head_dim],
            lambda_k1: vec![0.0; head_dim],
            lambda_q2: vec![0.0; head_dim],
            lambda_k2: vec![0.0; head_dim],
        };
        let subln_w = vec![1.0_f32; 2 * head_dim];

        let mut diff_out = vec![0.0_f32; seq_len * cfg.out_dim()];
        let mut diff_scratch = DiffAttnScratch::default();
        let mut base_out = vec![0.0_f32; seq_len * cfg.out_dim()];
        let mut base_scratch = SingleSoftmaxScratch::default();

        // Warm-up both.
        for _ in 0..20 {
            apply_differential_attention(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(&params),
                black_box(&subln_w),
                1e-6,
                black_box(&mut diff_out),
                seq_len,
                &cfg,
                &mut diff_scratch,
            );
            single_softmax_baseline(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(&mut base_out),
                seq_len,
                &cfg,
                &mut base_scratch,
            );
        }

        let iters = iterations_for_seq(seq_len);

        let us_base = average_us(iters, || {
            single_softmax_baseline(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(&mut base_out),
                seq_len,
                &cfg,
                &mut base_scratch,
            );
            black_box(&base_out);
        });

        let us_diff = average_us(iters, || {
            apply_differential_attention(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                black_box(&params),
                black_box(&subln_w),
                1e-6,
                black_box(&mut diff_out),
                seq_len,
                &cfg,
                &mut diff_scratch,
            );
            black_box(&diff_out);
        });

        let overhead = us_diff / us_base;
        println!("{name:<26} {seq_len:>8} {us_base:>14.3} {us_diff:>14.3} {overhead:>9.2}x");
    }
}

// ===================================================================
// Entry point
// ===================================================================

fn main() {
    println!("Differential Attention Benchmark");
    println!("=================================");
    bench_diff_attn();
    bench_mechanism_overhead();
    println!();
}
