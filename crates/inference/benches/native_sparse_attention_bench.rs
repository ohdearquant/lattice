//! Benchmarks for Native Sparse Attention (`apply_native_sparse_attention`).
//!
//! **This is a noisy, unstabilized wall-clock microbenchmark.** Results vary
//! run-to-run (typically ±10–30%) depending on CPU frequency scaling, thermal
//! state, OS scheduling, and cache temperature. Do NOT read the printed numbers
//! as stable facts; compare them as rough ranges across multiple runs.
//!
//! Measures:
//! - `apply_native_sparse_attention` throughput at representative sequence
//!   lengths (512, 2048, 8192). 64K is excluded: an unoptimized CPU prefill
//!   kernel at that length would take tens of seconds per run.
//! - **Sparse-vs-dense comparison.** A dense causal attention baseline with the
//!   *same per-token / per-head loop topology* is run at identical shapes.
//!   Per ADR-042 §Risks, the paper's "2–9×" is a GPU/FlashAttention-2 number;
//!   NSA may be SLOWER than dense at Lattice's current CPU targets. The
//!   ratio is printed as-is, honest whichever way it comes out.
//!
//! Run with:
//!   cargo bench -p lattice-inference --bench native_sparse_attention_bench
//!
//! For multiple-run variance, invoke several times and note the range of
//! the `nsa_us` and `ratio` columns.

use lattice_inference::attention::native_sparse::{
    NsaConfig, NsaScratch, NsaWeights, apply_native_sparse_attention,
};
use std::hint::black_box;
use std::time::Instant;

// ===================================================================
// Utilities
// ===================================================================

/// Linear congruential PRNG — same params as `differential_attention_bench.rs`.
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

/// Scale a vector by `s` in-place (keeps magnitudes small so φ outputs and
/// gate logits stay in a reasonable range).
fn scaled(mut v: Vec<f32>, s: f32) -> Vec<f32> {
    for x in &mut v {
        *x *= s;
    }
    v
}

/// Average microseconds per call, measured over `iters` iterations.
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

/// Iteration count scaled to sequence length so the bench finishes in
/// reasonable wall-clock time for an unoptimized CPU prefill kernel.
fn iterations_for_seq(seq_len: usize) -> usize {
    match seq_len {
        0..=512 => 20,
        513..=2048 => 5,
        _ => 1,
    }
}

// ===================================================================
// NSA config — paper defaults scaled to a benchmarkable size
//
// Paper §4.1 defaults: l=32, d=16, l'=64, n=16, w=512.
// Divisibility requirements: d | l (16 | 32 ✓) and d | l' (16 | 64 ✓).
// We keep these exact paper defaults but use a small head count and
// head_dim so the total per-call work is measurable at seq_len ≤ 8192.
// ===================================================================

/// A realistic small model config derived from the paper's §4.1 defaults.
/// `l=32, d=16, l'=64, n=16, w=512` are unchanged; heads are scaled down.
fn bench_cfg() -> NsaConfig {
    NsaConfig {
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        compress_block: 32,  // l=32
        compress_stride: 16, // d=16 (d | l: 32 % 16 == 0 ✓)
        select_block: 64,    // l'=64 (d | l': 64 % 16 == 0 ✓)
        num_selected: 16,    // n=16
        window: 512,         // w=512
    }
}

/// Construct `NsaWeights` with small-magnitude random values (×0.1 scale).
/// Small magnitudes keep the φ ReLU active and gate logits in a reasonable
/// range, avoiding degenerate all-zero or near-infinity outputs.
fn make_weights(cfg: &NsaConfig, model_dim: usize, seed: u64) -> NsaWeights {
    let phi_in = cfg.phi_in();
    let head_dim = cfg.head_dim;
    let num_kv_heads = cfg.num_kv_heads;
    let l = cfg.compress_block;
    let mut rng = Lcg::new(seed);

    let rv = |len: usize, rng: &mut Lcg| scaled(random_vec(len, rng), 0.1);

    NsaWeights {
        phi_k_w1: rv(phi_in * phi_in, &mut rng),
        phi_k_b1: rv(phi_in, &mut rng),
        phi_k_w2: rv(head_dim * phi_in, &mut rng),
        phi_k_b2: rv(head_dim, &mut rng),
        phi_v_w1: rv(phi_in * phi_in, &mut rng),
        phi_v_b1: rv(phi_in, &mut rng),
        phi_v_w2: rv(head_dim * phi_in, &mut rng),
        phi_v_b2: rv(head_dim, &mut rng),
        k_intrablock_pos: rv(num_kv_heads * l * head_dim, &mut rng),
        v_intrablock_pos: rv(num_kv_heads * l * head_dim, &mut rng),
        g_proj_w: rv(3 * cfg.num_heads * model_dim, &mut rng),
        g_proj_b: rv(3 * cfg.num_heads, &mut rng),
    }
}

// ===================================================================
// Sequence lengths
//
// 512   — short context, well within the sliding window.
// 2048  — medium; selection branch starts selecting multiple blocks.
// 8192  — long; stresses the compression branch.
// 64K is excluded: unoptimized CPU prefill at that length would take
// tens of seconds per iteration, making the bench unusable in CI.
// ===================================================================

const SEQ_LENS: &[usize] = &[512, 2048, 8192];

// ===================================================================
// Benchmark 1: NSA throughput
// ===================================================================

fn bench_nsa_throughput() {
    println!(
        "\n=== apply_native_sparse_attention throughput ===\
         \n(noisy wall-clock microbenchmark; ±10-30% run-to-run variance is normal)\
         \n{:<10} {:>8} {:>12} {:>12}",
        "cfg", "seq_len", "avg_us", "tok/ms"
    );
    println!(
        "{:<10} {:>8} {:>12} {:>12}",
        "----------", "--------", "------------", "------------",
    );

    let cfg = bench_cfg();
    // model_dim: use q_dim (same as the test helper in native_sparse.rs).
    let model_dim = cfg.q_dim();

    for &seq_len in SEQ_LENS {
        let seed = (seq_len as u64) ^ 0xCAFE_BABE_1234_5678;
        let weights = make_weights(&cfg, model_dim, seed);

        let mut rng = Lcg::new(seed.wrapping_add(1));
        // Independent K/V per branch (paper §3.3.3): the caller projects K and V
        // three times. Distinct random buffers mirror that.
        let q = scaled(random_vec(seq_len * cfg.q_dim(), &mut rng), 0.1);
        let q_rope = scaled(random_vec(seq_len * cfg.q_dim(), &mut rng), 0.1);
        let k_cmp = scaled(random_vec(seq_len * cfg.kv_dim(), &mut rng), 0.1);
        let k_slc = scaled(random_vec(seq_len * cfg.kv_dim(), &mut rng), 0.1);
        let k_win = scaled(random_vec(seq_len * cfg.kv_dim(), &mut rng), 0.1);
        let v_cmp = scaled(random_vec(seq_len * cfg.kv_dim(), &mut rng), 0.1);
        let v_slc = scaled(random_vec(seq_len * cfg.kv_dim(), &mut rng), 0.1);
        let v_win = scaled(random_vec(seq_len * cfg.kv_dim(), &mut rng), 0.1);
        let x = scaled(random_vec(seq_len * model_dim, &mut rng), 0.1);
        let mut out = vec![0.0_f32; seq_len * cfg.q_dim()];
        let mut scratch = NsaScratch::default();

        // Warm-up: prime the scratch allocations and instruction caches.
        for _ in 0..3 {
            apply_native_sparse_attention(
                black_box(&q),
                black_box(&q_rope),
                black_box(&k_cmp),
                black_box(&k_slc),
                black_box(&k_win),
                black_box(&v_cmp),
                black_box(&v_slc),
                black_box(&v_win),
                black_box(&x),
                black_box(&weights),
                black_box(&mut out),
                seq_len,
                &cfg,
                &mut scratch,
            );
        }

        let iters = iterations_for_seq(seq_len);
        let us = average_us(iters, || {
            apply_native_sparse_attention(
                black_box(&q),
                black_box(&q_rope),
                black_box(&k_cmp),
                black_box(&k_slc),
                black_box(&k_win),
                black_box(&v_cmp),
                black_box(&v_slc),
                black_box(&v_win),
                black_box(&x),
                black_box(&weights),
                black_box(&mut out),
                seq_len,
                &cfg,
                &mut scratch,
            );
            black_box(&out);
        });

        let tok_per_ms = seq_len as f64 / us * 1_000.0;
        println!(
            "{:<10} {:>8} {:>12.1} {:>12.1}",
            "nsa-paper", seq_len, us, tok_per_ms
        );
    }
}

// ===================================================================
// Dense causal attention baseline
//
// Same per-token / per-head loop topology as the NSA kernel: iterates
// over query positions qt=0..seq_len in an outer loop, then over KV
// heads, then over Q heads per KV head. This mirrors the structure of
// Steps 2d (sliding window) but extends the window to the full causal
// context (w = seq_len), giving standard full dense causal attention.
//
// This controls for the loop topology so the sparse-vs-dense ratio
// measures the net mechanism cost — φ MLP compression, three-branch
// construction, and the gated merge — rather than loop overhead
// differences. A comparison against a batched GEMM-based attention
// would conflate loop topology with mechanism cost.
//
// Note: this baseline has O(seq_len²) complexity, just like standard
// dense attention. NSA's compression and selection branches have lower
// complexity, but the sliding-window branch is O(seq_len × w) and the
// φ MLP is O(seq_len × phi_in²). Whether NSA is faster or slower on
// CPU depends on which term dominates — ADR-042 §Risks flags this as
// unproven. The printed ratio reflects the honest measurement.
// ===================================================================

#[derive(Default)]
struct DenseScratch {
    /// Causal attention scores per query: `[seq_len]` (reused per qt/head).
    scores: Vec<f32>,
}

impl DenseScratch {
    fn reserve_for(&mut self, seq_len: usize) {
        self.scores.resize(seq_len, 0.0);
    }
}

/// Dense causal attention with the same per-token / per-KV-head / per-Q-head
/// loop topology as `apply_native_sparse_attention`. No compression, no
/// selection, no gating — just one softmax attention map over the full causal
/// context per head per query token.
fn dense_causal_baseline(
    q_buf: &[f32],
    k_buf: &[f32],
    v_buf: &[f32],
    out: &mut [f32],
    seq_len: usize,
    cfg: &NsaConfig,
    scratch: &mut DenseScratch,
) {
    if seq_len == 0 {
        return;
    }
    scratch.reserve_for(seq_len);

    let head_dim = cfg.head_dim;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let n_rep = cfg.n_rep();
    let scale = (head_dim as f32).powf(-0.5);

    out.fill(0.0);

    for qt in 0..seq_len {
        for kv_h in 0..cfg.num_kv_heads {
            let q_head_start = kv_h * n_rep;
            for qh_local in 0..n_rep {
                let qh = q_head_start + qh_local;
                let q_off = qt * q_dim + qh * head_dim;
                let q_head = &q_buf[q_off..q_off + head_dim];

                // Scores: dot(q, k[pos]) * scale for pos=0..=qt (causal).
                let scores = &mut scratch.scores[..qt + 1];
                for pos in 0..=qt {
                    let k_off = pos * kv_dim + kv_h * head_dim;
                    let dot: f32 = q_head
                        .iter()
                        .zip(k_buf[k_off..k_off + head_dim].iter())
                        .map(|(&a, &b)| a * b)
                        .sum();
                    scores[pos] = dot * scale;
                }

                // Numerically stable softmax over the causally valid prefix.
                let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_val).exp();
                    sum += *s;
                }
                if sum > 0.0 {
                    let inv = 1.0 / sum;
                    for s in scores.iter_mut() {
                        *s *= inv;
                    }
                }

                // Accumulate weighted V into output.
                let out_off = qt * q_dim + qh * head_dim;
                for pos in 0..=qt {
                    let p = scores[pos];
                    let v_off = pos * kv_dim + kv_h * head_dim;
                    for dd in 0..head_dim {
                        out[out_off + dd] += p * v_buf[v_off + dd];
                    }
                }
            }
        }
    }
}

// ===================================================================
// Benchmark 2: sparse vs dense comparison
// ===================================================================

fn bench_sparse_vs_dense() {
    println!(
        "\n=== NSA vs dense causal attention (same loop topology) ===\
         \n(ADR-042 §Risks: the paper's 2-9x is a GPU/FlashAttention-2 number;\
         \n CPU ratio may be <1 — this reports the honest result either way)\
         \n(noisy microbenchmark; run multiple times and note the range of 'ratio')\
         \n{:<10} {:>8} {:>12} {:>12} {:>10}",
        "cfg", "seq_len", "dense_us", "nsa_us", "ratio"
    );
    println!(
        "{:<10} {:>8} {:>12} {:>12} {:>10}",
        "----------", "--------", "------------", "------------", "----------",
    );

    let cfg = bench_cfg();
    let model_dim = cfg.q_dim();

    for &seq_len in SEQ_LENS {
        let seed = (seq_len as u64) ^ 0xDEAD_BEEF_CAFE_1234;
        let weights = make_weights(&cfg, model_dim, seed);

        let mut rng = Lcg::new(seed.wrapping_add(7));
        // NSA takes 8 activation buffers (independent K/V per branch, paper §3.3.3).
        // The dense baseline is single-branch by definition; it borrows the window
        // branch's RoPE'd Q/K/V (`q_rope`, `k_win`, `v_win`) — dense causal is the
        // sliding window extended to the full context.
        let q = scaled(random_vec(seq_len * cfg.q_dim(), &mut rng), 0.1);
        let q_rope = scaled(random_vec(seq_len * cfg.q_dim(), &mut rng), 0.1);
        let k_cmp = scaled(random_vec(seq_len * cfg.kv_dim(), &mut rng), 0.1);
        let k_slc = scaled(random_vec(seq_len * cfg.kv_dim(), &mut rng), 0.1);
        let k_win = scaled(random_vec(seq_len * cfg.kv_dim(), &mut rng), 0.1);
        let v_cmp = scaled(random_vec(seq_len * cfg.kv_dim(), &mut rng), 0.1);
        let v_slc = scaled(random_vec(seq_len * cfg.kv_dim(), &mut rng), 0.1);
        let v_win = scaled(random_vec(seq_len * cfg.kv_dim(), &mut rng), 0.1);
        let x = scaled(random_vec(seq_len * model_dim, &mut rng), 0.1);

        let mut nsa_out = vec![0.0_f32; seq_len * cfg.q_dim()];
        let mut nsa_scratch = NsaScratch::default();
        let mut dense_out = vec![0.0_f32; seq_len * cfg.q_dim()];
        let mut dense_scratch = DenseScratch::default();

        // Warm-up both kernels.
        for _ in 0..3 {
            apply_native_sparse_attention(
                black_box(&q),
                black_box(&q_rope),
                black_box(&k_cmp),
                black_box(&k_slc),
                black_box(&k_win),
                black_box(&v_cmp),
                black_box(&v_slc),
                black_box(&v_win),
                black_box(&x),
                black_box(&weights),
                black_box(&mut nsa_out),
                seq_len,
                &cfg,
                &mut nsa_scratch,
            );
            dense_causal_baseline(
                black_box(&q_rope),
                black_box(&k_win),
                black_box(&v_win),
                black_box(&mut dense_out),
                seq_len,
                &cfg,
                &mut dense_scratch,
            );
        }

        let iters = iterations_for_seq(seq_len);

        let us_dense = average_us(iters, || {
            dense_causal_baseline(
                black_box(&q_rope),
                black_box(&k_win),
                black_box(&v_win),
                black_box(&mut dense_out),
                seq_len,
                &cfg,
                &mut dense_scratch,
            );
            black_box(&dense_out);
        });

        let us_nsa = average_us(iters, || {
            apply_native_sparse_attention(
                black_box(&q),
                black_box(&q_rope),
                black_box(&k_cmp),
                black_box(&k_slc),
                black_box(&k_win),
                black_box(&v_cmp),
                black_box(&v_slc),
                black_box(&v_win),
                black_box(&x),
                black_box(&weights),
                black_box(&mut nsa_out),
                seq_len,
                &cfg,
                &mut nsa_scratch,
            );
            black_box(&nsa_out);
        });

        // ratio = dense_us / nsa_us.
        // >1.0 means NSA is faster; <1.0 means NSA is slower than dense.
        let ratio = us_dense / us_nsa;
        println!(
            "{:<10} {:>8} {:>12.1} {:>12.1} {:>9.2}x",
            "nsa-paper", seq_len, us_dense, us_nsa, ratio
        );
    }
    println!("\n  ratio = dense_us / nsa_us  (>1 = NSA faster, <1 = NSA slower)");
}

// ===================================================================
// Entry point
// ===================================================================

fn main() {
    println!("Native Sparse Attention Benchmark");
    println!("===================================");
    println!("Config: num_heads=4, num_kv_heads=2, head_dim=64");
    println!("NSA:    l=32, d=16, l'=64, n=16, w=512  (paper §4.1 defaults)");
    bench_nsa_throughput();
    bench_sparse_vs_dense();
    println!();
}
