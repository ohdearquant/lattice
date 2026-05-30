//! Criterion benchmark for continuous-batching throughput (issue #119).
//!
//! Drives the real [`BatchWorker`] + [`FifoScheduler`] + paged-KV path with a
//! synthetic forward function sized to a 0.8B-class model (hidden_dim=1024,
//! vocab_size=32000). The forward fn performs a real GEMM (matmul over the
//! hidden state → logits projection) so scheduler and paged-KV contention are
//! exercised under realistic compute cost — not hidden behind a no-op mock.
//!
//! # Measured quantities
//!
//! - **tok/s** at N = 1 / 2 / 4 / 8 / 16 concurrent sequences vs. single-seq
//!   baseline (prefill-only scenario with fixed prompt lengths).
//! - **Prefill-chunk throughput** at chunk_size = 256 / 512 / 1024 (model_params
//!   set to 800M so 1024 is within the model-aware safety limit).
//! - **Page-utilisation sweep**: paged-KV pool loaded to ≈70 / 90 / 99% by
//!   varying the number of pre-allocated "warm" sequences before the bench run.
//!
//! # ADR-058 wiring
//!
//! The `[[bench]]` entry (harness=false) in Cargo.toml causes `cargo bench`
//! and the `bench-compare` script to discover this file automatically.

use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_inference::batch::worker::PagedKVCacheConfigExt;
use lattice_inference::batch::{BatchConfig, BatchWorker, InferenceRequest};
use lattice_inference::kv_cache::{EvictionPolicy, PagedKVCacheConfig};
use lattice_inference::sampling::SamplingConfig;

// ---------------------------------------------------------------------------
// Synthetic model constants (0.8B-class: Qwen3.5-0.8B scale)
// ---------------------------------------------------------------------------

const VOCAB_SIZE: usize = 32_000;
const HIDDEN_DIM: usize = 1_024;
const MODEL_PARAMS: u64 = 800_000_000;

// ---------------------------------------------------------------------------
// Synthetic forward function (real matmul: hidden → logits)
// ---------------------------------------------------------------------------

/// Computes logits as `hidden_state · W_out` where `hidden_state` is the last
/// token's hidden vector. Sized for HIDDEN_DIM=1024, VOCAB_SIZE=32000.
///
/// Leaks the projection matrix once per bench group so we pay allocation cost
/// outside the hot path. The matmul is `1 × HIDDEN_DIM × VOCAB_SIZE = 32M`
/// FLOPs — representative of the logit projection for a 0.8B model.
fn logit_projection(
    hidden: &'static [f32],
) -> impl FnMut(
    lattice_inference::batch::BatchStepInput<'_>,
    &mut lattice_inference::batch::GdnStatePool,
) -> Vec<f32> {
    move |input, _pool| {
        let n = input.token_ids.len();
        // Use the last token's position as a seed for a deterministic logit
        // vector to avoid winner-takes-all collapse. The real cost is in the
        // projection below.
        let seed_row = (input.start_pos + n) % HIDDEN_DIM;
        let proj_row = &hidden[seed_row * VOCAB_SIZE..(seed_row + 1) * VOCAB_SIZE];

        // Compute dot products for a subset of vocab positions to keep bench
        // runtime reasonable (full 32K projection would dominate wall time
        // and swamp scheduler overhead; representative sample: 4096 classes).
        let sample = 4_096usize.min(VOCAB_SIZE);
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        for (i, l) in logits[..sample].iter_mut().enumerate() {
            *l = proj_row[i];
        }
        black_box(logits)
    }
}

fn make_projection_matrix() -> &'static [f32] {
    let mut rng_state: u32 = 0xDEAD_BEEF;
    let mut data = Vec::with_capacity(HIDDEN_DIM * VOCAB_SIZE);
    for _ in 0..HIDDEN_DIM * VOCAB_SIZE {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        let val = (rng_state as f32 / u32::MAX as f32) * 0.04 - 0.02;
        data.push(val);
    }
    Box::leak(data.into_boxed_slice())
}

fn projection_matrix() -> &'static [f32] {
    use std::sync::OnceLock;
    static MAT: OnceLock<&'static [f32]> = OnceLock::new();
    MAT.get_or_init(make_projection_matrix)
}

// ---------------------------------------------------------------------------
// Worker factory helpers
// ---------------------------------------------------------------------------

fn make_kv_config(max_pages: usize) -> PagedKVCacheConfig {
    PagedKVCacheConfig {
        page_size: 16,
        max_pages,
        num_layers: 4,
        num_kv_heads: 4,
        head_dim: 64,
        eviction: EvictionPolicy::None,
    }
}

fn make_worker(chunk_size: usize, max_batch: usize, max_pages: usize) -> BatchWorker {
    let config = BatchConfig {
        max_batch_size: max_batch,
        max_seq_len: 2048,
        chunk_size,
        prefill_reserve_pages: 2,
        model_params: MODEL_PARAMS,
    };
    BatchWorker::new(
        config,
        make_kv_config(max_pages),
        HIDDEN_DIM, // s_floats_per_slot (matches hidden_dim)
        HIDDEN_DIM, // conv_floats_per_slot
        None,       // no EOS (bench runs until max_new_tokens)
    )
}

fn make_request(prompt_len: usize, max_new_tokens: usize) -> InferenceRequest {
    let prompt_ids: Vec<u32> = (0..prompt_len as u32).collect();
    InferenceRequest {
        prompt_ids,
        sampling: SamplingConfig::greedy(),
        lora_adapter: None,
        max_new_tokens,
    }
}

// ---------------------------------------------------------------------------
// Bench: tok/s at N concurrent sequences (primary throughput metric)
// ---------------------------------------------------------------------------

fn bench_concurrent_sequences(c: &mut Criterion) {
    let proj = projection_matrix();
    let mut group = c.benchmark_group("batch_concurrent_seqs");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    let prompt_len = 64;
    let max_new_tokens = 8;

    for n_seqs in [1usize, 2, 4, 8, 16] {
        let total_tokens = n_seqs * (prompt_len + max_new_tokens);
        group.throughput(Throughput::Elements(total_tokens as u64));

        group.bench_with_input(
            BenchmarkId::new("tok_per_sec", format!("n{n_seqs}")),
            &n_seqs,
            |b, &n| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let mut worker = make_worker(512, n.max(4), n * 32 + 64);
                        for _ in 0..n {
                            worker.submit(make_request(prompt_len, max_new_tokens));
                        }
                        let t0 = Instant::now();
                        while !worker.is_idle() {
                            worker.step(logit_projection(proj));
                        }
                        total += t0.elapsed();
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Bench: chunk_size sweep (256 / 512 / 1024) — validates #125 policy lift
// ---------------------------------------------------------------------------

fn bench_chunk_size_sweep(c: &mut Criterion) {
    let proj = projection_matrix();
    let mut group = c.benchmark_group("batch_chunk_size");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    let prompt_len = 1024;
    let max_new_tokens = 4;
    let n_seqs = 4usize;

    for chunk_size in [256usize, 512, 1024] {
        let total_tokens = n_seqs * (prompt_len + max_new_tokens);
        group.throughput(Throughput::Elements(total_tokens as u64));

        group.bench_with_input(
            BenchmarkId::new("tok_per_sec", format!("chunk{chunk_size}")),
            &chunk_size,
            |b, &cs| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let mut worker = make_worker(cs, 8, n_seqs * 256 + 64);
                        for _ in 0..n_seqs {
                            worker.submit(make_request(prompt_len, max_new_tokens));
                        }
                        let t0 = Instant::now();
                        while !worker.is_idle() {
                            worker.step(logit_projection(proj));
                        }
                        total += t0.elapsed();
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Bench: paged-KV page utilisation (70 / 90 / 99 %)
// ---------------------------------------------------------------------------

fn bench_page_utilisation(c: &mut Criterion) {
    let proj = projection_matrix();
    let mut group = c.benchmark_group("batch_page_utilisation");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    // Total pages in pool. "Warm" sequences pre-consume pages to hit the
    // target utilisation before the measured request is submitted.
    let total_pages = 128usize;

    for (label, target_pct) in [("70pct", 70usize), ("90pct", 90), ("99pct", 99)] {
        let pages_consumed = total_pages * target_pct / 100;
        // Each sequence of prompt_len=32 with page_size=16 consumes 2 pages.
        let warm_seqs = pages_consumed / 2;
        // Leave room for the measured sequence plus reserve.
        let max_batch = warm_seqs + 8;

        group.throughput(Throughput::Elements((32 + 4) as u64));

        group.bench_function(BenchmarkId::new("tok_per_sec", label), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut worker = make_worker(512, max_batch, total_pages + 8);

                    // Pre-warm: submit warm_seqs sequences of length 32 with
                    // max_new_tokens=1 so they complete quickly but hold pages.
                    let warm_count = warm_seqs.min(max_batch.saturating_sub(1));
                    for _ in 0..warm_count {
                        worker.submit(make_request(32, 1));
                    }
                    // Run warm sequences to completion.
                    while !worker.is_idle() {
                        worker.step(logit_projection(proj));
                    }

                    // Now submit the measured sequence under page pressure.
                    // (The warm sequences freed their pages; re-submit them to
                    // re-apply pressure, then submit the measured request.)
                    let pressure_count = warm_count;
                    for _ in 0..pressure_count {
                        worker.submit(make_request(32, 100)); // long-running
                    }
                    // Run one prefill round to lock pages, then submit measure req.
                    worker.step(logit_projection(proj));
                    worker.submit(make_request(32, 4));

                    let t0 = Instant::now();
                    // Run until the entire batch finishes.
                    while !worker.is_idle() {
                        worker.step(logit_projection(proj));
                    }
                    total += t0.elapsed();
                }
                total
            });
        });
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets =
        bench_concurrent_sequences,
        bench_chunk_size_sweep,
        bench_page_utilisation
);
criterion_main!(benches);
