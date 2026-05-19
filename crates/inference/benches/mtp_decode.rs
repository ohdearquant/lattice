//! MTP decode throughput benchmark.
//!
//! Measures end-to-end decode tok/s for two paths:
//!   - **baseline**: single-token greedy decode (no MTP).
//!   - **mtp**: MTP-enabled greedy decode (`LATTICE_MTP=1` code path).
//!
//! Both use `chat_completion` on the same prompt and token budget.  Criterion
//! handles statistical sampling; the `LATTICE_MTP` env var is set/cleared
//! around the MTP benchmark function.
//!
//! # Env
//! - `LATTICE_MODEL_DIR`  — path to model dir (default `~/.lattice/models/qwen3.5-0.8b`).
//!   If the dir does not contain `config.json` the benchmark group is skipped.
//!
//! # Run
//! ```text
//! cargo bench -p lattice-inference --features metal-gpu,f16 -- mtp_decode
//! ```
//!
//! # CI note
//! The benchmark compiles unconditionally but the body is gated on
//! `#[cfg(all(target_os = "macos", feature = "metal-gpu"))]` and the
//! model-dir existence check, so no GPU calls occur in CI.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::path::PathBuf;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// Number of tokens requested per benchmark iteration.
const N_TOKENS: usize = 32;

/// Prompt used for both baseline and MTP runs.
const BENCH_PROMPT: &str = "The quick brown fox jumps over the lazy dog. Once upon a time in a land far away, there lived a";

// ---------------------------------------------------------------------------
// Model loading helper
// ---------------------------------------------------------------------------

/// Resolve the model directory from env or the default cache location.
fn model_dir() -> Option<PathBuf> {
    let dir = if let Ok(v) = std::env::var("LATTICE_MODEL_DIR") {
        PathBuf::from(v)
    } else {
        let home = std::env::var("HOME").ok()?;
        PathBuf::from(format!("{home}/.lattice/models/qwen3.5-0.8b"))
    };
    if dir.join("config.json").exists() {
        Some(dir)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Benchmark: baseline (no MTP)
// ---------------------------------------------------------------------------

fn bench_baseline(c: &mut Criterion) {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        let _ = c;
        eprintln!("SKIP mtp_decode/baseline: requires macOS + metal-gpu feature");
        return;
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        use lattice_inference::forward::metal_qwen35::{ChatMessage, MetalQwen35State};
        use lattice_inference::model::{GenerateConfig, Qwen35Model};

        let Some(dir) = model_dir() else {
            eprintln!(
                "SKIP mtp_decode/baseline: model not found.\n\
                 Set LATTICE_MODEL_DIR or place model at ~/.lattice/models/qwen3.5-0.8b"
            );
            return;
        };

        let model = match Qwen35Model::from_safetensors(&dir) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("SKIP mtp_decode/baseline: model load failed: {e}");
                return;
            }
        };
        let cfg = model.config().clone();
        let tokenizer = model.tokenizer();

        let mut state = match MetalQwen35State::new(model.weights(), &cfg, 4096) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("SKIP mtp_decode/baseline: Metal init failed: {e}");
                return;
            }
        };

        // Greedy, no thinking — mirrors bench_decode_ab methodology.
        let gen_cfg = GenerateConfig {
            max_new_tokens: N_TOKENS,
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
            stop_token_ids: vec![],
            enable_thinking: false,
        };

        let history = vec![ChatMessage::user(BENCH_PROMPT)];

        // Warmup outside criterion's measurement loop.
        state.reset_state();
        let _ = state.chat_completion(&history, tokenizer, &gen_cfg);

        let mut group = c.benchmark_group("mtp_decode");
        group.warm_up_time(Duration::from_secs(5));
        group.measurement_time(Duration::from_secs(20));
        group.sample_size(10);
        group.throughput(Throughput::Elements(N_TOKENS as u64));

        group.bench_function(BenchmarkId::new("greedy", "baseline"), |b| {
            b.iter(|| {
                state.reset_state();
                let out = state.chat_completion(&history, tokenizer, &gen_cfg);
                // Return a value so the compiler cannot elide the call.
                out.completion_tokens
            });
        });

        group.finish();
    }
}

// ---------------------------------------------------------------------------
// Benchmark: MTP-enabled
// ---------------------------------------------------------------------------

fn bench_mtp(c: &mut Criterion) {
    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    {
        let _ = c;
        eprintln!("SKIP mtp_decode/mtp: requires macOS + metal-gpu feature");
        return;
    }

    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        use lattice_inference::forward::metal_qwen35::{ChatMessage, MetalQwen35State};
        use lattice_inference::model::{GenerateConfig, Qwen35Model};

        let Some(dir) = model_dir() else {
            eprintln!(
                "SKIP mtp_decode/mtp: model not found.\n\
                 Set LATTICE_MODEL_DIR or place model at ~/.lattice/models/qwen3.5-0.8b"
            );
            return;
        };

        let model = match Qwen35Model::from_safetensors(&dir) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("SKIP mtp_decode/mtp: model load failed: {e}");
                return;
            }
        };
        let cfg = model.config().clone();
        let tokenizer = model.tokenizer();

        let mut state = match MetalQwen35State::new(model.weights(), &cfg, 4096) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("SKIP mtp_decode/mtp: Metal init failed: {e}");
                return;
            }
        };

        // MTP path requires top_k <= 1 and temperature <= 0.0 (checked inside
        // generate before branching to generate_greedy_mtp).
        let gen_cfg = GenerateConfig {
            max_new_tokens: N_TOKENS,
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
            stop_token_ids: vec![],
            enable_thinking: false,
        };

        let history = vec![ChatMessage::user(BENCH_PROMPT)];

        // Activate the MTP code path for both warmup and measurement.
        // SAFETY: criterion runs bench functions sequentially.  No other thread
        // reads LATTICE_MTP while this function executes.
        unsafe {
            std::env::set_var("LATTICE_MTP", "1");
        }

        // Warmup.
        state.reset_state();
        let _ = state.chat_completion(&history, tokenizer, &gen_cfg);

        let mut group = c.benchmark_group("mtp_decode");
        group.warm_up_time(Duration::from_secs(5));
        group.measurement_time(Duration::from_secs(20));
        group.sample_size(10);
        group.throughput(Throughput::Elements(N_TOKENS as u64));

        group.bench_function(BenchmarkId::new("greedy", "mtp"), |b| {
            b.iter(|| {
                state.reset_state();
                let out = state.chat_completion(&history, tokenizer, &gen_cfg);
                out.completion_tokens
            });
        });

        group.finish();

        unsafe {
            std::env::remove_var("LATTICE_MTP");
        }
    }
}

// ---------------------------------------------------------------------------
// Criterion entry points
// ---------------------------------------------------------------------------

criterion_group!(benches, bench_baseline, bench_mtp);
criterion_main!(benches);
