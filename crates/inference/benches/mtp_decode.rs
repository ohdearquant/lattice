//! MTP decode throughput benchmark.
//!
//! Measures end-to-end decode tok/s for two paths:
//!   - **baseline**: greedy decode (MTP weights loaded but MTP gate disabled).
//!   - **mtp**: MTP-enabled greedy decode.
//!
//! Both paths load the **Q4-quantized** model via `from_q4_dir`, which is the
//! only constructor that populates MTP weights.  The f32 `new()` constructor
//! never loads MTP weights, so it cannot benchmark the MTP path.
//!
//! # Env
//! - `LATTICE_MODEL_DIR` — path to Q4 model dir (default `~/.lattice/models/qwen3.5-0.8b-q4`).
//!   Must contain `config.json` and `mtp_fc_weight.q4`.  Skipped if missing.
//! - `LATTICE_TOKENIZER_DIR` — path to dir with `tokenizer.json`
//!   (default `~/.lattice/models/qwen3.5-0.8b`).
//!
//! # Run
//! ```text
//! cargo bench -p lattice-inference --features metal-gpu,f16 -- mtp_decode
//! ```
//!
//! # CI note
//! Gated on `#[cfg(all(target_os = "macos", feature = "metal-gpu"))]` and
//! model-dir existence, so no GPU calls occur in CI.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::path::PathBuf;
use std::time::Duration;

const N_TOKENS: usize = 32;

const BENCH_PROMPT: &str =
    "Explain the theory of general relativity in simple terms, covering spacetime curvature and";

fn q4_model_dir() -> Option<PathBuf> {
    let dir = if let Ok(v) = std::env::var("LATTICE_MODEL_DIR") {
        PathBuf::from(v)
    } else {
        let home = std::env::var("HOME").ok()?;
        PathBuf::from(format!("{home}/.lattice/models/qwen3.5-0.8b-q4"))
    };
    if dir.join("config.json").exists() && dir.join("mtp_fc_weight.q4").exists() {
        Some(dir)
    } else {
        None
    }
}

fn tokenizer_dir() -> Option<PathBuf> {
    let dir = if let Ok(v) = std::env::var("LATTICE_TOKENIZER_DIR") {
        PathBuf::from(v)
    } else {
        let home = std::env::var("HOME").ok()?;
        PathBuf::from(format!("{home}/.lattice/models/qwen3.5-0.8b"))
    };
    if dir.join("tokenizer.json").exists() {
        Some(dir)
    } else {
        None
    }
}

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
        use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
        use lattice_inference::tokenizer::bpe::BpeTokenizer;

        let Some(dir) = q4_model_dir() else {
            eprintln!("SKIP mtp_decode/baseline: Q4 model not found (need mtp_fc_weight.q4)");
            return;
        };
        let Some(tok_dir) = tokenizer_dir() else {
            eprintln!("SKIP mtp_decode/baseline: tokenizer.json not found");
            return;
        };

        let cfg = match Qwen35Config::from_config_json(&dir.join("config.json")) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP mtp_decode/baseline: config parse failed: {e}");
                return;
            }
        };

        let tok_path = tok_dir.join("tokenizer.json");
        let tokenizer = match BpeTokenizer::from_tokenizer_json(&tok_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIP mtp_decode/baseline: tokenizer load failed: {e}");
                return;
            }
        };

        let mut state = match MetalQwen35State::from_q4_dir(&dir, &tok_path, &cfg, 4096) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("SKIP mtp_decode/baseline: Metal Q4 init failed: {e}");
                return;
            }
        };

        let gen_cfg = GenerateConfig {
            max_new_tokens: N_TOKENS,
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
            stop_token_ids: vec![],
            enable_thinking: false,
            enable_mtp: Some(false),
        };

        let history = vec![ChatMessage::user(BENCH_PROMPT)];

        state.reset_state();
        let warmup = state.chat_completion(&history, &tokenizer, &gen_cfg);
        let actual_tokens = warmup.completion_tokens;
        if actual_tokens < N_TOKENS {
            eprintln!(
                "WARN mtp_decode/baseline: warmup produced {actual_tokens}/{N_TOKENS} tokens \
                 (early stop). Throughput based on actual count."
            );
        }

        let mut group = c.benchmark_group("mtp_decode");
        group.warm_up_time(Duration::from_secs(5));
        group.measurement_time(Duration::from_secs(20));
        group.sample_size(10);
        group.throughput(Throughput::Elements(actual_tokens as u64));

        group.bench_function(BenchmarkId::new("greedy", "baseline"), |b| {
            b.iter(|| {
                state.reset_state();
                let out = state.chat_completion(&history, &tokenizer, &gen_cfg);
                out.completion_tokens
            });
        });

        group.finish();
    }
}

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
        use lattice_inference::model::qwen35_config::{GenerateConfig, Qwen35Config};
        use lattice_inference::tokenizer::bpe::BpeTokenizer;

        let Some(dir) = q4_model_dir() else {
            eprintln!("SKIP mtp_decode/mtp: Q4 model not found (need mtp_fc_weight.q4)");
            return;
        };
        let Some(tok_dir) = tokenizer_dir() else {
            eprintln!("SKIP mtp_decode/mtp: tokenizer.json not found");
            return;
        };

        let cfg = match Qwen35Config::from_config_json(&dir.join("config.json")) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("SKIP mtp_decode/mtp: config parse failed: {e}");
                return;
            }
        };

        let tok_path = tok_dir.join("tokenizer.json");
        let tokenizer = match BpeTokenizer::from_tokenizer_json(&tok_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIP mtp_decode/mtp: tokenizer load failed: {e}");
                return;
            }
        };

        let mut state = match MetalQwen35State::from_q4_dir(&dir, &tok_path, &cfg, 4096) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("SKIP mtp_decode/mtp: Metal Q4 init failed: {e}");
                return;
            }
        };

        if !state.has_mtp() {
            eprintln!(
                "SKIP mtp_decode/mtp: model loaded but MTP weights absent \
                 (mtp_num_hidden_layers=0 or mtp files missing)"
            );
            return;
        };

        let gen_cfg = GenerateConfig {
            max_new_tokens: N_TOKENS,
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: Some(42),
            stop_token_ids: vec![],
            enable_thinking: false,
            enable_mtp: Some(true),
        };

        let history = vec![ChatMessage::user(BENCH_PROMPT)];

        state.reset_state();
        let warmup = state.chat_completion(&history, &tokenizer, &gen_cfg);
        let actual_tokens = warmup.completion_tokens;
        if actual_tokens < N_TOKENS {
            eprintln!(
                "WARN mtp_decode/mtp: warmup produced {actual_tokens}/{N_TOKENS} tokens \
                 (early stop). Throughput based on actual count."
            );
        }

        let mut group = c.benchmark_group("mtp_decode");
        group.warm_up_time(Duration::from_secs(5));
        group.measurement_time(Duration::from_secs(20));
        group.sample_size(10);
        group.throughput(Throughput::Elements(actual_tokens as u64));

        group.bench_function(BenchmarkId::new("greedy", "mtp"), |b| {
            b.iter(|| {
                state.reset_state();
                let out = state.chat_completion(&history, &tokenizer, &gen_cfg);
                out.completion_tokens
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_baseline, bench_mtp);
criterion_main!(benches);
