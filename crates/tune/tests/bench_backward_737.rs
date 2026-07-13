//! A/B measurement harness for issue #737 stage 1 (scalar backward-loop
//! vectorization). Not run in CI: requires a real Qwen3.5-0.8B checkpoint on
//! disk at `$HOME/.lattice/models/qwen3.5-0.8b` (or `LATTICE_MODEL_DIR`).
//!
//! Exercises the real production path (`train_micro_lora`, `first_layer=19`,
//! `TOP_LAYER=23`), which materialises layers 19..=23 — a mix of GQA (19, 23)
//! and GDN (20, 21, 22) mixers, each carrying a LoRA slot — so one training
//! step drives every backward-tape primitive touched in this PR: `linear_vjp`
//! (o/q/k/v projections and the lm_head), `lora_vjp` (every LoRA slot),
//! `swiglu_backward`, and the lm_head forward logits.
//!
//! Run (release, single-threaded so wall-clock isn't perturbed by other
//! criterion/test workers):
//!
//! ```text
//! CARGO_TARGET_DIR=$HOME/.cache/shared-cargo-target/lattice \
//!   cargo test -p lattice-tune --release --test bench_backward_737 -- \
//!   --ignored --nocapture --test-threads=1
//! ```
use std::path::PathBuf;
use std::time::Instant;

use lattice_inference::model::qwen35::Qwen35Model;
use lattice_tune::lora::train::{MicroLoraConfig, TrainingPair, train_micro_lora};

fn model_dir() -> PathBuf {
    std::env::var("LATTICE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(".lattice")
                .join("models")
                .join("qwen3.5-0.8b")
        })
}

/// Deterministic xorshift64 token generator — timing does not depend on
/// token *values* (no data-dependent branching in the backward tape math),
/// only on shapes, so synthetic-but-in-vocab tokens are a sound stand-in for
/// a real dataset here.
fn synth_pairs(vocab: usize, n_pairs: usize, seq_len: usize) -> Vec<TrainingPair> {
    let mut state: u64 = 0x1234_5678_9abc_def1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    (0..n_pairs)
        .map(|_| {
            let tokens: Vec<u32> = (0..seq_len)
                .map(|_| (next() % vocab as u64) as u32)
                .collect();
            TrainingPair {
                tokens,
                completion_start: seq_len / 2,
            }
        })
        .collect()
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

#[test]
#[ignore = "requires a real Qwen3.5-0.8B checkpoint on disk; run explicitly for #737 A/B measurement"]
fn bench_train_micro_lora_one_step_737() {
    let dir = model_dir();
    let model = Qwen35Model::from_safetensors(&dir)
        .unwrap_or_else(|e| panic!("failed to load model at {}: {e}", dir.display()));

    let vocab = model.config().vocab_size;
    let seq_len = 48;
    let pairs = synth_pairs(vocab, 4, seq_len);

    let config = MicroLoraConfig {
        rank: 8,
        alpha: 16.0,
        first_layer: 19, // spans layers 19..=23: GQA(19), GDN(20,21,22), GQA(23)
        steps: 1,
        learning_rate: 1e-3,
        max_seq_len: 64,
    };

    const RUNS: usize = 5;
    let mut samples = Vec::with_capacity(RUNS);
    for i in 0..RUNS {
        let start = Instant::now();
        let _adapter = train_micro_lora(&model, &pairs, &config).expect("train_micro_lora");
        let secs = start.elapsed().as_secs_f64();
        eprintln!("run {i}: {secs:.3}s");
        samples.push(secs);
    }

    let med = median(samples.clone());
    eprintln!("samples={samples:?}");
    eprintln!(
        "median={med:.3}s over {RUNS} runs, first_layer=19..=23, seq_len={seq_len}, pairs={}",
        pairs.len()
    );
}
