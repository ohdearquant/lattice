//! Held-out NLL parity evidence for GDN-layer LoRA wired through
//! `train_micro_lora` (ADR-079 G1, issue #884).
//!
//! Ignored by default: requires a real Qwen3.5-0.8B checkpoint at
//! `$HOME/.lattice/models/qwen3.5-0.8b` and the same matched-config dataset
//! `data/lora-match-fit16/{train,valid}.jsonl` (16 train / 8 valid,
//! completion-only loss) that PR #191's on-par-with-MLX-LM comparison used —
//! gitignored local fixtures, not checked into the repo.
//!
//! Run with:
//!   cargo test -p lattice-tune --features train-backward,inference-hook,safetensors \
//!     --test gdn_lora_nll_parity -- --ignored --nocapture

#![cfg(all(
    feature = "train-backward",
    feature = "inference-hook",
    feature = "safetensors"
))]

use std::path::{Path, PathBuf};

use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::tokenizer::Tokenizer;
use lattice_tune::lora::train::{MicroLoraConfig, TrainingPair, train_micro_lora};

fn model_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".lattice/models/qwen3.5-0.8b")
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/lora-match-fit16")
}

/// Mirrors `train_common::load_jsonl`'s prompt/completion tokenization
/// exactly (bin-private module, not importable from a lib integration test).
fn load_pairs(path: &Path, tokenizer: &dyn Tokenizer, seq_len: usize) -> Vec<TrainingPair> {
    let text = std::fs::read_to_string(path).expect("read jsonl fixture");
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).expect("parse jsonl line");
        let prompt = v["prompt"].as_str().unwrap_or("").to_string();
        let completion = v["completion"].as_str().unwrap_or("").to_string();
        if prompt.is_empty() || completion.is_empty() {
            continue;
        }
        let mut full = prompt.clone();
        full.push_str(&completion);
        let prompt_tok = tokenizer.tokenize(&prompt);
        let full_tok = tokenizer.tokenize(&full);
        let prompt_ids: Vec<u32> = prompt_tok.input_ids[..prompt_tok.real_length].to_vec();
        let full_ids: Vec<u32> = full_tok.input_ids[..full_tok.real_length].to_vec();
        let total = full_ids.len();
        if total < 2 || total > seq_len {
            continue;
        }
        let completion_start = prompt_ids.len();
        if completion_start == 0 || completion_start >= total {
            continue;
        }
        out.push(TrainingPair {
            tokens: full_ids,
            completion_start,
        });
    }
    out
}

/// Mean held-out NLL over completion positions, scored through the real
/// inference forward pass (`compute_token_nlls`) with `adapter` installed via
/// `set_lora` — the actual CONSUME seam, not the training tape's own NLL.
/// `adapter = None` scores the frozen base model.
fn eval_held_out(
    model: &mut Qwen35Model,
    pairs: &[TrainingPair],
    adapter: Option<lattice_tune::lora::LoraAdapter>,
) -> f32 {
    match adapter {
        Some(a) => model.set_lora(Box::new(a)),
        None => model.set_lora(Box::new(lattice_inference::lora_hook::NoopLoraHook)),
    }
    let mut total = 0.0f32;
    let mut n = 0usize;
    for pair in pairs {
        let nlls = model
            .compute_token_nlls(&pair.tokens)
            .expect("compute_token_nlls");
        // nlls[i] scores tokens[i+1]; completion positions are
        // (completion_start - 1) ..= (tokens.len() - 2) in nlls' indexing.
        let start = pair.completion_start - 1;
        let end = pair.tokens.len() - 1;
        for &v in &nlls[start..end] {
            total += v;
            n += 1;
        }
    }
    total / n.max(1) as f32
}

#[test]
#[ignore = "requires a real model checkpoint + local dataset fixture; run explicitly"]
fn gdn_lora_held_out_nll_parity_vs_gqa_only() {
    let mdir = model_dir();
    let ddir = data_dir();
    assert!(
        mdir.join("config.json").exists(),
        "model checkpoint not found at {mdir:?} — set up ~/.lattice/models/qwen3.5-0.8b first"
    );
    assert!(
        ddir.join("train.jsonl").exists(),
        "dataset fixture not found at {ddir:?} — copy data/lora-match-fit16 from the main lattice checkout"
    );

    let mut model = Qwen35Model::from_safetensors(&mdir).expect("load model");
    let tokenizer = model.tokenizer().clone();

    // Matched config from PR #191's on-par-with-MLX-LM comparison: rank=8,
    // scale (alpha/rank)=20 => alpha=160, lr=1e-4, layers [19..=23],
    // completion-only loss, seq_len 128, 16 train / 8 valid samples.
    const SEQ_LEN: usize = 128;
    const RANK: usize = 8;
    const ALPHA: f32 = 160.0;
    const LR: f32 = 1e-4;
    const FIRST_LAYER: usize = 19;
    const STEPS: usize = 100;

    let train_pairs = load_pairs(&ddir.join("train.jsonl"), &tokenizer, SEQ_LEN);
    let valid_pairs = load_pairs(&ddir.join("valid.jsonl"), &tokenizer, SEQ_LEN);
    assert!(!train_pairs.is_empty(), "no train pairs loaded");
    assert!(!valid_pairs.is_empty(), "no valid pairs loaded");
    println!(
        "loaded {} train / {} valid pairs",
        train_pairs.len(),
        valid_pairs.len()
    );

    let base_nll = eval_held_out(&mut model, &valid_pairs, None);
    println!("base (no LoRA) held-out NLL: {base_nll:.4}");

    let gqa_cfg = MicroLoraConfig {
        rank: RANK,
        alpha: ALPHA,
        first_layer: FIRST_LAYER,
        steps: STEPS,
        learning_rate: LR,
        max_seq_len: SEQ_LEN,
        train_gdn: false,
    };
    let gqa_adapter =
        train_micro_lora(&model, &train_pairs, &gqa_cfg).expect("train_micro_lora (GQA-only)");
    let gqa_nll = eval_held_out(&mut model, &valid_pairs, Some(gqa_adapter));
    println!(
        "GQA-only (train_gdn=false) held-out NLL after {STEPS} steps: {gqa_nll:.4}  (delta {:+.4})",
        gqa_nll - base_nll
    );

    let gdn_cfg = MicroLoraConfig {
        train_gdn: true,
        ..gqa_cfg
    };
    let gdn_adapter =
        train_micro_lora(&model, &train_pairs, &gdn_cfg).expect("train_micro_lora (GQA+GDN)");

    // Acceptance criterion: non-empty GDN grads flowed — the adapter must
    // carry all 5 GDN projections for every GDN layer in [19..=23], and at
    // least one of them must have moved off its zero-B init.
    let gdn_modules = [
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    ];
    let mut any_nonzero = false;
    for layer_idx in FIRST_LAYER..=23 {
        for &module in &gdn_modules {
            if let Some(layer) = gdn_adapter.layers.get(&(layer_idx, module.to_string())) {
                if layer.b.iter().any(|&x| x != 0.0) {
                    any_nonzero = true;
                }
            }
        }
    }
    assert!(
        any_nonzero,
        "GDN LoRA adapter has all-zero B matrices after {STEPS} steps — grads did not flow"
    );

    let gdn_nll = eval_held_out(&mut model, &valid_pairs, Some(gdn_adapter));
    println!(
        "GQA+GDN (train_gdn=true) held-out NLL after {STEPS} steps: {gdn_nll:.4}  (delta {:+.4})",
        gdn_nll - base_nll
    );

    println!(
        "\n=== summary: base {base_nll:.4} | GQA-only {gqa_nll:.4} ({:+.4}) | GQA+GDN {gdn_nll:.4} ({:+.4}) ===",
        gqa_nll - base_nll,
        gdn_nll - base_nll
    );
}
