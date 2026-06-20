// Reverse-mode (exact) gradient LoRA trainer for Qwen3.5-0.8B.
//
// Milestone-1: rank-r LoRA on lm_head. lm_head is the top of the network, so
// its gradient needs no transformer backward — the input to lm_head is the
// final hidden state H_t (post final-norm), and the base weights are frozen, so
// H_t is a fixed per-position input. We cache (H_t, base_logits, target) once,
// then run real Adam on the LoRA factors:
//
//   logits_t = base_logits_t + scale · B · (A · H_t)
//   dL/dlogits = softmax(logits) - onehot(target)   (masked to completion, /n)
//   grad_B, grad_A = lora_vjp(dL/dlogits, H_t, A·H_t, A, B, ...)
//
// The LoRA delta is applied in BOTH the gradient and the NLL eval, so the
// reported curve reflects the trained adapter.
//
// Usage: train_grad --model-dir <path> --data-dir <path> [--steps 150]
//        [--lr 1e-3] [--max-train 6] [--seq-len 96]

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::Instant;

use lattice_inference::backward::ops::lora_vjp;
use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::tokenizer::Tokenizer;
use lattice_tune::lora::AdamState;

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn parse_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn usage() {
    eprintln!(
        "usage: train_grad [OPTIONS]

Options:
  --model-dir  <PATH>   Model directory (default: $HOME/.lattice/models/qwen3.5-0.8b)
  --data-dir   <PATH>   Dataset directory with train.jsonl
  --steps      <N>      Adam steps (default: 150)
  --lr         <F>      Learning rate (default: 1e-3)
  --rank       <N>      LoRA rank (default: 8)
  --alpha      <F>      LoRA alpha (default: 16.0)
  --seq-len    <N>      Max tokens per sample (default: 96)
  --max-train  <N>      Training samples cap (default: 6)
  --log-every  <N>      Print NLL every N steps (default: 10)
  -h, --help            Print this help"
    );
}

fn default_model_dir() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".lattice")
        .join("models")
        .join("qwen3.5-0.8b")
}

/// One training sample with prompt/completion split.
struct Sample {
    tokens: Vec<u32>,
    completion_start: usize,
}

fn load_jsonl(
    path: &Path,
    tokenizer: &dyn Tokenizer,
    seq_len: usize,
    max_samples: usize,
) -> Result<Vec<Sample>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        if out.len() >= max_samples {
            break;
        }
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
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
        out.push(Sample {
            tokens: full_ids,
            completion_start,
        });
    }
    Ok(out)
}

/// One completion position's frozen forward context. `base_logits = lm_head · h`
/// is fixed (base weights frozen), so the per-step cost is just the LoRA delta.
struct PosContext {
    h: Vec<f32>,           // [hidden] final hidden state, post final-norm
    base_logits: Vec<f32>, // [vocab] frozen base logits
    target: u32,
}

/// All completion positions of one sample, with their frozen forward context.
struct SampleCache {
    positions: Vec<PosContext>,
}

/// Run the real 24-layer forward once, capture H_t per completion position, and
/// precompute base_logits = lm_head · H_t. This is the only place the frozen
/// transformer runs; training reuses the cache.
fn build_cache(
    model: &Qwen35Model,
    lm_head: &[f32],
    sample: &Sample,
    hidden: usize,
    vocab: usize,
) -> Result<SampleCache, Box<dyn std::error::Error>> {
    let hiddens = model.forward_final_hidden(&sample.tokens)?;
    let seq_len = sample.tokens.len();
    let mut positions = Vec::new();
    for t in (sample.completion_start - 1)..seq_len - 1 {
        let h = &hiddens[t];
        let mut base_logits = vec![0.0f32; vocab];
        for (i, bl) in base_logits.iter_mut().enumerate() {
            *bl = lm_head[i * hidden..(i + 1) * hidden]
                .iter()
                .zip(h.iter())
                .map(|(w, x)| w * x)
                .sum();
        }
        positions.push(PosContext {
            h: h.clone(),
            base_logits,
            target: sample.tokens[t + 1],
        });
    }
    Ok(SampleCache { positions })
}

/// Apply the LoRA delta to a cached position and return (nll, logits, a_h).
fn forward_pos(
    pos: &PosContext,
    lora_a: &[f32],
    lora_b: &[f32],
    rank: usize,
    hidden: usize,
    _vocab: usize,
    scale: f32,
) -> (f32, Vec<f32>, Vec<f32>) {
    // a_h = A · h   [rank]
    let mut a_h = vec![0.0f32; rank];
    for (r, ah) in a_h.iter_mut().enumerate() {
        *ah = lora_a[r * hidden..(r + 1) * hidden]
            .iter()
            .zip(pos.h.iter())
            .map(|(w, x)| w * x)
            .sum();
    }
    // logits = base_logits + scale · B · a_h
    let mut logits = pos.base_logits.clone();
    for (i, l) in logits.iter_mut().enumerate() {
        let delta: f32 = lora_b[i * rank..(i + 1) * rank]
            .iter()
            .zip(a_h.iter())
            .map(|(b, ah)| b * ah)
            .sum();
        *l += scale * delta;
    }
    // masked CE at this position
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let nll = -((logits[pos.target as usize] - max) - sum_exp.ln());
    (nll, logits, a_h)
}

/// Mean masked NLL over all completion positions, with the LoRA adapter applied.
fn eval_nll(
    caches: &[SampleCache],
    lora_a: &[f32],
    lora_b: &[f32],
    rank: usize,
    hidden: usize,
    vocab: usize,
    scale: f32,
) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for c in caches {
        for pos in &c.positions {
            let (nll, _logits, _a_h) = forward_pos(pos, lora_a, lora_b, rank, hidden, vocab, scale);
            sum += nll as f64;
            n += 1;
        }
    }
    if n == 0 {
        f32::NAN
    } else {
        (sum / n as f64) as f32
    }
}

/// Mean masked NLL and the averaged LoRA gradients for one sample.
#[allow(clippy::too_many_arguments)]
fn nll_and_grad(
    cache: &SampleCache,
    lora_a: &[f32],
    lora_b: &[f32],
    rank: usize,
    hidden: usize,
    vocab: usize,
    scale: f32,
) -> (f32, Vec<f32>, Vec<f32>) {
    let mut grad_a = vec![0.0f32; rank * hidden];
    let mut grad_b = vec![0.0f32; vocab * rank];
    let mut nll_sum = 0.0f64;
    let n = cache.positions.len().max(1);

    for pos in &cache.positions {
        let (nll, logits, a_h) = forward_pos(pos, lora_a, lora_b, rank, hidden, vocab, scale);
        nll_sum += nll as f64;

        // dL/dlogits = softmax(logits) - onehot(target)
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
        let mut d_logits = vec![0.0f32; vocab];
        for (v, dl) in d_logits.iter_mut().enumerate() {
            let p = (logits[v] - max).exp() / sum_exp;
            *dl = p - if v == pos.target as usize { 1.0 } else { 0.0 };
        }

        let (gb, ga, _dx) = lora_vjp(
            &d_logits, &pos.h, &a_h, lora_a, lora_b, rank, hidden, vocab, scale,
        );
        for (g, v) in grad_a.iter_mut().zip(ga.iter()) {
            *g += v;
        }
        for (g, v) in grad_b.iter_mut().zip(gb.iter()) {
            *g += v;
        }
    }

    let inv = 1.0 / n as f32;
    for g in &mut grad_a {
        *g *= inv;
    }
    for g in &mut grad_b {
        *g *= inv;
    }
    ((nll_sum / n as f64) as f32, grad_a, grad_b)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if parse_flag(&args, "-h") || parse_flag(&args, "--help") {
        usage();
        return Ok(());
    }

    let model_dir = parse_arg(&args, "--model-dir")
        .map(PathBuf::from)
        .unwrap_or_else(default_model_dir);
    let data_dir = parse_arg(&args, "--data-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/claude-logs-lora"));
    let steps: usize = parse_arg(&args, "--steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);
    let lr: f32 = parse_arg(&args, "--lr")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let rank: usize = parse_arg(&args, "--rank")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let alpha: f32 = parse_arg(&args, "--alpha")
        .and_then(|s| s.parse().ok())
        .unwrap_or(16.0);
    let seq_len: usize = parse_arg(&args, "--seq-len")
        .and_then(|s| s.parse().ok())
        .unwrap_or(96);
    let max_train: usize = parse_arg(&args, "--max-train")
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    let log_every: usize = parse_arg(&args, "--log-every")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    println!("=== exact-gradient LoRA trainer (lm_head) ===");
    println!("model-dir:  {}", model_dir.display());
    println!("data-dir:   {}", data_dir.display());
    println!("steps:      {steps}  lr: {lr}  rank: {rank}  alpha: {alpha}");
    println!("max-train:  {max_train}  seq-len: {seq_len}");
    println!();

    println!("Loading model...");
    let t0 = Instant::now();
    let model = Qwen35Model::from_safetensors(&model_dir)
        .map_err(|e| format!("failed to load model: {e}"))?;
    println!("  loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let cfg = model.config().clone();
    let vocab = cfg.vocab_size;
    let hidden = cfg.hidden_size;
    let scale = alpha / rank as f32;
    println!(
        "  hidden={hidden}  vocab={vocab}  layers={}",
        cfg.num_hidden_layers
    );

    let tokenizer = model.tokenizer().clone();

    println!("\nLoading dataset from {}...", data_dir.display());
    let train_samples = load_jsonl(
        &data_dir.join("train.jsonl"),
        &tokenizer as &dyn Tokenizer,
        seq_len,
        max_train,
    )?;
    if train_samples.is_empty() {
        return Err(
            "no training samples (check --data-dir, train.jsonl must have prompt+completion fields)"
                .into(),
        );
    }
    println!("  {} samples loaded", train_samples.len());

    // Real forward once per sample: capture H_t + base_logits at completion positions.
    println!("\nBuilding forward cache (real 24-layer forward)...");
    let tcache = Instant::now();
    let (lm_head_slice, _final_norm, _embed) = model.head_weights();
    let lm_head = lm_head_slice.to_vec();
    let mut caches = Vec::new();
    let mut total_positions = 0usize;
    for s in &train_samples {
        let c = build_cache(&model, &lm_head, s, hidden, vocab)?;
        total_positions += c.positions.len();
        caches.push(c);
    }
    println!(
        "  {} completion positions cached in {:.1}s",
        total_positions,
        tcache.elapsed().as_secs_f64()
    );

    // TBV: cached base NLL must match the model's own compute_token_nlls.
    {
        let s0 = &train_samples[0];
        let model_nlls = model.compute_token_nlls(&s0.tokens)?;
        let start = s0.completion_start - 1;
        let end = s0.tokens.len() - 1;
        let model_masked: f32 = model_nlls[start..end].iter().sum::<f32>() / (end - start) as f32;
        let zero_b = vec![0.0f32; vocab * rank];
        let dummy_a = vec![0.0f32; rank * hidden];
        let cached_masked = eval_nll(&caches[..1], &dummy_a, &zero_b, rank, hidden, vocab, scale);
        println!(
            "  cache check (sample 0): model={model_masked:.5}  cached={cached_masked:.5}  diff={:.2e}",
            (model_masked - cached_masked).abs()
        );
    }

    // LoRA: A [rank, hidden] small random, B [vocab, rank] zero-init.
    let mut lora_a: Vec<f32> = {
        let mut rng = 0xFEED_FACE_u64;
        (0..rank * hidden)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                ((rng >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * 0.02
            })
            .collect()
    };
    let mut lora_b = vec![0.0f32; vocab * rank];

    let mut adam = AdamState::new();
    let beta1 = 0.9f32;
    let beta2 = 0.999f32;
    let eps_adam = 1e-8f32;

    let base_nll = eval_nll(&caches, &lora_a, &lora_b, rank, hidden, vocab, scale);
    println!("\n  step    0  train NLL: {base_nll:.4}");

    for step in 1..=steps {
        let cache = &caches[(step - 1) % caches.len()];
        let (_nll, ga, gb) = nll_and_grad(cache, &lora_a, &lora_b, rank, hidden, vocab, scale);

        adam.step(
            "lm_head_a",
            &mut lora_a,
            &ga,
            lr,
            beta1,
            beta2,
            eps_adam,
            0.0,
            false,
        );
        adam.step(
            "lm_head_b",
            &mut lora_b,
            &gb,
            lr,
            beta1,
            beta2,
            eps_adam,
            0.0,
            false,
        );

        if step % log_every == 0 || step == steps {
            let mean_nll = eval_nll(&caches, &lora_a, &lora_b, rank, hidden, vocab, scale);
            println!(
                "  step {step:4}  train NLL: {mean_nll:.4}  (delta from base: {:+.4})",
                mean_nll - base_nll
            );
        }
    }

    let final_nll = eval_nll(&caches, &lora_a, &lora_b, rank, hidden, vocab, scale);
    println!("\nTraining complete.");
    println!("  base NLL:  {base_nll:.4}");
    println!("  final NLL: {final_nll:.4}");
    println!("  reduction: {:+.4}", final_nll - base_nll);
    println!(
        "  lora_a norm: {:.4}   lora_b norm: {:.4}",
        lora_a.iter().map(|x| x * x).sum::<f32>().sqrt(),
        lora_b.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    Ok(())
}
