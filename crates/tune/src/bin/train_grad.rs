// Exact lm_head LoRA trainer — see docs/design.md (§train_grad).

use std::path::PathBuf;
use std::time::Instant;

use lattice_inference::backward::ops::lora_vjp;
use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::tokenizer::Tokenizer;
use lattice_tune::lora::AdamState;

mod train_common;
use train_common::{ArgView, Sample, load_jsonl, verify_tbv};

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

/// Typed CLI config for `train_grad`. Field defaults are the documented
/// contract in `usage()` and issue #845's flag table — the snapshot tests
/// below pin them.
#[derive(Debug)]
struct Config {
    model_dir: PathBuf,
    data_dir: PathBuf,
    steps: usize,
    lr: f32,
    rank: usize,
    alpha: f32,
    seq_len: usize,
    max_train: usize,
    log_every: usize,
}

fn parse_alpha(argv: &ArgView) -> Result<f32, String> {
    if !argv.flag("--alpha") {
        return Ok(16.0);
    }
    let value = argv
        .arg("--alpha")
        .ok_or_else(|| "invalid --alpha: missing value".to_string())?;
    let alpha = value
        .parse::<f32>()
        .map_err(|_| format!("invalid --alpha '{value}': expected a finite number"))?;
    if !alpha.is_finite() {
        return Err(format!("invalid --alpha '{value}': value must be finite"));
    }
    Ok(alpha)
}

fn parse_config(argv: &ArgView) -> Result<Config, String> {
    Ok(Config {
        model_dir: argv
            .arg("--model-dir")
            .map(PathBuf::from)
            .unwrap_or_else(train_common::default_model_dir),
        data_dir: argv
            .arg("--data-dir")
            .map(PathBuf::from)
            .unwrap_or_else(train_common::default_data_dir),
        steps: argv
            .arg("--steps")
            .and_then(|s| s.parse().ok())
            .unwrap_or(150),
        lr: argv
            .arg("--lr")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1e-3),
        rank: argv.arg("--rank").and_then(|s| s.parse().ok()).unwrap_or(8),
        alpha: parse_alpha(argv)?,
        seq_len: argv
            .arg("--seq-len")
            .and_then(|s| s.parse().ok())
            .unwrap_or(96),
        max_train: argv
            .arg("--max-train")
            .and_then(|s| s.parse().ok())
            .unwrap_or(6),
        log_every: argv
            .arg("--log-every")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let argv = ArgView::new(&args);
    if argv.flag("-h") || argv.flag("--help") {
        usage();
        return Ok(());
    }

    let Config {
        model_dir,
        data_dir,
        steps,
        lr,
        rank,
        alpha,
        seq_len,
        max_train,
        log_every,
    } = parse_config(&argv)
        .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;

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

    // Fail closed unless the cached base NLL matches the model's reference NLL.
    {
        let s0 = &train_samples[0];
        let model_nlls = model.compute_token_nlls(&s0.tokens)?;
        let start = s0.completion_start - 1;
        let end = s0.tokens.len() - 1;
        let model_masked: f32 = model_nlls[start..end].iter().sum::<f32>() / (end - start) as f32;
        let zero_b = vec![0.0f32; vocab * rank];
        let dummy_a = vec![0.0f32; rank * hidden];
        let cached_masked = eval_nll(&caches[..1], &dummy_a, &zero_b, rank, hidden, vocab, scale);
        let observation = verify_tbv(
            "train_grad cache check (sample 0)",
            model_masked,
            cached_masked,
        )?;
        println!(
            "  cache check (sample 0): model={model_masked:.5}  cached={cached_masked:.5}  diff={:.2e}",
            observation.diff
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

#[cfg(test)]
mod cli_contract_tests {
    use super::*;

    fn args(extra: &[&str]) -> Vec<String> {
        let mut v = vec!["train_grad".to_string()];
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    }

    #[test]
    fn defaults_match_documented_table() {
        let a = args(&[]);
        let cfg = parse_config(&ArgView::new(&a)).expect("default config must parse");
        assert_eq!(cfg.data_dir, PathBuf::from("data/lora-train"));
        assert_eq!(cfg.steps, 150);
        assert_eq!(cfg.lr, 1e-3);
        assert_eq!(cfg.rank, 8);
        assert_eq!(cfg.alpha, 16.0);
        assert_eq!(cfg.seq_len, 96);
        assert_eq!(cfg.max_train, 6);
        assert_eq!(cfg.log_every, 10);
        assert_eq!(cfg.model_dir, train_common::default_model_dir());
    }

    #[test]
    fn explicit_flags_override_defaults() {
        let a = args(&[
            "--model-dir",
            "/tmp/m",
            "--data-dir",
            "/tmp/d",
            "--steps",
            "5",
            "--lr",
            "2e-4",
            "--rank",
            "4",
            "--alpha",
            "8",
            "--seq-len",
            "32",
            "--max-train",
            "2",
            "--log-every",
            "1",
        ]);
        let cfg = parse_config(&ArgView::new(&a)).expect("explicit config must parse");
        assert_eq!(cfg.model_dir, PathBuf::from("/tmp/m"));
        assert_eq!(cfg.data_dir, PathBuf::from("/tmp/d"));
        assert_eq!(cfg.steps, 5);
        assert_eq!(cfg.lr, 2e-4);
        assert_eq!(cfg.rank, 4);
        assert_eq!(cfg.alpha, 8.0);
        assert_eq!(cfg.seq_len, 32);
        assert_eq!(cfg.max_train, 2);
        assert_eq!(cfg.log_every, 1);
    }

    #[test]
    fn rejects_invalid_alpha_values() {
        for alpha in ["nan", "inf", "-inf", "invalid"] {
            let a = args(&["--alpha", alpha]);
            let err = parse_config(&ArgView::new(&a))
                .expect_err("invalid --alpha must fail parsing before training");
            assert!(err.contains("--alpha"));
        }
    }

    #[test]
    fn rejects_missing_alpha_value() {
        let a = args(&["--alpha"]);
        let err = parse_config(&ArgView::new(&a))
            .expect_err("--alpha without a value must fail parsing before training");
        assert!(err.contains("--alpha"));
    }

    #[test]
    fn help_flags_detected_via_arg_view() {
        let a = args(&["-h"]);
        assert!(ArgView::new(&a).flag("-h"));
        let a = args(&["--help"]);
        assert!(ArgView::new(&a).flag("--help"));
        let a = args(&[]);
        assert!(!ArgView::new(&a).flag("-h"));
        assert!(!ArgView::new(&a).flag("--help"));
    }
}
