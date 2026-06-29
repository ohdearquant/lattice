// Reverse-mode (exact) gradient LoRA trainer for Qwen3.5-0.8B — layer 23.
//
// Milestone-2: rank-r LoRA on layer 23's q_proj and v_proj (the top GQA layer,
// so no GatedDeltaNet backward is involved). Unlike the lm_head trainer, the
// gradient flows through a full transformer block + the head:
//
//   h_in (frozen, layers 0..22)  ── cached once per sample
//   normed   = rms_norm(h_in,  pre_attn_norm)
//   attn_out = gated_GQA(normed; LoRA_q, LoRA_v)        ← trained
//   h_mid    = h_in + attn_out
//   ffn_out  = swiglu(rms_norm(h_mid, post_attn_norm))
//   h_out    = h_mid + ffn_out
//   logits   = lm_head · rms_norm(h_out, final_norm)
//   loss     = CE(logits, next_token)  over completion positions
//
// The frozen prefix (layers 0..22) is captured once via capture_attn_io, which
// returns h_in = the residual entering layer 23. Everything below layer 23 and
// every base weight is frozen; only the four LoRA factors move.
//
// Qwen3.5 RMSNorm is SHIFTED (x·inv·(1+gamma)); rms_norm_forward/rmsnorm_backward
// use plain gamma, so the layer/final norms get (1+gamma) precomputed weights.
// q_norm/k_norm are shifted inside gqa_forward_with_cache, so they stay raw.
//
// Usage: train_grad_layer23 --model-dir <path> --data-dir <path> [--steps 25]
//        [--lr 1e-3] [--rank 8] [--alpha 16] [--max-train 3] [--seq-len 64]

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::Instant;

use lattice_inference::backward::attention_gqa::{gqa_backward, gqa_forward_with_cache};
use lattice_inference::backward::ops::{linear_vjp, rmsnorm_backward, swiglu_backward};
use lattice_inference::backward::tape::{rms_norm_forward, swiglu_forward};
use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::tokenizer::Tokenizer;
use lattice_tune::lora::AdamState;

/// Top GQA (Full-attention) layer of Qwen3.5-0.8B. GQA layers are at
/// [3, 7, 11, 15, 19, 23]; 23 is the top one, so it needs no GDN backward.
const LAYER: usize = 23;

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
        "usage: train_grad_layer23 [OPTIONS]

Options:
  --model-dir  <PATH>   Model directory (default: $HOME/.lattice/models/qwen3.5-0.8b)
  --data-dir   <PATH>   Dataset directory with train.jsonl (default: data/lora-train)
  --steps      <N>      Adam steps (default: 25)
  --lr         <F>      Learning rate (default: 1e-3)
  --rank       <N>      LoRA rank (default: 8)
  --alpha      <F>      LoRA alpha (default: 16.0)
  --seq-len    <N>      Max tokens per sample (default: 64)
  --max-train  <N>      Training samples cap (default: 3)
  --log-every  <N>      Print NLL every N steps (default: 5)
  --save       <PATH>   Save trained adapter as PEFT safetensors (requires --features train-backward)
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

/// One sample's frozen layer-23 context: the residual entering layer 23 (from
/// the frozen prefix), the RoPE tables for its length, and the targets. All
/// fixed across training — only the LoRA factors change.
struct L23Cache {
    h_in: Vec<f32>, // [seq_len * hidden] residual entering layer 23
    cos: Vec<f32>,
    sin: Vec<f32>,
    tokens: Vec<u32>,
    completion_start: usize,
    seq_len: usize,
}

/// Borrowed frozen weights for layer 23 + head. Norm `*_shift` fields are the
/// `(1 + gamma)` shifted layer norms; `q_norm`/`k_norm` are raw (gqa shifts).
struct L23Weights<'a> {
    w_q: &'a [f32],
    w_k: &'a [f32],
    w_v: &'a [f32],
    w_o: &'a [f32],
    q_norm: &'a [f32],
    k_norm: &'a [f32],
    w_gate: &'a [f32],
    w_up: &'a [f32],
    w_down: &'a [f32],
    lm_head: &'a [f32],
    pre_shift: &'a [f32],
    post_shift: &'a [f32],
    final_shift: &'a [f32],
}

struct Lora<'a> {
    a_q: &'a [f32],
    b_q: &'a [f32],
    a_v: &'a [f32],
    b_v: &'a [f32],
    rank: usize,
    scale: f32,
}

#[derive(Clone, Copy)]
struct Dims {
    hidden: usize,
    vocab: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    inter: usize,
    eps: f32,
}

/// Per completion source-position forward intermediates needed by the backward.
struct PosFwd {
    t: usize,           // source position (predicts tokens[t+1])
    target: u32,        // tokens[t+1]
    logits: Vec<f32>,   // [vocab]
    post_ffn: Vec<f32>, // [hidden] final_norm input
    inv_final: f32,
    post_attn: Vec<f32>, // [hidden] post_attn_norm input
    inv_ffn: f32,
    gate_pre: Vec<f32>, // [inter] swiglu cache
    up_pre: Vec<f32>,   // [inter]
}

/// Layer-23 + head forward with the current LoRA. Returns the attention cache
/// (for gqa_backward), the full post-attn residual stream, and per-completion
/// intermediates. The pre_attn_norm is computed but its backward is skipped:
/// dx below layer 23 is discarded because the prefix is frozen.
struct L23Fwd {
    attn_cache: lattice_inference::backward::attention_gqa::AttnCache,
    positions: Vec<PosFwd>,
}

fn lm_head_logits(lm_head: &[f32], final_normed: &[f32], hidden: usize, vocab: usize) -> Vec<f32> {
    let mut logits = vec![0.0f32; vocab];
    for (i, l) in logits.iter_mut().enumerate() {
        *l = lm_head[i * hidden..(i + 1) * hidden]
            .iter()
            .zip(final_normed.iter())
            .map(|(w, x)| w * x)
            .sum();
    }
    logits
}

fn forward_layer23(cache: &L23Cache, w: &L23Weights, lora: &Lora, d: &Dims) -> L23Fwd {
    let h = d.hidden;
    let seq_len = cache.seq_len;

    // pre_attn_norm (shifted) per position → attention input.
    let mut normed_pre_attn = vec![0.0f32; seq_len * h];
    for t in 0..seq_len {
        let x_t = &cache.h_in[t * h..(t + 1) * h];
        let (n, _inv) = rms_norm_forward(x_t, w.pre_shift, d.eps);
        normed_pre_attn[t * h..(t + 1) * h].copy_from_slice(&n);
    }

    let (attn_out_seq, attn_cache) = gqa_forward_with_cache(
        &normed_pre_attn,
        w.w_q,
        w.w_k,
        w.w_v,
        w.w_o,
        w.q_norm,
        w.k_norm,
        Some(lora.a_q),
        Some(lora.b_q),
        Some(lora.a_v),
        Some(lora.b_v),
        lora.rank,
        lora.scale,
        seq_len,
        h,
        d.num_q_heads,
        d.num_kv_heads,
        d.head_dim,
        d.rope_dim,
        &cache.cos,
        &cache.sin,
        d.eps,
    );

    // h_mid = h_in + attn_out (full sequence; needed at every position for the
    // attention backward's residual, computed at completion positions for FFN).
    let mut post_attn = cache.h_in.clone();
    for (a, b) in post_attn.iter_mut().zip(attn_out_seq.iter()) {
        *a += *b;
    }

    // Completion source positions [completion_start-1, seq_len-1): predict t+1.
    let mut positions = Vec::new();
    for t in (cache.completion_start - 1)..seq_len - 1 {
        let h_mid_t = &post_attn[t * h..(t + 1) * h];
        let (normed_ffn, inv_ffn) = rms_norm_forward(h_mid_t, w.post_shift, d.eps);
        let (ffn_out, gate_pre, up_pre) =
            swiglu_forward(&normed_ffn, w.w_gate, w.w_up, w.w_down, h, d.inter);

        let mut h_out = h_mid_t.to_vec();
        for (o, f) in h_out.iter_mut().zip(ffn_out.iter()) {
            *o += *f;
        }
        let (final_normed, inv_final) = rms_norm_forward(&h_out, w.final_shift, d.eps);
        let logits = lm_head_logits(w.lm_head, &final_normed, h, d.vocab);

        positions.push(PosFwd {
            t,
            target: cache.tokens[t + 1],
            logits,
            post_ffn: h_out,
            inv_final,
            post_attn: h_mid_t.to_vec(),
            inv_ffn,
            gate_pre,
            up_pre,
        });
    }

    L23Fwd {
        attn_cache,
        positions,
    }
}

/// CE NLL for one position's logits against `target` (numerically stable).
fn position_nll(logits: &[f32], target: u32) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    let log_prob = (logits[target as usize] - max) as f64 - sum_exp.ln();
    (-log_prob) as f32
}

fn eval_nll(caches: &[L23Cache], w: &L23Weights, lora: &Lora, d: &Dims) -> f32 {
    let mut nll_sum = 0.0f64;
    let mut n = 0usize;
    for cache in caches {
        let fwd = forward_layer23(cache, w, lora, d);
        for p in &fwd.positions {
            nll_sum += position_nll(&p.logits, p.target) as f64;
            n += 1;
        }
    }
    (nll_sum / n.max(1) as f64) as f32
}

/// LoRA gradients accumulated across one sample.
struct Grads {
    a_q: Vec<f32>,
    b_q: Vec<f32>,
    a_v: Vec<f32>,
    b_v: Vec<f32>,
}

fn nll_and_grads(cache: &L23Cache, w: &L23Weights, lora: &Lora, d: &Dims) -> (f32, usize, Grads) {
    let h = d.hidden;
    let seq_len = cache.seq_len;
    let fwd = forward_layer23(cache, w, lora, d);
    let n_comp = fwd.positions.len().max(1) as f32;

    // d/d(logits) of CE, masked to completion + normalized by n_comp.
    let mut d_post_ffn = vec![0.0f32; seq_len * h];
    let mut nll_sum = 0.0f64;
    for p in &fwd.positions {
        nll_sum += position_nll(&p.logits, p.target) as f64;
        let max = p.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = p.logits.iter().map(|&l| (l - max).exp()).sum();
        let mut d_logits = vec![0.0f32; d.vocab];
        for (v, dl) in d_logits.iter_mut().enumerate() {
            let prob = (p.logits[v] - max).exp() / sum_exp;
            let indicator = if v == p.target as usize { 1.0 } else { 0.0 };
            *dl = (prob - indicator) / n_comp;
        }
        // lm_head VJP → final_norm backward → into the residual at position t.
        let d_final = linear_vjp(w.lm_head, &d_logits, h, d.vocab);
        let d_h_out = rmsnorm_backward(&p.post_ffn, w.final_shift, p.inv_final, &d_final);
        d_post_ffn[p.t * h..(p.t + 1) * h].copy_from_slice(&d_h_out);
    }

    // FFN backward → post_attn_norm backward → accumulate onto the residual.
    let mut d_attn_out = d_post_ffn.clone();
    for p in &fwd.positions {
        let d_ffn_out = &d_post_ffn[p.t * h..(p.t + 1) * h];
        let (d_normed_ffn, _dm) = swiglu_backward(
            d_ffn_out,
            &p.gate_pre,
            &p.up_pre,
            w.w_down,
            w.w_gate,
            w.w_up,
            h,
            d.inter,
        );
        let d_h_mid = rmsnorm_backward(&p.post_attn, w.post_shift, p.inv_ffn, &d_normed_ffn);
        for (j, dv) in d_h_mid.iter().enumerate() {
            d_attn_out[p.t * h + j] += *dv;
        }
    }

    // Attention backward. dx (grad to the frozen prefix) is discarded.
    let g = gqa_backward(
        &d_attn_out,
        &fwd.attn_cache,
        w.w_q,
        w.w_k,
        w.w_v,
        w.w_o,
        w.q_norm,
        w.k_norm,
        Some(lora.a_q),
        Some(lora.b_q),
        Some(lora.a_v),
        Some(lora.b_v),
        lora.rank,
        lora.scale,
    );

    (
        nll_sum as f32,
        fwd.positions.len(),
        Grads {
            a_q: g.grad_a_q,
            b_q: g.grad_b_q,
            a_v: g.grad_a_v,
            b_v: g.grad_b_v,
        },
    )
}

fn shifted(gamma: &[f32]) -> Vec<f32> {
    gamma.iter().map(|g| 1.0 + g).collect()
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
        .unwrap_or_else(|| PathBuf::from("data/lora-train"));
    let steps: usize = parse_arg(&args, "--steps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    let lr: f32 = parse_arg(&args, "--lr")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-3);
    let rank: usize = parse_arg(&args, "--rank")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let alpha: f32 = parse_arg(&args, "--alpha")
        .and_then(|s| s.parse().ok())
        .unwrap_or(16.0);
    let seq_len_cap: usize = parse_arg(&args, "--seq-len")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let max_train: usize = parse_arg(&args, "--max-train")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let log_every: usize = parse_arg(&args, "--log-every")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let save_path = parse_arg(&args, "--save");

    println!("=== exact-gradient LoRA trainer (layer {LAYER}, gated GQA) ===");
    println!("model-dir:  {}", model_dir.display());
    println!("data-dir:   {}", data_dir.display());
    println!("steps:      {steps}  lr: {lr}  rank: {rank}  alpha: {alpha}");
    println!("max-train:  {max_train}  seq-len: {seq_len_cap}");
    println!();

    println!("Loading model...");
    let t0 = Instant::now();
    let model =
        Qwen35Model::from_safetensors(&model_dir).map_err(|e| format!("load model: {e}"))?;
    println!("  loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let cfg = model.config().clone();
    let dims = Dims {
        hidden: cfg.hidden_size,
        vocab: cfg.vocab_size,
        num_q_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        rope_dim: cfg.rope_dim(),
        inter: cfg.intermediate_size,
        eps: cfg.rms_norm_eps,
    };
    let scale = alpha / rank as f32;
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();
    println!(
        "  hidden={}  vocab={}  q_dim={q_dim}  kv_dim={kv_dim}  inter={}",
        dims.hidden, dims.vocab, dims.inter
    );

    let (lm_head_s, final_norm_s, _embed) = model.head_weights();
    let lm_head = lm_head_s.to_vec();
    let final_shift = shifted(final_norm_s);

    let (w_q, w_k, w_v, w_o, q_norm, k_norm, pre_attn_norm, post_attn_norm, w_gate, w_up, w_down) =
        model
            .gqa_layer_weights(LAYER)
            .ok_or("layer 23 is not a Full+Dense GQA layer")?;
    let (w_q, w_k, w_v, w_o) = (w_q.to_vec(), w_k.to_vec(), w_v.to_vec(), w_o.to_vec());
    let (q_norm, k_norm) = (q_norm.to_vec(), k_norm.to_vec());
    let (w_gate, w_up, w_down) = (w_gate.to_vec(), w_up.to_vec(), w_down.to_vec());
    let pre_shift = shifted(pre_attn_norm);
    let post_shift = shifted(post_attn_norm);

    let weights = L23Weights {
        w_q: &w_q,
        w_k: &w_k,
        w_v: &w_v,
        w_o: &w_o,
        q_norm: &q_norm,
        k_norm: &k_norm,
        w_gate: &w_gate,
        w_up: &w_up,
        w_down: &w_down,
        lm_head: &lm_head,
        pre_shift: &pre_shift,
        post_shift: &post_shift,
        final_shift: &final_shift,
    };

    let tokenizer = model.tokenizer().clone();
    println!("\nLoading dataset from {}...", data_dir.display());
    let train_samples = load_jsonl(
        &data_dir.join("train.jsonl"),
        &tokenizer as &dyn Tokenizer,
        seq_len_cap,
        max_train,
    )?;
    if train_samples.is_empty() {
        return Err("no training samples (check --data-dir, train.jsonl prompt+completion)".into());
    }
    println!("  {} samples loaded", train_samples.len());

    // Cache the frozen prefix output (h_in entering layer 23) per sample.
    println!("\nBuilding layer-{LAYER} cache (frozen prefix forward)...");
    let tcache = Instant::now();
    let mut caches = Vec::new();
    let mut total_positions = 0usize;
    for s in &train_samples {
        let (h_in, _real_out) = model.capture_attn_io(&s.tokens, LAYER)?;
        let seq_len = s.tokens.len();
        let (cos, sin) = model.rope_cos_sin_tables(seq_len);
        let n_comp = seq_len - s.completion_start;
        total_positions += n_comp;
        caches.push(L23Cache {
            h_in,
            cos,
            sin,
            tokens: s.tokens.clone(),
            completion_start: s.completion_start,
            seq_len,
        });
    }
    println!(
        "  {} completion positions across {} samples in {:.1}s",
        total_positions,
        caches.len(),
        tcache.elapsed().as_secs_f64()
    );

    // LoRA init: A small random, B zero. At B=0 the delta is zero so the forward
    // reproduces the base model (TBV below); grad_B = scale·outer(g, A·x) ≠ 0 at
    // init, so B moves first, then A.
    let mut rng = 0xFEED_FACEu64;
    let mut rand_small = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                ((rng >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * 0.02
            })
            .collect()
    };
    let mut lora_a_q = rand_small(rank * dims.hidden);
    let mut lora_b_q = vec![0.0f32; 2 * q_dim * rank];
    let mut lora_a_v = rand_small(rank * dims.hidden);
    let mut lora_b_v = vec![0.0f32; kv_dim * rank];

    // TBV: with zero LoRA, the chain NLL must match the real model's own
    // compute_token_nlls — validates the whole layer-23 + head chain (shifted
    // norms, gate, FFN, lm_head) against the model, not just self-consistency.
    {
        let zero_lora = Lora {
            a_q: &lora_a_q,
            b_q: &lora_b_q,
            a_v: &lora_a_v,
            b_v: &lora_b_v,
            rank,
            scale,
        };
        let s0 = &train_samples[0];
        let model_nlls = model.compute_token_nlls(&s0.tokens)?;
        let start = s0.completion_start - 1;
        let end = s0.tokens.len() - 1;
        let model_masked: f32 = model_nlls[start..end].iter().sum::<f32>() / (end - start) as f32;
        let chain_masked = eval_nll(&caches[..1], &weights, &zero_lora, &dims);
        let diff = (model_masked - chain_masked).abs();
        println!(
            "\n  TBV (sample 0): model={model_masked:.5}  chain={chain_masked:.5}  diff={diff:.2e}"
        );
        if diff > 1e-2 {
            return Err(format!(
                "TBV failed: chain NLL diverges from real model by {diff:.3e} (>1e-2) — \
                 the layer-23 forward does not match the model"
            )
            .into());
        }
    }

    let mut adam = AdamState::new();
    let (beta1, beta2, eps_adam) = (0.9f32, 0.999f32, 1e-8f32);

    let lora0 = Lora {
        a_q: &lora_a_q,
        b_q: &lora_b_q,
        a_v: &lora_a_v,
        b_v: &lora_b_v,
        rank,
        scale,
    };
    let base_nll = eval_nll(&caches, &weights, &lora0, &dims);
    println!("\n  step    0  train NLL: {base_nll:.4}");

    let tstep = Instant::now();
    for step in 1..=steps {
        let cache = &caches[(step - 1) % caches.len()];
        let lora = Lora {
            a_q: &lora_a_q,
            b_q: &lora_b_q,
            a_v: &lora_a_v,
            b_v: &lora_b_v,
            rank,
            scale,
        };
        let (_nll, _n, grads) = nll_and_grads(cache, &weights, &lora, &dims);

        adam.step(
            "l23_a_q",
            &mut lora_a_q,
            &grads.a_q,
            lr,
            beta1,
            beta2,
            eps_adam,
            0.0,
            false,
        );
        adam.step(
            "l23_b_q",
            &mut lora_b_q,
            &grads.b_q,
            lr,
            beta1,
            beta2,
            eps_adam,
            0.0,
            false,
        );
        adam.step(
            "l23_a_v",
            &mut lora_a_v,
            &grads.a_v,
            lr,
            beta1,
            beta2,
            eps_adam,
            0.0,
            false,
        );
        adam.step(
            "l23_b_v",
            &mut lora_b_v,
            &grads.b_v,
            lr,
            beta1,
            beta2,
            eps_adam,
            0.0,
            false,
        );

        if step % log_every == 0 || step == steps {
            let lora_e = Lora {
                a_q: &lora_a_q,
                b_q: &lora_b_q,
                a_v: &lora_a_v,
                b_v: &lora_b_v,
                rank,
                scale,
            };
            let mean_nll = eval_nll(&caches, &weights, &lora_e, &dims);
            println!(
                "  step {step:4}  train NLL: {mean_nll:.4}  (delta from base: {:+.4})",
                mean_nll - base_nll
            );
        }
    }

    let lora_f = Lora {
        a_q: &lora_a_q,
        b_q: &lora_b_q,
        a_v: &lora_a_v,
        b_v: &lora_b_v,
        rank,
        scale,
    };
    let final_nll = eval_nll(&caches, &weights, &lora_f, &dims);
    println!(
        "\n=== done: base NLL {base_nll:.4} → final NLL {final_nll:.4} ({:+.4}) in {:.1}s ===",
        final_nll - base_nll,
        tstep.elapsed().as_secs_f64()
    );

    // Persist the trained low-rank factors as a PEFT-format adapter. The trainer
    // stores B as row-major [d_out, rank] and A as [rank, d_in] (see ops::lora_vjp),
    // which is exactly LoraLayer's convention -- no transpose. q_proj is gated, so
    // its d_out spans the full 2*q_dim output (Q rows + gate rows).
    if let Some(save_path) = save_path.as_deref() {
        #[cfg(feature = "safetensors")]
        {
            use lattice_tune::lora::{LoraAdapter, LoraConfig, LoraLayer};
            use std::collections::HashMap;
            let mut layers = HashMap::new();
            layers.insert(
                (LAYER, "q_proj".to_string()),
                LoraLayer {
                    a: lora_a_q.clone(),
                    b: lora_b_q.clone(),
                    d_in: dims.hidden,
                    d_out: 2 * q_dim,
                    rank,
                },
            );
            layers.insert(
                (LAYER, "v_proj".to_string()),
                LoraLayer {
                    a: lora_a_v.clone(),
                    b: lora_b_v.clone(),
                    d_in: dims.hidden,
                    d_out: kv_dim,
                    rank,
                },
            );
            let config = LoraConfig {
                rank,
                alpha,
                target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            };
            let adapter = LoraAdapter::new(config, layers);
            adapter
                .save_safetensors(std::path::Path::new(save_path))
                .map_err(|e| format!("save adapter: {e}"))?;
            println!("saved adapter (layer {LAYER}, q_proj+v_proj, rank {rank}) → {save_path}");
        }
        #[cfg(not(feature = "safetensors"))]
        {
            let _ = save_path;
            return Err("--save requires building with --features train-backward".into());
        }
    }

    Ok(())
}
