// Multi-layer reverse-mode (exact) gradient LoRA trainer for Qwen3.5-0.8B.
//
// Generalises train_grad_layer23 from a single GQA layer to a layer RANGE
// [first_layer ..= 23], dispatching per layer kind and propagating dx through
// the frozen GatedDeltaNet layers via gdn_backward. This is the full-depth
// backward tape: gradient from the CE loss flows back through the head, through
// every materialised layer's FFN + mixer + norms, accumulating LoRA grads at the
// GQA layers and threading dx through the frozen GDN layers in between.
//
//   h_in (frozen prefix 0..first_layer)  ── captured once per sample
//   for L in first_layer ..= 23:
//     normed_pre = rms_norm(h, pre_norm[L])
//     mixer_out  = GQA(normed_pre; LoRA)  | GDN(normed_pre)   (frozen if GDN)
//     h_mid      = h + mixer_out
//     ffn_out    = swiglu(rms_norm(h_mid, post_norm[L]))
//     h          = h_mid + ffn_out
//   logits = lm_head · rms_norm(h, final_norm)
//   loss   = CE(logits, next_token) over completion positions
//
// Default first_layer = 19, so the materialised stack is
//   19 (GQA+LoRA) → 20,21,22 (GDN, frozen) → 23 (GQA+LoRA).
// Layer-19's LoRA gradient is only correct if dx propagated correctly back
// through the three frozen GDN layers, so the finite-difference gradcheck on
// layer-19's params is the integration test for gdn_backward + the assembled
// residual/norm chain — exactly the gap the per-block self-consistent
// gradchecks cannot see.
//
// Qwen3.5 RMSNorm is SHIFTED (x·inv·(1+gamma)); rms_norm_forward/rmsnorm_backward
// take plain gamma, so layer/final norms get (1+gamma) precomputed weights.
// q_norm/k_norm are shifted inside gqa_forward_with_cache, so they stay raw.
//
// Usage: train_grad_full --model-dir <path> --data-dir <path> [--first-layer 19]
//        [--steps 25] [--lr 1e-3] [--rank 8] [--alpha 16] [--max-train 3]
//        [--seq-len 64] [--gradcheck]

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::Instant;

use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::tokenizer::Tokenizer;
use lattice_tune::lora::AdamState;
use lattice_tune::lora::train_core::{
    Dims, GdnDims, Head, LayerW, LoraParams, MixerKind, SeqCtx, TOP_LAYER, apply_adam_updates,
    eval_chain_nll, forward_full, nll_and_grads, rand_fill, shifted,
};

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
        "usage: train_grad_full [OPTIONS]

Options:
  --model-dir   <PATH>   Model directory (default: $HOME/.lattice/models/qwen3.5-0.8b)
  --data-dir    <PATH>   Dataset directory with train.jsonl + valid.jsonl (default: data/lora-train)
  --first-layer <N>      First materialised (trained) layer (default: 19)
  --steps       <N>      Adam steps (default: 25)
  --lr          <F>      Learning rate (default: 1e-3)
  --rank        <N>      LoRA rank (default: 8)
  --alpha       <F>      LoRA alpha (default: 16.0)
  --seq-len     <N>      Max tokens per sample (default: 64)
  --max-train   <N>      Training samples cap (default: 3)
  --max-valid   <N>      Held-out valid.jsonl samples for eval, 0=off (default: 16)
  --log-every   <N>      Print NLL every N steps (default: 5)
  --save        <PATH>   Save trained adapter as a PEFT safetensors file (requires --features safetensors)
  --gradcheck            Run finite-difference gradcheck instead of training
  --probe       <N>      Gradcheck entries probed per array per layer (default: 6)
  -h, --help             Print this help"
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

/// Capture the frozen-prefix output (h_in entering `first_layer`) and RoPE
/// tables per sample, yielding the `SeqCtx` set the tape forward consumes.
/// Returns the caches plus the total number of masked completion positions.
fn build_caches(
    model: &Qwen35Model,
    samples: &[Sample],
    first_layer: usize,
) -> Result<(Vec<SeqCtx>, usize), Box<dyn std::error::Error>> {
    let mut caches = Vec::with_capacity(samples.len());
    let mut total_positions = 0usize;
    for s in samples {
        let (h_in, _real_out) = model.capture_attn_io(&s.tokens, first_layer)?;
        let seq_len = s.tokens.len();
        let (cos, sin) = model.rope_cos_sin_tables(seq_len)?;
        total_positions += seq_len - s.completion_start;
        caches.push(SeqCtx {
            h_in,
            cos,
            sin,
            tokens: s.tokens.clone(),
            completion_start: s.completion_start,
            seq_len,
        });
    }
    Ok((caches, total_positions))
}

/// Indices of the `k` entries with the largest |grad| — the most informative
/// (least vacuous) entries to finite-difference.
fn top_k_indices(grad: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..grad.len()).collect();
    idx.sort_by(|&a, &b| grad[b].abs().partial_cmp(&grad[a].abs()).unwrap());
    idx.truncate(k);
    idx
}

/// Deterministic strided probes that cover entries top-k would skip (e.g.
/// zeroed-by-bug entries that self-select out of top-analytic).
fn strided_probes(len: usize, count: usize, seed: u64) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count);
    let mut rng = seed;
    for _ in 0..count {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        out.push((rng as usize) % len);
    }
    out.sort_unstable();
    out.dedup();
    out
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
    let first_layer: usize = parse_arg(&args, "--first-layer")
        .and_then(|s| s.parse().ok())
        .unwrap_or(19);
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
    let max_valid: usize = parse_arg(&args, "--max-valid")
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let log_every: usize = parse_arg(&args, "--log-every")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    if log_every == 0 {
        return Err("--log-every must be >= 1".into());
    }
    let gradcheck = parse_flag(&args, "--gradcheck");
    let probe: usize = parse_arg(&args, "--probe")
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    // Central-difference step. On the real f32 model (hidden 1024, vocab 248320,
    // multi-layer GDN recurrence) the NLL carries ~1e-6 roundoff, so too-small a
    // step is roundoff-dominated. Optimal central-FD step ≈ cbrt(roundoff) ≈ 4e-3.
    let fd_eps: f32 = parse_arg(&args, "--fd-eps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(4e-3);
    let save_path = parse_arg(&args, "--save");

    if first_layer > TOP_LAYER {
        return Err(format!("--first-layer {first_layer} must be <= {TOP_LAYER}").into());
    }

    println!(
        "=== multi-layer exact-gradient LoRA trainer (layers {first_layer}..={TOP_LAYER}) ==="
    );
    println!("model-dir:  {}", model_dir.display());
    println!("data-dir:   {}", data_dir.display());
    println!("steps:      {steps}  lr: {lr}  rank: {rank}  alpha: {alpha}");
    println!(
        "max-train:  {max_train}  seq-len: {seq_len_cap}  mode: {}",
        if gradcheck { "gradcheck" } else { "train" }
    );
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
        q_dim: cfg.full_q_dim(),
        kv_dim: cfg.full_kv_dim(),
        eps: cfg.rms_norm_eps,
    };
    let gdn_dims = GdnDims::from_cfg(&cfg);
    let scale = alpha / rank as f32;
    println!(
        "  hidden={}  vocab={}  q_dim={}  kv_dim={}  inter={}",
        dims.hidden, dims.vocab, dims.q_dim, dims.kv_dim, dims.inter
    );

    let (lm_head_s, final_norm_s, _embed) = model.head_weights();
    let lm_head = lm_head_s.to_vec();
    let final_shift = shifted(final_norm_s);
    let head = Head {
        lm_head: &lm_head,
        final_shift: &final_shift,
    };

    // Build the materialised layer stack [first_layer ..= 23], assigning a LoRA
    // slot to each GQA layer. All weight slices are borrowed from `model`.
    let mut layers: Vec<LayerW> = Vec::new();
    let mut slot_layers: Vec<usize> = Vec::new(); // global layer index per slot
    for layer_idx in first_layer..=TOP_LAYER {
        if let Some((w_q, w_k, w_v, w_o, q_norm, k_norm, pre, post, gate, up, down)) =
            model.gqa_layer_weights(layer_idx)
        {
            let slot = slot_layers.len();
            slot_layers.push(layer_idx);
            layers.push(LayerW {
                kind: MixerKind::Gqa,
                w_q,
                w_k,
                w_v,
                w_o,
                q_norm,
                k_norm,
                gdn: None,
                pre_shift: shifted(pre),
                post_shift: shifted(post),
                w_gate: gate,
                w_up: up,
                w_down: down,
                lora_slot: Some(slot),
            });
        } else if let Some((gdn, pre, post, gate, up, down)) = model.gdn_layer_weights(layer_idx) {
            layers.push(LayerW {
                kind: MixerKind::Gdn,
                w_q: &[],
                w_k: &[],
                w_v: &[],
                w_o: &[],
                q_norm: &[],
                k_norm: &[],
                gdn: Some(gdn),
                pre_shift: shifted(pre),
                post_shift: shifted(post),
                w_gate: gate,
                w_up: up,
                w_down: down,
                lora_slot: None,
            });
        } else {
            return Err(
                format!("layer {layer_idx} is neither a Full+Dense nor GDN+Dense layer").into(),
            );
        }
    }
    let num_slots = slot_layers.len();
    let kinds: String = layers
        .iter()
        .map(|l| match l.kind {
            MixerKind::Gqa => 'A',
            MixerKind::Gdn => 'D',
        })
        .collect();
    println!(
        "  materialised {} layers [{}]: {kinds}  ({} GQA LoRA slots at layers {:?})",
        layers.len(),
        (first_layer..=TOP_LAYER)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(","),
        num_slots,
        slot_layers
    );
    if num_slots == 0 {
        return Err("no GQA layers in range — nothing to train".into());
    }

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

    // Capture the frozen prefix output (h_in entering first_layer) per sample.
    println!("\nBuilding frozen-prefix cache (layers 0..{first_layer})...");
    let tcache = Instant::now();
    let (caches, total_positions) = build_caches(&model, &train_samples, first_layer)?;
    let logits_bytes = total_positions * dims.vocab * 4;
    const MAX_LOGITS_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GiB
    if logits_bytes > MAX_LOGITS_BYTES {
        return Err(format!(
            "logits buffer would require {} MiB ({} positions × {} vocab × 4B), \
             exceeds 2 GiB cap — reduce --seq-len or --max-train",
            logits_bytes / (1024 * 1024),
            total_positions,
            dims.vocab,
        )
        .into());
    }
    println!(
        "  {} completion positions across {} samples in {:.1}s",
        total_positions,
        caches.len(),
        tcache.elapsed().as_secs_f64()
    );

    // Held-out validation caches (valid.jsonl) — eval-only, never trained on.
    // The honest signal: train NLL falling while held-out NLL also falls is
    // learning; train NLL falling while held-out rises is memorisation.
    let valid_caches: Vec<SeqCtx> = if max_valid > 0 {
        match load_jsonl(
            &data_dir.join("valid.jsonl"),
            &tokenizer as &dyn Tokenizer,
            seq_len_cap,
            max_valid,
        ) {
            Ok(vs) if !vs.is_empty() => {
                let (vc, vpos) = build_caches(&model, &vs, first_layer)?;
                println!(
                    "  held-out: {vpos} completion positions across {} valid samples",
                    vc.len()
                );
                vc
            }
            _ => {
                println!("  held-out: valid.jsonl absent/empty — eval disabled");
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    // TBV: with zero LoRA (delta exactly 0), the chain NLL must match the real
    // model's own compute_token_nlls — validates the whole assembled forward
    // (every layer's shifted norms, GQA/GDN mixer, FFN, head) against the model.
    let zero_loras: Vec<LoraParams> = (0..num_slots)
        .map(|_| LoraParams::zeros(rank, &dims))
        .collect::<Result<Vec<_>, _>>()?;
    {
        let s0 = &train_samples[0];
        let model_nlls = model.compute_token_nlls(&s0.tokens)?;
        let start = s0.completion_start - 1;
        let end = s0.tokens.len() - 1;
        let model_masked: f32 = model_nlls[start..end].iter().sum::<f32>() / (end - start) as f32;
        let chain_masked = eval_chain_nll(
            &caches[..1],
            &layers,
            &zero_loras,
            &head,
            &dims,
            &gdn_dims,
            &cfg,
            rank,
            scale,
        )?;
        let diff = (model_masked - chain_masked).abs();
        println!(
            "\n  TBV (sample 0): model={model_masked:.5}  chain={chain_masked:.5}  diff={diff:.2e}"
        );
        if diff > 1e-2 {
            return Err(format!(
                "TBV failed: assembled chain NLL diverges from real model by {diff:.3e} (>1e-2)"
            )
            .into());
        }
    }

    // ---- Gradcheck mode: finite-difference the assembled tape's LoRA grads ----
    if gradcheck {
        println!("\n=== finite-difference gradcheck (probe {probe} entries/array/slot) ===");
        // Non-zero A AND B so grad_A and the gate path are non-vacuous.
        let mut rng = 0x1234_5678u64;
        let mut loras: Vec<LoraParams> = (0..num_slots)
            .map(|_| LoraParams {
                a_q: rand_fill(&mut rng, rank * dims.hidden, 0.05),
                b_q: rand_fill(&mut rng, 2 * dims.q_dim * rank, 0.05),
                a_v: rand_fill(&mut rng, rank * dims.hidden, 0.05),
                b_v: rand_fill(&mut rng, dims.kv_dim * rank, 0.05),
            })
            .collect();

        let fwd = forward_full(
            &caches[0], &layers, &loras, &head, &dims, &gdn_dims, &cfg, rank, scale,
        )?;
        let (_, _, analytic) =
            nll_and_grads(&fwd, &layers, &loras, &head, &dims, num_slots, rank, scale)?;

        println!("  fd-eps center {fd_eps:.0e}  (per-entry min over 0.25/0.5/1/2x)");
        let mut worst = 0.0f64;
        let mut all_pass = true;
        for slot in 0..num_slots {
            let layer_idx = slot_layers[slot];
            for (name, alen) in [
                ("a_q", analytic[slot].a_q.len()),
                ("b_q", analytic[slot].b_q.len()),
                ("a_v", analytic[slot].a_v.len()),
                ("b_v", analytic[slot].b_v.len()),
            ] {
                let agrad = match name {
                    "a_q" => &analytic[slot].a_q,
                    "b_q" => &analytic[slot].b_q,
                    "a_v" => &analytic[slot].a_v,
                    _ => &analytic[slot].b_v,
                };
                let mut idxs = top_k_indices(agrad, probe.min(alen));
                let seed = (slot as u64 * 4
                    + match name {
                        "a_q" => 0,
                        "b_q" => 1,
                        "a_v" => 2,
                        _ => 3,
                    })
                    ^ 0xDEAD;
                for p in strided_probes(alen, probe.min(alen), seed) {
                    if !idxs.contains(&p) {
                        idxs.push(p);
                    }
                }
                let mut max_rel = 0.0f64;
                let mut sum_rel = 0.0f64;
                for &k in &idxs {
                    let a = agrad[k];
                    // central FD on the chain NLL
                    let save = {
                        let arr = match name {
                            "a_q" => &mut loras[slot].a_q,
                            "b_q" => &mut loras[slot].b_q,
                            "a_v" => &mut loras[slot].a_v,
                            _ => &mut loras[slot].b_v,
                        };
                        arr[k]
                    };
                    let bump = |loras: &mut [LoraParams], val: f32| {
                        let arr = match name {
                            "a_q" => &mut loras[slot].a_q,
                            "b_q" => &mut loras[slot].b_q,
                            "a_v" => &mut loras[slot].a_v,
                            _ => &mut loras[slot].b_v,
                        };
                        arr[k] = val;
                    };
                    // min-over-eps: central FD has an unknown optimal step.
                    // If the analytic grad is correct, SOME step agrees well; if
                    // it is a structural error, NONE do — the per-entry min stays
                    // at the bug floor. Taking the min removes the arbitrary step
                    // choice — the honest correct-vs-buggy discriminator on a deep
                    // f32 backprop chain where any single step is roundoff- or
                    // truncation-dominated.
                    let eps_set = [fd_eps * 0.25, fd_eps * 0.5, fd_eps, fd_eps * 2.0];
                    let mut best = f64::INFINITY;
                    for &e in &eps_set {
                        bump(&mut loras, save + e);
                        let lp = eval_chain_nll(
                            &caches[..1],
                            &layers,
                            &loras,
                            &head,
                            &dims,
                            &gdn_dims,
                            &cfg,
                            rank,
                            scale,
                        )?;
                        bump(&mut loras, save - e);
                        let lm = eval_chain_nll(
                            &caches[..1],
                            &layers,
                            &loras,
                            &head,
                            &dims,
                            &gdn_dims,
                            &cfg,
                            rank,
                            scale,
                        )?;
                        let fd = (lp - lm) / (2.0 * e);
                        let rel = (a - fd).abs() as f64 / (a.abs().max(fd.abs()).max(1e-6)) as f64;
                        best = best.min(rel);
                    }
                    bump(&mut loras, save);
                    max_rel = max_rel.max(best);
                    sum_rel += best;
                }
                let mean_rel = sum_rel / idxs.len().max(1) as f64;
                worst = worst.max(max_rel);
                let ok = max_rel < 2e-2;
                all_pass &= ok;
                println!(
                    "  layer {layer_idx:2} slot {slot} {name:<3}: mean {mean_rel:.2e}  max {max_rel:.2e}  {}",
                    if ok { "ok" } else { "FAIL" }
                );
            }
        }
        println!("\n  worst rel-err across all probed entries: {worst:.2e}");
        if all_pass {
            println!(
                "PASS: assembled full-depth tape matches finite-difference (< 2e-2 min-over-eps)"
            );
            println!("      → dx propagates correctly through the frozen GDN layers into the");
            println!("        lower GQA layer's LoRA gradient.");
        } else {
            println!("FAIL: analytic gradient diverges from finite-difference");
            std::process::exit(1);
        }
        return Ok(());
    }

    // ---- Training mode ----
    // LoRA init: A ~ U(-1/sqrt(in), +1/sqrt(in)), B zero (delta=0 at init
    // reproduces the base; grad_B != 0 so B moves first). The 1/sqrt(in)
    // amplitude matches mlx_lm LoRALinear (tuner/lora.py) for on-par convergence.
    let init_amp = 1.0 / (dims.hidden as f32).sqrt();
    let mut rng = 0xFEED_FACEu64;
    let mut loras: Vec<LoraParams> = (0..num_slots)
        .map(|_| LoraParams {
            a_q: rand_fill(&mut rng, rank * dims.hidden, init_amp),
            b_q: vec![0.0; 2 * dims.q_dim * rank],
            a_v: rand_fill(&mut rng, rank * dims.hidden, init_amp),
            b_v: vec![0.0; dims.kv_dim * rank],
        })
        .collect();

    let mut adam = AdamState::new();
    let (beta1, beta2, eps_adam) = (0.9f32, 0.999f32, 1e-8f32);

    let eval_valid = |loras: &[LoraParams]| -> Result<Option<f32>, Box<dyn std::error::Error>> {
        if valid_caches.is_empty() {
            return Ok(None);
        }
        Ok(Some(eval_chain_nll(
            &valid_caches,
            &layers,
            loras,
            &head,
            &dims,
            &gdn_dims,
            &cfg,
            rank,
            scale,
        )?))
    };

    let base_nll = eval_chain_nll(
        &caches, &layers, &loras, &head, &dims, &gdn_dims, &cfg, rank, scale,
    )?;
    let base_valid = eval_valid(&loras)?;
    match base_valid {
        Some(v) => println!("\n  step    0  train NLL: {base_nll:.4}  held-out NLL: {v:.4}"),
        None => println!("\n  step    0  train NLL: {base_nll:.4}"),
    }

    let tstep = Instant::now();
    for step in 1..=steps {
        let ctx = &caches[(step - 1) % caches.len()];
        let fwd = forward_full(
            ctx, &layers, &loras, &head, &dims, &gdn_dims, &cfg, rank, scale,
        )?;
        let (_nll, _n, grads) =
            nll_and_grads(&fwd, &layers, &loras, &head, &dims, num_slots, rank, scale)?;

        apply_adam_updates(
            &mut adam,
            &mut loras,
            &grads,
            &slot_layers,
            lr,
            beta1,
            beta2,
            eps_adam,
        );

        if step % log_every == 0 || step == steps {
            let mean_nll = eval_chain_nll(
                &caches, &layers, &loras, &head, &dims, &gdn_dims, &cfg, rank, scale,
            )?;
            match eval_valid(&loras)? {
                Some(v) => println!(
                    "  step {step:4}  train NLL: {mean_nll:.4}  held-out NLL: {v:.4}  (train d {:+.4})",
                    mean_nll - base_nll
                ),
                None => println!(
                    "  step {step:4}  train NLL: {mean_nll:.4}  (delta from base: {:+.4})",
                    mean_nll - base_nll
                ),
            }
        }
    }

    let final_nll = eval_chain_nll(
        &caches, &layers, &loras, &head, &dims, &gdn_dims, &cfg, rank, scale,
    )?;
    let secs = tstep.elapsed().as_secs_f64();
    match (base_valid, eval_valid(&loras)?) {
        (Some(b), Some(f)) => println!(
            "\n=== done: train {base_nll:.4}→{final_nll:.4} ({:+.4})  |  held-out {b:.4}→{f:.4} ({:+.4})  in {secs:.1}s ===",
            final_nll - base_nll,
            f - b
        ),
        _ => println!(
            "\n=== done: base NLL {base_nll:.4} → final NLL {final_nll:.4} ({:+.4}) in {secs:.1}s ===",
            final_nll - base_nll
        ),
    }

    if let Some(ref path) = save_path {
        #[cfg(feature = "safetensors")]
        {
            use lattice_tune::lora::{LoraAdapter, LoraConfig, LoraLayer};
            use std::collections::HashMap;
            let mut adapter_layers = HashMap::new();
            for (slot, &li) in slot_layers.iter().enumerate() {
                adapter_layers.insert(
                    (li, "q_proj".to_string()),
                    LoraLayer {
                        a: loras[slot].a_q.clone(),
                        b: loras[slot].b_q.clone(),
                        d_in: dims.hidden,
                        d_out: 2 * dims.q_dim,
                        rank,
                    },
                );
                adapter_layers.insert(
                    (li, "v_proj".to_string()),
                    LoraLayer {
                        a: loras[slot].a_v.clone(),
                        b: loras[slot].b_v.clone(),
                        d_in: dims.hidden,
                        d_out: dims.kv_dim,
                        rank,
                    },
                );
            }
            let config = LoraConfig {
                rank,
                alpha,
                target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            };
            let adapter = LoraAdapter::new(config, adapter_layers);
            adapter
                .save_safetensors(std::path::Path::new(path))
                .map_err(|e| format!("save adapter: {e}"))?;
            println!("saved adapter ({num_slots} GQA slots, rank {rank}) → {path}");
        }
        #[cfg(not(feature = "safetensors"))]
        {
            let _ = path;
            return Err("--save requires the safetensors feature (--features safetensors)".into());
        }
    }

    Ok(())
}
