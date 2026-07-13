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

use std::path::PathBuf;
use std::time::Instant;

use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::tokenizer::Tokenizer;
use lattice_tune::lora::AdamState;
use lattice_tune::lora::train_core::{
    AdamConfig, Dims, GdnDims, GdnLoraParams, Head, LayerW, LoraParams, MixerKind, SeqCtx,
    SlotLayout, TOP_LAYER, TapeGeometry, TrainCtx, apply_adam_updates, apply_gdn_adam_updates,
    eval_chain_nll, forward_full, nll_and_grads, rand_fill, shifted,
};

mod train_common;
use train_common::{ArgView, load_jsonl, verify_tbv};

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
  --fd-eps      <F>      Gradcheck central-difference step (default: 4e-3)
  -h, --help             Print this help"
    );
}

/// Capture the frozen-prefix output (h_in entering `first_layer`) and RoPE
/// tables per sample, yielding the `SeqCtx` set the tape forward consumes.
/// Returns the caches plus the total number of masked completion positions.
fn build_caches(
    model: &Qwen35Model,
    samples: &[train_common::Sample],
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

/// Gradcheck-mode GDN LoRA initialization: every array (A and B) filled with
/// small non-zero random noise, so both `grad_A` and the gate path are
/// non-vacuous for the finite-difference probe.
///
/// Extracted into a standalone, testable function (#792): this is the exact
/// constructor that had drifted to `gdn_dims.num_kh` for `b_b`/`b_a` while
/// `GdnLoraParams::zeros` used the correct `value_heads`. Now routes through
/// `GdnLoraParams::shaped`, the single shape source of truth.
fn gradcheck_gdn_loras(
    num_gdn_slots: usize,
    rank: usize,
    hidden: usize,
    gdn_dims: &GdnDims,
    seed: u64,
) -> Vec<GdnLoraParams> {
    // rand_fill takes `&mut u64`; two closures can't each hold that mutable
    // borrow as separate `shaped` arguments, so thread it through a `Cell`
    // (each closure captures `&rng_cell`, a shared reference, so both can
    // coexist as separate arguments).
    let rng_cell = std::cell::Cell::new(seed);
    (0..num_gdn_slots)
        .map(|_| {
            GdnLoraParams::shaped(
                rank,
                hidden,
                gdn_dims,
                |n| {
                    let mut r = rng_cell.get();
                    let v = rand_fill(&mut r, n, 0.05);
                    rng_cell.set(r);
                    v
                },
                |n| {
                    let mut r = rng_cell.get();
                    let v = rand_fill(&mut r, n, 0.05);
                    rng_cell.set(r);
                    v
                },
            )
            .expect("gdn lora gradcheck-mode shapes should not overflow")
        })
        .collect()
}

/// Training-mode GDN LoRA initialization: A ~ U(-init_amp, +init_amp), B
/// zero (delta=0 at init reproduces the base; grad_B != 0 so B moves first).
///
/// Extracted into a standalone, testable function (#792): see
/// [`gradcheck_gdn_loras`] for why this must route through
/// `GdnLoraParams::shaped` rather than re-deriving `b_b`/`b_a`'s length
/// inline (the same drift existed at this call site independently).
fn zero_b_gdn_loras(
    num_gdn_slots: usize,
    rank: usize,
    hidden: usize,
    gdn_dims: &GdnDims,
    seed: u64,
    init_amp: f32,
) -> Vec<GdnLoraParams> {
    let mut rng = seed;
    (0..num_gdn_slots)
        .map(|_| {
            GdnLoraParams::shaped(
                rank,
                hidden,
                gdn_dims,
                |n| rand_fill(&mut rng, n, init_amp),
                |n| vec![0.0; n],
            )
            .expect("gdn lora training-mode shapes should not overflow")
        })
        .collect()
}

/// Typed CLI config for `train_grad_full`. Field defaults are the documented
/// contract in `usage()` and issue #845's flag table — the snapshot tests
/// below pin them. `--fd-eps` is undocumented-but-live (issue #845): kept
/// exactly as before, now also documented in `usage()`.
#[derive(Debug)]
struct Config {
    model_dir: PathBuf,
    data_dir: PathBuf,
    first_layer: usize,
    steps: usize,
    lr: f32,
    rank: usize,
    alpha: f32,
    seq_len_cap: usize,
    max_train: usize,
    max_valid: usize,
    log_every: usize,
    gradcheck: bool,
    probe: usize,
    fd_eps: f32,
    save_path: Option<String>,
}

fn parse_config(argv: &ArgView) -> Result<Config, String> {
    let log_every: usize = argv
        .arg("--log-every")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    if log_every == 0 {
        return Err("--log-every must be >= 1".to_string());
    }
    Ok(Config {
        model_dir: argv
            .arg("--model-dir")
            .map(PathBuf::from)
            .unwrap_or_else(train_common::default_model_dir),
        data_dir: argv
            .arg("--data-dir")
            .map(PathBuf::from)
            .unwrap_or_else(train_common::default_data_dir),
        first_layer: argv
            .arg("--first-layer")
            .and_then(|s| s.parse().ok())
            .unwrap_or(19),
        steps: argv
            .arg("--steps")
            .and_then(|s| s.parse().ok())
            .unwrap_or(25),
        lr: argv
            .arg("--lr")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1e-3),
        rank: argv.arg("--rank").and_then(|s| s.parse().ok()).unwrap_or(8),
        alpha: argv
            .arg("--alpha")
            .and_then(|s| s.parse().ok())
            .unwrap_or(16.0),
        seq_len_cap: argv
            .arg("--seq-len")
            .and_then(|s| s.parse().ok())
            .unwrap_or(64),
        max_train: argv
            .arg("--max-train")
            .and_then(|s| s.parse().ok())
            .unwrap_or(3),
        max_valid: argv
            .arg("--max-valid")
            .and_then(|s| s.parse().ok())
            .unwrap_or(16),
        log_every,
        gradcheck: argv.flag("--gradcheck"),
        probe: argv
            .arg("--probe")
            .and_then(|s| s.parse().ok())
            .unwrap_or(6),
        // Central-difference step. On the real f32 model (hidden 1024, vocab
        // 248320, multi-layer GDN recurrence) the NLL carries ~1e-6 roundoff,
        // so too-small a step is roundoff-dominated. Optimal central-FD step
        // ≈ cbrt(roundoff) ≈ 4e-3.
        fd_eps: argv
            .arg("--fd-eps")
            .and_then(|s| s.parse().ok())
            .unwrap_or(4e-3),
        save_path: argv.arg("--save"),
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
        first_layer,
        steps,
        lr,
        rank,
        alpha,
        seq_len_cap,
        max_train,
        max_valid,
        log_every,
        gradcheck,
        probe,
        fd_eps,
        save_path,
    } = parse_config(&argv)?;

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
    // slot to every layer: GQA layers get a slot into `loras` (surface-A,
    // q_proj/v_proj), GDN layers get a slot into `gdn_loras` (surface-B, the
    // 5 GDN projections — ported from lattice PR #202). All weight slices are
    // borrowed from `model`.
    let mut layers: Vec<LayerW> = Vec::new();
    let mut slot_layers: Vec<usize> = Vec::new(); // global layer index per GQA slot
    let mut gdn_slot_layers: Vec<usize> = Vec::new(); // global layer index per GDN slot
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
            let slot = gdn_slot_layers.len();
            gdn_slot_layers.push(layer_idx);
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
                lora_slot: Some(slot),
            });
        } else {
            return Err(
                format!("layer {layer_idx} is neither a Full+Dense nor GDN+Dense layer").into(),
            );
        }
    }
    let num_slots = slot_layers.len();
    let num_gdn_slots = gdn_slot_layers.len();
    let kinds: String = layers
        .iter()
        .map(|l| match l.kind {
            MixerKind::Gqa => 'A',
            MixerKind::Gdn => 'D',
        })
        .collect();
    println!(
        "  materialised {} layers [{}]: {kinds}  ({} GQA LoRA slots at layers {:?}, \
         {num_gdn_slots} GDN LoRA slots at layers {:?})",
        layers.len(),
        (first_layer..=TOP_LAYER)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(","),
        num_slots,
        slot_layers,
        gdn_slot_layers,
    );
    if num_slots == 0 && num_gdn_slots == 0 {
        return Err("no GQA or GDN layers in range — nothing to train".into());
    }

    let (beta1, beta2, eps_adam) = (0.9f32, 0.999f32, 1e-8f32);
    let train_ctx = TrainCtx::try_new(
        TapeGeometry::new(&dims, &gdn_dims, &cfg),
        rank,
        alpha,
        SlotLayout::new(&slot_layers, &gdn_slot_layers),
        AdamConfig::new(lr, beta1, beta2, eps_adam),
    )?;

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
    let zero_gdn_loras: Vec<GdnLoraParams> = (0..num_gdn_slots)
        .map(|_| GdnLoraParams::zeros(rank, dims.hidden, &gdn_dims))
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
            &zero_gdn_loras,
            &head,
            &train_ctx,
        )?;
        let observation = verify_tbv(
            "train_grad_full assembled chain check (sample 0)",
            model_masked,
            chain_masked,
        )?;
        println!(
            "\n  TBV (sample 0): model={model_masked:.5}  chain={chain_masked:.5}  diff={:.2e}",
            observation.diff
        );
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
        let mut gdn_loras: Vec<GdnLoraParams> =
            gradcheck_gdn_loras(num_gdn_slots, rank, dims.hidden, &gdn_dims, rng);

        let fwd = forward_full(&caches[0], &layers, &loras, &gdn_loras, &head, &train_ctx)?;
        let (_, _, analytic, gdn_analytic) =
            nll_and_grads(&fwd, &layers, &loras, &head, &train_ctx)?;

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
                            &gdn_loras,
                            &head,
                            &train_ctx,
                        )?;
                        bump(&mut loras, save - e);
                        let lm = eval_chain_nll(
                            &caches[..1],
                            &layers,
                            &loras,
                            &gdn_loras,
                            &head,
                            &train_ctx,
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
        // ---- GDN LoRA slots (surface-B, ported from lattice PR #202) ----
        for slot in 0..num_gdn_slots {
            let layer_idx = gdn_slot_layers[slot];
            for (name, alen) in [
                ("a_qkv", gdn_analytic[slot].a_qkv.len()),
                ("b_qkv", gdn_analytic[slot].b_qkv.len()),
                ("a_z", gdn_analytic[slot].a_z.len()),
                ("b_z", gdn_analytic[slot].b_z.len()),
                ("a_b", gdn_analytic[slot].a_b.len()),
                ("b_b", gdn_analytic[slot].b_b.len()),
                ("a_a", gdn_analytic[slot].a_a.len()),
                ("b_a", gdn_analytic[slot].b_a.len()),
                ("a_out", gdn_analytic[slot].a_out.len()),
                ("b_out", gdn_analytic[slot].b_out.len()),
            ] {
                fn field<'a>(g: &'a GdnLoraParams, name: &str) -> &'a [f32] {
                    match name {
                        "a_qkv" => &g.a_qkv,
                        "b_qkv" => &g.b_qkv,
                        "a_z" => &g.a_z,
                        "b_z" => &g.b_z,
                        "a_b" => &g.a_b,
                        "b_b" => &g.b_b,
                        "a_a" => &g.a_a,
                        "b_a" => &g.b_a,
                        "a_out" => &g.a_out,
                        _ => &g.b_out,
                    }
                }
                fn field_mut<'a>(g: &'a mut GdnLoraParams, name: &str) -> &'a mut [f32] {
                    match name {
                        "a_qkv" => &mut g.a_qkv,
                        "b_qkv" => &mut g.b_qkv,
                        "a_z" => &mut g.a_z,
                        "b_z" => &mut g.b_z,
                        "a_b" => &mut g.a_b,
                        "b_b" => &mut g.b_b,
                        "a_a" => &mut g.a_a,
                        "b_a" => &mut g.b_a,
                        "a_out" => &mut g.a_out,
                        _ => &mut g.b_out,
                    }
                }
                let agrad = field(&gdn_analytic[slot], name);
                let mut idxs = top_k_indices(agrad, probe.min(alen));
                let seed = (100
                    + slot as u64 * 10
                    + match name {
                        "a_qkv" => 0,
                        "b_qkv" => 1,
                        "a_z" => 2,
                        "b_z" => 3,
                        "a_b" => 4,
                        "b_b" => 5,
                        "a_a" => 6,
                        "b_a" => 7,
                        "a_out" => 8,
                        _ => 9,
                    })
                    ^ 0xBEEF;
                for p in strided_probes(alen, probe.min(alen), seed) {
                    if !idxs.contains(&p) {
                        idxs.push(p);
                    }
                }
                let mut max_rel = 0.0f64;
                let mut sum_rel = 0.0f64;
                for &k in &idxs {
                    let a = field(&gdn_analytic[slot], name)[k];
                    let save = field(&gdn_loras[slot], name)[k];
                    let eps_set = [fd_eps * 0.25, fd_eps * 0.5, fd_eps, fd_eps * 2.0];
                    let mut best = f64::INFINITY;
                    for &e in &eps_set {
                        field_mut(&mut gdn_loras[slot], name)[k] = save + e;
                        let lp = eval_chain_nll(
                            &caches[..1],
                            &layers,
                            &loras,
                            &gdn_loras,
                            &head,
                            &train_ctx,
                        )?;
                        field_mut(&mut gdn_loras[slot], name)[k] = save - e;
                        let lm = eval_chain_nll(
                            &caches[..1],
                            &layers,
                            &loras,
                            &gdn_loras,
                            &head,
                            &train_ctx,
                        )?;
                        let fd = (lp - lm) / (2.0 * e);
                        let rel = (a - fd).abs() as f64 / (a.abs().max(fd.abs()).max(1e-6)) as f64;
                        best = best.min(rel);
                    }
                    field_mut(&mut gdn_loras[slot], name)[k] = save;
                    max_rel = max_rel.max(best);
                    sum_rel += best;
                }
                let mean_rel = sum_rel / idxs.len().max(1) as f64;
                worst = worst.max(max_rel);
                let ok = max_rel < 2e-2;
                all_pass &= ok;
                println!(
                    "  layer {layer_idx:2} gdn-slot {slot} {name:<5}: mean {mean_rel:.2e}  max {max_rel:.2e}  {}",
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
    // Reuses `rng`'s current stream position as the seed (matches the prior
    // inline behavior of drawing from the same generator as `loras` above).
    let mut gdn_loras: Vec<GdnLoraParams> =
        zero_b_gdn_loras(num_gdn_slots, rank, dims.hidden, &gdn_dims, rng, init_amp);

    let mut adam = AdamState::new();

    let eval_valid = |loras: &[LoraParams],
                      gdn_loras: &[GdnLoraParams]|
     -> Result<Option<f32>, Box<dyn std::error::Error>> {
        if valid_caches.is_empty() {
            return Ok(None);
        }
        Ok(Some(eval_chain_nll(
            &valid_caches,
            &layers,
            loras,
            gdn_loras,
            &head,
            &train_ctx,
        )?))
    };

    let base_nll = eval_chain_nll(&caches, &layers, &loras, &gdn_loras, &head, &train_ctx)?;
    let base_valid = eval_valid(&loras, &gdn_loras)?;
    match base_valid {
        Some(v) => println!("\n  step    0  train NLL: {base_nll:.4}  held-out NLL: {v:.4}"),
        None => println!("\n  step    0  train NLL: {base_nll:.4}"),
    }

    let tstep = Instant::now();
    for step in 1..=steps {
        let ctx = &caches[(step - 1) % caches.len()];
        let fwd = forward_full(ctx, &layers, &loras, &gdn_loras, &head, &train_ctx)?;
        let (_nll, _n, grads, gdn_grads) = nll_and_grads(&fwd, &layers, &loras, &head, &train_ctx)?;

        apply_adam_updates(&mut adam, &mut loras, &grads, &train_ctx);
        apply_gdn_adam_updates(&mut adam, &mut gdn_loras, &gdn_grads, &train_ctx);

        if step % log_every == 0 || step == steps {
            let mean_nll = eval_chain_nll(&caches, &layers, &loras, &gdn_loras, &head, &train_ctx)?;
            match eval_valid(&loras, &gdn_loras)? {
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

    let final_nll = eval_chain_nll(&caches, &layers, &loras, &gdn_loras, &head, &train_ctx)?;
    let secs = tstep.elapsed().as_secs_f64();
    match (base_valid, eval_valid(&loras, &gdn_loras)?) {
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
            let mut target_modules = vec!["q_proj".to_string(), "v_proj".to_string()];
            if num_gdn_slots > 0 {
                target_modules.extend([
                    "in_proj_qkv".to_string(),
                    "in_proj_z".to_string(),
                    "in_proj_b".to_string(),
                    "in_proj_a".to_string(),
                    "out_proj".to_string(),
                ]);
                // GDN LoRA (surface-B, ported from lattice PR #202). All five
                // module names are recognised by `LoraAdapter::validate_against`
                // (crates/tune/src/lora/mod.rs, inference-hook feature).
                for (slot, &li) in gdn_slot_layers.iter().enumerate() {
                    let g = &gdn_loras[slot];
                    adapter_layers.insert(
                        (li, "in_proj_qkv".to_string()),
                        LoraLayer {
                            a: g.a_qkv.clone(),
                            b: g.b_qkv.clone(),
                            d_in: dims.hidden,
                            d_out: gdn_dims.qkv_dim,
                            rank,
                        },
                    );
                    adapter_layers.insert(
                        (li, "in_proj_z".to_string()),
                        LoraLayer {
                            a: g.a_z.clone(),
                            b: g.b_z.clone(),
                            d_in: dims.hidden,
                            d_out: gdn_dims.output_dim,
                            rank,
                        },
                    );
                    adapter_layers.insert(
                        (li, "in_proj_b".to_string()),
                        LoraLayer {
                            a: g.a_b.clone(),
                            b: g.b_b.clone(),
                            d_in: dims.hidden,
                            // beta is projected per VALUE head (matches the
                            // shipping gdn_fused forward and the f16 weight
                            // loader), not per key head (#792).
                            d_out: gdn_dims.value_heads,
                            rank,
                        },
                    );
                    adapter_layers.insert(
                        (li, "in_proj_a".to_string()),
                        LoraLayer {
                            a: g.a_a.clone(),
                            b: g.b_a.clone(),
                            d_in: dims.hidden,
                            // alpha is likewise projected per VALUE head.
                            d_out: gdn_dims.value_heads,
                            rank,
                        },
                    );
                    adapter_layers.insert(
                        (li, "out_proj".to_string()),
                        LoraLayer {
                            a: g.a_out.clone(),
                            b: g.b_out.clone(),
                            d_in: gdn_dims.output_dim,
                            d_out: dims.hidden,
                            rank,
                        },
                    );
                }
            }
            let config = LoraConfig {
                rank,
                alpha,
                target_modules,
            };
            let adapter = LoraAdapter::new(config, adapter_layers)
                .map_err(|e| format!("construct adapter: {e}"))?;
            adapter
                .save_safetensors(std::path::Path::new(path), None)
                .map_err(|e| format!("save adapter: {e}"))?;
            println!(
                "saved adapter ({num_slots} GQA slots, {num_gdn_slots} GDN slots, rank {rank}) → {path}"
            );
        }
        #[cfg(not(feature = "safetensors"))]
        {
            let _ = path;
            return Err("--save requires the safetensors feature (--features safetensors)".into());
        }
    }

    Ok(())
}

#[cfg(test)]
mod cli_contract_tests {
    use super::*;

    fn args(extra: &[&str]) -> Vec<String> {
        let mut v = vec!["train_grad_full".to_string()];
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    }

    #[test]
    fn defaults_match_documented_table() {
        let a = args(&[]);
        let cfg = parse_config(&ArgView::new(&a)).unwrap();
        assert_eq!(cfg.data_dir, PathBuf::from("data/lora-train"));
        assert_eq!(cfg.first_layer, 19);
        assert_eq!(cfg.steps, 25);
        assert_eq!(cfg.lr, 1e-3);
        assert_eq!(cfg.rank, 8);
        assert_eq!(cfg.alpha, 16.0);
        assert_eq!(cfg.seq_len_cap, 64);
        assert_eq!(cfg.max_train, 3);
        assert_eq!(cfg.max_valid, 16);
        assert_eq!(cfg.log_every, 5);
        assert!(!cfg.gradcheck);
        assert_eq!(cfg.probe, 6);
        assert_eq!(cfg.fd_eps, 4e-3);
        assert_eq!(cfg.save_path, None);
        assert_eq!(cfg.model_dir, train_common::default_model_dir());
    }

    #[test]
    fn explicit_flags_override_defaults() {
        let a = args(&[
            "--model-dir",
            "/tmp/m",
            "--data-dir",
            "/tmp/d",
            "--first-layer",
            "20",
            "--steps",
            "9",
            "--lr",
            "5e-4",
            "--rank",
            "16",
            "--alpha",
            "32",
            "--seq-len",
            "48",
            "--max-train",
            "1",
            "--max-valid",
            "0",
            "--log-every",
            "3",
            "--gradcheck",
            "--probe",
            "2",
            "--fd-eps",
            "1e-3",
            "--save",
            "/tmp/out.safetensors",
        ]);
        let cfg = parse_config(&ArgView::new(&a)).unwrap();
        assert_eq!(cfg.model_dir, PathBuf::from("/tmp/m"));
        assert_eq!(cfg.data_dir, PathBuf::from("/tmp/d"));
        assert_eq!(cfg.first_layer, 20);
        assert_eq!(cfg.steps, 9);
        assert_eq!(cfg.lr, 5e-4);
        assert_eq!(cfg.rank, 16);
        assert_eq!(cfg.alpha, 32.0);
        assert_eq!(cfg.seq_len_cap, 48);
        assert_eq!(cfg.max_train, 1);
        assert_eq!(cfg.max_valid, 0);
        assert_eq!(cfg.log_every, 3);
        assert!(cfg.gradcheck);
        assert_eq!(cfg.probe, 2);
        assert_eq!(cfg.fd_eps, 1e-3);
        assert_eq!(cfg.save_path, Some("/tmp/out.safetensors".to_string()));
    }

    #[test]
    fn log_every_zero_is_rejected() {
        let a = args(&["--log-every", "0"]);
        let err = parse_config(&ArgView::new(&a)).unwrap_err();
        assert!(err.contains("--log-every must be >= 1"));
    }

    #[test]
    fn help_flags_detected_via_arg_view() {
        let a = args(&["-h"]);
        assert!(ArgView::new(&a).flag("-h"));
        let a = args(&["--help"]);
        assert!(ArgView::new(&a).flag("--help"));
    }
}

#[cfg(test)]
mod gdn_lora_ctor_tests {
    use super::*;

    /// Asymmetric fixture (num_kh=2, value_heads=4) — the only shape that can
    /// distinguish "sized by key heads" from "sized by value heads"; a
    /// symmetric config (e.g. num_kh == value_heads) would pass either way.
    /// #792: `gradcheck_gdn_loras`/`zero_b_gdn_loras` (this binary's two GDN
    /// LoRA initializers) had independently re-derived `b_b`/`b_a` as
    /// `num_kh * rank` instead of `value_heads * rank`, breaking on exactly
    /// this shape class.
    fn asymmetric_gdn_dims() -> GdnDims {
        let key_dim = 8;
        let value_dim = 8;
        let num_kh = 2;
        let value_heads = 4;
        GdnDims {
            num_kh,
            value_heads,
            key_dim,
            value_dim,
            qkv_dim: 2 * key_dim * num_kh + value_heads * value_dim,
            output_dim: value_heads * value_dim,
            kernel_size: 3,
            scale: 1.0 / (key_dim as f32).sqrt(),
        }
    }

    #[test]
    fn gradcheck_gdn_loras_sizes_b_b_and_b_a_by_value_heads() {
        let gd = asymmetric_gdn_dims();
        let hidden = 16;
        let rank = 3;
        let loras = gradcheck_gdn_loras(2, rank, hidden, &gd, 0x1234_5678);
        assert_eq!(loras.len(), 2);
        for lora in &loras {
            assert_eq!(
                lora.b_b.len(),
                gd.value_heads * rank,
                "b_b must be sized by value_heads ({}), not num_kh ({})",
                gd.value_heads,
                gd.num_kh
            );
            assert_eq!(
                lora.b_a.len(),
                gd.value_heads * rank,
                "b_a must be sized by value_heads ({}), not num_kh ({})",
                gd.value_heads,
                gd.num_kh
            );
            // Non-zero: gradcheck mode fills both A and B with random noise.
            assert!(lora.b_b.iter().any(|&v| v != 0.0));
            assert!(lora.b_a.iter().any(|&v| v != 0.0));
        }
    }

    #[test]
    fn zero_b_gdn_loras_sizes_b_b_and_b_a_by_value_heads() {
        let gd = asymmetric_gdn_dims();
        let hidden = 16;
        let rank = 3;
        let init_amp = 1.0 / (hidden as f32).sqrt();
        let loras = zero_b_gdn_loras(2, rank, hidden, &gd, 0xFEED_FACE, init_amp);
        assert_eq!(loras.len(), 2);
        for lora in &loras {
            assert_eq!(
                lora.b_b.len(),
                gd.value_heads * rank,
                "b_b must be sized by value_heads ({}), not num_kh ({})",
                gd.value_heads,
                gd.num_kh
            );
            assert_eq!(
                lora.b_a.len(),
                gd.value_heads * rank,
                "b_a must be sized by value_heads ({}), not num_kh ({})",
                gd.value_heads,
                gd.num_kh
            );
            // Training-mode: B is zero at init (delta=0 reproduces the base).
            assert!(lora.b_b.iter().all(|&v| v == 0.0));
            assert!(lora.b_a.iter().all(|&v| v == 0.0));
            // A is non-zero random.
            assert!(lora.a_b.iter().any(|&v| v != 0.0));
            assert!(lora.a_a.iter().any(|&v| v != 0.0));
        }
    }
}
