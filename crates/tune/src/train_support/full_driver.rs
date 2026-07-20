//! Shared exact multi-layer LoRA training driver.

use std::path::PathBuf;
use std::time::Instant;

use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::tokenizer::Tokenizer;

use crate::lora::AdamState;
use crate::lora::train_core::{
    AdamConfig, Dims, GdnDims, GdnLoraParams, Head, LayerW, LoraParams, MixerKind, SeqCtx,
    SlotLayout, TOP_LAYER, TapeGeometry, TrainCtx, apply_adam_updates, apply_gdn_adam_updates,
    eval_chain_nll, forward_full, nll_and_grads, rand_fill, shifted,
};

use super::{Sample, load_jsonl, verify_tbv};

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

/// Initialize every GDN LoRA factor with nonzero noise for finite-difference checks.
/// See [`docs/design.md`](../../../docs/design.md#train_grad_full-details) for shape and non-vacuity invariants.
fn gradcheck_gdn_loras(
    num_gdn_slots: usize,
    rank: usize,
    hidden: usize,
    gdn_dims: &GdnDims,
    seed: u64,
) -> Vec<GdnLoraParams> {
    // `Cell` lets both `shaped` initializers advance one deterministic RNG state.
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

/// Initialize GDN LoRA with random A factors and zero B factors for training.
/// See [`docs/design.md`](../../../docs/design.md#train_grad_full-details) for shape and initialization invariants.
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

/// Fully resolved options for the shared multi-layer training driver.
#[derive(Debug)]
pub struct FullDriverConfig {
    pub model_dir: PathBuf,
    pub data_dir: PathBuf,
    pub first_layer: usize,
    pub steps: usize,
    pub lr: f32,
    pub rank: usize,
    pub alpha: f32,
    pub seq_len_cap: usize,
    pub max_train: usize,
    pub max_valid: usize,
    pub log_every: usize,
    pub gradcheck: bool,
    /// Whether gradcheck supplements top-gradient entries with deterministic
    /// strided probes.
    pub gradcheck_strided_probes: bool,
    pub probe: usize,
    pub fd_eps: f32,
    pub save_path: Option<String>,
    /// Fixed A-factor amplitude for compatibility callers; `None` uses the
    /// full trainer's `1 / sqrt(hidden)` initialization.
    pub a_init_amp: Option<f32>,
}

/// Numeric evidence returned by the shared driver.
pub struct FullDriverOutcome {
    /// Worst central-difference relative error in gradcheck mode.
    pub gradcheck_max_rel: Option<f64>,
}

/// Upper bound on LoRA rank accepted by [`run`]. Limits caller-controlled
/// buffer-size products (mirrors [`crate::lora::train`]'s bound).
const MAX_RANK: usize = 512;

/// Upper bound on training steps accepted by [`run`]. Limits unbounded
/// caller-controlled work (mirrors [`crate::lora::train`]'s bound).
const MAX_STEPS: usize = 100_000;

/// Upper bound on `--max-train` / `--max-valid` sample counts accepted by
/// [`run`]. Bounds tokenization and per-sample activation-cache work before
/// any of it runs; a real training or validation set this large is already
/// well beyond what a single CPU driver invocation is meant to process.
const MAX_SAMPLES: usize = 100_000;

/// Upper bound on `--seq-len-cap` accepted by [`run`]. Mirrors
/// [`crate::lora::train`]'s `MAX_TRAIN_SEQ_LEN`, the same per-pair sequence
/// ceiling used by the public micro-LoRA trainer.
const MAX_SEQ_LEN_CAP: usize = 8_192;

/// Upper bound on captured-logits memory for one cache set (training or
/// validation), matching the 2 GiB `f32` cap already enforced for the
/// training cache.
const MAX_LOGITS_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Upper bound on the frozen-prefix attention cache ([`build_caches`]'s
/// `h_in` + RoPE `cos`/`sin` buffers) for one sample set, in bytes. Bounds
/// the PRODUCT of raw token count and per-token buffer size — `MAX_SAMPLES`
/// and `MAX_SEQ_LEN_CAP` bound sample count and per-sample length
/// independently, and neither bounds their product (100_000 samples ×
/// 8_192 tokens = 819.2M raw tokens). Same 2 GiB order of magnitude as
/// [`MAX_LOGITS_BYTES`], the existing per-cache-set memory precedent for
/// this driver.
const MAX_CACHE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Upper bound on the per-sample training-tape allocation `forward_full`
/// builds for one [`SeqCtx`]. Same 2 GiB order of magnitude as
/// [`MAX_CACHE_BYTES`] and [`MAX_LOGITS_BYTES`].
const MAX_TAPE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Per-token `f32`-element count of every buffer a GQA layer's
/// `forward_full` pass retains, EXCLUDING `softmax_probs` — that term is not
/// linear in `seq_len` and is accounted separately by the caller. Mirrors
/// `lattice_inference::backward::attention_gqa::AttnCache` (`x_input`;
/// `q_raw`/`q_pre_rope`/`q_h`/`context`/`gate_z`, each `q_dim`;
/// `k_raw`/`k_pre_rope`/`v`, each `kv_dim`; `h_q`/`h_v`, each `rank`;
/// `cos_vals`/`sin_vals`, each `rope_dim / 2`) plus the shared `LayerFwd`
/// bookkeeping every materialized layer carries (`h_layer_in`/`h_mid`, each
/// `hidden`; `gate_pre`/`up_pre`, each `inter`).
fn gqa_layer_linear_elems_per_token(dims: &Dims, rank: usize) -> u64 {
    let q_dim = dims.q_dim as u64;
    let kv_dim = dims.kv_dim as u64;
    let hidden = dims.hidden as u64;
    let rope_half = (dims.rope_dim / 2) as u64;
    let rank = rank as u64;
    let attn_cache = hidden + 5 * q_dim + 3 * kv_dim + 2 * rank + 2 * rope_half;
    let layer_fwd = 2 * hidden + 2 * dims.inter as u64;
    attn_cache + layer_fwd
}

/// Per-token `f32`-element count of every buffer a GDN layer's
/// `forward_full` pass retains — all linear in `seq_len` (the recurrent
/// state `s_after` is one `key_dim * value_dim` block per token, not one per
/// token-pair). Mirrors `lattice_inference::attention::gdn_backward::GdnSaved`
/// (`inputs`: `hidden`; `qkv_proj`/`conv_out`: `qkv_dim` each;
/// `conv_buffers`: `qkv_dim * (kernel_size - 1)`; `z_proj`/`gated_buf`:
/// `output_dim` each; `beta_raw`/`alpha_proj`/`beta`/`g`/`q_norm`/`k_norm`/
/// `q_eps_norm`/`k_eps_norm`/`rms_vals`: `value_heads` each; `q_hat`/`k_hat`:
/// `value_heads * key_dim` each; `v`/`kv_mem`/`o_heads`/`silu_z`:
/// `value_heads * value_dim` each; `s_after`:
/// `value_heads * key_dim * value_dim`; `h_qkv`/`h_z`/`h_b`/`h_a`/`h_out`:
/// `rank` each) plus the shared `LayerFwd` bookkeeping.
fn gdn_layer_linear_elems_per_token(dims: &Dims, gdn_dims: &GdnDims, rank: usize) -> u64 {
    let vh = gdn_dims.value_heads as u64;
    let key_dim = gdn_dims.key_dim as u64;
    let value_dim = gdn_dims.value_dim as u64;
    let qkv_dim = gdn_dims.qkv_dim as u64;
    let output_dim = gdn_dims.output_dim as u64;
    let kernel = gdn_dims.kernel_size as u64;
    let hidden = dims.hidden as u64;
    let rank = rank as u64;
    let gdn_saved = hidden
        + 2 * qkv_dim
        + qkv_dim * kernel.saturating_sub(1)
        + 2 * output_dim
        + 9 * vh
        + 2 * vh * key_dim
        + 4 * vh * value_dim
        + vh * key_dim * value_dim
        + 5 * rank;
    let layer_fwd = 2 * hidden + 2 * dims.inter as u64;
    gdn_saved + layer_fwd
}

/// Reject a `seq_len_cap` whose worst-case per-sample `forward_full` tape —
/// summed across every materialized GQA and GDN layer — would exceed
/// [`MAX_TAPE_BYTES`]. `forward_full` builds and tears down one sample's tape
/// per training step (`run`'s `caches[(step - 1) % caches.len()]`), so the
/// peak allocation is bounded above by the single largest sample, which
/// `load_jsonl` never lets exceed `seq_len_cap`. The dominant term is the GQA
/// `softmax_probs` cache, which retains one causal row per query position
/// per head — `num_q_heads * seq_len * (seq_len + 1) / 2` `f32`s per GQA
/// layer, quadratic in `seq_len` — so a `seq_len_cap` that clears
/// [`check_cache_budget`]'s linear bound can still exhaust memory once
/// `forward_full` actually runs. Validates the derived total against a
/// ceiling (rather than lowering [`MAX_SEQ_LEN_CAP`] itself) so the bound
/// stays correct as head/layer counts change across models.
fn check_tape_budget(
    seq_len_cap: usize,
    dims: &Dims,
    gdn_dims: &GdnDims,
    rank: usize,
    num_gqa_layers: usize,
    num_gdn_layers: usize,
) -> Result<(), String> {
    let seq = seq_len_cap as u64;
    let quadratic_elems = dims.num_q_heads as u64 * seq * (seq + 1) / 2 * num_gqa_layers as u64;
    let linear_elems = num_gqa_layers as u64 * gqa_layer_linear_elems_per_token(dims, rank) * seq
        + num_gdn_layers as u64 * gdn_layer_linear_elems_per_token(dims, gdn_dims, rank) * seq;
    let total_bytes = quadratic_elems
        .saturating_add(linear_elems)
        .saturating_mul(4);
    if total_bytes > MAX_TAPE_BYTES {
        return Err(format!(
            "--seq-len-cap {seq_len_cap} across {num_gqa_layers} GQA + {num_gdn_layers} GDN \
             materialized layers would allocate a ~{} MiB training tape per sample (dominated \
             by the GQA softmax_probs cache, quadratic in seq-len), exceeds {} MiB cap - reduce \
             --seq-len-cap or --first-layer",
            total_bytes / (1024 * 1024),
            MAX_TAPE_BYTES / (1024 * 1024),
        ));
    }
    Ok(())
}

/// Reject a sample set whose projected [`build_caches`] allocation
/// (`sum(seq_len) * (hidden + rope_dim) * 4` bytes for `h_in` + `cos` +
/// `sin`) would exceed [`MAX_CACHE_BYTES`]. The caller must run this BEFORE
/// [`build_caches`] — the check exists to prevent the allocation, not to
/// report on it after the fact.
fn check_cache_budget(
    samples: &[Sample],
    hidden: usize,
    rope_dim: usize,
    label: &str,
) -> Result<(), String> {
    let per_token_bytes = (hidden + rope_dim) as u64 * 4;
    let total_tokens: u64 = samples.iter().map(|s| s.tokens.len() as u64).sum();
    let projected_bytes = total_tokens.saturating_mul(per_token_bytes);
    if projected_bytes > MAX_CACHE_BYTES {
        return Err(format!(
            "{label} frozen-prefix cache would require {} MiB ({total_tokens} raw tokens \
             across {} samples * (hidden {hidden} + rope_dim {rope_dim}) * 4B), \
             exceeds {} MiB cap - reduce --seq-len-cap or --max-{label}",
            projected_bytes / (1024 * 1024),
            samples.len(),
            MAX_CACHE_BYTES / (1024 * 1024),
        ));
    }
    Ok(())
}

/// Validate the loaded model's depth against the driver's fixed `TOP_LAYER`
/// range before any layer access. [`Qwen35Model::gqa_layer_weights`] and
/// [`Qwen35Model::gdn_layer_weights`] index layer storage directly by
/// position and panic rather than returning `None` when the index is out of
/// bounds, so a model shallower than `TOP_LAYER + 1` must be rejected here.
fn validate_loaded_depth(loaded_layer_count: usize) -> Result<(), String> {
    if TOP_LAYER >= loaded_layer_count {
        return Err(format!(
            "model has {loaded_layer_count} loaded layers; configured top layer \
             {TOP_LAYER} is out of range"
        ));
    }
    Ok(())
}

/// Run the shared full-depth training implementation.
pub fn run(config: FullDriverConfig) -> Result<FullDriverOutcome, Box<dyn std::error::Error>> {
    let FullDriverConfig {
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
        gradcheck_strided_probes,
        probe,
        fd_eps,
        save_path,
        a_init_amp,
    } = config;

    if first_layer > TOP_LAYER {
        return Err(format!("--first-layer {first_layer} must be <= {TOP_LAYER}").into());
    }
    if rank == 0 || rank > MAX_RANK {
        return Err(format!("--rank {rank} must be in 1..={MAX_RANK}").into());
    }
    if steps > MAX_STEPS {
        return Err(format!("--steps {steps} must be <= {MAX_STEPS}").into());
    }
    if log_every == 0 {
        return Err("--log-every must be > 0".into());
    }
    if max_train > MAX_SAMPLES {
        return Err(format!("--max-train {max_train} exceeds maximum {MAX_SAMPLES}").into());
    }
    if max_valid > MAX_SAMPLES {
        return Err(format!("--max-valid {max_valid} exceeds maximum {MAX_SAMPLES}").into());
    }
    if seq_len_cap == 0 || seq_len_cap > MAX_SEQ_LEN_CAP {
        return Err(format!("--seq-len-cap {seq_len_cap} must be in 1..={MAX_SEQ_LEN_CAP}").into());
    }
    if let Some(amplitude) = a_init_amp
        && (!amplitude.is_finite() || amplitude < 0.0)
    {
        return Err(
            format!("A initialization amplitude must be finite and >= 0, got {amplitude}").into(),
        );
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

    validate_loaded_depth(model.loaded_layer_count())
        .map_err(|e| format!("model at {}: {e}", model_dir.display()))?;

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

    // Assign GQA and GDN LoRA slots across the materialized range — see docs/design.md.
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
    check_tape_budget(
        seq_len_cap,
        &dims,
        &gdn_dims,
        rank,
        num_slots,
        num_gdn_slots,
    )?;

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

    check_cache_budget(&train_samples, dims.hidden, dims.rope_dim, "train")?;

    // Capture the frozen prefix output (h_in entering first_layer) per sample.
    println!("\nBuilding frozen-prefix cache (layers 0..{first_layer})...");
    let tcache = Instant::now();
    let (caches, total_positions) = build_caches(&model, &train_samples, first_layer)?;
    let logits_bytes = total_positions * dims.vocab * 4;
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

    // Hold out valid.jsonl from training; compare its NLL to the training NLL.
    let valid_caches: Vec<SeqCtx> = if max_valid > 0 {
        match load_jsonl(
            &data_dir.join("valid.jsonl"),
            &tokenizer as &dyn Tokenizer,
            seq_len_cap,
            max_valid,
        ) {
            Ok(vs) if !vs.is_empty() => {
                check_cache_budget(&vs, dims.hidden, dims.rope_dim, "valid")?;
                let (vc, vpos) = build_caches(&model, &vs, first_layer)?;
                let valid_logits_bytes = vpos * dims.vocab * 4;
                if valid_logits_bytes > MAX_LOGITS_BYTES {
                    return Err(format!(
                        "held-out logits buffer would require {} MiB ({vpos} positions × {} \
                         vocab × 4B), exceeds 2 GiB cap — reduce --seq-len or --max-valid",
                        valid_logits_bytes / (1024 * 1024),
                        dims.vocab,
                    )
                    .into());
                }
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

    // Fail closed unless the zero-adapter chain matches the real model NLL.
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
                if gradcheck_strided_probes {
                    for p in strided_probes(alen, probe.min(alen), seed) {
                        if !idxs.contains(&p) {
                            idxs.push(p);
                        }
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
                    // Keep the best epsilon per entry to resist deep-f32 roundoff — see docs/design.md.
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
                if gradcheck_strided_probes {
                    for p in strided_probes(alen, probe.min(alen), seed) {
                        if !idxs.contains(&p) {
                            idxs.push(p);
                        }
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
            return Err(format!(
                "analytic gradient diverges from finite-difference (worst relative error {worst:e})"
            )
            .into());
        }
        return Ok(FullDriverOutcome {
            gradcheck_max_rel: Some(worst),
        });
    }

    // ---- Training mode ----
    // Zero B preserves the base; the full CLI defaults to mlx_lm-compatible
    // scaling while compatibility callers can retain their historical scale.
    let init_amp = match a_init_amp {
        Some(amplitude) => amplitude,
        None => 1.0 / (dims.hidden as f32).sqrt(),
    };
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
            use crate::lora::{LoraAdapter, LoraConfig, LoraLayer};
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

    Ok(FullDriverOutcome {
        gradcheck_max_rel: None,
    })
}

#[cfg(test)]
mod run_bounds_tests {
    use super::*;

    fn base_config() -> FullDriverConfig {
        FullDriverConfig {
            model_dir: PathBuf::from("/nonexistent/model-dir"),
            data_dir: PathBuf::from("/nonexistent/data-dir"),
            first_layer: 19,
            steps: 10,
            lr: 1e-3,
            rank: 4,
            alpha: 8.0,
            seq_len_cap: 64,
            max_train: 3,
            max_valid: 0,
            log_every: 5,
            gradcheck: false,
            gradcheck_strided_probes: false,
            probe: 0,
            fd_eps: 1e-3,
            save_path: None,
            a_init_amp: None,
        }
    }

    /// `FullDriverOutcome` has no `Debug` impl, so `Result::unwrap_err` isn't
    /// available — extract the error message by hand.
    fn expect_err(result: Result<FullDriverOutcome, Box<dyn std::error::Error>>) -> String {
        match result {
            Ok(_) => panic!("expected run() to reject this config"),
            Err(e) => e.to_string(),
        }
    }

    /// Without the rank cap, an oversized rank reaches
    /// model loading and allocation instead of being rejected up front.
    #[test]
    fn run_rejects_rank_over_max() {
        let mut cfg = base_config();
        cfg.rank = MAX_RANK + 1;
        let err = expect_err(run(cfg));
        assert!(
            err.contains("--rank") && err.contains(&MAX_RANK.to_string()),
            "expected rank-over-cap rejection; got: {err}"
        );
    }

    /// Without the rank>0 guard, rank=0 reaches the tape
    /// and divides by zero deriving `scale = alpha / rank`.
    #[test]
    fn run_rejects_zero_rank() {
        let mut cfg = base_config();
        cfg.rank = 0;
        let err = expect_err(run(cfg));
        assert!(
            err.contains("--rank"),
            "expected zero-rank rejection; got: {err}"
        );
    }

    /// Without the step cap, `steps = usize::MAX` would
    /// monopolize a worker indefinitely instead of being rejected up front.
    #[test]
    fn run_rejects_steps_over_max() {
        let mut cfg = base_config();
        cfg.steps = MAX_STEPS + 1;
        let err = expect_err(run(cfg));
        assert!(
            err.contains("--steps") && err.contains(&MAX_STEPS.to_string()),
            "expected steps-over-cap rejection; got: {err}"
        );
    }

    /// Without `validate_loaded_depth`, a model shallower
    /// than `TOP_LAYER + 1` reaches `first_layer..=TOP_LAYER` layer access and
    /// panics indexing `weights.layers[layer_idx]` out of bounds instead of
    /// returning an error.
    #[test]
    fn validate_loaded_depth_rejects_shallow_model() {
        let err = validate_loaded_depth(TOP_LAYER).unwrap_err();
        assert!(
            err.contains("loaded layers") && err.contains("out of range"),
            "expected shallow-model rejection; got: {err}"
        );
    }

    #[test]
    fn validate_loaded_depth_accepts_sufficient_model() {
        validate_loaded_depth(TOP_LAYER + 1).unwrap();
        validate_loaded_depth(40).unwrap();
    }

    /// Without the `log_every == 0` guard, a caller
    /// passing `--log-every 0` with `steps > 0` reaches `step % log_every`
    /// in the training loop and panics on divide-by-zero.
    #[test]
    fn run_rejects_zero_log_every() {
        let mut cfg = base_config();
        cfg.log_every = 0;
        let err = expect_err(run(cfg));
        assert!(
            err.contains("--log-every"),
            "expected log-every rejection; got: {err}"
        );
    }

    /// Without the `--max-train` cap, an oversized value
    /// reaches JSONL tokenization and per-sample activation-cache work
    /// instead of being rejected up front.
    #[test]
    fn run_rejects_max_train_over_max() {
        let mut cfg = base_config();
        cfg.max_train = MAX_SAMPLES + 1;
        let err = expect_err(run(cfg));
        assert!(
            err.contains("--max-train") && err.contains(&MAX_SAMPLES.to_string()),
            "expected max-train-over-cap rejection; got: {err}"
        );
    }

    /// Without the `--max-valid` cap, an oversized value
    /// reaches JSONL tokenization and per-sample activation-cache work
    /// instead of being rejected up front.
    #[test]
    fn run_rejects_max_valid_over_max() {
        let mut cfg = base_config();
        cfg.max_valid = MAX_SAMPLES + 1;
        let err = expect_err(run(cfg));
        assert!(
            err.contains("--max-valid") && err.contains(&MAX_SAMPLES.to_string()),
            "expected max-valid-over-cap rejection; got: {err}"
        );
    }

    /// Without the `--seq-len-cap` cap, an oversized
    /// value reaches tokenization and per-sample activation-cache allocation
    /// (`seq_len * hidden` per sample) unbounded.
    #[test]
    fn run_rejects_seq_len_cap_over_max() {
        let mut cfg = base_config();
        cfg.seq_len_cap = MAX_SEQ_LEN_CAP + 1;
        let err = expect_err(run(cfg));
        assert!(
            err.contains("--seq-len-cap") && err.contains(&MAX_SEQ_LEN_CAP.to_string()),
            "expected seq-len-cap-over-cap rejection; got: {err}"
        );
    }

    #[test]
    fn run_rejects_zero_seq_len_cap() {
        let mut cfg = base_config();
        cfg.seq_len_cap = 0;
        let err = expect_err(run(cfg));
        assert!(
            err.contains("--seq-len-cap"),
            "expected zero-seq-len-cap rejection; got: {err}"
        );
    }

    /// `MAX_SAMPLES` and `MAX_SEQ_LEN_CAP` bound sample
    /// count and per-sample length independently; neither bounds their
    /// product. This sample set passes both per-dimension caps individually
    /// (far under `MAX_SAMPLES` samples, far under `MAX_SEQ_LEN_CAP` tokens
    /// each) but its aggregate `build_caches` allocation exceeds
    /// `MAX_CACHE_BYTES`. Without `check_cache_budget`, nothing in `run()`
    /// would have rejected this combination before `build_caches` started
    /// allocating.
    #[test]
    fn check_cache_budget_rejects_oversized_aggregate() {
        let hidden = 4096;
        let rope_dim = 128;
        let samples: Vec<Sample> = (0..2_000)
            .map(|_| Sample {
                tokens: vec![0u32; 100],
                completion_start: 0,
            })
            .collect();
        assert!(
            samples.len() < MAX_SAMPLES,
            "test setup must pass the sample-count cap"
        );
        assert!(
            samples.iter().all(|s| s.tokens.len() < MAX_SEQ_LEN_CAP),
            "test setup must pass the per-sample length cap"
        );
        let total_tokens: u64 = samples.iter().map(|s| s.tokens.len() as u64).sum();
        assert!(
            total_tokens * ((hidden + rope_dim) as u64) * 4 > MAX_CACHE_BYTES,
            "test setup must actually exceed the aggregate cache budget"
        );

        let err = check_cache_budget(&samples, hidden, rope_dim, "train").unwrap_err();
        assert!(
            err.contains("frozen-prefix cache") && err.contains("--max-train"),
            "expected aggregate cache-budget rejection; got: {err}"
        );
    }

    #[test]
    fn check_cache_budget_accepts_small_aggregate() {
        let samples: Vec<Sample> = (0..4)
            .map(|_| Sample {
                tokens: vec![0u32; 32],
                completion_start: 0,
            })
            .collect();
        check_cache_budget(&samples, 4096, 128, "train").unwrap();
    }

    fn tape_test_dims() -> Dims {
        Dims {
            hidden: 2048,
            vocab: 32_000,
            num_q_heads: 8,
            num_kv_heads: 2,
            head_dim: 128,
            rope_dim: 128,
            inter: 4096,
            q_dim: 1024,
            kv_dim: 256,
            eps: 1e-6,
        }
    }

    /// Smaller than [`tape_test_dims`] so its own linear tape term stays well
    /// under [`MAX_TAPE_BYTES`] at `MAX_SEQ_LEN_CAP` — isolating the
    /// quadratic `softmax_probs` term as the sole reason the full budget
    /// check rejects it.
    fn quadratic_dominant_dims() -> Dims {
        Dims {
            hidden: 1024,
            vocab: 32_000,
            num_q_heads: 8,
            num_kv_heads: 1,
            head_dim: 64,
            rope_dim: 64,
            inter: 2048,
            q_dim: 512,
            kv_dim: 128,
            eps: 1e-6,
        }
    }

    fn tape_test_gdn_dims() -> GdnDims {
        GdnDims {
            num_kh: 2,
            value_heads: 8,
            key_dim: 64,
            value_dim: 64,
            qkv_dim: 2 * 64 * 2 + 8 * 64,
            output_dim: 8 * 64,
            kernel_size: 4,
            scale: 1.0 / (64f32).sqrt(),
        }
    }

    /// A single sample at `seq_len_cap` clears [`check_cache_budget`] (linear
    /// in seq-len — only `h_in`/`cos`/`sin`), yet its `forward_full` tape
    /// across the same materialized GQA layers is quadratic in seq-len via
    /// `softmax_probs`. Without `check_tape_budget`, this combination would
    /// pass every existing guard and exhaust memory once training starts.
    #[test]
    fn check_tape_budget_rejects_quadratic_overflow_that_cache_budget_misses() {
        let dims = quadratic_dominant_dims();
        let gdn_dims = tape_test_gdn_dims();
        let seq_len_cap = MAX_SEQ_LEN_CAP; // 8_192
        let rank = 8;
        let num_gqa_layers = 2;

        let single_sample = vec![Sample {
            tokens: vec![0u32; seq_len_cap],
            completion_start: 0,
        }];
        check_cache_budget(&single_sample, dims.hidden, dims.rope_dim, "train")
            .expect("test setup: a single max-length sample must clear the linear cache budget");

        let seq = seq_len_cap as u64;
        let linear_only_bytes =
            num_gqa_layers as u64 * gqa_layer_linear_elems_per_token(&dims, rank) * seq * 4;
        assert!(
            linear_only_bytes <= MAX_TAPE_BYTES,
            "test setup must keep the linear tape term alone under budget, so the rejection \
             below is attributable to the quadratic softmax_probs term specifically; got \
             {linear_only_bytes} bytes"
        );

        let err = check_tape_budget(seq_len_cap, &dims, &gdn_dims, rank, num_gqa_layers, 0)
            .expect_err("expected quadratic tape-budget rejection");
        assert!(
            err.contains("--seq-len-cap") && err.contains("softmax_probs"),
            "expected tape-budget rejection naming the quadratic term; got: {err}"
        );
    }

    #[test]
    fn check_tape_budget_accepts_modest_config() {
        let dims = tape_test_dims();
        let gdn_dims = tape_test_gdn_dims();
        check_tape_budget(512, &dims, &gdn_dims, 16, 4, 4)
            .expect("a modest seq-len/layer-count combination must stay under the tape budget");
    }
}

#[cfg(test)]
mod gdn_lora_ctor_tests {
    use super::*;

    /// Use asymmetric head counts to expose GDN B-factor sizing drift.
    /// See [`docs/design.md`](../../../docs/design.md#train_grad_full-details) for the value-head invariant.
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
