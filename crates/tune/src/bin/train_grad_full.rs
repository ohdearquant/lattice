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
// apply (1+gamma) internally, so layer/final norm fields carry raw gamma.
// q_norm/k_norm are shifted inside gqa_forward_with_cache, so they stay raw.
//
// Usage: train_grad_full --model-dir <path> --data-dir <path> [--first-layer 19]
//        [--steps 25] [--lr 1e-3] [--rank 8] [--alpha 16] [--max-train 3]
//        [--seq-len 64] [--gradcheck]

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::Instant;

use lattice_inference::attention::gdn::GatedDeltaNetWeights;
use lattice_inference::attention::gdn_backward::{
    GdnGrads, GdnSaved, gdn_backward, gdn_forward_save,
};
use lattice_inference::backward::attention_gqa::{AttnCache, gqa_backward, gqa_forward_with_cache};
use lattice_inference::backward::ops::{linear_vjp, rmsnorm_backward, swiglu_backward};
use lattice_inference::backward::tape::{rms_norm_forward, swiglu_forward};
use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::model::qwen35_config::Qwen35Config;
use lattice_inference::tokenizer::Tokenizer;
use lattice_tune::lora::{AdamState, LoraAdapter, LoraConfig, LoraLayer};

const TOP_LAYER: usize = 23;

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
  --data-dir    <PATH>   Dataset directory with train.jsonl + valid.jsonl (default: data/claude-logs-lora)
  --first-layer <N>      First materialised (trained) layer (default: 19)
  --steps       <N>      Adam steps (default: 25)
  --lr          <F>      Learning rate (default: 1e-3)
  --rank        <N>      LoRA rank (default: 8)
  --alpha       <F>      LoRA alpha (default: 16.0)
  --seq-len     <N>      Max tokens per sample (default: 64)
  --max-train   <N>      Training samples cap (default: 3)
  --max-valid   <N>      Held-out valid.jsonl samples for eval, 0=off (default: 16)
  --log-every   <N>      Print NLL every N steps (default: 5)
  --gradcheck            Run finite-difference gradcheck instead of training
  --probe       <N>      Gradcheck entries probed per array per layer (default: 6)
  --json                 Emit @@lattice JSON events to stdout alongside human output
  --save <PATH>          After training, write a PEFT safetensors adapter to PATH
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
        let (cos, sin) = model.rope_cos_sin_tables(seq_len);
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

#[derive(Clone, Copy)]
struct Dims {
    hidden: usize,
    vocab: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    inter: usize,
    q_dim: usize,
    kv_dim: usize,
    eps: f32,
}

/// GDN config dims (shared by all GDN layers), captured once from the config.
#[derive(Clone, Copy)]
struct GdnDims {
    num_kh: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    qkv_dim: usize,
    output_dim: usize,
    kernel_size: usize,
    scale: f32,
}

impl GdnDims {
    fn from_cfg(cfg: &Qwen35Config) -> Self {
        let key_dim = cfg.linear_key_head_dim;
        Self {
            num_kh: cfg.linear_num_key_heads,
            value_heads: cfg.linear_num_value_heads(),
            key_dim,
            value_dim: cfg.linear_value_head_dim,
            qkv_dim: cfg.linear_qkv_dim(),
            output_dim: cfg.linear_output_dim(),
            kernel_size: cfg.linear_conv_kernel_dim,
            scale: 1.0 / (key_dim as f32).sqrt(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MixerKind {
    Gqa,
    Gdn,
}

/// Borrowed frozen weights for one materialised layer (lifetime tied to &model).
/// Norm `pre_norm`/`post_norm`/`final_norm` fields carry raw gamma; the shifted
/// primitives (rms_norm_forward/rmsnorm_backward) apply (1+gamma) internally.
/// `q_norm`/`k_norm` are raw (gqa shifts internally). `lora_slot` indexes the
/// mutable LoRA param array for GQA layers; `None` for frozen GDN layers.
struct LayerW<'a> {
    kind: MixerKind,
    // GQA mixer (valid iff kind == Gqa)
    w_q: &'a [f32],
    w_k: &'a [f32],
    w_v: &'a [f32],
    w_o: &'a [f32],
    q_norm: &'a [f32],
    k_norm: &'a [f32],
    // GDN mixer (valid iff kind == Gdn)
    gdn: Option<&'a GatedDeltaNetWeights>,
    // common
    pre_norm: Vec<f32>,
    post_norm: Vec<f32>,
    w_gate: &'a [f32],
    w_up: &'a [f32],
    w_down: &'a [f32],
    lora_slot: Option<usize>,
}

struct Head<'a> {
    lm_head: &'a [f32],
    final_norm: &'a [f32],
}

/// Mutable LoRA factors for one layer.
///
/// GQA layers populate a_q/b_q/a_v/b_v; GDN fields are empty Vec.
/// GDN layers populate a_qkv/b_qkv/a_z/b_z/a_b/b_b/a_a/b_a/a_out/b_out;
/// GQA fields are empty Vec.
#[derive(Clone)]
struct LoraParams {
    // GQA fields (empty for GDN slots)
    a_q: Vec<f32>,
    b_q: Vec<f32>,
    a_v: Vec<f32>,
    b_v: Vec<f32>,
    // GDN fields (empty for GQA slots)
    a_qkv: Vec<f32>,
    b_qkv: Vec<f32>,
    a_z: Vec<f32>,
    b_z: Vec<f32>,
    a_b: Vec<f32>,
    b_b: Vec<f32>,
    a_a: Vec<f32>,
    b_a: Vec<f32>,
    a_out: Vec<f32>,
    b_out: Vec<f32>,
}

impl LoraParams {
    fn zeros_gqa(rank: usize, d: &Dims) -> Self {
        Self {
            a_q: vec![0.0; rank * d.hidden],
            b_q: vec![0.0; 2 * d.q_dim * rank],
            a_v: vec![0.0; rank * d.hidden],
            b_v: vec![0.0; d.kv_dim * rank],
            a_qkv: Vec::new(),
            b_qkv: Vec::new(),
            a_z: Vec::new(),
            b_z: Vec::new(),
            a_b: Vec::new(),
            b_b: Vec::new(),
            a_a: Vec::new(),
            b_a: Vec::new(),
            a_out: Vec::new(),
            b_out: Vec::new(),
        }
    }

    fn zeros_gdn(rank: usize, d: &Dims, gd: &GdnDims) -> Self {
        Self {
            a_q: Vec::new(),
            b_q: Vec::new(),
            a_v: Vec::new(),
            b_v: Vec::new(),
            a_qkv: vec![0.0; rank * d.hidden],
            b_qkv: vec![0.0; gd.qkv_dim * rank],
            a_z: vec![0.0; rank * d.hidden],
            b_z: vec![0.0; gd.output_dim * rank],
            a_b: vec![0.0; rank * d.hidden],
            b_b: vec![0.0; gd.num_kh * rank],
            a_a: vec![0.0; rank * d.hidden],
            b_a: vec![0.0; gd.num_kh * rank],
            a_out: vec![0.0; rank * gd.output_dim],
            b_out: vec![0.0; d.hidden * rank],
        }
    }

    /// Backward-compatibility alias used in TBV and eval paths.
    fn zeros(rank: usize, d: &Dims) -> Self {
        Self::zeros_gqa(rank, d)
    }
}

/// LoRA gradients for one layer (same shapes as LoraParams).
type Grads = LoraParams;

/// One sample's frozen context entering `first_layer`.
struct SeqCtx {
    h_in: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    tokens: Vec<u32>,
    completion_start: usize,
    seq_len: usize,
}

#[allow(clippy::large_enum_variant)]
enum MixerCache {
    Gqa(AttnCache),
    Gdn(GdnSaved),
}

/// Saved forward activations for one materialised layer.
struct LayerFwd {
    h_layer_in: Vec<f32>, // [seq*hidden] residual entering the layer (pre-norm input)
    inv_pre: Vec<f32>,    // [seq] pre-norm inv_rms per position
    mixer: MixerCache,
    h_mid: Vec<f32>,         // [seq*hidden] residual after mixer (post-norm input)
    inv_ffn: Vec<f32>,       // [seq] post-norm inv_rms per position
    gate_pre: Vec<Vec<f32>>, // seq × [inter]
    up_pre: Vec<Vec<f32>>,   // seq × [inter]
}

/// Per completion source-position head intermediates.
struct HeadPos {
    t: usize,
    target: u32,
    logits: Vec<f32>,
    inv_final: f32,
}

struct FullFwd {
    layers: Vec<LayerFwd>,
    h_final: Vec<f32>, // [seq*hidden] residual entering the head
    positions: Vec<HeadPos>,
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

/// Shifted RMSNorm over a full sequence; returns (normed [seq*hidden], inv [seq]).
fn rmsnorm_seq(
    h: &[f32],
    shift: &[f32],
    hidden: usize,
    seq: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut normed = vec![0.0f32; seq * hidden];
    let mut inv = vec![0.0f32; seq];
    for t in 0..seq {
        let (n, iv) = rms_norm_forward(&h[t * hidden..(t + 1) * hidden], shift, eps);
        normed[t * hidden..(t + 1) * hidden].copy_from_slice(&n);
        inv[t] = iv;
    }
    (normed, inv)
}

#[allow(clippy::too_many_arguments)]
fn forward_full(
    ctx: &SeqCtx,
    layers: &[LayerW],
    loras: &[LoraParams],
    head: &Head,
    d: &Dims,
    gdn_dims: &GdnDims,
    cfg: &Qwen35Config,
    rank: usize,
    scale: f32,
) -> FullFwd {
    let hidden = d.hidden;
    let seq = ctx.seq_len;
    let mut h = ctx.h_in.clone();
    let mut layers_fwd: Vec<LayerFwd> = Vec::with_capacity(layers.len());

    for lw in layers {
        let h_layer_in = h.clone();
        let (normed_pre, inv_pre) = rmsnorm_seq(&h, &lw.pre_norm, hidden, seq, d.eps);

        let (mixer_out, mixer_cache) = match lw.kind {
            MixerKind::Gqa => {
                let lora = &loras[lw.lora_slot.expect("GQA layer must have a LoRA slot")];
                let (out, cache) = gqa_forward_with_cache(
                    &normed_pre,
                    lw.w_q,
                    lw.w_k,
                    lw.w_v,
                    lw.w_o,
                    lw.q_norm,
                    lw.k_norm,
                    Some(&lora.a_q),
                    Some(&lora.b_q),
                    Some(&lora.a_v),
                    Some(&lora.b_v),
                    rank,
                    scale,
                    seq,
                    hidden,
                    d.num_q_heads,
                    d.num_kv_heads,
                    d.head_dim,
                    d.rope_dim,
                    &ctx.cos,
                    &ctx.sin,
                    d.eps,
                );
                (out, MixerCache::Gqa(cache))
            }
            MixerKind::Gdn => {
                let mut saved = GdnSaved::new(
                    seq,
                    gdn_dims.num_kh,
                    gdn_dims.value_heads,
                    gdn_dims.key_dim,
                    gdn_dims.value_dim,
                    hidden,
                    gdn_dims.qkv_dim,
                    gdn_dims.output_dim,
                    gdn_dims.kernel_size,
                    gdn_dims.scale,
                    d.eps,
                );
                let mut out = vec![0.0f32; seq * hidden];
                let lora = &loras[lw.lora_slot.expect("GDN layer must have a LoRA slot")];
                gdn_forward_save(
                    &normed_pre,
                    lw.gdn.expect("GDN layer must have GDN weights"),
                    cfg,
                    &mut saved,
                    &mut out,
                    Some(&lora.a_qkv),
                    Some(&lora.b_qkv),
                    Some(&lora.a_z),
                    Some(&lora.b_z),
                    Some(&lora.a_b),
                    Some(&lora.b_b),
                    Some(&lora.a_a),
                    Some(&lora.b_a),
                    Some(&lora.a_out),
                    Some(&lora.b_out),
                    rank,
                    scale,
                );
                (out, MixerCache::Gdn(saved))
            }
        };

        let mut h_mid = h_layer_in.clone();
        for (a, b) in h_mid.iter_mut().zip(mixer_out.iter()) {
            *a += *b;
        }

        let (normed_ffn, inv_ffn) = rmsnorm_seq(&h_mid, &lw.post_norm, hidden, seq, d.eps);
        let mut gate_pre = Vec::with_capacity(seq);
        let mut up_pre = Vec::with_capacity(seq);
        let mut h_next = h_mid.clone();
        for t in 0..seq {
            let (ffn_out, gp, up) = swiglu_forward(
                &normed_ffn[t * hidden..(t + 1) * hidden],
                lw.w_gate,
                lw.w_up,
                lw.w_down,
                hidden,
                d.inter,
            );
            for (o, f) in h_next[t * hidden..(t + 1) * hidden]
                .iter_mut()
                .zip(ffn_out.iter())
            {
                *o += *f;
            }
            gate_pre.push(gp);
            up_pre.push(up);
        }

        layers_fwd.push(LayerFwd {
            h_layer_in,
            inv_pre,
            mixer: mixer_cache,
            h_mid,
            inv_ffn,
            gate_pre,
            up_pre,
        });
        h = h_next;
    }

    // Head at completion source positions [completion_start-1, seq-1).
    let mut positions = Vec::new();
    for t in (ctx.completion_start - 1)..seq - 1 {
        let (final_normed, inv_final) =
            rms_norm_forward(&h[t * hidden..(t + 1) * hidden], head.final_norm, d.eps);
        let logits = lm_head_logits(head.lm_head, &final_normed, hidden, d.vocab);
        positions.push(HeadPos {
            t,
            target: ctx.tokens[t + 1],
            logits,
            inv_final,
        });
    }

    FullFwd {
        layers: layers_fwd,
        h_final: h,
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

#[allow(clippy::too_many_arguments)]
fn eval_chain_nll(
    caches: &[SeqCtx],
    layers: &[LayerW],
    loras: &[LoraParams],
    head: &Head,
    d: &Dims,
    gdn_dims: &GdnDims,
    cfg: &Qwen35Config,
    rank: usize,
    scale: f32,
) -> f32 {
    let mut nll_sum = 0.0f64;
    let mut n = 0usize;
    for ctx in caches {
        let fwd = forward_full(ctx, layers, loras, head, d, gdn_dims, cfg, rank, scale);
        for p in &fwd.positions {
            nll_sum += position_nll(&p.logits, p.target) as f64;
            n += 1;
        }
    }
    (nll_sum / n.max(1) as f64) as f32
}

/// Full-depth backward over one sample. Returns (nll_sum, n_comp, per-slot grads).
#[allow(clippy::too_many_arguments)]
fn nll_and_grads(
    fwd: &FullFwd,
    layers: &[LayerW],
    loras: &[LoraParams],
    head: &Head,
    d: &Dims,
    gdn_dims: &GdnDims,
    num_slots: usize,
    slot_kinds: &[MixerKind],
    rank: usize,
    scale: f32,
) -> (f32, usize, Vec<Grads>) {
    let hidden = d.hidden;
    let seq = fwd.h_final.len() / hidden;
    let n_comp = fwd.positions.len().max(1) as f32;
    let mut grads: Vec<Grads> = (0..num_slots)
        .map(|s| match slot_kinds[s] {
            MixerKind::Gqa => LoraParams::zeros_gqa(rank, d),
            MixerKind::Gdn => LoraParams::zeros_gdn(rank, d, gdn_dims),
        })
        .collect();

    // ---- Head + CE backward ----
    let mut d_h = vec![0.0f32; seq * hidden];
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
        let d_final = linear_vjp(head.lm_head, &d_logits, hidden, d.vocab);
        let d_h_t = rmsnorm_backward(
            &fwd.h_final[p.t * hidden..(p.t + 1) * hidden],
            head.final_norm,
            p.inv_final,
            &d_final,
        );
        for (j, dv) in d_h_t.iter().enumerate() {
            d_h[p.t * hidden + j] += *dv;
        }
    }

    // ---- Layer backward, top → first ----
    for li in (0..layers.len()).rev() {
        let lw = &layers[li];
        let lf = &fwd.layers[li];

        // FFN backward (all positions). h_next = h_mid + ffn_out, so the residual
        // skip seeds d_h_mid with d_h, and the FFN path adds onto it.
        let mut d_h_mid = d_h.clone();
        for t in 0..seq {
            let d_ffn_out = &d_h[t * hidden..(t + 1) * hidden];
            let (d_normed_ffn, _dm) = swiglu_backward(
                d_ffn_out,
                &lf.gate_pre[t],
                &lf.up_pre[t],
                lw.w_down,
                lw.w_gate,
                lw.w_up,
                hidden,
                d.inter,
            );
            let d_hm = rmsnorm_backward(
                &lf.h_mid[t * hidden..(t + 1) * hidden],
                &lw.post_norm,
                lf.inv_ffn[t],
                &d_normed_ffn,
            );
            for (j, dv) in d_hm.iter().enumerate() {
                d_h_mid[t * hidden + j] += *dv;
            }
        }

        // Mixer backward → d_normed_pre (grad w.r.t. the pre-attn-normed input).
        // h_mid = h_layer_in + mixer_out, so d_mixer_out = d_h_mid.
        let d_normed_pre: Vec<f32> = match &lf.mixer {
            MixerCache::Gqa(cache) => {
                let slot = lw.lora_slot.expect("GQA layer must have a LoRA slot");
                let lora = &loras[slot];
                let g = gqa_backward(
                    &d_h_mid,
                    cache,
                    lw.w_q,
                    lw.w_k,
                    lw.w_v,
                    lw.w_o,
                    lw.q_norm,
                    lw.k_norm,
                    Some(&lora.a_q),
                    Some(&lora.b_q),
                    Some(&lora.a_v),
                    Some(&lora.b_v),
                    rank,
                    scale,
                );
                grads[slot] = Grads {
                    a_q: g.grad_a_q,
                    b_q: g.grad_b_q,
                    a_v: g.grad_a_v,
                    b_v: g.grad_b_v,
                };
                g.dx
            }
            MixerCache::Gdn(saved) => {
                let slot = lw.lora_slot.expect("GDN layer must have a LoRA slot");
                let g = gdn_backward(
                    &d_h_mid,
                    saved,
                    lw.gdn.expect("GDN layer must have GDN weights"),
                );
                grads[slot] = Grads {
                    a_q: Vec::new(),
                    b_q: Vec::new(),
                    a_v: Vec::new(),
                    b_v: Vec::new(),
                    a_qkv: g.grad_a_qkv,
                    b_qkv: g.grad_b_qkv,
                    a_z: g.grad_a_z,
                    b_z: g.grad_b_z,
                    a_b: g.grad_a_b,
                    b_b: g.grad_b_b,
                    a_a: g.grad_a_a,
                    b_a: g.grad_b_a,
                    a_out: g.grad_a_out,
                    b_out: g.grad_b_out,
                };
                g.dx
            }
        };

        // Pre-norm backward. h_mid = h_layer_in + mixer_out → the residual skip
        // seeds d_h_layer_in with d_h_mid; the mixer+norm path adds onto it.
        let mut d_h_new = d_h_mid;
        for t in 0..seq {
            let d_hl = rmsnorm_backward(
                &lf.h_layer_in[t * hidden..(t + 1) * hidden],
                &lw.pre_norm,
                lf.inv_pre[t],
                &d_normed_pre[t * hidden..(t + 1) * hidden],
            );
            for (j, dv) in d_hl.iter().enumerate() {
                d_h_new[t * hidden + j] += *dv;
            }
        }
        d_h = d_h_new;
    }

    (nll_sum as f32, fwd.positions.len(), grads)
}

/// xorshift small-random fill in [-amp, amp].
fn rand_fill(rng: &mut u64, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            ((*rng >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * amp
        })
        .collect()
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
        .unwrap_or_else(|| PathBuf::from("data/claude-logs-lora"));
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
    let emit_json = parse_flag(&args, "--json");
    let save_path: Option<PathBuf> = parse_arg(&args, "--save").map(PathBuf::from);

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
    let final_norm = final_norm_s.to_vec(); // raw gamma; rms_norm_forward applies (1+gamma)
    let head = Head {
        lm_head: &lm_head,
        final_norm: &final_norm,
    };

    // Build the materialised layer stack [first_layer ..= 23], assigning a LoRA
    // slot to each GQA or GDN layer. All weight slices are borrowed from `model`.
    let mut layers: Vec<LayerW> = Vec::new();
    let mut slot_layers: Vec<usize> = Vec::new(); // global layer index per slot
    let mut slot_kinds: Vec<MixerKind> = Vec::new(); // GQA or GDN per slot
    for layer_idx in first_layer..=TOP_LAYER {
        if let Some((w_q, w_k, w_v, w_o, q_norm, k_norm, pre, post, gate, up, down)) =
            model.gqa_layer_weights(layer_idx)
        {
            let slot = slot_layers.len();
            slot_layers.push(layer_idx);
            slot_kinds.push(MixerKind::Gqa);
            layers.push(LayerW {
                kind: MixerKind::Gqa,
                w_q,
                w_k,
                w_v,
                w_o,
                q_norm,
                k_norm,
                gdn: None,
                pre_norm: pre.to_vec(),
                post_norm: post.to_vec(),
                w_gate: gate,
                w_up: up,
                w_down: down,
                lora_slot: Some(slot),
            });
        } else if let Some((gdn, pre, post, gate, up, down)) = model.gdn_layer_weights(layer_idx) {
            let slot = slot_layers.len();
            slot_layers.push(layer_idx);
            slot_kinds.push(MixerKind::Gdn);
            layers.push(LayerW {
                kind: MixerKind::Gdn,
                w_q: &[],
                w_k: &[],
                w_v: &[],
                w_o: &[],
                q_norm: &[],
                k_norm: &[],
                gdn: Some(gdn),
                pre_norm: pre.to_vec(),
                post_norm: post.to_vec(),
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
    let kinds: String = layers
        .iter()
        .map(|l| match l.kind {
            MixerKind::Gqa => 'A',
            MixerKind::Gdn => 'D',
        })
        .collect();
    println!(
        "  materialised {} layers [{}]: {kinds}  ({} LoRA slots at layers {:?})",
        layers.len(),
        (first_layer..=TOP_LAYER)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(","),
        num_slots,
        slot_layers
    );
    if num_slots == 0 {
        return Err("no trainable layers in range — nothing to train".into());
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
        .map(|s| match slot_kinds[s] {
            MixerKind::Gqa => LoraParams::zeros_gqa(rank, &dims),
            MixerKind::Gdn => LoraParams::zeros_gdn(rank, &dims, &gdn_dims),
        })
        .collect();
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
        );
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
            .map(|s| match slot_kinds[s] {
                MixerKind::Gqa => LoraParams {
                    a_q: rand_fill(&mut rng, rank * dims.hidden, 0.05),
                    b_q: rand_fill(&mut rng, 2 * dims.q_dim * rank, 0.05),
                    a_v: rand_fill(&mut rng, rank * dims.hidden, 0.05),
                    b_v: rand_fill(&mut rng, dims.kv_dim * rank, 0.05),
                    a_qkv: Vec::new(),
                    b_qkv: Vec::new(),
                    a_z: Vec::new(),
                    b_z: Vec::new(),
                    a_b: Vec::new(),
                    b_b: Vec::new(),
                    a_a: Vec::new(),
                    b_a: Vec::new(),
                    a_out: Vec::new(),
                    b_out: Vec::new(),
                },
                MixerKind::Gdn => LoraParams {
                    a_q: Vec::new(),
                    b_q: Vec::new(),
                    a_v: Vec::new(),
                    b_v: Vec::new(),
                    a_qkv: rand_fill(&mut rng, rank * dims.hidden, 0.05),
                    b_qkv: rand_fill(&mut rng, gdn_dims.qkv_dim * rank, 0.05),
                    a_z: rand_fill(&mut rng, rank * dims.hidden, 0.05),
                    b_z: rand_fill(&mut rng, gdn_dims.output_dim * rank, 0.05),
                    a_b: rand_fill(&mut rng, rank * dims.hidden, 0.05),
                    b_b: rand_fill(&mut rng, gdn_dims.num_kh * rank, 0.05),
                    a_a: rand_fill(&mut rng, rank * dims.hidden, 0.05),
                    b_a: rand_fill(&mut rng, gdn_dims.num_kh * rank, 0.05),
                    a_out: rand_fill(&mut rng, rank * gdn_dims.output_dim, 0.05),
                    b_out: rand_fill(&mut rng, dims.hidden * rank, 0.05),
                },
            })
            .collect();

        let fwd = forward_full(
            &caches[0], &layers, &loras, &head, &dims, &gdn_dims, &cfg, rank, scale,
        );
        let (_, _, analytic) = nll_and_grads(
            &fwd,
            &layers,
            &loras,
            &head,
            &dims,
            &gdn_dims,
            num_slots,
            &slot_kinds,
            rank,
            scale,
        );

        println!("  fd-eps center {fd_eps:.0e}  (per-entry min over 0.25/0.5/1/2x)");
        let mut worst = 0.0f64;
        let mut all_pass = true;
        for slot in 0..num_slots {
            let layer_idx = slot_layers[slot];
            // Build the list of (name, len) pairs for this slot's kind.
            let arrays: &[(&str, usize)] = match slot_kinds[slot] {
                MixerKind::Gqa => &[
                    ("a_q", analytic[slot].a_q.len()),
                    ("b_q", analytic[slot].b_q.len()),
                    ("a_v", analytic[slot].a_v.len()),
                    ("b_v", analytic[slot].b_v.len()),
                ],
                MixerKind::Gdn => &[
                    ("a_qkv", analytic[slot].a_qkv.len()),
                    ("b_qkv", analytic[slot].b_qkv.len()),
                    ("a_z", analytic[slot].a_z.len()),
                    ("b_z", analytic[slot].b_z.len()),
                    ("a_b", analytic[slot].a_b.len()),
                    ("b_b", analytic[slot].b_b.len()),
                    ("a_a", analytic[slot].a_a.len()),
                    ("b_a", analytic[slot].b_a.len()),
                    ("a_out", analytic[slot].a_out.len()),
                    ("b_out", analytic[slot].b_out.len()),
                ],
            };
            for &(name, alen) in arrays {
                let agrad: &[f32] = match name {
                    "a_q" => &analytic[slot].a_q,
                    "b_q" => &analytic[slot].b_q,
                    "a_v" => &analytic[slot].a_v,
                    "b_v" => &analytic[slot].b_v,
                    "a_qkv" => &analytic[slot].a_qkv,
                    "b_qkv" => &analytic[slot].b_qkv,
                    "a_z" => &analytic[slot].a_z,
                    "b_z" => &analytic[slot].b_z,
                    "a_b" => &analytic[slot].a_b,
                    "b_b" => &analytic[slot].b_b,
                    "a_a" => &analytic[slot].a_a,
                    "b_a" => &analytic[slot].b_a,
                    "a_out" => &analytic[slot].a_out,
                    _ => &analytic[slot].b_out,
                };
                if alen == 0 {
                    // skip arrays that don't belong to this slot's kind
                    continue;
                }
                let mut idxs = top_k_indices(agrad, probe.min(alen));
                let seed = (slot as u64 * 10
                    + match name {
                        "a_q" => 0,
                        "b_q" => 1,
                        "a_v" => 2,
                        "b_v" => 3,
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
                        let arr: &mut Vec<f32> = match name {
                            "a_q" => &mut loras[slot].a_q,
                            "b_q" => &mut loras[slot].b_q,
                            "a_v" => &mut loras[slot].a_v,
                            "b_v" => &mut loras[slot].b_v,
                            "a_qkv" => &mut loras[slot].a_qkv,
                            "b_qkv" => &mut loras[slot].b_qkv,
                            "a_z" => &mut loras[slot].a_z,
                            "b_z" => &mut loras[slot].b_z,
                            "a_b" => &mut loras[slot].a_b,
                            "b_b" => &mut loras[slot].b_b,
                            "a_a" => &mut loras[slot].a_a,
                            "b_a" => &mut loras[slot].b_a,
                            "a_out" => &mut loras[slot].a_out,
                            _ => &mut loras[slot].b_out,
                        };
                        arr[k]
                    };
                    let bump = |loras: &mut [LoraParams], val: f32| {
                        let arr: &mut Vec<f32> = match name {
                            "a_q" => &mut loras[slot].a_q,
                            "b_q" => &mut loras[slot].b_q,
                            "a_v" => &mut loras[slot].a_v,
                            "b_v" => &mut loras[slot].b_v,
                            "a_qkv" => &mut loras[slot].a_qkv,
                            "b_qkv" => &mut loras[slot].b_qkv,
                            "a_z" => &mut loras[slot].a_z,
                            "b_z" => &mut loras[slot].b_z,
                            "a_b" => &mut loras[slot].a_b,
                            "b_b" => &mut loras[slot].b_b,
                            "a_a" => &mut loras[slot].a_a,
                            "b_a" => &mut loras[slot].b_a,
                            "a_out" => &mut loras[slot].a_out,
                            _ => &mut loras[slot].b_out,
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
                        );
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
                        );
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
        .map(|s| match slot_kinds[s] {
            MixerKind::Gqa => LoraParams {
                a_q: rand_fill(&mut rng, rank * dims.hidden, init_amp),
                b_q: vec![0.0; 2 * dims.q_dim * rank],
                a_v: rand_fill(&mut rng, rank * dims.hidden, init_amp),
                b_v: vec![0.0; dims.kv_dim * rank],
                a_qkv: Vec::new(),
                b_qkv: Vec::new(),
                a_z: Vec::new(),
                b_z: Vec::new(),
                a_b: Vec::new(),
                b_b: Vec::new(),
                a_a: Vec::new(),
                b_a: Vec::new(),
                a_out: Vec::new(),
                b_out: Vec::new(),
            },
            MixerKind::Gdn => LoraParams {
                a_q: Vec::new(),
                b_q: Vec::new(),
                a_v: Vec::new(),
                b_v: Vec::new(),
                a_qkv: rand_fill(&mut rng, rank * dims.hidden, init_amp),
                b_qkv: vec![0.0; gdn_dims.qkv_dim * rank],
                a_z: rand_fill(&mut rng, rank * dims.hidden, init_amp),
                b_z: vec![0.0; gdn_dims.output_dim * rank],
                a_b: rand_fill(&mut rng, rank * dims.hidden, init_amp),
                b_b: vec![0.0; gdn_dims.num_kh * rank],
                a_a: rand_fill(&mut rng, rank * dims.hidden, init_amp),
                b_a: vec![0.0; gdn_dims.num_kh * rank],
                a_out: rand_fill(&mut rng, rank * gdn_dims.output_dim, init_amp),
                b_out: vec![0.0; dims.hidden * rank],
            },
        })
        .collect();

    let mut adam = AdamState::new();
    let (beta1, beta2, eps_adam) = (0.9f32, 0.999f32, 1e-8f32);

    let eval_valid = |loras: &[LoraParams]| -> Option<f32> {
        (!valid_caches.is_empty()).then(|| {
            eval_chain_nll(
                &valid_caches,
                &layers,
                loras,
                &head,
                &dims,
                &gdn_dims,
                &cfg,
                rank,
                scale,
            )
        })
    };

    let base_nll = eval_chain_nll(
        &caches, &layers, &loras, &head, &dims, &gdn_dims, &cfg, rank, scale,
    );
    let base_valid = eval_valid(&loras);
    if !emit_json {
        match base_valid {
            Some(v) => println!("\n  step    0  train NLL: {base_nll:.4}  held-out NLL: {v:.4}"),
            None => println!("\n  step    0  train NLL: {base_nll:.4}"),
        }
    }
    if emit_json {
        let val_json = match base_valid {
            Some(v) => format!("{v:.6}"),
            None => "null".to_string(),
        };
        println!(
            "@@lattice {{\"ev\":\"train_step\",\"step\":0,\"loss\":{base_nll:.6},\"val_loss\":{val_json},\"lr\":{lr:.6}}}"
        );
    }

    let tstep = Instant::now();
    for step in 1..=steps {
        let ctx = &caches[(step - 1) % caches.len()];
        let fwd = forward_full(
            ctx, &layers, &loras, &head, &dims, &gdn_dims, &cfg, rank, scale,
        );
        let (_nll, _n, grads) = nll_and_grads(
            &fwd,
            &layers,
            &loras,
            &head,
            &dims,
            &gdn_dims,
            num_slots,
            &slot_kinds,
            rank,
            scale,
        );

        for slot in 0..num_slots {
            let li = slot_layers[slot];
            match slot_kinds[slot] {
                MixerKind::Gqa => {
                    adam.step(
                        &format!("l{li}_a_q"),
                        &mut loras[slot].a_q,
                        &grads[slot].a_q,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        0.0,
                        false,
                    );
                    adam.step(
                        &format!("l{li}_b_q"),
                        &mut loras[slot].b_q,
                        &grads[slot].b_q,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        0.0,
                        false,
                    );
                    adam.step(
                        &format!("l{li}_a_v"),
                        &mut loras[slot].a_v,
                        &grads[slot].a_v,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        0.0,
                        false,
                    );
                    adam.step(
                        &format!("l{li}_b_v"),
                        &mut loras[slot].b_v,
                        &grads[slot].b_v,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        0.0,
                        false,
                    );
                }
                MixerKind::Gdn => {
                    adam.step(
                        &format!("l{li}_a_qkv"),
                        &mut loras[slot].a_qkv,
                        &grads[slot].a_qkv,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        0.0,
                        false,
                    );
                    adam.step(
                        &format!("l{li}_b_qkv"),
                        &mut loras[slot].b_qkv,
                        &grads[slot].b_qkv,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        0.0,
                        false,
                    );
                    adam.step(
                        &format!("l{li}_a_z"),
                        &mut loras[slot].a_z,
                        &grads[slot].a_z,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        0.0,
                        false,
                    );
                    adam.step(
                        &format!("l{li}_b_z"),
                        &mut loras[slot].b_z,
                        &grads[slot].b_z,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        0.0,
                        false,
                    );
                    adam.step(
                        &format!("l{li}_a_b"),
                        &mut loras[slot].a_b,
                        &grads[slot].a_b,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        0.0,
                        false,
                    );
                    adam.step(
                        &format!("l{li}_b_b"),
                        &mut loras[slot].b_b,
                        &grads[slot].b_b,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        0.0,
                        false,
                    );
                    adam.step(
                        &format!("l{li}_a_a"),
                        &mut loras[slot].a_a,
                        &grads[slot].a_a,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        0.0,
                        false,
                    );
                    adam.step(
                        &format!("l{li}_b_a"),
                        &mut loras[slot].b_a,
                        &grads[slot].b_a,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        0.0,
                        false,
                    );
                    adam.step(
                        &format!("l{li}_a_out"),
                        &mut loras[slot].a_out,
                        &grads[slot].a_out,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        0.0,
                        false,
                    );
                    adam.step(
                        &format!("l{li}_b_out"),
                        &mut loras[slot].b_out,
                        &grads[slot].b_out,
                        lr,
                        beta1,
                        beta2,
                        eps_adam,
                        0.0,
                        false,
                    );
                }
            }
        }

        if step % log_every == 0 || step == steps {
            let mean_nll = eval_chain_nll(
                &caches, &layers, &loras, &head, &dims, &gdn_dims, &cfg, rank, scale,
            );
            let val_nll = eval_valid(&loras);
            if !emit_json {
                match val_nll {
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
            if emit_json {
                let val_json = match val_nll {
                    Some(v) => format!("{v:.6}"),
                    None => "null".to_string(),
                };
                println!(
                    "@@lattice {{\"ev\":\"train_step\",\"step\":{step},\"loss\":{mean_nll:.6},\"val_loss\":{val_json},\"lr\":{lr:.6}}}"
                );
            }
        }
    }

    let final_nll = eval_chain_nll(
        &caches, &layers, &loras, &head, &dims, &gdn_dims, &cfg, rank, scale,
    );
    let secs = tstep.elapsed().as_secs_f64();
    let final_valid = eval_valid(&loras);
    if !emit_json {
        match (base_valid, final_valid) {
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
    }

    // Save PEFT adapter if --save was specified.
    let saved_path: Option<String> = if let Some(ref out_path) = save_path {
        let mut lora_layers = std::collections::HashMap::new();
        for (slot, lp) in loras.iter().enumerate() {
            let layer_idx = slot_layers[slot];
            // q_proj: A=(rank, hidden), B=(2*q_dim, rank).
            // b_q was allocated as vec![0.0; 2 * dims.q_dim * rank], so d_out = 2*q_dim.
            lora_layers.insert(
                (layer_idx, "q_proj".to_string()),
                LoraLayer {
                    a: lp.a_q.clone(),
                    b: lp.b_q.clone(),
                    d_in: dims.hidden,
                    d_out: 2 * dims.q_dim,
                    rank,
                },
            );
            // v_proj: A=(rank, hidden), B=(kv_dim, rank).
            lora_layers.insert(
                (layer_idx, "v_proj".to_string()),
                LoraLayer {
                    a: lp.a_v.clone(),
                    b: lp.b_v.clone(),
                    d_in: dims.hidden,
                    d_out: dims.kv_dim,
                    rank,
                },
            );
        }
        let adapter = LoraAdapter::new(
            LoraConfig {
                rank,
                alpha,
                target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            },
            lora_layers,
        );
        match adapter.save_safetensors(out_path) {
            Ok(()) => {
                println!("  adapter saved to {}", out_path.display());
                Some(out_path.display().to_string())
            }
            Err(e) => {
                eprintln!(
                    "warning: failed to save adapter to {}: {e}",
                    out_path.display()
                );
                None
            }
        }
    } else {
        None
    };

    if emit_json {
        let best_val_json = match base_valid {
            Some(_) => {
                let bv = eval_valid(&loras).unwrap_or(f32::INFINITY);
                format!("{bv:.6}")
            }
            None => "null".to_string(),
        };
        let saved_json = match &saved_path {
            Some(p) => format!("\"{}\"", p.replace('\\', "\\\\").replace('"', "\\\"")),
            None => "null".to_string(),
        };
        println!(
            "@@lattice {{\"ev\":\"train_done\",\"base_nll\":{base_nll:.6},\"final_nll\":{final_nll:.6},\"best_val\":{best_val_json},\"duration_s\":{secs:.3},\"saved\":{saved_json}}}"
        );
    }

    Ok(())
}
