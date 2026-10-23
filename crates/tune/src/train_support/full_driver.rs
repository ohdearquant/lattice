//! Shared exact multi-layer LoRA training driver.

use std::path::PathBuf;
use std::time::Instant;

use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::tokenizer::Tokenizer;
use rayon::prelude::*;

use crate::lora::AdamState;
use crate::lora::train_core::{
    AdamConfig, ChainScores, Dims, GdnDims, GdnLoraParams, GdnModuleSelection, Head, LayerW,
    LoraParams, MixerKind, SeqCtx, SlotLayout, TOP_LAYER, TapeGeometry, TrainCtx,
    apply_adam_updates, apply_gdn_adam_updates, eval_chain_nll, eval_chain_scores, forward_full,
    nll_and_grads, rand_fill, shifted,
};

use super::{Sample, load_jsonl, verify_tbv};

const _: () = {
    const fn _assert_sync<T: Sync>() {}
    _assert_sync::<Qwen35Model>();
};

/// Capture the frozen-prefix output (h_in entering `first_layer`) and RoPE
/// tables per sample, yielding the `SeqCtx` set the tape forward consumes.
/// Returns the caches plus the total number of masked completion positions.
fn build_caches(
    model: &Qwen35Model,
    samples: &[Sample],
    first_layer: usize,
) -> Result<(Vec<SeqCtx>, usize), Box<dyn std::error::Error>> {
    let caches = samples
        .par_iter()
        .map(|s| {
            let (h_in, _real_out) = model.capture_attn_io(&s.tokens, first_layer)?;
            let seq_len = s.tokens.len();
            let (cos, sin) = model.rope_cos_sin_tables(seq_len)?;
            Ok(SeqCtx {
                h_in,
                cos,
                sin,
                tokens: s.tokens.clone(),
                completion_start: s.completion_start,
                seq_len,
            })
        })
        .collect::<Result<Vec<_>, lattice_inference::InferenceError>>()?;
    let total_positions = caches.iter().map(|s| s.seq_len - s.completion_start).sum();
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

fn training_lora(dims: &Dims, rank: usize, rng: &mut u64, init_amp: f32) -> LoraParams {
    LoraParams {
        a_q: rand_fill(rng, rank * dims.hidden, init_amp),
        b_q: vec![0.0; 2 * dims.q_dim * rank],
        a_v: rand_fill(rng, rank * dims.hidden, init_amp),
        b_v: vec![0.0; dims.kv_dim * rank],
    }
}

/// Draw the training LoRA slots from `seed`, returning them with the advanced RNG
/// state so the GDN initializer can continue the same stream.
///
/// THE SEED ARRIVES AS AN ARGUMENT AND THE CALLER KEEPS NO GENERATOR OF ITS OWN.
/// The earlier shape left `let mut rng = seed;` inline in [`run`], one line that no
/// test could reach: a test calling the per-slot helper directly still passed with
/// that line changed back to a literal, which was measured. Folding the whole step
/// into one seeded function removes the untestable line rather than testing around
/// it, so re-hardcoding the seed anywhere on this path now fails a test.
fn initial_training_loras(
    seed: u64,
    num_slots: usize,
    dims: &Dims,
    rank: usize,
    init_amp: f32,
) -> (Vec<LoraParams>, u64) {
    let mut rng = seed;
    let loras: Vec<LoraParams> = (0..num_slots)
        .map(|_| training_lora(dims, rank, &mut rng, init_amp))
        .collect();
    (loras, rng)
}

/// Overwrite freshly initialized training slots with a saved adapter's weights.
///
/// Strict in BOTH directions, and that is the whole point of the shape. Every
/// (slot layer, module) pair this run's geometry expects must be present, and every
/// key the adapter carries must be consumed by one of those expectations. A module
/// the save block below writes but no arm here reads would otherwise be dropped in
/// silence, and a dropped layer is indistinguishable from a badly trained adapter
/// once the numbers are in a table: the score is simply worse, with nothing naming
/// the reason. The unconsumed-key check is what turns that into a refusal.
///
/// The fused-kernel projections are required exactly when this run trains them,
/// mirroring the save side's condition. Loading an `--gdn-modules all` adapter into
/// a `served` run is therefore refused rather than silently truncated.
#[cfg(feature = "safetensors")]
#[allow(clippy::too_many_arguments)]
fn overwrite_from_adapter(
    adapter: &crate::lora::LoraAdapter,
    loras: &mut [LoraParams],
    slot_layers: &[usize],
    gdn_loras: &mut [GdnLoraParams],
    gdn_slot_layers: &[usize],
    dims: &Dims,
    gdn_dims: &GdnDims,
    rank: usize,
    gdn_modules: GdnModuleSelection,
) -> Result<usize, String> {
    fn take(
        adapter: &crate::lora::LoraAdapter,
        layer: usize,
        module: &str,
        d_in: usize,
        d_out: usize,
        rank: usize,
        a: &mut Vec<f32>,
        b: &mut Vec<f32>,
    ) -> Result<(), String> {
        let saved = adapter
            .layers()
            .get(&(layer, module.to_string()))
            .ok_or_else(|| format!("adapter carries no {module} at layer {layer}"))?;
        if saved.rank != rank || saved.d_in != d_in || saved.d_out != d_out {
            return Err(format!(
                "{module} at layer {layer}: adapter is rank {} ({} -> {}), this run is rank {rank} ({d_in} -> {d_out})",
                saved.rank, saved.d_in, saved.d_out
            ));
        }
        if saved.a.len() != rank * d_in || saved.b.len() != d_out * rank {
            return Err(format!(
                "{module} at layer {layer}: array lengths {} / {} disagree with the declared shape {rank}x{d_in} / {d_out}x{rank}",
                saved.a.len(),
                saved.b.len()
            ));
        }
        a.clear();
        a.extend_from_slice(&saved.a);
        b.clear();
        b.extend_from_slice(&saved.b);
        Ok(())
    }

    if loras.len() != slot_layers.len() || gdn_loras.len() != gdn_slot_layers.len() {
        return Err("slot count disagrees with the layer map".to_string());
    }

    let mut consumed = 0usize;
    for (slot, &li) in slot_layers.iter().enumerate() {
        let p = &mut loras[slot];
        take(
            adapter,
            li,
            "q_proj",
            dims.hidden,
            2 * dims.q_dim,
            rank,
            &mut p.a_q,
            &mut p.b_q,
        )?;
        take(
            adapter,
            li,
            "v_proj",
            dims.hidden,
            dims.kv_dim,
            rank,
            &mut p.a_v,
            &mut p.b_v,
        )?;
        consumed += 2;
    }
    for (slot, &li) in gdn_slot_layers.iter().enumerate() {
        let g = &mut gdn_loras[slot];
        take(
            adapter,
            li,
            "in_proj_qkv",
            dims.hidden,
            gdn_dims.qkv_dim,
            rank,
            &mut g.a_qkv,
            &mut g.b_qkv,
        )?;
        take(
            adapter,
            li,
            "in_proj_z",
            dims.hidden,
            gdn_dims.output_dim,
            rank,
            &mut g.a_z,
            &mut g.b_z,
        )?;
        take(
            adapter,
            li,
            "out_proj",
            gdn_dims.output_dim,
            dims.hidden,
            rank,
            &mut g.a_out,
            &mut g.b_out,
        )?;
        consumed += 3;
        if gdn_modules.trains_fused_kernel_projections() {
            take(
                adapter,
                li,
                "in_proj_b",
                dims.hidden,
                gdn_dims.value_heads,
                rank,
                &mut g.a_b,
                &mut g.b_b,
            )?;
            take(
                adapter,
                li,
                "in_proj_a",
                dims.hidden,
                gdn_dims.value_heads,
                rank,
                &mut g.a_a,
                &mut g.b_a,
            )?;
            consumed += 2;
        }
    }

    if consumed != adapter.layers().len() {
        let mut extra: Vec<String> = adapter
            .layers()
            .keys()
            .filter(|(li, m)| {
                let gqa = slot_layers.contains(li) && matches!(m.as_str(), "q_proj" | "v_proj");
                let gdn = gdn_slot_layers.contains(li)
                    && (matches!(m.as_str(), "in_proj_qkv" | "in_proj_z" | "out_proj")
                        || (gdn_modules.trains_fused_kernel_projections()
                            && matches!(m.as_str(), "in_proj_b" | "in_proj_a")));
                !(gqa || gdn)
            })
            .map(|(li, m)| format!("{m}@{li}"))
            .collect();
        extra.sort();
        return Err(format!(
            "adapter carries {} layer/module entries, this run consumes {consumed}; unread: {}",
            adapter.layers().len(),
            extra.join(", ")
        ));
    }

    Ok(consumed)
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
    /// LoRA initialization seed for training; gradcheck uses its fixed seed.
    pub seed: u64,
    pub log_every: usize,
    pub gradcheck: bool,
    /// Whether gradcheck supplements top-gradient entries with deterministic
    /// strided probes.
    pub gradcheck_strided_probes: bool,
    pub probe: usize,
    pub fd_eps: f32,
    pub save_path: Option<String>,
    /// Start from a saved adapter instead of a fresh initialization.
    ///
    /// Required for any evaluation of one adapter against data it was not
    /// trained on: without it this binary can only ever score the task it just
    /// trained, so a cross-task comparison is not expressible.
    pub load_path: Option<String>,
    /// Fixed A-factor amplitude for compatibility callers; `None` uses the
    /// full trainer's `1 / sqrt(hidden)` initialization.
    pub a_init_amp: Option<f32>,
    /// Which GDN projections to train and save.
    ///
    /// [`GdnModuleSelection::Served`] is the default everywhere a caller does not
    /// say otherwise, because an adapter carrying the other two cannot be loaded
    /// by the Metal forward and nothing says so until load time.
    pub gdn_modules: GdnModuleSelection,
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

/// Upper bound on simultaneously live vocabulary-sized logit buffers for one
/// sample.
///
/// [`forward_full`] retains one vocabulary-sized `f32` row per supervised
/// completion position. [`nll_and_grads`] also allocates one vocabulary-sized
/// reverse workspace row while those retained rows remain live.
const MAX_LOGITS_BYTES: usize = 2 * 1024 * 1024 * 1024;

const BACKWARD_LOGITS_WORKSPACE_ROWS: usize = 1;
const EVAL_LOGITS_WORKSPACE_ROWS: usize = 0;

/// Upper bound on the frozen-prefix attention cache ([`build_caches`]'s
/// `h_in` + RoPE `cos`/`sin` buffers) for one sample set, in bytes. Bounds
/// the PRODUCT of raw token count and per-token buffer size — `MAX_SAMPLES`
/// and `MAX_SEQ_LEN_CAP` bound sample count and per-sample length
/// independently, and neither bounds their product (100_000 samples ×
/// 8_192 tokens = 819.2M raw tokens). Same 2 GiB order of magnitude as the
/// per-sample live-logits bound in [`MAX_LOGITS_BYTES`].
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

/// Reject any sample whose simultaneously live vocabulary-sized logit rows
/// would exceed [`MAX_LOGITS_BYTES`].
///
/// The tape retains logits only for supervised completion positions, not the
/// prompt. `workspace_rows` accounts for vocabulary-sized buffers that coexist
/// with the tape, and only one sample tape is live at a time.
fn check_live_logits_budget(
    samples: &[Sample],
    vocab: usize,
    workspace_rows: usize,
    label: &str,
) -> Result<(), String> {
    let bytes_per_logit = std::mem::size_of::<f32>();
    for (sample_index, sample) in samples.iter().enumerate() {
        if sample.completion_start == 0 || sample.completion_start >= sample.tokens.len() {
            return Err(format!(
                "{label} sample {} has invalid completion_start {}; expected \
                 1 <= completion_start < {} for its {} tokens",
                sample_index + 1,
                sample.completion_start,
                sample.tokens.len(),
                sample.tokens.len(),
            ));
        }
        let completion_positions = sample.tokens.len() - sample.completion_start;
        let live_rows = completion_positions
            .checked_add(workspace_rows)
            .ok_or_else(|| {
                format!(
                    "{label} sample {} live-logits row count overflows usize \
                     ({completion_positions} supervised positions + {workspace_rows} \
                     workspace rows)",
                    sample_index + 1,
                )
            })?;
        let logits_bytes = live_rows
            .checked_mul(vocab)
            .and_then(|elements| elements.checked_mul(bytes_per_logit))
            .ok_or_else(|| {
                format!(
                    "{label} sample {} live-logits size overflows usize \
                     ({completion_positions} supervised positions + {workspace_rows} \
                     workspace rows = {live_rows} live rows; {vocab} vocab × {bytes_per_logit}B)",
                    sample_index + 1,
                )
            })?;
        if logits_bytes > MAX_LOGITS_BYTES {
            return Err(format!(
                "{label} sample {} live logits would require {} MiB \
                 ({completion_positions} supervised positions + {workspace_rows} workspace rows \
                 = {live_rows} live rows; {vocab} vocab × {bytes_per_logit}B), exceeds {} MiB \
                 cap - reduce --seq-len or shorten the completion",
                sample_index + 1,
                logits_bytes / (1024 * 1024),
                MAX_LOGITS_BYTES / (1024 * 1024),
            ));
        }
    }
    Ok(())
}

fn check_training_logits_budget(
    samples: &[Sample],
    vocab: usize,
    steps: usize,
    gradcheck: bool,
) -> Result<(), String> {
    if gradcheck {
        return check_live_logits_budget(
            &samples[..samples.len().min(1)],
            vocab,
            BACKWARD_LOGITS_WORKSPACE_ROWS,
            "gradcheck",
        );
    }

    check_live_logits_budget(samples, vocab, EVAL_LOGITS_WORKSPACE_ROWS, "training")?;
    check_live_logits_budget(
        &samples[..steps.min(samples.len())],
        vocab,
        BACKWARD_LOGITS_WORKSPACE_ROWS,
        "training",
    )
}

fn check_valid_logits_budget(
    samples: &[Sample],
    vocab: usize,
    gradcheck: bool,
) -> Result<(), String> {
    if gradcheck {
        return Ok(());
    }
    check_live_logits_budget(samples, vocab, EVAL_LOGITS_WORKSPACE_ROWS, "held-out")
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
        seed,
        log_every,
        gradcheck,
        gradcheck_strided_probes,
        probe,
        fd_eps,
        save_path,
        load_path,
        a_init_amp,
        gdn_modules,
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
        "max-train:  {max_train}  seq-len: {seq_len_cap}  seed: {seed}  mode: {}",
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
        model.config().eos_token_id,
        seq_len_cap,
        max_train,
    )?;
    if train_samples.is_empty() {
        return Err("no training samples (check --data-dir, train.jsonl prompt+completion)".into());
    }
    println!("  {} samples loaded", train_samples.len());

    check_cache_budget(&train_samples, dims.hidden, dims.rope_dim, "train")?;
    check_training_logits_budget(&train_samples, dims.vocab, steps, gradcheck)?;

    // Capture the frozen prefix output (h_in entering first_layer) per sample.
    println!("\nBuilding frozen-prefix cache (layers 0..{first_layer})...");
    let tcache = Instant::now();
    let (caches, total_positions) = build_caches(&model, &train_samples, first_layer)?;
    println!(
        "  {} completion positions across {} samples in {:.1}s ({} threads)",
        total_positions,
        caches.len(),
        tcache.elapsed().as_secs_f64(),
        rayon::current_num_threads()
    );

    // Hold out valid.jsonl from training; compare its NLL to the training NLL.
    let valid_caches: Vec<SeqCtx> = if max_valid > 0 {
        match load_jsonl(
            &data_dir.join("valid.jsonl"),
            &tokenizer as &dyn Tokenizer,
            model.config().eos_token_id,
            seq_len_cap,
            max_valid,
        ) {
            Ok(vs) if !vs.is_empty() => {
                check_cache_budget(&vs, dims.hidden, dims.rope_dim, "valid")?;
                check_valid_logits_budget(&vs, dims.vocab, gradcheck)?;
                let tvalid = Instant::now();
                let (vc, vpos) = build_caches(&model, &vs, first_layer)?;
                println!(
                    "  held-out: {vpos} completion positions across {} valid samples in {:.1}s ({} threads)",
                    vc.len(),
                    tvalid.elapsed().as_secs_f64(),
                    rayon::current_num_threads()
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
        let ttbv = Instant::now();
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
            "\n  TBV (sample 0): model={model_masked:.5}  chain={chain_masked:.5}  diff={:.2e}  in {:.1}s",
            observation.diff,
            ttbv.elapsed().as_secs_f64()
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

        let (_, _, analytic, gdn_analytic) = {
            let fwd = forward_full(&caches[0], &layers, &loras, &gdn_loras, &head, &train_ctx)?;
            nll_and_grads(&fwd, &layers, &loras, &head, &train_ctx)?
        };

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
    let (mut loras, rng) = initial_training_loras(seed, num_slots, &dims, rank, init_amp);
    // Reuses `rng`'s current stream position as the seed (matches the prior
    // inline behavior of drawing from the same generator as `loras` above).
    let mut gdn_loras: Vec<GdnLoraParams> =
        zero_b_gdn_loras(num_gdn_slots, rank, dims.hidden, &gdn_dims, rng, init_amp);

    if let Some(ref path) = load_path {
        #[cfg(feature = "safetensors")]
        {
            let adapter = crate::lora::load_peft_safetensors(std::path::Path::new(path))
                .map_err(|e| format!("load adapter: {e}"))?;
            let consumed = overwrite_from_adapter(
                &adapter,
                &mut loras,
                &slot_layers,
                &mut gdn_loras,
                &gdn_slot_layers,
                &dims,
                &gdn_dims,
                rank,
                gdn_modules,
            )?;
            println!("  loaded adapter: {consumed} layer/module entries from {path}");
        }
        #[cfg(not(feature = "safetensors"))]
        {
            let _ = path;
            return Err("--load requires the safetensors feature (--features safetensors)".into());
        }
    }

    let mut adam = AdamState::new();

    // Held-out scoring also reports teacher-forced agreement, because NLL alone
    // cannot say whether the adapter would emit the right tokens: two adapters can
    // sit at the same loss and disagree on every argmax. The counts print beside the
    // loss rather than replacing it; `eval_chain_scores` documents what the word
    // "exact" means here and what it does not.
    let eval_valid_scores = |loras: &[LoraParams],
                             gdn_loras: &[GdnLoraParams]|
     -> Result<Option<ChainScores>, Box<dyn std::error::Error>> {
        if valid_caches.is_empty() {
            return Ok(None);
        }
        Ok(Some(eval_chain_scores(
            &valid_caches,
            &layers,
            loras,
            gdn_loras,
            &head,
            &train_ctx,
        )?))
    };
    let eval_valid = |loras: &[LoraParams],
                      gdn_loras: &[GdnLoraParams]|
     -> Result<Option<f32>, Box<dyn std::error::Error>> {
        Ok(eval_valid_scores(loras, gdn_loras)?.map(|s| s.mean_nll))
    };

    // These two passes are reported because they are real work that used to print nothing at all.
    // A cost model built from the step-loop clock alone understates by a term that grows with the
    // pool and the held-out size, so every phase states its own duration and an external wall clock
    // becomes a CHECK on this log rather than the only place the time can be found.
    //
    // THE "ROUGHLY 46% OF THE RUN" FIGURE THIS COMMENT CARRIED IS WITHDRAWN. It was a ratio of
    // phase time to WALL time taken on a machine that thermally sleeps under load, and a sleep
    // inflates the denominator without touching the numerator: `Instant` is `mach_absolute_time`
    // and does not advance while the machine is asleep. That ratio therefore measured the box's
    // thermal behaviour mixed with this code's instrumentation gap, and the two cannot be
    // separated after the fact. The gap it described is real and was verified at source; only its
    // size came from the bad instrument.
    //
    // Replacement, from one externally stamped run at max-train 32 / max-valid 22 — named rather
    // than called "representative", because that word is not checkable: prologue phases totalled
    // 1895.4s (model load 3.7, prefix cache 959.9, held-out cache 350.5, consistency check 23.2,
    // baseline train 354.6, baseline held-out 203.5) against 533s inside the step loop. The phases
    // outside the loop are the large majority of the work, which is a stronger claim than the
    // withdrawn one rather than a weaker one.
    let tbase_train = Instant::now();
    let base_nll = eval_chain_nll(&caches, &layers, &loras, &gdn_loras, &head, &train_ctx)?;
    let base_train_secs = tbase_train.elapsed().as_secs_f64();
    let tbase_valid = Instant::now();
    // One pass, both readings. Calling `eval_valid` here and the scorer again below
    // would buy the agreement counts at the price of a second full held-out pass,
    // which at the sizes this runs at is the largest single cost outside the step
    // loop.
    let base_valid_scores = eval_valid_scores(&loras, &gdn_loras)?;
    let base_valid = base_valid_scores.map(|s| s.mean_nll);
    let base_valid_secs = tbase_valid.elapsed().as_secs_f64();
    println!(
        "  baseline scoring: train pass {base_train_secs:.1}s, held-out pass {base_valid_secs:.1}s"
    );
    match base_valid {
        Some(v) => println!("\n  step    0  train NLL: {base_nll:.4}  held-out NLL: {v:.4}"),
        None => println!("\n  step    0  train NLL: {base_nll:.4}"),
    }
    if let Some(sc) = base_valid_scores {
        println!(
            "  held-out agreement (teacher-forced, not a free-running decode): \
NLL {:.6}  tokens {}/{}  sequences {}/{}",
            sc.mean_nll, sc.positions_correct, sc.positions, sc.sequences_correct, sc.sequences
        );
    }

    // THREE CLOCKS, NOT ONE, AND A STEP COST THAT IS READ RATHER THAN DERIVED.
    //
    // What this replaces: a single clock started before the loop and read AFTER the epilogue's
    // extra train pass but BEFORE the epilogue's held-out pass. Its `in {secs}s` figure therefore
    // carried one whole scoring pass it never named and omitted a second one. Anybody recovering a
    // per-step cost from it — loop figure minus the in-loop scoring points, divided by steps —
    // subtracts from a quantity with an invisible term in it and gets an answer too large by that
    // term over the step count. Measured at 32-train/22-valid on a 2-step run: the unnamed pass was
    // 314.7s, inflating a step by ~157s against a true cost of tens of seconds. The reported
    // 847.7s decomposed exactly as 533s of loop plus that 314.7s, which is how the term was found.
    //
    // Each quantity now accumulates on its own clock and prints on its own line.
    let mut train_step_secs = 0.0f64;
    let mut in_loop_score_secs = 0.0f64;
    let mut score_points = 0usize;
    // The final step ALWAYS scores, because `step == steps` forces the log even when log_every does
    // not divide it. So after any run with steps >= 1 the epilogue re-scored weights the loop had
    // just scored, recomputing an identical value from an unchanged state: a full train pass plus a
    // full held-out pass, 558s at the sizes above, paid once per run for nothing. Carry the loop's
    // own last scoring forward instead of repeating it.
    let mut last_scored: Option<(f32, Option<f32>)> = None;
    let tstep = Instant::now();
    for step in 1..=steps {
        let tone = Instant::now();
        let ctx = &caches[(step - 1) % caches.len()];
        let (_nll, _n, grads, gdn_grads) = {
            let fwd = forward_full(ctx, &layers, &loras, &gdn_loras, &head, &train_ctx)?;
            nll_and_grads(&fwd, &layers, &loras, &head, &train_ctx)?
        };

        apply_adam_updates(&mut adam, &mut loras, &grads, &train_ctx);
        apply_gdn_adam_updates(
            &mut adam,
            &mut gdn_loras,
            &gdn_grads,
            &train_ctx,
            gdn_modules,
        );
        train_step_secs += tone.elapsed().as_secs_f64();

        if step % log_every == 0 || step == steps {
            let tscore = Instant::now();
            let mean_nll = eval_chain_nll(&caches, &layers, &loras, &gdn_loras, &head, &train_ctx)?;
            let valid_nll = eval_valid(&loras, &gdn_loras)?;
            in_loop_score_secs += tscore.elapsed().as_secs_f64();
            score_points += 1;
            last_scored = Some((mean_nll, valid_nll));
            match valid_nll {
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
    let secs = tstep.elapsed().as_secs_f64();

    // steps == 0 left the loop with nothing to have scored, and the state it would score is the
    // state the BASELINE already scored: no step ran, so nothing moved. The epilogue that used to
    // recompute it paid a full train pass plus a full held-out pass to reproduce a number already
    // printed, which on an eval-only run is the largest cost in the run. Carry the baseline forward
    // exactly as the loop carries its last scoring point.
    //
    // The old `epilogue_score_secs` clock is gone rather than left printing a constant zero: a
    // field whose only possible value is the empty one cannot carry the signal its name promises,
    // and a reader would take the zero for evidence the pass was skipped on paths where it never
    // ran in the first place.
    let (final_nll, final_valid) = last_scored.unwrap_or((base_nll, base_valid));

    let per_step = if steps > 0 {
        format!("{:.2}s/step", train_step_secs / steps as f64)
    } else {
        "no steps".to_string()
    };
    println!(
        "  step loop: {steps} steps in {train_step_secs:.1}s ({per_step}), \
in-loop scoring {in_loop_score_secs:.1}s over {score_points} point(s)"
    );

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
                // The list comes from the selection, never from a literal here: a
                // hand-written list is a second copy of the servable-module set and
                // drifts from the one the Metal loader refuses by.
                target_modules.extend(gdn_modules.modules().iter().map(|m| (*m).to_string()));
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
                    // Written only when they were TRAINED. Under the default they
                    // were never stepped, so `b_b`/`b_a` are still exactly zero and
                    // the layer would be an identity the loader refuses anyway.
                    if gdn_modules.trains_fused_kernel_projections() {
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
                    }
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
                dtype: "f32".into(),
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

/// MEASURED MUTATION COVERAGE, six live arms and one void one. Predicted before
/// running, reconciled after.
///
///   delete the unconsumed-key accounting         -> 2 red (ignored-module, fused pair)
///   never load v_proj                            -> 2 red (values, missing-entry)
///   drop the rank/width check                    -> 1 red (missing-entry)
///   hand take() (b_v, a_v) instead of (a_v, b_v)  -> 1 red (values)
///   require the fused projections always         -> 3 red
///   flip one ULP of every loaded A value         -> 3 red (values, fused, round trip)
///   round every loaded A value to a 1/1024 grid  -> 0 red, VOID
///
/// Two arms came back wider than predicted and one came back empty, and the empty
/// one is the one to keep in mind. The 1/1024 rounding changed nothing because the
/// fixture's ramp is built on 1/128, so every value was already on the coarser
/// grid: a lossy-copy arm the fixture cannot express certifies nothing, and it
/// reads as a passing suite. The one-ULP flip is its replacement precisely because
/// no fixture value can survive it.
///
/// The wider-than-predicted arms have a shared cause: the ignored-module fixture is
/// built from the SERVED save set, so an unconditional fused take refuses it before
/// the accounting it was written to exercise ever runs. A fixture drawn from one
/// selection cannot isolate a defect in the other selection's arm.
#[cfg(all(test, feature = "safetensors"))]
mod adapter_load_tests;

#[cfg(test)]
mod run_bounds_tests;

#[cfg(test)]
mod gdn_lora_ctor_tests;
