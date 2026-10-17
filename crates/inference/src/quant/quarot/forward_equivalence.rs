//! Forward-equivalence refuse-on-fail harness for QuaRot Qwen3.5 conversion
//! (step 3c-4 of ADR-044).
//!
//! ## What this module owns
//!
//! The converter binary (step 3c-5) prepares this gate after
//! `materialize_lm_head`, completes it after
//! `fuse_rmsnorms → absorb_rotations`, and does so before quantizing or
//! writing artifacts to disk. Direct map-backed callers use
//! [`assert_forward_equivalence_qwen35`]. The gate runs two checks and
//! refuses the conversion (`Err(InferenceError::Inference)`) when either
//! exceeds `tolerance`:
//!
//! 1. **Rotation-chain probe** — a linearized end-to-end forward (see
//!    §"What the probe does" below) that exercises the residual stream
//!    through one input projection + one output projection per attention
//!    layer plus the full MLP block. Catches errors that propagate
//!    through the residual stream.
//! 2. **Per-tensor matrix-equivalence check** — enumerates the
//!    config-required planned tensor set (via [`qwen_required_tensor_names`]
//!    filtered through [`RotationPlan`], plus `lm_head.weight`
//!    unconditionally for the post-materialize state), reconstructs the
//!    expected `W_rot` from the original-side source using the same
//!    primitives the pipeline does (column-multiply by `(1 + γ)` if
//!    fusion applies, then [`absorb_input_rotation_f64`] or
//!    [`absorb_output_rotation_f64`]), and compares **every stored
//!    element** to the actual rotated matrix. Catches per-tensor
//!    corruption / missed rotation absorption / missed fusion on tensors
//!    the chain probe does not consume — `k_proj`, `v_proj`, gate-z half
//!    of `q_proj`, `in_proj_qkv`, `in_proj_a`, `in_proj_b` — and refuses
//!    when a planned tensor is missing from its original source or the
//!    rotated working set.
//!
//! The two checks are complementary: (1) catches residual-stream
//! breakage holistically, (2) catches every planned tensor individually
//! including the ones (1) shortcuts past. The refuse semantic is the
//! same — `Err(InferenceError::Inference)` with `max_abs_error`,
//! `mean_abs_error`, `tolerance`, and (for chain probe failures)
//! `probe_tokens` in the diagnostic message.
//!
//! ## What the chain probe does (and does NOT do)
//!
//! The chain probe is **NOT a faithful Qwen3.5 forward.** It is a
//! deterministic *residual-stream walk* that consumes a deliberately
//! narrow subset of the rotation plan — one input projection + one
//! output projection per attention layer, plus the full dense MLP and
//! the embedding / lm_head edges. It does NOT exercise every planned
//! tensor (it skips `k_proj`, `v_proj`, the gate-z half of `q_proj`,
//! `in_proj_qkv`, `in_proj_a`, `in_proj_b`); the per-tensor
//! matrix-equivalence check is what covers those. The chain probe's
//! value is end-to-end residual-stream coverage; the per-tensor check's
//! value is per-matrix completeness:
//!
//! ```text
//!   h ← embed_tokens[token]
//!   for each layer L:
//!       h_pre ← (1 + γ_in) ⊙ rms_normalize(h)               [input_layernorm]
//!       if full_attention(L):
//!           q_full ← q_proj · h_pre                          [2*q_dim long]
//!           attn_out ← o_proj · q_full[0..q_dim]             [hidden]
//!       else:
//!           z ← in_proj_z · h_pre                            [output_dim]
//!           attn_out ← out_proj · z                          [hidden]
//!       h ← h + attn_out
//!       h_pre ← (1 + γ_post) ⊙ rms_normalize(h)              [post_attention_layernorm]
//!       gate ← gate_proj · h_pre                             [intermediate]
//!       up   ← up_proj · h_pre                               [intermediate]
//!       mid  ← gate + up        ← LINEARIZED (NOT silu(gate)⊙up)
//!       mlp_out ← down_proj · mid
//!       h ← h + mlp_out
//!   h_final ← (1 + γ_final) ⊙ rms_normalize(h)               [final_norm]
//!   logits ← lm_head · h_final                               [or embed_tokens fallback]
//! ```
//!
//! `silu(gate) ⊙ up` is replaced with `gate + up` so the probe is purely
//! linear in `h`. The QuaRot rotation absorption identities then hold
//! exactly across the chain — any divergence between `original` and
//! `rotated` logits is a pipeline bug, not a non-linearity artifact.
//!
//! For the GQA attention block the probe shortcuts through `q_proj`'s Q
//! half only (the first `full_q_dim` elements of the `[2 * full_q_dim,
//! hidden]` Q + gate_z fused row), feeds that to `o_proj` directly, and
//! skips the actual attention compute (softmax / RoPE / K, V). The
//! QuaRot v0 residual rotation does not depend on K, V being part of
//! the chain — they are input-side rotated for activation quantization
//! deferred to v1 and are still exercised by the per-layer RMSNorm
//! fusion targets (the fusion plan column-multiplies `(1 + γ_in)` into
//! all of `q_proj`, `k_proj`, `v_proj`).
//!
//! ### What the gate IS sufficient to catch
//!
//! - Missing rotation absorption on **every** tensor in the rotation
//!   plan — both the ones in the chain probe (`embed_tokens`, `q_proj`,
//!   `o_proj`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`,
//!   `down_proj`, `lm_head`) and the ones the per-tensor check adds
//!   coverage for (`k_proj`, `v_proj`, gate-z half of `q_proj`,
//!   `in_proj_qkv`, `in_proj_a`, `in_proj_b`).
//! - RMSNorm fusion missing: `(1 + γ)` left online for `input_layernorm`,
//!   `post_attention_layernorm`, or `final_norm`. The per-tensor check
//!   verifies the post-fusion algebraic identity holds for each fused
//!   downstream, so a partial fusion (e.g., fused into `q_proj` but not
//!   `k_proj`) is caught.
//! - `lm_head` not materialized when the input config is tied — the
//!   rotated probe needs a present `lm_head.weight` to consume the
//!   `(1 + γ_final)` column scale and the input-side rotation.
//! - `final_norm` fusion target not appended — γ_final left non-zero in
//!   the rotated set means the runtime double-applies the scale once
//!   it's also baked into lm_head.
//! - Wrong rotation (different `R` between the rotation passed to this
//!   gate and the one absorbed into the rotated working set, or wrong
//!   absorption side).
//! - Tensor-level corruption on any planned tensor (scaling, partial
//!   re-quantization, double-rotation), even on tensors the chain
//!   probe never matvecs against.
//!
//! ### What the gate is NOT sufficient to catch
//!
//! - Bugs that only manifest in `silu(gate) ⊙ up` — a real Qwen forward
//!   would exercise this; the probe linearizes it. Step 4 of ADR-044
//!   (perplexity bench on real calibration data) is the full check.
//! - Bugs in the full attention compute (softmax / RoPE) — the probe
//!   shortcuts through projections only.
//! - Quantization error — the gate runs in f64 on the pre-quantization
//!   working set. Q4 round-trip error is bounded by the Q4 bridge's own
//!   tests (`weights::q4_weights`).
//! - Tensors **outside** the rotation plan — e.g., GDN's
//!   `linear_attn.norm.weight` (plain-gamma, intentionally NOT fused per
//!   ADR-044 §Risks), `q_norm`/`k_norm`, `A_log`, `dt_bias`, `conv1d`.
//!   The plan and fusion plan together define the gate's correctness
//!   contract; tensors outside both are caller-responsibility.
//!
//! ## Tied vs untied originals
//!
//! For the converter flow, `materialize_lm_head_for_qwen35` runs before
//! gate preparation, so the original chain logits use the materialized
//! `lm_head.weight`. The post-rotation per-tensor check streams source
//! matrices from the checkpoint reader. For a tied input, it reconstructs
//! `lm_head.weight` from `embed_tokens.weight`, exactly matching
//! materialization even if a stray lm_head tensor exists on disk.
//!
//! For callers that take the `original` snapshot BEFORE materialization
//! (or for direct unit-level tests of a tied input) the probe falls back
//! to `embed_tokens.weight` when `lm_head.weight` is absent and the
//! config says `tie_word_embeddings=true`. This matches the runtime
//! `logits_weight()` semantics at `model/qwen35/weights.rs` and produces
//! the same logits as the rotated probe by construction (`lm_head_rot =
//! embed_tokens · diag(1 + γ_final) · R^T` cancels against `R · rmsnorm(h_orig)`).
//!
//! ## Scope
//!
//! Qwen3.5 dense (non-MoE) configs only. MoE is rejected explicitly here
//! — the rotation pipeline already refuses MoE (`fuse_rmsnorms` errors
//! via [`crate::quant::quarot::rmsnorm_fusion::qwen35_per_layer_fusion_plan`]),
//! and the probe has no expert-mixing path.

use std::collections::HashMap;
use std::mem::size_of;

use crate::error::InferenceError;
use crate::model::qwen35::qwen_required_tensor_names;
use crate::model::qwen35_config::Qwen35Config;
use crate::quant::quarot::hadamard::RandomizedHadamard;
use crate::quant::quarot::io::QuarotTensorReader;
use crate::quant::quarot::lm_head::{
    QWEN35_EMBED_TOKENS_NAME, QWEN35_FINAL_NORM_NAME, QWEN35_LM_HEAD_NAME,
    qwen35_final_norm_fusion_target,
};
use crate::quant::quarot::pipeline::TensorEntry;
use crate::quant::quarot::plan::{AbsorptionSide, RotationPlan};
use crate::quant::quarot::rmsnorm_fusion::{
    RmsNormFusionTarget, fuse_shifted_rmsnorm_into_next_layer_f64, qwen35_per_layer_fusion_plan,
};
use crate::quant::quarot::rotation::{absorb_input_rotation_f64, absorb_output_rotation_f64};

#[cfg(test)]
pub(crate) mod pre_admission_allocation_tracking;

/// Maximum retained bytes for chain-probe tokens, logit row descriptors, and
/// complete f64 logit rows in a prepared forward-equivalence snapshot.
///
/// The default four probes for Qwen3.5-0.8B retain about 7.6 MiB. A 64 MiB
/// ceiling permits 33 probes at its 248,320-token vocabulary while preventing
/// caller-controlled probe counts from adding an unbounded resident working set.
const MAX_FORWARD_EQUIVALENCE_PROBE_SNAPSHOT_BYTES: usize = 64 * 1024 * 1024;

/// Probe sample size + tolerance settings for
/// [`assert_forward_equivalence_qwen35`].
///
/// The default is `num_probe_tokens = 4`, `tolerance = 1e-5`, matching
/// the ADR-044 §"Step 3c contract" requirement
/// (`‖rotated_forward − original_forward‖ < 1e-5` on a small batch).
#[derive(Debug, Clone)]
pub struct ForwardEquivalenceConfig {
    /// Number of token IDs to probe. Must be > 0. The retained token and
    /// complete-logit evidence must fit the 64 MiB preparation budget.
    pub num_probe_tokens: usize,
    /// Max-abs-error threshold (per-logit). Must be > 0.
    pub tolerance: f64,
    /// Seed for the deterministic token sampler.
    pub seed: u64,
}

pub(crate) struct ForwardEquivalenceAdmission<'cfg, 'forward> {
    cfg: &'cfg Qwen35Config,
    forward_cfg: &'forward ForwardEquivalenceConfig,
}

impl Default for ForwardEquivalenceConfig {
    fn default() -> Self {
        Self {
            num_probe_tokens: 4,
            tolerance: 1e-5,
            seed: 0xCAFE_BABE_DEAD_BEEF,
        }
    }
}

/// Result of a successful [`assert_forward_equivalence_qwen35`] call.
///
/// The numerical metrics are diagnostic — `max_abs_error <= tolerance` is
/// the assertion contract that gated the `Ok` return.
#[derive(Debug, Clone)]
pub struct ForwardEquivalenceReport {
    /// Max-abs delta the gate observed, taken as the maximum of (a) the
    /// chain probe's per-logit max across all probe tokens and (b) the
    /// per-tensor matrix-equivalence check's per-element max across
    /// every planned tensor. A successful return guarantees this value
    /// is at or below `tolerance`.
    pub max_abs_error: f64,
    /// Mean absolute error across all `num_probe_tokens * vocab_size`
    /// chain-probe logits. **Chain-probe-only** — per-tensor deltas
    /// contribute to `max_abs_error` but not to this mean, because
    /// element counts vary per tensor and a single weighted mean would
    /// be skewed by the largest matrix.
    pub mean_abs_error: f64,
    /// Tokens actually probed by the chain probe (deterministic from
    /// `seed` and `vocab_size`).
    pub probe_tokens: Vec<u32>,
    /// Tolerance the gate evaluated against (echoed for downstream logging).
    pub tolerance: f64,
}

/// Original-side evidence retained while the converter mutates its f64
/// working set in place.
///
/// The chain probe consumes the full original model, but only its logits
/// are needed after rotation. Matrix equivalence separately needs the
/// original RMSNorm scales and one planned matrix at a time; the scales
/// live here while matrices are streamed from [`QuarotTensorReader`].
/// Borrowing the admitted configuration and prepared rotation prevents
/// completion from substituting either after the original evidence is captured.
pub(crate) struct ForwardEquivalenceSnapshot<'a> {
    cfg: &'a Qwen35Config,
    rotation: &'a RandomizedHadamard,
    probe_tokens: Vec<u32>,
    original_logits: Vec<Vec<f64>>,
    fusion_gammas: HashMap<String, TensorEntry>,
    tolerance: f64,
}

trait OriginalTensorSource {
    fn has_tensor(&self, name: &str) -> bool;

    fn load_tensor(&self, name: &str) -> Result<TensorEntry, InferenceError>;
}

impl OriginalTensorSource for HashMap<String, TensorEntry> {
    fn has_tensor(&self, name: &str) -> bool {
        self.contains_key(name)
    }

    fn load_tensor(&self, name: &str) -> Result<TensorEntry, InferenceError> {
        self.get(name)
            .cloned()
            .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
    }
}

struct ReaderOriginalTensorSource<'a> {
    reader: &'a QuarotTensorReader,
    tied_lm_head_from_embed: bool,
}

impl OriginalTensorSource for ReaderOriginalTensorSource<'_> {
    fn has_tensor(&self, name: &str) -> bool {
        if self.tied_lm_head_from_embed && name == QWEN35_LM_HEAD_NAME {
            return false;
        }
        self.reader.has_tensor(name)
    }

    fn load_tensor(&self, name: &str) -> Result<TensorEntry, InferenceError> {
        let (data, shape) = self.reader.read_tensor_f64(name)?;
        Ok(TensorEntry {
            name: name.to_string(),
            shape,
            data,
        })
    }
}

pub(crate) fn preflight_forward_equivalence_probe_budget(
    cfg: &Qwen35Config,
    forward_cfg: &ForwardEquivalenceConfig,
) -> Result<(), InferenceError> {
    let Some(logit_elements) = forward_cfg.num_probe_tokens.checked_mul(cfg.vocab_size) else {
        return reject_forward_equivalence_admission(format_args!(
            "assert_forward_equivalence_qwen35: retained chain-logit size overflow"
        ));
    };
    let Some(logit_bytes) = logit_elements.checked_mul(size_of::<f64>()) else {
        return reject_forward_equivalence_admission(format_args!(
            "assert_forward_equivalence_qwen35: retained chain-logit size overflow"
        ));
    };
    let Some(row_descriptor_bytes) = forward_cfg
        .num_probe_tokens
        .checked_mul(size_of::<Vec<f64>>())
    else {
        return reject_forward_equivalence_admission(format_args!(
            "assert_forward_equivalence_qwen35: retained chain-logit size overflow"
        ));
    };
    let Some(probe_token_bytes) = forward_cfg.num_probe_tokens.checked_mul(size_of::<u32>()) else {
        return reject_forward_equivalence_admission(format_args!(
            "assert_forward_equivalence_qwen35: retained chain-logit size overflow"
        ));
    };
    let Some(retained_bytes) = logit_bytes
        .checked_add(row_descriptor_bytes)
        .and_then(|bytes| bytes.checked_add(probe_token_bytes))
    else {
        return reject_forward_equivalence_admission(format_args!(
            "assert_forward_equivalence_qwen35: retained chain-logit size overflow"
        ));
    };
    if retained_bytes > MAX_FORWARD_EQUIVALENCE_PROBE_SNAPSHOT_BYTES {
        return reject_forward_equivalence_admission(format_args!(
            "assert_forward_equivalence_qwen35: retained chain-logit budget exceeded \
             (num_probe_tokens={}, vocab_size={}, required_bytes={retained_bytes}, \
             max_bytes={MAX_FORWARD_EQUIVALENCE_PROBE_SNAPSHOT_BYTES})",
            forward_cfg.num_probe_tokens, cfg.vocab_size
        ));
    }
    Ok(())
}

pub(crate) fn validate_forward_equivalence_admission<'cfg, 'forward>(
    cfg: &'cfg Qwen35Config,
    forward_cfg: &'forward ForwardEquivalenceConfig,
) -> Result<ForwardEquivalenceAdmission<'cfg, 'forward>, InferenceError> {
    if cfg.is_moe() {
        return reject_forward_equivalence_admission(format_args!(
            "assert_forward_equivalence_qwen35: MoE configs are deferred to v1 \
             (the rotation/fusion pipeline rejects MoE upstream; this probe \
             has no expert-mixing path)"
        ));
    }
    if forward_cfg.num_probe_tokens == 0 {
        return reject_forward_equivalence_admission(format_args!(
            "assert_forward_equivalence_qwen35: num_probe_tokens must be > 0"
        ));
    }
    if !forward_cfg.tolerance.is_finite() || forward_cfg.tolerance <= 0.0 {
        return reject_forward_equivalence_admission(format_args!(
            "assert_forward_equivalence_qwen35: tolerance must be a positive finite value, got {}",
            forward_cfg.tolerance
        ));
    }
    if cfg.vocab_size == 0 {
        return reject_forward_equivalence_admission(format_args!(
            "assert_forward_equivalence_qwen35: cfg.vocab_size must be > 0"
        ));
    }
    preflight_forward_equivalence_probe_budget(cfg, forward_cfg)?;
    Ok(ForwardEquivalenceAdmission { cfg, forward_cfg })
}

fn validate_forward_equivalence_rotation(
    cfg: &Qwen35Config,
    rotation: &RandomizedHadamard,
) -> Result<(), InferenceError> {
    if rotation.dim() != cfg.hidden_size {
        return Err(InferenceError::Inference(format!(
            "assert_forward_equivalence_qwen35: rotation.dim()={} != cfg.hidden_size={}",
            rotation.dim(),
            cfg.hidden_size
        )));
    }
    Ok(())
}

fn reject_forward_equivalence_admission<T>(
    message: std::fmt::Arguments<'_>,
) -> Result<T, InferenceError> {
    #[cfg(test)]
    pre_admission_allocation_tracking::mark_rejection();
    Err(InferenceError::Inference(message.to_string()))
}

fn full_fusion_plan(cfg: &Qwen35Config) -> Result<Vec<RmsNormFusionTarget>, InferenceError> {
    let mut fusion_plan = qwen35_per_layer_fusion_plan(cfg)?;
    fusion_plan.push(qwen35_final_norm_fusion_target());
    Ok(fusion_plan)
}

/// Capture the original-side evidence that cannot be recovered after the
/// converter mutates its working set.
///
/// This retains `num_probe_tokens * vocab_size` chain logits and one
/// `hidden_size` RMSNorm scale for each input/post-attention norm plus
/// the final norm. Retained chain-probe evidence is admitted against the
/// 64 MiB budget before any probe tokens or logits are allocated. Planned
/// matrices are not retained.
pub(crate) fn prepare_forward_equivalence_qwen35<'a>(
    original: &HashMap<String, TensorEntry>,
    cfg: &'a Qwen35Config,
    rotation: &'a RandomizedHadamard,
    forward_cfg: &ForwardEquivalenceConfig,
) -> Result<ForwardEquivalenceSnapshot<'a>, InferenceError> {
    let admission = validate_forward_equivalence_admission(cfg, forward_cfg)?;
    prepare_forward_equivalence_qwen35_after_admission(original, rotation, admission)
}

pub(crate) fn prepare_forward_equivalence_qwen35_after_admission<'cfg, 'forward>(
    original: &HashMap<String, TensorEntry>,
    rotation: &'cfg RandomizedHadamard,
    admission: ForwardEquivalenceAdmission<'cfg, 'forward>,
) -> Result<ForwardEquivalenceSnapshot<'cfg>, InferenceError> {
    let ForwardEquivalenceAdmission { cfg, forward_cfg } = admission;
    validate_forward_equivalence_rotation(cfg, rotation)?;

    let probe_tokens = deterministic_probe_tokens(
        forward_cfg.seed,
        forward_cfg.num_probe_tokens,
        cfg.vocab_size,
    );
    let original_logits = probe_tokens
        .iter()
        .map(|&token| rotation_chain_probe_qwen35(original, cfg, token))
        .collect::<Result<Vec<_>, _>>()?;

    let mut fusion_gammas = HashMap::new();
    for target in full_fusion_plan(cfg)? {
        if fusion_gammas.contains_key(&target.norm_tensor) {
            continue;
        }
        let gamma = original.get(&target.norm_tensor).ok_or_else(|| {
            InferenceError::Inference(format!(
                "prepare_forward_equivalence_qwen35: fusion gamma `{}` not in original \
                 working set",
                target.norm_tensor
            ))
        })?;
        fusion_gammas.insert(target.norm_tensor, gamma.clone());
    }

    Ok(ForwardEquivalenceSnapshot {
        cfg,
        rotation,
        probe_tokens,
        original_logits,
        fusion_gammas,
        tolerance: forward_cfg.tolerance,
    })
}

/// **Refuse-on-fail forward-equivalence gate** for QuaRot Qwen3.5 conversion.
///
/// Runs both the rotation-chain probe and the per-tensor rotation-equivalence
/// check (see module doc) on `original` and `rotated`. Returns
/// `Ok(report)` only when BOTH checks stay within `forward_cfg.tolerance`,
/// and **returns `Err(InferenceError::Inference)`** otherwise. The error
/// message names which check tripped and includes the observed max-abs
/// delta, the configured tolerance, and (for chain-probe failures) the
/// probe tokens — enough for the binary's caller to triage from logs
/// without re-running.
///
/// The converter binary (step 3c-5) MUST treat the `Err` return as a
/// hard refuse: do NOT proceed to quantization, do NOT write the output
/// `.q4` file, do NOT mutate the output `config.json`. The whole point
/// of this gate is to keep wrong-output artifacts off disk.
///
/// `rotation` MUST be the same [`RandomizedHadamard`] the caller passed
/// to [`crate::quant::quarot::pipeline::absorb_rotations`]. The
/// per-tensor check uses it to **reconstruct the expected `W_rot`**
/// from the original-side source via the same primitives the pipeline
/// uses (column-multiply by `(1 + γ)` if fusion applies, then
/// [`absorb_input_rotation_f64`] or [`absorb_output_rotation_f64`]),
/// and compares every stored element to the actual rotated matrix. A
/// mismatch between this rotation and the one absorbed into the
/// rotated working set is one of the bugs the gate explicitly catches
/// (per-tensor check will refuse).
///
/// # Errors
///
/// - `cfg.is_moe()` — MoE conversion is deferred to v1.
/// - `forward_cfg.num_probe_tokens == 0`, its retained chain-logit evidence
///   exceeds the 64 MiB budget (or its size calculation overflows), or
///   `tolerance` is not a positive finite value.
/// - `rotation.dim() != cfg.hidden_size`.
/// - Any planned tensor is missing from either working set, or has the
///   wrong shape or data length.
/// - Either the chain probe or the per-tensor check produces a non-finite
///   error or exceeds tolerance — the refuse-on-fail case. The error
///   message names which check tripped.
pub fn assert_forward_equivalence_qwen35(
    original: &HashMap<String, TensorEntry>,
    rotated: &HashMap<String, TensorEntry>,
    cfg: &Qwen35Config,
    rotation: &RandomizedHadamard,
    forward_cfg: &ForwardEquivalenceConfig,
) -> Result<ForwardEquivalenceReport, InferenceError> {
    let snapshot = prepare_forward_equivalence_qwen35(original, cfg, rotation, forward_cfg)?;
    assert_forward_equivalence_snapshot(snapshot, original, rotated)
}

/// Complete a prepared equivalence gate while streaming original planned
/// matrices from the mmap-backed checkpoint reader.
///
/// Configuration and rotation come only from `snapshot`, preserving the
/// preparation admission and dimension check through completion.
pub(crate) fn assert_prepared_forward_equivalence_qwen35(
    snapshot: ForwardEquivalenceSnapshot<'_>,
    reader: &QuarotTensorReader,
    rotated: &HashMap<String, TensorEntry>,
) -> Result<ForwardEquivalenceReport, InferenceError> {
    let original = ReaderOriginalTensorSource {
        reader,
        tied_lm_head_from_embed: snapshot.cfg.tie_word_embeddings,
    };
    assert_forward_equivalence_snapshot(snapshot, &original, rotated)
}

fn assert_forward_equivalence_snapshot<S: OriginalTensorSource>(
    snapshot: ForwardEquivalenceSnapshot<'_>,
    original: &S,
    rotated: &HashMap<String, TensorEntry>,
) -> Result<ForwardEquivalenceReport, InferenceError> {
    let ForwardEquivalenceSnapshot {
        cfg,
        rotation,
        probe_tokens,
        original_logits,
        fusion_gammas,
        tolerance,
    } = snapshot;
    if original_logits.len() != probe_tokens.len() {
        return Err(InferenceError::Inference(format!(
            "assert_forward_equivalence_qwen35: prepared probe count mismatch \
             (tokens={}, original_logits={})",
            probe_tokens.len(),
            original_logits.len()
        )));
    }

    let mut chain_max_abs = 0.0_f64;
    let mut chain_total_abs = 0.0_f64;
    let mut chain_count: usize = 0;

    for (&token, logits_orig) in probe_tokens.iter().zip(original_logits.iter()) {
        let logits_rot = rotation_chain_probe_qwen35(rotated, cfg, token)?;
        if logits_orig.len() != logits_rot.len() {
            return Err(InferenceError::Inference(format!(
                "assert_forward_equivalence_qwen35: probe logits length mismatch \
                 (original={}, rotated={}) on token {token}",
                logits_orig.len(),
                logits_rot.len()
            )));
        }
        for (logit_index, (a, b)) in logits_orig.iter().zip(logits_rot.iter()).enumerate() {
            let d = (a - b).abs();
            if !d.is_finite() {
                return Err(InferenceError::Inference(format!(
                    "forward-equivalence refused: chain probe non-finite error \
                     (token={token}, logit_index={logit_index}, original={a}, rotated={b}, \
                     abs_error={d}). Do NOT write conversion artifacts."
                )));
            }
            if d > chain_max_abs {
                chain_max_abs = d;
            }
            chain_total_abs += d;
            chain_count += 1;
        }
    }

    let chain_mean_abs = if chain_count > 0 {
        chain_total_abs / chain_count as f64
    } else {
        0.0
    };

    if chain_max_abs > tolerance {
        return Err(InferenceError::Inference(format!(
            "forward-equivalence refused: chain probe max_abs_error={chain_max_abs} \
             exceeds tolerance={tolerance} (mean_abs_error={chain_mean_abs}, \
             probe_tokens={probe_tokens:?}). \
             Do NOT write conversion artifacts — the pipeline produced logits \
             that disagree with the original model."
        )));
    }

    // 2. Per-tensor rotation-equivalence check. Covers every planned tensor,
    //    including the ones the chain probe shortcuts past (k_proj, v_proj,
    //    gate-z half of q_proj, in_proj_qkv, in_proj_a, in_proj_b).
    let rotation_plan = RotationPlan::qwen35_residual_stream_linear_layers();
    let fusion_plan = full_fusion_plan(cfg)?;
    let per_tensor_max_abs = check_per_tensor_rotation_equivalence(
        original,
        &fusion_gammas,
        rotated,
        cfg,
        rotation,
        &rotation_plan,
        &fusion_plan,
    )?;

    if per_tensor_max_abs > tolerance {
        return Err(InferenceError::Inference(format!(
            "forward-equivalence refused: per-tensor max_abs_error={per_tensor_max_abs} \
             exceeds tolerance={tolerance} (chain probe max={chain_max_abs}, \
             mean={chain_mean_abs}). \
             At least one planned tensor disagrees with the rotation/fusion algebra. \
             Do NOT write conversion artifacts."
        )));
    }

    let max_abs = chain_max_abs.max(per_tensor_max_abs);
    Ok(ForwardEquivalenceReport {
        max_abs_error: max_abs,
        mean_abs_error: chain_mean_abs,
        probe_tokens,
        tolerance,
    })
}

/// Per-tensor **matrix-equivalence** check for each [`RotationPlan`] rule.
///
/// For every config-required planned tensor (enumerated from
/// [`qwen_required_tensor_names`] plus `lm_head.weight` unconditionally
/// — the loader omits it for tied configs, but post-pipeline rotated
/// sets must have it), reconstructs the expected post-pipeline matrix
/// from the original-side source by applying the same primitives the
/// pipeline does:
///
/// - Input-side, fused: `source → fuse_shifted_rmsnorm_into_next_layer_f64 → absorb_input_rotation_f64`
/// - Input-side, no fusion (`embed_tokens` only): `source → absorb_input_rotation_f64`
/// - Output-side (`o_proj`, `out_proj`, `down_proj`): `source → absorb_output_rotation_f64`
///
/// Then compares the reconstructed matrix to `rotated[name]` element-wise
/// and tracks the max-abs delta across every planned tensor. This is a
/// **direct matrix equivalence** check — it has no nullspace (every
/// stored element of `W_rot` must match the reconstruction), unlike a
/// single-probe-vector check which could miss perturbations orthogonal
/// to the probe direction.
///
/// For tied configs where `original` lacks `lm_head.weight` (caller took
/// the snapshot before `materialize_lm_head_for_qwen35`), the
/// reconstruction uses `embed_tokens.weight` as the source for
/// `lm_head` — matching what `materialize_lm_head_for_qwen35` does
/// internally (clone, then the same fuse + rotate sequence).
///
/// # Errors
///
/// - Any expected planned tensor missing from either working set.
/// - Shape / data length mismatch on any planned tensor.
/// - Input-side tensor with `cols != hidden_size`, or output-side with
///   `rows != hidden_size` (rotation plan invariant violation).
/// - Output-side tensor unexpectedly carrying a fusion rule (rotation /
///   fusion plan inconsistency — should not happen with the Qwen3.5
///   plans).
fn check_per_tensor_rotation_equivalence<S: OriginalTensorSource>(
    original: &S,
    fusion_gammas: &HashMap<String, TensorEntry>,
    rotated: &HashMap<String, TensorEntry>,
    cfg: &Qwen35Config,
    rotation: &RandomizedHadamard,
    rotation_plan: &RotationPlan,
    fusion_plan: &[RmsNormFusionTarget],
) -> Result<f64, InferenceError> {
    let hidden = cfg.hidden_size;

    // downstream tensor name → norm tensor name (where γ lives in `original`).
    let mut fusion_gamma: HashMap<&str, &str> = HashMap::new();
    for target in fusion_plan {
        for downstream in &target.downstream_weights {
            fusion_gamma.insert(downstream.as_str(), target.norm_tensor.as_str());
        }
    }

    // Config-derived expected planned-tensor set. Filters
    // qwen_required_tensor_names through the rotation plan and appends
    // lm_head.weight unconditionally (tied configs omit it from
    // required-names but always need it post-materialize).
    let required = qwen_required_tensor_names(cfg);
    let mut expected_planned: Vec<String> = required
        .into_iter()
        .filter(|n| rotation_plan.for_tensor(n).is_some())
        .collect();
    let lm_head_name = QWEN35_LM_HEAD_NAME.to_string();
    if !expected_planned.iter().any(|n| n == &lm_head_name) {
        expected_planned.push(lm_head_name);
    }

    let mut max_abs = 0.0_f64;

    for expected_name in &expected_planned {
        let tensor_rotation = rotation_plan.for_tensor(expected_name).ok_or_else(|| {
            InferenceError::Inference(format!(
                "check_per_tensor_rotation_equivalence: expected planned tensor `{expected_name}` \
                 has no rotation rule (qwen_required_tensor_names/rotation_plan inconsistency)"
            ))
        })?;

        // Source tensor on the `original` side. Tied configs may have
        // taken the snapshot before materialize, in which case lm_head
        // falls back to embed_tokens (the source materialize would clone).
        let source_name = if original.has_tensor(expected_name) {
            expected_name.as_str()
        } else if expected_name == QWEN35_LM_HEAD_NAME && cfg.tie_word_embeddings {
            if !original.has_tensor(QWEN35_EMBED_TOKENS_NAME) {
                return Err(InferenceError::Inference(format!(
                    "check_per_tensor_rotation_equivalence: tied config requires \
                     either `{QWEN35_LM_HEAD_NAME}` or `{QWEN35_EMBED_TOKENS_NAME}` in the \
                     original tensor source as the source for lm_head reconstruction"
                )));
            }
            QWEN35_EMBED_TOKENS_NAME
        } else {
            return Err(InferenceError::Inference(format!(
                "check_per_tensor_rotation_equivalence: planned tensor `{expected_name}` \
                 missing from original tensor source"
            )));
        };
        let source = original.load_tensor(source_name).map_err(|err| {
            InferenceError::Inference(format!(
                "check_per_tensor_rotation_equivalence: failed to load original source \
                 `{source_name}` for planned tensor `{expected_name}`: {err}"
            ))
        })?;

        let actual = rotated.get(expected_name).ok_or_else(|| {
            InferenceError::Inference(format!(
                "check_per_tensor_rotation_equivalence: planned tensor `{expected_name}` \
                 missing from rotated working set"
            ))
        })?;

        if source.shape.len() != 2 || actual.shape != source.shape {
            return Err(InferenceError::Inference(format!(
                "check_per_tensor_rotation_equivalence: tensor `{expected_name}` shape mismatch \
                 (source={:?}, rotated={:?})",
                source.shape, actual.shape
            )));
        }
        let rows = source.shape[0];
        let cols = source.shape[1];
        let expected_len = rows.checked_mul(cols).ok_or_else(|| {
            InferenceError::Inference(format!(
                "check_per_tensor_rotation_equivalence: rows*cols overflow on `{expected_name}` \
                 ({rows}*{cols})"
            ))
        })?;
        if source.data.len() != expected_len || actual.data.len() != expected_len {
            return Err(InferenceError::Inference(format!(
                "check_per_tensor_rotation_equivalence: tensor `{expected_name}` data.len() mismatch \
                 (source={}, rotated={}, rows*cols={expected_len})",
                source.data.len(),
                actual.data.len()
            )));
        }

        let mut expected_rot = source.data;
        match tensor_rotation.side {
            AbsorptionSide::InputSide => {
                if cols != hidden {
                    return Err(InferenceError::Inference(format!(
                        "check_per_tensor_rotation_equivalence: input-side tensor `{expected_name}` \
                         cols={cols} != hidden={hidden} (rotation plan invariant violated)"
                    )));
                }
                if let Some(norm_name) = fusion_gamma.get(expected_name.as_str()) {
                    let norm = fusion_gammas.get(*norm_name).ok_or_else(|| {
                        InferenceError::Inference(format!(
                            "check_per_tensor_rotation_equivalence: fusion gamma source \
                             `{norm_name}` (for downstream `{expected_name}`) not in prepared \
                             original snapshot"
                        ))
                    })?;
                    if norm.shape.len() != 1 || norm.shape[0] != cols || norm.data.len() != cols {
                        return Err(InferenceError::Inference(format!(
                            "check_per_tensor_rotation_equivalence: fusion gamma `{norm_name}` \
                             shape/data mismatch (shape={:?}, data.len()={}, expected cols={cols})",
                            norm.shape,
                            norm.data.len()
                        )));
                    }
                    fuse_shifted_rmsnorm_into_next_layer_f64(
                        &mut expected_rot,
                        rows,
                        cols,
                        &norm.data,
                    )?;
                }
                absorb_input_rotation_f64(&mut expected_rot, rows, cols, rotation)?;
            }
            AbsorptionSide::OutputSide => {
                if rows != hidden {
                    return Err(InferenceError::Inference(format!(
                        "check_per_tensor_rotation_equivalence: output-side tensor `{expected_name}` \
                         rows={rows} != hidden={hidden} (rotation plan invariant violated)"
                    )));
                }
                if fusion_gamma.contains_key(expected_name.as_str()) {
                    return Err(InferenceError::Inference(format!(
                        "check_per_tensor_rotation_equivalence: output-side tensor `{expected_name}` \
                         unexpectedly has a fusion rule (rotation/fusion plan inconsistency)"
                    )));
                }
                absorb_output_rotation_f64(&mut expected_rot, rows, cols, rotation)?;
            }
        }

        let mut delta = 0.0_f64;
        for (element_index, (expected, actual)) in
            expected_rot.iter().zip(actual.data.iter()).enumerate()
        {
            let element_delta = (expected - actual).abs();
            if !element_delta.is_finite() {
                return Err(InferenceError::Inference(format!(
                    "forward-equivalence refused: per-tensor non-finite error \
                     (tensor={expected_name}, element_index={element_index}, \
                     expected={expected}, actual={actual}, abs_error={element_delta}). \
                     Do NOT write conversion artifacts."
                )));
            }
            if element_delta > delta {
                delta = element_delta;
            }
        }
        if delta > max_abs {
            max_abs = delta;
        }
    }

    Ok(max_abs)
}

fn deterministic_probe_tokens(seed: u64, n: usize, vocab_size: usize) -> Vec<u32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 32) % vocab_size as u64) as u32
        })
        .collect()
}

fn rotation_chain_probe_qwen35(
    tensors: &HashMap<String, TensorEntry>,
    cfg: &Qwen35Config,
    token: u32,
) -> Result<Vec<f64>, InferenceError> {
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let intermediate = cfg.intermediate_size;
    let full_q_dim = cfg.full_q_dim();
    let linear_output_dim = cfg.linear_output_dim();
    let eps = f64::from(cfg.rms_norm_eps);

    if (token as usize) >= vocab {
        return Err(InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: token id {token} out of range (vocab_size={vocab})"
        )));
    }

    // 1. Embed lookup.
    let embed = get_tensor_2d(tensors, QWEN35_EMBED_TOKENS_NAME, vocab, hidden)?;
    let row = (token as usize) * hidden;
    let mut h: Vec<f64> = embed.data[row..row + hidden].to_vec();

    // 2. Per-layer residual chain.
    for layer in 0..cfg.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{layer}");

        // Pre-attention norm (shifted).
        let gamma_in = get_tensor_1d(tensors, &format!("{prefix}.input_layernorm.weight"), hidden)?;
        let h_pre = rms_normalize_shifted(&h, &gamma_in.data, eps);

        // Linearized attention block.
        let attn_out = if cfg.is_full_attention(layer) {
            let q_proj = get_tensor_2d(
                tensors,
                &format!("{prefix}.self_attn.q_proj.weight"),
                2 * full_q_dim,
                hidden,
            )?;
            let o_proj = get_tensor_2d(
                tensors,
                &format!("{prefix}.self_attn.o_proj.weight"),
                hidden,
                full_q_dim,
            )?;
            let q_full = matvec_f64(&q_proj.data, 2 * full_q_dim, hidden, &h_pre);
            matvec_f64(&o_proj.data, hidden, full_q_dim, &q_full[..full_q_dim])
        } else {
            let in_proj_z = get_tensor_2d(
                tensors,
                &format!("{prefix}.linear_attn.in_proj_z.weight"),
                linear_output_dim,
                hidden,
            )?;
            let out_proj = get_tensor_2d(
                tensors,
                &format!("{prefix}.linear_attn.out_proj.weight"),
                hidden,
                linear_output_dim,
            )?;
            let z = matvec_f64(&in_proj_z.data, linear_output_dim, hidden, &h_pre);
            matvec_f64(&out_proj.data, hidden, linear_output_dim, &z)
        };
        add_in_place(&mut h, &attn_out);

        // Post-attention norm (shifted).
        let gamma_post = get_tensor_1d(
            tensors,
            &format!("{prefix}.post_attention_layernorm.weight"),
            hidden,
        )?;
        let h_pre_mlp = rms_normalize_shifted(&h, &gamma_post.data, eps);

        // Linearized MLP block: `gate + up` replaces `silu(gate) ⊙ up`.
        let gate_proj = get_tensor_2d(
            tensors,
            &format!("{prefix}.mlp.gate_proj.weight"),
            intermediate,
            hidden,
        )?;
        let up_proj = get_tensor_2d(
            tensors,
            &format!("{prefix}.mlp.up_proj.weight"),
            intermediate,
            hidden,
        )?;
        let down_proj = get_tensor_2d(
            tensors,
            &format!("{prefix}.mlp.down_proj.weight"),
            hidden,
            intermediate,
        )?;
        let gate = matvec_f64(&gate_proj.data, intermediate, hidden, &h_pre_mlp);
        let up = matvec_f64(&up_proj.data, intermediate, hidden, &h_pre_mlp);
        let mid: Vec<f64> = gate.iter().zip(up.iter()).map(|(a, b)| a + b).collect();
        let mlp_out = matvec_f64(&down_proj.data, hidden, intermediate, &mid);
        add_in_place(&mut h, &mlp_out);
    }

    // 3. Final norm (shifted).
    let gamma_final = get_tensor_1d(tensors, QWEN35_FINAL_NORM_NAME, hidden)?;
    let h_final = rms_normalize_shifted(&h, &gamma_final.data, eps);

    // 4. lm_head matvec — fall back to embed_tokens for tied originals.
    let lm_tensor = if tensors.contains_key(QWEN35_LM_HEAD_NAME) {
        get_tensor_2d(tensors, QWEN35_LM_HEAD_NAME, vocab, hidden)?
    } else if cfg.tie_word_embeddings {
        embed
    } else {
        return Err(InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: untied config requires `{QWEN35_LM_HEAD_NAME}` \
             in the working set (config says `tie_word_embeddings=false` and no fallback \
             is valid in that case)"
        )));
    };
    Ok(matvec_f64(&lm_tensor.data, vocab, hidden, &h_final))
}

fn get_tensor_2d<'a>(
    tensors: &'a HashMap<String, TensorEntry>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<&'a TensorEntry, InferenceError> {
    let t = tensors.get(name).ok_or_else(|| {
        InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: tensor `{name}` not in working set"
        ))
    })?;
    if t.shape.len() != 2 || t.shape[0] != rows || t.shape[1] != cols {
        return Err(InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: tensor `{name}` shape {:?} != expected [{rows}, {cols}]",
            t.shape
        )));
    }
    let expected = rows.checked_mul(cols).ok_or_else(|| {
        InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: rows*cols overflow on `{name}` ({rows}*{cols})"
        ))
    })?;
    if t.data.len() != expected {
        return Err(InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: tensor `{name}` data.len()={} != rows*cols {expected}",
            t.data.len()
        )));
    }
    Ok(t)
}

fn get_tensor_1d<'a>(
    tensors: &'a HashMap<String, TensorEntry>,
    name: &str,
    len: usize,
) -> Result<&'a TensorEntry, InferenceError> {
    let t = tensors.get(name).ok_or_else(|| {
        InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: tensor `{name}` not in working set"
        ))
    })?;
    if t.shape.len() != 1 || t.shape[0] != len {
        return Err(InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: tensor `{name}` shape {:?} != expected [{len}]",
            t.shape
        )));
    }
    if t.data.len() != len {
        return Err(InferenceError::Inference(format!(
            "rotation_chain_probe_qwen35: tensor `{name}` data.len()={} != expected {len}",
            t.data.len()
        )));
    }
    Ok(t)
}

fn rms_normalize_shifted(h: &[f64], gamma: &[f64], eps: f64) -> Vec<f64> {
    debug_assert_eq!(h.len(), gamma.len());
    let n = h.len();
    let sum_sq: f64 = h.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f64 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    h.iter()
        .zip(gamma.iter())
        .map(|(v, g)| v * inv_rms * (1.0 + g))
        .collect()
}

fn matvec_f64(w: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(w.len(), rows * cols);
    let mut y = vec![0.0_f64; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        y[r] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    }
    y
}

fn add_in_place(h: &mut [f64], addend: &[f64]) {
    debug_assert_eq!(h.len(), addend.len());
    for (a, b) in h.iter_mut().zip(addend.iter()) {
        *a += b;
    }
}

#[cfg(test)]
mod tests;
