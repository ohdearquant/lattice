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
pub(crate) mod pre_admission_allocation_tracking {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Phase {
        Inactive,
        WaitingForConverterBoundary,
        BeforeRejection,
        AfterRejection,
        WaitingForMaterializedWorkingSetBoundary,
        MaterializedWorkingSetBoundary,
        MaterializedWorkingSetBoundaryCompleted,
    }

    #[derive(Clone, Copy)]
    struct State {
        phase: Phase,
        before_rejection_allocation_calls: usize,
        after_rejection_allocation_calls: usize,
        materialized_working_set_allocation_calls: usize,
        rejection_seen: bool,
    }

    const INACTIVE: State = State {
        phase: Phase::Inactive,
        before_rejection_allocation_calls: 0,
        after_rejection_allocation_calls: 0,
        materialized_working_set_allocation_calls: 0,
        rejection_seen: false,
    };

    std::thread_local! {
        static STATE: Cell<State> = const { Cell::new(INACTIVE) };
    }

    struct TrackingAllocator;

    #[global_allocator]
    static GLOBAL: TrackingAllocator = TrackingAllocator;

    fn record_allocation() {
        let _ = STATE.try_with(|cell| {
            let mut state = cell.get();
            match state.phase {
                Phase::BeforeRejection => {
                    state.before_rejection_allocation_calls =
                        state.before_rejection_allocation_calls.saturating_add(1);
                }
                Phase::AfterRejection => {
                    state.after_rejection_allocation_calls =
                        state.after_rejection_allocation_calls.saturating_add(1);
                }
                Phase::MaterializedWorkingSetBoundary => {
                    state.materialized_working_set_allocation_calls = state
                        .materialized_working_set_allocation_calls
                        .saturating_add(1);
                }
                Phase::Inactive
                | Phase::WaitingForConverterBoundary
                | Phase::WaitingForMaterializedWorkingSetBoundary
                | Phase::MaterializedWorkingSetBoundaryCompleted => return,
            }
            cell.set(state);
        });
    }

    // SAFETY: allocation requests are observed without modification and then
    // forwarded unchanged to the system allocator.
    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record_allocation();
            System.alloc(layout)
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record_allocation();
            System.alloc_zeroed(layout)
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            record_allocation();
            System.realloc(ptr, layout, new_size)
        }
    }

    pub(crate) struct Guard {
        active: bool,
    }

    pub(crate) struct Observation {
        pub(crate) before_rejection_allocation_calls: usize,
        pub(crate) after_rejection_allocation_calls: usize,
        pub(crate) materialized_working_set_allocation_calls: usize,
        pub(crate) rejection_seen: bool,
        pub(crate) materialized_working_set_boundary_seen: bool,
        pub(crate) materialized_working_set_boundary_completed: bool,
    }

    fn start_in_phase(phase: Phase) -> Guard {
        STATE.with(|cell| {
            assert_eq!(
                cell.get().phase,
                Phase::Inactive,
                "allocation tracking already active"
            );
            cell.set(State { phase, ..INACTIVE });
        });
        Guard { active: true }
    }

    pub(super) fn start() -> Guard {
        start_in_phase(Phase::BeforeRejection)
    }

    pub(crate) fn start_at_converter_boundary() -> Guard {
        start_in_phase(Phase::WaitingForConverterBoundary)
    }

    pub(crate) fn start_at_materialized_working_set_boundary() -> Guard {
        start_in_phase(Phase::WaitingForMaterializedWorkingSetBoundary)
    }

    pub(crate) fn mark_converter_boundary() {
        let _ = STATE.try_with(|cell| {
            let mut state = cell.get();
            if state.phase == Phase::WaitingForConverterBoundary {
                state.phase = Phase::BeforeRejection;
                cell.set(state);
            }
        });
    }

    pub(crate) fn mark_materialized_working_set_boundary() {
        let _ = STATE.try_with(|cell| {
            let mut state = cell.get();
            if state.phase == Phase::WaitingForMaterializedWorkingSetBoundary {
                state.phase = Phase::MaterializedWorkingSetBoundary;
                cell.set(state);
            }
        });
    }

    pub(crate) fn mark_materialized_working_set_boundary_completed() {
        let _ = STATE.try_with(|cell| {
            let mut state = cell.get();
            if state.phase == Phase::MaterializedWorkingSetBoundary {
                state.phase = Phase::MaterializedWorkingSetBoundaryCompleted;
                cell.set(state);
            }
        });
    }

    pub(super) fn mark_rejection() {
        let _ = STATE.try_with(|cell| {
            let mut state = cell.get();
            if state.phase == Phase::BeforeRejection {
                state.phase = Phase::AfterRejection;
                state.rejection_seen = true;
                cell.set(state);
            }
        });
    }

    impl Guard {
        pub(crate) fn finish(mut self) -> Observation {
            self.active = false;
            STATE.with(|cell| {
                let state = cell.replace(INACTIVE);
                Observation {
                    before_rejection_allocation_calls: state.before_rejection_allocation_calls,
                    after_rejection_allocation_calls: state.after_rejection_allocation_calls,
                    materialized_working_set_allocation_calls: state
                        .materialized_working_set_allocation_calls,
                    rejection_seen: state.rejection_seen,
                    materialized_working_set_boundary_seen: matches!(
                        state.phase,
                        Phase::MaterializedWorkingSetBoundary
                            | Phase::MaterializedWorkingSetBoundaryCompleted
                    ),
                    materialized_working_set_boundary_completed: state.phase
                        == Phase::MaterializedWorkingSetBoundaryCompleted,
                }
            })
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            if self.active {
                let _ = STATE.try_with(|cell| cell.set(INACTIVE));
            }
        }
    }
}

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
pub(crate) struct ForwardEquivalenceSnapshot {
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
    let logit_elements = forward_cfg
        .num_probe_tokens
        .checked_mul(cfg.vocab_size)
        .ok_or_else(forward_equivalence_logit_size_overflow)?;
    let logit_bytes = logit_elements
        .checked_mul(size_of::<f64>())
        .ok_or_else(forward_equivalence_logit_size_overflow)?;
    let row_descriptor_bytes = forward_cfg
        .num_probe_tokens
        .checked_mul(size_of::<Vec<f64>>())
        .ok_or_else(forward_equivalence_logit_size_overflow)?;
    let probe_token_bytes = forward_cfg
        .num_probe_tokens
        .checked_mul(size_of::<u32>())
        .ok_or_else(forward_equivalence_logit_size_overflow)?;
    let retained_bytes = logit_bytes
        .checked_add(row_descriptor_bytes)
        .and_then(|bytes| bytes.checked_add(probe_token_bytes))
        .ok_or_else(forward_equivalence_logit_size_overflow)?;
    if retained_bytes > MAX_FORWARD_EQUIVALENCE_PROBE_SNAPSHOT_BYTES {
        #[cfg(test)]
        pre_admission_allocation_tracking::mark_rejection();
        return Err(InferenceError::Inference(format!(
            "assert_forward_equivalence_qwen35: retained chain-logit budget exceeded \
             (num_probe_tokens={}, vocab_size={}, required_bytes={retained_bytes}, \
             max_bytes={MAX_FORWARD_EQUIVALENCE_PROBE_SNAPSHOT_BYTES})",
            forward_cfg.num_probe_tokens, cfg.vocab_size
        )));
    }
    Ok(())
}

fn validate_forward_equivalence_inputs(
    cfg: &Qwen35Config,
    rotation: &RandomizedHadamard,
    forward_cfg: &ForwardEquivalenceConfig,
) -> Result<(), InferenceError> {
    if cfg.is_moe() {
        return Err(InferenceError::Inference(
            "assert_forward_equivalence_qwen35: MoE configs are deferred to v1 \
             (the rotation/fusion pipeline rejects MoE upstream; this probe \
             has no expert-mixing path)"
                .to_string(),
        ));
    }
    if forward_cfg.num_probe_tokens == 0 {
        return Err(InferenceError::Inference(
            "assert_forward_equivalence_qwen35: num_probe_tokens must be > 0".to_string(),
        ));
    }
    if !forward_cfg.tolerance.is_finite() || forward_cfg.tolerance <= 0.0 {
        return Err(InferenceError::Inference(format!(
            "assert_forward_equivalence_qwen35: tolerance must be a positive finite value, got {}",
            forward_cfg.tolerance
        )));
    }
    if cfg.vocab_size == 0 {
        return Err(InferenceError::Inference(
            "assert_forward_equivalence_qwen35: cfg.vocab_size must be > 0".to_string(),
        ));
    }
    preflight_forward_equivalence_probe_budget(cfg, forward_cfg)?;
    if rotation.dim() != cfg.hidden_size {
        return Err(InferenceError::Inference(format!(
            "assert_forward_equivalence_qwen35: rotation.dim()={} != cfg.hidden_size={}",
            rotation.dim(),
            cfg.hidden_size
        )));
    }
    Ok(())
}

fn forward_equivalence_logit_size_overflow() -> InferenceError {
    InferenceError::Inference(
        "assert_forward_equivalence_qwen35: retained chain-logit size overflow".to_string(),
    )
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
pub(crate) fn prepare_forward_equivalence_qwen35(
    original: &HashMap<String, TensorEntry>,
    cfg: &Qwen35Config,
    rotation: &RandomizedHadamard,
    forward_cfg: &ForwardEquivalenceConfig,
) -> Result<ForwardEquivalenceSnapshot, InferenceError> {
    validate_forward_equivalence_inputs(cfg, rotation, forward_cfg)?;

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
    assert_forward_equivalence_snapshot(snapshot, original, rotated, cfg, rotation)
}

/// Complete a prepared equivalence gate while streaming original planned
/// matrices from the mmap-backed checkpoint reader.
pub(crate) fn assert_prepared_forward_equivalence_qwen35(
    snapshot: ForwardEquivalenceSnapshot,
    reader: &QuarotTensorReader,
    rotated: &HashMap<String, TensorEntry>,
    cfg: &Qwen35Config,
    rotation: &RandomizedHadamard,
) -> Result<ForwardEquivalenceReport, InferenceError> {
    let original = ReaderOriginalTensorSource {
        reader,
        tied_lm_head_from_embed: cfg.tie_word_embeddings,
    };
    assert_forward_equivalence_snapshot(snapshot, &original, rotated, cfg, rotation)
}

fn assert_forward_equivalence_snapshot<S: OriginalTensorSource>(
    snapshot: ForwardEquivalenceSnapshot,
    original: &S,
    rotated: &HashMap<String, TensorEntry>,
    cfg: &Qwen35Config,
    rotation: &RandomizedHadamard,
) -> Result<ForwardEquivalenceReport, InferenceError> {
    let ForwardEquivalenceSnapshot {
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
mod tests {
    use super::*;
    use crate::model::qwen35_config::{LayerType, Qwen35Config, compute_layer_types};
    use crate::quant::quarot::hadamard::RandomizedHadamard;
    use crate::quant::quarot::lm_head::{
        materialize_lm_head_for_qwen35, qwen35_final_norm_fusion_target,
    };
    use crate::quant::quarot::pipeline::{absorb_rotations, fuse_rmsnorms};
    use crate::quant::quarot::plan::RotationPlan;
    use crate::quant::quarot::rmsnorm_fusion::qwen35_per_layer_fusion_plan;

    /// Tiny Qwen3.5 config tuned for tractable probe tests:
    /// - `hidden_size = 8` (power of 2 for Hadamard)
    /// - 2 layers, one GDN + one GQA so the probe exercises both branches
    /// - small intermediate/vocab so matvecs stay sub-millisecond
    fn tiny_test_cfg() -> Qwen35Config {
        let mut cfg = Qwen35Config::qwen35_0_8b();
        cfg.hidden_size = 8;
        cfg.num_hidden_layers = 2;
        cfg.vocab_size = 4;
        cfg.intermediate_size = 16;
        cfg.num_attention_heads = 2;
        cfg.num_key_value_heads = 1;
        cfg.head_dim = 4;
        cfg.linear_num_key_heads = 1;
        cfg.linear_key_head_dim = 2;
        cfg.linear_value_head_dim = 2;
        cfg.linear_num_value_heads = Some(1);
        cfg.full_attention_interval = 2;
        cfg.layer_types = compute_layer_types(cfg.num_hidden_layers, cfg.full_attention_interval);
        cfg.layer_mask = vec![true; cfg.num_hidden_layers];
        cfg.tie_word_embeddings = false;
        cfg.rms_norm_eps = 1e-6;
        cfg
    }

    fn tied_tiny_test_cfg() -> Qwen35Config {
        let mut cfg = tiny_test_cfg();
        cfg.tie_word_embeddings = true;
        cfg
    }

    fn synthetic_f64(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (state >> 11) as u32;
                (bits as f64 / u32::MAX as f64) - 0.5
            })
            .collect()
    }

    fn insert_tensor(
        tensors: &mut HashMap<String, TensorEntry>,
        name: &str,
        shape: Vec<usize>,
        data: Vec<f64>,
    ) {
        tensors.insert(
            name.to_string(),
            TensorEntry {
                name: name.to_string(),
                shape,
                data,
            },
        );
    }

    /// Build a working set with shapes consistent with `cfg`'s loader
    /// expectations. Includes every tensor the fusion plan and rotation
    /// plan reference, plus `embed_tokens` and `final_norm`. Does NOT
    /// include `lm_head.weight` — the caller adds it separately (or via
    /// `materialize_lm_head_for_qwen35` for tied configs).
    fn build_working_set(cfg: &Qwen35Config, seed: u64) -> HashMap<String, TensorEntry> {
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let intermediate = cfg.intermediate_size;
        let full_q_dim = cfg.full_q_dim();
        let full_kv_dim = cfg.full_kv_dim();
        let linear_qkv_dim = cfg.linear_qkv_dim();
        let linear_output_dim = cfg.linear_output_dim();
        let linear_num_heads = cfg.linear_num_key_heads;

        let mut tensors = HashMap::new();
        insert_tensor(
            &mut tensors,
            QWEN35_EMBED_TOKENS_NAME,
            vec![vocab, hidden],
            synthetic_f64(vocab * hidden, seed.wrapping_add(1)),
        );
        insert_tensor(
            &mut tensors,
            QWEN35_FINAL_NORM_NAME,
            vec![hidden],
            synthetic_f64(hidden, seed.wrapping_add(2)),
        );

        for i in 0..cfg.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{i}");
            let layer_seed = seed.wrapping_add(100 + i as u64);

            insert_tensor(
                &mut tensors,
                &format!("{prefix}.input_layernorm.weight"),
                vec![hidden],
                synthetic_f64(hidden, layer_seed.wrapping_add(1)),
            );
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
                vec![hidden],
                synthetic_f64(hidden, layer_seed.wrapping_add(2)),
            );

            if cfg.is_full_attention(i) {
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    vec![2 * full_q_dim, hidden],
                    synthetic_f64(2 * full_q_dim * hidden, layer_seed.wrapping_add(10)),
                );
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    vec![full_kv_dim, hidden],
                    synthetic_f64(full_kv_dim * hidden, layer_seed.wrapping_add(11)),
                );
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    vec![full_kv_dim, hidden],
                    synthetic_f64(full_kv_dim * hidden, layer_seed.wrapping_add(12)),
                );
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    vec![hidden, full_q_dim],
                    synthetic_f64(hidden * full_q_dim, layer_seed.wrapping_add(13)),
                );
            } else {
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    vec![linear_qkv_dim, hidden],
                    synthetic_f64(linear_qkv_dim * hidden, layer_seed.wrapping_add(20)),
                );
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.linear_attn.in_proj_z.weight"),
                    vec![linear_output_dim, hidden],
                    synthetic_f64(linear_output_dim * hidden, layer_seed.wrapping_add(21)),
                );
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.linear_attn.in_proj_a.weight"),
                    vec![linear_num_heads, hidden],
                    synthetic_f64(linear_num_heads * hidden, layer_seed.wrapping_add(22)),
                );
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.linear_attn.in_proj_b.weight"),
                    vec![linear_num_heads, hidden],
                    synthetic_f64(linear_num_heads * hidden, layer_seed.wrapping_add(23)),
                );
                insert_tensor(
                    &mut tensors,
                    &format!("{prefix}.linear_attn.out_proj.weight"),
                    vec![hidden, linear_output_dim],
                    synthetic_f64(hidden * linear_output_dim, layer_seed.wrapping_add(24)),
                );
            }

            insert_tensor(
                &mut tensors,
                &format!("{prefix}.mlp.gate_proj.weight"),
                vec![intermediate, hidden],
                synthetic_f64(intermediate * hidden, layer_seed.wrapping_add(30)),
            );
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.mlp.up_proj.weight"),
                vec![intermediate, hidden],
                synthetic_f64(intermediate * hidden, layer_seed.wrapping_add(31)),
            );
            insert_tensor(
                &mut tensors,
                &format!("{prefix}.mlp.down_proj.weight"),
                vec![hidden, intermediate],
                synthetic_f64(hidden * intermediate, layer_seed.wrapping_add(32)),
            );
        }

        tensors
    }

    fn full_pipeline_plans(
        cfg: &Qwen35Config,
    ) -> (
        Vec<crate::quant::quarot::rmsnorm_fusion::RmsNormFusionTarget>,
        RotationPlan,
    ) {
        let mut fusion = qwen35_per_layer_fusion_plan(cfg).unwrap();
        fusion.push(qwen35_final_norm_fusion_target());
        (fusion, RotationPlan::qwen35_residual_stream_linear_layers())
    }

    /// Sanity: the tiny test config really has one GDN + one GQA layer.
    #[test]
    fn tiny_cfg_has_one_full_and_one_linear_layer() {
        let cfg = tiny_test_cfg();
        assert_eq!(cfg.num_hidden_layers, 2);
        assert_eq!(cfg.layer_types[0], LayerType::LinearAttention);
        assert_eq!(cfg.layer_types[1], LayerType::FullAttention);
    }

    #[test]
    fn prepared_snapshot_enforces_configured_logit_budget_before_tensor_access() {
        let cfg = Qwen35Config::qwen35_0_8b();
        let original = HashMap::new();
        let rotation = RandomizedHadamard::new(0x51A5_EEED, cfg.hidden_size).unwrap();
        let bytes_per_probe =
            cfg.vocab_size * size_of::<f64>() + size_of::<Vec<f64>>() + size_of::<u32>();
        let max_probe_tokens = MAX_FORWARD_EQUIVALENCE_PROBE_SNAPSHOT_BYTES / bytes_per_probe;
        assert_eq!(max_probe_tokens, 33);
        assert!(max_probe_tokens * bytes_per_probe <= MAX_FORWARD_EQUIVALENCE_PROBE_SNAPSHOT_BYTES);
        assert!(
            (max_probe_tokens + 1) * bytes_per_probe > MAX_FORWARD_EQUIVALENCE_PROBE_SNAPSHOT_BYTES
        );

        let at_limit_cfg = ForwardEquivalenceConfig {
            num_probe_tokens: max_probe_tokens,
            ..Default::default()
        };
        let at_limit =
            prepare_forward_equivalence_qwen35(&original, &cfg, &rotation, &at_limit_cfg)
                .err()
                .expect("the empty fixture must fail at tensor access")
                .to_string();
        assert!(
            at_limit.contains(QWEN35_EMBED_TOKENS_NAME),
            "the largest in-budget probe count must reach tensor access: {at_limit}"
        );

        let over_limit_cfg = ForwardEquivalenceConfig {
            num_probe_tokens: max_probe_tokens + 1,
            ..Default::default()
        };
        let over_limit =
            prepare_forward_equivalence_qwen35(&original, &cfg, &rotation, &over_limit_cfg)
                .err()
                .expect("the over-budget request must fail admission")
                .to_string();
        assert!(
            over_limit.contains("retained chain-logit budget"),
            "one probe past the configured limit must be rejected before tensor access: \
             {over_limit}"
        );
        assert!(
            !over_limit.contains(QWEN35_EMBED_TOKENS_NAME),
            "budget rejection must precede tensor access: {over_limit}"
        );
    }

    #[test]
    fn prepared_snapshot_rejects_over_budget_before_any_owned_allocation() {
        let cfg = Qwen35Config::qwen35_0_8b();
        let original = HashMap::new();
        let rotation = RandomizedHadamard::new(0x51A5_EEED, cfg.hidden_size).unwrap();
        let bytes_per_probe =
            cfg.vocab_size * size_of::<f64>() + size_of::<Vec<f64>>() + size_of::<u32>();
        let max_probe_tokens = MAX_FORWARD_EQUIVALENCE_PROBE_SNAPSHOT_BYTES / bytes_per_probe;
        let forward_cfg = ForwardEquivalenceConfig {
            num_probe_tokens: max_probe_tokens + 1,
            ..Default::default()
        };

        let tracking = pre_admission_allocation_tracking::start();
        let result = prepare_forward_equivalence_qwen35(&original, &cfg, &rotation, &forward_cfg);
        let observation = tracking.finish();

        assert!(
            observation.rejection_seen,
            "the measured call must reach budget rejection"
        );
        assert_eq!(
            observation.before_rejection_allocation_calls, 0,
            "over-budget preparation allocated before rejecting"
        );
        assert!(
            observation.after_rejection_allocation_calls > 0,
            "the diagnostic allocation after budget rejection must be observed"
        );
        let error = result
            .err()
            .expect("the over-budget request must fail admission")
            .to_string();
        assert!(
            error.contains("retained chain-logit budget"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn prepared_snapshot_rejects_logit_budget_arithmetic_overflow() {
        let original = HashMap::new();
        for (num_probe_tokens, vocab_size) in [(2, usize::MAX / 2 + 1), (1, usize::MAX / 8 + 1)] {
            let mut cfg = tied_tiny_test_cfg();
            cfg.vocab_size = vocab_size;
            let rotation = RandomizedHadamard::new(0x51A5_EEED, cfg.hidden_size).unwrap();
            let forward_cfg = ForwardEquivalenceConfig {
                num_probe_tokens,
                ..Default::default()
            };
            let err = prepare_forward_equivalence_qwen35(&original, &cfg, &rotation, &forward_cfg)
                .err()
                .expect("overflowing budget arithmetic must fail admission")
                .to_string();
            assert!(
                err.contains("retained chain-logit size overflow"),
                "overflowing budget arithmetic must return an error: {err}"
            );
        }
    }

    /// Happy path (untied): build → materialize is a no-op (already untied
    /// with lm_head present) → fuse → rotate → harness must pass.
    #[test]
    fn forward_equivalence_passes_on_full_untied_pipeline() {
        let cfg = tiny_test_cfg();
        let mut original = build_working_set(&cfg, 1);
        // Untied: lm_head must be present pre-pipeline.
        insert_tensor(
            &mut original,
            QWEN35_LM_HEAD_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            synthetic_f64(cfg.vocab_size * cfg.hidden_size, 999),
        );
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        let (fusion, rot_plan) = full_pipeline_plans(&cfg);
        fuse_rmsnorms(&mut rotated, &fusion).unwrap();
        let rotation = RandomizedHadamard::new(0xC0FFEE, cfg.hidden_size).unwrap();
        absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

        let report = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap();
        assert!(report.max_abs_error < 1e-5, "unexpected delta: {report:?}");
        assert_eq!(report.probe_tokens.len(), 4);
        for &t in &report.probe_tokens {
            assert!((t as usize) < cfg.vocab_size);
        }
    }

    /// Happy path (tied): pre-pipeline lm_head is absent (probe falls
    /// back to embed_tokens); rotated set materializes + fuses + rotates.
    #[test]
    fn forward_equivalence_passes_on_full_tied_pipeline() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 2);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        let (fusion, rot_plan) = full_pipeline_plans(&cfg);
        fuse_rmsnorms(&mut rotated, &fusion).unwrap();
        let rotation = RandomizedHadamard::new(0xFEED_FACE, cfg.hidden_size).unwrap();
        absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

        let report = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap();
        assert!(report.max_abs_error < 1e-5, "unexpected delta: {report:?}");
    }

    /// Refuse: rotate without fusing the final_norm target. The mismatch
    /// `lm_head_rot · rmsnorm(R·h)` vs `embed_tokens · diag(1+γ_final) · rmsnorm(h)`
    /// produces a non-trivial delta and the gate must refuse.
    #[test]
    fn forward_equivalence_refuses_when_final_norm_fusion_skipped() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 4);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        // BUG: drop the final_norm fusion target — fuse only per-layer norms.
        let per_layer = qwen35_per_layer_fusion_plan(&cfg).unwrap();
        fuse_rmsnorms(&mut rotated, &per_layer).unwrap();
        let rotation = RandomizedHadamard::new(0xBADC0DE, cfg.hidden_size).unwrap();
        absorb_rotations(
            &mut rotated,
            &RotationPlan::qwen35_residual_stream_linear_layers(),
            &rotation,
        )
        .unwrap();

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("forward-equivalence refused"),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("exceeds tolerance"), "unexpected error: {msg}");
    }

    /// Refuse: a single tensor in the rotated set is silently corrupted
    /// after a correct pipeline run. Real-world analogue: a converter bug
    /// that re-quantizes a tensor a second time, or scales the wrong
    /// matrix. The harness must catch this even though the rotation
    /// algebra itself was applied correctly to every other tensor.
    #[test]
    fn forward_equivalence_refuses_on_corrupted_rotated_tensor() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 5);
        let mut rotated = original.clone();

        // Correct pipeline.
        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        let (fusion, rot_plan) = full_pipeline_plans(&cfg);
        fuse_rmsnorms(&mut rotated, &fusion).unwrap();
        let rotation = RandomizedHadamard::new(0x55AA_55AA, cfg.hidden_size).unwrap();
        absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

        // Corrupt one matrix after the pipeline.
        let lm = rotated
            .get_mut(QWEN35_LM_HEAD_NAME)
            .expect("lm_head should exist after materialize");
        for v in lm.data.iter_mut() {
            *v *= 1.25;
        }

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("forward-equivalence refused"),
            "unexpected error: {msg}"
        );
    }

    /// The poisoned rotated norm is consumed only by the chain probe, so
    /// removing the chain's non-finite guard lets this conversion pass.
    #[test]
    fn forward_equivalence_refuses_non_finite_chain_probe_error() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 32);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        let (fusion, rot_plan) = full_pipeline_plans(&cfg);
        fuse_rmsnorms(&mut rotated, &fusion).unwrap();
        let rotation = RandomizedHadamard::new(0xCA11_0BAD, cfg.hidden_size).unwrap();
        absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

        rotated.get_mut(QWEN35_FINAL_NORM_NAME).unwrap().data[0] = f64::NAN;

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("chain probe non-finite error"),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("abs_error=NaN"), "unexpected error: {msg}");
    }

    /// `k_proj` is excluded from the chain probe, so removing the
    /// per-tensor non-finite guard lets this conversion pass.
    #[test]
    fn forward_equivalence_refuses_non_finite_per_tensor_error() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 33);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        let (fusion, rot_plan) = full_pipeline_plans(&cfg);
        fuse_rmsnorms(&mut rotated, &fusion).unwrap();
        let rotation = RandomizedHadamard::new(0xFACE_FEED, cfg.hidden_size).unwrap();
        absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

        let k_name = "model.language_model.layers.1.self_attn.k_proj.weight";
        rotated.get_mut(k_name).unwrap().data[0] = f64::NAN;

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("per-tensor non-finite error"),
            "unexpected error: {msg}"
        );
        assert!(msg.contains(k_name), "unexpected error: {msg}");
        assert!(msg.contains("abs_error=NaN"), "unexpected error: {msg}");
    }

    /// Refuse: rotate without fusing per-layer norms first. Rotation
    /// absorbed AROUND a non-zero γ_in breaks the residual stream
    /// because `diag(1+γ) · R^T ≠ R^T · diag(1+γ)`.
    #[test]
    fn forward_equivalence_refuses_when_per_layer_fusion_skipped() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 6);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        // BUG: only fuse final_norm, skip per-layer.
        let final_only = vec![qwen35_final_norm_fusion_target()];
        fuse_rmsnorms(&mut rotated, &final_only).unwrap();
        let rotation = RandomizedHadamard::new(0xABCDEF, cfg.hidden_size).unwrap();
        absorb_rotations(
            &mut rotated,
            &RotationPlan::qwen35_residual_stream_linear_layers(),
            &rotation,
        )
        .unwrap();

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("forward-equivalence refused"),
            "unexpected error: {msg}"
        );
    }

    /// Untied original missing lm_head → harness errors with a useful
    /// message (vs. the silent fall-through that would happen for tied).
    #[test]
    fn forward_equivalence_errors_on_untied_original_missing_lm_head() {
        let cfg = tiny_test_cfg(); // untied
        let original = build_working_set(&cfg, 7); // no lm_head added
        let mut rotated = original.clone();
        insert_tensor(
            &mut rotated,
            QWEN35_LM_HEAD_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            synthetic_f64(cfg.vocab_size * cfg.hidden_size, 7777),
        );
        let rotation = RandomizedHadamard::new(1, cfg.hidden_size).unwrap();

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains(QWEN35_LM_HEAD_NAME), "unexpected error: {msg}");
        assert!(msg.contains("untied"), "unexpected error: {msg}");
    }

    #[test]
    fn forward_equivalence_errors_on_missing_required_tensor() {
        let cfg = tiny_test_cfg();
        let mut original = build_working_set(&cfg, 8);
        insert_tensor(
            &mut original,
            QWEN35_LM_HEAD_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            synthetic_f64(cfg.vocab_size * cfg.hidden_size, 888),
        );
        let rotated = original.clone();

        // Remove a required tensor from `original` to simulate a partial load.
        original.remove(QWEN35_FINAL_NORM_NAME);
        let rotation = RandomizedHadamard::new(2, cfg.hidden_size).unwrap();

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(QWEN35_FINAL_NORM_NAME),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains("not in working set"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn forward_equivalence_errors_on_shape_mismatch() {
        let cfg = tiny_test_cfg();
        let mut original = build_working_set(&cfg, 9);
        // Inject a wrong-shape embed_tokens.
        original.insert(
            QWEN35_EMBED_TOKENS_NAME.to_string(),
            TensorEntry {
                name: QWEN35_EMBED_TOKENS_NAME.to_string(),
                shape: vec![cfg.vocab_size, cfg.hidden_size + 1],
                data: vec![0.0; cfg.vocab_size * (cfg.hidden_size + 1)],
            },
        );
        insert_tensor(
            &mut original,
            QWEN35_LM_HEAD_NAME,
            vec![cfg.vocab_size, cfg.hidden_size],
            synthetic_f64(cfg.vocab_size * cfg.hidden_size, 999),
        );
        let rotated = original.clone();
        let rotation = RandomizedHadamard::new(3, cfg.hidden_size).unwrap();

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(QWEN35_EMBED_TOKENS_NAME),
            "unexpected error: {msg}"
        );
        assert!(msg.contains("shape"), "unexpected error: {msg}");
    }

    #[test]
    fn forward_equivalence_rejects_moe_config() {
        let cfg = Qwen35Config::qwen36_35b_a3b();
        assert!(cfg.is_moe());
        let original = HashMap::new();
        let rotated = HashMap::new();
        // MoE check returns before rotation.dim() is consulted; any
        // power-of-2-dim rotation suffices for the parameter slot.
        let rotation = RandomizedHadamard::new(0, 8).unwrap();
        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("MoE"), "unexpected error: {msg}");
    }

    #[test]
    fn forward_equivalence_rejects_zero_probe_tokens() {
        let cfg = tiny_test_cfg();
        let original = HashMap::new();
        let rotated = HashMap::new();
        let rotation = RandomizedHadamard::new(0, cfg.hidden_size).unwrap();
        let fc = ForwardEquivalenceConfig {
            num_probe_tokens: 0,
            ..Default::default()
        };
        let err = assert_forward_equivalence_qwen35(&original, &rotated, &cfg, &rotation, &fc)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("num_probe_tokens must be > 0"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn forward_equivalence_rejects_non_positive_tolerance() {
        let cfg = tiny_test_cfg();
        let original = HashMap::new();
        let rotated = HashMap::new();
        let rotation = RandomizedHadamard::new(0, cfg.hidden_size).unwrap();
        for bad in [0.0_f64, -1e-5, f64::NAN] {
            let fc = ForwardEquivalenceConfig {
                tolerance: bad,
                ..Default::default()
            };
            let err = assert_forward_equivalence_qwen35(&original, &rotated, &cfg, &rotation, &fc)
                .unwrap_err();
            let msg = format!("{err}");
            assert!(
                msg.contains("tolerance must be a positive finite value"),
                "unexpected error for tolerance={bad}: {msg}"
            );
        }
    }

    /// Refuse: rotation.dim() != cfg.hidden_size — caller passed a
    /// rotation built for the wrong space.
    #[test]
    fn forward_equivalence_rejects_rotation_dim_mismatch() {
        let cfg = tiny_test_cfg();
        let original = build_working_set(&cfg, 11);
        let rotated = original.clone();
        let rotation = RandomizedHadamard::new(0, cfg.hidden_size * 2).unwrap();
        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("rotation.dim()"), "unexpected error: {msg}");
        assert!(
            msg.contains(&format!("cfg.hidden_size={}", cfg.hidden_size)),
            "unexpected error: {msg}"
        );
    }

    /// Determinism: same seed → same probe tokens. Catches a clock-based or
    /// uninitialized RNG regression.
    #[test]
    fn probe_tokens_are_deterministic_in_seed() {
        let a = deterministic_probe_tokens(0xDEAD_BEEF, 4, 100);
        let b = deterministic_probe_tokens(0xDEAD_BEEF, 4, 100);
        assert_eq!(a, b);
        let c = deterministic_probe_tokens(0xDEAD_BEEF_u64.wrapping_add(1), 4, 100);
        assert_ne!(
            a, c,
            "different seeds should produce different probe tokens"
        );
    }

    /// Refuse error message includes the observed delta and tolerance —
    /// catches the regression where the error message would lose the
    /// diagnostic numbers operators need to triage in the binary.
    #[test]
    fn refuse_error_message_includes_max_abs_and_tolerance() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 10);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        // Real divergence: fuse only per-layer norms (skip final_norm) then
        // rotate. `diag(1 + γ_final) · R^T ≠ R^T · diag(1 + γ_final)` makes
        // the rotated lm_head produce different logits than the original.
        let per_layer = qwen35_per_layer_fusion_plan(&cfg).unwrap();
        fuse_rmsnorms(&mut rotated, &per_layer).unwrap();
        let rotation = RandomizedHadamard::new(0x1234_5678, cfg.hidden_size).unwrap();
        absorb_rotations(
            &mut rotated,
            &RotationPlan::qwen35_residual_stream_linear_layers(),
            &rotation,
        )
        .unwrap();

        let fc = ForwardEquivalenceConfig {
            tolerance: 1e-12,
            ..Default::default()
        };
        let err = assert_forward_equivalence_qwen35(&original, &rotated, &cfg, &rotation, &fc)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("max_abs_error="), "unexpected error: {msg}");
        assert!(
            msg.contains("exceeds tolerance="),
            "unexpected error: {msg}"
        );
        // Either chain-probe or per-tensor message format is acceptable —
        // both name the failing check explicitly.
        assert!(
            msg.contains("chain probe") || msg.contains("per-tensor"),
            "expected refuse message to name the failing check: {msg}"
        );
    }

    /// Per-tensor coverage helper: run the full pipeline correctly, then
    /// scale `victim_name` by `factor` in the rotated working set and
    /// assert the gate refuses. Covers the chain-probe-skipped planned
    /// tensors.
    fn assert_corrupting_planned_tensor_refuses(victim_name: &str, factor: f64, seed: u64) {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, seed);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        let (fusion, rot_plan) = full_pipeline_plans(&cfg);
        fuse_rmsnorms(&mut rotated, &fusion).unwrap();
        let rotation =
            RandomizedHadamard::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15), cfg.hidden_size)
                .unwrap();
        absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

        // Verify the pipeline is correct BEFORE corruption (so a refuse
        // after corruption is attributable to the corruption alone).
        assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_or_else(|e| {
            panic!("pre-corruption pipeline must pass for victim `{victim_name}`: {e}")
        });

        let victim = rotated
            .get_mut(victim_name)
            .unwrap_or_else(|| panic!("victim tensor `{victim_name}` missing from working set"));
        for v in victim.data.iter_mut() {
            *v *= factor;
        }

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("forward-equivalence refused"),
            "expected refuse for corrupted `{victim_name}`: {msg}"
        );
        assert!(
            msg.contains("per-tensor"),
            "corruption of `{victim_name}` should be caught by the per-tensor check: {msg}"
        );
    }

    /// Per-tensor coverage: corrupting `k_proj` is invisible to the chain
    /// probe (probe shortcuts through `q_proj` → `o_proj`) but must be
    /// caught by the per-tensor algebraic check. This regression case is
    /// the specific gap the chain probe alone leaves in PR #28.
    #[test]
    fn per_tensor_check_catches_corrupted_k_proj() {
        assert_corrupting_planned_tensor_refuses(
            "model.language_model.layers.1.self_attn.k_proj.weight",
            1000.0,
            21,
        );
    }

    /// Per-tensor coverage: `v_proj` is also chain-probe-skipped.
    #[test]
    fn per_tensor_check_catches_corrupted_v_proj() {
        assert_corrupting_planned_tensor_refuses(
            "model.language_model.layers.1.self_attn.v_proj.weight",
            -2.0,
            22,
        );
    }

    /// Per-tensor coverage: `q_proj` is shape `[2 * full_q_dim, hidden]`
    /// because Qwen3.5 fuses Q + gate_z. The chain probe takes only the
    /// first `full_q_dim` (the Q half). Corrupting the second half
    /// (gate_z) must still be caught.
    #[test]
    fn per_tensor_check_catches_corrupted_q_proj_gate_z_half() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 23);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        let (fusion, rot_plan) = full_pipeline_plans(&cfg);
        fuse_rmsnorms(&mut rotated, &fusion).unwrap();
        let rotation = RandomizedHadamard::new(0xAB12_34CD, cfg.hidden_size).unwrap();
        absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

        // Locate the layer index that is a full-attention layer in this
        // tiny config (the tied_tiny_test_cfg has one full and one linear).
        let full_layer = (0..cfg.num_hidden_layers)
            .find(|&i| cfg.is_full_attention(i))
            .expect("tied_tiny_test_cfg must have at least one full-attention layer");
        let q_name = format!("model.language_model.layers.{full_layer}.self_attn.q_proj.weight");
        let full_q_dim = cfg.full_q_dim();
        let hidden = cfg.hidden_size;
        let victim = rotated.get_mut(&q_name).unwrap();
        // Scale only the gate-z half (rows full_q_dim .. 2 * full_q_dim).
        for r in full_q_dim..(2 * full_q_dim) {
            for c in 0..hidden {
                victim.data[r * hidden + c] *= 3.0;
            }
        }

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("per-tensor"), "unexpected error: {msg}");
    }

    /// Per-tensor coverage: GDN `in_proj_qkv` is chain-probe-skipped (the
    /// GDN branch probes `in_proj_z`). Corruption must still refuse.
    #[test]
    fn per_tensor_check_catches_corrupted_in_proj_qkv() {
        assert_corrupting_planned_tensor_refuses(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            10.0,
            24,
        );
    }

    /// Per-tensor coverage: GDN `in_proj_a` is chain-probe-skipped.
    #[test]
    fn per_tensor_check_catches_corrupted_in_proj_a() {
        assert_corrupting_planned_tensor_refuses(
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            0.5,
            25,
        );
    }

    /// Per-tensor coverage: GDN `in_proj_b` is chain-probe-skipped.
    #[test]
    fn per_tensor_check_catches_corrupted_in_proj_b() {
        assert_corrupting_planned_tensor_refuses(
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            -1.0,
            26,
        );
    }

    /// Matrix-equivalence vs single-vector: the
    /// single-probe-vector implementation would miss a row
    /// perturbation orthogonal to the probe direction. The current
    /// matrix-equivalence implementation compares every stored element,
    /// so any non-zero element-level perturbation is caught. This test
    /// confirms by perturbing a single element of `k_proj` (chain probe
    /// never matvecs k_proj, single-vector check could miss the row).
    #[test]
    fn per_tensor_check_catches_single_element_perturbation_orthogonal_to_probe_vector() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 30);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        let (fusion, rot_plan) = full_pipeline_plans(&cfg);
        fuse_rmsnorms(&mut rotated, &fusion).unwrap();
        let rotation = RandomizedHadamard::new(0x9876_5432, cfg.hidden_size).unwrap();
        absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

        // Sanity: the correct pipeline passes.
        assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap();

        // Perturb a single element of k_proj. A single-probe-vector
        // check could mask this if the probe row was orthogonal; the
        // matrix-equivalence check sees the element directly.
        let k_name = "model.language_model.layers.1.self_attn.k_proj.weight";
        let victim = rotated.get_mut(k_name).unwrap();
        victim.data[0] += 0.5;

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("per-tensor"), "unexpected error: {msg}");
    }

    /// Refuse: planned tensor missing from BOTH `original` and
    /// `rotated`. Without an enumerated expected-planned-tensor set,
    /// an iteration-driven check would silently skip the absent tensor
    /// and the gate could return `Ok` despite a partial working set.
    /// The gate must enumerate the config-required set and refuse.
    #[test]
    fn per_tensor_check_errors_when_planned_tensor_missing_from_both_maps() {
        let cfg = tied_tiny_test_cfg();
        let mut original = build_working_set(&cfg, 31);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        let (fusion, rot_plan) = full_pipeline_plans(&cfg);
        fuse_rmsnorms(&mut rotated, &fusion).unwrap();
        let rotation = RandomizedHadamard::new(0xFEED_0BAD, cfg.hidden_size).unwrap();
        absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

        // Drop k_proj from BOTH maps. Chain probe never uses k_proj, so
        // it would happily pass. Per-tensor check must enumerate the
        // expected set and notice.
        let k_name = "model.language_model.layers.1.self_attn.k_proj.weight".to_string();
        assert!(original.remove(&k_name).is_some());
        assert!(rotated.remove(&k_name).is_some());

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains(&k_name), "unexpected error: {msg}");
        assert!(msg.contains("missing"), "unexpected error: {msg}");
    }

    /// `embed_tokens` corruption: covered by BOTH the chain probe and
    /// the per-tensor check. The chain probe runs first, so its message
    /// is what we see — but either is a valid refuse.
    #[test]
    fn either_check_catches_corrupted_embed_tokens() {
        let cfg = tied_tiny_test_cfg();
        let original = build_working_set(&cfg, 27);
        let mut rotated = original.clone();

        materialize_lm_head_for_qwen35(&mut rotated, &cfg).unwrap();
        let (fusion, rot_plan) = full_pipeline_plans(&cfg);
        fuse_rmsnorms(&mut rotated, &fusion).unwrap();
        let rotation = RandomizedHadamard::new(0xEDEDED, cfg.hidden_size).unwrap();
        absorb_rotations(&mut rotated, &rot_plan, &rotation).unwrap();

        let embed = rotated.get_mut(QWEN35_EMBED_TOKENS_NAME).unwrap();
        for v in embed.data.iter_mut() {
            *v *= 1.5;
        }

        let err = assert_forward_equivalence_qwen35(
            &original,
            &rotated,
            &cfg,
            &rotation,
            &ForwardEquivalenceConfig::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("forward-equivalence refused"),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains("chain probe") || msg.contains("per-tensor"),
            "expected refuse message to name the failing check: {msg}"
        );
    }
}
