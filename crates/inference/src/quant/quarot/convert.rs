//! High-level QuaRot Qwen3.5 conversion (ADR-044 step 3c-5).
//!
//! [`convert_quarot_qwen35`] reads `config.json` + SafeTensors from
//! `input_dir`, runs the full pipeline
//! (`materialize_lm_head` → `fuse_rmsnorms` → `absorb_rotations` →
//! forward equivalence) entirely in f64, and on success
//! writes the converted model to `output_dir`:
//!
//! - Planned (rotated) tensors → `<sanitized>.q4` via
//!   [`save_q4_file`].
//! - Other required weights (norms, `A_log`, `dt_bias`, `conv1d.weight`,
//!   etc.) → `<sanitized>.f16` with a `KHF1` header matching
//!   `bin/quantize_q4`'s convention.
//! - `quantize_index.json` — name/file/quantized/shape index for the
//!   runtime loader.
//! - `config.json` — mutated via
//!   [`untie_word_embeddings_in_config_json`] (no-op for untied input).
//!
//! **Refuse-on-fail**: when the forward-equivalence gate returns `Err`,
//! `convert_quarot_qwen35` returns the same `Err` immediately, **no files
//! are written**, and the output directory is left empty (or absent) so
//! a partial run cannot be mistaken for a successful one.
//!
//! `bin/quantize_quarot` is a thin argparse wrapper around this function;
//! direct library callers can use the same function with custom paths.

use std::fs;
use std::io::Write;
use std::path::Path;

use crate::error::InferenceError;
use crate::model::qwen35::qwen_required_tensor_names;
use crate::model::qwen35_config::Qwen35Config;
use crate::quant::quarot::forward_equivalence::{
    ForwardEquivalenceConfig, ForwardEquivalenceReport, assert_prepared_forward_equivalence_qwen35,
    prepare_forward_equivalence_qwen35_after_admission, validate_forward_equivalence_admission,
};
#[cfg(test)]
use crate::quant::quarot::forward_equivalence::{
    pre_admission_allocation_tracking, prepare_forward_equivalence_qwen35,
};
use crate::quant::quarot::hadamard::RandomizedHadamard;
use crate::quant::quarot::io::{ArtifactVersion, OnlineArtifactDescriptor, QuarotTensorReader};
use crate::quant::quarot::lm_head::{
    materialize_lm_head_for_qwen35, qwen35_final_norm_fusion_target,
    untie_word_embeddings_in_config_json,
};
use crate::quant::quarot::pipeline::{
    TensorEntry, absorb_rotations, fuse_rmsnorms, load_tensors_f64,
};
use crate::quant::quarot::plan::RotationPlan;
use crate::quant::quarot::rmsnorm_fusion::qwen35_per_layer_fusion_plan;
use crate::weights::ingress::DecodedTensorValidator;
use crate::weights::q4_weights::{q4_f32_to_finite_f16, quantize_f64_to_q4, save_q4_file};

/// Upper bound on `hidden_size` accepted by [`convert_quarot_qwen35`], well above any
/// real Qwen3.5 checkpoint, guarding `RandomizedHadamard::new`'s config-sized allocation
/// against a hostile or corrupted `config.json`.
const MAX_QUAROT_HIDDEN_SIZE: usize = 1 << 20;

/// CLI / library options for [`convert_quarot_qwen35`].
#[derive(Debug, Clone)]
pub struct ConversionOptions {
    /// Seed for the residual-stream Hadamard rotation. Must match the
    /// seed the runtime expects for adapter-aware code paths (v1+); in
    /// v0 the seed is just a knob for reproducibility.
    pub rotation_seed: u64,
    /// Forward-equivalence tolerance (passed through to the gate). The
    /// ADR-044 §"Step 3c contract" target is `1e-5`.
    pub tolerance: f64,
    /// Number of token IDs the chain probe samples (passed through).
    pub num_probe_tokens: usize,
    /// When `true`, run the full pipeline + forward-equivalence gate
    /// but skip every disk write. Useful for CI sanity passes.
    pub dry_run: bool,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            rotation_seed: 0xCAFE_BABE_DEAD_BEEF,
            tolerance: 1e-5,
            num_probe_tokens: 4,
            dry_run: false,
        }
    }
}

/// Summary of a successful [`convert_quarot_qwen35`] run.
#[derive(Debug, Clone)]
pub struct ConversionReport {
    /// Tensors that matched the rotation plan and were written as `.q4`.
    pub planned_quantized: usize,
    /// Tensors written as `.f16` (norms, biases, conv1d, `A_log`,
    /// `dt_bias`, etc.).
    pub kept_f16: usize,
    /// Sum of on-disk byte sizes of the language-model tensors the pipeline
    /// reads from the source checkpoint.
    ///
    /// Includes: every tensor in `required_names` (the language-model
    /// subset, with `embed_tokens` counted once), plus the on-disk spans of
    /// the MTP tensors that `write_mtp_weights_quarot` copies to the output
    /// as `.f16` files (these are not in `required_names`). `embed_tokens`
    /// is NOT double-counted for the tied lm_head: the output's
    /// `lm_head_weight.q4` is a second Q4 copy of that one source tensor,
    /// already accounted here.
    ///
    /// Note: the full multimodal file is ~1627 MiB for Qwen3.5-0.8B; this
    /// field counts only the processed language-model subset (the vision
    /// tower is not read), so it is smaller than the physical file footprint.
    pub total_bytes_in: u64,
    /// Sum of output tensor sizes in bytes (Q4 blocks + f16 payload +
    /// per-file headers).
    pub total_bytes_out: u64,
    /// Forward-equivalence gate output (the gate's `Ok` is what gated
    /// this report's existence).
    pub forward_equivalence: ForwardEquivalenceReport,
    /// `true` when the input config had `tie_word_embeddings = true` and
    /// the converter materialized `lm_head` and flipped the output
    /// config to untied.
    pub was_tied: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct IndexEntry {
    name: String,
    file: String,
    quantized: bool,
    shape: Vec<usize>,
    numel: usize,
}

/// Promotion status of a `quantize_quarot` artifact (issue #1103).
///
/// The converter's forward-equivalence gate (rotation correctness) and the
/// ADR-044 PPL acceptance gate (quantization quality, run separately via
/// `bin/eval_perplexity --q4-dir <baseline> --quarot-q4-dir <this output>`)
/// check different things. `convert_quarot_qwen35` can only ever establish
/// [`PromotionState::Unpromoted`] — it has no baseline Q4 directory or
/// corpus to measure PPL against. Only [`record_ppl_gate_result`], called
/// after the acceptance measurement actually runs, can flip this to
/// [`PromotionState::Promoted`] or [`PromotionState::Rejected`].
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PromotionState {
    /// The PPL acceptance gate has not been recorded against this
    /// artifact. Default state for every artifact `convert_quarot_qwen35`
    /// writes; a caller MUST NOT treat an unpromoted artifact as
    /// quality-validated.
    Unpromoted,
    /// A recorded PPL measurement passed the acceptance threshold
    /// (`delta < threshold`).
    Promoted,
    /// A recorded PPL measurement failed the acceptance threshold
    /// (`delta >= threshold`).
    Rejected,
}

/// A recorded ADR-044 dual-Q4 PPL acceptance measurement.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct PplGateRecord {
    pub unrotated_ppl: f64,
    pub quarot_ppl: f64,
    /// `quarot_ppl - unrotated_ppl`.
    pub delta: f64,
    pub delta_threshold: f64,
}

/// Promotion marker persisted in `quantize_index.json` (issue #1103). Makes
/// the two-step contract (forward-equivalence at convert time, PPL quality
/// gate at measurement time) explicit and durable on the artifact itself,
/// instead of letting a complete artifact + exit 0 read as "fully
/// validated" when the quality gate never ran.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct PromotionRecord {
    pub state: PromotionState,
    /// Human-readable explanation of `state` — always present so a reader
    /// never has to infer WHY from the state alone.
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ppl_gate: Option<PplGateRecord>,
}

impl PromotionRecord {
    fn unpromoted() -> Self {
        Self {
            state: PromotionState::Unpromoted,
            reason: "PPL acceptance gate has not been run against this artifact; run \
                     `eval_perplexity --q4-dir <unrotated baseline> --quarot-q4-dir <this \
                     output dir> --tokenizer-dir <src> --corpus-file <corpus>` to record a \
                     result before treating it as quality-validated (ADR-044 step 4, #1103)."
                .to_string(),
            ppl_gate: None,
        }
    }
}

/// Legacy artifacts (pre-#1103) and any manifest missing the `promotion`
/// field deserialize to this — still `Unpromoted`, with a reason that
/// distinguishes "never recorded" from "predates promotion tracking".
impl Default for PromotionRecord {
    fn default() -> Self {
        Self {
            state: PromotionState::Unpromoted,
            reason: "no promotion record present in quantize_index.json (artifact predates \
                     #1103 promotion tracking, or the gate has not been run)"
                .to_string(),
            ppl_gate: None,
        }
    }
}

/// Wire format for `quantize_index.json` produced by `convert_quarot_qwen35`.
///
/// ADR-051 §"quantize_quarot Binary Change": the rotation seed is the runtime's
/// authoritative source for reconstructing the QuaRot Hadamard sign vector. It
/// lives next to the tensor index so a loader can recover it without parsing
/// `config.json`.
///
/// The field is `Option<u64>` for wire-format reasons, not because a seedless
/// object-form manifest is a supported state. The only writer of this shape is
/// [`convert_quarot_qwen35`], which rotates unconditionally and has stamped
/// `Some(seed)` here since the struct was introduced, so "an older build wrote
/// an object-form manifest without a seed" describes a population that has
/// never existed. [`read_quarot_seed_from_index`] refuses that combination
/// when the legacy `config.json` channel is also silent. ADR-051's
/// absent-seed-means-unrotated clause is about the pre-existing bare-array
/// artifacts it promised to keep loadable, and those still load unchanged.
///
/// `online` is the schema-of-record home for [`OnlineArtifactDescriptor`] —
/// the ONE place a converter/loader reads or writes online-rotation
/// metadata, closing the dual-schema gap between this wire format and the
/// contract types in `io.rs`. `#[serde(default, ...)]` keeps every existing
/// V0 manifest (which never had this field) byte-compatible: absent on disk
/// deserializes to `None`, and `None` is never re-serialized back out.
///
/// `artifact_version` is a second, independent top-level signal carrying the
/// same [`ArtifactVersion`] a real V1 writer stamps on its manifest. It is
/// deliberately NOT read from `online.version` — `online` is optional and a
/// truncated write, a corrupted copy, or a hand edit can drop or null it
/// while leaving the rest of the manifest (including this field) intact.
/// Keying version detection on `artifact_version` means that combination is
/// caught as an incomplete V1 artifact rather than silently downgraded to
/// V0. See [`read_quarot_seed_from_index`] for the load-time contract.
#[derive(serde::Serialize, serde::Deserialize)]
struct QuantizeIndex {
    #[serde(skip_serializing_if = "Option::is_none")]
    quarot_seed: Option<u64>,
    tensors: Vec<IndexEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    online: Option<OnlineArtifactDescriptor>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    artifact_version: Option<ArtifactVersion>,
    /// Promotion marker (#1103). `#[serde(default)]` keeps every manifest
    /// written before this field existed byte-compatible on read: an
    /// absent field deserializes to [`PromotionRecord::default`], which is
    /// `Unpromoted` — the safe, fail-closed reading of "we don't know
    /// whether the gate ran" for a manifest older than the tracking itself.
    #[serde(default)]
    promotion: PromotionRecord,
}

/// Read the QuaRot rotation seed from `quantize_index.json` per ADR-051.
///
/// Delegates only the bounded, fail-closed byte read to
/// [`crate::quant::q4_manifest::read_manifest_bytes_bounded`] (issue #655 —
/// the one shared reader also backs `lattice doctor`'s inventory). Shape
/// normalization stays here, deliberately **not** unified with `doctor`'s
/// tolerant [`crate::quant::q4_manifest::parse_manifest`]: this reader's
/// accept/reject contract predates #655 and must not silently change.
/// Specifically:
///
/// - A bare top-level JSON array (`quantize_q4`'s shape) genuinely carries
///   no rotation seed. Its entries are **not validated** here — any array,
///   however malformed its entries, is `Ok(None)`, matching the pre-#655
///   reader exactly (a seed can only ever appear in the object form, so a
///   malformed array entry is `doctor`'s concern, not this reader's).
/// - An object form (`quantize_quarot`'s shape, `{"quarot_seed": ...,
///   "tensors": [...]}`) is parsed strictly via [`QuantizeIndex`] /
///   [`IndexEntry`]: every tensor entry must carry `name`, `file`,
///   `quantized`, `shape`, and `numel`, or the whole manifest is rejected.
///   A partially-formed object-form manifest is evidence of a corrupted or
///   in-progress write, not a legitimate "no seed" state.
///
/// Fail-closed contract (#504 remaining slice 2): a genuinely **absent**
/// file is `Ok(None)` — pre-ADR-051 artifacts never had this file, and the
/// caller falls back to the legacy `config.json` field
/// (`quarot_rotation_seed`) for those. A **present** file that fails to
/// read, exceeds the size cap, is not valid JSON, or (for the object form)
/// does not match the strict schema is `Err`. An object-form file that parses
/// but carries no `quarot_seed` is also `Err`, unless `cfg` supplies the
/// legacy seed: the object shape identifies the converter as its writer, and
/// that converter always rotates, so a seed absent from both channels is a
/// rotated artifact with no recoverable recipe rather than an unrotated one. A file that exists and
/// doesn't parse is evidence of truncation/corruption/tampering, not a
/// legitimate "no rotation" artifact, and silently falling back to the
/// legacy seed (or no rotation at all) would apply the wrong rotation to a
/// checkpoint that was actually QuaRot-rotated, silently corrupting
/// inference output instead of refusing to load (#630 end-to-end
/// verification against the real qwen3.5-0.8b-q4 checkpoint).
///
/// When the object form carries an `online` [`OnlineArtifactDescriptor`]
/// whose version is `V0Residual`, it is validated against `cfg`
/// (`OnlineArtifactDescriptor::validate`) before this function returns — a
/// manifest carrying a present-but-invalid online descriptor is rejected at
/// load time, the same fail-closed treatment as a malformed `tensors`
/// entry. A manifest with no `online` field (every V0 manifest, and any
/// object-form manifest predating this field) skips this check entirely.
///
/// A `V1Online` descriptor is **always** rejected here (`Err`, not `Ok`),
/// whether or not it is internally self-consistent — no forward path
/// executes R3/R4 online rotations at runtime yet, so returning the seed
/// for a V1 artifact would make `from_q4_dir` load it exactly like
/// `V0Residual` and silently skip the counter-rotations its Q4 weights
/// require, producing incorrect inference. This is the single load
/// boundary every `read_quarot_seed_from_index` caller passes through; a
/// `V1Online` artifact stays rejected until end-to-end runtime rotation
/// support lands. Because that rejection is unconditional, the version
/// check runs BEFORE `OnlineArtifactDescriptor::validate` rather than
/// after: `validate`'s R3/R4 layer-scope and asymmetric-tensor-name checks
/// are quadratic in attacker-controlled manifest content (declared layer
/// count, declared tensor-name count), and there is no reason to run that
/// scan on a descriptor this function refuses regardless of the outcome.
///
/// Version detection keys on the top-level `artifact_version` field, not on
/// `online`'s presence: `online` is optional and defaulted, so a manifest
/// with `artifact_version` absent (or `v0-residual`) but `online` present
/// falls through to the same present-online validation above, while a
/// manifest that declares `artifact_version: "v1-online-r3r4"` MUST carry a
/// complete, valid `online` descriptor or is rejected outright — an absent
/// or null `online` field on a manifest that declares itself V1 is treated
/// as an incomplete/corrupted V1 artifact, never silently downgraded to V0.
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) fn read_quarot_seed_from_index(
    q4_dir: &Path,
    cfg: &Qwen35Config,
) -> Result<Option<u64>, String> {
    let path = q4_dir.join("quantize_index.json");
    let Some(bytes) = crate::quant::q4_manifest::read_manifest_bytes_bounded(&path)? else {
        return Ok(None);
    };
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| format!("{}: malformed quantize_index.json: {e}", path.display()))?;
    if value.is_array() {
        // Bare array (quantize_q4 shape): genuinely no seed. Entries are
        // intentionally not validated here — see the doc comment above.
        return Ok(None);
    }
    let index: QuantizeIndex = serde_json::from_value(value)
        .map_err(|e| format!("{}: malformed quantize_index.json: {e}", path.display()))?;
    if matches!(index.artifact_version, Some(ArtifactVersion::V1Online)) {
        // The manifest declares itself V1 independently of whether `online`
        // made it through intact. An absent or null `online` here is not a
        // downgrade to V0 — it is an incomplete V1 artifact, and its Q4
        // weights are already counter-rotated for R3/R4 with no recipe left
        // to describe what to undo. Refuse rather than guess.
        if index.online.is_none() {
            return Err(format!(
                "{}: artifact_version declares v1-online-r3r4 but the online \
                 rotation descriptor is missing or null; refusing to load an \
                 incomplete V1 artifact",
                path.display()
            ));
        }
        // This runtime rejects every V1Online artifact outright, whether or
        // not its descriptor is internally self-consistent — no forward
        // path executes R3/R4 online rotations yet. The version check is
        // therefore resolved BEFORE `OnlineArtifactDescriptor::validate`
        // runs: that call's R3/R4 layer-scope and asymmetric-tensor-name
        // checks are O(layers^2 + tensor_names^2) over attacker-controlled
        // manifest content (a small malicious config declaring many layers
        // plus a matching V1 descriptor), and running them ahead of a
        // rejection this runtime always issues would let that manifest
        // drive the full quadratic scan before being refused.
        return Err(format!(
            "{}: this runtime does not yet execute V1 online rotation \
             recipes; artifact requires R3/R4 runtime support",
            path.display()
        ));
    }
    if index.quarot_seed.is_none() && cfg.quarot_rotation_seed.is_none() {
        // Object form means `convert_quarot_qwen35` wrote this manifest, and
        // that converter rotates unconditionally and stamps the seed into both
        // channels it owns (`quarot_seed` here, `quarot_rotation_seed` in the
        // emitted `config.json`). So an object-form manifest whose seed is
        // absent from BOTH channels describes rotated weights whose rotation
        // is no longer recoverable — the same "corrupted or in-progress write"
        // reading the entry-completeness check above already applies, extended
        // to the field that decides whether counter-rotation runs at all.
        //
        // The bare-array shape is untouched: it returns above, because for
        // `bin/quantize_q4.rs` output an absent seed is the only representable
        // state and genuinely means unrotated. The legacy `config.json`
        // channel is untouched too — a seed there yields `is_quarot = true`
        // with counter-rotation correctly applied, which is the fallback path
        // this refusal must never reach.
        //
        // Checked before the `online` descriptor validation below for the
        // reject-before-validate reason stated above: a manifest refused
        // regardless of the outcome must not first drive that O(layers^2 +
        // tensor_names^2) scan over attacker-controlled content.
        return Err(format!(
            "{}: object-form quantize_index.json (quantize_quarot shape) \
             carries no quarot_seed and config.json declares no \
             quarot_rotation_seed; refusing to load rotated weights with no \
             recoverable rotation seed",
            path.display()
        ));
    }
    if let Some(online) = &index.online {
        // Same reject-before-validate ordering as above, for the
        // `artifact_version` omitted/`V0Residual` but `online.version ==
        // V1Online` case (a manifest whose top-level tag lags its embedded
        // descriptor).
        if matches!(online.version, ArtifactVersion::V1Online) {
            return Err(format!(
                "{}: this runtime does not yet execute V1 online rotation \
                 recipes; artifact requires R3/R4 runtime support",
                path.display()
            ));
        }
        online.validate(Some(cfg)).map_err(|e| {
            format!(
                "{}: invalid online-artifact descriptor: {e}",
                path.display()
            )
        })?;
    }
    Ok(index.quarot_seed)
}

/// Record an ADR-044 dual-Q4 PPL acceptance measurement against a
/// `quantize_quarot` output directory (issue #1103).
///
/// Reads the existing `quantize_index.json` in `quarot_dir` (must be the
/// object-form QuaRot manifest — a bare-array `quantize_q4` manifest has no
/// `promotion` field to record against and is rejected), flips `promotion`
/// to [`PromotionState::Promoted`] when `quarot_ppl - unrotated_ppl <
/// delta_threshold` or [`PromotionState::Rejected`] otherwise, and writes
/// the manifest back. Every other field round-trips unchanged.
///
/// This function performs no PPL measurement itself — it only records a
/// measurement the caller already computed (`bin/eval_perplexity`'s
/// dual-Q4 mode calls this after printing its own verdict). Fail-closed: a
/// missing, malformed, or bare-array manifest is `Err`, never a silent
/// no-op — a caller that cannot durably record the result must not report
/// success either.
pub fn record_ppl_gate_result(
    quarot_dir: &Path,
    unrotated_ppl: f64,
    quarot_ppl: f64,
    delta_threshold: f64,
) -> Result<PromotionRecord, InferenceError> {
    let path = quarot_dir.join("quantize_index.json");
    let bytes = fs::read(&path).map_err(|e| {
        InferenceError::Inference(format!(
            "record_ppl_gate_result: failed to read {}: {e}",
            path.display()
        ))
    })?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        InferenceError::Inference(format!(
            "record_ppl_gate_result: {}: malformed quantize_index.json: {e}",
            path.display()
        ))
    })?;
    if value.is_array() {
        return Err(InferenceError::Inference(format!(
            "record_ppl_gate_result: {} is a bare-array manifest (quantize_q4 shape, no \
             promotion field); only a quantize_quarot object-form manifest can record a \
             PPL gate result",
            path.display()
        )));
    }
    let mut index: QuantizeIndex = serde_json::from_value(value).map_err(|e| {
        InferenceError::Inference(format!(
            "record_ppl_gate_result: {}: malformed quantize_index.json: {e}",
            path.display()
        ))
    })?;

    let delta = quarot_ppl - unrotated_ppl;
    let passed = delta < delta_threshold;
    let record = PromotionRecord {
        state: if passed {
            PromotionState::Promoted
        } else {
            PromotionState::Rejected
        },
        reason: if passed {
            format!(
                "ADR-044 PPL acceptance gate passed: delta {delta:+.6} < threshold \
                 {delta_threshold:.6} (quarot {quarot_ppl:.6} - unrotated {unrotated_ppl:.6})"
            )
        } else {
            format!(
                "ADR-044 PPL acceptance gate failed: delta {delta:+.6} >= threshold \
                 {delta_threshold:.6} (quarot {quarot_ppl:.6} - unrotated {unrotated_ppl:.6})"
            )
        },
        ppl_gate: Some(PplGateRecord {
            unrotated_ppl,
            quarot_ppl,
            delta,
            delta_threshold,
        }),
    };
    index.promotion = record.clone();

    let json = serde_json::to_string_pretty(&index).map_err(|e| {
        InferenceError::Inference(format!(
            "record_ppl_gate_result: failed to serialize {}: {e}",
            path.display()
        ))
    })?;
    fs::write(&path, json).map_err(|e| {
        InferenceError::Inference(format!(
            "record_ppl_gate_result: failed to write {}: {e}",
            path.display()
        ))
    })?;

    Ok(record)
}

/// Read the current promotion marker from a `quantize_quarot` output
/// directory (issue #1103). `Ok(PromotionRecord::default())` — i.e.
/// [`PromotionState::Unpromoted`] — for a present object-form manifest with
/// no `promotion` field (pre-#1103 artifact). `Err` for a missing manifest,
/// malformed JSON, or a bare-array (`quantize_q4`) manifest, which has no
/// promotion concept to read.
pub fn read_promotion_record(quarot_dir: &Path) -> Result<PromotionRecord, InferenceError> {
    let path = quarot_dir.join("quantize_index.json");
    let bytes = fs::read(&path).map_err(|e| {
        InferenceError::Inference(format!(
            "read_promotion_record: failed to read {}: {e}",
            path.display()
        ))
    })?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        InferenceError::Inference(format!(
            "read_promotion_record: {}: malformed quantize_index.json: {e}",
            path.display()
        ))
    })?;
    if value.is_array() {
        return Err(InferenceError::Inference(format!(
            "read_promotion_record: {} is a bare-array manifest (quantize_q4 shape); it has \
             no promotion field",
            path.display()
        )));
    }
    let index: QuantizeIndex = serde_json::from_value(value).map_err(|e| {
        InferenceError::Inference(format!(
            "read_promotion_record: {}: malformed quantize_index.json: {e}",
            path.display()
        ))
    })?;
    Ok(index.promotion)
}

fn inject_quarot_seed(json: &str, seed: u64) -> Result<String, InferenceError> {
    let mut value: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| InferenceError::Inference(format!("inject_quarot_seed: invalid JSON: {e}")))?;
    let obj = value.as_object_mut().ok_or_else(|| {
        InferenceError::Inference(
            "inject_quarot_seed: top-level JSON must be an object".to_string(),
        )
    })?;
    if let Some(text_config) = obj.get_mut("text_config")
        && let Some(text_obj) = text_config.as_object_mut()
    {
        text_obj.insert(
            "quarot_rotation_seed".to_string(),
            serde_json::Value::Number(seed.into()),
        );
    }
    obj.insert(
        "quarot_rotation_seed".to_string(),
        serde_json::Value::Number(seed.into()),
    );
    serde_json::to_string_pretty(&value).map_err(|e| {
        InferenceError::Inference(format!("inject_quarot_seed: serialize failed: {e}"))
    })
}

/// Compute the byte count that `write_f16_file` would write for a tensor
/// with `data_len` elements and `shape_len` dimensions, without performing
/// any I/O. Used by both the real write path (as a cross-check) and the
/// dry-run accounting path.
///
/// Layout mirrors `write_f16_file` exactly:
/// ```text
///   magic[4] version[4] ndim[4] dims[8*ndim] numel[8] payload[numel*2]
/// ```
fn f16_file_byte_count(data_len: usize, shape_len: usize) -> u64 {
    // Header: magic(4) + version(4) + ndim(4) + shape(8*ndim) + numel(8)
    let header: u64 = 4 + 4 + 4 + 8 * shape_len as u64 + 8;
    // Payload: each f64 element becomes one f16 (2 bytes).
    let payload: u64 = data_len as u64 * 2;
    header + payload
}

fn write_mtp_weights_quarot(
    reader: &QuarotTensorReader,
    source: &str,
    output_dir: &Path,
    dry_run: bool,
    index_entries: &mut Vec<IndexEntry>,
    kept_f16: &mut usize,
    _planned_quantized: &mut usize,
    total_bytes_out: &mut u64,
) -> Result<(), InferenceError> {
    // ADR-051 §"MTP tensors still safety-skipped in quantize_quarot": Phase 1 keeps
    // every MTP tensor as f16 (unquantized). The runtime counter-rotates on
    // unquantized weights; Phase 2 will rotate and quantize MTP tensors offline.
    // Splitting projections from norms here mirrors the loader split in
    // `load_mtp_weights_q4_dir` — projections are loaded as half buffers for
    // `gemv_decode_m1`, norms as f32 buffers for the RMSNorm kernels — but both
    // come from `.f16` files on disk.
    let proj_names = [
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.up_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
    ];
    let norm_names = [
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    ];

    let mut process_as_f16 = |name: &str| -> Result<(), InferenceError> {
        if !reader.has_tensor(name) {
            return Ok(());
        }
        let (data, shape) = reader.read_tensor_f64(name)?;
        let sanitized = sanitize_tensor_name(name);
        let file_name = format!("{sanitized}.f16");
        // Byte accounting uses the pure formula: same result as write_f16_file
        // would return, derived from shape and numel without any I/O.
        *total_bytes_out += f16_file_byte_count(data.len(), shape.len());
        if !dry_run {
            let out_path = output_dir.join(&file_name);
            write_f16_file(&out_path, source, name, &data, &shape)?;
        }
        *kept_f16 += 1;
        index_entries.push(IndexEntry {
            name: name.to_string(),
            file: file_name,
            quantized: false,
            shape: shape.clone(),
            numel: data.len(),
        });
        Ok(())
    };

    for name in &proj_names {
        process_as_f16(name)?;
    }
    for name in &norm_names {
        process_as_f16(name)?;
    }

    Ok(())
}

/// Refuse-on-fail QuaRot Qwen3.5 model conversion (ADR-044 §"Step 3c contract").
///
/// On success, writes the converted model to `output_dir` (created if
/// absent) and returns a [`ConversionReport`]. On the forward-equivalence
/// gate refusing, returns the gate's `Err` **without writing any output
/// files**.
///
/// `opts.dry_run = true` runs the full pipeline + gate but skips every
/// disk write, returning a report with real `planned_quantized`,
/// `kept_f16`, and `total_bytes_out` values computed using the same
/// formulas the write path applies (Q4: header + blocks×20; f16:
/// header + numel×2). This lets callers preview the output size and
/// compression ratio before committing the write. In dry-run the
/// output-directory layout validation (same-path, non-empty checks) is
/// also skipped — those constraints exist to keep the write path from
/// corrupting source artifacts, and dry-run produces no writes by
/// definition.
///
/// # Errors
///
/// - `input_dir/config.json` missing or invalid HF Qwen config.
/// - `cfg.hidden_size` not a power of 2 (v0 QuaRot requirement —
///   ADR-044 §Model coverage).
/// - `cfg.is_moe()` — MoE deferred to v1.
/// - SafeTensors reader fails (missing tensor, unsupported dtype, …).
/// - Pipeline error (`materialize_lm_head` / `fuse_rmsnorms` /
///   `absorb_rotations` propagated through).
/// - Forward-equivalence gate refuses — propagated unchanged.
/// - Disk I/O error during output write (file create, write,
///   directory create).
///
/// The emitted `config.json` carries `quarot_rotation_seed` so the runtime
/// can reconstruct the Hadamard rotation for MTP counter-rotation.
pub fn convert_quarot_qwen35(
    input_dir: &Path,
    output_dir: &Path,
    opts: &ConversionOptions,
) -> Result<ConversionReport, InferenceError> {
    // Path-layout validation runs FIRST so the cheap CLI footguns
    // (same dir, non-empty target) fail before any expensive tensor
    // work and before any disk write. Skipped in dry-run because the
    // function returns before any `fs::create_dir_all` or file write
    // happens, so the footguns cannot fire — callers may legitimately
    // dry-run against an existing populated output location or a
    // placeholder that happens to equal the input directory.
    if !opts.dry_run {
        validate_output_dir_layout(input_dir, output_dir)?;
    }

    let config_path = input_dir.join("config.json");
    let config_json = fs::read_to_string(&config_path).map_err(|e| {
        InferenceError::Inference(format!(
            "convert_quarot_qwen35: failed to read {}: {e}",
            config_path.display()
        ))
    })?;
    let cfg = Qwen35Config::from_config_json_str(&config_json)?;

    if !cfg.hidden_size.is_power_of_two() {
        return Err(InferenceError::Inference(format!(
            "convert_quarot_qwen35: hidden_size={} is not a power of 2; \
             QuaRot v0 only supports power-of-2 hidden dims \
             (see ADR-044 §Model coverage)",
            cfg.hidden_size
        )));
    }
    // `hidden_size` drives `RandomizedHadamard::new`'s sign-buffer allocation directly
    // (`Vec::with_capacity(hidden_size)` and per-block sign generation). A hostile or
    // corrupted `config.json` can pass the power-of-two check above with an absurd
    // value (e.g. `1 << 60`) and drive that allocation to a capacity-overflow abort or
    // OOM kill before any tensor bytes are read. No real Qwen3.5 model exceeds a few
    // tens of thousands of hidden dims; cap generously above that and reject anything
    // larger with a typed error before the allocation is ever attempted.
    if cfg.hidden_size > MAX_QUAROT_HIDDEN_SIZE {
        return Err(InferenceError::Inference(format!(
            "convert_quarot_qwen35: hidden_size={} exceeds the maximum supported value \
             ({MAX_QUAROT_HIDDEN_SIZE}); this is almost certainly a corrupted or hostile \
             config.json, not a real model",
            cfg.hidden_size
        )));
    }
    if cfg.is_moe() {
        return Err(InferenceError::Inference(
            "convert_quarot_qwen35: MoE configs are deferred to v1 (see ADR-044 §Out of v0)"
                .to_string(),
        ));
    }

    let forward_cfg = ForwardEquivalenceConfig {
        num_probe_tokens: opts.num_probe_tokens,
        tolerance: opts.tolerance,
        seed: opts.rotation_seed,
    };
    #[cfg(test)]
    pre_admission_allocation_tracking::mark_converter_boundary();
    let forward_admission = validate_forward_equivalence_admission(&cfg, &forward_cfg)?;

    #[cfg(test)]
    pre_admission_allocation_tracking::mark_reader_boundary();
    let reader = QuarotTensorReader::open(input_dir)?;
    let input_source = input_dir.display().to_string();
    let required_names = qwen_required_tensor_names(&cfg);
    // Measure the on-disk footprint of the language-model tensors the pipeline
    // reads and writes, using SafeTensors header byte spans
    // (`bytes_in = h.end - h.start`).  Same approach as `bin/quantize_q4`.  For a
    // bf16 checkpoint each element is 2 bytes on disk, so this is far smaller
    // than the 8-byte-per-element f64 working-copy size.
    //
    // This is the processed LM subset, intentionally SMALLER than the full
    // multimodal checkpoint on disk: QuaRot does not read or rewrite the vision
    // tower, so it is excluded from both the input and output bases (symmetric).
    //
    // `embed_tokens` is counted exactly ONCE (its real on-disk footprint).  When
    // `tie_word_embeddings` is true the output un-ties and writes TWO Q4 tensors
    // derived from it (`embed_tokens.q4` plus a materialized `lm_head_weight.q4`),
    // but both are copies of the single embed tensor already counted here, so the
    // lm_head is accounted on the input side.  Counting embed twice would make
    // the reported input exceed the physical model file.
    let mut total_bytes_in: u64 = required_names
        .iter()
        .map(|name| reader.tensor_byte_len(name))
        .collect::<Result<Vec<u64>, _>>()?
        .into_iter()
        .sum();
    // MTP tensors ARE a genuine adjustment: `write_mtp_weights_quarot` copies
    // them to output (kept as f16) but they are not in `required_names`, so add
    // their on-disk spans once to keep the input and output bases symmetric.
    if cfg.mtp_num_hidden_layers > 0 {
        let mtp_names = [
            "mtp.fc.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight",
            "mtp.layers.0.mlp.down_proj.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
        ];
        for name in &mtp_names {
            if reader.has_tensor(name) {
                total_bytes_in += reader.tensor_byte_len(name)?;
            }
        }
    }
    let mut working_set = load_tensors_f64(&reader, &required_names)?;

    let was_tied = cfg.tie_word_embeddings;
    if was_tied {
        working_set.reserve(1);
    }
    #[cfg(test)]
    pre_admission_allocation_tracking::mark_materialized_working_set_boundary();
    if was_tied {
        materialize_lm_head_for_qwen35(&mut working_set, &cfg)?;
    }

    let rotation = RandomizedHadamard::new(opts.rotation_seed, cfg.hidden_size)?;
    // The historical `working_set.clone()` regression sat exactly here, between
    // materialization and this prepare call (see git history at this path). Keep
    // the `MaterializedWorkingSetBoundary` phase open through the call below so
    // `converter_does_not_clone_materialized_working_set` observes allocations
    // at the former clone site instead of stopping at materialization.
    let equivalence_snapshot = prepare_forward_equivalence_qwen35_after_admission(
        &working_set,
        &rotation,
        forward_admission,
    )?;
    #[cfg(test)]
    pre_admission_allocation_tracking::mark_materialized_working_set_boundary_completed();

    let mut fusion_plan = qwen35_per_layer_fusion_plan(&cfg)?;
    fusion_plan.push(qwen35_final_norm_fusion_target());
    let rotation_plan = RotationPlan::qwen35_residual_stream_linear_layers();

    fuse_rmsnorms(&mut working_set, &fusion_plan)?;
    absorb_rotations(&mut working_set, &rotation_plan, &rotation)?;

    let forward_equivalence =
        assert_prepared_forward_equivalence_qwen35(equivalence_snapshot, &reader, &working_set)?;

    if !opts.dry_run {
        fs::create_dir_all(output_dir).map_err(|e| {
            InferenceError::Inference(format!(
                "convert_quarot_qwen35: failed to create output directory {}: {e}",
                output_dir.display()
            ))
        })?;
    }

    let mut names: Vec<String> = working_set.keys().cloned().collect();
    names.sort();

    let mut index_entries: Vec<IndexEntry> = Vec::with_capacity(names.len());
    let mut planned_quantized: usize = 0;
    let mut kept_f16: usize = 0;
    let mut total_bytes_out: u64 = 0;

    for name in &names {
        let entry: &TensorEntry = &working_set[name];
        let sanitized = sanitize_tensor_name(name);
        let is_planned = rotation_plan.for_tensor(name).is_some();

        if is_planned {
            if entry.shape.len() != 2 {
                return Err(InferenceError::Inference(format!(
                    "convert_quarot_qwen35: planned tensor `{name}` has shape {:?}, \
                     expected 2-D for Q4 quantization (rotation plan invariant violated)",
                    entry.shape
                )));
            }
            // Q4 file footprint: 4-byte magic + 4 version + 4 ndim +
            // 8*ndim shape + 8 original_len + 20 bytes per block (asymmetric).
            // Block count: original_len.div_ceil(32). Computed from shape WITHOUT
            // quantizing so the dry-run path produces the same number without
            // allocating the Q4 buffer.
            let header_bytes = (4 + 4 + 4 + 8 * entry.shape.len() + 8) as u64;
            let n_blocks = entry.data.len().div_ceil(32) as u64;
            total_bytes_out += header_bytes + n_blocks.saturating_mul(20);
            if !opts.dry_run {
                let q4 = quantize_f64_to_q4(&entry.data, &entry.shape)?;
                let file_name = format!("{sanitized}.q4");
                let out_path = output_dir.join(&file_name);
                save_q4_file(&out_path, &q4).map_err(|e| {
                    InferenceError::Inference(format!(
                        "convert_quarot_qwen35: failed to write {}: {e}",
                        out_path.display()
                    ))
                })?;
                index_entries.push(IndexEntry {
                    name: name.clone(),
                    file: file_name,
                    quantized: true,
                    shape: entry.shape.clone(),
                    numel: entry.data.len(),
                });
            }
            planned_quantized += 1;
        } else {
            // f16 file footprint computed from shape and numel without writing.
            total_bytes_out += f16_file_byte_count(entry.data.len(), entry.shape.len());
            if !opts.dry_run {
                let file_name = format!("{sanitized}.f16");
                let out_path = output_dir.join(&file_name);
                write_f16_file(&out_path, &input_source, name, &entry.data, &entry.shape)?;
                index_entries.push(IndexEntry {
                    name: name.clone(),
                    file: file_name,
                    quantized: false,
                    shape: entry.shape.clone(),
                    numel: entry.data.len(),
                });
            }
            kept_f16 += 1;
        }
    }

    if cfg.mtp_num_hidden_layers > 0 {
        write_mtp_weights_quarot(
            &reader,
            &input_source,
            output_dir,
            opts.dry_run,
            &mut index_entries,
            &mut kept_f16,
            &mut planned_quantized,
            &mut total_bytes_out,
        )?;
    }

    if !opts.dry_run {
        let index_path = output_dir.join("quantize_index.json");
        let index_record = QuantizeIndex {
            quarot_seed: Some(opts.rotation_seed),
            tensors: index_entries,
            // This converter only ever produces offline residual-rotation
            // (V0) artifacts today — no online-rotation recipe is produced
            // here yet.
            online: None,
            artifact_version: None,
            // #1103: every artifact this function writes starts
            // Unpromoted — the forward-equivalence gate above verified
            // rotation correctness, not quantization quality. Only
            // `record_ppl_gate_result` can advance this state.
            promotion: PromotionRecord::unpromoted(),
        };
        let index_json = serde_json::to_string_pretty(&index_record).map_err(|e| {
            InferenceError::Inference(format!(
                "convert_quarot_qwen35: failed to serialize quantize_index.json: {e}"
            ))
        })?;
        fs::write(&index_path, index_json).map_err(|e| {
            InferenceError::Inference(format!(
                "convert_quarot_qwen35: failed to write {}: {e}",
                index_path.display()
            ))
        })?;

        let mut output_config_json = untie_word_embeddings_in_config_json(&config_json)?;
        output_config_json = inject_quarot_seed(&output_config_json, opts.rotation_seed)?;
        let out_config_path = output_dir.join("config.json");
        fs::write(&out_config_path, &output_config_json).map_err(|e| {
            InferenceError::Inference(format!(
                "convert_quarot_qwen35: failed to write {}: {e}",
                out_config_path.display()
            ))
        })?;
    }

    Ok(ConversionReport {
        planned_quantized,
        kept_f16,
        total_bytes_in,
        total_bytes_out,
        forward_equivalence,
        was_tied,
    })
}

/// Refuse two CLI footguns that would otherwise let a failed conversion
/// leave the caller with corrupted source artifacts or a half-stale
/// output directory. The caller skips this validator in dry-run because
/// dry-run produces no writes — the footguns cannot fire — and callers
/// may want to dry-run against an existing populated output location
/// or a placeholder that happens to equal the input directory.
///
/// 1. `input_dir` and `output_dir` resolving to the **same canonical
///    path** — the converter would write a mutated (untied)
///    `config.json` on top of the source, then if the gate later
///    refused, the user would be left with a broken source checkpoint.
///    The runtime loader then takes the untied branch and demands a
///    `lm_head.weight` that never reached disk.
/// 2. A pre-existing **non-empty `output_dir`** — refuse-on-fail
///    short-circuits before any new files are written, so stale `.q4`
///    artifacts from a previous run would survive a gate failure and
///    the runtime would still pick them up. The PR-documented invariant
///    is "absent or empty after a refuse"; enforce it by requiring
///    `output_dir` to be empty (or absent) before we start.
///
/// Both checks fire before tensors are loaded, so the cost of bailing
/// is just a stat call.
fn validate_output_dir_layout(input_dir: &Path, output_dir: &Path) -> Result<(), InferenceError> {
    let input_canon = fs::canonicalize(input_dir).map_err(|e| {
        InferenceError::Inference(format!(
            "validate_output_dir_layout: cannot canonicalize input_dir {}: {e}",
            input_dir.display()
        ))
    })?;
    if !output_dir.exists() {
        return Ok(());
    }
    let output_canon = fs::canonicalize(output_dir).map_err(|e| {
        InferenceError::Inference(format!(
            "validate_output_dir_layout: cannot canonicalize output_dir {}: {e}",
            output_dir.display()
        ))
    })?;
    if input_canon == output_canon {
        return Err(InferenceError::Inference(format!(
            "validate_output_dir_layout: input and output directories resolve to the same \
             path ({}); refusing to overwrite source artifacts. Pass a separate \
             --output-dir to avoid corrupting the input checkpoint.",
            input_canon.display()
        )));
    }
    let mut entries = fs::read_dir(output_dir).map_err(|e| {
        InferenceError::Inference(format!(
            "validate_output_dir_layout: cannot read output_dir {}: {e}",
            output_dir.display()
        ))
    })?;
    if entries.next().is_some() {
        return Err(InferenceError::Inference(format!(
            "validate_output_dir_layout: output_dir {} is not empty; refusing to mix \
             new conversion output with pre-existing files. Remove the directory or \
             pass a fresh path — a refused conversion must not leave a partial mix \
             of stale + new artifacts.",
            output_canon.display()
        )));
    }
    Ok(())
}

fn sanitize_tensor_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Write a `KHF1`-headed `.f16` file matching the convention used by
/// `bin/quantize_q4` for non-quantized weights.
///
/// Layout:
/// ```text
///   magic[4]   = "KHF1"
///   version[4] = 1
///   ndim[4]    = shape.len() as u32
///   dims[8*ndim] = each dim as u64
///   numel[8]   = data.len() as u64
///   payload    = numel × u16 (IEEE-754 f16, little-endian)
/// ```
///
/// Returns the total number of bytes written. f64 → f16 goes through
/// f32 to share the existing converter (rounding-aware) — sufficient
/// for the runtime's f16 fast path.
fn write_f16_file(
    path: &Path,
    source: &str,
    tensor_name: &str,
    data: &[f64],
    shape: &[usize],
) -> Result<usize, InferenceError> {
    // Delay file creation until narrowing is proven finite so a rejected
    // tensor cannot look like a completed KHF1 artifact.
    let mut validator =
        DecodedTensorValidator::decoded_input(source, tensor_name, shape, "F16", data.len())?;
    let mut payload = Vec::with_capacity(data.len() * 2);
    for &value in data {
        let bits =
            q4_f32_to_finite_f16(value as f32).map_err(|bits| validator.reject_f16_bits(bits))?;
        validator.observe_finite();
        payload.extend_from_slice(&bits.to_le_bytes());
    }
    validator.finish()?;

    let mut file = fs::File::create(path).map_err(|e| {
        InferenceError::Inference(format!(
            "write_f16_file: failed to create {}: {e}",
            path.display()
        ))
    })?;
    let mut bytes_written: usize = 0;

    let mut write_all = |buf: &[u8]| -> Result<(), InferenceError> {
        file.write_all(buf).map_err(|e| {
            InferenceError::Inference(format!(
                "write_f16_file: write failure on {}: {e}",
                path.display()
            ))
        })
    };

    write_all(b"KHF1")?;
    bytes_written += 4;
    write_all(&1u32.to_le_bytes())?;
    bytes_written += 4;
    write_all(&(shape.len() as u32).to_le_bytes())?;
    bytes_written += 4;
    for &dim in shape {
        write_all(&(dim as u64).to_le_bytes())?;
        bytes_written += 8;
    }
    write_all(&(data.len() as u64).to_le_bytes())?;
    bytes_written += 8;

    write_all(&payload)?;
    bytes_written += payload.len();

    Ok(bytes_written)
}

#[cfg(test)]
mod tests;
