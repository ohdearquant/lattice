//! High-level QuaRot Qwen3.5 conversion (ADR-044 step 3c-5).
//!
//! [`convert_quarot_qwen35`] reads `config.json` + SafeTensors from
//! `input_dir`, runs the full pipeline
//! (`materialize_lm_head` → `fuse_rmsnorms` → `absorb_rotations` →
//! [`assert_forward_equivalence_qwen35`]) entirely in f64, and on success
//! writes the converted model to `output_dir`:
//!
//! - Planned (rotated) tensors → `<sanitized>.q4` via
//!   [`write_q4_tensor`] against an fd-bound exclusive no-follow file.
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
use std::os::fd::AsRawFd;
use std::path::Path;

use crate::error::InferenceError;
use crate::model::qwen35::qwen_required_tensor_names;
use crate::model::qwen35_config::Qwen35Config;
use crate::quant::quarot::forward_equivalence::{
    ForwardEquivalenceConfig, ForwardEquivalenceReport, assert_forward_equivalence_qwen35,
};
use crate::quant::quarot::hadamard::RandomizedHadamard;
use crate::quant::quarot::io::QuarotTensorReader;
use crate::quant::quarot::lm_head::{
    materialize_lm_head_for_qwen35, qwen35_final_norm_fusion_target,
    untie_word_embeddings_in_config_json,
};
use crate::quant::quarot::pipeline::{
    TensorEntry, absorb_rotations, fuse_rmsnorms, load_tensors_f64,
};
use crate::quant::quarot::plan::RotationPlan;
use crate::quant::quarot::rmsnorm_fusion::qwen35_per_layer_fusion_plan;
use crate::weights::q4_weights::{
    create_file_at, create_output_file, open_dir_nofollow, quantize_f64_to_q4, write_q4_tensor,
};

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

/// Wire format for `quantize_index.json` produced by `convert_quarot_qwen35`.
///
/// ADR-051 §"quantize_quarot Binary Change": the rotation seed is the runtime's
/// authoritative source for reconstructing the QuaRot Hadamard sign vector. It
/// lives next to the tensor index so a loader can recover it without parsing
/// `config.json`. `quantize_index.json` from older builds (no `quarot_seed`)
/// remains compatible — the field is `Option<u64>`.
#[derive(serde::Serialize, serde::Deserialize)]
struct QuantizeIndex {
    #[serde(skip_serializing_if = "Option::is_none")]
    quarot_seed: Option<u64>,
    tensors: Vec<IndexEntry>,
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
/// does not match the strict schema is `Err`. A file that exists and
/// doesn't parse is evidence of truncation/corruption/tampering, not a
/// legitimate "no rotation" artifact, and silently falling back to the
/// legacy seed (or no rotation at all) would apply the wrong rotation to a
/// checkpoint that was actually QuaRot-rotated, silently corrupting
/// inference output instead of refusing to load (#630 end-to-end
/// verification against the real qwen3.5-0.8b-q4 checkpoint).
#[cfg(any(test, feature = "metal-gpu"))]
pub(crate) fn read_quarot_seed_from_index(q4_dir: &Path) -> Result<Option<u64>, String> {
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
    Ok(index.quarot_seed)
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
    output_dir: &Path,
    output_dir_fd: Option<&std::os::fd::OwnedFd>,
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
        // Validate representability unconditionally: a
        // dry run must reject exactly what the real write below would
        // reject, not silently accept a value that later overflows f16.
        let framed = encode_f16_payload(&file_name, name, &data, &shape)?;
        if !dry_run {
            let dirfd = output_dir_fd.ok_or_else(|| {
                InferenceError::Inference(
                    "write_mtp_weights_quarot: missing output directory fd for a non-dry-run write"
                        .to_string(),
                )
            })?;
            create_file_at(dirfd.as_raw_fd(), &file_name)
                .and_then(|mut f| f.write_all(&framed))
                .map_err(|e| {
                    InferenceError::Inference(format!(
                        "write_mtp_weights_quarot: failed to write {} in {}: {e}",
                        file_name,
                        output_dir.display()
                    ))
                })?;
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
    if cfg.is_moe() {
        return Err(InferenceError::Inference(
            "convert_quarot_qwen35: MoE configs are deferred to v1 (see ADR-044 §Out of v0)"
                .to_string(),
        ));
    }

    let reader = QuarotTensorReader::open(input_dir)?;
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
        materialize_lm_head_for_qwen35(&mut working_set, &cfg)?;
    }

    let original_snapshot = working_set.clone();

    let mut fusion_plan = qwen35_per_layer_fusion_plan(&cfg)?;
    fusion_plan.push(qwen35_final_norm_fusion_target());
    let rotation_plan = RotationPlan::qwen35_residual_stream_linear_layers();
    let rotation = RandomizedHadamard::new(opts.rotation_seed, cfg.hidden_size)?;

    fuse_rmsnorms(&mut working_set, &fusion_plan)?;
    absorb_rotations(&mut working_set, &rotation_plan, &rotation)?;

    let forward_equivalence = assert_forward_equivalence_qwen35(
        &original_snapshot,
        &working_set,
        &cfg,
        &rotation,
        &ForwardEquivalenceConfig {
            num_probe_tokens: opts.num_probe_tokens,
            tolerance: opts.tolerance,
            seed: opts.rotation_seed,
        },
    )?;

    // Opened once, right after creation, and held for every write below:
    // binding the whole write session to one fd means an ancestor or
    // output-dir swap after this point cannot redirect any later write,
    // unlike re-joining and reopening `output_dir` by path for each file.
    let output_dir_fd = if !opts.dry_run {
        fs::create_dir_all(output_dir).map_err(|e| {
            InferenceError::Inference(format!(
                "convert_quarot_qwen35: failed to create output directory {}: {e}",
                output_dir.display()
            ))
        })?;
        Some(open_dir_nofollow(output_dir).map_err(|e| {
            InferenceError::Inference(format!(
                "convert_quarot_qwen35: failed to open output directory {}: {e}",
                output_dir.display()
            ))
        })?)
    } else {
        None
    };

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
            // Validate Q4 representability unconditionally: quantize_f64_to_q4
            // runs the same
            // encode_finite_q4_metadata scale+bias underflow guard that the
            // real write below depends on, so a dry run rejects exactly what
            // the real run would reject (e.g. a constant block whose bias
            // underflows to f16 zero) instead of only surfacing on the next
            // real run. Only the file write is gated on `!opts.dry_run`.
            let q4 = quantize_f64_to_q4(&entry.data, &entry.shape)?;
            let file_name = format!("{sanitized}.q4");
            if !opts.dry_run {
                let dirfd = output_dir_fd.as_ref().ok_or_else(|| {
                    InferenceError::Inference(
                        "convert_quarot_qwen35: missing output directory fd for a non-dry-run write"
                            .to_string(),
                    )
                })?;
                create_file_at(dirfd.as_raw_fd(), &file_name)
                    .and_then(|mut f| write_q4_tensor(&mut f, &q4))
                    .map_err(|e| {
                        InferenceError::Inference(format!(
                            "convert_quarot_qwen35: failed to write {file_name} in {}: {e}",
                            output_dir.display()
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
            let file_name = format!("{sanitized}.f16");
            // Validate representability unconditionally:
            // dry-run must reject exactly what the real write below would
            // reject (e.g. a finite kept value that overflows f16 narrowing),
            // not silently accept it and let the real run discover the
            // rejection later.
            let framed = encode_f16_payload(&file_name, name, &entry.data, &entry.shape)?;
            if !opts.dry_run {
                let dirfd = output_dir_fd.as_ref().ok_or_else(|| {
                    InferenceError::Inference(
                        "convert_quarot_qwen35: missing output directory fd for a non-dry-run write"
                            .to_string(),
                    )
                })?;
                create_file_at(dirfd.as_raw_fd(), &file_name)
                    .and_then(|mut f| f.write_all(&framed))
                    .map_err(|e| {
                        InferenceError::Inference(format!(
                            "convert_quarot_qwen35: failed to write {file_name} in {}: {e}",
                            output_dir.display()
                        ))
                    })?;
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
            output_dir,
            output_dir_fd.as_ref(),
            opts.dry_run,
            &mut index_entries,
            &mut kept_f16,
            &mut planned_quantized,
            &mut total_bytes_out,
        )?;
    }

    if !opts.dry_run {
        let dirfd = output_dir_fd.as_ref().ok_or_else(|| {
            InferenceError::Inference(
                "convert_quarot_qwen35: missing output directory fd for a non-dry-run write"
                    .to_string(),
            )
        })?;

        let index_record = QuantizeIndex {
            quarot_seed: Some(opts.rotation_seed),
            tensors: index_entries,
        };
        let index_json = serde_json::to_string_pretty(&index_record).map_err(|e| {
            InferenceError::Inference(format!(
                "convert_quarot_qwen35: failed to serialize quantize_index.json: {e}"
            ))
        })?;
        create_file_at(dirfd.as_raw_fd(), "quantize_index.json")
            .and_then(|mut f| f.write_all(index_json.as_bytes()))
            .map_err(|e| {
                InferenceError::Inference(format!(
                    "convert_quarot_qwen35: failed to write quantize_index.json in {}: {e}",
                    output_dir.display()
                ))
            })?;

        let mut output_config_json = untie_word_embeddings_in_config_json(&config_json)?;
        output_config_json = inject_quarot_seed(&output_config_json, opts.rotation_seed)?;
        create_file_at(dirfd.as_raw_fd(), "config.json")
            .and_then(|mut f| f.write_all(output_config_json.as_bytes()))
            .map_err(|e| {
                InferenceError::Inference(format!(
                    "convert_quarot_qwen35: failed to write config.json in {}: {e}",
                    output_dir.display()
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

/// Validate and encode one tensor's `KHF1`-framed `.f16` payload entirely in
/// memory, without touching the filesystem.
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
/// [`write_f16_file`] (the real-write path) and every dry-run byte-accounting
/// path call this same function, so a dry run rejects exactly what a real
/// write would reject — a finite kept tensor that overflows f16 narrowing
/// (e.g. `f32::MAX`) errors here regardless of `dry_run`, instead of only
/// surfacing on the next real run (dry-run/real-run representability
/// parity).
///
/// `source` labels the origin in error messages (a destination path for a
/// real write, or a synthetic label — e.g. the tensor name — when called
/// from a dry-run path with no output path yet). `tensor_name` is always the
/// original source tensor name (not derived from `source`), so a validation
/// failure traces back to the source tensor even when the output filename
/// has been sanitized.
///
/// # Errors
///
/// Returns an error when the tensor shape is inconsistent, a value is
/// non-finite before or after f16 encoding, or numel/shape overflow `usize`.
pub fn encode_f16_payload(
    source: &str,
    tensor_name: &str,
    data: &[f64],
    shape: &[usize],
) -> Result<Vec<u8>, InferenceError> {
    let mut payload = Vec::new();
    crate::weights::ingress::validate_ingested_tensor(
        crate::weights::ingress::IngestedTensor::encode_f16(
            source,
            tensor_name,
            shape,
            data,
            &mut payload,
        ),
    )?;

    let mut framed = Vec::with_capacity(4 + 4 + 4 + 8 * shape.len() + 8 + payload.len());
    framed.extend_from_slice(b"KHF1");
    framed.extend_from_slice(&1u32.to_le_bytes());
    framed.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for &dim in shape {
        framed.extend_from_slice(&(dim as u64).to_le_bytes());
    }
    framed.extend_from_slice(&(data.len() as u64).to_le_bytes());
    framed.extend_from_slice(&payload);
    Ok(framed)
}

/// Write a `KHF1`-headed `.f16` file matching the convention used by
/// `bin/quantize_q4` for non-quantized weights.
///
/// Returns the total number of bytes written. Validation and f64 → f16
/// conversion (via [`encode_f16_payload`]) finish before the destination is
/// created, so an invalid tensor cannot appear as a completed output
/// artifact.
///
/// # Errors
///
/// Returns an error when the tensor shape is inconsistent, a value is
/// non-finite before or after f16 encoding, or the output cannot be written.
pub fn write_f16_file(
    path: &Path,
    tensor_name: &str,
    data: &[f64],
    shape: &[usize],
) -> Result<usize, InferenceError> {
    let source = path.display().to_string();
    let framed = encode_f16_payload(&source, tensor_name, data, shape)?;

    let mut file = create_output_file(path).map_err(|e| {
        InferenceError::Inference(format!(
            "write_f16_file: failed to create {}: {e}",
            path.display()
        ))
    })?;
    file.write_all(&framed).map_err(|e| {
        InferenceError::Inference(format!(
            "write_f16_file: write failure on {}: {e}",
            path.display()
        ))
    })?;

    Ok(framed.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen35_config::{LayerType, compute_layer_types};
    use serde_json::Value;
    use std::path::PathBuf;

    // ------------------------------------------------------------------
    // Tiny test config + SafeTensors writer (local to this module to
    // avoid cross-module visibility juggling; mirrors the helper in
    // `io.rs` tests).
    // ------------------------------------------------------------------

    /// Tiny Qwen3.5 cfg with power-of-2 hidden=8, 2 layers (one GDN +
    /// one GQA), vocab=4. Tuned for tractable f64 matmul tests.
    fn tiny_cfg(tied: bool) -> Qwen35Config {
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
        cfg.linear_conv_kernel_dim = 4;
        cfg.full_attention_interval = 2;
        cfg.layer_types = compute_layer_types(cfg.num_hidden_layers, cfg.full_attention_interval);
        cfg.layer_mask = vec![true; cfg.num_hidden_layers];
        cfg.tie_word_embeddings = tied;
        cfg.rms_norm_eps = 1e-6;
        // head_dim (4) * partial_rotary_factor must derive an even rope_dim >= 2
        // (Qwen35Config parse guard, #401); 4 * 0.5 = 2.
        cfg.partial_rotary_factor = 0.5;
        cfg.rope_theta = 1_000_000.0;
        cfg.max_position_embeddings = 1024;
        cfg.eos_token_id = 3;
        cfg
    }

    /// Build a config.json string that parses back to a tiny test cfg.
    /// HF style: top-level `tie_word_embeddings` + nested `text_config`.
    /// MoE-specific fields (`num_experts`, etc.) are propagated when set
    /// so the converter's `is_moe()` reject path can be exercised.
    fn tiny_config_json(cfg: &Qwen35Config) -> String {
        let layer_types: Vec<Value> = cfg
            .layer_types
            .iter()
            .map(|t| match t {
                LayerType::FullAttention => Value::String("full_attention".into()),
                LayerType::LinearAttention => Value::String("linear_attention".into()),
            })
            .collect();
        let mut text_config = serde_json::Map::new();
        text_config.insert("hidden_size".into(), Value::from(cfg.hidden_size));
        text_config.insert(
            "num_hidden_layers".into(),
            Value::from(cfg.num_hidden_layers),
        );
        text_config.insert("vocab_size".into(), Value::from(cfg.vocab_size));
        text_config.insert(
            "intermediate_size".into(),
            Value::from(cfg.intermediate_size),
        );
        text_config.insert("rms_norm_eps".into(), Value::from(cfg.rms_norm_eps));
        text_config.insert(
            "num_attention_heads".into(),
            Value::from(cfg.num_attention_heads),
        );
        text_config.insert(
            "num_key_value_heads".into(),
            Value::from(cfg.num_key_value_heads),
        );
        text_config.insert("head_dim".into(), Value::from(cfg.head_dim));
        text_config.insert("rope_theta".into(), Value::from(cfg.rope_theta));
        text_config.insert(
            "partial_rotary_factor".into(),
            Value::from(cfg.partial_rotary_factor),
        );
        text_config.insert(
            "linear_num_key_heads".into(),
            Value::from(cfg.linear_num_key_heads),
        );
        if let Some(v) = cfg.linear_num_value_heads {
            text_config.insert("linear_num_value_heads".into(), Value::from(v));
        }
        text_config.insert(
            "linear_key_head_dim".into(),
            Value::from(cfg.linear_key_head_dim),
        );
        text_config.insert(
            "linear_value_head_dim".into(),
            Value::from(cfg.linear_value_head_dim),
        );
        text_config.insert(
            "linear_conv_kernel_dim".into(),
            Value::from(cfg.linear_conv_kernel_dim),
        );
        text_config.insert(
            "tie_word_embeddings".into(),
            Value::from(cfg.tie_word_embeddings),
        );
        text_config.insert(
            "full_attention_interval".into(),
            Value::from(cfg.full_attention_interval),
        );
        text_config.insert("layer_types".into(), Value::Array(layer_types));
        text_config.insert("eos_token_id".into(), Value::from(cfg.eos_token_id));
        text_config.insert(
            "max_position_embeddings".into(),
            Value::from(cfg.max_position_embeddings),
        );
        // MoE knobs only when present.
        if let Some(v) = cfg.num_experts {
            text_config.insert("num_experts".into(), Value::from(v));
        }
        if let Some(v) = cfg.num_experts_per_tok {
            text_config.insert("num_experts_per_tok".into(), Value::from(v));
        }
        if let Some(v) = cfg.moe_intermediate_size {
            text_config.insert("moe_intermediate_size".into(), Value::from(v));
        }
        if let Some(v) = cfg.shared_expert_intermediate_size {
            text_config.insert("shared_expert_intermediate_size".into(), Value::from(v));
        }

        serde_json::to_string_pretty(&serde_json::json!({
            "tie_word_embeddings": cfg.tie_word_embeddings,
            "text_config": Value::Object(text_config),
        }))
        .unwrap()
    }

    fn f32_to_bf16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let lsb = (bits >> 16) & 1;
        let rounding_bias = 0x7fff + lsb;
        ((bits.wrapping_add(rounding_bias)) >> 16) as u16
    }

    /// Minimal SafeTensors writer for the converter's input fixture.
    /// All tensors are stored as F32 (so `read_tensor_f64` round-trips
    /// without lossy conversions and the per-tensor matrix-equivalence
    /// gate stays within f64 noise).
    fn write_test_safetensors(path: &Path, tensors: &[(&str, Vec<usize>, &[f64])]) {
        let mut header = serde_json::Map::new();
        let mut payload: Vec<u8> = Vec::new();
        for (name, shape, values) in tensors {
            assert_eq!(values.len(), shape.iter().product::<usize>());
            let start = payload.len();
            for &v in *values {
                payload.extend_from_slice(&(v as f32).to_le_bytes());
            }
            let end = payload.len();
            let mut entry = serde_json::Map::new();
            entry.insert("dtype".into(), Value::String("F32".into()));
            entry.insert(
                "shape".into(),
                Value::Array(shape.iter().map(|d| Value::from(*d as u64)).collect()),
            );
            entry.insert(
                "data_offsets".into(),
                Value::Array(vec![Value::from(start as u64), Value::from(end as u64)]),
            );
            header.insert((*name).to_string(), Value::Object(entry));
        }
        let header_str = serde_json::to_string(&Value::Object(header)).unwrap();
        let mut file = fs::File::create(path).unwrap();
        file.write_all(&(header_str.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header_str.as_bytes()).unwrap();
        file.write_all(&payload).unwrap();
    }

    fn synth_data(n: usize, seed: u64) -> Vec<f64> {
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

    /// Write every required tensor for `cfg` to a single safetensors file.
    fn write_required_tensors_for(cfg: &Qwen35Config, path: &Path, seed: u64) {
        write_required_tensors_for_with_override(cfg, path, seed, None);
    }

    /// Like [`write_required_tensors_for`], but replaces the data of the
    /// tensor named `override_entry.0` with `override_entry.1` before
    /// writing. Used to force a specific planned (rotated) tensor to a
    /// controlled magnitude for representability-boundary tests.
    fn write_required_tensors_for_with_override(
        cfg: &Qwen35Config,
        path: &Path,
        seed: u64,
        override_entry: Option<(&str, Vec<f64>)>,
    ) {
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let intermediate = cfg.intermediate_size;
        let head_dim = cfg.head_dim;
        let full_q_dim = cfg.full_q_dim();
        let full_kv_dim = cfg.full_kv_dim();
        let linear_qkv_dim = cfg.linear_qkv_dim();
        let linear_output_dim = cfg.linear_output_dim();
        let linear_num_heads = cfg.linear_num_key_heads;
        let kernel = cfg.linear_conv_kernel_dim;

        // Build a vector of (name, shape, data) tuples then borrow-call the writer.
        let mut entries: Vec<(String, Vec<usize>, Vec<f64>)> = Vec::new();
        let mut s = seed;
        let mut next = |n: usize| -> Vec<f64> {
            s = s.wrapping_add(1);
            synth_data(n, s)
        };

        entries.push((
            "model.language_model.embed_tokens.weight".to_string(),
            vec![vocab, hidden],
            next(vocab * hidden),
        ));
        entries.push((
            "model.language_model.norm.weight".to_string(),
            vec![hidden],
            next(hidden),
        ));
        if !cfg.tie_word_embeddings {
            entries.push((
                "lm_head.weight".to_string(),
                vec![vocab, hidden],
                next(vocab * hidden),
            ));
        }

        for i in 0..cfg.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{i}");
            entries.push((
                format!("{prefix}.input_layernorm.weight"),
                vec![hidden],
                next(hidden),
            ));
            entries.push((
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![hidden],
                next(hidden),
            ));

            if cfg.is_full_attention(i) {
                entries.push((
                    format!("{prefix}.self_attn.q_proj.weight"),
                    vec![2 * full_q_dim, hidden],
                    next(2 * full_q_dim * hidden),
                ));
                entries.push((
                    format!("{prefix}.self_attn.k_proj.weight"),
                    vec![full_kv_dim, hidden],
                    next(full_kv_dim * hidden),
                ));
                entries.push((
                    format!("{prefix}.self_attn.v_proj.weight"),
                    vec![full_kv_dim, hidden],
                    next(full_kv_dim * hidden),
                ));
                entries.push((
                    format!("{prefix}.self_attn.o_proj.weight"),
                    vec![hidden, full_q_dim],
                    next(hidden * full_q_dim),
                ));
                entries.push((
                    format!("{prefix}.self_attn.q_norm.weight"),
                    vec![head_dim],
                    next(head_dim),
                ));
                entries.push((
                    format!("{prefix}.self_attn.k_norm.weight"),
                    vec![head_dim],
                    next(head_dim),
                ));
            } else {
                entries.push((
                    format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    vec![linear_qkv_dim, hidden],
                    next(linear_qkv_dim * hidden),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.in_proj_z.weight"),
                    vec![linear_output_dim, hidden],
                    next(linear_output_dim * hidden),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.in_proj_b.weight"),
                    vec![linear_num_heads, hidden],
                    next(linear_num_heads * hidden),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.in_proj_a.weight"),
                    vec![linear_num_heads, hidden],
                    next(linear_num_heads * hidden),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.A_log"),
                    vec![linear_num_heads],
                    next(linear_num_heads),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.dt_bias"),
                    vec![linear_num_heads],
                    next(linear_num_heads),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.conv1d.weight"),
                    vec![linear_qkv_dim, 1, kernel],
                    next(linear_qkv_dim * kernel),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.norm.weight"),
                    vec![linear_output_dim],
                    next(linear_output_dim),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.out_proj.weight"),
                    vec![hidden, linear_output_dim],
                    next(hidden * linear_output_dim),
                ));
            }

            entries.push((
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![intermediate, hidden],
                next(intermediate * hidden),
            ));
            entries.push((
                format!("{prefix}.mlp.up_proj.weight"),
                vec![intermediate, hidden],
                next(intermediate * hidden),
            ));
            entries.push((
                format!("{prefix}.mlp.down_proj.weight"),
                vec![hidden, intermediate],
                next(hidden * intermediate),
            ));
        }

        if let Some((name, data)) = &override_entry {
            let entry = entries
                .iter_mut()
                .find(|(n, _, _)| n == name)
                .unwrap_or_else(|| panic!("override target tensor `{name}` not found in fixture"));
            assert_eq!(
                entry.2.len(),
                data.len(),
                "override data length must match fixture tensor `{name}`'s numel"
            );
            entry.2 = data.clone();
        }

        let borrowed: Vec<(&str, Vec<usize>, &[f64])> = entries
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), d.as_slice()))
            .collect();
        write_test_safetensors(path, &borrowed);
    }

    fn write_input_dir(cfg: &Qwen35Config, dir: &Path, seed: u64) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("config.json"), tiny_config_json(cfg)).unwrap();
        write_required_tensors_for(cfg, &dir.join("model.safetensors"), seed);
    }

    /// Like [`write_input_dir`], but overrides one required tensor's data
    /// before writing — see [`write_required_tensors_for_with_override`].
    fn write_input_dir_with_override(
        cfg: &Qwen35Config,
        dir: &Path,
        seed: u64,
        override_entry: (&str, Vec<f64>),
    ) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("config.json"), tiny_config_json(cfg)).unwrap();
        write_required_tensors_for_with_override(
            cfg,
            &dir.join("model.safetensors"),
            seed,
            Some(override_entry),
        );
    }

    // ------------------------------------------------------------------
    // Happy path
    // ------------------------------------------------------------------

    #[test]
    #[cfg(unix)]
    fn create_output_file_refuses_a_preexisting_symlink() {
        // Every converter output write (`write_f16_file`, the Q4 write in
        // `convert_quarot_qwen35`, `quantize_index.json`, `config.json`)
        // routes through `create_output_file`. `validate_output_dir_layout`
        // only checks emptiness at the very start of the run, but the real
        // writes happen much later — after loading every tensor and running
        // the forward-equivalence gate — so a concurrent actor has a wide
        // window to plant a symlink at a fixed output filename pointing at
        // an external victim file. Mutation-sensitive: reverting
        // `create_output_file` to a plain `fs::File::create` makes this
        // test fail — the victim file would be truncated instead of the
        // open being refused.
        let tmp = tempfile::tempdir().unwrap();
        let victim = tmp.path().join("victim.txt");
        fs::write(&victim, b"do-not-touch").unwrap();
        let planted = tmp.path().join("config.json");
        std::os::unix::fs::symlink(&victim, &planted).unwrap();

        let err = create_output_file(&planted)
            .expect_err("a pre-existing symlink at the output path must be refused");
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::AlreadyExists,
            "unexpected error kind: {err}"
        );
        assert_eq!(
            fs::read(&victim).unwrap(),
            b"do-not-touch",
            "victim file reached through the planted symlink must be untouched"
        );
    }

    #[test]
    fn convert_quarot_qwen35_tied_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        let cfg = tiny_cfg(true);
        write_input_dir(&cfg, &input, 1);

        let report = convert_quarot_qwen35(
            &input,
            &output,
            &ConversionOptions {
                rotation_seed: 0xC0FFEE,
                tolerance: 1e-5,
                num_probe_tokens: 2,
                dry_run: false,
            },
        )
        .unwrap();

        assert!(report.was_tied);
        assert!(report.planned_quantized > 0);
        assert!(report.kept_f16 > 0);
        assert!(report.total_bytes_out > 0);
        assert!(report.forward_equivalence.max_abs_error <= 1e-5);

        assert!(output.join("config.json").exists());
        assert!(output.join("quantize_index.json").exists());

        // The materialized lm_head is rotated, so its .q4 file must be on disk
        // even though the tied input had no lm_head.weight tensor.
        let lm_head_q4 = output.join("lm_head_weight.q4");
        assert!(
            lm_head_q4.exists(),
            "lm_head .q4 should exist: {lm_head_q4:?}"
        );

        // Reload config and verify the untie flip survived JSON serialization.
        let out_cfg_str = fs::read_to_string(output.join("config.json")).unwrap();
        let out_cfg = Qwen35Config::from_config_json_str(&out_cfg_str).unwrap();
        assert!(
            !out_cfg.tie_word_embeddings,
            "output config must be untied after tied-input conversion"
        );

        // Index json contains every working-set tensor and the rotation seed.
        let idx_str = fs::read_to_string(output.join("quantize_index.json")).unwrap();
        let idx: serde_json::Value = serde_json::from_str(&idx_str).unwrap();
        let tensors = idx
            .get("tensors")
            .and_then(|v| v.as_array())
            .expect("quantize_index.json must have a `tensors` array");
        assert_eq!(tensors.len(), report.planned_quantized + report.kept_f16);
        assert!(
            idx.get("quarot_seed")
                .and_then(serde_json::Value::as_u64)
                .is_some(),
            "quantize_index.json must carry quarot_seed (ADR-051 contract)"
        );
    }

    #[test]
    fn convert_quarot_qwen35_untied_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        let cfg = tiny_cfg(false);
        write_input_dir(&cfg, &input, 2);

        let report = convert_quarot_qwen35(
            &input,
            &output,
            &ConversionOptions {
                rotation_seed: 0xFEED_FACE,
                tolerance: 1e-5,
                num_probe_tokens: 2,
                dry_run: false,
            },
        )
        .unwrap();

        assert!(!report.was_tied);
        assert!(report.planned_quantized > 0);
        assert!(output.join("config.json").exists());
        let out_cfg_str = fs::read_to_string(output.join("config.json")).unwrap();
        let out_cfg = Qwen35Config::from_config_json_str(&out_cfg_str).unwrap();
        assert!(!out_cfg.tie_word_embeddings);
    }

    // ------------------------------------------------------------------
    // Dry-run + refuse-on-fail + early-error contract
    // ------------------------------------------------------------------

    #[test]
    fn convert_quarot_qwen35_dry_run_writes_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        let cfg = tiny_cfg(true);
        write_input_dir(&cfg, &input, 3);

        let report = convert_quarot_qwen35(
            &input,
            &output,
            &ConversionOptions {
                rotation_seed: 0xDEADBEEF,
                tolerance: 1e-5,
                num_probe_tokens: 2,
                dry_run: true,
            },
        )
        .unwrap();

        // Dry-run now computes real byte counts and tensor counts (so the
        // Studio can show a meaningful compression ratio). The counts must be
        // positive and identical to what a real write would produce.
        assert!(
            report.planned_quantized > 0,
            "dry-run must report planned_quantized > 0"
        );
        assert!(report.kept_f16 > 0, "dry-run must report kept_f16 > 0");
        assert!(
            report.total_bytes_out > 0,
            "dry-run must report total_bytes_out > 0"
        );
        assert!(report.forward_equivalence.max_abs_error <= 1e-5);
        assert!(
            !output.exists(),
            "dry-run must not create the output directory"
        );
    }

    /// The Q4 quantized-tensor branch: the
    /// real-write path (`quantize_f64_to_q4` → `create_output_file` +
    /// `write_q4_tensor`) validates Q4
    /// representability via `encode_finite_q4_metadata`'s two-sided
    /// scale+bias underflow guard, but that call used to be gated behind
    /// `if !opts.dry_run`, so dry-run silently skipped it. A planned tensor
    /// whose post-rotation magnitude underflows Q4's symmetric-mode
    /// scale/bias to f16 zero (B4's constant-block repro, scaled to a whole
    /// tensor here since Hadamard rotation mixes every element along a
    /// planned tensor's rotated axis) must be rejected identically in
    /// dry-run and the real run.
    ///
    /// Mutation-sensitive: re-gating `quantize_f64_to_q4` behind `if
    /// !opts.dry_run` (the pre-fix state) makes the `dry_run: true`
    /// iteration of this test pass silently through validation, diverging
    /// from the `dry_run: false` iteration's `Err`.
    #[test]
    fn dry_run_rejects_q4_bias_underflow_that_the_real_run_would_also_reject() {
        let cfg = tiny_cfg(true);
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let tensor_name = "model.language_model.layers.0.mlp.down_proj.weight";

        for dry_run in [true, false] {
            let tmp = tempfile::tempdir().unwrap();
            let input = tmp.path().join("input");
            let output = tmp.path().join("output");
            // A uniformly tiny planned tensor: Hadamard rotation is
            // orthogonal (norm-preserving) along the rotated axis, so a
            // uniform 1e-9 input stays bounded within a small constant
            // factor of 1e-9 after rotation — well under the ~5.96e-8 f16
            // subnormal floor that `encode_finite_q4_metadata` guards
            // (symmetric-mode scale = abs_max/7, bias = -8*scale; both
            // derive from the same tiny abs_max and underflow together).
            let tiny = vec![1e-9_f64; hidden * intermediate];
            write_input_dir_with_override(&cfg, &input, 200, (tensor_name, tiny));

            let err = convert_quarot_qwen35(
                &input,
                &output,
                &ConversionOptions {
                    rotation_seed: 0xC0FFEE,
                    tolerance: 1e-5,
                    num_probe_tokens: 2,
                    dry_run,
                },
            )
            .expect_err(&format!(
                "a Q4 block whose scale/bias underflows to f16 zero must be rejected \
                 (dry_run={dry_run})"
            ));
            assert!(
                err.to_string().contains("underflows to f16 zero"),
                "unexpected error (dry_run={dry_run}): {err}"
            );
            if dry_run {
                // Dry-run performs no writes at all (`opts.dry_run` gates
                // every `fs::create_dir_all`/write in the function), so the
                // output directory must never come into existence.
                assert!(
                    !output.exists(),
                    "a dry-run rejection must not create output_dir"
                );
            }
            // For the real run, `output_dir` is created before the
            // per-tensor loop runs (so earlier-sorted tensors can stream
            // their writes), so a mid-loop rejection can leave it created
            // (possibly holding earlier tensors' output) — that pre-existing
            // real-run behavior is orthogonal to the dry-run/real-run
            // representability *parity* this test guards, which is: both
            // must reject the SAME input with the SAME error.
        }

        // A representable planned tensor passes both dry-run and the real run.
        for dry_run in [true, false] {
            let tmp = tempfile::tempdir().unwrap();
            let input = tmp.path().join("input");
            let output = tmp.path().join("output");
            write_input_dir(&cfg, &input, 201);

            let report = convert_quarot_qwen35(
                &input,
                &output,
                &ConversionOptions {
                    rotation_seed: 0xC0FFEE,
                    tolerance: 1e-5,
                    num_probe_tokens: 2,
                    dry_run,
                },
            )
            .unwrap_or_else(|e| {
                panic!("representable tensor must be accepted (dry_run={dry_run}): {e}")
            });
            assert!(report.planned_quantized > 0);
        }
    }

    /// Refuse-on-fail: tolerance set absurdly tight forces the gate to
    /// refuse. The converter must propagate the gate's `Err` and leave
    /// the output directory empty (or absent).
    #[test]
    fn convert_quarot_qwen35_refuses_when_tolerance_unmet() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        let cfg = tiny_cfg(true);
        write_input_dir(&cfg, &input, 4);

        let err = convert_quarot_qwen35(
            &input,
            &output,
            &ConversionOptions {
                rotation_seed: 0xAB12_34CD,
                tolerance: 0.0_f64.next_up(), // smallest positive — chain probe noise exceeds this
                num_probe_tokens: 2,
                dry_run: false,
            },
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("forward-equivalence refused") || msg.contains("exceeds tolerance"),
            "unexpected error: {msg}"
        );
        assert!(
            !output.exists(),
            "refused conversion must not create the output directory"
        );
    }

    #[test]
    fn convert_quarot_qwen35_errors_when_config_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        fs::create_dir_all(&input).unwrap();
        // No config.json written.

        let err =
            convert_quarot_qwen35(&input, &output, &ConversionOptions::default()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("config.json"), "unexpected error: {msg}");
    }

    #[test]
    fn convert_quarot_qwen35_rejects_non_power_of_two_hidden() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        let mut cfg = tiny_cfg(true);
        cfg.hidden_size = 10; // not power of 2
        fs::create_dir_all(&input).unwrap();
        fs::write(input.join("config.json"), tiny_config_json(&cfg)).unwrap();
        // No safetensors needed; reject happens before tensor load.

        let err =
            convert_quarot_qwen35(&input, &output, &ConversionOptions::default()).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("hidden_size=10") && msg.contains("power of 2"),
            "unexpected error: {msg}"
        );
        assert!(!output.exists());
    }

    #[test]
    fn convert_quarot_qwen35_rejects_moe_config() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        // Power-of-2 hidden so the MoE reject path is the one we actually
        // hit (not the power-of-2 pre-check).
        let mut moe_cfg = tiny_cfg(true);
        moe_cfg.num_experts = Some(2);
        moe_cfg.num_experts_per_tok = Some(1);
        moe_cfg.moe_intermediate_size = Some(moe_cfg.intermediate_size);
        fs::create_dir_all(&input).unwrap();
        fs::write(input.join("config.json"), tiny_config_json(&moe_cfg)).unwrap();

        let err =
            convert_quarot_qwen35(&input, &output, &ConversionOptions::default()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("MoE"), "unexpected error: {msg}");
        assert!(
            !output.exists(),
            "MoE-rejected conversion must not create output dir"
        );
    }

    // ------------------------------------------------------------------
    // Dry-run / real-write byte-count parity
    // ------------------------------------------------------------------

    /// Correctness gate: dry_run=true and dry_run=false on the same model
    /// must produce identical total_bytes_out values (and both > 0).
    /// Also verifies that dry-run wrote no output files.
    #[test]
    fn dry_run_bytes_out_matches_real_write() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output_dry = tmp.path().join("output_dry");
        let output_real = tmp.path().join("output_real");
        let cfg = tiny_cfg(false); // untied — no lm_head materialization side-effect
        write_input_dir(&cfg, &input, 99);

        let opts = ConversionOptions {
            rotation_seed: 0xABCD_5678,
            tolerance: 1e-5,
            num_probe_tokens: 2,
            dry_run: false,
        };

        let dry_report = convert_quarot_qwen35(
            &input,
            &output_dry,
            &ConversionOptions {
                dry_run: true,
                ..opts.clone()
            },
        )
        .unwrap();

        let real_report = convert_quarot_qwen35(&input, &output_real, &opts).unwrap();

        // Primary correctness assertion: byte counts are equal.
        assert_eq!(
            dry_report.total_bytes_out, real_report.total_bytes_out,
            "dry-run total_bytes_out ({}) must equal real-write total_bytes_out ({})",
            dry_report.total_bytes_out, real_report.total_bytes_out,
        );
        // Both must be positive — a zero here means accounting is broken.
        assert!(
            dry_report.total_bytes_out > 0,
            "total_bytes_out must be > 0; got 0 (accounting is broken)"
        );
        // Tensor counts must also match.
        assert_eq!(
            dry_report.planned_quantized, real_report.planned_quantized,
            "planned_quantized mismatch between dry and real"
        );
        assert_eq!(
            dry_report.kept_f16, real_report.kept_f16,
            "kept_f16 mismatch between dry and real"
        );
        // Dry-run must not have created an output directory.
        assert!(
            !output_dry.exists(),
            "dry-run must not create the output directory"
        );

        // Non-circular guard: the reported total must equal the SUM of the
        // actual on-disk tensor file sizes. A dry==real check alone is circular
        // (both sides apply the same formula); this catches drift between the
        // byte formula and what write_f16_file / save_q4_file actually write.
        let mut on_disk: u64 = 0;
        for dent in std::fs::read_dir(&output_real).unwrap() {
            let path = dent.unwrap().path();
            if matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("q4") | Some("f16")
            ) {
                on_disk += std::fs::metadata(&path).unwrap().len();
            }
        }
        assert_eq!(
            real_report.total_bytes_out, on_disk,
            "reported total_bytes_out ({}) must equal summed on-disk .q4/.f16 file sizes ({})",
            real_report.total_bytes_out, on_disk,
        );
    }

    /// Repeat the byte-count parity check with the tied model (triggers
    /// lm_head materialization, which adds one extra planned tensor).
    #[test]
    fn dry_run_bytes_out_matches_real_write_tied() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output_dry = tmp.path().join("output_dry");
        let output_real = tmp.path().join("output_real");
        let cfg = tiny_cfg(true);
        write_input_dir(&cfg, &input, 100);

        let opts = ConversionOptions {
            rotation_seed: 0xFACE_CAFE,
            tolerance: 1e-5,
            num_probe_tokens: 2,
            dry_run: false,
        };

        let dry_report = convert_quarot_qwen35(
            &input,
            &output_dry,
            &ConversionOptions {
                dry_run: true,
                ..opts.clone()
            },
        )
        .unwrap();

        let real_report = convert_quarot_qwen35(&input, &output_real, &opts).unwrap();

        assert_eq!(
            dry_report.total_bytes_out, real_report.total_bytes_out,
            "tied: dry-run total_bytes_out ({}) must equal real-write total_bytes_out ({})",
            dry_report.total_bytes_out, real_report.total_bytes_out,
        );
        assert!(dry_report.total_bytes_out > 0);
        assert!(
            !output_dry.exists(),
            "dry-run must not create the output directory"
        );
    }

    // ------------------------------------------------------------------
    // Output format spot-checks
    // ------------------------------------------------------------------

    #[test]
    fn sanitize_tensor_name_replaces_dots_and_slashes() {
        assert_eq!(
            sanitize_tensor_name("model.layers.0.mlp.gate_proj.weight"),
            "model_layers_0_mlp_gate_proj_weight"
        );
        assert_eq!(sanitize_tensor_name("lm_head.weight"), "lm_head_weight");
        assert_eq!(sanitize_tensor_name("a/b\\c"), "a_b_c");
    }

    /// f16 file readable: header is `KHF1\1\ndim\dims\numel\payload`.
    #[test]
    fn f16_file_has_khf1_header_and_correct_size() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("test.f16");
        let data = vec![1.5_f64, -2.5, 0.25, -0.125];
        let shape = vec![2_usize, 2];
        let bytes_written = write_f16_file(&p, "test-tensor", &data, &shape).unwrap();
        let raw = fs::read(&p).unwrap();
        assert_eq!(&raw[0..4], b"KHF1");
        assert_eq!(u32::from_le_bytes(raw[4..8].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(raw[8..12].try_into().unwrap()), 2);
        assert_eq!(u64::from_le_bytes(raw[12..20].try_into().unwrap()), 2);
        assert_eq!(u64::from_le_bytes(raw[20..28].try_into().unwrap()), 2);
        assert_eq!(u64::from_le_bytes(raw[28..36].try_into().unwrap()), 4);
        // Payload: 4 × 2 bytes = 8.
        assert_eq!(raw.len(), 36 + 8);
        assert_eq!(bytes_written, raw.len());
    }

    #[test]
    fn f16_file_rejects_invalid_encoded_values_before_creation() {
        for (case, value) in [
            ("nan", f64::NAN),
            ("positive-infinity", f64::INFINITY),
            ("negative-infinity", f64::NEG_INFINITY),
            ("finite-f16-overflow", f32::MAX as f64),
        ] {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join(format!("{case}.f16"));
            let err = write_f16_file(&path, "test-tensor", &[value], &[1])
                .expect_err("invalid f16 output must be rejected");
            assert!(
                err.to_string().contains("non-finite"),
                "unexpected {case} error: {err}"
            );
            assert!(
                !path.exists(),
                "validation must finish before publishing {case} output"
            );
        }
    }

    /// The shared KHF1 encoder preserves signed zero and f16 subnormals.
    #[test]
    fn f16_file_preserves_signed_zero_and_subnormal_values() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("edge-values.f16");
        write_f16_file(&path, "test-tensor", &[0.0, -0.0, 1e-7, 1e-40], &[4]).unwrap();
        let raw = fs::read(path).unwrap();
        let payload = &raw[28..];
        let bits: Vec<u16> = payload
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        assert_eq!(bits[0], 0x0000);
        assert_eq!(bits[1], 0x8000);
        assert_ne!(bits[2], 0, "f16 subnormal must not flush to zero");
        assert_eq!(bits[3], 0);
    }

    /// f32_to_bf16_bits helper smoke check (used by the test fixture
    /// writer; not exercised by the converter itself).
    #[test]
    fn f32_to_bf16_bits_canonical_values() {
        assert_eq!(f32_to_bf16_bits(0.0), 0);
        assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
        assert_eq!(f32_to_bf16_bits(-1.0), 0xbf80);
    }

    // ------------------------------------------------------------------
    // Path-layout refuses
    // ------------------------------------------------------------------

    /// When input and output paths resolve to the same canonical path, the
    /// converter must refuse before any write would corrupt the source
    /// `config.json`.
    #[test]
    fn convert_quarot_qwen35_rejects_same_input_output_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let cfg = tiny_cfg(true);
        write_input_dir(&cfg, &input, 50);
        let config_before = fs::read(input.join("config.json")).unwrap();

        let err = convert_quarot_qwen35(
            &input,
            &input, // same path
            &ConversionOptions::default(),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("same path"), "unexpected error: {msg}");
        // Source config must be byte-identical after the rejection.
        let config_after = fs::read(input.join("config.json")).unwrap();
        assert_eq!(
            config_before, config_after,
            "rejected conversion must not have mutated the source config.json"
        );
    }

    /// Even when the two paths differ literally (e.g., trailing slash,
    /// symlink), canonicalization must still catch the equivalence.
    #[test]
    fn convert_quarot_qwen35_rejects_same_input_output_dir_via_trailing_slash() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let cfg = tiny_cfg(true);
        write_input_dir(&cfg, &input, 51);
        let same_with_slash = tmp.path().join("input/.");

        let err = convert_quarot_qwen35(&input, &same_with_slash, &ConversionOptions::default())
            .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("same path"), "unexpected error: {msg}");
    }

    /// A pre-existing non-empty output directory must trigger refusal
    /// before any conversion work, so a previously-written `.q4` artifact
    /// cannot survive a gate failure and be picked up by the runtime
    /// loader.
    #[test]
    fn convert_quarot_qwen35_rejects_non_empty_output_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        let cfg = tiny_cfg(true);
        write_input_dir(&cfg, &input, 52);
        fs::create_dir_all(&output).unwrap();
        let stale_path = output.join("stale_artifact.q4");
        fs::write(&stale_path, b"old-q4-bytes").unwrap();

        let err =
            convert_quarot_qwen35(&input, &output, &ConversionOptions::default()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not empty"), "unexpected error: {msg}");
        // The stale artifact must still be on disk (untouched), because
        // we never started writing — the operator owns cleanup.
        assert!(stale_path.exists(), "stale file must not be deleted");
        let bytes = fs::read(&stale_path).unwrap();
        assert_eq!(&bytes[..], b"old-q4-bytes");
    }

    /// Dry-run must NOT enforce the write-mode same-dir refuse. A CI
    /// probe that points `--output-dir` at the same place as
    /// `--model-dir` is harmless because dry-run writes nothing, and
    /// the gate value is still useful as a fast pipeline sanity pass.
    #[test]
    fn convert_quarot_qwen35_dry_run_ignores_same_output_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let cfg = tiny_cfg(true);
        write_input_dir(&cfg, &input, 60);

        // Snapshot every byte under input to assert dry-run touches nothing.
        let listing_before = list_dir_recursive(&input);

        let report = convert_quarot_qwen35(
            &input,
            &input, // intentionally the same path
            &ConversionOptions {
                rotation_seed: 0xDEAD_C0DE,
                tolerance: 1e-5,
                num_probe_tokens: 2,
                dry_run: true,
            },
        )
        .unwrap();
        // Dry-run computes real byte counts; no files written.
        assert!(
            report.planned_quantized > 0,
            "dry-run must compute planned_quantized > 0"
        );
        assert!(
            report.total_bytes_out > 0,
            "dry-run must compute total_bytes_out > 0"
        );

        let listing_after = list_dir_recursive(&input);
        assert_eq!(
            listing_before, listing_after,
            "dry-run must not mutate the directory it shares with input"
        );
    }

    /// Dry-run with a non-empty pre-existing output_dir is also fine —
    /// no write happens, and the stale artifacts must survive the
    /// dry-run untouched.
    #[test]
    fn convert_quarot_qwen35_dry_run_ignores_non_empty_output_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        let cfg = tiny_cfg(true);
        write_input_dir(&cfg, &input, 61);
        fs::create_dir_all(&output).unwrap();
        let stale = output.join("stale.q4");
        fs::write(&stale, b"old-bytes").unwrap();

        let report = convert_quarot_qwen35(
            &input,
            &output,
            &ConversionOptions {
                rotation_seed: 0xBEEF_FACE,
                tolerance: 1e-5,
                num_probe_tokens: 2,
                dry_run: true,
            },
        )
        .unwrap();
        // Dry-run computes real byte counts; no files written.
        assert!(
            report.planned_quantized > 0,
            "dry-run must compute planned_quantized > 0"
        );
        assert!(report.kept_f16 > 0, "dry-run must compute kept_f16 > 0");
        assert!(
            report.total_bytes_out > 0,
            "dry-run must compute total_bytes_out > 0"
        );

        // Stale file must survive bit-for-bit; no new files in output.
        assert!(stale.exists(), "stale file must not be deleted in dry-run");
        assert_eq!(fs::read(&stale).unwrap(), b"old-bytes");
        let listing: Vec<_> = fs::read_dir(&output)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(listing.len(), 1, "dry-run must not add files: {listing:?}");
    }

    /// Recursive directory listing helper for "filesystem unchanged"
    /// assertions in dry-run tests. Returns (relative path, byte length)
    /// pairs sorted by path so two listings compare equal iff the
    /// filesystem state matches.
    fn list_dir_recursive(root: &Path) -> Vec<(PathBuf, u64)> {
        fn walk(root: &Path, dir: &Path, out: &mut Vec<(PathBuf, u64)>) {
            for entry in fs::read_dir(dir).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                let metadata = entry.metadata().unwrap();
                if metadata.is_dir() {
                    walk(root, &path, out);
                } else {
                    let rel = path.strip_prefix(root).unwrap().to_path_buf();
                    out.push((rel, metadata.len()));
                }
            }
        }
        let mut out = Vec::new();
        walk(root, root, &mut out);
        out.sort();
        out
    }

    /// Empty pre-existing output dir is fine — the converter populates it.
    #[test]
    fn convert_quarot_qwen35_accepts_empty_pre_existing_output_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        let cfg = tiny_cfg(true);
        write_input_dir(&cfg, &input, 53);
        fs::create_dir_all(&output).unwrap(); // empty

        let report = convert_quarot_qwen35(
            &input,
            &output,
            &ConversionOptions {
                rotation_seed: 0xABCD_EF01,
                tolerance: 1e-5,
                num_probe_tokens: 2,
                dry_run: false,
            },
        )
        .unwrap();
        assert!(report.planned_quantized > 0);
        assert!(output.join("config.json").exists());
    }

    // ------------------------------------------------------------------
    // MTP weight quantization tests
    // ------------------------------------------------------------------

    /// Build a `tiny_cfg` that includes one MTP layer and a `config.json`
    /// that round-trips `mtp_num_hidden_layers = 1`.
    fn tiny_cfg_with_mtp(tied: bool) -> Qwen35Config {
        let mut cfg = tiny_cfg(tied);
        cfg.mtp_num_hidden_layers = 1;
        cfg
    }

    /// Build a `config.json` string that includes `mtp_num_hidden_layers`.
    fn tiny_config_json_with_mtp(cfg: &Qwen35Config) -> String {
        // Start from the base JSON, then inject `mtp_num_hidden_layers` into
        // `text_config` via a JSON round-trip.
        let base = tiny_config_json(cfg);
        let mut root: serde_json::Value = serde_json::from_str(&base).unwrap();
        root.get_mut("text_config")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(
                "mtp_num_hidden_layers".into(),
                serde_json::Value::from(cfg.mtp_num_hidden_layers),
            );
        serde_json::to_string_pretty(&root).unwrap()
    }

    /// Write the minimal MTP tensors (matching the real Qwen3.5-0.8B shapes
    /// scaled down to the tiny test config's `hidden_size=8`) into `path`.
    ///
    /// Shapes are derived from the loader expectations in
    /// `metal_qwen35.rs:load_mtp_q4_weights`:
    ///   - fc.weight:              [hidden, 2*hidden]   → fc projects concat(embed, hidden)
    ///   - q_proj.weight:          [num_heads*head_dim*4, hidden]  — use 4*hidden rows
    ///   - k_proj.weight:          [num_kv_heads*head_dim, hidden] — use hidden rows
    ///   - v_proj.weight:          [num_kv_heads*head_dim, hidden] — use hidden rows
    ///   - o_proj.weight:          [hidden, num_heads*head_dim]    — use hidden rows
    ///   - gate/up_proj.weight:    [intermediate, hidden]
    ///   - down_proj.weight:       [hidden, intermediate]
    ///   - {input,post}_layernorm: [hidden]
    ///   - {q,k}_norm.weight:      [head_dim]
    ///   - {norm,pre_fc_norm_*}:   [hidden]
    ///
    /// All values are synthetic (same LCG used by `synth_data`).
    fn write_mtp_tensors_into(path: &Path, cfg: &Qwen35Config, mut seed: u64) {
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let head_dim = cfg.head_dim;

        // The existing tensors in `path` must be extended, not replaced.
        // SafeTensors are write-once files, so we need to append our MTP
        // tensors via `write_test_safetensors` on a new temp file, then
        // concatenate both sets into a single file.
        //
        // Simpler approach: read the existing file bytes, rewrite the combined
        // header+data. But that requires re-parsing SafeTensors.
        //
        // Easiest: write a SEPARATE second safetensors file, then merge both
        // sets of entries into one call to `write_test_safetensors`.
        // Since `write_required_tensors_for` already created the base file,
        // we parse it to extract its (name, shape, data) triples, append MTP
        // entries, and re-write to `path`.
        //
        // To avoid reimplementing the SafeTensors parser here we instead
        // write a NEW single safetensors file containing both main + MTP
        // tensors from scratch using known shapes. This matches what the
        // real model file looks like.

        let mut next = |n: usize| -> Vec<f64> {
            seed = seed.wrapping_add(1);
            synth_data(n, seed)
        };

        // Rebuild the main model tensors (same as `write_required_tensors_for`
        // but we need them to construct the combined file). We generate the
        // same tensors that `write_required_tensors_for` would, but since
        // the seed was already consumed we generate fresh synth data here.
        // The forward-equivalence test reads from the SAME file so the values
        // don't need to match the ones used in `write_required_tensors_for`
        // — we're building a combined file from scratch for the MTP test.
        let vocab = cfg.vocab_size;
        let full_q_dim = cfg.full_q_dim();
        let full_kv_dim = cfg.full_kv_dim();
        let linear_qkv_dim = cfg.linear_qkv_dim();
        let linear_output_dim = cfg.linear_output_dim();
        let linear_num_heads = cfg.linear_num_key_heads;
        let kernel = cfg.linear_conv_kernel_dim;

        let mut entries: Vec<(String, Vec<usize>, Vec<f64>)> = Vec::new();

        entries.push((
            "model.language_model.embed_tokens.weight".into(),
            vec![vocab, hidden],
            next(vocab * hidden),
        ));
        entries.push((
            "model.language_model.norm.weight".into(),
            vec![hidden],
            next(hidden),
        ));
        if !cfg.tie_word_embeddings {
            entries.push((
                "lm_head.weight".into(),
                vec![vocab, hidden],
                next(vocab * hidden),
            ));
        }

        for i in 0..cfg.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{i}");
            entries.push((
                format!("{prefix}.input_layernorm.weight"),
                vec![hidden],
                next(hidden),
            ));
            entries.push((
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![hidden],
                next(hidden),
            ));

            if cfg.is_full_attention(i) {
                entries.push((
                    format!("{prefix}.self_attn.q_proj.weight"),
                    vec![2 * full_q_dim, hidden],
                    next(2 * full_q_dim * hidden),
                ));
                entries.push((
                    format!("{prefix}.self_attn.k_proj.weight"),
                    vec![full_kv_dim, hidden],
                    next(full_kv_dim * hidden),
                ));
                entries.push((
                    format!("{prefix}.self_attn.v_proj.weight"),
                    vec![full_kv_dim, hidden],
                    next(full_kv_dim * hidden),
                ));
                entries.push((
                    format!("{prefix}.self_attn.o_proj.weight"),
                    vec![hidden, full_q_dim],
                    next(hidden * full_q_dim),
                ));
                entries.push((
                    format!("{prefix}.self_attn.q_norm.weight"),
                    vec![head_dim],
                    next(head_dim),
                ));
                entries.push((
                    format!("{prefix}.self_attn.k_norm.weight"),
                    vec![head_dim],
                    next(head_dim),
                ));
            } else {
                entries.push((
                    format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    vec![linear_qkv_dim, hidden],
                    next(linear_qkv_dim * hidden),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.in_proj_z.weight"),
                    vec![linear_output_dim, hidden],
                    next(linear_output_dim * hidden),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.in_proj_b.weight"),
                    vec![linear_num_heads, hidden],
                    next(linear_num_heads * hidden),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.in_proj_a.weight"),
                    vec![linear_num_heads, hidden],
                    next(linear_num_heads * hidden),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.A_log"),
                    vec![linear_num_heads],
                    next(linear_num_heads),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.dt_bias"),
                    vec![linear_num_heads],
                    next(linear_num_heads),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.conv1d.weight"),
                    vec![linear_qkv_dim, 1, kernel],
                    next(linear_qkv_dim * kernel),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.norm.weight"),
                    vec![linear_output_dim],
                    next(linear_output_dim),
                ));
                entries.push((
                    format!("{prefix}.linear_attn.out_proj.weight"),
                    vec![hidden, linear_output_dim],
                    next(hidden * linear_output_dim),
                ));
            }

            entries.push((
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![intermediate, hidden],
                next(intermediate * hidden),
            ));
            entries.push((
                format!("{prefix}.mlp.up_proj.weight"),
                vec![intermediate, hidden],
                next(intermediate * hidden),
            ));
            entries.push((
                format!("{prefix}.mlp.down_proj.weight"),
                vec![hidden, intermediate],
                next(hidden * intermediate),
            ));
        }

        // --- MTP tensors (shapes match the real Qwen3.5-0.8B MTP head scaled
        // to the tiny config's hidden=8, intermediate=16, head_dim=4) ---
        // fc.weight: [hidden, 2*hidden] — projects concat(embed_hidden, main_hidden)
        entries.push((
            "mtp.fc.weight".into(),
            vec![hidden, 2 * hidden],
            next(hidden * 2 * hidden),
        ));
        // Attention: use simple shapes that are 2-D and divisible by block size 32.
        // For the tiny test hidden=8 we'd get very small matrices; pad up to 32 elements.
        // Use hidden=8 as-is — quantize_f64_to_q4 pads the last block with zeros.
        entries.push((
            "mtp.layers.0.self_attn.q_proj.weight".into(),
            vec![4 * hidden, hidden],
            next(4 * hidden * hidden),
        ));
        entries.push((
            "mtp.layers.0.self_attn.k_proj.weight".into(),
            vec![hidden, hidden],
            next(hidden * hidden),
        ));
        entries.push((
            "mtp.layers.0.self_attn.v_proj.weight".into(),
            vec![hidden, hidden],
            next(hidden * hidden),
        ));
        entries.push((
            "mtp.layers.0.self_attn.o_proj.weight".into(),
            vec![hidden, 2 * hidden],
            next(hidden * 2 * hidden),
        ));
        entries.push((
            "mtp.layers.0.mlp.gate_proj.weight".into(),
            vec![intermediate, hidden],
            next(intermediate * hidden),
        ));
        entries.push((
            "mtp.layers.0.mlp.up_proj.weight".into(),
            vec![intermediate, hidden],
            next(intermediate * hidden),
        ));
        entries.push((
            "mtp.layers.0.mlp.down_proj.weight".into(),
            vec![hidden, intermediate],
            next(hidden * intermediate),
        ));
        // f16 tensors (norms + small vectors)
        entries.push((
            "mtp.layers.0.input_layernorm.weight".into(),
            vec![hidden],
            next(hidden),
        ));
        entries.push((
            "mtp.layers.0.post_attention_layernorm.weight".into(),
            vec![hidden],
            next(hidden),
        ));
        entries.push((
            "mtp.layers.0.self_attn.q_norm.weight".into(),
            vec![head_dim],
            next(head_dim),
        ));
        entries.push((
            "mtp.layers.0.self_attn.k_norm.weight".into(),
            vec![head_dim],
            next(head_dim),
        ));
        entries.push(("mtp.norm.weight".into(), vec![hidden], next(hidden)));
        entries.push((
            "mtp.pre_fc_norm_embedding.weight".into(),
            vec![hidden],
            next(hidden),
        ));
        entries.push((
            "mtp.pre_fc_norm_hidden.weight".into(),
            vec![hidden],
            next(hidden),
        ));

        let borrowed: Vec<(&str, Vec<usize>, &[f64])> = entries
            .iter()
            .map(|(n, s, d)| (n.as_str(), s.clone(), d.as_slice()))
            .collect();
        write_test_safetensors(path, &borrowed);
    }

    /// Write an input dir whose safetensors contains BOTH main model tensors
    /// and MTP tensors, with config.json that sets mtp_num_hidden_layers=1.
    fn write_input_dir_with_mtp(cfg: &Qwen35Config, dir: &Path, seed: u64) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("config.json"), tiny_config_json_with_mtp(cfg)).unwrap();
        // Write combined main+MTP safetensors in one shot.
        write_mtp_tensors_into(&dir.join("model.safetensors"), cfg, seed);
    }

    /// QuaRot converter emits MTP files in O-space (no rotation absorption).
    /// The runtime applies R^T to inputs and R to outputs at inference time
    /// (counter-rotate strategy, ADR-044 §MTP extension).
    #[test]
    fn convert_quarot_qwen35_emits_mtp_files_for_quarot() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        let cfg = tiny_cfg_with_mtp(true);
        write_input_dir_with_mtp(&cfg, &input, 70);

        let rotation_seed: u64 = 0xC0DE_BABE;
        let _report = convert_quarot_qwen35(
            &input,
            &output,
            &ConversionOptions {
                rotation_seed,
                tolerance: 1e-5,
                num_probe_tokens: 2,
                dry_run: false,
            },
        )
        .unwrap();

        // ADR-051 Phase 1: ALL 15 MTP tensors emitted as .f16 (no Q4). The runtime
        // applies counter-rotation on unquantized weights; Phase 2 will rotate and
        // quantize MTP tensors offline. Cross-check no MTP .q4 file leaked.
        let expected_f16 = [
            "mtp_fc_weight.f16",
            "mtp_layers_0_self_attn_q_proj_weight.f16",
            "mtp_layers_0_self_attn_k_proj_weight.f16",
            "mtp_layers_0_self_attn_v_proj_weight.f16",
            "mtp_layers_0_self_attn_o_proj_weight.f16",
            "mtp_layers_0_mlp_gate_proj_weight.f16",
            "mtp_layers_0_mlp_up_proj_weight.f16",
            "mtp_layers_0_mlp_down_proj_weight.f16",
            "mtp_layers_0_input_layernorm_weight.f16",
            "mtp_layers_0_post_attention_layernorm_weight.f16",
            "mtp_layers_0_self_attn_q_norm_weight.f16",
            "mtp_layers_0_self_attn_k_norm_weight.f16",
            "mtp_norm_weight.f16",
            "mtp_pre_fc_norm_embedding_weight.f16",
            "mtp_pre_fc_norm_hidden_weight.f16",
        ];
        for name in &expected_f16 {
            assert!(
                output.join(name).exists(),
                "MTP f16 file must be emitted: {name}"
            );
        }
        // No .q4 MTP file may be emitted in Phase 1.
        for name in &expected_f16 {
            let q4_variant = name.replace(".f16", ".q4");
            assert!(
                !output.join(&q4_variant).exists(),
                "MTP Q4 file must NOT be emitted in Phase 1: {q4_variant}"
            );
        }

        // quantize_index.json must carry quarot_seed (ADR-051 contract).
        let idx_str = fs::read_to_string(output.join("quantize_index.json")).unwrap();
        let idx_val: serde_json::Value = serde_json::from_str(&idx_str).unwrap();
        assert_eq!(
            idx_val
                .get("quarot_seed")
                .and_then(serde_json::Value::as_u64),
            Some(rotation_seed),
            "quantize_index.json must carry quarot_seed (ADR-051 contract)"
        );

        // Output config must retain mtp_num_hidden_layers=1 and carry quarot_rotation_seed
        // as a backwards-compatible diagnostic mirror of the index seed.
        let out_cfg_str = fs::read_to_string(output.join("config.json")).unwrap();
        let out_val: serde_json::Value = serde_json::from_str(&out_cfg_str).unwrap();
        assert_eq!(
            out_val
                .get("text_config")
                .and_then(|tc| tc.get("mtp_num_hidden_layers"))
                .and_then(serde_json::Value::as_u64),
            Some(1),
            "output config text_config.mtp_num_hidden_layers must be 1"
        );
        assert_eq!(
            out_val
                .get("quarot_rotation_seed")
                .and_then(serde_json::Value::as_u64),
            Some(rotation_seed),
            "output config must carry quarot_rotation_seed at top level"
        );
        assert_eq!(
            out_val
                .get("text_config")
                .and_then(|tc| tc.get("quarot_rotation_seed"))
                .and_then(serde_json::Value::as_u64),
            Some(rotation_seed),
            "output config text_config must carry quarot_rotation_seed"
        );
    }

    /// Counter-rotation equivalence gate: verify that apply_inverse followed by
    /// apply recovers the original vector (round-trip) for the hidden dimension
    /// used by the tiny test config.
    #[test]
    fn quarot_mtp_counter_rotation_roundtrip() {
        use crate::quant::quarot::hadamard::RandomizedHadamard;

        let hidden = 8usize; // tiny_cfg hidden_size
        let seed: u64 = 0xDEAD_C0DE;
        let rot = RandomizedHadamard::new(seed, hidden).unwrap();

        // Simulate R-space vector (what the QuaRot runtime would produce).
        let original: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.31 + 0.7).cos()).collect();
        let mut data = original.clone();

        // apply_inverse (R^T): R-space → O-space
        rot.apply_inverse(&mut data).unwrap();
        // apply (R): O-space → R-space
        rot.apply(&mut data).unwrap();

        for (i, (got, expected)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-4,
                "roundtrip failed at index {i}: got={got}, expected={expected}"
            );
        }
    }

    /// Verify inject_quarot_seed writes the seed into both text_config and top level.
    #[test]
    fn inject_quarot_seed_roundtrips_in_config_json() {
        let json = r#"{"text_config": {"hidden_size": 8}, "some_key": 1}"#;
        let seed: u64 = 0xCAFE_BABE;
        let output = inject_quarot_seed(json, seed).unwrap();
        let val: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(
            val.get("quarot_rotation_seed")
                .and_then(serde_json::Value::as_u64),
            Some(seed),
            "quarot_rotation_seed must be at top level"
        );
        assert_eq!(
            val.get("text_config")
                .and_then(|tc| tc.get("quarot_rotation_seed"))
                .and_then(serde_json::Value::as_u64),
            Some(seed),
            "quarot_rotation_seed must be inside text_config"
        );
    }

    /// When config has `mtp_num_hidden_layers > 0` but the checkpoint
    /// does NOT contain MTP tensors, the converter must succeed silently
    /// (skip MTP) without writing any `mtp.*` files.
    #[test]
    fn convert_quarot_qwen35_skips_mtp_when_tensors_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        // Use a config that says mtp_num_hidden_layers=1 but only write
        // main-model tensors (no MTP) to the safetensors file.
        let cfg = tiny_cfg_with_mtp(true);
        fs::create_dir_all(&input).unwrap();
        fs::write(input.join("config.json"), tiny_config_json_with_mtp(&cfg)).unwrap();
        // Write only the main model tensors — no MTP tensors in the file.
        write_required_tensors_for(&cfg, &input.join("model.safetensors"), 71);

        let report = convert_quarot_qwen35(
            &input,
            &output,
            &ConversionOptions {
                rotation_seed: 0xDEAD_BABE,
                tolerance: 1e-5,
                num_probe_tokens: 2,
                dry_run: false,
            },
        )
        .unwrap();

        // No mtp.* files should exist in the output directory.
        let entries: Vec<_> = fs::read_dir(&output)
            .unwrap()
            .filter_map(std::result::Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        for name in &entries {
            assert!(
                !name.starts_with("mtp"),
                "unexpected MTP file written when tensors were absent: {name}"
            );
        }

        // Main model must still have been converted successfully.
        assert!(report.planned_quantized > 0);
        assert!(output.join("config.json").exists());
    }

    /// When config has `mtp_num_hidden_layers == 0`, the converter must NOT
    /// write any MTP files — the zero-config gate skips MTP regardless of
    /// whether the checkpoint contains mtp.* tensors.
    #[test]
    fn convert_quarot_qwen35_skips_mtp_when_config_has_zero_layers() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("input");
        let output = tmp.path().join("output");
        // Force mtp_num_hidden_layers = 0 explicitly (tiny_cfg inherits 1
        // from qwen35_0_8b; override it here).
        let mut cfg = tiny_cfg(true);
        cfg.mtp_num_hidden_layers = 0;
        assert_eq!(cfg.mtp_num_hidden_layers, 0);
        // write_required_tensors_for only writes main model tensors (no MTP).
        // We write a plain config.json (no mtp_num_hidden_layers key so it
        // defaults to 0 on deserialize) alongside the main safetensors.
        fs::create_dir_all(&input).unwrap();
        fs::write(input.join("config.json"), tiny_config_json(&cfg)).unwrap();
        write_required_tensors_for(&cfg, &input.join("model.safetensors"), 72);

        let report = convert_quarot_qwen35(
            &input,
            &output,
            &ConversionOptions {
                rotation_seed: 0xCAFE_F00D,
                tolerance: 1e-5,
                num_probe_tokens: 2,
                dry_run: false,
            },
        )
        .unwrap();

        let entries: Vec<_> = fs::read_dir(&output)
            .unwrap()
            .filter_map(std::result::Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        for name in &entries {
            assert!(
                !name.starts_with("mtp"),
                "unexpected MTP file written for zero-MTP-layer config: {name}"
            );
        }
        assert!(report.planned_quantized > 0);
    }

    // ------------------------------------------------------------------
    // read_quarot_seed_from_index (#504 remaining slice 2: fail-closed
    // integrity for `quantize_index.json`).
    // ------------------------------------------------------------------

    #[test]
    fn read_quarot_seed_from_index_absent_file_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(
            read_quarot_seed_from_index(tmp.path()),
            Ok(None),
            "missing quantize_index.json must yield Ok(None), not an error"
        );
    }

    #[test]
    fn read_quarot_seed_from_index_without_key_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("quantize_index.json"), r#"{"tensors":[]}"#).unwrap();
        assert_eq!(
            read_quarot_seed_from_index(tmp.path()),
            Ok(None),
            "index without quarot_seed key must yield Ok(None)"
        );
    }

    #[test]
    fn read_quarot_seed_from_index_bare_array_is_none() {
        // `quantize_q4` (plain, non-rotated Q4 checkpoints) writes
        // `quantize_index.json` as a bare top-level tensor array, not the
        // `{quarot_seed, tensors}` object `quantize_quarot` produces. This
        // must be recognized as "no seed", not rejected as malformed —
        // reverting the array-shape check makes serde misparse the first
        // array element as the `quarot_seed: Option<u64>` field and error.
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("quantize_index.json"),
            r#"[{"name":"foo","file":"foo.q4","quantized":true,"shape":[2,2],"numel":4}]"#,
        )
        .unwrap();
        assert_eq!(
            read_quarot_seed_from_index(tmp.path()),
            Ok(None),
            "bare tensor-array quantize_index.json (plain quantize_q4 shape) must yield Ok(None)"
        );
    }

    #[test]
    fn read_quarot_seed_from_index_finds_seed() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("quantize_index.json"),
            r#"{"quarot_seed":13258600446175248384,"tensors":[]}"#,
        )
        .unwrap();
        assert_eq!(
            read_quarot_seed_from_index(tmp.path()),
            Ok(Some(13_258_600_446_175_248_384_u64)),
            "index with quarot_seed key must round-trip the u64 exactly"
        );
    }

    #[test]
    fn read_quarot_seed_from_index_rejects_malformed_json() {
        // #504 remaining slice 2: a *present* file that fails to parse must
        // be a hard error, not a silent None — the pre-fix behavior treated
        // corruption/truncation identically to "legitimately absent",
        // which would silently apply the wrong (or no) QuaRot rotation to
        // a checkpoint that actually needed one.
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("quantize_index.json"), "not json").unwrap();
        let err = read_quarot_seed_from_index(tmp.path())
            .expect_err("malformed quantize_index.json must be rejected, not silently None");
        assert!(
            err.contains("malformed quantize_index.json"),
            "error must name the malformed-index failure; got: {err}"
        );
    }

    #[test]
    fn read_quarot_seed_from_index_rejects_incomplete_object_form_entry() {
        // Regression test: unifying shape-normalization with `doctor`'s
        // tolerant parser would incorrectly *accept* an object-form
        // manifest whose tensor entries omit `quantized`/`shape`/`numel`
        // (doctor's historical leniency, appropriate for a tensor
        // inventory listing but not for the seed loader). The seed
        // loader's object-form schema has always required every
        // `IndexEntry` field; a partially formed entry here means
        // "corrupted or in-progress write", not "no seed", and must be
        // rejected.
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("quantize_index.json"),
            r#"{"quarot_seed":42,"tensors":[{"name":"foo","file":"foo.q4"}]}"#,
        )
        .unwrap();
        let err = read_quarot_seed_from_index(tmp.path()).expect_err(
            "object-form manifest with an incomplete tensor entry must be rejected, \
             not silently accepted",
        );
        assert!(
            err.contains("malformed quantize_index.json"),
            "error must name the malformed-index failure; got: {err}"
        );
    }

    #[test]
    fn read_quarot_seed_from_index_bare_array_with_malformed_entries_is_none() {
        // Regression test, opposite direction: unifying shape-normalization
        // with `doctor`'s parser would incorrectly *reject* a bare-array
        // manifest whose entries are missing required fields, because that
        // parser validates array entries the same way `doctor` does. The
        // seed loader has never validated array-shape entries at all — a
        // bare array can never carry a rotation seed regardless of how
        // malformed its entries are, so it must stay `Ok(None)`.
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("quantize_index.json"),
            r#"[{"name":"foo"}, "not even an object", 42]"#,
        )
        .unwrap();
        assert_eq!(
            read_quarot_seed_from_index(tmp.path()),
            Ok(None),
            "a bare array with malformed entries still carries no rotation seed \
             and must yield Ok(None), not an error"
        );
    }

    #[test]
    fn read_quarot_seed_from_index_rejects_wrong_schema_shape() {
        // `tensors` present but wrong type (string instead of an array) —
        // valid JSON, but does not match the `QuantizeIndex` schema.
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("quantize_index.json"),
            r#"{"quarot_seed":42,"tensors":"not-an-array"}"#,
        )
        .unwrap();
        assert!(
            read_quarot_seed_from_index(tmp.path()).is_err(),
            "schema-shape mismatch (tensors not an array) must be rejected"
        );
    }

    #[test]
    fn read_quarot_seed_from_index_rejects_truncated_file() {
        // A file that exists but was cut off mid-write (e.g. a crash
        // during `fs::write`) is invalid JSON and must fail closed.
        let tmp = tempfile::tempdir().unwrap();
        fs::write(
            tmp.path().join("quantize_index.json"),
            r#"{"quarot_seed":42,"tensors":[{"name":"foo","file":"foo.q4","quant"#,
        )
        .unwrap();
        assert!(
            read_quarot_seed_from_index(tmp.path()).is_err(),
            "truncated quantize_index.json must be rejected"
        );
    }

    #[test]
    fn read_quarot_seed_from_index_rejects_oversized_file() {
        // Bounded-read discipline (#504 remaining slice 1 pattern applied
        // here): a file far larger than any real index should ever be
        // must be rejected before it is fully read into memory.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("quantize_index.json");
        // One byte over the cap; write sparsely via set_len to avoid
        // actually allocating/writing 16 MiB+1 in the test.
        let f = fs::File::create(&path).unwrap();
        f.set_len(crate::quant::q4_manifest::MAX_QUANTIZE_INDEX_LEN + 1)
            .unwrap();
        drop(f);
        let err = read_quarot_seed_from_index(tmp.path())
            .expect_err("oversized quantize_index.json must be rejected");
        assert!(
            err.contains("too large"),
            "error must name the size-cap failure; got: {err}"
        );
    }
}
