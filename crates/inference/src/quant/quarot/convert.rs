//! High-level QuaRot Qwen3.5 conversion (ADR-044 step 3c-5).
//!
//! [`convert_quarot_qwen35`] reads `config.json` + SafeTensors from
//! `input_dir`, runs the full pipeline
//! (`materialize_lm_head` → `fuse_rmsnorms` → `absorb_rotations` →
//! [`assert_forward_equivalence_qwen35`]) entirely in f64, and on success
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
use crate::weights::q4_weights::{q4_f32_to_f16, quantize_f64_to_q4, save_q4_file};

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
    /// Sum of input tensor sizes in bytes (8 × element count, f64).
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

#[derive(serde::Serialize)]
struct IndexEntry {
    name: String,
    file: String,
    quantized: bool,
    shape: Vec<usize>,
    numel: usize,
}

/// Refuse-on-fail QuaRot Qwen3.5 model conversion (ADR-044 §"Step 3c contract").
///
/// On success, writes the converted model to `output_dir` (created if
/// absent) and returns a [`ConversionReport`]. On the forward-equivalence
/// gate refusing, returns the gate's `Err` **without writing any output
/// files**.
///
/// `opts.dry_run = true` runs the full pipeline + gate but skips every
/// disk write, returning a report with `planned_quantized = 0`,
/// `kept_f16 = 0`, `total_bytes_out = 0`.
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
pub fn convert_quarot_qwen35(
    input_dir: &Path,
    output_dir: &Path,
    opts: &ConversionOptions,
) -> Result<ConversionReport, InferenceError> {
    // Path-layout validation runs FIRST so the cheap CLI footguns
    // (same dir, non-empty target) fail before any expensive tensor
    // work and before any disk write.
    validate_output_dir_layout(input_dir, output_dir)?;

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
    let mut working_set = load_tensors_f64(&reader, &required_names)?;
    let total_bytes_in: u64 = working_set
        .values()
        .map(|t| (t.data.len() as u64).saturating_mul(8))
        .sum();

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

    if opts.dry_run {
        return Ok(ConversionReport {
            planned_quantized: 0,
            kept_f16: 0,
            total_bytes_in,
            total_bytes_out: 0,
            forward_equivalence,
            was_tied,
        });
    }

    fs::create_dir_all(output_dir).map_err(|e| {
        InferenceError::Inference(format!(
            "convert_quarot_qwen35: failed to create output directory {}: {e}",
            output_dir.display()
        ))
    })?;

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
            let q4 = quantize_f64_to_q4(&entry.data, &entry.shape);
            let file_name = format!("{sanitized}.q4");
            let out_path = output_dir.join(&file_name);
            save_q4_file(&out_path, &q4).map_err(|e| {
                InferenceError::Inference(format!(
                    "convert_quarot_qwen35: failed to write {}: {e}",
                    out_path.display()
                ))
            })?;
            // Q4 file footprint: 4-byte magic + 4 version + 4 ndim +
            // 8*ndim shape + 8 original_len + 18 bytes per block.
            let header_bytes = (4 + 4 + 4 + 8 * entry.shape.len() + 8) as u64;
            total_bytes_out += header_bytes + (q4.blocks.len() as u64).saturating_mul(18);
            planned_quantized += 1;
            index_entries.push(IndexEntry {
                name: name.clone(),
                file: file_name,
                quantized: true,
                shape: entry.shape.clone(),
                numel: entry.data.len(),
            });
        } else {
            let file_name = format!("{sanitized}.f16");
            let out_path = output_dir.join(&file_name);
            let bytes = write_f16_file(&out_path, &entry.data, &entry.shape)?;
            total_bytes_out += bytes as u64;
            kept_f16 += 1;
            index_entries.push(IndexEntry {
                name: name.clone(),
                file: file_name,
                quantized: false,
                shape: entry.shape.clone(),
                numel: entry.data.len(),
            });
        }
    }

    let index_path = output_dir.join("quantize_index.json");
    let index_json = serde_json::to_string_pretty(&index_entries).map_err(|e| {
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

    let output_config_json = untie_word_embeddings_in_config_json(&config_json)?;
    let out_config_path = output_dir.join("config.json");
    fs::write(&out_config_path, &output_config_json).map_err(|e| {
        InferenceError::Inference(format!(
            "convert_quarot_qwen35: failed to write {}: {e}",
            out_config_path.display()
        ))
    })?;

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
/// output directory:
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
fn write_f16_file(path: &Path, data: &[f64], shape: &[usize]) -> Result<usize, InferenceError> {
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

    let mut payload = Vec::with_capacity(data.len() * 2);
    for &v in data {
        // Use the subnormal-aware helper from `weights::q4_weights`. The
        // hand-rolled flush-to-zero variant in `bin/quantize_q4.rs`
        // silently rounds f16-subnormal-but-f32-normal values
        // (~1e-7 range) to zero — relevant here because every kept
        // tensor (norms, A_log, dt_bias, conv1d, etc.) passes through
        // this path. `q4_f32_to_f16` round-trips through the f16
        // subnormal range correctly.
        let h = q4_f32_to_f16(v as f32);
        payload.extend_from_slice(&h.to_le_bytes());
    }
    write_all(&payload)?;
    bytes_written += payload.len();

    Ok(bytes_written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen35_config::{LayerType, compute_layer_types};
    use serde_json::Value;

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
        cfg.partial_rotary_factor = 0.25;
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

    // ------------------------------------------------------------------
    // Happy path
    // ------------------------------------------------------------------

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

        // Index json contains every working-set tensor.
        let idx_str = fs::read_to_string(output.join("quantize_index.json")).unwrap();
        let idx: serde_json::Value = serde_json::from_str(&idx_str).unwrap();
        let arr = idx.as_array().expect("index is a JSON array");
        assert_eq!(arr.len(), report.planned_quantized + report.kept_f16);
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

        assert_eq!(report.planned_quantized, 0);
        assert_eq!(report.kept_f16, 0);
        assert_eq!(report.total_bytes_out, 0);
        assert!(report.forward_equivalence.max_abs_error <= 1e-5);
        assert!(
            !output.exists(),
            "dry-run must not create the output directory"
        );
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
        let bytes_written = write_f16_file(&p, &data, &shape).unwrap();
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

    /// Smoke check on `q4_f32_to_f16` (used by `write_f16_file`).
    /// Includes an f16-subnormal regression: codex round-1 flagged that
    /// the old local helper flushed every value below f16's smallest
    /// normal to zero, silently corrupting small-magnitude weights in
    /// kept tensors (e.g., `A_log`, `dt_bias`, GDN `linear_attn.norm`).
    #[test]
    fn q4_f32_to_f16_canonical_and_subnormal_values() {
        assert_eq!(q4_f32_to_f16(0.0), 0x0000);
        assert_eq!(q4_f32_to_f16(-0.0), 0x8000);
        assert_eq!(q4_f32_to_f16(1.0), 0x3c00);
        assert_eq!(q4_f32_to_f16(-1.0), 0xbc00);
        assert_eq!(q4_f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(q4_f32_to_f16(f32::NEG_INFINITY), 0xfc00);
        // f16 smallest positive normal is 2^-14 ≈ 6.103515625e-5; values
        // below that but above the f16 subnormal floor (2^-24) must NOT
        // flush to zero — they should encode as f16 subnormals.
        let h = q4_f32_to_f16(1e-7_f32);
        assert_ne!(
            h, 0,
            "1e-7 (an f16 subnormal range value) must not flush to zero \
             — that was the codex round-1 Medium"
        );
        // f32 subnormals (well below f16's subnormal range) DO round to zero
        // because there's no f16 representation for them.
        assert_eq!(q4_f32_to_f16(1e-40_f32), 0);
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
    // Path-layout refuses (codex round-1 Majors 1 + 2)
    // ------------------------------------------------------------------

    /// Major 1: when input and output paths resolve to the same canonical
    /// path, the converter must refuse before any write would corrupt
    /// the source `config.json`.
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

    /// Major 1 sibling: even when the two paths differ literally (e.g.,
    /// trailing slash, symlink), canonicalization must still catch the
    /// equivalence.
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

    /// Major 2: a pre-existing non-empty output directory must trigger
    /// refusal before any conversion work, so a previously-written `.q4`
    /// artifact cannot survive a gate failure and be picked up by the
    /// runtime loader.
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
}
