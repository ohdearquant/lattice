//! Converts safetensors F16/BF16 MLP weights to packed W3, producing a
//! complete loadable mixed `.w3`/`.q4`/`.f16` directory (issue #420).
//!
//! Mirrors `bin/quantize_q4.rs`'s directory-in/directory-out shape and
//! tensor classification, except dense MLP `gate_proj`/`up_proj`/`down_proj`
//! weights ([`is_w3_mlp_tensor_name`]) are shadowed to W3 *before* the Q4
//! classification runs. Everything else (attention, GDN, embeddings,
//! `lm_head`, norms) follows the existing Q4 converter behavior exactly.

use crate::error::InferenceError;
use crate::weights::f32_weights::parse_index;
use crate::weights::q4_weights::{quantize_bf16_to_q4, save_q4_file};
use crate::weights::w3_weights::{
    is_w3_mlp_tensor_name, quantize_bf16_to_w3, quantize_f16_to_w3, save_w3_file,
};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// Output container a tensor was routed to by [`classify_w3_mlp_output`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum W3MlpOutputFormat {
    W3,
    Q4,
    F16,
}

/// One row of `quantize_index.json`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct W3MlpIndexEntry {
    /// Source tensor name.
    pub name: String,
    /// Output file stem (relative to the output directory).
    pub file: String,
    /// Which container the tensor was written to.
    pub format: W3MlpOutputFormat,
    /// `true` for W3/Q4 (quantized), `false` for F16 (kept full-ish precision).
    pub quantized: bool,
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Number of original elements.
    pub numel: usize,
}

/// Summary counters returned by [`quantize_w3_mlp`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct W3MlpConvertReport {
    pub tensors_processed: usize,
    pub tensors_w3: usize,
    pub tensors_q4: usize,
    pub tensors_f16: usize,
    pub bytes_in: u64,
    pub bytes_out: u64,
}

/// Classify a tensor name into its output container.
///
/// Dense MLP `gate_proj`/`up_proj`/`down_proj` weights ([`is_w3_mlp_tensor_name`])
/// always route to [`W3MlpOutputFormat::W3`]. Everything else mirrors the Q4
/// converter's `should_quantize` rule.
pub fn classify_w3_mlp_output(name: &str) -> W3MlpOutputFormat {
    if is_w3_mlp_tensor_name(name) {
        return W3MlpOutputFormat::W3;
    }
    if should_quantize_non_mlp(name) {
        W3MlpOutputFormat::Q4
    } else {
        W3MlpOutputFormat::F16
    }
}

/// Mirrors `bin/quantize_q4.rs::should_quantize`, applied only to tensors
/// that [`classify_w3_mlp_output`] has already excluded from the W3 path.
fn should_quantize_non_mlp(name: &str) -> bool {
    if !name.ends_with(".weight") && !name.ends_with("lm_head.weight") {
        return false;
    }
    if name.ends_with("_proj.weight")
        || name.ends_with("_proj_a.weight")
        || name.ends_with("_proj_b.weight")
        || name.ends_with("_proj_qkv.weight")
        || name.ends_with("_proj_z.weight")
        || name.ends_with("gate_proj.weight")
        || name.ends_with("up_proj.weight")
        || name.ends_with("down_proj.weight")
        || name.ends_with("lm_head.weight")
        || name.ends_with("embed_tokens.weight")
    {
        return true;
    }
    if name.contains("norm.weight")
        || name.contains("norm_")
        || name.ends_with(".bias")
        || name.ends_with("A_log")
        || name.ends_with("dt_bias")
        || name.ends_with("conv1d.weight")
    {
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Minimal safetensors shard parser (mirrors bin/quantize_q4.rs::open_shard).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShardDType {
    F32,
    F16,
    BF16,
}

impl ShardDType {
    fn bytes_per_elem(self) -> usize {
        match self {
            ShardDType::F32 => 4,
            ShardDType::F16 | ShardDType::BF16 => 2,
        }
    }
}

struct TensorHeader {
    dtype: ShardDType,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

struct ShardData {
    mmap: Mmap,
    data_offset: usize,
    tensors: HashMap<String, TensorHeader>,
}

impl ShardData {
    fn tensor_bytes(&self, name: &str) -> &[u8] {
        let h = &self.tensors[name];
        &self.mmap[self.data_offset + h.start..self.data_offset + h.end]
    }
}

fn open_shard(path: &Path) -> Result<ShardData, InferenceError> {
    let file = std::fs::File::open(path).map_err(InferenceError::Io)?;
    // SAFETY: file is opened read-only; the mapping is dropped with `ShardData`.
    let mmap = unsafe { Mmap::map(&file) }.map_err(InferenceError::Io)?;

    if mmap.len() < 8 {
        return Err(InferenceError::InvalidSafetensors(format!(
            "{}: file too small to contain a safetensors header",
            path.display()
        )));
    }
    let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let data_offset = 8 + header_len;
    if data_offset > mmap.len() {
        return Err(InferenceError::InvalidSafetensors(format!(
            "{}: header extends past end of file (header_end={data_offset}, file_len={})",
            path.display(),
            mmap.len()
        )));
    }
    let header_str = std::str::from_utf8(&mmap[8..data_offset]).map_err(|e| {
        InferenceError::InvalidSafetensors(format!("{}: header is not UTF-8: {e}", path.display()))
    })?;
    let root: Value = serde_json::from_str(header_str).map_err(|e| {
        InferenceError::InvalidSafetensors(format!("{}: header JSON parse: {e}", path.display()))
    })?;
    let obj = root.as_object().ok_or_else(|| {
        InferenceError::InvalidSafetensors(format!(
            "{}: safetensors header is not a JSON object",
            path.display()
        ))
    })?;

    let mut tensors = HashMap::with_capacity(obj.len());
    for (name, entry) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype_str = entry["dtype"].as_str().ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!("{name}: missing dtype in shard header"))
        })?;
        let dtype = match dtype_str {
            "F32" => ShardDType::F32,
            "F16" => ShardDType::F16,
            "BF16" => ShardDType::BF16,
            other => {
                // Non-weight metadata tensors (e.g. tokenizer int buffers) in
                // unsupported dtypes are skipped; weight tensors are not.
                if name.ends_with(".weight") {
                    return Err(InferenceError::InvalidSafetensors(format!(
                        "{name}: unsupported dtype {other} for a weight tensor \
                         (W3/Q4 conversion requires F32/F16/BF16)"
                    )));
                }
                continue;
            }
        };
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .ok_or_else(|| InferenceError::InvalidSafetensors(format!("{name}: missing shape")))?
            .iter()
            .map(|v| {
                v.as_u64().map(|x| x as usize).ok_or_else(|| {
                    InferenceError::InvalidSafetensors(format!("{name}: non-u64 shape dim"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = entry["data_offsets"].as_array().ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!("{name}: missing data_offsets"))
        })?;
        if offsets.len() != 2 {
            return Err(InferenceError::InvalidSafetensors(format!(
                "{name}: data_offsets must have length 2"
            )));
        }
        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;
        let numel: usize = shape.iter().product();
        let expected_bytes = numel * dtype.bytes_per_elem();
        if end.saturating_sub(start) != expected_bytes {
            return Err(InferenceError::InvalidSafetensors(format!(
                "{name}: byte length mismatch — shape {shape:?} implies {expected_bytes} bytes, \
                 got {}",
                end.saturating_sub(start)
            )));
        }
        tensors.insert(
            name.clone(),
            TensorHeader {
                dtype,
                shape,
                start,
                end,
            },
        );
    }

    Ok(ShardData {
        mmap,
        data_offset,
        tensors,
    })
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

fn bf16_bytes_to_u16(raw: &[u8]) -> Vec<u16> {
    raw.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Convert f32 to an IEEE-754 F16 bit pattern with round-to-nearest-even.
///
/// Mirrors `bin/quantize_q4.rs::f32_to_f16`, used only for the F16 "keep as
/// f16" small-tensor path (mirroring the Q4 converter's F16 output, not the
/// W3 quantization math).
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

    if exp == 0xff {
        if frac == 0 {
            return sign | 0x7c00;
        }
        let mut payload = ((frac >> 13) as u16) & 0x03ff;
        if payload == 0 {
            payload = 1;
        }
        return sign | 0x7c00 | payload | 0x0200;
    }
    if exp == 0 {
        return sign;
    }
    let exp32 = exp - 127;
    if exp32 > 15 {
        return sign | 0x7c00;
    }
    if exp32 >= -14 {
        let frac16_raw = (frac >> 13) as u16;
        let round_bit = ((frac >> 12) & 1) as u16;
        let sticky = (frac & 0x0fff) != 0;
        let frac16 = frac16_raw
            + if round_bit == 1 && (sticky || (frac16_raw & 1) == 1) {
                1
            } else {
                0
            };
        let mut exp16 = (exp32 + 15) as u16;
        let mut frac16_final = frac16 & 0x03ff;
        if frac16 == 0x0400 {
            frac16_final = 0;
            exp16 += 1;
            if exp16 >= 0x1f {
                return sign | 0x7c00;
            }
        }
        return sign | (exp16 << 10) | frac16_final;
    }
    sign
}

fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

fn write_f16_file(
    out_path: &Path,
    shape: &[usize],
    numel: usize,
    f16_data: &[u8],
) -> Result<(), InferenceError> {
    use std::io::Write;
    let mut f = std::fs::File::create(out_path).map_err(InferenceError::Io)?;
    f.write_all(b"KHF1").map_err(InferenceError::Io)?;
    f.write_all(&1u32.to_le_bytes())
        .map_err(InferenceError::Io)?;
    f.write_all(&(shape.len() as u32).to_le_bytes())
        .map_err(InferenceError::Io)?;
    for &dim in shape {
        f.write_all(&(dim as u64).to_le_bytes())
            .map_err(InferenceError::Io)?;
    }
    f.write_all(&(numel as u64).to_le_bytes())
        .map_err(InferenceError::Io)?;
    f.write_all(f16_data).map_err(InferenceError::Io)
}

/// Convert a directory of sharded safetensors F16/BF16 MLP weights into a
/// complete, loadable mixed `.w3`/`.q4`/`.f16` directory.
///
/// Dense MLP `gate_proj`/`up_proj`/`down_proj` weight tensors
/// ([`is_w3_mlp_tensor_name`]) are packed as W3; every other tensor follows
/// the existing `quantize_q4` behavior (large weight matrices → Q4, small/
/// special tensors → F16). `output_dir` is a complete artifact loadable by
/// `MetalQwen35State::from_w3_mlp_dir` — see the design doc for the
/// fail-closed loader contract.
///
/// When `dry_run` is `true`, shards are parsed and tensors classified but no
/// files are written; the returned report still reflects the routing that
/// would have happened.
///
/// # Errors
///
/// Returns [`InferenceError::InvalidSafetensors`] if `model_dir` lacks a
/// valid `model.safetensors.index.json` or a shard fails to parse (including
/// an unsupported dtype for a weight tensor); [`InferenceError::InvalidInput`]
/// if a W3 MLP tensor has dtype `F32` (not yet supported by the W3 path) or
/// contains non-finite values; [`InferenceError::Io`] on any read/write
/// failure.
pub fn quantize_w3_mlp(
    model_dir: &Path,
    output_dir: &Path,
    dry_run: bool,
) -> Result<W3MlpConvertReport, InferenceError> {
    if !dry_run {
        std::fs::create_dir_all(output_dir).map_err(InferenceError::Io)?;
    }

    let shard_index = parse_index(model_dir)?;

    let mut shard_files: std::collections::BTreeSet<String> = Default::default();
    for v in shard_index.weight_map.values() {
        shard_files.insert(v.clone());
    }

    let mut shard_to_tensors: HashMap<String, Vec<String>> = HashMap::new();
    for (tensor_name, shard_name) in &shard_index.weight_map {
        shard_to_tensors
            .entry(shard_name.clone())
            .or_default()
            .push(tensor_name.clone());
    }

    let mut report = W3MlpConvertReport::default();
    let mut index_entries: Vec<W3MlpIndexEntry> = Vec::new();

    for shard_filename in &shard_files {
        let shard_path = model_dir.join(shard_filename);
        let shard = open_shard(&shard_path)?;

        let mut names = shard_to_tensors
            .get(shard_filename)
            .cloned()
            .unwrap_or_default();
        names.sort_by_key(|n| shard.tensors.get(n).map(|h| h.start).unwrap_or(usize::MAX));

        for tensor_name in &names {
            let Some(h) = shard.tensors.get(tensor_name.as_str()) else {
                continue;
            };
            let shape = h.shape.clone();
            let numel: usize = shape.iter().product();
            let bytes_in = (h.end - h.start) as u64;
            report.bytes_in += bytes_in;
            let raw_bytes = shard.tensor_bytes(tensor_name);

            let format = classify_w3_mlp_output(tensor_name);
            let sanitized = sanitize_tensor_name(tensor_name);

            match format {
                W3MlpOutputFormat::W3 => {
                    let tensor = match h.dtype {
                        ShardDType::BF16 => {
                            quantize_bf16_to_w3(&bf16_bytes_to_u16(raw_bytes), &shape)?
                        }
                        ShardDType::F16 => {
                            quantize_f16_to_w3(&bf16_bytes_to_u16(raw_bytes), &shape)?
                        }
                        ShardDType::F32 => {
                            return Err(InferenceError::InvalidInput(format!(
                                "{tensor_name}: F32 dense MLP tensors are not supported by the \
                                 W3 converter (v1 accepts BF16/F16 only); reject rather than \
                                 silently downcast, per the W3 fail-closed contract"
                            )));
                        }
                    };
                    let bytes_out =
                        (tensor.blocks.len() * crate::weights::w3_weights::W3_BLOCK_SIZE) as u64;
                    report.bytes_out += bytes_out;
                    let out_filename = format!("{sanitized}.w3");
                    if !dry_run {
                        save_w3_file(&output_dir.join(&out_filename), &tensor)?;
                    }
                    index_entries.push(W3MlpIndexEntry {
                        name: tensor_name.clone(),
                        file: out_filename,
                        format,
                        quantized: true,
                        shape,
                        numel,
                    });
                    report.tensors_w3 += 1;
                }
                W3MlpOutputFormat::Q4 => {
                    // Mirrors quantize_q4.rs: assumes BF16 source bytes for
                    // the quantize path, matching current Q4 converter behavior.
                    let bf16_vals = bf16_bytes_to_u16(raw_bytes);
                    let q4 = quantize_bf16_to_q4(&bf16_vals, &shape)?;
                    let bytes_out = (q4.blocks.len()
                        * std::mem::size_of::<crate::weights::q4_weights::Q4Block>())
                        as u64;
                    report.bytes_out += bytes_out;
                    let out_filename = format!("{sanitized}.q4");
                    if !dry_run {
                        save_q4_file(&output_dir.join(&out_filename), &q4)
                            .map_err(InferenceError::Io)?;
                    }
                    index_entries.push(W3MlpIndexEntry {
                        name: tensor_name.clone(),
                        file: out_filename,
                        format,
                        quantized: true,
                        shape,
                        numel,
                    });
                    report.tensors_q4 += 1;
                }
                W3MlpOutputFormat::F16 => {
                    let f16_data: Vec<u8> = match h.dtype {
                        ShardDType::BF16 => raw_bytes
                            .chunks_exact(2)
                            .flat_map(|c| {
                                let bf = u16::from_le_bytes([c[0], c[1]]);
                                f32_to_f16_bits(bf16_to_f32(bf)).to_le_bytes()
                            })
                            .collect(),
                        ShardDType::F16 => raw_bytes.to_vec(),
                        ShardDType::F32 => raw_bytes
                            .chunks_exact(4)
                            .flat_map(|c| {
                                let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                                f32_to_f16_bits(f).to_le_bytes()
                            })
                            .collect(),
                    };
                    report.bytes_out += f16_data.len() as u64;
                    let out_filename = format!("{sanitized}.f16");
                    if !dry_run {
                        write_f16_file(&output_dir.join(&out_filename), &shape, numel, &f16_data)?;
                    }
                    index_entries.push(W3MlpIndexEntry {
                        name: tensor_name.clone(),
                        file: out_filename,
                        format,
                        quantized: false,
                        shape,
                        numel,
                    });
                    report.tensors_f16 += 1;
                }
            }
            report.tensors_processed += 1;
        }
    }

    if !dry_run {
        let index_json = serde_json::to_string_pretty(&index_entries).map_err(|e| {
            InferenceError::InvalidInput(format!("failed to serialize quantize_index.json: {e}"))
        })?;
        std::fs::write(output_dir.join("quantize_index.json"), index_json)
            .map_err(InferenceError::Io)?;
        for extra in ["config.json", "tokenizer.json"] {
            let src = model_dir.join(extra);
            if src.exists() {
                std::fs::copy(&src, output_dir.join(extra)).map_err(InferenceError::Io)?;
            }
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn f32_to_bf16_bytes(v: f32) -> [u8; 2] {
        ((v.to_bits() >> 16) as u16).to_le_bytes()
    }

    /// Write a minimal single-shard safetensors model dir with the given
    /// `(name, shape, values)` tensors, all as BF16.
    fn write_fake_model_dir(dir: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
        std::fs::create_dir_all(dir).unwrap();
        let shard_name = "model.safetensors";

        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        let mut weight_map = serde_json::Map::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for &v in values {
                data.extend_from_slice(&f32_to_bf16_bytes(v));
            }
            let end = data.len();
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
            weight_map.insert((*name).to_string(), Value::String(shard_name.to_string()));
        }
        let header_json = serde_json::to_string(&Value::Object(header)).unwrap();
        let header_bytes = header_json.into_bytes();

        let mut shard_bytes = Vec::new();
        shard_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        shard_bytes.extend_from_slice(&header_bytes);
        shard_bytes.extend_from_slice(&data);
        std::fs::write(dir.join(shard_name), &shard_bytes).unwrap();

        let index = serde_json::json!({
            "metadata": {"total_size": shard_bytes.len()},
            "weight_map": Value::Object(weight_map),
        });
        let mut f = std::fs::File::create(dir.join("model.safetensors.index.json")).unwrap();
        f.write_all(serde_json::to_string(&index).unwrap().as_bytes())
            .unwrap();
    }

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("w3_mlp_convert_test_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn test_classify_w3_mlp_output_routes_dense_mlp_to_w3() {
        assert_eq!(
            classify_w3_mlp_output("model.language_model.layers.0.mlp.gate_proj.weight"),
            W3MlpOutputFormat::W3
        );
        assert_eq!(
            classify_w3_mlp_output("model.language_model.layers.0.mlp.down_proj.weight"),
            W3MlpOutputFormat::W3
        );
    }

    #[test]
    fn test_classify_w3_mlp_output_routes_moe_and_attention_to_q4() {
        assert_eq!(
            classify_w3_mlp_output("model.language_model.layers.0.mlp.experts.gate_proj.weight"),
            W3MlpOutputFormat::Q4
        );
        assert_eq!(
            classify_w3_mlp_output("model.language_model.layers.0.self_attn.q_proj.weight"),
            W3MlpOutputFormat::Q4
        );
        assert_eq!(
            classify_w3_mlp_output("model.language_model.embed_tokens.weight"),
            W3MlpOutputFormat::Q4
        );
    }

    #[test]
    fn test_classify_w3_mlp_output_routes_norms_to_f16() {
        assert_eq!(
            classify_w3_mlp_output("model.language_model.layers.0.input_layernorm.weight"),
            W3MlpOutputFormat::F16
        );
    }

    #[test]
    fn test_quantize_w3_mlp_happy_path_writes_w3_q4_f16() {
        let model_dir = tmp_dir("happy");
        let gate_vals: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 8.0).collect();
        let attn_vals: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 8.0).collect();
        let norm_vals: Vec<f32> = vec![1.0; 8];
        write_fake_model_dir(
            &model_dir,
            &[
                (
                    "model.language_model.layers.0.mlp.gate_proj.weight",
                    vec![2, 32],
                    gate_vals,
                ),
                (
                    "model.language_model.layers.0.self_attn.q_proj.weight",
                    vec![2, 32],
                    attn_vals,
                ),
                (
                    "model.language_model.layers.0.input_layernorm.weight",
                    vec![8],
                    norm_vals,
                ),
            ],
        );
        let output_dir = tmp_dir("happy_out");

        let report = quantize_w3_mlp(&model_dir, &output_dir, false).unwrap();
        assert_eq!(report.tensors_processed, 3);
        assert_eq!(report.tensors_w3, 1);
        assert_eq!(report.tensors_q4, 1);
        assert_eq!(report.tensors_f16, 1);

        assert!(
            output_dir
                .join("model_language_model_layers_0_mlp_gate_proj_weight.w3")
                .exists()
        );
        assert!(
            output_dir
                .join("model_language_model_layers_0_self_attn_q_proj_weight.q4")
                .exists()
        );
        assert!(
            output_dir
                .join("model_language_model_layers_0_input_layernorm_weight.f16")
                .exists()
        );
        assert!(output_dir.join("quantize_index.json").exists());

        // Round-trip the W3 file through the format layer's own loader.
        let loaded = crate::weights::w3_weights::load_w3_file(
            &output_dir.join("model_language_model_layers_0_mlp_gate_proj_weight.w3"),
        )
        .unwrap();
        assert_eq!(loaded.shape, vec![2, 32]);
        assert_eq!(loaded.original_len, 64);

        std::fs::remove_dir_all(&model_dir).ok();
        std::fs::remove_dir_all(&output_dir).ok();
    }

    /// Round-trip byte/value exactness: the `.w3` file the converter writes,
    /// once loaded back through `load_w3_file` (the format layer), must
    /// decode to *exactly* the same `W3Block`s (scale/bias f16 bit patterns
    /// and packed 3-bit codes) as quantizing the same source BF16 bytes
    /// directly with `quantize_bf16_to_w3` — not merely the same shape.
    /// `W3Block` derives `PartialEq`/`Eq`, so this is a bit-exact comparison,
    /// not an approximate one.
    #[test]
    fn test_quantize_w3_mlp_round_trip_is_byte_exact() {
        let model_dir = tmp_dir("roundtrip_exact");
        let name = "model.language_model.layers.0.mlp.gate_proj.weight";
        // 96 values spanning negative/positive/fractional bf16-representable
        // numbers across 3 blocks (32 elements each).
        let raw_vals: Vec<f32> = (0..96).map(|i| (i as f32 - 48.0) / 6.0).collect();
        write_fake_model_dir(&model_dir, &[(name, vec![3, 32], raw_vals.clone())]);
        let output_dir = tmp_dir("roundtrip_exact_out");

        let report = quantize_w3_mlp(&model_dir, &output_dir, false).unwrap();
        assert_eq!(report.tensors_w3, 1);

        let w3_path = output_dir.join("model_language_model_layers_0_mlp_gate_proj_weight.w3");
        let loaded = crate::weights::w3_weights::load_w3_file(&w3_path).unwrap();

        // Independently quantize the exact same bf16-truncated source values
        // (mirroring what write_fake_model_dir wrote to the safetensors shard)
        // and assert every block matches bit-for-bit.
        let raw_bf16: Vec<u16> = raw_vals
            .iter()
            .map(|&v| (v.to_bits() >> 16) as u16)
            .collect();
        let expected = quantize_bf16_to_w3(&raw_bf16, &[3, 32]).unwrap();

        assert_eq!(loaded.shape, expected.shape);
        assert_eq!(loaded.original_len, expected.original_len);
        assert_eq!(
            loaded.blocks.len(),
            expected.blocks.len(),
            "block count must match"
        );
        for (i, (got, want)) in loaded.blocks.iter().zip(expected.blocks.iter()).enumerate() {
            assert_eq!(got.scale, want.scale, "block {i} scale f16 bits mismatch");
            assert_eq!(got.bias, want.bias, "block {i} bias f16 bits mismatch");
            assert_eq!(got.packed, want.packed, "block {i} packed codes mismatch");
        }

        // And the dequantized f32 values must match exactly too (same f16
        // scale/bias, same codes -> identical arithmetic, no drift).
        let deq_loaded = crate::weights::w3_weights::dequantize_w3_to_f32(&loaded);
        let deq_expected = crate::weights::w3_weights::dequantize_w3_to_f32(&expected);
        assert_eq!(
            deq_loaded, deq_expected,
            "dequantized values must be bit-identical after the converter round-trip"
        );

        std::fs::remove_dir_all(&model_dir).ok();
        std::fs::remove_dir_all(&output_dir).ok();
    }

    #[test]
    fn test_quantize_w3_mlp_dry_run_writes_nothing() {
        let model_dir = tmp_dir("dry");
        write_fake_model_dir(
            &model_dir,
            &[(
                "model.language_model.layers.0.mlp.up_proj.weight",
                vec![32],
                vec![1.0; 32],
            )],
        );
        let output_dir = tmp_dir("dry_out");
        let _ = std::fs::remove_dir_all(&output_dir);

        let report = quantize_w3_mlp(&model_dir, &output_dir, true).unwrap();
        assert_eq!(report.tensors_w3, 1);
        assert!(
            !output_dir.exists(),
            "dry-run must not create the output directory"
        );

        std::fs::remove_dir_all(&model_dir).ok();
    }

    #[test]
    fn test_quantize_w3_mlp_rejects_f32_dense_mlp_tensor() {
        let model_dir = tmp_dir("f32reject");
        std::fs::create_dir_all(&model_dir).unwrap();
        let name = "model.language_model.layers.0.mlp.down_proj.weight";
        let values: Vec<f32> = vec![1.0; 32];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut header = serde_json::Map::new();
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": "F32", "shape": [32], "data_offsets": [0, data.len()]}),
        );
        let header_bytes = serde_json::to_string(&Value::Object(header))
            .unwrap()
            .into_bytes();
        let mut shard_bytes = Vec::new();
        shard_bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        shard_bytes.extend_from_slice(&header_bytes);
        shard_bytes.extend_from_slice(&data);
        std::fs::write(model_dir.join("model.safetensors"), &shard_bytes).unwrap();
        let mut weight_map = serde_json::Map::new();
        weight_map.insert(name.to_string(), Value::String("model.safetensors".into()));
        let index = serde_json::json!({"metadata": {}, "weight_map": Value::Object(weight_map)});
        std::fs::write(
            model_dir.join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        let output_dir = tmp_dir("f32reject_out");
        let result = quantize_w3_mlp(&model_dir, &output_dir, false);
        assert!(result.is_err(), "F32 dense MLP tensor must be rejected");

        std::fs::remove_dir_all(&model_dir).ok();
        std::fs::remove_dir_all(&output_dir).ok();
    }
}
