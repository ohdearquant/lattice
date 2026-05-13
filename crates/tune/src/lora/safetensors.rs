//! Load PEFT-format LoRA adapters from safetensors files.
//!
//! PEFT (Parameter-Efficient Fine-Tuning) saves LoRA weights with keys like:
//! ```text
//! base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight  -> (rank, d_in)
//! base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight  -> (d_out, rank)
//! base_model.model.model.layers.{i}.mlp.gate_proj.lora_A.weight     -> (rank, d_in)
//! base_model.model.model.layers.{i}.mlp.gate_proj.lora_B.weight     -> (d_out, rank)
//! ```
//!
//! This module parses those keys, extracts layer index and module name,
//! and loads the A/B matrix pairs into [`LoraLayer`] structs.

use super::{LoraAdapter, LoraConfig, LoraLayer};
use crate::error::TuneError;
use safetensors::Dtype;
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

/// A parsed PEFT tensor key.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PeftKey {
    /// Transformer layer index (0-based).
    layer_idx: usize,
    /// Module name: "q_proj", "v_proj", "gate_proj", etc.
    module: String,
    /// Whether this is the A or B matrix.
    matrix: LoraMatrix,
    /// Whether the tensor needs transposing (MLX format: A=(d_in, rank), B=(rank, d_out)).
    transposed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoraMatrix {
    A,
    B,
}

/// Parse a PEFT safetensors key into its components.
///
/// Accepted formats:
/// - `base_model.model.model.layers.{i}.self_attn.{module}.lora_A.weight`
/// - `base_model.model.model.layers.{i}.mlp.{module}.lora_B.weight`
///
/// Also handles the simpler HuggingFace format without the `base_model.model` prefix:
/// - `model.layers.{i}.self_attn.{module}.lora_A.weight`
/// - `model.layers.{i}.mlp.{module}.lora_B.weight`
///
/// Returns `None` for keys that don't match (e.g., non-LoRA metadata tensors).
fn parse_peft_key(key: &str) -> Option<PeftKey> {
    // Strip the trailing `.weight` if present
    let key = key.strip_suffix(".weight").unwrap_or(key);

    // Determine A or B (handles both PEFT uppercase and MLX lowercase)
    let (key, matrix, is_mlx) = if let Some(k) = key.strip_suffix(".lora_A") {
        (k, LoraMatrix::A, false)
    } else if let Some(k) = key.strip_suffix(".lora_B") {
        (k, LoraMatrix::B, false)
    } else if let Some(k) = key.strip_suffix(".lora_a") {
        (k, LoraMatrix::A, true)
    } else if let Some(k) = key.strip_suffix(".lora_b") {
        (k, LoraMatrix::B, true)
    } else {
        return None;
    };

    // Find "layers.{i}" segment and extract what follows
    let layers_marker = ".layers.";
    let layers_pos = key.find(layers_marker)?;
    let after_layers = &key[layers_pos + layers_marker.len()..];

    // Split: "{i}.{rest}" where rest is like "self_attn.q_proj" or "mlp.gate_proj"
    let dot_pos = after_layers.find('.')?;
    let layer_idx: usize = after_layers[..dot_pos].parse().ok()?;
    let rest = &after_layers[dot_pos + 1..];

    // Extract the module name (last segment after the block qualifier).
    // "self_attn.q_proj" -> "q_proj"
    // "mlp.gate_proj" -> "gate_proj"
    let module = rest.rsplit('.').next()?.to_string();
    if module.is_empty() {
        return None;
    }

    Some(PeftKey {
        layer_idx,
        module,
        matrix,
        transposed: is_mlx,
    })
}

/// Read a tensor from a safetensors file as a Vec<f32>.
///
/// Handles f32 and f16 dtypes (converting f16 to f32).
fn read_tensor_f32(
    tensors: &SafeTensors<'_>,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>), TuneError> {
    let tensor = tensors
        .tensor(name)
        .map_err(|e| TuneError::Serialization(format!("failed to read tensor '{name}': {e}")))?;

    let shape: Vec<usize> = tensor.shape().to_vec();
    let data = tensor.data();

    let values: Vec<f32> = match tensor.dtype() {
        Dtype::F32 => {
            if data.len() % 4 != 0 {
                return Err(TuneError::Serialization(format!(
                    "tensor '{name}' f32 data length {} not aligned to 4 bytes",
                    data.len()
                )));
            }
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        Dtype::F16 => {
            if data.len() % 2 != 0 {
                return Err(TuneError::Serialization(format!(
                    "tensor '{name}' f16 data length {} not aligned to 2 bytes",
                    data.len()
                )));
            }
            data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f16_to_f32(bits)
                })
                .collect()
        }
        Dtype::BF16 => {
            if data.len() % 2 != 0 {
                return Err(TuneError::Serialization(format!(
                    "tensor '{name}' bf16 data length {} not aligned to 2 bytes",
                    data.len()
                )));
            }
            data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    bf16_to_f32(bits)
                })
                .collect()
        }
        other => {
            return Err(TuneError::Serialization(format!(
                "tensor '{name}' has unsupported dtype {other:?}, expected F32/F16/BF16"
            )));
        }
    };

    if let Some((idx, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(TuneError::Serialization(format!(
            "tensor '{name}' contains non-finite value at index {idx}: {value}"
        )));
    }

    Ok((values, shape))
}

/// Convert IEEE 754 half-precision (f16) to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    if exp == 0 {
        if frac == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: value = (-1)^sign * 2^(-14) * (frac / 1024)
            let val = (frac as f32) / 1024.0 * (2.0f32).powi(-14);
            if sign == 1 { -val } else { val }
        }
    } else if exp == 31 {
        // Inf or NaN
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13))
        }
    } else {
        // Normalized: re-bias exponent from 15 to 127
        let f32_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (frac << 13))
    }
}

/// Convert bfloat16 to f32 (simply shift left by 16 bits).
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Load a LoRA adapter from a PEFT-format safetensors file.
///
/// Reads the file, parses all LoRA tensor keys, pairs A/B matrices per
/// (layer_idx, module), and assembles the adapter.
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read or is not valid safetensors
/// - A/B matrix pairs are incomplete (A without B or vice versa)
/// - Tensor shapes are inconsistent (ranks must match within a pair)
pub fn load_peft_safetensors(path: &Path) -> Result<LoraAdapter, TuneError> {
    const MAX_LORA_SIZE: u64 = 10 * 1024 * 1024 * 1024;
    let file_size = std::fs::metadata(path)
        .map_err(|e| {
            TuneError::Io(std::io::Error::new(
                e.kind(),
                format!("failed to read metadata for {}: {e}", path.display()),
            ))
        })?
        .len();
    if file_size > MAX_LORA_SIZE {
        return Err(TuneError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "LoRA file {} is {} bytes, exceeds maximum of {} bytes",
                path.display(),
                file_size,
                MAX_LORA_SIZE
            ),
        )));
    }
    let data = std::fs::read(path).map_err(|e| {
        TuneError::Io(std::io::Error::new(
            e.kind(),
            format!("failed to read LoRA adapter from {}: {e}", path.display()),
        ))
    })?;

    let tensors = SafeTensors::deserialize(&data).map_err(|e| {
        TuneError::Serialization(format!(
            "failed to parse safetensors from {}: {e}",
            path.display()
        ))
    })?;

    // Collect all parsed LoRA keys
    let names: Vec<String> = tensors.names().into_iter().map(String::from).collect();
    let mut a_tensors: HashMap<(usize, String), (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut b_tensors: HashMap<(usize, String), (Vec<f32>, Vec<usize>)> = HashMap::new();
    let mut target_modules = std::collections::BTreeSet::new();

    for name in &names {
        if let Some(peft_key) = parse_peft_key(name) {
            let key = (peft_key.layer_idx, peft_key.module.clone());
            let (data, shape) = read_tensor_f32(&tensors, name)?;
            target_modules.insert(peft_key.module.clone());

            // MLX format stores transposed: A=(d_in, rank), B=(rank, d_out)
            // PEFT format (what we expect): A=(rank, d_in), B=(d_out, rank)
            let (data, shape) = if peft_key.transposed && shape.len() == 2 {
                let (rows, cols) = (shape[0], shape[1]);
                let mut transposed = vec![0.0f32; data.len()];
                for r in 0..rows {
                    for c in 0..cols {
                        transposed[c * rows + r] = data[r * cols + c];
                    }
                }
                (transposed, vec![cols, rows])
            } else {
                (data, shape)
            };

            match peft_key.matrix {
                LoraMatrix::A => {
                    a_tensors.insert(key, (data, shape));
                }
                LoraMatrix::B => {
                    b_tensors.insert(key, (data, shape));
                }
            }
        }
    }

    if a_tensors.is_empty() && b_tensors.is_empty() {
        return Err(TuneError::Serialization(
            "LoRA adapter contains no lora_A/lora_B tensors".to_string(),
        ));
    }

    // Pair A and B matrices
    let mut layers: HashMap<(usize, String), LoraLayer> = HashMap::new();
    let mut rank: Option<usize> = None;

    for (key, (a_data, a_shape)) in &a_tensors {
        let (b_data, b_shape) = b_tensors.get(key).ok_or_else(|| {
            TuneError::Serialization(format!(
                "LoRA adapter has lora_A for layer {} module '{}' but no matching lora_B",
                key.0, key.1
            ))
        })?;

        // A shape: (rank, d_in), B shape: (d_out, rank)
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(TuneError::Serialization(format!(
                "LoRA matrices for layer {} module '{}' must be 2D, got A={:?} B={:?}",
                key.0, key.1, a_shape, b_shape
            )));
        }

        let a_rank = a_shape[0];
        let d_in = a_shape[1];
        let d_out = b_shape[0];
        let b_rank = b_shape[1];

        if a_rank != b_rank {
            return Err(TuneError::Serialization(format!(
                "LoRA rank mismatch for layer {} module '{}': A rank={}, B rank={}",
                key.0, key.1, a_rank, b_rank
            )));
        }

        // Verify all layers use the same rank
        match rank {
            None => rank = Some(a_rank),
            Some(r) if r != a_rank => {
                return Err(TuneError::Serialization(format!(
                    "inconsistent LoRA ranks: first seen rank={}, layer {} module '{}' has rank={}",
                    r, key.0, key.1, a_rank
                )));
            }
            _ => {}
        }

        // Verify data sizes match shapes
        let expected_a = a_rank.checked_mul(d_in).ok_or_else(|| {
            TuneError::Serialization(format!(
                "LoRA A shape overflow for layer {} module '{}': {} * {}",
                key.0, key.1, a_rank, d_in
            ))
        })?;
        if a_data.len() != expected_a {
            return Err(TuneError::Serialization(format!(
                "LoRA A data size mismatch for layer {} module '{}': expected {}, got {}",
                key.0,
                key.1,
                expected_a,
                a_data.len()
            )));
        }
        let expected_b = d_out.checked_mul(b_rank).ok_or_else(|| {
            TuneError::Serialization(format!(
                "LoRA B shape overflow for layer {} module '{}': {} * {}",
                key.0, key.1, d_out, b_rank
            ))
        })?;
        if b_data.len() != expected_b {
            return Err(TuneError::Serialization(format!(
                "LoRA B data size mismatch for layer {} module '{}': expected {}, got {}",
                key.0,
                key.1,
                expected_b,
                b_data.len()
            )));
        }

        layers.insert(
            key.clone(),
            LoraLayer {
                a: a_data.clone(),
                b: b_data.clone(),
                d_in,
                d_out,
                rank: a_rank,
            },
        );
    }

    // Check for orphaned B matrices
    for key in b_tensors.keys() {
        if !a_tensors.contains_key(key) {
            return Err(TuneError::Serialization(format!(
                "LoRA adapter has lora_B for layer {} module '{}' but no matching lora_A",
                key.0, key.1
            )));
        }
    }

    let rank = rank.unwrap_or(0);

    Ok(LoraAdapter {
        config: LoraConfig {
            rank,
            alpha: rank as f32, // default: alpha = rank => scale = 1.0
            target_modules: target_modules.into_iter().collect(),
        },
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_peft_key_self_attn() {
        let key = "base_model.model.model.layers.5.self_attn.q_proj.lora_A.weight";
        let parsed = parse_peft_key(key).unwrap();
        assert_eq!(parsed.layer_idx, 5);
        assert_eq!(parsed.module, "q_proj");
        assert_eq!(parsed.matrix, LoraMatrix::A);
    }

    #[test]
    fn test_parse_peft_key_mlp() {
        let key = "base_model.model.model.layers.12.mlp.gate_proj.lora_B.weight";
        let parsed = parse_peft_key(key).unwrap();
        assert_eq!(parsed.layer_idx, 12);
        assert_eq!(parsed.module, "gate_proj");
        assert_eq!(parsed.matrix, LoraMatrix::B);
    }

    #[test]
    fn test_parse_peft_key_simple_prefix() {
        let key = "model.layers.0.self_attn.v_proj.lora_A.weight";
        let parsed = parse_peft_key(key).unwrap();
        assert_eq!(parsed.layer_idx, 0);
        assert_eq!(parsed.module, "v_proj");
        assert_eq!(parsed.matrix, LoraMatrix::A);
    }

    #[test]
    fn test_parse_peft_key_no_weight_suffix() {
        let key = "base_model.model.model.layers.3.mlp.up_proj.lora_B";
        let parsed = parse_peft_key(key).unwrap();
        assert_eq!(parsed.layer_idx, 3);
        assert_eq!(parsed.module, "up_proj");
        assert_eq!(parsed.matrix, LoraMatrix::B);
    }

    #[test]
    fn test_parse_peft_key_rejects_non_lora() {
        assert!(parse_peft_key("model.layers.0.self_attn.q_proj.weight").is_none());
        assert!(parse_peft_key("some_random_tensor").is_none());
        assert!(parse_peft_key("").is_none());
    }

    #[test]
    fn test_parse_mlx_key_lowercase() {
        let key = "model.layers.3.self_attn.q_proj.lora_a";
        let parsed = parse_peft_key(key).unwrap();
        assert_eq!(parsed.layer_idx, 3);
        assert_eq!(parsed.module, "q_proj");
        assert_eq!(parsed.matrix, LoraMatrix::A);
        assert!(parsed.transposed);
    }

    #[test]
    fn test_parse_mlx_key_mlp() {
        let key = "model.layers.7.mlp.down_proj.lora_b";
        let parsed = parse_peft_key(key).unwrap();
        assert_eq!(parsed.layer_idx, 7);
        assert_eq!(parsed.module, "down_proj");
        assert_eq!(parsed.matrix, LoraMatrix::B);
        assert!(parsed.transposed);
    }

    #[test]
    fn test_peft_key_not_transposed() {
        let key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
        let parsed = parse_peft_key(key).unwrap();
        assert!(!parsed.transposed);
    }

    #[test]
    fn test_parse_peft_key_all_target_modules() {
        let modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ];
        for (module, block) in modules.iter().zip(
            [
                "self_attn",
                "self_attn",
                "self_attn",
                "self_attn",
                "mlp",
                "mlp",
                "mlp",
            ]
            .iter(),
        ) {
            let key = format!("base_model.model.model.layers.7.{block}.{module}.lora_A.weight");
            let parsed = parse_peft_key(&key).unwrap();
            assert_eq!(parsed.layer_idx, 7);
            assert_eq!(parsed.module, *module);
            assert_eq!(parsed.matrix, LoraMatrix::A);
        }
    }

    #[test]
    fn test_f16_to_f32_roundtrip() {
        // Test a few known values
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // -1.0 in f16 = 0xBC00
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
        // 0.0 in f16 = 0x0000
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // 0.5 in f16 = 0x3800
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32() {
        // 1.0 in bf16 = 0x3F80 (same as upper 16 bits of f32 1.0)
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        // 0.0
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }

    /// Helper: create a PEFT-format safetensors file with known LoRA weights.
    fn write_test_peft_safetensors(path: &std::path::Path, rank: usize, d_in: usize, d_out: usize) {
        use safetensors::Dtype;
        use safetensors::tensor::{TensorView, serialize};
        use std::collections::HashMap;

        // Layer 0, q_proj: A=(rank, d_in), B=(d_out, rank)
        let a_data: Vec<f32> = (0..rank * d_in).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..d_out * rank).map(|i| (i as f32) * 0.1).collect();

        // Layer 2, gate_proj: same dimensions
        let a2_data: Vec<f32> = (0..rank * d_in).map(|i| (i as f32) * -0.01).collect();
        let b2_data: Vec<f32> = (0..d_out * rank).map(|i| (i as f32) * -0.1).collect();

        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let a2_bytes: Vec<u8> = a2_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b2_bytes: Vec<u8> = b2_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut tensors = HashMap::new();
        tensors.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            TensorView::new(Dtype::F32, vec![rank, d_in], &a_bytes).unwrap(),
        );
        tensors.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            TensorView::new(Dtype::F32, vec![d_out, rank], &b_bytes).unwrap(),
        );
        tensors.insert(
            "base_model.model.model.layers.2.mlp.gate_proj.lora_A.weight".to_string(),
            TensorView::new(Dtype::F32, vec![rank, d_in], &a2_bytes).unwrap(),
        );
        tensors.insert(
            "base_model.model.model.layers.2.mlp.gate_proj.lora_B.weight".to_string(),
            TensorView::new(Dtype::F32, vec![d_out, rank], &b2_bytes).unwrap(),
        );

        let bytes = serialize(&tensors, &None).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn test_load_peft_safetensors_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter.safetensors");

        let rank = 4;
        let d_in = 8;
        let d_out = 8;
        write_test_peft_safetensors(&path, rank, d_in, d_out);

        let adapter = load_peft_safetensors(&path).unwrap();

        // Check config
        assert_eq!(adapter.config.rank, rank);
        assert_eq!(adapter.config.target_modules.len(), 2);
        assert!(
            adapter
                .config
                .target_modules
                .contains(&"q_proj".to_string())
        );
        assert!(
            adapter
                .config
                .target_modules
                .contains(&"gate_proj".to_string())
        );

        // Check we got both layers
        assert_eq!(adapter.layers.len(), 2);
        assert!(adapter.layers.contains_key(&(0, "q_proj".to_string())));
        assert!(adapter.layers.contains_key(&(2, "gate_proj".to_string())));

        // Check dimensions
        let q_lora = &adapter.layers[&(0, "q_proj".to_string())];
        assert_eq!(q_lora.rank, rank);
        assert_eq!(q_lora.d_in, d_in);
        assert_eq!(q_lora.d_out, d_out);
        assert_eq!(q_lora.a.len(), rank * d_in);
        assert_eq!(q_lora.b.len(), d_out * rank);

        // Verify actual values for A[0]
        assert!((q_lora.a[0] - 0.0).abs() < 1e-6);
        assert!((q_lora.a[1] - 0.01).abs() < 1e-6);
        assert!((q_lora.a[2] - 0.02).abs() < 1e-6);

        // Check gate_proj layer
        let g_lora = &adapter.layers[&(2, "gate_proj".to_string())];
        assert_eq!(g_lora.rank, rank);
        assert!((g_lora.a[1] - (-0.01)).abs() < 1e-6);
    }

    #[test]
    fn test_load_mlx_safetensors_transposed() {
        // MLX format: A=(d_in, rank), B=(rank, d_out) with lowercase keys
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapters.safetensors");

        use safetensors::Dtype;
        use safetensors::tensor::{TensorView, serialize};

        // rank=2, d_in=4, d_out=3
        // MLX A: (d_in=4, rank=2) -> after transpose: (rank=2, d_in=4)
        // MLX B: (rank=2, d_out=3) -> after transpose: (d_out=3, rank=2)
        let a_data: Vec<f32> = vec![
            // 4x2 matrix (d_in=4, rank=2)
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ];
        let b_data: Vec<f32> = vec![
            // 2x3 matrix (rank=2, d_out=3)
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
        ];

        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.lora_a".to_string(),
            TensorView::new(Dtype::F32, vec![4, 2], &a_bytes).unwrap(),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.lora_b".to_string(),
            TensorView::new(Dtype::F32, vec![2, 3], &b_bytes).unwrap(),
        );

        let bytes = serialize(&tensors, &None).unwrap();
        std::fs::write(&path, bytes).unwrap();

        let adapter = load_peft_safetensors(&path).unwrap();
        assert_eq!(adapter.config.rank, 2);

        let lora = &adapter.layers[&(0, "q_proj".to_string())];
        assert_eq!(lora.rank, 2);
        assert_eq!(lora.d_in, 4);
        assert_eq!(lora.d_out, 3);

        // Verify transpose: original A[0,0]=1, A[1,0]=3, A[2,0]=5, A[3,0]=7
        // Transposed A[0,0]=1, A[0,1]=3, A[0,2]=5, A[0,3]=7
        assert!((lora.a[0] - 1.0).abs() < 1e-6); // row 0
        assert!((lora.a[1] - 3.0).abs() < 1e-6);
        assert!((lora.a[2] - 5.0).abs() < 1e-6);
        assert!((lora.a[3] - 7.0).abs() < 1e-6);
        assert!((lora.a[4] - 2.0).abs() < 1e-6); // row 1
        assert!((lora.a[5] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_load_peft_safetensors_apply_correctness() {
        // End-to-end: write safetensors, load, apply, verify numerical result.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("adapter.safetensors");

        // rank=1, d_in=2, d_out=2 for simplicity
        // A = [[0.0, 0.01]]  (1x2)
        // B = [[0.0], [0.1]] (2x1)
        use safetensors::Dtype;
        use safetensors::tensor::{TensorView, serialize};

        let a_data: Vec<f32> = vec![0.0, 1.0]; // A = [[0, 1]]
        let b_data: Vec<f32> = vec![1.0, 0.0]; // B = [[1], [0]]
        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "model.layers.0.self_attn.v_proj.lora_A.weight".to_string(),
            TensorView::new(Dtype::F32, vec![1, 2], &a_bytes).unwrap(),
        );
        tensors.insert(
            "model.layers.0.self_attn.v_proj.lora_B.weight".to_string(),
            TensorView::new(Dtype::F32, vec![2, 1], &b_bytes).unwrap(),
        );

        let bytes = serialize(&tensors, &None).unwrap();
        std::fs::write(&path, bytes).unwrap();

        let adapter = load_peft_safetensors(&path).unwrap();
        assert_eq!(adapter.config.rank, 1);
        // Default alpha = rank = 1, so scale = 1.0

        let lora = &adapter.layers[&(0, "v_proj".to_string())];
        // x = [3, 5]
        // A @ x = [0*3 + 1*5] = [5]
        // B @ [5] = [1*5, 0*5] = [5, 0]
        // scale = 1.0 => output += [5, 0]
        let x = [3.0f32, 5.0];
        let mut output = [10.0, 20.0];
        super::super::apply_lora(lora, adapter.config.scale(), &x, &mut output);

        assert!((output[0] - 15.0).abs() < 1e-6);
        assert!((output[1] - 20.0).abs() < 1e-6);
    }

    fn write_raw_safetensors(path: &std::path::Path, header: &str, data: &[u8]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn test_rejects_truncated_safetensors_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.safetensors");
        std::fs::write(&path, 128u64.to_le_bytes()).unwrap();

        let err = load_peft_safetensors(&path).unwrap_err();
        assert!(err.to_string().contains("failed to parse safetensors"));
    }

    #[test]
    fn test_rejects_file_without_lora_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.safetensors");
        let header = r#"{"metadata.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        write_raw_safetensors(&path, header, &1.0f32.to_le_bytes());

        let err = load_peft_safetensors(&path).unwrap_err();
        assert!(err.to_string().contains("no lora_A/lora_B tensors"));
    }

    #[test]
    fn test_rejects_shape_product_overflow() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("overflow.safetensors");
        let max = usize::MAX;
        let header = format!(
            r#"{{"model.layers.0.self_attn.q_proj.lora_A.weight":{{"dtype":"F32","shape":[{max},2],"data_offsets":[0,0]}},"model.layers.0.self_attn.q_proj.lora_B.weight":{{"dtype":"F32","shape":[1,{max}],"data_offsets":[0,0]}}}}"#
        );
        write_raw_safetensors(&path, &header, &[]);

        let err = load_peft_safetensors(&path).unwrap_err();
        assert!(
            err.to_string().contains("shape overflow")
                || err.to_string().contains("failed to parse safetensors")
        );
    }

    #[test]
    fn test_rejects_non_finite_tensor_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nan.safetensors");

        use safetensors::Dtype;
        use safetensors::tensor::{TensorView, serialize};

        let a_data: Vec<f32> = vec![f32::NAN, 1.0];
        let b_data: Vec<f32> = vec![1.0, 2.0];
        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut tensors = std::collections::HashMap::new();
        tensors.insert(
            "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            TensorView::new(Dtype::F32, vec![1, 2], &a_bytes).unwrap(),
        );
        tensors.insert(
            "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            TensorView::new(Dtype::F32, vec![2, 1], &b_bytes).unwrap(),
        );

        let bytes = serialize(&tensors, &None).unwrap();
        std::fs::write(&path, bytes).unwrap();

        let err = load_peft_safetensors(&path).unwrap_err();
        assert!(err.to_string().contains("non-finite"));
    }
}
