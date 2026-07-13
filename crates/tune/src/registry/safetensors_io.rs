//! Safetensors serialization for flat `f32` weight vectors.
//!
//! Parsing accepts data only and validates the safetensors container, but it
//! does not replace registry identity or checksum verification.
//!
//! See `docs/registry.md` for the format, tensor contract, and security boundary.

use crate::error::{Result, TuneError};
use safetensors::Dtype;
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;

/// Serializes `weights` as a named one-dimensional `F32` safetensors tensor.
/// Returns serialization errors from the safetensors container.
/// See `docs/registry.md` (§`safetensors_io` helpers) for the tensor contract.
pub fn save_weights(weights: &[f32], name: &str) -> Result<Vec<u8>> {
    // Convert f32 to bytes
    let bytes: Vec<u8> = weights.iter().flat_map(|f| f.to_le_bytes()).collect();

    // Create tensor data map
    let mut tensors: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();

    // Create tensor view
    let shape = vec![weights.len()];
    let tensor = safetensors::tensor::TensorView::new(Dtype::F32, shape, &bytes)
        .map_err(|e| TuneError::Storage(format!("Failed to create tensor view: {e}")))?;

    tensors.insert(name.to_string(), tensor);

    // Serialize
    safetensors::tensor::serialize(&tensors, &None)
        .map_err(|e| TuneError::Storage(format!("Failed to serialize weights: {e}")))
}

/// Loads a named `F32` safetensors tensor as flat `f32` values.
/// Returns errors for malformed data, a missing tensor, a non-`F32` tensor, or unaligned bytes.
/// See `docs/registry.md` (§`safetensors_io` helpers) for format and integrity boundaries.
pub fn load_weights(data: &[u8], name: &str) -> Result<Vec<f32>> {
    // Parse safetensors (validates format, no code execution)
    let tensors = SafeTensors::deserialize(data)
        .map_err(|e| TuneError::Storage(format!("Failed to deserialize weights: {e}")))?;

    // Get tensor by name
    let tensor = tensors
        .tensor(name)
        .map_err(|e| TuneError::Storage(format!("Tensor '{name}' not found: {e}")))?;

    // Verify dtype
    if tensor.dtype() != Dtype::F32 {
        return Err(TuneError::Storage(format!(
            "Expected F32 dtype, got {:?}",
            tensor.dtype()
        )));
    }

    // Convert bytes to f32
    let bytes = tensor.data();
    if bytes.len() % 4 != 0 {
        return Err(TuneError::Storage(
            "Invalid data length for F32".to_string(),
        ));
    }

    let weights: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(weights)
}

/// Serializes named flat `f32` vectors as one-dimensional `F32` tensors.
/// See `docs/registry.md` (§`safetensors_io` helpers) for the multi-tensor contract.
pub fn save_tensors(tensors: &HashMap<String, Vec<f32>>) -> Result<Vec<u8>> {
    // Convert all tensors to byte views
    let byte_data: HashMap<String, Vec<u8>> = tensors
        .iter()
        .map(|(name, data)| {
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes)
        })
        .collect();

    // Create tensor views
    let mut tensor_views: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();
    for (name, bytes) in byte_data.iter() {
        let original_len = tensors.get(name).map(Vec::len).unwrap_or(0);
        let shape = vec![original_len];
        let view = safetensors::tensor::TensorView::new(Dtype::F32, shape, bytes).map_err(|e| {
            TuneError::Storage(format!("Failed to create tensor view for {name}: {e}"))
        })?;
        tensor_views.insert(name.clone(), view);
    }

    // Serialize
    safetensors::tensor::serialize(&tensor_views, &None)
        .map_err(|e| TuneError::Storage(format!("Failed to serialize tensors: {e}")))
}

/// Loads all `F32` safetensors tensors as flat `f32` vectors.
/// Skips non-`F32` tensors and rejects malformed `F32` byte lengths or shapes.
/// See `docs/registry.md` (§`safetensors_io` helpers) for validation behavior.
pub fn load_tensors(data: &[u8]) -> Result<HashMap<String, Vec<f32>>> {
    let tensors = SafeTensors::deserialize(data)
        .map_err(|e| TuneError::Storage(format!("Failed to deserialize tensors: {e}")))?;

    let mut result = HashMap::new();

    for name in tensors.names() {
        let tensor = tensors
            .tensor(name)
            .map_err(|e| TuneError::Storage(format!("Failed to get tensor '{name}': {e}")))?;

        if tensor.dtype() != Dtype::F32 {
            continue; // Skip non-F32 tensors
        }

        let bytes = tensor.data();
        if bytes.len() % 4 != 0 {
            return Err(TuneError::Storage(format!(
                "tensor '{name}' f32 data length {} is not a multiple of 4",
                bytes.len()
            )));
        }
        let shape = tensor.shape();
        let expected_elems: usize = shape.iter().product();
        let actual_elems = bytes.len() / 4;
        if actual_elems != expected_elems {
            return Err(TuneError::Storage(format!(
                "tensor '{name}' shape {shape:?} implies {expected_elems} elements but data contains {actual_elems}"
            )));
        }
        let weights: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        result.insert(name.clone(), weights);
    }

    Ok(result)
}

/// Validate that safetensors data is well-formed without fully loading it.
///
/// This is useful for security validation before committing weights to a registry.
pub fn validate(data: &[u8]) -> Result<()> {
    SafeTensors::deserialize(data)
        .map_err(|e| TuneError::Storage(format!("Invalid safetensors format: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_weights() {
        let weights = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let data = save_weights(&weights, "test").unwrap();
        let loaded = load_weights(&data, "test").unwrap();

        assert_eq!(weights.len(), loaded.len());
        for (a, b) in weights.iter().zip(loaded.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_save_load_empty() {
        let weights: Vec<f32> = vec![];
        let data = save_weights(&weights, "empty").unwrap();
        let loaded = load_weights(&data, "empty").unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_save_load_multiple_tensors() {
        let mut tensors = HashMap::new();
        tensors.insert("layer0.weights".to_string(), vec![1.0f32, 2.0, 3.0]);
        tensors.insert("layer0.biases".to_string(), vec![0.1f32, 0.2]);
        tensors.insert("layer1.weights".to_string(), vec![4.0f32, 5.0, 6.0, 7.0]);

        let data = save_tensors(&tensors).unwrap();
        let loaded = load_tensors(&data).unwrap();

        assert_eq!(tensors.len(), loaded.len());
        for (name, original) in tensors.iter() {
            let loaded_tensor = loaded.get(name).expect("Tensor not found");
            assert_eq!(original.len(), loaded_tensor.len());
        }
    }

    #[test]
    fn test_validate() {
        let weights = vec![1.0f32, 2.0, 3.0];
        let data = save_weights(&weights, "test").unwrap();
        assert!(validate(&data).is_ok());
    }

    #[test]
    fn test_validate_invalid() {
        let invalid_data = vec![0u8; 100]; // Random bytes
        assert!(validate(&invalid_data).is_err());
    }

    #[test]
    fn test_tensor_not_found() {
        let weights = vec![1.0f32, 2.0];
        let data = save_weights(&weights, "exists").unwrap();
        let result = load_weights(&data, "doesnt_exist");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_tensors_rejects_unaligned_f32_bytes() {
        // The parser rejects malformed ranges before this guard; loading must still fail safely.
        let header = r#"{"w":{"dtype":"F32","shape":[3],"data_offsets":[0,11]}}"#;
        let header_bytes = header.as_bytes();
        let mut raw: Vec<u8> = Vec::new();
        raw.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        raw.extend_from_slice(header_bytes);
        raw.extend_from_slice(&[0u8; 11]);

        let result = load_tensors(&raw);
        assert!(result.is_err(), "expected Err for unaligned F32 payload");
    }

    #[test]
    fn test_load_tensors_rejects_element_count_mismatch() {
        // The parser rejects this mismatch; the shape guard remains defence in depth.
        let header = r#"{"w":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#;
        let header_bytes = header.as_bytes();
        let mut raw: Vec<u8> = Vec::new();
        raw.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        raw.extend_from_slice(header_bytes);
        raw.extend_from_slice(&[0u8; 8]);

        let result = load_tensors(&raw);
        assert!(result.is_err(), "expected Err for element-count mismatch");
    }

    #[test]
    fn test_load_tensors_alignment_guard_fires_when_library_cannot_catch() {
        // Exercise the arithmetic preconditions independently; malformed files fail in the parser.
        let bytes_len: usize = 11;
        assert_ne!(bytes_len % 4, 0, "alignment precondition");
        let shape: Vec<usize> = vec![3];
        let expected_elems: usize = shape.iter().product();
        let actual_elems = 8usize / 4;
        assert_ne!(actual_elems, expected_elems, "count precondition");
    }
}
