//! Safe checkpoint serialization using safetensors format.
//!
//! This module provides secure model weight serialization that prevents
//! arbitrary code execution vulnerabilities found in formats like Python pickle.
//!
//! # Security Guarantees
//!
//! The safetensors format provides:
//! - **No code execution**: Pure data format, cannot contain executable code
//! - **Memory safety**: Zero-copy deserialization with bounds checking
//! - **Integrity**: Header contains tensor shapes and data types
//!
//! # Format
//!
//! ```text
//! [8 bytes: header size (little-endian u64)]
//! [header: JSON metadata + tensor info]
//! [tensor data: raw bytes, aligned]
//! ```
//!
//! # Example
//!
//! ```ignore
//! use lattice_tune::registry::safetensors_io::{save_weights, load_weights};
//!
//! // Save weights securely
//! let weights: Vec<f32> = vec![0.1, 0.2, 0.3];
//! let bytes = save_weights(&weights, "layer0.weights")?;
//!
//! // Load weights safely (validates format, no code execution)
//! let loaded: Vec<f32> = load_weights(&bytes, "layer0.weights")?;
//! ```

use crate::error::{Result, TuneError};
use safetensors::Dtype;
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;

/// Save model weights to safetensors format.
///
/// # Arguments
///
/// * `weights` - Model weights as f32 slice
/// * `name` - Tensor name (e.g., "layer0.weights")
///
/// # Returns
///
/// Serialized bytes in safetensors format
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

/// Load model weights from safetensors format.
///
/// # Arguments
///
/// * `data` - Serialized safetensors bytes
/// * `name` - Tensor name to load
///
/// # Returns
///
/// Deserialized weights as Vec<f32>
///
/// # Security
///
/// This function safely deserializes weights without executing any code.
/// The safetensors format only contains data, not executable code.
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

/// Save multiple named tensors to safetensors format.
///
/// # Arguments
///
/// * `tensors` - Map of tensor name to f32 data
///
/// # Returns
///
/// Serialized bytes in safetensors format
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
        let original_len = tensors.get(name).map(|v| v.len()).unwrap_or(0);
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

/// Load all tensors from safetensors format.
///
/// # Arguments
///
/// * `data` - Serialized safetensors bytes
///
/// # Returns
///
/// Map of tensor names to f32 data
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
        let weights: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        result.insert(name.to_string(), weights);
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
}
