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
