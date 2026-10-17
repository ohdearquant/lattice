use super::*;

#[test]
fn test_input_size_mismatch_display() {
    let err = FannError::InputSizeMismatch {
        expected: 10,
        actual: 5,
    };
    let msg = err.to_string();
    assert!(msg.contains("10"));
    assert!(msg.contains("5"));
}

#[test]
fn test_weight_count_mismatch_display() {
    let err = FannError::WeightCountMismatch {
        expected: 100,
        actual: 50,
    };
    assert!(err.to_string().contains("100"));
    assert!(err.to_string().contains("50"));
}

#[test]
fn test_invalid_layer_dimensions_display() {
    let err = FannError::InvalidLayerDimensions {
        inputs: 0,
        outputs: 10,
    };
    assert!(err.to_string().contains("inputs=0"));
}

#[test]
fn test_empty_network_display() {
    let err = FannError::EmptyNetwork;
    assert!(err.to_string().contains("no layers"));
}

#[test]
fn test_error_equality() {
    let err1 = FannError::InputSizeMismatch {
        expected: 10,
        actual: 5,
    };
    let err2 = FannError::InputSizeMismatch {
        expected: 10,
        actual: 5,
    };
    assert_eq!(err1, err2);
}

#[test]
fn test_shape_too_large_display() {
    let err = FannError::ShapeTooLarge {
        requested: 200_000_000,
        max: MAX_ALLOWED_ELEMENTS,
    };
    let msg = err.to_string();
    assert!(msg.contains("200000000"));
    assert!(msg.contains(&MAX_ALLOWED_ELEMENTS.to_string()));
}

#[test]
fn test_validate_allocation_size_ok() {
    assert!(validate_allocation_size(0).is_ok());
    assert!(validate_allocation_size(1000).is_ok());
    assert!(validate_allocation_size(MAX_ALLOWED_ELEMENTS).is_ok());
}

#[test]
fn test_validate_allocation_size_too_large() {
    let result = validate_allocation_size(MAX_ALLOWED_ELEMENTS + 1);
    assert!(matches!(result, Err(FannError::ShapeTooLarge { .. })));
}

#[test]
fn test_validate_layer_dimensions_ok() {
    assert!(validate_layer_dimensions(100, 100).is_ok());
    assert!(validate_layer_dimensions(1000, 1000).is_ok());
}

#[test]
fn test_validate_layer_dimensions_zero() {
    let result = validate_layer_dimensions(0, 100);
    assert!(matches!(
        result,
        Err(FannError::InvalidLayerDimensions { .. })
    ));

    let result = validate_layer_dimensions(100, 0);
    assert!(matches!(
        result,
        Err(FannError::InvalidLayerDimensions { .. })
    ));
}

#[test]
fn test_validate_layer_dimensions_too_large() {
    // 100_001 * 100_001 > MAX_ALLOWED_ELEMENTS (100M)
    let result = validate_layer_dimensions(100_001, 100_001);
    assert!(matches!(result, Err(FannError::ShapeTooLarge { .. })));
}

#[test]
fn test_validate_layer_dimensions_overflow() {
    // usize::MAX * 2 would overflow
    let result = validate_layer_dimensions(usize::MAX, 2);
    assert!(matches!(result, Err(FannError::ShapeTooLarge { .. })));
}
