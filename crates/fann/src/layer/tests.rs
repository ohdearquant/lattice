use super::*;

#[test]
fn test_layer_creation() {
    let layer = Layer::new(10, 5, Activation::ReLU).unwrap();
    assert_eq!(layer.num_inputs(), 10);
    assert_eq!(layer.num_outputs(), 5);
    assert_eq!(layer.weights().len(), 50);
    assert_eq!(layer.biases().len(), 5);
    assert_eq!(layer.num_params(), 55);
}

#[test]
fn test_layer_zeros() {
    let layer = Layer::zeros(4, 3, Activation::Sigmoid).unwrap();
    assert!(layer.weights().iter().all(|&w| w == 0.0));
    assert!(layer.biases().iter().all(|&b| b == 0.0));
}

#[test]
fn test_layer_with_weights() {
    let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let biases = vec![0.1, 0.2];
    let layer = Layer::with_weights(3, 2, weights, biases, Activation::Linear).unwrap();

    assert_eq!(layer.num_inputs(), 3);
    assert_eq!(layer.num_outputs(), 2);
}

#[test]
fn test_layer_with_weights_mismatch() {
    let weights = vec![1.0, 2.0]; // Wrong size
    let biases = vec![0.1, 0.2];
    let result = Layer::with_weights(3, 2, weights, biases, Activation::Linear);
    assert!(matches!(result, Err(FannError::WeightCountMismatch { .. })));
}

#[test]
fn test_layer_invalid_dimensions() {
    let result = Layer::new(0, 5, Activation::ReLU);
    assert!(matches!(
        result,
        Err(FannError::InvalidLayerDimensions { .. })
    ));

    let result = Layer::new(5, 0, Activation::ReLU);
    assert!(matches!(
        result,
        Err(FannError::InvalidLayerDimensions { .. })
    ));
}

#[test]
fn test_forward_linear() {
    // Simple 2->2 layer with known weights
    let weights = vec![1.0, 0.0, 0.0, 1.0]; // Identity
    let biases = vec![0.0, 0.0];
    let layer = Layer::with_weights(2, 2, weights, biases, Activation::Linear).unwrap();

    let input = [3.0, 4.0];
    let mut output = [0.0, 0.0];
    layer.forward(&input, &mut output).unwrap();

    assert!((output[0] - 3.0).abs() < 1e-5);
    assert!((output[1] - 4.0).abs() < 1e-5);
}

#[test]
fn test_forward_with_bias() {
    let weights = vec![1.0, 1.0]; // 1x2 - sum inputs
    let biases = vec![0.5];
    let layer = Layer::with_weights(2, 1, weights, biases, Activation::Linear).unwrap();

    let input = [2.0, 3.0];
    let mut output = [0.0];
    layer.forward(&input, &mut output).unwrap();

    assert!((output[0] - 5.5).abs() < 1e-5); // 2 + 3 + 0.5
}

#[test]
fn test_forward_relu() {
    let weights = vec![1.0, -1.0]; // 2x1 transposed
    let biases = vec![0.0, 0.0];
    let layer = Layer::with_weights(1, 2, weights, biases, Activation::ReLU).unwrap();

    let input = [2.0];
    let mut output = [0.0, 0.0];
    layer.forward(&input, &mut output).unwrap();

    assert!((output[0] - 2.0).abs() < 1e-5); // ReLU(2) = 2
    assert!((output[1] - 0.0).abs() < 1e-5); // ReLU(-2) = 0
}

#[test]
fn test_forward_input_mismatch() {
    let layer = Layer::new(10, 5, Activation::ReLU).unwrap();
    let input = [1.0; 8]; // Wrong size
    let mut output = [0.0; 5];

    let result = layer.forward(&input, &mut output);
    assert!(matches!(result, Err(FannError::InputSizeMismatch { .. })));
}

#[test]
fn test_get_set_weight() {
    let mut layer = Layer::zeros(3, 2, Activation::Linear).unwrap();

    assert!(layer.set_weight(0, 1, 42.0));
    assert_eq!(layer.get_weight(0, 1), Some(42.0));

    assert!(!layer.set_weight(10, 10, 1.0)); // Out of bounds
    assert_eq!(layer.get_weight(10, 10), None);
}

#[test]
fn test_layer_oversized_allocation_rejected() {
    // 100_001 * 100_001 > 100M (MAX_ALLOWED_ELEMENTS)
    let result = Layer::new(100_001, 100_001, Activation::ReLU);
    assert!(matches!(result, Err(FannError::ShapeTooLarge { .. })));

    // Also test zeros
    let result = Layer::zeros(100_001, 100_001, Activation::ReLU);
    assert!(matches!(result, Err(FannError::ShapeTooLarge { .. })));
}
