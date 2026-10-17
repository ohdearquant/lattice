use super::*;

#[test]
fn test_builder_simple() {
    let network = NetworkBuilder::new()
        .input(4)
        .hidden(8, Activation::ReLU)
        .output(2, Activation::Softmax)
        .build()
        .unwrap();

    assert_eq!(network.num_inputs(), 4);
    assert_eq!(network.num_outputs(), 2);
    assert_eq!(network.num_layers(), 2);
}

#[test]
fn test_builder_no_input() {
    let result = NetworkBuilder::new().hidden(8, Activation::ReLU).build();

    assert!(matches!(result, Err(FannError::InvalidBuilder(_))));
}

#[test]
fn test_builder_no_layers() {
    let result = NetworkBuilder::new().input(4).build();

    assert!(matches!(result, Err(FannError::InvalidBuilder(_))));
}

#[test]
fn test_builder_zero_size() {
    let result = NetworkBuilder::new()
        .input(4)
        .hidden(0, Activation::ReLU)
        .build();

    assert!(matches!(result, Err(FannError::InvalidBuilder(_))));
}

#[test]
fn test_dense_shorthand() {
    let network = NetworkBuilder::new()
        .input(4)
        .dense(8)
        .dense(4)
        .output(2, Activation::Softmax)
        .build()
        .unwrap();

    // dense() uses ReLU
    assert_eq!(network.layer(0).unwrap().activation(), Activation::ReLU);
    assert_eq!(network.layer(1).unwrap().activation(), Activation::ReLU);
    assert_eq!(network.layer(2).unwrap().activation(), Activation::Softmax);
}

#[test]
fn test_build_with_seed_reproducible() {
    let network1 = NetworkBuilder::new()
        .input(4)
        .hidden(8, Activation::ReLU)
        .output(2, Activation::Softmax)
        .build_with_seed(42)
        .unwrap();

    let network2 = NetworkBuilder::new()
        .input(4)
        .hidden(8, Activation::ReLU)
        .output(2, Activation::Softmax)
        .build_with_seed(42)
        .unwrap();

    // Same seed should produce identical weights
    for i in 0..network1.num_layers() {
        let layer1 = network1.layer(i).unwrap();
        let layer2 = network2.layer(i).unwrap();
        assert_eq!(layer1.weights(), layer2.weights());
        assert_eq!(layer1.biases(), layer2.biases());
    }
}

#[test]
fn test_build_with_seed_different_seeds() {
    let network1 = NetworkBuilder::new()
        .input(4)
        .hidden(8, Activation::ReLU)
        .output(2, Activation::Softmax)
        .build_with_seed(42)
        .unwrap();

    let network2 = NetworkBuilder::new()
        .input(4)
        .hidden(8, Activation::ReLU)
        .output(2, Activation::Softmax)
        .build_with_seed(123)
        .unwrap();

    // Different seeds should produce different weights
    let layer1 = network1.layer(0).unwrap();
    let layer2 = network2.layer(0).unwrap();
    assert_ne!(layer1.weights(), layer2.weights());
}

#[test]
fn test_builder_rejects_hidden_softmax() {
    // Softmax on a hidden layer should be rejected.
    let result = NetworkBuilder::new()
        .input(2)
        .hidden(4, Activation::Softmax)
        .output(1, Activation::Linear)
        .build();
    assert!(
        matches!(result, Err(FannError::InvalidBuilder(_))),
        "expected InvalidBuilder, got {result:?}"
    );

    // build_with_seed must enforce the same rule.
    let result = NetworkBuilder::new()
        .input(2)
        .hidden(4, Activation::Softmax)
        .output(1, Activation::Linear)
        .build_with_seed(0);
    assert!(
        matches!(result, Err(FannError::InvalidBuilder(_))),
        "expected InvalidBuilder from build_with_seed, got {result:?}"
    );
}

#[test]
fn test_builder_output_softmax_is_allowed() {
    // Softmax on the output layer must continue to work.
    let result = NetworkBuilder::new()
        .input(4)
        .hidden(8, Activation::ReLU)
        .output(3, Activation::Softmax)
        .build();
    assert!(
        result.is_ok(),
        "output Softmax should be allowed, got {result:?}"
    );

    let result = NetworkBuilder::new()
        .input(4)
        .hidden(8, Activation::ReLU)
        .output(3, Activation::Softmax)
        .build_with_seed(7);
    assert!(
        result.is_ok(),
        "output Softmax with seed should be allowed, got {result:?}"
    );
}
