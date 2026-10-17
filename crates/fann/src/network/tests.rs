use super::*;
use crate::activation::Activation;

#[test]
fn test_forward_simple() {
    let mut network = NetworkBuilder::new()
        .input(2)
        .output(2, Activation::Linear)
        .build()
        .unwrap();

    let input = [1.0, 2.0];
    let output = network.forward(&input).unwrap();
    assert_eq!(output.len(), 2);
}

#[test]
fn test_forward_multilayer() {
    let mut network = NetworkBuilder::new()
        .input(4)
        .hidden(8, Activation::ReLU)
        .hidden(4, Activation::ReLU)
        .output(2, Activation::Softmax)
        .build()
        .unwrap();

    let input = [1.0, 2.0, 3.0, 4.0];
    let output = network.forward(&input).unwrap();

    // Softmax output should sum to 1
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_forward_input_mismatch() {
    let mut network = NetworkBuilder::new()
        .input(4)
        .output(2, Activation::Linear)
        .build()
        .unwrap();

    let input = [1.0, 2.0]; // Wrong size
    let result = network.forward(&input);
    assert!(matches!(result, Err(FannError::InputSizeMismatch { .. })));
}

#[test]
fn release_nan_detection() {
    let layer = Layer::with_weights(1, 1, vec![1.0], vec![0.0], Activation::Linear).unwrap();
    let mut network = Network::new(vec![layer]).unwrap();

    let result = network.forward(&[f32::NAN]);

    assert!(matches!(
        result,
        Err(FannError::NumericInstability(message)) if message.contains("NaN")
    ));
}

#[test]
fn network_new_rejects_hidden_softmax() {
    let hidden = Layer::new(2, 4, Activation::Softmax).unwrap();
    let output = Layer::new(4, 1, Activation::Linear).unwrap();

    let result = Network::new(vec![hidden, output]);

    assert!(
        matches!(result, Err(FannError::InvalidBuilder(_))),
        "expected InvalidBuilder, got {result:?}"
    );
}

#[test]
fn test_total_params() {
    let network = NetworkBuilder::new()
        .input(10)
        .hidden(20, Activation::ReLU) // 10*20 + 20 = 220
        .output(5, Activation::Linear) // 20*5 + 5 = 105
        .build()
        .unwrap();

    assert_eq!(network.total_params(), 220 + 105);
}

#[test]
fn test_architecture_string() {
    let network = NetworkBuilder::new()
        .input(784)
        .hidden(128, Activation::ReLU)
        .output(10, Activation::Softmax)
        .build()
        .unwrap();

    let arch = network.architecture();
    assert!(arch.contains("784"));
    assert!(arch.contains("128"));
    assert!(arch.contains("10"));
    assert!(arch.contains("ReLU"));
    assert!(arch.contains("Softmax"));
}

#[test]
fn test_layer_access() {
    let network = NetworkBuilder::new()
        .input(4)
        .hidden(8, Activation::ReLU)
        .output(2, Activation::Linear)
        .build()
        .unwrap();

    assert!(network.layer(0).is_some());
    assert!(network.layer(1).is_some());
    assert!(network.layer(2).is_none());

    assert_eq!(network.layers().len(), 2);
}

#[test]
fn test_activations_access() {
    let mut network = NetworkBuilder::new()
        .input(2)
        .hidden(4, Activation::ReLU)
        .output(2, Activation::Linear)
        .build()
        .unwrap();

    let input = [1.0, 2.0];
    network.forward(&input).unwrap();

    // Can access intermediate activations
    let hidden_activations = network.activations(0).unwrap();
    assert_eq!(hidden_activations.len(), 4);

    let output_activations = network.activations(1).unwrap();
    assert_eq!(output_activations.len(), 2);
}

#[test]
fn test_clone_independence() {
    let mut network1 = NetworkBuilder::new()
        .input(2)
        .output(2, Activation::Linear)
        .build()
        .unwrap();

    let mut network2 = network1.clone();

    let input = [1.0, 2.0];
    let output1 = network1.forward(&input).unwrap().to_vec();
    let output2 = network2.forward(&input).unwrap().to_vec();

    // Cloned networks should produce same output
    for (a, b) in output1.iter().zip(output2.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}
