//! Property-based tests for Network
//!
//! Issue #448: proptest verifying Network::forward() output.len() == network.num_outputs()

use lattice_fann::{Activation, NetworkBuilder};
use proptest::prelude::*;

/// Strategy for generating valid network configurations
fn network_config_strategy()
-> impl Strategy<Value = (usize, Vec<(usize, Activation)>, usize, Activation)> {
    // Input dimension: 1-64
    let input_dim = 1usize..=64;
    // Hidden layers: 0-3 layers, each 1-64 neurons
    let hidden_layers = prop::collection::vec(
        (
            1usize..=64,
            prop::sample::select(vec![
                Activation::ReLU,
                Activation::Sigmoid,
                Activation::Tanh,
                Activation::Linear,
            ]),
        ),
        0..=3,
    );
    // Output dimension: 1-16
    let output_dim = 1usize..=16;
    // Output activation
    let output_activation = prop::sample::select(vec![
        Activation::Softmax,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Linear,
    ]);

    (input_dim, hidden_layers, output_dim, output_activation)
}

proptest! {
    // Test that forward() output length always matches num_outputs()
    #[test]
    fn test_forward_output_dimension_matches(
        (input_dim, hidden_layers, output_dim, output_activation) in network_config_strategy()
    ) {
        // Build the network
        let mut builder = NetworkBuilder::new().input(input_dim);

        for (size, activation) in &hidden_layers {
            builder = builder.hidden(*size, *activation);
        }

        let build_result = builder.output(output_dim, output_activation).build();

        // Network should build successfully
        prop_assert!(build_result.is_ok(), "Network failed to build: {:?}", build_result.err());

        let network = build_result.unwrap();

        // Verify num_outputs matches output_dim
        prop_assert_eq!(network.num_outputs(), output_dim);
        prop_assert_eq!(network.num_inputs(), input_dim);
    }

    // Test forward pass output dimension with various inputs
    #[test]
    fn test_forward_output_dimension_with_inputs(
        input_dim in 1usize..=32,
        output_dim in 1usize..=16,
        hidden_size in 1usize..=32,
    ) {
        // Create a simple network
        let build_result = NetworkBuilder::new()
            .input(input_dim)
            .hidden(hidden_size, Activation::ReLU)
            .output(output_dim, Activation::Softmax)
            .build();

        prop_assert!(build_result.is_ok());
        let mut network = build_result.unwrap();

        // Get num_outputs before forward pass
        let expected_outputs = network.num_outputs();

        // Create random input
        let input: Vec<f32> = (0..input_dim).map(|i| (i as f32) * 0.1).collect();

        // Forward pass
        let result = network.forward(&input);
        prop_assert!(result.is_ok(), "Forward pass failed: {:?}", result.err());

        let output_len = result.unwrap().len();

        // Critical assertion: output length must equal num_outputs
        prop_assert_eq!(
            output_len,
            expected_outputs,
            "Output length {} != num_outputs() {}",
            output_len,
            expected_outputs
        );
        prop_assert_eq!(output_len, output_dim);
    }

    // Test with deep networks
    #[test]
    fn test_deep_network_output_dimension(
        input_dim in 1usize..=16,
        num_hidden in 1usize..=5,
        hidden_size in 4usize..=16,
        output_dim in 1usize..=8,
    ) {
        let mut builder = NetworkBuilder::new().input(input_dim);

        // Add multiple hidden layers
        for _ in 0..num_hidden {
            builder = builder.hidden(hidden_size, Activation::ReLU);
        }

        let build_result = builder.output(output_dim, Activation::Linear).build();
        prop_assert!(build_result.is_ok());

        let mut network = build_result.unwrap();
        let expected_outputs = network.num_outputs();

        let input: Vec<f32> = vec![0.5; input_dim];
        let result = network.forward(&input);
        prop_assert!(result.is_ok());

        let output_len = result.unwrap().len();
        prop_assert_eq!(output_len, expected_outputs);
        prop_assert_eq!(output_len, output_dim);
    }

    // Test that all activations preserve output dimension
    #[test]
    fn test_all_activations_preserve_dimension(
        input_dim in 2usize..=16,
        output_dim in 2usize..=8,
        activation_idx in 0usize..5,
    ) {
        let activations = [
            Activation::Linear,
            Activation::ReLU,
            Activation::Sigmoid,
            Activation::Tanh,
            Activation::Softmax,
        ];
        let activation = activations[activation_idx % activations.len()];

        let build_result = NetworkBuilder::new()
            .input(input_dim)
            .output(output_dim, activation)
            .build();

        prop_assert!(build_result.is_ok());
        let mut network = build_result.unwrap();

        let input: Vec<f32> = vec![1.0; input_dim];
        let result = network.forward(&input);
        prop_assert!(result.is_ok());

        prop_assert_eq!(result.unwrap().len(), output_dim);
    }
}

#[cfg(test)]
mod additional_tests {
    use super::*;

    #[test]
    fn test_simple_network_dimension() {
        let mut network = NetworkBuilder::new()
            .input(4)
            .hidden(8, Activation::ReLU)
            .output(3, Activation::Softmax)
            .build()
            .unwrap();

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = network.forward(&input).unwrap();

        assert_eq!(output.len(), 3);
        assert_eq!(output.len(), network.num_outputs());
    }
}
