//! Fluent construction of dense feedforward networks.
//!
//! The builder records layer sizes and activations, then validates and creates
//! `Layer` objects at build time. It can use either entropy or a caller seed
//! for Xavier/Glorot initialization.
//!
//! See `docs/network.md` for build-time validation and buffer allocation.

use crate::activation::Activation;
use crate::error::{FannError, FannResult};
use crate::layer::Layer;
use crate::network::Network;

/// Fluent builder for a dense feedforward network.
/// See [`docs/network.md`](../../docs/network.md#networkbuilder) for construction examples.
#[derive(Debug, Clone, Default)]
pub struct NetworkBuilder {
    input_size: Option<usize>,
    layers: Vec<(usize, Activation)>,
}

impl NetworkBuilder {
    /// Create a new network builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input size for the network
    ///
    /// This must be called before adding any layers.
    pub fn input(mut self, size: usize) -> Self {
        self.input_size = Some(size);
        self
    }

    /// Add a hidden layer with the specified size and activation
    pub fn hidden(mut self, size: usize, activation: Activation) -> Self {
        self.layers.push((size, activation));
        self
    }

    /// Add the output layer with the specified size and activation
    ///
    /// This is semantically the same as `hidden` but improves readability.
    pub fn output(mut self, size: usize, activation: Activation) -> Self {
        self.layers.push((size, activation));
        self
    }

    /// Add a layer with default ReLU activation
    pub fn dense(self, size: usize) -> Self {
        self.hidden(size, Activation::ReLU)
    }

    /// Builds the configured network with entropy-seeded initialization.
    ///
    /// Returns an error for a missing or zero input, no layers, a zero-width
    /// layer, or Softmax before the output layer.
    pub fn build(self) -> FannResult<Network> {
        let input_size = self
            .input_size
            .ok_or_else(|| FannError::InvalidBuilder("Input size not specified".into()))?;

        if self.layers.is_empty() {
            return Err(FannError::InvalidBuilder("No layers specified".into()));
        }

        if input_size == 0 {
            return Err(FannError::InvalidBuilder("Input size cannot be 0".into()));
        }

        let mut layers = Vec::with_capacity(self.layers.len());
        let mut prev_size = input_size;

        for (i, (size, activation)) in self.layers.into_iter().enumerate() {
            if size == 0 {
                return Err(FannError::InvalidBuilder(format!("Layer {i} has size 0")));
            }
            let layer = Layer::new(prev_size, size, activation)?;
            layers.push(layer);
            prev_size = size;
        }

        Network::new(layers)
    }

    /// Builds the configured network with deterministic initialization from `seed`.
    ///
    /// The same architecture and seed produce identical parameters.
    /// See [`docs/network.md`](../../docs/network.md#networkbuilder) for reproducibility details.
    pub fn build_with_seed(self, seed: u64) -> FannResult<Network> {
        use rand::SeedableRng;

        let input_size = self
            .input_size
            .ok_or_else(|| FannError::InvalidBuilder("Input size not specified".into()))?;

        if self.layers.is_empty() {
            return Err(FannError::InvalidBuilder("No layers specified".into()));
        }

        if input_size == 0 {
            return Err(FannError::InvalidBuilder("Input size cannot be 0".into()));
        }

        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let mut layers = Vec::with_capacity(self.layers.len());
        let mut prev_size = input_size;

        for (i, (size, activation)) in self.layers.into_iter().enumerate() {
            if size == 0 {
                return Err(FannError::InvalidBuilder(format!("Layer {i} has size 0")));
            }
            let layer = Layer::new_with_rng(prev_size, size, activation, &mut rng)?;
            layers.push(layer);
            prev_size = size;
        }

        Network::new(layers)
    }
}

#[cfg(test)]
mod tests {
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
}
