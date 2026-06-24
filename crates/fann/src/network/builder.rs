//! Network builder for fluent API construction
//!
//! Provides `NetworkBuilder` for constructing neural networks with a fluent API.

use crate::activation::Activation;
use crate::error::{FannError, FannResult};
use crate::layer::Layer;
use crate::network::Network;

/// Builder for constructing neural networks with a fluent API
///
/// # Example
///
/// ```
/// use lattice_fann::{NetworkBuilder, Activation};
///
/// let network = NetworkBuilder::new()
///     .input(784)                          // MNIST input
///     .hidden(128, Activation::ReLU)       // Hidden layer 1
///     .hidden(64, Activation::ReLU)        // Hidden layer 2
///     .output(10, Activation::Softmax)     // 10 classes
///     .build()
///     .unwrap();
///
/// assert_eq!(network.num_inputs(), 784);
/// assert_eq!(network.num_outputs(), 10);
/// ```
#[derive(Debug, Clone, Default)]
pub struct NetworkBuilder {
    input_size: Option<usize>,
    layers: Vec<(usize, Activation)>,
}

/// Reject Softmax on any hidden (non-final) layer.
///
/// Softmax normalises over the simplex — meaningful only at the output.
/// On hidden layers the gradient degrades to the diagonal approximation
/// `s_i*(1-s_i)`, producing silently wrong gradients (tracked as FP-095).
fn validate_no_hidden_softmax(layers: &[(usize, Activation)]) -> FannResult<()> {
    if layers.len() < 2 {
        return Ok(());
    }
    let last_hidden = layers.len() - 1;
    for (_, activation) in &layers[..last_hidden] {
        if activation.is_softmax() {
            return Err(FannError::InvalidBuilder(
                "Softmax is only valid as the output-layer activation; \
                 hidden layers must use a pointwise activation"
                    .into(),
            ));
        }
    }
    Ok(())
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

    /// Build the network
    ///
    /// Returns an error if:
    /// - Input size was not specified
    /// - No layers were added
    /// - Any layer has size 0
    /// - Softmax is used on a hidden (non-output) layer
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

        validate_no_hidden_softmax(&self.layers)?;

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

    /// Build with specific seed for reproducible initialization
    ///
    /// Uses a seeded RNG for deterministic weight initialization,
    /// ensuring the same seed always produces the same network.
    ///
    /// # Example
    ///
    /// ```
    /// use lattice_fann::{NetworkBuilder, Activation};
    ///
    /// let network1 = NetworkBuilder::new()
    ///     .input(4)
    ///     .hidden(8, Activation::ReLU)
    ///     .output(2, Activation::Softmax)
    ///     .build_with_seed(42)
    ///     .unwrap();
    ///
    /// let network2 = NetworkBuilder::new()
    ///     .input(4)
    ///     .hidden(8, Activation::ReLU)
    ///     .output(2, Activation::Softmax)
    ///     .build_with_seed(42)
    ///     .unwrap();
    ///
    /// // Same seed produces identical weights
    /// assert_eq!(network1.layer(0).unwrap().weights(), network2.layer(0).unwrap().weights());
    /// ```
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

        validate_no_hidden_softmax(&self.layers)?;

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
