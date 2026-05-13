//! Neural network implementation
//!
//! Provides the `Network` struct for fast inference and `NetworkBuilder`
//! for fluent network construction.

mod builder;
mod serialization;

pub use builder::NetworkBuilder;

use crate::error::{FannError, FannResult};
use crate::layer::Layer;

/// Check for NaN or Inf values in a layer's output buffer.
///
/// Takes a layer index instead of a string to avoid format!() allocation
/// on every call in the forward pass hot path. The string is only
/// formatted in the error branch.
///
/// This check is always active, including release builds, so inference fails
/// closed instead of returning non-finite activations.
#[inline]
fn check_numeric_stability(values: &[f32], layer_index: usize) -> FannResult<()> {
    for (i, &v) in values.iter().enumerate() {
        if v.is_nan() {
            return Err(FannError::NumericInstability(format!(
                "NaN detected at index {i} in layer {layer_index} output"
            )));
        }
        if v.is_infinite() {
            return Err(FannError::NumericInstability(format!(
                "Inf detected at index {i} in layer {layer_index} output"
            )));
        }
    }
    Ok(())
}

/// Run a forward pass through layers using provided buffers.
///
/// This is the core forward-pass logic, extracted so that `forward_batch` can
/// share the (read-only) layer weights across threads without cloning the
/// entire network — only the mutable activation buffers are per-thread.
#[inline]
fn forward_into_buffers(
    layers: &[Layer],
    input: &[f32],
    buffers: &mut [Vec<f32>],
) -> FannResult<()> {
    // First layer takes input directly
    layers[0].forward(input, &mut buffers[0])?;
    check_numeric_stability(&buffers[0], 0)?;

    // Subsequent layers take previous layer's output
    for i in 1..layers.len() {
        let (prev, curr) = buffers.split_at_mut(i);
        layers[i].forward(&prev[i - 1], &mut curr[0])?;
        check_numeric_stability(&curr[0], i)?;
    }

    Ok(())
}

/// A feedforward neural network optimized for fast inference
///
/// The network pre-allocates all intermediate buffers at construction time
/// to avoid allocations during inference.
///
/// # Example
///
/// ```
/// use lattice_fann::{Network, NetworkBuilder, Activation};
///
/// // Build a simple network: 4 inputs -> 8 hidden (ReLU) -> 2 outputs (Softmax)
/// let mut network = NetworkBuilder::new()
///     .input(4)
///     .hidden(8, Activation::ReLU)
///     .output(2, Activation::Softmax)
///     .build()
///     .unwrap();
///
/// // Run inference
/// let input = [1.0, 2.0, 3.0, 4.0];
/// let output = network.forward(&input).unwrap();
/// assert_eq!(output.len(), 2);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Network {
    /// Network layers
    layers: Vec<Layer>,
    /// Pre-allocated buffers for intermediate activations
    /// buffers[i] holds output of layer i (or input for i=0)
    #[cfg_attr(feature = "serde", serde(skip))]
    buffers: Vec<Vec<f32>>,
}

impl Network {
    /// Create a network from a list of layers
    ///
    /// Validates that layer dimensions are compatible.
    pub fn new(layers: Vec<Layer>) -> FannResult<Self> {
        if layers.is_empty() {
            return Err(FannError::EmptyNetwork);
        }

        // Validate layer compatibility
        for i in 1..layers.len() {
            let prev_outputs = layers[i - 1].num_outputs();
            let curr_inputs = layers[i].num_inputs();
            if prev_outputs != curr_inputs {
                return Err(FannError::InvalidLayerDimensions {
                    inputs: curr_inputs,
                    outputs: prev_outputs,
                });
            }
        }

        // Allocate buffers for intermediate activations
        let buffers = layers
            .iter()
            .map(|layer| vec![0.0; layer.num_outputs()])
            .collect();

        Ok(Self { layers, buffers })
    }

    /// Run forward pass through the network
    ///
    /// Returns a reference to the output buffer (valid until next forward call).
    ///
    /// # Arguments
    /// * `input` - Input vector (must match network's input size)
    #[inline]
    pub fn forward(&mut self, input: &[f32]) -> FannResult<&[f32]> {
        let expected_inputs = self.num_inputs();
        if input.len() != expected_inputs {
            return Err(FannError::InputSizeMismatch {
                expected: expected_inputs,
                actual: input.len(),
            });
        }

        forward_into_buffers(&self.layers, input, &mut self.buffers)?;
        Ok(&self.buffers[self.buffers.len() - 1])
    }

    /// Async forward pass for API consistency with GpuNetwork
    ///
    /// This provides a unified async interface for CPU/GPU switching.
    /// Returns an owned Vec instead of a reference.
    ///
    /// # Arguments
    /// * `input` - Input vector (must match network's input size)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In an async context:
    /// use lattice_fann::{Network, NetworkBuilder, Activation};
    ///
    /// let mut network = NetworkBuilder::new()
    ///     .input(4)
    ///     .output(2, Activation::Softmax)
    ///     .build()
    ///     .unwrap();
    ///
    /// let input = vec![1.0, 2.0, 3.0, 4.0];
    /// let output = network.forward_async(&input).await.unwrap();
    /// assert_eq!(output.len(), 2);
    /// ```
    pub async fn forward_async(&mut self, input: &[f32]) -> FannResult<Vec<f32>> {
        // CPU forward is synchronous, just wrap in async for API consistency
        self.forward(input).map(<[f32]>::to_vec)
    }

    /// Get the number of inputs the network expects
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.layers[0].num_inputs()
    }

    /// Get the number of outputs the network produces
    #[inline]
    pub fn num_outputs(&self) -> usize {
        self.layers[self.layers.len() - 1].num_outputs()
    }

    /// Get the total number of parameters in the network
    #[inline]
    pub fn total_params(&self) -> usize {
        self.layers
            .iter()
            .map(super::layer::Layer::num_params)
            .sum()
    }

    /// Get the number of layers in the network
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get a reference to a specific layer
    #[inline]
    pub fn layer(&self, index: usize) -> Option<&Layer> {
        self.layers.get(index)
    }

    /// Get a mutable reference to a specific layer (for training)
    #[inline]
    pub fn layer_mut(&mut self, index: usize) -> Option<&mut Layer> {
        self.layers.get_mut(index)
    }

    /// Get all layers
    #[inline]
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Get all layers mutably (for training)
    #[inline]
    pub fn layers_mut(&mut self) -> &mut [Layer] {
        &mut self.layers
    }

    /// Get intermediate activations buffer for a layer (for debugging/training)
    #[inline]
    pub fn activations(&self, layer_index: usize) -> Option<&[f32]> {
        self.buffers.get(layer_index).map(std::vec::Vec::as_slice)
    }

    /// Get network architecture as a string
    pub fn architecture(&self) -> String {
        let mut parts = vec![self.num_inputs().to_string()];
        for layer in &self.layers {
            parts.push(format!("{:?}({})", layer.activation(), layer.num_outputs()));
        }
        parts.join(" -> ")
    }
}

#[cfg(feature = "parallel")]
impl Network {
    /// Run forward pass on multiple inputs in parallel
    ///
    /// Shares the (read-only) network weights across threads. Only the
    /// intermediate activation buffers are allocated per input, avoiding
    /// the cost of cloning all weight matrices.
    pub fn forward_batch(&self, inputs: &[Vec<f32>]) -> FannResult<Vec<Vec<f32>>> {
        use rayon::prelude::*;

        inputs
            .par_iter()
            .map(|input| {
                // Allocate only activation buffers — weights are shared via &self
                let mut buffers: Vec<Vec<f32>> = self
                    .layers
                    .iter()
                    .map(|layer| vec![0.0; layer.num_outputs()])
                    .collect();
                forward_into_buffers(&self.layers, input, &mut buffers)?;
                buffers.pop().ok_or(FannError::EmptyNetwork)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
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
}
