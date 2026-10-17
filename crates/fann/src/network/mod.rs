//! Feedforward-network execution and construction.
//!
//! `Network` validates layer connectivity, pre-allocates one activation buffer
//! per layer, and rejects non-finite layer outputs. `NetworkBuilder` constructs
//! compatible dense stacks.
//!
//! See `docs/network.md` for the execution model and binary format.

mod builder;
mod serialization;

pub use builder::NetworkBuilder;

use crate::error::{FannError, FannResult};
use crate::layer::Layer;

/// Reject non-finite layer output in every build; format an error only on failure.
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

/// Shared forward pass using caller-provided, per-layer activation buffers.
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

/// Feedforward network with reusable, per-layer activation buffers.
///
/// `forward` allocates no activation buffers; its output borrow remains valid
/// until the next mutable use of the network.
/// See [`docs/network.md`](../../docs/network.md#network) for execution and usage details.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "NetworkData"))]
pub struct Network {
    /// Network layers
    layers: Vec<Layer>,
    /// Pre-allocated buffers for intermediate activations
    /// `buffers[i]` holds the output of layer `i`.
    #[cfg_attr(feature = "serde", serde(skip))]
    buffers: Vec<Vec<f32>>,
}

/// Serde input that rebuilds skipped activation buffers through [`Network::new`].
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct NetworkData {
    layers: Vec<Layer>,
}

#[cfg(feature = "serde")]
impl TryFrom<NetworkData> for Network {
    type Error = FannError;

    fn try_from(data: NetworkData) -> Result<Self, Self::Error> {
        Network::new(data.layers)
    }
}

impl Network {
    /// Create a network from a list of layers
    ///
    /// Validates that layer dimensions are compatible and Softmax appears only
    /// on the output layer.
    pub fn new(layers: Vec<Layer>) -> FannResult<Self> {
        if layers.is_empty() {
            return Err(FannError::EmptyNetwork);
        }

        // Reject Softmax before the final layer: its derivative here is only diagonal (see ADR-023).
        if layers[..layers.len() - 1]
            .iter()
            .any(|layer| layer.activation().is_softmax())
        {
            return Err(FannError::InvalidBuilder(
                "Softmax is only valid as the output-layer activation; \
                 hidden layers must use a pointwise activation"
                    .into(),
            ));
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

    /// Runs `input` through all layers and borrows the final activation buffer.
    ///
    /// Returns an error when the input width is invalid or a layer output is non-finite.
    /// See [`docs/network.md`](../../docs/network.md#network) for buffer lifetime details.
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

    /// Runs the CPU forward pass behind the async GPU-compatible interface.
    ///
    /// Returns an owned copy of the output and propagates `forward` validation errors.
    /// See [`docs/network.md`](../../docs/network.md#networkforward_async) for interface rationale.
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
    /// Runs a forward pass for each input in parallel.
    ///
    /// Shares layer parameters and allocates independent activation buffers per input.
    /// See [`docs/network.md`](../../docs/network.md#networkforward_batch) for concurrency details.
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
mod tests;

#[cfg(all(test, feature = "serde"))]
mod serde_validation_tests;
