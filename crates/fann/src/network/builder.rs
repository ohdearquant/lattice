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
    /// This must be called before `build` or `build_with_seed`; call order is otherwise
    /// unconstrained.
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
mod tests;
