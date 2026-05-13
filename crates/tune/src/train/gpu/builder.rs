//! GPU trainer builder

use super::GpuTrainer;
use crate::error::{Result, TuneError};
use crate::train::config::TrainingConfig;
use lattice_fann::{Activation, NetworkBuilder};

/// Builder for GpuTrainer
pub struct GpuTrainerBuilder {
    input_size: usize,
    hidden_layers: Vec<(usize, Activation)>,
    output_size: usize,
    output_activation: Activation,
    config: TrainingConfig,
}

impl GpuTrainerBuilder {
    /// Create a new builder
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_layers: Vec::new(),
            output_size,
            output_activation: Activation::Softmax,
            config: TrainingConfig::default(),
        }
    }

    /// Add a hidden layer
    pub fn hidden(mut self, size: usize, activation: Activation) -> Self {
        self.hidden_layers.push((size, activation));
        self
    }

    /// Set output activation
    pub fn output_activation(mut self, activation: Activation) -> Self {
        self.output_activation = activation;
        self
    }

    /// Set training configuration
    pub fn config(mut self, config: TrainingConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the trainer
    pub fn build(self) -> Result<GpuTrainer> {
        let mut builder = NetworkBuilder::new().input(self.input_size);

        for (size, activation) in self.hidden_layers {
            builder = builder.hidden(size, activation);
        }

        let network = builder
            .output(self.output_size, self.output_activation)
            .build()
            .map_err(|e| TuneError::Training(format!("Network build failed: {e}")))?;

        GpuTrainer::new(network, self.config)
    }
}
