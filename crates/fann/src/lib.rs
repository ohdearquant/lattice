#![warn(clippy::all)]

//! lattice-fann: Fast neural network primitives for Lattice
//!
//! This crate provides lightweight neural network building blocks optimized
//! for fast CPU inference (<5ms). Designed for tiny models that need to run
//! quickly without GPU acceleration.
//!
//! # Features
//!
//! - **Fast inference**: Pre-allocated buffers, no allocations during forward pass
//! - **Fluent API**: `NetworkBuilder` for easy network construction
//! - **Common activations**: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU
//! - **Training support**: Basic backpropagation with momentum
//! - **Optional parallelism**: Feature-gated batch inference
//! - **Serialization**: Optional serde support
//!
//! # Quick Start
//!
//! ```
//! use lattice_fann::{Network, NetworkBuilder, Activation};
//!
//! // Build a simple classifier: 4 inputs -> 8 hidden -> 3 outputs
//! let mut network = NetworkBuilder::new()
//!     .input(4)
//!     .hidden(8, Activation::ReLU)
//!     .output(3, Activation::Softmax)
//!     .build()
//!     .unwrap();
//!
//! // Run inference
//! let input = [1.0, 2.0, 3.0, 4.0];
//! let output = network.forward(&input).unwrap();
//!
//! // Output is a probability distribution (sums to 1.0)
//! assert_eq!(output.len(), 3);
//! let sum: f32 = output.iter().sum();
//! assert!((sum - 1.0).abs() < 1e-5);
//! ```
//!
//! # Architecture
//!
//! ```text
//! NetworkBuilder --> Network --> [Layer, Layer, ...] --> output
//!                      |
//!                      +-- pre-allocated buffers (no alloc during inference)
//! ```
//!
//! # Training
//!
//! ```
//! use lattice_fann::{NetworkBuilder, Activation, BackpropTrainer, TrainingConfig, Trainer};
//!
//! let mut network = NetworkBuilder::new()
//!     .input(2)
//!     .hidden(4, Activation::Tanh)
//!     .output(1, Activation::Tanh)
//!     .build()
//!     .unwrap();
//!
//! // XOR training data
//! let inputs = vec![
//!     vec![0.0, 0.0],
//!     vec![0.0, 1.0],
//!     vec![1.0, 0.0],
//!     vec![1.0, 1.0],
//! ];
//! let targets = vec![
//!     vec![0.0],
//!     vec![1.0],
//!     vec![1.0],
//!     vec![0.0],
//! ];
//!
//! let mut trainer = BackpropTrainer::new();
//! let config = TrainingConfig::new()
//!     .learning_rate(0.5)
//!     .max_epochs(1000);
//!
//! let result = trainer.train(&mut network, &inputs, &targets, &config);
//! ```
//!
//! # Feature Flags
//!
//! - `std` (default): Enable standard library support
//! - `simd` (default): Enable SIMD optimizations for matrix operations
//! - `parallel`: Enable parallel batch inference via rayon
//! - `serde`: Enable serialization/deserialization support
//! - `gpu`: Enable GPU acceleration via wgpu (Metal/Vulkan/DX12)

#![warn(missing_docs)]

mod activation;
mod error;
mod layer;
mod network;
pub mod training;

#[cfg(feature = "gpu")]
pub mod gpu;

// Re-exports
pub use activation::Activation;
pub use error::{FannError, FannResult};
pub use layer::Layer;
pub use network::{Network, NetworkBuilder};
pub use training::{
    BackpropTrainer, GradientGuardStrategy, Trainer, TrainingConfig, TrainingResult,
};

#[cfg(feature = "gpu")]
pub use gpu::{GpuContext, GpuNetwork};

/// Prelude for common imports
pub mod prelude {
    pub use crate::{
        Activation, BackpropTrainer, FannError, FannResult, GradientGuardStrategy, Layer, Network,
        NetworkBuilder, Trainer, TrainingConfig, TrainingResult,
    };

    #[cfg(feature = "gpu")]
    pub use crate::gpu::{GpuContext, GpuNetwork};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_workflow() {
        // Build network
        let mut network = NetworkBuilder::new()
            .input(4)
            .hidden(8, Activation::ReLU)
            .hidden(4, Activation::ReLU)
            .output(2, Activation::Softmax)
            .build()
            .unwrap();

        // Verify architecture
        assert_eq!(network.num_inputs(), 4);
        assert_eq!(network.num_outputs(), 2);
        assert_eq!(network.num_layers(), 3);

        // Run inference
        let input = [1.0, 2.0, 3.0, 4.0];
        let output = network.forward(&input).unwrap();

        // Softmax output sums to 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_inference_speed() {
        use std::time::Instant;

        // Create a reasonably sized network
        let mut network = NetworkBuilder::new()
            .input(128)
            .hidden(256, Activation::ReLU)
            .hidden(128, Activation::ReLU)
            .hidden(64, Activation::ReLU)
            .output(10, Activation::Softmax)
            .build()
            .unwrap();

        let input: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();

        // Warm up
        for _ in 0..10 {
            network.forward(&input).unwrap();
        }

        // Benchmark
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            network.forward(&input).unwrap();
        }
        let elapsed = start.elapsed();

        let avg_us = elapsed.as_micros() as f64 / iterations as f64;
        println!("Average inference time: {avg_us:.2} us");

        // Should be well under 5ms (5000us)
        assert!(avg_us < 5000.0, "Inference too slow: {avg_us} us");
    }

    #[test]
    fn test_deterministic_output() {
        let mut network = NetworkBuilder::new()
            .input(4)
            .output(2, Activation::Linear)
            .build()
            .unwrap();

        let input = [1.0, 2.0, 3.0, 4.0];

        // Run multiple times
        let output1 = network.forward(&input).unwrap().to_vec();
        let output2 = network.forward(&input).unwrap().to_vec();

        // Should be deterministic
        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_parameter_count() {
        let network = NetworkBuilder::new()
            .input(784) // MNIST input
            .hidden(128, Activation::ReLU) // 784*128 + 128 = 100480
            .hidden(64, Activation::ReLU) // 128*64 + 64 = 8256
            .output(10, Activation::Softmax) // 64*10 + 10 = 650
            .build()
            .unwrap();

        assert_eq!(network.total_params(), 100480 + 8256 + 650);
    }
}
