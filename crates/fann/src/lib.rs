//! `lattice-fann` provides small dense neural networks for CPU-first inference.
//!
//! Use [`NetworkBuilder`] to assemble layers, [`Network`] to run them, and
//! [`BackpropTrainer`] to train them. The CPU forward path reuses activation
//! buffers; `parallel`, `serde`, and `gpu` add batch inference, persistence, and
//! optional GPU acceleration. `simd` accelerates supported CPU kernels.
//!
//! See `docs/design.md` for the crate architecture and `docs/network.md` for
//! the network, activation, and binary-format reference.

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
