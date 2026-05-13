//! Network serialization and deserialization
//!
//! Provides binary serialization for efficient network storage and loading.

use crate::activation::Activation;
use crate::error::{FannError, FannResult};
use crate::layer::Layer;
use crate::network::Network;

impl Network {
    /// Serialize the network to a compact binary format
    ///
    /// Format:
    /// - Magic: 4 bytes "FANN"
    /// - Version: u32 little-endian (1)
    /// - Num layers: u32 little-endian
    /// - For each layer:
    ///   - num_inputs: u32 little-endian
    ///   - num_outputs: u32 little-endian
    ///   - activation_type: u8 (0=Linear, 1=Sigmoid, 2=Tanh, 3=ReLU, 4=LeakyReLU, 5=Softmax)
    ///   - If LeakyReLU: alpha f32 little-endian
    ///   - weights: num_inputs * num_outputs f32s little-endian
    ///   - biases: num_outputs f32s little-endian
    ///
    /// # Example
    ///
    /// ```
    /// use lattice_fann::{Network, NetworkBuilder, Activation};
    ///
    /// let network = NetworkBuilder::new()
    ///     .input(4)
    ///     .hidden(8, Activation::ReLU)
    ///     .output(2, Activation::Softmax)
    ///     .build()
    ///     .unwrap();
    ///
    /// let bytes = network.to_bytes();
    /// let restored = Network::from_bytes(&bytes).unwrap();
    ///
    /// assert_eq!(network.num_inputs(), restored.num_inputs());
    /// assert_eq!(network.num_outputs(), restored.num_outputs());
    /// assert_eq!(network.num_layers(), restored.num_layers());
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Magic number "FANN"
        bytes.extend_from_slice(b"FANN");

        // Version (1)
        bytes.extend_from_slice(&1u32.to_le_bytes());

        // Number of layers
        bytes.extend_from_slice(&(self.layers().len() as u32).to_le_bytes());

        // Each layer
        for layer in self.layers() {
            // Dimensions
            bytes.extend_from_slice(&(layer.num_inputs() as u32).to_le_bytes());
            bytes.extend_from_slice(&(layer.num_outputs() as u32).to_le_bytes());

            // Activation type
            match layer.activation() {
                Activation::Linear => bytes.push(0),
                Activation::Sigmoid => bytes.push(1),
                Activation::Tanh => bytes.push(2),
                Activation::ReLU => bytes.push(3),
                Activation::LeakyReLU(alpha) => {
                    bytes.push(4);
                    bytes.extend_from_slice(&alpha.to_le_bytes());
                }
                Activation::Softmax => bytes.push(5),
            }

            // Weights
            for &w in layer.weights() {
                bytes.extend_from_slice(&w.to_le_bytes());
            }

            // Biases
            for &b in layer.biases() {
                bytes.extend_from_slice(&b.to_le_bytes());
            }
        }

        bytes
    }

    /// Deserialize a network from binary format
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Magic number is invalid
    /// - Version is unsupported
    /// - Data is truncated or malformed
    /// - Layer dimensions are invalid
    ///
    /// # Example
    ///
    /// ```
    /// use lattice_fann::{Network, NetworkBuilder, Activation};
    ///
    /// let original = NetworkBuilder::new()
    ///     .input(10)
    ///     .hidden(20, Activation::ReLU)
    ///     .output(5, Activation::Softmax)
    ///     .build()
    ///     .unwrap();
    ///
    /// let bytes = original.to_bytes();
    /// let restored = Network::from_bytes(&bytes).unwrap();
    ///
    /// assert_eq!(original.total_params(), restored.total_params());
    /// ```
    pub fn from_bytes(bytes: &[u8]) -> FannResult<Self> {
        let mut pos = 0;

        // Helper to read bytes
        let read_bytes = |pos: &mut usize, n: usize| -> FannResult<&[u8]> {
            if *pos + n > bytes.len() {
                return Err(FannError::InvalidBuilder(format!(
                    "Truncated data at offset {}: expected {} bytes, {} available",
                    *pos,
                    n,
                    bytes.len() - *pos
                )));
            }
            let slice = &bytes[*pos..*pos + n];
            *pos += n;
            Ok(slice)
        };

        // Read magic number
        let magic = read_bytes(&mut pos, 4)?;
        if magic != b"FANN" {
            return Err(FannError::InvalidBuilder(format!(
                "Invalid magic number: expected FANN, got {magic:?}"
            )));
        }

        // Read version
        let version_bytes: [u8; 4] = read_bytes(&mut pos, 4)?
            .try_into()
            .map_err(|_| FannError::InvalidBuilder("Failed to read version".into()))?;
        let version = u32::from_le_bytes(version_bytes);
        if version != 1 {
            return Err(FannError::InvalidBuilder(format!(
                "Unsupported version: {version}, expected 1"
            )));
        }

        // Read number of layers
        let num_layers_bytes: [u8; 4] = read_bytes(&mut pos, 4)?
            .try_into()
            .map_err(|_| FannError::InvalidBuilder("Failed to read num_layers".into()))?;
        let num_layers = u32::from_le_bytes(num_layers_bytes) as usize;

        if num_layers == 0 {
            return Err(FannError::EmptyNetwork);
        }

        let mut layers = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            // Read dimensions
            let num_inputs_bytes: [u8; 4] = read_bytes(&mut pos, 4)?
                .try_into()
                .map_err(|_| FannError::InvalidBuilder("Failed to read num_inputs".into()))?;
            let num_inputs = u32::from_le_bytes(num_inputs_bytes) as usize;

            let num_outputs_bytes: [u8; 4] = read_bytes(&mut pos, 4)?
                .try_into()
                .map_err(|_| FannError::InvalidBuilder("Failed to read num_outputs".into()))?;
            let num_outputs = u32::from_le_bytes(num_outputs_bytes) as usize;

            // Read activation type
            let activation_type = read_bytes(&mut pos, 1)?[0];
            let activation = match activation_type {
                0 => Activation::Linear,
                1 => Activation::Sigmoid,
                2 => Activation::Tanh,
                3 => Activation::ReLU,
                4 => {
                    let alpha_bytes: [u8; 4] =
                        read_bytes(&mut pos, 4)?.try_into().map_err(|_| {
                            FannError::InvalidBuilder("Failed to read LeakyReLU alpha".into())
                        })?;
                    Activation::LeakyReLU(f32::from_le_bytes(alpha_bytes))
                }
                5 => Activation::Softmax,
                _ => {
                    return Err(FannError::InvalidBuilder(format!(
                        "Unknown activation type {activation_type} at layer {layer_idx}"
                    )));
                }
            };

            // Read weights
            let weight_count = num_inputs * num_outputs;
            let weight_bytes = read_bytes(&mut pos, weight_count * 4)?;
            let weights: Vec<f32> = weight_bytes
                .chunks_exact(4)
                .map(|chunk| {
                    // SAFETY: chunks_exact(4) guarantees exactly 4 bytes per chunk
                    let arr: [u8; 4] = chunk
                        .try_into()
                        .expect("chunks_exact(4) guarantees 4 bytes");
                    f32::from_le_bytes(arr)
                })
                .collect();

            // Read biases
            let bias_bytes = read_bytes(&mut pos, num_outputs * 4)?;
            let biases: Vec<f32> = bias_bytes
                .chunks_exact(4)
                .map(|chunk| {
                    // SAFETY: chunks_exact(4) guarantees exactly 4 bytes per chunk
                    let arr: [u8; 4] = chunk
                        .try_into()
                        .expect("chunks_exact(4) guarantees 4 bytes");
                    f32::from_le_bytes(arr)
                })
                .collect();

            let layer = Layer::with_weights(num_inputs, num_outputs, weights, biases, activation)?;
            layers.push(layer);
        }

        Network::new(layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::NetworkBuilder;

    #[test]
    fn test_to_bytes_from_bytes_roundtrip() {
        let original = NetworkBuilder::new()
            .input(4)
            .hidden(8, Activation::ReLU)
            .hidden(4, Activation::Sigmoid)
            .output(2, Activation::Softmax)
            .build()
            .unwrap();

        let bytes = original.to_bytes();
        let restored = Network::from_bytes(&bytes).unwrap();

        assert_eq!(original.num_inputs(), restored.num_inputs());
        assert_eq!(original.num_outputs(), restored.num_outputs());
        assert_eq!(original.num_layers(), restored.num_layers());
        assert_eq!(original.total_params(), restored.total_params());

        // Verify weights match
        for i in 0..original.num_layers() {
            let orig_layer = original.layer(i).unwrap();
            let rest_layer = restored.layer(i).unwrap();

            assert_eq!(orig_layer.num_inputs(), rest_layer.num_inputs());
            assert_eq!(orig_layer.num_outputs(), rest_layer.num_outputs());
            assert_eq!(orig_layer.activation(), rest_layer.activation());
            assert_eq!(orig_layer.weights(), rest_layer.weights());
            assert_eq!(orig_layer.biases(), rest_layer.biases());
        }
    }

    #[test]
    fn test_to_bytes_from_bytes_with_leaky_relu() {
        let alpha = 0.1f32;
        let original = NetworkBuilder::new()
            .input(10)
            .hidden(20, Activation::LeakyReLU(alpha))
            .output(5, Activation::Linear)
            .build()
            .unwrap();

        let bytes = original.to_bytes();
        let restored = Network::from_bytes(&bytes).unwrap();

        // Verify LeakyReLU alpha is preserved
        assert_eq!(
            original.layer(0).unwrap().activation(),
            Activation::LeakyReLU(alpha)
        );
        assert_eq!(
            restored.layer(0).unwrap().activation(),
            Activation::LeakyReLU(alpha)
        );
    }

    #[test]
    fn test_from_bytes_invalid_magic() {
        let bytes = b"NOTF\x01\x00\x00\x00\x00\x00\x00\x00";
        let result = Network::from_bytes(bytes);
        assert!(matches!(result, Err(FannError::InvalidBuilder(_))));
    }

    #[test]
    fn test_from_bytes_truncated() {
        let bytes = b"FANN\x01\x00\x00\x00"; // Missing num_layers
        let result = Network::from_bytes(bytes);
        assert!(matches!(result, Err(FannError::InvalidBuilder(_))));
    }

    #[test]
    fn test_from_bytes_empty_network() {
        // Valid header with 0 layers
        let bytes = b"FANN\x01\x00\x00\x00\x00\x00\x00\x00";
        let result = Network::from_bytes(bytes);
        assert!(matches!(result, Err(FannError::EmptyNetwork)));
    }

    #[test]
    fn test_serialization_forward_consistency() {
        // Build with seed for reproducibility
        let mut original = NetworkBuilder::new()
            .input(4)
            .hidden(8, Activation::ReLU)
            .output(2, Activation::Softmax)
            .build_with_seed(42)
            .unwrap();

        let input = [1.0, 2.0, 3.0, 4.0];
        let original_output = original.forward(&input).unwrap().to_vec();

        // Serialize and deserialize
        let bytes = original.to_bytes();
        let mut restored = Network::from_bytes(&bytes).unwrap();
        let restored_output = restored.forward(&input).unwrap().to_vec();

        // Outputs should be identical
        for (orig, rest) in original_output.iter().zip(restored_output.iter()) {
            assert!((orig - rest).abs() < 1e-6);
        }
    }
}
