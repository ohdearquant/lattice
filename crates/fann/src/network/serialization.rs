//! Compact binary network serialization.
//!
//! The format is little-endian, versioned, and contains each layer's shape,
//! activation, weights, and biases. Parsing validates bounds before allocation
//! and requires an exact-length blob.
//!
//! See `docs/network.md` for the complete on-disk format and invariants.

use crate::activation::Activation;
use crate::error::{FannError, FannResult, validate_allocation_size, validate_layer_dimensions};
use crate::layer::Layer;
use crate::network::Network;

impl Network {
    /// Serializes this network in the version-1 little-endian binary format.
    /// See [`docs/network.md`](../../docs/network.md#network-serialization) for the wire format.
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

    /// Deserializes an exact-length version-1 network blob.
    ///
    /// Returns an error for malformed headers, records, dimensions, or trailing data.
    /// See [`docs/network.md`](../../docs/network.md#network-serialization) for validation rules.
    pub fn from_bytes(bytes: &[u8]) -> FannResult<Self> {
        let mut pos = 0;

        // Read only after an overflow-safe remaining-byte check.
        let read_bytes = |pos: &mut usize, n: usize| -> FannResult<&[u8]> {
            // Subtraction avoids overflow with hostile byte counts — see docs/network.md.
            let available = bytes.len().saturating_sub(*pos);
            if n > available {
                return Err(FannError::InvalidBuilder(format!(
                    "Truncated data at offset {}: expected {n} bytes, {available} available",
                    *pos,
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

        // Every layer needs 9 header bytes; bound counts before allocation — see docs/network.md.
        let remaining_for_layers = bytes.len().saturating_sub(pos);
        let max_plausible_layers = remaining_for_layers / 9;
        if num_layers > max_plausible_layers {
            return Err(FannError::InvalidBuilder(format!(
                "num_layers {num_layers} exceeds what the remaining {remaining_for_layers} \
                 bytes can encode (max plausible with 9 bytes/layer: {max_plausible_layers})"
            )));
        }
        // Apply the general allocation guard as defence in depth.
        validate_allocation_size(num_layers)?;

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

            // Validate hostile dimensions before collecting payloads — see docs/network.md.
            validate_layer_dimensions(num_inputs, num_outputs)?;

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

            // Checked payload sizing rejects hostile dimensions without allocation — see docs/network.md.
            let weight_count = num_inputs.checked_mul(num_outputs).ok_or_else(|| {
                FannError::InvalidBuilder(format!(
                    "layer {layer_idx}: num_inputs ({num_inputs}) * num_outputs \
                     ({num_outputs}) overflows"
                ))
            })?;
            let weight_byte_count = weight_count.checked_mul(4).ok_or_else(|| {
                FannError::InvalidBuilder(format!(
                    "layer {layer_idx}: weight byte count overflows ({weight_count} * 4)"
                ))
            })?;
            let weight_bytes = read_bytes(&mut pos, weight_byte_count)?;
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

            // Bias payload sizing uses the same checked-arithmetic boundary.
            let bias_byte_count = num_outputs.checked_mul(4).ok_or_else(|| {
                FannError::InvalidBuilder(format!(
                    "layer {layer_idx}: bias byte count overflows ({num_outputs} * 4)"
                ))
            })?;
            let bias_bytes = read_bytes(&mut pos, bias_byte_count)?;
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

        // The format is exact-length; trailing bytes are malformed — see docs/network.md.
        if pos != bytes.len() {
            return Err(FannError::InvalidBuilder(format!(
                "trailing bytes after parsing: consumed {pos} of {} bytes; \
                 the format requires an exact-length blob",
                bytes.len()
            )));
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

    // ── FIX 2: bounds-before-allocation + trailing-bytes tests ──────────────

    /// A header with num_layers = 0xFFFFFFFF and a short buffer must return
    /// Err(FannError::InvalidBuilder) without attempting any large allocation.
    ///
    /// Mutation that defeats this test: remove the remaining-bytes bounds check
    /// before Vec::with_capacity(num_layers).
    #[test]
    fn from_bytes_large_num_layers_short_buffer_returns_err() {
        // Only 12 bytes: magic + version + num_layers=0xFFFFFFFF.
        // remaining_for_layers = 0, max_plausible = 0 < 0xFFFFFFFF → Err.
        let mut bytes = b"FANN".to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        let result = Network::from_bytes(&bytes);
        assert!(
            matches!(result, Err(FannError::InvalidBuilder(_))),
            "large num_layers with short buffer must return InvalidBuilder, got {result:?}"
        );
    }

    /// A valid serialized blob with one extra trailing byte must return
    /// Err(FannError::InvalidBuilder).
    ///
    /// Mutation that defeats this test: remove the trailing-bytes check after
    /// the layer loop.
    #[test]
    fn from_bytes_trailing_bytes_returns_err() {
        let net = NetworkBuilder::new()
            .input(2)
            .hidden(4, Activation::ReLU)
            .output(2, Activation::Linear)
            .build()
            .unwrap();
        let mut bytes = net.to_bytes();
        bytes.push(0xAB); // one trailing byte
        let result = Network::from_bytes(&bytes);
        assert!(
            matches!(result, Err(FannError::InvalidBuilder(_))),
            "trailing byte must return InvalidBuilder, got {result:?}"
        );
    }

    /// A header declaring a huge weight count (num_inputs * num_outputs far
    /// above MAX_ALLOWED_ELEMENTS) must be rejected by the early per-layer size
    /// cap with Err(FannError::ShapeTooLarge), before any byte-count arithmetic,
    /// read, or allocation.
    ///
    /// With num_inputs=3_000_000_000 and num_outputs=2_000_000_000 the element
    /// count is 6e18 — a valid usize, but ~6e10x the 100M cap. validate_layer_dimensions
    /// (run right after the dimensions are read) rejects it via validate_allocation_size.
    /// The downstream weight_count.checked_mul(4) guard remains as defence in depth
    /// but is unreachable for these dimensions because the cap fires first.
    ///
    /// Mutation that defeats this test: remove the early validate_layer_dimensions
    /// call — the parser then reaches weight_count.checked_mul(4), which overflows
    /// (6e18 * 4 = 2.4e19 > usize::MAX) and returns InvalidBuilder instead.
    #[test]
    fn from_bytes_huge_weight_dims_rejected_by_size_cap() {
        let mut bytes = b"FANN".to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes()); // version=1
        bytes.extend_from_slice(&1u32.to_le_bytes()); // num_layers=1
        bytes.extend_from_slice(&3_000_000_000_u32.to_le_bytes()); // num_inputs
        bytes.extend_from_slice(&2_000_000_000_u32.to_le_bytes()); // num_outputs
        bytes.push(0); // activation=Linear
        // 21 bytes total; the layer bounds check passes (remaining=9, max=1 layer).
        let result = Network::from_bytes(&bytes);
        assert!(
            matches!(result, Err(FannError::ShapeTooLarge { .. })),
            "huge weight dims must hit the size cap before allocation, got {result:?}"
        );
    }

    /// A header with near-usize::MAX-product dimensions must be rejected cleanly
    /// (Err, no panic) by the early per-layer size cap. This pins that
    /// validate_layer_dimensions itself does not overflow/panic on extreme
    /// dimensions: its internal checked_mul handles the 2^62-1 product, and
    /// validate_allocation_size then rejects it with ShapeTooLarge.
    ///
    /// num_inputs = 2^31-1, num_outputs = 2^31+1 → product = 2^62-1 (a valid
    /// usize, ~4.6e10x the 100M cap). The cap fires before the weight byte-count
    /// arithmetic and before read_bytes is asked for the (usize::MAX-3) payload,
    /// so the overflow-safe read_bytes guard remains as defence in depth here.
    ///
    /// Mutation that defeats this test: remove the early validate_layer_dimensions
    /// call — the parser then reaches read_bytes(pos=21, usize::MAX-3) and returns
    /// a truncation InvalidBuilder instead of ShapeTooLarge.
    #[test]
    fn from_bytes_near_usize_max_dims_rejected_by_size_cap() {
        let mut bytes = b"FANN".to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes()); // version=1
        bytes.extend_from_slice(&1u32.to_le_bytes()); // num_layers=1
        bytes.extend_from_slice(&2_147_483_647_u32.to_le_bytes()); // num_inputs = 2^31-1
        bytes.extend_from_slice(&2_147_483_649_u32.to_le_bytes()); // num_outputs = 2^31+1
        bytes.push(0); // activation=Linear
        let result = Network::from_bytes(&bytes);
        assert!(
            matches!(result, Err(FannError::ShapeTooLarge { .. })),
            "near-usize::MAX dims must hit the size cap before allocation, got {result:?}"
        );
    }

    #[test]
    fn from_bytes_oversized_layer_dims_rejected_before_allocation() {
        // A hostile header declaring more than MAX_ALLOWED_ELEMENTS weights must
        // be rejected by the size cap BEFORE from_bytes reads or collects the
        // weight payload. This buffer carries no weight bytes at all: if the cap
        // were enforced only inside Layer::with_weights (after collection), the
        // parser would first attempt the oversized weight read and return a
        // truncation InvalidBuilder. Reverting the early validate_layer_dimensions
        // call makes this case return InvalidBuilder instead of ShapeTooLarge.
        let mut bytes = b"FANN".to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes()); // version=1
        bytes.extend_from_slice(&1u32.to_le_bytes()); // num_layers=1
        bytes.extend_from_slice(&100_000_001_u32.to_le_bytes()); // num_inputs = MAX_ALLOWED_ELEMENTS + 1
        bytes.extend_from_slice(&1u32.to_le_bytes()); // num_outputs=1
        bytes.push(0); // activation=Linear
        let result = Network::from_bytes(&bytes);
        assert!(
            matches!(result, Err(FannError::ShapeTooLarge { .. })),
            "oversized layer dims must hit the size cap before allocation, got {result:?}"
        );
    }
}
