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

        bytes.extend_from_slice(b"FANN");

        bytes.extend_from_slice(&1u32.to_le_bytes());

        bytes.extend_from_slice(&(self.layers().len() as u32).to_le_bytes());

        for layer in self.layers() {
            bytes.extend_from_slice(&(layer.num_inputs() as u32).to_le_bytes());
            bytes.extend_from_slice(&(layer.num_outputs() as u32).to_le_bytes());

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

            for &w in layer.weights() {
                bytes.extend_from_slice(&w.to_le_bytes());
            }

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
                    #[expect(
                        clippy::expect_used,
                        reason = "chunks_exact(4) yields exactly four bytes per chunk"
                    )]
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
                    #[expect(
                        clippy::expect_used,
                        reason = "chunks_exact(4) yields exactly four bytes per chunk"
                    )]
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
mod tests;
