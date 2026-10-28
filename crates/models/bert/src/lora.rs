//! LoRA projection geometry for BERT encoders and cross-encoders.

use lattice_compute::lora_hook::{LoraProjectionShape, LoraTargetGeometry};

/// **Unstable**: projection geometry for a BERT encoder or cross-encoder.
#[derive(Clone, Copy, Debug)]
pub struct BertLoraGeometry {
    num_hidden_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
}

impl BertLoraGeometry {
    /// Construct LoRA geometry from the model's layer count and projection widths.
    pub fn new(num_hidden_layers: usize, hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            num_hidden_layers,
            hidden_size,
            intermediate_size,
        }
    }
}

impl LoraTargetGeometry for BertLoraGeometry {
    fn num_layers(&self) -> usize {
        self.num_hidden_layers
    }

    fn projection_shape(
        &self,
        layer_idx: usize,
        module: &str,
    ) -> Result<LoraProjectionShape, String> {
        if layer_idx >= self.num_hidden_layers {
            return Err(format!(
                "layer {layer_idx} is out of range for BERT model with {} layers (module: {module})",
                self.num_hidden_layers
            ));
        }
        let (d_in, d_out) = match module {
            "query" | "key" | "value" | "attn_output" => (self.hidden_size, self.hidden_size),
            "ffn_intermediate" => (self.hidden_size, self.intermediate_size),
            "ffn_output" => (self.intermediate_size, self.hidden_size),
            _ => {
                return Err(format!(
                    "unknown BERT LoRA module '{module}' on layer {layer_idx}"
                ));
            }
        };
        Ok(LoraProjectionShape { d_in, d_out })
    }
}

#[cfg(test)]
mod tests;
