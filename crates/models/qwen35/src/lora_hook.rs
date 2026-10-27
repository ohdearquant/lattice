//! LoRA projection geometry of a Qwen3.5 model configuration.

use lattice_arch::qwen35_config::Qwen35Config;
use lattice_compute::lora_hook::{LoraProjectionShape, LoraTargetGeometry};

/// Return the LoRA projection shape for a Qwen3.5 layer and module.
///
/// Full-attention modules are valid only on full-attention layers, Gated
/// DeltaNet modules are valid only on linear-attention layers, and MLP modules
/// are valid on both. The returned geometry includes `in_proj_a` and
/// `in_proj_b`; a backend that cannot apply those projections must enforce that
/// capability restriction separately.
pub fn qwen35_projection_shape(
    config: &Qwen35Config,
    layer_idx: usize,
    module: &str,
) -> Result<LoraProjectionShape, String> {
    if layer_idx >= config.num_hidden_layers {
        return Err(format!(
            "layer {layer_idx} is out of range for Qwen3.5 model with {} layers",
            config.num_hidden_layers
        ));
    }

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let is_full = config.is_full_attention(layer_idx);
    let (d_in, d_out) = match (module, is_full) {
        ("q_proj", true) => (hidden, 2 * config.full_q_dim()),
        ("k_proj", true) => (hidden, config.full_kv_dim()),
        ("v_proj", true) => (hidden, config.full_kv_dim()),
        ("o_proj", true) => (config.full_q_dim(), hidden),
        ("in_proj_qkv", false) => (hidden, config.linear_qkv_dim()),
        ("in_proj_z", false) => (hidden, config.linear_output_dim()),
        ("in_proj_b" | "in_proj_a", false) => (hidden, config.linear_num_value_heads()),
        ("out_proj", false) => (config.linear_output_dim(), hidden),
        ("gate_proj", _) | ("up_proj", _) => (hidden, intermediate),
        ("down_proj", _) => (intermediate, hidden),
        ("q_proj" | "k_proj" | "v_proj" | "o_proj", false) => {
            return Err(format!(
                "module '{module}' is a full-attention projection but layer {layer_idx} is GDN"
            ));
        }
        ("in_proj_qkv" | "in_proj_z" | "in_proj_b" | "in_proj_a" | "out_proj", true) => {
            return Err(format!(
                "module '{module}' is a GDN projection but layer {layer_idx} is full-attention"
            ));
        }
        _ => return Err(format!("unknown LoRA module '{module}'")),
    };

    Ok(LoraProjectionShape { d_in, d_out })
}

/// **Unstable**: projection geometry borrowed from a Qwen3.5 model configuration.
#[derive(Clone, Copy, Debug)]
pub struct Qwen35LoraGeometry<'a> {
    config: &'a Qwen35Config,
}

impl<'a> Qwen35LoraGeometry<'a> {
    /// Borrow a model configuration for LoRA validation.
    pub fn new(config: &'a Qwen35Config) -> Self {
        Self { config }
    }
}

impl LoraTargetGeometry for Qwen35LoraGeometry<'_> {
    fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn projection_shape(
        &self,
        layer_idx: usize,
        module: &str,
    ) -> Result<LoraProjectionShape, String> {
        qwen35_projection_shape(self.config, layer_idx, module)
    }
}

#[cfg(test)]
mod tests;
