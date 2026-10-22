//! Qwen3.5 checkpoint loading into F16 weight containers.

use crate::error::InferenceError;
use crate::model::qwen35_config::Qwen35Config;
use crate::weights::SafetensorsFile;
use crate::weights::f16_weights::{
    F16AttentionWeights, F16CommonLayerWeights, F16FeedForwardWeights,
    F16FullAttentionLayerWeights, F16GatedDeltaNetWeights, F16ModelWeights, F16MoeLayerWeights,
    F16MoeRouter, F16RoutedExperts, F16SharedExpert,
};

#[inline]
fn checked_numel(shape: &[usize], name: &str) -> Result<usize, InferenceError> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            InferenceError::InvalidSafetensors(format!(
                "tensor {name} shape {shape:?} overflows usize element count"
            ))
        })
    })
}

#[inline]
fn expect_shape(name: &str, shape: &[usize], expected: &[usize]) -> Result<(), InferenceError> {
    if shape != expected {
        return Err(InferenceError::InvalidSafetensors(format!(
            "tensor {name} has shape {shape:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn load_f16_tensor_checked(
    sf: &SafetensorsFile,
    name: &str,
    expected: &[usize],
) -> Result<Vec<u16>, InferenceError> {
    let (data, shape) = sf.get_f16_tensor(name)?;
    expect_shape(name, shape, expected)?;
    let numel = checked_numel(shape, name)?;
    if data.len() != numel {
        return Err(InferenceError::InvalidSafetensors(format!(
            "tensor {name} has {} values, expected {numel}",
            data.len()
        )));
    }
    Ok(data)
}

fn load_f32_tensor_checked(
    sf: &SafetensorsFile,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, InferenceError> {
    let (data, shape) = sf.get_f32_tensor(name)?;
    expect_shape(name, shape, expected)?;
    let numel = checked_numel(shape, name)?;
    if data.len() != numel {
        return Err(InferenceError::InvalidSafetensors(format!(
            "tensor {name} has {} values, expected {numel}",
            data.len()
        )));
    }
    Ok(data.to_vec())
}

/// Load a conv1d weight tensor, tolerating 3D shape `[out, 1, kernel]` from safetensors.
///
/// Safetensors stores conv1d.weight as 3D `[conv_dim, 1, kernel_size]`, but we flatten
/// to 1D. We skip shape validation and only verify the element count matches.
fn load_conv1d_weight(
    sf: &SafetensorsFile,
    name: &str,
    conv_dim: usize,
    kernel_size: usize,
) -> Result<Vec<f32>, InferenceError> {
    let (data, shape) = sf.get_f32_tensor(name)?;
    let numel = checked_numel(shape, name)?;
    let expected_numel = conv_dim.checked_mul(kernel_size).ok_or_else(|| {
        InferenceError::InvalidSafetensors(format!(
            "conv1d {name}: conv_dim * kernel_size overflows"
        ))
    })?;
    if numel != expected_numel {
        return Err(InferenceError::InvalidSafetensors(format!(
            "conv1d tensor {name} has {numel} elements, expected {expected_numel} \
             (conv_dim={conv_dim}, kernel_size={kernel_size}, actual shape={shape:?})"
        )));
    }
    Ok(data.to_vec())
}

/// **Unstable**: load Qwen3.5-2B weights in f16 storage; tensor naming may change with model variants.
///
/// Load Qwen3.5-2B weights with f16 storage for large matrices.
///
/// Large projection matrices and embeddings are stored as packed f16 values; norms and other
/// small vectors remain `f32`.
pub fn load_f16_weights(
    sf: &SafetensorsFile,
    cfg: &Qwen35Config,
) -> Result<F16ModelWeights, InferenceError> {
    let hidden = cfg.hidden_size;
    let intermediate = cfg.intermediate_size;
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let value_heads = cfg.linear_num_value_heads();
    let kernel_size = cfg.linear_conv_kernel_dim;

    let embed_tokens = load_f16_tensor_checked(
        sf,
        "model.language_model.embed_tokens.weight",
        &[cfg.vocab_size, hidden],
    )?;
    let final_norm = load_f32_tensor_checked(sf, "model.language_model.norm.weight", &[hidden])?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

    for i in 0..cfg.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{i}");

        let input_layernorm =
            load_f32_tensor_checked(sf, &format!("{prefix}.input_layernorm.weight"), &[hidden])?;
        let post_attention_layernorm = load_f32_tensor_checked(
            sf,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[hidden],
        )?;

        let ffn = if cfg.is_moe() {
            let num_experts = cfg.num_experts.ok_or_else(|| {
                InferenceError::InvalidInput("MoE config missing num_experts".into())
            })?;
            let top_k = cfg.num_experts_per_tok.ok_or_else(|| {
                InferenceError::InvalidInput("MoE config missing num_experts_per_tok".into())
            })?;
            let moe_inter = cfg.moe_intermediate_size();
            let shared_inter = cfg.shared_expert_intermediate_size();

            let router = F16MoeRouter::new(
                // F1: router gate, shape [num_experts, hidden]
                load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.gate.weight"),
                    &[num_experts, hidden],
                )?,
                num_experts,
                top_k,
                hidden,
            )?;

            let experts = F16RoutedExperts::new(
                // F2: fused routed gate/up projections, shape [num_experts, 2 * moe_inter, hidden]
                load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.experts.gate_up_proj"),
                    &[num_experts, 2 * moe_inter, hidden],
                )?,
                // F3: routed down projections, shape [num_experts, hidden, moe_inter]
                load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.experts.down_proj"),
                    &[num_experts, hidden, moe_inter],
                )?,
                num_experts,
                hidden,
                moe_inter,
            )?;

            let shared_expert = F16SharedExpert::new(
                // F4: shared expert gate projection, shape [shared_inter, hidden]
                load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                    &[shared_inter, hidden],
                )?,
                // F5: shared expert up projection, shape [shared_inter, hidden]
                load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                    &[shared_inter, hidden],
                )?,
                // F6: shared expert down projection, shape [hidden, shared_inter]
                load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                    &[hidden, shared_inter],
                )?,
                // F7: shared expert scalar gate row, shape [1, hidden]
                load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.shared_expert_gate.weight"),
                    &[1, hidden],
                )?,
                hidden,
                shared_inter,
            )?;

            F16FeedForwardWeights::Moe(F16MoeLayerWeights {
                router,
                experts,
                shared_expert,
            })
        } else {
            F16FeedForwardWeights::Dense {
                gate_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    &[intermediate, hidden],
                )?,
                up_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    &[intermediate, hidden],
                )?,
                down_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    &[hidden, intermediate],
                )?,
            }
        };

        let common = F16CommonLayerWeights {
            input_layernorm,
            post_attention_layernorm,
            ffn,
        };

        let attn = if cfg.is_full_attention(i) {
            F16AttentionWeights::Full(F16FullAttentionLayerWeights {
                q_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    &[2 * q_dim, hidden],
                )?,
                k_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    &[kv_dim, hidden],
                )?,
                v_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    &[kv_dim, hidden],
                )?,
                o_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    &[hidden, q_dim],
                )?,
                q_norm: load_f32_tensor_checked(
                    sf,
                    &format!("{prefix}.self_attn.q_norm.weight"),
                    &[cfg.head_dim],
                )?,
                k_norm: load_f32_tensor_checked(
                    sf,
                    &format!("{prefix}.self_attn.k_norm.weight"),
                    &[cfg.head_dim],
                )?,
            })
        } else {
            // FIX CRITICAL-1: conv1d.weight is stored as 3D [qkv_dim, 1, kernel_size]
            // in safetensors, so we use load_conv1d_weight which skips shape validation
            // and only verifies element count.
            F16AttentionWeights::Linear(F16GatedDeltaNetWeights {
                in_proj_qkv: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    &[qkv_dim, hidden],
                )?,
                in_proj_qkv_rows: qkv_dim,
                in_proj_qkv_cols: hidden,
                in_proj_z: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.in_proj_z.weight"),
                    &[output_dim, hidden],
                )?,
                in_proj_z_rows: output_dim,
                in_proj_z_cols: hidden,
                in_proj_b: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.in_proj_b.weight"),
                    &[value_heads, hidden],
                )?,
                in_proj_b_rows: value_heads,
                in_proj_b_cols: hidden,
                in_proj_a: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.in_proj_a.weight"),
                    &[value_heads, hidden],
                )?,
                in_proj_a_rows: value_heads,
                in_proj_a_cols: hidden,
                a_log: load_f32_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.A_log"),
                    &[value_heads],
                )?,
                dt_bias: load_f32_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.dt_bias"),
                    &[value_heads],
                )?,
                conv1d_weight: load_conv1d_weight(
                    sf,
                    &format!("{prefix}.linear_attn.conv1d.weight"),
                    qkv_dim,
                    kernel_size,
                )?,
                conv_dim: qkv_dim,
                kernel_size,
                norm_weight: load_f32_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.norm.weight"),
                    &[cfg.linear_value_head_dim],
                )?,
                out_proj: load_f16_tensor_checked(
                    sf,
                    &format!("{prefix}.linear_attn.out_proj.weight"),
                    &[hidden, output_dim],
                )?,
                out_proj_rows: hidden,
                out_proj_cols: output_dim,
            })
        };

        layers.push((attn, common));
    }

    Ok(F16ModelWeights {
        embed_tokens,
        final_norm,
        layers,
    })
}

#[cfg(test)]
mod tests;
