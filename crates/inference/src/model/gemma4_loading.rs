//! Gemma 4 E2B safetensors -> [`Gemma4Weights`] loader (ADR-082 stage 5).
//!
//! Builds on the stage-2 preflight's tolerate-and-skip contract (Amendment
//! 1): KV-shared layers' `k_proj`/`v_proj`/`k_norm` checkpoint tensors are
//! never read here -- [`Gemma4LayerWeights`] carries `None` for those fields
//! on shared layers, and the forward pass resolves shared K/V through
//! [`crate::model::gemma4_cache::Gemma4KvCache`] instead. This module does
//! not call [`crate::model::gemma4_preflight::preflight_check`] directly (that
//! needs a full tensor-name/shape/dtype inventory, which a `TensorSource`
//! does not expose in bulk); it re-derives the same per-tensor shape
//! expectations directly against `cfg` and fails closed exactly like
//! `qwen35::loading::load_weights`.

use super::gemma4_config::Gemma4Config;
use super::gemma4_weights::{Gemma4LayerWeights, Gemma4Weights};
use crate::error::InferenceError;
use crate::weights::TensorSource;

const LM_PREFIX: &str = "model.language_model.";

fn load_tensor<T: TensorSource + ?Sized>(
    source: &mut T,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, InferenceError> {
    let (data, shape) = source.get_f32_tensor_owned(name)?;
    if shape != expected {
        return Err(InferenceError::ShapeMismatch {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual: shape,
        });
    }
    Ok(data)
}

fn load_scalar<T: TensorSource + ?Sized>(
    source: &mut T,
    name: &str,
) -> Result<f32, InferenceError> {
    let data = load_tensor(source, name, &[1])?;
    Ok(data[0])
}

pub(super) fn load_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &Gemma4Config,
) -> Result<Gemma4Weights, InferenceError> {
    let hidden = cfg.hidden_size;
    let per_layer_dim = cfg.hidden_size_per_layer_input;
    let ple_packed_dim = cfg.num_hidden_layers * per_layer_dim;

    let embed_tokens = load_tensor(
        source,
        &format!("{LM_PREFIX}embed_tokens.weight"),
        &[cfg.vocab_size, hidden],
    )?;
    let embed_tokens_per_layer = load_tensor(
        source,
        &format!("{LM_PREFIX}embed_tokens_per_layer.weight"),
        &[cfg.vocab_size, ple_packed_dim],
    )?;
    let norm = load_tensor(source, &format!("{LM_PREFIX}norm.weight"), &[hidden])?;
    let per_layer_model_projection = load_tensor(
        source,
        &format!("{LM_PREFIX}per_layer_model_projection.weight"),
        &[ple_packed_dim, hidden],
    )?;
    let per_layer_projection_norm = load_tensor(
        source,
        &format!("{LM_PREFIX}per_layer_projection_norm.weight"),
        &[per_layer_dim],
    )?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for layer in 0..cfg.num_hidden_layers {
        let prefix = format!("{LM_PREFIX}layers.{layer}.");
        let head_w = cfg.attn_head_dim(layer);
        let q_dim = cfg.num_attention_heads * head_w;
        let kv_dim = cfg.num_key_value_heads * head_w;
        let mlp_dim = cfg.mlp_intermediate_size(layer);

        let input_layernorm = load_tensor(
            source,
            &format!("{prefix}input_layernorm.weight"),
            &[hidden],
        )?;
        let post_attention_layernorm = load_tensor(
            source,
            &format!("{prefix}post_attention_layernorm.weight"),
            &[hidden],
        )?;
        let pre_feedforward_layernorm = load_tensor(
            source,
            &format!("{prefix}pre_feedforward_layernorm.weight"),
            &[hidden],
        )?;
        let post_feedforward_layernorm = load_tensor(
            source,
            &format!("{prefix}post_feedforward_layernorm.weight"),
            &[hidden],
        )?;
        let post_per_layer_input_norm = load_tensor(
            source,
            &format!("{prefix}post_per_layer_input_norm.weight"),
            &[hidden],
        )?;
        let layer_scalar = load_scalar(source, &format!("{prefix}layer_scalar"))?;
        let per_layer_input_gate = load_tensor(
            source,
            &format!("{prefix}per_layer_input_gate.weight"),
            &[per_layer_dim, hidden],
        )?;
        let per_layer_projection = load_tensor(
            source,
            &format!("{prefix}per_layer_projection.weight"),
            &[hidden, per_layer_dim],
        )?;

        let q_proj = load_tensor(
            source,
            &format!("{prefix}self_attn.q_proj.weight"),
            &[q_dim, hidden],
        )?;
        let o_proj = load_tensor(
            source,
            &format!("{prefix}self_attn.o_proj.weight"),
            &[hidden, q_dim],
        )?;
        let q_norm = load_tensor(
            source,
            &format!("{prefix}self_attn.q_norm.weight"),
            &[head_w],
        )?;

        let (k_proj, v_proj, k_norm) = if cfg.is_kv_shared_layer(layer) {
            (None, None, None)
        } else {
            (
                Some(load_tensor(
                    source,
                    &format!("{prefix}self_attn.k_proj.weight"),
                    &[kv_dim, hidden],
                )?),
                Some(load_tensor(
                    source,
                    &format!("{prefix}self_attn.v_proj.weight"),
                    &[kv_dim, hidden],
                )?),
                Some(load_tensor(
                    source,
                    &format!("{prefix}self_attn.k_norm.weight"),
                    &[head_w],
                )?),
            )
        };

        let gate_proj = load_tensor(
            source,
            &format!("{prefix}mlp.gate_proj.weight"),
            &[mlp_dim, hidden],
        )?;
        let up_proj = load_tensor(
            source,
            &format!("{prefix}mlp.up_proj.weight"),
            &[mlp_dim, hidden],
        )?;
        let down_proj = load_tensor(
            source,
            &format!("{prefix}mlp.down_proj.weight"),
            &[hidden, mlp_dim],
        )?;

        layers.push(Gemma4LayerWeights {
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_per_layer_input_norm,
            layer_scalar,
            per_layer_input_gate,
            per_layer_projection,
            q_proj,
            o_proj,
            q_norm,
            k_proj,
            v_proj,
            k_norm,
            gate_proj,
            up_proj,
            down_proj,
        });
    }

    Ok(Gemma4Weights {
        embed_tokens,
        embed_tokens_per_layer,
        norm,
        per_layer_model_projection,
        per_layer_projection_norm,
        layers,
    })
}
