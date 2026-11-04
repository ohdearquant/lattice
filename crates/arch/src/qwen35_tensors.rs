//! Qwen3.5 checkpoint tensor names: the per-layer prefix and the required-tensor list a config implies.
use crate::qwen35_config::Qwen35Config;
use lattice_inference_core::InferenceError;

/// Nested Qwen3.5 text-model tensor namespace.
#[doc(hidden)]
pub const LANGUAGE_MODEL_PREFIX: &str = "model.language_model.";
/// Flat Qwen3.5 text-model tensor namespace.
#[doc(hidden)]
pub const FLAT_MODEL_PREFIX: &str = "model.";
/// Embedding tensor that identifies the nested text-model namespace.
#[doc(hidden)]
pub const LANGUAGE_MODEL_EMBED: &str = "model.language_model.embed_tokens.weight";
/// Embedding tensor that identifies the flat text-model namespace.
#[doc(hidden)]
pub const FLAT_MODEL_EMBED: &str = "model.embed_tokens.weight";

/// Resolve a checkpoint's text-model namespace from its embedding anchors.
///
/// With neither anchor present, use the canonical namespace so the loader reports
/// the same missing-tensor error as it does for a canonical checkpoint.
///
/// # Errors
/// Returns an invalid-safetensors error when both anchors are present.
#[doc(hidden)]
pub fn resolve_tensor_prefix(
    has_language_model: bool,
    has_flat_model: bool,
) -> Result<&'static str, InferenceError> {
    match (has_language_model, has_flat_model) {
        (true, true) => Err(InferenceError::InvalidSafetensors(format!(
            "checkpoint contains both {LANGUAGE_MODEL_EMBED} and {FLAT_MODEL_EMBED}"
        ))),
        (false, true) => Ok(FLAT_MODEL_PREFIX),
        _ => Ok(LANGUAGE_MODEL_PREFIX),
    }
}

/// Canonical per-layer tensor-name prefix for a Qwen3.5 checkpoint
/// (`model.language_model.layers.{idx}`). Canonical checkpoint consumers use
/// this function, while loaders use [`qwen_layer_tensor_prefix_with_prefix`]
/// after resolving the namespace. Both share the same name construction so
/// the required-tensor list, QuaRot converter and artifact validator stay in sync.
pub fn qwen_layer_tensor_prefix(idx: usize) -> String {
    qwen_layer_tensor_prefix_with_prefix(LANGUAGE_MODEL_PREFIX, idx)
}

/// Build a layer prefix using the namespace returned by [`resolve_tensor_prefix`].
#[doc(hidden)]
pub fn qwen_layer_tensor_prefix_with_prefix(tensor_prefix: &str, idx: usize) -> String {
    format!("{tensor_prefix}layers.{idx}")
}

/// Return the complete list of required tensor names for a given config.
pub fn qwen_required_tensor_names(cfg: &Qwen35Config) -> Vec<String> {
    qwen_required_tensor_names_with_prefix(cfg, LANGUAGE_MODEL_PREFIX)
}

/// Build required names using the namespace returned by [`resolve_tensor_prefix`].
#[doc(hidden)]
pub fn qwen_required_tensor_names_with_prefix(
    cfg: &Qwen35Config,
    tensor_prefix: &str,
) -> Vec<String> {
    let mut names = Vec::new();
    names.push(format!("{tensor_prefix}embed_tokens.weight"));
    names.push(format!("{tensor_prefix}norm.weight"));
    if !cfg.tie_word_embeddings {
        names.push("lm_head.weight".to_string());
    }

    for i in 0..cfg.num_hidden_layers {
        let prefix = qwen_layer_tensor_prefix_with_prefix(tensor_prefix, i);
        names.push(format!("{prefix}.input_layernorm.weight"));
        names.push(format!("{prefix}.post_attention_layernorm.weight"));

        if cfg.is_full_attention(i) {
            names.push(format!("{prefix}.self_attn.q_proj.weight"));
            names.push(format!("{prefix}.self_attn.k_proj.weight"));
            names.push(format!("{prefix}.self_attn.v_proj.weight"));
            names.push(format!("{prefix}.self_attn.o_proj.weight"));
            names.push(format!("{prefix}.self_attn.q_norm.weight"));
            names.push(format!("{prefix}.self_attn.k_norm.weight"));
        } else {
            names.push(format!("{prefix}.linear_attn.in_proj_qkv.weight"));
            names.push(format!("{prefix}.linear_attn.in_proj_z.weight"));
            names.push(format!("{prefix}.linear_attn.in_proj_b.weight"));
            names.push(format!("{prefix}.linear_attn.in_proj_a.weight"));
            names.push(format!("{prefix}.linear_attn.A_log"));
            names.push(format!("{prefix}.linear_attn.dt_bias"));
            names.push(format!("{prefix}.linear_attn.conv1d.weight"));
            names.push(format!("{prefix}.linear_attn.norm.weight"));
            names.push(format!("{prefix}.linear_attn.out_proj.weight"));
        }

        if cfg.is_moe() {
            names.push(format!("{prefix}.mlp.gate.weight"));
            names.push(format!("{prefix}.mlp.experts.gate_up_proj"));
            names.push(format!("{prefix}.mlp.experts.down_proj"));
            names.push(format!("{prefix}.mlp.shared_expert.gate_proj.weight"));
            names.push(format!("{prefix}.mlp.shared_expert.up_proj.weight"));
            names.push(format!("{prefix}.mlp.shared_expert.down_proj.weight"));
            names.push(format!("{prefix}.mlp.shared_expert_gate.weight"));
        } else {
            names.push(format!("{prefix}.mlp.gate_proj.weight"));
            names.push(format!("{prefix}.mlp.up_proj.weight"));
            names.push(format!("{prefix}.mlp.down_proj.weight"));
        }
    }

    names
}
