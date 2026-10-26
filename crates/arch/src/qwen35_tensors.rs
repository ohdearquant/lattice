//! Qwen3.5 checkpoint tensor names: the per-layer prefix and the required-tensor list a config implies.
use crate::qwen35_config::Qwen35Config;

/// Canonical per-layer tensor-name prefix for a Qwen3.5 checkpoint
/// (`model.language_model.layers.{idx}`) — the single source of truth for
/// this namespace. Anything deriving a layer's tensor names (this module's
/// `qwen_required_tensor_names`, the QuaRot converter, the online-rotation
/// artifact validator) must go through this function rather than
/// re-deriving the prefix independently, so the namespace can't drift out
/// of sync with the loader.
pub fn qwen_layer_tensor_prefix(idx: usize) -> String {
    format!("model.language_model.layers.{idx}")
}

/// Return the complete list of required tensor names for a given config.
pub fn qwen_required_tensor_names(cfg: &Qwen35Config) -> Vec<String> {
    let mut names = Vec::new();
    names.push("model.language_model.embed_tokens.weight".to_string());
    names.push("model.language_model.norm.weight".to_string());
    if !cfg.tie_word_embeddings {
        names.push("lm_head.weight".to_string());
    }

    for i in 0..cfg.num_hidden_layers {
        let prefix = qwen_layer_tensor_prefix(i);
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
