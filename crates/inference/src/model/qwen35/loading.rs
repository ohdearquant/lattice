use super::weights::{
    AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights,
    FullAttentionLayerWeights, ModelWeights, MoeLayerWeights, MoeRouter, RoutedExperts,
    SharedExpert,
};
use crate::attention::gdn::GatedDeltaNetWeights;
use crate::error::InferenceError;
use crate::model::qwen35_config::Qwen35Config;
use crate::weights::TensorSource;

/// Return the complete list of required tensor names for a given config.
pub fn qwen_required_tensor_names(cfg: &Qwen35Config) -> Vec<String> {
    let mut names = Vec::new();
    names.push("model.language_model.embed_tokens.weight".to_string());
    names.push("model.language_model.norm.weight".to_string());
    if !cfg.tie_word_embeddings {
        names.push("lm_head.weight".to_string());
    }

    for i in 0..cfg.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{i}");
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

pub(super) fn validate_required_tensor_names<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &Qwen35Config,
) -> Result<(), InferenceError> {
    for name in qwen_required_tensor_names(cfg) {
        if !source.has_tensor(&name)? {
            return Err(InferenceError::MissingTensor(name));
        }
    }
    Ok(())
}

fn load_owned_tensor<T: TensorSource + ?Sized>(
    source: &mut T,
    name: &str,
) -> Result<Vec<f32>, InferenceError> {
    let (data, _) = source.get_f32_tensor_owned(name)?;
    Ok(data)
}

fn load_owned_tensor_checked<T: TensorSource + ?Sized>(
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

fn load_moe_ffn_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &Qwen35Config,
    prefix: &str,
    hidden: usize,
) -> Result<FeedForwardWeights, InferenceError> {
    let num_experts = cfg.num_experts.expect("MoE config has num_experts");
    let top_k = cfg
        .num_experts_per_tok
        .expect("MoE config has num_experts_per_tok");
    let moe_inter = cfg.moe_intermediate_size();
    let shared_inter = cfg.shared_expert_intermediate_size();

    let router = MoeRouter::new(
        load_owned_tensor_checked(
            source,
            &format!("{prefix}.mlp.gate.weight"),
            &[num_experts, hidden],
        )?,
        num_experts,
        top_k,
        hidden,
    )?;

    let experts = RoutedExperts::new(
        load_owned_tensor_checked(
            source,
            &format!("{prefix}.mlp.experts.gate_up_proj"),
            &[num_experts, 2 * moe_inter, hidden],
        )?,
        load_owned_tensor_checked(
            source,
            &format!("{prefix}.mlp.experts.down_proj"),
            &[num_experts, hidden, moe_inter],
        )?,
        num_experts,
        hidden,
        moe_inter,
    )?;

    let shared_expert = SharedExpert::new(
        load_owned_tensor_checked(
            source,
            &format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
            &[shared_inter, hidden],
        )?,
        load_owned_tensor_checked(
            source,
            &format!("{prefix}.mlp.shared_expert.up_proj.weight"),
            &[shared_inter, hidden],
        )?,
        load_owned_tensor_checked(
            source,
            &format!("{prefix}.mlp.shared_expert.down_proj.weight"),
            &[hidden, shared_inter],
        )?,
        load_owned_tensor_checked(
            source,
            &format!("{prefix}.mlp.shared_expert_gate.weight"),
            &[1, hidden],
        )?,
        hidden,
        shared_inter,
    )?;

    Ok(FeedForwardWeights::Moe(MoeLayerWeights {
        router,
        experts,
        shared_expert,
    }))
}

fn load_dense_ffn_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    prefix: &str,
) -> Result<FeedForwardWeights, InferenceError> {
    let gp = load_owned_tensor(source, &format!("{prefix}.mlp.gate_proj.weight"))?;
    let up = load_owned_tensor(source, &format!("{prefix}.mlp.up_proj.weight"))?;
    let dp = load_owned_tensor(source, &format!("{prefix}.mlp.down_proj.weight"))?;
    Ok(FeedForwardWeights::Dense(DenseFfnWeights {
        gate_proj: gp,
        up_proj: up,
        down_proj: dp,
    }))
}

fn load_full_attention_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    prefix: &str,
) -> Result<AttentionWeights, InferenceError> {
    let qw = load_owned_tensor(source, &format!("{prefix}.self_attn.q_proj.weight"))?;
    let kw = load_owned_tensor(source, &format!("{prefix}.self_attn.k_proj.weight"))?;
    let vw = load_owned_tensor(source, &format!("{prefix}.self_attn.v_proj.weight"))?;
    let ow = load_owned_tensor(source, &format!("{prefix}.self_attn.o_proj.weight"))?;
    let qn = load_owned_tensor(source, &format!("{prefix}.self_attn.q_norm.weight"))?;
    let kn = load_owned_tensor(source, &format!("{prefix}.self_attn.k_norm.weight"))?;
    Ok(AttentionWeights::Full(FullAttentionLayerWeights {
        q_proj: qw,
        k_proj: kw,
        v_proj: vw,
        o_proj: ow,
        q_norm: qn,
        k_norm: kn,
    }))
}

fn load_linear_attention_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    _cfg: &Qwen35Config,
    prefix: &str,
    hidden: usize,
    num_heads: usize,
    qkv_dim: usize,
    output_dim: usize,
    kernel_size: usize,
) -> Result<AttentionWeights, InferenceError> {
    let qkv = load_owned_tensor(source, &format!("{prefix}.linear_attn.in_proj_qkv.weight"))?;
    let z = load_owned_tensor(source, &format!("{prefix}.linear_attn.in_proj_z.weight"))?;
    let b = load_owned_tensor(source, &format!("{prefix}.linear_attn.in_proj_b.weight"))?;
    let a = load_owned_tensor(source, &format!("{prefix}.linear_attn.in_proj_a.weight"))?;
    let alog = load_owned_tensor(source, &format!("{prefix}.linear_attn.A_log"))?;
    let dtb = load_owned_tensor(source, &format!("{prefix}.linear_attn.dt_bias"))?;
    let cw = load_owned_tensor(source, &format!("{prefix}.linear_attn.conv1d.weight"))?;
    let nw = load_owned_tensor(source, &format!("{prefix}.linear_attn.norm.weight"))?;
    let op = load_owned_tensor(source, &format!("{prefix}.linear_attn.out_proj.weight"))?;

    // conv1d.weight is [conv_dim, 1, kernel_size] — reshape to [conv_dim, kernel_size]
    Ok(AttentionWeights::Linear(GatedDeltaNetWeights {
        in_proj_qkv: qkv,
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z: z,
        in_proj_z_rows: output_dim,
        in_proj_z_cols: hidden,
        in_proj_b: b,
        in_proj_b_rows: num_heads,
        in_proj_b_cols: hidden,
        in_proj_a: a,
        in_proj_a_rows: num_heads,
        in_proj_a_cols: hidden,
        a_log: alog,
        dt_bias: dtb,
        conv1d_weight: cw,
        conv_dim: qkv_dim,
        kernel_size,
        norm_weight: nw,
        out_proj: op,
        out_proj_rows: hidden,
        out_proj_cols: output_dim,
    }))
}

pub(super) fn load_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &Qwen35Config,
) -> Result<ModelWeights, InferenceError> {
    let hidden = cfg.hidden_size;
    let num_heads = cfg.linear_num_key_heads;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let kernel_size = cfg.linear_conv_kernel_dim;

    let embed_tokens = load_owned_tensor_checked(
        source,
        "model.language_model.embed_tokens.weight",
        &[cfg.vocab_size, hidden],
    )?;

    let lm_head = if cfg.tie_word_embeddings {
        None
    } else {
        Some(load_owned_tensor_checked(
            source,
            "lm_head.weight",
            &[cfg.vocab_size, hidden],
        )?)
    };

    let final_norm =
        load_owned_tensor_checked(source, "model.language_model.norm.weight", &[hidden])?;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

    for i in 0..cfg.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{i}");

        let iln = load_owned_tensor(source, &format!("{prefix}.input_layernorm.weight"))?;
        let paln = load_owned_tensor(source, &format!("{prefix}.post_attention_layernorm.weight"))?;

        let ffn = if cfg.is_moe() {
            load_moe_ffn_weights(source, cfg, &prefix, hidden)?
        } else {
            load_dense_ffn_weights(source, &prefix)?
        };

        let common = CommonLayerWeights {
            input_layernorm: iln,
            post_attention_layernorm: paln,
            ffn,
        };

        let attn = if cfg.is_full_attention(i) {
            load_full_attention_weights(source, &prefix)?
        } else {
            load_linear_attention_weights(
                source,
                cfg,
                &prefix,
                hidden,
                num_heads,
                qkv_dim,
                output_dim,
                kernel_size,
            )?
        };

        layers.push((attn, common));
    }

    Ok(ModelWeights {
        embed_tokens,
        lm_head,
        final_norm,
        layers,
    })
}
