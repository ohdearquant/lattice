//! Qwen3.5 tensor-name requirements, shape-checked owned tensor loading, layer-specific weight loaders, and `load_weights`.
use super::weights::{
    AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights,
    FullAttentionLayerWeights, ModelWeights, MoeLayerWeights, MoeRouter, RoutedExperts,
    SharedExpert,
};
use crate::attention::gdn::GatedDeltaNetWeights;
use crate::error::InferenceError;
use crate::model::qwen35_config::{Qwen35Config, ValidatedQwen35Config};
use crate::weights::TensorSource;

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

/// CLASS C ingress: takes `&ValidatedQwen35Config` rather than a raw `&Qwen35Config`, so an
/// unvalidated config cannot reach the required-tensor-name derivation for a checkpoint on
/// this (non-Metal) load path. See the CLASS C ingress table in the PR body.
pub(super) fn validate_required_tensor_names<T: TensorSource + ?Sized>(
    source: &mut T,
    validated_cfg: &ValidatedQwen35Config,
) -> Result<(), InferenceError> {
    let cfg: &Qwen35Config = validated_cfg;
    for name in qwen_required_tensor_names(cfg) {
        if !source.has_tensor(&name)? {
            return Err(InferenceError::MissingTensor(name));
        }
    }
    Ok(())
}

/// Loads and shape-checks a tensor, validating the declared shape against
/// `expected` *before* materializing the owned buffer. Rejecting a mismatch
/// at the metadata stage (via [`TensorSource::tensor_shape`]) avoids decoding
/// and copying a tensor the caller is about to discard, which matters for a
/// checkpoint carrying an oversized mismatched tensor. A tensor absent from
/// the source (`tensor_shape` returns `Ok(None)`) falls through to
/// `get_f32_tensor_owned`, which reports the missing-tensor error.
fn load_owned_tensor_checked<T: TensorSource + ?Sized>(
    source: &mut T,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, InferenceError> {
    // Check the header-declared shape before materializing tensor data: a source
    // that cannot report a shape up front returns None and falls through to the
    // post-load check below, but any source that can answer rejects a mismatch
    // before the (potentially huge) allocation and copy happen.
    if let Some(declared) = source.tensor_shape(name)?
        && declared != expected
    {
        return Err(InferenceError::ShapeMismatch {
            name: name.to_string(),
            expected: expected.to_vec(),
            actual: declared,
        });
    }
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

/// Doubles `value`, returning a typed error instead of panicking (debug) or wrapping
/// (release) when the config-derived dimension is too large for `usize`.
fn checked_double(value: usize, what: &str) -> Result<usize, InferenceError> {
    value
        .checked_mul(2)
        .ok_or_else(|| InferenceError::InvalidInput(format!("{what} overflows usize: 2 * {value}")))
}

/// Full-attention Q projection row count (`num_attention_heads * head_dim`), computed
/// with checked arithmetic so an attacker-controlled config.json cannot overflow it.
fn checked_full_q_dim(cfg: &Qwen35Config) -> Result<usize, InferenceError> {
    cfg.num_attention_heads
        .checked_mul(cfg.head_dim)
        .ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "full attention q_dim overflows usize: num_attention_heads({}) * head_dim({})",
                cfg.num_attention_heads, cfg.head_dim
            ))
        })
}

/// Full-attention KV projection row count (`num_key_value_heads * head_dim`), checked.
fn checked_full_kv_dim(cfg: &Qwen35Config) -> Result<usize, InferenceError> {
    cfg.num_key_value_heads
        .checked_mul(cfg.head_dim)
        .ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "full attention kv_dim overflows usize: num_key_value_heads({}) * head_dim({})",
                cfg.num_key_value_heads, cfg.head_dim
            ))
        })
}

/// GatedDeltaNet combined QKV projection row count (`Q + K + V`), checked. Mirrors
/// `Qwen35Config::linear_qkv_dim`'s formula but rejects overflow with a typed error
/// instead of panicking (debug) or wrapping (release).
fn checked_linear_qkv_dim(cfg: &Qwen35Config) -> Result<usize, InferenceError> {
    let k = cfg
        .linear_num_key_heads
        .checked_mul(cfg.linear_key_head_dim)
        .ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "linear attention k dim overflows usize: linear_num_key_heads({}) * linear_key_head_dim({})",
                cfg.linear_num_key_heads, cfg.linear_key_head_dim
            ))
        })?;
    let value_heads = cfg.linear_num_value_heads();
    let v = value_heads.checked_mul(cfg.linear_value_head_dim).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "linear attention v dim overflows usize: linear_num_value_heads({}) * linear_value_head_dim({})",
            value_heads, cfg.linear_value_head_dim
        ))
    })?;
    k.checked_add(k)
        .and_then(|qk| qk.checked_add(v))
        .ok_or_else(|| {
            InferenceError::InvalidInput(format!(
                "linear attention qkv_dim overflows usize: q({k}) + k({k}) + v({v})"
            ))
        })
}

/// GatedDeltaNet output projection row count (`linear_num_value_heads * linear_value_head_dim`), checked.
fn checked_linear_output_dim(cfg: &Qwen35Config) -> Result<usize, InferenceError> {
    let value_heads = cfg.linear_num_value_heads();
    value_heads.checked_mul(cfg.linear_value_head_dim).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "linear attention output_dim overflows usize: linear_num_value_heads({}) * linear_value_head_dim({})",
            value_heads, cfg.linear_value_head_dim
        ))
    })
}

fn load_moe_ffn_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &Qwen35Config,
    prefix: &str,
    hidden: usize,
) -> Result<FeedForwardWeights, InferenceError> {
    let num_experts = cfg
        .num_experts
        .ok_or_else(|| InferenceError::InvalidInput("MoE config missing num_experts".into()))?;
    let top_k = cfg.num_experts_per_tok.ok_or_else(|| {
        InferenceError::InvalidInput("MoE config missing num_experts_per_tok".into())
    })?;
    let moe_inter = cfg.moe_intermediate_size();
    let shared_inter = cfg.shared_expert_intermediate_size();
    let doubled_moe_inter = checked_double(
        moe_inter,
        "MoE gate_up_proj row count (2 * moe_intermediate_size)",
    )?;

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
            &[num_experts, doubled_moe_inter, hidden],
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
    cfg: &Qwen35Config,
    prefix: &str,
) -> Result<FeedForwardWeights, InferenceError> {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    // Checked against the declared safetensors axis shape (not just the flattened
    // element count) so a transposed tensor with the same element count — which would
    // otherwise be silently reinterpreted as row-major — is rejected here.
    let gp = load_owned_tensor_checked(
        source,
        &format!("{prefix}.mlp.gate_proj.weight"),
        &[inter, hidden],
    )?;
    let up = load_owned_tensor_checked(
        source,
        &format!("{prefix}.mlp.up_proj.weight"),
        &[inter, hidden],
    )?;
    let dp = load_owned_tensor_checked(
        source,
        &format!("{prefix}.mlp.down_proj.weight"),
        &[hidden, inter],
    )?;
    Ok(FeedForwardWeights::Dense(DenseFfnWeights {
        gate_proj: gp,
        up_proj: up,
        down_proj: dp,
    }))
}

fn load_full_attention_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &Qwen35Config,
    prefix: &str,
) -> Result<AttentionWeights, InferenceError> {
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    let q_dim = checked_full_q_dim(cfg)?;
    let kv_dim = checked_full_kv_dim(cfg)?;
    let doubled_q_dim = checked_double(
        q_dim,
        "full attention q_proj row count (2 * q_dim, fused sigmoid gate)",
    )?;
    let qw = load_owned_tensor_checked(
        source,
        &format!("{prefix}.self_attn.q_proj.weight"),
        &[doubled_q_dim, hidden],
    )?;
    let kw = load_owned_tensor_checked(
        source,
        &format!("{prefix}.self_attn.k_proj.weight"),
        &[kv_dim, hidden],
    )?;
    let vw = load_owned_tensor_checked(
        source,
        &format!("{prefix}.self_attn.v_proj.weight"),
        &[kv_dim, hidden],
    )?;
    let ow = load_owned_tensor_checked(
        source,
        &format!("{prefix}.self_attn.o_proj.weight"),
        &[hidden, q_dim],
    )?;
    let qn = load_owned_tensor_checked(
        source,
        &format!("{prefix}.self_attn.q_norm.weight"),
        &[head_dim],
    )?;
    let kn = load_owned_tensor_checked(
        source,
        &format!("{prefix}.self_attn.k_norm.weight"),
        &[head_dim],
    )?;
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
    cfg: &Qwen35Config,
    prefix: &str,
) -> Result<AttentionWeights, InferenceError> {
    let hidden = cfg.hidden_size;
    let qkv_dim = checked_linear_qkv_dim(cfg)?;
    let output_dim = checked_linear_output_dim(cfg)?;
    let kernel_size = cfg.linear_conv_kernel_dim;
    let value_heads = cfg.linear_num_value_heads();
    let value_dim = cfg.linear_value_head_dim;
    // Checked against the declared safetensors axis shape (not just the flattened
    // element count) so a transposed tensor with the same element count — which would
    // otherwise be silently reinterpreted as row-major — is rejected here.
    let qkv = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
        &[qkv_dim, hidden],
    )?;
    let z = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.in_proj_z.weight"),
        &[output_dim, hidden],
    )?;
    let b = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.in_proj_b.weight"),
        &[value_heads, hidden],
    )?;
    let a = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.in_proj_a.weight"),
        &[value_heads, hidden],
    )?;
    let alog = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.A_log"),
        &[value_heads],
    )?;
    let dtb = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.dt_bias"),
        &[value_heads],
    )?;
    // conv1d.weight's declared safetensors shape is [conv_dim, 1, kernel_size]; reshaped
    // to [conv_dim, kernel_size] below (conv_dim == qkv_dim for this architecture).
    let cw = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.conv1d.weight"),
        &[qkv_dim, 1, kernel_size],
    )?;
    let nw = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.norm.weight"),
        &[value_dim],
    )?;
    let op = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.out_proj.weight"),
        &[hidden, output_dim],
    )?;

    // conv1d.weight is [conv_dim, 1, kernel_size] — reshape to [conv_dim, kernel_size]
    Ok(AttentionWeights::Linear(GatedDeltaNetWeights {
        in_proj_qkv: qkv,
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z: z,
        in_proj_z_rows: output_dim,
        in_proj_z_cols: hidden,
        in_proj_b: b,
        in_proj_b_rows: value_heads,
        in_proj_b_cols: hidden,
        in_proj_a: a,
        in_proj_a_rows: value_heads,
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

/// Load all per-model and per-layer tensors, shape-validating each one against a
/// config-derived expected shape via [`load_owned_tensor_checked`]. A checkpoint
/// tensor whose shape does not match returns `Err(InferenceError::ShapeMismatch)`
/// rather than being accepted and later panicking downstream (e.g. in a GEMM
/// bounds assertion during the forward pass).
/// CLASS C ingress: takes `&ValidatedQwen35Config` rather than a raw `&Qwen35Config`, so an
/// unvalidated config cannot reach tensor materialization on this (non-Metal) load path.
/// See the CLASS C ingress table in the PR body.
pub(super) fn load_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    validated_cfg: &ValidatedQwen35Config,
) -> Result<ModelWeights, InferenceError> {
    let cfg: &Qwen35Config = validated_cfg;
    let hidden = cfg.hidden_size;

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
        let prefix = qwen_layer_tensor_prefix(i);

        let iln = load_owned_tensor_checked(
            source,
            &format!("{prefix}.input_layernorm.weight"),
            &[hidden],
        )?;
        let paln = load_owned_tensor_checked(
            source,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[hidden],
        )?;

        let ffn = if cfg.is_moe() {
            load_moe_ffn_weights(source, cfg, &prefix, hidden)?
        } else {
            load_dense_ffn_weights(source, cfg, &prefix)?
        };

        let common = CommonLayerWeights {
            input_layernorm: iln,
            post_attention_layernorm: paln,
            ffn,
        };

        let attn = if cfg.is_full_attention(i) {
            load_full_attention_weights(source, cfg, &prefix)?
        } else {
            load_linear_attention_weights(source, cfg, &prefix)?
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

#[cfg(test)]
mod tests;
