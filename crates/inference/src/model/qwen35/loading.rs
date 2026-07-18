//! Qwen3.5 tensor-name requirements, validation, owned tensor loaders, layer-specific weight
//! loaders, and `load_weights`.
//!
//! Every tensor `load_weights` reads is loaded through [`load_owned_tensor_checked`], which
//! compares the checkpoint's declared shape against a config-derived expected shape and returns
//! [`InferenceError::ShapeMismatch`] on mismatch. safetensors metadata describes each tensor's
//! own shape, not the consuming model's shape contract, so an unchecked load would accept a
//! finite but undersized tensor and only fail later, inside a forward-pass matmul. Routing every
//! tensor through the checked loader means a shape-incompatible checkpoint is rejected during
//! loading, before any forward pass runs.
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
    let num_experts = cfg
        .num_experts
        .ok_or_else(|| InferenceError::InvalidInput("MoE config missing num_experts".into()))?;
    let top_k = cfg.num_experts_per_tok.ok_or_else(|| {
        InferenceError::InvalidInput("MoE config missing num_experts_per_tok".into())
    })?;
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
    hidden: usize,
    intermediate: usize,
) -> Result<FeedForwardWeights, InferenceError> {
    let gp = load_owned_tensor_checked(
        source,
        &format!("{prefix}.mlp.gate_proj.weight"),
        &[intermediate, hidden],
    )?;
    let up = load_owned_tensor_checked(
        source,
        &format!("{prefix}.mlp.up_proj.weight"),
        &[intermediate, hidden],
    )?;
    let dp = load_owned_tensor_checked(
        source,
        &format!("{prefix}.mlp.down_proj.weight"),
        &[hidden, intermediate],
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
    hidden: usize,
) -> Result<AttentionWeights, InferenceError> {
    // q_proj carries a fused sigmoid gate alongside Q, doubling its output rows
    // relative to k_proj/v_proj (see FullAttentionLayerWeights / project_qkv).
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();
    let head_dim = cfg.head_dim;
    let qw = load_owned_tensor_checked(
        source,
        &format!("{prefix}.self_attn.q_proj.weight"),
        &[2 * q_dim, hidden],
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
    hidden: usize,
    _num_heads: usize,
    qkv_dim: usize,
    output_dim: usize,
    kernel_size: usize,
) -> Result<AttentionWeights, InferenceError> {
    let value_heads = cfg.linear_num_value_heads();
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
    let cw = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.conv1d.weight"),
        &[qkv_dim, 1, kernel_size],
    )?;
    // norm.weight is a single per-head-dim RMSNorm gamma shared across all value heads
    // (gdn.rs applies weights.norm_weight[..value_dim] identically to every head), so its
    // checkpoint shape is [linear_value_head_dim], not the full [output_dim].
    let nw = load_owned_tensor_checked(
        source,
        &format!("{prefix}.linear_attn.norm.weight"),
        &[cfg.linear_value_head_dim],
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

/// Assemble every required weight tensor for `cfg`. Each tensor is validated against a
/// config-derived expected shape as it is loaded (see the module docs); a checkpoint with a
/// finite but shape-incompatible tensor returns `Err(InferenceError::ShapeMismatch)` here
/// rather than constructing a model that panics during the forward pass.
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
            load_dense_ffn_weights(source, &prefix, hidden, cfg.intermediate_size)?
        };

        let common = CommonLayerWeights {
            input_layernorm: iln,
            post_attention_layernorm: paln,
            ffn,
        };

        let attn = if cfg.is_full_attention(i) {
            load_full_attention_weights(source, cfg, &prefix, hidden)?
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen35_config::{LayerType, Qwen35Config};

    struct NullTensorSource;

    /// A mock tensor source backed by a HashMap. Returns MissingTensor for unknown names.
    struct MockTensorSource {
        tensors: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
    }

    impl TensorSource for MockTensorSource {
        fn has_tensor(&mut self, name: &str) -> Result<bool, InferenceError> {
            Ok(self.tensors.contains_key(name))
        }
        fn tensor_shape(&mut self, name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
            Ok(self.tensors.get(name).map(|(_, s)| s.clone()))
        }
        fn get_f32_tensor_owned(
            &mut self,
            name: &str,
        ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
            self.tensors
                .get(name)
                .map(|(d, s)| (d.clone(), s.clone()))
                .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
        }
    }

    impl TensorSource for NullTensorSource {
        fn has_tensor(&mut self, _name: &str) -> Result<bool, InferenceError> {
            Ok(false)
        }
        fn tensor_shape(&mut self, _name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
            Ok(None)
        }
        fn get_f32_tensor_owned(
            &mut self,
            name: &str,
        ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
            Err(InferenceError::MissingTensor(name.to_string()))
        }
    }

    /// Verify that a key-head-shaped in_proj_b tensor in an asymmetric GDN config is rejected
    /// at load time (returns Err, not Ok or panic).
    #[test]
    fn gdn_decay_tensor_key_head_shape_rejected() {
        // qwen36_35b_a3b has linear_num_key_heads=16, linear_num_value_heads=Some(32) —
        // an asymmetric config so the shape check actually does something.
        let cfg = Qwen35Config::qwen36_35b_a3b();
        let hidden = cfg.hidden_size; // 2048
        let key_heads = cfg.linear_num_key_heads; // 16
        let value_heads = cfg.linear_num_value_heads(); // 32
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let kernel_size = cfg.linear_conv_kernel_dim;

        assert_ne!(
            key_heads, value_heads,
            "test requires asymmetric key/value heads"
        );

        let prefix = "layers.0";
        let mut src = MockTensorSource {
            tensors: [
                // qkv and z must be correctly shaped so the in_proj_b mismatch below is what
                // trips first.
                (
                    format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    (vec![0.0f32; qkv_dim * hidden], vec![qkv_dim, hidden]),
                ),
                (
                    format!("{prefix}.linear_attn.in_proj_z.weight"),
                    (vec![0.0f32; output_dim * hidden], vec![output_dim, hidden]),
                ),
                // in_proj_b is key-head-shaped ([16, hidden]) instead of value-head-shaped
                // ([32, hidden]) — this should be caught by load_owned_tensor_checked
                (
                    format!("{prefix}.linear_attn.in_proj_b.weight"),
                    (vec![0.0f32; key_heads * hidden], vec![key_heads, hidden]),
                ),
            ]
            .into_iter()
            .collect(),
        };

        let result = load_linear_attention_weights(
            &mut src,
            &cfg,
            prefix,
            hidden,
            key_heads,
            qkv_dim,
            output_dim,
            kernel_size,
        );

        match result {
            Err(InferenceError::ShapeMismatch {
                name,
                expected,
                actual,
            }) => {
                assert!(
                    name.contains("in_proj_b"),
                    "mismatch should name the in_proj_b tensor, got: {name}"
                );
                assert_eq!(
                    expected,
                    vec![value_heads, hidden],
                    "expected shape should be value-head-sized"
                );
                assert_eq!(
                    actual,
                    vec![key_heads, hidden],
                    "actual shape should reflect the key-head-shaped data we supplied"
                );
            }
            Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
            Ok(_) => panic!("expected Err for key-head-shaped decay tensor, got Ok"),
        }
    }

    #[test]
    fn moe_missing_num_experts_returns_err_not_panic() {
        let mut cfg = Qwen35Config::qwen36_35b_a3b();
        cfg.num_experts = None;
        cfg.num_experts_per_tok = Some(8);
        let mut src = NullTensorSource;
        let result = load_moe_ffn_weights(&mut src, &cfg, "layers.0", cfg.hidden_size);
        match result {
            Err(InferenceError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("num_experts"),
                    "message should name the field: {msg}"
                );
            }
            Err(e) => panic!("expected InvalidInput, got a different error: {e}"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    #[test]
    fn moe_missing_num_experts_per_tok_returns_err_not_panic() {
        let mut cfg = Qwen35Config::qwen36_35b_a3b();
        cfg.num_experts = Some(256);
        cfg.num_experts_per_tok = None;
        let mut src = NullTensorSource;
        let result = load_moe_ffn_weights(&mut src, &cfg, "layers.0", cfg.hidden_size);
        match result {
            Err(InferenceError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("num_experts_per_tok"),
                    "message should name the field: {msg}"
                );
            }
            Err(e) => panic!("expected InvalidInput, got a different error: {e}"),
            Ok(_) => panic!("expected Err, got Ok"),
        }
    }

    // ── Undersized-checkpoint rejection through `load_weights` (#1035) ─────────────────
    //
    // A finite but undersized safetensors tensor must be rejected by shape validation
    // during model assembly, not accepted and left to panic later in a forward-pass
    // matmul. Each test below builds a complete, correctly-shaped single-layer tensor
    // set, corrupts exactly one tensor's shape (row count reduced by one, data resized
    // to match so the tensor stays finite and self-consistent), and asserts that
    // `load_weights` returns `Err(InferenceError::ShapeMismatch)` naming that tensor.
    //
    // Mutation check: reverting the checked-loader routing in `load_dense_ffn_weights`,
    // `load_full_attention_weights`, or `load_linear_attention_weights` back to the
    // unchecked `load_owned_tensor` makes the corresponding test below fail (the
    // undersized tensor loads successfully and `load_weights` returns `Ok`, or a later
    // out-of-bounds read panics instead of returning a typed error).

    /// Minimal single-layer config with tiny dimensions, so the tensors constructed for
    /// these tests stay small. `layer_type` controls whether the one layer is full
    /// (GQA) or linear (GatedDeltaNet) attention; the FFN is always dense (`is_moe()`
    /// is false for every field here).
    fn tiny_config(layer_type: LayerType) -> Qwen35Config {
        Qwen35Config {
            hidden_size: 8,
            num_hidden_layers: 1,
            vocab_size: 4,
            intermediate_size: 16,
            rms_norm_eps: 1e-6,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 4,
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
            rope_parameters: None,
            linear_num_key_heads: 2,
            linear_num_value_heads: Some(2),
            linear_key_head_dim: 4,
            linear_value_head_dim: 4,
            linear_conv_kernel_dim: 4,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            tie_word_embeddings: true,
            full_attention_interval: 1,
            layer_types: vec![layer_type],
            layer_mask: vec![true],
            eos_token_id: 0,
            max_position_embeddings: 128,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            quarot_rotation_seed: None,
            vision_config: None,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
        }
    }

    /// Every required tensor for `tiny_config`'s single layer, each correctly shaped.
    fn full_valid_tensor_map(
        cfg: &Qwen35Config,
    ) -> std::collections::HashMap<String, (Vec<f32>, Vec<usize>)> {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let prefix = "model.language_model.layers.0";
        let mut m = std::collections::HashMap::new();

        m.insert(
            "model.language_model.embed_tokens.weight".to_string(),
            (
                vec![0.0f32; cfg.vocab_size * hidden],
                vec![cfg.vocab_size, hidden],
            ),
        );
        m.insert(
            "model.language_model.norm.weight".to_string(),
            (vec![0.0f32; hidden], vec![hidden]),
        );
        m.insert(
            format!("{prefix}.input_layernorm.weight"),
            (vec![0.0f32; hidden], vec![hidden]),
        );
        m.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            (vec![0.0f32; hidden], vec![hidden]),
        );
        m.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            (vec![0.0f32; inter * hidden], vec![inter, hidden]),
        );
        m.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            (vec![0.0f32; inter * hidden], vec![inter, hidden]),
        );
        m.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            (vec![0.0f32; hidden * inter], vec![hidden, inter]),
        );

        if cfg.is_full_attention(0) {
            let q_dim = cfg.full_q_dim();
            let kv_dim = cfg.full_kv_dim();
            let head_dim = cfg.head_dim;
            m.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                (vec![0.0f32; 2 * q_dim * hidden], vec![2 * q_dim, hidden]),
            );
            m.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                (vec![0.0f32; kv_dim * hidden], vec![kv_dim, hidden]),
            );
            m.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                (vec![0.0f32; kv_dim * hidden], vec![kv_dim, hidden]),
            );
            m.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                (vec![0.0f32; hidden * q_dim], vec![hidden, q_dim]),
            );
            m.insert(
                format!("{prefix}.self_attn.q_norm.weight"),
                (vec![0.0f32; head_dim], vec![head_dim]),
            );
            m.insert(
                format!("{prefix}.self_attn.k_norm.weight"),
                (vec![0.0f32; head_dim], vec![head_dim]),
            );
        } else {
            let qkv_dim = cfg.linear_qkv_dim();
            let output_dim = cfg.linear_output_dim();
            let value_heads = cfg.linear_num_value_heads();
            let kernel_size = cfg.linear_conv_kernel_dim;
            let value_dim = cfg.linear_value_head_dim;
            m.insert(
                format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                (vec![0.0f32; qkv_dim * hidden], vec![qkv_dim, hidden]),
            );
            m.insert(
                format!("{prefix}.linear_attn.in_proj_z.weight"),
                (vec![0.0f32; output_dim * hidden], vec![output_dim, hidden]),
            );
            m.insert(
                format!("{prefix}.linear_attn.in_proj_b.weight"),
                (
                    vec![0.0f32; value_heads * hidden],
                    vec![value_heads, hidden],
                ),
            );
            m.insert(
                format!("{prefix}.linear_attn.in_proj_a.weight"),
                (
                    vec![0.0f32; value_heads * hidden],
                    vec![value_heads, hidden],
                ),
            );
            m.insert(
                format!("{prefix}.linear_attn.A_log"),
                (vec![0.0f32; value_heads], vec![value_heads]),
            );
            m.insert(
                format!("{prefix}.linear_attn.dt_bias"),
                (vec![0.0f32; value_heads], vec![value_heads]),
            );
            m.insert(
                format!("{prefix}.linear_attn.conv1d.weight"),
                (
                    vec![0.0f32; qkv_dim * kernel_size],
                    vec![qkv_dim, 1, kernel_size],
                ),
            );
            m.insert(
                format!("{prefix}.linear_attn.norm.weight"),
                (vec![0.0f32; value_dim], vec![value_dim]),
            );
            m.insert(
                format!("{prefix}.linear_attn.out_proj.weight"),
                (vec![0.0f32; hidden * output_dim], vec![hidden, output_dim]),
            );
        }

        m
    }

    /// Shrink tensor `name`'s leading dimension by one row, keeping the data length
    /// consistent with the new (still finite) shape — an undersized-but-well-formed
    /// checkpoint tensor, not a corrupt/NaN one.
    fn shrink_first_dim(
        tensors: &mut std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
        name: &str,
    ) {
        let (data, shape) = tensors.get_mut(name).expect("tensor must be present");
        assert!(shape[0] > 0, "cannot shrink a zero-sized dimension");
        let row_len: usize = shape[1..].iter().product();
        shape[0] -= 1;
        data.truncate(data.len() - row_len);
    }

    #[test]
    fn dense_ffn_undersized_tensor_rejected() {
        let cfg = tiny_config(LayerType::LinearAttention);
        let mut tensors = full_valid_tensor_map(&cfg);
        let name = "model.language_model.layers.0.mlp.down_proj.weight";
        shrink_first_dim(&mut tensors, name);
        let mut src = MockTensorSource { tensors };

        match load_weights(&mut src, &cfg) {
            Err(InferenceError::ShapeMismatch { name: got_name, .. }) => assert_eq!(got_name, name),
            Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
            Ok(_) => panic!("expected Err for undersized dense-FFN tensor, got Ok"),
        }
    }

    #[test]
    fn full_attention_undersized_tensor_rejected() {
        let cfg = tiny_config(LayerType::FullAttention);
        let mut tensors = full_valid_tensor_map(&cfg);
        let name = "model.language_model.layers.0.self_attn.q_proj.weight";
        shrink_first_dim(&mut tensors, name);
        let mut src = MockTensorSource { tensors };

        match load_weights(&mut src, &cfg) {
            Err(InferenceError::ShapeMismatch { name: got_name, .. }) => assert_eq!(got_name, name),
            Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
            Ok(_) => panic!("expected Err for undersized attention tensor, got Ok"),
        }
    }

    #[test]
    fn gdn_undersized_tensor_rejected() {
        let cfg = tiny_config(LayerType::LinearAttention);
        let mut tensors = full_valid_tensor_map(&cfg);
        let name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
        shrink_first_dim(&mut tensors, name);
        let mut src = MockTensorSource { tensors };

        match load_weights(&mut src, &cfg) {
            Err(InferenceError::ShapeMismatch { name: got_name, .. }) => assert_eq!(got_name, name),
            Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
            Ok(_) => panic!("expected Err for undersized GDN tensor, got Ok"),
        }
    }
}
