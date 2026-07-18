//! Qwen3.5 tensor-name requirements, shape-checked owned tensor loading, layer-specific weight loaders, and `load_weights`.
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
    let intermediate = cfg.intermediate_size;
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
pub(super) fn load_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &Qwen35Config,
) -> Result<ModelWeights, InferenceError> {
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
mod tests {
    use super::*;
    use crate::model::qwen35_config::Qwen35Config;

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

    /// A mock tensor source whose declared metadata shape (`tensor_shape`) can differ
    /// from the actual stored tensor shape, and which counts calls to
    /// `get_f32_tensor_owned`. Used to prove the shape preflight in
    /// `load_owned_tensor_checked` rejects a mismatch from metadata alone, without
    /// ever materializing the owned buffer.
    struct CallCountingMockTensorSource {
        /// name -> (data, declared shape returned by `tensor_shape`)
        tensors: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
        owned_get_calls: std::cell::Cell<usize>,
    }

    impl TensorSource for CallCountingMockTensorSource {
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
            self.owned_get_calls.set(self.owned_get_calls.get() + 1);
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

        assert_ne!(
            key_heads, value_heads,
            "test requires asymmetric key/value heads"
        );

        let prefix = "layers.0";
        let mut src = MockTensorSource {
            tensors: [
                // qkv and z are correctly-shaped so the in_proj_b mismatch below is the
                // first (and only) error load_linear_attention_weights should surface.
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

        let result = load_linear_attention_weights(&mut src, &cfg, prefix);

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

    /// A finite but undersized dense-FFN `gate_proj` tensor must be rejected during
    /// loading (`Err(ShapeMismatch)`), not accepted and left to panic downstream in
    /// the GEMM bounds assertion during the forward pass.
    #[test]
    fn dense_ffn_undersized_gate_proj_shape_rejected() {
        let cfg = Qwen35Config::qwen35_0_8b();
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let prefix = "layers.0";
        let undersized_rows = intermediate / 2;

        let mut src = MockTensorSource {
            tensors: [(
                format!("{prefix}.mlp.gate_proj.weight"),
                (
                    vec![0.0f32; undersized_rows * hidden],
                    vec![undersized_rows, hidden],
                ),
            )]
            .into_iter()
            .collect(),
        };

        let result = load_dense_ffn_weights(&mut src, &cfg, prefix);

        match result {
            Err(InferenceError::ShapeMismatch {
                name,
                expected,
                actual,
            }) => {
                assert!(
                    name.contains("gate_proj"),
                    "mismatch should name the gate_proj tensor, got: {name}"
                );
                assert_eq!(expected, vec![intermediate, hidden]);
                assert_eq!(actual, vec![undersized_rows, hidden]);
            }
            Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
            Ok(_) => panic!("expected Err for undersized gate_proj, got Ok"),
        }
    }

    /// A finite but undersized full-attention `q_proj` tensor (half the required
    /// `2*q_dim` rows for the attn_output_gate=true fused Q+gate projection) must be
    /// rejected during loading, not accepted and left to panic downstream.
    #[test]
    fn full_attention_undersized_q_proj_shape_rejected() {
        let cfg = Qwen35Config::qwen35_0_8b();
        let hidden = cfg.hidden_size;
        let q_dim = cfg.full_q_dim();
        let prefix = "layers.0";
        let undersized_rows = q_dim; // expected 2*q_dim; supply only q_dim rows

        let mut src = MockTensorSource {
            tensors: [(
                format!("{prefix}.self_attn.q_proj.weight"),
                (
                    vec![0.0f32; undersized_rows * hidden],
                    vec![undersized_rows, hidden],
                ),
            )]
            .into_iter()
            .collect(),
        };

        let result = load_full_attention_weights(&mut src, &cfg, prefix);

        match result {
            Err(InferenceError::ShapeMismatch {
                name,
                expected,
                actual,
            }) => {
                assert!(
                    name.contains("q_proj"),
                    "mismatch should name the q_proj tensor, got: {name}"
                );
                assert_eq!(expected, vec![2 * q_dim, hidden]);
                assert_eq!(actual, vec![undersized_rows, hidden]);
            }
            Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
            Ok(_) => panic!("expected Err for undersized q_proj, got Ok"),
        }
    }

    /// A finite but undersized GDN `in_proj_qkv` tensor must be rejected during
    /// loading, not accepted and left to panic downstream.
    #[test]
    fn gdn_undersized_in_proj_qkv_shape_rejected() {
        let cfg = Qwen35Config::qwen35_0_8b();
        let hidden = cfg.hidden_size;
        let qkv_dim = cfg.linear_qkv_dim();
        let prefix = "layers.0";
        let undersized_rows = qkv_dim / 2;

        let mut src = MockTensorSource {
            tensors: [(
                format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                (
                    vec![0.0f32; undersized_rows * hidden],
                    vec![undersized_rows, hidden],
                ),
            )]
            .into_iter()
            .collect(),
        };

        let result = load_linear_attention_weights(&mut src, &cfg, prefix);

        match result {
            Err(InferenceError::ShapeMismatch {
                name,
                expected,
                actual,
            }) => {
                assert!(
                    name.contains("in_proj_qkv"),
                    "mismatch should name the in_proj_qkv tensor, got: {name}"
                );
                assert_eq!(expected, vec![qkv_dim, hidden]);
                assert_eq!(actual, vec![undersized_rows, hidden]);
            }
            Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
            Ok(_) => panic!("expected Err for undersized in_proj_qkv, got Ok"),
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

    // ── checked_* overflow-guard boundary tests ─────────────────────────────────────
    //
    // These dimension products are derived from config.json fields (an untrusted
    // checkpoint directory). A config with maliciously large but individually
    // in-range `usize` fields (e.g. num_attention_heads and head_dim each valid
    // on their own) can make the *product* overflow usize, which would silently
    // wrap to a small value in release builds and load a shape-incompatible
    // (undersized) tensor buffer, or panic in debug builds. `checked_*` must
    // reject the overflowing case with a typed `InvalidInput` error and accept a
    // large-but-non-overflowing product.

    #[test]
    fn checked_double_accepts_large_valid_and_rejects_overflow() {
        assert_eq!(checked_double(usize::MAX / 2, "x").unwrap(), usize::MAX - 1);
        assert!(checked_double(usize::MAX, "x").is_err());
    }

    #[test]
    fn checked_full_q_dim_accepts_large_valid_and_rejects_overflow() {
        let mut cfg = Qwen35Config::qwen35_0_8b();
        cfg.num_attention_heads = 1 << 40;
        cfg.head_dim = 1 << 20;
        assert_eq!(checked_full_q_dim(&cfg).unwrap(), 1 << 60);

        cfg.num_attention_heads = usize::MAX;
        cfg.head_dim = 2;
        match checked_full_q_dim(&cfg) {
            Err(InferenceError::InvalidInput(msg)) => assert!(msg.contains("q_dim")),
            other => panic!("expected InvalidInput overflow error, got {other:?}"),
        }
    }

    #[test]
    fn checked_full_kv_dim_accepts_large_valid_and_rejects_overflow() {
        let mut cfg = Qwen35Config::qwen35_0_8b();
        cfg.num_key_value_heads = 1 << 40;
        cfg.head_dim = 1 << 20;
        assert_eq!(checked_full_kv_dim(&cfg).unwrap(), 1 << 60);

        cfg.num_key_value_heads = usize::MAX;
        cfg.head_dim = 2;
        match checked_full_kv_dim(&cfg) {
            Err(InferenceError::InvalidInput(msg)) => assert!(msg.contains("kv_dim")),
            other => panic!("expected InvalidInput overflow error, got {other:?}"),
        }
    }

    #[test]
    fn checked_linear_qkv_dim_accepts_large_valid_and_rejects_overflow() {
        let mut cfg = Qwen35Config::qwen35_0_8b();
        cfg.linear_num_key_heads = 1 << 20;
        cfg.linear_key_head_dim = 1 << 20;
        cfg.linear_num_value_heads = Some(1 << 20);
        cfg.linear_value_head_dim = 1 << 20;
        // k = v = 1<<40, qkv = k + k + v = 3<<40 -- large but well within usize range.
        assert_eq!(checked_linear_qkv_dim(&cfg).unwrap(), 3 * (1usize << 40));

        cfg.linear_num_key_heads = usize::MAX;
        cfg.linear_key_head_dim = 2;
        match checked_linear_qkv_dim(&cfg) {
            Err(InferenceError::InvalidInput(msg)) => assert!(msg.contains("overflows usize")),
            other => panic!("expected InvalidInput overflow error, got {other:?}"),
        }
    }

    #[test]
    fn checked_linear_output_dim_accepts_large_valid_and_rejects_overflow() {
        let mut cfg = Qwen35Config::qwen35_0_8b();
        cfg.linear_num_value_heads = Some(1 << 40);
        cfg.linear_value_head_dim = 1 << 20;
        assert_eq!(checked_linear_output_dim(&cfg).unwrap(), 1 << 60);

        cfg.linear_num_value_heads = Some(usize::MAX);
        cfg.linear_value_head_dim = 2;
        match checked_linear_output_dim(&cfg) {
            Err(InferenceError::InvalidInput(msg)) => assert!(msg.contains("output_dim")),
            other => panic!("expected InvalidInput overflow error, got {other:?}"),
        }
    }

    // ── shape-preflight-before-materialization tests ───────────────────────────────
    //
    // `load_owned_tensor_checked` must reject a shape mismatch using only the
    // metadata `tensor_shape` returns, before ever calling `get_f32_tensor_owned`.
    // A checkpoint tensor whose declared shape is oversized should not pay the cost
    // of decoding/copying the owned buffer just to be rejected afterward.

    #[test]
    fn mismatched_shape_rejected_without_materializing_owned_buffer() {
        let name = "layers.0.mlp.gate_proj.weight";
        // Declared (metadata) shape is huge and mismatched; the actual stored data is
        // tiny, so if the preflight were skipped and get_f32_tensor_owned were called,
        // it would return this tiny buffer paired with the huge declared shape below.
        let declared_mismatched_shape = vec![1 << 20, 1 << 20];
        let mut src = CallCountingMockTensorSource {
            tensors: [(
                name.to_string(),
                (vec![0.0f32; 4], declared_mismatched_shape.clone()),
            )]
            .into_iter()
            .collect(),
            owned_get_calls: std::cell::Cell::new(0),
        };

        let result = load_owned_tensor_checked(&mut src, name, &[4, 4]);

        match result {
            Err(InferenceError::ShapeMismatch {
                name: got_name,
                expected,
                actual,
            }) => {
                assert_eq!(got_name, name);
                assert_eq!(expected, vec![4, 4]);
                assert_eq!(actual, declared_mismatched_shape);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
        assert_eq!(
            src.owned_get_calls.get(),
            0,
            "a mismatched tensor must be rejected via the metadata preflight \
             without ever materializing the owned buffer"
        );
    }

    #[test]
    fn correctly_shaped_tensor_still_loads() {
        let name = "layers.0.mlp.gate_proj.weight";
        let mut src = CallCountingMockTensorSource {
            tensors: [(name.to_string(), (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]))]
                .into_iter()
                .collect(),
            owned_get_calls: std::cell::Cell::new(0),
        };

        let result = load_owned_tensor_checked(&mut src, name, &[2, 2]);

        assert_eq!(result.unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            src.owned_get_calls.get(),
            1,
            "a correctly-shaped tensor must still be materialized exactly once"
        );
    }
}
