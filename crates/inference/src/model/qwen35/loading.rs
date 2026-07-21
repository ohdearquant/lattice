//! Qwen3.5 tensor-name requirements, validation, owned tensor loaders, layer-specific weight loaders, and `load_weights`.
use super::weights::{
    AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights,
    FullAttentionLayerWeights, ModelWeights, MoeLayerWeights, MoeRouter, RoutedExperts,
    SharedExpert,
};
use crate::attention::gdn::GatedDeltaNetWeights;
use crate::error::InferenceError;
use crate::model::qwen35_config::Qwen35Config;
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
    let gate_up_rows = moe_inter.checked_mul(2).ok_or_else(|| {
        InferenceError::InvalidInput(format!(
            "moe_intermediate_size {moe_inter} overflows usize when doubled for gate_up_proj shape"
        ))
    })?;

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
            &[num_experts, gate_up_rows, hidden],
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
    // `config.json` is untrusted input: use the checked accessors so a pathological
    // `num_attention_heads` / `num_key_value_heads` overflows into a typed error here
    // instead of wrapping to an undersized `q_dim`/`kv_dim` that later passes a
    // small-tensor shape check and panics or corrupts state once the forward pass
    // indexes by the original, unwrapped head count.
    let q_dim = cfg.checked_full_q_dim()?;
    let kv_dim = cfg.checked_full_kv_dim()?;
    let q_rows = crate::model::qwen35_config::checked_double(q_dim, "full_q_dim")?;
    // Checked against the declared safetensors axis shape (not just the flattened
    // element count) so a transposed tensor with the same element count — which would
    // otherwise be silently reinterpreted as row-major — is rejected here.
    let qw = load_owned_tensor_checked(
        source,
        &format!("{prefix}.self_attn.q_proj.weight"),
        &[q_rows, hidden],
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

pub(super) fn load_weights<T: TensorSource + ?Sized>(
    source: &mut T,
    cfg: &Qwen35Config,
) -> Result<ModelWeights, InferenceError> {
    let hidden = cfg.hidden_size;
    let num_heads = cfg.linear_num_key_heads;
    // `config.json` is untrusted input: use the checked accessors so a pathological
    // linear-attention head/dim combination overflows into a typed error here instead
    // of wrapping into an undersized `qkv_dim`/`output_dim` that later reaches the
    // forward pass under the original, unwrapped config.
    let qkv_dim = cfg.checked_linear_qkv_dim()?;
    let output_dim = cfg.checked_linear_output_dim()?;
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

    /// A mock tensor source that counts calls to `get_f32_tensor_owned`, so a test can
    /// assert a shape mismatch is caught from the header-declared shape without ever
    /// copying the tensor's data.
    struct CountingTensorSource {
        tensors: std::collections::HashMap<String, (Vec<f32>, Vec<usize>)>,
        materialize_calls: std::cell::Cell<usize>,
    }

    impl TensorSource for CountingTensorSource {
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
            self.materialize_calls.set(self.materialize_calls.get() + 1);
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
                // qkv and z are correctly cfg-shaped; the point of this test is the
                // in_proj_b mismatch below, not these tensors.
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

    /// A `gate_proj` tensor whose safetensors-declared axis shape is transposed
    /// (`[hidden, intermediate]` instead of `[intermediate, hidden]`) but has the same
    /// total element count must be rejected, not silently reinterpreted as row-major.
    /// Before the loader validated declared axis shapes, only the flattened element
    /// count was checked, so a transposed tensor with matching element count passed
    /// ingress and was misparsed, producing silently wrong outputs downstream.
    #[test]
    fn dense_ffn_rejects_transposed_gate_proj_with_matching_element_count() {
        let cfg = Qwen35Config {
            hidden_size: 4,
            intermediate_size: 8,
            ..Qwen35Config::qwen35_2b()
        };
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let prefix = "layers.0";

        let mut src = MockTensorSource {
            tensors: [
                // Declared as [hidden, intermediate] — the transpose of the expected
                // [intermediate, hidden] — with the same total element count (32).
                (
                    format!("{prefix}.mlp.gate_proj.weight"),
                    (vec![0.0f32; hidden * inter], vec![hidden, inter]),
                ),
                (
                    format!("{prefix}.mlp.up_proj.weight"),
                    (vec![0.0f32; inter * hidden], vec![inter, hidden]),
                ),
                (
                    format!("{prefix}.mlp.down_proj.weight"),
                    (vec![0.0f32; hidden * inter], vec![hidden, inter]),
                ),
            ]
            .into_iter()
            .collect(),
        };

        match load_dense_ffn_weights(&mut src, &cfg, prefix) {
            Err(InferenceError::ShapeMismatch {
                name,
                expected,
                actual,
            }) => {
                assert!(
                    name.contains("gate_proj"),
                    "mismatch should name the gate_proj tensor, got: {name}"
                );
                assert_eq!(expected, vec![inter, hidden]);
                assert_eq!(actual, vec![hidden, inter]);
            }
            Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
            Ok(_) => panic!("expected Err for transposed gate_proj, got Ok"),
        }
    }

    #[test]
    fn shape_mismatch_rejected_before_tensor_data_is_materialized() {
        let mut src = CountingTensorSource {
            tensors: [("t".to_string(), (vec![0.0f32; 4], vec![4, 1]))]
                .into_iter()
                .collect(),
            materialize_calls: std::cell::Cell::new(0),
        };

        let result = load_owned_tensor_checked(&mut src, "t", &[1, 4]);

        assert!(
            matches!(result, Err(InferenceError::ShapeMismatch { .. })),
            "expected ShapeMismatch, got {result:?}"
        );
        assert_eq!(
            src.materialize_calls.get(),
            0,
            "a declared-shape mismatch must be rejected before the tensor data is copied"
        );
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

    #[test]
    fn moe_gate_up_shape_overflow_returns_err_not_panic() {
        let mut cfg = Qwen35Config::qwen36_35b_a3b();
        cfg.num_experts = Some(256);
        cfg.num_experts_per_tok = Some(8);
        cfg.moe_intermediate_size = Some(usize::MAX);
        let hidden = cfg.hidden_size;
        // The router's gate.weight is present and correctly shaped so the loader
        // actually reaches the overflowing gate_up_proj shape computation instead of
        // failing earlier on a missing tensor.
        let mut src = MockTensorSource {
            tensors: [(
                "layers.0.mlp.gate.weight".to_string(),
                (vec![0.0f32; 256 * hidden], vec![256, hidden]),
            )]
            .into_iter()
            .collect(),
        };
        let result = load_moe_ffn_weights(&mut src, &cfg, "layers.0", hidden);
        match result {
            Err(InferenceError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("moe_intermediate_size"),
                    "message should name the field: {msg}"
                );
            }
            Err(e) => panic!("expected InvalidInput, got a different error: {e}"),
            Ok(_) => panic!("expected Err for overflowing moe_intermediate_size, got Ok"),
        }
    }

    /// A tiny, single-layer, dense-FFN, linear-attention-only config used to exercise
    /// `load_weights` end-to-end without a real checkpoint.
    fn tiny_linear_layer_cfg() -> Qwen35Config {
        Qwen35Config {
            hidden_size: 4,
            num_hidden_layers: 1,
            vocab_size: 4,
            intermediate_size: 8,
            linear_num_key_heads: 1,
            linear_key_head_dim: 2,
            linear_num_value_heads: Some(1),
            linear_value_head_dim: 2,
            linear_conv_kernel_dim: 2,
            layer_types: vec![LayerType::LinearAttention],
            layer_mask: vec![true],
            tie_word_embeddings: true,
            ..Qwen35Config::qwen35_2b()
        }
    }

    /// Build a complete, cfg-consistent tensor set for `tiny_linear_layer_cfg()`, with
    /// `input_layernorm`/`post_attention_layernorm` shaped `[hidden]` (or overridden via
    /// `iln_len`/`paln_len` to a malformed length).
    fn tiny_linear_layer_tensors(
        cfg: &Qwen35Config,
        iln_len: usize,
        paln_len: usize,
    ) -> MockTensorSource {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let qkv_dim = cfg.linear_qkv_dim();
        let output_dim = cfg.linear_output_dim();
        let value_heads = cfg.linear_num_value_heads();
        let kernel_size = cfg.linear_conv_kernel_dim;
        let prefix = "model.language_model.layers.0";

        MockTensorSource {
            tensors: [
                (
                    "model.language_model.embed_tokens.weight".to_string(),
                    (vec![0.0f32; vocab * hidden], vec![vocab, hidden]),
                ),
                (
                    "model.language_model.norm.weight".to_string(),
                    (vec![1.0f32; hidden], vec![hidden]),
                ),
                (
                    format!("{prefix}.input_layernorm.weight"),
                    (vec![1.0f32; iln_len], vec![iln_len]),
                ),
                (
                    format!("{prefix}.post_attention_layernorm.weight"),
                    (vec![1.0f32; paln_len], vec![paln_len]),
                ),
                (
                    format!("{prefix}.mlp.gate_proj.weight"),
                    (vec![0.0f32; inter * hidden], vec![inter, hidden]),
                ),
                (
                    format!("{prefix}.mlp.up_proj.weight"),
                    (vec![0.0f32; inter * hidden], vec![inter, hidden]),
                ),
                (
                    format!("{prefix}.mlp.down_proj.weight"),
                    (vec![0.0f32; hidden * inter], vec![hidden, inter]),
                ),
                (
                    format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    (vec![0.0f32; qkv_dim * hidden], vec![qkv_dim, hidden]),
                ),
                (
                    format!("{prefix}.linear_attn.in_proj_z.weight"),
                    (vec![0.0f32; output_dim * hidden], vec![output_dim, hidden]),
                ),
                (
                    format!("{prefix}.linear_attn.in_proj_b.weight"),
                    (
                        vec![0.0f32; value_heads * hidden],
                        vec![value_heads, hidden],
                    ),
                ),
                (
                    format!("{prefix}.linear_attn.in_proj_a.weight"),
                    (
                        vec![0.0f32; value_heads * hidden],
                        vec![value_heads, hidden],
                    ),
                ),
                (
                    format!("{prefix}.linear_attn.A_log"),
                    (vec![0.0f32; value_heads], vec![value_heads]),
                ),
                (
                    format!("{prefix}.linear_attn.dt_bias"),
                    (vec![0.0f32; value_heads], vec![value_heads]),
                ),
                (
                    format!("{prefix}.linear_attn.conv1d.weight"),
                    (
                        vec![0.0f32; qkv_dim * kernel_size],
                        vec![qkv_dim, 1, kernel_size],
                    ),
                ),
                (
                    format!("{prefix}.linear_attn.norm.weight"),
                    (
                        vec![1.0f32; cfg.linear_value_head_dim],
                        vec![cfg.linear_value_head_dim],
                    ),
                ),
                (
                    format!("{prefix}.linear_attn.out_proj.weight"),
                    (vec![0.0f32; hidden * output_dim], vec![hidden, output_dim]),
                ),
            ]
            .into_iter()
            .collect(),
        }
    }

    /// A malformed per-layer `input_layernorm` (length `hidden - 1`, e.g. from a
    /// checkpoint's `config.json` disagreeing with its own tensor) must be rejected by
    /// `load_weights`, not silently truncate: before this guard, the unchecked
    /// `load_owned_tensor` accepted any length and `qwen35_rms_norm` zipped only the
    /// shared prefix, producing wrong logits with no error signal.
    #[test]
    fn load_weights_rejects_malformed_input_layernorm() {
        let cfg = tiny_linear_layer_cfg();
        let hidden = cfg.hidden_size;
        let mut src = tiny_linear_layer_tensors(&cfg, hidden - 1, hidden);
        match load_weights(&mut src, &cfg) {
            Err(InferenceError::ShapeMismatch { name, .. }) => {
                assert!(
                    name.contains("input_layernorm"),
                    "error must name input_layernorm, got: {name}"
                );
            }
            Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
            Ok(_) => panic!("expected Err for malformed input_layernorm, got Ok"),
        }
    }

    /// Same as above for `post_attention_layernorm`.
    #[test]
    fn load_weights_rejects_malformed_post_attention_layernorm() {
        let cfg = tiny_linear_layer_cfg();
        let hidden = cfg.hidden_size;
        let mut src = tiny_linear_layer_tensors(&cfg, hidden, hidden - 1);
        match load_weights(&mut src, &cfg) {
            Err(InferenceError::ShapeMismatch { name, .. }) => {
                assert!(
                    name.contains("post_attention_layernorm"),
                    "error must name post_attention_layernorm, got: {name}"
                );
            }
            Err(e) => panic!("expected ShapeMismatch, got a different error: {e}"),
            Ok(_) => panic!("expected Err for malformed post_attention_layernorm, got Ok"),
        }
    }

    #[test]
    fn load_weights_accepts_correctly_shaped_layernorms() {
        let cfg = tiny_linear_layer_cfg();
        let hidden = cfg.hidden_size;
        let mut src = tiny_linear_layer_tensors(&cfg, hidden, hidden);
        load_weights(&mut src, &cfg).expect("correctly shaped layernorms must load");
    }

    /// A config with `num_attention_heads = 2^63` and `head_dim = 2` overflows
    /// `full_q_dim()`/`full_kv_dim()` in release builds. `load_full_attention_weights`
    /// must reject this via the checked accessors before deriving a wrapped, undersized
    /// expected tensor shape for `q_proj`/`k_proj`/`v_proj`/`o_proj` — a checkpoint that
    /// happens to match the wrapped shape would otherwise load successfully and panic
    /// (or corrupt state) once the forward pass indexes by the original, unwrapped head
    /// count.
    #[test]
    fn load_full_attention_weights_rejects_overflowing_config() {
        let cfg = Qwen35Config {
            num_attention_heads: 1 << 63,
            num_key_value_heads: 1 << 63,
            head_dim: 2,
            ..Qwen35Config::qwen35_2b()
        };
        let mut src = NullTensorSource;

        match load_full_attention_weights(&mut src, &cfg, "layers.0") {
            Err(InferenceError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("overflow"),
                    "error must describe the overflow, got: {msg}"
                );
            }
            Err(e) => panic!("expected InvalidInput, got a different error: {e}"),
            Ok(_) => panic!(
                "expected Err for overflowing full-attention config, got Ok (would derive a \
                 wrapped tensor shape and panic downstream)"
            ),
        }
    }

    /// Same overflow-rejection contract for the GatedDeltaNet dimension helpers used by
    /// `load_weights`: a pathological `linear_num_key_heads` / `linear_key_head_dim` /
    /// `linear_value_head_dim` combination must surface as a typed error at ingress
    /// instead of wrapping into an undersized `qkv_dim`/`output_dim`.
    #[test]
    fn load_weights_rejects_overflowing_linear_attention_config() {
        let cfg = Qwen35Config {
            linear_num_key_heads: 1 << 63,
            linear_num_value_heads: Some(1 << 63),
            linear_key_head_dim: 2,
            linear_value_head_dim: 2,
            ..Qwen35Config::qwen35_2b()
        };
        let mut src = NullTensorSource;

        match load_weights(&mut src, &cfg) {
            Err(InferenceError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("overflow"),
                    "error must describe the overflow, got: {msg}"
                );
            }
            Err(e) => panic!("expected InvalidInput, got a different error: {e}"),
            Ok(_) => panic!(
                "expected Err for overflowing linear-attention config, got Ok (would derive \
                 wrapped dimensions and panic downstream)"
            ),
        }
    }
}
