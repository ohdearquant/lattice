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

use super::gemma4_config::{GEMMA4_EXPECTED_DTYPE, Gemma4Config};
use super::gemma4_weights::{Gemma4LayerWeights, Gemma4Weights};
use crate::error::InferenceError;
use crate::weights::TensorSource;

const LM_PREFIX: &str = "model.language_model.";

/// Runtime replay of the stage-2 preflight's dtype contract
/// ([`GEMMA4_EXPECTED_DTYPE`]) against the tensor this call is about to
/// materialize as f32. `preflight_check` validates dtype from an offline
/// header manifest; nothing previously re-checked it at actual load time,
/// so a same-shaped F32/F16 tensor in a mislabeled or converted checkpoint
/// was silently accepted instead of failing closed. `None` (source cannot
/// report dtype) is not a failure -- see [`TensorSource::tensor_dtype`].
fn check_dtype<T: TensorSource + ?Sized>(source: &mut T, name: &str) -> Result<(), InferenceError> {
    if let Some(dtype) = source.tensor_dtype(name)?
        && dtype != GEMMA4_EXPECTED_DTYPE
    {
        return Err(InferenceError::Inference(format!(
            "gemma4 loading: tensor {name} has dtype {dtype:?}, expected \
             {GEMMA4_EXPECTED_DTYPE:?} -- the pinned Gemma 4 E2B checkpoint's language-model \
             tensors are BF16; a differently-dtyped tensor at this name means a wrong or \
             converted checkpoint, not a variant this loader supports"
        )));
    }
    Ok(())
}

fn load_tensor<T: TensorSource + ?Sized>(
    source: &mut T,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, InferenceError> {
    check_dtype(source, name)?;
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

#[cfg(test)]
mod tests {
    use super::super::gemma4_config::Gemma4LayerType;
    use super::*;
    use std::collections::HashMap;

    /// In-memory [`TensorSource`] carrying an explicit per-tensor dtype
    /// label, so tests can exercise [`check_dtype`]'s wiring through
    /// `load_weights` without a real safetensors file on disk.
    struct MockDtypeSource {
        tensors: HashMap<String, (Vec<f32>, Vec<usize>, String)>,
    }

    impl TensorSource for MockDtypeSource {
        fn has_tensor(&mut self, name: &str) -> Result<bool, InferenceError> {
            Ok(self.tensors.contains_key(name))
        }
        fn tensor_shape(&mut self, name: &str) -> Result<Option<Vec<usize>>, InferenceError> {
            Ok(self.tensors.get(name).map(|(_, s, _)| s.clone()))
        }
        fn tensor_dtype(&mut self, name: &str) -> Result<Option<String>, InferenceError> {
            Ok(self.tensors.get(name).map(|(_, _, d)| d.clone()))
        }
        fn get_f32_tensor_owned(
            &mut self,
            name: &str,
        ) -> Result<(Vec<f32>, Vec<usize>), InferenceError> {
            self.tensors
                .get(name)
                .map(|(d, s, _)| (d.clone(), s.clone()))
                .ok_or_else(|| InferenceError::MissingTensor(name.to_string()))
        }
    }

    /// Same tiny 6-layer geometry as `gemma4_cache::tests::tiny_config`
    /// (layers 4, 5 KV-shared) -- cheap enough to hand-populate every
    /// tensor `load_weights` requires.
    fn tiny_config() -> Gemma4Config {
        let layer_types = vec![
            Gemma4LayerType::SlidingAttention,
            Gemma4LayerType::FullAttention,
            Gemma4LayerType::SlidingAttention,
            Gemma4LayerType::SlidingAttention,
            Gemma4LayerType::FullAttention,
            Gemma4LayerType::SlidingAttention,
        ];
        Gemma4Config {
            hidden_size: 8,
            num_hidden_layers: 6,
            vocab_size: 32,
            intermediate_size: 16,
            rms_norm_eps: 1e-6,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            global_head_dim: 8,
            sliding_window: 4096,
            attention_k_eq_v: false,
            attention_bias: false,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            partial_rotary_factor: 0.5,
            layer_types,
            num_kv_shared_layers: 2,
            use_double_wide_mlp_raw: true,
            hidden_size_per_layer_input: 4,
            hidden_activation: "gelu_pytorch_tanh".to_string(),
            final_logit_softcapping: 30.0,
            tie_word_embeddings: true,
            eos_token_id: 1,
            max_position_embeddings: 4096,
        }
    }

    /// Populate every tensor `load_weights` reads for `cfg`, all declared
    /// BF16 (the correct contract), zero-valued.
    fn full_tensor_set(cfg: &Gemma4Config) -> HashMap<String, (Vec<f32>, Vec<usize>, String)> {
        let hidden = cfg.hidden_size;
        let per_layer_dim = cfg.hidden_size_per_layer_input;
        let ple_packed_dim = cfg.num_hidden_layers * per_layer_dim;
        let mut m = HashMap::new();
        let mut put = |name: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            m.insert(
                name,
                (vec![0f32; n], shape, GEMMA4_EXPECTED_DTYPE.to_string()),
            );
        };
        put(
            format!("{LM_PREFIX}embed_tokens.weight"),
            vec![cfg.vocab_size, hidden],
        );
        put(
            format!("{LM_PREFIX}embed_tokens_per_layer.weight"),
            vec![cfg.vocab_size, ple_packed_dim],
        );
        put(format!("{LM_PREFIX}norm.weight"), vec![hidden]);
        put(
            format!("{LM_PREFIX}per_layer_model_projection.weight"),
            vec![ple_packed_dim, hidden],
        );
        put(
            format!("{LM_PREFIX}per_layer_projection_norm.weight"),
            vec![per_layer_dim],
        );
        for layer in 0..cfg.num_hidden_layers {
            let prefix = format!("{LM_PREFIX}layers.{layer}.");
            let head_w = cfg.attn_head_dim(layer);
            let q_dim = cfg.num_attention_heads * head_w;
            let kv_dim = cfg.num_key_value_heads * head_w;
            let mlp_dim = cfg.mlp_intermediate_size(layer);
            put(format!("{prefix}input_layernorm.weight"), vec![hidden]);
            put(
                format!("{prefix}post_attention_layernorm.weight"),
                vec![hidden],
            );
            put(
                format!("{prefix}pre_feedforward_layernorm.weight"),
                vec![hidden],
            );
            put(
                format!("{prefix}post_feedforward_layernorm.weight"),
                vec![hidden],
            );
            put(
                format!("{prefix}post_per_layer_input_norm.weight"),
                vec![hidden],
            );
            put(format!("{prefix}layer_scalar"), vec![1]);
            put(
                format!("{prefix}per_layer_input_gate.weight"),
                vec![per_layer_dim, hidden],
            );
            put(
                format!("{prefix}per_layer_projection.weight"),
                vec![hidden, per_layer_dim],
            );
            put(
                format!("{prefix}self_attn.q_proj.weight"),
                vec![q_dim, hidden],
            );
            put(
                format!("{prefix}self_attn.o_proj.weight"),
                vec![hidden, q_dim],
            );
            put(format!("{prefix}self_attn.q_norm.weight"), vec![head_w]);
            if !cfg.is_kv_shared_layer(layer) {
                put(
                    format!("{prefix}self_attn.k_proj.weight"),
                    vec![kv_dim, hidden],
                );
                put(
                    format!("{prefix}self_attn.v_proj.weight"),
                    vec![kv_dim, hidden],
                );
                put(format!("{prefix}self_attn.k_norm.weight"), vec![head_w]);
            }
            put(
                format!("{prefix}mlp.gate_proj.weight"),
                vec![mlp_dim, hidden],
            );
            put(format!("{prefix}mlp.up_proj.weight"), vec![mlp_dim, hidden]);
            put(
                format!("{prefix}mlp.down_proj.weight"),
                vec![hidden, mlp_dim],
            );
        }
        m
    }

    #[test]
    fn well_formed_bf16_checkpoint_loads() {
        let cfg = tiny_config();
        let mut source = MockDtypeSource {
            tensors: full_tensor_set(&cfg),
        };
        load_weights(&mut source, &cfg).expect("an all-BF16 tensor set must load");
    }

    #[test]
    fn wrong_dtype_required_tensor_fails_closed_naming_tensor() {
        let cfg = tiny_config();
        let mut tensors = full_tensor_set(&cfg);
        let mutated_name = format!("{LM_PREFIX}embed_tokens.weight");
        let entry = tensors.get_mut(&mutated_name).unwrap();
        entry.2 = "F32".to_string();
        let mut source = MockDtypeSource { tensors };

        // `Gemma4Weights` has no `Debug` impl (deliberately -- it holds
        // multi-megabyte weight buffers), so `expect_err` can't be used here.
        let Err(err) = load_weights(&mut source, &cfg) else {
            panic!("a wrong-dtype required tensor must fail closed, not silently load");
        };
        let msg = err.to_string();
        assert!(
            msg.contains(&mutated_name),
            "error must name the tensor: {msg}"
        );
        assert!(
            msg.contains("F32"),
            "error must name the observed dtype: {msg}"
        );
        assert!(
            msg.contains(GEMMA4_EXPECTED_DTYPE),
            "error must name the expected dtype: {msg}"
        );
    }
}
