//! Qwen3.5 model type, safetensors loader, accessors, LoRA setter, train-backward weight accessors, and layer weight statistics.
use super::loading::{load_weights, validate_required_tensor_names};
use super::weights::{AttentionWeights, FeedForwardWeights, ModelWeights};
use crate::error::InferenceError;
use crate::model::qwen35_config::Qwen35Config;
use crate::rope::RopeTable;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::weights::{SafetensorsFile, ShardedSafetensors, TensorSource};
use std::path::Path;

/// **Unstable**: Qwen3.5-2B text generation model; Metal GPU path under development.
pub struct Qwen35Model {
    pub(crate) config: Qwen35Config,
    pub(crate) weights: ModelWeights,
    pub(crate) tokenizer: BpeTokenizer,
    pub(crate) rope: RopeTable,
    /// Optional LoRA adapter hook. Default: NoopLoraHook (no adapter).
    pub(crate) lora: Box<dyn crate::lora_hook::LoraHook>,
}

impl Qwen35Model {
    /// **Unstable**: load Qwen3.5-2B or Qwen3.6 from a local safetensors directory.
    pub fn from_safetensors(path: &Path) -> Result<Self, InferenceError> {
        let model_path = path.join("model.safetensors");
        let index_path = path.join("model.safetensors.index.json");
        let mut source: Box<dyn TensorSource> = if model_path.exists() {
            Box::new(SafetensorsFile::open(&model_path)?)
        } else if index_path.exists() {
            Box::new(ShardedSafetensors::open_index(&index_path)?)
        } else {
            return Err(InferenceError::ModelNotFound(format!(
                "missing model.safetensors or model.safetensors.index.json in {}",
                path.display()
            )));
        };

        let config = Qwen35Config::from_model_dir(path)?;

        validate_required_tensor_names(source.as_mut(), &config)?;

        let weights = load_weights(source.as_mut(), &config)?;

        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path)?;

        let rope_dim = config.rope_dim();
        let rope_max = config.max_position_embeddings.min(8192);
        // Qwen3.5 uses `theta^(-2i/rope_dim)` (not head_dim) — empirically
        // verified by PPL regression when using head_dim. The partial-RoPE
        // frequency spectrum is compressed into the first `rope_dim` dimensions
        // rather than sliced from a head_dim-wide spectrum.
        let rope = RopeTable::new(rope_dim, rope_max, config.rope_theta);

        Ok(Self {
            config,
            weights,
            tokenizer,
            rope,
            lora: Box::new(crate::lora_hook::NoopLoraHook),
        })
    }

    /// **Unstable**: attach a LoRA adapter hook; hook trait API may change.
    pub fn set_lora(&mut self, hook: Box<dyn crate::lora_hook::LoraHook>) {
        self.lora = hook;
    }

    /// **Unstable**: access raw model weights.
    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }

    /// **Unstable**: access Qwen3.5 configuration.
    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }

    /// **Unstable**: mutable access to the post-load Qwen3.5 configuration.
    ///
    /// Additive builder-style setter (`Qwen35Model.config` is a private
    /// field; this is the only way an external caller can adjust it after
    /// load). Intended for benchmark/tooling call sites that need to change
    /// generation behavior without adding a new field to the
    /// request-scoped [`GenerateConfig`](crate::model::qwen35_config::GenerateConfig)
    /// (which is a plain public-literal struct with a `Default` impl --
    /// `cargo-semver-checks` treats any new field there as
    /// `constructible_struct_adds_field`, a semver-major break).
    ///
    /// Established idiom already used throughout this crate's own test
    /// suite (e.g. `cfg.eos_token_id = u32::MAX`) to push the model's
    /// configured EOS token id out of the reachable vocab range, which
    /// `should_stop_token` (the single shared stop predicate every
    /// CPU/Metal decode loop calls) then never matches -- see
    /// `qwen35_generate --emit-phase-events` (PR #882), which combines this
    /// with `GenerateConfig::stop_token_ids: vec![]` (already public) to
    /// force continuation to the exact requested token count for a
    /// benchmark trial.
    pub fn config_mut(&mut self) -> &mut Qwen35Config {
        &mut self.config
    }

    /// **Unstable**: access the BPE tokenizer.
    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }

    /// **Unstable**: maximum sequence length the precomputed RoPE table can
    /// serve. `forward_step(token, position, ..)` panics if `position` is at
    /// or above this value, so callers driving long-context evaluation must
    /// keep their per-window position strictly below it.
    pub fn max_context(&self) -> usize {
        self.rope.max_positions()
    }

    /// **Unstable**: access raw embedding weights for debugging.
    pub fn embed_weights(&self) -> &[f32] {
        &self.weights.embed_tokens
    }

    /// **Unstable (train-backward)**: Return weight slices for a GQA layer that the
    /// backward trainer needs to compute gradients.
    ///
    /// Returns `None` if the requested layer is not a full-attention (GQA) layer.
    /// All returned slices are views into the model weight storage; their lifetime
    /// is tied to `&self`.
    ///
    /// Shapes for Qwen3.5-0.8B (24 layers, 8 Q-heads, 2 KV-heads, head_dim=256):
    ///   lm_head:  [vocab, hidden]
    ///   final_norm: [hidden]
    ///   embed: [vocab, hidden]
    ///   q_proj: [2*q_dim, hidden]
    ///   k_proj: [kv_dim, hidden]
    ///   v_proj: [kv_dim, hidden]
    ///   o_proj: [hidden, q_dim]
    ///   q_norm: [head_dim]
    ///   k_norm: [head_dim]
    ///   pre_attn_norm: [hidden]  (input_layernorm)
    ///   post_attn_norm: [hidden]  (post_attention_layernorm)
    ///   gate_proj: [inter, hidden]
    ///   up_proj: [inter, hidden]
    ///   down_proj: [hidden, inter]
    #[cfg(feature = "train-backward")]
    #[allow(clippy::type_complexity)]
    pub fn gqa_layer_weights(
        &self,
        layer: usize,
    ) -> Option<(
        &[f32], // q_proj
        &[f32], // k_proj
        &[f32], // v_proj
        &[f32], // o_proj
        &[f32], // q_norm
        &[f32], // k_norm
        &[f32], // pre_attn_norm
        &[f32], // post_attn_norm
        &[f32], // gate_proj
        &[f32], // up_proj
        &[f32], // down_proj
    )> {
        let (attn, common) = &self.weights.layers[layer];
        let full = match attn {
            AttentionWeights::Full(w) => w,
            AttentionWeights::Linear(_) => return None,
        };
        let dense = match &common.ffn {
            FeedForwardWeights::Dense(d) => d,
            FeedForwardWeights::Moe(_) => return None,
        };
        Some((
            &full.q_proj,
            &full.k_proj,
            &full.v_proj,
            &full.o_proj,
            &full.q_norm,
            &full.k_norm,
            &common.input_layernorm,
            &common.post_attention_layernorm,
            &dense.gate_proj,
            &dense.up_proj,
            &dense.down_proj,
        ))
    }

    /// **Unstable (train-backward)**: Return GDN (linear-attention) layer weights
    /// plus the layer's norm + Dense-FFN weights.
    ///
    /// Returns `None` if the layer is not a GatedDeltaNet layer or its FFN is not
    /// Dense. Tuple is `(gdn_mixer, input_layernorm, post_attention_layernorm,
    /// gate_proj, up_proj, down_proj)` — the GDN analogue of [`Self::gqa_layer_weights`].
    /// `gdn_mixer` is the frozen linear-attention block; `input_layernorm` is the
    /// pre-mixer shifted RMSNorm gamma. Used by the GDN differential test and the
    /// full-depth backward tape (`gdn_backward` for dx through frozen GDN layers,
    /// plus the layer's own FFN block).
    #[cfg(feature = "train-backward")]
    #[allow(clippy::type_complexity)]
    pub fn gdn_layer_weights(
        &self,
        layer: usize,
    ) -> Option<(
        &crate::attention::gdn::GatedDeltaNetWeights,
        &[f32], // input_layernorm
        &[f32], // post_attention_layernorm
        &[f32], // gate_proj
        &[f32], // up_proj
        &[f32], // down_proj
    )> {
        let (attn, common) = &self.weights.layers[layer];
        let gdn = match attn {
            AttentionWeights::Linear(w) => w,
            AttentionWeights::Full(_) => return None,
        };
        let dense = match &common.ffn {
            FeedForwardWeights::Dense(d) => d,
            FeedForwardWeights::Moe(_) => return None,
        };
        Some((
            gdn,
            &common.input_layernorm,
            &common.post_attention_layernorm,
            &dense.gate_proj,
            &dense.up_proj,
            &dense.down_proj,
        ))
    }

    /// **Unstable (train-backward)**: Return lm_head, final_norm, and embed slices.
    #[cfg(feature = "train-backward")]
    pub fn head_weights(&self) -> (&[f32], &[f32], &[f32]) {
        (
            self.weights.logits_weight(),
            &self.weights.final_norm,
            &self.weights.embed_tokens,
        )
    }

    /// **Unstable**: diagnostic weight statistics for a layer.
    pub fn layer_weight_stats(&self, layer: usize) -> Vec<(String, f32, f32)> {
        fn stats(name: &str, data: &[f32]) -> (String, f32, f32) {
            let n = data.len() as f32;
            let mean = data.iter().sum::<f32>() / n;
            let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
            (name.to_string(), mean, std)
        }
        let (attn, common) = &self.weights.layers[layer];
        let mut v = vec![stats("input_layernorm", &common.input_layernorm)];
        match &common.ffn {
            FeedForwardWeights::Dense(dense) => {
                v.push(stats("gate_proj", &dense.gate_proj));
            }
            FeedForwardWeights::Moe(moe) => {
                v.push(stats("router_gate", &moe.router.gate));
            }
        }
        match attn {
            AttentionWeights::Linear(w) => {
                v.push(stats("in_proj_qkv", &w.in_proj_qkv));
                v.push(stats("in_proj_z", &w.in_proj_z));
                v.push(stats("a_log", &w.a_log));
                v.push(stats("dt_bias", &w.dt_bias));
                v.push(stats("conv1d", &w.conv1d_weight));
                v.push(stats("norm", &w.norm_weight));
            }
            AttentionWeights::Full(w) => {
                v.push(stats("q_proj", &w.q_proj));
                v.push(stats("k_proj", &w.k_proj));
                v.push(stats("o_proj", &w.o_proj));
            }
        }
        v
    }
}
