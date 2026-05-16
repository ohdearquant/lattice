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

        let config_path = path.join("config.json");
        let config = if config_path.exists() {
            Qwen35Config::from_config_json(&config_path)?
        } else {
            Qwen35Config::qwen35_2b()
        };

        validate_required_tensor_names(source.as_mut(), &config)?;

        let weights = load_weights(source.as_mut(), &config)?;

        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path)?;

        let rope_dim = config.rope_dim();
        let rope_max = config.max_position_embeddings.min(8192);
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
