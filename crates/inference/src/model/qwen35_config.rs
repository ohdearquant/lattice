//! Configuration for Qwen hybrid attention models (3.5-2B and 3.6-35B-A3B).
//!
//! Qwen3.5-2B uses a hybrid architecture with two attention mechanisms:
//! - **GatedDeltaNet** (linear attention): 18 layers with recurrent state
//! - **Full GQA** (standard attention): 6 layers with KV cache
//!
//! Qwen3.6-35B-A3B extends this with a Mixture-of-Experts (MoE) FFN:
//! 256 routed experts, top-8 selection, 1 shared expert, separate lm_head.

use crate::error::InferenceError;
use std::path::Path;

/// Chat turn end token for Qwen models.
pub const QWEN_CHAT_IM_END_TOKEN_ID: u32 = 248_046;

/// Token IDs for Qwen3.6 thinking mode (248K vocab tokenizer).
pub const QWEN3_THINK_OPEN_TOKEN_ID: u32 = 248_068;
pub const QWEN3_THINK_CLOSE_TOKEN_ID: u32 = 248_069;
pub const QWEN3_NEWLINE_TOKEN_ID: u32 = 198;

/// Empty think block token sequence: `<think>\n\n</think>\n\n`.
/// Prefill this to disable chain-of-thought reasoning.
pub const QWEN3_NO_THINK_PREFIX: [u32; 6] = [
    QWEN3_THINK_OPEN_TOKEN_ID,
    QWEN3_NEWLINE_TOKEN_ID,
    QWEN3_NEWLINE_TOKEN_ID,
    QWEN3_THINK_CLOSE_TOKEN_ID,
    QWEN3_NEWLINE_TOKEN_ID,
    QWEN3_NEWLINE_TOKEN_ID,
];

/// **Unstable**: per-layer attention type selector for Qwen hybrid architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    /// GatedDeltaNet linear attention with recurrent state.
    LinearAttention,
    /// Standard grouped-query attention with KV cache.
    FullAttention,
}

/// **Unstable**: Qwen hybrid attention model configuration; fields evolving with model variants.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct Qwen35Config {
    // --- Core dimensions ---
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,

    // --- Full attention config (GQA) ---
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    /// Fraction of head_dim that gets RoPE applied (0.25 = first 64 of 256 dims).
    pub partial_rotary_factor: f32,
    /// Nested RoPE config (27B+ models store rope_theta here instead of flat).
    #[serde(default)]
    pub rope_parameters: Option<RopeParams>,

    // --- Linear attention config (GatedDeltaNet) ---
    pub linear_num_key_heads: usize,
    /// None means use the method `linear_num_value_heads()` default.
    #[serde(default)]
    pub linear_num_value_heads: Option<usize>,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,

    // --- MoE config ---
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<usize>,
    #[serde(default)]
    pub output_router_logits: bool,
    #[serde(default)]
    pub router_aux_loss_coef: Option<f32>,

    // --- Embedding/output projection ---
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    // --- MTP (Multi-Token Prediction) config ---
    /// Number of MTP transformer layers (0 = no MTP).
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    /// Whether the MTP module uses dedicated embeddings separate from the main model.
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,

    // --- Layer pattern ---
    /// Every Nth layer is full attention (4 = [lin, lin, lin, full]).
    pub full_attention_interval: usize,
    /// Precomputed per-layer type, length = num_hidden_layers.
    pub layer_types: Vec<LayerType>,
    /// Per-layer active mask; `true` = active, `false` = pruned (identity skip).
    /// Length must equal `num_hidden_layers`. Defaults to all-true (no pruning).
    #[serde(default)]
    pub layer_mask: Vec<bool>,

    // --- Generation ---
    pub eos_token_id: u32,
    pub max_position_embeddings: usize,
}

fn default_tie_word_embeddings() -> bool {
    true
}

// Nested rope_parameters in HF config.json (27B+ models nest rope_theta here).
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct RopeParams {
    #[serde(default)]
    pub rope_theta: f64,
}

// Private helper for HF config.json structure (outer wrapper with text_config).
#[derive(Debug, serde::Deserialize)]
struct HfQwenConfigFile {
    #[serde(default)]
    text_config: Option<Qwen35Config>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
}

impl Default for Qwen35Config {
    fn default() -> Self {
        Self::qwen36_35b_a3b()
    }
}

impl Qwen35Config {
    /// **Unstable**: default Qwen3.5-2B configuration; may change as model checkpoints update.
    pub fn qwen35_2b() -> Self {
        let num_hidden_layers = 24;
        let full_attention_interval = 4;
        let layer_types = compute_layer_types(num_hidden_layers, full_attention_interval);

        Self {
            hidden_size: 2048,
            num_hidden_layers,
            vocab_size: 248_320,
            intermediate_size: 6144,
            rms_norm_eps: 1e-6,
            // Full attention
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            // Linear attention (GatedDeltaNet)
            linear_num_key_heads: 16,
            linear_num_value_heads: Some(16),
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            // MoE (absent in Qwen3.5)
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            // Projection
            tie_word_embeddings: true,
            // Layer pattern
            full_attention_interval,
            layer_types,
            layer_mask: vec![true; num_hidden_layers],
            // Generation
            eos_token_id: 248_044,
            max_position_embeddings: 262_144,
            // MTP (absent in Qwen3.5)
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
        }
    }

    /// **Unstable**: Qwen3.5-0.8B configuration. Same hybrid architecture as the
    /// 2B (24 layers, `[linear, linear, linear, full] x 6`), scaled down. The
    /// released checkpoint is a vision-language model; this is its text decoder.
    pub fn qwen35_0_8b() -> Self {
        let num_hidden_layers = 24;
        let full_attention_interval = 4;
        let layer_types = compute_layer_types(num_hidden_layers, full_attention_interval);

        Self {
            hidden_size: 1024,
            num_hidden_layers,
            vocab_size: 248_320,
            intermediate_size: 3584,
            rms_norm_eps: 1e-6,
            // Full attention
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            // Linear attention (GatedDeltaNet)
            linear_num_key_heads: 16,
            linear_num_value_heads: Some(16),
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            // MoE (absent — 0.8B is dense)
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            // Projection
            tie_word_embeddings: true,
            // Layer pattern
            full_attention_interval,
            layer_types,
            layer_mask: vec![true; num_hidden_layers],
            // Generation
            eos_token_id: 248_044,
            max_position_embeddings: 262_144,
            // MTP: Qwen3.5-0.8B ships 1 MTP layer
            mtp_num_hidden_layers: 1,
            mtp_use_dedicated_embeddings: false,
        }
    }

    /// **Unstable**: Qwen3.6-35B-A3B text configuration defaults from HF `text_config`.
    pub fn qwen36_35b_a3b() -> Self {
        let num_hidden_layers = 40;
        let full_attention_interval = 4;
        let layer_types = compute_layer_types(num_hidden_layers, full_attention_interval);

        Self {
            hidden_size: 2048,
            num_hidden_layers,
            vocab_size: 248_320,
            intermediate_size: 6144,
            rms_norm_eps: 1e-6,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 256,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            linear_num_key_heads: 16,
            linear_num_value_heads: Some(32),
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            num_experts: Some(256),
            num_experts_per_tok: Some(8),
            moe_intermediate_size: Some(512),
            shared_expert_intermediate_size: Some(512),
            output_router_logits: false,
            router_aux_loss_coef: Some(0.001),
            tie_word_embeddings: false,
            full_attention_interval,
            layer_types,
            layer_mask: vec![true; num_hidden_layers],
            eos_token_id: 248_044,
            max_position_embeddings: 262_144,
            // MTP: Qwen3.6 has 1 MTP layer
            mtp_num_hidden_layers: 1,
            mtp_use_dedicated_embeddings: false,
        }
    }

    /// **Unstable**: Qwen3.6-27B dense configuration; fields from HF `text_config`.
    pub fn qwen36_27b() -> Self {
        let num_hidden_layers = 64;
        let full_attention_interval = 4;
        let layer_types = compute_layer_types(num_hidden_layers, full_attention_interval);

        Self {
            hidden_size: 5120,
            num_hidden_layers,
            vocab_size: 248_320,
            intermediate_size: 17408,
            rms_norm_eps: 1e-6,
            // Full attention (GA)
            num_attention_heads: 24,
            num_key_value_heads: 4,
            head_dim: 256,
            // rope_theta is nested under rope_parameters.rope_theta in config.json and not
            // directly deserializable; hardcode the value from rope_parameters.rope_theta.
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            // Linear attention (GatedDeltaNet)
            linear_num_key_heads: 16,
            linear_num_value_heads: Some(48),
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            // MoE (absent — 27B is dense)
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            // Projection
            tie_word_embeddings: false,
            // Layer pattern
            full_attention_interval,
            layer_types,
            layer_mask: vec![true; num_hidden_layers],
            // Generation
            eos_token_id: 248_044,
            max_position_embeddings: 262_144,
            // MTP: Qwen3.6 has 1 MTP layer
            mtp_num_hidden_layers: 1,
            mtp_use_dedicated_embeddings: false,
        }
    }

    /// Parse a HF config.json (which may wrap fields inside `text_config`).
    pub fn from_config_json(path: &Path) -> Result<Self, InferenceError> {
        let json = std::fs::read_to_string(path).map_err(InferenceError::Io)?;
        Self::from_config_json_str(&json)
    }

    /// Parse HF config.json text into a `Qwen35Config`.
    pub fn from_config_json_str(json: &str) -> Result<Self, InferenceError> {
        let parsed: HfQwenConfigFile = serde_json::from_str(json)
            .map_err(|e| InferenceError::Inference(format!("invalid Qwen config.json: {e}")))?;
        let mut cfg = parsed
            .text_config
            .unwrap_or_else(Qwen35Config::qwen36_35b_a3b);

        if let Some(tie) = parsed.tie_word_embeddings {
            cfg.tie_word_embeddings = tie;
        }
        // 27B+ models nest rope_theta under rope_parameters — extract if flat field is 0.
        if cfg.rope_theta == 0.0 {
            if let Some(rp) = &cfg.rope_parameters {
                if rp.rope_theta > 0.0 {
                    cfg.rope_theta = rp.rope_theta;
                }
            }
        }
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            cfg.layer_types =
                compute_layer_types(cfg.num_hidden_layers, cfg.full_attention_interval);
        }
        cfg.normalize_layer_mask();

        Ok(cfg)
    }

    /// Resolved linear value head count (falls back to 32 for Qwen3.6 if unset).
    pub fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads.unwrap_or(32)
    }

    /// Returns true when the model uses Mixture-of-Experts FFN layers.
    pub fn is_moe(&self) -> bool {
        self.num_experts.is_some()
            || self.num_experts_per_tok.is_some()
            || self.moe_intermediate_size.is_some()
            || self.shared_expert_intermediate_size.is_some()
    }

    /// Resolved MoE routed expert intermediate size (falls back to `intermediate_size`).
    pub fn moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_size.unwrap_or(self.intermediate_size)
    }

    /// Resolved shared expert intermediate size.
    pub fn shared_expert_intermediate_size(&self) -> usize {
        self.shared_expert_intermediate_size
            .unwrap_or_else(|| self.moe_intermediate_size())
    }

    /// **Unstable**: count of full-attention layers in the hybrid stack.
    pub fn num_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count()
    }

    /// **Unstable**: count of GatedDeltaNet linear-attention layers.
    pub fn num_linear_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count()
    }

    /// **Unstable**: Q projection dimension for full-attention layers.
    pub fn full_q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// **Unstable**: KV projection dimension for full-attention layers.
    pub fn full_kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// **Unstable**: number of RoPE dimensions (partial rotary factor applied).
    pub fn rope_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor) as usize
    }

    /// **Unstable**: total QKV projection output size for GatedDeltaNet layers.
    pub fn linear_qkv_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim   // Q
        + self.linear_num_key_heads * self.linear_key_head_dim // K
        + self.linear_num_value_heads() * self.linear_value_head_dim // V
    }

    /// **Unstable**: output dimension for GatedDeltaNet layers.
    pub fn linear_output_dim(&self) -> usize {
        self.linear_num_value_heads() * self.linear_value_head_dim
    }

    /// **Unstable**: returns true when layer `i` uses full GQA attention.
    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerType::LinearAttention)
            == LayerType::FullAttention
    }

    /// Normalizes `layer_mask` to `num_hidden_layers` all-true entries if length mismatches.
    fn normalize_layer_mask(&mut self) {
        if self.layer_mask.len() != self.num_hidden_layers {
            self.layer_mask = vec![true; self.num_hidden_layers];
        }
    }

    /// Returns true if layer `layer_idx` is active (not pruned).
    pub fn is_layer_active(&self, layer_idx: usize) -> bool {
        self.layer_mask.get(layer_idx).copied().unwrap_or(true)
    }

    /// Count of active (non-pruned) layers.
    pub fn num_active_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.is_layer_active(i))
            .count()
    }

    /// Count of active GatedDeltaNet linear-attention layers.
    pub fn num_active_linear_attention_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.is_layer_active(i) && !self.is_full_attention(i))
            .count()
    }

    /// Count of active full-attention (GQA) layers.
    pub fn num_active_full_attention_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&i| self.is_layer_active(i) && self.is_full_attention(i))
            .count()
    }

    /// Applies `mask` as the layer pruning mask. Panics if length or all-false.
    ///
    /// All built-in constructors (`qwen35_2b`, `qwen36_35b_a3b`, `qwen36_27b`) produce
    /// an all-true mask.  To enable pruning, call this method or [`Self::pruned_config`]
    /// after construction.
    pub fn apply_layer_mask(&mut self, mask: Vec<bool>) {
        assert_eq!(
            mask.len(),
            self.num_hidden_layers,
            "layer_mask length {} does not match num_hidden_layers {}",
            mask.len(),
            self.num_hidden_layers
        );
        assert!(
            mask.iter().any(|&active| active),
            "layer_mask must keep at least one active layer"
        );
        self.layer_mask = mask;
    }

    /// Returns a clone with `mask` applied as the pruning mask.
    pub fn pruned_config(&self, mask: Vec<bool>) -> Self {
        let mut cfg = self.clone();
        cfg.apply_layer_mask(mask);
        cfg
    }
}

/// **Unstable**: sampling configuration for text generation; temperature/top-k/top-p may expand.
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// Random seed for sampling. `None` = seed from system time.
    pub seed: Option<u64>,
    /// Additional stop token IDs (beyond EOS). Generation stops on any of these.
    pub stop_token_ids: Vec<u32>,
    /// When false, caller prepends `QWEN3_NO_THINK_SYSTEM_MSG` to disable chain-of-thought.
    pub enable_thinking: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            seed: None,
            stop_token_ids: vec![QWEN_CHAT_IM_END_TOKEN_ID],
            enable_thinking: true,
        }
    }
}

/// **Unstable**: text generation output struct; fields may expand with streaming support.
#[derive(Debug, Clone)]
pub struct GenerateOutput {
    /// Generated text (excluding prompt).
    pub text: String,
    /// Generated token IDs.
    pub token_ids: Vec<u32>,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    /// Total tokens generated (excluding prompt).
    pub generated_tokens: usize,
}

/// Compute the layer type pattern: every `interval`-th layer (1-indexed) is full attention.
/// For interval=4: layers 3, 7, 11, 15, 19, 23 are full (0-indexed).
pub(crate) fn compute_layer_types(num_layers: usize, interval: usize) -> Vec<LayerType> {
    (0..num_layers)
        .map(|i| {
            if (i + 1) % interval == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_construction_and_layer_types() {
        let cfg = Qwen35Config::qwen35_2b();

        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.vocab_size, 248_320);
        assert_eq!(cfg.layer_types.len(), 24);

        // Check the pattern: [lin, lin, lin, full] x 6
        assert_eq!(cfg.num_full_attention_layers(), 6);
        assert_eq!(cfg.num_linear_attention_layers(), 18);

        // Full attention at indices 3, 7, 11, 15, 19, 23
        let full_indices: Vec<usize> = cfg
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::FullAttention)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(full_indices, vec![3, 7, 11, 15, 19, 23]);

        // Linear attention at all other indices
        for i in [
            0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
        ] {
            assert_eq!(cfg.layer_types[i], LayerType::LinearAttention);
        }
    }

    #[test]
    fn test_generate_config_defaults() {
        let gen_cfg = GenerateConfig::default();
        assert_eq!(gen_cfg.max_new_tokens, 256);
        assert!((gen_cfg.temperature - 0.7).abs() < 1e-6);
        assert_eq!(gen_cfg.top_k, 50);
        assert!((gen_cfg.top_p - 0.9).abs() < 1e-6);
        assert!((gen_cfg.repetition_penalty - 1.1).abs() < 1e-6);
        assert!(
            gen_cfg.stop_token_ids.contains(&QWEN_CHAT_IM_END_TOKEN_ID),
            "default stop tokens must include im_end"
        );
    }

    #[test]
    fn test_dimension_helpers() {
        let cfg = Qwen35Config::qwen35_2b();

        // Full attention dims
        assert_eq!(cfg.full_q_dim(), 8 * 256); // 2048
        assert_eq!(cfg.full_kv_dim(), 2 * 256); // 512
        assert_eq!(cfg.rope_dim(), 64); // 0.25 * 256

        // Linear attention dims (Qwen3.5-2B: 16 value heads)
        // Q: 16*128=2048, K: 16*128=2048, V: 16*128=2048 → total 6144
        assert_eq!(cfg.linear_qkv_dim(), 6144);
        assert_eq!(cfg.linear_output_dim(), 2048); // 16 * 128
    }

    #[test]
    fn test_is_full_attention() {
        let cfg = Qwen35Config::qwen35_2b();
        assert!(!cfg.is_full_attention(0));
        assert!(!cfg.is_full_attention(1));
        assert!(!cfg.is_full_attention(2));
        assert!(cfg.is_full_attention(3));
        assert!(!cfg.is_full_attention(4));
        assert!(cfg.is_full_attention(7));
        assert!(cfg.is_full_attention(23));
        // Out of bounds returns false (linear)
        assert!(!cfg.is_full_attention(100));
    }

    #[test]
    fn test_qwen36_hf_config_fixture_parse_fields() {
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen36_config.json"
        ));

        let cfg = Qwen35Config::from_config_json_str(json).expect("Qwen3.6 HF config parses");

        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.linear_num_value_heads(), 32);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.num_experts, Some(256));
        assert_eq!(cfg.num_experts_per_tok, Some(8));
        assert_eq!(cfg.moe_intermediate_size, Some(512));
        assert_eq!(cfg.shared_expert_intermediate_size, Some(512));
        assert!(!cfg.tie_word_embeddings);
        assert!(cfg.is_moe());

        // 12 additional field assertions.
        assert_eq!(cfg.vocab_size, 248320);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.intermediate_size, 6144);
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_value_head_dim, 128);
        assert_eq!(cfg.linear_conv_kernel_dim, 4);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(cfg.eos_token_id, 248044_u32);
        assert_eq!(cfg.max_position_embeddings, 262144);
        assert_eq!(cfg.rope_theta, 10_000_000.0_f64);
        assert_eq!(cfg.partial_rotary_factor, 0.25_f32);

        let full_indices: Vec<usize> = cfg
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::FullAttention)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(full_indices, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);
    }

    #[test]
    fn test_qwen36_27b_preset_dimensions() {
        let cfg = Qwen35Config::qwen36_27b();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_hidden_layers, 64);
        assert_eq!(cfg.vocab_size, 248_320);
        assert_eq!(cfg.intermediate_size, 17408);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        assert!((cfg.rope_theta - 10_000_000.0_f64).abs() < 1.0);
        assert!((cfg.partial_rotary_factor - 0.25_f32).abs() < 1e-6);
        assert_eq!(cfg.eos_token_id, 248_044_u32);
        assert_eq!(cfg.max_position_embeddings, 262_144);
    }

    #[test]
    fn test_qwen36_27b_preset_layer_types() {
        let cfg = Qwen35Config::qwen36_27b();
        assert_eq!(cfg.layer_types.len(), 64);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(
            cfg.layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count(),
            48
        );
        assert_eq!(
            cfg.layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
            16
        );
        // Every 4th layer (1-indexed) is full attention: indices 3, 7, 11, ..., 63
        for i in 0..64_usize {
            let expected = (i + 1) % 4 == 0;
            assert_eq!(
                cfg.layer_types[i] == LayerType::FullAttention,
                expected,
                "layer {i} type mismatch"
            );
        }
    }

    #[test]
    fn test_qwen36_27b_preset_not_moe() {
        let cfg = Qwen35Config::qwen36_27b();
        assert!(!cfg.is_moe());
        assert!(cfg.num_experts.is_none());
        assert!(cfg.num_experts_per_tok.is_none());
        assert!(cfg.moe_intermediate_size.is_none());
        assert!(cfg.shared_expert_intermediate_size.is_none());
        assert!(cfg.router_aux_loss_coef.is_none());
        assert!(!cfg.output_router_logits);
    }

    #[test]
    fn test_qwen36_27b_preset_gdn_fields() {
        let cfg = Qwen35Config::qwen36_27b();
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_num_value_heads, Some(48));
        assert_eq!(cfg.linear_num_value_heads(), 48);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_value_head_dim, 128);
        assert_eq!(cfg.linear_conv_kernel_dim, 4);
        // Untied embeddings — separate lm_head.weight
        assert!(!cfg.tie_word_embeddings);
        // MTP
        assert_eq!(cfg.mtp_num_hidden_layers, 1);
        assert!(!cfg.mtp_use_dedicated_embeddings);
    }

    #[test]
    fn test_qwen36_27b_from_config_json() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let path =
            std::path::PathBuf::from(format!("{}/.lattice/models/qwen3.6-27b/config.json", home));
        if !path.exists() {
            return; // model not downloaded; skip
        }
        let cfg = Qwen35Config::from_config_json(&path).expect("27B config.json parses");
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_hidden_layers, 64);
        assert_eq!(cfg.vocab_size, 248_320);
        assert_eq!(cfg.intermediate_size, 17408);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.layer_types.len(), 64);
        assert!(!cfg.is_moe());
        assert_eq!(cfg.mtp_num_hidden_layers, 1);
        // rope_theta is nested under rope_parameters — from_config_json_str now extracts it.
        assert!(
            (cfg.rope_theta - 10_000_000.0_f64).abs() < 1.0,
            "rope_theta should be extracted from nested rope_parameters"
        );
    }

    // --- layer_mask tests ---

    #[test]
    fn test_layer_mask_default_all_true_27b() {
        let cfg = Qwen35Config::qwen36_27b();
        assert_eq!(cfg.layer_mask.len(), 64);
        assert!(cfg.layer_mask.iter().all(|&active| active));
        assert_eq!(cfg.num_active_layers(), 64);
        assert_eq!(cfg.num_active_linear_attention_layers(), 48);
        assert_eq!(cfg.num_active_full_attention_layers(), 16);
    }

    #[test]
    fn test_num_active_layers_partial_mask() {
        let mut cfg = Qwen35Config::qwen36_27b();
        // Deactivate layer 0 (GDN), layer 3 (GQA), layer 4 (GDN).
        let mut mask = vec![true; 64];
        mask[0] = false;
        mask[3] = false;
        mask[4] = false;
        cfg.apply_layer_mask(mask);
        assert_eq!(cfg.num_active_layers(), 61);
        assert_eq!(cfg.num_active_linear_attention_layers(), 46);
        assert_eq!(cfg.num_active_full_attention_layers(), 15);
    }

    #[test]
    #[should_panic(expected = "layer_mask length")]
    fn test_apply_layer_mask_wrong_length_panics() {
        let mut cfg = Qwen35Config::qwen36_27b();
        cfg.apply_layer_mask(vec![true; 32]);
    }

    #[test]
    fn test_pruned_config_preserves_fields() {
        let cfg = Qwen35Config::qwen36_27b();
        let mut mask = vec![true; 64];
        mask[5] = false;
        mask[10] = false;
        let pruned = cfg.pruned_config(mask.clone());
        assert_eq!(pruned.hidden_size, cfg.hidden_size);
        assert_eq!(pruned.num_hidden_layers, cfg.num_hidden_layers);
        assert_eq!(pruned.vocab_size, cfg.vocab_size);
        assert_eq!(pruned.layer_types, cfg.layer_types);
        assert_eq!(pruned.layer_mask, mask);
        assert_eq!(pruned.num_active_layers(), 62);
    }

    #[test]
    #[should_panic(expected = "at least one active layer")]
    fn test_apply_layer_mask_all_false_panics() {
        let mut cfg = Qwen35Config::qwen36_27b();
        cfg.apply_layer_mask(vec![false; 64]);
    }

    #[test]
    fn test_layer_mask_normalizes_on_parse() {
        let json = r#"{
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "full_attention_interval": 4,
                "eos_token_id": 1
            }
        }"#;
        let cfg = Qwen35Config::from_config_json_str(json).unwrap();
        assert_eq!(
            cfg.layer_mask.len(),
            4,
            "normalize_layer_mask must fill mask to num_hidden_layers"
        );
        assert!(
            cfg.layer_mask.iter().all(|&v| v),
            "normalized mask must be all-true"
        );
    }

    #[test]
    fn test_generate_config_enable_thinking_default_and_toggle() {
        let default_cfg = GenerateConfig::default();
        assert!(
            default_cfg.enable_thinking,
            "default must have thinking enabled"
        );

        let no_think = GenerateConfig {
            enable_thinking: false,
            ..GenerateConfig::default()
        };
        assert!(!no_think.enable_thinking);
        // Other fields unaffected
        assert_eq!(no_think.max_new_tokens, 256);
        assert!((no_think.temperature - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_qwen36_27b_layer_distribution_matches_config_json() {
        // Verify compute_layer_types(64, 4) produces the pattern from config.json:
        // [linear, linear, linear, full] × 16 times
        let types = compute_layer_types(64, 4);
        for chunk_start in (0..64).step_by(4) {
            assert_eq!(types[chunk_start], LayerType::LinearAttention);
            assert_eq!(types[chunk_start + 1], LayerType::LinearAttention);
            assert_eq!(types[chunk_start + 2], LayerType::LinearAttention);
            assert_eq!(types[chunk_start + 3], LayerType::FullAttention);
        }
    }

    #[test]
    fn test_qwen35_config_backward_compat() {
        // Qwen3.5 config (no MoE fields in JSON) must still deserialize correctly.
        let json = r#"{
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "vocab_size": 248320,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "rms_norm_eps": 0.000001,
                "intermediate_size": 6144,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 4,
                "eos_token_id": 248044,
                "max_position_embeddings": 262144,
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25
            }
        }"#;

        let cfg = Qwen35Config::from_config_json_str(json).expect("backward-compat config parses");
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.linear_num_value_heads(), 16);
        assert!(!cfg.is_moe(), "Qwen3.5 must not be detected as MoE");
        assert!(
            cfg.tie_word_embeddings,
            "default tie_word_embeddings is true"
        );
        assert_eq!(cfg.num_experts, None);
        assert_eq!(cfg.num_experts_per_tok, None);
        assert_eq!(
            cfg.mtp_num_hidden_layers, 0,
            "Qwen3.5 mtp_num_hidden_layers must default to 0"
        );
        assert!(!cfg.mtp_use_dedicated_embeddings);
    }

    #[test]
    fn test_qwen35_0_8b_preset_dimensions() {
        let cfg = Qwen35Config::qwen35_0_8b();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.vocab_size, 248_320);
        assert_eq!(cfg.intermediate_size, 3584);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.rope_dim(), 64); // 256 * 0.25
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_num_value_heads(), 16);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_value_head_dim, 128);
        assert_eq!(cfg.eos_token_id, 248_044);
        assert_eq!(cfg.max_position_embeddings, 262_144);
        assert_eq!(cfg.mtp_num_hidden_layers, 1);
        assert!(cfg.tie_word_embeddings);
        assert!(!cfg.is_moe(), "Qwen3.5-0.8B is dense, not MoE");
        // Same hybrid pattern as the 2B: [linear, linear, linear, full] x 6.
        assert_eq!(cfg.layer_types.len(), 24);
        assert_eq!(cfg.num_full_attention_layers(), 6);
        assert_eq!(cfg.num_linear_attention_layers(), 18);
    }

    #[test]
    fn test_qwen35_0_8b_config_json_fixture_parses() {
        // Parse the real released config.json (downloaded verbatim) — proves
        // from_config_json handles the 0.8B checkpoint, not just my transcription.
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen35_0_8b_config.json"
        ));
        let cfg =
            Qwen35Config::from_config_json_str(json).expect("Qwen3.5-0.8B config.json parses");

        // Core dims must match the released checkpoint.
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.vocab_size, 248_320);
        assert_eq!(cfg.intermediate_size, 3584);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_num_value_heads(), 16);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_value_head_dim, 128);
        assert_eq!(cfg.linear_conv_kernel_dim, 4);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(cfg.eos_token_id, 248_044);
        assert_eq!(cfg.max_position_embeddings, 262_144);
        assert_eq!(cfg.mtp_num_hidden_layers, 1);

        // Released checkpoint is a dense vision-language model — neither the MoE
        // fields nor the vision wrapper may leak into the text config.
        assert!(!cfg.is_moe(), "Qwen3.5-0.8B is dense, not MoE");

        // rope_theta and partial_rotary_factor are nested under rope_parameters
        // in this checkpoint; verify they resolve to the correct values.
        assert_eq!(cfg.rope_theta, 10_000_000.0);
        assert!((cfg.partial_rotary_factor - 0.25).abs() < 1e-6);
        assert_eq!(cfg.rope_dim(), 64);

        // layer_types comes from the explicit JSON array: [lin, lin, lin, full] x 6.
        assert_eq!(cfg.layer_types.len(), 24);
        assert_eq!(cfg.num_full_attention_layers(), 6);
        assert_eq!(cfg.num_linear_attention_layers(), 18);
        let full_indices: Vec<usize> = cfg
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::FullAttention)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(full_indices, vec![3, 7, 11, 15, 19, 23]);

        // tie_word_embeddings is taken from the outer wrapper.
        assert!(cfg.tie_word_embeddings);
    }
}
