//! Configuration for Qwen hybrid attention models (3.5-2B and 3.6-35B-A3B).
//!
//! Qwen3.5-2B uses a hybrid architecture with two attention mechanisms:
//! - **GatedDeltaNet** (linear attention): 18 layers with recurrent state
//! - **Full GQA** (standard attention): 6 layers with KV cache
//!
//! Qwen3.6-35B-A3B extends this with a Mixture-of-Experts (MoE) FFN:
//! 256 routed experts, top-8 selection, 1 shared expert, separate lm_head.

use crate::error::InferenceError;
use crate::grammar::GrammarEngine;
use crate::stop_reason::StopReason;
use std::path::Path;
use std::sync::Arc;

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
    /// Rotation seed used during QuaRot conversion. `None` for non-QuaRot artifacts.
    /// Used at runtime to reconstruct `RandomizedHadamard` for MTP counter-rotation.
    #[serde(default)]
    pub quarot_rotation_seed: Option<u64>,

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

// Nested rope_parameters in HF config.json (many models nest rope_theta and
// partial_rotary_factor here instead of at the top level of text_config).
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct RopeParams {
    #[serde(default)]
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
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
            quarot_rotation_seed: None,
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
            quarot_rotation_seed: None,
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
            quarot_rotation_seed: None,
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
            quarot_rotation_seed: None,
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
        // Many models nest rope_theta and partial_rotary_factor under rope_parameters
        // instead of at the text_config level — extract when the flat fields are unset.
        if let Some(rp) = &cfg.rope_parameters {
            if cfg.rope_theta == 0.0 && rp.rope_theta > 0.0 {
                cfg.rope_theta = rp.rope_theta;
            }
            if let Some(prf) = rp.partial_rotary_factor {
                cfg.partial_rotary_factor = prf;
            }
        }
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            // compute_layer_types uses `(i + 1) % interval`; a zero interval (from an
            // explicit `"full_attention_interval": 0`, or the container `#[serde(default)]`
            // fallback when a preset's interval is zero) would panic with a remainder
            // divide-by-zero. Malformed input must surface as a typed error, not a panic.
            if cfg.full_attention_interval == 0 {
                return Err(InferenceError::Inference(
                    "invalid Qwen config.json: full_attention_interval must be > 0 when \
                     layer_types is absent or its length differs from num_hidden_layers"
                        .to_string(),
                ));
            }
            cfg.layer_types =
                compute_layer_types(cfg.num_hidden_layers, cfg.full_attention_interval);
        }
        cfg.normalize_layer_mask();

        // Structural invariants. A parseable-but-malformed config.json can set these to zero
        // or inconsistent values that survive serde yet cause a downstream divide-by-zero,
        // out-of-bounds index, or unsigned underflow at model construction / forward (e.g.
        // `num_q_heads / num_kv_heads` in the GQA path, `head_vec[rope_dim / 2 + i]` in
        // partial RoPE, `linear_conv_kernel_dim - 1` in the GatedDeltaNet conv buffer).
        // Surface them as typed errors at this single load-time choke point rather than as a
        // panic deep in the forward pass. Presets satisfy all of these by construction.
        if cfg.num_attention_heads == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: num_attention_heads must be > 0".to_string(),
            ));
        }
        if cfg.num_key_value_heads == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: num_key_value_heads must be > 0".to_string(),
            ));
        }
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: num_attention_heads ({}) must be divisible by \
                 num_key_value_heads ({})",
                cfg.num_attention_heads, cfg.num_key_value_heads
            )));
        }
        if cfg.head_dim == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: head_dim must be > 0".to_string(),
            ));
        }
        if cfg.num_hidden_layers == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: num_hidden_layers must be > 0".to_string(),
            ));
        }
        if cfg.linear_conv_kernel_dim == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: linear_conv_kernel_dim must be > 0".to_string(),
            ));
        }
        // `rope_dim = (head_dim * partial_rotary_factor) as usize` is rotated in place over a
        // `head_dim`-length head slice; a factor > 1.0 makes `rope_dim` exceed `head_dim` and
        // indexes `head_vec[rope_dim / 2 + i]` out of bounds. Require a finite fraction.
        if !(cfg.partial_rotary_factor.is_finite()
            && cfg.partial_rotary_factor >= 0.0
            && cfg.partial_rotary_factor <= 1.0)
        {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: partial_rotary_factor ({}) must be in [0.0, 1.0]",
                cfg.partial_rotary_factor
            )));
        }
        // rope_dim = (head_dim * partial_rotary_factor) as usize.  apply_partial_rope pairs
        // dimensions as (i, half+i) for i in 0..half, where half = rope_dim / 2.  An odd
        // rope_dim silently truncates: only 2*(rope_dim/2) dims are rotated, leaving one dim
        // inside the documented "first rope_dim dimensions" range untouched — wrong output
        // with no error signal.  rope_dim == 0 makes RopeTable::max_positions() return 0,
        // which causes every non-empty-sequence call to the capacity-guarded APIs to fail
        // instead of the intended no-op.  rope_dim > head_dim indexes head_vec[half + i] past
        // the head_dim-length slice: the partial_rotary_factor <= 1.0 check above bounds this
        // in real arithmetic, but rope_dim() casts head_dim through f32, so a head_dim above
        // f32's exact-integer range (2^24) can round UP and derive rope_dim > head_dim even at
        // factor 1.0.  Reject all three fail-closed; no-RoPE variants that need rope_dim==0
        // require an explicit dispatch path (Refs #401).
        let rope_dim = cfg.rope_dim();
        if rope_dim < 2 || !rope_dim.is_multiple_of(2) || rope_dim > cfg.head_dim {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: derived rope_dim ({rope_dim}) must be even, >= 2, \
                 and <= head_dim ({hd}) (partial_rotary_factor={prf})",
                hd = cfg.head_dim,
                prf = cfg.partial_rotary_factor,
            )));
        }
        // The GatedDeltaNet fused path divides by these head counts: `value_heads / key_heads`
        // (gdn_fused.rs ratio) and `h / ratio` per value head. A parseable config with
        // `linear_num_key_heads == 0`, `linear_num_value_heads == 0`, or value-heads not a
        // positive multiple of key-heads (ratio == 0) is an integer divide-by-zero panic deep in
        // the recurrence. Real GDN configs are key=16/value=32 (ratio 2); reject the rest here.
        let key_heads = cfg.linear_num_key_heads;
        let value_heads = cfg.linear_num_value_heads();
        if key_heads == 0 {
            return Err(InferenceError::Inference(
                "invalid Qwen config.json: linear_num_key_heads must be > 0".to_string(),
            ));
        }
        if value_heads == 0 || !value_heads.is_multiple_of(key_heads) {
            return Err(InferenceError::Inference(format!(
                "invalid Qwen config.json: linear_num_value_heads ({value_heads}) must be a \
                 positive multiple of linear_num_key_heads ({key_heads})"
            )));
        }

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

    /// **Unstable**: number of layers that hold a KV cache.
    ///
    /// Only full-attention (GQA) layers carry a growing KV cache. The
    /// GatedDeltaNet linear-attention layers carry fixed-size recurrent state
    /// that does not grow with sequence length. Therefore KV memory scales with
    /// `num_full_attention_layers()`, not `num_hidden_layers`.
    ///
    /// For qwen3.5-0.8B this is 6 (not 24). A regression to all-24-layer KV
    /// allocation would 4× decode memory; this method makes the invariant
    /// testable.
    pub fn kv_cache_layer_count(&self) -> usize {
        self.num_full_attention_layers()
    }

    /// **Unstable**: total KV cache bytes consumed per input token.
    ///
    /// Formula: `num_full_attention_layers × 2 (K and V) × full_kv_dim × dtype_bytes`.
    ///
    /// Pass `dtype_bytes = 2` for f16, `dtype_bytes = 4` for f32.
    ///
    /// For qwen3.5-0.8B with f16:
    /// `6 × 2 × 512 × 2 = 12_288 B/token ≈ 48 MiB at a 4096-token context`.
    ///
    /// Numeric identity (preset-specific, NOT a general formula): at `dtype_bytes = 1`,
    /// `kv_bytes_per_token(1) = 6 × 2 × 512 = 6_144 = 24 × 256 = num_hidden_layers × head_dim`.
    /// Since `kv_bytes_per_token(1) = num_full × 2 × num_kv_heads × head_dim`, it equals
    /// `num_hidden_layers × head_dim` exactly when `num_full × 2 × num_kv_heads == num_hidden_layers`
    /// — true for 0.8B (`6 × 2 × 2 == 24`) but not in general. Do not rely on it across configs.
    pub fn kv_bytes_per_token(&self, dtype_bytes: usize) -> usize {
        self.num_full_attention_layers() * 2 * self.full_kv_dim() * dtype_bytes
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
#[derive(Clone)]
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
    /// Enable multi-token prediction when the model has MTP weights loaded.
    /// Replaces the `LATTICE_MTP` env var for programmatic control.
    /// `None` = defer to `LATTICE_MTP` env var (backwards-compatible default).
    pub enable_mtp: Option<bool>,
    /// Optional grammar-constrained decoding engine (ADR-046).
    ///
    /// When set, `mask_logits` is called on CPU logits before sampling on every step.
    /// The Metal path copies logits to CPU before sampling — no additional GPU transfer needed.
    pub grammar: Option<Arc<GrammarEngine>>,
    /// Additional string-level stop sequences. When any appears in the output, generation
    /// halts and the matched text is excluded. Empty = disabled (default; parity-safe).
    pub stop_strings: Vec<String>,
    /// Reasoning-budget forcing (s1-style): after this many reasoning tokens are
    /// generated without a `</think>`, force-inject `</think>` to commit the model
    /// to an answer. `None` or `Some(0)` = disabled (no behaviour change).
    pub reasoning_budget: Option<usize>,
}

impl std::fmt::Debug for GenerateConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenerateConfig")
            .field("max_new_tokens", &self.max_new_tokens)
            .field("temperature", &self.temperature)
            .field("top_k", &self.top_k)
            .field("top_p", &self.top_p)
            .field("repetition_penalty", &self.repetition_penalty)
            .field("seed", &self.seed)
            .field("stop_token_ids", &self.stop_token_ids)
            .field("enable_thinking", &self.enable_thinking)
            .field("enable_mtp", &self.enable_mtp)
            .field("grammar", &self.grammar.as_ref().map(|_| "<GrammarEngine>"))
            .field("stop_strings", &self.stop_strings)
            .field("reasoning_budget", &self.reasoning_budget)
            .finish()
    }
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
            enable_mtp: None,
            grammar: None,
            stop_strings: vec![],
            reasoning_budget: None,
        }
    }
}

/// Decide whether to force-close the thinking block this step (s1 budget forcing).
///
/// Returns `Some(close_id)` to override the sampled token with `</think>`, else `None`.
/// All conditions must hold: budget enabled and non-zero, thinking block is still open,
/// enough tokens have been generated. Returns `None` immediately if any guard fails so
/// the common disabled path costs a single `Option::None` check per step.
#[inline]
pub(crate) fn force_close_think(
    reasoning_budget: Option<usize>,
    enable_thinking: bool,
    thinking_closed: bool,
    generated_so_far: usize,
    close_id: Option<u32>,
) -> Option<u32> {
    let budget = reasoning_budget?;
    let close = close_id?;
    if enable_thinking && !thinking_closed && budget > 0 && generated_so_far >= budget {
        Some(close)
    } else {
        None
    }
}

/// Decode-loop iteration cap.
///
/// When a reasoning budget is active, reasoning tokens get their OWN budget (`rb`) ON TOP
/// of the answer budget (`max_new_tokens`), plus **1** for the forced `</think>` delimiter
/// itself: worst case `rb + max_new_tokens + 1` total tokens. The +1 is necessary because
/// the forced `</think>` is an extra token that is not part of either the reasoning content
/// or the answer — omitting it leaves the answer one token short (off-by-one).
///
/// Without a budget (`None` or `Some(0)`) the cap is unchanged (`max_new_tokens`), so the
/// disabled path is byte-identical to the pre-budget behaviour (parity-safe).
#[inline]
pub(crate) fn decode_cap(reasoning_budget: Option<usize>, max_new_tokens: usize) -> usize {
    match reasoning_budget {
        // rb reasoning-content tokens + 1 forced </think> delimiter + max_new_tokens answer tokens.
        Some(rb) if rb > 0 => rb.saturating_add(max_new_tokens).saturating_add(1),
        _ => max_new_tokens,
    }
}

/// **Unstable**: text generation output struct; fields may expand with streaming support.
///
/// # Stop-token contract (#613)
///
/// When generation ends because a stop condition was hit (EOS, a configured
/// stop token, or a stop string), that terminating token is **excluded** from
/// `token_ids` and `text` — it is never appended to the output. Every
/// generation entry point across this crate (CPU and Metal) honours this
/// contract (see the `stop_token_contract` test module for the cross-path
/// regression sweep). `generated_tokens` always equals `token_ids.len()`.
#[derive(Debug, Clone)]
pub struct GenerateOutput {
    /// Generated text (excluding prompt).
    pub text: String,
    /// Generated token IDs (excluding any terminating stop token — see the
    /// stop-token contract above).
    pub token_ids: Vec<u32>,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    /// Total tokens generated (excluding prompt).
    pub generated_tokens: usize,
    /// True when generation ended via a stop condition (EOS, a stop token, or a
    /// stop string); false when it ended by reaching `max_new_tokens`. Serve maps
    /// this to the OpenAI `finish_reason` ("stop" vs "length").
    pub stopped: bool,
    /// Why generation terminated. `Some` on every real generation exit; `None` only on
    /// non-generation returns that have no issue-listed cause.
    pub stop_reason: Option<StopReason>,
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
            std::path::PathBuf::from(format!("{home}/.lattice/models/qwen3.6-27b/config.json"));
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
    fn test_zero_full_attention_interval_errors_not_panics() {
        // A config.json with full_attention_interval: 0 must return a clean
        // InferenceError, never panic. layer_types is omitted, so the container
        // #[serde(default)] fills it from the preset (whose length differs from
        // num_hidden_layers: 4), forcing the recompute branch that calls
        // compute_layer_types(4, 0) -> (i + 1) % 0 -> divide-by-zero panic
        // without the guard.
        let json = r#"{
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "full_attention_interval": 0,
                "eos_token_id": 1
            }
        }"#;
        let result = Qwen35Config::from_config_json_str(json);
        assert!(
            result.is_err(),
            "full_attention_interval: 0 must yield an InferenceError, not panic"
        );
    }

    #[test]
    fn test_zero_num_key_value_heads_errors_not_panics() {
        // An explicit num_key_value_heads: 0 survives serde but reaches a divide-by-zero
        // (`num_q_heads / num_kv_heads`) and a hard `assert!(num_kv_heads > 0)` in the GQA
        // forward path. Reject at parse time. Omitted fields fall back to the valid preset.
        // The substring assert proves THIS guard fired (not an unrelated default-zero field):
        // `#[serde(default)]` + `Default = qwen36_35b_a3b()` means every omitted field carries a
        // valid preset value, so the one explicit bad field is the only one that can trip.
        let json = r#"{"text_config": {"num_key_value_heads": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("num_key_value_heads: 0 must yield an InferenceError, not a panic")
            .to_string();
        assert!(
            err.contains("num_key_value_heads"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_indivisible_head_counts_error_not_panics() {
        // num_attention_heads not divisible by num_key_value_heads truncates the GQA group
        // count and over-runs the KV row (OOB read) on the unasserted release path.
        let json = r#"{"text_config": {"num_attention_heads": 3, "num_key_value_heads": 2}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("indivisible head counts must yield an InferenceError, not OOB/panic")
            .to_string();
        assert!(err.contains("divisible"), "wrong guard fired: {err}");
    }

    #[test]
    fn test_zero_head_dim_errors() {
        let json = r#"{"text_config": {"head_dim": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("head_dim: 0 must yield an InferenceError")
            .to_string();
        assert!(err.contains("head_dim"), "wrong guard fired: {err}");
    }

    #[test]
    fn test_zero_num_hidden_layers_errors() {
        let json = r#"{"text_config": {"num_hidden_layers": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("num_hidden_layers: 0 must yield an InferenceError")
            .to_string();
        assert!(
            err.contains("num_hidden_layers"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_zero_linear_conv_kernel_dim_errors() {
        // `linear_conv_kernel_dim - 1` underflows usize (panics in debug, wraps to a ~16 EiB
        // allocation in release) in the GatedDeltaNet conv-buffer sizing.
        let json = r#"{"text_config": {"linear_conv_kernel_dim": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("linear_conv_kernel_dim: 0 must yield an InferenceError, not underflow")
            .to_string();
        assert!(
            err.contains("linear_conv_kernel_dim"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_zero_linear_num_key_heads_errors() {
        // gdn_fused divides `value_heads / linear_num_key_heads`; a parseable 0 is a hard
        // integer divide-by-zero panic deep in the GatedDeltaNet recurrence (codex #342 finding).
        let json = r#"{"text_config": {"linear_num_key_heads": 0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("linear_num_key_heads: 0 must yield an InferenceError, not divide-by-zero")
            .to_string();
        assert!(
            err.contains("linear_num_key_heads"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_value_heads_below_key_heads_errors() {
        // value_heads < key_heads makes the integer `ratio = value_heads / key_heads == 0`, then
        // `h / ratio` is a divide-by-zero panic. value=1/key=16 (preset) hits ratio 0.
        let json = r#"{"text_config": {"linear_num_value_heads": 1}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("value_heads < key_heads must yield an InferenceError, not divide-by-zero")
            .to_string();
        assert!(
            err.contains("linear_num_value_heads"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_value_heads_multiple_of_key_heads_accepted() {
        // Boundary: the real GDN shape (key 16, value 32 → ratio 2) must pass the divisibility
        // guard. Guards against the new check wrongly rejecting legitimate asymmetric heads.
        let json = r#"{"text_config": {"linear_num_key_heads": 16, "linear_num_value_heads": 32}}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "key 16 / value 32 (ratio 2) is a real GDN config and must be accepted"
        );
    }

    #[test]
    fn test_partial_rotary_factor_above_one_errors() {
        // rope_dim = (head_dim * factor); factor > 1 makes rope_dim exceed head_dim and
        // indexes head_vec[rope_dim/2 + i] out of bounds in apply_partial_rope.
        let json = r#"{"text_config": {"partial_rotary_factor": 3.0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("partial_rotary_factor > 1.0 must yield an InferenceError, not OOB")
            .to_string();
        assert!(
            err.contains("partial_rotary_factor"),
            "wrong guard fired: {err}"
        );
    }

    #[test]
    fn test_partial_rotary_factor_one_accepted() {
        // Boundary: factor == 1.0 makes rope_dim == head_dim (full rotary), which is in range.
        // Guards against an off-by-one in the [0.0, 1.0] range check.
        let json = r#"{"text_config": {"partial_rotary_factor": 1.0}}"#;
        assert!(
            Qwen35Config::from_config_json_str(json).is_ok(),
            "partial_rotary_factor == 1.0 (full rotary) must be accepted"
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // rope_dim invariant tests (issue #401)
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn test_odd_rope_dim_errors_not_panics() {
        // Mutation contract: removing the `rope_dim < 2 || rope_dim % 2 != 0` guard must
        // make this test FAIL (the call returns Ok instead of Err).
        //
        // head_dim=10, partial_rotary_factor=0.3 → rope_dim = (10 * 0.3) as usize = 3 (odd).
        // apply_partial_rope uses half = rope_dim / 2 = 1, rotating only pair (0,1) and
        // leaving dim 2 (inside the declared rotate range) silently unrotated — wrong output.
        let json = r#"{"text_config": {"head_dim": 10, "partial_rotary_factor": 0.3}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("odd rope_dim must yield an InferenceError, not silent wrong output")
            .to_string();
        assert!(err.contains("rope_dim"), "wrong guard fired: {err}");
    }

    #[test]
    fn test_zero_rope_dim_errors_not_panics() {
        // Mutation contract: removing the `rope_dim < 2 || rope_dim % 2 != 0` guard must
        // make this test FAIL (the call returns Ok instead of Err).
        //
        // partial_rotary_factor=0.0 → rope_dim = 0.  RopeTable::new(0, ..) gives
        // max_positions()=0, so every non-empty-sequence call to the capacity-guarded APIs
        // rejects the input rather than applying a no-op — contrary to caller expectations.
        // Reject fail-closed until a dedicated no-RoPE dispatch path exists (Refs #401).
        let json = r#"{"text_config": {"partial_rotary_factor": 0.0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("zero rope_dim must yield an InferenceError, not capacity-zero surprise")
            .to_string();
        assert!(err.contains("rope_dim"), "wrong guard fired: {err}");
    }

    #[test]
    fn test_rope_dim_exceeds_head_dim_via_f32_rounding_errors() {
        // Mutation contract: removing the `rope_dim > cfg.head_dim` guard must make this
        // test FAIL (the call returns Ok instead of Err).
        //
        // partial_rotary_factor=1.0 keeps rope_dim <= head_dim in real arithmetic, but
        // rope_dim() casts head_dim through f32: head_dim=16_777_219 (2^24 + 3) rounds UP to
        // 16_777_220, so rope_dim=16_777_220 > head_dim. That value is even and >= 2, so it
        // slips the lower-bound/parity guard and would index head_vec[rope_dim/2 + i] one past
        // the head_dim-length slice in apply_partial_rope. Fail closed at parse time.
        let json = r#"{"text_config": {"head_dim": 16777219, "partial_rotary_factor": 1.0}}"#;
        let err = Qwen35Config::from_config_json_str(json)
            .expect_err("rope_dim > head_dim (f32 rounding) must yield an InferenceError, not OOB")
            .to_string();
        assert!(err.contains("rope_dim"), "wrong guard fired: {err}");
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

    // ──────────────────────────────────────────────────────────────────────
    // KV-cache layer-count invariant tests (issue #170)
    // ──────────────────────────────────────────────────────────────────────

    /// KV cache layer count is the number of full-attention layers, NOT all layers.
    ///
    /// This is the primary regression guard: a silent switch to all-24-layer KV
    /// allocation would change `kv_cache_layer_count()` from 6 to 24 and cause
    /// this test to fail.
    #[test]
    fn kv_layer_count_excludes_linear_layers() {
        let cfg = Qwen35Config::qwen35_0_8b();
        assert_eq!(
            cfg.kv_cache_layer_count(),
            6,
            "must be full-attention count"
        );
        assert_ne!(
            cfg.kv_cache_layer_count(),
            cfg.num_hidden_layers,
            "kv_cache_layer_count must not equal num_hidden_layers (would 4× decode memory)"
        );
    }

    /// Full + linear layers must sum to total hidden layers for every preset.
    ///
    /// `compute_layer_types` produces exactly `num_hidden_layers` entries, each
    /// either FullAttention or LinearAttention, so this invariant must always hold.
    /// If any preset fails here it indicates a config bug.
    #[test]
    fn full_plus_linear_equals_total() {
        for (name, cfg) in [
            ("qwen35_0_8b", Qwen35Config::qwen35_0_8b()),
            ("qwen35_2b", Qwen35Config::qwen35_2b()),
            ("qwen36_35b_a3b", Qwen35Config::qwen36_35b_a3b()),
            ("qwen36_27b", Qwen35Config::qwen36_27b()),
        ] {
            assert_eq!(
                cfg.num_full_attention_layers() + cfg.num_linear_attention_layers(),
                cfg.num_hidden_layers,
                "{name}: full + linear must equal num_hidden_layers"
            );
        }
    }

    /// KV bytes per token for f16 matches the expected 12_288 B for qwen3.5-0.8B.
    ///
    /// Formula: `num_full(6) × 2(K+V) × full_kv_dim(512) × dtype_bytes(2) = 12_288`.
    /// At a 4096-token context this is 48 MiB.
    #[test]
    fn kv_bytes_per_token_f16() {
        let cfg = Qwen35Config::qwen35_0_8b();
        // 6 layers × 2 (K+V) × 512 (kv_dim) × 2 (f16) = 12_288 B/token = 48 MiB @ 4096 ctx
        assert_eq!(cfg.kv_bytes_per_token(2), 12_288);
    }

    /// Numeric identity: kv_bytes_per_token(1) == num_hidden_layers * head_dim for 0.8B.
    ///
    /// `6 × 2 × 512 × 1 = 6_144 = 24 × 256`. This is a coincidence specific to
    /// the 0.8B parameters (full_attention_interval=4, num_kv_heads=2); it does
    /// NOT generalise across configs and must not be used as a formula.
    #[test]
    fn kv_bytes_per_token_identity() {
        let cfg = Qwen35Config::qwen35_0_8b();
        assert_eq!(cfg.kv_bytes_per_token(1), 6_144);
        // Coincidence: matches num_hidden_layers × head_dim for this specific preset only.
        assert_eq!(
            cfg.kv_bytes_per_token(1),
            cfg.num_hidden_layers * cfg.head_dim
        );
    }

    // ── decode_cap unit tests ────────────────────────────────────────────────

    #[test]
    fn decode_cap_none_budget_returns_max() {
        // Disabled path must be byte-identical to the pre-budget behaviour.
        assert_eq!(decode_cap(None, 512), 512);
        assert_eq!(decode_cap(None, 0), 0);
    }

    #[test]
    fn decode_cap_zero_budget_returns_max() {
        // Some(0) is treated as disabled.
        assert_eq!(decode_cap(Some(0), 512), 512);
        assert_eq!(decode_cap(Some(0), 1), 1);
    }

    #[test]
    fn decode_cap_nonzero_budget_adds_budgets() {
        // Worst case = rb + max_new_tokens + 1 (the +1 is the forced </think> delimiter).
        // Mutation-sensitive: revert to rb+max and these assertions fail.
        assert_eq!(decode_cap(Some(2048), 512), 2561);
        assert_eq!(decode_cap(Some(1), 1), 3);
        assert_eq!(decode_cap(Some(100), 200), 301);
    }

    #[test]
    fn decode_cap_saturates_on_overflow() {
        // saturating_add must not wrap on usize::MAX inputs.
        assert_eq!(decode_cap(Some(usize::MAX), 1), usize::MAX);
        assert_eq!(decode_cap(Some(1), usize::MAX), usize::MAX);
    }

    // ── force_close_think unit tests ────────────────────────────────────────

    #[test]
    fn force_close_think_disabled_when_budget_none() {
        // budget=None → always returns None regardless of other args.
        assert_eq!(
            force_close_think(None, true, false, 100, Some(99)),
            None,
            "None budget must disable forcing"
        );
    }

    #[test]
    fn force_close_think_disabled_when_budget_zero() {
        // budget=Some(0) → budget > 0 guard fails → None.
        assert_eq!(
            force_close_think(Some(0), true, false, 100, Some(99)),
            None,
            "budget=0 must disable forcing"
        );
    }

    #[test]
    fn force_close_think_disabled_when_enable_thinking_false() {
        // enable_thinking=false → no reasoning block → forcing is a no-op.
        assert_eq!(
            force_close_think(Some(10), false, false, 20, Some(99)),
            None,
            "enable_thinking=false must disable forcing"
        );
    }

    #[test]
    fn force_close_think_disabled_when_already_closed() {
        // thinking_closed=true → block already closed → should not force again.
        assert_eq!(
            force_close_think(Some(10), true, true, 20, Some(99)),
            None,
            "already-closed thinking block must not force again"
        );
    }

    #[test]
    fn force_close_think_disabled_when_close_id_none() {
        // close_id=None → model has no </think> token → forcing is a no-op.
        assert_eq!(
            force_close_think(Some(10), true, false, 20, None),
            None,
            "close_id=None must disable forcing"
        );
    }

    #[test]
    fn force_close_think_fires_at_budget_boundary() {
        let close_id = 248_069_u32;
        // generated_so_far == budget → should force (mutation: >= not >).
        assert_eq!(
            force_close_think(Some(10), true, false, 10, Some(close_id)),
            Some(close_id),
            "must force when generated_so_far equals budget"
        );
        // generated_so_far > budget → also forces.
        assert_eq!(
            force_close_think(Some(10), true, false, 11, Some(close_id)),
            Some(close_id),
            "must force when generated_so_far exceeds budget"
        );
    }

    #[test]
    fn force_close_think_does_not_fire_before_budget() {
        let close_id = 248_069_u32;
        // generated_so_far < budget → must NOT force (mutation-sensitive boundary).
        assert_eq!(
            force_close_think(Some(10), true, false, 9, Some(close_id)),
            None,
            "must not force when generated_so_far is one below budget"
        );
        assert_eq!(
            force_close_think(Some(10), true, false, 0, Some(close_id)),
            None,
            "must not force when zero tokens generated"
        );
    }
}
