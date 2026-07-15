//! Configuration for the Gemma 4 E2B text decoder (ADR-082 stage 2).
//!
//! Gemma 4 E2B's decoder alternates sliding-window local attention with full
//! (global) attention every `sliding_window_pattern`-th layer, and shares K/V
//! state across its final `num_kv_shared_layers` layers. Per ADR-082
//! Amendment 1, "shared" is a *functional* runtime property only: the
//! checkpoint still carries `k_proj`/`v_proj` (and `k_norm`) weights for
//! every layer, at a width keyed by attention type (`head_dim` on sliding
//! layers, `global_head_dim` on global layers) rather than by sharing. This
//! module parses and validates the config; [`crate::model::gemma4_preflight`]
//! is what applies the resulting per-layer geometry to a tensor inventory.

use crate::error::InferenceError;
use std::path::Path;

/// Expected safetensors dtype for every Gemma 4 E2B language-model tensor
/// (header-verified, ADR-082 G16/G5).
pub const GEMMA4_EXPECTED_DTYPE: &str = "BF16";

/// Per-layer attention kind (ADR-082 G3). Orthogonal to KV sharing: sharing
/// is keyed on layer *index* (the final `num_kv_shared_layers` layers),
/// while this schedule is keyed on layer *position modulo pattern*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gemma4LayerType {
    SlidingAttention,
    FullAttention,
}

/// **Unstable**: Gemma 4 E2B text-decoder configuration; fields evolving with
/// model variants. Parsed from the `text_config` object of a Gemma 4 E2B
/// `config.json` (vision/audio towers are out of scope for this stage; see
/// ADR-082 stages 5-9).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct Gemma4Config {
    // --- Core dimensions (G2) ---
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    /// MLP intermediate size for non-KV-shared layers. KV-shared layers use
    /// double this width (G7) — see [`Self::mlp_intermediate_size`].
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,

    // --- Attention geometry (G3/G4) ---
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Head width on `sliding_attention` layers.
    pub head_dim: usize,
    /// Head width on `full_attention` (global) layers.
    pub global_head_dim: usize,
    pub sliding_window: usize,

    // --- Dual RoPE (G8) ---
    /// Global-layer RoPE base (proportional, over `partial_rotary_factor` of
    /// `global_head_dim`).
    pub rope_theta: f64,
    /// Sliding-layer RoPE base (standard, full `head_dim`).
    pub rope_local_base_freq: f64,
    /// Fraction of `global_head_dim` rotated on global layers. Sliding layers
    /// always use full rotary (`head_dim`, not scaled by this factor).
    pub partial_rotary_factor: f32,

    // --- Layer schedule (G3) ---
    /// Every Nth layer (1-indexed) is `full_attention`; the rest are
    /// `sliding_attention`. Used to derive `layer_types` when the checkpoint
    /// config omits (or mis-sizes) the explicit array.
    pub sliding_window_pattern: usize,
    /// Precomputed per-layer attention kind, length = `num_hidden_layers`.
    #[serde(default)]
    pub layer_types: Vec<Gemma4LayerType>,

    // --- KV sharing (G5, Amendment 1) ---
    /// Count of trailing decoder layers that functionally share K/V state
    /// with the last non-shared layer of their attention type. The
    /// checkpoint still carries (unused) `k_proj`/`v_proj`/`k_norm` weights
    /// on these layers — see [`Self::first_kv_shared_layer_idx`].
    pub num_kv_shared_layers: usize,

    // --- Per-layer embeddings / PLE (G9) ---
    /// Width of the per-layer identity embedding gated into each layer.
    pub hidden_size_per_layer_input: usize,

    // --- Embedding/output projection ---
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    // --- Generation ---
    pub eos_token_id: u32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
}

fn default_tie_word_embeddings() -> bool {
    true
}

fn default_max_position_embeddings() -> usize {
    131_072
}

// Private helper for the HF config.json structure: Gemma 4's text decoder
// fields live under a `text_config` object, a sibling of `vision_config` /
// `audio_config` (out of scope here) at the top level — the same wrapper
// shape as Qwen3.5's `HfQwenConfigFile` in `qwen35_config.rs`.
#[derive(Debug, serde::Deserialize)]
struct HfGemma4ConfigFile {
    #[serde(default)]
    text_config: Option<Gemma4Config>,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
}

impl Default for Gemma4Config {
    fn default() -> Self {
        Self::e2b()
    }
}

impl Gemma4Config {
    /// **Unstable**: default `google/gemma-4-E2B-it` text-decoder configuration
    /// (ADR-082 G2-G9; layer-schedule and KV-sharing indices header-verified
    /// against the committed Stage-0 tensor manifest, PR #991).
    pub fn e2b() -> Self {
        let num_hidden_layers = 35;
        let sliding_window_pattern = 5;
        let layer_types = compute_layer_types(num_hidden_layers, sliding_window_pattern);

        Self {
            hidden_size: 1536,
            num_hidden_layers,
            vocab_size: 262_144,
            intermediate_size: 6144,
            rms_norm_eps: 1e-6,
            num_attention_heads: 8,
            num_key_value_heads: 1,
            head_dim: 256,
            global_head_dim: 512,
            sliding_window: 512,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            partial_rotary_factor: 0.25,
            sliding_window_pattern,
            layer_types,
            num_kv_shared_layers: 20,
            hidden_size_per_layer_input: 256,
            tie_word_embeddings: true,
            eos_token_id: 1,
            max_position_embeddings: 131_072,
        }
    }

    /// Parse a HF Gemma 4 `config.json` (text decoder fields nested under
    /// `text_config`).
    pub fn from_config_json(path: &Path) -> Result<Self, InferenceError> {
        let json = std::fs::read_to_string(path).map_err(InferenceError::Io)?;
        Self::from_config_json_str(&json)
    }

    /// Resolve the Gemma 4 config for a model directory, requiring a real
    /// `config.json` — mirrors `Qwen35Config::from_model_dir` (#923): a
    /// missing file is a hard, descriptive error naming the directory rather
    /// than a silently-substituted E2B preset.
    pub fn from_model_dir(dir: &Path) -> Result<Self, InferenceError> {
        let config_path = dir.join("config.json");
        if !config_path.exists() {
            return Err(InferenceError::ModelNotFound(format!(
                "missing config.json in {} -- no Gemma 4 architecture preset is inferred from \
                 a config-less directory",
                dir.display()
            )));
        }
        Self::from_config_json(&config_path)
    }

    /// Parse HF Gemma 4 `config.json` text into a `Gemma4Config`.
    pub fn from_config_json_str(json: &str) -> Result<Self, InferenceError> {
        let parsed: HfGemma4ConfigFile = serde_json::from_str(json)
            .map_err(|e| InferenceError::Inference(format!("invalid Gemma 4 config.json: {e}")))?;
        let mut cfg = parsed.text_config.unwrap_or_else(Gemma4Config::e2b);

        if let Some(tie) = parsed.tie_word_embeddings {
            cfg.tie_word_embeddings = tie;
        }

        if cfg.layer_types.len() != cfg.num_hidden_layers {
            if cfg.sliding_window_pattern == 0 {
                return Err(InferenceError::Inference(
                    "invalid Gemma 4 config.json: sliding_window_pattern must be > 0 when \
                     layer_types is absent or its length differs from num_hidden_layers"
                        .to_string(),
                ));
            }
            cfg.layer_types =
                compute_layer_types(cfg.num_hidden_layers, cfg.sliding_window_pattern);
        }

        cfg.validate()?;
        Ok(cfg)
    }

    /// Structural invariants every downstream consumer (loader preflight,
    /// eventual forward pass) depends on. A parseable-but-malformed
    /// `config.json` can set these to zero or inconsistent values that
    /// survive serde yet cause a downstream divide-by-zero, out-of-bounds
    /// index, or a silently-wrong tensor-shape expectation. Surface them as
    /// typed errors here, at the single load-time choke point, rather than
    /// deep in the preflight or forward pass. Called both at the
    /// `from_config_json_str` parse boundary and again by
    /// [`crate::model::gemma4_preflight::preflight_check`], since callers can
    /// construct a `Gemma4Config` directly and skip the parse boundary.
    pub fn validate(&self) -> Result<(), InferenceError> {
        if self.num_hidden_layers == 0 {
            return Err(InferenceError::Inference(
                "invalid Gemma 4 config.json: num_hidden_layers must be > 0".to_string(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(InferenceError::Inference(
                "invalid Gemma 4 config.json: hidden_size must be > 0".to_string(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(InferenceError::Inference(
                "invalid Gemma 4 config.json: vocab_size must be > 0".to_string(),
            ));
        }
        if self.intermediate_size == 0 {
            return Err(InferenceError::Inference(
                "invalid Gemma 4 config.json: intermediate_size must be > 0".to_string(),
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(InferenceError::Inference(
                "invalid Gemma 4 config.json: num_attention_heads must be > 0".to_string(),
            ));
        }
        if self.num_key_value_heads == 0 {
            return Err(InferenceError::Inference(
                "invalid Gemma 4 config.json: num_key_value_heads must be > 0".to_string(),
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: num_attention_heads ({}) must be divisible by \
                 num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.head_dim == 0 {
            return Err(InferenceError::Inference(
                "invalid Gemma 4 config.json: head_dim must be > 0".to_string(),
            ));
        }
        if self.global_head_dim == 0 {
            return Err(InferenceError::Inference(
                "invalid Gemma 4 config.json: global_head_dim must be > 0".to_string(),
            ));
        }
        if self.sliding_window == 0 {
            return Err(InferenceError::Inference(
                "invalid Gemma 4 config.json: sliding_window must be > 0".to_string(),
            ));
        }
        if self.hidden_size_per_layer_input == 0 {
            return Err(InferenceError::Inference(
                "invalid Gemma 4 config.json: hidden_size_per_layer_input must be > 0".to_string(),
            ));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: layer_types has {} entries but \
                 num_hidden_layers is {}",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        if self.num_kv_shared_layers > self.num_hidden_layers {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: num_kv_shared_layers ({}) must be <= \
                 num_hidden_layers ({})",
                self.num_kv_shared_layers, self.num_hidden_layers
            )));
        }
        if !(self.rope_theta.is_finite() && self.rope_theta > 0.0) {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: rope_theta ({}) must be finite and > 0",
                self.rope_theta
            )));
        }
        if !(self.rope_local_base_freq.is_finite() && self.rope_local_base_freq > 0.0) {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: rope_local_base_freq ({}) must be finite and > 0",
                self.rope_local_base_freq
            )));
        }
        if !(self.partial_rotary_factor.is_finite()
            && self.partial_rotary_factor > 0.0
            && self.partial_rotary_factor <= 1.0)
        {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: partial_rotary_factor ({}) must be in (0.0, 1.0]",
                self.partial_rotary_factor
            )));
        }
        // Global-layer rope_dim = (global_head_dim * partial_rotary_factor) as usize;
        // apply_partial_rope-style code pairs dimensions as (i, half+i) for i in
        // 0..rope_dim/2 — an odd or zero rope_dim silently mis-rotates or
        // capacity-zeros (see qwen35_config.rs's identical guard, issue #401).
        let global_rope_dim = self.global_rope_dim();
        if global_rope_dim < 2
            || !global_rope_dim.is_multiple_of(2)
            || global_rope_dim > self.global_head_dim
        {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: derived global rope_dim ({global_rope_dim}) must \
                 be even, >= 2, and <= global_head_dim ({}) (partial_rotary_factor={})",
                self.global_head_dim, self.partial_rotary_factor
            )));
        }
        Ok(())
    }

    /// Index of the first layer that functionally shares K/V state (ADR-082
    /// G5/Amendment 1): the final `num_kv_shared_layers` layers.
    pub fn first_kv_shared_layer_idx(&self) -> usize {
        self.num_hidden_layers
            .saturating_sub(self.num_kv_shared_layers)
    }

    /// True when `layer_idx` functionally shares K/V state (reads
    /// `shared_kv_states[layer_type]` instead of its own `k_proj`/`v_proj`).
    /// The checkpoint still carries those layers' K/V weights (Amendment 1)
    /// — the loader's tolerate-and-skip contract, not this predicate,
    /// governs whether they get loaded.
    pub fn is_kv_shared_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.first_kv_shared_layer_idx()
    }

    /// True when `layer_idx` uses full (global) attention. Orthogonal to
    /// [`Self::is_kv_shared_layer`] — sharing is keyed on trailing index,
    /// this is keyed on `layer_types`.
    pub fn is_global_layer(&self, layer_idx: usize) -> bool {
        self.layer_types.get(layer_idx).copied() == Some(Gemma4LayerType::FullAttention)
    }

    /// Attention head width for `layer_idx`: `global_head_dim` on global
    /// layers, `head_dim` on sliding layers (G4).
    pub fn attn_head_dim(&self, layer_idx: usize) -> usize {
        if self.is_global_layer(layer_idx) {
            self.global_head_dim
        } else {
            self.head_dim
        }
    }

    /// MLP intermediate size for `layer_idx`: double-wide on KV-shared
    /// layers (G7), per [`Self::use_double_wide_mlp`].
    pub fn mlp_intermediate_size(&self, layer_idx: usize) -> usize {
        if self.use_double_wide_mlp(layer_idx) {
            self.intermediate_size * 2
        } else {
            self.intermediate_size
        }
    }

    /// Whether `layer_idx` uses the double-wide (`2 * intermediate_size`)
    /// MLP. Per ADR-082 Amendment 1 ("G2/G7 corroboration"), the reference
    /// runtime sets this identically to [`Self::is_kv_shared_layer`] — the
    /// checkpoint's own `mlp.*` shapes are therefore an independent,
    /// second structural observable for the shared-layer set, which
    /// [`crate::model::gemma4_preflight::preflight_check`] cross-checks
    /// against this config-derived value.
    pub fn use_double_wide_mlp(&self, layer_idx: usize) -> bool {
        self.is_kv_shared_layer(layer_idx)
    }

    /// Number of RoPE dimensions rotated on global (full-attention) layers.
    pub fn global_rope_dim(&self) -> usize {
        (self.global_head_dim as f32 * self.partial_rotary_factor) as usize
    }
}

/// Compute the layer type pattern: every `interval`-th layer (1-indexed) is
/// full (global) attention, the rest are sliding. For interval=5 over 35
/// layers this yields global layers at indices 4, 9, 14, 19, 24, 29, 34 —
/// the header-verified ADR-082 G3 schedule.
pub(crate) fn compute_layer_types(num_layers: usize, interval: usize) -> Vec<Gemma4LayerType> {
    (0..num_layers)
        .map(|i| {
            if (i + 1) % interval == 0 {
                Gemma4LayerType::FullAttention
            } else {
                Gemma4LayerType::SlidingAttention
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2b_preset_matches_adr_082_dimensions() {
        let cfg = Gemma4Config::e2b();
        assert_eq!(cfg.num_hidden_layers, 35);
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.vocab_size, 262_144);
        assert_eq!(cfg.intermediate_size, 6144);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 1);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.global_head_dim, 512);
        assert_eq!(cfg.sliding_window, 512);
        assert_eq!(cfg.num_kv_shared_layers, 20);
        assert_eq!(cfg.hidden_size_per_layer_input, 256);
        assert!(cfg.tie_word_embeddings);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn e2b_preset_layer_schedule_matches_g3() {
        let cfg = Gemma4Config::e2b();
        let global_indices: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|&i| cfg.is_global_layer(i))
            .collect();
        assert_eq!(global_indices, vec![4, 9, 14, 19, 24, 29, 34]);
    }

    #[test]
    fn e2b_preset_kv_sharing_matches_amendment_1() {
        let cfg = Gemma4Config::e2b();
        assert_eq!(cfg.first_kv_shared_layer_idx(), 15);
        let shared: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|&i| cfg.is_kv_shared_layer(i))
            .collect();
        assert_eq!(shared.len(), 20);
        assert_eq!(shared[0], 15);
        assert_eq!(*shared.last().unwrap(), 34);
    }

    #[test]
    fn use_double_wide_mlp_matches_kv_shared_set() {
        let cfg = Gemma4Config::e2b();
        for i in 0..cfg.num_hidden_layers {
            assert_eq!(cfg.use_double_wide_mlp(i), cfg.is_kv_shared_layer(i));
        }
        assert_eq!(cfg.mlp_intermediate_size(0), 6144);
        assert_eq!(cfg.mlp_intermediate_size(15), 12288);
        assert_eq!(cfg.mlp_intermediate_size(34), 12288);
        assert_eq!(cfg.mlp_intermediate_size(14), 6144);
    }

    #[test]
    fn attn_head_dim_follows_layer_type_not_sharing() {
        let cfg = Gemma4Config::e2b();
        // Layer 4: global, non-shared -> global_head_dim.
        assert_eq!(cfg.attn_head_dim(4), 512);
        // Layer 15: sliding, shared -> head_dim (width is keyed on type, not sharing).
        assert_eq!(cfg.attn_head_dim(15), 256);
        // Layer 24: global, shared -> global_head_dim.
        assert_eq!(cfg.attn_head_dim(24), 512);
        // Layer 0: sliding, non-shared -> head_dim.
        assert_eq!(cfg.attn_head_dim(0), 256);
    }

    fn minimal_config_json() -> String {
        r#"{
            "text_config": {
                "hidden_size": 1536,
                "num_hidden_layers": 35,
                "vocab_size": 262144,
                "intermediate_size": 6144,
                "rms_norm_eps": 1e-6,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "global_head_dim": 512,
                "sliding_window": 512,
                "rope_theta": 1000000.0,
                "rope_local_base_freq": 10000.0,
                "partial_rotary_factor": 0.25,
                "sliding_window_pattern": 5,
                "num_kv_shared_layers": 20,
                "hidden_size_per_layer_input": 256,
                "eos_token_id": 1
            }
        }"#
        .to_string()
    }

    #[test]
    fn minimal_config_json_parses_and_derives_layer_types() {
        let cfg = Gemma4Config::from_config_json_str(&minimal_config_json())
            .expect("minimal Gemma 4 config.json parses");
        assert_eq!(cfg.layer_types.len(), 35);
        assert_eq!(cfg.first_kv_shared_layer_idx(), 15);
        let global_indices: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|&i| cfg.is_global_layer(i))
            .collect();
        assert_eq!(global_indices, vec![4, 9, 14, 19, 24, 29, 34]);
    }

    #[test]
    fn zero_sliding_window_pattern_errors_not_panics() {
        let mut json: serde_json::Value = serde_json::from_str(&minimal_config_json()).unwrap();
        json["text_config"]["sliding_window_pattern"] = serde_json::json!(0);
        json["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("layer_types");
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("sliding_window_pattern: 0 must yield an InferenceError, not panic");
        assert!(err.to_string().contains("sliding_window_pattern"));
    }

    #[test]
    fn zero_num_key_value_heads_errors_naming_field() {
        let mut json: serde_json::Value = serde_json::from_str(&minimal_config_json()).unwrap();
        json["text_config"]["num_key_value_heads"] = serde_json::json!(0);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("num_key_value_heads: 0 must yield an InferenceError");
        assert!(err.to_string().contains("num_key_value_heads"));
    }

    #[test]
    fn num_kv_shared_layers_exceeding_total_errors_naming_field() {
        let mut json: serde_json::Value = serde_json::from_str(&minimal_config_json()).unwrap();
        json["text_config"]["num_kv_shared_layers"] = serde_json::json!(36);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("num_kv_shared_layers > num_hidden_layers must yield an InferenceError");
        assert!(err.to_string().contains("num_kv_shared_layers"));
    }

    #[test]
    fn zero_head_dim_errors_naming_field() {
        let mut json: serde_json::Value = serde_json::from_str(&minimal_config_json()).unwrap();
        json["text_config"]["head_dim"] = serde_json::json!(0);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("head_dim: 0 must yield an InferenceError");
        assert!(err.to_string().contains("head_dim"));
    }

    #[test]
    fn zero_global_head_dim_errors_naming_field() {
        let mut json: serde_json::Value = serde_json::from_str(&minimal_config_json()).unwrap();
        json["text_config"]["global_head_dim"] = serde_json::json!(0);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("global_head_dim: 0 must yield an InferenceError");
        assert!(err.to_string().contains("global_head_dim"));
    }

    #[test]
    fn partial_rotary_factor_zero_errors_naming_field() {
        let mut json: serde_json::Value = serde_json::from_str(&minimal_config_json()).unwrap();
        json["text_config"]["partial_rotary_factor"] = serde_json::json!(0.0);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("partial_rotary_factor: 0.0 must yield an InferenceError");
        assert!(err.to_string().contains("partial_rotary_factor"));
    }

    #[test]
    fn partial_rotary_factor_above_one_errors_naming_field() {
        let mut json: serde_json::Value = serde_json::from_str(&minimal_config_json()).unwrap();
        json["text_config"]["partial_rotary_factor"] = serde_json::json!(1.5);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("partial_rotary_factor > 1.0 must yield an InferenceError");
        assert!(err.to_string().contains("partial_rotary_factor"));
    }

    #[test]
    fn from_model_dir_errors_on_missing_config_json() {
        let tmp = tempfile::tempdir().unwrap();
        let err = Gemma4Config::from_model_dir(tmp.path())
            .expect_err("a directory with no config.json must be a hard error");
        let msg = err.to_string();
        assert!(msg.contains("config.json"));
        assert!(msg.contains(&tmp.path().display().to_string()));
    }

    #[test]
    fn from_model_dir_loads_a_real_config_json() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("config.json"), minimal_config_json()).unwrap();
        let cfg = Gemma4Config::from_model_dir(tmp.path())
            .expect("a directory with a valid config.json must load");
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 35);
    }
}
