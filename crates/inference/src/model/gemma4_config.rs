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
//!
//! Parsing is fail-closed by construction: every forward-relevant field is
//! read from an explicit, non-defaulted raw struct that mirrors the pinned
//! target `config.json` (`google/gemma-4-E2B-it` @
//! `9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf`, fixture
//! `tests/fixtures/gemma4/e2b_config.json`) field-for-field, including the
//! two nested `rope_parameters` records. A missing or wrong-shaped field is a
//! serde error naming that field; there is no substitution of the E2B preset
//! for an absent or partial `text_config`.

use crate::error::InferenceError;
use std::path::Path;

/// Expected safetensors dtype for every Gemma 4 E2B language-model tensor
/// (header-verified, ADR-082 G16/G5).
pub const GEMMA4_EXPECTED_DTYPE: &str = "BF16";

/// The only `hidden_activation` value this loader supports (ADR-082 G7).
const EXPECTED_HIDDEN_ACTIVATION: &str = "gelu_pytorch_tanh";

/// The only `final_logit_softcapping` value this loader supports (ADR-082
/// G10, target `config.json`).
const EXPECTED_FINAL_LOGIT_SOFTCAPPING: f32 = 30.0;

/// Binding E2B global-layer schedule (ADR-082 G3, header/config-verified):
/// `full_attention` at exactly these zero-based indices, `sliding_attention`
/// everywhere else. This is an exact observable of the pinned checkpoint, not
/// a permissive default -- a checkpoint-provided `layer_types` that disagrees
/// at any position is rejected, never repaired.
const EXPECTED_GLOBAL_LAYER_INDICES: &[usize] = &[4, 9, 14, 19, 24, 29, 34];

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
/// model variants. Built exclusively via [`Self::from_config_json_str`] (or
/// its file-path wrappers) or [`Self::e2b`] -- there is no `Deserialize`
/// impl on this type itself, so a caller cannot bypass the fail-closed raw
/// parse in [`HfGemma4TextConfig`] by deserializing this struct directly.
#[derive(Debug, Clone)]
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
    /// `attention_k_eq_v` from the target config: whether K and V share
    /// weights. E2B is `false` (G4) -- the family-card's general "K=V"
    /// description does not apply to this checkpoint; `true` is unsupported
    /// by this loader and rejected in [`Self::validate`].
    pub attention_k_eq_v: bool,
    /// `attention_bias` from the target config. E2B is `false`; `true` is
    /// unsupported by this loader and rejected in [`Self::validate`].
    pub attention_bias: bool,

    // --- Dual RoPE (G8) ---
    /// Global-layer RoPE base, from `rope_parameters.full_attention.rope_theta`
    /// (proportional, over `partial_rotary_factor` of `global_head_dim`).
    pub rope_theta: f64,
    /// Sliding-layer RoPE base, from
    /// `rope_parameters.sliding_attention.rope_theta` (standard, full
    /// `head_dim`).
    pub rope_local_base_freq: f64,
    /// Fraction of `global_head_dim` rotated on global layers, from
    /// `rope_parameters.full_attention.partial_rotary_factor`. Sliding
    /// layers always use full rotary (`head_dim`, not scaled by this
    /// factor).
    pub partial_rotary_factor: f32,

    // --- Layer schedule (G3) ---
    /// Per-layer attention kind, length = `num_hidden_layers`. Parsed
    /// verbatim from the checkpoint's own `layer_types` array and checked in
    /// [`Self::validate`] against the exact binding E2B schedule -- never
    /// regenerated from a pattern.
    pub layer_types: Vec<Gemma4LayerType>,

    // --- KV sharing (G5, Amendment 1) ---
    /// Count of trailing decoder layers that functionally share K/V state
    /// with the last non-shared layer of their attention type. The
    /// checkpoint still carries (unused) `k_proj`/`v_proj`/`k_norm` weights
    /// on these layers — see [`Self::first_kv_shared_layer_idx`].
    pub num_kv_shared_layers: usize,

    /// Raw `use_double_wide_mlp` flag from the target config. Cross-checked
    /// in [`Self::validate`] against `num_kv_shared_layers`: this loader
    /// requires it `true` whenever any layer is KV-shared, since the
    /// reference runtime sets the double-wide MLP identically to
    /// `is_kv_shared_layer` (Amendment 1, "G2/G7 corroboration") and a raw
    /// `false` here would silently disagree with that invariant. See
    /// [`Self::use_double_wide_mlp`].
    pub use_double_wide_mlp_raw: bool,

    // --- Per-layer embeddings / PLE (G9) ---
    /// Width of the per-layer identity embedding gated into each layer.
    pub hidden_size_per_layer_input: usize,

    // --- Output (G10) ---
    /// MLP/activation nonlinearity name. Only `"gelu_pytorch_tanh"` is
    /// supported by this loader; any other value is rejected in
    /// [`Self::validate`].
    pub hidden_activation: String,
    /// Tanh soft-cap applied to final logits. Only `30.0` (the target
    /// config's value) is supported by this loader; any other value is
    /// rejected in [`Self::validate`].
    pub final_logit_softcapping: f32,

    // --- Embedding/output projection ---
    pub tie_word_embeddings: bool,

    // --- Generation ---
    pub eos_token_id: u32,
    pub max_position_embeddings: usize,
}

fn default_tie_word_embeddings() -> bool {
    true
}

fn default_max_position_embeddings() -> usize {
    131_072
}

/// Raw `text_config` shape, mirroring the pinned target `config.json`
/// field-for-field (`tests/fixtures/gemma4/e2b_config.json`). No
/// container-level `#[serde(default)]`: every field here that affects
/// forward-pass shape or numerics is required, so a missing or mistyped
/// field is a serde error naming that field rather than a silent
/// substitution. Only genuinely optional fields (`tie_word_embeddings`,
/// `max_position_embeddings`) carry an explicit field-level default.
#[derive(Debug, serde::Deserialize)]
struct HfGemma4TextConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    intermediate_size: usize,
    rms_norm_eps: f32,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    global_head_dim: usize,
    sliding_window: usize,
    attention_k_eq_v: bool,
    attention_bias: bool,
    rope_parameters: HfRopeParameters,
    layer_types: Vec<Gemma4LayerType>,
    num_kv_shared_layers: usize,
    use_double_wide_mlp: bool,
    hidden_size_per_layer_input: usize,
    hidden_activation: String,
    final_logit_softcapping: f32,
    eos_token_id: u32,
    #[serde(default = "default_tie_word_embeddings")]
    tie_word_embeddings: bool,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
}

/// Raw `text_config.rope_parameters`: two nested per-attention-type RoPE
/// records, as the target `config.json` actually nests them (ADR-082
/// Amendment 1) -- not the flat `rope_theta` /
/// `rope_local_base_freq` shape this loader previously (incorrectly)
/// expected at the top level of `text_config`.
#[derive(Debug, serde::Deserialize)]
struct HfRopeParameters {
    full_attention: HfRopeParamFullAttention,
    sliding_attention: HfRopeParamSlidingAttention,
}

#[derive(Debug, serde::Deserialize)]
struct HfRopeParamFullAttention {
    rope_theta: f64,
    partial_rotary_factor: f32,
}

#[derive(Debug, serde::Deserialize)]
struct HfRopeParamSlidingAttention {
    rope_theta: f64,
}

/// Private helper for the HF config.json structure: Gemma 4's text decoder
/// fields live under a required `text_config` object, a sibling of
/// `vision_config` / `audio_config` (out of scope here) at the top level —
/// the same wrapper shape as Qwen3.5's `HfQwenConfigFile` in
/// `qwen35_config.rs`. `text_config` is *not* `Option`: an absent
/// `text_config` (e.g. `{}`) is a hard parse error naming `text_config`,
/// never a silent substitution of the E2B preset.
#[derive(Debug, serde::Deserialize)]
struct HfGemma4ConfigFile {
    text_config: HfGemma4TextConfig,
}

impl Gemma4Config {
    /// **Unstable**: default `google/gemma-4-E2B-it` text-decoder configuration
    /// (ADR-082 G2-G9; layer-schedule and KV-sharing indices header-verified
    /// against the committed Stage-0 tensor manifest, PR #991). Used as a
    /// hardcoded starting point for tests and callers that want the known-good
    /// preset directly rather than parsing a `config.json`; parsing a real
    /// `config.json` never falls back to this value on missing/partial input.
    pub fn e2b() -> Self {
        let num_hidden_layers = 35;
        let layer_types = compute_layer_types(num_hidden_layers, 5);

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
            attention_k_eq_v: false,
            attention_bias: false,
            rope_theta: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            partial_rotary_factor: 0.25,
            layer_types,
            num_kv_shared_layers: 20,
            use_double_wide_mlp_raw: true,
            hidden_size_per_layer_input: 256,
            hidden_activation: EXPECTED_HIDDEN_ACTIVATION.to_string(),
            final_logit_softcapping: EXPECTED_FINAL_LOGIT_SOFTCAPPING,
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

    /// Parse HF Gemma 4 `config.json` text into a `Gemma4Config`. Fail-closed:
    /// `text_config` and every forward-relevant field within it are required
    /// by [`HfGemma4TextConfig`]'s schema (no container-level
    /// `#[serde(default)]`), so a missing or partial `text_config` is a
    /// serde error naming the absent field rather than a silently-completed
    /// E2B preset.
    pub fn from_config_json_str(json: &str) -> Result<Self, InferenceError> {
        let parsed: HfGemma4ConfigFile = serde_json::from_str(json)
            .map_err(|e| InferenceError::Inference(format!("invalid Gemma 4 config.json: {e}")))?;
        let raw = parsed.text_config;

        let cfg = Gemma4Config {
            hidden_size: raw.hidden_size,
            num_hidden_layers: raw.num_hidden_layers,
            vocab_size: raw.vocab_size,
            intermediate_size: raw.intermediate_size,
            rms_norm_eps: raw.rms_norm_eps,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            global_head_dim: raw.global_head_dim,
            sliding_window: raw.sliding_window,
            attention_k_eq_v: raw.attention_k_eq_v,
            attention_bias: raw.attention_bias,
            rope_theta: raw.rope_parameters.full_attention.rope_theta,
            rope_local_base_freq: raw.rope_parameters.sliding_attention.rope_theta,
            partial_rotary_factor: raw.rope_parameters.full_attention.partial_rotary_factor,
            layer_types: raw.layer_types,
            num_kv_shared_layers: raw.num_kv_shared_layers,
            use_double_wide_mlp_raw: raw.use_double_wide_mlp,
            hidden_size_per_layer_input: raw.hidden_size_per_layer_input,
            hidden_activation: raw.hidden_activation,
            final_logit_softcapping: raw.final_logit_softcapping,
            tie_word_embeddings: raw.tie_word_embeddings,
            eos_token_id: raw.eos_token_id,
            max_position_embeddings: raw.max_position_embeddings,
        };

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
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: head_dim ({}) must be even and > 0 \
                 (stride-half RoPE pairs (i, head_dim/2 + i))",
                self.head_dim
            )));
        }
        if self.global_head_dim == 0 || !self.global_head_dim.is_multiple_of(2) {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: global_head_dim ({}) must be even and > 0 \
                 (stride-half RoPE pairs (i, head_dim/2 + i))",
                self.global_head_dim
            )));
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
        // layer_types is an exact observable of the pinned E2B checkpoint
        // (ADR-082 G3), not a repairable default: length must match
        // num_hidden_layers AND the global-layer positions must match the
        // binding schedule exactly. A same-length schedule with a global
        // layer at the wrong index is rejected here, naming layer_types --
        // it is never regenerated from a pattern.
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: layer_types has {} entries but \
                 num_hidden_layers is {}",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        let global_indices: Vec<usize> = (0..self.layer_types.len())
            .filter(|&i| self.layer_types[i] == Gemma4LayerType::FullAttention)
            .collect();
        if global_indices != EXPECTED_GLOBAL_LAYER_INDICES {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: layer_types full_attention indices {global_indices:?} \
                 do not match the binding Gemma 4 E2B schedule {EXPECTED_GLOBAL_LAYER_INDICES:?}"
            )));
        }
        if self.num_kv_shared_layers > self.num_hidden_layers {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: num_kv_shared_layers ({}) must be <= \
                 num_hidden_layers ({})",
                self.num_kv_shared_layers, self.num_hidden_layers
            )));
        }
        // ADR-082 Amendment 1: use_double_wide_mlp must not be a
        // tautological function of is_kv_shared_layer. The raw checkpoint
        // flag is required to agree with the derived KV-shared set: the
        // reference runtime sets it identically (Amendment 1, "G2/G7
        // corroboration"), so a raw `false` while any layer is KV-shared is
        // a structural config/checkpoint disagreement, not a valid variant.
        if self.num_kv_shared_layers > 0 && !self.use_double_wide_mlp_raw {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: use_double_wide_mlp is false but \
                 num_kv_shared_layers ({}) implies is_kv_shared_layer is true for {} layer(s) -- \
                 Gemma 4 E2B requires use_double_wide_mlp=true whenever KV-shared layers exist",
                self.num_kv_shared_layers, self.num_kv_shared_layers
            )));
        }
        if self.hidden_activation != EXPECTED_HIDDEN_ACTIVATION {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: hidden_activation ({:?}) is unsupported -- only \
                 {EXPECTED_HIDDEN_ACTIVATION:?} is implemented",
                self.hidden_activation
            )));
        }
        if self.final_logit_softcapping != EXPECTED_FINAL_LOGIT_SOFTCAPPING {
            return Err(InferenceError::Inference(format!(
                "invalid Gemma 4 config.json: final_logit_softcapping ({}) is unsupported -- \
                 only {EXPECTED_FINAL_LOGIT_SOFTCAPPING} is implemented",
                self.final_logit_softcapping
            )));
        }
        if self.attention_k_eq_v {
            return Err(InferenceError::Inference(
                "invalid Gemma 4 config.json: attention_k_eq_v=true is unsupported -- the \
                 Gemma 4 E2B checkpoint this loader targets has attention_k_eq_v=false"
                    .to_string(),
            ));
        }
        if self.attention_bias {
            return Err(InferenceError::Inference(
                "invalid Gemma 4 config.json: attention_bias=true is unsupported -- the \
                 Gemma 4 E2B checkpoint this loader targets has attention_bias=false"
                    .to_string(),
            ));
        }
        // The forward pass always projects logits through `embed_tokens`
        // (the tied convention) and the loader never requests a distinct
        // `lm_head.weight`. tie_word_embeddings=false on an otherwise
        // E2B-shaped config would load successfully and silently produce
        // wrong logits by using the input embedding table as the output
        // head. Reject it here rather than load-then-diverge.
        if !self.tie_word_embeddings {
            return Err(InferenceError::Inference(
                "invalid Gemma 4 config.json: tie_word_embeddings=false is unsupported -- this \
                 loader only implements the tied embed_tokens/lm_head projection the pinned \
                 Gemma 4 E2B checkpoint uses; an untied lm_head.weight is never read"
                    .to_string(),
            ));
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
    /// MLP: the raw checkpoint `use_double_wide_mlp` flag AND `layer_idx`
    /// being KV-shared. Not a tautological restatement of
    /// [`Self::is_kv_shared_layer`] -- [`Self::validate`] rejects any config
    /// where the raw flag disagrees with the KV-shared set, so by the time
    /// this is called on a validated config the two conjuncts already agree,
    /// but the raw flag is still an observed input, not a derived constant.
    /// [`crate::model::gemma4_preflight::preflight_check`] cross-checks this
    /// against the checkpoint's own `mlp.*` shapes as a second, independent
    /// structural observable (Amendment 1, "G2/G7 corroboration").
    pub fn use_double_wide_mlp(&self, layer_idx: usize) -> bool {
        self.use_double_wide_mlp_raw && self.is_kv_shared_layer(layer_idx)
    }

    /// Number of RoPE dimensions rotated on global (full-attention) layers.
    pub fn global_rope_dim(&self) -> usize {
        (self.global_head_dim as f32 * self.partial_rotary_factor) as usize
    }
}

/// Compute the layer type pattern: every `interval`-th layer (1-indexed) is
/// full (global) attention, the rest are sliding. For interval=5 over 35
/// layers this yields global layers at indices 4, 9, 14, 19, 24, 29, 34 —
/// the header-verified ADR-082 G3 schedule. Used only to build the hardcoded
/// [`Gemma4Config::e2b`] preset -- checkpoint-provided `layer_types` is never
/// regenerated this way (see [`Gemma4Config::validate`]).
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
    use std::path::PathBuf;

    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("gemma4")
            .join("e2b_config.json")
    }

    fn pinned_config_json() -> String {
        std::fs::read_to_string(fixture_path()).expect("read committed pinned target config.json")
    }

    fn pinned_config_value() -> serde_json::Value {
        serde_json::from_str(&pinned_config_json()).expect("pinned fixture is valid JSON")
    }

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

    #[test]
    fn pinned_config_fixture_parses_and_matches_adr_082() {
        let cfg = Gemma4Config::from_config_json_str(&pinned_config_json())
            .expect("committed pinned google/gemma-4-E2B-it config.json must parse");
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 35);
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
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert_eq!(cfg.rope_local_base_freq, 10_000.0);
        assert_eq!(cfg.partial_rotary_factor, 0.25);
        assert!(cfg.use_double_wide_mlp_raw);
        assert_eq!(cfg.hidden_activation, "gelu_pytorch_tanh");
        assert_eq!(cfg.final_logit_softcapping, 30.0);
        assert!(!cfg.attention_k_eq_v);
        assert!(!cfg.attention_bias);
        let global_indices: Vec<usize> = (0..cfg.num_hidden_layers)
            .filter(|&i| cfg.is_global_layer(i))
            .collect();
        assert_eq!(global_indices, vec![4, 9, 14, 19, 24, 29, 34]);
    }

    #[test]
    fn absent_text_config_errors_naming_field() {
        let err = Gemma4Config::from_config_json_str("{}")
            .expect_err("an absent text_config must yield an InferenceError, not the E2B preset");
        assert!(
            err.to_string().contains("text_config"),
            "error must name text_config: {err}"
        );
    }

    #[test]
    fn partial_text_config_errors_naming_missing_field() {
        // A text_config present but missing a forward-relevant field (here
        // rope_parameters) must be rejected naming that field, not silently
        // completed from the E2B preset.
        let json = r#"{"text_config": {"hidden_size": 1536}}"#;
        let err = Gemma4Config::from_config_json_str(json)
            .expect_err("a partial text_config must yield an InferenceError");
        assert!(
            err.to_string().contains("rope_parameters")
                || err.to_string().contains("num_hidden_layers"),
            "error must name a missing required field: {err}"
        );
    }

    #[test]
    fn missing_nested_rope_full_attention_errors_naming_field() {
        let mut json = pinned_config_value();
        json["text_config"]["rope_parameters"]
            .as_object_mut()
            .unwrap()
            .remove("full_attention");
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("a missing rope_parameters.full_attention must yield an InferenceError");
        assert!(
            err.to_string().contains("full_attention"),
            "error must name full_attention: {err}"
        );
    }

    #[test]
    fn missing_nested_rope_sliding_attention_theta_errors_naming_field() {
        let mut json = pinned_config_value();
        json["text_config"]["rope_parameters"]["sliding_attention"]
            .as_object_mut()
            .unwrap()
            .remove("rope_theta");
        let err = Gemma4Config::from_config_json_str(&json.to_string()).expect_err(
            "a missing rope_parameters.sliding_attention.rope_theta must yield an InferenceError",
        );
        assert!(
            err.to_string().contains("rope_theta"),
            "error must name rope_theta: {err}"
        );
    }

    #[test]
    fn nested_rope_theta_mutation_is_observed_not_ignored() {
        let mut json = pinned_config_value();
        json["text_config"]["rope_parameters"]["full_attention"]["rope_theta"] =
            serde_json::json!(42.0);
        let cfg = Gemma4Config::from_config_json_str(&json.to_string())
            .expect("a plausible rope_theta mutation still parses");
        assert_eq!(
            cfg.rope_theta, 42.0,
            "nested rope_parameters.full_attention.rope_theta must flow through, not be \
             silently replaced by the E2B preset's 1_000_000.0"
        );
    }

    #[test]
    fn use_double_wide_mlp_false_errors_naming_both_fields() {
        let mut json = pinned_config_value();
        json["text_config"]["use_double_wide_mlp"] = serde_json::json!(false);
        let err = Gemma4Config::from_config_json_str(&json.to_string()).expect_err(
            "use_double_wide_mlp=false while KV-shared layers exist must yield an InferenceError",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("use_double_wide_mlp"),
            "must name use_double_wide_mlp: {msg}"
        );
        assert!(
            msg.contains("is_kv_shared_layer") || msg.contains("num_kv_shared_layers"),
            "must name the KV-shared-layer field: {msg}"
        );
    }

    #[test]
    fn odd_head_dim_errors_naming_field() {
        // gemma4_apply_rope's stride-half pairing requires even widths; an
        // odd head_dim must die here with a typed error, not deep in the
        // forward pass at the op-level assert.
        let mut json = pinned_config_value();
        json["text_config"]["head_dim"] = serde_json::json!(255);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("an odd head_dim must yield an InferenceError");
        let msg = err.to_string();
        assert!(
            msg.contains("head_dim (255)") && msg.contains("even"),
            "error must name head_dim and evenness: {msg}"
        );
    }

    #[test]
    fn odd_global_head_dim_errors_naming_field() {
        let mut json = pinned_config_value();
        json["text_config"]["global_head_dim"] = serde_json::json!(511);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("an odd global_head_dim must yield an InferenceError");
        let msg = err.to_string();
        assert!(
            msg.contains("global_head_dim (511)") && msg.contains("even"),
            "error must name global_head_dim and evenness: {msg}"
        );
    }

    #[test]
    fn wrong_hidden_activation_errors_naming_field() {
        let mut json = pinned_config_value();
        json["text_config"]["hidden_activation"] = serde_json::json!("gelu");
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("an unsupported hidden_activation must yield an InferenceError");
        assert!(
            err.to_string().contains("hidden_activation"),
            "error must name hidden_activation: {err}"
        );
    }

    #[test]
    fn wrong_final_logit_softcapping_errors_naming_field() {
        let mut json = pinned_config_value();
        json["text_config"]["final_logit_softcapping"] = serde_json::json!(50.0);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("an unsupported final_logit_softcapping must yield an InferenceError");
        assert!(
            err.to_string().contains("final_logit_softcapping"),
            "error must name final_logit_softcapping: {err}"
        );
    }

    #[test]
    fn tie_word_embeddings_false_errors_naming_field() {
        let mut json = pinned_config_value();
        json["text_config"]["tie_word_embeddings"] = serde_json::json!(false);
        let err = Gemma4Config::from_config_json_str(&json.to_string()).expect_err(
            "tie_word_embeddings=false must yield an InferenceError: this loader never reads \
             a distinct lm_head.weight",
        );
        assert!(
            err.to_string().contains("tie_word_embeddings"),
            "error must name tie_word_embeddings: {err}"
        );
    }

    #[test]
    fn attention_k_eq_v_true_errors_naming_field() {
        let mut json = pinned_config_value();
        json["text_config"]["attention_k_eq_v"] = serde_json::json!(true);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("attention_k_eq_v=true must yield an InferenceError for this checkpoint");
        assert!(
            err.to_string().contains("attention_k_eq_v"),
            "error must name attention_k_eq_v: {err}"
        );
    }

    #[test]
    fn attention_bias_true_errors_naming_field() {
        let mut json = pinned_config_value();
        json["text_config"]["attention_bias"] = serde_json::json!(true);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("attention_bias=true must yield an InferenceError for this checkpoint");
        assert!(
            err.to_string().contains("attention_bias"),
            "error must name attention_bias: {err}"
        );
    }

    #[test]
    fn layer_types_length_34_errors_naming_field() {
        let mut json = pinned_config_value();
        let arr = json["text_config"]["layer_types"].as_array_mut().unwrap();
        arr.pop();
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("a length-34 layer_types must yield an InferenceError");
        assert!(
            err.to_string().contains("layer_types"),
            "error must name layer_types: {err}"
        );
    }

    #[test]
    fn layer_types_length_36_errors_naming_field() {
        let mut json = pinned_config_value();
        let arr = json["text_config"]["layer_types"].as_array_mut().unwrap();
        arr.push(serde_json::json!("sliding_attention"));
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("a length-36 layer_types must yield an InferenceError");
        assert!(
            err.to_string().contains("layer_types"),
            "error must name layer_types: {err}"
        );
    }

    #[test]
    fn layer_types_wrong_first_global_index_errors_naming_field() {
        let mut json = pinned_config_value();
        let arr = json["text_config"]["layer_types"].as_array_mut().unwrap();
        // Move the first global layer from index 4 to index 3.
        arr[3] = serde_json::json!("full_attention");
        arr[4] = serde_json::json!("sliding_attention");
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("a layer_types with the wrong first global index must be rejected");
        assert!(
            err.to_string().contains("layer_types"),
            "error must name layer_types: {err}"
        );
    }

    #[test]
    fn layer_types_wrong_final_layer_type_errors_naming_field() {
        let mut json = pinned_config_value();
        let arr = json["text_config"]["layer_types"].as_array_mut().unwrap();
        // Layer 34 must be full_attention (final, forced global); flip it.
        let last = arr.len() - 1;
        arr[last] = serde_json::json!("sliding_attention");
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("a layer_types with the wrong final layer type must be rejected");
        assert!(
            err.to_string().contains("layer_types"),
            "error must name layer_types: {err}"
        );
    }

    #[test]
    fn num_kv_shared_layers_exceeding_total_errors_naming_field() {
        let mut json = pinned_config_value();
        json["text_config"]["num_kv_shared_layers"] = serde_json::json!(36);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("num_kv_shared_layers > num_hidden_layers must yield an InferenceError");
        assert!(err.to_string().contains("num_kv_shared_layers"));
    }

    #[test]
    fn zero_head_dim_errors_naming_field() {
        let mut json = pinned_config_value();
        json["text_config"]["head_dim"] = serde_json::json!(0);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("head_dim: 0 must yield an InferenceError");
        assert!(err.to_string().contains("head_dim"));
    }

    #[test]
    fn zero_global_head_dim_errors_naming_field() {
        let mut json = pinned_config_value();
        json["text_config"]["global_head_dim"] = serde_json::json!(0);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("global_head_dim: 0 must yield an InferenceError");
        assert!(err.to_string().contains("global_head_dim"));
    }

    #[test]
    fn partial_rotary_factor_zero_errors_naming_field() {
        let mut json = pinned_config_value();
        json["text_config"]["rope_parameters"]["full_attention"]["partial_rotary_factor"] =
            serde_json::json!(0.0);
        let err = Gemma4Config::from_config_json_str(&json.to_string())
            .expect_err("partial_rotary_factor: 0.0 must yield an InferenceError");
        assert!(err.to_string().contains("partial_rotary_factor"));
    }

    #[test]
    fn partial_rotary_factor_above_one_errors_naming_field() {
        let mut json = pinned_config_value();
        json["text_config"]["rope_parameters"]["full_attention"]["partial_rotary_factor"] =
            serde_json::json!(1.5);
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
        std::fs::write(tmp.path().join("config.json"), pinned_config_json()).unwrap();
        let cfg = Gemma4Config::from_model_dir(tmp.path())
            .expect("a directory with a valid config.json must load");
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.num_hidden_layers, 35);
    }
}
