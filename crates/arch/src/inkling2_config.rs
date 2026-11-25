//! Parsed and validated `config.json` for the Inkling 2 looped models (`inkling2_loop`).
//!
//! Only the text tower is parsed. The vision tower, the multi-token-prediction head and the
//! quantization block are recognised and left unparsed, and `Inkling2Config::unparsed_sections`
//! names the ones present. Several text fields are never read by the model definition this
//! parser follows: it builds every projection without bias, never adds a shared expert, never
//! weights the value norm, and derives the attention span of each layer from its index. Those
//! fields are therefore accepted only with the value that matches that behaviour, so a
//! checkpoint relying on a different value is refused instead of being run with the wrong
//! semantics. Geometry fields are required: lattice does not fill in a missing dimension from a
//! default.

use std::path::Path;

use lattice_inference_core::InferenceError;
use lattice_weights::bounded_config::{MAX_CONFIG_JSON_BYTES, read_config_json_bounded};
use serde::Deserialize;
use serde_json::Value;

use crate::deepseek_v4_inspection::BoundedValue;

const MAX_GEOMETRY: usize = 16_777_216;
const MAX_LOOP_PASSES: usize = 64;
const MAX_CONV_KERNEL: usize = 64;
/// Every sixth layer, counting from one, uses full attention; the others use a sliding window.
const GLOBAL_LAYER_PERIOD: usize = 6;
const DEFAULT_IMAGE_TOKEN_ID: usize = 200_054;

/// Attention span of one decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionSpan {
    /// Causal attention over the whole context.
    Global,
    /// Causal attention limited to the sliding window.
    Sliding,
}

/// Head geometry of one attention span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadGeometry {
    /// Query heads.
    pub num_attention_heads: usize,
    /// Key/value heads; divides `num_attention_heads`.
    pub num_key_value_heads: usize,
    /// Width of one head.
    pub head_dim: usize,
}

/// The looped core: layers `core_start..core_end` run `num_passes` times in sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopConfig {
    /// First looped layer; a positive multiple of the attention-residual block length.
    pub core_start: usize,
    /// One past the last looped layer; a multiple of the attention-residual block length.
    pub core_end: usize,
    /// Number of passes over the looped layers, at least one.
    pub num_passes: usize,
    /// Rank of the gate that mixes one pass into the next.
    pub gate_rank: usize,
}

/// Validated text-tower configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2TextConfig {
    /// Hidden width.
    pub hidden_size: usize,
    /// Stored decoder layers, not counting repeated passes.
    pub num_hidden_layers: usize,
    /// Rows of the stored embedding and unembedding, including padding rows.
    pub vocab_size: usize,
    /// Logits the model produces; rows from here to `vocab_size` are padding.
    pub unpadded_vocab_size: usize,
    /// Head geometry of the full-attention layers.
    pub global_heads: HeadGeometry,
    /// Head geometry of the sliding-window layers.
    pub sliding_heads: HeadGeometry,
    /// Sliding window length in tokens.
    pub sliding_window_size: usize,
    /// Width of the relative-position projection.
    pub d_rel: usize,
    /// Relative-position extent of the full-attention layers.
    pub rel_extent: usize,
    /// Context length below which the full-attention query scale is not raised.
    pub log_scaling_n_floor: usize,
    /// Strength of the full-attention query scale growth with context length.
    pub log_scaling_alpha: f64,
    /// RMS norm epsilon.
    pub rms_norm_eps: f64,
    /// L2 bound applied to queries and keys.
    pub qk_l2_clip_threshold: f64,
    /// L2 bound applied to the relative-position projection.
    pub rel_l2_clip_threshold: f64,
    /// Taps of the short convolutions.
    pub conv_kernel_size: usize,
    /// Tanh softcap on the final logits.
    pub final_logit_softcapping: f64,
    /// Layers per attention-residual block.
    pub n_layers_per_attn_res_block: usize,
    /// Routed experts per layer.
    pub n_routed_experts: usize,
    /// Experts selected per token.
    pub num_experts_per_tok: usize,
    /// Expert intermediate width.
    pub moe_intermediate_size: usize,
    /// Softcap on the expert gate projection.
    pub mlp_gate_softcap: f64,
    /// Softcap on the expert up projection.
    pub mlp_up_softcap: f64,
    /// Factor applied to the normalised routing weights.
    pub route_scale: f64,
    /// Declared context maximum, not an allocated capacity.
    pub max_position_embeddings: usize,
    /// Attention span of each stored layer, by index.
    pub layer_spans: Vec<AttentionSpan>,
    /// The looped core.
    pub loop_config: LoopConfig,
}

impl Inkling2TextConfig {
    /// Decoder layers applied in one forward pass, counting every pass of the looped core.
    pub fn total_layer_applications(&self) -> usize {
        let looped = self.loop_config.core_end - self.loop_config.core_start;
        self.num_hidden_layers + (self.loop_config.num_passes - 1) * looped
    }

    /// Attention span of stored layer `layer`, or `None` past the last layer.
    pub fn span(&self, layer: usize) -> Option<AttentionSpan> {
        self.layer_spans.get(layer).copied()
    }

    /// Head geometry used by layers of the given span.
    pub fn heads(&self, span: AttentionSpan) -> HeadGeometry {
        match span {
            AttentionSpan::Global => self.global_heads,
            AttentionSpan::Sliding => self.sliding_heads,
        }
    }

    /// Relative-position extent used by layers of the given span: the sliding window for
    /// sliding layers, `rel_extent` for full-attention layers.
    pub fn relative_extent(&self, span: AttentionSpan) -> usize {
        match span {
            AttentionSpan::Global => self.rel_extent,
            AttentionSpan::Sliding => self.sliding_window_size,
        }
    }

    /// Whether stored layer `layer` opens a new attention-residual block.
    pub fn starts_attention_residual_block(&self, layer: usize) -> bool {
        layer.is_multiple_of(self.n_layers_per_attn_res_block)
    }
}

/// Validated model configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2Config {
    /// The text tower.
    pub text: Inkling2TextConfig,
    /// End-of-sequence token.
    pub eos_token_id: usize,
    /// Image placeholder token.
    pub image_token_id: usize,
    /// Top-level sections present in the file that this parser does not read.
    pub unparsed_sections: Vec<String>,
}

/// Parse and validate the `config.json` at `path` using the shared bounded reader.
pub fn parse_config(path: &Path) -> Result<Inkling2Config, InferenceError> {
    let text = read_config_json_bounded(path, "Inkling 2 config")?;
    parse_config_json(&text)
}

/// Parse and validate config JSON text.
pub fn parse_config_json(text: &str) -> Result<Inkling2Config, InferenceError> {
    if text.len() as u64 > MAX_CONFIG_JSON_BYTES {
        return Err(error("config exceeds MAX_CONFIG_JSON_BYTES"));
    }
    let BoundedValue(value): BoundedValue =
        serde_json::from_str(text).map_err(|reason| error(&format!("invalid JSON: {reason}")))?;
    let root: RawRoot =
        serde_json::from_value(value).map_err(|reason| error(&format!("{reason}")))?;
    root.validate()
}

fn error(message: &str) -> InferenceError {
    InferenceError::Inference(format!("Inkling 2 config: {message}"))
}

fn expect_flag(name: &str, value: Option<bool>, required: bool) -> Result<(), InferenceError> {
    match value {
        Some(found) if found != required => Err(error(&format!(
            "{name} = {found} is not supported; only {required} is implemented"
        ))),
        _ => Ok(()),
    }
}

fn derived_span(layer: usize) -> AttentionSpan {
    if (layer + 1).is_multiple_of(GLOBAL_LAYER_PERIOD) {
        AttentionSpan::Global
    } else {
        AttentionSpan::Sliding
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRoot {
    architectures: Vec<String>,
    model_type: String,
    dtype: Option<String>,
    eos_token_id: usize,
    image_token_id: Option<usize>,
    text_config: RawText,
    vision_config: Option<Value>,
    mtp_config: Option<Value>,
    quantization_config: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawText {
    model_type: Option<String>,
    torch_dtype: Option<String>,
    #[serde(alias = "model_max_length")]
    max_position_embeddings: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    unpadded_vocab_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    swa_num_attention_heads: usize,
    swa_num_key_value_heads: usize,
    swa_head_dim: usize,
    #[serde(alias = "sliding_window")]
    sliding_window_size: usize,
    d_rel: usize,
    rel_extent: usize,
    log_scaling_n_floor: usize,
    log_scaling_alpha: f64,
    rms_norm_eps: f64,
    qk_l2_clip_threshold: f64,
    rel_l2_clip_threshold: f64,
    #[serde(alias = "sconv_kernel_size")]
    conv_kernel_size: usize,
    final_logit_softcapping: f64,
    n_layers_per_attn_res_block: usize,
    #[serde(alias = "num_local_experts")]
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    #[serde(alias = "intermediate_size")]
    moe_intermediate_size: usize,
    mlp_gate_softcap: f64,
    mlp_up_softcap: f64,
    #[serde(alias = "routed_scaling_factor")]
    route_scale: f64,
    scoring_func: Option<String>,
    hidden_act: Option<String>,
    layer_types: Option<Vec<String>>,
    local_layer_ids: Option<Vec<usize>>,
    q_bias: Option<bool>,
    o_bias: Option<bool>,
    use_value_norm_no_gamma: Option<bool>,
    attn_res_normalize_values: Option<bool>,
    use_gate_bias: Option<bool>,
    n_shared_experts: Option<usize>,
    loop_config: RawLoop,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLoop {
    core_start: usize,
    core_end: usize,
    num_passes: usize,
    gate_rank: usize,
    is_varloop: Option<bool>,
}

impl RawRoot {
    fn validate(self) -> Result<Inkling2Config, InferenceError> {
        if self.model_type != "inkling2_loop"
            || self.architectures.as_slice() != ["Inkling2LoopForConditionalGeneration"]
        {
            return Err(error("unexpected model_type or architecture"));
        }
        if self.dtype.as_deref().is_some_and(str::is_empty) {
            return Err(error("dtype must not be empty"));
        }
        let text = self.text_config.validate()?;
        if self.eos_token_id >= text.unpadded_vocab_size {
            return Err(error("eos_token_id outside the unpadded vocabulary"));
        }
        let image_token_id = self.image_token_id.unwrap_or(DEFAULT_IMAGE_TOKEN_ID);
        if image_token_id >= text.unpadded_vocab_size {
            return Err(error("image_token_id outside the unpadded vocabulary"));
        }
        let mut unparsed_sections = Vec::new();
        if let Some(vision) = &self.vision_config {
            let declared = vision
                .get("text_hidden_size")
                .and_then(Value::as_u64)
                .ok_or_else(|| error("vision_config.text_hidden_size is missing"))?;
            if declared != text.hidden_size as u64 {
                return Err(error(
                    "vision_config.text_hidden_size does not match text_config.hidden_size",
                ));
            }
            unparsed_sections.push("vision_config".to_owned());
        }
        if self.mtp_config.is_some() {
            unparsed_sections.push("mtp_config".to_owned());
        }
        if self.quantization_config.is_some() {
            unparsed_sections.push("quantization_config".to_owned());
        }
        Ok(Inkling2Config {
            text,
            eos_token_id: self.eos_token_id,
            image_token_id,
            unparsed_sections,
        })
    }
}

impl RawText {
    fn validate(self) -> Result<Inkling2TextConfig, InferenceError> {
        if self
            .model_type
            .as_deref()
            .is_some_and(|kind| kind != "inkling2_loop_text")
        {
            return Err(error("unexpected text_config.model_type"));
        }
        if self.torch_dtype.as_deref().is_some_and(str::is_empty) {
            return Err(error("text_config.torch_dtype must not be empty"));
        }
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("vocab_size", self.vocab_size),
            ("unpadded_vocab_size", self.unpadded_vocab_size),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("swa_num_attention_heads", self.swa_num_attention_heads),
            ("swa_num_key_value_heads", self.swa_num_key_value_heads),
            ("swa_head_dim", self.swa_head_dim),
            ("sliding_window_size", self.sliding_window_size),
            ("d_rel", self.d_rel),
            ("rel_extent", self.rel_extent),
            ("log_scaling_n_floor", self.log_scaling_n_floor),
            ("conv_kernel_size", self.conv_kernel_size),
            (
                "n_layers_per_attn_res_block",
                self.n_layers_per_attn_res_block,
            ),
            ("n_routed_experts", self.n_routed_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("max_position_embeddings", self.max_position_embeddings),
            ("loop_config.gate_rank", self.loop_config.gate_rank),
        ] {
            if value == 0 {
                return Err(error(&format!("{name} must be positive")));
            }
            if value > MAX_GEOMETRY {
                return Err(error(&format!("{name} exceeds the supported geometry")));
            }
        }
        for (name, value) in [
            ("log_scaling_alpha", self.log_scaling_alpha),
            ("rms_norm_eps", self.rms_norm_eps),
            ("qk_l2_clip_threshold", self.qk_l2_clip_threshold),
            ("rel_l2_clip_threshold", self.rel_l2_clip_threshold),
            ("final_logit_softcapping", self.final_logit_softcapping),
            ("mlp_gate_softcap", self.mlp_gate_softcap),
            ("mlp_up_softcap", self.mlp_up_softcap),
            ("route_scale", self.route_scale),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(error(&format!("{name} must be finite and positive")));
            }
        }
        if self.conv_kernel_size > MAX_CONV_KERNEL {
            return Err(error(
                "conv_kernel_size exceeds the supported kernel length",
            ));
        }
        if self.unpadded_vocab_size > self.vocab_size {
            return Err(error("unpadded_vocab_size exceeds vocab_size"));
        }
        if self.num_experts_per_tok > self.n_routed_experts {
            return Err(error("num_experts_per_tok exceeds n_routed_experts"));
        }
        let global_heads = heads(
            "",
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )?;
        let sliding_heads = heads(
            "swa_",
            self.swa_num_attention_heads,
            self.swa_num_key_value_heads,
            self.swa_head_dim,
        )?;
        self.vocab_size
            .checked_mul(self.hidden_size)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| error("embedding byte count overflow"))?;
        self.n_routed_experts
            .checked_mul(self.moe_intermediate_size)
            .and_then(|n| n.checked_mul(self.hidden_size))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| error("expert byte count overflow"))?;

        let scoring = self.scoring_func.as_deref().unwrap_or("sigmoid");
        if scoring != "sigmoid" {
            return Err(error(&format!(
                "scoring_func {scoring:?} is not supported; only \"sigmoid\" is implemented"
            )));
        }
        let activation = self.hidden_act.as_deref().unwrap_or("silu");
        if activation != "silu" {
            return Err(error(&format!(
                "hidden_act {activation:?} is not supported; only \"silu\" is implemented"
            )));
        }
        expect_flag("q_bias", self.q_bias, false)?;
        expect_flag("o_bias", self.o_bias, false)?;
        expect_flag(
            "use_value_norm_no_gamma",
            self.use_value_norm_no_gamma,
            true,
        )?;
        expect_flag(
            "attn_res_normalize_values",
            self.attn_res_normalize_values,
            true,
        )?;
        expect_flag("use_gate_bias", self.use_gate_bias, true)?;
        expect_flag("loop_config.is_varloop", self.loop_config.is_varloop, false)?;
        if self.n_shared_experts.is_some_and(|n| n != 0) {
            return Err(error(
                "n_shared_experts must be 0; shared experts are not implemented",
            ));
        }

        let layer_spans: Vec<AttentionSpan> =
            (0..self.num_hidden_layers).map(derived_span).collect();
        if let Some(types) = &self.layer_types {
            let matches = types.len() == layer_spans.len()
                && types.iter().zip(&layer_spans).all(|(kind, span)| {
                    matches!(
                        (kind.as_str(), span),
                        ("hybrid", AttentionSpan::Global)
                            | ("hybrid_sliding", AttentionSpan::Sliding)
                    )
                });
            if !matches {
                return Err(error(
                    "layer_types must place full attention on every sixth layer",
                ));
            }
        }
        if let Some(local) = &self.local_layer_ids {
            let expected = layer_spans
                .iter()
                .enumerate()
                .filter(|(_, span)| **span == AttentionSpan::Sliding)
                .map(|(layer, _)| layer);
            if !local.iter().copied().eq(expected) {
                return Err(error(
                    "local_layer_ids must list exactly the layers that are not every sixth layer",
                ));
            }
        }

        let raw_loop = self.loop_config;
        let block = self.n_layers_per_attn_res_block;
        if raw_loop.num_passes == 0 || raw_loop.num_passes > MAX_LOOP_PASSES {
            return Err(error("loop_config.num_passes must be between 1 and 64"));
        }
        if raw_loop.core_start == 0 {
            return Err(error("loop_config.core_start must be positive"));
        }
        if raw_loop.core_start >= raw_loop.core_end || raw_loop.core_end > self.num_hidden_layers {
            return Err(error(
                "loop_config must satisfy core_start < core_end <= num_hidden_layers",
            ));
        }
        if !raw_loop.core_start.is_multiple_of(block) || !raw_loop.core_end.is_multiple_of(block) {
            return Err(error(
                "loop_config core_start and core_end must be multiples of n_layers_per_attn_res_block",
            ));
        }

        Ok(Inkling2TextConfig {
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            vocab_size: self.vocab_size,
            unpadded_vocab_size: self.unpadded_vocab_size,
            global_heads,
            sliding_heads,
            sliding_window_size: self.sliding_window_size,
            d_rel: self.d_rel,
            rel_extent: self.rel_extent,
            log_scaling_n_floor: self.log_scaling_n_floor,
            log_scaling_alpha: self.log_scaling_alpha,
            rms_norm_eps: self.rms_norm_eps,
            qk_l2_clip_threshold: self.qk_l2_clip_threshold,
            rel_l2_clip_threshold: self.rel_l2_clip_threshold,
            conv_kernel_size: self.conv_kernel_size,
            final_logit_softcapping: self.final_logit_softcapping,
            n_layers_per_attn_res_block: block,
            n_routed_experts: self.n_routed_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            moe_intermediate_size: self.moe_intermediate_size,
            mlp_gate_softcap: self.mlp_gate_softcap,
            mlp_up_softcap: self.mlp_up_softcap,
            route_scale: self.route_scale,
            max_position_embeddings: self.max_position_embeddings,
            layer_spans,
            loop_config: LoopConfig {
                core_start: raw_loop.core_start,
                core_end: raw_loop.core_end,
                num_passes: raw_loop.num_passes,
                gate_rank: raw_loop.gate_rank,
            },
        })
    }
}

fn heads(
    prefix: &str,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
) -> Result<HeadGeometry, InferenceError> {
    if !num_attention_heads.is_multiple_of(num_key_value_heads) {
        return Err(error(&format!(
            "{prefix}num_attention_heads must be a multiple of {prefix}num_key_value_heads"
        )));
    }
    num_attention_heads
        .checked_mul(head_dim)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| error(&format!("{prefix}attention byte count overflow")))?;
    Ok(HeadGeometry {
        num_attention_heads,
        num_key_value_heads,
        head_dim,
    })
}
