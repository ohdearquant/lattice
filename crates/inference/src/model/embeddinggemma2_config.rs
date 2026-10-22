//! EmbeddingGemma 2 text-tower configuration.
//!
//! Parsed from the `text_config` object of the checkpoint's `config.json`
//! (a config that has no `text_config` wrapper is read as the text config
//! itself). Every forward-relevant field is required: a missing field is an
//! error naming it, and a value this implementation does not support is
//! rejected instead of being silently replaced by a default.

use crate::error::InferenceError;
use crate::model::config_file::read_config_json_bounded;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

const MAX_DIM: usize = 1 << 20;
const MAX_LAYERS: usize = 1024;

/// The attention pattern of one encoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingGemma2LayerKind {
    /// Bidirectional attention restricted to keys with `|q - k| <= sliding_window`.
    Sliding,
    /// Bidirectional attention over every key.
    Full,
}

/// Attention geometry of one encoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingGemma2LayerShape {
    /// Width of one attention head.
    pub head_dim: usize,
    /// Number of key/value heads.
    pub num_key_value_heads: usize,
}

/// Validated EmbeddingGemma 2 text-tower configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingGemma2Config {
    /// Token vocabulary size.
    pub vocab_size: usize,
    /// Residual stream width.
    pub hidden_size: usize,
    /// Width of the gated feed-forward layer.
    pub intermediate_size: usize,
    /// Number of encoder layers.
    pub num_hidden_layers: usize,
    /// Number of query heads (shared by every layer).
    pub num_attention_heads: usize,
    /// Width of the per-layer input signal that gates each layer's third residual block.
    pub hidden_size_per_layer_input: usize,
    /// Width of the final per-token projection, which is the native embedding size.
    pub embedding_dim: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Inclusive radius of the sliding attention window, in tokens.
    pub sliding_window: usize,
    /// Per-layer attention pattern.
    pub layer_types: Vec<EmbeddingGemma2LayerKind>,
    /// Per-layer attention geometry.
    pub layer_shapes: Vec<EmbeddingGemma2LayerShape>,
    /// RoPE base for sliding layers.
    pub rope_theta_sliding: f64,
    /// RoPE base for full-attention layers.
    pub rope_theta_full: f64,
    /// Id of the token placed before the text on every input.
    pub bos_token_id: u32,
    /// Id of the token placed after the text on every input.
    pub eos_token_id: u32,
    /// Multiplier applied to the token embeddings. Defaults to `sqrt(hidden_size)` in f32,
    /// which is what the reference computes when it runs in f32. A reference run in bf16
    /// rounds this value to bf16 first (22.625 for a hidden size of 512).
    pub embed_scale: f32,
}

#[derive(Deserialize)]
struct RawRopeEntry {
    rope_theta: f64,
    rope_type: String,
}

#[derive(Deserialize)]
struct RawRope {
    full_attention: RawRopeEntry,
    sliding_attention: RawRopeEntry,
}

#[derive(Deserialize)]
struct RawLayerOverride {
    head_dim: Option<usize>,
    num_key_value_heads: Option<usize>,
}

#[derive(Deserialize)]
struct RawTextConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    hidden_size_per_layer_input: usize,
    embedding_dim: usize,
    rms_norm_eps: f64,
    sliding_window: usize,
    layer_types: Vec<String>,
    hidden_activation: String,
    #[serde(default = "default_bos_token_id")]
    bos_token_id: serde_json::Value,
    #[serde(default = "default_eos_token_id")]
    eos_token_id: serde_json::Value,
    rope_parameters: RawRope,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    per_layer_config: Option<BTreeMap<String, RawLayerOverride>>,
}

fn default_bos_token_id() -> serde_json::Value {
    serde_json::json!(2)
}

fn default_eos_token_id() -> serde_json::Value {
    serde_json::json!(1)
}

/// The checkpoint's top-level multimodal token layout: the ids that mark
/// image slots in the text sequence.
///
/// Read from `config.json`'s top-level keys, not from `text_config`. The
/// released checkpoint uses `image_token_id` 258880, `boi_token_id` 255999 and
/// `eoi_token_id` 258882; a config that omits them makes the model text-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ImagePartsConfig {
    /// Placeholder id replaced by one image soft row.
    pub(crate) image_token_id: u32,
    /// Beginning-of-image marker id.
    pub(crate) boi_token_id: u32,
    /// End-of-image marker id.
    pub(crate) eoi_token_id: u32,
}

/// Reads the image token layout from a `config.json` document's top level.
/// Returns `None` when any of the three ids is absent or out of range, which
/// keeps a text-only checkpoint loadable.
pub(crate) fn image_parts_config(json: &str, vocab_size: usize) -> Option<ImagePartsConfig> {
    let root: serde_json::Value = serde_json::from_str(json).ok()?;
    let id = |name: &str| {
        root.get(name)
            .and_then(serde_json::Value::as_u64)
            .filter(|&value| (value as usize) < vocab_size)
            .and_then(|value| u32::try_from(value).ok())
    };
    Some(ImagePartsConfig {
        image_token_id: id("image_token_id")?,
        boi_token_id: id("boi_token_id")?,
        eoi_token_id: id("eoi_token_id")?,
    })
}

/// A special-token id: one non-negative integer below the vocabulary size.
fn token_id(
    name: &str,
    value: &serde_json::Value,
    vocab_size: usize,
) -> Result<u32, InferenceError> {
    value
        .as_u64()
        .filter(|&id| (id as usize) < vocab_size)
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| {
            InferenceError::Inference(format!(
                "EmbeddingGemma 2 config: {name} {value} must be one integer below vocab_size {vocab_size}"
            ))
        })
}

impl EmbeddingGemma2Config {
    /// Reads and parses `<dir>/config.json`.
    pub fn from_model_dir(dir: &Path) -> Result<Self, InferenceError> {
        let path = dir.join("config.json");
        if !path.exists() {
            return Err(InferenceError::ModelNotFound(format!(
                "missing config.json in {}",
                dir.display()
            )));
        }
        let text = read_config_json_bounded(&path, "config.json")?;
        Self::from_config_json_str(&text)
    }

    /// Parses a `config.json` document, using its `text_config` object when present.
    pub fn from_config_json_str(json: &str) -> Result<Self, InferenceError> {
        let root: serde_json::Value = serde_json::from_str(json).map_err(|e| {
            InferenceError::Inference(format!("invalid EmbeddingGemma 2 config.json: {e}"))
        })?;
        let text = match root.get("text_config") {
            Some(inner) => inner.clone(),
            None => root,
        };
        let raw: RawTextConfig = serde_json::from_value(text).map_err(|e| {
            InferenceError::Inference(format!("invalid EmbeddingGemma 2 text_config: {e}"))
        })?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: RawTextConfig) -> Result<Self, InferenceError> {
        let bad =
            |what: String| InferenceError::Inference(format!("EmbeddingGemma 2 config: {what}"));

        if raw.hidden_activation != "gelu_pytorch_tanh" {
            return Err(bad(format!(
                "hidden_activation {:?} is not supported (expected gelu_pytorch_tanh)",
                raw.hidden_activation
            )));
        }
        if raw.attention_bias {
            return Err(bad("attention_bias=true is not supported".to_string()));
        }
        for (name, entry) in [
            ("full_attention", &raw.rope_parameters.full_attention),
            ("sliding_attention", &raw.rope_parameters.sliding_attention),
        ] {
            if entry.rope_type != "default" {
                return Err(bad(format!(
                    "rope_type {:?} for {name} is not supported (expected default)",
                    entry.rope_type
                )));
            }
            if !(entry.rope_theta.is_finite() && entry.rope_theta > 1.0) {
                return Err(bad(format!(
                    "rope_theta {} for {name} must be finite and above 1",
                    entry.rope_theta
                )));
            }
        }
        if !(raw.rms_norm_eps.is_finite() && raw.rms_norm_eps > 0.0) {
            return Err(bad(format!(
                "rms_norm_eps {} must be finite and positive",
                raw.rms_norm_eps
            )));
        }
        for (name, value) in [
            ("vocab_size", raw.vocab_size),
            ("hidden_size", raw.hidden_size),
            ("intermediate_size", raw.intermediate_size),
            ("num_attention_heads", raw.num_attention_heads),
            ("num_key_value_heads", raw.num_key_value_heads),
            ("head_dim", raw.head_dim),
            (
                "hidden_size_per_layer_input",
                raw.hidden_size_per_layer_input,
            ),
            ("embedding_dim", raw.embedding_dim),
            ("sliding_window", raw.sliding_window),
        ] {
            if value == 0 || value > MAX_DIM {
                return Err(bad(format!("{name}={value} is outside 1..={MAX_DIM}")));
            }
        }
        if raw.num_hidden_layers == 0 || raw.num_hidden_layers > MAX_LAYERS {
            return Err(bad(format!(
                "num_hidden_layers={} is outside 1..={MAX_LAYERS}",
                raw.num_hidden_layers
            )));
        }
        if raw.layer_types.len() != raw.num_hidden_layers {
            return Err(bad(format!(
                "layer_types has {} entries for {} layers",
                raw.layer_types.len(),
                raw.num_hidden_layers
            )));
        }

        let mut layer_types = Vec::with_capacity(raw.num_hidden_layers);
        for (i, name) in raw.layer_types.iter().enumerate() {
            layer_types.push(match name.as_str() {
                "sliding_attention" => EmbeddingGemma2LayerKind::Sliding,
                "full_attention" => EmbeddingGemma2LayerKind::Full,
                other => return Err(bad(format!("layer {i} has unknown layer type {other:?}"))),
            });
        }
        // The reference forces the final layer to full attention.
        if let Some(last) = layer_types.last_mut() {
            *last = EmbeddingGemma2LayerKind::Full;
        }

        let mut layer_shapes = vec![
            EmbeddingGemma2LayerShape {
                head_dim: raw.head_dim,
                num_key_value_heads: raw.num_key_value_heads,
            };
            raw.num_hidden_layers
        ];
        for (key, over) in raw.per_layer_config.iter().flatten() {
            let index: usize = key
                .parse()
                .map_err(|_| bad(format!("per_layer_config key {key:?} is not a layer index")))?;
            let shape = layer_shapes.get_mut(index).ok_or_else(|| {
                bad(format!(
                    "per_layer_config key {key:?} is outside the {} layers",
                    raw.num_hidden_layers
                ))
            })?;
            if let Some(head_dim) = over.head_dim {
                shape.head_dim = head_dim;
            }
            if let Some(kv) = over.num_key_value_heads {
                shape.num_key_value_heads = kv;
            }
        }

        let mut kind_shape: [Option<EmbeddingGemma2LayerShape>; 2] = [None, None];
        for (i, (kind, shape)) in layer_types.iter().zip(&layer_shapes).enumerate() {
            if shape.head_dim == 0 || shape.head_dim > MAX_DIM || !shape.head_dim.is_multiple_of(2)
            {
                return Err(bad(format!(
                    "layer {i} head_dim {} must be even and within 1..={MAX_DIM}",
                    shape.head_dim
                )));
            }
            if shape.num_key_value_heads == 0
                || !raw
                    .num_attention_heads
                    .is_multiple_of(shape.num_key_value_heads)
            {
                return Err(bad(format!(
                    "layer {i} num_key_value_heads {} must divide num_attention_heads {}",
                    shape.num_key_value_heads, raw.num_attention_heads
                )));
            }
            // RoPE tables are built once per layer type, so a type must have one geometry.
            let slot = &mut kind_shape[usize::from(*kind == EmbeddingGemma2LayerKind::Full)];
            match slot {
                None => *slot = Some(*shape),
                Some(first) if first != shape => {
                    return Err(bad(format!(
                        "layer {i} geometry differs from the other {kind:?} layers"
                    )));
                }
                Some(_) => {}
            }
        }

        Ok(Self {
            vocab_size: raw.vocab_size,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            hidden_size_per_layer_input: raw.hidden_size_per_layer_input,
            embedding_dim: raw.embedding_dim,
            rms_norm_eps: raw.rms_norm_eps as f32,
            sliding_window: raw.sliding_window,
            layer_types,
            layer_shapes,
            rope_theta_sliding: raw.rope_parameters.sliding_attention.rope_theta,
            rope_theta_full: raw.rope_parameters.full_attention.rope_theta,
            bos_token_id: token_id("bos_token_id", &raw.bos_token_id, raw.vocab_size)?,
            eos_token_id: token_id("eos_token_id", &raw.eos_token_id, raw.vocab_size)?,
            embed_scale: (raw.hidden_size as f32).sqrt(),
        })
    }
}

#[cfg(test)]
mod tests;
