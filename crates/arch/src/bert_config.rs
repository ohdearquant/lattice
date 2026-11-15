//! BERT model configuration and config.json parsing.

use lattice_inference_core::InferenceError;

// Per-field ceilings bound pathological table, scratch and iteration dimensions.
// Their scales follow the corresponding architecture-wide Qwen config limits;
// independent field limits do not bound products or guarantee allocation success.
const MAX_VOCAB_SIZE: usize = 4_000_000;
const MAX_HIDDEN_SIZE: usize = 1_048_576;
const MAX_HIDDEN_LAYERS: usize = 4_096;
const MAX_ATTENTION_HEADS: usize = 8_192;
const MAX_INTERMEDIATE_SIZE: usize = 1_048_576;
const MAX_POSITION_EMBEDDINGS: usize = 4_194_304;
const MAX_TYPE_VOCAB_SIZE: usize = 4_000_000;

/// **Stable** (provisional): BERT model configuration; consumed by `lattice-embed`
/// via `BertModel::config()`. Field additions are backward-compatible; field
/// removals or type changes require a SemVer bump.
#[derive(Debug, Clone)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
    /// Index of the position-embedding row the first token reads. BERT numbers
    /// positions from 0. RoBERTa-family checkpoints (`xlm-roberta`, `roberta`,
    /// `camembert`) number them from `pad_token_id + 1`, leaving the rows below
    /// for padding.
    pub position_offset: usize,
}

impl BertConfig {
    /// **Stable** (provisional): factory for BGE-small-en-v1.5 configuration.
    ///
    /// Note: the published Hugging Face config uses 12 attention heads with
    /// hidden_size=384, giving head_dim=32.
    pub fn bge_small() -> Self {
        Self {
            vocab_size: 30_522,
            hidden_size: 384,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 1_536,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            position_offset: 0,
        }
    }

    /// **Stable** (provisional): factory for BGE-base-en-v1.5 configuration.
    pub fn bge_base() -> Self {
        Self {
            vocab_size: 30_522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3_072,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            position_offset: 0,
        }
    }

    /// **Stable** (provisional): factory for BGE-large-en-v1.5 configuration.
    pub fn bge_large() -> Self {
        Self {
            vocab_size: 30_522,
            hidden_size: 1_024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4_096,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            position_offset: 0,
        }
    }

    /// **Unstable**: derived convenience; may be replaced by a struct field.
    /// Callers constructing this public struct directly must validate it first.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// **Unstable**: number of positions a sequence can use, the position table
    /// size less [`position_offset`](Self::position_offset). Callers constructing
    /// this public struct directly must validate it first.
    pub fn max_positions(&self) -> usize {
        self.max_position_embeddings - self.position_offset
    }

    /// **Unstable**: parse a `BertConfig` from `config.json` text, without
    /// touching the filesystem; signature may change as the from-bytes model
    /// loading API matures.
    ///
    /// This is the string counterpart to the file-based `config.json` parsing
    /// `from_directory` performs internally. Used by
    /// `BertModel::from_bytes`; unlike `from_directory`, there is no
    /// fallback to inferring the config from safetensors tensor shapes when a
    /// required key is missing.
    pub fn from_json_str(config_json: &str) -> Result<Self, InferenceError> {
        let config = parse_config_json_str(config_json)?;
        config.validate().map_err(|error| match error {
            InferenceError::Inference(message) => {
                InferenceError::Inference(format!("invalid BERT config.json: {message}"))
            }
            other => other,
        })?;
        Ok(config)
    }

    /// **Unstable**: validate BERT geometry and normalization configuration.
    pub fn validate(&self) -> Result<(), InferenceError> {
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("intermediate_size", self.intermediate_size),
            ("max_position_embeddings", self.max_position_embeddings),
            ("type_vocab_size", self.type_vocab_size),
        ] {
            if value == 0 {
                return Err(InferenceError::Inference(format!("{name} must be > 0")));
            }
        }
        for (name, value, max) in [
            ("vocab_size", self.vocab_size, MAX_VOCAB_SIZE),
            ("hidden_size", self.hidden_size, MAX_HIDDEN_SIZE),
            (
                "num_hidden_layers",
                self.num_hidden_layers,
                MAX_HIDDEN_LAYERS,
            ),
            (
                "num_attention_heads",
                self.num_attention_heads,
                MAX_ATTENTION_HEADS,
            ),
            (
                "intermediate_size",
                self.intermediate_size,
                MAX_INTERMEDIATE_SIZE,
            ),
            (
                "max_position_embeddings",
                self.max_position_embeddings,
                MAX_POSITION_EMBEDDINGS,
            ),
            ("type_vocab_size", self.type_vocab_size, MAX_TYPE_VOCAB_SIZE),
        ] {
            if value > max {
                return Err(InferenceError::Inference(format!(
                    "{name} ({value}) exceeds maximum {max}"
                )));
            }
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(InferenceError::Inference(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if !(self.layer_norm_eps.is_finite() && self.layer_norm_eps > 0.0) {
            return Err(InferenceError::Inference(
                "layer_norm_eps must be finite and > 0".into(),
            ));
        }
        if self.position_offset >= self.max_position_embeddings {
            return Err(InferenceError::Inference(format!(
                "position_offset ({}) must be below max_position_embeddings ({})",
                self.position_offset, self.max_position_embeddings
            )));
        }
        Ok(())
    }
}

#[derive(serde::Deserialize)]
struct BertConfigFields {
    #[serde(default, deserialize_with = "present_raw")]
    vocab_size: Option<Box<serde_json::value::RawValue>>,
    #[serde(default, deserialize_with = "present_raw")]
    hidden_size: Option<Box<serde_json::value::RawValue>>,
    #[serde(default, deserialize_with = "present_raw")]
    num_hidden_layers: Option<Box<serde_json::value::RawValue>>,
    #[serde(default, deserialize_with = "present_raw")]
    num_attention_heads: Option<Box<serde_json::value::RawValue>>,
    #[serde(default, deserialize_with = "present_raw")]
    intermediate_size: Option<Box<serde_json::value::RawValue>>,
    #[serde(default, deserialize_with = "present_raw")]
    max_position_embeddings: Option<Box<serde_json::value::RawValue>>,
    #[serde(default, deserialize_with = "present_raw")]
    type_vocab_size: Option<Box<serde_json::value::RawValue>>,
    #[serde(default, deserialize_with = "present_raw")]
    layer_norm_eps: Option<Box<serde_json::value::RawValue>>,
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default, deserialize_with = "present_raw")]
    pad_token_id: Option<Box<serde_json::value::RawValue>>,
}

/// `model_type` values whose checkpoints number positions from `pad_token_id + 1`.
const PADDING_OFFSET_MODEL_TYPES: [&str; 3] = ["xlm-roberta", "roberta", "camembert"];

fn present_raw<'de, D>(
    deserializer: D,
) -> Result<Option<Box<serde_json::value::RawValue>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    serde::Deserialize::deserialize(deserializer).map(Some)
}

// Retain the previous scanner's lexical numeric conversion, including its handling
// of escaped strings, while taking field boundaries only from the root JSON object.
fn scalar<'a>(
    field: &'a Option<Box<serde_json::value::RawValue>>,
    key: &str,
) -> Result<&'a str, InferenceError> {
    let raw = field
        .as_ref()
        .ok_or_else(|| InferenceError::Inference(format!("config.json missing key {key}")))?
        .get();
    if let Some(rest) = raw.strip_prefix('"') {
        let end = rest
            .find('"')
            .ok_or_else(|| InferenceError::Inference(format!("config.json missing key {key}")))?;
        Ok(&rest[..end])
    } else {
        let end = raw
            .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
            .unwrap_or(raw.len());
        Ok(raw[..end].trim())
    }
}

fn parse_config_json_str(text: &str) -> Result<BertConfig, InferenceError> {
    if !text.trim_start().starts_with('{') {
        return Err(InferenceError::Inference(
            "invalid config.json: expected object".into(),
        ));
    }
    let fields: BertConfigFields = serde_json::from_str(text)
        .map_err(|e| InferenceError::Inference(format!("invalid config.json: {e}")))?;
    let get_usize = |field, key| {
        scalar(field, key)?
            .parse::<usize>()
            .map_err(|e| InferenceError::Inference(format!("invalid usize for {key}: {e}")))
    };
    let get_f32 = |field, key| {
        scalar(field, key)?
            .parse::<f32>()
            .map_err(|e| InferenceError::Inference(format!("invalid f32 for {key}: {e}")))
    };
    let position_offset = match fields.model_type.as_deref() {
        None | Some("bert") => 0,
        Some(model_type) if PADDING_OFFSET_MODEL_TYPES.contains(&model_type) => {
            get_usize(&fields.pad_token_id, "pad_token_id")?
                .checked_add(1)
                .ok_or_else(|| InferenceError::Inference("pad_token_id overflows".into()))?
        }
        Some(model_type) => {
            return Err(InferenceError::UnsupportedModel(format!(
                "config.json model_type \"{model_type}\" is not a BERT or RoBERTa encoder"
            )));
        }
    };
    Ok(BertConfig {
        vocab_size: get_usize(&fields.vocab_size, "vocab_size")?,
        hidden_size: get_usize(&fields.hidden_size, "hidden_size")?,
        num_hidden_layers: get_usize(&fields.num_hidden_layers, "num_hidden_layers")?,
        num_attention_heads: get_usize(&fields.num_attention_heads, "num_attention_heads")?,
        intermediate_size: get_usize(&fields.intermediate_size, "intermediate_size")?,
        max_position_embeddings: get_usize(
            &fields.max_position_embeddings,
            "max_position_embeddings",
        )?,
        type_vocab_size: get_usize(&fields.type_vocab_size, "type_vocab_size")?,
        layer_norm_eps: get_f32(&fields.layer_norm_eps, "layer_norm_eps")?,
        position_offset,
    })
}

#[cfg(test)]
mod tests;
