//! Qwen3 embedding model configuration (`QwenConfig`).

use crate::error::InferenceError;

/// **Unstable**: Qwen3 embedding model configuration; consumed by `QwenModel` loader.
#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
}

impl QwenConfig {
    /// **Unstable**: hardcoded Qwen3-Embedding-0.6B defaults.
    pub fn qwen3_embedding_0_6b() -> Self {
        Self {
            vocab_size: 151_669,
            hidden_size: 1024,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 3072,
            max_position_embeddings: 32_768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }

    /// **Unstable**: total Q projection output dimension.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// **Unstable**: total KV projection output dimension.
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// **Unstable**: GQA group size (Q heads per KV head).
    pub fn num_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[derive(serde::Deserialize)]
struct QwenConfigFile {
    vocab_size: Option<serde_json::Value>,
    hidden_size: Option<serde_json::Value>,
    num_hidden_layers: Option<serde_json::Value>,
    num_attention_heads: Option<serde_json::Value>,
    num_key_value_heads: Option<serde_json::Value>,
    head_dim: Option<serde_json::Value>,
    intermediate_size: Option<serde_json::Value>,
    max_position_embeddings: Option<serde_json::Value>,
    rms_norm_eps: Option<serde_json::Value>,
    rope_theta: Option<serde_json::Value>,
}

impl QwenConfig {
    /// **Unstable**: parse and validate a top-level Qwen3 embedding config.
    ///
    /// # Errors
    /// Returns an error for invalid JSON, missing fields, or unsupported geometry.
    pub fn from_json_str(text: &str) -> Result<Self, InferenceError> {
        let parsed: QwenConfigFile = serde_json::from_str(text)
            .map_err(|error| InferenceError::Inference(format!("invalid config.json: {error}")))?;
        let as_usize = |value: Option<&serde_json::Value>| {
            value
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
        };
        let get = |value: Option<&serde_json::Value>, key: &str| -> Result<usize, InferenceError> {
            as_usize(value)
                .ok_or_else(|| InferenceError::Inference(format!("config.json missing key {key}")))
        };

        let rope_theta = parsed
            .rope_theta
            .as_ref()
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1_000_000.0);
        let rms_norm_eps = parsed
            .rms_norm_eps
            .as_ref()
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
            .unwrap_or(1e-6);

        let num_attention_heads = get(parsed.num_attention_heads.as_ref(), "num_attention_heads")?;
        let num_key_value_heads = get(parsed.num_key_value_heads.as_ref(), "num_key_value_heads")?;
        // Both head counts are used as divisors (head_dim inference below,
        // QwenConfig::num_groups, GQA grouping). A zero from a caller-supplied
        // config.json would otherwise be an integer divide-by-zero panic, so reject
        // it as InvalidInput here rather than crashing deep in the forward pass.
        if num_attention_heads == 0 || num_key_value_heads == 0 {
            return Err(InferenceError::InvalidInput(format!(
                "config.json: num_attention_heads ({num_attention_heads}) and \
             num_key_value_heads ({num_key_value_heads}) must both be non-zero"
            )));
        }
        // GQA grouping requires num_attention_heads divisible by num_key_value_heads.
        // `GqaConfig::groups()` only `debug_assert`s this (stripped in release), and
        // `apply_gqa_attention` hard-asserts it; a non-divisible caller-supplied
        // config.json would otherwise panic deep in the forward pass. Reject it here
        // at load time, mirroring `Qwen35Config::from_json_str`.
        if num_attention_heads % num_key_value_heads != 0 {
            return Err(InferenceError::InvalidInput(format!(
                "config.json: num_attention_heads ({num_attention_heads}) must be \
             divisible by num_key_value_heads ({num_key_value_heads})"
            )));
        }
        let hidden_size = get(parsed.hidden_size.as_ref(), "hidden_size")?;
        // head_dim may be explicit in config, or inferred from hidden_size / num_heads.
        let head_dim =
            as_usize(parsed.head_dim.as_ref()).unwrap_or(hidden_size / num_attention_heads);
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(InferenceError::InvalidInput(format!(
                "config.json: head_dim ({head_dim}) must be non-zero and even for RoPE"
            )));
        }

        let config = QwenConfig {
            vocab_size: get(parsed.vocab_size.as_ref(), "vocab_size")?,
            hidden_size,
            num_hidden_layers: get(parsed.num_hidden_layers.as_ref(), "num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size: get(parsed.intermediate_size.as_ref(), "intermediate_size")?,
            max_position_embeddings: get(
                parsed.max_position_embeddings.as_ref(),
                "max_position_embeddings",
            )?,
            rms_norm_eps,
            rope_theta,
        };
        let q_dim = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| {
                InferenceError::InvalidInput(
                    "config.json: num_attention_heads * head_dim overflows usize".into(),
                )
            })?;
        let kv_dim = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| {
                InferenceError::InvalidInput(
                    "config.json: num_key_value_heads * head_dim overflows usize".into(),
                )
            })?;
        let doubled_kv_dim = kv_dim.checked_mul(2).ok_or_else(|| {
            InferenceError::InvalidInput("config.json: 2 * kv_dim overflows usize".into())
        })?;
        q_dim.checked_add(doubled_kv_dim).ok_or_else(|| {
            InferenceError::InvalidInput("config.json: q_dim + 2 * kv_dim overflows usize".into())
        })?;
        config.intermediate_size.checked_mul(2).ok_or_else(|| {
            InferenceError::InvalidInput(
                "config.json: 2 * intermediate_size overflows usize".into(),
            )
        })?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests;
