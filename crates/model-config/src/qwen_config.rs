//! Qwen3 embedding model configuration (`QwenConfig`).

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
