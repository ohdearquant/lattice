//! Configuration for the BitNet b1.58 2B4T decoder-only model.
//!
//! BitNet b1.58 uses ternary weights {-1, 0, +1} with 2-bit I2_S packing.
//! The model is a 2B-parameter LLM with 30 layers, GQA (4:1), ReLU^2 activation,
//! SubLN normalization, and tied word embeddings.

/// **Unstable**: BitNet b1.58 2B4T model configuration; architecture under initial integration.
#[derive(Debug, Clone)]
pub struct BitNetConfig {
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads (GQA).
    pub num_key_value_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// FFN intermediate size.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_position_embeddings: usize,
    /// RoPE base frequency.
    pub rope_theta: f64,
    /// RMS normalization epsilon.
    pub rms_norm_eps: f32,
    /// Whether lm_head shares weights with embed_tokens.
    pub tie_word_embeddings: bool,
}

impl BitNetConfig {
    /// **Unstable**: default BitNet b1.58 2B4T configuration.
    pub fn bitnet_2b4t() -> Self {
        Self {
            num_hidden_layers: 30,
            hidden_size: 2560,
            num_attention_heads: 20,
            num_key_value_heads: 5,
            head_dim: 128,
            intermediate_size: 6912,
            vocab_size: 128_256,
            max_position_embeddings: 4096,
            rope_theta: 500_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
        }
    }

    /// **Unstable**: total KV projection dimension.
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// **Unstable**: total Q projection dimension.
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// **Unstable**: GQA ratio (Q heads per KV head).
    pub fn gqa_ratio(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_values() {
        let cfg = BitNetConfig::bitnet_2b4t();
        assert_eq!(cfg.num_hidden_layers, 30);
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_attention_heads, 20);
        assert_eq!(cfg.num_key_value_heads, 5);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.intermediate_size, 6912);
        assert_eq!(cfg.vocab_size, 128_256);
        assert_eq!(cfg.max_position_embeddings, 4096);
        assert!((cfg.rope_theta - 500_000.0).abs() < 1e-6);
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert!(cfg.tie_word_embeddings);
    }

    #[test]
    fn test_kv_dim() {
        let cfg = BitNetConfig::bitnet_2b4t();
        // 5 KV heads * 128 = 640
        assert_eq!(cfg.kv_dim(), 640);
    }

    #[test]
    fn test_q_dim() {
        let cfg = BitNetConfig::bitnet_2b4t();
        // 20 heads * 128 = 2560
        assert_eq!(cfg.q_dim(), 2560);
    }

    #[test]
    fn test_gqa_ratio() {
        let cfg = BitNetConfig::bitnet_2b4t();
        // 20 / 5 = 4:1 GQA
        assert_eq!(cfg.gqa_ratio(), 4);
    }

    #[test]
    fn test_hidden_size_equals_q_dim() {
        let cfg = BitNetConfig::bitnet_2b4t();
        // For BitNet 2B4T, hidden_size == num_heads * head_dim
        assert_eq!(cfg.hidden_size, cfg.q_dim());
    }
}
