//! Gemma 4 E2B weight storage (ADR-082 stage 5).
//!
//! Mirrors the qwen35 weight-storage split (`qwen35::weights`): a per-layer
//! struct plus a top-level container. Per ADR-082 Amendment 1, K/V-shared
//! layers carry `None` for `k_proj`/`v_proj`/`k_norm`. Their checkpoint
//! tensors exist but are tolerate-and-skipped at load time
//! (`gemma4_loading`) and never wired into the forward pass; the forward
//! pass instead resolves those layers' K/V through
//! [`crate::model::gemma4_cache::Gemma4KvCache`]'s donor-slot indirection.

/// Per-layer Gemma 4 weights.
pub(crate) struct Gemma4LayerWeights {
    pub(crate) input_layernorm: Vec<f32>,            // [hidden]
    pub(crate) post_attention_layernorm: Vec<f32>,   // [hidden]
    pub(crate) pre_feedforward_layernorm: Vec<f32>,  // [hidden]
    pub(crate) post_feedforward_layernorm: Vec<f32>, // [hidden]
    pub(crate) post_per_layer_input_norm: Vec<f32>,  // [hidden]
    /// `layer_scalar` checkpoint tensor, shape `[1]`, stored unwrapped.
    pub(crate) layer_scalar: f32,
    pub(crate) per_layer_input_gate: Vec<f32>, // [per_layer_dim, hidden]
    pub(crate) per_layer_projection: Vec<f32>, // [hidden, per_layer_dim]

    pub(crate) q_proj: Vec<f32>, // [num_attention_heads * head_w, hidden]
    pub(crate) o_proj: Vec<f32>, // [hidden, num_attention_heads * head_w]
    pub(crate) q_norm: Vec<f32>, // [head_w]

    /// `None` on KV-shared layers (ADR-082 Amendment 1): those layers have
    /// no `k_proj`/`v_proj`/`k_norm` weights loaded, and read their donor's
    /// K/V via [`crate::model::gemma4_cache::Gemma4KvCache`] instead.
    pub(crate) k_proj: Option<Vec<f32>>, // [kv_dim, hidden]
    pub(crate) v_proj: Option<Vec<f32>>, // [kv_dim, hidden]
    pub(crate) k_norm: Option<Vec<f32>>, // [head_w]

    pub(crate) gate_proj: Vec<f32>, // [mlp_dim, hidden]
    pub(crate) up_proj: Vec<f32>,   // [mlp_dim, hidden]
    pub(crate) down_proj: Vec<f32>, // [hidden, mlp_dim]
}

/// **Unstable**: Gemma 4 E2B weight storage; layout tied to checkpoint format.
pub(crate) struct Gemma4Weights {
    pub(crate) embed_tokens: Vec<f32>,               // [vocab, hidden]
    pub(crate) embed_tokens_per_layer: Vec<f32>,     // [vocab, num_hidden_layers * per_layer_dim]
    pub(crate) norm: Vec<f32>,                       // [hidden]
    pub(crate) per_layer_model_projection: Vec<f32>, // [num_hidden_layers * per_layer_dim, hidden]
    pub(crate) per_layer_projection_norm: Vec<f32>,  // [per_layer_dim]
    pub(crate) layers: Vec<Gemma4LayerWeights>,
}
