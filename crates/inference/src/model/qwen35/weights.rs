use crate::attention::gdn::GatedDeltaNetWeights;
use crate::error::InferenceError;

/// Weights for a full-attention (GQA) layer.
/// Note: q_proj is [2*q_dim, hidden] because attn_output_gate=true —
/// it produces both Q and the output gate Z in a single projection.
pub(crate) struct FullAttentionLayerWeights {
    pub(crate) q_proj: Vec<f32>, // [2*q_dim, hidden] (Q + gate_z)
    pub(crate) k_proj: Vec<f32>, // [kv_dim, hidden]
    pub(crate) v_proj: Vec<f32>, // [kv_dim, hidden]
    pub(crate) o_proj: Vec<f32>, // [hidden, q_dim]
    pub(crate) q_norm: Vec<f32>, // [head_dim]
    pub(crate) k_norm: Vec<f32>, // [head_dim]
}

/// Weights common to all layers (norms + FFN).
pub(crate) struct CommonLayerWeights {
    pub(crate) input_layernorm: Vec<f32>,          // [hidden]
    pub(crate) post_attention_layernorm: Vec<f32>, // [hidden]
    pub(crate) ffn: FeedForwardWeights,
}

/// Per-layer FFN weights: either dense SwiGLU or MoE.
pub(crate) enum FeedForwardWeights {
    Dense(DenseFfnWeights),
    Moe(MoeLayerWeights),
}

/// Dense SwiGLU FFN weights (Qwen3.5 and non-MoE layers).
pub(crate) struct DenseFfnWeights {
    pub(crate) gate_proj: Vec<f32>, // [intermediate, hidden]
    pub(crate) up_proj: Vec<f32>,   // [intermediate, hidden]
    pub(crate) down_proj: Vec<f32>, // [hidden, intermediate]
}

/// Per-layer weight storage.
pub(crate) enum AttentionWeights {
    Linear(GatedDeltaNetWeights),
    Full(FullAttentionLayerWeights),
}

/// **Unstable**: Qwen3.5-2B weight storage; layout tied to checkpoint format.
pub struct ModelWeights {
    pub(crate) embed_tokens: Vec<f32>,    // [vocab_size, hidden]
    pub(crate) lm_head: Option<Vec<f32>>, // [vocab_size, hidden] when tie_word_embeddings=false
    pub(crate) final_norm: Vec<f32>,      // [hidden]
    pub(crate) layers: Vec<(AttentionWeights, CommonLayerWeights)>,
}

impl ModelWeights {
    /// Returns the output projection weights; falls back to embed_tokens when tied.
    pub(crate) fn logits_weight(&self) -> &[f32] {
        self.lm_head.as_deref().unwrap_or(&self.embed_tokens)
    }
}

// -----------------------------------------------------------------------
// MoE types (Qwen3.6-35B-A3B: 256 routed experts, top-8, 1 shared expert)
// -----------------------------------------------------------------------

/// Top-k router for Mixture-of-Experts layers.
#[derive(Debug)]
pub(crate) struct MoeRouter {
    pub(crate) gate: Vec<f32>, // [num_experts, hidden_size], bias-free
    pub(crate) num_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) hidden_size: usize,
}

impl MoeRouter {
    pub(crate) fn new(
        gate: Vec<f32>,
        num_experts: usize,
        num_experts_per_tok: usize,
        hidden_size: usize,
    ) -> Result<Self, InferenceError> {
        let expected = num_experts
            .checked_mul(hidden_size)
            .ok_or_else(|| InferenceError::Inference("MoE router shape overflow".into()))?;
        if gate.len() != expected {
            return Err(InferenceError::ShapeMismatch {
                name: "mlp.gate.weight".to_string(),
                expected: vec![num_experts, hidden_size],
                actual: vec![gate.len()],
            });
        }
        if num_experts_per_tok == 0 || num_experts_per_tok > num_experts {
            return Err(InferenceError::UnsupportedModel(format!(
                "invalid MoE top_k={num_experts_per_tok} for num_experts={num_experts}"
            )));
        }
        Ok(Self {
            gate,
            num_experts,
            num_experts_per_tok,
            hidden_size,
        })
    }
}

/// Routed (sparse) expert weight storage for MoE layers.
#[derive(Debug)]
pub(crate) struct RoutedExperts {
    pub(crate) gate_up_proj: Vec<f32>, // [num_experts, 2 * intermediate_size, hidden_size]
    pub(crate) down_proj: Vec<f32>,    // [num_experts, hidden_size, intermediate_size]
    pub(crate) num_experts: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
}

impl RoutedExperts {
    pub(crate) fn new(
        gate_up_proj: Vec<f32>,
        down_proj: Vec<f32>,
        num_experts: usize,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, InferenceError> {
        let expected_gate_up = num_experts
            .checked_mul(2 * intermediate_size)
            .and_then(|x| x.checked_mul(hidden_size))
            .ok_or_else(|| {
                InferenceError::Inference("RoutedExperts gate_up_proj shape overflow".into())
            })?;
        let expected_down = num_experts
            .checked_mul(hidden_size)
            .and_then(|x| x.checked_mul(intermediate_size))
            .ok_or_else(|| {
                InferenceError::Inference("RoutedExperts down_proj shape overflow".into())
            })?;
        if gate_up_proj.len() != expected_gate_up {
            return Err(InferenceError::ShapeMismatch {
                name: "mlp.experts.gate_up_proj".to_string(),
                expected: vec![num_experts, 2 * intermediate_size, hidden_size],
                actual: vec![gate_up_proj.len()],
            });
        }
        if down_proj.len() != expected_down {
            return Err(InferenceError::ShapeMismatch {
                name: "mlp.experts.down_proj".to_string(),
                expected: vec![num_experts, hidden_size, intermediate_size],
                actual: vec![down_proj.len()],
            });
        }
        Ok(Self {
            gate_up_proj,
            down_proj,
            num_experts,
            hidden_size,
            intermediate_size,
        })
    }
}

/// Shared (always-active) expert weights for Qwen3.6 MoE layers.
#[derive(Debug)]
pub(crate) struct SharedExpert {
    pub(crate) gate_proj: Vec<f32>, // [intermediate_size, hidden_size]
    pub(crate) up_proj: Vec<f32>,   // [intermediate_size, hidden_size]
    pub(crate) down_proj: Vec<f32>, // [hidden_size, intermediate_size]
    pub(crate) shared_expert_gate: Vec<f32>, // [1, hidden_size] (scalar gate)
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
}

impl SharedExpert {
    pub(crate) fn new(
        gate_proj: Vec<f32>,
        up_proj: Vec<f32>,
        down_proj: Vec<f32>,
        shared_expert_gate: Vec<f32>,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, InferenceError> {
        let up_shape = intermediate_size
            .checked_mul(hidden_size)
            .ok_or_else(|| InferenceError::Inference("shared expert shape overflow".into()))?;
        let down_shape = hidden_size
            .checked_mul(intermediate_size)
            .ok_or_else(|| InferenceError::Inference("shared expert down shape overflow".into()))?;
        if gate_proj.len() != up_shape {
            return Err(InferenceError::ShapeMismatch {
                name: "mlp.shared_expert.gate_proj.weight".to_string(),
                expected: vec![intermediate_size, hidden_size],
                actual: vec![gate_proj.len()],
            });
        }
        if up_proj.len() != up_shape {
            return Err(InferenceError::ShapeMismatch {
                name: "mlp.shared_expert.up_proj.weight".to_string(),
                expected: vec![intermediate_size, hidden_size],
                actual: vec![up_proj.len()],
            });
        }
        if down_proj.len() != down_shape {
            return Err(InferenceError::ShapeMismatch {
                name: "mlp.shared_expert.down_proj.weight".to_string(),
                expected: vec![hidden_size, intermediate_size],
                actual: vec![down_proj.len()],
            });
        }
        if shared_expert_gate.len() != hidden_size {
            return Err(InferenceError::ShapeMismatch {
                name: "mlp.shared_expert_gate.weight".to_string(),
                expected: vec![1, hidden_size],
                actual: vec![shared_expert_gate.len()],
            });
        }
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            shared_expert_gate,
            hidden_size,
            intermediate_size,
        })
    }
}

/// All MoE FFN weights for one layer.
pub(crate) struct MoeLayerWeights {
    pub(crate) router: MoeRouter,
    pub(crate) experts: RoutedExperts,
    pub(crate) shared_expert: SharedExpert,
}
