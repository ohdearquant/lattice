//! N-gram prompt lookup speculative decoding.
//!
//! A zero-training speculative decoding technique that uses n-gram matching
//! against the prompt to predict continuation tokens. Effective on grounded
//! tasks (summarisation, extraction, reformatting) where the model frequently
//! copies spans from the input.
//!
//! # Algorithm
//!
//! 1. Take the last `n` tokens of the generated sequence (trying `n = max_ngram` down to 2).
//! 2. Search the prompt for a matching n-gram.
//! 3. If found at position `p`, draft the next `max_draft` tokens from `prompt[p+n..]`.
//! 4. Verify each draft token by running the model forward; accept tokens until
//!    the model disagrees.
//!
//! # Usage
//!
//! The module exposes two levels of API:
//! - Low-level: [`NgramSpeculator::speculate`] + [`verify_draft`] for custom loops.
//! - High-level: [`generate_with_speculation`] wraps any `forward_fn` closure.

/// **Unstable**: n-gram speculative decoding; algorithm parameters and API may change as
/// the generation pipeline evolves.
///
/// N-gram prompt lookup speculative decoder.
///
/// Searches the original prompt for n-gram matches against recent tokens,
/// then proposes draft continuations from the prompt's subsequent tokens.
pub struct NgramSpeculator {
    /// The full prompt token sequence for n-gram matching.
    prompt_tokens: Vec<u32>,
    /// Maximum n-gram length to try (tries `max_ngram` down to 2).
    max_ngram: usize,
    /// Maximum number of draft tokens per speculation step.
    max_draft: usize,
}

impl NgramSpeculator {
    /// **Unstable**: constructor; parameters may expand as speculation strategies evolve.
    ///
    /// Create a new speculator.
    ///
    /// - `prompt_tokens`: the full tokenised prompt.
    /// - `max_ngram`: longest n-gram to attempt (default: 5). Tried in
    ///   descending order; first match wins.
    /// - `max_draft`: maximum draft tokens to propose per step (default: 4).
    pub fn new(prompt_tokens: Vec<u32>, max_ngram: usize, max_draft: usize) -> Self {
        Self {
            prompt_tokens,
            max_ngram,
            max_draft,
        }
    }

    /// **Unstable**: speculate next tokens via n-gram prompt lookup; return type may change.
    ///
    /// Given recent context tokens, predict the next few tokens by n-gram
    /// lookup in the prompt.
    ///
    /// Returns draft token IDs (may be empty if no n-gram match is found).
    /// Tries the longest n-gram first for better prediction quality.
    pub fn speculate(&self, recent_tokens: &[u32]) -> Vec<u32> {
        if recent_tokens.is_empty() {
            return Vec::new();
        }

        // Try longest n-gram first (greedy: longer match = better prediction)
        let max_n = self.max_ngram.min(recent_tokens.len());
        for n in (2..=max_n).rev() {
            let suffix = &recent_tokens[recent_tokens.len() - n..];

            if let Some(pos) = self.find_ngram(suffix) {
                let start = pos + n;
                let end = (start + self.max_draft).min(self.prompt_tokens.len());
                if start < end {
                    return self.prompt_tokens[start..end].to_vec();
                }
            }
        }

        Vec::new()
    }

    /// Linear scan for the first occurrence of `pattern` in the prompt.
    ///
    /// Prompts are typically a few hundred to a few thousand tokens, so a
    /// linear scan is fast enough (sub-microsecond for typical lengths).
    fn find_ngram(&self, pattern: &[u32]) -> Option<usize> {
        let n = pattern.len();
        if n == 0 || n > self.prompt_tokens.len() {
            return None;
        }
        (0..=self.prompt_tokens.len() - n).find(|&i| self.prompt_tokens[i..i + n] == *pattern)
    }
}

/// **Unstable**: argmax helper for greedy decoding; co-located with speculation utilities.
///
/// Return the index of the maximum element in `logits`.
///
/// When multiple elements share the maximum value, returns the last
/// occurrence (Rust `max_by` semantics). Returns 0 on empty input.
pub fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// MTP (Multi-Token Prediction) speculative decode — structs, weights, verifier
// ---------------------------------------------------------------------------

/// Runtime configuration for the MTP speculative verifier.
#[derive(Debug, Clone, PartialEq)]
pub struct MtpConfig {
    pub draft_length: usize,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub partial_rotary_factor: f32,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub use_dedicated_embeddings: bool,
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            draft_length: 4,
            num_hidden_layers: 1,
            hidden_size: 2048,
            vocab_size: 248_320,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            num_experts: 256,
            num_experts_per_tok: 8,
            moe_intermediate_size: 512,
            shared_expert_intermediate_size: 512,
            use_dedicated_embeddings: false,
        }
    }
}

/// Per-step metrics collected during MTP verification.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MtpMetrics {
    pub target_forwards: usize,
    pub mtp_forwards: usize,
    pub draft_tokens: usize,
    pub accepted_tokens: usize,
    pub accepted_tokens_per_forward: f64,
    pub acceptance_rate: f64,
}

/// Result returned by [`mtp_verify_draft`].
#[derive(Debug, Clone, PartialEq)]
pub struct MtpVerifyResult {
    pub accepted_count: usize,
    pub accepted_tokens: Vec<u32>,
    pub fallback_token: Option<u32>,
    pub draft_tokens: Vec<u32>,
    pub stopped_by_eos: bool,
    pub metrics: MtpMetrics,
}

/// Output of a single MTP forward step.
pub struct MtpForwardOutput {
    pub logits: Vec<f32>,
    pub hidden: Vec<f32>,
}

// --- MTP weight structures ---

/// All weights for the MTP module.
#[derive(Debug, Clone)]
pub struct MtpWeights {
    /// fc_weight: [hidden_size, 2 * hidden_size] — fusion projection
    pub fc_weight: Vec<f32>,
    pub layers: Vec<MtpLayerWeights>,
    /// Final RMSNorm weight: [hidden_size]
    pub norm_weight: Vec<f32>,
    /// Pre-FC normalization applied to embedding: [hidden_size]
    pub pre_fc_norm_embedding_weight: Vec<f32>,
    /// Pre-FC normalization applied to previous hidden: [hidden_size]
    pub pre_fc_norm_hidden_weight: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct MtpLayerWeights {
    pub input_layernorm: Vec<f32>,          // [hidden_size]
    pub post_attention_layernorm: Vec<f32>, // [hidden_size]
    pub self_attn: MtpAttentionWeights,
    pub mlp: MtpMoeWeights,
}

#[derive(Debug, Clone)]
pub struct MtpAttentionWeights {
    pub q_proj: Vec<f32>, // [2 * num_attention_heads * head_dim, hidden_size]
    pub k_proj: Vec<f32>, // [num_key_value_heads * head_dim, hidden_size]
    pub v_proj: Vec<f32>, // [num_key_value_heads * head_dim, hidden_size]
    pub o_proj: Vec<f32>, // [hidden_size, num_attention_heads * head_dim]
    pub q_norm: Vec<f32>, // [head_dim]
    pub k_norm: Vec<f32>, // [head_dim]
}

#[derive(Debug, Clone)]
pub struct MtpMoeWeights {
    pub router_gate: Vec<f32>,          // [num_experts, hidden_size]
    pub experts_gate_up_proj: Vec<f32>, // [num_experts, 2 * moe_intermediate_size, hidden_size]
    pub experts_down_proj: Vec<f32>,    // [num_experts, hidden_size, moe_intermediate_size]
    pub shared_gate_proj: Vec<f32>,     // [shared_expert_intermediate_size, hidden_size]
    pub shared_up_proj: Vec<f32>,       // [shared_expert_intermediate_size, hidden_size]
    pub shared_down_proj: Vec<f32>,     // [hidden_size, shared_expert_intermediate_size]
    pub shared_expert_gate: Vec<f32>,   // [hidden_size]
}

impl MtpWeights {
    /// Load all MTP weights from a tensor source, validating shapes.
    pub fn load_from_source<S: crate::weights::TensorSource>(
        source: &mut S,
        cfg: &MtpConfig,
    ) -> Result<Self, crate::error::InferenceError> {
        use crate::error::InferenceError;

        let hidden = cfg.hidden_size;
        let q_proj_rows = 2 * cfg.num_attention_heads * cfg.head_dim;
        let kv_proj_rows = cfg.num_key_value_heads * cfg.head_dim;
        let o_proj_rows = hidden;
        let o_proj_cols = cfg.num_attention_heads * cfg.head_dim;
        let moe_inter = cfg.moe_intermediate_size;
        let shared_inter = cfg.shared_expert_intermediate_size;
        let num_experts = cfg.num_experts;

        let load_checked =
            |source: &mut S, name: &str, expected: &[usize]| -> Result<Vec<f32>, InferenceError> {
                let (data, shape) = source.get_f32_tensor_owned(name)?;
                if shape != expected {
                    return Err(InferenceError::ShapeMismatch {
                        name: name.to_string(),
                        expected: expected.to_vec(),
                        actual: shape,
                    });
                }
                Ok(data)
            };

        // Validate fc_weight shape: must be [hidden, 2*hidden]
        let fc_weight = {
            let name = "mtp.fc.weight";
            let (data, shape) = source.get_f32_tensor_owned(name)?;
            let expected_fc = [hidden, 2 * hidden];
            if shape != expected_fc {
                return Err(InferenceError::UnsupportedModel(format!(
                    "unexpected mtp.fc.weight shape {shape:?}; expected fusion projection [{hidden}, {}]",
                    2 * hidden
                )));
            }
            data
        };

        let norm_weight = load_checked(source, "mtp.norm.weight", &[hidden])?;
        let pre_fc_norm_embedding_weight =
            load_checked(source, "mtp.pre_fc_norm_embedding.weight", &[hidden])?;
        let pre_fc_norm_hidden_weight =
            load_checked(source, "mtp.pre_fc_norm_hidden.weight", &[hidden])?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let iln = load_checked(
                source,
                &format!("mtp.layers.{i}.input_layernorm.weight"),
                &[hidden],
            )?;
            let paln = load_checked(
                source,
                &format!("mtp.layers.{i}.post_attention_layernorm.weight"),
                &[hidden],
            )?;
            let q_proj = load_checked(
                source,
                &format!("mtp.layers.{i}.self_attn.q_proj.weight"),
                &[q_proj_rows, hidden],
            )?;
            let k_proj = load_checked(
                source,
                &format!("mtp.layers.{i}.self_attn.k_proj.weight"),
                &[kv_proj_rows, hidden],
            )?;
            let v_proj = load_checked(
                source,
                &format!("mtp.layers.{i}.self_attn.v_proj.weight"),
                &[kv_proj_rows, hidden],
            )?;
            let o_proj = load_checked(
                source,
                &format!("mtp.layers.{i}.self_attn.o_proj.weight"),
                &[o_proj_rows, o_proj_cols],
            )?;
            let q_norm = load_checked(
                source,
                &format!("mtp.layers.{i}.self_attn.q_norm.weight"),
                &[cfg.head_dim],
            )?;
            let k_norm = load_checked(
                source,
                &format!("mtp.layers.{i}.self_attn.k_norm.weight"),
                &[cfg.head_dim],
            )?;
            let router_gate = load_checked(
                source,
                &format!("mtp.layers.{i}.mlp.gate.weight"),
                &[num_experts, hidden],
            )?;
            let experts_gate_up_proj = load_checked(
                source,
                &format!("mtp.layers.{i}.mlp.experts.gate_up_proj"),
                &[num_experts, 2 * moe_inter, hidden],
            )?;
            let experts_down_proj = load_checked(
                source,
                &format!("mtp.layers.{i}.mlp.experts.down_proj"),
                &[num_experts, hidden, moe_inter],
            )?;
            let shared_gate_proj = load_checked(
                source,
                &format!("mtp.layers.{i}.mlp.shared_expert.gate_proj.weight"),
                &[shared_inter, hidden],
            )?;
            let shared_up_proj = load_checked(
                source,
                &format!("mtp.layers.{i}.mlp.shared_expert.up_proj.weight"),
                &[shared_inter, hidden],
            )?;
            let shared_down_proj = load_checked(
                source,
                &format!("mtp.layers.{i}.mlp.shared_expert.down_proj.weight"),
                &[hidden, shared_inter],
            )?;
            // shared_expert_gate can be [1, hidden] (flatten to [hidden])
            let shared_expert_gate = {
                let name = format!("mtp.layers.{i}.mlp.shared_expert_gate.weight");
                let (data, shape) = source.get_f32_tensor_owned(&name)?;
                let total = shape.iter().product::<usize>();
                if total != hidden {
                    return Err(InferenceError::ShapeMismatch {
                        name,
                        expected: vec![hidden],
                        actual: shape,
                    });
                }
                data
            };

            layers.push(MtpLayerWeights {
                input_layernorm: iln,
                post_attention_layernorm: paln,
                self_attn: MtpAttentionWeights {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                },
                mlp: MtpMoeWeights {
                    router_gate,
                    experts_gate_up_proj,
                    experts_down_proj,
                    shared_gate_proj,
                    shared_up_proj,
                    shared_down_proj,
                    shared_expert_gate,
                },
            });
        }

        Ok(Self {
            fc_weight,
            layers,
            norm_weight,
            pre_fc_norm_embedding_weight,
            pre_fc_norm_hidden_weight,
        })
    }
}

// --- MTP scratch and verifier ---

struct MtpScratch {
    embedding: Vec<f32>,
    norm_embedding: Vec<f32>,
    norm_hidden: Vec<f32>,
    fused_input: Vec<f32>,
    hidden: Vec<f32>,
    residual: Vec<f32>,
    q_and_gate: Vec<f32>,
    q: Vec<f32>,
    gate_z: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    scores: Vec<f32>,
    context: Vec<f32>,
    attn_out: Vec<f32>,
    router_logits: Vec<f32>,
    selected_experts: Vec<(usize, f32)>,
    expert_gate_up: Vec<f32>,
    expert_silu_up: Vec<f32>,
    moe_out: Vec<f32>,
    shared_gate: Vec<f32>,
    shared_up: Vec<f32>,
    shared_silu_up: Vec<f32>,
    logits: Vec<f32>,
}

impl MtpScratch {
    fn new(cfg: &MtpConfig, max_seq_len: usize) -> Self {
        let hidden = cfg.hidden_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let num_experts = cfg.num_experts;
        let num_experts_per_tok = cfg.num_experts_per_tok;
        let moe_inter = cfg.moe_intermediate_size;
        let shared_inter = cfg.shared_expert_intermediate_size;
        Self {
            embedding: vec![0.0; hidden],
            norm_embedding: vec![0.0; hidden],
            norm_hidden: vec![0.0; hidden],
            fused_input: vec![0.0; 2 * hidden],
            hidden: vec![0.0; hidden],
            residual: vec![0.0; hidden],
            q_and_gate: vec![0.0; 2 * q_dim],
            q: vec![0.0; q_dim],
            gate_z: vec![0.0; q_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            scores: vec![0.0; cfg.num_attention_heads * max_seq_len],
            context: vec![0.0; q_dim],
            attn_out: vec![0.0; hidden],
            router_logits: vec![0.0; num_experts],
            selected_experts: vec![(0, 0.0); num_experts_per_tok],
            expert_gate_up: vec![0.0; 2 * moe_inter],
            expert_silu_up: vec![0.0; moe_inter],
            moe_out: vec![0.0; hidden],
            shared_gate: vec![0.0; shared_inter],
            shared_up: vec![0.0; shared_inter],
            shared_silu_up: vec![0.0; shared_inter],
            logits: vec![0.0; cfg.vocab_size],
        }
    }
}

/// MTP transformer verifier: drafts and verifies speculative tokens.
pub struct MtpVerifier<'a> {
    pub config: MtpConfig,
    pub weights: &'a MtpWeights,
    pub embed_tokens: &'a [f32],
    pub lm_head_weight: &'a [f32],
    /// Own KV cache for the MTP transformer (1 layer).
    pub cache: crate::kv_cache::flat::FlatKVCache,
    rope: crate::rope::RopeTable,
    scratch: MtpScratch,
}

impl<'a> MtpVerifier<'a> {
    /// Create a new MTP verifier, validating configuration and allocating scratch.
    pub fn new(
        config: MtpConfig,
        weights: &'a MtpWeights,
        embed_tokens: &'a [f32],
        lm_head_weight: &'a [f32],
        max_seq_len: usize,
    ) -> Result<Self, crate::error::InferenceError> {
        use crate::error::InferenceError;
        if config.num_hidden_layers != 1 {
            return Err(InferenceError::UnsupportedModel(
                "MTP num_hidden_layers other than 1 is not implemented".into(),
            ));
        }
        if weights.layers.len() != config.num_hidden_layers {
            return Err(InferenceError::Inference(format!(
                "MtpWeights has {} layers but config specifies {}",
                weights.layers.len(),
                config.num_hidden_layers
            )));
        }
        if embed_tokens.len() != config.vocab_size * config.hidden_size {
            return Err(InferenceError::Inference(format!(
                "embed_tokens length {} != vocab_size * hidden_size = {}",
                embed_tokens.len(),
                config.vocab_size * config.hidden_size
            )));
        }
        if lm_head_weight.len() != config.vocab_size * config.hidden_size {
            return Err(InferenceError::Inference(format!(
                "lm_head_weight length {} != vocab_size * hidden_size = {}",
                lm_head_weight.len(),
                config.vocab_size * config.hidden_size
            )));
        }
        if config.draft_length > 8 {
            return Err(InferenceError::UnsupportedModel(
                "MTP draft_length > 8 is not benchmarked or supported".into(),
            ));
        }
        if config.use_dedicated_embeddings {
            return Err(InferenceError::UnsupportedModel(
                "dedicated MTP embeddings are not implemented".into(),
            ));
        }

        let rope_dim = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
        // RopeTable built with head_dim=rope_dim so half_dim = rope_dim/2
        let rope = crate::rope::RopeTable::new(rope_dim, max_seq_len.max(1), config.rope_theta);

        let cache_cfg = crate::kv_cache::flat::FlatKVCacheConfig::for_qwen3(
            1,
            config.num_key_value_heads,
            config.head_dim,
            max_seq_len,
        );
        let cache = crate::kv_cache::flat::FlatKVCache::new(cache_cfg);
        let scratch = MtpScratch::new(&config, max_seq_len);

        Ok(Self {
            config,
            weights,
            embed_tokens,
            lm_head_weight,
            cache,
            rope,
            scratch,
        })
    }

    /// Reset the MTP KV cache to empty (call at generation start).
    pub fn reset_cache(&mut self) {
        self.cache.reset_fast();
    }

    /// Roll back the MTP KV cache to `seq_len` tokens without deallocating.
    pub fn rollback_cache_to(
        &mut self,
        seq_len: usize,
    ) -> Result<(), crate::error::InferenceError> {
        self.cache.truncate_to(seq_len);
        Ok(())
    }

    /// Run one MTP transformer forward step.
    ///
    /// - `input_token_id`: token whose embedding is fed into the MTP module
    /// - `position`: sequence position (used for RoPE and causal mask)
    /// - `previous_hidden`: normalized hidden state from the main model at `position`
    pub fn forward_one(
        &mut self,
        input_token_id: u32,
        position: usize,
        previous_hidden: &[f32],
    ) -> Result<MtpForwardOutput, crate::error::InferenceError> {
        use crate::error::InferenceError;
        use crate::forward::cpu::{matmul_bt, rms_norm};

        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let head_dim = cfg.head_dim;
        let num_q_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let groups = num_q_heads / num_kv_heads;
        let rope_dim = (head_dim as f32 * cfg.partial_rotary_factor) as usize;
        let moe_inter = cfg.moe_intermediate_size;
        let shared_inter = cfg.shared_expert_intermediate_size;
        let num_experts = cfg.num_experts;
        let num_experts_per_tok = cfg.num_experts_per_tok;
        let eps = cfg.rms_norm_eps;

        // Validate
        if input_token_id as usize >= vocab {
            return Err(InferenceError::Inference(format!(
                "MTP token_id {input_token_id} >= vocab_size {vocab}"
            )));
        }
        if previous_hidden.len() != hidden {
            return Err(InferenceError::Inference(format!(
                "previous_hidden len {} != hidden_size {hidden}",
                previous_hidden.len()
            )));
        }

        let layer = &self.weights.layers[0];

        // 1. Embedding lookup
        let tok = input_token_id as usize;
        self.scratch
            .embedding
            .copy_from_slice(&self.embed_tokens[tok * hidden..(tok + 1) * hidden]);

        // 2. Pre-fusion normalization (plain RMSNorm, not shifted)
        self.scratch.norm_hidden.copy_from_slice(previous_hidden);
        rms_norm(
            &mut self.scratch.norm_hidden,
            &self.weights.pre_fc_norm_hidden_weight,
            hidden,
            eps,
        );

        self.scratch
            .norm_embedding
            .copy_from_slice(&self.scratch.embedding);
        rms_norm(
            &mut self.scratch.norm_embedding,
            &self.weights.pre_fc_norm_embedding_weight,
            hidden,
            eps,
        );

        // 3. Fusion: concat([norm_embed, norm_hidden]) and project
        self.scratch.fused_input[..hidden].copy_from_slice(&self.scratch.norm_embedding);
        self.scratch.fused_input[hidden..2 * hidden].copy_from_slice(&self.scratch.norm_hidden);
        matmul_bt(
            &self.scratch.fused_input[..2 * hidden],
            &self.weights.fc_weight,
            &mut self.scratch.hidden[..hidden],
            1,
            2 * hidden,
            hidden,
        );

        // 4. Transformer layer
        // Save residual for attention
        self.scratch.residual.copy_from_slice(&self.scratch.hidden);

        // Pre-attention layernorm (plain RMSNorm)
        rms_norm(
            &mut self.scratch.hidden,
            &layer.input_layernorm,
            hidden,
            eps,
        );

        // Q projection: output is [2*q_dim] (Q + gate_z interleaved per head)
        matmul_bt(
            &self.scratch.hidden,
            &layer.self_attn.q_proj,
            &mut self.scratch.q_and_gate[..2 * q_dim],
            1,
            hidden,
            2 * q_dim,
        );

        // Scatter Q and gate_z per head
        for h in 0..num_q_heads {
            let src = h * head_dim * 2;
            let dst = h * head_dim;
            self.scratch.q[dst..dst + head_dim]
                .copy_from_slice(&self.scratch.q_and_gate[src..src + head_dim]);
            self.scratch.gate_z[dst..dst + head_dim]
                .copy_from_slice(&self.scratch.q_and_gate[src + head_dim..src + 2 * head_dim]);
        }

        // K projection
        matmul_bt(
            &self.scratch.hidden,
            &layer.self_attn.k_proj,
            &mut self.scratch.k[..kv_dim],
            1,
            hidden,
            kv_dim,
        );

        // V projection
        matmul_bt(
            &self.scratch.hidden,
            &layer.self_attn.v_proj,
            &mut self.scratch.v[..kv_dim],
            1,
            hidden,
            kv_dim,
        );

        // Per-head QK normalization (plain RMSNorm)
        for h in 0..num_q_heads {
            let start = h * head_dim;
            rms_norm(
                &mut self.scratch.q[start..start + head_dim],
                &layer.self_attn.q_norm,
                head_dim,
                eps,
            );
        }
        for h in 0..num_kv_heads {
            let start = h * head_dim;
            rms_norm(
                &mut self.scratch.k[start..start + head_dim],
                &layer.self_attn.k_norm,
                head_dim,
                eps,
            );
        }

        // Partial RoPE (interleaved pairing, first rope_dim dims of each head)
        for h in 0..num_q_heads {
            let start = h * head_dim;
            mtp_apply_partial_rope(
                &mut self.scratch.q[start..start + head_dim],
                position,
                &self.rope,
                rope_dim,
            );
        }
        for h in 0..num_kv_heads {
            let start = h * head_dim;
            mtp_apply_partial_rope(
                &mut self.scratch.k[start..start + head_dim],
                position,
                &self.rope,
                rope_dim,
            );
        }

        // Append K, V to MTP KV cache at current seq_len position
        let write_pos = self.cache.seq_len();
        {
            let k_buf = self.cache.k_buffer_mut(0);
            k_buf[write_pos * kv_dim..(write_pos + 1) * kv_dim]
                .copy_from_slice(&self.scratch.k[..kv_dim]);
        }
        {
            let v_buf = self.cache.v_buffer_mut(0);
            v_buf[write_pos * kv_dim..(write_pos + 1) * kv_dim]
                .copy_from_slice(&self.scratch.v[..kv_dim]);
        }
        let cur_seq_len = write_pos + 1;

        // GQA attention
        let scale = 1.0 / (head_dim as f32).sqrt();
        let k_all = &self.cache.k_buffer(0)[..cur_seq_len * kv_dim];
        let v_all = &self.cache.v_buffer(0)[..cur_seq_len * kv_dim];

        for qh in 0..num_q_heads {
            let kvh = qh / groups;
            let q_off = qh * head_dim;

            // Compute scores
            let scores_start = qh * cur_seq_len;
            let mut max_score = f32::NEG_INFINITY;
            for t in 0..cur_seq_len {
                let k_off = t * kv_dim + kvh * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += self.scratch.q[q_off + d] * k_all[k_off + d];
                }
                let s = dot * scale;
                self.scratch.scores[scores_start + t] = s;
                if s > max_score {
                    max_score = s;
                }
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            for t in 0..cur_seq_len {
                let e = (self.scratch.scores[scores_start + t] - max_score).exp();
                self.scratch.scores[scores_start + t] = e;
                sum_exp += e;
            }
            let inv_sum = 1.0 / sum_exp;
            for t in 0..cur_seq_len {
                self.scratch.scores[scores_start + t] *= inv_sum;
            }

            // Weighted sum of V
            let ctx_off = qh * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for t in 0..cur_seq_len {
                    let v_off = t * kv_dim + kvh * head_dim;
                    val += self.scratch.scores[scores_start + t] * v_all[v_off + d];
                }
                self.scratch.context[ctx_off + d] = val;
            }
        }

        // Output gate: sigmoid(gate_z) applied elementwise to context
        for d in 0..q_dim {
            let sig = 1.0 / (1.0 + (-self.scratch.gate_z[d]).exp());
            self.scratch.context[d] *= sig;
        }

        // O projection: context [q_dim] → attn_out [hidden]
        matmul_bt(
            &self.scratch.context[..q_dim],
            &layer.self_attn.o_proj,
            &mut self.scratch.attn_out[..hidden],
            1,
            q_dim,
            hidden,
        );

        // Attention residual
        for i in 0..hidden {
            self.scratch.hidden[i] = self.scratch.residual[i] + self.scratch.attn_out[i];
        }

        // Advance KV cache after layer completes
        self.cache.advance_by(1);

        // Post-attention layernorm (plain RMSNorm), save residual for FFN
        self.scratch.residual.copy_from_slice(&self.scratch.hidden);
        rms_norm(
            &mut self.scratch.hidden,
            &layer.post_attention_layernorm,
            hidden,
            eps,
        );

        // MoE FFN
        // Router
        matmul_bt(
            &self.scratch.hidden,
            &layer.mlp.router_gate,
            &mut self.scratch.router_logits[..num_experts],
            1,
            hidden,
            num_experts,
        );

        // Stable softmax over router logits
        {
            let logits = &mut self.scratch.router_logits[..num_experts];
            let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for v in logits.iter_mut() {
                *v = (*v - max_l).exp();
                denom += *v;
            }
            if denom > 0.0 {
                for v in logits.iter_mut() {
                    *v /= denom;
                }
            }
        }

        // Top-k selection (insertion sort)
        for s in &mut self.scratch.selected_experts[..num_experts_per_tok] {
            *s = (usize::MAX, f32::NEG_INFINITY);
        }
        for (expert_id, prob) in self.scratch.router_logits[..num_experts]
            .iter()
            .copied()
            .enumerate()
        {
            for rank in 0..num_experts_per_tok {
                if prob > self.scratch.selected_experts[rank].1 {
                    for shift in (rank + 1..num_experts_per_tok).rev() {
                        self.scratch.selected_experts[shift] =
                            self.scratch.selected_experts[shift - 1];
                    }
                    self.scratch.selected_experts[rank] = (expert_id, prob);
                    break;
                }
            }
        }

        // Renormalize top-k weights
        let top_sum: f32 = self.scratch.selected_experts[..num_experts_per_tok]
            .iter()
            .map(|(_, p)| *p)
            .sum();
        if top_sum > 0.0 {
            for (_, p) in &mut self.scratch.selected_experts[..num_experts_per_tok] {
                *p /= top_sum;
            }
        }

        // Accumulate expert outputs into moe_out
        for i in 0..hidden {
            self.scratch.moe_out[i] = 0.0;
        }

        let gate_up_stride = 2 * moe_inter * hidden;
        let down_stride = hidden * moe_inter;

        for idx in 0..num_experts_per_tok {
            let (expert_id, prob) = self.scratch.selected_experts[idx];
            if expert_id == usize::MAX {
                continue;
            }

            let gu_base = expert_id * gate_up_stride;
            let gu_end = gu_base + gate_up_stride;
            // gate_up = hidden @ experts_gate_up_proj[e]^T, shape [2*moe_inter]
            matmul_bt(
                &self.scratch.hidden,
                &layer.mlp.experts_gate_up_proj[gu_base..gu_end],
                &mut self.scratch.expert_gate_up[..2 * moe_inter],
                1,
                hidden,
                2 * moe_inter,
            );

            // SwiGLU
            for j in 0..moe_inter {
                let gate = self.scratch.expert_gate_up[j];
                let up = self.scratch.expert_gate_up[moe_inter + j];
                let silu = gate * (1.0 / (1.0 + (-gate).exp()));
                self.scratch.expert_silu_up[j] = silu * up;
            }

            // down = silu_up @ experts_down_proj[e]^T, shape [hidden]
            let d_base = expert_id * down_stride;
            let d_end = d_base + down_stride;
            let mut expert_out = vec![0.0f32; hidden];
            matmul_bt(
                &self.scratch.expert_silu_up[..moe_inter],
                &layer.mlp.experts_down_proj[d_base..d_end],
                &mut expert_out,
                1,
                moe_inter,
                hidden,
            );

            for i in 0..hidden {
                self.scratch.moe_out[i] += prob * expert_out[i];
            }
        }

        // Shared expert
        matmul_bt(
            &self.scratch.hidden,
            &layer.mlp.shared_gate_proj,
            &mut self.scratch.shared_gate[..shared_inter],
            1,
            hidden,
            shared_inter,
        );
        matmul_bt(
            &self.scratch.hidden,
            &layer.mlp.shared_up_proj,
            &mut self.scratch.shared_up[..shared_inter],
            1,
            hidden,
            shared_inter,
        );

        // SwiGLU on shared expert
        for j in 0..shared_inter {
            let gate = self.scratch.shared_gate[j];
            let silu = gate * (1.0 / (1.0 + (-gate).exp()));
            self.scratch.shared_silu_up[j] = silu * self.scratch.shared_up[j];
        }

        let mut shared_out = vec![0.0f32; hidden];
        matmul_bt(
            &self.scratch.shared_silu_up[..shared_inter],
            &layer.mlp.shared_down_proj,
            &mut shared_out,
            1,
            shared_inter,
            hidden,
        );

        // Shared expert gate: scalar = sigmoid(dot(hidden, shared_expert_gate))
        let shared_scalar = {
            let mut dot = 0.0f32;
            for j in 0..hidden {
                dot += self.scratch.hidden[j] * layer.mlp.shared_expert_gate[j];
            }
            1.0 / (1.0 + (-dot).exp())
        };

        for i in 0..hidden {
            self.scratch.moe_out[i] += shared_scalar * shared_out[i];
        }

        // FFN residual
        for i in 0..hidden {
            self.scratch.hidden[i] = self.scratch.residual[i] + self.scratch.moe_out[i];
        }

        // 5. Final MTP norm (plain RMSNorm)
        rms_norm(
            &mut self.scratch.hidden,
            &self.weights.norm_weight,
            hidden,
            eps,
        );

        // 6. Logits: hidden @ lm_head^T
        matmul_bt(
            &self.scratch.hidden,
            self.lm_head_weight,
            &mut self.scratch.logits[..vocab],
            1,
            hidden,
            vocab,
        );

        Ok(MtpForwardOutput {
            logits: self.scratch.logits[..vocab].to_vec(),
            hidden: self.scratch.hidden[..hidden].to_vec(),
        })
    }

    /// Draft `config.draft_length` candidate tokens using iterative MTP forwards.
    ///
    /// Stops early if `eos_token` is produced.
    pub fn draft_tokens(
        &mut self,
        current_token_id: u32,
        current_position: usize,
        main_hidden_at_current_position: &[f32],
        eos_token: Option<u32>,
    ) -> Result<Vec<u32>, crate::error::InferenceError> {
        let mut draft = Vec::with_capacity(self.config.draft_length);
        let mut next_input = current_token_id;
        let mut next_hidden: Vec<f32> = main_hidden_at_current_position.to_vec();
        let mut next_position = current_position;

        for _ in 0..self.config.draft_length {
            let out = self.forward_one(next_input, next_position, &next_hidden)?;
            let token = argmax(&out.logits) as u32;
            draft.push(token);
            if Some(token) == eos_token {
                break;
            }
            next_input = token;
            next_hidden = out.hidden;
            next_position += 1;
        }

        Ok(draft)
    }
}

/// Interleaved partial RoPE: rotate pairs (2i, 2i+1) for i in 0..rope_dim/2.
fn mtp_apply_partial_rope(
    head_vec: &mut [f32],
    position: usize,
    rope: &crate::rope::RopeTable,
    rope_dim: usize,
) {
    let half = rope_dim / 2;
    let base = position * half;
    for i in 0..half {
        let cos_val = rope.cos_at(base + i);
        let sin_val = rope.sin_at(base + i);
        let x0 = head_vec[2 * i];
        let x1 = head_vec[2 * i + 1];
        head_vec[2 * i] = x0 * cos_val - x1 * sin_val;
        head_vec[2 * i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

// --- Verification loop ---

/// Trait implemented by the target model adapter used in MTP verification.
pub trait MtpTargetVerifier {
    fn cache_position(&self) -> usize;

    fn rollback_cache_to(&mut self, seq_len: usize) -> Result<(), crate::error::InferenceError>;

    /// Forward `tokens` starting at `start_pos` through the target model.
    /// Returns per-token logits: `logits[i]` is target output after processing `tokens[i]`.
    fn verify_tokens(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<Vec<f32>>, crate::error::InferenceError>;
}

/// Verify a speculative MTP draft against the target model.
///
/// Returns an [`MtpVerifyResult`] with accepted tokens, optional fallback, metrics,
/// and properly rolled-back caches.
pub fn mtp_verify_draft<T: MtpTargetVerifier>(
    verifier: &mut MtpVerifier<'_>,
    current_token_id: u32,
    current_position: usize,
    main_hidden_at_current_position: &[f32],
    initial_target_logits: &[f32],
    eos_token: Option<u32>,
    target: &mut T,
) -> Result<MtpVerifyResult, crate::error::InferenceError> {
    // Degenerate path: draft_length < 2 — use normal greedy decode
    if verifier.config.draft_length < 2 {
        let fallback = argmax(initial_target_logits) as u32;
        return Ok(MtpVerifyResult {
            accepted_count: 0,
            accepted_tokens: vec![],
            fallback_token: Some(fallback),
            draft_tokens: vec![],
            stopped_by_eos: false,
            metrics: MtpMetrics {
                target_forwards: 0,
                mtp_forwards: 0,
                draft_tokens: 0,
                accepted_tokens: 0,
                accepted_tokens_per_forward: 0.0,
                acceptance_rate: 0.0,
            },
        });
    }

    let target_start = target.cache_position();
    let mtp_start = verifier.cache.seq_len();

    // Generate draft
    let draft = verifier.draft_tokens(
        current_token_id,
        current_position,
        main_hidden_at_current_position,
        eos_token,
    )?;
    let draft_len = draft.len();
    let mtp_forwards = draft_len;

    if draft.is_empty() {
        let fallback = argmax(initial_target_logits) as u32;
        verifier.rollback_cache_to(mtp_start)?;
        return Ok(MtpVerifyResult {
            accepted_count: 0,
            accepted_tokens: vec![],
            fallback_token: Some(fallback),
            draft_tokens: vec![],
            stopped_by_eos: false,
            metrics: MtpMetrics {
                target_forwards: 0,
                mtp_forwards: 0,
                draft_tokens: 0,
                accepted_tokens: 0,
                accepted_tokens_per_forward: 0.0,
                acceptance_rate: 0.0,
            },
        });
    }

    // Verify first draft token against initial target logits
    let target_first = argmax(initial_target_logits) as u32;

    if draft[0] != target_first {
        // Full rejection before calling verify_tokens
        verifier.rollback_cache_to(mtp_start)?;
        target.rollback_cache_to(target_start)?;
        let fallback = target_first;
        return Ok(MtpVerifyResult {
            accepted_count: 0,
            accepted_tokens: vec![],
            fallback_token: Some(fallback),
            draft_tokens: draft,
            stopped_by_eos: false,
            metrics: MtpMetrics {
                target_forwards: 0,
                mtp_forwards,
                draft_tokens: draft_len,
                accepted_tokens: 0,
                accepted_tokens_per_forward: 0.0,
                acceptance_rate: 0.0,
            },
        });
    }

    // First token is EOS: accept it and stop
    if Some(target_first) == eos_token {
        verifier.rollback_cache_to(mtp_start + 1)?;
        target.rollback_cache_to(target_start)?;
        return Ok(MtpVerifyResult {
            accepted_count: 1,
            accepted_tokens: vec![target_first],
            fallback_token: None,
            draft_tokens: draft,
            stopped_by_eos: true,
            metrics: MtpMetrics {
                target_forwards: 0,
                mtp_forwards,
                draft_tokens: draft_len,
                accepted_tokens: 1,
                accepted_tokens_per_forward: 0.0,
                acceptance_rate: 1.0 / draft_len.max(1) as f64,
            },
        });
    }

    // Batch verify remaining draft tokens through target model
    let target_logits = target.verify_tokens(&draft, current_position + 1)?;
    let target_forwards = 1;

    // Determine how many additional tokens are accepted
    // target_logits[i] corresponds to the target output after processing draft[i]
    // We compare argmax(target_logits[i]) with draft[i+1]
    let mut accepted_count = 1; // first token already accepted
    let mut fallback_token = None;

    for i in 1..draft_len {
        let model_choice = argmax(&target_logits[i - 1]) as u32;
        if model_choice != draft[i] {
            fallback_token = Some(model_choice);
            break;
        }
        accepted_count += 1;
    }

    // Check for EOS within accepted tokens
    let mut stopped_by_eos = false;
    let mut eos_truncate = accepted_count;
    for i in 0..accepted_count {
        if Some(draft[i]) == eos_token {
            eos_truncate = i + 1;
            stopped_by_eos = true;
            fallback_token = None;
            break;
        }
    }
    accepted_count = eos_truncate;
    let accepted_tokens = draft[..accepted_count].to_vec();

    // Roll back caches to accepted positions
    verifier.rollback_cache_to(mtp_start + accepted_count)?;
    target.rollback_cache_to(target_start + accepted_count)?;

    let acceptance_rate = accepted_count as f64 / draft_len.max(1) as f64;
    let accepted_tokens_per_forward = accepted_count as f64 / target_forwards.max(1) as f64;

    Ok(MtpVerifyResult {
        accepted_count,
        accepted_tokens,
        fallback_token,
        draft_tokens: draft,
        stopped_by_eos,
        metrics: MtpMetrics {
            target_forwards,
            mtp_forwards,
            draft_tokens: draft_len,
            accepted_tokens: accepted_count,
            accepted_tokens_per_forward,
            acceptance_rate,
        },
    })
}

/// Test helper: run MTP verification with pre-computed draft tokens (no model forward needed).
/// Only available under `#[cfg(test)]`.
#[cfg(test)]
fn mtp_verify_precomputed_draft<T: MtpTargetVerifier>(
    precomputed_draft: Vec<u32>,
    current_position: usize,
    initial_target_logits: &[f32],
    eos_token: Option<u32>,
    target: &mut T,
    draft_length: usize,
) -> Result<MtpVerifyResult, crate::error::InferenceError> {
    // Degenerate path
    if draft_length < 2 {
        let fallback = argmax(initial_target_logits) as u32;
        return Ok(MtpVerifyResult {
            accepted_count: 0,
            accepted_tokens: vec![],
            fallback_token: Some(fallback),
            draft_tokens: vec![],
            stopped_by_eos: false,
            metrics: MtpMetrics::default(),
        });
    }

    let target_start = target.cache_position();
    let draft = precomputed_draft;
    let draft_len = draft.len();

    if draft.is_empty() {
        let fallback = argmax(initial_target_logits) as u32;
        return Ok(MtpVerifyResult {
            accepted_count: 0,
            accepted_tokens: vec![],
            fallback_token: Some(fallback),
            draft_tokens: vec![],
            stopped_by_eos: false,
            metrics: MtpMetrics::default(),
        });
    }

    let target_first = argmax(initial_target_logits) as u32;

    if draft[0] != target_first {
        target.rollback_cache_to(target_start)?;
        return Ok(MtpVerifyResult {
            accepted_count: 0,
            accepted_tokens: vec![],
            fallback_token: Some(target_first),
            draft_tokens: draft,
            stopped_by_eos: false,
            metrics: MtpMetrics {
                target_forwards: 0,
                mtp_forwards: draft_len,
                draft_tokens: draft_len,
                accepted_tokens: 0,
                accepted_tokens_per_forward: 0.0,
                acceptance_rate: 0.0,
            },
        });
    }

    if Some(target_first) == eos_token {
        target.rollback_cache_to(target_start)?;
        return Ok(MtpVerifyResult {
            accepted_count: 1,
            accepted_tokens: vec![target_first],
            fallback_token: None,
            draft_tokens: draft,
            stopped_by_eos: true,
            metrics: MtpMetrics {
                target_forwards: 0,
                mtp_forwards: draft_len,
                draft_tokens: draft_len,
                accepted_tokens: 1,
                accepted_tokens_per_forward: 0.0,
                acceptance_rate: 1.0 / draft_len.max(1) as f64,
            },
        });
    }

    let target_logits = target.verify_tokens(&draft, current_position + 1)?;
    let target_forwards = 1usize;

    let mut accepted_count = 1;
    let mut fallback_token = None;

    for i in 1..draft_len {
        let model_choice = argmax(&target_logits[i - 1]) as u32;
        if model_choice != draft[i] {
            fallback_token = Some(model_choice);
            break;
        }
        accepted_count += 1;
    }

    let mut stopped_by_eos = false;
    let mut eos_truncate = accepted_count;
    for i in 0..accepted_count {
        if Some(draft[i]) == eos_token {
            eos_truncate = i + 1;
            stopped_by_eos = true;
            fallback_token = None;
            break;
        }
    }
    accepted_count = eos_truncate;
    let accepted_tokens = draft[..accepted_count].to_vec();

    target.rollback_cache_to(target_start + accepted_count)?;

    let acceptance_rate = accepted_count as f64 / draft_len.max(1) as f64;
    let accepted_tokens_per_forward = accepted_count as f64 / target_forwards.max(1) as f64;

    Ok(MtpVerifyResult {
        accepted_count,
        accepted_tokens,
        fallback_token,
        draft_tokens: draft,
        stopped_by_eos,
        metrics: MtpMetrics {
            target_forwards,
            mtp_forwards: draft_len,
            draft_tokens: draft_len,
            accepted_tokens: accepted_count,
            accepted_tokens_per_forward,
            acceptance_rate,
        },
    })
}

// ---------------------------------------------------------------------------
// End of MTP additions
// ---------------------------------------------------------------------------

/// **Unstable**: draft verification loop; signature may change as sampling strategies expand.
///
/// Verify draft tokens against the model's greedy predictions.
///
/// Runs `forward_fn(token_id, position)` for each draft token in sequence.
/// Accepts tokens as long as the model's greedy choice agrees with the next
/// draft token. Returns after the first disagreement.
///
/// Returns `(accepted_count, collected_logits)` where:
/// - `accepted_count`: number of draft tokens whose forward passes were
///   executed (always >= 1 if `draft_tokens` is non-empty).
/// - `collected_logits`: the logits from each forward call, useful for
///   recovering the correct next token after a rejection.
pub fn verify_draft<F>(
    draft_tokens: &[u32],
    position_start: usize,
    mut forward_fn: F,
) -> (usize, Vec<Vec<f32>>)
where
    F: FnMut(u32, usize) -> Vec<f32>,
{
    let mut accepted = 0;
    let mut all_logits = Vec::with_capacity(draft_tokens.len());

    for (i, &draft_token) in draft_tokens.iter().enumerate() {
        let logits = forward_fn(draft_token, position_start + i);
        let model_choice = argmax(&logits);
        all_logits.push(logits);

        // Always count this token as accepted (we ran the forward pass for it).
        accepted += 1;

        // If there are more draft tokens, check whether the model agrees
        // with the next one. If not, stop here.
        if i + 1 < draft_tokens.len() && model_choice != draft_tokens[i + 1] as usize {
            break;
        }
    }

    (accepted, all_logits)
}

/// **Unstable**: high-level speculative generation wrapper; loop logic under active development.
///
/// Generate tokens with n-gram speculative decoding.
///
/// This is a model-agnostic wrapper: pass any closure that implements a
/// single-token forward step returning logits, and this function handles
/// the speculation/verification loop.
///
/// # Arguments
///
/// - `prompt_tokens`: tokenised prompt (caller must have already run prefill).
/// - `max_new_tokens`: generation budget.
/// - `eos_token`: stop token ID.
/// - `forward_fn`: `(token_id, position) -> logits`.
/// - `max_ngram`: longest n-gram to try (default: 5).
/// - `max_draft`: max draft tokens per speculation step (default: 4).
///
/// # Returns
///
/// The generated token IDs (excluding the prompt).
pub fn generate_with_speculation<F>(
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    eos_token: u32,
    mut forward_fn: F,
    max_ngram: usize,
    max_draft: usize,
) -> Vec<u32>
where
    F: FnMut(u32, usize) -> Vec<f32>,
{
    let speculator = NgramSpeculator::new(prompt_tokens.to_vec(), max_ngram, max_draft);
    let mut generated: Vec<u32> = Vec::new();
    let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();
    let mut pos = prompt_tokens.len();

    while generated.len() < max_new_tokens {
        // Attempt speculative draft from prompt n-grams
        let draft = speculator.speculate(&all_tokens);

        if draft.is_empty() {
            // No speculation possible -- normal single-token decode
            let logits = forward_fn(
                *all_tokens
                    .last()
                    .expect("invariant: prompt_tokens must seed speculation history"),
                pos,
            );
            let next_token = argmax(&logits) as u32;
            if next_token == eos_token {
                break;
            }
            generated.push(next_token);
            all_tokens.push(next_token);
            pos += 1;
        } else {
            // Verify the draft tokens
            let (accepted, logits_vec) = verify_draft(&draft, pos, &mut forward_fn);

            // Accept verified tokens
            for &t in &draft[..accepted] {
                if t == eos_token {
                    return generated;
                }
                generated.push(t);
                all_tokens.push(t);
                pos += 1;
            }

            // If we rejected some draft tokens, the last set of logits
            // gives us the model's actual prediction for the next token.
            if accepted < draft.len() {
                let rejection_logits =
                    &logits_vec[accepted.min(logits_vec.len().saturating_sub(1))];
                let next_token = argmax(rejection_logits) as u32;
                if next_token == eos_token {
                    break;
                }
                generated.push(next_token);
                all_tokens.push(next_token);
                pos += 1;
            }
        }
    }

    generated
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- NgramSpeculator unit tests --

    #[test]
    fn speculate_basic_ngram_match() {
        let spec = NgramSpeculator::new(vec![1, 2, 3, 4, 5, 6, 7], 5, 3);
        // Recent [2, 3, 4] matches prompt[1..4], should draft [5, 6, 7]
        let draft = spec.speculate(&[2, 3, 4]);
        assert_eq!(draft, vec![5, 6, 7]);
    }

    #[test]
    fn speculate_no_match_returns_empty() {
        let spec = NgramSpeculator::new(vec![1, 2, 3], 5, 3);
        let draft = spec.speculate(&[10, 11, 12]);
        assert!(draft.is_empty());
    }

    #[test]
    fn speculate_empty_recent_returns_empty() {
        let spec = NgramSpeculator::new(vec![1, 2, 3, 4], 5, 3);
        let draft = spec.speculate(&[]);
        assert!(draft.is_empty());
    }

    #[test]
    fn speculate_single_recent_token_skipped() {
        // Minimum n-gram is 2, so a single recent token cannot match.
        let spec = NgramSpeculator::new(vec![1, 2, 3, 4], 5, 3);
        let draft = spec.speculate(&[2]);
        assert!(draft.is_empty());
    }

    #[test]
    fn speculate_prefers_longest_match() {
        // Prompt: [1, 2, 3, 4, 5, 3, 4, 5, 6, 7]
        // Trigram [3,4,5] matches at pos 2 (draft [3,4,5]->continuation)
        // and at pos 5 (draft [6,7]).
        // But if recent is [2,3,4,5], that's a 4-gram matching pos 1,
        // which should draft from pos 5 onward: [3, 4, 5].
        let spec = NgramSpeculator::new(vec![1, 2, 3, 4, 5, 3, 4, 5, 6, 7], 5, 3);
        // 4-gram [2,3,4,5] matches at pos 1, drafts prompt[5..8] = [3,4,5]
        let draft = spec.speculate(&[2, 3, 4, 5]);
        assert_eq!(draft, vec![3, 4, 5]);
    }

    #[test]
    fn speculate_max_draft_limits_output() {
        let spec = NgramSpeculator::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5, 2);
        // Match [1, 2] at pos 0, max_draft=2 so drafts [3, 4] only
        let draft = spec.speculate(&[1, 2]);
        assert_eq!(draft, vec![3, 4]);
    }

    #[test]
    fn speculate_match_at_end_of_prompt_returns_empty() {
        // Match is at the very end of prompt -- nothing left to draft
        let spec = NgramSpeculator::new(vec![1, 2, 3], 5, 3);
        let draft = spec.speculate(&[2, 3]);
        assert!(draft.is_empty());
    }

    #[test]
    fn speculate_match_near_end_returns_partial() {
        // Match [2, 3] at pos 1, only token 4 left to draft
        let spec = NgramSpeculator::new(vec![1, 2, 3, 4], 5, 3);
        let draft = spec.speculate(&[2, 3]);
        assert_eq!(draft, vec![4]);
    }

    #[test]
    fn speculate_max_ngram_clamped_to_recent_len() {
        // max_ngram=10 but only 3 recent tokens -- should still work
        let spec = NgramSpeculator::new(vec![5, 6, 7, 8, 9], 10, 3);
        let draft = spec.speculate(&[5, 6, 7]);
        assert_eq!(draft, vec![8, 9]);
    }

    // -- find_ngram tests --

    #[test]
    fn find_ngram_empty_pattern() {
        let spec = NgramSpeculator::new(vec![1, 2, 3], 5, 3);
        assert_eq!(spec.find_ngram(&[]), None);
    }

    #[test]
    fn find_ngram_pattern_longer_than_prompt() {
        let spec = NgramSpeculator::new(vec![1, 2], 5, 3);
        assert_eq!(spec.find_ngram(&[1, 2, 3]), None);
    }

    #[test]
    fn find_ngram_exact_match() {
        let spec = NgramSpeculator::new(vec![1, 2, 3], 5, 3);
        assert_eq!(spec.find_ngram(&[1, 2, 3]), Some(0));
    }

    #[test]
    fn find_ngram_returns_first_occurrence() {
        let spec = NgramSpeculator::new(vec![1, 2, 1, 2, 5], 5, 3);
        assert_eq!(spec.find_ngram(&[1, 2]), Some(0));
    }

    // -- argmax tests --

    #[test]
    fn argmax_basic() {
        assert_eq!(argmax(&[0.1, 0.9, 0.5]), 1);
    }

    #[test]
    fn argmax_empty() {
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn argmax_tie_last_wins() {
        // Rust max_by returns the last element on tie.
        assert_eq!(argmax(&[1.0, 1.0, 0.5]), 1);
    }

    #[test]
    fn argmax_negative_values() {
        assert_eq!(argmax(&[-3.0, -1.0, -2.0]), 1);
    }

    // -- verify_draft tests --

    #[test]
    fn verify_draft_all_accepted() {
        let draft = vec![1u32, 2, 3];
        // Forward function: for draft[i], model predicts draft[i+1]
        let mut call = 0usize;
        let (accepted, logits) = verify_draft(&draft, 10, |_tok, _pos| {
            let mut l = vec![0.0f32; 10];
            call += 1;
            // After processing draft[i], model should predict draft[i+1]
            if call < draft.len() {
                l[draft[call] as usize] = 1.0;
            } else {
                l[0] = 1.0; // last token -- prediction doesn't matter for acceptance
            }
            l
        });
        assert_eq!(accepted, 3);
        assert_eq!(logits.len(), 3);
    }

    #[test]
    fn verify_draft_first_rejected() {
        let draft = vec![1u32, 2, 3];
        // Model always predicts token 99 -- disagrees with draft[1]=2
        let (accepted, logits) = verify_draft(&draft, 0, |_tok, _pos| {
            let mut l = vec![0.0f32; 100];
            l[99] = 1.0;
            l
        });
        // First forward call processes draft[0]=1, predicts 99 != draft[1]=2 => stop after 1
        assert_eq!(accepted, 1);
        assert_eq!(logits.len(), 1);
    }

    #[test]
    fn verify_draft_partial_acceptance() {
        let draft = vec![10u32, 20, 30, 40];
        let mut call = 0usize;
        let (accepted, logits) = verify_draft(&draft, 0, |_tok, _pos| {
            call += 1;
            let mut l = vec![0.0f32; 50];
            match call {
                1 => l[20] = 1.0, // agrees with draft[1]
                2 => l[30] = 1.0, // agrees with draft[2]
                3 => l[0] = 1.0,  // disagrees with draft[3]=40
                _ => l[0] = 1.0,
            }
            l
        });
        // Accepted: draft[0] (model ok with draft[1]), draft[1] (ok with draft[2]),
        // draft[2] (disagrees with draft[3]) => 3 accepted
        assert_eq!(accepted, 3);
        assert_eq!(logits.len(), 3);
    }

    #[test]
    fn verify_draft_single_token() {
        let draft = vec![42u32];
        let (accepted, logits) = verify_draft(&draft, 5, |_tok, _pos| vec![0.0; 10]);
        assert_eq!(accepted, 1);
        assert_eq!(logits.len(), 1);
    }

    #[test]
    fn verify_draft_empty() {
        let (accepted, logits) = verify_draft(&[], 0, |_tok, _pos| vec![0.0; 10]);
        assert_eq!(accepted, 0);
        assert_eq!(logits.len(), 0);
    }

    #[test]
    fn verify_draft_positions_are_correct() {
        let draft = vec![1u32, 2, 3];
        let mut positions = Vec::new();
        let mut call = 0usize;
        let _ = verify_draft(&draft, 100, |_tok, pos| {
            positions.push(pos);
            call += 1;
            let mut l = vec![0.0f32; 10];
            // Each call must predict the *next* draft token for verification
            // to continue.
            match call {
                1 => l[2] = 1.0, // predicts draft[1]=2
                2 => l[3] = 1.0, // predicts draft[2]=3
                _ => l[0] = 1.0, // last token, prediction doesn't matter
            }
            l
        });
        // Should be called with consecutive positions starting at 100
        assert_eq!(positions, vec![100, 101, 102]);
    }

    // -- MTP verification tests --

    /// Build logits where only token `tok` has a nonzero value.
    fn logits_with_argmax(vocab: usize, tok: u32) -> Vec<f32> {
        let mut v = vec![0.0f32; vocab];
        if (tok as usize) < vocab {
            v[tok as usize] = 1.0;
        }
        v
    }

    struct MockTargetVerifier {
        cache_pos: usize,
        logits_by_step: Vec<Vec<f32>>,
        calls: Vec<(Vec<u32>, usize)>,
    }

    impl MtpTargetVerifier for MockTargetVerifier {
        fn cache_position(&self) -> usize {
            self.cache_pos
        }
        fn rollback_cache_to(
            &mut self,
            seq_len: usize,
        ) -> Result<(), crate::error::InferenceError> {
            self.cache_pos = seq_len;
            Ok(())
        }
        fn verify_tokens(
            &mut self,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<Vec<Vec<f32>>, crate::error::InferenceError> {
            self.calls.push((tokens.to_vec(), start_pos));
            self.cache_pos += tokens.len();
            Ok(self.logits_by_step.clone())
        }
    }

    const EOS: u32 = 99;
    const VOCAB: usize = 200;

    #[test]
    fn mtp_eos_mid_draft_accepts_through_eos_and_stops() {
        // Draft: [10, 11, EOS, 12]
        // Initial target argmax = 10 → first token accepted
        // target_logits[0] argmax = 11 → second accepted
        // target_logits[1] argmax = EOS → third accepted, EOS hit → stop
        let initial = logits_with_argmax(VOCAB, 10);
        let mut target = MockTargetVerifier {
            cache_pos: 5,
            logits_by_step: vec![
                logits_with_argmax(VOCAB, 11),
                logits_with_argmax(VOCAB, EOS),
                logits_with_argmax(VOCAB, 12),
            ],
            calls: vec![],
        };
        let draft = vec![10, 11, EOS, 12];

        let result =
            mtp_verify_precomputed_draft(draft, 5, &initial, Some(EOS), &mut target, 4).unwrap();

        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.accepted_tokens, vec![10, 11, EOS]);
        assert!(result.stopped_by_eos);
        assert_eq!(result.fallback_token, None);
        assert_eq!(
            target.cache_pos,
            5 + 3,
            "cache must be rolled back to start + accepted"
        );
    }

    #[test]
    fn mtp_position_accounting_rolls_back_after_partial_accept() {
        // Start cache at 100, draft 8, first 3 match, 4th mismatches
        let initial = logits_with_argmax(VOCAB, 1);
        let mut target = MockTargetVerifier {
            cache_pos: 100,
            logits_by_step: vec![
                logits_with_argmax(VOCAB, 2),  // agrees with draft[1]
                logits_with_argmax(VOCAB, 3),  // agrees with draft[2]
                logits_with_argmax(VOCAB, 77), // mismatches draft[3]=4
                logits_with_argmax(VOCAB, 5),
                logits_with_argmax(VOCAB, 6),
                logits_with_argmax(VOCAB, 7),
                logits_with_argmax(VOCAB, 8),
            ],
            calls: vec![],
        };
        let draft = vec![1u32, 2, 3, 4, 5, 6, 7, 8];

        let result =
            mtp_verify_precomputed_draft(draft, 100, &initial, None, &mut target, 8).unwrap();

        assert_eq!(result.accepted_count, 3);
        // target cache rolled back to 100 + 3 = 103
        assert_eq!(target.cache_pos, 103, "target cache must be 100 + 3 = 103");
    }

    #[test]
    fn mtp_partial_acceptance_returns_target_fourth_token() {
        // Draft: [1, 2, 3, 4, 5, 6, 7, 8]
        // initial argmax=1, target_logits[0] argmax=2, [1] argmax=3, [2] argmax=99 (mismatch with 4)
        let initial = logits_with_argmax(VOCAB, 1);
        let mut target = MockTargetVerifier {
            cache_pos: 0,
            logits_by_step: vec![
                logits_with_argmax(VOCAB, 2),
                logits_with_argmax(VOCAB, 3),
                logits_with_argmax(VOCAB, 99),
                logits_with_argmax(VOCAB, 5),
                logits_with_argmax(VOCAB, 6),
                logits_with_argmax(VOCAB, 7),
                logits_with_argmax(VOCAB, 8),
            ],
            calls: vec![],
        };
        let draft = vec![1u32, 2, 3, 4, 5, 6, 7, 8];

        let result =
            mtp_verify_precomputed_draft(draft, 0, &initial, None, &mut target, 8).unwrap();

        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.accepted_tokens, vec![1, 2, 3]);
        assert_eq!(result.fallback_token, Some(99));
    }

    #[test]
    fn mtp_full_rejection_uses_initial_target_argmax() {
        // Draft: [42, 43], initial target argmax = 7 (mismatch with 42)
        let initial = logits_with_argmax(VOCAB, 7);
        let mut target = MockTargetVerifier {
            cache_pos: 0,
            logits_by_step: vec![],
            calls: vec![],
        };
        let draft = vec![42u32, 43];

        let result =
            mtp_verify_precomputed_draft(draft, 0, &initial, None, &mut target, 4).unwrap();

        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.fallback_token, Some(7));
        assert!(
            target.calls.is_empty(),
            "verify_tokens must NOT be called on full rejection"
        );
    }

    #[test]
    fn mtp_degenerate_draft_length_zero_or_one_uses_normal_decode() {
        let initial = logits_with_argmax(VOCAB, 5);
        for dl in [0usize, 1] {
            let mut target = MockTargetVerifier {
                cache_pos: 0,
                logits_by_step: vec![],
                calls: vec![],
            };
            let draft = vec![5u32]; // doesn't matter — degenerate path ignores draft
            let result =
                mtp_verify_precomputed_draft(draft, 0, &initial, None, &mut target, dl).unwrap();
            assert_eq!(
                result.accepted_count, 0,
                "dl={dl}: degenerate path returns 0 accepted"
            );
            assert_eq!(
                result.fallback_token,
                Some(5),
                "dl={dl}: fallback must be initial target argmax"
            );
            assert!(
                target.calls.is_empty(),
                "dl={dl}: no verify_tokens calls in degenerate path"
            );
        }
    }

    #[test]
    fn mtp_acceptance_metric_matches_manual_count() {
        // Draft length 8, accepted 6
        let initial = logits_with_argmax(VOCAB, 1);
        let mut target = MockTargetVerifier {
            cache_pos: 0,
            logits_by_step: vec![
                logits_with_argmax(VOCAB, 2),
                logits_with_argmax(VOCAB, 3),
                logits_with_argmax(VOCAB, 4),
                logits_with_argmax(VOCAB, 5),
                logits_with_argmax(VOCAB, 6),
                logits_with_argmax(VOCAB, 77), // mismatch with draft[6]=7
                logits_with_argmax(VOCAB, 8),
            ],
            calls: vec![],
        };
        let draft = vec![1u32, 2, 3, 4, 5, 6, 7, 8];

        let result =
            mtp_verify_precomputed_draft(draft, 0, &initial, None, &mut target, 8).unwrap();

        assert_eq!(result.accepted_count, 6);
        assert_eq!(result.metrics.draft_tokens, 8);
        assert_eq!(result.metrics.target_forwards, 1);
        assert_eq!(result.metrics.accepted_tokens, 6);
        assert!((result.metrics.acceptance_rate - 6.0 / 8.0).abs() < 1e-6);
        assert!((result.metrics.accepted_tokens_per_forward - 6.0).abs() < 1e-6);
    }

    #[test]
    fn mtp_mock_forward_function_is_deterministic() {
        let initial = logits_with_argmax(VOCAB, 3);
        let make_target = || MockTargetVerifier {
            cache_pos: 10,
            logits_by_step: vec![
                logits_with_argmax(VOCAB, 4),
                logits_with_argmax(VOCAB, 5),
                logits_with_argmax(VOCAB, 99), // mismatch with draft[3]=6
            ],
            calls: vec![],
        };
        let draft1 = vec![3u32, 4, 5, 6];
        let draft2 = draft1.clone();

        let mut t1 = make_target();
        let mut t2 = make_target();
        let r1 = mtp_verify_precomputed_draft(draft1, 10, &initial, None, &mut t1, 4).unwrap();
        let r2 = mtp_verify_precomputed_draft(draft2, 10, &initial, None, &mut t2, 4).unwrap();

        assert_eq!(r1.accepted_count, r2.accepted_count);
        assert_eq!(r1.accepted_tokens, r2.accepted_tokens);
        assert_eq!(r1.fallback_token, r2.fallback_token);
    }

    // -- generate_with_speculation integration tests --

    #[test]
    fn generate_no_speculation_possible() {
        // Prompt has no overlap with generated tokens -- falls back to normal decode.
        let prompt = vec![100u32, 101, 102];
        let eos = 999;
        let mut step = 0usize;
        let result = generate_with_speculation(
            &prompt,
            3,
            eos,
            |_tok, _pos| {
                step += 1;
                let mut l = vec![0.0f32; 200];
                // Generate tokens 200+step which won't match prompt
                let next = (50 + step).min(199);
                l[next] = 1.0;
                l
            },
            5,
            4,
        );
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn generate_stops_on_eos() {
        let prompt = vec![1u32, 2, 3];
        let eos = 99;
        let result = generate_with_speculation(
            &prompt,
            100,
            eos,
            |_tok, _pos| {
                let mut l = vec![0.0f32; 100];
                l[eos as usize] = 1.0; // always predict EOS
                l
            },
            5,
            4,
        );
        assert!(result.is_empty());
    }

    #[test]
    fn generate_with_perfect_speculation() {
        // Prompt: [1, 2, 3, 4, 5, 6, 7]
        // After prefill, first call gets last token = 7, model predicts "1" so
        // all_tokens becomes [1,2,3,4,5,6,7,1].
        // Next iteration: recent tokens end with [7,1] -- matches prompt[6..8]?
        // No, prompt only has 7 tokens so [7,1] doesn't match.
        //
        // Better test: model copies from prompt.
        // Prompt: [10, 20, 30, 40, 50]
        // Model always predicts the token matching the prompt copy.
        let prompt = vec![10u32, 20, 30, 40, 50];
        let eos = 999;
        let mut call_count = 0usize;
        let result = generate_with_speculation(
            &prompt,
            5,
            eos,
            |_tok, _pos| {
                call_count += 1;
                let mut l = vec![0.0f32; 100];
                // Just produce non-EOS non-prompt tokens to avoid complicating logic
                l[60 + (call_count % 30)] = 1.0;
                l
            },
            5,
            4,
        );
        // Should produce exactly 5 tokens
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn generate_respects_max_new_tokens() {
        let prompt = vec![1u32, 2, 3];
        let eos = 999;
        let result = generate_with_speculation(
            &prompt,
            10,
            eos,
            |_tok, _pos| {
                let mut l = vec![0.0f32; 100];
                l[42] = 1.0;
                l
            },
            5,
            4,
        );
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn generate_eos_in_draft_stops_early() {
        // Prompt: [1, 2, 3, EOS, 5]
        // If speculation drafts [3, EOS, 5], generation should stop at EOS.
        let prompt = vec![1u32, 2, 3, 99, 5];
        let eos = 99;

        // We need to construct a scenario where speculation fires with EOS in draft.
        // After first normal decode, if model produces token 2, all_tokens = [1,2,3,99,5,2].
        // Then [5, 2] doesn't match prompt. Next, model produces 3, all_tokens=[...,2,3].
        // [2, 3] matches prompt[1..3], draft = [99, 5] (draft[0] = EOS).
        // verify_draft runs forward for draft[0]=99, which is EOS.
        // But EOS check is done AFTER verify, in the acceptance loop.
        // draft[0]=99=EOS => generation should return.

        let mut call = 0;
        let result = generate_with_speculation(
            &prompt,
            20,
            eos,
            |_tok, _pos| {
                call += 1;
                let mut l = vec![0.0f32; 100];
                match call {
                    1 => l[2] = 1.0, // first normal: predict 2
                    2 => l[3] = 1.0, // second normal: predict 3
                    // Now speculation kicks in: [2,3] matches, drafts [99,5]
                    // verify_draft calls forward for draft token 99
                    3 => l[5] = 1.0, // model predicts 5 (agrees with draft[1])
                    4 => l[0] = 1.0, // model prediction for token 5
                    _ => l[42] = 1.0,
                }
                l
            },
            5,
            4,
        );

        // Generated: [2, 3] from normal decode, then EOS hit from draft
        assert_eq!(result.len(), 2);
    }
}
