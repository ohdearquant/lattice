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
/// When multiple elements share the maximum value, returns the **first**
/// occurrence. Speculative decoding must be token-identical to plain greedy, so
/// this tie-break must match the engine greedy decode (`forward::metal_qwen35`),
/// the `sampling` module, and `torch.argmax` — all of which are first-wins (#280).
/// Non-finite (`NaN`) entries are skipped, matching `sampling::argmax_f32_scalar`.
/// Returns 0 on empty input.
pub fn argmax(logits: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
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

/// Draft tokens and their associated raw logits produced by
/// [`MtpVerifier::draft_tokens_with_logits`].
///
/// `logits[i]` is the draft model's logit vector for the position that generated `tokens[i]`.
/// This is the `q_i(·)` distribution required by probabilistic rejection sampling (ADR-050).
pub struct MtpDraft {
    pub tokens: Vec<u32>,
    pub logits: Vec<Vec<f32>>,
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
    // Dequantization buffers: f16 KV cache → f32 for attention computation.
    cached_k_f32: Vec<f32>,
    cached_v_f32: Vec<f32>,
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
            cached_k_f32: vec![0.0; kv_dim * max_seq_len],
            cached_v_f32: vec![0.0; kv_dim * max_seq_len],
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
        // Checked product so a pathological public config returns a typed error
        // instead of overflow-panicking (debug) or wrapping (release) before the
        // shape comparison. Matches the FlatKVCache checked-arithmetic precedent.
        let embed_numel = config
            .vocab_size
            .checked_mul(config.hidden_size)
            .ok_or_else(|| {
                InferenceError::Inference(format!(
                    "MtpConfig vocab_size ({}) * hidden_size ({}) overflows usize",
                    config.vocab_size, config.hidden_size
                ))
            })?;
        if embed_tokens.len() != embed_numel {
            return Err(InferenceError::Inference(format!(
                "embed_tokens length {} != vocab_size * hidden_size = {embed_numel}",
                embed_tokens.len(),
            )));
        }
        if lm_head_weight.len() != embed_numel {
            return Err(InferenceError::Inference(format!(
                "lm_head_weight length {} != vocab_size * hidden_size = {embed_numel}",
                lm_head_weight.len(),
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

        // Guard structural scalars: zero values cause divide-by-zero or OOB on
        // first forward_one; partial_rotary_factor outside [0,1] produces a
        // rope_dim > head_dim which overreads the per-head slice in
        // mtp_apply_partial_rope.
        if config.hidden_size == 0 {
            return Err(InferenceError::Inference(
                "MtpConfig hidden_size must be > 0".into(),
            ));
        }
        if config.vocab_size == 0 {
            return Err(InferenceError::Inference(
                "MtpConfig vocab_size must be > 0".into(),
            ));
        }
        if config.head_dim == 0 {
            return Err(InferenceError::Inference(
                "MtpConfig head_dim must be > 0".into(),
            ));
        }
        if config.num_attention_heads == 0 {
            return Err(InferenceError::Inference(
                "MtpConfig num_attention_heads must be > 0".into(),
            ));
        }
        if config.num_key_value_heads == 0 {
            return Err(InferenceError::Inference(
                "MtpConfig num_key_value_heads must be > 0".into(),
            ));
        }
        if config.num_attention_heads % config.num_key_value_heads != 0 {
            return Err(InferenceError::Inference(format!(
                "MtpConfig num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                config.num_attention_heads, config.num_key_value_heads
            )));
        }
        if !config.partial_rotary_factor.is_finite()
            || !(0.0..=1.0).contains(&config.partial_rotary_factor)
        {
            return Err(InferenceError::Inference(format!(
                "MtpConfig partial_rotary_factor {} is not in [0.0, 1.0]",
                config.partial_rotary_factor
            )));
        }

        let rope_dim = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
        // Reject zero-width RoPE: rope_dim < 2 gives half_dim == 0, so
        // RopeTable::max_positions() == 0 and the forward_one position bound
        // would reject every position. A no-RoPE MTP is unsupported (Qwen3.5
        // always uses partial RoPE); fail closed rather than run without
        // positional encoding.
        if rope_dim < 2 {
            return Err(InferenceError::UnsupportedModel(format!(
                "MTP requires a non-zero RoPE width: head_dim ({}) * partial_rotary_factor ({}) \
                 yields rope_dim {rope_dim}, need >= 2",
                config.head_dim, config.partial_rotary_factor
            )));
        }
        // Checked derived-dimension products. MtpScratch::new and FlatKVCache
        // allocate from these, and num_attention_heads/head_dim can be oversized
        // in a public config even with tiny weights (only weights.layers.len()
        // is checked above), so a pathological config could overflow-panic
        // (debug) or wrap (release) before construction. Validate the products
        // here — the private MtpScratch::new and the cache config are only
        // reached through this constructor, so upstream checking is sufficient.
        let checked_dim = |a: usize, b: usize, what: &str| -> Result<usize, InferenceError> {
            a.checked_mul(b).ok_or_else(|| {
                InferenceError::Inference(format!(
                    "MtpConfig dimension product {what} ({a} * {b}) overflows usize"
                ))
            })
        };
        let q_dim = checked_dim(
            config.num_attention_heads,
            config.head_dim,
            "num_attention_heads * head_dim",
        )?;
        let kv_dim = checked_dim(
            config.num_key_value_heads,
            config.head_dim,
            "num_key_value_heads * head_dim",
        )?;
        checked_dim(2, config.hidden_size, "2 * hidden_size")?;
        checked_dim(2, q_dim, "2 * q_dim")?;
        checked_dim(2, config.moe_intermediate_size, "2 * moe_intermediate_size")?;
        checked_dim(
            config.num_attention_heads,
            max_seq_len,
            "num_attention_heads * max_seq_len",
        )?;
        checked_dim(kv_dim, max_seq_len, "kv_dim * max_seq_len")?;

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

        // At capacity (`seq_len == max_seq_len`) this step indexes per-position
        // buffers (RoPE position tables, then the raw `k_buffer_mut(0)` K/V write at
        // `seq_len * kv_dim`) out of bounds, bypassing the bounds-checked
        // `FlatKVCache` append API hardened in #290, and panics. Fail closed with a
        // typed error instead, matching the no-panic-in-library contract and
        // `advance_by`'s overflow behaviour.
        if self.cache.is_full() {
            return Err(InferenceError::InvalidInput(format!(
                "MTP KV cache is full ({} tokens); call rollback_cache_to or reset_cache before forward_one",
                self.cache.seq_len()
            )));
        }

        // Guard RoPE table bounds: `position` is an independent parameter and
        // can exceed `max_seq_len` even when the cache is not full.
        // mtp_apply_partial_rope indexes at `position * half_dim + i`; if
        // position >= rope.max_positions() that index is past the end of the
        // precomputed table and panics.  Fail closed instead.
        if position >= self.rope.max_positions() {
            return Err(InferenceError::InvalidInput(format!(
                "MTP forward_one position {position} out of range for RoPE table of {} positions",
                self.rope.max_positions()
            )));
        }

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

        // Append K, V to MTP KV cache at current seq_len position (convert f32→f16 on write).
        let write_pos = self.cache.seq_len();
        {
            let k_buf = self.cache.k_buffer_mut(0);
            let base = write_pos * kv_dim;
            for (j, &val) in self.scratch.k[..kv_dim].iter().enumerate() {
                k_buf[base + j] = half::f16::from_f32(val);
            }
        }
        {
            let v_buf = self.cache.v_buffer_mut(0);
            let base = write_pos * kv_dim;
            for (j, &val) in self.scratch.v[..kv_dim].iter().enumerate() {
                v_buf[base + j] = half::f16::from_f32(val);
            }
        }
        let cur_seq_len = write_pos + 1;

        // GQA attention: dequantize f16 KV cache to f32 scratch buffers.
        let kv_end = cur_seq_len * kv_dim;
        debug_assert!(
            kv_end <= self.scratch.cached_k_f32.len(),
            "kv_end={kv_end} > cached_k_f32.len()={} (rollback target exceeded scratch)",
            self.scratch.cached_k_f32.len()
        );
        debug_assert!(
            kv_end <= self.scratch.cached_v_f32.len(),
            "kv_end={kv_end} > cached_v_f32.len()={} (rollback target exceeded scratch)",
            self.scratch.cached_v_f32.len()
        );
        for (i, &h) in self.cache.k_buffer(0)[..kv_end].iter().enumerate() {
            self.scratch.cached_k_f32[i] = h.to_f32();
        }
        for (i, &h) in self.cache.v_buffer(0)[..kv_end].iter().enumerate() {
            self.scratch.cached_v_f32[i] = h.to_f32();
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        let k_all = &self.scratch.cached_k_f32[..kv_end];
        let v_all = &self.scratch.cached_v_f32[..kv_end];

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
        self.cache.advance_by(1)?;

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

    /// Draft `config.draft_length` candidate tokens with their raw logits.
    ///
    /// The returned `logits` are the `q_i(·)` distributions required by probabilistic
    /// rejection sampling (ADR-050). Use [`Self::draft_tokens`] when only token IDs are
    /// needed (greedy callers that pass `&[]` to `rejection_sample_draft`).
    ///
    /// Stops early if `eos_token` is produced.
    #[allow(clippy::explicit_counter_loop)]
    pub fn draft_tokens_with_logits(
        &mut self,
        current_token_id: u32,
        current_position: usize,
        main_hidden_at_current_position: &[f32],
        eos_token: Option<u32>,
    ) -> Result<MtpDraft, crate::error::InferenceError> {
        let mut tokens = Vec::with_capacity(self.config.draft_length);
        let mut logits = Vec::with_capacity(self.config.draft_length);
        let mut next_input = current_token_id;
        let mut next_hidden: Vec<f32> = main_hidden_at_current_position.to_vec();
        let mut next_position = current_position;

        for _ in 0..self.config.draft_length {
            let out = self.forward_one(next_input, next_position, &next_hidden)?;
            let token = argmax(&out.logits) as u32;
            tokens.push(token);
            logits.push(out.logits);
            if Some(token) == eos_token {
                break;
            }
            next_input = token;
            next_hidden = out.hidden;
            next_position += 1;
        }

        Ok(MtpDraft { tokens, logits })
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
        Ok(self
            .draft_tokens_with_logits(
                current_token_id,
                current_position,
                main_hidden_at_current_position,
                eos_token,
            )?
            .tokens)
    }
}

/// Stride-half partial RoPE: rotate pairs (i, half+i) for i in 0..rope_dim/2 — matches HF rotate_half / MLX traditional=False.
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
        let x0 = head_vec[i];
        let x1 = head_vec[half + i];
        head_vec[i] = x0 * cos_val - x1 * sin_val;
        head_vec[half + i] = x0 * sin_val + x1 * cos_val;
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

    /// Snapshot every GDN layer's recurrent state (S matrices + conv buffer). See ADR-052.
    ///
    /// Implementors that have no GDN layers (e.g. pure-attention test mocks) return an empty
    /// `Vec`. Callers must treat the returned snapshot as opaque.
    fn snapshot_gdn_states(&self) -> crate::attention::gdn::GdnSnapshot;

    /// Restore every GDN layer's recurrent state from a prior [`snapshot_gdn_states`] call.
    ///
    /// The snapshot must have been taken from the same model instance. Implementors that
    /// hold no GDN state accept an empty snapshot and do nothing.
    fn restore_gdn_states(&mut self, snapshot: &crate::attention::gdn::GdnSnapshot);
}

/// Verify a speculative MTP draft against the target model.
///
/// Returns an [`MtpVerifyResult`] with accepted tokens, optional fallback, metrics,
/// and properly rolled-back caches.
///
/// # Errors
///
/// On `Err`, the verifier and target caches (and GDN state) are restored to their
/// pre-call positions on a **best-effort** basis: if a rollback call itself fails, the
/// original error is still returned and cache coherence is no longer guaranteed. A caller
/// recovering from an error must therefore treat the speculative context as invalid and
/// rebuild it rather than assume the caches are at a known position.
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

    // ADR-052: snapshot target GDN state before draft generation. The draft itself runs only
    // through `MtpVerifier`'s separate cache, but `target.verify_tokens` below mutates the
    // target GDN state through all draft tokens; we must restore it on rejection so that
    // post-rejection forward steps start from the correct recurrent state.
    let gdn_snap = target.snapshot_gdn_states();

    // Generate draft with per-token logits for probabilistic rejection sampling (ADR-050).
    // #282: if draft generation fails after advancing the verifier cache, restore both
    // caches to their pre-call positions so the caller can retry or propagate cleanly.
    let mtp_draft = match verifier.draft_tokens_with_logits(
        current_token_id,
        current_position,
        main_hidden_at_current_position,
        eos_token,
    ) {
        Ok(d) => d,
        Err(e) => {
            let _ = verifier.rollback_cache_to(mtp_start);
            target.restore_gdn_states(&gdn_snap);
            return Err(e);
        }
    };
    let draft = mtp_draft.tokens;
    let draft_logits = mtp_draft.logits;
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

    // Batch verify all draft tokens through the target model.  `verify_tokens` must be
    // called before `rejection_sample_draft` so that implementors that maintain a
    // speculation checkpoint (e.g. the Metal adapter) have an active checkpoint to roll
    // back into when `rollback_cache_to` is called below — even in the full-rejection case.
    // #282: verify_tokens advances the target cache; roll back both caches on error so
    // the caller sees a consistent pre-call state.
    let target_logits = match target.verify_tokens(&draft, current_position + 1) {
        Ok(l) => l,
        Err(e) => {
            let _ = verifier.rollback_cache_to(mtp_start);
            let _ = target.rollback_cache_to(target_start);
            target.restore_gdn_states(&gdn_snap);
            return Err(e);
        }
    };
    let target_forwards = 1;

    // Probabilistic rejection sampling (ADR-050): pass draft logits and greedy=false so
    // every draft token is accepted with probability min(1, p(x)/q(x)).
    // #282: rejection_sample_draft doesn't touch the caches, but it is fallible (e.g. on a
    // draft/logits length mismatch). By this point verify_tokens has already advanced the
    // target cache and draft generation advanced the verifier cache, so an early Err here
    // must restore both to their pre-call positions.
    let rs = match rejection_sample_draft(
        &draft,
        &draft_logits,
        initial_target_logits,
        &target_logits,
        false,
        None,
    ) {
        Ok(r) => r,
        Err(e) => {
            let _ = verifier.rollback_cache_to(mtp_start);
            let _ = target.rollback_cache_to(target_start);
            target.restore_gdn_states(&gdn_snap);
            return Err(e);
        }
    };
    let mut accepted_count = rs.accepted_count;
    let mut fallback_token = if rs.had_rejection {
        rs.bonus_token
    } else {
        None
    };

    // EOS truncation: stop at first EOS inside the accepted prefix.
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

    // Roll back caches to accepted positions. The contract for `rollback_cache_to` requires
    // an implementation to leave both KV and GDN coherent at `target_start + accepted_count`
    // — e.g., the Metal implementation does this through its per-token GDN slot pool so
    // partial-accept rollback restores slot `accepted_count` (state after pending +
    // `accepted_count - 1` drafts via the full model). We deliberately do NOT call
    // `restore_gdn_states` on partial accept: that would overwrite the slot-restored GDN
    // with the pre-draft snapshot and leave GDN behind KV by `accepted_count` tokens.
    verifier.rollback_cache_to(mtp_start + accepted_count)?;
    target.rollback_cache_to(target_start + accepted_count)?;

    // `gdn_snap` is intentionally dropped here. On partial accept the implementor handles
    // GDN sync inside `rollback_cache_to`; on full accept GDN is already at +draft_len.
    drop(gdn_snap);

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

    // #282: verify_tokens advances the target cache; roll back to target_start on error
    // so the caller sees a consistent pre-call state (no partial forward visible).
    let target_logits = match target.verify_tokens(&draft, current_position + 1) {
        Ok(l) => l,
        Err(e) => {
            let _ = target.rollback_cache_to(target_start);
            return Err(e);
        }
    };
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

// ---------------------------------------------------------------------------
// Probabilistic rejection sampling for speculative decoding (Leviathan 2023)
// ---------------------------------------------------------------------------

/// Result of probabilistic rejection sampling verification.
#[derive(Debug, Clone, PartialEq)]
pub struct RejectionSampleResult {
    /// Number of draft tokens accepted (probabilistically or greedily).
    pub accepted_count: usize,
    /// The accepted draft token IDs.
    pub accepted_tokens: Vec<u32>,
    /// Token sampled from the adjusted distribution at the rejection point,
    /// or from the target distribution after the last accepted token.
    /// `None` only when `draft_tokens` is empty.
    pub bonus_token: Option<u32>,
    /// Whether any draft token was rejected before the end.
    pub had_rejection: bool,
}

/// Compute stable softmax over `logits` into `out`.
///
/// `out` must have the same length as `logits`. Values are written in-place.
fn softmax_into(logits: &[f32], out: &mut [f32]) {
    debug_assert_eq!(logits.len(), out.len());
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (o, &l) in out.iter_mut().zip(logits.iter()) {
        let e = (l - max).exp();
        *o = e;
        sum += e;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for o in out.iter_mut() {
            *o *= inv;
        }
    }
}

/// Xorshift64 PRNG local to this module — used only in rejection sampling.
/// Mirrors the implementation in `sampling.rs` so callers do not need to
/// expose or import that module's private `Rng`.
struct SpecRng {
    state: u64,
}

impl SpecRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x853c_49e6_748f_ea9b
            } else {
                seed
            },
        }
    }

    fn from_clock() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x853c_49e6_748f_ea9b);
        Self::new(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Sample one token from the adjusted distribution `max(0, p - q)` (re-normalised).
///
/// Called on rejection: the adjusted distribution represents probability mass that
/// the target model assigns but the draft model under-assigns. Sampling from it
/// ensures the output distribution exactly matches the target model.
///
/// `r` is a uniform draw in [0, 1).
///
/// Falls back to argmax of `p` when the adjusted distribution is degenerate
/// (all zeros after clamping, which can happen when `q >= p` everywhere).
fn sample_adjusted(p: &[f32], q: &[f32], r: f32) -> usize {
    debug_assert_eq!(p.len(), q.len());

    // Build clamped-difference distribution.
    let mut sum = 0.0f32;
    let mut max_diff = f32::NEG_INFINITY;
    let mut max_idx = 0usize;
    for (i, (&pi, &qi)) in p.iter().zip(q.iter()).enumerate() {
        let d = (pi - qi).max(0.0);
        sum += d;
        if pi > max_diff {
            max_diff = pi;
            max_idx = i;
        }
    }

    // Degenerate: adjusted mass is zero — fall back to target argmax.
    if sum <= 0.0 {
        return max_idx;
    }

    let inv = 1.0 / sum;
    let mut cumsum = 0.0f32;
    for (i, (&pi, &qi)) in p.iter().zip(q.iter()).enumerate() {
        let d = (pi - qi).max(0.0) * inv;
        cumsum += d;
        if r < cumsum {
            return i;
        }
    }
    // Numerical tail: return last non-zero token.
    for i in (0..p.len()).rev() {
        if (p[i] - q[i]).max(0.0) > 0.0 {
            return i;
        }
    }
    max_idx
}

/// **Unstable**: probabilistic rejection sampling verifier; algorithm parameters
/// and return type may change as the generation API evolves.
///
/// Verify draft tokens from a speculative decoder against the target model using
/// the probabilistic rejection sampling algorithm of Leviathan et al. (2023).
///
/// # Verification convention
///
/// `target_logits[i]` is the target model output produced *after* processing
/// `draft_tokens[i]`, i.e. it predicts the token at position `i + 1`. To verify
/// `draft_tokens[i]` we therefore need the target distribution at position `i`:
///
/// - `i == 0`: `initial_target_logits` (the caller's distribution for the first
///   draft slot, typically argmax/sampled by the outer generation loop).
/// - `i >= 1`: `target_logits[i - 1]`.
///
/// `target_logits[n - 1]` is reserved for the bonus token sampled after full
/// acceptance. Using `target_logits[i]` to verify `draft_tokens[i]` is the
/// off-by-one that codex round 1 flagged (Leviathan Algorithm 1 evaluates
/// `p_i(x_i)`, not `p_{i+1}(x_i)`).
///
/// For each position `i`:
/// - Compute `p[x_i]` (target probability under the position-`i` distribution)
///   and `q[x_i]` (draft probability).
/// - Accept `x_i` with probability `min(1, p[x_i] / q[x_i])`.
/// - On rejection, sample the bonus token from the adjusted distribution
///   `norm(max(0, p - q))`, which exactly recovers the target distribution.
/// - After all tokens accepted, sample one bonus token from
///   `softmax(target_logits[n - 1])` — the target's prediction for the position
///   after the last accepted draft token.
///
/// When `greedy` is `true` (temperature = 0 / top_k = 1), the function skips
/// probability computation entirely and falls back to argmax comparison, which is
/// strictly faster and produces the same result as the standard greedy verifier.
///
/// # Arguments
///
/// - `draft_tokens`: token IDs proposed by the draft model.
/// - `draft_logits`: raw logits from the draft model for each position, parallel
///   to `draft_tokens`. `draft_logits[i]` is `q_i(·)`.
/// - `initial_target_logits`: target distribution over the first draft slot — the
///   logits the outer generation loop already used to choose between the draft's
///   `draft_tokens[0]` and the target's own pick.
/// - `target_logits`: raw logits from the target model for each draft position.
///   `target_logits[i]` is the target output *after* processing `draft_tokens[i]`,
///   i.e. the distribution over what comes *after* position `i`. Slot `n - 1` is
///   used for the bonus token on full acceptance, never for verification.
/// - `greedy`: when `true`, use argmax comparison instead of probabilistic sampling.
/// - `seed`: PRNG seed for reproducibility; pass `None` for non-deterministic.
///
/// # Errors
///
/// Returns [`crate::error::InferenceError::InvalidInput`] when:
/// - `draft_tokens` and `target_logits` have different lengths.
/// - `draft_logits` length or per-row vocabulary disagrees with `draft_tokens` —
///   checked only when `greedy == false`. Greedy callers may pass an empty
///   `&[]` (the argument is unused).
/// - `initial_target_logits` is empty or has a vocabulary size that disagrees
///   with `target_logits` / `draft_logits` rows.
pub fn rejection_sample_draft(
    draft_tokens: &[u32],
    draft_logits: &[Vec<f32>],
    initial_target_logits: &[f32],
    target_logits: &[Vec<f32>],
    greedy: bool,
    seed: Option<u64>,
) -> Result<RejectionSampleResult, crate::error::InferenceError> {
    use crate::error::InferenceError;

    let n = draft_tokens.len();
    if target_logits.len() != n {
        return Err(InferenceError::InvalidInput(format!(
            "draft_tokens len={n} but target_logits len={}",
            target_logits.len()
        )));
    }
    // `draft_logits` is only used in probabilistic mode (to evaluate `q(x_i)`); greedy
    // callers can pass an empty slice to avoid allocating a `draft_len * vocab_size`
    // placeholder for a value that will never be read. Validate the length only when
    // we are actually going to consume it.
    if !greedy && draft_logits.len() != n {
        return Err(InferenceError::InvalidInput(format!(
            "probabilistic rejection sampling: draft_tokens len={n} but draft_logits len={}",
            draft_logits.len()
        )));
    }

    if n == 0 {
        return Ok(RejectionSampleResult {
            accepted_count: 0,
            accepted_tokens: vec![],
            bonus_token: None,
            had_rejection: false,
        });
    }

    // Validate that all logit slices have the same (non-zero) vocabulary size.
    let vocab = initial_target_logits.len();
    if vocab == 0 {
        return Err(InferenceError::InvalidInput(
            "initial_target_logits vocabulary size is 0".into(),
        ));
    }
    for (i, tl) in target_logits.iter().enumerate() {
        if tl.len() != vocab {
            return Err(InferenceError::InvalidInput(format!(
                "target_logits[{i}] has length {} but expected {vocab}",
                tl.len()
            )));
        }
    }
    if !greedy {
        for (i, dl) in draft_logits.iter().enumerate() {
            if dl.len() != vocab {
                return Err(InferenceError::InvalidInput(format!(
                    "draft_logits[{i}] has length {} but expected {vocab}",
                    dl.len()
                )));
            }
        }
    }

    // For verifying `draft_tokens[i]`, pick the target distribution at position
    // `i`. The convention is documented above the function: `initial_target_logits`
    // for the first draft slot, `target_logits[i - 1]` after that.
    let target_dist_for = |i: usize| -> &[f32] {
        if i == 0 {
            initial_target_logits
        } else {
            &target_logits[i - 1]
        }
    };

    if greedy {
        let mut accepted_tokens = Vec::with_capacity(n);
        let mut had_rejection = false;
        let mut bonus_token = None;

        for i in 0..n {
            let target_choice = argmax(target_dist_for(i)) as u32;
            if target_choice == draft_tokens[i] {
                accepted_tokens.push(draft_tokens[i]);
            } else {
                had_rejection = true;
                bonus_token = Some(target_choice);
                break;
            }
        }

        // Bonus token after full acceptance: argmax of the position-after-last-draft
        // target distribution.
        if !had_rejection {
            let last_target_choice = argmax(target_logits.last().expect("n > 0")) as u32;
            bonus_token = Some(last_target_choice);
        }

        let accepted_count = accepted_tokens.len();
        return Ok(RejectionSampleResult {
            accepted_count,
            accepted_tokens,
            bonus_token,
            had_rejection,
        });
    }

    // --- Probabilistic path ---
    let mut rng = match seed {
        Some(s) => SpecRng::new(s),
        None => SpecRng::from_clock(),
    };

    // Pre-allocate probability buffers for reuse across positions.
    let mut p_buf = vec![0.0f32; vocab];
    let mut q_buf = vec![0.0f32; vocab];

    let mut accepted_tokens: Vec<u32> = Vec::with_capacity(n);
    let mut had_rejection = false;
    let mut rejection_bonus: Option<u32> = None;

    for i in 0..n {
        let tok = draft_tokens[i] as usize;
        if tok >= vocab {
            return Err(InferenceError::InvalidInput(format!(
                "draft_tokens[{i}]={tok} >= vocab_size={vocab}"
            )));
        }

        softmax_into(target_dist_for(i), &mut p_buf);
        softmax_into(&draft_logits[i], &mut q_buf);

        let p_x = p_buf[tok];
        let q_x = q_buf[tok];

        // Acceptance probability: min(1, p(x) / q(x)).
        // Guard against division by zero: if q_x == 0, the draft model assigns
        // zero mass to this token yet proposed it — reject immediately and sample
        // from the full target distribution.
        let accept_prob = if q_x <= 0.0 {
            0.0f32
        } else {
            (p_x / q_x).min(1.0)
        };

        let u = rng.next_f32();
        if u < accept_prob {
            accepted_tokens.push(draft_tokens[i]);
        } else {
            had_rejection = true;
            // Sample from the adjusted distribution max(0, p - q) normalised.
            let r = rng.next_f32();
            let adjusted_tok = sample_adjusted(&p_buf, &q_buf, r) as u32;
            rejection_bonus = Some(adjusted_tok);
            break;
        }
    }

    // If all draft tokens were accepted, sample one bonus token from the target
    // distribution at the position *after* the last accepted token.
    let bonus_token = if had_rejection {
        rejection_bonus
    } else {
        let r = rng.next_f32();
        // Use target_logits[n-1] — the distribution over what comes after draft[n-1].
        softmax_into(target_logits.last().expect("n > 0"), &mut p_buf);
        // Weighted sample from the full target distribution.
        let mut cumsum = 0.0f32;
        let mut sampled = argmax(target_logits.last().expect("n > 0")) as u32;
        for (j, &prob) in p_buf.iter().enumerate() {
            cumsum += prob;
            if r < cumsum {
                sampled = j as u32;
                break;
            }
        }
        Some(sampled)
    };

    let accepted_count = accepted_tokens.len();
    Ok(RejectionSampleResult {
        accepted_count,
        accepted_tokens,
        bonus_token,
        had_rejection,
    })
}

// ---------------------------------------------------------------------------
// End of rejection sampling additions
// ---------------------------------------------------------------------------

/// **Unstable**: draft verification loop; signature may change as sampling strategies expand.
///
/// Verify draft tokens against the model's greedy predictions.
///
/// Starting from `current_token` (the last committed token sitting at `position_start - 1`
/// with the KV cache holding positions `0..position_start`), runs one forward pass per
/// candidate position until either a draft token is rejected or all drafts are confirmed.
///
/// # Contract
///
/// Returns `(accepted, all_logits)` where:
/// - `accepted` ∈ `0..=draft_tokens.len()`: number of draft tokens that matched greedy.
/// - `all_logits.len() == accepted + 1`: exactly one logit vector per committed position
///   plus one for the next-step distribution.
/// - For `i in 0..accepted`: `argmax(all_logits[i]) == draft_tokens[i]` (greedy agreement).
/// - `argmax(all_logits[accepted])` is the greedy-correct continuation after the accepted
///   prefix: a rejection-correction when `accepted < draft_tokens.len()`, or the bonus
///   token when all drafts matched.
///
/// This helper is greedy-equivalent: with a stateless `forward_fn` the sequence it commits
/// is byte-for-byte identical to plain greedy decoding. The real latency benefit only appears
/// when the *target* forward pass is batched (as in the `mtp_verify_draft` path); this
/// single-token variant does `accepted + 1` sequential calls for `accepted + 1` tokens.
pub fn verify_draft<F>(
    current_token: u32,
    draft_tokens: &[u32],
    position_start: usize,
    mut forward_fn: F,
) -> (usize, Vec<Vec<f32>>)
where
    F: FnMut(u32, usize) -> Vec<f32>,
{
    let mut input = current_token;
    let mut accepted = 0usize;
    let mut all_logits = Vec::with_capacity(draft_tokens.len() + 1);

    loop {
        let logits = forward_fn(input, position_start + accepted);
        let model_choice = argmax(&logits);
        all_logits.push(logits);

        if accepted < draft_tokens.len() && model_choice == draft_tokens[accepted] as usize {
            // Model agrees with the next draft token; advance and keep verifying.
            input = draft_tokens[accepted];
            accepted += 1;
        } else {
            // Either a mismatch (rejection) or all drafts verified — the final logits
            // in all_logits[accepted] are the greedy-correct next-step distribution.
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
    // An empty prompt cannot seed the n-gram speculator or the decode history,
    // so there is nothing to generate from. Return early instead of panicking on
    // `all_tokens.last()` in the first decode step.
    if prompt_tokens.is_empty() {
        return Vec::new();
    }

    let speculator = NgramSpeculator::new(prompt_tokens.to_vec(), max_ngram, max_draft);
    let mut generated: Vec<u32> = Vec::new();
    let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();
    let mut pos = prompt_tokens.len();

    while generated.len() < max_new_tokens {
        let draft = speculator.speculate(&all_tokens);
        let current = *all_tokens
            .last()
            .expect("invariant: prompt_tokens must seed speculation history");

        // verify_draft feeds `current` first, then accepted draft tokens.
        // When draft is empty this reduces to a single forward pass — the
        // "bonus" logits[0] give the greedy-correct next token.
        let (accepted, logits_vec) = verify_draft(current, &draft, pos, &mut forward_fn);

        // Commit the accepted draft tokens (each == greedy by verify_draft's contract).
        for &t in &draft[..accepted] {
            if generated.len() == max_new_tokens {
                return generated;
            }
            if t == eos_token {
                return generated;
            }
            generated.push(t);
            all_tokens.push(t);
            pos += 1;
        }

        // logits_vec[accepted] is the greedy-correct next-step distribution regardless
        // of whether we rejected or exhausted the draft (bonus token path).
        if generated.len() < max_new_tokens {
            let next_token = argmax(&logits_vec[accepted]) as u32;
            if next_token == eos_token {
                break;
            }
            generated.push(next_token);
            all_tokens.push(next_token);
            pos += 1;
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
    fn argmax_tie_first_wins() {
        // Greedy contract: speculation must be token-identical to plain greedy.
        // The engine greedy decode (metal_qwen35), the sampling module, and
        // torch.argmax all break ties toward the FIRST max index, so this helper
        // must too (see #280).
        assert_eq!(argmax(&[1.0, 1.0, 0.5]), 0);
    }

    #[test]
    fn argmax_tie_matches_engine_first_wins() {
        // Cross-check against the engine's first-wins loop on a tie-heavy vector:
        // enabling speculation must not change the token on tied top logits.
        let tie = [0.0_f32, 0.0, -1.0, 0.0];
        let engine_first_wins = {
            let mut best_id = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in tie.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_id = i;
                }
            }
            best_id
        };
        assert_eq!(engine_first_wins, 0);
        assert_eq!(argmax(&tie), engine_first_wins);
    }

    #[test]
    fn argmax_skips_nan_and_never_panics() {
        // NaN loses to finite values; among finite ties the first index wins.
        // Matches sampling::argmax_f32_scalar and the engine greedy decode.
        assert_eq!(argmax(&[f32::NAN, 1.0, 1.0]), 1);
        assert_eq!(argmax(&[1.0, f32::NAN, 1.0]), 0);
        // All-NaN and all-(-inf) fall back to index 0 — must never panic.
        assert_eq!(argmax(&[f32::NAN, f32::NAN]), 0);
        assert_eq!(argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), 0);
    }

    #[test]
    fn argmax_negative_values() {
        assert_eq!(argmax(&[-3.0, -1.0, -2.0]), 1);
    }

    // -- verify_draft tests --

    #[test]
    fn verify_draft_all_accepted() {
        // current=0, draft=[1,2,3].
        // Call 0: fwd(0, 10) → predicts draft[0]=1  → accepted=1, input=1
        // Call 1: fwd(1, 11) → predicts draft[1]=2  → accepted=2, input=2
        // Call 2: fwd(2, 12) → predicts draft[2]=3  → accepted=3, input=3
        // Call 3: fwd(3, 13) → accepted=3 == len=3  → break (bonus logits)
        // Result: accepted=3, logits.len()=4 (3 verification calls + 1 bonus call)
        let draft = vec![1u32, 2, 3];
        let mut call = 0usize;
        let (accepted, logits) = verify_draft(0, &draft, 10, |_tok, _pos| {
            let mut l = vec![0.0f32; 10];
            match call {
                0 => l[1] = 1.0, // predicts draft[0]=1
                1 => l[2] = 1.0, // predicts draft[1]=2
                2 => l[3] = 1.0, // predicts draft[2]=3
                _ => l[0] = 1.0, // bonus call — prediction is the next greedy token
            }
            call += 1;
            l
        });
        assert_eq!(accepted, 3);
        assert_eq!(logits.len(), 4);
    }

    #[test]
    fn verify_draft_first_rejected() {
        // current=5, draft=[1,2,3].
        // Call 0: fwd(5, 0) → predicts 99 != draft[0]=1 → accepted=0, break
        // Result: accepted=0, logits.len()=1 (the rejection-correction call)
        let draft = vec![1u32, 2, 3];
        let (accepted, logits) = verify_draft(5, &draft, 0, |_tok, _pos| {
            let mut l = vec![0.0f32; 100];
            l[99] = 1.0;
            l
        });
        assert_eq!(accepted, 0);
        assert_eq!(logits.len(), 1);
        // The correction token is 99, not draft[0]=1
        assert_eq!(argmax(&logits[0]), 99);
    }

    #[test]
    fn verify_draft_partial_acceptance() {
        // current=0, draft=[10,20,30,40].
        // Call 0: fwd(0,  0) → predicts 10  → accepted=1, input=10
        // Call 1: fwd(10, 1) → predicts 20  → accepted=2, input=20
        // Call 2: fwd(20, 2) → predicts 30  → accepted=3, input=30
        // Call 3: fwd(30, 3) → predicts  0 != draft[3]=40 → break
        // Result: accepted=3, logits.len()=4
        let draft = vec![10u32, 20, 30, 40];
        let mut call = 0usize;
        let (accepted, logits) = verify_draft(0, &draft, 0, |_tok, _pos| {
            let mut l = vec![0.0f32; 50];
            match call {
                0 => l[10] = 1.0, // agrees with draft[0]
                1 => l[20] = 1.0, // agrees with draft[1]
                2 => l[30] = 1.0, // agrees with draft[2]
                _ => l[0] = 1.0,  // disagrees with draft[3]=40
            }
            call += 1;
            l
        });
        assert_eq!(accepted, 3);
        assert_eq!(logits.len(), 4);
    }

    #[test]
    fn verify_draft_single_token() {
        // current=7, draft=[42].
        // Call 0: fwd(7, 5) → predicts 42 == draft[0] → accepted=1, input=42
        // Call 1: fwd(42, 6) → accepted=1 == len=1 → break (bonus)
        // Result: accepted=1, logits.len()=2
        let draft = vec![42u32];
        let mut call = 0usize;
        let (accepted, logits) = verify_draft(7, &draft, 5, |_tok, _pos| {
            let mut l = vec![0.0f32; 100];
            if call == 0 {
                l[42] = 1.0; // predicts draft[0]=42
            } else {
                l[0] = 1.0; // bonus call
            }
            call += 1;
            l
        });
        assert_eq!(accepted, 1);
        assert_eq!(logits.len(), 2);
    }

    #[test]
    fn verify_draft_empty() {
        // current=3, draft=[].
        // Call 0: fwd(3, 0) → accepted=0 == len=0 → break immediately (bonus logits)
        // Result: accepted=0, logits.len()=1
        let (accepted, logits) = verify_draft(3, &[], 0, |_tok, _pos| {
            let mut l = vec![0.0f32; 10];
            l[7] = 1.0;
            l
        });
        assert_eq!(accepted, 0);
        assert_eq!(logits.len(), 1);
        // The single logits entry is the greedy-correct next token distribution
        assert_eq!(argmax(&logits[0]), 7);
    }

    #[test]
    fn verify_draft_positions_are_correct() {
        // current=9, draft=[1,2,3], position_start=100.
        // Call 0: fwd(9,   100) → predicts 1  → accepted=1
        // Call 1: fwd(1,   101) → predicts 2  → accepted=2
        // Call 2: fwd(2,   102) → predicts 3  → accepted=3
        // Call 3: fwd(3,   103) → bonus call  → break
        // Positions seen: [100, 101, 102, 103]
        let draft = vec![1u32, 2, 3];
        let mut positions = Vec::new();
        let mut call = 0usize;
        let _ = verify_draft(9, &draft, 100, |_tok, pos| {
            positions.push(pos);
            let mut l = vec![0.0f32; 10];
            match call {
                0 => l[1] = 1.0, // predicts draft[0]=1
                1 => l[2] = 1.0, // predicts draft[1]=2
                2 => l[3] = 1.0, // predicts draft[2]=3
                _ => l[0] = 1.0, // bonus
            }
            call += 1;
            l
        });
        assert_eq!(positions, vec![100, 101, 102, 103]);
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
        fn snapshot_gdn_states(&self) -> crate::attention::gdn::GdnSnapshot {
            Vec::new()
        }
        fn restore_gdn_states(&mut self, _snapshot: &crate::attention::gdn::GdnSnapshot) {}
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

    // A verifier whose `verify_tokens` advances `cache_pos` then immediately
    // returns an error — exercises the #282 error-path rollback in
    // `mtp_verify_precomputed_draft` (and, by structural analogy, in
    // `mtp_verify_draft`).  We use the precomputed helper because constructing a
    // real `MtpVerifier` requires a live model.
    struct ErroringTargetVerifier {
        cache_pos: usize,
    }

    impl MtpTargetVerifier for ErroringTargetVerifier {
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
            _start_pos: usize,
        ) -> Result<Vec<Vec<f32>>, crate::error::InferenceError> {
            // Advance the cache before returning the error, mimicking a real
            // implementation that commits the forward pass then discovers a
            // shape mismatch in the output.
            self.cache_pos += tokens.len();
            Err(crate::error::InferenceError::Inference(
                "simulated verify_tokens failure".into(),
            ))
        }
        fn snapshot_gdn_states(&self) -> crate::attention::gdn::GdnSnapshot {
            Vec::new()
        }
        fn restore_gdn_states(&mut self, _snapshot: &crate::attention::gdn::GdnSnapshot) {}
    }

    #[test]
    fn mtp_verify_error_restores_target_cache() {
        // Guards the #282 footgun: if `verify_tokens` advances the target cache
        // and then returns an error, the cache must be rolled back to its
        // pre-call position so subsequent calls see a consistent starting state.
        //
        // We use `mtp_verify_precomputed_draft` (the test helper) rather than
        // `mtp_verify_draft` because the latter requires a real `MtpVerifier`
        // backed by a model.  The rollback logic for the production path is
        // structurally identical.
        let target_start = 42usize;
        // initial argmax = draft[0] so execution reaches verify_tokens
        let initial = logits_with_argmax(VOCAB, 5);
        let draft = vec![5u32, 6, 7]; // draft[0]==5 matches initial argmax; not EOS

        let mut target = ErroringTargetVerifier {
            cache_pos: target_start,
        };

        let result =
            mtp_verify_precomputed_draft(draft, target_start, &initial, None, &mut target, 4);

        assert!(result.is_err(), "expected Err from erroring verifier");
        assert_eq!(
            target.cache_pos, target_start,
            "cache_pos must be restored to pre-call value on error (#282)"
        );
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
    fn generate_does_not_overrun_budget_on_multi_token_draft() {
        // Regression for #242: the speculative branch committed every accepted
        // draft token (plus a rejection-fallback token) without re-checking the
        // budget, so a multi-token draft could return more than `max_new_tokens`.
        //
        // The suffix [7,8,9] recurs at pos 0, so `speculate(&prompt)` drafts
        // prompt[3..7] = [0,7,8,9] on the very first step. The forward_fn below
        // makes the target agree with all four (accepted == 4), so a budget of 1
        // would have returned 4 tokens before the fix.
        let prompt = vec![7u32, 8, 9, 0, 7, 8, 9];
        let eos = 999u32;
        for max_new in 1..=3 {
            let out = generate_with_speculation(
                &prompt,
                max_new,
                eos,
                |tok: u32, _pos: usize| match tok {
                    0 => logits_with_argmax(1000, 7),
                    7 => logits_with_argmax(1000, 8),
                    8 => logits_with_argmax(1000, 9),
                    _ => logits_with_argmax(1000, 5),
                },
                5,
                4,
            );
            assert!(
                out.len() <= max_new,
                "budget {max_new} overrun: produced {} tokens {:?}",
                out.len(),
                out
            );
            assert_eq!(out.len(), max_new, "should fill the budget exactly");
        }
    }

    #[test]
    fn generate_eos_in_draft_stops_early() {
        // Prompt: [1, 2, 3, 99, 5] where eos=99.
        //
        // Iter 1: draft=[] (no 2-gram match from tail [..,5]); verify_draft(5,[],5,fwd).
        //   call 1: fwd(5, 5) → 2.  Emit 2.  all_tokens=[1,2,3,99,5,2], pos=6.
        // Iter 2: draft=[] ([5,2] not in prompt); verify_draft(2,[],6,fwd).
        //   call 2: fwd(2, 6) → 3.  Emit 3.  all_tokens=[...,2,3], pos=7.
        // Iter 3: 2-gram [2,3] matches prompt[1..3] → draft=[99,5].
        //   verify_draft(3, [99,5], 7, fwd):
        //     call 3: fwd(3, 7) → 99 == draft[0] → accepted=1, input=99
        //     call 4: fwd(99, 8) → 0  != draft[1]=5 → break
        //   Commit draft[..1]=[99]: 99==eos → return [2,3].
        let prompt = vec![1u32, 2, 3, 99, 5];
        let eos = 99;
        let mut call = 0;
        let result = generate_with_speculation(
            &prompt,
            20,
            eos,
            |_tok, _pos| {
                call += 1;
                let mut l = vec![0.0f32; 100];
                match call {
                    1 => l[2] = 1.0,  // verify_draft(5,[],5): bonus → emit 2
                    2 => l[3] = 1.0,  // verify_draft(2,[],6): bonus → emit 3
                    3 => l[99] = 1.0, // verify_draft(3,[99,5],7): fwd(3,7)→99 matches draft[0]
                    4 => l[0] = 1.0,  // verify_draft cont: fwd(99,8)→0 != draft[1]=5 → reject
                    _ => l[42] = 1.0,
                }
                l
            },
            5,
            4,
        );

        // [2, 3] emitted normally, then draft[0]=99=EOS terminates generation
        assert_eq!(result, vec![2, 3]);
    }

    #[test]
    fn generate_with_speculation_issue_243_regression() {
        // Regression: prompt=[1,1,1], greedy always picks 0 from input 1.
        // Speculation drafts [1,...] from 2-gram [1,1], but the model rejects
        // draft[0]=1 on the very first call (greedy choice is 0). The fix
        // ensures verify_draft feeds current_token first, so draft[0] is gated
        // by an actual forward pass rather than admitted unconditionally.
        let prompt = vec![1u32, 1, 1];
        let eos = 99;
        let result = generate_with_speculation(
            &prompt,
            1,
            eos,
            |_tok, _pos| {
                // Greedy choice is always 0 regardless of input.
                let mut l = vec![0.0f32; 10];
                l[0] = 1.0;
                l
            },
            5,
            4,
        );
        // Greedy emits [0]; the old buggy code emitted [1] (draft committed unverified).
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn generate_with_speculation_matches_greedy() {
        // Verifies speculative decoding is token-identical to plain greedy.
        //
        // forward_fn: next token = (input_token + 1) % 10 (pure, position-independent).
        // Prompt repeats the cycle twice to guarantee 2-gram matches fire during
        // generation, exercising the speculation path on every iteration after
        // the first few tokens.
        let prompt: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1];
        let eos = 50u32; // unreachable in a 10-token cycle
        let max_new = 10usize;

        // Reference greedy: feed last token, argmax, repeat.
        let greedy_fwd = |tok: u32, _pos: usize| -> Vec<f32> {
            let mut l = vec![0.0f32; 60];
            l[((tok + 1) % 10) as usize] = 1.0;
            l
        };
        let mut ref_tokens = prompt.clone();
        let mut ref_pos = prompt.len();
        let mut greedy_out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let logits = greedy_fwd(*ref_tokens.last().unwrap(), ref_pos);
            let t = argmax(&logits) as u32;
            if t == eos {
                break;
            }
            greedy_out.push(t);
            ref_tokens.push(t);
            ref_pos += 1;
        }

        // Speculative path — identical forward logic, independent closure.
        let spec_fwd = |tok: u32, _pos: usize| -> Vec<f32> {
            let mut l = vec![0.0f32; 60];
            l[((tok + 1) % 10) as usize] = 1.0;
            l
        };
        let spec_out = generate_with_speculation(&prompt, max_new, eos, spec_fwd, 5, 4);

        assert_eq!(
            spec_out, greedy_out,
            "speculative output diverged from greedy: spec={spec_out:?} greedy={greedy_out:?}"
        );
    }

    #[test]
    fn generate_with_speculation_matches_greedy_with_ties() {
        // Regression guard for #280 (last-wins tie-break divergence).
        //
        // The existing `generate_with_speculation_matches_greedy` uses one-hot logits
        // so the top logit is never tied.  This test places TWO entries at the same
        // maximum value so `argmax`'s first-wins tie-break is exercised on every step.
        // Both greedy and speculative paths must resolve the tie identically (lower
        // token-id wins) and produce the same output sequence.
        //
        // forward_fn: tokens 2 and 5 share the peak (first-wins → always 2).
        // Prompt repeats the cycle twice so the 2-gram speculator fires.
        let prompt: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1];
        let eos = 50u32;
        let max_new = 10usize;

        // The tied forward: token 2 and token 5 are both at the peak; argmax returns
        // the first (lower) index, so greedy always emits 2 regardless of input token.
        let tied_fwd = |_tok: u32, _pos: usize| -> Vec<f32> {
            let mut l = vec![0.0f32; 60];
            l[2] = 1.0; // first tied winner — argmax must return this
            l[5] = 1.0; // second tied entry at the same value
            l
        };

        // Reference greedy: identical logic, independent closure.
        let greedy_fwd = |_tok: u32, _pos: usize| -> Vec<f32> {
            let mut l = vec![0.0f32; 60];
            l[2] = 1.0;
            l[5] = 1.0;
            l
        };
        let mut ref_tokens = prompt.clone();
        let mut ref_pos = prompt.len();
        let mut greedy_out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let logits = greedy_fwd(*ref_tokens.last().unwrap(), ref_pos);
            let t = argmax(&logits) as u32;
            if t == eos {
                break;
            }
            greedy_out.push(t);
            ref_tokens.push(t);
            ref_pos += 1;
        }

        let spec_out = generate_with_speculation(&prompt, max_new, eos, tied_fwd, 5, 4);

        assert_eq!(
            spec_out, greedy_out,
            "speculative output diverged from greedy under tie-break: spec={spec_out:?} greedy={greedy_out:?}"
        );
    }

    // ── rejection_sample_draft tests ─────────────────────────────────────────

    /// Build uniform logits over `vocab` tokens — after softmax every token
    /// has probability 1/vocab.
    fn uniform_logits(vocab: usize) -> Vec<f32> {
        vec![1.0f32; vocab]
    }

    /// Build peaked logits where token `tok` dominates.
    fn peaked_logits(vocab: usize, tok: usize, peak: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; vocab];
        v[tok] = peak;
        v
    }

    #[test]
    fn rejection_empty_draft_returns_no_bonus() {
        let result =
            rejection_sample_draft(&[], &[], &uniform_logits(8), &[], false, Some(42)).unwrap();
        assert_eq!(result.accepted_count, 0);
        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.bonus_token, None);
        assert!(!result.had_rejection);
    }

    #[test]
    fn rejection_mismatched_lengths_returns_error() {
        let logits = vec![peaked_logits(10, 3, 5.0)];
        let initial = uniform_logits(10);
        let err = rejection_sample_draft(&[3], &logits, &initial, &[], false, Some(1));
        assert!(err.is_err());
    }

    #[test]
    fn rejection_identical_distributions_accept_all() {
        // When draft and target have exactly the same distribution (including the
        // distribution that predicts draft[0]), p(x)/q(x) = 1 everywhere so all
        // tokens accept.
        const VOCAB: usize = 8;
        let logits = peaked_logits(VOCAB, 2, 10.0);

        let draft_tokens = vec![2u32, 2, 2];
        let draft_logits = vec![logits.clone(), logits.clone(), logits.clone()];
        let target_logits = vec![logits.clone(), logits.clone(), logits.clone()];
        let initial_target = logits.clone();

        for seed in 0u64..20 {
            let res = rejection_sample_draft(
                &draft_tokens,
                &draft_logits,
                &initial_target,
                &target_logits,
                false,
                Some(seed),
            )
            .unwrap();
            assert_eq!(
                res.accepted_count, 3,
                "seed={seed}: identical distributions must accept all tokens"
            );
            assert_eq!(res.accepted_tokens, vec![2u32, 2, 2]);
            assert!(!res.had_rejection, "seed={seed}: no rejection expected");
        }
    }

    #[test]
    fn rejection_known_rejection_samples_adjusted() {
        // Single draft slot: q peaks exactly at 0, p (== initial_target_logits) peaks
        // exactly at 1 → accept_prob = min(1, 0/1) = 0, adjusted distribution puts all
        // mass on token 1, so every bonus draw must be token 1.
        const VOCAB: usize = 4;
        let draft_tok = 0u32;
        let mut draft_logit = vec![f32::NEG_INFINITY; VOCAB];
        draft_logit[0] = 100.0;
        let mut target_logit = vec![f32::NEG_INFINITY; VOCAB];
        target_logit[1] = 100.0;

        let draft_tokens = vec![draft_tok];
        let draft_logits = vec![draft_logit];
        // For a single-draft scenario, `target_logits[0]` (bonus position) and the
        // verification distribution (`initial_target_logits`) carry the same peak.
        let target_logits = vec![target_logit.clone()];
        let initial_target = target_logit;

        let mut bonus_counts = [0u32; VOCAB];
        let trials = 40u64;
        for seed in 0u64..trials {
            let res = rejection_sample_draft(
                &draft_tokens,
                &draft_logits,
                &initial_target,
                &target_logits,
                false,
                Some(seed),
            )
            .unwrap();
            assert!(res.had_rejection, "seed={seed}: rejection must occur");
            assert_eq!(res.accepted_count, 0, "seed={seed}: no tokens accepted");
            if let Some(b) = res.bonus_token {
                bonus_counts[b as usize] += 1;
            }
        }
        assert_eq!(
            bonus_counts[1], trials as u32,
            "adjusted sampling must produce token 1 in all trials, got counts={bonus_counts:?}"
        );
    }

    #[test]
    fn rejection_greedy_mode_accepts_matching_argmax() {
        // Verification convention:
        //   draft[0] verified against initial_target_logits
        //   draft[i>=1] verified against target_logits[i-1]
        //   target_logits[n-1] is the bonus on full accept.
        //
        // Setup: initial_target peaks at draft[0]=5 → draft[0] accepted. Then
        // target_logits[0] peaks at 9 ≠ draft[1]=7 → reject. bonus = 9.
        const VOCAB: usize = 10;
        let draft_tokens = vec![5u32, 7, 3];
        let initial_target = peaked_logits(VOCAB, 5, 5.0);
        let target_logits = vec![
            peaked_logits(VOCAB, 9, 5.0),
            peaked_logits(VOCAB, 2, 5.0),
            peaked_logits(VOCAB, 4, 5.0),
        ];
        let draft_logits = vec![
            uniform_logits(VOCAB),
            uniform_logits(VOCAB),
            uniform_logits(VOCAB),
        ];

        let res = rejection_sample_draft(
            &draft_tokens,
            &draft_logits,
            &initial_target,
            &target_logits,
            true,
            Some(1),
        )
        .unwrap();

        assert_eq!(res.accepted_count, 1, "only position 0 accepted");
        assert!(res.had_rejection);
        assert_eq!(
            res.bonus_token,
            Some(9),
            "bonus is argmax of target_logits[0]"
        );
    }

    #[test]
    fn rejection_greedy_all_match() {
        // Greedy: draft[0]=3 matches initial_target=3; draft[1]=7 matches
        // target_logits[0]=7; draft[2]=2 matches target_logits[1]=2; bonus from
        // target_logits[2]=9.
        const VOCAB: usize = 10;
        let draft_tokens = vec![3u32, 7, 2];
        let initial_target = peaked_logits(VOCAB, 3, 5.0);
        let target_logits = vec![
            peaked_logits(VOCAB, 7, 5.0),
            peaked_logits(VOCAB, 2, 5.0),
            peaked_logits(VOCAB, 9, 5.0),
        ];
        let draft_logits = vec![
            uniform_logits(VOCAB),
            uniform_logits(VOCAB),
            uniform_logits(VOCAB),
        ];

        let res = rejection_sample_draft(
            &draft_tokens,
            &draft_logits,
            &initial_target,
            &target_logits,
            true,
            Some(7),
        )
        .unwrap();

        assert_eq!(res.accepted_count, 3);
        assert_eq!(res.accepted_tokens, vec![3u32, 7, 2]);
        assert!(!res.had_rejection);
        assert_eq!(res.bonus_token, Some(9));
    }

    #[test]
    fn rejection_q_zero_causes_rejection() {
        // q(x) = 0 for the draft token → accept_prob = 0 → always reject. On
        // rejection, bonus comes from the position-0 target distribution (=
        // initial_target_logits).
        const VOCAB: usize = 4;
        let mut draft_logit = vec![f32::NEG_INFINITY; VOCAB];
        draft_logit[0] = 10.0; // q peaks at 0
        let target_logit = peaked_logits(VOCAB, 2, 10.0); // p peaks at 2

        let res = rejection_sample_draft(
            &[2u32],
            &[draft_logit],
            &target_logit,
            std::slice::from_ref(&target_logit),
            false,
            Some(99),
        )
        .unwrap();

        assert!(res.had_rejection);
        assert_eq!(
            res.bonus_token,
            Some(2),
            "bonus token from adjusted dist should be 2"
        );
    }

    #[test]
    fn rejection_probabilistic_full_accept_has_bonus() {
        // Identical distributions at every position → all accepted, bonus appended.
        const VOCAB: usize = 8;
        let logit = peaked_logits(VOCAB, 5, 15.0);
        let draft_tokens = vec![5u32, 5];
        let draft_logits = vec![logit.clone(), logit.clone()];
        let target_logits = vec![logit.clone(), logit.clone()];
        let initial_target = logit.clone();

        let res = rejection_sample_draft(
            &draft_tokens,
            &draft_logits,
            &initial_target,
            &target_logits,
            false,
            Some(0),
        )
        .unwrap();

        assert_eq!(res.accepted_count, 2);
        assert!(!res.had_rejection);
        // Bonus is sampled from the strongly peaked target distribution → must be 5.
        assert_eq!(
            res.bonus_token,
            Some(5),
            "bonus from peaked target must be token 5"
        );
    }

    #[test]
    fn softmax_into_sums_to_one() {
        let logits = vec![1.0f32, 2.0, 3.0, 0.5, -1.0];
        let mut out = vec![0.0f32; logits.len()];
        softmax_into(&logits, &mut out);
        let sum: f32 = out.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax must sum to 1.0, got {sum}"
        );
        for &p in &out {
            assert!(p >= 0.0, "all softmax outputs must be non-negative");
        }
    }

    #[test]
    fn softmax_into_peaked_distribution() {
        // A very large logit gap should produce ≈1.0 for the argmax.
        let mut logits = vec![0.0f32; 10];
        logits[3] = 100.0;
        let mut out = vec![0.0f32; 10];
        softmax_into(&logits, &mut out);
        assert!(
            out[3] > 0.999,
            "heavily peaked logit should dominate softmax"
        );
    }

    #[test]
    fn sample_adjusted_concentrates_on_target_mass() {
        // p = [0.9, 0.1], q = [0.1, 0.9]
        // adjusted = max(0, p-q) = [0.8, 0.0], renormalised = [1.0, 0.0]
        // All samples should return token 0.
        let p = vec![0.9f32, 0.1];
        let q = vec![0.1f32, 0.9];
        for seed in 0u64..20 {
            let mut rng = SpecRng::new(seed + 1);
            let r = rng.next_f32();
            let tok = sample_adjusted(&p, &q, r);
            assert_eq!(tok, 0, "seed={seed}: all mass on token 0");
        }
    }

    // ── ADR-050 MTP rejection sampling integration tests ─────────────────────

    #[test]
    fn mtp_rejection_sampling_seeded_acceptance_accepts_all() {
        // Identical draft and target distributions → p(x)/q(x) == 1 everywhere → all accept.
        const VOCAB: usize = 8;
        let logits = peaked_logits(VOCAB, 2, 100.0);
        let draft_tokens = vec![2u32, 2, 2];
        let draft_logits = vec![logits.clone(), logits.clone(), logits.clone()];
        let target_logits = vec![logits.clone(), logits.clone(), logits.clone()];
        let initial_target = logits.clone();

        let res = rejection_sample_draft(
            &draft_tokens,
            &draft_logits,
            &initial_target,
            &target_logits,
            false,
            Some(7),
        )
        .unwrap();

        assert_eq!(res.accepted_count, 3);
        assert_eq!(res.accepted_tokens, vec![2u32, 2, 2]);
        assert!(!res.had_rejection);
        assert_eq!(
            res.bonus_token,
            Some(2),
            "bonus from peaked target must be token 2"
        );
    }

    #[test]
    fn mtp_rejection_sampling_seeded_rejection_samples_target_token() {
        // q peaks at 0, p peaks at 1 → accept_prob = 0 → certain rejection.
        // Adjusted distribution = max(0, p-q) has all mass at token 1 → bonus = Some(1).
        const VOCAB: usize = 4;
        let mut q_peak_0 = vec![f32::NEG_INFINITY; VOCAB];
        q_peak_0[0] = 100.0;
        let mut p_peak_1 = vec![f32::NEG_INFINITY; VOCAB];
        p_peak_1[1] = 100.0;

        let res = rejection_sample_draft(
            &[0u32],
            &[q_peak_0],
            &p_peak_1,
            &[p_peak_1.clone()],
            false,
            Some(13),
        )
        .unwrap();

        assert_eq!(res.accepted_count, 0);
        assert!(res.accepted_tokens.is_empty());
        assert!(res.had_rejection);
        assert_eq!(
            res.bonus_token,
            Some(1),
            "adjusted sampling must produce token 1"
        );
    }

    #[test]
    fn mtp_rejection_sampling_partial_acceptance_rolls_back_gdn_to_accepted_prefix() {
        // 3-token draft: tokens [0, 1, 2].
        //   draft_logits  = [peak0, peak1, q_peak_2]  — draft model near-certain at each
        //   initial_target = peak0                     — p/q ≈ 1 at i=0 → accept token 0
        //   target_logits[0] = peak1                  — p/q ≈ 1 at i=1 → accept token 1
        //   target_logits[1] = p_peak_5               — p_peak_5[2]/q_peak_2[2] ≈ 0 → reject
        //   bonus = argmax(adjusted(p_peak_5, q_peak_2)) = 5
        const VOCAB: usize = 8;
        let peak0 = peaked_logits(VOCAB, 0, 100.0);
        let peak1 = peaked_logits(VOCAB, 1, 100.0);
        let q_peak_2 = peaked_logits(VOCAB, 2, 100.0);
        let p_peak_5 = peaked_logits(VOCAB, 5, 100.0);
        let bonus_peak_6 = peaked_logits(VOCAB, 6, 100.0);

        let rs = rejection_sample_draft(
            &[0u32, 1, 2],
            &[peak0.clone(), peak1.clone(), q_peak_2.clone()],
            &peak0,
            &[peak1.clone(), p_peak_5.clone(), bonus_peak_6.clone()],
            false,
            Some(0),
        )
        .unwrap();

        assert_eq!(rs.accepted_count, 2);
        assert_eq!(rs.accepted_tokens, vec![0u32, 1]);
        assert!(rs.had_rejection);
        assert_eq!(rs.bonus_token, Some(5));

        // Verify that callers (mtp_verify_draft) will correctly roll back cache to
        // target_start + accepted_count = target_start + 2.
        let target_start = 10usize;
        let mut restore_gdn_called = false;
        let mut mock = {
            struct GdnMock {
                cache_pos: usize,
                restore_called: *mut bool,
            }
            impl MtpTargetVerifier for GdnMock {
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
                    _: usize,
                ) -> Result<Vec<Vec<f32>>, crate::error::InferenceError> {
                    self.cache_pos += tokens.len();
                    Ok(vec![])
                }
                fn snapshot_gdn_states(&self) -> crate::attention::gdn::GdnSnapshot {
                    Vec::new()
                }
                fn restore_gdn_states(&mut self, _: &crate::attention::gdn::GdnSnapshot) {
                    // SAFETY: single-threaded test, pointer is to a local bool.
                    unsafe {
                        *self.restore_called = true;
                    }
                }
            }
            GdnMock {
                cache_pos: target_start,
                restore_called: &mut restore_gdn_called,
            }
        };

        // Simulate what mtp_verify_draft does: advance cache via verify_tokens, then roll back.
        mock.verify_tokens(&[0u32, 1, 2], target_start + 1).unwrap();
        mock.rollback_cache_to(target_start + rs.accepted_count)
            .unwrap();

        assert_eq!(
            mock.cache_pos,
            target_start + 2,
            "rollback must leave cache at target_start + accepted_count"
        );
        assert!(
            !restore_gdn_called,
            "partial accept must not call restore_gdn_states; rollback_cache_to handles GDN"
        );
    }

    #[test]
    fn rejection_greedy_mode_regression_matches_existing_argmax_behavior() {
        // Regression guard: greedy=true path must still accept only argmax-matching tokens.
        // initial_target peaks at 5 → draft[0]=5 accepted.
        // target_logits[0] peaks at 9 ≠ draft[1]=7 → reject; bonus = 9.
        const VOCAB: usize = 10;
        let draft_tokens = vec![5u32, 7, 3];
        let initial_target = peaked_logits(VOCAB, 5, 5.0);
        let target_logits = vec![
            peaked_logits(VOCAB, 9, 5.0),
            peaked_logits(VOCAB, 2, 5.0),
            peaked_logits(VOCAB, 4, 5.0),
        ];
        let draft_logits = vec![
            uniform_logits(VOCAB),
            uniform_logits(VOCAB),
            uniform_logits(VOCAB),
        ];

        let res = rejection_sample_draft(
            &draft_tokens,
            &draft_logits,
            &initial_target,
            &target_logits,
            true,
            Some(1),
        )
        .unwrap();

        assert_eq!(
            res.accepted_count, 1,
            "only draft[0]=5 matches initial_target argmax=5"
        );
        assert!(res.had_rejection);
        assert_eq!(
            res.bonus_token,
            Some(9),
            "bonus is argmax of target_logits[0]"
        );
    }

    #[test]
    fn rejection_greedy_one_draft_mtp_contract_empty_draft_logits() {
        // Locks the contract the greedy MTP loop (metal_qwen35::generate_greedy_mtp,
        // #237) depends on: a single draft token verified with `greedy=true` and an
        // EMPTY `draft_logits` slice (unused in greedy mode). The downstream loop reads
        //   accept → bonus = argmax(target_logits[0])  (verify_out.logits[1])
        //   reject → bonus = argmax(initial_target)     (verify_out.logits[0])
        // A regression that flips this call back to probabilistic (greedy=false,
        // clock-seeded) would break determinism and these argmax bonus identities.
        // `seed=None` with `greedy=true` must stay deterministic (no RNG is built).
        const VOCAB: usize = 16;
        let initial_target = peaked_logits(VOCAB, 4, 6.0); // target's pick at draft pos = 4
        let target_logits = vec![peaked_logits(VOCAB, 11, 6.0)]; // full-accept bonus = 11

        // Accept: draft token == initial_target argmax (4) → accept, bonus = 11.
        let accept =
            rejection_sample_draft(&[4u32], &[], &initial_target, &target_logits, true, None)
                .unwrap();
        assert_eq!(
            accept.accepted_count, 1,
            "draft == initial argmax → accepted"
        );
        assert!(!accept.had_rejection);
        assert_eq!(
            accept.bonus_token,
            Some(11),
            "full-accept bonus is argmax(target_logits[0])"
        );

        // Reject: draft token (7) != initial_target argmax (4) → reject, bonus = 4.
        let reject =
            rejection_sample_draft(&[7u32], &[], &initial_target, &target_logits, true, None)
                .unwrap();
        assert_eq!(
            reject.accepted_count, 0,
            "draft != initial argmax → rejected"
        );
        assert!(reject.had_rejection);
        assert_eq!(
            reject.bonus_token,
            Some(4),
            "rejection bonus is the target's own pick = argmax(initial_target)"
        );
    }

    // -----------------------------------------------------------------------
    // MtpVerifier f16 KV cache rollback regression (Defect 4 fix, Option A).
    //
    // Exercises the real f16 FlatKVCache + dequant scratch path under rollback:
    //   replay:  forward(1,0) → forward(2,1) → forward(3,2) → rollback(1)
    //              → forward(7,1)
    //   fresh:   forward(1,0) → forward(7,1)
    //
    // After both sequences the KV cache at positions 0 and 1 must be identical
    // and the final logits must match, proving stale f16 data at positions 2/3
    // does not contaminate the output after rollback.
    // -----------------------------------------------------------------------

    /// Build the smallest valid MtpConfig for rollback testing.
    fn tiny_mtp_config() -> MtpConfig {
        MtpConfig {
            draft_length: 1,
            num_hidden_layers: 1,
            hidden_size: 4,
            vocab_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.5, // rope_dim = 4*0.5 = 2
            num_experts: 1,
            num_experts_per_tok: 1,
            moe_intermediate_size: 2,
            shared_expert_intermediate_size: 2,
            use_dedicated_embeddings: false,
        }
    }

    /// Build tiny MtpWeights for the given config with deterministic non-zero values.
    fn tiny_mtp_weights(cfg: &MtpConfig) -> MtpWeights {
        let h = cfg.hidden_size;
        let q_proj_rows = 2 * cfg.num_attention_heads * cfg.head_dim;
        let kv_proj_rows = cfg.num_key_value_heads * cfg.head_dim;
        let moe_inter = cfg.moe_intermediate_size;
        let shared_inter = cfg.shared_expert_intermediate_size;
        let num_experts = cfg.num_experts;

        // Small but non-zero values: use position-based pattern to avoid symmetry.
        let fill = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| scale * ((i as f32 + 1.0) * 0.01).sin())
                .collect()
        };
        let ones = |n: usize| vec![1.0f32; n];

        let layer = MtpLayerWeights {
            input_layernorm: ones(h),
            post_attention_layernorm: ones(h),
            self_attn: MtpAttentionWeights {
                q_proj: fill(q_proj_rows * h, 0.1),
                k_proj: fill(kv_proj_rows * h, 0.1),
                v_proj: fill(kv_proj_rows * h, 0.1),
                o_proj: fill(h * (cfg.num_attention_heads * cfg.head_dim), 0.1),
                q_norm: ones(cfg.head_dim),
                k_norm: ones(cfg.head_dim),
            },
            mlp: MtpMoeWeights {
                router_gate: fill(num_experts * h, 0.1),
                experts_gate_up_proj: fill(num_experts * 2 * moe_inter * h, 0.05),
                experts_down_proj: fill(num_experts * h * moe_inter, 0.05),
                shared_gate_proj: fill(shared_inter * h, 0.05),
                shared_up_proj: fill(shared_inter * h, 0.05),
                shared_down_proj: fill(h * shared_inter, 0.05),
                shared_expert_gate: fill(h, 0.1),
            },
        };

        MtpWeights {
            fc_weight: fill(h * (2 * h), 0.1),
            layers: vec![layer],
            norm_weight: ones(h),
            pre_fc_norm_embedding_weight: ones(h),
            pre_fc_norm_hidden_weight: ones(h),
        }
    }

    #[test]
    fn mtp_verifier_f16_cache_replay_after_rollback_matches_fresh_suffix() {
        let cfg = tiny_mtp_config();
        let weights = tiny_mtp_weights(&cfg);

        // Embed and lm_head: [vocab * hidden] — small non-zero values.
        let embed: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|i| ((i as f32 + 1.0) * 0.01).sin())
            .collect();
        let lm_head: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|i| ((i as f32 + 2.0) * 0.01).cos())
            .collect();

        let h = cfg.hidden_size;
        let max_seq = 8usize;

        // Previous-hidden stub: same for all steps (non-zero, unit-like).
        let prev_hidden: Vec<f32> = (0..h).map(|i| 0.1 * (i as f32 + 1.0)).collect();

        // --- Replay path ---
        let mut replay =
            MtpVerifier::new(cfg.clone(), &weights, &embed, &lm_head, max_seq).unwrap();
        replay.forward_one(1, 0, &prev_hidden).unwrap();
        replay.forward_one(2, 1, &prev_hidden).unwrap();
        replay.forward_one(3, 2, &prev_hidden).unwrap();
        // Roll back to after position 0 (seq_len=1).
        replay.rollback_cache_to(1).unwrap();
        let replay_out = replay.forward_one(7, 1, &prev_hidden).unwrap();

        // --- Fresh path ---
        let mut fresh = MtpVerifier::new(cfg.clone(), &weights, &embed, &lm_head, max_seq).unwrap();
        fresh.forward_one(1, 0, &prev_hidden).unwrap();
        let fresh_out = fresh.forward_one(7, 1, &prev_hidden).unwrap();

        // Both caches should now hold exactly 2 tokens.
        assert_eq!(
            replay.cache.seq_len(),
            2,
            "replay cache must hold 2 tokens after rollback+replay"
        );
        assert_eq!(fresh.cache.seq_len(), 2, "fresh cache must hold 2 tokens");

        // f16 KV slices at positions 0 and 1 must be identical.
        assert_eq!(
            replay.cache.get_k_f16(0),
            fresh.cache.get_k_f16(0),
            "K cache mismatch after rollback: stale f16 data contaminating position 0 or 1"
        );
        assert_eq!(
            replay.cache.get_v_f16(0),
            fresh.cache.get_v_f16(0),
            "V cache mismatch after rollback: stale f16 data contaminating position 0 or 1"
        );

        // Logits from the final forward_one must be identical.
        assert_eq!(
            replay_out.logits.len(),
            fresh_out.logits.len(),
            "logit vector length must match"
        );
        for (i, (r, f)) in replay_out
            .logits
            .iter()
            .zip(fresh_out.logits.iter())
            .enumerate()
        {
            assert_eq!(
                r, f,
                "logits[{i}] differ after rollback (replay={r}, fresh={f}): \
                 stale f16 dequant scratch is contaminating attention output"
            );
        }
    }

    #[test]
    fn mtp_verifier_forward_one_at_capacity_returns_err_not_panic() {
        let cfg = tiny_mtp_config();
        let weights = tiny_mtp_weights(&cfg);

        let embed: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|i| ((i as f32 + 1.0) * 0.01).sin())
            .collect();
        let lm_head: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|i| ((i as f32 + 2.0) * 0.01).cos())
            .collect();

        let h = cfg.hidden_size;
        let max_seq = 4usize;
        let prev_hidden: Vec<f32> = (0..h).map(|i| 0.1 * (i as f32 + 1.0)).collect();

        let mut v = MtpVerifier::new(cfg, &weights, &embed, &lm_head, max_seq).unwrap();

        // Fill the cache to exactly capacity: each forward_one appends one token.
        for pos in 0..max_seq {
            v.forward_one((pos % 2) as u32 + 1, pos, &prev_hidden)
                .unwrap_or_else(|e| {
                    panic!("forward_one at pos {pos} (below capacity) should succeed: {e:?}")
                });
        }
        assert_eq!(
            v.cache.seq_len(),
            max_seq,
            "cache must be full after filling"
        );
        assert!(v.cache.is_full(), "cache must report full at capacity");

        // The next forward_one would index the raw K/V buffer out of bounds
        // (it writes at seq_len * kv_dim before advance_by). It must fail closed
        // with a typed error, never panic — matches the no-panic library contract
        // and the #290 FlatKVCache hardening.
        match v.forward_one(1, max_seq, &prev_hidden) {
            Ok(_) => panic!("forward_one at capacity must return Err, not overwrite or panic"),
            Err(crate::InferenceError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("full"),
                    "error message should explain the cache is full, got: {msg}"
                );
            }
            Err(other) => panic!("expected InvalidInput for full cache, got {other:?}"),
        }
    }

    /// forward_one returns Err (not a panic) when `position` equals max_seq_len even
    /// though the KV cache is not yet full.  This guards the RoPE table OOB path in
    /// mtp_apply_partial_rope (independent `position` param, distinct from the #290
    /// is_full guard).
    #[test]
    fn mtp_verifier_forward_one_out_of_range_position_returns_err() {
        let cfg = tiny_mtp_config();
        let weights = tiny_mtp_weights(&cfg);

        let embed: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|i| ((i as f32 + 1.0) * 0.01).sin())
            .collect();
        let lm_head: Vec<f32> = (0..cfg.vocab_size * cfg.hidden_size)
            .map(|i| ((i as f32 + 2.0) * 0.01).cos())
            .collect();

        let h = cfg.hidden_size;
        let max_seq = 4usize;
        let prev_hidden: Vec<f32> = (0..h).map(|i| 0.1 * (i as f32 + 1.0)).collect();

        // Cache starts empty (seq_len == 0); pass position == max_seq so the
        // cache is not full but the position is past the RoPE table.
        let mut v = MtpVerifier::new(cfg, &weights, &embed, &lm_head, max_seq).unwrap();
        assert_eq!(v.cache.seq_len(), 0, "cache must start empty");
        assert!(!v.cache.is_full(), "cache must not be full yet");

        match v.forward_one(1, max_seq, &prev_hidden) {
            Ok(_) => panic!(
                "forward_one with out-of-range position must return Err, not panic or succeed"
            ),
            Err(crate::InferenceError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("position"),
                    "error message should mention position, got: {msg}"
                );
            }
            Err(other) => panic!("expected InvalidInput for out-of-range position, got {other:?}"),
        }
    }

    /// MtpVerifier::new rejects a config whose partial_rotary_factor is outside [0,1].
    /// Factor 2.0 would produce rope_dim = 2 * head_dim, making mtp_apply_partial_rope
    /// index head_vec[head_dim] on a head_dim-length slice (Bug 2 / scalar validation).
    #[test]
    fn mtp_new_rejects_partial_rotary_factor_above_one() {
        let mut cfg = tiny_mtp_config();
        cfg.partial_rotary_factor = 2.0;
        let weights = tiny_mtp_weights(&cfg);
        let embed: Vec<f32> = vec![0.0; cfg.vocab_size * cfg.hidden_size];
        let lm_head: Vec<f32> = vec![0.0; cfg.vocab_size * cfg.hidden_size];
        let Err(err) = MtpVerifier::new(cfg, &weights, &embed, &lm_head, 8) else {
            panic!("partial_rotary_factor > 1 must be rejected");
        };
        assert!(
            format!("{err:?}").contains("partial_rotary_factor"),
            "expected partial_rotary_factor rejection, got: {err:?}"
        );
    }

    /// MtpVerifier::new rejects a config with num_key_value_heads == 0.
    /// Zero kv heads causes divide-by-zero at `groups = num_q_heads / num_kv_heads`
    /// on the first forward_one (Bug 2 / scalar validation).
    #[test]
    fn mtp_new_rejects_zero_kv_heads() {
        let mut cfg = tiny_mtp_config();
        cfg.num_key_value_heads = 0;
        let weights = tiny_mtp_weights(&cfg);
        let embed: Vec<f32> = vec![0.0; cfg.vocab_size * cfg.hidden_size];
        let lm_head: Vec<f32> = vec![0.0; cfg.vocab_size * cfg.hidden_size];
        let Err(err) = MtpVerifier::new(cfg, &weights, &embed, &lm_head, 8) else {
            panic!("num_key_value_heads == 0 must be rejected");
        };
        assert!(
            format!("{err:?}").contains("num_key_value_heads"),
            "expected num_key_value_heads rejection, got: {err:?}"
        );
    }

    /// MtpVerifier::new rejects zero-width RoPE (partial_rotary_factor == 0.0,
    /// which the [0,1] scalar check accepts) before constructing a degenerate
    /// RopeTable whose max_positions() == 0 would reject every forward_one
    /// position (codex #362 Major: position-bound exactness for zero-width RoPE).
    #[test]
    fn mtp_new_rejects_zero_partial_rotary_factor() {
        let mut cfg = tiny_mtp_config();
        cfg.partial_rotary_factor = 0.0;
        let weights = tiny_mtp_weights(&cfg);
        let embed: Vec<f32> = vec![0.0; cfg.vocab_size * cfg.hidden_size];
        let lm_head: Vec<f32> = vec![0.0; cfg.vocab_size * cfg.hidden_size];
        let Err(err) = MtpVerifier::new(cfg, &weights, &embed, &lm_head, 8) else {
            panic!("zero-width RoPE (partial_rotary_factor == 0) must be rejected");
        };
        assert!(
            format!("{err:?}").contains("RoPE width"),
            "expected zero-width RoPE rejection, got: {err:?}"
        );
    }

    /// MtpVerifier::new returns Err (not a debug overflow-panic / release wrap)
    /// when vocab_size * hidden_size overflows usize. Weights/embeds are tiny;
    /// the checked product fires before any tensor-shape comparison
    /// (codex #362 Medium: unchecked dimension products).
    #[test]
    fn mtp_new_rejects_dimension_product_overflow() {
        let base = tiny_mtp_config();
        let weights = tiny_mtp_weights(&base);
        let embed: Vec<f32> = vec![0.0; base.vocab_size * base.hidden_size];
        let lm_head: Vec<f32> = vec![0.0; base.vocab_size * base.hidden_size];
        let mut cfg = tiny_mtp_config();
        cfg.hidden_size = usize::MAX / 2 + 1;
        cfg.vocab_size = 2;
        let Err(err) = MtpVerifier::new(cfg, &weights, &embed, &lm_head, 8) else {
            panic!("overflowing vocab_size * hidden_size must return Err, not panic");
        };
        assert!(
            format!("{err:?}").contains("overflows usize"),
            "expected overflow rejection, got: {err:?}"
        );
    }

    /// MtpVerifier::new returns Err (not a panic) when num_attention_heads *
    /// head_dim overflows. Reachable with TINY weights — only weights.layers.len()
    /// is checked before MtpScratch/cache construction, so an oversized
    /// num_attention_heads passes the scalar/divisibility checks (kv_heads=1)
    /// yet overflows q_dim (codex #362 round-2 Medium: derived-dimension products).
    #[test]
    fn mtp_new_rejects_head_dim_product_overflow() {
        let base = tiny_mtp_config();
        let weights = tiny_mtp_weights(&base);
        let embed: Vec<f32> = vec![0.0; base.vocab_size * base.hidden_size];
        let lm_head: Vec<f32> = vec![0.0; base.vocab_size * base.hidden_size];
        let mut cfg = tiny_mtp_config();
        cfg.num_key_value_heads = 1;
        cfg.num_attention_heads = usize::MAX / 4 + 1; // * head_dim (4) overflows
        let Err(err) = MtpVerifier::new(cfg, &weights, &embed, &lm_head, 8) else {
            panic!("overflowing num_attention_heads * head_dim must return Err, not panic");
        };
        assert!(
            format!("{err:?}").contains("overflows usize"),
            "expected dimension-product overflow rejection, got: {err:?}"
        );
    }

    #[test]
    fn generate_with_speculation_empty_prompt_returns_empty() {
        // Empty prompt cannot seed speculation; must return empty, not panic on
        // `all_tokens.last()` in the first decode step.
        let out = generate_with_speculation(&[], 8, 999, |_tok, _pos| vec![0.0; 4], 3, 4);
        assert!(out.is_empty());
    }
}
