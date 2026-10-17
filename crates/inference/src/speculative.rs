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
//! - Low-level: `NgramSpeculator::speculate` + `verify_draft` for custom loops.
//! - High-level: `generate_with_speculation` wraps any `forward_fn` closure.

/// Upper bound applied to a caller-supplied `max_ngram`.
///
/// [`NgramSpeculator::speculate`] tries every n-gram length from `max_ngram`
/// down to 2, and each attempt scans the whole prompt comparing `n` elements at
/// every offset, so the work per generated token grows with the square of the
/// attempted length and with prompt length. An unbounded `max_ngram` therefore
/// lets a caller turn one generation request into an arbitrarily long CPU-bound
/// search before a single token is produced.
///
/// The default is 5, and a match longer than a few dozen tokens is vanishingly
/// rare in practice, so clamping here costs nothing real and removes the
/// pathological case. The clamp is applied once, in
/// [`NgramSpeculator::new`], so every entry point inherits the same bound.
pub const MAX_NGRAM_LIMIT: usize = 64;

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
    /// Maximum n-gram length to try (tries `max_ngram` down to 2), already
    /// clamped to [`MAX_NGRAM_LIMIT`] by [`NgramSpeculator::new`].
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
    ///   descending order; first match wins. Values above
    ///   [`MAX_NGRAM_LIMIT`] are clamped to it.
    /// - `max_draft`: maximum draft tokens to propose per step (default: 4).
    pub fn new(prompt_tokens: Vec<u32>, max_ngram: usize, max_draft: usize) -> Self {
        Self {
            prompt_tokens,
            max_ngram: max_ngram.min(MAX_NGRAM_LIMIT),
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
                let end = start
                    .saturating_add(self.max_draft)
                    .min(self.prompt_tokens.len());
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

impl MtpConfig {
    /// Build the MTP config for a Qwen3.5 checkpoint from the model's own config.
    ///
    /// Every field comes from `cfg` except `draft_length`, which is a runtime choice of the
    /// caller rather than a property of the checkpoint.
    ///
    /// A routed-MoE model maps its expert fields across (`num_experts`, `num_experts_per_tok`,
    /// `moe_intermediate_size` and `shared_expert_intermediate_size`, with the same fallbacks
    /// the model config applies). A dense model, whose MTP layer carries a plain SwiGLU MLP,
    /// is expressed in the same layout as one expert, top-1 routing and a zero-width shared
    /// expert, with `moe_intermediate_size` set to the dense `intermediate_size`.
    /// [`MtpWeights::load_from_source`] reads the matching tensors for either shape.
    ///
    /// Errors when the model has no MTP layer, or is MoE but leaves the expert count or the
    /// experts per token unset.
    pub fn from_qwen35(
        cfg: &crate::model::qwen35_config::Qwen35Config,
        draft_length: usize,
    ) -> Result<Self, crate::error::InferenceError> {
        use crate::error::InferenceError;

        if cfg.mtp_num_hidden_layers == 0 {
            return Err(InferenceError::UnsupportedModel(
                "the model config declares no MTP layer (mtp_num_hidden_layers = 0)".into(),
            ));
        }
        let (experts, per_tok, moe_inter, shared_inter) = if cfg.is_moe() {
            let experts = cfg.num_experts.ok_or_else(|| {
                InferenceError::UnsupportedModel("MoE model config does not set num_experts".into())
            })?;
            let per_tok = cfg.num_experts_per_tok.ok_or_else(|| {
                InferenceError::UnsupportedModel(
                    "MoE model config does not set num_experts_per_tok".into(),
                )
            })?;
            (
                experts,
                per_tok,
                cfg.moe_intermediate_size(),
                cfg.shared_expert_intermediate_size(),
            )
        } else {
            (1, 1, cfg.intermediate_size, 0)
        };
        Ok(Self {
            draft_length,
            num_hidden_layers: cfg.mtp_num_hidden_layers,
            hidden_size: cfg.hidden_size,
            vocab_size: cfg.vocab_size,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta,
            partial_rotary_factor: cfg.partial_rotary_factor,
            num_experts: experts,
            num_experts_per_tok: per_tok,
            moe_intermediate_size: moe_inter,
            shared_expert_intermediate_size: shared_inter,
            use_dedicated_embeddings: cfg.mtp_use_dedicated_embeddings,
        })
    }
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
    /// Final RMSNorm weight: `[hidden_size]`
    pub norm_weight: Vec<f32>,
    /// Pre-FC normalization applied to embedding: `[hidden_size]`
    pub pre_fc_norm_embedding_weight: Vec<f32>,
    /// Pre-FC normalization applied to previous hidden: `[hidden_size]`
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

/// How one MTP layer's MLP is stored in a checkpoint.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MtpMlpLayout {
    /// `mlp.{gate,up,down}_proj.weight`: a single SwiGLU MLP.
    Dense,
    /// `mlp.experts.{gate_up_proj,down_proj}` plus a router and a shared expert.
    Routed,
}

impl MtpMlpLayout {
    /// Decide the layout of layer `layer` from which tensors the source holds.
    ///
    /// Exactly one of the two key families must be present; both or neither is an error that
    /// names the layer and the keys found.
    fn detect<S: crate::weights::TensorSource>(
        source: &mut S,
        layer: usize,
    ) -> Result<Self, crate::error::InferenceError> {
        use crate::error::InferenceError;

        const DENSE: [&str; 3] = [
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];
        const ROUTED: [&str; 2] = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"];
        let mut found = |keys: &[&'static str]| -> Result<Vec<&'static str>, InferenceError> {
            let mut present = Vec::new();
            for key in keys {
                if source.has_tensor(&format!("mtp.layers.{layer}.{key}"))? {
                    present.push(*key);
                }
            }
            Ok(present)
        };
        let dense = found(&DENSE)?;
        let routed = found(&ROUTED)?;
        match (dense.is_empty(), routed.is_empty()) {
            (false, true) => Ok(Self::Dense),
            (true, false) => Ok(Self::Routed),
            (false, false) => Err(InferenceError::UnsupportedModel(format!(
                "mtp.layers.{layer}: MLP layout is ambiguous, found dense keys {dense:?} and \
                 expert keys {routed:?}"
            ))),
            (true, true) => Err(InferenceError::UnsupportedModel(format!(
                "mtp.layers.{layer}: no MLP tensors found, expected dense keys {DENSE:?} or \
                 expert keys {ROUTED:?}"
            ))),
        }
    }
}

impl MtpWeights {
    /// Load all MTP weights from a tensor source, validating shapes.
    ///
    /// The MLP layout is detected per layer. A routed layer reads the expert, router and
    /// shared-expert tensors. A dense layer (`mlp.{gate,up,down}_proj.weight`, as in the
    /// Qwen3.5-0.8B checkpoint) is stored in the same fields as one expert with top-1 routing
    /// and a zero-width shared expert, which evaluates identically to the plain SwiGLU MLP: the
    /// router softmax over one expert is exactly 1.0 and the shared expert is not evaluated.
    /// The config must then say `num_experts = 1`, `num_experts_per_tok = 1` and
    /// `shared_expert_intermediate_size = 0`, with `moe_intermediate_size` the dense
    /// intermediate size (which the tensor shapes enforce). [`MtpConfig::from_qwen35`] produces
    /// such a config for a dense model.
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
                // Check the header-declared shape before materializing tensor data: a
                // source that cannot report a shape up front returns None and falls
                // through to the post-load check below, but any source that can
                // answer rejects a mismatch before the (potentially huge) allocation
                // and copy happen.
                if let Some(declared) = source.tensor_shape(name)?
                    && declared != expected
                {
                    return Err(InferenceError::ShapeMismatch {
                        name: name.to_string(),
                        expected: expected.to_vec(),
                        actual: declared,
                    });
                }
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
            let expected_fc = [hidden, 2 * hidden];
            if let Some(declared) = source.tensor_shape(name)?
                && declared != expected_fc
            {
                return Err(InferenceError::UnsupportedModel(format!(
                    "unexpected mtp.fc.weight shape {declared:?}; expected fusion projection [{hidden}, {}]",
                    2 * hidden
                )));
            }
            let (data, shape) = source.get_f32_tensor_owned(name)?;
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
            let mlp = match MtpMlpLayout::detect(source, i)? {
                MtpMlpLayout::Routed => {
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
                        if let Some(declared) = source.tensor_shape(&name)?
                            && declared.iter().product::<usize>() != hidden
                        {
                            return Err(InferenceError::ShapeMismatch {
                                name,
                                expected: vec![hidden],
                                actual: declared,
                            });
                        }
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

                    MtpMoeWeights {
                        router_gate,
                        experts_gate_up_proj,
                        experts_down_proj,
                        shared_gate_proj,
                        shared_up_proj,
                        shared_down_proj,
                        shared_expert_gate,
                    }
                }
                MtpMlpLayout::Dense => {
                    if num_experts != 1 || cfg.num_experts_per_tok != 1 || shared_inter != 0 {
                        return Err(InferenceError::Inference(format!(
                            "mtp.layers.{i} has a dense MLP, which needs num_experts = 1, \
                             num_experts_per_tok = 1 and shared_expert_intermediate_size = 0, \
                             but the config has {num_experts}, {} and {shared_inter}",
                            cfg.num_experts_per_tok
                        )));
                    }
                    let gate = load_checked(
                        source,
                        &format!("mtp.layers.{i}.mlp.gate_proj.weight"),
                        &[moe_inter, hidden],
                    )?;
                    let up = load_checked(
                        source,
                        &format!("mtp.layers.{i}.mlp.up_proj.weight"),
                        &[moe_inter, hidden],
                    )?;
                    let down = load_checked(
                        source,
                        &format!("mtp.layers.{i}.mlp.down_proj.weight"),
                        &[hidden, moe_inter],
                    )?;
                    // `forward_one` reads each expert's `[2 * moe_inter, hidden]` block as the
                    // gate rows followed by the up rows.
                    let mut experts_gate_up_proj = gate;
                    experts_gate_up_proj.extend_from_slice(&up);
                    MtpMoeWeights {
                        router_gate: vec![0.0; hidden],
                        experts_gate_up_proj,
                        experts_down_proj: down,
                        shared_gate_proj: Vec::new(),
                        shared_up_proj: Vec::new(),
                        shared_down_proj: Vec::new(),
                        shared_expert_gate: vec![0.0; hidden],
                    }
                }
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
                mlp,
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
        if !config
            .num_attention_heads
            .is_multiple_of(config.num_key_value_heads)
        {
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
        // try_new re-validates all dimension products at the allocation boundary.
        // The checked_dim calls above provide an upstream guard; try_new ensures
        // no unchecked product reaches Vec::with_capacity even if a future
        // refactor removes or reorders those guards.
        let cache = crate::kv_cache::flat::FlatKVCache::try_new(cache_cfg)?;
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

    /// Compute this position's MTP K/V row and write it at the cache cursor without advancing
    /// the cursor.
    ///
    /// Shared by [`Self::forward_one`] and [`Self::append_kv_row`], so a row appended for an
    /// already-committed token is built by the same code a full step uses: embedding lookup,
    /// pre-fusion norms, fusion projection, input norm, K and V projections, K norm and partial
    /// RoPE. On return `scratch.hidden` holds the input-normed hidden state and
    /// `scratch.residual` the fused hidden state before that norm; `forward_one` continues from
    /// there.
    fn stage_kv_row(
        &mut self,
        token_id: u32,
        position: usize,
        previous_hidden: &[f32],
    ) -> Result<(), crate::error::InferenceError> {
        use crate::error::InferenceError;
        use crate::forward::cpu::matmul_bt;
        use crate::model::qwen35::qwen35_rms_norm;

        // At capacity (`seq_len == max_seq_len`) this step indexes per-position
        // buffers (RoPE position tables, then the raw `k_buffer_mut(0)` K/V write at
        // `seq_len * kv_dim`) out of bounds, bypassing the bounds-checked
        // `FlatKVCache` append API hardened in #290, and panics. Fail closed with a
        // typed error instead, matching the no-panic-in-library contract and
        // `advance_by`'s overflow behaviour.
        if self.cache.is_full() {
            return Err(InferenceError::InvalidInput(format!(
                "MTP KV cache is full ({} tokens); call rollback_cache_to or reset_cache before writing another row",
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
                "MTP position {position} out of range for RoPE table of {} positions",
                self.rope.max_positions()
            )));
        }

        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let head_dim = cfg.head_dim;
        let num_kv_heads = cfg.num_key_value_heads;
        let rope_dim = (head_dim as f32 * cfg.partial_rotary_factor) as usize;
        let eps = cfg.rms_norm_eps;

        // Validate
        if token_id as usize >= vocab {
            return Err(InferenceError::Inference(format!(
                "MTP token_id {token_id} >= vocab_size {vocab}"
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
        let tok = token_id as usize;
        self.scratch
            .embedding
            .copy_from_slice(&self.embed_tokens[tok * hidden..(tok + 1) * hidden]);

        // 2. Pre-fusion normalization (shifted RMSNorm: x / rms(x) * (1 + gamma))
        self.scratch.norm_hidden.copy_from_slice(previous_hidden);
        qwen35_rms_norm(
            &mut self.scratch.norm_hidden,
            &self.weights.pre_fc_norm_hidden_weight,
            hidden,
            eps,
        );

        self.scratch
            .norm_embedding
            .copy_from_slice(&self.scratch.embedding);
        qwen35_rms_norm(
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

        // Pre-attention layernorm (shifted RMSNorm)
        qwen35_rms_norm(
            &mut self.scratch.hidden,
            &layer.input_layernorm,
            hidden,
            eps,
        );

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

        // Per-head K normalization (shifted RMSNorm over each head's head_dim row)
        for h in 0..num_kv_heads {
            let start = h * head_dim;
            qwen35_rms_norm(
                &mut self.scratch.k[start..start + head_dim],
                &layer.self_attn.k_norm,
                head_dim,
                eps,
            );
        }

        // Partial RoPE (stride-half pairing; see mtp_apply_partial_rope)
        for h in 0..num_kv_heads {
            let start = h * head_dim;
            mtp_apply_partial_rope(
                &mut self.scratch.k[start..start + head_dim],
                position,
                &self.rope,
                rope_dim,
            );
        }

        // Write K, V to MTP KV cache at current seq_len position (convert f32→f16 on write).
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
        Ok(())
    }

    /// Append one K/V-only row to the MTP cache for `token_id` at absolute `position`.
    ///
    /// `previous_hidden` has the same meaning as in [`Self::forward_one`]: the target's
    /// post-final-norm hidden state whose output predicted `token_id`. The row is written by the
    /// same code as a full step up to its cache write and nothing after it runs: a row appended
    /// for an already-committed token is only ever a future key, so the query, attention, MoE
    /// FFN and logits are not computed. Advances the cache by one row.
    pub(crate) fn append_kv_row(
        &mut self,
        token_id: u32,
        position: usize,
        previous_hidden: &[f32],
    ) -> Result<(), crate::error::InferenceError> {
        self.stage_kv_row(token_id, position, previous_hidden)?;
        self.cache.advance_by(1)?;
        Ok(())
    }

    /// Run one MTP transformer forward step.
    ///
    /// - `input_token_id`: token whose embedding is fed into the MTP module
    /// - `position`: sequence position of `input_token_id` (used for RoPE and causal mask)
    /// - `previous_hidden`: post-final-norm hidden state of the target model whose output
    ///   (through the output projection) predicted `input_token_id`, i.e. the target's hidden
    ///   state at `position - 1`. Each cache row therefore pairs a hidden state with the
    ///   embedding of the token that follows it. [`Self::draft_tokens_with_logits`] feeds the
    ///   caller's target hidden state to its first step and this head's own output hidden state
    ///   to the later ones.
    pub fn forward_one(
        &mut self,
        input_token_id: u32,
        position: usize,
        previous_hidden: &[f32],
    ) -> Result<MtpForwardOutput, crate::error::InferenceError> {
        use crate::forward::cpu::matmul_bt;
        use crate::model::qwen35::qwen35_rms_norm;

        self.stage_kv_row(input_token_id, position, previous_hidden)?;

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

        let layer = &self.weights.layers[0];

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

        // Per-head Q normalization (shifted RMSNorm over each head's head_dim row)
        for h in 0..num_q_heads {
            let start = h * head_dim;
            qwen35_rms_norm(
                &mut self.scratch.q[start..start + head_dim],
                &layer.self_attn.q_norm,
                head_dim,
                eps,
            );
        }

        // Partial RoPE (stride-half pairing; see mtp_apply_partial_rope)
        for h in 0..num_q_heads {
            let start = h * head_dim;
            mtp_apply_partial_rope(
                &mut self.scratch.q[start..start + head_dim],
                position,
                &self.rope,
                rope_dim,
            );
        }

        // This position's K/V row was staged above at the cache cursor; the cursor advances
        // after attention.
        let write_pos = self.cache.seq_len();
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
            for t in 0..cur_seq_len {
                let k_off = t * kv_dim + kvh * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += self.scratch.q[q_off + d] * k_all[k_off + d];
                }
                self.scratch.scores[scores_start + t] = dot * scale;
            }

            // ADR-080 C1: fail-closed softmax row contract (#785).
            // This is MtpVerifier's own GQA attention over its KV cache, not a
            // vocabulary/logit softmax -- a NaN/`+inf` cached score must zero the
            // whole row, not silently poison every weight via `1.0 / sum_exp`.
            let row = &mut self.scratch.scores[scores_start..scores_start + cur_seq_len];
            let (max_score, any_nan) = crate::attention::softmax_row::row_max_and_any_nan(row);
            if crate::attention::softmax_row::row_fails_closed_pre_exp(max_score, any_nan) {
                row.fill(0.0);
            } else {
                let mut sum_exp = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max_score).exp();
                    sum_exp += *v;
                }
                crate::attention::softmax_row::finalize_row(row, sum_exp);
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

        // Post-attention layernorm (shifted RMSNorm), save residual for FFN
        self.scratch.residual.copy_from_slice(&self.scratch.hidden);
        qwen35_rms_norm(
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

        // Shared expert. A zero-width one (the dense-MLP layout) adds nothing, so it is not
        // evaluated: that skips zero-sized matmuls whose handling differs by backend.
        if shared_inter > 0 {
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
        }

        // FFN residual
        for i in 0..hidden {
            self.scratch.hidden[i] = self.scratch.residual[i] + self.scratch.moe_out[i];
        }

        // 5. Final MTP norm (shifted RMSNorm)
        qwen35_rms_norm(
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
    /// `hidden_predicting_current_token` is the target's post-final-norm hidden state whose
    /// output predicted `current_token_id` (the target's hidden state at
    /// `current_position - 1`); see [`Self::forward_one`]. It is not the hidden state computed
    /// after processing `current_token_id` itself.
    ///
    /// Stops early if `eos_token` is produced.
    #[allow(clippy::explicit_counter_loop)]
    pub fn draft_tokens_with_logits(
        &mut self,
        current_token_id: u32,
        current_position: usize,
        hidden_predicting_current_token: &[f32],
        eos_token: Option<u32>,
    ) -> Result<MtpDraft, crate::error::InferenceError> {
        let mut tokens = Vec::with_capacity(self.config.draft_length);
        let mut logits = Vec::with_capacity(self.config.draft_length);
        let mut next_input = current_token_id;
        let mut next_hidden: Vec<f32> = hidden_predicting_current_token.to_vec();
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
    /// See [`Self::draft_tokens_with_logits`] for the meaning of
    /// `hidden_predicting_current_token`.
    ///
    /// Stops early if `eos_token` is produced.
    pub fn draft_tokens(
        &mut self,
        current_token_id: u32,
        current_position: usize,
        hidden_predicting_current_token: &[f32],
        eos_token: Option<u32>,
    ) -> Result<Vec<u32>, crate::error::InferenceError> {
        Ok(self
            .draft_tokens_with_logits(
                current_token_id,
                current_position,
                hidden_predicting_current_token,
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
    ///
    /// `start_pos` must equal [`Self::cache_position`]. Implementations must reject a
    /// mismatch before forwarding any token so KV placement and positional encoding
    /// cannot observe different positions.
    fn verify_tokens(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<Vec<f32>>, crate::error::InferenceError>;

    /// Post-final-norm hidden state of the most recently processed token: the vector the
    /// target's output projection read to produce that token's logits.
    ///
    /// [`mtp_verify_draft_with_seed`] reads it once, before it calls
    /// [`Self::verify_tokens_with_hidden`]. At that point the most recently processed token is
    /// the pending token, so the value is the hidden state that predicted the first draft token.
    /// The default returns `None`, meaning the implementation cannot supply it. An
    /// implementation must return `None` rather than a hidden state in any other basis, because
    /// a wrong vector is worse than a missing one.
    fn last_hidden(&self) -> Option<Vec<f32>> {
        None
    }

    /// Same forward as [`Self::verify_tokens`], additionally returning the target's
    /// post-final-norm hidden states when the implementation can supply them.
    ///
    /// The first element is exactly what `verify_tokens` returns. The second is `Some(h)` only
    /// when `h[i]` is the target's post-final-norm hidden state after processing `tokens[i]`,
    /// with one row per token and every row `hidden_size` long. `None` means the implementation
    /// cannot supply every row; it must return `None` rather than a partial, approximated or
    /// differently normalized set. The default delegates to `verify_tokens` and returns `None`.
    ///
    /// [`mtp_verify_draft_with_seed`] uses these rows together with [`Self::last_hidden`] to
    /// keep the MTP cache aligned with the target. When either is `None` it behaves exactly as
    /// it did before these methods existed.
    #[allow(clippy::type_complexity)]
    fn verify_tokens_with_hidden(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<Vec<f32>>>), crate::error::InferenceError> {
        Ok((self.verify_tokens(tokens, start_pos)?, None))
    }

    /// Snapshot every GDN layer's recurrent state (S matrices + conv buffer). See ADR-052.
    ///
    /// Implementors that have no GDN layers (e.g. pure-attention test mocks) return an empty
    /// `Vec`. Callers must treat the returned snapshot as opaque.
    fn snapshot_gdn_states(&self) -> crate::attention::gdn::GdnSnapshot;

    /// Restore every GDN layer's recurrent state from a prior `snapshot_gdn_states` call.
    ///
    /// The snapshot must have been taken from the same model instance. Implementors that
    /// hold no GDN state accept an empty snapshot and do nothing.
    fn restore_gdn_states(&mut self, snapshot: &crate::attention::gdn::GdnSnapshot);
}

/// Fixed default seed used by [`mtp_verify_draft`] when no seed is supplied.
///
/// #329: the trait-level MTP verify wrapper must never fall back to
/// [`SpecRng::from_clock`] the way the lower-level [`rejection_sample_draft`] does for
/// its own `seed=None`. `MTP_VERIFY_DEFAULT_SEED` gives `mtp_verify_draft`'s probabilistic
/// rejection sampling a fixed, deterministic-by-design default: identical inputs always
/// produce identical accepted tokens, matching the same reproducible-sampling contract
/// [`crate::sampling::Sampler::with_seed`] documents. This value is arbitrary (mirrors
/// [`SpecRng::new`]'s zero-seed fallback) and carries no other meaning.
const MTP_VERIFY_DEFAULT_SEED: u64 = 0x853c_49e6_748f_ea9b;

/// Verify a speculative MTP draft against the target model, with an explicit seed for the
/// probabilistic rejection-sampling RNG.
///
/// Returns an [`MtpVerifyResult`] with accepted tokens, optional fallback, metrics,
/// and properly rolled-back caches.
///
/// # Seed contract
///
/// - `Some(seed)`: probabilistic rejection sampling draws from an RNG seeded with `seed`.
///   Identical inputs and the same `seed` reproduce identical accepted tokens and
///   fallback token across runs — thread the caller's `GenerateConfig.seed` here to match
///   the rest of a reproducible generation run. Note `SpecRng::new` maps `seed == 0` to
///   its internal fallback constant, so `Some(0)` behaves identically to that fallback
///   seed rather than as a distinct zero-seeded stream (reproducibility still holds).
/// - `None`: uses `MTP_VERIFY_DEFAULT_SEED`, a fixed deterministic default — **not**
///   `SpecRng::from_clock`. This is documented compatibility behavior for
///   [`mtp_verify_draft`], the reverse of the lower-level [`rejection_sample_draft`], whose
///   own `seed=None` is intentionally clock-seeded for non-reproducible standalone use.
///
/// # Temperature contract (#388 / ADR-050)
///
/// `temperature` is threaded straight through to [`rejection_sample_draft`]'s `p`/`q`
/// softmax — pass the caller's `GenerateConfig.temperature` here so the acceptance ratio
/// `min(1, p(x)/q(x))` is computed under the same distribution the target decoder actually
/// samples from. Must be finite and `> 0`; `rejection_sample_draft` fails closed on
/// `InvalidInput` otherwise.
///
/// # Hidden-state contract
///
/// `hidden_predicting_current_token` is the target's post-final-norm hidden state whose output
/// predicted `current_token_id` (the hidden state at `current_position - 1`). The target has
/// already processed `current_token_id`: `initial_target_logits` is its output for that token
/// and `target.cache_position()` is `current_position + 1`.
///
/// Each MTP cache row pairs a target hidden state with the embedding of the token that follows
/// it. Drafting writes the row for `current_token_id` from the caller's hidden state and the
/// row for each further draft token from the draft head's own hidden state. When the target
/// supplies [`MtpTargetVerifier::last_hidden`] and
/// [`MtpTargetVerifier::verify_tokens_with_hidden`] rows, the rows for the accepted draft tokens
/// are rewritten from the target's own hidden states instead: the MTP cache is rolled back to
/// just after the row for `current_token_id`, then one row is appended per accepted draft
/// token, the first from `last_hidden` and each later one from the verify row of the draft
/// token before it. The cache then ends at the same logical position as the target's whether
/// the draft was fully accepted, partly accepted or rejected, and the row for the last
/// accepted token is no longer missing. Without both inputs the cache is rolled back to the
/// drafted rows of the accepted prefix, as before.
///
/// # Errors
///
/// On `Err`, the verifier and target caches (and GDN state) are restored to their
/// pre-call positions on a **best-effort** basis: if a rollback call itself fails, the
/// original error is still returned and cache coherence is no longer guaranteed. A caller
/// recovering from an error must therefore treat the speculative context as invalid and
/// rebuild it rather than assume the caches are at a known position.
pub fn mtp_verify_draft_with_seed<T: MtpTargetVerifier>(
    verifier: &mut MtpVerifier<'_>,
    current_token_id: u32,
    current_position: usize,
    hidden_predicting_current_token: &[f32],
    initial_target_logits: &[f32],
    eos_token: Option<u32>,
    target: &mut T,
    temperature: f32,
    seed: Option<u64>,
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
        hidden_predicting_current_token,
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
    // The hidden state after the pending token must be read before `verify_tokens_with_hidden`
    // advances the target past it.
    let last_hidden = target.last_hidden();
    let (target_logits, hidden_rows) =
        match target.verify_tokens_with_hidden(&draft, current_position + 1) {
            Ok(l) => l,
            Err(e) => {
                let _ = verifier.rollback_cache_to(mtp_start);
                let _ = target.rollback_cache_to(target_start);
                target.restore_gdn_states(&gdn_snap);
                return Err(e);
            }
        };
    let target_forwards = 1;
    if let Some(rows) = hidden_rows.as_deref()
        && rows.len() != draft_len
    {
        let _ = verifier.rollback_cache_to(mtp_start);
        let _ = target.rollback_cache_to(target_start);
        target.restore_gdn_states(&gdn_snap);
        return Err(crate::error::InferenceError::InvalidInput(format!(
            "verify_tokens_with_hidden returned {} hidden rows for {draft_len} tokens",
            rows.len()
        )));
    }

    // Probabilistic rejection sampling (ADR-050): pass draft logits and greedy=false so
    // every draft token is accepted with probability min(1, p(x)/q(x)).
    // #282: rejection_sample_draft doesn't touch the caches, but it is fallible (e.g. on a
    // draft/logits length mismatch). By this point verify_tokens has already advanced the
    // target cache and draft generation advanced the verifier cache, so an early Err here
    // must restore both to their pre-call positions.
    // #329: never pass this wrapper's `seed=None` straight through — that would reach
    // `SpecRng::from_clock()` inside `rejection_sample_draft` and make `mtp_verify_draft`
    // non-reproducible. Resolve to `MTP_VERIFY_DEFAULT_SEED` first so the wrapper's own
    // `seed=None` stays deterministic-by-design.
    let rejection_seed = Some(seed.unwrap_or(MTP_VERIFY_DEFAULT_SEED));
    let rs = match rejection_sample_draft(
        &draft,
        &draft_logits,
        initial_target_logits,
        &target_logits,
        false,
        temperature,
        rejection_seed,
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
    // #389: rs.bonus_token is defined for both rejection-correction and full-accept
    // (see rejection_sample_draft), so expose it unconditionally here — dropping it on
    // full accept would discard a token the target already forward-passed and sampled.
    let mut fallback_token = rs.bonus_token;

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
    // With the target's hidden states the accepted draft rows are rewritten below, so keep only
    // the row for `current_token_id`, whose hidden state came from the caller.
    let target_hiddens = last_hidden.as_deref().zip(hidden_rows.as_deref());
    let mtp_keep = if target_hiddens.is_some() {
        mtp_start + 1
    } else {
        mtp_start + accepted_count
    };
    verifier.rollback_cache_to(mtp_keep)?;
    target.rollback_cache_to(target_start + accepted_count)?;

    // Drafting writes one MTP row per draft INPUT (the current token and every draft token but
    // the last) and takes the hidden state of every row after the first from the draft head
    // itself. Replace them with rows built from the target's hidden states: draft token `k`
    // (1-based, at `current_position + k`) pairs with the hidden state that predicted it, which
    // is `last_hidden` for the first and the verify row of the previous draft token after that.
    if let Some((first, rows)) = target_hiddens {
        for k in 1..=accepted_count {
            let position = current_position + k;
            if verifier.cache.is_full() || position >= verifier.rope.max_positions() {
                // No room for the row; the next draft step reports the same condition.
                break;
            }
            let predictor = if k == 1 {
                first
            } else {
                rows[k - 2].as_slice()
            };
            if let Err(e) = verifier.append_kv_row(draft[k - 1], position, predictor) {
                let _ = verifier.rollback_cache_to(mtp_start);
                let _ = target.rollback_cache_to(target_start);
                target.restore_gdn_states(&gdn_snap);
                return Err(e);
            }
        }
    }

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

/// Verify a speculative MTP draft against the target model.
///
/// Compatibility shim over [`mtp_verify_draft_with_seed`] that always passes `seed=None`,
/// i.e. `MTP_VERIFY_DEFAULT_SEED` — see that function's "Seed contract" section for what
/// `None` means here. Callers that want to thread a caller-controlled seed (e.g. a
/// `GenerateConfig.seed`) should call [`mtp_verify_draft_with_seed`] directly.
///
/// `temperature` (ADR-050 `T`) is forwarded unchanged — pass the caller's
/// `GenerateConfig.temperature` so the acceptance ratio is computed under the same
/// distribution the target decoder samples from (#388).
///
/// `hidden_predicting_current_token` and the MTP cache alignment are described under
/// "Hidden-state contract" on [`mtp_verify_draft_with_seed`].
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
    hidden_predicting_current_token: &[f32],
    initial_target_logits: &[f32],
    eos_token: Option<u32>,
    target: &mut T,
    temperature: f32,
) -> Result<MtpVerifyResult, crate::error::InferenceError> {
    mtp_verify_draft_with_seed(
        verifier,
        current_token_id,
        current_position,
        hidden_predicting_current_token,
        initial_target_logits,
        eos_token,
        target,
        temperature,
        None,
    )
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

/// Compute stable softmax over `logits / temperature` into `out` (ADR-050).
///
/// `scratch` and `out` must have the same length as `logits`; `scratch` is
/// overwritten with the temperature-scaled logits. `temperature == 1.0` skips
/// the scaling pass entirely. `temperature` must already be validated finite
/// and `> 0` by the caller, and `1.0 / temperature` must be finite. A
/// non-positive divisor would sign-flip or NaN the scaled logits, and an
/// infinite reciprocal turns the centered maximum into `0.0 * inf = NaN`.
fn softmax_into_temperature(
    logits: &[f32],
    temperature: f32,
    scratch: &mut [f32],
    out: &mut [f32],
) {
    if temperature == 1.0 {
        softmax_into(logits, out);
        return;
    }
    let inv = 1.0 / temperature;
    // Center only when `inv > 1.0`: that is the one regime where scaling a huge
    // finite logit can overflow to +inf and poison the row. Output for
    // `temperature >= 1.0` stays bit-identical to the uncentered formula.
    let center = if inv > 1.0 {
        let raw = logits
            .iter()
            .copied()
            .filter(|l| l.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);
        if raw.is_finite() { raw } else { 0.0 }
    } else {
        0.0
    };
    for (s, &l) in scratch.iter_mut().zip(logits.iter()) {
        *s = (l - center) * inv;
    }
    softmax_into(scratch, out);
}

/// Validate that a logit row is usable for probabilistic softmax sampling.
///
/// The probabilistic path turns every logit row into a probability distribution
/// via [`softmax_into`]. A `NaN` or `+inf` logit poisons that softmax: the sum
/// becomes `NaN`, the `sum > 0.0` normalization guard is skipped, and `out` is
/// left holding un-normalized / `NaN` mass, which silently corrupts the
/// acceptance test `min(1, p/q)` and breaks ADR-050's distribution-preservation
/// guarantee. `-inf` is a legitimate zero-probability sentinel, but only when at
/// least one finite logit remains so the softmax is well defined. Fail closed on
/// everything else (same bug class as the #327 / #335 / #361 sampling hardening).
fn validate_probabilistic_logits(
    label: &str,
    logits: &[f32],
) -> Result<(), crate::error::InferenceError> {
    use crate::error::InferenceError;
    let mut has_finite = false;
    for (i, &l) in logits.iter().enumerate() {
        if l.is_nan() || (l.is_infinite() && l.is_sign_positive()) {
            return Err(InferenceError::InvalidInput(format!(
                "{label}[{i}] must be finite or -inf for probabilistic rejection sampling, got {l}"
            )));
        }
        has_finite |= l.is_finite();
    }
    if !has_finite {
        return Err(InferenceError::InvalidInput(format!(
            "{label} has no finite logit; probabilistic rejection sampling needs a well-defined softmax"
        )));
    }
    Ok(())
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
/// acceptance. Using `target_logits[i]` to verify `draft_tokens[i]` would be
/// an off-by-one against the algorithm (Leviathan Algorithm 1 evaluates
/// `p_i(x_i)`, not `p_{i+1}(x_i)`).
///
/// For each position `i`:
/// - Compute `p[x_i] = softmax(target_logits_i / T)[x_i]` (target probability
///   under the position-`i` distribution) and `q[x_i] = softmax(draft_logits_i /
///   T)[x_i]` (draft probability), per ADR-050.
/// - Accept `x_i` with probability `min(1, p[x_i] / q[x_i])`.
/// - On rejection, sample the bonus token from the adjusted distribution
///   `norm(max(0, p - q))`, which exactly recovers the target distribution.
/// - After all tokens accepted, sample one bonus token from
///   `softmax(target_logits[n - 1] / T)` — the target's prediction for the
///   position after the last accepted draft token.
///
/// When `greedy` is `true` (temperature = 0 / top_k = 1), the function skips
/// probability computation entirely and falls back to argmax comparison, which is
/// strictly faster and produces the same result as the standard greedy verifier.
/// `temperature` is ignored in that case (argmax is invariant to positive scaling).
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
/// - `temperature`: sampling temperature (ADR-050 `T`) applied to both `p` and `q`
///   before softmax in the probabilistic path. Must be finite and `> 0` when
///   `greedy == false`; ignored when `greedy == true`.
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
/// - `temperature` is non-finite or `<= 0.0`, or `1.0 / temperature` is not
///   finite (checked only when `greedy == false`).
pub fn rejection_sample_draft(
    draft_tokens: &[u32],
    draft_logits: &[Vec<f32>],
    initial_target_logits: &[f32],
    target_logits: &[Vec<f32>],
    greedy: bool,
    temperature: f32,
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
        // Value finiteness: the probabilistic path softmaxes every row, so a
        // NaN/+inf logit would silently poison the distribution and corrupt the
        // accept/reject decision. The greedy path is robust (argmax skips NaN),
        // so this guard is probabilistic-only.
        validate_probabilistic_logits("initial_target_logits", initial_target_logits)?;
        for (i, tl) in target_logits.iter().enumerate() {
            validate_probabilistic_logits(&format!("target_logits[{i}]"), tl)?;
        }
        for (i, dl) in draft_logits.iter().enumerate() {
            validate_probabilistic_logits(&format!("draft_logits[{i}]"), dl)?;
        }
        // ADR-050 defines T as a caller-provided divisor of both p and q's logits;
        // a non-finite or non-positive T would poison every softmax (NaN, or a
        // negative-T sign flip that turns the max logit into the min), so fail
        // closed rather than silently substituting some other value (#388).
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(InferenceError::InvalidInput(format!(
                "probabilistic rejection sampling requires a finite temperature > 0, got {temperature}"
            )));
        }
        // A tiny positive temperature whose reciprocal overflows to +inf would
        // turn the centered maximum into `0.0 * inf = NaN` in the softmax, so
        // refuse it too rather than produce a non-distribution.
        if !(1.0 / temperature).is_finite() {
            return Err(InferenceError::InvalidInput(format!(
                "probabilistic rejection sampling requires a temperature whose reciprocal is finite, got {temperature:e}"
            )));
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
            #[expect(
                clippy::expect_used,
                reason = "target_logits carries one distribution per draft position, so it is never empty here"
            )]
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

    // Pre-allocate probability buffers for reuse across positions. `scaled` is
    // scratch space for `softmax_into_temperature`'s temperature-divided logits.
    let mut p_buf = vec![0.0f32; vocab];
    let mut q_buf = vec![0.0f32; vocab];
    let mut scaled = vec![0.0f32; vocab];

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

        softmax_into_temperature(target_dist_for(i), temperature, &mut scaled, &mut p_buf);
        softmax_into_temperature(&draft_logits[i], temperature, &mut scaled, &mut q_buf);

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
        #[expect(
            clippy::expect_used,
            reason = "target_logits carries one distribution per draft position, so it is never empty here"
        )]
        softmax_into_temperature(
            target_logits.last().expect("n > 0"),
            temperature,
            &mut scaled,
            &mut p_buf,
        );
        // Weighted sample from the full target distribution.
        let mut cumsum = 0.0f32;
        #[expect(
            clippy::expect_used,
            reason = "target_logits carries one distribution per draft position, so it is never empty here"
        )]
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
/// # Metal state warning
///
/// Do not pass a live
/// [`MetalQwen35State::forward_step`](crate::forward::metal_qwen35::MetalQwen35State::forward_step)
/// callback after Metal prefill. This compatibility wrapper feeds the final
/// prompt token at `position = prompt_tokens.len()` on its first callback,
/// which would duplicate that token in the Metal KV/GDN state. Use
/// [`MetalQwen35State::generate_with_speculation`](crate::forward::metal_qwen35::MetalQwen35State::generate_with_speculation)
/// instead; it owns admission, prefill, and state advancement.
///
/// # Arguments
///
/// - `prompt_tokens`: tokenised prompt (caller must have already run prefill).
/// - `max_new_tokens`: generation budget.
/// - `eos_token`: stop token ID.
/// - `forward_fn`: `(token_id, position) -> logits`.
/// - `max_ngram`: longest n-gram to try (default: 5); clamped to
///   [`MAX_NGRAM_LIMIT`].
/// - `max_draft`: max draft tokens per speculation step (default: 4).
///
/// # Returns
///
/// The generated token IDs (excluding the prompt).
///
/// # Errors
///
/// Returns [`crate::error::InferenceError::Inference`] with the message
/// `"empty prompt"` when `prompt_tokens` is empty.
pub fn generate_with_speculation<F>(
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    eos_token: u32,
    mut forward_fn: F,
    max_ngram: usize,
    max_draft: usize,
) -> Result<Vec<u32>, crate::error::InferenceError>
where
    F: FnMut(u32, usize) -> Vec<f32>,
{
    if prompt_tokens.is_empty() {
        return Err(crate::error::InferenceError::Inference(
            "empty prompt".into(),
        ));
    }

    let speculator = NgramSpeculator::new(prompt_tokens.to_vec(), max_ngram, max_draft);
    let mut generated: Vec<u32> = Vec::new();
    let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();
    let mut pos = prompt_tokens.len();

    while generated.len() < max_new_tokens {
        let draft = speculator.speculate(&all_tokens);
        #[expect(
            clippy::expect_used,
            reason = "prompt_tokens seeds all_tokens before the loop"
        )]
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
                return Ok(generated);
            }
            if t == eos_token {
                return Ok(generated);
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

    Ok(generated)
}

/// Drive n-gram speculative generation from logits produced by a completed
/// prompt prefill.
///
/// Unlike [`generate_with_speculation`], the callback receives each newly
/// emitted token exactly once at its true absolute position. The final token
/// admitted by `max_new_tokens` is also forwarded so the caller's live state
/// represents the full returned sequence.
#[cfg(any(test, all(target_os = "macos", feature = "metal-gpu")))]
pub(crate) fn generate_with_speculation_from_prefill<F>(
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    eos_token: u32,
    mut logits: Vec<f32>,
    mut forward_fn: F,
    max_ngram: usize,
    max_draft: usize,
) -> Result<Vec<u32>, crate::error::InferenceError>
where
    F: FnMut(u32, usize) -> Result<Vec<f32>, crate::error::InferenceError>,
{
    if prompt_tokens.is_empty() {
        return Err(crate::error::InferenceError::Inference(
            "empty prompt".into(),
        ));
    }

    let speculator = NgramSpeculator::new(prompt_tokens.to_vec(), max_ngram, max_draft);
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut all_tokens = prompt_tokens.to_vec();
    let mut position = prompt_tokens.len();

    while generated.len() < max_new_tokens {
        let draft = speculator.speculate(&all_tokens);
        let mut rejected = false;

        for &candidate in &draft {
            if generated.len() == max_new_tokens {
                break;
            }

            let model_choice = argmax(&logits) as u32;
            if model_choice == eos_token {
                return Ok(generated);
            }

            let emitted = if model_choice == candidate {
                candidate
            } else {
                rejected = true;
                model_choice
            };
            generated.push(emitted);
            all_tokens.push(emitted);
            logits = forward_fn(emitted, position)?;
            position += 1;

            if rejected {
                break;
            }
        }

        if generated.len() == max_new_tokens || rejected {
            continue;
        }

        let bonus = argmax(&logits) as u32;
        if bonus == eos_token {
            break;
        }
        generated.push(bonus);
        all_tokens.push(bonus);
        logits = forward_fn(bonus, position)?;
        position += 1;
    }

    Ok(generated)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
