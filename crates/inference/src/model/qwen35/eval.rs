//! Language-model evaluation primitives for Qwen3.5: per-token NLLs and
//! strided sliding-window perplexity.
//!
//! Step 4a of [ADR-044](../../../../../docs/adr/ADR-044-quarot-rotated-quantization.md).
//! The CPU forward path here is the baseline that step 4b will compare against
//! Q4 / QuaRot-Q4 measurements (on the Metal runtime).
//!
//! The strided methodology matches the [HuggingFace fixed-length-model perplexity
//! recipe](https://huggingface.co/docs/transformers/en/perplexity): each global
//! token is scored exactly once with up to `window - stride` tokens of preceding
//! context, eliminating the cold-start bias of non-overlapping chunks.

use super::cache::{ForwardScratch, KvCache};
use super::model::Qwen35Model;
use crate::attention::gdn::GatedDeltaNetState;
use crate::error::InferenceError;

/// Configuration for [`Qwen35Model::compute_perplexity`].
#[derive(Clone, Copy, Debug)]
pub struct PerplexityConfig {
    /// Context window length in tokens. Each window scores at most
    /// `window - 1` positions (the first token has no predecessor).
    pub window: usize,
    /// Tokens advanced between successive windows. Each global token is
    /// scored under the first window whose context covers it.
    pub stride: usize,
}

impl Default for PerplexityConfig {
    fn default() -> Self {
        Self {
            window: 512,
            stride: 256,
        }
    }
}

/// Result of [`Qwen35Model::compute_perplexity`].
#[derive(Clone, Debug)]
pub struct PerplexityReport {
    /// Perplexity: `exp(mean_nll)`. Lower is better.
    pub ppl: f64,
    /// Mean cross-entropy in nats per scored token: `total_nll / num_tokens_scored`.
    pub mean_nll: f64,
    /// Sum of NLLs across every scored token, in nats.
    pub total_nll: f64,
    /// Tokens that contributed to the average. With `stride < window` and
    /// `tokens.len() <= window` this equals `tokens.len() - 1`; otherwise
    /// it is the same value, since each non-first global token is scored
    /// exactly once.
    pub num_tokens_scored: usize,
    /// Number of windows the harness ran over the corpus.
    pub num_windows: usize,
    /// Window size passed in [`PerplexityConfig`], echoed for the report.
    pub window: usize,
    /// Stride passed in [`PerplexityConfig`], echoed for the report.
    pub stride: usize,
}

impl Qwen35Model {
    /// **Unstable**: compute per-position cross-entropy NLLs for an autoregressive
    /// forward pass over `tokens`.
    ///
    /// Returns a `Vec<f32>` of length `tokens.len() - 1`. The value at index `i`
    /// is `-log p(tokens[i + 1] | tokens[0..=i])` as computed by the model, where
    /// `p` is the softmax over the full vocabulary at decode step `i`.
    ///
    /// The first token has no predecessor to score against, so it is not
    /// represented in the output. The KV cache and GDN recurrent state are
    /// constructed fresh inside this function; the caller's state (if any) is
    /// not touched.
    ///
    /// Errors if `tokens.len() < 2`, if any `token_id >= cfg.vocab_size`, or
    /// if `tokens.len() > self.max_context()` (the precomputed RoPE table
    /// only covers positions `0..max_context`; the underlying `forward_step`
    /// would otherwise panic on out-of-range RoPE indexing).
    pub fn compute_token_nlls(&self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError> {
        let cfg = &self.config;
        if tokens.len() < 2 {
            return Err(InferenceError::Inference(format!(
                "compute_token_nlls: need at least 2 tokens, got {}",
                tokens.len()
            )));
        }
        let max_context = self.max_context();
        if tokens.len() > max_context {
            return Err(InferenceError::Inference(format!(
                "compute_token_nlls: tokens.len() ({}) exceeds RoPE capacity ({}); \
                 use a shorter window or load the model with a larger context table",
                tokens.len(),
                max_context
            )));
        }
        if let Some((bad_idx, &bad)) = tokens
            .iter()
            .enumerate()
            .find(|&(_, &t)| (t as usize) >= cfg.vocab_size)
        {
            return Err(InferenceError::Inference(format!(
                "compute_token_nlls: tokens[{bad_idx}]={bad} >= vocab_size {}",
                cfg.vocab_size
            )));
        }

        let num_linear = cfg.num_linear_attention_layers();
        let num_full = cfg.num_full_attention_layers();
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
            .map(|_| GatedDeltaNetState::new(cfg))
            .collect();
        let mut kv_cache = KvCache::new(num_full);
        let mut scratch = ForwardScratch::new();

        let mut nlls = Vec::with_capacity(tokens.len() - 1);

        for (pos, &token_id) in tokens.iter().enumerate() {
            self.forward_step(token_id, pos, &mut gdn_states, &mut kv_cache, &mut scratch);

            if pos + 1 < tokens.len() {
                let next = tokens[pos + 1] as usize;
                let nll = log_softmax_nll(&scratch.logits[..cfg.vocab_size], next);
                nlls.push(nll);
            }

            if pos < tokens.len() - 1 {
                kv_cache.seq_len += 1;
            }
        }

        Ok(nlls)
    }

    /// **Unstable**: compute strided sliding-window perplexity over `tokens`.
    ///
    /// Mirrors the HuggingFace fixed-length-model recipe: walk the corpus in
    /// overlapping windows of `cfg.window` tokens, advancing `cfg.stride` tokens
    /// per step, and aggregate NLLs across windows so each global token is
    /// scored under the first window whose context covers it. Perplexity is
    /// `exp(total_nll / num_tokens_scored)`.
    ///
    /// Each window resets the KV cache + GDN state, so context never crosses a
    /// window boundary — `window - stride` tokens of context precede each
    /// scored token (zero context for the first `stride` tokens of the corpus).
    ///
    /// `stride` must be strictly less than `window`. Equal values would
    /// produce disjoint windows whose first token (the global boundary token)
    /// has no predecessor inside its own window and would therefore go
    /// unscored — silently breaking the "scored exactly once" invariant.
    /// Callers that want maximally-disjoint windows should pass
    /// `stride = window - 1`.
    pub fn compute_perplexity(
        &self,
        tokens: &[u32],
        cfg: &PerplexityConfig,
    ) -> Result<PerplexityReport, InferenceError> {
        if cfg.window < 2 {
            return Err(InferenceError::Inference(format!(
                "compute_perplexity: window ({}) must be >= 2",
                cfg.window
            )));
        }
        if cfg.stride == 0 {
            return Err(InferenceError::Inference(
                "compute_perplexity: stride must be > 0".into(),
            ));
        }
        if cfg.stride >= cfg.window {
            return Err(InferenceError::Inference(format!(
                "compute_perplexity: stride ({}) must be < window ({}); \
                 stride == window silently drops every window-boundary token",
                cfg.stride, cfg.window
            )));
        }
        let max_context = self.max_context();
        if cfg.window > max_context {
            return Err(InferenceError::Inference(format!(
                "compute_perplexity: window ({}) exceeds RoPE capacity ({}); \
                 use a shorter window or load the model with a larger context table",
                cfg.window, max_context
            )));
        }
        if tokens.len() < 2 {
            return Err(InferenceError::Inference(format!(
                "compute_perplexity: need at least 2 tokens, got {}",
                tokens.len()
            )));
        }

        let n = tokens.len();
        let mut total_nll: f64 = 0.0;
        let mut num_scored: usize = 0;
        let mut num_windows: usize = 0;
        // Highest global target index already scored (0 = nothing scored yet;
        // valid token indices are 1..n-1 inclusive, since target 0 is the
        // unscored first token).
        let mut last_scored: usize = 0;
        let mut begin: usize = 0;

        loop {
            let end = (begin + cfg.window).min(n);
            if end - begin < 2 {
                break;
            }

            let nlls = self.compute_token_nlls(&tokens[begin..end])?;
            for (i, &nll) in nlls.iter().enumerate() {
                let global_target = begin + i + 1;
                if global_target > last_scored {
                    total_nll += nll as f64;
                    num_scored += 1;
                }
            }
            // Last NLL in this window predicts global token (end - 1).
            last_scored = end - 1;
            num_windows += 1;

            if end >= n {
                break;
            }
            begin += cfg.stride;
        }

        if num_scored == 0 {
            return Err(InferenceError::Inference(
                "compute_perplexity: scored zero tokens (corpus shorter than 2?)".into(),
            ));
        }

        let mean_nll = total_nll / (num_scored as f64);
        let ppl = mean_nll.exp();

        Ok(PerplexityReport {
            ppl,
            mean_nll,
            total_nll,
            num_tokens_scored: num_scored,
            num_windows,
            window: cfg.window,
            stride: cfg.stride,
        })
    }
}

/// Numerically stable `-log softmax(logits)[target]`.
///
/// Computed as `log_sum_exp(logits) - logits[target]` with the standard
/// max-subtraction trick, so the running sum stays in `[0, vocab_size]`
/// even for logits with large magnitudes.
fn log_softmax_nll(logits: &[f32], target: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp: f64 = 0.0;
    for &l in logits {
        sum_exp += ((l - max) as f64).exp();
    }
    let log_sum_exp = (max as f64) + sum_exp.ln();
    (log_sum_exp - logits[target] as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::gdn::GatedDeltaNetWeights;
    use crate::lora_hook::NoopLoraHook;
    use crate::model::qwen35::ModelWeights;
    use crate::model::qwen35_config::{LayerType, Qwen35Config, compute_layer_types};
    use crate::rope::RopeTable;
    use crate::tokenizer::bpe::BpeTokenizer;

    /// Deterministic xorshift RNG → uniform noise in `[-scale, scale]`.
    fn rand_vec(state: &mut u64, len: usize, scale: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            out.push(((x >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale);
        }
        out
    }

    /// Minimal byte-level BPE tokenizer; `compute_token_nlls` never touches it,
    /// but `Qwen35Model` requires a non-trivial tokenizer field.
    fn test_tokenizer() -> BpeTokenizer {
        let json = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true },
  "post_processor": null,
  "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "<unk>",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": { "<unk>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, " ": 6 },
    "merges": []
  }
}"#;
        BpeTokenizer::from_tokenizer_json_str(json).expect("eval test tokenizer parses")
    }

    fn test_config() -> Qwen35Config {
        let num_hidden_layers = 4;
        let full_attention_interval = 4;
        Qwen35Config {
            hidden_size: 64,
            num_hidden_layers,
            vocab_size: 97,
            intermediate_size: 128,
            rms_norm_eps: 1e-6,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            rope_parameters: None,
            linear_num_key_heads: 4,
            linear_num_value_heads: Some(4),
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            output_router_logits: false,
            router_aux_loss_coef: None,
            tie_word_embeddings: true,
            full_attention_interval,
            layer_types: compute_layer_types(num_hidden_layers, full_attention_interval),
            layer_mask: vec![true; num_hidden_layers],
            eos_token_id: 96,
            max_position_embeddings: 1024,
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
        }
    }

    fn build_model(cfg: Qwen35Config, seed: u64) -> Qwen35Model {
        use crate::model::qwen35::{
            AttentionWeights, CommonLayerWeights, DenseFfnWeights, FeedForwardWeights,
            FullAttentionLayerWeights,
        };

        let mut rng = seed | 1;
        let h = cfg.hidden_size;

        let embed_tokens = rand_vec(&mut rng, cfg.vocab_size * h, 0.02);
        let final_norm = rand_vec(&mut rng, h, 0.02);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_type in &cfg.layer_types {
            let common = CommonLayerWeights {
                input_layernorm: rand_vec(&mut rng, h, 0.02),
                post_attention_layernorm: rand_vec(&mut rng, h, 0.02),
                ffn: FeedForwardWeights::Dense(DenseFfnWeights {
                    gate_proj: rand_vec(&mut rng, cfg.intermediate_size * h, 0.02),
                    up_proj: rand_vec(&mut rng, cfg.intermediate_size * h, 0.02),
                    down_proj: rand_vec(&mut rng, h * cfg.intermediate_size, 0.02),
                }),
            };
            let attn = match layer_type {
                LayerType::LinearAttention => {
                    let qkv_dim = cfg.linear_qkv_dim();
                    let output_dim = cfg.linear_output_dim();
                    let nh = cfg.linear_num_key_heads;
                    let kernel = cfg.linear_conv_kernel_dim;
                    AttentionWeights::Linear(GatedDeltaNetWeights {
                        in_proj_qkv: rand_vec(&mut rng, qkv_dim * h, 0.02),
                        in_proj_qkv_rows: qkv_dim,
                        in_proj_qkv_cols: h,
                        in_proj_z: rand_vec(&mut rng, output_dim * h, 0.02),
                        in_proj_z_rows: output_dim,
                        in_proj_z_cols: h,
                        in_proj_b: rand_vec(&mut rng, nh * h, 0.02),
                        in_proj_b_rows: nh,
                        in_proj_b_cols: h,
                        in_proj_a: rand_vec(&mut rng, nh * h, 0.02),
                        in_proj_a_rows: nh,
                        in_proj_a_cols: h,
                        a_log: rand_vec(&mut rng, nh, 0.02),
                        dt_bias: rand_vec(&mut rng, nh, 0.02),
                        conv1d_weight: rand_vec(&mut rng, qkv_dim * kernel, 0.02),
                        conv_dim: qkv_dim,
                        kernel_size: kernel,
                        norm_weight: rand_vec(&mut rng, output_dim, 0.02),
                        out_proj: rand_vec(&mut rng, h * output_dim, 0.02),
                        out_proj_rows: h,
                        out_proj_cols: output_dim,
                    })
                }
                LayerType::FullAttention => {
                    let q_dim = cfg.full_q_dim();
                    let kv_dim = cfg.full_kv_dim();
                    AttentionWeights::Full(FullAttentionLayerWeights {
                        q_proj: rand_vec(&mut rng, 2 * q_dim * h, 0.02),
                        k_proj: rand_vec(&mut rng, kv_dim * h, 0.02),
                        v_proj: rand_vec(&mut rng, kv_dim * h, 0.02),
                        o_proj: rand_vec(&mut rng, h * q_dim, 0.02),
                        q_norm: rand_vec(&mut rng, cfg.head_dim, 0.02),
                        k_norm: rand_vec(&mut rng, cfg.head_dim, 0.02),
                    })
                }
            };
            layers.push((attn, common));
        }

        let rope = RopeTable::new(
            cfg.rope_dim(),
            cfg.max_position_embeddings.min(8192),
            cfg.rope_theta,
        );

        Qwen35Model {
            config: cfg,
            weights: ModelWeights {
                embed_tokens,
                lm_head: None,
                final_norm,
                layers,
            },
            tokenizer: test_tokenizer(),
            rope,
            lora: Box::new(NoopLoraHook),
        }
    }

    #[test]
    fn log_softmax_nll_uniform_logits_is_log_vocab() {
        let logits = vec![0.5f32; 32];
        let nll = log_softmax_nll(&logits, 0);
        let expected = (32.0f64).ln() as f32;
        assert!(
            (nll - expected).abs() < 1e-5,
            "uniform logits NLL = log(vocab_size); got {nll}, expected {expected}"
        );
    }

    #[test]
    fn log_softmax_nll_one_hot_target_is_near_zero() {
        let mut logits = vec![-100.0f32; 16];
        logits[3] = 100.0;
        let nll = log_softmax_nll(&logits, 3);
        assert!(nll.abs() < 1e-3, "one-hot target NLL ~ 0; got {nll}");
    }

    #[test]
    fn log_softmax_nll_one_hot_wrong_target_is_large() {
        let mut logits = vec![-100.0f32; 16];
        logits[3] = 100.0;
        let nll = log_softmax_nll(&logits, 7);
        assert!(nll > 100.0, "wrong one-hot target NLL >> 0; got {nll}");
    }

    #[test]
    fn log_softmax_nll_handles_large_magnitudes_without_overflow() {
        let mut logits = vec![1.0e6_f32; 8];
        logits[4] = 1.0e6 + 2.0;
        let nll = log_softmax_nll(&logits, 4);
        // Max-subtraction puts the target at 0 and the other 7 logits at -2,
        // so NLL = log(7 * e^-2 + 1) - 0 ≈ 0.6665 (independent of the 1e6
        // offset; naïve unstable code would compute exp(1e6) -> +inf).
        let expected: f64 = (7.0_f64 * (-2.0_f64).exp() + 1.0).ln();
        assert!(
            ((nll as f64) - expected).abs() < 1e-3,
            "shifted-uniform NLL = log(7*e^-2 + 1); got {nll}, expected {expected}"
        );
    }

    #[test]
    fn compute_token_nlls_returns_one_less_than_input_length() {
        let cfg = test_config();
        let model = build_model(cfg, 0xCAFE_F00D);
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let nlls = model
            .compute_token_nlls(&tokens)
            .expect("compute_token_nlls succeeds on valid tokens");
        assert_eq!(
            nlls.len(),
            tokens.len() - 1,
            "NLL output length = tokens.len() - 1 (one prediction per non-first token)"
        );
    }

    #[test]
    fn compute_token_nlls_all_values_are_finite_and_non_negative() {
        let cfg = test_config();
        let model = build_model(cfg, 0xBAD_C0DE);
        let tokens: Vec<u32> = (1u32..=8).collect();
        let nlls = model.compute_token_nlls(&tokens).expect("nlls ok");
        for (i, &nll) in nlls.iter().enumerate() {
            assert!(
                nll.is_finite(),
                "nlls[{i}] = {nll} must be finite (no overflow/underflow)"
            );
            assert!(
                nll >= -1e-3,
                "nlls[{i}] = {nll} must be non-negative (cross-entropy of a probability is ≥ 0)"
            );
        }
    }

    #[test]
    fn compute_token_nlls_is_deterministic() {
        let cfg = test_config();
        let model = build_model(cfg, 0xDEAD_BEEF);
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7];
        let a = model.compute_token_nlls(&tokens).unwrap();
        let b = model.compute_token_nlls(&tokens).unwrap();
        assert_eq!(
            a, b,
            "compute_token_nlls must produce bit-identical output across calls"
        );
    }

    #[test]
    fn compute_token_nlls_rejects_too_short_input() {
        let cfg = test_config();
        let model = build_model(cfg, 0xFACE_FEED);
        assert!(model.compute_token_nlls(&[]).is_err());
        assert!(model.compute_token_nlls(&[1]).is_err());
        assert!(model.compute_token_nlls(&[1, 2]).is_ok());
    }

    #[test]
    fn compute_token_nlls_rejects_out_of_vocab_token() {
        let cfg = test_config();
        let vocab = cfg.vocab_size as u32;
        let model = build_model(cfg, 0xC001_BEAD);
        let err = model
            .compute_token_nlls(&[1, vocab, 3])
            .expect_err("out-of-vocab token must error");
        let msg = format!("{err}");
        assert!(
            msg.contains(&format!("{vocab}")),
            "error must name the bad token id; got: {msg}"
        );
    }

    #[test]
    fn compute_perplexity_equals_exp_mean_nll() {
        let cfg = test_config();
        let model = build_model(cfg, 0xC0DE_FACE);
        let tokens: Vec<u32> = (1u32..=20).collect();
        let report = model
            .compute_perplexity(
                &tokens,
                &PerplexityConfig {
                    window: 10,
                    stride: 5,
                },
            )
            .expect("ppl ok");
        let expected_ppl = report.mean_nll.exp();
        assert!(
            (report.ppl - expected_ppl).abs() < 1e-9,
            "ppl must equal exp(mean_nll); got ppl={}, exp(mean_nll)={}",
            report.ppl,
            expected_ppl
        );
        assert!(
            (report.mean_nll - report.total_nll / report.num_tokens_scored as f64).abs() < 1e-9,
            "mean_nll must equal total_nll / num_tokens_scored"
        );
    }

    #[test]
    fn compute_perplexity_scores_each_token_exactly_once_under_stride() {
        let cfg = test_config();
        let model = build_model(cfg, 0xBEEF_CAFE);
        // 20 tokens, window=10, stride=5
        // → windows starting at [0, 5, 10] (third one ends at 20, covers tail)
        //   window 0: tokens[0..10],  scores positions 1..9 globally  (9 tokens)
        //   window 1: tokens[5..15],  scores positions 10..14 globally (5 tokens)
        //   window 2: tokens[10..20], scores positions 15..19 globally (5 tokens)
        // Total: 19 = tokens.len() - 1 (every non-first token scored once)
        let tokens: Vec<u32> = (1u32..=20).collect();
        let report = model
            .compute_perplexity(
                &tokens,
                &PerplexityConfig {
                    window: 10,
                    stride: 5,
                },
            )
            .unwrap();
        assert_eq!(
            report.num_tokens_scored,
            tokens.len() - 1,
            "each non-first token must be scored exactly once under non-overlapping stride windows"
        );
        assert_eq!(report.num_windows, 3);
    }

    #[test]
    fn compute_perplexity_single_window_covers_short_corpus() {
        let cfg = test_config();
        let model = build_model(cfg, 0xFEED_BEEF);
        let tokens: Vec<u32> = (1u32..=8).collect();
        // Window larger than corpus → single window scores all (N-1) tokens.
        let report = model
            .compute_perplexity(
                &tokens,
                &PerplexityConfig {
                    window: 16,
                    stride: 8,
                },
            )
            .unwrap();
        assert_eq!(report.num_windows, 1);
        assert_eq!(report.num_tokens_scored, 7);
    }

    #[test]
    fn compute_perplexity_validates_config() {
        let cfg = test_config();
        let model = build_model(cfg, 0xC0DE_C0DE);
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        assert!(
            model
                .compute_perplexity(
                    &tokens,
                    &PerplexityConfig {
                        window: 1,
                        stride: 1,
                    }
                )
                .is_err(),
            "window < 2 must error"
        );
        assert!(
            model
                .compute_perplexity(
                    &tokens,
                    &PerplexityConfig {
                        window: 4,
                        stride: 0,
                    }
                )
                .is_err(),
            "stride == 0 must error"
        );
        assert!(
            model
                .compute_perplexity(
                    &tokens,
                    &PerplexityConfig {
                        window: 4,
                        stride: 8,
                    }
                )
                .is_err(),
            "stride > window must error"
        );
        assert!(
            model
                .compute_perplexity(
                    &tokens,
                    &PerplexityConfig {
                        window: 4,
                        stride: 4,
                    }
                )
                .is_err(),
            "stride == window must error (would silently drop boundary tokens)"
        );
        assert!(
            model
                .compute_perplexity(
                    &[1],
                    &PerplexityConfig {
                        window: 4,
                        stride: 2,
                    }
                )
                .is_err(),
            "corpus < 2 tokens must error"
        );
    }

    #[test]
    fn compute_perplexity_matches_compute_token_nlls_on_single_window() {
        let cfg = test_config();
        let model = build_model(cfg, 0xA110_CA7E);
        let tokens: Vec<u32> = (1u32..=12).collect();
        // window covers the whole corpus → strided harness collapses to a
        // single window and the aggregated NLL must equal the direct sum.
        let nlls = model.compute_token_nlls(&tokens).unwrap();
        let direct_total: f64 = nlls.iter().map(|&x| x as f64).sum();
        let report = model
            .compute_perplexity(
                &tokens,
                &PerplexityConfig {
                    window: tokens.len(),
                    stride: tokens.len() - 1,
                },
            )
            .unwrap();
        assert_eq!(report.num_windows, 1);
        assert_eq!(report.num_tokens_scored, tokens.len() - 1);
        assert!(
            (report.total_nll - direct_total).abs() < 1e-9,
            "single-window strided PPL must agree with compute_token_nlls sum: got {} vs {}",
            report.total_nll,
            direct_total
        );
    }

    #[test]
    fn compute_perplexity_rejects_stride_equal_window() {
        // Round-1 codex finding: stride == window silently dropped every
        // window-boundary token (the first token of each disjoint window has
        // no predecessor inside its own window). The harness now errors
        // before that can happen.
        let cfg = test_config();
        let model = build_model(cfg, 0xDADA_F00D);
        let tokens: Vec<u32> = (1u32..=20).collect();
        let err = model
            .compute_perplexity(
                &tokens,
                &PerplexityConfig {
                    window: 10,
                    stride: 10,
                },
            )
            .expect_err("stride == window must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("stride") && msg.contains("window"),
            "error must name both `stride` and `window`; got: {msg}"
        );
    }

    #[test]
    fn compute_perplexity_rejects_window_above_rope_capacity() {
        // Round-1 codex finding: the public Result-returning API was able
        // to panic by indexing the precomputed RoPE table past its end.
        // The harness now validates window <= max_context up-front.
        let cfg = test_config();
        let max_context = {
            let model = build_model(cfg.clone(), 0xC0FF_EE42);
            model.max_context()
        };
        let model = build_model(cfg, 0xC0FF_EE42);
        let tokens: Vec<u32> = (1..=8).cycle().take(max_context + 4).collect();
        let err = model
            .compute_perplexity(
                &tokens,
                &PerplexityConfig {
                    window: max_context + 1,
                    stride: 8,
                },
            )
            .expect_err("window > max_context must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("RoPE") && msg.contains(&format!("{max_context}")),
            "error must mention the RoPE capacity; got: {msg}"
        );
    }

    #[test]
    fn compute_token_nlls_rejects_tokens_above_rope_capacity() {
        let cfg = test_config();
        let max_context = {
            let model = build_model(cfg.clone(), 0xC0FF_EE43);
            model.max_context()
        };
        let model = build_model(cfg, 0xC0FF_EE43);
        let tokens: Vec<u32> = (1..=8).cycle().take(max_context + 1).collect();
        let err = model
            .compute_token_nlls(&tokens)
            .expect_err("tokens.len() > max_context must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("RoPE") && msg.contains(&format!("{max_context}")),
            "error must mention the RoPE capacity; got: {msg}"
        );
    }
}
