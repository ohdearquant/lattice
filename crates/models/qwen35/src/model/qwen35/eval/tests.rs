use super::*;
use crate::attention::gdn::GatedDeltaNetWeights;
use crate::attention::gdn::require_qwen35_gdn_geometry;
use crate::model::qwen35::ModelWeights;
use lattice_arch::qwen35_config::{LayerType, Qwen35Config, compute_layer_types};
use lattice_compute::lora_hook::NoopLoraHook;
use lattice_tokenizer::bpe::BpeTokenizer;

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
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
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
                gate_proj: rand_vec(&mut rng, cfg.intermediate_size * h, 0.02).into(),
                up_proj: rand_vec(&mut rng, cfg.intermediate_size * h, 0.02).into(),
                down_proj: rand_vec(&mut rng, h * cfg.intermediate_size, 0.02).into(),
            }),
        };
        let attn = match layer_type {
            LayerType::LinearAttention => {
                let qkv_dim = cfg.linear_qkv_dim();
                let output_dim = cfg.linear_output_dim();
                let nh = cfg.linear_num_key_heads;
                let kernel = cfg.linear_conv_kernel_dim;
                AttentionWeights::Linear(GatedDeltaNetWeights {
                    in_proj_qkv: rand_vec(&mut rng, qkv_dim * h, 0.02).into(),
                    in_proj_qkv_rows: qkv_dim,
                    in_proj_qkv_cols: h,
                    in_proj_z: rand_vec(&mut rng, output_dim * h, 0.02).into(),
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
                    out_proj: rand_vec(&mut rng, h * output_dim, 0.02).into(),
                    out_proj_rows: h,
                    out_proj_cols: output_dim,
                })
            }
            LayerType::FullAttention => {
                let q_dim = cfg.full_q_dim();
                let kv_dim = cfg.full_kv_dim();
                AttentionWeights::Full(FullAttentionLayerWeights {
                    q_proj: rand_vec(&mut rng, 2 * q_dim * h, 0.02).into(),
                    k_proj: rand_vec(&mut rng, kv_dim * h, 0.02).into(),
                    v_proj: rand_vec(&mut rng, kv_dim * h, 0.02).into(),
                    o_proj: rand_vec(&mut rng, h * q_dim, 0.02).into(),
                    q_norm: rand_vec(&mut rng, cfg.head_dim, 0.02),
                    k_norm: rand_vec(&mut rng, cfg.head_dim, 0.02),
                })
            }
        };
        layers.push((attn, common));
    }

    let rope = crate::model::qwen35::cpu_rope_table(&cfg);

    Qwen35Model {
        config: cfg,
        weights: ModelWeights {
            embed_tokens: embed_tokens.into(),
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
    //   window 0: tokens[0..10],  scores global targets 1..9   (9 tokens; local
    //     prior counts ramp 1..9)
    //   window 1: tokens[5..15],  scores global targets 10..14 (5 tokens; local
    //     prior counts run 5..9 → ≥ window - stride and ≤ window - 1)
    //   window 2: tokens[10..20], scores global targets 15..19 (5 tokens; local
    //     prior counts run 5..9)
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
    // Finding: stride == window silently dropped every
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
    // Finding: the public Result-returning API was able
    // to panic by indexing the precomputed RoPE table past its end.
    // The harness now validates window <= max_context up-front. After
    // the step-4b refactor the message comes from the shared aggregator
    // and uses the generic "backend context capacity" wording so it
    // covers the Metal KV-cache case too.
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
        msg.contains("context capacity") && msg.contains(&format!("{max_context}")),
        "error must mention the backend context capacity and the bound; got: {msg}"
    );
}

#[test]
fn compute_perplexity_accepts_long_window_on_short_corpus() {
    // Finding: the long-window guard was rejecting safe
    // short-corpus invocations. The effective slice length is
    // `min(window, tokens.len())`, so an oversized window with a small
    // corpus must succeed — the per-window clip keeps every slice
    // within the RoPE table.
    let cfg = test_config();
    let max_context = {
        let model = build_model(cfg.clone(), 0xC0FF_EE44);
        model.max_context()
    };
    let model = build_model(cfg, 0xC0FF_EE44);
    let tokens: Vec<u32> = (1u32..=8).collect();
    assert!(tokens.len() < max_context, "guard assumption");
    let report = model
        .compute_perplexity(
            &tokens,
            &PerplexityConfig {
                window: max_context + 1_000,
                stride: 8,
            },
        )
        .expect("short corpus with oversized window must succeed");
    assert_eq!(report.num_windows, 1);
    assert_eq!(report.num_tokens_scored, tokens.len() - 1);
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

// ── Issue #403 preflight regression tests ──────────────────────────────────
//
// Mutation contract: each test below MUST FAIL when its corresponding guard is
// removed (i.e. when the `if prompt_ids.len() > max_context` / `if seq_len >
// max_context` branch is deleted). Reverting that guard leaves the underlying
// `forward_step` / `cos_at` / `sin_at` calls able to index the RoPE table out
// of range, which panics in release — the test would then panic instead of
// asserting `is_err()`.

#[test]
fn forward_prompt_debug_rejects_over_capacity_prompt() {
    // Mutation contract: removing the `if prompt_ids.len() > max_context` guard
    // in debug.rs causes this test to panic (out-of-bounds RoPE index) instead
    // of asserting the Err variant. The panic changes the test result from PASS
    // to FAIL (with a panic message, not an assertion failure).
    let cfg = test_config();
    let max_context = {
        let model = build_model(cfg.clone(), 0xD3B0_6403);
        model.max_context()
    };
    let model = build_model(cfg, 0xD3B0_6403);
    // One token past the RoPE table boundary.
    let prompt_ids: Vec<u32> = (1u32..=4).cycle().take(max_context + 1).collect();
    let err = model
        .forward_prompt_debug(&prompt_ids)
        .expect_err("prompt_ids.len() > max_context must return Err, not panic");
    let msg = format!("{err}");
    assert!(
        msg.contains("RoPE table capacity") && msg.contains(&format!("{max_context}")),
        "error must name the RoPE table capacity limit; got: {msg}"
    );
}

#[test]
fn forward_prompt_debug_rejects_empty_prompt() {
    let cfg = test_config();
    let model = build_model(cfg, 0xE850_0403);
    let err = model
        .forward_prompt_debug(&[])
        .expect_err("empty prompt_ids must return Err");
    let msg = format!("{err}");
    assert!(
        msg.contains("empty"),
        "error message must mention empty; got: {msg}"
    );
}

#[cfg(feature = "train-backward")]
#[test]
fn rope_cos_sin_tables_rejects_over_capacity_seq_len() {
    // Mutation contract: removing the `if seq_len > max_context` guard in the
    // `rope_cos_sin_tables` body causes this test to panic (out-of-bounds slice
    // index inside the `cos_at`/`sin_at` loop) instead of asserting the Err
    // variant. The panic changes the test result from PASS to FAIL.
    let cfg = test_config();
    let max_context = {
        let model = build_model(cfg.clone(), 0xC0FE_0403);
        model.max_context()
    };
    let model = build_model(cfg, 0xC0FE_0403);
    let err = model
        .rope_cos_sin_tables(max_context + 1)
        .expect_err("seq_len > max_context must return Err, not panic");
    let msg = format!("{err}");
    assert!(
        msg.contains("RoPE table capacity") && msg.contains(&format!("{max_context}")),
        "error must name the RoPE table capacity limit; got: {msg}"
    );
}

#[cfg(feature = "train-backward")]
#[test]
fn rope_cos_sin_tables_accepts_exact_capacity() {
    // Exactly at the boundary must succeed (seq_len == max_context is valid).
    let cfg = test_config();
    let max_context = {
        let model = build_model(cfg.clone(), 0xC0FE_1403);
        model.max_context()
    };
    let model = build_model(cfg, 0xC0FE_1403);
    let result = model.rope_cos_sin_tables(max_context);
    assert!(result.is_ok(), "seq_len == max_context must succeed");
    let (cos, sin) = result.unwrap();
    let half = model.config.rope_dim() / 2;
    assert_eq!(cos.len(), max_context * half);
    assert_eq!(sin.len(), max_context * half);
}

/// Last-position logits read by hand: drive `forward_step` over the prompt and copy the
/// scratch logits, independent of `forward_prompt_debug` and of the scorer.
fn hand_read_last_logits(model: &Qwen35Model, prompt: &[u32]) -> Vec<f32> {
    let cfg = &model.config;
    let mut gdn_states: Vec<GatedDeltaNetState> = (0..cfg.num_linear_attention_layers())
        .map(|_| GatedDeltaNetState::from_geometry(&require_qwen35_gdn_geometry(cfg)))
        .collect();
    let mut kv_cache = KvCache::new(cfg.num_full_attention_layers());
    let mut scratch = ForwardScratch::new();
    for (pos, &token) in prompt.iter().enumerate() {
        model.forward_step(token, pos, &mut gdn_states, &mut kv_cache, &mut scratch);
        if pos + 1 < prompt.len() {
            kv_cache.seq_len += 1;
        }
    }
    scratch.logits[..cfg.vocab_size].to_vec()
}

#[test]
fn score_option_letters_logits_equal_the_hand_read_last_position_bitwise() {
    let model = build_model(test_config(), 0x0A71_0C01);
    let prompt: Vec<u32> = vec![11, 42, 7, 7, 63, 5, 20];
    let letters: Vec<u32> = vec![33, 8, 90, 1];

    let scores = model
        .score_option_letters(&prompt, &letters)
        .expect("valid prompt and letters");

    let last = hand_read_last_logits(&model, &prompt);
    let expected: Vec<u32> = letters
        .iter()
        .map(|&i| last[i as usize].to_bits())
        .collect();
    let actual: Vec<u32> = scores.logits.iter().map(|l| l.to_bits()).collect();
    assert_eq!(
        actual, expected,
        "scorer logits are the last-position logits at the letters"
    );

    // The fixture must be able to tell the last position from an earlier one: the same read
    // at position 0 gives different values, and the letters do not all tie.
    let first = hand_read_last_logits(&model, &prompt[..1]);
    let at_first: Vec<u32> = letters
        .iter()
        .map(|&i| first[i as usize].to_bits())
        .collect();
    assert_ne!(
        at_first, expected,
        "position 0 must differ from the last position"
    );
    assert!(
        scores
            .logits
            .windows(2)
            .any(|w| w[0].to_bits() != w[1].to_bits()),
        "fixture letters must not all tie"
    );
}

#[test]
fn score_option_letters_summary_matches_the_full_row_and_repeats_bit_identically() {
    let model = build_model(test_config(), 0x0A71_0C02);
    let prompt: Vec<u32> = vec![3, 14, 15, 92, 65];
    let letters: Vec<u32> = vec![10, 20, 30];

    let first = model
        .score_option_letters(&prompt, &letters)
        .expect("scores");
    let second = model
        .score_option_letters(&prompt, &letters)
        .expect("scores");
    assert_eq!(
        first.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        second
            .logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        first.probs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        second.probs.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(first.label_mass.to_bits(), second.label_mass.to_bits());

    let sum: f32 = first.probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "probs sum to {sum}");
    assert!((0.0..=1.0).contains(&first.label_mass));

    // label_mass is the mass of the full softmax over the hand-read row.
    let row = hand_read_last_logits(&model, &prompt);
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f64 = row.iter().map(|&l| f64::from(l - max).exp()).sum();
    let on_letters: f64 = letters
        .iter()
        .map(|&i| f64::from(row[i as usize] - max).exp())
        .sum();
    assert!((f64::from(first.label_mass) - on_letters / total).abs() < 1e-6);
}

#[test]
fn score_option_letters_depends_on_the_whole_prompt() {
    let model = build_model(test_config(), 0x0A71_0C03);
    let letters = [4_u32, 5, 6];
    let a = model
        .score_option_letters(&[9, 8, 7], &letters)
        .expect("scores");
    let b = model
        .score_option_letters(&[1, 8, 7], &letters)
        .expect("scores");
    assert_ne!(
        a.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "changing an earlier prompt token must change the last-position logits"
    );
}

#[test]
fn score_option_letters_refuses_bad_input_before_any_forward_pass() {
    let mut cfg = test_config();
    cfg.max_position_embeddings = 8;
    let model = build_model(cfg, 0x0A71_0C04);
    assert_eq!(model.max_context(), 8);
    let vocab = model.config.vocab_size as u32;

    let invalid = |result: Result<OptionScores, InferenceError>, needle: &str| match result {
        Err(InferenceError::InvalidInput(message)) => {
            assert!(message.contains(needle), "{message}");
        }
        other => panic!("expected InvalidInput containing {needle:?}, got {other:?}"),
    };

    invalid(
        model.score_option_letters(&[], &[1, 2]),
        "must not be empty",
    );
    invalid(
        model.score_option_letters(&[1, 2], &[]),
        "must not be empty",
    );
    invalid(
        model.score_option_letters(&[1, 2], &[1, vocab]),
        "out of range",
    );
    invalid(model.score_option_letters(&[1, 2], &[3, 4, 3]), "duplicate");
    // A prompt token outside the vocabulary must be an error, not the panic `forward_step` raises.
    invalid(
        model.score_option_letters(&[1, vocab], &[1, 2]),
        "prompt_ids[1]",
    );
    invalid(
        model.score_option_letters(&[1; 9], &[1, 2]),
        "exceeds RoPE capacity",
    );

    // The boundary itself is accepted.
    model
        .score_option_letters(&[1; 8], &[1, 2])
        .expect("a prompt of exactly max_context tokens is valid");
}

mod readout_tests;

fn serial_and_batched(
    model: &Qwen35Model,
    prompt: &[u32],
    letters: &[u32],
) -> (OptionScores, OptionScores) {
    let serial = model
        .score_option_letters(prompt, letters)
        .expect("serial scores");
    let batched = model
        .score_option_letters_batched(prompt, letters)
        .expect("batched scores");
    (serial, batched)
}

/// The largest absolute difference, or the first non-finite one, so a
/// tolerance check fails on a NaN instead of folding it away.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        let diff = (x - y).abs();
        if !diff.is_finite() {
            return diff;
        }
        max = max.max(diff);
    }
    max
}

#[test]
fn batched_option_scores_match_the_serial_scorer_across_prompt_lengths() {
    let model = build_model(test_config(), 0x0A71_0C11);
    let letters: Vec<u32> = vec![33, 8, 90, 1, 64];
    let mut rng = 0x5EED_0C11_u64;
    // 1 and 7 sit inside one attention tile; 150 and 300 cross several.
    for len in [1usize, 7, 150, 300] {
        let prompt: Vec<u32> = rand_vec(&mut rng, len, 1.0)
            .iter()
            .map(|v| ((v.abs() * 96.0) as u32).min(95))
            .collect();
        let (serial, batched) = serial_and_batched(&model, &prompt, &letters);
        let scale = serial
            .logits
            .iter()
            .fold(0.0_f32, |m, v| m.max(v.abs()))
            .max(1e-6);
        let logit_diff = max_abs_diff(&serial.logits, &batched.logits);
        assert!(
            logit_diff <= 1e-4 * scale,
            "len {len}: logits differ by {logit_diff} at scale {scale}"
        );
        assert!(
            max_abs_diff(&serial.probs, &batched.probs) <= 1e-5,
            "len {len}: probabilities differ"
        );
        assert!(
            serial
                .logits
                .windows(2)
                .any(|w| (w[0] - w[1]).abs() > 1e3 * logit_diff.max(f32::EPSILON * scale)),
            "len {len}: the fixture letters must be separated by far more than the path difference"
        );
    }
}

#[test]
fn batched_option_scores_depend_on_the_whole_prompt() {
    let model = build_model(test_config(), 0x0A71_0C12);
    let letters = [4_u32, 5, 6];
    let a = model
        .score_option_letters_batched(&[9, 8, 7], &letters)
        .expect("scores");
    let b = model
        .score_option_letters_batched(&[1, 8, 7], &letters)
        .expect("scores");
    assert_ne!(
        a.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "changing an earlier prompt token must change the last-position logits"
    );
}

#[test]
fn batched_option_scoring_falls_back_to_the_serial_scorer_for_moe() {
    let mut model = build_model(test_config(), 0x0A71_0C13);
    model.config.num_experts = Some(1);
    model.config.num_experts_per_tok = Some(1);
    assert!(model.config.is_moe());
    let (serial, batched) = serial_and_batched(&model, &[3, 14, 15, 92, 65], &[10, 20, 30]);
    assert_eq!(
        serial
            .logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        batched
            .logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "a config the batched prefill refuses takes the serial path"
    );
}

#[test]
fn batched_option_scoring_refuses_bad_input_like_the_serial_scorer() {
    let mut cfg = test_config();
    cfg.max_position_embeddings = 8;
    let model = build_model(cfg, 0x0A71_0C14);
    let vocab = model.config.vocab_size as u32;
    let cases: [(&[u32], &[u32]); 5] = [
        (&[], &[1, 2]),
        (&[1, 2], &[]),
        (&[1, 2], &[1, vocab]),
        (&[1, vocab], &[1, 2]),
        (&[1; 9], &[1, 2]),
    ];
    for (prompt, letters) in cases {
        let serial = model.score_option_letters(prompt, letters);
        let batched = model.score_option_letters_batched(prompt, letters);
        match (serial, batched) {
            (Err(InferenceError::InvalidInput(a)), Err(InferenceError::InvalidInput(b))) => {
                assert_eq!(a, b);
            }
            other => panic!("expected the same InvalidInput from both, got {other:?}"),
        }
    }
}

fn random_prompt(rng: &mut u64, len: usize) -> Vec<u32> {
    rand_vec(rng, len, 1.0)
        .iter()
        .map(|v| ((v.abs() * 96.0) as u32).min(95))
        .collect()
}

fn assert_scores_close(expected: &OptionScores, actual: &OptionScores, what: &str) {
    let scale = expected
        .logits
        .iter()
        .fold(0.0_f32, |m, v| m.max(v.abs()))
        .max(1e-6);
    let diff = max_abs_diff(&expected.logits, &actual.logits);
    assert!(
        diff <= 1e-4 * scale,
        "{what}: logits differ by {diff} at scale {scale}"
    );
    assert!(
        max_abs_diff(&expected.probs, &actual.probs) <= 1e-5,
        "{what}: probabilities differ"
    );
}

#[test]
fn shared_prefix_rows_match_scoring_each_row_on_its_own() {
    let model = build_model(test_config(), 0x0A71_0C21);
    let mut rng = 0x5EED_0C21_u64;
    // The shared state crosses two attention tiles; the suffixes are one
    // token, a few tokens, and longer than a tile.
    let state = random_prompt(&mut rng, 150);
    let suffixes = [
        random_prompt(&mut rng, 1),
        random_prompt(&mut rng, 5),
        random_prompt(&mut rng, 80),
    ];
    let letters: [&[u32]; 3] = [&[33, 8, 90], &[1, 2], &[64, 65, 66, 67]];
    let prompts: Vec<Vec<u32>> = suffixes
        .iter()
        .map(|suffix| [state.as_slice(), suffix.as_slice()].concat())
        .collect();
    let rows: Vec<(&[u32], &[u32])> = prompts
        .iter()
        .zip(letters)
        .map(|(p, l)| (p.as_slice(), l))
        .collect();
    assert!(shared_scoring_prefix_len(&rows) >= 150);
    let shared = model
        .score_option_rows_shared_prefix(&rows)
        .expect("scores");
    assert_eq!(shared.len(), rows.len());
    for (index, ((prompt_ids, letter_ids), scores)) in rows.iter().zip(&shared).enumerate() {
        let alone = model
            .score_option_letters_batched(prompt_ids, letter_ids)
            .expect("batched");
        assert_scores_close(&alone, scores, &format!("row {index} vs batched"));
        let serial = model
            .score_option_letters(prompt_ids, letter_ids)
            .expect("serial");
        assert_scores_close(&serial, scores, &format!("row {index} vs serial"));
    }
}

#[test]
fn rows_that_share_no_prefix_or_end_where_another_continues_are_scored() {
    let model = build_model(test_config(), 0x0A71_0C22);
    let disjoint: [(&[u32], &[u32]); 2] = [(&[1, 2, 3], &[4, 5]), (&[6, 7], &[4, 5])];
    let long: Vec<u32> = (0..40).map(|i| (i * 7 % 90) as u32).collect();
    let nested: [(&[u32], &[u32]); 2] = [(&long[..20], &[4, 5]), (&long, &[4, 5])];
    for rows in [&disjoint[..], &nested[..]] {
        let shared = model.score_option_rows_shared_prefix(rows).expect("scores");
        assert_eq!(shared.len(), rows.len());
        for ((prompt_ids, letter_ids), scores) in rows.iter().zip(&shared) {
            let alone = model
                .score_option_letters_batched(prompt_ids, letter_ids)
                .expect("batched");
            assert_scores_close(&alone, scores, "row");
        }
    }
}

#[test]
fn shared_prefix_length_leaves_every_row_a_token() {
    let none: [(&[u32], &[u32]); 0] = [];
    let one: [(&[u32], &[u32]); 1] = [(&[1, 2, 3], &[9])];
    let diverging: [(&[u32], &[u32]); 2] = [(&[1, 2, 3], &[9]), (&[1, 2, 4], &[9])];
    let nested: [(&[u32], &[u32]); 2] = [(&[1, 2], &[9]), (&[1, 2, 3], &[9])];
    let disjoint: [(&[u32], &[u32]); 2] = [(&[5, 2], &[9]), (&[1, 2], &[9])];
    assert_eq!(shared_scoring_prefix_len(&none), 0);
    assert_eq!(shared_scoring_prefix_len(&one), 2);
    assert_eq!(shared_scoring_prefix_len(&diverging), 2);
    assert_eq!(shared_scoring_prefix_len(&nested), 1);
    assert_eq!(shared_scoring_prefix_len(&disjoint), 0);
}

#[test]
fn shared_prefix_scoring_falls_back_to_the_serial_scorer_for_moe() {
    let mut model = build_model(test_config(), 0x0A71_0C23);
    model.config.num_experts = Some(1);
    model.config.num_experts_per_tok = Some(1);
    assert!(model.config.is_moe());
    let rows: [(&[u32], &[u32]); 2] =
        [(&[3, 14, 15, 92], &[10, 20]), (&[3, 14, 15, 65], &[10, 20])];
    let shared = model
        .score_option_rows_shared_prefix(&rows)
        .expect("scores");
    for ((prompt_ids, letter_ids), scores) in rows.iter().zip(&shared) {
        let serial = model
            .score_option_letters(prompt_ids, letter_ids)
            .expect("serial");
        assert_eq!(
            serial
                .logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            scores
                .logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn shared_prefix_scoring_refuses_a_bad_row_like_the_serial_scorer() {
    let model = build_model(test_config(), 0x0A71_0C24);
    let rows: [(&[u32], &[u32]); 2] = [(&[1, 2, 3], &[4]), (&[], &[4])];
    match (
        model.score_option_rows_shared_prefix(&rows),
        model.score_option_letters(&[], &[4]),
    ) {
        (Err(InferenceError::InvalidInput(a)), Err(InferenceError::InvalidInput(b))) => {
            assert_eq!(a, b);
        }
        other => panic!("expected the same InvalidInput from both, got {other:?}"),
    }
}

#[test]
fn a_scoring_prefix_serves_several_continuations_like_whole_prompts() {
    let model = build_model(test_config(), 0x0A71_0C31);
    let mut rng = 0x5EED_0C31_u64;
    let state = random_prompt(&mut rng, 70);
    let prefix = model.prefill_scoring_prefix(&state).expect("prefix");
    assert_eq!(prefix.ids(), state.as_slice());
    let letters = [4_u32, 9, 33];
    // The same prefix is read twice for one suffix: a continuation must not
    // change the prefix it started from.
    for suffix_len in [1, 12, 1, 40] {
        let prompt = [state.as_slice(), &random_prompt(&mut rng, suffix_len)].concat();
        let after = model
            .score_option_letters_after_prefix(&prefix, &prompt, &letters)
            .expect("after prefix");
        let whole = model
            .score_option_letters_batched(&prompt, &letters)
            .expect("whole");
        assert_scores_close(&whole, &after, &format!("suffix {suffix_len}"));
    }
    let again = [state.as_slice(), &[7, 7, 7]].concat();
    let first = model
        .score_option_letters_after_prefix(&prefix, &again, &letters)
        .expect("first");
    let second = model
        .score_option_letters_after_prefix(&prefix, &again, &letters)
        .expect("second");
    assert_eq!(
        first.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        second
            .logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn a_prompt_that_does_not_continue_the_prefix_is_refused() {
    let model = build_model(test_config(), 0x0A71_0C32);
    let prefix = model.prefill_scoring_prefix(&[5, 6, 7]).expect("prefix");
    for prompt in [&[5_u32, 6, 7][..], &[5, 6], &[5, 9, 7, 1], &[1, 5, 6, 7]] {
        match model.score_option_letters_after_prefix(&prefix, prompt, &[1, 2]) {
            Err(InferenceError::InvalidInput(message)) => {
                assert!(message.contains("does not continue"), "{message}");
            }
            other => panic!("expected a refusal for {prompt:?}, got {other:?}"),
        }
    }
}

/// A cancellation check that counts its calls and answers true from call
/// `cancels_on` on.
fn cancel_on_call(cancels_on: usize, calls: &std::cell::Cell<usize>) -> impl FnMut() -> bool + '_ {
    move || {
        calls.set(calls.get() + 1);
        calls.get() >= cancels_on
    }
}

#[test]
fn scoring_with_cancel_asks_before_each_layer_and_stops_there() {
    let model = build_model(test_config(), 0x0A71_0C36);
    let layers = model.config.num_hidden_layers;
    assert!(layers >= 2, "the test needs a layer to stop before");
    let letters = [4_u32, 9];
    let prompt = [5_u32, 6, 7, 8];
    let calls = std::cell::Cell::new(0);
    let prefix = model
        .prefill_scoring_prefix_with_cancel(&prompt[..3], &mut cancel_on_call(usize::MAX, &calls))
        .expect("prefix")
        .expect("not cancelled");
    assert_eq!(calls.get(), layers);
    for stop in [1, layers] {
        calls.set(0);
        let stopped = model
            .prefill_scoring_prefix_with_cancel(&prompt[..3], &mut cancel_on_call(stop, &calls))
            .expect("prefix");
        assert!(stopped.is_none());
        assert_eq!(calls.get(), stop);
        calls.set(0);
        let stopped = model
            .score_option_letters_batched_with_cancel(
                &prompt,
                &letters,
                &mut cancel_on_call(stop, &calls),
            )
            .expect("batched");
        assert!(stopped.is_none());
        assert_eq!(calls.get(), stop);
        calls.set(0);
        let stopped = model
            .score_option_letters_after_prefix_with_cancel(
                &prefix,
                &prompt,
                &letters,
                &mut cancel_on_call(stop, &calls),
            )
            .expect("after prefix");
        assert!(stopped.is_none());
        assert_eq!(calls.get(), stop);
    }
    // A check that never fires leaves the scores of the plain call.
    calls.set(0);
    let scored = model
        .score_option_letters_after_prefix_with_cancel(
            &prefix,
            &prompt,
            &letters,
            &mut cancel_on_call(usize::MAX, &calls),
        )
        .expect("after prefix")
        .expect("not cancelled");
    assert_eq!(calls.get(), layers);
    let plain = model
        .score_option_letters_after_prefix(&prefix, &prompt, &letters)
        .expect("plain");
    assert_eq!(
        scored
            .logits
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        plain.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn a_scoring_prefix_is_refused_for_moe_and_for_a_prefix_without_room() {
    let mut cfg = test_config();
    cfg.max_position_embeddings = 8;
    let model = build_model(cfg, 0x0A71_0C33);
    assert!(matches!(
        model.prefill_scoring_prefix(&[1; 8]),
        Err(InferenceError::InvalidInput(_))
    ));
    assert!(model.prefill_scoring_prefix(&[1; 7]).is_ok());
    assert!(matches!(
        model.prefill_scoring_prefix(&[]),
        Err(InferenceError::InvalidInput(_))
    ));
    let mut moe = build_model(test_config(), 0x0A71_0C34);
    moe.config.num_experts = Some(1);
    moe.config.num_experts_per_tok = Some(1);
    assert!(matches!(
        moe.prefill_scoring_prefix(&[1, 2, 3]),
        Err(InferenceError::UnsupportedModel(_))
    ));
}

#[test]
fn a_batched_prefill_refuses_a_cache_that_does_not_hold_its_positions() {
    let model = build_model(test_config(), 0x0A71_0C35);
    let cfg = &model.config;
    let geometry = qwen35_gdn_geometry(cfg).expect("geometry");
    let states = |count: usize| -> Vec<GatedDeltaNetState> {
        (0..count)
            .map(|_| GatedDeltaNetState::from_geometry(&geometry))
            .collect()
    };
    let mut claims_positions = KvCache::new(cfg.num_full_attention_layers());
    claims_positions.seq_len = 3;
    let mut fresh = KvCache::new(cfg.num_full_attention_layers());
    let cases: [(Vec<GatedDeltaNetState>, &mut KvCache); 2] = [
        (
            states(cfg.num_linear_attention_layers()),
            &mut claims_positions,
        ),
        (states(cfg.num_linear_attention_layers() + 1), &mut fresh),
    ];
    for (mut gdn, cache) in cases {
        match model.prefill_tokens_batched_for_generate(&[1, 2], &mut gdn, cache) {
            Err(InferenceError::InvalidInput(message)) => {
                assert!(message.contains("does not match"), "{message}");
            }
            other => panic!("expected a refusal, got {:?}", other.map(|l| l.len())),
        }
    }
}
