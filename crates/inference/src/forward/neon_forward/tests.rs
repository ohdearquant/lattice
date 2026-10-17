use super::*;
use crate::decoder::standalone_cpu::parity;
use crate::model::qwen35_config::LayerType;
use crate::stop_reason::StopReason;

/// Helper: create a zero Q8_0 packed weight buffer for [n, k].
fn zero_packed(n: usize, k: usize) -> Vec<u8> {
    pack_weights_q8(&vec![0.0f32; n * k], n, k).unwrap()
}

/// `generate_q8_neon` is `pub` in a published crate, so its signature is
/// part of the surface a later change to its body must not move. The
/// matching pins for `generate_f16` and `generate_q8` live in `cpu_f16` and
/// `cpu_q8`.
#[test]
#[allow(clippy::type_complexity)]
fn generate_q8_neon_signature_is_pinned() {
    let _generate: fn(
        &Q8NeonModel,
        &Qwen35Config,
        &BpeTokenizer,
        &RopeTable,
        &str,
        &GenerateConfig,
    ) -> Result<GenerateOutput, InferenceError> = generate_q8_neon;
}

#[test]
fn test_quantize_model_produces_valid_packed_sizes() {
    let cfg = Qwen35Config::qwen35_2b();
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let inter = cfg.intermediate_size;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();

    // Q8_0 packed size: n * (k/32) * 36 bytes
    let q8_packed_size = |n: usize, k: usize| -> usize {
        assert_eq!(k % 32, 0, "k={k} must be multiple of 32");
        n * (k / 32) * 36
    };

    // Build a minimal ModelWeights with known-size zero tensors
    let gdn_w = GatedDeltaNetWeights {
        in_proj_qkv: vec![0.0; qkv_dim * hidden],
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z: vec![0.0; output_dim * hidden],
        in_proj_z_rows: output_dim,
        in_proj_z_cols: hidden,
        in_proj_b: vec![0.0; cfg.linear_num_value_heads() * hidden],
        in_proj_b_rows: cfg.linear_num_value_heads(),
        in_proj_b_cols: hidden,
        in_proj_a: vec![0.0; cfg.linear_num_value_heads() * hidden],
        in_proj_a_rows: cfg.linear_num_value_heads(),
        in_proj_a_cols: hidden,
        a_log: vec![0.0; cfg.linear_num_value_heads()],
        dt_bias: vec![0.0; cfg.linear_num_value_heads()],
        conv1d_weight: vec![0.0; qkv_dim * cfg.linear_conv_kernel_dim],
        conv_dim: qkv_dim,
        kernel_size: cfg.linear_conv_kernel_dim,
        norm_weight: vec![0.0; cfg.linear_value_head_dim],
        out_proj: vec![0.0; hidden * output_dim],
        out_proj_rows: hidden,
        out_proj_cols: output_dim,
    };

    let full_w = FullAttentionLayerWeights {
        q_proj: vec![0.0; 2 * q_dim * hidden],
        k_proj: vec![0.0; kv_dim * hidden],
        v_proj: vec![0.0; kv_dim * hidden],
        o_proj: vec![0.0; hidden * q_dim],
        q_norm: vec![0.0; cfg.head_dim],
        k_norm: vec![0.0; cfg.head_dim],
    };

    let make_common_w = || CommonLayerWeights {
        input_layernorm: vec![0.0; hidden],
        post_attention_layernorm: vec![0.0; hidden],
        ffn: crate::model::qwen35::FeedForwardWeights::Dense(
            crate::model::qwen35::DenseFfnWeights {
                gate_proj: vec![0.0; inter * hidden],
                up_proj: vec![0.0; inter * hidden],
                down_proj: vec![0.0; hidden * inter],
            },
        ),
    };

    // Build one linear + one full layer to test both paths
    let weights = ModelWeights {
        embed_tokens: vec![0.0; vocab * hidden],
        lm_head: None,
        final_norm: vec![0.0; hidden],
        layers: vec![
            (AttentionWeights::Linear(gdn_w), make_common_w()),
            (AttentionWeights::Full(full_w), make_common_w()),
        ],
    };

    let q8 = quantize_model(&weights, &cfg).unwrap();

    // Check lm_head packed size
    assert_eq!(q8.lm_head_packed.len(), q8_packed_size(vocab, hidden));
    assert_eq!(q8.lm_head_rows, vocab);
    assert_eq!(q8.lm_head_cols, hidden);

    // Check embed_tokens preserved
    assert_eq!(q8.embed_tokens.len(), vocab * hidden);

    // Check layer 0 (linear)
    match &q8.layers[0].0 {
        Q8NeonAttentionWeights::Linear(gdn) => {
            assert_eq!(
                gdn.in_proj_qkv_packed.len(),
                q8_packed_size(qkv_dim, hidden)
            );
            assert_eq!(
                gdn.in_proj_z_packed.len(),
                q8_packed_size(output_dim, hidden)
            );
            assert_eq!(
                gdn.out_proj_packed.len(),
                q8_packed_size(hidden, output_dim)
            );
            assert_eq!(gdn.a_log.len(), cfg.linear_num_key_heads);
        }
        Q8NeonAttentionWeights::Full(_) => panic!("expected Linear layer"),
    }

    // Check layer 1 (full)
    match &q8.layers[1].0 {
        Q8NeonAttentionWeights::Full(full) => {
            assert_eq!(full.q_proj_packed.len(), q8_packed_size(2 * q_dim, hidden));
            assert_eq!(full.k_proj_packed.len(), q8_packed_size(kv_dim, hidden));
            assert_eq!(full.v_proj_packed.len(), q8_packed_size(kv_dim, hidden));
            assert_eq!(full.o_proj_packed.len(), q8_packed_size(hidden, q_dim));
        }
        Q8NeonAttentionWeights::Linear(_) => panic!("expected Full layer"),
    }

    // Check common weights
    let c = &q8.layers[0].1;
    assert_eq!(c.gate_proj_packed.len(), q8_packed_size(inter, hidden));
    assert_eq!(c.up_proj_packed.len(), q8_packed_size(inter, hidden));
    assert_eq!(c.down_proj_packed.len(), q8_packed_size(hidden, inter));
    assert_eq!(c.input_layernorm.len(), hidden);
}

/// A `ModelWeights` whose GDN `in_proj_b_rows`/`in_proj_a_rows`/`a_log`/`dt_bias` are
/// internally consistent (all finite, all agree with each other) but disagree with
/// `cfg.linear_num_value_heads()` must be rejected by the public NEON `quantize_model`
/// with a typed `Err`, not a panic. Before `pack_gdn_weights` called
/// `validate_gdn_shapes`, this geometry reached assert-based `pack_weights_q8`
/// (self-consistent rows/cols always satisfy `weights.len() == n * k`) and panicked
/// downstream in `matmul_q8_neon_into` at inference time instead of failing at
/// ingress — the NEON sibling of the guarded CPU Q8 path.
#[test]
fn quantize_model_rejects_gdn_value_heads_disagreeing_with_cfg() {
    let cfg = Qwen35Config::qwen35_2b();
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let inter = cfg.intermediate_size;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();

    // Internally consistent value-head count (5) that disagrees with
    // cfg.linear_num_value_heads() (16 for qwen35_2b).
    let bad_value_heads = 5;
    assert_ne!(bad_value_heads, cfg.linear_num_value_heads());
    let gdn_w = GatedDeltaNetWeights {
        in_proj_qkv: vec![0.0; qkv_dim * hidden],
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z: vec![0.0; output_dim * hidden],
        in_proj_z_rows: output_dim,
        in_proj_z_cols: hidden,
        in_proj_b: vec![0.0; bad_value_heads * hidden],
        in_proj_b_rows: bad_value_heads,
        in_proj_b_cols: hidden,
        in_proj_a: vec![0.0; bad_value_heads * hidden],
        in_proj_a_rows: bad_value_heads,
        in_proj_a_cols: hidden,
        a_log: vec![0.0; bad_value_heads],
        dt_bias: vec![0.0; bad_value_heads],
        conv1d_weight: vec![0.0; qkv_dim * cfg.linear_conv_kernel_dim],
        conv_dim: qkv_dim,
        kernel_size: cfg.linear_conv_kernel_dim,
        norm_weight: vec![0.0; cfg.linear_value_head_dim],
        out_proj: vec![0.0; hidden * output_dim],
        out_proj_rows: hidden,
        out_proj_cols: output_dim,
    };

    let common_w = CommonLayerWeights {
        input_layernorm: vec![0.0; hidden],
        post_attention_layernorm: vec![0.0; hidden],
        ffn: crate::model::qwen35::FeedForwardWeights::Dense(
            crate::model::qwen35::DenseFfnWeights {
                gate_proj: vec![0.0; inter * hidden],
                up_proj: vec![0.0; inter * hidden],
                down_proj: vec![0.0; hidden * inter],
            },
        ),
    };

    let weights = ModelWeights {
        embed_tokens: vec![0.0; vocab * hidden],
        lm_head: None,
        final_norm: vec![0.0; hidden],
        layers: vec![(AttentionWeights::Linear(gdn_w), common_w)],
    };

    match quantize_model(&weights, &cfg) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("in_proj_b"),
                "error must name in_proj_b, got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!(
            "expected Err for GDN value-head count disagreeing with cfg, got Ok \
                 (would panic in matmul_q8_neon_into downstream)"
        ),
    }
}

/// A config with `num_attention_heads = 2^63` and `head_dim = 2` overflows
/// `full_q_dim()`/`full_kv_dim()` (`num_attention_heads * head_dim`) to a small
/// wrapped value in release builds. `pack_full_attn_weights` must reject this via
/// the checked accessors before packing a Q/O tensor sized to the wrapped
/// dimension while `forward_step_q8_neon` would still index by the original,
/// unwrapped head count.
#[test]
fn quantize_model_rejects_full_attention_overflowing_config() {
    let cfg = Qwen35Config {
        num_attention_heads: 1 << 63,
        num_key_value_heads: 1 << 63,
        head_dim: 2,
        ..Qwen35Config::qwen35_2b()
    };
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let inter = cfg.intermediate_size;

    // Content doesn't matter — the overflow must be caught before any tensor
    // geometry derived from it is even consulted.
    let full_w = FullAttentionLayerWeights {
        q_proj: vec![0.0; 4],
        k_proj: vec![0.0; 4],
        v_proj: vec![0.0; 4],
        o_proj: vec![0.0; 4],
        q_norm: vec![0.0; cfg.head_dim],
        k_norm: vec![0.0; cfg.head_dim],
    };
    let common_w = CommonLayerWeights {
        input_layernorm: vec![0.0; hidden],
        post_attention_layernorm: vec![0.0; hidden],
        ffn: crate::model::qwen35::FeedForwardWeights::Dense(
            crate::model::qwen35::DenseFfnWeights {
                gate_proj: vec![0.0; inter * hidden],
                up_proj: vec![0.0; inter * hidden],
                down_proj: vec![0.0; hidden * inter],
            },
        ),
    };
    let weights = ModelWeights {
        embed_tokens: vec![0.0; vocab * hidden],
        lm_head: None,
        final_norm: vec![0.0; hidden],
        layers: vec![(AttentionWeights::Full(full_w), common_w)],
    };

    match quantize_model(&weights, &cfg) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("overflow"),
                "error must describe the overflow, got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!(
            "expected Err for overflowing full-attention config, got Ok (would pack an \
                 undersized tensor and panic in forward_step_q8_neon downstream)"
        ),
    }
}

/// The NEON GDN path retains `a_log`, `dt_bias`, `conv1d_weight`, and `norm_weight`
/// as f32 without quantizing them, so they must be explicitly finite-checked (via
/// the shared `validate_gdn_shapes`) instead of copied verbatim. A `ModelWeights`
/// carrying a NaN in `a_log` must be rejected at ingress, not accepted to poison
/// the GDN decay recurrence at decode time.
#[test]
fn quantize_model_rejects_non_finite_gdn_a_log() {
    let cfg = Qwen35Config::qwen35_2b();
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let inter = cfg.intermediate_size;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let value_heads = cfg.linear_num_value_heads();

    let mut gdn_w = GatedDeltaNetWeights {
        in_proj_qkv: vec![0.0; qkv_dim * hidden],
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z: vec![0.0; output_dim * hidden],
        in_proj_z_rows: output_dim,
        in_proj_z_cols: hidden,
        in_proj_b: vec![0.0; value_heads * hidden],
        in_proj_b_rows: value_heads,
        in_proj_b_cols: hidden,
        in_proj_a: vec![0.0; value_heads * hidden],
        in_proj_a_rows: value_heads,
        in_proj_a_cols: hidden,
        a_log: vec![0.0; value_heads],
        dt_bias: vec![0.0; value_heads],
        conv1d_weight: vec![0.0; qkv_dim * cfg.linear_conv_kernel_dim],
        conv_dim: qkv_dim,
        kernel_size: cfg.linear_conv_kernel_dim,
        norm_weight: vec![0.0; cfg.linear_value_head_dim],
        out_proj: vec![0.0; hidden * output_dim],
        out_proj_rows: hidden,
        out_proj_cols: output_dim,
    };
    gdn_w.a_log[0] = f32::NAN;

    let common_w = CommonLayerWeights {
        input_layernorm: vec![0.0; hidden],
        post_attention_layernorm: vec![0.0; hidden],
        ffn: crate::model::qwen35::FeedForwardWeights::Dense(
            crate::model::qwen35::DenseFfnWeights {
                gate_proj: vec![0.0; inter * hidden],
                up_proj: vec![0.0; inter * hidden],
                down_proj: vec![0.0; hidden * inter],
            },
        ),
    };
    let weights = ModelWeights {
        embed_tokens: vec![0.0; vocab * hidden],
        lm_head: None,
        final_norm: vec![0.0; hidden],
        layers: vec![(AttentionWeights::Linear(gdn_w), common_w)],
    };

    match quantize_model(&weights, &cfg) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(msg.contains("a_log"), "error must name a_log, got: {msg}");
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!(
            "expected Err for non-finite a_log, got Ok (would poison the GDN decay \
                 recurrence at decode time)"
        ),
    }
}

/// A short `post_attention_layernorm` (shorter than `cfg.hidden_size`) must be
/// rejected by the NEON packing path, not silently accepted: `qwen35_rms_norm`
/// zips gamma against the `hidden`-length activation row, so a short gamma would
/// leave trailing hidden values unnormalized instead of failing loudly — silent
/// wrong output, not a panic.
#[test]
fn quantize_model_rejects_short_post_attention_layernorm() {
    let cfg = Qwen35Config::qwen35_2b();
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let inter = cfg.intermediate_size;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let value_heads = cfg.linear_num_value_heads();

    let gdn_w = GatedDeltaNetWeights {
        in_proj_qkv: vec![0.0; qkv_dim * hidden],
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z: vec![0.0; output_dim * hidden],
        in_proj_z_rows: output_dim,
        in_proj_z_cols: hidden,
        in_proj_b: vec![0.0; value_heads * hidden],
        in_proj_b_rows: value_heads,
        in_proj_b_cols: hidden,
        in_proj_a: vec![0.0; value_heads * hidden],
        in_proj_a_rows: value_heads,
        in_proj_a_cols: hidden,
        a_log: vec![0.0; value_heads],
        dt_bias: vec![0.0; value_heads],
        conv1d_weight: vec![0.0; qkv_dim * cfg.linear_conv_kernel_dim],
        conv_dim: qkv_dim,
        kernel_size: cfg.linear_conv_kernel_dim,
        norm_weight: vec![0.0; cfg.linear_value_head_dim],
        out_proj: vec![0.0; hidden * output_dim],
        out_proj_rows: hidden,
        out_proj_cols: output_dim,
    };
    let common_w = CommonLayerWeights {
        input_layernorm: vec![0.0; hidden],
        post_attention_layernorm: vec![0.0; hidden - 1], // shorter than cfg.hidden_size
        ffn: crate::model::qwen35::FeedForwardWeights::Dense(
            crate::model::qwen35::DenseFfnWeights {
                gate_proj: vec![0.0; inter * hidden],
                up_proj: vec![0.0; inter * hidden],
                down_proj: vec![0.0; hidden * inter],
            },
        ),
    };
    let weights = ModelWeights {
        embed_tokens: vec![0.0; vocab * hidden],
        lm_head: None,
        final_norm: vec![0.0; hidden],
        layers: vec![(AttentionWeights::Linear(gdn_w), common_w)],
    };

    match quantize_model(&weights, &cfg) {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("post_attention_layernorm"),
                "error must name post_attention_layernorm, got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidInput, got: {e}"),
        Ok(_) => panic!(
            "expected Err for short post_attention_layernorm, got Ok (would silently \
                 leave trailing hidden values unnormalized)"
        ),
    }
}

#[test]
fn test_forward_step_q8_neon_zero_weights_produces_zero_logits() {
    let cfg = Qwen35Config::qwen35_2b();
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let inter = cfg.intermediate_size;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let num_heads = cfg.linear_num_key_heads;
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();

    // Build a minimal 2-layer model (1 linear + 1 full) with zero weights
    let make_linear = || Q8NeonGdnWeights {
        in_proj_qkv_packed: zero_packed(qkv_dim, hidden),
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z_packed: zero_packed(output_dim, hidden),
        in_proj_z_rows: output_dim,
        in_proj_z_cols: hidden,
        in_proj_b_packed: zero_packed(num_heads, hidden),
        in_proj_b_rows: num_heads,
        in_proj_b_cols: hidden,
        in_proj_a_packed: zero_packed(num_heads, hidden),
        in_proj_a_rows: num_heads,
        in_proj_a_cols: hidden,
        out_proj_packed: zero_packed(hidden, output_dim),
        out_proj_rows: hidden,
        out_proj_cols: output_dim,
        a_log: vec![0.0; num_heads],
        dt_bias: vec![0.0; num_heads],
        conv1d_weight: vec![0.0; qkv_dim * cfg.linear_conv_kernel_dim],
        conv_dim: qkv_dim,
        kernel_size: cfg.linear_conv_kernel_dim,
        norm_weight: vec![0.0; cfg.linear_value_head_dim],
    };

    let make_full = || Q8NeonFullAttnWeights {
        q_proj_packed: zero_packed(2 * q_dim, hidden),
        q_proj_rows: 2 * q_dim,
        q_proj_cols: hidden,
        k_proj_packed: zero_packed(kv_dim, hidden),
        k_proj_rows: kv_dim,
        k_proj_cols: hidden,
        v_proj_packed: zero_packed(kv_dim, hidden),
        v_proj_rows: kv_dim,
        v_proj_cols: hidden,
        o_proj_packed: zero_packed(hidden, q_dim),
        o_proj_rows: hidden,
        o_proj_cols: q_dim,
        q_norm: vec![0.0; cfg.head_dim],
        k_norm: vec![0.0; cfg.head_dim],
    };

    let make_common = || Q8NeonCommonWeights {
        input_layernorm: vec![0.0; hidden],
        post_attention_layernorm: vec![0.0; hidden],
        gate_proj_packed: zero_packed(inter, hidden),
        gate_proj_rows: inter,
        gate_proj_cols: hidden,
        up_proj_packed: zero_packed(inter, hidden),
        up_proj_rows: inter,
        up_proj_cols: hidden,
        down_proj_packed: zero_packed(hidden, inter),
        down_proj_rows: hidden,
        down_proj_cols: inter,
    };

    // Build config for just 2 layers to keep test fast
    let mut test_cfg = cfg.clone();
    test_cfg.num_hidden_layers = 2;
    test_cfg.layer_types = vec![
        crate::model::qwen35_config::LayerType::LinearAttention,
        crate::model::qwen35_config::LayerType::FullAttention,
    ];

    let model = Q8NeonModel {
        embed_tokens: vec![0.0; vocab * hidden],
        final_norm: vec![0.0; hidden],
        lm_head_packed: zero_packed(vocab, hidden),
        lm_head_rows: vocab,
        lm_head_cols: hidden,
        layers: vec![
            (Q8NeonAttentionWeights::Linear(make_linear()), make_common()),
            (Q8NeonAttentionWeights::Full(make_full()), make_common()),
        ],
    };

    let rope_dim = test_cfg.rope_dim();
    let rope_max = test_cfg.max_position_embeddings.min(8192);
    let rope = RopeTable::new(rope_dim, rope_max, test_cfg.rope_theta);

    let num_linear = test_cfg.num_linear_attention_layers();
    let num_full = test_cfg.num_full_attention_layers();
    let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
        .map(|_| GatedDeltaNetState::new(&test_cfg))
        .collect();
    let mut kv_cache = KvCache::new(num_full);
    let mut scratch = ForwardScratch::new();

    forward_step_q8_neon(
        &model,
        &test_cfg,
        &rope,
        0, // token_id
        0, // position
        &mut gdn_states,
        &mut kv_cache,
        &mut scratch,
    );

    // With all-zero weights and embeddings, all logits should be zero
    for (i, &v) in scratch.logits[..test_cfg.vocab_size].iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "logit[{i}] = {v}, expected ~0.0 with zero weights"
        );
    }
}

#[test]
fn test_gdn_step_q8_neon_zero_weights_produces_zero_output() {
    let cfg = Qwen35Config::qwen35_2b();
    let hidden = cfg.hidden_size;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let value_heads = cfg.linear_num_value_heads();

    let weights = Q8NeonGdnWeights {
        in_proj_qkv_packed: zero_packed(qkv_dim, hidden),
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z_packed: zero_packed(output_dim, hidden),
        in_proj_z_rows: output_dim,
        in_proj_z_cols: hidden,
        in_proj_b_packed: zero_packed(value_heads, hidden),
        in_proj_b_rows: value_heads,
        in_proj_b_cols: hidden,
        in_proj_a_packed: zero_packed(value_heads, hidden),
        in_proj_a_rows: value_heads,
        in_proj_a_cols: hidden,
        out_proj_packed: zero_packed(hidden, output_dim),
        out_proj_rows: hidden,
        out_proj_cols: output_dim,
        a_log: vec![0.0; value_heads],
        dt_bias: vec![0.0; value_heads],
        conv1d_weight: vec![0.0; qkv_dim * cfg.linear_conv_kernel_dim],
        conv_dim: qkv_dim,
        kernel_size: cfg.linear_conv_kernel_dim,
        norm_weight: vec![0.0; cfg.linear_value_head_dim],
    };

    let mut state = GatedDeltaNetState::new(&cfg);
    let input = vec![0.0f32; hidden];
    let mut output = vec![0.0f32; hidden];
    let mut gdn_scratch = GatedDeltaNetFusedScratch::default();
    let mut x_q_scratch = Vec::new();

    gdn_step_q8_neon(
        &input,
        &mut state,
        &weights,
        &cfg,
        &mut gdn_scratch,
        &mut x_q_scratch,
        &mut output,
    );

    for (i, &v) in output[..hidden].iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "output[{i}] = {v}, expected 0.0 with zero weights + zero input"
        );
    }
}

#[test]
fn test_ffn_step_q8_neon_zero_weights() {
    let cfg = Qwen35Config::qwen35_2b();
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;

    let common = Q8NeonCommonWeights {
        input_layernorm: vec![0.0; hidden],
        post_attention_layernorm: vec![0.0; hidden],
        gate_proj_packed: zero_packed(inter, hidden),
        gate_proj_rows: inter,
        gate_proj_cols: hidden,
        up_proj_packed: zero_packed(inter, hidden),
        up_proj_rows: inter,
        up_proj_cols: hidden,
        down_proj_packed: zero_packed(hidden, inter),
        down_proj_rows: hidden,
        down_proj_cols: inter,
    };

    let mut scratch = ForwardScratch::new();
    scratch.ensure_capacity(&cfg, 1);
    scratch.ffn_out[..hidden].fill(0.0);

    ffn_step_q8_neon(&common, &mut scratch, hidden);

    for (i, &v) in scratch.ffn_out[..hidden].iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "ffn_out[{i}] = {v}, expected 0.0 with zero weights"
        );
    }
}

/// Regression test for #392: NEON Q8 RoPE must use stride-half pairing (i, half+i), not
/// interleaved (2i, 2i+1).
///
/// Mirror of `test_full_attn_step_q8_rope_stride_half_parity` in cpu_q8.rs, using the
/// NEON packed weight format and `full_attention_step_q8_neon`.  The identity K-projection
/// propagates quantised input into k_buf with no W_k quantisation error.  A stride-half
/// reference reproduces the expected k_cache; max_diff < 1e-4 with the fix, ~0.9 with the
/// interleaved bug (verified by mutation testing).
#[test]
fn test_full_attn_step_q8_neon_rope_stride_half_parity() {
    let head_dim: usize = 32;
    let num_q_heads: usize = 1;
    let num_kv_heads: usize = 1;
    let hidden: usize = 64;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let position: usize = 3;

    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 2,
        vocab_size: 128,
        intermediate_size: 128,
        rms_norm_eps: 1e-6,
        num_attention_heads: num_q_heads,
        num_key_value_heads: num_kv_heads,
        head_dim,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.5, // rope_dim = 16, half = 8
        rope_parameters: None,
        linear_num_key_heads: 2,
        linear_num_value_heads: Some(2),
        linear_key_head_dim: 32,
        linear_value_head_dim: 32,
        linear_conv_kernel_dim: 4,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: 2,
        layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
        layer_mask: vec![true; 2],
        eos_token_id: 127,
        max_position_embeddings: 512,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    let rope_dim = cfg.rope_dim(); // = 16
    let half = rope_dim / 2; // = 8
    let rope = RopeTable::new(rope_dim, 512, cfg.rope_theta);

    // W_k packed identity [kv_dim=32, hidden=64]: row j = e_j → k_buf[j] = x[j].
    // Q8_0 block packs scale=1/127 and one i8=127; dequant gives exact 1.0 × x[j].
    let identity_packed = {
        let mut mat = vec![0.0f32; kv_dim * hidden];
        for j in 0..kv_dim {
            mat[j * hidden + j] = 1.0;
        }
        pack_weights_q8(&mat, kv_dim, hidden).unwrap()
    };

    // W_q = identity for first q_dim rows (Q part), zeros for next q_dim rows (gate part).
    // Row j selects input[j] exactly so scratch.q_buf is non-trivial and Q-loop mutation
    // changes the assertion result.
    let q_identity_packed = {
        let mut mat = vec![0.0f32; 2 * q_dim * hidden];
        for j in 0..q_dim {
            mat[j * hidden + j] = 1.0;
        }
        pack_weights_q8(&mat, 2 * q_dim, hidden).unwrap()
    };

    let weights = Q8NeonFullAttnWeights {
        q_proj_packed: q_identity_packed,
        q_proj_rows: 2 * q_dim,
        q_proj_cols: hidden,
        k_proj_packed: identity_packed,
        k_proj_rows: kv_dim,
        k_proj_cols: hidden,
        v_proj_packed: zero_packed(kv_dim, hidden),
        v_proj_rows: kv_dim,
        v_proj_cols: hidden,
        o_proj_packed: zero_packed(hidden, q_dim),
        o_proj_rows: hidden,
        o_proj_cols: q_dim,
        q_norm: vec![0.0f32; head_dim],
        k_norm: vec![0.0f32; head_dim],
    };

    let input: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.07).collect();

    let mut scratch = ForwardScratch::new();
    scratch.ensure_capacity(&cfg, 2);
    scratch.attn_out[..hidden].copy_from_slice(&input);

    let mut kv_cache = KvCache::new(1);
    full_attention_step_q8_neon(
        &weights,
        0,
        position,
        &mut kv_cache,
        &mut scratch,
        &cfg,
        &rope,
        hidden,
    );

    // Reference: same matmul + QK-norm + stride-half RoPE as the production path.

    // --- K reference ---
    let mut k_ref = vec![0.0f32; kv_dim];
    matmul_q8_neon_into(
        &input,
        &weights.k_proj_packed,
        kv_dim,
        hidden,
        &mut k_ref,
        &mut scratch.x_q_scratch,
    );
    qwen35_rms_norm(&mut k_ref, &weights.k_norm, head_dim, cfg.rms_norm_eps);

    let base = position * half;
    for i in 0..half {
        let cos_val = rope.cos_at(base + i);
        let sin_val = rope.sin_at(base + i);
        let x0 = k_ref[i];
        let x1 = k_ref[half + i];
        k_ref[i] = x0 * cos_val - x1 * sin_val;
        k_ref[half + i] = x0 * sin_val + x1 * cos_val;
    }

    let k_cached = &kv_cache.k[0][..kv_dim];
    let max_k_diff = k_cached
        .iter()
        .zip(k_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_k_diff < 1e-4,
        "NEON Q8 K-loop stride-half RoPE diverges from reference: max_k_diff = {max_k_diff:.6}. \
             With interleaved pairing the diff is O(0.1-1). Bug: #392."
    );

    // --- Q reference (guards the Q loop mutation) ---
    // The production scatter copies q_and_gate[0..q_dim] → scratch.q_buf[0..q_dim]
    // for head 0 (num_q_heads=1).
    let q_proj_dim = 2 * q_dim;
    let mut q_and_gate_ref = vec![0.0f32; q_proj_dim];
    matmul_q8_neon_into(
        &input,
        &weights.q_proj_packed,
        q_proj_dim,
        hidden,
        &mut q_and_gate_ref,
        &mut scratch.x_q_scratch,
    );
    let mut q_ref = q_and_gate_ref[..q_dim].to_vec();
    qwen35_rms_norm(&mut q_ref, &weights.q_norm, head_dim, cfg.rms_norm_eps);

    for i in 0..half {
        let cos_val = rope.cos_at(base + i);
        let sin_val = rope.sin_at(base + i);
        let x0 = q_ref[i];
        let x1 = q_ref[half + i];
        q_ref[i] = x0 * cos_val - x1 * sin_val;
        q_ref[half + i] = x0 * sin_val + x1 * cos_val;
    }

    let max_q_diff = scratch.q_buf[..q_dim]
        .iter()
        .zip(q_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_q_diff < 1e-4,
        "NEON Q8 Q-loop stride-half RoPE diverges from reference: max_q_diff = {max_q_diff:.6}. \
             With interleaved pairing the diff is O(0.1-1). Bug: #392."
    );
}

// -----------------------------------------------------------------------
// Nonzero parity test: captures current allocating NEON logits as baseline.
// After GDN/full-attention buffer migration (i2), re-run and verify the
// first 16 logits are within 1e-6 of these captured constants.
// -----------------------------------------------------------------------

/// Small deterministic test model — 2 layers, all dims multiples of 32.
fn make_nonzero_q8_neon_test_model() -> (Qwen35Config, Q8NeonModel, RopeTable) {
    let hidden: usize = 64;
    let vocab: usize = 128;
    let inter: usize = 128;
    let num_attn_heads: usize = 2;
    let num_kv_heads: usize = 1;
    let head_dim: usize = 32;
    let q_dim = num_attn_heads * head_dim; // 64
    let kv_dim = num_kv_heads * head_dim; // 32
    let lin_key_heads: usize = 2;
    let lin_val_heads: usize = 2;
    let lin_key_dim: usize = 32;
    let lin_val_dim: usize = 32;
    let lin_qkv_dim = lin_key_heads * lin_key_dim * 2 + lin_val_heads * lin_val_dim; // 192
    let lin_output_dim = lin_val_heads * lin_val_dim; // 64
    let kernel_size: usize = 4;

    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 2,
        vocab_size: vocab,
        intermediate_size: inter,
        rms_norm_eps: 1e-6,
        num_attention_heads: num_attn_heads,
        num_key_value_heads: num_kv_heads,
        head_dim,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.5,
        rope_parameters: None,
        linear_num_key_heads: lin_key_heads,
        linear_num_value_heads: Some(lin_val_heads),
        linear_key_head_dim: lin_key_dim,
        linear_value_head_dim: lin_val_dim,
        linear_conv_kernel_dim: kernel_size,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: 2,
        layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
        layer_mask: vec![true; 2],
        eos_token_id: 127,
        max_position_embeddings: 512,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    let rope_dim = (head_dim as f32 * cfg.partial_rotary_factor) as usize; // 16
    let rope = RopeTable::new(rope_dim, cfg.max_position_embeddings, cfg.rope_theta);

    // Deterministic weight generator: LCG producing small floats.
    let mut seed: u64 = 0xdead_beef_cafe_babe;
    let mut next_weight = |n: usize, k: usize| -> Vec<u8> {
        let floats: Vec<f32> = (0..n * k)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((seed >> 33) as f32 / u32::MAX as f32) * 0.04 - 0.02
            })
            .collect();
        // LCG output is bounded — always finite.
        pack_weights_q8(&floats, n, k).unwrap()
    };

    let gdn_w = Q8NeonGdnWeights {
        in_proj_qkv_packed: next_weight(lin_qkv_dim, hidden),
        in_proj_qkv_rows: lin_qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z_packed: next_weight(lin_output_dim, hidden),
        in_proj_z_rows: lin_output_dim,
        in_proj_z_cols: hidden,
        in_proj_b_packed: next_weight(lin_key_heads, hidden),
        in_proj_b_rows: lin_key_heads,
        in_proj_b_cols: hidden,
        in_proj_a_packed: next_weight(lin_key_heads, hidden),
        in_proj_a_rows: lin_key_heads,
        in_proj_a_cols: hidden,
        out_proj_packed: next_weight(hidden, lin_output_dim),
        out_proj_rows: hidden,
        out_proj_cols: lin_output_dim,
        a_log: vec![0.0f32; lin_key_heads],
        dt_bias: vec![0.0f32; lin_key_heads],
        conv1d_weight: vec![0.01f32; lin_qkv_dim * kernel_size],
        conv_dim: lin_qkv_dim,
        kernel_size,
        norm_weight: vec![0.0f32; lin_val_dim],
    };

    let full_w = Q8NeonFullAttnWeights {
        q_proj_packed: next_weight(2 * q_dim, hidden),
        q_proj_rows: 2 * q_dim,
        q_proj_cols: hidden,
        k_proj_packed: next_weight(kv_dim, hidden),
        k_proj_rows: kv_dim,
        k_proj_cols: hidden,
        v_proj_packed: next_weight(kv_dim, hidden),
        v_proj_rows: kv_dim,
        v_proj_cols: hidden,
        o_proj_packed: next_weight(hidden, q_dim),
        o_proj_rows: hidden,
        o_proj_cols: q_dim,
        q_norm: vec![0.0f32; head_dim],
        k_norm: vec![0.0f32; head_dim],
    };

    let common_w = |seed: &mut u64| {
        let mut nw = |n: usize, k: usize| -> Vec<u8> {
            let floats: Vec<f32> = (0..n * k)
                .map(|_| {
                    *seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((*seed >> 33) as f32 / u32::MAX as f32) * 0.04 - 0.02
                })
                .collect();
            // LCG output is bounded — always finite.
            pack_weights_q8(&floats, n, k).unwrap()
        };
        Q8NeonCommonWeights {
            input_layernorm: vec![0.0f32; hidden],
            post_attention_layernorm: vec![0.0f32; hidden],
            gate_proj_packed: nw(inter, hidden),
            gate_proj_rows: inter,
            gate_proj_cols: hidden,
            up_proj_packed: nw(inter, hidden),
            up_proj_rows: inter,
            up_proj_cols: hidden,
            down_proj_packed: nw(hidden, inter),
            down_proj_rows: hidden,
            down_proj_cols: inter,
        }
    };
    let common0 = common_w(&mut seed);
    let common1 = common_w(&mut seed);

    let embed_tokens: Vec<f32> = (0..vocab * hidden)
        .map(|i| {
            let mut s = (i as u64).wrapping_mul(0x9e3779b9_7f4a7c15);
            s ^= s >> 33;
            s &= 0xFFFF;
            s as f32 / 0x10000_u64 as f32 * 0.04 - 0.02
        })
        .collect();
    // Fixture embed is bounded hash output — always finite.
    let lm_head_packed = pack_weights_q8(&embed_tokens, vocab, hidden).unwrap();

    let model = Q8NeonModel {
        embed_tokens,
        final_norm: vec![0.0f32; hidden],
        lm_head_packed,
        lm_head_rows: vocab,
        lm_head_cols: hidden,
        layers: vec![
            (Q8NeonAttentionWeights::Linear(gdn_w), common0),
            (Q8NeonAttentionWeights::Full(full_w), common1),
        ],
    };

    (cfg, model, rope)
}

#[test]
fn test_forward_step_q8_neon_into_migration_preserves_nonzero_logits() {
    let (cfg, model, rope) = make_nonzero_q8_neon_test_model();
    let num_linear = cfg.num_linear_attention_layers();
    let num_full = cfg.num_full_attention_layers();

    let run_two_steps = || -> Vec<f32> {
        let mut gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
            .map(|_| GatedDeltaNetState::new(&cfg))
            .collect();
        let mut kv_cache = KvCache::new(num_full);
        let mut scratch = ForwardScratch::new();

        forward_step_q8_neon(
            &model,
            &cfg,
            &rope,
            7,
            0,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
        );
        kv_cache.seq_len += 1;
        forward_step_q8_neon(
            &model,
            &cfg,
            &rope,
            11,
            1,
            &mut gdn_states,
            &mut kv_cache,
            &mut scratch,
        );

        scratch.logits[..16].to_vec()
    };

    let logits_a = run_two_steps();
    let logits_b = run_two_steps();

    // Verify determinism (two identical runs must agree bitwise).
    assert_eq!(
        logits_a, logits_b,
        "forward_step_q8_neon is non-deterministic"
    );

    // Verify at least one non-zero logit (non-trivial model).
    assert!(
        logits_a.iter().any(|&v| v.abs() > 1e-9),
        "all 16 logits are zero — check weight generation"
    );

    // Baseline capture: hardcoded first 16 logits from the pre-migration allocating path.
    // Captured from `make_nonzero_q8_neon_test_model()` with seed 0xdead_beef_cafe_babe.
    // After i2 applies the zero-allocation migration, re-run this test; it must still pass.
    // To recapture: add `eprintln!("{:?}", &logits_a[..16]);` before this block, run test,
    // then update the array below.
    let expected: [f32; 16] = [
        -0.036519982,
        0.04271806,
        0.0027702842,
        0.007089529,
        -0.074217916,
        0.001136966,
        0.07316325,
        0.027880985,
        -0.027898913,
        -0.10935478,
        0.04180234,
        0.08315472,
        0.008195344,
        -0.07999688,
        0.014749594,
        0.028362377,
    ];
    for (i, (&actual, &exp)) in logits_a.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - exp).abs() <= 1e-6,
            "logit[{i}] mismatch: actual={actual:.8}, expected={exp:.8}"
        );
    }
}

// -----------------------------------------------------------------------
// generate_q8_neon output-stream goldens
// -----------------------------------------------------------------------
//
// Own fixture rather than reusing `make_nonzero_q8_neon_test_model` above:
// that fixture's `eos_token_id: 127` sits inside its `vocab_size: 128`, so
// an LCG-driven argmax could land on it "by luck" during a multi-step
// decode and make a golden's stop behaviour depend on incidental EOS
// timing rather than the `GenerateConfig` under test. Same LCG recipe and
// layer shapes as `make_nonzero_q8_neon_test_model` / the f16 and Q8
// sibling generate-golden fixtures, `eos_token_id` moved out of range.
//
// The NEON-vs-scalar architecture question:
// `matmul_q8_neon_into` (`forward/neon.rs`) is the only architecture-gated
// computation this wrapper's forward pass reaches -- GDN recurrence and
// GQA attention here use the plain scalar `gated_rms_norm`/`l2_normalize_vec`
// from `crate::attention::gdn`, not the `simd_*` dispatch helpers in
// `gdn_fused.rs` that `cpu_f16`/`cpu_q8` use for their GDN steps (see this
// module's own doc comment: "GDN recurrence and GQA attention: f32"). Per
// block, both the NEON and the scalar kernel in `neon.rs` sum the same
// 32 `i8 * i8` products into one `i32` -- the NEON kernel groups that sum
// differently (two 16-lane `vmull_s8`/`vmlal_s8` widen-multiplies plus a
// `vpadalq_s16`/`vaddvq_s32` horizontal reduction, vs. the scalar kernel's
// sequential `for i in 0..32`), but `i32` addition here is associative and
// never overflows (each product is bounded by 127*127, 32 of them can't
// approach `i32::MAX`), so the grouping difference cannot change the
// summed value. Both kernels then apply the *same single*
// `(dot as f32) * block_scale * x_scale` multiply, once per block, in the
// same block order (`neon.rs:76-104` scalar, `:118-207` NEON), so the two
// kernels are bit-identical by construction, not just close.
// `q8_neon_matmul_matches_scalar_kernel_directly` below proves that
// for the packed shapes this fixture actually uses, on whichever kernel
// this host's `#[cfg(target_arch)]` selects; the token-id-level goldens
// below therefore need no per-`target_arch` variant.
//
// Regenerating the goldens: run
//   cargo test -p lattice-inference --lib \
//     forward::neon_forward::tests::print_q8_neon_generate_goldens -- --ignored --nocapture
// and copy the printed `token_ids`/`stop_reason`/`stopped`/`text` lines into
// the four `#[test]` functions below.
fn make_nonzero_q8_neon_generate_fixture() -> (Qwen35Config, Q8NeonModel, RopeTable) {
    use crate::model::qwen35_config::LayerType;

    let hidden: usize = 64;
    let vocab: usize = 128;
    let inter: usize = 128;
    let num_attn_heads: usize = 2;
    let num_kv_heads: usize = 1;
    let head_dim: usize = 32;
    let q_dim = num_attn_heads * head_dim; // 64
    let kv_dim = num_kv_heads * head_dim; // 32
    let lin_key_heads: usize = 2;
    let lin_val_heads: usize = 2;
    let lin_key_dim: usize = 32;
    let lin_val_dim: usize = 32;
    let lin_qkv_dim = lin_key_heads * lin_key_dim * 2 + lin_val_heads * lin_val_dim; // 192
    let lin_output_dim = lin_val_heads * lin_val_dim; // 64
    let kernel_size: usize = 4;

    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 2,
        vocab_size: vocab,
        intermediate_size: inter,
        rms_norm_eps: 1e-6,
        num_attention_heads: num_attn_heads,
        num_key_value_heads: num_kv_heads,
        head_dim,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.5,
        rope_parameters: None,
        linear_num_key_heads: lin_key_heads,
        linear_num_value_heads: Some(lin_val_heads),
        linear_key_head_dim: lin_key_dim,
        linear_value_head_dim: lin_val_dim,
        linear_conv_kernel_dim: kernel_size,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: 2,
        layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
        layer_mask: vec![true; 2],
        eos_token_id: 999_999,
        max_position_embeddings: 512,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    let rope_dim = (head_dim as f32 * cfg.partial_rotary_factor) as usize; // 16
    let rope = RopeTable::new(rope_dim, cfg.max_position_embeddings, cfg.rope_theta);

    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    fn lcg_step(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*seed >> 33) as f32 / u32::MAX as f32
    }
    fn next_small_vec(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| lcg_step(seed) * scale - scale / 2.0)
            .collect()
    }
    // Norm gains, decay-gate params and conv taps: distinct, non-uniform
    // per-element draws (never a repeated literal), matching
    // `cpu_q8.rs`'s fixture recipe so all three wrappers' goldens are
    // built from the same LCG stream. Drawn BEFORE `next_weight` is
    // even defined: that closure holds a unique borrow of `seed` for
    // its whole life once created, so any `next_small_vec(&mut seed, ..)`
    // call sitting inside that span conflicts.
    let gdn_a_log = next_small_vec(&mut seed, lin_key_heads, 1.0);
    let gdn_dt_bias = next_small_vec(&mut seed, lin_key_heads, 1.0);
    let gdn_conv1d_weight = next_small_vec(&mut seed, lin_qkv_dim * kernel_size, 0.04);
    let gdn_norm_weight: Vec<f32> = next_small_vec(&mut seed, lin_val_dim, 1.0)
        .iter()
        .map(|g| g + 1.0)
        .collect();
    // q_norm/k_norm gains at scale 8.0, tried together with the v_proj/
    // o_proj scale below (round 2/3, measured insufficient to reach
    // position sensitivity; see `crates/inference/src/forward/cpu_q8.rs`'s
    // equivalent comment for the full measurement history). Kept at this
    // scale because it strictly widens the draw range this fixture's
    // content-sensitivity property uses, without regressing it.
    let full_q_norm = next_small_vec(&mut seed, head_dim, 8.0);
    let full_k_norm = next_small_vec(&mut seed, head_dim, 8.0);

    let mut next_weight = |n: usize, k: usize, scale: f32| -> Vec<u8> {
        let floats: Vec<f32> = (0..n * k)
            .map(|_| lcg_step(&mut seed) * scale - scale / 2.0)
            .collect();
        pack_weights_q8(&floats, n, k).unwrap()
    };

    let gdn_w = Q8NeonGdnWeights {
        in_proj_qkv_packed: next_weight(lin_qkv_dim, hidden, 0.04),
        in_proj_qkv_rows: lin_qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z_packed: next_weight(lin_output_dim, hidden, 0.04),
        in_proj_z_rows: lin_output_dim,
        in_proj_z_cols: hidden,
        in_proj_b_packed: next_weight(lin_key_heads, hidden, 0.04),
        in_proj_b_rows: lin_key_heads,
        in_proj_b_cols: hidden,
        in_proj_a_packed: next_weight(lin_key_heads, hidden, 0.04),
        in_proj_a_rows: lin_key_heads,
        in_proj_a_cols: hidden,
        out_proj_packed: next_weight(hidden, lin_output_dim, 0.04),
        out_proj_rows: hidden,
        out_proj_cols: lin_output_dim,
        a_log: gdn_a_log,
        dt_bias: gdn_dt_bias,
        conv1d_weight: gdn_conv1d_weight,
        conv_dim: lin_qkv_dim,
        kernel_size,
        norm_weight: gdn_norm_weight,
    };

    let full_w = Q8NeonFullAttnWeights {
        q_proj_packed: next_weight(2 * q_dim, hidden, 0.04),
        q_proj_rows: 2 * q_dim,
        q_proj_cols: hidden,
        k_proj_packed: next_weight(kv_dim, hidden, 0.04),
        k_proj_rows: kv_dim,
        k_proj_cols: hidden,
        v_proj_packed: next_weight(kv_dim, hidden, 4.0),
        v_proj_rows: kv_dim,
        v_proj_cols: hidden,
        o_proj_packed: next_weight(hidden, q_dim, 4.0),
        o_proj_rows: hidden,
        o_proj_cols: q_dim,
        q_norm: full_q_norm,
        k_norm: full_k_norm,
    };

    let common_w = |seed: &mut u64| {
        let input_layernorm = next_small_vec(seed, hidden, 1.0);
        let post_attention_layernorm = next_small_vec(seed, hidden, 1.0);
        let mut nw = |n: usize, k: usize| -> Vec<u8> {
            let floats: Vec<f32> = (0..n * k).map(|_| lcg_step(seed) * 0.04 - 0.02).collect();
            pack_weights_q8(&floats, n, k).unwrap()
        };
        Q8NeonCommonWeights {
            input_layernorm,
            post_attention_layernorm,
            gate_proj_packed: nw(inter, hidden),
            gate_proj_rows: inter,
            gate_proj_cols: hidden,
            up_proj_packed: nw(inter, hidden),
            up_proj_rows: inter,
            up_proj_cols: hidden,
            down_proj_packed: nw(hidden, inter),
            down_proj_rows: hidden,
            down_proj_cols: inter,
        }
    };
    let common0 = common_w(&mut seed);
    let common1 = common_w(&mut seed);

    let embed_tokens: Vec<f32> = (0..vocab * hidden)
        .map(|_| lcg_step(&mut seed) * 0.04 - 0.02)
        .collect();
    let lm_head_packed = pack_weights_q8(&embed_tokens, vocab, hidden).unwrap();

    let model = Q8NeonModel {
        embed_tokens,
        final_norm: next_small_vec(&mut seed, hidden, 1.0),
        lm_head_packed,
        lm_head_rows: vocab,
        lm_head_cols: hidden,
        layers: vec![
            (Q8NeonAttentionWeights::Linear(gdn_w), common0),
            (Q8NeonAttentionWeights::Full(full_w), common1),
        ],
    };

    (cfg, model, rope)
}

/// Tokenizer covering every id in the nonzero fixture's `vocab_size` (128).
/// Ids 0..=7 keep the single-character tokens the other tests in this file
/// already rely on; the golden prompt `"world"` encodes to five of them. Ids 8..128 get
/// distinct two-character printable tokens so every id the decode loop can
/// produce has real text instead of falling through to an empty string.
fn nonzero_fixture_tokenizer_q8_neon() -> BpeTokenizer {
    use std::collections::HashMap;
    let mut vocab_map: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o", "w", "r", "d", "!"].iter().enumerate() {
        vocab_map.insert((*c).to_string(), i as u32);
    }
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for id in 8u32..128 {
        let j = (id - 8) as usize;
        let a = ALPHABET[j / ALPHABET.len()] as char;
        let b = ALPHABET[j % ALPHABET.len()] as char;
        vocab_map.insert(format!("{a}{b}"), id);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    BpeTokenizer::from_vocab_and_merges(vocab_map, merges).unwrap()
}

/// Direct primitive-level cross-check: the dispatching `matmul_q8_neon_into`
/// (NEON on aarch64, scalar elsewhere, see `forward/neon.rs`) against the
/// always-compiled `matvec_q8_scalar` called directly with the identical
/// quantized input, for a shape drawn from this fixture (`lin_output_dim x
/// hidden` = 64x64, two Q8_0 blocks per row). On aarch64 this is a genuine
/// NEON-vs-scalar comparison; on every other target `matmul_q8_neon_into`
/// already calls the same scalar kernel, so the assertion holds trivially
/// there.
///
/// Mutation check: perturbing one packed weight byte on a copy fed only to
/// the scalar call (not to `matmul_q8_neon_into`) must turn this red, since
/// the whole point is that the two argument sets are normally identical.
#[test]
fn q8_neon_matmul_matches_scalar_kernel_directly() {
    use crate::forward::neon::{matvec_q8_scalar, quantize_vec_q8_into};

    let hidden = 64usize;
    let out_dim = 64usize; // matches lin_output_dim x hidden in this fixture
    let mut seed: u64 = 0x1234_5678_9abc_def0;
    let mut next_f32 = || -> f32 {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let x: Vec<f32> = (0..hidden).map(|_| next_f32()).collect();
    let w_floats: Vec<f32> = (0..out_dim * hidden).map(|_| next_f32()).collect();
    let packed = pack_weights_q8(&w_floats, out_dim, hidden).unwrap();

    let mut x_q_scratch: Vec<i8> = Vec::new();
    let x_scale = quantize_vec_q8_into(&x, &mut x_q_scratch);

    let mut dispatched = vec![0.0f32; out_dim];
    matmul_q8_neon_into(
        &x,
        &packed,
        out_dim,
        hidden,
        &mut dispatched,
        &mut x_q_scratch.clone(),
    );

    let mut scalar_direct = vec![0.0f32; out_dim];
    matvec_q8_scalar(
        &x_q_scratch,
        x_scale,
        &packed,
        out_dim,
        hidden,
        &mut scalar_direct,
    );

    assert_eq!(
        dispatched, scalar_direct,
        "matmul_q8_neon_into's dispatched kernel must be bit-identical to the \
             scalar kernel called directly with the same quantized input -- the \
             per-block int8 dot product is exact and both kernels apply the same \
             single (dot as f32) * block_scale * x_scale multiply per block in the \
             same accumulation order"
    );
}

/// Writer path for the four `generate_q8_neon` goldens below. Not run by
/// default; see the module-level comment above
/// `make_nonzero_q8_neon_generate_fixture`.
#[test]
#[ignore = "golden writer: run explicitly to (re)print the pinned values"]
fn print_q8_neon_generate_goldens() {
    let (cfg, model, rope) = make_nonzero_q8_neon_generate_fixture();
    let tokenizer = nonzero_fixture_tokenizer_q8_neon();

    let greedy_cfg = GenerateConfig {
        max_new_tokens: 6,
        temperature: 0.0,
        seed: Some(7),
        ..Default::default()
    };
    let greedy = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "world", &greedy_cfg)
        .expect("greedy golden must succeed");
    println!("GREEDY token_ids={:?}", greedy.token_ids);
    println!(
        "GREEDY stopped={} stop_reason={:?}",
        greedy.stopped, greedy.stop_reason
    );
    println!("GREEDY text={:?}", greedy.text);

    let seeded_cfg = GenerateConfig {
        max_new_tokens: 6,
        temperature: 0.8,
        top_k: 5,
        top_p: 0.9,
        repetition_penalty: 1.1,
        seed: Some(1234),
        ..Default::default()
    };
    let seeded = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "world", &seeded_cfg)
        .expect("seeded golden must succeed");
    println!("SEEDED token_ids={:?}", seeded.token_ids);
    println!(
        "SEEDED stopped={} stop_reason={:?}",
        seeded.stopped, seeded.stop_reason
    );
    println!("SEEDED text={:?}", seeded.text);

    assert!(
        greedy.token_ids.len() >= 2,
        "need at least 2 greedy tokens to derive a mid-stream stop token"
    );
    let stop_tok = greedy.token_ids[1];
    let stop_cfg = GenerateConfig {
        max_new_tokens: 6,
        temperature: 0.0,
        seed: Some(7),
        stop_token_ids: vec![stop_tok],
        ..Default::default()
    };
    let stopped = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "world", &stop_cfg)
        .expect("stop-token golden must succeed");
    println!("STOP stop_tok={stop_tok}");
    println!("STOP token_ids={:?}", stopped.token_ids);
    println!(
        "STOP stopped={} stop_reason={:?}",
        stopped.stopped, stopped.stop_reason
    );
    println!("STOP text={:?}", stopped.text);

    let one_tok_cfg = GenerateConfig {
        max_new_tokens: 1,
        temperature: 0.0,
        seed: Some(7),
        ..Default::default()
    };
    let one_tok = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "world", &one_tok_cfg)
        .expect("max_new_tokens=1 golden must succeed");
    println!("ONE token_ids={:?}", one_tok.token_ids);
    println!(
        "ONE stopped={} stop_reason={:?}",
        one_tok.stopped, one_tok.stop_reason
    );
    println!("ONE text={:?}", one_tok.text);
}

/// `generate_q8_neon` greedy decode (`temperature: 0.0`). Values captured
/// from `print_q8_neon_generate_goldens`; identical to `generate_f16`'s and
/// `generate_q8`'s greedy goldens for this fixture.
///
/// Mutation check: changing `max_new_tokens: 6` to `5` must turn
/// `token_ids` red (truncated sequence). `seed` is not the mutated field
/// here: at `temperature: 0.0` the sampler takes the degenerate-temperature
/// argmax short-circuit before the RNG is ever consulted, so a seed change
/// on a greedy golden cannot move the output.
#[test]
fn generate_q8_neon_greedy_golden() {
    let (cfg, model, rope) = make_nonzero_q8_neon_generate_fixture();
    let tokenizer = nonzero_fixture_tokenizer_q8_neon();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 6,
        temperature: 0.0,
        seed: Some(7),
        ..Default::default()
    };
    let out = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "world", &gen_cfg)
        .expect("greedy golden must succeed");
    assert_eq!(
        out.prompt_tokens, 5,
        "the prompt must span several prefill positions"
    );
    assert_eq!(out.token_ids, vec![36, 80, 112, 62, 103, 86]);
    assert!(!out.stopped);
    assert_eq!(
        out.stop_reason,
        Some(crate::stop_reason::StopReason::Length)
    );
    assert_eq!(out.text, "aCbkbQa2bHbq");
}

/// `generate_q8_neon` with temperature/top_k/top_p/repetition_penalty all
/// set (seeded sampling, not greedy).
///
/// Mutation check: changing `top_k: 5` to `top_k: 1` must turn `token_ids`
/// red against this golden.
#[test]
fn generate_q8_neon_seeded_sampling_golden() {
    let (cfg, model, rope) = make_nonzero_q8_neon_generate_fixture();
    let tokenizer = nonzero_fixture_tokenizer_q8_neon();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 6,
        temperature: 0.8,
        top_k: 5,
        top_p: 0.9,
        repetition_penalty: 1.1,
        seed: Some(1234),
        ..Default::default()
    };
    let out = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "world", &gen_cfg)
        .expect("seeded golden must succeed");
    assert_eq!(out.token_ids, vec![36, 80, 103, 112, 87, 86]);
    assert!(!out.stopped);
    assert_eq!(
        out.stop_reason,
        Some(crate::stop_reason::StopReason::Length)
    );
    assert_eq!(out.text, "aCbkbHbQbrbq");
}

/// `generate_q8_neon` stopping on a `stop_token_ids` match. `stop_tok = 80`
/// is the greedy golden's own second token (see
/// `generate_q8_neon_greedy_golden`), so this proves an actual decode-loop
/// stop.
///
/// Mutation check: removing `stop_token_ids` must turn
/// `stopped`/`stop_reason`/`token_ids.len()` red against this golden.
#[test]
fn generate_q8_neon_stop_token_golden() {
    let (cfg, model, rope) = make_nonzero_q8_neon_generate_fixture();
    let tokenizer = nonzero_fixture_tokenizer_q8_neon();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 6,
        temperature: 0.0,
        seed: Some(7),
        stop_token_ids: vec![80],
        ..Default::default()
    };
    let out = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "world", &gen_cfg)
        .expect("stop-token golden must succeed");
    assert_eq!(out.token_ids, vec![36]);
    assert!(out.stopped);
    assert_eq!(out.stop_reason, Some(crate::stop_reason::StopReason::Eos));
    assert_eq!(out.text, "aC");
}

/// `generate_q8_neon` with `max_new_tokens: 1`, the boundary case where the
/// decode loop must stop after exactly one token.
///
/// Mutation check: changing `max_new_tokens` to `2` must turn
/// `token_ids.len()` red against this golden.
#[test]
fn generate_q8_neon_max_new_tokens_one_golden() {
    let (cfg, model, rope) = make_nonzero_q8_neon_generate_fixture();
    let tokenizer = nonzero_fixture_tokenizer_q8_neon();
    let gen_cfg = GenerateConfig {
        max_new_tokens: 1,
        temperature: 0.0,
        seed: Some(7),
        ..Default::default()
    };
    let out = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "world", &gen_cfg)
        .expect("max_new_tokens=1 golden must succeed");
    assert_eq!(out.token_ids, vec![36]);
    assert!(!out.stopped);
    assert_eq!(
        out.stop_reason,
        Some(crate::stop_reason::StopReason::Length)
    );
    assert_eq!(out.text, "aC");
}

/// The fixture the goldens above use, bundled for the shared standalone-session checks.
struct SessionFixture {
    cfg: Qwen35Config,
    model: Q8NeonModel,
    rope: RopeTable,
    tokenizer: BpeTokenizer,
}

impl SessionFixture {
    fn new() -> Self {
        let (cfg, model, rope) = make_nonzero_q8_neon_generate_fixture();
        Self {
            cfg,
            model,
            rope,
            tokenizer: nonzero_fixture_tokenizer_q8_neon(),
        }
    }

    fn standalone(&self) -> StandaloneWeights<'_> {
        StandaloneWeights::Q8Neon(&self.model)
    }

    fn public(&self, gen_cfg: &GenerateConfig) -> Result<GenerateOutput, InferenceError> {
        generate_q8_neon(
            &self.model,
            &self.cfg,
            &self.tokenizer,
            &self.rope,
            "world",
            gen_cfg,
        )
    }

    fn forward(
        &self,
        token_id: u32,
        position: usize,
        gdn_states: &mut [GatedDeltaNetState],
        kv_cache: &mut KvCache,
        scratch: &mut ForwardScratch,
    ) -> Result<(), InferenceError> {
        forward_step_q8_neon(
            &self.model,
            &self.cfg,
            &self.rope,
            token_id,
            position,
            gdn_states,
            kv_cache,
            scratch,
        );
        Ok(())
    }
}

#[test]
fn standalone_session_matches_recorded_outputs_deterministic_cases() {
    let f = SessionFixture::new();
    for (name, gen_cfg, recorded) in parity::deterministic_cases() {
        parity::assert_matches_recorded(
            &|c| f.public(c),
            f.standalone(),
            &f.cfg,
            &f.tokenizer,
            &f.rope,
            "world",
            name,
            &gen_cfg,
            &recorded,
        );
    }
}

#[test]
fn standalone_session_matches_recorded_outputs_seeded() {
    let f = SessionFixture::new();
    parity::assert_matches_recorded(
        &|c| f.public(c),
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        "seeded",
        &parity::seeded_case(),
        &parity::seeded_recorded(),
    );
}

#[test]
fn standalone_session_logits_match_reference_replay() {
    let f = SessionFixture::new();
    parity::assert_logits_replay(
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        &parity::greedy_case(),
        4,
        &|t, p, g, k, s| f.forward(t, p, g, k, s),
    );
}

#[test]
fn standalone_session_rejects_a_stale_prediction() {
    let f = SessionFixture::new();
    parity::assert_rejects_stale_prediction(
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        &parity::greedy_case(),
    );
}

#[test]
fn standalone_session_cancellation_precedes_state_change() {
    let f = SessionFixture::new();
    parity::assert_cancellation_precedes_state_change(
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        &parity::greedy_case(),
    );
}

#[test]
fn standalone_session_refuses_undeclared_controls() {
    let f = SessionFixture::new();
    parity::assert_refuses_undeclared_controls(
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        &parity::greedy_case(),
    );
}

#[test]
fn standalone_session_select_refuses_config_controls() {
    let f = SessionFixture::new();
    parity::assert_select_refuses_config_controls(
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        &parity::seeded_case(),
    );
}

#[test]
fn standalone_session_finish_invalidates_the_live_prediction() {
    let f = SessionFixture::new();
    parity::assert_finish_invalidates_the_live_prediction(
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        &parity::greedy_case(),
    );
}

#[test]
fn standalone_session_refuses_a_second_prefill() {
    let f = SessionFixture::new();
    parity::assert_second_prefill_is_refused(
        f.standalone(),
        &f.cfg,
        &f.tokenizer,
        &f.rope,
        "world",
        &parity::greedy_case(),
    );
}

#[test]
fn test_all_projection_dims_are_multiples_of_32() {
    // Q8_0 requires K to be a multiple of 32. Verify all our dims qualify.
    let cfg = Qwen35Config::qwen35_2b();
    let dims_to_check = [
        ("hidden_size", cfg.hidden_size),
        ("intermediate_size", cfg.intermediate_size),
        ("full_q_dim", cfg.full_q_dim()),
        ("full_kv_dim", cfg.full_kv_dim()),
        ("2*full_q_dim", 2 * cfg.full_q_dim()),
        ("linear_qkv_dim", cfg.linear_qkv_dim()),
        ("linear_output_dim", cfg.linear_output_dim()),
    ];
    for (name, dim) in &dims_to_check {
        assert_eq!(
            dim % 32,
            0,
            "{name} = {dim} is not a multiple of 32 (required for Q8_0)"
        );
    }
}

/// Fail-closed: a non-finite cached K (corrupt activation) drives a score
/// row non-finite; without the `sum_exp > 0.0` else-branch the NaN inv_sum
/// poisons `scratch.context`. NEON sibling of
/// `test_full_attn_step_q8_nan_score_fails_closed` in cpu_q8.rs, with the
/// NaN injected into a prior cached K position.
#[test]
fn test_full_attn_step_q8_neon_nan_score_fails_closed() {
    let head_dim: usize = 32;
    let hidden: usize = 64;
    let q_dim = head_dim; // 1 q head
    let kv_dim = head_dim; // 1 kv head

    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 2,
        vocab_size: 128,
        intermediate_size: 128,
        rms_norm_eps: 1e-6,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.5,
        rope_parameters: None,
        linear_num_key_heads: 2,
        linear_num_value_heads: Some(2),
        linear_key_head_dim: 32,
        linear_value_head_dim: 32,
        linear_conv_kernel_dim: 4,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: 2,
        layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
        layer_mask: vec![true; 2],
        eos_token_id: 127,
        max_position_embeddings: 512,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };
    let rope = RopeTable::new(cfg.rope_dim(), 512, cfg.rope_theta);

    // Identity q/k/v projections so q_buf, k_buf, v_buf are nonzero finite.
    let identity = |rows: usize| {
        let mut mat = vec![0.0f32; rows * hidden];
        for j in 0..rows.min(hidden) {
            mat[j * hidden + j] = 1.0;
        }
        // Identity matrices contain only 0.0 and 1.0 — always finite.
        pack_weights_q8(&mat, rows, hidden).unwrap()
    };
    let weights = Q8NeonFullAttnWeights {
        q_proj_packed: identity(2 * q_dim),
        q_proj_rows: 2 * q_dim,
        q_proj_cols: hidden,
        k_proj_packed: identity(kv_dim),
        k_proj_rows: kv_dim,
        k_proj_cols: hidden,
        v_proj_packed: identity(kv_dim),
        v_proj_rows: kv_dim,
        v_proj_cols: hidden,
        o_proj_packed: zero_packed(hidden, q_dim),
        o_proj_rows: hidden,
        o_proj_cols: q_dim,
        q_norm: vec![0.0f32; head_dim],
        k_norm: vec![0.0f32; head_dim],
    };

    let input: Vec<f32> = (0..hidden).map(|i| (i as f32 + 1.0) * 0.05).collect();
    let mut scratch = ForwardScratch::new();
    scratch.ensure_capacity(&cfg, 4);
    let mut kv_cache = KvCache::new(1);
    kv_cache.reserve(4, kv_dim);

    // Position 0: real finite K/V appended at cache index 0.
    scratch.attn_out[..hidden].copy_from_slice(&input);
    full_attention_step_q8_neon(
        &weights,
        0,
        0,
        &mut kv_cache,
        &mut scratch,
        &cfg,
        &rope,
        hidden,
    );

    // ADR-080 C1 (#785) clean-row parity check: before any
    // poisoning, this well-formed single-position row (identity q/k/v
    // projections, non-zero input) must normalize through `finalize_row`,
    // not be incidentally swept into the fail-closed zero branch.
    assert_eq!(
        scratch.scores[0], 1.0,
        "clean single-position row must normalize to 1.0 before poisoning"
    );
    assert!(
        scratch.context[..q_dim].iter().any(|&v| v != 0.0),
        "clean row's context must be a real (non-degenerate) result"
    );

    // Corrupt the prior cached K (position 0) and advance to position 1 so
    // the score loop reads the NaN at t = 0.
    kv_cache.seq_len = 1;
    kv_cache.k[0][0] = f32::NAN;

    scratch.attn_out[..hidden].copy_from_slice(&input);
    full_attention_step_q8_neon(
        &weights,
        0,
        1,
        &mut kv_cache,
        &mut scratch,
        &cfg,
        &rope,
        hidden,
    );

    for (d, &v) in scratch.context[..q_dim].iter().enumerate() {
        assert!(
            v.is_finite(),
            "context[{d}] = {v} is non-finite; NEON Q8 softmax must fail \
                 closed on a NaN score row"
        );
    }
}

/// Fail-closed: an overflowing decay gate (large a_log, very negative
/// dt_bias) makes exp(a_log)=+inf and inf*softplus(...)=NaN without the
/// `.min(f32::MAX)` clamp, poisoning the recurrent state. NEON sibling of
/// `test_gdn_q8_decay_gate_overflow_fails_closed` in cpu_q8.rs.
#[test]
fn test_gdn_step_q8_neon_decay_gate_overflow_fails_closed() {
    let cfg = Qwen35Config::qwen35_2b();
    let hidden = cfg.hidden_size;
    let qkv_dim = cfg.linear_qkv_dim();
    let output_dim = cfg.linear_output_dim();
    let num_heads = cfg.linear_num_key_heads;

    let mut weights = Q8NeonGdnWeights {
        in_proj_qkv_packed: zero_packed(qkv_dim, hidden),
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z_packed: zero_packed(output_dim, hidden),
        in_proj_z_rows: output_dim,
        in_proj_z_cols: hidden,
        in_proj_b_packed: zero_packed(num_heads, hidden),
        in_proj_b_rows: num_heads,
        in_proj_b_cols: hidden,
        in_proj_a_packed: zero_packed(num_heads, hidden),
        in_proj_a_rows: num_heads,
        in_proj_a_cols: hidden,
        out_proj_packed: zero_packed(hidden, output_dim),
        out_proj_rows: hidden,
        out_proj_cols: output_dim,
        a_log: vec![0.0; num_heads],
        dt_bias: vec![0.0; num_heads],
        conv1d_weight: vec![0.0; qkv_dim * cfg.linear_conv_kernel_dim],
        conv_dim: qkv_dim,
        kernel_size: cfg.linear_conv_kernel_dim,
        norm_weight: vec![0.0; cfg.linear_value_head_dim],
    };
    // Overflow the decay gate on head 0: exp(100) -> +inf, and exp(-200) underflows to
    // 0.0 in f32, so softplus is exactly 0.0. Zero in_proj_a makes alpha exactly 0, so
    // softplus sees dt_bias alone. A nonzero softplus makes inf * sp = inf, which never
    // produces the NaN this test guards.
    weights.a_log[0] = 100.0;
    weights.dt_bias[0] = -200.0;
    assert_eq!(softplus(weights.dt_bias[0]), 0.0);

    let mut state = GatedDeltaNetState::new(&cfg);
    let input = vec![0.05f32; hidden];
    let mut output = vec![0.0f32; hidden];
    let mut gdn_scratch = GatedDeltaNetFusedScratch::default();
    let mut x_q_scratch = Vec::new();

    gdn_step_q8_neon(
        &input,
        &mut state,
        &weights,
        &cfg,
        &mut gdn_scratch,
        &mut x_q_scratch,
        &mut output,
    );

    for (i, &v) in output[..hidden].iter().enumerate() {
        assert!(
            v.is_finite(),
            "output[{i}] = {v} non-finite; decay gate must not overflow to NaN"
        );
    }
    for (i, &v) in state.s_matrices.iter().enumerate() {
        assert!(
            v.is_finite(),
            "state.s_matrices[{i}] = {v} non-finite; decay gate overflow poisoned state"
        );
    }
}

/// generate_q8_neon must reject a request whose prompt + max_new_tokens
/// exceeds the RoPE context window before allocating caches or
/// dereferencing weights. NEON sibling of
/// `test_generate_q8_rejects_context_overflow` in cpu_q8.rs.
#[test]
fn test_generate_q8_neon_rejects_context_overflow() {
    use std::collections::HashMap;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let model = Q8NeonModel {
        embed_tokens: vec![],
        final_norm: vec![],
        lm_head_packed: vec![],
        lm_head_rows: 0,
        lm_head_cols: 0,
        layers: vec![],
    };
    let gen_cfg = GenerateConfig {
        max_new_tokens: usize::MAX,
        ..Default::default()
    };

    let err = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "hello", &gen_cfg)
        .expect_err("expected context-window rejection");
    let msg = format!("{err}");
    assert!(
        msg.contains("context window"),
        "error should mention context window, got: {msg}"
    );
}

/// `generate_q8_neon` must stop on a token in `stop_token_ids` even when
/// that token differs from `eos_token_id`.
///
/// Setup: hidden=64 (must be multiple of 32 for Q8_0), vocab=64, all-zero
/// weights → logits all 0 → greedy always picks token 0.
/// Config has eos_token_id=5 (not 0) and stop_token_ids=[0]; `should_stop_token`
/// must check membership in `stop_token_ids`, not just equality with
/// `eos_token_id`, so generation stops on token 0 without emitting it.
#[test]
fn test_generate_q8_neon_honors_stop_token_ids() {
    use std::collections::HashMap;

    // Q8_0 requires hidden % 32 == 0; use 64.
    let hidden = 64usize;
    let vocab = 64usize;

    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 0,
        vocab_size: vocab,
        intermediate_size: 64,
        rms_norm_eps: 1e-6,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 64,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.5,
        rope_parameters: None,
        linear_num_key_heads: 1,
        linear_num_value_heads: Some(1),
        linear_key_head_dim: 64,
        linear_value_head_dim: 64,
        linear_conv_kernel_dim: 64,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: 2,
        layer_types: vec![],
        layer_mask: vec![],
        // eos is 5 so token 0 is NOT eos — stop_token_ids=[0] is the
        // distinct stop path we are testing.
        eos_token_id: 5,
        max_position_embeddings: 512,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    // All-zero packed weights: scale=0, all i8=0 → logits all 0 → greedy picks 0.
    let model = Q8NeonModel {
        embed_tokens: vec![0.0f32; vocab * hidden],
        final_norm: vec![0.0f32; hidden],
        lm_head_packed: zero_packed(vocab, hidden),
        lm_head_rows: vocab,
        lm_head_cols: hidden,
        layers: vec![],
    };

    // rope_dim = head_dim * partial_rotary_factor = 64 * 0.5 = 32 (multiple of 32 ✓)
    let rope = RopeTable::new(32, 64, 10_000.0);

    let mut vocab_map: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab_map.insert((*c).to_string(), i as u32);
    }
    let merges = vec![("h".to_string(), "e".to_string())];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab_map, merges).unwrap();

    let gen_cfg = GenerateConfig {
        max_new_tokens: 4,
        stop_token_ids: vec![0], // token 0 is the stop signal, NOT eos (5)
        temperature: 0.0,        // greedy: all-zero logits always yield token 0
        ..Default::default()
    };

    let out = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "h", &gen_cfg)
        .expect("generate_q8_neon must succeed with valid stop_token_ids");

    assert_eq!(
        out.generated_tokens, 0,
        "generate_q8_neon must stop immediately when the first greedy token (0) \
             is in stop_token_ids — got {} generated tokens instead",
        out.generated_tokens
    );
}

/// `generate_q8_neon` must also stop when the stop token first appears in the
/// **decode loop**, not only at the post-prefill check.
///
/// Fixture: a "bouncing" 0-layer Q8 model.
///   embed[0] = [-1, 1, 0, ..., 0]  (64 dims, first two non-zero)
///   embed[1] = [ 1, 1, 0, ..., 0]
///   lm_head_packed = pack_weights_q8 of the same matrix (mimics tied weights)
///   final_norm gamma = [-2, 0, ..., 0]
///
/// The negative gamma at dim-0 flips that component after RMSNorm, creating a
/// deterministic bounce between tokens 0 and 1 (Q8 rounding preserves ordering):
///   from token 1: hidden ∝ [-c, +c, 0, …] → Q8 dot → logit[0] > logit[1]
///   from token 0: hidden ∝ [+c, +c, 0, …] → Q8 dot → logit[1] > logit[0]
///
/// Greedy sequence from prompt "e" (→ token 1, eos_token_id=5):
///   post-prefill  → token 0  (not stop=1)
///   decode step 1 → token 1  (stop) → decode-loop fires
///
/// Mutation proof: reverting ONLY the decode-loop `should_stop_token` check
/// (line 987 at time of writing) to `next_id == cfg.eos_token_id` leaves
/// token 1 uncaught (1 ≠ eos=5), the sequence continues, and generated_tokens
/// becomes ≥ 2 — failing the assertion below.
#[test]
fn test_generate_q8_neon_honors_stop_token_ids_decode_loop() {
    use std::collections::HashMap;

    // Q8_0 requires hidden % 32 == 0; use 64 (same as the post-prefill test).
    let hidden = 64usize;
    let vocab = 64usize;

    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 0,
        vocab_size: vocab,
        intermediate_size: 64,
        rms_norm_eps: 1e-6,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 64,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.5,
        rope_parameters: None,
        linear_num_key_heads: 1,
        linear_num_value_heads: Some(1),
        linear_key_head_dim: 64,
        linear_value_head_dim: 64,
        linear_conv_kernel_dim: 64,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: 2,
        layer_types: vec![],
        layer_mask: vec![],
        // eos=5 so the stop at token 1 is detectable only via stop_token_ids.
        eos_token_id: 5,
        max_position_embeddings: 512,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    // The negative gamma at dim-0 creates a "bounce" each decode step.
    // from embed[1]=[1,1,0,…]: hidden∝[-c,+c,0,…] → Q8 matmul → logit[0] > 0 wins
    // from embed[0]=[-1,1,0,…]: hidden∝[+c,+c,0,…] → Q8 matmul → logit[1] > 0 wins
    let mut embed_f32 = vec![0.0f32; vocab * hidden];
    embed_f32[0] = -1.0; // token 0, dim 0
    embed_f32[1] = 1.0; // token 0, dim 1
    embed_f32[hidden] = 1.0; // token 1, dim 0
    embed_f32[hidden + 1] = 1.0; // token 1, dim 1

    // Embed values are ±1.0 or 0.0 — all finite.
    let lm_head_packed = pack_weights_q8(&embed_f32, vocab, hidden).unwrap();

    let mut final_norm = vec![0.0f32; hidden];
    final_norm[0] = -2.0; // flip dim-0 sign after RMSNorm to drive the bounce

    let model = Q8NeonModel {
        embed_tokens: embed_f32,
        final_norm,
        lm_head_packed,
        lm_head_rows: vocab,
        lm_head_cols: hidden,
        layers: vec![],
    };

    let rope = RopeTable::new(32, 64, 10_000.0);

    let mut vocab_map: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab_map.insert((*c).to_string(), i as u32);
    }
    let merges = vec![("h".to_string(), "e".to_string())];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab_map, merges).unwrap();

    let gen_cfg = GenerateConfig {
        max_new_tokens: 10,
        stop_token_ids: vec![1], // stop on token 1 mid-decode-loop; eos_token_id=5≠1
        temperature: 0.0,        // greedy: deterministic bouncing sequence
        ..Default::default()
    };

    // Prompt "e" → token 1.
    // Post-prefill generates token 0 (not stop=1).
    // Decode step 1 generates token 1 → decode-loop stop fires.
    let out = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "e", &gen_cfg)
        .expect("generate_q8_neon must succeed");

    assert_eq!(
        out.generated_tokens, 1,
        "generate_q8_neon must stop at decode-loop step 1 when token 1 is in \
             stop_token_ids — got {} tokens; reverting only the decode-loop check \
             lets token 1 through and produces ≥ 2 tokens",
        out.generated_tokens
    );
    assert!(
        out.stopped,
        "generate_q8_neon must set stopped=true when the decode-loop stop fires"
    );
}

/// `generate_q8_neon` must reject an empty prompt with a typed
/// `Err(Inference("empty prompt"))` before any weight dereference or
/// state allocation (#856): this is one of the three CPU forward paths
/// the shared `check_prompt_not_empty` preflight unifies with the four
/// Metal paths, which used to silently accept an empty prompt and
/// return an empty `Ok`. See docs/generation-entrypoint-matrix.md row 2.
///
/// Mutation sensitivity: bypassing the shared preparation at this entry
/// point makes the function proceed past the guard with a
/// zero-length prompt, either panicking in the prefill/decode loop or
/// producing a non-`Inference` error — this assert fails either way.
#[test]
fn generate_q8_neon_rejects_empty_prompt() {
    use crate::error::InferenceError;
    use std::collections::HashMap;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let model = Q8NeonModel {
        embed_tokens: vec![],
        final_norm: vec![],
        lm_head_packed: vec![],
        lm_head_rows: 0,
        lm_head_cols: 0,
        layers: vec![],
    };
    let gen_cfg = GenerateConfig::default();

    let result = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "", &gen_cfg);
    assert!(
        matches!(result, Err(InferenceError::Inference(ref msg)) if msg.contains("empty prompt")),
        "generate_q8_neon must reject an empty prompt with Err(Inference(\"empty \
             prompt\")) (#856); got {result:?}"
    );
}

/// A standalone NEON driver can receive a tokenizer whose vocabulary is
/// larger than the supplied model config. The boundary ID `vocab_size`
/// must be rejected before `forward_step_q8_neon` slices the embedding
/// table.
///
/// Mutation sensitivity: removing the shared setup's
/// `check_prompt_ids_in_vocab` call lets this request reach the unchecked
/// embedding slice and panic instead of returning `InvalidInput`.
#[test]
fn generate_q8_neon_rejects_out_of_vocab_prompt_id() {
    use std::collections::HashMap;

    let hidden = 32usize;
    let vocab = 8usize;
    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 0,
        vocab_size: vocab,
        intermediate_size: 32,
        rms_norm_eps: 1e-6,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 32,
        rope_theta: 10_000.0,
        partial_rotary_factor: 0.5,
        rope_parameters: None,
        linear_num_key_heads: 1,
        linear_num_value_heads: Some(1),
        linear_key_head_dim: 32,
        linear_value_head_dim: 32,
        linear_conv_kernel_dim: 32,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        full_attention_interval: 2,
        layer_types: vec![],
        layer_mask: vec![],
        eos_token_id: 5,
        max_position_embeddings: 512,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };
    let model = Q8NeonModel {
        embed_tokens: vec![0.0f32; vocab * hidden],
        final_norm: vec![0.0f32; hidden],
        lm_head_packed: zero_packed(vocab, hidden),
        lm_head_rows: vocab,
        lm_head_cols: hidden,
        layers: vec![],
    };
    let rope = RopeTable::new(16, 64, 10_000.0);
    let mut vocab_map: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o", "w", "r", "d", "!"].iter().enumerate() {
        vocab_map.insert((*c).to_string(), i as u32);
    }
    vocab_map.insert("z".to_string(), cfg.vocab_size as u32);
    let mismatched_tokenizer = BpeTokenizer::from_vocab_and_merges(vocab_map, vec![])
        .expect("tokenizer with an OOV vocab entry still constructs");
    let gen_cfg = GenerateConfig {
        max_new_tokens: 1,
        ..Default::default()
    };

    let err = generate_q8_neon(&model, &cfg, &mismatched_tokenizer, &rope, "z", &gen_cfg)
        .expect_err("an out-of-vocabulary prompt token id must be rejected, not panic");
    assert!(
        matches!(err, crate::error::InferenceError::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

/// `generate_q8_neon` must reject a `GenerateConfig` that sets `logprobs` with
/// a typed `InvalidInput` error before sampling any token (#585): this wrapper
/// has not wired per-step log-probability capture into its decode loop.
///
/// Sibling of `generate_q8_neon_rejects_grammar_config_before_sampling` /
/// `..._stop_strings_..` / `..._reasoning_budget_..`, which already cover the
/// other three `StandaloneCpu`-contract guards; this one was the gap.
/// Mutation sensitivity: removing the `check_logprobs_not_set`
/// call in `GenerationEntryContract::StandaloneCpu`'s `validate_capabilities`
/// arm makes the function proceed past the guard and attempt to forward with
/// empty weights, producing a panic or a non-`InvalidInput` error — this
/// assert fails either way.
#[test]
fn generate_q8_neon_rejects_logprobs_config_before_sampling() {
    use crate::error::InferenceError;
    use std::collections::HashMap;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let model = Q8NeonModel {
        embed_tokens: vec![],
        final_norm: vec![],
        lm_head_packed: vec![],
        lm_head_rows: 0,
        lm_head_cols: 0,
        layers: vec![],
    };

    let gen_cfg = GenerateConfig {
        logprobs: Some(0),
        ..Default::default()
    };

    let result = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "hello", &gen_cfg);
    assert!(
        matches!(
            result,
            Err(InferenceError::InvalidInput(ref message))
                if message.contains("per-token logprobs are not yet supported")
        ),
        "generate_q8_neon must fail closed with the exact logprobs-unsupported \
             InvalidInput (#585); got {result:?}"
    );
}

/// `generate_q8_neon` must reject a `GenerateConfig` that sets `grammar` with
/// a typed `InvalidInput` error before sampling any token (#397/#398).
///
/// Before the fix, grammar was silently ignored and unconstrained output was
/// produced. The guard fires before any weight dereference or state allocation,
/// so empty weight vecs are sufficient.
///
/// Mutation sensitivity: removing the `check_grammar_not_set` call makes the
/// function proceed past the guard and attempt to forward with empty weights,
/// producing a panic or a non-`InvalidInput` error — this assert fails either way.
#[test]
fn generate_q8_neon_rejects_grammar_config_before_sampling() {
    use crate::error::InferenceError;
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::collections::HashMap;
    use std::sync::Arc;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let model = Q8NeonModel {
        embed_tokens: vec![],
        final_norm: vec![],
        lm_head_packed: vec![],
        lm_head_rows: 0,
        lm_head_cols: 0,
        layers: vec![],
    };

    let spec = GrammarSpec::Gbnf("root ::= \"t\" | \"f\"\n".to_string());
    let grammar_vocab = vec![b"t".to_vec(), b"f".to_vec()];
    let engine = GrammarEngine::new(&spec, grammar_vocab).expect("trivial grammar must compile");

    let gen_cfg = GenerateConfig {
        grammar: Some(Arc::new(engine)),
        ..Default::default()
    };

    let result = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "hello", &gen_cfg);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "generate_q8_neon must fail closed with InvalidInput when grammar is set \
             (#397/#398); got {result:?}"
    );
}

/// `generate_q8_neon` must reject a `GenerateConfig` that sets `stop_strings`
/// with a typed `InvalidInput` error before sampling any token (ADR-080 C3, #783).
///
/// Mutation sensitivity: removing the `check_stop_strings_not_set` call makes the
/// function proceed past the guard and attempt to forward with empty weights,
/// producing a panic or a non-`InvalidInput` error — this assert fails either way.
#[test]
fn generate_q8_neon_rejects_stop_strings_config_before_sampling() {
    use crate::error::InferenceError;
    use std::collections::HashMap;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let model = Q8NeonModel {
        embed_tokens: vec![],
        final_norm: vec![],
        lm_head_packed: vec![],
        lm_head_rows: 0,
        lm_head_cols: 0,
        layers: vec![],
    };

    let gen_cfg = GenerateConfig {
        stop_strings: vec!["</s>".to_string()],
        ..Default::default()
    };

    let result = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "hello", &gen_cfg);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "generate_q8_neon must fail closed with InvalidInput when stop_strings is set \
             (ADR-080 C3, #783); got {result:?}"
    );
}

/// `generate_q8_neon` must reject a `GenerateConfig` that sets `reasoning_budget`
/// with a typed `InvalidInput` error before sampling any token (ADR-080 C3, #783).
///
/// Mutation sensitivity: removing the `check_reasoning_budget_not_set` call makes
/// the function proceed past the guard and attempt to forward with empty weights,
/// producing a panic or a non-`InvalidInput` error — this assert fails either way.
#[test]
fn generate_q8_neon_rejects_reasoning_budget_config_before_sampling() {
    use crate::error::InferenceError;
    use std::collections::HashMap;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab.insert((*c).to_string(), i as u32);
    }
    let merges = vec![
        ("h".to_string(), "e".to_string()),
        ("he".to_string(), "l".to_string()),
    ];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let model = Q8NeonModel {
        embed_tokens: vec![],
        final_norm: vec![],
        lm_head_packed: vec![],
        lm_head_rows: 0,
        lm_head_cols: 0,
        layers: vec![],
    };

    let gen_cfg = GenerateConfig {
        reasoning_budget: Some(16),
        ..Default::default()
    };

    let result = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "hello", &gen_cfg);
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "generate_q8_neon must fail closed with InvalidInput when reasoning_budget is \
             set (ADR-080 C3, #783); got {result:?}"
    );
}

/// `generate_q8_neon` with `max_new_tokens == 0` must return zero generated
/// tokens without running prefill or sampling anything (#612, 3rd recurrence
/// of the #226/#456 bug class).
///
/// The guard fires before any weight dereference or state allocation, so
/// empty weight vecs are sufficient — mirrors the grammar-guard test above.
///
/// Mutation sensitivity: removing the `max_new_tokens == 0` early return
/// causes the function to run prefill (against empty weight vecs, which
/// would panic) and sample one token, so `generated_tokens` becomes 1
/// instead of 0 and the assertion below fails.
#[test]
fn generate_q8_neon_max_new_tokens_zero_returns_empty() {
    use std::collections::HashMap;

    let mut vocab: HashMap<String, u32> = HashMap::new();
    for (i, c) in ["h", "e", "l", "o"].iter().enumerate() {
        vocab.insert((*c).to_string(), i as u32);
    }
    let merges = vec![("h".to_string(), "e".to_string())];
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, merges).unwrap();

    let cfg = Qwen35Config::qwen35_2b();
    let rope = RopeTable::new(cfg.rope_dim(), 8, cfg.rope_theta);
    let model = Q8NeonModel {
        embed_tokens: vec![],
        final_norm: vec![],
        lm_head_packed: vec![],
        lm_head_rows: 0,
        lm_head_cols: 0,
        layers: vec![],
    };

    let gen_cfg = GenerateConfig {
        max_new_tokens: 0,
        ..Default::default()
    };

    let out = generate_q8_neon(&model, &cfg, &tokenizer, &rope, "hello", &gen_cfg)
        .expect("max_new_tokens=0 must succeed, not error");

    assert_eq!(
        out.generated_tokens, 0,
        "max_new_tokens=0 must produce zero generated tokens"
    );
    assert!(
        out.token_ids.is_empty(),
        "max_new_tokens=0 must produce an empty token list"
    );
    assert_eq!(
        out.stop_reason,
        Some(StopReason::Length),
        "max_new_tokens=0 must report stop_reason=Length"
    );
}
