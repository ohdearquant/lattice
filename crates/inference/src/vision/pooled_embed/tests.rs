use super::*;
use crate::model::qwen35_config::{LayerType, RopeParams, VisionModelConfig};
use crate::vision::checkpoint::{VisualBlockWeights, VisualMergerWeights};
use crate::weights::f16_weights::{
    F16AttentionWeights, F16CommonLayerWeights, F16FeedForwardWeights, F16FullAttentionLayerWeights,
};

/// Deterministic pseudo-random f32 fill (xorshift LCG), mirroring the
/// pattern already used by `qwen35_vit.rs`'s own unit tests, so weights
/// are non-trivial (not all-zero/identity) without needing a real
/// checkpoint.
fn pseudo_random_fill(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 0.2 - 0.1
    };
    (0..n).map(|_| next()).collect()
}

fn tiny_vision_cfg() -> VisionModelConfig {
    VisionModelConfig {
        depth: 1,
        hidden_size: 8,
        num_heads: 2,
        patch_size: 2,
        spatial_merge_size: 2,
        out_hidden_size: 8, // must equal decoder hidden_size below
        temporal_patch_size: 1,
        num_position_embeddings: 16,
        in_channels: 3,
        deepstack_visual_indexes: vec![],
        intermediate_size: None,
    }
}

fn tiny_vision_weights(vision_cfg: &VisionModelConfig, seed: u32) -> Qwen35VisionWeights {
    let hidden = vision_cfg.hidden_size;
    let patch_len = vision_cfg.in_channels
        * vision_cfg.temporal_patch_size
        * vision_cfg.patch_size
        * vision_cfg.patch_size;
    let mlp_dim = 2 * hidden;
    let merge_in = vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size * hidden;

    let block = VisualBlockWeights {
        qkv_weight: pseudo_random_fill(seed, 3 * hidden * hidden),
        qkv_bias: pseudo_random_fill(seed.wrapping_add(1), 3 * hidden),
        proj_weight: pseudo_random_fill(seed.wrapping_add(2), hidden * hidden),
        proj_bias: pseudo_random_fill(seed.wrapping_add(3), hidden),
        fc1_weight: pseudo_random_fill(seed.wrapping_add(4), mlp_dim * hidden),
        fc1_bias: pseudo_random_fill(seed.wrapping_add(5), mlp_dim),
        fc2_weight: pseudo_random_fill(seed.wrapping_add(6), hidden * mlp_dim),
        fc2_bias: pseudo_random_fill(seed.wrapping_add(7), hidden),
        norm1_weight: vec![1.0; hidden],
        norm1_bias: vec![0.0; hidden],
        norm2_weight: vec![1.0; hidden],
        norm2_bias: vec![0.0; hidden],
    };

    Qwen35VisionWeights {
        patch_embed_weight: pseudo_random_fill(seed.wrapping_add(8), hidden * patch_len),
        patch_embed_weight_shape: vec![
            hidden,
            vision_cfg.in_channels,
            vision_cfg.temporal_patch_size,
            vision_cfg.patch_size,
            vision_cfg.patch_size,
        ],
        patch_embed_bias: pseudo_random_fill(seed.wrapping_add(9), hidden),
        pos_embed: pseudo_random_fill(
            seed.wrapping_add(10),
            vision_cfg.num_position_embeddings * hidden,
        ),
        blocks: vec![block],
        merger: VisualMergerWeights {
            fc1_weight: pseudo_random_fill(seed.wrapping_add(11), merge_in * merge_in),
            fc1_bias: pseudo_random_fill(seed.wrapping_add(12), merge_in),
            fc2_weight: pseudo_random_fill(
                seed.wrapping_add(13),
                vision_cfg.out_hidden_size * merge_in,
            ),
            fc2_bias: pseudo_random_fill(seed.wrapping_add(14), vision_cfg.out_hidden_size),
            norm_weight: vec![1.0; hidden],
            norm_bias: vec![0.0; hidden],
        },
    }
}

/// A minimal one-layer full-attention decoder + vision config wired
/// together: small enough to hand-construct, non-trivial (pseudo-random)
/// projections so the pipeline is actually exercised end to end.
fn tiny_vlm_fixture() -> (Qwen35Config, F16ModelWeights, Qwen35VisionWeights) {
    let hidden = 8usize;
    let vocab = 16usize;
    let vision_cfg = tiny_vision_cfg();

    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 1,
        vocab_size: vocab,
        intermediate_size: 4,
        rms_norm_eps: 1e-6,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: hidden,
        rope_theta: 1.0e7,
        partial_rotary_factor: 1.0,
        rope_parameters: Some(RopeParams {
            rope_theta: 1.0e7,
            partial_rotary_factor: Some(1.0),
            mrope_section: Some(vec![2, 1, 1]),
            mrope_interleaved: Some(true),
        }),
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
        full_attention_interval: 1,
        layer_types: vec![LayerType::FullAttention],
        layer_mask: vec![true],
        eos_token_id: 999,
        max_position_embeddings: 512,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        quarot_rotation_seed: None,
        vision_config: Some(vision_cfg.clone()),
        image_token_id: Some(9),
        video_token_id: None,
        vision_start_token_id: Some(10),
        vision_end_token_id: Some(11),
    };

    let to_f16 = |src: &[f32]| -> Vec<u16> {
        let mut dst = vec![0u16; src.len()];
        crate::weights::f16_weights::f32_to_f16_slice(src, &mut dst);
        dst
    };

    let embed_tokens_f32 = pseudo_random_fill(777, vocab * hidden);
    // q_proj packs [Q, gate] interleaved per head (see
    // `full_attention_step_f16`), so its width is `2 * q_dim`, not `q_dim`.
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();
    let full_weights = F16FullAttentionLayerWeights {
        q_proj: to_f16(&pseudo_random_fill(101, 2 * q_dim * hidden)),
        k_proj: to_f16(&pseudo_random_fill(102, kv_dim * hidden)),
        v_proj: to_f16(&pseudo_random_fill(103, kv_dim * hidden)),
        o_proj: to_f16(&pseudo_random_fill(104, hidden * q_dim)),
        q_norm: vec![0.0f32; hidden],
        k_norm: vec![0.0f32; hidden],
    };
    let common = F16CommonLayerWeights {
        input_layernorm: vec![0.0f32; hidden],
        post_attention_layernorm: vec![0.0f32; hidden],
        ffn: F16FeedForwardWeights::Dense {
            gate_proj: to_f16(&vec![0.0f32; 4 * hidden]),
            up_proj: to_f16(&vec![0.0f32; 4 * hidden]),
            down_proj: to_f16(&vec![0.0f32; hidden * 4]),
        },
    };
    let weights = F16ModelWeights {
        embed_tokens: to_f16(&embed_tokens_f32),
        final_norm: vec![0.0f32; hidden],
        layers: vec![(F16AttentionWeights::Full(full_weights), common)],
    };

    let vision_weights = tiny_vision_weights(&vision_cfg, 555);
    (cfg, weights, vision_weights)
}

fn make_test_png(w: u32, h: u32, seed: u8) -> Vec<u8> {
    use image::RgbImage;
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = ((x + y + seed as u32) % 256) as u8;
            img.put_pixel(x, y, image::Rgb([v, v, v]));
        }
    }
    let mut buf = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .unwrap();
    buf
}

fn tiny_tokenizer() -> BpeTokenizer {
    let mut vocab_map = std::collections::HashMap::new();
    for (i, c) in ["describe", "this", "image"].iter().enumerate() {
        vocab_map.insert((*c).to_string(), i as u32);
    }
    BpeTokenizer::from_vocab_and_merges(vocab_map, vec![]).expect("tokenizer constructs")
}

#[test]
fn embed_image_from_bytes_is_deterministic() {
    let (cfg, weights, vision_weights) = tiny_vlm_fixture();
    let tokenizer = tiny_tokenizer();
    let png = make_test_png(8, 8, 0);

    let v1 = embed_image_from_bytes_f16(
        &weights,
        &cfg,
        &vision_weights,
        &tokenizer,
        &png,
        "describe this image",
        PoolingStrategy::MeanVisualTokens,
    )
    .expect("embed_image_from_bytes_f16 succeeds");
    let v2 = embed_image_from_bytes_f16(
        &weights,
        &cfg,
        &vision_weights,
        &tokenizer,
        &png,
        "describe this image",
        PoolingStrategy::MeanVisualTokens,
    )
    .expect("embed_image_from_bytes_f16 succeeds");

    assert_eq!(
        v1, v2,
        "same image + prompt must produce an identical vector"
    );
    assert_eq!(v1.len(), cfg.hidden_size);
    assert!(v1.iter().all(|x| x.is_finite()));
    let norm: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "expected unit norm, got {norm}");
}

#[test]
fn embed_image_from_bytes_discriminates_different_images() {
    let (cfg, weights, vision_weights) = tiny_vlm_fixture();
    let tokenizer = tiny_tokenizer();

    let png_a = make_test_png(8, 8, 0);
    let png_b = make_test_png(8, 8, 200);

    let emb_a = embed_image_from_bytes_f16(
        &weights,
        &cfg,
        &vision_weights,
        &tokenizer,
        &png_a,
        "describe this image",
        PoolingStrategy::MeanVisualTokens,
    )
    .expect("embed succeeds");
    let emb_b = embed_image_from_bytes_f16(
        &weights,
        &cfg,
        &vision_weights,
        &tokenizer,
        &png_b,
        "describe this image",
        PoolingStrategy::MeanVisualTokens,
    )
    .expect("embed succeeds");

    let dot: f32 = emb_a.iter().zip(&emb_b).map(|(x, y)| x * y).sum();
    assert!(
        dot < 0.999,
        "two different images must not collapse to near-identical embeddings, got cosine {dot}"
    );
}

#[test]
fn embed_image_from_bytes_rejects_non_vlm_checkpoint() {
    let (mut cfg, weights, vision_weights) = tiny_vlm_fixture();
    cfg.vision_config = None;
    let tokenizer = tiny_tokenizer();
    let png = make_test_png(8, 8, 0);

    let err = embed_image_from_bytes_f16(
        &weights,
        &cfg,
        &vision_weights,
        &tokenizer,
        &png,
        "describe this image",
        PoolingStrategy::MeanVisualTokens,
    )
    .expect_err("a checkpoint with no vision_config must be rejected");
    assert!(matches!(err, InferenceError::InvalidInput(_)));
}

#[test]
fn embed_image_from_bytes_rejects_misaligned_image() {
    let (cfg, weights, vision_weights) = tiny_vlm_fixture();
    let tokenizer = tiny_tokenizer();
    // factor = patch_size(2) * merge(2) = 4; 6 is not a multiple of 4.
    let png = make_test_png(6, 4, 0);

    let err = embed_image_from_bytes_f16(
        &weights,
        &cfg,
        &vision_weights,
        &tokenizer,
        &png,
        "describe this image",
        PoolingStrategy::MeanVisualTokens,
    )
    .expect_err("a misaligned image must be rejected, not panic");
    assert!(matches!(err, InferenceError::InvalidInput(_)));
}

/// Metal/CPU pooled-output parity for the tiny synthetic fixture (same
/// weights, same image, same prompt scaffold). This checks the wiring
/// added in this change (merger + prompt-scaffold + decoder pooling
/// reused identically by both entry points) rather than ViT-level
/// numerics. Two separate, differently-scoped artifacts already cover the
/// ViT forward itself, and neither is this test:
/// - `qwen35_vit_metal.rs`'s own `metal_forward_matches_cpu_reference_small_shapes`
///   test uses this *same* tiny geometry (and therefore also exercises the
///   CPU-fallback branch, since these shapes sit far below the Metal
///   dispatch threshold) at a tight `< 1e-4` max-abs-diff tolerance — this
///   test reuses that same tolerance for the same reason (tiny geometry,
///   deterministic fixture, no GPU-rounding slack to budget for).
/// - `tests/vision_s3b_vit_metal_gate_test.rs` is the one gated at
///   cosine > 0.999, on the *real* checkpoint geometry (depth 2,
///   hidden 768, 12 heads) against the committed real-image golden
///   fixture, where every GEMM clears the dispatch threshold and
///   genuinely runs on the GPU — that test, not this one and not
///   `metal_forward_matches_cpu_reference_small_shapes`, is what
///   validates real Metal dispatch.
///
/// Mutation-sensitivity: swapping the `vision_start`/`vision_end`
/// arguments in `embed_image_from_bytes_f16_metal`'s call to
/// `pool_from_pre_merger_hidden_states` (this file) reorders the
/// assembled `input_ids` scaffold on the Metal path only, changing which
/// decoder positions get pooled — the CPU path's output would stay
/// fixed, so `max_abs_diff` would jump well past `1e-4` and this test
/// would fail.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn embed_image_from_bytes_metal_matches_cpu_reference() {
    let (cfg, weights, vision_weights) = tiny_vlm_fixture();
    let tokenizer = tiny_tokenizer();
    let png = make_test_png(8, 8, 0);

    let cpu = embed_image_from_bytes_f16(
        &weights,
        &cfg,
        &vision_weights,
        &tokenizer,
        &png,
        "describe this image",
        PoolingStrategy::MeanVisualTokens,
    )
    .expect("cpu embed succeeds");
    let metal = embed_image_from_bytes_f16_metal(
        &weights,
        &cfg,
        &vision_weights,
        &tokenizer,
        &png,
        "describe this image",
        PoolingStrategy::MeanVisualTokens,
    )
    .expect("metal embed succeeds");

    assert_eq!(cpu.len(), metal.len());
    let max_abs_diff = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff < 1e-4,
        "cpu vs metal pooled embedding diverged: max_abs_diff={max_abs_diff}"
    );
}

/// Off this cfg gate (no macOS, or `metal-gpu` off), the Metal entry
/// must fail closed with a distinct error rather than silently running
/// the CPU ViT forward.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
#[test]
fn embed_image_from_bytes_metal_fails_closed_without_metal_gpu() {
    let (cfg, weights, vision_weights) = tiny_vlm_fixture();
    let tokenizer = tiny_tokenizer();
    let png = make_test_png(8, 8, 0);

    let err = embed_image_from_bytes_f16_metal(
        &weights,
        &cfg,
        &vision_weights,
        &tokenizer,
        &png,
        "describe this image",
        PoolingStrategy::MeanVisualTokens,
    )
    .expect_err("Metal entry must fail without the metal-gpu feature, not silently run CPU");
    assert!(
        matches!(err, InferenceError::UnsupportedModel(_)),
        "expected a distinct UnsupportedModel error, got {err:?}"
    );
}
