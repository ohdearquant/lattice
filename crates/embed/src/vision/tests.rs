use super::*;
use lattice_inference::model::qwen35_config::{LayerType, RopeParams, VisionModelConfig};
use lattice_inference::vision::checkpoint::{
    VisualBlockWeights, VisualMergerWeights, resolve_qwen35_single_decoder_safetensors,
};
use lattice_inference::weights::f16_weights::{
    F16AttentionWeights, F16CommonLayerWeights, F16FeedForwardWeights,
    F16FullAttentionLayerWeights, f32_to_f16_slice,
};

/// Deterministic pseudo-random f32 fill (xorshift LCG), matching the
/// fixture builder in `lattice_inference::vision::pooled_embed`'s own
/// unit tests, so this crate's wrapper is exercised against
/// non-trivial weights without needing a real checkpoint.
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
        in_channels: 1,
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
        f32_to_f16_slice(src, &mut dst);
        dst
    };

    let embed_tokens_f32 = pseudo_random_fill(777, vocab * hidden);
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

/// Single-character vocab: with no merges, a byte-level BPE tokenizer
/// falls back to per-character tokens, so (unlike `tiny_tokenizer`'s
/// whole-word entries) this actually produces non-empty `real_length`
/// output — required by `embed_text_vlm_f16`'s empty-prompt guard.
/// Mirrors the tokenizer `cpu_f16.rs`'s own `embed_text_vlm_f16` tests use.
fn single_char_tokenizer() -> BpeTokenizer {
    let mut vocab_map = std::collections::HashMap::new();
    for (i, c) in ["a", "b", "c"].iter().enumerate() {
        vocab_map.insert((*c).to_string(), i as u32);
    }
    BpeTokenizer::from_vocab_and_merges(vocab_map, vec![]).expect("tokenizer constructs")
}

fn tiny_vlm_checkpoint_shapes() -> Vec<(String, Vec<usize>)> {
    let hidden = 8usize;
    let mut shapes = vec![
        (
            "model.language_model.embed_tokens.weight".to_string(),
            vec![16, hidden],
        ),
        ("model.language_model.norm.weight".to_string(), vec![hidden]),
        (
            "model.language_model.layers.0.input_layernorm.weight".to_string(),
            vec![hidden],
        ),
        (
            "model.language_model.layers.0.post_attention_layernorm.weight".to_string(),
            vec![hidden],
        ),
        (
            "model.language_model.layers.0.mlp.gate_proj.weight".to_string(),
            vec![4, hidden],
        ),
        (
            "model.language_model.layers.0.mlp.up_proj.weight".to_string(),
            vec![4, hidden],
        ),
        (
            "model.language_model.layers.0.mlp.down_proj.weight".to_string(),
            vec![hidden, 4],
        ),
        (
            "model.language_model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![16, hidden],
        ),
        (
            "model.language_model.layers.0.self_attn.k_proj.weight".to_string(),
            vec![hidden, hidden],
        ),
        (
            "model.language_model.layers.0.self_attn.v_proj.weight".to_string(),
            vec![hidden, hidden],
        ),
        (
            "model.language_model.layers.0.self_attn.o_proj.weight".to_string(),
            vec![hidden, hidden],
        ),
        (
            "model.language_model.layers.0.self_attn.q_norm.weight".to_string(),
            vec![hidden],
        ),
        (
            "model.language_model.layers.0.self_attn.k_norm.weight".to_string(),
            vec![hidden],
        ),
        (
            "model.visual.patch_embed.proj.weight".to_string(),
            vec![hidden, 3, 1, 2, 2],
        ),
        (
            "model.visual.patch_embed.proj.bias".to_string(),
            vec![hidden],
        ),
        (
            "model.visual.pos_embed.weight".to_string(),
            vec![16, hidden],
        ),
        (
            "model.visual.merger.linear_fc1.weight".to_string(),
            vec![32, 32],
        ),
        ("model.visual.merger.linear_fc1.bias".to_string(), vec![32]),
        (
            "model.visual.merger.linear_fc2.weight".to_string(),
            vec![hidden, 32],
        ),
        (
            "model.visual.merger.linear_fc2.bias".to_string(),
            vec![hidden],
        ),
        ("model.visual.merger.norm.weight".to_string(), vec![hidden]),
        ("model.visual.merger.norm.bias".to_string(), vec![hidden]),
    ];
    for (suffix, shape) in [
        ("attn.qkv.weight", vec![24, hidden]),
        ("attn.qkv.bias", vec![24]),
        ("attn.proj.weight", vec![hidden, hidden]),
        ("attn.proj.bias", vec![hidden]),
        ("mlp.linear_fc1.weight", vec![32, hidden]),
        ("mlp.linear_fc1.bias", vec![32]),
        ("mlp.linear_fc2.weight", vec![hidden, 32]),
        ("mlp.linear_fc2.bias", vec![hidden]),
        ("norm1.weight", vec![hidden]),
        ("norm1.bias", vec![hidden]),
        ("norm2.weight", vec![hidden]),
        ("norm2.bias", vec![hidden]),
    ] {
        shapes.push((format!("model.visual.blocks.0.{suffix}"), shape));
    }
    shapes
}

fn write_f32_safetensors(path: &Path, shapes: &[(String, Vec<usize>)]) {
    write_f32_safetensors_with_offset(path, shapes, 0.0);
}

fn write_f32_safetensors_with_offset(path: &Path, shapes: &[(String, Vec<usize>)], offset: f32) {
    let mut header_parts = Vec::with_capacity(shapes.len());
    let mut data = Vec::new();
    for (i, (name, shape)) in shapes.iter().enumerate() {
        let start = data.len();
        let numel: usize = shape.iter().product();
        for _ in 0..numel {
            data.extend_from_slice(&(offset + (i + 1) as f32 / 100.0).to_le_bytes());
        }
        let end = data.len();
        let shape = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        header_parts.push(format!(
            r#""{name}":{{"dtype":"F32","shape":[{shape}],"data_offsets":[{start},{end}]}}"#
        ));
    }
    let header = format!("{{{}}}", header_parts.join(","));
    let mut bytes = Vec::with_capacity(8 + header.len() + data.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&data);
    std::fs::write(path, bytes).expect("write safetensors fixture");
}

fn write_tiny_tokenizer_json(dir: &Path) {
    let tokenizer = r#"{
            "model": {
                "type": "BPE",
                "vocab": {
                    "a": 0, "b": 1, "c": 2, "d": 3,
                    "e": 4, "f": 5, "g": 6, "h": 7,
                    "i": 8, "j": 9, "k": 10, "l": 11,
                    "m": 12, "n": 13, "o": 14, "p": 15
                },
                "merges": []
            }
        }"#;
    std::fs::write(dir.join("tokenizer.json"), tokenizer).expect("write tokenizer.json");
}

fn write_tiny_vlm_checkpoint(dir: &Path, indexed: bool) {
    write_tiny_vlm_checkpoint_with_max_position_embeddings(dir, indexed, 512);
}

fn write_tiny_vlm_checkpoint_with_max_position_embeddings(
    dir: &Path,
    indexed: bool,
    max_position_embeddings: usize,
) {
    let config = format!(
        r#"{{
            "text_config": {{
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "vocab_size": 16,
                "intermediate_size": 4,
                "rms_norm_eps": 0.000001,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 1.0,
                "rope_parameters": {{
                    "rope_theta": 10000000.0,
                    "partial_rotary_factor": 1.0,
                    "mrope_section": [2, 1, 1],
                    "mrope_interleaved": true
                }},
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "tie_word_embeddings": true,
                "full_attention_interval": 1,
                "layer_types": ["full_attention"],
                "layer_mask": [true],
                "eos_token_id": 15,
                "max_position_embeddings": {max_position_embeddings}
            }},
            "vision_config": {{
                "depth": 1,
                "hidden_size": 8,
                "num_heads": 2,
                "patch_size": 2,
                "spatial_merge_size": 2,
                "out_hidden_size": 8,
                "temporal_patch_size": 1,
                "num_position_embeddings": 16,
                "in_channels": 3,
                "deepstack_visual_indexes": []
            }},
            "image_token_id": 9,
            "vision_start_token_id": 10,
            "vision_end_token_id": 11,
            "tie_word_embeddings": true
        }}"#
    );
    std::fs::write(dir.join("config.json"), &config).expect("write config.json");
    write_tiny_tokenizer_json(dir);

    let shapes = tiny_vlm_checkpoint_shapes();
    let shard_name = if indexed {
        "model-00001-of-00001.safetensors"
    } else {
        "model.safetensors"
    };
    write_f32_safetensors(&dir.join(shard_name), &shapes);
    if indexed {
        let weight_map = shapes
            .iter()
            .map(|(name, _)| format!(r#""{name}":"{shard_name}""#))
            .collect::<Vec<_>>()
            .join(",");
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            format!(r#"{{"weight_map":{{{weight_map}}}}}"#),
        )
        .expect("write one-shard index");
    }
}

/// The embed-crate wrapper must return the exact same vector as calling
/// the raw inference-crate primitive directly: wiring adds no numerical
/// difference. This is the core claim of this module (a wire-through,
/// not a reimplementation).
#[test]
fn embed_image_matches_raw_inference_primitive() {
    let (cfg, weights, vision_weights) = tiny_vlm_fixture();
    let tokenizer = tiny_tokenizer();
    let png = make_test_png(8, 8, 0);

    let model = VisionEmbeddingModel::new(
        weights.clone(),
        cfg.clone(),
        vision_weights.clone(),
        tokenizer.clone(),
    );
    let via_wrapper = model
        .embed_image(
            &png,
            "describe this image",
            PoolingStrategy::MeanVisualTokens,
        )
        .expect("wrapper embed_image succeeds");

    let via_raw = embed_image_from_bytes_f16(
        &weights,
        &cfg,
        &vision_weights,
        &tokenizer,
        &png,
        "describe this image",
        PoolingStrategy::MeanVisualTokens,
    )
    .expect("raw primitive succeeds");

    assert_eq!(
        via_wrapper, via_raw,
        "embed-crate wrapper must return the identical vector to the raw primitive"
    );
}

/// Same wire-through claim as
/// `embed_image_matches_raw_inference_primitive`, for the Metal entry
/// point: the crate wrapper must add no numerical difference relative to
/// calling `embed_image_from_bytes_f16_metal` directly.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[test]
fn embed_image_metal_matches_raw_inference_primitive() {
    use lattice_inference::vision::embed_image_from_bytes_f16_metal;

    let (cfg, weights, vision_weights) = tiny_vlm_fixture();
    let tokenizer = tiny_tokenizer();
    let png = make_test_png(8, 8, 0);

    let model = VisionEmbeddingModel::new(
        weights.clone(),
        cfg.clone(),
        vision_weights.clone(),
        tokenizer.clone(),
    );
    let via_wrapper = model
        .embed_image_metal(
            &png,
            "describe this image",
            PoolingStrategy::MeanVisualTokens,
        )
        .expect("wrapper embed_image_metal succeeds");

    let via_raw = embed_image_from_bytes_f16_metal(
        &weights,
        &cfg,
        &vision_weights,
        &tokenizer,
        &png,
        "describe this image",
        PoolingStrategy::MeanVisualTokens,
    )
    .expect("raw metal primitive succeeds");

    assert_eq!(
        via_wrapper, via_raw,
        "embed-crate Metal wrapper must return the identical vector to the raw primitive"
    );
}

/// Off this cfg gate, the wrapper must surface the CPU/Metal
/// distinction the inference crate makes (`InferenceError::
/// UnsupportedModel`) as `EmbedError::InferenceFailed`, never silently
/// substituting `embed_image`'s CPU output.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
#[test]
fn embed_image_metal_fails_closed_without_metal_gpu() {
    let (cfg, weights, vision_weights) = tiny_vlm_fixture();
    let tokenizer = tiny_tokenizer();
    let png = make_test_png(8, 8, 0);
    let model = VisionEmbeddingModel::new(weights, cfg, vision_weights, tokenizer);

    let err = model
        .embed_image_metal(
            &png,
            "describe this image",
            PoolingStrategy::MeanVisualTokens,
        )
        .expect_err("Metal wrapper must fail without the metal-gpu feature");
    assert!(matches!(err, EmbedError::InferenceFailed(_)));
}

#[test]
fn embed_image_is_deterministic_and_normalized() {
    let (cfg, weights, vision_weights) = tiny_vlm_fixture();
    let tokenizer = tiny_tokenizer();
    let png = make_test_png(8, 8, 0);
    let model = VisionEmbeddingModel::new(weights, cfg.clone(), vision_weights, tokenizer);

    let v1 = model
        .embed_image(
            &png,
            "describe this image",
            PoolingStrategy::MeanVisualTokens,
        )
        .expect("embed succeeds");
    let v2 = model
        .embed_image(
            &png,
            "describe this image",
            PoolingStrategy::MeanVisualTokens,
        )
        .expect("embed succeeds");

    assert_eq!(
        v1, v2,
        "same image + prompt must produce an identical vector"
    );
    assert_eq!(v1.len(), model.dimensions());
    assert!(v1.iter().all(|x| x.is_finite()));
    let norm: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "expected unit norm, got {norm}");
}

#[test]
fn embed_image_rejects_non_vlm_checkpoint() {
    let (mut cfg, weights, vision_weights) = tiny_vlm_fixture();
    cfg.vision_config = None;
    let tokenizer = tiny_tokenizer();
    let png = make_test_png(8, 8, 0);
    let model = VisionEmbeddingModel::new(weights, cfg, vision_weights, tokenizer);

    let err = model
        .embed_image(
            &png,
            "describe this image",
            PoolingStrategy::MeanVisualTokens,
        )
        .expect_err("a checkpoint with no vision_config must be rejected");
    let msg = err.to_string();
    assert!(matches!(err, EmbedError::InvalidInput(_)));
    assert!(
        msg.contains("vision_config"),
        "error must name the missing field, got: {msg}"
    );
}

#[test]
fn embed_image_rejects_misaligned_image() {
    let (cfg, weights, vision_weights) = tiny_vlm_fixture();
    let tokenizer = tiny_tokenizer();
    // factor = patch_size(2) * merge(2) = 4; 6 is not a multiple of 4.
    let png = make_test_png(6, 4, 0);
    let model = VisionEmbeddingModel::new(weights, cfg, vision_weights, tokenizer);

    let err = model
        .embed_image(
            &png,
            "describe this image",
            PoolingStrategy::MeanVisualTokens,
        )
        .expect_err("a misaligned image must be rejected, not panic");
    assert!(matches!(err, EmbedError::InvalidInput(_)));
}

/// Issue #1408 sibling hole (`embed_image`): unlike the sibling
/// `lattice_inference::serve::embeddings::EmbeddingModel::
/// image_scaffold_token_count`, whose one production caller always
/// passes an empty `prompt` (so its own doc comment notes the
/// truncation-vs-window mismatch is not live there), `embed_image` is a
/// general public entry point where a caller can pass any `prompt`. An
/// 8x8 test image against this fixture's tiny vision config produces 4
/// merged patch pads, so its scaffold (`vision_start` + 4 pads +
/// `vision_end`) is 6 tokens -- already over a 5-token window with an
/// empty prompt, so this exercises the scaffold-only side of the
/// admission check without needing any text tokens at all.
#[test]
fn embed_image_rejects_when_scaffold_exceeds_context_window() {
    let (mut cfg, weights, vision_weights) = tiny_vlm_fixture();
    cfg.max_position_embeddings = 5;
    let tokenizer = tiny_tokenizer();
    let model = VisionEmbeddingModel::new(weights, cfg, vision_weights, tokenizer);
    let png = make_test_png(8, 8, 0);

    let err = model
        .embed_image(&png, "", PoolingStrategy::MeanVisualTokens)
        .expect_err("scaffold tokens alone exceeding the context window must be rejected");
    assert!(matches!(err, EmbedError::InvalidInput(_)), "got: {err:?}");
    let msg = err.to_string();
    assert!(
        msg.contains('6'),
        "error must name the scaffold token total (6), got: {msg}"
    );
    assert!(
        msg.contains('5'),
        "error must name the context window (5), got: {msg}"
    );
}

/// Sibling boundary case: scaffold tokens landing exactly at the context
/// window (empty prompt, so no text tokens add to the total) must still
/// embed successfully.
#[test]
fn embed_image_accepts_when_scaffold_exactly_fits_context_window() {
    let (mut cfg, weights, vision_weights) = tiny_vlm_fixture();
    cfg.max_position_embeddings = 6;
    let tokenizer = tiny_tokenizer();
    let model = VisionEmbeddingModel::new(weights, cfg, vision_weights, tokenizer);
    let png = make_test_png(8, 8, 0);

    model
        .embed_image(&png, "", PoolingStrategy::MeanVisualTokens)
        .expect("scaffold tokens exactly at the context window boundary must still embed");
}

/// Same property as `embed_image_rejects_when_scaffold_exceeds_context_window`,
/// for the Metal-dispatching sibling: the admission check must run (and
/// reject) before `embed_image_metal` ever attempts a Metal dispatch, so
/// this must reject identically on a build/platform with no Metal
/// support at all -- no GPU is touched by this test.
#[test]
fn embed_image_metal_rejects_when_scaffold_exceeds_context_window() {
    let (mut cfg, weights, vision_weights) = tiny_vlm_fixture();
    cfg.max_position_embeddings = 5;
    let tokenizer = tiny_tokenizer();
    let model = VisionEmbeddingModel::new(weights, cfg, vision_weights, tokenizer);
    let png = make_test_png(8, 8, 0);

    let err = model
        .embed_image_metal(&png, "", PoolingStrategy::MeanVisualTokens)
        .expect_err(
            "scaffold tokens alone exceeding the context window must be rejected \
                         before any Metal dispatch is attempted",
        );
    assert!(matches!(err, EmbedError::InvalidInput(_)), "got: {err:?}");
}

#[test]
fn embed_text_matches_raw_inference_primitive() {
    let (cfg, weights, vision_weights) = tiny_vlm_fixture();
    let tokenizer = single_char_tokenizer();
    let model = VisionEmbeddingModel::new(
        weights.clone(),
        cfg.clone(),
        vision_weights,
        tokenizer.clone(),
    );

    let via_wrapper = model
        .embed_text("abc", PoolingStrategy::LastToken)
        .expect("wrapper embed_text succeeds");
    let via_raw = lattice_inference::forward::cpu_f16::embed_text_vlm_f16(
        &weights,
        &cfg,
        &tokenizer,
        "abc",
        PoolingStrategy::LastToken,
    )
    .expect("raw primitive succeeds");

    assert_eq!(via_wrapper, via_raw);
}

/// Issue #1408 (library-path admission guard, `embed_text`): a prompt
/// whose PRE-truncation token count exceeds the checkpoint's context
/// window must be rejected as caller input, before the forward pass ever
/// runs. This fixture's tokenizer cap (4096, `capped_tokenizer` never
/// lowers it) sits well above the 1-token window, so before this guard
/// existed "abc" tokenized to its full 3 real ids and only failed
/// downstream, as `EmbedError::InferenceFailed`, once the assembled
/// request's length was compared against `max_position_embeddings` deep
/// inside the forward pass -- a caller-input problem misreported as a
/// runtime one. A checkpoint whose window instead sits between the
/// tokenizer's default and a wider `max_position_embeddings` (the
/// scenario `capped_tokenizer` exists for) would raise the tokenizer's
/// own cap to match, so the same over-window prompt would truncate
/// silently there and never reach any error at all. Reverting the
/// `check_prompt_admission` call in `embed_text` reddens this.
#[test]
fn embed_text_rejects_prompt_exceeding_context_window_before_truncation() {
    let (mut cfg, weights, vision_weights) = tiny_vlm_fixture();
    cfg.max_position_embeddings = 1;
    let tokenizer = single_char_tokenizer();
    let model = VisionEmbeddingModel::new(weights, cfg, vision_weights, tokenizer);

    let err = model
        .embed_text("abc", PoolingStrategy::LastToken)
        .expect_err(
            "a prompt whose pre-truncation length exceeds the context window must be rejected",
        );
    assert!(
        matches!(err, EmbedError::InvalidInput(_)),
        "over-window prompt must be rejected as caller input, not embedded from a \
             truncated prefix, got: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains('3'),
        "error must name the pre-truncation token count (3), got: {msg}"
    );
    assert!(
        msg.contains('1'),
        "error must name the context window (1), got: {msg}"
    );
}

/// Sibling boundary case: a prompt whose pre-truncation length lands
/// exactly at the context window must still embed successfully -- the
/// admission guard must not reject an at-or-below-window prompt.
#[test]
fn embed_text_accepts_prompt_exactly_at_context_window() {
    let (mut cfg, weights, vision_weights) = tiny_vlm_fixture();
    cfg.max_position_embeddings = 1;
    let tokenizer = single_char_tokenizer();
    let model = VisionEmbeddingModel::new(weights, cfg, vision_weights, tokenizer);

    model
        .embed_text("a", PoolingStrategy::LastToken)
        .expect("a prompt exactly at the context window boundary must still embed");
}

#[test]
fn resolve_single_shard_rejects_multi_shard_index() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let index_path = tmp.path().join("model.safetensors.index.json");
    std::fs::write(
        &index_path,
        r#"{"metadata":{},"weight_map":{"a":"shard1.safetensors","b":"shard2.safetensors"}}"#,
    )
    .expect("write index");

    let err = resolve_qwen35_single_decoder_safetensors(tmp.path())
        .expect_err("multi-shard must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("sharded across 2 files"), "got: {msg}");
}

#[test]
fn resolve_single_shard_rejects_missing_checkpoint() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let err = resolve_qwen35_single_decoder_safetensors(tmp.path())
        .expect_err("missing checkpoint must be rejected");
    assert!(matches!(err, InferenceError::ModelNotFound(_)));
}

#[test]
fn from_directory_without_checkpoint_reports_actionable_error() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let config_json = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../inference/tests/fixtures/qwen35_0_8b_config.json"
    ));
    std::fs::write(tmp.path().join("config.json"), config_json).expect("write config.json");
    write_tiny_tokenizer_json(tmp.path());

    let Err(err) = VisionEmbeddingModel::from_directory(tmp.path()) else {
        panic!("a directory with no checkpoint must be rejected")
    };
    assert!(matches!(err, EmbedError::ModelInitialization(_)));
    let msg = err.to_string();
    assert!(
        msg.contains("model.safetensors") && msg.contains("model.safetensors.index.json"),
        "error must name the supported checkpoint layouts, got: {msg}"
    );
}

#[test]
fn from_directory_rejects_missing_tokenizer_before_checkpoint_materialization() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_tiny_vlm_checkpoint(tmp.path(), false);
    std::fs::remove_file(tmp.path().join("tokenizer.json")).expect("remove tokenizer fixture");
    std::fs::write(tmp.path().join("model.safetensors"), u64::MAX.to_le_bytes())
        .expect("replace checkpoint with a corrupt header");

    let Err(err) = VisionEmbeddingModel::from_directory(tmp.path()) else {
        panic!("a checkpoint without tokenizer.json must be rejected")
    };
    assert!(matches!(err, EmbedError::ModelInitialization(_)));
    let msg = err.to_string();
    assert!(msg.contains("tokenizer.json"), "got: {msg}");
    assert!(
        !msg.contains("vision weights") && !msg.contains("decoder weights"),
        "tokenizer admission must fail before tensor materialization, got: {msg}"
    );
}

#[test]
fn from_directory_rejects_quantized_checkpoint_before_tensor_loading() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join("quantize_index.json"), b"not valid json")
        .expect("write quantized checkpoint sentinel");

    let Err(err) = VisionEmbeddingModel::from_directory(tmp.path()) else {
        panic!("the f16 pooled decoder loader must reject quantized checkpoints")
    };
    assert!(matches!(err, EmbedError::ModelInitialization(_)));
    let msg = err.to_string();
    assert!(msg.contains("quantize_index.json"), "got: {msg}");
    assert!(msg.contains("not supported"), "got: {msg}");
    assert!(
        !msg.contains("config.json"),
        "the unsupported file-set must fail before unrelated component loading, got: {msg}"
    );
}

#[test]
fn from_directory_loads_single_model_safetensors_without_index() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_tiny_vlm_checkpoint(tmp.path(), false);

    let model = VisionEmbeddingModel::from_directory(tmp.path())
        .expect("single-file VLM checkpoint must load without a synthetic index");
    assert_eq!(model.dimensions(), 8);
}

/// Issue #1408, sibling-crate half: a `tokenizer.json` with no explicit
/// cap parses through `BpeTokenizer::from_tokenizer_json` at its
/// constructor default (4096, `DEFAULT_BPE_MAX_SEQ_LEN` in
/// `lattice_inference::tokenizer::bpe`). A checkpoint whose
/// `max_position_embeddings` sits above that default (here 5000, still
/// under the 8192 ceiling `capped_tokenizer` applies) must come out of
/// `from_directory` with its tokenizer cap raised to match -- otherwise
/// prompts between 4097 and 5000 tokens are silently truncated before
/// the forward pass ever sees them, even though the checkpoint can
/// process them. Reverting `VisionEmbeddingModel::new`'s call to
/// `capped_tokenizer` reddens this (cap stays at the tokenizer's own
/// default, 4096, strictly below `max_position_embeddings`, 5000).
#[test]
fn from_directory_raises_tokenizer_cap_to_context_window_above_tokenizer_default() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_tiny_vlm_checkpoint_with_max_position_embeddings(tmp.path(), false, 5000);

    let model = VisionEmbeddingModel::from_directory(tmp.path())
        .expect("VLM checkpoint with a wide context window must load");
    assert_eq!(
        model.tokenizer.max_seq_len(),
        5000,
        "tokenizer cap must be raised to max_position_embeddings, not left at the \
             tokenizer's own 4096 default"
    );
}

#[test]
fn single_file_and_one_shard_index_produce_identical_image_embeddings() {
    let single = tempfile::tempdir().expect("single tempdir");
    let indexed = tempfile::tempdir().expect("indexed tempdir");
    write_tiny_vlm_checkpoint(single.path(), false);
    write_tiny_vlm_checkpoint(indexed.path(), true);

    let single_model = VisionEmbeddingModel::from_directory(single.path())
        .expect("single-file VLM checkpoint loads");
    let indexed_model = VisionEmbeddingModel::from_directory(indexed.path())
        .expect("equivalent one-shard indexed VLM checkpoint loads");
    let image = make_test_png(4, 4, 17);
    let from_single = single_model
        .embed_image(&image, "a", PoolingStrategy::MeanVisualTokens)
        .expect("single-file image embedding succeeds");
    let from_index = indexed_model
        .embed_image(&image, "a", PoolingStrategy::MeanVisualTokens)
        .expect("indexed image embedding succeeds");

    assert_eq!(
        from_single, from_index,
        "equivalent single-file and indexed layouts must produce parity embeddings"
    );
}

#[test]
fn from_directory_rejects_index_map_that_contradicts_opened_shard_header() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_tiny_vlm_checkpoint(tmp.path(), true);
    std::fs::write(
        tmp.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"not.a.real.tensor":"model-00001-of-00001.safetensors"}}"#,
    )
    .expect("replace index with contradictory weight_map");

    let Err(err) = VisionEmbeddingModel::from_directory(tmp.path()) else {
        panic!("an authoritative index that omits the physical tensors must be rejected")
    };
    let msg = err.to_string();
    assert!(
        msg.contains("weight_map/header inventory mismatch"),
        "got: {msg}"
    );
}

#[cfg(unix)]
#[test]
fn from_directory_binds_visual_and_decoder_weights_across_path_replacement() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_tiny_vlm_checkpoint(tmp.path(), false);
    let checkpoint_path = tmp.path().join("model.safetensors");
    let replacement = tmp.path().join("replacement-checkpoint");
    write_f32_safetensors_with_offset(&replacement, &tiny_vlm_checkpoint_shapes(), 10.0);

    let model = with_after_visual_load_hook(
        move || {
            std::fs::rename(&replacement, &checkpoint_path)
                .expect("atomically replace checkpoint pathname with checkpoint B");
        },
        || {
            VisionEmbeddingModel::from_directory(tmp.path())
                .expect("constructor keeps both components on checkpoint A")
        },
    );

    assert_eq!(
        model.vision_weights.patch_embed_weight[0], 0.14,
        "visual weights must remain bound to checkpoint A"
    );
    let mut expected_embed = [0u16];
    f32_to_f16_slice(&[0.01], &mut expected_embed);
    assert_eq!(
        model.weights.embed_tokens[0], expected_embed[0],
        "decoder weights must remain bound to checkpoint A"
    );
}

#[test]
fn resolve_single_shard_prefers_existing_index_over_plain_file() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join("model.safetensors"), b"plain").expect("write convenience file");
    std::fs::write(
        tmp.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"tensor":"indexed.safetensors"}}"#,
    )
    .expect("write index");

    let resolved =
        resolve_qwen35_single_decoder_safetensors(tmp.path()).expect("single-shard index resolves");
    assert_eq!(resolved, tmp.path().join("indexed.safetensors"));
}

#[test]
fn resolve_single_shard_rejects_index_entry_escaping_model_directory() {
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        tmp.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"tensor":"../outside.safetensors"}}"#,
    )
    .expect("write index");

    let err = resolve_qwen35_single_decoder_safetensors(tmp.path())
        .expect_err("an index entry must not escape the checkpoint directory");
    assert!(
        err.to_string().contains("escapes the model directory"),
        "got: {err}"
    );
}
