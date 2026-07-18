//! Raw image bytes -> pooled, L2-normalized embedding: wires the existing
//! real Qwen3.5-0.8B vision pipeline (`qwen35_vit::{preprocess_qwen35_image,
//! qwen35_vit_forward}`, `qwen35_merger::qwen35_merger_forward`) into the
//! decoder-side pooling in [`crate::forward::cpu_f16`]
//! (`prefill_hidden_states_f16`, `embed_image_f16`, `PoolingStrategy`).
//!
//! ## Scope
//!
//! The scaffold this assembles — `[vision_start] + [image_pad; N] +
//! [vision_end] + tokenized(prompt)` — is a minimal, documented
//! approximation of the real HF chat template (no system role, no
//! `<think></think>` block, no BOS). It is sufficient to exercise correct
//! decoder injection and pooling; callers that need exact chat-template
//! parity should assemble their own `input_ids` (matching
//! `tests/fixtures/vision/input_ids.json`'s layout) and call
//! [`crate::forward::cpu_f16::embed_image_f16`] directly.
//!
//! Retrieval quality against the base Qwen3.5-0.8B *instruct* checkpoint is
//! unvalidated — see [`crate::forward::cpu_f16::PoolingStrategy`]'s doc
//! comment.

use super::checkpoint::Qwen35VisionWeights;
use super::multimodal::Qwen35VisionRequest;
use super::qwen35_merger::qwen35_merger_forward;
use super::qwen35_vit::{preprocess_qwen35_image, qwen35_vit_forward};
use crate::error::InferenceError;
use crate::forward::cpu_f16::{PoolingStrategy, embed_image_f16};
use crate::model::qwen35_config::Qwen35Config;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::common::Tokenizer;
use crate::weights::f16_weights::F16ModelWeights;

/// Encode `image_bytes` through the real Qwen3.5-0.8B vision pipeline
/// (preprocess -> ViT -> merger), assemble a minimal vision-prompt scaffold
/// around `prompt`, run decoder prefill, and return a pooled, L2-normalized
/// `[cfg.hidden_size]` embedding vector (2048 for the 0.8B checkpoint).
///
/// # Errors
///
/// Returns [`InferenceError::InvalidInput`] if `cfg` has no `vision_config`,
/// `image_token_id`, `vision_start_token_id`, or `vision_end_token_id` (a
/// non-vision-language checkpoint); if `image_bytes` cannot be decoded or
/// its dimensions are not an exact multiple of `patch_size *
/// spatial_merge_size` (see [`preprocess_qwen35_image`]'s scope note); or if
/// the assembled request fails [`Qwen35VisionRequest::validate`].
pub fn embed_image_from_bytes_f16(
    weights: &F16ModelWeights,
    cfg: &Qwen35Config,
    vision_weights: &Qwen35VisionWeights,
    tokenizer: &BpeTokenizer,
    image_bytes: &[u8],
    prompt: &str,
    pooling: PoolingStrategy,
) -> Result<Vec<f32>, InferenceError> {
    let vision_cfg = cfg.vision_config.as_ref().ok_or_else(|| {
        InferenceError::InvalidInput(
            "checkpoint has no vision_config; embed_image_from_bytes_f16 requires a \
             vision-language checkpoint"
                .to_string(),
        )
    })?;
    let image_token_id = cfg
        .image_token_id
        .ok_or_else(|| InferenceError::InvalidInput("checkpoint has no image_token_id".into()))?;
    let vision_start = cfg.vision_start_token_id.ok_or_else(|| {
        InferenceError::InvalidInput("checkpoint has no vision_start_token_id".into())
    })?;
    let vision_end = cfg.vision_end_token_id.ok_or_else(|| {
        InferenceError::InvalidInput("checkpoint has no vision_end_token_id".into())
    })?;

    let (pixel_values, grid) = preprocess_qwen35_image(image_bytes, vision_cfg)
        .map_err(|e| InferenceError::InvalidInput(format!("image preprocessing failed: {e}")))?;
    let pre_merger = qwen35_vit_forward(vision_weights, vision_cfg, &pixel_values, grid)
        .map_err(|e| InferenceError::InvalidInput(format!("ViT forward failed: {e}")))?;
    let post_merger = qwen35_merger_forward(&vision_weights.merger, vision_cfg, &pre_merger)
        .map_err(|e| InferenceError::InvalidInput(format!("merger forward failed: {e}")))?;

    let merge_sq = vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size;
    if merge_sq == 0 || !grid.num_patches().is_multiple_of(merge_sq) {
        return Err(InferenceError::InvalidInput(format!(
            "image grid {grid:?} patch count is not a multiple of spatial_merge_size^2"
        )));
    }
    let num_pads = grid.num_patches() / merge_sq;

    let text_ids = {
        let out = tokenizer.tokenize(prompt);
        out.input_ids[..out.real_length].to_vec()
    };

    let mut input_ids = Vec::with_capacity(2 + num_pads + text_ids.len());
    input_ids.push(vision_start);
    input_ids.extend(std::iter::repeat_n(image_token_id, num_pads));
    input_ids.push(vision_end);
    input_ids.extend(text_ids);

    let request = Qwen35VisionRequest {
        input_ids,
        image_grids: vec![grid],
        post_merger_rows: post_merger,
        image_token_id,
        spatial_merge_size: vision_cfg.spatial_merge_size,
        decoder_hidden_size: cfg.hidden_size,
    };

    embed_image_f16(weights, cfg, &request, pooling)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen35_config::{LayerType, RopeParams, VisionModelConfig};
    use crate::vision::checkpoint::{VisualBlockWeights, VisualMergerWeights};
    use crate::weights::f16_weights::{
        F16AttentionWeights, F16CommonLayerWeights, F16FeedForwardWeights,
        F16FullAttentionLayerWeights,
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
}
