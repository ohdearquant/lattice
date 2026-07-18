//! Image (and image+text) embedding through the Qwen3.5 vision-language
//! pooled-embedding pipeline (ADR-069 S5, #1007).
//!
//! This is a wire-through: [`VisionEmbeddingModel::embed_image`] and
//! [`VisionEmbeddingModel::embed_text`] call straight into
//! `lattice_inference::vision::embed_image_from_bytes_f16` /
//! `lattice_inference::forward::cpu_f16::embed_text_vlm_f16`, the same
//! pooling + L2-normalization contract #1007 established. No new math lives
//! here — only checkpoint loading (mirroring the directory-loading pattern
//! `service::native` uses for the BERT/Qwen text models) and error mapping.
//!
//! [`VisionEmbeddingModel::from_directory`] supports only checkpoints that
//! carry a `model.safetensors.index.json` (or `quantize_index.json`)
//! manifest naming exactly one decoder shard (matches the Qwen3.5-0.8B
//! checkpoint shape) — the vision-tensor loader requires that manifest and
//! runs before decoder-shard resolution, so a directory with only a plain
//! `model.safetensors` and no manifest is rejected. Callers with pre-loaded
//! components, or a multi-shard checkpoint, can assemble their own weights
//! and call [`VisionEmbeddingModel::new`] directly.

use crate::error::{EmbedError, Result};
use lattice_inference::InferenceError;
use lattice_inference::model::qwen35_config::Qwen35Config;
use lattice_inference::tokenizer::bpe::BpeTokenizer;
use lattice_inference::vision::checkpoint::{Qwen35VisionWeights, load_qwen35_vision_weights};
use lattice_inference::vision::embed_image_from_bytes_f16;
use lattice_inference::weights::SafetensorsFile;
use lattice_inference::weights::f16_weights::{F16ModelWeights, load_f16_weights};
use std::path::Path;

pub use lattice_inference::forward::cpu_f16::PoolingStrategy;

/// A loaded Qwen3.5 vision-language checkpoint, ready to pool image (and
/// image+text) embeddings.
///
/// See [`docs/model.md`](../docs/model.md) for the general model-loading design; this type
/// follows the same "load once, reuse" shape as `NativeEmbeddingService`'s wrapped models.
pub struct VisionEmbeddingModel {
    weights: F16ModelWeights,
    config: Qwen35Config,
    vision_weights: Qwen35VisionWeights,
    tokenizer: BpeTokenizer,
}

impl VisionEmbeddingModel {
    /// Compose a model from already-loaded components (no I/O).
    ///
    /// Use this when the checkpoint spans multiple safetensors shards (not
    /// supported by [`Self::from_directory`]) or when components are shared
    /// across other in-process model instances.
    pub fn new(
        weights: F16ModelWeights,
        config: Qwen35Config,
        vision_weights: Qwen35VisionWeights,
        tokenizer: BpeTokenizer,
    ) -> Self {
        Self {
            weights,
            config,
            vision_weights,
            tokenizer,
        }
    }

    /// Load a Qwen3.5 vision-language checkpoint directory: `config.json`,
    /// `tokenizer.json`, the `model.visual.*` vision-encoder tensors, and a
    /// single-shard decoder checkpoint. The directory must carry a
    /// `model.safetensors.index.json` (or `quantize_index.json`) manifest
    /// naming exactly one decoder shard file — the vision-tensor loader
    /// requires one of those manifests and runs before decoder-shard
    /// resolution, so a plain `model.safetensors` alone (with no manifest)
    /// is not sufficient. The canonical Qwen3.5-0.8B HF layout (a one-shard
    /// index) satisfies this.
    ///
    /// # Errors
    ///
    /// Returns [`EmbedError::ModelInitialization`] if `config.json` is
    /// missing or invalid, if the checkpoint has no `vision_config`, if
    /// neither manifest is present, if the decoder weights are sharded
    /// across more than one file, or if any component tensor fails to load.
    pub fn from_directory(dir: &Path) -> Result<Self> {
        let config = Qwen35Config::from_model_dir(dir)
            .map_err(|e| EmbedError::ModelInitialization(format!("config.json: {e}")))?;
        let vision_cfg = config.vision_config.clone().ok_or_else(|| {
            EmbedError::ModelInitialization(format!(
                "{} has no vision_config; not a vision-language checkpoint",
                dir.display()
            ))
        })?;

        let vision_weights = load_qwen35_vision_weights(dir, &vision_cfg)
            .map_err(|e| EmbedError::ModelInitialization(format!("vision weights: {e}")))?;

        let shard_path = resolve_single_shard(dir)?;
        let sf = SafetensorsFile::open(&shard_path).map_err(|e| {
            EmbedError::ModelInitialization(format!("opening {}: {e}", shard_path.display()))
        })?;
        let weights = load_f16_weights(&sf, &config)
            .map_err(|e| EmbedError::ModelInitialization(format!("decoder weights: {e}")))?;

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path).map_err(|e| {
            EmbedError::ModelInitialization(format!("{}: {e}", tokenizer_path.display()))
        })?;

        Ok(Self::new(weights, config, vision_weights, tokenizer))
    }

    /// Pool an image (plus an optional text prompt) into a single
    /// L2-normalized `[dimensions()]` embedding vector.
    ///
    /// Same scaffold and pooling contract as
    /// [`lattice_inference::vision::embed_image_from_bytes_f16`] (see that
    /// function's docs for the exact prompt-assembly layout).
    ///
    /// # Errors
    ///
    /// Returns [`EmbedError::InvalidInput`] if `image_bytes` cannot be
    /// decoded, its dimensions are not compatible with the checkpoint's
    /// patch/merge geometry, or the assembled request otherwise fails
    /// validation (the error message names the offending field). Returns
    /// [`EmbedError::InferenceFailed`] for every other underlying failure —
    /// e.g. the prompt plus image tokens exceeding the checkpoint's context
    /// window.
    pub fn embed_image(
        &self,
        image_bytes: &[u8],
        prompt: &str,
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>> {
        embed_image_from_bytes_f16(
            &self.weights,
            &self.config,
            &self.vision_weights,
            &self.tokenizer,
            image_bytes,
            prompt,
            pooling,
        )
        .map_err(map_inference_error)
    }

    /// Pool a text-only prompt through the same decoder + pooling path as
    /// [`Self::embed_image`], landing in the same vector space.
    ///
    /// # Errors
    ///
    /// Returns [`EmbedError::InvalidInput`] if the prompt is empty or
    /// tokenizes to an out-of-vocabulary id. Returns
    /// [`EmbedError::InferenceFailed`] for every other underlying failure —
    /// e.g. the prompt exceeding the checkpoint's context window.
    pub fn embed_text(&self, prompt: &str, pooling: PoolingStrategy) -> Result<Vec<f32>> {
        lattice_inference::forward::cpu_f16::embed_text_vlm_f16(
            &self.weights,
            &self.config,
            &self.tokenizer,
            prompt,
            pooling,
        )
        .map_err(map_inference_error)
    }

    /// Output embedding dimension (the checkpoint's decoder hidden size).
    pub fn dimensions(&self) -> usize {
        self.config.hidden_size
    }
}

/// Map an inference-layer error to the embed crate's two-variant contract:
/// caller-supplied-input problems stay distinguishable from every other
/// (model/runtime) failure, so callers can tell "fix your request" apart
/// from "retry or report a bug" (see `embed_image`/`embed_text` docs).
fn map_inference_error(e: InferenceError) -> EmbedError {
    match e {
        InferenceError::InvalidInput(msg) => EmbedError::InvalidInput(msg),
        other => EmbedError::InferenceFailed(other.to_string()),
    }
}

/// Resolve the single safetensors shard `load_f16_weights` needs. By the time
/// this runs, [`load_qwen35_vision_weights`] has already required a
/// `model.safetensors.index.json` or `quantize_index.json` manifest to exist
/// in `model_dir` (see module docs) — so a convenience `model.safetensors`
/// file (often a symlink some local checkouts add alongside the manifest) is
/// checked first as a cheap shortcut when present, then falls back to
/// resolving the shard named by the index. This mirrors
/// `Qwen35Model::from_safetensors`'s plain-then-index precedence, but
/// resolves the concrete shard path a single-`SafetensorsFile` loader needs
/// (multi-shard checkpoints are out of scope here — see module docs).
fn resolve_single_shard(model_dir: &Path) -> Result<std::path::PathBuf> {
    let plain = model_dir.join("model.safetensors");
    if plain.exists() {
        return Ok(plain);
    }
    let index = lattice_inference::weights::parse_index(model_dir).map_err(|e| {
        EmbedError::ModelInitialization(format!(
            "no model.safetensors in {} and no valid model.safetensors.index.json: {e}",
            model_dir.display()
        ))
    })?;
    let mut shards: Vec<&str> = index.weight_map.values().map(String::as_str).collect();
    shards.sort_unstable();
    shards.dedup();
    match shards.as_slice() {
        [one] => Ok(model_dir.join(one)),
        [] => Err(EmbedError::ModelInitialization(format!(
            "empty weight_map in {}",
            model_dir.join("model.safetensors.index.json").display()
        ))),
        _ => Err(EmbedError::ModelInitialization(format!(
            "checkpoint at {} is sharded across {} files; VisionEmbeddingModel::from_directory \
             only supports single-shard checkpoints -- use VisionEmbeddingModel::new with \
             manually loaded components instead",
            model_dir.display(),
            shards.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_inference::model::qwen35_config::{LayerType, RopeParams, VisionModelConfig};
    use lattice_inference::vision::checkpoint::{VisualBlockWeights, VisualMergerWeights};
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

    /// A runtime (non-input) failure -- the prompt exceeding the checkpoint's
    /// context window, surfaced as `InferenceError::Inference` from the
    /// shared prefill path (cpu_f16.rs) -- must map to
    /// `EmbedError::InferenceFailed`, not `EmbedError::InvalidInput`: the
    /// prompt itself is well-formed, the checkpoint just can't fit it.
    #[test]
    fn embed_text_maps_context_overflow_to_inference_failed() {
        let (mut cfg, weights, vision_weights) = tiny_vlm_fixture();
        cfg.max_position_embeddings = 1;
        let tokenizer = single_char_tokenizer();
        let model = VisionEmbeddingModel::new(weights, cfg, vision_weights, tokenizer);

        let err = model
            .embed_text("abc", PoolingStrategy::LastToken)
            .expect_err("a prompt longer than max_position_embeddings must fail");
        assert!(
            matches!(err, EmbedError::InferenceFailed(_)),
            "context-window overflow is a runtime failure, not caller-input validation, got: {err:?}"
        );
        assert!(
            err.to_string().contains("context window"),
            "error should retain the underlying context-window detail, got: {err}"
        );
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

        let err = resolve_single_shard(tmp.path()).expect_err("multi-shard must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("sharded across 2 files"), "got: {msg}");
    }

    #[test]
    fn resolve_single_shard_rejects_missing_manifest() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let err = resolve_single_shard(tmp.path()).expect_err("missing manifest must be rejected");
        assert!(matches!(err, EmbedError::ModelInitialization(_)));
    }

    /// `from_directory`'s documented contract requires an index/quantize
    /// manifest (the vision-tensor loader runs before decoder-shard
    /// resolution and has no plain-file fallback). A directory with a valid
    /// config.json (including `vision_config`) but no manifest at all must
    /// fail with an actionable, named error -- not merely `expect_err` on
    /// some opaque error -- pinning the real (manifest-required) behavior
    /// rather than the previously-documented (plain-file-sufficient) one.
    #[test]
    fn from_directory_without_manifest_reports_actionable_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let config_json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../inference/tests/fixtures/qwen35_0_8b_config.json"
        ));
        std::fs::write(tmp.path().join("config.json"), config_json).expect("write config.json");

        let Err(err) = VisionEmbeddingModel::from_directory(tmp.path()) else {
            panic!("a directory with no index/quantize manifest must be rejected")
        };
        assert!(matches!(err, EmbedError::ModelInitialization(_)));
        let msg = err.to_string();
        assert!(
            msg.contains("model.safetensors.index.json") && msg.contains("quantize_index.json"),
            "error must name the missing manifest(s), got: {msg}"
        );
    }
}
