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
//! [`VisionEmbeddingModel::from_directory`] accepts either a
//! `model.safetensors.index.json` naming exactly one decoder shard (the
//! Qwen3.5-0.8B layout), or an unindexed directory containing exactly one
//! safetensors file named `model.safetensors`. The indexed path is
//! authoritative when both are present. The unindexed path parses the
//! safetensors header and validates the exact `model.visual.*` inventory
//! before tensor payloads are materialized. Decoder and visual tensors are
//! materialized from the same mmap-backed open file, so a path replacement
//! during loading cannot mix two checkpoint versions. For an indexed
//! one-shard layout, the authoritative `weight_map` must exactly match that
//! opened shard's header inventory. `quantize_index.json` is rejected: this
//! loader constructs an f16 decoder and cannot bind a quantized visual file
//! set coherently. Callers with pre-loaded components, or a multi-shard
//! decoder checkpoint, can assemble their own weights and call
//! [`VisionEmbeddingModel::new`] directly.

use crate::error::{EmbedError, Result};
use lattice_inference::InferenceError;
use lattice_inference::model::qwen35_config::Qwen35Config;
use lattice_inference::tokenizer::bpe::BpeTokenizer;
use lattice_inference::tokenizer::common::Tokenizer as _;
use lattice_inference::vision::checkpoint::{
    Qwen35VisionWeights, load_qwen35_vision_weights_from_safetensors,
    open_qwen35_single_decoder_safetensors,
};
use lattice_inference::vision::qwen35_vit::preprocess_qwen35_image;
use lattice_inference::vision::{embed_image_from_bytes_f16, embed_image_from_bytes_f16_metal};
use lattice_inference::weights::f16_weights::{F16ModelWeights, load_f16_weights};
use std::path::Path;

pub use lattice_inference::forward::cpu_f16::PoolingStrategy;

/// Raises `tokenizer`'s own truncation cap to `config`'s
/// `max_position_embeddings.min(8192)` decode window when it sits below it,
/// otherwise returns `tokenizer` unchanged. Only ever raises the cap, never
/// lowers it.
///
/// Mirrors `lattice_inference::serve::embeddings`'s identically-named
/// guard (issue #1408): without it, a caller-supplied `BpeTokenizer` left at
/// its constructor default (4096) truncates any prompt between 4097 tokens
/// and this checkpoint's real window before [`embed_text_vlm_f16`] or
/// [`embed_image_from_bytes_f16`] ever see it, silently discarding content
/// the model could otherwise process. Unlike the `lattice_inference` sibling,
/// this crate has no admission guard ahead of the forward pass to reject an
/// over-window input outright -- the tokenizer's own cap is the only thing
/// standing between "process it" and "silently shorten it", so it must never
/// sit below the checkpoint's window.
fn capped_tokenizer(tokenizer: BpeTokenizer, config: &Qwen35Config) -> BpeTokenizer {
    let max_context = max_context(config);
    if tokenizer.max_seq_len() < max_context {
        tokenizer.with_max_seq_len(max_context)
    } else {
        tokenizer
    }
}

/// This checkpoint's usable decoder window: `max_position_embeddings.min(8192)`,
/// the same value [`capped_tokenizer`] raises the tokenizer cap to. Shared
/// source of truth for the admission checks below.
fn max_context(config: &Qwen35Config) -> usize {
    config.max_position_embeddings.min(8192)
}

/// Rejects `prompt` whose PRE-truncation token count, plus `reserved`
/// non-text scaffold tokens (image delimiters + pad tokens; `0` for a
/// text-only call), exceeds `max_context` -- i.e. would otherwise be
/// silently shortened by the tokenizer's own truncation before the forward
/// pass ever sees it (issue #1408). Mirrors
/// `lattice_inference::serve::embeddings::check_item_fits_window`'s
/// pre-truncation-count admission for the sibling wire route: `real_length`
/// can never exceed the tokenizer's own cap, so comparing against it (as
/// opposed to `pre_truncation_len`) can never observe an over-window input
/// once the cap sits at or above `max_context` -- the rejection would become
/// unreachable and the input silently embedded from a truncated prefix.
///
/// # Errors
///
/// Returns [`EmbedError::InvalidInput`] naming the pre-truncation count, the
/// reserved scaffold count, and `max_context`.
fn check_prompt_admission(
    tokenizer: &BpeTokenizer,
    prompt: &str,
    reserved: usize,
    max_context: usize,
) -> Result<()> {
    let pre_truncation_len = tokenizer.tokenize(prompt).pre_truncation_len;
    let total = pre_truncation_len.saturating_add(reserved);
    if total > max_context {
        return Err(EmbedError::InvalidInput(format!(
            "prompt tokenizes to {pre_truncation_len} tokens before truncation ({total} \
             including {reserved} scaffold tokens), exceeding the model's context window of \
             {max_context} tokens"
        )));
    }
    Ok(())
}

/// Real (pre-decode) scaffold token count an image consumes:
/// `vision_start_token_id` + `vision_end_token_id` + the checkpoint's
/// per-image pad-token count derived from the patch grid, mirroring
/// `lattice_inference::serve::embeddings::EmbeddingModel::image_scaffold_token_count`'s
/// identical arithmetic. Computed via a separate, cheap preprocessing call
/// (not the caller's own forward pass) so the admission check below can run
/// before the (expensive) ViT forward.
///
/// # Errors
///
/// Returns [`EmbedError::InvalidInput`] if the checkpoint has no
/// `vision_config`, or if the image cannot be decoded/preprocessed.
fn image_scaffold_reserve(config: &Qwen35Config, image_bytes: &[u8]) -> Result<usize> {
    let vision_cfg = config.vision_config.as_ref().ok_or_else(|| {
        EmbedError::InvalidInput(
            "checkpoint has no vision_config; not a vision-language checkpoint".to_string(),
        )
    })?;
    let (_pixel_values, grid) = preprocess_qwen35_image(image_bytes, vision_cfg)
        .map_err(|e| EmbedError::InvalidInput(format!("image preprocessing failed: {e}")))?;
    let merge_sq = vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size;
    if merge_sq == 0 || !grid.num_patches().is_multiple_of(merge_sq) {
        return Err(EmbedError::InvalidInput(format!(
            "image grid {grid:?} patch count is not a multiple of spatial_merge_size^2"
        )));
    }
    Ok(2 + grid.num_patches() / merge_sq)
}

#[cfg(test)]
thread_local! {
    static AFTER_VISUAL_LOAD_HOOK: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        std::cell::RefCell::new(None);
}

#[cfg(test)]
fn run_after_visual_load_hook() {
    let hook = AFTER_VISUAL_LOAD_HOOK.with(|slot| slot.borrow_mut().take());
    if let Some(hook) = hook {
        hook();
    }
}

#[cfg(test)]
fn with_after_visual_load_hook<T>(hook: impl FnOnce() + 'static, action: impl FnOnce() -> T) -> T {
    struct ClearHookOnDrop;
    impl Drop for ClearHookOnDrop {
        fn drop(&mut self) {
            AFTER_VISUAL_LOAD_HOOK.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }

    AFTER_VISUAL_LOAD_HOOK.with(|slot| {
        let previous = slot.borrow_mut().replace(Box::new(hook));
        assert!(
            previous.is_none(),
            "visual-load test hook already installed"
        );
    });
    let _clear_on_drop = ClearHookOnDrop;
    let result = action();
    AFTER_VISUAL_LOAD_HOOK.with(|slot| {
        assert!(
            slot.borrow().is_none(),
            "VisionEmbeddingModel::from_directory did not traverse the visual-load test hook"
        );
    });
    result
}

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
    ///
    /// Raises `tokenizer`'s own truncation cap to `config`'s context window
    /// when it sits below it (see [`capped_tokenizer`]) -- see that
    /// function's docs for why: without this, a caller-supplied tokenizer
    /// left at its default cap silently truncates any prompt between 4097
    /// tokens and the checkpoint's real window before it ever reaches the
    /// forward pass. [`Self::from_directory`] goes through this constructor
    /// too, so it gets the same guarantee.
    pub fn new(
        weights: F16ModelWeights,
        config: Qwen35Config,
        vision_weights: Qwen35VisionWeights,
        tokenizer: BpeTokenizer,
    ) -> Self {
        let tokenizer = capped_tokenizer(tokenizer, &config);
        Self {
            weights,
            config,
            vision_weights,
            tokenizer,
        }
    }

    /// Load a Qwen3.5 vision-language checkpoint directory: `config.json`,
    /// `tokenizer.json`, the `model.visual.*` vision-encoder tensors, and a
    /// single-shard f16 decoder checkpoint. An existing
    /// `model.safetensors.index.json` is authoritative. Without an index, the
    /// directory must contain exactly one `*.safetensors` file, named
    /// `model.safetensors`; its header must carry the exact `model.visual.*`
    /// inventory implied by `vision_config`. A `quantize_index.json` is
    /// rejected because this constructor has no quantized decoder loader. The
    /// chosen safetensors file is opened once and that same reader supplies
    /// both visual and decoder tensors.
    ///
    /// # Errors
    ///
    /// Returns [`EmbedError::ModelInitialization`] if `config.json` is
    /// missing or invalid, if the checkpoint has no `vision_config`, if a
    /// quantized checkpoint is present, if the unindexed safetensors layout is
    /// missing or ambiguous, if the decoder weights are sharded across more
    /// than one file, or if any component tensor fails to load.
    pub fn from_directory(dir: &Path) -> Result<Self> {
        let quantized_index = dir.join("quantize_index.json");
        match std::fs::symlink_metadata(&quantized_index) {
            Ok(_) => {
                return Err(EmbedError::ModelInitialization(format!(
                    "{} is present, but quantized checkpoints are not supported by \
                     VisionEmbeddingModel::from_directory's f16 decoder loader",
                    quantized_index.display()
                )));
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(EmbedError::ModelInitialization(format!(
                    "failed to inspect {}: {err}",
                    quantized_index.display()
                )));
            }
        }

        let config = Qwen35Config::from_model_dir(dir)
            .map_err(|e| EmbedError::ModelInitialization(format!("config.json: {e}")))?;
        let vision_cfg = config.vision_config.clone().ok_or_else(|| {
            EmbedError::ModelInitialization(format!(
                "{} has no vision_config; not a vision-language checkpoint",
                dir.display()
            ))
        })?;

        // Tokenizer parsing is independent of tensor payloads, so reject a
        // missing or malformed tokenizer before materializing multi-GB model
        // weights. The checkpoint itself is still opened only once below so
        // visual and decoder tensors stay bound to one file descriptor.
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path).map_err(|e| {
            EmbedError::ModelInitialization(format!("{}: {e}", tokenizer_path.display()))
        })?;

        let (mut sf, shard_path) = open_qwen35_single_decoder_safetensors(dir)
            .map_err(|e| EmbedError::ModelInitialization(format!("decoder checkpoint: {e}")))?;
        let vision_weights =
            load_qwen35_vision_weights_from_safetensors(&mut sf, &shard_path, &vision_cfg)
                .map_err(|e| EmbedError::ModelInitialization(format!("vision weights: {e}")))?;
        #[cfg(test)]
        run_after_visual_load_hook();
        let weights = load_f16_weights(&sf, &config)
            .map_err(|e| EmbedError::ModelInitialization(format!("decoder weights: {e}")))?;

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
    /// patch/merge geometry, the assembled request otherwise fails
    /// validation (the error message names the offending field), or the
    /// image's scaffold tokens plus `prompt`'s pre-truncation token count
    /// exceed the checkpoint's context window (issue #1408: rejected before
    /// the tokenizer's own truncation could silently shorten `prompt`).
    /// Returns [`EmbedError::InferenceFailed`] for every other underlying
    /// failure.
    pub fn embed_image(
        &self,
        image_bytes: &[u8],
        prompt: &str,
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>> {
        let reserved = image_scaffold_reserve(&self.config, image_bytes)?;
        check_prompt_admission(&self.tokenizer, prompt, reserved, max_context(&self.config))?;
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

    /// Metal-dispatching sibling of [`Self::embed_image`]: runs the ViT
    /// forward pass on the Metal GPU instead of the CPU (see
    /// [`lattice_inference::vision::embed_image_from_bytes_f16_metal`]).
    /// Same scaffold, pooling contract, and error semantics as
    /// [`Self::embed_image`], with one addition: on a build/platform with no
    /// Metal support, this returns [`EmbedError::InferenceFailed`] (Metal
    /// unavailability is a runtime-backend failure, not caller-input
    /// validation) rather than silently falling back to
    /// [`Self::embed_image`]'s CPU path — callers that want a fallback must
    /// call [`Self::embed_image`] themselves.
    ///
    /// # Errors
    ///
    /// See [`Self::embed_image`]'s docs.
    pub fn embed_image_metal(
        &self,
        image_bytes: &[u8],
        prompt: &str,
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>> {
        let reserved = image_scaffold_reserve(&self.config, image_bytes)?;
        check_prompt_admission(&self.tokenizer, prompt, reserved, max_context(&self.config))?;
        embed_image_from_bytes_f16_metal(
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
    /// Returns [`EmbedError::InvalidInput`] if the prompt is empty, tokenizes
    /// to an out-of-vocabulary id, or its pre-truncation token count exceeds
    /// the checkpoint's context window (issue #1408: rejected before the
    /// tokenizer's own truncation could silently shorten it). Returns
    /// [`EmbedError::InferenceFailed`] for every other underlying failure.
    pub fn embed_text(&self, prompt: &str, pooling: PoolingStrategy) -> Result<Vec<f32>> {
        check_prompt_admission(&self.tokenizer, prompt, 0, max_context(&self.config))?;
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

#[cfg(test)]
mod tests {
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

    fn write_f32_safetensors_with_offset(
        path: &Path,
        shapes: &[(String, Vec<usize>)],
        offset: f32,
    ) {
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
        std::fs::write(tmp.path().join("model.safetensors"), b"plain")
            .expect("write convenience file");
        std::fs::write(
            tmp.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"tensor":"indexed.safetensors"}}"#,
        )
        .expect("write index");

        let resolved = resolve_qwen35_single_decoder_safetensors(tmp.path())
            .expect("single-shard index resolves");
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
}
