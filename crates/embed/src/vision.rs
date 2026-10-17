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
    /// when it sits below it (see `capped_tokenizer`) -- see that
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
mod tests;
