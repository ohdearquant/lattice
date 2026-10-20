//! Pooled Qwen vision-language embedding model, served through the
//! `lattice` binary's `/v1/embeddings` route.
//!
//! [`EmbeddingModel`] loads the same way
//! `lattice-embed`'s `VisionEmbeddingModel` does -- an f16-packed decoder plus
//! vision-encoder weights from one safetensors checkpoint -- but is defined
//! in this crate rather than reused from `lattice-embed`, because
//! `lattice-embed` depends on `lattice-inference` (not the other way around):
//! a serving binary in this crate cannot name `lattice-embed`'s type without
//! a dependency cycle. This loader calls the same public
//! [`crate::vision::checkpoint`] / [`crate::weights::f16_weights`] functions
//! `VisionEmbeddingModel::from_directory` wraps; no pooling or checkpoint
//! math is duplicated, only the small amount of loading glue.

use std::path::Path;

use crate::error::InferenceError;
use crate::forward::cpu_f16::{PoolingStrategy, embed_text_vlm_f16};
use crate::model::qwen35_config::Qwen35Config;
use crate::tokenizer::bpe::BpeTokenizer;
use crate::tokenizer::common::Tokenizer as _;
use crate::vision::checkpoint::{
    Qwen35VisionWeights, load_qwen35_vision_weights_from_safetensors,
    open_qwen35_single_decoder_safetensors,
};
use crate::vision::qwen35_vit::preprocess_qwen35_image;
use crate::vision::{embed_image_from_bytes_f16, embed_image_from_bytes_f16_metal};
use crate::weights::f16_weights::{F16ModelWeights, load_f16_weights};

/// Raises `tokenizer`'s own truncation cap to `max_context` when it sits
/// below it, otherwise returns `tokenizer` unchanged.
///
/// This is the shared enforcement point for the invariant that no constructed
/// [`EmbeddingModel`] may have a tokenizer cap
/// below its own [`EmbeddingModel::max_context`]. If it did,
/// the pre-truncation-count admission check (see
/// [`crate::serve::embeddings::embed_items`]) would still admit an over-window item -- its
/// pre-truncation count is compared against `max_context`, which the
/// tokenizer cap sits below -- but the tokenizer would then silently
/// truncate it before the forward pass, embedding a truncated prefix
/// instead of rejecting the input. Only ever raises the cap, never lowers
/// it: a caller-supplied tokenizer already capped at or above
/// `max_context` (e.g. a fixture deliberately testing a *narrower* cap
/// than its own window) is left untouched.
fn capped_tokenizer(tokenizer: BpeTokenizer, max_context: usize) -> BpeTokenizer {
    if tokenizer.max_seq_len() < max_context {
        tokenizer.with_max_seq_len(max_context)
    } else {
        tokenizer
    }
}

/// How a loader reads a checkpoint, recorded in a router artifact's
/// representation so a gate cannot be served by a loader it was not trained
/// through. Two names exist because this workspace ships two serving binaries
/// whose embedders really do read the bytes differently: one shared constant
/// would make every cross-binary comparison pass by construction, which is
/// the one way this check can be wrong without ever failing.
///
/// `EmbeddingModel::from_directory` in this module -- a Qwen3.5 f16 decoder.
/// The `lattice` binary reads this one; it is the only binary that checks a
/// router gate.
pub const QWEN35_F16_DECODER_LOADER: &str = "qwen35-f16-decoder";

/// A loaded Qwen3.5 vision-language checkpoint used to serve pooled text and
/// image embeddings. See the module doc comment for why this loader lives
/// in this crate instead of reusing `lattice-embed::vision::VisionEmbeddingModel`.
pub struct EmbeddingModel {
    weights: F16ModelWeights,
    config: Qwen35Config,
    vision_weights: Qwen35VisionWeights,
    tokenizer: BpeTokenizer,
}

impl EmbeddingModel {
    /// Compose a model from already-loaded components (no I/O). Used by
    /// tests and by callers that share components with another in-process
    /// model instance.
    ///
    /// Raises `tokenizer`'s own truncation cap to this checkpoint's
    /// [`Self::max_context`] window when it sits below it (see
    /// `capped_tokenizer`), so this constructor gives the same guarantee
    /// [`Self::from_directory`] does: no `EmbeddingModel`, however
    /// constructed, can have a tokenizer cap below its own `max_context()`.
    /// Without this, a caller passing a default `BpeTokenizer` (whose
    /// constructors cap at 4096) alongside a checkpoint config whose
    /// `max_position_embeddings.min(8192)` exceeds 4096 would reproduce
    /// the same cap mismatch through this constructor instead of the loader.
    pub fn new(
        weights: F16ModelWeights,
        config: Qwen35Config,
        vision_weights: Qwen35VisionWeights,
        tokenizer: BpeTokenizer,
    ) -> Self {
        let max_context = config.max_position_embeddings.min(8192);
        let tokenizer = capped_tokenizer(tokenizer, max_context);
        Self {
            weights,
            config,
            vision_weights,
            tokenizer,
        }
    }

    /// Load a Qwen3.5 vision-language checkpoint directory for embedding
    /// service. Same layout and error conditions as `lattice-embed`'s
    /// `VisionEmbeddingModel::from_directory`: `config.json` must declare a
    /// `vision_config`, the directory must resolve to exactly one decoder
    /// safetensors shard, and a `quantize_index.json` (native Q4) checkpoint
    /// is rejected -- this loader only understands the f16 decoder format.
    pub fn from_directory(dir: &Path) -> Result<Self, String> {
        let quantized_index = dir.join("quantize_index.json");
        if quantized_index.exists() {
            return Err(format!(
                "{} is present, but quantized checkpoints are not supported by the embeddings \
                 f16 decoder loader",
                quantized_index.display()
            ));
        }

        let config = Qwen35Config::from_model_dir(dir).map_err(|e| format!("config.json: {e}"))?;
        let vision_cfg = config.vision_config.clone().ok_or_else(|| {
            format!(
                "{} has no vision_config; not a vision-language checkpoint",
                dir.display()
            )
        })?;

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path)
            .map_err(|e| format!("{}: {e}", tokenizer_path.display()))?;

        let (mut sf, shard_path) = open_qwen35_single_decoder_safetensors(dir)
            .map_err(|e| format!("decoder checkpoint: {e}"))?;
        let vision_weights =
            load_qwen35_vision_weights_from_safetensors(&mut sf, &shard_path, &vision_cfg)
                .map_err(|e| format!("vision weights: {e}"))?;
        let weights =
            load_f16_weights(&sf, &config).map_err(|e| format!("decoder weights: {e}"))?;

        // Routed through `Self::new` (not a bare struct literal) so the
        // tokenizer's truncation cap is raised to this checkpoint's context
        // window exactly as any other construction path would be -- see
        // `Self::new`'s and `capped_tokenizer`'s docs for why: otherwise,
        // whenever `max_position_embeddings.min(8192)` exceeds
        // `BpeTokenizer`'s `DEFAULT_BPE_MAX_SEQ_LEN` (4096), the tokenizer
        // can silently truncate an input to 4096 tokens while
        // `check_item_fits_window` admits it (its pre-truncation count is
        // still under the wider window), embedding a truncated prefix
        // instead of rejecting it.
        Ok(Self::new(weights, config, vision_weights, tokenizer))
    }

    /// Output embedding dimension (the checkpoint's decoder hidden size).
    pub fn dimensions(&self) -> usize {
        self.config.hidden_size
    }

    /// Real (possibly truncated) tokenized length of `text`, used for
    /// `usage.prompt_tokens` -- this must report what the pooled forward
    /// pass actually consumes, not what the caller sent. Never use this for
    /// a context-window admission check: see `Self::tokenize_lengths`.
    pub fn tokenize_len(&self, text: &str) -> usize {
        self.tokenizer.tokenize(text).real_length
    }

    /// Tokenizes `text` once, returning `(pre_truncation_len, real_length)`
    /// -- the count the context-window admission guard
    /// (`check_item_fits_window` in the pooled wire contract) must compare against, and the count the
    /// forward pass actually consumes for `usage.prompt_tokens`,
    /// respectively. Exists so a caller that needs both counts (the
    /// pooled wire contract's text branch) tokenizes once instead of twice;
    /// [`Self::tokenize_len`] above remains the single-value accessor for a
    /// caller (e.g. [`Self::image_scaffold_token_count`]) that only needs
    /// the real, post-truncation count.
    ///
    /// `pre_truncation_len` must be used for admission, never `real_length`:
    /// `real_length` can never exceed the tokenizer's `max_seq_len`, so
    /// comparing it against `max_context()` can never observe an
    /// over-window input once `max_seq_len <= max_context()` -- the
    /// rejection becomes unreachable and an over-long input is silently
    /// embedded from a truncated prefix instead. See
    /// `TokenizedInput::pre_truncation_len`'s doc for the same reasoning
    /// applied to the sibling `lattice_serve` text-embeddings route.
    fn tokenize_lengths(&self, text: &str) -> (usize, usize) {
        let tokenized = self.tokenizer.tokenize(text);
        (tokenized.pre_truncation_len, tokenized.real_length)
    }

    /// Maximum decoder scaffold length this checkpoint can process in one
    /// pooled prefill, mirroring the exact cap the chat serving path derives
    /// from the same field for its RoPE table (`Qwen35Model::from_safetensors`:
    /// `config.max_position_embeddings.min(8192)`). Used as the source of
    /// truth for `check_item_fits_window`.
    pub fn max_context(&self) -> usize {
        self.config.max_position_embeddings.min(8192)
    }

    /// Real scaffold token count an image item consumes: one
    /// `vision_start_token_id`, one `vision_end_token_id`, the checkpoint's
    /// per-image pad-token count derived from the patch grid (mirroring
    /// `pooled_embed.rs`'s own `image_pad_positions`/`num_pads` arithmetic,
    /// via a separate, cheap preprocessing call rather than by reaching into
    /// that module's internals), and `prompt`'s tokenized length. Used both
    /// for `usage.prompt_tokens` accounting and (via the returned count) can
    /// be compared against [`Self::max_context`] the same way a text item is.
    ///
    /// Uses [`Self::tokenize_len`] (the real, possibly-truncated count), not
    /// the pre-truncation count `Self::tokenize_lengths` also exposes, for
    /// the `prompt` component -- unlike the text item guard, this is not a
    /// live gap today: this
    /// method's production call site (the pooled wire contract's image branch)
    /// always passes `prompt = ""`, whose tokenized length is `0` either
    /// way, so no input can currently exercise the truncation-vs-window
    /// mismatch through this path. A future caller that passes a real
    /// `prompt` here would need the same admission/usage split the text
    /// branch got; left as real-length-only for now rather than adding an
    /// unexercised second accessor for a code path with no non-empty-prompt
    /// caller.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError::InvalidInput`] if the checkpoint has no
    /// `vision_config`, or if the image cannot be decoded/preprocessed (see
    /// [`preprocess_qwen35_image`]'s error conditions).
    pub fn image_scaffold_token_count(
        &self,
        image_bytes: &[u8],
        prompt: &str,
    ) -> Result<usize, InferenceError> {
        let vision_cfg = self.config.vision_config.as_ref().ok_or_else(|| {
            InferenceError::InvalidInput(
                "checkpoint has no vision_config; cannot count image scaffold tokens".to_string(),
            )
        })?;
        let (_pixel_values, grid) =
            preprocess_qwen35_image(image_bytes, vision_cfg).map_err(|e| {
                InferenceError::InvalidInput(format!("image preprocessing failed: {e}"))
            })?;
        let merge_sq = vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size;
        if merge_sq == 0 || !grid.num_patches().is_multiple_of(merge_sq) {
            return Err(InferenceError::InvalidInput(format!(
                "image grid {grid:?} patch count is not a multiple of spatial_merge_size^2"
            )));
        }
        let pads = grid.num_patches() / merge_sq;
        // vision_start + pads + vision_end + prompt tokens.
        Ok(2 + pads + self.tokenize_len(prompt))
    }

    /// Pool a text-only prompt through the same decoder + pooling path used
    /// for images, landing in the same vector space (see
    /// [`embed_text_vlm_f16`]).
    pub fn embed_text(
        &self,
        prompt: &str,
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>, InferenceError> {
        embed_text_vlm_f16(
            &self.weights,
            &self.config,
            &self.tokenizer,
            prompt,
            pooling,
        )
    }

    /// Pool an image into a single L2-normalized embedding vector, running
    /// the ViT forward pass on the CPU.
    pub fn embed_image(
        &self,
        image_bytes: &[u8],
        prompt: &str,
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>, InferenceError> {
        embed_image_from_bytes_f16(
            &self.weights,
            &self.config,
            &self.vision_weights,
            &self.tokenizer,
            image_bytes,
            prompt,
            pooling,
        )
    }

    /// Metal-dispatching sibling of [`Self::embed_image`]: runs the ViT
    /// forward pass on the Metal GPU. Returns
    /// [`InferenceError::UnsupportedModel`] when no Metal device is
    /// available (build-level or runtime) rather than silently falling back
    /// -- see [`Self::embed_image_best_effort`] for the fallback policy this
    /// module actually serves requests through.
    pub fn embed_image_metal(
        &self,
        image_bytes: &[u8],
        prompt: &str,
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>, InferenceError> {
        embed_image_from_bytes_f16_metal(
            &self.weights,
            &self.config,
            &self.vision_weights,
            &self.tokenizer,
            image_bytes,
            prompt,
            pooling,
        )
    }

    /// Serving policy for image items: try the Metal ViT dispatch first,
    /// falling back to the CPU ViT forward only when Metal is unavailable
    /// (build-level or runtime) rather than for every error -- a real
    /// inference failure (bad image, invalid assembled request) must still
    /// surface as such, not retry silently on the CPU path and mask it.
    pub fn embed_image_best_effort(
        &self,
        image_bytes: &[u8],
        prompt: &str,
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>, InferenceError> {
        match self.embed_image_metal(image_bytes, prompt, pooling) {
            Err(InferenceError::UnsupportedModel(_)) => {
                self.embed_image(image_bytes, prompt, pooling)
            }
            other => other,
        }
    }
}

impl crate::serving_embeddings::PooledEmbeddingService for EmbeddingModel {
    fn representation(&self) -> crate::serving_embeddings::EmbeddingRepresentation {
        crate::serving_embeddings::EmbeddingRepresentation {
            loader_format: QWEN35_F16_DECODER_LOADER,
            dimensions: self.dimensions(),
        }
    }

    fn max_context(&self) -> usize {
        EmbeddingModel::max_context(self)
    }

    fn text_counts(&self, text: &str) -> crate::serving_embeddings::TokenCounts {
        let (pre_truncation, embedded) = self.tokenize_lengths(text);
        crate::serving_embeddings::TokenCounts {
            pre_truncation,
            embedded,
        }
    }

    fn image_scaffold_len(&self, image: &[u8]) -> Result<usize, InferenceError> {
        self.image_scaffold_token_count(image, "")
    }

    fn embed_text(&self, text: &str, pooling: PoolingStrategy) -> Result<Vec<f32>, InferenceError> {
        EmbeddingModel::embed_text(self, text, pooling)
    }

    fn embed_image(
        &self,
        image: &[u8],
        pooling: PoolingStrategy,
    ) -> Result<Vec<f32>, InferenceError> {
        self.embed_image_best_effort(image, "", pooling)
    }
}

#[cfg(test)]
mod service_tests;

// The pooled route's existing tests read this model's private fields, so they
// compile as a descendant of this module; the inline module keeps their
// nested files resolving where they already live.
#[cfg(test)]
#[path = "../../../serve/embeddings"]
mod route_tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::serve::embeddings::*;

    mod tests;
}
