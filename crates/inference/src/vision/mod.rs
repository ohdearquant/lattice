//! Vision encoder module. Three vision paths live here: the real Qwen3.5 path,
//! the PaddleOCR-VL path, and the inert Qwen3-VL scaffold. PaddleOCR-VL has
//! its own weights and config, but shares `apply_rope_inplace` and
//! `multihead_attention_full` from `qwen35_vit`, plus `batch_matvec`, `gelu`,
//! and `layer_norm` from `vit`, and `gelu_exact` from `qwen35_merger`.
//!
//! ## The real path: Qwen3.5-0.8B (ADR-069, S1-S3b shipped)
//!
//! [`checkpoint::load_qwen35_vision_weights`](crate::vision::checkpoint::load_qwen35_vision_weights) loads the real
//! `model.visual.*` tensors (S1/S2); [`qwen35_vit::qwen35_vit_forward`](crate::vision::qwen35_vit::qwen35_vit_forward) runs
//! the depth-12/hidden-768 forward pass over them, producing pre-merger
//! hidden states `[num_patches, hidden_size]` (S3a — CPU reference, gated at
//! cosine > 0.999 vs a committed HF differential golden in
//! `tests/vision_s3_vit_forward_test.rs`, wired as a required CI job).
//! [`qwen35_vit_metal::qwen35_vit_forward_metal`](crate::vision::qwen35_vit_metal::qwen35_vit_forward_metal) is the Metal port (S3b),
//! gated against the S3a CPU reference under the machine GPU lock in
//! `tests/vision_s3b_vit_metal_gate_test.rs`.
//! [`qwen35_merger::qwen35_merger_forward`](crate::vision::qwen35_merger::qwen35_merger_forward) runs the spatial-merge +
//! projection MLP over the pre-merger hidden states, producing the
//! post-merger visual embeddings `[num_patches / spatial_merge_size^2,
//! out_hidden_size]` that the decoder consumes (S4 — gated at cosine >
//! 0.999 vs a committed HF differential golden in
//! `tests/vision_s4_merger_test.rs`). S5 (decoder injection + M-RoPE) is
//! unimplemented — **this is the path future vision work should build on.**
//!
//! ```text
//! image bytes → preprocess_qwen35_image → qwen35_vit_forward
//!   → [num_patches, hidden_size] pre-merger hidden states
//!   → qwen35_merger_forward
//!   → [num_patches / spatial_merge_size^2, out_hidden_size] post-merger visual embeddings
//!   (S5 not yet implemented: no decoder injection)
//! ```
//!
//! ## The inert scaffold: hypothetical Qwen3-VL 7B (ADR-049)
//!
//! [`vit::ViT`](crate::vision::vit::ViT) + [`merger::MlpMerger`](crate::vision::merger::MlpMerger) + [`preprocess`](crate::vision::preprocess) + [`VisionEncoder`](crate::vision::VisionEncoder)
//! below implement a *different*, structurally incompatible forward pass
//! (three separate Q/K/V projections with per-head RMSNorm, windowed
//! attention, a from-scratch 2D RoPE, ImageNet normalization stats) for a
//! 7B-parameter checkpoint shape that has never been loaded against real
//! weights. Nothing in the tree calls `VisionEncoder::new` outside its own
//! unit tests below. It is kept for a future 7B checkpoint if one is
//! targeted; do not extend it to consume the Qwen3.5-0.8B checkpoint — the
//! two attention/normalization conventions are incompatible (see
//! `checkpoint.rs` and `qwen35_vit.rs` module docs for the specific deltas).
//!
//! ```text
//! image bytes
//!   → preprocess (resize 448×448, normalize, patch grid)
//!   → ViT forward (patch embed, 2D RoPE, 27 transformer blocks)
//!   → MLP merger (spatial merge 2×2, project to decoder dim)
//!   → VisionOutput (196 visual tokens × d_model)
//!   → MultimodalInput (patch_embeddings + text_tokens)
//!   → generate_multimodal (prepend visual to text embeddings → decode loop)
//! ```
//!
//! ### Scaffold v0 scope (per ADR-049)
//!
//! - Fixed resolution: 448×448, 16×16 patches, spatial_merge_size=2
//! - Qwen3-VL only — no multi-family generalization
//! - CPU-side ViT forward pass (Metal GPU path is a v1 item)
//! - No dynamic resolution tiling, no multi-image inputs, no video
//!
//! ### Scaffold usage
//!
//! ```no_run
//! use lattice_inference::vision::{VisionEncoder, VisionConfig, MultimodalInput};
//! use lattice_inference::vision::vit::VisionWeights;
//! use lattice_inference::vision::merger::MlpMergerWeights;
//!
//! // Construct after loading weights from safetensors checkpoint
//! let cfg = VisionConfig::qwen3_vl_7b();
//! // let enc = VisionEncoder::new(vit_weights, mlp_weights, cfg).unwrap();
//! // let vision_out = enc.encode(&image_bytes).unwrap();
//! // let mm_input = MultimodalInput { ... };
//! // state.generate_multimodal(mm_input, gen_cfg);
//! ```

pub mod checkpoint;
pub mod config;
pub mod merger;
pub mod multimodal;
pub mod paddleocr_preprocess;
pub mod paddleocr_vit;
pub mod pooled_embed;
pub mod preprocess;
pub mod qwen35_merger;
pub mod qwen35_mrope;
pub mod qwen35_vit;
pub mod qwen35_vit_metal;
pub mod vit;

pub use config::VisionConfig;
pub use multimodal::MultimodalInput;
pub use pooled_embed::{embed_image_from_bytes_f16, embed_image_from_bytes_f16_metal};

use merger::{MlpMerger, MlpMergerWeights};
use preprocess::PreprocessConfig;
use vit::{ViT, VisionWeights};

/// Errors produced by the vision encoder pipeline.
///
/// Marked `#[non_exhaustive]` so that distinguishing a new failure mode is a
/// minor change rather than a major one. Adding `DimensionsExceeded` was a
/// breaking change only because this enum was exhaustive; that cost is paid
/// once here.
#[derive(Debug)]
#[non_exhaustive]
pub enum VisionError {
    /// Raw image bytes could not be decoded (unsupported format, corrupt data).
    ImageDecode(String),
    /// A weight or activation tensor had unexpected dimensions.
    ShapeMismatch {
        expected: usize,
        actual: usize,
        context: String,
    },
    /// A configuration value is invalid (zero dimension, indivisible sizes, etc.).
    InvalidConfig(String),
    /// The decoded image's declared pixel dimensions, pre-merge patch grid,
    /// or resulting preprocessed buffer exceed a serving-side budget --
    /// distinct from [`Self::ImageDecode`] (the image itself is well-formed,
    /// just larger than this server will process) and from
    /// [`Self::InvalidConfig`] (a checkpoint/config problem, not a
    /// per-request one). Callers map this to a client-safe, retriable
    /// rejection rather than the generic image-decode error.
    DimensionsExceeded(String),
    /// I/O error during weight loading.
    Io(std::io::Error),
}

impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ImageDecode(msg) => write!(f, "Vision image decode error: {msg}"),
            Self::ShapeMismatch {
                expected,
                actual,
                context,
            } => write!(
                f,
                "Vision shape mismatch ({context}): expected {expected}, got {actual}"
            ),
            Self::InvalidConfig(msg) => write!(f, "Vision invalid config: {msg}"),
            Self::DimensionsExceeded(msg) => write!(f, "Vision dimensions exceeded: {msg}"),
            Self::Io(e) => write!(f, "Vision I/O error: {e}"),
        }
    }
}

impl std::error::Error for VisionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for VisionError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// VisionOutput
// ---------------------------------------------------------------------------

/// Output of the end-to-end vision encoder pipeline.
///
/// Patch embeddings have been spatially merged and projected to the decoder
/// hidden dimension. They are ready to be prepended to text token embeddings.
#[derive(Debug, Clone)]
pub struct VisionOutput {
    /// Projected patch embeddings: flat `[visual_tokens * d_model]` f32.
    pub patch_embeddings: Vec<f32>,
    /// Raw patch count before spatial merge (e.g., 784 for 448×448, patch=16).
    pub raw_patches: usize,
    /// Visual token count after spatial merge (e.g., 196 for merge_size=2).
    pub visual_tokens: usize,
    /// Decoder hidden dimension.
    pub d_model: usize,
}

// ---------------------------------------------------------------------------
// VisionEncoder
// ---------------------------------------------------------------------------

/// End-to-end vision encoder: image bytes → projected patch embeddings.
///
/// Encapsulates the full pipeline:
/// 1. CPU preprocessing (resize, normalize, patch extraction)
/// 2. ViT forward pass
/// 3. MLP merger (spatial merge + MLP projection)
///
/// Allocate once; reuse across calls (`encode` takes `&self`).
pub struct VisionEncoder {
    vit: ViT,
    merger: MlpMerger,
    preprocess_cfg: PreprocessConfig,
}

impl VisionEncoder {
    /// Construct the vision encoder from pre-loaded weights.
    ///
    /// - `vit_weights`: loaded from `vision_model.*` keys in the checkpoint
    /// - `mlp_weights`: loaded from `vision_model.merger.*` keys
    /// - `config`: must match the actual Qwen3-VL checkpoint dimensions
    ///
    /// The MLP merger's `d_hidden` is derived from `config.d_mlp` (= 4 × d_model
    /// within the ViT; not the decoder `d_decoder`). Callers should set
    /// `mlp_d_hidden` to the actual intermediate dimension from the checkpoint.
    pub fn new(
        vit_weights: VisionWeights,
        mlp_weights: MlpMergerWeights,
        config: VisionConfig,
        mlp_d_hidden: usize,
    ) -> Result<Self, VisionError> {
        config.validate()?;
        let d_vit = config.d_model;
        let d_model = config.d_decoder;
        let merge_size = config.spatial_merge_size;
        let merger = MlpMerger::new(mlp_weights, d_vit, mlp_d_hidden, d_model, merge_size)?;
        let preprocess_cfg = PreprocessConfig {
            image_size: config.image_size,
            patch_size: config.patch_size,
            mean: preprocess::QWEN3_VL_IMAGE_MEAN,
            std: preprocess::QWEN3_VL_IMAGE_STD,
        };
        let vit = ViT::new(vit_weights, config)?;
        Ok(Self {
            vit,
            merger,
            preprocess_cfg,
        })
    }

    /// Encode raw image bytes (JPEG or PNG) through ViT + spatial merge + MLP.
    ///
    /// Returns `VisionOutput` with `[visual_tokens, d_model]` patch embeddings.
    ///
    /// # Errors
    ///
    /// Returns `VisionError::ImageDecode` if the bytes cannot be decoded.
    /// Returns shape/config errors if the ViT or merger dimensions are
    /// inconsistent (these indicate incorrect weight loading, not runtime issues).
    pub fn encode(&self, image_bytes: &[u8]) -> Result<VisionOutput, VisionError> {
        // Step 1: CPU preprocessing
        let img_tensor = preprocess::preprocess(image_bytes, &self.preprocess_cfg)?;

        // Step 2: ViT forward pass → [n_patches, d_vit]
        let vit_out = self.vit.forward(&img_tensor)?;
        let raw_patches = img_tensor.n_patches;

        // Step 3: MLP merger → [visual_tokens, d_model]
        let projected = self.merger.merge_and_project(&vit_out, raw_patches)?;
        let merge_sq = self.vit.config.spatial_merge_size.pow(2);
        let visual_tokens = raw_patches / merge_sq;
        let d_model = self.vit.config.d_decoder;

        Ok(VisionOutput {
            patch_embeddings: projected,
            raw_patches,
            visual_tokens,
            d_model,
        })
    }

    /// Access the underlying ViT config.
    pub fn config(&self) -> &VisionConfig {
        &self.vit.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
