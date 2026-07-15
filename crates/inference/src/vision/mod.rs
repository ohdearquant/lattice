//! Vision encoder module for Qwen3-VL multimodal inference (ADR-049).
//!
//! ## Architecture
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
//! ## v0 scope (per ADR-049)
//!
//! - Fixed resolution: 448×448, 16×16 patches, spatial_merge_size=2
//! - Qwen3-VL only — no multi-family generalization
//! - CPU-side ViT forward pass (Metal GPU path is a v1 item)
//! - No dynamic resolution tiling, no multi-image inputs, no video
//!
//! ## Usage
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
pub mod preprocess;
pub mod qwen35_vit;
pub mod vit;

pub use config::VisionConfig;
pub use multimodal::MultimodalInput;

use merger::{MlpMerger, MlpMergerWeights};
use preprocess::PreprocessConfig;
use vit::{ViT, VisionWeights};

/// Errors produced by the vision encoder pipeline.
#[derive(Debug)]
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
mod tests {
    use super::*;
    use merger::MlpMergerWeights;
    use vit::{AttentionWeights, MlpWeights, ViTBlockWeights};

    /// Build a test `VisionEncoder` with zero/identity weights for shape testing.
    fn test_encoder(cfg: VisionConfig) -> VisionEncoder {
        let d = cfg.d_model;
        let patch_len = (cfg.patch_size as usize).pow(2) * 3;
        let d_mlp_vit = cfg.d_mlp;
        let d_decoder = cfg.d_decoder;
        let merge = cfg.spatial_merge_size;

        let make_block = || ViTBlockWeights {
            ln1_weight: vec![1.0f32; d],
            ln1_bias: vec![0.0f32; d],
            attn: AttentionWeights {
                q_proj: identity_w(d),
                k_proj: identity_w(d),
                v_proj: identity_w(d),
                o_proj: identity_w(d),
                q_norm_weight: vec![1.0f32; d],
                k_norm_weight: vec![1.0f32; d],
            },
            ln2_weight: vec![1.0f32; d],
            ln2_bias: vec![0.0f32; d],
            mlp: MlpWeights {
                gate_proj: vec![0.0f32; d_mlp_vit * d],
                up_proj: vec![0.0f32; d_mlp_vit * d],
                down_proj: vec![0.0f32; d * d_mlp_vit],
            },
        };

        let vit_weights = vit::VisionWeights {
            patch_embed_weight: rect_identity(d, patch_len),
            patch_embed_bias: vec![0.0f32; d],
            norm_weight: vec![1.0f32; d],
            norm_bias: vec![0.0f32; d],
            blocks: (0..cfg.n_layers).map(|_| make_block()).collect(),
        };

        let d_in_mlp = d * merge * merge;
        let d_hidden_mlp = d * 2;
        let mlp_weights = MlpMergerWeights {
            w1: vec![0.0f32; d_hidden_mlp * d_in_mlp],
            b1: vec![0.0f32; d_hidden_mlp],
            w2: vec![0.0f32; d_decoder * d_hidden_mlp],
            b2: vec![0.0f32; d_decoder],
        };

        VisionEncoder::new(vit_weights, mlp_weights, cfg, d_hidden_mlp)
            .expect("test encoder construction")
    }

    fn identity_w(n: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        w
    }

    fn rect_identity(rows: usize, cols: usize) -> Vec<f32> {
        let min_dim = rows.min(cols);
        let mut w = vec![0.0f32; rows * cols];
        for i in 0..min_dim {
            w[i * cols + i] = 1.0;
        }
        w
    }

    fn tiny_cfg() -> VisionConfig {
        let image_size = 8u32;
        let patch_size = 4u32;
        let n_patches = ((image_size / patch_size) as usize).pow(2); // 4
        let d_model = 8usize;
        VisionConfig {
            image_size,
            patch_size,
            n_patches,
            d_model,
            n_heads: 2,
            n_layers: 1,
            spatial_merge_size: 2,
            global_attn_every: 1,
            window_size: 2,
            mlp_ratio: 2,
            use_gelu: true,
            d_decoder: 16,
            d_mlp: d_model * 2,
        }
    }

    fn make_test_png(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        use image::RgbImage;
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.put_pixel(x, y, image::Rgb([r, g, b]));
            }
        }
        let mut buf = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .unwrap();
        buf
    }

    #[test]
    fn vision_encoder_output_shape() {
        let cfg = tiny_cfg();
        let enc = test_encoder(cfg.clone());
        let png = make_test_png(16, 16, 100, 150, 200);
        let out = enc.encode(&png).expect("encode");
        assert_eq!(out.raw_patches, cfg.n_patches);
        assert_eq!(out.visual_tokens, cfg.visual_tokens());
        assert_eq!(out.d_model, cfg.d_decoder);
        assert_eq!(
            out.patch_embeddings.len(),
            cfg.visual_tokens() * cfg.d_decoder
        );
    }

    #[test]
    fn vision_encoder_output_finite() {
        let cfg = tiny_cfg();
        let enc = test_encoder(cfg);
        let png = make_test_png(16, 16, 50, 100, 150);
        let out = enc.encode(&png).expect("encode");
        for &v in &out.patch_embeddings {
            assert!(v.is_finite(), "VisionOutput contains non-finite: {v}");
        }
    }

    #[test]
    fn vision_encoder_bad_image_rejected() {
        let cfg = tiny_cfg();
        let enc = test_encoder(cfg);
        let result = enc.encode(b"not an image");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), VisionError::ImageDecode(_)));
    }

    #[test]
    fn multimodal_input_construction_from_vision_output() {
        let cfg = tiny_cfg();
        let enc = test_encoder(cfg.clone());
        let png = make_test_png(16, 16, 0, 0, 0);
        let out = enc.encode(&png).expect("encode");

        let mm = MultimodalInput {
            patch_embeddings: out.patch_embeddings.clone(),
            raw_patches: out.raw_patches,
            visual_tokens: out.visual_tokens,
            d_model: out.d_model,
            text_tokens: vec![100, 200, 300],
        };
        mm.validate().expect("MultimodalInput valid");
        assert_eq!(mm.total_sequence_len(), out.visual_tokens + 3);
    }

    #[test]
    fn vision_error_display_coverage() {
        let e1 = VisionError::ImageDecode("bad bytes".into());
        assert!(e1.to_string().contains("bad bytes"));

        let e2 = VisionError::ShapeMismatch {
            expected: 10,
            actual: 5,
            context: "test ctx".into(),
        };
        assert!(e2.to_string().contains("10"));
        assert!(e2.to_string().contains("5"));

        let e3 = VisionError::InvalidConfig("zero dim".into());
        assert!(e3.to_string().contains("zero dim"));
    }
}
