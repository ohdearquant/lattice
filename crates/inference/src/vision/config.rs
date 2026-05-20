//! Vision encoder configuration for Qwen3-VL.
//!
//! All defaults match the Qwen3-VL-7B-Instruct Hugging Face `VisionConfig`.
//! This is a v0 implementation targeting Qwen3-VL only; multi-family support
//! is deferred per ADR-049.

/// Configuration for the Qwen3-VL ViT encoder.
///
/// Field defaults are taken from `Qwen/Qwen3-VL-7B-Instruct` `config.json`
/// under the `vision_config` key.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Input image height and width (square, pixels). Qwen3-VL default: 448.
    pub image_size: u32,
    /// Patch side length (pixels). Qwen3-VL default: 16.
    pub patch_size: u32,
    /// Number of raw patches = (image_size / patch_size)^2. Computed field.
    pub n_patches: usize,
    /// ViT hidden dimension. Qwen3-VL 7B: 1152.
    pub d_model: usize,
    /// Number of attention heads in the ViT. Qwen3-VL 7B: 16.
    pub n_heads: usize,
    /// Number of ViT transformer layers. Qwen3-VL default: 27.
    pub n_layers: usize,
    /// Spatial merge factor (applied in both H and W). Qwen3-VL default: 2.
    /// After merge: visual_tokens = n_patches / (spatial_merge_size^2).
    pub spatial_merge_size: usize,
    /// Every `global_attn_every`-th block uses full (global) attention;
    /// all other blocks use window attention. Qwen3-VL: 4 (every 4th block).
    pub global_attn_every: usize,
    /// Window size for windowed self-attention (patches per side).
    /// Qwen3-VL: 16 (covers a 256-patch window in a 784-patch grid).
    pub window_size: usize,
    /// MLP projection hidden dimension inside VisionEncoder. Typically 4×d_model.
    pub mlp_ratio: usize,
    /// ViT MLP activation: GELU is used in Qwen3-VL.
    pub use_gelu: bool,
    /// MLP merger output dimension: must match the decoder's hidden_size.
    pub d_decoder: usize,
    /// ViT MLP intermediate dimension (d_model * mlp_ratio).
    pub d_mlp: usize,
}

impl VisionConfig {
    /// Qwen3-VL-7B defaults (Hugging Face `Qwen/Qwen3-VL-7B-Instruct`).
    ///
    /// The decoder hidden dimension (d_decoder) must be set to match the
    /// actual decoder model. For Qwen3.5-2B this is 2048; for Qwen3.5-7B it
    /// is 3584. Callers should always override `d_decoder` explicitly.
    pub fn qwen3_vl_7b() -> Self {
        let image_size = 448u32;
        let patch_size = 16u32;
        let n_patches = ((image_size / patch_size) as usize).pow(2);
        let d_model = 1152usize;
        let mlp_ratio = 4usize;
        Self {
            image_size,
            patch_size,
            n_patches,
            d_model,
            n_heads: 16,
            n_layers: 27,
            spatial_merge_size: 2,
            global_attn_every: 4,
            window_size: 16,
            mlp_ratio,
            use_gelu: true,
            // Caller must set this to the actual decoder hidden size.
            d_decoder: 3584,
            d_mlp: d_model * mlp_ratio,
        }
    }

    /// Number of visual tokens delivered to the decoder after spatial merge.
    pub fn visual_tokens(&self) -> usize {
        self.n_patches / (self.spatial_merge_size * self.spatial_merge_size)
    }

    /// Head dimension derived from d_model / n_heads.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Validate that dimensions are internally consistent.
    pub fn validate(&self) -> Result<(), super::VisionError> {
        if self.patch_size == 0 {
            return Err(super::VisionError::InvalidConfig(
                "patch_size must be > 0".into(),
            ));
        }
        if self.image_size % self.patch_size != 0 {
            return Err(super::VisionError::InvalidConfig(format!(
                "image_size {} must be divisible by patch_size {}",
                self.image_size, self.patch_size
            )));
        }
        let expected_n_patches = ((self.image_size / self.patch_size) as usize).pow(2);
        if self.n_patches != expected_n_patches {
            return Err(super::VisionError::InvalidConfig(format!(
                "n_patches {} inconsistent with image_size={} patch_size={}; expected {}",
                self.n_patches, self.image_size, self.patch_size, expected_n_patches
            )));
        }
        if self.n_patches % (self.spatial_merge_size * self.spatial_merge_size) != 0 {
            return Err(super::VisionError::InvalidConfig(format!(
                "n_patches {} must be divisible by spatial_merge_size^2={}",
                self.n_patches,
                self.spatial_merge_size * self.spatial_merge_size
            )));
        }
        if self.d_model == 0 {
            return Err(super::VisionError::InvalidConfig(
                "d_model must be > 0".into(),
            ));
        }
        if self.n_heads == 0 || self.d_model % self.n_heads != 0 {
            return Err(super::VisionError::InvalidConfig(format!(
                "d_model {} must be divisible by n_heads {}",
                self.d_model, self.n_heads
            )));
        }
        if self.n_layers == 0 {
            return Err(super::VisionError::InvalidConfig(
                "n_layers must be > 0".into(),
            ));
        }
        if self.d_decoder == 0 {
            return Err(super::VisionError::InvalidConfig(
                "d_decoder must be > 0".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_vl_7b_defaults_are_consistent() {
        let cfg = VisionConfig::qwen3_vl_7b();
        assert_eq!(cfg.image_size, 448);
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.n_patches, 784); // (448/16)^2
        assert_eq!(cfg.d_model, 1152);
        assert_eq!(cfg.n_heads, 16);
        assert_eq!(cfg.n_layers, 27);
        assert_eq!(cfg.spatial_merge_size, 2);
        assert_eq!(cfg.visual_tokens(), 196); // 784 / 4
        assert_eq!(cfg.head_dim(), 72); // 1152 / 16
        assert_eq!(cfg.d_mlp, 4608); // 1152 * 4
        cfg.validate().expect("default config validates");
    }

    #[test]
    fn validation_rejects_indivisible_image_patch() {
        let mut cfg = VisionConfig::qwen3_vl_7b();
        cfg.image_size = 449; // not divisible by 16
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validation_rejects_zero_d_model() {
        let mut cfg = VisionConfig::qwen3_vl_7b();
        cfg.d_model = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validation_rejects_misaligned_heads() {
        let mut cfg = VisionConfig::qwen3_vl_7b();
        cfg.n_heads = 7; // 1152 is not divisible by 7
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn visual_tokens_count() {
        let cfg = VisionConfig::qwen3_vl_7b();
        // 784 raw patches / (2*2) = 196
        assert_eq!(cfg.visual_tokens(), 196);
    }
}
