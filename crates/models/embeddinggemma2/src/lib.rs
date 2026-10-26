//! EmbeddingGemma 2 text and image encoders with optional Metal execution.

pub mod config;
pub mod encoder;
pub mod family;
pub mod image;
pub mod metal;
pub mod model;
mod vision;

#[doc(inline)]
pub use encoder::EmbeddingGemma2Encoder;
#[doc(inline)]
pub use model::EmbeddingGemma2Model;

/// Shader sources for facade measurement targets.
#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
#[doc(hidden)]
pub mod shaders {
    /// The unchanged `embeddinggemma2.metal` source.
    pub const EMBEDDINGGEMMA2: &str = include_str!("../shaders/embeddinggemma2.metal");
    /// The unchanged `embeddinggemma2_attention.metal` source.
    pub const EMBEDDINGGEMMA2_ATTENTION: &str =
        include_str!("../shaders/embeddinggemma2_attention.metal");
    /// The unchanged `embeddinggemma2_vision.metal` source.
    pub const EMBEDDINGGEMMA2_VISION: &str =
        include_str!("../shaders/embeddinggemma2_vision.metal");
    /// The unchanged `gemm_f32_sg.metal` source.
    pub const GEMM_F32_SG: &str = include_str!("../shaders/gemm_f32_sg.metal");
    /// The unchanged `attention_sg_hd64.metal` source.
    pub const ATTENTION_SG_HD64: &str = include_str!("../shaders/attention_sg_hd64.metal");
}
