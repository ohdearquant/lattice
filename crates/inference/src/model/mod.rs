//! Model module index for BERT, BitNet config, cross-encoder, Qwen, Qwen3.5, and Qwen3.5 config, with BERT/CrossEncoder/Qwen/Qwen3.5 type re-exports.
#[cfg(feature = "bert")]
pub mod bert;
pub mod bitnet_config;
#[cfg(feature = "bert")]
pub mod cross_encoder;
#[cfg(feature = "embeddinggemma2")]
pub mod embeddinggemma2;
pub mod embeddinggemma2_config;
/// RGB image input and internal patch preparation for EmbeddingGemma 2.
#[cfg(feature = "embeddinggemma2")]
pub mod embeddinggemma2_image;
#[cfg(feature = "ernie45")]
pub mod ernie45;
#[cfg(feature = "gemma4")]
pub mod gemma4_cache;
pub mod gemma4_config;
#[cfg(feature = "gemma4")]
pub mod gemma4_model;
pub mod gemma4_ops;
#[cfg(feature = "gemma4")]
pub mod gemma4_preflight;
#[cfg(feature = "paddleocr-vl")]
pub mod paddleocr_vl;
pub mod qwen;
pub mod qwen35;
pub mod qwen35_config;

// Preserve the original owning names and model-level loading module.
#[cfg(feature = "bert")]
pub use self::bert::{BertConfig, BertModel, weight_loading};
#[cfg(feature = "bert")]
pub use self::cross_encoder::CrossEncoderModel;
#[cfg(feature = "embeddinggemma2")]
pub use self::embeddinggemma2::EmbeddingGemma2Model;
// Re-export key types from other model modules
#[cfg(feature = "gemma4")]
pub use self::gemma4_model::Gemma4Model;
pub use self::qwen::QwenConfig;
#[cfg(feature = "qwen3")]
pub use self::qwen::{LayerTimings, ProfileTimings, QwenModel};
pub use self::qwen35::{HiddenPooling, Qwen35Model};
#[deprecated(
    since = "0.11.0",
    note = "moved under ADR-092; use `lattice_inference::GenerateConfig`"
)]
pub type GenerateConfig = crate::generation::GenerateConfig;
