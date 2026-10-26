use super::*;
use crate::vision::VisionError;
use lattice_arch::qwen35_config::VisionModelConfig;

#[cfg(not(feature = "image"))]
#[test]
fn image_decoders_require_image_feature() {
    #[cfg(feature = "embeddinggemma2")]
    use crate::error::InferenceError;
    #[cfg(feature = "embeddinggemma2")]
    use crate::model::embeddinggemma2_image::EmbeddingGemma2Rgb8Image;
    use crate::vision::preprocess::{PreprocessConfig, preprocess};

    let bytes = b"\x89PNG\r\n\x1a\n";
    let cfg = tiny_cfg();
    let expected = "image decoding is not compiled in; enable the `image` feature";
    let errors = [
        preprocess(bytes, &PreprocessConfig::qwen3_vl()).unwrap_err(),
        preprocess_qwen35_image(bytes, &cfg).unwrap_err(),
        preprocess_qwen35_image_for_serve(bytes, &cfg).unwrap_err(),
        preprocess_qwen35_image_for_embedding_serve(bytes, &cfg).unwrap_err(),
    ];
    for error in errors {
        assert!(matches!(error, VisionError::ImageDecode(message) if message == expected));
    }
    #[cfg(feature = "embeddinggemma2")]
    assert!(matches!(
        EmbeddingGemma2Rgb8Image::from_encoded(bytes),
        Err(InferenceError::InvalidInput(message))
            if message == format!("EmbeddingGemma 2 image: {expected}")
    ));
    let mut zero_patch = PreprocessConfig::qwen3_vl();
    zero_patch.patch_size = 0;
    assert!(matches!(
        preprocess(bytes, &zero_patch),
        Err(VisionError::InvalidConfig(_))
    ));
}

fn tiny_cfg() -> VisionModelConfig {
    VisionModelConfig {
        depth: 1,
        hidden_size: 8,
        num_heads: 2,
        patch_size: 2,
        spatial_merge_size: 2,
        out_hidden_size: 8,
        temporal_patch_size: 1,
        num_position_embeddings: 16, // side = 4
        in_channels: 3,
        deepstack_visual_indexes: vec![],
        intermediate_size: None,
    }
}
