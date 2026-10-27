//! PaddleOCR-VL vision encoder and projector.

pub use lattice_model_ernie45::vision::paddleocr_vit::{
    MAX_VISION_CHECKPOINT_BYTES, MAX_VISION_HEADS, MAX_VISION_HIDDEN_SIZE,
    MAX_VISION_INTERMEDIATE_SIZE, MAX_VISION_LAYERS, MAX_VISION_PATCH_SIZE,
    MAX_VISION_POSITION_SIDE, MAX_VISION_TEXT_HIDDEN_SIZE, MAX_VISION_TOKENS, PROJECTOR_NORM_EPS,
    PaddleOcrVisionConfig, PaddleOcrVisionTrace, PaddleOcrVisionWeights, PaddleOcrVitLayerWeights,
    VISION_ROPE_THETA, paddleocr_vision_forward, paddleocr_vision_forward_trace,
};
