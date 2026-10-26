//! PaddleOCR-VL image preprocessing.

pub use lattice_model_ernie45::vision::paddleocr_preprocess::{
    MAX_INPUT_IMAGE_SIDE, MAX_INPUT_PIXELS, MAX_PIXEL_BUDGET, MAX_RESIZE_FACTOR,
    MAX_RESIZED_IMAGE_SIDE, PaddleOcrImageProcessorConfig, PreprocessedImage,
    SMART_RESIZE_PIXEL_SLACK, preprocess_rgb8, resize_bicubic_rgb8, smart_resize,
};
