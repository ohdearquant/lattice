//! PaddleOCR-VL model and prompt preparation.

pub use lattice_model_ernie45::model::paddleocr_vl::{
    EOS_ID, IMAGE_END_ID, IMAGE_PLACEHOLDER_ID, IMAGE_START_ID, PaddleOcrVlModel, PrefillTrace,
    argmax_u32,
};
