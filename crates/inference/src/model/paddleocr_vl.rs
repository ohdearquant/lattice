//! PaddleOCR-VL model and prompt preparation.
//!
//! **Compatibility path.** The items here are re-exported from `lattice-model-ernie45` so that code
//! written against the pre-split `lattice-inference` keeps compiling. These low-level re-exports
//! are not a stable API and will be removed at 1.0; import from `lattice-model-ernie45` directly.

pub use lattice_model_ernie45::model::paddleocr_vl::{
    EOS_ID, IMAGE_END_ID, IMAGE_PLACEHOLDER_ID, IMAGE_START_ID, PaddleOcrVlModel, PrefillTrace,
    argmax_u32,
};
