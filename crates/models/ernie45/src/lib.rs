//! ERNIE4.5 configuration, checkpoint loading, CPU and optional Metal execution.
//! PaddleOCR-VL vision and prompt preparation are enabled by `paddleocr-vl`.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

pub mod forward;
pub mod model;

#[cfg(feature = "paddleocr-vl")]
pub mod vision;
