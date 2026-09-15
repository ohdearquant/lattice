//! Ad-hoc harness: run PaddleOCR-VL-1.6 greedy decode over an arbitrary
//! document image and print the emitted token ids and text.
//!
//! Every other PaddleOCR-VL gate in this directory is fixture-driven — it
//! compares one committed image against one committed golden. There is no way
//! to point the model at an image that is not a fixture, which makes any
//! cross-stack comparison against a second document impossible without one.
//! This fills that gap and nothing else: it asserts no goldens, so it is not a
//! gate and cannot fail as one.
//!
//! `#[ignore]`d, so `cargo test` never picks it up; it runs only when named.
//!
//! ```bash
//! LATTICE_POCR_IMAGE=/path/to/page.png \
//!   cargo test --release -p lattice-inference --features f16 \
//!     --test paddleocr_vl_adhoc_ocr -- --ignored --nocapture
//! ```
//!
//! Environment:
//! - `LATTICE_POCR_IMAGE`      required; PNG or JPEG, decoded to RGB8.
//! - `LATTICE_POCR_PROMPT`     default `"OCR:"` — the prompt the fixtures use.
//! - `LATTICE_POCR_MAX_TOKENS` default 24 — matches the e2e fixture's window.
//! - `LATTICE_POCR_MODEL_DIR`  default `~/.lattice/models/paddleocr-vl-1.6`.
//!
//! Release only, for the same reason the e2e gate is: this is a ~0.9B decoder
//! running on CPU, and prefill over a full-page image is minutes, not seconds.

#![cfg(feature = "f16")]

use std::path::PathBuf;

use lattice_inference::model::paddleocr_vl::PaddleOcrVlModel;
use lattice_inference::tokenizer::common::Tokenizer;
use lattice_inference::tokenizer::gemma_bpe::GemmaBpeTokenizer;

fn model_dir() -> PathBuf {
    match std::env::var_os("LATTICE_POCR_MODEL_DIR") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(std::env::var_os("HOME").expect("HOME is set"))
            .join(".lattice/models/paddleocr-vl-1.6"),
    }
}

#[test]
#[ignore = "ad-hoc harness: needs LATTICE_POCR_IMAGE and the ~1.9 GB checkpoint"]
fn ocr_arbitrary_image() {
    // Panic rather than skip on a missing input. This harness is only ever run
    // deliberately, so a silent skip would print a reassuring nothing at the
    // exact moment the caller is waiting for a number.
    let image_path = PathBuf::from(
        std::env::var_os("LATTICE_POCR_IMAGE")
            .expect("set LATTICE_POCR_IMAGE to the image to run OCR over"),
    );
    let prompt = std::env::var("LATTICE_POCR_PROMPT").unwrap_or_else(|_| "OCR:".to_string());
    let max_new_tokens: usize = std::env::var("LATTICE_POCR_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);

    let dir = model_dir();
    assert!(
        dir.join("model.safetensors").exists(),
        "checkpoint missing at {} (set LATTICE_POCR_MODEL_DIR)",
        dir.display()
    );

    let png_bytes =
        std::fs::read(&image_path).unwrap_or_else(|e| panic!("read {}: {e}", image_path.display()));
    let image = image::load_from_memory(&png_bytes)
        .unwrap_or_else(|e| panic!("decode {}: {e}", image_path.display()))
        .to_rgb8();
    let (w, h) = (image.width() as usize, image.height() as usize);
    let rgb = image.as_raw();

    let model = PaddleOcrVlModel::load(&dir).expect("checkpoint loads");
    // The checkpoint carries no tokenizer.json on this target; the pinned
    // tokenizer is the in-repo fixture the tokenizer gate already holds, so
    // this harness and the gates tokenize identically.
    let tokenizer = GemmaBpeTokenizer::from_ernie_tokenizer_json(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/paddleocr_vl/tokenizer/tokenizer.json"),
    )
    .expect("pinned tokenizer loads");

    println!("image      {} ({w}x{h} RGB8)", image_path.display());
    println!("prompt     {prompt:?}   max_new_tokens {max_new_tokens}");

    let t0 = std::time::Instant::now();
    let ids = model
        .generate_greedy(&tokenizer, rgb, h, w, &prompt, max_new_tokens)
        .expect("greedy decode");
    let elapsed = t0.elapsed().as_secs_f64();

    let text = tokenizer
        .decode(&ids)
        .unwrap_or_else(|| "<detokenize returned None>".to_string());

    println!("elapsed    {elapsed:.2}s");
    println!("emitted    {} tokens", ids.len());
    println!("ids        {ids:?}");
    println!("text       {text:?}");
}
