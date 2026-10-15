//! Manual paired timing of the real cached PaddleOCR request entry point.
//!
//! Model loading and input preparation are outside request timing. Every request
//! still assembles the image/prompt, allocates its cache, and prefills it. Budget
//! one emits from prefill; budget eight additionally performs seven decode steps.
//! Neither timing isolates decode from the other request phases.

use super::{EOS_ID, PaddleOcrVlModel};
use crate::tokenizer::gemma_bpe::GemmaBpeTokenizer;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::hint::black_box;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

const BUDGETS: [usize; 2] = [1, 8];
const SAMPLES: usize = 3;
const REVISION: &str = "c5630abae1d940eafe0697512a0325494b02ab42";

#[derive(Deserialize)]
struct Golden {
    revision: String,
    prompt_text: String,
    image: ImageGolden,
    greedy: GreedyGolden,
}

#[derive(Deserialize)]
struct ImageGolden {
    file: String,
    png_sha256: String,
    rgb_sha256: String,
    hw: [usize; 2],
}

#[derive(Deserialize)]
struct GreedyGolden {
    max_new_tokens: usize,
    tokens: Vec<u32>,
}

fn file_hash(path: &Path) -> String {
    let mut file = File::open(path).unwrap_or_else(|error| {
        panic!("open {}: {error}", path.display());
    });
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65_536];
    loop {
        let count = file.read(&mut buffer).expect("read checkpoint hash input");
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    format!("{:x}", hash.finalize())
}

fn tokens_hash(tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn check_tokens(tokens: &[u32], golden: &Golden, budget: usize, vocab: usize) {
    assert_eq!(
        tokens.len(),
        budget,
        "early termination shortens timed work"
    );
    assert_eq!(tokens, &golden.greedy.tokens[..budget], "greedy prefix");
    assert!(
        tokens
            .iter()
            .all(|&token| token != EOS_ID && (token as usize) < vocab)
    );
    if budget > 1 {
        assert!(
            tokens.windows(2).any(|pair| pair[0] != pair[1]),
            "the multi-step fixture must make different token decisions"
        );
    }
}

#[test]
#[ignore = "manual release timing; requires the pinned PaddleOCR checkpoint"]
fn bench_generate_greedy_cached_requests() {
    if cfg!(debug_assertions) {
        panic!("run the PaddleOCR timing harness with --release");
    }
    if !cfg!(feature = "f16") {
        panic!("this harness requires --features f16");
    }
    let setup = Instant::now();
    let model_dir = PathBuf::from(
        std::env::var_os("LATTICE_POCR_MODEL_DIR")
            .expect("set LATTICE_POCR_MODEL_DIR to the pinned checkpoint directory"),
    );
    assert!(model_dir.is_absolute(), "checkpoint path must be absolute");
    let checkpoint_hashes = [
        ("config.json", file_hash(&model_dir.join("config.json"))),
        (
            "preprocessor_config.json",
            file_hash(&model_dir.join("preprocessor_config.json")),
        ),
        (
            "model.safetensors",
            file_hash(&model_dir.join("model.safetensors")),
        ),
    ];
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/paddleocr_vl");
    let golden_path = fixtures.join("e2e/e2e_goldens.json");
    let golden: Golden =
        serde_json::from_slice(&std::fs::read(&golden_path).expect("read committed e2e golden"))
            .expect("parse committed e2e golden");
    assert_eq!(golden.revision, REVISION, "checkpoint revision contract");
    assert_eq!(golden.prompt_text, "OCR:");
    assert_eq!(golden.image.file, "e2e_image.png");
    assert_eq!(golden.image.hw, [252, 448]);
    assert_eq!(golden.greedy.max_new_tokens, 24);
    assert_eq!(golden.greedy.tokens.len(), 24);
    let png_path = fixtures.join("e2e").join(&golden.image.file);
    assert_eq!(file_hash(&png_path), golden.image.png_sha256);
    let image = image::load_from_memory(&std::fs::read(&png_path).expect("read committed PNG"))
        .expect("decode committed PNG")
        .to_rgb8();
    let (w, h) = (image.width() as usize, image.height() as usize);
    assert_eq!([h, w], golden.image.hw);
    let rgb = image.as_raw();
    assert_eq!(
        format!("{:x}", Sha256::digest(rgb)),
        golden.image.rgb_sha256
    );
    let tokenizer_path = fixtures.join("tokenizer/tokenizer.json");
    let tokenizer =
        GemmaBpeTokenizer::from_ernie_tokenizer_json(&tokenizer_path).expect("pinned tokenizer");
    let model_start = Instant::now();
    let model = PaddleOcrVlModel::load(&model_dir).expect("pinned checkpoint loads");
    let model_load_ms = model_start.elapsed().as_secs_f64() * 1_000.0;
    let setup_ms = setup.elapsed().as_secs_f64() * 1_000.0;
    println!(
        "ERNIE_DECODE_SCRATCH_INPUT {}",
        serde_json::json!({
            "schema": 1,
            "fixture_checkpoint_revision": REVISION,
            "checkpoint_sha256": checkpoint_hashes,
            "golden_sha256": file_hash(&golden_path),
            "tokenizer_sha256": file_hash(&tokenizer_path),
            "png_sha256": golden.image.png_sha256,
            "rgb_sha256": golden.image.rgb_sha256,
            "image_hw": [h, w],
            "prompt": golden.prompt_text,
            "expected_tokens": &golden.greedy.tokens[..8],
            "budgets": BUDGETS,
            "warmups_per_budget": 1,
            "samples_per_budget": SAMPLES,
            "model_load_ms": model_load_ms,
            "setup_including_hashing_and_model_load_ms": setup_ms,
            "timed_scope": "whole_generate_greedy_request_including_vision_prompt_cache_prefill_and_decode",
            "isolated_decode_timing": false,
        })
    );

    for budget in BUDGETS {
        let tokens = model
            .generate_greedy(&tokenizer, rgb, h, w, &golden.prompt_text, budget)
            .expect("warmup cached generation");
        check_tokens(&tokens, &golden, budget, model.config().vocab_size);
        black_box(tokens);
        println!("ERNIE_DECODE_SCRATCH_WARMUP budget={budget} tokens={budget}");
    }

    let mut requests = 0usize;
    for sample in 0..SAMPLES {
        let order = if sample % 2 == 0 {
            BUDGETS
        } else {
            [BUDGETS[1], BUDGETS[0]]
        };
        for budget in order {
            let start = Instant::now();
            let tokens = model
                .generate_greedy(
                    black_box(&tokenizer),
                    black_box(rgb),
                    h,
                    w,
                    black_box(&golden.prompt_text),
                    budget,
                )
                .expect("timed cached generation");
            let request_ms = start.elapsed().as_secs_f64() * 1_000.0;
            check_tokens(&tokens, &golden, budget, model.config().vocab_size);
            assert!(request_ms.is_finite() && request_ms > 0.0);
            println!(
                "ERNIE_DECODE_SCRATCH_SAMPLE {}",
                serde_json::json!({
                    "schema": 1,
                    "sample": sample,
                    "budget": budget,
                    "generated_tokens": tokens.len(),
                    "required_decode_steps": budget - 1,
                    "request_ms": request_ms,
                    "tokens_sha256": tokens_hash(&tokens),
                })
            );
            black_box(tokens);
            requests += 1;
        }
    }
    assert_eq!(requests, BUDGETS.len() * SAMPLES);
    println!("ERNIE_DECODE_SCRATCH_COMPLETE groups=2 samples=6 warmups=2 requests=8");
}
