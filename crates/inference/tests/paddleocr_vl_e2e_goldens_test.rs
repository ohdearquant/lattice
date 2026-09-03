//! PaddleOCR-VL-1.6 end-to-end CPU forward vs the HF reference: one image
//! plus the `"OCR:"` prompt, greedy decode, cached and uncached.
//!
//! The committed fixture (`fixtures/paddleocr_vl/e2e/e2e_goldens.json` +
//! `e2e_image.png`) was captured from the pinned checkpoint's own
//! modeling/processing source running under HF transformers 4.57.1 (torch
//! 2.13.0, PIL 12.3.0, CPU f32, eager attention, no cache) — the same
//! reference stack as the decoder/vision/preprocess/tokenizer gates this
//! composes. This test drives `PaddleOcrVlModel::prefill` +
//! `generate_greedy` over the same image and prompt and holds every field
//! the fixture records:
//!
//! - image decode (sha256 of the RGB8 bytes),
//! - `grid_thw` / `resized_hw`,
//! - prompt ids (exact, one-pass HF tokenization of the template),
//! - `position_ids` (exact, all three mrope rows),
//! - projector rows (first/last first8, mean_abs),
//! - spliced embeddings (last-token first8, mean_abs),
//! - prompt logits (argmax at every position, last-token first8 / top-5 /
//!   mean_abs),
//! - the 24 greedy tokens (exact, in order), produced by the KV-cached
//!   loop and again by the uncached re-forward loop, which must agree
//!   with each other as well as with the fixture.
//!
//! **Fail-closed contract** (mirrors `paddleocr_vl_decoder_goldens_test.rs`):
//! the ~1.9 GB checkpoint is not committed. With `LATTICE_POCR_MODEL_DIR`
//! unset and the default `~/.lattice/models/paddleocr-vl-1.6` absent, this
//! test prints a skip line and returns. With `LATTICE_POCR_GATE_ENFORCE=1`,
//! a missing checkpoint panics instead of skipping.
//!
//! Run (release only — the uncached reference loop re-forwards the ~0.9B
//! decoder over the full sequence per step):
//! ```bash
//! cargo test --release -p lattice-inference --features f16 \
//!     --test paddleocr_vl_e2e_goldens_test -- --nocapture
//! ```

#[test]
fn e2e_forward_matches_hf_reference() {
    // The pinned checkpoint stores every tensor as BF16; reading it through
    // `SafetensorsFile` requires the `f16` feature. Without it the gate
    // skips (or panics under enforce) exactly like a missing checkpoint,
    // mirroring `paddleocr_vl_decoder_goldens_test.rs`.
    #[cfg(not(feature = "f16"))]
    {
        if std::env::var("LATTICE_POCR_GATE_ENFORCE").as_deref() == Ok("1") {
            panic!(
                "LATTICE_POCR_GATE_ENFORCE=1 but the `f16` feature is not enabled — the \
                 checkpoint's BF16 tensors require it"
            );
        }
        eprintln!("SKIP paddleocr_vl_e2e_goldens_test: f16 feature disabled");
    }
    #[cfg(feature = "f16")]
    gate::run();
}

/// Everything that needs the `f16` feature (the BF16 checkpoint read and
/// the golden comparison) lives here so the default-feature build carries
/// no unreachable items — `cargo clippy --all-targets -D warnings` without
/// `f16` is a CI gate.
#[cfg(feature = "f16")]
mod gate {
    use crate::gate_types::*;
    use lattice_inference::model::paddleocr_vl::PaddleOcrVlModel;
    use lattice_inference::tokenizer::gemma_bpe::GemmaBpeTokenizer;
    use std::path::PathBuf;

    fn model_dir() -> Option<PathBuf> {
        let dir = match std::env::var_os("LATTICE_POCR_MODEL_DIR") {
            Some(d) => PathBuf::from(d),
            None => {
                PathBuf::from(std::env::var_os("HOME")?).join(".lattice/models/paddleocr-vl-1.6")
            }
        };
        if dir.join("model.safetensors").is_file() && dir.join("config.json").is_file() {
            Some(dir)
        } else {
            None
        }
    }

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/paddleocr_vl/e2e")
    }

    pub fn run() {
        let Some(dir) = model_dir() else {
            if std::env::var("LATTICE_POCR_GATE_ENFORCE").as_deref() == Ok("1") {
                panic!(
                    "LATTICE_POCR_GATE_ENFORCE=1 but the PaddleOCR-VL checkpoint is missing \
                     (set LATTICE_POCR_MODEL_DIR or place it at ~/.lattice/models/paddleocr-vl-1.6)"
                );
            }
            eprintln!(
                "SKIP paddleocr_vl_e2e_goldens_test: checkpoint not found (set \
                 LATTICE_POCR_MODEL_DIR to run)"
            );
            return;
        };

        let golden: Golden = serde_json::from_str(
            &std::fs::read_to_string(fixture_dir().join("e2e_goldens.json"))
                .expect("read e2e_goldens.json"),
        )
        .expect("valid e2e_goldens.json");
        assert_eq!(
            golden.revision, "c5630abae1d940eafe0697512a0325494b02ab42",
            "fixture revision drifted from the pinned checkpoint"
        );

        // 1. Image decode: the PNG bytes must be exactly the fixture, and
        // the decoded RGB8 plane exactly the reference's input.
        let png_bytes =
            std::fs::read(fixture_dir().join(&golden.image.file)).expect("read e2e_image.png");
        assert_eq!(
            hex(sha256(&png_bytes)),
            golden.image.png_sha256,
            "e2e_image.png bytes drifted from the fixture hash"
        );
        let image = image::load_from_memory(&png_bytes)
            .expect("decode PNG")
            .to_rgb8();
        let (w, h) = (image.width() as usize, image.height() as usize);
        assert_eq!(
            (h, w),
            (golden.image.hw[0] as usize, golden.image.hw[1] as usize)
        );
        let rgb = image.as_raw();
        assert_eq!(
            hex(sha256(rgb)),
            golden.image.rgb_sha256,
            "decoded RGB8 plane does not hash to the reference input"
        );

        let model = PaddleOcrVlModel::load(&dir).expect("checkpoint loads");
        // The checkpoint directory on this target carries no
        // tokenizer.json; the pinned tokenizer is the in-repo fixture the
        // tokenizer gate already holds.
        let tokenizer = GemmaBpeTokenizer::from_ernie_tokenizer_json(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/paddleocr_vl/tokenizer/tokenizer.json"),
        )
        .expect("pinned tokenizer loads");

        // 2. Image processor + vision + prompt + splice + decoder prefill.
        let t0 = std::time::Instant::now();
        let prefill = model
            .prefill(&tokenizer, rgb, h, w, &golden.prompt_text)
            .expect("prefill");
        let prefill_s = t0.elapsed().as_secs_f64();

        assert_eq!(
            (prefill.grid_thw.1, prefill.grid_thw.2),
            (
                golden.image.grid_thw[1] as usize,
                golden.image.grid_thw[2] as usize
            ),
            "grid_thw"
        );
        assert_eq!(
            prefill.grid_thw.0, golden.image.grid_thw[0] as usize,
            "grid temporal"
        );
        assert_eq!(
            (prefill.resized_hw.0, prefill.resized_hw.1),
            (
                golden.image.resized_hw[0] as usize,
                golden.image.resized_hw[1] as usize
            ),
            "resized_hw"
        );
        assert_eq!(
            golden.image.num_image_tokens as usize,
            (golden.image.grid_thw[1] / 2) as usize * (golden.image.grid_thw[2] / 2) as usize,
            "fixture placeholder count vs merged grid"
        );

        // 3. Prompt ids: exact one-pass tokenization of the template.
        assert_eq!(
            prefill.prompt_ids, golden.input_ids,
            "prompt ids diverge from the HF template tokenization"
        );

        // 4. Position ids: all three mrope rows, every column.
        for row in 0..3 {
            for (col, (g, e)) in prefill
                .positions
                .iter()
                .map(|p| p[row])
                .zip(golden.position_ids[row].iter())
                .enumerate()
            {
                assert_eq!(
                    g, *e,
                    "position_ids row {row} col {col}: lattice {g} vs HF {e}"
                );
            }
        }

        // 5. Projector rows (the spliced-in image embedding).
        let hidden = model.config().hidden_size;
        let n_img = golden.image.num_image_tokens as usize;
        let proj = &prefill.projector_rows;
        assert_eq!(proj.len(), n_img * hidden, "projector row count");
        let mut worst = 0f32;
        worst = worst.max(assert_slice_close(
            &proj[..8],
            &golden.projector.first_row_first8,
            "projector first_row_first8",
        ));
        worst = worst.max(assert_slice_close(
            &proj[(n_img - 1) * hidden..][..8],
            &golden.projector.last_row_first8,
            "projector last_row_first8",
        ));
        worst = worst.max(assert_scalar_close(
            mean_abs(proj),
            golden.projector.mean_abs,
            "projector mean_abs",
        ));

        // 6. Spliced embeddings: last prompt token (text row) + aggregate.
        let spliced = &prefill.spliced_embeds;
        assert_eq!(
            spliced.len(),
            prefill.prompt_ids.len() * hidden,
            "spliced length"
        );
        worst = worst.max(assert_slice_close(
            &spliced[(prefill.prompt_ids.len() - 1) * hidden..][..8],
            &golden.spliced_embeds.last_tok_first8,
            "spliced last_tok_first8",
        ));
        worst = worst.max(assert_scalar_close(
            mean_abs(spliced),
            golden.spliced_embeds.mean_abs,
            "spliced mean_abs",
        ));

        // 7. Prompt logits: greedy choice at every position, exact.
        let vocab = model.config().vocab_size;
        assert_eq!(vocab, golden.prompt_logits.vocab as usize, "vocab size");
        let s = prefill.prompt_ids.len();
        let logits = &prefill.logits;
        assert_eq!(logits.len(), s * vocab, "logits shape");
        for (t, &expect_argmax) in golden.prompt_logits.argmax_per_pos.iter().enumerate() {
            let row = &logits[t * vocab..][..vocab];
            let argmax = argmax(row);
            assert_eq!(
                argmax, expect_argmax as usize,
                "pos {t}: greedy token diverged from HF ({argmax} vs HF {expect_argmax})"
            );
        }
        let last_row = &logits[(s - 1) * vocab..][..vocab];
        worst = worst.max(assert_slice_close(
            &last_row[..8],
            &golden.prompt_logits.last_tok_first8,
            "prompt logits last_tok_first8",
        ));
        for &(id, val) in &golden.prompt_logits.last_tok_top5 {
            assert!(
                close(last_row[id], val),
                "prompt logits top-5 id {id}: lattice {} vs HF {val}",
                last_row[id]
            );
        }
        worst = worst.max(assert_scalar_close(
            mean_abs(last_row),
            golden.prompt_logits.last_tok_mean_abs,
            "prompt logits last_tok_mean_abs",
        ));

        // 8. Greedy decode with the KV cache: every token reproduced
        // exactly. The fixture records per-step top1/top2 margins; the
        // smallest recorded margin is 3.78 (step 7: 16.072 - 12.289), far
        // above the 0.05 threshold, so all 24 steps are compared exactly
        // with no early stop.
        let t1 = std::time::Instant::now();
        let generated = model
            .generate_greedy(
                &tokenizer,
                rgb,
                h,
                w,
                &golden.prompt_text,
                golden.greedy.max_new_tokens,
            )
            .expect("greedy decode");
        let greedy_s = t1.elapsed().as_secs_f64();
        assert_eq!(
            generated.len(),
            golden.greedy.tokens.len(),
            "generated {} tokens; golden has {}",
            generated.len(),
            golden.greedy.tokens.len()
        );
        for (i, (g, e)) in generated
            .iter()
            .zip(golden.greedy.tokens.iter())
            .enumerate()
        {
            assert_eq!(g, e, "greedy step {i}");
        }

        // 9. The cache's own acceptance: the cached loop above and the
        // uncached re-forward loop must produce the same tokens on the
        // same input. Holding both against the golden would not settle
        // this — the golden is 24 tokens of one image, and a cache that
        // diverged only after the horizon, or only on another prompt,
        // would still match it. The two loops share every stage except
        // the attention read, so an equality here is the invariant
        // stated as an equation.
        let t2 = std::time::Instant::now();
        let uncached = model
            .generate_greedy_uncached(
                &tokenizer,
                rgb,
                h,
                w,
                &golden.prompt_text,
                golden.greedy.max_new_tokens,
            )
            .expect("uncached greedy decode");
        let uncached_s = t2.elapsed().as_secs_f64();
        assert_eq!(
            generated, uncached,
            "cached and uncached greedy sequences diverged"
        );

        // Read the two greedy timings with care, and do not quote them as a
        // decode benchmark. Each call runs the whole pipeline, image
        // processor and vision encoder included, and on this fixture that
        // shared prefix dominates: the prompt is 157 tokens, so the 24
        // full-sequence decoder forwards the uncached loop makes, against
        // the cached loop's 24 single-token steps, are worth a few seconds
        // against a total near 230s. Measured 2026-09-03, one run: cached
        // 229.39s against uncached 224.49s, i.e. the difference came out
        // negative, which is the shared vision pass varying by more than
        // the decoder work being compared. What the cache saves per step
        // is a structural property of the code (one token instead of the
        // whole sequence); isolating it is a bench-harness job.
        println!(
            "paddleocr_vl_e2e: prefill {prefill_s:.2}s, greedy cached {greedy_s:.2}s vs \
             uncached {uncached_s:.2}s ({} steps, single run), worst |diff| {worst:.2e} \
             (ATOL=RTOL={ATOL:e}) — all fields within tolerance, greedy exact {} of {}, \
             cached == uncached",
            generated.len(),
            generated.len(),
            golden.greedy.tokens.len()
        );
    }
}

#[cfg(feature = "f16")]
mod gate_types {
    use serde::Deserialize;
    use sha2::{Digest, Sha256};

    /// Golden shape of `e2e_goldens.json`.
    #[derive(Deserialize)]
    pub struct Golden {
        pub revision: String,
        pub image: ImageGolden,
        pub prompt_text: String,
        pub input_ids: Vec<u32>,
        pub position_ids: Vec<Vec<u32>>,
        #[allow(dead_code)]
        pub rope_delta: i64,
        pub projector: ProjectorGolden,
        pub spliced_embeds: SplicedGolden,
        pub prompt_logits: PromptLogitsGolden,
        pub greedy: GreedyGolden,
    }

    #[derive(Deserialize)]
    pub struct ImageGolden {
        pub file: String,
        pub png_sha256: String,
        pub rgb_sha256: String,
        pub hw: [u64; 2],
        pub resized_hw: [u64; 2],
        pub grid_thw: [u64; 3],
        pub num_image_tokens: u64,
    }

    #[derive(Deserialize)]
    pub struct ProjectorGolden {
        pub first_row_first8: Vec<f32>,
        pub last_row_first8: Vec<f32>,
        pub mean_abs: f32,
    }

    #[derive(Deserialize)]
    pub struct SplicedGolden {
        pub last_tok_first8: Vec<f32>,
        pub mean_abs: f32,
    }

    #[derive(Deserialize)]
    pub struct PromptLogitsGolden {
        pub vocab: u64,
        pub argmax_per_pos: Vec<u64>,
        pub last_tok_first8: Vec<f32>,
        pub last_tok_top5: Vec<(usize, f32)>,
        pub last_tok_mean_abs: f32,
    }

    #[derive(Deserialize)]
    pub struct GreedyGolden {
        pub max_new_tokens: usize,
        pub tokens: Vec<u32>,
        #[allow(dead_code)]
        pub text: String,
        #[allow(dead_code)]
        pub steps: Vec<serde_json::Value>,
    }

    /// Element-wise tolerance, inherited from the decoder gate. The two
    /// sides load bit-identical bf16 weights and compute in f32, so
    /// divergence is accumulation-order noise (BLAS vs torch GEMM
    /// reduction trees) compounded across 18 layers plus the vision tower.
    /// Measured worst |diff| on the golden-compared fields of this fixture
    /// (run 2026-09-01, release, this machine): 9.54e-5 — below the
    /// decoder gate's 3.1e-4 floor, consistent with the extra vision
    /// tower not adding measurable divergence. The bound keeps ~20x
    /// headroom over that without admitting any structural
    /// error: an h/w position swap changes every image-block angle
    /// (argmax diverges immediately), a section-row offset in the doubled
    /// mrope layout moves logits by O(1e-2) while the 1-D decoder gate
    /// still passes, and a reversed splice replaces 144 of 157 embedding
    /// rows.
    pub const ATOL: f32 = 2e-3;
    pub const RTOL: f32 = 2e-3;

    pub fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= ATOL + RTOL * b.abs()
    }

    pub fn assert_slice_close(actual: &[f32], expected: &[f32], what: &str) -> f32 {
        assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
        let mut worst = 0f32;
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                close(a, e),
                "{what}[{i}]: lattice {a} vs HF {e} (|diff| {})",
                (a - e).abs()
            );
            worst = worst.max((a - e).abs());
        }
        worst
    }

    pub fn assert_scalar_close(a: f32, b: f32, what: &str) -> f32 {
        assert!(
            close(a, b),
            "{what}: lattice {a} vs HF {b} (|diff| {})",
            (a - b).abs()
        );
        (a - b).abs()
    }

    pub fn mean_abs(buf: &[f32]) -> f32 {
        buf.iter().map(|v| v.abs()).sum::<f32>() / buf.len() as f32
    }

    pub fn argmax(row: &[f32]) -> usize {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .expect("non-empty row")
    }

    pub fn sha256(bytes: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hasher.finalize().into()
    }

    pub fn hex(bytes: [u8; 32]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}
