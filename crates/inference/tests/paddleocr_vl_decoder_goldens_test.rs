//! PaddleOCR-VL ERNIE-4.5 text-decoder forward vs the HF reference.
//!
//! The committed fixture (`fixtures/paddleocr_vl/decoder/decoder_goldens.json`)
//! holds per-checkpoint activation summaries and logits captured from the
//! pinned checkpoint's own modeling source (revision noted in the fixture)
//! running under HF transformers on CPU in f32 (eager attention, no cache).
//! This test replays the same token sequences through
//! `lattice_inference::model::ernie45` and holds every checkpoint against the
//! goldens.
//!
//! **Fail-closed contract** (mirrors `vision_s3_vit_forward_test.rs`): the
//! ~1.9 GB checkpoint is not committed. With `LATTICE_POCR_MODEL_DIR` unset
//! and the default `~/.lattice/models/paddleocr-vl-1.6` absent, this test
//! prints a skip line and returns. With `LATTICE_POCR_GATE_ENFORCE=1`, a
//! missing checkpoint panics instead of skipping.
//!
//! Run:
//! ```bash
//! cargo test --release -p lattice-inference --features f16 \
//!     --test paddleocr_vl_decoder_goldens_test -- --nocapture
//! ```

#[test]
fn decoder_forward_matches_hf_reference() {
    // The pinned checkpoint stores every text tensor as BF16; reading it
    // through `SafetensorsFile` requires the `f16` feature. Without it the
    // gate skips (or panics under enforce) exactly like a missing checkpoint,
    // mirroring `vision_s3_vit_forward_test.rs`.
    #[cfg(not(feature = "f16"))]
    {
        if std::env::var("LATTICE_POCR_GATE_ENFORCE").as_deref() == Ok("1") {
            panic!(
                "LATTICE_POCR_GATE_ENFORCE=1 but the `f16` feature is not enabled — the \
                 checkpoint's BF16 tensors require it"
            );
        }
        eprintln!("SKIP paddleocr_vl_decoder_goldens_test: f16 feature disabled");
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
    use lattice_inference::model::ernie45::{Ernie45Config, Ernie45Model, Ernie45Weights};
    use lattice_inference::weights::SafetensorsFile;
    use serde::Deserialize;
    use std::path::PathBuf;

    /// Absolute floor plus relative term for element-wise activation and logit
    /// comparisons: the two sides load bit-identical bf16 weights and both
    /// compute in f32, so divergence is pure accumulation-order noise (BLAS vs
    /// torch GEMM reduction trees) compounded across 18 layers. Measured
    /// worst error on the golden-compared fields across the four fixture cases
    /// was 3.1e-4 (full-tensor worst 3.7e-3, concentrated on layer-17
    /// massive-activation dims where |x| ~ 1e2, i.e. ~4e-5 relative); the bounds
    /// below leave ~6x headroom over that without admitting any structural error
    /// (a RoPE pairing flip, a wrong GQA head mapping, or an approximate-exp
    /// softmax each move activations by O(1e-2) or more — all three were run as
    /// mutations against this gate and each reddened it).
    const ATOL: f32 = 2e-3;
    const RTOL: f32 = 2e-3;

    #[derive(Deserialize)]
    struct Golden {
        revision: String,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        id: String,
        ids: Vec<u32>,
        checkpoints: Vec<Checkpoint>,
        logits: LogitsGolden,
    }

    #[derive(Deserialize)]
    struct Checkpoint {
        name: String,
        last_tok_first8: Vec<f32>,
        mean_abs: f32,
    }

    #[derive(Deserialize)]
    struct LogitsGolden {
        argmax_per_pos: Vec<usize>,
        last_tok_first8: Vec<f32>,
        last_tok_top5: Vec<(usize, f32)>,
        last_tok_mean_abs: f32,
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= ATOL + RTOL * b.abs()
    }

    fn assert_slice_close(actual: &[f32], expected: &[f32], what: &str) -> f32 {
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

    pub fn run() {
        let Some(dir) = model_dir() else {
            if std::env::var("LATTICE_POCR_GATE_ENFORCE").as_deref() == Ok("1") {
                panic!(
                    "LATTICE_POCR_GATE_ENFORCE=1 but the PaddleOCR-VL checkpoint is missing \
                     (set LATTICE_POCR_MODEL_DIR or place it at ~/.lattice/models/paddleocr-vl-1.6)"
                );
            }
            eprintln!(
                "SKIP paddleocr_vl_decoder_goldens_test: checkpoint not found (set \
                 LATTICE_POCR_MODEL_DIR to run)"
            );
            return;
        };

        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/paddleocr_vl/decoder/decoder_goldens.json");
        let golden: Golden =
            serde_json::from_str(&std::fs::read_to_string(&fixture).expect("read fixture"))
                .expect("valid decoder_goldens.json");
        assert_eq!(
            golden.revision, "c5630abae1d940eafe0697512a0325494b02ab42",
            "fixture revision drifted from the pinned checkpoint"
        );
        assert!(
            golden.cases.len() >= 4,
            "goldens shrank to {} cases",
            golden.cases.len()
        );

        let cfg = Ernie45Config::from_config_json(&dir.join("config.json")).expect("config loads");
        let mut source =
            SafetensorsFile::open(&dir.join("model.safetensors")).expect("open weights");
        let weights = Ernie45Weights::load(&mut source, &cfg).expect("weights load");
        let model = Ernie45Model::new(cfg, weights);

        for case in &golden.cases {
            let trace = model.forward_trace(&case.ids).expect("forward");
            let s = case.ids.len();
            let h = model.config().hidden_size;
            let vocab = model.config().vocab_size;

            // Checkpoints arrive in generator order: embed, layer_0..N-1, final_norm.
            assert_eq!(
                case.checkpoints.len(),
                model.config().num_hidden_layers + 2,
                "case {}: unexpected checkpoint count",
                case.id
            );
            let mut worst = 0f32;
            for (idx, ck) in case.checkpoints.iter().enumerate() {
                let buf: &[f32] = match idx {
                    0 => &trace.embed,
                    i if i <= model.config().num_hidden_layers => &trace.layer_outputs[i - 1],
                    _ => &trace.final_norm,
                };
                let expected_name = match idx {
                    0 => "embed".to_string(),
                    i if i <= model.config().num_hidden_layers => format!("layer_{}", i - 1),
                    _ => "final_norm".to_string(),
                };
                assert_eq!(ck.name, expected_name, "case {}: checkpoint order", case.id);
                let last = &buf[(s - 1) * h..][..8.min(h)];
                worst = worst.max(assert_slice_close(
                    last,
                    &ck.last_tok_first8,
                    &format!("case {} {} last_tok_first8", case.id, ck.name),
                ));
                let mean_abs = buf.iter().map(|v| v.abs()).sum::<f32>() / buf.len() as f32;
                assert!(
                    close(mean_abs, ck.mean_abs),
                    "case {} {}: mean_abs {mean_abs} vs HF {} — aggregate drift",
                    case.id,
                    ck.name,
                    ck.mean_abs
                );
            }

            // Logits: greedy choice at every position must agree exactly.
            for (t, &expect_argmax) in case.logits.argmax_per_pos.iter().enumerate() {
                let row = &trace.logits[t * vocab..][..vocab];
                let argmax = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .expect("non-empty row");
                assert_eq!(
                    argmax, expect_argmax,
                    "case {} pos {t}: greedy token diverged from HF",
                    case.id
                );
            }
            let last_row = &trace.logits[(s - 1) * vocab..][..vocab];
            worst = worst.max(assert_slice_close(
                &last_row[..8],
                &case.logits.last_tok_first8,
                &format!("case {} logits last_tok_first8", case.id),
            ));
            for &(id, val) in &case.logits.last_tok_top5 {
                assert!(
                    close(last_row[id], val),
                    "case {}: top-5 logit id {id}: lattice {} vs HF {val}",
                    case.id,
                    last_row[id]
                );
            }
            let mean_abs = last_row.iter().map(|v| v.abs()).sum::<f32>() / vocab as f32;
            assert!(
                close(mean_abs, case.logits.last_tok_mean_abs),
                "case {}: logits mean_abs {mean_abs} vs HF {}",
                case.id,
                case.logits.last_tok_mean_abs
            );
            println!(
                "case {}: S={s} all checkpoints within tolerance, worst |diff| {worst:.2e}",
                case.id
            );
        }
    }
}
