//! PaddleOCR-VL vision encoder + projector forward vs the HF reference.
//!
//! The committed fixture (`fixtures/paddleocr_vl/vision/vision_goldens.json`)
//! holds per-checkpoint activation summaries captured from the pinned
//! checkpoint's own modeling source (revision noted in the fixture) running
//! under HF transformers on CPU in f32 (eager attention, `use_rope=True`,
//! `interpolate_pos_encoding=True`, `return_pooler_output=False`, the OCR
//! path's own call shape). The input is a formula-generated patch tensor
//! (no image codec, no resize) that this test regenerates bit-exactly, so
//! the gate isolates the encoder + projector algebra from preprocessing.
//! Regenerate with `scripts/gen_paddleocr_vision_goldens.py`; the adjacent
//! fixture manifest records its pinned reference inputs and runtime.
//!
//! **Fail-closed contract** (mirrors `paddleocr_vl_decoder_goldens_test.rs`):
//! the ~1.9 GB checkpoint is not committed. With `LATTICE_POCR_MODEL_DIR`
//! unset and the default `~/.lattice/models/paddleocr-vl-1.6` absent, this
//! test prints a skip line and returns. With `LATTICE_POCR_GATE_ENFORCE=1`, a
//! missing checkpoint panics instead of skipping.
//!
//! Run:
//! ```bash
//! cargo test --release -p lattice-inference --features f16 \
//!     --test paddleocr_vl_vision_goldens_test -- --nocapture
//! ```

#[test]
fn vision_forward_matches_hf_reference() {
    #[cfg(not(feature = "f16"))]
    {
        if std::env::var("LATTICE_POCR_GATE_ENFORCE").as_deref() == Ok("1") {
            panic!(
                "LATTICE_POCR_GATE_ENFORCE=1 but the `f16` feature is not enabled — the \
                 checkpoint's BF16 tensors require it"
            );
        }
        eprintln!("SKIP paddleocr_vl_vision_goldens_test: f16 feature disabled");
    }
    #[cfg(feature = "f16")]
    gate::run();
}

#[cfg(feature = "f16")]
mod gate {
    use lattice_inference::vision::paddleocr_vit::{
        PaddleOcrVisionConfig, PaddleOcrVisionWeights, paddleocr_vision_forward,
        paddleocr_vision_forward_trace,
    };
    use lattice_inference::weights::SafetensorsFile;
    use serde::Deserialize;
    use std::path::PathBuf;

    /// Retain the original summary bounds. A CPU f32 calibration measured
    /// 3.672e-5 worst HF disagreement over the 94 projector summary values.
    /// The 2.6e-4 quoted here previously is the 96-patch case's worst over
    /// the whole compared surface, and it is attained on an encoder
    /// checkpoint rather than on the projector: it is unchanged under the
    /// projector-only tanh-GELU substitution, which moves the other two
    /// cases to 6.83e-4 and 4.99e-4. That substitution's worst summary
    /// residual against HF is 6.828e-4 and passes these bounds.
    const ATOL: f32 = 1e-3;
    const RTOL: f32 = 1e-3;

    /// The first row's maximum scans all 1024 channels without averaging.
    /// Across the three calibration cases, the smallest GELU-swap signal
    /// (4.325e-4) exceeds the largest full-first-row HF disagreement
    /// (3.958e-5) by 10.9x. This common bound leaves at least 3.1x measured
    /// noise headroom and 3.2x mutant-residual separation. These are single
    /// CPU-host observations, not a bound on every accumulation order.
    const PROJECTOR_FIRST_ROW_MAX_ABS_ATOL: f32 = 1.25e-4;

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Golden {
        revision: String,
        dtype: String,
        pixel_formula: String,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Case {
        id: String,
        grid_h: usize,
        grid_w: usize,
        checkpoints: Vec<Checkpoint>,
        projector: ProjectorGolden,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Checkpoint {
        name: String,
        last_tok_first8: Vec<f32>,
        first_tok_first8: Vec<f32>,
        mean_abs: f32,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ProjectorGolden {
        rows: usize,
        first_row_first8: Vec<f32>,
        last_row_first8: Vec<f32>,
        mean_abs: f32,
        row_mean_abs: Vec<f32>,
        first_row_max_abs: f32,
    }

    fn close(a: f32, e: f32) -> bool {
        (a - e).abs() <= ATOL + RTOL * e.abs()
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

    fn mean_abs(x: &[f32]) -> f32 {
        x.iter().map(|v| v.abs()).sum::<f32>() / x.len() as f32
    }

    fn assert_projector_first_row_max_abs(results: &[(&str, f32, f32)]) {
        let mut failures = Vec::new();
        for &(id, actual, expected) in results {
            let error = (actual - expected).abs();
            println!(
                "case {id}: projector.first_row_max_abs lattice {actual} vs HF {expected}, \
                 |diff| {error:.8e}"
            );
            if !actual.is_finite()
                || !expected.is_finite()
                || error > PROJECTOR_FIRST_ROW_MAX_ABS_ATOL
            {
                failures.push(id);
            }
        }
        assert!(
            failures.is_empty(),
            "projector.first_row_max_abs exceeded absolute bound \
             {PROJECTOR_FIRST_ROW_MAX_ABS_ATOL} in cases {failures:?}"
        );
    }

    /// `pixel[i, c, py, px] = ((i*7 + c*13 + py*3 + px*5) % 17) / 8 - 1`, the
    /// generator's formula; every value is an exact f32.
    fn formula_patches(grid_h: usize, grid_w: usize, patch: usize, channels: usize) -> Vec<f32> {
        let n = grid_h * grid_w;
        let mut out = Vec::with_capacity(n * channels * patch * patch);
        for i in 0..n {
            for c in 0..channels {
                for py in 0..patch {
                    for px in 0..patch {
                        let k = (i * 7 + c * 13 + py * 3 + px * 5) % 17;
                        out.push(k as f32 / 8.0 - 1.0);
                    }
                }
            }
        }
        out
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
                "SKIP paddleocr_vl_vision_goldens_test: checkpoint not found (set \
                 LATTICE_POCR_MODEL_DIR to run)"
            );
            return;
        };

        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/paddleocr_vl/vision/vision_goldens.json");
        let golden: Golden =
            serde_json::from_str(&std::fs::read_to_string(&fixture).expect("read fixture"))
                .expect("valid vision_goldens.json");
        assert_eq!(
            golden.revision, "c5630abae1d940eafe0697512a0325494b02ab42",
            "fixture revision drifted from the pinned checkpoint"
        );
        assert_eq!(
            golden.dtype,
            "weights bf16 upcast to f32, eager attention, use_rope=True, interpolate_pos_encoding=True",
            "fixture dtype metadata does not match this test"
        );
        assert_eq!(
            golden.pixel_formula,
            "pixel[i,c,py,px] = ((i*7 + c*13 + py*3 + px*5) % 17) / 8 - 1; i = raster patch index",
            "fixture pixel formula metadata does not match this test"
        );
        assert_eq!(
            golden
                .cases
                .iter()
                .map(|case| (case.id.as_str(), case.grid_h, case.grid_w))
                .collect::<Vec<_>>(),
            [("g4x4", 4, 4), ("g6x10", 6, 10), ("g12x8", 12, 8)],
            "fixture cases differ from the calibrated grids"
        );

        let cfg = PaddleOcrVisionConfig::from_config_json(&dir.join("config.json"))
            .expect("config loads");
        let mut source =
            SafetensorsFile::open(&dir.join("model.safetensors")).expect("open weights");
        let weights = PaddleOcrVisionWeights::load(&mut source, &cfg).expect("weights load");
        assert_eq!(cfg.text_hidden_size, 1024, "projector channel count");
        let mut projector_maxima = Vec::new();

        for case in &golden.cases {
            let (gh, gw) = (case.grid_h, case.grid_w);
            let n = gh * gw;
            let h = cfg.hidden_size;
            let pixels = formula_patches(gh, gw, cfg.patch_size, cfg.num_channels);
            let trace =
                paddleocr_vision_forward_trace(&weights, &cfg, &pixels, gh, gw).expect("forward");

            assert_eq!(
                case.checkpoints.len(),
                cfg.num_hidden_layers + 2,
                "case {}: unexpected checkpoint count",
                case.id
            );
            let mut worst = 0f32;
            for (idx, ck) in case.checkpoints.iter().enumerate() {
                let (buf, expected_name): (&[f32], String) = match idx {
                    0 => (&trace.embed, "embed".to_string()),
                    i if i <= cfg.num_hidden_layers => {
                        (&trace.layer_outputs[i - 1], format!("layer_{}", i - 1))
                    }
                    _ => (&trace.post_layernorm, "post_layernorm".to_string()),
                };
                assert_eq!(ck.name, expected_name, "case {}: checkpoint order", case.id);
                assert_eq!(buf.len(), n * h, "case {} {}: length", case.id, ck.name);
                worst = worst.max(assert_slice_close(
                    &buf[..8],
                    &ck.first_tok_first8,
                    &format!("case {} {} first_tok_first8", case.id, ck.name),
                ));
                worst = worst.max(assert_slice_close(
                    &buf[(n - 1) * h..][..8],
                    &ck.last_tok_first8,
                    &format!("case {} {} last_tok_first8", case.id, ck.name),
                ));
                let ma = mean_abs(buf);
                assert!(
                    close(ma, ck.mean_abs),
                    "case {} {}: mean_abs {ma} vs HF {} — aggregate drift",
                    case.id,
                    ck.name,
                    ck.mean_abs
                );
            }

            let t = cfg.text_hidden_size;
            let rows = case.projector.rows;
            assert_eq!(rows, n / 4, "case {}: projector row count", case.id);
            assert_eq!(
                trace.projector.len(),
                rows * t,
                "case {}: projector length",
                case.id
            );
            worst = worst.max(assert_slice_close(
                &trace.projector[..8],
                &case.projector.first_row_first8,
                &format!("case {} projector first_row_first8", case.id),
            ));
            worst = worst.max(assert_slice_close(
                &trace.projector[(rows - 1) * t..][..8],
                &case.projector.last_row_first8,
                &format!("case {} projector last_row_first8", case.id),
            ));
            assert!(
                close(mean_abs(&trace.projector), case.projector.mean_abs),
                "case {}: projector mean_abs {} vs HF {}",
                case.id,
                mean_abs(&trace.projector),
                case.projector.mean_abs
            );
            assert_eq!(case.projector.row_mean_abs.len(), rows);
            for (r, &e) in case.projector.row_mean_abs.iter().enumerate() {
                let ma = mean_abs(&trace.projector[r * t..(r + 1) * t]);
                assert!(
                    close(ma, e),
                    "case {} projector row {r}: mean_abs {ma} vs HF {e} — merge order or \
                     per-block drift",
                    case.id
                );
            }
            let projector =
                paddleocr_vision_forward(&weights, &cfg, &pixels, gh, gw).expect("forward");
            assert_eq!(
                projector, trace.projector,
                "case {}: production output",
                case.id
            );
            assert!(
                trace.projector[..t].iter().all(|value| value.is_finite()),
                "case {}: nonfinite projector first row",
                case.id
            );
            projector_maxima.push((
                case.id.as_str(),
                trace.projector[..t]
                    .iter()
                    .map(|value| value.abs())
                    .fold(0f32, f32::max),
                case.projector.first_row_max_abs,
            ));
            println!(
                "case {}: grid {gh}x{gw} ({n} patches) all checkpoints within tolerance, \
                 worst |diff| {worst:.2e}",
                case.id
            );
        }
        // Complete every original assertion before the tighter projector check.
        println!("[POCR_VISION_LEGACY_GATE] executed=true cases=3");
        assert_projector_first_row_max_abs(&projector_maxima);
        println!("[POCR_PROJECTOR_GELU_GATE] executed=true cases=3");
    }
}
