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
        PaddleOcrVisionConfig, PaddleOcrVisionWeights, paddleocr_vision_forward_trace,
    };
    use lattice_inference::weights::SafetensorsFile;
    use serde::Deserialize;
    use std::path::PathBuf;

    /// Both sides load bit-identical bf16 weights and compute in f32, so the
    /// divergence is accumulation-order noise compounded across 27 blocks plus
    /// the projector's 4608-wide reductions. Measured worst error on the
    /// compared fields across the three fixture grids was 2.6e-4 (on the
    /// 96-patch case's projector rows); the bounds below leave ~4x headroom.
    /// Two structural mutations were run against this gate and each reddened
    /// it on the first compared checkpoint (bilinear position-embedding
    /// interpolation with the align-corners convention instead of the
    /// half-pixel one; swapped row/column RoPE axes). A third, tanh-approximate instead of exact-erf GELU in the
    /// projector, moves the compared rows by at most 6.8e-4 and is NOT
    /// discriminated by these bounds: the fixture summarises projector rows by
    /// their first eight values and per-row mean |x|, and the two GELU
    /// variants differ by O(1e-3) only near |x| ~ 2 before a 4608-wide linear
    /// averages it down.
    const ATOL: f32 = 1e-3;
    const RTOL: f32 = 1e-3;

    #[derive(Deserialize)]
    struct Golden {
        revision: String,
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        id: String,
        grid_h: usize,
        grid_w: usize,
        checkpoints: Vec<Checkpoint>,
        projector: ProjectorGolden,
    }

    #[derive(Deserialize)]
    struct Checkpoint {
        name: String,
        last_tok_first8: Vec<f32>,
        first_tok_first8: Vec<f32>,
        mean_abs: f32,
    }

    #[derive(Deserialize)]
    struct ProjectorGolden {
        rows: usize,
        first_row_first8: Vec<f32>,
        last_row_first8: Vec<f32>,
        mean_abs: f32,
        row_mean_abs: Vec<f32>,
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
        assert!(
            golden.cases.len() >= 3,
            "goldens shrank to {} cases",
            golden.cases.len()
        );

        let cfg = PaddleOcrVisionConfig::from_config_json(&dir.join("config.json"))
            .expect("config loads");
        let mut source =
            SafetensorsFile::open(&dir.join("model.safetensors")).expect("open weights");
        let weights = PaddleOcrVisionWeights::load(&mut source, &cfg).expect("weights load");

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
            println!(
                "case {}: grid {gh}x{gw} ({n} patches) all checkpoints within tolerance, \
                 worst |diff| {worst:.2e}",
                case.id
            );
        }
    }
}
