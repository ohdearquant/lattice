//! Reverso against reference outputs from the upstream PyTorch implementation.
//!
//! `LATTICE_REVERSO_GOLDEN_DIR` names a directory holding `prep.safetensors` and, per model
//! (`reverso`, `reverso_small`, `reverso_nano`), `args.json`, `model.safetensors` and
//! `goldens.safetensors`. Without it the test prints SKIP and passes.

use std::path::{Path, PathBuf};

use lattice_inference::forecast::{PredictOptions, Reverso, ReversoConfig, prep};
use lattice_inference::weights::SafetensorsFile;

const OUTPUT_TOL: f32 = 1e-3;
const LAYER_TOL: f32 = 1e-2;

fn golden_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LATTICE_REVERSO_GOLDEN_DIR")?;
    Some(PathBuf::from(dir))
}

fn tensor(file: &SafetensorsFile, name: &str) -> (Vec<f32>, Vec<usize>) {
    let (data, shape) = file
        .get_f32_tensor(name)
        .unwrap_or_else(|e| panic!("golden tensor {name}: {e}"));
    (data.to_vec(), shape.to_vec())
}

/// Reads an f32 tensor straight from the file layout. The library reader refuses non-finite
/// values, and the context-preparation inputs carry `NaN` on purpose.
fn raw_f32(path: &Path, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let meta = &header[name];
    assert_eq!(meta["dtype"], "F32", "{name} dtype");
    let start = 8 + header_len + meta["data_offsets"][0].as_u64().unwrap() as usize;
    let end = 8 + header_len + meta["data_offsets"][1].as_u64().unwrap() as usize;
    bytes[start..end]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// max |a - b| / (max |b| + 1)
fn scaled_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale = b.iter().map(|y| y.abs()).fold(0.0f32, f32::max) + 1.0;
    worst / scale
}

fn check_model(dir: &Path, name: &str) {
    let model_dir = dir.join(name);
    let model = Reverso::load(&model_dir).unwrap_or_else(|e| panic!("{name}: load: {e}"));
    let goldens = SafetensorsFile::open(&model_dir.join("goldens.safetensors")).unwrap();
    let cfg = model.config().clone();
    let (l, d, ol) = (cfg.seq_len, cfg.d_model, cfg.output_token_len);

    let (inputs, shape) = tensor(&goldens, "inputs");
    assert_eq!(shape, vec![4, l]);
    let series: Vec<&[f32]> = inputs.chunks_exact(l).collect();

    let (_, layers) = model.forward_with_layer_outputs(series[0]).unwrap();
    for (j, got) in layers.iter().enumerate() {
        let (want, shape) = tensor(&goldens, &format!("layer{j:02}"));
        assert_eq!(shape, vec![d, l], "{name} layer{j:02} shape");
        let mut worst = 0.0f32;
        let mut scale = 0.0f32;
        for c in 0..d {
            for t in 0..l {
                let w = want[c * l + t];
                worst = worst.max((got[t * d + c] - w).abs());
                scale = scale.max(w.abs());
            }
        }
        let err = worst / scale.max(1e-6);
        println!("{name} layer{j:02} max_abs={worst:.3e} scaled={err:.3e}");
        assert!(
            err <= LAYER_TOL,
            "{name} layer{j:02}: scaled error {err:.3e}"
        );
    }

    let (one_call, _) = tensor(&goldens, "one_call");
    for (i, s) in series.iter().enumerate() {
        let got = model.forward(s).unwrap();
        let err = scaled_error(&got, &one_call[i * ol..(i + 1) * ol]);
        println!("{name} one_call[{i}] scaled={err:.3e}");
        assert!(err <= OUTPUT_TOL, "{name} one_call[{i}]: {err:.3e}");
    }

    let (rollout, _) = tensor(&goldens, "rollout200");
    let (rollout_neg, _) = tensor(&goldens, "rollout200_neg");
    let (flip, _) = tensor(&goldens, "flip200");
    for (i, s) in series.iter().enumerate() {
        let range = i * 200..(i + 1) * 200;
        let got = model.rollout(s, 200).unwrap();
        let err = scaled_error(&got, &rollout[range.clone()]);
        let negated: Vec<f32> = s.iter().map(|v| -v).collect();
        let got_neg = model.rollout(&negated, 200).unwrap();
        let err_neg = scaled_error(&got_neg, &rollout_neg[range.clone()]);
        let options = PredictOptions {
            downsample: 1,
            flip_invariance: true,
        };
        let got_flip = model.predict(s, 200, options).unwrap();
        let err_flip = scaled_error(&got_flip, &flip[range]);
        println!("{name} series {i}: rollout={err:.3e} negated={err_neg:.3e} flip={err_flip:.3e}");
        assert!(err <= OUTPUT_TOL, "{name} rollout[{i}]: {err:.3e}");
        assert!(
            err_neg <= OUTPUT_TOL,
            "{name} rollout_neg[{i}]: {err_neg:.3e}"
        );
        assert!(err_flip <= OUTPUT_TOL, "{name} flip[{i}]: {err_flip:.3e}");
    }
}

#[test]
fn reverso_models_match_the_reference() {
    let Some(dir) = golden_dir() else {
        println!("SKIP: LATTICE_REVERSO_GOLDEN_DIR is not set");
        return;
    };
    for name in ["reverso_nano", "reverso_small", "reverso"] {
        check_model(&dir, name);
    }
}

#[test]
fn context_preparation_matches_the_reference_bit_for_bit() {
    let Some(dir) = golden_dir() else {
        println!("SKIP: LATTICE_REVERSO_GOLDEN_DIR is not set");
        return;
    };
    let path = dir.join("prep.safetensors");
    for case in ["short300", "nan_mid", "nan_lead", "one_valid", "all_nan"] {
        let raw = raw_f32(&path, &format!("{case}.raw"));
        let want = raw_f32(&path, &format!("{case}.prepared"));
        if case != "short300" {
            assert!(
                raw.iter().any(|v| v.is_nan()),
                "control: {case} carries missing values"
            );
        }
        let mut got = prep::left_pad(&raw, 2048);
        prep::impute_missing(&mut got);
        let mismatches = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("prep {case}: {mismatches} of {} values differ", want.len());
        assert_eq!(got.len(), want.len(), "{case} length");
        assert_eq!(mismatches, 0, "{case}");
    }
}

/// The full checkpoint carries an output position table and its query projection. A loader
/// configured without them must refuse the leftover tensors instead of ignoring them.
#[test]
fn a_checkpoint_with_unused_tensors_is_refused() {
    let Some(dir) = golden_dir() else {
        println!("SKIP: LATTICE_REVERSO_GOLDEN_DIR is not set");
        return;
    };
    let model_dir = dir.join("reverso");
    let args = std::fs::read_to_string(model_dir.join("args.json")).unwrap();
    let config = ReversoConfig::from_args_json(&args).unwrap();
    assert!(
        config.use_output_pe,
        "control: the full model uses output positions"
    );
    let stripped = ReversoConfig {
        use_output_pe: false,
        ..config
    };
    let file = SafetensorsFile::open(&model_dir.join("model.safetensors")).unwrap();
    let err = Reverso::from_safetensors(stripped, &file)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("post_pe_q_proj.weight"),
        "unexpected error: {err}"
    );
}
