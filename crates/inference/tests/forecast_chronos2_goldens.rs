//! Chronos-2 against reference outputs from the upstream PyTorch implementation.
//!
//! `LATTICE_CHRONOS2_GOLDEN_DIR` names a directory holding the published `config.json` and
//! `model.safetensors` beside `goldens.safetensors` and `goldens.json`. Without it the test prints
//! SKIP and passes.

use std::path::{Path, PathBuf};

use lattice_inference::forecast::{Chronos2, Chronos2Series};

const OUTPUT_TOL: f32 = 1e-3;
const LAYER_TOL: f32 = 1e-3;
const BATCH_TOL: f32 = 1e-5;

fn golden_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LATTICE_CHRONOS2_GOLDEN_DIR")?;
    Some(PathBuf::from(dir))
}

/// Reads an f32 tensor and its shape straight from the file layout. The library reader refuses
/// non-finite values, and the contexts carry `NaN` on purpose.
fn raw_f32(path: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    let bytes = std::fs::read(path).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let meta = &header[name];
    assert_eq!(meta["dtype"], "F32", "{name} dtype");
    let shape = meta["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap() as usize)
        .collect();
    let start = 8 + header_len + meta["data_offsets"][0].as_u64().unwrap() as usize;
    let end = 8 + header_len + meta["data_offsets"][1].as_u64().unwrap() as usize;
    let data = bytes[start..end]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    (data, shape)
}

/// max |a - b| / (max |b| + 1)
fn scaled_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    worst / (b.iter().map(|y| y.abs()).fold(0.0f32, f32::max) + 1.0)
}

/// max |a - b| / max |b|
fn relative_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    worst / b.iter().map(|y| y.abs()).fold(0.0f32, f32::max)
}

fn has_tensor(path: &Path, name: &str) -> bool {
    let bytes = std::fs::read(path).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    header.get(name).is_some()
}

fn batch_of(goldens: &Path, case: &str, groups: &[u64]) -> Vec<Chronos2Series> {
    let (ctx, shape) = raw_f32(goldens, &format!("{case}.context"));
    let len = shape[1];
    let future_name = format!("{case}.future_covariates");
    let future = has_tensor(goldens, &future_name).then(|| raw_f32(goldens, &future_name));
    (0..shape[0])
        .map(|b| Chronos2Series {
            context: ctx[b * len..(b + 1) * len].to_vec(),
            group_id: groups[b],
            future_covariates: future
                .as_ref()
                .map(|(f, fs)| f[b * fs[1]..(b + 1) * fs[1]].to_vec()),
        })
        .collect()
}

#[test]
fn chronos2_matches_the_reference() {
    let Some(dir) = golden_dir() else {
        println!("SKIP chronos2_matches_the_reference: LATTICE_CHRONOS2_GOLDEN_DIR is not set");
        return;
    };
    let model = Chronos2::load(&dir).unwrap_or_else(|e| panic!("load: {e}"));
    let goldens = dir.join("goldens.safetensors");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("goldens.json")).unwrap()).unwrap();
    let cases = meta["cases"].as_object().unwrap();
    assert_eq!(cases.len(), 8, "goldens.json case count");
    let nq = model.config().quantiles.len();

    for (case, info) in cases {
        let groups: Vec<u64> = info["group_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g.as_u64().unwrap())
            .collect();
        let patches = info["patches"].as_u64().unwrap() as usize;
        let batch = batch_of(&goldens, case, &groups);
        let (want, want_shape) = raw_f32(&goldens, &format!("{case}.quantile_preds"));
        assert_eq!(
            want_shape,
            vec![batch.len(), nq, patches * 16],
            "{case} shape"
        );

        let (got, trace) = model
            .forward_with_trace(&batch, patches)
            .unwrap_or_else(|e| panic!("{case}: {e}"));
        let got: Vec<f32> = got.iter().flat_map(|f| f.values.clone()).collect();
        let err = scaled_error(&got, &want);
        println!("{case} quantile_preds scaled_error {err:.3e}");
        assert!(
            err <= OUTPUT_TOL,
            "{case}: quantile_preds scaled error {err:.3e}"
        );

        if case == "a" {
            let mut layers = vec![("embeds".to_string(), trace.embeddings.clone())];
            for (i, b) in trace.blocks.iter().enumerate() {
                layers.push((format!("block{i:02}"), b.clone()));
            }
            layers.push(("final".to_string(), trace.final_state.clone()));
            assert_eq!(layers.len(), 14, "layer count");
            for (name, ours) in layers {
                let (reference, _) = raw_f32(&goldens, &format!("a.{name}"));
                let err = relative_error(&ours, &reference);
                println!("a {name} relative_error {err:.3e}");
                assert!(err <= LAYER_TOL, "a {name}: relative error {err:.3e}");
            }
        }

        if case == "f" {
            for (i, series) in batch.iter().enumerate() {
                let single = model
                    .forward(std::slice::from_ref(series), patches)
                    .unwrap()
                    .remove(0);
                let err = scaled_error(
                    &single.values,
                    &got[i * nq * patches * 16..][..nq * patches * 16],
                );
                println!("f series {i} single-vs-batch scaled_error {err:.3e}");
                assert!(err <= BATCH_TOL, "f series {i}: single vs batch {err:.3e}");
            }
        }
    }
}

#[test]
fn a_loader_refuses_tensors_the_configuration_does_not_use() {
    let Some(dir) = golden_dir() else {
        println!("SKIP a_loader_refuses_tensors_the_configuration_does_not_use");
        return;
    };
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let fewer = text.replacen("\"num_layers\": 12", "\"num_layers\": 11", 1);
    assert_ne!(fewer, text, "fixture edit did not apply");
    let config = lattice_inference::forecast::Chronos2Config::from_config_json(&fewer).unwrap();
    let file =
        lattice_inference::weights::SafetensorsFile::open(&dir.join("model.safetensors")).unwrap();
    let err = Chronos2::from_safetensors(config, &file)
        .unwrap_err()
        .to_string();
    assert!(err.contains("encoder.block.11."), "{err}");
}
