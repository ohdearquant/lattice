//! Toto 2.0 against reference outputs from the upstream PyTorch implementation.
//!
//! `LATTICE_TOTO2_GOLDEN_DIR` names a directory holding `m22/` and `m4/` (each the published
//! `config.json` and `model.safetensors`) beside `goldens-22m.{safetensors,json}` and
//! `goldens-4m.{safetensors,json}`. Without it the tests print SKIP and pass.

use std::path::{Path, PathBuf};

use lattice_inference::forecast::{Toto2, Toto2Options, Toto2Variate};

const OUTPUT_TOL: f32 = 1e-3;
const LAYER_TOL: f32 = 1e-3;
const BATCH_TOL: f32 = 1e-5;

fn golden_dir() -> Option<PathBuf> {
    Some(PathBuf::from(std::env::var_os("LATTICE_TOTO2_GOLDEN_DIR")?))
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

fn variates(goldens: &Path, case: &str, ids: &[u64]) -> Vec<Toto2Variate> {
    let (raw, shape) = raw_f32(goldens, &format!("{case}.raw"));
    let len = shape[1];
    (0..shape[0])
        .map(|v| Toto2Variate {
            context: raw[v * len..(v + 1) * len].to_vec(),
            series_id: ids[v],
        })
        .collect()
}

/// Reference quantiles `[q, variate, step]` regrouped per variate as `[q, step]`.
fn per_variate(reference: &[f32], nq: usize, n_var: usize, horizon: usize) -> Vec<Vec<f32>> {
    (0..n_var)
        .map(|v| {
            (0..nq)
                .flat_map(|q| {
                    let at = (q * n_var + v) * horizon;
                    reference[at..at + horizon].to_vec()
                })
                .collect()
        })
        .collect()
}

fn check_size(size: &str, expected_cases: usize) {
    let Some(dir) = golden_dir() else {
        println!("SKIP toto2 {size}: LATTICE_TOTO2_GOLDEN_DIR is not set");
        return;
    };
    let model_dir = dir.join(if size == "22m" { "m22" } else { "m4" });
    let model = Toto2::load(&model_dir).unwrap_or_else(|e| panic!("load {size}: {e}"));
    let goldens = dir.join(format!("goldens-{size}.safetensors"));
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join(format!("goldens-{size}.json"))).unwrap(),
    )
    .unwrap();
    let cases = meta["cases"].as_object().unwrap();
    assert_eq!(
        cases.len(),
        expected_cases,
        "goldens-{size}.json case count"
    );
    let nq = model.quantile_levels().len();
    let mut compared = 0;

    for (case, info) in cases {
        let block = info["decode_block_size"].as_u64().map(|b| b as usize);
        let ids: Vec<u64> = info["series_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g.as_u64().unwrap())
            .collect();
        let horizon = info["horizon"].as_u64().unwrap() as usize;
        let batch = variates(&goldens, case, &ids);
        let (want, want_shape) = raw_f32(&goldens, &format!("{case}.quantiles"));
        assert_eq!(want_shape, vec![nq, batch.len(), horizon], "{case} shape");
        let want = per_variate(&want, nq, batch.len(), horizon);

        let (got, trace) = match block {
            // Block decoding: three passes for case k (24, 24 and 2 patches), the later two
            // reading the context through the key/value cache.
            Some(decode_block_size) => {
                let options = Toto2Options {
                    decode_block_size,
                    ..Toto2Options::default()
                };
                let got = model
                    .forecast_with_options(&batch, horizon, options)
                    .unwrap_or_else(|e| panic!("{size} {case}: {e}"));
                (got, Default::default())
            }
            None => model
                .forecast_with_trace(&batch, horizon)
                .unwrap_or_else(|e| panic!("{size} {case}: {e}")),
        };
        for (v, (g, w)) in got.iter().zip(&want).enumerate() {
            let err = scaled_error(&g.values, w);
            println!("{size} {case} variate {v} quantiles scaled_error {err:.3e}");
            assert!(
                err <= OUTPUT_TOL,
                "{size} {case} variate {v}: scaled error {err:.3e}"
            );
        }
        compared += 1;

        if size == "22m" && case == "a" {
            let mut layers = vec![
                ("loc".to_string(), trace.loc.clone()),
                ("scale".to_string(), trace.scale.clone()),
                ("embed".to_string(), trace.embeddings.clone()),
            ];
            for (i, l) in trace.layers.iter().enumerate() {
                layers.push((format!("layer{i:02}"), l.clone()));
            }
            layers.push(("final".to_string(), trace.final_state.clone()));
            assert_eq!(layers.len(), 4 + model.config().num_layers, "layer count");
            for (name, ours) in layers {
                let (reference, _) = raw_f32(&goldens, &format!("a.{name}"));
                let err = relative_error(&ours, &reference);
                println!("{size} a {name} relative_error {err:.3e}");
                assert!(
                    err <= LAYER_TOL,
                    "{size} a {name}: relative error {err:.3e}"
                );
            }

            // A block that covers the horizon is the single pass, bit for bit.
            let patch = model.config().patch_size;
            let covering = Toto2Options {
                decode_block_size: horizon.div_ceil(patch) * patch,
                ..Toto2Options::default()
            };
            let same = model
                .forecast_with_options(&batch, horizon, covering)
                .unwrap();
            assert_eq!(
                same[0].values, got[0].values,
                "covering block vs single pass"
            );
            let uneven = Toto2Options {
                decode_block_size: patch + 1,
                ..Toto2Options::default()
            };
            let err = model
                .forecast_with_options(&batch, horizon, uneven)
                .unwrap_err()
                .to_string();
            assert!(err.contains("multiple of the patch size"), "{err}");
        }

        if size == "22m" && case == "g" {
            // The third variate has its own series id, so it must forecast the same alone.
            let alone = model
                .forecast(std::slice::from_ref(&batch[2]), horizon)
                .unwrap()
                .remove(0);
            let err = scaled_error(&alone.values, &got[2].values);
            println!("{size} g variate 2 alone-vs-joint scaled_error {err:.3e}");
            assert!(err <= BATCH_TOL, "g variate 2 alone vs joint {err:.3e}");
        }
    }
    assert!(compared > 0, "{size}: no case compared");
}

#[test]
fn toto2_22m_matches_the_reference() {
    check_size("22m", 9);
}

#[test]
fn toto2_4m_matches_the_reference() {
    check_size("4m", 2);
}

#[test]
fn a_loader_refuses_tensors_the_configuration_does_not_use() {
    let Some(dir) = golden_dir() else {
        println!("SKIP a_loader_refuses_tensors_the_configuration_does_not_use");
        return;
    };
    let text = std::fs::read_to_string(dir.join("m22/config.json")).unwrap();
    let fewer = text
        .replacen("\"num_layers\": 6", "\"num_layers\": 3", 1)
        .replacen("\"layer_group_size\": 6", "\"layer_group_size\": 3", 1);
    assert_ne!(fewer, text, "fixture edit did not apply");
    let config = lattice_inference::forecast::Toto2Config::from_config_json(&fewer).unwrap();
    let file =
        lattice_inference::weights::SafetensorsFile::open(&dir.join("m22/model.safetensors"))
            .unwrap();
    let err = Toto2::from_safetensors(config, &file)
        .unwrap_err()
        .to_string();
    assert!(err.contains("transformer.layers.3."), "{err}");
}
