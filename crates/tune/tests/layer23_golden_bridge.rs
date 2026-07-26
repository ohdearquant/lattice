//! Real-checkpoint equivalence gate for the layer-23 compatibility command.

use std::path::{Path, PathBuf};
use std::process::Command;

use lattice_tune::lora::LoraAdapter;
use lattice_tune::train_support as train_common;

// The removed hand-inlined reverse pass and the shared tape accumulate f32
// gradients in a different order; 1e-5 covers that rounding without accepting
// the 1e-4 mutation exercised below.
const TENSOR_ABS_TOLERANCE: f32 = 1e-5;
const NLL_ABS_TOLERANCE: f32 = 1e-4;

#[derive(Clone)]
struct AdapterTensor {
    name: String,
    shape: (usize, usize, usize),
    a: Vec<f32>,
    b: Vec<f32>,
}

fn tensors(adapter: &LoraAdapter) -> Vec<AdapterTensor> {
    let mut tensors: Vec<_> = adapter
        .layers()
        .iter()
        .map(|((layer, module), value)| AdapterTensor {
            name: format!("layer.{layer}.{module}"),
            shape: (value.d_in, value.d_out, value.rank),
            a: value.a.clone(),
            b: value.b.clone(),
        })
        .collect();
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    tensors
}

fn compare_values(name: &str, left: &[f32], right: &[f32], tolerance: f32) -> Result<(), String> {
    if left.len() != right.len() {
        return Err(format!(
            "{name} length differs: {} != {}",
            left.len(),
            right.len()
        ));
    }
    for (index, (&left_value, &right_value)) in left.iter().zip(right).enumerate() {
        let diff = (left_value - right_value).abs();
        if !diff.is_finite() || diff > tolerance {
            return Err(format!(
                "{name}[{index}] differs by {diff:e}, tolerance {tolerance:e}"
            ));
        }
    }
    Ok(())
}

fn compare_tensors(
    left: &[AdapterTensor],
    right: &[AdapterTensor],
    tolerance: f32,
) -> Result<(), String> {
    if left.len() != right.len() {
        return Err(format!(
            "tensor count differs: {} != {}",
            left.len(),
            right.len()
        ));
    }
    for (left_tensor, right_tensor) in left.iter().zip(right) {
        if left_tensor.name != right_tensor.name {
            return Err(format!(
                "tensor name differs: {} != {}",
                left_tensor.name, right_tensor.name
            ));
        }
        if left_tensor.shape != right_tensor.shape {
            return Err(format!(
                "{} shape differs: {:?} != {:?}",
                left_tensor.name, left_tensor.shape, right_tensor.shape
            ));
        }
        compare_values(
            &format!("{}.A", left_tensor.name),
            &left_tensor.a,
            &right_tensor.a,
            tolerance,
        )?;
        compare_values(
            &format!("{}.B", left_tensor.name),
            &left_tensor.b,
            &right_tensor.b,
            tolerance,
        )?;
    }
    Ok(())
}

fn metadata_value<'a>(metadata: &'a str, key: &str) -> &'a str {
    metadata
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("legacy metadata is missing {key}"))
}

fn stdout_value(stdout: &str, line_marker: &str, value_marker: &str) -> f32 {
    let line = stdout
        .lines()
        .find(|line| line.contains(line_marker))
        .unwrap_or_else(|| panic!("driver output is missing {line_marker:?}"));
    let value = line
        .split_once(value_marker)
        .unwrap_or_else(|| panic!("driver output line is missing {value_marker:?}: {line}"))
        .1
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("driver output has no value after {value_marker:?}: {line}"));
    value
        .parse()
        .unwrap_or_else(|_| panic!("driver output value is not f32: {value}"))
}

fn assert_close(name: &str, left: f32, right: f32, tolerance: f32) {
    let diff = (left - right).abs();
    assert!(
        diff.is_finite() && diff <= tolerance,
        "{name} differs by {diff:e}: {left} != {right}, tolerance {tolerance:e}"
    );
}

fn assert_roundtrip(adapter: &LoraAdapter, path: &Path) {
    adapter.save_safetensors(path, None).unwrap();
    let reloaded = LoraAdapter::from_safetensors(path).unwrap();
    compare_tensors(&tensors(adapter), &tensors(&reloaded), 0.0).unwrap();
}

#[test]
fn compatibility_shim_matches_legacy_layer23_golden() {
    if std::env::var_os("LATTICE_LAYER23_GOLDEN_GATE_ENFORCE").is_none() {
        eprintln!(
            "layer-23 golden bridge skipped; set LATTICE_LAYER23_GOLDEN_GATE_ENFORCE=1 with LATTICE_MODEL_DIR"
        );
        return;
    }

    let model_dir = PathBuf::from(
        std::env::var_os("LATTICE_MODEL_DIR")
            .expect("LATTICE_MODEL_DIR is required when the layer-23 gate is enforced"),
    );
    assert!(
        model_dir.join("config.json").is_file(),
        "enforced layer-23 gate requires a provisioned model at {}",
        model_dir.display()
    );

    let fixture_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/layer23_golden_v1");
    let temp = tempfile::tempdir().unwrap();
    let actual_path = temp.path().join("actual.safetensors");
    let output = Command::new(env!("CARGO_BIN_EXE_train_grad_layer23"))
        .args([
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--data-dir",
            fixture_dir.to_str().unwrap(),
            "--steps",
            "1",
            "--lr",
            "1e-3",
            "--rank",
            "1",
            "--alpha",
            "2",
            "--seq-len",
            "32",
            "--max-train",
            "1",
            "--log-every",
            "1",
            "--save",
            actual_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "compatibility driver failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("train_grad_full --first-layer 23"));

    let metadata = std::fs::read_to_string(fixture_dir.join("legacy_run.txt")).unwrap();
    let legacy_tbv_model: f32 = metadata_value(&metadata, "tbv_model_nll").parse().unwrap();
    let legacy_tbv_chain: f32 = metadata_value(&metadata, "tbv_chain_nll").parse().unwrap();
    let legacy_tbv_diff: f32 = metadata_value(&metadata, "tbv_diff").parse().unwrap();
    let legacy_step0: f32 = metadata_value(&metadata, "step0_nll").parse().unwrap();
    let legacy_final: f32 = metadata_value(&metadata, "final_nll").parse().unwrap();

    let actual_tbv_model = stdout_value(&stdout, "TBV (sample 0)", "model=");
    let actual_tbv_chain = stdout_value(&stdout, "TBV (sample 0)", "chain=");
    let actual_tbv_diff = stdout_value(&stdout, "TBV (sample 0)", "diff=");
    let actual_step0 = stdout_value(&stdout, "step    0", "train NLL:");
    let actual_final = stdout_value(&stdout, "step    1", "train NLL:");

    assert!(legacy_tbv_diff <= 1e-2);
    assert!(actual_tbv_diff <= 1e-2);
    assert_close(
        "TBV model NLL",
        legacy_tbv_model,
        actual_tbv_model,
        NLL_ABS_TOLERANCE,
    );
    assert_close(
        "TBV chain NLL",
        legacy_tbv_chain,
        actual_tbv_chain,
        NLL_ABS_TOLERANCE,
    );
    assert_close("step-0 NLL", legacy_step0, actual_step0, NLL_ABS_TOLERANCE);
    assert_close("final NLL", legacy_final, actual_final, NLL_ABS_TOLERANCE);

    let legacy =
        LoraAdapter::from_safetensors(&fixture_dir.join("legacy_adapter.safetensors")).unwrap();
    let actual = LoraAdapter::from_safetensors(&actual_path).unwrap();
    let legacy_tensors = tensors(&legacy);
    let actual_tensors = tensors(&actual);
    compare_tensors(&legacy_tensors, &actual_tensors, TENSOR_ABS_TOLERANCE).unwrap();

    let mut mutated = actual_tensors.clone();
    mutated[0].a[0] += TENSOR_ABS_TOLERANCE * 10.0;
    assert!(
        compare_tensors(&legacy_tensors, &mutated, TENSOR_ABS_TOLERANCE).is_err(),
        "the bridge must reject a deliberate adapter-array mutation"
    );

    assert_roundtrip(&legacy, &temp.path().join("legacy-roundtrip.safetensors"));
    assert_roundtrip(&actual, &temp.path().join("actual-roundtrip.safetensors"));

    let gradcheck = train_common::full_driver::run(train_common::full_driver::FullDriverConfig {
        model_dir,
        data_dir: fixture_dir.clone(),
        first_layer: 23,
        steps: 1,
        lr: 1e-3,
        rank: 1,
        alpha: 2.0,
        seq_len_cap: 32,
        max_train: 1,
        max_valid: 0,
        log_every: 1,
        gradcheck: true,
        gradcheck_strided_probes: false,
        probe: 1,
        fd_eps: 4e-3,
        save_path: None,
        a_init_amp: Some(0.02),
    })
    .unwrap();
    let worst_relative_error = gradcheck
        .gradcheck_max_rel
        .expect("gradcheck mode must report its worst relative error");
    assert!(
        worst_relative_error < 1e-2,
        "layer-23 central-difference max_rel {worst_relative_error:e} must be < 1e-2"
    );

    let report = format!(
        "layer-23 golden bridge PASS\nTBV legacy={legacy_tbv_chain:.5} shared={actual_tbv_chain:.5}\nstep0 legacy={legacy_step0:.4} shared={actual_step0:.4}\nfinal legacy={legacy_final:.4} shared={actual_final:.4}\ntensor_abs_tolerance={TENSOR_ABS_TOLERANCE:e}\nnll_abs_tolerance={NLL_ABS_TOLERANCE:e}\ncentral_difference_max_rel={worst_relative_error:e}\nmutation_check=PASS\nroundtrip_check=PASS\n"
    );
    print!("{report}");
    if let Some(path) = std::env::var_os("LATTICE_LAYER23_GOLDEN_REPORT") {
        std::fs::write(path, report).unwrap();
    }
}
