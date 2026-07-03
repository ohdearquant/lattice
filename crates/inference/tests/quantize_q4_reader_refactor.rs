//! Byte-identity regression gate for the `quantize_q4` binary (issue #23).
//!
//! `quantize_q4` moved off an inline SafeTensors parser onto
//! `QuarotTensorReader`. The `.q4` output format and the Q4 quantization math
//! did not change, so the binary's output for a fixed input must stay
//! byte-for-byte identical to what it produced before that refactor.
//!
//! The golden `.q4` files under `tests/fixtures/quantize_q4_reader_refactor/`
//! were captured by running the pre-refactor binary (inline parser) on the
//! fixture built by `write_sharded_fixture`/`write_single_file_fixture` below.
//! The refactor no longer has that old binary to compare against at test
//! time, so the committed golden bytes are the behavior contract instead of
//! a within-test before/after.

use std::io::Write;
use std::path::Path;
use std::process::Command;

const GATE_PROJ_GOLDEN: &[u8] =
    include_bytes!("fixtures/quantize_q4_reader_refactor/gate_proj_golden.q4");
const K_PROJ_GOLDEN: &[u8] =
    include_bytes!("fixtures/quantize_q4_reader_refactor/k_proj_golden.q4");

/// BF16 bit pattern for an arbitrary finite `f32`: truncate to the top 16
/// bits, matching `QuarotTensorReader`'s BF16 decode (`f32::from_bits((bits
/// as u32) << 16)`).
fn f32_to_bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// Deterministic, varied, finite fill values — matches the values used to
/// generate the committed golden fixture.
fn gen_values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 13) % 200) as i64 - 100) as f32 / 4.0)
        .collect()
}

fn write_safetensors(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut payload: Vec<u8> = Vec::new();
    for (name, shape, values) in tensors {
        let start = payload.len();
        for &v in values {
            payload.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes());
        }
        let end = payload.len();

        let mut entry = serde_json::Map::new();
        entry.insert("dtype".into(), serde_json::Value::String("BF16".into()));
        entry.insert(
            "shape".into(),
            serde_json::Value::Array(shape.iter().map(|d| serde_json::Value::from(*d)).collect()),
        );
        entry.insert(
            "data_offsets".into(),
            serde_json::Value::Array(vec![
                serde_json::Value::from(start),
                serde_json::Value::from(end),
            ]),
        );
        header.insert((*name).to_string(), serde_json::Value::Object(entry));
    }
    let header_bytes = serde_json::to_string(&serde_json::Value::Object(header))
        .expect("header serializes")
        .into_bytes();
    let mut file = std::fs::File::create(path).expect("create safetensors file");
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .expect("write header length");
    file.write_all(&header_bytes).expect("write header");
    file.write_all(&payload).expect("write payload");
}

/// Same three tensors used to generate the committed golden: one Q4 tensor
/// spanning two full blocks plus a partial tail, one exactly-one-block Q4
/// tensor, and one kept `.f16` norm tensor.
fn fixture_tensors() -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    vec![
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![2, 33],
            gen_values(66, 1),
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![1, 32],
            gen_values(32, 2),
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![4],
            gen_values(4, 3),
        ),
    ]
}

fn write_sharded_fixture(model_dir: &Path) {
    std::fs::create_dir_all(model_dir).expect("create model dir");
    let shard_name = "model-00001-of-00001.safetensors";
    write_safetensors(&model_dir.join(shard_name), &fixture_tensors());

    let weight_map: serde_json::Map<String, serde_json::Value> = fixture_tensors()
        .into_iter()
        .map(|(name, _, _)| {
            (
                name.to_string(),
                serde_json::Value::String(shard_name.into()),
            )
        })
        .collect();
    let index = serde_json::json!({
        "metadata": {},
        "weight_map": serde_json::Value::Object(weight_map),
    });
    std::fs::write(
        model_dir.join("model.safetensors.index.json"),
        serde_json::to_string(&index).expect("index serializes"),
    )
    .expect("write index");
}

fn write_single_file_fixture(model_dir: &Path) {
    std::fs::create_dir_all(model_dir).expect("create model dir");
    write_safetensors(&model_dir.join("model.safetensors"), &fixture_tensors());
}

fn run_quantize_q4(model_dir: &Path, output_dir: &Path) {
    let bin = env!("CARGO_BIN_EXE_quantize_q4");
    let status = Command::new(bin)
        .arg("--model-dir")
        .arg(model_dir)
        .arg("--output-dir")
        .arg(output_dir)
        .status()
        .expect("spawn quantize_q4");
    assert!(status.success(), "quantize_q4 exited with {status}");
}

fn assert_q4_matches_golden(output_dir: &Path) {
    let gate_proj =
        std::fs::read(output_dir.join("model_layers_0_mlp_gate_proj_weight.q4")).unwrap();
    let k_proj =
        std::fs::read(output_dir.join("model_layers_0_self_attn_k_proj_weight.q4")).unwrap();
    assert_eq!(
        gate_proj, GATE_PROJ_GOLDEN,
        "gate_proj .q4 bytes diverged from the pre-refactor golden"
    );
    assert_eq!(
        k_proj, K_PROJ_GOLDEN,
        "k_proj .q4 bytes diverged from the pre-refactor golden"
    );
}

#[test]
fn sharded_fixture_matches_pre_refactor_q4_golden() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_dir = dir.path().join("model");
    let output_dir = dir.path().join("out");
    write_sharded_fixture(&model_dir);

    run_quantize_q4(&model_dir, &output_dir);
    assert_q4_matches_golden(&output_dir);

    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(output_dir.join("quantize_index.json")).unwrap(),
    )
    .expect("index parses");
    let entries = index.as_array().expect("index is an array");
    assert_eq!(entries.len(), 3);
    let names: Vec<&str> = entries
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"model.layers.0.mlp.gate_proj.weight"));
    assert!(names.contains(&"model.layers.0.self_attn.k_proj.weight"));
    assert!(names.contains(&"model.layers.0.input_layernorm.weight"));
    let quantized_flags: Vec<bool> = entries
        .iter()
        .map(|e| e["quantized"].as_bool().unwrap())
        .collect();
    assert_eq!(quantized_flags.iter().filter(|&&q| q).count(), 2);
}

#[test]
fn single_file_fixture_matches_same_q4_golden() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model_dir = dir.path().join("model");
    let output_dir = dir.path().join("out");
    write_single_file_fixture(&model_dir);

    run_quantize_q4(&model_dir, &output_dir);
    assert_q4_matches_golden(&output_dir);
}
