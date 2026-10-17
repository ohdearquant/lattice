use super::*;
use std::fs;
use std::path::PathBuf;

fn write_header(path: &Path, tensors: &[(&str, &str)]) {
    let mut header = serde_json::Map::new();
    for (name, dtype) in tensors {
        header.insert(
            (*name).to_owned(),
            serde_json::json!({ "dtype": dtype, "shape": [1], "data_offsets": [0, 0] }),
        );
    }
    let json = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&json);
    fs::write(path, bytes).unwrap();
}

fn write_config(dir: &Path, config: Value) {
    fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
}

fn mlx_config() -> Value {
    serde_json::json!({
        "model_type": "qwen3_5",
        "quantization": { "group_size": 64, "bits": 4, "mode": "affine" },
        "quantization_config": { "group_size": 64, "bits": 4, "mode": "affine" }
    })
}

const AFFINE: [(&str, &str); 4] = [
    ("layer.q.weight", "U32"),
    ("layer.q.scales", "BF16"),
    ("layer.q.biases", "BF16"),
    ("norm.weight", "BF16"),
];

fn dir_with(tensors: &[(&str, &str)], config: Option<Value>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_header(&dir.path().join("model.safetensors"), tensors);
    if let Some(config) = config {
        write_config(dir.path(), config);
    }
    dir
}

#[track_caller]
fn assert_refused(dir: &Path, needle: &str) {
    let detection = inspect_format(dir);
    assert_eq!(detection.format(), ModelFormat::Refused, "{detection:?}");
    let reason = detection.refusal().unwrap_or_default();
    assert!(
        reason.contains(needle),
        "refusal {reason:?} lacks {needle:?}"
    );
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("mlx_headers")
}

/// Rebuild a captured repository as header-only files: its `config.json`,
/// its index when it had one, and one header-only safetensors file per
/// shard.
fn materialize(fixture: &str) -> tempfile::TempDir {
    let path = fixtures_dir().join(fixture);
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let record: Value = serde_json::from_str(&text).unwrap();
    let dir = tempfile::tempdir().unwrap();
    write_config(dir.path(), record["config"].clone());
    let shards = record["shards"].as_object().unwrap();
    let mut weight_map = serde_json::Map::new();
    for (shard, tensors) in shards {
        let tensors: Vec<(&str, &str)> = tensors
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, dtype)| (name.as_str(), dtype.as_str().unwrap()))
            .collect();
        write_header(&dir.path().join(shard), &tensors);
        for (name, _) in tensors {
            weight_map.insert(name.to_owned(), Value::String(shard.clone()));
        }
    }
    if record["index"].as_bool().unwrap() {
        fs::write(
            dir.path().join(SAFETENSORS_INDEX),
            serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
        )
        .unwrap();
    }
    dir
}

#[test]
fn captured_mlx_repositories_detect_as_mlx_quantized() {
    let fixtures = [
        "Qwen3.5-4B-MLX-4bit.json",
        "Qwen3.5-9B-MLX-4bit.json",
        "gemma-4-e4b-it-4bit.json",
        "Qwen3-Embedding-0.6B-4bit-DWQ.json",
    ];
    for fixture in fixtures {
        let dir = materialize(fixture);
        let detection = inspect_format(dir.path());
        assert_eq!(
            detection.format(),
            ModelFormat::MlxQuantized,
            "{fixture}: {detection:?}"
        );
    }
}

#[test]
fn captured_repository_without_its_declaration_is_refused_not_read_as_floating_point() {
    let dir = materialize("Qwen3.5-4B-MLX-4bit.json");
    write_config(dir.path(), serde_json::json!({ "model_type": "qwen3_5" }));
    assert_refused(dir.path(), "declares a quantization");
}

#[test]
fn sharded_capture_refuses_an_unlisted_shard() {
    let dir = materialize("Qwen3.5-9B-MLX-4bit.json");
    assert_eq!(detect_format(dir.path()), ModelFormat::MlxQuantized);
    write_header(&dir.path().join("extra.safetensors"), &[("x", "BF16")]);
    assert_refused(dir.path(), "does not list: extra.safetensors");
}

#[test]
fn affine_triad_without_a_declaration_refuses() {
    let dir = dir_with(
        &AFFINE,
        Some(serde_json::json!({ "model_type": "qwen3_5" })),
    );
    assert_refused(dir.path(), "declares a quantization");
    let no_config = dir_with(&AFFINE, None);
    assert_refused(no_config.path(), "declares a quantization");
}

#[test]
fn declared_affine_triad_is_mlx_quantized_at_top_level_and_in_text_config() {
    let dir = dir_with(&AFFINE, Some(mlx_config()));
    assert_eq!(detect_format(dir.path()), ModelFormat::MlxQuantized);
    let nested = dir_with(
        &AFFINE,
        Some(serde_json::json!({
            "text_config": { "quantization": { "group_size": 64, "bits": 4 } }
        })),
    );
    assert_eq!(detect_format(nested.path()), ModelFormat::MlxQuantized);
}

#[test]
fn partial_and_mixed_modules_refuse() {
    let config = Some(mlx_config());
    let orphan_scales = dir_with(
        &[
            ("layer.q.weight", "U32"),
            ("layer.q.scales", "BF16"),
            ("layer.q.biases", "BF16"),
            ("layer.k.scales", "BF16"),
        ],
        config.clone(),
    );
    assert_refused(orphan_scales.path(), "layer.k.scales");
    let no_biases = dir_with(
        &[("layer.q.weight", "U32"), ("layer.q.scales", "F16")],
        config.clone(),
    );
    assert_refused(no_biases.path(), "F16 scales and no biases");
    let mixed = dir_with(
        &[
            ("layer.q.weight", "U32"),
            ("layer.q.scales", "F16"),
            ("layer.q.biases", "BF16"),
        ],
        config.clone(),
    );
    assert_refused(mixed.path(), "F16 scales and BF16 biases");
    let affine_global = dir_with(
        &[
            ("layer.q.weight", "U32"),
            ("layer.q.scales", "BF16"),
            ("layer.q.biases", "BF16"),
            ("layer.q.global_scale", "F32"),
        ],
        config.clone(),
    );
    assert_refused(affine_global.path(), "carries a global_scale");
    let orphan_global = dir_with(
        &[
            ("layer.q.weight", "U32"),
            ("layer.q.scales", "BF16"),
            ("layer.q.biases", "BF16"),
            ("layer.k.global_scale", "F32"),
        ],
        config.clone(),
    );
    assert_refused(orphan_global.path(), "layer.k.global_scale");
    let byte_scales_with_biases = dir_with(
        &[
            ("layer.q.weight", "U32"),
            ("layer.q.scales", "U8"),
            ("layer.q.biases", "BF16"),
        ],
        config.clone(),
    );
    assert_refused(byte_scales_with_biases.path(), "beside biases");
    let weight_without_scales = dir_with(
        &[
            ("layer.q.weight", "U32"),
            ("layer.q.scales", "BF16"),
            ("layer.q.biases", "BF16"),
            ("layer.k.weight", "U32"),
        ],
        config.clone(),
    );
    assert_refused(weight_without_scales.path(), "layer.k.weight (U32)");
    let nothing_quantized = dir_with(&[("norm.weight", "BF16")], config);
    assert_refused(
        nothing_quantized.path(),
        "no tensor forms a quantized module",
    );
}

#[test]
fn non_affine_module_is_claimed_and_reported_by_its_tensors() {
    let dir = dir_with(
        &[
            ("layer.q.weight", "U32"),
            ("layer.q.scales", "F8_E8M0"),
            ("layer.q.global_scale", "F32"),
            ("norm.weight", "BF16"),
        ],
        Some(serde_json::json!({
            "quantization": { "group_size": 32, "bits": 4, "mode": "mxfp4" }
        })),
    );
    assert_eq!(detect_format(dir.path()), ModelFormat::MlxQuantized);
    let bad_global = dir_with(
        &[
            ("layer.q.weight", "U32"),
            ("layer.q.scales", "F8_E8M0"),
            ("layer.q.global_scale", "BF16"),
        ],
        Some(mlx_config()),
    );
    assert_refused(bad_global.path(), "BF16 global_scale");
}

#[test]
fn foreign_declarations_refuse() {
    let top = dir_with(
        &[("w", "BF16")],
        Some(serde_json::json!({ "quantization_config": { "quant_method": "gptq" } })),
    );
    assert_refused(top.path(), "config.json quantization_config");
    let nested = dir_with(
        &[("w", "BF16")],
        Some(serde_json::json!({
            "text_config": { "quantization_config": { "quant_method": "awq" } }
        })),
    );
    assert_refused(nested.path(), "text_config.quantization_config");
    let hf = dir_with(&[("w", "BF16")], Some(serde_json::json!({})));
    fs::write(hf.path().join("hf_quant_config.json"), b"{}").unwrap();
    assert_refused(hf.path(), "hf_quant_config.json");
}

#[test]
fn undeclared_quantized_dtypes_refuse() {
    for dtype in QUANTIZED_DTYPES {
        let dir = dir_with(&[("w", dtype), ("norm.weight", "BF16")], None);
        assert_refused(dir.path(), &format!("w ({dtype})"));
    }
    let global = dir_with(&[("w.global_scale", "F32")], None);
    assert_refused(global.path(), "w.global_scale (F32)");
    let declared_unclaimed = dir_with(
        &[
            ("layer.q.weight", "U32"),
            ("layer.q.scales", "BF16"),
            ("layer.q.biases", "BF16"),
            ("router.weight", "F8_E4M3"),
        ],
        Some(mlx_config()),
    );
    assert_refused(declared_unclaimed.path(), "router.weight (F8_E4M3)");
}

#[test]
fn layout_ambiguity_refuses() {
    let second = dir_with(&[("w", "BF16")], None);
    write_header(&second.path().join("other.safetensors"), &[("v", "BF16")]);
    assert_refused(second.path(), "must be the only one");

    let mixed = dir_with(&[("w", "BF16")], None);
    fs::write(mixed.path().join("model.gguf"), b"GGUF").unwrap();
    assert_refused(mixed.path(), "both GGUF and safetensors");

    let gguf = tempfile::tempdir().unwrap();
    fs::write(gguf.path().join("model.gguf"), b"GGUF").unwrap();
    assert_eq!(detect_format(gguf.path()), ModelFormat::Gguf);
    fs::write(gguf.path().join("other.gguf"), b"GGUF").unwrap();
    assert_refused(gguf.path(), "2 GGUF files");
}

#[test]
fn index_must_match_the_shard_headers() {
    let dir = dir_with(&[("w", "BF16"), ("v", "BF16")], None);
    let index = dir.path().join(SAFETENSORS_INDEX);
    fs::write(
        &index,
        br#"{"weight_map": {"w": "model.safetensors", "v": "model.safetensors"}}"#,
    )
    .unwrap();
    assert_eq!(detect_format(dir.path()), ModelFormat::Safetensors);
    fs::write(&index, br#"{"weight_map": {"w": "model.safetensors"}}"#).unwrap();
    assert_refused(dir.path(), "tensor v in shard model.safetensors is missing");
    fs::write(
            &index,
            br#"{"weight_map": {"w": "model.safetensors", "v": "model.safetensors", "u": "model.safetensors"}}"#,
        )
        .unwrap();
    assert_refused(dir.path(), "maps u to model.safetensors");
    fs::write(
        &index,
        br#"{"weight_map": {"w": "./model.safetensors", "v": "./model.safetensors"}}"#,
    )
    .unwrap();
    assert_eq!(detect_format(dir.path()), ModelFormat::Safetensors);
}

/// A `model.safetensors` symlink to the one shard its index lists (a
/// layout in real model directories) is that shard, so it is not an
/// unlisted file. A link to an unlisted file, and a plain copy of the
/// shard, are still refused.
#[cfg(unix)]
#[test]
fn a_symlink_to_a_listed_shard_is_not_an_unlisted_file() {
    let shard = "model.safetensors-00001-of-00001.safetensors";
    let dir = tempfile::tempdir().unwrap();
    write_header(&dir.path().join(shard), &[("w", "BF16")]);
    fs::write(
        dir.path().join(SAFETENSORS_INDEX),
        format!(r#"{{"weight_map": {{"w": "{shard}"}}}}"#),
    )
    .unwrap();
    let link = dir.path().join(SAFETENSORS_FILE);
    std::os::unix::fs::symlink(shard, &link).unwrap();
    assert_eq!(detect_format(dir.path()), ModelFormat::Safetensors);

    fs::remove_file(&link).unwrap();
    let outside = tempfile::tempdir().unwrap();
    write_header(&outside.path().join("other.safetensors"), &[("w", "BF16")]);
    std::os::unix::fs::symlink(outside.path().join("other.safetensors"), &link).unwrap();
    assert_refused(dir.path(), "does not list: model.safetensors");

    fs::remove_file(&link).unwrap();
    fs::copy(dir.path().join(shard), &link).unwrap();
    assert_refused(dir.path(), "does not list: model.safetensors");
}

/// A converted directory carries `quantize_index.json` and `.q4` files and
/// may keep a `config.json` naming a quantization. Native Q4 detection
/// runs before any configuration rule, so it stays `Q4`.
#[test]
fn native_q4_output_with_a_quantization_object_stays_q4() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("model_layers_0_weight.q4"), b"stub").unwrap();
    write_config(dir.path(), mlx_config());
    assert_eq!(detect_format(dir.path()), ModelFormat::Q4);
}

#[test]
fn a_non_object_quantization_refuses() {
    let dir = dir_with(&AFFINE, Some(serde_json::json!({ "quantization": true })));
    assert_refused(dir.path(), "is not an object");
}

/// An unreadable `config.json` declares nothing: a floating-point checkpoint
/// stays `Safetensors` for its loader to report, and a quantized one is
/// still refused by its dtypes.
#[test]
fn an_unreadable_config_declares_nothing() {
    let float = dir_with(&[("w", "BF16")], None);
    fs::write(float.path().join("config.json"), b"{").unwrap();
    assert_eq!(detect_format(float.path()), ModelFormat::Safetensors);
    let quantized = dir_with(&AFFINE, None);
    fs::write(quantized.path().join("config.json"), b"{").unwrap();
    assert_refused(quantized.path(), "declares a quantization");
}

/// `hf_quant_config.json` is a declaration of its own: an unreadable
/// `config.json` beside it cannot hide it, whatever the tensors are.
#[test]
fn an_unreadable_config_does_not_hide_hf_quant_config() {
    for tensors in [&[("w", "BF16")][..], &AFFINE[..]] {
        let dir = dir_with(tensors, None);
        fs::write(dir.path().join("config.json"), b"{").unwrap();
        fs::write(dir.path().join("hf_quant_config.json"), b"{}").unwrap();
        assert_refused(dir.path(), "hf_quant_config.json");
    }
}

#[test]
fn unloadable_message_names_the_format_or_the_refusal() {
    let mlx = dir_with(&AFFINE, Some(mlx_config()));
    assert!(unloadable_format_message(mlx.path()).contains("MLX quantized"));
    let refused = dir_with(&AFFINE, None);
    assert!(unloadable_format_message(refused.path()).contains("declares a quantization"));
    let empty = tempfile::tempdir().unwrap();
    assert!(unloadable_format_message(empty.path()).contains("not a recognized"));
}
