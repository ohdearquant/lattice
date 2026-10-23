use super::*;
use std::fs;

fn tempdir(name: &str) -> std::path::PathBuf {
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "lattice-model-format-test-{name}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    fs::create_dir_all(&dir).expect("create tempdir");
    dir
}

/// A safetensors file holding only a header that declares `tensors`
/// (name, dtype), each with shape `[1]`. Detection reads no payload.
fn write_header(path: &Path, tensors: &[(&str, &str)]) {
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_owned(),
        serde_json::json!({ "format": "pt" }),
    );
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

#[test]
fn detect_format_safetensors_file() {
    let dir = tempdir("safetensors-file");
    write_header(&dir.join("model.safetensors"), &[("w", "F32")]);
    assert_eq!(detect_format(&dir), ModelFormat::Safetensors);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn detect_format_refuses_an_unreadable_header() {
    let dir = tempdir("safetensors-stub");
    fs::write(dir.join("model.safetensors"), b"stub").unwrap();
    let detection = inspect_format(&dir);
    assert_eq!(detection.format(), ModelFormat::Refused);
    assert!(
        detection.refusal().is_some_and(|r| r.contains("too short")),
        "{detection:?}"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn detect_format_refuses_an_index_with_no_shards() {
    let dir = tempdir("safetensors-index");
    fs::write(dir.join("model.safetensors.index.json"), b"{}").unwrap();
    assert_eq!(detect_format(&dir), ModelFormat::Refused);
    fs::write(
        dir.join("model.safetensors.index.json"),
        br#"{"weight_map": {}}"#,
    )
    .unwrap();
    let detection = inspect_format(&dir);
    assert_eq!(detection.format(), ModelFormat::Refused);
    assert!(
        detection
            .refusal()
            .is_some_and(|r| r.contains("empty weight_map")),
        "{detection:?}"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn detect_format_q4_dir() {
    let dir = tempdir("q4");
    fs::write(dir.join("model_layers_0_weight.q4"), b"stub").unwrap();
    fs::write(dir.join("config.json"), b"{}").unwrap();
    assert_eq!(detect_format(&dir), ModelFormat::Q4);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn detect_format_prefers_safetensors_over_q4_files() {
    // A directory that (unusually) has both a safetensors file and a
    // stray .q4 file must resolve as Safetensors — the safetensors
    // loader path is untouched and takes priority.
    let dir = tempdir("mixed");
    write_header(&dir.join("model.safetensors"), &[("w", "F32")]);
    fs::write(dir.join("leftover.q4"), b"stub").unwrap();
    assert_eq!(detect_format(&dir), ModelFormat::Safetensors);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn detect_format_prefers_safetensors_index_over_q4_files() {
    // A directory that has an index.json (sharded safetensors) and a
    // stray .q4 file must also resolve as Safetensors. This is the
    // precedence the three benchmark binaries' old inline heuristic got
    // wrong: `!model.safetensors.exists() && has_q4_file` never checked
    // index.json, so a directory with an index file plus .q4 files used
    // to route those binaries to the Q4 loader. The canonical detector
    // checks both sentinel files before falling through to .q4, so this
    // case now resolves to Safetensors everywhere.
    let dir = tempdir("mixed-index");
    write_header(&dir.join("model.safetensors"), &[("w", "F32")]);
    fs::write(
        dir.join("model.safetensors.index.json"),
        br#"{"weight_map": {"w": "model.safetensors"}}"#,
    )
    .unwrap();
    fs::write(dir.join("leftover.q4"), b"stub").unwrap();
    assert_eq!(detect_format(&dir), ModelFormat::Safetensors);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn detect_format_empty_dir_is_unknown() {
    let dir = tempdir("empty");
    assert_eq!(detect_format(&dir), ModelFormat::Unknown);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn detect_format_unrelated_files_is_unknown() {
    let dir = tempdir("unrelated");
    fs::write(dir.join("readme.txt"), b"hello").unwrap();
    fs::write(dir.join("config.json"), b"{}").unwrap();
    assert_eq!(detect_format(&dir), ModelFormat::Unknown);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn detect_format_unreadable_dir_is_unknown() {
    // A directory that does not exist at all: `read_dir` fails, and the
    // fail-closed path (`.ok()` -> `None` -> `is_some() == false`) must
    // resolve to `Unknown` rather than panicking or defaulting to a
    // recognized format.
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "lattice-model-format-test-does-not-exist-{}",
        std::process::id()
    ));
    assert_eq!(detect_format(&dir), ModelFormat::Unknown);
}

#[test]
fn metal_gpu_required_message_mentions_rebuild_flags() {
    let msg = metal_gpu_required_message(Path::new("/tmp/some-q4-dir"));
    assert!(msg.contains("metal-gpu"));
    assert!(msg.contains("--features"));
}

#[test]
fn unrecognized_format_message_mentions_expected_files() {
    let msg = unrecognized_format_message(Path::new("/tmp/bogus"));
    assert!(msg.contains("model.safetensors"));
    assert!(msg.contains(".q4"));
}

#[test]
fn detect_family_names_gemma4_only_from_the_top_level_model_type() {
    let dir = tempdir("family-gemma4");
    fs::write(
        dir.join("config.json"),
        br#"{"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}}"#,
    )
    .unwrap();
    assert_eq!(detect_family(&dir), ModelFamily::Gemma4);
    assert_eq!(ModelFamily::Gemma4.name(), "gemma4");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn detect_family_ignores_nested_model_type_values() {
    // A Qwen3.5 vision-language config nests other `model_type` values;
    // only the top-level key decides.
    let dir = tempdir("family-nested");
    fs::write(
        dir.join("config.json"),
        br#"{"model_type": "qwen3_5", "text_config": {"model_type": "gemma4"}}"#,
    )
    .unwrap();
    assert_eq!(detect_family(&dir), ModelFamily::Qwen35);
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn detect_family_falls_back_to_qwen35_for_anything_unclaimed() {
    let missing = tempdir("family-missing");
    assert_eq!(detect_family(&missing), ModelFamily::Qwen35);
    let malformed = tempdir("family-malformed");
    fs::write(malformed.join("config.json"), b"{not json").unwrap();
    assert_eq!(detect_family(&malformed), ModelFamily::Qwen35);
    let other = tempdir("family-other");
    fs::write(other.join("config.json"), br#"{"model_type": 4}"#).unwrap();
    assert_eq!(detect_family(&other), ModelFamily::Qwen35);
    let absent = std::env::temp_dir().join(format!(
        "lattice-model-format-test-family-absent-{}",
        std::process::id()
    ));
    assert_eq!(detect_family(&absent), ModelFamily::Qwen35);
    for dir in [missing, malformed, other] {
        fs::remove_dir_all(&dir).ok();
    }
}

// Fail-closed without metal-gpu: a Q4 directory must never silently
// fall back to the CPU safetensors loader. This test only compiles
// (and only means anything) when the binary is built WITHOUT the
// metal-gpu feature — it asserts that the code path this binary
// would take for a Q4 directory is the explicit error message above,
// never `Qwen35Model::from_safetensors`.
#[cfg(not(feature = "metal-gpu"))]
#[test]
fn q4_dir_without_metal_gpu_feature_fails_closed() {
    let dir = tempdir("q4-no-metal");
    fs::write(dir.join("model_layers_0_weight.q4"), b"stub").unwrap();
    assert_eq!(detect_format(&dir), ModelFormat::Q4);
    // Scope: detection + message content only. This proves a Q4 dir
    // classifies as `ModelFormat::Q4` (so a caller's `match` takes the
    // fail-closed branch, never a safetensors load) and that the error
    // names the rebuild flags. It does not drive any binary's `main`
    // itself — those exit the process, which a unit test cannot cross.
    let msg = metal_gpu_required_message(&dir);
    assert!(msg.contains("metal-gpu"));
    fs::remove_dir_all(&dir).ok();
}
