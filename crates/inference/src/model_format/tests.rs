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
