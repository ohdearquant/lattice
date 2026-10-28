//! Offline syntax, semantic, entry-point, and Rust-wiring checks for the standalone GEMM shader.

use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};

const GEMM_FILE: &str = "gemm.wgsl";
const GEMM_SOURCE: &str = include_str!("../src/forward/wgsl/gemm.wgsl");
const GEMM_ENTRY_POINTS: &[&str] = &["gemm_bt", "gemm_nn"];

#[test]
fn gemm_case_source_is_the_file_it_names() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/forward/wgsl")
        .join(GEMM_FILE);
    let on_disk = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("shader {GEMM_FILE} is unreadable: {error}"));
    assert_eq!(
        on_disk, GEMM_SOURCE,
        "{GEMM_FILE} case does not carry the file it names"
    );
}

#[test]
fn gemm_wgsl_parses_validates_and_exports_expected_entry_points() {
    let module = wgsl::parse_str(GEMM_SOURCE)
        .unwrap_or_else(|error| panic!("{GEMM_FILE} failed WGSL parsing: {error}"));
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|error| panic!("{GEMM_FILE} failed WGSL validation: {error}"));
    for expected in GEMM_ENTRY_POINTS {
        assert!(
            module
                .entry_points
                .iter()
                .any(|entry| entry.name == *expected),
            "{GEMM_FILE} does not export expected entry point {expected}"
        );
    }
}

#[test]
fn gpu_gemm_loads_the_gemm_shader_file() {
    let gemm_source = include_str!("../src/forward/gpu_gemm.rs");
    let gemm_source_compact: String = gemm_source.split_whitespace().collect();
    assert!(
        gemm_source_compact
            .contains(r#"constGEMM_SHADER:&str=concat!("\n",include_str!("wgsl/gemm.wgsl"));"#),
        "gpu_gemm.rs must load gemm.wgsl"
    );
    assert!(
        !gemm_source.contains("r#\""),
        "gpu_gemm.rs must not retain inline WGSL"
    );
}
