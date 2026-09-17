//! Offline syntax, semantic, entry-point, and Rust-wiring checks for WGPU shaders.

use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};

struct ShaderCase {
    file: &'static str,
    source: &'static str,
    entry_points: &'static [&'static str],
}

const SHADERS: &[ShaderCase] = &[
    ShaderCase {
        file: "matmul_bt.wgsl",
        source: include_str!("../src/forward/wgsl/matmul_bt.wgsl"),
        entry_points: &["gemm_bt"],
    },
    ShaderCase {
        file: "rms_norm.wgsl",
        source: include_str!("../src/forward/wgsl/rms_norm.wgsl"),
        entry_points: &["rms_norm"],
    },
    ShaderCase {
        file: "copy.wgsl",
        source: include_str!("../src/forward/wgsl/copy.wgsl"),
        entry_points: &["copy_kernel"],
    },
    ShaderCase {
        file: "add.wgsl",
        source: include_str!("../src/forward/wgsl/add.wgsl"),
        entry_points: &["add_kernel"],
    },
    ShaderCase {
        file: "silu.wgsl",
        source: include_str!("../src/forward/wgsl/silu.wgsl"),
        entry_points: &["silu_kernel"],
    },
    ShaderCase {
        file: "mul.wgsl",
        source: include_str!("../src/forward/wgsl/mul.wgsl"),
        entry_points: &["mul_kernel"],
    },
    ShaderCase {
        file: "rope.wgsl",
        source: include_str!("../src/forward/wgsl/rope.wgsl"),
        entry_points: &["rope_kernel"],
    },
    ShaderCase {
        file: "attention_scores.wgsl",
        source: include_str!("../src/forward/wgsl/attention_scores.wgsl"),
        entry_points: &["attention_scores"],
    },
    ShaderCase {
        file: "attention_softmax.wgsl",
        source: include_str!("../src/forward/wgsl/attention_softmax.wgsl"),
        entry_points: &["attention_softmax"],
    },
    ShaderCase {
        file: "attention_context.wgsl",
        source: include_str!("../src/forward/wgsl/attention_context.wgsl"),
        entry_points: &["attention_context"],
    },
    ShaderCase {
        file: "gemm.wgsl",
        source: include_str!("../src/forward/wgsl/gemm.wgsl"),
        entry_points: &["gemm_bt", "gemm_nn"],
    },
];

#[test]
fn every_shader_case_source_is_the_file_it_names() {
    // `file` and the path inside `source`'s `include_str!` are written out twice by
    // hand, so nothing binds them: a case carrying a neighbour's source would validate
    // one shader twice, never validate the other, and name the wrong file in any
    // failure it did report. Read each named file off disk and require equality.
    let wgsl_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/forward/wgsl");
    assert!(!SHADERS.is_empty(), "shader case inventory is empty");
    for case in SHADERS {
        let on_disk = std::fs::read_to_string(wgsl_dir.join(case.file))
            .unwrap_or_else(|error| panic!("shader {} is unreadable: {error}", case.file));
        assert_eq!(
            on_disk, case.source,
            "shader case {} does not carry the contents of the file it names",
            case.file
        );
    }
}

#[test]
fn wgsl_artifacts_parse_validate_and_export_expected_entry_points() {
    for case in SHADERS {
        let module = wgsl::parse_str(case.source)
            .unwrap_or_else(|error| panic!("{} failed WGSL parsing: {error}", case.file));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|error| panic!("{} failed WGSL validation: {error}", case.file));

        for expected in case.entry_points {
            assert!(
                module
                    .entry_points
                    .iter()
                    .any(|entry| entry.name == *expected),
                "{} does not export expected entry point {expected}",
                case.file
            );
        }
    }
}

#[test]
fn rust_sources_load_every_wgsl_artifact() {
    let shader_constants = include_str!("../src/forward/gpu/inner/shaders.rs");
    let shader_constants_compact: String = shader_constants.split_whitespace().collect();
    let mappings = [
        ("MATMUL_BT_SHADER", "matmul_bt.wgsl"),
        ("RMS_NORM_SHADER", "rms_norm.wgsl"),
        ("COPY_SHADER", "copy.wgsl"),
        ("ADD_SHADER", "add.wgsl"),
        ("SILU_SHADER", "silu.wgsl"),
        ("MUL_SHADER", "mul.wgsl"),
        ("ROPE_SHADER", "rope.wgsl"),
        ("ATTENTION_SCORES_SHADER", "attention_scores.wgsl"),
        ("ATTENTION_SOFTMAX_SHADER", "attention_softmax.wgsl"),
        ("ATTENTION_CONTEXT_SHADER", "attention_context.wgsl"),
    ];

    for (constant, file) in mappings {
        let declaration = format!(
            r#"pub(super)const{constant}:&str=concat!("\n",include_str!("../../wgsl/{file}"));"#
        );
        assert!(
            shader_constants_compact.contains(&declaration),
            "{constant} must load {file}"
        );
    }
    assert!(
        !shader_constants.contains("r#\""),
        "gpu/inner/shaders.rs must not retain inline WGSL"
    );

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
