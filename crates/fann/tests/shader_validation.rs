//! WGSL shader validation tests using naga
//!
//! These tests validate shader syntax and semantics without requiring GPU hardware.
//! Naga is the same shader compiler used by wgpu, ensuring these tests catch
//! the same errors that would occur during actual shader compilation.

use naga::front::wgsl;
use naga::valid::{Capabilities, ValidationFlags, Validator};

/// Helper to validate a WGSL shader string
fn validate_wgsl(source: &str) -> Result<(), String> {
    let module = wgsl::parse_str(source).map_err(|e| format!("Parse error: {e}"))?;

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    validator
        .validate(&module)
        .map_err(|e| format!("Validation error: {e}"))?;

    Ok(())
}

/// Verify shader contains expected patterns
fn check_shader_patterns(source: &str, patterns: &[&str]) {
    for pattern in patterns {
        assert!(
            source.contains(pattern),
            "Shader missing expected pattern: {pattern}"
        );
    }
}

// ============================================================================
// Matrix Operations Shaders
// ============================================================================

#[test]
fn test_matmul_shader_valid() {
    let source = include_str!("../src/gpu/shaders.rs");
    // Extract MATMUL_SHADER constant value
    let matmul_start = source.find("pub const MATMUL_SHADER: &str = r#\"").unwrap();
    let matmul_source = &source[matmul_start..];
    let shader_start = matmul_source.find("r#\"").unwrap() + 3;
    let shader_end = matmul_source[shader_start..].find("\"#;").unwrap();
    let shader = &matmul_source[shader_start..shader_start + shader_end];

    validate_wgsl(shader).expect("MATMUL_SHADER should be valid WGSL");
    check_shader_patterns(
        shader,
        &[
            "@compute",
            "@workgroup_size(32, 1, 1)",
            "var<uniform>",
            "var<storage",
        ],
    );
}

#[test]
fn test_matmul_relu_shader_valid() {
    let source = include_str!("../src/gpu/shaders.rs");
    let start = source
        .find("pub const MATMUL_RELU_SHADER: &str = r#\"")
        .unwrap();
    let rest = &source[start..];
    let shader_start = rest.find("r#\"").unwrap() + 3;
    let shader_end = rest[shader_start..].find("\"#;").unwrap();
    let shader = &rest[shader_start..shader_start + shader_end];

    validate_wgsl(shader).expect("MATMUL_RELU_SHADER should be valid WGSL");
    check_shader_patterns(shader, &["@compute", "max(0.0, sum)"]);
}

// ============================================================================
// Activation Function Shaders
// ============================================================================

#[test]
fn test_relu_shader_valid() {
    let source = include_str!("../src/gpu/shaders.rs");
    let start = source.find("pub const RELU_SHADER: &str = r#\"").unwrap();
    let rest = &source[start..];
    let shader_start = rest.find("r#\"").unwrap() + 3;
    let shader_end = rest[shader_start..].find("\"#;").unwrap();
    let shader = &rest[shader_start..shader_start + shader_end];

    validate_wgsl(shader).expect("RELU_SHADER should be valid WGSL");
    check_shader_patterns(
        shader,
        &[
            "@compute",
            "@workgroup_size(256, 1, 1)",
            "max(data[idx], 0.0)",
        ],
    );
}

#[test]
fn test_leaky_relu_shader_valid() {
    let source = include_str!("../src/gpu/shaders.rs");
    let start = source
        .find("pub const LEAKY_RELU_SHADER: &str = r#\"")
        .unwrap();
    let rest = &source[start..];
    let shader_start = rest.find("r#\"").unwrap() + 3;
    let shader_end = rest[shader_start..].find("\"#;").unwrap();
    let shader = &rest[shader_start..shader_start + shader_end];

    validate_wgsl(shader).expect("LEAKY_RELU_SHADER should be valid WGSL");
    check_shader_patterns(shader, &["@compute", "alpha", "select"]);
}

#[test]
fn test_sigmoid_shader_valid() {
    let source = include_str!("../src/gpu/shaders.rs");
    let start = source
        .find("pub const SIGMOID_SHADER: &str = r#\"")
        .unwrap();
    let rest = &source[start..];
    let shader_start = rest.find("r#\"").unwrap() + 3;
    let shader_end = rest[shader_start..].find("\"#;").unwrap();
    let shader = &rest[shader_start..shader_start + shader_end];

    validate_wgsl(shader).expect("SIGMOID_SHADER should be valid WGSL");
    check_shader_patterns(shader, &["@compute", "if (x >= 0.0)", "exp(-x)", "exp(x)"]);
    assert!(
        !shader.contains("clamp("),
        "Sigmoid must preserve extreme inputs"
    );
}

#[test]
fn test_tanh_shader_valid() {
    let source = include_str!("../src/gpu/shaders.rs");
    let start = source.find("pub const TANH_SHADER: &str = r#\"").unwrap();
    let rest = &source[start..];
    let shader_start = rest.find("r#\"").unwrap() + 3;
    let shader_end = rest[shader_start..].find("\"#;").unwrap();
    let shader = &rest[shader_start..shader_start + shader_end];

    validate_wgsl(shader).expect("TANH_SHADER should be valid WGSL");
    check_shader_patterns(shader, &["@compute", "tanh(data[idx])"]);
    assert!(
        !shader.contains("clamp("),
        "Tanh must preserve extreme inputs"
    );
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

#[test]
fn test_activation_shaders_preserve_extreme_inputs() {
    let source = include_str!("../src/gpu/shaders.rs");

    assert!(
        source.contains("if (x >= 0.0)"),
        "Sigmoid should use a sign-stable branch"
    );
    assert!(
        source.contains("data[idx] = tanh(data[idx]);"),
        "Tanh should evaluate the original input"
    );
}

// ============================================================================
// Workgroup Size Tests
// ============================================================================

#[test]
fn test_shader_workgroup_sizes() {
    let source = include_str!("../src/gpu/shaders.rs");

    // MatMul uses 32 (Apple Silicon SIMD lanes)
    let matmul_section = &source[..source.find("RELU_SHADER").unwrap()];
    assert!(
        matmul_section.contains("@workgroup_size(32, 1, 1)"),
        "MatMul should use workgroup size 32 for Apple Silicon"
    );

    // Element-wise ops use 256 for throughput
    let relu_start = source.find("RELU_SHADER").unwrap();
    let relu_section = &source[relu_start..];
    assert!(
        relu_section.contains("@workgroup_size(256, 1, 1)"),
        "Element-wise ops should use workgroup size 256"
    );
}
