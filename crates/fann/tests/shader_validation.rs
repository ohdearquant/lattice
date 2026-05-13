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
    // Check for numerical stability clamping
    check_shader_patterns(shader, &["@compute", "clamp", "exp"]);
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
    check_shader_patterns(shader, &["@compute", "clamp", "tanh"]);
}

// ============================================================================
// Softmax Shaders (3-pass implementation)
// ============================================================================

#[test]
fn test_softmax_max_shader_valid() {
    let source = include_str!("../src/gpu/shaders.rs");
    let start = source
        .find("pub const SOFTMAX_MAX_SHADER: &str = r#\"")
        .unwrap();
    let rest = &source[start..];
    let shader_start = rest.find("r#\"").unwrap() + 3;
    let shader_end = rest[shader_start..].find("\"#;").unwrap();
    let shader = &rest[shader_start..shader_start + shader_end];

    validate_wgsl(shader).expect("SOFTMAX_MAX_SHADER should be valid WGSL");
    check_shader_patterns(
        shader,
        &[
            "@compute",
            "workgroupBarrier",
            "var<workgroup>",
            "shared_max",
        ],
    );
}

#[test]
fn test_softmax_exp_sum_shader_valid() {
    let source = include_str!("../src/gpu/shaders.rs");
    let start = source
        .find("pub const SOFTMAX_EXP_SUM_SHADER: &str = r#\"")
        .unwrap();
    let rest = &source[start..];
    let shader_start = rest.find("r#\"").unwrap() + 3;
    let shader_end = rest[shader_start..].find("\"#;").unwrap();
    let shader = &rest[shader_start..shader_start + shader_end];

    validate_wgsl(shader).expect("SOFTMAX_EXP_SUM_SHADER should be valid WGSL");
    // Check for numerical stability (exp(x - max))
    check_shader_patterns(shader, &["@compute", "exp", "workgroupBarrier"]);
}

#[test]
fn test_softmax_norm_shader_valid() {
    let source = include_str!("../src/gpu/shaders.rs");
    let start = source
        .find("pub const SOFTMAX_NORM_SHADER: &str = r#\"")
        .unwrap();
    let rest = &source[start..];
    let shader_start = rest.find("r#\"").unwrap() + 3;
    let shader_end = rest[shader_start..].find("\"#;").unwrap();
    let shader = &rest[shader_start..shader_start + shader_end];

    validate_wgsl(shader).expect("SOFTMAX_NORM_SHADER should be valid WGSL");
    // Check for epsilon to prevent division by zero
    check_shader_patterns(shader, &["@compute", "1e-10"]);
}

// ============================================================================
// Optimizer Shaders
// ============================================================================

#[test]
fn test_sgd_momentum_shader_valid() {
    let source = include_str!("../src/gpu/shaders.rs");
    let start = source
        .find("pub const SGD_MOMENTUM_SHADER: &str = r#\"")
        .unwrap();
    let rest = &source[start..];
    let shader_start = rest.find("r#\"").unwrap() + 3;
    let shader_end = rest[shader_start..].find("\"#;").unwrap();
    let shader = &rest[shader_start..shader_start + shader_end];

    validate_wgsl(shader).expect("SGD_MOMENTUM_SHADER should be valid WGSL");
    check_shader_patterns(
        shader,
        &[
            "@compute",
            "momentum",
            "learning_rate",
            "velocity",
            "weights",
        ],
    );
}

#[test]
fn test_adam_shader_valid() {
    let source = include_str!("../src/gpu/shaders.rs");
    let start = source.find("pub const ADAM_SHADER: &str = r#\"").unwrap();
    let rest = &source[start..];
    let shader_start = rest.find("r#\"").unwrap() + 3;
    let shader_end = rest[shader_start..].find("\"#;").unwrap();
    let shader = &rest[shader_start..shader_start + shader_end];

    validate_wgsl(shader).expect("ADAM_SHADER should be valid WGSL");
    // Check for Adam-specific parameters
    check_shader_patterns(
        shader,
        &["@compute", "beta1", "beta2", "epsilon", "bias_correction"],
    );
}

#[test]
fn test_adamw_shader_valid() {
    let source = include_str!("../src/gpu/shaders.rs");
    let start = source.find("pub const ADAMW_SHADER: &str = r#\"").unwrap();
    let rest = &source[start..];
    let shader_start = rest.find("r#\"").unwrap() + 3;
    let shader_end = rest[shader_start..].find("\"#;").unwrap();
    let shader = &rest[shader_start..shader_start + shader_end];

    validate_wgsl(shader).expect("ADAMW_SHADER should be valid WGSL");
    // Check for weight decay (AdamW specific)
    check_shader_patterns(shader, &["@compute", "weight_decay", "beta1", "beta2"]);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

#[test]
fn test_shader_numerical_stability_patterns() {
    let source = include_str!("../src/gpu/shaders.rs");

    // Sigmoid should clamp inputs to prevent exp() overflow
    assert!(
        source.contains("clamp(data[idx], -10.0, 10.0)"),
        "Sigmoid should clamp inputs for numerical stability"
    );

    // Tanh should clamp inputs
    assert!(
        source.contains("clamp(data[idx], -5.0, 5.0)"),
        "Tanh should clamp inputs for numerical stability"
    );

    // Softmax should use max subtraction for stability
    assert!(
        source.contains("exp(data[idx] - m)"),
        "Softmax should subtract max for numerical stability"
    );

    // Division should have epsilon guard
    assert!(
        source.contains("1e-10") || source.contains("1e-8"),
        "Division operations should have epsilon guards"
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
