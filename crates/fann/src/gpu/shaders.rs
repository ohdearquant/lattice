//! WGSL compute shaders for neural network operations
//!
//! All shaders follow these patterns:
//! - Numerical stability guards (clamping, max-subtraction)
//! - 4x loop unrolling for memory bandwidth
//! - Proper workgroup sizes for operation type

/// Matrix-vector multiplication with 4x unrolling
///
/// Workgroup size 32 matches Apple Silicon 32-lane SIMD
pub const MATMUL_SHADER: &str = r#"
struct Uniforms {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> input_vec: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_vec: array<f32>;

// Workgroup size 32 matches Apple Silicon SIMD lanes
@compute @workgroup_size(32, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= uniforms.rows) {
        return;
    }

    let cols = uniforms.cols;
    let row_start = row * cols;

    var sum: f32 = bias[row];

    // 4x loop unrolling for better memory bandwidth
    let vectorized_cols = cols & ~3u; // Round down to multiple of 4
    var col: u32 = 0u;

    for (; col < vectorized_cols; col += 4u) {
        let base = row_start + col;
        // Fused multiply-add operations
        sum += weights[base] * input_vec[col];
        sum += weights[base + 1u] * input_vec[col + 1u];
        sum += weights[base + 2u] * input_vec[col + 2u];
        sum += weights[base + 3u] * input_vec[col + 3u];
    }

    // Handle remaining elements
    for (; col < cols; col += 1u) {
        sum += weights[row_start + col] * input_vec[col];
    }

    output_vec[row] = sum;
}
"#;

/// Fused matmul + ReLU for reduced kernel launches
pub const MATMUL_RELU_SHADER: &str = r#"
struct Uniforms {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> input_vec: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_vec: array<f32>;

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= uniforms.rows) {
        return;
    }

    let cols = uniforms.cols;
    let row_start = row * cols;
    var sum: f32 = bias[row];

    let vectorized_cols = cols & ~3u;
    var col: u32 = 0u;

    for (; col < vectorized_cols; col += 4u) {
        let base = row_start + col;
        sum += weights[base] * input_vec[col];
        sum += weights[base + 1u] * input_vec[col + 1u];
        sum += weights[base + 2u] * input_vec[col + 2u];
        sum += weights[base + 3u] * input_vec[col + 3u];
    }

    for (; col < cols; col += 1u) {
        sum += weights[row_start + col] * input_vec[col];
    }

    // Fused ReLU activation
    output_vec[row] = max(0.0, sum);
}
"#;

/// ReLU activation - element-wise max(0, x)
///
/// Workgroup size 256 for element-wise throughput
pub const RELU_SHADER: &str = r#"
struct Uniforms {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) {
        return;
    }
    data[idx] = max(data[idx], 0.0);
}
"#;

/// Leaky ReLU activation - max(alpha * x, x)
pub const LEAKY_RELU_SHADER: &str = r#"
struct Uniforms {
    size: u32,
    alpha: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) {
        return;
    }
    let x = data[idx];
    data[idx] = select(uniforms.alpha * x, x, x > 0.0);
}
"#;

/// Sigmoid activation with numerical stability
///
/// Uses sign-dependent formulas to keep the exponent non-positive
pub const SIGMOID_SHADER: &str = r#"
struct Uniforms {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) {
        return;
    }

    let x = data[idx];
    if (x >= 0.0) {
        data[idx] = 1.0 / (1.0 + exp(-x));
    } else {
        let ex = exp(x);
        data[idx] = ex / (1.0 + ex);
    }
}
"#;

/// Tanh activation with numerical stability
pub const TANH_SHADER: &str = r#"
struct Uniforms {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) {
        return;
    }

    data[idx] = tanh(data[idx]);
}
"#;
