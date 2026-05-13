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
/// Uses clamping to prevent exp() overflow
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

    // Clamp to [-10, 10] to prevent exp() overflow
    // exp(88) overflows f32, exp(10) = 22026 is safe
    let x = clamp(data[idx], -10.0, 10.0);
    data[idx] = 1.0 / (1.0 + exp(-x));
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

    // Clamp to [-5, 5] - tanh saturates faster than sigmoid
    let x = clamp(data[idx], -5.0, 5.0);
    data[idx] = tanh(x);
}
"#;

/// Softmax - max finding pass
///
/// Reduces to find max value for numerical stability
pub const SOFTMAX_MAX_SHADER: &str = r#"
struct Uniforms {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

var<workgroup> shared_max: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = global_id.x;
    let lid = local_id.x;

    // Load value or -inf if out of bounds
    var val: f32 = -3.4028235e+38; // -FLT_MAX
    if (idx < uniforms.size) {
        val = data[idx];
    }
    shared_max[lid] = val;
    workgroupBarrier();

    // Parallel reduction for max
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            shared_max[lid] = max(shared_max[lid], shared_max[lid + s]);
        }
        workgroupBarrier();
    }

    // Write result from first thread
    if (lid == 0u) {
        result[0] = shared_max[0];
    }
}
"#;

/// Softmax - exp and sum pass
///
/// Computes exp(x - max) and accumulates sum
pub const SOFTMAX_EXP_SUM_SHADER: &str = r#"
struct Uniforms {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
@group(0) @binding(2) var<storage, read> max_val: array<f32>;
@group(0) @binding(3) var<storage, read_write> sum_out: array<f32>;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = global_id.x;
    let lid = local_id.x;
    let m = max_val[0];

    var val: f32 = 0.0;
    if (idx < uniforms.size) {
        // exp(x - max) for numerical stability
        val = exp(data[idx] - m);
        data[idx] = val;
    }
    shared_sum[lid] = val;
    workgroupBarrier();

    // Parallel reduction for sum
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            shared_sum[lid] = shared_sum[lid] + shared_sum[lid + s];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        sum_out[0] = shared_sum[0];
    }
}
"#;

/// Softmax - normalize pass
///
/// Divides each element by sum
pub const SOFTMAX_NORM_SHADER: &str = r#"
struct Uniforms {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;
@group(0) @binding(2) var<storage, read> sum_val: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) {
        return;
    }
    // Add epsilon to prevent division by zero
    let sum = sum_val[0] + 1e-10;
    data[idx] = data[idx] / sum;
}
"#;

/// SGD with momentum weight update
pub const SGD_MOMENTUM_SHADER: &str = r#"
struct Uniforms {
    size: u32,
    learning_rate: f32,
    momentum: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read> gradients: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocity: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) {
        return;
    }

    // v = momentum * v - lr * grad
    let v = uniforms.momentum * velocity[idx] - uniforms.learning_rate * gradients[idx];
    velocity[idx] = v;
    // w = w + v
    weights[idx] = weights[idx] + v;
}
"#;

/// Adam optimizer weight update
pub const ADAM_SHADER: &str = r#"
struct Uniforms {
    size: u32,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: f32,  // timestep for bias correction
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read> gradients: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;  // First moment
@group(0) @binding(4) var<storage, read_write> v: array<f32>;  // Second moment

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) {
        return;
    }

    let g = gradients[idx];

    // Update biased first moment estimate
    let new_m = uniforms.beta1 * m[idx] + (1.0 - uniforms.beta1) * g;
    m[idx] = new_m;

    // Update biased second raw moment estimate
    let new_v = uniforms.beta2 * v[idx] + (1.0 - uniforms.beta2) * g * g;
    v[idx] = new_v;

    // Bias correction
    let bias_correction1 = 1.0 - pow(uniforms.beta1, uniforms.t);
    let bias_correction2 = 1.0 - pow(uniforms.beta2, uniforms.t);

    let m_hat = new_m / bias_correction1;
    let v_hat = new_v / bias_correction2;

    // Update weights - use max to prevent division by near-zero
    let denom = max(sqrt(v_hat) + uniforms.epsilon, 1e-8);
    weights[idx] = weights[idx] - uniforms.learning_rate * m_hat / denom;
}
"#;

/// AdamW optimizer (decoupled weight decay)
pub const ADAMW_SHADER: &str = r#"
struct Uniforms {
    size: u32,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    t: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read> gradients: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= uniforms.size) {
        return;
    }

    let g = gradients[idx];
    let w = weights[idx];

    // Update moments
    let new_m = uniforms.beta1 * m[idx] + (1.0 - uniforms.beta1) * g;
    let new_v = uniforms.beta2 * v[idx] + (1.0 - uniforms.beta2) * g * g;
    m[idx] = new_m;
    v[idx] = new_v;

    // Bias correction
    let bias_correction1 = 1.0 - pow(uniforms.beta1, uniforms.t);
    let bias_correction2 = 1.0 - pow(uniforms.beta2, uniforms.t);

    let m_hat = new_m / bias_correction1;
    let v_hat = new_v / bias_correction2;

    // AdamW: decoupled weight decay applied separately - use max to prevent division by near-zero
    let denom = max(sqrt(v_hat) + uniforms.epsilon, 1e-8);
    weights[idx] = w - uniforms.learning_rate * (m_hat / denom + uniforms.weight_decay * w);
}
"#;
