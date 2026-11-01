#include <metal_stdlib>
using namespace metal;

// EmbeddingGemma 2 text-encoder kernels, f32 throughout. The shared matmul, RMSNorm, copy, add
// and stride-half RoPE kernels come from the other shader files; this file adds what they lack.

// ===== GELU (tanh form) times a second operand =====
// gate = gelu_tanh(gate) * other. The exact tanh is used on purpose: the rational
// approximation used by the CPU fast path has an error larger than the parity tolerance.
kernel void eg2_gelu_mul(
    device float* gate        [[buffer(0)]],
    device const float* other [[buffer(1)]],
    constant uint& count      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const float SQRT_2_OVER_PI = 0.7978846f;
    const float COEFF = 0.044715f;
    const float x = gate[gid];
    const float inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    gate[gid] = 0.5f * x * (1.0f + precise::tanh(inner)) * other[gid];
}

// ===== RMSNorm without a learned weight (the value norm) =====
// x[row] = x[row] * rsqrt(mean(x[row]^2) + eps). One threadgroup of 256 threads per row.
kernel void eg2_rms_norm_noweight(
    device float* x          [[buffer(0)]],
    constant uint& row_len   [[buffer(1)]],
    constant uint& num_rows  [[buffer(2)]],
    constant float& eps      [[buffer(3)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]])
{
    if (gid >= num_rows) return;
    constexpr uint RMS_WG = 256;
    const uint base = gid * row_len;

    threadgroup float shared[RMS_WG];
    float local_sum = 0.0f;
    for (uint i = lid; i < row_len; i += tgs) {
        const float v = x[base + i];
        local_sum += v * v;
    }
    const float inv = rms_inv_from_local_sum(shared, local_sum, lid, tgs, row_len, eps);

    for (uint i = lid; i < row_len; i += tgs) {
        x[base + i] = x[base + i] * inv;
    }
}

// ===== x *= s =====
kernel void eg2_scale(
    device float* x          [[buffer(0)]],
    constant float& s        [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    x[gid] = x[gid] * s;
}

// Parameters of eg2_attention_hd<N> (see embeddinggemma2_attention.metal).
struct Eg2AttentionParams {
    uint seq_len;
    uint heads;
    uint kv_heads;
    uint window;       // inclusive radius, read only when use_window != 0
    uint use_window;
    float scale;
    uint _pad0;
    uint _pad1;
};
