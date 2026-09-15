#include <metal_stdlib>
using namespace metal;

// Compact tables preserve the CPU reference's f32 angles and stride-half pairs.
kernel void ernie45_rope(
    device float* x             [[buffer(0)]],
    device const float* cos_tab  [[buffer(1)]],
    device const float* sin_tab  [[buffer(2)]],
    constant uint& num_tokens   [[buffer(3)]],
    constant uint& num_heads    [[buffer(4)]],
    constant uint& head_dim     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint half_dim = head_dim / 2;
    const uint total_pairs = num_tokens * num_heads * half_dim;
    if (gid >= total_pairs) return;

    const uint pair = gid % half_dim;
    const uint head = (gid / half_dim) % num_heads;
    const uint token = gid / (num_heads * half_dim);
    const uint base = (token * num_heads + head) * head_dim;
    const uint angle = token * half_dim + pair;
    const float c = cos_tab[angle];
    const float s = sin_tab[angle];
    const float lo = x[base + pair];
    const float hi = x[base + half_dim + pair];
    x[base + pair] = lo * c - hi * s;
    x[base + half_dim + pair] = hi * c + lo * s;
}
