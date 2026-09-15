#include <metal_stdlib>
using namespace metal;

// Host validation bounds every token ID and both flattened indexing products.
kernel void ernie45_embedding(
    device const float* weights [[buffer(0)]],
    device const uint* ids      [[buffer(1)]],
    device float* hidden        [[buffer(2)]],
    constant uint& hidden_dim   [[buffer(3)]],
    constant uint& total        [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= total) return;
    const uint token = gid / hidden_dim;
    const uint column = gid % hidden_dim;
    hidden[gid] = weights[ids[token] * hidden_dim + column];
}
