#include <metal_stdlib>
using namespace metal;

// Match fused_attention's float4 reduction and online-softmax tile order so
// single-row decode does not introduce a second attention arithmetic contract.
kernel void ernie45_decode_attention(
    device const float4* Q4 [[buffer(0)]],
    device const float4* K4 [[buffer(1)]],
    device const float4* V4 [[buffer(2)]],
    device float4* O4 [[buffer(3)]],
    constant uint& cache_len [[buffer(4)]],
    constant uint& kv_dim4 [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint kv_head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint HEAD_DIM4 = __FA_HEAD_DIM__u / 4u;
    constexpr uint GROUPS = __FA_GQA_GROUPS__u;
    constexpr uint TILE_K = 16u;
    constexpr uint THREADS = GROUPS * 32u;
    static_assert(HEAD_DIM4 == 32u, "ERNIE decode requires head_dim 128");
    const uint q_head = kv_head * GROUPS + tid / 32u;
    const uint q_base4 = q_head * HEAD_DIM4 + lane;
    const float4 q_frag = Q4[q_base4];
    threadgroup float4 K_tile[TILE_K][HEAD_DIM4];
    threadgroup float4 V_tile[TILE_K][HEAD_DIM4];
    float4 o_frag = float4(0.0f);
    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (uint k_start = 0; k_start < cache_len; k_start += TILE_K) {
        const uint tile_len = min(TILE_K, cache_len - k_start);
        for (uint idx = tid; idx < tile_len * HEAD_DIM4; idx += THREADS) {
            const uint tk = idx / HEAD_DIM4;
            const uint d4 = idx % HEAD_DIM4;
            const uint offset = (k_start + tk) * kv_dim4 + kv_head * HEAD_DIM4 + d4;
            K_tile[tk][d4] = K4[offset];
            V_tile[tk][d4] = V4[offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scores[TILE_K];
        float tile_max = -INFINITY;
        for (uint tk = 0; tk < tile_len; ++tk) {
            const float partial = dot(q_frag, K_tile[tk][lane]);
            const float s = simd_sum(partial) * scale;
            scores[tk] = s;
            tile_max = max(tile_max, s);
        }
        const float m_new = max(m_i, tile_max);
        const float alpha = exp(m_i - m_new);
        float l_new = l_i * alpha;
        float4 o_update = float4(0.0f);
        for (uint tk = 0; tk < tile_len; ++tk) {
            const float p_ij = exp(scores[tk] - m_new);
            l_new += p_ij;
            o_update += p_ij * V_tile[tk][lane];
        }
        o_frag = o_frag * alpha + o_update;
        l_i = l_new;
        m_i = m_new;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Literal assignment preserves the existing fail-closed softmax contract.
    if (isfinite(l_i) && l_i > 0.0f) {
        O4[q_base4] = o_frag * (1.0f / l_i);
    } else {
        O4[q_base4] = float4(0.0f);
    }
}
