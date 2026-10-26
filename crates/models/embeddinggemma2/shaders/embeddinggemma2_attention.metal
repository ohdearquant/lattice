#include <metal_stdlib>
using namespace metal;

// ===== Bidirectional grouped-query attention =====
// One simdgroup per query row, eight rows per threadgroup, one query head per threadgroup column.
// Keys and values are staged in threadgroup memory and shared by the eight rows. The softmax is
// online, so no [seq, seq] matrix exists anywhere. Every key a row may see is visited:
//   full layer     : all keys
//   sliding layer  : keys j with |i - j| <= window
// A lane owns the float4 chunks lane, lane + 32, ... of the head, so any head_dim that is a
// multiple of 4 works; lanes past the last chunk contribute zero. The template parameter
// __HD__ is substituted by the host once per distinct head width.

kernel void eg2_attention_hd__HD__(
    device const float4* Q4 [[buffer(0)]],
    device const float4* K4 [[buffer(1)]],
    device const float4* V4 [[buffer(2)]],
    device float4* O4       [[buffer(3)]],
    constant Eg2AttentionParams& p [[buffer(4)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid  [[thread_index_in_threadgroup]])
{
    constexpr uint HD = __HD__u;
    constexpr uint HD4 = HD / 4u;
    constexpr uint NC = (HD4 + 31u) / 32u;
    constexpr uint TK_BY_MEMORY = 2048u / HD;
    constexpr uint TK = TK_BY_MEMORY > 16u ? 16u : (TK_BY_MEMORY > 0u ? TK_BY_MEMORY : 1u);
    constexpr uint ROWS = 8u;
    constexpr uint THREADS = ROWS * 32u;

    const uint q_head = tgp.x;
    if (q_head >= p.heads) return;
    const uint kv_head = q_head / (p.heads / p.kv_heads);
    const uint q_dim4 = p.heads * HD4;
    const uint kv_dim4 = p.kv_heads * HD4;

    const uint q_block_start = tgp.y * ROWS;
    const uint q_block_last = min(p.seq_len, q_block_start + ROWS) - 1u;
    const uint qi = q_block_start + sg;
    const bool row_active = qi < p.seq_len;

    uint tg_lo = 0u;
    uint tg_hi = p.seq_len;
    uint row_lo = 0u;
    uint row_hi = p.seq_len;
    if (p.use_window != 0u) {
        tg_lo = q_block_start > p.window ? q_block_start - p.window : 0u;
        tg_hi = min(p.seq_len, q_block_last + p.window + 1u);
        row_lo = qi > p.window ? qi - p.window : 0u;
        row_hi = min(p.seq_len, qi + p.window + 1u);
    }

    threadgroup float4 K_tile[TK][HD4];
    threadgroup float4 V_tile[TK][HD4];

    float4 q_frag[NC];
    float4 o_frag[NC];
    for (uint c = 0u; c < NC; ++c) {
        const uint d4 = lane + 32u * c;
        q_frag[c] = (row_active && d4 < HD4) ? Q4[qi * q_dim4 + q_head * HD4 + d4] : float4(0.0f);
        o_frag[c] = float4(0.0f);
    }
    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (uint k_start = tg_lo; k_start < tg_hi; k_start += TK) {
        const uint tile_len = min(TK, tg_hi - k_start);

        for (uint idx = tid; idx < tile_len * HD4; idx += THREADS) {
            const uint tk = idx / HD4;
            const uint d4 = idx % HD4;
            const uint base = (k_start + tk) * kv_dim4 + kv_head * HD4 + d4;
            K_tile[tk][d4] = K4[base];
            V_tile[tk][d4] = V4[base];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_active) {
            float scores[TK];
            float tile_max = -INFINITY;
            for (uint tk = 0u; tk < tile_len; ++tk) {
                const uint key = k_start + tk;
                float s = -INFINITY;
                if (key >= row_lo && key < row_hi) {
                    float partial = 0.0f;
                    for (uint c = 0u; c < NC; ++c) {
                        const uint d4 = lane + 32u * c;
                        if (d4 < HD4) {
                            partial += dot(q_frag[c], K_tile[tk][d4]);
                        }
                    }
                    s = simd_sum(partial) * p.scale;
                    if (isnan(s)) {
                        l_i = NAN;
                    }
                }
                scores[tk] = s;
                tile_max = max(tile_max, s);
            }

            if (tile_max > -INFINITY) {
                const float m_new = max(m_i, tile_max);
                const float alpha = exp(m_i - m_new);
                float l_new = l_i * alpha;
                float4 o_update[NC];
                for (uint c = 0u; c < NC; ++c) {
                    o_update[c] = float4(0.0f);
                }
                for (uint tk = 0u; tk < tile_len; ++tk) {
                    const float s = scores[tk];
                    if (s > -INFINITY) {
                        const float p_ij = exp(s - m_new);
                        l_new += p_ij;
                        for (uint c = 0u; c < NC; ++c) {
                            const uint d4 = lane + 32u * c;
                            if (d4 < HD4) {
                                o_update[c] += p_ij * V_tile[tk][d4];
                            }
                        }
                    }
                }
                for (uint c = 0u; c < NC; ++c) {
                    o_frag[c] = o_frag[c] * alpha + o_update[c];
                }
                l_i = l_new;
                m_i = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_active) {
        // Every row sees at least its own key, so a healthy row has a positive finite
        // denominator. Anything else is written as NaN so the host's finite-output check
        // rejects the whole result instead of consuming a silently zeroed row.
        const bool healthy = isfinite(l_i) && l_i > 0.0f;
        for (uint c = 0u; c < NC; ++c) {
            const uint d4 = lane + 32u * c;
            if (d4 < HD4) {
                O4[qi * q_dim4 + q_head * HD4 + d4] = healthy ? o_frag[c] / l_i : float4(NAN);
            }
        }
    }
}
