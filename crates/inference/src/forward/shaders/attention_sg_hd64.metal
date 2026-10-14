#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 300)
#include <metal_simdgroup_matrix>
using namespace metal;
// Bidirectional attention at head width 64 on simdgroup matrices, without a window. Compiled
// after the file that declares Eg2AttentionParams.
// Grid (heads, ceil(seq/32)); 128 threads = 4 simdgroups, each owning 8 query rows. Keys are
// visited in tiles of 32: S = Q K^T as matrix products, an online softmax by lanes (lane l holds
// row l/4 and keys 8*(l%4)..+8 of its simdgroup's block), the running output rescaled by
// diag(alpha) as a matrix product, then O += P V. Queries are scaled once as they are staged.
// A row whose denominator is not positive and finite is written as NaN, so the host's
// finite-output check rejects the result instead of consuming a silently zeroed row.
kernel void eg2_attention_sg_hd64(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O       [[buffer(3)]],
    constant Eg2AttentionParams& p [[buffer(4)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid  [[thread_index_in_threadgroup]])
{
    constexpr uint HD = 64;
    constexpr uint HD4 = HD / 4;
    constexpr uint BQ = 32;
    constexpr uint BK = 32;
    constexpr uint KT_STRIDE = 40;
    constexpr uint THREADS = 128;

    const uint head = tgp.x;
    if (head >= p.heads) return;
    const uint kv_head = head / (p.heads / p.kv_heads);
    const uint q_stride = p.heads * HD;
    const uint kv_stride = p.kv_heads * HD;
    const uint n = p.seq_len;
    const uint q0 = tgp.y * BQ;

    threadgroup float Kt[HD][KT_STRIDE];   // key tile, transposed: Kt[d][key]
    threadgroup float4 Vtg4[BK][HD4];      // value tile; stages the query block first and the output last
    threadgroup float* Vtg = (threadgroup float*)Vtg4;
    threadgroup float Stg[4][8][BK];       // per simdgroup: scores, then probabilities
    threadgroup float Dg[4][8][8];         // per simdgroup: diag(alpha)

    for (uint j = tid; j < BQ * HD4; j += THREADS) {
        const uint r = j / HD4;
        const uint c4 = j % HD4;
        const uint gq = q0 + r;
        const float4 v = gq < n
            ? *((device const float4*)(Q + gq * q_stride + head * HD + c4 * 4)) * p.scale
            : float4(0.0f);
        Vtg4[r][c4] = v;
    }
    for (uint j = tid; j < 4 * 8 * 8; j += THREADS) {
        Dg[j / 64][(j / 8) % 8][j % 8] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 qf[8];
    for (uint d = 0; d < 8; ++d) {
        simdgroup_load(qf[d], Vtg + sg * 8 * HD + d * 8, HD);
    }
    const simdgroup_float8x8 zero = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 of[8];
    for (uint d = 0; d < 8; ++d) {
        of[d] = zero;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = lane / 4;
    const uint c0 = (lane % 4) * 8;
    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (uint k0 = 0; k0 < n; k0 += BK) {
        for (uint j = tid; j < BK * HD4; j += THREADS) {
            const uint kr = j / HD4;
            const uint c4 = j % HD4;
            const uint gk = k0 + kr;
            float4 kv = float4(0.0f);
            float4 vv = float4(0.0f);
            if (gk < n) {
                const uint base = gk * kv_stride + kv_head * HD + c4 * 4;
                kv = *((device const float4*)(K + base));
                vv = *((device const float4*)(V + base));
            }
            Kt[c4 * 4 + 0][kr] = kv.x;
            Kt[c4 * 4 + 1][kr] = kv.y;
            Kt[c4 * 4 + 2][kr] = kv.z;
            Kt[c4 * 4 + 3][kr] = kv.w;
            Vtg4[kr][c4] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kb = 0; kb < 4; ++kb) {
            simdgroup_float8x8 s = zero;
            for (uint d = 0; d < 8; ++d) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, &Kt[d * 8][kb * 8], KT_STRIDE);
                simdgroup_multiply_accumulate(s, qf[d], kt, s);
            }
            simdgroup_store(s, &Stg[sg][0][kb * 8], BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* srow = &Stg[sg][row][c0];
        float x[8];
        float tile_max = -INFINITY;
        for (uint i = 0; i < 8; ++i) {
            const float v = (k0 + c0 + i < n) ? srow[i] : -INFINITY;
            x[i] = v;
            tile_max = max(tile_max, v);
        }
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 2));
        const float m_new = max(m_i, tile_max);
        const float alpha = exp(m_i - m_new);
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            const float e = exp(x[i] - m_new);
            srow[i] = e;
            sum += e;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l_i = l_i * alpha + sum;
        m_i = m_new;
        if (lane % 4 == 0) {
            Dg[sg][row][row] = alpha;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 da;
        simdgroup_load(da, &Dg[sg][0][0], 8);
        for (uint d = 0; d < 8; ++d) {
            simdgroup_float8x8 scaled;
            simdgroup_multiply(scaled, da, of[d]);
            of[d] = scaled;
        }
        for (uint kb = 0; kb < 4; ++kb) {
            simdgroup_float8x8 pm;
            simdgroup_load(pm, &Stg[sg][0][kb * 8], BK);
            for (uint d = 0; d < 8; ++d) {
                simdgroup_float8x8 vm;
                simdgroup_load(vm, Vtg + kb * 8 * HD + d * 8, HD);
                simdgroup_multiply_accumulate(of[d], pm, vm, of[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint d = 0; d < 8; ++d) {
        simdgroup_store(of[d], Vtg + sg * 8 * HD + d * 8, HD);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint gq = q0 + sg * 8 + row;
    if (gq < n) {
        const bool healthy = isfinite(l_i) && l_i > 0.0f;
        const uint col0 = (lane % 4) * 16;
        for (uint c = 0; c < 4; ++c) {
            const float4 v = Vtg4[sg * 8 + row][col0 / 4 + c];
            *((device float4*)(O + gq * q_stride + head * HD + col0 + c * 4)) = healthy ? v / l_i : float4(NAN);
        }
    }
}
#endif
