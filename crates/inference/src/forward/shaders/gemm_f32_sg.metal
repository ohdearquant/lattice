#include <metal_stdlib>
#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 300)
#include <metal_simdgroup_matrix>
using namespace metal;
// Y[M,N] = X[M,K] . W[N,K]^T with f32 operands and f32 accumulation on simdgroup matrices.
// Tile BM=64 x BN=32 x BK=32, 128 threads = 4 simdgroups, each owning a 32x16 block of Y.
// Requires K % 4 == 0 (float4 staging); M, N and the K tail are zero-padded in the tiles.
kernel void gemm_f32_sg(
    device const float* X  [[buffer(0)]],
    device const float* W  [[buffer(1)]],
    device float* Y        [[buffer(2)]],
    constant uint& M       [[buffer(3)]],
    constant uint& N       [[buffer(4)]],
    constant uint& K       [[buffer(5)]],
    uint3 tg               [[threadgroup_position_in_grid]],
    uint tid               [[thread_index_in_threadgroup]],
    uint sg                [[simdgroup_index_in_threadgroup]])
{
    constexpr uint BM = 64;
    constexpr uint BN = 32;
    constexpr uint BK = 32;
    constexpr uint BN_PAD = 40;
    constexpr uint THREADS = 128;
    const uint m0 = tg.y * BM;
    const uint n0 = tg.x * BN;
    threadgroup float Xtg[64][32];
    threadgroup float Wtg[32][40];
    threadgroup float Ytg[32][16];
    const uint sg_m_base = (sg / 2) * 32;
    const uint sg_n_base = (sg % 2) * 16;
    // The scalar constructor builds a diagonal matrix; the accumulators need every element zero.
    const simdgroup_float8x8 zero = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc00 = zero, acc01 = zero, acc10 = zero, acc11 = zero;
    simdgroup_float8x8 acc20 = zero, acc21 = zero, acc30 = zero, acc31 = zero;
    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint j = tid; j < BM * (BK / 4); j += THREADS) {
            uint mi = j / (BK / 4); uint kb4 = j % (BK / 4);
            uint gm = m0 + mi; uint gk = k0 + kb4 * 4;
            float4 v = (gm < M && gk < K) ? *((device const float4*)(X + gm * K + gk)) : float4(0.0f);
            Xtg[mi][kb4 * 4 + 0] = v.x; Xtg[mi][kb4 * 4 + 1] = v.y;
            Xtg[mi][kb4 * 4 + 2] = v.z; Xtg[mi][kb4 * 4 + 3] = v.w;
        }
        for (uint j = tid; j < BN * (BK / 4); j += THREADS) {
            uint ni = j / (BK / 4); uint kb4 = j % (BK / 4);
            uint gn = n0 + ni; uint gk = k0 + kb4 * 4;
            float4 v = (gn < N && gk < K) ? *((device const float4*)(W + gn * K + gk)) : float4(0.0f);
            Wtg[kb4 * 4 + 0][ni] = v.x; Wtg[kb4 * 4 + 1][ni] = v.y;
            Wtg[kb4 * 4 + 2][ni] = v.z; Wtg[kb4 * 4 + 3][ni] = v.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a0, a1, a2, a3, b0, b1;
            simdgroup_load(a0, &Xtg[sg_m_base + 0][kk], BK);
            simdgroup_load(a1, &Xtg[sg_m_base + 8][kk], BK);
            simdgroup_load(a2, &Xtg[sg_m_base + 16][kk], BK);
            simdgroup_load(a3, &Xtg[sg_m_base + 24][kk], BK);
            simdgroup_load(b0, &Wtg[kk][sg_n_base + 0], BN_PAD);
            simdgroup_load(b1, &Wtg[kk][sg_n_base + 8], BN_PAD);
            simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
            simdgroup_multiply_accumulate(acc20, a2, b0, acc20);
            simdgroup_multiply_accumulate(acc21, a2, b1, acc21);
            simdgroup_multiply_accumulate(acc30, a3, b0, acc30);
            simdgroup_multiply_accumulate(acc31, a3, b1, acc31);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (m0 + BM <= M && n0 + BN <= N) {
        device float* y = Y + (m0 + sg_m_base) * N + n0 + sg_n_base;
        simdgroup_store(acc00, y + 0 * N + 0, N); simdgroup_store(acc01, y + 0 * N + 8, N);
        simdgroup_store(acc10, y + 8 * N + 0, N); simdgroup_store(acc11, y + 8 * N + 8, N);
        simdgroup_store(acc20, y + 16 * N + 0, N); simdgroup_store(acc21, y + 16 * N + 8, N);
        simdgroup_store(acc30, y + 24 * N + 0, N); simdgroup_store(acc31, y + 24 * N + 8, N);
        return;
    }
    for (uint ssg = 0; ssg < 4; ssg++) {
        if (sg == ssg) {
            simdgroup_store(acc00, &Ytg[0][0], 16); simdgroup_store(acc01, &Ytg[0][8], 16);
            simdgroup_store(acc10, &Ytg[8][0], 16); simdgroup_store(acc11, &Ytg[8][8], 16);
            simdgroup_store(acc20, &Ytg[16][0], 16); simdgroup_store(acc21, &Ytg[16][8], 16);
            simdgroup_store(acc30, &Ytg[24][0], 16); simdgroup_store(acc31, &Ytg[24][8], 16);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint smb = (ssg / 2) * 32; uint snb = (ssg % 2) * 16;
        for (uint i = tid; i < 512; i += THREADS) {
            uint lm = i / 16; uint ln = i % 16;
            uint gm = m0 + smb + lm; uint gn = n0 + snb + ln;
            if (gm < M && gn < N) { Y[gm * N + gn] = Ytg[lm][ln]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
#endif
