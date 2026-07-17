#include <metal_stdlib>
#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 300)
#include <metal_simdgroup_matrix>
using namespace metal;
// ===== Q3 GEMM tiled: 3-bit plane-split (2+1) x float32, simdgroup-matrix MMA =====
// Q3 block: [f16 scale (2B)][f16 bias/min (2B)][12 bytes packed 2+1 plane-split] = 16 bytes / 32 weights.
// Plane-split layout (see weights/q3_weights.rs module docs):
//   packed[0..8]  low-2-bit plane, 4 values/byte
//   packed[8..12] high-1-bit plane, 8 values/byte
//   dequant: q[i] = low2(i) | (hi(i) << 2); weight = q * scale + bias
// Mirrors gemm_q4_tiled's BM=64 x BN=32 x BK=32 tiling (PR #270/#283 occupancy
// structure) with the Q3 16-byte block payload substituted for Q4's 20-byte one.
kernel void gemm_q3_tiled(
    device const uchar* QW [[buffer(0)]],
    device const float* X  [[buffer(1)]],
    device float* Y        [[buffer(2)]],
    constant uint& M       [[buffer(3)]],
    constant uint& N       [[buffer(4)]],
    constant uint& K       [[buffer(5)]],
    uint3 tg               [[threadgroup_position_in_grid]],
    uint tid               [[thread_index_in_threadgroup]],
    uint lane              [[thread_index_in_simdgroup]],
    uint sg                [[simdgroup_index_in_threadgroup]])
{
    constexpr uint BM = 64;
    constexpr uint BN = 32;
    constexpr uint BK = 32;
    constexpr uint BN_PAD = 40;
    constexpr uint Q3_BYTES = 16;
    constexpr uint THREADS = 128;
    const uint m0 = tg.y * BM;
    const uint n0 = tg.x * BN;
    const uint nb = K / 32;
    const uint row_bytes = nb * Q3_BYTES;
    threadgroup half Xtg[64][32];
    threadgroup uchar Qraw[32][Q3_BYTES];
    threadgroup half Wtg[32][40];
    threadgroup float Ytg[32][16];
    threadgroup float Zero[8][8];
    if (tid < 64) { Zero[tid / 8][tid % 8] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sg_m_base = (sg / 2) * 32;
    const uint sg_n_base = (sg % 2) * 16;
    simdgroup_float8x8 acc00,acc01,acc10,acc11,acc20,acc21,acc30,acc31;
    simdgroup_load(acc00,&Zero[0][0],8); simdgroup_load(acc01,&Zero[0][0],8);
    simdgroup_load(acc10,&Zero[0][0],8); simdgroup_load(acc11,&Zero[0][0],8);
    simdgroup_load(acc20,&Zero[0][0],8); simdgroup_load(acc21,&Zero[0][0],8);
    simdgroup_load(acc30,&Zero[0][0],8); simdgroup_load(acc31,&Zero[0][0],8);
    for (uint k0 = 0; k0 < K; k0 += BK) {
        const uint kb = k0 / 32;
        for (uint j = tid; j < BM*(BK/4); j += THREADS) {
            uint mi=j/(BK/4); uint kb4=j%(BK/4); uint gm=m0+mi; uint gk4=k0+kb4*4;
            float4 v=(gm<M)?*((device const float4*)(X+gm*K+gk4)):float4(0.0f);
            Xtg[mi][kb4*4+0]=half(v.x); Xtg[mi][kb4*4+1]=half(v.y);
            Xtg[mi][kb4*4+2]=half(v.z); Xtg[mi][kb4*4+3]=half(v.w); }
        for (uint i = tid; i < BN*Q3_BYTES; i += THREADS) {
            uint ni=i/Q3_BYTES; uint off=i%Q3_BYTES; uint gn=n0+ni; uchar v=0;
            if(gn<N){v=QW[gn*row_bytes+kb*Q3_BYTES+off];} Qraw[ni][off]=v; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // One thread per (column, value-in-block) pair — 32 values/block, unlike
        // Q4's 2-values-per-byte loop, since 3-bit values aren't byte-aligned.
        for (uint i = tid; i < BN*32; i += THREADS) {
            uint ni=i/32; uint vi=i%32;
            ushort sb=ushort(Qraw[ni][0])|(ushort(Qraw[ni][1])<<8);
            ushort bb=ushort(Qraw[ni][2])|(ushort(Qraw[ni][3])<<8);
            float d=float(as_type<half>(sb));
            float b=float(as_type<half>(bb));
            uchar low2 = (Qraw[ni][4 + vi/4] >> ((vi%4)*2)) & 0x3;
            uchar hi   = (Qraw[ni][4 + 8 + vi/8] >> (vi%8)) & 0x1;
            uchar q = low2 | (hi << 2);
            Wtg[vi][ni] = half(float(q)*d + b); }
        for (uint i = tid; i < BK*8; i += THREADS) {
            uint kk=i/8; uint pn=BN+(i%8); Wtg[kk][pn]=half(0.0f); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        [[unroll]]
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 a0,a1,a2,a3,b0,b1;
            simdgroup_load(a0,&Xtg[sg_m_base+ 0][kk],BK);
            simdgroup_load(a1,&Xtg[sg_m_base+ 8][kk],BK);
            simdgroup_load(a2,&Xtg[sg_m_base+16][kk],BK);
            simdgroup_load(a3,&Xtg[sg_m_base+24][kk],BK);
            simdgroup_load(b0,&Wtg[kk][sg_n_base+0],BN_PAD);
            simdgroup_load(b1,&Wtg[kk][sg_n_base+8],BN_PAD);
            simdgroup_multiply_accumulate(acc00,a0,b0,acc00);
            simdgroup_multiply_accumulate(acc01,a0,b1,acc01);
            simdgroup_multiply_accumulate(acc10,a1,b0,acc10);
            simdgroup_multiply_accumulate(acc11,a1,b1,acc11);
            simdgroup_multiply_accumulate(acc20,a2,b0,acc20);
            simdgroup_multiply_accumulate(acc21,a2,b1,acc21);
            simdgroup_multiply_accumulate(acc30,a3,b0,acc30);
            simdgroup_multiply_accumulate(acc31,a3,b1,acc31); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint ssg = 0; ssg < 4; ssg++) {
        if (sg == ssg) {
            simdgroup_store(acc00,&Ytg[ 0][0],16); simdgroup_store(acc01,&Ytg[ 0][8],16);
            simdgroup_store(acc10,&Ytg[ 8][0],16); simdgroup_store(acc11,&Ytg[ 8][8],16);
            simdgroup_store(acc20,&Ytg[16][0],16); simdgroup_store(acc21,&Ytg[16][8],16);
            simdgroup_store(acc30,&Ytg[24][0],16); simdgroup_store(acc31,&Ytg[24][8],16); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint smb=(ssg/2)*32; uint snb=(ssg%2)*16;
        for (uint i = tid; i < 512; i += THREADS) {
            uint lm=i/16; uint ln=i%16;
            uint gm=m0+smb+lm; uint gn=n0+snb+ln;
            if(gm<M&&gn<N){Y[gm*N+gn]=Ytg[lm][ln];} }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
#endif
