#include <metal_stdlib>
#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 300)
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void gemm_q8_tiled(
    device const char* QW  [[buffer(0)]],
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
    constexpr uint BN = 64;
    constexpr uint BK = 32;
    constexpr uint BN_PAD = 72;
    constexpr uint THREADS = 128;
    const uint m0 = tg.y * BM;
    const uint n0 = tg.x * BN;
    const uint nb = K / 32;
    const uint row_bytes = nb * 34;
    threadgroup half Xtg[64][32];
    threadgroup char Qraw[64][32];
    threadgroup half Qscale[64];
    threadgroup half Wtg[32][72];
    threadgroup float Ytg[32][32];
    threadgroup float Zero[8][8];
    if (tid < 64) { Zero[tid / 8][tid % 8] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint sg_m_base = (sg / 2) * 32;
    const uint sg_n_base = (sg % 2) * 32;
    simdgroup_float8x8 acc00,acc01,acc02,acc03;
    simdgroup_float8x8 acc10,acc11,acc12,acc13;
    simdgroup_float8x8 acc20,acc21,acc22,acc23;
    simdgroup_float8x8 acc30,acc31,acc32,acc33;
    simdgroup_load(acc00,&Zero[0][0],8); simdgroup_load(acc01,&Zero[0][0],8);
    simdgroup_load(acc02,&Zero[0][0],8); simdgroup_load(acc03,&Zero[0][0],8);
    simdgroup_load(acc10,&Zero[0][0],8); simdgroup_load(acc11,&Zero[0][0],8);
    simdgroup_load(acc12,&Zero[0][0],8); simdgroup_load(acc13,&Zero[0][0],8);
    simdgroup_load(acc20,&Zero[0][0],8); simdgroup_load(acc21,&Zero[0][0],8);
    simdgroup_load(acc22,&Zero[0][0],8); simdgroup_load(acc23,&Zero[0][0],8);
    simdgroup_load(acc30,&Zero[0][0],8); simdgroup_load(acc31,&Zero[0][0],8);
    simdgroup_load(acc32,&Zero[0][0],8); simdgroup_load(acc33,&Zero[0][0],8);
    for (uint k0 = 0; k0 < K; k0 += BK) {
        const uint kb = k0 / 32;
        for (uint j = tid; j < BM*(BK/4); j += THREADS) {
            uint mi=j/(BK/4); uint kb4=j%(BK/4); uint gm=m0+mi; uint gk4=k0+kb4*4;
            float4 v=(gm<M)?*((device const float4*)(X+gm*K+gk4)):float4(0.0f);
            Xtg[mi][kb4*4+0]=half(v.x); Xtg[mi][kb4*4+1]=half(v.y);
            Xtg[mi][kb4*4+2]=half(v.z); Xtg[mi][kb4*4+3]=half(v.w); }
        for (uint i = tid; i < BN; i += THREADS) {
            uint gn = n0 + i;
            if (gn < N) {
                ushort sb = ushort(QW[gn*row_bytes+kb*34+0]&0xff) | (ushort(QW[gn*row_bytes+kb*34+1]&0xff)<<8);
                Qscale[i] = as_type<half>(sb);
            } else { Qscale[i] = half(0.0f); }
        }
        for (uint i = tid; i < BN*BK; i += THREADS) {
            uint ni=i/BK; uint ki=i%BK; uint gn=n0+ni;
            Qraw[ni][ki] = (gn<N) ? QW[gn*row_bytes+kb*34+2+ki] : char(0); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < BN*BK; i += THREADS) {
            uint ni=i/BK; uint ki=i%BK;
            Wtg[ki][ni] = half(float(Qraw[ni][ki]) * float(Qscale[ni])); }
        for (uint i = tid; i < BK*8; i += THREADS) {
            uint kk=i/8; uint pn=BN+(i%8); Wtg[kk][pn]=half(0.0f); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        [[unroll]]
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_half8x8 a0,a1,a2,a3,b0,b1,b2,b3;
            simdgroup_load(a0,&Xtg[sg_m_base+ 0][kk],BK);
            simdgroup_load(a1,&Xtg[sg_m_base+ 8][kk],BK);
            simdgroup_load(a2,&Xtg[sg_m_base+16][kk],BK);
            simdgroup_load(a3,&Xtg[sg_m_base+24][kk],BK);
            simdgroup_load(b0,&Wtg[kk][sg_n_base+0],BN_PAD);
            simdgroup_load(b1,&Wtg[kk][sg_n_base+8],BN_PAD);
            simdgroup_load(b2,&Wtg[kk][sg_n_base+16],BN_PAD);
            simdgroup_load(b3,&Wtg[kk][sg_n_base+24],BN_PAD);
            simdgroup_multiply_accumulate(acc00,a0,b0,acc00);
            simdgroup_multiply_accumulate(acc01,a0,b1,acc01);
            simdgroup_multiply_accumulate(acc02,a0,b2,acc02);
            simdgroup_multiply_accumulate(acc03,a0,b3,acc03);
            simdgroup_multiply_accumulate(acc10,a1,b0,acc10);
            simdgroup_multiply_accumulate(acc11,a1,b1,acc11);
            simdgroup_multiply_accumulate(acc12,a1,b2,acc12);
            simdgroup_multiply_accumulate(acc13,a1,b3,acc13);
            simdgroup_multiply_accumulate(acc20,a2,b0,acc20);
            simdgroup_multiply_accumulate(acc21,a2,b1,acc21);
            simdgroup_multiply_accumulate(acc22,a2,b2,acc22);
            simdgroup_multiply_accumulate(acc23,a2,b3,acc23);
            simdgroup_multiply_accumulate(acc30,a3,b0,acc30);
            simdgroup_multiply_accumulate(acc31,a3,b1,acc31);
            simdgroup_multiply_accumulate(acc32,a3,b2,acc32);
            simdgroup_multiply_accumulate(acc33,a3,b3,acc33); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint ssg = 0; ssg < 4; ssg++) {
        if (sg == ssg) {
            simdgroup_store(acc00,&Ytg[ 0][ 0],32); simdgroup_store(acc01,&Ytg[ 0][ 8],32);
            simdgroup_store(acc02,&Ytg[ 0][16],32); simdgroup_store(acc03,&Ytg[ 0][24],32);
            simdgroup_store(acc10,&Ytg[ 8][ 0],32); simdgroup_store(acc11,&Ytg[ 8][ 8],32);
            simdgroup_store(acc12,&Ytg[ 8][16],32); simdgroup_store(acc13,&Ytg[ 8][24],32);
            simdgroup_store(acc20,&Ytg[16][ 0],32); simdgroup_store(acc21,&Ytg[16][ 8],32);
            simdgroup_store(acc22,&Ytg[16][16],32); simdgroup_store(acc23,&Ytg[16][24],32);
            simdgroup_store(acc30,&Ytg[24][ 0],32); simdgroup_store(acc31,&Ytg[24][ 8],32);
            simdgroup_store(acc32,&Ytg[24][16],32); simdgroup_store(acc33,&Ytg[24][24],32); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint smb=(ssg/2)*32; uint snb=(ssg%2)*32;
        for (uint i = tid; i < 1024; i += THREADS) {
            uint lm=i/32; uint ln=i%32;
            uint gm=m0+smb+lm; uint gn=n0+snb+ln;
            if(gm<M&&gn<N){Y[gm*N+gn]=Ytg[lm][ln];} }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
#endif
