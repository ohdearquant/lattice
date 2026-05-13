//! Direct Metal compute shader forward pass for Qwen3-Embedding-0.6B.
//!
//! Bypasses wgpu abstraction for lower per-dispatch overhead on Apple Silicon.
//! All buffers use `StorageModeShared` for zero-copy unified memory access.
//! The entire 28-layer forward pass is encoded into a single `MTLCommandBuffer`.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod inner {
    use crate::model::qwen::QwenConfig;
    use crate::weights::QwenWeights;
    use metal::*;

    // ---------------------------------------------------------------------------
    // MSL Compute Shaders
    // ---------------------------------------------------------------------------

    const MSL_TEMPLATE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ===== Tiled GEMM: C[M,N] = A[M,K] @ B[N,K]^T =====
// B stored row-major as [N,K], accessed transposed.
kernel void matmul_bt(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]])
{
    constexpr uint TILE = 16;
    threadgroup float tA[TILE][TILE];
    threadgroup float tB[TILE][TILE];

    uint row = gid.y;
    uint col = gid.x;
    float sum = 0.0f;

    for (uint t = 0; t < (K + TILE - 1) / TILE; t++) {
        uint ak = t * TILE + lid.x;
        uint bk = t * TILE + lid.y;

        tA[lid.y][lid.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        tB[lid.y][lid.x] = (col < N && bk < K) ? B[col * K + bk] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            sum += tA[lid.y][k] * tB[k][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ===== RMS Norm: row-wise normalization with gamma =====
// x[row_len] = x[row_len] * gamma[row_len] / sqrt(mean(x^2) + eps)
// One threadgroup per row, reduction within threadgroup.
kernel void rms_norm(
    device float* x           [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    constant uint& row_len    [[buffer(2)]],
    constant uint& num_rows   [[buffer(3)]],
    constant float& eps       [[buffer(4)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]])
{
    if (gid >= num_rows) return;
    constexpr uint RMS_WG = 256;
    uint base = gid * row_len;

    // Each thread accumulates sum-of-squares for its strided elements.
    threadgroup float shared[RMS_WG];
    float local_sum = 0.0f;
    for (uint i = lid; i < row_len; i += tgs) {
        float v = x[base + i];
        local_sum += v * v;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction.
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared[0] / float(row_len) + eps);

    // Scale each element.
    for (uint i = lid; i < row_len; i += tgs) {
        x[base + i] = x[base + i] * rms * gamma[i];
    }
}

// ===== RoPE: rotary position embedding =====
// Operates on x[total_heads, head_dim] at a given position offset.
// cos/sin tables: [max_seq_len, half_dim].
kernel void rope(
    device float* x              [[buffer(0)]],
    device const float* cos_tab  [[buffer(1)]],
    device const float* sin_tab  [[buffer(2)]],
    constant uint& seq_len       [[buffer(3)]],
    constant uint& num_heads     [[buffer(4)]],
    constant uint& head_dim      [[buffer(5)]],
    constant uint& stride        [[buffer(6)]],  // row stride in x (q_dim or kv_dim)
    uint gid [[thread_position_in_grid]])
{
    uint half_dim = head_dim / 2;
    uint total_pairs = seq_len * num_heads * half_dim;
    if (gid >= total_pairs) return;

    // Decompose linear index into (pos, head, pair).
    uint pair = gid % half_dim;
    uint tmp = gid / half_dim;
    uint head = tmp % num_heads;
    uint pos = tmp / num_heads;

    uint base = pos * stride + head * head_dim;
    uint cs_base = pos * half_dim;

    float cos_val = cos_tab[cs_base + pair];
    float sin_val = sin_tab[cs_base + pair];

    float x0 = x[base + pair];
    float x1 = x[base + half_dim + pair];
    x[base + pair]            = x0 * cos_val - x1 * sin_val;
    x[base + half_dim + pair] = x0 * sin_val + x1 * cos_val;
}

// ===== Attention scores: Q @ K^T per head with GQA =====
// Q: [seq, q_dim], K: [seq, kv_dim]
// out: [num_heads, seq, seq]
// Applies scale and causal mask in-place.
kernel void attn_scores(
    device const float* Q    [[buffer(0)]],
    device const float* K    [[buffer(1)]],
    device float* out        [[buffer(2)]],
    constant uint& seq_len   [[buffer(3)]],
    constant uint& head_dim  [[buffer(4)]],
    constant uint& num_heads [[buffer(5)]],
    constant uint& num_kv_heads [[buffer(6)]],
    constant uint& q_dim     [[buffer(7)]],
    constant uint& kv_dim    [[buffer(8)]],
    constant float& scale    [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = num_heads * seq_len * seq_len;
    if (gid >= total) return;

    uint kj = gid % seq_len;
    uint tmp = gid / seq_len;
    uint qi = tmp % seq_len;
    uint h = tmp / seq_len;

    // Causal mask: future positions get -inf.
    if (kj > qi) {
        out[gid] = -1e9f;
        return;
    }

    uint groups = num_heads / num_kv_heads;
    uint kv_h = h / groups;

    float dot = 0.0f;
    uint q_base = qi * q_dim + h * head_dim;
    uint k_base = kj * kv_dim + kv_h * head_dim;
    for (uint d = 0; d < head_dim; d++) {
        dot += Q[q_base + d] * K[k_base + d];
    }

    out[gid] = dot * scale;
}

// ===== Softmax over last dimension =====
// in/out: [num_rows, row_len]
// One threadgroup per row.
kernel void attn_softmax(
    device float* data       [[buffer(0)]],
    constant uint& row_len   [[buffer(1)]],
    constant uint& num_rows  [[buffer(2)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]])
{
    if (gid >= num_rows) return;
    constexpr uint SOFTMAX_WG = 256;
    uint base = gid * row_len;

    // Find max.
    threadgroup float shared[SOFTMAX_WG];
    float local_max = -1e30f;
    for (uint i = lid; i < row_len; i += tgs) {
        local_max = max(local_max, data[base + i]);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];

    // Exp and sum.
    float local_sum = 0.0f;
    for (uint i = lid; i < row_len; i += tgs) {
        float e = exp(data[base + i] - max_val);
        data[base + i] = e;
        local_sum += e;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_val = shared[0];

    // Normalize.
    float inv_sum = (sum_val > 0.0f) ? (1.0f / sum_val) : 0.0f;
    for (uint i = lid; i < row_len; i += tgs) {
        data[base + i] *= inv_sum;
    }
}

// ===== Attention context: scores @ V per head =====
// scores: [num_heads, seq, seq], V: [seq, kv_dim]
// out: [seq, q_dim]
kernel void attn_context(
    device const float* scores  [[buffer(0)]],
    device const float* V       [[buffer(1)]],
    device float* out           [[buffer(2)]],
    constant uint& seq_len      [[buffer(3)]],
    constant uint& head_dim     [[buffer(4)]],
    constant uint& num_heads    [[buffer(5)]],
    constant uint& num_kv_heads [[buffer(6)]],
    constant uint& q_dim        [[buffer(7)]],
    constant uint& kv_dim       [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = num_heads * seq_len * head_dim;
    if (gid >= total) return;

    uint d = gid % head_dim;
    uint tmp = gid / head_dim;
    uint pos = tmp % seq_len;
    uint h = tmp / seq_len;

    uint groups = num_heads / num_kv_heads;
    uint kv_h = h / groups;

    float sum = 0.0f;
    uint score_base = h * seq_len * seq_len + pos * seq_len;
    for (uint j = 0; j < seq_len; j++) {
        sum += scores[score_base + j] * V[j * kv_dim + kv_h * head_dim + d];
    }

    out[pos * q_dim + h * head_dim + d] = sum;
}

// ===== Rectangular GEMM kernels for small-M projections =====
// Specialized for Qwen3 shapes: small M (seq_len), large N and K.
// A in registers, only B in threadgroup memory -> eliminates tA staging.
// float4 loads throughout for better memory throughput.

// NOTE: BLOCK_COLS, K_STEP, K_STEP4 defined as function-local constexpr
// inside each GEMM kernel to guarantee compile-time constant evaluation.

// Variant A: 1 row x 8 cols per thread. Best for M <= 8 (SHORT text).
kernel void gemm_rega1x8_r8(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid             [[thread_position_in_grid]],
    ushort2 tid           [[thread_position_in_threadgroup]])
{
    constexpr ushort BLOCK_COLS = 64, K_STEP = 32, K_STEP4 = 8;
    constexpr ushort COLS_PER_THREAD = 8;
    constexpr ushort TG_X = BLOCK_COLS / COLS_PER_THREAD;
    const uint row = gid.y;
    const uint col_base = gid.x * COLS_PER_THREAD;
    const uint block_col_base = (gid.x - uint(tid.x)) * COLS_PER_THREAD;
    const uint K4 = K >> 2;
    device const float4* A4 = (device const float4*)A;
    device const float4* B4 = (device const float4*)B;
    threadgroup float4 tB[BLOCK_COLS][K_STEP4];
    float acc[COLS_PER_THREAD];
    for (ushort j = 0; j < COLS_PER_THREAD; ++j) { acc[j] = 0.0f; }
    for (uint k0 = 0; k0 < K; k0 += K_STEP) {
        if (tid.y == 0) {
            for (uint c = tid.x; c < BLOCK_COLS; c += TG_X) {
                const uint gcol = block_col_base + c;
                for (uint kk4 = 0; kk4 < K_STEP4; ++kk4) {
                    const uint gk4 = (k0 >> 2) + kk4;
                    tB[c][kk4] = (gcol < N && gk4 < K4) ? B4[gcol * K4 + gk4] : float4(0.0f);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < M) {
            const device float4* a_ptr = A4 + row * K4 + (k0 >> 2);
            for (uint kk4 = 0; kk4 < K_STEP4; ++kk4) {
                const float4 av = a_ptr[kk4];
                const uint c0 = uint(tid.x) * COLS_PER_THREAD;
                acc[0] += dot(av, tB[c0 + 0][kk4]);
                acc[1] += dot(av, tB[c0 + 1][kk4]);
                acc[2] += dot(av, tB[c0 + 2][kk4]);
                acc[3] += dot(av, tB[c0 + 3][kk4]);
                acc[4] += dot(av, tB[c0 + 4][kk4]);
                acc[5] += dot(av, tB[c0 + 5][kk4]);
                acc[6] += dot(av, tB[c0 + 6][kk4]);
                acc[7] += dot(av, tB[c0 + 7][kk4]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M) {
        for (ushort j = 0; j < COLS_PER_THREAD; ++j) {
            const uint col = col_base + j;
            if (col < N) { C[row * N + col] = acc[j]; }
        }
    }
}

// Variant B: 2 rows x 4 cols per thread. Best for M > 8 (MEDIUM/LONG text).
// r16 variant: 16-row threadgroup (good occupancy for M <= 128).
kernel void gemm_rega2x4_r16(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid             [[thread_position_in_grid]],
    ushort2 tid           [[thread_position_in_threadgroup]])
{
    constexpr ushort BLOCK_COLS = 64, K_STEP = 32, K_STEP4 = 8;
    constexpr ushort COLS_PER_THREAD = 4;
    constexpr ushort TG_X = BLOCK_COLS / COLS_PER_THREAD;
    const uint row0 = gid.y * 2;
    const uint row1 = row0 + 1;
    const uint col_base = gid.x * COLS_PER_THREAD;
    const uint block_col_base = (gid.x - uint(tid.x)) * COLS_PER_THREAD;
    const uint K4 = K >> 2;
    device const float4* A4 = (device const float4*)A;
    device const float4* B4 = (device const float4*)B;
    threadgroup float4 tB[BLOCK_COLS][K_STEP4];
    float acc0[COLS_PER_THREAD];
    float acc1[COLS_PER_THREAD];
    for (ushort j = 0; j < COLS_PER_THREAD; ++j) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
    for (uint k0 = 0; k0 < K; k0 += K_STEP) {
        if (tid.y == 0) {
            for (uint c = tid.x; c < BLOCK_COLS; c += TG_X) {
                const uint gcol = block_col_base + c;
                for (uint kk4 = 0; kk4 < K_STEP4; ++kk4) {
                    const uint gk4 = (k0 >> 2) + kk4;
                    tB[c][kk4] = (gcol < N && gk4 < K4) ? B4[gcol * K4 + gk4] : float4(0.0f);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool valid0 = row0 < M;
        const bool valid1 = row1 < M;
        const device float4* a0_ptr = valid0 ? (A4 + row0 * K4 + (k0 >> 2)) : nullptr;
        const device float4* a1_ptr = valid1 ? (A4 + row1 * K4 + (k0 >> 2)) : nullptr;
        for (uint kk4 = 0; kk4 < K_STEP4; ++kk4) {
            const uint c0 = uint(tid.x) * COLS_PER_THREAD;
            const float4 b0 = tB[c0 + 0][kk4];
            const float4 b1 = tB[c0 + 1][kk4];
            const float4 b2 = tB[c0 + 2][kk4];
            const float4 b3 = tB[c0 + 3][kk4];
            if (valid0) {
                const float4 av0 = a0_ptr[kk4];
                acc0[0] += dot(av0, b0);
                acc0[1] += dot(av0, b1);
                acc0[2] += dot(av0, b2);
                acc0[3] += dot(av0, b3);
            }
            if (valid1) {
                const float4 av1 = a1_ptr[kk4];
                acc1[0] += dot(av1, b0);
                acc1[1] += dot(av1, b1);
                acc1[2] += dot(av1, b2);
                acc1[3] += dot(av1, b3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 < M) {
        for (ushort j = 0; j < COLS_PER_THREAD; ++j) {
            const uint col = col_base + j;
            if (col < N) { C[row0 * N + col] = acc0[j]; }
        }
    }
    if (row1 < M) {
        for (ushort j = 0; j < COLS_PER_THREAD; ++j) {
            const uint col = col_base + j;
            if (col < N) { C[row1 * N + col] = acc1[j]; }
        }
    }
}

// r32 variant: 32-row threadgroup (better B reuse for LONG text / large N).
kernel void gemm_rega2x4_r32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid             [[thread_position_in_grid]],
    ushort2 tid           [[thread_position_in_threadgroup]])
{
    constexpr ushort BLOCK_COLS = 64, K_STEP = 32, K_STEP4 = 8;
    constexpr ushort COLS_PER_THREAD = 4;
    constexpr ushort TG_X = BLOCK_COLS / COLS_PER_THREAD;
    const uint row0 = gid.y * 2;
    const uint row1 = row0 + 1;
    const uint col_base = gid.x * COLS_PER_THREAD;
    const uint block_col_base = (gid.x - uint(tid.x)) * COLS_PER_THREAD;
    const uint K4 = K >> 2;
    device const float4* A4 = (device const float4*)A;
    device const float4* B4 = (device const float4*)B;
    threadgroup float4 tB[BLOCK_COLS][K_STEP4];
    float acc0[COLS_PER_THREAD];
    float acc1[COLS_PER_THREAD];
    for (ushort j = 0; j < COLS_PER_THREAD; ++j) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
    for (uint k0 = 0; k0 < K; k0 += K_STEP) {
        if (tid.y == 0) {
            for (uint c = tid.x; c < BLOCK_COLS; c += TG_X) {
                const uint gcol = block_col_base + c;
                for (uint kk4 = 0; kk4 < K_STEP4; ++kk4) {
                    const uint gk4 = (k0 >> 2) + kk4;
                    tB[c][kk4] = (gcol < N && gk4 < K4) ? B4[gcol * K4 + gk4] : float4(0.0f);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool valid0 = row0 < M;
        const bool valid1 = row1 < M;
        const device float4* a0_ptr = valid0 ? (A4 + row0 * K4 + (k0 >> 2)) : nullptr;
        const device float4* a1_ptr = valid1 ? (A4 + row1 * K4 + (k0 >> 2)) : nullptr;
        for (uint kk4 = 0; kk4 < K_STEP4; ++kk4) {
            const uint c0 = uint(tid.x) * COLS_PER_THREAD;
            const float4 b0 = tB[c0 + 0][kk4];
            const float4 b1 = tB[c0 + 1][kk4];
            const float4 b2 = tB[c0 + 2][kk4];
            const float4 b3 = tB[c0 + 3][kk4];
            if (valid0) {
                const float4 av0 = a0_ptr[kk4];
                acc0[0] += dot(av0, b0);
                acc0[1] += dot(av0, b1);
                acc0[2] += dot(av0, b2);
                acc0[3] += dot(av0, b3);
            }
            if (valid1) {
                const float4 av1 = a1_ptr[kk4];
                acc1[0] += dot(av1, b0);
                acc1[1] += dot(av1, b1);
                acc1[2] += dot(av1, b2);
                acc1[3] += dot(av1, b3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 < M) {
        for (ushort j = 0; j < COLS_PER_THREAD; ++j) {
            const uint col = col_base + j;
            if (col < N) { C[row0 * N + col] = acc0[j]; }
        }
    }
    if (row1 < M) {
        for (ushort j = 0; j < COLS_PER_THREAD; ++j) {
            const uint col = col_base + j;
            if (col < N) { C[row1 * N + col] = acc1[j]; }
        }
    }
}

// ===== Fused Attention: Q@K^T + causal softmax + scores@V in one kernel =====
// Eliminates the global scores buffer. Online softmax in registers.
// FA_HEAD_DIM and FA_GQA_GROUPS are injected from model config at compile time.
// Q/O in registers, K/V tiles in threadgroup memory.

struct FusedAttentionParams {
    uint seq_len;
    uint q_dim4;        // q_dim / 4
    uint kv_dim4;       // kv_dim / 4
    uint num_kv_heads;
    float scale;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

kernel void fused_attention(
    device const float4* Q4 [[buffer(0)]],
    device const float4* K4 [[buffer(1)]],
    device const float4* V4 [[buffer(2)]],
    device float4* O4       [[buffer(3)]],
    constant FusedAttentionParams& p [[buffer(4)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr uint FA_HEAD_DIM     = __FA_HEAD_DIM__u;  // injected from model config
    constexpr uint FA_HEAD_DIM4    = FA_HEAD_DIM / 4u;
    constexpr uint FA_GQA_GROUPS   = __FA_GQA_GROUPS__u;
    constexpr uint FA_TILE_Q       = 4u;
    constexpr uint FA_TILE_K       = 16u;
    constexpr uint FA_SIMD_WIDTH   = 32u;
    constexpr uint FA_ROWS_PER_TG  = FA_TILE_Q * FA_GQA_GROUPS;
    constexpr uint FA_THREADS_PER_TG = FA_ROWS_PER_TG * FA_SIMD_WIDTH;

    const uint kv_head = tgp.x;
    const uint q_block = tgp.y;

    if (kv_head >= p.num_kv_heads) return;

    const uint local_row    = tid / FA_SIMD_WIDTH;
    const uint q_head_local = local_row / FA_TILE_Q;
    const uint q_row_local  = local_row % FA_TILE_Q;

    const uint q_block_start = q_block * FA_TILE_Q;
    const uint q_block_end   = min(p.seq_len, q_block_start + FA_TILE_Q);
    const uint qi            = q_block_start + q_row_local;
    const bool row_active    = (qi < p.seq_len);

    const uint q_head = kv_head * FA_GQA_GROUPS + q_head_local;

    threadgroup float4 K_tile[FA_TILE_K][FA_HEAD_DIM4];
    threadgroup float4 V_tile[FA_TILE_K][FA_HEAD_DIM4];

    float4 q_frag = float4(0.0f);
    float4 o_frag = float4(0.0f);
    float m_i = -INFINITY;
    float l_i = 0.0f;

    if (row_active) {
        const uint q_base4 = qi * p.q_dim4 + q_head * FA_HEAD_DIM4 + lane;
        q_frag = Q4[q_base4];
    }

    for (uint k_start = 0; k_start < p.seq_len; k_start += FA_TILE_K) {
        if (k_start >= q_block_end) break;

        const uint tile_len_global = min(FA_TILE_K, p.seq_len - k_start);

        for (uint idx = tid; idx < tile_len_global * FA_HEAD_DIM4; idx += FA_THREADS_PER_TG) {
            const uint tk = idx / FA_HEAD_DIM4;
            const uint d4 = idx % FA_HEAD_DIM4;
            const uint kj = k_start + tk;
            const uint kv_base4 = kj * p.kv_dim4 + kv_head * FA_HEAD_DIM4 + d4;
            K_tile[tk][d4] = K4[kv_base4];
            V_tile[tk][d4] = V4[kv_base4];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint row_tile_len = 0u;
        if (row_active) {
            const uint last_k_exclusive = qi + 1u;
            if (last_k_exclusive > k_start) {
                row_tile_len = min(FA_TILE_K, last_k_exclusive - k_start);
            }
        }

        if (row_tile_len > 0u) {
            float scores[FA_TILE_K];
            float tile_max = -INFINITY;

            for (uint tk = 0; tk < row_tile_len; ++tk) {
                const float partial = dot(q_frag, K_tile[tk][lane]);
                const float s = simd_sum(partial) * p.scale;
                scores[tk] = s;
                tile_max = max(tile_max, s);
            }

            const float m_new = max(m_i, tile_max);
            const float alpha = exp(m_i - m_new);
            float l_new = l_i * alpha;
            float4 o_update = float4(0.0f);

            for (uint tk = 0; tk < row_tile_len; ++tk) {
                const float p_ij = exp(scores[tk] - m_new);
                l_new += p_ij;
                o_update += p_ij * V_tile[tk][lane];
            }

            o_frag = o_frag * alpha + o_update;
            l_i = l_new;
            m_i = m_new;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_active) {
        const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        const uint out_base4 = qi * p.q_dim4 + q_head * FA_HEAD_DIM4 + lane;
        O4[out_base4] = o_frag * inv_l;
    }
}

// ===== Fused SiLU * up elementwise =====
// gate = silu(gate) * up
kernel void silu_mul(
    device float* gate       [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float g = gate[gid];
    float s = g / (1.0f + exp(-g));
    gate[gid] = s * up[gid];
}

// ===== Copy: dst = src =====
kernel void copy_buf(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& count    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    dst[gid] = src[gid];
}

// ===== Add: dst += src =====
kernel void add_buf(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& count    [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    dst[gid] += src[gid];
}

// ===== BF16 utilities =====
inline float bf16_to_float(ushort x) {
    return as_type<float>(uint(x) << 16);
}

inline float silu_act(float x) {
    return x / (1.0f + exp(-x));
}

// ===== BF16 Rectangular GEMM: C[M,N] = A[M,K](f32) @ B[N,K](bf16)^T =====
// B stored as ushort (BF16), decoded in threadgroup memory staging.

// Inline helper: decode ushort4 (BF16) to float4.
inline float4 bf16x4_to_float4(ushort4 v) {
    return float4(bf16_to_float(v.x), bf16_to_float(v.y),
                  bf16_to_float(v.z), bf16_to_float(v.w));
}

// Variant A bf16: 1 row x 8 cols per thread. Best for M <= 8 (SHORT).
kernel void gemm_rega1x8_r8_bf16(
    device const float* A    [[buffer(0)]],
    device const ushort* B16 [[buffer(1)]],
    device float* C          [[buffer(2)]],
    constant uint& M         [[buffer(3)]],
    constant uint& N         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    uint2 gid                [[thread_position_in_grid]],
    ushort2 tid              [[thread_position_in_threadgroup]])
{
    constexpr ushort BLOCK_COLS = 64, K_STEP = 32, K_STEP4 = 8;
    constexpr ushort COLS_PER_THREAD = 8;
    constexpr ushort TG_X = BLOCK_COLS / COLS_PER_THREAD;
    const uint row = gid.y;
    const uint col_base = gid.x * COLS_PER_THREAD;
    const uint block_col_base = (gid.x - uint(tid.x)) * COLS_PER_THREAD;
    const uint K4 = K >> 2;
    device const float4* A4 = (device const float4*)A;
    device const ushort4* B4_u = (device const ushort4*)B16;
    threadgroup float4 tB[BLOCK_COLS][K_STEP4];
    float acc[COLS_PER_THREAD];
    for (ushort j = 0; j < COLS_PER_THREAD; ++j) { acc[j] = 0.0f; }
    for (uint k0 = 0; k0 < K; k0 += K_STEP) {
        if (tid.y == 0) {
            for (uint c = tid.x; c < BLOCK_COLS; c += TG_X) {
                const uint gcol = block_col_base + c;
                for (uint kk4 = 0; kk4 < K_STEP4; ++kk4) {
                    const uint gk4 = (k0 >> 2) + kk4;
                    tB[c][kk4] = (gcol < N && gk4 < K4)
                        ? bf16x4_to_float4(B4_u[gcol * K4 + gk4])
                        : float4(0.0f);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < M) {
            const device float4* a_ptr = A4 + row * K4 + (k0 >> 2);
            for (uint kk4 = 0; kk4 < K_STEP4; ++kk4) {
                const float4 av = a_ptr[kk4];
                const uint c0 = uint(tid.x) * COLS_PER_THREAD;
                acc[0] += dot(av, tB[c0 + 0][kk4]);
                acc[1] += dot(av, tB[c0 + 1][kk4]);
                acc[2] += dot(av, tB[c0 + 2][kk4]);
                acc[3] += dot(av, tB[c0 + 3][kk4]);
                acc[4] += dot(av, tB[c0 + 4][kk4]);
                acc[5] += dot(av, tB[c0 + 5][kk4]);
                acc[6] += dot(av, tB[c0 + 6][kk4]);
                acc[7] += dot(av, tB[c0 + 7][kk4]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M) {
        for (ushort j = 0; j < COLS_PER_THREAD; ++j) {
            const uint col = col_base + j;
            if (col < N) { C[row * N + col] = acc[j]; }
        }
    }
}

// Variant B bf16 r16: 2 rows x 4 cols per thread. Best for M > 8.
kernel void gemm_rega2x4_r16_bf16(
    device const float* A    [[buffer(0)]],
    device const ushort* B16 [[buffer(1)]],
    device float* C          [[buffer(2)]],
    constant uint& M         [[buffer(3)]],
    constant uint& N         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    uint2 gid                [[thread_position_in_grid]],
    ushort2 tid              [[thread_position_in_threadgroup]])
{
    constexpr ushort BLOCK_COLS = 64, K_STEP = 32, K_STEP4 = 8;
    constexpr ushort COLS_PER_THREAD = 4;
    constexpr ushort TG_X = BLOCK_COLS / COLS_PER_THREAD;
    const uint row0 = gid.y * 2;
    const uint row1 = row0 + 1;
    const uint col_base = gid.x * COLS_PER_THREAD;
    const uint block_col_base = (gid.x - uint(tid.x)) * COLS_PER_THREAD;
    const uint K4 = K >> 2;
    device const float4* A4 = (device const float4*)A;
    device const ushort4* B4_u = (device const ushort4*)B16;
    threadgroup float4 tB[BLOCK_COLS][K_STEP4];
    float acc0[COLS_PER_THREAD];
    float acc1[COLS_PER_THREAD];
    for (ushort j = 0; j < COLS_PER_THREAD; ++j) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
    for (uint k0 = 0; k0 < K; k0 += K_STEP) {
        if (tid.y == 0) {
            for (uint c = tid.x; c < BLOCK_COLS; c += TG_X) {
                const uint gcol = block_col_base + c;
                for (uint kk4 = 0; kk4 < K_STEP4; ++kk4) {
                    const uint gk4 = (k0 >> 2) + kk4;
                    tB[c][kk4] = (gcol < N && gk4 < K4)
                        ? bf16x4_to_float4(B4_u[gcol * K4 + gk4])
                        : float4(0.0f);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool valid0 = row0 < M;
        const bool valid1 = row1 < M;
        const device float4* a0_ptr = valid0 ? (A4 + row0 * K4 + (k0 >> 2)) : nullptr;
        const device float4* a1_ptr = valid1 ? (A4 + row1 * K4 + (k0 >> 2)) : nullptr;
        for (uint kk4 = 0; kk4 < K_STEP4; ++kk4) {
            const uint c0 = uint(tid.x) * COLS_PER_THREAD;
            const float4 b0 = tB[c0 + 0][kk4];
            const float4 b1 = tB[c0 + 1][kk4];
            const float4 b2 = tB[c0 + 2][kk4];
            const float4 b3 = tB[c0 + 3][kk4];
            if (valid0) {
                const float4 av0 = a0_ptr[kk4];
                acc0[0] += dot(av0, b0); acc0[1] += dot(av0, b1);
                acc0[2] += dot(av0, b2); acc0[3] += dot(av0, b3);
            }
            if (valid1) {
                const float4 av1 = a1_ptr[kk4];
                acc1[0] += dot(av1, b0); acc1[1] += dot(av1, b1);
                acc1[2] += dot(av1, b2); acc1[3] += dot(av1, b3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 < M) {
        for (ushort j = 0; j < COLS_PER_THREAD; ++j) {
            const uint col = col_base + j;
            if (col < N) { C[row0 * N + col] = acc0[j]; }
        }
    }
    if (row1 < M) {
        for (ushort j = 0; j < COLS_PER_THREAD; ++j) {
            const uint col = col_base + j;
            if (col < N) { C[row1 * N + col] = acc1[j]; }
        }
    }
}

// Variant B bf16 r32: 2 rows x 4 cols, 32-row TG for LONG.
kernel void gemm_rega2x4_r32_bf16(
    device const float* A    [[buffer(0)]],
    device const ushort* B16 [[buffer(1)]],
    device float* C          [[buffer(2)]],
    constant uint& M         [[buffer(3)]],
    constant uint& N         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    uint2 gid                [[thread_position_in_grid]],
    ushort2 tid              [[thread_position_in_threadgroup]])
{
    constexpr ushort BLOCK_COLS = 64, K_STEP = 32, K_STEP4 = 8;
    constexpr ushort COLS_PER_THREAD = 4;
    constexpr ushort TG_X = BLOCK_COLS / COLS_PER_THREAD;
    const uint row0 = gid.y * 2;
    const uint row1 = row0 + 1;
    const uint col_base = gid.x * COLS_PER_THREAD;
    const uint block_col_base = (gid.x - uint(tid.x)) * COLS_PER_THREAD;
    const uint K4 = K >> 2;
    device const float4* A4 = (device const float4*)A;
    device const ushort4* B4_u = (device const ushort4*)B16;
    threadgroup float4 tB[BLOCK_COLS][K_STEP4];
    float acc0[COLS_PER_THREAD];
    float acc1[COLS_PER_THREAD];
    for (ushort j = 0; j < COLS_PER_THREAD; ++j) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
    for (uint k0 = 0; k0 < K; k0 += K_STEP) {
        if (tid.y == 0) {
            for (uint c = tid.x; c < BLOCK_COLS; c += TG_X) {
                const uint gcol = block_col_base + c;
                for (uint kk4 = 0; kk4 < K_STEP4; ++kk4) {
                    const uint gk4 = (k0 >> 2) + kk4;
                    tB[c][kk4] = (gcol < N && gk4 < K4)
                        ? bf16x4_to_float4(B4_u[gcol * K4 + gk4])
                        : float4(0.0f);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool valid0 = row0 < M;
        const bool valid1 = row1 < M;
        const device float4* a0_ptr = valid0 ? (A4 + row0 * K4 + (k0 >> 2)) : nullptr;
        const device float4* a1_ptr = valid1 ? (A4 + row1 * K4 + (k0 >> 2)) : nullptr;
        for (uint kk4 = 0; kk4 < K_STEP4; ++kk4) {
            const uint c0 = uint(tid.x) * COLS_PER_THREAD;
            const float4 b0 = tB[c0 + 0][kk4];
            const float4 b1 = tB[c0 + 1][kk4];
            const float4 b2 = tB[c0 + 2][kk4];
            const float4 b3 = tB[c0 + 3][kk4];
            if (valid0) {
                const float4 av0 = a0_ptr[kk4];
                acc0[0] += dot(av0, b0); acc0[1] += dot(av0, b1);
                acc0[2] += dot(av0, b2); acc0[3] += dot(av0, b3);
            }
            if (valid1) {
                const float4 av1 = a1_ptr[kk4];
                acc1[0] += dot(av1, b0); acc1[1] += dot(av1, b1);
                acc1[2] += dot(av1, b2); acc1[3] += dot(av1, b3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row0 < M) {
        for (ushort j = 0; j < COLS_PER_THREAD; ++j) {
            const uint col = col_base + j;
            if (col < N) { C[row0 * N + col] = acc0[j]; }
        }
    }
    if (row1 < M) {
        for (ushort j = 0; j < COLS_PER_THREAD; ++j) {
            const uint col = col_base + j;
            if (col < N) { C[row1 * N + col] = acc1[j]; }
        }
    }
}

// ===== Fused QK Norm + RoPE (Fusion C from R3-08) =====
// One threadgroup per (token, head, family). family 0=Q, 1=K.
// Replaces: Q-norm + K-norm + RoPE-Q + RoPE-K (4 dispatches → 1).

#define FUSED_C_HEAD_DIM __FUSED_C_HEAD_DIM__u  // injected from model config
#define FUSED_C_HALF_DIM  __FUSED_C_HALF_DIM__u
#define FUSED_C_THREADS   __FUSED_C_THREADS__u

struct FusedQkNormRopeParams {
    uint seq_len;
    uint q_heads;      // 16
    uint k_heads;      // 8
    uint q_stride;     // 2048
    uint k_stride;     // 1024
    float eps;
};

kernel void fused_qk_norm_rope(
    device float*        q                   [[buffer(0)]],
    device float*        k                   [[buffer(1)]],
    device const float*  q_norm_weight       [[buffer(2)]],
    device const float*  k_norm_weight       [[buffer(3)]],
    device const float*  rope_cos            [[buffer(4)]],
    device const float*  rope_sin            [[buffer(5)]],
    constant FusedQkNormRopeParams& p        [[buffer(6)]],
    uint tid_local                           [[thread_index_in_threadgroup]],
    uint3 tg_pos                             [[threadgroup_position_in_grid]])
{
    const uint pos    = tg_pos.x;
    const uint head   = tg_pos.y;
    const uint family = tg_pos.z;

    if (pos >= p.seq_len) return;
    const bool is_q = (family == 0u);
    const uint head_count = is_q ? p.q_heads : p.k_heads;
    if (head >= head_count) return;

    threadgroup float tg_reduce[FUSED_C_THREADS];

    device float* vec = is_q ? q : k;
    device const float* norm_weight = is_q ? q_norm_weight : k_norm_weight;
    const uint stride = is_q ? p.q_stride : p.k_stride;

    const uint base = pos * stride + head * FUSED_C_HEAD_DIM;
    const uint lo   = tid_local;
    const uint hi   = tid_local + FUSED_C_HALF_DIM;

    const float x_lo = vec[base + lo];
    const float x_hi = vec[base + hi];

    tg_reduce[tid_local] = x_lo * x_lo + x_hi * x_hi;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = FUSED_C_THREADS >> 1; offset > 0; offset >>= 1) {
        if (tid_local < offset) {
            tg_reduce[tid_local] += tg_reduce[tid_local + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt(tg_reduce[0] / float(FUSED_C_HEAD_DIM) + p.eps);

    const float n_lo = x_lo * inv_rms * norm_weight[lo];
    const float n_hi = x_hi * inv_rms * norm_weight[hi];

    const float c = rope_cos[pos * FUSED_C_HALF_DIM + tid_local];
    const float s = rope_sin[pos * FUSED_C_HALF_DIM + tid_local];

    vec[base + lo] = n_lo * c - n_hi * s;
    vec[base + hi] = n_lo * s + n_hi * c;
}
"#;

    // ---------------------------------------------------------------------------
    // Weight and Activation Buffer Structs
    // ---------------------------------------------------------------------------

    struct MetalLayerWeights {
        q_proj_weight: Buffer,
        k_proj_weight: Buffer,
        v_proj_weight: Buffer,
        o_proj_weight: Buffer,
        q_norm_weight: Buffer,
        k_norm_weight: Buffer,
        input_layernorm_weight: Buffer,
        gate_proj_weight: Buffer,
        up_proj_weight: Buffer,
        down_proj_weight: Buffer,
        post_attention_layernorm_weight: Buffer,
    }

    /// Params for fused QK-norm + RoPE kernel.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct FusedQkNormRopeParams {
        seq_len: u32,
        q_heads: u32,
        k_heads: u32,
        q_stride: u32,
        k_stride: u32,
        eps: f32,
    }

    struct MetalActivations {
        hidden: Buffer,
        residual: Buffer,
        q: Buffer,
        k: Buffer,
        v: Buffer,
        attn_out: Buffer,
        gate: Buffer,
        up: Buffer,
        ffn_out: Buffer,
    }

    /// Configuration mirroring QwenConfig but with pre-computed derived values.
    #[derive(Clone)]
    struct Qwen3GpuConfig {
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rms_norm_eps: f32,
        q_dim: usize,
        kv_dim: usize,
        max_seq_len: usize,
        /// Debug-only layer limit (from METAL_LAYERS env var at init time).
        layer_limit: Option<usize>,
    }

    /// Validate model shape for Metal fused kernels and return the GQA group count.
    ///
    /// Enforces structural requirements only — head_dim must be divisible by 4
    /// (for float4 operations) and nonzero, kv_heads nonzero, and q_heads
    /// divisible by kv_heads. The exact values are injected into the MSL source
    /// at compile time via `msl_source_for`.
    fn validate_fused_kernel_shape(config: &QwenConfig) -> Result<usize, String> {
        if config.head_dim == 0 {
            return Err("Metal fused attention requires nonzero head_dim".to_string());
        }
        if config.head_dim % 4 != 0 {
            return Err(format!(
                "Metal fused attention requires head_dim divisible by 4 (for float4 ops); got {}",
                config.head_dim
            ));
        }
        if config.num_key_value_heads == 0 {
            return Err("Metal fused attention requires nonzero num_key_value_heads".to_string());
        }
        if config.num_attention_heads % config.num_key_value_heads != 0 {
            return Err(format!(
                "Metal fused attention requires num_attention_heads divisible by num_key_value_heads; got {}/{}",
                config.num_attention_heads, config.num_key_value_heads
            ));
        }
        let gqa_groups = config.num_attention_heads / config.num_key_value_heads;
        Ok(gqa_groups)
    }

    /// Produce MSL source with model-specific constants substituted in.
    ///
    /// Derived values injected:
    /// - `half_dim = head_dim / 2`  (RoPE rotation pairs, also threadgroup size)
    /// `FA_HEAD_DIM4` is derived from `FA_HEAD_DIM` inside the MSL, not injected separately.
    fn msl_source_for(head_dim: usize, gqa_groups: usize) -> String {
        let half_dim = head_dim / 2;
        let threads = head_dim / 2;
        MSL_TEMPLATE
            .replace("__FA_HEAD_DIM__", &head_dim.to_string())
            .replace("__FA_GQA_GROUPS__", &gqa_groups.to_string())
            .replace("__FUSED_C_HEAD_DIM__", &head_dim.to_string())
            .replace("__FUSED_C_HALF_DIM__", &half_dim.to_string())
            .replace("__FUSED_C_THREADS__", &threads.to_string())
    }

    /// Direct Metal forward pass for Qwen3-Embedding.
    ///
    /// All weight buffers are persistent in unified memory (`StorageModeShared`).
    /// Activation buffers are pre-allocated for `max_seq_len`.
    /// A single `MTLCommandBuffer` encodes all 28 layers per forward call.
    pub struct MetalForwardPass {
        #[allow(dead_code)]
        // TODO(#1958): Metal roadmap — device handle kept alive for GPU lifetime management
        device: Device,
        queue: CommandQueue,
        matmul_pipeline: ComputePipelineState,
        rms_norm_pipeline: ComputePipelineState,
        fused_attention_pipeline: ComputePipelineState,
        fused_qk_norm_rope_pipeline: ComputePipelineState,
        silu_mul_pipeline: ComputePipelineState,
        copy_pipeline: ComputePipelineState,
        add_pipeline: ComputePipelineState,
        layer_weights: Vec<MetalLayerWeights>,
        final_norm_weight: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        activations: MetalActivations,
        config: Qwen3GpuConfig,
    }

    // Helper: create a Metal buffer from a float slice.
    fn make_buffer(device: &Device, data: &[f32], label: &str) -> Buffer {
        let byte_len = std::mem::size_of_val(data) as u64;
        let buf = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        buf.set_label(label);
        buf
    }

    // Helper: create a zero-filled buffer of `num_floats` capacity.
    fn make_zero_buffer(device: &Device, num_floats: usize, label: &str) -> Buffer {
        let byte_len = (num_floats * std::mem::size_of::<f32>()) as u64;
        let buf = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
        buf.set_label(label);
        buf
    }

    // Helper: ceil-divide.
    fn div_ceil(a: u64, b: u64) -> u64 {
        a.div_ceil(b)
    }

    impl MetalForwardPass {
        /// Create from CPU weights. Uploads all weights to unified memory buffers.
        ///
        /// `max_seq_len` caps the pre-allocated activation buffer size.
        /// For embedding workloads, 512 is typically sufficient.
        pub fn new(
            config: &QwenConfig,
            weights: &QwenWeights<'_>,
            max_seq_len: usize,
        ) -> Result<Self, String> {
            let device =
                Device::system_default().ok_or_else(|| "No Metal device found".to_string())?;

            tracing::info!(
                name = device.name(),
                "Metal GPU initialized for Qwen3 forward pass"
            );

            let queue = device.new_command_queue();

            // Validate model shape and extract GQA group count before shader compilation.
            let gqa_groups = validate_fused_kernel_shape(config)?;

            // Compile shaders with model-specific constants substituted in.
            let opts = CompileOptions::new();
            let msl = msl_source_for(config.head_dim, gqa_groups);
            let library = device
                .new_library_with_source(&msl, &opts)
                .map_err(|e| format!("Metal shader compilation failed: {e}"))?;

            let make_pipeline = |name: &str| -> Result<ComputePipelineState, String> {
                let func = library
                    .get_function(name, None)
                    .map_err(|e| format!("function '{name}' not found: {e}"))?;
                device
                    .new_compute_pipeline_state_with_function(&func)
                    .map_err(|e| format!("pipeline for '{name}' failed: {e}"))
            };

            let matmul_pipeline = make_pipeline("matmul_bt")?;
            let rms_norm_pipeline = make_pipeline("rms_norm")?;
            let fused_attention_pipeline = make_pipeline("fused_attention")?;
            let fused_qk_norm_rope_pipeline = make_pipeline("fused_qk_norm_rope")?;
            let silu_mul_pipeline = make_pipeline("silu_mul")?;
            let copy_pipeline = make_pipeline("copy_buf")?;
            let add_pipeline = make_pipeline("add_buf")?;

            let q_dim = config.q_dim();
            let kv_dim = config.kv_dim();
            // Read METAL_LAYERS env var once at init (not per-forward).
            let layer_limit = std::env::var("METAL_LAYERS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok());

            let gpu_config = Qwen3GpuConfig {
                hidden_size: config.hidden_size,
                num_hidden_layers: config.num_hidden_layers,
                num_attention_heads: config.num_attention_heads,
                num_key_value_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                intermediate_size: config.intermediate_size,
                rms_norm_eps: config.rms_norm_eps,
                q_dim,
                kv_dim,
                max_seq_len,
                layer_limit,
            };

            // Upload per-layer weights: norms as f32, projections as BF16.
            let mut layer_weights_vec = Vec::with_capacity(config.num_hidden_layers);
            for (i, lw) in weights.layers.iter().enumerate() {
                layer_weights_vec.push(MetalLayerWeights {
                    // f32 projection weights (fallback)
                    q_proj_weight: make_buffer(
                        &device,
                        lw.q_proj_weight.data,
                        &format!("L{i}.q_proj"),
                    ),
                    k_proj_weight: make_buffer(
                        &device,
                        lw.k_proj_weight.data,
                        &format!("L{i}.k_proj"),
                    ),
                    v_proj_weight: make_buffer(
                        &device,
                        lw.v_proj_weight.data,
                        &format!("L{i}.v_proj"),
                    ),
                    o_proj_weight: make_buffer(
                        &device,
                        lw.o_proj_weight.data,
                        &format!("L{i}.o_proj"),
                    ),
                    gate_proj_weight: make_buffer(
                        &device,
                        lw.gate_proj_weight.data,
                        &format!("L{i}.gate"),
                    ),
                    up_proj_weight: make_buffer(
                        &device,
                        lw.up_proj_weight.data,
                        &format!("L{i}.up"),
                    ),
                    down_proj_weight: make_buffer(
                        &device,
                        lw.down_proj_weight.data,
                        &format!("L{i}.down"),
                    ),
                    // Norm weights f32
                    q_norm_weight: make_buffer(
                        &device,
                        lw.q_norm_weight.data,
                        &format!("L{i}.q_norm"),
                    ),
                    k_norm_weight: make_buffer(
                        &device,
                        lw.k_norm_weight.data,
                        &format!("L{i}.k_norm"),
                    ),
                    input_layernorm_weight: make_buffer(
                        &device,
                        lw.input_layernorm_weight.data,
                        &format!("L{i}.in_norm"),
                    ),
                    post_attention_layernorm_weight: make_buffer(
                        &device,
                        lw.post_attention_layernorm_weight.data,
                        &format!("L{i}.post_norm"),
                    ),
                });
            }

            let final_norm_weight = make_buffer(&device, weights.norm_weight.data, "final_norm");

            // Build and upload RoPE tables.
            let rope_max = max_seq_len.min(config.max_position_embeddings);
            let (cos_data, sin_data) =
                build_rope_flat(config.head_dim, rope_max, config.rope_theta);
            let rope_cos = make_buffer(&device, &cos_data, "rope_cos");
            let rope_sin = make_buffer(&device, &sin_data, "rope_sin");

            // Allocate activation buffers for max_seq_len.
            let s = max_seq_len;
            let activations = MetalActivations {
                hidden: make_zero_buffer(&device, s * config.hidden_size, "act_hidden"),
                residual: make_zero_buffer(&device, s * config.hidden_size, "act_residual"),
                q: make_zero_buffer(&device, s * q_dim, "act_q"),
                k: make_zero_buffer(&device, s * kv_dim, "act_k"),
                v: make_zero_buffer(&device, s * kv_dim, "act_v"),
                attn_out: make_zero_buffer(&device, s * q_dim, "act_attn_out"),
                gate: make_zero_buffer(&device, s * config.intermediate_size, "act_gate"),
                up: make_zero_buffer(&device, s * config.intermediate_size, "act_up"),
                ffn_out: make_zero_buffer(&device, s * config.hidden_size, "act_ffn_out"),
            };

            Ok(Self {
                device,
                queue,
                matmul_pipeline,
                rms_norm_pipeline,
                fused_attention_pipeline,
                fused_qk_norm_rope_pipeline,
                silu_mul_pipeline,
                copy_pipeline,
                add_pipeline,
                layer_weights: layer_weights_vec,
                final_norm_weight,
                rope_cos,
                rope_sin,
                activations,
                config: gpu_config,
            })
        }

        /// Run the forward pass on GPU.
        ///
        /// `hidden_input` is `[seq_len, hidden_size]` — the result of CPU embedding gather.
        /// Returns `[seq_len, hidden_size]` hidden states after all transformer layers + final norm.
        ///
        /// Takes `&mut self` because activation buffers are written via unified memory pointers.
        pub fn forward(
            &mut self,
            hidden_input: &[f32],
            seq_len: usize,
        ) -> Result<Vec<f32>, String> {
            let hidden = self.config.hidden_size;
            if seq_len == 0 || seq_len > self.config.max_seq_len {
                return Err(format!(
                    "seq_len {seq_len} out of range [1, {}]",
                    self.config.max_seq_len
                ));
            }
            if hidden_input.len() != seq_len * hidden {
                return Err(format!(
                    "hidden_input length {} != seq_len({seq_len}) * hidden_size({hidden})",
                    hidden_input.len()
                ));
            }

            let cfg = &self.config;
            let q_dim = cfg.q_dim as u32;
            let kv_dim = cfg.kv_dim as u32;
            let num_heads = cfg.num_attention_heads as u32;
            let num_kv_heads = cfg.num_key_value_heads as u32;
            let hidden_u = cfg.hidden_size as u32;
            let inter_u = cfg.intermediate_size as u32;
            let seq_u = seq_len as u32;
            let eps = cfg.rms_norm_eps;
            let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();

            let hidden_elems = (seq_len * hidden) as u32;
            let inter_elems = (seq_len * cfg.intermediate_size) as u32;

            // SAFETY: `self.activations.hidden` is a Metal buffer with StorageModeShared,
            // giving us a CPU-accessible pointer to GPU-visible unified memory. We have
            // exclusive access through `&mut self`, and the buffer was allocated with at
            // least `max_seq_len * hidden_size` f32s in `new()`.
            unsafe {
                let dst = self.activations.hidden.contents() as *mut f32;
                std::ptr::copy_nonoverlapping(hidden_input.as_ptr(), dst, hidden_input.len());
            }

            let cmd = self.queue.new_command_buffer();

            let layer_limit = cfg.layer_limit.unwrap_or(cfg.num_hidden_layers);

            for layer_idx in 0..layer_limit.min(cfg.num_hidden_layers) {
                let lw = &self.layer_weights[layer_idx];
                let enc = cmd.new_compute_command_encoder();

                // 1. Copy hidden -> residual
                self.dispatch_copy(
                    enc,
                    &self.activations.hidden,
                    &self.activations.residual,
                    hidden_elems,
                );

                // 2. Input LayerNorm (in-place on hidden)
                self.dispatch_rms_norm(
                    enc,
                    &self.activations.hidden,
                    &lw.input_layernorm_weight,
                    hidden_u,
                    seq_u,
                    eps,
                );

                // 3-5. Q/K/V projections
                self.dispatch_matmul(
                    enc,
                    &self.activations.hidden,
                    &lw.q_proj_weight,
                    &self.activations.q,
                    seq_u,
                    q_dim,
                    hidden_u,
                );
                self.dispatch_matmul(
                    enc,
                    &self.activations.hidden,
                    &lw.k_proj_weight,
                    &self.activations.k,
                    seq_u,
                    kv_dim,
                    hidden_u,
                );
                self.dispatch_matmul(
                    enc,
                    &self.activations.hidden,
                    &lw.v_proj_weight,
                    &self.activations.v,
                    seq_u,
                    kv_dim,
                    hidden_u,
                );
                // 6-9. Fused QK-norm + RoPE (4 dispatches → 1)
                self.dispatch_fused_qk_norm_rope(
                    enc,
                    seq_u,
                    num_heads,
                    num_kv_heads,
                    q_dim,
                    kv_dim,
                    eps,
                    &lw.q_norm_weight,
                    &lw.k_norm_weight,
                );

                // 10-12. Fused attention: Q@K^T + causal softmax + scores@V
                // Replaces 3 separate dispatches with 1 fused kernel.
                // Online softmax in registers, no global scores buffer.
                self.dispatch_fused_attention(enc, seq_u, q_dim, kv_dim, num_kv_heads, scale);

                // 13. O projection
                self.dispatch_matmul(
                    enc,
                    &self.activations.attn_out,
                    &lw.o_proj_weight,
                    &self.activations.hidden,
                    seq_u,
                    hidden_u,
                    q_dim,
                );

                // 14. Add residual: hidden += residual
                self.dispatch_add(
                    enc,
                    &self.activations.residual,
                    &self.activations.hidden,
                    hidden_elems,
                );

                // 15. Copy hidden -> residual (for FFN residual)
                self.dispatch_copy(
                    enc,
                    &self.activations.hidden,
                    &self.activations.residual,
                    hidden_elems,
                );

                // 16. Post-attention LayerNorm (in-place on hidden)
                self.dispatch_rms_norm(
                    enc,
                    &self.activations.hidden,
                    &lw.post_attention_layernorm_weight,
                    hidden_u,
                    seq_u,
                    eps,
                );

                // 17. Gate projection
                self.dispatch_matmul(
                    enc,
                    &self.activations.hidden,
                    &lw.gate_proj_weight,
                    &self.activations.gate,
                    seq_u,
                    inter_u,
                    hidden_u,
                );
                // 18. Up projection
                self.dispatch_matmul(
                    enc,
                    &self.activations.hidden,
                    &lw.up_proj_weight,
                    &self.activations.up,
                    seq_u,
                    inter_u,
                    hidden_u,
                );
                // 19. Fused SiLU(gate) * up -> gate
                self.dispatch_silu_mul(enc, inter_elems);
                // 20. Down projection
                self.dispatch_matmul(
                    enc,
                    &self.activations.gate,
                    &lw.down_proj_weight,
                    &self.activations.ffn_out,
                    seq_u,
                    hidden_u,
                    inter_u,
                );

                // 21. hidden = residual + ffn_out
                // Copy residual to hidden, then add ffn_out.
                self.dispatch_copy(
                    enc,
                    &self.activations.residual,
                    &self.activations.hidden,
                    hidden_elems,
                );
                self.dispatch_add(
                    enc,
                    &self.activations.ffn_out,
                    &self.activations.hidden,
                    hidden_elems,
                );

                enc.end_encoding();
            }

            // Final RMSNorm.
            {
                let enc = cmd.new_compute_command_encoder();
                self.dispatch_rms_norm(
                    enc,
                    &self.activations.hidden,
                    &self.final_norm_weight,
                    hidden_u,
                    seq_u,
                    eps,
                );
                enc.end_encoding();
            }

            // Submit and wait.
            cmd.commit();
            cmd.wait_until_completed();

            // SAFETY: Read back from the same unified memory buffer we wrote to above.
            // The GPU command buffer has completed (wait_until_completed), so the data
            // is coherent. Buffer size >= max_seq_len * hidden_size from allocation.
            let out_len = seq_len * hidden;
            let mut output = vec![0.0f32; out_len];
            // SAFETY: out_len is within the allocated hidden activation buffer and
            // output has the same number of f32 slots.
            unsafe {
                let src = self.activations.hidden.contents() as *const f32;
                std::ptr::copy_nonoverlapping(src, output.as_mut_ptr(), out_len);
            }
            Ok(output)
        }

        // -----------------------------------------------------------------------
        // Dispatch helpers
        // -----------------------------------------------------------------------

        fn dispatch_matmul(
            &self,
            enc: &ComputeCommandEncoderRef,
            a: &Buffer,
            b: &Buffer,
            c: &Buffer,
            m: u32,
            n: u32,
            k: u32,
        ) {
            // On Metal 4 (macOS 26+), the basic 16x16 tiled GEMM is 4.7x faster than
            // rect GEMM variants for small M. The rect kernels suffer from poor occupancy
            // on Metal 4's scheduler. Use matmul_bt for all shapes.
            enc.set_compute_pipeline_state(&self.matmul_pipeline);
            enc.set_buffer(0, Some(a), 0);
            enc.set_buffer(1, Some(b), 0);
            enc.set_buffer(2, Some(c), 0);
            enc.set_bytes(3, 4, &m as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            // Must use dispatch_thread_groups (not dispatch_threads) because
            // the tiled GEMM needs ALL threads in each threadgroup to fill the
            // shared tA/tB tiles. dispatch_threads creates non-uniform TGs at
            // edges, leaving tB partially uninitialized.
            let tile = 16u64;
            enc.dispatch_thread_groups(
                MTLSize::new((n as u64).div_ceil(tile), (m as u64).div_ceil(tile), 1),
                MTLSize::new(tile, tile, 1),
            );
        }

        /// Fused QK-norm + RoPE: replaces 4 dispatches (Q-norm, K-norm, RoPE-Q, RoPE-K) with 1.
        #[allow(clippy::too_many_arguments)]
        fn dispatch_fused_qk_norm_rope(
            &self,
            enc: &ComputeCommandEncoderRef,
            seq_len: u32,
            num_heads: u32,
            num_kv_heads: u32,
            q_dim: u32,
            kv_dim: u32,
            eps: f32,
            q_norm_weight: &Buffer,
            k_norm_weight: &Buffer,
        ) {
            let params = FusedQkNormRopeParams {
                seq_len,
                q_heads: num_heads,
                k_heads: num_kv_heads,
                q_stride: q_dim,
                k_stride: kv_dim,
                eps,
            };

            enc.set_compute_pipeline_state(&self.fused_qk_norm_rope_pipeline);
            enc.set_buffer(0, Some(&self.activations.q), 0);
            enc.set_buffer(1, Some(&self.activations.k), 0);
            enc.set_buffer(2, Some(q_norm_weight), 0);
            enc.set_buffer(3, Some(k_norm_weight), 0);
            enc.set_buffer(4, Some(&self.rope_cos), 0);
            enc.set_buffer(5, Some(&self.rope_sin), 0);
            enc.set_bytes(
                6,
                std::mem::size_of::<FusedQkNormRopeParams>() as u64,
                &params as *const FusedQkNormRopeParams as *const std::ffi::c_void,
            );

            // Grid: (seq_len, max(q_heads, k_heads), 2)
            // z=0 for Q, z=1 for K. Kernel guards against head >= head_count.
            let max_heads = num_heads.max(num_kv_heads);
            let grid = MTLSize::new(seq_len as u64, max_heads as u64, 2);
            let tg = MTLSize::new(64, 1, 1);
            enc.dispatch_thread_groups(grid, tg);
        }

        fn dispatch_rms_norm(
            &self,
            enc: &ComputeCommandEncoderRef,
            x: &Buffer,
            gamma: &Buffer,
            row_len: u32,
            num_rows: u32,
            eps: f32,
        ) {
            enc.set_compute_pipeline_state(&self.rms_norm_pipeline);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(gamma), 0);
            enc.set_bytes(2, 4, &row_len as *const u32 as *const _);
            enc.set_bytes(3, 4, &num_rows as *const u32 as *const _);
            enc.set_bytes(4, 4, &eps as *const f32 as *const _);

            // One threadgroup per row. Shader uses threadgroup_position_in_grid
            // as the row index and thread_position_in_threadgroup for strided access.
            let wg = 256u64;
            let grid = MTLSize::new(num_rows as u64, 1, 1);
            let tg = MTLSize::new(wg, 1, 1);
            enc.dispatch_thread_groups(grid, tg);
        }

        /// Fused attention: Q@K^T + causal softmax + scores@V in one kernel.
        /// Eliminates the global scores buffer. Online softmax in registers.
        fn dispatch_fused_attention(
            &self,
            enc: &ComputeCommandEncoderRef,
            seq_len: u32,
            q_dim: u32,
            kv_dim: u32,
            num_kv_heads: u32,
            scale: f32,
        ) {
            if seq_len == 0 {
                return;
            }

            #[repr(C)]
            struct Params {
                seq_len: u32,
                q_dim4: u32,
                kv_dim4: u32,
                num_kv_heads: u32,
                scale: f32,
                _pad0: u32,
                _pad1: u32,
                _pad2: u32,
            }

            let params = Params {
                seq_len,
                q_dim4: q_dim / 4,
                kv_dim4: kv_dim / 4,
                num_kv_heads,
                scale,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };

            const TILE_Q: u32 = 4;
            let gqa_groups: u32 =
                (self.config.num_attention_heads / self.config.num_key_value_heads) as u32;
            const SIMD_WIDTH: u32 = 32;
            let threads_per_tg: u64 = (TILE_Q * gqa_groups * SIMD_WIDTH) as u64;

            let q_blocks = div_ceil(seq_len as u64, TILE_Q as u64);

            enc.set_compute_pipeline_state(&self.fused_attention_pipeline);
            enc.set_buffer(0, Some(&self.activations.q), 0);
            enc.set_buffer(1, Some(&self.activations.k), 0);
            enc.set_buffer(2, Some(&self.activations.v), 0);
            enc.set_buffer(3, Some(&self.activations.attn_out), 0);
            enc.set_bytes(
                4,
                std::mem::size_of::<Params>() as u64,
                &params as *const Params as *const _,
            );

            // 2D grid: x = kv_head, y = query block
            let tg_count = MTLSize::new(num_kv_heads as u64, q_blocks, 1);
            let tg_size = MTLSize::new(threads_per_tg, 1, 1);
            enc.dispatch_thread_groups(tg_count, tg_size);
        }

        fn dispatch_silu_mul(&self, enc: &ComputeCommandEncoderRef, count: u32) {
            enc.set_compute_pipeline_state(&self.silu_mul_pipeline);
            enc.set_buffer(0, Some(&self.activations.gate), 0);
            enc.set_buffer(1, Some(&self.activations.up), 0);
            enc.set_bytes(2, 4, &count as *const u32 as *const _);

            let wg = 256u64;
            let grid = MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1);
            let tg = MTLSize::new(wg, 1, 1);
            enc.dispatch_threads(grid, tg);
        }

        fn dispatch_copy(
            &self,
            enc: &ComputeCommandEncoderRef,
            src: &Buffer,
            dst: &Buffer,
            count: u32,
        ) {
            enc.set_compute_pipeline_state(&self.copy_pipeline);
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst), 0);
            enc.set_bytes(2, 4, &count as *const u32 as *const _);

            let wg = 256u64;
            let grid = MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1);
            let tg = MTLSize::new(wg, 1, 1);
            enc.dispatch_threads(grid, tg);
        }

        fn dispatch_add(
            &self,
            enc: &ComputeCommandEncoderRef,
            src: &Buffer,
            dst: &Buffer,
            count: u32,
        ) {
            enc.set_compute_pipeline_state(&self.add_pipeline);
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst), 0);
            enc.set_bytes(2, 4, &count as *const u32 as *const _);

            let wg = 256u64;
            let grid = MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1);
            let tg = MTLSize::new(wg, 1, 1);
            enc.dispatch_threads(grid, tg);
        }
    }

    /// Build flat cos/sin RoPE tables: [max_seq_len * half_dim] each.
    fn build_rope_flat(head_dim: usize, max_seq_len: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
        let half_dim = head_dim / 2;
        let mut cos = Vec::with_capacity(max_seq_len * half_dim);
        let mut sin = Vec::with_capacity(max_seq_len * half_dim);

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
                let angle = pos as f64 * freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }

        (cos, sin)
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub use inner::MetalForwardPass;

// Stub for non-macOS or non-metal builds.
/// **Unstable**: Metal GPU forward pass; stub on non-macOS; full impl behind metal-gpu feature.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub struct MetalForwardPass;

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
impl MetalForwardPass {
    /// **Unstable**: construct Metal forward pass; always fails without metal-gpu feature.
    pub fn new(
        _config: &crate::model::qwen::QwenConfig,
        _weights: &crate::weights::QwenWeights<'_>,
        _max_seq_len: usize,
    ) -> Result<Self, String> {
        Err("Metal GPU not available (requires macOS + metal-gpu feature)".into())
    }

    /// **Unstable**: run Metal forward pass; always fails without metal-gpu feature.
    pub fn forward(&mut self, _hidden_input: &[f32], _seq_len: usize) -> Result<Vec<f32>, String> {
        Err("Metal GPU not available".into())
    }
}
