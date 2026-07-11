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

// ===== Optimized f16-weight GEMM/GEMV suite =====
// GemmParams used by all optimized kernels.
struct GemmParams {
    uint M;
    uint N;
    uint K;
    uint lda;  // K for tightly packed A
    uint ldb;  // K for tightly packed B
    uint ldc;  // N for tightly packed C
};

// Simdgroup tree reduction (Apple M-series: 32-lane simdgroups).
inline float simdgroup_reduce_add_f32(float v, uint lane_id, uint simd_width) {
    for (uint offset = simd_width >> 1; offset > 0; offset >>= 1) {
        float other = simd_shuffle_down(v, offset);
        if (lane_id + offset < simd_width) { v += other; }
    }
    return v;
}

// ===== Decode hot path (M==1): one threadgroup per output element =====
// float4/half4 vectorized loads, simdgroup reduction across K.
template<uint TG_THREADS>
inline void gemv_decode_core(device const float* A,
                             device const half*  B,
                             device float*       C,
                             constant GemmParams& p,
                             uint n, uint tid, uint simd_lane,
                             uint simdgroup_id, uint simd_width,
                             threadgroup float* sg_partials) {
    if (n >= p.N) return;
    const device half* brow = B + n * p.ldb;
    float partial = 0.0f;

    // Fast path: float4/half4 vectorization.
    if (((p.K & 3u) == 0u) && ((p.ldb & 3u) == 0u)) {
        const device float4* a4 = (const device float4*)A;
        const device half4*  b4 = (const device half4*)brow;
        for (uint k4 = tid; k4 < (p.K >> 2); k4 += TG_THREADS) {
            partial += dot(a4[k4], float4(b4[k4]));
        }
    } else if (((p.K & 1u) == 0u) && ((p.ldb & 1u) == 0u)) {
        const device float2* a2 = (const device float2*)A;
        const device half2*  b2 = (const device half2*)brow;
        for (uint k2 = tid; k2 < (p.K >> 1); k2 += TG_THREADS) {
            const float2 av = a2[k2];
            const half2  bv = b2[k2];
            partial = fma(av.x, float(bv.x), partial);
            partial = fma(av.y, float(bv.y), partial);
        }
    } else {
        for (uint k = tid; k < p.K; k += TG_THREADS) {
            partial = fma(A[k], float(brow[k]), partial);
        }
    }

    // Simdgroup reduction then cross-simdgroup fold.
    partial = simdgroup_reduce_add_f32(partial, simd_lane, simd_width);
    if (simd_lane == 0u) { sg_partials[simdgroup_id] = partial; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_id == 0u) {
        const uint sg_count = (TG_THREADS + simd_width - 1u) / simd_width;
        float block_sum = (simd_lane < sg_count) ? sg_partials[simd_lane] : 0.0f;
        block_sum = simdgroup_reduce_add_f32(block_sum, simd_lane, simd_width);
        if (simd_lane == 0u) { C[n] = block_sum; }
    }
}

// M==1 decode GEMV, 256 threads per threadgroup (best for K~2048 on M2).
// Launch: threadgroups=(N,1,1), threads=(256,1,1)
kernel void gemv_decode_m1(device const float* A [[buffer(0)]],
                           device const half*  B [[buffer(1)]],
                           device float*       C [[buffer(2)]],
                           constant GemmParams& p [[buffer(3)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint3 tg_pos [[threadgroup_position_in_grid]],
                           uint simd_lane [[thread_index_in_simdgroup]],
                           uint simdgroup_id [[simdgroup_index_in_threadgroup]],
                           uint simd_width [[threads_per_simdgroup]]) {
    threadgroup float sg_partials[8]; // 256/32
    gemv_decode_core<256>(A, B, C, p, tg_pos.x, tid,
                          simd_lane, simdgroup_id, simd_width, sg_partials);
}

// ===== Q8_0 GEMV Decode: int8 x float32 with simd_sum reduction =====
// Q8_0 block: [f16 scale (2B)][32 x int8 (32B)] = 34 bytes per block.
// NR=2 output rows per threadgroup, 4 simdgroups of 32 threads = 128 threads.
// Dispatch: threadgroups=(ceil(N/2), 1, 1), threads=(32, 4, 1)
kernel void gemv_q8_decode(
    device const float* x        [[buffer(0)]],
    device const char*  qweight  [[buffer(1)]],
    device float*       y        [[buffer(2)]],
    constant uint& N             [[buffer(3)]],
    constant uint& K             [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR = 2;
    const uint NSG = 4;
    const uint nb = K / 32;
    const uint row_bytes = nb * 34;
    const uint first_row = tgpig.x * NR;
    const uint ix = tiisg / 4;
    const uint il = tiisg % 4;

    float sumf[NR] = {0.0f};
    const uint ib_start = sgitg * 8 + ix;
    const uint ib_stride = NSG * 8;
    device const float* yb = x + ib_start * 32 + il * 8;

    for (uint ib = ib_start; ib < nb; ib += ib_stride) {
        float yl[8];
        for (uint i = 0; i < 8; i++) yl[i] = yb[i];
        yb += ib_stride * 32;

        for (uint row = 0; row < NR; row++) {
            uint r = first_row + row;
            if (r >= N) continue;
            device const char* base = qweight + r * row_bytes + ib * 34;
            device const char* qs = base + 2 + il * 8;
            half d = *((device const half*)base);
            float sumq = 0.0f;
            for (uint i = 0; i < 8; i++) sumq += float(qs[i]) * yl[i];
            sumf[row] += sumq * float(d);
        }
    }

    for (uint row = 0; row < NR; row++) sumf[row] = simd_sum(sumf[row]);

    threadgroup float shared[NR][4];
    if (tiisg == 0) {
        for (uint row = 0; row < NR; row++) shared[row][sgitg] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        for (uint row = 0; row < NR; row++) {
            uint r = first_row + row;
            if (r < N) {
                y[r] = shared[row][0] + shared[row][1] + shared[row][2] + shared[row][3];
            }
        }
    }
}

// ===== Q8_0 GEMV Decode (wide): NR=4 variant for large-N matmuls (lm_head) =====
// Halves threadgroup count vs NR=2, better for N > 8192 where scheduling dominates.
// Dispatch: threadgroups=(ceil(N/4), 1, 1), threads=(32, 4, 1)
kernel void gemv_q8_decode_wide(
    device const float* x        [[buffer(0)]],
    device const char*  qweight  [[buffer(1)]],
    device float*       y        [[buffer(2)]],
    constant uint& N             [[buffer(3)]],
    constant uint& K             [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR = 4;
    const uint NSG = 4;
    const uint nb = K / 32;
    const uint row_bytes = nb * 34;
    const uint first_row = tgpig.x * NR;
    const uint ix = tiisg / 4;
    const uint il = tiisg % 4;

    float sumf[NR] = {0.0f};
    const uint ib_start = sgitg * 8 + ix;
    const uint ib_stride = NSG * 8;
    device const float* yb = x + ib_start * 32 + il * 8;

    for (uint ib = ib_start; ib < nb; ib += ib_stride) {
        float yl[8];
        for (uint i = 0; i < 8; i++) yl[i] = yb[i];
        yb += ib_stride * 32;

        for (uint row = 0; row < NR; row++) {
            uint r = first_row + row;
            if (r >= N) continue;
            device const char* base = qweight + r * row_bytes + ib * 34;
            device const char* qs = base + 2 + il * 8;
            half d = *((device const half*)base);
            float sumq = 0.0f;
            for (uint i = 0; i < 8; i++) sumq += float(qs[i]) * yl[i];
            sumf[row] += sumq * float(d);
        }
    }

    for (uint row = 0; row < NR; row++) sumf[row] = simd_sum(sumf[row]);

    threadgroup float shared[NR][4];
    if (tiisg == 0) {
        for (uint row = 0; row < NR; row++) shared[row][sgitg] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        for (uint row = 0; row < NR; row++) {
            uint r = first_row + row;
            if (r < N) {
                y[r] = shared[row][0] + shared[row][1] + shared[row][2] + shared[row][3];
            }
        }
    }
}

// ===== FP16-weight GEMV Decode (wide): NR=4 rows per threadgroup =====
// Same scheduling as gemv_q8_decode_wide but reads half-precision weights
// directly (no Q8 block structure). Dispatched for large-N matmuls (lm_head)
// where the NR=1 gemv_decode_m1 kernel creates excessive threadgroup count.
// Dispatch: threadgroups=(ceil(N/4), 1, 1), threads=(32, 4, 1)
kernel void gemv_decode_wide_f16(
    device const float* x       [[buffer(0)]],
    device const half*  W       [[buffer(1)]],
    device float*       y       [[buffer(2)]],
    constant uint& N            [[buffer(3)]],
    constant uint& K            [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR = 4;
    const uint NSG = 4;
    const uint nb = K / 32;
    const uint first_row = tgpig.x * NR;
    const uint ix = tiisg / 4;
    const uint il = tiisg % 4;

    float sumf[NR] = {0.0f};
    const uint ib_start = sgitg * 8 + ix;
    const uint ib_stride = NSG * 8;
    device const float* xb = x + ib_start * 32 + il * 8;

    for (uint ib = ib_start; ib < nb; ib += ib_stride) {
        float xl[8];
        for (uint i = 0; i < 8; i++) xl[i] = xb[i];
        xb += ib_stride * 32;

        for (uint row = 0; row < NR; row++) {
            uint r = first_row + row;
            if (r >= N) continue;
            device const half* wrow = W + r * K + ib * 32 + il * 8;
            float dot = 0.0f;
            for (uint i = 0; i < 8; i++) dot += xl[i] * float(wrow[i]);
            sumf[row] += dot;
        }
    }

    for (uint row = 0; row < NR; row++) sumf[row] = simd_sum(sumf[row]);

    threadgroup float shared[NR][4];
    if (tiisg == 0) {
        for (uint row = 0; row < NR; row++) shared[row][sgitg] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        for (uint row = 0; row < NR; row++) {
            uint r = first_row + row;
            if (r < N) {
                y[r] = shared[row][0] + shared[row][1] + shared[row][2] + shared[row][3];
            }
        }
    }
}

// ===== Qwen3.5 RMS Norm: x = x * (1 + gamma) / sqrt(mean(x^2) + eps) =====
// Shifted norm: (1 + gamma) instead of plain gamma.
// One threadgroup per row.
kernel void rms_norm_qwen35(
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

    threadgroup float shared[RMS_WG];
    float local_sum = 0.0f;
    for (uint i = lid; i < row_len; i += tgs) {
        float v = x[base + i];
        local_sum += v * v;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared[0] / float(row_len) + eps);

    // Qwen3.5: shifted RMSNorm (1 + gamma). Empirically required — plain
    // gamma produces garbage logits on this model.
    for (uint i = lid; i < row_len; i += tgs) {
        x[base + i] = x[base + i] * rms * (1.0f + gamma[i]);
    }
}

// ===== Stride-Half Partial RoPE for Qwen3.5 full-attention layers =====
// Pairs are (i, half+i) for i in 0..half_rope_dim — HF rotate_half convention,
// matching MLX nn.RoPE(traditional=False). Kernel name kept for ABI continuity.
// Only first rope_dim dimensions are rotated; the rest are untouched.
// Operates on x[num_heads * head_dim] for a single position.
kernel void partial_rope_interleaved(
    device float* x              [[buffer(0)]],
    device const float* cos_tab  [[buffer(1)]],
    device const float* sin_tab  [[buffer(2)]],
    constant uint& num_heads     [[buffer(3)]],
    constant uint& head_dim      [[buffer(4)]],
    constant uint& half_rope_dim [[buffer(5)]],
    constant uint& pos_offset    [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint total_pairs = num_heads * half_rope_dim;
    if (gid >= total_pairs) return;

    uint pair = gid % half_rope_dim;
    uint head = gid / half_rope_dim;

    uint base = head * head_dim;
    uint cs_base = pos_offset * half_rope_dim;

    float cos_val = cos_tab[cs_base + pair];
    float sin_val = sin_tab[cs_base + pair];

    // Stride-half: rotate (pair, half_rope_dim + pair) within each head
    uint idx0 = base + pair;
    uint idx1 = base + half_rope_dim + pair;
    float x0 = x[idx0];
    float x1 = x[idx1];
    x[idx0] = x0 * cos_val - x1 * sin_val;
    x[idx1] = x0 * sin_val + x1 * cos_val;
}

// ===== Qwen3.5 QK-norm: per-head RMS norm with (1+gamma) =====
// x: [num_heads * head_dim], norms each head independently.
kernel void per_head_rms_norm(
    device float* x           [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    constant uint& num_heads  [[buffer(2)]],
    constant uint& head_dim   [[buffer(3)]],
    constant float& eps       [[buffer(4)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]])
{
    if (gid >= num_heads) return;
    constexpr uint NORM_WG = 256;
    uint base = gid * head_dim;

    threadgroup float shared[NORM_WG];
    float local_sum = 0.0f;
    for (uint i = lid; i < head_dim; i += tgs) {
        float v = x[base + i];
        local_sum += v * v;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared[lid] += shared[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared[0] / float(head_dim) + eps);

    // Qwen3.5: shifted RMSNorm (1 + gamma) — per-head variant for q_norm/k_norm.
    for (uint i = lid; i < head_dim; i += tgs) {
        x[base + i] = x[base + i] * rms * (1.0f + gamma[i]);
    }
}

// ===== FlashAttention-style GQA decode (H1+H2+H4+H5) =====
// Single-pass online softmax. One threadgroup per KV head processes all query heads in the
// GQA group simultaneously. QK computed once per (query_head, cache_tile), scores stored in
// threadgroup memory and reused for V accumulation. Q loaded once into threadgroup memory.
// Grid: [num_kv_heads, 1, 1].  Threads: [256, 1, 1] — one thread per output dimension.
//
// KVT is the KV cache element type (float for f32 path, half for f16 path).
// float(...) casts are no-ops for float and widen half→float for the f16 path.
#define DEFINE_DECODE_ATTENTION(NAME, KVT) \
kernel void NAME( \
    device const float* q        [[buffer(0)]], \
    device const KVT*   k_cache  [[buffer(1)]], \
    device const KVT*   v_cache  [[buffer(2)]], \
    device float* out            [[buffer(3)]], \
    constant uint& cache_len     [[buffer(4)]], \
    constant uint& head_dim      [[buffer(5)]], \
    constant uint& num_q_heads   [[buffer(6)]], \
    constant uint& num_kv_heads  [[buffer(7)]], \
    constant uint& q_dim         [[buffer(8)]], \
    constant uint& kv_dim        [[buffer(9)]], \
    constant float& scale        [[buffer(10)]], \
    uint gid  [[threadgroup_position_in_grid]], \
    uint lid  [[thread_position_in_threadgroup]], \
    uint tgs  [[threads_per_threadgroup]]) \
{ \
    if (cache_len == 0) return; \
    constexpr uint HEAD_DIM    = 256; \
    constexpr uint MAX_GRP     = 8; \
    constexpr uint TILE_TOKENS = 256; \
    if (head_dim != HEAD_DIM || num_kv_heads == 0) return; \
    const uint kvh = gid; \
    if (kvh >= num_kv_heads) return; \
    if ((num_q_heads % num_kv_heads) != 0) return; \
    const uint group_size = num_q_heads / num_kv_heads; \
    if (group_size == 0 || group_size > MAX_GRP) return; \
    const uint qh_base = kvh * group_size; \
    threadgroup float q_s    [MAX_GRP * HEAD_DIM]; \
    threadgroup float score_s[MAX_GRP * TILE_TOKENS]; \
    threadgroup float reduce_s[TILE_TOKENS]; \
    threadgroup float m_s    [MAX_GRP]; \
    threadgroup float l_s    [MAX_GRP]; \
    threadgroup float alpha_s[MAX_GRP]; \
    if (lid < group_size) { \
        m_s[lid] = -INFINITY; \
        l_s[lid] = 0.0f; \
    } \
    for (uint idx = lid; idx < group_size * HEAD_DIM; idx += tgs) { \
        q_s[idx] = q[qh_base * HEAD_DIM + idx]; \
    } \
    float acc[MAX_GRP]; \
    for (uint qi = 0; qi < MAX_GRP; qi++) acc[qi] = 0.0f; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint tile_start = 0; tile_start < cache_len; tile_start += TILE_TOKENS) { \
        const uint tile_count = min(TILE_TOKENS, cache_len - tile_start); \
        if (lid < tile_count) { \
            float dot[MAX_GRP]; \
            for (uint qi = 0; qi < MAX_GRP; qi++) dot[qi] = 0.0f; \
            const uint k_base = (tile_start + lid) * kv_dim + kvh * HEAD_DIM; \
            for (uint d = 0; d < HEAD_DIM; d++) { \
                const float kd = float(k_cache[k_base + d]); \
                for (uint qi = 0; qi < group_size; qi++) { \
                    dot[qi] += q_s[qi * HEAD_DIM + d] * kd; \
                } \
            } \
            for (uint qi = 0; qi < group_size; qi++) { \
                score_s[qi * TILE_TOKENS + lid] = dot[qi] * scale; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint qi = 0; qi < group_size; qi++) { \
            reduce_s[lid] = (lid < tile_count) ? score_s[qi * TILE_TOKENS + lid] : -INFINITY; \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            for (uint s = tgs >> 1; s > 0; s >>= 1) { \
                if (lid < s) reduce_s[lid] = max(reduce_s[lid], reduce_s[lid + s]); \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
            } \
            const float tile_max = reduce_s[0]; \
            if (lid == 0) { \
                const float m_old = m_s[qi]; \
                const float m_new = max(m_old, tile_max); \
                alpha_s[qi] = isfinite(m_old) ? exp(m_old - m_new) : 0.0f; \
                m_s[qi]     = m_new; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            if (lid < tile_count) { \
                score_s[qi * TILE_TOKENS + lid] = \
                    exp(score_s[qi * TILE_TOKENS + lid] - m_s[qi]); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            reduce_s[lid] = (lid < tile_count) ? score_s[qi * TILE_TOKENS + lid] : 0.0f; \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            for (uint s = tgs >> 1; s > 0; s >>= 1) { \
                if (lid < s) reduce_s[lid] += reduce_s[lid + s]; \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
            } \
            if (lid == 0) { \
                l_s[qi] = alpha_s[qi] * l_s[qi] + reduce_s[0]; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        for (uint qi = 0; qi < group_size; qi++) { \
            acc[qi] *= alpha_s[qi]; \
        } \
        const uint d = lid; \
        for (uint local_t = 0; local_t < tile_count; local_t++) { \
            const float v = float(v_cache[(tile_start + local_t) * kv_dim + kvh * HEAD_DIM + d]); \
            for (uint qi = 0; qi < group_size; qi++) { \
                acc[qi] += score_s[qi * TILE_TOKENS + local_t] * v; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    for (uint qi = 0; qi < group_size; qi++) { \
        const uint qh  = qh_base + qi; \
        const float dn = l_s[qi]; \
        out[qh * HEAD_DIM + lid] = dn > 0.0f ? acc[qi] / dn : 0.0f; \
    } \
}

DEFINE_DECODE_ATTENTION(decode_attention,     float)
DEFINE_DECODE_ATTENTION(decode_attention_f16, half)

// ===== Partitioned flash decode — partial kernel (H3) =====
// One threadgroup per (KV head, partition). Runs the same tiled online-softmax as
// decode_attention but over partition_tokens-length slice of the KV cache.
// Writes (m, l, acc[head_dim]) partial state for each query head in the GQA group.
// Partials layout: partials[((part * num_q_heads + qh) * (HEAD_DIM+2)) + offset]
//   offset 0 = running max m, offset 1 = running sum l, offset 2+ = acc[d].
// Grid: [num_kv_heads, num_partitions, 1].  Threads: [256, 1, 1].
//
// KVT is the KV cache element type (float for f32 path, half for f16 path).
#define DEFINE_DECODE_ATTENTION_FLASH_PARTIAL(NAME, KVT) \
kernel void NAME( \
    device const float* q           [[buffer(0)]], \
    device const KVT*   k_cache     [[buffer(1)]], \
    device const KVT*   v_cache     [[buffer(2)]], \
    device float* attn_partials     [[buffer(3)]], \
    constant uint& cache_len        [[buffer(4)]], \
    constant uint& head_dim         [[buffer(5)]], \
    constant uint& num_q_heads      [[buffer(6)]], \
    constant uint& num_kv_heads     [[buffer(7)]], \
    constant uint& q_dim            [[buffer(8)]], \
    constant uint& kv_dim           [[buffer(9)]], \
    constant float& scale           [[buffer(10)]], \
    constant uint& partition_tokens [[buffer(11)]], \
    uint3 gid3  [[threadgroup_position_in_grid]], \
    uint3 lid3  [[thread_position_in_threadgroup]], \
    uint3 tgs3  [[threads_per_threadgroup]]) \
{ \
    constexpr uint HEAD_DIM    = 256; \
    constexpr uint MAX_GRP     = 8; \
    constexpr uint TILE_TOKENS = 256; \
    constexpr uint STRIDE      = HEAD_DIM + 2; \
    const uint lid          = lid3.x; \
    const uint tgs          = tgs3.x; \
    const uint kvh          = gid3.x; \
    const uint partition_id = gid3.y; \
    if (head_dim != HEAD_DIM || partition_tokens == 0 || num_kv_heads == 0) return; \
    if (kvh >= num_kv_heads) return; \
    if ((num_q_heads % num_kv_heads) != 0) return; \
    const uint part_start = partition_id * partition_tokens; \
    if (part_start >= cache_len) return; \
    const uint part_end = min(cache_len, part_start + partition_tokens); \
    const uint group_size = num_q_heads / num_kv_heads; \
    if (group_size == 0 || group_size > MAX_GRP) return; \
    const uint qh_base = kvh * group_size; \
    threadgroup float q_s    [MAX_GRP * HEAD_DIM]; \
    threadgroup float score_s[MAX_GRP * TILE_TOKENS]; \
    threadgroup float reduce_s[TILE_TOKENS]; \
    threadgroup float m_s    [MAX_GRP]; \
    threadgroup float l_s    [MAX_GRP]; \
    threadgroup float alpha_s[MAX_GRP]; \
    if (lid < group_size) { \
        m_s[lid] = -INFINITY; \
        l_s[lid] = 0.0f; \
    } \
    for (uint idx = lid; idx < group_size * HEAD_DIM; idx += tgs) { \
        q_s[idx] = q[qh_base * HEAD_DIM + idx]; \
    } \
    float acc[MAX_GRP]; \
    for (uint qi = 0; qi < MAX_GRP; qi++) acc[qi] = 0.0f; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint tile_start = part_start; tile_start < part_end; tile_start += TILE_TOKENS) { \
        const uint tile_count = min(TILE_TOKENS, part_end - tile_start); \
        if (lid < tile_count) { \
            float dot[MAX_GRP]; \
            for (uint qi = 0; qi < MAX_GRP; qi++) dot[qi] = 0.0f; \
            const uint k_base = (tile_start + lid) * kv_dim + kvh * HEAD_DIM; \
            for (uint d = 0; d < HEAD_DIM; d++) { \
                const float kd = float(k_cache[k_base + d]); \
                for (uint qi = 0; qi < group_size; qi++) { \
                    dot[qi] += q_s[qi * HEAD_DIM + d] * kd; \
                } \
            } \
            for (uint qi = 0; qi < group_size; qi++) { \
                score_s[qi * TILE_TOKENS + lid] = dot[qi] * scale; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint qi = 0; qi < group_size; qi++) { \
            reduce_s[lid] = (lid < tile_count) ? score_s[qi * TILE_TOKENS + lid] : -INFINITY; \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            for (uint s = tgs >> 1; s > 0; s >>= 1) { \
                if (lid < s) reduce_s[lid] = max(reduce_s[lid], reduce_s[lid + s]); \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
            } \
            const float tile_max = reduce_s[0]; \
            if (lid == 0) { \
                const float m_old = m_s[qi]; \
                const float m_new = max(m_old, tile_max); \
                alpha_s[qi] = isfinite(m_old) ? exp(m_old - m_new) : 0.0f; \
                m_s[qi]     = m_new; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            if (lid < tile_count) { \
                score_s[qi * TILE_TOKENS + lid] = \
                    exp(score_s[qi * TILE_TOKENS + lid] - m_s[qi]); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            reduce_s[lid] = (lid < tile_count) ? score_s[qi * TILE_TOKENS + lid] : 0.0f; \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            for (uint s = tgs >> 1; s > 0; s >>= 1) { \
                if (lid < s) reduce_s[lid] += reduce_s[lid + s]; \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
            } \
            if (lid == 0) { \
                l_s[qi] = alpha_s[qi] * l_s[qi] + reduce_s[0]; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        for (uint qi = 0; qi < group_size; qi++) acc[qi] *= alpha_s[qi]; \
        const uint d = lid; \
        for (uint local_t = 0; local_t < tile_count; local_t++) { \
            const float v = float(v_cache[(tile_start + local_t) * kv_dim + kvh * HEAD_DIM + d]); \
            for (uint qi = 0; qi < group_size; qi++) { \
                acc[qi] += score_s[qi * TILE_TOKENS + local_t] * v; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    const uint d = lid; \
    for (uint qi = 0; qi < group_size; qi++) { \
        const uint qh   = qh_base + qi; \
        const uint base = (partition_id * num_q_heads + qh) * STRIDE; \
        if (lid == 0) { \
            attn_partials[base + 0] = m_s[qi]; \
            attn_partials[base + 1] = l_s[qi]; \
        } \
        attn_partials[base + 2 + d] = acc[qi]; \
    } \
}

DEFINE_DECODE_ATTENTION_FLASH_PARTIAL(decode_attention_flash_partial,     float)
DEFINE_DECODE_ATTENTION_FLASH_PARTIAL(decode_attention_flash_partial_f16, half)

// ===== Partitioned flash decode — reduce kernel (H3) =====
// Combines partition partial states into the final attention output.
// Each thread lid owns output dimension d=lid; iterates over partitions in registers.
// No threadgroup memory needed — all accumulation is per-thread.
// Grid: [num_kv_heads, 1, 1].  Threads: [256, 1, 1].
kernel void decode_attention_flash_reduce(
    device const float* attn_partials [[buffer(0)]],
    device float* out                 [[buffer(1)]],
    constant uint& num_q_heads        [[buffer(2)]],
    constant uint& num_kv_heads       [[buffer(3)]],
    constant uint& num_partitions     [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    constexpr uint HEAD_DIM = 256;
    constexpr uint MAX_GRP  = 8;
    constexpr uint STRIDE   = HEAD_DIM + 2; // m + l + acc[256]

    if (num_kv_heads == 0) return;
    const uint kvh = gid;
    if (kvh >= num_kv_heads) return;
    if ((num_q_heads % num_kv_heads) != 0) return;
    const uint group_size = num_q_heads / num_kv_heads;
    if (group_size == 0 || group_size > MAX_GRP) return;
    const uint qh_base = kvh * group_size;
    const uint d = lid; // output dimension owned by this thread

    float m_acc[MAX_GRP], l_acc[MAX_GRP], acc[MAX_GRP];
    for (uint qi = 0; qi < group_size; qi++) {
        m_acc[qi] = -INFINITY;
        l_acc[qi] = 0.0f;
        acc[qi]   = 0.0f;
    }

    // Combine partitions using the same online recurrence
    for (uint p = 0; p < num_partitions; p++) {
        for (uint qi = 0; qi < group_size; qi++) {
            const uint qh     = qh_base + qi;
            const uint base   = (p * num_q_heads + qh) * STRIDE;
            const float m_p   = attn_partials[base + 0];
            const float l_p   = attn_partials[base + 1];
            const float acc_d = attn_partials[base + 2 + d];

            if (!isfinite(m_p)) continue; // empty partition

            const float m_new = max(m_acc[qi], m_p);
            const float a_acc = isfinite(m_acc[qi]) ? exp(m_acc[qi] - m_new) : 0.0f;
            const float a_p   = exp(m_p - m_new);
            l_acc[qi]  = a_acc * l_acc[qi]  + a_p * l_p;
            acc[qi]    = a_acc * acc[qi]     + a_p * acc_d;
            m_acc[qi]  = m_new;
        }
    }

    for (uint qi = 0; qi < group_size; qi++) {
        const uint qh = qh_base + qi;
        out[qh * HEAD_DIM + d] = l_acc[qi] > 0.0f ? acc[qi] / l_acc[qi] : 0.0f;
    }
}

// ===== Output gating: out[i] *= sigmoid(gate[i]) =====
kernel void sigmoid_gate(
    device float* out        [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float g = gate[gid];
    float sig = 1.0f / (1.0f + exp(-g));
    out[gid] = out[gid] * sig;
}

// ===== Scatter Q and gate from interleaved q_proj output =====
// q_proj output: [num_heads * head_dim * 2] with (Q_h, gate_h) per head
// Scatter into separate Q[num_heads * head_dim] and gate[num_heads * head_dim]
kernel void scatter_q_gate(
    device const float* qg_interleaved [[buffer(0)]],
    device float* q_out                [[buffer(1)]],
    device float* gate_out             [[buffer(2)]],
    constant uint& num_heads           [[buffer(3)]],
    constant uint& head_dim            [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = num_heads * head_dim;
    if (gid >= total) return;

    uint head = gid / head_dim;
    uint d = gid % head_dim;

    uint src_base = head * head_dim * 2;
    q_out[gid] = qg_interleaved[src_base + d];
    gate_out[gid] = qg_interleaved[src_base + head_dim + d];
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

// Dense decode fused gate||up kernel: first half = gate, second half = up.
kernel void silu_mul_fused(
    device float* data    [[buffer(0)]],
    constant uint& count  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float gate = data[gid];
    float up   = data[gid + count];
    float s    = gate / (1.0f + exp(-gate));
    data[gid]  = s * up;
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

// ===== Fused residual add + RMS norm =====
// Combines: residual_out = base + delta; normed_out = rms_norm(residual_out) * (1+gamma)
// Replaces 4 dispatches (copy+add+copy+rms_norm) with 1.
// One threadgroup, 256 threads.
kernel void fused_residual_add_norm(
    device const float* base     [[buffer(0)]],   // pre-norm residual
    device const float* delta    [[buffer(1)]],   // attention/FFN output to add
    device float* residual_out   [[buffer(2)]],   // updated residual (base + delta)
    device float* normed_out     [[buffer(3)]],   // RMS-normed result for next sublayer
    device const float* gamma    [[buffer(4)]],   // norm weights
    constant uint& row_len       [[buffer(5)]],
    constant float& eps          [[buffer(6)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]])
{
    constexpr uint WG = 256;
    threadgroup float shared[WG];

    // Phase 1: residual_out = base + delta, accumulate sum of squares
    float local_sum = 0.0f;
    for (uint i = lid; i < row_len; i += tgs) {
        float val = base[i] + delta[i];
        residual_out[i] = val;
        local_sum += val * val;
    }

    // Reduce sum of squares
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(shared[0] / float(row_len) + eps);

    // Phase 2: normed_out = residual_out * rms * (1 + gamma) (Qwen3.5 shifted RMSNorm).
    for (uint i = lid; i < row_len; i += tgs) {
        normed_out[i] = residual_out[i] * rms * (1.0f + gamma[i]);
    }
}

// ===== Copy with offset: dst[dst_offset + i] = src[i] for i in 0..count =====
kernel void copy_buf_offset(
    device const float* src    [[buffer(0)]],
    device float* dst          [[buffer(1)]],
    constant uint& count       [[buffer(2)]],
    constant uint& dst_offset  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    dst[dst_offset + gid] = src[gid];
}

// Variant that narrows f32 src → half dst; used for f16 KV cache writes.
// Offsets are element-counted (same semantic as copy_buf_offset, different element size).
kernel void copy_buf_offset_f16(
    device const float* src    [[buffer(0)]],
    device half*  dst          [[buffer(1)]],
    constant uint& count       [[buffer(2)]],
    constant uint& dst_offset  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    dst[dst_offset + gid] = half(src[gid]);
}

// ===== Q4 GEMV Decode: asymmetric 4-bit x float32 with simd_sum reduction =====
// Q4 block: [f16 scale (2B)][f16 bias/min (2B)][16 bytes packed 4-bit] = 20 bytes per 32 weights.
// Packed nibbles are unsigned sequential pairs: real_value = nibble * scale + bias.
// NR=2 output rows per threadgroup, 4 simdgroups of 32 = 128 threads.
// Dispatch: threadgroups=(ceil(N/2), 1, 1), threads=(32, 4, 1)
kernel void gemv_q4_decode(
    device const float* x        [[buffer(0)]],
    device const char*  qweight  [[buffer(1)]],
    device float*       y        [[buffer(2)]],
    constant uint& N             [[buffer(3)]],
    constant uint& K             [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR = 2;
    const uint NSG = 4;
    const uint nb = K / 32;           // number of Q4 blocks per row
    const uint row_bytes = nb * 20;   // 20 bytes per asymmetric block (scale + bias + 16 nibbles)
    const uint first_row = tgpig.x * NR;
    const uint ix = tiisg / 4;        // block sub-index (0..7)
    const uint il = tiisg % 4;        // lane within block (0..3)

    float sumf[NR] = {0.0f};
    const uint ib_start = sgitg * 8 + ix;
    const uint ib_stride = NSG * 8;

    // Each lane handles 8 weights (4 bytes of packed nibbles = 8 nibbles)
    device const float* yb = x + ib_start * 32 + il * 8;

    for (uint ib = ib_start; ib < nb; ib += ib_stride) {
        float yl[8];
        float yl_sum = 0.0f;
        for (uint i = 0; i < 8; i++) { yl[i] = yb[i]; yl_sum += yb[i]; }
        yb += ib_stride * 32;

        for (uint row = 0; row < NR; row++) {
            uint r = first_row + row;
            if (r >= N) continue;
            device const uchar* base = (device const uchar*)(qweight + r * row_bytes + ib * 20);
            float d = float(*((device const half*)base));         // scale
            float b = float(*((device const half*)(base + 2)));   // bias (min)
            device const uchar* qs = base + 4 + il * 4;  // 4 bytes = 8 nibbles

            // Asymmetric dequant: w = nibble * d + b. The bias contribution
            // factors out of the inner loop: sum(nibble*x)*d + sum(x)*b.
            float sumq_dot = 0.0f;
            for (uint i = 0; i < 4; i++) {
                uchar byte_val = qs[i];
                float n0 = float(byte_val & 0xF);
                float n1 = float(byte_val >> 4);
                sumq_dot += n0 * yl[i * 2] + n1 * yl[i * 2 + 1];
            }
            sumf[row] += sumq_dot * d + b * yl_sum;
        }
    }

    for (uint row = 0; row < NR; row++) sumf[row] = simd_sum(sumf[row]);

    threadgroup float shared[NR][4];
    if (tiisg == 0) {
        for (uint row = 0; row < NR; row++) shared[row][sgitg] = sumf[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        for (uint row = 0; row < NR; row++) {
            uint r = first_row + row;
            if (r < N) {
                y[r] = shared[row][0] + shared[row][1] + shared[row][2] + shared[row][3];
            }
        }
    }
}

// ===== GDN: Depthwise conv1d + SiLU =====
// conv_buf: [conv_dim * buf_len] persistent shift register (buf_len = kernel_size - 1)
// One thread per channel.
kernel void conv1d_depthwise_silu(
    device float* conv_buf             [[buffer(0)]],
    device const float* input          [[buffer(1)]],
    device const float* weight         [[buffer(2)]],
    device float* output               [[buffer(3)]],
    constant uint& conv_dim            [[buffer(4)]],
    constant uint& kernel_size         [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= conv_dim) return;
    uint buf_len = kernel_size - 1;
    device float* buf = conv_buf + gid * buf_len;
    device const float* w = weight + gid * kernel_size;

    float sum = 0.0f;
    for (uint t = 0; t < buf_len; t++) sum += buf[t] * w[t];
    float x = input[gid];
    sum += x * w[buf_len];
    output[gid] = sum / (1.0f + exp(-sum));  // SiLU

    for (uint t = 0; t + 1 < buf_len; t++) buf[t] = buf[t + 1];
    if (buf_len > 0) buf[buf_len - 1] = x;
}

// ===== GDN: Fused per-head recurrence =====
// One threadgroup per head, 128 threads (4 simdgroups).
// S stored transposed: S^T[value_dim, key_dim] for contiguous row access.
struct GdnRecurParams {
    uint key_dim;          // 128
    uint value_dim;        // 128
    uint num_key_heads;    // 16
    uint num_value_heads;  // 48 (or 16 for symmetric)
    uint hidden_size;      // 2048
    uint q_total;          // num_key_heads * key_dim
    uint v_offset;         // q_total + num_key_heads * key_dim
    float scale;           // 1/sqrt(key_dim)
    float eps;             // rms_norm_eps
};

kernel void gdn_recurrence_fused(
    device float* S_all              [[buffer(0)]],   // [num_value_heads, vd, kd]
    device const float* conv_out     [[buffer(1)]],   // [qkv_dim]
    device const float* z_proj       [[buffer(2)]],   // [output_dim]
    device const float* hidden_in    [[buffer(3)]],   // [hidden_size]
    device const half*  in_proj_b    [[buffer(4)]],   // [num_value_heads, hidden_size]
    device const half*  in_proj_a    [[buffer(5)]],   // [num_value_heads, hidden_size]
    device const float* a_log        [[buffer(6)]],   // [num_value_heads]
    device const float* dt_bias      [[buffer(7)]],   // [num_value_heads]
    device const float* norm_w       [[buffer(8)]],   // [value_dim]
    device float* output             [[buffer(9)]],   // [output_dim]
    constant GdnRecurParams& p       [[buffer(10)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    uint h = tgpig;
    if (h >= p.num_value_heads) return;
    uint ratio = p.num_value_heads / p.num_key_heads;
    uint k_head = h / ratio;
    uint kd = p.key_dim;
    uint vd = p.value_dim;
    uint hd = p.hidden_size;

    threadgroup float sg_buf[4];
    threadgroup float q_tg[128];
    threadgroup float k_tg[128];

    // Beta = sigmoid(hidden @ in_proj_b[h]^T)
    device const half* wb = in_proj_b + h * hd;
    float bp = 0.0f;
    for (uint i = tid; i < hd; i += 128) bp += float(wb[i]) * hidden_in[i];
    bp = simd_sum(bp);
    if (simd_lane == 0) sg_buf[sgitg] = bp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { bp = 0; for (uint s = 0; s < 4; s++) bp += sg_buf[s]; sg_buf[0] = 1.0f / (1.0f + exp(-bp)); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float beta_val = sg_buf[0];
    // WAR guard: all simdgroups must finish reading sg_buf[0] (beta) above before the
    // alpha reduction below overwrites sg_buf[sgitg]. Without this barrier a fast
    // simdgroup can clobber slot 0 with an alpha partial while a lagging simdgroup is
    // still reading beta_val out of it.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Alpha
    device const half* wa = in_proj_a + h * hd;
    float ap = 0.0f;
    for (uint i = tid; i < hd; i += 128) ap += float(wa[i]) * hidden_in[i];
    ap = simd_sum(ap);
    if (simd_lane == 0) sg_buf[sgitg] = ap;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { ap = 0; for (uint s = 0; s < 4; s++) ap += sg_buf[s]; sg_buf[0] = ap; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float alpha_val = sg_buf[0];
    // WAR guard: all simdgroups must finish reading sg_buf[0] (alpha) above before the
    // Q-normalize reduction below overwrites sg_buf[sgitg].
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // L2 normalize Q
    // ADR-080 C1 fail-closed (#850): NaN * 0.0f == NaN under IEEE-754, so a NaN lane in
    // q_val survives a plain `q_val *= qs` even when qs is correctly 0.0f from the
    // sg_buf[0] guard. Assign the literal 0.0f directly to the whole vector on an
    // invalid (non-finite or near-zero) norm instead of multiplying a poisoned lane
    // through a zeroed reciprocal.
    float q_val = (tid < kd) ? conv_out[k_head * kd + tid] : 0.0f;
    float q_sq = simd_sum(q_val * q_val);
    if (simd_lane == 0) sg_buf[sgitg] = q_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { q_sq = 0; for (uint s = 0; s < 4; s++) q_sq += sg_buf[s]; sg_buf[0] = q_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool q_valid = isfinite(sg_buf[0]) && sg_buf[0] > 1e-12f;
    float qs = q_valid ? rsqrt(sg_buf[0]) : 0.0f;
    q_val = q_valid ? (q_val * qs) : 0.0f;
    if (tid < kd) q_tg[tid] = q_val;
    // WAR guard: all simdgroups must finish reading sg_buf[0] (the Q norm) above
    // before the K-normalize reduction below overwrites sg_buf[sgitg]. Without this
    // barrier a lagging simdgroup can read the K partial sum in place of the Q norm.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // L2 normalize K
    float k_val = (tid < kd) ? conv_out[p.q_total + k_head * kd + tid] : 0.0f;
    float k_sq = simd_sum(k_val * k_val);
    if (simd_lane == 0) sg_buf[sgitg] = k_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { k_sq = 0; for (uint s = 0; s < 4; s++) k_sq += sg_buf[s]; sg_buf[0] = k_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool k_valid = isfinite(sg_buf[0]) && sg_buf[0] > 1e-12f;
    float ks = k_valid ? rsqrt(sg_buf[0]) : 0.0f;
    k_val = k_valid ? (k_val * ks) : 0.0f;
    if (tid < kd) k_tg[tid] = k_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Decay gate
    float a = min(exp(a_log[h]), FLT_MAX);  // clamp +inf (parity w/ CPU gdn.rs): inf*0 = NaN poisons GDN state
    float sp = log(1.0f + exp(alpha_val + dt_bias[h]));
    float g = exp(-a * sp);

    // k[:] @ q[:] — shared dot product, same for all value rows in this head
    float kq_part = (tid < kd) ? (k_tg[tid] * q_tg[tid]) : 0.0f;
    kq_part = simd_sum(kq_part);
    if (simd_lane == 0) sg_buf[sgitg] = kq_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float kq = 0.0f; for (uint s = 0; s < 4; s++) kq += sg_buf[s]; sg_buf[0] = kq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_dot_q = sg_buf[0];
    // WAR guard: all simdgroups must finish reading sg_buf[0] (k_dot_q) above before the
    // gated-RMS-norm reduction below overwrites sg_buf[sgitg].
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Single pass: compute kv_mem and old_q together (saves one S read vs original)
    device float* S_h = S_all + h * vd * kd;
    float kv_mem = 0.0f;
    float old_q = 0.0f;
    if (tid < vd) {
        device float* sr = S_h + tid * kd;
        for (uint i = 0; i < kd; i++) {
            float s = sr[i];
            kv_mem += s * k_tg[i];
            old_q  += s * q_tg[i];
        }
    }

    // delta = (v - g * kv_mem) * beta
    float v_val = (tid < vd) ? conv_out[p.v_offset + h * vd + tid] : 0.0f;
    float delta = (v_val - g * kv_mem) * beta_val;

    // Update S; compute output algebraically — no re-read of S needed
    float out_val = 0.0f;
    if (tid < vd) {
        device float* sr = S_h + tid * kd;
        for (uint i = 0; i < kd; i++) sr[i] = fma(k_tg[i], delta, g * sr[i]);
        out_val = (g * old_q + delta * k_dot_q) * p.scale;
    }

    // Gated RMS norm: out = (x / rms(x)) * gamma * silu(z)
    float osq = simd_sum(out_val * out_val);
    if (simd_lane == 0) sg_buf[sgitg] = osq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { osq = 0; for (uint s = 0; s < 4; s++) osq += sg_buf[s]; sg_buf[0] = osq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(sg_buf[0] / float(vd) + p.eps);

    if (tid < vd) {
        float normed = out_val * inv_rms * norm_w[tid];
        float z_val = z_proj[h * vd + tid];
        float silu_z = z_val / (1.0f + exp(-z_val));
        output[h * vd + tid] = normed * silu_z;
    }
}

// ===== GDN: Specialized fused recurrence for Qwen3.6-27B (H2 + H4) =====
// Constexpr dims: kd=128, vd=128, hd=5120, ratio=3 (48vh / 16kh).
// H4: constexpr loop bounds enable compiler scheduling.
// H2: thread-local s_cache holds each thread's S row, eliminating the
//     second device-memory read that appeared in the original update loop
//     (`g * sr[i]` reread despite the comment claiming no reread needed).
// Same buffer layout as gdn_recurrence_fused — drop-in dispatch replacement.
kernel void gdn_recurrence_fused_q36(
    device float* S_all              [[buffer(0)]],
    device const float* conv_out     [[buffer(1)]],
    device const float* z_proj       [[buffer(2)]],
    device const float* hidden_in    [[buffer(3)]],
    device const half*  in_proj_b    [[buffer(4)]],
    device const half*  in_proj_a    [[buffer(5)]],
    device const float* a_log        [[buffer(6)]],
    device const float* dt_bias      [[buffer(7)]],
    device const float* norm_w       [[buffer(8)]],
    device float* output             [[buffer(9)]],
    constant GdnRecurParams& p       [[buffer(10)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kd = 128;
    constexpr uint vd = 128;
    constexpr uint hd = 5120;

    uint h = tgpig;
    if (h >= p.num_value_heads) return;
    uint k_head = h / 3u;  // ratio = num_value_heads / num_key_heads = 48 / 16 = 3

    threadgroup float sg_buf[4];
    threadgroup float q_tg[128];
    threadgroup float k_tg[128];

    // Beta = sigmoid(hidden @ in_proj_b[h])
    device const half* wb = in_proj_b + h * hd;
    float bp = 0.0f;
    for (uint i = tid; i < hd; i += 128) bp += float(wb[i]) * hidden_in[i];
    bp = simd_sum(bp);
    if (simd_lane == 0) sg_buf[sgitg] = bp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { bp = 0; for (uint s = 0; s < 4; s++) bp += sg_buf[s]; sg_buf[0] = 1.0f / (1.0f + exp(-bp)); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float beta_val = sg_buf[0];
    // WAR guard: all simdgroups must finish reading sg_buf[0] (beta) above before the
    // alpha reduction below overwrites sg_buf[sgitg].
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Alpha = hidden @ in_proj_a[h]
    device const half* wa = in_proj_a + h * hd;
    float ap = 0.0f;
    for (uint i = tid; i < hd; i += 128) ap += float(wa[i]) * hidden_in[i];
    ap = simd_sum(ap);
    if (simd_lane == 0) sg_buf[sgitg] = ap;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { ap = 0; for (uint s = 0; s < 4; s++) ap += sg_buf[s]; sg_buf[0] = ap; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float alpha_val = sg_buf[0];
    // WAR guard: all simdgroups must finish reading sg_buf[0] (alpha) above before the
    // Q-normalize reduction below overwrites sg_buf[sgitg].
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // L2 normalize Q — kd=128 == thread_count, no ARRAY-BOUNDS ternary needed, but the
    // norm-validity ternary below is still required (ADR-080 C1 fail-closed, #850): a
    // plain `q_val * qs` lets a NaN lane survive a correctly-zeroed qs (NaN * 0.0f ==
    // NaN under IEEE-754), so the whole vector is assigned 0.0f directly when invalid.
    float q_val = conv_out[k_head * kd + tid];
    float q_sq = simd_sum(q_val * q_val);
    if (simd_lane == 0) sg_buf[sgitg] = q_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { q_sq = 0; for (uint s = 0; s < 4; s++) q_sq += sg_buf[s]; sg_buf[0] = q_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool q_valid = isfinite(sg_buf[0]) && sg_buf[0] > 1e-12f;
    float qs = q_valid ? rsqrt(sg_buf[0]) : 0.0f;
    q_tg[tid] = q_valid ? (q_val * qs) : 0.0f;
    // WAR guard: all simdgroups must finish reading sg_buf[0] (the Q norm) above
    // before the K-normalize reduction below overwrites sg_buf[sgitg].
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // L2 normalize K
    float k_val = conv_out[p.q_total + k_head * kd + tid];
    float k_sq = simd_sum(k_val * k_val);
    if (simd_lane == 0) sg_buf[sgitg] = k_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { k_sq = 0; for (uint s = 0; s < 4; s++) k_sq += sg_buf[s]; sg_buf[0] = k_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool k_valid = isfinite(sg_buf[0]) && sg_buf[0] > 1e-12f;
    float ks_inv = k_valid ? rsqrt(sg_buf[0]) : 0.0f;
    k_tg[tid] = k_valid ? (k_val * ks_inv) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Decay gate
    float a = min(exp(a_log[h]), FLT_MAX);  // clamp +inf (parity w/ CPU gdn.rs): inf*0 = NaN poisons GDN state
    float sp = log(1.0f + exp(alpha_val + dt_bias[h]));
    float g = exp(-a * sp);

    // k dot q (scalar, same for all value rows in this head)
    float kq_part = k_tg[tid] * q_tg[tid];
    kq_part = simd_sum(kq_part);
    if (simd_lane == 0) sg_buf[sgitg] = kq_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float kq = 0.0f; for (uint s = 0; s < 4; s++) kq += sg_buf[s]; sg_buf[0] = kq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_dot_q = sg_buf[0];
    // WAR guard: all simdgroups must finish reading sg_buf[0] (k_dot_q) above before the
    // gated-RMS-norm reduction below overwrites sg_buf[sgitg].
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // H2: load S row into thread-local cache, compute kv_mem and old_q in one pass.
    // s_cache holds the values so the update loop reads cache instead of device memory.
    device float* S_h = S_all + h * vd * kd;
    float kv_mem = 0.0f;
    float old_q = 0.0f;
    float s_cache[128];
    device float* sr = S_h + tid * kd;
    for (uint i = 0; i < kd; i++) {
        float s = sr[i];
        s_cache[i] = s;
        kv_mem += s * k_tg[i];
        old_q  += s * q_tg[i];
    }

    float v_val = conv_out[p.v_offset + h * vd + tid];
    float delta = (v_val - g * kv_mem) * beta_val;

    // Update S using cached values — no second device read.
    for (uint i = 0; i < kd; i++) sr[i] = fma(k_tg[i], delta, g * s_cache[i]);
    float out_val = (g * old_q + delta * k_dot_q) * p.scale;

    // Gated RMS norm: out = (x / rms(x)) * gamma * silu(z)
    float osq = simd_sum(out_val * out_val);
    if (simd_lane == 0) sg_buf[sgitg] = osq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { osq = 0; for (uint s = 0; s < 4; s++) osq += sg_buf[s]; sg_buf[0] = osq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(sg_buf[0] / float(vd) + p.eps);

    float normed = out_val * inv_rms * norm_w[tid];
    float z_val = z_proj[h * vd + tid];
    float silu_z = z_val / (1.0f + exp(-z_val));
    output[h * vd + tid] = normed * silu_z;
}

// ===== GDN H1+H3: Three-kernel sharded path for Qwen3.6-27B =====
// H3: gdn_precompute_keys dispatches per VALUE head (num_value_heads TGs × 128 threads).
//     Scratch uses a split layout (floats):
//       Key section:   key_stride = 2*kd+1; key_base(kh) = kh * key_stride
//                      [q_norm(0..kd) | k_norm(kd..2kd) | k_dot_q]  — one row per key head
//       Value section: value_base = num_key_heads * key_stride; stride = 2
//                      [beta | g]  — one slot per value head
//       Total size: num_key_heads * (2*kd+1) + num_value_heads * 2
// H1: gdn_recurrence_sharded expands from 48 TG/layer to 1536 TG/layer (32x48).
//     Each TG owns 4 S-rows x 32 key lanes; simd_sum reduces kv/old_q within a row.
// gdn_norm_silu: RMSNorm + SiLU gate on raw_out, writes result to gdn_qkvz for output proj.

kernel void gdn_precompute_keys(
    device const float* conv_out     [[buffer(0)]],
    device const float* hidden_in    [[buffer(1)]],
    device const half*  in_proj_b    [[buffer(2)]],  // [num_value_heads, hidden_size]
    device const half*  in_proj_a    [[buffer(3)]],  // [num_value_heads, hidden_size]
    device const float* a_log        [[buffer(4)]],  // [num_value_heads]
    device const float* dt_bias      [[buffer(5)]],  // [num_value_heads]
    device float* key_scratch        [[buffer(6)]],
    constant GdnRecurParams& p       [[buffer(7)]],
    uint h         [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint sgitg     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kd = 128;
    constexpr uint hd = 5120;

    if (h >= p.num_value_heads) return;
    uint ratio  = p.num_value_heads / p.num_key_heads;
    uint k_head = h / ratio;

    threadgroup float sg_buf[4];

    // Beta = sigmoid(hidden @ in_proj_b[h])  — per VALUE head
    device const half* wb = in_proj_b + h * hd;
    float bp = 0.0f;
    for (uint i = tid; i < hd; i += kd) bp += float(wb[i]) * hidden_in[i];
    bp = simd_sum(bp);
    if (simd_lane == 0) sg_buf[sgitg] = bp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { bp = 0; for (uint s = 0; s < 4; s++) bp += sg_buf[s]; sg_buf[0] = 1.0f / (1.0f + exp(-bp)); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float beta_val = sg_buf[0];
    // WAR guard: all simdgroups must finish reading sg_buf[0] (beta) above before the
    // alpha reduction below overwrites sg_buf[sgitg].
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Alpha = hidden @ in_proj_a[h]  — per VALUE head
    device const half* wa = in_proj_a + h * hd;
    float ap = 0.0f;
    for (uint i = tid; i < hd; i += kd) ap += float(wa[i]) * hidden_in[i];
    ap = simd_sum(ap);
    if (simd_lane == 0) sg_buf[sgitg] = ap;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { ap = 0; for (uint s = 0; s < 4; s++) ap += sg_buf[s]; sg_buf[0] = ap; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float alpha_val = sg_buf[0];
    // WAR guard: all simdgroups must finish reading sg_buf[0] (alpha) above before the
    // Q-normalize reduction below overwrites sg_buf[sgitg].
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Q/K stay per KEY head (repeat_interleave: k_head = h / ratio)
    // ADR-080 C1 fail-closed (#850): assign 0.0f to the whole vector directly on an
    // invalid (non-finite or near-zero) norm, never `value * guarded_zero_reciprocal`
    // (NaN * 0.0f == NaN under IEEE-754).
    float q_val = conv_out[k_head * kd + tid];
    float q_sq  = simd_sum(q_val * q_val);
    if (simd_lane == 0) sg_buf[sgitg] = q_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { q_sq = 0; for (uint s = 0; s < 4; s++) q_sq += sg_buf[s]; sg_buf[0] = q_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool q_valid      = isfinite(sg_buf[0]) && sg_buf[0] > 1e-12f;
    float qs          = q_valid ? rsqrt(sg_buf[0]) : 0.0f;
    float q_norm_val  = q_valid ? (q_val * qs) : 0.0f;
    // WAR guard: all simdgroups must finish reading sg_buf[0] (the Q norm) above
    // before the K-normalize reduction below overwrites sg_buf[sgitg].
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float k_val = conv_out[p.q_total + k_head * kd + tid];
    float k_sq  = simd_sum(k_val * k_val);
    if (simd_lane == 0) sg_buf[sgitg] = k_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { k_sq = 0; for (uint s = 0; s < 4; s++) k_sq += sg_buf[s]; sg_buf[0] = k_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool k_valid      = isfinite(sg_buf[0]) && sg_buf[0] > 1e-12f;
    float ks_inv      = k_valid ? rsqrt(sg_buf[0]) : 0.0f;
    float k_norm_val  = k_valid ? (k_val * ks_inv) : 0.0f;
    // WAR guard: all simdgroups must finish reading sg_buf[0] (the K norm) above
    // before the k_dot_q reduction below overwrites sg_buf[sgitg].
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Decay gate — per VALUE head
    float a  = min(exp(a_log[h]), FLT_MAX);  // clamp +inf (parity w/ CPU gdn.rs): inf*0 = NaN poisons GDN state
    float sp = log(1.0f + exp(alpha_val + dt_bias[h]));
    float g  = exp(-a * sp);

    float kq_part = k_norm_val * q_norm_val;
    kq_part = simd_sum(kq_part);
    if (simd_lane == 0) sg_buf[sgitg] = kq_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float kq = 0.0f; for (uint s = 0; s < 4; s++) kq += sg_buf[s]; sg_buf[0] = kq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_dot_q = sg_buf[0];

    // Split scratch layout:
    //   Key section:   key_stride = 2*kd+1; one row per key head → [q_norm | k_norm | k_dot_q]
    //   Value section: value_base = num_key_heads * key_stride; one slot per value head → [beta | g]
    constexpr uint key_stride = 2 * kd + 1;
    uint value_base = p.num_key_heads * key_stride;

    // Write beta/g for every value head (each TG writes its own slot — no conflict)
    device float* vs = key_scratch + value_base + h * 2;
    if (tid == 0) {
        vs[0] = beta_val;
        vs[1] = g;
    }

    // Write q_norm/k_norm/k_dot_q only from the unique representative value head for this key head.
    // For each kh, only h = kh * ratio satisfies (h % ratio == 0) — no two TGs write the same row.
    if ((h % ratio) == 0) {
        device float* ks_out = key_scratch + k_head * key_stride;
        ks_out[tid]      = q_norm_val;
        ks_out[kd + tid] = k_norm_val;
        if (tid == 0) ks_out[2 * kd] = k_dot_q;
    }
}

// Grid: (val_d/4, num_value_heads, 1) = (32, 48, 1). Threads: (32, 4, 1).
// tgpig.x = row_block (0..31), tgpig.y = value head (0..47).
// tid2.x = key lane (0..31), tid2.y = row within block (0..3).
// Each lane owns key indices: lane, lane+32, lane+64, lane+96.
kernel void gdn_recurrence_sharded(
    device float*        S_all       [[buffer(0)]],
    device const float*  conv_out    [[buffer(1)]],
    device const float*  key_scratch [[buffer(2)]],
    device float*        raw_out     [[buffer(3)]],
    constant GdnRecurParams& p       [[buffer(4)]],
    uint2 tgpig    [[threadgroup_position_in_grid]],
    uint2 tid2     [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    constexpr uint kd = 128;
    constexpr uint vd = 128;

    uint h   = tgpig.y;
    uint row = tgpig.x * 4 + tid2.y;
    if (row >= vd) return;

    uint k_head = h / (p.num_value_heads / p.num_key_heads);
    device float*       sr = S_all       + h * vd * kd + row * kd;

    // Split scratch layout (mirrors gdn_precompute_keys):
    //   Key section:   key_stride = 2*kd+1; ks[0..kd)=q_norm, ks[kd..2kd)=k_norm, ks[2*kd]=k_dot_q
    //   Value section: value_base = num_key_heads * key_stride; vs[0]=beta, vs[1]=g
    constexpr uint key_stride = 2 * kd + 1;
    uint value_base = p.num_key_heads * key_stride;
    device const float* ks = key_scratch + k_head * key_stride;
    device const float* vs = key_scratch + value_base + h * 2;

    float kv    = 0.0f;
    float old_q = 0.0f;
    float s_vals[4];
    uint  i_vals[4];
    uint  n_vals = 0;

    for (uint i = tid2.x; i < kd; i += 32) {
        float s   = sr[i];
        s_vals[n_vals] = s;
        i_vals[n_vals] = i;
        n_vals++;
        kv    += s * ks[kd + i];
        old_q += s * ks[i];
    }

    kv    = simd_sum(kv);
    old_q = simd_sum(old_q);

    float beta    = vs[0];
    float g       = vs[1];
    float k_dot_q = ks[2 * kd];
    float v       = conv_out[p.v_offset + h * vd + row];
    float delta   = (v - g * kv) * beta;

    for (uint n = 0; n < n_vals; n++) {
        sr[i_vals[n]] = fma(ks[kd + i_vals[n]], delta, g * s_vals[n]);
    }

    if (tid2.x == 0) {
        raw_out[h * vd + row] = (g * old_q + delta * k_dot_q) * p.scale;
    }
}

// Grid: (num_value_heads, 1, 1) = (48, 1, 1). Threads: (128, 1, 1).
// Reads raw_out (recurrence output) and z_proj (Z gate from gdn_qkvz at z_byte_off).
// Writes gated RMSNorm result back to gdn_qkvz for the downstream output projection.
kernel void gdn_norm_silu(
    device const float* raw_out  [[buffer(0)]],
    device const float* z_proj   [[buffer(1)]],
    device const float* norm_w   [[buffer(2)]],
    device float*       output   [[buffer(3)]],
    constant GdnRecurParams& p   [[buffer(4)]],
    uint h      [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint sgitg  [[simdgroup_index_in_threadgroup]])
{
    constexpr uint vd = 128;

    threadgroup float sg_buf[4];

    float out_val = raw_out[h * vd + tid];

    float osq = simd_sum(out_val * out_val);
    if (simd_lane == 0) sg_buf[sgitg] = osq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { osq = 0; for (uint s = 0; s < 4; s++) osq += sg_buf[s]; sg_buf[0] = osq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(sg_buf[0] / float(vd) + p.eps);

    float normed = out_val * inv_rms * norm_w[tid];
    float z_val  = z_proj[h * vd + tid];
    float silu_z = z_val / (1.0f + exp(-z_val));
    output[h * vd + tid] = normed * silu_z;
}

// ===== Q8_0 batch GEMM: C[M,N] = A[M,K] @ Q8_B[N, K/32*34]^T =====
// Extends the NR=2 GEMV pattern to batch M>1 with NM=4 M-rows per TG.
// Each TG handles NR=2 output columns × NM=4 input rows.
// B (weight) data loaded once per (ib,ri) is reused across NM rows (L1 cache).
// Dispatch: thread_groups=(ceil(N/2), ceil(M/4), 1), threads=(32, 4, 1)
kernel void gemm_q8(
    device const float* A    [[buffer(0)]],
    device const char*  B    [[buffer(1)]],
    device float*       C    [[buffer(2)]],
    constant uint& M         [[buffer(3)]],
    constant uint& N         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR = 2;
    const uint NM = 4;
    const uint NSG = 4;
    const uint nb = K / 32;
    const uint row_bytes = nb * 34;
    const uint first_col = tgpig.x * NR;
    const uint first_row = tgpig.y * NM;
    const uint ix = tiisg / 4;
    const uint il = tiisg % 4;

    float sumf[NR * NM];
    for (uint i = 0; i < NR * NM; i++) sumf[i] = 0.0f;

    const uint ib_start = sgitg * 8 + ix;
    const uint ib_stride = NSG * 8;

    for (uint ib = ib_start; ib < nb; ib += ib_stride) {
        for (uint mi = 0; mi < NM; mi++) {
            uint m = first_row + mi;
            if (m >= M) continue;

            device const float* yb = A + m * K + ib * 32 + il * 8;
            float yl[8];
            for (uint i = 0; i < 8; i++) yl[i] = yb[i];

            for (uint ri = 0; ri < NR; ri++) {
                uint col = first_col + ri;
                if (col >= N) continue;
                device const char* base = B + col * row_bytes + ib * 34;
                device const char* qs = base + 2 + il * 8;
                half d = *((device const half*)base);
                float sumq = 0.0f;
                for (uint i = 0; i < 8; i++) sumq += float(qs[i]) * yl[i];
                sumf[mi * NR + ri] += sumq * float(d);
            }
        }
    }

    for (uint i = 0; i < NR * NM; i++) sumf[i] = simd_sum(sumf[i]);

    threadgroup float shared[NR * NM][4];
    if (tiisg == 0) {
        for (uint i = 0; i < NR * NM; i++) shared[i][sgitg] = sumf[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        for (uint mi = 0; mi < NM; mi++) {
            uint m = first_row + mi;
            if (m >= M) continue;
            for (uint ri = 0; ri < NR; ri++) {
                uint col = first_col + ri;
                if (col < N) {
                    uint idx = mi * NR + ri;
                    C[m * N + col] = shared[idx][0] + shared[idx][1] +
                                     shared[idx][2] + shared[idx][3];
                }
            }
        }
    }
}

// ===== Q4 (asymmetric) batch GEMM: Y[M,N] = X[M,K] @ Q4_W[N, K/32*20]^T =====
// Scalar fallback cloned from gemm_q8: NM=4 M rows, NR=2 output columns,
// NSG=4 SIMD groups. Q4 blocks are [f16 scale][f16 bias][16 packed nibble bytes].
// Dequantization: w = nibble * scale + bias (asymmetric min-max).
// Dispatch: thread_groups=(ceil(N/2), ceil(M/4), 1), threads=(32, 4, 1)
kernel void gemm_q4(
    device const uchar* QW   [[buffer(0)]],
    device const float* X    [[buffer(1)]],
    device float*       Y    [[buffer(2)]],
    constant uint& M         [[buffer(3)]],
    constant uint& N         [[buffer(4)]],
    constant uint& K         [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint NR = 2;
    const uint NM = 4;
    const uint NSG = 4;
    const uint nb = K / 32;
    const uint row_bytes = nb * 20;
    const uint first_col = tgpig.x * NR;
    const uint first_row = tgpig.y * NM;
    const uint ix = tiisg / 4;
    const uint il = tiisg % 4;

    float sumf[NR * NM];
    for (uint i = 0; i < NR * NM; i++) sumf[i] = 0.0f;

    const uint ib_start = sgitg * 8 + ix;
    const uint ib_stride = NSG * 8;

    for (uint ib = ib_start; ib < nb; ib += ib_stride) {
        for (uint mi = 0; mi < NM; mi++) {
            uint m = first_row + mi;
            if (m >= M) continue;

            device const float* xb = X + m * K + ib * 32 + il * 8;
            float xv[8];
            float xv_sum = 0.0f;
            for (uint i = 0; i < 8; i++) { xv[i] = xb[i]; xv_sum += xb[i]; }

            for (uint ri = 0; ri < NR; ri++) {
                uint col = first_col + ri;
                if (col >= N) continue;

                uint block_start = col * row_bytes + ib * 20;
                ushort scale_bits = ushort(QW[block_start]) |
                    (ushort(QW[block_start + 1]) << 8);
                ushort bias_bits = ushort(QW[block_start + 2]) |
                    (ushort(QW[block_start + 3]) << 8);
                float d = float(as_type<half>(scale_bits));
                float b = float(as_type<half>(bias_bits));

                // Asymmetric: w = nibble * d + b. The bias contribution
                // factors out as b * sum(x[8]).
                float sumq_dot = 0.0f;
                for (uint byte_i = 0; byte_i < 4; byte_i++) {
                    uchar packed = QW[block_start + 4 + il * 4 + byte_i];
                    int q0 = int(packed & 0x0f);
                    int q1 = int(packed >> 4);
                    sumq_dot += float(q0) * xv[byte_i * 2] +
                                float(q1) * xv[byte_i * 2 + 1];
                }

                sumf[mi * NR + ri] += sumq_dot * d + b * xv_sum;
            }
        }
    }

    for (uint i = 0; i < NR * NM; i++) sumf[i] = simd_sum(sumf[i]);

    threadgroup float shared[NR * NM][4];
    if (tiisg == 0) {
        for (uint i = 0; i < NR * NM; i++) shared[i][sgitg] = sumf[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        for (uint mi = 0; mi < NM; mi++) {
            uint m = first_row + mi;
            if (m >= M) continue;
            for (uint ri = 0; ri < NR; ri++) {
                uint col = first_col + ri;
                if (col < N) {
                    uint idx = mi * NR + ri;
                    Y[m * N + col] = shared[idx][0] + shared[idx][1] +
                                     shared[idx][2] + shared[idx][3];
                }
            }
        }
    }
}

// ===== GPU Top-K Logit Selection =====
// Selects the top-k (logit, token_id) pairs from a flat f32 logits buffer.
// Two-pass algorithm: per-tile first pass, then iterative merge passes.
// MAX_TOP_K = 256; TOPK_TILE = 1280 (256 threads × 5 items each).

struct TopKCandidate {
    float logit;
    uint  token_id;
};

inline TopKCandidate topk_sentinel() {
    return TopKCandidate{-INFINITY, UINT_MAX};
}

// Returns true if a should rank before b (higher logit first; ties → lower id).
inline bool topk_better(TopKCandidate a, TopKCandidate b) {
    if (isnan(a.logit)) return false;
    if (isnan(b.logit)) return true;
    if (a.logit != b.logit) return a.logit > b.logit;
    return a.token_id < b.token_id;
}

// Full descending bitonic sort of tg[0..2048] with 256 threads.
// Barriers are placed after each sub-stage distance level.
inline void bitonic_sort_desc_2048(threadgroup TopKCandidate* tg, uint tid) {
    for (uint step = 2u; step <= 2048u; step <<= 1u) {
        for (uint dist = step >> 1u; dist > 0u; dist >>= 1u) {
            for (uint idx = tid; idx < 2048u; idx += 256u) {
                uint ixj = idx ^ dist;
                if (ixj > idx) {
                    bool swap_cond;
                    if ((idx & step) != 0u) {
                        // ascending sub-net: put smaller at lower index
                        swap_cond = topk_better(tg[idx], tg[ixj]);
                    } else {
                        // descending sub-net: put larger at lower index
                        swap_cond = topk_better(tg[ixj], tg[idx]);
                    }
                    if (swap_cond) {
                        TopKCandidate tmp = tg[idx];
                        tg[idx] = tg[ixj];
                        tg[ixj] = tmp;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// First pass: each threadgroup processes 1280 logits and writes top_k candidates.
// Grid: (ceil(vocab_size/1280), 1, 1)  Threads: (256, 1, 1)
kernel void logits_topk_first_pass(
    device const float*   logits      [[buffer(0)]],
    device TopKCandidate* partial_out [[buffer(1)]],
    constant uint&        vocab_size  [[buffer(2)]],
    constant uint&        top_k       [[buffer(3)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup TopKCandidate tg[2048];
    const uint TILE = 1280u;
    uint base = tgid * TILE;

    // Load 1280 logits (5 per thread) into tg[0..1280].
    for (uint j = 0u; j < 5u; ++j) {
        uint slot = tid + j * 256u;
        uint gidx = base + slot;
        if (gidx < vocab_size) {
            float v = logits[gidx];
            tg[slot] = isnan(v) ? topk_sentinel() : TopKCandidate{v, gidx};
        } else {
            tg[slot] = topk_sentinel();
        }
    }
    // Pad tg[1280..2048] with sentinels.
    for (uint slot = TILE + tid; slot < 2048u; slot += 256u) {
        tg[slot] = topk_sentinel();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bitonic_sort_desc_2048(tg, tid);

    // Write first top_k candidates to the output scratch buffer.
    for (uint out = tid; out < top_k; out += 256u) {
        partial_out[tgid * top_k + out] = tg[out];
    }
}

// Merge pass: merges fan_in groups of top_k candidates into one group of top_k.
// Grid: (ceil(input_groups/fan_in), 1, 1)  Threads: (256, 1, 1)
kernel void logits_topk_merge_pass(
    device const TopKCandidate* in_buf      [[buffer(0)]],
    device       TopKCandidate* out_buf     [[buffer(1)]],
    constant uint&              input_groups [[buffer(2)]],
    constant uint&              top_k        [[buffer(3)]],
    constant uint&              fan_in       [[buffer(4)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup TopKCandidate tg[2048];
    uint start_group = tgid * fan_in;
    uint max_items   = min(fan_in * top_k, 2048u);

    // Load fan_in * top_k candidates (padded to 2048 with sentinels).
    for (uint slot = tid; slot < 2048u; slot += 256u) {
        if (slot < max_items) {
            uint g  = slot / top_k;
            uint ci = slot % top_k;
            uint global_group = start_group + g;
            tg[slot] = (global_group < input_groups)
                       ? in_buf[global_group * top_k + ci]
                       : topk_sentinel();
        } else {
            tg[slot] = topk_sentinel();
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bitonic_sort_desc_2048(tg, tid);

    // Write top_k candidates to output.
    for (uint out = tid; out < top_k; out += 256u) {
        out_buf[tgid * top_k + out] = tg[out];
    }
}

// ===== k=1 Dedicated Argmax Reduction =====
// Replaces bitonic sort for k=1. Target: <100 µs for 248,320 logits.

inline TopKCandidate simd_argmax(TopKCandidate v, uint simd_width) {
    for (uint off = simd_width >> 1u; off > 0u; off >>= 1u) {
        TopKCandidate other;
        other.logit    = simd_shuffle_down(v.logit,    off);
        other.token_id = simd_shuffle_down(v.token_id, off);
        if (topk_better(other, v)) v = other;
    }
    return v;
}

// First pass: each 1024-thread group produces one winner.
// Grid: (ceil(vocab_size/1024), 1, 1)  Threads: (1024, 1, 1)
kernel void logits_argmax_first(
    device const float*   logits     [[buffer(0)]],
    device TopKCandidate* group_out  [[buffer(1)]],
    constant uint&        vocab_size [[buffer(2)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lane       [[thread_index_in_simdgroup]],
    uint sgid       [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]])
{
    threadgroup TopKCandidate sg_winners[32];
    TopKCandidate best = topk_sentinel();
    uint idx = tgid * 1024u + tid;
    if (idx < vocab_size) {
        float v = logits[idx];
        best = isnan(v) ? topk_sentinel() : TopKCandidate{v, idx};
    }
    best = simd_argmax(best, simd_width);
    if (lane == 0u) sg_winners[sgid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TopKCandidate tg_best = (tid < 32u) ? sg_winners[tid] : topk_sentinel();
    if (sgid == 0u) {
        tg_best = simd_argmax(tg_best, simd_width);
        if (lane == 0u) group_out[tgid] = tg_best;
    }
}

// Merge pass: one 1024-thread group reduces all group winners to one candidate.
// Grid: (1, 1, 1)  Threads: (1024, 1, 1)
kernel void logits_argmax_merge(
    device const TopKCandidate* group_in    [[buffer(0)]],
    device TopKCandidate*       out         [[buffer(1)]],
    constant uint&              group_count [[buffer(2)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint lane       [[thread_index_in_simdgroup]],
    uint sgid       [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]])
{
    threadgroup TopKCandidate sg_winners[32];
    TopKCandidate best = topk_sentinel();
    for (uint i = tid; i < group_count; i += 1024u) {
        TopKCandidate c = group_in[i];
        if (topk_better(c, best)) best = c;
    }
    best = simd_argmax(best, simd_width);
    if (lane == 0u) sg_winners[sgid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TopKCandidate final_best = (tid < 32u) ? sg_winners[tid] : topk_sentinel();
    if (sgid == 0u) {
        final_best = simd_argmax(final_best, simd_width);
        if (lane == 0u) out[0] = final_best;
    }
}

// ===== k>1 Fast Top-K First Pass (1024-element tile) =====
// Halves threadgroup memory vs 2048-element version (8KB vs 16KB).
// Improves occupancy ~2× and reduces sort stages 66 -> 55.

inline void bitonic_sort_desc_1024(threadgroup TopKCandidate* tg, uint tid) {
    for (uint step = 2u; step <= 1024u; step <<= 1u) {
        for (uint dist = step >> 1u; dist > 0u; dist >>= 1u) {
            for (uint idx = tid; idx < 1024u; idx += 256u) {
                uint ixj = idx ^ dist;
                if (ixj > idx) {
                    bool swap_cond;
                    if ((idx & step) != 0u) {
                        swap_cond = topk_better(tg[idx], tg[ixj]);
                    } else {
                        swap_cond = topk_better(tg[ixj], tg[idx]);
                    }
                    if (swap_cond) {
                        TopKCandidate tmp = tg[idx];
                        tg[idx] = tg[ixj];
                        tg[ixj] = tmp;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Fast first pass: tile=1024, 8KB threadgroup memory, no padding waste.
// Grid: (ceil(vocab_size/1024), 1, 1)  Threads: (256, 1, 1)
kernel void logits_topk_fast_first(
    device const float*   logits      [[buffer(0)]],
    device TopKCandidate* partial_out [[buffer(1)]],
    constant uint&        vocab_size  [[buffer(2)]],
    constant uint&        top_k       [[buffer(3)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup TopKCandidate tg[1024];
    const uint TILE = 1024u;
    uint base = tgid * TILE;
    for (uint j = 0u; j < 4u; ++j) {
        uint slot = tid + j * 256u;
        uint gidx = base + slot;
        if (gidx < vocab_size) {
            float v = logits[gidx];
            tg[slot] = isnan(v) ? topk_sentinel() : TopKCandidate{v, gidx};
        } else {
            tg[slot] = topk_sentinel();
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bitonic_sort_desc_1024(tg, tid);
    for (uint out = tid; out < top_k; out += 256u) {
        partial_out[tgid * top_k + out] = tg[out];
    }
}

// ===== lm_head Two-Stage Block Top-K (issue #171) =====
// Stage 1: fuses the lm_head GEMV with a block-local exact argmax/top-K
// selection so the sampler never materializes the full [vocab_size] logit
// buffer. One threadgroup owns ROWS_PER_TG consecutive vocab rows ("a tile");
// each of the ROWS_PER_TG threads computes one row's full HIDDEN-length dot
// product, then the whole tile is bitonic-sorted and the top LOCAL_K local
// winners are emitted as compact (logit, token_id) pairs. Stage 2 (the
// existing `logits_argmax_merge` / `logits_topk_merge_pass` kernels, unchanged)
// then reduces the compact candidates from every tile to the global result.
//
// EXACTNESS: Stage 2 only sees the union of tile-local lists. For every
// global top-K token t, Stage 1 for tile(t) must emit t in that tile's exact
// local top-K list. Because every row in the tile participates in a full
// bitonic sort of the tile (not an approximate reduction), the local list is
// exact by construction. If LOCAL_K were ever reduced below the requested K,
// or the tile selection made approximate, a distribution where all global
// winners cluster inside one tile would silently drop valid global top-K
// items — this is pinned by an adversarial clustered-tile test.
//
// Requires ROWS_PER_TG == 256 and dispatch threads == (256, 1, 1): the tile
// is bitonic-sorted via the existing `bitonic_sort_desc_1024` helper, whose
// inner stride is hardcoded to the 256-thread launch config.
constant uint HIDDEN      [[function_constant(0)]];
constant uint ROWS_PER_TG [[function_constant(1)]];
constant uint LOCAL_K     [[function_constant(2)]];

// Dispatch: threadgroups=(ceil(vocab_size/ROWS_PER_TG), 1, 1), threads=(256, 1, 1)
kernel void lm_head_block_topk_f16(
    device const float*   hidden_in  [[buffer(0)]],
    device const half*    weight     [[buffer(1)]],
    device TopKCandidate* partial_out [[buffer(2)]],
    constant uint&        vocab_size [[buffer(3)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup TopKCandidate tg[1024];

    // DISPATCH GEOMETRY: tg_pos.x (tgid) owns vocab rows
    // [tgid*ROWS_PER_TG, tgid*ROWS_PER_TG+ROWS_PER_TG). Tail guard: rows past
    // vocab_size must rank last (sentinel), never silently win a slot.
    uint row = tgid * ROWS_PER_TG + tid;
    TopKCandidate cand = topk_sentinel();
    if (row < vocab_size) {
        device const half* wrow = weight + (ulong)row * (ulong)HIDDEN;
        float dot = 0.0f;
        for (uint i = 0; i < HIDDEN; i++) {
            dot += hidden_in[i] * float(wrow[i]);
        }
        cand = isnan(dot) ? topk_sentinel() : TopKCandidate{dot, row};
    }
    tg[tid] = cand;
    for (uint slot = 256u + tid; slot < 1024u; slot += 256u) {
        tg[slot] = topk_sentinel();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bitonic_sort_desc_1024(tg, tid);

    for (uint out = tid; out < LOCAL_K; out += 256u) {
        partial_out[tgid * LOCAL_K + out] = tg[out];
    }
}

// Q4_0 variant of `lm_head_block_topk_f16`. Dequant matches `gemv_q4_decode`:
// asymmetric blocks of 32 weights, 20 bytes each (f16 scale + f16 bias + 16
// packed-nibble bytes), w = nibble * scale + bias.
// Dispatch: threadgroups=(ceil(vocab_size/ROWS_PER_TG), 1, 1), threads=(256, 1, 1)
kernel void lm_head_block_topk_q4(
    device const float*   hidden_in  [[buffer(0)]],
    device const uchar*   qweight    [[buffer(1)]],
    device TopKCandidate* partial_out [[buffer(2)]],
    constant uint&        vocab_size [[buffer(3)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup TopKCandidate tg[1024];

    uint row = tgid * ROWS_PER_TG + tid;
    TopKCandidate cand = topk_sentinel();
    if (row < vocab_size) {
        const uint nb = HIDDEN / 32u;
        const uint row_bytes = nb * 20u;
        device const uchar* base_row = qweight + (ulong)row * (ulong)row_bytes;
        float dot = 0.0f;
        for (uint blk = 0; blk < nb; blk++) {
            device const uchar* base = base_row + blk * 20u;
            float d    = float(*((device const half*)base));
            float bias = float(*((device const half*)(base + 2)));
            device const uchar* qs = base + 4;
            uint x_base = blk * 32u;
            for (uint byte_i = 0; byte_i < 16u; byte_i++) {
                uchar packed = qs[byte_i];
                float n0 = float(packed & 0x0fu);
                float n1 = float(packed >> 4u);
                dot += hidden_in[x_base + byte_i * 2u] * (n0 * d + bias);
                dot += hidden_in[x_base + byte_i * 2u + 1u] * (n1 * d + bias);
            }
        }
        cand = isnan(dot) ? topk_sentinel() : TopKCandidate{dot, row};
    }
    tg[tid] = cand;
    for (uint slot = 256u + tid; slot < 1024u; slot += 256u) {
        tg[slot] = topk_sentinel();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bitonic_sort_desc_1024(tg, tid);

    for (uint out = tid; out < LOCAL_K; out += 256u) {
        partial_out[tgid * LOCAL_K + out] = tg[out];
    }
}

// ===== Hierarchical k=50 SIMD-Group Tournament =====
// Two-stage exact selection without bitonic sort or threadgroup barriers in the
// selection loop.  Only one threadgroup barrier is used (between stage 1 and 2).
//
// First pass: logits_topk_select50_first
//   Grid: (ceil(vocab_size/1024), 1, 1)  Threads: (256, 1, 1)
//   Each 256-thread group processes 1024 logits.  8 SIMD groups (width=32) each
//   independently select top-50 from their 128-element sub-tile, then SIMD group 0
//   merges the 8*50 candidates to the tile top-50.
//
// Merge pass: logits_topk_select50_merge
//   Grid: (ceil(groups/fan_in), 1, 1)  Threads: (256, 1, 1)  fan_in=16
//   SIMD group 0 tournament-selects top-50 from up to fan_in*50 candidates.
//   Other threads return immediately (no barriers needed).

constant uint TOPK50 = 50u;
constant uint SELECT50_THREADS = 256u;
constant uint SELECT50_ITEMS_PER_THREAD = 4u;
constant uint SELECT50_TILE = 1024u;
constant uint SELECT50_MAX_SIMDGROUPS = 16u;
constant uint SELECT50_MERGE_LOCAL_MAX = 50u;

kernel void logits_topk_select50_first(
    device const float*   logits      [[buffer(0)]],
    device TopKCandidate* partial_out [[buffer(1)]],
    constant uint&        vocab_size  [[buffer(2)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lane       [[thread_index_in_simdgroup]],
    uint sgid       [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]])
{
    threadgroup TopKCandidate sg_top[SELECT50_MAX_SIMDGROUPS * TOPK50];

    if (simd_width < 16u) return;

    uint num_sg   = SELECT50_THREADS / simd_width;
    uint sg_tile  = simd_width * SELECT50_ITEMS_PER_THREAD;
    uint base     = tgid * SELECT50_TILE + sgid * sg_tile;

    thread TopKCandidate local[SELECT50_ITEMS_PER_THREAD];
    uint selected_mask = 0u;

    for (uint j = 0u; j < SELECT50_ITEMS_PER_THREAD; ++j) {
        uint idx = base + lane + j * simd_width;
        if (idx < vocab_size) {
            float v = logits[idx];
            local[j] = isnan(v) ? topk_sentinel() : TopKCandidate{v, idx};
        } else {
            local[j] = topk_sentinel();
        }
    }

    // Stage 1: each SIMD group selects its exact top-50 from its 128-element sub-tile.
    for (uint rank = 0u; rank < TOPK50; ++rank) {
        TopKCandidate lane_best  = topk_sentinel();
        uint          lane_best_j = UINT_MAX;
        for (uint j = 0u; j < SELECT50_ITEMS_PER_THREAD; ++j) {
            if (((selected_mask >> j) & 1u) == 0u && topk_better(local[j], lane_best)) {
                lane_best   = local[j];
                lane_best_j = j;
            }
        }

        TopKCandidate winner = simd_argmax(lane_best, simd_width);
        winner.logit    = simd_broadcast(winner.logit,    0u);
        winner.token_id = simd_broadcast(winner.token_id, 0u);

        if (lane == 0u) {
            sg_top[sgid * TOPK50 + rank] = winner;
        }
        if (lane_best_j != UINT_MAX && local[lane_best_j].token_id == winner.token_id) {
            selected_mask |= (1u << lane_best_j);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: SIMD group 0 merges all sub-tile top-50 lists to the tile top-50.
    if (sgid == 0u) {
        thread TopKCandidate merge_vals[SELECT50_MERGE_LOCAL_MAX];
        ulong merge_selected   = 0ul;
        uint  candidate_count  = num_sg * TOPK50;

        for (uint j = 0u; j < SELECT50_MERGE_LOCAL_MAX; ++j) {
            uint slot    = lane + j * simd_width;
            merge_vals[j] = (slot < candidate_count) ? sg_top[slot] : topk_sentinel();
        }

        for (uint rank = 0u; rank < TOPK50; ++rank) {
            TopKCandidate lane_best   = topk_sentinel();
            uint          lane_best_j  = UINT_MAX;
            for (uint j = 0u; j < SELECT50_MERGE_LOCAL_MAX; ++j) {
                if (((merge_selected >> j) & 1ul) == 0ul && topk_better(merge_vals[j], lane_best)) {
                    lane_best   = merge_vals[j];
                    lane_best_j = j;
                }
            }

            TopKCandidate winner = simd_argmax(lane_best, simd_width);
            winner.logit    = simd_broadcast(winner.logit,    0u);
            winner.token_id = simd_broadcast(winner.token_id, 0u);

            if (lane == 0u) {
                partial_out[tgid * TOPK50 + rank] = winner;
            }
            if (lane_best_j != UINT_MAX && merge_vals[lane_best_j].token_id == winner.token_id) {
                merge_selected |= (1ul << lane_best_j);
            }
        }
    }
}

// Merge pass: tournament-select top-50 from up to fan_in first-pass groups.
// Only SIMD group 0 does work (no threadgroup barriers).
// Buffer layout: in_buf[0], out_buf[1], input_groups[2], fan_in[3]
kernel void logits_topk_select50_merge(
    device const TopKCandidate* in_buf       [[buffer(0)]],
    device       TopKCandidate* out_buf      [[buffer(1)]],
    constant uint&              input_groups [[buffer(2)]],
    constant uint&              fan_in       [[buffer(3)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lane       [[thread_index_in_simdgroup]],
    uint sgid       [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]])
{
    if (simd_width < 16u) return;
    if (sgid != 0u) return;

    thread TopKCandidate vals[SELECT50_MERGE_LOCAL_MAX];
    ulong selected = 0ul;

    uint start_group      = tgid * fan_in;
    uint remaining_groups = (start_group < input_groups) ? (input_groups - start_group) : 0u;
    uint groups_here      = min(fan_in, remaining_groups);
    uint candidate_count  = groups_here * TOPK50;

    for (uint j = 0u; j < SELECT50_MERGE_LOCAL_MAX; ++j) {
        uint slot = lane + j * simd_width;
        if (slot < candidate_count) {
            uint g   = slot / TOPK50;
            uint ci  = slot - g * TOPK50;
            vals[j]  = in_buf[(start_group + g) * TOPK50 + ci];
        } else {
            vals[j] = topk_sentinel();
        }
    }

    for (uint rank = 0u; rank < TOPK50; ++rank) {
        TopKCandidate lane_best   = topk_sentinel();
        uint          lane_best_j  = UINT_MAX;
        for (uint j = 0u; j < SELECT50_MERGE_LOCAL_MAX; ++j) {
            if (((selected >> j) & 1ul) == 0ul && topk_better(vals[j], lane_best)) {
                lane_best   = vals[j];
                lane_best_j = j;
            }
        }

        TopKCandidate winner = simd_argmax(lane_best, simd_width);
        winner.logit    = simd_broadcast(winner.logit,    0u);
        winner.token_id = simd_broadcast(winner.token_id, 0u);

        if (lane == 0u) {
            out_buf[tgid * TOPK50 + rank] = winner;
        }
        if (lane_best_j != UINT_MAX && vals[lane_best_j].token_id == winner.token_id) {
            selected |= (1ul << lane_best_j);
        }
    }
}

// ===== Fused copy-then-RMS-norm =====
// Saves pre-norm state: residual_out = src (copy)
// Then normalizes src in-place: src = rms_norm(src) * (1 + gamma)
// Replaces 2 dispatches (copy_buf + rms_norm_qwen35) with 1.
// One threadgroup, 256 threads; assumes 1 row (decode path).
kernel void copy_and_rms_norm(
    device float* src            [[buffer(0)]],  // hidden — normalized in-place
    device float* residual_out   [[buffer(1)]],  // receives pre-norm copy of src
    device const float* gamma    [[buffer(2)]],
    constant uint& row_len       [[buffer(3)]],
    constant float& eps          [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    constexpr uint WG = 256;
    threadgroup float shared[WG];

    // Phase 1: copy src → residual_out and accumulate sum of squares
    float local_sum = 0.0f;
    for (uint i = lid; i < row_len; i += tgs) {
        float v = src[i];
        residual_out[i] = v;
        local_sum += v * v;
    }

    // Reduce sum of squares
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(shared[0] / float(row_len) + eps);

    // Phase 2: normalize src in-place with Qwen3.5 shifted RMSNorm (1 + gamma).
    for (uint i = lid; i < row_len; i += tgs) {
        src[i] = src[i] * rms * (1.0f + gamma[i]);
    }
}

// ===== Batch copy-then-RMS-norm: multi-row version for prefill =====
// Each threadgroup processes one row. gid = row index.
kernel void copy_and_rms_norm_batch(
    device float* src            [[buffer(0)]],
    device float* residual_out   [[buffer(1)]],
    device const float* gamma    [[buffer(2)]],
    constant uint& row_len       [[buffer(3)]],
    constant uint& num_rows      [[buffer(4)]],
    constant float& eps          [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    if (gid >= num_rows) return;
    constexpr uint WG = 256;
    threadgroup float shared[WG];
    uint base = gid * row_len;

    float local_sum = 0.0f;
    for (uint i = lid; i < row_len; i += tgs) {
        float v = src[base + i];
        residual_out[base + i] = v;
        local_sum += v * v;
    }

    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(shared[0] / float(row_len) + eps);

    for (uint i = lid; i < row_len; i += tgs) {
        src[base + i] = src[base + i] * rms * (1.0f + gamma[i]);
    }
}

// ===== Batch fused residual-add-then-norm: multi-row version for prefill =====
// residual_out[row] = base[row] + delta[row]; normed_out[row] = rms_norm(residual_out[row])
kernel void fused_residual_add_norm_batch(
    device const float* base     [[buffer(0)]],
    device const float* delta    [[buffer(1)]],
    device float* residual_out   [[buffer(2)]],
    device float* normed_out     [[buffer(3)]],
    device const float* gamma    [[buffer(4)]],
    constant uint& row_len       [[buffer(5)]],
    constant uint& num_rows      [[buffer(6)]],
    constant float& eps          [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    if (gid >= num_rows) return;
    constexpr uint WG = 256;
    threadgroup float shared[WG];
    uint base_off = gid * row_len;

    float local_sum = 0.0f;
    for (uint i = lid; i < row_len; i += tgs) {
        float val = base[base_off + i] + delta[base_off + i];
        residual_out[base_off + i] = val;
        local_sum += val * val;
    }

    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = rsqrt(shared[0] / float(row_len) + eps);

    for (uint i = lid; i < row_len; i += tgs) {
        normed_out[base_off + i] = residual_out[base_off + i] * rms * (1.0f + gamma[i]);
    }
}

// ===== Fused add-into-residual + copy-to-hidden =====
// residual[i] += src[i]; dst[i] = residual[i]
// Replaces 2 dispatches (add_buf + copy_buf) with 1.
// Used at end of each transformer layer.
kernel void add_and_copy(
    device const float* src   [[buffer(0)]],  // ffn_out — added into residual
    device float* residual    [[buffer(1)]],  // accumulates src
    device float* dst         [[buffer(2)]],  // hidden — receives updated residual
    constant uint& count      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float r = residual[gid] + src[gid];
    residual[gid] = r;
    dst[gid] = r;
}

// ===== LoRA GEMV Phase 1: intermediate = A @ x (ADR-045 step 2) =====
// A is row-major float32 [rank, K]. One threadgroup per rank-row.
// 128 threads (4 simdgroups × 32), simd_sum reduction over K.
// Dispatch: threadgroups=(rank, 1, 1), threads=(32, 4, 1)
kernel void lora_gemv_a(
    device const float* x       [[buffer(0)]],
    device const float* A       [[buffer(1)]],
    device float* intermediate  [[buffer(2)]],
    constant uint& rank_val     [[buffer(3)]],
    constant uint& K            [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    uint r = tgpig.x;
    if (r >= rank_val) return;

    device const float* row = A + r * K;
    float partial = 0.0f;
    for (uint k = tiisg + sgitg * 32; k < K; k += 128) {
        partial += row[k] * x[k];
    }
    partial = simd_sum(partial);

    threadgroup float sg_sums[4];
    if (tiisg == 0) sg_sums[sgitg] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) {
        intermediate[r] = sg_sums[0] + sg_sums[1] + sg_sums[2] + sg_sums[3];
    }
}

// ===== LoRA GEMV Phase 2: y += scale * B @ intermediate (ADR-045 step 2) =====
// B is row-major float32 [N, rank]. One thread per output element.
// rank ≤ 64, so the inner loop is trivially short — no reduction needed.
// Dispatch: threadgroups=(ceil(N/256), 1, 1), threads=(256, 1, 1)
kernel void lora_gemv_b_accum(
    device const float* intermediate  [[buffer(0)]],
    device const float* B             [[buffer(1)]],
    device float* y                   [[buffer(2)]],
    constant uint& N                  [[buffer(3)]],
    constant uint& rank_val           [[buffer(4)]],
    constant float& scale             [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    device const float* row = B + gid * rank_val;
    float sum = 0.0f;
    for (uint j = 0; j < rank_val; j++) {
        sum += row[j] * intermediate[j];
    }
    y[gid] += scale * sum;
}

// ===== MoE: zero a float buffer =====
// Used to clear scratch_out before accumulating expert outputs.
kernel void zero_buf(
    device float* buf   [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    buf[gid] = 0.0f;
}

// ===== MoE expert GEMV with expert-major f16 weight layout =====
// W layout: [num_experts, N, K] f16 contiguous.
// Expert e's weight matrix starts at element offset: e * N * K.
// This is the same gemv_decode_core template but takes an element-offset into W.
// buffer(0): x          [K] f32 activation
// buffer(1): W          [num_experts * N * K] f16
// buffer(2): out        [N] f32 output
// buffer(3): GemmParams  {M=1, N, K, lda=K, ldb=K, ldc=N}
// buffer(4): W_elem_offset  uint — element index of this expert's weight start
// Dispatch: threadgroups=(N,1,1), threads=(256,1,1)
kernel void moe_expert_gemv(
    device const float* x             [[buffer(0)]],
    device const half*  W             [[buffer(1)]],
    device float*       out           [[buffer(2)]],
    constant GemmParams& p            [[buffer(3)]],
    constant uint&       W_elem_off   [[buffer(4)]],
    uint n            [[threadgroup_position_in_grid]],
    uint tid          [[thread_index_in_threadgroup]],
    uint simd_lane    [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint simd_width   [[threads_per_simdgroup]])
{
    threadgroup float sg_partials[8]; // 256/32 simdgroups
    gemv_decode_core<256>(x, W + W_elem_off, out, p, n, tid,
                          simd_lane, simdgroup_id, simd_width, sg_partials);
}

// ===== MoE: scale-and-accumulate =====
// Accumulates a scaled expert output into the running sum.
// scratch_out[i] += router_weight * expert_out[i]
// Dispatch: threadgroups=1, threads=hidden (one thread per output element)
kernel void moe_scale_add(
    device float* scratch_out         [[buffer(0)]],
    device const float* expert_out    [[buffer(1)]],
    constant float& router_weight     [[buffer(2)]],
    constant uint& hidden             [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= hidden) return;
    scratch_out[gid] = fma(router_weight, expert_out[gid], scratch_out[gid]);
}

// ===== MoE: sigmoid-gated shared expert accumulate =====
// gate_val = sigmoid(dot(x, shared_gate_w))
// scratch_out[i] += gate_val * shared_out[i]
// Runs after shared expert down-projection result lands in expert_out.
kernel void moe_shared_gate_add(
    device float* scratch_out         [[buffer(0)]],
    device const float* expert_out    [[buffer(1)]],
    constant float& gate_val          [[buffer(2)]],
    constant uint& hidden             [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= hidden) return;
    scratch_out[gid] = fma(gate_val, expert_out[gid], scratch_out[gid]);
}

// ===== Batch scatter Q and gate from interleaved q_proj output (Win 1) =====
// Processes num_tokens rows in one dispatch.
// Source: q[num_tokens, num_heads * 2 * head_dim] with (Q_h, gate_h) per head.
// Output: q_out[num_tokens, num_heads * head_dim], gate_out same shape.
kernel void scatter_q_gate_batch(
    device const float* qg_interleaved [[buffer(0)]],
    device float* q_out                [[buffer(1)]],
    device float* gate_out             [[buffer(2)]],
    constant uint& num_tokens          [[buffer(3)]],
    constant uint& num_heads           [[buffer(4)]],
    constant uint& head_dim            [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    uint q_dim = num_heads * head_dim;
    uint total = num_tokens * q_dim;
    if (gid >= total) return;

    uint t    = gid / q_dim;
    uint rem  = gid % q_dim;
    uint head = rem / head_dim;
    uint d    = rem % head_dim;

    // Source row: token t, head layout (Q_h, gate_h) per head in q_proj output.
    uint src_base = t * (2u * q_dim) + head * (2u * head_dim);
    q_out[gid]    = qg_interleaved[src_base + d];
    gate_out[gid] = qg_interleaved[src_base + head_dim + d];
}

// ===== Batch per-head RMS norm (Win 1) =====
// One threadgroup per (token, head) pair; same math as per_head_rms_norm.
kernel void per_head_rms_norm_batch(
    device float* x              [[buffer(0)]],
    device const float* gamma    [[buffer(1)]],
    constant uint& num_tokens    [[buffer(2)]],
    constant uint& num_heads     [[buffer(3)]],
    constant uint& head_dim      [[buffer(4)]],
    constant float& eps          [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    if (gid >= num_tokens * num_heads) return;

    uint t    = gid / num_heads;
    uint head = gid % num_heads;
    uint base = t * (num_heads * head_dim) + head * head_dim;

    constexpr uint NORM_WG = 256;
    threadgroup float shared[NORM_WG];

    float local_sum = 0.0f;
    for (uint i = lid; i < head_dim; i += tgs) {
        float v = x[base + i];
        local_sum += v * v;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared[0] / float(head_dim) + eps);

    // Qwen3.5 shifted RMSNorm: same (1 + gamma) convention as per_head_rms_norm.
    for (uint i = lid; i < head_dim; i += tgs) {
        x[base + i] = x[base + i] * rms * (1.0f + gamma[i]);
    }
}

// ===== Batch stride-half partial RoPE (Win 1) =====
// Extends partial_rope_interleaved to num_tokens rows.
// RoPE absolute position for token t is base_pos + t, not chunk-local t.
kernel void partial_rope_batch(
    device float* x                 [[buffer(0)]],
    device const float* cos_tab     [[buffer(1)]],
    device const float* sin_tab     [[buffer(2)]],
    constant uint& num_tokens       [[buffer(3)]],
    constant uint& num_heads        [[buffer(4)]],
    constant uint& head_dim         [[buffer(5)]],
    constant uint& half_rope_dim    [[buffer(6)]],
    constant uint& base_pos         [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint total_pairs = num_tokens * num_heads * half_rope_dim;
    if (gid >= total_pairs) return;

    uint pair = gid % half_rope_dim;
    uint head = (gid / half_rope_dim) % num_heads;
    uint t    = gid / (num_heads * half_rope_dim);

    uint base    = t * (num_heads * head_dim) + head * head_dim;
    // Absolute position keeps RoPE consistent across chunk boundaries.
    uint cs_base = (base_pos + t) * half_rope_dim;

    float cos_val = cos_tab[cs_base + pair];
    float sin_val = sin_tab[cs_base + pair];

    // Stride-half pairing: (pair, half_rope_dim + pair), matching HF rotate_half.
    uint idx0 = base + pair;
    uint idx1 = base + half_rope_dim + pair;
    float x0 = x[idx0];
    float x1 = x[idx1];
    x[idx0] = x0 * cos_val - x1 * sin_val;
    x[idx1] = x0 * sin_val + x1 * cos_val;
}

// ===== Batch K/V cache store (Win 1) =====
// Copies num_tokens rows of K and V into cache buffers in one dispatch.
// Cache layout is token-major: cache[row * kv_dim + d] where row = base_pos + t.
kernel void copy_kv_cache_batch(
    device const float* k_src   [[buffer(0)]],
    device const float* v_src   [[buffer(1)]],
    device float* k_cache       [[buffer(2)]],
    device float* v_cache       [[buffer(3)]],
    constant uint& num_tokens   [[buffer(4)]],
    constant uint& kv_dim       [[buffer(5)]],
    constant uint& base_pos     [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = num_tokens * kv_dim;
    if (gid >= total) return;

    uint t = gid / kv_dim;
    uint d = gid % kv_dim;
    uint src = t * kv_dim + d;
    uint dst = (base_pos + t) * kv_dim + d;
    k_cache[dst] = k_src[src];
    v_cache[dst] = v_src[src];
}

// Variant that narrows f32 src → half dst for the f16 KV cache batch-write path.
kernel void copy_kv_cache_batch_f16(
    device const float* k_src   [[buffer(0)]],
    device const float* v_src   [[buffer(1)]],
    device half* k_cache        [[buffer(2)]],
    device half* v_cache        [[buffer(3)]],
    constant uint& num_tokens   [[buffer(4)]],
    constant uint& kv_dim       [[buffer(5)]],
    constant uint& base_pos     [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = num_tokens * kv_dim;
    if (gid >= total) return;

    uint t = gid / kv_dim;
    uint d = gid % kv_dim;
    uint src = t * kv_dim + d;
    uint dst = (base_pos + t) * kv_dim + d;
    k_cache[dst] = half(k_src[src]);
    v_cache[dst] = half(v_src[src]);
}

// ===== Batched causal prefill attention (Win 2) =====
// Replaces the per-token decode_attention loop for full-attention prefill chunks.
// Grid: [num_kv_heads, num_tokens, 1]. Threads: [256, 1, 1] — one thread per output dim.
// One threadgroup per (kv_head, query_token) processes all Q heads in the GQA group.
//
// KVT is the KV cache element type (float for f32 path, half for f16 path).
// GRP is the compile-time MAX_GRP specialization (4/6/8) — see the Rust-side
// `prefill_maxgrp_suffix` helper for how a loaded model's group_size selects
// among the g4/g6/g8 instantiations below.
#define DEFINE_PREFILL_ATTENTION_BATCHED_CAUSAL(NAME, KVT, GRP) \
kernel void NAME( \
    device const float* q          [[buffer(0)]], \
    device const KVT*   k_cache    [[buffer(1)]], \
    device const KVT*   v_cache    [[buffer(2)]], \
    device float* out              [[buffer(3)]], \
    constant uint& base_pos        [[buffer(4)]], \
    constant uint& num_tokens      [[buffer(5)]], \
    constant uint& cache_len_total [[buffer(6)]], \
    constant uint& head_dim        [[buffer(7)]], \
    constant uint& num_q_heads     [[buffer(8)]], \
    constant uint& num_kv_heads    [[buffer(9)]], \
    constant uint& q_dim           [[buffer(10)]], \
    constant uint& kv_dim          [[buffer(11)]], \
    constant float& scale          [[buffer(12)]], \
    uint3 gid3 [[threadgroup_position_in_grid]], \
    uint3 lid3 [[thread_position_in_threadgroup]], \
    uint3 tgs3 [[threads_per_threadgroup]]) \
{ \
    constexpr uint HEAD_DIM    = 256; \
    constexpr uint MAX_GRP     = GRP; \
    constexpr uint TILE_TOKENS = 256; \
    const uint lid = lid3.x; \
    const uint tgs = tgs3.x; \
    if (head_dim != HEAD_DIM || num_kv_heads == 0) return; \
    const uint kvh = gid3.x; \
    const uint qt  = gid3.y; \
    if (kvh >= num_kv_heads || qt >= num_tokens) return; \
    if ((num_q_heads % num_kv_heads) != 0) return; \
    const uint group_size = num_q_heads / num_kv_heads; \
    if (group_size == 0 || group_size > MAX_GRP) return; \
    const uint qh_base = kvh * group_size; \
    const uint causal_len = min(cache_len_total, base_pos + qt + 1u); \
    if (causal_len == 0) { \
        for (uint qi = 0; qi < group_size; qi++) { \
            out[qt * q_dim + (qh_base + qi) * HEAD_DIM + lid] = 0.0f; \
        } \
        return; \
    } \
    threadgroup float q_s    [MAX_GRP * HEAD_DIM]; \
    threadgroup float score_s[MAX_GRP * TILE_TOKENS]; \
    threadgroup float reduce_s[TILE_TOKENS]; \
    threadgroup float m_s    [MAX_GRP]; \
    threadgroup float l_s    [MAX_GRP]; \
    threadgroup float alpha_s[MAX_GRP]; \
    if (lid < group_size) { \
        m_s[lid] = -INFINITY; \
        l_s[lid] = 0.0f; \
    } \
    const uint q_row_base = qt * q_dim + qh_base * HEAD_DIM; \
    for (uint idx = lid; idx < group_size * HEAD_DIM; idx += tgs) { \
        q_s[idx] = q[q_row_base + idx]; \
    } \
    float acc[MAX_GRP]; \
    for (uint qi = 0; qi < MAX_GRP; qi++) acc[qi] = 0.0f; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint tile_start = 0; tile_start < causal_len; tile_start += TILE_TOKENS) { \
        const uint tile_count = min(TILE_TOKENS, causal_len - tile_start); \
        if (lid < tile_count) { \
            float dot[MAX_GRP]; \
            for (uint qi = 0; qi < MAX_GRP; qi++) dot[qi] = 0.0f; \
            const uint k_base = (tile_start + lid) * kv_dim + kvh * HEAD_DIM; \
            for (uint d = 0; d < HEAD_DIM; d++) { \
                const float kd = float(k_cache[k_base + d]); \
                for (uint qi = 0; qi < group_size; qi++) { \
                    dot[qi] += q_s[qi * HEAD_DIM + d] * kd; \
                } \
            } \
            for (uint qi = 0; qi < group_size; qi++) { \
                score_s[qi * TILE_TOKENS + lid] = dot[qi] * scale; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint qi = 0; qi < group_size; qi++) { \
            reduce_s[lid] = (lid < tile_count) ? score_s[qi * TILE_TOKENS + lid] : -INFINITY; \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            for (uint s = tgs >> 1; s > 0; s >>= 1) { \
                if (lid < s) reduce_s[lid] = max(reduce_s[lid], reduce_s[lid + s]); \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
            } \
            const float tile_max = reduce_s[0]; \
            if (lid == 0) { \
                const float m_old = m_s[qi]; \
                const float m_new = max(m_old, tile_max); \
                alpha_s[qi] = isfinite(m_old) ? exp(m_old - m_new) : 0.0f; \
                m_s[qi]     = m_new; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            if (lid < tile_count) { \
                score_s[qi * TILE_TOKENS + lid] = \
                    exp(score_s[qi * TILE_TOKENS + lid] - m_s[qi]); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            reduce_s[lid] = (lid < tile_count) ? score_s[qi * TILE_TOKENS + lid] : 0.0f; \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            for (uint s = tgs >> 1; s > 0; s >>= 1) { \
                if (lid < s) reduce_s[lid] += reduce_s[lid + s]; \
                threadgroup_barrier(mem_flags::mem_threadgroup); \
            } \
            if (lid == 0) { \
                l_s[qi] = alpha_s[qi] * l_s[qi] + reduce_s[0]; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
        for (uint qi = 0; qi < group_size; qi++) { \
            acc[qi] *= alpha_s[qi]; \
        } \
        const uint d = lid; \
        for (uint local_t = 0; local_t < tile_count; local_t++) { \
            const float v = float(v_cache[(tile_start + local_t) * kv_dim + kvh * HEAD_DIM + d]); \
            for (uint qi = 0; qi < group_size; qi++) { \
                acc[qi] += score_s[qi * TILE_TOKENS + local_t] * v; \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    for (uint qi = 0; qi < group_size; qi++) { \
        const uint qh  = qh_base + qi; \
        const float dn = l_s[qi]; \
        out[qt * q_dim + qh * HEAD_DIM + lid] = dn > 0.0f ? acc[qi] / dn : 0.0f; \
    } \
}

// g4/g6/g8: MAX_GRP specializations selected at pipeline-build time by the
// Rust-side `prefill_maxgrp_suffix(group_size)` helper. g8 is byte-for-byte
// equivalent to the pre-specialization kernel (MAX_GRP=8 was the fleet-max
// hardcode this replaces); g4/g6 trade unreachable-for-that-model MAX_GRP
// rows for more threadgroup-memory occupancy (measured -17.4% TTFT@4096 on
// the 0.8b model at MAX_GRP=4).
DEFINE_PREFILL_ATTENTION_BATCHED_CAUSAL(prefill_attention_batched_causal_g4,     float, 4)
DEFINE_PREFILL_ATTENTION_BATCHED_CAUSAL(prefill_attention_batched_causal_g4_f16, half,  4)
DEFINE_PREFILL_ATTENTION_BATCHED_CAUSAL(prefill_attention_batched_causal_g6,     float, 6)
DEFINE_PREFILL_ATTENTION_BATCHED_CAUSAL(prefill_attention_batched_causal_g6_f16, half,  6)
DEFINE_PREFILL_ATTENTION_BATCHED_CAUSAL(prefill_attention_batched_causal_g8,     float, 8)
DEFINE_PREFILL_ATTENTION_BATCHED_CAUSAL(prefill_attention_batched_causal_g8_f16, half,  8)

// ===== GDN Chunked Prefill Scan (C=32 specialized) =====
// Implements chunked-parallel WY/UT scan for GatedDeltaNet prefill.
// Default path for 0.8B shape (kd=vd=128, 16 heads, ratio=1); opt-out via LATTICE_GDN_CHUNKED=0.
// Serial path (gdn_recurrence_fused) is the fallback for unsupported shapes or when opted out.

struct GdnChunkParams {
    uint  key_dim;         // 128
    uint  value_dim;       // 128
    uint  num_key_heads;   // 16
    uint  num_value_heads; // 16
    uint  hidden_size;     // 1024
    uint  q_total;         // num_key_heads * key_dim = 2048
    uint  v_offset;        // q_total * 2 = 4096
    uint  qkv_dim;         // 6144
    uint  output_dim;      // 2048
    uint  chunk_size;      // 32
    uint  n_tokens;
    uint  num_chunks;
    uint  active_chunk;
    float scale;           // 1/sqrt(128)
    float eps;             // rms_norm_eps
};

// Kernel 1: Conv1d+SiLU, Q/K L2-norm, beta, log_alpha, V for each (chunk, value_head).
// Grid: (num_chunks, num_value_heads, 1). Threads: (32, 4, 1) = 128 per TG.
// Each thread `tid` handles dim `tid` of Q, K, V (tid in [0,127]).
// kh = h / ratio for Q/K channel offsets; decay params (beta, a_log, dt_bias, in_proj_b/a) indexed by h.
kernel void gdn_chunk_materialize_c32(
    device float*       conv_buf    [[buffer(0)]],  // [qkv_dim, buf_len] rolling shift register
    device const float* gdn_qkv     [[buffer(1)]],  // [n_tokens, qkv_dim]
    device const float* conv_weight [[buffer(2)]],  // [qkv_dim, kernel_size]
    device const float* hidden_in   [[buffer(3)]],  // [n_tokens, hidden_size]
    device const half*  in_proj_b   [[buffer(4)]],  // [num_value_heads, hidden_size]
    device const half*  in_proj_a   [[buffer(5)]],  // [num_value_heads, hidden_size]
    device const float* a_log       [[buffer(6)]],  // [num_value_heads]
    device const float* dt_bias     [[buffer(7)]],  // [num_value_heads]
    device float*       out_q       [[buffer(8)]],  // [num_chunks, num_value_heads, C, key_dim]
    device float*       out_k       [[buffer(9)]],  // [num_chunks, num_value_heads, C, key_dim]
    device float*       out_v       [[buffer(10)]], // [num_chunks, num_value_heads, C, value_dim]
    device float*       out_bla     [[buffer(11)]], // [num_chunks, num_value_heads, C, 2]
    constant GdnChunkParams& p      [[buffer(12)]],
    uint3 tgpig    [[threadgroup_position_in_grid]],
    uint3 tpitg    [[thread_position_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  sgitg    [[simdgroup_index_in_threadgroup]])
{
    uint chunk_idx = tgpig.x;
    uint h         = tgpig.y;
    if (chunk_idx >= p.num_chunks || h >= p.num_value_heads) return;

    uint tid = tpitg.y * 32u + tpitg.x;
    uint kd  = p.key_dim;
    uint vd  = p.value_dim;
    uint hd  = p.hidden_size;
    constexpr uint ks      = 4u;
    constexpr uint buf_len = ks - 1u;  // 3

    uint ratio      = p.num_value_heads / p.num_key_heads;
    uint kh         = h / ratio;  // Q/K use key-head index; decay params use h
    uint chunk_base = chunk_idx * p.chunk_size;
    uint ci         = min(p.chunk_size, p.n_tokens - chunk_base);
    uint chunk_head = chunk_idx * p.num_value_heads + h;

    // Channel offset for this thread in the QKV buffer layout
    uint q_ch = kh * kd + tid;
    uint k_ch = p.q_total + kh * kd + tid;
    uint v_ch = p.v_offset + h * vd + tid;

    threadgroup float sg_buf[4];

    for (uint j = 0u; j < ci; j++) {
        uint global_row = chunk_base + j;
        uint head_row   = chunk_head * p.chunk_size + j;

        // Depthwise conv1d+SiLU for Q, K, V channels of this thread
        float q_raw = 0.0f, k_raw = 0.0f, v_raw = 0.0f;
        for (uint tap = 0u; tap < ks; tap++) {
            int src = (int)(global_row + tap) - (int)buf_len;
            float xq, xk, xv;
            if (src < 0) {
                uint bi = global_row + tap;
                xq = conv_buf[q_ch * buf_len + bi];
                xk = conv_buf[k_ch * buf_len + bi];
                xv = conv_buf[v_ch * buf_len + bi];
            } else {
                uint sr = (uint)src;
                xq = gdn_qkv[sr * p.qkv_dim + q_ch];
                xk = gdn_qkv[sr * p.qkv_dim + k_ch];
                xv = gdn_qkv[sr * p.qkv_dim + v_ch];
            }
            q_raw += xq * conv_weight[q_ch * ks + tap];
            k_raw += xk * conv_weight[k_ch * ks + tap];
            v_raw += xv * conv_weight[v_ch * ks + tap];
        }
        // SiLU: x / (1 + exp(-x))
        float q_silu = q_raw / (1.0f + exp(-q_raw));
        float k_silu = k_raw / (1.0f + exp(-k_raw));
        float v_silu = v_raw / (1.0f + exp(-v_raw));

        // L2 normalize Q (reduce over 128 dims, 4 simdgroups × 32 lanes)
        // ADR-080 C1 fail-closed (#850): assign 0.0f to the whole vector directly on an
        // invalid (non-finite or near-zero) norm, never `value * guarded_zero_reciprocal`
        // (NaN * 0.0f == NaN under IEEE-754).
        float qsq = simd_sum(q_silu * q_silu);
        if (simd_lane == 0) sg_buf[sgitg] = qsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float s = 0.0f;
            for (uint si = 0u; si < 4u; si++) s += sg_buf[si];
            sg_buf[0] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool q_valid = isfinite(sg_buf[0]) && sg_buf[0] > 1e-12f;
        float qs_inv = q_valid ? rsqrt(sg_buf[0]) : 0.0f;
        // WAR guard: all simdgroups must finish reading sg_buf[0] above before the
        // K-normalize reduction below overwrites sg_buf[sgitg]. Without this barrier a
        // lagging simdgroup reads the K-sum in place of the Q-sum (cross-simdgroup race).
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // L2 normalize K
        float ksq = simd_sum(k_silu * k_silu);
        if (simd_lane == 0) sg_buf[sgitg] = ksq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float s = 0.0f;
            for (uint si = 0u; si < 4u; si++) s += sg_buf[si];
            sg_buf[0] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool k_valid = isfinite(sg_buf[0]) && sg_buf[0] > 1e-12f;
        float ks_inv = k_valid ? rsqrt(sg_buf[0]) : 0.0f;
        // WAR guard: same hazard as above — every simdgroup must read the K-sum from
        // sg_buf[0] before the beta reduction below overwrites sg_buf[sgitg].
        threadgroup_barrier(mem_flags::mem_threadgroup);

        out_q[head_row * kd + tid] = q_valid ? (q_silu * qs_inv) : 0.0f;
        out_k[head_row * kd + tid] = k_valid ? (k_silu * ks_inv) : 0.0f;
        out_v[head_row * vd + tid] = v_silu;

        // Beta = sigmoid(hidden[j] @ in_proj_b[h])  — per value head
        {
            device const half*  wb = in_proj_b + h * hd;
            device const float* hr = hidden_in + global_row * hd;
            float bp = 0.0f;
            for (uint i = tid; i < hd; i += 128u) bp += float(wb[i]) * hr[i];
            bp = simd_sum(bp);
            if (simd_lane == 0) sg_buf[sgitg] = bp;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float s = 0.0f;
                for (uint si = 0u; si < 4u; si++) s += sg_buf[si];
                out_bla[head_row * 2u] = 1.0f / (1.0f + exp(-s));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Log-alpha = -exp(a_log[h]) * softplus(hidden[j] @ in_proj_a[h] + dt_bias[h])  — per value head
        {
            device const half*  wa = in_proj_a + h * hd;
            device const float* hr = hidden_in + global_row * hd;
            float ap = 0.0f;
            for (uint i = tid; i < hd; i += 128u) ap += float(wa[i]) * hr[i];
            ap = simd_sum(ap);
            if (simd_lane == 0) sg_buf[sgitg] = ap;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float s = 0.0f;
                for (uint si = 0u; si < 4u; si++) s += sg_buf[si];
                float a  = exp(a_log[h]);
                float sp = log(1.0f + exp(s + dt_bias[h]));
                // Floor the per-token log-decay so the chunked cumulative-log scan stays
                // closed under saturated decay.  An overflowing softplus drives -a*sp to
                // -inf; the prefix-log differences in gdn_chunk_solve_c32 then form
                // -inf - -inf = NaN and poison all downstream GDN state (gamma_logs, qk_l,
                // k_right, S_all).  The serial recurrence maps the same input to
                // g = exp(-inf) = 0 (a finite full reset).  -88 reproduces that
                // (exp(-88) ~ 6e-39 ~ 0) while keeping the cumsum and its differences
                // finite; it only ever clamps values whose exp is already ~0, so finite
                // inputs are bit-unchanged.  fmax (not max) also sanitises a NaN from
                // exp(a_log)=inf * sp=0.  Mirrored by GDN_LOG_ALPHA_FLOOR in the CPU
                // algebra regression test.
                out_bla[head_row * 2u + 1u] = fmax(-a * sp, -88.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Conv-buf update removed from this kernel to eliminate the read/write race
    // (MAJ-2 fix): chunk 0 reads conv_buf while the last chunk was writing it in
    // the same all-chunks dispatch.  The update is now done in a separate
    // gdn_chunk_conv_buf_update_c32 dispatch that runs after this one completes.
}

// Kernel 1b: Update rolling conv buffer after the all-chunks materialize dispatch completes.
// Dispatched with a (1, num_value_heads, 1) grid — single-threadgroup, one dispatch after
// gdn_chunk_materialize_c32.  Reads conv_buf only from the pre-prefill taps (final_src < 0)
// or from gdn_qkv (final_src >= 0); no other threadgroup reads or writes conv_buf at the
// same time, so there is no race.  Thread (32, 4, 1) = 128 per TG, same width as materialize.
kernel void gdn_chunk_conv_buf_update_c32(
    device float*       conv_buf    [[buffer(0)]],  // [qkv_dim, buf_len] rolling shift register
    device const float* gdn_qkv    [[buffer(1)]],  // [n_tokens, qkv_dim]
    constant GdnChunkParams& p     [[buffer(2)]],
    uint3 tgpig    [[threadgroup_position_in_grid]],
    uint3 tpitg    [[thread_position_in_threadgroup]])
{
    // tgpig.x is always 0 (grid is (1, num_value_heads, 1)).
    uint h   = tgpig.y;
    if (h >= p.num_value_heads) return;

    uint tid = tpitg.y * 32u + tpitg.x;
    uint kd  = p.key_dim;
    uint vd  = p.value_dim;
    constexpr uint ks      = 4u;
    constexpr uint buf_len = ks - 1u;  // 3

    uint kh   = h;  // ratio=1 for 0.8B
    uint q_ch = kh * kd + tid;
    uint k_ch = p.q_total + kh * kd + tid;
    uint v_ch = p.v_offset + h * vd + tid;

    // Only write for valid channel lanes (128 total threads; kd=vd=128).
    if (tid >= kd) return;

    for (uint t = 0u; t < buf_len; t++) {
        int final_src = (int)p.n_tokens + (int)t - (int)buf_len;
        float xq, xk, xv;
        if (final_src < 0) {
            // Still within the pre-prefill taps that were snapshotted before materialize ran.
            // Read from the OLD conv_buf positions — no other kernel is modifying these now.
            uint bi = (uint)((int)buf_len + final_src);
            xq = conv_buf[q_ch * buf_len + bi];
            xk = conv_buf[k_ch * buf_len + bi];
            xv = conv_buf[v_ch * buf_len + bi];
        } else {
            uint sr = (uint)final_src;
            xq = gdn_qkv[sr * p.qkv_dim + q_ch];
            xk = gdn_qkv[sr * p.qkv_dim + k_ch];
            xv = gdn_qkv[sr * p.qkv_dim + v_ch];
        }
        conv_buf[q_ch * buf_len + t] = xq;
        conv_buf[k_ch * buf_len + t] = xk;
        conv_buf[v_ch * buf_len + t] = xv;
    }
}

// Kernel 2: Compute log-gamma, KKT, QKT*L, W/U forward substitution, K_right.
// Grid: (num_chunks, num_value_heads, 1). Threads: (32, 4, 1) = 128 per TG.
// Sequential over j (forward substitution); parallel over dim `lane = tid`.
kernel void gdn_chunk_solve_c32(
    device const float* q          [[buffer(0)]],  // [chunks, heads, C, key_dim]
    device const float* k          [[buffer(1)]],  // [chunks, heads, C, key_dim]
    device const float* v_in       [[buffer(2)]],  // [chunks, heads, C, value_dim]
    device const float* bla        [[buffer(3)]],  // [chunks, heads, C, 2] beta+log_alpha
    device float*       gamma      [[buffer(4)]],  // [chunks, heads, C] log cumulative gamma
    device float*       gamma_end  [[buffer(5)]],  // [chunks, heads]
    device float*       kkt_out    [[buffer(6)]],  // [chunks, heads, C, C]
    device float*       qk_l       [[buffer(7)]],  // [chunks, heads, C, C]
    device float*       out_w      [[buffer(8)]],  // [chunks, heads, C, key_dim]
    device float*       out_u      [[buffer(9)]],  // [chunks, heads, C, value_dim]
    device float*       k_right    [[buffer(10)]], // [chunks, heads, C, key_dim]
    constant GdnChunkParams& p     [[buffer(11)]],
    uint3 tgpig    [[threadgroup_position_in_grid]],
    uint3 tpitg    [[thread_position_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  sgitg    [[simdgroup_index_in_threadgroup]])
{
    uint chunk_idx = tgpig.x;
    uint h         = tgpig.y;
    if (chunk_idx >= p.num_chunks || h >= p.num_value_heads) return;

    uint tid = tpitg.y * 32u + tpitg.x;
    uint kd  = p.key_dim;
    uint vd  = p.value_dim;

    uint chunk_base = chunk_idx * p.chunk_size;
    uint ci         = min(p.chunk_size, p.n_tokens - chunk_base);
    uint chunk_head = chunk_idx * p.num_value_heads + h;

    threadgroup float sg_buf[4];
    threadgroup float kkt_scalar;

    // Phase 1: Compute log-gamma cumsum (same for all threads; stored log-form)
    float gamma_logs[32];
    float log_g = 0.0f;
    for (uint j = 0u; j < ci; j++) {
        uint hr = chunk_head * p.chunk_size + j;
        log_g += bla[hr * 2u + 1u];
        gamma_logs[j] = log_g;
    }
    float log_ge = (ci > 0u) ? gamma_logs[ci - 1u] : 0.0f;

    if (tid == 0u) {
        for (uint j = 0u; j < ci; j++) {
            gamma[chunk_head * p.chunk_size + j] = gamma_logs[j];
        }
        gamma_end[chunk_head] = log_ge;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 2: Forward substitution for W, U; inline KKT/QKL computation.
    for (uint j = 0u; j < ci; j++) {
        uint hr_j      = chunk_head * p.chunk_size + j;
        float beta_j   = bla[hr_j * 2u];
        float k_j_lane = k[hr_j * kd + tid];
        float q_j_lane = q[hr_j * kd + tid];
        float v_j_lane = v_in[hr_j * vd + tid];

        float w_lane = beta_j * k_j_lane;
        float u_lane = beta_j * v_j_lane;

        for (uint kk = 0u; kk < j; kk++) {
            uint  hr_k    = chunk_head * p.chunk_size + kk;
            float k_k     = k[hr_k * kd + tid];
            float q_k_dot = simd_sum(q_j_lane * k_k);
            float k_k_dot = simd_sum(k_j_lane * k_k);
            if (simd_lane == 0u) sg_buf[sgitg] = k_k_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float s = 0.0f;
                for (uint si = 0u; si < 4u; si++) s += sg_buf[si];
                kkt_scalar = s;
                kkt_out[hr_j * p.chunk_size + kk] = s;
            }
            // WAR guard: tid==0 must finish reading every sg_buf[si] for the KKT sum above
            // before any simdgroup overwrites sg_buf[sgitg] with the Q-dot partial below.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Gather Q dot separately using sg_buf after KKT
            float qk_partial = q_k_dot;
            if (simd_lane == 0u) sg_buf[sgitg] = qk_partial;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float kkt_val;
            {
                kkt_val = kkt_scalar;
                if (tid == 0u) {
                    float sq = 0.0f;
                    for (uint si = 0u; si < 4u; si++) sq += sg_buf[si];
                    float l_jk = exp(gamma_logs[j] - gamma_logs[kk]);
                    qk_l[hr_j * p.chunk_size + kk] = sq * l_jk;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float w_k    = out_w[hr_k * kd + tid];
            float u_k    = out_u[hr_k * vd + tid];
            float l_jk   = exp(gamma_logs[j] - gamma_logs[kk]);
            w_lane -= beta_j * kkt_val * w_k;
            u_lane -= beta_j * l_jk * kkt_val * u_k;
        }

        // Self-dot for kkt[j,j] and qkl[j,j] (diagonal: L[j,j]=1)
        {
            float k2 = simd_sum(k_j_lane * k_j_lane);
            if (simd_lane == 0u) sg_buf[sgitg] = k2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float qd = simd_sum(q_j_lane * k_j_lane);
            if (simd_lane == 0u) sg_buf[sgitg] = qd;  // note: overwrites k2 partial — compute serially
            // Recompute k2 with sgitg-local partial; use separate round
            // Compute kkt_jj:
            float k2p = simd_sum(k_j_lane * k_j_lane);
            if (simd_lane == 0u) sg_buf[sgitg] = k2p;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float sk = 0.0f;
                for (uint si = 0u; si < 4u; si++) sk += sg_buf[si];
                kkt_out[hr_j * p.chunk_size + j] = sk;
            }
            // WAR guard: tid==0 must finish reading sg_buf for kkt[j,j] before the
            // qkl[j,j] reduction below overwrites sg_buf[sgitg].
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Compute qkl_jj:
            float qdp = simd_sum(q_j_lane * k_j_lane);
            if (simd_lane == 0u) sg_buf[sgitg] = qdp;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float sq = 0.0f;
                for (uint si = 0u; si < 4u; si++) sq += sg_buf[si];
                qk_l[hr_j * p.chunk_size + j] = sq;  // L[j,j] = exp(0) = 1
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        out_w[hr_j * kd + tid]   = w_lane;
        out_u[hr_j * vd + tid]   = u_lane;
        k_right[hr_j * kd + tid] = exp(log_ge - gamma_logs[j]) * k_j_lane;
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// Kernel 3: Compute R = U - gamma * W @ S0^T, raw_out = (Q@S0^T * gamma + QKL @ R) * scale.
// Dispatched per-chunk in sequential order.
// Grid: (ceil(vd/8), num_value_heads, 1). Threads: (32, 8, 1) = 256 per TG.
// Thread x handles k-dimension partial (stride 32); thread y handles v within 8-value tile.
kernel void gdn_chunk_residual_output_c32(
    device const float* S_all      [[buffer(0)]],  // [num_value_heads, vd, kd] value-major
    device const float* q          [[buffer(1)]],  // [chunks, heads, C, key_dim]
    device const float* w_sc       [[buffer(2)]],  // [chunks, heads, C, key_dim]
    device const float* u_sc       [[buffer(3)]],  // [chunks, heads, C, value_dim]
    device const float* gamma      [[buffer(4)]],  // [chunks, heads, C] log-form
    device const float* qk_l       [[buffer(5)]],  // [chunks, heads, C, C]
    device float*       out_r      [[buffer(6)]],  // [chunks, heads, C, value_dim]
    device float*       raw_out    [[buffer(7)]],  // [n_tokens, output_dim] scaled
    constant GdnChunkParams& p     [[buffer(8)]],
    uint3 tgpig    [[threadgroup_position_in_grid]],
    uint3 tpitg    [[thread_position_in_threadgroup]])
{
    uint v_tile    = tgpig.x;
    uint h         = tgpig.y;
    uint chunk_idx = p.active_chunk;
    if (v_tile >= (p.value_dim + 7u) / 8u || h >= p.num_value_heads) return;

    // Thread layout (32, 8, 1): x = k-partial stride, y = v within 8-tile
    uint lane_k = tpitg.x;
    uint lane_v = tpitg.y;
    uint v      = v_tile * 8u + lane_v;
    if (v >= p.value_dim) return;

    uint kd         = p.key_dim;
    uint vd         = p.value_dim;
    uint chunk_base = chunk_idx * p.chunk_size;
    uint ci         = min(p.chunk_size, p.n_tokens - chunk_base);
    uint chunk_head = chunk_idx * p.num_value_heads + h;

    device const float* S0 = S_all + h * vd * kd + v * kd;

    for (uint j = 0u; j < ci; j++) {
        uint global_row = chunk_base + j;
        uint head_row   = chunk_head * p.chunk_size + j;
        float gamma_j   = exp(gamma[head_row]);

        // Partial sums over k (32 threads in x, stride 32 covers kd=128)
        float w_dot = 0.0f, q_dot = 0.0f;
        for (uint k = lane_k; k < kd; k += 32u) {
            float s0k = S0[k];
            w_dot += w_sc[head_row * kd + k] * s0k;
            q_dot += q[head_row * kd + k] * s0k;
        }
        // simd_sum over lane_k (simdgroup = lane_v for (32,8,1) layout)
        float w_sum = simd_sum(w_dot);
        float q_sum = simd_sum(q_dot);

        float u_jv = u_sc[head_row * vd + v];
        float R_jv = u_jv - gamma_j * w_sum;
        // Only one thread per (head_row, v) may write out_r to avoid many-writer device race.
        // All lane_k threads compute the same R_jv after simd_sum, so lane_k==0 is the owner.
        if (lane_k == 0u) out_r[head_row * vd + v] = R_jv;
        threadgroup_barrier(mem_flags::mem_device);

        float O_inter = gamma_j * q_sum;
        float O_intra = 0.0f;
        for (uint r = 0u; r <= j; r++) {
            uint  hr_r  = chunk_head * p.chunk_size + r;
            float qkl_jr = qk_l[(chunk_head * p.chunk_size + j) * p.chunk_size + r];
            float R_rv  = out_r[hr_r * vd + v];
            O_intra += qkl_jr * R_rv;
        }

        // Same value across all lane_k; restrict write to lane_k==0 to avoid device race.
        if (lane_k == 0u) {
            raw_out[global_row * p.output_dim + h * vd + v] = (O_inter + O_intra) * p.scale;
        }
    }
}

// Kernel 4: S_end = gamma_end * S0 + R^T @ K_right (outer-product state update).
// Dispatched per-chunk in sequential order after gdn_chunk_residual_output_c32.
// Grid: (ceil(kd/16), ceil(vd/16), num_value_heads). Threads: (16, 16, 1) = 256 per TG.
kernel void gdn_chunk_state_update_c32(
    device float*       S_all      [[buffer(0)]],  // [num_value_heads, vd, kd] read/write
    device const float* r_sc       [[buffer(1)]],  // [chunks, heads, C, value_dim]
    device const float* k_right    [[buffer(2)]],  // [chunks, heads, C, key_dim]
    device const float* gamma_end  [[buffer(3)]],  // [chunks, heads]
    constant GdnChunkParams& p     [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]])
{
    uint k_tile    = tgpig.x;
    uint v_tile    = tgpig.y;
    uint h         = tgpig.z;
    uint chunk_idx = p.active_chunk;
    if (h >= p.num_value_heads) return;

    uint tx = tpitg.x;
    uint ty = tpitg.y;
    uint k  = k_tile * 16u + tx;
    uint v  = v_tile * 16u + ty;
    if (k >= p.key_dim || v >= p.value_dim) return;

    uint kd         = p.key_dim;
    uint vd         = p.value_dim;
    uint chunk_base = chunk_idx * p.chunk_size;
    uint ci         = min(p.chunk_size, p.n_tokens - chunk_base);
    uint chunk_head = chunk_idx * p.num_value_heads + h;

    float g_end  = exp(gamma_end[chunk_head]);
    uint  s_idx  = h * vd * kd + v * kd + k;
    float acc    = g_end * S_all[s_idx];

    for (uint j = 0u; j < ci; j++) {
        uint  hr  = chunk_head * p.chunk_size + j;
        float Rjv = r_sc[hr * vd + v];
        float Krjk = k_right[hr * kd + k];
        acc = fma(Rjv, Krjk, acc);
    }
    S_all[s_idx] = acc;
}

// Kernel 5: Per-token per-head RMS norm + gated SiLU; overwrites gdn_z output.
// Grid: (n_tokens, num_value_heads, 1). Threads: (128, 1, 1).
kernel void gdn_chunk_norm_silu_c32(
    device const float* raw_out    [[buffer(0)]],  // [n_tokens, output_dim]
    device const float* gdn_z_in   [[buffer(1)]],  // [n_tokens, output_dim] Z gate input
    device const float* norm_w     [[buffer(2)]],  // [value_dim]
    device float*       gdn_z_out  [[buffer(3)]],  // [n_tokens, output_dim]
    constant GdnChunkParams& p     [[buffer(4)]],
    uint3 tgpig    [[threadgroup_position_in_grid]],
    uint  tid      [[thread_index_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  sgitg    [[simdgroup_index_in_threadgroup]])
{
    uint row = tgpig.x;
    uint h   = tgpig.y;
    if (row >= p.n_tokens || h >= p.num_value_heads) return;

    uint vd  = p.value_dim;
    uint base = row * p.output_dim + h * vd;
    float x  = (tid < vd) ? raw_out[base + tid] : 0.0f;

    threadgroup float sg_buf[4];

    float sq = simd_sum(x * x);
    if (simd_lane == 0) sg_buf[sgitg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0.0f;
        for (uint si = 0u; si < 4u; si++) s += sg_buf[si];
        sg_buf[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(sg_buf[0] / float(vd) + p.eps);

    if (tid < vd) {
        float normed  = x * inv_rms * norm_w[tid];
        float z       = gdn_z_in[base + tid];
        float silu_z  = z / (1.0f + exp(-z));
        gdn_z_out[base + tid] = normed * silu_z;
    }
}
