//! Metal GPU forward pass for Qwen3.5-2B hybrid attention model.
//!
//! Hybrid architecture: 18 GatedDeltaNet (linear) + 6 GQA (full) layers.
//! Pattern: [linear, linear, linear, full] x 6 = 24 layers.
//!
//! Design decisions:
//! - **All weights in unified memory** (StorageModeShared): zero-copy CPU/GPU access.
//! - **GPU dispatches for projections and FFN**: the big matmuls that dominate compute.
//! - **CPU-side GDN recurrence**: the S matrix update is sequential per-token and only
//!   ~17% of total compute. Runs on CPU reading from shared Metal buffers.
//! - **GPU dispatches for full attention**: Q/K/V projections, QK-norm, partial RoPE,
//!   fused attention, output gating, and O projection all on GPU.
//! - **Single command buffer per layer**: encode all dispatches, commit, wait.
//! - **Qwen3.5 RMS norm**: uses (1 + gamma) shifted normalization.
//! - **Interleaved partial RoPE**: only first rope_dim dimensions rotated.
//!
//! ## Half-precision weight storage (f16)
//!
//! Large projection weight matrices are stored as IEEE-754 half-precision (`f16`/`half`)
//! in Metal buffers to halve memory bandwidth. Activations, accumulators, norms, and
//! outputs remain `f32`. The MSL kernel `matmul_bt_half` loads weight tiles as `half`
//! and widens to `float` before accumulation, producing identical tiling and dispatch
//! geometry as the `f32` path.
//!
//! Buffers stored as f16: all projection weights (QKV, Z, B, A, out, gate, up, down,
//! q/k/v/o_proj) and embed_tokens. Buffers kept as f32: norm weights, bias/log scalars,
//! conv1d weights (small, read by CPU), RoPE tables, activation scratch buffers.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod inner {
    use crate::attention::gdn::GatedDeltaNetState;
    use crate::attention::gdn_fused::GatedDeltaNetFusedScratch;
    #[cfg(debug_assertions)]
    use crate::attention::gdn_fused::{
        conv1d_silu_fused, simd_decay_and_rank1_update, simd_gated_rms_norm, simd_l2_normalize,
        simd_matvec_transpose,
    };
    use crate::model::qwen35::{AttentionWeights, ModelWeights};
    use crate::model::qwen35_config::{GenerateConfig, GenerateOutput, Qwen35Config};
    use crate::sampling::CandidateSet;
    use crate::tokenizer::bpe::BpeTokenizer;
    use crate::tokenizer::common::Tokenizer;
    use crate::weights::q4_weights::quantize_row_q4_0;
    use metal::*;

    // ---------------------------------------------------------------------------
    // MSL Compute Shaders
    // ---------------------------------------------------------------------------

    const MSL_SOURCE: &str = r#"
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

    // Qwen3.5: (1 + gamma) shifted normalization
    for (uint i = lid; i < row_len; i += tgs) {
        x[base + i] = x[base + i] * rms * (1.0f + gamma[i]);
    }
}

// ===== Interleaved Partial RoPE for Qwen3.5 full-attention layers =====
// Pairs are (2i, 2i+1) for i in 0..half_rope_dim.
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

    // Interleaved: rotate (2*pair, 2*pair+1) within each head
    uint idx0 = base + 2 * pair;
    uint idx1 = base + 2 * pair + 1;
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

    for (uint i = lid; i < head_dim; i += tgs) {
        x[base + i] = x[base + i] * rms * (1.0f + gamma[i]);
    }
}

// ===== FlashAttention-style GQA decode (H1+H2+H4+H5) =====
// Single-pass online softmax. One threadgroup per KV head processes all query heads in the
// GQA group simultaneously. QK computed once per (query_head, cache_tile), scores stored in
// threadgroup memory and reused for V accumulation. Q loaded once into threadgroup memory.
// Grid: [num_kv_heads, 1, 1].  Threads: [256, 1, 1] — one thread per output dimension.
kernel void decode_attention(
    device const float* q        [[buffer(0)]],
    device const float* k_cache  [[buffer(1)]],
    device const float* v_cache  [[buffer(2)]],
    device float* out            [[buffer(3)]],
    constant uint& cache_len     [[buffer(4)]],
    constant uint& head_dim      [[buffer(5)]],
    constant uint& num_q_heads   [[buffer(6)]],
    constant uint& num_kv_heads  [[buffer(7)]],
    constant uint& q_dim         [[buffer(8)]],
    constant uint& kv_dim        [[buffer(9)]],
    constant float& scale        [[buffer(10)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]])
{
    if (cache_len == 0) return;
    constexpr uint HEAD_DIM    = 256;
    constexpr uint MAX_GRP     = 8;   // max GQA group size supported
    constexpr uint TILE_TOKENS = 256; // cache tokens per tile; one thread per token during QK

    if (head_dim != HEAD_DIM || num_kv_heads == 0) return;
    const uint kvh = gid;
    if (kvh >= num_kv_heads) return;
    if ((num_q_heads % num_kv_heads) != 0) return;
    const uint group_size = num_q_heads / num_kv_heads;
    if (group_size == 0 || group_size > MAX_GRP) return;
    const uint qh_base = kvh * group_size;

    // Threadgroup memory (~17 KB total for MAX_GRP=8, HEAD_DIM=256, TILE_TOKENS=256)
    threadgroup float q_s    [MAX_GRP * HEAD_DIM];      // Q for all heads in this KV group
    threadgroup float score_s[MAX_GRP * TILE_TOKENS];   // raw QK scores, then unnorm weights
    threadgroup float reduce_s[TILE_TOKENS];             // reduction scratch
    threadgroup float m_s    [MAX_GRP];                  // running max per Q head
    threadgroup float l_s    [MAX_GRP];                  // running denominator per Q head
    threadgroup float alpha_s[MAX_GRP];                  // exp(m_old - m_new) per Q head

    // Initialize running softmax state
    if (lid < group_size) {
        m_s[lid] = -INFINITY;
        l_s[lid] = 0.0f;
    }

    // Load Q for all query heads in this GQA group into threadgroup memory (H4)
    for (uint idx = lid; idx < group_size * HEAD_DIM; idx += tgs) {
        q_s[idx] = q[qh_base * HEAD_DIM + idx];
    }

    // Per-thread register accumulators: thread lid owns output dimension d=lid (H1)
    float acc[MAX_GRP];
    for (uint qi = 0; qi < MAX_GRP; qi++) acc[qi] = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Tiled online-softmax loop (H1 + H2) ===
    for (uint tile_start = 0; tile_start < cache_len; tile_start += TILE_TOKENS) {
        const uint tile_count = min(TILE_TOKENS, cache_len - tile_start);

        // QK scoring: thread lid owns one cache token (lid < tile_count).
        // Loads K once and accumulates dot products for all GQA query heads (H2).
        if (lid < tile_count) {
            float dot[MAX_GRP];
            for (uint qi = 0; qi < MAX_GRP; qi++) dot[qi] = 0.0f;
            const uint k_base = (tile_start + lid) * kv_dim + kvh * HEAD_DIM;
            for (uint d = 0; d < HEAD_DIM; d++) {
                const float kd = k_cache[k_base + d];
                for (uint qi = 0; qi < group_size; qi++) {
                    dot[qi] += q_s[qi * HEAD_DIM + d] * kd;
                }
            }
            for (uint qi = 0; qi < group_size; qi++) {
                score_s[qi * TILE_TOKENS + lid] = dot[qi] * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online-softmax state update per Q head (H5 — proper barriers, separate arrays)
        for (uint qi = 0; qi < group_size; qi++) {
            // Tile max reduction over valid tokens
            reduce_s[lid] = (lid < tile_count) ? score_s[qi * TILE_TOKENS + lid] : -INFINITY;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs >> 1; s > 0; s >>= 1) {
                if (lid < s) reduce_s[lid] = max(reduce_s[lid], reduce_s[lid + s]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const float tile_max = reduce_s[0];

            if (lid == 0) {
                const float m_old = m_s[qi];
                const float m_new = max(m_old, tile_max);
                alpha_s[qi] = isfinite(m_old) ? exp(m_old - m_new) : 0.0f;
                m_s[qi]     = m_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Overwrite raw scores with unnormalized softmax weights p = exp(score - m_new)
            if (lid < tile_count) {
                score_s[qi * TILE_TOKENS + lid] =
                    exp(score_s[qi * TILE_TOKENS + lid] - m_s[qi]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Tile sum reduction of p values
            reduce_s[lid] = (lid < tile_count) ? score_s[qi * TILE_TOKENS + lid] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs >> 1; s > 0; s >>= 1) {
                if (lid < s) reduce_s[lid] += reduce_s[lid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0) {
                l_s[qi] = alpha_s[qi] * l_s[qi] + reduce_s[0];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Rescale previous tile's V accumulators before adding new tile (H1)
        for (uint qi = 0; qi < group_size; qi++) {
            acc[qi] *= alpha_s[qi];
        }

        // V accumulation: thread lid owns output dimension d=lid; reuses p from score_s (H1)
        const uint d = lid;
        for (uint local_t = 0; local_t < tile_count; local_t++) {
            const float v = v_cache[(tile_start + local_t) * kv_dim + kvh * HEAD_DIM + d];
            for (uint qi = 0; qi < group_size; qi++) {
                acc[qi] += score_s[qi * TILE_TOKENS + local_t] * v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization and output write
    for (uint qi = 0; qi < group_size; qi++) {
        const uint qh  = qh_base + qi;
        const float dn = l_s[qi];
        out[qh * HEAD_DIM + lid] = dn > 0.0f ? acc[qi] / dn : 0.0f;
    }
}

// ===== Partitioned flash decode — partial kernel (H3) =====
// One threadgroup per (KV head, partition). Runs the same tiled online-softmax as
// decode_attention but over partition_tokens-length slice of the KV cache.
// Writes (m, l, acc[head_dim]) partial state for each query head in the GQA group.
// Partials layout: partials[((part * num_q_heads + qh) * (HEAD_DIM+2)) + offset]
//   offset 0 = running max m, offset 1 = running sum l, offset 2+ = acc[d].
// Grid: [num_kv_heads, num_partitions, 1].  Threads: [256, 1, 1].
kernel void decode_attention_flash_partial(
    device const float* q           [[buffer(0)]],
    device const float* k_cache     [[buffer(1)]],
    device const float* v_cache     [[buffer(2)]],
    device float* attn_partials     [[buffer(3)]],
    constant uint& cache_len        [[buffer(4)]],
    constant uint& head_dim         [[buffer(5)]],
    constant uint& num_q_heads      [[buffer(6)]],
    constant uint& num_kv_heads     [[buffer(7)]],
    constant uint& q_dim            [[buffer(8)]],
    constant uint& kv_dim           [[buffer(9)]],
    constant float& scale           [[buffer(10)]],
    constant uint& partition_tokens [[buffer(11)]],
    uint3 gid3  [[threadgroup_position_in_grid]],
    uint3 lid3  [[thread_position_in_threadgroup]],
    uint3 tgs3  [[threads_per_threadgroup]])
{
    constexpr uint HEAD_DIM    = 256;
    constexpr uint MAX_GRP     = 8;
    constexpr uint TILE_TOKENS = 256;
    constexpr uint STRIDE      = HEAD_DIM + 2; // m + l + acc[256]

    const uint lid          = lid3.x;
    const uint tgs          = tgs3.x;
    const uint kvh          = gid3.x;
    const uint partition_id = gid3.y;
    if (head_dim != HEAD_DIM || partition_tokens == 0 || num_kv_heads == 0) return;
    if (kvh >= num_kv_heads) return;
    if ((num_q_heads % num_kv_heads) != 0) return;

    const uint part_start = partition_id * partition_tokens;
    if (part_start >= cache_len) return;
    const uint part_end = min(cache_len, part_start + partition_tokens);

    const uint group_size = num_q_heads / num_kv_heads;
    if (group_size == 0 || group_size > MAX_GRP) return;
    const uint qh_base = kvh * group_size;

    threadgroup float q_s    [MAX_GRP * HEAD_DIM];
    threadgroup float score_s[MAX_GRP * TILE_TOKENS];
    threadgroup float reduce_s[TILE_TOKENS];
    threadgroup float m_s    [MAX_GRP];
    threadgroup float l_s    [MAX_GRP];
    threadgroup float alpha_s[MAX_GRP];

    if (lid < group_size) {
        m_s[lid] = -INFINITY;
        l_s[lid] = 0.0f;
    }
    for (uint idx = lid; idx < group_size * HEAD_DIM; idx += tgs) {
        q_s[idx] = q[qh_base * HEAD_DIM + idx];
    }

    float acc[MAX_GRP];
    for (uint qi = 0; qi < MAX_GRP; qi++) acc[qi] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = part_start; tile_start < part_end; tile_start += TILE_TOKENS) {
        const uint tile_count = min(TILE_TOKENS, part_end - tile_start);

        if (lid < tile_count) {
            float dot[MAX_GRP];
            for (uint qi = 0; qi < MAX_GRP; qi++) dot[qi] = 0.0f;
            const uint k_base = (tile_start + lid) * kv_dim + kvh * HEAD_DIM;
            for (uint d = 0; d < HEAD_DIM; d++) {
                const float kd = k_cache[k_base + d];
                for (uint qi = 0; qi < group_size; qi++) {
                    dot[qi] += q_s[qi * HEAD_DIM + d] * kd;
                }
            }
            for (uint qi = 0; qi < group_size; qi++) {
                score_s[qi * TILE_TOKENS + lid] = dot[qi] * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint qi = 0; qi < group_size; qi++) {
            reduce_s[lid] = (lid < tile_count) ? score_s[qi * TILE_TOKENS + lid] : -INFINITY;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs >> 1; s > 0; s >>= 1) {
                if (lid < s) reduce_s[lid] = max(reduce_s[lid], reduce_s[lid + s]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const float tile_max = reduce_s[0];
            if (lid == 0) {
                const float m_old = m_s[qi];
                const float m_new = max(m_old, tile_max);
                alpha_s[qi] = isfinite(m_old) ? exp(m_old - m_new) : 0.0f;
                m_s[qi]     = m_new;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < tile_count) {
                score_s[qi * TILE_TOKENS + lid] =
                    exp(score_s[qi * TILE_TOKENS + lid] - m_s[qi]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            reduce_s[lid] = (lid < tile_count) ? score_s[qi * TILE_TOKENS + lid] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs >> 1; s > 0; s >>= 1) {
                if (lid < s) reduce_s[lid] += reduce_s[lid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0) {
                l_s[qi] = alpha_s[qi] * l_s[qi] + reduce_s[0];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint qi = 0; qi < group_size; qi++) acc[qi] *= alpha_s[qi];

        const uint d = lid;
        for (uint local_t = 0; local_t < tile_count; local_t++) {
            const float v = v_cache[(tile_start + local_t) * kv_dim + kvh * HEAD_DIM + d];
            for (uint qi = 0; qi < group_size; qi++) {
                acc[qi] += score_s[qi * TILE_TOKENS + local_t] * v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write partial state to global buffer
    const uint d = lid;
    for (uint qi = 0; qi < group_size; qi++) {
        const uint qh   = qh_base + qi;
        const uint base = (partition_id * num_q_heads + qh) * STRIDE;
        if (lid == 0) {
            attn_partials[base + 0] = m_s[qi];
            attn_partials[base + 1] = l_s[qi];
        }
        attn_partials[base + 2 + d] = acc[qi];
    }
}

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

    // Phase 2: normed_out = residual_out * rms * (1 + gamma)
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

// ===== Q4_0 GEMV Decode: 4-bit x float32 with simd_sum reduction =====
// Q4_0 block: [f16 scale (2B)][16 bytes packed 4-bit] = 18 bytes per 32 weights.
// Packed as unsigned offset: real_value = (nibble - 8) * scale.
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
    const uint nb = K / 32;           // number of Q4_0 blocks per row
    const uint row_bytes = nb * 18;   // 18 bytes per block
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
        for (uint i = 0; i < 8; i++) yl[i] = yb[i];
        yb += ib_stride * 32;

        for (uint row = 0; row < NR; row++) {
            uint r = first_row + row;
            if (r >= N) continue;
            device const uchar* base = (device const uchar*)(qweight + r * row_bytes + ib * 18);
            half d = *((device const half*)base);
            device const uchar* qs = base + 2 + il * 4;  // 4 bytes = 8 nibbles

            float sumq = 0.0f;
            for (uint i = 0; i < 4; i++) {
                uchar byte_val = qs[i];
                float v0 = float(int(byte_val & 0xF) - 8);
                float v1 = float(int(byte_val >> 4) - 8);
                sumq += v0 * yl[i * 2] + v1 * yl[i * 2 + 1];
            }
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
    device const half*  in_proj_b    [[buffer(4)]],   // [num_key_heads, hidden_size]
    device const half*  in_proj_a    [[buffer(5)]],   // [num_key_heads, hidden_size]
    device const float* a_log        [[buffer(6)]],   // [num_key_heads]
    device const float* dt_bias      [[buffer(7)]],   // [num_key_heads]
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

    // Beta = sigmoid(hidden @ in_proj_b[k_head]^T)
    device const half* wb = in_proj_b + k_head * hd;
    float bp = 0.0f;
    for (uint i = tid; i < hd; i += 128) bp += float(wb[i]) * hidden_in[i];
    bp = simd_sum(bp);
    if (simd_lane == 0) sg_buf[sgitg] = bp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { bp = 0; for (uint s = 0; s < 4; s++) bp += sg_buf[s]; sg_buf[0] = 1.0f / (1.0f + exp(-bp)); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float beta_val = sg_buf[0];

    // Alpha
    device const half* wa = in_proj_a + k_head * hd;
    float ap = 0.0f;
    for (uint i = tid; i < hd; i += 128) ap += float(wa[i]) * hidden_in[i];
    ap = simd_sum(ap);
    if (simd_lane == 0) sg_buf[sgitg] = ap;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { ap = 0; for (uint s = 0; s < 4; s++) ap += sg_buf[s]; sg_buf[0] = ap; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float alpha_val = sg_buf[0];

    // L2 normalize Q
    float q_val = (tid < kd) ? conv_out[k_head * kd + tid] : 0.0f;
    float q_sq = simd_sum(q_val * q_val);
    if (simd_lane == 0) sg_buf[sgitg] = q_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { q_sq = 0; for (uint s = 0; s < 4; s++) q_sq += sg_buf[s]; sg_buf[0] = q_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qs = (sg_buf[0] > 1e-12f) ? rsqrt(sg_buf[0]) : 0.0f;
    q_val *= qs;
    if (tid < kd) q_tg[tid] = q_val;

    // L2 normalize K
    float k_val = (tid < kd) ? conv_out[p.q_total + k_head * kd + tid] : 0.0f;
    float k_sq = simd_sum(k_val * k_val);
    if (simd_lane == 0) sg_buf[sgitg] = k_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { k_sq = 0; for (uint s = 0; s < 4; s++) k_sq += sg_buf[s]; sg_buf[0] = k_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ks = (sg_buf[0] > 1e-12f) ? rsqrt(sg_buf[0]) : 0.0f;
    k_val *= ks;
    if (tid < kd) k_tg[tid] = k_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Decay gate
    float a = exp(a_log[k_head]);
    float sp = log(1.0f + exp(alpha_val + dt_bias[k_head]));
    float g = exp(-a * sp);

    // k[:] @ q[:] — shared dot product, same for all value rows in this head
    float kq_part = (tid < kd) ? (k_tg[tid] * q_tg[tid]) : 0.0f;
    kq_part = simd_sum(kq_part);
    if (simd_lane == 0) sg_buf[sgitg] = kq_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float kq = 0.0f; for (uint s = 0; s < 4; s++) kq += sg_buf[s]; sg_buf[0] = kq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_dot_q = sg_buf[0];

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

    // Beta = sigmoid(hidden @ in_proj_b[k_head])
    device const half* wb = in_proj_b + k_head * hd;
    float bp = 0.0f;
    for (uint i = tid; i < hd; i += 128) bp += float(wb[i]) * hidden_in[i];
    bp = simd_sum(bp);
    if (simd_lane == 0) sg_buf[sgitg] = bp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { bp = 0; for (uint s = 0; s < 4; s++) bp += sg_buf[s]; sg_buf[0] = 1.0f / (1.0f + exp(-bp)); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float beta_val = sg_buf[0];

    // Alpha = hidden @ in_proj_a[k_head]
    device const half* wa = in_proj_a + k_head * hd;
    float ap = 0.0f;
    for (uint i = tid; i < hd; i += 128) ap += float(wa[i]) * hidden_in[i];
    ap = simd_sum(ap);
    if (simd_lane == 0) sg_buf[sgitg] = ap;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { ap = 0; for (uint s = 0; s < 4; s++) ap += sg_buf[s]; sg_buf[0] = ap; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float alpha_val = sg_buf[0];

    // L2 normalize Q — kd=128 == thread_count, no ternary guard needed
    float q_val = conv_out[k_head * kd + tid];
    float q_sq = simd_sum(q_val * q_val);
    if (simd_lane == 0) sg_buf[sgitg] = q_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { q_sq = 0; for (uint s = 0; s < 4; s++) q_sq += sg_buf[s]; sg_buf[0] = q_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qs = (sg_buf[0] > 1e-12f) ? rsqrt(sg_buf[0]) : 0.0f;
    q_tg[tid] = q_val * qs;

    // L2 normalize K
    float k_val = conv_out[p.q_total + k_head * kd + tid];
    float k_sq = simd_sum(k_val * k_val);
    if (simd_lane == 0) sg_buf[sgitg] = k_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { k_sq = 0; for (uint s = 0; s < 4; s++) k_sq += sg_buf[s]; sg_buf[0] = k_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ks_inv = (sg_buf[0] > 1e-12f) ? rsqrt(sg_buf[0]) : 0.0f;
    k_tg[tid] = k_val * ks_inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Decay gate
    float a = exp(a_log[k_head]);
    float sp = log(1.0f + exp(alpha_val + dt_bias[k_head]));
    float g = exp(-a * sp);

    // k dot q (scalar, same for all value rows in this head)
    float kq_part = k_tg[tid] * q_tg[tid];
    kq_part = simd_sum(kq_part);
    if (simd_lane == 0) sg_buf[sgitg] = kq_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float kq = 0.0f; for (uint s = 0; s < 4; s++) kq += sg_buf[s]; sg_buf[0] = kq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_dot_q = sg_buf[0];

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
// H3: gdn_precompute_keys hoists key-head preprocessing (beta, g, q_norm, k_norm, k_dot_q)
//     that the fused kernel repeated 3x per key head (ratio = 48 vh / 16 kh = 3).
//     Scratch layout per key head: [q_norm(0..128) | k_norm(128..256) | beta | g | k_dot_q]
// H1: gdn_recurrence_sharded expands from 48 TG/layer to 1536 TG/layer (32x48).
//     Each TG owns 4 S-rows x 32 key lanes; simd_sum reduces kv/old_q within a row.
// gdn_norm_silu: RMSNorm + SiLU gate on raw_out, writes result to gdn_qkvz for output proj.

kernel void gdn_precompute_keys(
    device const float* conv_out     [[buffer(0)]],
    device const float* hidden_in    [[buffer(1)]],
    device const half*  in_proj_b    [[buffer(2)]],
    device const half*  in_proj_a    [[buffer(3)]],
    device const float* a_log        [[buffer(4)]],
    device const float* dt_bias      [[buffer(5)]],
    device float* key_scratch        [[buffer(6)]],
    constant GdnRecurParams& p       [[buffer(7)]],
    uint k_head    [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint sgitg     [[simdgroup_index_in_threadgroup]])
{
    constexpr uint kd = 128;
    constexpr uint hd = 5120;

    threadgroup float sg_buf[4];

    device const half* wb = in_proj_b + k_head * hd;
    float bp = 0.0f;
    for (uint i = tid; i < hd; i += kd) bp += float(wb[i]) * hidden_in[i];
    bp = simd_sum(bp);
    if (simd_lane == 0) sg_buf[sgitg] = bp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { bp = 0; for (uint s = 0; s < 4; s++) bp += sg_buf[s]; sg_buf[0] = 1.0f / (1.0f + exp(-bp)); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float beta_val = sg_buf[0];

    device const half* wa = in_proj_a + k_head * hd;
    float ap = 0.0f;
    for (uint i = tid; i < hd; i += kd) ap += float(wa[i]) * hidden_in[i];
    ap = simd_sum(ap);
    if (simd_lane == 0) sg_buf[sgitg] = ap;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { ap = 0; for (uint s = 0; s < 4; s++) ap += sg_buf[s]; sg_buf[0] = ap; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float alpha_val = sg_buf[0];

    float q_val = conv_out[k_head * kd + tid];
    float q_sq  = simd_sum(q_val * q_val);
    if (simd_lane == 0) sg_buf[sgitg] = q_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { q_sq = 0; for (uint s = 0; s < 4; s++) q_sq += sg_buf[s]; sg_buf[0] = q_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float qs          = (sg_buf[0] > 1e-12f) ? rsqrt(sg_buf[0]) : 0.0f;
    float q_norm_val  = q_val * qs;

    float k_val = conv_out[p.q_total + k_head * kd + tid];
    float k_sq  = simd_sum(k_val * k_val);
    if (simd_lane == 0) sg_buf[sgitg] = k_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { k_sq = 0; for (uint s = 0; s < 4; s++) k_sq += sg_buf[s]; sg_buf[0] = k_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ks_inv      = (sg_buf[0] > 1e-12f) ? rsqrt(sg_buf[0]) : 0.0f;
    float k_norm_val  = k_val * ks_inv;

    float a  = exp(a_log[k_head]);
    float sp = log(1.0f + exp(alpha_val + dt_bias[k_head]));
    float g  = exp(-a * sp);

    float kq_part = k_norm_val * q_norm_val;
    kq_part = simd_sum(kq_part);
    if (simd_lane == 0) sg_buf[sgitg] = kq_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float kq = 0.0f; for (uint s = 0; s < 4; s++) kq += sg_buf[s]; sg_buf[0] = kq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_dot_q = sg_buf[0];

    device float* ks_out = key_scratch + k_head * (2 * kd + 3);
    ks_out[tid]      = q_norm_val;
    ks_out[kd + tid] = k_norm_val;
    if (tid == 0) {
        ks_out[2 * kd]     = beta_val;
        ks_out[2 * kd + 1] = g;
        ks_out[2 * kd + 2] = k_dot_q;
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
    device const float* ks = key_scratch + k_head * (2 * kd + 3);

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

    float beta    = ks[2 * kd];
    float g       = ks[2 * kd + 1];
    float k_dot_q = ks[2 * kd + 2];
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

// ===== Q4_0 batch GEMM: Y[M,N] = X[M,K] @ Q4_W[N, K/32*18]^T =====
// Scalar fallback cloned from gemm_q8: NM=4 M rows, NR=2 output columns,
// NSG=4 SIMD groups. Q4_0 blocks are [f16 scale][16 packed nibble bytes].
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
    const uint row_bytes = nb * 18;
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
            for (uint i = 0; i < 8; i++) xv[i] = xb[i];

            for (uint ri = 0; ri < NR; ri++) {
                uint col = first_col + ri;
                if (col >= N) continue;

                uint block_start = col * row_bytes + ib * 18;
                ushort scale_bits = ushort(QW[block_start]) |
                    (ushort(QW[block_start + 1]) << 8);
                half dh = as_type<half>(scale_bits);
                float d = float(dh);

                float sumq = 0.0f;
                for (uint byte_i = 0; byte_i < 4; byte_i++) {
                    uchar packed = QW[block_start + 2 + il * 4 + byte_i];
                    int q0 = int(packed & 0x0f) - 8;
                    int q1 = int(packed >> 4) - 8;
                    sumq += (float(q0) * d) * xv[byte_i * 2] +
                            (float(q1) * d) * xv[byte_i * 2 + 1];
                }

                sumf[mi * NR + ri] += sumq;
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

    // Phase 2: normalize src in-place (src still holds original values)
    for (uint i = lid; i < row_len; i += tgs) {
        src[i] = src[i] * rms * (1.0f + gamma[i]);
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
"#;

    const MSL_Q4_TILED_SOURCE: &str = concat!(
        "#include <metal_stdlib>\n",
        "#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 300)\n",
        "#include <metal_simdgroup_matrix>\n",
        "using namespace metal;\n",
        "kernel void gemm_q4_tiled(\n",
        "    device const uchar* QW [[buffer(0)]],\n",
        "    device const float* X  [[buffer(1)]],\n",
        "    device float* Y        [[buffer(2)]],\n",
        "    constant uint& M       [[buffer(3)]],\n",
        "    constant uint& N       [[buffer(4)]],\n",
        "    constant uint& K       [[buffer(5)]],\n",
        "    uint3 tg               [[threadgroup_position_in_grid]],\n",
        "    uint tid               [[thread_index_in_threadgroup]],\n",
        "    uint lane              [[thread_index_in_simdgroup]],\n",
        "    uint sg                [[simdgroup_index_in_threadgroup]])\n",
        "{\n",
        "    constexpr uint BM = 64;\n",
        "    constexpr uint BN = 32;\n",
        "    constexpr uint BK = 32;\n",
        "    constexpr uint BN_PAD = 40;\n",
        "    constexpr uint Q4_PAD = 32;\n",
        "    constexpr uint THREADS = 128;\n",
        "    const uint m0 = tg.y * BM;\n",
        "    const uint n0 = tg.x * BN;\n",
        "    const uint nb = K / 32;\n",
        "    const uint row_bytes = nb * 18;\n",
        "    threadgroup float Xtg[64][32];\n",
        "    threadgroup uchar Qraw[32][32];\n",
        "    threadgroup float Wtg[32][40];\n",
        "    threadgroup float Ytg[32][16];\n",
        "    threadgroup float Zero[8][8];\n",
        "    if (tid < 64) { Zero[tid / 8][tid % 8] = 0.0f; }\n",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n",
        "    const uint sg_m_base = (sg / 2) * 32;\n",
        "    const uint sg_n_base = (sg % 2) * 16;\n",
        "    simdgroup_float8x8 acc00,acc01,acc10,acc11,acc20,acc21,acc30,acc31;\n",
        "    simdgroup_load(acc00,&Zero[0][0],8); simdgroup_load(acc01,&Zero[0][0],8);\n",
        "    simdgroup_load(acc10,&Zero[0][0],8); simdgroup_load(acc11,&Zero[0][0],8);\n",
        "    simdgroup_load(acc20,&Zero[0][0],8); simdgroup_load(acc21,&Zero[0][0],8);\n",
        "    simdgroup_load(acc30,&Zero[0][0],8); simdgroup_load(acc31,&Zero[0][0],8);\n",
        "    for (uint k0 = 0; k0 < K; k0 += BK) {\n",
        "        const uint kb = k0 / 32;\n",
        "        for (uint i = tid; i < BM*BK; i += THREADS) {\n",
        "            uint mi=i/BK; uint kk=i%BK; uint gm=m0+mi; uint gk=k0+kk;\n",
        "            Xtg[mi][kk]=(gm<M&&gk<K)?X[gm*K+gk]:0.0f; }\n",
        "        for (uint i = tid; i < BN*Q4_PAD; i += THREADS) {\n",
        "            uint ni=i/Q4_PAD; uint off=i%Q4_PAD; uint gn=n0+ni; uchar v=0;\n",
        "            if(gn<N&&off<18){v=QW[gn*row_bytes+kb*18+off];} Qraw[ni][off]=v; }\n",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n",
        "        for (uint i = tid; i < BN*16; i += THREADS) {\n",
        "            uint ni=i/16; uint byte_i=i%16;\n",
        "            ushort sb=ushort(Qraw[ni][0])|(ushort(Qraw[ni][1])<<8);\n",
        "            float d=float(as_type<half>(sb));\n",
        "            uchar packed=Qraw[ni][2+byte_i];\n",
        "            Wtg[2*byte_i+0][ni]=float(int(packed&0x0f)-8)*d;\n",
        "            Wtg[2*byte_i+1][ni]=float(int(packed>>4)-8)*d; }\n",
        "        for (uint i = tid; i < BK*8; i += THREADS) {\n",
        "            uint kk=i/8; uint pn=BN+(i%8); Wtg[kk][pn]=0.0f; }\n",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n",
        "        for (uint kk = 0; kk < BK; kk += 8) {\n",
        "            simdgroup_float8x8 a0,a1,a2,a3,b0,b1;\n",
        "            simdgroup_load(a0,&Xtg[sg_m_base+ 0][kk],BK);\n",
        "            simdgroup_load(a1,&Xtg[sg_m_base+ 8][kk],BK);\n",
        "            simdgroup_load(a2,&Xtg[sg_m_base+16][kk],BK);\n",
        "            simdgroup_load(a3,&Xtg[sg_m_base+24][kk],BK);\n",
        "            simdgroup_load(b0,&Wtg[kk][sg_n_base+0],BN_PAD);\n",
        "            simdgroup_load(b1,&Wtg[kk][sg_n_base+8],BN_PAD);\n",
        "            simdgroup_multiply_accumulate(acc00,a0,b0,acc00);\n",
        "            simdgroup_multiply_accumulate(acc01,a0,b1,acc01);\n",
        "            simdgroup_multiply_accumulate(acc10,a1,b0,acc10);\n",
        "            simdgroup_multiply_accumulate(acc11,a1,b1,acc11);\n",
        "            simdgroup_multiply_accumulate(acc20,a2,b0,acc20);\n",
        "            simdgroup_multiply_accumulate(acc21,a2,b1,acc21);\n",
        "            simdgroup_multiply_accumulate(acc30,a3,b0,acc30);\n",
        "            simdgroup_multiply_accumulate(acc31,a3,b1,acc31); }\n",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n",
        "    }\n",
        "    for (uint ssg = 0; ssg < 4; ssg++) {\n",
        "        if (sg == ssg) {\n",
        "            simdgroup_store(acc00,&Ytg[ 0][0],16); simdgroup_store(acc01,&Ytg[ 0][8],16);\n",
        "            simdgroup_store(acc10,&Ytg[ 8][0],16); simdgroup_store(acc11,&Ytg[ 8][8],16);\n",
        "            simdgroup_store(acc20,&Ytg[16][0],16); simdgroup_store(acc21,&Ytg[16][8],16);\n",
        "            simdgroup_store(acc30,&Ytg[24][0],16); simdgroup_store(acc31,&Ytg[24][8],16); }\n",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n",
        "        uint smb=(ssg/2)*32; uint snb=(ssg%2)*16;\n",
        "        for (uint i = tid; i < 512; i += THREADS) {\n",
        "            uint lm=i/16; uint ln=i%16;\n",
        "            uint gm=m0+smb+lm; uint gn=n0+snb+ln;\n",
        "            if(gm<M&&gn<N){Y[gm*N+gn]=Ytg[lm][ln];} }\n",
        "        threadgroup_barrier(mem_flags::mem_threadgroup);\n",
        "    }\n",
        "}\n",
        "#endif\n"
    );

    // ---------------------------------------------------------------------------
    // GPU Buffer Structures
    // ---------------------------------------------------------------------------

    /// A quantized weight buffer, paired with the byte offset within that buffer where
    /// the actual Q4 payload starts.
    ///
    /// For whole-file mmap buffers (H1 optimisation): `buffer` spans the entire
    /// mmapped file and `payload_offset` skips the file header.  For regular
    /// `new_buffer_with_data` buffers (Q8_0 path and concatenated Q4 tensors):
    /// `payload_offset` is 0.  Dispatch helpers always bind via
    /// `enc.set_buffer(slot, Some(&qw.buffer), qw.payload_offset)` so both cases
    /// are handled uniformly.
    struct Q4WeightBuf {
        buffer: Buffer,
        /// Byte offset of Q4 payload within `buffer`.
        payload_offset: u64,
        // Keeps the mmap alive as long as the Metal buffer needs the memory.
        _mmap: Option<memmap2::Mmap>,
    }

    impl Q4WeightBuf {
        /// Wrap a normally-allocated Metal buffer (payload at offset 0).
        fn from_buffer(buffer: Buffer) -> Self {
            Q4WeightBuf {
                buffer,
                payload_offset: 0,
                _mmap: None,
            }
        }

        /// Wrap a no-copy Metal buffer backed by `mmap` with a non-zero payload offset.
        fn from_mmap(buffer: Buffer, payload_offset: u64, mmap: memmap2::Mmap) -> Self {
            Q4WeightBuf {
                buffer,
                payload_offset,
                _mmap: Some(mmap),
            }
        }
    }

    /// Open `path`, mmap the whole file, and register it as a Metal no-copy
    /// `StorageModeShared` buffer.  Stores the mmap owner inside the returned
    /// [`Q4WeightBuf`] so the pages remain valid for the lifetime of the buffer.
    ///
    /// # Safety invariant
    /// The mmap is read-only (`MAP_PRIVATE`) and the model files must not be
    /// modified while the process is running.
    #[cfg(feature = "metal-gpu")]
    fn mmap_q4_weight(
        device: &Device,
        path: &std::path::Path,
        label: &str,
    ) -> Result<Q4WeightBuf, String> {
        use crate::weights::q4_weights::read_q4_header;

        let file = std::fs::File::open(path)
            .map_err(|e| format!("failed to open {}: {e}", path.display()))?;
        let header = read_q4_header(&file)
            .map_err(|e| format!("failed to parse Q4 header {}: {e}", path.display()))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file) }
            .map_err(|e| format!("failed to mmap {}: {e}", path.display()))?;
        assert!(
            mmap.len() as u64 >= header.payload_offset,
            "Q4 file truncated: {} bytes < payload_offset {}",
            mmap.len(),
            header.payload_offset
        );

        // The mmap pointer is page-aligned (guaranteed by the OS).  We create the Metal
        // buffer over the *whole* file so the pointer alignment requirement of
        // new_buffer_with_bytes_no_copy is satisfied.  The Q4 payload is then addressed
        // via enc.set_buffer(slot, ..., header.payload_offset) at dispatch time.
        let buf = device.new_buffer_with_bytes_no_copy(
            mmap.as_ptr().cast(),
            mmap.len() as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        buf.set_label(label);

        Ok(Q4WeightBuf::from_mmap(buf, header.payload_offset, mmap))
    }

    /// Merge two Q4 files into a single concatenated Q4 file and write it to `out_path`.
    ///
    /// Reads only the raw bytes (no deserialization) and prepends a new KHQ4 header
    /// reflecting the merged shape.  Uses a temp-file + atomic rename so a crashed
    /// mid-write never leaves a partial file at the final path.
    ///
    /// Returns `Err` if the model directory is read-only or I/O fails — callers
    /// must fall back to the CPU concat path in that case.
    #[cfg(feature = "metal-gpu")]
    fn write_merged_qkvz(
        qkv_path: &std::path::Path,
        z_path: &std::path::Path,
        out_path: &std::path::Path,
    ) -> Result<(), String> {
        use crate::weights::q4_weights::read_q4_header;
        use std::io::{Read, Seek, SeekFrom, Write};

        // Open and parse headers — then seek back to payload start for a direct byte copy.
        let qkv_file = std::fs::File::open(qkv_path)
            .map_err(|e| format!("open {}: {e}", qkv_path.display()))?;
        let qkv_hdr =
            read_q4_header(&qkv_file).map_err(|e| format!("header {}: {e}", qkv_path.display()))?;
        let mut qkv_rdr = qkv_file;
        qkv_rdr
            .seek(SeekFrom::Start(qkv_hdr.payload_offset))
            .map_err(|e| format!("seek {}: {e}", qkv_path.display()))?;
        let qkv_payload_len = std::fs::metadata(qkv_path)
            .map_err(|e| format!("metadata {}: {e}", qkv_path.display()))?
            .len()
            - qkv_hdr.payload_offset;
        let mut qkv_payload = Vec::with_capacity(qkv_payload_len as usize);
        qkv_rdr
            .read_to_end(&mut qkv_payload)
            .map_err(|e| format!("read {}: {e}", qkv_path.display()))?;

        let z_file =
            std::fs::File::open(z_path).map_err(|e| format!("open {}: {e}", z_path.display()))?;
        let z_hdr =
            read_q4_header(&z_file).map_err(|e| format!("header {}: {e}", z_path.display()))?;
        let mut z_rdr = z_file;
        z_rdr
            .seek(SeekFrom::Start(z_hdr.payload_offset))
            .map_err(|e| format!("seek {}: {e}", z_path.display()))?;
        let z_payload_len = std::fs::metadata(z_path)
            .map_err(|e| format!("metadata {}: {e}", z_path.display()))?
            .len()
            - z_hdr.payload_offset;
        let mut z_payload = Vec::with_capacity(z_payload_len as usize);
        z_rdr
            .read_to_end(&mut z_payload)
            .map_err(|e| format!("read {}: {e}", z_path.display()))?;

        // Merged shape: rows = qkv_rows + z_rows, cols = hidden (shared)
        let merged_rows = qkv_hdr.shape[0] + z_hdr.shape[0];
        let cols = if qkv_hdr.shape.len() >= 2 {
            qkv_hdr.shape[1]
        } else {
            1
        };
        let original_len = qkv_hdr.original_len + z_hdr.original_len;

        // Write to a temp file then rename atomically so partial writes are never trusted.
        let tmp = out_path.with_extension("q4.tmp");
        let write_result = (|| -> Result<(), String> {
            let mut f = std::io::BufWriter::new(
                std::fs::File::create(&tmp)
                    .map_err(|e| format!("create {}: {e}", tmp.display()))?,
            );
            // KHQ4 header: magic(4) + version(4) + ndim=2(4) + shape[0](8) + shape[1](8) + original_len(8)
            f.write_all(b"KHQ4").map_err(|e| e.to_string())?;
            f.write_all(&1u32.to_le_bytes())
                .map_err(|e| e.to_string())?;
            f.write_all(&2u32.to_le_bytes())
                .map_err(|e| e.to_string())?;
            f.write_all(&(merged_rows as u64).to_le_bytes())
                .map_err(|e| e.to_string())?;
            f.write_all(&(cols as u64).to_le_bytes())
                .map_err(|e| e.to_string())?;
            f.write_all(&(original_len as u64).to_le_bytes())
                .map_err(|e| e.to_string())?;
            f.write_all(&qkv_payload).map_err(|e| e.to_string())?;
            f.write_all(&z_payload).map_err(|e| e.to_string())?;
            Ok(())
        })();

        if write_result.is_err() {
            let _ = std::fs::remove_file(&tmp);
            return write_result;
        }

        std::fs::rename(&tmp, out_path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!("rename: {e}")
        })
    }

    /// Weights for a GatedDeltaNet (linear attention) layer on GPU.
    struct MetalGdnLayerWeights {
        in_proj_qkv: Q4WeightBuf,  // [qkv_dim, hidden] — Q4/Q8
        in_proj_z: Q4WeightBuf,    // [output_dim, hidden] — Q4/Q8
        in_proj_qkvz: Q4WeightBuf, // [qkv_dim+output_dim, hidden] — Q4/Q8; concat of qkv||z rows for fused decode GEMV
        in_proj_b: Buffer,         // [num_heads, hidden] — f16 (CPU-read via unified mem)
        in_proj_a: Buffer,         // [num_heads, hidden] — f16 (CPU-read via unified mem)
        a_log: Buffer,             // [num_heads] — f32 (small, precision-sensitive)
        dt_bias: Buffer,           // [num_heads] — f32 (small, precision-sensitive)
        conv1d_weight: Buffer,     // [qkv_dim, kernel_size] — f32 (small, CPU-read)
        norm_weight: Buffer,       // [value_dim] — f32 (small, CPU-read)
        out_proj: Q4WeightBuf,     // [hidden, output_dim] — Q4/Q8
    }

    /// Weights for a full-attention (GQA) layer on GPU.
    struct MetalFullLayerWeights {
        q_proj: Q4WeightBuf, // [2*q_dim, hidden] (Q + gate_z interleaved) — Q4/Q8
        k_proj: Q4WeightBuf, // [kv_dim, hidden] — Q4/Q8
        v_proj: Q4WeightBuf, // [kv_dim, hidden] — Q4/Q8
        o_proj: Q4WeightBuf, // [hidden, q_dim] — Q4/Q8
        q_norm: Buffer,      // [head_dim] — f32 (small, precision-sensitive)
        k_norm: Buffer,      // [head_dim] — f32 (small, precision-sensitive)
    }

    /// Per-layer weights on GPU, discriminated by layer type.
    enum MetalLayerAttnWeights {
        Linear(MetalGdnLayerWeights),
        Full(MetalFullLayerWeights),
    }

    /// Common per-layer weights (norms + MLP) on GPU.
    struct MetalCommonLayerWeights {
        input_layernorm: Buffer,          // [hidden] — f32 (norm)
        post_attention_layernorm: Buffer, // [hidden] — f32 (norm)
        gate_proj: Q4WeightBuf,           // [intermediate, hidden] — Q4/Q8
        up_proj: Q4WeightBuf,             // [intermediate, hidden] — Q4/Q8
        down_proj: Q4WeightBuf,           // [hidden, intermediate] — Q4/Q8
    }

    /// Maximum draft tokens verified in one MTP speculation round (first=target, second=MTP draft).
    const MTP_VERIFY_MAX_TOKENS: usize = 2;

    struct MetalMtpDenseMlpWeights {
        gate_proj: Q4WeightBuf, // [intermediate, hidden]
        up_proj: Q4WeightBuf,   // [intermediate, hidden]
        down_proj: Q4WeightBuf, // [hidden, intermediate]
    }

    struct MetalMtpLayerWeights {
        input_layernorm: Buffer,          // [hidden]
        post_attention_layernorm: Buffer, // [hidden]
        q_proj: Q4WeightBuf,              // [2*q_dim, hidden]
        k_proj: Q4WeightBuf,              // [kv_dim, hidden]
        v_proj: Q4WeightBuf,              // [kv_dim, hidden]
        o_proj: Q4WeightBuf,              // [hidden, q_dim]
        q_norm: Buffer,                   // [head_dim]
        k_norm: Buffer,                   // [head_dim]
        mlp: MetalMtpDenseMlpWeights,
    }

    struct MetalMtpWeights {
        fc: Q4WeightBuf,               // [hidden, 2*hidden]
        pre_fc_norm_embedding: Buffer, // [hidden]
        pre_fc_norm_hidden: Buffer,    // [hidden]
        layers: Vec<MetalMtpLayerWeights>,
        norm: Buffer, // [hidden]
    }

    struct MetalMtpCache {
        k_buf: Buffer,
        v_buf: Buffer,
        seq_len: usize,
        max_cache_len: usize,
    }

    impl MetalMtpCache {
        fn reset(&mut self) {
            self.seq_len = 0;
        }
    }

    struct MetalMtpActivations {
        fused: Buffer,       // [2*hidden] — normed embedding concat normed hidden
        hidden: Buffer,      // [hidden]
        residual: Buffer,    // [hidden]
        q: Buffer,           // [2*q_dim] interleaved Q+gate from q_proj
        q_separated: Buffer, // [q_dim] Q after scatter
        gate_z: Buffer,      // [q_dim] gate after scatter
        k: Buffer,           // [kv_dim]
        v: Buffer,           // [kv_dim]
        attn_out: Buffer,    // [q_dim.max(hidden)]
        gate: Buffer,        // [intermediate]
        up: Buffer,          // [intermediate]
        ffn_out: Buffer,     // [hidden]
        logits: Buffer,      // [vocab]
    }

    struct MetalMtpRuntime {
        weights: MetalMtpWeights,
        cache: MetalMtpCache,
        activations: MetalMtpActivations,
    }

    /// Per-session MTP mutable state (cache + activations only; weights live on the engine).
    pub(crate) struct MetalMtpSession {
        pub(crate) cache: MetalMtpCache,
        pub(crate) activations: MetalMtpActivations,
    }

    /// Per-speculation-round GDN state checkpoint pool.
    ///
    /// Stores up to `max_tokens + 1` snapshots (slot 0 = base, slot k = after k tokens).
    struct MetalGdnCheckpointPool {
        max_tokens: usize,
        conv_slots: Vec<Vec<Buffer>>, // [slot][linear_layer]
        s_slots: Vec<Vec<Buffer>>,    // [slot][linear_layer]
        active_base_seq_len: Option<usize>,
        mtp_base_seq_len: Option<usize>,
        /// Batch-verifier repair: (token_id, position) to replay on rejection for GDN correctness.
        batch_repair_token: Option<(u32, usize)>,
    }

    impl MetalGdnCheckpointPool {
        fn new(
            device: &Device,
            max_tokens: usize,
            conv_bufs: &[Buffer],
            s_matrices: &[Buffer],
        ) -> Self {
            let num_layers = conv_bufs.len();
            let slots = max_tokens + 1;
            let conv_slots = (0..slots)
                .map(|slot| {
                    (0..num_layers)
                        .map(|i| {
                            let buf = device.new_buffer(
                                conv_bufs[i].length(),
                                MTLResourceOptions::StorageModeShared,
                            );
                            buf.set_label(&format!("gdn_ckpt_conv_s{slot}_l{i}"));
                            buf
                        })
                        .collect()
                })
                .collect();
            let s_slots = (0..slots)
                .map(|slot| {
                    (0..num_layers)
                        .map(|i| {
                            let buf = device.new_buffer(
                                s_matrices[i].length(),
                                MTLResourceOptions::StorageModeShared,
                            );
                            buf.set_label(&format!("gdn_ckpt_s_s{slot}_l{i}"));
                            buf
                        })
                        .collect()
                })
                .collect();
            Self {
                max_tokens,
                conv_slots,
                s_slots,
                active_base_seq_len: None,
                mtp_base_seq_len: None,
                batch_repair_token: None,
            }
        }
    }

    struct MetalVerifyOutput {
        logits: Vec<Vec<f32>>,  // per-token full-vocab logits
        final_hidden: Vec<f32>, // pre-final hidden of last verified row
    }

    struct MetalStepOutput {
        logits: Vec<f32>,
        pre_final_hidden: Vec<f32>,
    }

    struct MetalMtpForwardOutput {
        token_id: u32,
        logits: Vec<f32>,
    }

    /// Decoding metrics emitted when LATTICE_MTP=1.
    #[derive(Default)]
    struct MetalMtpDecodeMetrics {
        rounds: usize,
        mtp_forwards: usize,
        verify_calls: usize,
        accepted_extra_tokens: usize,
        fallback_tokens: usize,
        mtp_ms: f64,
        verify_ms: f64,
        rollback_ms: f64,
    }

    /// Reusable activation buffers on GPU (pre-allocated for single-token decode).
    struct MetalQwen35Activations {
        hidden: Buffer,      // [hidden_size]
        residual: Buffer,    // [hidden_size]
        attn_out: Buffer,    // [max(q_dim, hidden_size)] for attention output
        q: Buffer,           // [2 * q_dim] for Q+gate interleaved from q_proj
        q_separated: Buffer, // [q_dim] Q after scatter
        gate_z: Buffer,      // [q_dim] gate after scatter
        k: Buffer,           // [kv_dim]
        v: Buffer,           // [kv_dim]
        gate: Buffer,        // [intermediate_size]
        up: Buffer,          // [intermediate_size]
        ffn_out: Buffer,     // [hidden_size]
        // GDN projection buffers
        gdn_qkv: Buffer,         // [qkv_dim]  — used by batch/prefill and CPU paths
        gdn_z: Buffer,           // [output_dim] — used by batch/prefill and CPU paths
        gdn_qkvz: Buffer,        // [qkv_dim+output_dim] — decode-only fused projection scratch
        gdn_key_scratch: Buffer, // [num_key_heads * (2*key_dim+3)] — H3 precomputed key-head values
        gdn_raw_out: Buffer,     // [output_dim] — H1 raw recurrence output before norm+SiLU
        logits: Buffer,          // [vocab_size]
        // Top-k compact readback buffers (zero-initialized; sized for MAX_TOP_K=256).
        topk_scratch_a: Buffer, // [first_pass_groups * MAX_TOP_K] TopKCandidate pairs
        topk_scratch_b: Buffer, // same — ping-pong merge target
        // Flash decode partitioned path (H3): [max_partitions * num_q_heads * (head_dim+2)] f32
        attn_partials: Buffer,
        // MTP: pre-final hidden state (before final RMSNorm) for the last processed token.
        pre_final_hidden: Buffer, // [hidden_size]
        // MTP: logits for all K verified tokens (K <= MTP_VERIFY_MAX_TOKENS).
        verify_logits: Buffer, // [MTP_VERIFY_MAX_TOKENS * vocab_size]
    }

    /// KV cache for full-attention layers stored in Metal unified memory.
    struct MetalKvCache {
        /// k_cache[full_layer_idx]: flat [max_cache_len * kv_dim]
        k_bufs: Vec<Buffer>,
        /// v_cache[full_layer_idx]: flat [max_cache_len * kv_dim]
        v_bufs: Vec<Buffer>,
        /// Current number of tokens cached.
        seq_len: usize,
        kv_dim: usize,
        max_cache_len: usize,
    }

    /// Compiled MSL pipeline state objects.
    struct MetalQwen35Pipelines {
        gemv_decode: ComputePipelineState,
        gemv_q8: ComputePipelineState,
        rms_norm: ComputePipelineState,
        partial_rope: ComputePipelineState,
        per_head_rms_norm: ComputePipelineState,
        decode_attention: ComputePipelineState,
        sigmoid_gate: ComputePipelineState,
        scatter_q_gate: ComputePipelineState,
        silu_mul: ComputePipelineState,
        copy: ComputePipelineState,
        copy_offset: ComputePipelineState,
        add: ComputePipelineState,
        gemv_q4: ComputePipelineState,
        gemm_q4: ComputePipelineState,
        gemm_q4_tiled: Option<ComputePipelineState>,
        conv1d_silu: ComputePipelineState,
        gdn_recurrence: ComputePipelineState,
        fused_residual_add_norm: ComputePipelineState,
        copy_and_rms_norm: ComputePipelineState,
        add_and_copy: ComputePipelineState,
        gemm_q8: ComputePipelineState,
        topk_first_pass: ComputePipelineState,
        topk_merge_pass: ComputePipelineState,
        argmax_first: ComputePipelineState,
        argmax_merge: ComputePipelineState,
        topk_fast_first: ComputePipelineState,
        // Hierarchical k=50 SIMD-group tournament kernels (R2)
        topk_select50_first: ComputePipelineState,
        topk_select50_merge: ComputePipelineState,
        // Flash decode partitioned kernels (H3)
        decode_attn_partial: ComputePipelineState,
        decode_attn_reduce: ComputePipelineState,
        // H2+H4: Qwen3.6-27B specialized GDN kernel (optional, falls back to gdn_recurrence)
        gdn_recurrence_q36: Option<ComputePipelineState>,
        // H1+H3: Three-kernel sharded path (optional; used when all three are present)
        gdn_precompute_keys: Option<ComputePipelineState>,
        gdn_recurrence_sharded: Option<ComputePipelineState>,
        gdn_norm_silu: Option<ComputePipelineState>,
        // ADR-045: LoRA GEMV kernels (always compiled, zero cost when unused)
        lora_gemv_a: ComputePipelineState,
        lora_gemv_b_accum: ComputePipelineState,
    }

    // -----------------------------------------------------------------------
    // GPU Top-K routing
    // -----------------------------------------------------------------------

    /// Which GPU kernel path handles top-k for the current generate call.
    ///
    /// Computed once by `choose_gpu_topk_route`; stored in `compact_route` so
    /// the dispatch path doesn't re-evaluate env vars per token.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum GpuTopkRoute {
        /// CPU fallback — no GPU top-k dispatch; `compact_topk` stays 0.
        CpuFallback,
        /// k=1 two-stage simdgroup argmax — reserved for a future faster kernel.
        /// R1 bench (2026-04-26, M2 Max): CPU NEON 89 µs vs GPU 240 µs (2.70×); dead path.
        #[allow(dead_code)]
        Argmax,
        /// k=2..=64 simdgroup selection — reserved for a future non-bitonic kernel.
        /// Bench prototype failed gate by 2.4-2.8×; not reachable from production.
        #[allow(dead_code)]
        Select64,
        /// k=65..=256 experimental — reserved for a future non-bitonic kernel.
        #[allow(dead_code)]
        Select256Experimental,
        /// k=50 hierarchical SIMD-group tournament (two-stage, no bitonic sort).
        /// Requires LATTICE_COMPACT_TOPK and LATTICE_COMPACT_TOPK_SELECT env vars.
        HierarchicalK50,
    }

    /// Decide which GPU top-k route to take for a given `top_k` and env flags.
    ///
    /// `compact_env`: `LATTICE_COMPACT_TOPK` is set.
    /// `selection_env`: `LATTICE_COMPACT_TOPK_SELECT` is set (unlocks k>1 routes).
    fn choose_gpu_topk_route(top_k: usize, compact_env: bool, selection_env: bool) -> GpuTopkRoute {
        // R1 benchmark (2026-04-26, M2 Max, vocab=248,320):
        //   CPU NEON argmax k=1 = 89 µs; GPU argmax k=1 = 240 µs (2.70×, t=137, p<<0.0001).
        //   GPU k>1 (bitonic/select64) = 811–940 µs (2.5-3.0× vs CPU 314–366 µs).
        //   Hierarchical k=50 tournament under evaluation (R2, gated by both env flags).
        if compact_env && selection_env && top_k == 50 {
            return GpuTopkRoute::HierarchicalK50;
        }
        GpuTopkRoute::CpuFallback
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum QuantFormat {
        Q8_0,
        Q4_0,
    }

    impl Default for QuantFormat {
        fn default() -> Self {
            QuantFormat::Q8_0
        }
    }

    impl QuantFormat {
        fn from_env() -> Option<Self> {
            match std::env::var("LATTICE_QUANT_FORMAT")
                .ok()?
                .to_uppercase()
                .as_str()
            {
                "Q8_0" | "Q8" => Some(QuantFormat::Q8_0),
                "Q4_0" | "Q4" => Some(QuantFormat::Q4_0),
                _ => None,
            }
        }

        fn from_model_size(cfg: &Qwen35Config) -> Self {
            let h = cfg.hidden_size;
            let i = cfg.intermediate_size;
            let n = cfg.num_hidden_layers;
            let v = cfg.vocab_size;
            let est_params = n * (4 * h * h + 3 * h * i) + v * h;
            if est_params > 2_000_000_000 {
                QuantFormat::Q4_0
            } else {
                QuantFormat::Q8_0
            }
        }

        fn resolve(cfg: &Qwen35Config) -> Self {
            if let Some(env_fmt) = Self::from_env() {
                return env_fmt;
            }
            Self::from_model_size(cfg)
        }
    }

    /// Shared immutable model state for Qwen3.5/3.6 Metal GPU inference.
    ///
    /// Holds all state that does not change between requests: compiled Metal pipelines,
    /// layer weights, embedding tables, RoPE cosine/sine tables, and model config.
    /// Designed to be wrapped in `Arc<MetalQwen35Engine>` for concurrent multi-session
    /// serving — one engine shared across many `InferenceSession` instances.
    ///
    /// **Deferred**: The public concurrent API
    /// (`Engine::forward_step(&self, session: &mut InferenceSession, ...)`) is deferred
    /// to a follow-up PR; this PR delivers the struct-level separation as prerequisite
    /// infrastructure. For now, use `MetalQwen35State` as the backward-compatible
    /// single-session entry point.
    pub struct MetalQwen35Engine {
        #[allow(dead_code)]
        // TODO(#1958): Metal roadmap — device handle kept alive for GPU lifetime management
        pub(crate) device: Device,
        pub(crate) queue: CommandQueue,
        pub(crate) pipelines: MetalQwen35Pipelines,
        pub(crate) layer_weights: Vec<(MetalLayerAttnWeights, MetalCommonLayerWeights)>,
        pub(crate) embed_tokens: Buffer, // [vocab_size, hidden] — f16
        pub(crate) embed_tokens_q8: Q4WeightBuf, // [vocab_size, hidden] — Q8_0/Q4 for logits GEMV
        pub(crate) final_norm: Buffer,   // [hidden] — f32
        pub(crate) rope_cos: Buffer,     // [max_pos * half_rope_dim] — f32
        pub(crate) rope_sin: Buffer,     // [max_pos * half_rope_dim] — f32
        pub(crate) config: Qwen35Config,
        pub(crate) quant_format: QuantFormat,
        // Immutable MTP weights only; cache/activations live on InferenceSession.
        pub(crate) mtp_weights: Option<MetalMtpWeights>,
    }

    /// Per-request mutable inference state.
    ///
    /// Holds all state that evolves during a single inference request: KV cache,
    /// GDN recurrent state, activation scratch buffers, MTP session cache, and
    /// sampling state. One instance is required per concurrent inference request.
    /// Created via `MetalQwen35Engine::new_session(max_cache_len)`.
    pub struct InferenceSession {
        pub(crate) activations: MetalQwen35Activations,
        pub(crate) gdn_states: Vec<GatedDeltaNetState>,
        #[allow(dead_code)]
        // TODO(#1958): Metal roadmap — GDN GPU scratch buffer, wired in future GPU-GDN pass
        pub(crate) gdn_scratch: GatedDeltaNetFusedScratch,
        pub(crate) gdn_gpu_conv_bufs: Vec<Buffer>, // [num_linear_layers] each [qkv_dim * buf_len]
        pub(crate) gdn_gpu_s_matrices: Vec<Buffer>, // [num_linear_layers] each [num_heads * vd * kd]
        pub(crate) gdn_gpu_conv_out: Buffer,        // [qkv_dim] temporary conv1d output
        pub(crate) kv_cache: MetalKvCache,
        /// Maximum batch prefill length (activation buffers sized for this).
        pub(crate) max_prefill: usize,
        /// When >0, forward_step dispatches GPU top-k; set to gen_cfg.top_k before generate loops.
        pub(crate) compact_topk: usize,
        /// Routing decision for the current generate call.
        pub(crate) compact_route: GpuTopkRoute,
        /// Populated by forward_step / forward_prefill when compact_topk > 0.
        pub(crate) compact_result: Vec<crate::sampling::Candidate>,
        // MTP per-session cache and activations (None if model has no MTP).
        pub(crate) mtp: Option<MetalMtpSession>,
        pub(crate) gdn_checkpoints: Option<MetalGdnCheckpointPool>,
        pub(crate) last_pre_final_hidden: Vec<f32>,
        /// Authoritative decode cursor; kept in sync with kv_cache.seq_len.
        pub(crate) position: usize,
    }

    impl InferenceSession {
        /// Set both the logical position and the KV cache sequence length atomically.
        pub(crate) fn set_position(&mut self, position: usize) {
            self.position = position;
            self.kv_cache.seq_len = position;
        }
    }

    // ---------------------------------------------------------------------------
    // LoRA adapter state (ADR-045 step 3)
    // ---------------------------------------------------------------------------

    /// Metal buffers for a single LoRA-adapted projection.
    struct MetalLoraProjection {
        a_buf: Buffer,
        b_buf: Buffer,
        rank: u32,
        d_in: u32,
        d_out: u32,
    }

    /// Loaded LoRA adapter state on Metal GPU.
    pub(crate) struct MetalLoraAdapter {
        /// Nested lookup: layer_idx → module_name → projection. No allocation on lookup.
        projections: Vec<std::collections::HashMap<String, MetalLoraProjection>>,
        scale: f32,
        intermediate: Buffer,
        max_rank: u32,
    }

    impl MetalLoraAdapter {
        fn get_projection(&self, layer_idx: usize, module: &str) -> Option<&MetalLoraProjection> {
            self.projections.get(layer_idx)?.get(module)
        }
    }

    /// Input data for a single LoRA layer to be loaded onto GPU.
    pub struct LoraLayerData {
        /// Transformer layer index (0-based).
        pub layer_idx: usize,
        /// Projection module name (e.g., `"q_proj"`, `"o_proj"`).
        pub module: String,
        /// A matrix, row-major `(rank × d_in)`.
        pub a: Vec<f32>,
        /// B matrix, row-major `(d_out × rank)`.
        pub b: Vec<f32>,
        /// LoRA rank.
        pub rank: usize,
        /// Input dimension.
        pub d_in: usize,
        /// Output dimension.
        pub d_out: usize,
    }

    /// **Unstable**: Metal GPU forward pass for Qwen3.5/3.6; kernel set and weight layout evolving.
    ///
    /// Backward-compatible convenience wrapper bundling one `MetalQwen35Engine` (immutable
    /// model state) with one `InferenceSession` (mutable per-request state). Existing callers
    /// continue using this type unchanged. For concurrent multi-session serving, construct the
    /// engine and sessions separately once the public concurrent API ships.
    pub struct MetalQwen35State {
        pub(crate) engine: MetalQwen35Engine,
        pub(crate) session: InferenceSession,
        pub(crate) lora: Option<MetalLoraAdapter>,
    }

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    // Flash decode shape invariants: HEAD_DIM=256, GQA group ∈ [1,8].
    const METAL_FLASH_HEAD_DIM: usize = 256;
    const METAL_FLASH_MAX_GQA_GROUP: usize = 8;
    const METAL_FLASH_PARTITION_TOKENS: usize = 1024;

    fn validate_flash_decode_shape(
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Result<(), String> {
        if head_dim != METAL_FLASH_HEAD_DIM {
            return Err(format!(
                "Metal FlashAttention decode supports head_dim={} only; got {}",
                METAL_FLASH_HEAD_DIM, head_dim
            ));
        }
        if num_q_heads == 0 || num_kv_heads == 0 {
            return Err(format!(
                "Metal FlashAttention decode requires nonzero heads; got num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}"
            ));
        }
        if num_q_heads % num_kv_heads != 0 {
            return Err(format!(
                "Metal FlashAttention decode requires num_q_heads divisible by num_kv_heads; got {num_q_heads}/{num_kv_heads}"
            ));
        }
        let group_size = num_q_heads / num_kv_heads;
        if group_size == 0 || group_size > METAL_FLASH_MAX_GQA_GROUP {
            return Err(format!(
                "Metal FlashAttention decode supports GQA group size 1..={}; got {}",
                METAL_FLASH_MAX_GQA_GROUP, group_size
            ));
        }
        let expected_q_dim = num_q_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Metal FlashAttention q_dim overflow".to_string())?;
        let expected_kv_dim = num_kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| "Metal FlashAttention kv_dim overflow".to_string())?;
        if q_dim != expected_q_dim {
            return Err(format!(
                "Metal FlashAttention q_dim mismatch: got {q_dim}, expected {expected_q_dim}"
            ));
        }
        if kv_dim != expected_kv_dim {
            return Err(format!(
                "Metal FlashAttention kv_dim mismatch: got {kv_dim}, expected {expected_kv_dim}"
            ));
        }
        Ok(())
    }

    /// Create a Metal buffer holding f32 data in shared (unified) memory.
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

    /// Create a Metal buffer holding f16 (half-precision) data converted from f32.
    ///
    /// Converts each f32 element to IEEE-754 half precision at load time. The resulting
    /// buffer is half the size of the f32 equivalent and is consumed by `matmul_bt_half`.
    fn make_buffer_f16(device: &Device, data: &[f32], label: &str) -> Buffer {
        let f16_data: Vec<u16> = data.iter().map(|&x| f32_to_f16(x)).collect();
        let byte_len = (f16_data.len() * std::mem::size_of::<u16>()) as u64;
        let buf = device.new_buffer_with_data(
            f16_data.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        buf.set_label(label);
        buf
    }

    /// Q8_0 block: f16 scale (2 bytes) + 32 int8 values = 34 bytes.
    const QK8_0: usize = 32;
    const Q8_0_BLOCK_SIZE: usize = 2 + QK8_0;
    const QK4_0: usize = 32;
    const Q4_0_BLOCK_SIZE: usize = 18;

    /// Quantize a row of f32 weights to Q8_0 blocks.
    fn quantize_row_q8_0(src: &[f32]) -> Vec<u8> {
        assert_eq!(src.len() % QK8_0, 0);
        let nblocks = src.len() / QK8_0;
        let mut packed = Vec::with_capacity(nblocks * Q8_0_BLOCK_SIZE);
        for b in 0..nblocks {
            let block = &src[b * QK8_0..(b + 1) * QK8_0];
            let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            packed.extend_from_slice(&f32_to_f16(d).to_ne_bytes());
            for &v in block {
                packed.push((v * id).round().clamp(-127.0, 127.0) as i8 as u8);
            }
        }
        packed
    }

    /// Create a Metal buffer with Q8_0 quantized weights from f32 data.
    fn make_buffer_q8_0(device: &Device, data: &[f32], label: &str) -> Buffer {
        assert_eq!(data.len() % QK8_0, 0);
        let packed = quantize_row_q8_0(data);
        let byte_len = packed.len() as u64;
        let buf = device.new_buffer_with_data(
            packed.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        buf.set_label(label);
        buf
    }

    fn make_buffer_q4_0(device: &Device, data: &[f32], label: &str) -> Buffer {
        assert_eq!(
            data.len() % QK4_0,
            0,
            "make_buffer_q4_0: data length {} not divisible by {QK4_0}",
            data.len()
        );
        let packed = quantize_row_q4_0(data);
        let byte_len = packed.len() as u64;
        let buf = device.new_buffer_with_data(
            packed.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        buf.set_label(label);
        buf
    }

    fn make_zero_buffer(device: &Device, num_floats: usize, label: &str) -> Buffer {
        let byte_len = (num_floats * std::mem::size_of::<f32>()) as u64;
        let buf = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
        buf.set_label(label);
        buf
    }

    /// Allocate a zero-initialised byte buffer (not float-sized).
    /// Used for top-k candidate scratch buffers whose element size is 8 bytes.
    fn make_zero_byte_buffer(device: &Device, byte_len: usize, label: &str) -> Buffer {
        let buf = device.new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
        buf.set_label(label);
        buf
    }

    /// GemmParams layout matching the MSL GemmParams struct (6 × u32 = 24 bytes).
    #[repr(C)]
    struct GemmParams {
        m: u32,
        n: u32,
        k: u32,
        lda: u32,
        ldb: u32,
        ldc: u32,
    }

    fn div_ceil(a: u64, b: u64) -> u64 {
        a.div_ceil(b)
    }

    /// Convert f32 to IEEE-754 half-precision (f16) stored as u16 bits.
    ///
    /// Handles signed zero, subnormals, infinities, and NaN. Uses round-to-nearest-even
    /// for the mantissa truncation to minimize systematic bias.
    #[inline]
    fn f32_to_f16(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let frac = bits & 0x007f_ffff;

        // Inf or NaN
        if exp == 0xff {
            if frac == 0 {
                return sign | 0x7c00; // infinity
            }
            // NaN: preserve some payload, ensure it stays NaN
            let mut payload = ((frac >> 13) as u16) & 0x03ff;
            if payload == 0 {
                payload = 1; // quiet NaN needs nonzero mantissa
            }
            payload |= 0x0200; // set quiet bit
            return sign | 0x7c00 | payload;
        }

        // Zero or f32 subnormal (becomes f16 zero)
        if exp == 0 {
            return sign;
        }

        let exp32 = exp - 127; // unbiased exponent

        // Overflow to infinity
        if exp32 > 15 {
            return sign | 0x7c00;
        }

        // Normal f16 range
        if exp32 >= -14 {
            let mut exp16 = (exp32 + 15) as u16;
            let mut frac16 = round_shift_right_even(frac, 13) as u16;

            // Mantissa overflow -> carry into exponent
            if frac16 == 0x0400 {
                frac16 = 0;
                exp16 += 1;
                if exp16 >= 0x1f {
                    return sign | 0x7c00; // overflow to infinity
                }
            }

            return sign | (exp16 << 10) | frac16;
        }

        // Subnormal f16 range
        let mant = frac | 0x0080_0000; // add implicit 1
        let shift = (-exp32 - 1) as u32;
        if shift >= 32 {
            return sign; // underflow to zero
        }

        let frac16 = round_shift_right_even(mant, shift) as u16;
        if frac16 == 0 {
            return sign;
        }
        // If rounding carried into exponent bit, it becomes smallest normal
        if frac16 == 0x0400 {
            return sign | 0x0400;
        }

        sign | frac16
    }

    /// Round-to-nearest-even right shift for mantissa truncation.
    #[inline]
    fn round_shift_right_even(value: u32, shift: u32) -> u32 {
        if shift == 0 {
            return value;
        }
        if shift >= 32 {
            return 0;
        }

        let base = value >> shift;
        let mask = (1u32 << shift) - 1;
        let remainder = value & mask;
        let half = 1u32 << (shift - 1);

        if remainder > half || (remainder == half && (base & 1) != 0) {
            base + 1
        } else {
            base
        }
    }

    /// Convert f16 bits (u16) back to f32.
    ///
    /// Used for CPU-side reads of f16 weight buffers when needed (e.g., small
    /// GDN projections dispatched on CPU).
    #[inline]
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 0x1) as u32;
        let exp = ((bits >> 10) & 0x1f) as u32;
        let frac = (bits & 0x03ff) as u32;

        let f32_bits = match (exp, frac) {
            // Zero
            (0, 0) => sign << 31,
            // Subnormal
            (0, _) => {
                let mut mant = frac;
                let mut e = -14i32;
                while (mant & 0x0400) == 0 {
                    mant <<= 1;
                    e -= 1;
                }
                mant &= 0x03ff;
                (sign << 31) | (((e + 127) as u32) << 23) | (mant << 13)
            }
            // Infinity
            (0x1f, 0) => (sign << 31) | 0x7f80_0000,
            // NaN
            (0x1f, _) => (sign << 31) | 0x7f80_0000 | (frac << 13),
            // Normal
            _ => (sign << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (frac << 13),
        };

        f32::from_bits(f32_bits)
    }

    /// Read f32 slice from a Metal shared-mode buffer.
    ///
    /// SAFETY: Buffer must be StorageModeShared with sufficient length and no
    /// concurrent GPU writes (command buffer completed or not yet submitted).
    unsafe fn read_buffer(buf: &Buffer, len: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        let mut out = vec![0.0f32; len];
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), len);
        out
    }

    /// Read f32 slice from a Metal shared-mode buffer starting at a given element offset.
    ///
    /// SAFETY: Same requirements as `read_buffer`; `offset_f32 + len` must be within bounds.
    unsafe fn read_buffer_offset(buf: &Buffer, offset_f32: usize, len: usize) -> Vec<f32> {
        let ptr = (buf.contents() as *const f32).add(offset_f32);
        std::slice::from_raw_parts(ptr, len).to_vec()
    }

    /// Read f16 data from a Metal shared-mode buffer, converting to f32.
    ///
    /// Used for CPU-side reads of f16 weight buffers (GDN small projections).
    ///
    /// SAFETY: Buffer must be StorageModeShared with sufficient length and no
    /// concurrent GPU writes (command buffer completed or not yet submitted).
    unsafe fn read_buffer_f16(buf: &Buffer, len: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const u16;
        let mut out = vec![0.0f32; len];
        for i in 0..len {
            out[i] = f16_to_f32(*ptr.add(i));
        }
        out
    }

    /// Write f32 slice into a Metal shared-mode buffer.
    ///
    /// SAFETY: Buffer must be StorageModeShared with sufficient capacity and no
    /// concurrent GPU reads (command buffer completed or not yet submitted).
    unsafe fn write_buffer(buf: &Buffer, data: &[f32]) {
        let dst = buf.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
    }

    /// Build interleaved partial RoPE cos/sin tables.
    /// For Qwen3.5: partial_rotary_factor=0.25, head_dim=256 => rope_dim=64, half=32.
    /// Tables: [max_pos, half_rope_dim] each.
    fn build_rope_interleaved(rope_dim: usize, max_pos: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
        let half = rope_dim / 2;
        let mut cos_data = Vec::with_capacity(max_pos * half);
        let mut sin_data = Vec::with_capacity(max_pos * half);

        for pos in 0..max_pos {
            for i in 0..half {
                // freq for interleaved pair i: theta^(-2i/rope_dim)
                let freq = 1.0 / theta.powf(2.0 * i as f64 / rope_dim as f64);
                let angle = pos as f64 * freq;
                cos_data.push(angle.cos() as f32);
                sin_data.push(angle.sin() as f32);
            }
        }

        (cos_data, sin_data)
    }

    impl MetalKvCache {
        fn new(
            device: &Device,
            num_full_layers: usize,
            kv_dim: usize,
            max_cache_len: usize,
        ) -> Self {
            let mut k_bufs = Vec::with_capacity(num_full_layers);
            let mut v_bufs = Vec::with_capacity(num_full_layers);
            for i in 0..num_full_layers {
                k_bufs.push(make_zero_buffer(
                    device,
                    max_cache_len * kv_dim,
                    &format!("kv_k_{i}"),
                ));
                v_bufs.push(make_zero_buffer(
                    device,
                    max_cache_len * kv_dim,
                    &format!("kv_v_{i}"),
                ));
            }
            Self {
                k_bufs,
                v_bufs,
                seq_len: 0,
                kv_dim,
                max_cache_len,
            }
        }

        /// Append a K/V pair into the cache for a given full-attention layer.
        /// Writes directly into the shared Metal buffer via CPU pointer.
        ///
        /// Returns `Err` if the cache is full; callers must handle this before calling
        /// to avoid a buffer overflow in release builds.
        fn append_kv(
            &mut self,
            full_layer_idx: usize,
            k_vec: &[f32],
            v_vec: &[f32],
        ) -> Result<(), String> {
            if self.seq_len >= self.max_cache_len {
                return Err(format!(
                    "KV cache full: seq_len {} >= max_cache_len {}",
                    self.seq_len, self.max_cache_len
                ));
            }
            let offset = self.seq_len * self.kv_dim;
            // SAFETY: seq_len < max_cache_len (checked above); buffers are
            // StorageModeShared and we have exclusive access (no GPU command
            // buffer is in flight during this call); offset + k_vec.len() ==
            // (seq_len + 1) * kv_dim which is ≤ max_cache_len * kv_dim (the
            // allocated buffer size).
            unsafe {
                let k_ptr = self.k_bufs[full_layer_idx].contents() as *mut f32;
                std::ptr::copy_nonoverlapping(k_vec.as_ptr(), k_ptr.add(offset), k_vec.len());
                let v_ptr = self.v_bufs[full_layer_idx].contents() as *mut f32;
                std::ptr::copy_nonoverlapping(v_vec.as_ptr(), v_ptr.add(offset), v_vec.len());
            }
            Ok(())
        }

        fn reset(&mut self) {
            self.seq_len = 0;
            // No need to zero the buffers — seq_len controls valid range.
        }
    }

    // -----------------------------------------------------------------------
    // Layer importance scoring — calibration harness
    // -----------------------------------------------------------------------

    /// Per-layer hidden-state snapshot for calibration scoring.
    struct LayerTrace {
        layer_idx: usize,
        input: Vec<f32>,
        output: Vec<f32>,
    }

    /// Layer importance score derived from hidden-state cosine similarity.
    #[derive(Debug, Clone)]
    pub struct LayerImportanceScore {
        pub layer_idx: usize,
        pub layer_type: crate::model::qwen35_config::LayerType,
        /// Mean cosine similarity of input vs output hidden state across calibration prompts.
        /// High cosine → small change → safer prune candidate.
        pub mean_cosine: f32,
        /// Importance proxy: `1.0 - mean_cosine`.
        pub importance: f32,
    }

    /// Pruning plan returned by [`MetalQwen35State::score_layer_importance`].
    #[derive(Debug, Clone)]
    pub struct LayerPruningPlan {
        /// Layers sorted by `mean_cosine` descending (highest cosine first = best prune candidate).
        pub scores: Vec<LayerImportanceScore>,
        /// Recommended per-layer mask (`false` = prune, `true` = keep).
        pub recommended_mask: Vec<bool>,
        /// Set when `prune_layers` exceeds 20 % of the total layer count.
        pub warning: Option<String>,
    }

    /// Numerically stable cosine similarity using f64 accumulators.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut aa = 0.0f64;
        let mut bb = 0.0f64;
        for (&x, &y) in a.iter().zip(b) {
            let x = x as f64;
            let y = y as f64;
            dot += x * y;
            aa += x * x;
            bb += y * y;
        }
        if aa == 0.0 || bb == 0.0 {
            return 0.0;
        }
        (dot / (aa.sqrt() * bb.sqrt())) as f32
    }

    impl MetalQwen35Engine {
        /// Load immutable model resources from CPU weights. Does NOT allocate any session state.
        ///
        /// `max_cache_len` is NOT accepted here; pass it to `new_session` instead.
        pub fn new(weights: &ModelWeights, cfg: &Qwen35Config) -> Result<Self, String> {
            let device =
                Device::system_default().ok_or_else(|| "No Metal device found".to_string())?;

            tracing::info!(
                name = device.name(),
                "Metal GPU initialized for Qwen3.5-2B forward pass (f16 weights)"
            );

            let queue = device.new_command_queue();

            let quant_format = QuantFormat::resolve(cfg);
            tracing::info!(
                ?quant_format,
                "Weight quantization format for Metal forward pass"
            );

            validate_flash_decode_shape(
                cfg.head_dim,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.num_attention_heads * cfg.head_dim,
                cfg.num_key_value_heads * cfg.head_dim,
            )?;

            // Compile shaders
            let opts = CompileOptions::new();
            let library = device
                .new_library_with_source(MSL_SOURCE, &opts)
                .map_err(|e| format!("Metal shader compilation failed: {e}"))?;

            let make_pipeline = |name: &str| -> Result<ComputePipelineState, String> {
                let func = library
                    .get_function(name, None)
                    .map_err(|e| format!("function '{name}' not found: {e}"))?;
                device
                    .new_compute_pipeline_state_with_function(&func)
                    .map_err(|e| format!("pipeline for '{name}' failed: {e}"))
            };

            let make_optional_gemm_q4_tiled = || -> Option<ComputePipelineState> {
                if !device.supports_family(MTLGPUFamily::Apple9) {
                    return None;
                }
                let tiled_opts = CompileOptions::new();
                tiled_opts.set_language_version(MTLLanguageVersion::V3_0);
                let lib = device
                    .new_library_with_source(MSL_Q4_TILED_SOURCE, &tiled_opts)
                    .ok()?;
                let func = lib.get_function("gemm_q4_tiled", None).ok()?;
                device.new_compute_pipeline_state_with_function(&func).ok()
            };

            let pipelines = MetalQwen35Pipelines {
                gemv_decode: make_pipeline("gemv_decode_m1")?,
                gemv_q8: make_pipeline("gemv_q8_decode")?,
                gemv_q4: make_pipeline("gemv_q4_decode")?,
                gemm_q4: make_pipeline("gemm_q4")?,
                gemm_q4_tiled: make_optional_gemm_q4_tiled(),
                rms_norm: make_pipeline("rms_norm_qwen35")?,
                partial_rope: make_pipeline("partial_rope_interleaved")?,
                per_head_rms_norm: make_pipeline("per_head_rms_norm")?,
                decode_attention: make_pipeline("decode_attention")?,
                sigmoid_gate: make_pipeline("sigmoid_gate")?,
                scatter_q_gate: make_pipeline("scatter_q_gate")?,
                silu_mul: make_pipeline("silu_mul")?,
                copy: make_pipeline("copy_buf")?,
                copy_offset: make_pipeline("copy_buf_offset")?,
                add: make_pipeline("add_buf")?,
                conv1d_silu: make_pipeline("conv1d_depthwise_silu")?,
                gdn_recurrence: make_pipeline("gdn_recurrence_fused")?,
                gdn_recurrence_q36: library
                    .get_function("gdn_recurrence_fused_q36", None)
                    .ok()
                    .and_then(|f| device.new_compute_pipeline_state_with_function(&f).ok()),
                gdn_precompute_keys: library
                    .get_function("gdn_precompute_keys", None)
                    .ok()
                    .and_then(|f| device.new_compute_pipeline_state_with_function(&f).ok()),
                gdn_recurrence_sharded: library
                    .get_function("gdn_recurrence_sharded", None)
                    .ok()
                    .and_then(|f| device.new_compute_pipeline_state_with_function(&f).ok()),
                gdn_norm_silu: library
                    .get_function("gdn_norm_silu", None)
                    .ok()
                    .and_then(|f| device.new_compute_pipeline_state_with_function(&f).ok()),
                fused_residual_add_norm: make_pipeline("fused_residual_add_norm")?,
                copy_and_rms_norm: make_pipeline("copy_and_rms_norm")?,
                add_and_copy: make_pipeline("add_and_copy")?,
                gemm_q8: make_pipeline("gemm_q8")?,
                topk_first_pass: make_pipeline("logits_topk_first_pass")?,
                topk_merge_pass: make_pipeline("logits_topk_merge_pass")?,
                argmax_first: make_pipeline("logits_argmax_first")?,
                argmax_merge: make_pipeline("logits_argmax_merge")?,
                topk_fast_first: make_pipeline("logits_topk_fast_first")?,
                topk_select50_first: make_pipeline("logits_topk_select50_first")?,
                topk_select50_merge: make_pipeline("logits_topk_select50_merge")?,
                decode_attn_partial: make_pipeline("decode_attention_flash_partial")?,
                decode_attn_reduce: make_pipeline("decode_attention_flash_reduce")?,
                lora_gemv_a: make_pipeline("lora_gemv_a")?,
                lora_gemv_b_accum: make_pipeline("lora_gemv_b_accum")?,
            };

            // Upload per-layer weights
            let quant_tag = match quant_format {
                QuantFormat::Q8_0 => "q8",
                QuantFormat::Q4_0 => "q4",
            };
            let make_buffer_quant = |device: &Device, data: &[f32], label: &str| -> Buffer {
                match quant_format {
                    QuantFormat::Q8_0 => make_buffer_q8_0(device, data, label),
                    QuantFormat::Q4_0 => make_buffer_q4_0(device, data, label),
                }
            };

            let mut layer_weights = Vec::with_capacity(cfg.num_hidden_layers);
            for (i, (attn_w, common_w)) in weights.layers.iter().enumerate() {
                let dense_ffn = match &common_w.ffn {
                    crate::model::qwen35::FeedForwardWeights::Dense(d) => d,
                    crate::model::qwen35::FeedForwardWeights::Moe(_) => {
                        return Err(format!(
                            "Metal forward pass does not support MoE FFN at layer {i}"
                        ));
                    }
                };
                let common = MetalCommonLayerWeights {
                    // Norms stay f32 (small, precision-sensitive)
                    input_layernorm: make_buffer(
                        &device,
                        &common_w.input_layernorm,
                        &format!("L{i}.in_norm"),
                    ),
                    post_attention_layernorm: make_buffer(
                        &device,
                        &common_w.post_attention_layernorm,
                        &format!("L{i}.post_norm"),
                    ),
                    // FFN projections to q8_0 (large, bandwidth-bound)
                    gate_proj: Q4WeightBuf::from_buffer(make_buffer_quant(
                        &device,
                        &dense_ffn.gate_proj,
                        &format!("L{i}.gate.{quant_tag}"),
                    )),
                    up_proj: Q4WeightBuf::from_buffer(make_buffer_quant(
                        &device,
                        &dense_ffn.up_proj,
                        &format!("L{i}.up.{quant_tag}"),
                    )),
                    down_proj: Q4WeightBuf::from_buffer(make_buffer_quant(
                        &device,
                        &dense_ffn.down_proj,
                        &format!("L{i}.down.{quant_tag}"),
                    )),
                };

                let attn = match attn_w {
                    AttentionWeights::Linear(gdn_w) => {
                        MetalLayerAttnWeights::Linear(MetalGdnLayerWeights {
                            // Large projections to f16
                            in_proj_qkv: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &gdn_w.in_proj_qkv,
                                &format!("L{i}.gdn.qkv.{quant_tag}"),
                            )),
                            in_proj_z: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &gdn_w.in_proj_z,
                                &format!("L{i}.gdn.z.{quant_tag}"),
                            )),
                            in_proj_qkvz: {
                                let mut combined = gdn_w.in_proj_qkv.to_vec();
                                combined.extend_from_slice(&gdn_w.in_proj_z);
                                Q4WeightBuf::from_buffer(make_buffer_quant(
                                    &device,
                                    &combined,
                                    &format!("L{i}.gdn.qkvz.{quant_tag}"),
                                ))
                            },
                            // in_proj_b/a: keep f16 — CPU reads these for GDN recurrence
                            in_proj_b: make_buffer_f16(
                                &device,
                                &gdn_w.in_proj_b,
                                &format!("L{i}.gdn.b.f16"),
                            ),
                            in_proj_a: make_buffer_f16(
                                &device,
                                &gdn_w.in_proj_a,
                                &format!("L{i}.gdn.a.f16"),
                            ),
                            // Small scalars stay f32 (precision-sensitive, CPU-read)
                            a_log: make_buffer(&device, &gdn_w.a_log, &format!("L{i}.gdn.a_log")),
                            dt_bias: make_buffer(
                                &device,
                                &gdn_w.dt_bias,
                                &format!("L{i}.gdn.dt_bias"),
                            ),
                            // Conv1d and norm: f32 (small, read by CPU directly)
                            conv1d_weight: make_buffer(
                                &device,
                                &gdn_w.conv1d_weight,
                                &format!("L{i}.gdn.conv1d"),
                            ),
                            norm_weight: make_buffer(
                                &device,
                                &gdn_w.norm_weight,
                                &format!("L{i}.gdn.norm"),
                            ),
                            // Output projection: f16 (large)
                            out_proj: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &gdn_w.out_proj,
                                &format!("L{i}.gdn.out.{quant_tag}"),
                            )),
                        })
                    }
                    AttentionWeights::Full(full_w) => {
                        MetalLayerAttnWeights::Full(MetalFullLayerWeights {
                            // Large projections to f16
                            q_proj: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &full_w.q_proj,
                                &format!("L{i}.full.q.{quant_tag}"),
                            )),
                            k_proj: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &full_w.k_proj,
                                &format!("L{i}.full.k.{quant_tag}"),
                            )),
                            v_proj: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &full_w.v_proj,
                                &format!("L{i}.full.v.{quant_tag}"),
                            )),
                            o_proj: Q4WeightBuf::from_buffer(make_buffer_quant(
                                &device,
                                &full_w.o_proj,
                                &format!("L{i}.full.o.{quant_tag}"),
                            )),
                            // Norm weights stay f32 (small, precision-sensitive)
                            q_norm: make_buffer(
                                &device,
                                &full_w.q_norm,
                                &format!("L{i}.full.q_norm"),
                            ),
                            k_norm: make_buffer(
                                &device,
                                &full_w.k_norm,
                                &format!("L{i}.full.k_norm"),
                            ),
                        })
                    }
                };

                layer_weights.push((attn, common));
            }

            // Embedding matrix: f16 (large, used for both embed and tied lm_head)
            // embed_tokens: keep f16 for embedding lookup (CPU reads it).
            // Add a Q8_0 version for the logits projection (GPU matmul).
            let embed_tokens = make_buffer_f16(&device, &weights.embed_tokens, "embed_tokens.f16");
            let embed_tokens_q8 = Q4WeightBuf::from_buffer(make_buffer_quant(
                &device,
                &weights.embed_tokens,
                &format!("embed_tokens.{quant_tag}"),
            ));
            // Final norm: f32 (small, precision-sensitive)
            let final_norm = make_buffer(&device, &weights.final_norm, "final_norm");

            // Build RoPE tables for partial interleaved rotation (f32).
            // Engine does not know the session cache length; size to max_position_embeddings.
            let rope_dim = cfg.rope_dim();
            let rope_max = cfg.max_position_embeddings;
            let (cos_data, sin_data) = build_rope_interleaved(rope_dim, rope_max, cfg.rope_theta);
            let rope_cos = make_buffer(&device, &cos_data, "rope_cos");
            let rope_sin = make_buffer(&device, &sin_data, "rope_sin");

            Ok(Self {
                device,
                queue,
                pipelines,
                layer_weights,
                embed_tokens,
                embed_tokens_q8,
                final_norm,
                rope_cos,
                rope_sin,
                config: cfg.clone(),
                quant_format,
                mtp_weights: None,
            })
        }

        /// Allocate per-session mutable state for a single concurrent request.
        pub fn new_session(&self, max_cache_len: usize) -> Result<InferenceSession, String> {
            if max_cache_len == 0 {
                return Err("new_session: max_cache_len must be > 0".into());
            }
            if max_cache_len > self.config.max_position_embeddings {
                return Err(format!(
                    "new_session: max_cache_len {max_cache_len} exceeds max_position_embeddings {}",
                    self.config.max_position_embeddings
                ));
            }
            let cfg = &self.config;
            let device = &self.device;
            let hidden = cfg.hidden_size;
            let q_dim = cfg.full_q_dim();
            let kv_dim = cfg.full_kv_dim();
            let qkv_dim = cfg.linear_qkv_dim();
            let output_dim = cfg.linear_output_dim();
            let inter = cfg.intermediate_size;
            let max_prefill = max_cache_len.min(512);
            let bp = max_prefill;

            let activations = MetalQwen35Activations {
                hidden: make_zero_buffer(device, bp * hidden, "act_hidden"),
                residual: make_zero_buffer(device, bp * hidden, "act_residual"),
                attn_out: make_zero_buffer(device, bp * q_dim.max(hidden), "act_attn_out"),
                q: make_zero_buffer(device, bp * 2 * q_dim, "act_q_interleaved"),
                q_separated: make_zero_buffer(device, bp * q_dim, "act_q"),
                gate_z: make_zero_buffer(device, bp * q_dim, "act_gate_z"),
                k: make_zero_buffer(device, bp * kv_dim, "act_k"),
                v: make_zero_buffer(device, bp * kv_dim, "act_v"),
                gate: make_zero_buffer(device, bp * inter, "act_gate"),
                up: make_zero_buffer(device, bp * inter, "act_up"),
                ffn_out: make_zero_buffer(device, bp * hidden, "act_ffn_out"),
                gdn_qkv: make_zero_buffer(device, bp * qkv_dim, "act_gdn_qkv"),
                gdn_z: make_zero_buffer(device, bp * output_dim, "act_gdn_z"),
                gdn_qkvz: make_zero_buffer(device, qkv_dim + output_dim, "act_gdn_qkvz"),
                gdn_key_scratch: make_zero_buffer(
                    device,
                    cfg.linear_num_key_heads * (2 * cfg.linear_key_head_dim + 3),
                    "act_gdn_key_scratch",
                ),
                gdn_raw_out: make_zero_buffer(device, output_dim, "act_gdn_raw_out"),
                logits: make_zero_buffer(device, cfg.vocab_size, "act_logits"),
                topk_scratch_a: {
                    let groups = cfg.vocab_size.div_ceil(1024);
                    make_zero_byte_buffer(device, groups * 256 * 8, "topk_scratch_a")
                },
                topk_scratch_b: {
                    let groups = cfg.vocab_size.div_ceil(1024);
                    make_zero_byte_buffer(device, groups * 256 * 8, "topk_scratch_b")
                },
                attn_partials: {
                    let max_partitions = max_cache_len.div_ceil(1024);
                    let stride = cfg.head_dim + 2;
                    make_zero_buffer(
                        device,
                        max_partitions * cfg.num_attention_heads * stride,
                        "attn_partials",
                    )
                },
                pre_final_hidden: make_zero_buffer(device, hidden, "act_pre_final_hidden"),
                verify_logits: make_zero_buffer(
                    device,
                    MTP_VERIFY_MAX_TOKENS * cfg.vocab_size,
                    "act_verify_logits",
                ),
            };

            let num_linear = cfg.num_linear_attention_layers();
            let num_full = cfg.num_full_attention_layers();
            let gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
                .map(|_| GatedDeltaNetState::new(cfg))
                .collect();
            let gdn_scratch = GatedDeltaNetFusedScratch::default();

            let num_value_heads = cfg.linear_num_value_heads();
            let key_dim = cfg.linear_key_head_dim;
            let value_dim = cfg.linear_value_head_dim;
            let qkv_dim2 = cfg.linear_qkv_dim();
            let buf_len = cfg.linear_conv_kernel_dim.saturating_sub(1);
            let gdn_gpu_conv_bufs: Vec<Buffer> = (0..num_linear)
                .map(|i| make_zero_buffer(device, qkv_dim2 * buf_len, &format!("gdn_conv_{i}")))
                .collect();
            let gdn_gpu_s_matrices: Vec<Buffer> = (0..num_linear)
                .map(|i| {
                    make_zero_buffer(
                        device,
                        num_value_heads * value_dim * key_dim,
                        &format!("gdn_s_{i}"),
                    )
                })
                .collect();
            let gdn_gpu_conv_out = make_zero_buffer(device, qkv_dim2, "gdn_conv_out");
            let kv_cache = MetalKvCache::new(device, num_full, kv_dim, max_cache_len);

            // Allocate MTP session state if engine has MTP weights loaded.
            let mtp = self.mtp_weights.as_ref().map(|_| {
                let q_dim_mtp = cfg.full_q_dim();
                MetalMtpSession {
                    cache: MetalMtpCache {
                        k_buf: make_zero_buffer(device, max_cache_len * kv_dim, "mtp.kv_k"),
                        v_buf: make_zero_buffer(device, max_cache_len * kv_dim, "mtp.kv_v"),
                        seq_len: 0,
                        max_cache_len,
                    },
                    activations: MetalMtpActivations {
                        fused: make_zero_buffer(device, 2 * hidden, "mtp.act_fused"),
                        hidden: make_zero_buffer(device, hidden, "mtp.act_hidden"),
                        residual: make_zero_buffer(device, hidden, "mtp.act_residual"),
                        q: make_zero_buffer(device, 2 * q_dim_mtp, "mtp.act_q"),
                        q_separated: make_zero_buffer(device, q_dim_mtp, "mtp.act_q_sep"),
                        gate_z: make_zero_buffer(device, q_dim_mtp, "mtp.act_gate_z"),
                        k: make_zero_buffer(device, kv_dim, "mtp.act_k"),
                        v: make_zero_buffer(device, kv_dim, "mtp.act_v"),
                        attn_out: make_zero_buffer(
                            device,
                            q_dim_mtp.max(hidden),
                            "mtp.act_attn_out",
                        ),
                        gate: make_zero_buffer(device, inter, "mtp.act_gate"),
                        up: make_zero_buffer(device, inter, "mtp.act_up"),
                        ffn_out: make_zero_buffer(device, hidden, "mtp.act_ffn_out"),
                        logits: make_zero_buffer(device, cfg.vocab_size, "mtp.act_logits"),
                    },
                }
            });

            let gdn_checkpoints = if mtp.is_some() {
                Some(MetalGdnCheckpointPool::new(
                    device,
                    MTP_VERIFY_MAX_TOKENS,
                    &gdn_gpu_conv_bufs,
                    &gdn_gpu_s_matrices,
                ))
            } else {
                None
            };

            Ok(InferenceSession {
                activations,
                gdn_states,
                gdn_scratch,
                gdn_gpu_conv_bufs,
                gdn_gpu_s_matrices,
                gdn_gpu_conv_out,
                kv_cache,
                max_prefill,
                compact_topk: 0,
                compact_route: GpuTopkRoute::CpuFallback,
                compact_result: Vec::new(),
                mtp,
                gdn_checkpoints,
                last_pre_final_hidden: vec![0.0f32; hidden],
                position: 0,
            })
        }
    }

    /// Per-phase timing accumulator for `forward_step_inner`.
    #[derive(Default)]
    struct StepProfile {
        embedding_us: u128,
        projection_us: u128,
        gdn_recurrence_us: u128,
        gqa_attention_us: u128,
        mlp_us: u128,
        final_us: u128,
    }

    impl MetalQwen35State {
        /// **Unstable**: create from CPU weights; weight layout and f16 conversion may change.
        pub fn new(
            weights: &ModelWeights,
            cfg: &Qwen35Config,
            max_cache_len: usize,
        ) -> Result<Self, String> {
            let engine = MetalQwen35Engine::new(weights, cfg)?;
            let session = engine.new_session(max_cache_len)?;
            Ok(Self {
                engine,
                session,
                lora: None,
            })
        }

        /// Load a LoRA adapter onto the Metal GPU for inference.
        ///
        /// Applies rotation corrections for QuaRot-converted bases when `quarot_seed`
        /// is provided, then uploads all A/B matrices to Metal shared-memory buffers.
        ///
        /// # Errors
        ///
        /// Returns an error if:
        /// - An adapter is already loaded (call `unload_lora_adapter` first)
        /// - Any module name is not in the rotation plan (when `quarot_seed` is set)
        /// - Matrix dimensions are inconsistent
        pub fn load_lora_adapter(
            &mut self,
            mut layers: Vec<LoraLayerData>,
            scale: f32,
            quarot_seed: Option<u64>,
        ) -> Result<(), crate::error::InferenceError> {
            use crate::error::InferenceError;
            use crate::quant::quarot::lora::{LoraLayerMut, rotate_adapter_for_quarot};
            use crate::quant::quarot::plan::RotationPlan;

            if self.lora.is_some() {
                return Err(InferenceError::Inference(
                    "LoRA adapter already loaded; call unload_lora_adapter first".into(),
                ));
            }

            let hidden_dim = self.engine.config.hidden_size;

            if let Some(seed) = quarot_seed {
                let plan = RotationPlan::qwen35_residual_stream_linear_layers();
                let lora_muts: Vec<LoraLayerMut<'_>> = layers
                    .iter_mut()
                    .map(|l| LoraLayerMut {
                        layer_idx: l.layer_idx,
                        module: &l.module,
                        a: &mut l.a,
                        b: &mut l.b,
                        rank: l.rank,
                        d_in: l.d_in,
                        d_out: l.d_out,
                    })
                    .collect();
                rotate_adapter_for_quarot(lora_muts, seed, hidden_dim, &plan)?;
            }

            let device = &self.engine.device;
            let num_layers = self.engine.config.num_hidden_layers;
            let mut max_rank = 0u32;
            let mut projections: Vec<std::collections::HashMap<String, MetalLoraProjection>> = (0
                ..num_layers)
                .map(|_| std::collections::HashMap::new())
                .collect();

            for layer in &layers {
                if layer.rank == 0 {
                    return Err(InferenceError::Inference(format!(
                        "load_lora_adapter: layer {} module '{}' has rank=0",
                        layer.layer_idx, layer.module
                    )));
                }
                if layer.d_in == 0 || layer.d_out == 0 {
                    return Err(InferenceError::Inference(format!(
                        "load_lora_adapter: layer {} module '{}' has zero dimension \
                         (d_in={}, d_out={})",
                        layer.layer_idx, layer.module, layer.d_in, layer.d_out
                    )));
                }
                if layer.layer_idx >= num_layers {
                    return Err(InferenceError::Inference(format!(
                        "load_lora_adapter: layer_idx {} >= num_hidden_layers {}",
                        layer.layer_idx, num_layers
                    )));
                }
                let expected_a = layer.rank.checked_mul(layer.d_in).ok_or_else(|| {
                    InferenceError::Inference(format!(
                        "load_lora_adapter: layer {} module '{}' rank*d_in overflow",
                        layer.layer_idx, layer.module
                    ))
                })?;
                let expected_b = layer.d_out.checked_mul(layer.rank).ok_or_else(|| {
                    InferenceError::Inference(format!(
                        "load_lora_adapter: layer {} module '{}' d_out*rank overflow",
                        layer.layer_idx, layer.module
                    ))
                })?;
                if layer.a.len() != expected_a {
                    return Err(InferenceError::Inference(format!(
                        "load_lora_adapter: layer {} module '{}' A length {} != rank*d_in {}",
                        layer.layer_idx,
                        layer.module,
                        layer.a.len(),
                        expected_a
                    )));
                }
                if layer.b.len() != expected_b {
                    return Err(InferenceError::Inference(format!(
                        "load_lora_adapter: layer {} module '{}' B length {} != d_out*rank {}",
                        layer.layer_idx,
                        layer.module,
                        layer.b.len(),
                        expected_b
                    )));
                }
                let rank = u32::try_from(layer.rank).map_err(|_| {
                    InferenceError::Inference(format!(
                        "load_lora_adapter: rank {} exceeds u32",
                        layer.rank
                    ))
                })?;
                let d_in = u32::try_from(layer.d_in).map_err(|_| {
                    InferenceError::Inference(format!(
                        "load_lora_adapter: d_in {} exceeds u32",
                        layer.d_in
                    ))
                })?;
                let d_out = u32::try_from(layer.d_out).map_err(|_| {
                    InferenceError::Inference(format!(
                        "load_lora_adapter: d_out {} exceeds u32",
                        layer.d_out
                    ))
                })?;
                if rank > max_rank {
                    max_rank = rank;
                }
                let a_buf = device.new_buffer_with_data(
                    layer.a.as_ptr() as *const _,
                    (layer.a.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let b_buf = device.new_buffer_with_data(
                    layer.b.as_ptr() as *const _,
                    (layer.b.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                projections[layer.layer_idx].insert(
                    layer.module.clone(),
                    MetalLoraProjection {
                        a_buf,
                        b_buf,
                        rank,
                        d_in,
                        d_out,
                    },
                );
            }

            let intermediate =
                device.new_buffer((max_rank as u64) * 4, MTLResourceOptions::StorageModeShared);

            self.lora = Some(MetalLoraAdapter {
                projections,
                scale,
                intermediate,
                max_rank,
            });

            Ok(())
        }

        /// Unload the currently loaded LoRA adapter, freeing GPU buffers.
        pub fn unload_lora_adapter(&mut self) {
            self.lora = None;
        }

        /// Returns true if a LoRA adapter is currently loaded.
        pub fn has_lora_adapter(&self) -> bool {
            self.lora.is_some()
        }

        /// Dispatch LoRA GEMV for a single projection: y += scale * B @ (A @ x).
        ///
        /// No-op if no adapter is loaded or no adapter exists for the given layer/module.
        fn dispatch_lora_if_active(
            &self,
            enc: &ComputeCommandEncoderRef,
            x: &Buffer,
            y: &Buffer,
            layer_idx: usize,
            module: &str,
        ) {
            let Some(adapter) = &self.lora else { return };
            let Some(proj) = adapter.get_projection(layer_idx, module) else {
                return;
            };

            // Phase 1: intermediate = A @ x
            enc.set_compute_pipeline_state(&self.engine.pipelines.lora_gemv_a);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(&proj.a_buf), 0);
            enc.set_buffer(2, Some(&adapter.intermediate), 0);
            enc.set_bytes(3, 4, &proj.rank as *const u32 as *const _);
            enc.set_bytes(4, 4, &proj.d_in as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(proj.rank as u64, 1, 1),
                MTLSize::new(32, 4, 1),
            );

            // Phase 2: y += scale * B @ intermediate
            enc.set_compute_pipeline_state(&self.engine.pipelines.lora_gemv_b_accum);
            enc.set_buffer(0, Some(&adapter.intermediate), 0);
            enc.set_buffer(1, Some(&proj.b_buf), 0);
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 4, &proj.d_out as *const u32 as *const _);
            enc.set_bytes(4, 4, &proj.rank as *const u32 as *const _);
            enc.set_bytes(5, 4, &adapter.scale as *const f32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(proj.d_out.div_ceil(256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        /// Copy live GDN conv/S buffers into the given checkpoint slot (blocking GPU blit).
        fn checkpoint_gdn_to_slot(
            &mut self,
            slot: usize,
        ) -> Result<(), crate::error::InferenceError> {
            if self.session.gdn_checkpoints.is_none() {
                return Err(crate::error::InferenceError::Inference(
                    "GDN checkpoint pool not initialized".into(),
                ));
            }
            let num_layers = self.session.gdn_gpu_conv_bufs.len();
            let conv_sizes: Vec<u64> = self
                .session
                .gdn_gpu_conv_bufs
                .iter()
                .map(|b| b.length())
                .collect();
            let s_sizes: Vec<u64> = self
                .session
                .gdn_gpu_s_matrices
                .iter()
                .map(|b| b.length())
                .collect();
            let cmd = self.engine.queue.new_command_buffer();
            let enc = cmd.new_blit_command_encoder();
            for i in 0..num_layers {
                enc.copy_from_buffer(
                    &self.session.gdn_gpu_conv_bufs[i],
                    0,
                    &self.session.gdn_checkpoints.as_ref().unwrap().conv_slots[slot][i],
                    0,
                    conv_sizes[i],
                );
                enc.copy_from_buffer(
                    &self.session.gdn_gpu_s_matrices[i],
                    0,
                    &self.session.gdn_checkpoints.as_ref().unwrap().s_slots[slot][i],
                    0,
                    s_sizes[i],
                );
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            Ok(())
        }

        /// Restore live GDN conv/S buffers from the given checkpoint slot (blocking GPU blit).
        fn restore_gdn_slot_blocking(&mut self, slot: usize) {
            let num_layers = self.session.gdn_gpu_conv_bufs.len();
            let conv_sizes: Vec<u64> = self
                .session
                .gdn_gpu_conv_bufs
                .iter()
                .map(|b| b.length())
                .collect();
            let s_sizes: Vec<u64> = self
                .session
                .gdn_gpu_s_matrices
                .iter()
                .map(|b| b.length())
                .collect();
            let cmd = self.engine.queue.new_command_buffer();
            let enc = cmd.new_blit_command_encoder();
            for i in 0..num_layers {
                enc.copy_from_buffer(
                    &self.session.gdn_checkpoints.as_ref().unwrap().conv_slots[slot][i],
                    0,
                    &self.session.gdn_gpu_conv_bufs[i],
                    0,
                    conv_sizes[i],
                );
                enc.copy_from_buffer(
                    &self.session.gdn_checkpoints.as_ref().unwrap().s_slots[slot][i],
                    0,
                    &self.session.gdn_gpu_s_matrices[i],
                    0,
                    s_sizes[i],
                );
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        /// Roll back KV cache and GDN state to `seq_len` after a speculative rejection.
        ///
        /// Restores from slot `seq_len - active_base_seq_len` in the checkpoint pool.
        fn rollback_speculative_state_to(
            &mut self,
            seq_len: usize,
        ) -> Result<(), crate::error::InferenceError> {
            let base = self
                .session
                .gdn_checkpoints
                .as_ref()
                .and_then(|p| p.active_base_seq_len)
                .ok_or_else(|| {
                    crate::error::InferenceError::Inference(
                        "No active speculation checkpoint".into(),
                    )
                })?;
            let slot = seq_len.checked_sub(base).ok_or_else(|| {
                crate::error::InferenceError::Inference(format!(
                    "rollback seq_len {seq_len} < base {base}"
                ))
            })?;

            // Extract batch-repair info and MTP base before any mutable borrows.
            let batch_repair = self
                .session
                .gdn_checkpoints
                .as_mut()
                .and_then(|p| p.batch_repair_token.take());
            let mtp_base = self
                .session
                .gdn_checkpoints
                .as_ref()
                .and_then(|p| p.mtp_base_seq_len);

            if let Some((repair_token, repair_pos)) = batch_repair {
                // Batch verifier path: restore GDN base, then replay the first verify token so
                // the GDN state lands at S_{repair_pos} (= state after pending_token was processed).
                self.restore_gdn_slot_blocking(0);
                // Set seq_len so forward_step_inner writes KV to the correct slot.
                self.session.kv_cache.seq_len = repair_pos;
                // Repair step: reprocesses pending_token, advancing GDN and capturing hidden.
                let repair_out = self.forward_step_inner(repair_token, repair_pos, true);
                self.session.last_pre_final_hidden = repair_out.pre_final_hidden;
                self.session.kv_cache.seq_len = seq_len;
            } else {
                // Sequential verifier path: slot-based GDN restore.
                self.restore_gdn_slot_blocking(slot);
                self.session.kv_cache.seq_len = seq_len;
            }

            // Bug 3 fix: restore MTP KV cache position to prevent ghost entries.
            if let Some(base_mtp) = mtp_base {
                if let Some(ref mut mtp) = self.session.mtp {
                    mtp.cache.seq_len = base_mtp + slot;
                }
            }
            Ok(())
        }

        /// Run the target model sequentially on `tokens` starting at `start_pos`,
        /// checkpointing GDN state after each token for speculative rollback.
        ///
        /// Returns per-token logits and the pre-final hidden state of the last token.
        /// This is the H1 sequential path (V_2≈2.0); a batch-GEMM verifier is a follow-up.
        fn verify_tokens_batched(
            &mut self,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<MetalVerifyOutput, crate::error::InferenceError> {
            if self.session.gdn_checkpoints.is_none() {
                return Err(crate::error::InferenceError::Inference(
                    "GDN checkpoint pool required for verify_tokens_batched".into(),
                ));
            }
            if let Some(ref mut p) = self.session.gdn_checkpoints {
                p.active_base_seq_len = Some(start_pos);
                p.mtp_base_seq_len = self.session.mtp.as_ref().map(|m| m.cache.seq_len);
            }
            self.checkpoint_gdn_to_slot(0)?;
            let mut all_logits: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
            let mut final_hidden = Vec::new();
            for (i, &token) in tokens.iter().enumerate() {
                let out = self.forward_step_inner(token, start_pos + i, true);
                self.checkpoint_gdn_to_slot(i + 1)?;
                all_logits.push(out.logits);
                final_hidden = out.pre_final_hidden;
            }
            Ok(MetalVerifyOutput {
                logits: all_logits,
                final_hidden,
            })
        }

        /// Batch-GEMM verifier (activate with `LATTICE_MTP_BATCH=1`).
        ///
        /// Processes K verify tokens layer-by-layer using batch GEMM for all weight projections,
        /// with sequential GDN recurrence within each layer. Reduces weight-load cost vs the
        /// sequential verifier: each weight matrix is loaded once for K outputs instead of K times.
        ///
        /// **GDN rollback strategy**: Checkpoints GDN base (slot 0) before the batch. On rejection,
        /// `rollback_speculative_state_to` restores slot 0 and runs one repair forward step for the
        /// pending token to re-establish the correct intermediate GDN state.
        ///
        /// Expected V_2: ~1.35 at α=100%, ~1.60 at α=75%, ~2.30 at α=5%.
        fn verify_tokens_batch_gemm(
            &mut self,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<MetalVerifyOutput, crate::error::InferenceError> {
            let n = tokens.len();
            if n == 0 || n > MTP_VERIFY_MAX_TOKENS {
                return Err(crate::error::InferenceError::Inference(format!(
                    "verify_batch: bad token count {n} (max {MTP_VERIFY_MAX_TOKENS})"
                )));
            }
            if self.session.gdn_checkpoints.is_none() {
                return Err(crate::error::InferenceError::Inference(
                    "GDN checkpoint pool required for verify_tokens_batch_gemm".into(),
                ));
            }
            assert!(
                start_pos + n <= self.session.kv_cache.max_cache_len,
                "verify_batch: would overflow KV cache (start={start_pos} n={n} max={})",
                self.session.kv_cache.max_cache_len
            );

            let cfg = self.engine.config.clone();
            let hidden = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            let kv_dim = cfg.full_kv_dim();
            let q_dim = cfg.full_q_dim();
            let num_q_heads = cfg.num_attention_heads;
            let num_kv_heads = cfg.num_key_value_heads;
            let head_dim = cfg.head_dim;
            let half_rope_dim = (cfg.rope_dim() / 2) as u32;
            let m = n as u32;

            // Checkpoint GDN base + record repair token for rollback.
            if let Some(ref mut p) = self.session.gdn_checkpoints {
                p.active_base_seq_len = Some(start_pos);
                p.mtp_base_seq_len = self.session.mtp.as_ref().map(|mtp| mtp.cache.seq_len);
                p.batch_repair_token = if n > 0 {
                    Some((tokens[0], start_pos))
                } else {
                    None
                };
            }
            self.checkpoint_gdn_to_slot(0)?;

            // Batch embedding: f16 → f32 for all K tokens into hidden[0..K*hidden].
            // SAFETY: embed_tokens is StorageModeShared f16, no GPU in flight;
            // each id is validated < vocab_size before computing offset.
            unsafe {
                let src_base = self.engine.embed_tokens.contents() as *const u16;
                let dst_base = self.session.activations.hidden.contents() as *mut f32;
                for (t, &id) in tokens.iter().enumerate() {
                    assert!(
                        (id as usize) < cfg.vocab_size,
                        "verify_batch: token {id} >= vocab_size {}",
                        cfg.vocab_size
                    );
                    let src = src_base.add(id as usize * hidden);
                    let dst = dst_base.add(t * hidden);
                    for i in 0..hidden {
                        *dst.add(i) = f16_to_f32(*src.add(i));
                    }
                }
            }

            let mut active_layer_idx = 0usize;
            let mut linear_idx = 0usize;
            let mut full_idx = 0usize;

            let cmd = self.engine.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            for layer_i in 0..cfg.num_hidden_layers {
                if !cfg.is_layer_active(layer_i) {
                    continue;
                }
                let compact_idx = active_layer_idx;
                active_layer_idx += 1;
                let is_linear = matches!(
                    &self.engine.layer_weights[compact_idx].0,
                    MetalLayerAttnWeights::Linear(_)
                );
                let (_, common_w) = &self.engine.layer_weights[compact_idx];
                let w_in_norm = &common_w.input_layernorm as *const Buffer;
                let w_post_norm = &common_w.post_attention_layernorm as *const Buffer;
                let w_gate = &common_w.gate_proj as *const Q4WeightBuf;
                let w_up = &common_w.up_proj as *const Q4WeightBuf;
                let w_down = &common_w.down_proj as *const Q4WeightBuf;

                if is_linear {
                    // GDN layer: batch projections + sequential recurrence (causal dependency).
                    let (w_qkv, w_z, w_b, w_a, w_alog, w_dtb, w_conv, w_norm, w_out) = {
                        let gdn_w = match &self.engine.layer_weights[compact_idx].0 {
                            MetalLayerAttnWeights::Linear(w) => w,
                            _ => unreachable!(),
                        };
                        (
                            &gdn_w.in_proj_qkv as *const Q4WeightBuf,
                            &gdn_w.in_proj_z as *const Q4WeightBuf,
                            &gdn_w.in_proj_b as *const Buffer,
                            &gdn_w.in_proj_a as *const Buffer,
                            &gdn_w.a_log as *const Buffer,
                            &gdn_w.dt_bias as *const Buffer,
                            &gdn_w.conv1d_weight as *const Buffer,
                            &gdn_w.norm_weight as *const Buffer,
                            &gdn_w.out_proj as *const Q4WeightBuf,
                        )
                    };
                    let qkv_d = cfg.linear_qkv_dim();
                    let out_d = cfg.linear_output_dim();
                    let num_h = cfg.linear_num_key_heads;
                    let key_d = cfg.linear_key_head_dim;
                    let val_d = cfg.linear_value_head_dim;
                    let ks = cfg.linear_conv_kernel_dim as u32;

                    // Batch: copy hidden → residual, norm hidden.
                    self.dispatch_copy(
                        enc,
                        &self.session.activations.hidden,
                        &self.session.activations.residual,
                        m * hidden as u32,
                    );
                    // SAFETY: w_in_norm is a live layer-owned buffer pointer.
                    unsafe {
                        self.dispatch_rms_norm(
                            enc,
                            &self.session.activations.hidden,
                            &*w_in_norm,
                            hidden as u32,
                            m,
                            cfg.rms_norm_eps,
                        );
                    }

                    // Batch: QKV + Z projections (GEMM, M=K — one weight load for K outputs).
                    // SAFETY: w_qkv, w_z are live layer-owned buffers.
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_qkv,
                            &self.session.activations.gdn_qkv,
                            0,
                            m,
                            qkv_d as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_z,
                            &self.session.activations.gdn_z,
                            0,
                            m,
                            out_d as u32,
                            hidden as u32,
                        );
                    }

                    // Sequential: conv1d + recurrence per token (inherently causal).
                    // SAFETY: Token offsets are within m-sized activation buffers;
                    // conv/state/weight buffers are live for the command buffer.
                    for t in 0..n {
                        let qkv_off = (t * qkv_d) as u64 * 4;
                        let z_off = (t * out_d) as u64 * 4;
                        let h_off = (t * hidden) as u64 * 4;
                        unsafe {
                            enc.set_compute_pipeline_state(&self.engine.pipelines.conv1d_silu);
                            enc.set_buffer(0, Some(&self.session.gdn_gpu_conv_bufs[linear_idx]), 0);
                            enc.set_buffer(1, Some(&self.session.activations.gdn_qkv), qkv_off);
                            enc.set_buffer(2, Some(&*w_conv), 0);
                            enc.set_buffer(3, Some(&self.session.gdn_gpu_conv_out), 0);
                            let qd = qkv_d as u32;
                            enc.set_bytes(4, 4, &qd as *const u32 as *const _);
                            enc.set_bytes(5, 4, &ks as *const u32 as *const _);
                            let wg = 256u64;
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(qkv_d as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );

                            #[repr(C)]
                            struct GdnRecurParams {
                                key_dim: u32,
                                value_dim: u32,
                                num_key_heads: u32,
                                num_value_heads: u32,
                                hidden_size: u32,
                                q_total: u32,
                                v_offset: u32,
                                scale: f32,
                                eps: f32,
                            }
                            let num_vh = cfg.linear_num_value_heads();
                            let q_total = (num_h * key_d) as u32;
                            let params = GdnRecurParams {
                                key_dim: key_d as u32,
                                value_dim: val_d as u32,
                                num_key_heads: num_h as u32,
                                num_value_heads: num_vh as u32,
                                hidden_size: hidden as u32,
                                q_total,
                                v_offset: q_total * 2,
                                scale: 1.0 / (key_d as f32).sqrt(),
                                eps: cfg.rms_norm_eps,
                            };
                            enc.set_compute_pipeline_state(&self.engine.pipelines.gdn_recurrence);
                            enc.set_buffer(
                                0,
                                Some(&self.session.gdn_gpu_s_matrices[linear_idx]),
                                0,
                            );
                            enc.set_buffer(1, Some(&self.session.gdn_gpu_conv_out), 0);
                            enc.set_buffer(2, Some(&self.session.activations.gdn_z), z_off);
                            enc.set_buffer(3, Some(&self.session.activations.hidden), h_off);
                            enc.set_buffer(4, Some(&*w_b), 0);
                            enc.set_buffer(5, Some(&*w_a), 0);
                            enc.set_buffer(6, Some(&*w_alog), 0);
                            enc.set_buffer(7, Some(&*w_dtb), 0);
                            enc.set_buffer(8, Some(&*w_norm), 0);
                            enc.set_buffer(9, Some(&self.session.activations.gdn_z), z_off);
                            enc.set_bytes(
                                10,
                                std::mem::size_of::<GdnRecurParams>() as u64,
                                &params as *const GdnRecurParams as *const _,
                            );
                            enc.dispatch_thread_groups(
                                MTLSize::new(num_vh as u64, 1, 1),
                                MTLSize::new(32, 4, 1),
                            );
                        }
                    }

                    // Batch: out_proj + residual + post-norm + MLP.
                    // SAFETY: w_out, w_gate, w_up, w_down are live layer-owned buffers.
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.gdn_z,
                            0,
                            &*w_out,
                            &self.session.activations.attn_out,
                            0,
                            m,
                            hidden as u32,
                            out_d as u32,
                        );
                    }
                    self.dispatch_add(
                        enc,
                        &self.session.activations.attn_out,
                        &self.session.activations.residual,
                        m * hidden as u32,
                    );
                    self.dispatch_copy(
                        enc,
                        &self.session.activations.residual,
                        &self.session.activations.hidden,
                        m * hidden as u32,
                    );
                    unsafe {
                        self.dispatch_rms_norm(
                            enc,
                            &self.session.activations.hidden,
                            &*w_post_norm,
                            hidden as u32,
                            m,
                            cfg.rms_norm_eps,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_gate,
                            &self.session.activations.gate,
                            0,
                            m,
                            inter as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_up,
                            &self.session.activations.up,
                            0,
                            m,
                            inter as u32,
                            hidden as u32,
                        );
                    }
                    self.dispatch_silu_mul(enc, m * inter as u32);
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.gate,
                            0,
                            &*w_down,
                            &self.session.activations.ffn_out,
                            0,
                            m,
                            hidden as u32,
                            inter as u32,
                        );
                    }
                    self.dispatch_add(
                        enc,
                        &self.session.activations.ffn_out,
                        &self.session.activations.residual,
                        m * hidden as u32,
                    );
                    self.dispatch_copy(
                        enc,
                        &self.session.activations.residual,
                        &self.session.activations.hidden,
                        m * hidden as u32,
                    );
                    linear_idx += 1;
                } else {
                    // Full-attention layer: batch QKV/O/MLP projections, sequential per-token ops.
                    let (w_q, w_k, w_v, w_o, w_qn, w_kn) = {
                        let full_w = match &self.engine.layer_weights[compact_idx].0 {
                            MetalLayerAttnWeights::Full(w) => w,
                            _ => unreachable!(),
                        };
                        (
                            &full_w.q_proj as *const Q4WeightBuf,
                            &full_w.k_proj as *const Q4WeightBuf,
                            &full_w.v_proj as *const Q4WeightBuf,
                            &full_w.o_proj as *const Q4WeightBuf,
                            &full_w.q_norm as *const Buffer,
                            &full_w.k_norm as *const Buffer,
                        )
                    };
                    let scale = 1.0f32 / (head_dim as f32).sqrt();

                    // Batch: copy + in-norm.
                    self.dispatch_copy(
                        enc,
                        &self.session.activations.hidden,
                        &self.session.activations.residual,
                        m * hidden as u32,
                    );
                    // SAFETY: w_in_norm is a live layer-owned buffer pointer.
                    unsafe {
                        self.dispatch_rms_norm(
                            enc,
                            &self.session.activations.hidden,
                            &*w_in_norm,
                            hidden as u32,
                            m,
                            cfg.rms_norm_eps,
                        );
                    }

                    // Batch: Q/K/V projections.
                    // SAFETY: w_q, w_k, w_v are live layer-owned buffers.
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_q,
                            &self.session.activations.q,
                            0,
                            m,
                            (2 * q_dim) as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_k,
                            &self.session.activations.k,
                            0,
                            m,
                            kv_dim as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_v,
                            &self.session.activations.v,
                            0,
                            m,
                            kv_dim as u32,
                            hidden as u32,
                        );
                    }

                    // Sequential: scatter, norm, RoPE, KV store, attention, gate (causal).
                    // KV cache writes at absolute positions start_pos+t; attention sees cache[0..start_pos+t+1].
                    // SAFETY: Per-token offsets are within m-sized activation buffers;
                    // KV cache buffers are live StorageModeShared, sized for max_cache_len.
                    for t in 0..n {
                        let q_off = (t * 2 * q_dim) as u64 * 4;
                        let qs_off = (t * q_dim) as u64 * 4;
                        let gz_off = (t * q_dim) as u64 * 4;
                        let k_off = (t * kv_dim) as u64 * 4;
                        let v_off = (t * kv_dim) as u64 * 4;
                        let ao_off = (t * q_dim) as u64 * 4;
                        let abs_pos = start_pos + t;
                        let cache_len = (abs_pos + 1) as u32;
                        let kv_dst_off = (abs_pos * kv_dim) as u32;

                        unsafe {
                            // Scatter Q + gate from interleaved q_proj.
                            enc.set_compute_pipeline_state(&self.engine.pipelines.scatter_q_gate);
                            enc.set_buffer(0, Some(&self.session.activations.q), q_off);
                            enc.set_buffer(1, Some(&self.session.activations.q_separated), qs_off);
                            enc.set_buffer(2, Some(&self.session.activations.gate_z), gz_off);
                            let nh = num_q_heads as u32;
                            let hd = head_dim as u32;
                            enc.set_bytes(3, 4, &nh as *const u32 as *const _);
                            enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                            let wg = 256u64;
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );

                            // Per-head RMS norm Q and K.
                            enc.set_compute_pipeline_state(
                                &self.engine.pipelines.per_head_rms_norm,
                            );
                            enc.set_buffer(0, Some(&self.session.activations.q_separated), qs_off);
                            enc.set_buffer(1, Some(&*w_qn), 0);
                            enc.set_bytes(2, 4, &nh as *const u32 as *const _);
                            enc.set_bytes(3, 4, &hd as *const u32 as *const _);
                            enc.set_bytes(4, 4, &cfg.rms_norm_eps as *const f32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(nh as u64, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                            enc.set_buffer(0, Some(&self.session.activations.k), k_off);
                            enc.set_buffer(1, Some(&*w_kn), 0);
                            let nkh = num_kv_heads as u32;
                            enc.set_bytes(2, 4, &nkh as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(nkh as u64, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );

                            // Partial RoPE for Q and K at absolute position (start_pos + t).
                            let pos = abs_pos as u32;
                            enc.set_compute_pipeline_state(&self.engine.pipelines.partial_rope);
                            enc.set_buffer(0, Some(&self.session.activations.q_separated), qs_off);
                            enc.set_buffer(1, Some(&self.engine.rope_cos), 0);
                            enc.set_buffer(2, Some(&self.engine.rope_sin), 0);
                            enc.set_bytes(3, 4, &nh as *const u32 as *const _);
                            enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                            enc.set_bytes(5, 4, &half_rope_dim as *const u32 as *const _);
                            enc.set_bytes(6, 4, &pos as *const u32 as *const _);
                            let wg = 256u64;
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );
                            enc.set_buffer(0, Some(&self.session.activations.k), k_off);
                            let nkh = num_kv_heads as u32;
                            enc.set_bytes(3, 4, &nkh as *const u32 as *const _);
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(kv_dim as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );
                        }

                        // KV store at absolute position (start_pos + t).
                        enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset);
                        enc.set_buffer(0, Some(&self.session.activations.k), k_off);
                        enc.set_buffer(1, Some(&self.session.kv_cache.k_bufs[full_idx]), 0);
                        let cnt = kv_dim as u32;
                        enc.set_bytes(2, 4, &cnt as *const u32 as *const _);
                        enc.set_bytes(3, 4, &kv_dst_off as *const u32 as *const _);
                        let wg = 256u64;
                        enc.dispatch_threads(
                            MTLSize::new(div_ceil(kv_dim as u64, wg) * wg, 1, 1),
                            MTLSize::new(wg, 1, 1),
                        );
                        enc.set_buffer(0, Some(&self.session.activations.v), v_off);
                        enc.set_buffer(1, Some(&self.session.kv_cache.v_bufs[full_idx]), 0);
                        enc.dispatch_threads(
                            MTLSize::new(div_ceil(kv_dim as u64, wg) * wg, 1, 1),
                            MTLSize::new(wg, 1, 1),
                        );

                        // Causal attention: Q[t] against full cache[0..start_pos+t+1].
                        enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attention);
                        enc.set_buffer(0, Some(&self.session.activations.q_separated), qs_off);
                        enc.set_buffer(1, Some(&self.session.kv_cache.k_bufs[full_idx]), 0);
                        enc.set_buffer(2, Some(&self.session.kv_cache.v_bufs[full_idx]), 0);
                        enc.set_buffer(3, Some(&self.session.activations.attn_out), ao_off);
                        enc.set_bytes(4, 4, &cache_len as *const u32 as *const _);
                        let hd = head_dim as u32;
                        let nqh = num_q_heads as u32;
                        let nkh = num_kv_heads as u32;
                        let qd = q_dim as u32;
                        let kvd = kv_dim as u32;
                        enc.set_bytes(5, 4, &hd as *const u32 as *const _);
                        enc.set_bytes(6, 4, &nqh as *const u32 as *const _);
                        enc.set_bytes(7, 4, &nkh as *const u32 as *const _);
                        enc.set_bytes(8, 4, &qd as *const u32 as *const _);
                        enc.set_bytes(9, 4, &kvd as *const u32 as *const _);
                        enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new(nqh as u64, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );

                        // Sigmoid gate.
                        let cnt = q_dim as u32;
                        enc.set_compute_pipeline_state(&self.engine.pipelines.sigmoid_gate);
                        enc.set_buffer(0, Some(&self.session.activations.attn_out), ao_off);
                        enc.set_buffer(1, Some(&self.session.activations.gate_z), gz_off);
                        enc.set_bytes(2, 4, &cnt as *const u32 as *const _);
                        let wg = 256u64;
                        enc.dispatch_threads(
                            MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                            MTLSize::new(wg, 1, 1),
                        );
                    }

                    // Batch: O projection + residual + post-norm + MLP.
                    // SAFETY: w_o, w_gate, w_up, w_down are live layer-owned buffers.
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.attn_out,
                            0,
                            &*w_o,
                            &self.session.activations.ffn_out,
                            0,
                            m,
                            hidden as u32,
                            q_dim as u32,
                        );
                    }
                    self.dispatch_add(
                        enc,
                        &self.session.activations.ffn_out,
                        &self.session.activations.residual,
                        m * hidden as u32,
                    );
                    self.dispatch_copy(
                        enc,
                        &self.session.activations.residual,
                        &self.session.activations.hidden,
                        m * hidden as u32,
                    );
                    unsafe {
                        self.dispatch_rms_norm(
                            enc,
                            &self.session.activations.hidden,
                            &*w_post_norm,
                            hidden as u32,
                            m,
                            cfg.rms_norm_eps,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_gate,
                            &self.session.activations.gate,
                            0,
                            m,
                            inter as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_up,
                            &self.session.activations.up,
                            0,
                            m,
                            inter as u32,
                            hidden as u32,
                        );
                    }
                    self.dispatch_silu_mul(enc, m * inter as u32);
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.gate,
                            0,
                            &*w_down,
                            &self.session.activations.ffn_out,
                            0,
                            m,
                            hidden as u32,
                            inter as u32,
                        );
                    }
                    self.dispatch_add(
                        enc,
                        &self.session.activations.ffn_out,
                        &self.session.activations.residual,
                        m * hidden as u32,
                    );
                    self.dispatch_copy(
                        enc,
                        &self.session.activations.residual,
                        &self.session.activations.hidden,
                        m * hidden as u32,
                    );
                    full_idx += 1;
                }
            } // end layer loop

            // Capture pre-final hidden for last token (for MTP next round).
            let last_off = ((n - 1) * hidden) as u64 * 4;
            if self.session.mtp.is_some() {
                enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset);
                enc.set_buffer(0, Some(&self.session.activations.hidden), last_off);
                enc.set_buffer(1, Some(&self.session.activations.pre_final_hidden), 0);
                let cnt = hidden as u32;
                let dst_off = 0u32;
                enc.set_bytes(2, 4, &cnt as *const u32 as *const _);
                enc.set_bytes(3, 4, &dst_off as *const u32 as *const _);
                let wg = 256u64;
                enc.dispatch_threads(
                    MTLSize::new(div_ceil(hidden as u64, wg) * wg, 1, 1),
                    MTLSize::new(wg, 1, 1),
                );
            }

            // Final RMSNorm for all K tokens (batch).
            self.dispatch_rms_norm(
                enc,
                &self.session.activations.hidden,
                &self.engine.final_norm,
                hidden as u32,
                m,
                cfg.rms_norm_eps,
            );

            // Logits GEMM for all K tokens → verify_logits[K * vocab_size].
            self.dispatch_gemm(
                enc,
                &self.session.activations.hidden,
                0,
                &self.engine.embed_tokens_q8,
                &self.session.activations.verify_logits,
                0,
                m,
                cfg.vocab_size as u32,
                hidden as u32,
            );

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // Advance KV cache position.
            self.session.kv_cache.seq_len = start_pos + n;

            // Read back per-token logits from verify_logits buffer.
            let vocab = cfg.vocab_size;
            let flat = unsafe { read_buffer(&self.session.activations.verify_logits, n * vocab) };
            let all_logits: Vec<Vec<f32>> = (0..n)
                .map(|t| flat[t * vocab..(t + 1) * vocab].to_vec())
                .collect();

            // Read back and update pre-final hidden (on accept, used by next mtp_forward_one).
            let final_hidden = if self.session.mtp.is_some() {
                let h = unsafe { read_buffer(&self.session.activations.pre_final_hidden, hidden) };
                self.session.last_pre_final_hidden = h.clone();
                h
            } else {
                Vec::new()
            };

            Ok(MetalVerifyOutput {
                logits: all_logits,
                final_hidden,
            })
        }

        /// Draft one extra token using the MTP module.
        ///
        /// Runs the single MTP attention+MLP layer on top of the target model's
        /// pre-final hidden state (`self.session.last_pre_final_hidden`) to predict
        /// `pending_token + 1`.  Returns the draft token id and logits.
        ///
        /// Panics if `self.session.mtp` is None.
        fn mtp_forward_one(
            &mut self,
            pending_token: u32,
            position: usize,
        ) -> MetalMtpForwardOutput {
            let cfg = self.engine.config.clone();
            let hidden = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            let num_q_heads = cfg.num_attention_heads;
            let num_kv_heads = cfg.num_key_value_heads;
            let head_dim = cfg.head_dim;
            let q_dim = cfg.full_q_dim();
            let kv_dim = cfg.full_kv_dim();
            let half_rope_dim = (cfg.rope_dim() / 2) as u32;
            let scale = 1.0_f32 / (head_dim as f32).sqrt();

            // ---- Phase 1: CPU ----

            // 1. Embedding lookup (f16 → f32), same token as was just generated.
            let mut normed_embed = vec![0.0f32; hidden];
            assert!((pending_token as usize) < cfg.vocab_size);
            unsafe {
                let src = (self.engine.embed_tokens.contents() as *const u16)
                    .add(pending_token as usize * hidden);
                for i in 0..hidden {
                    normed_embed[i] = f16_to_f32(*src.add(i));
                }
            }

            // 2. CPU RMSNorm of embedding using pre_fc_norm_embedding weights.
            {
                let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
                let gamma = unsafe {
                    std::slice::from_raw_parts(
                        mtp_weights.pre_fc_norm_embedding.contents() as *const f32,
                        hidden,
                    )
                };
                let mut sum_sq = 0.0f32;
                for &v in normed_embed.iter() {
                    sum_sq += v * v;
                }
                let inv_rms = 1.0 / (sum_sq / hidden as f32 + cfg.rms_norm_eps).sqrt();
                for (v, &g) in normed_embed.iter_mut().zip(gamma.iter()) {
                    *v = *v * inv_rms * g;
                }
            }

            // 3. CPU RMSNorm of target pre-final hidden using pre_fc_norm_hidden weights.
            let mut normed_hidden = self.session.last_pre_final_hidden.clone();
            {
                let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
                let gamma = unsafe {
                    std::slice::from_raw_parts(
                        mtp_weights.pre_fc_norm_hidden.contents() as *const f32,
                        hidden,
                    )
                };
                let mut sum_sq = 0.0f32;
                for &v in normed_hidden.iter() {
                    sum_sq += v * v;
                }
                let inv_rms = 1.0 / (sum_sq / hidden as f32 + cfg.rms_norm_eps).sqrt();
                for (v, &g) in normed_hidden.iter_mut().zip(gamma.iter()) {
                    *v = *v * inv_rms * g;
                }
            }

            // 4. CPU concat [normed_embed || normed_hidden] into fused buffer (2*hidden).
            {
                let mtp = self.session.mtp.as_ref().unwrap();
                let dst = mtp.activations.fused.contents() as *mut f32;
                unsafe {
                    for (i, &v) in normed_embed.iter().enumerate() {
                        *dst.add(i) = v;
                    }
                    for (i, &v) in normed_hidden.iter().enumerate() {
                        *dst.add(hidden + i) = v;
                    }
                }
            }

            // ---- Phase 2: GPU ----
            // Collect raw pointers before encoder to avoid simultaneous borrow conflicts.
            let (
                w_fc,
                w_input_ln,
                w_post_attn_ln,
                w_q_proj,
                w_k_proj,
                w_v_proj,
                w_o_proj,
                w_q_norm,
                w_k_norm,
                w_gate_proj,
                w_up_proj,
                w_down_proj,
                w_norm,
            ) = {
                let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
                let lw = &mtp_weights.layers[0];
                (
                    &mtp_weights.fc as *const Q4WeightBuf,
                    &lw.input_layernorm as *const Buffer,
                    &lw.post_attention_layernorm as *const Buffer,
                    &lw.q_proj as *const Q4WeightBuf,
                    &lw.k_proj as *const Q4WeightBuf,
                    &lw.v_proj as *const Q4WeightBuf,
                    &lw.o_proj as *const Q4WeightBuf,
                    &lw.q_norm as *const Buffer,
                    &lw.k_norm as *const Buffer,
                    &lw.mlp.gate_proj as *const Q4WeightBuf,
                    &lw.mlp.up_proj as *const Q4WeightBuf,
                    &lw.mlp.down_proj as *const Q4WeightBuf,
                    &mtp_weights.norm as *const Buffer,
                )
            };
            let (
                buf_fused,
                buf_hidden,
                buf_residual,
                buf_q,
                buf_q_sep,
                buf_gate_z,
                buf_k,
                buf_v,
                buf_attn_out,
                buf_gate,
                buf_up,
                buf_ffn_out,
                buf_logits,
            ) = {
                let a = &self.session.mtp.as_ref().unwrap().activations;
                (
                    &a.fused as *const Buffer,
                    &a.hidden as *const Buffer,
                    &a.residual as *const Buffer,
                    &a.q as *const Buffer,
                    &a.q_separated as *const Buffer,
                    &a.gate_z as *const Buffer,
                    &a.k as *const Buffer,
                    &a.v as *const Buffer,
                    &a.attn_out as *const Buffer,
                    &a.gate as *const Buffer,
                    &a.up as *const Buffer,
                    &a.ffn_out as *const Buffer,
                    &a.logits as *const Buffer,
                )
            };
            let (buf_k_cache, buf_v_cache, mtp_seq_len) = {
                let c = &self.session.mtp.as_ref().unwrap().cache;
                (
                    &c.k_buf as *const Buffer,
                    &c.v_buf as *const Buffer,
                    c.seq_len,
                )
            };

            let cmd = self.engine.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            let wg = 256u64;

            unsafe {
                // fc: fused (2*hidden) → hidden
                self.dispatch_matmul(
                    enc,
                    &*buf_fused,
                    &*w_fc,
                    &*buf_hidden,
                    1,
                    hidden as u32,
                    (2 * hidden) as u32,
                );

                // Pre-attention: copy hidden → residual, RMSNorm hidden in-place.
                self.dispatch_copy_and_rms_norm(
                    enc,
                    &*buf_hidden,
                    &*buf_residual,
                    &*w_input_ln,
                    hidden as u32,
                    cfg.rms_norm_eps,
                );

                // Q/K/V projections.
                self.dispatch_matmul(
                    enc,
                    &*buf_hidden,
                    &*w_q_proj,
                    &*buf_q,
                    1,
                    (2 * q_dim) as u32,
                    hidden as u32,
                );
                self.dispatch_matmul(
                    enc,
                    &*buf_hidden,
                    &*w_k_proj,
                    &*buf_k,
                    1,
                    kv_dim as u32,
                    hidden as u32,
                );
                self.dispatch_matmul(
                    enc,
                    &*buf_hidden,
                    &*w_v_proj,
                    &*buf_v,
                    1,
                    kv_dim as u32,
                    hidden as u32,
                );

                // Scatter Q+gate from q → q_separated, gate_z (inline).
                enc.set_compute_pipeline_state(&self.engine.pipelines.scatter_q_gate);
                enc.set_buffer(0, Some(&*buf_q), 0);
                enc.set_buffer(1, Some(&*buf_q_sep), 0);
                enc.set_buffer(2, Some(&*buf_gate_z), 0);
                let nh = num_q_heads as u32;
                let hd = head_dim as u32;
                enc.set_bytes(3, 4, &nh as *const u32 as *const _);
                enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                enc.dispatch_threads(
                    MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                    MTLSize::new(wg, 1, 1),
                );

                // Per-head RMS norm on Q and K.
                self.dispatch_per_head_rms_norm(
                    enc,
                    &*buf_q_sep,
                    &*w_q_norm,
                    num_q_heads as u32,
                    hd,
                    cfg.rms_norm_eps,
                );
                self.dispatch_per_head_rms_norm(
                    enc,
                    &*buf_k,
                    &*w_k_norm,
                    num_kv_heads as u32,
                    hd,
                    cfg.rms_norm_eps,
                );

                // Partial RoPE.
                self.dispatch_partial_rope(
                    enc,
                    &*buf_q_sep,
                    num_q_heads as u32,
                    hd,
                    half_rope_dim,
                    position as u32,
                );
                self.dispatch_partial_rope(
                    enc,
                    &*buf_k,
                    num_kv_heads as u32,
                    hd,
                    half_rope_dim,
                    position as u32,
                );

                // Append K,V to MTP KV cache.
                let kv_cache_off = (mtp_seq_len * kv_dim) as u32;
                self.dispatch_copy_offset(enc, &*buf_k, &*buf_k_cache, kv_dim as u32, kv_cache_off);
                self.dispatch_copy_offset(enc, &*buf_v, &*buf_v_cache, kv_dim as u32, kv_cache_off);

                // Flash decode attention (direct path — MTP cache is always short).
                let cur_mtp_seq = (mtp_seq_len + 1) as u32;
                enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attention);
                enc.set_buffer(0, Some(&*buf_q_sep), 0);
                enc.set_buffer(1, Some(&*buf_k_cache), 0);
                enc.set_buffer(2, Some(&*buf_v_cache), 0);
                enc.set_buffer(3, Some(&*buf_attn_out), 0);
                enc.set_bytes(4, 4, &cur_mtp_seq as *const u32 as *const _);
                enc.set_bytes(5, 4, &hd as *const u32 as *const _);
                let nq = num_q_heads as u32;
                let nkv = num_kv_heads as u32;
                let qd = q_dim as u32;
                let kvd = kv_dim as u32;
                enc.set_bytes(6, 4, &nq as *const u32 as *const _);
                enc.set_bytes(7, 4, &nkv as *const u32 as *const _);
                enc.set_bytes(8, 4, &qd as *const u32 as *const _);
                enc.set_bytes(9, 4, &kvd as *const u32 as *const _);
                enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_kv_heads as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );

                // Sigmoid gate: attn_out *= sigmoid(gate_z) (inline).
                enc.set_compute_pipeline_state(&self.engine.pipelines.sigmoid_gate);
                enc.set_buffer(0, Some(&*buf_attn_out), 0);
                enc.set_buffer(1, Some(&*buf_gate_z), 0);
                let qd_u32 = q_dim as u32;
                enc.set_bytes(2, 4, &qd_u32 as *const u32 as *const _);
                enc.dispatch_threads(
                    MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                    MTLSize::new(wg, 1, 1),
                );

                // O projection: attn_out → ffn_out.
                self.dispatch_matmul(
                    enc,
                    &*buf_attn_out,
                    &*w_o_proj,
                    &*buf_ffn_out,
                    1,
                    hidden as u32,
                    q_dim as u32,
                );
                // Copy ffn_out → attn_out for use as delta in fused residual+norm.
                self.dispatch_copy(enc, &*buf_ffn_out, &*buf_attn_out, hidden as u32);

                // Post-attention residual + norm (MLP input).
                self.dispatch_fused_residual_add_norm(
                    enc,
                    &*buf_residual,
                    &*buf_attn_out,
                    &*buf_residual,
                    &*buf_hidden,
                    &*w_post_attn_ln,
                    hidden as u32,
                    cfg.rms_norm_eps,
                );

                // Dense MLP.
                self.dispatch_matmul(
                    enc,
                    &*buf_hidden,
                    &*w_gate_proj,
                    &*buf_gate,
                    1,
                    inter as u32,
                    hidden as u32,
                );
                self.dispatch_matmul(
                    enc,
                    &*buf_hidden,
                    &*w_up_proj,
                    &*buf_up,
                    1,
                    inter as u32,
                    hidden as u32,
                );
                // SiLU-mul (inline).
                enc.set_compute_pipeline_state(&self.engine.pipelines.silu_mul);
                enc.set_buffer(0, Some(&*buf_gate), 0);
                enc.set_buffer(1, Some(&*buf_up), 0);
                let inter_u32 = inter as u32;
                enc.set_bytes(2, 4, &inter_u32 as *const u32 as *const _);
                enc.dispatch_threads(
                    MTLSize::new(div_ceil(inter as u64, wg) * wg, 1, 1),
                    MTLSize::new(wg, 1, 1),
                );
                self.dispatch_matmul(
                    enc,
                    &*buf_gate,
                    &*w_down_proj,
                    &*buf_ffn_out,
                    1,
                    hidden as u32,
                    inter as u32,
                );

                // MLP residual add.
                self.dispatch_add_and_copy(
                    enc,
                    &*buf_ffn_out,
                    &*buf_residual,
                    &*buf_hidden,
                    hidden as u32,
                );

                // Final RMSNorm.
                self.dispatch_rms_norm(
                    enc,
                    &*buf_hidden,
                    &*w_norm,
                    hidden as u32,
                    1,
                    cfg.rms_norm_eps,
                );

                // Logits GEMV using shared lm_head (embed_tokens_q8).
                self.dispatch_gemm(
                    enc,
                    &*buf_hidden,
                    0,
                    &self.engine.embed_tokens_q8,
                    &*buf_logits,
                    0,
                    1,
                    cfg.vocab_size as u32,
                    hidden as u32,
                );
            }

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // ---- Phase 3: CPU ----
            let logits = unsafe { read_buffer(&*buf_logits, cfg.vocab_size) };
            let draft_token = logits
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);

            // Advance MTP KV cache.
            if let Some(ref mut mtp) = self.session.mtp {
                mtp.cache.seq_len += 1;
            }

            MetalMtpForwardOutput {
                token_id: draft_token,
                logits,
            }
        }

        /// Internal single-token forward step.
        ///
        /// When `capture_hidden` is true (or when `self.session.mtp` is loaded), copies the
        /// pre-final hidden state into `self.session.activations.pre_final_hidden` before the
        /// final RMSNorm, then reads it back to `self.session.last_pre_final_hidden`.
        fn forward_step_inner(
            &mut self,
            token_id: u32,
            position: usize,
            capture_hidden: bool,
        ) -> MetalStepOutput {
            let cfg = self.engine.config.clone();
            let hidden = cfg.hidden_size;

            // --- Per-phase timing (enabled by env LATTICE_PROFILE=1) ---
            let profiling = std::env::var_os("LATTICE_PROFILE").is_some();

            if std::env::var_os("LATTICE_GDN_CPU").is_some() {
                #[cfg(not(debug_assertions))]
                panic!("LATTICE_GDN_CPU=1 is debug-only and cannot be used in release decode");
            }

            let mut prof = StepProfile::default();
            let mut gdn_gpu_dispatches = 0usize;
            let gdn_cpu_dispatches = 0usize;
            let expected_gdn_dispatches = cfg.num_active_linear_attention_layers();
            let step_start = std::time::Instant::now();

            // --- Embedding lookup (CPU, read f16 embed_tokens, write f32 hidden) ---
            let t0 = std::time::Instant::now();
            let embed_offset = token_id as usize * hidden;
            assert!(
                (token_id as usize) < cfg.vocab_size,
                "forward_step: token_id {token_id} >= vocab_size {}",
                cfg.vocab_size
            );
            // SAFETY: embed_tokens is StorageModeShared f16, no GPU in flight;
            // token_id validated < vocab_size above so embed_offset is in bounds.
            unsafe {
                let src = (self.engine.embed_tokens.contents() as *const u16).add(embed_offset);
                let dst = self.session.activations.hidden.contents() as *mut f32;
                for i in 0..hidden {
                    *dst.add(i) = f16_to_f32(*src.add(i));
                }
            }

            if profiling {
                prof.embedding_us += t0.elapsed().as_micros();
            }

            let mut active_layer_idx = 0usize;
            let mut linear_idx = 0usize;
            let mut full_idx = 0usize;

            let kv_dim = cfg.full_kv_dim();

            // Single command buffer for ALL 24 layers — no intermediate waits.
            // Raw pointer breaks the borrow chain: cmd is ref-counted by Metal
            // and outlives the queue borrow. The encode_* methods need &mut self
            // (for session state) but don't touch engine.queue.
            let cmd = unsafe {
                &*(self.engine.queue.new_command_buffer() as *const metal::CommandBufferRef)
            };
            let enc = cmd.new_compute_command_encoder();

            for layer_i in 0..cfg.num_hidden_layers {
                if !cfg.is_layer_active(layer_i) {
                    continue;
                }
                let compact_idx = active_layer_idx;
                active_layer_idx += 1;
                let is_linear = matches!(
                    &self.engine.layer_weights[compact_idx].0,
                    MetalLayerAttnWeights::Linear(_)
                );
                if is_linear {
                    gdn_gpu_dispatches += self.encode_gdn_layer(
                        enc,
                        compact_idx,
                        linear_idx,
                        position,
                        &cfg,
                        &mut prof,
                        profiling,
                    );
                    linear_idx += 1;
                } else {
                    self.encode_gqa_layer(
                        enc,
                        compact_idx,
                        full_idx,
                        position,
                        kv_dim,
                        &cfg,
                        &mut prof,
                        profiling,
                    );
                    full_idx += 1;
                }
            } // end layer loop

            let topk_which =
                self.encode_final_head(enc, &cfg, capture_hidden, &mut prof, profiling);

            // Single submit for entire forward pass + optional top-k.
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            debug_assert_eq!(
                gdn_gpu_dispatches, expected_gdn_dispatches,
                "GDN GPU recurrence dispatch count mismatch"
            );

            if profiling {
                let total_us = step_start.elapsed().as_micros();
                let accounted_us = prof.embedding_us
                    + prof.projection_us
                    + prof.gdn_recurrence_us
                    + prof.gqa_attention_us
                    + prof.mlp_us
                    + prof.final_us;
                let other_us = total_us.saturating_sub(accounted_us);
                eprintln!(
                    "[PROFILE] step {position}: total_us={total_us} \
                     embedding_us={} projection_us={} gdn_recurrence_us={} \
                     gqa_attention_us={} mlp_us={} final_us={} other_us={} \
                     gdn_gpu_dispatches={} expected_gdn_dispatches={} gdn_cpu_dispatches={}",
                    prof.embedding_us,
                    prof.projection_us,
                    prof.gdn_recurrence_us,
                    prof.gqa_attention_us,
                    prof.mlp_us,
                    prof.final_us,
                    other_us,
                    gdn_gpu_dispatches,
                    expected_gdn_dispatches,
                    gdn_cpu_dispatches,
                );
            }

            // Read back pre-final hidden when requested (for MTP input).
            // SAFETY: GPU completed, pre_final_hidden is StorageModeShared.
            let pre_final_hidden = if capture_hidden || self.session.mtp.is_some() {
                let h = unsafe { read_buffer(&self.session.activations.pre_final_hidden, hidden) };
                self.session.last_pre_final_hidden = h.clone();
                h
            } else {
                Vec::new()
            };

            let logits = if let Some(which) = topk_which {
                // Compact path: read k*(f32+u32)=k*8 bytes instead of vocab*4 bytes.
                // SAFETY: GPU completed, buffers are StorageModeShared.
                let k = self.session.compact_topk;
                let candidates = unsafe { self.read_topk_candidates(which, k) };
                self.session.compact_result = candidates;
                vec![] // generate loop reads compact_result, not these logits
            } else {
                // Full path: read vocab_size f32 logits back to host.
                // SAFETY: GPU completed, buffer is StorageModeShared.
                unsafe { read_buffer(&self.session.activations.logits, cfg.vocab_size) }
            };
            self.session.kv_cache.seq_len += 1;

            MetalStepOutput {
                logits,
                pre_final_hidden,
            }
        }

        /// **Unstable**: single-token forward step; kernel dispatch strategy evolving.
        ///
        /// Run a single token through the full model. Returns logits [vocab_size].
        pub fn forward_step(&mut self, token_id: u32, position: usize) -> Vec<f32> {
            self.forward_step_inner(token_id, position, false).logits
        }

        /// **Unstable**: batch prompt prefill; prefill kernel and fallback threshold may change.
        ///
        /// Batch prefill: process all prompt tokens at once using GEMM.
        ///
        /// Returns logits for the LAST token. Updates KV cache and GDN state.
        /// Uses batch GEMM (M=prompt_len) for projections instead of per-token GEMV,
        /// giving ~10-20x speedup on prefill for typical prompt lengths.
        ///
        /// Falls back to sequential forward_step for prompts longer than max_prefill.
        pub fn forward_prefill(&mut self, token_ids: &[u32]) -> Vec<f32> {
            let n = token_ids.len();
            if n == 0 {
                return vec![0.0; self.engine.config.vocab_size];
            }
            if n == 1 {
                let logits = self.forward_step(token_ids[0], 0);
                return logits;
            }
            // Fall back to sequential for long prompts
            if n > self.session.max_prefill {
                let mut last_logits = Vec::new();
                for (pos, &id) in token_ids.iter().enumerate() {
                    last_logits = self.forward_step(id, pos);
                }
                return last_logits;
            }

            assert!(
                n <= self.session.kv_cache.max_cache_len,
                "prefill length {} exceeds max_cache_len {}",
                n,
                self.session.kv_cache.max_cache_len
            );
            let cfg = self.engine.config.clone();
            let m = n as u32;
            let hidden = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            let kv_dim = cfg.full_kv_dim();
            let q_dim = cfg.full_q_dim();
            let num_q_heads = cfg.num_attention_heads;
            let num_kv_heads = cfg.num_key_value_heads;
            let head_dim = cfg.head_dim;
            let half_rope_dim = (cfg.rope_dim() / 2) as u32;

            // Batch embedding: f16 → f32 for all N tokens
            // SAFETY: embed_tokens is StorageModeShared f16, no GPU in flight;
            // each id is validated < vocab_size before computing the offset.
            unsafe {
                let src_base = self.engine.embed_tokens.contents() as *const u16;
                let dst_base = self.session.activations.hidden.contents() as *mut f32;
                for (t, &id) in token_ids.iter().enumerate() {
                    assert!(
                        (id as usize) < cfg.vocab_size,
                        "forward_prefill: token_ids[{t}]={id} >= vocab_size {}",
                        cfg.vocab_size
                    );
                    let src = src_base.add(id as usize * hidden);
                    let dst = dst_base.add(t * hidden);
                    for i in 0..hidden {
                        *dst.add(i) = f16_to_f32(*src.add(i));
                    }
                }
            }

            let mut active_layer_idx = 0usize;
            let mut linear_idx = 0usize;
            let mut full_idx = 0usize;

            // Single command buffer for entire prefill (all 24 layers + final logits)
            let cmd = self.engine.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            for layer_i in 0..cfg.num_hidden_layers {
                if !cfg.is_layer_active(layer_i) {
                    continue;
                }
                let compact_idx = active_layer_idx;
                active_layer_idx += 1;
                let is_linear = matches!(
                    &self.engine.layer_weights[compact_idx].0,
                    MetalLayerAttnWeights::Linear(_)
                );

                // Extract weight pointers (avoids borrow conflict with &mut self)
                let (_, common_w) = &self.engine.layer_weights[compact_idx];
                let w_in_norm = &common_w.input_layernorm as *const Buffer;
                let w_post_norm = &common_w.post_attention_layernorm as *const Buffer;
                let w_gate = &common_w.gate_proj as *const Q4WeightBuf;
                let w_up = &common_w.up_proj as *const Q4WeightBuf;
                let w_down = &common_w.down_proj as *const Q4WeightBuf;

                if is_linear {
                    // ========= GDN layer: batch projections + sequential recurrence =========
                    let (w_qkv, w_z, w_b, w_a, w_alog, w_dtb, w_conv, w_norm, w_out) = {
                        let gdn_w = match &self.engine.layer_weights[compact_idx].0 {
                            MetalLayerAttnWeights::Linear(w) => w,
                            _ => unreachable!(),
                        };
                        (
                            &gdn_w.in_proj_qkv as *const Q4WeightBuf,
                            &gdn_w.in_proj_z as *const Q4WeightBuf,
                            &gdn_w.in_proj_b as *const Buffer,
                            &gdn_w.in_proj_a as *const Buffer,
                            &gdn_w.a_log as *const Buffer,
                            &gdn_w.dt_bias as *const Buffer,
                            &gdn_w.conv1d_weight as *const Buffer,
                            &gdn_w.norm_weight as *const Buffer,
                            &gdn_w.out_proj as *const Q4WeightBuf,
                        )
                    };
                    let qkv_d = cfg.linear_qkv_dim();
                    let out_d = cfg.linear_output_dim();
                    let num_h = cfg.linear_num_key_heads;
                    let key_d = cfg.linear_key_head_dim;
                    let val_d = cfg.linear_value_head_dim;
                    let ks = cfg.linear_conv_kernel_dim as u32;

                    // Batch: copy hidden → residual, norm hidden
                    self.dispatch_copy(
                        enc,
                        &self.session.activations.hidden,
                        &self.session.activations.residual,
                        m * hidden as u32,
                    );
                    // SAFETY: The input norm buffer pointer is live for this command
                    // buffer and row_len/num_rows match the batch activation layout.
                    unsafe {
                        self.dispatch_rms_norm(
                            enc,
                            &self.session.activations.hidden,
                            &*w_in_norm,
                            hidden as u32,
                            m,
                            cfg.rms_norm_eps,
                        );
                    }

                    // Batch: QKV + Z projections (GEMM)
                    // SAFETY: Raw GDN projection pointers are live layer-owned buffers;
                    // GEMM dimensions match [m, hidden] by [qkv_d/out_d, hidden].
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_qkv,
                            &self.session.activations.gdn_qkv,
                            0,
                            m,
                            qkv_d as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_z,
                            &self.session.activations.gdn_z,
                            0,
                            m,
                            out_d as u32,
                            hidden as u32,
                        );
                    }

                    // Sequential: conv1d + recurrence per token (causal dependency)
                    for t in 0..n {
                        let qkv_off = (t * qkv_d) as u64 * 4;
                        let z_off = (t * out_d) as u64 * 4;
                        let h_off = (t * hidden) as u64 * 4;

                        // Conv1d + SiLU
                        // SAFETY: Token offsets are within m-sized activation buffers;
                        // conv/state/weight buffers are live for the command buffer.
                        unsafe {
                            enc.set_compute_pipeline_state(&self.engine.pipelines.conv1d_silu);
                            enc.set_buffer(0, Some(&self.session.gdn_gpu_conv_bufs[linear_idx]), 0);
                            enc.set_buffer(1, Some(&self.session.activations.gdn_qkv), qkv_off);
                            enc.set_buffer(2, Some(&*w_conv), 0);
                            enc.set_buffer(3, Some(&self.session.gdn_gpu_conv_out), 0);
                            let qd = qkv_d as u32;
                            enc.set_bytes(4, 4, &qd as *const u32 as *const _);
                            enc.set_bytes(5, 4, &ks as *const u32 as *const _);
                            let wg = 256u64;
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(qkv_d as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );
                        }

                        // GDN recurrence
                        // SAFETY: Token offsets are within the batch activation buffers;
                        // GdnRecurParams dimensions are derived from the allocation config.
                        unsafe {
                            #[repr(C)]
                            struct GdnRecurParams {
                                key_dim: u32,
                                value_dim: u32,
                                num_key_heads: u32,
                                num_value_heads: u32,
                                hidden_size: u32,
                                q_total: u32,
                                v_offset: u32,
                                scale: f32,
                                eps: f32,
                            }
                            let num_vh = cfg.linear_num_value_heads();
                            let q_total = (num_h * key_d) as u32;
                            let params = GdnRecurParams {
                                key_dim: key_d as u32,
                                value_dim: val_d as u32,
                                num_key_heads: num_h as u32,
                                num_value_heads: num_vh as u32,
                                hidden_size: hidden as u32,
                                q_total,
                                v_offset: q_total * 2,
                                scale: 1.0 / (key_d as f32).sqrt(),
                                eps: cfg.rms_norm_eps,
                            };
                            enc.set_compute_pipeline_state(&self.engine.pipelines.gdn_recurrence);
                            enc.set_buffer(
                                0,
                                Some(&self.session.gdn_gpu_s_matrices[linear_idx]),
                                0,
                            );
                            enc.set_buffer(1, Some(&self.session.gdn_gpu_conv_out), 0);
                            enc.set_buffer(2, Some(&self.session.activations.gdn_z), z_off);
                            enc.set_buffer(3, Some(&self.session.activations.hidden), h_off);
                            enc.set_buffer(4, Some(&*w_b), 0);
                            enc.set_buffer(5, Some(&*w_a), 0);
                            enc.set_buffer(6, Some(&*w_alog), 0);
                            enc.set_buffer(7, Some(&*w_dtb), 0);
                            enc.set_buffer(8, Some(&*w_norm), 0);
                            enc.set_buffer(9, Some(&self.session.activations.gdn_z), z_off);
                            enc.set_bytes(
                                10,
                                std::mem::size_of::<GdnRecurParams>() as u64,
                                &params as *const GdnRecurParams as *const _,
                            );
                            enc.dispatch_thread_groups(
                                MTLSize::new(num_vh as u64, 1, 1),
                                MTLSize::new(32, 4, 1),
                            );
                        }
                    }

                    // Batch: out_proj
                    // SAFETY: The output projection pointer is live for this command
                    // buffer and GEMM dimensions match [m, out_d] by [hidden, out_d].
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.gdn_z,
                            0,
                            &*w_out,
                            &self.session.activations.attn_out,
                            0,
                            m,
                            hidden as u32,
                            out_d as u32,
                        );
                    }

                    // Batch: residual + norm for MLP
                    self.dispatch_add(
                        enc,
                        &self.session.activations.attn_out,
                        &self.session.activations.residual,
                        m * hidden as u32,
                    );
                    self.dispatch_copy(
                        enc,
                        &self.session.activations.residual,
                        &self.session.activations.hidden,
                        m * hidden as u32,
                    );
                    // SAFETY: The post-attention norm pointer is live and hidden/m
                    // match the activation rows in the preallocated buffers.
                    unsafe {
                        self.dispatch_rms_norm(
                            enc,
                            &self.session.activations.hidden,
                            &*w_post_norm,
                            hidden as u32,
                            m,
                            cfg.rms_norm_eps,
                        );
                    }

                    // Batch: MLP (gate + up + silu_mul + down)
                    // SAFETY: Gate/up pointers are live layer-owned buffers; GEMM
                    // dimensions match [m, hidden] by [inter, hidden].
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_gate,
                            &self.session.activations.gate,
                            0,
                            m,
                            inter as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_up,
                            &self.session.activations.up,
                            0,
                            m,
                            inter as u32,
                            hidden as u32,
                        );
                    }
                    self.dispatch_silu_mul(enc, m * inter as u32);
                    // SAFETY: Down projection pointer is live and dimensions match
                    // [m, inter] by [hidden, inter].
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.gate,
                            0,
                            &*w_down,
                            &self.session.activations.ffn_out,
                            0,
                            m,
                            hidden as u32,
                            inter as u32,
                        );
                    }

                    // Batch: end-of-layer residual
                    self.dispatch_add(
                        enc,
                        &self.session.activations.ffn_out,
                        &self.session.activations.residual,
                        m * hidden as u32,
                    );
                    self.dispatch_copy(
                        enc,
                        &self.session.activations.residual,
                        &self.session.activations.hidden,
                        m * hidden as u32,
                    );

                    linear_idx += 1;
                } else {
                    // ========= Full attention layer: batch proj + sequential attention =========
                    let (w_q, w_k, w_v, w_o, w_qn, w_kn) = {
                        let full_w = match &self.engine.layer_weights[compact_idx].0 {
                            MetalLayerAttnWeights::Full(w) => w,
                            _ => unreachable!(),
                        };
                        (
                            &full_w.q_proj as *const Q4WeightBuf,
                            &full_w.k_proj as *const Q4WeightBuf,
                            &full_w.v_proj as *const Q4WeightBuf,
                            &full_w.o_proj as *const Q4WeightBuf,
                            &full_w.q_norm as *const Buffer,
                            &full_w.k_norm as *const Buffer,
                        )
                    };
                    let scale = 1.0f32 / (head_dim as f32).sqrt();

                    // Batch: copy + norm
                    self.dispatch_copy(
                        enc,
                        &self.session.activations.hidden,
                        &self.session.activations.residual,
                        m * hidden as u32,
                    );
                    // SAFETY: The input norm buffer pointer is live for this command
                    // buffer and row_len/num_rows match the batch activation layout.
                    unsafe {
                        self.dispatch_rms_norm(
                            enc,
                            &self.session.activations.hidden,
                            &*w_in_norm,
                            hidden as u32,
                            m,
                            cfg.rms_norm_eps,
                        );
                    }

                    // Batch: Q/K/V projections (GEMM)
                    // SAFETY: Raw attention projection pointers are live layer-owned
                    // buffers; GEMM dimensions match batch hidden/q/kv layouts.
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_q,
                            &self.session.activations.q,
                            0,
                            m,
                            (2 * q_dim) as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_k,
                            &self.session.activations.k,
                            0,
                            m,
                            kv_dim as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_v,
                            &self.session.activations.v,
                            0,
                            m,
                            kv_dim as u32,
                            hidden as u32,
                        );
                    }

                    // Per-token: scatter, norm, RoPE, cache store, attention, gate
                    for t in 0..n {
                        let q_off = (t * 2 * q_dim) as u64 * 4;
                        let qs_off = (t * q_dim) as u64 * 4;
                        let gz_off = (t * q_dim) as u64 * 4;
                        let k_off = (t * kv_dim) as u64 * 4;
                        let v_off = (t * kv_dim) as u64 * 4;
                        let ao_off = (t * q_dim) as u64 * 4;

                        // Scatter Q + gate from interleaved q_proj
                        // SAFETY: Per-token q/gate offsets are within the batch
                        // activation buffers, and q_dim is num_q_heads * head_dim.
                        unsafe {
                            enc.set_compute_pipeline_state(&self.engine.pipelines.scatter_q_gate);
                            enc.set_buffer(0, Some(&self.session.activations.q), q_off);
                            enc.set_buffer(1, Some(&self.session.activations.q_separated), qs_off);
                            enc.set_buffer(2, Some(&self.session.activations.gate_z), gz_off);
                            let nh = num_q_heads as u32;
                            let hd = head_dim as u32;
                            enc.set_bytes(3, 4, &nh as *const u32 as *const _);
                            enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                            let wg = 256u64;
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );
                        }

                        // Per-head RMS norm Q and K
                        // SAFETY: Q/K norm buffers are live and per-token offsets
                        // are within activation buffers sized from the same config.
                        unsafe {
                            enc.set_compute_pipeline_state(
                                &self.engine.pipelines.per_head_rms_norm,
                            );
                            enc.set_buffer(0, Some(&self.session.activations.q_separated), qs_off);
                            enc.set_buffer(1, Some(&*w_qn), 0);
                            let nh = num_q_heads as u32;
                            let hd = head_dim as u32;
                            enc.set_bytes(2, 4, &nh as *const u32 as *const _);
                            enc.set_bytes(3, 4, &hd as *const u32 as *const _);
                            enc.set_bytes(4, 4, &cfg.rms_norm_eps as *const f32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(nh as u64, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );

                            enc.set_buffer(0, Some(&self.session.activations.k), k_off);
                            enc.set_buffer(1, Some(&*w_kn), 0);
                            let nkh = num_kv_heads as u32;
                            enc.set_bytes(2, 4, &nkh as *const u32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(nkh as u64, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                        }

                        // Partial RoPE for Q and K
                        // SAFETY: RoPE table buffers cover max positions and
                        // per-token Q/K offsets are within activation buffers.
                        unsafe {
                            let pos = t as u32;
                            enc.set_compute_pipeline_state(&self.engine.pipelines.partial_rope);
                            enc.set_buffer(0, Some(&self.session.activations.q_separated), qs_off);
                            enc.set_buffer(1, Some(&self.engine.rope_cos), 0);
                            enc.set_buffer(2, Some(&self.engine.rope_sin), 0);
                            let nh = num_q_heads as u32;
                            let hd = head_dim as u32;
                            enc.set_bytes(3, 4, &nh as *const u32 as *const _);
                            enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                            enc.set_bytes(5, 4, &half_rope_dim as *const u32 as *const _);
                            enc.set_bytes(6, 4, &pos as *const u32 as *const _);
                            let wg = 256u64;
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );

                            enc.set_buffer(0, Some(&self.session.activations.k), k_off);
                            let nkh = num_kv_heads as u32;
                            enc.set_bytes(3, 4, &nkh as *const u32 as *const _);
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(kv_dim as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );
                        }

                        // Store K, V to cache
                        {
                            let dst_offset = (t * kv_dim) as u32;
                            enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset);
                            enc.set_buffer(0, Some(&self.session.activations.k), k_off);
                            enc.set_buffer(1, Some(&self.session.kv_cache.k_bufs[full_idx]), 0);
                            let cnt = kv_dim as u32;
                            enc.set_bytes(2, 4, &cnt as *const u32 as *const _);
                            enc.set_bytes(3, 4, &dst_offset as *const u32 as *const _);
                            let wg = 256u64;
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(kv_dim as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );

                            enc.set_buffer(0, Some(&self.session.activations.v), v_off);
                            enc.set_buffer(1, Some(&self.session.kv_cache.v_bufs[full_idx]), 0);
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(kv_dim as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );
                        }

                        // Causal attention: query Q[t] against cache[0..t+1]
                        {
                            let cache_len = (t + 1) as u32;
                            enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attention);
                            enc.set_buffer(0, Some(&self.session.activations.q_separated), qs_off);
                            enc.set_buffer(1, Some(&self.session.kv_cache.k_bufs[full_idx]), 0);
                            enc.set_buffer(2, Some(&self.session.kv_cache.v_bufs[full_idx]), 0);
                            enc.set_buffer(3, Some(&self.session.activations.attn_out), ao_off);
                            enc.set_bytes(4, 4, &cache_len as *const u32 as *const _);
                            let hd = head_dim as u32;
                            let nqh = num_q_heads as u32;
                            let nkh = num_kv_heads as u32;
                            let qd = q_dim as u32;
                            let kvd = kv_dim as u32;
                            enc.set_bytes(5, 4, &hd as *const u32 as *const _);
                            enc.set_bytes(6, 4, &nqh as *const u32 as *const _);
                            enc.set_bytes(7, 4, &nkh as *const u32 as *const _);
                            enc.set_bytes(8, 4, &qd as *const u32 as *const _);
                            enc.set_bytes(9, 4, &kvd as *const u32 as *const _);
                            enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                            enc.dispatch_thread_groups(
                                MTLSize::new(nqh as u64, 1, 1),
                                MTLSize::new(256, 1, 1),
                            );
                        }

                        // Sigmoid gate
                        {
                            let cnt = q_dim as u32;
                            enc.set_compute_pipeline_state(&self.engine.pipelines.sigmoid_gate);
                            enc.set_buffer(0, Some(&self.session.activations.attn_out), ao_off);
                            enc.set_buffer(1, Some(&self.session.activations.gate_z), gz_off);
                            enc.set_bytes(2, 4, &cnt as *const u32 as *const _);
                            let wg = 256u64;
                            enc.dispatch_threads(
                                MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                                MTLSize::new(wg, 1, 1),
                            );
                        }
                    }

                    // Batch: O projection (attn_out[N, q_dim] → ffn_out[N, hidden])
                    // SAFETY: O-projection pointer is live and dimensions match
                    // [m, q_dim] by [hidden, q_dim].
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.attn_out,
                            0,
                            &*w_o,
                            &self.session.activations.ffn_out,
                            0,
                            m,
                            hidden as u32,
                            q_dim as u32,
                        );
                    }

                    // Batch: residual + norm for MLP
                    self.dispatch_add(
                        enc,
                        &self.session.activations.ffn_out,
                        &self.session.activations.residual,
                        m * hidden as u32,
                    );
                    self.dispatch_copy(
                        enc,
                        &self.session.activations.residual,
                        &self.session.activations.hidden,
                        m * hidden as u32,
                    );
                    // SAFETY: The post-attention norm pointer is live and hidden/m
                    // match the activation rows in the preallocated buffers.
                    unsafe {
                        self.dispatch_rms_norm(
                            enc,
                            &self.session.activations.hidden,
                            &*w_post_norm,
                            hidden as u32,
                            m,
                            cfg.rms_norm_eps,
                        );
                    }

                    // Batch: MLP
                    // SAFETY: Gate/up pointers are live layer-owned buffers; GEMM
                    // dimensions match [m, hidden] by [inter, hidden].
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_gate,
                            &self.session.activations.gate,
                            0,
                            m,
                            inter as u32,
                            hidden as u32,
                        );
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.hidden,
                            0,
                            &*w_up,
                            &self.session.activations.up,
                            0,
                            m,
                            inter as u32,
                            hidden as u32,
                        );
                    }
                    self.dispatch_silu_mul(enc, m * inter as u32);
                    // SAFETY: Down projection pointer is live and dimensions match
                    // [m, inter] by [hidden, inter].
                    unsafe {
                        self.dispatch_gemm(
                            enc,
                            &self.session.activations.gate,
                            0,
                            &*w_down,
                            &self.session.activations.ffn_out,
                            0,
                            m,
                            hidden as u32,
                            inter as u32,
                        );
                    }

                    // Batch: end-of-layer residual
                    self.dispatch_add(
                        enc,
                        &self.session.activations.ffn_out,
                        &self.session.activations.residual,
                        m * hidden as u32,
                    );
                    self.dispatch_copy(
                        enc,
                        &self.session.activations.residual,
                        &self.session.activations.hidden,
                        m * hidden as u32,
                    );

                    full_idx += 1;
                }
            }

            // Final: RMS norm + logits for LAST token only
            let last_off = ((n - 1) * hidden) as u64 * 4;
            // Capture pre-final hidden for last token before RMSNorm overwrites it.
            if self.session.mtp.is_some() {
                enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset);
                enc.set_buffer(0, Some(&self.session.activations.hidden), last_off);
                enc.set_buffer(1, Some(&self.session.activations.pre_final_hidden), 0);
                let cnt = hidden as u32;
                let dst_off = 0u32;
                enc.set_bytes(2, 4, &cnt as *const u32 as *const _);
                enc.set_bytes(3, 4, &dst_off as *const u32 as *const _);
                let wg = 256u64;
                enc.dispatch_threads(
                    MTLSize::new(div_ceil(hidden as u64, wg) * wg, 1, 1),
                    MTLSize::new(wg, 1, 1),
                );
            }
            {
                enc.set_compute_pipeline_state(&self.engine.pipelines.rms_norm);
                enc.set_buffer(0, Some(&self.session.activations.hidden), last_off);
                enc.set_buffer(1, Some(&self.engine.final_norm), 0);
                let rl = hidden as u32;
                let nr = 1u32;
                enc.set_bytes(2, 4, &rl as *const u32 as *const _);
                enc.set_bytes(3, 4, &nr as *const u32 as *const _);
                enc.set_bytes(4, 4, &cfg.rms_norm_eps as *const f32 as *const _);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            }
            // Logits: GEMV on last token
            self.dispatch_gemm(
                enc,
                &self.session.activations.hidden,
                last_off,
                &self.engine.embed_tokens_q8,
                &self.session.activations.logits,
                0,
                1,
                cfg.vocab_size as u32,
                hidden as u32,
            );

            // Append top-k in the same encoder when compact mode is active.
            let topk_which = if self.session.compact_topk > 0 {
                let k = self.session.compact_topk as u32;
                Some(self.dispatch_topk_enc(enc, cfg.vocab_size as u32, k))
            } else {
                None
            };

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            if self.session.mtp.is_some() {
                // SAFETY: GPU completed, pre_final_hidden is StorageModeShared.
                self.session.last_pre_final_hidden =
                    unsafe { read_buffer(&self.session.activations.pre_final_hidden, hidden) };
            }
            self.session.kv_cache.seq_len = n;

            if let Some(which) = topk_which {
                // SAFETY: GPU completed, buffers are StorageModeShared.
                let k = self.session.compact_topk;
                let candidates = unsafe { self.read_topk_candidates(which, k) };
                self.session.compact_result = candidates;
                vec![]
            } else {
                // SAFETY: The command buffer has completed and logits is a
                // StorageModeShared buffer sized for vocab_size f32 values.
                unsafe { read_buffer(&self.session.activations.logits, cfg.vocab_size) }
            }
        }

        /// MTP greedy decode loop (LATTICE_MTP=1, greedy only).
        ///
        /// Each round: draft one extra token via MTP, verify 2 tokens with the target
        /// model, accept if the target agrees, roll back and use target's token otherwise.
        fn generate_greedy_mtp(
            &mut self,
            prefill_logits: &[f32],
            prompt_len: usize,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
        ) -> GenerateOutput {
            let cfg = self.engine.config.clone();

            // Greedy argmax from prefill logits.
            let pending_first = prefill_logits
                .iter()
                .copied()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);

            let is_stop = |id: u32| id == cfg.eos_token_id || gen_cfg.stop_token_ids.contains(&id);

            if is_stop(pending_first) {
                return GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                };
            }

            let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
            let mut pending_token = pending_first;
            let mut metrics = MetalMtpDecodeMetrics::default();

            while generated_ids.len() < gen_cfg.max_new_tokens {
                let pos = self.session.kv_cache.seq_len;
                if pos >= self.session.kv_cache.max_cache_len.saturating_sub(2) {
                    // Not enough room for 2 more tokens — flush pending and stop.
                    generated_ids.push(pending_token);
                    break;
                }

                // --- MTP draft phase ---
                let t_mtp = std::time::Instant::now();
                let draft = self.mtp_forward_one(pending_token, pos);
                metrics.mtp_ms += t_mtp.elapsed().as_secs_f64() * 1000.0;
                metrics.mtp_forwards += 1;

                if is_stop(draft.token_id) {
                    // Draft is EOS — flush pending, stop before draft.
                    generated_ids.push(pending_token);
                    break;
                }

                // --- Verify phase (LATTICE_MTP_BATCH=1 selects batch-GEMM verifier) ---
                let use_batch = std::env::var_os("LATTICE_MTP_BATCH").is_some();
                let t_verify = std::time::Instant::now();
                let verify_result = if use_batch {
                    self.verify_tokens_batch_gemm(&[pending_token, draft.token_id], pos)
                } else {
                    self.verify_tokens_batched(&[pending_token, draft.token_id], pos)
                };
                let verify_out = match verify_result {
                    Ok(v) => v,
                    Err(_) => {
                        // Fallback: accept pending, advance normally.
                        self.session.kv_cache.seq_len = pos + 1;
                        generated_ids.push(pending_token);
                        metrics.fallback_tokens += 1;
                        pending_token = draft.token_id;
                        continue;
                    }
                };
                metrics.verify_ms += t_verify.elapsed().as_secs_f64() * 1000.0;
                metrics.verify_calls += 1;

                // Target's prediction for position pos+1 (conditioned on pending_token).
                let target_token1 = verify_out.logits[0]
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);

                if draft.token_id == target_token1 {
                    // Draft accepted: output pending + draft, advance by 2.
                    // Clear batch repair token — rollback won't be called this round.
                    if let Some(ref mut p) = self.session.gdn_checkpoints {
                        p.batch_repair_token = None;
                    }
                    generated_ids.push(pending_token);
                    generated_ids.push(draft.token_id);
                    metrics.accepted_extra_tokens += 1;

                    let next_token = verify_out.logits[1]
                        .iter()
                        .copied()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0);

                    // State is already at pos+2 after verify.
                    if is_stop(next_token) || generated_ids.len() >= gen_cfg.max_new_tokens {
                        break;
                    }
                    pending_token = next_token;
                } else {
                    // Draft rejected: output pending, roll back to pos+1.
                    generated_ids.push(pending_token);
                    let t_rb = std::time::Instant::now();
                    let _ = self.rollback_speculative_state_to(pos + 1);
                    metrics.rollback_ms += t_rb.elapsed().as_secs_f64() * 1000.0;

                    pending_token = target_token1;
                    if is_stop(pending_token) || generated_ids.len() >= gen_cfg.max_new_tokens {
                        break;
                    }
                }
                metrics.rounds += 1;
            }

            // Last update to pre_final_hidden from the verify output.
            self.session.last_pre_final_hidden = {
                let pos = self.session.kv_cache.seq_len;
                if pos > 0 {
                    // Re-run one step to get fresh hidden for any downstream caller.
                    // (verify_tokens_batched already left last_pre_final_hidden current.)
                    self.session.last_pre_final_hidden.clone()
                } else {
                    vec![0.0f32; cfg.hidden_size]
                }
            };

            if std::env::var("LATTICE_MTP_VERBOSE").is_ok() {
                eprintln!(
                    "[MTP] rounds={} mtp_fwd={} verify={} accepted_extra={} rollbacks={} fallbacks={} mtp_ms={:.1} verify_ms={:.1} rb_ms={:.1}",
                    metrics.rounds,
                    metrics.mtp_forwards,
                    metrics.verify_calls,
                    metrics.accepted_extra_tokens,
                    metrics.verify_calls - metrics.accepted_extra_tokens,
                    metrics.fallback_tokens,
                    metrics.mtp_ms,
                    metrics.verify_ms,
                    metrics.rollback_ms,
                );
            }

            let text = decode_tokens(tokenizer, &generated_ids);
            GenerateOutput {
                text,
                token_ids: generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
            }
        }

        /// **Unstable**: generate text from a prompt; sampling parameters and output format may change.
        ///
        /// Generate text from a prompt.
        pub fn generate(
            &mut self,
            prompt: &str,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
        ) -> GenerateOutput {
            let cfg = self.engine.config.clone();

            // Initialize RNG
            let mut rng_state = match gen_cfg.seed {
                Some(s) => {
                    if s == 0 {
                        1
                    } else {
                        s
                    }
                }
                None => {
                    use std::time::SystemTime;
                    let t = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0x12345678_9abcdef0);
                    if t == 0 { 1 } else { t }
                }
            };

            // Tokenize prompt
            let input = tokenizer.tokenize(prompt);
            let prompt_ids: Vec<u32> = input.input_ids[..input.real_length].to_vec();
            let prompt_len = prompt_ids.len();

            if prompt_len == 0 {
                return GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: 0,
                    generated_tokens: 0,
                };
            }

            // Reset state for new generation
            self.reset_state();

            let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
            let mut all_ids = prompt_ids.clone();

            let route = choose_gpu_topk_route(
                gen_cfg.top_k,
                std::env::var("LATTICE_COMPACT_TOPK").is_ok(),
                std::env::var("LATTICE_COMPACT_TOPK_SELECT").is_ok(),
            );
            // Repetition penalty requires full logits — disable compact mode when active.
            let use_compact = route != GpuTopkRoute::CpuFallback
                && (gen_cfg.repetition_penalty == 1.0 || all_ids.is_empty());
            self.session.compact_route = if use_compact {
                route
            } else {
                GpuTopkRoute::CpuFallback
            };
            if use_compact {
                self.session.compact_topk = gen_cfg.top_k;
            }

            // Batch prefill: process all prompt tokens at once (GEMM)
            let prefill_logits = self.forward_prefill(&prompt_ids);

            // MTP greedy path: env-gated, greedy (top_k<=1) only.
            let use_mtp = self.session.mtp.is_some()
                && std::env::var("LATTICE_MTP").is_ok()
                && gen_cfg.top_k <= 1
                && gen_cfg.temperature <= 0.0
                && !use_compact;
            if use_mtp {
                if use_compact {
                    self.session.compact_topk = 0;
                    self.session.compact_route = GpuTopkRoute::CpuFallback;
                }
                return self.generate_greedy_mtp(&prefill_logits, prompt_len, tokenizer, gen_cfg);
            }

            let next_id = if use_compact {
                sample_from_candidates(
                    &self.session.compact_result,
                    gen_cfg,
                    &all_ids,
                    &mut rng_state,
                )
            } else {
                sample_token(&prefill_logits, gen_cfg, &all_ids, &mut rng_state)
            };

            let is_stop = |id: u32| -> bool {
                id == cfg.eos_token_id || gen_cfg.stop_token_ids.contains(&id)
            };

            if is_stop(next_id) {
                if use_compact {
                    self.session.compact_topk = 0;
                    self.session.compact_route = GpuTopkRoute::CpuFallback;
                }
                return GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                };
            }

            generated_ids.push(next_id);
            all_ids.push(next_id);

            // Autoregressive decode
            for _ in 1..gen_cfg.max_new_tokens {
                if self.session.kv_cache.seq_len >= self.session.kv_cache.max_cache_len {
                    break;
                }
                let pos = self.session.kv_cache.seq_len;
                let last_token = *all_ids
                    .last()
                    .expect("invariant: prompt or previous sample populated all_ids");

                let step_logits = self.forward_step(last_token, pos);
                let next_id = if use_compact {
                    sample_from_candidates(
                        &self.session.compact_result,
                        gen_cfg,
                        &all_ids,
                        &mut rng_state,
                    )
                } else {
                    sample_token(&step_logits, gen_cfg, &all_ids, &mut rng_state)
                };

                if is_stop(next_id) {
                    break;
                }

                generated_ids.push(next_id);
                all_ids.push(next_id);
                if self.session.kv_cache.seq_len >= self.session.kv_cache.max_cache_len {
                    break;
                }
            }

            if use_compact {
                self.session.compact_topk = 0;
                self.session.compact_route = GpuTopkRoute::CpuFallback;
            }

            // Detokenize
            let text = decode_tokens(tokenizer, &generated_ids);

            GenerateOutput {
                text,
                token_ids: generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
            }
        }

        /// **Unstable**: reset recurrent state; state buffer layout may change.
        ///
        /// Reset all recurrent state for a new generation.
        pub fn reset_state(&mut self) {
            let cfg = &self.engine.config;
            self.session.gdn_states = (0..cfg.num_active_linear_attention_layers())
                .map(|_| GatedDeltaNetState::new(cfg))
                .collect();
            // Zero GPU GDN buffers
            for buf in &self.session.gdn_gpu_conv_bufs {
                // SAFETY: Each buffer is StorageModeShared, no command buffer is
                // in flight during reset_state, and length() is the allocated size.
                unsafe {
                    let ptr = buf.contents() as *mut u8;
                    std::ptr::write_bytes(ptr, 0, buf.length() as usize);
                }
            }
            for buf in &self.session.gdn_gpu_s_matrices {
                // SAFETY: Each buffer is StorageModeShared, no command buffer is
                // in flight during reset_state, and length() is the allocated size.
                unsafe {
                    let ptr = buf.contents() as *mut u8;
                    std::ptr::write_bytes(ptr, 0, buf.length() as usize);
                }
            }
            self.session.kv_cache.reset();
            if let Some(ref mut mtp) = self.session.mtp {
                mtp.cache.reset();
            }
            if let Some(ref mut pool) = self.session.gdn_checkpoints {
                pool.active_base_seq_len = None;
            }
            self.session.last_pre_final_hidden = vec![0.0f32; self.engine.config.hidden_size];
        }

        // ===================================================================
        // GatedDeltaNet layer: GPU projections + CPU recurrence
        // ===================================================================

        /// Process a single token through a GatedDeltaNet layer (CPU path, debug-only).
        ///
        /// Strategy: GPU handles the 2 large projections (QKV, Z) and
        /// the output projection. CPU handles conv1d + recurrence via unified memory.
        /// Beta/alpha projections are small (16 rows) so done on CPU.
        ///
        /// QKV, Z, and out_proj use f16 weights via `dispatch_matmul_half`.
        /// Beta/alpha projections (in_proj_b, in_proj_a) also use f16 weights
        /// but are read via CPU and converted to f32 for the CPU-side matmul.
        ///
        /// Requires `LATTICE_GDN_CPU=1`. Not available in release builds.
        #[cfg(debug_assertions)]
        fn gdn_layer_step_by_idx(
            &mut self,
            layer_idx: usize,
            state_idx: usize,
            cfg: &Qwen35Config,
        ) {
            let hidden = cfg.hidden_size;
            let num_heads = cfg.linear_num_key_heads;
            let value_heads = cfg.linear_num_value_heads();
            let ratio = value_heads / num_heads;
            let key_dim = cfg.linear_key_head_dim;
            let value_dim = cfg.linear_value_head_dim;
            let qkv_dim = cfg.linear_qkv_dim();
            let output_dim = cfg.linear_output_dim();
            let kernel_size = cfg.linear_conv_kernel_dim;

            // Extract raw pointers to weight buffers, releasing the borrow on
            // self.engine.layer_weights before we need &mut self for dispatch/state.
            // SAFETY: layer_weights is never mutated during forward pass, so the
            // pointers remain valid. We dereference them inside unsafe blocks below.
            let (
                w_in_proj_qkv,
                w_in_proj_z,
                w_in_proj_b,
                w_in_proj_a,
                w_a_log,
                w_dt_bias,
                w_conv1d,
                w_norm,
                w_out_proj,
            ) = {
                let gdn_w = match &self.engine.layer_weights[layer_idx].0 {
                    MetalLayerAttnWeights::Linear(w) => w,
                    _ => unreachable!("gdn_layer_step called on non-linear layer"),
                };
                (
                    &gdn_w.in_proj_qkv as *const Q4WeightBuf,
                    &gdn_w.in_proj_z as *const Q4WeightBuf,
                    &gdn_w.in_proj_b as *const Buffer,
                    &gdn_w.in_proj_a as *const Buffer,
                    &gdn_w.a_log as *const Buffer,
                    &gdn_w.dt_bias as *const Buffer,
                    &gdn_w.conv1d_weight as *const Buffer,
                    &gdn_w.norm_weight as *const Buffer,
                    &gdn_w.out_proj as *const Q4WeightBuf,
                )
            };

            // --- GPU: 2 large projections (f16 weights) ---
            // SAFETY: Weight buffer pointers are valid for the lifetime of self.
            // self.engine.layer_weights is not mutated during forward pass.
            let cmd = self.engine.queue.new_command_buffer();
            {
                let enc = cmd.new_compute_command_encoder();
                // SAFETY: Raw projection pointers are live layer-owned buffers;
                // dispatch dimensions match [qkv_dim/output_dim, hidden].
                unsafe {
                    self.dispatch_matmul(
                        enc,
                        &self.session.activations.hidden,
                        &*w_in_proj_qkv,
                        &self.session.activations.gdn_qkv,
                        1,
                        qkv_dim as u32,
                        hidden as u32,
                    );
                    self.dispatch_matmul(
                        enc,
                        &self.session.activations.hidden,
                        &*w_in_proj_z,
                        &self.session.activations.gdn_z,
                        1,
                        output_dim as u32,
                        hidden as u32,
                    );
                }
                enc.end_encoding();
            }
            cmd.commit();
            cmd.wait_until_completed();

            // --- CPU: read projections from shared memory ---
            // SAFETY: The projection command buffer completed and the buffer is
            // StorageModeShared with qkv_dim f32 values.
            let qkv_proj = unsafe { read_buffer(&self.session.activations.gdn_qkv, qkv_dim) };
            // SAFETY: The projection command buffer completed and the buffer is
            // StorageModeShared with output_dim f32 values.
            let z_proj = unsafe { read_buffer(&self.session.activations.gdn_z, output_dim) };
            // SAFETY: No GPU writes are in flight and hidden is the activation length.
            let hidden_vec = unsafe { read_buffer(&self.session.activations.hidden, hidden) };

            // Read small weight vectors for CPU-side projections
            // a_log, dt_bias: f32 buffers (read directly)
            // in_proj_b, in_proj_a: f16 buffers (read and convert)
            // conv1d_weight, norm_weight: f32 buffers (read directly)
            // SAFETY: Layer parameter buffers are StorageModeShared and sized
            // during initialization from the same model config.
            let a_log = unsafe { read_buffer(&*w_a_log, num_heads) };
            // SAFETY: Layer parameter buffers are StorageModeShared and sized
            // during initialization from the same model config.
            let dt_bias = unsafe { read_buffer(&*w_dt_bias, num_heads) };
            // SAFETY: The f16 projection buffer has num_heads * hidden elements.
            let in_proj_b = unsafe { read_buffer_f16(&*w_in_proj_b, num_heads * hidden) };
            // SAFETY: The f16 projection buffer has num_heads * hidden elements.
            let in_proj_a = unsafe { read_buffer_f16(&*w_in_proj_a, num_heads * hidden) };
            // SAFETY: Conv1d weights are StorageModeShared and sized qkv_dim * kernel_size.
            let conv1d_weight = unsafe { read_buffer(&*w_conv1d, qkv_dim * kernel_size) };
            // SAFETY: Norm weights are StorageModeShared and sized value_dim.
            let norm_weight = unsafe { read_buffer(&*w_norm, value_dim) };

            // Beta projection (small): [num_heads, hidden] @ hidden -> [num_heads]
            let mut beta_proj = vec![0.0f32; num_heads];
            crate::forward::cpu::matmul_bt(
                &hidden_vec,
                &in_proj_b,
                &mut beta_proj,
                1,
                hidden,
                num_heads,
            );
            for b in &mut beta_proj {
                *b = 1.0 / (1.0 + (-*b).exp());
            }

            // Alpha projection (small): [num_heads, hidden] @ hidden -> [num_heads]
            let mut alpha_proj = vec![0.0f32; num_heads];
            crate::forward::cpu::matmul_bt(
                &hidden_vec,
                &in_proj_a,
                &mut alpha_proj,
                1,
                hidden,
                num_heads,
            );

            // --- CPU: conv1d + SiLU ---
            let mut conv_output = vec![0.0f32; qkv_dim];
            conv1d_silu_fused(
                &qkv_proj,
                &mut self.session.gdn_states[state_idx].conv_buffer,
                &conv1d_weight,
                &mut conv_output,
                qkv_dim,
                kernel_size,
            );

            // --- CPU: per-head recurrence ---
            let q_total = num_heads * key_dim;
            let k_total = num_heads * key_dim;
            let v_offset = q_total + k_total;
            let scale = 1.0 / (key_dim as f32).sqrt();
            let mut output_heads = vec![0.0f32; output_dim];

            let state = &mut self.session.gdn_states[state_idx];

            for h in 0..value_heads {
                let kh = h / ratio;
                let q_start = kh * key_dim;
                let k_start = q_total + kh * key_dim;
                let v_start = v_offset + h * value_dim;

                let mut q_head = conv_output[q_start..q_start + key_dim].to_vec();
                let mut k_head = conv_output[k_start..k_start + key_dim].to_vec();
                let v = &conv_output[v_start..v_start + value_dim];

                simd_l2_normalize(&mut q_head);
                simd_l2_normalize(&mut k_head);

                let a = a_log[kh].exp();
                let sp = softplus(alpha_proj[kh] + dt_bias[kh]);
                let g = (-a * sp).exp();

                let s_off = h * key_dim * value_dim;
                let s = &mut state.s_matrices[s_off..s_off + key_dim * value_dim];

                let mut kv_mem = vec![0.0f32; value_dim];
                simd_matvec_transpose(s, &k_head, &mut kv_mem, key_dim, value_dim);

                let beta_h = beta_proj[kh];
                let mut delta = vec![0.0f32; value_dim];
                for j in 0..value_dim {
                    delta[j] = (v[j] - kv_mem[j] * g) * beta_h;
                }

                simd_decay_and_rank1_update(s, &k_head, &delta, g, key_dim, value_dim);

                let out_start = h * value_dim;
                let out_head = &mut output_heads[out_start..out_start + value_dim];
                simd_matvec_transpose(s, &q_head, out_head, key_dim, value_dim);
                for val in out_head.iter_mut() {
                    *val *= scale;
                }
            }

            // --- CPU: gated RMS norm ---
            let gamma = &norm_weight[..value_dim];
            let mut gated_norm_buf = vec![0.0f32; output_dim];
            for h in 0..value_heads {
                let start = h * value_dim;
                let end = start + value_dim;
                simd_gated_rms_norm(
                    &output_heads[start..end],
                    &z_proj[start..end],
                    gamma,
                    &mut gated_norm_buf[start..end],
                    cfg.rms_norm_eps,
                );
            }

            // --- GPU: output projection (f16 weights) ---
            // SAFETY: gdn_z is StorageModeShared, no command buffer is in flight,
            // and gated_norm_buf length matches output_dim.
            unsafe { write_buffer(&self.session.activations.gdn_z, &gated_norm_buf) };

            let cmd = self.engine.queue.new_command_buffer();
            {
                let enc = cmd.new_compute_command_encoder();
                // SAFETY: Raw output projection pointer is live and dimensions
                // match [1, output_dim] by [hidden, output_dim].
                unsafe {
                    self.dispatch_matmul(
                        enc,
                        &self.session.activations.gdn_z,
                        &*w_out_proj,
                        &self.session.activations.attn_out,
                        1,
                        hidden as u32,
                        output_dim as u32,
                    );
                }
                enc.end_encoding();
            }
            cmd.commit();
            cmd.wait_until_completed();
        }

        /// CPU-only portion of GDN layer: conv1d + recurrence + gated norm (debug-only).
        /// Reads QKV/Z projections from GPU shared buffers (written by CMD A),
        /// writes gated norm output to gdn_z buffer (read by CMD B for out_proj).
        /// Requires `LATTICE_GDN_CPU=1`. Not available in release builds.
        #[cfg(debug_assertions)]
        fn gdn_cpu_recurrence(&mut self, layer_idx: usize, state_idx: usize, cfg: &Qwen35Config) {
            let hidden = cfg.hidden_size;
            let num_heads = cfg.linear_num_key_heads;
            let value_heads = cfg.linear_num_value_heads();
            let ratio = value_heads / num_heads;
            let key_dim = cfg.linear_key_head_dim;
            let value_dim = cfg.linear_value_head_dim;
            let qkv_dim = cfg.linear_qkv_dim();
            let output_dim = cfg.linear_output_dim();
            let kernel_size = cfg.linear_conv_kernel_dim;

            let (w_in_proj_b, w_in_proj_a, w_a_log, w_dt_bias, w_conv1d, w_norm) = {
                let gdn_w = match &self.engine.layer_weights[layer_idx].0 {
                    MetalLayerAttnWeights::Linear(w) => w,
                    _ => unreachable!(),
                };
                (
                    &gdn_w.in_proj_b as *const Buffer,
                    &gdn_w.in_proj_a as *const Buffer,
                    &gdn_w.a_log as *const Buffer,
                    &gdn_w.dt_bias as *const Buffer,
                    &gdn_w.conv1d_weight as *const Buffer,
                    &gdn_w.norm_weight as *const Buffer,
                )
            };

            // Read GPU outputs from shared memory
            // SAFETY: The producer command buffer completed and the buffer is
            // StorageModeShared with qkv_dim f32 values.
            let qkv_proj = unsafe { read_buffer(&self.session.activations.gdn_qkv, qkv_dim) };
            // SAFETY: The producer command buffer completed and the buffer is
            // StorageModeShared with output_dim f32 values.
            let z_proj = unsafe { read_buffer(&self.session.activations.gdn_z, output_dim) };
            // SAFETY: No GPU writes are in flight and hidden is the activation length.
            let hidden_vec = unsafe { read_buffer(&self.session.activations.hidden, hidden) };

            // SAFETY: Layer parameter buffers are StorageModeShared and sized
            // during initialization from the same model config.
            let a_log_v = unsafe { read_buffer(&*w_a_log, num_heads) };
            // SAFETY: Layer parameter buffers are StorageModeShared and sized
            // during initialization from the same model config.
            let dt_bias_v = unsafe { read_buffer(&*w_dt_bias, num_heads) };
            // SAFETY: The f16 projection buffer has num_heads * hidden elements.
            let in_proj_b = unsafe { read_buffer_f16(&*w_in_proj_b, num_heads * hidden) };
            // SAFETY: The f16 projection buffer has num_heads * hidden elements.
            let in_proj_a = unsafe { read_buffer_f16(&*w_in_proj_a, num_heads * hidden) };
            // SAFETY: Conv1d weights are StorageModeShared and sized qkv_dim * kernel_size.
            let conv1d_weight = unsafe { read_buffer(&*w_conv1d, qkv_dim * kernel_size) };
            // SAFETY: Norm weights are StorageModeShared and sized value_dim.
            let norm_weight = unsafe { read_buffer(&*w_norm, value_dim) };

            // Beta/alpha projections
            let mut beta_proj = vec![0.0f32; num_heads];
            crate::forward::cpu::matmul_bt(
                &hidden_vec,
                &in_proj_b,
                &mut beta_proj,
                1,
                hidden,
                num_heads,
            );
            for b in &mut beta_proj {
                *b = 1.0 / (1.0 + (-*b).exp());
            }

            let mut alpha_proj = vec![0.0f32; num_heads];
            crate::forward::cpu::matmul_bt(
                &hidden_vec,
                &in_proj_a,
                &mut alpha_proj,
                1,
                hidden,
                num_heads,
            );

            // Conv1d + SiLU
            let mut conv_output = vec![0.0f32; qkv_dim];
            conv1d_silu_fused(
                &qkv_proj,
                &mut self.session.gdn_states[state_idx].conv_buffer,
                &conv1d_weight,
                &mut conv_output,
                qkv_dim,
                kernel_size,
            );

            // Per-head recurrence
            let q_total = num_heads * key_dim;
            let k_total = num_heads * key_dim;
            let v_offset = q_total + k_total;
            let scale = 1.0 / (key_dim as f32).sqrt();
            let mut output_heads = vec![0.0f32; output_dim];
            let state = &mut self.session.gdn_states[state_idx];

            for h in 0..value_heads {
                let kh = h / ratio;
                let q_start = kh * key_dim;
                let k_start = q_total + kh * key_dim;
                let v_start = v_offset + h * value_dim;

                let mut q_head = conv_output[q_start..q_start + key_dim].to_vec();
                let mut k_head = conv_output[k_start..k_start + key_dim].to_vec();
                let v = &conv_output[v_start..v_start + value_dim];

                simd_l2_normalize(&mut q_head);
                simd_l2_normalize(&mut k_head);

                let a = a_log_v[kh].exp();
                let sp = softplus(alpha_proj[kh] + dt_bias_v[kh]);
                let g = (-a * sp).exp();

                let s_off = h * key_dim * value_dim;
                let s = &mut state.s_matrices[s_off..s_off + key_dim * value_dim];

                let mut kv_mem = vec![0.0f32; value_dim];
                simd_matvec_transpose(s, &k_head, &mut kv_mem, key_dim, value_dim);

                let beta_h = beta_proj[kh];
                let mut delta = vec![0.0f32; value_dim];
                for j in 0..value_dim {
                    delta[j] = (v[j] - kv_mem[j] * g) * beta_h;
                }

                simd_decay_and_rank1_update(s, &k_head, &delta, g, key_dim, value_dim);

                let out_start = h * value_dim;
                let out_head = &mut output_heads[out_start..out_start + value_dim];
                simd_matvec_transpose(s, &q_head, out_head, key_dim, value_dim);
                for val in out_head.iter_mut() {
                    *val *= scale;
                }
            }

            // Gated RMS norm
            let gamma = &norm_weight[..value_dim];
            let mut gated_norm_buf = vec![0.0f32; output_dim];
            for h in 0..value_heads {
                let start = h * value_dim;
                let end = start + value_dim;
                simd_gated_rms_norm(
                    &output_heads[start..end],
                    &z_proj[start..end],
                    gamma,
                    &mut gated_norm_buf[start..end],
                    cfg.rms_norm_eps,
                );
            }

            // Write result to gdn_z buffer for CMD B's output projection
            // SAFETY: gdn_z is StorageModeShared, no command buffer is in flight,
            // and gated_norm_buf length matches output_dim.
            unsafe { write_buffer(&self.session.activations.gdn_z, &gated_norm_buf) };
        }

        // ===================================================================
        // Full attention layer: fully on GPU
        // ===================================================================

        fn full_attention_layer_step_by_idx(
            &mut self,
            layer_idx: usize,
            cache_idx: usize,
            position: usize,
            cfg: &Qwen35Config,
        ) -> Result<(), String> {
            let hidden = cfg.hidden_size;
            let q_dim = cfg.full_q_dim();
            let kv_dim = cfg.full_kv_dim();
            let head_dim = cfg.head_dim;
            let num_q_heads = cfg.num_attention_heads;
            let num_kv_heads = cfg.num_key_value_heads;
            let rope_dim = cfg.rope_dim();
            let half_rope_dim = (rope_dim / 2) as u32;

            // Extract weight buffer pointers to avoid holding borrow on layer_weights.
            // SAFETY: layer_weights is never mutated during forward pass.
            let (w_q_proj, w_k_proj, w_v_proj, w_o_proj, w_q_norm, w_k_norm) = {
                let full_w = match &self.engine.layer_weights[layer_idx].0 {
                    MetalLayerAttnWeights::Full(w) => w,
                    _ => unreachable!("full_attention_layer_step called on non-full layer"),
                };
                (
                    &full_w.q_proj as *const Q4WeightBuf,
                    &full_w.k_proj as *const Q4WeightBuf,
                    &full_w.v_proj as *const Q4WeightBuf,
                    &full_w.o_proj as *const Q4WeightBuf,
                    &full_w.q_norm as *const Buffer,
                    &full_w.k_norm as *const Buffer,
                )
            };

            // --- GPU: Q (+ gate), K, V projections (f16) + norm + RoPE ---
            let cmd = self.engine.queue.new_command_buffer();
            {
                let enc = cmd.new_compute_command_encoder();
                // SAFETY: Raw Q/K/V projection pointers are live layer-owned
                // buffers and dispatch dimensions match hidden/q/kv layouts.
                unsafe {
                    self.dispatch_matmul(
                        enc,
                        &self.session.activations.hidden,
                        &*w_q_proj,
                        &self.session.activations.q,
                        1,
                        (2 * q_dim) as u32,
                        hidden as u32,
                    );
                    self.dispatch_matmul(
                        enc,
                        &self.session.activations.hidden,
                        &*w_k_proj,
                        &self.session.activations.k,
                        1,
                        kv_dim as u32,
                        hidden as u32,
                    );
                    self.dispatch_matmul(
                        enc,
                        &self.session.activations.hidden,
                        &*w_v_proj,
                        &self.session.activations.v,
                        1,
                        kv_dim as u32,
                        hidden as u32,
                    );
                }

                self.dispatch_scatter_q_gate(enc, num_q_heads as u32, head_dim as u32);

                // SAFETY: Q/K norm buffer pointers are live and head counts/head_dim
                // match the activation buffers allocated from the same config.
                unsafe {
                    self.dispatch_per_head_rms_norm(
                        enc,
                        &self.session.activations.q_separated,
                        &*w_q_norm,
                        num_q_heads as u32,
                        head_dim as u32,
                        cfg.rms_norm_eps,
                    );
                    self.dispatch_per_head_rms_norm(
                        enc,
                        &self.session.activations.k,
                        &*w_k_norm,
                        num_kv_heads as u32,
                        head_dim as u32,
                        cfg.rms_norm_eps,
                    );
                }

                self.dispatch_partial_rope(
                    enc,
                    &self.session.activations.q_separated,
                    num_q_heads as u32,
                    head_dim as u32,
                    half_rope_dim,
                    position as u32,
                );
                self.dispatch_partial_rope(
                    enc,
                    &self.session.activations.k,
                    num_kv_heads as u32,
                    head_dim as u32,
                    half_rope_dim,
                    position as u32,
                );

                enc.end_encoding();
            }
            cmd.commit();
            cmd.wait_until_completed();

            // --- CPU: append K/V to cache (read from shared memory) ---
            // SAFETY: The Q/K/V command buffer completed and k buffer has kv_dim f32 values.
            let k_vec = unsafe { read_buffer(&self.session.activations.k, kv_dim) };
            // SAFETY: The Q/K/V command buffer completed and v buffer has kv_dim f32 values.
            let v_vec = unsafe { read_buffer(&self.session.activations.v, kv_dim) };
            self.session.kv_cache.append_kv(cache_idx, &k_vec, &v_vec)?;
            let cur_seq_len = self.session.kv_cache.seq_len + 1;

            // --- GPU: decode attention + gating + O projection (f16 weights) ---
            let scale = 1.0 / (head_dim as f32).sqrt();
            let cmd = self.engine.queue.new_command_buffer();
            {
                let enc = cmd.new_compute_command_encoder();
                self.dispatch_decode_attention(
                    enc,
                    &self.session.kv_cache.k_bufs[cache_idx],
                    &self.session.kv_cache.v_bufs[cache_idx],
                    cur_seq_len as u32,
                    head_dim as u32,
                    num_q_heads as u32,
                    num_kv_heads as u32,
                    q_dim as u32,
                    kv_dim as u32,
                    scale,
                );

                self.dispatch_sigmoid_gate(enc, q_dim as u32);

                // SAFETY: The O-projection pointer is live and dimensions match
                // [1, q_dim] by [hidden, q_dim].
                unsafe {
                    self.dispatch_matmul(
                        enc,
                        &self.session.activations.attn_out,
                        &*w_o_proj,
                        &self.session.activations.ffn_out,
                        1,
                        hidden as u32,
                        q_dim as u32,
                    );
                }
                self.dispatch_copy(
                    enc,
                    &self.session.activations.ffn_out,
                    &self.session.activations.attn_out,
                    hidden as u32,
                );

                enc.end_encoding();
            }
            cmd.commit();
            cmd.wait_until_completed();
            Ok(())
        }

        // ===================================================================
        // Layer-encoding helpers (called from forward_step_inner)
        // ===================================================================

        /// Encode all Metal commands for a single GDN (linear-attention) layer.
        ///
        /// All commands are appended to the already-open encoder `enc`; no new
        /// command buffer is created here.  Returns `1` on every call (one GPU
        /// recurrence dispatch issued per GDN layer).
        fn encode_gdn_layer(
            &mut self,
            enc: &ComputeCommandEncoderRef,
            compact_idx: usize,
            linear_idx: usize,
            _position: usize,
            cfg: &Qwen35Config,
            prof: &mut StepProfile,
            profiling: bool,
        ) -> usize {
            let hidden = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            let (
                w_in_proj_qkvz,
                w_in_proj_b,
                w_in_proj_a,
                w_a_log,
                w_dt_bias,
                w_conv1d,
                w_norm,
                w_out_proj,
            ) = {
                let gdn_w = match &self.engine.layer_weights[compact_idx].0 {
                    MetalLayerAttnWeights::Linear(w) => w,
                    _ => unreachable!(),
                };
                (
                    &gdn_w.in_proj_qkvz as *const Q4WeightBuf,
                    &gdn_w.in_proj_b as *const Buffer,
                    &gdn_w.in_proj_a as *const Buffer,
                    &gdn_w.a_log as *const Buffer,
                    &gdn_w.dt_bias as *const Buffer,
                    &gdn_w.conv1d_weight as *const Buffer,
                    &gdn_w.norm_weight as *const Buffer,
                    &gdn_w.out_proj as *const Q4WeightBuf,
                )
            };
            let (_, common_w) = &self.engine.layer_weights[compact_idx];
            let qkv_d = cfg.linear_qkv_dim();
            let out_d = cfg.linear_output_dim();
            let ks = cfg.linear_conv_kernel_dim as u32;
            let num_h = cfg.linear_num_key_heads;
            let key_d = cfg.linear_key_head_dim;
            let val_d = cfg.linear_value_head_dim;

            // Pre-norm: save residual + normalize hidden in one dispatch
            self.dispatch_copy_and_rms_norm(
                enc,
                &self.session.activations.hidden,
                &self.session.activations.residual,
                &common_w.input_layernorm,
                hidden as u32,
                cfg.rms_norm_eps,
            );

            // QKV + Z projections
            // SAFETY: Raw projection buffer pointers were taken from
            // self.engine.layer_weights and remain valid while self is borrowed;
            // H4 fusion: one wider GEMV writes QKV||Z rows to gdn_qkvz (saves 1 dispatch/GDN layer).
            // SAFETY: w_in_proj_qkvz holds contiguous Q4 rows [0..qkv_d) from qkv and
            // [qkv_d..qkv_d+out_d) from z; output gdn_qkvz is (qkv_d+out_d) floats.
            let proj_t0 = profiling.then(|| std::time::Instant::now());
            unsafe {
                self.dispatch_matmul(
                    enc,
                    &self.session.activations.hidden,
                    &*w_in_proj_qkvz,
                    &self.session.activations.gdn_qkvz,
                    1,
                    (qkv_d + out_d) as u32,
                    hidden as u32,
                );
            }
            if let Some(t0) = proj_t0 {
                prof.projection_us += t0.elapsed().as_micros();
            }

            // Conv1d + SiLU (GPU)
            // SAFETY: Encoder commands reference live Metal buffers owned by
            // self/layer weights; qkv_d and kernel_size match activation and
            // weight buffer dimensions allocated during initialization.
            unsafe {
                enc.set_compute_pipeline_state(&self.engine.pipelines.conv1d_silu);
                enc.set_buffer(0, Some(&self.session.gdn_gpu_conv_bufs[linear_idx]), 0);
                enc.set_buffer(1, Some(&self.session.activations.gdn_qkvz), 0); // QKV at offset 0 of fused buffer
                enc.set_buffer(2, Some(&*w_conv1d), 0);
                enc.set_buffer(3, Some(&self.session.gdn_gpu_conv_out), 0);
                let qd = qkv_d as u32;
                enc.set_bytes(4, 4, &qd as *const u32 as *const _);
                enc.set_bytes(5, 4, &ks as *const u32 as *const _);
                let wg = 256u64;
                enc.dispatch_threads(
                    MTLSize::new(div_ceil(qkv_d as u64, wg) * wg, 1, 1),
                    MTLSize::new(wg, 1, 1),
                );
            }

            // GDN recurrence (GPU) — one threadgroup per value head
            // SAFETY: All buffers are StorageModeShared and live for the
            // command buffer; dimensions in GdnRecurParams are derived from
            // the model config used to allocate the GDN state buffers.
            let gdn_t0 = profiling.then(|| std::time::Instant::now());
            unsafe {
                #[repr(C)]
                struct GdnRecurParams {
                    key_dim: u32,
                    value_dim: u32,
                    num_key_heads: u32,
                    num_value_heads: u32,
                    hidden_size: u32,
                    q_total: u32,
                    v_offset: u32,
                    scale: f32,
                    eps: f32,
                }
                let num_vh = cfg.linear_num_value_heads();
                let q_total = (num_h * key_d) as u32;
                let params = GdnRecurParams {
                    key_dim: key_d as u32,
                    value_dim: val_d as u32,
                    num_key_heads: num_h as u32,
                    num_value_heads: num_vh as u32,
                    hidden_size: hidden as u32,
                    q_total,
                    v_offset: q_total * 2,
                    scale: 1.0 / (key_d as f32).sqrt(),
                    eps: cfg.rms_norm_eps,
                };
                let z_byte_off = (qkv_d as u64) * 4; // byte offset to Z portion in gdn_qkvz
                let use_q36 =
                    key_d == 128 && val_d == 128 && hidden == 5120 && num_h == 16 && num_vh == 48;
                // H1+H3 three-kernel path: use when all three sharded kernels compiled.
                let use_h1h3 = use_q36
                    && self.engine.pipelines.gdn_precompute_keys.is_some()
                    && self.engine.pipelines.gdn_recurrence_sharded.is_some()
                    && self.engine.pipelines.gdn_norm_silu.is_some();
                if use_h1h3 {
                    let p_bytes = std::mem::size_of::<GdnRecurParams>() as u64;
                    let p_ptr = &params as *const GdnRecurParams as *const _;

                    // H3: precompute key-head values once — 16 TGs × 128 threads
                    enc.set_compute_pipeline_state(
                        self.engine.pipelines.gdn_precompute_keys.as_ref().unwrap(),
                    );
                    enc.set_buffer(0, Some(&self.session.gdn_gpu_conv_out), 0);
                    enc.set_buffer(1, Some(&self.session.activations.hidden), 0);
                    enc.set_buffer(2, Some(&*w_in_proj_b), 0);
                    enc.set_buffer(3, Some(&*w_in_proj_a), 0);
                    enc.set_buffer(4, Some(&*w_a_log), 0);
                    enc.set_buffer(5, Some(&*w_dt_bias), 0);
                    enc.set_buffer(6, Some(&self.session.activations.gdn_key_scratch), 0);
                    enc.set_bytes(7, p_bytes, p_ptr);
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_h as u64, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );

                    // H1: sharded recurrence — (val_d/4) × num_vh TGs, 32×4 threads
                    enc.set_compute_pipeline_state(
                        self.engine
                            .pipelines
                            .gdn_recurrence_sharded
                            .as_ref()
                            .unwrap(),
                    );
                    enc.set_buffer(0, Some(&self.session.gdn_gpu_s_matrices[linear_idx]), 0);
                    enc.set_buffer(1, Some(&self.session.gdn_gpu_conv_out), 0);
                    enc.set_buffer(2, Some(&self.session.activations.gdn_key_scratch), 0);
                    enc.set_buffer(3, Some(&self.session.activations.gdn_raw_out), 0);
                    enc.set_bytes(4, p_bytes, p_ptr);
                    enc.dispatch_thread_groups(
                        MTLSize::new((val_d as u64).div_ceil(4), num_vh as u64, 1),
                        MTLSize::new(32, 4, 1),
                    );

                    // Norm+SiLU: one TG per value head — 48 TGs × 128 threads
                    enc.set_compute_pipeline_state(
                        self.engine.pipelines.gdn_norm_silu.as_ref().unwrap(),
                    );
                    enc.set_buffer(0, Some(&self.session.activations.gdn_raw_out), 0);
                    enc.set_buffer(1, Some(&self.session.activations.gdn_qkvz), z_byte_off);
                    enc.set_buffer(2, Some(&*w_norm), 0);
                    enc.set_buffer(3, Some(&self.session.activations.gdn_qkvz), z_byte_off);
                    enc.set_bytes(4, p_bytes, p_ptr);
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_vh as u64, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );
                } else {
                    // Fallback: H2+H4 fused kernel or generic
                    let recur_pipe = use_q36
                        .then(|| self.engine.pipelines.gdn_recurrence_q36.as_ref())
                        .flatten()
                        .unwrap_or(&self.engine.pipelines.gdn_recurrence);
                    enc.set_compute_pipeline_state(recur_pipe);
                    enc.set_buffer(0, Some(&self.session.gdn_gpu_s_matrices[linear_idx]), 0);
                    enc.set_buffer(1, Some(&self.session.gdn_gpu_conv_out), 0);
                    enc.set_buffer(2, Some(&self.session.activations.gdn_qkvz), z_byte_off);
                    enc.set_buffer(3, Some(&self.session.activations.hidden), 0);
                    enc.set_buffer(4, Some(&*w_in_proj_b), 0);
                    enc.set_buffer(5, Some(&*w_in_proj_a), 0);
                    enc.set_buffer(6, Some(&*w_a_log), 0);
                    enc.set_buffer(7, Some(&*w_dt_bias), 0);
                    enc.set_buffer(8, Some(&*w_norm), 0);
                    enc.set_buffer(9, Some(&self.session.activations.gdn_qkvz), z_byte_off);
                    enc.set_bytes(
                        10,
                        std::mem::size_of::<GdnRecurParams>() as u64,
                        &params as *const GdnRecurParams as *const _,
                    );
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_vh as u64, 1, 1),
                        MTLSize::new(128, 1, 1),
                    );
                }
            }
            if let Some(t0) = gdn_t0 {
                prof.gdn_recurrence_us += t0.elapsed().as_micros();
            }

            // Output projection — reads norm output from Z portion of fused buffer.
            // SAFETY: w_out_proj and gdn_qkvz are live for this command buffer;
            // z_byte_off points to out_d norm values written by gdn_recurrence.
            let out_proj_t0 = profiling.then(|| std::time::Instant::now());
            unsafe {
                self.dispatch_gemm(
                    enc,
                    &self.session.activations.gdn_qkvz,
                    (qkv_d as u64) * 4,
                    &*w_out_proj,
                    &self.session.activations.attn_out,
                    0,
                    1,
                    hidden as u32,
                    out_d as u32,
                );
            }
            if let Some(t0) = out_proj_t0 {
                prof.projection_us += t0.elapsed().as_micros();
            }

            // MLP: fused residual+add+norm replaces 4 dispatches with 1
            // residual = residual + attn_out; hidden = norm(residual)
            let mlp_t0 = profiling.then(|| std::time::Instant::now());
            self.dispatch_fused_residual_add_norm(
                enc,
                &self.session.activations.residual,
                &self.session.activations.attn_out,
                &self.session.activations.residual,
                &self.session.activations.hidden,
                &common_w.post_attention_layernorm,
                hidden as u32,
                cfg.rms_norm_eps,
            );
            self.dispatch_matmul(
                enc,
                &self.session.activations.hidden,
                &common_w.gate_proj,
                &self.session.activations.gate,
                1,
                inter as u32,
                hidden as u32,
            );
            self.dispatch_matmul(
                enc,
                &self.session.activations.hidden,
                &common_w.up_proj,
                &self.session.activations.up,
                1,
                inter as u32,
                hidden as u32,
            );
            self.dispatch_silu_mul(enc, inter as u32);
            self.dispatch_matmul(
                enc,
                &self.session.activations.gate,
                &common_w.down_proj,
                &self.session.activations.ffn_out,
                1,
                hidden as u32,
                inter as u32,
            );
            // End of layer: residual += ffn_out, hidden = residual (fused)
            self.dispatch_add_and_copy(
                enc,
                &self.session.activations.ffn_out,
                &self.session.activations.residual,
                &self.session.activations.hidden,
                hidden as u32,
            );
            if let Some(t0) = mlp_t0 {
                prof.mlp_us += t0.elapsed().as_micros();
            }

            1 // one GDN GPU dispatch per layer
        }

        /// Encode all Metal commands for a single GQA (full-attention) layer.
        ///
        /// All commands are appended to the already-open encoder `enc`; no new
        /// command buffer is created here.
        #[allow(clippy::too_many_arguments)]
        fn encode_gqa_layer(
            &mut self,
            enc: &ComputeCommandEncoderRef,
            compact_idx: usize,
            full_idx: usize,
            position: usize,
            kv_dim: usize,
            cfg: &Qwen35Config,
            prof: &mut StepProfile,
            profiling: bool,
        ) {
            let hidden = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            // GQA: SINGLE command buffer — pre-norm + projections + KV copy + attention + MLP
            let (w_q_proj, w_k_proj, w_v_proj, w_o_proj, w_q_norm, w_k_norm) = {
                let full_w = match &self.engine.layer_weights[compact_idx].0 {
                    MetalLayerAttnWeights::Full(w) => w,
                    _ => unreachable!(),
                };
                (
                    &full_w.q_proj as *const Q4WeightBuf,
                    &full_w.k_proj as *const Q4WeightBuf,
                    &full_w.v_proj as *const Q4WeightBuf,
                    &full_w.o_proj as *const Q4WeightBuf,
                    &full_w.q_norm as *const Buffer,
                    &full_w.k_norm as *const Buffer,
                )
            };
            let (_, common_w) = &self.engine.layer_weights[compact_idx];
            let q_dim = cfg.full_q_dim();
            let head_dim = cfg.head_dim;
            let num_q_heads = cfg.num_attention_heads;
            let num_kv_heads = cfg.num_key_value_heads;
            let half_rope_dim = (cfg.rope_dim() / 2) as u32;
            assert!(
                self.session.kv_cache.seq_len < self.session.kv_cache.max_cache_len,
                "KV cache overflow: seq_len {} >= max_cache_len {}",
                self.session.kv_cache.seq_len,
                self.session.kv_cache.max_cache_len
            );
            let kv_cache_offset = (self.session.kv_cache.seq_len * kv_dim) as u32;
            let cur_seq_len = (self.session.kv_cache.seq_len + 1) as u32;
            let scale = 1.0 / (head_dim as f32).sqrt();

            // Pre-norm: save residual + normalize hidden in one dispatch
            self.dispatch_copy_and_rms_norm(
                enc,
                &self.session.activations.hidden,
                &self.session.activations.residual,
                &common_w.input_layernorm,
                hidden as u32,
                cfg.rms_norm_eps,
            );

            // Q/K/V projections + scatter + norms + RoPE
            // SAFETY: Raw layer weight pointers were taken from
            // self.engine.layer_weights and remain valid while self is borrowed;
            // dispatch dimensions match the preallocated activation buffers.
            let gqa_proj_t0 = profiling.then(|| std::time::Instant::now());
            unsafe {
                self.dispatch_matmul(
                    enc,
                    &self.session.activations.hidden,
                    &*w_q_proj,
                    &self.session.activations.q,
                    1,
                    (2 * q_dim) as u32,
                    hidden as u32,
                );
                self.dispatch_matmul(
                    enc,
                    &self.session.activations.hidden,
                    &*w_k_proj,
                    &self.session.activations.k,
                    1,
                    kv_dim as u32,
                    hidden as u32,
                );
                self.dispatch_matmul(
                    enc,
                    &self.session.activations.hidden,
                    &*w_v_proj,
                    &self.session.activations.v,
                    1,
                    kv_dim as u32,
                    hidden as u32,
                );
            }
            if let Some(t0) = gqa_proj_t0 {
                prof.projection_us += t0.elapsed().as_micros();
            }
            self.dispatch_scatter_q_gate(enc, num_q_heads as u32, head_dim as u32);
            // SAFETY: Q/K norm buffers are live layer-owned buffers and the
            // head counts/head_dim come from the same config used to size them.
            unsafe {
                self.dispatch_per_head_rms_norm(
                    enc,
                    &self.session.activations.q_separated,
                    &*w_q_norm,
                    num_q_heads as u32,
                    head_dim as u32,
                    cfg.rms_norm_eps,
                );
                self.dispatch_per_head_rms_norm(
                    enc,
                    &self.session.activations.k,
                    &*w_k_norm,
                    num_kv_heads as u32,
                    head_dim as u32,
                    cfg.rms_norm_eps,
                );
            }
            self.dispatch_partial_rope(
                enc,
                &self.session.activations.q_separated,
                num_q_heads as u32,
                head_dim as u32,
                half_rope_dim,
                position as u32,
            );
            self.dispatch_partial_rope(
                enc,
                &self.session.activations.k,
                num_kv_heads as u32,
                head_dim as u32,
                half_rope_dim,
                position as u32,
            );

            // GPU KV cache copy
            self.dispatch_copy_offset(
                enc,
                &self.session.activations.k,
                &self.session.kv_cache.k_bufs[full_idx],
                kv_dim as u32,
                kv_cache_offset,
            );
            self.dispatch_copy_offset(
                enc,
                &self.session.activations.v,
                &self.session.kv_cache.v_bufs[full_idx],
                kv_dim as u32,
                kv_cache_offset,
            );

            // Decode attention + gating + O projection
            let gqa_attn_t0 = profiling.then(|| std::time::Instant::now());
            self.dispatch_decode_attention(
                enc,
                &self.session.kv_cache.k_bufs[full_idx],
                &self.session.kv_cache.v_bufs[full_idx],
                cur_seq_len,
                head_dim as u32,
                num_q_heads as u32,
                num_kv_heads as u32,
                q_dim as u32,
                kv_dim as u32,
                scale,
            );
            self.dispatch_sigmoid_gate(enc, q_dim as u32);
            if let Some(t0) = gqa_attn_t0 {
                prof.gqa_attention_us += t0.elapsed().as_micros();
            }
            // SAFETY: The O-projection buffer pointer is live for the command
            // buffer and dimensions match [hidden, q_dim].
            let gqa_oproj_t0 = profiling.then(|| std::time::Instant::now());
            unsafe {
                self.dispatch_matmul(
                    enc,
                    &self.session.activations.attn_out,
                    &*w_o_proj,
                    &self.session.activations.ffn_out,
                    1,
                    hidden as u32,
                    q_dim as u32,
                );
            }
            self.dispatch_copy(
                enc,
                &self.session.activations.ffn_out,
                &self.session.activations.attn_out,
                hidden as u32,
            );
            if let Some(t0) = gqa_oproj_t0 {
                prof.projection_us += t0.elapsed().as_micros();
            }

            // MLP: fused residual+add+norm replaces 4 dispatches with 1
            let gqa_mlp_t0 = profiling.then(|| std::time::Instant::now());
            self.dispatch_fused_residual_add_norm(
                enc,
                &self.session.activations.residual,
                &self.session.activations.attn_out,
                &self.session.activations.residual,
                &self.session.activations.hidden,
                &common_w.post_attention_layernorm,
                hidden as u32,
                cfg.rms_norm_eps,
            );
            self.dispatch_matmul(
                enc,
                &self.session.activations.hidden,
                &common_w.gate_proj,
                &self.session.activations.gate,
                1,
                inter as u32,
                hidden as u32,
            );
            self.dispatch_matmul(
                enc,
                &self.session.activations.hidden,
                &common_w.up_proj,
                &self.session.activations.up,
                1,
                inter as u32,
                hidden as u32,
            );
            self.dispatch_silu_mul(enc, inter as u32);
            self.dispatch_matmul(
                enc,
                &self.session.activations.gate,
                &common_w.down_proj,
                &self.session.activations.ffn_out,
                1,
                hidden as u32,
                inter as u32,
            );
            // End of layer: residual += ffn_out, hidden = residual (fused)
            self.dispatch_add_and_copy(
                enc,
                &self.session.activations.ffn_out,
                &self.session.activations.residual,
                &self.session.activations.hidden,
                hidden as u32,
            );
            if let Some(t0) = gqa_mlp_t0 {
                prof.mlp_us += t0.elapsed().as_micros();
            }
        }

        /// Encode the final head: optional pre-final hidden capture, RMS norm,
        /// logit GEMV, and optional top-k.
        ///
        /// All commands are appended to the already-open encoder `enc`.
        /// Returns `Some(which)` when compact top-k is active, `None` otherwise.
        fn encode_final_head(
            &mut self,
            enc: &ComputeCommandEncoderRef,
            cfg: &Qwen35Config,
            capture_hidden: bool,
            prof: &mut StepProfile,
            profiling: bool,
        ) -> Option<u8> {
            let hidden = cfg.hidden_size;

            // === Capture pre-final hidden for MTP (before RMSNorm overwrites) ===
            if capture_hidden || self.session.mtp.is_some() {
                self.dispatch_copy(
                    enc,
                    &self.session.activations.hidden,
                    &self.session.activations.pre_final_hidden,
                    hidden as u32,
                );
            }

            // === Final RMS norm + logits (same command buffer) ===
            let final_t0 = profiling.then(|| std::time::Instant::now());
            self.dispatch_rms_norm(
                enc,
                &self.session.activations.hidden,
                &self.engine.final_norm,
                hidden as u32,
                1,
                cfg.rms_norm_eps,
            );
            self.dispatch_matmul(
                enc,
                &self.session.activations.hidden,
                &self.engine.embed_tokens_q8,
                &self.session.activations.logits,
                1,
                cfg.vocab_size as u32,
                hidden as u32,
            );
            if let Some(t0) = final_t0 {
                prof.final_us += t0.elapsed().as_micros();
            }

            // If compact mode, append top-k kernels to the same encoder before commit.
            // This avoids a second command-buffer round-trip (~18 µs overhead).
            if self.session.compact_topk > 0 {
                let k = self.session.compact_topk as u32;
                Some(self.dispatch_topk_enc(enc, cfg.vocab_size as u32, k))
            } else {
                None
            }
        }

        // ===================================================================
        // Dispatch helpers
        // ===================================================================

        /// Dispatch f16-weight matmul: C[M,N] = A[M,K] @ B_half[N,K]^T.
        ///
        /// A is f32 (activations), B is f16 (weights), C is f32 (output).
        /// The MSL kernel loads weight tiles as `half` and widens to `float`
        /// before accumulation, maintaining f32 precision in the dot product.
        /// Dispatch optimized GEMV for M=1 decode (one threadgroup per output element).
        /// Uses gemv_decode_m1: float4/half4 vectorized loads + simdgroup reduction.
        fn dispatch_matmul_half(
            &self,
            enc: &ComputeCommandEncoderRef,
            a: &Buffer,
            b: &Buffer,
            c: &Buffer,
            m: u32,
            n: u32,
            k: u32,
        ) {
            let params = GemmParams {
                m,
                n,
                k,
                lda: k,
                ldb: k,
                ldc: n,
            };
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_decode);
            enc.set_buffer(0, Some(a), 0);
            enc.set_buffer(1, Some(b), 0);
            enc.set_buffer(2, Some(c), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<GemmParams>() as u64,
                &params as *const GemmParams as *const _,
            );
            // One threadgroup per output element N, 256 threads per group.
            enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
        }

        /// Dispatch Q8_0 GEMV for M=1 decode. 2 rows per threadgroup, 128 threads.
        /// Uses int8 × f32 direct multiply + simd_sum — no dequantization step.
        fn dispatch_matmul_q8(
            &self,
            enc: &ComputeCommandEncoderRef,
            x: &Buffer,       // activation [M,K] float (M=1 for decode)
            qw: &Q4WeightBuf, // Q8_0 packed weights [N, K/32 * 34]
            y: &Buffer,       // output [M,N] float
            _m: u32,          // always 1 for decode (ignored)
            n: u32,
            k: u32,
        ) {
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_q8);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(&qw.buffer), 0);
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
            enc.set_bytes(4, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(2) as u64, 1, 1),
                MTLSize::new(32, 4, 1), // 128 threads: 32 lanes × 4 simdgroups
            );
        }

        /// Q8 matmul with batch support and buffer offsets.
        /// M=1 uses GEMV (decode hot path), M>1 uses GEMM (batch prefill).
        fn dispatch_gemm_q8(
            &self,
            enc: &ComputeCommandEncoderRef,
            x: &Buffer,
            x_offset: u64, // byte offset into x
            qw: &Q4WeightBuf,
            y: &Buffer,
            y_offset: u64, // byte offset into y
            m: u32,
            n: u32,
            k: u32,
        ) {
            if m <= 1 {
                // GEMV decode path (M=1)
                enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_q8);
                enc.set_buffer(0, Some(x), x_offset);
                enc.set_buffer(1, Some(&qw.buffer), 0);
                enc.set_buffer(2, Some(y), y_offset);
                enc.set_bytes(3, 4, &n as *const u32 as *const _);
                enc.set_bytes(4, 4, &k as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(n.div_ceil(2) as u64, 1, 1),
                    MTLSize::new(32, 4, 1),
                );
            } else {
                // Batch GEMM (prefill path): NR=2 cols × NM=4 rows per TG
                enc.set_compute_pipeline_state(&self.engine.pipelines.gemm_q8);
                enc.set_buffer(0, Some(x), x_offset);
                enc.set_buffer(1, Some(&qw.buffer), 0);
                enc.set_buffer(2, Some(y), y_offset);
                enc.set_bytes(3, 4, &m as *const u32 as *const _);
                enc.set_bytes(4, 4, &n as *const u32 as *const _);
                enc.set_bytes(5, 4, &k as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(n.div_ceil(2) as u64, m.div_ceil(4) as u64, 1),
                    MTLSize::new(32, 4, 1), // 128 threads: same as GEMV
                );
            }
        }

        fn dispatch_matmul_q4(
            &self,
            enc: &ComputeCommandEncoderRef,
            x: &Buffer,       // activation [1,K] float
            qw: &Q4WeightBuf, // Q4_0 packed weights [N, K/32 * 18]; payload at qw.payload_offset
            y: &Buffer,       // output [1,N] float
            _m: u32,
            n: u32,
            k: u32,
        ) {
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_q4);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(&qw.buffer), qw.payload_offset);
            enc.set_buffer(2, Some(y), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
            enc.set_bytes(4, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(2) as u64, 1, 1),
                MTLSize::new(32, 4, 1),
            );
        }

        fn dispatch_gemm_q4(
            &self,
            enc: &ComputeCommandEncoderRef,
            x: &Buffer,
            x_offset: u64,
            qw: &Q4WeightBuf,
            y: &Buffer,
            y_offset: u64,
            m: u32,
            n: u32,
            k: u32,
        ) {
            if m == 0 || n == 0 {
                return;
            }
            assert!(
                k > 0 && k % 32 == 0,
                "dispatch_gemm_q4 requires K to be non-zero and divisible by 32, got {k}"
            );

            if m == 1 {
                enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_q4);
                enc.set_buffer(0, Some(x), x_offset);
                enc.set_buffer(1, Some(&qw.buffer), qw.payload_offset);
                enc.set_buffer(2, Some(y), y_offset);
                enc.set_bytes(3, 4, &n as *const u32 as *const _);
                enc.set_bytes(4, 4, &k as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(n.div_ceil(2) as u64, 1, 1),
                    MTLSize::new(32, 4, 1),
                );
            } else if let Some(tiled) = self.engine.pipelines.gemm_q4_tiled.as_ref() {
                enc.set_compute_pipeline_state(tiled);
                enc.set_buffer(0, Some(&qw.buffer), qw.payload_offset);
                enc.set_buffer(1, Some(x), x_offset);
                enc.set_buffer(2, Some(y), y_offset);
                enc.set_bytes(3, 4, &m as *const u32 as *const _);
                enc.set_bytes(4, 4, &n as *const u32 as *const _);
                enc.set_bytes(5, 4, &k as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(n.div_ceil(32) as u64, m.div_ceil(64) as u64, 1),
                    MTLSize::new(32, 4, 1),
                );
            } else {
                enc.set_compute_pipeline_state(&self.engine.pipelines.gemm_q4);
                enc.set_buffer(0, Some(&qw.buffer), qw.payload_offset);
                enc.set_buffer(1, Some(x), x_offset);
                enc.set_buffer(2, Some(y), y_offset);
                enc.set_bytes(3, 4, &m as *const u32 as *const _);
                enc.set_bytes(4, 4, &n as *const u32 as *const _);
                enc.set_bytes(5, 4, &k as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(n.div_ceil(2) as u64, m.div_ceil(4) as u64, 1),
                    MTLSize::new(32, 4, 1),
                );
            }
        }

        fn dispatch_matmul(
            &self,
            enc: &ComputeCommandEncoderRef,
            x: &Buffer,
            qw: &Q4WeightBuf,
            y: &Buffer,
            m: u32,
            n: u32,
            k: u32,
        ) {
            match self.engine.quant_format {
                QuantFormat::Q8_0 => self.dispatch_matmul_q8(enc, x, qw, y, m, n, k),
                QuantFormat::Q4_0 => self.dispatch_matmul_q4(enc, x, qw, y, m, n, k),
            }
        }

        fn dispatch_gemm(
            &self,
            enc: &ComputeCommandEncoderRef,
            x: &Buffer,
            x_offset: u64,
            qw: &Q4WeightBuf,
            y: &Buffer,
            y_offset: u64,
            m: u32,
            n: u32,
            k: u32,
        ) {
            match self.engine.quant_format {
                QuantFormat::Q8_0 => {
                    self.dispatch_gemm_q8(enc, x, x_offset, qw, y, y_offset, m, n, k)
                }
                QuantFormat::Q4_0 => {
                    self.dispatch_gemm_q4(enc, x, x_offset, qw, y, y_offset, m, n, k)
                }
            }
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
            enc.set_compute_pipeline_state(&self.engine.pipelines.rms_norm);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(gamma), 0);
            enc.set_bytes(2, 4, &row_len as *const u32 as *const _);
            enc.set_bytes(3, 4, &num_rows as *const u32 as *const _);
            enc.set_bytes(4, 4, &eps as *const f32 as *const _);
            let wg = 256u64;
            enc.dispatch_thread_groups(MTLSize::new(num_rows as u64, 1, 1), MTLSize::new(wg, 1, 1));
        }

        fn dispatch_per_head_rms_norm(
            &self,
            enc: &ComputeCommandEncoderRef,
            x: &Buffer,
            gamma: &Buffer,
            num_heads: u32,
            head_dim: u32,
            eps: f32,
        ) {
            enc.set_compute_pipeline_state(&self.engine.pipelines.per_head_rms_norm);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(gamma), 0);
            enc.set_bytes(2, 4, &num_heads as *const u32 as *const _);
            enc.set_bytes(3, 4, &head_dim as *const u32 as *const _);
            enc.set_bytes(4, 4, &eps as *const f32 as *const _);
            let wg = 256u64;
            enc.dispatch_thread_groups(
                MTLSize::new(num_heads as u64, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
        }

        fn dispatch_partial_rope(
            &self,
            enc: &ComputeCommandEncoderRef,
            x: &Buffer,
            num_heads: u32,
            head_dim: u32,
            half_rope_dim: u32,
            position: u32,
        ) {
            enc.set_compute_pipeline_state(&self.engine.pipelines.partial_rope);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(&self.engine.rope_cos), 0);
            enc.set_buffer(2, Some(&self.engine.rope_sin), 0);
            enc.set_bytes(3, 4, &num_heads as *const u32 as *const _);
            enc.set_bytes(4, 4, &head_dim as *const u32 as *const _);
            enc.set_bytes(5, 4, &half_rope_dim as *const u32 as *const _);
            enc.set_bytes(6, 4, &position as *const u32 as *const _);
            let total_pairs = num_heads * half_rope_dim;
            let wg = 256u64;
            enc.dispatch_threads(
                MTLSize::new(div_ceil(total_pairs as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
        }

        #[allow(clippy::too_many_arguments)]
        fn dispatch_decode_attention(
            &self,
            enc: &ComputeCommandEncoderRef,
            k_cache: &Buffer,
            v_cache: &Buffer,
            cache_len: u32,
            head_dim: u32,
            num_q_heads: u32,
            num_kv_heads: u32,
            q_dim: u32,
            kv_dim: u32,
            scale: f32,
        ) {
            const PARTITION_TOKENS: u32 = METAL_FLASH_PARTITION_TOKENS as u32;
            const DIRECT_THRESHOLD: u32 = 512; // use direct path for short caches

            validate_flash_decode_shape(
                head_dim as usize,
                num_q_heads as usize,
                num_kv_heads as usize,
                q_dim as usize,
                kv_dim as usize,
            )
            .expect("invalid Metal FlashAttention decode shape");

            // Common buffer setup (same layout for both direct and partial kernels)
            let set_common_bufs = |enc: &ComputeCommandEncoderRef, out_or_partials: &Buffer| {
                enc.set_buffer(0, Some(&self.session.activations.q_separated), 0);
                enc.set_buffer(1, Some(k_cache), 0);
                enc.set_buffer(2, Some(v_cache), 0);
                enc.set_buffer(3, Some(out_or_partials), 0);
                enc.set_bytes(4, 4, &cache_len as *const u32 as *const _);
                enc.set_bytes(5, 4, &head_dim as *const u32 as *const _);
                enc.set_bytes(6, 4, &num_q_heads as *const u32 as *const _);
                enc.set_bytes(7, 4, &num_kv_heads as *const u32 as *const _);
                enc.set_bytes(8, 4, &q_dim as *const u32 as *const _);
                enc.set_bytes(9, 4, &kv_dim as *const u32 as *const _);
                enc.set_bytes(10, 4, &scale as *const f32 as *const _);
            };

            if cache_len <= DIRECT_THRESHOLD {
                // Direct grouped flash decode: one threadgroup per KV head (H1+H2+H4+H5)
                enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attention);
                set_common_bufs(enc, &self.session.activations.attn_out);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_kv_heads as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            } else {
                // Partitioned flash decode (H3): partial kernel + reduce kernel.
                // Split KV cache into PARTITION_TOKENS-token chunks for better occupancy.
                let num_partitions = cache_len.div_ceil(PARTITION_TOKENS);

                // Partial pass: one TG per (KV head, partition)
                enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attn_partial);
                set_common_bufs(enc, &self.session.activations.attn_partials);
                enc.set_bytes(11, 4, &PARTITION_TOKENS as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_kv_heads as u64, num_partitions as u64, 1),
                    MTLSize::new(256, 1, 1),
                );

                // Reduce pass: one TG per KV head, combines all partitions
                enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attn_reduce);
                enc.set_buffer(0, Some(&self.session.activations.attn_partials), 0);
                enc.set_buffer(1, Some(&self.session.activations.attn_out), 0);
                enc.set_bytes(2, 4, &num_q_heads as *const u32 as *const _);
                enc.set_bytes(3, 4, &num_kv_heads as *const u32 as *const _);
                enc.set_bytes(4, 4, &num_partitions as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_kv_heads as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
        }

        fn dispatch_sigmoid_gate(&self, enc: &ComputeCommandEncoderRef, count: u32) {
            enc.set_compute_pipeline_state(&self.engine.pipelines.sigmoid_gate);
            enc.set_buffer(0, Some(&self.session.activations.attn_out), 0);
            enc.set_buffer(1, Some(&self.session.activations.gate_z), 0);
            enc.set_bytes(2, 4, &count as *const u32 as *const _);
            let wg = 256u64;
            enc.dispatch_threads(
                MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
        }

        fn dispatch_scatter_q_gate(
            &self,
            enc: &ComputeCommandEncoderRef,
            num_heads: u32,
            head_dim: u32,
        ) {
            enc.set_compute_pipeline_state(&self.engine.pipelines.scatter_q_gate);
            enc.set_buffer(0, Some(&self.session.activations.q), 0);
            enc.set_buffer(1, Some(&self.session.activations.q_separated), 0);
            enc.set_buffer(2, Some(&self.session.activations.gate_z), 0);
            enc.set_bytes(3, 4, &num_heads as *const u32 as *const _);
            enc.set_bytes(4, 4, &head_dim as *const u32 as *const _);
            let total = num_heads * head_dim;
            let wg = 256u64;
            enc.dispatch_threads(
                MTLSize::new(div_ceil(total as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
        }

        fn dispatch_silu_mul(&self, enc: &ComputeCommandEncoderRef, count: u32) {
            enc.set_compute_pipeline_state(&self.engine.pipelines.silu_mul);
            enc.set_buffer(0, Some(&self.session.activations.gate), 0);
            enc.set_buffer(1, Some(&self.session.activations.up), 0);
            enc.set_bytes(2, 4, &count as *const u32 as *const _);
            let wg = 256u64;
            enc.dispatch_threads(
                MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
        }

        fn dispatch_copy(
            &self,
            enc: &ComputeCommandEncoderRef,
            src: &Buffer,
            dst: &Buffer,
            count: u32,
        ) {
            enc.set_compute_pipeline_state(&self.engine.pipelines.copy);
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst), 0);
            enc.set_bytes(2, 4, &count as *const u32 as *const _);
            let wg = 256u64;
            enc.dispatch_threads(
                MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
        }

        fn dispatch_copy_offset(
            &self,
            enc: &ComputeCommandEncoderRef,
            src: &Buffer,
            dst: &Buffer,
            count: u32,
            dst_offset: u32,
        ) {
            enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset);
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst), 0);
            enc.set_bytes(2, 4, &count as *const u32 as *const _);
            enc.set_bytes(3, 4, &dst_offset as *const u32 as *const _);
            let wg = 256u64;
            enc.dispatch_threads(
                MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
        }

        fn dispatch_add(
            &self,
            enc: &ComputeCommandEncoderRef,
            src: &Buffer,
            dst: &Buffer,
            count: u32,
        ) {
            enc.set_compute_pipeline_state(&self.engine.pipelines.add);
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst), 0);
            enc.set_bytes(2, 4, &count as *const u32 as *const _);
            let wg = 256u64;
            enc.dispatch_threads(
                MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
        }

        /// Fused residual add + RMS norm.
        /// residual_out = base + delta; normed_out = rms_norm(residual_out) * (1+gamma)
        /// Replaces 4 dispatches (copy+add+copy+rms_norm) with 1.
        fn dispatch_fused_residual_add_norm(
            &self,
            enc: &ComputeCommandEncoderRef,
            base: &Buffer,
            delta: &Buffer,
            residual_out: &Buffer,
            normed_out: &Buffer,
            gamma: &Buffer,
            row_len: u32,
            eps: f32,
        ) {
            enc.set_compute_pipeline_state(&self.engine.pipelines.fused_residual_add_norm);
            enc.set_buffer(0, Some(base), 0);
            enc.set_buffer(1, Some(delta), 0);
            enc.set_buffer(2, Some(residual_out), 0);
            enc.set_buffer(3, Some(normed_out), 0);
            enc.set_buffer(4, Some(gamma), 0);
            enc.set_bytes(5, 4, &row_len as *const u32 as *const _);
            enc.set_bytes(6, 4, &eps as *const f32 as *const _);
            let wg = 256u64;
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(wg, 1, 1));
        }

        /// Fused copy-then-RMS-norm for decode pre-norm: saves `src` to `residual_out`
        /// and normalizes `src` in-place.  Replaces `dispatch_copy` + `dispatch_rms_norm`.
        fn dispatch_copy_and_rms_norm(
            &self,
            enc: &ComputeCommandEncoderRef,
            src: &Buffer,
            residual_out: &Buffer,
            gamma: &Buffer,
            row_len: u32,
            eps: f32,
        ) {
            enc.set_compute_pipeline_state(&self.engine.pipelines.copy_and_rms_norm);
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(residual_out), 0);
            enc.set_buffer(2, Some(gamma), 0);
            enc.set_bytes(3, 4, &row_len as *const u32 as *const _);
            enc.set_bytes(4, 4, &eps as *const f32 as *const _);
            let wg = 256u64;
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(wg, 1, 1));
        }

        /// Fused add-into-residual + copy-to-hidden for decode end-of-layer.
        /// `residual[i] += src[i]; dst[i] = residual[i]`.
        /// Replaces `dispatch_add` + `dispatch_copy`.
        fn dispatch_add_and_copy(
            &self,
            enc: &ComputeCommandEncoderRef,
            src: &Buffer,
            residual: &Buffer,
            dst: &Buffer,
            count: u32,
        ) {
            enc.set_compute_pipeline_state(&self.engine.pipelines.add_and_copy);
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(residual), 0);
            enc.set_buffer(2, Some(dst), 0);
            enc.set_bytes(3, 4, &count as *const u32 as *const _);
            let wg = 256u64;
            enc.dispatch_threads(
                MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
        }

        // -----------------------------------------------------------------------
        // GPU Top-K dispatch helpers
        // -----------------------------------------------------------------------

        /// Dispatch top-k kernels into `enc` (same command buffer as the logits GEMV).
        ///
        /// Runs first-pass + iterative merge passes.  All dispatches are in the
        /// same encoder so the GPU executes them in order without extra synchronisation.
        ///
        /// Returns 0 if the final result is in `topk_scratch_a`, 1 for `topk_scratch_b`.
        /// The caller reads from that buffer after `wait_until_completed()`.
        fn dispatch_topk_enc(&self, enc: &ComputeCommandEncoderRef, vocab_size: u32, k: u32) -> u8 {
            // compact_route must be set before dispatch; CpuFallback means compact_topk=0.
            debug_assert!(
                self.session.compact_route != GpuTopkRoute::CpuFallback,
                "dispatch_topk_enc called with CpuFallback route"
            );
            if k == 1 {
                // Dedicated argmax: two passes, no sorting.
                let groups = vocab_size.div_ceil(1024);
                enc.set_compute_pipeline_state(&self.engine.pipelines.argmax_first);
                enc.set_buffer(0, Some(&self.session.activations.logits), 0);
                enc.set_buffer(1, Some(&self.session.activations.topk_scratch_a), 0);
                enc.set_bytes(2, 4, &vocab_size as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(groups as u64, 1, 1),
                    MTLSize::new(1024, 1, 1),
                );

                enc.set_compute_pipeline_state(&self.engine.pipelines.argmax_merge);
                enc.set_buffer(0, Some(&self.session.activations.topk_scratch_a), 0);
                enc.set_buffer(1, Some(&self.session.activations.topk_scratch_b), 0);
                enc.set_bytes(2, 4, &groups as *const u32 as *const _);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));

                return 1; // result in scratch_b[0]
            }

            // k > 1: hierarchical k=50 SIMD-group tournament (no bitonic sort).
            debug_assert_eq!(k, 50, "only HierarchicalK50 route is supported for k>1");
            debug_assert_eq!(self.session.compact_route, GpuTopkRoute::HierarchicalK50);

            let tile = 1024u32;
            let first_pass_groups = vocab_size.div_ceil(tile);

            enc.set_compute_pipeline_state(&self.engine.pipelines.topk_select50_first);
            enc.set_buffer(0, Some(&self.session.activations.logits), 0);
            enc.set_buffer(1, Some(&self.session.activations.topk_scratch_a), 0);
            enc.set_bytes(2, 4, &vocab_size as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(first_pass_groups as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );

            let mut current_groups = first_pass_groups;
            let mut which: u8 = 0;

            while current_groups > 1 {
                let fan_in: u32 = 16u32.min(current_groups);
                let out_groups = current_groups.div_ceil(fan_in);

                let (in_buf, out_buf) = if which == 0 {
                    (
                        &self.session.activations.topk_scratch_a,
                        &self.session.activations.topk_scratch_b,
                    )
                } else {
                    (
                        &self.session.activations.topk_scratch_b,
                        &self.session.activations.topk_scratch_a,
                    )
                };
                enc.set_compute_pipeline_state(&self.engine.pipelines.topk_select50_merge);
                enc.set_buffer(0, Some(in_buf), 0);
                enc.set_buffer(1, Some(out_buf), 0);
                enc.set_bytes(2, 4, &current_groups as *const u32 as *const _);
                enc.set_bytes(3, 4, &fan_in as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(out_groups as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );

                current_groups = out_groups;
                which = 1 - which;
            }

            which
        }

        /// Read top-k compact candidates from the appropriate scratch buffer.
        ///
        /// SAFETY: GPU command buffer must have completed before calling.
        unsafe fn read_topk_candidates(
            &self,
            which: u8,
            k: usize,
        ) -> Vec<crate::sampling::Candidate> {
            // GPU layout: struct TopKCandidate { float logit; uint token_id; }
            #[repr(C)]
            #[derive(Copy, Clone)]
            struct GpuCandidate {
                logit: f32,
                token_id: u32,
            }
            let buf = if which == 0 {
                &self.session.activations.topk_scratch_a
            } else {
                &self.session.activations.topk_scratch_b
            };
            let ptr = buf.contents() as *const GpuCandidate;
            let mut out = Vec::with_capacity(k);
            for i in 0..k {
                let gc = *ptr.add(i);
                out.push(crate::sampling::Candidate {
                    token_id: gc.token_id,
                    logit: gc.logit,
                });
            }
            out
        }

        // ===================================================================
        // Layer importance scoring
        // ===================================================================

        /// Run a prefix-masked forward pass capturing the last-token hidden state
        /// before and after each active layer for importance scoring.
        ///
        /// For N active layers, performs N sequential model passes where pass `i`
        /// activates only layers `[active_0 .. active_i]`.  Each pass processes all
        /// tokens in `token_ids` via `forward_step_inner`.  Because preceding layers
        /// are identical across passes, `output[i]` is the exact post-layer-i hidden
        /// state from the full-model run.
        ///
        /// Intended for offline calibration only.
        /// Cost: O(N²) in active layers — performs N passes of increasing length
        /// (pass i activates layers 0..=i, total layer-token work ≈ N²/2).
        /// Acceptable for calibration-only usage where N ≤ 64.
        fn forward_prefill_layer_traces_last_token(
            &mut self,
            token_ids: &[u32],
        ) -> Result<Vec<LayerTrace>, crate::error::InferenceError> {
            if token_ids.is_empty() {
                return Err(crate::error::InferenceError::Inference(
                    "forward_prefill_layer_traces_last_token: empty token sequence".to_string(),
                ));
            }
            let orig_cfg = self.engine.config.clone();
            let hidden = orig_cfg.hidden_size;
            let n = token_ids.len();

            let active_indices: Vec<usize> = (0..orig_cfg.num_hidden_layers)
                .filter(|&i| orig_cfg.is_layer_active(i))
                .collect();

            // Input to the first active layer = embedding of the last token.
            let last_id = *token_ids.last().unwrap();
            assert!(
                (last_id as usize) < orig_cfg.vocab_size,
                "token_id {last_id} >= vocab_size {}",
                orig_cfg.vocab_size
            );
            // SAFETY: embed_tokens is StorageModeShared f16, no GPU command in flight.
            let first_input: Vec<f32> = unsafe {
                let src = (self.engine.embed_tokens.contents() as *const u16)
                    .add(last_id as usize * hidden);
                (0..hidden).map(|i| f16_to_f32(*src.add(i))).collect()
            };

            let mut traces: Vec<LayerTrace> = Vec::with_capacity(active_indices.len());
            let mut prev_output: Vec<f32> = first_input;

            for (active_pos, &layer_i) in active_indices.iter().enumerate() {
                // Mask: activate only layers active_indices[0..=active_pos].
                let mut trace_mask = vec![false; orig_cfg.num_hidden_layers];
                for j in 0..=active_pos {
                    trace_mask[active_indices[j]] = true;
                }
                self.engine.config.layer_mask = trace_mask;
                self.reset_state();

                // Process all tokens; capture pre-final hidden for the last one.
                for (pos, &id) in token_ids.iter().enumerate() {
                    let is_last = pos == n - 1;
                    let _ = self.forward_step_inner(id, pos, is_last);
                }

                // last_pre_final_hidden = post-last-active-layer hidden for the last token.
                let output_hidden = self.session.last_pre_final_hidden.clone();
                let input_hidden = prev_output.clone();
                prev_output = output_hidden.clone();

                traces.push(LayerTrace {
                    layer_idx: layer_i,
                    input: input_hidden,
                    output: output_hidden,
                });
            }

            // Restore original config and leave model in a clean state.
            self.engine.config.layer_mask = orig_cfg.layer_mask;
            self.reset_state();

            Ok(traces)
        }

        /// Score each active layer's importance by measuring how much it changes the
        /// last-token hidden state across `calibration_prompts`.
        ///
        /// Layers with high mean cosine similarity change the hidden state least and
        /// are the safest prune candidates.  Returns a [`LayerPruningPlan`] containing
        /// per-layer scores (sorted highest cosine first), a type-balanced recommended
        /// mask for `prune_layers` removals, and an optional Gromov warning when
        /// `prune_layers` exceeds 20 % of the total layer count.
        pub fn score_layer_importance(
            &mut self,
            calibration_prompts: &[Vec<u32>],
            prune_layers: usize,
        ) -> Result<LayerPruningPlan, crate::error::InferenceError> {
            use crate::model::qwen35_config::LayerType;
            let cfg = self.engine.config.clone();

            let mut sums = vec![0.0f64; cfg.num_hidden_layers];
            let mut counts = vec![0usize; cfg.num_hidden_layers];

            for prompt in calibration_prompts {
                let traces = self.forward_prefill_layer_traces_last_token(prompt)?;
                for trace in &traces {
                    let cos = cosine_similarity(&trace.input, &trace.output);
                    sums[trace.layer_idx] += cos as f64;
                    counts[trace.layer_idx] += 1;
                }
            }

            let mut scores: Vec<LayerImportanceScore> = (0..cfg.num_hidden_layers)
                .filter(|&i| cfg.is_layer_active(i))
                .map(|i| {
                    let mean_cosine = if counts[i] == 0 {
                        0.0
                    } else {
                        (sums[i] / counts[i] as f64) as f32
                    };
                    LayerImportanceScore {
                        layer_idx: i,
                        layer_type: cfg.layer_types[i],
                        mean_cosine,
                        importance: 1.0 - mean_cosine,
                    }
                })
                .collect();

            // Highest cosine first = lowest importance = best prune candidate.
            scores.sort_by(|a, b| b.mean_cosine.total_cmp(&a.mean_cosine));

            // Type-balanced quota: ~1 GQA removed per 3 GDN.
            let gqa_quota = (prune_layers + 2) / 4;
            let gdn_quota = prune_layers.saturating_sub(gqa_quota);
            let mut gdn_left = gdn_quota;
            let mut gqa_left = gqa_quota;
            let mut recommended_mask = cfg.layer_mask.clone();

            for score in &scores {
                match score.layer_type {
                    LayerType::LinearAttention if gdn_left > 0 => {
                        recommended_mask[score.layer_idx] = false;
                        gdn_left -= 1;
                    }
                    LayerType::FullAttention if gqa_left > 0 => {
                        recommended_mask[score.layer_idx] = false;
                        gqa_left -= 1;
                    }
                    _ => {}
                }
                if gdn_left == 0 && gqa_left == 0 {
                    break;
                }
            }

            let warning = if prune_layers * 5 > cfg.num_hidden_layers {
                Some(format!(
                    "Gromov warning: requested prune_layers={prune_layers} exceeds 20% of {} layers",
                    cfg.num_hidden_layers
                ))
            } else {
                None
            };

            Ok(LayerPruningPlan {
                scores,
                recommended_mask,
                warning,
            })
        }
    }

    // -----------------------------------------------------------------------
    // Compact sampling helper (replaces private sample_token)
    // -----------------------------------------------------------------------

    /// Sample from a compact candidate set returned by GPU top-k.
    ///
    /// The candidates have already been selected by the GPU (no full-vocab
    /// scan needed).  Repetition penalty is applied CPU-side to the compact
    /// set as a round-0 approximation (accurate when recently-repeated tokens
    /// are not ranked far below the top-k boundary).
    fn sample_from_candidates(
        candidates: &[crate::sampling::Candidate],
        cfg: &GenerateConfig,
        previous_ids: &[u32],
        rng_state: &mut u64,
    ) -> u32 {
        use crate::sampling::CandidateSet;
        let mut cs = CandidateSet::from_candidates(candidates.to_vec());

        if cfg.repetition_penalty != 1.0 {
            cs.apply_repetition_penalty(previous_ids, cfg.repetition_penalty);
        }
        if cfg.temperature <= 0.0 {
            return cs.argmax();
        }
        cs.apply_temperature(cfg.temperature);

        let raw = xorshift64(rng_state);
        let r = (raw as f64 / u64::MAX as f64) as f32;
        cs.sample_top_p(cfg.top_p, r)
    }

    fn softplus(x: f32) -> f32 {
        if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
    }

    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// Full-logit sampling fallback (used when top_k == 0 or top_k > MAX_TOP_K).
    /// Implements the same pipeline as sample_from_candidates but builds the
    /// CandidateSet from raw logits.
    fn sample_token(
        logits: &[f32],
        cfg: &GenerateConfig,
        previous_ids: &[u32],
        rng_state: &mut u64,
    ) -> u32 {
        use crate::sampling::CandidateSet;
        let mut cs = CandidateSet::from_full_logits(logits);
        cs.apply_repetition_penalty(previous_ids, cfg.repetition_penalty);
        if cfg.temperature <= 0.0 {
            return cs.argmax();
        }
        cs.apply_temperature(cfg.temperature);
        cs.retain_top_k(cfg.top_k);
        let r = (xorshift64(rng_state) as f64 / u64::MAX as f64) as f32;
        cs.sample_top_p(cfg.top_p, r)
    }

    fn decode_tokens(tokenizer: &BpeTokenizer, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|id| tokenizer.token_for_id(*id))
            .map(crate::tokenizer::bpe::byte_decode_token)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Chat Completion API
    // -----------------------------------------------------------------------

    /// **Unstable**: chat conversation role; variants may extend for tool/function roles.
    ///
    /// Role in a chat conversation.
    #[derive(Debug, Clone, PartialEq)]
    pub enum ChatRole {
        System,
        User,
        Assistant,
    }

    impl ChatRole {
        fn as_str(&self) -> &str {
            match self {
                ChatRole::System => "system",
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
            }
        }
    }

    /// **Unstable**: single chat message; fields may expand with tool call support.
    ///
    /// A single message in a chat conversation.
    #[derive(Debug, Clone)]
    pub struct ChatMessage {
        pub role: ChatRole,
        pub content: String,
    }

    impl ChatMessage {
        /// **Unstable**: construct a system message.
        pub fn system(content: impl Into<String>) -> Self {
            Self {
                role: ChatRole::System,
                content: content.into(),
            }
        }
        /// **Unstable**: construct a user message.
        pub fn user(content: impl Into<String>) -> Self {
            Self {
                role: ChatRole::User,
                content: content.into(),
            }
        }
        /// **Unstable**: construct an assistant message.
        pub fn assistant(content: impl Into<String>) -> Self {
            Self {
                role: ChatRole::Assistant,
                content: content.into(),
            }
        }
    }

    /// **Unstable**: output from chat completion; fields may expand with streaming and usage stats.
    ///
    /// Output from chat completion.
    #[derive(Debug, Clone)]
    pub struct ChatCompletionOutput {
        pub message: ChatMessage,
        pub prompt_tokens: usize,
        pub completion_tokens: usize,
    }

    /// **Unstable**: format messages into Qwen3.5 chat template; template format may change.
    ///
    /// Format messages into Qwen3.5 chat template.
    /// Template: <|im_start|>{role}\n{content}<|im_end|>\n
    /// Final assistant turn left open for generation.
    pub fn format_chat_template(messages: &[ChatMessage]) -> String {
        let mut prompt = String::new();
        for msg in messages {
            prompt.push_str("<|im_start|>");
            prompt.push_str(msg.role.as_str());
            prompt.push('\n');
            prompt.push_str(&msg.content);
            prompt.push_str("<|im_end|>\n");
        }
        // Open assistant turn for generation
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }

    impl MetalQwen35State {
        /// **Unstable**: chat completion; stop token handling and output format may change.
        ///
        /// Chat completion: format messages with Qwen3.5 template, generate response.
        /// Stops on <|im_end|> or EOS or max tokens.
        pub fn chat_completion(
            &mut self,
            messages: &[ChatMessage],
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
        ) -> ChatCompletionOutput {
            let prompt = format_chat_template(messages);
            // Add <|im_end|> as stop token
            let mut cfg = gen_cfg.clone();
            if let Some(im_end_id) = tokenizer.special_token_id("<|im_end|>") {
                if !cfg.stop_token_ids.contains(&im_end_id) {
                    cfg.stop_token_ids.push(im_end_id);
                }
            }
            let result = self.generate(&prompt, tokenizer, &cfg);
            let text = result.text.trim_end().to_string();
            ChatCompletionOutput {
                message: ChatMessage::assistant(text),
                prompt_tokens: result.prompt_tokens,
                completion_tokens: result.generated_tokens,
            }
        }

        /// **Unstable**: streaming generation; callback signature may change.
        ///
        /// Streaming generation: calls `on_token` for each generated token.
        /// Returns total generated token count.
        /// The callback receives (token_text, token_id) and returns true to continue.
        pub fn generate_streaming<F>(
            &mut self,
            prompt: &str,
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
            mut on_token: F,
        ) -> GenerateOutput
        where
            F: FnMut(&str, u32) -> bool,
        {
            let cfg = self.engine.config.clone();

            let mut rng_state = match gen_cfg.seed {
                Some(s) => {
                    if s == 0 {
                        1
                    } else {
                        s
                    }
                }
                None => {
                    use std::time::SystemTime;
                    let t = SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0x12345678_9abcdef0);
                    if t == 0 { 1 } else { t }
                }
            };

            let input = tokenizer.tokenize(prompt);
            let prompt_ids: Vec<u32> = input.input_ids[..input.real_length].to_vec();
            let prompt_len = prompt_ids.len();

            if prompt_len == 0 {
                return GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: 0,
                    generated_tokens: 0,
                };
            }

            self.reset_state();
            let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
            let mut all_ids = prompt_ids.clone();

            let route = choose_gpu_topk_route(
                gen_cfg.top_k,
                std::env::var("LATTICE_COMPACT_TOPK").is_ok(),
                std::env::var("LATTICE_COMPACT_TOPK_SELECT").is_ok(),
            );
            let use_compact = route != GpuTopkRoute::CpuFallback
                && (gen_cfg.repetition_penalty == 1.0 || all_ids.is_empty());
            self.session.compact_route = if use_compact {
                route
            } else {
                GpuTopkRoute::CpuFallback
            };
            if use_compact {
                self.session.compact_topk = gen_cfg.top_k;
            }

            // Batch prefill
            let prefill_logits = self.forward_prefill(&prompt_ids);
            let next_id = if use_compact {
                sample_from_candidates(
                    &self.session.compact_result,
                    gen_cfg,
                    &all_ids,
                    &mut rng_state,
                )
            } else {
                sample_token(&prefill_logits, gen_cfg, &all_ids, &mut rng_state)
            };

            let is_stop = |id: u32| -> bool {
                id == cfg.eos_token_id || gen_cfg.stop_token_ids.contains(&id)
            };

            if is_stop(next_id) {
                if use_compact {
                    self.session.compact_topk = 0;
                    self.session.compact_route = GpuTopkRoute::CpuFallback;
                }
                return GenerateOutput {
                    text: String::new(),
                    token_ids: vec![],
                    prompt_tokens: prompt_len,
                    generated_tokens: 0,
                };
            }

            generated_ids.push(next_id);
            all_ids.push(next_id);
            let token_text = decode_tokens(tokenizer, &[next_id]);
            if !on_token(&token_text, next_id) {
                if use_compact {
                    self.session.compact_topk = 0;
                    self.session.compact_route = GpuTopkRoute::CpuFallback;
                }
                let text = decode_tokens(tokenizer, &generated_ids);
                return GenerateOutput {
                    text,
                    token_ids: generated_ids.clone(),
                    prompt_tokens: prompt_len,
                    generated_tokens: generated_ids.len(),
                };
            }

            // Autoregressive decode with streaming
            for _ in 1..gen_cfg.max_new_tokens {
                if self.session.kv_cache.seq_len >= self.session.kv_cache.max_cache_len {
                    break;
                }
                let pos = self.session.kv_cache.seq_len;
                let last_token = *all_ids
                    .last()
                    .expect("invariant: prompt or previous sample populated all_ids");
                let step_logits = self.forward_step(last_token, pos);
                let next_id = if use_compact {
                    sample_from_candidates(
                        &self.session.compact_result,
                        gen_cfg,
                        &all_ids,
                        &mut rng_state,
                    )
                } else {
                    sample_token(&step_logits, gen_cfg, &all_ids, &mut rng_state)
                };

                if is_stop(next_id) {
                    break;
                }

                generated_ids.push(next_id);
                all_ids.push(next_id);
                let token_text = decode_tokens(tokenizer, &[next_id]);
                if !on_token(&token_text, next_id) {
                    break;
                }
                if self.session.kv_cache.seq_len >= self.session.kv_cache.max_cache_len {
                    break;
                }
            }

            if use_compact {
                self.session.compact_topk = 0;
                self.session.compact_route = GpuTopkRoute::CpuFallback;
            }

            let text = decode_tokens(tokenizer, &generated_ids);
            GenerateOutput {
                text,
                token_ids: generated_ids.clone(),
                prompt_tokens: prompt_len,
                generated_tokens: generated_ids.len(),
            }
        }

        // ---------------------------------------------------------------------------
        // Q4 direct-load helpers
        // ---------------------------------------------------------------------------

        /// Build the sanitized file-stem from a tensor name (dots/slashes → underscores).
        fn q4_tensor_path(
            dir: &std::path::Path,
            tensor_name: &str,
            ext: &str,
        ) -> std::path::PathBuf {
            let sanitized: String = tensor_name
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            dir.join(format!("{sanitized}.{ext}"))
        }

        /// Create a Metal buffer from pre-quantized Q4 block bytes (raw `[u8]`).
        ///
        /// The bytes are the serialized `Q4Block` array as written by `save_q4_file`
        /// (after the file header), i.e. `n_blocks × 18` raw bytes.
        fn make_buffer_from_q4_raw(device: &Device, raw: &[u8], label: &str) -> Buffer {
            let buf = device.new_buffer_with_data(
                raw.as_ptr() as *const _,
                raw.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            buf.set_label(label);
            buf
        }

        /// Dequantize a [`Q4Tensor`] to f16 (as `u16` bit patterns) and create a Metal buffer.
        ///
        /// Used for `embed_tokens`: the CPU embedding lookup reads f16 values from this buffer,
        /// so we need an f16 Metal buffer even when the on-disk format is Q4.
        fn make_buffer_f16_from_q4(
            device: &Device,
            tensor: &crate::weights::q4_weights::Q4Tensor,
            label: &str,
        ) -> Buffer {
            use crate::weights::q4_weights::{q4_f16_to_f32, q4_f32_to_f16};
            let mut f16_data: Vec<u16> = Vec::with_capacity(tensor.original_len);
            for block in &tensor.blocks {
                let scale = q4_f16_to_f32(block.scale);
                for b in 0..16 {
                    let byte_val = block.packed[b];
                    let w0 = ((byte_val & 0x0f) as f32 - 8.0) * scale;
                    let w1 = ((byte_val >> 4) as f32 - 8.0) * scale;
                    f16_data.push(q4_f32_to_f16(w0));
                    f16_data.push(q4_f32_to_f16(w1));
                }
            }
            f16_data.truncate(tensor.original_len);
            let byte_len = (f16_data.len() * std::mem::size_of::<u16>()) as u64;
            let buf = device.new_buffer_with_data(
                f16_data.as_ptr() as *const _,
                byte_len,
                MTLResourceOptions::StorageModeShared,
            );
            buf.set_label(label);
            buf
        }

        /// Load raw Q4 block bytes from a `.q4` file (strips the file header).
        ///
        /// Returns the `Q4Block` bytes and the `original_len` (for validation).
        fn load_q4_raw_bytes(path: &std::path::Path) -> Result<(Vec<u8>, usize), String> {
            use crate::weights::q4_weights::load_q4_file;
            let tensor = load_q4_file(path)
                .map_err(|e| format!("failed to load Q4 file {}: {e}", path.display()))?;
            let n_blocks = tensor.blocks.len();
            // Re-serialise the blocks into raw bytes (18 bytes each, Q4Block is #[repr(C)]).
            // SAFETY: Q4Block is #[repr(C)] size 18, alignment 2 (from leading
            // `scale: u16`); byte-cast is valid because target element type is u8.
            let raw: Vec<u8> = unsafe {
                std::slice::from_raw_parts(tensor.blocks.as_ptr().cast::<u8>(), n_blocks * 18)
                    .to_vec()
            };
            Ok((raw, tensor.original_len))
        }

        /// Load MTP weights from a Q4 directory and build `MetalMtpRuntime`.
        ///
        /// Returns `None` if the model has no MTP layers or if any weight file is missing
        /// (soft failure — caller falls back to non-MTP decode).
        fn load_mtp_q4_weights(
            q4_dir: &std::path::Path,
            cfg: &Qwen35Config,
            device: &Device,
        ) -> Option<MetalMtpWeights> {
            use crate::weights::q4_weights::load_f16_tensor_file;

            if cfg.mtp_num_hidden_layers == 0 {
                return None;
            }

            let q4p = |name: &str| MetalQwen35State::q4_tensor_path(q4_dir, name, "q4");
            let f16p = |name: &str| MetalQwen35State::q4_tensor_path(q4_dir, name, "f16");

            // Helper: load an f16 file as a f32 Metal buffer; return None on missing.
            let load_f16_buf = |name: &str, label: &str| -> Option<Buffer> {
                let path = f16p(name);
                if !path.exists() {
                    return None;
                }
                let (vals, _) = load_f16_tensor_file(&path).ok()?;
                Some(make_buffer(device, &vals, label))
            };

            // Helper: mmap-load a Q4 file; return None on missing.
            let load_q4 = |name: &str, label: &str| -> Option<Q4WeightBuf> {
                let path = q4p(name);
                if !path.exists() {
                    return None;
                }
                mmap_q4_weight(device, &path, label).ok()
            };

            // --- Layer 0 weights ---
            let input_layernorm = load_f16_buf(
                "mtp.layers.0.input_layernorm.weight",
                "mtp.l0.input_layernorm",
            )?;
            let post_attention_layernorm = load_f16_buf(
                "mtp.layers.0.post_attention_layernorm.weight",
                "mtp.l0.post_attn_layernorm",
            )?;
            let q_proj = load_q4("mtp.layers.0.self_attn.q_proj.weight", "mtp.l0.q_proj")?;
            let k_proj = load_q4("mtp.layers.0.self_attn.k_proj.weight", "mtp.l0.k_proj")?;
            let v_proj = load_q4("mtp.layers.0.self_attn.v_proj.weight", "mtp.l0.v_proj")?;
            let o_proj = load_q4("mtp.layers.0.self_attn.o_proj.weight", "mtp.l0.o_proj")?;
            let q_norm = load_f16_buf("mtp.layers.0.self_attn.q_norm.weight", "mtp.l0.q_norm")?;
            let k_norm = load_f16_buf("mtp.layers.0.self_attn.k_norm.weight", "mtp.l0.k_norm")?;
            let gate_proj = load_q4("mtp.layers.0.mlp.gate_proj.weight", "mtp.l0.gate_proj")?;
            let up_proj = load_q4("mtp.layers.0.mlp.up_proj.weight", "mtp.l0.up_proj")?;
            let down_proj = load_q4("mtp.layers.0.mlp.down_proj.weight", "mtp.l0.down_proj")?;

            // --- Top-level MTP weights ---
            let fc = load_q4("mtp.fc.weight", "mtp.fc")?;
            let pre_fc_norm_embedding = load_f16_buf(
                "mtp.pre_fc_norm_embedding.weight",
                "mtp.pre_fc_norm_embedding",
            )?;
            let pre_fc_norm_hidden =
                load_f16_buf("mtp.pre_fc_norm_hidden.weight", "mtp.pre_fc_norm_hidden")?;
            let norm = load_f16_buf("mtp.norm.weight", "mtp.norm")?;

            eprintln!("[mtp] Loaded MTP layer 0 weights from {}", q4_dir.display());
            Some(MetalMtpWeights {
                fc,
                pre_fc_norm_embedding,
                pre_fc_norm_hidden,
                layers: vec![MetalMtpLayerWeights {
                    input_layernorm,
                    post_attention_layernorm,
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm,
                    k_norm,
                    mlp: MetalMtpDenseMlpWeights {
                        gate_proj,
                        up_proj,
                        down_proj,
                    },
                }],
                norm,
            })
        }

        /// Create a new [`MetalQwen35State`] by loading pre-quantized Q4/F16 files directly
        /// from a directory, without going through the full f32 weight representation.
        ///
        /// This avoids the ~51 GB peak RAM usage of the `new()` path (which loads all weights
        /// as `Vec<f32>` and then quantizes on the Metal side) and instead reads Q4/f16 files
        /// written by `foundation/inference/src/bin/quantize_q4.rs`.
        ///
        /// # File layout
        ///
        /// `q4_dir` must contain files produced by `quantize_q4`:
        /// - Large projections (`*_proj*.weight`, `gate_proj`, `up_proj`, `down_proj`,
        ///   `embed_tokens`, `lm_head`) → `<sanitized_name>.q4`
        /// - Small tensors (norms, biases, A_log, dt_bias, conv1d) → `<sanitized_name>.f16`
        ///
        /// # Errors
        ///
        /// Returns `Err(String)` on missing files, I/O failures, or Metal initialisation
        /// failures.
        pub fn from_q4_dir(
            q4_dir: &std::path::Path,
            tokenizer_path: &std::path::Path,
            cfg: &Qwen35Config,
            max_cache_len: usize,
        ) -> Result<Self, String> {
            use crate::weights::q4_weights::{load_f16_tensor_file, load_q4_file};

            let device =
                Device::system_default().ok_or_else(|| "No Metal device found".to_string())?;

            tracing::info!(
                name = device.name(),
                "Metal GPU initialized for Q4 direct-load forward pass"
            );

            let queue = device.new_command_queue();

            // Q4 dir always uses Q4_0 format.
            let quant_format = QuantFormat::Q4_0;

            validate_flash_decode_shape(
                cfg.head_dim,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.num_attention_heads * cfg.head_dim,
                cfg.num_key_value_heads * cfg.head_dim,
            )?;

            // Cache-capacity validation, matching `new_session` so a
            // `from_q4_dir` call cannot construct a runtime whose KV cap
            // outruns the RoPE table (`partial_rope_interleaved` indexes
            // `cos_tab` / `sin_tab` by `position`; without this guard a
            // `--max-cache-len > max_position_embeddings` argument would
            // race past the precomputed table at decode time).
            if max_cache_len == 0 {
                return Err("from_q4_dir: max_cache_len must be > 0".to_string());
            }
            if max_cache_len > cfg.max_position_embeddings {
                return Err(format!(
                    "from_q4_dir: max_cache_len {max_cache_len} exceeds max_position_embeddings {} \
                     (the RoPE table is sized to max_position_embeddings; running past it would \
                     produce out-of-bounds GPU reads or incorrect rotary positions)",
                    cfg.max_position_embeddings
                ));
            }

            // Two-way config/artifact contract for the logits head:
            //
            // - `tie_word_embeddings=false` (e.g., QuaRot output where
            //   step 3c flips the flag and materializes a fused
            //   `(1 + γ_final)` + rotation `lm_head.weight`) MUST ship
            //   the rotated head as its own `lm_head.weight.q4`. Falling
            //   back to `embed_tokens` here would silently feed the wrong
            //   matrix into the final logits GEMV.
            // - `tie_word_embeddings=true` (the legitimate "embed_tokens
            //   IS the head" case) MUST NOT ship `lm_head.weight.q4` —
            //   if it does, the on-disk config-flip was likely skipped
            //   (HF Qwen3.5 stores the flag in both `text_config` and
            //   top level; `Qwen35Config::from_config_json_str` takes
            //   the top-level value, so a partial update can leave the
            //   parsed flag stale). The materialized head would then be
            //   dead weight and the runtime would silently fall back to
            //   `embed_tokens` — exactly the contamination ADR-044
            //   §"Step 3c contract" warns against and `quant::quarot::lm_head`
            //   flags at the converter boundary.
            //
            // Both mismatches let `eval_perplexity --quarot-q4-dir`
            // print a PPL delta and PASS/FAIL verdict on the wrong
            // logits head, so the loader refuses both directions before
            // any tensor I/O.
            let lm_head_path = Self::q4_tensor_path(q4_dir, "lm_head.weight", "q4");
            let lm_head_present = lm_head_path.exists();
            if !cfg.tie_word_embeddings && !lm_head_present {
                return Err(format!(
                    "from_q4_dir: tie_word_embeddings=false but lm_head.q4 is missing at {}; \
                     the runtime requires the materialized lm_head matrix (ADR-044 step 3c) \
                     and must not fall back to embed_tokens, which would yield wrong logits \
                     and a misleading perplexity report",
                    lm_head_path.display()
                ));
            }
            if cfg.tie_word_embeddings && lm_head_present {
                return Err(format!(
                    "from_q4_dir: tie_word_embeddings=true but lm_head.q4 is present at {}; \
                     the on-disk config likely missed the step-3c `tie_word_embeddings=false` \
                     flip (HF Qwen3.5 carries the flag in both nested `text_config` and the \
                     top level — see `Qwen35Config::from_config_json_str`). Loading would \
                     silently ignore the materialized head and fall back to embed_tokens, \
                     yielding wrong logits and a misleading perplexity report",
                    lm_head_path.display()
                ));
            }

            // Compile shaders (same as new()).
            let opts = CompileOptions::new();
            let library = device
                .new_library_with_source(MSL_SOURCE, &opts)
                .map_err(|e| format!("Metal shader compilation failed: {e}"))?;

            let make_pipeline = |name: &str| -> Result<ComputePipelineState, String> {
                let func = library
                    .get_function(name, None)
                    .map_err(|e| format!("function '{name}' not found: {e}"))?;
                device
                    .new_compute_pipeline_state_with_function(&func)
                    .map_err(|e| format!("pipeline for '{name}' failed: {e}"))
            };

            let make_optional_gemm_q4_tiled = || -> Option<ComputePipelineState> {
                if !device.supports_family(MTLGPUFamily::Apple9) {
                    return None;
                }
                let tiled_opts = CompileOptions::new();
                tiled_opts.set_language_version(MTLLanguageVersion::V3_0);
                let lib = device
                    .new_library_with_source(MSL_Q4_TILED_SOURCE, &tiled_opts)
                    .ok()?;
                let func = lib.get_function("gemm_q4_tiled", None).ok()?;
                device.new_compute_pipeline_state_with_function(&func).ok()
            };

            let pipelines = MetalQwen35Pipelines {
                gemv_decode: make_pipeline("gemv_decode_m1")?,
                gemv_q8: make_pipeline("gemv_q8_decode")?,
                gemv_q4: make_pipeline("gemv_q4_decode")?,
                gemm_q4: make_pipeline("gemm_q4")?,
                gemm_q4_tiled: make_optional_gemm_q4_tiled(),
                rms_norm: make_pipeline("rms_norm_qwen35")?,
                partial_rope: make_pipeline("partial_rope_interleaved")?,
                per_head_rms_norm: make_pipeline("per_head_rms_norm")?,
                decode_attention: make_pipeline("decode_attention")?,
                sigmoid_gate: make_pipeline("sigmoid_gate")?,
                scatter_q_gate: make_pipeline("scatter_q_gate")?,
                silu_mul: make_pipeline("silu_mul")?,
                copy: make_pipeline("copy_buf")?,
                copy_offset: make_pipeline("copy_buf_offset")?,
                add: make_pipeline("add_buf")?,
                conv1d_silu: make_pipeline("conv1d_depthwise_silu")?,
                gdn_recurrence: make_pipeline("gdn_recurrence_fused")?,
                gdn_recurrence_q36: library
                    .get_function("gdn_recurrence_fused_q36", None)
                    .ok()
                    .and_then(|f| device.new_compute_pipeline_state_with_function(&f).ok()),
                gdn_precompute_keys: library
                    .get_function("gdn_precompute_keys", None)
                    .ok()
                    .and_then(|f| device.new_compute_pipeline_state_with_function(&f).ok()),
                gdn_recurrence_sharded: library
                    .get_function("gdn_recurrence_sharded", None)
                    .ok()
                    .and_then(|f| device.new_compute_pipeline_state_with_function(&f).ok()),
                gdn_norm_silu: library
                    .get_function("gdn_norm_silu", None)
                    .ok()
                    .and_then(|f| device.new_compute_pipeline_state_with_function(&f).ok()),
                fused_residual_add_norm: make_pipeline("fused_residual_add_norm")?,
                copy_and_rms_norm: make_pipeline("copy_and_rms_norm")?,
                add_and_copy: make_pipeline("add_and_copy")?,
                gemm_q8: make_pipeline("gemm_q8")?,
                topk_first_pass: make_pipeline("logits_topk_first_pass")?,
                topk_merge_pass: make_pipeline("logits_topk_merge_pass")?,
                argmax_first: make_pipeline("logits_argmax_first")?,
                argmax_merge: make_pipeline("logits_argmax_merge")?,
                topk_fast_first: make_pipeline("logits_topk_fast_first")?,
                topk_select50_first: make_pipeline("logits_topk_select50_first")?,
                topk_select50_merge: make_pipeline("logits_topk_select50_merge")?,
                decode_attn_partial: make_pipeline("decode_attention_flash_partial")?,
                decode_attn_reduce: make_pipeline("decode_attention_flash_reduce")?,
                lora_gemv_a: make_pipeline("lora_gemv_a")?,
                lora_gemv_b_accum: make_pipeline("lora_gemv_b_accum")?,
            };

            let hidden = cfg.hidden_size;
            let q_dim = cfg.full_q_dim();
            let kv_dim = cfg.full_kv_dim();
            let qkv_dim = cfg.linear_qkv_dim();
            let output_dim = cfg.linear_output_dim();
            let inter = cfg.intermediate_size;

            // ----------------------------------------------------------------
            // Phase timers — measures I/O and Metal buffer creation costs.
            // ----------------------------------------------------------------
            let t_total = std::time::Instant::now();
            let mut dur_a = std::time::Duration::ZERO; // Phase A: qkvz I/O + Metal copies
            let dur_b_cell = std::cell::Cell::new(std::time::Duration::ZERO); // Phase B: F16 loads
            let dur_c_cell = std::cell::Cell::new(std::time::Duration::ZERO); // Phase C: mmap Q4

            // ----------------------------------------------------------------
            // Helper: mmap a Q4 file and create a Metal no-copy buffer.
            // Zero CPU copies — GPU pages fault lazily from mmap'd file.
            // ----------------------------------------------------------------
            let load_q4_buf = |name: &str, label: &str| -> Result<Q4WeightBuf, String> {
                let t = std::time::Instant::now();
                let path = Self::q4_tensor_path(q4_dir, name, "q4");
                let result = mmap_q4_weight(&device, &path, label);
                dur_c_cell.set(dur_c_cell.get() + t.elapsed());
                result
            };

            // ----------------------------------------------------------------
            // Helper: load an F16 file, convert to f32, create a Metal f32 buffer.
            // ----------------------------------------------------------------
            let load_f16_buf_f32 = |name: &str, label: &str| -> Result<Buffer, String> {
                let t = std::time::Instant::now();
                let path = Self::q4_tensor_path(q4_dir, name, "f16");
                let (values, _shape) = load_f16_tensor_file(&path)
                    .map_err(|e| format!("failed to load f16 file {}: {e}", path.display()))?;
                let buf = make_buffer(&device, &values, label);
                dur_b_cell.set(dur_b_cell.get() + t.elapsed());
                Ok(buf)
            };

            // ----------------------------------------------------------------
            // H3: Merge-on-first-load for in_proj_qkvz.
            // On first load, concatenate qkv+z payloads into a single merged .q4 file
            // so that subsequent loads can use zero-copy mmap (like all other Q4 weights).
            // Merged filename encodes source sizes → stale cache auto-invalidated on model update.
            // Falls back to CPU concat if model dir is read-only.
            // Metal buffer creation stays on the main thread (Metal is not thread-safe).
            // ----------------------------------------------------------------
            let t_a_io = std::time::Instant::now();
            let mut qkvz_merge_map: std::collections::HashMap<usize, Option<std::path::PathBuf>> = {
                use rayon::prelude::*;
                let pairs: Vec<Result<(usize, Option<std::path::PathBuf>), String>> =
                    (0..cfg.num_hidden_layers)
                        .into_par_iter()
                        .filter(|&i| cfg.is_layer_active(i) && !cfg.is_full_attention(i))
                        .map(|i| {
                            let prefix = format!("model.language_model.layers.{i}");
                            let qkv_p = Self::q4_tensor_path(
                                q4_dir,
                                &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                                "q4",
                            );
                            let z_p = Self::q4_tensor_path(
                                q4_dir,
                                &format!("{prefix}.linear_attn.in_proj_z.weight"),
                                "q4",
                            );
                            let qkv_size = std::fs::metadata(&qkv_p)
                                .map_err(|e| format!("metadata {}: {e}", qkv_p.display()))?.len();
                            let z_size = std::fs::metadata(&z_p)
                                .map_err(|e| format!("metadata {}: {e}", z_p.display()))?.len();
                            // Filename encodes source sizes for cache-invalidation.
                            let merged_path = q4_dir.join(
                                format!("merged_qkvz_{i}_{qkv_size}_{z_size}.q4")
                            );
                            // Expected size: 36-byte header (ndim=2, payload_offset=36)
                            // + qkv payload + z payload.  All source weight files are 2D
                            // so payload_offset=36 for each.
                            let expected_size = 36u64 + (qkv_size - 36) + (z_size - 36);
                            let is_valid = std::fs::metadata(&merged_path)
                                .ok()
                                .map(|m| m.len() == expected_size)
                                .unwrap_or(false);
                            if !is_valid {
                                let _ = std::fs::remove_file(&merged_path);
                                if let Err(e) = write_merged_qkvz(&qkv_p, &z_p, &merged_path) {
                                    eprintln!("[load] merge-on-first-load failed layer {i}: {e}; using CPU fallback");
                                    return Ok((i, None));
                                }
                            }
                            Ok((i, Some(merged_path)))
                        })
                        .collect();
                pairs.into_iter().collect::<Result<_, _>>()?
            };
            dur_a += t_a_io.elapsed();

            // ----------------------------------------------------------------
            // Per-layer weights
            // ----------------------------------------------------------------
            let mut layer_weights: Vec<(MetalLayerAttnWeights, MetalCommonLayerWeights)> =
                Vec::with_capacity(cfg.num_active_layers());

            for i in 0..cfg.num_hidden_layers {
                if !cfg.is_layer_active(i) {
                    tracing::debug!(layer = i, "skipping pruned layer weight loading");
                    continue;
                }
                let prefix = format!("model.language_model.layers.{i}");

                // Common layer weights
                let common = MetalCommonLayerWeights {
                    input_layernorm: load_f16_buf_f32(
                        &format!("{prefix}.input_layernorm.weight"),
                        &format!("L{i}.in_norm"),
                    )?,
                    post_attention_layernorm: load_f16_buf_f32(
                        &format!("{prefix}.post_attention_layernorm.weight"),
                        &format!("L{i}.post_norm"),
                    )?,
                    gate_proj: load_q4_buf(
                        &format!("{prefix}.mlp.gate_proj.weight"),
                        &format!("L{i}.gate.q4"),
                    )?,
                    up_proj: load_q4_buf(
                        &format!("{prefix}.mlp.up_proj.weight"),
                        &format!("L{i}.up.q4"),
                    )?,
                    down_proj: load_q4_buf(
                        &format!("{prefix}.mlp.down_proj.weight"),
                        &format!("L{i}.down.q4"),
                    )?,
                };

                let attn = if cfg.is_full_attention(i) {
                    MetalLayerAttnWeights::Full(MetalFullLayerWeights {
                        q_proj: load_q4_buf(
                            &format!("{prefix}.self_attn.q_proj.weight"),
                            &format!("L{i}.full.q.q4"),
                        )?,
                        k_proj: load_q4_buf(
                            &format!("{prefix}.self_attn.k_proj.weight"),
                            &format!("L{i}.full.k.q4"),
                        )?,
                        v_proj: load_q4_buf(
                            &format!("{prefix}.self_attn.v_proj.weight"),
                            &format!("L{i}.full.v.q4"),
                        )?,
                        o_proj: load_q4_buf(
                            &format!("{prefix}.self_attn.o_proj.weight"),
                            &format!("L{i}.full.o.q4"),
                        )?,
                        // q_norm and k_norm are norm.weight → stored as .f16
                        q_norm: load_f16_buf_f32(
                            &format!("{prefix}.self_attn.q_norm.weight"),
                            &format!("L{i}.full.q_norm"),
                        )?,
                        k_norm: load_f16_buf_f32(
                            &format!("{prefix}.self_attn.k_norm.weight"),
                            &format!("L{i}.full.k_norm"),
                        )?,
                    })
                } else {
                    // in_proj_b and in_proj_a are quantized on disk (ends_with _proj_b.weight /
                    // _proj_a.weight → Q4), but in new() they become f16 Metal buffers for the
                    // CPU GDN recurrence path.  Load Q4 → dequantize → f16 Metal buffer.
                    let load_q4_as_f16_buf = |name: &str, label: &str| -> Result<Buffer, String> {
                        let t = std::time::Instant::now();
                        let path = Self::q4_tensor_path(q4_dir, name, "q4");
                        let tensor = load_q4_file(&path).map_err(|e| {
                            format!("failed to load Q4 file {}: {e}", path.display())
                        })?;
                        let buf = Self::make_buffer_f16_from_q4(&device, &tensor, label);
                        dur_b_cell.set(dur_b_cell.get() + t.elapsed());
                        Ok(buf)
                    };

                    MetalLayerAttnWeights::Linear(MetalGdnLayerWeights {
                        in_proj_qkv: load_q4_buf(
                            &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                            &format!("L{i}.gdn.qkv.q4"),
                        )?,
                        in_proj_z: load_q4_buf(
                            &format!("{prefix}.linear_attn.in_proj_z.weight"),
                            &format!("L{i}.gdn.z.q4"),
                        )?,
                        in_proj_qkvz: {
                            let t_qkvz = std::time::Instant::now();
                            let r = match qkvz_merge_map.remove(&i) {
                                Some(Some(merged_path)) => mmap_q4_weight(
                                    &device,
                                    &merged_path,
                                    &format!("L{i}.gdn.qkvz.q4"),
                                )?,
                                _ => {
                                    // Fallback: model dir is read-only — CPU concat path
                                    let pfx = format!("model.language_model.layers.{i}");
                                    let qkv_p = Self::q4_tensor_path(
                                        q4_dir,
                                        &format!("{pfx}.linear_attn.in_proj_qkv.weight"),
                                        "q4",
                                    );
                                    let z_p = Self::q4_tensor_path(
                                        q4_dir,
                                        &format!("{pfx}.linear_attn.in_proj_z.weight"),
                                        "q4",
                                    );
                                    let (mut raw, _) = Self::load_q4_raw_bytes(&qkv_p)?;
                                    let (z_raw, _) = Self::load_q4_raw_bytes(&z_p)?;
                                    raw.extend_from_slice(&z_raw);
                                    Q4WeightBuf::from_buffer(Self::make_buffer_from_q4_raw(
                                        &device,
                                        &raw,
                                        &format!("L{i}.gdn.qkvz.q4"),
                                    ))
                                }
                            };
                            dur_a += t_qkvz.elapsed();
                            r
                        },
                        // in_proj_b and in_proj_a: Q4 on disk, but stored as f16 Metal buffers
                        in_proj_b: load_q4_as_f16_buf(
                            &format!("{prefix}.linear_attn.in_proj_b.weight"),
                            &format!("L{i}.gdn.b.f16"),
                        )?,
                        in_proj_a: load_q4_as_f16_buf(
                            &format!("{prefix}.linear_attn.in_proj_a.weight"),
                            &format!("L{i}.gdn.a.f16"),
                        )?,
                        // Small scalars: f32 Metal buffers (CPU-read in GDN recurrence)
                        a_log: load_f16_buf_f32(
                            &format!("{prefix}.linear_attn.A_log"),
                            &format!("L{i}.gdn.a_log"),
                        )?,
                        dt_bias: load_f16_buf_f32(
                            &format!("{prefix}.linear_attn.dt_bias"),
                            &format!("L{i}.gdn.dt_bias"),
                        )?,
                        conv1d_weight: load_f16_buf_f32(
                            &format!("{prefix}.linear_attn.conv1d.weight"),
                            &format!("L{i}.gdn.conv1d"),
                        )?,
                        norm_weight: load_f16_buf_f32(
                            &format!("{prefix}.linear_attn.norm.weight"),
                            &format!("L{i}.gdn.norm"),
                        )?,
                        out_proj: load_q4_buf(
                            &format!("{prefix}.linear_attn.out_proj.weight"),
                            &format!("L{i}.gdn.out.q4"),
                        )?,
                    })
                };

                layer_weights.push((attn, common));
            }

            let t_d = std::time::Instant::now();

            // ----------------------------------------------------------------
            // embed_tokens: Q4 on disk.
            //   - embed_tokens (f16 buf): for CPU embedding lookup
            //   - embed_tokens_q8 (Q4 buf): for logits GEMV on GPU
            // ----------------------------------------------------------------
            let embed_name = "model.language_model.embed_tokens.weight";
            let embed_path = Self::q4_tensor_path(q4_dir, embed_name, "q4");
            let embed_tensor = load_q4_file(&embed_path)
                .map_err(|e| format!("failed to load embed_tokens Q4 file: {e}"))?;
            // f16 buffer for CPU embedding lookup
            let embed_tokens =
                Self::make_buffer_f16_from_q4(&device, &embed_tensor, "embed_tokens.f16");
            // Q4 buffer for GPU logits GEMV — mmap no-copy
            let embed_q4_path = Self::q4_tensor_path(q4_dir, embed_name, "q4");
            let embed_tokens_q8 = mmap_q4_weight(&device, &embed_q4_path, "embed_tokens.q4")?;

            // ----------------------------------------------------------------
            // final_norm: stored as .f16 (it's a norm.weight tensor)
            // ----------------------------------------------------------------
            let final_norm = load_f16_buf_f32("model.language_model.norm.weight", "final_norm")?;

            // ----------------------------------------------------------------
            // lm_head: only present when tie_word_embeddings = false.
            //
            // Step-4b loader contract: when `!cfg.tie_word_embeddings` we
            // require `lm_head.weight.q4` on disk — the early-return guard
            // at the top of `from_q4_dir` already rejected the missing-file
            // case, so this branch only sees a path that exists. We load
            // it into the `embed_tokens_q8` slot because that buffer is
            // what the runtime's logits GEMV indexes; the field is shared
            // between tied (embed_tokens IS the head) and untied
            // (separate fused-and-rotated head) artifacts.
            // ----------------------------------------------------------------
            let embed_tokens_q8 = if !cfg.tie_word_embeddings {
                let lm_head_path = Self::q4_tensor_path(q4_dir, "lm_head.weight", "q4");
                mmap_q4_weight(&device, &lm_head_path, "lm_head.q4")?
            } else {
                embed_tokens_q8
            };

            // ----------------------------------------------------------------
            // RoPE tables (same as new())
            // ----------------------------------------------------------------
            let rope_dim = cfg.rope_dim();
            let rope_max = cfg.max_position_embeddings.min(max_cache_len + 64);
            let (cos_data, sin_data) = build_rope_interleaved(rope_dim, rope_max, cfg.rope_theta);
            let rope_cos = make_buffer(&device, &cos_data, "rope_cos");
            let rope_sin = make_buffer(&device, &sin_data, "rope_sin");

            // ----------------------------------------------------------------
            // Activation buffers (same as new())
            // ----------------------------------------------------------------
            let max_prefill = max_cache_len.min(512);
            let bp = max_prefill;
            let activations = MetalQwen35Activations {
                hidden: make_zero_buffer(&device, bp * hidden, "act_hidden"),
                residual: make_zero_buffer(&device, bp * hidden, "act_residual"),
                attn_out: make_zero_buffer(&device, bp * q_dim.max(hidden), "act_attn_out"),
                q: make_zero_buffer(&device, bp * 2 * q_dim, "act_q_interleaved"),
                q_separated: make_zero_buffer(&device, bp * q_dim, "act_q"),
                gate_z: make_zero_buffer(&device, bp * q_dim, "act_gate_z"),
                k: make_zero_buffer(&device, bp * kv_dim, "act_k"),
                v: make_zero_buffer(&device, bp * kv_dim, "act_v"),
                gate: make_zero_buffer(&device, bp * inter, "act_gate"),
                up: make_zero_buffer(&device, bp * inter, "act_up"),
                ffn_out: make_zero_buffer(&device, bp * hidden, "act_ffn_out"),
                gdn_qkv: make_zero_buffer(&device, bp * qkv_dim, "act_gdn_qkv"),
                gdn_z: make_zero_buffer(&device, bp * output_dim, "act_gdn_z"),
                gdn_qkvz: make_zero_buffer(&device, qkv_dim + output_dim, "act_gdn_qkvz"),
                gdn_key_scratch: make_zero_buffer(
                    &device,
                    cfg.linear_num_key_heads * (2 * cfg.linear_key_head_dim + 3),
                    "act_gdn_key_scratch",
                ),
                gdn_raw_out: make_zero_buffer(&device, output_dim, "act_gdn_raw_out"),
                logits: make_zero_buffer(&device, cfg.vocab_size, "act_logits"),
                topk_scratch_a: {
                    let groups = cfg.vocab_size.div_ceil(1024);
                    make_zero_byte_buffer(&device, groups * 256 * 8, "topk_scratch_a")
                },
                topk_scratch_b: {
                    let groups = cfg.vocab_size.div_ceil(1024);
                    make_zero_byte_buffer(&device, groups * 256 * 8, "topk_scratch_b")
                },
                attn_partials: {
                    let max_partitions = max_cache_len.div_ceil(1024);
                    let stride = cfg.head_dim + 2;
                    make_zero_buffer(
                        &device,
                        max_partitions * cfg.num_attention_heads * stride,
                        "attn_partials",
                    )
                },
                pre_final_hidden: make_zero_buffer(&device, hidden, "act_pre_final_hidden"),
                verify_logits: make_zero_buffer(
                    &device,
                    MTP_VERIFY_MAX_TOKENS * cfg.vocab_size,
                    "act_verify_logits",
                ),
            };

            // ----------------------------------------------------------------
            // GDN recurrent states (same as new())
            // ----------------------------------------------------------------
            let num_linear = cfg.num_active_linear_attention_layers();
            let num_full = cfg.num_active_full_attention_layers();
            let gdn_states: Vec<GatedDeltaNetState> = (0..num_linear)
                .map(|_| GatedDeltaNetState::new(cfg))
                .collect();
            let gdn_scratch = GatedDeltaNetFusedScratch::default();

            let num_value_heads = cfg.linear_num_value_heads();
            let key_dim = cfg.linear_key_head_dim;
            let value_dim = cfg.linear_value_head_dim;
            let qkv_dim = cfg.linear_qkv_dim();
            let buf_len = cfg.linear_conv_kernel_dim.saturating_sub(1);
            let gdn_gpu_conv_bufs: Vec<Buffer> = (0..num_linear)
                .map(|i| make_zero_buffer(&device, qkv_dim * buf_len, &format!("gdn_conv_{i}")))
                .collect();
            let gdn_gpu_s_matrices: Vec<Buffer> = (0..num_linear)
                .map(|i| {
                    make_zero_buffer(
                        &device,
                        num_value_heads * value_dim * key_dim,
                        &format!("gdn_s_{i}"),
                    )
                })
                .collect();
            let gdn_gpu_conv_out = make_zero_buffer(&device, qkv_dim, "gdn_conv_out");

            let kv_cache = MetalKvCache::new(&device, num_full, kv_dim, max_cache_len);

            let dur_d = t_d.elapsed();
            eprintln!(
                "[load-timer] Phase A (qkvz parallel I/O + Metal copy): {:.3}s",
                dur_a.as_secs_f64()
            );
            eprintln!(
                "[load-timer] Phase B (F16 buffer loads): {:.3}s",
                dur_b_cell.get().as_secs_f64()
            );
            eprintln!(
                "[load-timer] Phase C (mmap Q4 weight loads): {:.3}s",
                dur_c_cell.get().as_secs_f64()
            );
            eprintln!(
                "[load-timer] Phase D (embed/rope/activations/GDN): {:.3}s",
                dur_d.as_secs_f64()
            );
            eprintln!(
                "[load-timer] Total: {:.3}s",
                t_total.elapsed().as_secs_f64()
            );

            // tokenizer_path is accepted but not stored — tokenizer is loaded by the caller.
            let _ = tokenizer_path;

            // Load MTP weights (cache+activations go into session, not engine).
            let mtp_weights_opt = Self::load_mtp_q4_weights(q4_dir, cfg, &device);
            let mtp_session = mtp_weights_opt.as_ref().map(|_| {
                let cache = MetalMtpCache {
                    k_buf: make_zero_buffer(&device, max_cache_len * kv_dim, "mtp.kv_k"),
                    v_buf: make_zero_buffer(&device, max_cache_len * kv_dim, "mtp.kv_v"),
                    seq_len: 0,
                    max_cache_len,
                };
                let mtp_activations = MetalMtpActivations {
                    fused: make_zero_buffer(&device, 2 * hidden, "mtp.act_fused"),
                    hidden: make_zero_buffer(&device, hidden, "mtp.act_hidden"),
                    residual: make_zero_buffer(&device, hidden, "mtp.act_residual"),
                    q: make_zero_buffer(&device, 2 * q_dim, "mtp.act_q"),
                    q_separated: make_zero_buffer(&device, q_dim, "mtp.act_q_sep"),
                    gate_z: make_zero_buffer(&device, q_dim, "mtp.act_gate_z"),
                    k: make_zero_buffer(&device, kv_dim, "mtp.act_k"),
                    v: make_zero_buffer(&device, kv_dim, "mtp.act_v"),
                    attn_out: make_zero_buffer(&device, q_dim.max(hidden), "mtp.act_attn_out"),
                    gate: make_zero_buffer(&device, inter, "mtp.act_gate"),
                    up: make_zero_buffer(&device, inter, "mtp.act_up"),
                    ffn_out: make_zero_buffer(&device, hidden, "mtp.act_ffn_out"),
                    logits: make_zero_buffer(&device, cfg.vocab_size, "mtp.act_logits"),
                };
                MetalMtpSession {
                    cache,
                    activations: mtp_activations,
                }
            });
            let gdn_checkpoints = if mtp_weights_opt.is_some() {
                Some(MetalGdnCheckpointPool::new(
                    &device,
                    MTP_VERIFY_MAX_TOKENS,
                    &gdn_gpu_conv_bufs,
                    &gdn_gpu_s_matrices,
                ))
            } else {
                None
            };

            Ok(Self {
                engine: MetalQwen35Engine {
                    device,
                    queue,
                    pipelines,
                    layer_weights,
                    embed_tokens,
                    embed_tokens_q8,
                    final_norm,
                    rope_cos,
                    rope_sin,
                    config: cfg.clone(),
                    quant_format,
                    mtp_weights: mtp_weights_opt,
                },
                session: InferenceSession {
                    activations,
                    gdn_states,
                    gdn_scratch,
                    gdn_gpu_conv_bufs,
                    gdn_gpu_s_matrices,
                    gdn_gpu_conv_out,
                    kv_cache,
                    max_prefill,
                    compact_topk: 0,
                    compact_route: GpuTopkRoute::CpuFallback,
                    compact_result: Vec::new(),
                    mtp: mtp_session,
                    gdn_checkpoints,
                    last_pre_final_hidden: vec![0.0f32; hidden],
                    position: 0,
                },
                lora: None,
            })
        }

        /// **Unstable**: streaming chat completion; callback signature may change.
        ///
        /// Streaming chat completion with token-by-token callback.
        pub fn chat_completion_streaming<F>(
            &mut self,
            messages: &[ChatMessage],
            tokenizer: &BpeTokenizer,
            gen_cfg: &GenerateConfig,
            on_token: F,
        ) -> ChatCompletionOutput
        where
            F: FnMut(&str, u32) -> bool,
        {
            let prompt = format_chat_template(messages);
            let mut cfg = gen_cfg.clone();
            if let Some(im_end_id) = tokenizer.special_token_id("<|im_end|>") {
                if !cfg.stop_token_ids.contains(&im_end_id) {
                    cfg.stop_token_ids.push(im_end_id);
                }
            }
            let result = self.generate_streaming(&prompt, tokenizer, &cfg, on_token);
            let text = result.text.trim_end().to_string();
            ChatCompletionOutput {
                message: ChatMessage::assistant(text),
                prompt_tokens: result.prompt_tokens,
                completion_tokens: result.generated_tokens,
            }
        }
    }

    // -----------------------------------------------------------------------
    // Perplexity harness (ADR-044 step 4b)
    // -----------------------------------------------------------------------

    impl MetalQwen35State {
        /// **Unstable**: largest window length the Metal forward path can serve
        /// in a single call. Bounded by the KV-cache capacity chosen at session
        /// construction (`new_session(max_cache_len)` or
        /// `MetalQwen35State::new(_, _, max_cache_len)`), NOT by the RoPE table
        /// — `new_session` already rejects `max_cache_len >
        /// max_position_embeddings`, so the KV cap is always the tighter bound.
        /// Callers driving the perplexity harness must keep their effective
        /// window strictly at or below this value; `forward_step` panics
        /// otherwise (KV-cache full assertion).
        pub fn max_context(&self) -> usize {
            self.session.kv_cache.max_cache_len
        }

        /// **Unstable**: Metal Q4 sibling of [`crate::model::qwen35::Qwen35Model::compute_token_nlls`].
        ///
        /// Computes per-position cross-entropy NLLs for an autoregressive
        /// forward pass over `tokens` on the Metal forward path. Resets all
        /// recurrent state ([`MetalQwen35State::reset_state`]) before stepping
        /// so a single call does not depend on the caller's previous KV cache
        /// / GDN state, and subsequent calls do not contaminate each other —
        /// this is what makes the function safe to use as the per-window NLL
        /// kernel inside [`crate::model::qwen35::run_strided_perplexity`].
        ///
        /// Returns a `Vec<f32>` of length `tokens.len() - 1`. The value at
        /// index `i` is `-log p(tokens[i + 1] | tokens[0..=i])`, where the
        /// softmax runs over the full vocabulary at decode step `i`.
        ///
        /// Errors if `tokens.len() < 2`, if `tokens.len() > self.max_context()`
        /// (the KV cache would overflow during the walk), or if any
        /// `token_id >= cfg.vocab_size`.
        pub fn compute_token_nlls(
            &mut self,
            tokens: &[u32],
        ) -> Result<Vec<f32>, crate::error::InferenceError> {
            if tokens.len() < 2 {
                return Err(crate::error::InferenceError::Inference(format!(
                    "compute_token_nlls: need at least 2 tokens, got {}",
                    tokens.len()
                )));
            }
            let max_context = self.max_context();
            if tokens.len() > max_context {
                return Err(crate::error::InferenceError::Inference(format!(
                    "compute_token_nlls: tokens.len() ({}) exceeds Metal KV-cache capacity ({}); \
                     use a shorter window or rebuild the session with a larger max_cache_len",
                    tokens.len(),
                    max_context
                )));
            }
            let vocab_size = self.engine.config.vocab_size;
            if let Some((bad_idx, &bad)) = tokens
                .iter()
                .enumerate()
                .find(|&(_, &t)| (t as usize) >= vocab_size)
            {
                return Err(crate::error::InferenceError::Inference(format!(
                    "compute_token_nlls: tokens[{bad_idx}]={bad} >= vocab_size {vocab_size}"
                )));
            }

            self.reset_state();

            let mut nlls = Vec::with_capacity(tokens.len() - 1);
            for (pos, &token_id) in tokens.iter().enumerate() {
                let logits = self.forward_step(token_id, pos);
                if pos + 1 < tokens.len() {
                    let next = tokens[pos + 1] as usize;
                    nlls.push(crate::model::qwen35::log_softmax_nll(
                        &logits[..vocab_size],
                        next,
                    ));
                }
            }

            Ok(nlls)
        }

        /// **Unstable**: Metal Q4 sibling of [`crate::model::qwen35::Qwen35Model::compute_perplexity`].
        ///
        /// Thin wrapper around [`crate::model::qwen35::run_strided_perplexity`]
        /// driving the shared aggregator with this state's
        /// [`Self::compute_token_nlls`] as the per-window NLL kernel. Each
        /// window resets recurrent state, so context never crosses a window
        /// boundary — identical semantics to the CPU path.
        ///
        /// Use this on a [`MetalQwen35State`] loaded via [`Self::from_q4_dir`]
        /// to measure unrotated-Q4 PPL or rotated-Q4 PPL (the output of
        /// `bin/quantize_quarot`); the rotated-vs-unrotated delta is the
        /// acceptance number for ADR-044 step 4.
        pub fn compute_perplexity(
            &mut self,
            tokens: &[u32],
            cfg: &crate::model::qwen35::PerplexityConfig,
        ) -> Result<crate::model::qwen35::PerplexityReport, crate::error::InferenceError> {
            let max_ctx = self.max_context();
            crate::model::qwen35::run_strided_perplexity(tokens, cfg, max_ctx, |slice| {
                self.compute_token_nlls(slice)
            })
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::model::qwen35::{
            CommonLayerWeights, DenseFfnWeights, FeedForwardWeights, FullAttentionLayerWeights,
        };
        use crate::model::qwen35_config::LayerType;

        #[test]
        fn test_f32_f16_roundtrip_basic() {
            // Verify f32 -> f16 -> f32 roundtrip for typical weight values
            let test_values: Vec<f32> = vec![
                0.0, 1.0, -1.0, 0.5, -0.5, 0.001, -0.001, 0.25, 0.75, 1.5, -1.5, 65504.0, -65504.0,
                // Small subnormal range
                5.96e-8, -5.96e-8,
            ];

            for &v in &test_values {
                let bits = f32_to_f16(v);
                let back = f16_to_f32(bits);
                if v == 0.0 {
                    assert_eq!(back, 0.0, "zero roundtrip failed");
                } else {
                    let rel_err = ((back - v) / v).abs();
                    assert!(
                        rel_err < 0.002,
                        "roundtrip error for {v}: got {back}, rel_err={rel_err}"
                    );
                }
            }
        }

        #[test]
        fn test_f32_f16_special_values() {
            // Infinity
            let inf_bits = f32_to_f16(f32::INFINITY);
            assert_eq!(inf_bits, 0x7c00, "positive infinity");
            let neg_inf_bits = f32_to_f16(f32::NEG_INFINITY);
            assert_eq!(neg_inf_bits, 0xfc00, "negative infinity");

            // NaN
            let nan_bits = f32_to_f16(f32::NAN);
            assert_eq!(nan_bits & 0x7c00, 0x7c00, "NaN exponent");
            assert_ne!(nan_bits & 0x03ff, 0, "NaN mantissa nonzero");
            let back = f16_to_f32(nan_bits);
            assert!(back.is_nan(), "NaN roundtrip");

            // Signed zero
            let pos_zero = f32_to_f16(0.0f32);
            let neg_zero = f32_to_f16(-0.0f32);
            assert_eq!(pos_zero, 0x0000);
            assert_eq!(neg_zero, 0x8000);
        }

        #[test]
        fn test_f32_f16_overflow_underflow() {
            // Values too large for f16 -> infinity
            let big = f32_to_f16(100_000.0);
            assert_eq!(big, 0x7c00, "overflow to infinity");

            // Values too small for f16 -> zero
            let tiny = f32_to_f16(1e-20);
            assert_eq!(tiny, 0x0000, "underflow to zero");
        }

        #[test]
        fn test_f32_f16_weight_distribution() {
            // Simulate typical weight distribution (normal, mean 0, std ~0.02)
            let mut rng = 12345u64;
            let weights: Vec<f32> = (0..1000)
                .map(|_| {
                    // Simple Box-Muller approximation using xorshift
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let u = (rng as f64) / (u64::MAX as f64);
                    (u * 2.0 - 1.0) as f32 * 0.05
                })
                .collect();

            let mut max_err: f32 = 0.0;
            for &w in &weights {
                let bits = f32_to_f16(w);
                let back = f16_to_f32(bits);
                let err = (back - w).abs();
                max_err = max_err.max(err);
            }

            // For weights in [-0.05, 0.05], f16 should be accurate to ~1e-5
            assert!(
                max_err < 1e-4,
                "max roundtrip error for typical weights: {max_err}"
            );
        }

        #[test]
        fn test_read_buffer_f16_conversion() {
            // Verify read_buffer_f16 produces correct f32 values
            let device = match Device::system_default() {
                Some(d) => d,
                None => return, // skip on non-Metal systems
            };

            let test_f32 = vec![1.0f32, -0.5, 0.25, 0.0, 100.0];
            let buf = make_buffer_f16(&device, &test_f32, "test_f16");
            // SAFETY: The test buffer was just allocated from test_f32 and has
            // exactly test_f32.len() f16 elements in StorageModeShared memory.
            let recovered = unsafe { read_buffer_f16(&buf, test_f32.len()) };

            for (i, (&orig, &rec)) in test_f32.iter().zip(recovered.iter()).enumerate() {
                if orig == 0.0 {
                    assert_eq!(rec, 0.0, "index {i}: zero");
                } else {
                    let rel_err = ((rec - orig) / orig).abs();
                    assert!(
                        rel_err < 0.002,
                        "index {i}: orig={orig}, recovered={rec}, rel_err={rel_err}"
                    );
                }
            }
        }

        #[test]
        fn test_gemv_decode_kernel_compiles() {
            // Verify the MSL source compiles and GEMV pipeline is creatable.
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };

            let opts = CompileOptions::new();
            let library = device
                .new_library_with_source(MSL_SOURCE, &opts)
                .expect("MSL compilation failed");

            // Both matmul kernels should exist
            let _f32_fn = library
                .get_function("matmul_bt", None)
                .expect("matmul_bt not found");
            let _gemv_fn = library
                .get_function("gemv_decode_m1", None)
                .expect("gemv_decode_m1 not found");

            // Both should create valid pipelines
            let _f32_pipe = device
                .new_compute_pipeline_state_with_function(&_f32_fn)
                .expect("matmul_bt pipeline failed");
            let _gemv_pipe = device
                .new_compute_pipeline_state_with_function(&_gemv_fn)
                .expect("gemv_decode_m1 pipeline failed");
        }

        #[test]
        fn test_gemv_decode_numerical() {
            // Run a small GEMM through both f32 and f16 paths, compare results.
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };

            let opts = CompileOptions::new();
            let library = device
                .new_library_with_source(MSL_SOURCE, &opts)
                .expect("MSL compile");

            let f32_fn = library
                .get_function("matmul_bt", None)
                .expect("test setup: matmul_bt kernel exists");
            let gemv_fn = library
                .get_function("gemv_decode_m1", None)
                .expect("test setup: gemv_decode_m1 kernel exists");
            let pipe_f32 = device
                .new_compute_pipeline_state_with_function(&f32_fn)
                .expect("test setup: matmul_bt pipeline builds");
            let pipe_gemv = device
                .new_compute_pipeline_state_with_function(&gemv_fn)
                .expect("test setup: gemv_decode_m1 pipeline builds");

            let queue = device.new_command_queue();

            // A[1, 128], B[64, 128] -> C[1, 64]
            let m = 1u32;
            let n = 64u32;
            let k = 128u32;

            // Generate deterministic test data
            let mut rng = 42u64;
            let next_f = |rng: &mut u64| -> f32 {
                *rng ^= *rng << 13;
                *rng ^= *rng >> 7;
                *rng ^= *rng << 17;
                ((*rng as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32 * 0.1
            };

            let a_data: Vec<f32> = (0..(m * k) as usize).map(|_| next_f(&mut rng)).collect();
            let b_data: Vec<f32> = (0..(n * k) as usize).map(|_| next_f(&mut rng)).collect();

            // Create buffers
            let a_buf = make_buffer(&device, &a_data, "A");
            let b_buf_f32 = make_buffer(&device, &b_data, "B_f32");
            let b_buf_f16 = make_buffer_f16(&device, &b_data, "B_f16");
            let c_buf_f32 = make_zero_buffer(&device, (m * n) as usize, "C_f32");
            let c_buf_f16 = make_zero_buffer(&device, (m * n) as usize, "C_f16");

            // Run f32 path (matmul_bt, tiled)
            let cmd = queue.new_command_buffer();
            {
                let tile = 16u64;
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipe_f32);
                enc.set_buffer(0, Some(&a_buf), 0);
                enc.set_buffer(1, Some(&b_buf_f32), 0);
                enc.set_buffer(2, Some(&c_buf_f32), 0);
                enc.set_bytes(3, 4, &m as *const u32 as *const _);
                enc.set_bytes(4, 4, &n as *const u32 as *const _);
                enc.set_bytes(5, 4, &k as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(
                        (n as u64 + tile - 1) / tile,
                        (m as u64 + tile - 1) / tile,
                        1,
                    ),
                    MTLSize::new(tile, tile, 1),
                );
                enc.end_encoding();
            }
            cmd.commit();
            cmd.wait_until_completed();

            // Run GEMV f16 path (M=1)
            let cmd = queue.new_command_buffer();
            {
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipe_gemv);
                enc.set_buffer(0, Some(&a_buf), 0);
                enc.set_buffer(1, Some(&b_buf_f16), 0);
                enc.set_buffer(2, Some(&c_buf_f16), 0);
                let params = GemmParams {
                    m,
                    n,
                    k,
                    lda: k,
                    ldb: k,
                    ldc: n,
                };
                enc.set_bytes(
                    3,
                    std::mem::size_of::<GemmParams>() as u64,
                    &params as *const GemmParams as *const _,
                );
                enc.dispatch_thread_groups(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
                enc.end_encoding();
            }
            cmd.commit();
            cmd.wait_until_completed();

            // Compare results
            // SAFETY: The command buffer completed and c_buf_f32 has m*n f32 values.
            let result_f32 = unsafe { read_buffer(&c_buf_f32, (m * n) as usize) };
            // SAFETY: The command buffer completed and c_buf_f16 has m*n f32 values.
            let result_f16 = unsafe { read_buffer(&c_buf_f16, (m * n) as usize) };

            let mut max_diff: f32 = 0.0;
            for (i, (&v32, &v16)) in result_f32.iter().zip(result_f16.iter()).enumerate() {
                let diff = (v32 - v16).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                // Per-element tolerance: half precision with K=128 accumulation
                // can introduce error ~K * eps_f16 * max_weight ~= 128 * 0.001 * 0.1 = 0.01
                assert!(
                    diff < 0.05,
                    "element {i}: f32={v32}, f16={v16}, diff={diff}"
                );
            }

            // Overall max diff should be small
            assert!(
                max_diff < 0.05,
                "max diff between f32 and f16 GEMM: {max_diff}"
            );
        }

        #[test]
        fn test_make_buffer_f16_halves_size() {
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };

            let data = vec![1.0f32; 1024];
            let buf_f32 = make_buffer(&device, &data, "f32");
            let buf_f16 = make_buffer_f16(&device, &data, "f16");

            assert_eq!(buf_f32.length(), 1024 * 4); // 4 bytes per f32
            assert_eq!(buf_f16.length(), 1024 * 2); // 2 bytes per f16
        }

        // -----------------------------------------------------------------------
        // decode_attention parity tests
        //
        // Compare the old 3-pass reference kernel against the new flash GQA kernel
        // for all 6 required configurations and key edge cases.
        //
        // Tolerance gate (from optimization_plan.md §5):
        //   max_abs_diff <= 5e-2, mean_abs_diff <= 5e-3
        // Measured in optimizations.md: max_abs_diff < 1e-7 for all 6 configs.
        // -----------------------------------------------------------------------

        /// Old 3-pass decode kernel preserved as the correctness reference.
        const MSL_DECODE_REFERENCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void decode_attention_reference(
    device const float* q        [[buffer(0)]],
    device const float* k_cache  [[buffer(1)]],
    device const float* v_cache  [[buffer(2)]],
    device float* out            [[buffer(3)]],
    constant uint& cache_len     [[buffer(4)]],
    constant uint& head_dim      [[buffer(5)]],
    constant uint& num_q_heads   [[buffer(6)]],
    constant uint& num_kv_heads  [[buffer(7)]],
    constant uint& q_dim         [[buffer(8)]],
    constant uint& kv_dim        [[buffer(9)]],
    constant float& scale        [[buffer(10)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    if (gid >= num_q_heads) return;
    if (cache_len == 0) return;
    constexpr uint ATTN_WG = 256;
    const uint qh = gid;
    const uint kvh = qh / (num_q_heads / num_kv_heads);
    device const float* q_head = q + qh * head_dim;
    threadgroup float shared[ATTN_WG];

    float local_max = -1e30f;
    for (uint t = lid; t < cache_len; t += tgs) {
        device const float* k_t = k_cache + t * kv_dim + kvh * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) dot += q_head[d] * k_t[d];
        local_max = max(local_max, dot * scale);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] = max(shared[lid], shared[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];

    float local_exp_sum = 0.0f;
    for (uint t = lid; t < cache_len; t += tgs) {
        device const float* k_t = k_cache + t * kv_dim + kvh * head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) dot += q_head[d] * k_t[d];
        local_exp_sum += exp(dot * scale - max_val);
    }
    shared[lid] = local_exp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_exp = shared[0];
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    device float* out_head = out + qh * head_dim;
    for (uint d = lid; d < head_dim; d += tgs) {
        float acc = 0.0f;
        for (uint t = 0; t < cache_len; t++) {
            device const float* k_t = k_cache + t * kv_dim + kvh * head_dim;
            float dot = 0.0f;
            for (uint dd = 0; dd < head_dim; dd++) dot += q_head[dd] * k_t[dd];
            acc += exp(dot * scale - max_val) * inv_sum
                 * v_cache[t * kv_dim + kvh * head_dim + d];
        }
        out_head[d] = acc;
    }
}
"#;

        fn xorshift32_parity(state: &mut u32) -> f32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            (x as f32 / u32::MAX as f32) - 0.5
        }

        /// Run old reference kernel and new flash kernel on identical inputs.
        /// Returns `(max_abs_diff, mean_abs_diff, nan_count)`.
        fn parity_check(
            device: &Device,
            queue: &CommandQueue,
            pipe_ref: &ComputePipelineState,
            pipe_flash: &ComputePipelineState,
            pipe_partial: &ComputePipelineState,
            pipe_reduce: &ComputePipelineState,
            num_q_heads: u32,
            num_kv_heads: u32,
            cache_len: u32,
        ) -> (f32, f32, usize) {
            const HEAD_DIM: u32 = 256;
            const PARTITION_TOKENS: u32 = 1024;
            const DIRECT_THRESHOLD: u32 = 512;
            let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
            let q_dim = num_q_heads * HEAD_DIM;
            let kv_dim = num_kv_heads * HEAD_DIM;
            let num_partitions = cache_len.div_ceil(PARTITION_TOKENS);

            let q_buf =
                device.new_buffer((q_dim as u64) * 4, MTLResourceOptions::StorageModeShared);
            let k_buf = device.new_buffer(
                (cache_len as u64) * (kv_dim as u64) * 4,
                MTLResourceOptions::StorageModeShared,
            );
            let v_buf = device.new_buffer(
                (cache_len as u64) * (kv_dim as u64) * 4,
                MTLResourceOptions::StorageModeShared,
            );
            let out_ref =
                device.new_buffer((q_dim as u64) * 4, MTLResourceOptions::StorageModeShared);
            let out_flash =
                device.new_buffer((q_dim as u64) * 4, MTLResourceOptions::StorageModeShared);
            let partials_buf = device.new_buffer(
                (num_partitions as u64) * (num_q_heads as u64) * ((HEAD_DIM + 2) as u64) * 4,
                MTLResourceOptions::StorageModeShared,
            );

            let mut rng: u32 = 0xDEAD_BEEF;
            // SAFETY: buffers are StorageModeShared, freshly allocated, correctly sized.
            unsafe {
                let q =
                    std::slice::from_raw_parts_mut(q_buf.contents() as *mut f32, q_dim as usize);
                for v in q.iter_mut() {
                    *v = xorshift32_parity(&mut rng);
                }
                let k = std::slice::from_raw_parts_mut(
                    k_buf.contents() as *mut f32,
                    (cache_len * kv_dim) as usize,
                );
                for v in k.iter_mut() {
                    *v = xorshift32_parity(&mut rng);
                }
                let vd = std::slice::from_raw_parts_mut(
                    v_buf.contents() as *mut f32,
                    (cache_len * kv_dim) as usize,
                );
                for v in vd.iter_mut() {
                    *v = xorshift32_parity(&mut rng);
                }
            }

            // Reference: one threadgroup per Q head
            {
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pipe_ref);
                enc.set_buffer(0, Some(&q_buf), 0);
                enc.set_buffer(1, Some(&k_buf), 0);
                enc.set_buffer(2, Some(&v_buf), 0);
                enc.set_buffer(3, Some(&out_ref), 0);
                enc.set_bytes(4, 4, &cache_len as *const u32 as *const _);
                enc.set_bytes(5, 4, &HEAD_DIM as *const u32 as *const _);
                enc.set_bytes(6, 4, &num_q_heads as *const u32 as *const _);
                enc.set_bytes(7, 4, &num_kv_heads as *const u32 as *const _);
                enc.set_bytes(8, 4, &q_dim as *const u32 as *const _);
                enc.set_bytes(9, 4, &kv_dim as *const u32 as *const _);
                enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_q_heads as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }

            // Flash: direct (cache_len <= 512) or partitioned (cache_len > 512)
            {
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                if cache_len <= DIRECT_THRESHOLD {
                    enc.set_compute_pipeline_state(pipe_flash);
                    enc.set_buffer(0, Some(&q_buf), 0);
                    enc.set_buffer(1, Some(&k_buf), 0);
                    enc.set_buffer(2, Some(&v_buf), 0);
                    enc.set_buffer(3, Some(&out_flash), 0);
                    enc.set_bytes(4, 4, &cache_len as *const u32 as *const _);
                    enc.set_bytes(5, 4, &HEAD_DIM as *const u32 as *const _);
                    enc.set_bytes(6, 4, &num_q_heads as *const u32 as *const _);
                    enc.set_bytes(7, 4, &num_kv_heads as *const u32 as *const _);
                    enc.set_bytes(8, 4, &q_dim as *const u32 as *const _);
                    enc.set_bytes(9, 4, &kv_dim as *const u32 as *const _);
                    enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_kv_heads as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    enc.end_encoding();
                } else {
                    enc.set_compute_pipeline_state(pipe_partial);
                    enc.set_buffer(0, Some(&q_buf), 0);
                    enc.set_buffer(1, Some(&k_buf), 0);
                    enc.set_buffer(2, Some(&v_buf), 0);
                    enc.set_buffer(3, Some(&partials_buf), 0);
                    enc.set_bytes(4, 4, &cache_len as *const u32 as *const _);
                    enc.set_bytes(5, 4, &HEAD_DIM as *const u32 as *const _);
                    enc.set_bytes(6, 4, &num_q_heads as *const u32 as *const _);
                    enc.set_bytes(7, 4, &num_kv_heads as *const u32 as *const _);
                    enc.set_bytes(8, 4, &q_dim as *const u32 as *const _);
                    enc.set_bytes(9, 4, &kv_dim as *const u32 as *const _);
                    enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                    enc.set_bytes(11, 4, &PARTITION_TOKENS as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_kv_heads as u64, num_partitions as u64, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    enc.set_compute_pipeline_state(pipe_reduce);
                    enc.set_buffer(0, Some(&partials_buf), 0);
                    enc.set_buffer(1, Some(&out_flash), 0);
                    enc.set_bytes(2, 4, &num_q_heads as *const u32 as *const _);
                    enc.set_bytes(3, 4, &num_kv_heads as *const u32 as *const _);
                    enc.set_bytes(4, 4, &num_partitions as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(num_kv_heads as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    enc.end_encoding();
                }
                cmd.commit();
                cmd.wait_until_completed();
            }

            // SAFETY: both command buffers completed; buffers are f32 StorageModeShared.
            unsafe {
                let a =
                    std::slice::from_raw_parts(out_ref.contents() as *const f32, q_dim as usize);
                let b =
                    std::slice::from_raw_parts(out_flash.contents() as *const f32, q_dim as usize);
                let mut max_d = 0.0f32;
                let mut sum_d = 0.0f32;
                let mut nans = 0usize;
                for (x, y) in a.iter().zip(b.iter()) {
                    if !y.is_finite() {
                        nans += 1;
                    }
                    let d = (x - y).abs();
                    if d > max_d {
                        max_d = d;
                    }
                    sum_d += d;
                }
                (max_d, sum_d / a.len() as f32, nans)
            }
        }

        /// Compile all three flash pipelines and the reference pipeline once.
        /// Returns None on non-Metal systems (test is skipped).
        fn build_parity_pipelines(
            device: &Device,
        ) -> Option<(
            ComputePipelineState,
            ComputePipelineState,
            ComputePipelineState,
            ComputePipelineState,
        )> {
            let opts = CompileOptions::new();
            let lib_ref = device
                .new_library_with_source(MSL_DECODE_REFERENCE, &opts)
                .ok()?;
            let fn_ref = lib_ref
                .get_function("decode_attention_reference", None)
                .ok()?;
            let pipe_ref = device
                .new_compute_pipeline_state_with_function(&fn_ref)
                .ok()?;

            let lib_flash = device
                .new_library_with_source(MSL_SOURCE, &opts)
                .expect("MSL_SOURCE must compile");
            let pipe_flash = device
                .new_compute_pipeline_state_with_function(
                    &lib_flash
                        .get_function("decode_attention", None)
                        .expect("decode_attention not in MSL_SOURCE"),
                )
                .expect("decode_attention pipeline");
            let pipe_partial = device
                .new_compute_pipeline_state_with_function(
                    &lib_flash
                        .get_function("decode_attention_flash_partial", None)
                        .expect("decode_attention_flash_partial not in MSL_SOURCE"),
                )
                .expect("flash_partial pipeline");
            let pipe_reduce = device
                .new_compute_pipeline_state_with_function(
                    &lib_flash
                        .get_function("decode_attention_flash_reduce", None)
                        .expect("decode_attention_flash_reduce not in MSL_SOURCE"),
                )
                .expect("flash_reduce pipeline");

            Some((pipe_ref, pipe_flash, pipe_partial, pipe_reduce))
        }

        const STRICT_MAX_ABS_TOL: f32 = 1e-4;

        fn assert_decode_attention_parity_case(
            device: &Device,
            queue: &CommandQueue,
            pipes: &(
                ComputePipelineState,
                ComputePipelineState,
                ComputePipelineState,
                ComputePipelineState,
            ),
            num_q_heads: u32,
            num_kv_heads: u32,
            cache_len: u32,
        ) {
            let (pipe_ref, pipe_flash, pipe_partial, pipe_reduce) = pipes;
            let (max_d, mean_d, nans) = parity_check(
                device,
                queue,
                pipe_ref,
                pipe_flash,
                pipe_partial,
                pipe_reduce,
                num_q_heads,
                num_kv_heads,
                cache_len,
            );
            assert_eq!(
                nans, 0,
                "{num_q_heads}Q/{num_kv_heads}KV cache={cache_len}: NaN in flash output"
            );
            assert!(
                max_d < STRICT_MAX_ABS_TOL,
                "{num_q_heads}Q/{num_kv_heads}KV cache={cache_len}: max_abs_diff={max_d:.3e}, mean_abs_diff={mean_d:.3e}, tolerance={STRICT_MAX_ABS_TOL:.3e}"
            );
        }

        /// Strict parity gate: required matrix {128,512,2048} × {8Q/2KV,16Q/2KV}, tolerance 1e-4.
        #[test]
        fn test_decode_attention_parity_required_matrix() {
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };
            let queue = device.new_command_queue();
            let pipes = match build_parity_pipelines(&device) {
                Some(p) => p,
                None => return,
            };

            for &(num_q_heads, num_kv_heads) in &[(8u32, 2u32), (16u32, 2u32)] {
                for &cache_len in &[128u32, 512u32, 2048u32] {
                    assert_decode_attention_parity_case(
                        &device,
                        &queue,
                        &pipes,
                        num_q_heads,
                        num_kv_heads,
                        cache_len,
                    );
                }
            }
        }

        // --- 6 required configurations ---

        #[test]
        fn test_decode_attention_parity_8q2kv_512() {
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };
            let queue = device.new_command_queue();
            let (pr, pf, pp, prr) = match build_parity_pipelines(&device) {
                Some(p) => p,
                None => return,
            };
            let (max_d, _mean_d, nans) =
                parity_check(&device, &queue, &pr, &pf, &pp, &prr, 8, 2, 512);
            assert_eq!(nans, 0, "8Q/2KV cache=512: NaN in flash output");
            assert!(
                max_d < STRICT_MAX_ABS_TOL,
                "8Q/2KV cache=512: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
            );
        }

        #[test]
        fn test_decode_attention_parity_8q2kv_4096() {
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };
            let queue = device.new_command_queue();
            let (pr, pf, pp, prr) = match build_parity_pipelines(&device) {
                Some(p) => p,
                None => return,
            };
            let (max_d, _mean_d, nans) =
                parity_check(&device, &queue, &pr, &pf, &pp, &prr, 8, 2, 4096);
            assert_eq!(nans, 0, "8Q/2KV cache=4096: NaN in flash output");
            assert!(
                max_d < STRICT_MAX_ABS_TOL,
                "8Q/2KV cache=4096: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
            );
        }

        #[test]
        #[ignore = "slow: reference kernel ~721ms at cache_len=32768; run with --include-ignored"]
        fn test_decode_attention_parity_8q2kv_32768() {
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };
            let queue = device.new_command_queue();
            let (pr, pf, pp, prr) = match build_parity_pipelines(&device) {
                Some(p) => p,
                None => return,
            };
            let (max_d, _mean_d, nans) =
                parity_check(&device, &queue, &pr, &pf, &pp, &prr, 8, 2, 32768);
            assert_eq!(nans, 0, "8Q/2KV cache=32768: NaN in flash output");
            assert!(
                max_d < STRICT_MAX_ABS_TOL,
                "8Q/2KV cache=32768: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
            );
        }

        #[test]
        fn test_decode_attention_parity_16q2kv_512() {
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };
            let queue = device.new_command_queue();
            let (pr, pf, pp, prr) = match build_parity_pipelines(&device) {
                Some(p) => p,
                None => return,
            };
            let (max_d, _mean_d, nans) =
                parity_check(&device, &queue, &pr, &pf, &pp, &prr, 16, 2, 512);
            assert_eq!(nans, 0, "16Q/2KV cache=512: NaN in flash output");
            assert!(
                max_d < STRICT_MAX_ABS_TOL,
                "16Q/2KV cache=512: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
            );
        }

        #[test]
        fn test_decode_attention_parity_16q2kv_4096() {
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };
            let queue = device.new_command_queue();
            let (pr, pf, pp, prr) = match build_parity_pipelines(&device) {
                Some(p) => p,
                None => return,
            };
            let (max_d, _mean_d, nans) =
                parity_check(&device, &queue, &pr, &pf, &pp, &prr, 16, 2, 4096);
            assert_eq!(nans, 0, "16Q/2KV cache=4096: NaN in flash output");
            assert!(
                max_d < STRICT_MAX_ABS_TOL,
                "16Q/2KV cache=4096: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
            );
        }

        #[test]
        #[ignore = "slow: reference kernel ~723ms at cache_len=32768; run with --include-ignored"]
        fn test_decode_attention_parity_16q2kv_32768() {
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };
            let queue = device.new_command_queue();
            let (pr, pf, pp, prr) = match build_parity_pipelines(&device) {
                Some(p) => p,
                None => return,
            };
            let (max_d, _mean_d, nans) =
                parity_check(&device, &queue, &pr, &pf, &pp, &prr, 16, 2, 32768);
            assert_eq!(nans, 0, "16Q/2KV cache=32768: NaN in flash output");
            assert!(
                max_d < STRICT_MAX_ABS_TOL,
                "16Q/2KV cache=32768: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
            );
        }

        // --- Edge cases (from auditor static_analysis.md) ---

        /// cache_len=1: first-token decode. The only cached token gets weight 1.0,
        /// so flash output must equal the old reference exactly (same f32 arithmetic).
        #[test]
        fn test_decode_attention_edge_cache_len_1() {
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };
            let queue = device.new_command_queue();
            let (pr, pf, pp, prr) = match build_parity_pipelines(&device) {
                Some(p) => p,
                None => return,
            };
            // Use 8Q/2KV; first token is on the direct path (cache_len=1 <= 512)
            let (max_d, mean_d, nans) = parity_check(&device, &queue, &pr, &pf, &pp, &prr, 8, 2, 1);
            assert_eq!(nans, 0, "cache_len=1: NaN in output");
            // Tighter tolerance: single-token softmax = 1.0 exactly, output = V[0]
            assert!(
                max_d <= 1e-5,
                "cache_len=1: max_abs_diff={max_d:.3e} (expected near 0)"
            );
            assert!(
                mean_d <= 1e-6,
                "cache_len=1: mean_abs_diff={mean_d:.3e} (expected near 0)"
            );
        }

        /// cache_len=257: crosses the 256-token TILE_TOKENS boundary (direct path).
        #[test]
        fn test_decode_attention_edge_cache_len_257() {
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };
            let queue = device.new_command_queue();
            let (pr, pf, pp, prr) = match build_parity_pipelines(&device) {
                Some(p) => p,
                None => return,
            };
            let (max_d, _mean_d, nans) =
                parity_check(&device, &queue, &pr, &pf, &pp, &prr, 8, 2, 257);
            assert_eq!(nans, 0, "cache_len=257: NaN in output");
            assert!(
                max_d < STRICT_MAX_ABS_TOL,
                "cache_len=257: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
            );
        }

        /// cache_len=513: just above DIRECT_THRESHOLD — forces partitioned path.
        #[test]
        fn test_decode_attention_edge_cache_len_513() {
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };
            let queue = device.new_command_queue();
            let (pr, pf, pp, prr) = match build_parity_pipelines(&device) {
                Some(p) => p,
                None => return,
            };
            let (max_d, _mean_d, nans) =
                parity_check(&device, &queue, &pr, &pf, &pp, &prr, 8, 2, 513);
            assert_eq!(nans, 0, "cache_len=513: NaN in output");
            assert!(
                max_d < STRICT_MAX_ABS_TOL,
                "cache_len=513: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
            );
        }

        /// cache_len=1025: crosses PARTITION_TOKENS boundary in the partitioned path.
        #[test]
        fn test_decode_attention_edge_cache_len_1025() {
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };
            let queue = device.new_command_queue();
            let (pr, pf, pp, prr) = match build_parity_pipelines(&device) {
                Some(p) => p,
                None => return,
            };
            let (max_d, _mean_d, nans) =
                parity_check(&device, &queue, &pr, &pf, &pp, &prr, 8, 2, 1025);
            assert_eq!(nans, 0, "cache_len=1025: NaN in output");
            assert!(
                max_d < STRICT_MAX_ABS_TOL,
                "cache_len=1025: max_abs_diff={max_d:.3e} > {STRICT_MAX_ABS_TOL:.3e}"
            );
        }

        // ── Route-logic unit tests (pure Rust, no Metal device needed) ────────

        #[test]
        fn test_choose_gpu_topk_route_all_cases() {
            // k=0 → always CPU
            assert_eq!(
                choose_gpu_topk_route(0, true, true),
                GpuTopkRoute::CpuFallback,
                "k=0"
            );
            // compact_env=false → always CPU
            assert_eq!(
                choose_gpu_topk_route(1, false, false),
                GpuTopkRoute::CpuFallback,
                "k=1, compact_env=false"
            );
            assert_eq!(
                choose_gpu_topk_route(50, false, true),
                GpuTopkRoute::CpuFallback,
                "k=50, compact_env=false"
            );
            // k=1, compact_env=true → CpuFallback (GPU 2.70× slower than CPU NEON after R1)
            assert_eq!(
                choose_gpu_topk_route(1, true, false),
                GpuTopkRoute::CpuFallback,
                "k=1, compact_env=true, no select"
            );
            assert_eq!(
                choose_gpu_topk_route(1, true, true),
                GpuTopkRoute::CpuFallback,
                "k=1, compact_env=true, select"
            );
            // k=2..=64, compact_env=true, no selection_env → CpuFallback
            for k in [2usize, 10, 50, 64] {
                assert_eq!(
                    choose_gpu_topk_route(k, true, false),
                    GpuTopkRoute::CpuFallback,
                    "k={k}, no LATTICE_COMPACT_TOPK_SELECT"
                );
            }
            // k=50 with both env flags → HierarchicalK50 (R2 experimental).
            assert_eq!(
                choose_gpu_topk_route(50, true, true),
                GpuTopkRoute::HierarchicalK50,
                "k=50 with both env flags must use HierarchicalK50"
            );
            // Other k>1 values must not enter any GPU route.
            for k in [2usize, 10, 49, 51, 64, 65, 128, 256] {
                assert_eq!(
                    choose_gpu_topk_route(k, true, true),
                    GpuTopkRoute::CpuFallback,
                    "k={k} must not enter any GPU top-k route"
                );
            }
            // k>256 → always CpuFallback
            assert_eq!(
                choose_gpu_topk_route(257, true, true),
                GpuTopkRoute::CpuFallback,
                "k=257"
            );
            assert_eq!(
                choose_gpu_topk_route(1000, true, true),
                GpuTopkRoute::CpuFallback,
                "k=1000"
            );
        }

        // ── GPU-CPU argmax parity tests (requires Metal device) ──────────────

        #[test]
        fn test_gpu_argmax_parity_k1() {
            let device = match Device::system_default() {
                Some(d) => d,
                None => return,
            };
            let queue = device.new_command_queue();
            let opts = CompileOptions::new();
            let library = device
                .new_library_with_source(MSL_SOURCE, &opts)
                .expect("MSL compile");
            let first_fn = library
                .get_function("logits_argmax_first", None)
                .expect("logits_argmax_first");
            let merge_fn = library
                .get_function("logits_argmax_merge", None)
                .expect("logits_argmax_merge");
            let first_pipe = device
                .new_compute_pipeline_state_with_function(&first_fn)
                .expect("argmax_first pipeline");
            let merge_pipe = device
                .new_compute_pipeline_state_with_function(&merge_fn)
                .expect("argmax_merge pipeline");

            // Dispatch GPU argmax and return the winning token_id.
            let gpu_argmax = |logits: &[f32]| -> u32 {
                let vocab_size = logits.len() as u32;
                let groups = vocab_size.div_ceil(1024);
                let in_buf = make_buffer(&device, logits, "logits");
                // scratch_a: groups × TopKCandidate (8 bytes = float logit + uint token_id)
                let scratch_a_zeros: Vec<f32> = vec![0.0; (groups as usize) * 2];
                let scratch_a = make_buffer(&device, &scratch_a_zeros, "scratch_a");
                // scratch_b: 1 TopKCandidate
                let scratch_b = make_buffer(&device, &[0.0f32; 2], "scratch_b");

                let cmd = queue.new_command_buffer();
                {
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&first_pipe);
                    enc.set_buffer(0, Some(&in_buf), 0);
                    enc.set_buffer(1, Some(&scratch_a), 0);
                    enc.set_bytes(2, 4, &vocab_size as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(groups as u64, 1, 1),
                        MTLSize::new(1024, 1, 1),
                    );
                    enc.set_compute_pipeline_state(&merge_pipe);
                    enc.set_buffer(0, Some(&scratch_a), 0);
                    enc.set_buffer(1, Some(&scratch_b), 0);
                    enc.set_bytes(2, 4, &groups as *const u32 as *const _);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
                    enc.end_encoding();
                }
                cmd.commit();
                cmd.wait_until_completed();

                // SAFETY: command buffer completed; scratch_b layout is
                // TopKCandidate { float logit; uint token_id; } — token_id at bytes [4..8].
                unsafe {
                    let ptr = scratch_b.contents() as *const u8;
                    let token_id_bytes = std::slice::from_raw_parts(ptr.add(4), 4);
                    u32::from_le_bytes([
                        token_id_bytes[0],
                        token_id_bytes[1],
                        token_id_bytes[2],
                        token_id_bytes[3],
                    ])
                }
            };

            // Mirror of sampling.rs:argmax_f32 — NaN never beats a real value.
            let cpu_argmax = |logits: &[f32]| -> u32 {
                let mut best_idx = 0u32;
                let mut best_val = f32::NEG_INFINITY;
                for (i, &v) in logits.iter().enumerate() {
                    if v > best_val {
                        best_val = v;
                        best_idx = i as u32;
                    }
                }
                best_idx
            };

            struct Case {
                name: &'static str,
                logits: Vec<f32>,
                expect_gpu_eq_cpu: bool,
            }

            // When all values are NaN the CPU returns 0 (initial best_idx) but the
            // GPU returns UINT_MAX (sentinel token_id).  This is a documented
            // divergence; in practice logits never contain only NaN at inference time.
            let cases: Vec<Case> = vec![
                Case {
                    name: "single element",
                    logits: vec![1.0f32],
                    expect_gpu_eq_cpu: true,
                },
                Case {
                    name: "small random (5 elem)",
                    logits: vec![0.1f32, 0.9, 0.3, 0.7, 0.5],
                    expect_gpu_eq_cpu: true,
                },
                Case {
                    name: "sorted descending (max first)",
                    logits: (0u32..256).rev().map(|i| i as f32).collect(),
                    expect_gpu_eq_cpu: true,
                },
                Case {
                    name: "reverse sorted (max last)",
                    logits: (0u32..256).map(|i| i as f32).collect(),
                    expect_gpu_eq_cpu: true,
                },
                Case {
                    name: "all same values (tie → lowest id = 0)",
                    logits: vec![3.14f32; 4096],
                    expect_gpu_eq_cpu: true,
                },
                Case {
                    name: "full vocab 248320 random",
                    logits: {
                        let mut rng = 42u64;
                        (0..248_320)
                            .map(|_| {
                                rng ^= rng << 13;
                                rng ^= rng >> 7;
                                rng ^= rng << 17;
                                ((rng as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32
                            })
                            .collect()
                    },
                    expect_gpu_eq_cpu: true,
                },
                Case {
                    name: "partial NaN (non-NaN max wins)",
                    logits: vec![1.0f32, f32::NAN, 5.0, f32::NAN, 3.0],
                    expect_gpu_eq_cpu: true,
                },
                Case {
                    name: "-Inf values (finite max wins)",
                    logits: vec![f32::NEG_INFINITY, 2.0f32, f32::NEG_INFINITY, 1.0],
                    expect_gpu_eq_cpu: true,
                },
            ];

            let mut failures = Vec::new();
            for case in &cases {
                let cpu = cpu_argmax(&case.logits);
                let gpu = gpu_argmax(&case.logits);
                let matches = gpu == cpu;
                if case.expect_gpu_eq_cpu && !matches {
                    failures.push(format!("MISMATCH {}: cpu={cpu} gpu={gpu}", case.name));
                }
            }

            assert!(
                failures.is_empty(),
                "GPU-CPU argmax parity failures:\n{}",
                failures.join("\n")
            );
        }

        fn tiny_metal_qwen35_fixture() -> (Qwen35Config, ModelWeights) {
            let hidden = 512usize;
            let intermediate = 64usize;
            let vocab = 64usize;
            let cfg = Qwen35Config {
                hidden_size: hidden,
                num_hidden_layers: 1,
                vocab_size: vocab,
                intermediate_size: intermediate,
                rms_norm_eps: 1e-6,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                head_dim: 256,
                rope_theta: 10_000_000.0,
                partial_rotary_factor: 0.25,
                rope_parameters: None,
                linear_num_key_heads: 1,
                linear_num_value_heads: Some(1),
                linear_key_head_dim: 16,
                linear_value_head_dim: 16,
                linear_conv_kernel_dim: 4,
                num_experts: None,
                num_experts_per_tok: None,
                moe_intermediate_size: None,
                shared_expert_intermediate_size: None,
                output_router_logits: false,
                router_aux_loss_coef: None,
                tie_word_embeddings: true,
                mtp_num_hidden_layers: 0,
                mtp_use_dedicated_embeddings: false,
                full_attention_interval: 1,
                layer_types: vec![LayerType::FullAttention],
                layer_mask: vec![true],
                eos_token_id: (vocab - 1) as u32,
                max_position_embeddings: 128,
            };

            let mut embed_tokens = vec![0.0f32; vocab * hidden];
            for token in 0..vocab {
                embed_tokens[token * hidden] = match token % 3 {
                    0 => -1.0,
                    1 => 0.0,
                    _ => 1.0,
                };
            }
            embed_tokens[42 * hidden] = 1.0;

            let common = CommonLayerWeights {
                input_layernorm: vec![1.0; hidden],
                post_attention_layernorm: vec![1.0; hidden],
                ffn: FeedForwardWeights::Dense(DenseFfnWeights {
                    gate_proj: vec![0.0; intermediate * hidden],
                    up_proj: vec![0.0; intermediate * hidden],
                    down_proj: vec![0.0; hidden * intermediate],
                }),
            };

            let full = FullAttentionLayerWeights {
                q_proj: vec![0.0; 2 * cfg.full_q_dim() * hidden],
                k_proj: vec![0.0; cfg.full_kv_dim() * hidden],
                v_proj: vec![0.0; cfg.full_kv_dim() * hidden],
                o_proj: vec![0.0; hidden * cfg.full_q_dim()],
                q_norm: vec![1.0; cfg.head_dim],
                k_norm: vec![1.0; cfg.head_dim],
            };

            let weights = ModelWeights {
                embed_tokens,
                lm_head: None,
                final_norm: vec![1.0; hidden],
                layers: vec![(AttentionWeights::Full(full), common)],
            };
            (cfg, weights)
        }

        #[test]
        fn test_metal_qwen35_golden_logit_snapshot_forward_step_token_42_pos_0() {
            let Some(_) = Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 16)
                .expect("tiny MetalQwen35State fixture constructs");

            let logits = state.forward_step(42, 0);
            assert_eq!(logits.len(), cfg.vocab_size);
            let actual = &logits[..10];
            let expected = [
                -22.6216266_f32,
                0.0,
                22.6216266,
                -22.6216266,
                0.0,
                22.6216266,
                -22.6216266,
                0.0,
                22.6216266,
                -22.6216266,
            ];
            let max_abs_diff = actual
                .iter()
                .zip(expected.iter())
                .map(|(a, e)| (a - e).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_abs_diff < 1e-4,
                "golden first-10 logits changed: actual={actual:?} expected={expected:?} max_abs_diff={max_abs_diff}"
            );
        }

        #[test]
        fn test_metal_qwen35_kv_cache_determinism_replay_5_tokens() {
            let Some(_) = Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let tokens = [42_u32, 7, 13, 8, 42];
            let mut state = MetalQwen35State::new(&weights, &cfg, 16)
                .expect("tiny MetalQwen35State fixture constructs");

            // First pass
            let mut first_logits: Vec<Vec<f32>> = Vec::new();
            for &token in &tokens {
                let pos = state.session.kv_cache.seq_len;
                let logits = state.forward_step(token, pos);
                assert_eq!(
                    state.session.kv_cache.seq_len,
                    pos + 1,
                    "forward_step must advance seq_len exactly once"
                );
                first_logits.push(logits);
            }
            let first_seq_len = state.session.kv_cache.seq_len;

            state.reset_state();

            // Second pass (replay)
            let mut second_logits: Vec<Vec<f32>> = Vec::new();
            for &token in &tokens {
                let pos = state.session.kv_cache.seq_len;
                let logits = state.forward_step(token, pos);
                assert_eq!(
                    state.session.kv_cache.seq_len,
                    pos + 1,
                    "forward_step must advance seq_len exactly once"
                );
                second_logits.push(logits);
            }
            let second_seq_len = state.session.kv_cache.seq_len;

            assert_eq!(first_seq_len, tokens.len(), "first pass seq_len");
            assert_eq!(second_seq_len, tokens.len(), "second pass seq_len");
            for (step, (first, second)) in first_logits.iter().zip(second_logits.iter()).enumerate()
            {
                let max_abs_diff = first
                    .iter()
                    .zip(second.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    max_abs_diff < 1e-4,
                    "replay logits diverged at step {step}: max_abs_diff={max_abs_diff}"
                );
            }
        }

        #[test]
        fn test_metal_qwen35_engine_session_isolation() {
            let Some(_) = Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();

            // Build engine, then create two independent sessions from it.
            let engine = MetalQwen35Engine::new(&weights, &cfg)
                .expect("MetalQwen35Engine constructs from tiny fixture");
            let session_a = engine.new_session(16).expect("session_a constructs");
            let session_b = engine.new_session(16).expect("session_b constructs");

            // Both sessions start at position 0.
            assert_eq!(session_a.position, 0, "session_a starts at position 0");
            assert_eq!(session_b.position, 0, "session_b starts at position 0");
            assert_eq!(
                session_a.kv_cache.seq_len, 0,
                "session_a kv_cache starts empty"
            );
            assert_eq!(
                session_b.kv_cache.seq_len, 0,
                "session_b kv_cache starts empty"
            );

            // Wrap engine + session_a into a full state and run forward steps.
            let mut state_a = MetalQwen35State {
                engine,
                session: session_a,
                lora: None,
            };
            let _logits = state_a.forward_step(42, 0);
            let _logits = state_a.forward_step(7, 1);

            // session_a (inside state_a) has advanced.
            assert_eq!(
                state_a.session.kv_cache.seq_len, 2,
                "state_a advanced 2 steps"
            );

            // Create a fresh session from the same engine — must start at 0.
            let fresh_session = state_a
                .engine
                .new_session(16)
                .expect("fresh session constructs");
            assert_eq!(
                fresh_session.position, 0,
                "fresh session position independent of state_a"
            );
            assert_eq!(
                fresh_session.kv_cache.seq_len, 0,
                "fresh session kv_cache independent"
            );

            // session_b was never touched; its state should match a fresh session.
            assert_eq!(
                session_b.position, 0,
                "session_b untouched by state_a steps"
            );
            assert_eq!(
                session_b.kv_cache.seq_len, 0,
                "session_b kv_cache untouched"
            );
        }

        // -------------------------------------------------------------------
        // Perplexity harness (ADR-044 step 4b)
        // -------------------------------------------------------------------

        #[test]
        fn metal_compute_token_nlls_matches_manual_forward_loop() {
            let Some(_) = Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 16)
                .expect("tiny MetalQwen35State fixture constructs");
            let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];

            // Reference: manual reset + forward loop + log_softmax matches what
            // compute_token_nlls advertises.
            state.reset_state();
            let mut expected = Vec::with_capacity(tokens.len() - 1);
            for (pos, &tok) in tokens.iter().enumerate() {
                let logits = state.forward_step(tok, pos);
                if pos + 1 < tokens.len() {
                    expected.push(crate::model::qwen35::log_softmax_nll(
                        &logits[..cfg.vocab_size],
                        tokens[pos + 1] as usize,
                    ));
                }
            }

            // Subject under test — resets internally, returns identical NLLs.
            let got = state
                .compute_token_nlls(&tokens)
                .expect("compute_token_nlls ok on valid tokens");
            assert_eq!(got.len(), tokens.len() - 1);
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (g - e).abs() < 1e-4,
                    "metal compute_token_nlls[{i}] diverged from manual loop: got={g} expected={e}"
                );
            }
        }

        #[test]
        fn metal_compute_token_nlls_is_repeatable_under_self_reset() {
            // The Metal harness resets recurrent state at the start of each
            // call. Two back-to-back calls on the same input must therefore
            // produce identical NLLs even though `forward_step` mutates the
            // session's KV cache and GDN state in place.
            let Some(_) = Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 16)
                .expect("tiny MetalQwen35State fixture constructs");
            let tokens: Vec<u32> = vec![1, 7, 13, 8, 42];

            let first = state.compute_token_nlls(&tokens).expect("first call ok");
            let second = state.compute_token_nlls(&tokens).expect("second call ok");
            assert_eq!(first.len(), second.len());
            for (i, (a, b)) in first.iter().zip(second.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "back-to-back NLLs diverged at {i}: {a} vs {b}"
                );
            }
        }

        #[test]
        fn metal_compute_token_nlls_rejects_oversized_input() {
            let Some(_) = Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            // KV cap = 4. compute_token_nlls(tokens) with tokens.len() > 4 must
            // refuse — would otherwise overflow the cache mid-window.
            let mut state = MetalQwen35State::new(&weights, &cfg, 4)
                .expect("tiny MetalQwen35State fixture constructs with max_cache_len=4");
            let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
            let err = state
                .compute_token_nlls(&tokens)
                .expect_err("tokens.len() > max_cache_len must error");
            let msg = format!("{err}");
            assert!(
                msg.contains("KV-cache") && msg.contains('4'),
                "error must name the KV-cache cap; got: {msg}"
            );
        }

        #[test]
        fn metal_compute_token_nlls_rejects_out_of_vocab_token() {
            let Some(_) = Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 16)
                .expect("tiny MetalQwen35State fixture constructs");
            let vocab = cfg.vocab_size as u32;
            let err = state
                .compute_token_nlls(&[1, vocab, 3])
                .expect_err("out-of-vocab token must error");
            let msg = format!("{err}");
            assert!(
                msg.contains(&format!("{vocab}")),
                "error must name the bad token id; got: {msg}"
            );
        }

        #[test]
        fn metal_compute_perplexity_equals_exp_mean_nll() {
            let Some(_) = Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 16)
                .expect("tiny MetalQwen35State fixture constructs");
            let tokens: Vec<u32> = (1u32..=12).collect();
            let report = state
                .compute_perplexity(
                    &tokens,
                    &crate::model::qwen35::PerplexityConfig {
                        window: 6,
                        stride: 3,
                    },
                )
                .expect("ppl ok");
            let expected_ppl = report.mean_nll.exp();
            assert!(
                (report.ppl - expected_ppl).abs() < 1e-9,
                "ppl must equal exp(mean_nll); got ppl={}, exp(mean_nll)={}",
                report.ppl,
                expected_ppl
            );
            assert!(
                (report.mean_nll - report.total_nll / report.num_tokens_scored as f64).abs() < 1e-9,
                "mean_nll must equal total_nll / num_tokens_scored"
            );
            assert!(report.ppl.is_finite() && report.ppl > 0.0);
        }

        #[test]
        fn metal_compute_perplexity_matches_compute_token_nlls_on_single_window() {
            let Some(_) = Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 16)
                .expect("tiny MetalQwen35State fixture constructs");
            let tokens: Vec<u32> = (1u32..=8).collect();
            // window covers the whole corpus → strided harness collapses to a
            // single window and the aggregated NLL must equal the direct sum.
            let direct_nlls = state.compute_token_nlls(&tokens).expect("direct ok");
            let direct_total: f64 = direct_nlls.iter().map(|&x| x as f64).sum();

            let report = state
                .compute_perplexity(
                    &tokens,
                    &crate::model::qwen35::PerplexityConfig {
                        window: tokens.len(),
                        stride: tokens.len() - 1,
                    },
                )
                .expect("ppl ok");
            assert_eq!(report.num_windows, 1);
            assert_eq!(report.num_tokens_scored, tokens.len() - 1);
            assert!(
                (report.total_nll - direct_total).abs() < 1e-5,
                "single-window strided PPL must agree with compute_token_nlls sum: got {} vs {}",
                report.total_nll,
                direct_total
            );
        }

        #[test]
        fn metal_compute_perplexity_rejects_window_above_max_cache_len() {
            // Counterpart of the CPU test:
            // `compute_perplexity_rejects_window_above_rope_capacity`. The
            // Metal-side cap is the KV-cache size, not the RoPE table.
            let Some(_) = Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let max_cache_len = 8usize;
            let mut state = MetalQwen35State::new(&weights, &cfg, max_cache_len)
                .expect("tiny MetalQwen35State fixture constructs");
            // Long enough corpus that the effective window > max_cache_len.
            let tokens: Vec<u32> = (1..=8).cycle().take(max_cache_len + 4).collect();
            let err = state
                .compute_perplexity(
                    &tokens,
                    &crate::model::qwen35::PerplexityConfig {
                        window: max_cache_len + 1,
                        stride: 4,
                    },
                )
                .expect_err("window > max_cache_len must error");
            let msg = format!("{err}");
            assert!(
                msg.contains("context capacity") && msg.contains(&format!("{max_cache_len}")),
                "error must name backend context capacity + cap; got: {msg}"
            );
        }

        // -------------------------------------------------------------------
        // from_q4_dir refuse-on-misconfig (round-1 codex findings on PR #32)
        // -------------------------------------------------------------------

        #[test]
        fn from_q4_dir_rejects_max_cache_len_above_max_position_embeddings() {
            // Round-1 codex Major: `from_q4_dir` was missing the
            // `max_cache_len <= cfg.max_position_embeddings` guard that
            // `new_session` enforces. Without it, `--max-cache-len 999999
            // --window 999999` would slip past the strided-PPL aggregator
            // (which only knows the KV cap) and panic / read out-of-bounds
            // GPU memory in `partial_rope_interleaved`. The loader now
            // matches `new_session` and refuses up-front.
            let Some(_) = Device::system_default() else {
                return;
            };
            let (mut cfg, _weights) = tiny_metal_qwen35_fixture();
            cfg.max_position_embeddings = 64; // small RoPE table
            let tmp =
                std::env::temp_dir().join(format!("lattice-q4-test-mpe-{}", std::process::id()));
            // The validator runs before any file I/O so an empty / non-
            // existent dir is fine for this regression.
            let err = match MetalQwen35State::from_q4_dir(
                &tmp,
                std::path::Path::new("/dev/null"),
                &cfg,
                1024,
            ) {
                Ok(_) => panic!("max_cache_len > max_position_embeddings must error"),
                Err(e) => e,
            };
            assert!(
                err.contains("max_cache_len") && err.contains("max_position_embeddings"),
                "error must name both bounds; got: {err}"
            );
        }

        #[test]
        fn from_q4_dir_rejects_zero_max_cache_len() {
            let Some(_) = Device::system_default() else {
                return;
            };
            let (cfg, _weights) = tiny_metal_qwen35_fixture();
            let tmp =
                std::env::temp_dir().join(format!("lattice-q4-test-zero-{}", std::process::id()));
            let err = match MetalQwen35State::from_q4_dir(
                &tmp,
                std::path::Path::new("/dev/null"),
                &cfg,
                0,
            ) {
                Ok(_) => panic!("max_cache_len = 0 must error"),
                Err(e) => e,
            };
            assert!(
                err.contains("max_cache_len"),
                "error must name max_cache_len; got: {err}"
            );
        }

        #[test]
        fn from_q4_dir_rejects_tied_config_with_lm_head_artifact() {
            // Round-2 codex Blocker: the round-1 fix only caught the
            // `tie_word_embeddings=false` + missing `lm_head.q4` half of
            // the artifact contract. The opposite mismatch —
            // `tie_word_embeddings=true` but a `lm_head_weight.q4`
            // present on disk — was still accepted, with the loader
            // silently ignoring the materialized head (the converter's
            // config-flip was likely skipped because HF Qwen3.5 stores
            // the flag in both nested `text_config` and the top level).
            // The loader now refuses both directions before any tensor
            // I/O.
            let Some(_) = Device::system_default() else {
                return;
            };
            let (cfg, _weights) = tiny_metal_qwen35_fixture();
            assert!(
                cfg.tie_word_embeddings,
                "tiny fixture must be tied for this test"
            );
            let tmp =
                std::env::temp_dir().join(format!("lattice-q4-test-tied-{}", std::process::id()));
            std::fs::create_dir_all(&tmp).expect("tempdir create");
            // Drop a placeholder lm_head_weight.q4 in the dir; the file
            // contents don't matter — the preflight only checks
            // existence, before any loader touches the bytes.
            let lm_head_file = tmp.join("lm_head_weight.q4");
            std::fs::write(&lm_head_file, b"placeholder").expect("write lm_head placeholder");

            let err = match MetalQwen35State::from_q4_dir(
                &tmp,
                std::path::Path::new("/dev/null"),
                &cfg,
                16,
            ) {
                Ok(_) => panic!("tied config with stray lm_head.q4 must error"),
                Err(e) => e,
            };
            assert!(
                err.contains("lm_head") && err.contains("tie_word_embeddings"),
                "error must name both lm_head and tie_word_embeddings; got: {err}"
            );
            // Cleanup; ignore errors.
            let _ = std::fs::remove_file(&lm_head_file);
            let _ = std::fs::remove_dir_all(&tmp);
        }

        #[test]
        fn from_q4_dir_rejects_untied_config_without_lm_head_artifact() {
            // Round-1 codex Blocker: the loader silently fell back to
            // `embed_tokens` when `tie_word_embeddings=false` and
            // `lm_head.q4` was missing. That is exactly the contamination
            // ADR-044 step 3c forbids — a QuaRot artifact missing its
            // materialized fused-and-rotated head would still produce a
            // PPL and a PASS verdict using the wrong logits matrix. The
            // loader now refuses before any tensor I/O.
            let Some(_) = Device::system_default() else {
                return;
            };
            let (mut cfg, _weights) = tiny_metal_qwen35_fixture();
            cfg.tie_word_embeddings = false; // make this an untied config
            let tmp =
                std::env::temp_dir().join(format!("lattice-q4-test-untied-{}", std::process::id()));
            std::fs::create_dir_all(&tmp).expect("tempdir create");
            // Empty dir → lm_head.weight.q4 does not exist.
            let err = match MetalQwen35State::from_q4_dir(
                &tmp,
                std::path::Path::new("/dev/null"),
                &cfg,
                16,
            ) {
                Ok(_) => panic!("untied config without lm_head artifact must error"),
                Err(e) => e,
            };
            assert!(
                err.contains("lm_head") && err.contains("tie_word_embeddings"),
                "error must name both lm_head and tie_word_embeddings; got: {err}"
            );
            // Cleanup; ignore errors (the dir may have been removed by other tests).
            let _ = std::fs::remove_dir_all(&tmp);
        }

        #[test]
        fn lora_gemv_kernel_matches_cpu_reference() {
            let Some(device) = Device::system_default() else {
                return;
            };
            let rank: u32 = 16;
            let k: u32 = 1024;
            let n: u32 = 512;
            let scale: f32 = 0.5;

            // Synthetic data
            let mut rng_state = 0xCAFE_BABE_u64;
            let mut rand_f32 = || -> f32 {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rng_state >> 11) as u32 as f32 / u32::MAX as f32) - 0.5
            };
            let x: Vec<f32> = (0..k).map(|_| rand_f32()).collect();
            let a: Vec<f32> = (0..(rank * k)).map(|_| rand_f32()).collect();
            let b: Vec<f32> = (0..(n * rank)).map(|_| rand_f32()).collect();
            let y_init: Vec<f32> = (0..n).map(|_| rand_f32()).collect();

            // CPU reference: y += scale * B @ (A @ x)
            let mut intermediate_cpu = vec![0.0f32; rank as usize];
            for r in 0..rank as usize {
                let row = &a[r * k as usize..(r + 1) * k as usize];
                intermediate_cpu[r] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            }
            let mut y_expected = y_init.clone();
            for i in 0..n as usize {
                let row = &b[i * rank as usize..(i + 1) * rank as usize];
                let dot: f32 = row
                    .iter()
                    .zip(intermediate_cpu.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                y_expected[i] += scale * dot;
            }

            // GPU execution
            let opts = CompileOptions::new();
            let library = device
                .new_library_with_source(MSL_SOURCE, &opts)
                .expect("MSL compile");
            let pipeline_a = {
                let f = library
                    .get_function("lora_gemv_a", None)
                    .expect("lora_gemv_a");
                device
                    .new_compute_pipeline_state_with_function(&f)
                    .expect("pipeline lora_gemv_a")
            };
            let pipeline_b = {
                let f = library
                    .get_function("lora_gemv_b_accum", None)
                    .expect("lora_gemv_b_accum");
                device
                    .new_compute_pipeline_state_with_function(&f)
                    .expect("pipeline lora_gemv_b_accum")
            };

            let make_buf = |data: &[f32]| -> Buffer {
                device.new_buffer_with_data(
                    data.as_ptr() as *const _,
                    (data.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            };
            let x_buf = make_buf(&x);
            let a_buf = make_buf(&a);
            let b_buf = make_buf(&b);
            let y_buf = make_buf(&y_init);
            let inter_buf =
                device.new_buffer((rank as u64) * 4, MTLResourceOptions::StorageModeShared);

            let queue = device.new_command_queue();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            // Phase 1
            enc.set_compute_pipeline_state(&pipeline_a);
            enc.set_buffer(0, Some(&x_buf), 0);
            enc.set_buffer(1, Some(&a_buf), 0);
            enc.set_buffer(2, Some(&inter_buf), 0);
            enc.set_bytes(3, 4, &rank as *const u32 as *const _);
            enc.set_bytes(4, 4, &k as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(rank as u64, 1, 1), MTLSize::new(32, 4, 1));

            // Phase 2
            enc.set_compute_pipeline_state(&pipeline_b);
            enc.set_buffer(0, Some(&inter_buf), 0);
            enc.set_buffer(1, Some(&b_buf), 0);
            enc.set_buffer(2, Some(&y_buf), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
            enc.set_bytes(4, 4, &rank as *const u32 as *const _);
            enc.set_bytes(5, 4, &scale as *const f32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(n.div_ceil(256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // SAFETY: y_buf is StorageModeShared, GPU work completed, length matches allocation.
            let y_gpu: &[f32] =
                unsafe { std::slice::from_raw_parts(y_buf.contents() as *const f32, n as usize) };

            let max_diff = y_expected
                .iter()
                .zip(y_gpu.iter())
                .map(|(e, g)| (e - g).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-3,
                "LoRA GEMV GPU vs CPU diverged: max_abs_diff={max_diff}"
            );
        }

        #[test]
        fn load_adapter_and_dispatch_lora_if_active() {
            let Some(device) = Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state =
                MetalQwen35State::new(&weights, &cfg, 4).expect("tiny MetalQwen35State fixture");

            let hidden = cfg.hidden_size;
            let rank = 4usize;
            let scale = 2.0f32;

            let mut rng = 0xABCD_u64;
            let mut rand_f32 = || -> f32 {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rng >> 11) as u32 as f32 / u32::MAX as f32) - 0.5
            };

            let a: Vec<f32> = (0..rank * hidden).map(|_| rand_f32()).collect();
            let b: Vec<f32> = (0..hidden * rank).map(|_| rand_f32()).collect();

            let layers = vec![LoraLayerData {
                layer_idx: 0,
                module: "q_proj".into(),
                a: a.clone(),
                b: b.clone(),
                rank,
                d_in: hidden,
                d_out: hidden,
            }];

            // Load without QuaRot rotation (unrotated base)
            state.load_lora_adapter(layers, scale, None).unwrap();
            assert!(state.has_lora_adapter());

            // Dispatch through the production wrapper
            let x: Vec<f32> = (0..hidden).map(|_| rand_f32()).collect();
            let y_init: Vec<f32> = (0..hidden).map(|_| rand_f32()).collect();

            let x_buf = device.new_buffer_with_data(
                x.as_ptr() as *const _,
                (x.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let y_buf = device.new_buffer_with_data(
                y_init.as_ptr() as *const _,
                (y_init.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let cmd = state.engine.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            state.dispatch_lora_if_active(&enc, &x_buf, &y_buf, 0, "q_proj");
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // CPU reference
            let mut intermediate = vec![0.0f32; rank];
            for r in 0..rank {
                intermediate[r] = a[r * hidden..(r + 1) * hidden]
                    .iter()
                    .zip(x.iter())
                    .map(|(ai, xi)| ai * xi)
                    .sum();
            }
            let mut y_expected = y_init.clone();
            for i in 0..hidden {
                let dot: f32 = b[i * rank..(i + 1) * rank]
                    .iter()
                    .zip(intermediate.iter())
                    .map(|(bi, ii)| bi * ii)
                    .sum();
                y_expected[i] += scale * dot;
            }

            // SAFETY: y_buf is StorageModeShared, GPU work completed, length matches allocation.
            let y_gpu: &[f32] =
                unsafe { std::slice::from_raw_parts(y_buf.contents() as *const f32, hidden) };
            let max_diff = y_expected
                .iter()
                .zip(y_gpu.iter())
                .map(|(e, g)| (e - g).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-3,
                "dispatch_lora_if_active GPU vs CPU: max_abs_diff={max_diff}"
            );

            // Verify no-op for wrong layer/module
            let y_buf2 = device.new_buffer_with_data(
                y_init.as_ptr() as *const _,
                (y_init.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let cmd2 = state.engine.queue.new_command_buffer();
            let enc2 = cmd2.new_compute_command_encoder();
            state.dispatch_lora_if_active(&enc2, &x_buf, &y_buf2, 0, "v_proj");
            enc2.end_encoding();
            cmd2.commit();
            cmd2.wait_until_completed();

            // SAFETY: y_buf2 is StorageModeShared, GPU work completed, length matches allocation.
            let y_noop: &[f32] =
                unsafe { std::slice::from_raw_parts(y_buf2.contents() as *const f32, hidden) };
            assert_eq!(
                y_noop,
                &y_init[..],
                "dispatch for non-loaded module should be no-op"
            );

            // Unload
            state.unload_lora_adapter();
            assert!(!state.has_lora_adapter());
        }

        // ── malformed-load validation tests ──────────────────────────────────
        // These tests exercise the validation gates in load_lora_adapter and do
        // not require a live GPU device (errors are returned before any buffer
        // allocation). Each case must return Err, never panic.

        fn make_valid_layer(hidden: usize, rank: usize) -> LoraLayerData {
            LoraLayerData {
                layer_idx: 0,
                module: "q_proj".into(),
                a: vec![0.0f32; rank * hidden],
                b: vec![0.0f32; hidden * rank],
                rank,
                d_in: hidden,
                d_out: hidden,
            }
        }

        #[test]
        fn load_lora_adapter_rejects_short_a() {
            let Some(_dev) = metal::Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
            let hidden = cfg.hidden_size;
            let rank = 4usize;
            let mut layer = make_valid_layer(hidden, rank);
            layer.a.pop(); // one element short of rank * d_in
            let result = state.load_lora_adapter(vec![layer], 1.0, None);
            assert!(result.is_err(), "short A must return Err");
        }

        #[test]
        fn load_lora_adapter_rejects_short_b() {
            let Some(_dev) = metal::Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
            let hidden = cfg.hidden_size;
            let rank = 4usize;
            let mut layer = make_valid_layer(hidden, rank);
            layer.b.pop(); // one element short of d_out * rank
            let result = state.load_lora_adapter(vec![layer], 1.0, None);
            assert!(result.is_err(), "short B must return Err");
        }

        #[test]
        fn load_lora_adapter_rejects_zero_rank() {
            let Some(_dev) = metal::Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
            let hidden = cfg.hidden_size;
            let layer = LoraLayerData {
                layer_idx: 0,
                module: "q_proj".into(),
                a: vec![],
                b: vec![],
                rank: 0,
                d_in: hidden,
                d_out: hidden,
            };
            let result = state.load_lora_adapter(vec![layer], 1.0, None);
            assert!(result.is_err(), "zero rank must return Err");
        }

        #[test]
        fn load_lora_adapter_rejects_zero_d_in() {
            let Some(_dev) = metal::Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
            let layer = LoraLayerData {
                layer_idx: 0,
                module: "q_proj".into(),
                a: vec![],
                b: vec![],
                rank: 4,
                d_in: 0,
                d_out: cfg.hidden_size,
            };
            let result = state.load_lora_adapter(vec![layer], 1.0, None);
            assert!(result.is_err(), "zero d_in must return Err");
        }

        #[test]
        fn load_lora_adapter_rejects_out_of_range_layer_idx() {
            let Some(_dev) = metal::Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
            let hidden = cfg.hidden_size;
            let rank = 4usize;
            // cfg.num_hidden_layers == 1, so layer_idx=1 is out of range
            let mut layer = make_valid_layer(hidden, rank);
            layer.layer_idx = cfg.num_hidden_layers; // == num_hidden_layers, out of range
            let result = state.load_lora_adapter(vec![layer], 1.0, None);
            assert!(result.is_err(), "out-of-range layer_idx must return Err");
        }

        #[test]
        fn load_lora_adapter_rejects_wrong_a_length_mismatched_dims() {
            let Some(_dev) = metal::Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
            let hidden = cfg.hidden_size;
            let rank = 4usize;
            // Declare rank=4, d_in=hidden, but supply A with wrong inner dim
            let layer = LoraLayerData {
                layer_idx: 0,
                module: "q_proj".into(),
                a: vec![0.0f32; rank * (hidden + 1)], // extra column — length mismatch
                b: vec![0.0f32; hidden * rank],
                rank,
                d_in: hidden,
                d_out: hidden,
            };
            let result = state.load_lora_adapter(vec![layer], 1.0, None);
            assert!(result.is_err(), "A length != rank*d_in must return Err");
        }

        #[test]
        fn load_lora_adapter_rejects_wrong_b_length_mismatched_dims() {
            let Some(_dev) = metal::Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
            let hidden = cfg.hidden_size;
            let rank = 4usize;
            let layer = LoraLayerData {
                layer_idx: 0,
                module: "q_proj".into(),
                a: vec![0.0f32; rank * hidden],
                b: vec![0.0f32; (hidden + 1) * rank], // extra row — length mismatch
                rank,
                d_in: hidden,
                d_out: hidden,
            };
            let result = state.load_lora_adapter(vec![layer], 1.0, None);
            assert!(result.is_err(), "B length != d_out*rank must return Err");
        }

        #[test]
        fn load_lora_adapter_rejects_quarot_with_short_a() {
            let Some(_dev) = metal::Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
            let hidden = cfg.hidden_size;
            let rank = 4usize;
            let mut layer = make_valid_layer(hidden, rank);
            layer.a.pop(); // short A triggers validation error
            // quarot_seed = Some triggers rotate_adapter_for_quarot path, but
            // validation after that catches the mismatch.
            let result = state.load_lora_adapter(vec![layer], 1.0, Some(42));
            assert!(result.is_err(), "quarot path: short A must return Err");
        }

        #[test]
        fn load_lora_adapter_rejects_quarot_with_short_b() {
            let Some(_dev) = metal::Device::system_default() else {
                return;
            };
            let (cfg, weights) = tiny_metal_qwen35_fixture();
            let mut state = MetalQwen35State::new(&weights, &cfg, 4).expect("tiny fixture");
            let hidden = cfg.hidden_size;
            let rank = 4usize;
            let mut layer = make_valid_layer(hidden, rank);
            layer.b.pop();
            let result = state.load_lora_adapter(vec![layer], 1.0, Some(42));
            assert!(result.is_err(), "quarot path: short B must return Err");
        }
    }

    impl crate::speculative::MtpTargetVerifier for MetalQwen35State {
        fn cache_position(&self) -> usize {
            self.session.kv_cache.seq_len
        }

        fn rollback_cache_to(
            &mut self,
            seq_len: usize,
        ) -> Result<(), crate::error::InferenceError> {
            self.rollback_speculative_state_to(seq_len)
        }

        fn verify_tokens(
            &mut self,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<Vec<Vec<f32>>, crate::error::InferenceError> {
            let out = self.verify_tokens_batched(tokens, start_pos)?;
            Ok(out.logits)
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub use inner::{
    ChatCompletionOutput, ChatMessage, ChatRole, LayerImportanceScore, LayerPruningPlan,
    LoraLayerData, MetalQwen35State, format_chat_template,
};

// Stub for non-macOS or non-metal builds.
/// **Unstable**: Metal Qwen3.5 forward state; stub on non-macOS; full impl behind metal-gpu feature.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub struct MetalQwen35State;

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
impl MetalQwen35State {
    /// **Unstable**: construct Metal Qwen3.5 state; always fails without metal-gpu feature.
    pub fn new(
        _weights: &crate::model::qwen35::ModelWeights,
        _cfg: &crate::model::qwen35_config::Qwen35Config,
        _max_cache_len: usize,
    ) -> Result<Self, String> {
        Err("Metal GPU not available (requires macOS + metal-gpu feature)".into())
    }

    /// **Unstable**: load Q4/F16 directly stub; always fails without metal-gpu feature.
    pub fn from_q4_dir(
        _q4_dir: &std::path::Path,
        _tokenizer_path: &std::path::Path,
        _cfg: &crate::model::qwen35_config::Qwen35Config,
        _max_cache_len: usize,
    ) -> Result<Self, String> {
        Err("Metal GPU not available (requires macOS + metal-gpu feature)".into())
    }

    /// **Unstable**: Metal single-token forward step stub.
    pub fn forward_step(&mut self, _token_id: u32, _position: usize) -> Vec<f32> {
        Vec::new()
    }

    /// **Unstable**: Metal generate stub; always returns empty output without metal-gpu feature.
    pub fn generate(
        &mut self,
        _prompt: &str,
        _tokenizer: &crate::tokenizer::bpe::BpeTokenizer,
        _cfg: &crate::model::qwen35_config::GenerateConfig,
    ) -> crate::model::qwen35_config::GenerateOutput {
        crate::model::qwen35_config::GenerateOutput {
            text: String::new(),
            token_ids: vec![],
            prompt_tokens: 0,
            generated_tokens: 0,
        }
    }

    /// **Unstable**: reset recurrent KV state stub.
    pub fn reset_state(&mut self) {}

    /// **Unstable**: max context stub; always 0 without metal-gpu feature.
    pub fn max_context(&self) -> usize {
        0
    }

    /// **Unstable**: Metal Q4 per-token NLL stub; always errors without metal-gpu feature.
    pub fn compute_token_nlls(
        &mut self,
        _tokens: &[u32],
    ) -> Result<Vec<f32>, crate::error::InferenceError> {
        Err(crate::error::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }

    /// **Unstable**: Metal Q4 perplexity stub; always errors without metal-gpu feature.
    pub fn compute_perplexity(
        &mut self,
        _tokens: &[u32],
        _cfg: &crate::model::qwen35::PerplexityConfig,
    ) -> Result<crate::model::qwen35::PerplexityReport, crate::error::InferenceError> {
        Err(crate::error::InferenceError::Inference(
            "Metal GPU not available (requires macOS + metal-gpu feature)".into(),
        ))
    }
}
