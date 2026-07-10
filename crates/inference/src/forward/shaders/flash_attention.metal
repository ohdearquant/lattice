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
