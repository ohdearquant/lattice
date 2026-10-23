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
    float rms = rms_inv_from_local_sum(shared, local_sum, lid, tgs, head_dim, eps);

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
        // invalid (non-finite) norm, never `value * guarded_zero_reciprocal`
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
        bool q_valid = isfinite(sg_buf[0]);
        float qs_inv = q_valid ? rsqrt(sg_buf[0] + GDN_QK_NORM_EPS) : 0.0f;
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
        bool k_valid = isfinite(sg_buf[0]);
        float ks_inv = k_valid ? rsqrt(sg_buf[0] + GDN_QK_NORM_EPS) : 0.0f;
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
                float sp = gdn_softplus(s + dt_bias[h]);
                // Floor the per-token log-decay so the chunked cumulative-log scan stays
                // closed under saturated decay.  An overflowing decay product drives -a*sp to
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
