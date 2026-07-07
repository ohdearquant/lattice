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
