//! Criterion benchmarks for Metal decode_attention: flash vs reference baseline.
//!
//! Measures single-token decode_attention host-submit-wait latency for:
//!   {8Q/2KV, 16Q/2KV} × {cache_len: 512, 2048, 8192}
//! head_dim = 256.
//!
//! Two benchmark groups allow direct Criterion comparison:
//!   - decode_attention_reference: old 3-pass kernel
//!   - decode_attention_flash:     new flash GQA decode (direct + partial/reduce)
//!
//! Metal-only (macOS + --features metal-gpu):
//!   cargo bench -p lattice-inference --features "f16,metal-gpu" --bench decode_attn_bench
//!
//! CPU-only builds silently skip all benchmarks.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

// ---------------------------------------------------------------------------
// MSL — baseline (old 3-pass QK-recomputing kernel)
// ---------------------------------------------------------------------------

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const MSL_REFERENCE: &str = r#"
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
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]])
{
    if (gid >= num_q_heads) return;
    if (cache_len == 0) return;

    constexpr uint ATTN_WG = 256;
    uint qh = gid;
    uint groups = num_q_heads / num_kv_heads;
    uint kvh = qh / groups;
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
            acc += exp(dot * scale - max_val) * inv_sum * v_cache[t * kv_dim + kvh * head_dim + d];
        }
        out_head[d] = acc;
    }
}
"#;

// ---------------------------------------------------------------------------
// MSL — flash direct grouped decode
// ---------------------------------------------------------------------------

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const MSL_FLASH_DIRECT: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void decode_attention_flash(
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
    constexpr uint MAX_GRP     = 8;
    constexpr uint TILE_TOKENS = 256;

    const uint kvh        = gid;
    if (kvh >= num_kv_heads) return;
    const uint group_size = num_q_heads / num_kv_heads;
    const uint qh_base    = kvh * group_size;

    threadgroup float q_s    [MAX_GRP * HEAD_DIM];
    threadgroup float score_s[MAX_GRP * TILE_TOKENS];
    threadgroup float reduce_s[TILE_TOKENS];
    threadgroup float m_s    [MAX_GRP];
    threadgroup float l_s    [MAX_GRP];
    threadgroup float alpha_s[MAX_GRP];

    if (lid < group_size) { m_s[lid] = -INFINITY; l_s[lid] = 0.0f; }
    for (uint idx = lid; idx < group_size * HEAD_DIM; idx += tgs)
        q_s[idx] = q[qh_base * HEAD_DIM + idx];

    float acc[MAX_GRP];
    for (uint qi = 0; qi < MAX_GRP; qi++) acc[qi] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tile_start = 0; tile_start < cache_len; tile_start += TILE_TOKENS) {
        const uint tile_count = min(TILE_TOKENS, cache_len - tile_start);

        if (lid < tile_count) {
            float dot[MAX_GRP];
            for (uint qi = 0; qi < MAX_GRP; qi++) dot[qi] = 0.0f;
            const uint k_base = (tile_start + lid) * kv_dim + kvh * HEAD_DIM;
            for (uint d = 0; d < HEAD_DIM; d++) {
                const float kd = k_cache[k_base + d];
                for (uint qi = 0; qi < group_size; qi++)
                    dot[qi] += q_s[qi * HEAD_DIM + d] * kd;
            }
            for (uint qi = 0; qi < group_size; qi++)
                score_s[qi * TILE_TOKENS + lid] = dot[qi] * scale;
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
            if (lid < tile_count)
                score_s[qi * TILE_TOKENS + lid] =
                    exp(score_s[qi * TILE_TOKENS + lid] - m_s[qi]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            reduce_s[lid] = (lid < tile_count) ? score_s[qi * TILE_TOKENS + lid] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs >> 1; s > 0; s >>= 1) {
                if (lid < s) reduce_s[lid] += reduce_s[lid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0) l_s[qi] = alpha_s[qi] * l_s[qi] + reduce_s[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint qi = 0; qi < group_size; qi++) acc[qi] *= alpha_s[qi];

        const uint d = lid;
        for (uint local_t = 0; local_t < tile_count; local_t++) {
            const float v = v_cache[(tile_start + local_t) * kv_dim + kvh * HEAD_DIM + d];
            for (uint qi = 0; qi < group_size; qi++)
                acc[qi] += score_s[qi * TILE_TOKENS + local_t] * v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint qi = 0; qi < group_size; qi++) {
        const uint qh = qh_base + qi;
        const float dn = l_s[qi];
        out[qh * HEAD_DIM + lid] = dn > 0.0f ? acc[qi] / dn : 0.0f;
    }
}
"#;

// ---------------------------------------------------------------------------
// MSL — flash partial + reduce kernels
// ---------------------------------------------------------------------------

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
const MSL_FLASH_PARTIAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void decode_attention_flash_partial_bench(
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
    constexpr uint STRIDE      = HEAD_DIM + 2;

    const uint lid          = lid3.x;
    const uint tgs          = tgs3.x;
    const uint kvh          = gid3.x;
    const uint partition_id = gid3.y;
    if (kvh >= num_kv_heads) return;
    const uint part_start = partition_id * partition_tokens;
    if (part_start >= cache_len) return;
    const uint part_end   = min(cache_len, part_start + partition_tokens);

    const uint group_size = num_q_heads / num_kv_heads;
    const uint qh_base    = kvh * group_size;

    threadgroup float q_s    [MAX_GRP * HEAD_DIM];
    threadgroup float score_s[MAX_GRP * TILE_TOKENS];
    threadgroup float reduce_s[TILE_TOKENS];
    threadgroup float m_s    [MAX_GRP];
    threadgroup float l_s    [MAX_GRP];
    threadgroup float alpha_s[MAX_GRP];

    if (lid < group_size) { m_s[lid] = -INFINITY; l_s[lid] = 0.0f; }
    for (uint idx = lid; idx < group_size * HEAD_DIM; idx += tgs)
        q_s[idx] = q[qh_base * HEAD_DIM + idx];

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
                for (uint qi = 0; qi < group_size; qi++)
                    dot[qi] += q_s[qi * HEAD_DIM + d] * kd;
            }
            for (uint qi = 0; qi < group_size; qi++)
                score_s[qi * TILE_TOKENS + lid] = dot[qi] * scale;
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
            if (lid < tile_count)
                score_s[qi * TILE_TOKENS + lid] =
                    exp(score_s[qi * TILE_TOKENS + lid] - m_s[qi]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            reduce_s[lid] = (lid < tile_count) ? score_s[qi * TILE_TOKENS + lid] : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = tgs >> 1; s > 0; s >>= 1) {
                if (lid < s) reduce_s[lid] += reduce_s[lid + s];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (lid == 0) l_s[qi] = alpha_s[qi] * l_s[qi] + reduce_s[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint qi = 0; qi < group_size; qi++) acc[qi] *= alpha_s[qi];
        const uint d = lid;
        for (uint local_t = 0; local_t < tile_count; local_t++) {
            const float v = v_cache[(tile_start + local_t) * kv_dim + kvh * HEAD_DIM + d];
            for (uint qi = 0; qi < group_size; qi++)
                acc[qi] += score_s[qi * TILE_TOKENS + local_t] * v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
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

kernel void decode_attention_flash_reduce_bench(
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
    constexpr uint STRIDE   = HEAD_DIM + 2;
    const uint kvh = gid;
    if (kvh >= num_kv_heads) return;
    const uint group_size = num_q_heads / num_kv_heads;
    const uint qh_base    = kvh * group_size;
    const uint d          = lid;

    float m_acc[MAX_GRP], l_acc[MAX_GRP], acc[MAX_GRP];
    for (uint qi = 0; qi < group_size; qi++) {
        m_acc[qi] = -INFINITY; l_acc[qi] = 0.0f; acc[qi] = 0.0f;
    }
    for (uint p = 0; p < num_partitions; p++) {
        for (uint qi = 0; qi < group_size; qi++) {
            const uint qh     = qh_base + qi;
            const uint base   = (p * num_q_heads + qh) * STRIDE;
            const float m_p   = attn_partials[base + 0];
            const float l_p   = attn_partials[base + 1];
            const float acc_d = attn_partials[base + 2 + d];
            if (!isfinite(m_p)) continue;
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
"#;

// ---------------------------------------------------------------------------
// PRNG (same seed as profile_metal_decode.rs for reproducible buffer contents)
// ---------------------------------------------------------------------------

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn xorshift32(state: &mut u32) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    (x as f32 / u32::MAX as f32) - 0.5
}

// ---------------------------------------------------------------------------
// Benchmark: reference (old 3-pass) vs flash decode at {512, 2048, 8192}
// ---------------------------------------------------------------------------

fn bench_reference_decode(c: &mut Criterion) {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        use metal::*;

        const HEAD_DIM: u32 = 256;

        let device = match Device::system_default() {
            Some(d) => d,
            None => {
                eprintln!("[decode_attn_bench] No Metal device — skipping reference benchmarks");
                return;
            }
        };
        let queue = device.new_command_queue();

        let compile_opts = CompileOptions::new();
        eprint!("[decode_attn_bench] Compiling reference kernel...");
        let lib_ref = device
            .new_library_with_source(MSL_REFERENCE, &compile_opts)
            .expect("MSL_REFERENCE compile failed");
        let pipe_ref = {
            let f = lib_ref
                .get_function("decode_attention_reference", None)
                .unwrap();
            device.new_compute_pipeline_state_with_function(&f).unwrap()
        };
        eprintln!(" done");

        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

        let configs: &[(u32, u32, u32, &str)] = &[
            (8, 2, 512, "8q2kv"),
            (8, 2, 2048, "8q2kv"),
            (8, 2, 8192, "8q2kv"),
            (16, 2, 512, "16q2kv"),
            (16, 2, 2048, "16q2kv"),
            (16, 2, 8192, "16q2kv"),
        ];

        let mut group = c.benchmark_group("decode_attention_reference");
        group.throughput(Throughput::Elements(1));
        group.warm_up_time(Duration::from_secs(2));
        group.measurement_time(Duration::from_secs(20));
        group.sample_size(50);

        for &(nqh, nkh, clen, label) in configs {
            let q_dim = nqh * HEAD_DIM;
            let kv_dim = nkh * HEAD_DIM;

            let q_buf =
                device.new_buffer((q_dim as u64) * 4, MTLResourceOptions::StorageModeShared);
            let k_buf = device.new_buffer(
                (clen as u64) * (kv_dim as u64) * 4,
                MTLResourceOptions::StorageModeShared,
            );
            let v_buf = device.new_buffer(
                (clen as u64) * (kv_dim as u64) * 4,
                MTLResourceOptions::StorageModeShared,
            );
            let out_buf =
                device.new_buffer((q_dim as u64) * 4, MTLResourceOptions::StorageModeShared);

            let mut rng = 0xDEAD_BEEFu32;
            unsafe {
                let q =
                    std::slice::from_raw_parts_mut(q_buf.contents() as *mut f32, q_dim as usize);
                for v in q.iter_mut() {
                    *v = xorshift32(&mut rng);
                }
                let k = std::slice::from_raw_parts_mut(
                    k_buf.contents() as *mut f32,
                    (clen * kv_dim) as usize,
                );
                for v in k.iter_mut() {
                    *v = xorshift32(&mut rng);
                }
                let vd = std::slice::from_raw_parts_mut(
                    v_buf.contents() as *mut f32,
                    (clen * kv_dim) as usize,
                );
                for v in vd.iter_mut() {
                    *v = xorshift32(&mut rng);
                }
            }

            group.bench_function(BenchmarkId::new(label, clen), |b| {
                b.iter(|| {
                    let cmd = queue.new_command_buffer();
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&pipe_ref);
                    enc.set_buffer(0, Some(&q_buf), 0);
                    enc.set_buffer(1, Some(&k_buf), 0);
                    enc.set_buffer(2, Some(&v_buf), 0);
                    enc.set_buffer(3, Some(&out_buf), 0);
                    enc.set_bytes(4, 4, &clen as *const u32 as *const _);
                    enc.set_bytes(5, 4, &HEAD_DIM as *const u32 as *const _);
                    enc.set_bytes(6, 4, &nqh as *const u32 as *const _);
                    enc.set_bytes(7, 4, &nkh as *const u32 as *const _);
                    enc.set_bytes(8, 4, &q_dim as *const u32 as *const _);
                    enc.set_bytes(9, 4, &kv_dim as *const u32 as *const _);
                    enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(nqh as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    black_box(())
                });
            });
        }

        group.finish();
    }

    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    let _ = c;
}

fn bench_flash_decode(c: &mut Criterion) {
    #[cfg(all(target_os = "macos", feature = "metal-gpu"))]
    {
        use metal::*;

        const HEAD_DIM: u32 = 256;
        const PARTITION_TOKENS: u32 = 1024;
        const DIRECT_THRESHOLD: u32 = 512;

        let device = match Device::system_default() {
            Some(d) => d,
            None => {
                eprintln!("[decode_attn_bench] No Metal device — skipping flash benchmarks");
                return;
            }
        };
        let queue = device.new_command_queue();

        let compile_opts = CompileOptions::new();

        eprint!("[decode_attn_bench] Compiling flash direct kernel...");
        let lib_direct = device
            .new_library_with_source(MSL_FLASH_DIRECT, &compile_opts)
            .expect("MSL_FLASH_DIRECT compile failed");
        let pipe_direct = {
            let f = lib_direct
                .get_function("decode_attention_flash", None)
                .unwrap();
            device.new_compute_pipeline_state_with_function(&f).unwrap()
        };
        eprintln!(" done");

        eprint!("[decode_attn_bench] Compiling flash partial+reduce kernels...");
        let lib_partial = device
            .new_library_with_source(MSL_FLASH_PARTIAL, &compile_opts)
            .expect("MSL_FLASH_PARTIAL compile failed");
        let pipe_partial = {
            let f = lib_partial
                .get_function("decode_attention_flash_partial_bench", None)
                .unwrap();
            device.new_compute_pipeline_state_with_function(&f).unwrap()
        };
        let pipe_reduce = {
            let f = lib_partial
                .get_function("decode_attention_flash_reduce_bench", None)
                .unwrap();
            device.new_compute_pipeline_state_with_function(&f).unwrap()
        };
        eprintln!(" done");

        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

        let configs: &[(u32, u32, u32, &str)] = &[
            (8, 2, 512, "8q2kv"),
            (8, 2, 2048, "8q2kv"),
            (8, 2, 8192, "8q2kv"),
            (16, 2, 512, "16q2kv"),
            (16, 2, 2048, "16q2kv"),
            (16, 2, 8192, "16q2kv"),
        ];

        let mut group = c.benchmark_group("decode_attention_flash");
        group.throughput(Throughput::Elements(1));
        group.warm_up_time(Duration::from_secs(3));
        group.measurement_time(Duration::from_secs(30));
        group.sample_size(50);

        for &(nqh, nkh, clen, label) in configs {
            let q_dim = nqh * HEAD_DIM;
            let kv_dim = nkh * HEAD_DIM;
            let num_partitions = clen.div_ceil(PARTITION_TOKENS);
            let stride = HEAD_DIM + 2;

            let q_buf =
                device.new_buffer((q_dim as u64) * 4, MTLResourceOptions::StorageModeShared);
            let k_buf = device.new_buffer(
                (clen as u64) * (kv_dim as u64) * 4,
                MTLResourceOptions::StorageModeShared,
            );
            let v_buf = device.new_buffer(
                (clen as u64) * (kv_dim as u64) * 4,
                MTLResourceOptions::StorageModeShared,
            );
            let out_buf =
                device.new_buffer((q_dim as u64) * 4, MTLResourceOptions::StorageModeShared);
            let partials_buf = device.new_buffer(
                (num_partitions as u64) * (nqh as u64) * (stride as u64) * 4,
                MTLResourceOptions::StorageModeShared,
            );

            let mut rng = 0xDEAD_BEEFu32;
            unsafe {
                let q =
                    std::slice::from_raw_parts_mut(q_buf.contents() as *mut f32, q_dim as usize);
                for v in q.iter_mut() {
                    *v = xorshift32(&mut rng);
                }
                let k = std::slice::from_raw_parts_mut(
                    k_buf.contents() as *mut f32,
                    (clen * kv_dim) as usize,
                );
                for v in k.iter_mut() {
                    *v = xorshift32(&mut rng);
                }
                let vd = std::slice::from_raw_parts_mut(
                    v_buf.contents() as *mut f32,
                    (clen * kv_dim) as usize,
                );
                for v in vd.iter_mut() {
                    *v = xorshift32(&mut rng);
                }
            }

            group.bench_function(BenchmarkId::new(label, clen), |b| {
                b.iter(|| {
                    let cmd = queue.new_command_buffer();
                    let enc = cmd.new_compute_command_encoder();

                    if clen <= DIRECT_THRESHOLD {
                        enc.set_compute_pipeline_state(&pipe_direct);
                        enc.set_buffer(0, Some(&q_buf), 0);
                        enc.set_buffer(1, Some(&k_buf), 0);
                        enc.set_buffer(2, Some(&v_buf), 0);
                        enc.set_buffer(3, Some(&out_buf), 0);
                        enc.set_bytes(4, 4, &clen as *const u32 as *const _);
                        enc.set_bytes(5, 4, &HEAD_DIM as *const u32 as *const _);
                        enc.set_bytes(6, 4, &nqh as *const u32 as *const _);
                        enc.set_bytes(7, 4, &nkh as *const u32 as *const _);
                        enc.set_bytes(8, 4, &q_dim as *const u32 as *const _);
                        enc.set_bytes(9, 4, &kv_dim as *const u32 as *const _);
                        enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new(nkh as u64, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                    } else {
                        enc.set_compute_pipeline_state(&pipe_partial);
                        enc.set_buffer(0, Some(&q_buf), 0);
                        enc.set_buffer(1, Some(&k_buf), 0);
                        enc.set_buffer(2, Some(&v_buf), 0);
                        enc.set_buffer(3, Some(&partials_buf), 0);
                        enc.set_bytes(4, 4, &clen as *const u32 as *const _);
                        enc.set_bytes(5, 4, &HEAD_DIM as *const u32 as *const _);
                        enc.set_bytes(6, 4, &nqh as *const u32 as *const _);
                        enc.set_bytes(7, 4, &nkh as *const u32 as *const _);
                        enc.set_bytes(8, 4, &q_dim as *const u32 as *const _);
                        enc.set_bytes(9, 4, &kv_dim as *const u32 as *const _);
                        enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                        enc.set_bytes(11, 4, &PARTITION_TOKENS as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new(nkh as u64, num_partitions as u64, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        enc.set_compute_pipeline_state(&pipe_reduce);
                        enc.set_buffer(0, Some(&partials_buf), 0);
                        enc.set_buffer(1, Some(&out_buf), 0);
                        enc.set_bytes(2, 4, &nqh as *const u32 as *const _);
                        enc.set_bytes(3, 4, &nkh as *const u32 as *const _);
                        enc.set_bytes(4, 4, &num_partitions as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new(nkh as u64, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                    }

                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    black_box(())
                });
            });
        }

        group.finish();
    }

    #[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
    let _ = c;
}

criterion_group!(benches, bench_reference_decode, bench_flash_decode);
criterion_main!(benches);
