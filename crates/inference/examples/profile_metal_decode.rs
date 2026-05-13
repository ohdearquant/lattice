//! Benchmark for decode_attention before/after flash rewrite.
//!
//! Measures isolated single-token decode_attention host-submit-wait latency for six
//! configurations:  {8Q/2KV, 16Q/2KV} × {cache_len: 512, 4096, 32768}
//! head_dim = 256.
//!
//! Reports:
//!  - "baseline": old 3-pass QK-recomputing kernel
//!  - "flash":    new flash GQA decode (H1+H2+H3+H4+H5)
//!  - parity:     max_abs_diff / mean_abs_diff between outputs
//!
//! stdout: JSONL (machine-readable)
//! stderr: human-readable table + parity
//!
//! Run:
//!   cargo run --release -p lattice-inference --example profile_metal_decode \
//!     --features "f16,metal-gpu"

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
fn main() {
    eprintln!("profile_metal_decode requires --features f16,metal-gpu on macOS");
    std::process::exit(1);
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
fn main() {
    use metal::*;
    use std::time::Instant;

    // -----------------------------------------------------------------------
    // MSL — baseline (old 3-pass kernel), renamed to decode_attention_reference
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // MSL — flash direct grouped decode (decode_attention, H1+H2+H4+H5)
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // MSL — flash partial + reduce kernels (H3, for cache_len > 512)
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    fn xorshift32(state: &mut u32) -> f32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        (x as f32 / u32::MAX as f32) - 0.5
    }

    fn stats(samples: &[f64]) -> (f64, f64, f64, f64, f64) {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let stdev = var.sqrt();
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = sorted[((sorted.len() - 1) as f64 * 0.50) as usize];
        let p99 = sorted[((sorted.len() - 1) as f64 * 0.99) as usize];
        (mean, stdev, p50, p99, min)
    }

    fn max_mean_abs_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
        let mut max_d = 0.0f32;
        let mut sum_d = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > max_d {
                max_d = d;
            }
            sum_d += d;
        }
        (max_d, sum_d / a.len() as f32)
    }

    // -----------------------------------------------------------------------
    // Metal setup
    // -----------------------------------------------------------------------
    let device = match Device::system_default() {
        Some(d) => d,
        None => {
            eprintln!("No Metal device");
            std::process::exit(1);
        }
    };
    let queue = device.new_command_queue();
    eprintln!("[bench] device: {}", device.name());

    let compile = |src: &str| -> Library {
        device
            .new_library_with_source(src, &CompileOptions::new())
            .expect("MSL compile failed")
    };
    let pipeline = |lib: &Library, name: &str| -> ComputePipelineState {
        let func = lib.get_function(name, None).expect("fn not found");
        device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline failed")
    };

    eprint!("[bench] Compiling reference kernel...");
    let lib_ref = compile(MSL_REFERENCE);
    let pipe_ref = pipeline(&lib_ref, "decode_attention_reference");
    eprintln!(" done");

    eprint!("[bench] Compiling flash direct kernel...");
    let lib_flash = compile(MSL_FLASH_DIRECT);
    let pipe_flash_direct = pipeline(&lib_flash, "decode_attention_flash");
    eprintln!(" done");

    eprint!("[bench] Compiling flash partial+reduce kernels...");
    let lib_partial = compile(MSL_FLASH_PARTIAL);
    let pipe_partial = pipeline(&lib_partial, "decode_attention_flash_partial_bench");
    let pipe_reduce = pipeline(&lib_partial, "decode_attention_flash_reduce_bench");
    eprintln!(" done");

    // -----------------------------------------------------------------------
    // Benchmark configurations
    // -----------------------------------------------------------------------
    struct Cfg {
        label: &'static str,
        num_q_heads: u32,
        num_kv_heads: u32,
        cache_len: u32,
        warmup: usize,
        measure: usize,
    }

    let cfgs: &[Cfg] = &[
        Cfg {
            label: "8Q/2KV",
            num_q_heads: 8,
            num_kv_heads: 2,
            cache_len: 512,
            warmup: 100,
            measure: 1000,
        },
        Cfg {
            label: "8Q/2KV",
            num_q_heads: 8,
            num_kv_heads: 2,
            cache_len: 4096,
            warmup: 30,
            measure: 300,
        },
        Cfg {
            label: "8Q/2KV",
            num_q_heads: 8,
            num_kv_heads: 2,
            cache_len: 32768,
            warmup: 3,
            measure: 30,
        },
        Cfg {
            label: "16Q/2KV",
            num_q_heads: 16,
            num_kv_heads: 2,
            cache_len: 512,
            warmup: 100,
            measure: 1000,
        },
        Cfg {
            label: "16Q/2KV",
            num_q_heads: 16,
            num_kv_heads: 2,
            cache_len: 4096,
            warmup: 30,
            measure: 300,
        },
        Cfg {
            label: "16Q/2KV",
            num_q_heads: 16,
            num_kv_heads: 2,
            cache_len: 32768,
            warmup: 3,
            measure: 30,
        },
    ];

    const HEAD_DIM: u32 = 256;
    const PARTITION_TOKENS: u32 = 1024;
    const DIRECT_THRESHOLD: u32 = 512;
    let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();

    const PARITY_MAX_ABS: f32 = 1e-4;
    let mut gate_failures: Vec<String> = Vec::new();

    eprintln!();
    eprintln!(
        "  {:<12}  {:>9}  {:>10}  {:>10}  {:>9}  {:>10}  {:>10}  {:>10}",
        "config",
        "cache_len",
        "base_p50_us",
        "flash_p50_us",
        "speedup",
        "max_diff",
        "mean_diff",
        "parity"
    );
    eprintln!("  {:-<103}", "");

    for cfg in cfgs {
        let nqh = cfg.num_q_heads;
        let nkh = cfg.num_kv_heads;
        let clen = cfg.cache_len;
        let q_dim = nqh * HEAD_DIM;
        let kv_dim = nkh * HEAD_DIM;

        // Allocate buffers
        let q_buf = device.new_buffer((q_dim as u64) * 4, MTLResourceOptions::StorageModeShared);
        let k_buf = device.new_buffer(
            (clen as u64) * (kv_dim as u64) * 4,
            MTLResourceOptions::StorageModeShared,
        );
        let v_buf = device.new_buffer(
            (clen as u64) * (kv_dim as u64) * 4,
            MTLResourceOptions::StorageModeShared,
        );
        let out_ref = device.new_buffer((q_dim as u64) * 4, MTLResourceOptions::StorageModeShared);
        let out_new = device.new_buffer((q_dim as u64) * 4, MTLResourceOptions::StorageModeShared);

        // Partials buffer for H3 path
        let num_partitions = clen.div_ceil(PARTITION_TOKENS);
        let stride = HEAD_DIM + 2;
        let partials_buf = device.new_buffer(
            (num_partitions as u64) * (nqh as u64) * (stride as u64) * 4,
            MTLResourceOptions::StorageModeShared,
        );

        // Fill with deterministic values
        let mut rng: u32 = 0xDEAD_BEEF;
        unsafe {
            let q = std::slice::from_raw_parts_mut(q_buf.contents() as *mut f32, q_dim as usize);
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

        // Helper: dispatch reference kernel (old, one TG per Q head)
        let dispatch_ref = |out: &Buffer| -> f64 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipe_ref);
            enc.set_buffer(0, Some(&q_buf), 0);
            enc.set_buffer(1, Some(&k_buf), 0);
            enc.set_buffer(2, Some(&v_buf), 0);
            enc.set_buffer(3, Some(out), 0);
            enc.set_bytes(4, 4, &clen as *const u32 as *const _);
            enc.set_bytes(5, 4, &HEAD_DIM as *const u32 as *const _);
            enc.set_bytes(6, 4, &nqh as *const u32 as *const _);
            enc.set_bytes(7, 4, &nkh as *const u32 as *const _);
            enc.set_bytes(8, 4, &q_dim as *const u32 as *const _);
            enc.set_bytes(9, 4, &kv_dim as *const u32 as *const _);
            enc.set_bytes(10, 4, &scale as *const f32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(nqh as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            let t = Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            t.elapsed().as_secs_f64() * 1e6
        };

        // Helper: dispatch new flash kernel (direct for short cache, partitioned for long)
        let dispatch_flash = |out: &Buffer| -> f64 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            let t;
            if clen <= DIRECT_THRESHOLD {
                enc.set_compute_pipeline_state(&pipe_flash_direct);
                enc.set_buffer(0, Some(&q_buf), 0);
                enc.set_buffer(1, Some(&k_buf), 0);
                enc.set_buffer(2, Some(&v_buf), 0);
                enc.set_buffer(3, Some(out), 0);
                enc.set_bytes(4, 4, &clen as *const u32 as *const _);
                enc.set_bytes(5, 4, &HEAD_DIM as *const u32 as *const _);
                enc.set_bytes(6, 4, &nqh as *const u32 as *const _);
                enc.set_bytes(7, 4, &nkh as *const u32 as *const _);
                enc.set_bytes(8, 4, &q_dim as *const u32 as *const _);
                enc.set_bytes(9, 4, &kv_dim as *const u32 as *const _);
                enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                enc.dispatch_thread_groups(MTLSize::new(nkh as u64, 1, 1), MTLSize::new(256, 1, 1));
                enc.end_encoding();
                t = Instant::now();
                cmd.commit();
                cmd.wait_until_completed();
            } else {
                // Partial pass
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
                // Reduce pass
                enc.set_compute_pipeline_state(&pipe_reduce);
                enc.set_buffer(0, Some(&partials_buf), 0);
                enc.set_buffer(1, Some(out), 0);
                enc.set_bytes(2, 4, &nqh as *const u32 as *const _);
                enc.set_bytes(3, 4, &nkh as *const u32 as *const _);
                enc.set_bytes(4, 4, &num_partitions as *const u32 as *const _);
                enc.dispatch_thread_groups(MTLSize::new(nkh as u64, 1, 1), MTLSize::new(256, 1, 1));
                enc.end_encoding();
                t = Instant::now();
                cmd.commit();
                cmd.wait_until_completed();
            }
            t.elapsed().as_secs_f64() * 1e6
        };

        // Parity check: run both once and compare outputs
        let _ = dispatch_ref(&out_ref);
        let _ = dispatch_flash(&out_new);

        let (max_diff, mean_diff) = unsafe {
            let a = std::slice::from_raw_parts(out_ref.contents() as *const f32, q_dim as usize);
            let b = std::slice::from_raw_parts(out_new.contents() as *const f32, q_dim as usize);
            max_mean_abs_diff(a, b)
        };
        let parity = if max_diff < PARITY_MAX_ABS {
            "PASS"
        } else {
            "FAIL"
        };

        // Save golden reference logits for downstream parity verification
        {
            let golden_dir = std::path::Path::new("benchmarks/golden_logits");
            let _ = std::fs::create_dir_all(golden_dir);
            let label_safe = cfg.label.to_lowercase().replace('/', "");
            let golden_path = golden_dir.join(format!("golden_{}_{}_ref.f32", label_safe, clen));
            unsafe {
                let bytes =
                    std::slice::from_raw_parts(out_ref.contents() as *const u8, q_dim as usize * 4);
                match std::fs::write(&golden_path, bytes) {
                    Ok(()) => eprintln!(
                        "[bench] Golden ref saved: {:?} ({} f32 values)",
                        golden_path, q_dim
                    ),
                    Err(e) => eprintln!("[bench] Warning: golden save failed: {e}"),
                }
            }
        }

        // Warmup reference
        eprint!("  {:<12}  {:>9}  warming ref...", cfg.label, clen);
        for _ in 0..cfg.warmup {
            let _ = dispatch_ref(&out_ref);
        }

        // Measure reference
        let mut base_samples = Vec::with_capacity(cfg.measure);
        for _ in 0..cfg.measure {
            base_samples.push(dispatch_ref(&out_ref));
        }

        // Warmup flash
        eprint!("\r  {:<12}  {:>9}  warming flash...", cfg.label, clen);
        for _ in 0..cfg.warmup {
            let _ = dispatch_flash(&out_new);
        }

        // Measure flash
        let mut flash_samples = Vec::with_capacity(cfg.measure);
        for _ in 0..cfg.measure {
            flash_samples.push(dispatch_flash(&out_new));
        }

        let (base_mean, base_std, base_p50, base_p99, _) = stats(&base_samples);
        let (flash_mean, flash_std, flash_p50, flash_p99, _) = stats(&flash_samples);
        let base_median = base_p50;
        let flash_median = flash_p50;
        let base_tokens_per_sec = 1_000_000.0 / base_median;
        let flash_tokens_per_sec = 1_000_000.0 / flash_median;
        let speedup = base_median / flash_median;
        let base_cv = 100.0 * base_std / base_mean;
        let flash_cv = 100.0 * flash_std / flash_mean;

        let speedup_gate = speedup >= 1.10;
        let cv_gate = base_cv < 20.0 && flash_cv < 20.0;
        let parity_gate = parity == "PASS";
        if !(speedup_gate && cv_gate && parity_gate) {
            gate_failures.push(format!(
                "{} cache_len={}: speedup={:.3}x, base_cv={:.1}%, flash_cv={:.1}%, parity={}",
                cfg.label, clen, speedup, base_cv, flash_cv, parity
            ));
        }

        eprintln!(
            "\r  {:<12}  {:>9}  {:>10.1}  {:>10.1}  {:>9.2}x  {:>10.2e}  {:>10.2e}  {:>10}",
            cfg.label, clen, base_median, flash_median, speedup, max_diff, mean_diff, parity
        );

        // JSONL baseline
        println!(
            r#"{{"heads":"{}", "cache_len":{}, "impl":"baseline", "mean_us":{:.2}, "median_latency_us":{:.2}, "stdev_us":{:.2}, "p99_us":{:.2}, "cv_pct":{:.1}, "throughput_tokens_per_sec":{:.2}, "n":{}}}"#,
            cfg.label,
            clen,
            base_mean,
            base_median,
            base_std,
            base_p99,
            base_cv,
            base_tokens_per_sec,
            base_samples.len()
        );
        // JSONL flash
        println!(
            r#"{{"heads":"{}", "cache_len":{}, "impl":"flash", "mean_us":{:.2}, "median_latency_us":{:.2}, "stdev_us":{:.2}, "p99_us":{:.2}, "cv_pct":{:.1}, "throughput_tokens_per_sec":{:.2}, "speedup":{:.3}, "max_abs_diff":{:.2e}, "mean_abs_diff":{:.2e}, "parity":"{}", "n":{}}}"#,
            cfg.label,
            clen,
            flash_mean,
            flash_median,
            flash_std,
            flash_p99,
            flash_cv,
            flash_tokens_per_sec,
            speedup,
            max_diff,
            mean_diff,
            parity,
            flash_samples.len()
        );
    }

    eprintln!();
    eprintln!("[bench] Timing: host_submit_wait_us (includes command-buffer overhead).");
    eprintln!("[bench] Parity gate: max_abs_diff < 1e-4.");
    eprintln!("[bench] Speedup gate: median speedup >= 1.10x at cache_len >= 512, CV < 20%.");

    if std::env::var_os("LATTICE_DECODE_ENFORCE_GATE").is_some() && !gate_failures.is_empty() {
        eprintln!("[bench] Gate failures:");
        for failure in &gate_failures {
            eprintln!("[bench]   {failure}");
        }
        std::process::exit(2);
    }
}
