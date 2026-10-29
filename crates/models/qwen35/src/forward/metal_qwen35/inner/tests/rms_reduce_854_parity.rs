    use super::*;

    const ORACLE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_qwen35_pre_854_oracle(
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

    for (uint i = lid; i < row_len; i += tgs) {
        x[base + i] = x[base + i] * rms * (1.0f + gamma[i]);
    }
}

kernel void per_head_rms_norm_pre_854_oracle(
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

kernel void fused_residual_add_norm_pre_854_oracle(
    device const float* base     [[buffer(0)]],
    device const float* delta    [[buffer(1)]],
    device float* residual_out   [[buffer(2)]],
    device float* normed_out     [[buffer(3)]],
    device const float* gamma    [[buffer(4)]],
    constant uint& row_len       [[buffer(5)]],
    constant float& eps          [[buffer(6)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]])
{
    constexpr uint WG = 256;
    threadgroup float shared[WG];

    float local_sum = 0.0f;
    for (uint i = lid; i < row_len; i += tgs) {
        float val = base[i] + delta[i];
        residual_out[i] = val;
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
        normed_out[i] = residual_out[i] * rms * (1.0f + gamma[i]);
    }
}

kernel void copy_and_rms_norm_pre_854_oracle(
    device float* src            [[buffer(0)]],
    device float* residual_out   [[buffer(1)]],
    device const float* gamma    [[buffer(2)]],
    constant uint& row_len       [[buffer(3)]],
    constant float& eps          [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    constexpr uint WG = 256;
    threadgroup float shared[WG];

    float local_sum = 0.0f;
    for (uint i = lid; i < row_len; i += tgs) {
        float v = src[i];
        residual_out[i] = v;
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
        src[i] = src[i] * rms * (1.0f + gamma[i]);
    }
}

kernel void copy_and_rms_norm_batch_pre_854_oracle(
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

kernel void fused_residual_add_norm_batch_pre_854_oracle(
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

kernel void per_head_rms_norm_batch_pre_854_oracle(
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

    for (uint i = lid; i < head_dim; i += tgs) {
        x[base + i] = x[base + i] * rms * (1.0f + gamma[i]);
    }
}
"#;

    fn f32_buf(device: &Device, data: &[f32]) -> Buffer {
        device.new_buffer_with_data(
            data.as_ptr() as *const _,
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn read_f32(buf: &Buffer, len: usize) -> Vec<f32> {
        // SAFETY: buffer is StorageModeShared, written by a completed
        // command buffer, and sized for `len` f32 elements.
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, len).to_vec() }
    }

    fn lcg_vec(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (((state >> 32) as u32) as f32 / u32::MAX as f32 - 0.5) * 4.0
            })
            .collect()
    }

    /// Builds `num_rows` rows of `row_len` elements each, with row 0 forced to
    /// an exact all-near-zero constant and row 1 forced to the exact smallest
    /// positive f32 subnormal (`f32::from_bits(1)`, ~1.4e-45) — two distinct
    /// epsilon-floor edge cases a whole-row-of-ordinary-magnitude fixture
    /// cannot exercise: the reduction's `rsqrt(sum_sq / width + eps)` is
    /// dominated by `eps` only when `sum_sq` itself is vanishingly small
    /// across the *whole* row, not just one lane. Remaining rows are ordinary
    /// pseudo-random data (regression coverage).
    fn row_batch(seed: u64, row_len: usize, num_rows: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(row_len * num_rows);
        for r in 0..num_rows {
            match r {
                0 => v.extend(std::iter::repeat_n(1e-20f32, row_len)),
                1 => v.extend(std::iter::repeat_n(f32::from_bits(1), row_len)),
                _ => v.extend(lcg_vec(seed + r as u64, row_len)),
            }
        }
        v
    }

    /// Confirms `row_batch`'s row 0 / row 1 actually landed as the exact
    /// near-zero / subnormal constants (construction proof) and prints their
    /// sum-of-squares so a test run makes the new coverage visible, not just
    /// implicit in a later bit-identical comparison.
    fn assert_edge_rows(label: &str, data: &[f32], row_len: usize) {
        let near_zero = &data[0..row_len];
        let subnormal = &data[row_len..2 * row_len];
        assert!(
            near_zero.iter().all(|v| *v == 1e-20f32),
            "{label}: row 0 fixture must be the exact all-near-zero constant"
        );
        assert!(
            subnormal.iter().all(|v| v.to_bits() == 1),
            "{label}: row 1 fixture must be the exact f32 subnormal (from_bits(1))"
        );
        eprintln!(
            "[RMS_854_FIXTURE] {label}: row0(near_zero) sum_sq={:.3e} \
                     row1(subnormal) sum_sq={:.3e}",
            near_zero
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>(),
            subnormal
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>(),
        );
    }

    fn pipeline_for(device: &Device, lib: &Library, name: &str) -> ComputePipelineState {
        let func = lib
            .get_function(name, None)
            .unwrap_or_else(|e| panic!("kernel {name} missing: {e}"));
        device
            .new_compute_pipeline_state_with_function(&func)
            .unwrap_or_else(|e| panic!("pipeline for {name} failed: {e}"))
    }

    /// rms_norm_qwen35(x, gamma, row_len, num_rows, eps) — in-place on x.
    #[allow(clippy::too_many_arguments)]
    fn run_rms_norm_qwen35(
        device: &Device,
        queue: &CommandQueue,
        lib: &Library,
        name: &str,
        x: &[f32],
        gamma: &[f32],
        row_len: u32,
        num_rows: u32,
        eps: f32,
    ) -> Vec<f32> {
        let pipe = pipeline_for(device, lib, name);
        let x_buf = f32_buf(device, x);
        let gamma_buf = f32_buf(device, gamma);
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&gamma_buf), 0);
        enc.set_bytes(2, 4, &row_len as *const u32 as *const _);
        enc.set_bytes(3, 4, &num_rows as *const u32 as *const _);
        enc.set_bytes(4, 4, &eps as *const f32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(num_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        read_f32(&x_buf, x.len())
    }

    /// per_head_rms_norm(x, gamma, num_heads, head_dim, eps) — in-place on x.
    #[allow(clippy::too_many_arguments)]
    fn run_per_head_rms_norm(
        device: &Device,
        queue: &CommandQueue,
        lib: &Library,
        name: &str,
        x: &[f32],
        gamma: &[f32],
        num_heads: u32,
        head_dim: u32,
        eps: f32,
    ) -> Vec<f32> {
        let pipe = pipeline_for(device, lib, name);
        let x_buf = f32_buf(device, x);
        let gamma_buf = f32_buf(device, gamma);
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&gamma_buf), 0);
        enc.set_bytes(2, 4, &num_heads as *const u32 as *const _);
        enc.set_bytes(3, 4, &head_dim as *const u32 as *const _);
        enc.set_bytes(4, 4, &eps as *const f32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(num_heads as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        read_f32(&x_buf, x.len())
    }

    /// fused_residual_add_norm(base, delta, residual_out, normed_out, gamma,
    /// row_len, eps) — single threadgroup, returns normed_out.
    #[allow(clippy::too_many_arguments)]
    fn run_fused_residual_add_norm(
        device: &Device,
        queue: &CommandQueue,
        lib: &Library,
        name: &str,
        base: &[f32],
        delta: &[f32],
        gamma: &[f32],
        row_len: u32,
        eps: f32,
    ) -> Vec<f32> {
        let pipe = pipeline_for(device, lib, name);
        let base_buf = f32_buf(device, base);
        let delta_buf = f32_buf(device, delta);
        let residual_buf = f32_buf(device, &vec![0.0f32; row_len as usize]);
        let normed_buf = f32_buf(device, &vec![0.0f32; row_len as usize]);
        let gamma_buf = f32_buf(device, gamma);
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&base_buf), 0);
        enc.set_buffer(1, Some(&delta_buf), 0);
        enc.set_buffer(2, Some(&residual_buf), 0);
        enc.set_buffer(3, Some(&normed_buf), 0);
        enc.set_buffer(4, Some(&gamma_buf), 0);
        enc.set_bytes(5, 4, &row_len as *const u32 as *const _);
        enc.set_bytes(6, 4, &eps as *const f32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        read_f32(&normed_buf, row_len as usize)
    }

    /// copy_and_rms_norm(src, residual_out, gamma, row_len, eps) — single
    /// threadgroup, returns src (normalized in place).
    #[allow(clippy::too_many_arguments)]
    fn run_copy_and_rms_norm(
        device: &Device,
        queue: &CommandQueue,
        lib: &Library,
        name: &str,
        src: &[f32],
        gamma: &[f32],
        row_len: u32,
        eps: f32,
    ) -> Vec<f32> {
        let pipe = pipeline_for(device, lib, name);
        let src_buf = f32_buf(device, src);
        let residual_buf = f32_buf(device, &vec![0.0f32; row_len as usize]);
        let gamma_buf = f32_buf(device, gamma);
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&src_buf), 0);
        enc.set_buffer(1, Some(&residual_buf), 0);
        enc.set_buffer(2, Some(&gamma_buf), 0);
        enc.set_bytes(3, 4, &row_len as *const u32 as *const _);
        enc.set_bytes(4, 4, &eps as *const f32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        read_f32(&src_buf, row_len as usize)
    }

    /// copy_and_rms_norm_batch(src, residual_out, gamma, row_len, num_rows,
    /// eps) — returns src (normalized in place).
    #[allow(clippy::too_many_arguments)]
    fn run_copy_and_rms_norm_batch(
        device: &Device,
        queue: &CommandQueue,
        lib: &Library,
        name: &str,
        src: &[f32],
        gamma: &[f32],
        row_len: u32,
        num_rows: u32,
        eps: f32,
    ) -> Vec<f32> {
        let pipe = pipeline_for(device, lib, name);
        let src_buf = f32_buf(device, src);
        let residual_buf = f32_buf(device, &vec![0.0f32; src.len()]);
        let gamma_buf = f32_buf(device, gamma);
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&src_buf), 0);
        enc.set_buffer(1, Some(&residual_buf), 0);
        enc.set_buffer(2, Some(&gamma_buf), 0);
        enc.set_bytes(3, 4, &row_len as *const u32 as *const _);
        enc.set_bytes(4, 4, &num_rows as *const u32 as *const _);
        enc.set_bytes(5, 4, &eps as *const f32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(num_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        read_f32(&src_buf, src.len())
    }

    /// fused_residual_add_norm_batch(base, delta, residual_out, normed_out,
    /// gamma, row_len, num_rows, eps) — returns normed_out.
    #[allow(clippy::too_many_arguments)]
    fn run_fused_residual_add_norm_batch(
        device: &Device,
        queue: &CommandQueue,
        lib: &Library,
        name: &str,
        base: &[f32],
        delta: &[f32],
        gamma: &[f32],
        row_len: u32,
        num_rows: u32,
        eps: f32,
    ) -> Vec<f32> {
        let pipe = pipeline_for(device, lib, name);
        let n = (row_len * num_rows) as usize;
        let base_buf = f32_buf(device, base);
        let delta_buf = f32_buf(device, delta);
        let residual_buf = f32_buf(device, &vec![0.0f32; n]);
        let normed_buf = f32_buf(device, &vec![0.0f32; n]);
        let gamma_buf = f32_buf(device, gamma);
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&base_buf), 0);
        enc.set_buffer(1, Some(&delta_buf), 0);
        enc.set_buffer(2, Some(&residual_buf), 0);
        enc.set_buffer(3, Some(&normed_buf), 0);
        enc.set_buffer(4, Some(&gamma_buf), 0);
        enc.set_bytes(5, 4, &row_len as *const u32 as *const _);
        enc.set_bytes(6, 4, &num_rows as *const u32 as *const _);
        enc.set_bytes(7, 4, &eps as *const f32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(num_rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        read_f32(&normed_buf, n)
    }

    /// per_head_rms_norm_batch(x, gamma, num_tokens, num_heads, head_dim,
    /// eps) — returns x (normalized in place).
    #[allow(clippy::too_many_arguments)]
    fn run_per_head_rms_norm_batch(
        device: &Device,
        queue: &CommandQueue,
        lib: &Library,
        name: &str,
        x: &[f32],
        gamma: &[f32],
        num_tokens: u32,
        num_heads: u32,
        head_dim: u32,
        eps: f32,
    ) -> Vec<f32> {
        let pipe = pipeline_for(device, lib, name);
        let x_buf = f32_buf(device, x);
        let gamma_buf = f32_buf(device, gamma);
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&gamma_buf), 0);
        enc.set_bytes(2, 4, &num_tokens as *const u32 as *const _);
        enc.set_bytes(3, 4, &num_heads as *const u32 as *const _);
        enc.set_bytes(4, 4, &head_dim as *const u32 as *const _);
        enc.set_bytes(5, 4, &eps as *const f32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((num_tokens * num_heads) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        read_f32(&x_buf, x.len())
    }

    #[test]
    fn qwen35_rms_kernels_match_pre_854_oracle_on_finite_input() {
        // Fail-closed device probe (matches `metal::inner::tests::metal_test_device`'s
        // convention): a runner with no Metal device reports a distinct skip, unless
        // LATTICE_METAL_TEST_ENFORCE=1 is set, in which case a missing device is a
        // hard failure rather than a silent pass. Without this, a CI runner that lost
        // its Metal device would report this parity gate as `ok` without ever
        // compiling or comparing either library.
        let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
        let Some(device) = Device::system_default() else {
            eprintln!(
                "[METAL_TEST_SKIP] context=qwen35_rms_kernels_match_pre_854_oracle_on_finite_input \
                         reason=no_metal_device"
            );
            assert!(
                !enforce,
                "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present \
                         (qwen35_rms_kernels_match_pre_854_oracle_on_finite_input)"
            );
            return;
        };
        let _gpu = gpu_test_lock();
        let opts = CompileOptions::new();
        let oracle_lib = device
            .new_library_with_source(ORACLE_MSL, &opts)
            .expect("frozen pre-#854 oracle MSL must compile");
        let live_lib = device
            .new_library_with_source(MSL_SOURCE, &opts)
            .expect("live post-#854 MSL_SOURCE must compile");
        let queue = device.new_command_queue();

        // Non-multiple-of-256 dims exercise the strided accumulation loop's
        // tail; includes small/negative/near-zero lanes.
        let row_len = 130u32;
        let num_rows = 3u32;
        let head_dim = 64u32;
        let num_heads = 5u32;
        let num_tokens = 3u32;
        let eps = 1e-6f32;

        // 1. rms_norm_qwen35
        {
            let x = row_batch(11, row_len as usize, num_rows as usize);
            assert_edge_rows("rms_norm_qwen35", &x, row_len as usize);
            let gamma = lcg_vec(12, row_len as usize);
            let oracle = run_rms_norm_qwen35(
                &device,
                &queue,
                &oracle_lib,
                "rms_norm_qwen35_pre_854_oracle",
                &x,
                &gamma,
                row_len,
                num_rows,
                eps,
            );
            let live = run_rms_norm_qwen35(
                &device,
                &queue,
                &live_lib,
                "rms_norm_qwen35",
                &x,
                &gamma,
                row_len,
                num_rows,
                eps,
            );
            assert_eq!(
                oracle, live,
                "rms_norm_qwen35 diverged from pre-#854 oracle"
            );
        }

        // 2. per_head_rms_norm
        {
            let x = row_batch(21, head_dim as usize, num_heads as usize);
            assert_edge_rows("per_head_rms_norm", &x, head_dim as usize);
            let gamma = lcg_vec(22, head_dim as usize);
            let oracle = run_per_head_rms_norm(
                &device,
                &queue,
                &oracle_lib,
                "per_head_rms_norm_pre_854_oracle",
                &x,
                &gamma,
                num_heads,
                head_dim,
                eps,
            );
            let live = run_per_head_rms_norm(
                &device,
                &queue,
                &live_lib,
                "per_head_rms_norm",
                &x,
                &gamma,
                num_heads,
                head_dim,
                eps,
            );
            assert_eq!(
                oracle, live,
                "per_head_rms_norm diverged from pre-#854 oracle"
            );
        }

        // 3. fused_residual_add_norm — single threadgroup, no row/head batch
        // dimension, so near-zero/subnormal residual-input coverage runs as
        // separate dispatches instead of dedicated rows within one dispatch.
        for (label, base, delta) in [
            (
                "normal",
                lcg_vec(31, row_len as usize),
                lcg_vec(32, row_len as usize),
            ),
            (
                "near_zero_residual",
                vec![1e-20f32; row_len as usize],
                vec![1e-20f32; row_len as usize],
            ),
            (
                "subnormal_residual",
                vec![f32::from_bits(1); row_len as usize],
                vec![f32::from_bits(1); row_len as usize],
            ),
        ] {
            let gamma = lcg_vec(33, row_len as usize);
            let oracle = run_fused_residual_add_norm(
                &device,
                &queue,
                &oracle_lib,
                "fused_residual_add_norm_pre_854_oracle",
                &base,
                &delta,
                &gamma,
                row_len,
                eps,
            );
            let live = run_fused_residual_add_norm(
                &device,
                &queue,
                &live_lib,
                "fused_residual_add_norm",
                &base,
                &delta,
                &gamma,
                row_len,
                eps,
            );
            assert_eq!(
                oracle, live,
                "fused_residual_add_norm ({label} fixture) diverged from pre-#854 oracle"
            );
            eprintln!("[RMS_854_FIXTURE] fused_residual_add_norm: {label} fixture executed");
        }

        // 4. copy_and_rms_norm — single threadgroup, no row/head batch
        // dimension, so near-zero/subnormal coverage runs as separate
        // dispatches instead of dedicated rows within one dispatch.
        for (label, src) in [
            ("normal", lcg_vec(41, row_len as usize)),
            ("near_zero_residual", vec![1e-20f32; row_len as usize]),
            (
                "subnormal_residual",
                vec![f32::from_bits(1); row_len as usize],
            ),
        ] {
            let gamma = lcg_vec(42, row_len as usize);
            let oracle = run_copy_and_rms_norm(
                &device,
                &queue,
                &oracle_lib,
                "copy_and_rms_norm_pre_854_oracle",
                &src,
                &gamma,
                row_len,
                eps,
            );
            let live = run_copy_and_rms_norm(
                &device,
                &queue,
                &live_lib,
                "copy_and_rms_norm",
                &src,
                &gamma,
                row_len,
                eps,
            );
            assert_eq!(
                oracle, live,
                "copy_and_rms_norm ({label} fixture) diverged from pre-#854 oracle"
            );
            eprintln!("[RMS_854_FIXTURE] copy_and_rms_norm: {label} fixture executed");
        }

        // 5. copy_and_rms_norm_batch
        {
            let src = row_batch(51, row_len as usize, num_rows as usize);
            assert_edge_rows("copy_and_rms_norm_batch", &src, row_len as usize);
            let gamma = lcg_vec(52, row_len as usize);
            let oracle = run_copy_and_rms_norm_batch(
                &device,
                &queue,
                &oracle_lib,
                "copy_and_rms_norm_batch_pre_854_oracle",
                &src,
                &gamma,
                row_len,
                num_rows,
                eps,
            );
            let live = run_copy_and_rms_norm_batch(
                &device,
                &queue,
                &live_lib,
                "copy_and_rms_norm_batch",
                &src,
                &gamma,
                row_len,
                num_rows,
                eps,
            );
            assert_eq!(
                oracle, live,
                "copy_and_rms_norm_batch diverged from pre-#854 oracle"
            );
        }

        // 6. fused_residual_add_norm_batch — both residual inputs (base and
        // delta) get dedicated near-zero/subnormal rows, since this kernel's
        // reduction runs over the post-add residual sum, not either input
        // alone.
        {
            let base = row_batch(61, row_len as usize, num_rows as usize);
            assert_edge_rows(
                "fused_residual_add_norm_batch/base",
                &base,
                row_len as usize,
            );
            let delta = row_batch(62, row_len as usize, num_rows as usize);
            assert_edge_rows(
                "fused_residual_add_norm_batch/delta",
                &delta,
                row_len as usize,
            );
            let gamma = lcg_vec(63, row_len as usize);
            let oracle = run_fused_residual_add_norm_batch(
                &device,
                &queue,
                &oracle_lib,
                "fused_residual_add_norm_batch_pre_854_oracle",
                &base,
                &delta,
                &gamma,
                row_len,
                num_rows,
                eps,
            );
            let live = run_fused_residual_add_norm_batch(
                &device,
                &queue,
                &live_lib,
                "fused_residual_add_norm_batch",
                &base,
                &delta,
                &gamma,
                row_len,
                num_rows,
                eps,
            );
            assert_eq!(
                oracle, live,
                "fused_residual_add_norm_batch diverged from pre-#854 oracle"
            );
        }

        // 7. per_head_rms_norm_batch — row-major layout is
        // base = t*(num_heads*head_dim) + head*head_dim, so a batch of
        // num_heads*num_tokens rows of width head_dim lines up exactly
        // with row_batch's row-major ordering.
        {
            let x = row_batch(71, head_dim as usize, (num_heads * num_tokens) as usize);
            assert_edge_rows("per_head_rms_norm_batch", &x, head_dim as usize);
            let gamma = lcg_vec(72, head_dim as usize);
            let oracle = run_per_head_rms_norm_batch(
                &device,
                &queue,
                &oracle_lib,
                "per_head_rms_norm_batch_pre_854_oracle",
                &x,
                &gamma,
                num_tokens,
                num_heads,
                head_dim,
                eps,
            );
            let live = run_per_head_rms_norm_batch(
                &device,
                &queue,
                &live_lib,
                "per_head_rms_norm_batch",
                &x,
                &gamma,
                num_tokens,
                num_heads,
                head_dim,
                eps,
            );
            assert_eq!(
                oracle, live,
                "per_head_rms_norm_batch diverged from pre-#854 oracle"
            );
        }
    }
