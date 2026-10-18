mod checkpoint_refusals;
mod chunked_prefill;
mod cross_turn_kv_cache;
mod decode_attention;
mod dispatch;
mod gdn_chunked_scan;
mod grammar;
mod layers;
mod lm_head;
mod lora;
mod moe;
mod mtp_hidden_tap;
mod mtp_verify;
mod mtp_weight_load;
mod multimodal_vision;
mod path_proof_bytes;
mod perplexity;
mod pooled_embeddings;
mod prefix_cache_disposition;
mod prefix_cache_route;
mod prefix_cache_slots;
mod q3;
mod q4_signed_scale;
mod q8_nan_guard;
mod self_spec;
mod speculative_route;
mod stop_token_sweep;
mod suffix_preflight;

use super::super::{
    LM_HEAD_TOPK_TIE_EPSILON, LM_HEAD_TOPK_TIE_EPSILON_Q4, TopkSetAgreement,
    topk_set_agreement_or_boundary_tie,
};
// Fixture helpers for the MTP Q4/F16 loader tests below now live in
// the portable `mtp_resolve_tests` module (moved out of
// `mod inner` so the pure resolver tests compile without
// `metal-gpu` -- these Device-gated integration tests still share
// the same fixture writers).
use super::super::mtp_weights::{
    mtp_resolve_tests::{
        mtp_proj_names_and_shapes, q4_const_roundtrip, write_full_mtp_fixture,
        write_tiny_f16_fixture, write_tiny_q4_fixture, write_tiny_q4_fixture_per_expert_row,
        write_tiny_q4_fixture_per_row,
    },
    mtp_tensor_path,
};
use super::*;
use crate::measurement::gpu_test_lock;
use crate::model::qwen35::{
    CommonLayerWeights, DenseFfnWeights, FeedForwardWeights, FullAttentionLayerWeights,
};
use crate::model::qwen35_config::LayerType;
use crate::tokenizer::common::Tokenizer;

/// Behavioral guard for the allocation policy `from_q4_dir` uses to size
/// typed Metal GDN state (`active_gdn_layer_indices`, called at the
/// `num_linear` assignment above the `MetalGdnState::new` call in
/// `from_q4_dir`). Exercises the same pure decision the constructor makes
/// — which layer indices are GatedDeltaNet AND unpruned — without a GPU
/// or a model fixture download.
#[test]
fn q4_constructor_allocates_typed_gdn_state_for_active_layers() {
    let (cfg, _weights) = tiny_hybrid_fixture();
    // tiny_hybrid_fixture: layer_types = [Linear, Linear, Linear, Full],
    // layer_mask all active — every linear-attention layer is selected,
    // the full-attention layer (index 3) is excluded.
    assert_eq!(super::active_gdn_layer_indices(&cfg), vec![0, 1, 2]);
    assert_eq!(
        super::active_gdn_layer_indices(&cfg).len(),
        cfg.num_active_linear_attention_layers()
    );

    let mut pruned = cfg.clone();
    pruned.layer_mask[1] = false;
    assert_eq!(
        super::active_gdn_layer_indices(&pruned),
        vec![0, 2],
        "a pruned linear-attention layer must not receive typed GDN state"
    );
    assert_eq!(
        super::active_gdn_layer_indices(&pruned).len(),
        pruned.num_active_linear_attention_layers()
    );
}

#[test]
fn shared_pipeline_builder_resolves_every_required_pipeline() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(device) = Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _gpu = gpu_test_lock();
    let (cfg, _) = tiny_hybrid_fixture();

    let pipelines = build_pipelines(&device, &cfg)
        .expect("every required pipeline lookup in the shared builder must resolve");
    assert_eq!(
        pipelines.lm_head_block_topk_f16.len(),
        LM_HEAD_LOCAL_KS.len()
    );
    assert_eq!(
        pipelines.lm_head_block_topk_q4.len(),
        LM_HEAD_LOCAL_KS.len()
    );
}

/// #854: guards against a reintroduced manual copy of the RMS-norm
/// reduction tree across qwen35.metal's seven RMS-norm-family kernels
/// (rms_norm_qwen35, per_head_rms_norm, fused_residual_add_norm,
/// copy_and_rms_norm, copy_and_rms_norm_batch, fused_residual_add_norm_batch,
/// per_head_rms_norm_batch). `rms_inv_from_local_sum` must have exactly one
/// definition (the shared fragment) plus exactly seven call sites in the
/// assembled qwen35.metal translation unit.
#[test]
fn qwen35_rms_norm_kernels_use_shared_reduction_helper() {
    let helper_occurrences = MSL_SOURCE.matches("rms_inv_from_local_sum(").count();
    assert_eq!(
        helper_occurrences, 8,
        "expected 1 definition (rms_reduce.metal fragment) + 7 call sites in \
                 the assembled qwen35.metal, got {helper_occurrences}"
    );
    let inline_leftover = MSL_SOURCE.matches("shared[lid] = local_sum;").count();
    assert_eq!(
        inline_leftover, 1,
        "expected exactly one occurrence, inside rms_inv_from_local_sum's own \
                 body; any additional occurrence is a reintroduced manual copy of the \
                 reduction tree (no thirteenth manual copy)"
    );
}

/// #854 numeric parity fixtures (pre-refactor vs. post-refactor) for all
/// seven qwen35.metal RMS-norm-family kernels. Each kernel's pre-refactor
/// inline reduction tree is frozen verbatim below (renamed with a
/// `_pre_854_oracle` suffix, bodies otherwise byte-identical to what this
/// PR replaced) and compiled as an independent oracle library; the live
/// post-refactor kernel is dispatched from the real `MSL_SOURCE`. Both run
/// on the same GPU with the same compile options and identical finite
/// input — the only source difference is inline-body vs. helper-call — so
/// bit-identical output proves the extraction preserved barrier count, tree
/// order, and the rsqrt expression for every call site (RMS-norm is a
/// nonlinearity; f32 reduction order is part of the contract).
mod rms_reduce_854_parity {
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
            .expect("live post-#854 qwen35.metal (MSL_SOURCE) must compile");
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
}

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

// Mutation proof: reverting the is_finite() guard makes this test pass NaN
// through undetected (NaN.abs() is NaN, m.max(NaN) == m, so amax stays 0.0,
// scale stays 0.0, and NaN * 0.0 = NaN → clamp → 0 silently). With the guard
// the function returns Err before any NaN reaches the scale computation.
#[test]
fn quantize_row_q8_0_rejects_nan() {
    let mut row = vec![0.0f32; 32];
    row[7] = f32::NAN;
    assert!(
        quantize_row_q8_0(&row).is_err(),
        "quantize_row_q8_0 must return Err when a block element is NaN"
    );
}

#[test]
fn quantize_row_q8_0_rejects_inf() {
    let mut row = vec![0.5f32; 32];
    row[0] = f32::INFINITY;
    assert!(
        quantize_row_q8_0(&row).is_err(),
        "quantize_row_q8_0 must return Err when a block element is +Inf"
    );
}

/// Sweep all 65 536 u16 bit patterns and assert that `convert_f16_row` produces
/// bit-identical output to the scalar `f16_to_f32` reference for every pattern
/// (regression guard: `convert_f16_row` is itself scalar-only post-#568, so this
/// mainly pins the loop's correctness against the reference implementation).
///
/// NaN comparison policy: both outputs are NaN ⇒ pass (any NaN payload is
/// acceptable; the bit pattern of a NaN is not architecturally meaningful).
/// Non-NaN values are compared by exact bit equality, which is correct because
/// f16→f32 widening is lossless for every representable value including
/// subnormals, ±0, and ±∞.
#[test]
fn test_convert_f16_row_matches_scalar_all_patterns() {
    let mut src = [0u16; 65536];
    for (i, v) in src.iter_mut().enumerate() {
        *v = i as u16;
    }
    let mut neon_out = vec![0.0f32; 65536];
    // SAFETY: src and neon_out are both 65536 elements, stack-allocated and valid.
    unsafe {
        convert_f16_row(src.as_ptr(), neon_out.as_mut_ptr(), 65536);
    }
    for bits in 0u16..=0xffffu16 {
        let scalar = f16_to_f32(bits);
        let fast = neon_out[bits as usize];
        if scalar.is_nan() {
            assert!(
                fast.is_nan(),
                "bits=0x{bits:04x}: scalar=NaN but fast={fast}"
            );
        } else {
            assert_eq!(
                scalar.to_bits(),
                fast.to_bits(),
                "bits=0x{bits:04x}: scalar={scalar} fast={fast}"
            );
        }
    }
}

#[test]
fn test_read_buffer_f16_conversion() {
    // Verify read_buffer_f16 produces correct f32 values
    let Some(device) = Device::system_default() else {
        return;
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
    let Some(device) = Device::system_default() else {
        return;
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
    let _gpu_guard = gpu_test_lock();
    // Run a small GEMM through both f32 and f16 paths, compare results.
    let Some(device) = Device::system_default() else {
        return;
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
            MTLSize::new((n as u64).div_ceil(tile), (m as u64).div_ceil(tile), 1),
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
    let Some(device) = Device::system_default() else {
        return;
    };

    let data = vec![1.0f32; 1024];
    let buf_f32 = make_buffer(&device, &data, "f32");
    let buf_f16 = make_buffer_f16(&device, &data, "f16");

    assert_eq!(buf_f32.length(), 1024 * 4); // 4 bytes per f32
    assert_eq!(buf_f16.length(), 1024 * 2); // 2 bytes per f16
}

/// Old 3-pass decode kernel preserved as the correctness reference.
const MSL_DECODE_REFERENCE: &str = include_str!("../../shaders/decode_reference.metal");

/// Serializes tests in this module that mutate `LATTICE_COMPACT_TOPK` in the
/// real process environment. `set_var`/`remove_var` are `unsafe` because they
/// can race with a read on another thread; this is the same per-variable lock
/// convention `with_self_spec_env` (below) uses for `LATTICE_SELF_SPEC`.
/// Restores the prior value (including "was unset") on the way out, even if
/// `f` panics.
fn with_compact_topk_env<R>(value: &str, f: impl FnOnce() -> R) -> R {
    use std::sync::Mutex;
    static ENV_LOCK: Mutex<()> = Mutex::new(());
    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let prior = std::env::var("LATTICE_COMPACT_TOPK").ok();
    // SAFETY: serialized by `ENV_LOCK` above — this lock only guards writers
    // of LATTICE_COMPACT_TOPK against each other; it does not stop other
    // tests from reading the environment concurrently.
    unsafe {
        std::env::set_var("LATTICE_COMPACT_TOPK", value);
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    // SAFETY: see above.
    unsafe {
        match &prior {
            Some(v) => std::env::set_var("LATTICE_COMPACT_TOPK", v),
            None => std::env::remove_var("LATTICE_COMPACT_TOPK"),
        }
    }
    match result {
        Ok(r) => r,
        Err(payload) => std::panic::resume_unwind(payload),
    }
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
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
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

fn tiny_metal_qwen35_vision_fixture() -> (Qwen35Config, ModelWeights) {
    let (mut cfg, weights) = tiny_metal_qwen35_fixture();
    cfg.rope_parameters = Some(crate::model::qwen35_config::RopeParams {
        rope_theta: 10_000_000.0,
        partial_rotary_factor: Some(0.25),
        mrope_section: Some(vec![11, 11, 10]),
        mrope_interleaved: Some(true),
    });
    cfg.vision_config = Some(crate::model::qwen35_config::VisionModelConfig {
        depth: 1,
        hidden_size: 4,
        num_heads: 1,
        patch_size: 1,
        spatial_merge_size: 2,
        out_hidden_size: cfg.hidden_size,
        temporal_patch_size: 1,
        num_position_embeddings: 4,
        in_channels: 3,
        deepstack_visual_indexes: Vec::new(),
        intermediate_size: Some(8),
    });
    cfg.image_token_id = Some(5);
    cfg.video_token_id = Some(6);
    cfg.vision_start_token_id = Some(7);
    cfg.vision_end_token_id = Some(8);
    (cfg, weights)
}

fn synthetic_mtp_weights_for_test(device: &Device, cfg: &Qwen35Config) -> MetalMtpWeights {
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let q_dim = cfg.full_q_dim();
    let kv_dim = cfg.full_kv_dim();

    // fc shape: [hidden, 2*hidden]; identity on the embedding half so MTP
    // output equals the normed embedding when attention/MLP weights are zero.
    let mut fc = vec![0.0f32; hidden * 2 * hidden];
    for i in 0..hidden {
        fc[i * (2 * hidden) + i] = 1.0;
    }

    MetalMtpWeights {
        fc: make_buffer_f16(device, &fc, "test.mtp.fc"),
        pre_fc_norm_embedding: make_buffer(
            device,
            &vec![1.0; hidden],
            "test.mtp.pre_fc_norm_embedding",
        ),
        pre_fc_norm_hidden: make_buffer(device, &vec![1.0; hidden], "test.mtp.pre_fc_norm_hidden"),
        layers: vec![MetalMtpLayerWeights {
            input_layernorm: make_buffer(device, &vec![1.0; hidden], "test.mtp.input_ln"),
            post_attention_layernorm: make_buffer(
                device,
                &vec![1.0; hidden],
                "test.mtp.post_attn_ln",
            ),
            q_proj: make_buffer_f16(device, &vec![0.0; 2 * q_dim * hidden], "test.mtp.q_proj"),
            k_proj: make_buffer_f16(device, &vec![0.0; kv_dim * hidden], "test.mtp.k_proj"),
            v_proj: make_buffer_f16(device, &vec![0.0; kv_dim * hidden], "test.mtp.v_proj"),
            o_proj: make_buffer_f16(device, &vec![0.0; hidden * q_dim], "test.mtp.o_proj"),
            q_norm: make_buffer(device, &vec![1.0; cfg.head_dim], "test.mtp.q_norm"),
            k_norm: make_buffer(device, &vec![1.0; cfg.head_dim], "test.mtp.k_norm"),
            mlp: MetalMtpDenseMlpWeights {
                gate_proj: make_buffer_f16(
                    device,
                    &vec![0.0; inter * hidden],
                    "test.mtp.gate_proj",
                ),
                up_proj: make_buffer_f16(device, &vec![0.0; inter * hidden], "test.mtp.up_proj"),
                down_proj: make_buffer_f16(
                    device,
                    &vec![0.0; hidden * inter],
                    "test.mtp.down_proj",
                ),
            },
        }],
        norm: make_buffer(device, &vec![1.0; hidden], "test.mtp.norm"),
        hidden_tap: MtpHiddenTap::PostFinalNorm,
    }
}

fn metal_state_with_synthetic_mtp_for_test(
    weights: &ModelWeights,
    cfg: &Qwen35Config,
    rotation: Option<crate::quant::quarot::hadamard::RandomizedHadamard>,
) -> MetalQwen35State {
    let mut engine = MetalQwen35Engine::new(weights, cfg)
        .expect("tiny MetalQwen35Engine with MTP fixture constructs");
    let mut mtp_weights = synthetic_mtp_weights_for_test(&engine.device, cfg);
    // Mirror the loader's decision: a rotated checkpoint keeps the pre-norm hidden.
    mtp_weights.hidden_tap = MtpHiddenTap::for_checkpoint(rotation.is_some());
    engine.mtp_weights = Some(mtp_weights);
    engine.quarot_rotation = rotation;
    let session = engine.new_session(16).expect("tiny MTP session constructs");
    MetalQwen35State {
        engine,
        session,
        lora: None,
        use_gdn_chunked: true,
        use_kv_f16: false,
        cross_turn_prefix_cache: MetalCrossTurnPrefixCache::default(),
        path_proof_enabled: false,
        path_proof: PathProofCounters::default(),
    }
}

// Issue #1341: a K=1 draft rejection must leave the MTP KV cache cursor at the
// row `mtp_forward_one` actually wrote, not one row past it. `mtp_forward_one`
// writes exactly one row (for `pending_token`) and advances `mtp.cache.seq_len`
// by 1 before verification even starts; `verify_tokens_batched` then snapshots
// that already-advanced value into `mtp_base_seq_len`. On rejection,
// `rollback_speculative_state_to` adds the verifier's accepted-token `slot` on
// top of that snapshot, double-counting the draft's own advance.
//
// A zeroed `fc` projection collapses the MTP head's output to the zero vector
// for any input, so `mtp_forward_one` always drafts token 0 (all-zero logits,
// first-wins argmax). `tiny_metal_qwen35_fixture`'s target stack has zero
// attention/FFN weights too, so its prediction after processing token `X` is
// driven entirely by `RMSNorm(embed_tokens[X])` through the tied lm_head — for
// `X == 2` (`embed_tokens[2] `'s only nonzero component is `+1`, and index 2 is
// the first vocab entry in that residue class) that prediction is token 2
// itself, which never equals the constant draft token 0. Round 1 is therefore
// a deterministic reject, driven through the real `generate_greedy_mtp` path.
fn constant_zero_draft_mtp_weights_for_test(
    device: &Device,
    cfg: &Qwen35Config,
) -> MetalMtpWeights {
    let mut weights = synthetic_mtp_weights_for_test(device, cfg);
    weights.fc = make_buffer_f16(
        device,
        &vec![0.0f32; cfg.hidden_size * 2 * cfg.hidden_size],
        "test.mtp.fc.constant_zero_draft",
    );
    weights
}

fn metal_state_with_constant_zero_draft_mtp_for_test(
    weights: &ModelWeights,
    cfg: &Qwen35Config,
) -> MetalQwen35State {
    let mut engine = MetalQwen35Engine::new(weights, cfg)
        .expect("tiny MetalQwen35Engine with constant-draft MTP fixture constructs");
    engine.mtp_weights = Some(constant_zero_draft_mtp_weights_for_test(
        &engine.device,
        cfg,
    ));
    let session = engine.new_session(16).expect("tiny MTP session constructs");
    MetalQwen35State {
        engine,
        session,
        lora: None,
        use_gdn_chunked: true,
        use_kv_f16: false,
        cross_turn_prefix_cache: MetalCrossTurnPrefixCache::default(),
        path_proof_enabled: false,
        path_proof: PathProofCounters::default(),
    }
}

fn assert_forward_rows_close(label: &str, left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len(), "{label}: row lengths must match");
    let max_abs_diff = left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_abs_diff < 1e-4, "{label}: max_abs_diff={max_abs_diff}");
}

/// Byte-for-byte snapshot of every KV buffer (K and V, every layer),
/// including unwritten rows beyond `seq_len`. Used to prove a rejected
/// call performed no GPU write anywhere in the cache, not just that
/// `seq_len` held still.
fn snapshot_kv_bytes(state: &MetalQwen35State) -> Vec<Vec<u8>> {
    state
        .session
        .kv_cache
        .k_bufs
        .iter()
        .chain(state.session.kv_cache.v_bufs.iter())
        .map(|buf| {
            // SAFETY: StorageModeShared, no command buffer in flight
            // between test calls (each forward_*_with_hidden call
            // `wait_until_completed`s before returning).
            unsafe {
                let ptr = buf.contents() as *const u8;
                std::slice::from_raw_parts(ptr, buf.length() as usize).to_vec()
            }
        })
        .collect()
}

/// Byte-for-byte snapshot of every live GDN recurrent-state buffer.
fn snapshot_gdn_bytes(state: &MetalQwen35State) -> Vec<Vec<u8>> {
    state
        .session
        .gdn_gpu_state
        .layers()
        .map(super::gdn_state::MetalGdnLayerState::s_matrix)
        .chain(
            state
                .session
                .gdn_gpu_state
                .layers()
                .map(super::gdn_state::MetalGdnLayerState::conv_buffer),
        )
        .map(|buf| {
            // SAFETY: StorageModeShared, and the verifier preflight runs
            // before creating a command buffer, so no GPU write is in flight.
            unsafe {
                let ptr = buf.contents() as *const u8;
                std::slice::from_raw_parts(ptr, buf.length() as usize).to_vec()
            }
        })
        .collect()
}

/// Helper used by both Q4-GEMM differential tests: dispatch a single GEMM kernel
/// and read back Y[M, N] from the returned buffer.  The caller provides the
/// command queue and either the naive `gemm_q4` or tiled `gemm_q4_tiled` pipeline.
/// Buffer bindings match `dispatch_gemm_q4`: buf0=QW, buf1=X, buf2=Y, bytes 3-5=m/n/k.
fn run_gemm_q4_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    qw_buf: &Buffer,
    x_buf: &Buffer,
    y_buf: &Buffer,
    m: u32,
    n: u32,
    k: u32,
    tg_x: u64,
    tg_y: u64,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(qw_buf), 0);
    enc.set_buffer(1, Some(x_buf), 0);
    enc.set_buffer(2, Some(y_buf), 0);
    enc.set_bytes(3, 4, &m as *const u32 as *const _);
    enc.set_bytes(4, 4, &n as *const u32 as *const _);
    enc.set_bytes(5, 4, &k as *const u32 as *const _);
    enc.dispatch_thread_groups(MTLSize::new(tg_x, tg_y, 1), MTLSize::new(32, 4, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

/// Build a Q4 weight buffer and f32 reference for shape [N, K] (output × input).
/// Uses a deterministic LCG so every run produces the same synthetic weights.
/// Calls `quantize_row_q4_0` on each output row — matching the kernel index formula
/// `QW[col * (K/32 * 20) + ib * 20]`.
fn make_q4_weight_ref(
    device: &Device,
    seed: u64,
    n: usize, // output dim
    k: usize, // input dim — must be divisible by 32
) -> (Buffer, Vec<f32>) {
    assert_eq!(k % 32, 0, "K must be divisible by 32 for Q4_0 GEMM");
    let mut rng = seed;
    let mut next = || -> f32 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map u64 to [-1, 1] with moderate range — representative of projection weights.
        ((rng >> 11) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    // Generate row-major [N, K] weight matrix.
    let weights_f32: Vec<f32> = (0..n * k).map(|_| next()).collect();
    // Quantize each output row independently — matches kernel's per-row block layout.
    use crate::weights::q4_weights::dequantize_row_q4_0;
    let mut packed_bytes = Vec::with_capacity(n * (k / 32) * 20);
    let mut deq_weights = Vec::with_capacity(n * k);
    for row in weights_f32.chunks_exact(k) {
        let row_packed = quantize_row_q4_0(row).unwrap();
        deq_weights.extend_from_slice(&dequantize_row_q4_0(&row_packed, k));
        packed_bytes.extend_from_slice(&row_packed);
    }
    let buf = device.new_buffer_with_data(
        packed_bytes.as_ptr() as *const _,
        packed_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    buf.set_label("test_qw_q4");
    // Return: GPU buffer + dequantized f32 weights for CPU reference matmul.
    (buf, deq_weights)
}

/// CPU f32 reference: Y[m, n] = sum_k X[m, k] * W[n, k]  (W is transposed).
fn cpu_matmul_ref(x: &[f32], w_deq: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += x[mi * k + ki] * w_deq[ni * k + ki];
            }
            y[mi * n + ni] = acc;
        }
    }
    y
}

// NaN-honest max|a-b|: a `.fold(0.0, f32::max)` silently drops a NaN/Inf
// operand (IEEE maxNum keeps the non-NaN side), letting a catastrophically
// wrong output read as 0.0 and slip past a `<= tol` gate. Surface it instead.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut max = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if !d.is_finite() {
            return d;
        }
        if d > max {
            max = d;
        }
    }
    max
}

fn make_valid_layer(hidden: usize, rank: usize) -> LoraLayerData {
    // o_proj: d_in = full_q_dim() = hidden, d_out = hidden (equal in tiny fixture).
    // Using o_proj keeps A/B sizes as rank*hidden and hidden*rank while satisfying
    // shape validation.
    LoraLayerData {
        layer_idx: 0,
        module: "o_proj".into(),
        a: vec![0.0f32; rank * hidden],
        b: vec![0.0f32; hidden * rank],
        rank,
        d_in: hidden,
        d_out: hidden,
    }
}

/// Minimal 4-layer hybrid fixture: [GDN, GDN, GDN, GQA] mirroring the
/// Qwen3.5 pattern at small dimension. Sufficient to exercise the
/// self-speculative draft path (GDN-only forward) and verify path (full).
fn tiny_hybrid_fixture() -> (Qwen35Config, ModelWeights) {
    use crate::attention::gdn::GatedDeltaNetWeights;
    // Metal FlashAttention decode requires head_dim=256 (see
    // `validate_flash_decode_shape`). Hidden size = num_attention_heads * head_dim
    // so the Q/K/V projections fit the per-head layout used by the decode kernel.
    let hidden = 512usize;
    let intermediate = 64usize;
    let vocab = 32usize;
    let num_kh = 1usize;
    let kd = 16usize;
    // vd * num_kh = GDN out_dim = the contraction dim K of the out_proj Q8
    // GEMM, which the GEMV/GEMM kernel requires to be a multiple of 32 (real
    // configs use vd=128). 16 made the GDN-only GPU tests crash on real Apple
    // hardware (and silently skip in paravirtual CI where no Metal device exists).
    let vd = 32usize;
    let qkv_dim = num_kh * kd * 2 + num_kh * vd; // Q+K+V
    let out_dim = num_kh * vd;
    let ks = 4usize;
    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 4,
        vocab_size: vocab,
        intermediate_size: intermediate,
        rms_norm_eps: 1e-6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 256,
        rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        rope_parameters: None,
        linear_num_key_heads: num_kh,
        linear_num_value_heads: Some(num_kh),
        linear_key_head_dim: kd,
        linear_value_head_dim: vd,
        linear_conv_kernel_dim: ks,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        full_attention_interval: 4,
        layer_types: vec![
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ],
        layer_mask: vec![true; 4],
        eos_token_id: (vocab - 1) as u32,
        max_position_embeddings: 128,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    let make_common = || CommonLayerWeights {
        input_layernorm: vec![1.0; hidden],
        post_attention_layernorm: vec![1.0; hidden],
        ffn: FeedForwardWeights::Dense(DenseFfnWeights {
            gate_proj: vec![0.0; intermediate * hidden],
            up_proj: vec![0.0; intermediate * hidden],
            down_proj: vec![0.0; hidden * intermediate],
        }),
    };

    let make_gdn = || {
        AttentionWeights::Linear(GatedDeltaNetWeights {
            in_proj_qkv: vec![0.0; qkv_dim * hidden],
            in_proj_qkv_rows: qkv_dim,
            in_proj_qkv_cols: hidden,
            in_proj_z: vec![0.0; out_dim * hidden],
            in_proj_z_rows: out_dim,
            in_proj_z_cols: hidden,
            in_proj_b: vec![0.0; num_kh * hidden],
            in_proj_b_rows: num_kh,
            in_proj_b_cols: hidden,
            in_proj_a: vec![0.0; num_kh * hidden],
            in_proj_a_rows: num_kh,
            in_proj_a_cols: hidden,
            a_log: vec![-1.0; num_kh],
            dt_bias: vec![0.0; num_kh],
            conv1d_weight: vec![0.0; qkv_dim * ks],
            conv_dim: qkv_dim,
            kernel_size: ks,
            norm_weight: vec![1.0; out_dim],
            out_proj: vec![0.0; hidden * out_dim],
            out_proj_rows: hidden,
            out_proj_cols: out_dim,
        })
    };

    let make_full = || {
        AttentionWeights::Full(FullAttentionLayerWeights {
            q_proj: vec![0.0; 2 * cfg.full_q_dim() * hidden],
            k_proj: vec![0.0; cfg.full_kv_dim() * hidden],
            v_proj: vec![0.0; cfg.full_kv_dim() * hidden],
            o_proj: vec![0.0; hidden * cfg.full_q_dim()],
            q_norm: vec![1.0; cfg.head_dim],
            k_norm: vec![1.0; cfg.head_dim],
        })
    };

    let mut embed = vec![0.0f32; vocab * hidden];
    for tok in 0..vocab {
        embed[tok * hidden] = if tok % 2 == 0 { 1.0 } else { -1.0 };
    }

    let weights = ModelWeights {
        embed_tokens: embed,
        lm_head: None,
        final_norm: vec![1.0; hidden],
        layers: vec![
            (make_gdn(), make_common()),
            (make_gdn(), make_common()),
            (make_gdn(), make_common()),
            (make_full(), make_common()),
        ],
    };
    (cfg, weights)
}

/// `tiny_hybrid_fixture` with token embeddings that spread the letter logits: every row
/// has two non-zero entries, so the logit at a letter depends on the last prompt token and
/// differs from letter to letter. The full-attention layer also carries one token-mixing
/// path: q_proj and k_proj stay zero, so attention is uniform over the positions seen so
/// far, value channel 0 reads hidden[0], and o_proj routes that channel into hidden[1].
/// The last position's letter logits therefore depend on every earlier prompt token.
pub(super) fn option_scoring_fixture() -> (Qwen35Config, ModelWeights) {
    let (cfg, mut weights) = tiny_hybrid_fixture();
    let hidden = cfg.hidden_size;
    let q_dim = cfg.full_q_dim();
    for token in 0..cfg.vocab_size {
        let row = &mut weights.embed_tokens[token * hidden..(token + 1) * hidden];
        row[0] = (token % 5) as f32 - 2.0;
        row[1] = (token % 3) as f32 - 1.0;
    }
    let Some((AttentionWeights::Full(full), _)) = weights.layers.last_mut() else {
        panic!("the last fixture layer is full attention");
    };
    full.v_proj[0] = 1.0;
    // o_proj is [hidden, q_dim] row-major: row 1, column 0 writes head 0 channel 0 into hidden[1].
    full.o_proj[q_dim] = 1.0;
    (cfg, weights)
}

fn f32_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Single full-attention MoE layer — batched prefill does not support MoE
/// (`assert_batched_prefill_dense_only`), so this is the minimal fixture for
/// exercising the up-front rejection path before any state mutation.
fn tiny_moe_prefill_fixture() -> (Qwen35Config, ModelWeights) {
    use crate::model::qwen35::{MoeLayerWeights, MoeRouter, RoutedExperts, SharedExpert};

    let hidden = 512usize;
    let vocab = 32usize;
    let num_experts = 2usize;
    let top_k = 1usize;
    let inter = 16usize;
    let s_inter = 16usize;
    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 1,
        vocab_size: vocab,
        intermediate_size: 64,
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
        num_experts: Some(num_experts),
        num_experts_per_tok: Some(top_k),
        moe_intermediate_size: Some(inter),
        shared_expert_intermediate_size: Some(s_inter),
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        full_attention_interval: 1,
        layer_types: vec![LayerType::FullAttention],
        layer_mask: vec![true],
        eos_token_id: (vocab - 1) as u32,
        max_position_embeddings: 64,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    let full = AttentionWeights::Full(FullAttentionLayerWeights {
        q_proj: vec![0.0; 2 * cfg.full_q_dim() * hidden],
        k_proj: vec![0.0; cfg.full_kv_dim() * hidden],
        v_proj: vec![0.0; cfg.full_kv_dim() * hidden],
        o_proj: vec![0.0; hidden * cfg.full_q_dim()],
        q_norm: vec![1.0; cfg.head_dim],
        k_norm: vec![1.0; cfg.head_dim],
    });

    let router = MoeRouter::new(vec![0.0; num_experts * hidden], num_experts, top_k, hidden)
        .expect("moe router fixture shape");
    let experts = RoutedExperts::new(
        vec![0.0; num_experts * 2 * inter * hidden],
        vec![0.0; num_experts * hidden * inter],
        num_experts,
        hidden,
        inter,
    )
    .expect("moe routed experts fixture shape");
    let shared_expert = SharedExpert::new(
        vec![0.0; s_inter * hidden],
        vec![0.0; s_inter * hidden],
        vec![0.0; hidden * s_inter],
        vec![0.0; hidden],
        hidden,
        s_inter,
    )
    .expect("moe shared expert fixture shape");

    let common = CommonLayerWeights {
        input_layernorm: vec![1.0; hidden],
        post_attention_layernorm: vec![1.0; hidden],
        ffn: FeedForwardWeights::Moe(MoeLayerWeights {
            router,
            experts,
            shared_expert,
        }),
    };

    // Non-zero so a wrongly-ordered guard (embedding copy running before
    // the MoE rejection) is observable as a mutated `activations.hidden`.
    let mut embed = vec![0.0f32; vocab * hidden];
    for tok in 0..vocab {
        embed[tok * hidden] = if tok % 2 == 0 { 1.0 } else { -1.0 };
    }

    let weights = ModelWeights {
        embed_tokens: embed,
        lm_head: None,
        final_norm: vec![1.0; hidden],
        layers: vec![(full, common)],
    };
    (cfg, weights)
}

#[cfg(feature = "bench-internals")]
fn tiny_chunked_prefill_fixture() -> (Qwen35Config, ModelWeights) {
    use crate::attention::gdn::GatedDeltaNetWeights;

    let hidden = 1024usize;
    let intermediate = 32usize;
    let vocab = 32usize;
    let num_kh = 16usize;
    let kd = 128usize;
    let vd = 128usize;
    let qkv_dim = num_kh * (2 * kd + vd);
    let out_dim = num_kh * vd;
    let ks = 4usize;
    let cfg = Qwen35Config {
        hidden_size: hidden,
        num_hidden_layers: 2,
        vocab_size: vocab,
        intermediate_size: intermediate,
        rms_norm_eps: 1e-6,
        num_attention_heads: 4,
        num_key_value_heads: 1,
        head_dim: 256,
        rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        rope_parameters: None,
        linear_num_key_heads: num_kh,
        linear_num_value_heads: Some(num_kh),
        linear_key_head_dim: kd,
        linear_value_head_dim: vd,
        linear_conv_kernel_dim: ks,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        shared_expert_intermediate_size: None,
        output_router_logits: false,
        router_aux_loss_coef: None,
        tie_word_embeddings: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: false,
        full_attention_interval: 2,
        layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
        layer_mask: vec![true; 2],
        eos_token_id: (vocab - 1) as u32,
        max_position_embeddings: 64,
        quarot_rotation_seed: None,
        vision_config: None,
        image_token_id: None,
        video_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
    };

    let patterned = |len: usize, seed: usize, scale: f32| -> Vec<f32> {
        (0..len)
            .map(|i| {
                let bucket = (i.wrapping_mul(37).wrapping_add(seed)) % 31;
                (bucket as f32 - 15.0) * scale
            })
            .collect()
    };
    let make_common = || CommonLayerWeights {
        input_layernorm: patterned(hidden, 1, 0.002),
        post_attention_layernorm: patterned(hidden, 2, 0.002),
        ffn: FeedForwardWeights::Dense(DenseFfnWeights {
            gate_proj: patterned(intermediate * hidden, 3, 0.001),
            up_proj: patterned(intermediate * hidden, 4, 0.001),
            down_proj: patterned(hidden * intermediate, 5, 0.001),
        }),
    };
    let gdn = AttentionWeights::Linear(GatedDeltaNetWeights {
        in_proj_qkv: patterned(qkv_dim * hidden, 6, 0.0005),
        in_proj_qkv_rows: qkv_dim,
        in_proj_qkv_cols: hidden,
        in_proj_z: patterned(out_dim * hidden, 7, 0.0005),
        in_proj_z_rows: out_dim,
        in_proj_z_cols: hidden,
        in_proj_b: patterned(num_kh * hidden, 8, 0.001),
        in_proj_b_rows: num_kh,
        in_proj_b_cols: hidden,
        in_proj_a: patterned(num_kh * hidden, 9, 0.001),
        in_proj_a_rows: num_kh,
        in_proj_a_cols: hidden,
        a_log: patterned(num_kh, 10, 0.01),
        dt_bias: patterned(num_kh, 11, 0.01),
        conv1d_weight: patterned(qkv_dim * ks, 12, 0.002),
        conv_dim: qkv_dim,
        kernel_size: ks,
        norm_weight: patterned(out_dim, 13, 0.002),
        out_proj: patterned(hidden * out_dim, 14, 0.0005),
        out_proj_rows: hidden,
        out_proj_cols: out_dim,
    });
    let full = AttentionWeights::Full(FullAttentionLayerWeights {
        q_proj: patterned(2 * cfg.full_q_dim() * hidden, 15, 0.0005),
        k_proj: patterned(cfg.full_kv_dim() * hidden, 16, 0.0005),
        v_proj: patterned(cfg.full_kv_dim() * hidden, 17, 0.0005),
        o_proj: patterned(hidden * cfg.full_q_dim(), 18, 0.0005),
        q_norm: patterned(cfg.head_dim, 19, 0.002),
        k_norm: patterned(cfg.head_dim, 20, 0.002),
    });
    let weights = ModelWeights {
        embed_tokens: patterned(vocab * hidden, 21, 0.002),
        lm_head: None,
        final_norm: patterned(hidden, 22, 0.002),
        layers: vec![(gdn, make_common()), (full, make_common())],
    };
    (cfg, weights)
}

#[cfg(feature = "bench-internals")]
#[derive(Debug, PartialEq)]
struct PrefillSchedulerArtifacts {
    position: usize,
    kv_rows: Vec<(Vec<u8>, Vec<u8>)>,
    gdn_snapshot: crate::attention::gdn::GdnSnapshot,
    next_token: usize,
}

#[cfg(feature = "bench-internals")]
fn prefill_scheduler_artifacts(
    state: &mut MetalQwen35State,
    token_ids: &[u32],
    next_input: u32,
) -> PrefillSchedulerArtifacts {
    use crate::speculative::MtpTargetVerifier as _;

    let kv_row_bytes = if state.use_kv_f16 { 2 } else { 4 };
    let used_bytes = token_ids.len() * state.engine.config.full_kv_dim() * kv_row_bytes;
    let kv_rows = state
        .session
        .kv_cache
        .k_bufs
        .iter()
        .zip(&state.session.kv_cache.v_bufs)
        .map(|(k, v)| {
            // SAFETY: Both buffers are StorageModeShared, the scheduler waited for its
            // command buffers, and used_bytes covers only initialized prefix rows.
            unsafe {
                let k_bytes =
                    std::slice::from_raw_parts(k.contents() as *const u8, used_bytes).to_vec();
                let v_bytes =
                    std::slice::from_raw_parts(v.contents() as *const u8, used_bytes).to_vec();
                (k_bytes, v_bytes)
            }
        })
        .collect();
    let position = state.session.position();
    let gdn_snapshot = state.snapshot_gdn_states();
    let next_logits = state.forward_step(next_input, token_ids.len());
    PrefillSchedulerArtifacts {
        position,
        kv_rows,
        gdn_snapshot,
        next_token: argmax_f32(&next_logits),
    }
}

/// CPU oracle: exact top-K token ids by full-logit value, descending.
fn cpu_topk_token_ids(logits: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx
}

fn load_real_qwen35_0_8b_or_skip() -> Option<crate::model::qwen35::Qwen35Model> {
    let model_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
        .join(".lattice/models/qwen3.5-0.8b");
    // Mirror `Qwen35Model::from_safetensors`: weights are present in
    // EITHER the single-file layout (`model.safetensors`) or the
    // sharded layout (`model.safetensors.index.json`) — an HF
    // `snapshot_download` of this model ships only the sharded form.
    let weights_present = model_dir.join("model.safetensors").exists()
        || model_dir.join("model.safetensors.index.json").exists();
    if !weights_present
        || !model_dir.join("config.json").exists()
        || !model_dir.join("tokenizer.json").exists()
    {
        eprintln!(
            "skipping real-checkpoint test: model files missing at {}",
            model_dir.display()
        );
        return None;
    }
    Some(
        crate::model::qwen35::Qwen35Model::from_safetensors(&model_dir).expect("load qwen3.5-0.8b"),
    )
}

/// Qwen3.5 vision (ADR-069 Metal S5, MP2) capability gate: a Metal
/// device and a real checkpoint are both required to exercise the
/// injection path. Fails closed (`LATTICE_METAL_TEST_ENFORCE=1`)
/// instead of returning `None` when either is missing, matching the
/// `forward::metal::metal_test_device` / `f16_kv_metal_path_executes_and_reports_capability`
/// convention — otherwise a runner without a Metal device or
/// checkpoint would report these tests as passing without ever
/// exercising the injection path.
fn require_metal_and_real_checkpoint_or_skip(
    context: &str,
) -> Option<crate::model::qwen35::Qwen35Model> {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    if Device::system_default().is_none() {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present ({context})"
        );
        eprintln!("[METAL_TEST_SKIP] context={context} reason=no_metal_device");
        return None;
    }
    match load_real_qwen35_0_8b_or_skip() {
        Some(model) => Some(model),
        None => {
            assert!(
                !enforce,
                "LATTICE_METAL_TEST_ENFORCE=1 but qwen3.5-0.8b checkpoint missing ({context})"
            );
            eprintln!("[METAL_TEST_SKIP] context={context} reason=no_checkpoint");
            None
        }
    }
}

/// Run `f` with `LATTICE_SELF_SPEC=1` in the process environment, then remove it.
///
/// Rust 2024 marks `set_var`/`remove_var` as `unsafe` because they can race with
/// reads on other threads. We serialize self-spec tests through `ENV_LOCK` so this
/// runs single-threaded with respect to the env var; the env var is only consulted
/// at session construction (line 3778 / 10374), so no concurrent reads occur during
/// the test body itself.
fn with_self_spec_env<R>(f: impl FnOnce() -> R) -> R {
    use std::sync::Mutex;
    static ENV_LOCK: Mutex<()> = Mutex::new(());
    let _guard = ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // SAFETY: only this serialized closure mutates LATTICE_SELF_SPEC. The lock
    // forbids concurrent test threads from reading or writing it.
    unsafe {
        std::env::set_var("LATTICE_SELF_SPEC", "1");
    }
    let r = f();
    // SAFETY: same justification as above.
    unsafe {
        std::env::remove_var("LATTICE_SELF_SPEC");
    }
    r
}

fn hidden_parity_model_dir() -> Option<std::path::PathBuf> {
    let expand = |s: &str| match s.strip_prefix("~/") {
        Some(rest) => match std::env::var("HOME") {
            Ok(home) => format!("{home}/{rest}"),
            Err(_) => s.to_string(),
        },
        None => s.to_string(),
    };
    if let Ok(d) = std::env::var("LATTICE_MODEL_DIR") {
        let p = std::path::PathBuf::from(expand(&d));
        return p.join("config.json").is_file().then_some(p);
    }
    let home = std::env::var("HOME").ok()?;
    let p = std::path::PathBuf::from(home).join(".lattice/models/qwen3.5-0.8b");
    p.join("config.json").is_file().then_some(p)
}

fn minimal_bpe_tokenizer() -> crate::tokenizer::bpe::BpeTokenizer {
    use std::collections::HashMap;
    let mut vocab: HashMap<String, u32> = HashMap::new();
    // A handful of canonical specials plus a few bytes — generate_greedy_self_spec
    // only uses the tokenizer for final decode, so vocab content doesn't need to
    // cover the model's full id range.
    vocab.insert("<|endoftext|>".to_string(), 0);
    vocab.insert("<|im_start|>".to_string(), 1);
    vocab.insert("<|im_end|>".to_string(), 2);
    for i in 0..32u32 {
        vocab.insert(format!("t{i}"), 3 + i);
    }
    crate::tokenizer::bpe::BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
        .expect("minimal tokenizer build")
}

/// Single-character vocabulary covering every id in `tiny_hybrid_fixture`'s
/// 32-token vocab, so any sampled id round-trips through both the BPE
/// encoder (for prompt text) and decoder (for streamed output) without
/// landing on an out-of-vocab fallback. Unlike `minimal_bpe_tokenizer`,
/// every entry is exactly one character: with no merge rules, `bpe_merge`
/// never combines characters, so only single-character vocab keys are
/// reachable on the encode side.
fn single_char_vocab_tokenizer() -> crate::tokenizer::bpe::BpeTokenizer {
    use std::collections::HashMap;
    let mut vocab: HashMap<String, u32> = HashMap::new();
    for i in 0u32..32 {
        let byte = if i < 26 {
            b'a' + i as u8
        } else {
            b'A' + (i - 26) as u8
        };
        vocab.insert((byte as char).to_string(), i);
    }
    crate::tokenizer::bpe::BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
        .expect("single-char vocab tokenizer build")
}

/// Shared setup for the #1448 rejection tests: the `LATTICE_METAL_TEST_ENFORCE`
/// device contract, the machine-wide GPU lock, the MoE fixture, and state
/// construction, in one place so an entry-point addition or a setup change is
/// one edit instead of one per test. Runs `test` while the lock is held; on a
/// device-less machine it asserts the enforce contract and runs nothing.
fn with_moe_prefill_state(
    max_cache_len: usize,
    test: impl FnOnce(&Qwen35Config, &mut MetalQwen35State),
) {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = metal::Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _guard = gpu_test_lock();
    let (cfg, weights) = tiny_moe_prefill_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, max_cache_len)
        .expect("moe prefill fixture must construct");
    test(&cfg, &mut state);
}

fn assert_context_budget_error<T: std::fmt::Debug>(
    result: &Result<T, crate::error::InferenceError>,
    prompt_len: usize,
    effective_new: usize,
    max_context: usize,
) {
    let expected = format!(
        "prompt ({prompt_len} tokens) plus effective decode cap ({effective_new} tokens; \
                 max_new_tokens={effective_new}) exceeds model context window ({max_context})"
    );
    assert!(
        matches!(result, Err(crate::error::InferenceError::Inference(message)) if message == &expected),
        "expected shared context-budget error {expected:?}, got {result:?}"
    );
}

/// Build a token sequence of 650-700 tokens from varied prose paragraphs.
/// Uses the provided tokenizer; grows text by appending paragraphs until
/// the encoded length is ≥ 650, then truncates to ≤ 680.
fn long_real_text_tokens(tokenizer: &BpeTokenizer) -> Vec<u32> {
    let paragraphs = [
        "During a late engineering review, the team walked through the inference trace \
                 one layer at a time. They checked where state changed, which buffers were reused, \
                 and how a long request should preserve every earlier token.",
        "The prompt continued with ordinary prose about debugging, benchmarks, release notes, \
                 and careful handoffs. It used full sentences, punctuation, and varied vocabulary \
                 so tokenization looked like real input rather than a repeated numeric pattern.",
        "A second reviewer asked for evidence at the boundary between chunks. The answer \
                 described rotary positions, cache rows, recurrent memory, and causal attention \
                 in concrete terms before any optimization was accepted.",
        "Verification across chunk boundaries requires that each token's absolute position \
                 index matches the RoPE table row, the KV cache write offset, and the attention \
                 causal mask — all three must use the same absolute coordinate, not a chunk-local one.",
    ];
    let mut text = String::new();
    let mut ids = Vec::new();
    for i in 0..256usize {
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str(paragraphs[i % paragraphs.len()]);
        let input = tokenizer.tokenize(&text);
        ids = input.input_ids[..input.real_length].to_vec();
        if ids.len() >= 650 {
            break;
        }
    }
    if ids.len() > 680 {
        ids.truncate(680);
    }
    ids
}

fn argmax_f32(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    best
}

/// Mirror `Qwen35Model::from_safetensors`: weights are present in
/// EITHER the single-file layout (`model.safetensors`) or the
/// sharded layout (`model.safetensors.index.json`) — an HF
/// `snapshot_download` of this model ships only the sharded form.
fn qwen35_checkpoint_weights_present(model_dir: &std::path::Path) -> bool {
    model_dir.join("model.safetensors").exists()
        || model_dir.join("model.safetensors.index.json").exists()
}

fn cross_turn_test_gen_cfg(seed: u64, max_new_tokens: usize) -> GenerateConfig {
    GenerateConfig {
        min_p: 0.0,
        max_new_tokens,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(seed),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    }
}

/// Byte-per-token vocab matching `single_char_vocab_tokenizer`'s id
/// order (`'a'..='z'` then `'A'..='F'`), so a `GrammarEngine` built
/// over it can accept/reject real single-character tokens instead of
/// the all-empty fixture the prefill-only tests above use.
fn single_char_vocab_bytes() -> Vec<Vec<u8>> {
    (0u32..32)
        .map(|i| {
            let byte = if i < 26 {
                b'a' + i as u8
            } else {
                b'A' + (i - 26) as u8
            };
            vec![byte]
        })
        .collect()
}

/// Greedy grammar-constrained `GenerateConfig` over the single-char
/// vocab: `gbnf` and the token cap vary per test, everything else is
/// the shared deterministic-decode baseline the tests above use.
fn single_char_grammar_cfg(gbnf: &str, max_new_tokens: usize) -> crate::generation::GenerateConfig {
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    let spec = GrammarSpec::Gbnf(gbnf.to_string());
    let engine = Arc::new(
        GrammarEngine::new(&spec, single_char_vocab_bytes())
            .expect("grammar engine builds over single-char vocab"),
    );
    crate::generation::GenerateConfig {
        min_p: 0.0,
        max_new_tokens,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: Some(engine),
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    }
}
