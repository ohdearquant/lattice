use super::{
    LM_HEAD_LOCAL_KS, LM_HEAD_ROWS_PER_TG, MSL_Q3_TILED_SOURCE, MSL_Q4_TILED_SOURCE,
    MSL_Q8_TILED_SOURCE, MSL_SOURCE, MetalQwen35Pipelines, prefill_maxgrp_suffix,
};
use lattice_arch::qwen35_config::Qwen35Config;
use metal::{
    CompileOptions, ComputePipelineState, Device, FunctionConstantValues, Library, MTLDataType,
    MTLGPUFamily, MTLLanguageVersion,
};

/// Compile one Stage-1 lm_head block-top-k pipeline variant with the
/// HIDDEN / ROWS_PER_TG / LOCAL_K function constants baked in.
fn make_lm_head_block_pipeline(
    device: &Device,
    library: &Library,
    name: &str,
    hidden: u32,
    rows_per_tg: u32,
    local_k: u32,
) -> Result<ComputePipelineState, String> {
    let constants = FunctionConstantValues::new();
    constants.set_constant_value_at_index(
        &hidden as *const u32 as *const std::ffi::c_void,
        MTLDataType::UInt,
        0,
    );
    constants.set_constant_value_at_index(
        &rows_per_tg as *const u32 as *const std::ffi::c_void,
        MTLDataType::UInt,
        1,
    );
    constants.set_constant_value_at_index(
        &local_k as *const u32 as *const std::ffi::c_void,
        MTLDataType::UInt,
        2,
    );
    let func = library.get_function(name, Some(constants)).map_err(|e| {
        format!(
            "function '{name}' (HIDDEN={hidden}, ROWS_PER_TG={rows_per_tg}, LOCAL_K={local_k}) not found: {e}"
        )
    })?;
    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| format!("pipeline for '{name}' LOCAL_K={local_k} failed: {e}"))
}

/// Compile all five LOCAL_K variants of a Stage-1 lm_head block-top-k kernel.
fn make_lm_head_block_pipelines(
    device: &Device,
    library: &Library,
    name: &str,
    hidden: u32,
) -> Result<[ComputePipelineState; 5], String> {
    let mut out: Vec<ComputePipelineState> = Vec::with_capacity(LM_HEAD_LOCAL_KS.len());
    for &local_k in LM_HEAD_LOCAL_KS.iter() {
        out.push(make_lm_head_block_pipeline(
            device,
            library,
            name,
            hidden,
            LM_HEAD_ROWS_PER_TG,
            local_k,
        )?);
    }
    out.try_into().map_err(|_| {
        format!(
            "expected {} pipeline variants for '{name}'",
            LM_HEAD_LOCAL_KS.len()
        )
    })
}

pub(super) fn build_pipelines(
    device: &Device,
    cfg: &Qwen35Config,
) -> Result<MetalQwen35Pipelines, String> {
    let prefill_grp_suffix =
        prefill_maxgrp_suffix(cfg.num_attention_heads / cfg.num_key_value_heads)?;

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
        // simdgroup_float8x8 matrix ops are available since Apple7 (M1).
        // The compile/pipeline `.ok()?` chain below falls back to the
        // naive gemm_q4 on any device where the V3.0 source fails.
        if !device.supports_family(MTLGPUFamily::Apple7) {
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

    let make_optional_gemm_q3_tiled = || -> Option<ComputePipelineState> {
        // Same Apple7 simdgroup-matrix gate as Q4/Q8; falls back to the
        // gemv_q3_decode-only (M=1) path on any device/compile failure.
        if !device.supports_family(MTLGPUFamily::Apple7) {
            return None;
        }
        let tiled_opts = CompileOptions::new();
        tiled_opts.set_language_version(MTLLanguageVersion::V3_0);
        let lib = device
            .new_library_with_source(MSL_Q3_TILED_SOURCE, &tiled_opts)
            .ok()?;
        let func = lib.get_function("gemm_q3_tiled", None).ok()?;
        device.new_compute_pipeline_state_with_function(&func).ok()
    };

    let make_optional_gemm_q8_tiled = || -> Option<ComputePipelineState> {
        // Q8_0 simdgroup-matrix tiled GEMM: same Apple7 gate as Q4.
        if !device.supports_family(MTLGPUFamily::Apple7) {
            return None;
        }
        let tiled_opts = CompileOptions::new();
        tiled_opts.set_language_version(MTLLanguageVersion::V3_0);
        let lib = device
            .new_library_with_source(MSL_Q8_TILED_SOURCE, &tiled_opts)
            .ok()?;
        let func = lib.get_function("gemm_q8_tiled", None).ok()?;
        device.new_compute_pipeline_state_with_function(&func).ok()
    };

    Ok(MetalQwen35Pipelines {
        gemv_decode: make_pipeline("gemv_decode_m1")?,
        gemv_decode_wide: make_pipeline("gemv_decode_wide_f16")?,
        gemv_q8: make_pipeline("gemv_q8_decode")?,
        gemv_q4: make_pipeline("gemv_q4_decode")?,
        gemm_q4: make_pipeline("gemm_q4")?,
        gemm_q4_tiled: make_optional_gemm_q4_tiled(),
        gemv_q3: make_pipeline("gemv_q3_decode")?,
        gemm_q3_tiled: make_optional_gemm_q3_tiled(),
        rms_norm: make_pipeline("rms_norm_qwen35")?,
        partial_rope: make_pipeline("partial_rope_interleaved")?,
        per_head_rms_norm: make_pipeline("per_head_rms_norm")?,
        decode_attention: make_pipeline("decode_attention")?,
        sigmoid_gate: make_pipeline("sigmoid_gate")?,
        scatter_q_gate: make_pipeline("scatter_q_gate")?,
        silu_mul: make_pipeline("silu_mul")?,
        silu_mul_fused: make_pipeline("silu_mul_fused")?,
        copy: make_pipeline("copy_buf")?,
        copy_offset: make_pipeline("copy_buf_offset")?,
        conv1d_silu: make_pipeline("conv1d_depthwise_silu")?,
        gdn_recurrence: make_pipeline("gdn_recurrence_fused")?,
        gdn_chunk_materialize_c32: make_pipeline("gdn_chunk_materialize_c32")?,
        gdn_chunk_solve_c32: make_pipeline("gdn_chunk_solve_c32")?,
        gdn_chunk_residual_output_c32: make_pipeline("gdn_chunk_residual_output_c32")?,
        gdn_chunk_state_update_c32: make_pipeline("gdn_chunk_state_update_c32")?,
        gdn_chunk_norm_silu_c32: make_pipeline("gdn_chunk_norm_silu_c32")?,
        gdn_chunk_conv_buf_update_c32: make_pipeline("gdn_chunk_conv_buf_update_c32")?,
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
        copy_and_rms_norm_batch: make_pipeline("copy_and_rms_norm_batch")?,
        fused_residual_add_norm_batch: make_pipeline("fused_residual_add_norm_batch")?,
        gemm_q8: make_pipeline("gemm_q8")?,
        gemm_q8_tiled: make_optional_gemm_q8_tiled(),
        topk_merge_pass: make_pipeline("logits_topk_merge_pass")?,
        argmax_first: make_pipeline("logits_argmax_first")?,
        argmax_merge: make_pipeline("logits_argmax_merge")?,
        topk_select50_first: make_pipeline("logits_topk_select50_first")?,
        topk_select50_merge: make_pipeline("logits_topk_select50_merge")?,
        decode_attn_partial: make_pipeline("decode_attention_flash_partial")?,
        decode_attn_reduce: make_pipeline("decode_attention_flash_reduce")?,
        lora_gemv_a: make_pipeline("lora_gemv_a")?,
        lora_gemv_b_accum: make_pipeline("lora_gemv_b_accum")?,
        lora_gemm_a_rows: make_pipeline("lora_gemm_a_rows")?,
        lora_gemm_b_accum_rows: make_pipeline("lora_gemm_b_accum_rows")?,
        // ADR-053: MoE Metal dispatch kernels
        moe_expert_gemv: make_pipeline("moe_expert_gemv")?,
        moe_scale_add: make_pipeline("moe_scale_add")?,
        moe_shared_gate_add: make_pipeline("moe_shared_gate_add")?,
        moe_zero_buf: make_pipeline("zero_buf")?,
        moe_router_topk: make_pipeline("moe_router_topk")?,
        moe_shared_gate_scalar: make_pipeline("moe_shared_gate_scalar")?,
        moe_expert_gemv_routed: make_pipeline("moe_expert_gemv_routed")?,
        moe_scale_add_routed: make_pipeline("moe_scale_add_routed")?,
        scatter_q_gate_batch: make_pipeline("scatter_q_gate_batch")?,
        per_head_rms_norm_batch: make_pipeline("per_head_rms_norm_batch")?,
        partial_rope_batch: make_pipeline("partial_rope_batch")?,
        copy_kv_cache_batch: make_pipeline("copy_kv_cache_batch")?,
        prefill_attention_batched: make_pipeline(&format!(
            "prefill_attention_batched_causal_{prefill_grp_suffix}"
        ))?,
        copy_offset_f16: make_pipeline("copy_buf_offset_f16")?,
        copy_kv_cache_batch_f16: make_pipeline("copy_kv_cache_batch_f16")?,
        decode_attention_f16: make_pipeline("decode_attention_f16")?,
        decode_attn_partial_f16: make_pipeline("decode_attention_flash_partial_f16")?,
        prefill_attention_batched_f16: make_pipeline(&format!(
            "prefill_attention_batched_causal_{prefill_grp_suffix}_f16"
        ))?,
        lm_head_block_topk_f16: make_lm_head_block_pipelines(
            device,
            &library,
            "lm_head_block_topk_f16",
            cfg.hidden_size as u32,
        )?,
        lm_head_block_topk_q4: make_lm_head_block_pipelines(
            device,
            &library,
            "lm_head_block_topk_q4",
            cfg.hidden_size as u32,
        )?,
    })
}
