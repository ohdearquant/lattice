use std::sync::{Arc, atomic::Ordering};

use wgpu::util::DeviceExt;

use super::api::checked_mul;
use super::api::{
    GpuForwardError, GpuRuntimeConfig, Qwen3Config, Qwen3LayerWeights, Qwen3Weights, Result,
};
use super::params::{AllocationStats, PARAM_SLOT_BYTES};
use super::util::bytes_f32;

pub(super) struct GpuLayerWeights {
    pub(super) q_proj_weight: wgpu::Buffer,
    pub(super) k_proj_weight: wgpu::Buffer,
    pub(super) v_proj_weight: wgpu::Buffer,
    pub(super) o_proj_weight: wgpu::Buffer,
    pub(super) q_norm_weight: wgpu::Buffer,
    pub(super) k_norm_weight: wgpu::Buffer,
    pub(super) input_layernorm_weight: wgpu::Buffer,
    pub(super) gate_proj_weight: wgpu::Buffer,
    pub(super) up_proj_weight: wgpu::Buffer,
    pub(super) down_proj_weight: wgpu::Buffer,
    pub(super) post_attention_layernorm_weight: wgpu::Buffer,
}

pub(super) struct GpuWeights {
    pub(super) layers: Vec<GpuLayerWeights>,
    pub(super) norm_weight: wgpu::Buffer,
    pub(super) rope_cos: wgpu::Buffer,
    pub(super) rope_sin: wgpu::Buffer,
    pub(super) embed_tokens_gpu: Option<wgpu::Buffer>,
}

pub(super) struct ActivationBuffers {
    pub(super) hidden: wgpu::Buffer,
    pub(super) residual: wgpu::Buffer,
    pub(super) q: wgpu::Buffer,
    pub(super) k: wgpu::Buffer,
    pub(super) v: wgpu::Buffer,
    pub(super) scores: wgpu::Buffer,
    pub(super) attn_out: wgpu::Buffer,
    pub(super) gate: wgpu::Buffer,
    pub(super) up: wgpu::Buffer,
    pub(super) params: wgpu::Buffer,
    pub(super) readback: wgpu::Buffer,
}

pub(super) fn upload_weights(
    stats: &Arc<AllocationStats>,
    device: &wgpu::Device,
    config: &Qwen3Config,
    runtime: &GpuRuntimeConfig,
    weights: &Qwen3Weights,
    rope_cos: &[f32],
    rope_sin: &[f32],
) -> Result<GpuWeights> {
    let mut gpu_layers = Vec::with_capacity(config.num_hidden_layers);
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        gpu_layers.push(upload_layer_weights(stats, device, layer_idx, layer));
    }

    let norm_weight = tracked_create_storage_buffer_init_f32(
        stats,
        device,
        "final_norm_weight",
        &weights.norm_weight,
    );
    let rope_cos_buf = tracked_create_storage_buffer_init_f32(stats, device, "rope_cos", rope_cos);
    let rope_sin_buf = tracked_create_storage_buffer_init_f32(stats, device, "rope_sin", rope_sin);

    let embed_tokens_gpu = maybe_upload_embeddings(stats, device, runtime, weights)?;

    Ok(GpuWeights {
        layers: gpu_layers,
        norm_weight,
        rope_cos: rope_cos_buf,
        rope_sin: rope_sin_buf,
        embed_tokens_gpu,
    })
}

fn upload_layer_weights(
    stats: &Arc<AllocationStats>,
    device: &wgpu::Device,
    layer_idx: usize,
    layer: &Qwen3LayerWeights,
) -> GpuLayerWeights {
    GpuLayerWeights {
        q_proj_weight: tracked_create_storage_buffer_init_f32(
            stats,
            device,
            &format!("layer_{layer_idx}_q_proj"),
            &layer.q_proj_weight,
        ),
        k_proj_weight: tracked_create_storage_buffer_init_f32(
            stats,
            device,
            &format!("layer_{layer_idx}_k_proj"),
            &layer.k_proj_weight,
        ),
        v_proj_weight: tracked_create_storage_buffer_init_f32(
            stats,
            device,
            &format!("layer_{layer_idx}_v_proj"),
            &layer.v_proj_weight,
        ),
        o_proj_weight: tracked_create_storage_buffer_init_f32(
            stats,
            device,
            &format!("layer_{layer_idx}_o_proj"),
            &layer.o_proj_weight,
        ),
        q_norm_weight: tracked_create_storage_buffer_init_f32(
            stats,
            device,
            &format!("layer_{layer_idx}_q_norm"),
            &layer.q_norm_weight,
        ),
        k_norm_weight: tracked_create_storage_buffer_init_f32(
            stats,
            device,
            &format!("layer_{layer_idx}_k_norm"),
            &layer.k_norm_weight,
        ),
        input_layernorm_weight: tracked_create_storage_buffer_init_f32(
            stats,
            device,
            &format!("layer_{layer_idx}_input_ln"),
            &layer.input_layernorm_weight,
        ),
        gate_proj_weight: tracked_create_storage_buffer_init_f32(
            stats,
            device,
            &format!("layer_{layer_idx}_gate_proj"),
            &layer.gate_proj_weight,
        ),
        up_proj_weight: tracked_create_storage_buffer_init_f32(
            stats,
            device,
            &format!("layer_{layer_idx}_up_proj"),
            &layer.up_proj_weight,
        ),
        down_proj_weight: tracked_create_storage_buffer_init_f32(
            stats,
            device,
            &format!("layer_{layer_idx}_down_proj"),
            &layer.down_proj_weight,
        ),
        post_attention_layernorm_weight: tracked_create_storage_buffer_init_f32(
            stats,
            device,
            &format!("layer_{layer_idx}_post_attn_ln"),
            &layer.post_attention_layernorm_weight,
        ),
    }
}

fn maybe_upload_embeddings(
    stats: &Arc<AllocationStats>,
    device: &wgpu::Device,
    runtime: &GpuRuntimeConfig,
    weights: &Qwen3Weights,
) -> Result<Option<wgpu::Buffer>> {
    if !runtime.upload_embeddings_to_gpu {
        return Ok(None);
    }
    let bytes = bytes_f32(weights.embed_tokens.len())?;
    if bytes <= device.limits().max_buffer_size
        && bytes <= device.limits().max_storage_buffer_binding_size as u64
    {
        Ok(Some(tracked_create_storage_buffer_init_f32(
            stats,
            device,
            "embed_tokens_gpu",
            &weights.embed_tokens,
        )))
    } else {
        Ok(None)
    }
}

pub(super) fn create_activation_buffers(
    stats: &Arc<AllocationStats>,
    device: &wgpu::Device,
    config: &Qwen3Config,
    runtime: &GpuRuntimeConfig,
    param_slots: usize,
) -> Result<ActivationBuffers> {
    let hidden_bytes = bytes_f32(checked_mul(
        runtime.max_seq_len,
        config.hidden_size,
        "hidden",
    )?)?;
    let q_bytes = bytes_f32(checked_mul(runtime.max_seq_len, config.q_dim(), "q")?)?;
    let kv_bytes = bytes_f32(checked_mul(runtime.max_seq_len, config.kv_dim(), "kv")?)?;
    let scores_bytes = bytes_f32(checked_mul(
        checked_mul(runtime.max_seq_len, runtime.max_seq_len, "scores")?,
        config.num_attention_heads,
        "scores heads",
    )?)?;
    let intermediate_bytes = bytes_f32(checked_mul(
        runtime.max_seq_len,
        config.intermediate_size,
        "intermediate",
    )?)?;
    let params_bytes = (checked_mul(param_slots, PARAM_SLOT_BYTES, "param buffer bytes")?) as u64;
    let readback_bytes = hidden_bytes;

    let activation_usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    Ok(ActivationBuffers {
        hidden: tracked_create_buffer(stats, device, "hidden", hidden_bytes, activation_usage),
        residual: tracked_create_buffer(stats, device, "residual", hidden_bytes, activation_usage),
        q: tracked_create_buffer(stats, device, "q_buf", q_bytes, activation_usage),
        k: tracked_create_buffer(stats, device, "k_buf", kv_bytes, activation_usage),
        v: tracked_create_buffer(stats, device, "v_buf", kv_bytes, activation_usage),
        scores: tracked_create_buffer(stats, device, "scores", scores_bytes, activation_usage),
        attn_out: tracked_create_buffer(stats, device, "attn_out", q_bytes, activation_usage),
        gate: tracked_create_buffer(
            stats,
            device,
            "gate_buf",
            intermediate_bytes,
            activation_usage,
        ),
        up: tracked_create_buffer(
            stats,
            device,
            "up_buf",
            intermediate_bytes,
            activation_usage,
        ),
        params: tracked_create_buffer(
            stats,
            device,
            "params",
            params_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ),
        readback: tracked_create_buffer(
            stats,
            device,
            "readback",
            readback_bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        ),
    })
}

pub(super) fn tracked_create_storage_buffer_init_f32(
    stats: &AllocationStats,
    device: &wgpu::Device,
    label: &str,
    data: &[f32],
) -> wgpu::Buffer {
    stats.user_buffer_creations.fetch_add(1, Ordering::Relaxed);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

pub(super) fn tracked_create_buffer(
    stats: &AllocationStats,
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    stats.user_buffer_creations.fetch_add(1, Ordering::Relaxed);
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}
