use std::num::NonZeroU64;

use super::api::{Qwen3Config, Result};
use super::buffers::{ActivationBuffers, GpuWeights};
use super::params::PARAM_SLOT_BYTES;
use super::pipelines::BindGroupLayouts;

pub(super) struct LayerBindGroups {
    pub(super) input_norm: wgpu::BindGroup,
    pub(super) q_proj: wgpu::BindGroup,
    pub(super) k_proj: wgpu::BindGroup,
    pub(super) v_proj: wgpu::BindGroup,
    pub(super) q_head_norm: wgpu::BindGroup,
    pub(super) k_head_norm: wgpu::BindGroup,
    pub(super) o_proj: wgpu::BindGroup,
    pub(super) post_attn_norm: wgpu::BindGroup,
    pub(super) gate_proj: wgpu::BindGroup,
    pub(super) up_proj: wgpu::BindGroup,
    pub(super) down_proj: wgpu::BindGroup,
}

pub(super) struct GlobalBindGroups {
    pub(super) copy_hidden_to_residual: wgpu::BindGroup,
    pub(super) add_hidden_residual: wgpu::BindGroup,
    pub(super) rope_q: wgpu::BindGroup,
    pub(super) rope_k: wgpu::BindGroup,
    pub(super) attn_scores: wgpu::BindGroup,
    pub(super) attn_softmax: wgpu::BindGroup,
    pub(super) attn_context: wgpu::BindGroup,
    pub(super) silu_gate: wgpu::BindGroup,
    pub(super) mul_gate_up: wgpu::BindGroup,
    pub(super) final_norm: wgpu::BindGroup,
}

pub(super) struct BindGroups {
    pub(super) layers: Vec<LayerBindGroups>,
    pub(super) globals: GlobalBindGroups,
}

pub(super) fn create_bind_groups(
    device: &wgpu::Device,
    layouts: &BindGroupLayouts,
    weights: &GpuWeights,
    activations: &ActivationBuffers,
    config: &Qwen3Config,
) -> Result<BindGroups> {
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer in &weights.layers {
        layers.push(LayerBindGroups {
            input_norm: two_buffer_bind_group(
                device,
                &layouts.two_buffer_with_params,
                "bg_input_norm",
                &activations.hidden,
                &layer.input_layernorm_weight,
                &activations.params,
            ),
            q_proj: three_buffer_bind_group(
                device,
                &layouts.three_buffer_with_params,
                "bg_q_proj",
                &activations.hidden,
                &layer.q_proj_weight,
                &activations.q,
                &activations.params,
            ),
            k_proj: three_buffer_bind_group(
                device,
                &layouts.three_buffer_with_params,
                "bg_k_proj",
                &activations.hidden,
                &layer.k_proj_weight,
                &activations.k,
                &activations.params,
            ),
            v_proj: three_buffer_bind_group(
                device,
                &layouts.three_buffer_with_params,
                "bg_v_proj",
                &activations.hidden,
                &layer.v_proj_weight,
                &activations.v,
                &activations.params,
            ),
            q_head_norm: two_buffer_bind_group(
                device,
                &layouts.two_buffer_with_params,
                "bg_q_head_norm",
                &activations.q,
                &layer.q_norm_weight,
                &activations.params,
            ),
            k_head_norm: two_buffer_bind_group(
                device,
                &layouts.two_buffer_with_params,
                "bg_k_head_norm",
                &activations.k,
                &layer.k_norm_weight,
                &activations.params,
            ),
            o_proj: three_buffer_bind_group(
                device,
                &layouts.three_buffer_with_params,
                "bg_o_proj",
                &activations.attn_out,
                &layer.o_proj_weight,
                &activations.hidden,
                &activations.params,
            ),
            post_attn_norm: two_buffer_bind_group(
                device,
                &layouts.two_buffer_with_params,
                "bg_post_attn_norm",
                &activations.hidden,
                &layer.post_attention_layernorm_weight,
                &activations.params,
            ),
            gate_proj: three_buffer_bind_group(
                device,
                &layouts.three_buffer_with_params,
                "bg_gate_proj",
                &activations.hidden,
                &layer.gate_proj_weight,
                &activations.gate,
                &activations.params,
            ),
            up_proj: three_buffer_bind_group(
                device,
                &layouts.three_buffer_with_params,
                "bg_up_proj",
                &activations.hidden,
                &layer.up_proj_weight,
                &activations.up,
                &activations.params,
            ),
            down_proj: three_buffer_bind_group(
                device,
                &layouts.three_buffer_with_params,
                "bg_down_proj",
                &activations.gate,
                &layer.down_proj_weight,
                &activations.hidden,
                &activations.params,
            ),
        });
    }

    let globals = GlobalBindGroups {
        copy_hidden_to_residual: two_buffer_bind_group(
            device,
            &layouts.two_buffer_with_params,
            "bg_copy_hidden_to_residual",
            &activations.residual,
            &activations.hidden,
            &activations.params,
        ),
        add_hidden_residual: two_buffer_bind_group(
            device,
            &layouts.two_buffer_with_params,
            "bg_add_hidden_residual",
            &activations.hidden,
            &activations.residual,
            &activations.params,
        ),
        rope_q: rope_bind_group(
            device,
            &layouts.rope_with_params,
            "bg_rope_q",
            &activations.q,
            &weights.rope_cos,
            &weights.rope_sin,
            &activations.params,
        ),
        rope_k: rope_bind_group(
            device,
            &layouts.rope_with_params,
            "bg_rope_k",
            &activations.k,
            &weights.rope_cos,
            &weights.rope_sin,
            &activations.params,
        ),
        attn_scores: three_buffer_bind_group(
            device,
            &layouts.three_buffer_with_params,
            "bg_attn_scores",
            &activations.q,
            &activations.k,
            &activations.scores,
            &activations.params,
        ),
        attn_softmax: one_buffer_bind_group(
            device,
            &layouts.one_buffer_with_params,
            "bg_attn_softmax",
            &activations.scores,
            &activations.params,
        ),
        attn_context: three_buffer_bind_group(
            device,
            &layouts.three_buffer_with_params,
            "bg_attn_context",
            &activations.scores,
            &activations.v,
            &activations.attn_out,
            &activations.params,
        ),
        silu_gate: one_buffer_bind_group(
            device,
            &layouts.one_buffer_with_params,
            "bg_silu_gate",
            &activations.gate,
            &activations.params,
        ),
        mul_gate_up: two_buffer_bind_group(
            device,
            &layouts.two_buffer_with_params,
            "bg_mul_gate_up",
            &activations.gate,
            &activations.up,
            &activations.params,
        ),
        final_norm: two_buffer_bind_group(
            device,
            &layouts.two_buffer_with_params,
            "bg_final_norm",
            &activations.hidden,
            &weights.norm_weight,
            &activations.params,
        ),
    };

    Ok(BindGroups { layers, globals })
}

fn params_binding(buffer: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: Some(
            NonZeroU64::new(PARAM_SLOT_BYTES as u64)
                .expect("invariant: PARAM_SLOT_BYTES is non-zero"),
        ),
    })
}

fn three_buffer_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    c: &wgpu::Buffer,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_binding(params),
            },
        ],
    })
}

fn two_buffer_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    a: &wgpu::Buffer,
    b: &wgpu::Buffer,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_binding(params),
            },
        ],
    })
}

fn one_buffer_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    a: &wgpu::Buffer,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_binding(params),
            },
        ],
    })
}

fn rope_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    x: &wgpu::Buffer,
    cos: &wgpu::Buffer,
    sin: &wgpu::Buffer,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cos.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sin.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_binding(params),
            },
        ],
    })
}
