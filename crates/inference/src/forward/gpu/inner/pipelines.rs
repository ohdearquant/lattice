use std::{borrow::Cow, num::NonZeroU64};

use super::params::PARAM_SLOT_BYTES;
use super::shaders::{
    ADD_SHADER, ATTENTION_CONTEXT_SHADER, ATTENTION_SCORES_SHADER, ATTENTION_SOFTMAX_SHADER,
    COPY_SHADER, MATMUL_BT_SHADER, MUL_SHADER, RMS_NORM_SHADER, ROPE_SHADER, SILU_SHADER,
};

pub(super) struct BindGroupLayouts {
    pub(super) three_buffer_with_params: wgpu::BindGroupLayout,
    pub(super) two_buffer_with_params: wgpu::BindGroupLayout,
    pub(super) one_buffer_with_params: wgpu::BindGroupLayout,
    pub(super) rope_with_params: wgpu::BindGroupLayout,
}

pub(super) struct Pipelines {
    pub(super) matmul_bt: wgpu::ComputePipeline,
    pub(super) rms_norm: wgpu::ComputePipeline,
    pub(super) copy: wgpu::ComputePipeline,
    pub(super) add: wgpu::ComputePipeline,
    pub(super) silu: wgpu::ComputePipeline,
    pub(super) mul: wgpu::ComputePipeline,
    pub(super) rope: wgpu::ComputePipeline,
    pub(super) attn_scores: wgpu::ComputePipeline,
    pub(super) attn_softmax: wgpu::ComputePipeline,
    pub(super) attn_context: wgpu::ComputePipeline,
}

pub(super) fn create_bind_group_layouts(device: &wgpu::Device) -> BindGroupLayouts {
    let param_size =
        NonZeroU64::new(PARAM_SLOT_BYTES as u64).expect("invariant: PARAM_SLOT_BYTES is non-zero");

    let three_buffer_with_params =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout_3buf_params"),
            entries: &[
                storage_entry(0, true, false, None),
                storage_entry(1, true, false, None),
                storage_entry(2, false, false, None),
                storage_entry(3, true, true, Some(param_size)),
            ],
        });

    let two_buffer_with_params =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout_2buf_params"),
            entries: &[
                storage_entry(0, false, false, None),
                storage_entry(1, true, false, None),
                storage_entry(2, true, true, Some(param_size)),
            ],
        });

    let one_buffer_with_params =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout_1buf_params"),
            entries: &[
                storage_entry(0, false, false, None),
                storage_entry(1, true, true, Some(param_size)),
            ],
        });

    let rope_with_params = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("layout_rope_params"),
        entries: &[
            storage_entry(0, false, false, None),
            storage_entry(1, true, false, None),
            storage_entry(2, true, false, None),
            storage_entry(3, true, true, Some(param_size)),
        ],
    });

    BindGroupLayouts {
        three_buffer_with_params,
        two_buffer_with_params,
        one_buffer_with_params,
        rope_with_params,
    }
}

pub(super) fn create_pipelines(device: &wgpu::Device, layouts: &BindGroupLayouts) -> Pipelines {
    let three_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout_3buf"),
        bind_group_layouts: &[&layouts.three_buffer_with_params],
        push_constant_ranges: &[],
    });
    let two_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout_2buf"),
        bind_group_layouts: &[&layouts.two_buffer_with_params],
        push_constant_ranges: &[],
    });
    let one_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout_1buf"),
        bind_group_layouts: &[&layouts.one_buffer_with_params],
        push_constant_ranges: &[],
    });
    let rope_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout_rope"),
        bind_group_layouts: &[&layouts.rope_with_params],
        push_constant_ranges: &[],
    });

    let matmul_shader = shader_module(device, "matmul_bt", MATMUL_BT_SHADER);
    let rms_shader = shader_module(device, "rms_norm", RMS_NORM_SHADER);
    let copy_shader = shader_module(device, "copy", COPY_SHADER);
    let add_shader = shader_module(device, "add", ADD_SHADER);
    let silu_shader = shader_module(device, "silu", SILU_SHADER);
    let mul_shader = shader_module(device, "mul", MUL_SHADER);
    let rope_shader = shader_module(device, "rope", ROPE_SHADER);
    let attn_scores_shader = shader_module(device, "attn_scores", ATTENTION_SCORES_SHADER);
    let attn_softmax_shader = shader_module(device, "attn_softmax", ATTENTION_SOFTMAX_SHADER);
    let attn_context_shader = shader_module(device, "attn_context", ATTENTION_CONTEXT_SHADER);

    Pipelines {
        matmul_bt: compute_pipeline(
            device,
            &three_layout,
            &matmul_shader,
            Some("gemm_bt"),
            "pipeline_matmul_bt",
        ),
        rms_norm: compute_pipeline(
            device,
            &two_layout,
            &rms_shader,
            Some("rms_norm"),
            "pipeline_rms_norm",
        ),
        copy: compute_pipeline(
            device,
            &two_layout,
            &copy_shader,
            Some("copy_kernel"),
            "pipeline_copy",
        ),
        add: compute_pipeline(
            device,
            &two_layout,
            &add_shader,
            Some("add_kernel"),
            "pipeline_add",
        ),
        silu: compute_pipeline(
            device,
            &one_layout,
            &silu_shader,
            Some("silu_kernel"),
            "pipeline_silu",
        ),
        mul: compute_pipeline(
            device,
            &two_layout,
            &mul_shader,
            Some("mul_kernel"),
            "pipeline_mul",
        ),
        rope: compute_pipeline(
            device,
            &rope_layout,
            &rope_shader,
            Some("rope_kernel"),
            "pipeline_rope",
        ),
        attn_scores: compute_pipeline(
            device,
            &three_layout,
            &attn_scores_shader,
            Some("attention_scores"),
            "pipeline_attn_scores",
        ),
        attn_softmax: compute_pipeline(
            device,
            &one_layout,
            &attn_softmax_shader,
            Some("attention_softmax"),
            "pipeline_attn_softmax",
        ),
        attn_context: compute_pipeline(
            device,
            &three_layout,
            &attn_context_shader,
            Some("attention_context"),
            "pipeline_attn_context",
        ),
    }
}

pub(super) fn storage_entry(
    binding: u32,
    read_only: bool,
    has_dynamic_offset: bool,
    min_binding_size: Option<NonZeroU64>,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset,
            min_binding_size,
        },
        count: None,
    }
}

pub(super) fn shader_module(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
    })
}

pub(super) fn compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry_point: Option<&str>,
    label: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module,
        entry_point,
        compilation_options: Default::default(),
        cache: None,
    })
}
