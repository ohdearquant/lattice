use std::sync::{Arc, Mutex, atomic::Ordering, mpsc};

use super::api::checked_mul;
use super::api::{GpuForwardError, GpuRuntimeConfig, Qwen3Config, Qwen3Weights, Result};
use super::bind_groups::{BindGroups, create_bind_groups};
use super::buffers::{ActivationBuffers, GpuWeights, create_activation_buffers, upload_weights};
use super::dims::{ForwardDims, validate_model_config, validate_runtime_feasibility};
use super::dispatch::{
    dispatch_add, dispatch_attention_context, dispatch_attention_scores,
    dispatch_attention_softmax, dispatch_copy, dispatch_matmul, dispatch_mul, dispatch_rms_norm,
    dispatch_rope, dispatch_silu,
};
use super::params::{AllocationStats, EXTRA_PARAM_SLOTS, PARAM_SLOTS_PER_LAYER, ParamPacker};
use super::pipelines::{BindGroupLayouts, Pipelines, create_bind_group_layouts, create_pipelines};
use super::util::{build_rope_tables, bytes_f32};

/// **Unstable**: GPU model state holding device, pipelines, and weight buffers; layout evolving.
pub struct GpuModelState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: Qwen3Config,
    pub runtime: GpuRuntimeConfig,
    weights: GpuWeights,
    activations: ActivationBuffers,
    bind_groups: BindGroups,
    pipelines: Pipelines,
    embed_tokens_cpu: Arc<[f32]>,
    stats: Arc<AllocationStats>,
    execution_lock: Mutex<()>,
}

impl GpuModelState {
    /// **Unstable**: create GPU model state; pipeline compilation and buffer layout may change.
    pub fn new(
        config: Qwen3Config,
        weights: &Qwen3Weights,
        runtime: GpuRuntimeConfig,
    ) -> Result<Self> {
        weights.validate(&config)?;
        validate_model_config(&config)?;

        if runtime.max_seq_len == 0 {
            return Err(GpuForwardError::InvalidInput(
                "runtime.max_seq_len must be > 0".to_string(),
            ));
        }
        if runtime.max_seq_len > config.max_position_embeddings {
            return Err(GpuForwardError::InvalidInput(format!(
                "runtime.max_seq_len {} exceeds model max_position_embeddings {}",
                runtime.max_seq_len, config.max_position_embeddings
            )));
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or(GpuForwardError::NoAdapter)?;

        let info = adapter.get_info();
        tracing::info!(
            name = info.name,
            backend = ?info.backend,
            "wgpu GPU initialized for Qwen3 forward pass"
        );

        let required_limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("qwen3_gpu_forward_device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;

        let stats = Arc::new(AllocationStats::default());
        let embed_tokens_cpu: Arc<[f32]> = Arc::from(weights.embed_tokens.clone());

        validate_runtime_feasibility(&device.limits(), &config, &runtime)?;

        let (rope_cos, rope_sin) =
            build_rope_tables(config.head_dim, runtime.max_seq_len, config.rope_theta);

        let gpu_weights = upload_weights(
            &stats, &device, &config, &runtime, weights, &rope_cos, &rope_sin,
        )?;
        let param_slots = checked_mul(
            config.num_hidden_layers,
            PARAM_SLOTS_PER_LAYER,
            "param slots",
        )? + EXTRA_PARAM_SLOTS;
        let activations =
            create_activation_buffers(&stats, &device, &config, &runtime, param_slots)?;
        let layouts = create_bind_group_layouts(&device);
        let pipelines = create_pipelines(&device, &layouts);
        let bind_groups =
            create_bind_groups(&device, &layouts, &gpu_weights, &activations, &config)?;

        Ok(Self {
            device,
            queue,
            config,
            runtime,
            weights: gpu_weights,
            activations,
            bind_groups,
            pipelines,
            embed_tokens_cpu,
            stats,
            execution_lock: Mutex::new(()),
        })
    }

    /// **Unstable**: GPU forward pass; shader dispatch and output format may change.
    pub fn forward(&self, input_ids: &[u32], seq_len: usize) -> Result<Vec<f32>> {
        let _guard = self
            .execution_lock
            .lock()
            .map_err(|_| GpuForwardError::ExecutionLockPoisoned)?;

        self.validate_forward_request(input_ids, seq_len)?;

        let dims = ForwardDims::from_config(&self.config, seq_len)?;
        let hidden_cpu = self.gather_embeddings(input_ids, seq_len)?;
        self.queue.write_buffer(
            &self.activations.hidden,
            0,
            bytemuck::cast_slice(&hidden_cpu),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qwen3_forward_encoder"),
            });

        let mut params = ParamPacker::new(self.max_param_slots());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("qwen3_forward_compute"),
                timestamp_writes: None,
            });

            for layer_idx in 0..self.config.num_hidden_layers {
                self.dispatch_layer_qkv(&mut pass, layer_idx, &dims, &mut params)?;
                self.dispatch_layer_attn(&mut pass, layer_idx, &dims, &mut params)?;
                self.dispatch_layer_ffn(&mut pass, layer_idx, &dims, &mut params)?;
            }

            dispatch_rms_norm(
                &mut pass,
                &self.pipelines.rms_norm,
                &self.bind_groups.globals.final_norm,
                dims.seq,
                dims.hidden,
                self.config.rms_norm_eps,
                &mut params,
            )?;
        }

        self.queue
            .write_buffer(&self.activations.params, 0, params.as_bytes());

        let active_hidden_bytes = bytes_f32(dims.hidden_elems as usize)?;
        encoder.copy_buffer_to_buffer(
            &self.activations.hidden,
            0,
            &self.activations.readback,
            0,
            active_hidden_bytes,
        );

        self.queue.submit(Some(encoder.finish()));
        self.stats.queue_submits.fetch_add(1, Ordering::Relaxed);

        let slice = self.activations.readback.slice(0..active_hidden_bytes);
        let (tx, rx) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(GpuForwardError::BufferMap(e)),
            Err(_) => return Err(GpuForwardError::ChannelClosed),
        }

        let mapped = slice.get_mapped_range();
        let mut out = vec![0.0f32; dims.hidden_elems as usize];
        for (dst, chunk) in out.iter_mut().zip(mapped.chunks_exact(4)) {
            *dst = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        drop(mapped);
        self.activations.readback.unmap();
        Ok(out)
    }

    /// **Unstable**: number of wgpu queue submissions; diagnostic counter.
    pub fn submit_count(&self) -> u64 {
        self.stats.queue_submits.load(Ordering::Relaxed)
    }

    /// **Unstable**: number of user-side buffer creations; diagnostic counter.
    pub fn user_buffer_creation_count(&self) -> u64 {
        self.stats.user_buffer_creations.load(Ordering::Relaxed)
    }

    /// **Unstable**: whether embeddings are resident in GPU memory.
    pub fn has_gpu_embedding_buffer(&self) -> bool {
        self.weights.embed_tokens_gpu.is_some()
    }

    fn validate_forward_request(&self, input_ids: &[u32], seq_len: usize) -> Result<()> {
        if input_ids.len() != seq_len {
            return Err(GpuForwardError::InvalidInput(format!(
                "input_ids length {} does not match seq_len {seq_len}",
                input_ids.len()
            )));
        }
        if seq_len == 0 {
            return Err(GpuForwardError::InvalidInput(
                "seq_len must be > 0".to_string(),
            ));
        }
        if seq_len > self.runtime.max_seq_len {
            return Err(GpuForwardError::InvalidInput(format!(
                "seq_len {seq_len} exceeds runtime.max_seq_len {}",
                self.runtime.max_seq_len
            )));
        }
        if seq_len > self.config.max_position_embeddings {
            return Err(GpuForwardError::InvalidInput(format!(
                "seq_len {seq_len} exceeds model max_position_embeddings {}",
                self.config.max_position_embeddings
            )));
        }
        Ok(())
    }

    fn gather_embeddings(&self, input_ids: &[u32], seq_len: usize) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let mut dense = vec![0.0f32; seq_len * hidden];
        for (i, &tok_id) in input_ids.iter().enumerate() {
            let tok_id = tok_id as usize;
            if tok_id >= self.config.vocab_size {
                return Err(GpuForwardError::InvalidInput(format!(
                    "token id {} out of range [0, {})",
                    tok_id, self.config.vocab_size
                )));
            }
            let src = &self.embed_tokens_cpu[tok_id * hidden..(tok_id + 1) * hidden];
            dense[i * hidden..(i + 1) * hidden].copy_from_slice(src);
        }
        Ok(dense)
    }

    fn max_param_slots(&self) -> usize {
        self.config.num_hidden_layers * PARAM_SLOTS_PER_LAYER + EXTRA_PARAM_SLOTS
    }

    fn dispatch_layer_qkv(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        layer_idx: usize,
        dims: &ForwardDims,
        params: &mut ParamPacker,
    ) -> Result<()> {
        let layer = &self.bind_groups.layers[layer_idx];
        let globals = &self.bind_groups.globals;
        dispatch_copy(
            pass,
            &self.pipelines.copy,
            &globals.copy_hidden_to_residual,
            dims.hidden_elems,
            params,
        )?;
        dispatch_rms_norm(
            pass,
            &self.pipelines.rms_norm,
            &layer.input_norm,
            dims.seq,
            dims.hidden,
            self.config.rms_norm_eps,
            params,
        )?;
        dispatch_matmul(
            pass,
            &self.pipelines.matmul_bt,
            &layer.q_proj,
            dims.seq,
            dims.q_dim,
            dims.hidden,
            params,
        )?;
        dispatch_matmul(
            pass,
            &self.pipelines.matmul_bt,
            &layer.k_proj,
            dims.seq,
            dims.kv_dim,
            dims.hidden,
            params,
        )?;
        dispatch_matmul(
            pass,
            &self.pipelines.matmul_bt,
            &layer.v_proj,
            dims.seq,
            dims.kv_dim,
            dims.hidden,
            params,
        )?;
        Ok(())
    }

    fn dispatch_layer_attn(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        layer_idx: usize,
        dims: &ForwardDims,
        params: &mut ParamPacker,
    ) -> Result<()> {
        let layer = &self.bind_groups.layers[layer_idx];
        let globals = &self.bind_groups.globals;
        dispatch_rms_norm(
            pass,
            &self.pipelines.rms_norm,
            &layer.q_head_norm,
            dims.seq_heads,
            dims.head_dim,
            self.config.rms_norm_eps,
            params,
        )?;
        dispatch_rms_norm(
            pass,
            &self.pipelines.rms_norm,
            &layer.k_head_norm,
            dims.seq_kv_heads,
            dims.head_dim,
            self.config.rms_norm_eps,
            params,
        )?;
        dispatch_rope(
            pass,
            &self.pipelines.rope,
            &globals.rope_q,
            dims.seq,
            dims.num_heads,
            dims.head_dim,
            params,
        )?;
        dispatch_rope(
            pass,
            &self.pipelines.rope,
            &globals.rope_k,
            dims.seq,
            dims.num_kv_heads,
            dims.head_dim,
            params,
        )?;
        dispatch_attention_scores(
            pass,
            &self.pipelines.attn_scores,
            &globals.attn_scores,
            dims,
            params,
        )?;
        dispatch_attention_softmax(
            pass,
            &self.pipelines.attn_softmax,
            &globals.attn_softmax,
            dims,
            params,
        )?;
        dispatch_attention_context(
            pass,
            &self.pipelines.attn_context,
            &globals.attn_context,
            dims,
            params,
        )?;
        dispatch_matmul(
            pass,
            &self.pipelines.matmul_bt,
            &layer.o_proj,
            dims.seq,
            dims.hidden,
            dims.q_dim,
            params,
        )?;
        dispatch_add(
            pass,
            &self.pipelines.add,
            &globals.add_hidden_residual,
            dims.hidden_elems,
            params,
        )?;
        Ok(())
    }

    fn dispatch_layer_ffn(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        layer_idx: usize,
        dims: &ForwardDims,
        params: &mut ParamPacker,
    ) -> Result<()> {
        let layer = &self.bind_groups.layers[layer_idx];
        let globals = &self.bind_groups.globals;
        dispatch_copy(
            pass,
            &self.pipelines.copy,
            &globals.copy_hidden_to_residual,
            dims.hidden_elems,
            params,
        )?;
        dispatch_rms_norm(
            pass,
            &self.pipelines.rms_norm,
            &layer.post_attn_norm,
            dims.seq,
            dims.hidden,
            self.config.rms_norm_eps,
            params,
        )?;
        dispatch_matmul(
            pass,
            &self.pipelines.matmul_bt,
            &layer.gate_proj,
            dims.seq,
            dims.intermediate,
            dims.hidden,
            params,
        )?;
        dispatch_matmul(
            pass,
            &self.pipelines.matmul_bt,
            &layer.up_proj,
            dims.seq,
            dims.intermediate,
            dims.hidden,
            params,
        )?;
        dispatch_silu(
            pass,
            &self.pipelines.silu,
            &globals.silu_gate,
            dims.intermediate_elems,
            params,
        )?;
        dispatch_mul(
            pass,
            &self.pipelines.mul,
            &globals.mul_gate_up,
            dims.intermediate_elems,
            params,
        )?;
        dispatch_matmul(
            pass,
            &self.pipelines.matmul_bt,
            &layer.down_proj,
            dims.seq,
            dims.hidden,
            dims.intermediate,
            params,
        )?;
        dispatch_add(
            pass,
            &self.pipelines.add,
            &globals.add_hidden_residual,
            dims.hidden_elems,
            params,
        )?;
        Ok(())
    }
}
