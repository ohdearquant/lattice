//! Direct Metal compute shader forward pass for Qwen3-Embedding-0.6B.
//!
//! Bypasses wgpu abstraction for lower per-dispatch overhead on Apple Silicon.
//! All buffers use `StorageModeShared` for zero-copy unified memory access.
//! The entire 28-layer forward pass is encoded into a single `MTLCommandBuffer`.

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
mod inner {
    use crate::model::qwen::QwenConfig;
    use crate::weights::QwenWeights;
    use metal::*;

    // ---------------------------------------------------------------------------
    // MSL Compute Shaders
    // ---------------------------------------------------------------------------

    const MSL_TEMPLATE: &str = include_str!("shaders/flash_attention.metal");

    // ---------------------------------------------------------------------------
    // Weight and Activation Buffer Structs
    // ---------------------------------------------------------------------------

    struct MetalLayerWeights {
        q_proj_weight: Buffer,
        k_proj_weight: Buffer,
        v_proj_weight: Buffer,
        o_proj_weight: Buffer,
        q_norm_weight: Buffer,
        k_norm_weight: Buffer,
        input_layernorm_weight: Buffer,
        gate_proj_weight: Buffer,
        up_proj_weight: Buffer,
        down_proj_weight: Buffer,
        post_attention_layernorm_weight: Buffer,
    }

    /// Params for fused QK-norm + RoPE kernel.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct FusedQkNormRopeParams {
        seq_len: u32,
        q_heads: u32,
        k_heads: u32,
        q_stride: u32,
        k_stride: u32,
        eps: f32,
    }

    struct MetalActivations {
        hidden: Buffer,
        residual: Buffer,
        q: Buffer,
        k: Buffer,
        v: Buffer,
        attn_out: Buffer,
        gate: Buffer,
        up: Buffer,
        ffn_out: Buffer,
    }

    /// Configuration mirroring QwenConfig but with pre-computed derived values.
    #[derive(Clone)]
    struct Qwen3GpuConfig {
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        rms_norm_eps: f32,
        q_dim: usize,
        kv_dim: usize,
        max_seq_len: usize,
        /// Debug-only layer limit (from METAL_LAYERS env var at init time).
        layer_limit: Option<usize>,
    }

    /// Validate model shape for Metal fused kernels and return the GQA group count.
    ///
    /// Enforces structural requirements only — head_dim must be divisible by 4
    /// (for float4 operations) and nonzero, kv_heads nonzero, and q_heads
    /// divisible by kv_heads. The exact values are injected into the MSL source
    /// at compile time via `msl_source_for`.
    fn validate_fused_kernel_shape(config: &QwenConfig) -> Result<usize, String> {
        if config.head_dim == 0 {
            return Err("Metal fused attention requires nonzero head_dim".to_string());
        }
        if !config.head_dim.is_multiple_of(4) {
            return Err(format!(
                "Metal fused attention requires head_dim divisible by 4 (for float4 ops); got {}",
                config.head_dim
            ));
        }
        if config.num_key_value_heads == 0 {
            return Err("Metal fused attention requires nonzero num_key_value_heads".to_string());
        }
        if !config
            .num_attention_heads
            .is_multiple_of(config.num_key_value_heads)
        {
            return Err(format!(
                "Metal fused attention requires num_attention_heads divisible by num_key_value_heads; got {}/{}",
                config.num_attention_heads, config.num_key_value_heads
            ));
        }
        let gqa_groups = config.num_attention_heads / config.num_key_value_heads;
        Ok(gqa_groups)
    }

    /// Produce MSL source with model-specific constants substituted in.
    ///
    /// Derived values injected:
    /// - `half_dim = head_dim / 2`  (RoPE rotation pairs, also threadgroup size)
    ///   `FA_HEAD_DIM4` is derived from `FA_HEAD_DIM` inside the MSL, not injected separately.
    fn msl_source_for(head_dim: usize, gqa_groups: usize) -> String {
        let half_dim = head_dim / 2;
        let threads = head_dim / 2;
        MSL_TEMPLATE
            .replace("__FA_HEAD_DIM__", &head_dim.to_string())
            .replace("__FA_GQA_GROUPS__", &gqa_groups.to_string())
            .replace("__FUSED_C_HEAD_DIM__", &head_dim.to_string())
            .replace("__FUSED_C_HALF_DIM__", &half_dim.to_string())
            .replace("__FUSED_C_THREADS__", &threads.to_string())
    }

    /// Direct Metal forward pass for Qwen3-Embedding.
    ///
    /// All weight buffers are persistent in unified memory (`StorageModeShared`).
    /// Activation buffers are pre-allocated for `max_seq_len`.
    /// A single `MTLCommandBuffer` encodes all 28 layers per forward call.
    pub struct MetalForwardPass {
        #[allow(dead_code)]
        // TODO(#1958): Metal roadmap — device handle kept alive for GPU lifetime management
        device: Device,
        queue: CommandQueue,
        matmul_pipeline: ComputePipelineState,
        rms_norm_pipeline: ComputePipelineState,
        fused_attention_pipeline: ComputePipelineState,
        fused_qk_norm_rope_pipeline: ComputePipelineState,
        silu_mul_pipeline: ComputePipelineState,
        copy_pipeline: ComputePipelineState,
        add_pipeline: ComputePipelineState,
        layer_weights: Vec<MetalLayerWeights>,
        final_norm_weight: Buffer,
        rope_cos: Buffer,
        rope_sin: Buffer,
        activations: MetalActivations,
        config: Qwen3GpuConfig,
    }

    // Helper: create a Metal buffer from a float slice.
    fn make_buffer(device: &Device, data: &[f32], label: &str) -> Buffer {
        let byte_len = std::mem::size_of_val(data) as u64;
        let buf = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );
        buf.set_label(label);
        buf
    }

    // Helper: create a zero-filled buffer of `num_floats` capacity.
    fn make_zero_buffer(device: &Device, num_floats: usize, label: &str) -> Buffer {
        let byte_len = (num_floats * std::mem::size_of::<f32>()) as u64;
        let buf = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
        buf.set_label(label);
        buf
    }

    // Helper: ceil-divide.
    fn div_ceil(a: u64, b: u64) -> u64 {
        a.div_ceil(b)
    }

    impl MetalForwardPass {
        /// Create from CPU weights. Uploads all weights to unified memory buffers.
        ///
        /// `max_seq_len` caps the pre-allocated activation buffer size.
        /// For embedding workloads, 512 is typically sufficient.
        pub fn new(
            config: &QwenConfig,
            weights: &QwenWeights<'_>,
            max_seq_len: usize,
        ) -> Result<Self, String> {
            let device =
                Device::system_default().ok_or_else(|| "No Metal device found".to_string())?;

            tracing::info!(
                name = device.name(),
                "Metal GPU initialized for Qwen3 forward pass"
            );

            let queue = device.new_command_queue();

            // Validate model shape and extract GQA group count before shader compilation.
            let gqa_groups = validate_fused_kernel_shape(config)?;

            // Compile shaders with model-specific constants substituted in.
            let opts = CompileOptions::new();
            let msl = msl_source_for(config.head_dim, gqa_groups);
            let library = device
                .new_library_with_source(&msl, &opts)
                .map_err(|e| format!("Metal shader compilation failed: {e}"))?;

            let make_pipeline = |name: &str| -> Result<ComputePipelineState, String> {
                let func = library
                    .get_function(name, None)
                    .map_err(|e| format!("function '{name}' not found: {e}"))?;
                device
                    .new_compute_pipeline_state_with_function(&func)
                    .map_err(|e| format!("pipeline for '{name}' failed: {e}"))
            };

            let matmul_pipeline = make_pipeline("matmul_bt")?;
            let rms_norm_pipeline = make_pipeline("rms_norm")?;
            let fused_attention_pipeline = make_pipeline("fused_attention")?;
            let fused_qk_norm_rope_pipeline = make_pipeline("fused_qk_norm_rope")?;
            let silu_mul_pipeline = make_pipeline("silu_mul")?;
            let copy_pipeline = make_pipeline("copy_buf")?;
            let add_pipeline = make_pipeline("add_buf")?;

            let q_dim = config.q_dim();
            let kv_dim = config.kv_dim();
            // Read METAL_LAYERS env var once at init (not per-forward).
            let layer_limit = std::env::var("METAL_LAYERS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok());

            let gpu_config = Qwen3GpuConfig {
                hidden_size: config.hidden_size,
                num_hidden_layers: config.num_hidden_layers,
                num_attention_heads: config.num_attention_heads,
                num_key_value_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                intermediate_size: config.intermediate_size,
                rms_norm_eps: config.rms_norm_eps,
                q_dim,
                kv_dim,
                max_seq_len,
                layer_limit,
            };

            // Upload per-layer weights: norms as f32, projections as BF16.
            let mut layer_weights_vec = Vec::with_capacity(config.num_hidden_layers);
            for (i, lw) in weights.layers.iter().enumerate() {
                layer_weights_vec.push(MetalLayerWeights {
                    // f32 projection weights (fallback)
                    q_proj_weight: make_buffer(
                        &device,
                        lw.q_proj_weight.data,
                        &format!("L{i}.q_proj"),
                    ),
                    k_proj_weight: make_buffer(
                        &device,
                        lw.k_proj_weight.data,
                        &format!("L{i}.k_proj"),
                    ),
                    v_proj_weight: make_buffer(
                        &device,
                        lw.v_proj_weight.data,
                        &format!("L{i}.v_proj"),
                    ),
                    o_proj_weight: make_buffer(
                        &device,
                        lw.o_proj_weight.data,
                        &format!("L{i}.o_proj"),
                    ),
                    gate_proj_weight: make_buffer(
                        &device,
                        lw.gate_proj_weight.data,
                        &format!("L{i}.gate"),
                    ),
                    up_proj_weight: make_buffer(
                        &device,
                        lw.up_proj_weight.data,
                        &format!("L{i}.up"),
                    ),
                    down_proj_weight: make_buffer(
                        &device,
                        lw.down_proj_weight.data,
                        &format!("L{i}.down"),
                    ),
                    // Norm weights f32
                    q_norm_weight: make_buffer(
                        &device,
                        lw.q_norm_weight.data,
                        &format!("L{i}.q_norm"),
                    ),
                    k_norm_weight: make_buffer(
                        &device,
                        lw.k_norm_weight.data,
                        &format!("L{i}.k_norm"),
                    ),
                    input_layernorm_weight: make_buffer(
                        &device,
                        lw.input_layernorm_weight.data,
                        &format!("L{i}.in_norm"),
                    ),
                    post_attention_layernorm_weight: make_buffer(
                        &device,
                        lw.post_attention_layernorm_weight.data,
                        &format!("L{i}.post_norm"),
                    ),
                });
            }

            let final_norm_weight = make_buffer(&device, weights.norm_weight.data, "final_norm");

            // Build and upload RoPE tables.
            let rope_max = max_seq_len.min(config.max_position_embeddings);
            let (cos_data, sin_data) =
                build_rope_flat(config.head_dim, rope_max, config.rope_theta);
            let rope_cos = make_buffer(&device, &cos_data, "rope_cos");
            let rope_sin = make_buffer(&device, &sin_data, "rope_sin");

            // Allocate activation buffers for max_seq_len.
            let s = max_seq_len;
            let activations = MetalActivations {
                hidden: make_zero_buffer(&device, s * config.hidden_size, "act_hidden"),
                residual: make_zero_buffer(&device, s * config.hidden_size, "act_residual"),
                q: make_zero_buffer(&device, s * q_dim, "act_q"),
                k: make_zero_buffer(&device, s * kv_dim, "act_k"),
                v: make_zero_buffer(&device, s * kv_dim, "act_v"),
                attn_out: make_zero_buffer(&device, s * q_dim, "act_attn_out"),
                gate: make_zero_buffer(&device, s * config.intermediate_size, "act_gate"),
                up: make_zero_buffer(&device, s * config.intermediate_size, "act_up"),
                ffn_out: make_zero_buffer(&device, s * config.hidden_size, "act_ffn_out"),
            };

            Ok(Self {
                device,
                queue,
                matmul_pipeline,
                rms_norm_pipeline,
                fused_attention_pipeline,
                fused_qk_norm_rope_pipeline,
                silu_mul_pipeline,
                copy_pipeline,
                add_pipeline,
                layer_weights: layer_weights_vec,
                final_norm_weight,
                rope_cos,
                rope_sin,
                activations,
                config: gpu_config,
            })
        }

        /// Run the forward pass on GPU.
        ///
        /// `hidden_input` is `[seq_len, hidden_size]` — the result of CPU embedding gather.
        /// Returns `[seq_len, hidden_size]` hidden states after all transformer layers + final norm.
        ///
        /// Takes `&mut self` because activation buffers are written via unified memory pointers.
        pub fn forward(
            &mut self,
            hidden_input: &[f32],
            seq_len: usize,
        ) -> Result<Vec<f32>, String> {
            let hidden = self.config.hidden_size;
            if seq_len == 0 || seq_len > self.config.max_seq_len {
                return Err(format!(
                    "seq_len {seq_len} out of range [1, {}]",
                    self.config.max_seq_len
                ));
            }
            if hidden_input.len() != seq_len * hidden {
                return Err(format!(
                    "hidden_input length {} != seq_len({seq_len}) * hidden_size({hidden})",
                    hidden_input.len()
                ));
            }

            let cfg = &self.config;
            let q_dim = cfg.q_dim as u32;
            let kv_dim = cfg.kv_dim as u32;
            let num_heads = cfg.num_attention_heads as u32;
            let num_kv_heads = cfg.num_key_value_heads as u32;
            let hidden_u = cfg.hidden_size as u32;
            let inter_u = cfg.intermediate_size as u32;
            let seq_u = seq_len as u32;
            let eps = cfg.rms_norm_eps;
            let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();

            let hidden_elems = (seq_len * hidden) as u32;
            let inter_elems = (seq_len * cfg.intermediate_size) as u32;

            // SAFETY: `self.activations.hidden` is a Metal buffer with StorageModeShared,
            // giving us a CPU-accessible pointer to GPU-visible unified memory. We have
            // exclusive access through `&mut self`, and the buffer was allocated with at
            // least `max_seq_len * hidden_size` f32s in `new()`.
            unsafe {
                let dst = self.activations.hidden.contents() as *mut f32;
                std::ptr::copy_nonoverlapping(hidden_input.as_ptr(), dst, hidden_input.len());
            }

            let cmd = self.queue.new_command_buffer();

            let layer_limit = cfg.layer_limit.unwrap_or(cfg.num_hidden_layers);

            for layer_idx in 0..layer_limit.min(cfg.num_hidden_layers) {
                let lw = &self.layer_weights[layer_idx];
                let enc = cmd.new_compute_command_encoder();

                // 1. Copy hidden -> residual
                self.dispatch_copy(
                    enc,
                    &self.activations.hidden,
                    &self.activations.residual,
                    hidden_elems,
                );

                // 2. Input LayerNorm (in-place on hidden)
                self.dispatch_rms_norm(
                    enc,
                    &self.activations.hidden,
                    &lw.input_layernorm_weight,
                    hidden_u,
                    seq_u,
                    eps,
                );

                // 3-5. Q/K/V projections
                self.dispatch_matmul(
                    enc,
                    &self.activations.hidden,
                    &lw.q_proj_weight,
                    &self.activations.q,
                    seq_u,
                    q_dim,
                    hidden_u,
                );
                self.dispatch_matmul(
                    enc,
                    &self.activations.hidden,
                    &lw.k_proj_weight,
                    &self.activations.k,
                    seq_u,
                    kv_dim,
                    hidden_u,
                );
                self.dispatch_matmul(
                    enc,
                    &self.activations.hidden,
                    &lw.v_proj_weight,
                    &self.activations.v,
                    seq_u,
                    kv_dim,
                    hidden_u,
                );
                // 6-9. Fused QK-norm + RoPE (4 dispatches → 1)
                self.dispatch_fused_qk_norm_rope(
                    enc,
                    seq_u,
                    num_heads,
                    num_kv_heads,
                    q_dim,
                    kv_dim,
                    eps,
                    &lw.q_norm_weight,
                    &lw.k_norm_weight,
                );

                // 10-12. Fused attention: Q@K^T + causal softmax + scores@V
                // Replaces 3 separate dispatches with 1 fused kernel.
                // Online softmax in registers, no global scores buffer.
                self.dispatch_fused_attention(enc, seq_u, q_dim, kv_dim, num_kv_heads, scale);

                // 13. O projection
                self.dispatch_matmul(
                    enc,
                    &self.activations.attn_out,
                    &lw.o_proj_weight,
                    &self.activations.hidden,
                    seq_u,
                    hidden_u,
                    q_dim,
                );

                // 14. Add residual: hidden += residual
                self.dispatch_add(
                    enc,
                    &self.activations.residual,
                    &self.activations.hidden,
                    hidden_elems,
                );

                // 15. Copy hidden -> residual (for FFN residual)
                self.dispatch_copy(
                    enc,
                    &self.activations.hidden,
                    &self.activations.residual,
                    hidden_elems,
                );

                // 16. Post-attention LayerNorm (in-place on hidden)
                self.dispatch_rms_norm(
                    enc,
                    &self.activations.hidden,
                    &lw.post_attention_layernorm_weight,
                    hidden_u,
                    seq_u,
                    eps,
                );

                // 17. Gate projection
                self.dispatch_matmul(
                    enc,
                    &self.activations.hidden,
                    &lw.gate_proj_weight,
                    &self.activations.gate,
                    seq_u,
                    inter_u,
                    hidden_u,
                );
                // 18. Up projection
                self.dispatch_matmul(
                    enc,
                    &self.activations.hidden,
                    &lw.up_proj_weight,
                    &self.activations.up,
                    seq_u,
                    inter_u,
                    hidden_u,
                );
                // 19. Fused SiLU(gate) * up -> gate
                self.dispatch_silu_mul(enc, inter_elems);
                // 20. Down projection
                self.dispatch_matmul(
                    enc,
                    &self.activations.gate,
                    &lw.down_proj_weight,
                    &self.activations.ffn_out,
                    seq_u,
                    hidden_u,
                    inter_u,
                );

                // 21. hidden = residual + ffn_out
                // Copy residual to hidden, then add ffn_out.
                self.dispatch_copy(
                    enc,
                    &self.activations.residual,
                    &self.activations.hidden,
                    hidden_elems,
                );
                self.dispatch_add(
                    enc,
                    &self.activations.ffn_out,
                    &self.activations.hidden,
                    hidden_elems,
                );

                enc.end_encoding();
            }

            // Final RMSNorm.
            {
                let enc = cmd.new_compute_command_encoder();
                self.dispatch_rms_norm(
                    enc,
                    &self.activations.hidden,
                    &self.final_norm_weight,
                    hidden_u,
                    seq_u,
                    eps,
                );
                enc.end_encoding();
            }

            // Submit and wait.
            cmd.commit();
            cmd.wait_until_completed();

            // SAFETY: Read back from the same unified memory buffer we wrote to above.
            // The GPU command buffer has completed (wait_until_completed), so the data
            // is coherent. Buffer size >= max_seq_len * hidden_size from allocation.
            let out_len = seq_len * hidden;
            let mut output = vec![0.0f32; out_len];
            // SAFETY: out_len is within the allocated hidden activation buffer and
            // output has the same number of f32 slots.
            unsafe {
                let src = self.activations.hidden.contents() as *const f32;
                std::ptr::copy_nonoverlapping(src, output.as_mut_ptr(), out_len);
            }
            Ok(output)
        }

        // -----------------------------------------------------------------------
        // Dispatch helpers
        // -----------------------------------------------------------------------

        fn dispatch_matmul(
            &self,
            enc: &ComputeCommandEncoderRef,
            a: &Buffer,
            b: &Buffer,
            c: &Buffer,
            m: u32,
            n: u32,
            k: u32,
        ) {
            // On Metal 4 (macOS 26+), the basic 16x16 tiled GEMM is 4.7x faster than
            // rect GEMM variants for small M. The rect kernels suffer from poor occupancy
            // on Metal 4's scheduler. Use matmul_bt for all shapes.
            enc.set_compute_pipeline_state(&self.matmul_pipeline);
            enc.set_buffer(0, Some(a), 0);
            enc.set_buffer(1, Some(b), 0);
            enc.set_buffer(2, Some(c), 0);
            enc.set_bytes(3, 4, &m as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
            enc.set_bytes(5, 4, &k as *const u32 as *const _);
            // Must use dispatch_thread_groups (not dispatch_threads) because
            // the tiled GEMM needs ALL threads in each threadgroup to fill the
            // shared tA/tB tiles. dispatch_threads creates non-uniform TGs at
            // edges, leaving tB partially uninitialized.
            let tile = 16u64;
            enc.dispatch_thread_groups(
                MTLSize::new((n as u64).div_ceil(tile), (m as u64).div_ceil(tile), 1),
                MTLSize::new(tile, tile, 1),
            );
        }

        /// Fused QK-norm + RoPE: replaces 4 dispatches (Q-norm, K-norm, RoPE-Q, RoPE-K) with 1.
        #[allow(clippy::too_many_arguments)]
        fn dispatch_fused_qk_norm_rope(
            &self,
            enc: &ComputeCommandEncoderRef,
            seq_len: u32,
            num_heads: u32,
            num_kv_heads: u32,
            q_dim: u32,
            kv_dim: u32,
            eps: f32,
            q_norm_weight: &Buffer,
            k_norm_weight: &Buffer,
        ) {
            let params = FusedQkNormRopeParams {
                seq_len,
                q_heads: num_heads,
                k_heads: num_kv_heads,
                q_stride: q_dim,
                k_stride: kv_dim,
                eps,
            };

            enc.set_compute_pipeline_state(&self.fused_qk_norm_rope_pipeline);
            enc.set_buffer(0, Some(&self.activations.q), 0);
            enc.set_buffer(1, Some(&self.activations.k), 0);
            enc.set_buffer(2, Some(q_norm_weight), 0);
            enc.set_buffer(3, Some(k_norm_weight), 0);
            enc.set_buffer(4, Some(&self.rope_cos), 0);
            enc.set_buffer(5, Some(&self.rope_sin), 0);
            enc.set_bytes(
                6,
                std::mem::size_of::<FusedQkNormRopeParams>() as u64,
                &params as *const FusedQkNormRopeParams as *const std::ffi::c_void,
            );

            // Grid: (seq_len, max(q_heads, k_heads), 2)
            // z=0 for Q, z=1 for K. Kernel guards against head >= head_count.
            let max_heads = num_heads.max(num_kv_heads);
            let grid = MTLSize::new(seq_len as u64, max_heads as u64, 2);
            let tg = MTLSize::new(64, 1, 1);
            enc.dispatch_thread_groups(grid, tg);
        }

        fn dispatch_rms_norm(
            &self,
            enc: &ComputeCommandEncoderRef,
            x: &Buffer,
            gamma: &Buffer,
            row_len: u32,
            num_rows: u32,
            eps: f32,
        ) {
            enc.set_compute_pipeline_state(&self.rms_norm_pipeline);
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(gamma), 0);
            enc.set_bytes(2, 4, &row_len as *const u32 as *const _);
            enc.set_bytes(3, 4, &num_rows as *const u32 as *const _);
            enc.set_bytes(4, 4, &eps as *const f32 as *const _);

            // One threadgroup per row. Shader uses threadgroup_position_in_grid
            // as the row index and thread_position_in_threadgroup for strided access.
            let wg = 256u64;
            let grid = MTLSize::new(num_rows as u64, 1, 1);
            let tg = MTLSize::new(wg, 1, 1);
            enc.dispatch_thread_groups(grid, tg);
        }

        /// Fused attention: Q@K^T + causal softmax + scores@V in one kernel.
        /// Eliminates the global scores buffer. Online softmax in registers.
        fn dispatch_fused_attention(
            &self,
            enc: &ComputeCommandEncoderRef,
            seq_len: u32,
            q_dim: u32,
            kv_dim: u32,
            num_kv_heads: u32,
            scale: f32,
        ) {
            if seq_len == 0 {
                return;
            }

            #[repr(C)]
            struct Params {
                seq_len: u32,
                q_dim4: u32,
                kv_dim4: u32,
                num_kv_heads: u32,
                scale: f32,
                _pad0: u32,
                _pad1: u32,
                _pad2: u32,
            }

            let params = Params {
                seq_len,
                q_dim4: q_dim / 4,
                kv_dim4: kv_dim / 4,
                num_kv_heads,
                scale,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };

            const TILE_Q: u32 = 4;
            let gqa_groups: u32 =
                (self.config.num_attention_heads / self.config.num_key_value_heads) as u32;
            const SIMD_WIDTH: u32 = 32;
            let threads_per_tg: u64 = (TILE_Q * gqa_groups * SIMD_WIDTH) as u64;

            let q_blocks = div_ceil(seq_len as u64, TILE_Q as u64);

            enc.set_compute_pipeline_state(&self.fused_attention_pipeline);
            enc.set_buffer(0, Some(&self.activations.q), 0);
            enc.set_buffer(1, Some(&self.activations.k), 0);
            enc.set_buffer(2, Some(&self.activations.v), 0);
            enc.set_buffer(3, Some(&self.activations.attn_out), 0);
            enc.set_bytes(
                4,
                std::mem::size_of::<Params>() as u64,
                &params as *const Params as *const _,
            );

            // 2D grid: x = kv_head, y = query block
            let tg_count = MTLSize::new(num_kv_heads as u64, q_blocks, 1);
            let tg_size = MTLSize::new(threads_per_tg, 1, 1);
            enc.dispatch_thread_groups(tg_count, tg_size);
        }

        fn dispatch_silu_mul(&self, enc: &ComputeCommandEncoderRef, count: u32) {
            enc.set_compute_pipeline_state(&self.silu_mul_pipeline);
            enc.set_buffer(0, Some(&self.activations.gate), 0);
            enc.set_buffer(1, Some(&self.activations.up), 0);
            enc.set_bytes(2, 4, &count as *const u32 as *const _);

            let wg = 256u64;
            let grid = MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1);
            let tg = MTLSize::new(wg, 1, 1);
            enc.dispatch_threads(grid, tg);
        }

        fn dispatch_copy(
            &self,
            enc: &ComputeCommandEncoderRef,
            src: &Buffer,
            dst: &Buffer,
            count: u32,
        ) {
            enc.set_compute_pipeline_state(&self.copy_pipeline);
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst), 0);
            enc.set_bytes(2, 4, &count as *const u32 as *const _);

            let wg = 256u64;
            let grid = MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1);
            let tg = MTLSize::new(wg, 1, 1);
            enc.dispatch_threads(grid, tg);
        }

        fn dispatch_add(
            &self,
            enc: &ComputeCommandEncoderRef,
            src: &Buffer,
            dst: &Buffer,
            count: u32,
        ) {
            enc.set_compute_pipeline_state(&self.add_pipeline);
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst), 0);
            enc.set_bytes(2, 4, &count as *const u32 as *const _);

            let wg = 256u64;
            let grid = MTLSize::new(div_ceil(count as u64, wg) * wg, 1, 1);
            let tg = MTLSize::new(wg, 1, 1);
            enc.dispatch_threads(grid, tg);
        }
    }

    /// Build flat cos/sin RoPE tables: [max_seq_len * half_dim] each.
    fn build_rope_flat(head_dim: usize, max_seq_len: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
        let half_dim = head_dim / 2;
        let mut cos = Vec::with_capacity(max_seq_len * half_dim);
        let mut sin = Vec::with_capacity(max_seq_len * half_dim);

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
                let angle = pos as f64 * freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }

        (cos, sin)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::weights::{QwenLayerWeights, Tensor1D, Tensor2D};

        /// ADR-066 D3.1 / ADR-080 C1 (PR #794): a capability-gated
        /// Metal test must report a distinct skip or fail closed when
        /// `LATTICE_METAL_TEST_ENFORCE=1` is set, never silently early-return `ok` —
        /// otherwise a runner that loses its Metal device (or never had one) greens every
        /// gate that depends on this module without ever executing the Metal path.
        /// Mirrors the `forward::metal_qwen35` enforce convention already used by the Q4
        /// and grammar fail-closed CI steps.
        fn metal_test_device(context: &str) -> Option<Device> {
            match Device::system_default() {
                Some(device) => Some(device),
                None => {
                    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
                    assert!(
                        !enforce,
                        "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present ({context})"
                    );
                    eprintln!("[METAL_TEST_SKIP] context={context} reason=no_metal_device");
                    None
                }
            }
        }

        /// Guards the `flash_attention.metal` extraction: the file loaded via
        /// `include_str!` and run through `msl_source_for`'s token substitution must
        /// still produce a complete, compilable shader library. Mutation-sensitive —
        /// corrupting the `.metal` file or dropping a `.replace()` fails this test.
        #[test]
        fn msl_template_assembles_and_compiles() {
            // Representative Qwen3-Embedding dims (head_dim=128, gqa_groups=2).
            let msl = msl_source_for(128, 2);
            // Every injected placeholder must be substituted — no token may leak into MSL.
            assert!(
                !msl.contains("__"),
                "unsubstituted placeholder token remains in assembled MSL"
            );
            // Compile the assembled shader and confirm every pipeline function resolves.
            // Fails closed under LATTICE_METAL_TEST_ENFORCE=1 if no Metal device is present
            // (see `metal_test_device`); otherwise reports a distinct skip.
            let Some(device) = metal_test_device("msl_template_assembles_and_compiles") else {
                return;
            };
            let opts = CompileOptions::new();
            let library = device
                .new_library_with_source(&msl, &opts)
                .expect("assembled flash_attention.metal must compile");
            for name in [
                "matmul_bt",
                "rms_norm",
                "fused_attention",
                "fused_qk_norm_rope",
                "silu_mul",
                "copy_buf",
                "add_buf",
            ] {
                library
                    .get_function(name, None)
                    .unwrap_or_else(|e| panic!("kernel '{name}' missing from library: {e}"));
            }
            // ADR-080 C1 (#791): attn_scores/attn_softmax/attn_context were dead
            // code (no `make_pipeline` call referenced them, and the dead
            // `attn_softmax` carried the same multiply-through-zero fail-open
            // defect as #789) and were deleted from the shader source. Guard
            // against silently resurrecting them.
            for dead_name in ["attn_scores", "attn_softmax", "attn_context"] {
                assert!(
                    library.get_function(dead_name, None).is_err(),
                    "'{dead_name}' should have been deleted as dead code (ADR-080 C1, #791) \
                     but is present in the assembled MSL"
                );
            }
        }

        /// ADR-080 C1 (#789) regression fixture: corrupts a single entry of
        /// `q_proj_weight` with `corrupt_value` so one attention head's Q vector
        /// carries that value at every query position (K/V/embeddings/residual stay
        /// completely clean), then runs the live `MetalForwardPass::forward` boundary
        /// and returns its output. Shared by the NaN and +inf invalid-row regressions
        /// below (ADR-066 D2 invariant #1's non-finite-row classes) so both exercise
        /// the identical construction and the identical finalize fix.
        ///
        /// Returns `None` only when there is no Metal device and
        /// `LATTICE_METAL_TEST_ENFORCE` is unset (see `metal_test_device`); with the
        /// enforce var set, a missing device or failed pass construction panics
        /// instead of skipping.
        fn run_fused_attention_with_corrupted_q_lane(
            context: &str,
            corrupt_value: f32,
        ) -> Option<Vec<f32>> {
            let _lock = gpu_test_lock();
            let _device_probe = metal_test_device(context)?;

            let config = QwenConfig {
                vocab_size: 8,
                hidden_size: 32,
                num_hidden_layers: 1,
                num_attention_heads: 4,
                num_key_value_heads: 2,
                head_dim: 8,
                intermediate_size: 48,
                max_position_embeddings: 32,
                rms_norm_eps: 1e-6,
                rope_theta: 1_000_000.0,
            };
            let hidden = config.hidden_size;
            let q_dim = config.q_dim();
            let kv_dim = config.kv_dim();
            let intermediate = config.intermediate_size;

            let mut rng = TestLcg::new(0x5EED_BAAD_F00Du64);
            let embed_tokens_flat = test_random_vec(&mut rng, config.vocab_size * hidden);
            let norm_weight_flat = test_random_positive_vec(&mut rng, hidden);

            // Corrupt one entry of q_proj_weight so a single attention head's
            // Q vector is NaN at every query position, while K/V/embeddings/
            // residual stay completely clean -- isolates the softmax finalize
            // as the only place the NaN could be cured.
            let corrupt_head: usize = 1;
            let corrupt_head_dim: usize = 3;
            let corrupt_out_idx = corrupt_head * config.head_dim + corrupt_head_dim;
            let corrupt_in_idx = 5usize;

            let mut q_proj_flat = test_random_vec(&mut rng, q_dim * hidden);
            q_proj_flat[corrupt_out_idx * hidden + corrupt_in_idx] = corrupt_value;
            let k_proj_flat = test_random_vec(&mut rng, kv_dim * hidden);
            let v_proj_flat = test_random_vec(&mut rng, kv_dim * hidden);
            let o_proj_flat = test_random_vec(&mut rng, hidden * q_dim);
            let q_norm_flat = test_random_positive_vec(&mut rng, config.head_dim);
            let k_norm_flat = test_random_positive_vec(&mut rng, config.head_dim);
            let input_ln_flat = test_random_positive_vec(&mut rng, hidden);
            let gate_proj_flat = test_random_vec(&mut rng, intermediate * hidden);
            let up_proj_flat = test_random_vec(&mut rng, intermediate * hidden);
            let down_proj_flat = test_random_vec(&mut rng, hidden * intermediate);
            let post_ln_flat = test_random_positive_vec(&mut rng, hidden);

            let layer = QwenLayerWeights {
                q_proj_weight: Tensor2D {
                    data: &q_proj_flat,
                    rows: q_dim,
                    cols: hidden,
                },
                k_proj_weight: Tensor2D {
                    data: &k_proj_flat,
                    rows: kv_dim,
                    cols: hidden,
                },
                v_proj_weight: Tensor2D {
                    data: &v_proj_flat,
                    rows: kv_dim,
                    cols: hidden,
                },
                o_proj_weight: Tensor2D {
                    data: &o_proj_flat,
                    rows: hidden,
                    cols: q_dim,
                },
                q_norm_weight: Tensor1D {
                    data: &q_norm_flat,
                    len: config.head_dim,
                },
                k_norm_weight: Tensor1D {
                    data: &k_norm_flat,
                    len: config.head_dim,
                },
                input_layernorm_weight: Tensor1D {
                    data: &input_ln_flat,
                    len: hidden,
                },
                gate_proj_weight: Tensor2D {
                    data: &gate_proj_flat,
                    rows: intermediate,
                    cols: hidden,
                },
                up_proj_weight: Tensor2D {
                    data: &up_proj_flat,
                    rows: intermediate,
                    cols: hidden,
                },
                down_proj_weight: Tensor2D {
                    data: &down_proj_flat,
                    rows: hidden,
                    cols: intermediate,
                },
                post_attention_layernorm_weight: Tensor1D {
                    data: &post_ln_flat,
                    len: hidden,
                },
                fused_qkv: vec![0.0f32; (q_dim + 2 * kv_dim) * hidden],
                qkv_out_dim: q_dim + 2 * kv_dim,
                fused_gate_up: vec![0.0f32; (2 * intermediate) * hidden],
                gate_up_out_dim: 2 * intermediate,
            };
            let weights = QwenWeights {
                embed_tokens: Tensor2D {
                    data: &embed_tokens_flat,
                    rows: config.vocab_size,
                    cols: hidden,
                },
                norm_weight: Tensor1D {
                    data: &norm_weight_flat,
                    len: hidden,
                },
                layers: vec![layer],
            };

            let input_ids: Vec<u32> = vec![1, 3, 2, 5, 6];
            let seq_len = input_ids.len();

            let mut hidden_input = vec![0.0f32; seq_len * hidden];
            for (i, &tok) in input_ids.iter().enumerate() {
                let tok = tok as usize;
                hidden_input[i * hidden..(i + 1) * hidden]
                    .copy_from_slice(&embed_tokens_flat[tok * hidden..(tok + 1) * hidden]);
            }

            let pass = MetalForwardPass::new(&config, &weights, 8);
            let mut pass = match pass {
                Ok(pass) => pass,
                Err(err) => {
                    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
                    assert!(
                        !enforce,
                        "LATTICE_METAL_TEST_ENFORCE=1 but MetalForwardPass::new failed \
                         ({context}): {err}"
                    );
                    eprintln!(
                        "[METAL_TEST_SKIP] context={context} reason=forward_pass_construct_failed \
                         error={err}"
                    );
                    return None;
                }
            };

            let metal_out = pass
                .forward(&hidden_input, seq_len)
                .expect("Metal forward should not itself error");

            Some(metal_out)
        }

        /// ADR-080 C1 (#789) regression, NaN class: the fused_attention finalize must
        /// fail closed by ASSIGNMENT when the online-softmax denominator is
        /// non-positive or non-finite (here: a NaN score from a corrupted Q lane), not
        /// by multiplying the (possibly already-NaN) numerator through a zeroed
        /// reciprocal -- `NaN * 0.0f == NaN` under IEEE-754, so a `* inv_l` finalize
        /// cannot recover a poisoned row.
        ///
        /// Mutation-sensitive: reverting the fix in flash_attention.metal back to
        /// `O4[out_base4] = o_frag * inv_l;` makes this test fail (160/160 NaN
        /// elements instead of 0), confirmed at PR time (round 1) and reconfirmed
        /// after the fix-round refactor (round 2).
        #[test]
        fn fused_attention_fails_closed_on_nan_q_lane() {
            let Some(metal_out) = run_fused_attention_with_corrupted_q_lane(
                "fused_attention_fails_closed_on_nan_q_lane",
                f32::NAN,
            ) else {
                return;
            };

            let nan_count = metal_out.iter().filter(|v| v.is_nan()).count();
            assert_eq!(
                nan_count,
                0,
                "fused_attention must fail closed (zero output) on a NaN-poisoned \
                 Q lane instead of propagating NaN through the finalize multiply \
                 (ADR-080 C1, #789); got {nan_count}/{} NaN elements",
                metal_out.len()
            );
        }

        /// ADR-080 C1 / ADR-066 D2 invariant #1, +inf class: the same corrupted-Q-lane
        /// construction as the NaN test above, but with the poisoned weight entry set
        /// to `f32::INFINITY` instead of `f32::NAN`. Because only one weight entry is
        /// corrupted and the rest of the accumulation over `hidden` dims is finite, the
        /// resulting Q component for that head goes to +inf or -inf depending on the
        /// sign of the corresponding hidden-input activation at each query position --
        /// both are covered by the `!is_finite()` check below. This reaches the
        /// finalize's non-finite-denominator branch through a distinct arithmetic path
        /// from the NaN test (an infinite accumulator/denominator rather than a NaN
        /// one), so it is not redundant with it.
        ///
        /// Mutation-sensitive by the same finalize revert as the NaN test above (same
        /// shared shader guard).
        #[test]
        fn fused_attention_fails_closed_on_inf_q_lane() {
            let Some(metal_out) = run_fused_attention_with_corrupted_q_lane(
                "fused_attention_fails_closed_on_inf_q_lane",
                f32::INFINITY,
            ) else {
                return;
            };

            let non_finite_count = metal_out.iter().filter(|v| !v.is_finite()).count();
            assert_eq!(
                non_finite_count,
                0,
                "fused_attention must fail closed (zero output) on an inf-poisoned \
                 Q lane instead of propagating inf/NaN through the finalize multiply \
                 (ADR-080 C1, #789; ADR-066 D2 invariant #1); got {non_finite_count}/{} \
                 non-finite elements",
                metal_out.len()
            );
        }

        // ADR-066 D2 invariant #1, all-(-inf) class: NOT covered by a direct
        // kernel-dispatch fixture here, by design, not oversight (PR #794 round-1
        // review, PR #794). `fused_attention` takes no external mask buffer -- the
        // only way a score row goes to all -inf is every Q.K dot product in that row
        // underflowing to -inf, and the live kernel always includes the causal
        // self-key (Q_i . K_i), which stays finite for any finite Q/K pair. Forcing a
        // genuine all-(-inf) row would require injecting -inf directly into the score
        // buffer, i.e. bypassing the live entry point this suite is scoped to (the
        // corrupted-weight construction above always produces a *finite* Q/K pair
        // whose dot product can go to NaN or +-inf, never a row of literal -inf by
        // construction). The finalize guard's arithmetic is the same for this class as
        // for the +inf class above (`l_i` non-positive or non-finite triggers the same
        // zero-assignment branch), so the +inf regression already exercises the shared
        // fail-closed code path; there is no code path reachable from a live forward
        // call that this suite could exercise differently for all-(-inf) specifically.

        struct TestLcg(u64);

        impl TestLcg {
            fn new(seed: u64) -> Self {
                Self(seed)
            }
            fn next_u32(&mut self) -> u32 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                (self.0 >> 32) as u32
            }
            fn next_f32(&mut self) -> f32 {
                let x = self.next_u32() as f32 / u32::MAX as f32;
                (x - 0.5) * 0.2
            }
        }

        fn test_random_vec(rng: &mut TestLcg, len: usize) -> Vec<f32> {
            (0..len).map(|_| rng.next_f32()).collect()
        }

        fn test_random_positive_vec(rng: &mut TestLcg, len: usize) -> Vec<f32> {
            (0..len).map(|_| 0.5 + rng.next_f32().abs()).collect()
        }

        /// Serializes GPU-driving tests onto the single shared Metal device,
        /// in-process and machine-wide (mirrors `forward::metal_qwen35`'s
        /// `gpu_test_lock`; the in-process `Mutex` here is a separate instance
        /// but the machine-wide `flock` on the same fixed path still
        /// serializes correctly across both, since `flock` mutual exclusion
        /// is per-path, not per-`Mutex`-instance).
        struct GpuTestGuard {
            _process: std::sync::MutexGuard<'static, ()>,
            _machine: std::fs::File,
        }

        fn gpu_test_lock() -> GpuTestGuard {
            use std::sync::Mutex;
            static GPU_LOCK: Mutex<()> = Mutex::new(());
            let process = GPU_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);

            const LOCK_PATH: &str = "/tmp/lion-metal-gpu-test.lock";
            const LOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30 * 60);
            let file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(false)
                .open(LOCK_PATH)
                .unwrap_or_else(|e| panic!("gpu_test_lock: cannot open {LOCK_PATH}: {e}"));
            let deadline = std::time::Instant::now() + LOCK_TIMEOUT;
            loop {
                match file.try_lock() {
                    Ok(()) => break,
                    Err(std::fs::TryLockError::WouldBlock) => {
                        if std::time::Instant::now() >= deadline {
                            panic!(
                                "gpu_test_lock: another process has held {LOCK_PATH} for over \
                                 {}s -- inspect `lsof {LOCK_PATH}`",
                                LOCK_TIMEOUT.as_secs()
                            );
                        }
                        std::thread::sleep(std::time::Duration::from_millis(500));
                    }
                    Err(std::fs::TryLockError::Error(e)) => {
                        panic!("gpu_test_lock: flock on {LOCK_PATH} failed: {e}")
                    }
                }
            }
            GpuTestGuard {
                _process: process,
                _machine: file,
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal-gpu"))]
pub use inner::MetalForwardPass;

// Stub for non-macOS or non-metal builds.
/// **Unstable**: Metal GPU forward pass; stub on non-macOS; full impl behind metal-gpu feature.
#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
pub struct MetalForwardPass;

#[cfg(not(all(target_os = "macos", feature = "metal-gpu")))]
impl MetalForwardPass {
    /// **Unstable**: construct Metal forward pass; always fails without metal-gpu feature.
    pub fn new(
        _config: &crate::model::qwen::QwenConfig,
        _weights: &crate::weights::QwenWeights<'_>,
        _max_seq_len: usize,
    ) -> Result<Self, String> {
        Err("Metal GPU not available (requires macOS + metal-gpu feature)".into())
    }

    /// **Unstable**: run Metal forward pass; always fails without metal-gpu feature.
    pub fn forward(&mut self, _hidden_input: &[f32], _seq_len: usize) -> Result<Vec<f32>, String> {
        Err("Metal GPU not available".into())
    }
}
