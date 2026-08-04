use super::super::*;

impl MetalQwen35State {
    /// Encode all Metal commands for a single GDN (linear-attention) layer.
    ///
    /// All commands are appended to the already-open encoder `enc`; no new
    /// command buffer is created here.  Returns `1` on every call (one GPU
    /// recurrence dispatch issued per GDN layer).
    pub(in super::super) fn encode_gdn_layer(
        &mut self,
        enc: &ComputeCommandEncoderRef,
        compact_idx: usize,
        linear_idx: usize,
        position: usize,
        layer_idx: usize,
        cfg: &Qwen35Config,
        prof: &mut StepProfile,
        profiling: bool,
        with_mlp: bool,
    ) -> usize {
        let hidden = cfg.hidden_size;
        let (
            w_in_proj_qkvz,
            w_in_proj_b,
            w_in_proj_a,
            w_a_log,
            w_dt_bias,
            w_conv1d,
            w_norm,
            w_out_proj,
        ) = {
            let MetalLayerAttnWeights::Linear(gdn_w) = &self.engine.layer_weights[compact_idx].0
            else {
                unreachable!()
            };
            (
                &gdn_w.in_proj_qkvz as *const Q4WeightBuf,
                &gdn_w.in_proj_b as *const Buffer,
                &gdn_w.in_proj_a as *const Buffer,
                &gdn_w.a_log as *const Buffer,
                &gdn_w.dt_bias as *const Buffer,
                &gdn_w.conv1d_weight as *const Buffer,
                &gdn_w.norm_weight as *const Buffer,
                &gdn_w.out_proj as *const Q4WeightBuf,
            )
        };
        let (_, common_w) = &self.engine.layer_weights[compact_idx];
        let qkv_d = cfg.linear_qkv_dim();
        let out_d = cfg.linear_output_dim();
        let ks = cfg.linear_conv_kernel_dim as u32;
        let num_h = cfg.linear_num_key_heads;
        let key_d = cfg.linear_key_head_dim;
        let val_d = cfg.linear_value_head_dim;

        // Pre-norm: save residual + normalize hidden in one dispatch
        self.dispatch_copy_and_rms_norm(
            enc,
            &self.session.activations.hidden,
            &self.session.activations.residual,
            &common_w.input_layernorm,
            hidden as u32,
            cfg.rms_norm_eps,
        );

        // QKV + Z projections
        // SAFETY: Raw projection buffer pointers were taken from
        // self.engine.layer_weights and remain valid while self is borrowed;
        // H4 fusion: one wider GEMV writes QKV||Z rows to gdn_qkvz (saves 1 dispatch/GDN layer).
        // SAFETY: w_in_proj_qkvz holds contiguous Q4 rows [0..qkv_d) from qkv and
        // [qkv_d..qkv_d+out_d) from z; output gdn_qkvz is (qkv_d+out_d) floats.
        let proj_t0 = profiling.then(std::time::Instant::now);
        unsafe {
            self.dispatch_matmul(
                enc,
                &self.session.activations.hidden,
                &*w_in_proj_qkvz,
                &self.session.activations.gdn_qkvz,
                1,
                (qkv_d + out_d) as u32,
                hidden as u32,
            );
        }
        if let Some(t0) = proj_t0 {
            prof.projection_us += t0.elapsed().as_micros();
        }

        // LoRA for in_proj_qkv: accumulates into QKV portion (offset 0) of gdn_qkvz.
        self.dispatch_lora_if_active(
            enc,
            &self.session.activations.hidden,
            0,
            &self.session.activations.gdn_qkvz,
            0,
            layer_idx,
            "in_proj_qkv",
        );
        // LoRA for in_proj_z: accumulates into Z portion (byte offset qkv_d * 4) of gdn_qkvz.
        self.dispatch_lora_if_active(
            enc,
            &self.session.activations.hidden,
            0,
            &self.session.activations.gdn_qkvz,
            (qkv_d as u64) * 4,
            layer_idx,
            "in_proj_z",
        );
        // in_proj_b and in_proj_a: rejected at load time (consumed inside fused kernels).

        // Conv1d + SiLU (GPU)
        // SAFETY: Encoder commands reference live Metal buffers owned by
        // self/layer weights; qkv_d and kernel_size match activation and
        // weight buffer dimensions allocated during initialization.
        unsafe {
            enc.set_compute_pipeline_state(&self.engine.pipelines.conv1d_silu);
            enc.set_buffer(0, Some(&self.session.gdn_gpu_conv_bufs[linear_idx]), 0);
            enc.set_buffer(1, Some(&self.session.activations.gdn_qkvz), 0); // QKV at offset 0 of fused buffer
            enc.set_buffer(2, Some(&*w_conv1d), 0);
            enc.set_buffer(3, Some(&self.session.gdn_gpu_conv_out), 0);
            let qd = qkv_d as u32;
            enc.set_bytes(4, 4, &qd as *const u32 as *const _);
            enc.set_bytes(5, 4, &ks as *const u32 as *const _);
            let wg = 256u64;
            enc.dispatch_threads(
                MTLSize::new(div_ceil(qkv_d as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
        }

        // GDN recurrence (GPU) — one threadgroup per value head
        // SAFETY: All buffers are StorageModeShared and live for the
        // command buffer; dimensions in GdnRecurParams are derived from
        // the model config used to allocate the GDN state buffers.
        let gdn_t0 = profiling.then(std::time::Instant::now);
        unsafe {
            #[repr(C)]
            struct GdnRecurParams {
                key_dim: u32,
                value_dim: u32,
                num_key_heads: u32,
                num_value_heads: u32,
                hidden_size: u32,
                q_total: u32,
                v_offset: u32,
                scale: f32,
                eps: f32,
            }
            let num_vh = cfg.linear_num_value_heads();
            let q_total = (num_h * key_d) as u32;
            let params = GdnRecurParams {
                key_dim: key_d as u32,
                value_dim: val_d as u32,
                num_key_heads: num_h as u32,
                num_value_heads: num_vh as u32,
                hidden_size: hidden as u32,
                q_total,
                v_offset: q_total * 2,
                scale: 1.0 / (key_d as f32).sqrt(),
                eps: cfg.rms_norm_eps,
            };
            let z_byte_off = (qkv_d as u64) * 4; // byte offset to Z portion in gdn_qkvz
            let use_q36 =
                key_d == 128 && val_d == 128 && hidden == 5120 && num_h == 16 && num_vh == 48;
            // Chunked GDN prefill is prefill-only; single-token decode always uses serial path.
            if self.use_gdn_chunked && !use_q36 {
                tracing::debug!(
                    "chunked GDN prefill is prefill-only; encode_gdn_layer keeps serial recurrence"
                );
            }
            // H1+H3 three-kernel path: use when all three sharded kernels compiled.
            let use_h1h3 = use_q36
                && self.engine.pipelines.gdn_precompute_keys.is_some()
                && self.engine.pipelines.gdn_recurrence_sharded.is_some()
                && self.engine.pipelines.gdn_norm_silu.is_some();
            if use_h1h3 {
                let p_bytes = std::mem::size_of::<GdnRecurParams>() as u64;
                let p_ptr = &params as *const GdnRecurParams as *const _;

                // H3: precompute per-value-head decay/beta/g + per-key-head Q/K norms — num_vh TGs × 128 threads
                enc.set_compute_pipeline_state(
                    self.engine.pipelines.gdn_precompute_keys.as_ref().unwrap(),
                );
                enc.set_buffer(0, Some(&self.session.gdn_gpu_conv_out), 0);
                enc.set_buffer(1, Some(&self.session.activations.hidden), 0);
                enc.set_buffer(2, Some(&*w_in_proj_b), 0);
                enc.set_buffer(3, Some(&*w_in_proj_a), 0);
                enc.set_buffer(4, Some(&*w_a_log), 0);
                enc.set_buffer(5, Some(&*w_dt_bias), 0);
                enc.set_buffer(6, Some(&self.session.activations.gdn_key_scratch), 0);
                enc.set_bytes(7, p_bytes, p_ptr);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_vh as u64, 1, 1),
                    MTLSize::new(128, 1, 1),
                );

                // H1: sharded recurrence — (val_d/4) × num_vh TGs, 32×4 threads
                enc.set_compute_pipeline_state(
                    self.engine
                        .pipelines
                        .gdn_recurrence_sharded
                        .as_ref()
                        .unwrap(),
                );
                enc.set_buffer(0, Some(&self.session.gdn_gpu_s_matrices[linear_idx]), 0);
                enc.set_buffer(1, Some(&self.session.gdn_gpu_conv_out), 0);
                enc.set_buffer(2, Some(&self.session.activations.gdn_key_scratch), 0);
                enc.set_buffer(3, Some(&self.session.activations.gdn_raw_out), 0);
                enc.set_bytes(4, p_bytes, p_ptr);
                enc.dispatch_thread_groups(
                    MTLSize::new((val_d as u64).div_ceil(4), num_vh as u64, 1),
                    MTLSize::new(32, 4, 1),
                );

                // Norm+SiLU: one TG per value head — 48 TGs × 128 threads
                enc.set_compute_pipeline_state(
                    self.engine.pipelines.gdn_norm_silu.as_ref().unwrap(),
                );
                enc.set_buffer(0, Some(&self.session.activations.gdn_raw_out), 0);
                enc.set_buffer(1, Some(&self.session.activations.gdn_qkvz), z_byte_off);
                enc.set_buffer(2, Some(&*w_norm), 0);
                enc.set_buffer(3, Some(&self.session.activations.gdn_qkvz), z_byte_off);
                enc.set_bytes(4, p_bytes, p_ptr);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_vh as u64, 1, 1),
                    MTLSize::new(128, 1, 1),
                );
            } else {
                // Fallback: H2+H4 fused kernel or generic
                let recur_pipe = use_q36
                    .then_some(self.engine.pipelines.gdn_recurrence_q36.as_ref())
                    .flatten()
                    .unwrap_or(&self.engine.pipelines.gdn_recurrence);
                enc.set_compute_pipeline_state(recur_pipe);
                enc.set_buffer(0, Some(&self.session.gdn_gpu_s_matrices[linear_idx]), 0);
                enc.set_buffer(1, Some(&self.session.gdn_gpu_conv_out), 0);
                enc.set_buffer(2, Some(&self.session.activations.gdn_qkvz), z_byte_off);
                enc.set_buffer(3, Some(&self.session.activations.hidden), 0);
                enc.set_buffer(4, Some(&*w_in_proj_b), 0);
                enc.set_buffer(5, Some(&*w_in_proj_a), 0);
                enc.set_buffer(6, Some(&*w_a_log), 0);
                enc.set_buffer(7, Some(&*w_dt_bias), 0);
                enc.set_buffer(8, Some(&*w_norm), 0);
                enc.set_buffer(9, Some(&self.session.activations.gdn_qkvz), z_byte_off);
                enc.set_bytes(
                    10,
                    std::mem::size_of::<GdnRecurParams>() as u64,
                    &params as *const GdnRecurParams as *const _,
                );
                enc.dispatch_thread_groups(
                    MTLSize::new(num_vh as u64, 1, 1),
                    MTLSize::new(128, 1, 1),
                );
            }
        }
        if let Some(t0) = gdn_t0 {
            prof.gdn_recurrence_us += t0.elapsed().as_micros();
        }

        // Output projection — reads norm output from Z portion of fused buffer.
        // SAFETY: w_out_proj and gdn_qkvz are live for this command buffer;
        // z_byte_off points to out_d norm values written by gdn_recurrence.
        let out_proj_t0 = profiling.then(std::time::Instant::now);
        unsafe {
            self.dispatch_gemm(
                enc,
                &self.session.activations.gdn_qkvz,
                (qkv_d as u64) * 4,
                &*w_out_proj,
                &self.session.activations.attn_out,
                0,
                1,
                hidden as u32,
                out_d as u32,
            );
        }
        if let Some(t0) = out_proj_t0 {
            prof.projection_us += t0.elapsed().as_micros();
        }

        // LoRA for out_proj: x is the Z-normed portion of gdn_qkvz (at qkv_d * 4 bytes),
        // y is attn_out (offset 0).
        self.dispatch_lora_if_active(
            enc,
            &self.session.activations.gdn_qkvz,
            (qkv_d as u64) * 4,
            &self.session.activations.attn_out,
            0,
            layer_idx,
            "out_proj",
        );

        if with_mlp {
            self.encode_mlp_block(enc, compact_idx, layer_idx, position, cfg, prof, profiling);
        }

        1 // one GDN GPU dispatch per layer
    }
}
