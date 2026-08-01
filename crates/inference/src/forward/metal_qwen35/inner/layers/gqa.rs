use super::super::*;

impl MetalQwen35State {
    /// Encode all Metal commands for a single GQA (full-attention) layer.
    ///
    /// All commands are appended to the already-open encoder `enc`; no new
    /// command buffer is created here.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn encode_gqa_layer(
        &mut self,
        enc: &ComputeCommandEncoderRef,
        compact_idx: usize,
        full_idx: usize,
        position: usize,
        kv_dim: usize,
        layer_idx: usize,
        cfg: &Qwen35Config,
        prof: &mut StepProfile,
        profiling: bool,
        with_mlp: bool,
        mrope_override: Option<(&Buffer, &Buffer)>,
    ) {
        let hidden = cfg.hidden_size;
        // GQA: SINGLE command buffer — pre-norm + projections + KV copy + attention + MLP
        let (w_q_proj, w_k_proj, w_v_proj, w_o_proj, w_q_norm, w_k_norm) = {
            let MetalLayerAttnWeights::Full(full_w) = &self.engine.layer_weights[compact_idx].0
            else {
                unreachable!()
            };
            (
                &full_w.q_proj as *const Q4WeightBuf,
                &full_w.k_proj as *const Q4WeightBuf,
                &full_w.v_proj as *const Q4WeightBuf,
                &full_w.o_proj as *const Q4WeightBuf,
                &full_w.q_norm as *const Buffer,
                &full_w.k_norm as *const Buffer,
            )
        };
        let (_, common_w) = &self.engine.layer_weights[compact_idx];
        let q_dim = cfg.full_q_dim();
        let head_dim = cfg.head_dim;
        let num_q_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let half_rope_dim = (cfg.rope_dim() / 2) as u32;
        assert!(
            self.session.kv_cache.seq_len < self.session.kv_cache.max_cache_len,
            "KV cache overflow: seq_len {} >= max_cache_len {}",
            self.session.kv_cache.seq_len,
            self.session.kv_cache.max_cache_len
        );
        let kv_cache_offset = (self.session.kv_cache.seq_len * kv_dim) as u32;
        let cur_seq_len = (self.session.kv_cache.seq_len + 1) as u32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Pre-norm: save residual + normalize hidden in one dispatch
        self.dispatch_copy_and_rms_norm(
            enc,
            &self.session.activations.hidden,
            &self.session.activations.residual,
            &common_w.input_layernorm,
            hidden as u32,
            cfg.rms_norm_eps,
        );

        // Q/K/V projections + scatter + norms + RoPE
        // SAFETY: Raw layer weight pointers were taken from
        // self.engine.layer_weights and remain valid while self is borrowed;
        // dispatch dimensions match the preallocated activation buffers.
        let gqa_proj_t0 = profiling.then(std::time::Instant::now);
        unsafe {
            self.dispatch_matmul(
                enc,
                &self.session.activations.hidden,
                &*w_q_proj,
                &self.session.activations.q,
                1,
                (2 * q_dim) as u32,
                hidden as u32,
            );
            self.dispatch_matmul(
                enc,
                &self.session.activations.hidden,
                &*w_k_proj,
                &self.session.activations.k,
                1,
                kv_dim as u32,
                hidden as u32,
            );
            self.dispatch_matmul(
                enc,
                &self.session.activations.hidden,
                &*w_v_proj,
                &self.session.activations.v,
                1,
                kv_dim as u32,
                hidden as u32,
            );
        }
        // LoRA for Q/K/V projections
        self.dispatch_lora_if_active(
            enc,
            &self.session.activations.hidden,
            0,
            &self.session.activations.q,
            0,
            layer_idx,
            "q_proj",
        );
        self.dispatch_lora_if_active(
            enc,
            &self.session.activations.hidden,
            0,
            &self.session.activations.k,
            0,
            layer_idx,
            "k_proj",
        );
        self.dispatch_lora_if_active(
            enc,
            &self.session.activations.hidden,
            0,
            &self.session.activations.v,
            0,
            layer_idx,
            "v_proj",
        );
        if let Some(t0) = gqa_proj_t0 {
            prof.projection_us += t0.elapsed().as_micros();
        }
        self.dispatch_scatter_q_gate(enc, num_q_heads as u32, head_dim as u32);
        // SAFETY: Q/K norm buffers are live layer-owned buffers and the
        // head counts/head_dim come from the same config used to size them.
        unsafe {
            self.dispatch_per_head_rms_norm(
                enc,
                &self.session.activations.q_separated,
                &*w_q_norm,
                num_q_heads as u32,
                head_dim as u32,
                cfg.rms_norm_eps,
            );
            self.dispatch_per_head_rms_norm(
                enc,
                &self.session.activations.k,
                &*w_k_norm,
                num_kv_heads as u32,
                head_dim as u32,
                cfg.rms_norm_eps,
            );
        }
        self.dispatch_partial_rope(
            enc,
            &self.session.activations.q_separated,
            num_q_heads as u32,
            head_dim as u32,
            half_rope_dim,
            position as u32,
            mrope_override,
        );
        self.dispatch_partial_rope(
            enc,
            &self.session.activations.k,
            num_kv_heads as u32,
            head_dim as u32,
            half_rope_dim,
            position as u32,
            mrope_override,
        );

        // GPU KV cache copy
        self.dispatch_copy_offset_kv(
            enc,
            &self.session.activations.k,
            &self.session.kv_cache.k_bufs[full_idx],
            kv_dim as u32,
            kv_cache_offset,
        );
        self.dispatch_copy_offset_kv(
            enc,
            &self.session.activations.v,
            &self.session.kv_cache.v_bufs[full_idx],
            kv_dim as u32,
            kv_cache_offset,
        );

        // Decode attention + gating + O projection
        let gqa_attn_t0 = profiling.then(std::time::Instant::now);
        self.dispatch_decode_attention(
            enc,
            &self.session.kv_cache.k_bufs[full_idx],
            &self.session.kv_cache.v_bufs[full_idx],
            cur_seq_len,
            head_dim as u32,
            num_q_heads as u32,
            num_kv_heads as u32,
            q_dim as u32,
            kv_dim as u32,
            scale,
        );
        self.dispatch_sigmoid_gate(enc, q_dim as u32);
        if let Some(t0) = gqa_attn_t0 {
            prof.gqa_attention_us += t0.elapsed().as_micros();
        }
        // SAFETY: The O-projection buffer pointer is live for the command
        // buffer and dimensions match [hidden, q_dim].
        let gqa_oproj_t0 = profiling.then(std::time::Instant::now);
        unsafe {
            self.dispatch_matmul(
                enc,
                &self.session.activations.attn_out,
                &*w_o_proj,
                &self.session.activations.ffn_out,
                1,
                hidden as u32,
                q_dim as u32,
            );
        }
        // LoRA for o_proj: x = attn_out (gated attention output), y = ffn_out
        self.dispatch_lora_if_active(
            enc,
            &self.session.activations.attn_out,
            0,
            &self.session.activations.ffn_out,
            0,
            layer_idx,
            "o_proj",
        );
        self.dispatch_copy(
            enc,
            &self.session.activations.ffn_out,
            &self.session.activations.attn_out,
            hidden as u32,
        );
        if let Some(t0) = gqa_oproj_t0 {
            prof.projection_us += t0.elapsed().as_micros();
        }

        if with_mlp {
            self.encode_mlp_block(enc, compact_idx, layer_idx, position, cfg, prof, profiling);
        }
    }
}
