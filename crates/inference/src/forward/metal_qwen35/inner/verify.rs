#[cfg(feature = "gdn-state-counters")]
use super::GdnStateCopyKind;
use super::{
    GdnStateTrafficScope, MTP_VERIFY_MAX_TOKENS, MetalFfnWeights, MetalLayerAttnWeights,
    MetalQwen35State, MetalVerifyOutput, Q4_0_BLOCK_SIZE, Q4WeightBuf, Q8_0_BLOCK_SIZE, QK4_0,
    QK8_0, QuantFormat, div_ceil, read_buffer,
};
use metal::{Buffer, MTLSize};

impl MetalQwen35State {
    // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
    // dispatch creates command buffers on the same long-lived worker thread.
    // Unmeasured individually; fixed on the structural argument, not on numbers.
    pub(super) fn checkpoint_gdn_to_slot(
        &mut self,
        slot: usize,
        _traffic_scope: GdnStateTrafficScope,
    ) -> Result<(), crate::error::InferenceError> {
        objc::rc::autoreleasepool(|| self.checkpoint_gdn_to_slot_dispatch(slot, _traffic_scope))
    }

    /// Copy live GDN conv/S buffers into the given checkpoint slot (blocking GPU blit).
    fn checkpoint_gdn_to_slot_dispatch(
        &mut self,
        slot: usize,
        _traffic_scope: GdnStateTrafficScope,
    ) -> Result<(), crate::error::InferenceError> {
        let num_layers = self.session.gdn_gpu_state.len();
        let conv_sizes: Vec<u64> = self
            .session
            .gdn_gpu_state
            .layers()
            .map(|layer| layer.conv_buffer().length())
            .collect();
        let s_sizes: Vec<u64> = self
            .session
            .gdn_gpu_state
            .layers()
            .map(|layer| layer.s_matrix().length())
            .collect();
        // Resolve the checkpoint pool BEFORE allocating Metal command objects
        // so an uninitialized pool returns Err without leaking an un-ended
        // blit encoder or uncommitted command buffer on the error path.
        let pool = self.session.gdn_checkpoints.as_ref().ok_or_else(|| {
            crate::error::InferenceError::Inference("GDN checkpoint pool not initialized".into())
        })?;
        let cmd = self.engine.queue.new_command_buffer();
        let enc = cmd.new_blit_command_encoder();
        for i in 0..num_layers {
            let layer = self.session.gdn_gpu_state.layer(i);
            enc.copy_from_buffer(
                layer.conv_buffer(),
                0,
                &pool.conv_slots[slot][i],
                0,
                conv_sizes[i],
            );
            enc.copy_from_buffer(layer.s_matrix(), 0, &pool.s_slots[slot][i], 0, s_sizes[i]);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        #[cfg(feature = "gdn-state-counters")]
        self.record_gdn_copy(_traffic_scope, GdnStateCopyKind::Snapshot);
        Ok(())
    }

    // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
    // dispatch creates command buffers on the same long-lived worker thread.
    // Unmeasured individually; fixed on the structural argument, not on numbers.
    pub(super) fn restore_gdn_slot_blocking(
        &mut self,
        slot: usize,
        _traffic_scope: GdnStateTrafficScope,
    ) -> Result<(), crate::error::InferenceError> {
        objc::rc::autoreleasepool(|| self.restore_gdn_slot_blocking_dispatch(slot, _traffic_scope))
    }

    /// Restore live GDN conv/S buffers from the given checkpoint slot (blocking GPU blit).
    fn restore_gdn_slot_blocking_dispatch(
        &mut self,
        slot: usize,
        _traffic_scope: GdnStateTrafficScope,
    ) -> Result<(), crate::error::InferenceError> {
        let num_layers = self.session.gdn_gpu_state.len();
        let conv_sizes: Vec<u64> = self
            .session
            .gdn_gpu_state
            .layers()
            .map(|layer| layer.conv_buffer().length())
            .collect();
        let s_sizes: Vec<u64> = self
            .session
            .gdn_gpu_state
            .layers()
            .map(|layer| layer.s_matrix().length())
            .collect();
        // Resolve the checkpoint pool BEFORE allocating Metal command objects
        // so an uninitialized pool returns Err without leaking an un-ended
        // blit encoder or uncommitted command buffer on the error path.
        let pool = self.session.gdn_checkpoints.as_ref().ok_or_else(|| {
            crate::error::InferenceError::Inference("GDN checkpoint pool not initialized".into())
        })?;
        let cmd = self.engine.queue.new_command_buffer();
        let enc = cmd.new_blit_command_encoder();
        for i in 0..num_layers {
            let layer = self.session.gdn_gpu_state.layer(i);
            enc.copy_from_buffer(
                &pool.conv_slots[slot][i],
                0,
                layer.conv_buffer(),
                0,
                conv_sizes[i],
            );
            enc.copy_from_buffer(&pool.s_slots[slot][i], 0, layer.s_matrix(), 0, s_sizes[i]);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        #[cfg(feature = "gdn-state-counters")]
        self.record_gdn_copy(_traffic_scope, GdnStateCopyKind::Restore);
        Ok(())
    }

    /// Roll back KV cache and GDN state to `seq_len` after a speculative rejection.
    ///
    /// Restores from slot `seq_len - active_base_seq_len` in the checkpoint pool.
    pub(super) fn rollback_speculative_state_to(
        &mut self,
        seq_len: usize,
    ) -> Result<(), crate::error::InferenceError> {
        let base = self
            .session
            .gdn_checkpoints
            .as_ref()
            .and_then(|p| p.active_base_seq_len)
            .ok_or_else(|| {
                crate::error::InferenceError::Inference("No active speculation checkpoint".into())
            })?;
        let slot = seq_len.checked_sub(base).ok_or_else(|| {
            crate::error::InferenceError::Inference(format!(
                "rollback seq_len {seq_len} < base {base}"
            ))
        })?;

        // Extract batch-repair info and MTP base before any mutable borrows.
        let batch_repair = self
            .session
            .gdn_checkpoints
            .as_mut()
            .and_then(|p| p.batch_repair_token.take());
        let mtp_base = self
            .session
            .gdn_checkpoints
            .as_ref()
            .and_then(|p| p.mtp_base_seq_len);

        if let Some((repair_token, repair_pos)) = batch_repair {
            // Batch verifier path: restore GDN base, then replay the first verify token so
            // the GDN state lands at S_{repair_pos} (= state after pending_token was processed).
            self.restore_gdn_slot_blocking(0, GdnStateTrafficScope::MtpVerify)?;
            // Set seq_len so forward_step_inner writes KV to the correct slot.
            self.session.rewind_position(repair_pos);
            // Repair step: reprocesses pending_token, advancing GDN and capturing hidden.
            #[cfg(feature = "gdn-state-counters")]
            let repair_out = self.forward_step_inner_with_traffic_scope(
                repair_token,
                repair_pos,
                true,
                GdnStateTrafficScope::MtpVerify,
            );
            #[cfg(not(feature = "gdn-state-counters"))]
            let repair_out = self.forward_step_inner(
                repair_token,
                repair_pos,
                true,
                crate::forward::signpost::Scope::NotDecode,
            );
            self.session.last_pre_final_hidden = repair_out.pre_final_hidden;
            self.session.mark_pre_final_hidden(repair_pos + 1);
            self.session.rewind_position(seq_len);
        } else {
            // Sequential verifier path: slot-based GDN restore.
            self.restore_gdn_slot_blocking(slot, GdnStateTrafficScope::MtpVerify)?;
            self.session.rewind_position(seq_len);
        }

        // Bug 3 fix: restore MTP KV cache position to prevent ghost entries.
        if let Some(base_mtp) = mtp_base
            && let Some(ref mut mtp) = self.session.mtp
        {
            mtp.cache.seq_len = base_mtp + slot;
        }
        Ok(())
    }

    /// Run the target model sequentially on `tokens` starting at `start_pos`,
    /// checkpointing GDN state after each token for speculative rollback.
    ///
    /// Returns per-token logits and the pre-final hidden state of the last token.
    /// This is the H1 sequential path (V_2≈2.0); a batch-GEMM verifier is a follow-up.
    pub(super) fn verify_tokens_batched(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<MetalVerifyOutput, crate::error::InferenceError> {
        self.verify_tokens_batched_impl(tokens, start_pos, false)
    }

    /// `verify_tokens_batched`, optionally retaining every verified row's raw
    /// pre-final hidden in `MetalVerifyOutput::pre_final_hidden_rows`. The
    /// rows come from the same per-token forward that produced the logits.
    fn verify_tokens_batched_impl(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        keep_all_rows: bool,
    ) -> Result<MetalVerifyOutput, crate::error::InferenceError> {
        self.check_forward_range_capacity(start_pos, tokens.len(), true)?;
        self.check_live_cursor("verify_tokens_batched", start_pos)?;
        if self.session.gdn_checkpoints.is_none() {
            return Err(crate::error::InferenceError::Inference(
                "GDN checkpoint pool required for verify_tokens_batched".into(),
            ));
        }
        if let Some(ref mut p) = self.session.gdn_checkpoints {
            p.active_base_seq_len = Some(start_pos);
            p.mtp_base_seq_len = self.session.mtp.as_ref().map(|m| m.cache.seq_len);
        }
        self.checkpoint_gdn_to_slot(0, GdnStateTrafficScope::MtpVerify)?;
        let mut all_logits: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
        let mut final_hidden = Vec::new();
        let mut first_pre_final_hidden = Vec::new();
        let mut pre_final_hidden_rows: Vec<Vec<f32>> = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            #[cfg(feature = "gdn-state-counters")]
            let out = self.forward_step_inner_with_traffic_scope(
                token,
                start_pos + i,
                true,
                GdnStateTrafficScope::MtpVerify,
            );
            #[cfg(not(feature = "gdn-state-counters"))]
            let out = self.forward_step_inner(
                token,
                start_pos + i,
                true,
                crate::forward::signpost::Scope::NotDecode,
            );
            self.checkpoint_gdn_to_slot(i + 1, GdnStateTrafficScope::MtpVerify)?;
            all_logits.push(out.logits);
            if keep_all_rows {
                pre_final_hidden_rows.push(out.pre_final_hidden.clone());
            }
            if i == 0 {
                // lattice#1396: the pending token's own pre-final hidden,
                // needed to append the accepted draft's MTP-cache row and
                // to restore `last_pre_final_hidden` on a K=1 reject —
                // captured here for free (this step already computes it)
                // before the next iteration's `final_hidden` overwrite.
                first_pre_final_hidden = out.pre_final_hidden.clone();
            }
            final_hidden = out.pre_final_hidden;
        }
        Ok(MetalVerifyOutput {
            logits: all_logits,
            final_hidden,
            first_pre_final_hidden,
            pre_final_hidden_rows,
        })
    }

    // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
    // dispatch creates command buffers on the same long-lived worker thread.
    // Unmeasured individually; fixed on the structural argument, not on numbers.
    pub(super) fn verify_tokens_batch_gemm(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<MetalVerifyOutput, crate::error::InferenceError> {
        objc::rc::autoreleasepool(|| self.verify_tokens_batch_gemm_dispatch(tokens, start_pos))
    }

    /// Batch-GEMM verifier (activate with `LATTICE_MTP_BATCH=1`).
    ///
    /// Processes K verify tokens layer-by-layer using batch GEMM for all weight projections,
    /// with sequential GDN recurrence within each layer. Reduces weight-load cost vs the
    /// sequential verifier: each weight matrix is loaded once for K outputs instead of K times.
    ///
    /// **GDN rollback strategy**: Checkpoints GDN base (slot 0) before the batch. On rejection,
    /// `rollback_speculative_state_to` restores slot 0 and runs one repair forward step for the
    /// pending token to re-establish the correct intermediate GDN state.
    ///
    /// Expected V_2: ~1.35 at α=100%, ~1.60 at α=75%, ~2.30 at α=5%.
    fn verify_tokens_batch_gemm_dispatch(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<MetalVerifyOutput, crate::error::InferenceError> {
        self.check_live_cursor("verify_tokens_batch_gemm", start_pos)?;
        let n = tokens.len();
        if n == 0 || n > MTP_VERIFY_MAX_TOKENS {
            return Err(crate::error::InferenceError::Inference(format!(
                "verify_batch: bad token count {n} (max {MTP_VERIFY_MAX_TOKENS})"
            )));
        }
        self.check_forward_range_capacity(start_pos, n, false)?;
        if self.session.gdn_checkpoints.is_none() {
            return Err(crate::error::InferenceError::Inference(
                "GDN checkpoint pool required for verify_tokens_batch_gemm".into(),
            ));
        }

        let cfg = self.engine.config.clone();
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let kv_dim = cfg.full_kv_dim();
        let q_dim = cfg.full_q_dim();
        let num_q_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let half_rope_dim = (cfg.rope_dim() / 2) as u32;
        let m = n as u32;

        // Checkpoint GDN base + record repair token for rollback.
        if let Some(ref mut p) = self.session.gdn_checkpoints {
            p.active_base_seq_len = Some(start_pos);
            p.mtp_base_seq_len = self.session.mtp.as_ref().map(|mtp| mtp.cache.seq_len);
            p.batch_repair_token = if n > 0 {
                Some((tokens[0], start_pos))
            } else {
                None
            };
        }
        self.checkpoint_gdn_to_slot(0, GdnStateTrafficScope::MtpVerify)?;

        // Batch embedding: one row per token into hidden[0..K*hidden].
        let dst_base = self.session.activations.hidden.contents() as *mut f32;
        for (t, &id) in tokens.iter().enumerate() {
            assert!(
                (id as usize) < cfg.vocab_size,
                "verify_batch: token {id} >= vocab_size {}",
                cfg.vocab_size
            );
            // SAFETY: dst row t holds hidden f32 values and is disjoint from the table.
            unsafe {
                self.engine
                    .embed_tokens
                    .write_row(id as usize, dst_base.add(t * hidden), hidden);
            }
        }

        let mut active_layer_idx = 0usize;
        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        #[cfg(feature = "gdn-state-counters")]
        let mut gdn_verify_layer_dispatches = 0usize;

        let cmd = self.engine.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        for layer_i in 0..cfg.num_hidden_layers {
            if !cfg.is_layer_active(layer_i) {
                continue;
            }
            let compact_idx = active_layer_idx;
            active_layer_idx += 1;
            let is_linear = matches!(
                &self.engine.layer_weights[compact_idx].0,
                MetalLayerAttnWeights::Linear(_)
            );
            let (_, common_w) = &self.engine.layer_weights[compact_idx];
            let w_in_norm = &common_w.input_layernorm as *const Buffer;
            let w_post_norm = &common_w.post_attention_layernorm as *const Buffer;
            let (w_gate_up, w_down, gate_byte_size) = match &common_w.ffn {
                MetalFfnWeights::Dense {
                    gate_up_proj,
                    down_proj,
                } => {
                    let byte_size: u64 = match self.engine.quant_format {
                        QuantFormat::Q8_0 => (inter * (hidden / QK8_0) * Q8_0_BLOCK_SIZE) as u64,
                        QuantFormat::Q4_0 => (inter * (hidden / QK4_0) * Q4_0_BLOCK_SIZE) as u64,
                    };
                    (
                        gate_up_proj as *const Q4WeightBuf,
                        down_proj as *const Q4WeightBuf,
                        byte_size,
                    )
                }
                MetalFfnWeights::Moe(_) => {
                    panic!(
                        "verify_tokens_batch_gemm: MoE layers are not supported in batch \
                            prefill (M>1). ADR-053 v1 targets decode-mode only."
                    );
                }
            };

            if is_linear {
                // GDN layer: batch projections + sequential recurrence (causal dependency).
                let (w_qkv, w_z, w_b, w_a, w_alog, w_dtb, w_conv, w_norm, w_out) = {
                    let MetalLayerAttnWeights::Linear(gdn_w) =
                        &self.engine.layer_weights[compact_idx].0
                    else {
                        unreachable!()
                    };
                    (
                        &gdn_w.in_proj_qkv as *const Q4WeightBuf,
                        &gdn_w.in_proj_z as *const Q4WeightBuf,
                        &gdn_w.in_proj_b as *const Buffer,
                        &gdn_w.in_proj_a as *const Buffer,
                        &gdn_w.a_log as *const Buffer,
                        &gdn_w.dt_bias as *const Buffer,
                        &gdn_w.conv1d_weight as *const Buffer,
                        &gdn_w.norm_weight as *const Buffer,
                        &gdn_w.out_proj as *const Q4WeightBuf,
                    )
                };
                let qkv_d = cfg.linear_qkv_dim();
                let out_d = cfg.linear_output_dim();
                let num_h = cfg.linear_num_key_heads;
                let key_d = cfg.linear_key_head_dim;
                let val_d = cfg.linear_value_head_dim;
                let ks = cfg.linear_conv_kernel_dim as u32;

                // Batch: fused copy hidden → residual + norm hidden in-place.
                // SAFETY: w_in_norm is a live layer-owned buffer pointer (dropped only when
                // the model is destroyed). Activation buffers hidden/residual are sized for
                // m * hidden f32 elements. The Metal command buffer enc is active.
                unsafe {
                    self.dispatch_copy_and_rms_norm_batch(
                        enc,
                        &self.session.activations.hidden,
                        &self.session.activations.residual,
                        &*w_in_norm,
                        hidden as u32,
                        m,
                        cfg.rms_norm_eps,
                    );
                }

                // Batch: QKV + Z projections (GEMM, M=K — one weight load for K outputs).
                // SAFETY: w_qkv, w_z are live layer-owned buffers.
                unsafe {
                    self.dispatch_gemm(
                        enc,
                        &self.session.activations.hidden,
                        0,
                        &*w_qkv,
                        &self.session.activations.gdn_qkv,
                        0,
                        m,
                        qkv_d as u32,
                        hidden as u32,
                    );
                    self.dispatch_gemm(
                        enc,
                        &self.session.activations.hidden,
                        0,
                        &*w_z,
                        &self.session.activations.gdn_z,
                        0,
                        m,
                        out_d as u32,
                        hidden as u32,
                    );
                }

                // Sequential: conv1d + recurrence per token (inherently causal).
                // SAFETY: Token offsets are within m-sized activation buffers;
                // conv/state/weight buffers are live for the command buffer.
                for t in 0..n {
                    let qkv_off = (t * qkv_d) as u64 * 4;
                    let z_off = (t * out_d) as u64 * 4;
                    let h_off = (t * hidden) as u64 * 4;
                    unsafe {
                        enc.set_compute_pipeline_state(&self.engine.pipelines.conv1d_silu);
                        enc.set_buffer(
                            0,
                            Some(self.session.gdn_gpu_state.layer(linear_idx).conv_buffer()),
                            0,
                        );
                        enc.set_buffer(1, Some(&self.session.activations.gdn_qkv), qkv_off);
                        enc.set_buffer(2, Some(&*w_conv), 0);
                        enc.set_buffer(3, Some(&self.session.gdn_gpu_conv_out), 0);
                        let qd = qkv_d as u32;
                        enc.set_bytes(4, 4, &qd as *const u32 as *const _);
                        enc.set_bytes(5, 4, &ks as *const u32 as *const _);
                        let wg = 256u64;
                        enc.dispatch_threads(
                            MTLSize::new(div_ceil(qkv_d as u64, wg) * wg, 1, 1),
                            MTLSize::new(wg, 1, 1),
                        );

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
                        enc.set_compute_pipeline_state(&self.engine.pipelines.gdn_recurrence);
                        enc.set_buffer(
                            0,
                            Some(self.session.gdn_gpu_state.layer(linear_idx).s_matrix()),
                            0,
                        );
                        enc.set_buffer(1, Some(&self.session.gdn_gpu_conv_out), 0);
                        enc.set_buffer(2, Some(&self.session.activations.gdn_z), z_off);
                        enc.set_buffer(3, Some(&self.session.activations.hidden), h_off);
                        enc.set_buffer(4, Some(&*w_b), 0);
                        enc.set_buffer(5, Some(&*w_a), 0);
                        enc.set_buffer(6, Some(&*w_alog), 0);
                        enc.set_buffer(7, Some(&*w_dtb), 0);
                        enc.set_buffer(8, Some(&*w_norm), 0);
                        enc.set_buffer(9, Some(&self.session.activations.gdn_z), z_off);
                        enc.set_bytes(
                            10,
                            std::mem::size_of::<GdnRecurParams>() as u64,
                            &params as *const GdnRecurParams as *const _,
                        );
                        enc.dispatch_thread_groups(
                            MTLSize::new(num_vh as u64, 1, 1),
                            MTLSize::new(32, 4, 1),
                        );
                    }
                    #[cfg(feature = "gdn-state-counters")]
                    {
                        gdn_verify_layer_dispatches += 1;
                    }
                }

                // Batch: out_proj + residual + post-norm + MLP.
                // SAFETY: w_out, w_gate_up, w_down are live layer-owned buffers.
                unsafe {
                    self.dispatch_gemm(
                        enc,
                        &self.session.activations.gdn_z,
                        0,
                        &*w_out,
                        &self.session.activations.attn_out,
                        0,
                        m,
                        hidden as u32,
                        out_d as u32,
                    );
                }
                // Batch: fused residual-add + norm for MLP.
                // SAFETY: w_post_norm is a live layer-owned buffer pointer.
                unsafe {
                    self.dispatch_fused_residual_add_norm_batch(
                        enc,
                        &self.session.activations.residual,
                        &self.session.activations.attn_out,
                        &self.session.activations.residual,
                        &self.session.activations.hidden,
                        &*w_post_norm,
                        hidden as u32,
                        m,
                        cfg.rms_norm_eps,
                    );
                    self.dispatch_gemm_at(
                        enc,
                        &self.session.activations.hidden,
                        0,
                        &*w_gate_up,
                        0,
                        &self.session.activations.gate,
                        0,
                        m,
                        inter as u32,
                        hidden as u32,
                    );
                    self.dispatch_gemm_at(
                        enc,
                        &self.session.activations.hidden,
                        0,
                        &*w_gate_up,
                        gate_byte_size,
                        &self.session.activations.up,
                        0,
                        m,
                        inter as u32,
                        hidden as u32,
                    );
                }
                self.dispatch_silu_mul(enc, m * inter as u32);
                unsafe {
                    self.dispatch_gemm(
                        enc,
                        &self.session.activations.gate,
                        0,
                        &*w_down,
                        &self.session.activations.ffn_out,
                        0,
                        m,
                        hidden as u32,
                        inter as u32,
                    );
                }
                // Batch: fused end-of-layer residual add + copy.
                self.dispatch_add_and_copy(
                    enc,
                    &self.session.activations.ffn_out,
                    &self.session.activations.residual,
                    &self.session.activations.hidden,
                    m * hidden as u32,
                );
                linear_idx += 1;
            } else {
                // Full-attention layer: batch QKV/O/MLP projections, sequential per-token ops.
                let (w_q, w_k, w_v, w_o, w_qn, w_kn) = {
                    let MetalLayerAttnWeights::Full(full_w) =
                        &self.engine.layer_weights[compact_idx].0
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
                let scale = 1.0f32 / (head_dim as f32).sqrt();

                // Batch: fused copy hidden → residual + norm hidden in-place.
                // SAFETY: w_in_norm is a live layer-owned buffer pointer.
                unsafe {
                    self.dispatch_copy_and_rms_norm_batch(
                        enc,
                        &self.session.activations.hidden,
                        &self.session.activations.residual,
                        &*w_in_norm,
                        hidden as u32,
                        m,
                        cfg.rms_norm_eps,
                    );
                }

                // Batch: Q/K/V projections.
                // SAFETY: w_q, w_k, w_v are live layer-owned buffers.
                unsafe {
                    self.dispatch_gemm(
                        enc,
                        &self.session.activations.hidden,
                        0,
                        &*w_q,
                        &self.session.activations.q,
                        0,
                        m,
                        (2 * q_dim) as u32,
                        hidden as u32,
                    );
                    self.dispatch_gemm(
                        enc,
                        &self.session.activations.hidden,
                        0,
                        &*w_k,
                        &self.session.activations.k,
                        0,
                        m,
                        kv_dim as u32,
                        hidden as u32,
                    );
                    self.dispatch_gemm(
                        enc,
                        &self.session.activations.hidden,
                        0,
                        &*w_v,
                        &self.session.activations.v,
                        0,
                        m,
                        kv_dim as u32,
                        hidden as u32,
                    );
                }

                // Sequential: scatter, norm, RoPE, KV store, attention, gate (causal).
                // KV cache writes at absolute positions start_pos+t; attention sees cache[0..start_pos+t+1].
                // SAFETY: Per-token offsets are within m-sized activation buffers;
                // KV cache buffers are live StorageModeShared, sized for max_cache_len.
                for t in 0..n {
                    let q_off = (t * 2 * q_dim) as u64 * 4;
                    let qs_off = (t * q_dim) as u64 * 4;
                    let gz_off = (t * q_dim) as u64 * 4;
                    let k_off = (t * kv_dim) as u64 * 4;
                    let v_off = (t * kv_dim) as u64 * 4;
                    let ao_off = (t * q_dim) as u64 * 4;
                    let abs_pos = start_pos + t;
                    let cache_len = (abs_pos + 1) as u32;
                    let kv_dst_off = (abs_pos * kv_dim) as u32;

                    unsafe {
                        // Scatter Q + gate from interleaved q_proj.
                        enc.set_compute_pipeline_state(&self.engine.pipelines.scatter_q_gate);
                        enc.set_buffer(0, Some(&self.session.activations.q), q_off);
                        enc.set_buffer(1, Some(&self.session.activations.q_separated), qs_off);
                        enc.set_buffer(2, Some(&self.session.activations.gate_z), gz_off);
                        let nh = num_q_heads as u32;
                        let hd = head_dim as u32;
                        enc.set_bytes(3, 4, &nh as *const u32 as *const _);
                        enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                        let wg = 256u64;
                        enc.dispatch_threads(
                            MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                            MTLSize::new(wg, 1, 1),
                        );

                        // Per-head RMS norm Q and K.
                        enc.set_compute_pipeline_state(&self.engine.pipelines.per_head_rms_norm);
                        enc.set_buffer(0, Some(&self.session.activations.q_separated), qs_off);
                        enc.set_buffer(1, Some(&*w_qn), 0);
                        enc.set_bytes(2, 4, &nh as *const u32 as *const _);
                        enc.set_bytes(3, 4, &hd as *const u32 as *const _);
                        enc.set_bytes(4, 4, &cfg.rms_norm_eps as *const f32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new(nh as u64, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );
                        enc.set_buffer(0, Some(&self.session.activations.k), k_off);
                        enc.set_buffer(1, Some(&*w_kn), 0);
                        let nkh = num_kv_heads as u32;
                        enc.set_bytes(2, 4, &nkh as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new(nkh as u64, 1, 1),
                            MTLSize::new(256, 1, 1),
                        );

                        // Partial RoPE for Q and K at absolute position (start_pos + t).
                        let pos = abs_pos as u32;
                        enc.set_compute_pipeline_state(&self.engine.pipelines.partial_rope);
                        enc.set_buffer(0, Some(&self.session.activations.q_separated), qs_off);
                        enc.set_buffer(1, Some(&self.engine.rope_cos), 0);
                        enc.set_buffer(2, Some(&self.engine.rope_sin), 0);
                        enc.set_bytes(3, 4, &nh as *const u32 as *const _);
                        enc.set_bytes(4, 4, &hd as *const u32 as *const _);
                        enc.set_bytes(5, 4, &half_rope_dim as *const u32 as *const _);
                        enc.set_bytes(6, 4, &pos as *const u32 as *const _);
                        let wg = 256u64;
                        enc.dispatch_threads(
                            MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                            MTLSize::new(wg, 1, 1),
                        );
                        enc.set_buffer(0, Some(&self.session.activations.k), k_off);
                        let nkh = num_kv_heads as u32;
                        enc.set_bytes(3, 4, &nkh as *const u32 as *const _);
                        enc.dispatch_threads(
                            MTLSize::new(div_ceil(kv_dim as u64, wg) * wg, 1, 1),
                            MTLSize::new(wg, 1, 1),
                        );
                    }

                    // KV store at absolute position (start_pos + t).
                    if self.use_kv_f16 {
                        enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset_f16);
                    } else {
                        enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset);
                    }
                    enc.set_buffer(0, Some(&self.session.activations.k), k_off);
                    enc.set_buffer(1, Some(&self.session.kv_cache.k_bufs[full_idx]), 0);
                    let cnt = kv_dim as u32;
                    enc.set_bytes(2, 4, &cnt as *const u32 as *const _);
                    enc.set_bytes(3, 4, &kv_dst_off as *const u32 as *const _);
                    let wg = 256u64;
                    enc.dispatch_threads(
                        MTLSize::new(div_ceil(kv_dim as u64, wg) * wg, 1, 1),
                        MTLSize::new(wg, 1, 1),
                    );
                    enc.set_buffer(0, Some(&self.session.activations.v), v_off);
                    enc.set_buffer(1, Some(&self.session.kv_cache.v_bufs[full_idx]), 0);
                    enc.dispatch_threads(
                        MTLSize::new(div_ceil(kv_dim as u64, wg) * wg, 1, 1),
                        MTLSize::new(wg, 1, 1),
                    );

                    // Causal attention: Q[t] against full cache[0..start_pos+t+1].
                    if self.use_kv_f16 {
                        enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attention_f16);
                    } else {
                        enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attention);
                    }
                    enc.set_buffer(0, Some(&self.session.activations.q_separated), qs_off);
                    enc.set_buffer(1, Some(&self.session.kv_cache.k_bufs[full_idx]), 0);
                    enc.set_buffer(2, Some(&self.session.kv_cache.v_bufs[full_idx]), 0);
                    enc.set_buffer(3, Some(&self.session.activations.attn_out), ao_off);
                    enc.set_bytes(4, 4, &cache_len as *const u32 as *const _);
                    let hd = head_dim as u32;
                    let nqh = num_q_heads as u32;
                    let nkh = num_kv_heads as u32;
                    let qd = q_dim as u32;
                    let kvd = kv_dim as u32;
                    enc.set_bytes(5, 4, &hd as *const u32 as *const _);
                    enc.set_bytes(6, 4, &nqh as *const u32 as *const _);
                    enc.set_bytes(7, 4, &nkh as *const u32 as *const _);
                    enc.set_bytes(8, 4, &qd as *const u32 as *const _);
                    enc.set_bytes(9, 4, &kvd as *const u32 as *const _);
                    enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                    // Grid axis = num_kv_heads: the kernel maps gid → kvh and handles
                    // each GQA group of Q heads internally per threadgroup. Dispatching
                    // nqh wastes (nqh - nkh) threadgroups that the kernel's early-return
                    // guard discards; output is bit-identical but scheduling is nkh-sized.
                    enc.dispatch_thread_groups(
                        MTLSize::new(nkh as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );

                    // Sigmoid gate.
                    let cnt = q_dim as u32;
                    enc.set_compute_pipeline_state(&self.engine.pipelines.sigmoid_gate);
                    enc.set_buffer(0, Some(&self.session.activations.attn_out), ao_off);
                    enc.set_buffer(1, Some(&self.session.activations.gate_z), gz_off);
                    enc.set_bytes(2, 4, &cnt as *const u32 as *const _);
                    let wg = 256u64;
                    enc.dispatch_threads(
                        MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                        MTLSize::new(wg, 1, 1),
                    );
                }

                // Batch: O projection + residual + post-norm + MLP.
                // SAFETY: w_o, w_gate_up, w_down are live layer-owned buffers.
                unsafe {
                    self.dispatch_gemm(
                        enc,
                        &self.session.activations.attn_out,
                        0,
                        &*w_o,
                        &self.session.activations.ffn_out,
                        0,
                        m,
                        hidden as u32,
                        q_dim as u32,
                    );
                }
                // Batch: fused residual-add + norm for MLP.
                // SAFETY: w_post_norm is a live layer-owned buffer pointer.
                unsafe {
                    self.dispatch_fused_residual_add_norm_batch(
                        enc,
                        &self.session.activations.residual,
                        &self.session.activations.ffn_out,
                        &self.session.activations.residual,
                        &self.session.activations.hidden,
                        &*w_post_norm,
                        hidden as u32,
                        m,
                        cfg.rms_norm_eps,
                    );
                    self.dispatch_gemm_at(
                        enc,
                        &self.session.activations.hidden,
                        0,
                        &*w_gate_up,
                        0,
                        &self.session.activations.gate,
                        0,
                        m,
                        inter as u32,
                        hidden as u32,
                    );
                    self.dispatch_gemm_at(
                        enc,
                        &self.session.activations.hidden,
                        0,
                        &*w_gate_up,
                        gate_byte_size,
                        &self.session.activations.up,
                        0,
                        m,
                        inter as u32,
                        hidden as u32,
                    );
                }
                self.dispatch_silu_mul(enc, m * inter as u32);
                unsafe {
                    self.dispatch_gemm(
                        enc,
                        &self.session.activations.gate,
                        0,
                        &*w_down,
                        &self.session.activations.ffn_out,
                        0,
                        m,
                        hidden as u32,
                        inter as u32,
                    );
                }
                // Batch: fused end-of-layer residual add + copy.
                self.dispatch_add_and_copy(
                    enc,
                    &self.session.activations.ffn_out,
                    &self.session.activations.residual,
                    &self.session.activations.hidden,
                    m * hidden as u32,
                );
                full_idx += 1;
            }
        } // end layer loop

        // Capture pre-final hidden for last token (for MTP next round).
        let last_off = ((n - 1) * hidden) as u64 * 4;
        if self.session.mtp.is_some() {
            enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset);
            enc.set_buffer(0, Some(&self.session.activations.hidden), last_off);
            enc.set_buffer(1, Some(&self.session.activations.pre_final_hidden), 0);
            let cnt = hidden as u32;
            let dst_off = 0u32;
            enc.set_bytes(2, 4, &cnt as *const u32 as *const _);
            enc.set_bytes(3, 4, &dst_off as *const u32 as *const _);
            let wg = 256u64;
            enc.dispatch_threads(
                MTLSize::new(div_ceil(hidden as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
        }

        // Final RMSNorm for all K tokens (batch).
        self.dispatch_rms_norm(
            enc,
            &self.session.activations.hidden,
            &self.engine.final_norm,
            hidden as u32,
            m,
            cfg.rms_norm_eps,
        );
        // Site 1 of 3. `last_off` is the last of the K verified tokens.
        self.encode_capture_final_hidden(enc, last_off, hidden);

        // Logits GEMM for all K tokens → verify_logits[K * vocab_size].
        self.dispatch_gemm(
            enc,
            &self.session.activations.hidden,
            0,
            &self.engine.embed_tokens_q8,
            &self.session.activations.verify_logits,
            0,
            m,
            cfg.vocab_size as u32,
            hidden as u32,
        );

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        #[cfg(feature = "gdn-state-counters")]
        self.record_gdn_layer_access(GdnStateTrafficScope::MtpVerify, gdn_verify_layer_dispatches);

        // Advance KV cache position.
        self.session.kv_cache.seq_len = start_pos + n;

        // Read back per-token logits from verify_logits buffer.
        let vocab = cfg.vocab_size;
        let flat = unsafe { read_buffer(&self.session.activations.verify_logits, n * vocab) };
        let all_logits: Vec<Vec<f32>> = (0..n)
            .map(|t| flat[t * vocab..(t + 1) * vocab].to_vec())
            .collect();

        // Read back and update pre-final hidden (on accept, used by next mtp_forward_one).
        let final_hidden = if self.session.mtp.is_some() {
            let h = unsafe { read_buffer(&self.session.activations.pre_final_hidden, hidden) };
            self.session.last_pre_final_hidden = h.clone();
            self.session.mark_pre_final_hidden(start_pos + n);
            h
        } else {
            // No write: the cursor moved past the marker (or, with the verified range
            // starting below it, rewrote the history the marker described).
            if start_pos < self.session.last_hidden_cursor.unwrap_or(0) {
                self.session.last_hidden_cursor = None;
            }
            Vec::new()
        };

        // lattice#1396: raw pre-final hidden of the FIRST verified row (the
        // pending token's own position), needed to append the accepted
        // draft's MTP-cache row. `activations.residual` holds this value
        // untouched: the layer loop's own end-of-layer fused add+copy
        // (`dispatch_add_and_copy`) leaves `residual[i] == hidden[i]` for
        // every one of the `n` verified rows right after the last layer,
        // and unlike `activations.hidden`, `residual` is never written
        // again by the batch RMSNorm dispatched above — that dispatch
        // only mutates `hidden` in place. No extra GPU dispatch is needed;
        // this is a second CPU-side read of an already-computed buffer,
        // the same pattern the MTP prefill capture uses on `hidden` itself.
        let first_pre_final_hidden = if self.session.mtp.is_some() {
            unsafe { read_buffer(&self.session.activations.residual, hidden) }
        } else {
            Vec::new()
        };

        Ok(MetalVerifyOutput {
            logits: all_logits,
            final_hidden,
            first_pre_final_hidden,
            pre_final_hidden_rows: Vec::new(),
        })
    }
}

// `MtpTargetVerifier` is a public trait and this impl's
// mutating methods (`rollback_cache_to`, `verify_tokens`,
// `verify_tokens_with_hidden`) advance live KV/GDN state exactly like
// `forward_step`/`forward_prefill`. They are not wired into any live Metal
// generate loop (Metal MTP decode uses `mtp_greedy_round`, a self-contained
// mechanism) and only tests call `mtp_verify_draft` in this repository, but
// a consumer holding a `&mut MetalQwen35State` and `use`-ing this trait can
// call them directly, so they are part of the public raw-forward boundary
// and must clear the retained cross-turn entry before mutating, same as the
// inherent methods.
impl crate::speculative::MtpTargetVerifier for MetalQwen35State {
    fn cache_position(&self) -> usize {
        self.session.kv_cache.seq_len
    }

    fn rollback_cache_to(&mut self, seq_len: usize) -> Result<(), crate::error::InferenceError> {
        self.cross_turn_prefix_cache.clear();
        self.rollback_speculative_state_to(seq_len)
    }

    fn verify_tokens(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<Vec<f32>>, crate::error::InferenceError> {
        self.check_live_cursor("MtpTargetVerifier::verify_tokens", start_pos)?;
        self.cross_turn_prefix_cache.clear();
        let out = self.verify_tokens_batched(tokens, start_pos)?;
        Ok(out.logits)
    }

    // Some only when the buffered hidden state is provably the one the target's final
    // norm read for the token at `cursor - 1` of the live history: the freshness marker
    // is set at every write of the buffer, left behind by a cursor move that did not
    // write it, and cleared by a rewind below it, a reset and a cross-turn restore. A
    // rotated checkpoint keeps its hidden states in the rotated basis, so it never
    // supplies one.
    fn last_hidden(&self) -> Option<Vec<f32>> {
        if self.checkpoint_is_rotated()
            || self.session.last_hidden_cursor != Some(self.session.kv_cache.seq_len)
            || self.session.last_pre_final_hidden.len() != self.engine.config.hidden_size
        {
            return None;
        }
        let mut hidden = self.session.last_pre_final_hidden.clone();
        self.apply_final_norm_cpu(&mut hidden);
        Some(hidden)
    }

    #[allow(clippy::type_complexity)]
    fn verify_tokens_with_hidden(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<Vec<f32>>>), crate::error::InferenceError> {
        // A rotated checkpoint keeps its hidden states in the rotated basis and stores a
        // neutral final-norm scale, so it cannot supply post-final-norm rows in the
        // original basis. Behave exactly as `verify_tokens`: same forward, same state.
        if self.checkpoint_is_rotated() {
            return Ok((self.verify_tokens(tokens, start_pos)?, None));
        }
        self.check_live_cursor("MtpTargetVerifier::verify_tokens_with_hidden", start_pos)?;
        self.cross_turn_prefix_cache.clear();
        let out = self.verify_tokens_batched_impl(tokens, start_pos, true)?;
        let hidden = self.engine.config.hidden_size;
        // Every row or none: a missing row is recoverable, a wrong one is not.
        if out.pre_final_hidden_rows.len() != tokens.len()
            || out.pre_final_hidden_rows.iter().any(|r| r.len() != hidden)
        {
            return Ok((out.logits, None));
        }
        let mut rows = out.pre_final_hidden_rows;
        for row in &mut rows {
            self.apply_final_norm_cpu(row);
        }
        Ok((out.logits, Some(rows)))
    }

    fn snapshot_gdn_states(&self) -> crate::attention::gdn::GdnSnapshot {
        let num_layers = self.session.gdn_gpu_state.len();
        let mut snap = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = self.session.gdn_gpu_state.layer(i);
            let conv_buf = layer.conv_buffer();
            let s_buf = layer.s_matrix();
            let conv_floats = (conv_buf.length() / 4) as usize;
            let s_floats = (s_buf.length() / 4) as usize;
            // SAFETY: GPU buffers are StorageModeShared (allocated with
            // MTLResourceOptions::StorageModeShared in `new`/`from_q4_dir`), so
            // `contents()` points to host-readable memory. `length()` is the
            // exact allocated byte length and is divisible by 4 (we always
            // allocate f32 buffers). Callers of `snapshot_gdn_states` invoke it
            // outside any in-flight command buffer (in `mtp_verify_draft` before
            // `target.verify_tokens`), so no GPU write can race with this read.
            let conv = unsafe {
                let ptr = conv_buf.contents() as *const f32;
                std::slice::from_raw_parts(ptr, conv_floats).to_vec()
            };
            let s = unsafe {
                let ptr = s_buf.contents() as *const f32;
                std::slice::from_raw_parts(ptr, s_floats).to_vec()
            };
            snap.push((s, conv));
        }
        snap
    }

    fn restore_gdn_states(&mut self, snapshot: &crate::attention::gdn::GdnSnapshot) {
        if let Err(error) = self.validate_gdn_restore_snapshot(snapshot) {
            tracing::warn!(
                error = %error,
                "refusing malformed GDN state restore"
            );
            return;
        }
        self.restore_gdn_states_validated(snapshot);
    }
}

impl MetalQwen35State {
    pub(super) fn validate_gdn_restore_snapshot(
        &self,
        snapshot: &crate::attention::gdn::GdnSnapshot,
    ) -> Result<(), crate::error::InferenceError> {
        use crate::error::InferenceError;

        let num_layers = self.session.gdn_gpu_state.len();
        if snapshot.len() != num_layers {
            return Err(InferenceError::PrefixCache(format!(
                "restore_gdn_states: snapshot has {} layers, expected {num_layers}",
                snapshot.len()
            )));
        }
        for (i, (s_snap, conv_snap)) in snapshot.iter().enumerate() {
            let layer = self.session.gdn_gpu_state.layer(i);
            let conv_buf = layer.conv_buffer();
            let s_buf = layer.s_matrix();
            let conv_floats = (conv_buf.length() / 4) as usize;
            let s_floats = (s_buf.length() / 4) as usize;
            if conv_snap.len() != conv_floats || s_snap.len() != s_floats {
                return Err(InferenceError::PrefixCache(format!(
                    "restore_gdn_states: layer {i} shape mismatch: conv {} != \
                     {conv_floats} or s {} != {s_floats}",
                    conv_snap.len(),
                    s_snap.len()
                )));
            }
        }
        Ok(())
    }

    pub(super) fn restore_gdn_states_validated(
        &mut self,
        snapshot: &crate::attention::gdn::GdnSnapshot,
    ) {
        for (i, (s_snap, conv_snap)) in snapshot.iter().enumerate() {
            let layer = self.session.gdn_gpu_state.layer(i);
            let conv_buf = layer.conv_buffer();
            let s_buf = layer.s_matrix();
            // SAFETY: see snapshot_gdn_states. StorageModeShared lets the CPU write
            // the buffer directly; validation above proves each snapshot exactly
            // matches its destination, and callers invoke this outside any in-flight
            // command buffer (rejection branch in `mtp_verify_draft`).
            unsafe {
                let dst = conv_buf.contents() as *mut f32;
                std::ptr::copy_nonoverlapping(conv_snap.as_ptr(), dst, conv_snap.len());
            }
            unsafe {
                let dst = s_buf.contents() as *mut f32;
                std::ptr::copy_nonoverlapping(s_snap.as_ptr(), dst, s_snap.len());
            }
            if i < self.session.gdn_states.len() {
                self.session.gdn_states[i].restore_from(&(s_snap.clone(), conv_snap.clone()));
            }
        }
    }
}
