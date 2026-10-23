use super::{
    MetalMtpForwardOutput, MetalQwen35State, MtpHiddenTap, div_ceil, read_buffer, write_buffer,
};
use metal::{Buffer, MTLSize};

impl MetalQwen35State {
    // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
    // dispatch creates command buffers on the same long-lived worker thread.
    // Unmeasured individually; fixed on the structural argument, not on numbers.
    pub(super) fn mtp_forward_one(
        &mut self,
        pending_token: u32,
        position: usize,
    ) -> MetalMtpForwardOutput {
        objc::rc::autoreleasepool(|| self.mtp_forward_one_dispatch(pending_token, position))
    }

    /// RMSNorm for the MTP head's two CPU-computed pre-fc norms
    /// (`pre_fc_norm_embedding`, `pre_fc_norm_hidden`). Applies this
    /// architecture's shifted convention, `x * inv_rms * (1.0 + gamma)`,
    /// matching every GPU-kernel RMSNorm in this forward pass and the
    /// model's own `final_norm`. Plain `gamma` (no `+1`) is a different,
    /// incompatible convention and silently inverts the sign of most
    /// channels on a checkpoint whose norm weights center negative.
    pub(super) fn mtp_pre_fc_rmsnorm(x: &mut [f32], gamma: &[f32], eps: f32) {
        debug_assert_eq!(x.len(), gamma.len());
        let mut sum_sq = 0.0f32;
        for &v in x.iter() {
            sum_sq += v * v;
        }
        let inv_rms = 1.0 / (sum_sq / x.len() as f32 + eps).sqrt();
        for (v, &g) in x.iter_mut().zip(gamma.iter()) {
            *v = *v * inv_rms * (1.0 + g);
        }
    }

    /// Brings a captured (pre-final-norm) target hidden to the form the MTP
    /// head's hidden-state input expects, per [`MtpHiddenTap`]: under
    /// `PostFinalNorm` the target's final RMSNorm is applied here (CPU, same
    /// shifted convention and scale buffer the GPU final norm uses); under
    /// `PreFinalNorm` the hidden is left untouched. Both MTP entry points call
    /// this after the QuaRot counter-rotation and before `pre_fc_norm_hidden`.
    fn mtp_apply_hidden_tap(&self, hidden: &mut [f32]) {
        if self.mtp_hidden_tap() != Some(MtpHiddenTap::PostFinalNorm) {
            return;
        }
        self.apply_final_norm_cpu(hidden);
    }

    /// The target's final RMSNorm on the CPU: the shifted `x / rms(x) * (1 + gamma)`
    /// convention with the engine's final-norm scale, the same one the GPU final norm
    /// uses. It is only meaningful on a hidden in the original (unrotated) basis of a
    /// checkpoint whose final-norm scale is stored, i.e. not a rotated checkpoint. Shared
    /// by [`Self::mtp_apply_hidden_tap`] and the `MtpTargetVerifier` hidden-state methods.
    pub(super) fn apply_final_norm_cpu(&self, hidden: &mut [f32]) {
        debug_assert!(self.engine.final_norm.length() as usize >= hidden.len() * 4);
        // SAFETY: `final_norm` is a StorageModeShared f32 buffer of `hidden_size`
        // values written at load; `hidden` has `hidden_size` values.
        let gamma = unsafe {
            std::slice::from_raw_parts(
                self.engine.final_norm.contents() as *const f32,
                hidden.len(),
            )
        };
        Self::mtp_pre_fc_rmsnorm(hidden, gamma, self.engine.config.rms_norm_eps);
    }

    /// Draft one extra token using the MTP module.
    ///
    /// Runs the single MTP attention+MLP layer on the target model's hidden
    /// state to predict `pending_token + 1`. The hidden state is captured before
    /// the target's final norm (`self.session.last_pre_final_hidden`); the head
    /// receives it after the final norm on unrotated checkpoints and before it
    /// on rotated ones (see [`MtpHiddenTap`], [`Self::mtp_apply_hidden_tap`]).
    /// Returns the draft token id and logits.
    ///
    /// Panics if `self.session.mtp` is None.
    fn mtp_forward_one_dispatch(
        &mut self,
        pending_token: u32,
        position: usize,
    ) -> MetalMtpForwardOutput {
        let cfg = self.engine.config.clone();
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let num_q_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let q_dim = cfg.full_q_dim();
        let kv_dim = cfg.full_kv_dim();
        let half_rope_dim = (cfg.rope_dim() / 2) as u32;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        // ---- Phase 1: CPU ----

        // 1. Embedding lookup, same token as was just generated.
        let mut normed_embed = vec![0.0f32; hidden];
        assert!((pending_token as usize) < cfg.vocab_size);
        // SAFETY: normed_embed holds hidden f32 values, disjoint from the embedding table.
        unsafe {
            self.engine
                .embed_tokens
                .write_row(pending_token as usize, &mut normed_embed);
        }

        if let Some(ref rot) = self.engine.quarot_rotation {
            // Rotation dim is built from `cfg.hidden_size` at session construction
            // (see `from_q4_dir` in `weights.rs`), and `normed_embed` is `vec![0.0f32; hidden]`,
            // so dimensions cannot mismatch here. The `Result` API is preserved for
            // type compatibility; `debug_assert` catches programming errors in tests.
            debug_assert_eq!(rot.dim(), normed_embed.len());
            let _ = rot.apply_inverse(&mut normed_embed);
        }

        // 2. CPU RMSNorm of embedding using pre_fc_norm_embedding weights.
        {
            let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
            let gamma = unsafe {
                std::slice::from_raw_parts(
                    mtp_weights.pre_fc_norm_embedding.contents() as *const f32,
                    hidden,
                )
            };
            Self::mtp_pre_fc_rmsnorm(&mut normed_embed, gamma, cfg.rms_norm_eps);
        }

        // 3. CPU RMSNorm of the target hidden using pre_fc_norm_hidden weights. The
        // captured hidden is pre-final-norm; `mtp_apply_hidden_tap` applies the
        // target's final norm first on unrotated checkpoints.
        let mut normed_hidden = self.session.last_pre_final_hidden.clone();
        if let Some(ref rot) = self.engine.quarot_rotation {
            // See note above: rotation dim == hidden_size by construction.
            debug_assert_eq!(rot.dim(), normed_hidden.len());
            let _ = rot.apply_inverse(&mut normed_hidden);
        }
        self.mtp_apply_hidden_tap(&mut normed_hidden);
        {
            let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
            let gamma = unsafe {
                std::slice::from_raw_parts(
                    mtp_weights.pre_fc_norm_hidden.contents() as *const f32,
                    hidden,
                )
            };
            Self::mtp_pre_fc_rmsnorm(&mut normed_hidden, gamma, cfg.rms_norm_eps);
        }

        // 4. CPU concat [normed_embed || normed_hidden] into fused buffer (2*hidden).
        {
            let mtp = self.session.mtp.as_ref().unwrap();
            let dst = mtp.activations.fused.contents() as *mut f32;
            unsafe {
                for (i, &v) in normed_embed.iter().enumerate() {
                    *dst.add(i) = v;
                }
                for (i, &v) in normed_hidden.iter().enumerate() {
                    *dst.add(hidden + i) = v;
                }
            }
        }

        // ---- Phase 2: GPU ----
        // Collect raw pointers before encoder to avoid simultaneous borrow conflicts.
        let (
            w_fc,
            w_input_ln,
            w_post_attn_ln,
            w_q_proj,
            w_k_proj,
            w_v_proj,
            w_o_proj,
            w_q_norm,
            w_k_norm,
            w_gate_proj,
            w_up_proj,
            w_down_proj,
            w_norm,
        ) = {
            let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
            let lw = &mtp_weights.layers[0];
            (
                &mtp_weights.fc as *const Buffer,
                &lw.input_layernorm as *const Buffer,
                &lw.post_attention_layernorm as *const Buffer,
                &lw.q_proj as *const Buffer,
                &lw.k_proj as *const Buffer,
                &lw.v_proj as *const Buffer,
                &lw.o_proj as *const Buffer,
                &lw.q_norm as *const Buffer,
                &lw.k_norm as *const Buffer,
                &lw.mlp.gate_proj as *const Buffer,
                &lw.mlp.up_proj as *const Buffer,
                &lw.mlp.down_proj as *const Buffer,
                &mtp_weights.norm as *const Buffer,
            )
        };
        let (
            buf_fused,
            buf_hidden,
            buf_residual,
            buf_q,
            buf_q_sep,
            buf_gate_z,
            buf_k,
            buf_v,
            buf_attn_out,
            buf_gate,
            buf_up,
            buf_ffn_out,
            buf_logits,
        ) = {
            let a = &self.session.mtp.as_ref().unwrap().activations;
            (
                &a.fused as *const Buffer,
                &a.hidden as *const Buffer,
                &a.residual as *const Buffer,
                &a.q as *const Buffer,
                &a.q_separated as *const Buffer,
                &a.gate_z as *const Buffer,
                &a.k as *const Buffer,
                &a.v as *const Buffer,
                &a.attn_out as *const Buffer,
                &a.gate as *const Buffer,
                &a.up as *const Buffer,
                &a.ffn_out as *const Buffer,
                &a.logits as *const Buffer,
            )
        };
        let (buf_k_cache, buf_v_cache, mtp_seq_len) = {
            let c = &self.session.mtp.as_ref().unwrap().cache;
            (
                &c.k_buf as *const Buffer,
                &c.v_buf as *const Buffer,
                c.seq_len,
            )
        };

        let cmd = self.engine.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let wg = 256u64;

        unsafe {
            // fc: fused (2*hidden) → hidden
            // ADR-051 Phase 1: MTP projections are f16; `dispatch_matmul_half`
            // expects (a: f32 act, b: f16 weight, c: f32 out) with the M=1
            // decode GEMV pipeline (`gemv_decode_m1`).
            self.dispatch_matmul_half(
                enc,
                &*buf_fused,
                &*w_fc,
                &*buf_hidden,
                1,
                hidden as u32,
                (2 * hidden) as u32,
            );

            // Pre-attention: copy hidden → residual, RMSNorm hidden in-place.
            self.dispatch_copy_and_rms_norm(
                enc,
                &*buf_hidden,
                &*buf_residual,
                &*w_input_ln,
                hidden as u32,
                cfg.rms_norm_eps,
            );

            // Q/K/V projections (f16 weights, Phase 1).
            self.dispatch_matmul_half(
                enc,
                &*buf_hidden,
                &*w_q_proj,
                &*buf_q,
                1,
                (2 * q_dim) as u32,
                hidden as u32,
            );
            self.dispatch_matmul_half(
                enc,
                &*buf_hidden,
                &*w_k_proj,
                &*buf_k,
                1,
                kv_dim as u32,
                hidden as u32,
            );
            self.dispatch_matmul_half(
                enc,
                &*buf_hidden,
                &*w_v_proj,
                &*buf_v,
                1,
                kv_dim as u32,
                hidden as u32,
            );

            // Scatter Q+gate from q → q_separated, gate_z (inline).
            enc.set_compute_pipeline_state(&self.engine.pipelines.scatter_q_gate);
            enc.set_buffer(0, Some(&*buf_q), 0);
            enc.set_buffer(1, Some(&*buf_q_sep), 0);
            enc.set_buffer(2, Some(&*buf_gate_z), 0);
            let nh = num_q_heads as u32;
            let hd = head_dim as u32;
            enc.set_bytes(3, 4, &nh as *const u32 as *const _);
            enc.set_bytes(4, 4, &hd as *const u32 as *const _);
            enc.dispatch_threads(
                MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );

            // Per-head RMS norm on Q and K.
            self.dispatch_per_head_rms_norm(
                enc,
                &*buf_q_sep,
                &*w_q_norm,
                num_q_heads as u32,
                hd,
                cfg.rms_norm_eps,
            );
            self.dispatch_per_head_rms_norm(
                enc,
                &*buf_k,
                &*w_k_norm,
                num_kv_heads as u32,
                hd,
                cfg.rms_norm_eps,
            );

            // Partial RoPE. MTP speculative decoding is text-only in this
            // stage (ADR-069 RECON §5 open question); no M-RoPE override.
            self.dispatch_partial_rope(
                enc,
                &*buf_q_sep,
                num_q_heads as u32,
                hd,
                half_rope_dim,
                position as u32,
                None,
            );
            self.dispatch_partial_rope(
                enc,
                &*buf_k,
                num_kv_heads as u32,
                hd,
                half_rope_dim,
                position as u32,
                None,
            );

            // Append K,V to MTP KV cache.
            let kv_cache_off = (mtp_seq_len * kv_dim) as u32;
            self.dispatch_copy_offset(enc, &*buf_k, &*buf_k_cache, kv_dim as u32, kv_cache_off);
            self.dispatch_copy_offset(enc, &*buf_v, &*buf_v_cache, kv_dim as u32, kv_cache_off);

            // Flash decode attention (direct path — MTP cache is always short).
            let cur_mtp_seq = (mtp_seq_len + 1) as u32;
            enc.set_compute_pipeline_state(&self.engine.pipelines.decode_attention);
            enc.set_buffer(0, Some(&*buf_q_sep), 0);
            enc.set_buffer(1, Some(&*buf_k_cache), 0);
            enc.set_buffer(2, Some(&*buf_v_cache), 0);
            enc.set_buffer(3, Some(&*buf_attn_out), 0);
            enc.set_bytes(4, 4, &cur_mtp_seq as *const u32 as *const _);
            enc.set_bytes(5, 4, &hd as *const u32 as *const _);
            let nq = num_q_heads as u32;
            let nkv = num_kv_heads as u32;
            let qd = q_dim as u32;
            let kvd = kv_dim as u32;
            enc.set_bytes(6, 4, &nq as *const u32 as *const _);
            enc.set_bytes(7, 4, &nkv as *const u32 as *const _);
            enc.set_bytes(8, 4, &qd as *const u32 as *const _);
            enc.set_bytes(9, 4, &kvd as *const u32 as *const _);
            enc.set_bytes(10, 4, &scale as *const f32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(num_kv_heads as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );

            // Sigmoid gate: attn_out *= sigmoid(gate_z) (inline).
            enc.set_compute_pipeline_state(&self.engine.pipelines.sigmoid_gate);
            enc.set_buffer(0, Some(&*buf_attn_out), 0);
            enc.set_buffer(1, Some(&*buf_gate_z), 0);
            let qd_u32 = q_dim as u32;
            enc.set_bytes(2, 4, &qd_u32 as *const u32 as *const _);
            enc.dispatch_threads(
                MTLSize::new(div_ceil(q_dim as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );

            // O projection: attn_out → ffn_out (f16 weights, Phase 1).
            self.dispatch_matmul_half(
                enc,
                &*buf_attn_out,
                &*w_o_proj,
                &*buf_ffn_out,
                1,
                hidden as u32,
                q_dim as u32,
            );
            // Copy ffn_out → attn_out for use as delta in fused residual+norm.
            self.dispatch_copy(enc, &*buf_ffn_out, &*buf_attn_out, hidden as u32);

            // Post-attention residual + norm (MLP input).
            self.dispatch_fused_residual_add_norm(
                enc,
                &*buf_residual,
                &*buf_attn_out,
                &*buf_residual,
                &*buf_hidden,
                &*w_post_attn_ln,
                hidden as u32,
                cfg.rms_norm_eps,
            );

            // Dense MLP (f16 weights, Phase 1).
            self.dispatch_matmul_half(
                enc,
                &*buf_hidden,
                &*w_gate_proj,
                &*buf_gate,
                1,
                inter as u32,
                hidden as u32,
            );
            self.dispatch_matmul_half(
                enc,
                &*buf_hidden,
                &*w_up_proj,
                &*buf_up,
                1,
                inter as u32,
                hidden as u32,
            );
            // SiLU-mul (inline).
            enc.set_compute_pipeline_state(&self.engine.pipelines.silu_mul);
            enc.set_buffer(0, Some(&*buf_gate), 0);
            enc.set_buffer(1, Some(&*buf_up), 0);
            let inter_u32 = inter as u32;
            enc.set_bytes(2, 4, &inter_u32 as *const u32 as *const _);
            enc.dispatch_threads(
                MTLSize::new(div_ceil(inter as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
            self.dispatch_matmul_half(
                enc,
                &*buf_gate,
                &*w_down_proj,
                &*buf_ffn_out,
                1,
                hidden as u32,
                inter as u32,
            );

            // MLP residual add.
            self.dispatch_add_and_copy(
                enc,
                &*buf_ffn_out,
                &*buf_residual,
                &*buf_hidden,
                hidden as u32,
            );

            // Final RMSNorm.
            self.dispatch_rms_norm(
                enc,
                &*buf_hidden,
                &*w_norm,
                hidden as u32,
                1,
                cfg.rms_norm_eps,
            );
        }

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        if let Some(ref rot) = self.engine.quarot_rotation {
            // SAFETY: `buf_hidden` is the MTP `hidden` activation buffer, allocated
            // with capacity `hidden` f32 elements at session construction. The
            // command buffer that wrote it (`cmd`) has been awaited above via
            // `wait_until_completed()`, so the contents are visible to the host.
            let mut h_out = unsafe { read_buffer(&*buf_hidden, hidden) };
            debug_assert_eq!(rot.dim(), h_out.len());
            let _ = rot.apply(&mut h_out);
            // SAFETY: same buffer, no other command buffer in flight. We write
            // back the rotated values before the next command buffer (`cmd2`) is
            // submitted on the next line.
            unsafe { write_buffer(&*buf_hidden, &h_out) };
        }

        let cmd2 = self.engine.queue.new_command_buffer();
        let enc2 = cmd2.new_compute_command_encoder();
        unsafe {
            // Logits GEMV using shared lm_head (embed_tokens_q8).
            self.dispatch_gemm(
                enc2,
                &*buf_hidden,
                0,
                &self.engine.embed_tokens_q8,
                &*buf_logits,
                0,
                1,
                cfg.vocab_size as u32,
                hidden as u32,
            );
        }
        enc2.end_encoding();
        cmd2.commit();
        cmd2.wait_until_completed();

        // ---- Phase 3: CPU ----
        // PR #787: this was the one surviving
        // production last-wins `max_by` argmax after the #280 fix -- the
        // MTP draft-token pick, called from `generate_greedy_mtp`. Swapped
        // for the shared first-wins `argmax_f32_first_wins` helper so a
        // tie at the draft boundary resolves the same way as every other
        // greedy pick in the crate (a differing draft tie can only reduce
        // speculative acceptance, never change final target output, but
        // the duplicate last-wins contract itself is the defect).
        let logits = unsafe { read_buffer(&*buf_logits, cfg.vocab_size) };
        let draft_token = crate::sampling::argmax_f32_first_wins(&logits);

        // Advance MTP KV cache.
        if let Some(ref mut mtp) = self.session.mtp {
            mtp.cache.seq_len += 1;
        }

        MetalMtpForwardOutput {
            token_id: draft_token,
            logits,
        }
    }

    /// Rebuilds the MTP draft head's KV cache from the prompt just processed by
    /// `forward_prefill`/`forward_prefill_all_logits` (#1340).
    ///
    /// Without this, `reset_state` zeroes the MTP cache and the only prefill on
    /// the Metal path fills the *target* model's cache, so the first
    /// `mtp_forward_one` call of a generation dispatches attention with
    /// `cache_len = 1` — the draft can only ever attend to the row it just
    /// wrote, independent of the prompt.
    ///
    /// `mtp_forward_one(pending_token, pos)` pairs `pending_token`'s embedding
    /// with `last_pre_final_hidden` — the target's hidden state, captured before
    /// the final norm, from the position *before* `pending_token`, i.e. the
    /// hidden state that predicted it. This mirrors that pairing across the whole
    /// prompt: for each position `p` in `1..prompt_ids.len()`, it pairs
    /// `prompt_ids[p]` with the target's captured hidden state at position
    /// `p - 1`, stored in `self.session.mtp_prefill_hidden` by
    /// `forward_prefill_batched_chunk`. Each append then brings the captured
    /// hidden to the head's input per [`MtpHiddenTap`], exactly as the live
    /// draft does.
    /// Position 0 has no predecessor hidden state (nothing came before the first
    /// prompt token) and is intentionally not represented in the MTP cache, so a
    /// prompt of length `L` yields `L - 1` prefilled entries; the live call for
    /// the first generated token then appends the `L`-th, giving that call's
    /// attention dispatch `cache_len == L` instead of `1`.
    ///
    /// No-op when the session has no MTP head, when the prompt is too short to
    /// yield a pair, or when `mtp_prefill_hidden` was not populated for this
    /// exact prompt (e.g. the LoRA-active prefill path, which forwards token by
    /// token via `forward_step` and does not feed the accumulator) — captured
    /// length is asserted against `prompt_ids.len() * hidden_size` and anything
    /// else is treated as "nothing to prefill" rather than misinterpreted.
    pub(super) fn mtp_prefill(&mut self, prompt_ids: &[u32]) {
        let captured = std::mem::take(&mut self.session.mtp_prefill_hidden);
        if self.session.mtp.is_none() {
            return;
        }
        let hidden = self.engine.config.hidden_size;
        let n = prompt_ids.len();
        if n < 2 || captured.len() != n * hidden {
            return;
        }
        for p in 1..n {
            let hidden_in = &captured[(p - 1) * hidden..p * hidden];
            self.mtp_prefill_append(prompt_ids[p], hidden_in, p);
        }
    }

    // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
    // dispatch creates command buffers on the same long-lived worker thread.
    // Unmeasured individually; fixed on the structural argument, not on numbers.
    pub(super) fn mtp_prefill_append(&mut self, token_id: u32, hidden_in: &[f32], position: usize) {
        objc::rc::autoreleasepool(|| {
            self.mtp_prefill_append_dispatch(token_id, hidden_in, position)
        })
    }

    /// Appends one MTP cache entry (K/V only — no Q, attention, MLP, or logits)
    /// for `token_id` at absolute position `position`, using `hidden_in` (the
    /// target's hidden state captured before the final norm) in place of
    /// `self.session.last_pre_final_hidden`; the head's hidden-state input is
    /// derived from it per [`MtpHiddenTap`]. Mirrors the CPU phase and GPU
    /// dispatch sequence of `mtp_forward_one` up through the KV cache append —
    /// everything after that point (attention, gating, MLP, logits) only matters
    /// for a position acting as a *query*, and a prefilled position is only ever
    /// a future *key*.
    fn mtp_prefill_append_dispatch(&mut self, token_id: u32, hidden_in: &[f32], position: usize) {
        let cfg = self.engine.config.clone();
        let hidden = cfg.hidden_size;
        let kv_dim = cfg.full_kv_dim();
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim as u32;
        let half_rope_dim = (cfg.rope_dim() / 2) as u32;

        // ---- Phase 1: CPU (mirrors mtp_forward_one's embedding + hidden norm + fuse) ----
        let mut normed_embed = vec![0.0f32; hidden];
        assert!((token_id as usize) < cfg.vocab_size);
        unsafe {
            // SAFETY: normed_embed holds hidden f32 values, disjoint from the table.
            self.engine
                .embed_tokens
                .write_row(token_id as usize, &mut normed_embed);
        }
        if let Some(ref rot) = self.engine.quarot_rotation {
            debug_assert_eq!(rot.dim(), normed_embed.len());
            let _ = rot.apply_inverse(&mut normed_embed);
        }
        {
            let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
            let gamma = unsafe {
                std::slice::from_raw_parts(
                    mtp_weights.pre_fc_norm_embedding.contents() as *const f32,
                    hidden,
                )
            };
            Self::mtp_pre_fc_rmsnorm(&mut normed_embed, gamma, cfg.rms_norm_eps);
        }

        let mut normed_hidden = hidden_in.to_vec();
        if let Some(ref rot) = self.engine.quarot_rotation {
            debug_assert_eq!(rot.dim(), normed_hidden.len());
            let _ = rot.apply_inverse(&mut normed_hidden);
        }
        self.mtp_apply_hidden_tap(&mut normed_hidden);
        {
            let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
            let gamma = unsafe {
                std::slice::from_raw_parts(
                    mtp_weights.pre_fc_norm_hidden.contents() as *const f32,
                    hidden,
                )
            };
            Self::mtp_pre_fc_rmsnorm(&mut normed_hidden, gamma, cfg.rms_norm_eps);
        }

        {
            let mtp = self.session.mtp.as_ref().unwrap();
            let dst = mtp.activations.fused.contents() as *mut f32;
            unsafe {
                for (i, &v) in normed_embed.iter().enumerate() {
                    *dst.add(i) = v;
                }
                for (i, &v) in normed_hidden.iter().enumerate() {
                    *dst.add(hidden + i) = v;
                }
            }
        }

        // ---- Phase 2: GPU (K/V-only: fc, input-norm, k/v proj, k-norm, rope, cache append) ----
        let (w_fc, w_input_ln, w_k_proj, w_v_proj, w_k_norm) = {
            let mtp_weights = self.engine.mtp_weights.as_ref().unwrap();
            let lw = &mtp_weights.layers[0];
            (
                &mtp_weights.fc as *const Buffer,
                &lw.input_layernorm as *const Buffer,
                &lw.k_proj as *const Buffer,
                &lw.v_proj as *const Buffer,
                &lw.k_norm as *const Buffer,
            )
        };
        let (buf_fused, buf_hidden, buf_residual, buf_k, buf_v) = {
            let a = &self.session.mtp.as_ref().unwrap().activations;
            (
                &a.fused as *const Buffer,
                &a.hidden as *const Buffer,
                &a.residual as *const Buffer,
                &a.k as *const Buffer,
                &a.v as *const Buffer,
            )
        };
        let (buf_k_cache, buf_v_cache, mtp_seq_len) = {
            let c = &self.session.mtp.as_ref().unwrap().cache;
            (
                &c.k_buf as *const Buffer,
                &c.v_buf as *const Buffer,
                c.seq_len,
            )
        };

        let cmd = self.engine.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        unsafe {
            self.dispatch_matmul_half(
                enc,
                &*buf_fused,
                &*w_fc,
                &*buf_hidden,
                1,
                hidden as u32,
                (2 * hidden) as u32,
            );
            self.dispatch_copy_and_rms_norm(
                enc,
                &*buf_hidden,
                &*buf_residual,
                &*w_input_ln,
                hidden as u32,
                cfg.rms_norm_eps,
            );
            self.dispatch_matmul_half(
                enc,
                &*buf_hidden,
                &*w_k_proj,
                &*buf_k,
                1,
                kv_dim as u32,
                hidden as u32,
            );
            self.dispatch_matmul_half(
                enc,
                &*buf_hidden,
                &*w_v_proj,
                &*buf_v,
                1,
                kv_dim as u32,
                hidden as u32,
            );
            self.dispatch_per_head_rms_norm(
                enc,
                &*buf_k,
                &*w_k_norm,
                num_kv_heads as u32,
                head_dim,
                cfg.rms_norm_eps,
            );
            self.dispatch_partial_rope(
                enc,
                &*buf_k,
                num_kv_heads as u32,
                head_dim,
                half_rope_dim,
                position as u32,
                None,
            );
            let kv_cache_off = (mtp_seq_len * kv_dim) as u32;
            self.dispatch_copy_offset(enc, &*buf_k, &*buf_k_cache, kv_dim as u32, kv_cache_off);
            self.dispatch_copy_offset(enc, &*buf_v, &*buf_v_cache, kv_dim as u32, kv_cache_off);
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        if let Some(ref mut mtp) = self.session.mtp {
            mtp.cache.seq_len += 1;
        }
    }
}
