use super::{
    GDN_CHUNK_SIZE, GdnChunkParams, GdnChunkedWeights, MetalCommonLayerWeights, MetalFfnWeights,
    MetalLayerAttnWeights, MetalQwen35State, Q4_0_BLOCK_SIZE, Q8_0_BLOCK_SIZE, QK4_0, QK8_0,
    QuantFormat, Qwen35Config, div_ceil,
};
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

impl MetalQwen35State {
    /// **Unstable**: fallible batch prompt prefill; kernel strategy may change.
    ///
    /// Returns logits for the last token while advancing KV and GDN state. This
    /// is the non-panicking raw-prefill route; existing callers of
    /// [`Self::forward_prefill`] can migrate by handling the returned
    /// [`Result`]. Empty input retains `forward_prefill`'s existing zero-logit
    /// result.
    ///
    /// # Fresh-prompt semantics
    ///
    /// This call always dispatches from position 0. It requires a fresh
    /// session (`kv_cache.seq_len == 0` and GDN recurrent state at its initial
    /// condition) and does not reset on the caller's behalf. Call
    /// [`Self::reset_state`] before starting a new prompt on a reused session.
    ///
    /// # Errors
    ///
    /// Returns [`lattice_inference_core::InferenceError::InvalidInput`] before GPU
    /// dispatch for a non-fresh session, an out-of-vocabulary token, or a token
    /// range beyond the session capacity. Rejection leaves session state unchanged.
    pub fn try_forward_prefill(
        &mut self,
        token_ids: &[u32],
    ) -> Result<Vec<f32>, lattice_inference_core::InferenceError> {
        self.check_raw_prefill_fresh_session("try_forward_prefill")?;
        self.check_forward_token_ids("try_forward_prefill", token_ids)?;
        self.check_forward_range_capacity(0, token_ids.len(), false)?;
        self.cross_turn_prefix_cache.clear();
        Ok(self.forward_prefill_impl(token_ids, false, Vec::new()))
    }

    /// **Unstable**: batch prompt prefill; prefill kernel and fallback threshold may change.
    ///
    /// Returns logits for the LAST token while updating KV cache and GDN state.
    /// Dense models use batched GEMM in max_prefill-sized chunks, including
    /// active LoRA adapters. Models with MoE layers use per-token
    /// [`Self::forward_step`] calls, with or without an adapter.
    ///
    /// # Fresh-prompt semantics
    ///
    /// This call always dispatches from position 0. It requires a fresh
    /// session (`kv_cache.seq_len == 0` and GDN recurrent state at its
    /// initial condition), does not reset state on the caller's behalf,
    /// and does not append to an existing prefix. Call [`Self::reset_state`]
    /// before reusing a session for a new prompt.
    ///
    /// # Panics
    ///
    /// Panics when [`Self::try_forward_prefill`] returns an error: the session
    /// is not fresh, a token is outside the vocabulary, or the token range
    /// exceeds capacity. New code should use the fallible route and handle its
    /// typed input error. This compatibility wrapper preserves the existing
    /// return type for callers that treat those conditions as contract
    /// violations.
    ///
    /// # Cross-turn cache invalidation (#516)
    ///
    /// Public raw-forward entry point — see [`Self::forward_step`]'s doc
    /// comment for the invariant this enforces. The clear happens inside
    /// [`Self::try_forward_prefill`], which this wraps. No production caller in
    /// this crate remains — the generation entry points prefill through the
    /// fallible route directly — so this wrapper's callers are this module's own
    /// tests, the bench and profiling examples, and any external consumer that
    /// wants the panicking form.
    pub fn forward_prefill(&mut self, token_ids: &[u32]) -> Vec<f32> {
        match self.try_forward_prefill(token_ids) {
            Ok(logits) => logits,
            Err(error) => panic!("{error}"),
        }
    }

    /// **Unstable**: score a list of option letters with one prompt prefill (ADR-097).
    ///
    /// Resets the recurrent and KV state, runs exactly one
    /// [`Self::try_forward_prefill`] over `prompt_ids` and reads the last-position logits at
    /// `letter_ids`. There is no sampling and no decode step. The returned
    /// [`OptionScores`](lattice_compute::option_scoring::OptionScores) holds the k raw logits in the
    /// caller's order, their softmax over the k, and the full-vocabulary mass on the k letters.
    ///
    /// The scorer never loads or unloads an adapter: whichever adapter the state holds is the
    /// one in effect. Dense models apply an active adapter inside batched prefill.
    /// Models with MoE layers use per-token steps, with or without an adapter.
    /// Because the call resets the state first, any session that was in progress is
    /// discarded, and the state is left advanced by the prompt, so the next call resets again.
    ///
    /// # Errors
    ///
    /// Returns [`lattice_inference_core::InferenceError::InvalidInput`] before any state reset or GPU
    /// dispatch for an empty `prompt_ids`, a prompt token outside the vocabulary, a prompt
    /// longer than the session capacity, an empty `letter_ids`, a letter id outside the
    /// vocabulary or a repeated letter id; a rejected call leaves the session untouched.
    /// Returns [`lattice_inference_core::InferenceError::Inference`] when the logits contain
    /// a non-finite value.
    pub fn score_option_letters(
        &mut self,
        prompt_ids: &[u32],
        letter_ids: &[u32],
    ) -> Result<lattice_compute::option_scoring::OptionScores, lattice_inference_core::InferenceError>
    {
        if prompt_ids.is_empty() {
            return Err(lattice_inference_core::InferenceError::InvalidInput(
                "score_option_letters: prompt_ids must not be empty".into(),
            ));
        }
        lattice_compute::option_scoring::check_letter_ids(
            self.engine.config.vocab_size,
            letter_ids,
        )?;
        self.check_forward_token_ids("score_option_letters", prompt_ids)?;
        self.check_forward_range_capacity(0, prompt_ids.len(), false)?;
        self.reset_state();
        let logits = self.try_forward_prefill(prompt_ids)?;
        lattice_compute::option_scoring::option_scores_from_logits(&logits, letter_ids)
    }

    /// **Unstable**: fallible prompt prefill with explicit hidden readback.
    ///
    /// Returns `(last_token_logits, pre_final_hidden)`. The hidden row is
    /// captured for the prompt's last token before final RMSNorm. Batched,
    /// chunked, one-token, and MoE per-token paths advance KV/GDN state by
    /// exactly the same token range as [`Self::forward_prefill`].
    ///
    /// # Fresh-prompt semantics
    ///
    /// This call always dispatches from position 0. It requires a fresh
    /// session (`kv_cache.seq_len == 0` and GDN recurrent state at its
    /// initial condition) and rejects otherwise — it does not reset the
    /// session on the caller's behalf, and it does not implement append
    /// semantics against an existing prefix. Calling it against a session
    /// with live state would overwrite the existing KV prefix rows while
    /// leaving GDN recurrent state conditioned on the tokens it overwrote,
    /// an unreconstructible mix. Call [`Self::reset_state`] first to start a
    /// new prompt on a session that has already generated.
    ///
    /// # Errors
    ///
    /// Returns [`lattice_inference_core::InferenceError::InvalidInput`] before GPU
    /// dispatch for an empty prompt, an out-of-vocabulary token, a prompt
    /// whose token range exceeds the session capacity, or a session that is
    /// not fresh (see above). On rejection, session state (KV cache, GDN state,
    /// hidden-readback counters) is left unchanged.
    pub fn forward_prefill_with_hidden(
        &mut self,
        token_ids: &[u32],
    ) -> Result<(Vec<f32>, Vec<f32>), lattice_inference_core::InferenceError> {
        use lattice_inference_core::InferenceError;

        if token_ids.is_empty() {
            return Err(InferenceError::InvalidInput(
                "forward_prefill_with_hidden: token_ids must not be empty".into(),
            ));
        }
        self.check_forward_token_ids("forward_prefill_with_hidden", token_ids)?;
        self.check_hidden_prefill_fresh_session()?;
        self.check_forward_range_capacity(0, token_ids.len(), false)?;
        self.cross_turn_prefix_cache.clear();

        if token_ids.len() == 1 {
            return self.forward_step_with_hidden(token_ids[0], 0);
        }
        if self.engine.has_moe_layer {
            // The batched schedulers have no MoE schedule; per-token steps do.
            let last_index = token_ids.len() - 1;
            for (position, &token_id) in token_ids[..last_index].iter().enumerate() {
                self.try_forward_step(token_id, position)?;
            }
            return self.forward_step_with_hidden(token_ids[last_index], last_index);
        }

        let max_prefill = self.session.max_prefill;
        if token_ids.len() <= max_prefill {
            let logits = self.forward_prefill_batched_chunk(token_ids, 0, false, true, true);
            return Ok((logits, self.session.last_pre_final_hidden.clone()));
        }

        let mut start_pos = 0usize;
        let mut last_logits = Vec::new();
        for chunk in token_ids.chunks(max_prefill) {
            let is_last = start_pos + chunk.len() == token_ids.len();
            last_logits =
                self.forward_prefill_batched_chunk(chunk, start_pos, false, is_last, is_last);
            start_pos += chunk.len();
        }
        Ok((last_logits, self.session.last_pre_final_hidden.clone()))
    }

    /// **Unstable**: prefill that returns logits for ALL `n` positions, not
    /// just the last. Output is `n * vocab_size` f32 values, row-major:
    /// `logits[i * vocab_size .. (i+1) * vocab_size]` predicts `tokens[i+1]`.
    /// Models with MoE layers collect per-token logits in the same row-major layout.
    ///
    /// Used by perplexity evaluation. Allocates a per-call buffer of size
    /// `n * vocab_size * 4` bytes — for `vocab_size=248K` and `n=128` that's
    /// ~127MB. Callers should cap `n` accordingly (typical PPL window: 128).
    ///
    /// # Fresh-prompt semantics
    ///
    /// This call always dispatches from position 0. It requires a fresh
    /// session (`kv_cache.seq_len == 0` and GDN recurrent state at its
    /// initial condition), does not reset state on the caller's behalf,
    /// and does not append to an existing prefix. Call [`Self::reset_state`]
    /// before reusing a session for a new prompt.
    ///
    /// # Panics
    ///
    /// Panics if any `token_ids[i] >= vocab_size`. See [`forward_prefill`] for details.
    ///
    /// # Errors
    ///
    /// Returns [`lattice_inference_core::InferenceError::InvalidInput`] before GPU
    /// dispatch when the session is not fresh (see above). Returns an error
    /// when more than one position is requested while a LoRA adapter is active,
    /// because the batched all-position path does not apply LoRA projections.
    /// MoE all-position output also returns `InvalidInput` if its aggregate
    /// buffer size is not representable or cannot be reserved.
    ///
    /// # Cross-turn cache invalidation (#516)
    ///
    /// Public raw-forward entry point — see [`Self::forward_step`]'s doc
    /// comment for the invariant this enforces.
    pub fn forward_prefill_all_logits(
        &mut self,
        token_ids: &[u32],
    ) -> Result<Vec<f32>, lattice_inference_core::InferenceError> {
        self.check_raw_prefill_fresh_session("forward_prefill_all_logits")?;
        if token_ids.len() > 1 && self.lora.is_some() {
            return Err(lattice_inference_core::InferenceError::Inference(
                "forward_prefill_all_logits: all-position prefill does not apply LoRA \
                     adapters; unload the adapter before requesting all-position logits"
                    .into(),
            ));
        }
        let moe_logits = if self.engine.has_moe_layer && token_ids.len() > 1 {
            Self::prepare_moe_prefill_logits(token_ids.len(), self.engine.config.vocab_size)?
        } else {
            Vec::new()
        };
        self.cross_turn_prefix_cache.clear();
        Ok(self.forward_prefill_impl(token_ids, true, moe_logits))
    }

    fn forward_prefill_impl(
        &mut self,
        token_ids: &[u32],
        all_positions: bool,
        moe_logits: Vec<f32>,
    ) -> Vec<f32> {
        // Start of a fresh prompt pass: drop any hidden states captured by a
        // previous prefill so `mtp_prefill` never pairs this prompt's tokens
        // with a stale prior prompt's hidden states (#1340).
        self.session.mtp_prefill_hidden.clear();
        let n = token_ids.len();
        let vocab = self.engine.config.vocab_size;
        // Validate all ids once at the entry point (O(seq_len)) before any GPU work.
        // Do NOT duplicate this inside the hot embedding loop; the inner chunk code
        // relies on this pre-scan and omits the per-token guard.
        for (t, &id) in token_ids.iter().enumerate() {
            assert!(
                (id as usize) < vocab,
                "forward_prefill: token_ids[{t}]={id} out of range: vocab_size is {vocab}",
            );
        }
        if n == 0 {
            return if all_positions {
                vec![]
            } else {
                vec![0.0; vocab]
            };
        }
        if n == 1 {
            return if all_positions && self.engine.has_moe_layer {
                self.forward_step_with_dense_logits(token_ids[0], 0)
            } else {
                self.forward_step(token_ids[0], 0)
            };
        }
        assert!(
            n <= self.session.kv_cache.max_cache_len,
            "prefill length {} exceeds max_cache_len {}",
            n,
            self.session.kv_cache.max_cache_len
        );
        if self.engine.has_moe_layer {
            let mut logits = moe_logits;
            for (pos, &id) in token_ids.iter().enumerate() {
                let step_logits = if all_positions {
                    self.forward_step_with_dense_logits(id, pos)
                } else {
                    self.forward_step(id, pos)
                };
                if all_positions {
                    logits.extend(step_logits);
                } else {
                    logits = step_logits;
                }
            }
            return logits;
        }
        let max_prefill = self.session.max_prefill;
        if n <= max_prefill {
            // Only/last chunk — its logits are always the caller's answer.
            return self.forward_prefill_batched_chunk(token_ids, 0, all_positions, true, false);
        }
        // Chunked batched prefill: each chunk is one command buffer (preserving the
        // n≤512 fast path within each chunk). GDN recurrent state threads across
        // boundaries automatically — session GPU buffers are mutated in place and
        // never reset between chunks within a single request.
        if all_positions {
            let mut all_logits = Vec::with_capacity(n * vocab);
            let mut start_pos = 0usize;
            for chunk in token_ids.chunks(max_prefill) {
                // Perplexity needs every chunk's per-position logits.
                all_logits.extend(
                    self.forward_prefill_batched_chunk(chunk, start_pos, true, true, false),
                );
                start_pos += chunk.len();
            }
            all_logits
        } else {
            let mut last_logits = Vec::new();
            let mut start_pos = 0usize;
            for chunk in token_ids.chunks(max_prefill) {
                // Only the LAST chunk's tail (RMSNorm+lm_head+MTP+top-k) is ever kept —
                // intermediate chunks still commit their layer work and advance
                // set_position, they just skip the terminal tail (Experiment B).
                let is_last = start_pos + chunk.len() == n;
                last_logits =
                    self.forward_prefill_batched_chunk(chunk, start_pos, false, is_last, false);
                start_pos += chunk.len();
            }
            last_logits
        }
    }

    pub(super) fn copy_prefill_embeddings(&self, token_ids: &[u32], hidden: usize) {
        // SAFETY: callers validate ids < vocab_size; hidden rows hold hidden f32 values.
        unsafe {
            let dst_base = self.session.activations.hidden.contents() as *mut f32;
            for (t, &id) in token_ids.iter().enumerate() {
                self.engine.embed_tokens.write_row(
                    id as usize,
                    std::slice::from_raw_parts_mut(dst_base.add(t * hidden), hidden),
                );
            }
        }
    }

    fn encode_prefill_dense_mlp(
        &self,
        enc: &ComputeCommandEncoderRef,
        common_w: &MetalCommonLayerWeights,
        attention_delta: &Buffer,
        cfg: &Qwen35Config,
        layer_idx: usize,
        m: u32,
    ) {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let (w_gate_up, w_down, gate_byte_size) = match &common_w.ffn {
            MetalFfnWeights::Dense {
                gate_up_proj,
                down_proj,
            } => {
                let byte_size = match self.engine.quant_format {
                    QuantFormat::Q8_0 => (inter * (hidden / QK8_0) * Q8_0_BLOCK_SIZE) as u64,
                    QuantFormat::Q4_0 => (inter * (hidden / QK4_0) * Q4_0_BLOCK_SIZE) as u64,
                };
                (gate_up_proj, down_proj, byte_size)
            }
            MetalFfnWeights::Moe(_) => {
                panic!("batch prefill does not support MoE layers when M>1")
            }
        };

        self.dispatch_fused_residual_add_norm_batch(
            enc,
            &self.session.activations.residual,
            attention_delta,
            &self.session.activations.residual,
            &self.session.activations.hidden,
            &common_w.post_attention_layernorm,
            hidden as u32,
            m,
            cfg.rms_norm_eps,
        );
        self.dispatch_gemm_at(
            enc,
            &self.session.activations.hidden,
            0,
            w_gate_up,
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
            w_gate_up,
            gate_byte_size,
            &self.session.activations.up,
            0,
            m,
            inter as u32,
            hidden as u32,
        );
        // Adapter deltas land on the separate gate / up matrices before silu_mul.
        self.dispatch_lora_rows_if_active(
            enc,
            &self.session.activations.hidden,
            0,
            hidden as u32,
            &self.session.activations.gate,
            0,
            inter as u32,
            m,
            layer_idx,
            "gate_proj",
        );
        self.dispatch_lora_rows_if_active(
            enc,
            &self.session.activations.hidden,
            0,
            hidden as u32,
            &self.session.activations.up,
            0,
            inter as u32,
            m,
            layer_idx,
            "up_proj",
        );
        self.dispatch_silu_mul(enc, m * inter as u32);
        self.dispatch_gemm(
            enc,
            &self.session.activations.gate,
            0,
            w_down,
            &self.session.activations.ffn_out,
            0,
            m,
            hidden as u32,
            inter as u32,
        );
        // down_proj reads the post-silu_mul gate matrix and adds into ffn_out before
        // the residual add.
        self.dispatch_lora_rows_if_active(
            enc,
            &self.session.activations.gate,
            0,
            inter as u32,
            &self.session.activations.ffn_out,
            0,
            hidden as u32,
            m,
            layer_idx,
            "down_proj",
        );
        self.dispatch_add_and_copy(
            enc,
            &self.session.activations.ffn_out,
            &self.session.activations.residual,
            &self.session.activations.hidden,
            m * hidden as u32,
        );
    }

    pub(super) fn encode_gdn_prefill_before_recurrence(
        &self,
        enc: &ComputeCommandEncoderRef,
        compact_idx: usize,
        layer_idx: usize,
        cfg: &Qwen35Config,
        m: u32,
    ) {
        let (attn_w, common_w) = &self.engine.layer_weights[compact_idx];
        let MetalLayerAttnWeights::Linear(gdn_w) = attn_w else {
            unreachable!()
        };
        self.dispatch_copy_and_rms_norm_batch(
            enc,
            &self.session.activations.hidden,
            &self.session.activations.residual,
            &common_w.input_layernorm,
            cfg.hidden_size as u32,
            m,
            cfg.rms_norm_eps,
        );
        self.dispatch_gemm(
            enc,
            &self.session.activations.hidden,
            0,
            &gdn_w.in_proj_qkv,
            &self.session.activations.gdn_qkv,
            0,
            m,
            cfg.linear_qkv_dim() as u32,
            cfg.hidden_size as u32,
        );
        self.dispatch_lora_rows_if_active(
            enc,
            &self.session.activations.hidden,
            0,
            cfg.hidden_size as u32,
            &self.session.activations.gdn_qkv,
            0,
            cfg.linear_qkv_dim() as u32,
            m,
            layer_idx,
            "in_proj_qkv",
        );
        self.dispatch_gemm(
            enc,
            &self.session.activations.hidden,
            0,
            &gdn_w.in_proj_z,
            &self.session.activations.gdn_z,
            0,
            m,
            cfg.linear_output_dim() as u32,
            cfg.hidden_size as u32,
        );
        // The recurrence reads and then overwrites gdn_z, so the in_proj_z delta must
        // be in place before it runs.
        self.dispatch_lora_rows_if_active(
            enc,
            &self.session.activations.hidden,
            0,
            cfg.hidden_size as u32,
            &self.session.activations.gdn_z,
            0,
            cfg.linear_output_dim() as u32,
            m,
            layer_idx,
            "in_proj_z",
        );
    }

    pub(super) fn encode_gdn_prefill_recurrence(
        &self,
        enc: &ComputeCommandEncoderRef,
        compact_idx: usize,
        linear_idx: usize,
        cfg: &Qwen35Config,
        n: usize,
        chunked_enabled: bool,
    ) {
        let MetalLayerAttnWeights::Linear(gdn_w) = &self.engine.layer_weights[compact_idx].0 else {
            unreachable!()
        };
        let chunk_params = if chunked_enabled {
            Self::make_gdn_chunk_params(cfg, n)
        } else {
            None
        };
        if let Some(params) = chunk_params {
            let weights = GdnChunkedWeights {
                conv1d_weight: &gdn_w.conv1d_weight,
                in_proj_b: &gdn_w.in_proj_b,
                in_proj_a: &gdn_w.in_proj_a,
                a_log: &gdn_w.a_log,
                dt_bias: &gdn_w.dt_bias,
                norm_weight: &gdn_w.norm_weight,
            };
            self.dispatch_gdn_chunked_prefill_layer(enc, linear_idx, weights, params);
            return;
        }

        let qkv_d = cfg.linear_qkv_dim();
        let out_d = cfg.linear_output_dim();
        let num_h = cfg.linear_num_key_heads;
        let key_d = cfg.linear_key_head_dim;
        let val_d = cfg.linear_value_head_dim;
        let num_vh = cfg.linear_num_value_heads();
        let q_total = (num_h * key_d) as u32;
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
        let params = GdnRecurParams {
            key_dim: key_d as u32,
            value_dim: val_d as u32,
            num_key_heads: num_h as u32,
            num_value_heads: num_vh as u32,
            hidden_size: cfg.hidden_size as u32,
            q_total,
            v_offset: q_total * 2,
            scale: 1.0 / (key_d as f32).sqrt(),
            eps: cfg.rms_norm_eps,
        };
        for t in 0..n {
            let qkv_off = (t * qkv_d) as u64 * 4;
            let z_off = (t * out_d) as u64 * 4;
            let h_off = (t * cfg.hidden_size) as u64 * 4;

            enc.set_compute_pipeline_state(&self.engine.pipelines.conv1d_silu);
            enc.set_buffer(
                0,
                Some(self.session.gdn_gpu_state.layer(linear_idx).conv_buffer()),
                0,
            );
            enc.set_buffer(1, Some(&self.session.activations.gdn_qkv), qkv_off);
            enc.set_buffer(2, Some(&gdn_w.conv1d_weight), 0);
            enc.set_buffer(3, Some(&self.session.gdn_gpu_conv_out), 0);
            let qd = qkv_d as u32;
            let ks = cfg.linear_conv_kernel_dim as u32;
            enc.set_bytes(4, 4, &qd as *const u32 as *const _);
            enc.set_bytes(5, 4, &ks as *const u32 as *const _);
            let wg = 256u64;
            enc.dispatch_threads(
                MTLSize::new(div_ceil(qkv_d as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );

            enc.set_compute_pipeline_state(&self.engine.pipelines.gdn_recurrence);
            enc.set_buffer(
                0,
                Some(self.session.gdn_gpu_state.layer(linear_idx).s_matrix()),
                0,
            );
            enc.set_buffer(1, Some(&self.session.gdn_gpu_conv_out), 0);
            enc.set_buffer(2, Some(&self.session.activations.gdn_z), z_off);
            enc.set_buffer(3, Some(&self.session.activations.hidden), h_off);
            enc.set_buffer(4, Some(&gdn_w.in_proj_b), 0);
            enc.set_buffer(5, Some(&gdn_w.in_proj_a), 0);
            enc.set_buffer(6, Some(&gdn_w.a_log), 0);
            enc.set_buffer(7, Some(&gdn_w.dt_bias), 0);
            enc.set_buffer(8, Some(&gdn_w.norm_weight), 0);
            enc.set_buffer(9, Some(&self.session.activations.gdn_z), z_off);
            enc.set_bytes(
                10,
                std::mem::size_of::<GdnRecurParams>() as u64,
                &params as *const GdnRecurParams as *const _,
            );
            enc.dispatch_thread_groups(MTLSize::new(num_vh as u64, 1, 1), MTLSize::new(32, 4, 1));
        }
    }

    pub(super) fn encode_gdn_prefill_after_recurrence(
        &self,
        enc: &ComputeCommandEncoderRef,
        compact_idx: usize,
        layer_idx: usize,
        cfg: &Qwen35Config,
        m: u32,
    ) {
        let (attn_w, common_w) = &self.engine.layer_weights[compact_idx];
        let MetalLayerAttnWeights::Linear(gdn_w) = attn_w else {
            unreachable!()
        };
        self.dispatch_gemm(
            enc,
            &self.session.activations.gdn_z,
            0,
            &gdn_w.out_proj,
            &self.session.activations.attn_out,
            0,
            m,
            cfg.hidden_size as u32,
            cfg.linear_output_dim() as u32,
        );
        // out_proj reads the post-recurrence gdn_z rows and adds into attn_out before
        // the MLP's residual-add consumes it.
        self.dispatch_lora_rows_if_active(
            enc,
            &self.session.activations.gdn_z,
            0,
            cfg.linear_output_dim() as u32,
            &self.session.activations.attn_out,
            0,
            cfg.hidden_size as u32,
            m,
            layer_idx,
            "out_proj",
        );
        self.encode_prefill_dense_mlp(
            enc,
            common_w,
            &self.session.activations.attn_out,
            cfg,
            layer_idx,
            m,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_full_attention_prefill_layer(
        &self,
        enc: &ComputeCommandEncoderRef,
        compact_idx: usize,
        layer_idx: usize,
        full_idx: usize,
        cfg: &Qwen35Config,
        n: usize,
        m: u32,
        start_pos: usize,
    ) {
        let hidden = cfg.hidden_size;
        let kv_dim = cfg.full_kv_dim();
        let q_dim = cfg.full_q_dim();
        let num_q_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let half_rope_dim = (cfg.rope_dim() / 2) as u32;
        let (attn_w, common_w) = &self.engine.layer_weights[compact_idx];
        let MetalLayerAttnWeights::Full(full_w) = attn_w else {
            unreachable!()
        };

        self.dispatch_copy_and_rms_norm_batch(
            enc,
            &self.session.activations.hidden,
            &self.session.activations.residual,
            &common_w.input_layernorm,
            hidden as u32,
            m,
            cfg.rms_norm_eps,
        );
        self.dispatch_gemm(
            enc,
            &self.session.activations.hidden,
            0,
            &full_w.q_proj,
            &self.session.activations.q,
            0,
            m,
            (2 * q_dim) as u32,
            hidden as u32,
        );
        // Q rows carry both the query and gate lanes (width 2 * q_dim); the delta must
        // land before the scatter, per-head norm, RoPE and KV write consume them.
        self.dispatch_lora_rows_if_active(
            enc,
            &self.session.activations.hidden,
            0,
            hidden as u32,
            &self.session.activations.q,
            0,
            (2 * q_dim) as u32,
            m,
            layer_idx,
            "q_proj",
        );
        self.dispatch_gemm(
            enc,
            &self.session.activations.hidden,
            0,
            &full_w.k_proj,
            &self.session.activations.k,
            0,
            m,
            kv_dim as u32,
            hidden as u32,
        );
        self.dispatch_lora_rows_if_active(
            enc,
            &self.session.activations.hidden,
            0,
            hidden as u32,
            &self.session.activations.k,
            0,
            kv_dim as u32,
            m,
            layer_idx,
            "k_proj",
        );
        self.dispatch_gemm(
            enc,
            &self.session.activations.hidden,
            0,
            &full_w.v_proj,
            &self.session.activations.v,
            0,
            m,
            kv_dim as u32,
            hidden as u32,
        );
        self.dispatch_lora_rows_if_active(
            enc,
            &self.session.activations.hidden,
            0,
            hidden as u32,
            &self.session.activations.v,
            0,
            kv_dim as u32,
            m,
            layer_idx,
            "v_proj",
        );

        let base_pos = start_pos as u32;
        let nqh = num_q_heads as u32;
        let nkh = num_kv_heads as u32;
        let hd = head_dim as u32;
        self.dispatch_scatter_q_gate_batch(enc, m, nqh, hd);
        self.dispatch_per_head_rms_norm_batch(
            enc,
            &self.session.activations.q_separated,
            &full_w.q_norm,
            m,
            nqh,
            hd,
            cfg.rms_norm_eps,
        );
        self.dispatch_per_head_rms_norm_batch(
            enc,
            &self.session.activations.k,
            &full_w.k_norm,
            m,
            nkh,
            hd,
            cfg.rms_norm_eps,
        );
        self.dispatch_partial_rope_batch(
            enc,
            &self.session.activations.q_separated,
            m,
            nqh,
            hd,
            half_rope_dim,
            base_pos,
        );
        self.dispatch_partial_rope_batch(
            enc,
            &self.session.activations.k,
            m,
            nkh,
            hd,
            half_rope_dim,
            base_pos,
        );
        self.dispatch_copy_kv_cache_batch(
            enc,
            &self.session.activations.k,
            &self.session.activations.v,
            &self.session.kv_cache.k_bufs[full_idx],
            &self.session.kv_cache.v_bufs[full_idx],
            m,
            kv_dim as u32,
            base_pos,
        );

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        if self
            .dispatch_prefill_attention_batched(
                enc,
                &self.session.kv_cache.k_bufs[full_idx],
                &self.session.kv_cache.v_bufs[full_idx],
                base_pos,
                m,
                hd,
                nqh,
                nkh,
                q_dim as u32,
                kv_dim as u32,
                scale,
            )
            .is_err()
        {
            for t in 0..n {
                let abs_pos = start_pos + t;
                let qs_off = (t * q_dim) as u64 * 4;
                let ao_off = (t * q_dim) as u64 * 4;
                let cache_len = (abs_pos + 1) as u32;
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
                enc.set_bytes(5, 4, &hd as *const u32 as *const _);
                enc.set_bytes(6, 4, &nqh as *const u32 as *const _);
                enc.set_bytes(7, 4, &nkh as *const u32 as *const _);
                let qd = q_dim as u32;
                let kvd = kv_dim as u32;
                enc.set_bytes(8, 4, &qd as *const u32 as *const _);
                enc.set_bytes(9, 4, &kvd as *const u32 as *const _);
                enc.set_bytes(10, 4, &scale as *const f32 as *const _);
                enc.dispatch_thread_groups(MTLSize::new(nkh as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
        }

        self.dispatch_sigmoid_gate(enc, m * q_dim as u32);
        self.dispatch_gemm(
            enc,
            &self.session.activations.attn_out,
            0,
            &full_w.o_proj,
            &self.session.activations.ffn_out,
            0,
            m,
            hidden as u32,
            q_dim as u32,
        );
        // o_proj reads the sigmoid-gated attention rows and adds into ffn_out before
        // the MLP's residual-add consumes it.
        self.dispatch_lora_rows_if_active(
            enc,
            &self.session.activations.attn_out,
            0,
            q_dim as u32,
            &self.session.activations.ffn_out,
            0,
            hidden as u32,
            m,
            layer_idx,
            "o_proj",
        );
        self.encode_prefill_dense_mlp(
            enc,
            common_w,
            &self.session.activations.ffn_out,
            cfg,
            layer_idx,
            m,
        );
    }

    // -----------------------------------------------------------------------
    // GDN chunked-scan helpers (default ON; LATTICE_GDN_CHUNKED=0 opts out)
    // -----------------------------------------------------------------------

    pub(super) fn supports_gdn_chunked_prefill(cfg: &Qwen35Config) -> bool {
        cfg.hidden_size == 1024
            && cfg.linear_num_key_heads == 16
            && cfg.linear_num_value_heads() == 16
            && cfg.linear_key_head_dim == 128
            && cfg.linear_value_head_dim == 128
            && cfg.linear_conv_kernel_dim == 4
    }

    pub(super) fn make_gdn_chunk_params(
        cfg: &Qwen35Config,
        n_tokens: usize,
    ) -> Option<GdnChunkParams> {
        if !Self::supports_gdn_chunked_prefill(cfg) || n_tokens == 0 {
            return None;
        }
        let n_tokens = u32::try_from(n_tokens).ok()?;
        let num_chunks = n_tokens.div_ceil(GDN_CHUNK_SIZE as u32);
        let key_dim = cfg.linear_key_head_dim as u32;
        let value_dim = cfg.linear_value_head_dim as u32;
        let num_key_heads = cfg.linear_num_key_heads as u32;
        let num_value_heads = cfg.linear_num_value_heads() as u32;
        let q_total = num_key_heads * key_dim;
        Some(GdnChunkParams {
            key_dim,
            value_dim,
            num_key_heads,
            num_value_heads,
            hidden_size: cfg.hidden_size as u32,
            q_total,
            v_offset: q_total * 2,
            qkv_dim: cfg.linear_qkv_dim() as u32,
            output_dim: cfg.linear_output_dim() as u32,
            chunk_size: GDN_CHUNK_SIZE as u32,
            n_tokens,
            num_chunks,
            active_chunk: 0,
            scale: 1.0 / (cfg.linear_key_head_dim as f32).sqrt(),
            eps: cfg.rms_norm_eps,
        })
    }
}
