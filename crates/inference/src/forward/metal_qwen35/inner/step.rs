#[cfg(doc)]
use super::MetalCrossTurnPrefixCache;
use super::{
    GdnStateTrafficScope, GpuCandidate, MetalLayerAttnWeights, MetalQwen35State, MetalStepOutput,
    QuantFormat, Qwen35Config, StepProfile, div_ceil, make_buffer, read_buffer,
    resolve_block_local_k,
};
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};
use std::sync::atomic::Ordering;

impl MetalQwen35State {
    /// Internal single-token forward step.
    ///
    /// When `capture_hidden` is true (or when `self.session.mtp` is loaded), copies the
    /// pre-final hidden state into `self.session.activations.pre_final_hidden` before the
    /// final RMSNorm, then reads it back to `self.session.last_pre_final_hidden`.
    pub(super) fn forward_step_inner(
        &mut self,
        token_id: u32,
        position: usize,
        capture_hidden: bool,
        signpost_scope: crate::forward::signpost::Scope,
    ) -> MetalStepOutput {
        self.forward_step_inner_impl(
            token_id,
            position,
            capture_hidden,
            false,
            true,
            GdnStateTrafficScope::Decode,
            None,
            None,
            signpost_scope,
        )
    }

    /// Same as `forward_step_inner`, but attributes GDN state traffic to the
    /// given scope instead of the `Decode` default. Used by the MTP verify and
    /// speculative-repair paths so their state traffic is not misattributed to
    /// (and does not double count against) normal decode.
    #[cfg(feature = "gdn-state-counters")]
    pub(super) fn forward_step_inner_with_traffic_scope(
        &mut self,
        token_id: u32,
        position: usize,
        capture_hidden: bool,
        scope: GdnStateTrafficScope,
    ) -> MetalStepOutput {
        // Both callers (repair replay, batched verifier) are MTP-verify
        // work, never autoregressive decode — always `NotDecode`.
        self.forward_step_inner_impl(
            token_id,
            position,
            capture_hidden,
            false,
            true,
            scope,
            None,
            None,
            crate::forward::signpost::Scope::NotDecode,
        )
    }

    /// Qwen3.5 vision (ADR-069 Metal S5 / MP2): run a single decoder step with the
    /// token-embedding lookup REPLACED by a caller-supplied row (HF `masked_scatter`
    /// semantics — mirrors `cpu_f16::forward_step_f16`'s `injected_embedding` contract).
    /// `token_id` is unused when injecting (the embedding lookup itself is skipped) but
    /// is still required to size-check against `vocab_size` in the non-injected caller
    /// paths that share `forward_step_inner_impl`; callers here pass `0` (always valid).
    ///
    /// The caller (`generate_multimodal_vision`) has already validated `embedding`'s
    /// length and finiteness before this is reached — this is a private, non-`pub`
    /// entry point, so that validation is not repeated per call.
    ///
    /// `generate_multimodal_vision` itself now calls
    /// [`Self::forward_step_injected_mrope`] (ADR-069 MP3 needs a per-token
    /// cos/sin row at every call site); this plain-injection entry point is
    /// retained because unit tests exercise the MP2 injection contract in
    /// isolation, without the M-RoPE table machinery.
    #[allow(dead_code)] // exercised by injection-only unit tests, see doc comment
    pub(super) fn forward_step_injected(&mut self, embedding: &[f32], position: usize) -> Vec<f32> {
        self.cross_turn_prefix_cache.clear();
        // Never reached on a production path (see doc comment) — always `NotDecode`.
        self.forward_step_inner_impl(
            0,
            position,
            false,
            false,
            true,
            GdnStateTrafficScope::Decode,
            Some(embedding),
            None,
            crate::forward::signpost::Scope::NotDecode,
        )
        .logits
    }

    /// Qwen3.5 vision (ADR-069 Metal MP3): same REPLACE-injection contract as
    /// [`Self::forward_step_injected`], plus a per-token interleaved-axis
    /// M-RoPE cos/sin row for the six GQA layers (see
    /// `forward_step_inner_impl`'s `mrope_cos_sin` doc). Used for image-pad
    /// prefill positions, where `injected_embedding` and `mrope_cos_sin` are
    /// both `Some`.
    ///
    /// `emit_head` gates the terminal RMSNorm + lm_head dispatch, same
    /// contract as `forward_step_inner_impl`'s parameter of the same name —
    /// pass `false` for a multimodal prefill position whose logits will
    /// never be sampled (KV/GDN state still advances); pass `true` for every
    /// decode step and for the prompt's final prefill position.
    pub(super) fn forward_step_injected_mrope(
        &mut self,
        embedding: &[f32],
        position: usize,
        mrope_cos_sin: Option<(&[f32], &[f32])>,
        emit_head: bool,
        signpost_scope: crate::forward::signpost::Scope,
    ) -> Vec<f32> {
        self.cross_turn_prefix_cache.clear();
        self.forward_step_inner_impl(
            0,
            position,
            false,
            false,
            emit_head,
            GdnStateTrafficScope::Decode,
            Some(embedding),
            mrope_cos_sin,
            signpost_scope,
        )
        .logits
    }

    /// Qwen3.5 vision (ADR-069 Metal MP3): ordinary token-id embedding lookup
    /// (no injection), plus a per-token M-RoPE cos/sin row — used for
    /// non-image-pad positions in a multimodal request (text tokens and
    /// vision delimiters keep the normal embedding lookup, but still rotate
    /// with the 3-axis table once an image has shifted the position
    /// coordinate) and for every decode step after a multimodal prefill.
    ///
    /// `emit_head`: see [`Self::forward_step_injected_mrope`]'s doc.
    pub(super) fn forward_step_mrope(
        &mut self,
        token_id: u32,
        position: usize,
        mrope_cos_sin: Option<(&[f32], &[f32])>,
        emit_head: bool,
        signpost_scope: crate::forward::signpost::Scope,
    ) -> Vec<f32> {
        self.cross_turn_prefix_cache.clear();
        self.forward_step_inner_impl(
            token_id,
            position,
            false,
            false,
            emit_head,
            GdnStateTrafficScope::Decode,
            None,
            mrope_cos_sin,
            signpost_scope,
        )
        .logits
    }

    // lattice#1584: `new_command_buffer` hands back a borrowed reference to an
    // autoreleased, unretained object, so every command buffer and encoder this
    // dispatch creates stays alive until some pool drains. The serving worker thread
    // is long-lived by construction, so its outermost pool never drains during a
    // session. Measured on this funnel at 3000 decode steps: 1.3-1.5 KB per step
    // accumulates without this pool and nothing accumulates with it, at identical
    // throughput.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_step_inner_impl(
        &mut self,
        token_id: u32,
        position: usize,
        capture_hidden: bool,
        skip_logits_readback: bool,
        emit_head: bool,
        _traffic_scope: GdnStateTrafficScope,
        injected_embedding: Option<&[f32]>,
        mrope_cos_sin: Option<(&[f32], &[f32])>,
        signpost_scope: crate::forward::signpost::Scope,
    ) -> MetalStepOutput {
        objc::rc::autoreleasepool(|| {
            self.forward_step_inner_impl_dispatch(
                token_id,
                position,
                capture_hidden,
                skip_logits_readback,
                emit_head,
                _traffic_scope,
                injected_embedding,
                mrope_cos_sin,
                signpost_scope,
            )
        })
    }

    /// `mrope_cos_sin`, when supplied (Qwen3.5 vision M-RoPE, ADR-069 MP3),
    /// replaces the six GQA layers' 1-D `engine.rope_cos`/`rope_sin` table
    /// lookup (keyed by `position`) with this caller-supplied interleaved-axis
    /// row for every full-attention layer in this step — mirrors
    /// `cpu_f16::full_attention_step_f16`'s `mrope_cos_sin` parameter. GDN
    /// layers never consume RoPE and are unaffected either way.
    ///
    /// `emit_head`, when `false`, skips the terminal RMSNorm + lm_head dispatch
    /// (`encode_final_head`) and its GPU readback entirely — every layer still
    /// runs, so the KV cache and GDN recurrent state are fully advanced, but
    /// `logits` comes back empty (issue #1336: a multimodal prefill position
    /// whose logits are never sampled has no reason to pay for the vocabulary
    /// projection). Forced back to `true` whenever `capture_hidden` or an
    /// active MTP session needs the pre-final hidden state this step, so a
    /// caller cannot accidentally starve MTP capture by passing `false`.
    #[allow(clippy::too_many_arguments)]
    fn forward_step_inner_impl_dispatch(
        &mut self,
        token_id: u32,
        position: usize,
        capture_hidden: bool,
        skip_logits_readback: bool,
        emit_head: bool,
        _traffic_scope: GdnStateTrafficScope,
        injected_embedding: Option<&[f32]>,
        mrope_cos_sin: Option<(&[f32], &[f32])>,
        signpost_scope: crate::forward::signpost::Scope,
    ) -> MetalStepOutput {
        let _signpost_step = crate::forward::signpost::interval_in(
            signpost_scope,
            crate::forward::signpost::Label::DecodeStep,
        );
        let cfg = self.engine.config.clone();
        let hidden = cfg.hidden_size;

        // A caller that wants pre-final hidden capture (MTP) still needs the
        // terminal RMSNorm dispatched even if it passed `emit_head=false`
        // (only the multimodal prefill loop does today, and never with MTP
        // active) — this keeps the flag safe by construction rather than by
        // caller discipline. `mtp_active` is this request's own commitment
        // to the MTP path (set by `generate()`'s `use_mtp`, cleared by
        // `reset_state()`), not `self.session.mtp.is_some()`: the latter
        // reports only that the checkpoint carries MTP weights, which stays
        // true even when this request runs with `enable_mtp: Some(false)`
        // (#1336 round 2 — that combination must still honor
        // `emit_head=false` for the multimodal prefill loop).
        let run_head = emit_head || capture_hidden || self.session.mtp_active;

        // Build the single-token M-RoPE cos/sin buffers once for this step (all
        // six GQA layers rotate with the same per-token row); `None` keeps every
        // `dispatch_partial_rope` call on the unchanged 1-D `engine.rope_cos`/
        // `rope_sin` path, so text-only decode is structurally untouched.
        let mrope_row_bufs: Option<(Buffer, Buffer)> = mrope_cos_sin.map(|(cos_row, sin_row)| {
            (
                make_buffer(&self.engine.device, cos_row, "mrope_cos_row"),
                make_buffer(&self.engine.device, sin_row, "mrope_sin_row"),
            )
        });
        let mrope_override: Option<(&Buffer, &Buffer)> =
            mrope_row_bufs.as_ref().map(|(c, s)| (c, s));

        // --- Per-phase timing (enabled by env LATTICE_PROFILE=1) ---
        let profiling = crate::env_switch_enabled("LATTICE_PROFILE");
        let decode_profiling = crate::env_switch_enabled("LATTICE_DECODE_PROFILE");

        if crate::env_switch_enabled("LATTICE_GDN_CPU") {
            #[cfg(not(debug_assertions))]
            panic!("LATTICE_GDN_CPU=1 is debug-only and cannot be used in release decode");
        }

        let mut prof = StepProfile::default();
        let mut gdn_gpu_dispatches = 0usize;
        let gdn_cpu_dispatches = 0usize;
        let expected_gdn_dispatches = cfg.num_active_linear_attention_layers();
        let step_start = std::time::Instant::now();

        // --- Embedding lookup (CPU row read, write f32 hidden) ---
        // Qwen3.5 vision (ADR-069 Metal S5 / MP2): `injected_embedding` REPLACES this
        // lookup with a caller-supplied row (HF `masked_scatter`; mirrors
        // `cpu_f16::forward_step_f16`'s contract) instead of reading `embed_tokens`.
        let t0 = std::time::Instant::now();
        match injected_embedding {
            Some(row) => {
                debug_assert_eq!(
                    row.len(),
                    hidden,
                    "forward_step_injected: caller must pre-validate row length"
                );
                // SAFETY: hidden buffer is StorageModeShared f32, no GPU in flight;
                // caller (forward_step_injected) guarantees row.len() == hidden.
                unsafe {
                    let dst = self.session.activations.hidden.contents() as *mut f32;
                    std::ptr::copy_nonoverlapping(row.as_ptr(), dst, hidden);
                }
            }
            None => {
                assert!(
                    (token_id as usize) < cfg.vocab_size,
                    "forward_step: token_id {token_id} >= vocab_size {}",
                    cfg.vocab_size
                );
                // SAFETY: token_id < vocab_size; hidden buffer row holds hidden f32 values.
                unsafe {
                    let dst = self.session.activations.hidden.contents() as *mut f32;
                    self.engine
                        .embed_tokens
                        .write_row(token_id as usize, dst, hidden);
                }
            }
        }

        if profiling {
            prof.embedding_us += t0.elapsed().as_micros();
        }

        let mut active_layer_idx = 0usize;
        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;

        let kv_dim = cfg.full_kv_dim();

        if decode_profiling {
            // Per-group command-buffer split path (LATTICE_DECODE_PROFILE=1).
            // Each layer gets its own command buffer committed and waited before the next.
            // GPU time = Instant::now() wraps commit()+wait_until_completed() per layer.
            // This serializes cross-kernel GPU pipelining; absolute throughput is lower
            // than the fused path.
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
                // SAFETY: same raw-pointer idiom as the fused path below.
                // cmd is ref-counted by Metal and outlives the queue borrow.
                // encode_* methods need &mut self but don't touch engine.queue.
                let layer_cmd = unsafe {
                    &*(self.engine.queue.new_command_buffer() as *const metal::CommandBufferRef)
                };
                let layer_enc = layer_cmd.new_compute_command_encoder();
                if is_linear {
                    // Mixer-only (no MLP): measures pure GDN recurrence cost.
                    let gdn_dispatches_this_layer = self.encode_gdn_layer(
                        layer_enc,
                        compact_idx,
                        linear_idx,
                        position,
                        layer_i,
                        &cfg,
                        &mut prof,
                        profiling,
                        false,
                    );
                    gdn_gpu_dispatches += gdn_dispatches_this_layer;
                    #[cfg(feature = "gdn-state-counters")]
                    self.record_gdn_layer_access(_traffic_scope, gdn_dispatches_this_layer);
                    linear_idx += 1;
                    layer_enc.end_encoding();
                    let t_mixer = std::time::Instant::now();
                    {
                        let _signpost_commit = crate::forward::signpost::interval_in(
                            signpost_scope,
                            crate::forward::signpost::Label::DecodeCbCommit,
                        );
                        layer_cmd.commit();
                    }
                    {
                        let _signpost_wait = crate::forward::signpost::interval_in(
                            signpost_scope,
                            crate::forward::signpost::Label::DecodeCbWait,
                        );
                        layer_cmd.wait_until_completed();
                    }
                    prof.gpu_gdn_mixer_us += t_mixer.elapsed().as_micros();
                    // MLP-only command buffer for this GDN layer.
                    let mlp_cmd = unsafe {
                        &*(self.engine.queue.new_command_buffer() as *const metal::CommandBufferRef)
                    };
                    let mlp_enc = mlp_cmd.new_compute_command_encoder();
                    self.encode_mlp_block(
                        mlp_enc,
                        compact_idx,
                        layer_i,
                        position,
                        &cfg,
                        &mut prof,
                        false,
                    );
                    mlp_enc.end_encoding();
                    let t_mlp = std::time::Instant::now();
                    {
                        let _signpost_commit = crate::forward::signpost::interval_in(
                            signpost_scope,
                            crate::forward::signpost::Label::DecodeCbCommit,
                        );
                        mlp_cmd.commit();
                    }
                    {
                        let _signpost_wait = crate::forward::signpost::interval_in(
                            signpost_scope,
                            crate::forward::signpost::Label::DecodeCbWait,
                        );
                        mlp_cmd.wait_until_completed();
                    }
                    prof.gpu_mlp_us += t_mlp.elapsed().as_micros();
                } else {
                    // Attn-only (no MLP): measures pure GQA attention cost.
                    self.encode_gqa_layer(
                        layer_enc,
                        compact_idx,
                        full_idx,
                        position,
                        kv_dim,
                        layer_i,
                        &cfg,
                        &mut prof,
                        profiling,
                        false,
                        mrope_override,
                    );
                    full_idx += 1;
                    layer_enc.end_encoding();
                    let t_attn = std::time::Instant::now();
                    {
                        let _signpost_commit = crate::forward::signpost::interval_in(
                            signpost_scope,
                            crate::forward::signpost::Label::DecodeCbCommit,
                        );
                        layer_cmd.commit();
                    }
                    {
                        let _signpost_wait = crate::forward::signpost::interval_in(
                            signpost_scope,
                            crate::forward::signpost::Label::DecodeCbWait,
                        );
                        layer_cmd.wait_until_completed();
                    }
                    prof.gpu_gqa_attn_us += t_attn.elapsed().as_micros();
                    // MLP-only command buffer for this GQA layer.
                    let mlp_cmd = unsafe {
                        &*(self.engine.queue.new_command_buffer() as *const metal::CommandBufferRef)
                    };
                    let mlp_enc = mlp_cmd.new_compute_command_encoder();
                    self.encode_mlp_block(
                        mlp_enc,
                        compact_idx,
                        layer_i,
                        position,
                        &cfg,
                        &mut prof,
                        false,
                    );
                    mlp_enc.end_encoding();
                    let t_mlp = std::time::Instant::now();
                    {
                        let _signpost_commit = crate::forward::signpost::interval_in(
                            signpost_scope,
                            crate::forward::signpost::Label::DecodeCbCommit,
                        );
                        mlp_cmd.commit();
                    }
                    {
                        let _signpost_wait = crate::forward::signpost::interval_in(
                            signpost_scope,
                            crate::forward::signpost::Label::DecodeCbWait,
                        );
                        mlp_cmd.wait_until_completed();
                    }
                    prof.gpu_mlp_us += t_mlp.elapsed().as_micros();
                }
                // Keep legacy aggregates for backward compat with existing tooling.
                prof.gpu_gdn_us = prof.gpu_gdn_mixer_us;
                prof.gpu_gqa_us = prof.gpu_gqa_attn_us;
            }

            let head_cmd = unsafe {
                &*(self.engine.queue.new_command_buffer() as *const metal::CommandBufferRef)
            };
            let head_enc = head_cmd.new_compute_command_encoder();
            let topk_which_inner = if run_head {
                self.encode_final_head(head_enc, &cfg, capture_hidden, &mut prof, profiling)
            } else {
                None
            };
            head_enc.end_encoding();
            let t_gpu = std::time::Instant::now();
            {
                let _signpost_commit = crate::forward::signpost::interval_in(
                    signpost_scope,
                    crate::forward::signpost::Label::DecodeCbCommit,
                );
                head_cmd.commit();
            }
            {
                let _signpost_wait = crate::forward::signpost::interval_in(
                    signpost_scope,
                    crate::forward::signpost::Label::DecodeCbWait,
                );
                head_cmd.wait_until_completed();
            }
            prof.gpu_lm_head_us += t_gpu.elapsed().as_micros();

            let total_gpu_us = prof.gpu_gdn_mixer_us
                + prof.gpu_gqa_attn_us
                + prof.gpu_mlp_us
                + prof.gpu_lm_head_us;
            let pct = |v: u128| -> f64 {
                if total_gpu_us == 0 {
                    0.0
                } else {
                    v as f64 / total_gpu_us as f64 * 100.0
                }
            };
            eprintln!("[DECODE_PROFILE] step {position}: total_gpu_us={total_gpu_us}");
            eprintln!(
                "  gdn_mixer:   {:>8} µs  ({:5.1}%)  [18 GDN layers: norm+in_proj+conv1d+recurrence+out_proj (no MLP)]",
                prof.gpu_gdn_mixer_us,
                pct(prof.gpu_gdn_mixer_us)
            );
            eprintln!(
                "  gqa_attn:    {:>8} µs  ({:5.1}%)  [6 GQA layers: norm+qkv_proj+attention+o_proj (no MLP)]",
                prof.gpu_gqa_attn_us,
                pct(prof.gpu_gqa_attn_us)
            );
            eprintln!(
                "  mlp:         {:>8} µs  ({:5.1}%)  [all 24 layers: post_attn_norm+gate_up_proj+silu_mul+down_proj]",
                prof.gpu_mlp_us,
                pct(prof.gpu_mlp_us)
            );
            eprintln!(
                "  lm_head:     {:>8} µs  ({:5.1}%)  [final norm + lm_head GEMV]",
                prof.gpu_lm_head_us,
                pct(prof.gpu_lm_head_us)
            );
            eprintln!(
                "  NOTE: GPU time under serialized per-group command buffers (2 cmd bufs/layer + 1 for lm_head). Absolute throughput is lower than the fused path."
            );

            debug_assert_eq!(
                gdn_gpu_dispatches, expected_gdn_dispatches,
                "GDN GPU recurrence dispatch count mismatch"
            );

            if profiling {
                let total_us = step_start.elapsed().as_micros();
                let accounted_us = prof.embedding_us
                    + prof.projection_us
                    + prof.gdn_recurrence_us
                    + prof.gqa_attention_us
                    + prof.mlp_us
                    + prof.final_us;
                let other_us = total_us.saturating_sub(accounted_us);
                eprintln!(
                    "[PROFILE] step {position}: total_us={total_us} \
                         embedding_us={} projection_us={} gdn_recurrence_us={} \
                         gqa_attention_us={} mlp_us={} final_us={} other_us={} \
                         gdn_gpu_dispatches={} expected_gdn_dispatches={} gdn_cpu_dispatches={}",
                    prof.embedding_us,
                    prof.projection_us,
                    prof.gdn_recurrence_us,
                    prof.gqa_attention_us,
                    prof.mlp_us,
                    prof.final_us,
                    other_us,
                    gdn_gpu_dispatches,
                    expected_gdn_dispatches,
                    gdn_cpu_dispatches,
                );
            }

            // SAFETY: GPU completed (wait_until_completed called above for head_cmd).
            let pre_final_hidden = if capture_hidden || self.session.mtp.is_some() {
                if capture_hidden && self.path_proof_enabled {
                    self.path_proof
                        .decode_hidden_readback
                        .fetch_add(1, Ordering::Relaxed);
                }
                let _signpost_host_read = crate::forward::signpost::interval_in(
                    signpost_scope,
                    crate::forward::signpost::Label::DecodeHostScalarRead,
                );
                let h = unsafe { read_buffer(&self.session.activations.pre_final_hidden, hidden) };
                self.session.last_pre_final_hidden = h.clone();
                self.session
                    .mark_pre_final_hidden(self.session.kv_cache.seq_len + 1);
                h
            } else {
                Vec::new()
            };

            let logits = if !run_head || skip_logits_readback {
                vec![]
            } else if let Some(which) = topk_which_inner {
                if self.path_proof_enabled {
                    self.path_proof
                        .decode_compact_candidate_logit_readback
                        .fetch_add(1, Ordering::Relaxed);
                    self.path_proof
                        .decode_compact_candidate_logit_readback_bytes
                        .fetch_add(
                            (self.session.compact_topk * std::mem::size_of::<GpuCandidate>())
                                as u64,
                            Ordering::Relaxed,
                        );
                }
                let _signpost_host_read = crate::forward::signpost::interval_in(
                    signpost_scope,
                    crate::forward::signpost::Label::DecodeHostScalarRead,
                );
                let k = self.session.compact_topk;
                let candidates = unsafe { self.read_topk_candidates(which, k) };
                self.session.compact_result = candidates;
                vec![]
            } else {
                if self.path_proof_enabled {
                    self.path_proof
                        .decode_full_vocab_logit_readback
                        .fetch_add(1, Ordering::Relaxed);
                    self.path_proof
                        .decode_full_vocab_logit_readback_bytes
                        .fetch_add(
                            (cfg.vocab_size * std::mem::size_of::<f32>()) as u64,
                            Ordering::Relaxed,
                        );
                }
                let _signpost_host_read = crate::forward::signpost::interval_in(
                    signpost_scope,
                    crate::forward::signpost::Label::DecodeHostScalarRead,
                );
                unsafe { read_buffer(&self.session.activations.logits, cfg.vocab_size) }
            };
            self.session.kv_cache.seq_len += 1;

            return MetalStepOutput {
                logits,
                pre_final_hidden,
                _signpost_step,
            };
        }

        // Fused single-command-buffer path (default, zero-cost when LATTICE_DECODE_PROFILE is off).
        // Single command buffer for ALL 24 layers — no intermediate waits.
        // Raw pointer breaks the borrow chain: cmd is ref-counted by Metal
        // and outlives the queue borrow. The encode_* methods need &mut self
        // (for session state) but don't touch engine.queue.
        let cmd =
            unsafe { &*(self.engine.queue.new_command_buffer() as *const metal::CommandBufferRef) };
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
            if is_linear {
                let gdn_dispatches_this_layer = self.encode_gdn_layer(
                    enc,
                    compact_idx,
                    linear_idx,
                    position,
                    layer_i,
                    &cfg,
                    &mut prof,
                    profiling,
                    true,
                );
                gdn_gpu_dispatches += gdn_dispatches_this_layer;
                #[cfg(feature = "gdn-state-counters")]
                self.record_gdn_layer_access(_traffic_scope, gdn_dispatches_this_layer);
                linear_idx += 1;
            } else {
                self.encode_gqa_layer(
                    enc,
                    compact_idx,
                    full_idx,
                    position,
                    kv_dim,
                    layer_i,
                    &cfg,
                    &mut prof,
                    profiling,
                    true,
                    mrope_override,
                );
                full_idx += 1;
            }
        } // end layer loop

        let topk_which = if run_head {
            self.encode_final_head(enc, &cfg, capture_hidden, &mut prof, profiling)
        } else {
            None
        };

        // Single submit for entire forward pass + optional top-k.
        enc.end_encoding();
        {
            let _signpost_commit = crate::forward::signpost::interval_in(
                signpost_scope,
                crate::forward::signpost::Label::DecodeCbCommit,
            );
            cmd.commit();
        }
        {
            let _signpost_wait = crate::forward::signpost::interval_in(
                signpost_scope,
                crate::forward::signpost::Label::DecodeCbWait,
            );
            cmd.wait_until_completed();
        }

        // Diagnostic: dump post-final-norm hidden state magnitudes for
        // divergence analysis vs MLX. Set LATTICE_HIDDEN_DUMP=path to enable.
        if let Ok(dump_path) = std::env::var("LATTICE_HIDDEN_DUMP") {
            // SAFETY: hidden buffer is StorageModeShared, GPU completed (just waited).
            // Reads `cfg.hidden_size` f32 values starting at offset 0 (post-RMSNorm hidden
            // for position 0 — only valid for decode/single-token forward_step).
            let hidden_vec: Vec<f32> = unsafe {
                let ptr = self.session.activations.hidden.contents() as *const f32;
                std::slice::from_raw_parts(ptr, cfg.hidden_size).to_vec()
            };
            let mag = hidden_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
            let max = hidden_vec.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let mean = hidden_vec.iter().sum::<f32>() / hidden_vec.len() as f32;
            eprintln!(
                "[hidden-dump] token={token_id} pos={position} L2_norm={mag:.4} max_abs={max:.4} mean={mean:.4}"
            );
            let bytes: Vec<u8> = hidden_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&dump_path, &bytes).expect("failed to dump hidden");
        }

        debug_assert_eq!(
            gdn_gpu_dispatches, expected_gdn_dispatches,
            "GDN GPU recurrence dispatch count mismatch"
        );

        if profiling {
            let total_us = step_start.elapsed().as_micros();
            let accounted_us = prof.embedding_us
                + prof.projection_us
                + prof.gdn_recurrence_us
                + prof.gqa_attention_us
                + prof.mlp_us
                + prof.final_us;
            let other_us = total_us.saturating_sub(accounted_us);
            eprintln!(
                "[PROFILE] step {position}: total_us={total_us} \
                     embedding_us={} projection_us={} gdn_recurrence_us={} \
                     gqa_attention_us={} mlp_us={} final_us={} other_us={} \
                     gdn_gpu_dispatches={} expected_gdn_dispatches={} gdn_cpu_dispatches={}",
                prof.embedding_us,
                prof.projection_us,
                prof.gdn_recurrence_us,
                prof.gqa_attention_us,
                prof.mlp_us,
                prof.final_us,
                other_us,
                gdn_gpu_dispatches,
                expected_gdn_dispatches,
                gdn_cpu_dispatches,
            );
        }

        // Read back pre-final hidden when requested (for MTP input).
        // SAFETY: GPU completed, pre_final_hidden is StorageModeShared.
        let pre_final_hidden = if capture_hidden || self.session.mtp.is_some() {
            if capture_hidden && self.path_proof_enabled {
                self.path_proof
                    .decode_hidden_readback
                    .fetch_add(1, Ordering::Relaxed);
            }
            let _signpost_host_read = crate::forward::signpost::interval_in(
                signpost_scope,
                crate::forward::signpost::Label::DecodeHostScalarRead,
            );
            let h = unsafe { read_buffer(&self.session.activations.pre_final_hidden, hidden) };
            self.session.last_pre_final_hidden = h.clone();
            self.session
                .mark_pre_final_hidden(self.session.kv_cache.seq_len + 1);
            h
        } else {
            Vec::new()
        };

        let logits = if !run_head || skip_logits_readback {
            vec![]
        } else if let Some(which) = topk_which {
            // Compact path: read k*(f32+u32)=k*8 bytes instead of vocab*4 bytes.
            // SAFETY: GPU completed, buffers are StorageModeShared.
            if self.path_proof_enabled {
                self.path_proof
                    .decode_compact_candidate_logit_readback
                    .fetch_add(1, Ordering::Relaxed);
                self.path_proof
                    .decode_compact_candidate_logit_readback_bytes
                    .fetch_add(
                        (self.session.compact_topk * std::mem::size_of::<GpuCandidate>()) as u64,
                        Ordering::Relaxed,
                    );
            }
            let _signpost_host_read = crate::forward::signpost::interval_in(
                signpost_scope,
                crate::forward::signpost::Label::DecodeHostScalarRead,
            );
            let k = self.session.compact_topk;
            let candidates = unsafe { self.read_topk_candidates(which, k) };
            self.session.compact_result = candidates;
            vec![] // generate loop reads compact_result, not these logits
        } else {
            // Full path: read vocab_size f32 logits back to host.
            // SAFETY: GPU completed, buffer is StorageModeShared.
            if self.path_proof_enabled {
                self.path_proof
                    .decode_full_vocab_logit_readback
                    .fetch_add(1, Ordering::Relaxed);
                self.path_proof
                    .decode_full_vocab_logit_readback_bytes
                    .fetch_add(
                        (cfg.vocab_size * std::mem::size_of::<f32>()) as u64,
                        Ordering::Relaxed,
                    );
            }
            let _signpost_host_read = crate::forward::signpost::interval_in(
                signpost_scope,
                crate::forward::signpost::Label::DecodeHostScalarRead,
            );
            unsafe { read_buffer(&self.session.activations.logits, cfg.vocab_size) }
        };
        self.session.kv_cache.seq_len += 1;

        MetalStepOutput {
            logits,
            pre_final_hidden,
            _signpost_step,
        }
    }

    fn check_forward_step_capacity(
        &self,
        position: usize,
    ) -> Result<(), crate::error::InferenceError> {
        use crate::error::InferenceError;

        let max_cache_len = self.session.kv_cache.max_cache_len;
        if position >= max_cache_len {
            return Err(InferenceError::InvalidInput(format!(
                "forward_step: position {position} exceeds max position {}",
                max_cache_len - 1
            )));
        }
        if self.session.kv_cache.seq_len >= max_cache_len {
            return Err(InferenceError::InvalidInput(format!(
                "forward_step: KV cache is full at {} tokens (max_cache_len {max_cache_len})",
                self.session.kv_cache.seq_len
            )));
        }
        Ok(())
    }

    fn check_forward_token_id(
        &self,
        entry_point: &str,
        token_id: u32,
    ) -> Result<(), crate::error::InferenceError> {
        if token_id as usize >= self.engine.config.vocab_size {
            return Err(crate::error::InferenceError::InvalidInput(format!(
                "{entry_point}: token_id {token_id} out of range: vocab_size is {}",
                self.engine.config.vocab_size
            )));
        }
        Ok(())
    }

    pub(super) fn check_forward_token_ids(
        &self,
        entry_point: &str,
        token_ids: &[u32],
    ) -> Result<(), crate::error::InferenceError> {
        for (index, &token_id) in token_ids.iter().enumerate() {
            if token_id as usize >= self.engine.config.vocab_size {
                return Err(crate::error::InferenceError::InvalidInput(format!(
                    "{entry_point}: token_ids[{index}]={token_id} out of range: vocab_size \
                         is {}",
                    self.engine.config.vocab_size
                )));
            }
        }
        Ok(())
    }

    pub(super) fn validate_live_cursor(
        entry_point: &str,
        supplied_position: usize,
        live_cursor: usize,
    ) -> Result<(), crate::error::InferenceError> {
        if supplied_position != live_cursor {
            return Err(crate::error::InferenceError::InvalidInput(format!(
                "{entry_point}: supplied position {supplied_position} does not match the \
                     live cache cursor {live_cursor}; KV placement and attention length derive \
                     from kv_cache.seq_len while RoPE uses the supplied position"
            )));
        }
        Ok(())
    }

    pub(super) fn check_live_cursor(
        &self,
        entry_point: &str,
        supplied_position: usize,
    ) -> Result<(), crate::error::InferenceError> {
        Self::validate_live_cursor(
            entry_point,
            supplied_position,
            self.session.kv_cache.seq_len,
        )
    }

    pub(super) fn validate_hidden_prefill_fresh_session(
        seq_len: usize,
        gdn_state_is_initial: bool,
    ) -> Result<(), crate::error::InferenceError> {
        if seq_len != 0 || !gdn_state_is_initial {
            return Err(crate::error::InferenceError::InvalidInput(format!(
                "forward_prefill_with_hidden: requires a fresh session (kv_cache.seq_len \
                     == 0 and GDN recurrent state at its initial condition), found \
                     kv_cache.seq_len={seq_len}; this call always dispatches from position 0, so \
                     calling it against a session with live state would overwrite the \
                     existing KV prefix while leaving GDN state conditioned on the tokens it \
                     overwrote. Call reset_state() first to start a new prompt."
            )));
        }
        Ok(())
    }

    /// The fresh-session predicate: `kv_cache.seq_len == 0 && gdn_state_is_initial()`,
    /// with `gdn_state_is_initial()` treated as vacuously satisfied when the session
    /// has no GDN layers at all — there is no recurrent state for such a session to
    /// hold live, so `seq_len == 0` is already the complete freshness criterion.
    /// The ONLY place this predicate is computed — every caller that needs it goes
    /// through here (or through [`Self::check_hidden_prefill_fresh_session`], which
    /// wraps it) so a caller can never check `seq_len` without also checking GDN state.
    fn prefill_session_freshness(&self) -> (usize, bool) {
        let seq_len = self.session.kv_cache.seq_len;
        let gdn_fresh = !self.has_gdn_layers() || self.gdn_state_is_initial();
        (seq_len, seq_len == 0 && gdn_fresh)
    }

    pub(super) fn check_hidden_prefill_fresh_session(
        &self,
    ) -> Result<(), crate::error::InferenceError> {
        let (seq_len, fresh) = self.prefill_session_freshness();
        Self::validate_hidden_prefill_fresh_session(seq_len, fresh)
    }

    pub(super) fn check_raw_prefill_fresh_session(
        &self,
        entry_point: &str,
    ) -> Result<(), crate::error::InferenceError> {
        let (seq_len, fresh) = self.prefill_session_freshness();
        if !fresh {
            return Err(crate::error::InferenceError::InvalidInput(format!(
                "{entry_point}: requires a fresh session (kv_cache.seq_len == 0 and GDN \
                     recurrent state at its initial condition), found \
                     kv_cache.seq_len={seq_len}; this call always dispatches from position \
                     0. Call reset_state() first to start a new prompt."
            )));
        }
        Ok(())
    }

    /// Batched prefill has no MoE schedule: `assert_batched_prefill_dense_only`
    /// panics on it. Reported here as a typed error so a caller sees a rejection
    /// instead of the process aborting; the assert stays as the internal
    /// invariant for direct callers of the batched chunk schedulers.
    ///
    /// This is the unconditional form: it rejects whenever the model has an MoE
    /// layer, for call sites every remaining branch of which is batched.
    /// [`Self::forward_prefill_from`] calls it after its single-token and
    /// LoRA-on-MoE fast paths have already returned via per-token forward steps
    /// (which support MoE). Entry points whose per-token routing is expressible as a
    /// predicate on the input use
    /// [`Self::check_prefill_moe_batched_unsupported`], which applies the
    /// token-count/LoRA gate before delegating here.
    pub(super) fn reject_moe_batched(
        &self,
        entry_point: &str,
    ) -> Result<(), crate::error::InferenceError> {
        if self.engine.has_moe_layer {
            return Err(crate::error::InferenceError::UnsupportedModel(format!(
                "{entry_point}: batched prefill (GEMM path) does not support MoE \
                     layers. Decode one token at a time via forward_step for MoE \
                     models."
            )));
        }
        Ok(())
    }

    /// Gated form of [`Self::reject_moe_batched`] for entry points that route a
    /// single-token or LoRA-active input to the per-token path, where MoE is
    /// supported (a LoRA-active dense model takes the batched schedulers, which
    /// apply the adapter; this gate only matters when an MoE layer exists): only
    /// a multi-token, no-LoRA input reaches the batched
    /// schedulers, so only that shape is rejected.
    ///
    /// Wired into the public entry points that can reach the batched schedulers
    /// through `forward_prefill_impl` or the with-hidden chunk loop:
    /// [`Self::try_forward_prefill`], [`Self::forward_prefill_with_hidden`], and
    /// [`Self::forward_prefill_all_logits`]. The suffix primitive
    /// [`Self::forward_prefill_from`] — reached by the prefix-cache generation
    /// path — uses the unconditional form above. Every request-facing caller
    /// prefills through one of the four: `generate`,
    /// `generate_with_speculation`, `generate_streaming`,
    /// `generate_streaming_with_cancel`, and `embed_tokens` through the
    /// first, `compute_token_nlls` through the third,
    /// `generate_streaming_with_prefix_cache*` through the fourth — so an
    /// MoE model reports this input instead of aborting the process.
    ///
    /// [`Self::forward_prefill`] is the exception, by construction: it unwraps
    /// [`Self::try_forward_prefill`]'s result, so it still aborts. Its remaining
    /// callers are this module's tests, the bench and profiling examples, and any
    /// external consumer that wants the panicking form.
    pub(super) fn check_prefill_moe_batched_unsupported(
        &self,
        entry_point: &str,
        token_count: usize,
    ) -> Result<(), crate::error::InferenceError> {
        if token_count <= 1 || self.lora.is_some() {
            return Ok(());
        }
        self.reject_moe_batched(entry_point)
    }

    /// Pure capacity precondition for a Metal dispatch spanning `token_count` positions
    /// starting at `start_pos` — every position `start_pos..start_pos + token_count` must
    /// land inside the RoPE table / KV cache, and (when `gdn_pool_capacity` is `Some`)
    /// `token_count` must not exceed the fixed-size GDN checkpoint pool. Checked before any
    /// of it reaches Metal dispatch or GDN checkpoint mutation, so a caller-supplied token
    /// count can never leave session state partially advanced on rejection.
    ///
    /// `gdn_pool_capacity` bounds callers that checkpoint one pool slot per processed token
    /// (slot 0 = base, slot k = after k tokens, so valid slots are `0..=capacity` and at
    /// most `capacity` tokens may be processed after the base checkpoint).
    ///
    /// Free function (no `&self`) so it is unit-testable on CPU without a Metal device.
    pub(super) fn validate_dispatch_capacity(
        start_pos: usize,
        token_count: usize,
        kv_capacity: usize,
        gdn_pool_capacity: Option<usize>,
    ) -> Result<(), crate::error::InferenceError> {
        use crate::error::InferenceError;

        if token_count == 0 {
            return Ok(());
        }
        let last_position = start_pos.saturating_add(token_count - 1);
        if last_position >= kv_capacity {
            return Err(InferenceError::InvalidInput(format!(
                "start_pos {start_pos} + {token_count} tokens reaches position \
                     {last_position}, exceeding max position {}",
                kv_capacity - 1
            )));
        }
        if let Some(pool_capacity) = gdn_pool_capacity
            && token_count > pool_capacity
        {
            return Err(InferenceError::InvalidInput(format!(
                "start_pos {start_pos} + {token_count} tokens exceeds GDN checkpoint pool \
                     capacity {pool_capacity} (checkpoint slots 0..={pool_capacity})"
            )));
        }
        Ok(())
    }

    /// Method form of [`Self::validate_dispatch_capacity`], resolving the KV/RoPE bound
    /// from live session state.
    ///
    /// Scope, stated precisely because the obvious paraphrase of it is wrong: this is
    /// shared by every caller that dispatches a caller-supplied *range*,
    /// `start_pos..start_pos + token_count`. It is not on the single-position decode
    /// path. `forward_step_decode` and `forward_step_greedy_argmax` take one position
    /// and are bounded by their caller instead: each decode loop tests the pending
    /// position against `kv_cache.max_cache_len` and breaks with `StopReason::KvFull`
    /// before dispatching, with the headroom that loop needs. There are three distinct
    /// headroom requirements, not three loops — the loops are more numerous than the
    /// shapes, and reading the list below as a list of loops is the mistake this
    /// paragraph exists to prevent. The MTP loop reserves two slots
    /// (`max_cache_len.saturating_sub(2)`); the self-speculative loop reserves
    /// `SELF_SPEC_MAX_DRAFT + 1`; every other decode loop, including the multimodal and
    /// vision ones, needs a single free slot and tests `seq_len >= max_cache_len`
    /// directly. The streaming entry `generate_streaming_with_cancel` decodes through
    /// the decoder session instead, whose `decode` refuses a full cache with an error;
    /// its admission bound keeps every request clear of that refusal.
    /// Read as "every Metal dispatch goes through here" this comment is false, and a
    /// reader who checks it against the decode path will correctly conclude the guard is
    /// bypassed.
    ///
    /// Pass `check_gdn_pool = true` for callers that checkpoint one GDN pool slot per
    /// token (`verify_tokens_batched`); other callers pass `false` since they never
    /// advance the GDN checkpoint past the base slot.
    pub(super) fn check_forward_range_capacity(
        &self,
        start_pos: usize,
        token_count: usize,
        check_gdn_pool: bool,
    ) -> Result<(), crate::error::InferenceError> {
        let gdn_pool_capacity = if check_gdn_pool {
            self.session.gdn_checkpoints.as_ref().map(|p| p.max_tokens)
        } else {
            None
        };
        Self::validate_dispatch_capacity(
            start_pos,
            token_count,
            self.session.kv_cache.max_cache_len,
            gdn_pool_capacity,
        )
    }

    /// True when the session has at least one GatedDeltaNet linear-attention
    /// layer, i.e. `gdn_state_is_initial()` has a population to answer about.
    /// A session with zero GDN layers holds no recurrent state at all — that
    /// is a distinct fact from "the recurrent state is clean" and callers must
    /// check this before asking [`Self::gdn_state_is_initial`], never conflate
    /// the two by treating an empty population as if it were a clean one.
    pub(super) fn has_gdn_layers(&self) -> bool {
        self.session.gdn_gpu_state.len() != 0
    }

    /// True when every GDN recurrent-state buffer (S matrices and conv1d
    /// rolling buffers, for every linear-attention layer) holds its
    /// post-`reset_state()` zero value. Reads the live GPU buffers directly
    /// (same buffers and safety invariant as `snapshot_gdn_states`), so this
    /// reflects the state Metal dispatch will actually see, not the CPU
    /// `gdn_states` mirror.
    ///
    /// # Panics
    ///
    /// Panics when the session has no GDN layers (`has_gdn_layers()` is
    /// false). A vacuous "true" there would silently conflate "no buffers to
    /// check" with "buffers checked and clean" — two different facts a caller
    /// may need to tell apart. Callers that consider both cases equally fresh
    /// (there is nothing to protect when there is no GDN state) must say so
    /// explicitly at the call site, e.g. `!self.has_gdn_layers() ||
    /// self.gdn_state_is_initial()`, as [`Self::prefill_session_freshness`]
    /// does.
    pub(super) fn gdn_state_is_initial(&self) -> bool {
        let num_layers = self.session.gdn_gpu_state.len();
        assert!(
            num_layers > 0,
            "gdn_state_is_initial: session has no GDN layers; check has_gdn_layers() \
                 first — an empty population and a clean population are not the same fact"
        );
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
            // allocate f32 buffers). Callers invoke this before their own GPU
            // dispatch, with no command buffer in flight, so no GPU write can
            // race with this read — same invariant as `snapshot_gdn_states`.
            let conv_nonzero = unsafe {
                let ptr = conv_buf.contents() as *const f32;
                std::slice::from_raw_parts(ptr, conv_floats)
                    .iter()
                    .any(|&v| v != 0.0)
            };
            if conv_nonzero {
                return false;
            }
            let s_nonzero = unsafe {
                let ptr = s_buf.contents() as *const f32;
                std::slice::from_raw_parts(ptr, s_floats)
                    .iter()
                    .any(|&v| v != 0.0)
            };
            if s_nonzero {
                return false;
            }
        }
        true
    }

    /// **Unstable**: fallible single-token forward step; kernel dispatch strategy evolving.
    ///
    /// Run a single token through the full model. Returns logits [vocab_size].
    ///
    /// `position` must equal the live cache cursor (`kv_cache.seq_len`). KV
    /// placement and attention length derive from the live cursor while RoPE
    /// uses the supplied `position`; sparse or out-of-order positions are not
    /// supported.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::InferenceError::InvalidInput`] before GPU dispatch
    /// when `token_id >= vocab_size`, when `position` does not equal
    /// `kv_cache.seq_len`, or when `position` or the next KV-cache row is
    /// outside the session capacity. On rejection, session state is left
    /// unchanged.
    pub fn try_forward_step(
        &mut self,
        token_id: u32,
        position: usize,
    ) -> Result<Vec<f32>, crate::error::InferenceError> {
        self.check_forward_token_id("try_forward_step", token_id)?;
        self.check_forward_step_capacity(position)?;
        self.check_live_cursor("try_forward_step", position)?;
        self.cross_turn_prefix_cache.clear();
        let output = self.forward_step_inner(
            token_id,
            position,
            false,
            crate::forward::signpost::Scope::NotDecode,
        );
        Ok(output.logits)
    }

    /// **Unstable**: fallible single-token forward with explicit hidden readback.
    ///
    /// Returns `(logits, pre_final_hidden)`. The hidden row is captured before
    /// the final RMSNorm overwrites the activation buffer. Existing
    /// [`Self::try_forward_step`] and generation paths retain their current
    /// readback behavior.
    ///
    /// `position` must equal the live cache cursor (`kv_cache.seq_len`): KV
    /// placement and attention length derive from `kv_cache.seq_len` while
    /// RoPE rotates by the supplied `position`, so a mismatched `position`
    /// would rotate the token for a position other than the KV row it is
    /// actually written to. Callers driving an out-of-order or sparse
    /// position sequence are not supported by this entry point.
    ///
    /// # Errors
    ///
    /// Returns [`crate::error::InferenceError::InvalidInput`] before GPU
    /// dispatch when `token_id`, `position`, or the next KV-cache row is
    /// outside the session capacity, or when `position` does not match
    /// `kv_cache.seq_len`. On rejection, session state (KV cache, GDN
    /// state, hidden-readback counters) is left unchanged.
    pub fn forward_step_with_hidden(
        &mut self,
        token_id: u32,
        position: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), crate::error::InferenceError> {
        self.check_forward_token_id("forward_step_with_hidden", token_id)?;
        self.check_forward_step_capacity(position)?;
        self.check_live_cursor("forward_step_with_hidden", position)?;
        self.cross_turn_prefix_cache.clear();
        let output = self.forward_step_inner(
            token_id,
            position,
            true,
            crate::forward::signpost::Scope::NotDecode,
        );
        Ok((output.logits, output.pre_final_hidden))
    }

    /// **Unstable**: single-token forward step; kernel dispatch strategy evolving.
    ///
    /// Run a single token through the full model. Returns logits [vocab_size].
    ///
    /// # Panics
    ///
    /// Panics if `token_id >= vocab_size`, if `position` does not equal the
    /// live cache cursor (`kv_cache.seq_len`), or if `position` or the next
    /// KV-cache row exceeds the session capacity. Use [`Self::try_forward_step`]
    /// to receive a typed input error. These checks run before GPU dispatch.
    ///
    /// # Cross-turn cache invalidation (#516)
    ///
    /// This is a public raw-forward entry point that advances live KV/GDN
    /// state outside the cache-aware generation family's ownership, so it
    /// must clear any retained [`MetalCrossTurnPrefixCache`] entry before
    /// mutating — otherwise a later `generate_streaming_with_prefix_cache`
    /// call could restore a GDN snapshot for KV rows this call already
    /// overwrote. Safe to call from inside the cache-aware decode loop
    /// itself: by the time that loop reaches its own `forward_step` call,
    /// `restore_cross_turn_prefix` has already `take()`-n (ExactAppend) or
    /// `reset_state()` has already cleared (FullRefill) the slot's entry,
    /// so this clear is a no-op there — the cache-aware path never has a
    /// live entry saved while it is still generating.
    pub fn forward_step(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        // General-purpose raw entry point: reused for prompt prefill (a
        // token-at-a-time loop, e.g. `forward_prefill_impl`,
        // `forward_prefill_from`, multimodal prefill) as well as by
        // external library consumers calling it directly, so it cannot
        // assume decode scope. The production autoregressive decode
        // loops call `forward_step_decode` instead (below), which is
        // `Scope::Decode`. Delegates to `try_forward_step` so this path
        // gets the same before-dispatch capacity validation as external
        // callers of the fallible API.
        match self.try_forward_step(token_id, position) {
            Ok(logits) => logits,
            Err(error) => panic!("{error}"),
        }
    }

    /// Same contract as [`Self::forward_step`], for the production
    /// autoregressive decode loops (`generate`, `generate_streaming*`,
    /// `generate_streaming_with_prefix_cache*`, and
    /// `crate::decoder::qwen_metal::QwenMetalSession`) only — marks the
    /// `decode.*` signpost interval `Scope::Decode` so it is not silent.
    /// Not `pub`: external consumers get the general-purpose
    /// `forward_step`, which stays `Scope::NotDecode` since it cannot
    /// know whether the caller is decoding or prefilling.
    pub(crate) fn forward_step_decode(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        self.cross_turn_prefix_cache.clear();
        self.forward_step_inner(
            token_id,
            position,
            false,
            crate::forward::signpost::Scope::Decode,
        )
        .logits
    }

    /// Zero-copy greedy argmax: run forward pass then scan GPU shared buffer
    /// directly for the argmax token ID, avoiding 993KB allocation+copy.
    pub(crate) fn forward_step_greedy_argmax(&mut self, token_id: u32, position: usize) -> u32 {
        let cfg = self.engine.config.clone();
        // Binding (not discarding) the returned guard keeps `decode.step`
        // alive through the host-read/argmax scan below, matching its
        // documented span (through logits/top-k readback) — the scan is
        // logically part of this step, not a separate one.
        let _step_guard = self.forward_step_inner_impl(
            token_id,
            position,
            false,
            true,
            true,
            GdnStateTrafficScope::Decode,
            None,
            None,
            crate::forward::signpost::Scope::Decode,
        );

        // Fused host read + greedy sample: this scan both reads the
        // shared logits buffer off the GPU and picks the argmax token in
        // the same pass (the whole point of the zero-copy path), so it
        // carries both signpost labels for the span it covers.
        let _signpost_host_read = crate::forward::signpost::interval(
            crate::forward::signpost::Label::DecodeHostScalarRead,
        );
        let _signpost_sample =
            crate::forward::signpost::interval(crate::forward::signpost::Label::DecodeSample);
        // SAFETY: GPU completed (wait_until_completed called in forward_step_inner).
        // logits buffer is StorageModeShared — CPU can read it directly.
        unsafe {
            let ptr = self.session.activations.logits.contents() as *const f32;
            let vocab_size = cfg.vocab_size;
            let mut best_id = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for i in 0..vocab_size {
                let v = *ptr.add(i);
                if v > best_val {
                    best_val = v;
                    best_id = i as u32;
                }
            }
            best_id
        }
    }

    // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
    // dispatch creates command buffers on the same long-lived worker thread.
    // Unmeasured individually; fixed on the structural argument, not on numbers.
    pub(super) fn forward_step_gdn_only(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        objc::rc::autoreleasepool(|| self.forward_step_gdn_only_dispatch(token_id, position))
    }

    /// Run a token through GDN layers only, skipping all GQA layers.
    ///
    /// GQA layers are bypassed: the residual stream passes through unmodified,
    /// the KV cache is NOT advanced, and `session.kv_cache.seq_len` is NOT
    /// incremented. The function returns full-vocab logits computed from the
    /// GDN-only hidden state via final-norm + lm_head.
    ///
    /// Caller is responsible for checkpointing GDN state before calling.
    fn forward_step_gdn_only_dispatch(&mut self, token_id: u32, position: usize) -> Vec<f32> {
        let cfg = self.engine.config.clone();
        let hidden = cfg.hidden_size;

        // Embedding lookup — identical to forward_step_inner.
        // SAFETY: token_id < vocab_size; hidden buffer row holds hidden f32 values.
        unsafe {
            let dst = self.session.activations.hidden.contents() as *mut f32;
            self.engine
                .embed_tokens
                .write_row(token_id as usize, dst, hidden);
        }

        let mut active_layer_idx = 0usize;
        let mut linear_idx = 0usize;

        // Single command buffer for all GDN layers (GQA layers produce no encoder work).
        // Raw pointer breaks the borrow chain; identical idiom to forward_step_inner.
        let cmd =
            unsafe { &*(self.engine.queue.new_command_buffer() as *const metal::CommandBufferRef) };
        let enc = cmd.new_compute_command_encoder();

        let mut prof = StepProfile::default();
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
            if is_linear {
                self.encode_gdn_layer(
                    enc,
                    compact_idx,
                    linear_idx,
                    position,
                    layer_i,
                    &cfg,
                    &mut prof,
                    false,
                    true,
                );
                #[cfg(feature = "gdn-state-counters")]
                self.record_gdn_layer_access(GdnStateTrafficScope::Decode, 1);
                linear_idx += 1;
            }
            // GQA layers: residual stream passes through without modification.
        }

        // Final head (RMSNorm + lm_head) — capture_hidden=false, no MTP side-effect.
        self.encode_final_head(enc, &cfg, false, &mut prof, false);

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Read logits — do NOT increment kv_cache.seq_len.
        // SAFETY: GPU completed, buffer is StorageModeShared.
        unsafe { read_buffer(&self.session.activations.logits, cfg.vocab_size) }
    }

    // ===================================================================
    // Layer-encoding helpers (called from forward_step_inner)
    // ===================================================================

    /// Copy the last token's hidden state into `activations.final_hidden`,
    /// AFTER the final RMSNorm has been applied to it.
    ///
    /// One implementation with three call sites, one per forward path that
    /// dispatches the final norm, so the paths cannot drift apart. Grep
    /// `engine.final_norm` to enumerate them; every hit is either a
    /// dispatch immediately followed by a call to this, or the weight
    /// buffer's own definition.
    ///
    /// Placement after the norm is the whole point. The norm runs in place
    /// on `activations.hidden`, so the same address holds a pre-norm vector
    /// before it and a post-norm vector after, and the two differ by the
    /// norm's rescale. A capture encoded before the norm returns
    /// well-formed, correctly-sized, plausible numbers from the wrong point
    /// in the network — which is exactly what `pre_final_hidden` above
    /// holds for MTP, and exactly why this is a separate buffer.
    ///
    /// `src_row_offset_bytes` selects the row of `activations.hidden`
    /// holding the token of interest: the last row on the batch paths, row
    /// zero on the single-token decode path.
    pub(super) fn encode_capture_final_hidden(
        &self,
        enc: &ComputeCommandEncoderRef,
        src_row_offset_bytes: u64,
        hidden: usize,
    ) {
        if !self.session.capture_final_hidden {
            return;
        }
        // `&self`, not `&mut self`: two of the three call sites hold a live
        // immutable borrow of `self.engine.queue`'s command buffer across
        // the encode, so a `&mut self` helper cannot be called there at
        // all. The record is therefore an atomic rather than a plain bool.
        self.session
            .final_hidden_captured
            .store(true, std::sync::atomic::Ordering::Relaxed);
        enc.set_compute_pipeline_state(&self.engine.pipelines.copy_offset);
        enc.set_buffer(
            0,
            Some(&self.session.activations.hidden),
            src_row_offset_bytes,
        );
        enc.set_buffer(1, Some(&self.session.activations.final_hidden), 0);
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

    /// Encode the final head: optional pre-final hidden capture, RMS norm,
    /// logit GEMV, and optional top-k.
    ///
    /// All commands are appended to the already-open encoder `enc`.
    /// Returns `Some(which)` when compact top-k is active, `None` otherwise.
    pub(super) fn encode_final_head(
        &mut self,
        enc: &ComputeCommandEncoderRef,
        cfg: &Qwen35Config,
        capture_hidden: bool,
        prof: &mut StepProfile,
        profiling: bool,
    ) -> Option<u8> {
        let hidden = cfg.hidden_size;

        // === Capture pre-final hidden for MTP (before RMSNorm overwrites) ===
        if capture_hidden || self.session.mtp.is_some() {
            self.dispatch_copy(
                enc,
                &self.session.activations.hidden,
                &self.session.activations.pre_final_hidden,
                hidden as u32,
            );
        }

        // === Final RMS norm + logits (same command buffer) ===
        let final_t0 = profiling.then(std::time::Instant::now);
        self.dispatch_rms_norm(
            enc,
            &self.session.activations.hidden,
            &self.engine.final_norm,
            hidden as u32,
            1,
            cfg.rms_norm_eps,
        );
        // Site 3 of 3. Single-token path: the token of interest is row 0.
        self.encode_capture_final_hidden(enc, 0, hidden);
        // Issue #171: lm_head two-stage block-top-k. When the route requires
        // only argmax/top-k (never the full logit vector), skip the full
        // [vocab_size] GEMV entirely — Stage 1 fuses the GEMV with a
        // block-local exact reduction, Stage 2 (below) reduces those compact
        // candidates to the global result. This is the threadgroup-flood fix:
        // the full-logit GEMV alone dispatches tens of thousands of
        // threadgroups (ceil(vocab/4) or ceil(vocab/2)); Stage 1 dispatches
        // ceil(vocab/LM_HEAD_ROWS_PER_TG), ~100x fewer.
        let block_local_k = resolve_block_local_k(self.session.compact_route);
        if let Some((local_k, slot)) = block_local_k {
            let candidate_groups =
                self.dispatch_lm_head_block_stage1_enc(enc, cfg, 0, local_k, slot);
            let which = self.dispatch_block_topk_merge_enc(enc, candidate_groups, local_k);
            if let Some(t0) = final_t0 {
                prof.final_us += t0.elapsed().as_micros();
            }
            return Some(which);
        }

        // lm_head: for Q8 format use FP16 wide kernel (NR=4, fewer TGs than
        // gemv_decode_m1 which dispatches one TG per vocab row). For Q4 use Q4 path.
        match self.engine.quant_format {
            QuantFormat::Q8_0 => {
                let vocab_n = cfg.vocab_size as u32;
                let hidden_k = hidden as u32;
                enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_decode_wide);
                enc.set_buffer(0, Some(&self.session.activations.hidden), 0);
                enc.set_buffer(1, Some(self.engine.embed_tokens.f16_buffer()), 0);
                enc.set_buffer(2, Some(&self.session.activations.logits), 0);
                enc.set_bytes(3, 4, &vocab_n as *const u32 as *const _);
                enc.set_bytes(4, 4, &hidden_k as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(vocab_n.div_ceil(4) as u64, 1, 1),
                    MTLSize::new(32, 4, 1),
                );
            }
            QuantFormat::Q4_0 => {
                self.dispatch_matmul(
                    enc,
                    &self.session.activations.hidden,
                    &self.engine.embed_tokens_q8,
                    &self.session.activations.logits,
                    1,
                    cfg.vocab_size as u32,
                    hidden as u32,
                );
            }
        }
        if let Some(t0) = final_t0 {
            prof.final_us += t0.elapsed().as_micros();
        }

        // If compact mode, append top-k kernels to the same encoder before commit.
        // This avoids a second command-buffer round-trip (~18 µs overhead).
        if self.session.compact_topk > 0 {
            let k = self.session.compact_topk as u32;
            Some(self.dispatch_topk_enc(enc, cfg.vocab_size as u32, k))
        } else {
            None
        }
    }
}
