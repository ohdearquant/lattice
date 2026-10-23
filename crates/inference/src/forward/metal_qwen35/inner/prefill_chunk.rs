#[cfg(feature = "bench-internals")]
use super::{GDN_CHUNK_SIZE, bench_support, read_buffer_offset};
use super::{
    GpuCandidate, MetalLayerAttnWeights, MetalQwen35State, QuantFormat, div_ceil, read_buffer,
    resolve_block_local_k,
};
use metal::{MTLResourceOptions, MTLSize};
use std::sync::atomic::Ordering;

impl MetalQwen35State {
    /// Every active layer's FFN must be dense (`MetalFfnWeights::Dense`) for the
    /// batched (M>1 GEMM) prefill schedulers — `encode_prefill_dense_mlp` panics
    /// on `MetalFfnWeights::Moe`. Called before any embedding copy, command
    /// buffer, or per-layer encode so an unsupported config is rejected with zero
    /// state mutation instead of panicking mid-encode after earlier layers'
    /// GDN/KV work has already been recorded (or, for schedulers that split
    /// encoding across multiple command buffers, already committed to the GPU).
    fn assert_batched_prefill_dense_only(&self) {
        assert!(
            !self.engine.has_moe_layer,
            "batched prefill does not support MoE layers (M>1 GEMM path); \
                 use decode-mode forward_step for MoE models instead"
        );
    }

    // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
    // dispatch creates command buffers on the same long-lived worker thread.
    // Unmeasured individually; fixed on the structural argument, not on numbers.
    pub(super) fn forward_prefill_batched_chunk(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        all_positions: bool,
        emit_logits: bool,
        capture_hidden: bool,
    ) -> Vec<f32> {
        objc::rc::autoreleasepool(|| {
            self.forward_prefill_batched_chunk_dispatch(
                token_ids,
                start_pos,
                all_positions,
                emit_logits,
                capture_hidden,
            )
        })
    }

    /// Batched single-command-buffer prefill for a contiguous token slice starting
    /// at absolute position `start_pos`.
    ///
    /// Writes full-attention K/V rows `start_pos..start_pos+n` and advances GDN
    /// recurrent state in the session GPU buffers. Called by `forward_prefill_impl`
    /// once per chunk; for `n ≤ max_prefill` it IS the entire prefill (one command
    /// buffer, all 24 layers, final logits).
    ///
    /// `emit_logits` gates ONLY the terminal RMSNorm, lm_head, hidden capture/readback,
    /// and top-k tail, which is purely terminal (it does not feed the residual
    /// stream, KV cache, or GDN recurrent state — those are fully advanced by the
    /// per-layer loop above it). When `false`, the layer command buffer for this
    /// chunk still commits and `set_position` still advances; only the wasted tail
    /// work (and its GPU readback) is skipped. Callers with an intermediate chunk of
    /// a chunked, last-token-only (`!all_positions`) prefill — whose tail output is
    /// always discarded — pass `false`; every other caller passes `true`.
    fn forward_prefill_batched_chunk_dispatch(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        all_positions: bool,
        emit_logits: bool,
        capture_hidden: bool,
    ) -> Vec<f32> {
        let n = token_ids.len();
        debug_assert!(
            n <= self.session.max_prefill,
            "forward_prefill_batched_chunk: n={n} exceeds max_prefill={}",
            self.session.max_prefill
        );
        assert!(
            start_pos + n <= self.session.kv_cache.max_cache_len,
            "prefill chunk start_pos={start_pos} + n={n} exceeds max_cache_len {}",
            self.session.kv_cache.max_cache_len
        );
        self.assert_batched_prefill_dense_only();
        let cfg = self.engine.config.clone();
        let m = n as u32;
        let hidden = cfg.hidden_size;

        self.copy_prefill_embeddings(token_ids, hidden);

        let mut active_layer_idx = 0usize;
        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;

        let chunked_requested = self.use_gdn_chunked;
        let chunked_supported = Self::supports_gdn_chunked_prefill(&cfg);
        if chunked_requested && !chunked_supported {
            tracing::warn!(
                "chunked GDN prefill unsupported for this shape (0.8B only); \
                     falling back to serial scan (set LATTICE_GDN_CHUNKED=0 to suppress)"
            );
        }
        let chunked_enabled = chunked_requested && chunked_supported;

        // Single command buffer for entire prefill (all 24 layers + final logits)
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

            if is_linear {
                self.encode_gdn_prefill_before_recurrence(enc, compact_idx, layer_i, &cfg, m);
                self.encode_gdn_prefill_recurrence(
                    enc,
                    compact_idx,
                    linear_idx,
                    &cfg,
                    n,
                    chunked_enabled,
                );
                self.encode_gdn_prefill_after_recurrence(enc, compact_idx, layer_i, &cfg, m);
                linear_idx += 1;
            } else {
                self.encode_full_attention_prefill_layer(
                    enc,
                    compact_idx,
                    layer_i,
                    full_idx,
                    &cfg,
                    n,
                    m,
                    start_pos,
                );
                full_idx += 1;
            }
        }

        // MTP prefill capture (#1340): `activations.hidden` currently holds the
        // raw, pre-final-RMSNorm target hidden state for every one of this
        // chunk's `n` positions. The tail below overwrites row `n-1` (default)
        // or every row (`all_positions`) with its post-RMSNorm value in place,
        // so this read must land before that dispatch runs — hence ending and
        // committing the layer-loop's command buffer here instead of after the
        // tail. `mtp_prefill` (called from `generate()`) drains the accumulated
        // rows to rebuild the MTP draft head's KV cache from the prompt. Gated
        // on `mtp.is_some()` so a session without an MTP head pays nothing extra
        // — the pre-existing single-command-buffer path below is untouched.
        let (cmd, enc) = if self.session.mtp.is_some() {
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            // SAFETY: the command buffer just completed; `hidden` (the
            // activations buffer, not the local `hidden: usize` row-length) is
            // StorageModeShared and the layer loop above wrote exactly `n`
            // valid rows into it.
            let raw_hidden = unsafe { read_buffer(&self.session.activations.hidden, n * hidden) };
            self.session
                .mtp_prefill_hidden
                .extend_from_slice(&raw_hidden);

            if !emit_logits {
                self.session.set_position(start_pos + n);
                return vec![];
            }

            let cmd = self.engine.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            (cmd, enc)
        } else {
            (cmd, enc)
        };

        if !emit_logits {
            // Intermediate chunk of a chunked, `!all_positions` prefill: the caller
            // only ever keeps the LAST chunk's logits, so the terminal tail below
            // (MTP capture, RMSNorm, lm_head, top-k, and their GPU readback) is pure
            // waste here — it does not feed the residual stream, KV cache, or GDN
            // recurrent state, which the per-layer loop above has already fully
            // advanced. Still commit this chunk's layer command buffer and advance
            // the session cursor exactly as the emit_logits==true path does below.
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            self.session.set_position(start_pos + n);
            return vec![];
        }

        // ---------- Final: RMSNorm + lm_head ----------
        // Two modes:
        //   - all_positions=false: RMSNorm + lm_head GEMV for last token only
        //     (default — used by chat/decode prefill, returns vocab_size f32)
        //   - all_positions=true:  RMSNorm + lm_head GEMM for all n tokens
        //     (used by perplexity eval, returns n*vocab_size f32, allocates a
        //     per-call buffer of n*vocab*4 bytes)
        let last_off = ((n - 1) * hidden) as u64 * 4;

        // Per-call output buffer for all-positions mode (n * vocab f32).
        // Allocated lazily to keep the chat path zero-overhead.
        let ppl_buf = if all_positions {
            let bytes = (n * cfg.vocab_size * 4) as u64;
            Some(
                self.engine
                    .device
                    .new_buffer(bytes, MTLResourceOptions::StorageModeShared),
            )
        } else {
            None
        };

        // Pre-final-hidden capture (last-token only; not needed for ppl).
        if !all_positions && (capture_hidden || self.session.mtp.is_some()) {
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

        // RMSNorm — single-token (last) or all n tokens.
        {
            enc.set_compute_pipeline_state(&self.engine.pipelines.rms_norm);
            let (off, nr) = if all_positions {
                (0u64, n as u32)
            } else {
                (last_off, 1u32)
            };
            enc.set_buffer(0, Some(&self.session.activations.hidden), off);
            enc.set_buffer(1, Some(&self.engine.final_norm), 0);
            let rl = hidden as u32;
            enc.set_bytes(2, 4, &rl as *const u32 as *const _);
            enc.set_bytes(3, 4, &nr as *const u32 as *const _);
            enc.set_bytes(4, 4, &cfg.rms_norm_eps as *const f32 as *const _);
            // One threadgroup per row in batched mode, one for single-row mode.
            enc.dispatch_thread_groups(MTLSize::new(nr as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
        // Site 2 of 3. `last_off` is normed under both branches above:
        // `all_positions` norms every row including the last, and the
        // single-token branch norms exactly that row.
        self.encode_capture_final_hidden(enc, last_off, hidden);

        // lm_head — for Q8 format (auto-quantized from F16 source), use the
        // FP16 embed_tokens buffer to avoid the ~0.79 PPL gap from per-row
        // symmetric Q8 quantization (which can't represent outlier rows
        // cleanly, corrupting hidden·embed dot products vs MLX).
        // For Q4 format (from .q4 dirs), keep using the Q4 path because
        // it is faster and handles QuaRot rotations; a Q4 checkpoint has no f16 table.
        // Issue #171: when the last-token route only needs argmax/top-k (never
        // the full logit vector), skip the full [vocab_size] GEMV entirely —
        // Stage 1 fuses the GEMV with a block-local exact reduction, Stage 2
        // (dispatch_block_topk_merge_enc) reduces those compact candidates to
        // the global result. all_positions (perplexity) always takes the
        // full-logit path below; it never sets compact_route to a Block route
        // in the generate loop, but we gate here too as defense in depth.
        let block_local_k = if all_positions {
            None
        } else {
            resolve_block_local_k(self.session.compact_route)
        };

        let topk_which = if let Some((local_k, slot)) = block_local_k {
            let candidate_groups =
                self.dispatch_lm_head_block_stage1_enc(enc, &cfg, last_off, local_k, slot);
            Some(self.dispatch_block_topk_merge_enc(enc, candidate_groups, local_k))
        } else {
            let use_fp16_lm_head = matches!(self.engine.quant_format, QuantFormat::Q8_0);
            if use_fp16_lm_head {
                let vocab_n = cfg.vocab_size as u32;
                let hidden_k = hidden as u32;
                if let Some(ref pb) = ppl_buf {
                    for p in 0..n {
                        let hidden_off = (p * hidden) as u64 * 4;
                        let logits_off = (p * cfg.vocab_size) as u64 * 4;
                        enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_decode_wide);
                        enc.set_buffer(0, Some(&self.session.activations.hidden), hidden_off);
                        enc.set_buffer(1, Some(self.engine.embed_tokens.f16_buffer()), 0);
                        enc.set_buffer(2, Some(pb), logits_off);
                        enc.set_bytes(3, 4, &vocab_n as *const u32 as *const _);
                        enc.set_bytes(4, 4, &hidden_k as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            MTLSize::new(vocab_n.div_ceil(4) as u64, 1, 1),
                            MTLSize::new(32, 4, 1),
                        );
                    }
                } else {
                    enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_decode_wide);
                    enc.set_buffer(0, Some(&self.session.activations.hidden), last_off);
                    enc.set_buffer(1, Some(self.engine.embed_tokens.f16_buffer()), 0);
                    enc.set_buffer(2, Some(&self.session.activations.logits), 0);
                    enc.set_bytes(3, 4, &vocab_n as *const u32 as *const _);
                    enc.set_bytes(4, 4, &hidden_k as *const u32 as *const _);
                    enc.dispatch_thread_groups(
                        MTLSize::new(vocab_n.div_ceil(4) as u64, 1, 1),
                        MTLSize::new(32, 4, 1),
                    );
                }
            } else if let Some(ref pb) = ppl_buf {
                self.dispatch_gemm(
                    enc,
                    &self.session.activations.hidden,
                    0,
                    &self.engine.embed_tokens_q8,
                    pb,
                    0,
                    n as u32,
                    cfg.vocab_size as u32,
                    hidden as u32,
                );
            } else {
                self.dispatch_gemm(
                    enc,
                    &self.session.activations.hidden,
                    last_off,
                    &self.engine.embed_tokens_q8,
                    &self.session.activations.logits,
                    0,
                    1,
                    cfg.vocab_size as u32,
                    hidden as u32,
                );
            }

            // Append top-k in the same encoder when compact mode is active (last-token only).
            if !all_positions && self.session.compact_topk > 0 {
                let k = self.session.compact_topk as u32;
                Some(self.dispatch_topk_enc(enc, cfg.vocab_size as u32, k))
            } else {
                None
            }
        };

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // `set_position` clears the freshness marker, so it runs before the capture
        // below, which marks the new cursor.
        self.session.set_position(start_pos + n);
        if !all_positions && (capture_hidden || self.session.mtp.is_some()) {
            if capture_hidden && self.path_proof_enabled {
                self.path_proof
                    .prefill_hidden_readback
                    .fetch_add(1, Ordering::Relaxed);
            }
            // SAFETY: GPU completed, pre_final_hidden is StorageModeShared.
            self.session.last_pre_final_hidden =
                unsafe { read_buffer(&self.session.activations.pre_final_hidden, hidden) };
            self.session.mark_pre_final_hidden(start_pos + n);
        }

        if let Some(pb) = ppl_buf {
            // SAFETY: GPU completed, ppl_buf is StorageModeShared and sized n*vocab.
            if self.path_proof_enabled {
                self.path_proof
                    .prefill_full_vocab_logit_readback
                    .fetch_add(1, Ordering::Relaxed);
                // `n *` is the whole reason the byte counter exists: this site and
                // the ordinary one below share a counter and move different amounts.
                self.path_proof
                    .prefill_full_vocab_logit_readback_bytes
                    .fetch_add(
                        (n * cfg.vocab_size * std::mem::size_of::<f32>()) as u64,
                        Ordering::Relaxed,
                    );
            }
            unsafe { read_buffer(&pb, n * cfg.vocab_size) }
        } else if let Some(which) = topk_which {
            // SAFETY: GPU completed, buffers are StorageModeShared.
            if self.path_proof_enabled {
                self.path_proof
                    .prefill_compact_candidate_logit_readback
                    .fetch_add(1, Ordering::Relaxed);
                self.path_proof
                    .prefill_compact_candidate_logit_readback_bytes
                    .fetch_add(
                        (self.session.compact_topk * std::mem::size_of::<GpuCandidate>()) as u64,
                        Ordering::Relaxed,
                    );
            }
            let k = self.session.compact_topk;
            let candidates = unsafe { self.read_topk_candidates(which, k) };
            self.session.compact_result = candidates;
            vec![]
        } else {
            // SAFETY: The command buffer has completed and logits is a
            // StorageModeShared buffer sized for vocab_size f32 values.
            if self.path_proof_enabled {
                self.path_proof
                    .prefill_full_vocab_logit_readback
                    .fetch_add(1, Ordering::Relaxed);
                self.path_proof
                    .prefill_full_vocab_logit_readback_bytes
                    .fetch_add(
                        (cfg.vocab_size * std::mem::size_of::<f32>()) as u64,
                        Ordering::Relaxed,
                    );
            }
            unsafe { read_buffer(&self.session.activations.logits, cfg.vocab_size) }
        }
    }

    // lattice#1584: same autorelease contract as `forward_step_inner_impl`. This
    // dispatch creates command buffers on the same long-lived worker thread.
    // Unmeasured individually; fixed on the structural argument, not on numbers.
    #[cfg(feature = "bench-internals")]
    pub(super) fn forward_prefill_chunk_gdn_isolated(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        capture: &[bench_support::GdnCaptureRequest],
    ) -> (
        bench_support::GdnIsolatedChunkTiming,
        Vec<bench_support::GdnLayerCapture>,
    ) {
        objc::rc::autoreleasepool(|| {
            self.forward_prefill_chunk_gdn_isolated_dispatch(token_ids, start_pos, capture)
        })
    }

    /// GDN-recurrence-isolating variant of [`Self::forward_prefill_batched_chunk`],
    /// built for the #175 bench harness (`bin/bench_gdn_prefill_ab.rs`).
    ///
    /// Structurally identical to `forward_prefill_batched_chunk`: same op sequence,
    /// same buffers, same dispatch calls, same per-layer branch (GDN vs full
    /// attention). The ONLY difference is *where command-buffer boundaries fall*:
    /// immediately before and after each GDN layer's recurrence dispatch (either
    /// `dispatch_gdn_chunked_prefill_layer`'s C32 family, or the serial
    /// conv1d+recurrence per-token loop) the current encoder is ended and its
    /// command buffer committed + `wait_until_completed`, a fresh command buffer
    /// is opened for the GDN dispatch alone, then another fresh one for the rest of
    /// the layer. This isolates the GDN segment's CPU wall-clock time from the
    /// surrounding GEMMs/norms so the two scans' recurrence cost can be compared
    /// without the O(n^2) attention layers (or the GDN layers' own non-recurrence
    /// GEMMs) swamping the signal.
    ///
    /// No GDN kernel, dispatch argument, threadgroup geometry, or buffer differs
    /// from the production path — only the encoder/command-buffer boundaries
    /// change. This cannot introduce a new GDN-state race: within one encoder,
    /// Metal already serializes same-buffer read/write hazards between
    /// consecutive dispatches (the production path's correctness already depends
    /// on this), and an explicit `commit` + `wait_until_completed` between two
    /// *separate* command buffers on the same StorageModeShared buffers is a
    /// strictly stronger ordering guarantee than that. Splitting into more,
    /// smaller command buffers can only add synchronization stalls, never change
    /// what a kernel reads or writes.
    ///
    /// Always skips the terminal RMSNorm/lm_head/top-k tail (mirrors
    /// `forward_prefill_batched_chunk`'s `emit_logits = false` path) — logits are
    /// never needed for a dispatch-timing measurement, skipping the tail removes a
    /// large amount of code unrelated to GDN isolation, and the tail's cost is
    /// identical between the serial and chunked paths regardless (it depends on
    /// neither), so omitting it does not bias the serial-vs-chunked comparison.
    ///
    /// Handles exactly one `max_prefill`-sized chunk starting at `start_pos`
    /// (`n = token_ids.len() <= self.session.max_prefill`, asserted below); the
    /// caller drives the `max_prefill`-sized chunking loop for prompts longer than
    /// `max_prefill`, exactly as `forward_prefill_impl` does for the production
    /// path, threading GDN recurrent state across chunk boundaries via the
    /// session's GPU buffers (unchanged by this method).
    ///
    /// Gated on `bench-internals` (like `bench_support` itself, which owns this
    /// method's return type) — this is bench-only isolation plumbing, not part of
    /// the production dispatch path, so it stays out of the default `metal-gpu`
    /// build the way `bench_support`'s other helpers do.
    ///
    /// `capture`: optional read-only revalidation hooks (gdn175 S1b). For every
    /// entry whose `linear_idx` matches the GDN layer currently being processed,
    /// immediately after that layer's GDN command buffer has completed (and
    /// before the next GDN layer's materialize kernel can overwrite the shared
    /// scratch buffers), this reads back that entry's `value_head`'s per-token
    /// q/k/v/beta/log_alpha straight off the session's `gdn_chunk` scratch —
    /// see `capture_gdn_chunk_scratch`. Pass `&[]` for the normal timing-only
    /// path (unchanged behavior/dispatch either way; this never perturbs what
    /// gets dispatched, only what gets read back afterward).
    #[cfg(feature = "bench-internals")]
    fn forward_prefill_chunk_gdn_isolated_dispatch(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        capture: &[bench_support::GdnCaptureRequest],
    ) -> (
        bench_support::GdnIsolatedChunkTiming,
        Vec<bench_support::GdnLayerCapture>,
    ) {
        let n = token_ids.len();
        assert!(
            n <= self.session.max_prefill,
            "forward_prefill_chunk_gdn_isolated: n={n} exceeds max_prefill={}",
            self.session.max_prefill
        );
        assert!(
            start_pos + n <= self.session.kv_cache.max_cache_len,
            "prefill chunk start_pos={start_pos} + n={n} exceeds max_cache_len {}",
            self.session.kv_cache.max_cache_len
        );
        self.assert_batched_prefill_dense_only();
        let cfg = self.engine.config.clone();
        let m = n as u32;
        let hidden = cfg.hidden_size;

        self.copy_prefill_embeddings(token_ids, hidden);

        let mut active_layer_idx = 0usize;
        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;

        let chunked_requested = self.use_gdn_chunked;
        let chunked_supported = Self::supports_gdn_chunked_prefill(&cfg);
        // Bench-internals path: fail LOUD instead of falling back. The
        // production path degrades gracefully to the serial recurrence, but a
        // bench that did the same would measure the serial path while the
        // harness labels the arm "chunked" — a silent instrument corruption.
        assert!(
            !chunked_requested || chunked_supported,
            "GDN chunked prefill requested (set_gdn_chunked(true)) but this config \
                 does not support the chunked path — refusing to silently measure the \
                 serial recurrence under a 'chunked' label"
        );
        let chunked_enabled = chunked_requested && chunked_supported;

        let mut timing = bench_support::GdnIsolatedChunkTiming::default();
        let mut captured: Vec<bench_support::GdnLayerCapture> = Vec::new();

        let mut cmd = self.engine.queue.new_command_buffer();
        let mut enc = cmd.new_compute_command_encoder();
        let mut seg_start = std::time::Instant::now();

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
                self.encode_gdn_prefill_before_recurrence(enc, compact_idx, layer_i, &cfg, m);

                // ---- GDN isolation boundary: close the pre-GDN (non-GDN) segment ----
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                timing.non_gdn_ms += seg_start.elapsed().as_secs_f64() * 1000.0;

                let gdn_cmd = self.engine.queue.new_command_buffer();
                let gdn_enc = gdn_cmd.new_compute_command_encoder();
                let gdn_start = std::time::Instant::now();

                self.encode_gdn_prefill_recurrence(
                    gdn_enc,
                    compact_idx,
                    linear_idx,
                    &cfg,
                    n,
                    chunked_enabled,
                );

                gdn_enc.end_encoding();
                gdn_cmd.commit();
                gdn_cmd.wait_until_completed();
                timing.gdn_ms += gdn_start.elapsed().as_secs_f64() * 1000.0;
                timing.num_gdn_layers += 1;

                // gdn175 S1b: read-only capture, right after this layer's GDN
                // command buffer completed and before the next GDN layer's
                // materialize kernel can overwrite the shared gdn_chunk scratch
                // (see capture_gdn_chunk_scratch's doc comment for the safety
                // argument). `linear_idx` here is still this layer's index —
                // it only increments at the bottom of this branch.
                for req in capture {
                    if req.linear_idx == linear_idx {
                        captured.push(self.capture_gdn_chunk_scratch(
                            linear_idx,
                            req.value_head,
                            n,
                        ));
                    }
                }

                // ---- reopen a fresh segment for the rest of this layer + subsequent layers ----
                cmd = self.engine.queue.new_command_buffer();
                enc = cmd.new_compute_command_encoder();
                seg_start = std::time::Instant::now();

                self.encode_gdn_prefill_after_recurrence(enc, compact_idx, layer_i, &cfg, m);
                linear_idx += 1;
            } else {
                self.encode_full_attention_prefill_layer(
                    enc,
                    compact_idx,
                    layer_i,
                    full_idx,
                    &cfg,
                    n,
                    m,
                    start_pos,
                );
                full_idx += 1;
            }
        }

        // Close the trailing non-GDN segment (mirrors the emit_logits=false early
        // return in forward_prefill_batched_chunk: still commit + advance the
        // session cursor, just never compute logits).
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        timing.non_gdn_ms += seg_start.elapsed().as_secs_f64() * 1000.0;

        self.session.set_position(start_pos + n);
        (timing, captured)
    }

    /// gdn175 S1b: read-only capture of one (GDN layer, value head)'s
    /// materialized per-token chunkwise-scan inputs — q/k/v/beta/log_alpha —
    /// off the session's shared `gdn_chunk` scratch buffers, de-chunked from
    /// `[chunks, heads, C, dim]` back to `[n_tokens, dim]` token order.
    ///
    /// SAFETY / correctness argument (verified against
    /// `forward/shaders/qwen35.metal`): `sc.q`/`sc.k`/`sc.v`/`sc.beta_log_alpha`
    /// are written ONLY by `gdn_chunk_materialize_c32` and read (`device const`,
    /// never written) by `gdn_chunk_solve_c32` and
    /// `gdn_chunk_residual_output_c32`; `gdn_chunk_conv_buf_update_c32`,
    /// `gdn_chunk_state_update_c32`, and `gdn_chunk_norm_silu_c32` never touch
    /// them at all. So once this layer's `gdn_cmd.wait_until_completed()` has
    /// returned (the call site, immediately before this method runs), these
    /// four buffers hold their final values for this layer and stay untouched
    /// until the NEXT GDN layer's own materialize dispatch reuses the same
    /// session-owned scratch. Reading them here is a pure CPU-side copy off a
    /// StorageModeShared buffer with no GPU work in flight — it cannot perturb
    /// the dispatch this call is capturing.
    ///
    /// De-chunking: `dispatch_gdn_chunk_materialize_c32` writes row
    /// `((chunk*num_value_heads + head)*GDN_CHUNK_SIZE + j)*dim + d` for token
    /// `chunk*GDN_CHUNK_SIZE + j` (see `qwen35.metal`'s `head_row` computation);
    /// only the LAST active chunk is short (materialize's `ci = min(chunk_size,
    /// n_tokens - chunk_base)` loop bound leaves the tail rows at their
    /// zero-initialized default). Reading exactly `rows_in_chunk * dim`
    /// contiguous elements per chunk therefore already skips the zero-padded
    /// tail rows and yields the valid tokens in order — no separate reshuffle
    /// needed after concatenating the per-chunk reads.
    ///
    /// `log_alpha` is returned RAW (natural-log, per-token, NOT cumulative) —
    /// the S1b spec's `alpha[t] = exp(log_alpha[t])` transform is left to the
    /// caller so the oracle-feeding step stays visibly separate from capture.
    #[cfg(feature = "bench-internals")]
    fn capture_gdn_chunk_scratch(
        &self,
        linear_idx: usize,
        value_head: usize,
        n_tokens: usize,
    ) -> bench_support::GdnLayerCapture {
        let cfg = &self.engine.config;
        let num_vh = cfg.linear_num_value_heads();
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        assert!(
            value_head < num_vh,
            "capture_gdn_chunk_scratch: value_head={value_head} out of range \
                 (num_value_heads={num_vh})"
        );
        assert!(
            n_tokens > 0,
            "capture_gdn_chunk_scratch: n_tokens must be > 0"
        );

        let num_chunks_active = n_tokens.div_ceil(GDN_CHUNK_SIZE);
        let tail = n_tokens - (num_chunks_active - 1) * GDN_CHUNK_SIZE;

        let sc = &self.session.activations.gdn_chunk;
        let mut q = Vec::with_capacity(n_tokens * kd);
        let mut k = Vec::with_capacity(n_tokens * kd);
        let mut v = Vec::with_capacity(n_tokens * vd);
        let mut beta = Vec::with_capacity(n_tokens);
        let mut log_alpha = Vec::with_capacity(n_tokens);

        for chunk in 0..num_chunks_active {
            let rows = if chunk + 1 == num_chunks_active {
                tail
            } else {
                GDN_CHUNK_SIZE
            };
            let base_row = (chunk * num_vh + value_head) * GDN_CHUNK_SIZE;
            // SAFETY: see this method's doc comment — GPU completed, read-only,
            // no concurrent write to sc.q/k/v/beta_log_alpha at this point.
            unsafe {
                q.extend_from_slice(&read_buffer_offset(&sc.q, base_row * kd, rows * kd));
                k.extend_from_slice(&read_buffer_offset(&sc.k, base_row * kd, rows * kd));
                v.extend_from_slice(&read_buffer_offset(&sc.v, base_row * vd, rows * vd));
                let bla = read_buffer_offset(&sc.beta_log_alpha, base_row * 2, rows * 2);
                for pair in bla.chunks_exact(2) {
                    beta.push(pair[0]);
                    log_alpha.push(pair[1]);
                }
            }
        }

        debug_assert_eq!(q.len(), n_tokens * kd);
        debug_assert_eq!(k.len(), n_tokens * kd);
        debug_assert_eq!(v.len(), n_tokens * vd);
        debug_assert_eq!(beta.len(), n_tokens);
        debug_assert_eq!(log_alpha.len(), n_tokens);

        bench_support::GdnLayerCapture {
            linear_idx,
            value_head,
            length: n_tokens,
            dk: kd,
            dv: vd,
            q,
            k,
            v,
            beta,
            log_alpha,
        }
    }
}
