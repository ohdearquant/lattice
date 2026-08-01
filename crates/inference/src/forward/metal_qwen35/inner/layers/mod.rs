mod gdn;

use super::*;

impl MetalQwen35State {
    /// Dispatch LoRA GEMV for a single projection: y += scale * B @ (A @ x).
    ///
    /// `x_byte_offset` is the byte offset into `x` where the input vector starts (usually 0,
    /// non-zero for sub-slices of fused buffers such as the Z portion of `gdn_qkvz`).
    /// `y_byte_offset` is the byte offset into `y` where the output vector starts (usually 0,
    /// non-zero when accumulating into a sub-slice of a fused output buffer).
    ///
    /// No-op if no adapter is loaded or no adapter exists for the given layer/module.
    pub(super) fn dispatch_lora_if_active(
        &self,
        enc: &ComputeCommandEncoderRef,
        x: &Buffer,
        x_byte_offset: u64,
        y: &Buffer,
        y_byte_offset: u64,
        layer_idx: usize,
        module: &str,
    ) {
        let Some(adapter) = &self.lora else { return };
        let Some(proj) = adapter.get_projection(layer_idx, module) else {
            return;
        };

        // Phase 1: intermediate = A @ x  (x read from x_byte_offset)
        enc.set_compute_pipeline_state(&self.engine.pipelines.lora_gemv_a);
        enc.set_buffer(0, Some(x), x_byte_offset);
        enc.set_buffer(1, Some(&proj.a_buf), 0);
        enc.set_buffer(2, Some(&adapter.intermediate), 0);
        enc.set_bytes(3, 4, &proj.rank as *const u32 as *const _);
        enc.set_bytes(4, 4, &proj.d_in as *const u32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(proj.rank as u64, 1, 1), MTLSize::new(32, 4, 1));

        // Phase 2: y += scale * B @ intermediate  (y written at y_byte_offset)
        enc.set_compute_pipeline_state(&self.engine.pipelines.lora_gemv_b_accum);
        enc.set_buffer(0, Some(&adapter.intermediate), 0);
        enc.set_buffer(1, Some(&proj.b_buf), 0);
        enc.set_buffer(2, Some(y), y_byte_offset);
        enc.set_bytes(3, 4, &proj.d_out as *const u32 as *const _);
        enc.set_bytes(4, 4, &proj.rank as *const u32 as *const _);
        enc.set_bytes(5, 4, &adapter.scale as *const f32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(proj.d_out.div_ceil(256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    /// Encode the MLP tail shared by every layer (both GDN and GQA).
    ///
    /// Dispatches: fused residual-add-norm → gate_up_proj GEMV → LoRA(gate/up) →
    /// silu_mul → down_proj GEMV → LoRA(down) → residual-add-copy.
    ///
    /// Pure extraction from `encode_gdn_layer` / `encode_gqa_layer`; the fused
    /// path is byte-for-byte identical to inlining.
    ///
    /// # Preconditions
    /// - `self.session.activations.residual` holds the post-attention residual.
    /// - `self.session.activations.attn_out` holds the attention output.
    /// - `self.session.activations.hidden` is scratch space.
    ///
    /// # Postconditions
    /// - `self.session.activations.residual` and `.hidden` hold the post-FFN state.
    pub(super) fn encode_mlp_block(
        &mut self,
        enc: &ComputeCommandEncoderRef,
        compact_idx: usize,
        layer_idx: usize,
        position: usize,
        cfg: &Qwen35Config,
        prof: &mut StepProfile,
        profiling: bool,
    ) {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let (_, common_w) = &self.engine.layer_weights[compact_idx];
        let mlp_t0 = profiling.then(std::time::Instant::now);
        self.dispatch_fused_residual_add_norm(
            enc,
            &self.session.activations.residual,
            &self.session.activations.attn_out,
            &self.session.activations.residual,
            &self.session.activations.hidden,
            &common_w.post_attention_layernorm,
            hidden as u32,
            cfg.rms_norm_eps,
        );
        // SAFETY: FFN weight buffers are live for the command buffer lifetime.
        match &common_w.ffn {
            MetalFfnWeights::Dense {
                gate_up_proj,
                down_proj,
            } => {
                let w_gate_up = gate_up_proj as *const Q4WeightBuf;
                let w_down = down_proj as *const Q4WeightBuf;
                unsafe {
                    self.dispatch_matmul(
                        enc,
                        &self.session.activations.hidden,
                        &*w_gate_up,
                        &self.session.activations.gate,
                        1,
                        (2 * inter) as u32,
                        hidden as u32,
                    );
                }
                self.dispatch_lora_if_active(
                    enc,
                    &self.session.activations.hidden,
                    0,
                    &self.session.activations.gate,
                    0,
                    layer_idx,
                    "gate_proj",
                );
                self.dispatch_lora_if_active(
                    enc,
                    &self.session.activations.hidden,
                    0,
                    &self.session.activations.gate,
                    (inter * std::mem::size_of::<f32>()) as u64,
                    layer_idx,
                    "up_proj",
                );
                self.dispatch_silu_mul_fused(enc, &self.session.activations.gate, inter as u32);
                unsafe {
                    self.dispatch_matmul(
                        enc,
                        &self.session.activations.gate,
                        &*w_down,
                        &self.session.activations.ffn_out,
                        1,
                        hidden as u32,
                        inter as u32,
                    );
                }
                self.dispatch_lora_if_active(
                    enc,
                    &self.session.activations.gate,
                    0,
                    &self.session.activations.ffn_out,
                    0,
                    layer_idx,
                    "down_proj",
                );
            }
            MetalFfnWeights::Moe(moe_bufs) => {
                let moe_ptr = moe_bufs.as_ref() as *const MoeMetalBuffers;
                // SAFETY: moe_bufs is owned by layer_weights which is live.
                unsafe {
                    self.encode_moe_ffn(enc, &*moe_ptr, cfg, layer_idx, position);
                }
            }
        }
        self.dispatch_add_and_copy(
            enc,
            &self.session.activations.ffn_out,
            &self.session.activations.residual,
            &self.session.activations.hidden,
            hidden as u32,
        );
        if let Some(t0) = mlp_t0 {
            prof.mlp_us += t0.elapsed().as_micros();
        }
    }

    /// Encode a MoE FFN step into an already-open compute command encoder.
    ///
    /// Implements ADR-053 D1–D4: CPU routing + single-encoder GPU dispatch for
    /// all routed experts plus the shared expert, accumulating with router weights
    /// via `moe_scale_add` / `moe_shared_gate_add` kernels.
    ///
    /// # Preconditions
    /// - `moe` is a live reference to a `MoeMetalBuffers` owned by `self.engine.layer_weights`.
    /// - `self.session.activations.hidden` holds the post-RMSNorm hidden state ([hidden] f32).
    /// - The encoder `enc` is open and NOT yet committed.
    ///
    /// # Postconditions
    /// - `self.session.activations.ffn_out` holds the MoE FFN output ([hidden] f32)
    ///   ready for residual accumulation by the caller.
    ///
    /// # Safety
    /// Reads `moe.router_gate.contents()` and `moe.shared_expert_gate.contents()` as
    /// `*const f32` for CPU routing. Valid because both buffers use `StorageModeShared`
    /// and no GPU work on those buffers is in flight (router runs before encoding starts).
    /// Reads `self.session.activations.hidden.contents()` similarly (GPU idle at this point
    /// — this is called before the encoder dispatches any hidden-state kernel).
    unsafe fn encode_moe_ffn(
        &self,
        enc: &ComputeCommandEncoderRef,
        moe: &MoeMetalBuffers,
        _cfg: &Qwen35Config,
        layer_idx: usize,
        token_idx: usize,
    ) {
        let hidden = moe.hidden;
        let inter = moe.inter;
        let shared_inter = moe.shared_inter;
        let num_experts = moe.num_experts;
        let top_k = moe.top_k;

        // ── Step 0: Zero the accumulator on GPU ───────────────────────────────
        let hidden_u32 = hidden as u32;
        enc.set_compute_pipeline_state(&self.engine.pipelines.moe_zero_buf);
        enc.set_buffer(0, Some(&moe.scratch_out), 0);
        enc.set_bytes(1, 4, &hidden_u32 as *const u32 as *const _);
        let wg = 256u64;
        enc.dispatch_threads(
            MTLSize::new(div_ceil(hidden as u64, wg) * wg, 1, 1),
            MTLSize::new(wg, 1, 1),
        );

        // ── Step 1: CPU routing ───────────────────────────────────────────────
        // Read hidden activation (post-RMSNorm, CPU-accessible StorageModeShared).
        let hidden_ptr = self.session.activations.hidden.contents() as *const f32;
        let hidden_slice = std::slice::from_raw_parts(hidden_ptr, hidden);

        // Read router gate weights [num_experts, hidden] f32.
        let gate_ptr = moe.router_gate.contents() as *const f32;
        let gate_slice = std::slice::from_raw_parts(gate_ptr, num_experts * hidden);

        // Compute router logits: logits[e] = dot(hidden, gate_w[e])
        let mut logits = vec![0.0f32; num_experts];
        for e in 0..num_experts {
            let row = &gate_slice[e * hidden..(e + 1) * hidden];
            let mut acc = 0.0f32;
            for i in 0..hidden {
                acc += hidden_slice[i] * row[i];
            }
            logits[e] = acc;
        }

        // Softmax over all experts (numerically stable).
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for v in logits.iter_mut() {
            *v = (*v - max_logit).exp();
            denom += *v;
        }
        if denom > 0.0 {
            for v in logits.iter_mut() {
                *v /= denom;
            }
        }

        // Select top-k experts (insertion-sort into fixed-size array).
        let mut selected: Vec<(usize, f32)> = vec![(usize::MAX, f32::NEG_INFINITY); top_k];
        for (e, &prob) in logits.iter().enumerate() {
            for rank in 0..top_k {
                if prob > selected[rank].1 {
                    for shift in (rank + 1..top_k).rev() {
                        selected[shift] = selected[shift - 1];
                    }
                    selected[rank] = (e, prob);
                    break;
                }
            }
        }

        // Test-only override (see `FORCED_MOE_EXPERTS_FOR_TEST`): replace
        // the CPU-computed selection with a caller-chosen expert list so
        // tests can drive a specific `ExpertSlotCache::resolve` sequence
        // through this real, encoded command buffer.
        #[cfg(test)]
        FORCED_MOE_EXPERTS_FOR_TEST.with(|c| {
            if let Some(forced_ids) = c.borrow().as_ref() {
                assert_eq!(
                    forced_ids.len(),
                    top_k,
                    "set_forced_moe_experts_for_test: forced expert list length must equal top_k"
                );
                let weight = 1.0f32 / top_k as f32;
                for (slot, &expert_id) in selected.iter_mut().zip(forced_ids.iter()) {
                    *slot = (expert_id, weight);
                }
            }
        });

        // Renormalize selected weights to sum=1.
        let top_sum: f32 = selected.iter().map(|(_, p)| *p).sum();
        if top_sum > 0.0 {
            for (_, prob) in selected.iter_mut() {
                *prob /= top_sum;
            }
        }

        // #682 Stage 4: optional routing-divergence trace. Single
        // branch-on-None (`with` + `borrow` on an unset thread-local) when
        // disarmed, the production default — see `MOE_ROUTING_TRACE`.
        MOE_ROUTING_TRACE.with(|c| {
            if let Some(trace) = c.borrow_mut().as_mut() {
                trace.push(MoeRoutingTraceRecord {
                    layer_idx,
                    token_idx,
                    selected_ids: selected.iter().map(|(id, _)| *id).collect(),
                    gate_weights: selected.iter().map(|(_, w)| *w).collect(),
                });
            }
        });

        // ── Step 2: Shared expert (always active) ─────────────────────────────
        // Extracted into a closure (called exactly once, either inline
        // below or from inside the prefetch scope's spawn/join window
        // just after this) so its CPU encode time can overlap the
        // routed-expert dequant phase without duplicating this body —
        // see `moe_expert_cache`'s module doc comment.
        let encode_step2 = || {
            // Compute scalar sigmoid gate on CPU.
            // SAFETY: `moe.shared_expert_gate` is a valid
            // StorageModeShared buffer of at least `hidden` f32
            // elements (same layout `MoeMetalBuffers` was constructed
            // with) — same raw-pointer access pattern as this
            // function's Step 1 above, just inside a nested closure
            // (which does not inherit `encode_moe_ffn`'s
            // `unsafe fn`-body allowance, so this needs its own block).
            let sg_slice = unsafe {
                let sg_ptr = moe.shared_expert_gate.contents() as *const f32;
                std::slice::from_raw_parts(sg_ptr, hidden)
            };
            let gate_logit: f32 = hidden_slice
                .iter()
                .zip(sg_slice.iter())
                .map(|(x, g)| x * g)
                .sum();
            let shared_gate_val = 1.0f32 / (1.0 + (-gate_logit).exp());

            // Shared expert gate projection GEMV: scratch_gate[shared_inter] = W_gate * hidden
            let shared_inter_u32 = shared_inter as u32;
            let params_gate_up_sh = GemmParams {
                m: 1,
                n: shared_inter_u32,
                k: hidden_u32,
                lda: hidden_u32,
                ldb: hidden_u32,
                ldc: shared_inter_u32,
            };
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_decode);
            enc.set_buffer(0, Some(&self.session.activations.hidden), 0);
            enc.set_buffer(1, Some(&moe.shared_gate_proj), 0);
            enc.set_buffer(2, Some(&moe.scratch_gate), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<GemmParams>() as u64,
                &params_gate_up_sh as *const GemmParams as *const _,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(shared_inter as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );

            // Shared expert up projection GEMV: scratch_up[shared_inter] = W_up * hidden
            let params_up_sh = GemmParams {
                m: 1,
                n: shared_inter_u32,
                k: hidden_u32,
                lda: hidden_u32,
                ldb: hidden_u32,
                ldc: shared_inter_u32,
            };
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_decode);
            enc.set_buffer(0, Some(&self.session.activations.hidden), 0);
            enc.set_buffer(1, Some(&moe.shared_up_proj), 0);
            enc.set_buffer(2, Some(&moe.scratch_up), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<GemmParams>() as u64,
                &params_up_sh as *const GemmParams as *const _,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(shared_inter as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );

            // SiLU-mul in-place on scratch_gate (gate[i] = silu(gate[i]) * up[i]).
            let count_sh = shared_inter as u32;
            enc.set_compute_pipeline_state(&self.engine.pipelines.silu_mul);
            enc.set_buffer(0, Some(&moe.scratch_gate), 0);
            enc.set_buffer(1, Some(&moe.scratch_up), 0);
            enc.set_bytes(2, 4, &count_sh as *const u32 as *const _);
            enc.dispatch_threads(
                MTLSize::new(div_ceil(shared_inter as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );

            // Shared expert down projection GEMV: scratch_expert_out[hidden] = W_down * scratch_gate
            let params_down_sh = GemmParams {
                m: 1,
                n: hidden_u32,
                k: shared_inter_u32,
                lda: shared_inter_u32,
                ldb: shared_inter_u32,
                ldc: hidden_u32,
            };
            enc.set_compute_pipeline_state(&self.engine.pipelines.gemv_decode);
            enc.set_buffer(0, Some(&moe.scratch_gate), 0);
            enc.set_buffer(1, Some(&moe.shared_down_proj), 0);
            enc.set_buffer(2, Some(&moe.scratch_expert_out), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<GemmParams>() as u64,
                &params_down_sh as *const GemmParams as *const _,
            );
            enc.dispatch_thread_groups(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));

            // Accumulate: scratch_out += shared_gate_val * scratch_expert_out.
            enc.set_compute_pipeline_state(&self.engine.pipelines.moe_shared_gate_add);
            enc.set_buffer(0, Some(&moe.scratch_out), 0);
            enc.set_buffer(1, Some(&moe.scratch_expert_out), 0);
            enc.set_bytes(2, 4, &shared_gate_val as *const f32 as *const _);
            enc.set_bytes(3, 4, &hidden_u32 as *const u32 as *const _);
            enc.dispatch_threads(
                MTLSize::new(div_ceil(hidden as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
        };

        // #682 Stage 2: plan every routed expert's cache load NOW —
        // right after routing, before Step 2 is encoded — then spawn
        // the dequant work for cache misses onto background thread(s)
        // and encode Step 2 (above) while it runs, joining + applying
        // results only after Step 2 is fully encoded. Cold-expert
        // mmap reads + CPU dequant then genuinely overlap the
        // shared-expert GEMV encoding instead of blocking in front of
        // it (any already-cached routed experts cost nothing extra
        // either way). This must run before `RoutedExpertStorage::
        // Cached`'s `begin_token()` bookkeeping is otherwise
        // consulted, and Step 3 below relies on every selected expert
        // already being resident (`apply_prefetch_results` already
        // run) by the time its loop runs. `RoutedExpertStorage::Eager`
        // (the safetensors-upload path) has no cache to prefetch into
        // — Step 2 just runs inline with nothing to overlap.
        if let RoutedExpertStorage::Cached { gate_up, down } = &moe.routed {
            gate_up.borrow_mut().begin_token();
            down.borrow_mut().begin_token();

            let expert_ids: Vec<usize> = selected
                .iter()
                .filter(|&&(id, _)| id != usize::MAX)
                .map(|&(id, _)| id)
                .collect();

            #[cfg(test)]
            let parallel = MOE_PREFETCH_PARALLEL_FOR_TEST.with(|c| c.borrow().unwrap_or(true));
            #[cfg(not(test))]
            let parallel = true;

            // `spawn_dequant`'s fault-injection hook is exercised
            // directly against `ExpertSlotCache` (see
            // `moe_prefetch_dequant_panic_recovers_via_readiness_reload`)
            // rather than through this real encode path: a panic that
            // unwinds through a live, not-yet-`endEncoding`'d Metal
            // command encoder is an unrecoverable process abort, not
            // something any amount of `catch_unwind` here could make
            // safe to test.
            let panic_on_expert: Option<usize> = None;

            #[cfg(test)]
            let ordering_gate = MOE_PREFETCH_ORDERING_GATE_FOR_TEST.with(|c| c.borrow_mut().take());

            let gate_up_tasks = gate_up.borrow_mut().plan_prefetch(&expert_ids);
            let down_tasks = down.borrow_mut().plan_prefetch(&expert_ids);

            let gate_up_ref = gate_up.borrow();
            let down_ref = down.borrow();

            let (gate_up_join, down_join) = std::thread::scope(|scope| {
                let gate_up_handle = gate_up_ref.spawn_dequant(
                    gate_up_tasks,
                    parallel,
                    panic_on_expert,
                    #[cfg(test)]
                    ordering_gate,
                    scope,
                );
                let down_handle = down_ref.spawn_dequant(
                    down_tasks,
                    parallel,
                    panic_on_expert,
                    #[cfg(test)]
                    None,
                    scope,
                );

                encode_step2();

                #[cfg(test)]
                MOE_PREFETCH_STEP2_DONE_TX_FOR_TEST.with(|c| {
                    if let Some(tx) = c.borrow_mut().take() {
                        let _ = tx.send(());
                    }
                });

                (
                    gate_up_handle.map(std::thread::ScopedJoinHandle::join),
                    down_handle.map(std::thread::ScopedJoinHandle::join),
                )
            });

            drop(gate_up_ref);
            drop(down_ref);

            // Apply whichever side(s) succeeded BEFORE propagating any
            // panic, so a failure on one cache never discards the
            // other's already-completed dequant work (its slots are
            // marked ready and become immediately usable; only the
            // failed side's slots stay unready, to be reloaded by the
            // next `plan_prefetch` that needs them).
            let mut panic_payload = None;
            match gate_up_join {
                Some(Ok(results)) => gate_up.borrow_mut().apply_prefetch_results(&results),
                Some(Err(e)) => panic_payload = Some(e),
                None => {}
            }
            match down_join {
                Some(Ok(results)) => down.borrow_mut().apply_prefetch_results(&results),
                Some(Err(e)) => {
                    panic_payload.get_or_insert(e);
                }
                None => {}
            }
            if let Some(e) = panic_payload {
                std::panic::resume_unwind(e);
            }
        } else {
            encode_step2();
        }

        // ── Step 3: Routed experts ────────────────────────────────────────────
        let inter_u32 = inter as u32;

        // #682 Stage 2: `RoutedExpertStorage::Cached`'s "touched this
        // token" bookkeeping was already cleared, and every selected
        // expert already prefetched into its slot, right after
        // routing (see above, before Step 2) — this loop now only
        // looks resident slots up (`get_prefetched`), it never
        // triggers a miss/dequant itself.

        for &(expert_id, router_weight) in selected.iter() {
            if expert_id == usize::MAX {
                // Unfilled slot (fewer than top_k experts passed threshold).
                continue;
            }

            // Resolve this expert's gate_up/down buffers plus the
            // element offsets to address gate vs. up within the
            // gate_up buffer. Eager: offsets are global (`expert_id *
            // ...`) into the one giant resident buffer. Cached: the
            // resolved buffer already IS just this expert's slice, so
            // offsets are local (`0` for gate/down, `inter * hidden`
            // for up within the fused per-expert gate_up slot).
            let (gate_up_buf, gate_elem_off, up_elem_off, down_buf, down_elem_off) =
                match &moe.routed {
                    RoutedExpertStorage::Eager { gate_up, down } => {
                        // Expert e gate half starts at element: e * 2 * inter * hidden
                        // Expert e up half starts at element:  e * 2 * inter * hidden + inter * hidden
                        let gate_off = (expert_id * 2 * inter * hidden) as u32;
                        let up_off = (expert_id * 2 * inter * hidden + inter * hidden) as u32;
                        // Expert e down half starts at element: e * hidden * inter
                        let down_off = (expert_id * hidden * inter) as u32;
                        (gate_up.clone(), gate_off, up_off, down.clone(), down_off)
                    }
                    RoutedExpertStorage::Cached { gate_up, down } => {
                        // #682 Stage 2: already prefetched above, right
                        // after routing — this is a pure lookup, no
                        // hit/miss accounting or eviction here.
                        let gate_up_buf = gate_up.borrow().get_prefetched(expert_id).clone();
                        let down_buf = down.borrow().get_prefetched(expert_id).clone();
                        (gate_up_buf, 0u32, (inter * hidden) as u32, down_buf, 0u32)
                    }
                };

            // Gate GEMV: scratch_gate[inter] = W_gate[e] * hidden
            let params_gate = GemmParams {
                m: 1,
                n: inter_u32,
                k: hidden_u32,
                lda: hidden_u32,
                ldb: hidden_u32,
                ldc: inter_u32,
            };
            enc.set_compute_pipeline_state(&self.engine.pipelines.moe_expert_gemv);
            enc.set_buffer(0, Some(&self.session.activations.hidden), 0);
            enc.set_buffer(1, Some(&gate_up_buf), 0);
            enc.set_buffer(2, Some(&moe.scratch_gate), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<GemmParams>() as u64,
                &params_gate as *const GemmParams as *const _,
            );
            enc.set_bytes(4, 4, &gate_elem_off as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));

            // Up GEMV: scratch_up[inter] = W_up[e] * hidden
            let params_up = GemmParams {
                m: 1,
                n: inter_u32,
                k: hidden_u32,
                lda: hidden_u32,
                ldb: hidden_u32,
                ldc: inter_u32,
            };
            enc.set_compute_pipeline_state(&self.engine.pipelines.moe_expert_gemv);
            enc.set_buffer(0, Some(&self.session.activations.hidden), 0);
            enc.set_buffer(1, Some(&gate_up_buf), 0);
            enc.set_buffer(2, Some(&moe.scratch_up), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<GemmParams>() as u64,
                &params_up as *const GemmParams as *const _,
            );
            enc.set_bytes(4, 4, &up_elem_off as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(inter as u64, 1, 1), MTLSize::new(256, 1, 1));

            // SiLU-mul in-place on scratch_gate.
            let count_r = inter as u32;
            enc.set_compute_pipeline_state(&self.engine.pipelines.silu_mul);
            enc.set_buffer(0, Some(&moe.scratch_gate), 0);
            enc.set_buffer(1, Some(&moe.scratch_up), 0);
            enc.set_bytes(2, 4, &count_r as *const u32 as *const _);
            enc.dispatch_threads(
                MTLSize::new(div_ceil(inter as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );

            // Down GEMV: scratch_expert_out[hidden] = W_down[e] * scratch_gate
            let params_down = GemmParams {
                m: 1,
                n: hidden_u32,
                k: inter_u32,
                lda: inter_u32,
                ldb: inter_u32,
                ldc: hidden_u32,
            };
            enc.set_compute_pipeline_state(&self.engine.pipelines.moe_expert_gemv);
            enc.set_buffer(0, Some(&moe.scratch_gate), 0);
            enc.set_buffer(1, Some(&down_buf), 0);
            enc.set_buffer(2, Some(&moe.scratch_expert_out), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<GemmParams>() as u64,
                &params_down as *const GemmParams as *const _,
            );
            enc.set_bytes(4, 4, &down_elem_off as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(hidden as u64, 1, 1), MTLSize::new(256, 1, 1));

            // Scale-and-accumulate: scratch_out += router_weight * scratch_expert_out.
            enc.set_compute_pipeline_state(&self.engine.pipelines.moe_scale_add);
            enc.set_buffer(0, Some(&moe.scratch_out), 0);
            enc.set_buffer(1, Some(&moe.scratch_expert_out), 0);
            enc.set_bytes(2, 4, &router_weight as *const f32 as *const _);
            enc.set_bytes(3, 4, &hidden_u32 as *const u32 as *const _);
            enc.dispatch_threads(
                MTLSize::new(div_ceil(hidden as u64, wg) * wg, 1, 1),
                MTLSize::new(wg, 1, 1),
            );
        }

        // ── Step 4: Copy accumulator → ffn_out ───────────────────────────────
        // `dispatch_copy` uses session.activations buffers; dispatch manually here.
        enc.set_compute_pipeline_state(&self.engine.pipelines.copy);
        enc.set_buffer(0, Some(&moe.scratch_out), 0);
        enc.set_buffer(1, Some(&self.session.activations.ffn_out), 0);
        enc.set_bytes(2, 4, &hidden_u32 as *const u32 as *const _);
        enc.dispatch_threads(
            MTLSize::new(div_ceil(hidden as u64, wg) * wg, 1, 1),
            MTLSize::new(wg, 1, 1),
        );
    }
}
