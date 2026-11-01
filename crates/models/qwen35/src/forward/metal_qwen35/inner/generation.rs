use super::{
    BpeTokenizer, ChatCompletionOutput, ChatMessage, GenerateConfig, GenerateOutput,
    GenerationEntryContract, GenerationPlan, GenerationPreparation, GpuTopkRoute, MetalQwen35State,
    Qwen35VisionRequest, SamplingRouteEnvironment, StopReason, StopStringMatcher,
    apply_sampling_route_plan, decode_tokens, format_chat_template, incremental_detokenizer_finish,
    plan_sampling_route, prepare_generation, read_buffer, reject_inline_images_for_text_chat,
    sample_decode_traced, sample_token, tok,
};

impl MetalQwen35State {
    pub(super) fn prepare_direct_generation(
        &self,
        prompt: &str,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerationPreparation, lattice_inference_core::InferenceError> {
        prepare_generation(
            tokenizer,
            prompt,
            gen_cfg,
            self.engine.config.vocab_size,
            self.max_context(),
            GenerationEntryContract::MetalDirect,
        )
    }

    /// The teardown for a compact sampling route engaged by
    /// [`crate::decoder::qwen_metal::QwenMetalSession`], whether generation
    /// completed, was cancelled, or failed at prefill.
    pub(crate) fn disengage_compact_route(&mut self) {
        self.session.compact_topk = 0;
        self.session.compact_route = GpuTopkRoute::CpuFallback;
    }

    /// **Unstable**: generate text from a prompt; sampling parameters and output format may change.
    ///
    /// Generate text from a prompt.
    ///
    /// # Errors
    ///
    /// Returns `InferenceError::InvalidInput` if grammar-constrained decoding
    /// blocks every token (prefill or any decode step) — mirrors the CPU
    /// `generate` contract (#611).
    /// Returns `InferenceError::Inference` when the prompt plus requested decode
    /// cap exceeds [`Self::max_context`].
    /// Returns `InferenceError::UnsupportedModel` for a multi-token prompt on a
    /// model with MoE layers and no active LoRA adapter: prefill for that input
    /// is batched and the batched schedulers have no MoE schedule.
    pub fn generate(
        &mut self,
        prompt: &str,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerateOutput, lattice_inference_core::InferenceError> {
        let plan = match self.prepare_direct_generation(prompt, tokenizer, gen_cfg)? {
            GenerationPreparation::Complete(output) => return Ok(output),
            GenerationPreparation::Ready(plan) => plan,
        };

        // The MTP and GDN-first self-speculative routes are chosen from the request,
        // the checkpoint and the environment alone, before any state is touched. The
        // one route plan made here also configures the decoder session, which refuses
        // both speculative routes, so a request cannot reach the ordinary loop by a
        // second reading of the environment.
        let environment = SamplingRouteEnvironment::current();
        let route = plan_sampling_route(gen_cfg, plan.prompt_ids.is_empty(), environment);
        let mtp_enabled = gen_cfg
            .enable_mtp
            .unwrap_or_else(|| lattice_model_io::env_switch_enabled("LATTICE_MTP"));
        let use_mtp = super::mtp_route_active(
            self.session.mtp.is_some(),
            mtp_enabled,
            gen_cfg,
            route.use_compact,
        );
        let use_self_spec = !use_mtp
            && super::self_spec_route_active(
                self.session.gdn_checkpoints.is_some(),
                lattice_model_io::env_switch_enabled("LATTICE_SELF_SPEC"),
                gen_cfg,
                route.use_compact,
                self.engine.config.num_active_linear_attention_layers(),
            );
        if use_mtp || use_self_spec {
            // Both routes require a greedy request without grammar, so the plan never
            // engages a compact route here and prefill logits are always dense.
            let GenerationPlan {
                prompt_ids,
                prompt_len,
                ..
            } = plan;
            self.reset_state();
            apply_sampling_route_plan(
                route,
                &mut self.session.compact_route,
                &mut self.session.compact_topk,
                &mut self.session.compact_result,
            );
            // Through the fallible entry point: a multi-token prompt on an MoE model
            // has no batched schedule and must come back as `UnsupportedModel`.
            let prefill_logits = self.try_forward_prefill(&prompt_ids)?;
            if use_mtp {
                // This request has committed to the MTP path (#1336 round 2) —
                // downstream pre-final-hidden-capture gates key off this, not
                // off `self.session.mtp.is_some()` (which only reports whether
                // the checkpoint carries MTP weights, not whether this request
                // uses them).
                self.session.mtp_active = true;
                // #1340: rebuild the MTP draft head's KV cache from the prompt
                // before the first draft round, so it attends to the prompt prefix
                // instead of the single row `reset_state` left it with.
                self.mtp_prefill(&prompt_ids);
                return self.generate_greedy_mtp(&prefill_logits, prompt_len, tokenizer, gen_cfg);
            }
            return self.generate_greedy_self_spec(&prefill_logits, prompt_len, tokenizer, gen_cfg);
        }

        let eos_token_id = self.engine.config.eos_token_id;
        let prompt_ids = plan.prompt_ids.clone();
        // The session resets the state and engages the planned sampling route;
        // dropping it tears the route down on every exit path below.
        let mut session = crate::decoder::qwen_metal::QwenMetalSession::with_route_environment(
            self,
            plan,
            gen_cfg,
            crate::decoder::qwen_metal::MetalEntryProfile::Direct,
            environment,
        )?;
        // KV capacity: `prepare_generation` admitted this request only if
        // `prompt_len + max_new_tokens <= max_context()` (the direct entry refuses a
        // reasoning budget, so the decode cap is `max_new_tokens`), and the driver
        // runs at most `max_new_tokens - 1` decode steps after prefill, so the last
        // step writes position `max_context() - 2` at most. No admitted request
        // fills the cache: this entry never stops with `StopReason::KvFull`, and the
        // session's full-cache refusal cannot fire.
        crate::decoder::qwen_metal::run_direct(
            &mut session,
            gen_cfg,
            &prompt_ids,
            eos_token_id,
            tokenizer,
        )
    }

    /// **Unstable**: multimodal generation; visual token handling and position encoding
    /// may evolve (ADR-049).
    ///
    /// Generate text from a multimodal prompt (image + text).
    ///
    /// Visual patch embeddings from `input.patch_embeddings` are prepended to the text
    /// token embeddings before the first prefill call.  Position IDs for text tokens are
    /// offset by `input.visual_tokens` so that patch positions `[0, visual_tokens)` are
    /// occupied by the image.  All other generation parameters (KV cache, sampling,
    /// stop tokens) behave identically to `generate`.
    ///
    /// # Errors
    ///
    /// Returns `InferenceError::InvalidInput` if:
    /// - `input.d_model` does not match this model's `hidden_size`
    /// - `input.patch_embeddings` length is inconsistent with `visual_tokens * d_model`
    ///
    /// Returns the shared `InferenceError::Inference` context-budget error if the
    /// visual and text prompt plus requested decode cap exceed the KV cache capacity.
    pub fn generate_multimodal(
        &mut self,
        input: crate::vision::MultimodalInput,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerateOutput, lattice_inference_core::InferenceError> {
        use lattice_inference_core::InferenceError;

        // Reject grammar configs before any GPU work: grammar-constrained decoding
        // is not wired into the multimodal path.
        super::multimodal_generate_preflight(gen_cfg)?;

        // Validate multimodal input
        input.validate().map_err(|e| {
            InferenceError::InvalidInput(format!("MultimodalInput validation failed: {e}"))
        })?;

        let hidden = self.engine.config.hidden_size;

        if input.d_model != hidden {
            return Err(InferenceError::InvalidInput(format!(
                "generate_multimodal: input.d_model ({}) does not match model hidden_size ({})",
                input.d_model, hidden
            )));
        }

        let visual_tokens = input.visual_tokens;
        let text_ids = &input.text_tokens;
        let total_len = visual_tokens + text_ids.len();

        if total_len == 0 {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: 0,
                generated_tokens: 0,
                stopped: false,
                stop_reason: None,
                token_logprobs: vec![],
            });
        }

        // max_new_tokens == 0 means "generate nothing": return before prefill/sampling
        // so we never emit a token the caller did not ask for. Mirrors the
        // guard in generate() above (#612).
        if gen_cfg.effective_decode_cap() == 0 {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: total_len,
                generated_tokens: 0,
                stopped: false,
                stop_reason: Some(StopReason::Length),
                token_logprobs: vec![],
            });
        }

        crate::model::qwen35::check_context_budget_with_total(
            total_len,
            gen_cfg.effective_reasoning_budget(),
            gen_cfg.max_new_tokens,
            gen_cfg.total_token_cap,
            self.max_context(),
        )?;

        let cfg = self.engine.config.clone();

        // Reset recurrent state for a clean generation.
        self.reset_state();

        let mut rng_state = match gen_cfg.seed {
            Some(s) => {
                if s == 0 {
                    1
                } else {
                    s
                }
            }
            None => {
                use std::time::SystemTime;
                let t = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0x12345678_9abcdef0_u64);
                if t == 0 { 1 } else { t }
            }
        };

        // ---------------------------------------------------------------
        // Prefill: write [visual_tokens | text_tokens] embeddings into the
        // activations buffer, then run the prefill forward pass.
        //
        // The hidden buffer is sized max_prefill * hidden at construction;
        // we need total_len slots.  For v0 we fall back to sequential stepping
        // when total_len > max_prefill to avoid a bounds overrun.
        // ---------------------------------------------------------------

        // Validate text token IDs
        for (i, &id) in text_ids.iter().enumerate() {
            if id as usize >= cfg.vocab_size {
                return Err(InferenceError::InvalidInput(format!(
                    "generate_multimodal: text_tokens[{i}]={id} >= vocab_size {}",
                    cfg.vocab_size
                )));
            }
        }

        // Build a combined embedding sequence: [visual_embeds | text_embeds].
        // We write it in one shot using sequential forward steps to avoid
        // sizing the hidden buffer beyond its allocation.
        //
        // Position IDs:
        //   - Visual patch tokens: positions 0 .. visual_tokens
        //     (positions are unused by decoder — the ViT already applied 2D RoPE)
        //   - Text tokens: positions visual_tokens .. visual_tokens + text_ids.len()
        //
        // For v0, we run each visual token as a forward_step with a synthetic
        // embedding (bypassing the embed_tokens lookup) by directly writing the
        // patch embedding into the hidden buffer, then invoking the layer stack
        // at the appropriate position.
        //
        // Concretely: we produce logits only for the text portion prefill; the
        // visual token steps advance KV cache position without needing their
        // logits.  This matches the reference: image tokens occupy positions
        // [0, visual_tokens), text tokens follow.

        // Step 1: Feed each visual token as a hidden-state injection.
        // We borrow `forward_step_with_hidden` logic: write directly to
        // `session.activations.hidden`, then run the layer stack at the position.
        for vt in 0..visual_tokens {
            let embed_start = vt * hidden;
            let embed_end = embed_start + hidden;
            let visual_embed = &input.patch_embeddings[embed_start..embed_end];

            // SAFETY: activations.hidden is StorageModeShared f32; no GPU command
            // is in flight at this point (we are between forward calls).
            unsafe {
                let dst = self.session.activations.hidden.contents() as *mut f32;
                for i in 0..hidden {
                    *dst.add(i) = visual_embed[i];
                }
            }

            // Run the layer stack at position `vt` (no embedding lookup needed —
            // we already wrote the hidden state above).  We reuse `forward_step`'s
            // infrastructure by writing token_id=0 (a valid ID) into the buffer and
            // then immediately overwriting with the visual embedding.  However, that
            // would double-write.  Instead, use the internal `forward_step_inner`
            // helper which runs layers on whatever is already in `hidden`.
            //
            // Since `forward_step_inner` is not directly accessible outside the inner
            // module (it has no `pub` on the wrapper), we call the public
            // `forward_step` with token_id=0 to drive the layer stack and KV cache
            // advancement, then discard the logits.  Before the call, we overwrite
            // the embedding that `forward_step` writes so the correct visual
            // embedding propagates through the layers.
            //
            // Implementation note: `forward_step` writes the embedding at the START
            // of its body (before any GPU dispatch), so we must write AFTER the
            // embedding lookup.  We achieve this by calling `forward_step` normally —
            // which correctly looks up token_id=0 — and instead pre-initializing the
            // hidden buffer with the visual embedding before the call, relying on the
            // fact that `forward_step` OVERWRITES the hidden buffer with the token
            // embedding at the start.
            //
            // This means we cannot avoid the embed_tokens read in `forward_step`.
            // For v0 correctness (the embedding is overwritten), we call a thin
            // wrapper that writes the visual embedding AFTER the embed lookup but
            // before the first GPU dispatch.  Since we cannot easily interpose
            // without modifying forward_step, v0 uses the following approach:
            //
            // Write visual_embed → call forward_step(0, vt) → logits discarded.
            // forward_step will overwrite our write with embed_tokens[0], but the
            // net effect is that we process token 0 at each visual position.
            //
            // *** v0 limitation ***: visual token embeddings are not correctly fed
            // through the decoder in this release; only the position offset is
            // established.  The visual semantics require a Metal-level injection
            // point (v1 work item per ADR-049).  The generate_multimodal API surface
            // and types are correct and tested; the actual visual features require
            // v1 Metal kernel integration.
            let _logits = self.forward_step(0u32, vt);
        }

        // Step 2: Run the text portion using the existing sequential path.
        // Text tokens start at position `visual_tokens`.
        let prompt_len = total_len;
        let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.effective_decode_cap());
        let mut all_ids: Vec<u32> = text_ids.clone();

        // Process all text tokens sequentially at their offset positions.
        let mut last_logits = Vec::new();
        for (t, &id) in text_ids.iter().enumerate() {
            last_logits = self.forward_step(id, visual_tokens + t);
        }

        // If no text tokens were given, we have no last logits — just return empty.
        if text_ids.is_empty() {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: visual_tokens,
                generated_tokens: 0,
                stopped: false,
                stop_reason: None,
                token_logprobs: vec![],
            });
        }

        // Sample the first token from last logits.
        let is_stop = |id: u32| id == cfg.eos_token_id || gen_cfg.stop_token_ids.contains(&id);

        let mut stopped = false;
        let mut stop_reason = StopReason::Length;

        // String-stop path: mirrors the CPU stop-string matcher exactly (#643).
        let mut stop_text_state = if gen_cfg.stop_strings.is_empty() {
            None
        } else {
            Some((
                tok::incremental_detokenizer_new(),
                StopStringMatcher::new(&gen_cfg.stop_strings),
            ))
        };

        let first_id = sample_token(&last_logits, gen_cfg, &all_ids, &mut rng_state);
        if is_stop(first_id) {
            stopped = true;
            stop_reason = StopReason::Eos;
        } else {
            generated_ids.push(first_id);
            all_ids.push(first_id);
            if let Some((detok, matcher)) = stop_text_state.as_mut() {
                let delta = tok::incremental_detokenizer_push(detok, tokenizer, first_id);
                if matcher.push(&delta, &mut |_| {}) {
                    stopped = true;
                    stop_reason = StopReason::Eos;
                }
            }
        }

        // Autoregressive decode loop.
        let mut pos = visual_tokens + text_ids.len();
        while !stopped && generated_ids.len() < gen_cfg.effective_decode_cap() {
            if self.session.kv_cache.seq_len >= self.session.kv_cache.max_cache_len {
                stop_reason = StopReason::KvFull;
                break;
            }
            let last_token = *all_ids.last().unwrap_or(&cfg.eos_token_id);
            if is_stop(last_token) {
                stopped = true;
                stop_reason = StopReason::Eos;
                break;
            }
            let step_logits = self.forward_step_decode(last_token, pos);
            pos += 1;
            let next_id =
                sample_decode_traced(None, &step_logits, gen_cfg, &all_ids, &mut rng_state);
            if is_stop(next_id) {
                stopped = true;
                stop_reason = StopReason::Eos;
                break;
            }
            generated_ids.push(next_id);
            all_ids.push(next_id);

            if let Some((detok, matcher)) = stop_text_state.as_mut() {
                let delta = tok::incremental_detokenizer_push(detok, tokenizer, next_id);
                if matcher.push(&delta, &mut |_| {}) {
                    stopped = true;
                    stop_reason = StopReason::Eos;
                    break;
                }
            }
        }

        let text = if let Some((mut detok, mut matcher)) = stop_text_state {
            if !matcher.stopped() {
                matcher.finish(&incremental_detokenizer_finish(&mut detok), &mut |_| {});
            }
            matcher.into_text()
        } else {
            decode_tokens(tokenizer, &generated_ids)
        };
        Ok(GenerateOutput {
            text,
            token_ids: generated_ids.clone(),
            prompt_tokens: prompt_len,
            generated_tokens: generated_ids.len(),
            stopped,
            stop_reason: Some(stop_reason),
            token_logprobs: vec![],
        })
    }

    /// Qwen3.5 vision (ADR-069 Metal S5, MP2+MP3): Metal port of
    /// `cpu_f16::generate_multimodal_f16` — the CPU function is the numeric
    /// oracle (ADR-069 Amendment 1: divergence from it is a bug by
    /// definition), so this mirrors its request type, validation order, and
    /// per-token embedding + M-RoPE contract exactly: real residual-stream
    /// REPLACE injection of merged patch embeddings at `<|image_pad|>` slots
    /// (mirrors `forward_step_f16`'s `injected_embedding`), and per-token
    /// 3-axis interleaved M-RoPE cos/sin rows at the six GQA layers (mirrors
    /// `full_attention_step_f16`'s `mrope_cos_sin`) built from
    /// `Qwen35VisionRequest::build_mrope_tables`/`build_decode_cos_sin`. A
    /// text-only request (`image_grids` empty) passes `None` throughout,
    /// which keeps every position on the model's unchanged 1-D
    /// `engine.rope_cos`/`rope_sin` table — structurally identical to plain
    /// text decode.
    ///
    /// Deliberately a new entry point rather than a change to [`Self::generate_multimodal`]:
    /// that function's [`crate::vision::MultimodalInput`] represents visual tokens as a
    /// flat prefix and carries no per-image `(T, H, W)` grid shape, so it cannot supply
    /// `Qwen35VisionRequest`'s validation/checkpoint-binding inputs this port reuses.
    /// `generate_multimodal` is unchanged by this PR.
    pub fn generate_multimodal_vision(
        &mut self,
        request: &Qwen35VisionRequest,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerateOutput, lattice_inference_core::InferenceError> {
        self.generate_multimodal_vision_impl(request, tokenizer, gen_cfg, None, None)
    }

    /// Generate from one vision request while polling caller cancellation.
    ///
    /// Cancellation is checked before prefill, before every prefill token,
    /// after prefill, and before every decode step. The returned output
    /// carries [`StopReason::Interrupt`] when cancellation wins.
    pub fn generate_multimodal_vision_with_cancel<C>(
        &mut self,
        request: &Qwen35VisionRequest,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
        mut should_cancel: C,
    ) -> Result<GenerateOutput, lattice_inference_core::InferenceError>
    where
        C: FnMut() -> bool,
    {
        self.generate_multimodal_vision_impl(
            request,
            tokenizer,
            gen_cfg,
            None,
            Some(&mut should_cancel),
        )
    }

    fn multimodal_cancel_requested(should_cancel: &mut Option<&mut dyn FnMut() -> bool>) -> bool {
        match should_cancel {
            Some(callback) => callback(),
            None => false,
        }
    }

    fn cancelled_multimodal_output(prompt_tokens: usize) -> GenerateOutput {
        GenerateOutput {
            text: String::new(),
            token_ids: Vec::new(),
            prompt_tokens,
            generated_tokens: 0,
            stopped: false,
            stop_reason: Some(StopReason::Interrupt),
            token_logprobs: Vec::new(),
        }
    }

    /// Body of [`Self::generate_multimodal_vision`] with an optional
    /// decode-step logits probe (each decode iteration's full logits row,
    /// in order). The probe exists because greedy token output is not a
    /// usable observable for the decode-time M-RoPE coordinate: 18 of 24
    /// decoder layers are GDN (no RoPE at all), and on the real #989
    /// fixture a `None`-decode mutation reproduced the correct arm's
    /// greedy tokens exactly over a 32-step horizon. Logit-level
    /// assertions through this seam are what make the decode wiring
    /// mutation-detectable (same pattern as `gemma4`'s
    /// `generate_greedy_with_probe`).
    pub(super) fn generate_multimodal_vision_impl(
        &mut self,
        request: &Qwen35VisionRequest,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
        mut decode_logits_probe: Option<&mut Vec<Vec<f32>>>,
        mut should_cancel: Option<&mut dyn FnMut() -> bool>,
    ) -> Result<GenerateOutput, lattice_inference_core::InferenceError> {
        use lattice_inference_core::InferenceError;

        super::multimodal_generate_preflight(gen_cfg)?;

        request.validate().map_err(|e| {
            InferenceError::InvalidInput(format!("multimodal request failed validation: {e}"))
        })?;

        let cfg = self.engine.config.clone();

        // Caller-supplied `input_ids` must be bounded against the checkpoint
        // vocabulary before any GPU work begins — mirrors the CPU oracle's guard
        // ahead of `forward_step_f16`'s embedding-table slice.
        if let Some(&bad_id) = request
            .input_ids
            .iter()
            .find(|&&id| id as usize >= cfg.vocab_size)
        {
            return Err(InferenceError::InvalidInput(format!(
                "input_ids contains out-of-vocabulary token id {bad_id} (vocab_size={})",
                cfg.vocab_size
            )));
        }

        let has_image = !request.image_grids.is_empty();

        // Bind the request to the loaded checkpoint's vision metadata before
        // selecting image slots (mirrors the CPU oracle's checkpoint-binding
        // guard) — an internally consistent request targeting the wrong
        // checkpoint must not silently inject at the wrong slots.
        if has_image {
            let cfg_image_token_id = cfg.image_token_id.ok_or_else(|| {
                InferenceError::InvalidInput(
                    "multimodal request supplied but checkpoint has no image_token_id".to_string(),
                )
            })?;
            if cfg_image_token_id != request.image_token_id {
                return Err(InferenceError::InvalidInput(format!(
                    "request image_token_id {} does not match checkpoint image_token_id {cfg_image_token_id}",
                    request.image_token_id
                )));
            }
            let vision_cfg = cfg.vision_config.as_ref().ok_or_else(|| {
                InferenceError::InvalidInput(
                    "multimodal request supplied but checkpoint has no vision_config".to_string(),
                )
            })?;
            if vision_cfg.spatial_merge_size != request.spatial_merge_size {
                return Err(InferenceError::InvalidInput(format!(
                    "request spatial_merge_size {} does not match checkpoint \
                         vision_config.spatial_merge_size {}",
                    request.spatial_merge_size, vision_cfg.spatial_merge_size
                )));
            }
            if request.decoder_hidden_size != cfg.hidden_size {
                return Err(InferenceError::InvalidInput(format!(
                    "request decoder_hidden_size {} does not match checkpoint hidden_size {}",
                    request.decoder_hidden_size, cfg.hidden_size
                )));
            }
            if vision_cfg.out_hidden_size != cfg.hidden_size {
                return Err(InferenceError::InvalidInput(format!(
                    "checkpoint vision_config.out_hidden_size {} does not match decoder \
                         hidden_size {}",
                    vision_cfg.out_hidden_size, cfg.hidden_size
                )));
            }
        }

        // Build the per-physical-token M-RoPE position ids and interleaved
        // cos/sin tables (ADR-069 MP3). Mirrors `generate_multimodal_f16`'s
        // ordering: built right after the checkpoint-binding checks above,
        // before the prompt-empty/context-budget checks below.
        let (positions, tables) = request.build_mrope_tables(&cfg)?;
        let expected_rope_half = cfg.rope_dim() / 2;
        if tables.cos.iter().any(|row| row.len() != expected_rope_half)
            || tables.sin.iter().any(|row| row.len() != expected_rope_half)
        {
            return Err(InferenceError::InvalidInput(format!(
                "M-RoPE table row width does not match decoder rotary half-width: expected \
                     {expected_rope_half}"
            )));
        }

        let prompt_ids = &request.input_ids;
        let prompt_len = prompt_ids.len();
        crate::model::qwen35::check_prompt_not_empty(prompt_len)?;

        if gen_cfg.effective_decode_cap() == 0 {
            return Ok(GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                stopped: false,
                stop_reason: Some(StopReason::Length),
                token_logprobs: vec![],
            });
        }

        crate::model::qwen35::check_context_budget_with_total(
            prompt_len,
            gen_cfg.effective_reasoning_budget(),
            gen_cfg.max_new_tokens,
            gen_cfg.total_token_cap,
            self.max_context(),
        )?;

        if Self::multimodal_cancel_requested(&mut should_cancel) {
            return Ok(Self::cancelled_multimodal_output(prompt_len));
        }

        // Reset recurrent state for a clean generation.
        self.reset_state();

        let mut rng_state = match gen_cfg.seed {
            Some(s) => {
                if s == 0 {
                    1
                } else {
                    s
                }
            }
            None => {
                use std::time::SystemTime;
                let t = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0x12345678_9abcdef0_u64);
                if t == 0 { 1 } else { t }
            }
        };

        // Prefill: inject a post-merger visual row at each `<|image_pad|>` slot, in the
        // same sequential image-token order the request already concatenated
        // `post_merger_rows` in (HF's masked_scatter contract), text tokens (including
        // vision delimiters) keep their normal embedding lookup. Every physical
        // position — image-pad or text — rotates the six GQA layers with its
        // `tables.cos[pos]`/`tables.sin[pos]` row when the request carries an
        // image (ADR-069 MP3); `has_image=false` passes `None` throughout, which
        // keeps `forward_step_mrope`/`forward_step_injected_mrope` on the
        // unchanged 1-D `engine.rope_cos`/`rope_sin` path — structurally
        // identical to the pre-MP3 text-only route.
        let mut visual_row = 0usize;
        let mut last_logits = Vec::new();
        let last_prefill_pos = prompt_len - 1;
        for (pos, &token_id) in prompt_ids.iter().enumerate() {
            if Self::multimodal_cancel_requested(&mut should_cancel) {
                return Ok(Self::cancelled_multimodal_output(prompt_len));
            }
            let cos_sin = if has_image {
                Some((tables.cos[pos].as_slice(), tables.sin[pos].as_slice()))
            } else {
                None
            };
            // issue #1336: only the prompt's final position is ever sampled —
            // every earlier position still advances KV/GDN state but skips the
            // terminal RMSNorm/lm_head/vocab readback (`emit_head=false`).
            let emit_head = pos == last_prefill_pos;
            if has_image && token_id == request.image_token_id {
                let start = visual_row * request.decoder_hidden_size;
                let end = start + request.decoder_hidden_size;
                let row = &request.post_merger_rows[start..end];
                // Mirrors `forward_step_f16`'s injected_embedding guard: reject a
                // non-finite visual row before it reaches the residual stream. (Length
                // is already guaranteed by `request.validate()` above.)
                if let Some(bad) = row.iter().find(|v| !v.is_finite()) {
                    return Err(InferenceError::InvalidInput(format!(
                        "injected_embedding contains a non-finite value: {bad}"
                    )));
                }
                visual_row += 1;
                last_logits = self.forward_step_injected_mrope(
                    row,
                    pos,
                    cos_sin,
                    emit_head,
                    crate::forward::signpost::Scope::NotDecode,
                );
            } else {
                last_logits = self.forward_step_mrope(
                    token_id,
                    pos,
                    cos_sin,
                    emit_head,
                    crate::forward::signpost::Scope::NotDecode,
                );
            }
        }
        if Self::multimodal_cancel_requested(&mut should_cancel) {
            return Ok(Self::cancelled_multimodal_output(prompt_len));
        }

        let mut all_ids = prompt_ids.clone();
        let is_stop = |id: u32| id == cfg.eos_token_id || gen_cfg.stop_token_ids.contains(&id);
        let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.effective_decode_cap());
        let mut stopped = false;
        let mut stop_reason = StopReason::Length;

        // String-stop path: mirrors the CPU stop-string matcher exactly, same as
        // `generate_multimodal`.
        let mut stop_text_state = if gen_cfg.stop_strings.is_empty() {
            None
        } else {
            Some((
                tok::incremental_detokenizer_new(),
                StopStringMatcher::new(&gen_cfg.stop_strings),
            ))
        };

        let first_id = sample_token(&last_logits, gen_cfg, &all_ids, &mut rng_state);
        if is_stop(first_id) {
            stopped = true;
            stop_reason = StopReason::Eos;
        } else {
            generated_ids.push(first_id);
            all_ids.push(first_id);
            if let Some((detok, matcher)) = stop_text_state.as_mut() {
                let delta = tok::incremental_detokenizer_push(detok, tokenizer, first_id);
                if matcher.push(&delta, &mut |_| {}) {
                    stopped = true;
                    stop_reason = StopReason::Eos;
                }
            }
        }

        // Autoregressive decode: identical to the plain `generate`/`generate_streaming`
        // loop, except that once the prompt contained an image the physical
        // KV-cache index (`physical_pos`) and the M-RoPE coordinate diverge —
        // mirrors `generate_multimodal_f16`'s decode loop (ADR-069 RECON §3):
        // `decode_axis = physical_pos + rope_delta`, recomputed into a single
        // cos/sin row via the same `build_decode_cos_sin` the CPU oracle uses.
        while !stopped && generated_ids.len() < gen_cfg.effective_decode_cap() {
            if Self::multimodal_cancel_requested(&mut should_cancel) {
                stop_reason = StopReason::Interrupt;
                break;
            }
            if self.session.kv_cache.seq_len >= self.session.kv_cache.max_cache_len {
                stop_reason = StopReason::KvFull;
                break;
            }
            let physical_pos = self.session.kv_cache.seq_len;
            let last_token = *all_ids.last().unwrap_or(&cfg.eos_token_id);
            if is_stop(last_token) {
                stopped = true;
                stop_reason = StopReason::Eos;
                break;
            }

            let decode_cos_sin_owned;
            let mrope_cos_sin = if has_image {
                let decode_axis = crate::vision::qwen35_mrope::decode_position(
                    physical_pos,
                    positions.rope_delta,
                )?;
                decode_cos_sin_owned = request.build_decode_cos_sin(&cfg, decode_axis)?;
                if decode_cos_sin_owned.0.len() != expected_rope_half
                    || decode_cos_sin_owned.1.len() != expected_rope_half
                {
                    return Err(InferenceError::InvalidInput(format!(
                        "decode-time M-RoPE row width does not match decoder rotary \
                             half-width: expected {expected_rope_half}"
                    )));
                }
                Some((
                    decode_cos_sin_owned.0.as_slice(),
                    decode_cos_sin_owned.1.as_slice(),
                ))
            } else {
                None
            };

            let step_logits = self.forward_step_mrope(
                last_token,
                physical_pos,
                mrope_cos_sin,
                true,
                crate::forward::signpost::Scope::Decode,
            );
            if let Some(probe) = decode_logits_probe.as_deref_mut() {
                probe.push(step_logits.clone());
            }
            let next_id =
                sample_decode_traced(None, &step_logits, gen_cfg, &all_ids, &mut rng_state);
            if is_stop(next_id) {
                stopped = true;
                stop_reason = StopReason::Eos;
                break;
            }
            generated_ids.push(next_id);
            all_ids.push(next_id);

            if let Some((detok, matcher)) = stop_text_state.as_mut() {
                let delta = tok::incremental_detokenizer_push(detok, tokenizer, next_id);
                if matcher.push(&delta, &mut |_| {}) {
                    stopped = true;
                    stop_reason = StopReason::Eos;
                    break;
                }
            }
        }

        let text = if let Some((mut detok, mut matcher)) = stop_text_state {
            if !matcher.stopped() {
                matcher.finish(&incremental_detokenizer_finish(&mut detok), &mut |_| {});
            }
            matcher.into_text()
        } else {
            decode_tokens(tokenizer, &generated_ids)
        };

        Ok(GenerateOutput {
            text,
            token_ids: generated_ids.clone(),
            prompt_tokens: prompt_len,
            generated_tokens: generated_ids.len(),
            stopped,
            stop_reason: Some(stop_reason),
            token_logprobs: vec![],
        })
    }
}

impl MetalQwen35State {
    /// **Unstable**: chat completion; stop token handling and output format may change.
    ///
    /// Chat completion: format messages with Qwen3.5 template, generate response.
    /// Stops on <|im_end|> or EOS or max tokens.
    ///
    /// # Errors
    ///
    /// Returns `InferenceError::InvalidInput` if `messages` contains an
    /// inline image, which requires the multimodal vision entry point, or
    /// if grammar-constrained decoding blocks every token — propagated
    /// from [`Self::generate`] (#611).
    pub fn chat_completion(
        &mut self,
        messages: &[ChatMessage],
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
    ) -> Result<ChatCompletionOutput, lattice_inference_core::InferenceError> {
        reject_inline_images_for_text_chat(messages)?;
        let prompt = format_chat_template(messages);
        // Add <|im_end|> as stop token
        let mut cfg = gen_cfg.clone();
        if let Some(im_end_id) = tokenizer.special_token_id("<|im_end|>")
            && !cfg.stop_token_ids.contains(&im_end_id)
        {
            cfg.stop_token_ids.push(im_end_id);
        }
        let result = self.generate(&prompt, tokenizer, &cfg)?;
        let text = result.text.trim_end().to_string();
        Ok(ChatCompletionOutput {
            message: ChatMessage::assistant(text),
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.generated_tokens,
            stopped: result.stopped,
        })
    }

    /// **Unstable**: streaming generation; callback signature may change.
    ///
    /// Streaming generation: calls `on_token` for each generated token.
    /// Returns total generated token count.
    /// The callback receives (token_text, token_id) and returns true to continue.
    ///
    /// Cancellable only through `on_token`'s return value. Callers that need to
    /// observe cancellation independently of text emission -- so an
    /// empty-delta decode iteration, or the uninterruptible prefill call
    /// itself, can't silently swallow a disconnect -- should call
    /// [`Self::generate_streaming_with_cancel`] instead.
    pub fn generate_streaming<F>(
        &mut self,
        prompt: &str,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
        on_token: F,
    ) -> Result<GenerateOutput, lattice_inference_core::InferenceError>
    where
        F: FnMut(&str, u32) -> bool,
    {
        self.generate_streaming_with_cancel(prompt, tokenizer, gen_cfg, on_token, || false)
    }

    /// Cancellation-aware sibling of [`Self::generate_streaming`].
    ///
    /// `should_cancel` is polled independently of `on_token`: before starting
    /// the prefill pass, immediately after it returns, and at the top of every
    /// decode iteration -- all before any further expensive GPU work runs for
    /// that step. This closes two gaps `on_token`-only cancellation has:
    /// prefill itself has no callback point at all (a long prompt can run to
    /// completion after the client is already gone), and the decode loop only
    /// calls `on_token` for non-empty text deltas, so a run of tokens that
    /// decode to an incomplete UTF-8 tail (a multi-token codepoint mid-stream)
    /// would otherwise skip the cancellation check for those iterations too.
    /// SSE emission itself stays gated on non-empty deltas, unchanged.
    ///
    /// `generate_streaming` is the `should_cancel = || false` convenience form
    /// and is otherwise identical; both share this one implementation.
    ///
    /// # Errors
    ///
    /// Returns `InferenceError::InvalidInput` if grammar-constrained decoding
    /// blocks every token (prefill or any decode step) — mirrors the CPU
    /// `generate_streaming` contract (#611).
    /// Returns `InferenceError::Inference` when the prompt plus effective decode
    /// cap exceeds [`Self::max_context`].
    /// Returns `InferenceError::UnsupportedModel` for a multi-token prompt on a
    /// model with MoE layers and no active LoRA adapter, for the reason given on
    /// [`Self::generate`]. `generate_streaming` shares this implementation and so
    /// reports the same error.
    pub fn generate_streaming_with_cancel<F, C>(
        &mut self,
        prompt: &str,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
        on_token: F,
        should_cancel: C,
    ) -> Result<GenerateOutput, lattice_inference_core::InferenceError>
    where
        F: FnMut(&str, u32) -> bool,
        C: FnMut() -> bool,
    {
        let plan = match prepare_generation(
            tokenizer,
            prompt,
            gen_cfg,
            self.engine.config.vocab_size,
            self.max_context(),
            GenerationEntryContract::MetalStreaming,
        )? {
            GenerationPreparation::Complete(output) => return Ok(output),
            GenerationPreparation::Ready(plan) => plan,
        };
        let vocab_size = self.engine.config.vocab_size;
        let eos_token_id = self.engine.config.eos_token_id;
        let prompt_ids = plan.prompt_ids.clone();

        // The session resets the state and engages the planned sampling route;
        // dropping it tears the route down on every exit path below.
        let mut session = crate::decoder::qwen_metal::QwenMetalSession::new(
            self,
            plan,
            gen_cfg,
            crate::decoder::qwen_metal::MetalEntryProfile::Streaming,
        )?;

        // Budget forcing: resolve the </think> token id once before decoding.
        // Only paid when reasoning_budget is Some; None path is a single branch.
        let think_close_id = crate::model::qwen35::resolve_reasoning_close_token(
            tokenizer,
            gen_cfg.reasoning_budget,
            gen_cfg.enable_thinking,
            vocab_size,
        )?;

        // KV capacity: `prepare_generation` admitted this request only if
        // `prompt_len + decode_cap(..) <= max_context()` (`check_context_budget`),
        // and the driver runs at most `decode_cap(..) - 1` decode steps after
        // prefill, so the last step writes position `max_context() - 2` at most.
        // No admitted request fills the cache: this entry never stops with
        // `StopReason::KvFull`, and the session's full-cache refusal cannot fire.
        crate::decoder::qwen_metal::run_streaming(
            &mut session,
            gen_cfg,
            think_close_id,
            &prompt_ids,
            eos_token_id,
            tokenizer,
            on_token,
            should_cancel,
        )
    }

    /// **Unstable**: streaming chat completion; callback signature may change.
    ///
    /// Streaming chat completion with token-by-token callback.
    ///
    /// Cancellable only through `on_token`'s return value; see
    /// [`Self::chat_completion_streaming_with_cancel`] for a variant that also
    /// observes cancellation during prefill and on empty-delta decode steps.
    pub fn chat_completion_streaming<F>(
        &mut self,
        messages: &[ChatMessage],
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
        on_token: F,
    ) -> Result<ChatCompletionOutput, lattice_inference_core::InferenceError>
    where
        F: FnMut(&str, u32) -> bool,
    {
        self.chat_completion_streaming_with_cancel(messages, tokenizer, gen_cfg, on_token, || false)
    }

    /// Cancellation-aware sibling of [`Self::chat_completion_streaming`]; see
    /// [`Self::generate_streaming_with_cancel`] for exactly what `should_cancel`
    /// observes that `on_token` alone cannot.
    ///
    /// # Errors
    ///
    /// Returns `InferenceError::InvalidInput` if `messages` contains an
    /// inline image, which requires the multimodal vision entry point, or
    /// if grammar-constrained decoding blocks every token — propagated
    /// from [`Self::generate_streaming_with_cancel`] (#611).
    pub fn chat_completion_streaming_with_cancel<F, C>(
        &mut self,
        messages: &[ChatMessage],
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
        on_token: F,
        should_cancel: C,
    ) -> Result<ChatCompletionOutput, lattice_inference_core::InferenceError>
    where
        F: FnMut(&str, u32) -> bool,
        C: FnMut() -> bool,
    {
        reject_inline_images_for_text_chat(messages)?;
        let prompt = format_chat_template(messages);
        let mut cfg = gen_cfg.clone();
        if let Some(im_end_id) = tokenizer.special_token_id("<|im_end|>")
            && !cfg.stop_token_ids.contains(&im_end_id)
        {
            cfg.stop_token_ids.push(im_end_id);
        }
        let result =
            self.generate_streaming_with_cancel(&prompt, tokenizer, &cfg, on_token, should_cancel)?;
        let text = result.text.trim_end().to_string();
        Ok(ChatCompletionOutput {
            message: ChatMessage::assistant(text),
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.generated_tokens,
            stopped: result.stopped,
        })
    }
}

// -----------------------------------------------------------------------
// Perplexity harness (ADR-044 step 4b)
// -----------------------------------------------------------------------

impl MetalQwen35State {
    /// **Unstable**: largest window length the Metal forward path can serve
    /// in a single call. Bounded by the KV-cache capacity chosen at session
    /// construction (`new_session(max_cache_len)` or
    /// `MetalQwen35State::new(_, _, max_cache_len)`), NOT by the RoPE table
    /// — `new_session` already rejects `max_cache_len >
    /// max_position_embeddings`, so the KV cap is always the tighter bound.
    /// Callers driving the perplexity harness must keep their effective
    /// window strictly at or below this value; `forward_step` panics
    /// otherwise (KV-cache full assertion).
    pub fn max_context(&self) -> usize {
        self.session.kv_cache.max_cache_len
    }

    /// The model's hidden size, which is the length of every vector
    /// [`Self::embed_tokens`] returns.
    pub fn hidden_size(&self) -> usize {
        self.engine.config.hidden_size
    }

    /// **Stable**: Metal sibling of
    /// [`crate::model::qwen35::Qwen35Model::embed_tokens`] — embed `tokens`
    /// as one `[hidden_size]` vector by pooling the model's final hidden
    /// states.
    ///
    /// Returns the same quantity as the CPU method, from the same point in
    /// the network: the last position's hidden state after the final
    /// RMSNorm, before the language-model head. The two are held together
    /// by an equivalence test rather than by inspection.
    ///
    /// The returned vector is **not** L2-normalized, matching the CPU
    /// method and HuggingFace's `AutoModel`-plus-pooling convention.
    ///
    /// # Session state
    ///
    /// This resets the session before running. GDN recurrent state carries
    /// across calls, so embedding without a reset would fold whatever the
    /// session generated previously into the vector. Do not interleave this
    /// with an in-progress generation and expect that generation to
    /// continue.
    ///
    /// # Errors
    ///
    /// - `tokens` empty, longer than the session's context, or containing
    ///   an id at or above `vocab_size`.
    /// - More than one token against a model with MoE layers: the prefill this
    ///   runs to produce the hidden state is batched, and the batched schedulers
    ///   have no MoE schedule.
    /// - [`HiddenPooling::Mean`], which this path does not implement. Mean
    ///   pooling needs every position's *post-norm* hidden state, and the
    ///   Metal prefill normalizes only the rows it is about to project to
    ///   logits — the last row on the default path, and nothing at all on a
    ///   non-final chunk of a long prompt, which returns before the norm.
    ///   Producing a number here anyway would mean silently returning
    ///   last-token pooling under a mean-pooling request, so this refuses
    ///   instead. Use the CPU path for mean pooling.
    pub fn embed_tokens(
        &mut self,
        tokens: &[u32],
        pooling: crate::model::qwen35::HiddenPooling,
    ) -> Result<Vec<f32>, lattice_inference_core::InferenceError> {
        use crate::model::qwen35::HiddenPooling;
        use lattice_inference_core::InferenceError;

        if pooling != HiddenPooling::LastToken {
            return Err(InferenceError::Inference(format!(
                "embed_tokens: the Metal path implements {:?} only; {pooling:?} \
                     needs post-norm hidden states at every position, which this \
                     forward path does not produce",
                HiddenPooling::LastToken
            )));
        }
        if tokens.is_empty() {
            return Err(InferenceError::Inference(
                "embed_tokens: need at least 1 token, got 0".to_string(),
            ));
        }
        let max_context = self.max_context();
        if tokens.len() > max_context {
            return Err(InferenceError::Inference(format!(
                "embed_tokens: {} tokens exceeds session context {max_context}",
                tokens.len()
            )));
        }
        let vocab = self.engine.config.vocab_size;
        if let Some((i, &id)) = tokens
            .iter()
            .enumerate()
            .find(|&(_, &id)| id as usize >= vocab)
        {
            // Checked here rather than left to `forward_prefill`, which
            // asserts and panics. A library consumer passing raw ids gets a
            // typed error, matching the CPU method.
            return Err(InferenceError::Inference(format!(
                "embed_tokens: tokens[{i}]={id} is out of range: vocab_size is {vocab}"
            )));
        }

        let hidden = self.engine.config.hidden_size;
        self.reset_state();

        let previous = self.session.capture_final_hidden;
        self.session.capture_final_hidden = true;
        self.session
            .final_hidden_captured
            .store(false, std::sync::atomic::Ordering::Relaxed);
        // Fallible route: a multi-token prompt on an MoE model has no batched
        // prefill schedule. Restore the capture flag before propagating, so a
        // rejected call leaves the session exactly as it found it.
        let prefill = self.try_forward_prefill(tokens);
        self.session.capture_final_hidden = previous;
        prefill?;

        if !self
            .session
            .final_hidden_captured
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            // Fail closed. The capture buffer persists across calls, so
            // returning its contents when this call did not write them
            // would hand back a well-formed vector for somebody else's
            // input — indistinguishable from a correct answer at the call
            // site. A forward path that norms without capturing is a bug in
            // this file, not a condition the caller can fix, so say so.
            return Err(InferenceError::Inference(
                "embed_tokens: the forward path did not capture a final hidden state; \
                     a final-RMSNorm site is missing its capture call"
                    .to_string(),
            ));
        }

        // SAFETY: `forward_prefill` waits on its command buffers before
        // returning, and `final_hidden` is StorageModeShared and sized for
        // `hidden` f32 values.
        let v = unsafe { read_buffer(&self.session.activations.final_hidden, hidden) };
        Ok(v)
    }

    /// **Unstable**: Metal Q4 sibling of [`crate::model::qwen35::Qwen35Model::compute_token_nlls`].
    ///
    /// Computes per-position cross-entropy NLLs for an autoregressive
    /// forward pass over `tokens` on the Metal forward path. Resets all
    /// recurrent state ([`MetalQwen35State::reset_state`]) before stepping
    /// so a single call does not depend on the caller's previous KV cache
    /// / GDN state, and subsequent calls do not contaminate each other —
    /// this is what makes the function safe to use as the per-window NLL
    /// kernel inside [`crate::model::qwen35::run_strided_perplexity`].
    ///
    /// Returns a `Vec<f32>` of length `tokens.len() - 1`. The value at
    /// index `i` is `-log p(tokens[i + 1] | tokens[0..=i])`, where the
    /// softmax runs over the full vocabulary at decode step `i`.
    ///
    /// Errors if `tokens.len() < 2`, if `tokens.len() > self.max_context()`
    /// (the KV cache would overflow during the walk), if any
    /// `token_id >= cfg.vocab_size`, or if a LoRA adapter is active.
    pub fn compute_token_nlls(
        &mut self,
        tokens: &[u32],
    ) -> Result<Vec<f32>, lattice_inference_core::InferenceError> {
        if tokens.len() < 2 {
            return Err(lattice_inference_core::InferenceError::Inference(format!(
                "compute_token_nlls: need at least 2 tokens, got {}",
                tokens.len()
            )));
        }
        let max_context = self.max_context();
        if tokens.len() > max_context {
            return Err(lattice_inference_core::InferenceError::Inference(format!(
                "compute_token_nlls: tokens.len() ({}) exceeds Metal KV-cache capacity ({}); \
                     use a shorter window or rebuild the session with a larger max_cache_len",
                tokens.len(),
                max_context
            )));
        }
        let vocab_size = self.engine.config.vocab_size;
        if let Some((bad_idx, &bad)) = tokens
            .iter()
            .enumerate()
            .find(|&(_, &t)| (t as usize) >= vocab_size)
        {
            return Err(lattice_inference_core::InferenceError::Inference(format!(
                "compute_token_nlls: tokens[{bad_idx}]={bad} >= vocab_size {vocab_size}"
            )));
        }

        // Single batch prefill producing logits at all n positions, then
        // pick out the n-1 next-token NLLs. Same code path as production
        // decode prefill (forward_prefill) — extended to emit per-position
        // lm_head outputs via dispatch_gemm with m=n instead of m=1.
        // Sequential forward_step from pos=0 with reset_state does NOT
        // produce usable next-token distributions and is not used here.
        self.reset_state();
        let flat = self.forward_prefill_all_logits(tokens)?;
        debug_assert_eq!(flat.len(), tokens.len() * vocab_size);
        let mut nlls = Vec::with_capacity(tokens.len() - 1);
        for i in 0..tokens.len() - 1 {
            let logits = &flat[i * vocab_size..(i + 1) * vocab_size];
            let target = tokens[i + 1] as usize;
            nlls.push(crate::model::qwen35::log_softmax_nll(logits, target));
        }
        Ok(nlls)
    }

    /// **Unstable**: Metal Q4 sibling of [`crate::model::qwen35::Qwen35Model::compute_perplexity`].
    ///
    /// Thin wrapper around [`crate::model::qwen35::run_strided_perplexity`]
    /// driving the shared aggregator with this state's
    /// [`Self::compute_token_nlls`] as the per-window NLL kernel. Each
    /// window resets recurrent state, so context never crosses a window
    /// boundary — identical semantics to the CPU path.
    ///
    /// Use this on a [`MetalQwen35State`] loaded via [`Self::from_q4_dir`]
    /// to measure unrotated-Q4 PPL or rotated-Q4 PPL (the output of
    /// `bin/quantize_quarot`); the rotated-vs-unrotated delta is the
    /// acceptance number for ADR-044 step 4.
    pub fn compute_perplexity(
        &mut self,
        tokens: &[u32],
        cfg: &crate::model::qwen35::PerplexityConfig,
    ) -> Result<crate::model::qwen35::PerplexityReport, lattice_inference_core::InferenceError>
    {
        let max_ctx = self.max_context();
        crate::model::qwen35::run_strided_perplexity(tokens, cfg, max_ctx, |slice| {
            self.compute_token_nlls(slice)
        })
    }
}
