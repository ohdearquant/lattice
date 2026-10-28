use super::*;
use lattice_tokenizer::Tokenizer as _;

// -----------------------------------------------------------------------
// Cross-turn KV prefix cache: Metal-specific runtime (#462)
// -----------------------------------------------------------------------

impl MetalQwen35State {
    /// Cache-aware sibling of [`Self::generate_streaming`].
    ///
    /// Detects the longest shared token prefix between `slot_id`'s
    /// cached entry and this prompt; when it is an exact, fully
    /// represented append-only prefix (`PrefixReuseMode::ExactAppend`),
    /// restores the GDN snapshot, sets the logical position, and
    /// prefills only the divergent suffix via `forward_prefill_from`.
    /// Otherwise falls back to `reset_state` + full prefill — byte-for-
    /// byte the same path `generate_streaming` takes, so the decode loop
    /// below is identical in both cases and produces the same generated
    /// token IDs as the non-cached path for the same conversation
    /// (token-identity invariant; see design.md).
    ///
    /// `generate_streaming` and `chat_completion_streaming` are left
    /// untouched and never use this cache, so their behavior stays
    /// deterministic and unaffected by any cache state.
    ///
    /// MoE suffixes and full prefills use per-token forward steps, with or
    /// without a LoRA adapter.
    ///
    /// # Errors
    ///
    /// Returns the shared context-budget error before cache restore or eviction
    /// when the prompt plus effective decode cap exceeds [`Self::max_context`].
    pub fn generate_streaming_with_prefix_cache<F>(
        &mut self,
        slot_id: lattice_compute::kv_cache::CrossTurnSlotId,
        prompt: &str,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
        on_token: F,
    ) -> Result<CachedGenerateOutput, lattice_inference_core::InferenceError>
    where
        F: FnMut(&str, u32) -> bool,
    {
        self.generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            prompt,
            tokenizer,
            gen_cfg,
            on_token,
            || false,
        )
    }

    /// Cancellation-aware sibling of [`Self::generate_streaming_with_prefix_cache`];
    /// see [`Self::generate_streaming_with_cancel`] for exactly what
    /// `should_cancel` observes that `on_token` alone cannot. Fail-closed
    /// invariant: every path that can observe a cancellation runs strictly
    /// after `restore_cross_turn_prefix` has already `take`n the slot's
    /// entry (for `ExactAppend`) or `reset_state` has cleared the whole
    /// cache (for `FullRefill`), so an early cancel-return never re-saves
    /// and the slot is left empty rather than holding a stale/partial
    /// entry — the same guarantee the existing error path documents
    /// (#516).
    pub fn generate_streaming_with_prefix_cache_and_cancel<F, C>(
        &mut self,
        slot_id: lattice_compute::kv_cache::CrossTurnSlotId,
        prompt: &str,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
        on_token: F,
        should_cancel: C,
    ) -> Result<CachedGenerateOutput, lattice_inference_core::InferenceError>
    where
        F: FnMut(&str, u32) -> bool,
        C: FnMut() -> bool,
    {
        self.generate_streaming_with_prefix_cache_with_trace(
            slot_id,
            prompt,
            tokenizer,
            gen_cfg,
            on_token,
            should_cancel,
        )
        .map(|(output, _trace)| output)
    }

    /// The prefix-cache entry's one implementation: the output
    /// [`Self::generate_streaming_with_prefix_cache_and_cancel`] returns, with
    /// the driver trace of the request that produced it. The trace is all
    /// zeroes when the request returned before reaching the driver.
    pub(crate) fn generate_streaming_with_prefix_cache_with_trace<F, C>(
        &mut self,
        slot_id: lattice_compute::kv_cache::CrossTurnSlotId,
        prompt: &str,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
        on_token: F,
        should_cancel: C,
    ) -> Result<
        (CachedGenerateOutput, crate::decoder::driver::DriverTrace),
        lattice_inference_core::InferenceError,
    >
    where
        F: FnMut(&str, u32) -> bool,
        C: FnMut() -> bool,
    {
        // #856/#922/#827/#1354: prompt tokenization, the empty-prompt
        // guard, and the prompt-plus-decode-budget context bound are
        // delegated to the same `prepare_generation` shared preparation
        // `generate_streaming_with_cancel` already routes through,
        // closing the last unmigrated generation entry point (issue
        // #827). This uses `GenerationEntryContract::MetalPrefixCacheStreaming`
        // (#1354), not the plain-streaming `MetalStreaming` variant this
        // call used to reuse: `MetalStreaming`'s permissive
        // `validate_capabilities` exists because plain streaming
        // supports both `logprobs` and `enable_mtp`, but this path does
        // not, so folding those checks into that variant would have
        // wrongly restricted plain streaming too. `MetalPrefixCacheStreaming`
        // carries its own `check_logprobs_not_set` /
        // `check_mtp_not_requested` guards in `validate_before_tokenization`
        // (run ahead of tokenization inside `prepare_generation`, same
        // position these two guards ran in as an explicit preflight
        // here before this change), so the error text and precedence
        // for a request that violates more than one guard at once are
        // unchanged: this call must still return here, before any
        // cache/session state mutation and before `_inner`'s
        // destructive-on-`Err` recovery arm is reachable, so a rejected
        // call never evicts a valid pre-existing cross-turn entry it
        // never touched (PR #787).
        let prompt_prep = crate::model::qwen35::prepare_generation(
            tokenizer,
            prompt,
            gen_cfg,
            self.engine.config.vocab_size,
            self.max_context(),
            crate::model::qwen35::GenerationEntryContract::MetalPrefixCacheStreaming,
        )?;
        let gen_plan = match prompt_prep {
            crate::model::qwen35::GenerationPreparation::Complete(output) => {
                // Zero-budget requests return before any state mutation,
                // leaving an existing cache entry exactly as-is.
                return Ok((
                    CachedGenerateOutput {
                        cache: CrossTurnCacheStats {
                            slot_id,
                            prompt_tokens: output.prompt_tokens,
                            reused_tokens: 0,
                            prefetched_tokens: 0,
                            mode: lattice_compute::kv_cache::PrefixReuseMode::FullRefill,
                        },
                        output,
                    },
                    crate::decoder::driver::DriverTrace::default(),
                ));
            }
            crate::model::qwen35::GenerationPreparation::Ready(plan) => plan,
        };

        // #835: validate the suffix a reuse plan would select for this
        // slot BEFORE calling `_inner` at all -- same reasoning as the
        // `check_logprobs_not_set`/`check_mtp_not_requested` guards
        // `prepare_generation` already ran above, ahead of tokenization
        // (PR #787, #1354): the `match` below unconditionally calls
        // `reset_state()` and evicts the cache slot on ANY `Err` from
        // `_inner`, so a suffix-validation-only rejection (out-of-vocab
        // token, empty suffix, or a suffix overflowing `max_cache_len`)
        // must return here, before `_inner` runs `restore_cross_turn_prefix`
        // (or `_inner`'s own destructive-on-`Err` recovery is even
        // reachable), or a valid pre-existing cross-turn entry this call
        // never touched would be destroyed alongside it.
        //
        // `prepare_generation`'s zero-budget short-circuit above already
        // returned for `max_new_tokens == 0`, so `max_new_tokens > 0` is
        // guaranteed from here on.
        let metadata = self.cross_turn_metadata(tokenizer);
        let plan = self.plan_cross_turn_reuse(slot_id, &metadata, &gen_plan.prompt_ids);
        self.plan_prefix_request(&gen_plan.prompt_ids, &plan)?;

        // Budget forcing: validate the </think> token id here too,
        // before `_inner` is even called -- same reasoning as the
        // suffix-plan preflight immediately above (#835) and the
        // `check_logprobs_not_set`/`check_mtp_not_requested` preflights
        // above that (PR #787). `_inner` resolves this again (defense
        // in depth, mirroring the suffix-plan double-check), but only
        // THIS early return -- before the destructive `match` below
        // runs `restore_cross_turn_prefix` or `reset_state()` --
        // guarantees a missing/out-of-range </think> token never
        // destroys a valid pre-existing cross-turn entry the request
        // never touched.
        //
        // Gated on `max_new_tokens > 0`, same as the suffix-plan
        // preflight above: `_inner`'s own zero-budget guard (before its
        // `resolve_reasoning_close_token` call) is
        // unreachable-before-return for a zero-token request, so an
        // unconditional resolve here would disagree with every other
        // generation path (plain CPU `generate()`, non-cached-prefix
        // Metal `generate_streaming()`) -- both short-circuit
        // `max_new_tokens == 0` to a zero-token `StopReason::Length`
        // BEFORE resolving the close marker, so a `reasoning_budget`
        // set alongside `max_new_tokens == 0` succeeds there even with
        // a missing/invalid `</think>` marker.
        if gen_cfg.max_new_tokens > 0 {
            crate::model::qwen35::resolve_reasoning_close_token(
                tokenizer,
                gen_cfg.reasoning_budget,
                gen_cfg.enable_thinking,
                self.engine.config.vocab_size,
            )?;
        }

        match self.generate_streaming_with_prefix_cache_and_cancel_inner(
            slot_id,
            gen_plan,
            tokenizer,
            gen_cfg,
            on_token,
            should_cancel,
        ) {
            Ok(out) => Ok(out),
            Err(e) => {
                // Fail-closed: state and cache may disagree after a
                // restore/prefill/save error, so drop both rather than
                // risk a later turn reusing an inconsistent boundary.
                self.reset_state();
                self.cross_turn_prefix_cache.remove(slot_id);
                Err(e)
            }
        }
    }

    /// Restores the plan's reusable boundary (or resets the state for a full
    /// refill), runs the suffix through the shared decoder driver over a
    /// [`crate::decoder::qwen_metal::QwenMetalSession`], then commits the
    /// cross-turn slot as the request's exit dictates (see
    /// `crate::decoder::qwen_metal::run_prefix_cache_streaming`).
    ///
    /// The preflights the public entry ran (suffix content, `</think>` id) run
    /// again here, before any state or slot is touched, so this function's own
    /// ordering invariant does not rest on its one caller.
    fn generate_streaming_with_prefix_cache_and_cancel_inner<F, C>(
        &mut self,
        slot_id: lattice_compute::kv_cache::CrossTurnSlotId,
        gen_plan: crate::model::qwen35::GenerationPlan,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
        on_token: F,
        should_cancel: C,
    ) -> Result<
        (CachedGenerateOutput, crate::decoder::driver::DriverTrace),
        lattice_inference_core::InferenceError,
    >
    where
        F: FnMut(&str, u32) -> bool,
        C: FnMut() -> bool,
    {
        use crate::decoder::qwen_metal::{
            PrefixCommit, QwenMetalSession, run_prefix_cache_streaming,
        };
        use lattice_compute::kv_cache::PrefixReuseMode;

        let prompt_ids = gen_plan.prompt_ids.clone();
        let prompt_len = prompt_ids.len();
        debug_assert!(
            prompt_len > 0,
            "the wrapper's check_prompt_not_empty (#856) must reject an \
                 empty prompt before calling _inner"
        );
        debug_assert!(
            gen_cfg.max_new_tokens > 0,
            "the wrapper's prepare_generation zero-budget short-circuit \
                 (#827) must return before calling _inner"
        );

        let metadata = self.cross_turn_metadata(tokenizer);
        let plan = self.plan_cross_turn_reuse(slot_id, &metadata, &prompt_ids);
        self.plan_prefix_request(&prompt_ids, &plan)?;

        let think_close_id = crate::model::qwen35::resolve_reasoning_close_token(
            tokenizer,
            gen_cfg.reasoning_budget,
            gen_cfg.enable_thinking,
            self.engine.config.vocab_size,
        )?;

        // The consumed entry (ExactAppend / ReplayFromCheckpoint) is
        // carried to the end-of-generation save so its checkpoint ring
        // and boundary snapshot survive across turns (#590).
        let mut consumed_entry = match plan.mode {
            PrefixReuseMode::ExactAppend | PrefixReuseMode::ReplayFromCheckpoint { .. } => {
                self.restore_cross_turn_prefix(slot_id, &plan)?
            }
            // FullRefill, and any mode this build does not know (the enum is
            // non-exhaustive in lattice-compute), resets and prefills.
            _ => {
                self.reset_state();
                None
            }
        };

        // Plain history omits the generation prompt's think frame, so the
        // shared prefix can end just after this turn's assistant opener.
        let mut opener = String::new();
        crate::forward::metal_qwen35::push_chat_generation_open(&mut opener);
        let opener = tokenizer.tokenize(&opener);
        let turn_checkpoint_at =
            generation_turn_start(&prompt_ids, &opener.input_ids[..opener.real_length]);

        // The session borrows the state for the whole run and tears the
        // sampling route down when it drops, before the commit below.
        let eos_token_id = self.engine.config.eos_token_id;
        let (run, turn_checkpoint) = {
            let mut session = QwenMetalSession::over_restored_state(
                self,
                gen_plan,
                plan.suffix_start,
                gen_cfg,
                SamplingRouteEnvironment::current(),
            )?
            .with_turn_checkpoint_at(turn_checkpoint_at);
            let run = run_prefix_cache_streaming(
                &mut session,
                gen_cfg,
                think_close_id,
                &prompt_ids,
                eos_token_id,
                tokenizer,
                on_token,
                should_cancel,
            )?;
            (run, session.take_turn_checkpoint())
        };

        match run.commit {
            PrefixCommit::Leave => self.cross_turn_prefix_cache.remove(slot_id),
            PrefixCommit::Save { silent_step } => {
                // The silent step forwards the last pushed token so the next
                // turn can reuse through the full assistant output. It
                // samples and emits nothing. A full KV cache leaves the
                // boundary one token short, which stays consistent: KV, GDN
                // and `represented_len` all agree there.
                if silent_step && let Some(&last_pushed_id) = run.output.token_ids.last() {
                    let seq_len = self.session.kv_cache.seq_len;
                    if seq_len < self.session.kv_cache.max_cache_len {
                        let _ = self.forward_step_decode(last_pushed_id, seq_len);
                    }
                }
                let represented_len = self.session.kv_cache.seq_len;
                debug_assert!(represented_len >= prompt_len);
                let mut represented_token_ids = prompt_ids;
                represented_token_ids
                    .extend_from_slice(&run.output.token_ids[..represented_len - prompt_len]);
                self.save_cross_turn_prefix_or_clear(
                    slot_id,
                    metadata,
                    represented_token_ids,
                    consumed_entry.take(),
                    turn_checkpoint.map(|(len, snapshot)| MetalGdnCheckpoint { len, snapshot }),
                );
            }
        }

        Ok((
            CachedGenerateOutput {
                output: run.output,
                cache: CrossTurnCacheStats {
                    slot_id,
                    prompt_tokens: prompt_len,
                    reused_tokens: if run.prefill_ran {
                        plan.reusable_len
                    } else {
                        0
                    },
                    prefetched_tokens: if run.prefill_ran { plan.suffix_len } else { 0 },
                    mode: plan.mode,
                },
            },
            run.trace,
        ))
    }

    /// Cache-aware sibling of [`Self::chat_completion_streaming`]: same
    /// ChatML formatting and `<|im_end|>` stop-token handling, routed
    /// through [`Self::generate_streaming_with_prefix_cache`] instead of
    /// `generate_streaming`.
    pub fn chat_completion_streaming_with_prefix_cache<F>(
        &mut self,
        slot_id: lattice_compute::kv_cache::CrossTurnSlotId,
        messages: &[ChatMessage],
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
        on_token: F,
    ) -> Result<CachedChatCompletionOutput, lattice_inference_core::InferenceError>
    where
        F: FnMut(&str, u32) -> bool,
    {
        self.chat_completion_streaming_with_prefix_cache_and_cancel(
            slot_id,
            messages,
            tokenizer,
            gen_cfg,
            on_token,
            || false,
        )
    }

    /// Cancellation-aware sibling of
    /// [`Self::chat_completion_streaming_with_prefix_cache`]: same ChatML
    /// formatting and `<|im_end|>` stop-token handling, routed through
    /// [`Self::generate_streaming_with_prefix_cache_and_cancel`] instead
    /// of `generate_streaming_with_prefix_cache`. See
    /// [`Self::chat_completion_streaming_with_cancel`] for the analogous
    /// non-cache entry point this mirrors.
    ///
    /// # Errors
    ///
    /// Returns `InferenceError::InvalidInput` if `messages` contains an
    /// inline image, which requires the multimodal vision entry point, or
    /// if the delegated generation path rejects the input.
    pub fn chat_completion_streaming_with_prefix_cache_and_cancel<F, C>(
        &mut self,
        slot_id: lattice_compute::kv_cache::CrossTurnSlotId,
        messages: &[ChatMessage],
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
        on_token: F,
        should_cancel: C,
    ) -> Result<CachedChatCompletionOutput, lattice_inference_core::InferenceError>
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
        let cached = self.generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            &prompt,
            tokenizer,
            &cfg,
            on_token,
            should_cancel,
        )?;
        let text = cached.output.text.trim_end().to_string();
        Ok(CachedChatCompletionOutput {
            output: ChatCompletionOutput {
                message: ChatMessage::assistant(text),
                prompt_tokens: cached.output.prompt_tokens,
                completion_tokens: cached.output.generated_tokens,
                stopped: cached.output.stopped,
            },
            cache: cached.cache,
        })
    }
}

// The next request renders prior assistant content immediately after the opener.
fn generation_turn_start(prompt_ids: &[u32], opener_ids: &[u32]) -> Option<usize> {
    if opener_ids.is_empty() || opener_ids.len() > prompt_ids.len() {
        return None;
    }
    prompt_ids
        .windows(opener_ids.len())
        .rposition(|window| window == opener_ids)
        .map(|start| start + opener_ids.len())
}

#[cfg(test)]
mod tests {
    use super::generation_turn_start;
    use lattice_compute::kv_cache::{
        AdapterId, CrossTurnPrefixEntry, CrossTurnPrefixMetadata, CrossTurnSlotId, KvPrefixHandle,
        PrefixReuseMode, plan_prefix_reuse_for_slot,
    };

    fn metadata() -> CrossTurnPrefixMetadata {
        CrossTurnPrefixMetadata {
            model_fingerprint: 1,
            tokenizer_fingerprint: 2,
            adapter_id: AdapterId::BASE,
            vocab_size: 1000,
            max_cache_len: 4096,
            kv_f16: false,
            rope_theta_bits: 0,
            partial_rotary_factor_bits: None,
            layer_pattern_hash: 42,
            chat_template_version: 1,
        }
    }

    fn entry(
        token_ids: Vec<u32>,
        represented_len: usize,
        gdn_snapshot_len: usize,
    ) -> CrossTurnPrefixEntry {
        CrossTurnPrefixEntry {
            slot_id: CrossTurnSlotId::DEFAULT,
            metadata: metadata(),
            token_ids,
            represented_len,
            kv: KvPrefixHandle {
                represented_len,
                num_full_attention_layers: 6,
                kv_dim: 512,
                max_cache_len: 4096,
                kv_f16: false,
            },
            gdn_snapshot_len,
        }
    }

    #[test]
    fn generation_turn_start_ends_at_the_last_opener() {
        let opener = [4, 5];
        assert_eq!(generation_turn_start(&[1, 4, 5, 6, 7], &opener), Some(3));
        assert_eq!(
            generation_turn_start(&[1, 4, 5, 8, 9, 4, 5, 6], &opener),
            Some(7)
        );
        assert_eq!(generation_turn_start(&[1, 2, 4, 5], &opener), Some(4));
        assert_eq!(generation_turn_start(&[1, 4, 6, 5], &opener), None);
        assert_eq!(generation_turn_start(&[4], &opener), None);
        assert_eq!(generation_turn_start(&[1, 4, 5], &[]), None);
    }

    #[test]
    fn a_checkpoint_at_the_turn_start_replays_a_plain_history_continuation() {
        let opener = [4, 5];
        let think = [6, 7];
        let first_prompt: Vec<u32> = [&[1, 2, 3][..], &opener, &think].concat();
        let turn_start = generation_turn_start(&first_prompt, &opener).expect("opener present");
        assert_eq!(turn_start, 5);
        let answer = [8, 9];
        let held: Vec<u32> = [&first_prompt[..], &answer].concat();
        let e = entry(held.clone(), held.len(), held.len());
        let next_prompt: Vec<u32> = [
            &[1, 2, 3][..],
            &opener,
            &answer,
            &[10, 1, 2, 11],
            &opener,
            &think,
        ]
        .concat();

        let plan = plan_prefix_reuse_for_slot(
            CrossTurnSlotId::DEFAULT,
            Some(&e),
            &metadata(),
            &next_prompt,
            &[turn_start],
        );
        assert_eq!(
            plan.mode,
            PrefixReuseMode::ReplayFromCheckpoint {
                checkpoint_len: turn_start
            }
        );
        assert_eq!(plan.reusable_len, turn_start);
        assert_eq!(plan.suffix_len, next_prompt.len() - turn_start);

        let plan = plan_prefix_reuse_for_slot(
            CrossTurnSlotId::DEFAULT,
            Some(&e),
            &metadata(),
            &next_prompt,
            &[],
        );
        assert_eq!(plan.mode, PrefixReuseMode::FullRefill);
    }
}
