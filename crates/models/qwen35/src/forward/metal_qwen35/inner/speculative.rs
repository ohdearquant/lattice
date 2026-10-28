use super::{
    GdnStateTrafficScope, MetalMtpDecodeMetrics, MetalQwen35State, MtpHiddenTap,
    SELF_SPEC_MAX_DRAFT, SelfSpecMetrics, SpeculativeMetrics, SpeculativeRoute,
    SpeculativeRouteRecord, decode_tokens,
};
use crate::generation::{GenerateConfig, GenerateOutput};
use crate::stop_reason::StopReason;
use lattice_tokenizer::bpe::BpeTokenizer;

impl MetalQwen35State {
    /// MTP greedy decode (LATTICE_MTP=1, greedy only).
    ///
    /// Each round drafts one extra token via MTP, verifies two tokens with the target
    /// model, accepts the draft if the target agrees, and rolls back and uses the
    /// target's token otherwise.
    ///
    /// The default (sequential) verifier runs through the shared speculative driver,
    /// which applies the shared decode policy to the verified, committed tokens only.
    /// The batch-GEMM verifier (`LATTICE_MTP_BATCH` on, no adapter loaded) is a legacy
    /// experimental route that keeps its own loop: it is selected exactly as before,
    /// never falls back to the sequential verifier, and records
    /// [`SpeculativeRoute::MtpBatchGemmLegacy`] so a request on it is distinguishable
    /// from shared-route coverage.
    pub(super) fn generate_greedy_mtp(
        &mut self,
        prefill_logits: &[f32],
        prompt_len: usize,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerateOutput, lattice_inference_core::InferenceError> {
        let use_batch = super::use_batch_gemm_verifier(
            lattice_model_io::env_switch_enabled("LATTICE_MTP_BATCH"),
            self.lora.is_some(),
        );
        if use_batch {
            return Ok(self.generate_greedy_mtp_batch_gemm_legacy(
                prefill_logits,
                prompt_len,
                tokenizer,
                gen_cfg,
            ));
        }
        self.generate_speculative_shared(
            SpeculativeMetrics::mtp(),
            prefill_logits,
            prompt_len,
            tokenizer,
            gen_cfg,
        )
    }

    /// GDN-first self-speculative decode.
    ///
    /// Drafts K tokens via GDN-only forwards (cheap O(n) recurrence, no KV cache
    /// writes), then verifies with the full model (GDN+GQA) in batch. Accepts the
    /// longest agreeing prefix and rolls back GDN state on partial rejection. The
    /// committed tokens go through the shared speculative driver.
    ///
    /// Activated by `LATTICE_SELF_SPEC=1`. Requires greedy decode (temperature <= 0,
    /// top_k <= 1). Requires `gdn_checkpoints` to be allocated in the session.
    pub(super) fn generate_greedy_self_spec(
        &mut self,
        prefill_logits: &[f32],
        prompt_len: usize,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerateOutput, lattice_inference_core::InferenceError> {
        self.generate_speculative_shared(
            SpeculativeMetrics::self_spec(),
            prefill_logits,
            prompt_len,
            tokenizer,
            gen_cfg,
        )
    }

    /// Runs a request on a shared speculative route: the decoder session hands the
    /// driver verified rounds and the driver applies the decode policy to them.
    pub(super) fn generate_speculative_shared(
        &mut self,
        metrics: SpeculativeMetrics,
        prefill_logits: &[f32],
        prompt_len: usize,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
    ) -> Result<GenerateOutput, lattice_inference_core::InferenceError> {
        let eos_token_id = self.engine.config.eos_token_id;
        let mut session = crate::decoder::qwen_metal::QwenMetalSpeculativeSession::new(
            self,
            metrics,
            prefill_logits,
        );
        crate::decoder::qwen_metal::run_speculative_direct(
            &mut session,
            gen_cfg,
            prompt_len,
            eos_token_id,
            tokenizer,
        )
    }

    /// Records which speculative route the current request took and, under the
    /// route's verbose switch, prints the route marker line.
    pub(crate) fn record_speculative_route(
        &mut self,
        route: SpeculativeRoute,
        trace: crate::decoder::SpeculativeTrace,
    ) {
        self.session.speculative_route = Some(SpeculativeRouteRecord { route, trace });
        let (switch, tag) = match route {
            SpeculativeRoute::SelfSpecShared => ("LATTICE_SELF_SPEC_VERBOSE", "SELF_SPEC"),
            SpeculativeRoute::MtpShared | SpeculativeRoute::MtpBatchGemmLegacy => {
                ("LATTICE_MTP_VERBOSE", "MTP")
            }
        };
        if lattice_model_io::env_switch_enabled(switch) {
            eprintln!(
                "[{tag}] route={} shared_policy={} driver_rounds={} offered={}",
                route.label(),
                route.uses_shared_policy(),
                trace.rounds,
                trace.offered,
            );
        }
    }

    /// One verification round of a shared speculative route. `pending` is the token
    /// the target already chose for the next position; `room` is how many more tokens
    /// the request may commit. The returned round holds only tokens the target model
    /// evaluated and agreed with, plus the target's prediction that follows them.
    pub(crate) fn speculative_round(
        &mut self,
        metrics: &mut SpeculativeMetrics,
        pending: u32,
        room: usize,
        is_stop: &dyn Fn(u32) -> bool,
    ) -> crate::decoder::VerifiedRound {
        match metrics {
            SpeculativeMetrics::Mtp(m) => self.mtp_sequential_round(m, pending, room, is_stop),
            SpeculativeMetrics::SelfSpec(m) => self.self_spec_round(m, pending, room, is_stop),
        }
    }

    /// What the loops this driver replaced did once their last round ended: the MTP
    /// route refreshes `last_pre_final_hidden` and both routes print their verbose
    /// counters.
    pub(crate) fn finish_speculative(&mut self, metrics: &SpeculativeMetrics) {
        match metrics {
            SpeculativeMetrics::Mtp(metrics) => {
                self.session.last_pre_final_hidden = {
                    let pos = self.session.kv_cache.seq_len;
                    if pos > 0 {
                        // Re-run one step to get fresh hidden for any downstream caller.
                        // (verify_tokens_batched already left last_pre_final_hidden current.)
                        self.session.last_pre_final_hidden.clone()
                    } else {
                        self.session.last_hidden_cursor = None;
                        vec![0.0f32; self.engine.config.hidden_size]
                    }
                };

                if lattice_model_io::env_switch_enabled("LATTICE_MTP_VERBOSE") {
                    eprintln!(
                        "[MTP] rounds={} mtp_fwd={} verify={} accepted_extra={} rollbacks={} fallbacks={} mtp_ms={:.1} verify_ms={:.1} rb_ms={:.1} tap={}",
                        metrics.rounds,
                        metrics.mtp_forwards,
                        metrics.verify_calls,
                        metrics.accepted_extra_tokens,
                        metrics.verify_calls - metrics.accepted_extra_tokens,
                        metrics.fallback_tokens,
                        metrics.mtp_ms,
                        metrics.verify_ms,
                        metrics.rollback_ms,
                        self.mtp_hidden_tap().map_or("none", MtpHiddenTap::label),
                    );
                }
            }
            SpeculativeMetrics::SelfSpec(metrics) => {
                if lattice_model_io::env_switch_enabled("LATTICE_SELF_SPEC_VERBOSE") {
                    eprintln!(
                        "[SELF_SPEC] rounds={} draft_fwd={} verify={} accepted_extra={} fallbacks={} draft_ms={:.1} verify_ms={:.1} rb_ms={:.1}",
                        metrics.rounds,
                        metrics.draft_forwards,
                        metrics.verify_calls,
                        metrics.accepted_extra_tokens,
                        metrics.fallback_tokens,
                        metrics.draft_ms,
                        metrics.verify_ms,
                        metrics.rollback_ms,
                    );
                }
            }
        }
    }

    /// One MTP round under the sequential verifier.
    ///
    /// Drafts one token for the position after `pending_token`, verifies
    /// `[pending_token, draft]` with the target, and returns the tokens the target
    /// committed. A rejected draft is rolled back here and never appears in the
    /// returned round. The stop-token contract is `mtp_greedy_round`'s: a stop token
    /// is excluded from `committed` and reported as `next`.
    fn mtp_sequential_round(
        &mut self,
        metrics: &mut MetalMtpDecodeMetrics,
        pending_token: u32,
        room: usize,
        is_stop: &dyn Fn(u32) -> bool,
    ) -> crate::decoder::VerifiedRound {
        use crate::decoder::VerifiedRound;
        let pos = self.session.kv_cache.seq_len;
        if pos >= self.session.kv_cache.max_cache_len.saturating_sub(2) {
            // Not enough room for 2 more tokens: commit pending and stop.
            return VerifiedRound {
                committed: vec![pending_token],
                next: None,
                cache_full: true,
            };
        }

        // --- MTP draft phase ---
        // `mtp_forward_one` writes exactly one MTP KV cache row (for
        // `pending_token`) and advances `mtp.cache.seq_len` past it before this
        // round's verify/rollback machinery runs. Snapshot the pre-draft cursor
        // so a rejection restores the MTP cache to its state as of the top of
        // this round, not as of after the draft's own write: `verify_tokens_batched`
        // captures `mtp_base_seq_len` from whatever the cursor reads when *it* runs,
        // which is already past that row (#1341).
        let mtp_seq_len_pre_draft = self.session.mtp.as_ref().map(|m| m.cache.seq_len);
        let t_mtp = std::time::Instant::now();
        let draft = self.mtp_forward_one(pending_token, pos);
        metrics.mtp_ms += t_mtp.elapsed().as_secs_f64() * 1000.0;
        metrics.mtp_forwards += 1;

        // NOTE: We intentionally do NOT short-circuit on draft EOS here.
        // The draft's EOS guess must be verified against the target model.
        // If the target agrees, the round commits pending and reports the stop.
        // If the target disagrees, the reject path replaces the draft as usual.
        // Short-circuiting before verify caused concern #2 of #237: a wrong
        // draft-EOS would silently truncate generation early.

        // --- Verify phase (sequential verifier) ---
        let t_verify = std::time::Instant::now();
        let verify_result = self.verify_tokens_batched(&[pending_token, draft.token_id], pos);
        let Ok(verify_out) = verify_result else {
            // Fallback: commit pending only; the draft stays unverified and is only
            // a candidate, so the driver checks it before it becomes pending.
            self.session.rewind_position(pos + 1);
            metrics.fallback_tokens += 1;
            return VerifiedRound {
                committed: vec![pending_token],
                next: Some(draft.token_id),
                cache_full: false,
            };
        };
        metrics.verify_ms += t_verify.elapsed().as_secs_f64() * 1000.0;
        metrics.verify_calls += 1;
        // Correct the base the verifier just captured: it read the MTP cursor
        // after the draft phase already advanced it by one row, so restoring
        // `base_mtp + slot` on rejection would double-count that row and leave
        // the cursor one slot past the last one `mtp_forward_one` actually wrote.
        if let Some(ref mut p) = self.session.gdn_checkpoints {
            p.mtp_base_seq_len = mtp_seq_len_pre_draft;
        }

        // Route the accept/reject decision through `rejection_sample_draft` so
        // the live MTP loop and the trait-level `mtp_verify_draft` share the
        // same verifier implementation (ADR-050). This is the GREEDY MTP route
        // (entered only when top_k<=1 && temperature<=0.0), so verification uses
        // `greedy=true`: argmax-only acceptance that is deterministic and
        // token-for-token equivalent to plain greedy `generate()`. Passing
        // `greedy=false` here drove a clock-seeded RNG, making greedy MTP output
        // non-deterministic and not argmax-equivalent (#237). In greedy mode the
        // verifier never reads `draft_logits` (it may even be `&[]` per the
        // contract), but we still hand it the draft's own logits: the value is
        // ignored on the greedy path and this keeps `draft.logits` a live read.
        //   draft_tokens          = [draft.token_id]
        //   draft_logits          = [draft.logits]        (ignored in greedy mode)
        //   initial_target_logits = verify_out.logits[0]  (predicts draft position)
        //   target_logits         = [verify_out.logits[1]] (bonus on full accept)
        // temperature is ignored on the greedy=true path (argmax is invariant to
        // positive scaling), so any finite positive placeholder is safe here:
        // this route is entered only when gen_cfg.temperature <= 0.0, so the real
        // config value cannot satisfy `> 0` validation anyway.
        let Ok(rs) = crate::speculative::rejection_sample_draft(
            &[draft.token_id],
            std::slice::from_ref(&draft.logits),
            &verify_out.logits[0],
            std::slice::from_ref(&verify_out.logits[1]),
            true,
            1.0,
            None,
        ) else {
            // Fallback: commit pending only, as for a failed verify.
            self.session.rewind_position(pos + 1);
            metrics.fallback_tokens += 1;
            return VerifiedRound {
                committed: vec![pending_token],
                next: Some(draft.token_id),
                cache_full: false,
            };
        };

        // GPU state mutations must happen before the pure decision function:
        // - Accept: clear batch_repair_token (rollback won't run this round),
        //   and append the MTP-cache row `mtp_forward_one` never writes for
        //   the accepted draft token itself.
        // - Reject: roll back KV cache and GDN state to pos+1.
        let accepted = rs.accepted_count == 1;
        if accepted {
            if let Some(ref mut p) = self.session.gdn_checkpoints {
                p.batch_repair_token = None;
            }
            // lattice#1396: `mtp_forward_one` above wrote only the row for
            // `pending_token` at `pos`; a full accept commits `draft.token_id`
            // at `pos + 1` without ever giving it its own MTP-cache row, so
            // the next round's `mtp_forward_one(bonus_token, pos + 2)` lands
            // one physical slot behind its RoPE position (a positional hole
            // per accepted transition). `mtp_prefill_append` is the same
            // K/V-only append primitive `mtp_prefill` uses to backfill
            // historical prompt positions: a prefilled position is only
            // ever a future key, exactly this case. Pair the accepted
            // token's embedding with `first_pre_final_hidden`, the verify
            // pass's pre-final hidden for `pending_token`'s own position
            // (the hidden state that predicted the accepted draft), matching
            // the pairing `mtp_prefill_append`'s own doc comment establishes.
            self.mtp_prefill_append(draft.token_id, &verify_out.first_pre_final_hidden, pos + 1);
            metrics.accepted_extra_tokens += 1;
        } else {
            let t_rb = std::time::Instant::now();
            let _ = self.rollback_speculative_state_to(pos + 1);
            metrics.rollback_ms += t_rb.elapsed().as_secs_f64() * 1000.0;
            // The sequential verifier's rollback restores GDN state and the KV
            // cache cursor but not `last_pre_final_hidden`, which
            // `verify_tokens_batched` last wrote for the rejected draft token.
            // Restore it to the hidden state that predicted the target's real
            // replacement, or the next MTP draft is fed the wrong input.
            self.session.last_pre_final_hidden = verify_out.first_pre_final_hidden.clone();
            self.session.mark_pre_final_hidden(pos + 1);
        }

        // Delegate the pure emit/stop/continue decision to `mtp_greedy_round`.
        // `bonus_token` meaning depends on the accept/reject outcome:
        //   accept -> argmax(verify_out.logits[1])  (target pred for pos+2)
        //   reject -> argmax(verify_out.logits[0])  (target's replacement at pos+1)
        // Both are encoded in rs.bonus_token by `rejection_sample_draft`.
        let bonus_token = rs.bonus_token.unwrap_or(0);
        // A round can commit up to two tokens (pending and an accepted draft), and
        // the request may have room for fewer: clip to `room` so greedy MTP never
        // exceeds `max_new_tokens`. Plain greedy `generate()` enters only
        // `max_new_tokens - 1` decode steps, so it stops at the cap even when the
        // next token would be EOS.
        match super::mtp_greedy_round(
            pending_token,
            draft.token_id,
            accepted,
            bonus_token,
            is_stop,
        ) {
            super::MtpRoundOutcome::EmitAndStop(mut tokens) => {
                // Stop-token contract (#613): `tokens` holds only the committed
                // (non-stop) tokens for this round. The stop token that ended it is
                // reported as `next`, and the driver counts it as a stop only when the
                // request still had room for it (#632).
                let stop = if accepted && is_stop(draft.token_id) {
                    draft.token_id
                } else {
                    bonus_token
                };
                tokens.truncate(room);
                VerifiedRound {
                    committed: tokens,
                    next: Some(stop),
                    cache_full: false,
                }
            }
            super::MtpRoundOutcome::EmitAndContinue {
                mut emit,
                next_pending,
            } => {
                if emit.len() < room {
                    metrics.rounds += 1;
                }
                emit.truncate(room);
                VerifiedRound {
                    committed: emit,
                    next: Some(next_pending),
                    cache_full: false,
                }
            }
        }
    }

    /// One GDN-first self-speculative round: drafts up to `SELF_SPEC_MAX_DRAFT`
    /// tokens with GDN-only forwards, verifies `[pending] ++ drafts` with the full
    /// model, and returns `pending` plus the longest draft prefix the target agreed
    /// with. A rejected draft is rolled back here and never appears in the returned
    /// round.
    fn self_spec_round(
        &mut self,
        metrics: &mut SelfSpecMetrics,
        pending_token: u32,
        room: usize,
        is_stop: &dyn Fn(u32) -> bool,
    ) -> crate::decoder::VerifiedRound {
        use crate::decoder::VerifiedRound;
        // Shared first-wins argmax helper (ADR-080 C3, #783) rather than a
        // hand-rolled `max_by` closure, which kept the LAST tied maximum
        // instead of the engine-wide-contract first tied maximum.
        let argmax_logits = crate::sampling::argmax_f32_first_wins;

        let pos = self.session.kv_cache.seq_len;
        if pos + SELF_SPEC_MAX_DRAFT + 1 >= self.session.kv_cache.max_cache_len {
            return VerifiedRound {
                committed: vec![pending_token],
                next: None,
                cache_full: true,
            };
        }

        // --- Draft phase: K GDN-only forwards ---
        let t_draft = std::time::Instant::now();
        // Self-spec uses slot-based rollback: `verify_tokens_batched` populates one
        // slot per verify token (slot k = state after k tokens processed by the full
        // model from the *pre-draft* base). We do NOT take a `batch_repair_token`
        // shortcut here: that path replays only the pending token on rejection,
        // leaving GDN state inconsistent with the KV cache after partial acceptance.
        if let Some(ref mut p) = self.session.gdn_checkpoints {
            p.active_base_seq_len = Some(pos);
            p.mtp_base_seq_len = None;
            p.batch_repair_token = None;
        }
        // Slot 0 reserves the pre-draft GDN state. `forward_step_gdn_only` below
        // mutates the live GDN buffers; if we did not capture pre-draft state here,
        // the slot-based rollback (which expects verify_tokens_batched's slots
        // 1..=K to be full-model forwards from this pre-draft base) would be
        // computed against a contaminated reference. With no checkpoint pool, or a
        // failed capture, fall through to a single-token decode.
        if self.session.gdn_checkpoints.is_none()
            || self
                .checkpoint_gdn_to_slot(0, GdnStateTrafficScope::Decode)
                .is_err()
        {
            let logits = self
                .forward_step_inner(
                    pending_token,
                    pos,
                    false,
                    crate::forward::signpost::Scope::Decode,
                )
                .logits;
            let next = argmax_logits(&logits);
            if !is_stop(next) && 1 < room {
                metrics.fallback_tokens += 1;
            }
            return VerifiedRound {
                committed: vec![pending_token],
                next: Some(next),
                cache_full: false,
            };
        }

        let mut draft_tokens: Vec<u32> = Vec::with_capacity(SELF_SPEC_MAX_DRAFT);
        let first_draft_logits = self.forward_step_gdn_only(pending_token, pos);
        let first_draft = argmax_logits(&first_draft_logits);
        metrics.draft_forwards += 1;
        if !is_stop(first_draft) {
            draft_tokens.push(first_draft);
            let mut cur_draft = first_draft;
            for _ in 1..SELF_SPEC_MAX_DRAFT {
                if is_stop(cur_draft) {
                    break;
                }
                let d_pos = pos + draft_tokens.len();
                let logits = self.forward_step_gdn_only(cur_draft, d_pos);
                let next = argmax_logits(&logits);
                metrics.draft_forwards += 1;
                draft_tokens.push(next);
                cur_draft = next;
            }
        }
        metrics.draft_ms += t_draft.elapsed().as_secs_f64() * 1000.0;

        // --- Verify phase: full model over [pending_token] ++ draft_tokens ---
        let mut verify_input: Vec<u32> = Vec::with_capacity(1 + draft_tokens.len());
        verify_input.push(pending_token);
        verify_input.extend_from_slice(&draft_tokens);

        // Restore the pre-draft GDN state from slot 0 before verification. The
        // GDN-only draft forwards above mutated the live GDN buffers; without
        // this restore, `verify_tokens_batched` would compose full-model forwards
        // on top of the GDN-only draft mutations, producing slots that do NOT
        // match the canonical "pre-draft + N full-model tokens" semantics that
        // slot-based rollback relies on.
        let restore_ok = self
            .restore_gdn_slot_blocking(0, GdnStateTrafficScope::Decode)
            .is_ok();

        let t_verify = std::time::Instant::now();
        let Ok(verify_out) = (if restore_ok {
            self.verify_tokens_batched(&verify_input, pos)
        } else {
            Err(lattice_inference_core::InferenceError::Inference(
                "GDN pre-verify restore failed".into(),
            ))
        }) else {
            // Verification failed: commit pending; the first draft stays unverified
            // and is only a candidate for the driver to check.
            self.session.rewind_position(pos + 1);
            metrics.fallback_tokens += 1;
            return VerifiedRound {
                committed: vec![pending_token],
                next: Some(argmax_logits(&first_draft_logits)),
                cache_full: false,
            };
        };
        metrics.verify_ms += t_verify.elapsed().as_secs_f64() * 1000.0;
        metrics.verify_calls += 1;

        // verify_out.logits[i] = full-model output after processing verify_input[i].
        // Logits[0] predicts what comes after pending_token.
        // Compare logits[i] argmax with draft_tokens[i] for i in 0..draft_len.

        // pending_token is always accepted (full model processed it above).
        let mut committed: Vec<u32> = Vec::with_capacity(1 + draft_tokens.len());
        committed.push(pending_token);

        let mut accepted_drafts = 0usize;
        let mut rejection_next: Option<u32> = None;

        for (i, &draft) in draft_tokens.iter().enumerate() {
            if committed.len() >= room {
                break;
            }
            let target = argmax_logits(&verify_out.logits[i]);
            if target == draft {
                if is_stop(draft) {
                    // Stop-token contract (#613): the accepted draft is itself a
                    // stop and is never committed. Room was available (checked
                    // above), so the driver reports a genuine stop.
                    return VerifiedRound {
                        committed,
                        next: Some(draft),
                        cache_full: false,
                    };
                }
                committed.push(draft);
                accepted_drafts += 1;
                metrics.accepted_extra_tokens += 1;
            } else {
                rejection_next = Some(target);
                break;
            }
        }

        if let Some(next_token) = rejection_next {
            // Partial or full rejection: roll back GDN/KV to accepted prefix boundary.
            let t_rb = std::time::Instant::now();
            let _ = self.rollback_speculative_state_to(pos + accepted_drafts + 1);
            metrics.rollback_ms += t_rb.elapsed().as_secs_f64() * 1000.0;
            if !is_stop(next_token) && committed.len() < room {
                metrics.rounds += 1;
            }
            return VerifiedRound {
                committed,
                next: Some(next_token),
                cache_full: false,
            };
        }

        // Full acceptance: next pending = full-model prediction after last verified token.
        if let Some(ref mut p) = self.session.gdn_checkpoints {
            p.batch_repair_token = None;
        }
        let last_idx = draft_tokens
            .len()
            .min(verify_out.logits.len().saturating_sub(1));
        let next_pending = argmax_logits(&verify_out.logits[last_idx]);
        metrics.rounds += 1;
        VerifiedRound {
            committed,
            next: Some(next_pending),
            cache_full: false,
        }
    }

    /// The legacy batch-GEMM MTP loop (`LATTICE_MTP_BATCH` on, no adapter loaded).
    ///
    /// Kept as its own loop: the batch-GEMM verifier is an experimental route that
    /// is not part of the shared decode policy, and it never falls back to the
    /// sequential verifier. Its MoE panic is the verifier's own and is left alone.
    pub(super) fn generate_greedy_mtp_batch_gemm_legacy(
        &mut self,
        prefill_logits: &[f32],
        prompt_len: usize,
        tokenizer: &BpeTokenizer,
        gen_cfg: &GenerateConfig,
    ) -> GenerateOutput {
        self.record_speculative_route(
            SpeculativeRoute::MtpBatchGemmLegacy,
            crate::decoder::SpeculativeTrace::default(),
        );
        let cfg = self.engine.config.clone();

        // Mirror canonical plain greedy generation: max_new_tokens == 0
        // means "generate nothing".  Return before sampling so we never emit a
        // token the caller did not ask for — including the case-A prefill-EOS path
        // below, which would otherwise emit one stop token for a zero budget.
        if gen_cfg.max_new_tokens == 0 {
            return GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                stopped: false,
                stop_reason: Some(StopReason::Length),
                token_logprobs: vec![],
            };
        }

        // Greedy argmax from prefill logits. Shared first-wins helper
        // (ADR-080 C3, #783) rather than a hand-rolled `max_by`, which
        // keeps the LAST tied maximum instead of the engine-wide-contract
        // first tied maximum.
        let pending_first = crate::sampling::argmax_f32_first_wins(prefill_logits);

        let is_stop = |id: u32| id == cfg.eos_token_id || gen_cfg.stop_token_ids.contains(&id);

        if is_stop(pending_first) {
            // Stop-token contract (#613): the terminating token is excluded
            // from token_ids/text. max_new_tokens == 0 already returned above,
            // so budget was available for this token — this is a genuine
            // stop, not one clipped by the cap.
            return GenerateOutput {
                text: String::new(),
                token_ids: vec![],
                prompt_tokens: prompt_len,
                generated_tokens: 0,
                stopped: true,
                stop_reason: Some(StopReason::Eos),
                token_logprobs: vec![],
            };
        }

        let mut generated_ids: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
        let mut pending_token = pending_first;
        let mut metrics = MetalMtpDecodeMetrics::default();
        let mut stopped = false;
        let mut stop_reason = StopReason::Length;

        while generated_ids.len() < gen_cfg.max_new_tokens {
            let pos = self.session.kv_cache.seq_len;
            if pos >= self.session.kv_cache.max_cache_len.saturating_sub(2) {
                // Not enough room for 2 more tokens — flush pending and stop.
                generated_ids.push(pending_token);
                stop_reason = StopReason::KvFull;
                break;
            }

            // --- MTP draft phase ---
            // `mtp_forward_one` writes exactly one MTP KV cache row (for
            // `pending_token`) and advances `mtp.cache.seq_len` past it before this
            // round's verify/rollback machinery runs. Snapshot the pre-draft cursor
            // so a rejection restores the MTP cache to its state as of the top of
            // this round, not as of after the draft's own write — `verify_tokens_batched`/
            // `verify_tokens_batch_gemm` capture `mtp_base_seq_len` from whatever the
            // cursor reads when *they* run, which is already past that row (#1341).
            let mtp_seq_len_pre_draft = self.session.mtp.as_ref().map(|m| m.cache.seq_len);
            let t_mtp = std::time::Instant::now();
            let draft = self.mtp_forward_one(pending_token, pos);
            metrics.mtp_ms += t_mtp.elapsed().as_secs_f64() * 1000.0;
            metrics.mtp_forwards += 1;

            // NOTE: We intentionally do NOT short-circuit on draft EOS here.
            // The draft's EOS guess must be verified against the target model.
            // If the target agrees → emit pending + EOS, stop.
            // If the target disagrees → reject path replaces the draft as usual.
            // Short-circuiting before verify caused concern #2 of #237: a wrong
            // draft-EOS would silently truncate generation early.

            // --- Verify phase (LATTICE_MTP_BATCH=1 selects batch-GEMM verifier,
            // unless an adapter is loaded: that verifier applies none) ---
            let t_verify = std::time::Instant::now();
            let verify_result =
                self.verify_tokens_batch_gemm(&[pending_token, draft.token_id], pos);
            let Ok(verify_out) = verify_result else {
                // Fallback: accept pending, advance normally.
                self.session.rewind_position(pos + 1);
                generated_ids.push(pending_token);
                metrics.fallback_tokens += 1;
                pending_token = draft.token_id;
                continue;
            };
            metrics.verify_ms += t_verify.elapsed().as_secs_f64() * 1000.0;
            metrics.verify_calls += 1;
            // Correct the base the verifier just captured: it read the MTP cursor
            // after the draft phase already advanced it by one row, so restoring
            // `base_mtp + slot` on rejection would double-count that row and leave
            // the cursor one slot past the last one `mtp_forward_one` actually wrote.
            if let Some(ref mut p) = self.session.gdn_checkpoints {
                p.mtp_base_seq_len = mtp_seq_len_pre_draft;
            }

            // Route the accept/reject decision through `rejection_sample_draft` so
            // the live MTP loop and the trait-level `mtp_verify_draft` share the
            // same verifier implementation (ADR-050). This is the GREEDY MTP loop
            // (entered only when top_k<=1 && temperature<=0.0), so verification uses
            // `greedy=true`: argmax-only acceptance that is deterministic and
            // token-for-token equivalent to plain greedy `generate()`. Passing
            // `greedy=false` here drove a clock-seeded RNG, making greedy MTP output
            // non-deterministic and not argmax-equivalent (#237). In greedy mode the
            // verifier never reads `draft_logits` (it may even be `&[]` per the
            // contract), but we still hand it the draft's own logits: the value is
            // ignored on the greedy path and this keeps `draft.logits` a live read.
            //   draft_tokens          = [draft.token_id]
            //   draft_logits          = [draft.logits]        (ignored in greedy mode)
            //   initial_target_logits = verify_out.logits[0]  (predicts draft position)
            //   target_logits         = [verify_out.logits[1]] (bonus on full accept)
            // temperature is ignored on the greedy=true path (argmax is invariant to
            // positive scaling), so any finite positive placeholder is safe here —
            // this loop is entered only when gen_cfg.temperature <= 0.0 (see comment
            // above), so the real config value cannot satisfy `> 0` validation anyway.
            let Ok(rs) = crate::speculative::rejection_sample_draft(
                &[draft.token_id],
                std::slice::from_ref(&draft.logits),
                &verify_out.logits[0],
                std::slice::from_ref(&verify_out.logits[1]),
                true,
                1.0,
                None,
            ) else {
                // Fallback: accept pending, advance normally.
                self.session.rewind_position(pos + 1);
                generated_ids.push(pending_token);
                metrics.fallback_tokens += 1;
                pending_token = draft.token_id;
                continue;
            };

            // GPU state mutations must happen before the pure decision function:
            // - Accept: clear batch_repair_token (rollback won't run this round),
            //   and append the MTP-cache row `mtp_forward_one` never writes for
            //   the accepted draft token itself.
            // - Reject: roll back KV cache and GDN state to pos+1.
            let accepted = rs.accepted_count == 1;
            if accepted {
                if let Some(ref mut p) = self.session.gdn_checkpoints {
                    p.batch_repair_token = None;
                }
                // lattice#1396: `mtp_forward_one` above wrote only the row for
                // `pending_token` at `pos`; a full accept commits `draft.token_id`
                // at `pos + 1` without ever giving it its own MTP-cache row, so
                // the next round's `mtp_forward_one(bonus_token, pos + 2)` lands
                // one physical slot behind its RoPE position (a positional hole
                // per accepted transition). `mtp_prefill_append` is the same
                // K/V-only append primitive `mtp_prefill` uses to backfill
                // historical prompt positions — a prefilled position is only
                // ever a future key, exactly this case. Pair the accepted
                // token's embedding with `first_pre_final_hidden`, the verify
                // pass's pre-final hidden for `pending_token`'s own position
                // (the hidden state that predicted the accepted draft), matching
                // the pairing `mtp_prefill_append`'s own doc comment establishes.
                self.mtp_prefill_append(
                    draft.token_id,
                    &verify_out.first_pre_final_hidden,
                    pos + 1,
                );
                metrics.accepted_extra_tokens += 1;
            } else {
                let t_rb = std::time::Instant::now();
                let _ = self.rollback_speculative_state_to(pos + 1);
                metrics.rollback_ms += t_rb.elapsed().as_secs_f64() * 1000.0;
                // The batch-GEMM verifier's own repair-replay restores
                // `last_pre_final_hidden`, so nothing is restored here.
            }

            // Delegate the pure emit/stop/continue decision to `mtp_greedy_round`.
            // `bonus_token` meaning depends on the accept/reject outcome:
            //   accept → argmax(verify_out.logits[1])  (target pred for pos+2)
            //   reject → argmax(verify_out.logits[0])  (target's replacement at pos+1)
            // Both are encoded in rs.bonus_token by `rejection_sample_draft`.
            let bonus_token = rs.bonus_token.unwrap_or(0);
            // The loop guard (`generated_ids.len() < max_new_tokens`) guarantees at
            // least one slot remains here, but a round can emit up to three tokens
            // (case B: [pending, draft, bonus]).  Clip every emission to the
            // remaining budget so greedy MTP never exceeds `max_new_tokens` — plain
            // greedy `generate()` enters only `max_new_tokens - 1` decode steps, so
            // it stops at the cap even when the next token would be EOS.
            match super::mtp_greedy_round(
                pending_token,
                draft.token_id,
                accepted,
                bonus_token,
                is_stop,
            ) {
                super::MtpRoundOutcome::EmitAndStop(tokens) => {
                    let remaining = gen_cfg.max_new_tokens - generated_ids.len();
                    let emit_len = tokens.len().min(remaining);
                    generated_ids.extend_from_slice(&tokens[..emit_len]);
                    // Stop-token contract (#613): `tokens` holds only the committed
                    // (non-stop) tokens for this round — the stop token itself is
                    // never in it. #632: `emit_len == tokens.len()`
                    // alone is not sufficient — if the committed payload exactly
                    // fills `remaining`, the excluded stop token would have landed
                    // one position *beyond* `max_new_tokens`, so plain greedy would
                    // never have reached it either and reports Length, not Eos.
                    // `stopped` is therefore true only when every committed token
                    // was emitted AND at least one more slot remained for the
                    // (excluded) stop token itself — i.e. the stop was genuinely
                    // reached within budget, not merely coincident with the cap.
                    stopped = emit_len == tokens.len() && remaining > tokens.len();
                    if stopped {
                        stop_reason = StopReason::Eos;
                    }
                    break;
                }
                super::MtpRoundOutcome::EmitAndContinue { emit, next_pending } => {
                    let remaining = gen_cfg.max_new_tokens - generated_ids.len();
                    generated_ids.extend_from_slice(&emit[..emit.len().min(remaining)]);
                    if generated_ids.len() >= gen_cfg.max_new_tokens {
                        break;
                    }
                    pending_token = next_pending;
                }
            }
            metrics.rounds += 1;
        }

        // Last update to pre_final_hidden from the verify output.
        self.session.last_pre_final_hidden = {
            let pos = self.session.kv_cache.seq_len;
            if pos > 0 {
                // Re-run one step to get fresh hidden for any downstream caller.
                // (verify_tokens_batched already left last_pre_final_hidden current.)
                self.session.last_pre_final_hidden.clone()
            } else {
                self.session.last_hidden_cursor = None;
                vec![0.0f32; cfg.hidden_size]
            }
        };

        if lattice_model_io::env_switch_enabled("LATTICE_MTP_VERBOSE") {
            eprintln!(
                "[MTP] rounds={} mtp_fwd={} verify={} accepted_extra={} rollbacks={} fallbacks={} mtp_ms={:.1} verify_ms={:.1} rb_ms={:.1} tap={}",
                metrics.rounds,
                metrics.mtp_forwards,
                metrics.verify_calls,
                metrics.accepted_extra_tokens,
                metrics.verify_calls - metrics.accepted_extra_tokens,
                metrics.fallback_tokens,
                metrics.mtp_ms,
                metrics.verify_ms,
                metrics.rollback_ms,
                self.mtp_hidden_tap().map_or("none", MtpHiddenTap::label),
            );
        }

        let text = decode_tokens(tokenizer, &generated_ids);
        GenerateOutput {
            text,
            token_ids: generated_ids.clone(),
            prompt_tokens: prompt_len,
            generated_tokens: generated_ids.len(),
            stopped,
            stop_reason: Some(stop_reason),
            token_logprobs: vec![],
        }
    }

    /// **Unstable**: greedy n-gram speculative generation over raw token ids.
    ///
    /// Owns the complete Metal lifecycle: validates the request before
    /// mutation, resets recurrent state, prefills the prompt once, then
    /// forwards every emitted non-EOS token at its absolute position. On
    /// success with a non-zero generation budget, the live KV/GDN state
    /// therefore represents `prompt_tokens.len() + generated.len()` tokens.
    /// For a valid prompt, a zero generation budget returns an empty vector
    /// without resetting existing state.
    ///
    /// Models with MoE layers prefill per token, with or without a LoRA adapter.
    ///
    /// # Errors
    ///
    /// Returns [`lattice_inference_core::InferenceError::Inference`] for an empty
    /// prompt or, for a non-zero generation budget, when the prompt plus
    /// budget exceeds [`Self::max_context`]. Returns
    /// [`lattice_inference_core::InferenceError::InvalidInput`] when a prompt token
    /// cannot index the configured embedding table. Admission errors occur
    /// before reset, prefill, allocation of generation history, or GPU work.
    ///
    /// `max_ngram` is clamped to
    /// [`crate::speculative::MAX_NGRAM_LIMIT`]; the n-gram search scans the
    /// prompt once per attempted length, so an unbounded value would let a
    /// caller spend arbitrary CPU before the first token.
    ///
    pub fn generate_with_speculation(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        eos_token: u32,
        max_ngram: usize,
        max_draft: usize,
    ) -> Result<Vec<u32>, lattice_inference_core::InferenceError> {
        crate::model::qwen35::check_prompt_not_empty(prompt_tokens.len())?;
        crate::model::qwen35::check_prompt_ids_in_vocab(
            prompt_tokens,
            self.engine.config.vocab_size,
        )?;
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }
        crate::model::qwen35::check_context_budget(
            prompt_tokens.len(),
            None,
            max_new_tokens,
            self.max_context(),
        )?;

        self.reset_state();
        // Return prefill admission errors before starting speculation.
        let prefill_logits = self.try_forward_prefill(prompt_tokens)?;
        crate::speculative::generate_with_speculation_from_prefill(
            prompt_tokens,
            max_new_tokens,
            eos_token,
            prefill_logits,
            |token, position| self.try_forward_step(token, position),
            max_ngram,
            max_draft,
        )
    }
}
