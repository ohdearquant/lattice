//! Cross-turn KV prefix cache on a Metal state.

use super::*;
// ---------------------------------------------------------------------
// Cross-turn KV prefix cache (#462): Metal integration tests.
//
// `single_char_vocab_tokenizer` gives a 32-token, merge-free vocab
// (one token per ASCII letter) so growing a conversation by string
// concatenation is guaranteed to grow the token sequence by an exact
// append — no re-tokenization boundary can shift earlier tokens,
// which keeps these tests focused on the cache logic itself rather
// than on tokenizer edge cases.
// ---------------------------------------------------------------------

/// The correctness gate for #462: a growing multi-turn conversation run
/// through `generate_streaming_with_prefix_cache` on one long-lived state
/// must produce byte-identical token IDs, turn by turn, to running each
/// turn's full accumulated prompt through plain `generate_streaming` on a
/// fresh state (i.e. the pre-#462 "re-prefill everything every turn"
/// behavior). If cross-turn KV/GDN reuse ever perturbs a single sampled
/// token, this test goes red.
#[test]
fn cross_turn_cache_multiturn_token_identity_matches_full_reprefill() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    // Keep EOS unreachable so every turn generates its full budget,
    // guaranteeing the conversation keeps growing turn over turn.
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut cached_state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");

    let user_turns = ["a", "bc", "d", "ef", "g"];
    let mut conversation = String::new();
    let mut saw_exact_append = false;

    for (turn_idx, user_text) in user_turns.iter().enumerate() {
        conversation.push_str(user_text);
        let gen_cfg = cross_turn_test_gen_cfg(42, 3);

        let mut cached_delta_calls = 0u32;
        let cached_out = cached_state
            .generate_streaming_with_prefix_cache(
                slot_id,
                &conversation,
                &tokenizer,
                &gen_cfg,
                |_delta, _id| {
                    cached_delta_calls += 1;
                    true
                },
            )
            .expect("cache-aware generation must not error on a valid conversation");

        // Independent, from-scratch reference: a brand new state, full
        // re-prefill of the entire accumulated conversation so far —
        // exactly the behavior cross-turn reuse must reproduce.
        let mut reference_state =
            MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
        let reference_gen_cfg = cross_turn_test_gen_cfg(42, 3);
        let reference_out = reference_state
            .generate_streaming(
                &conversation,
                &tokenizer,
                &reference_gen_cfg,
                |_delta, _id| true,
            )
            .expect("no grammar configured; must not fail closed");

        assert_eq!(
            cached_out.output.token_ids, reference_out.token_ids,
            "turn {turn_idx}: cache-aware token IDs must match full re-prefill \
                     exactly (conversation so far: {conversation:?})"
        );
        assert_eq!(
            cached_out.output.text, reference_out.text,
            "turn {turn_idx}: cache-aware decoded text must match full re-prefill"
        );
        assert_eq!(
            cached_out.output.stop_reason, reference_out.stop_reason,
            "turn {turn_idx}: stop reason must match full re-prefill"
        );

        if matches!(
            cached_out.cache.mode,
            crate::kv_cache::PrefixReuseMode::ExactAppend
        ) {
            saw_exact_append = true;
            assert!(
                cached_out.cache.reused_tokens > 0,
                "turn {turn_idx}: ExactAppend must report a nonzero reused prefix"
            );
        }

        // Grow the conversation with the (validated-identical) reply so
        // the next turn's prompt is a real append onto this one.
        conversation.push_str(&cached_out.output.text);
    }

    assert!(
        saw_exact_append,
        "at least one later turn must actually exercise ExactAppend reuse — \
                 otherwise this test only proves the FullRefill fallback path works"
    );
}

/// Sibling of the above, but exercised through the `ChatMessage`-based
/// `chat_completion_streaming_with_prefix_cache` wrapper rather than raw
/// prompt strings — this is the exact entry point `chat_metal`'s
/// interactive REPL calls (#462), so it must reproduce plain
/// `chat_completion_streaming`'s output turn-for-turn on a growing
/// conversation history, including `format_chat_template`'s ChatML
/// framing and `<|im_end|>` stop-token handling.
#[test]
fn cross_turn_cache_chat_completion_matches_full_reprefill() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    const LEGACY_FIXTURE_CONTEXT: usize = 128;
    const CHAT_FIXTURE_CONTEXT: usize = 192;

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    // Keep EOS unreachable so every turn generates its full budget,
    // guaranteeing the history keeps growing turn over turn (mirrors the
    // raw-prompt sibling test above).
    cfg.eos_token_id = u32::MAX;
    cfg.max_position_embeddings = CHAT_FIXTURE_CONTEXT;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut cached_state =
        MetalQwen35State::new(&weights, &cfg, CHAT_FIXTURE_CONTEXT).expect("tiny hybrid fixture");

    let user_turns = ["a", "bc", "d", "ef"];
    let mut history: Vec<ChatMessage> = vec![ChatMessage::system("")];
    let mut saw_exact_append = false;
    let mut prompt_lengths = Vec::with_capacity(user_turns.len());

    for (turn_idx, user_text) in user_turns.iter().enumerate() {
        history.push(ChatMessage::user(*user_text));
        let gen_cfg = cross_turn_test_gen_cfg(42, 2);
        let rendered_prompt = format_chat_template(&history);
        let prompt_len = tokenizer.tokenize(&rendered_prompt).real_length;
        prompt_lengths.push(prompt_len);
        assert!(
            prompt_len + gen_cfg.max_new_tokens <= CHAT_FIXTURE_CONTEXT,
            "turn {turn_idx}: fixture context must cover the rendered prompt and decode cap"
        );

        let cached_out = cached_state
            .chat_completion_streaming_with_prefix_cache(
                slot_id,
                &history,
                &tokenizer,
                &gen_cfg,
                |_delta, _id| true,
            )
            .expect("cache-aware chat completion must not error on a valid history");

        // Independent, from-scratch reference: a brand new state, the
        // plain (non-cache) chat_completion_streaming path over the same
        // history — exactly chat_metal's pre-#462 behavior (reset_state +
        // full re-prefill of the whole ChatML-formatted history every turn).
        let mut reference_state = MetalQwen35State::new(&weights, &cfg, CHAT_FIXTURE_CONTEXT)
            .expect("tiny hybrid fixture");
        let reference_gen_cfg = cross_turn_test_gen_cfg(42, 2);
        let reference_out = reference_state
            .chat_completion_streaming(&history, &tokenizer, &reference_gen_cfg, |_delta, _id| true)
            .expect("reference chat completion must not error on a valid history");

        assert_eq!(
            cached_out.output.message.content, reference_out.message.content,
            "turn {turn_idx}: cache-aware chat completion must match full re-prefill exactly"
        );
        assert_eq!(
            cached_out.output.prompt_tokens, reference_out.prompt_tokens,
            "turn {turn_idx}: prompt_tokens must match full re-prefill"
        );
        assert_eq!(
            cached_out.output.completion_tokens, reference_out.completion_tokens,
            "turn {turn_idx}: completion_tokens must match full re-prefill"
        );

        if matches!(
            cached_out.cache.mode,
            crate::kv_cache::PrefixReuseMode::ExactAppend
        ) {
            saw_exact_append = true;
            assert!(
                cached_out.cache.reused_tokens > 0,
                "turn {turn_idx}: ExactAppend must report a nonzero reused prefix"
            );
        }

        // Grow history with the (validated-identical) reply so the next
        // turn's prompt is a real append onto this one, matching real
        // REPL usage in chat_metal.
        history.push(cached_out.output.message.clone());
    }

    assert_eq!(
        prompt_lengths,
        [51, 92, 132, 173],
        "record the four-turn ChatML footprint so template growth cannot silently \
                 invalidate this cache-parity fixture"
    );
    assert!(
        prompt_lengths
            .iter()
            .any(|&len| len + 2 > LEGACY_FIXTURE_CONTEXT),
        "the fixture must exercise history beyond the obsolete 128-token window"
    );
    assert!(
        saw_exact_append,
        "at least one later turn must actually exercise ExactAppend reuse through the \
                 ChatMessage-based wrapper — otherwise this test only proves the FullRefill \
                 path works"
    );
}

/// The correctness gate for #590: editing an earlier turn (a prompt
/// that diverges from the cached history mid-way, at or after a
/// retained sparse GDN checkpoint boundary) must take the
/// `ReplayFromCheckpoint` path — GDN state restored from the ring
/// snapshot, KV rows below the boundary reused in place, divergent
/// suffix prefilled forward — and produce byte-identical token IDs
/// to running the edited conversation as a full re-prefill on a
/// fresh state. If checkpoint replay ever perturbs a single sampled
/// token, this test goes red.
///
/// Mutation sensitivity: reverting the #590 restore arm to its old
/// fail-closed error makes the edited turn error out (the plan
/// selects `ReplayFromCheckpoint`, which the old arm rejected);
/// reverting the ring carry in `save_cross_turn_prefix` leaves the
/// plan at `FullRefill` and the mode assertion fails; disabling the
/// carry-time `checkpoint_survives_save` pruning lets the stale
/// turn-2 boundary survive the edit and the turn-5 probe's
/// `checkpoint_len == l1_len` assertion fails.
#[test]
fn cross_turn_replay_from_checkpoint_matches_full_reprefill() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut cached_state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let gen_cfg = cross_turn_test_gen_cfg(42, 3);

    // Turn 1: establish the entry whose boundary becomes the ring
    // checkpoint after turn 2's save.
    let turn1_prompt = "abc".to_string();
    let turn1 = cached_state
        .generate_streaming_with_prefix_cache(
            slot_id,
            &turn1_prompt,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
        )
        .expect("turn 1 must succeed");
    let history_through_turn1 = format!("{turn1_prompt}{}", turn1.output.text);

    // Turn 2: exact append. Its save must promote turn 1's boundary
    // snapshot into the checkpoint ring.
    let turn2_prompt = format!("{history_through_turn1}de");
    let turn2 = cached_state
        .generate_streaming_with_prefix_cache(
            slot_id,
            &turn2_prompt,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
        )
        .expect("turn 2 must succeed");
    assert_eq!(
        turn2.cache.mode,
        crate::kv_cache::PrefixReuseMode::ExactAppend,
        "turn 2 must exercise ExactAppend so its save carries turn 1's boundary \
                 into the checkpoint ring"
    );

    // Turn 3: second exact append — its save carries turn 1's
    // boundary forward AND promotes turn 2's boundary, so the ring
    // now holds two checkpoints at different depths.
    let turn2_full = format!("{turn2_prompt}{}", turn2.output.text);
    let turn3_prompt = format!("{turn2_full}gh");
    let turn3 = cached_state
        .generate_streaming_with_prefix_cache(
            slot_id,
            &turn3_prompt,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
        )
        .expect("turn 3 must succeed");
    assert_eq!(
        turn3.cache.mode,
        crate::kv_cache::PrefixReuseMode::ExactAppend,
        "turn 3 must exercise ExactAppend so the ring holds both the \
                 turn-1 and turn-2 boundaries"
    );
    // Single-char ASCII vocab: 1 byte == 1 token, so boundary
    // lengths are plain byte lengths of the transcript strings.
    let l1_len = history_through_turn1.len();
    let l2_len = turn2_full.len();

    // Turn 4: EDIT turn 2 — same history through turn 1, different
    // user text after it. The shared prefix with the cached entry
    // ends where "de" was; the deepest eligible ring checkpoint at
    // or below that divergence is turn 1's boundary (turn 2's is
    // deeper than the divergence and must be ineligible), so the
    // planner must select ReplayFromCheckpoint at exactly `l1_len`.
    let edited_prompt = format!("{history_through_turn1}wxyz");
    let edited = cached_state
        .generate_streaming_with_prefix_cache(
            slot_id,
            &edited_prompt,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
        )
        .expect("edited-history turn must succeed");
    assert_eq!(
        edited.cache.mode,
        crate::kv_cache::PrefixReuseMode::ReplayFromCheckpoint {
            checkpoint_len: l1_len
        },
        "editing turn 2 must replay from turn 1's boundary checkpoint"
    );
    assert!(
        edited.cache.reused_tokens > 0,
        "replay must report a nonzero reused prefix"
    );

    // Reference: the edited conversation as a full re-prefill on a
    // brand-new state — exactly the behavior replay must reproduce.
    let mut reference_state =
        MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let reference_gen_cfg = cross_turn_test_gen_cfg(42, 3);
    let reference_out = reference_state
        .generate_streaming(&edited_prompt, &tokenizer, &reference_gen_cfg, |_, _| true)
        .expect("reference re-prefill generate_streaming");
    assert_eq!(
        edited.output.token_ids, reference_out.token_ids,
        "replayed-from-checkpoint token IDs must match full re-prefill exactly"
    );
    assert_eq!(
        edited.output.text, reference_out.text,
        "replayed-from-checkpoint decoded text must match full re-prefill"
    );
    assert_eq!(
        edited.output.stop_reason, reference_out.stop_reason,
        "stop reason must match full re-prefill"
    );

    // Turn 5: probe that the save after the replay PRUNED the stale
    // turn-2 boundary (its history diverged) while keeping turn 1's.
    // Diverge at exactly the old turn-2 boundary depth: a stale
    // checkpoint surviving there would be the deepest eligible one,
    // the planner would pick it, and GDN state from the abandoned
    // timeline would drive generation. The correct ring holds only
    // turn 1's boundary at or below this divergence.
    let edited_full = format!("{edited_prompt}{}", edited.output.text);
    assert!(
        edited_full.len() > l2_len,
        "edited transcript must extend past the old turn-2 boundary"
    );
    // Pick a divergent continuation guaranteed to differ from the
    // generated char at the divergence point (vocab is a-z + A-F).
    let cont = edited_full.as_bytes()[l2_len];
    let div = if cont == b'A' { "BB" } else { "AA" };
    let probe_prompt = format!("{}{div}", &edited_full[..l2_len]);
    let probe = cached_state
        .generate_streaming_with_prefix_cache(
            slot_id,
            &probe_prompt,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
        )
        .expect("stale-checkpoint probe turn must succeed");
    assert_eq!(
        probe.cache.mode,
        crate::kv_cache::PrefixReuseMode::ReplayFromCheckpoint {
            checkpoint_len: l1_len
        },
        "the post-replay save must prune the stale turn-2 boundary; \
                 only turn 1's checkpoint may serve this divergence"
    );
    let mut probe_reference_state =
        MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let probe_reference = probe_reference_state
        .generate_streaming(
            &probe_prompt,
            &tokenizer,
            &cross_turn_test_gen_cfg(42, 3),
            |_, _| true,
        )
        .expect("probe reference generate_streaming");
    assert_eq!(
        probe.output.token_ids, probe_reference.token_ids,
        "probe token IDs must match full re-prefill exactly"
    );

    // A follow-up append onto the probe history must still reuse
    // cleanly.
    let followup_prompt = format!("{probe_prompt}{}h", probe.output.text);
    let followup = cached_state
        .generate_streaming_with_prefix_cache(
            slot_id,
            &followup_prompt,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
        )
        .expect("follow-up append after replay must succeed");
    assert_eq!(
        followup.cache.mode,
        crate::kv_cache::PrefixReuseMode::ExactAppend,
        "the post-replay save must leave a reusable entry for a plain append"
    );
    let mut followup_reference_state =
        MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let followup_reference = followup_reference_state
        .generate_streaming(
            &followup_prompt,
            &tokenizer,
            &cross_turn_test_gen_cfg(42, 3),
            |_, _| true,
        )
        .expect("followup reference generate_streaming");
    assert_eq!(
        followup.output.token_ids, followup_reference.token_ids,
        "post-replay append token IDs must match full re-prefill exactly"
    );
}
/// Prefix-mismatch invalidation: when the new prompt does not share the
/// cached entry's history at all, the planner must fall back to
/// `FullRefill` (never attempt to graft an unrelated suffix onto stale
/// KV/GDN state), and the output must still be correct.
#[test]
fn cross_turn_cache_prefix_mismatch_falls_back_to_full_refill() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let gen_cfg = cross_turn_test_gen_cfg(7, 2);

    let first = state
        .generate_streaming_with_prefix_cache(slot_id, "abc", &tokenizer, &gen_cfg, |_, _| true)
        .expect("first turn must succeed");
    assert_eq!(
        first.cache.mode,
        crate::kv_cache::PrefixReuseMode::FullRefill,
        "an empty cache must always plan FullRefill"
    );

    // Completely different conversation: zero shared token prefix
    // with the cached entry from "abc".
    let second = state
        .generate_streaming_with_prefix_cache(slot_id, "xyz", &tokenizer, &gen_cfg, |_, _| true)
        .expect("divergent-prefix turn must still succeed");
    assert_eq!(
        second.cache.mode,
        crate::kv_cache::PrefixReuseMode::FullRefill,
        "a prompt sharing no prefix with the cached entry must invalidate to FullRefill"
    );

    let mut reference_state =
        MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let reference_out = reference_state
        .generate_streaming("xyz", &tokenizer, &gen_cfg, |_, _| true)
        .expect("no grammar configured; must not fail closed");
    assert_eq!(
        second.output.token_ids, reference_out.token_ids,
        "after a prefix-mismatch invalidation, output must still match full re-prefill"
    );
}

/// `clear_cross_turn_prefix_cache` must force the next call to plan
/// `FullRefill` even when the new prompt would otherwise be a valid
/// `ExactAppend` continuation of the (now-cleared) entry.
#[test]
fn cross_turn_cache_clear_forces_full_refill_next_call() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let gen_cfg = cross_turn_test_gen_cfg(3, 2);

    let first = state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &gen_cfg, |_, _| true)
        .expect("first turn must succeed");
    let mut conversation = "ab".to_string();
    conversation.push_str(&first.output.text);

    state.clear_cross_turn_prefix_cache();

    let second = state
        .generate_streaming_with_prefix_cache(
            slot_id,
            &conversation,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
        )
        .expect("second turn after clear must succeed");
    assert_eq!(
        second.cache.mode,
        crate::kv_cache::PrefixReuseMode::FullRefill,
        "a cleared cache slot must never be treated as a valid cached entry"
    );
}
