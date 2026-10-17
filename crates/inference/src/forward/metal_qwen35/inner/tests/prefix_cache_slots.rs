//! Prefix-cache slot isolation and eviction regressions.

use super::*;
// -------------------------------------------------------------------
// #516 remediation — regression coverage for
// D1/D3/D4/D5.
// -------------------------------------------------------------------

/// D1 (slot isolation): a divergent-prompt call on a
/// different slot must evict the single retained entry (D1's
/// single-live-entry cache), so a later append to the ORIGINAL
/// slot's prefix cannot silently restore GDN state left by a
/// different slot's generation while attention reads whatever KV
/// rows the other slot's generation wrote.
#[test]
fn cross_turn_cache_slot_isolation_evicts_other_slot() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_a = crate::kv_cache::CrossTurnSlotId::new(1);
    let slot_b = crate::kv_cache::CrossTurnSlotId::new(2);
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let gen_cfg = cross_turn_test_gen_cfg(11, 2);

    // Save slot A.
    let first = state
        .generate_streaming_with_prefix_cache(slot_a, "ab", &tokenizer, &gen_cfg, |_, _| true)
        .expect("slot A first turn must succeed");
    let mut conv_a = "ab".to_string();
    conv_a.push_str(&first.output.text);

    // Divergent-prompt call on slot B: under D1, `insert` replaces
    // whatever is retained regardless of slot, so this evicts A's
    // entry and retains B's.
    state
        .generate_streaming_with_prefix_cache(slot_b, "xyz", &tokenizer, &gen_cfg, |_, _| true)
        .expect("slot B turn must succeed");

    // Append-only continuation of A's prefix, issued on slot A
    // AFTER B evicted the single retained entry: must NOT
    // ExactAppend (there is no A entry left to restore from).
    let second = state
        .generate_streaming_with_prefix_cache(slot_a, &conv_a, &tokenizer, &gen_cfg, |_, _| true)
        .expect("slot A second turn must succeed");
    assert_eq!(
        second.cache.reused_tokens, 0,
        "slot A must not reuse GDN/KV state left by slot B's generation"
    );
    assert_ne!(
        second.cache.mode,
        crate::kv_cache::PrefixReuseMode::ExactAppend,
        "a different-slot eviction must force a FullRefill, never ExactAppend"
    );

    let mut reference_state =
        MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let reference_out = reference_state
        .generate_streaming(&conv_a, &tokenizer, &gen_cfg, |_, _| true)
        .expect("no grammar configured; must not fail closed");
    assert_eq!(
        second.output.token_ids, reference_out.token_ids,
        "after cross-slot eviction, output must still match full re-prefill"
    );
}

/// D3 (plain-path leg): interleaving a plain
/// `generate_streaming` call (which goes through `reset_state`)
/// between two cache-aware calls must invalidate the retained
/// entry — the plain path overwrites live KV/GDN state without
/// ever calling `save_cross_turn_prefix`.
///
/// Mutation sensitivity: reverting D3 (removing
/// `self.cross_turn_prefix_cache.clear()` from `reset_state`) makes
/// this test fail, because the stale entry would still report
/// `ExactAppend` against state the plain path already overwrote.
#[test]
fn cross_turn_cache_interleaved_plain_path_invalidates() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let gen_cfg = cross_turn_test_gen_cfg(21, 2);

    // Save a cross-turn entry.
    let first = state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &gen_cfg, |_, _| true)
        .expect("cache-aware first turn must succeed");
    let mut conversation = "ab".to_string();
    conversation.push_str(&first.output.text);

    // Plain path: a different prompt, routed through `reset_state`,
    // never touches the cross-turn cache's save path.
    let _ = state.generate_streaming("zzzzzz", &tokenizer, &gen_cfg, |_, _| true);

    // A genuine append (strictly longer than the cached entry) onto
    // the ORIGINAL prefix through the cache-aware path again: if the
    // stale entry survived, this satisfies D5's `len > shared` guard
    // too (it is NOT an exact-equal retry), so it would still look
    // like a valid ExactAppend continuation of `conversation` and
    // this test would only be sensitive to D3, not accidentally
    // masked by D5's separate guard.
    let mut appended = conversation.clone();
    appended.push('q');
    let second = state
        .generate_streaming_with_prefix_cache(slot_id, &appended, &tokenizer, &gen_cfg, |_, _| true)
        .expect("cache-aware second turn must succeed");
    assert_eq!(
        second.cache.reused_tokens, 0,
        "an interleaved plain generate_streaming call must invalidate the retained entry"
    );

    let mut reference_state =
        MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let reference_out = reference_state
        .generate_streaming(&appended, &tokenizer, &gen_cfg, |_, _| true)
        .expect("no grammar configured; must not fail closed");
    assert_eq!(
        second.output.token_ids, reference_out.token_ids,
        "after an interleaved plain-path reset, output must still match full re-prefill"
    );
}

/// D5: retrying the exact same prompt through the
/// cache-aware path must not hit the empty-suffix invariant error
/// in `forward_prefill_from` — the planner must fall back to
/// `FullRefill` when the new prompt exactly equals the entry's
/// represented tokens.
///
/// Mutation sensitivity: reverting D5 (dropping the
/// `new_prompt_ids.len() > shared` conjunct in
/// `plan_prefix_reuse`) makes this test fail: the planner would
/// again produce `ExactAppend` with `suffix_len == 0`, and
/// `generate_streaming_with_prefix_cache` would return
/// `Err(InferenceError::PrefixCache(..))` instead of `Ok`.
#[test]
fn cross_turn_cache_exact_equal_retry_full_refills() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let gen_cfg = cross_turn_test_gen_cfg(5, 2);

    let first = state
        .generate_streaming_with_prefix_cache(slot_id, "abc", &tokenizer, &gen_cfg, |_, _| true)
        .expect("first call must succeed");

    // The saved entry represents `prompt + generated tokens`, NOT
    // just the original prompt (see the `represented_token_ids`
    // construction in `generate_streaming_with_prefix_cache_inner`).
    // To actually exercise the exact-equal boundary the second call
    // must retry with THIS accumulated conversation — retrying with
    // the bare "abc" prompt again would only exercise the ordinary
    // mid-history-divergence FullRefill path (shared < represented_len),
    // never reaching the `shared == represented_len` branch this test
    // targets.
    let mut conversation = "abc".to_string();
    conversation.push_str(&first.output.text);

    let second = state
        .generate_streaming_with_prefix_cache(
            slot_id,
            &conversation,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
        )
        .expect("exact-equal prompt retry must not error");
    assert_eq!(
        second.cache.reused_tokens, 0,
        "an exact-equal prompt retry must plan FullRefill, not a zero-length ExactAppend"
    );

    let mut reference_state =
        MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let reference_out = reference_state
        .generate_streaming(&conversation, &tokenizer, &gen_cfg, |_, _| true)
        .expect("no grammar configured; must not fail closed");
    assert_eq!(
        second.output.token_ids, reference_out.token_ids,
        "exact-equal prompt retry output must match full re-prefill"
    );
}

/// D4: LoRA adapter load/unload must invalidate the
/// retained cross-turn entry. The adapter-identity hash in
/// `CrossTurnPrefixMetadata` is shape-based (max_rank/scale/
/// projection count), not content-based, so two different adapters
/// with the same shape can collide on the same `AdapterId` — the
/// unconditional clear on load/unload is the actual correctness
/// mechanism, not the metadata hash. Asserted directly against the
/// cache struct (this `mod tests` is a child module of
/// `metal_qwen35`, so the private `entry` field is visible here).
#[test]
fn cross_turn_cache_lora_load_unload_invalidates() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let gen_cfg = cross_turn_test_gen_cfg(9, 2);

    state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &gen_cfg, |_, _| true)
        .expect("save call must succeed");
    assert!(
        state.cross_turn_prefix_cache.entry.is_some(),
        "precondition: a save must actually retain an entry"
    );

    // `tiny_hybrid_fixture`'s only FullAttention layer is index 3
    // (layers 0-2 are LinearAttention/GDN); `make_valid_layer`
    // defaults to layer_idx 0, which is only valid against the
    // single-FullAttention-layer fixture other LoRA tests use.
    let mut lora_layer = make_valid_layer(512, 2);
    lora_layer.layer_idx = 3;
    state
        .load_lora_adapter(vec![lora_layer], 1.0, None)
        .expect("load_lora_adapter with a valid small fixture must succeed");
    assert!(
        state.cross_turn_prefix_cache.entry.is_none(),
        "load_lora_adapter must clear the retained cross-turn entry"
    );

    // Re-save under the loaded adapter, then unload and check
    // invalidation again.
    state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &gen_cfg, |_, _| true)
        .expect("save call under LoRA must succeed");
    assert!(
        state.cross_turn_prefix_cache.entry.is_some(),
        "precondition: a save under LoRA must actually retain an entry"
    );

    state.unload_lora_adapter();
    assert!(
        state.cross_turn_prefix_cache.entry.is_none(),
        "unload_lora_adapter must clear the retained cross-turn entry"
    );
}

/// D7: a public raw-forward call
/// (`forward_step`) interleaved between two cache-aware calls on the
/// SAME slot must invalidate the retained entry, exactly like
/// `generate_streaming`'s `reset_state()` does for D3's
/// plain-path leg (`cross_turn_cache_interleaved_plain_path_invalidates`
/// above) — `forward_step` mutates the same live KV/GDN buffers
/// without ever routing through `reset_state` or the cache's own
/// save path, so it is a second, independent way to leave a stale
/// entry pointing at overwritten state.
///
/// Uses a strictly-longer continuation of the ORIGINAL conversation
/// (append one extra character) for the second call, same as D3's
/// test: an exact-equal retry would independently trip D5's
/// `FullRefill` guard (`cross_turn_cache_exact_equal_retry_full_refills`)
/// and mask whether D7's clear is actually doing anything.
///
/// Mutation sensitivity: reverting D7 (removing
/// `self.cross_turn_prefix_cache.clear()` from `forward_step`) makes
/// this test fail — see the reverse-apply evidence recorded in the
/// PR body.
#[test]
fn cross_turn_cache_interleaved_raw_forward_invalidates() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let gen_cfg = cross_turn_test_gen_cfg(31, 2);

    // Save a cross-turn entry via the cache-aware path (slot A's
    // only slot, since `MetalCrossTurnPrefixCache` retains at most
    // one live entry across the whole state — see D1).
    let first = state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &gen_cfg, |_, _| true)
        .expect("cache-aware first turn must succeed");
    let mut conversation = "ab".to_string();
    conversation.push_str(&first.output.text);
    assert!(
        state.cross_turn_prefix_cache.entry.is_some(),
        "precondition: the first cache-aware call must actually retain an entry"
    );

    // Interleave a raw public forward call on an unrelated token
    // stream. This is the exact stale path: `forward_step`
    // advances `self.session.kv_cache` / GDN state directly, bypassing
    // both `reset_state()` (D3's guard) and the cache-aware save path.
    let live_cursor = state.session.kv_cache.seq_len;
    let _ = state.forward_step(0, live_cursor);

    // The raw call must have invalidated the retained entry outright
    // (not just left it stale) — this is what D7 adds.
    assert!(
        state.cross_turn_prefix_cache.entry.is_none(),
        "forward_step must clear the retained cross-turn entry before mutating live state"
    );

    // Strict-append continuation of the ORIGINAL slot's conversation:
    // if the entry had survived, this shape would still satisfy D5's
    // `len > shared` guard and look like a valid ExactAppend, so the
    // assertion below is genuinely exercising D7 and not being masked
    // by D5.
    let mut appended = conversation.clone();
    appended.push('q');
    let second = state
        .generate_streaming_with_prefix_cache(slot_id, &appended, &tokenizer, &gen_cfg, |_, _| true)
        .expect("cache-aware second turn must succeed");
    assert_eq!(
        second.cache.reused_tokens, 0,
        "an interleaved raw forward_step call must force a full refill, not a stale restore"
    );
    assert_ne!(
        second.cache.mode,
        crate::kv_cache::PrefixReuseMode::ExactAppend,
        "a raw-forward-invalidated entry must never be restored via ExactAppend"
    );

    let mut reference_state =
        MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let reference_out = reference_state
        .generate_streaming(&appended, &tokenizer, &gen_cfg, |_, _| true)
        .expect("no grammar configured; must not fail closed");
    assert_eq!(
        second.output.token_ids, reference_out.token_ids,
        "after an interleaved raw forward_step call, output must still match full re-prefill"
    );
}

/// #516: `set_gdn_chunked` swaps the GDN prefill
/// algorithm, and the two scans are not bit-exact (measured logit
/// drift up to ~2e-1 at chunk boundaries). A cache entry saved under
/// one mode must not seed a restore whose full-refill reference would
/// use the other mode. Flipping the mode must evict the retained
/// entry; a no-op set must NOT (so benchmarks re-asserting the current
/// mode keep their cache).
#[test]
fn cross_turn_cache_invalidated_by_gdn_mode_flip() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let gen_cfg = cross_turn_test_gen_cfg(13, 2);

    let first = state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &gen_cfg, |_, _| true)
        .expect("cache-aware first turn must succeed");
    let mut conversation = "ab".to_string();
    conversation.push_str(&first.output.text);
    assert!(
        state.cross_turn_prefix_cache.entry.is_some(),
        "precondition: the first cache-aware call must actually retain an entry"
    );

    // Re-asserting the CURRENT mode is a no-op and must keep the entry.
    let current_mode = state.use_gdn_chunked;
    state.set_gdn_chunked(current_mode);
    assert!(
        state.cross_turn_prefix_cache.entry.is_some(),
        "setting the already-active GDN mode must not evict the cache entry"
    );

    // Flipping the mode must evict it outright.
    state.set_gdn_chunked(!current_mode);
    assert!(
        state.cross_turn_prefix_cache.entry.is_none(),
        "flipping the GDN prefill mode must clear the retained cross-turn entry"
    );

    // Strict-append continuation: if the entry had survived the flip,
    // this shape would satisfy the ExactAppend guards, so the
    // assertions below genuinely exercise the mode-flip eviction.
    let mut appended = conversation.clone();
    appended.push('q');
    let second = state
        .generate_streaming_with_prefix_cache(slot_id, &appended, &tokenizer, &gen_cfg, |_, _| true)
        .expect("cache-aware second turn must succeed");
    assert_eq!(
        second.cache.reused_tokens, 0,
        "a GDN mode flip must force a full refill, not a cross-mode restore"
    );
    assert_ne!(
        second.cache.mode,
        crate::kv_cache::PrefixReuseMode::ExactAppend,
        "an entry saved under the other GDN prefill mode must never be restored via ExactAppend"
    );
}

/// #462 serve-wiring correctness gate for
/// `generate_streaming_with_prefix_cache_and_cancel`: an immediate
/// `should_cancel` (true before any GPU work runs) must return
/// `StopReason::Interrupt` with zero generated tokens, and must not
/// leave live state or the retained cache entry in a shape that
/// corrupts a later, uncancelled call on the same state — i.e. the
/// fail-closed guarantee the doc comment on
/// `generate_streaming_with_prefix_cache_and_cancel` promises.
#[test]
fn cross_turn_cache_and_cancel_immediate_leaves_state_usable() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let gen_cfg = cross_turn_test_gen_cfg(7, 3);

    // Turn 1: establish a normal, saved cache entry.
    let turn1 = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            "a",
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || false,
        )
        .expect("turn 1 must not error");
    assert!(
        turn1.output.generated_tokens > 0,
        "precondition: turn 1 must actually generate tokens"
    );

    // Turn 2: should_cancel is already true, so no GPU work for this
    // turn should ever run. Must surface as Interrupt, not an error,
    // and must not generate any tokens.
    let turn2 = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            "ab",
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || true,
        )
        .expect("an immediate cancel must not surface as an engine error");
    assert_eq!(
        turn2.output.stop_reason,
        Some(StopReason::Interrupt),
        "an immediate should_cancel=true must stop with Interrupt"
    );
    assert_eq!(
        turn2.output.generated_tokens, 0,
        "an immediate cancel must not generate any tokens"
    );

    // Turn 3: uncancelled call on the same conversation prefix as the
    // cancelled turn 2. If the cancel path had left live KV/GDN state
    // or the retained cache entry inconsistent, this would either
    // error or diverge from a from-scratch reference.
    let mut reference_state =
        MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let reference_out = reference_state
        .generate_streaming("ab", &tokenizer, &gen_cfg, |_, _| true)
        .expect("reference generate_streaming must not error");

    let turn3 = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            "ab",
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || false,
        )
        .expect("turn 3 must not error after a cancelled turn 2");
    assert_eq!(
        turn3.output.token_ids, reference_out.token_ids,
        "post-cancel generation must still match full re-prefill exactly"
    );
}

/// Serve-boundary correctness gate: `lattice_serve` always calls the
/// cancel-aware prefix-cache API on the single `CrossTurnSlotId::DEFAULT`
/// slot regardless of which logical conversation issued the request.
/// Two interleaved conversations sharing that one slot (A1, B1, A2)
/// must therefore behave exactly like the plain slot-isolation case
/// above: A2's shared prefix with the retained entry (now B's) is not
/// a match, so the planner must fall back to `FullRefill`, and the
/// output must still be correct.
#[test]
fn cross_turn_cache_and_cancel_same_slot_interleave_falls_back_to_full_refill() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let gen_cfg = cross_turn_test_gen_cfg(23, 2);

    // A1: establish a saved entry on the DEFAULT slot.
    let a1 = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            "ab",
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || false,
        )
        .expect("A1 must succeed");
    let mut conv_a = "ab".to_string();
    conv_a.push_str(&a1.output.text);

    // B1: an unrelated conversation on the SAME slot -- exactly what
    // `lattice_serve` produces when two different clients' requests
    // interleave through its single worker thread. Under the
    // single-entry invariant this replaces A's retained entry.
    state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            "xyz",
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || false,
        )
        .expect("B1 must succeed");

    // A2: append-only continuation of A's own conversation, issued
    // right after B1 evicted the shared slot. Must miss cleanly and
    // fall back to FullRefill -- never restore B's GDN/KV state.
    let a2 = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            &conv_a,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || false,
        )
        .expect("A2 must succeed");
    assert_eq!(
        a2.cache.reused_tokens, 0,
        "A2 must not reuse state left by B1's interleaved generation"
    );
    assert_ne!(
        a2.cache.mode,
        crate::kv_cache::PrefixReuseMode::ExactAppend,
        "same-slot interleaving must force FullRefill for A2, never ExactAppend"
    );

    let mut reference_state =
        MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let reference_out = reference_state
        .generate_streaming(&conv_a, &tokenizer, &gen_cfg, |_, _| true)
        .expect("reference generate_streaming must not error");
    assert_eq!(
        a2.output.token_ids, reference_out.token_ids,
        "after same-slot interleaved eviction, A2's output must still match \
                 full re-prefill exactly -- this is the parity the plain (uncached) \
                 path already guaranteed before #462"
    );
}

/// Checkpoint 2 of the cancellation contract: a `should_cancel`
/// observed right after the suffix prefill returns (but before any
/// decode work) must stop with `Interrupt`, generate zero tokens, and
/// leave state usable for a later uncancelled call on the same
/// conversation.
#[test]
fn cross_turn_cache_and_cancel_after_prefill_leaves_state_usable() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let gen_cfg = cross_turn_test_gen_cfg(9, 3);

    // Turn 1: establish a saved entry.
    let turn1 = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            "a",
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || false,
        )
        .expect("turn 1 must not error");
    assert!(turn1.output.generated_tokens > 0);

    // Turn 2: should_cancel is false on the first poll (checkpoint 1,
    // before the suffix prefill) and true on the second (checkpoint 2,
    // immediately after it returns) -- so the suffix prefill runs but
    // no decode work ever does.
    let poll_count = std::cell::Cell::new(0u32);
    let turn2 = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            "ab",
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || {
                let n = poll_count.get() + 1;
                poll_count.set(n);
                n >= 2
            },
        )
        .expect("a checkpoint-2 cancel must not surface as an engine error");
    assert!(
        poll_count.get() >= 2,
        "precondition: should_cancel must have been polled at least twice \
                 (checkpoint 1 then checkpoint 2) for this test to exercise checkpoint 2"
    );
    assert_eq!(
        turn2.output.stop_reason,
        Some(StopReason::Interrupt),
        "a checkpoint-2 cancel (right after the suffix prefill) must stop \
                 with Interrupt"
    );
    assert_eq!(
        turn2.output.generated_tokens, 0,
        "a checkpoint-2 cancel must not generate any tokens -- the suffix \
                 prefill produced only the (unsampled) next-token logits"
    );

    // Turn 3: uncancelled call on the same conversation. If checkpoint
    // 2's early return had left live KV/GDN state or the retained
    // cache entry inconsistent, this would either error or diverge.
    let mut reference_state =
        MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let reference_out = reference_state
        .generate_streaming("ab", &tokenizer, &gen_cfg, |_, _| true)
        .expect("reference generate_streaming must not error");
    let turn3 = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            "ab",
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || false,
        )
        .expect("turn 3 must not error after a checkpoint-2-cancelled turn 2");
    assert_eq!(
        turn3.output.token_ids, reference_out.token_ids,
        "post-checkpoint-2-cancel generation must still match full re-prefill exactly"
    );
}

/// Checkpoint 3 of the cancellation contract: a `should_cancel`
/// observed at the top of a decode iteration, after at least one token
/// has already been forwarded, must stop with `Interrupt`, report
/// exactly the tokens actually forwarded before the cancel was
/// observed, and -- per the doc comment on
/// `generate_streaming_with_prefix_cache_and_cancel` -- this case is
/// state-consistent and safe to save: a later append onto exactly the
/// text produced so far must still hit `ExactAppend` and match a full
/// re-prefill of the same extended conversation.
#[test]
fn cross_turn_cache_and_cancel_mid_decode_leaves_state_usable() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let gen_cfg = cross_turn_test_gen_cfg(17, 4);

    // should_cancel is polled once at checkpoint 1, once at checkpoint
    // 2, then once per decode iteration at checkpoint 3. Cancel on the
    // second decode-loop poll (the 4th poll overall) so exactly two
    // tokens (the prefill-sampled one plus one decode-loop token) are
    // forwarded before the cancelled iteration.
    let poll_count = std::cell::Cell::new(0u32);
    let turn1 = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            "ab",
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || {
                let n = poll_count.get() + 1;
                poll_count.set(n);
                n >= 4
            },
        )
        .expect("a mid-decode cancel must not surface as an engine error");
    assert_eq!(
        turn1.output.stop_reason,
        Some(StopReason::Interrupt),
        "a checkpoint-3 (mid-decode) cancel must stop with Interrupt"
    );
    assert!(
        turn1.output.generated_tokens >= 1
            && turn1.output.generated_tokens < gen_cfg.max_new_tokens,
        "a checkpoint-3 cancel must report exactly the tokens forwarded before \
                 the cancel was observed, not zero and not the full cap: got {}",
        turn1.output.generated_tokens
    );

    // The conversation through exactly the delivered text, plus a new
    // suffix: resubmitting the cancelled turn's own history
    // byte-for-byte with nothing new to append is itself a
    // FullRefill case (see `cross_turn_cache_exact_equal_retry_full_refills`),
    // so a real client turn always adds at least one new token before
    // its next request.
    let history = format!("ab{}", turn1.output.text);
    let followup_prompt = format!("{history}q");

    // Reference: the full uncancelled conversation through this exact
    // point, on a fresh state.
    let mut reference_state =
        MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let reference_out = reference_state
        .generate_streaming(&followup_prompt, &tokenizer, &gen_cfg, |_, _| true)
        .expect("reference generate_streaming must not error");

    // A follow-up append onto exactly the delivered text must hit
    // ExactAppend, proving the cancelled turn's save left a
    // consistent, reusable boundary.
    let turn2 = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            &followup_prompt,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || false,
        )
        .expect("follow-up turn after mid-decode cancel must not error");
    assert_eq!(
        turn2.cache.mode,
        crate::kv_cache::PrefixReuseMode::ExactAppend,
        "a mid-decode cancel is state-consistent and safe to save -- the \
                 follow-up append must reuse it, not fall back to FullRefill"
    );
    assert_eq!(
        turn2.output.token_ids, reference_out.token_ids,
        "post-mid-decode-cancel generation must still match full re-prefill exactly"
    );
}

/// Client-disconnect contract, distinct from the `should_cancel` tests
/// above: `on_token` returning `false` mid-decode -- the SSE-send-failed
/// signal a real HTTP disconnect produces -- must never let a later
/// request be served text the client was never actually sent.
/// Re-issuing the ORIGINAL (pre-disconnect) prompt must reproduce the
/// reference output for that original prompt exactly, proving the
/// retained entry cannot silently graft the undelivered continuation
/// onto a request that never asked for it.
#[test]
fn cross_turn_cache_and_cancel_on_token_disconnect_never_serves_undelivered_tokens() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let gen_cfg = cross_turn_test_gen_cfg(31, 3);
    let prompt = "ab".to_string();

    // Simulate an SSE client that hangs up after its first delta: the
    // callback returns `false` from then on, exactly like
    // `run_worker_loop`'s `on_token` once `cb_tx.send` starts failing.
    let sent_count = std::cell::Cell::new(0u32);
    let disconnect = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            &prompt,
            &tokenizer,
            &gen_cfg,
            |_delta, _id| {
                let n = sent_count.get() + 1;
                sent_count.set(n);
                n < 2
            },
            || false,
        )
        .expect("a disconnect must not surface as an engine error");
    assert_eq!(
        disconnect.output.stop_reason,
        Some(StopReason::Interrupt),
        "on_token returning false must stop with Interrupt"
    );

    // The client only ever has the original prompt -- it never saw
    // `disconnect.output.text`. Re-issuing exactly that original
    // prompt must reproduce the same output a fresh full re-prefill
    // of that prompt would: the planner's exact-prefix check can only
    // ExactAppend when the shared length matches the represented
    // length, so a shorter re-submitted prompt cannot silently
    // inherit the undelivered suffix.
    let mut reference_state =
        MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let reference_out = reference_state
        .generate_streaming(&prompt, &tokenizer, &gen_cfg, |_, _| true)
        .expect("reference generate_streaming must not error");
    let replay = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            &prompt,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || false,
        )
        .expect("re-issuing the original prompt must not error");
    assert_eq!(
        replay.output.token_ids, reference_out.token_ids,
        "re-issuing exactly the client's last-known prompt must reproduce the \
                 same output a full re-prefill would -- the client can never receive \
                 tokens generated after it disconnected"
    );
}

/// Regression test: an `on_token` rejection mid-decode must exclude the
/// rejected token from the persisted cache boundary, not just leave a
/// safe-by-coincidence fallback. This is a stronger check than output
/// parity alone (re-submitting a *shorter*, already-fully-cached prompt
/// always falls back to `FullRefill` regardless of what garbage the
/// cache might hold underneath, which would silently mask this bug) --
/// it inspects the reuse *plan* the corrupted-vs-fixed cache entry
/// produces for a follow-up prompt that extends past exactly the
/// accepted tokens.
#[test]
fn cross_turn_cache_and_cancel_on_token_rejection_excludes_rejected_token_from_reuse_plan() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let prompt = "ab".to_string();
    let prompt_len = tokenizer.tokenize(&prompt).real_length;
    let gen_cfg = cross_turn_test_gen_cfg(41, 3);

    // Deterministic (temperature 0, top_k 1): an independent, uncached
    // run of the same prompt/config tells us exactly which token the
    // decode loop's first iteration will reject below, without
    // hardcoding a sampled char that could drift with the fixture.
    let mut reference_state =
        MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let reference = reference_state
        .generate_streaming(&prompt, &tokenizer, &gen_cfg, |_, _| true)
        .expect("reference generate_streaming must not error");
    assert!(
        reference.token_ids.len() >= 2,
        "fixture must generate at least 2 tokens for this test to reach the \
                 mid-decode on_token rejection branch"
    );
    let accepted_id = reference.token_ids[0];
    let rejected_id = reference.token_ids[1];
    let accepted_text = decode_tokens(&tokenizer, &[accepted_id]);
    let rejected_text = decode_tokens(&tokenizer, &[rejected_id]);

    // A follow-up suffix char guaranteed to diverge from whatever the
    // (buggy or fixed) cache entry actually holds at the rejected
    // token's position.
    let diverging_char = ('a'..='z')
        .chain('A'..='Z')
        .map(|c| c.to_string())
        .find(|c| c != &rejected_text)
        .expect("single_char_vocab_tokenizer has 52 letters, at least one differs");

    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let call_count = std::cell::Cell::new(0u32);
    let turn1 = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            &prompt,
            &tokenizer,
            &gen_cfg,
            |_delta, _id| {
                let n = call_count.get() + 1;
                call_count.set(n);
                // Accept only the first generated token (the
                // pre-decode-loop prefill sample); reject the second,
                // which is produced inside the decode loop -- the
                // rejected-token case the cache-boundary logic below accounts for.
                n < 2
            },
            || false,
        )
        .expect("on_token rejection must not surface as an engine error");
    assert_eq!(
        turn1.output.stop_reason,
        Some(StopReason::Interrupt),
        "on_token returning false must stop with Interrupt"
    );
    assert_eq!(
        turn1.output.token_ids.first().copied(),
        Some(accepted_id),
        "the accepted token must match the deterministic reference"
    );

    // The client only ever received `accepted_text`; it never saw
    // `rejected_text`. Its own next request extends the conversation
    // from what it actually has, plus new content of its own.
    let followup_prompt = format!("{prompt}{accepted_text}{diverging_char}");
    let followup = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            &followup_prompt,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || false,
        )
        .expect("follow-up turn must not error");

    // The reuse plan's own reported boundary is the direct check:
    // it must reuse through exactly `prompt_len + 1` tokens (the
    // prompt plus the one token the client actually received) and
    // never through `prompt_len + 2` (prompt plus the rejected
    // token). Without excluding the rejected token, the persisted entry's
    // `represented_len` is `prompt_len + 2`, its trailing token is
    // `rejected_id` (not `diverging_char`'s id), so the shared
    // prefix against `followup_prompt`'s tokens stops at
    // `prompt_len + 1` -- short of the (wrong) `represented_len` --
    // and the planner falls back to `FullRefill` instead, with
    // `reused_tokens == 0`.
    assert_eq!(
        followup.cache.mode,
        crate::kv_cache::PrefixReuseMode::ExactAppend,
        "a follow-up that extends exactly past the tokens the client actually \
                 received must reuse via ExactAppend; FullRefill here means the \
                 persisted cache entry still includes the token on_token rejected"
    );
    assert_eq!(
        followup.cache.reused_tokens,
        prompt_len + 1,
        "the reused boundary must cover the prompt plus exactly the one \
                 accepted token -- not the rejected token"
    );

    // Output parity: the follow-up's result must match an independent
    // full re-prefill of the same follow-up prompt.
    let mut parity_state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let parity_reference = parity_state
        .generate_streaming(&followup_prompt, &tokenizer, &gen_cfg, |_, _| true)
        .expect("parity reference generate_streaming must not error");
    assert_eq!(
        followup.output.token_ids, parity_reference.token_ids,
        "ExactAppend reuse must reproduce exactly what a full re-prefill of the \
                 follow-up prompt would produce"
    );
}

/// A caller rejecting the natural-end tail flush must end the prefix-cache
/// path with `Interrupt`, like a mid-decode rejection, and must leave no
/// cross-turn entry: the rejected text belongs to generated tokens, so no
/// saved boundary may include them.
#[test]
fn cross_turn_cache_rejected_tail_flush_reports_interrupt_and_saves_no_entry() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _gpu_guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;

    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let prompt = "ab".to_string();
    let mut gen_cfg = cross_turn_test_gen_cfg(41, 3);
    // 'Z' is outside single_char_vocab_tokenizer's alphabet, so this never
    // matches; it only makes the stop matcher hold back one byte, which
    // reaches the caller through the tail flush alone.
    gen_cfg.stop_strings = vec!["ZZ".to_string()];

    // Control: the same request with every emit accepted runs to the cap and
    // saves an entry for the slot.
    let mut control_state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let mut control_emits: Vec<String> = Vec::new();
    let control = control_state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            &prompt,
            &tokenizer,
            &gen_cfg,
            |delta, _id| {
                control_emits.push(delta.to_string());
                true
            },
            || false,
        )
        .expect("control run must not error");
    assert_eq!(control.output.stop_reason, Some(StopReason::Length));
    assert_eq!(control.output.token_ids.len(), 3);
    assert_eq!(
        control_emits.len(),
        3,
        "fixture shape: two decode-loop emits plus one tail flush, got {control_emits:?}"
    );
    assert!(
        control_state.cross_turn_prefix_cache.get(slot_id).is_some(),
        "control: an accepted run must save a cross-turn entry, or the assertion \
                 below proves nothing"
    );

    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let mut calls = 0usize;
    let turn = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            &prompt,
            &tokenizer,
            &gen_cfg,
            |_delta, _id| {
                calls += 1;
                calls < control_emits.len()
            },
            || false,
        )
        .expect("a rejected tail flush must not surface as an engine error");
    assert_eq!(
        calls,
        control_emits.len(),
        "the only rejected call must be the final one, the tail flush"
    );
    assert_eq!(
        turn.output.token_ids, control.output.token_ids,
        "the rejection lands after decoding ends, so the same tokens were generated"
    );
    assert_eq!(
        turn.output.stop_reason,
        Some(StopReason::Interrupt),
        "a rejected tail flush must stop with Interrupt"
    );
    assert!(
        !turn.output.stopped,
        "Interrupt always reports stopped: false"
    );
    assert!(
        state.cross_turn_prefix_cache.get(slot_id).is_none(),
        "a rejected tail flush must not persist a boundary covering tokens whose \
                 text the caller never received"
    );
}
