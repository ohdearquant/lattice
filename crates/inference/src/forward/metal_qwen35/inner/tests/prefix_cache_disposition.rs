//! What each terminal path of `generate_streaming_with_prefix_cache_and_cancel`
//! leaves in the cross-turn slot. A request either empties the slot or saves a
//! boundary that live KV and GDN state represent exactly, and every test names
//! which one it expects and checks the saved boundary against live state.

use super::*;
use crate::kv_cache::{CrossTurnSlotId, PrefixReuseMode};

fn metal_device_present() -> bool {
    let present = Device::system_default().is_some();
    assert!(
        present || std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_none(),
        "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
    );
    present
}

/// The token ids of the boundary a finished request left in `slot_id`, or
/// `None` for an empty slot. A saved entry must agree with live state: its
/// length is the represented length, its GDN snapshot sits at that same
/// length, and the live KV cursor is exactly there.
fn saved_boundary(state: &MetalQwen35State, slot_id: CrossTurnSlotId) -> Option<Vec<u32>> {
    let entry = state.cross_turn_prefix_cache.get(slot_id)?;
    let represented_len = entry.generic.represented_len;
    assert_eq!(
        entry.generic.token_ids.len(),
        represented_len,
        "a saved entry's token ids must cover exactly its represented length"
    );
    assert_eq!(
        entry.generic.gdn_snapshot_len, represented_len,
        "a saved entry's GDN snapshot must sit at its represented length"
    );
    assert_eq!(
        state.session.kv_cache.seq_len, represented_len,
        "live KV must sit exactly at a saved entry's represented length"
    );
    Some(entry.generic.token_ids.clone())
}

#[test]
fn fresh_request_that_hits_the_cap_saves_the_prompt_and_every_generated_token() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;

    let turn = state
        .generate_streaming_with_prefix_cache(
            slot_id,
            "ab",
            &tokenizer,
            &cross_turn_test_gen_cfg(7, 3),
            |_, _| true,
        )
        .expect("a fresh request must not error");
    assert_eq!(turn.cache.mode, PrefixReuseMode::FullRefill);
    assert_eq!(turn.cache.reused_tokens, 0);
    assert_eq!(turn.output.stop_reason, Some(StopReason::Length));
    assert_eq!(turn.output.token_ids.len(), 3);

    let prompt_len = turn.cache.prompt_tokens;
    let boundary = saved_boundary(&state, slot_id)
        .expect("a request that ran to its cap must save its boundary");
    assert_eq!(
        boundary.len(),
        prompt_len + 3,
        "the silent final step must bring the last generated token into the boundary"
    );
    assert_eq!(&boundary[prompt_len..], turn.output.token_ids.as_slice());
}

#[test]
fn exact_append_request_that_hits_the_cap_saves_the_extended_boundary() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;

    let turn1 = state
        .generate_streaming_with_prefix_cache(
            slot_id,
            "ab",
            &tokenizer,
            &cross_turn_test_gen_cfg(7, 3),
            |_, _| true,
        )
        .expect("turn 1 must not error");
    let first_boundary_len = turn1.cache.prompt_tokens + 3;

    let prompt2 = format!("ab{}q", turn1.output.text);
    let turn2 = state
        .generate_streaming_with_prefix_cache(
            slot_id,
            &prompt2,
            &tokenizer,
            &cross_turn_test_gen_cfg(8, 2),
            |_, _| true,
        )
        .expect("turn 2 must not error");
    assert_eq!(turn2.cache.mode, PrefixReuseMode::ExactAppend);
    assert_eq!(turn2.cache.reused_tokens, first_boundary_len);
    assert_eq!(turn2.cache.prefetched_tokens, 1);
    assert_eq!(turn2.output.stop_reason, Some(StopReason::Length));
    assert_eq!(turn2.output.token_ids.len(), 2);

    let prompt_len = turn2.cache.prompt_tokens;
    let boundary = saved_boundary(&state, slot_id)
        .expect("an exact-append request that ran to its cap must save its boundary");
    assert_eq!(boundary.len(), prompt_len + 2);
    assert_eq!(&boundary[prompt_len..], turn2.output.token_ids.as_slice());
}

#[test]
fn request_over_a_mismatching_entry_replaces_it_with_a_fresh_boundary() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;

    state
        .generate_streaming_with_prefix_cache(
            slot_id,
            "ab",
            &tokenizer,
            &cross_turn_test_gen_cfg(7, 3),
            |_, _| true,
        )
        .expect("turn 1 must not error");
    let first_boundary = saved_boundary(&state, slot_id).expect("precondition: a warm slot");

    let turn2 = state
        .generate_streaming_with_prefix_cache(
            slot_id,
            "xyz",
            &tokenizer,
            &cross_turn_test_gen_cfg(9, 2),
            |_, _| true,
        )
        .expect("a mismatching request must not error");
    assert_eq!(turn2.cache.mode, PrefixReuseMode::FullRefill);
    assert_eq!(turn2.cache.reused_tokens, 0);

    let prompt_len = turn2.cache.prompt_tokens;
    let boundary = saved_boundary(&state, slot_id)
        .expect("a request that replaced the entry must save its own boundary");
    assert_eq!(boundary.len(), prompt_len + 2);
    assert_ne!(
        boundary[0], first_boundary[0],
        "the saved boundary must belong to the new prompt, not the invalidated entry"
    );
    assert_eq!(&boundary[prompt_len..], turn2.output.token_ids.as_slice());
}

#[test]
fn cancel_before_prefill_over_an_exact_append_plan_leaves_the_slot_empty() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;
    let gen_cfg = cross_turn_test_gen_cfg(7, 3);

    let turn1 = state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &gen_cfg, |_, _| true)
        .expect("turn 1 must not error");
    assert!(
        saved_boundary(&state, slot_id).is_some(),
        "precondition: turn 1 leaves a warm slot"
    );

    let prompt2 = format!("ab{}q", turn1.output.text);
    let turn2 = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            &prompt2,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || true,
        )
        .expect("a cancel before prefill must not surface as an engine error");
    assert_eq!(turn2.output.stop_reason, Some(StopReason::Interrupt));
    assert_eq!(turn2.output.generated_tokens, 0);
    assert!(!turn2.output.stopped);
    assert_eq!(turn2.cache.mode, PrefixReuseMode::ExactAppend);
    assert_eq!(turn2.cache.reused_tokens, 0);
    assert_eq!(turn2.cache.prefetched_tokens, 0);
    assert!(
        saved_boundary(&state, slot_id).is_none(),
        "a cancel before prefill must leave the slot empty, not re-save the consumed entry"
    );
}

#[test]
fn cancel_before_prefill_over_a_mismatching_entry_leaves_the_slot_empty() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;
    let gen_cfg = cross_turn_test_gen_cfg(7, 3);

    state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &gen_cfg, |_, _| true)
        .expect("turn 1 must not error");
    assert!(
        saved_boundary(&state, slot_id).is_some(),
        "precondition: turn 1 leaves a warm slot"
    );

    let turn2 = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            "xyz",
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || true,
        )
        .expect("a cancel before prefill must not surface as an engine error");
    assert_eq!(turn2.output.stop_reason, Some(StopReason::Interrupt));
    assert_eq!(turn2.output.generated_tokens, 0);
    assert_eq!(turn2.cache.mode, PrefixReuseMode::FullRefill);
    assert!(
        saved_boundary(&state, slot_id).is_none(),
        "a full refill discards the warm entry, and a cancel before prefill must not save"
    );
}

#[test]
fn cancel_after_prefill_leaves_the_slot_empty() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;
    let gen_cfg = cross_turn_test_gen_cfg(7, 3);

    let turn1 = state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &gen_cfg, |_, _| true)
        .expect("turn 1 must not error");
    let first_boundary_len = turn1.cache.prompt_tokens + 3;

    let prompt2 = format!("ab{}q", turn1.output.text);
    let poll_count = std::cell::Cell::new(0u32);
    let turn2 = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            &prompt2,
            &tokenizer,
            &gen_cfg,
            |_, _| true,
            || {
                let n = poll_count.get() + 1;
                poll_count.set(n);
                n >= 2
            },
        )
        .expect("a cancel after prefill must not surface as an engine error");
    assert_eq!(
        poll_count.get(),
        2,
        "the cancel must have been observed at the checkpoint right after prefill"
    );
    assert_eq!(turn2.output.stop_reason, Some(StopReason::Interrupt));
    assert_eq!(turn2.output.generated_tokens, 0);
    assert!(!turn2.output.stopped);
    assert_eq!(turn2.cache.mode, PrefixReuseMode::ExactAppend);
    assert_eq!(turn2.cache.reused_tokens, first_boundary_len);
    assert_eq!(turn2.cache.prefetched_tokens, 1);
    assert!(
        saved_boundary(&state, slot_id).is_none(),
        "a cancel after prefill must leave the slot empty"
    );
}

#[test]
fn caller_rejecting_the_first_token_leaves_the_slot_empty() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;
    let gen_cfg = cross_turn_test_gen_cfg(7, 3);

    let turn1 = state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &gen_cfg, |_, _| true)
        .expect("turn 1 must not error");
    assert!(
        saved_boundary(&state, slot_id).is_some(),
        "precondition: turn 1 leaves a warm slot"
    );

    let prompt2 = format!("ab{}q", turn1.output.text);
    let turn2 = state
        .generate_streaming_with_prefix_cache(slot_id, &prompt2, &tokenizer, &gen_cfg, |_, _| false)
        .expect("a rejected first token must not surface as an engine error");
    assert_eq!(turn2.output.stop_reason, Some(StopReason::Interrupt));
    assert_eq!(turn2.output.generated_tokens, 1);
    assert!(!turn2.output.stopped);
    assert_eq!(turn2.cache.mode, PrefixReuseMode::ExactAppend);
    assert!(
        saved_boundary(&state, slot_id).is_none(),
        "only the prefill-sampled token was produced and it was never forwarded; \
         nothing may be saved"
    );
}

#[test]
fn cancel_at_the_top_of_a_decode_iteration_saves_the_forwarded_boundary() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;

    // Polls 1 and 2 are the two checkpoints around prefill, poll 3 is the top of
    // the first decode iteration and poll 4 the top of the second, where the
    // cancel lands: two tokens were pushed, and only the first was forwarded.
    let poll_count = std::cell::Cell::new(0u32);
    let turn = state
        .generate_streaming_with_prefix_cache_and_cancel(
            slot_id,
            "ab",
            &tokenizer,
            &cross_turn_test_gen_cfg(17, 4),
            |_, _| true,
            || {
                let n = poll_count.get() + 1;
                poll_count.set(n);
                n >= 4
            },
        )
        .expect("a cancel at a decode boundary must not surface as an engine error");
    assert_eq!(turn.output.stop_reason, Some(StopReason::Interrupt));
    assert!(!turn.output.stopped);
    assert_eq!(turn.output.token_ids.len(), 2);

    let prompt_len = turn.cache.prompt_tokens;
    let boundary = saved_boundary(&state, slot_id)
        .expect("a cancel at the top of a decode iteration must save the forwarded boundary");
    assert_eq!(
        boundary.len(),
        prompt_len + 1,
        "the boundary covers the prompt and the one forwarded token, not the pushed one"
    );
    assert_eq!(&boundary[prompt_len..], &turn.output.token_ids[..1]);
}

#[test]
fn caller_rejecting_a_decode_token_saves_the_boundary_before_it() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;

    // The first two tokens are accepted and the third, sampled in the second
    // decode iteration, is rejected. Both accepted tokens were forwarded by
    // then, and the rejected one was not.
    let delivered = std::cell::Cell::new(0u32);
    let turn = state
        .generate_streaming_with_prefix_cache(
            slot_id,
            "ab",
            &tokenizer,
            &cross_turn_test_gen_cfg(41, 4),
            |_, _| {
                let n = delivered.get() + 1;
                delivered.set(n);
                n < 3
            },
        )
        .expect("a rejected decode token must not surface as an engine error");
    assert_eq!(turn.output.stop_reason, Some(StopReason::Interrupt));
    assert!(!turn.output.stopped);
    assert_eq!(turn.output.token_ids.len(), 3);

    let prompt_len = turn.cache.prompt_tokens;
    let boundary = saved_boundary(&state, slot_id)
        .expect("a rejected decode token must still save the forwarded boundary");
    assert_eq!(
        boundary.len(),
        prompt_len + 2,
        "the boundary must cover the prompt and both forwarded tokens, never the rejected one"
    );
    assert_eq!(&boundary[prompt_len..], &turn.output.token_ids[..2]);
}

#[test]
fn stop_token_on_the_first_sample_saves_the_prompt_only_boundary() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;
    // The grammar forces the first sample to be id 0, which is also a stop token.
    let mut gen_cfg = single_char_grammar_cfg("root ::= \"a\" \"a\"\n", 4);
    gen_cfg.stop_token_ids = vec![0];

    let turn = state
        .generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true)
        .expect("a stop token on the first sample must not error");
    assert!(turn.output.stopped);
    assert_eq!(turn.output.stop_reason, Some(StopReason::Eos));
    assert!(turn.output.token_ids.is_empty());

    let prompt_len = turn.cache.prompt_tokens;
    let boundary = saved_boundary(&state, slot_id)
        .expect("a stop on the first sample must save the prompt boundary");
    assert_eq!(boundary.len(), prompt_len);
}

#[test]
fn stop_token_inside_the_decode_loop_saves_every_generated_token() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;
    // The grammar forces "a", "b", "c" in order and id 2 ("c") is a stop token, so the
    // loop stops on its second iteration with both earlier tokens already forwarded.
    let mut gen_cfg = single_char_grammar_cfg("root ::= \"a\" \"b\" \"c\"\n", 5);
    gen_cfg.stop_token_ids = vec![2];

    let turn = state
        .generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true)
        .expect("a stop token inside the loop must not error");
    assert!(turn.output.stopped);
    assert_eq!(turn.output.stop_reason, Some(StopReason::Eos));
    assert_eq!(turn.output.token_ids, vec![0, 1]);
    assert_eq!(turn.output.text, "ab");

    let prompt_len = turn.cache.prompt_tokens;
    let boundary = saved_boundary(&state, slot_id)
        .expect("a stop token inside the loop must save the forwarded boundary");
    assert_eq!(
        boundary.len(),
        prompt_len + 2,
        "every generated token was forwarded before the stop token was sampled"
    );
    assert_eq!(&boundary[prompt_len..], [0, 1]);
}

#[test]
fn grammar_completed_by_the_first_sample_saves_the_prompt_only_boundary() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;
    let gen_cfg = single_char_grammar_cfg("root ::= \"a\"\n", 4);

    let turn = state
        .generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true)
        .expect("a grammar completed by the first sample must not error");
    assert!(turn.output.stopped);
    assert_eq!(turn.output.stop_reason, Some(StopReason::Grammar));
    assert_eq!(turn.output.token_ids, vec![0]);

    let prompt_len = turn.cache.prompt_tokens;
    let boundary = saved_boundary(&state, slot_id)
        .expect("a grammar completed by the first sample must save a boundary");
    assert_eq!(
        boundary.len(),
        prompt_len,
        "the completing token was never forwarded, so it stays out of the boundary"
    );
}

#[test]
fn grammar_completed_inside_the_decode_loop_saves_the_boundary_one_token_short() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;
    let gen_cfg = single_char_grammar_cfg("root ::= \"a\" \"a\"\n", 4);

    let turn = state
        .generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true)
        .expect("a grammar completed inside the loop must not error");
    assert!(turn.output.stopped);
    assert_eq!(turn.output.stop_reason, Some(StopReason::Grammar));
    assert_eq!(turn.output.token_ids, vec![0, 0]);
    assert_eq!(turn.output.text, "aa");

    let prompt_len = turn.cache.prompt_tokens;
    let boundary = saved_boundary(&state, slot_id)
        .expect("a grammar completed inside the loop must save a boundary");
    assert_eq!(
        boundary.len(),
        prompt_len + 1,
        "the completing token was never forwarded, so the boundary stops one token short"
    );
    assert_eq!(&boundary[prompt_len..], [0]);
}

#[test]
fn stop_string_on_the_first_sample_leaves_the_slot_empty() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;
    // The grammar forces the first sample to "a", which the stop string matches.
    let mut gen_cfg = single_char_grammar_cfg("root ::= \"a\" \"a\"\n", 4);
    gen_cfg.stop_strings = vec!["a".to_string()];

    let turn = state
        .generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true)
        .expect("a stop string on the first sample must not error");
    assert!(turn.output.stopped);
    assert_eq!(turn.output.stop_reason, Some(StopReason::Eos));
    assert_eq!(turn.output.text, "");
    assert!(
        saved_boundary(&state, slot_id).is_none(),
        "tokens behind a stop-string match must never be saved"
    );
}

#[test]
fn stop_string_inside_the_decode_loop_leaves_the_slot_empty() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;
    // The grammar forces "a", "b", "c" in order and the stop string matches the
    // second token, which is produced inside the decode loop.
    let mut gen_cfg = single_char_grammar_cfg("root ::= \"a\" \"b\" \"c\"\n", 5);
    gen_cfg.stop_strings = vec!["b".to_string()];

    let turn = state
        .generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true)
        .expect("a stop string inside the loop must not error");
    assert!(turn.output.stopped);
    assert_eq!(turn.output.stop_reason, Some(StopReason::Eos));
    assert_eq!(turn.output.text, "a");
    assert!(
        saved_boundary(&state, slot_id).is_none(),
        "tokens behind a stop-string match must never be saved"
    );
}

#[test]
fn zero_token_budget_leaves_the_warm_slot_untouched() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;

    state
        .generate_streaming_with_prefix_cache(
            slot_id,
            "ab",
            &tokenizer,
            &cross_turn_test_gen_cfg(7, 3),
            |_, _| true,
        )
        .expect("turn 1 must not error");
    let warm_boundary = saved_boundary(&state, slot_id).expect("precondition: a warm slot");

    let turn2 = state
        .generate_streaming_with_prefix_cache(
            slot_id,
            "xyz",
            &tokenizer,
            &cross_turn_test_gen_cfg(1, 0),
            |_, _| true,
        )
        .expect("a zero-budget request must not error");
    assert_eq!(turn2.output.generated_tokens, 0);
    assert_eq!(turn2.cache.mode, PrefixReuseMode::FullRefill);
    assert_eq!(turn2.cache.reused_tokens, 0);
    assert_eq!(
        saved_boundary(&state, slot_id),
        Some(warm_boundary),
        "a zero-budget request returns before touching any state, warm entry included"
    );
}
