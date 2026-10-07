//! What the cross-turn slot holds after each of the streaming exits the tests
//! below drive; it is not a map of every return in
//! `generate_streaming_with_prefix_cache_and_cancel`. A request either empties
//! the slot or saves a boundary, and every test names which one it expects.
//! A saved boundary is compared against an independently computed token
//! sequence (prompt ids written out from the fixture vocabulary, generated ids
//! fixed by a grammar or taken from a no-cache reference run), never against
//! the turn's own output, and its lengths are checked against the live KV
//! cursor. The checks do not read KV or GDN contents.

use super::*;
use crate::kv_cache::{CrossTurnSlotId, PrefixReuseMode};

pub(super) fn metal_device_present() -> bool {
    let present = Device::system_default().is_some();
    assert!(
        present || std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_none(),
        "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
    );
    present
}

/// The id the fixture vocabulary gives one character: lowercase letters are
/// ids 0..26 and `A`.. continue from 26. Written out here so the expected
/// sequences do not pass through the tokenizer under test.
pub(super) fn id_of(c: char) -> u32 {
    match c {
        'a'..='z' => c as u32 - 'a' as u32,
        'A'..='F' => 26 + c as u32 - 'A' as u32,
        _ => panic!("{c:?} is outside the fixture vocabulary"),
    }
}

pub(super) fn ids_of(text: &str) -> Vec<u32> {
    text.chars().map(id_of).collect()
}

fn text_of(ids: &[u32]) -> String {
    ids.iter()
        .map(|&id| match id {
            0..=25 => char::from(b'a' + id as u8),
            26..=31 => char::from(b'A' + (id - 26) as u8),
            _ => panic!("id {id} is outside the fixture vocabulary"),
        })
        .collect()
}

/// The ids a greedy request generates for `prompt`, from a fresh state that
/// has no cross-turn cache involved: the full re-prefill path that cross-turn
/// reuse must reproduce.
fn reference_generated_ids(
    weights: &ModelWeights,
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    gen_cfg: &GenerateConfig,
) -> Vec<u32> {
    let mut reference = MetalQwen35State::new(weights, cfg, 64).expect("tiny hybrid fixture");
    reference
        .generate_streaming(prompt, tokenizer, gen_cfg, |_, _| true)
        .expect("a reference run without a grammar must not fail")
        .token_ids
}

/// A saved entry must hold exactly `expected` as its token ids, its GDN
/// snapshot must sit at that length, and the live KV cursor must be there too.
fn assert_saved_boundary(state: &MetalQwen35State, slot_id: CrossTurnSlotId, expected: &[u32]) {
    let entry = state
        .cross_turn_prefix_cache
        .get(slot_id)
        .expect("the request must have saved a boundary");
    assert_eq!(
        entry.generic.represented_len,
        expected.len(),
        "a saved entry must represent exactly the expected boundary"
    );
    assert_eq!(
        entry.generic.token_ids.as_slice(),
        expected,
        "a saved entry's token ids must be the expected prompt and generated ids"
    );
    assert_eq!(
        entry.generic.gdn_snapshot_len,
        expected.len(),
        "a saved entry's GDN snapshot must sit at its represented length"
    );
    assert_eq!(
        state.session.kv_cache.seq_len,
        expected.len(),
        "live KV must sit exactly at a saved entry's represented length"
    );
}

fn assert_slot_empty(state: &MetalQwen35State, slot_id: CrossTurnSlotId, why: &str) {
    assert!(
        state.cross_turn_prefix_cache.get(slot_id).is_none(),
        "{why}"
    );
}

/// Runs the first turn of a conversation ("ab", three greedy tokens) and
/// returns the boundary it must have saved: the prompt ids and every generated
/// id, the last of them brought into KV by the silent final step.
fn warm_the_slot_with_a_first_turn(
    state: &mut MetalQwen35State,
    weights: &ModelWeights,
    cfg: &Qwen35Config,
    tokenizer: &BpeTokenizer,
    slot_id: CrossTurnSlotId,
) -> Vec<u32> {
    let gen_cfg = cross_turn_test_gen_cfg(7, 3);
    state
        .generate_streaming_with_prefix_cache(slot_id, "ab", tokenizer, &gen_cfg, |_, _| true)
        .expect("the first turn must not error");
    let mut expected = ids_of("ab");
    expected.extend(reference_generated_ids(
        weights, cfg, tokenizer, "ab", &gen_cfg,
    ));
    assert_eq!(
        expected.len(),
        5,
        "precondition: two prompt ids, three generated"
    );
    assert_saved_boundary(state, slot_id, &expected);
    expected
}

/// The single-character vocabulary with id 30 re-spelled as `</think>`, and a
/// greedy config whose reasoning budget closes on that token, under `gbnf`.
fn thinking_fixture(
    gbnf: &str,
    reasoning_budget: usize,
    max_new_tokens: usize,
) -> (BpeTokenizer, GenerateConfig) {
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::collections::HashMap;
    use std::sync::Arc;

    let mut vocab_bytes = single_char_vocab_bytes();
    vocab_bytes[30] = b"</think>".to_vec();
    let vocab: HashMap<String, u32> = vocab_bytes
        .iter()
        .zip(0u32..)
        .map(|(bytes, id)| (String::from_utf8_lossy(bytes).into_owned(), id))
        .collect();
    let tokenizer = BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
        .expect("thinking vocab tokenizer build");
    let engine = Arc::new(
        GrammarEngine::new(&GrammarSpec::Gbnf(gbnf.to_string()), vocab_bytes)
            .expect("grammar engine builds over the thinking vocab"),
    );
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: true,
        enable_mtp: Some(false),
        grammar: Some(engine),
        stop_strings: vec![],
        reasoning_budget: Some(reasoning_budget),
        logprobs: None,
    };
    (tokenizer, gen_cfg)
}

/// The single-character vocabulary with id 1 decoding to the lone byte 0xE4,
/// which the detokenizer holds until the final flush renders it as U+FFFD,
/// though a grammar reads it as `b`; and id 30 spelled `</think>`. Returns the
/// tokenizer and the bytes the grammar engine reads each id as.
pub(super) fn lone_byte_vocabulary() -> (BpeTokenizer, Vec<Vec<u8>>) {
    use std::collections::HashMap;

    let mut grammar_bytes = single_char_vocab_bytes();
    grammar_bytes[30] = b"</think>".to_vec();
    let mut vocab: HashMap<String, u32> = grammar_bytes
        .iter()
        .zip(0u32..)
        .map(|(bytes, id)| (String::from_utf8_lossy(bytes).into_owned(), id))
        .collect();
    vocab.remove("b");
    vocab.insert("\u{e4}".to_string(), 1);
    let tokenizer =
        BpeTokenizer::from_vocab_and_merges(vocab, Vec::new()).expect("lone-byte tokenizer");
    (tokenizer, grammar_bytes)
}

/// A greedy request over the lone-byte vocabulary whose grammar fixes the
/// generated ids. A reasoning budget turns thinking on.
fn lone_byte_cfg(
    gbnf: &str,
    max_new_tokens: usize,
    stop_token_ids: &[u32],
    stop_strings: &[&str],
    reasoning_budget: Option<usize>,
) -> GenerateConfig {
    use crate::grammar::{GrammarEngine, GrammarSpec};

    let (_, grammar_bytes) = lone_byte_vocabulary();
    let engine = GrammarEngine::new(&GrammarSpec::Gbnf(gbnf.to_string()), grammar_bytes)
        .expect("grammar engine builds over the lone-byte vocabulary");
    GenerateConfig {
        grammar: Some(std::sync::Arc::new(engine)),
        stop_token_ids: stop_token_ids.to_vec(),
        stop_strings: stop_strings.iter().map(ToString::to_string).collect(),
        enable_thinking: reasoning_budget.is_some(),
        reasoning_budget,
        ..cross_turn_test_gen_cfg(1, max_new_tokens)
    }
}

/// Runs one request with the prompt "ac" over the lone-byte vocabulary on a
/// fresh state, and returns the state with the request's output.
fn run_lone_byte_request(
    gen_cfg: &GenerateConfig,
) -> (MetalQwen35State, crate::generation::GenerateOutput) {
    let (tokenizer, _) = lone_byte_vocabulary();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let turn = state
        .generate_streaming_with_prefix_cache(
            CrossTurnSlotId::DEFAULT,
            "ac",
            &tokenizer,
            gen_cfg,
            |_, _| true,
        )
        .expect("a lone-byte request must not error");
    (state, turn.output)
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
    let gen_cfg = cross_turn_test_gen_cfg(7, 3);

    let turn = state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &gen_cfg, |_, _| true)
        .expect("a fresh request must not error");
    assert_eq!(turn.cache.mode, PrefixReuseMode::FullRefill);
    assert_eq!(turn.cache.reused_tokens, 0);
    assert_eq!(turn.output.stop_reason, Some(StopReason::Length));
    assert_eq!(turn.output.token_ids.len(), 3);

    // The silent final step brings the last generated token into KV, so the
    // boundary is the prompt and all three generated ids.
    let mut expected = ids_of("ab");
    expected.extend(reference_generated_ids(
        &weights, &cfg, &tokenizer, "ab", &gen_cfg,
    ));
    assert_eq!(expected.len(), 5);
    assert_saved_boundary(&state, slot_id, &expected);
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

    let first_boundary =
        warm_the_slot_with_a_first_turn(&mut state, &weights, &cfg, &tokenizer, slot_id);

    let prompt2 = format!("{}q", text_of(&first_boundary));
    let gen_cfg2 = cross_turn_test_gen_cfg(8, 2);
    let turn2 = state
        .generate_streaming_with_prefix_cache(slot_id, &prompt2, &tokenizer, &gen_cfg2, |_, _| true)
        .expect("turn 2 must not error");
    assert_eq!(turn2.cache.mode, PrefixReuseMode::ExactAppend);
    assert_eq!(turn2.cache.reused_tokens, first_boundary.len());
    assert_eq!(turn2.cache.prefetched_tokens, 1);
    assert_eq!(turn2.output.stop_reason, Some(StopReason::Length));
    assert_eq!(turn2.output.token_ids.len(), 2);

    let mut expected = ids_of(&prompt2);
    expected.extend(reference_generated_ids(
        &weights, &cfg, &tokenizer, &prompt2, &gen_cfg2,
    ));
    assert_eq!(expected.len(), first_boundary.len() + 1 + 2);
    assert_saved_boundary(&state, slot_id, &expected);
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

    warm_the_slot_with_a_first_turn(&mut state, &weights, &cfg, &tokenizer, slot_id);

    let gen_cfg2 = cross_turn_test_gen_cfg(9, 2);
    let turn2 = state
        .generate_streaming_with_prefix_cache(slot_id, "xyz", &tokenizer, &gen_cfg2, |_, _| true)
        .expect("a mismatching request must not error");
    assert_eq!(turn2.cache.mode, PrefixReuseMode::FullRefill);
    assert_eq!(turn2.cache.reused_tokens, 0);

    // The saved boundary belongs to the new prompt: its ids start with "xyz",
    // which no id of the invalidated entry's prompt ("ab") can satisfy.
    let mut expected = ids_of("xyz");
    expected.extend(reference_generated_ids(
        &weights, &cfg, &tokenizer, "xyz", &gen_cfg2,
    ));
    assert_eq!(expected.len(), 5);
    assert_saved_boundary(&state, slot_id, &expected);
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

    let first_boundary =
        warm_the_slot_with_a_first_turn(&mut state, &weights, &cfg, &tokenizer, slot_id);

    let prompt2 = format!("{}q", text_of(&first_boundary));
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
    assert_slot_empty(
        &state,
        slot_id,
        "a cancel before prefill must leave the slot empty, not re-save the consumed entry",
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

    warm_the_slot_with_a_first_turn(&mut state, &weights, &cfg, &tokenizer, slot_id);

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
    assert_slot_empty(
        &state,
        slot_id,
        "a full refill discards the warm entry, and a cancel before prefill must not save",
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

    let first_boundary =
        warm_the_slot_with_a_first_turn(&mut state, &weights, &cfg, &tokenizer, slot_id);

    let prompt2 = format!("{}q", text_of(&first_boundary));
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
    assert_eq!(turn2.cache.reused_tokens, first_boundary.len());
    assert_eq!(turn2.cache.prefetched_tokens, 1);
    assert_slot_empty(
        &state,
        slot_id,
        "a cancel after prefill must leave the slot empty",
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

    let first_boundary =
        warm_the_slot_with_a_first_turn(&mut state, &weights, &cfg, &tokenizer, slot_id);

    let prompt2 = format!("{}q", text_of(&first_boundary));
    let turn2 = state
        .generate_streaming_with_prefix_cache(slot_id, &prompt2, &tokenizer, &gen_cfg, |_, _| false)
        .expect("a rejected first token must not surface as an engine error");
    assert_eq!(turn2.output.stop_reason, Some(StopReason::Interrupt));
    assert_eq!(turn2.output.generated_tokens, 1);
    assert!(!turn2.output.stopped);
    assert_eq!(turn2.cache.mode, PrefixReuseMode::ExactAppend);
    assert_slot_empty(
        &state,
        slot_id,
        "only the prefill-sampled token was produced and it was never forwarded; \
         nothing may be saved",
    );
}

#[test]
fn cancel_at_the_top_of_the_first_decode_iteration_saves_the_prompt_only_boundary() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;

    // Polls 1 and 2 are the two checkpoints around prefill and poll 3 is the top
    // of the first decode iteration, where the cancel lands: the prefill-derived
    // token is pushed but its forward step is the work this iteration skips, so
    // live KV holds the prompt and nothing else.
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
                n >= 3
            },
        )
        .expect("a cancel at a decode boundary must not surface as an engine error");
    assert_eq!(
        poll_count.get(),
        3,
        "the cancel must have been observed at the top of the first decode iteration"
    );
    assert_eq!(turn.output.stop_reason, Some(StopReason::Interrupt));
    assert!(!turn.output.stopped);
    assert_eq!(turn.output.token_ids.len(), 1);

    assert_saved_boundary(&state, slot_id, &ids_of("ab"));
}

#[test]
fn cancel_at_the_top_of_a_later_decode_iteration_saves_the_forwarded_boundary() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;
    let gen_cfg = cross_turn_test_gen_cfg(17, 4);

    // Polls 1 and 2 are the two checkpoints around prefill, poll 3 is the top of
    // the first decode iteration and poll 4 the top of the second, where the
    // cancel lands: two tokens were pushed, and only the first was forwarded.
    let poll_count = std::cell::Cell::new(0u32);
    let turn = state
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
        .expect("a cancel at a decode boundary must not surface as an engine error");
    assert_eq!(turn.output.stop_reason, Some(StopReason::Interrupt));
    assert!(!turn.output.stopped);
    assert_eq!(turn.output.token_ids.len(), 2);

    // The boundary covers the prompt and the one forwarded token, not the pushed one.
    let generated = reference_generated_ids(&weights, &cfg, &tokenizer, "ab", &gen_cfg);
    let mut expected = ids_of("ab");
    expected.push(generated[0]);
    assert_saved_boundary(&state, slot_id, &expected);
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
    let gen_cfg = cross_turn_test_gen_cfg(41, 4);

    // The first two tokens are accepted and the third, sampled in the second
    // decode iteration, is rejected. Both accepted tokens were forwarded by
    // then, and the rejected one was not.
    let delivered = std::cell::Cell::new(0u32);
    let turn = state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &gen_cfg, |_, _| {
            let n = delivered.get() + 1;
            delivered.set(n);
            n < 3
        })
        .expect("a rejected decode token must not surface as an engine error");
    assert_eq!(turn.output.stop_reason, Some(StopReason::Interrupt));
    assert!(!turn.output.stopped);
    assert_eq!(turn.output.token_ids.len(), 3);

    // The boundary covers the prompt and both forwarded tokens, never the rejected one.
    let generated = reference_generated_ids(&weights, &cfg, &tokenizer, "ab", &gen_cfg);
    let mut expected = ids_of("ab");
    expected.extend_from_slice(&generated[..2]);
    assert_saved_boundary(&state, slot_id, &expected);
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

    assert_saved_boundary(&state, slot_id, &ids_of("a"));
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

    // Every generated token was forwarded before the stop token was sampled.
    assert_saved_boundary(&state, slot_id, &[0, 0, 1]);
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

    // The completing token was never forwarded, so it stays out of the boundary.
    assert_saved_boundary(&state, slot_id, &[0]);
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

    // The completing token was never forwarded, so the boundary stops one token short.
    assert_saved_boundary(&state, slot_id, &[0, 0]);
}

#[test]
fn grammar_rejecting_a_budget_forced_close_saves_the_boundary_without_forwarding_twice() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let (tokenizer, gen_cfg) = thinking_fixture("root ::= \"a\" \"a\"\n", 1, 4);
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;

    // The grammar forces "a" (id 0) as the first sample. With a reasoning budget
    // of one, the first decode iteration forwards that token and then overrides
    // the sampled token with `</think>`, which the grammar rejects before it is
    // pushed. The saved boundary is the prompt and the one forwarded token.
    let turn = state
        .generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true)
        .expect("a budget-forced token the grammar rejects is a stop, not an error");
    assert_eq!(turn.output.stop_reason, Some(StopReason::Grammar));
    assert!(
        !turn.output.stopped,
        "a grammar rejection reports stopped: false, as the streaming entry does"
    );
    assert_eq!(turn.output.token_ids, vec![0]);

    assert_saved_boundary(&state, slot_id, &[0, 0]);
}

#[test]
fn answer_budget_exhausted_before_the_cap_saves_every_generated_token() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    // The grammar forces `</think>` (id 30) as the first sample and then "a"
    // (id 0) six times, so it never completes inside the cap. With a reasoning
    // budget of 3 and 2 answer tokens the cap is 3 + 2 + 1 = 6 tokens, but the
    // model closed thinking on token 1, so the answer budget runs out after
    // token 3 and the loop breaks there instead of exhausting the cap.
    let (tokenizer, gen_cfg) = thinking_fixture(
        "root ::= \"</think>\" \"a\" \"a\" \"a\" \"a\" \"a\" \"a\"\n",
        3,
        2,
    );
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;

    let turn = state
        .generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true)
        .expect("an exhausted answer budget must not error");
    assert!(!turn.output.stopped);
    assert_eq!(turn.output.stop_reason, Some(StopReason::Length));
    assert_eq!(
        turn.output.token_ids,
        vec![30, 0, 0],
        "the loop must break on the answer budget, three tokens short of the cap"
    );

    // The iteration that pushed the third token forwarded the second, and the
    // silent final step forwards the third: prompt "a" (0) plus all three.
    assert_saved_boundary(&state, slot_id, &[0, 30, 0, 0]);
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
    assert_slot_empty(
        &state,
        slot_id,
        "tokens behind a stop-string match must never be saved",
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
    assert_slot_empty(
        &state,
        slot_id,
        "tokens behind a stop-string match must never be saved",
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

    let warm_boundary =
        warm_the_slot_with_a_first_turn(&mut state, &weights, &cfg, &tokenizer, slot_id);

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
    // A zero-budget request returns before touching any state, warm entry included.
    assert_saved_boundary(&state, slot_id, &warm_boundary);
}

#[test]
fn stop_string_spanning_two_tokens_leaves_the_slot_empty() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;
    // The grammar forces "a", "b", "c" in order and the stop string "bc" is only
    // complete once the third token is pushed.
    let mut gen_cfg = single_char_grammar_cfg("root ::= \"a\" \"b\" \"c\"\n", 5);
    gen_cfg.stop_strings = vec!["bc".to_string()];

    let turn = state
        .generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true)
        .expect("a stop string spanning two tokens must not error");
    assert!(turn.output.stopped);
    assert_eq!(turn.output.stop_reason, Some(StopReason::Eos));
    assert_eq!(turn.output.token_ids, vec![0, 1, 2]);
    assert_eq!(turn.output.text, "a");
    assert_slot_empty(
        &state,
        slot_id,
        "tokens behind a stop-string match must never be saved",
    );
}

#[test]
fn stop_string_held_back_and_never_matched_saves_every_generated_token() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let tokenizer = single_char_vocab_tokenizer();
    let (mut cfg, weights) = tiny_hybrid_fixture();
    cfg.eos_token_id = u32::MAX;
    let mut state = MetalQwen35State::new(&weights, &cfg, 64).expect("tiny hybrid fixture");
    let slot_id = CrossTurnSlotId::DEFAULT;
    // The grammar forces "a" .. "e" and the cap stops the request after "c". The
    // stop string "cde" holds the "c" back, and the final flush releases it
    // without a match.
    let mut gen_cfg = single_char_grammar_cfg("root ::= \"a\" \"b\" \"c\" \"d\" \"e\"\n", 3);
    gen_cfg.stop_strings = vec!["cde".to_string()];

    let turn = state
        .generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true)
        .expect("a held-back stop string must not error");
    assert!(!turn.output.stopped);
    assert_eq!(turn.output.stop_reason, Some(StopReason::Length));
    assert_eq!(turn.output.token_ids, vec![0, 1, 2]);
    assert_eq!(turn.output.text, "abc");

    // The silent final step forwards the last token, so the boundary is the
    // prompt and all three generated ids.
    assert_saved_boundary(&state, slot_id, &[0, 0, 1, 2]);
}

#[test]
fn lone_byte_request_that_hits_the_cap_saves_every_generated_token() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let gen_cfg = lone_byte_cfg("root ::= \"a\" \"b\" \"b\" \"b\"\n", 2, &[], &[], None);

    let (state, output) = run_lone_byte_request(&gen_cfg);
    assert!(!output.stopped);
    assert_eq!(output.stop_reason, Some(StopReason::Length));
    assert_eq!(output.token_ids, vec![0, 1]);
    assert_eq!(output.text, "a\u{fffd}");

    let mut expected = ids_of("ac");
    expected.extend([0, 1]);
    assert_saved_boundary(&state, CrossTurnSlotId::DEFAULT, &expected);
}

#[test]
fn lone_byte_flush_completing_a_stop_string_after_the_cap_leaves_the_slot_empty() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let gen_cfg = lone_byte_cfg(
        "root ::= \"a\" \"b\" \"b\" \"b\"\n",
        2,
        &[],
        &["\u{fffd}"],
        None,
    );

    let (state, output) = run_lone_byte_request(&gen_cfg);
    assert!(output.stopped);
    assert_eq!(output.stop_reason, Some(StopReason::Eos));
    assert_eq!(output.token_ids, vec![0, 1]);
    assert_eq!(
        output.text, "a",
        "the stop string must have been completed by the flush of the held byte"
    );
    assert_slot_empty(
        &state,
        CrossTurnSlotId::DEFAULT,
        "a stop string completed by the final flush must not save the tokens behind it",
    );
}

#[test]
fn lone_byte_stop_token_saves_the_tokens_it_forwarded() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let gen_cfg = lone_byte_cfg("root ::= \"a\" \"b\" \"c\"\n", 5, &[2], &[], None);

    let (state, output) = run_lone_byte_request(&gen_cfg);
    assert!(output.stopped);
    assert_eq!(output.stop_reason, Some(StopReason::Eos));
    assert_eq!(output.token_ids, vec![0, 1]);
    assert_eq!(output.text, "a\u{fffd}");

    let mut expected = ids_of("ac");
    expected.extend([0, 1]);
    assert_saved_boundary(&state, CrossTurnSlotId::DEFAULT, &expected);
}

#[test]
fn lone_byte_stop_token_then_flush_completed_stop_string_leaves_the_slot_empty() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    // The stop token ends the request on an opened prediction that is never
    // pushed, and the stop string is completed only afterwards, by the flush of
    // the byte the detokenizer was holding.
    let gen_cfg = lone_byte_cfg("root ::= \"a\" \"b\" \"c\"\n", 5, &[2], &["\u{fffd}"], None);

    let (state, output) = run_lone_byte_request(&gen_cfg);
    assert!(output.stopped);
    assert_eq!(output.stop_reason, Some(StopReason::Eos));
    assert_eq!(output.token_ids, vec![0, 1]);
    assert_eq!(
        output.text, "a",
        "the stop string must have been completed by the flush of the held byte"
    );
    assert_slot_empty(
        &state,
        CrossTurnSlotId::DEFAULT,
        "a stop string completed by the final flush after a stop token must not save the \
         tokens behind it",
    );
}

#[test]
fn lone_byte_grammar_completed_by_the_first_sample_saves_the_prompt_only_boundary() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let gen_cfg = lone_byte_cfg("root ::= \"b\"\n", 4, &[], &[], None);

    let (state, output) = run_lone_byte_request(&gen_cfg);
    assert!(output.stopped);
    assert_eq!(output.stop_reason, Some(StopReason::Grammar));
    assert_eq!(output.token_ids, vec![1]);
    assert_eq!(output.text, "\u{fffd}");

    // The completing token was never forwarded, so it stays out of the boundary.
    assert_saved_boundary(&state, CrossTurnSlotId::DEFAULT, &ids_of("ac"));
}

#[test]
fn budget_rejection_after_a_held_byte_saves_the_forwarded_boundary() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    // The grammar forces the lone-byte token first. With a reasoning budget of
    // one, the first decode iteration forwards it and then overrides the sampled
    // token with `</think>`, which the grammar rejects before it is pushed.
    let gen_cfg = lone_byte_cfg("root ::= \"b\" \"b\"\n", 4, &[], &[], Some(1));

    let (state, output) = run_lone_byte_request(&gen_cfg);
    assert!(!output.stopped);
    assert_eq!(output.stop_reason, Some(StopReason::Grammar));
    assert_eq!(output.token_ids, vec![1]);
    assert_eq!(output.text, "\u{fffd}");

    let mut expected = ids_of("ac");
    expected.push(1);
    assert_saved_boundary(&state, CrossTurnSlotId::DEFAULT, &expected);
}

#[test]
fn budget_rejection_after_a_held_byte_then_flush_completed_stop_string_leaves_the_slot_empty() {
    let _gpu_guard = gpu_test_lock();
    if !metal_device_present() {
        return;
    }
    let gen_cfg = lone_byte_cfg("root ::= \"b\" \"b\"\n", 4, &[], &["\u{fffd}"], Some(1));

    let (state, output) = run_lone_byte_request(&gen_cfg);
    assert!(output.stopped);
    assert_eq!(output.stop_reason, Some(StopReason::Eos));
    assert_eq!(output.token_ids, vec![1]);
    assert_eq!(
        output.text, "",
        "the stop string must have been completed by the flush of the held byte"
    );
    assert_slot_empty(
        &state,
        CrossTurnSlotId::DEFAULT,
        "a stop string completed by the final flush after a grammar rejection must not save the \
         tokens behind it",
    );
}
