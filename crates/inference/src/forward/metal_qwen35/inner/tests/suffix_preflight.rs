//! Suffix-content preflight in `plan_prefix_request`.

use super::*;
// -----------------------------------------------------------------------
// #835: suffix-content preflight -- `plan_prefix_request` must run
// BEFORE `restore_cross_turn_prefix`/`reset_state`, both in the pure
// planner (unit tests below) and end-to-end through the public
// `generate_streaming_with_prefix_cache` entry point (integration
// test below).
//
// `single_char_vocab_tokenizer`'s vocab exactly matches
// `tiny_hybrid_fixture`'s `vocab_size` (32) and has no `<unk>`
// fallback, so `tokenize()` can only ever emit ids in `0..32` (any
// other character is silently dropped, never mapped to an
// out-of-range id -- see `BpeTokenizer::encode_piece_to_ids`). With
// that tokenizer the out-of-vocab and empty-suffix branches are
// therefore not directly reachable through the public API (the
// end-to-end test below reaches out-of-vocab with a tokenizer whose
// vocab extends past `vocab_size`), exactly like
// `forward_prefill_from`'s own equivalent checks, which
// `plan_prefix_request` mirrors byte-for-byte; they are exercised
// here as direct (white-box) unit tests of the pure planner
// instead. The overflow branch is likewise unreachable through
// `generate_streaming_with_prefix_cache` for the SAME reason
// `forward_prefill_from`'s original overflow check was for the
// prefix-cache caller: `plan.suffix_start + suffix.len()` always
// equals `prompt_len` (arithmetic identity over
// `&prompt_ids[plan.suffix_start..]`), and `prompt_len >
// max_context()` (== `max_cache_len`) is already rejected earlier,
// with an `Ok`/`KvFull` result, before a plan is ever computed.
// -----------------------------------------------------------------------

/// Direct (white-box) unit test: an out-of-vocab suffix token id
/// must be rejected with the same `InferenceError::InvalidInput`
/// shape `forward_prefill_from` uses.
#[test]
fn plan_prefix_request_rejects_out_of_vocab_suffix_token() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    use crate::error::InferenceError;
    use crate::kv_cache::{PrefixRestorePlan, PrefixReuseMode};

    let (cfg, weights) = tiny_hybrid_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    // cfg.vocab_size == 32 (tiny_hybrid_fixture), so id 32 is out of range.
    let prompt_ids: Vec<u32> = vec![0, 1, 32];
    let plan = PrefixRestorePlan {
        mode: PrefixReuseMode::FullRefill,
        shared_token_prefix_len: 0,
        reusable_len: 0,
        suffix_start: 0,
        suffix_len: prompt_ids.len(),
        old_represented_len: 0,
    };

    let result = state.plan_prefix_request(&prompt_ids, &plan);
    match result {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("out of range"),
                "expected an out-of-range message, got: {msg}"
            );
        }
        other => {
            panic!("expected InvalidInput for an out-of-vocab suffix token; got {other:?}")
        }
    }
}

/// Direct (white-box) unit test: an empty suffix must be rejected
/// with the same `InferenceError::PrefixCache` shape
/// `forward_prefill_from` uses.
#[test]
fn plan_prefix_request_rejects_empty_suffix() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    use crate::error::InferenceError;
    use crate::kv_cache::{PrefixRestorePlan, PrefixReuseMode};

    let (cfg, weights) = tiny_hybrid_fixture();
    let state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let prompt_ids: Vec<u32> = vec![0, 1, 2];
    // suffix_start == prompt_ids.len() -> the selected suffix is empty.
    let plan = PrefixRestorePlan {
        mode: PrefixReuseMode::ReplayFromCheckpoint { checkpoint_len: 3 },
        shared_token_prefix_len: 3,
        reusable_len: 3,
        suffix_start: 3,
        suffix_len: 0,
        old_represented_len: 3,
    };

    let result = state.plan_prefix_request(&prompt_ids, &plan);
    match result {
        Err(InferenceError::PrefixCache(msg)) => {
            assert!(
                msg.contains("empty suffix"),
                "expected an empty-suffix message, got: {msg}"
            );
        }
        other => panic!("expected PrefixCache for an empty suffix; got {other:?}"),
    }
}

/// Direct (white-box) unit test: a suffix whose
/// `suffix_start + len` overflows `max_cache_len` must be rejected
/// with the same `InferenceError::InvalidInput` shape
/// `forward_prefill_from` uses.
#[test]
fn plan_prefix_request_rejects_suffix_overflowing_max_cache_len() {
    let Some(_) = metal::Device::system_default() else {
        return;
    };
    let _guard = gpu_test_lock();
    use crate::error::InferenceError;
    use crate::kv_cache::{PrefixRestorePlan, PrefixReuseMode};

    let (cfg, weights) = tiny_hybrid_fixture();
    // max_cache_len = 32.
    let state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    // 40 valid ids (all < vocab_size=32 by value % 26), total length
    // 40 > max_cache_len 32.
    let prompt_ids: Vec<u32> = (0..40u32).map(|i| i % 26).collect();
    let plan = PrefixRestorePlan {
        mode: PrefixReuseMode::FullRefill,
        shared_token_prefix_len: 0,
        reusable_len: 0,
        suffix_start: 0,
        suffix_len: prompt_ids.len(),
        old_represented_len: 0,
    };

    let result = state.plan_prefix_request(&prompt_ids, &plan);
    match result {
        Err(InferenceError::InvalidInput(msg)) => {
            assert!(
                msg.contains("exceeds max_cache_len"),
                "expected an exceeds-max_cache_len message, got: {msg}"
            );
        }
        other => {
            panic!("expected InvalidInput for a suffix overflowing max_cache_len; got {other:?}")
        }
    }
}

/// End-to-end regression test (#1757): a follow-up prompt that ends
/// exactly on a retained checkpoint boundary must be served, not
/// refused.
///
/// The warm-up call retains a real entry for `"abcdef"` plus its
/// sampled token; a synthetic checkpoint is then injected at
/// `len == 2` (the same boundary a real earlier turn's
/// `save_cross_turn_prefix` could have retained). A follow-up request
/// for `"ab"` shares its entire (2-token) length with both the entry
/// and the checkpoint, so replaying from that checkpoint would leave
/// an empty suffix, which `forward_prefill_from` rejects. The planner
/// must skip that checkpoint and plan `FullRefill`, so the request
/// succeeds and its output equals a full-refill baseline for `"ab"`
/// on a fresh state. Serving it by `FullRefill` legitimately replaces
/// the warm entry, including the injected checkpoint whose snapshot
/// does not represent `len == 2`.
///
/// Mutation sensitivity: dropping the `len < new_prompt_ids.len()`
/// conjunct in `plan_prefix_reuse` makes the planner select
/// `ReplayFromCheckpoint { checkpoint_len: 2 }` with an empty suffix
/// again, and the `"ab"` request returns `Err(PrefixCache(..))`.
#[test]
fn generate_streaming_with_prefix_cache_prompt_ending_on_checkpoint_full_refills() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = metal::Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _guard = gpu_test_lock();

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;

    // Warm up with "abcdef" (6 valid, distinct single-char tokens).
    let warm_cfg = cross_turn_test_gen_cfg(11, 1);
    state
        .generate_streaming_with_prefix_cache(slot_id, "abcdef", &tokenizer, &warm_cfg, |_, _| true)
        .expect("warm-up call must succeed");

    // Inject a synthetic checkpoint at len=2 on the live warm entry,
    // simulating a retained earlier-turn boundary. Its snapshot
    // content is not a valid len=2 state; the planner must not
    // select it for a prompt that ends on it.
    {
        let entry = state
            .cross_turn_prefix_cache
            .entry
            .as_mut()
            .expect("precondition: warm-up call must retain a cache entry");
        let dummy_snapshot = entry.gdn_snapshot.clone();
        entry.checkpoints.push(MetalGdnCheckpoint {
            len: 2,
            snapshot: dummy_snapshot,
        });
    }

    let warm_token_ids_before = state
        .cross_turn_prefix_cache
        .entry
        .as_ref()
        .unwrap()
        .generic
        .token_ids
        .clone();

    // Follow-up "ab": shares its full (2-token) length with the
    // entry AND the injected checkpoint at len=2.
    let result = state
        .generate_streaming_with_prefix_cache(
            slot_id,
            "ab",
            &tokenizer,
            &cross_turn_test_gen_cfg(12, 1),
            |_, _| true,
        )
        .expect("a prompt ending on a checkpoint boundary must be served");
    assert_eq!(
        result.cache.mode,
        crate::kv_cache::PrefixReuseMode::FullRefill,
        "a checkpoint at the prompt's end leaves no suffix to replay; \
                 the planner must fall back to FullRefill"
    );
    assert_eq!(
        result.cache.reused_tokens, 0,
        "FullRefill reuses no cached tokens"
    );

    let entry_after = state
        .cross_turn_prefix_cache
        .entry
        .as_ref()
        .map(|e| e.generic.token_ids.clone());
    assert_ne!(
        entry_after,
        Some(warm_token_ids_before),
        "serving the request by FullRefill must replace the warm entry"
    );

    let mut baseline_state =
        MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let baseline = baseline_state
        .generate_streaming_with_cancel(
            "ab",
            &tokenizer,
            &cross_turn_test_gen_cfg(12, 1),
            |_, _| true,
            || false,
        )
        .expect("full-refill baseline must succeed");
    assert_eq!(
        result.output.token_ids, baseline.token_ids,
        "output for a prompt ending on a checkpoint must match a \
                 full-refill baseline bit-for-bit (token-identity invariant)"
    );
}

/// End-to-end regression test (#835): a request that
/// `plan_prefix_request` refuses must fail through
/// `generate_streaming_with_prefix_cache` WITHOUT consuming a warm
/// entry or touching live KV/GDN state.
///
/// The refusal is reached through the out-of-vocab branch. The
/// prefix-cache entry contract performs no prompt-id vocab check
/// before planning, so a caller-supplied tokenizer whose vocab maps
/// `'G'` to id 32 (one past the tiny fixture's `vocab_size`) gets
/// that id as far as `plan_prefix_request`. The warm-up call
/// retains an entry using only in-range characters; the follow-up
/// prompt is that entry's own tokens plus `'G'`, which plans
/// `ExactAppend` -- the mode whose `restore_cross_turn_prefix`
/// `take()`s the entry out of the slot.
///
/// Mutation sensitivity: moving the `plan_prefix_request` calls in
/// both the public wrapper and `_inner` to after their destructive
/// `restore_cross_turn_prefix`/`reset_state` match makes the entry
/// disappear before the same refusal is returned, so the
/// entry-preservation assertion fails.
#[test]
fn generate_streaming_with_prefix_cache_refusal_preserves_warm_entry() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = metal::Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _guard = gpu_test_lock();

    use crate::error::InferenceError;
    use std::collections::HashMap;

    let (cfg, weights) = tiny_hybrid_fixture();
    // `single_char_vocab_tokenizer`'s 32 entries plus 'G' -> 32,
    // which is out of range for the fixture's `vocab_size`.
    let mut vocab: HashMap<String, u32> = HashMap::new();
    for i in 0u32..33 {
        let byte = if i < 26 {
            b'a' + i as u8
        } else {
            b'A' + (i - 26) as u8
        };
        vocab.insert((byte as char).to_string(), i);
    }
    assert_eq!(
        cfg.vocab_size, 32,
        "precondition: 'G' -> 32 must be out of range"
    );
    let tokenizer = crate::tokenizer::bpe::BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
        .expect("oversized single-char vocab tokenizer build");

    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;

    state
        .generate_streaming_with_prefix_cache(
            slot_id,
            "abcdef",
            &tokenizer,
            &cross_turn_test_gen_cfg(11, 1),
            |_, _| true,
        )
        .expect("warm-up call must succeed");

    let warm_token_ids_before = state
        .cross_turn_prefix_cache
        .entry
        .as_ref()
        .expect("precondition: warm-up call must retain a cache entry")
        .generic
        .token_ids
        .clone();
    let kv_seq_len_before = state.session.kv_cache.seq_len;
    let position_before = state.session.position();

    // The entry's own tokens (the prompt plus the sampled token, all
    // below 32) rendered back through the bijective single-char
    // vocab, then one out-of-range token.
    let mut follow_up_prompt: String = warm_token_ids_before
        .iter()
        .map(|&id| {
            if id < 26 {
                (b'a' + id as u8) as char
            } else {
                (b'A' + (id - 26) as u8) as char
            }
        })
        .collect();
    follow_up_prompt.push('G');

    let mut expected_ids = warm_token_ids_before.clone();
    expected_ids.push(32);
    let encoded = {
        use crate::tokenizer::common::Tokenizer;
        tokenizer.tokenize(&follow_up_prompt)
    };
    assert_eq!(
        &encoded.input_ids[..encoded.real_length],
        expected_ids.as_slice(),
        "precondition: the follow-up must tokenize to the entry's ids plus id 32"
    );
    let metadata = state.cross_turn_metadata(&tokenizer);
    let plan = state.plan_cross_turn_reuse(slot_id, &metadata, &expected_ids);
    assert_eq!(
        plan.mode,
        crate::kv_cache::PrefixReuseMode::ExactAppend,
        "precondition: the follow-up must plan the entry-consuming ExactAppend"
    );

    let result = state.generate_streaming_with_prefix_cache(
        slot_id,
        &follow_up_prompt,
        &tokenizer,
        &cross_turn_test_gen_cfg(12, 1),
        |_, _| true,
    );
    match &result {
        Err(InferenceError::InvalidInput(msg)) => assert!(
            msg.contains("out of range"),
            "expected the plan_prefix_request out-of-range refusal, got: {msg}"
        ),
        other => {
            panic!("an out-of-vocab suffix must be refused by plan_prefix_request; got {other:?}")
        }
    }

    let entry_after = state
        .cross_turn_prefix_cache
        .entry
        .as_ref()
        .map(|e| e.generic.token_ids.clone());
    assert_eq!(
        entry_after,
        Some(warm_token_ids_before),
        "a refused request must not evict or consume the warm entry -- \
                 validation must run before restore_cross_turn_prefix's take()"
    );
    assert_eq!(
        state.session.kv_cache.seq_len, kv_seq_len_before,
        "live KV length must be unchanged by a refused request"
    );
    assert_eq!(
        state.session.position(),
        position_before,
        "live GDN position must be unchanged by a refused request"
    );
}
