//! Grammar-constrained decoding: fail-closed admission, step-zero terminals and the decode loop.

use super::*;
// -----------------------------------------------------------------------
// Grammar fail-closed integration tests (#611)
//
// These three tests are the hardware-gated end-to-end counterpart to the
// pure-function `has_finite_logit_tests` module further down this file:
// that module proves the *predicate* is correct in isolation, these prove
// it is actually *wired in* at each real public Metal entry point, using
// the same technique as the CPU reference test
// `grammar_wiring_mask_logits_called_in_generate` in
// `model/qwen35/generation.rs`: a `GrammarEngine` built over a vocab
// table of all-empty byte sequences. `VocabPartition::build` rejects
// empty entries (they can never advance the PDA), so the precomputed
// bitmask for the initial state is all-zeros and `mask_logits` sets
// every one of `tiny_hybrid_fixture`'s 32 logits to `NEG_INFINITY`
// before the very first sample — a real, not simulated, all-blocked
// mask flowing through the real Metal forward path.
//
// Scope: these three cover the post-prefill masking site in each
// function. The decode-loop site — a second, structurally identical
// `has_finite_logit` guard a few lines below the prefill one in each
// of the three functions (`generate`, `generate_streaming_with_cancel`,
// `generate_streaming_with_prefix_cache_inner`) — is reachable only
// from step 2 onward and needs a grammar that allows exactly one
// token and then blocks everything. #611: that gap is covered separately by the
// "DECODE-LOOP integration tests" block below this one, which builds
// exactly that allow-then-block fixture.
//
// Mutation sensitivity (all three): reverting the Metal-side
// `has_finite_logit` guard at the corresponding prefill site (or
// stubbing it to always return `true`) leaves the all-`NEG_INFINITY`
// row in place; the sampler's non-finite-max short-circuit then
// silently returns token 0 and each `matches!(result, Err(...))`
// assertion below fails instead of passing — this was verified by
// temporarily reverting the guard and re-running these three tests
// (see PR description).

/// #611: `generate`'s post-prefill grammar-masking site must fail
/// closed via `has_finite_logit`, exactly like the CPU path.
#[test]
fn generate_fails_closed_on_all_blocking_grammar() {
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
    use crate::generation::GenerateConfig;
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    // vocab_size=32 in `tiny_hybrid_fixture`; an all-empty-byte-sequence
    // vocab table makes every one of the 32 entries unreachable by the
    // PDA, so the initial-state mask blocks every logit.
    let vocab_bytes: Vec<Vec<u8>> = vec![vec![]; 32];
    let spec = GrammarSpec::Gbnf("root ::= \"ok\"\n".to_string());
    let engine = Arc::new(
        GrammarEngine::new(&spec, vocab_bytes).expect("grammar engine builds with empty vocab"),
    );

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: Some(engine),
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let result = state.generate("a", &tokenizer, &gen_cfg);
    assert!(
        matches!(result, Err(InferenceError::GrammarConstraintBlocked(_))),
        "grammar blocking every token must fail closed via generate(); got {result:?}"
    );
}

/// #611: `generate_streaming`'s post-prefill grammar-masking site must
/// fail closed too. `generate_streaming` is a pure
/// `should_cancel = || false` delegate to
/// `generate_streaming_with_cancel` (see that method's doc comment),
/// so this one GPU run exercises both public entry points' shared
/// implementation without a redundant second pass.
#[test]
fn generate_streaming_fails_closed_on_all_blocking_grammar() {
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
    use crate::generation::GenerateConfig;
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let vocab_bytes: Vec<Vec<u8>> = vec![vec![]; 32];
    let spec = GrammarSpec::Gbnf("root ::= \"ok\"\n".to_string());
    let engine = Arc::new(
        GrammarEngine::new(&spec, vocab_bytes).expect("grammar engine builds with empty vocab"),
    );

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: Some(engine),
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let mut on_token_calls = 0u32;
    let result = state.generate_streaming("a", &tokenizer, &gen_cfg, |_delta, _id| {
        on_token_calls += 1;
        true
    });

    assert!(
        matches!(result, Err(InferenceError::GrammarConstraintBlocked(_))),
        "grammar blocking every token must fail closed via generate_streaming(); \
                 got {result:?}"
    );
    assert_eq!(
        on_token_calls, 0,
        "a prefill-blocked grammar must fail before any token ever reaches on_token"
    );
}

/// #611: `generate_streaming_with_prefix_cache`'s post-prefill
/// grammar-masking site (inside
/// `generate_streaming_with_prefix_cache_inner`) must fail closed too,
/// AND the public wrapper's error path must not leave a stale cache
/// entry behind: unlike the CPU-only `_inner`, the public wrapper
/// resets state and evicts `slot_id`'s cache entry on any `Err` before
/// re-raising it (see that method's doc comment), so a
/// grammar-fail-closed turn can never leave the cache in a state a
/// later turn could incorrectly reuse.
#[test]
fn generate_streaming_with_prefix_cache_fails_closed_on_all_blocking_grammar() {
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
    use crate::generation::GenerateConfig;
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;

    let vocab_bytes: Vec<Vec<u8>> = vec![vec![]; 32];
    let spec = GrammarSpec::Gbnf("root ::= \"ok\"\n".to_string());
    let engine = Arc::new(
        GrammarEngine::new(&spec, vocab_bytes).expect("grammar engine builds with empty vocab"),
    );

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: Some(engine),
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let result =
        state.generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true);

    assert!(
        matches!(result, Err(InferenceError::GrammarConstraintBlocked(_))),
        "grammar blocking every token must fail closed via \
                 generate_streaming_with_prefix_cache(); got {result:?}"
    );
    assert!(
        state.cross_turn_prefix_cache.entry.is_none(),
        "a grammar-fail-closed turn must not leave a stale cache entry behind"
    );
}

// -----------------------------------------------------------------------
// Grammar terminal-at-step-zero integration tests (#1153)
//
// The all-blocking tests above prove the incomplete/dead-end half of
// each post-prefill branch. An epsilon grammar reaches the same
// all-NEG_INFINITY mask from an accepting state: no token may extend
// the empty document, so that result is a clean Grammar stop rather
// than GrammarConstraintBlocked. The fixture asserts both
// preconditions before any GPU work so a parser or vocabulary change
// cannot make these tests pass through a different branch.

fn terminal_step_zero_grammar_cfg() -> GenerateConfig {
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    let engine = Arc::new(
        GrammarEngine::new(
            &GrammarSpec::Gbnf("root ::= \"\"\n".to_string()),
            single_char_vocab_bytes(),
        )
        .expect("epsilon grammar builds over single-char vocab"),
    );
    let mut grammar_state = engine.initial_state();
    assert!(
        engine.is_complete_without_continuation(&grammar_state),
        "epsilon grammar must start complete with no legal continuation"
    );
    let mut logits = vec![0.0; 32];
    engine
        .mask_logits(&mut grammar_state, &mut logits)
        .expect("local epsilon engine and full vocabulary logits must mask");
    assert!(
        !crate::forward::metal_qwen35::has_finite_logit(&logits),
        "terminal epsilon grammar must mask every continuation"
    );

    GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: Some(engine),
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    }
}

fn assert_terminal_step_zero_output(output: &GenerateOutput) {
    assert_eq!(output.text, "");
    assert!(output.token_ids.is_empty());
    assert_eq!(output.generated_tokens, 0);
    assert!(output.stopped, "terminal grammar is a successful stop");
    assert_eq!(output.stop_reason, Some(StopReason::Grammar));
    assert!(output.token_logprobs.is_empty());
}

/// #1153: `generate` must distinguish an initially terminal grammar
/// from the incomplete all-blocking grammar covered above.
#[test]
fn generate_terminal_grammar_at_step_zero_is_success() {
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

    let output = state
        .generate("a", &tokenizer, &terminal_step_zero_grammar_cfg())
        .expect("an initially terminal grammar is a successful stop");
    assert_terminal_step_zero_output(&output);
}

/// #1153: the shared streaming implementation must make the same
/// terminal-versus-dead-end distinction without emitting a token.
#[test]
fn generate_streaming_terminal_grammar_at_step_zero_is_success() {
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
    let mut on_token_calls = 0usize;

    let output = state
        .generate_streaming(
            "a",
            &tokenizer,
            &terminal_step_zero_grammar_cfg(),
            |_, _| {
                on_token_calls += 1;
                true
            },
        )
        .expect("an initially terminal grammar is a successful streaming stop");
    assert_terminal_step_zero_output(&output);
    assert_eq!(
        on_token_calls, 0,
        "step-zero completion must stop before any token reaches on_token"
    );
}

/// #1153: the prefix-cache Metal path must also return a clean
/// terminal stop rather than entering its destructive error recovery.
#[test]
fn generate_streaming_with_prefix_cache_terminal_grammar_at_step_zero_is_success() {
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
    let mut on_token_calls = 0usize;

    let result = state
        .generate_streaming_with_prefix_cache(
            slot_id,
            "a",
            &tokenizer,
            &terminal_step_zero_grammar_cfg(),
            |_, _| {
                on_token_calls += 1;
                true
            },
        )
        .expect("an initially terminal grammar is a successful cached streaming stop");
    assert_terminal_step_zero_output(&result.output);
    assert_eq!(
        on_token_calls, 0,
        "step-zero completion must stop before any token reaches on_token"
    );
}

/// `generate_streaming_with_prefix_cache` must reject an empty
/// prompt with a typed `Err(Inference("empty prompt"))` (#856),
/// unifying it with all three CPU forward paths instead of the
/// empty `Ok(CachedGenerateOutput { output: GenerateOutput {
/// stopped: false, stop_reason: None, .. }, .. })` `_inner` used to
/// return. The guard now lives in the public wrapper
/// (`generate_streaming_with_prefix_cache_and_cancel`), before
/// `_inner` is ever called.
///
/// A fresh-state / `entry ==
/// None` assertion alone does not exercise the property that
/// motivated putting the guard in the wrapper rather than
/// `_inner` -- a *warmed* slot survives the rejection, because the
/// guard returns before `_inner`'s unconditional Err-recovery
/// eviction ever runs. This mirrors
/// `..._rejects_logprobs_request_preserves_warm_entry` immediately
/// below: warm the slot first, submit the empty-prompt request,
/// assert both the exact error and that the pre-existing entry is
/// unchanged afterward.
///
/// Comparing only `generic.token_ids` does not prove
/// "byte-identical" survival; `MetalCrossTurnPrefixEntry` also
/// carries `gdn_snapshot` and sparse `checkpoints`. This test
/// compares every field: `generic` (`kv_cache::CrossTurnPrefixEntry`,
/// which derives `PartialEq`/`Eq` over `slot_id`, `metadata`,
/// `token_ids`, `represented_len`, the `kv` handle -- itself a
/// small `Copy` value type of offsets/lengths, not a raw device
/// pointer -- and `gdn_snapshot_len`) via a direct `==`, plus
/// `gdn_snapshot` (`Vec<(Vec<f32>, Vec<f32>)>`) and each
/// `checkpoints` entry's `(len, snapshot)` via element-wise `==`
/// (neither type derives `PartialEq` itself, but their constituent
/// `Vec<f32>`/tuple/`usize` fields all do). There is no field on
/// `MetalCrossTurnPrefixEntry` this leaves uncompared.
///
/// Mutation sensitivity: reverting the `check_prompt_not_empty`
/// call in the wrapper makes this fail two ways -- the call
/// proceeds to `_inner` and returns `Ok` instead of `Err`, AND
/// (if the guard were instead duplicated as an `Err` inside
/// `_inner`, the alternative bug this guards against) the warmed
/// entry would be evicted by `_inner`'s destructive recovery arm.
#[test]
fn generate_streaming_with_prefix_cache_rejects_empty_prompt() {
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

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;

    let warm_cfg = cross_turn_test_gen_cfg(9, 2);
    state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &warm_cfg, |_, _| true)
        .expect("warm-up call must succeed");
    let warm_entry = state
        .cross_turn_prefix_cache
        .entry
        .as_ref()
        .expect("precondition: the warm-up call must retain a cache entry");
    let warm_generic = warm_entry.generic.clone();
    let warm_gdn_snapshot = warm_entry.gdn_snapshot.clone();
    let warm_checkpoints: Vec<(usize, crate::attention::gdn::GdnSnapshot)> = warm_entry
        .checkpoints
        .iter()
        .map(|c| (c.len, c.snapshot.clone()))
        .collect();

    let rejected_cfg = cross_turn_test_gen_cfg(9, 2);
    let result = state.generate_streaming_with_prefix_cache(
        slot_id,
        "",
        &tokenizer,
        &rejected_cfg,
        |_, _| true,
    );
    assert!(
        matches!(result, Err(InferenceError::Inference(ref msg)) if msg.contains("empty prompt")),
        "generate_streaming_with_prefix_cache must reject an empty \
                 prompt with Err(Inference(\"empty prompt\")) (#856); got {result:?}"
    );

    let entry_after = state.cross_turn_prefix_cache.entry.as_ref().expect(
        "an empty-prompt rejection must not evict a pre-existing \
                     cache entry it never touched -- the guard runs before \
                     _inner's destructive Err-recovery block, same invariant \
                     as the logprobs/MTP preflight tests",
    );
    assert_eq!(
        entry_after.generic, warm_generic,
        "the retained entry's generic (slot_id/metadata/token_ids/\
                 represented_len/kv handle/gdn_snapshot_len) must be \
                 unchanged by a rejected empty-prompt request"
    );
    assert_eq!(
        entry_after.gdn_snapshot, warm_gdn_snapshot,
        "the retained entry's GDN recurrent-state snapshot must be \
                 unchanged by a rejected empty-prompt request"
    );
    let after_checkpoints: Vec<(usize, crate::attention::gdn::GdnSnapshot)> = entry_after
        .checkpoints
        .iter()
        .map(|c| (c.len, c.snapshot.clone()))
        .collect();
    assert_eq!(
        after_checkpoints, warm_checkpoints,
        "the retained entry's sparse GDN checkpoints must be \
                 unchanged by a rejected empty-prompt request"
    );
}

/// PR #787: `generate_streaming_with_prefix_cache` never wires per-token
/// logprobs capture into its decode loop. Before the fix, a caller
/// requesting `gen_cfg.logprobs` got `Ok` back with silently empty
/// `token_logprobs` -- the exact silent-omission defect ADR-080 C3's
/// apply-or-fail-closed contract exists to close. The guard must
/// reject the request before any cache/state mutation, so no cache
/// entry is created (or consumed) for the rejected call.
///
/// Mutation sensitivity: removing the `check_logprobs_not_set` call
/// from `GenerationEntryContract::MetalPrefixCacheStreaming`'s
/// `validate_before_tokenization` (#1354) makes this assertion fail
/// (the call returns `Ok` with empty `token_logprobs` instead of
/// `Err`).
#[test]
fn generate_streaming_with_prefix_cache_rejects_logprobs_request() {
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
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 2,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: None,
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: Some(5),
    };

    let result =
        state.generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true);

    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "a logprobs request must fail closed via \
                 generate_streaming_with_prefix_cache() instead of silently \
                 returning empty token_logprobs; got {result:?}"
    );
    assert!(
        state.cross_turn_prefix_cache.entry.is_none(),
        "a logprobs-rejected call must not create or leave a cache \
                 entry behind -- the guard runs before any state mutation"
    );
}

/// PR #787: an earlier fix guarded the
/// NO-entry case, but not the case where a valid entry already
/// exists. The public wrapper
/// `generate_streaming_with_prefix_cache_and_cancel` used to catch
/// EVERY `Err` from `_inner` -- including a preflight-only rejection
/// that never attempted cache/state mutation -- and unconditionally
/// call `reset_state()` + remove the slot. A caller who warms a live
/// cross-turn entry and then makes one invalid (`logprobs`-set)
/// call would have that valid entry destroyed even though the
/// rejected call touched nothing.
///
/// Mutation sensitivity: hoisting the preflight checks in
/// `generate_streaming_with_prefix_cache_and_cancel` back below the
/// `match` (i.e. routing them through `_inner` again) makes this
/// test fail: the wrapper's `Err` arm evicts the warm entry.
#[test]
fn generate_streaming_with_prefix_cache_rejects_logprobs_request_preserves_warm_entry() {
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
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;

    let warm_cfg = cross_turn_test_gen_cfg(9, 2);
    state
        .generate_streaming_with_prefix_cache(slot_id, "ab", &tokenizer, &warm_cfg, |_, _| true)
        .expect("warm-up call must succeed");
    let warm_entry = state
        .cross_turn_prefix_cache
        .entry
        .as_ref()
        .expect("precondition: the warm-up call must retain a cache entry")
        .generic
        .token_ids
        .clone();

    let rejected_cfg = GenerateConfig {
        min_p: 0.0,
        logprobs: Some(0),
        ..cross_turn_test_gen_cfg(9, 2)
    };
    let result = state.generate_streaming_with_prefix_cache(
        slot_id,
        "ab",
        &tokenizer,
        &rejected_cfg,
        |_, _| true,
    );
    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "the logprobs-set call must still fail closed; got {result:?}"
    );

    let entry_after = state
        .cross_turn_prefix_cache
        .entry
        .as_ref()
        .map(|e| e.generic.token_ids.clone());
    assert_eq!(
        entry_after,
        Some(warm_entry),
        "a preflight-only rejection must not evict a pre-existing \
                 cache entry it never touched -- the public wrapper must \
                 return before its destructive Err-recovery block runs"
    );
}

/// PR #787: the prefix-cache path accepts
/// `GenerateConfig` but never reads `enable_mtp`, unlike the direct
/// Metal `generate()` entry point, which resolves it and delegates
/// to `generate_greedy_mtp`. An active MTP request (explicit
/// `Some(true)` or `LATTICE_MTP` set with `enable_mtp: None`) must
/// be rejected before any cache/state mutation rather than silently
/// falling back to plain per-token decode with no MTP applied and
/// no indication the request was ignored.
///
/// Mutation sensitivity: removing the `check_mtp_not_requested` call
/// from `GenerationEntryContract::MetalPrefixCacheStreaming`'s
/// `validate_before_tokenization` (#1354) makes this assertion fail
/// (the call returns `Ok` instead of `Err`).
#[test]
fn generate_streaming_with_prefix_cache_rejects_active_mtp_request() {
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

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;

    let gen_cfg = crate::generation::GenerateConfig {
        min_p: 0.0,
        enable_mtp: Some(true),
        ..cross_turn_test_gen_cfg(1, 2)
    };

    let result =
        state.generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true);

    assert!(
        matches!(result, Err(InferenceError::InvalidInput(_))),
        "an explicit enable_mtp: Some(true) request must fail closed \
                 via generate_streaming_with_prefix_cache() instead of \
                 silently falling back to plain decode; got {result:?}"
    );
    assert!(
        state.cross_turn_prefix_cache.entry.is_none(),
        "an MTP-rejected call must not create or leave a cache \
                 entry behind -- the guard runs before any state mutation"
    );
}

/// #1354: a request that violates two guards at once must surface
/// the *specific* error the pre-refactor call order produces, not
/// merely *an* error -- a single-violation suite cannot distinguish
/// "checks order preserved" from "checks order silently changed"
/// because every single-violation test still passes either way.
/// `logprobs` and `enable_mtp` were both checked before tokenization
/// in the public wrapper, `logprobs` first; an empty prompt is only
/// discovered afterward, inside `prepare_generation`. A caller that
/// sets `logprobs` on an empty-prompt request must therefore still
/// see the logprobs error, not "empty prompt" -- and a caller that
/// sets both `logprobs` and `enable_mtp` must see the logprobs
/// error, not the MTP one, because logprobs is checked first.
///
/// Mutation sensitivity: folding either preflight into
/// `prepare_generation`'s post-tokenization `validate_capabilities`
/// (rather than a pre-tokenization hook) flips the empty-prompt
/// cases to `Err(Inference("empty prompt"))`; swapping the
/// logprobs/MTP check order flips the combined case to the MTP
/// error text.
#[test]
fn generate_streaming_with_prefix_cache_multi_violation_requests_return_the_earlier_guards_error() {
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
    use crate::generation::GenerateConfig;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;

    // logprobs + enable_mtp both set: logprobs must win (checked first).
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let both_capabilities_cfg = GenerateConfig {
        min_p: 0.0,
        enable_mtp: Some(true),
        logprobs: Some(0),
        ..cross_turn_test_gen_cfg(1, 2)
    };
    let result = state.generate_streaming_with_prefix_cache(
        slot_id,
        "a",
        &tokenizer,
        &both_capabilities_cfg,
        |_, _| true,
    );
    assert!(
        matches!(
            result,
            Err(InferenceError::InvalidInput(ref msg)) if msg.contains("logprobs")
        ),
        "logprobs and enable_mtp set together must reject with the \
                 logprobs error (checked before MTP); got {result:?}"
    );

    // logprobs set + empty prompt: logprobs must win (checked before
    // tokenization discovers the prompt is empty).
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let logprobs_and_empty_cfg = GenerateConfig {
        min_p: 0.0,
        logprobs: Some(0),
        ..cross_turn_test_gen_cfg(1, 2)
    };
    let result = state.generate_streaming_with_prefix_cache(
        slot_id,
        "",
        &tokenizer,
        &logprobs_and_empty_cfg,
        |_, _| true,
    );
    assert!(
        matches!(
            result,
            Err(InferenceError::InvalidInput(ref msg)) if msg.contains("logprobs")
        ),
        "logprobs set on an empty-prompt request must reject with the \
                 logprobs error, not \"empty prompt\"; got {result:?}"
    );

    // enable_mtp set + empty prompt: MTP must win (checked before
    // tokenization discovers the prompt is empty).
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let mtp_and_empty_cfg = GenerateConfig {
        min_p: 0.0,
        enable_mtp: Some(true),
        ..cross_turn_test_gen_cfg(1, 2)
    };
    let result = state.generate_streaming_with_prefix_cache(
        slot_id,
        "",
        &tokenizer,
        &mtp_and_empty_cfg,
        |_, _| true,
    );
    assert!(
        matches!(
            result,
            Err(InferenceError::InvalidInput(ref msg)) if msg.contains("enable_mtp")
        ),
        "enable_mtp set on an empty-prompt request must reject with \
                 the MTP error, not \"empty prompt\"; got {result:?}"
    );
}

/// #827: `generate_streaming_with_prefix_cache_and_cancel_inner` was
/// the last of the seven Qwen3.5 generation entry points that did not
/// route through the shared `prepare_generation` preparation --
/// it derived its RNG seed with its own inline copy of the
/// normalize-seed arithmetic instead. This is a behavioural check,
/// not a textual one (grepping the source for `prepare_generation`
/// would prove the call is written, not that admission actually
/// runs through it): it drives the already-migrated
/// `generate_streaming_with_cancel` sibling and the prefix-cache
/// entry point with the identical over-budget prompt/config on
/// identically-configured fixtures, and asserts both reject with
/// the exact same `check_context_budget` text -- the shared
/// `Self::MetalDirect | Self::MetalStreaming | Self::MetalPrefixCacheStreaming`
/// arm `validate_context` owns inside `prepare_generation` (#1354).
/// Both entry points now resolve that arm through the one shared
/// call site instead of two independently maintained copies, so a
/// future change to the bound (or to which contract either path is
/// threaded through) cannot silently diverge between the two
/// without this test catching it.
#[test]
fn generate_streaming_with_prefix_cache_context_rejection_matches_migrated_sibling() {
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

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let gen_cfg = cross_turn_test_gen_cfg(1, 1);
    let over_budget_prompt = "a".repeat(32);

    let mut streaming = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let streaming_result = streaming.generate_streaming_with_cancel(
        &over_budget_prompt,
        &tokenizer,
        &gen_cfg,
        |_, _| true,
        || false,
    );
    assert_context_budget_error(&streaming_result, 32, 1, 32);

    let mut cached = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;
    let cached_result = cached.generate_streaming_with_prefix_cache(
        slot_id,
        &over_budget_prompt,
        &tokenizer,
        &gen_cfg,
        |_, _| true,
    );
    assert_context_budget_error(&cached_result, 32, 1, 32);

    let (Err(InferenceError::Inference(streaming_msg)), Err(InferenceError::Inference(cached_msg))) =
        (&streaming_result, &cached_result)
    else {
        panic!(
            "both entry points must reject with InferenceError::Inference; \
                     got streaming={streaming_result:?} cached={cached_result:?}"
        );
    };
    assert_eq!(
        streaming_msg, cached_msg,
        "generate_streaming_with_prefix_cache_and_cancel must reject an \
                 over-budget prompt with the exact same check_context_budget text \
                 generate_streaming_with_cancel produces (#827): both route \
                 through prepare_generation's shared validate_context arm, \
                 albeit via different contract variants \
                 (MetalStreaming / MetalPrefixCacheStreaming, #1354)"
    );
}

// -----------------------------------------------------------------------
// Grammar DECODE-LOOP integration tests (#611 fail-closed + #1064
// completion-is-success)
//
// The three tests above only reach the post-prefill masking site: their
// fixture blocks every vocab entry from step 0. The decode loop of each
// function (`generate`, `generate_streaming_with_cancel`,
// `generate_streaming_with_prefix_cache_inner`) carries two distinct
// grammar contracts, each pinned by its own fixture below:
//
// 1. COMPLETION (#1064): GBNF `root ::= "a"`. Token id 0 in
//    `single_char_vocab_tokenizer` is `'a'`; after `'a'` is sampled and
//    the grammar advances past it, the state is complete with no legal
//    continuation (`is_complete_without_continuation` is true). That is
//    a SUCCESSFUL stop: the generated text must come back with
//    `stopped: true` / `StopReason::Grammar`, exactly like the CPU
//    paths. Each path checks completion at two sites, mirroring the
//    CPU `grammar_complete` / `grammar_complete_without_continuation`
//    placement: recorded immediately after the INITIAL advance (so a
//    zero-iteration decode loop — `max_new_tokens: 1` — cannot fall
//    through to Length) and re-checked after every IN-LOOP advance (so
//    completion on the final allowed token cannot either). Before
//    #1064 the Metal loops missed completion entirely, ran one more
//    mask (all 32 logits blocked), and misreported the completed
//    generation as GrammarConstraintBlocked; the first #1064 fix
//    probed only at the loop top, which no zero-iteration or
//    final-iteration run ever reaches, misreporting those as Length.
//    Each check is load-bearing: removing the post-initial-advance
//    recording fails the `max_one_token` tests below (Length, not
//    Grammar); removing a post-advance in-loop check fails that
//    path's `final_token` test the same way; removing the loop-top
//    `if stopped` break restores the GrammarConstraintBlocked
//    misclassification in the `max_new_tokens: 2` completion tests.
//
// 2. DEAD END (#611): GBNF `root ::= "a" "!"`. `'!'` is not in the
//    32-entry vocab (`'a'..='z'`, `'A'..='F'`), so after `'a'` the
//    grammar requires a byte no token can supply: the state is NOT
//    complete and the next mask blocks all 32 tokens. That must remain
//    a hard `GrammarConstraintBlocked` error via the decode-loop
//    `has_finite_logit` guard; without that guard the sampler's
//    non-finite-max short-circuit silently returns a token instead
//    of failing.
//
// The dead-end fixture needs `max_new_tokens: 2` so the loop actually
// reaches the decode iteration after the prefill-derived token; with
// `max_new_tokens: 1` the function returns at the cap and never masks
// a second step. The completion fixtures deliberately run at BOTH
// caps: `max_new_tokens: 2` pins the loop-top break, and
// `max_new_tokens: 1` pins the post-initial-advance recording that a
// zero-iteration loop depends on. `root ::= "a" "a"` at
// `max_new_tokens: 2` pins the in-loop post-advance check (completion
// lands exactly on the final allowed token).

/// #1064: a grammar that completes after one token is a successful
/// stop via `generate`, not a constraint failure.
#[test]
fn generate_decode_loop_completed_grammar_is_success() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = metal::Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _guard = gpu_test_lock();

    use crate::generation::GenerateConfig;
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let spec = GrammarSpec::Gbnf("root ::= \"a\"\n".to_string());
    let engine = Arc::new(
        GrammarEngine::new(&spec, single_char_vocab_bytes())
            .expect("grammar engine builds over single-char vocab"),
    );

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 2,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: Some(engine),
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let result = state.generate("a", &tokenizer, &gen_cfg);
    let output = result.expect("a completed grammar is a successful stop, not an error");
    assert_eq!(
        output.text, "a",
        "the completing token must be in the output text"
    );
    assert_eq!(output.generated_tokens, 1);
    assert!(output.stopped, "grammar completion is a natural stop");
    assert_eq!(output.stop_reason, Some(StopReason::Grammar));
}

/// #1064: a grammar that completes after one token is a successful
/// stop via `generate_streaming`, and the completing token's delta
/// reaches `on_token` exactly once.
#[test]
fn generate_streaming_decode_loop_completed_grammar_is_success() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = metal::Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _guard = gpu_test_lock();

    use crate::generation::GenerateConfig;
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let spec = GrammarSpec::Gbnf("root ::= \"a\"\n".to_string());
    let engine = Arc::new(
        GrammarEngine::new(&spec, single_char_vocab_bytes())
            .expect("grammar engine builds over single-char vocab"),
    );

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 2,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: Some(engine),
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let mut on_token_calls = 0u32;
    let mut streamed = String::new();
    let result = state.generate_streaming("a", &tokenizer, &gen_cfg, |delta, _id| {
        on_token_calls += 1;
        streamed.push_str(delta);
        true
    });

    let output = result.expect("a completed grammar is a successful stop, not an error");
    assert_eq!(output.text, "a");
    assert!(output.stopped, "grammar completion is a natural stop");
    assert_eq!(output.stop_reason, Some(StopReason::Grammar));
    assert_eq!(
        on_token_calls, 1,
        "exactly the completing token's delta reaches on_token"
    );
    assert_eq!(streamed, "a");
}

/// #1064: a grammar that completes after one token is a successful
/// stop via `generate_streaming_with_prefix_cache` too.
#[test]
fn generate_streaming_with_prefix_cache_decode_loop_completed_grammar_is_success() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = metal::Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _guard = gpu_test_lock();

    use crate::generation::GenerateConfig;
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;

    let spec = GrammarSpec::Gbnf("root ::= \"a\"\n".to_string());
    let engine = Arc::new(
        GrammarEngine::new(&spec, single_char_vocab_bytes())
            .expect("grammar engine builds over single-char vocab"),
    );

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 2,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: Some(engine),
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let result =
        state.generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true);

    let output = result
        .expect("a completed grammar is a successful stop, not an error")
        .output;
    assert_eq!(output.text, "a");
    assert!(output.stopped, "grammar completion is a natural stop");
    assert_eq!(output.stop_reason, Some(StopReason::Grammar));
}

/// One-token regression (#1064 follow-up): with `max_new_tokens: 1`
/// the decode loop runs zero iterations, so only the
/// post-initial-advance completion recording can classify the stop.
/// Before the fix `generate` fell through to `stopped: false` /
/// `StopReason::Length` for a grammar its only token had completed.
#[test]
fn generate_max_one_token_completed_grammar_is_success() {
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
    let gen_cfg = single_char_grammar_cfg("root ::= \"a\"\n", 1);

    let output = state
        .generate("a", &tokenizer, &gen_cfg)
        .expect("a completed grammar is a successful stop, not an error");
    assert_eq!(
        output.text, "a",
        "the completing token must be in the output text"
    );
    assert_eq!(output.generated_tokens, 1);
    assert!(
        output.stopped,
        "first-token grammar completion at max_new_tokens=1 is a \
                 natural stop, not a length cap"
    );
    assert_eq!(output.stop_reason, Some(StopReason::Grammar));
}

/// One-token regression (#1064 follow-up) via `generate_streaming`:
/// the completing token's delta must still reach `on_token` exactly
/// once, and the stop must classify as Grammar, not Length.
#[test]
fn generate_streaming_max_one_token_completed_grammar_is_success() {
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
    let gen_cfg = single_char_grammar_cfg("root ::= \"a\"\n", 1);

    let mut on_token_calls = 0u32;
    let mut streamed = String::new();
    let output = state
        .generate_streaming("a", &tokenizer, &gen_cfg, |delta, _id| {
            on_token_calls += 1;
            streamed.push_str(delta);
            true
        })
        .expect("a completed grammar is a successful stop, not an error");

    assert_eq!(output.text, "a");
    assert_eq!(output.generated_tokens, 1);
    assert!(
        output.stopped,
        "first-token grammar completion at max_new_tokens=1 is a \
                 natural stop, not a length cap"
    );
    assert_eq!(output.stop_reason, Some(StopReason::Grammar));
    assert_eq!(
        on_token_calls, 1,
        "exactly the completing token's delta reaches on_token"
    );
    assert_eq!(streamed, "a");
}

/// One-token regression (#1064 follow-up) via
/// `generate_streaming_with_prefix_cache`.
#[test]
fn generate_streaming_with_prefix_cache_max_one_token_completed_grammar_is_success() {
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
    let gen_cfg = single_char_grammar_cfg("root ::= \"a\"\n", 1);

    let output = state
        .generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true)
        .expect("a completed grammar is a successful stop, not an error")
        .output;
    assert_eq!(output.text, "a");
    assert_eq!(output.generated_tokens, 1);
    assert!(
        output.stopped,
        "first-token grammar completion at max_new_tokens=1 is a \
                 natural stop, not a length cap"
    );
    assert_eq!(output.stop_reason, Some(StopReason::Grammar));
}

/// Final-token regression (#1064 follow-up): `root ::= "a" "a"` at
/// `max_new_tokens: 2` completes on the decode loop's LAST allowed
/// iteration, so no following loop top exists and only the in-loop
/// post-advance completion check can classify the stop. Before the
/// fix `generate` exited by count and misreported Length.
#[test]
fn generate_final_token_grammar_completion_is_grammar_stop() {
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
    let gen_cfg = single_char_grammar_cfg("root ::= \"a\" \"a\"\n", 2);

    let output = state
        .generate("a", &tokenizer, &gen_cfg)
        .expect("a completed grammar is a successful stop, not an error");
    assert_eq!(
        output.text, "aa",
        "both grammar-forced tokens must be in the output text"
    );
    assert_eq!(output.generated_tokens, 2);
    assert!(
        output.stopped,
        "grammar completion on the final allowed token is a natural \
                 stop, not a length cap"
    );
    assert_eq!(output.stop_reason, Some(StopReason::Grammar));
}

/// Final-token regression (#1064 follow-up) via `generate_streaming`.
#[test]
fn generate_streaming_final_token_grammar_completion_is_grammar_stop() {
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
    let gen_cfg = single_char_grammar_cfg("root ::= \"a\" \"a\"\n", 2);

    let mut streamed = String::new();
    let output = state
        .generate_streaming("a", &tokenizer, &gen_cfg, |delta, _id| {
            streamed.push_str(delta);
            true
        })
        .expect("a completed grammar is a successful stop, not an error");

    assert_eq!(output.text, "aa");
    assert_eq!(output.generated_tokens, 2);
    assert!(
        output.stopped,
        "grammar completion on the final allowed token is a natural \
                 stop, not a length cap"
    );
    assert_eq!(output.stop_reason, Some(StopReason::Grammar));
    assert_eq!(streamed, "aa");
}

/// Final-token regression (#1064 follow-up) via
/// `generate_streaming_with_prefix_cache`.
#[test]
fn generate_streaming_with_prefix_cache_final_token_grammar_completion_is_grammar_stop() {
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
    let gen_cfg = single_char_grammar_cfg("root ::= \"a\" \"a\"\n", 2);

    let output = state
        .generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true)
        .expect("a completed grammar is a successful stop, not an error")
        .output;
    assert_eq!(output.text, "aa");
    assert_eq!(output.generated_tokens, 2);
    assert!(
        output.stopped,
        "grammar completion on the final allowed token is a natural \
                 stop, not a length cap"
    );
    assert_eq!(output.stop_reason, Some(StopReason::Grammar));
}

/// #611: a grammar that dead-ends mid-document (`root ::= "a" "!"`,
/// with `'!'` absent from the vocab) must still fail closed at
/// `generate`'s decode-loop `has_finite_logit` guard: the state is not
/// complete, so the all-blocked mask is a real constraint failure.
#[test]
fn generate_decode_loop_fails_closed_on_grammar_dead_end() {
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
    use crate::generation::GenerateConfig;
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let spec = GrammarSpec::Gbnf("root ::= \"a\" \"!\"\n".to_string());
    let engine = Arc::new(
        GrammarEngine::new(&spec, single_char_vocab_bytes())
            .expect("grammar engine builds over single-char vocab"),
    );

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 2,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: Some(engine),
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let result = state.generate("a", &tokenizer, &gen_cfg);
    assert!(
        matches!(result, Err(InferenceError::GrammarConstraintBlocked(_))),
        "a grammar whose only continuation byte has no vocab token must \
                 fail closed at the decode-loop guard; got {result:?}"
    );
}

/// #611: the same mid-document dead end must fail closed via
/// `generate_streaming` as well, with at most the first (legal) token
/// reaching `on_token`.
#[test]
fn generate_streaming_decode_loop_fails_closed_on_grammar_dead_end() {
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
    use crate::generation::GenerateConfig;
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");

    let spec = GrammarSpec::Gbnf("root ::= \"a\" \"!\"\n".to_string());
    let engine = Arc::new(
        GrammarEngine::new(&spec, single_char_vocab_bytes())
            .expect("grammar engine builds over single-char vocab"),
    );

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 2,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: Some(engine),
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let mut on_token_calls = 0u32;
    let result = state.generate_streaming("a", &tokenizer, &gen_cfg, |_delta, _id| {
        on_token_calls += 1;
        true
    });

    assert!(
        matches!(result, Err(InferenceError::GrammarConstraintBlocked(_))),
        "a mid-document grammar dead end must fail closed via \
                 generate_streaming(); got {result:?}"
    );
    assert!(
        on_token_calls <= 1,
        "the decode-loop guard must fire before a second (bogus) token \
                 ever reaches on_token; got {on_token_calls} calls"
    );
}

/// #611: the same mid-document dead end must fail closed via
/// `generate_streaming_with_prefix_cache`, and the error path must not
/// leave a cache entry behind.
#[test]
fn generate_streaming_with_prefix_cache_decode_loop_fails_closed_on_grammar_dead_end() {
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
    use crate::generation::GenerateConfig;
    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::sync::Arc;

    let tokenizer = single_char_vocab_tokenizer();
    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;

    let spec = GrammarSpec::Gbnf("root ::= \"a\" \"!\"\n".to_string());
    let engine = Arc::new(
        GrammarEngine::new(&spec, single_char_vocab_bytes())
            .expect("grammar engine builds over single-char vocab"),
    );

    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 2,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: Some(1),
        stop_token_ids: vec![],
        enable_thinking: false,
        enable_mtp: Some(false),
        grammar: Some(engine),
        stop_strings: vec![],
        reasoning_budget: None,
        logprobs: None,
    };

    let result =
        state.generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |_, _| true);

    assert!(
        matches!(result, Err(InferenceError::GrammarConstraintBlocked(_))),
        "a mid-document grammar dead end must fail closed via \
                 generate_streaming_with_prefix_cache(); got {result:?}"
    );
    assert!(
        state.cross_turn_prefix_cache.entry.is_none(),
        "a decode-loop grammar-fail-closed turn must not leave a stale \
                 cache entry behind"
    );
}

/// The reasoning budget forces `</think>` on the first decode step, and a
/// grammar that expects another `a` rejects it. The request ends with
/// `StopReason::Grammar`, and the boundary it saves must be one live
/// KV/GDN state actually represents: the rejected step already forwarded
/// the one pushed token, so nothing may forward it a second time.
#[test]
fn generate_streaming_with_prefix_cache_budget_forced_close_grammar_rejection_saves_consistent_boundary()
 {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = metal::Device::system_default() else {
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present"
        );
        return;
    };
    let _guard = gpu_test_lock();

    use crate::grammar::{GrammarEngine, GrammarSpec};
    use std::collections::HashMap;
    use std::sync::Arc;

    const THINK_CLOSE_ID: usize = 30;
    let mut vocab_bytes = single_char_vocab_bytes();
    vocab_bytes[THINK_CLOSE_ID] = b"</think>".to_vec();
    let vocab: HashMap<String, u32> = vocab_bytes
        .iter()
        .zip(0u32..)
        .map(|(bytes, id)| (String::from_utf8_lossy(bytes).into_owned(), id))
        .collect();
    let tokenizer = crate::tokenizer::bpe::BpeTokenizer::from_vocab_and_merges(vocab, Vec::new())
        .expect("thinking vocab tokenizer build");
    let engine = Arc::new(
        GrammarEngine::new(
            &GrammarSpec::Gbnf("root ::= \"a\" \"a\"\n".to_string()),
            vocab_bytes,
        )
        .expect("grammar engine builds over the thinking vocab"),
    );
    let gen_cfg = GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 4,
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
        reasoning_budget: Some(1),
        logprobs: None,
    };

    let (cfg, weights) = tiny_hybrid_fixture();
    let mut state = MetalQwen35State::new(&weights, &cfg, 32).expect("tiny hybrid fixture");
    let slot_id = crate::kv_cache::CrossTurnSlotId::DEFAULT;

    let mut streamed = String::new();
    let turn = state
        .generate_streaming_with_prefix_cache(slot_id, "a", &tokenizer, &gen_cfg, |delta, _id| {
            streamed.push_str(delta);
            true
        })
        .expect("a budget-forced token the grammar rejects is a stop, not an error");

    assert_eq!(turn.output.stop_reason, Some(StopReason::Grammar));
    assert_eq!(
        turn.output.token_ids,
        vec![0],
        "the forced </think> is rejected before it is pushed"
    );
    assert_eq!(turn.output.text, "a");
    assert_eq!(streamed, "a");
    assert!(
        !turn.output.stopped,
        "a grammar rejection reports stopped: false, as the streaming entry does"
    );

    let prompt_len = turn.cache.prompt_tokens;
    let entry = state
        .cross_turn_prefix_cache
        .get(slot_id)
        .expect("a grammar rejection saves the boundary live state represents");
    let represented_len = entry.generic.represented_len;
    assert_eq!(
        represented_len,
        prompt_len + 1,
        "the boundary covers the prompt and the one pushed token the rejected \
                 step forwarded, and nothing beyond it"
    );
    assert_eq!(entry.generic.token_ids.len(), represented_len);
    assert_eq!(entry.generic.token_ids[prompt_len..], [0]);
    assert_eq!(entry.generic.gdn_snapshot_len, represented_len);
    assert_eq!(state.session.kv_cache.seq_len, represented_len);

    let follow_up = state
        .generate_streaming_with_prefix_cache(
            slot_id,
            &format!("a{}q", turn.output.text),
            &tokenizer,
            &cross_turn_test_gen_cfg(3, 2),
            |_, _| true,
        )
        .expect("a follow-up on the saved boundary must not error");
    assert_eq!(
        follow_up.cache.mode,
        crate::kv_cache::PrefixReuseMode::ExactAppend,
        "the saved boundary must be reusable by a turn that extends it"
    );
    assert_eq!(follow_up.cache.reused_tokens, represented_len);
}
