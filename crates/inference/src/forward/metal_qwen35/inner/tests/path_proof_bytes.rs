//! `PathProofCounters::reset()` coverage, asserted directly.
//!
//! The failure this guards is a field added to the struct and missed in
//! `reset()`. It does not look like a bug: the counter simply carries its value
//! into the next probe, so it surfaces as one test reading another test's
//! traffic, which reads as a flake and gets re-run rather than investigated.
//!
//! A hand-written list of field names would inherit that same defect for the
//! next field added, so the list is pinned to the struct by size. Every field is
//! an `AtomicU64`, so the struct's size divided by 8 is its field count; if a
//! seventeenth field appears, the first assertion fails and names what to do.
//! That is the whole point of writing it this way rather than as sixteen
//! `assert_eq!(.., 0)` lines that would still pass on the day the defect lands.

use std::sync::atomic::{AtomicU64, Ordering};

use super::super::PathProofCounters;
// The parent test module's own imports and fixtures (the tiny Metal fixture,
// the GPU lock, the sampling-route helpers) reach this file the same way
// they reach `dispatch.rs`.
use super::*;

/// A field's name and a borrow of it, so the assertions below can report which
/// field failed rather than which index did.
type Accessor = (
    &'static str,
    for<'a> fn(&'a PathProofCounters) -> &'a AtomicU64,
);

/// One accessor per field, in declaration order. Kept beside the size assertion
/// below, which is what makes the list complete rather than merely plausible.
const ACCESSORS: [Accessor; 16] = [
    ("prefill_kv_batch", |c| &c.prefill_kv_batch),
    ("prefill_attn_batched", |c| &c.prefill_attn_batched),
    ("prefill_hidden_readback", |c| &c.prefill_hidden_readback),
    ("prefill_full_vocab_logit_readback", |c| {
        &c.prefill_full_vocab_logit_readback
    }),
    ("prefill_compact_candidate_logit_readback", |c| {
        &c.prefill_compact_candidate_logit_readback
    }),
    ("decode_kv_copy", |c| &c.decode_kv_copy),
    ("decode_attn_direct", |c| &c.decode_attn_direct),
    ("decode_attn_split_partial", |c| {
        &c.decode_attn_split_partial
    }),
    ("decode_attn_split_reduce", |c| &c.decode_attn_split_reduce),
    ("decode_hidden_readback", |c| &c.decode_hidden_readback),
    ("decode_full_vocab_logit_readback", |c| {
        &c.decode_full_vocab_logit_readback
    }),
    ("decode_compact_candidate_logit_readback", |c| {
        &c.decode_compact_candidate_logit_readback
    }),
    ("prefill_full_vocab_logit_readback_bytes", |c| {
        &c.prefill_full_vocab_logit_readback_bytes
    }),
    ("prefill_compact_candidate_logit_readback_bytes", |c| {
        &c.prefill_compact_candidate_logit_readback_bytes
    }),
    ("decode_full_vocab_logit_readback_bytes", |c| {
        &c.decode_full_vocab_logit_readback_bytes
    }),
    ("decode_compact_candidate_logit_readback_bytes", |c| {
        &c.decode_compact_candidate_logit_readback_bytes
    }),
];

#[test]
fn path_proof_counters_reset_clears_every_field() {
    assert_eq!(
        std::mem::size_of::<AtomicU64>(),
        8,
        "the field-count derivation below assumes an 8-byte AtomicU64"
    );
    assert_eq!(
        std::mem::size_of::<PathProofCounters>(),
        ACCESSORS.len() * std::mem::size_of::<AtomicU64>(),
        "PathProofCounters has {} AtomicU64-sized fields but ACCESSORS lists {}. \
         A field was added: add it to PathProofCounters::reset() and to ACCESSORS. \
         (If the new field is not an AtomicU64, this derivation no longer holds \
         and the assertion needs rewriting rather than adjusting.)",
        std::mem::size_of::<PathProofCounters>() / std::mem::size_of::<AtomicU64>(),
        ACCESSORS.len()
    );

    let counters = PathProofCounters::default();

    // Distinct values, so a reset() line that clears the wrong field twice and
    // misses its neighbour is still caught by the neighbour's own assertion.
    for (i, (_, get)) in ACCESSORS.iter().enumerate() {
        get(&counters).store(i as u64 + 1, Ordering::Relaxed);
    }

    // Control: the post-condition below is vacuous if the loop above wrote
    // nothing, and "every field is zero" is exactly what a no-op loop produces.
    for (i, (name, get)) in ACCESSORS.iter().enumerate() {
        assert_eq!(
            get(&counters).load(Ordering::Relaxed),
            i as u64 + 1,
            "{name} did not take the value this test wrote, so the reset \
             assertion below would pass without testing anything"
        );
    }

    counters.reset();

    for (name, get) in ACCESSORS.iter() {
        assert_eq!(
            get(&counters).load(Ordering::Relaxed),
            0,
            "PathProofCounters::reset() left {name} set; it is missing a store"
        );
    }
}

/// The byte counters' reason for existing, made executable rather than argued.
///
/// Both arms below perform exactly one decode logit readback, so both report an
/// event count of 1. They differ only in which field that 1 lands in, and a
/// probe that asserts the *shape* of the readback is therefore satisfied by
/// either one. The transfer sizes are what separate them, and at the production
/// vocabulary (151936) with `local_k = 64` the separation is a factor near 1187.
/// The fixture's vocabulary is tiny, so this test asserts the exact byte
/// expressions instead of that ratio: the ratio is a property of the
/// expressions, not of the fixture, and pinning it here would make the test
/// depend on a fixture dimension it has no reason to care about.
///
/// What this covers, stated because the instrument is wider than the test: the
/// ordinary decode readback. `forward_step_inner_impl_dispatch` carries a second
/// pair of readback sites under `LATTICE_DECODE_PROFILE`, and the prefill sites
/// are a third pair; all are instrumented, none are exercised here, because the
/// profiling variant is selected by a process-global environment variable that a
/// test sharing this binary must not set. A mutation aimed at the profiling site
/// leaves this test green, which is a statement about its reach and not about the
/// counters.
///
/// The substitution is reached through a supported public knob rather than an
/// edited line: `logprobs: Some(_)` forces the full-logit path at
/// `plan_sampling_route`, because compact sampling cannot provide full-logit
/// logprob semantics. That makes this a standing regression test — if the
/// logprobs request ever silently takes the compact route, its readback
/// collapses from `vocab_size * 4` to `k * 8` bytes and this reddens — rather
/// than a one-off mutation that is gone once the diff is restored.
#[test]
fn compact_and_full_logit_readback_differ_in_bytes_while_both_count_one() {
    let enforce = std::env::var_os("LATTICE_METAL_TEST_ENFORCE").is_some();
    let Some(_) = Device::system_default() else {
        eprintln!(
            "[METAL_TEST_SKIP] context=compact_and_full_logit_readback_differ_in_bytes_while_both_count_one \
             reason=no_metal_device"
        );
        assert!(
            !enforce,
            "LATTICE_METAL_TEST_ENFORCE=1 but no Metal device present \
             (compact_and_full_logit_readback_differ_in_bytes_while_both_count_one)"
        );
        return;
    };
    let _gpu = gpu_test_lock();

    // One of the precompiled `LM_HEAD_LOCAL_KS` Stage-1 variants, so the compact
    // arm actually resolves a block-top-k route instead of falling back.
    const LOCAL_K: usize = 8;
    let base = || GenerateConfig {
        min_p: 0.0,
        max_new_tokens: 1,
        top_k: LOCAL_K,
        // top_p < 1.0 requires the approximate-top-p opt-in, which this test
        // does not set; an exact-nucleus request would route to CpuFallback.
        top_p: 1.0,
        repetition_penalty: 1.0,
        grammar: None,
        logprobs: None,
        ..GenerateConfig::default()
    };

    let environment = SamplingRouteEnvironment {
        compact: true,
        selection: false,
        approximate_top_p: false,
    };
    // The environment is passed rather than read from the process:
    // `SamplingRouteEnvironment::current()` reads `LATTICE_COMPACT_TOPK` out of
    // the process environment, and a test that set it would be setting it for
    // every other test sharing the binary.
    let compact_plan = plan_sampling_route(&base(), true, environment);
    let exact_plan = plan_sampling_route(
        &GenerateConfig {
            min_p: 0.0,
            logprobs: Some(1),
            ..base()
        },
        true,
        environment,
    );
    assert!(
        compact_plan.use_compact,
        "the compact arm must engage the compact route, or it is measuring the \
         same path as the other arm and the comparison below is vacuous"
    );
    assert_eq!(compact_plan.compact_topk, LOCAL_K);
    assert!(
        !exact_plan.use_compact,
        "logprobs: Some(_) must force the full-logit path"
    );
    assert_eq!(exact_plan.compact_topk, 0);

    let (cfg, weights) = tiny_metal_qwen35_fixture();
    let run = |plan| {
        let mut state = MetalQwen35State::new(&weights, &cfg, 16)
            .expect("tiny MetalQwen35State fixture constructs");
        state.path_proof_enabled = true;
        state.reset_path_proof_counters();
        apply_sampling_route_plan(
            plan,
            &mut state.session.compact_route,
            &mut state.session.compact_topk,
            &mut state.session.compact_result,
        );
        let _ = state.forward_step(42, 0);
        state.logit_readback_path_proof_snapshot()
    };
    let compact = run(compact_plan);
    let exact = run(exact_plan);

    // The counts. Both are 1, which is the point.
    assert_eq!(
        compact.decode_compact_candidate, 1,
        "compact arm: {compact:?}"
    );
    assert_eq!(compact.decode_full_vocab, 0, "compact arm: {compact:?}");
    assert_eq!(exact.decode_full_vocab, 1, "full-logit arm: {exact:?}");
    assert_eq!(
        exact.decode_compact_candidate, 0,
        "full-logit arm: {exact:?}"
    );

    // The bytes. These are what the counts above cannot say.
    assert_eq!(
        compact.decode_compact_candidate_bytes,
        (LOCAL_K * std::mem::size_of::<GpuCandidate>()) as u64,
        "compact readback must move k candidates: {compact:?}"
    );
    assert_eq!(
        compact.decode_full_vocab_bytes, 0,
        "compact arm: {compact:?}"
    );
    assert_eq!(
        exact.decode_full_vocab_bytes,
        (cfg.vocab_size * std::mem::size_of::<f32>()) as u64,
        "full-logit readback must move the whole vocabulary: {exact:?}"
    );
    assert_eq!(
        exact.decode_compact_candidate_bytes, 0,
        "full-logit arm: {exact:?}"
    );
    assert!(
        exact.decode_full_vocab_bytes > compact.decode_compact_candidate_bytes,
        "the substitution must be visible as a larger transfer: {exact:?} vs {compact:?}"
    );
}
