//! Criterion coverage for the grammar mask hot path.
//!
//! The synthetic vocabulary makes different multi-byte tokens
//! context-dependent in different PDA states. This isolates the cost of the
//! runtime recheck set without model files or tokenizer fixtures.

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use lattice_inference::grammar::engine::{
    context_recheck_candidates, context_recheck_simulated, enable_mask_profiling, take_mask_profile,
};
use lattice_inference::grammar::pda::GrammarState;
use lattice_inference::grammar::{GrammarEngine, GrammarSpec};
use std::time::Duration;

const STATE_COUNT: usize = 64;
const CONTEXT_TOKEN_COUNT: usize = 16_384;

fn fixture() -> (GrammarEngine, GrammarState, usize) {
    let state_bytes: Vec<u8> = (b'!'..=b'~')
        .filter(|byte| !matches!(byte, b'"' | b'\\'))
        .take(STATE_COUNT)
        .collect();
    assert_eq!(state_bytes.len(), STATE_COUNT);

    let literal = String::from_utf8(state_bytes.clone()).expect("fixture is printable ASCII");
    let spec = GrammarSpec::Gbnf(format!("root ::= \"{literal}\"\n"));

    let mut vocab = Vec::with_capacity(STATE_COUNT + CONTEXT_TOKEN_COUNT);
    vocab.extend(state_bytes.iter().copied().map(|byte| vec![byte]));
    for token_id in 0..CONTEXT_TOKEN_COUNT {
        vocab.push(vec![state_bytes[token_id % STATE_COUNT], 0]);
    }
    let vocab_size = vocab.len();
    let engine = GrammarEngine::new(&spec, vocab).expect("fixture grammar must compile");
    assert!(
        !engine.exceeds_state_budget(),
        "fixture must exercise the precomputed mask path"
    );

    let mut state = engine.initial_state();
    let measured_state = STATE_COUNT / 2;
    for token_id in 0..measured_state {
        assert!(
            engine.advance(&mut state, token_id as u32),
            "single-byte fixture token must advance the PDA"
        );
    }

    enable_mask_profiling();
    let mut probe_logits = vec![0.0; vocab_size];
    engine
        .mask_logits(&mut state, &mut probe_logits)
        .expect("fixture logits match the vocabulary");
    let profile = take_mask_profile();
    assert_eq!(profile.precomputed_calls, 1);
    assert_eq!(profile.context_recheck_calls, 1);
    assert_eq!(profile.fallback_calls, 0);
    // The context-dependent vocabulary is spread evenly across STATE_COUNT
    // literal-byte positions (each position's literal byte prefixes
    // CONTEXT_TOKEN_COUNT / STATE_COUNT of the two-byte tokens), and only the
    // group matching the queried state clears the precomputed bitmask to
    // reach `simulate_token`.
    // A precomputed-mask regression that pre-blocks (or never routes)
    // candidates into the runtime recheck loop collapses this to 0 while
    // `context_recheck_calls` above stays 1, since that counter only proves
    // the loop *ran*, not that it did work.
    let expected_simulated = (CONTEXT_TOKEN_COUNT / STATE_COUNT) as u64;
    assert_eq!(context_recheck_simulated(), expected_simulated);
    // `context_recheck_simulated` alone does not discriminate the candidate
    // *set* the loop iterated: every non-matching-group candidate is already
    // blocked by the precomputed bitmask before it can reach
    // `simulate_token`, so it reads `expected_simulated` whether the loop
    // walked this state's local group or the global union across all
    // states. `context_recheck_candidates` counts loop entries instead
    // (before that bitmask short-circuit), so it is sized by which set was
    // iterated: this state's local group is exactly
    // `CONTEXT_TOKEN_COUNT / STATE_COUNT` candidates (state-local routing,
    // what this PR implements), whereas the global union is all
    // `CONTEXT_TOKEN_COUNT` context-dependent candidates across every state
    // (the pre-PR fallback). A regression that routes the loop back onto
    // the global union changes this count from `expected_simulated` to
    // `CONTEXT_TOKEN_COUNT` while `context_recheck_simulated` stays flat.
    let expected_candidates = expected_simulated;
    assert_eq!(context_recheck_candidates(), expected_candidates);
    assert_eq!(
        probe_logits
            .iter()
            .filter(|logit| logit.is_finite())
            .count(),
        1,
        "runtime rechecks must reject every partial multi-byte token"
    );
    assert!(probe_logits[measured_state].is_finite());

    (engine, state, vocab_size)
}

fn bench_context_rechecks(c: &mut Criterion) {
    let (engine, state, vocab_size) = fixture();
    let mut group = c.benchmark_group("grammar_mask");
    group.throughput(Throughput::Elements(vocab_size as u64));
    group.bench_function("state_sparse_context_rechecks", |b| {
        b.iter_batched_ref(
            || (state.clone(), vec![0.0; vocab_size]),
            |(state, logits)| {
                engine
                    .mask_logits(black_box(state), black_box(logits.as_mut_slice()))
                    .expect("fixture logits match the vocabulary");
                black_box(logits.as_slice());
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .sample_size(60);
    targets = bench_context_rechecks
}
criterion_main!(benches);
