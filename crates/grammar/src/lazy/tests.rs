use super::*;
use crate::engine::GrammarEngine;

fn fixture(budget: usize) -> (GrammarEngine, LazyGrammarEngine) {
    let spec = GrammarSpec::Gbnf("root ::= \"ab\" | \"ac\"\n".into());
    let vocab = vec![
        vec![],
        b"a".to_vec(),
        b"b".to_vec(),
        b"c".to_vec(),
        b"ab".to_vec(),
        b"z".to_vec(),
    ];
    (
        GrammarEngine::new(&spec, vocab.clone()).unwrap(),
        LazyGrammarEngine::new(&spec, vocab, budget).unwrap(),
    )
}

#[test]
fn lazy_masks_match_eager_and_oracle_across_advance_and_hits() {
    let (eager, lazy) = fixture(4096);
    let mut state = lazy.initial_state();
    for token in [Some(1), Some(2), None] {
        let initial = vec![7.0, f32::NEG_INFINITY, 3.0, 4.0, 5.0, 6.0, 9.0];
        let mut expected = initial.clone();
        let mut eager_state = state.clone();
        eager.mask_logits(&mut eager_state, &mut expected).unwrap();
        for _ in 0..2 {
            let mut actual = initial.clone();
            lazy.mask_logits(&state, &mut actual).unwrap();
            assert_eq!(actual, expected);
        }
        let mut oracle = initial.clone();
        eager.mask_by_simulation(&state, &mut oracle).unwrap();
        assert_eq!(expected, oracle);
        if let Some(token) = token {
            assert!(lazy.advance(&mut state, token).unwrap());
        }
    }
    assert!(!lazy.cache.lock().unwrap().entries.is_empty());
}

#[test]
fn cache_budget_eviction_and_oversize_preserve_exact_masks() {
    let (eager, lazy) = fixture(0);
    let state = lazy.initial_state();
    let mut logits = vec![0.0; 6];
    lazy.mask_logits(&state, &mut logits).unwrap();
    assert!(lazy.cache.lock().unwrap().entries.is_empty());
    let mut oracle = vec![0.0; 6];
    eager.mask_by_simulation(&state, &mut oracle).unwrap();
    assert_eq!(logits, oracle);

    let (_, lazy) = fixture(0);
    let mut later = state.clone();
    assert!(lazy.advance(&mut later, 1).unwrap());
    let bytes = Cache::entry_bytes(&state, 1)
        .unwrap()
        .max(Cache::entry_bytes(&later, 1).unwrap());
    let budget = size_of::<Entry>() + bytes;
    *lazy.cache.lock().unwrap() = Cache {
        entries: Vec::with_capacity(1),
        retained: size_of::<Entry>(),
        budget,
    };
    for current in [&state, &later, &state] {
        let mut actual = vec![0.0; 6];
        lazy.mask_logits(current, &mut actual).unwrap();
        let mut expected = vec![0.0; 6];
        eager.mask_by_simulation(current, &mut expected).unwrap();
        assert_eq!(actual, expected);
        let cache = lazy.cache.lock().unwrap();
        assert!(cache.retained <= budget);
        assert_eq!(cache.entries.len(), 1);
        assert!(cache.entries[0].key.matches(current));
    }
}

#[test]
fn full_ordered_stack_and_completion_are_cache_identity() {
    let (_, lazy) = fixture(4096);
    let state = lazy.initial_state();
    let mut changed = state.clone();
    changed.complete = !state.complete;
    let mut other = state.clone();
    other.stacks[0][0].sym_pos += 1;
    let mut cache = lazy.cache.lock().unwrap();
    cache.insert(&state, vec![1]);
    assert!(cache.entries[0].key.matches(&state));
    assert!(!cache.entries[0].key.matches(&changed));
    assert!(!cache.entries[0].key.matches(&other));
    changed = state.clone();
    changed.partial_token_bytes.push(42);
    assert!(cache.entries[0].key.matches(&changed));
    let mut ordered = state.clone();
    ordered.stacks.push(other.stacks[0].clone());
    cache.insert(&ordered, vec![1]);
    ordered.stacks.swap(0, 1);
    assert!(!cache.entries[1].key.matches(&ordered));
}

#[test]
fn short_logits_do_not_change_and_engines_do_not_share_cache() {
    let (_, lazy) = fixture(4096);
    let (_, other) = fixture(4096);
    let state = lazy.initial_state();
    let mut short = vec![9.0; 5];
    assert!(lazy.mask_logits(&state, &mut short).is_err());
    assert_eq!(short, vec![9.0; 5]);
    let mut logits = vec![0.0; 6];
    lazy.mask_logits(&state, &mut logits).unwrap();
    assert_eq!(lazy.cache.lock().unwrap().entries.len(), 1);
    assert_eq!(other.cache.lock().unwrap().entries.len(), 0);
}

#[test]
fn concurrent_mask_requests_match_oracle() {
    let (eager, lazy) = fixture(4096);
    let state = lazy.initial_state();
    let mut expected = vec![0.0; 6];
    eager.mask_by_simulation(&state, &mut expected).unwrap();
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let lazy = &lazy;
            let state = &state;
            let expected = &expected;
            scope.spawn(move || {
                for _ in 0..5 {
                    let mut actual = vec![0.0; 6];
                    lazy.mask_logits(state, &mut actual).unwrap();
                    assert_eq!(&actual, expected);
                }
            });
        }
    });
    let cache = lazy.cache.lock().unwrap();
    assert_eq!(cache.entries.len(), 1);
    assert!(cache.retained <= cache.budget);
}

#[test]
fn matcher_limit_does_not_cache_or_change_logits() {
    use crate::pda::{MAX_LIVE_STACKS, Rule, Symbol};
    let (_, mut lazy) = fixture(4096);
    let mut alts = vec![Vec::new()];
    alts.extend(vec![
        vec![Symbol::Terminal(b'a'), Symbol::Terminal(b'b')];
        MAX_LIVE_STACKS + 1
    ]);
    lazy.grammar = CompiledGrammar {
        rules: vec![Rule {
            name: "root".into(),
            alts,
        }],
    };
    let state = lazy.initial_state();
    let mut logits = vec![3.0; 6];
    assert!(lazy.mask_logits(&state, &mut logits).is_err());
    assert_eq!(logits, vec![3.0; 6]);
    assert!(lazy.cache.lock().unwrap().entries.is_empty());
}

#[test]
fn oversized_entry_with_available_slot_stays_uncached() {
    let (eager, lazy) = fixture(0);
    let state = lazy.initial_state();
    let fixed = size_of::<Entry>();
    let bytes = Cache::entry_bytes(&state, 1).unwrap();
    assert!(bytes > 0);
    let budget = fixed + bytes - 1;
    *lazy.cache.lock().unwrap() = Cache {
        entries: Vec::with_capacity(1),
        retained: fixed,
        budget,
    };

    let mut expected = vec![0.0; 6];
    eager.mask_by_simulation(&state, &mut expected).unwrap();
    let mut actual = vec![0.0; 6];
    lazy.mask_logits(&state, &mut actual).unwrap();
    assert_eq!(actual, expected);
    let cache = lazy.cache.lock().unwrap();
    assert_eq!(cache.entries.capacity(), 1);
    assert!(cache.entries.is_empty());
    assert_eq!(cache.retained, fixed);
    assert!(cache.retained <= cache.budget);
}
