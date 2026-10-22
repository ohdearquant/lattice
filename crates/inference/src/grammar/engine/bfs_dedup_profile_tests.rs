use super::*;
use crate::grammar::pda::StackFrame;

fn reference_enumeration(
    grammar: &CompiledGrammar,
    vocab_bytes: &[Vec<u8>],
    max_states: usize,
) -> Result<(Vec<GrammarState>, [u64; 3]), StackLimitError> {
    let initial = initial_grammar_state(grammar);
    let mut queue: Vec<GrammarState> = vec![initial.clone()];
    let mut visited: Vec<GrammarState> = vec![initial];
    let mut head = 0;
    let mut checks = 0;
    let mut comparisons = 0;
    let mut duplicates = 0;

    while head < queue.len() && visited.len() < max_states {
        let state = queue[head].clone();
        head += 1;

        for token_bytes in vocab_bytes {
            if token_bytes.is_empty() {
                continue;
            }
            let (result, next_state) = simulate_token(&state, grammar, token_bytes);
            if result == SimResult::StackLimitExceeded {
                return Err(StackLimitError);
            }
            if result == SimResult::Accept || result == SimResult::ContextDependent {
                checks += 1;
                let mut found = false;
                for candidate in &visited {
                    comparisons += 1;
                    if candidate.stacks == next_state.stacks
                        && candidate.complete == next_state.complete
                    {
                        found = true;
                        break;
                    }
                }
                duplicates += u64::from(found);
                if !found {
                    visited.push(next_state.clone());
                    if visited.len() < max_states {
                        queue.push(next_state);
                    }
                }
            }
        }
    }

    Ok((visited, [checks, comparisons, duplicates]))
}

fn assert_state_order(actual: &[GrammarState], expected: &[GrammarState]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.stacks, expected.stacks);
        assert_eq!(actual.complete, expected.complete);
        assert_eq!(actual.partial_token_bytes, expected.partial_token_bytes);
    }
}

fn counts(profile: BfsDedupProfile) -> [u64; 3] {
    [
        profile.membership_checks,
        profile.state_comparisons,
        profile.duplicate_states,
    ]
}

fn assert_profile_matches(spec: &GrammarSpec, vocab: &[Vec<u8>], cap: usize) -> Vec<GrammarState> {
    let grammar = match spec {
        GrammarSpec::JsonSchema(schema) => compile(schema).unwrap(),
        GrammarSpec::Gbnf(gbnf) => parse_gbnf(gbnf).unwrap(),
    };
    let (expected, expected_counts) = reference_enumeration(&grammar, vocab, cap).unwrap();
    let ordinary = enumerate_grammar_states(&grammar, vocab, cap).unwrap();
    let (profiled, profile) = profile_grammar_states(&grammar, vocab, cap).unwrap();
    assert_state_order(&ordinary, &expected);
    assert_state_order(&profiled, &expected);
    assert_eq!(counts(profile), expected_counts);
    assert_eq!(
        profile.membership_checks - profile.duplicate_states,
        (expected.len() - 1) as u64
    );
    let (public_count, public_profile) = profile_reachable_states(spec, vocab, cap).unwrap();
    assert_eq!(public_count, expected.len());
    assert_eq!(counts(public_profile), expected_counts);
    assert_eq!(
        probe_reachable_states(spec, vocab, cap).unwrap(),
        expected.len()
    );
    profiled
}

#[test]
fn bfs_profile_counts_short_circuit_comparisons_and_duplicates() {
    let spec = GrammarSpec::Gbnf("root ::= \"a\"".into());
    let vocab = vec![b"a".to_vec(), b"a".to_vec(), b"b".to_vec(), Vec::new()];
    let expected = probe_reachable_states(&spec, &vocab, 256).unwrap();
    let (states, profile) = profile_reachable_states(&spec, &vocab, 256).unwrap();
    assert_eq!(states, expected);
    assert_eq!(states, 2);
    assert_eq!(profile.membership_checks, 2);
    assert_eq!(profile.state_comparisons, 3);
    assert_eq!(profile.duplicate_states, 1);
    assert_profile_matches(&spec, &vocab, 256);
}

proptest::proptest! {
    #[test]
    fn bfs_profile_preserves_state_counts_and_accounts_for_every_scan(
        tokens in proptest::collection::vec(0usize..10, 0..30),
        cap in 0usize..18,
        nested in proptest::bool::ANY,
    ) {
        let spec = GrammarSpec::Gbnf(if nested {
            "root ::= child \"z\"\nchild ::= \"a\" \"x\" | \"b\" \"y\"".into()
        } else {
            "root ::= \"a\" | \"b\"".into()
        });
        let choices: [&[u8]; 10] = [b"a", b"b", b"aa", b"bad", b"", b"x", b"y", b"z", b"axz", b"byz"];
        let vocab: Vec<Vec<u8>> = tokens.iter().map(|&index| choices[index].to_vec()).collect();
        let expected = probe_reachable_states(&spec, &vocab, cap).unwrap();
        let (states, profile) = profile_reachable_states(&spec, &vocab, cap).unwrap();
        proptest::prop_assert_eq!(states, expected);
        proptest::prop_assert!(profile.state_comparisons >= profile.membership_checks);
        proptest::prop_assert!(profile.duplicate_states <= profile.membership_checks);
        proptest::prop_assert_eq!(
            profile.membership_checks - profile.duplicate_states,
            (states - 1) as u64,
        );
        assert_profile_matches(&spec, &vocab, cap);
    }
}

#[test]
fn bfs_profile_empty_vocab_and_invalid_grammar_are_explicit() {
    let spec = GrammarSpec::Gbnf("root ::= \"a\"".into());
    let (states, profile) = profile_reachable_states(&spec, &[], 256).unwrap();
    assert_eq!(states, 1);
    assert_eq!(profile.membership_checks, 0);
    assert_eq!(profile.state_comparisons, 0);
    assert_eq!(profile.dedup_ns, 0);
    assert_eq!(profile.empty_interval_ns, 0);
    assert_eq!(profile.duplicate_states, 0);
    assert!(
        profile_reachable_states(&GrammarSpec::Gbnf("not a grammar".into()), &[], 256).is_err()
    );
}

#[test]
fn bfs_profile_preserves_order_and_cap_overshoot_for_both_grammar_formats() {
    let spec = GrammarSpec::Gbnf("root ::= \"a\" \"x\" | \"b\" \"y\" | \"c\" \"z\"".into());
    let vocab: Vec<Vec<u8>> = ["a", "b", "c", "x", "y", "z", "a", "", "bad"]
        .iter()
        .map(|token| token.as_bytes().to_vec())
        .collect();
    let states = assert_profile_matches(&spec, &vocab, 2);
    assert_eq!(
        states.len(),
        4,
        "the current expansion may overshoot the cap"
    );
    assert_profile_matches(&spec, &vocab, 256);
    let mut reordered_vocab = vocab.clone();
    reordered_vocab.swap(0, 2);
    let reordered = assert_profile_matches(&spec, &reordered_vocab, 2);
    assert_eq!(reordered.len(), states.len());
    assert_ne!(states[1].stacks, reordered[1].stacks);
    for cap in [0, 1] {
        assert_eq!(assert_profile_matches(&spec, &vocab, cap).len(), 1);
    }
    let spec = GrammarSpec::JsonSchema(serde_json::json!({"type": "boolean"}));
    let vocab: Vec<Vec<u8>> = ["t", "f", "rue", "alse", "true", "false", "true", "", "bad"]
        .iter()
        .map(|token| token.as_bytes().to_vec())
        .collect();
    for cap in [0, 1, 2, 4, 256] {
        assert_profile_matches(&spec, &vocab, cap);
    }
}

#[test]
fn bfs_profile_identity_uses_every_ordered_frame_and_completion_only() {
    let original = GrammarState {
        stacks: vec![
            vec![
                StackFrame {
                    rule_id: 1,
                    alt_idx: 2,
                    sym_pos: 3,
                },
                StackFrame {
                    rule_id: 4,
                    alt_idx: 5,
                    sym_pos: 6,
                },
            ],
            vec![
                StackFrame {
                    rule_id: 7,
                    alt_idx: 8,
                    sym_pos: 9,
                },
                StackFrame {
                    rule_id: 10,
                    alt_idx: 11,
                    sym_pos: 12,
                },
            ],
        ],
        partial_token_bytes: b"partial".to_vec(),
        complete: false,
    };
    for stack in 0..2 {
        for frame in 0..2 {
            for field in 0..3 {
                let mut changed = original.clone();
                let frame = &mut changed.stacks[stack][frame];
                match field {
                    0 => frame.rule_id += 1,
                    1 => frame.alt_idx += 1,
                    _ => frame.sym_pos += 1,
                }
                assert!(!states_equal(&original, &changed));
                assert!(!states_equal(&changed, &original));
            }
        }
    }
    let mut changed = original.clone();
    changed.complete = true;
    assert!(!states_equal(&original, &changed));
    let mut changed = original.clone();
    changed.stacks.swap(0, 1);
    assert!(!states_equal(&original, &changed));
    let mut changed = original.clone();
    changed.stacks[0].swap(0, 1);
    assert!(!states_equal(&original, &changed));
    let mut changed = original.clone();
    changed.stacks.pop();
    assert!(!states_equal(&original, &changed));
    let mut changed = original.clone();
    changed.partial_token_bytes = b"other".to_vec();
    assert!(states_equal(&original, &changed));
    assert!(states_equal(&changed, &original));
}

#[test]
fn bfs_profile_preserves_stack_limit_errors() {
    let spec = GrammarSpec::Gbnf("root ::= root".into());
    let grammar = parse_gbnf("root ::= root").unwrap();
    let vocab = vec![b"a".to_vec()];
    assert!(reference_enumeration(&grammar, &vocab, 256).is_err());
    assert!(enumerate_grammar_states(&grammar, &vocab, 256).is_err());
    assert!(profile_grammar_states(&grammar, &vocab, 256).is_err());
    let expected = probe_reachable_states(&spec, &vocab, 256).unwrap_err();
    let actual = profile_reachable_states(&spec, &vocab, 256).unwrap_err();
    assert_eq!(actual.to_string(), expected.to_string());
    assert!(actual.to_string().contains("too deeply nested"));
}
