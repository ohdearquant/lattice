use super::*;

/// Build a grammar that matches exactly `b"ab"`.
fn ab_grammar() -> CompiledGrammar {
    let mut b = GrammarBuilder::new();
    b.add_rule(
        "root",
        vec![vec![Symbol::Terminal(b'a'), Symbol::Terminal(b'b')]],
    );
    b.build()
}

/// Grammar: root = 'a' | 'b'
fn or_grammar() -> CompiledGrammar {
    let mut b = GrammarBuilder::new();
    b.add_rule(
        "root",
        vec![vec![Symbol::Terminal(b'a')], vec![Symbol::Terminal(b'b')]],
    );
    b.build()
}

/// Grammar: root = digit+  where digit = '0' | '1' | ... | '9'
fn digits_grammar() -> CompiledGrammar {
    let mut b = GrammarBuilder::new();
    let digit_id = b.reserve("digit");
    let digit_alts: Vec<Alt> = (b'0'..=b'9')
        .map(|byte| vec![Symbol::Terminal(byte)])
        .collect();
    b.set_alts(digit_id, digit_alts).unwrap();

    // root = digit digit_rest
    // digit_rest = digit digit_rest | ε  (implemented as digit_rest = [empty alt])
    let rest_id = b.reserve("digit_rest");
    b.set_alts(
        rest_id,
        vec![
            vec![Symbol::NonTerminal(digit_id), Symbol::NonTerminal(rest_id)],
            vec![], // epsilon
        ],
    )
    .unwrap();

    let root_id = b.reserve("root");
    b.set_alts(
        root_id,
        vec![vec![
            Symbol::NonTerminal(digit_id),
            Symbol::NonTerminal(rest_id),
        ]],
    )
    .unwrap();
    // Ensure root is at index 0.
    let mut grammar = b.build();
    // Swap root to position 0.
    let root_pos = grammar.rules.iter().position(|r| r.name == "root").unwrap();
    grammar.rules.swap(0, root_pos);
    // Fix up any NonTerminal references after the swap.
    let orig_root_id = root_pos;
    let swapped_to_id = 0usize;
    if orig_root_id != 0 {
        for rule in &mut grammar.rules {
            for alt in &mut rule.alts {
                for sym in alt.iter_mut() {
                    if let Symbol::NonTerminal(rid) = sym {
                        if *rid == orig_root_id {
                            *rid = swapped_to_id;
                        } else if *rid == 0 {
                            *rid = orig_root_id;
                        }
                    }
                }
            }
        }
    }
    grammar
}

/// Minimal grammar isolating the trailing-comma class, free of the
/// JSON-schema compiler:
///   root = '[' body ']'
///   body = elem tail | ε
///   tail = ',' elem tail | ε
///   elem = 'x'
/// A trailing comma (`[x,]`) must reject: once the `,` is consumed only the
/// parse that took `tail`'s `',' elem tail` alternative is alive, and it
/// needs an `elem` next, not `]`. The ε arm of `tail` belongs to another
/// stack that never saw the `,`.  The nullable `body`/`tail` ε arms must
/// still let valid forms through (refs #353).
fn comma_list_grammar() -> CompiledGrammar {
    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root"); // index 0
    let body_id = b.reserve("body");
    let tail_id = b.reserve("tail");
    let elem_id = b.reserve("elem");
    b.set_alts(elem_id, vec![vec![Symbol::Terminal(b'x')]])
        .unwrap();
    b.set_alts(
        tail_id,
        vec![
            vec![
                Symbol::Terminal(b','),
                Symbol::NonTerminal(elem_id),
                Symbol::NonTerminal(tail_id),
            ],
            vec![], // epsilon
        ],
    )
    .unwrap();
    b.set_alts(
        body_id,
        vec![
            vec![Symbol::NonTerminal(elem_id), Symbol::NonTerminal(tail_id)],
            vec![], // epsilon
        ],
    )
    .unwrap();
    b.set_alts(
        root_id,
        vec![vec![
            Symbol::Terminal(b'['),
            Symbol::NonTerminal(body_id),
            Symbol::Terminal(b']'),
        ]],
    )
    .unwrap();
    b.build()
}

fn accepts_str(g: &CompiledGrammar, input: &[u8]) -> bool {
    let state = GrammarState::initial();
    let (result, final_state) = simulate_token(&state, g, input);
    result == SimResult::Accept && final_state.is_complete()
}

#[test]
fn comma_list_rejects_trailing_comma() {
    let g = comma_list_grammar();
    assert!(accepts_str(&g, b"[]")); // nullable body reaches ε, no byte consumed
    assert!(accepts_str(&g, b"[x]")); // clean tail reaches ε
    assert!(accepts_str(&g, b"[x,x]")); // nested tail
    assert!(!accepts_str(&g, b"[x,]")); // trailing comma: the ε stack never saw the ','
    assert!(!accepts_str(&g, b"[x,x,]")); // same, one level deeper
    assert!(!accepts_str(&g, b"[,]")); // leading comma
}

/// Grammar: root = "ab" | "ac" (two alternatives with one shared first byte).
fn shared_first_byte_grammar() -> CompiledGrammar {
    let mut b = GrammarBuilder::new();
    b.add_rule(
        "root",
        vec![
            vec![Symbol::Terminal(b'a'), Symbol::Terminal(b'b')],
            vec![Symbol::Terminal(b'a'), Symbol::Terminal(b'c')],
        ],
    );
    b.build()
}

#[test]
fn alternatives_sharing_a_first_byte_both_stay_alive() {
    let g = shared_first_byte_grammar();
    for follower in [b'b', b'c'] {
        let mut state = GrammarState::initial();
        assert_eq!(advance_byte(&mut state, &g, b'a'), StepResult::Accepted);
        assert_eq!(
            state.stacks.len(),
            2,
            "one stack per alternative that took 'a'"
        );
        assert!(
            state.stacks.windows(2).all(|pair| pair[0] < pair[1]),
            "stacks must be sorted and free of duplicates"
        );
        assert!(!state.is_complete());
        assert_eq!(
            advance_byte(&mut state, &g, follower),
            StepResult::Accepted,
            "follower {:?}",
            follower as char
        );
        assert!(state.is_complete());
    }

    let mut state = GrammarState::initial();
    assert_eq!(advance_byte(&mut state, &g, b'a'), StepResult::Accepted);
    let before = state.clone();
    assert_eq!(advance_byte(&mut state, &g, b'd'), StepResult::Rejected);
    assert_eq!(state.stacks, before.stacks);
}

/// `root ::= "ab" | "a" tail`, `tail ::= "x" | ""`. After "a" the first
/// stack (alternative 0) still needs "b" while the second can finish
/// through the nullable `tail`, so completeness has to look past the first
/// stack.
#[test]
fn completion_is_judged_over_every_live_stack() {
    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root");
    let tail_id = b.reserve("tail");
    b.set_alts(tail_id, vec![vec![Symbol::Terminal(b'x')], vec![]])
        .unwrap();
    b.set_alts(
        root_id,
        vec![
            vec![Symbol::Terminal(b'a'), Symbol::Terminal(b'b')],
            vec![Symbol::Terminal(b'a'), Symbol::NonTerminal(tail_id)],
        ],
    )
    .unwrap();
    let g = b.build();

    let mut state = GrammarState::initial();
    assert_eq!(advance_byte(&mut state, &g, b'a'), StepResult::Accepted);
    assert_eq!(state.stacks.len(), 2);
    assert!(state.is_complete(), "\"a\" completes through `tail`");
    assert!(accepts_str(&g, b"ab"));
    assert!(accepts_str(&g, b"ax"));
    assert!(!accepts_str(&g, b"abx"));
}

/// `root ::= x root | ""`, `x ::= "a" | "a"`. The two `x` alternatives lead
/// to the same parse, so without merging identical stacks the live set
/// would double on every byte.
#[test]
fn identical_stacks_are_merged_after_every_byte() {
    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root");
    let x_id = b.reserve("x");
    b.set_alts(
        x_id,
        vec![vec![Symbol::Terminal(b'a')], vec![Symbol::Terminal(b'a')]],
    )
    .unwrap();
    b.set_alts(
        root_id,
        vec![
            vec![Symbol::NonTerminal(x_id), Symbol::NonTerminal(root_id)],
            vec![],
        ],
    )
    .unwrap();
    let g = b.build();

    let mut state = GrammarState::initial();
    for i in 0..40 {
        assert_eq!(
            advance_byte(&mut state, &g, b'a'),
            StepResult::Accepted,
            "byte {i}"
        );
        assert_eq!(
            state.stacks.len(),
            1,
            "byte {i}: identical stacks must collapse into one"
        );
    }
    assert!(state.is_complete());
}

/// `n` alternatives that all start with "a" and differ only in their index,
/// so each one is a distinct live stack after that byte.
fn many_distinct_alternatives_grammar(n: usize) -> CompiledGrammar {
    CompiledGrammar {
        rules: vec![Rule {
            name: "root".to_string(),
            alts: vec![vec![Symbol::Terminal(b'a'), Symbol::Terminal(b'b')]; n],
        }],
    }
}

#[test]
fn live_stack_limit_is_inclusive_and_reported_distinctly() {
    let at_limit = many_distinct_alternatives_grammar(MAX_LIVE_STACKS);
    let mut state = GrammarState::initial();
    assert_eq!(
        advance_byte(&mut state, &at_limit, b'a'),
        StepResult::Accepted
    );
    assert_eq!(state.stacks.len(), MAX_LIVE_STACKS);

    let past_limit = many_distinct_alternatives_grammar(MAX_LIVE_STACKS + 1);
    let mut state = GrammarState::initial();
    let before = state.clone();
    assert_eq!(
        advance_byte(&mut state, &past_limit, b'a'),
        StepResult::StackLimitExceeded
    );
    assert_eq!(state.stacks, before.stacks);
    assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
    assert_eq!(state.complete, before.complete);

    // The token-level classification keeps the same distinction instead of
    // folding the limit into a rejection.
    let (result, _) = simulate_token(&GrammarState::initial(), &past_limit, b"ab");
    assert_eq!(result, SimResult::StackLimitExceeded);

    // A byte that no alternative takes is still an ordinary rejection.
    let mut state = GrammarState::initial();
    assert_eq!(
        advance_byte(&mut state, &past_limit, b'z'),
        StepResult::Rejected
    );
}

/// `root ::= root "a" | "b"` is left-recursive with a base case: every
/// level of expansion yields one more distinct stack, so the live-stack
/// limit reports it instead of the byte looking rejected.
#[test]
fn left_recursive_grammar_with_a_base_case_reports_the_stack_limit() {
    let grammar = CompiledGrammar {
        rules: vec![Rule {
            name: "root".to_string(),
            alts: vec![
                vec![Symbol::NonTerminal(0), Symbol::Terminal(b'a')],
                vec![Symbol::Terminal(b'b')],
            ],
        }],
    };
    let mut state = GrammarState::initial();
    assert_eq!(
        advance_byte(&mut state, &grammar, b'b'),
        StepResult::StackLimitExceeded
    );
}

/// `root ::= "a" E` with `E` a rule that has no alternatives (it matches
/// the empty string): the completion analysis must treat `E` as nullable,
/// as execution does when it steps past the reference.
#[test]
fn referenced_rule_without_alternatives_is_nullable() {
    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root");
    let empty_id = b.reserve("E");
    b.set_alts(
        root_id,
        vec![vec![Symbol::Terminal(b'a'), Symbol::NonTerminal(empty_id)]],
    )
    .unwrap();
    let g = b.build();

    let mut state = GrammarState::initial();
    assert_eq!(advance_byte(&mut state, &g, b'a'), StepResult::Accepted);
    assert!(
        state.is_complete(),
        "\"a\" completes through the empty rule"
    );
    assert!(accepts_str(&g, b"a"));
    assert!(!accepts_str(&g, b"ab"));

    // The same rule reached before any byte: `root ::= E`.
    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root");
    let empty_id = b.reserve("E");
    b.set_alts(root_id, vec![vec![Symbol::NonTerminal(empty_id)]])
        .unwrap();
    assert!(initial_grammar_state(&b.build()).is_complete());
}

/// `GrammarState::initial()` takes no grammar, so `complete` starts `false`
/// even when the root accepts the empty string.
#[test]
fn initial_state_does_not_consult_the_grammar() {
    let mut b = GrammarBuilder::new();
    b.add_rule("root", vec![vec![]]);
    let g = b.build();

    assert!(!GrammarState::initial().is_complete());
    // The grammar-aware initializer sees the same root as nullable.
    assert!(initial_grammar_state(&g).is_complete());
}

/// `root ::= ""`: the empty input is in the language, so the grammar-aware
/// constructor reports completion before any byte.
#[test]
fn for_grammar_reports_an_empty_root_complete_before_any_byte() {
    let mut b = GrammarBuilder::new();
    b.add_rule("root", vec![vec![]]);
    let g = b.build();

    assert!(GrammarState::for_grammar(&g).is_complete());
}

/// `root ::= "a" | ""` and `root ::= E` (with `E ::= ""`) are nullable
/// without the root's first alternative being empty.
#[test]
fn for_grammar_reports_an_optional_and_a_delegating_root_complete() {
    let mut b = GrammarBuilder::new();
    b.add_rule("root", vec![vec![Symbol::Terminal(b'a')], vec![]]);
    assert!(GrammarState::for_grammar(&b.build()).is_complete());

    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root");
    let empty_id = b.reserve("E");
    b.set_alts(root_id, vec![vec![Symbol::NonTerminal(empty_id)]])
        .unwrap();
    b.set_alts(empty_id, vec![vec![]]).unwrap();
    assert!(GrammarState::for_grammar(&b.build()).is_complete());
}

/// Control: a root that needs at least one byte is not complete before any
/// byte, whichever constructor builds the state.
#[test]
fn for_grammar_reports_a_non_nullable_root_incomplete() {
    for g in [ab_grammar(), or_grammar(), digits_grammar()] {
        assert!(!GrammarState::for_grammar(&g).is_complete());
    }
}

/// The grammar-aware constructor differs from `initial()` only in
/// `complete`, so it is a drop-in start state for `advance_byte`.
#[test]
fn for_grammar_differs_from_initial_only_in_complete() {
    let g = ab_grammar();
    let from_grammar = GrammarState::for_grammar(&g);
    let initial = GrammarState::initial();
    assert_eq!(from_grammar.stacks, initial.stacks);
    assert_eq!(
        from_grammar.partial_token_bytes,
        initial.partial_token_bytes
    );
    assert_eq!(from_grammar.complete, initial.complete);

    let mut state = GrammarState::for_grammar(&g);
    assert_eq!(advance_byte(&mut state, &g, b'a'), StepResult::Accepted);
    assert_eq!(advance_byte(&mut state, &g, b'b'), StepResult::Accepted);
    assert!(state.is_complete());
}

/// `root ::= "a" root | ""` accepts every run of `a` bytes, each one a
/// level deeper on the stack. The byte that needs the stack past
/// `MAX_PDA_DEPTH` is reported as a limit, not as a rejection, and leaves
/// the state as it was.
#[test]
fn recursion_past_the_depth_cap_reports_the_stack_limit() {
    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root");
    b.set_alts(
        root_id,
        vec![
            vec![Symbol::Terminal(b'a'), Symbol::NonTerminal(root_id)],
            vec![],
        ],
    )
    .unwrap();
    let g = b.build();

    let mut state = GrammarState::initial();
    for i in 0..MAX_PDA_DEPTH {
        assert_eq!(
            advance_byte(&mut state, &g, b'a'),
            StepResult::Accepted,
            "byte {i} is inside the depth cap"
        );
    }
    assert!(state.is_complete());
    let before = state.clone();
    assert_eq!(
        advance_byte(&mut state, &g, b'a'),
        StepResult::StackLimitExceeded
    );
    assert_eq!(state.stacks, before.stacks);
    assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
}

/// `root ::= E{n} "a"` with `E` a rule without alternatives: one stack, one
/// structural step per `E`, no sibling and no depth. Choosing root's
/// alternative is one more step, so `MAX_EXPANSION_STEPS - 1` references
/// fit the budget and `MAX_EXPANSION_STEPS` do not.
fn empty_rule_run_grammar(references: usize) -> CompiledGrammar {
    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root");
    let empty_id = b.reserve("E");
    let mut alt = vec![Symbol::NonTerminal(empty_id); references];
    alt.push(Symbol::Terminal(b'a'));
    b.set_alts(root_id, vec![alt]).unwrap();
    b.build()
}

/// The expansion budget is charged for every structural step a stack
/// takes, not once per stack taken off the pending list: a single stack
/// walking a long run of nullable references pays for each of them.
#[test]
fn expansion_budget_is_charged_per_structural_step() {
    let within = empty_rule_run_grammar(MAX_EXPANSION_STEPS - 1);
    let mut state = GrammarState::initial();
    assert_eq!(
        advance_byte(&mut state, &within, b'a'),
        StepResult::Accepted
    );

    let over = empty_rule_run_grammar(MAX_EXPANSION_STEPS);
    let mut state = GrammarState::initial();
    assert_eq!(
        advance_byte(&mut state, &over, b'a'),
        StepResult::StackLimitExceeded
    );
}

/// Creating siblings counts against `MAX_LIVE_STACKS` as they are made: a
/// rule with far more matching alternatives than the limit fails after the
/// first `MAX_LIVE_STACKS + 1` clones instead of cloning all of them first.
#[test]
fn sibling_generation_stops_at_the_live_stack_limit() {
    let g = many_distinct_alternatives_grammar(4 * MAX_LIVE_STACKS);
    let mut stack = vec![StackFrame {
        rule_id: 0,
        alt_idx: UNCHOSEN_ALT,
        sym_pos: 0,
    }];
    let mut pending = Vec::new();
    let mut next = Vec::new();
    assert_eq!(
        open_top(&mut stack, &g, b'a', &mut pending, &mut next),
        Err(StackLimitError)
    );
    assert_eq!(
        pending.len(),
        MAX_LIVE_STACKS + 1,
        "sibling creation must stop as soon as the limit is passed"
    );
}

#[test]
fn ab_grammar_accepts_ab() {
    let g = ab_grammar();
    let mut state = GrammarState::initial();
    assert_eq!(advance_byte(&mut state, &g, b'a'), StepResult::Accepted);
    assert!(!state.is_complete()); // not done yet
    assert_eq!(advance_byte(&mut state, &g, b'b'), StepResult::Accepted);
    assert!(state.is_complete());
}

#[test]
fn ab_grammar_rejects_ba() {
    let g = ab_grammar();
    let mut state = GrammarState::initial();
    assert_eq!(advance_byte(&mut state, &g, b'b'), StepResult::Rejected);
}

#[test]
fn ab_grammar_rejects_partial_a_then_wrong() {
    let g = ab_grammar();
    let mut state = GrammarState::initial();
    advance_byte(&mut state, &g, b'a');
    assert_eq!(advance_byte(&mut state, &g, b'x'), StepResult::Rejected);
}

#[test]
fn rejected_byte_leaves_state_intact_and_resumes() {
    // A rejected byte must not corrupt the matcher: the consumed prefix
    // stays committed and the correct continuation still completes. This
    // locks the rollback-on-reject contract that `advance_byte` relies on
    // (`step_stacks` builds the next stack set separately and `advance_byte`
    // only commits it on success, so no outer snapshot is needed).
    //
    // The grammar must be *nested* so the rejecting byte is checked
    // against a stack with a child frame on it:
    //   root  ::= "a" child
    //   child ::= "bc"
    // Feeding `a`,`b` descends into `child` (frame pushed, one byte
    // consumed). The wrong byte at child's second position matches no
    // stack, and the state must keep that child frame so the correct `c`
    // can still complete. A flat grammar never exercises this and would
    // make the test vacuous.
    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root");
    let child_id = b.reserve("child");
    b.set_alts(
        child_id,
        vec![vec![Symbol::Terminal(b'b'), Symbol::Terminal(b'c')]],
    )
    .unwrap();
    b.set_alts(
        root_id,
        vec![vec![Symbol::Terminal(b'a'), Symbol::NonTerminal(child_id)]],
    )
    .unwrap();
    let g = b.build();

    let mut state = GrammarState::initial();
    assert_eq!(advance_byte(&mut state, &g, b'a'), StepResult::Accepted);
    assert_eq!(advance_byte(&mut state, &g, b'b'), StepResult::Accepted);
    assert_eq!(advance_byte(&mut state, &g, b'x'), StepResult::Rejected);
    assert_eq!(advance_byte(&mut state, &g, b'c'), StepResult::Accepted);
    assert!(state.complete);
}

#[test]
fn or_grammar_accepts_a_or_b() {
    let g = or_grammar();
    let mut s = GrammarState::initial();
    assert_eq!(advance_byte(&mut s, &g, b'a'), StepResult::Accepted);

    let mut s2 = GrammarState::initial();
    assert_eq!(advance_byte(&mut s2, &g, b'b'), StepResult::Accepted);
}

#[test]
fn or_grammar_rejects_c() {
    let g = or_grammar();
    let mut s = GrammarState::initial();
    assert_eq!(advance_byte(&mut s, &g, b'c'), StepResult::Rejected);
}

#[test]
fn dead_child_alternative_does_not_block_a_sibling() {
    let mut builder = GrammarBuilder::new();
    let root_id = builder.reserve("root");
    let dead_id = builder.reserve("dead");
    builder
        .set_alts(dead_id, vec![vec![Symbol::Terminal(b'y')]])
        .unwrap();
    builder
        .set_alts(
            root_id,
            vec![
                vec![Symbol::NonTerminal(dead_id)],
                vec![Symbol::Terminal(b'x')],
            ],
        )
        .unwrap();
    let grammar = builder.build();
    let mut state = GrammarState::initial();

    assert_eq!(
        advance_byte(&mut state, &grammar, b'x'),
        StepResult::Accepted
    );
    assert!(state.complete);
}

/// A production-valid nested grammar must reach a root sibling without
/// growing the native call stack while it exhausts uncommitted parents.
///
/// The 64 KiB native thread stack makes a recursive parent walk overflow before
/// it can accept `x`; the iterative walk completes within the same bound.
#[test]
fn deeply_nested_parent_fallback_accepts_on_bounded_stack() {
    let depth = MAX_PDA_DEPTH / 2;
    let mut rules = Vec::with_capacity(depth);
    rules.push(Rule {
        name: "root".to_string(),
        alts: vec![vec![Symbol::NonTerminal(1)], vec![Symbol::Terminal(b'x')]],
    });
    for rule_id in 1..depth - 1 {
        rules.push(Rule {
            name: String::new(),
            alts: vec![vec![Symbol::NonTerminal(rule_id + 1)]],
        });
    }
    rules.push(Rule {
        name: String::new(),
        alts: vec![vec![Symbol::Terminal(b'y')]],
    });
    let grammar = CompiledGrammar { rules };

    std::thread::Builder::new()
        .stack_size(64 * 1024)
        .spawn(move || {
            let mut state = GrammarState::initial();
            assert_eq!(
                advance_byte(&mut state, &grammar, b'x'),
                StepResult::Accepted
            );
            assert!(state.complete);
        })
        .expect("bounded-stack regression thread spawns")
        .join()
        .expect("iterative fallback must not overflow the bounded stack");
}

/// A long *acyclic* chain of distinct nullable wrapper rules has no cycle
/// for `remaining_is_nullable`'s `visited` guard to catch, so unlike the
/// cyclic case its only historical bound was call-frame depth.
/// `is_accepting` runs on `initial_grammar_state`, before any byte is
/// consumed — a single-frame state referencing the head of such a chain
/// must not overflow the native stack even though `state.stacks.len()`
/// never leaves 1 (the nullability walk descends the *static* rule graph,
/// not the PDA execution stack `MAX_PDA_DEPTH` bounds).
#[test]
fn deeply_nested_nullable_chain_accepts_on_bounded_stack() {
    let depth = MAX_PDA_DEPTH;
    let mut rules = Vec::with_capacity(depth);
    for rule_id in 0..depth - 1 {
        rules.push(Rule {
            name: String::new(),
            alts: vec![vec![Symbol::NonTerminal(rule_id + 1)]],
        });
    }
    // The chain's tail is epsilon, so the whole chain is nullable.
    rules.push(Rule {
        name: String::new(),
        alts: vec![vec![]],
    });
    let grammar = CompiledGrammar { rules };

    std::thread::Builder::new()
        .stack_size(64 * 1024)
        .spawn(move || {
            let state = initial_grammar_state(&grammar);
            assert!(state.is_complete());
        })
        .expect("bounded-stack regression thread spawns")
        .join()
        .expect("iterative nullability walk must not overflow the bounded stack");
}

fn nested_terminal_grammar(depth: usize, terminal: u8) -> CompiledGrammar {
    let mut rules = Vec::with_capacity(depth);
    for rule_id in 0..depth - 1 {
        rules.push(Rule {
            name: String::new(),
            alts: vec![vec![Symbol::NonTerminal(rule_id + 1)]],
        });
    }
    rules.push(Rule {
        name: String::new(),
        alts: vec![vec![Symbol::Terminal(terminal)]],
    });
    CompiledGrammar { rules }
}

#[test]
fn nesting_depth_limit_matches_recursive_boundary() {
    let at_limit = nested_terminal_grammar(MAX_PDA_DEPTH, b'x');
    let mut state = GrammarState::initial();
    assert_eq!(
        advance_byte(&mut state, &at_limit, b'x'),
        StepResult::Accepted
    );

    // One rule deeper needs a stack past the cap: the matcher reports its
    // limit instead of rejecting a byte the grammar accepts.
    let past_limit = nested_terminal_grammar(MAX_PDA_DEPTH + 1, b'x');
    let mut state = GrammarState::initial();
    assert_eq!(
        advance_byte(&mut state, &past_limit, b'x'),
        StepResult::StackLimitExceeded
    );
}

/// Grammar: root = "a" nonterm | "x" ; nonterm = "cd"
/// Root reserved first so it lands at index 0.
fn leading_terminal_then_nt_grammar() -> CompiledGrammar {
    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root");
    let nt_id = b.reserve("nonterm");
    b.set_alts(
        nt_id,
        vec![vec![Symbol::Terminal(b'c'), Symbol::Terminal(b'd')]],
    )
    .unwrap();
    b.set_alts(
        root_id,
        vec![
            vec![Symbol::Terminal(b'a'), Symbol::NonTerminal(nt_id)],
            vec![Symbol::Terminal(b'x')],
        ],
    )
    .unwrap();
    b.build()
}

#[test]
fn leading_terminal_then_nt_accepts_valid() {
    let g = leading_terminal_then_nt_grammar();
    // "acd" via alt-0, "x" via alt-1 must both still be accepted.
    let s0 = GrammarState::initial();
    let (r_acd, _) = simulate_token(&s0, &g, b"acd");
    assert_eq!(r_acd, SimResult::Accept);
    let s1 = GrammarState::initial();
    let (r_x, _) = simulate_token(&s1, &g, b"x");
    assert_eq!(r_x, SimResult::Accept);
}

#[test]
fn simulate_token_full_match() {
    let g = ab_grammar();
    let state = GrammarState::initial();
    let (result, _) = simulate_token(&state, &g, b"ab");
    assert_eq!(result, SimResult::Accept);
}

#[test]
fn simulate_token_reject() {
    let g = ab_grammar();
    let state = GrammarState::initial();
    let (result, _) = simulate_token(&state, &g, b"ba");
    assert_eq!(result, SimResult::Reject);
}

#[test]
fn simulate_token_partial_is_context_dependent() {
    let g = ab_grammar();
    let state = GrammarState::initial();
    // Token "ax" — first byte 'a' accepted, second 'x' rejected mid-token.
    let (result, _) = simulate_token(&state, &g, b"ax");
    assert_eq!(result, SimResult::ContextDependent);
}

#[test]
fn state_partial_bytes_recorded() {
    let g = ab_grammar();
    let mut state = GrammarState::initial();
    advance_byte(&mut state, &g, b'a');
    assert_eq!(state.partial_token_bytes, vec![b'a']);
    advance_byte(&mut state, &g, b'b');
    assert_eq!(state.partial_token_bytes, vec![b'a', b'b']);
}

#[test]
fn any_byte_matches_any_value() {
    let mut b = GrammarBuilder::new();
    b.add_rule("root", vec![vec![Symbol::AnyByte]]);
    let g = b.build();
    for byte in [b'a', b'z', b'0', b'\n', 0xffu8] {
        let mut s = GrammarState::initial();
        assert_eq!(advance_byte(&mut s, &g, byte), StepResult::Accepted);
        assert!(s.is_complete());
    }
}

#[test]
fn digits_grammar_accepts_single_digit() {
    let g = digits_grammar();
    let state = GrammarState::initial();
    let (result, _) = simulate_token(&state, &g, b"5");
    assert_eq!(result, SimResult::Accept);
}

#[test]
fn digits_grammar_accepts_multi_digit() {
    let g = digits_grammar();
    let state = GrammarState::initial();
    let (result, final_state) = simulate_token(&state, &g, b"123");
    assert_eq!(result, SimResult::Accept);
    assert!(final_state.is_complete());
}

#[test]
fn digits_grammar_rejects_letter() {
    let g = digits_grammar();
    let state = GrammarState::initial();
    let (result, _) = simulate_token(&state, &g, b"abc");
    assert_eq!(result, SimResult::Reject);
}

#[test]
fn grammar_builder_reserve_idempotent() {
    let mut builder = GrammarBuilder::new();
    let id1 = builder.reserve("foo");
    let id2 = builder.reserve("foo");
    assert_eq!(id1, id2);
}

#[test]
fn missing_root_rule_id_rejects_without_panicking() {
    let grammar = CompiledGrammar { rules: Vec::new() };
    let mut state = GrammarState::initial();
    let before = state.clone();

    assert_eq!(
        advance_byte(&mut state, &grammar, b'x'),
        StepResult::Rejected
    );
    assert_eq!(state.stacks, before.stacks);
    assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
    assert_eq!(state.complete, before.complete);
}

#[test]
fn out_of_range_state_rule_id_rejects_without_panicking() {
    let grammar = CompiledGrammar {
        rules: vec![Rule {
            name: "root".to_string(),
            alts: vec![vec![Symbol::Terminal(b'x')]],
        }],
    };
    let mut state = GrammarState {
        stacks: vec![vec![StackFrame {
            rule_id: 1,
            alt_idx: 0,
            sym_pos: 0,
        }]],
        partial_token_bytes: Vec::new(),
        complete: false,
    };
    let before = state.clone();

    assert_eq!(
        advance_byte(&mut state, &grammar, b'x'),
        StepResult::Rejected
    );
    assert_eq!(state.stacks, before.stacks);
    assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
    assert_eq!(state.complete, before.complete);
}

#[test]
fn out_of_range_non_terminal_rule_id_rejects_without_panicking() {
    let grammar = CompiledGrammar {
        rules: vec![Rule {
            name: "root".to_string(),
            alts: vec![vec![Symbol::NonTerminal(1)]],
        }],
    };
    let mut state = GrammarState::initial();
    let before = state.clone();

    assert_eq!(
        advance_byte(&mut state, &grammar, b'x'),
        StepResult::Rejected
    );
    assert_eq!(state.stacks, before.stacks);
    assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
    assert_eq!(state.complete, before.complete);
}

#[test]
fn set_alts_out_of_range_id_returns_err() {
    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root");
    assert_eq!(root_id, 0);

    // No rule was ever reserved at id 5: out of range for a 1-rule builder.
    let err = b
        .set_alts(5, vec![vec![Symbol::Terminal(b'x')]])
        .expect_err("set_alts on an unreserved id must return Err, not panic");
    assert!(err.0.contains('5'), "error should name the bad id: {err:?}");
}

#[test]
fn set_alts_in_range_id_succeeds() {
    let mut b = GrammarBuilder::new();
    let id = b.reserve("root");
    b.set_alts(id, vec![vec![Symbol::Terminal(b'x')]])
        .expect("set_alts on a freshly reserved id must succeed");
    let g = b.build();
    assert!(accepts_str(&g, b"x"));
}

#[test]
fn out_of_range_alt_idx_rejects_without_panicking() {
    // A directly-constructed StackFrame (public fields) can carry an
    // alt_idx past the rule's alternative count even though the rule id
    // itself is valid and non-empty. The main advance loop must reject
    // this the same way it already rejects an out-of-range rule id,
    // rather than indexing `rule.alts[alt_idx]` unchecked.
    let grammar = CompiledGrammar {
        rules: vec![Rule {
            name: "root".to_string(),
            alts: vec![vec![Symbol::Terminal(b'x')]], // exactly one alt
        }],
    };
    let mut state = GrammarState {
        stacks: vec![vec![StackFrame {
            rule_id: 0,
            alt_idx: 7, // no alt at index 7
            sym_pos: 0,
        }]],
        partial_token_bytes: Vec::new(),
        complete: false,
    };
    let before = state.clone();

    assert_eq!(
        advance_byte(&mut state, &grammar, b'x'),
        StepResult::Rejected
    );
    assert_eq!(state.stacks, before.stacks);
    assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
    assert_eq!(state.complete, before.complete);
}

#[test]
fn dangling_ancestor_rule_id_rejects_on_mismatch_without_panicking() {
    // A non-top (ancestor) frame can carry a rule id that no longer
    // exists in the grammar — the state a dangling NonTerminal reference
    // would leave behind. The top frame is valid but its only alt
    // mismatches the incoming byte, so the stack is dropped without ever
    // being popped into the invalid ancestor. The byte must reject, not
    // index `grammar.rules[rule_id]` unchecked.
    let grammar = CompiledGrammar {
        rules: vec![Rule {
            name: "child".to_string(),
            alts: vec![vec![Symbol::Terminal(b'z')]],
        }],
    };
    let mut state = GrammarState {
        stacks: vec![vec![
            StackFrame {
                rule_id: 999, // dangling: no such rule
                alt_idx: 0,
                sym_pos: 0,
            },
            StackFrame {
                rule_id: 0, // valid, but its only alt won't match b'x'
                alt_idx: 0,
                sym_pos: 0,
            },
        ]],
        partial_token_bytes: Vec::new(),
        complete: false,
    };
    let before = state.clone();

    assert_eq!(
        advance_byte(&mut state, &grammar, b'x'),
        StepResult::Rejected
    );
    assert_eq!(state.stacks, before.stacks);
    assert_eq!(state.partial_token_bytes, before.partial_token_bytes);
    assert_eq!(state.complete, before.complete);
}

/// `r ::= "" | "a"` (epsilon first), `root ::= r "b"`.
fn epsilon_first_alt_grammar() -> CompiledGrammar {
    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root");
    let r_id = b.reserve("r");
    b.set_alts(r_id, vec![vec![], vec![Symbol::Terminal(b'a')]])
        .unwrap();
    b.set_alts(
        root_id,
        vec![vec![Symbol::NonTerminal(r_id), Symbol::Terminal(b'b')]],
    )
    .unwrap();
    b.build()
}

/// Same shape, epsilon last: `r ::= "a" | ""`. Control proving the fix
/// does not depend on alternative order (this direction already worked).
fn epsilon_last_alt_grammar() -> CompiledGrammar {
    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root");
    let r_id = b.reserve("r");
    b.set_alts(r_id, vec![vec![Symbol::Terminal(b'a')], vec![]])
        .unwrap();
    b.set_alts(
        root_id,
        vec![vec![Symbol::NonTerminal(r_id), Symbol::Terminal(b'b')]],
    )
    .unwrap();
    b.build()
}

/// `r ::= "" | "a"`, `root ::= r "a" "c"`: accepting "aac" needs both of
/// `r`'s choices to stay open across the first byte (#322).
fn epsilon_first_multi_byte_grammar() -> CompiledGrammar {
    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root");
    let r_id = b.reserve("r");
    b.set_alts(r_id, vec![vec![], vec![Symbol::Terminal(b'a')]])
        .unwrap();
    b.set_alts(
        root_id,
        vec![vec![
            Symbol::NonTerminal(r_id),
            Symbol::Terminal(b'a'),
            Symbol::Terminal(b'c'),
        ]],
    )
    .unwrap();
    b.build()
}

/// Deeply nested `r_i ::= "" | (r_{i+1} "a") | (r_{i+1} "b")`, base case
/// `r_depth ::= "" | "a" | "b"`; each level doubles fresh re-exploration
/// of the next, so exploring every parse is ~2^depth work.
fn nested_nullable_branch_grammar(depth: usize) -> CompiledGrammar {
    let mut b = GrammarBuilder::new();
    let root_id = b.reserve("root");
    let levels: Vec<usize> = (0..=depth).map(|i| b.reserve(&format!("r{i}"))).collect();
    b.set_alts(
        levels[depth],
        vec![
            vec![],
            vec![Symbol::Terminal(b'a')],
            vec![Symbol::Terminal(b'b')],
        ],
    )
    .unwrap();
    for i in (0..depth).rev() {
        let next = levels[i + 1];
        b.set_alts(
            levels[i],
            vec![
                vec![],
                vec![Symbol::NonTerminal(next), Symbol::Terminal(b'a')],
                vec![Symbol::NonTerminal(next), Symbol::Terminal(b'b')],
            ],
        )
        .unwrap();
    }
    b.set_alts(
        root_id,
        vec![vec![Symbol::NonTerminal(levels[0]), Symbol::Terminal(b'z')]],
    )
    .unwrap();
    b.build()
}

/// #322: an epsilon-first alternative must not permanently commit. `"ab"`
/// falls back to `r`'s `"a"` once `root`'s `"b"` mismatches; `"b"` keeps
/// working via the epsilon choice; `"c"` / `"aab"` still reject.
#[test]
fn epsilon_first_alternative_is_retried_on_later_mismatch() {
    let g = epsilon_first_alt_grammar();
    assert!(accepts_str(&g, b"ab"), "\"ab\" must accept via r ::= \"a\"");
    assert!(accepts_str(&g, b"b"), "\"b\" must accept via r ::= \"\"");
    assert!(!accepts_str(&g, b"c"), "\"c\" matches neither alternative");
    assert!(
        !accepts_str(&g, b"aab"),
        "\"aab\" has one 'a' too many for r ::= \"\" | \"a\""
    );
}

/// Epsilon *last* control: must keep accepting both forms, unaffected by
/// alternative order (this direction predates #322 and must not regress).
#[test]
fn epsilon_last_alternative_still_accepts_both_forms() {
    let g = epsilon_last_alt_grammar();
    assert!(accepts_str(&g, b"ab"), "\"ab\" must accept via r ::= \"a\"");
    assert!(accepts_str(&g, b"b"), "\"b\" must accept via r ::= \"\"");
    assert!(!accepts_str(&g, b"c"));
    assert!(!accepts_str(&g, b"aab"));
}

/// `r`'s epsilon choice for "aac" is still open after byte 1: the stack
/// that took `r ::= "a"` and the stack that took `r ::= ""` both stay
/// alive, and byte 2 settles which one the input follows.
#[test]
fn epsilon_first_choice_stays_open_across_a_later_byte() {
    let g = epsilon_first_multi_byte_grammar();
    assert!(
        accepts_str(&g, b"aac"),
        "\"aac\" must accept via r ::= \"a\""
    );
}

/// Exploring every parse of `depth`-nested branches is ~2^depth work,
/// intractable if unbounded; completing at all proves the expansion budget
/// capped it. The step reports the limit rather than a rejection, because
/// the matcher gave up before it could tell whether "x" fits.
#[test]
fn nested_nullable_alternatives_report_the_expansion_limit() {
    let g = nested_nullable_branch_grammar(40);
    let mut state = GrammarState::initial();
    assert_eq!(
        advance_byte(&mut state, &g, b'x'),
        StepResult::StackLimitExceeded
    );
    let (result, _) = simulate_token(&GrammarState::initial(), &g, b"x");
    assert_eq!(result, SimResult::StackLimitExceeded);
}

#[test]
fn dangling_ancestor_rule_id_stops_collapse_without_panicking() {
    // Same dangling-ancestor shape as above, but reached via the
    // post-match `collapse_exhausted` walk instead of a mismatch:
    // the top frame's byte DOES match and exhausts its only alt, so
    // popping it advances into the invalid ancestor while collapsing.
    let grammar = CompiledGrammar {
        rules: vec![Rule {
            name: "child".to_string(),
            alts: vec![vec![Symbol::Terminal(b'x')]],
        }],
    };
    let mut state = GrammarState {
        stacks: vec![vec![
            StackFrame {
                rule_id: 999, // dangling: no such rule
                alt_idx: 0,
                sym_pos: 0,
            },
            StackFrame {
                rule_id: 0, // valid; matches b'x' and then exhausts
                alt_idx: 0,
                sym_pos: 0,
            },
        ]],
        partial_token_bytes: Vec::new(),
        complete: false,
    };

    // Must not panic. The byte match itself still succeeds (the top
    // frame's own terminal matched); what must not panic is the
    // subsequent collapse into the dangling ancestor.
    let result = advance_byte(&mut state, &grammar, b'x');
    assert_eq!(result, StepResult::Accepted);
}

#[test]
fn root_is_rule_zero_when_present() {
    let grammar = ab_grammar();

    let root = grammar.root().expect("a grammar with rules has a root");

    assert_eq!(root.name, "root");
}

#[test]
fn root_of_a_grammar_without_rules_is_none() {
    let grammar = GrammarBuilder::new().build();

    assert!(grammar.root().is_none());
}
