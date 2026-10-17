use super::*;

/// Build a tiny vocab: token 0 = b"t", token 1 = b"f", token 2 = b"x".
fn tiny_vocab() -> Vec<Vec<u8>> {
    vec![b"t".to_vec(), b"f".to_vec(), b"x".to_vec()]
}

/// Grammar spec: boolean (true | false)
fn bool_spec() -> GrammarSpec {
    GrammarSpec::JsonSchema(serde_json::json!({"type": "boolean"}))
}

#[test]
fn engine_new_from_json_schema() {
    let vocab = tiny_vocab();
    let spec = bool_spec();
    let result = GrammarEngine::new(&spec, vocab);
    assert!(result.is_ok(), "engine construction should succeed");
}

#[test]
fn engine_new_from_gbnf() {
    let vocab = tiny_vocab();
    let spec = GrammarSpec::Gbnf("root ::= \"t\" | \"f\"\n".to_string());
    let result = GrammarEngine::new(&spec, vocab);
    assert!(result.is_ok());
}

/// Regression (issue #343): a left-recursive GBNF grammar (`root ::= root`)
/// must not hang. Before the PDA depth cap, `enumerate_grammar_states` grew
/// the PDA stack without bound inside a single `simulate_token` call. The
/// depth cap stops that growth and construction reports the stack limit
/// instead of returning.
#[test]
fn cyclic_gbnf_does_not_hang() {
    let vocab = tiny_vocab();
    let spec = GrammarSpec::Gbnf("root ::= root\n".to_string());
    let err = match GrammarEngine::new(&spec, vocab) {
        Ok(_) => panic!("a cyclic grammar cannot be matched and must not construct"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("too deeply nested"), "{err}");
}

/// Regression (issue #343): a JSON-Schema `$ref` cycle compiles to a cyclic
/// grammar and previously hung at construction via the same PDA mechanism.
#[test]
fn cyclic_ref_schema_does_not_hang() {
    let vocab = tiny_vocab();
    let spec = GrammarSpec::JsonSchema(serde_json::json!({
        "$ref": "#/$defs/Node",
        "$defs": { "Node": { "$ref": "#/$defs/Node" } }
    }));
    let err = match GrammarEngine::new(&spec, vocab) {
        Ok(_) => panic!("a cyclic $ref schema cannot be matched and must not construct"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("too deeply nested"), "{err}");
}

/// Regression (issue #343): an array schema with an absurd `maxItems`
/// (near `u64::MAX`) previously overflowed the stack in `build_bounded_tail`.
/// It must now be rejected with a typed error at the parse boundary.
#[test]
fn array_maxitems_overflow_rejected() {
    let vocab = tiny_vocab();
    let spec = GrammarSpec::JsonSchema(serde_json::json!({
        "type": "array",
        "items": { "type": "boolean" },
        "maxItems": 18446744073709551615u64
    }));
    let err = match GrammarEngine::new(&spec, vocab) {
        Ok(_) => panic!("absurd maxItems must be rejected, not overflow the stack"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("maxItems"),
        "error should name the offending field: {err}"
    );
}

/// Regression (issue #343): the `minItems` sibling — a near-`u64::MAX`
/// `minItems` drives the required-item loop ~1.8e19 times. Reject it too.
#[test]
fn array_minitems_overflow_rejected() {
    let vocab = tiny_vocab();
    let spec = GrammarSpec::JsonSchema(serde_json::json!({
        "type": "array",
        "items": { "type": "boolean" },
        "minItems": 18446744073709551615u64
    }));
    let err = match GrammarEngine::new(&spec, vocab) {
        Ok(_) => panic!("absurd minItems must be rejected, not hang"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("minItems"),
        "error should name the offending field: {err}"
    );
}

/// Regression (issue #343, finding B): a long *acyclic* `$defs`
/// reference chain (`N0→N1→…→Nk`, each a distinct unseen ref) recurses
/// `compile_schema` once per link. Without the depth cap this overflowed the
/// stack at compile time, independent of the PDA cycle fix. A 2000-link
/// chain (well past MAX_SCHEMA_DEPTH=512) must reject cleanly.
#[test]
fn deep_ref_chain_rejected() {
    let n = 2000usize;
    let mut defs = serde_json::Map::new();
    for i in 0..n {
        let target = if i + 1 < n {
            serde_json::json!({ "$ref": format!("#/$defs/N{}", i + 1) })
        } else {
            serde_json::json!({ "type": "boolean" })
        };
        defs.insert(format!("N{i}"), target);
    }
    let schema = serde_json::json!({ "$ref": "#/$defs/N0", "$defs": defs });
    let spec = GrammarSpec::JsonSchema(schema);
    let err = match GrammarEngine::new(&spec, tiny_vocab()) {
        Ok(_) => panic!("deep $ref chain must be rejected, not overflow the stack"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("depth"), "error should mention depth: {err}");
}

/// A short acyclic `$ref` chain (well under the cap) must still compile.
#[test]
fn shallow_ref_chain_accepted() {
    let n = 64usize;
    let mut defs = serde_json::Map::new();
    for i in 0..n {
        let target = if i + 1 < n {
            serde_json::json!({ "$ref": format!("#/$defs/N{}", i + 1) })
        } else {
            serde_json::json!({ "type": "boolean" })
        };
        defs.insert(format!("N{i}"), target);
    }
    let schema = serde_json::json!({ "$ref": "#/$defs/N0", "$defs": defs });
    let result = GrammarEngine::new(&GrammarSpec::JsonSchema(schema), tiny_vocab());
    assert!(result.is_ok(), "a 64-link $ref chain should compile");
}

/// Boundary (issue #343, finding D): `maxItems` one past the cap is
/// rejected; a small in-range `maxItems` still compiles.
#[test]
fn array_maxitems_boundary() {
    let over = GrammarSpec::JsonSchema(serde_json::json!({
        "type": "array", "items": { "type": "boolean" }, "maxItems": 4097
    }));
    assert!(
        GrammarEngine::new(&over, tiny_vocab()).is_err(),
        "maxItems just past the cap must be rejected"
    );
    let ok = GrammarSpec::JsonSchema(serde_json::json!({
        "type": "array", "items": { "type": "boolean" }, "maxItems": 8
    }));
    assert!(
        GrammarEngine::new(&ok, tiny_vocab()).is_ok(),
        "a small in-range maxItems must still compile"
    );
}

#[test]
fn mask_logits_blocks_disallowed() {
    // Boolean grammar: only "true" or "false" tokens allowed.
    // vocab: token 0 = b"true", token 1 = b"false", token 2 = b"other"
    let vocab = vec![b"true".to_vec(), b"false".to_vec(), b"other".to_vec()];
    let spec = GrammarSpec::JsonSchema(serde_json::json!({"type": "boolean"}));
    let engine = GrammarEngine::new(&spec, vocab).unwrap();

    let mut state = engine.initial_state();
    let mut logits = vec![1.0f32, 2.0f32, 3.0f32];
    engine
        .mask_logits(&mut state, &mut logits)
        .expect("matching vocab length");

    // "other" should be blocked.
    assert_eq!(
        logits[2],
        f32::NEG_INFINITY,
        "token 'other' should be blocked"
    );
    // "true" or "false" tokens should be allowed (at least one).
    assert!(
        logits[0] > f32::NEG_INFINITY || logits[1] > f32::NEG_INFINITY,
        "at least one of 'true'/'false' must be allowed"
    );
}

#[test]
fn advance_updates_state() {
    // null grammar: "null"
    let vocab = vec![b"n".to_vec(), b"u".to_vec(), b"l".to_vec(), b"l".to_vec()];
    let spec = GrammarSpec::JsonSchema(serde_json::json!({"type": "null"}));
    let engine = GrammarEngine::new(&spec, vocab).unwrap();

    let mut state = engine.initial_state();
    // Advance by token 0 ('n'): should be accepted.
    assert!(engine.advance(&mut state, 0).unwrap());
}

#[test]
fn advance_rejects_wrong_token() {
    // "null" grammar — token 0 = b"x" should be rejected.
    let vocab = vec![b"x".to_vec(), b"n".to_vec()];
    let spec = GrammarSpec::JsonSchema(serde_json::json!({"type": "null"}));
    let engine = GrammarEngine::new(&spec, vocab).unwrap();

    let mut state = engine.initial_state();
    // Token 0 = 'x' should be rejected in the "null" grammar.
    let result = engine.advance(&mut state, 0).unwrap();
    assert!(!result, "'x' should be rejected for null grammar");
}

#[test]
fn initial_state_not_complete() {
    let vocab = vec![b"t".to_vec()];
    let spec = GrammarSpec::Gbnf("root ::= \"test\"\n".to_string());
    let engine = GrammarEngine::new(&spec, vocab).unwrap();
    let state = engine.initial_state();
    assert!(!state.is_complete(), "initial state should not be complete");
}

#[test]
fn nullable_initial_state_is_complete_without_continuation() {
    let vocab = vec![b"a".to_vec()];
    let spec = GrammarSpec::Gbnf("root ::= \"\"\n".to_string());
    let engine = GrammarEngine::new(&spec, vocab).unwrap();
    let mut state = engine.initial_state();

    assert!(state.is_complete());
    assert!(engine.is_complete_without_continuation(&state));

    let mut logits = vec![0.0];
    enable_mask_profiling();
    engine
        .mask_logits(&mut state, &mut logits)
        .expect("locally constructed engine and matching logits length must mask");
    let profile = take_mask_profile();
    assert_eq!(logits, vec![f32::NEG_INFINITY]);
    assert_eq!(profile.precomputed_calls, 1);
    assert_eq!(profile.fallback_calls, 0);
}

#[test]
fn nullable_initial_state_with_continuation_is_not_terminal() {
    let vocab = vec![b"a".to_vec()];
    let spec = GrammarSpec::Gbnf("root ::= \"a\"?\n".to_string());
    let engine = GrammarEngine::new(&spec, vocab).unwrap();
    let mut state = engine.initial_state();

    assert!(state.is_complete());
    assert!(!engine.is_complete_without_continuation(&state));

    let mut logits = vec![0.0];
    enable_mask_profiling();
    engine
        .mask_logits(&mut state, &mut logits)
        .expect("locally constructed engine and matching logits length must mask");
    let profile = take_mask_profile();
    assert_eq!(logits, vec![0.0]);
    assert_eq!(profile.precomputed_calls, 1);
    assert_eq!(profile.fallback_calls, 0);
}

#[test]
fn complete_state_without_continuation_is_terminal() {
    let vocab = vec![b"a".to_vec()];
    let spec = GrammarSpec::Gbnf("root ::= \"a\"\n".to_string());
    let engine = GrammarEngine::new(&spec, vocab).unwrap();
    let mut state = engine.initial_state();

    assert!(engine.advance(&mut state, 0).unwrap());
    assert!(engine.is_complete_without_continuation(&state));
}

#[test]
fn complete_state_with_continuation_is_not_terminal() {
    let vocab = vec![b"a".to_vec()];
    let spec = GrammarSpec::Gbnf("root ::= \"a\"+\n".to_string());
    let engine = GrammarEngine::new(&spec, vocab).unwrap();
    let mut state = engine.initial_state();

    assert!(engine.advance(&mut state, 0).unwrap());
    assert!(state.is_complete());
    assert!(!engine.is_complete_without_continuation(&state));
    assert!(engine.advance(&mut state, 0).unwrap());
}

#[test]
fn engine_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GrammarEngine>();
}

#[test]
fn mask_logits_logits_shorter_returns_error() {
    let vocab = vec![b"a".to_vec(), b"b".to_vec()];
    let spec = GrammarSpec::JsonSchema(serde_json::json!({"type": "null"}));
    let engine = GrammarEngine::new(&spec, vocab).unwrap();
    let mut state = engine.initial_state();
    let mut logits = vec![0.0f32];
    let error = engine
        .mask_logits(&mut state, &mut logits)
        .expect_err("logits shorter than vocab_size must return an error");
    assert!(error.0.contains("logits length 1"));
    assert!(error.0.contains("vocabulary size 2"));

    let simulation_error = engine
        .mask_by_simulation(&state, &mut logits)
        .expect_err("the public simulation path must reject the same mismatch");
    assert!(simulation_error.0.contains("logits length 1"));
    assert!(simulation_error.0.contains("vocabulary size 2"));
}

#[test]
fn mask_logits_rechecks_only_the_current_states_candidates() {
    let spec = GrammarSpec::Gbnf("root ::= \"abcd\"\n".to_string());
    let vocab = vec![
        b"a".to_vec(),
        b"b".to_vec(),
        b"c".to_vec(),
        b"d".to_vec(),
        b"ax".to_vec(),
        b"bx".to_vec(),
        b"cx".to_vec(),
        b"dx".to_vec(),
    ];
    let engine = GrammarEngine::new(&spec, vocab).expect("fixture grammar must compile");
    let mut state = engine.initial_state();
    assert!(
        engine.advance(&mut state, 0).unwrap(),
        "token 'a' must advance"
    );

    reset_context_recheck_candidates_for_test();
    let mut actual = vec![0.0; 8];
    engine
        .mask_logits(&mut state, &mut actual)
        .expect("fixture logits match the vocabulary");
    assert_eq!(
        take_context_recheck_candidates_for_test(),
        1,
        "state after 'a' must inspect only the 'bx' partial token"
    );

    let mut oracle = vec![0.0; 8];
    engine
        .mask_by_simulation(&state, &mut oracle)
        .expect("fixture logits match the vocabulary");
    assert_eq!(actual, oracle);
}

/// Two alternatives that share their first byte stay reachable through the
/// engine: after "a" both continuations are allowed, and the state after
/// "a" is one of the precomputed partition states.
#[test]
fn shared_first_byte_alternatives_stay_reachable_through_the_engine() {
    let spec = GrammarSpec::Gbnf("root ::= \"ab\" | \"ac\"\n".to_string());
    let vocab = vec![
        b"a".to_vec(),
        b"b".to_vec(),
        b"c".to_vec(),
        b"d".to_vec(),
        b"ab".to_vec(),
        b"ac".to_vec(),
        b"ad".to_vec(),
    ];
    let engine = GrammarEngine::new(&spec, vocab).expect("fixture grammar must compile");
    let mut state = engine.initial_state();
    assert!(
        engine.advance(&mut state, 0).unwrap(),
        "token 'a' must advance"
    );
    assert!(
        engine.find_state_id(&state).is_some(),
        "the state after 'a' must be a precomputed partition state"
    );

    let mut logits = vec![0.0f32; 7];
    engine
        .mask_logits(&mut state, &mut logits)
        .expect("fixture logits match the vocabulary");
    let allowed: Vec<bool> = logits.iter().map(|l| *l != f32::NEG_INFINITY).collect();
    assert_eq!(
        allowed,
        vec![false, true, true, false, false, false, false],
        "after 'a' only 'b' and 'c' continue a parse"
    );

    let mut via_b = state.clone();
    assert!(engine.advance(&mut via_b, 1).unwrap(), "'ab' must complete");
    assert!(via_b.is_complete());
    let mut via_c = state.clone();
    assert!(engine.advance(&mut via_c, 2).unwrap(), "'ac' must complete");
    assert!(via_c.is_complete());
    let mut via_d = state;
    assert!(
        !engine.advance(&mut via_d, 3).unwrap(),
        "'ad' must be rejected"
    );
}

/// Byte-at-a-time walk through a schema whose optional keys share the
/// prefix `"al`, including the member that skips the first optional key.
/// At every step the mask must leave the next byte of the valid input
/// allowed, and the finished input must be complete.
#[test]
fn optional_keys_sharing_a_prefix_are_reachable_through_mask_logits() {
    let spec = GrammarSpec::JsonSchema(serde_json::json!({
        "type": "object",
        "properties": {
            "alpha": {"type": "integer"},
            "alps": {"type": "integer"}
        },
        "additionalProperties": false
    }));
    let vocab: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
    let engine = GrammarEngine::new(&spec, vocab).expect("fixture schema must compile");

    for input in [
        &br#"{"alpha":1}"#[..],
        &br#"{"alps":2}"#[..],
        &br#"{"alpha":1,"alps":2}"#[..],
        &b"{}"[..],
    ] {
        let mut state = engine.initial_state();
        for (at, &byte) in input.iter().enumerate() {
            let mut logits = vec![0.0f32; 256];
            engine
                .mask_logits(&mut state, &mut logits)
                .expect("fixture logits match the vocabulary");
            assert!(
                logits[byte as usize] != f32::NEG_INFINITY,
                "{}: byte {at} ({:?}) was masked although it continues a valid object",
                String::from_utf8_lossy(input),
                byte as char
            );
            assert!(
                engine.advance(&mut state, u32::from(byte)).unwrap(),
                "{}: byte {at} must advance",
                String::from_utf8_lossy(input)
            );
        }
        assert!(
            state.is_complete(),
            "{} must end in a complete state",
            String::from_utf8_lossy(input)
        );
    }
}

/// A grammar that needs more live stacks than the matcher allows is
/// refused at construction with an error that names the limit, instead of
/// being built over a matcher that silently drops stacks.
#[test]
fn engine_construction_reports_the_live_stack_limit() {
    let alternatives: Vec<String> = (0..=crate::grammar::pda::MAX_LIVE_STACKS)
        .map(|i| format!("\"a\" \"{i}\""))
        .collect();
    let spec = GrammarSpec::Gbnf(format!("root ::= {}\n", alternatives.join(" | ")));
    let vocab = vec![b"a".to_vec(), b"0".to_vec()];
    let err = match GrammarEngine::new(&spec, vocab) {
        Ok(_) => panic!("a grammar past the live-stack limit must not construct"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("live stacks"),
        "the error must name the stack limit: {err}"
    );
}

/// The runtime paths report the live-stack limit instead of masking the
/// affected tokens as if the grammar had rejected them: the exact
/// simulation and trie masks return an error, `advance` fails, and a state
/// whose continuations cannot be classified is not called terminal.
#[test]
fn runtime_paths_report_the_live_stack_limit() {
    use crate::grammar::pda::{CompiledGrammar, MAX_LIVE_STACKS, Rule, Symbol};

    // The root is nullable (its first alternative is empty), so the initial
    // state is accepting; every other alternative starts with "a".
    let mut alts = vec![Vec::new()];
    alts.extend(vec![
        vec![Symbol::Terminal(b'a'), Symbol::Terminal(b'b')];
        MAX_LIVE_STACKS + 1
    ]);
    let over_limit = CompiledGrammar {
        rules: vec![Rule {
            name: "root".to_string(),
            alts,
        }],
    };
    let vocab = vec![b"a".to_vec(), b"z".to_vec()];
    let spec = GrammarSpec::Gbnf("root ::= \"a\"\n".to_string());
    let mut engine = GrammarEngine::new(&spec, vocab).expect("fixture grammar must compile");
    engine.grammar = over_limit;

    let mut state = initial_grammar_state(&engine.grammar);
    assert!(
        state.is_complete(),
        "the nullable root accepts the empty input"
    );
    let mut logits = vec![0.0f32; 2];
    let err = engine
        .mask_by_simulation(&state, &mut logits)
        .expect_err("simulation must report the stack limit");
    assert!(err.0.contains("live stacks"), "{}", err.0);
    assert_eq!(logits, vec![0.0, 0.0], "no token may be masked on an error");

    let mut logits = vec![0.0f32; 2];
    assert!(
        engine.mask_by_trie(&state, &mut logits).is_err(),
        "the trie mask must report the stack limit"
    );

    let err = engine
        .advance(&mut state, 0)
        .expect_err("advance must report the stack limit as an error");
    assert!(err.0.contains("live stacks"), "{}", err.0);
    assert!(
        !engine.is_complete_without_continuation(&state),
        "an unclassifiable continuation must not read as a terminal state"
    );
}

#[test]
fn unknown_state_fallback_is_sound() {
    // Soundness regression: when a grammar exceeds MAX_GRAMMAR_STATES the
    // BFS enumeration is truncated, so a deep runtime state is not in the
    // partition. `find_state_id` must NOT silently fall back to the
    // initial-state mask (which over-allows tokens that are valid only at
    // position 0). It must compute the exact mask by simulation.
    //
    // Grammar: a literal chain `"` + 300×`a` + `"` (>256 PDA states).
    // Vocab: token 0 = b"\"" (quote), token 1 = b"a".
    // At a deep mid-chain state the only valid next token is `a`; the
    // closing quote `"` is invalid there but IS valid at state 0.
    use crate::grammar::pda::{CompiledGrammar, Rule, Symbol};

    let mut chain = vec![Symbol::Terminal(b'"')];
    chain.extend(std::iter::repeat_n(Symbol::Terminal(b'a'), 300));
    chain.push(Symbol::Terminal(b'"'));
    let grammar = CompiledGrammar {
        rules: vec![Rule {
            name: "root".to_string(),
            alts: vec![chain],
        }],
    };
    let vocab = vec![b"\"".to_vec(), b"a".to_vec()];

    let spec = GrammarSpec::Gbnf("root ::= \"placeholder\"\n".to_string());
    // Build engine then swap in the hand-built grammar + partition so the
    // test does not depend on json_schema const-compilation internals.
    let mut engine = GrammarEngine::new(&spec, vocab.clone()).unwrap();
    let states = enumerate_grammar_states(
        &grammar,
        &vocab,
        crate::grammar::vocab_partition::MAX_GRAMMAR_STATES,
    )
    .expect("fixture stays inside the stack limits");
    // Enumeration must have hit the cap, otherwise the test is vacuous.
    assert_eq!(
        states.len(),
        crate::grammar::vocab_partition::MAX_GRAMMAR_STATES,
        "grammar must exceed the state cap for this regression to bite"
    );
    engine.partition = VocabPartition::build(&grammar, states, &vocab);
    engine.grammar = grammar;

    // Drive into a deep state beyond the enumerated cap: quote + 270 a's.
    let mut state = engine.initial_state();
    assert!(
        engine.advance(&mut state, 0).unwrap(),
        "opening quote accepted"
    );
    for _ in 0..270 {
        assert!(
            engine.advance(&mut state, 1).unwrap(),
            "mid-chain 'a' accepted"
        );
    }

    // The deep state must NOT be in the enumerated partition — otherwise
    // this test exercises the precomputed fast path, not the simulation
    // fallback it is meant to cover.
    assert!(
        engine.find_state_id(&state).is_none(),
        "deep state must be unknown to the capped partition (else the \
             simulation fallback is not the path under test)"
    );

    let mut logits = vec![1.0f32, 1.0f32];
    engine
        .mask_logits(&mut state, &mut logits)
        .expect("matching vocab length");

    assert!(
        logits[1] > f32::NEG_INFINITY,
        "valid mid-chain token 'a' must remain allowed at the deep state"
    );
    assert_eq!(
        logits[0],
        f32::NEG_INFINITY,
        "invalid closing-quote token must be blocked at the deep state \
             (state-0 fallback would wrongly allow it)"
    );
}

#[test]
fn bitmask_and_correctness_large_vocab() {
    // Build a vocab of 130 tokens: tokens 0..4 = b"true" chars,
    // rest = b"x".  Only token 0 starts "true"/"false" in a boolean grammar.
    let mut vocab: Vec<Vec<u8>> = vec![b"true".to_vec(), b"false".to_vec()];
    for i in 2..130 {
        vocab.push(format!("tok{i}").into_bytes());
    }
    let spec = GrammarSpec::JsonSchema(serde_json::json!({"type": "boolean"}));
    let engine = GrammarEngine::new(&spec, vocab).unwrap();

    let mut state = engine.initial_state();
    let mut logits = vec![1.0f32; 130];
    engine
        .mask_logits(&mut state, &mut logits)
        .expect("matching vocab length");

    // At least tokens 0 and 1 should be allowed.
    assert!(
        logits[0] > f32::NEG_INFINITY,
        "token 'true' should be allowed"
    );
    assert!(
        logits[1] > f32::NEG_INFINITY,
        "token 'false' should be allowed"
    );
    // All other tokens should be blocked.
    for i in 2..130 {
        assert_eq!(logits[i], f32::NEG_INFINITY, "token {i} should be blocked");
    }
}

// -----------------------------------------------------------------
// Trie-mask differential tests (issue #734 fix)
//
// `mask_by_trie` must reproduce `mask_by_simulation`'s mask bit for
// bit. These tests harvest a corpus of real grammar states from a
// generation trajectory against the exact schema shape
// `gramperf_profile` profiles (crates/inference/src/bin/
// gramperf_profile.rs), then compare the two algorithms directly.
//
// They are equivalence tests: both algorithms run the same
// `advance_byte` matcher, so they pin the trie walk to the per-token
// simulation, not the matcher's language. The language is pinned by the
// matcher tests in `pda.rs` and
// `tests/grammar_shared_first_byte_alternatives.rs`.
// -----------------------------------------------------------------

/// The #734-shape schema from `gramperf_profile`: 4 nested object
/// levels, 3 array fields, 6 six-member string enums. Reused verbatim
/// (not approximated) so this corpus exercises the same structural
/// over-cap behavior the profiling report measured.
fn trie_diff_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "object",
                        "properties": {
                            "level3": {
                                "type": "object",
                                "properties": {
                                    "level4": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string", "enum": ["active", "inactive", "pending", "archived", "deleted", "draft"]},
                                            "value": {"type": "integer"}
                                        },
                                        "required": ["status", "value"]
                                    }
                                },
                                "required": ["level4"]
                            },
                            "category": {"type": "string", "enum": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]}
                        },
                        "required": ["level3", "category"]
                    },
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["level2", "tags"]
            },
            "items": {"type": "array", "items": {"type": "integer"}},
            "flags": {"type": "array", "items": {"type": "boolean"}},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent", "critical", "none"]},
            "region": {"type": "string", "enum": ["us", "eu", "apac", "latam", "mea", "other"]},
            "mode": {"type": "string", "enum": ["sync", "async", "batch", "stream", "manual", "auto"]},
            "role": {"type": "string", "enum": ["admin", "user", "guest", "owner", "viewer", "editor"]}
        },
        "required": ["level1", "items", "flags", "priority", "region", "mode", "role"]
    })
}

/// Synthetic vocabulary sized for fast unit tests (not the real 248K
/// tokenizer vocab): every single byte (so the trie branches fully at
/// every depth, matching the real vocab's near-total root fan-out)
/// plus multi-byte literal fragments drawn from the schema's property
/// names, enum members, and JSON keywords, so the trie also exercises
/// prefix-shared, straight-line multi-byte runs.
fn trie_diff_vocab() -> Vec<Vec<u8>> {
    let mut vocab: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
    let fragments: &[&str] = &[
        "\"level1\"",
        "\"level2\"",
        "\"level3\"",
        "\"level4\"",
        "\"status\"",
        "\"value\"",
        "\"category\"",
        "\"tags\"",
        "\"items\"",
        "\"flags\"",
        "\"priority\"",
        "\"region\"",
        "\"mode\"",
        "\"role\"",
        "\"active\"",
        "\"inactive\"",
        "\"pending\"",
        "\"archived\"",
        "\"deleted\"",
        "\"draft\"",
        "\"alpha\"",
        "\"beta\"",
        "\"gamma\"",
        "\"delta\"",
        "\"epsilon\"",
        "\"zeta\"",
        "\"low\"",
        "\"medium\"",
        "\"high\"",
        "\"urgent\"",
        "\"critical\"",
        "\"none\"",
        "\"us\"",
        "\"eu\"",
        "\"apac\"",
        "\"latam\"",
        "\"mea\"",
        "\"other\"",
        "\"sync\"",
        "\"async\"",
        "\"batch\"",
        "\"stream\"",
        "\"manual\"",
        "\"auto\"",
        "\"admin\"",
        "\"user\"",
        "\"guest\"",
        "\"owner\"",
        "\"viewer\"",
        "\"editor\"",
        "true",
        "false",
        "null",
    ];
    for f in fragments {
        vocab.push(f.as_bytes().to_vec());
    }
    vocab
}

/// A minified, schema-valid JSON instance. `serde_json::Value::Object`
/// is a `BTreeMap` in this workspace (no `preserve_order` feature), so
/// `compile_object`'s property order — and therefore the required-key
/// order the PDA enforces — is alphabetical-by-key, not `properties`
/// declaration order or `required`-array order; this instance's key
/// order at every nesting level is alphabetical to match (top:
/// flags, items, level1, mode, priority, region, role; level2:
/// category, level3 — note `category` sorts before `level3`). Walking
/// this byte-by-byte through the PDA is the "generation trajectory"
/// the corpus is harvested from: its byte positions span the initial
/// state (byte 0), deep nesting (into level1/level2/level3/level4),
/// inside-string (mid `"active`), inside-enum-literal (mid `"alpha`,
/// `"low`, `"sync`, ...), post-number (right after `42`, `1`, `2`,
/// `3`), and near-complete (the last few closing-brace bytes) — the
/// edge-state categories the correctness bar requires, without needing
/// a separate hand-built state per category.
const TRIE_DIFF_INSTANCE: &str = r#"{"flags":[true,false],"items":[1,2,3],"level1":{"level2":{"category":"alpha","level3":{"level4":{"status":"active","value":42}}},"tags":["x","y"]},"mode":"sync","priority":"low","region":"us","role":"admin"}"#;

/// Harvest a corpus of distinct grammar states by advancing `engine`
/// byte-by-byte through [`TRIE_DIFF_INSTANCE`], deduping by PDA
/// identity (`states_equal`, same identity `find_state_id` uses).
/// Panics with the failing byte position if the instance is rejected —
/// a silent partial trajectory would silently shrink the corpus
/// instead of failing the test that depends on its size.
fn harvest_trajectory_states(engine: &GrammarEngine) -> Vec<GrammarState> {
    let mut state = engine.initial_state();
    let mut corpus = vec![state.clone()];
    for (i, &b) in TRIE_DIFF_INSTANCE.as_bytes().iter().enumerate() {
        let step = advance_byte(&mut state, &engine.grammar, b);
        assert_eq!(
            step,
            StepResult::Accepted,
            "trajectory instance rejected at byte {i} ({:?}); fixture is out of \
                 sync with the schema",
            b as char
        );
        if !corpus.iter().any(|s| states_equal(s, &state)) {
            corpus.push(state.clone());
        }
    }
    assert!(
        state.is_complete(),
        "trajectory instance must fully complete the grammar"
    );
    corpus
}

#[test]
fn trie_mask_byte_identical_to_oracle_over_corpus() {
    let vocab = trie_diff_vocab();
    let spec = GrammarSpec::JsonSchema(trie_diff_schema());
    let engine = GrammarEngine::new(&spec, vocab.clone()).unwrap();
    assert!(
        engine.exceeds_state_budget(),
        "fixture must exceed the state cap for this differential to exercise \
             the fallback path (the point of this test)"
    );

    let corpus = harvest_trajectory_states(&engine);
    assert!(
        corpus.len() >= 20,
        "need at least 20 distinct harvested states, got {}",
        corpus.len()
    );

    let mut over_cap_states = 0usize;
    for (i, state) in corpus.iter().enumerate() {
        let mut oracle_logits = vec![0.0f32; vocab.len()];
        let mut trie_logits = vec![0.0f32; vocab.len()];
        engine
            .mask_by_simulation(state, &mut oracle_logits)
            .expect("matching vocab length");
        engine
            .mask_by_trie(state, &mut trie_logits)
            .expect("fixture stays inside the stack limits");
        if engine.find_state_id(state).is_none() {
            over_cap_states += 1;
        }
        for tok in 0..vocab.len() {
            assert_eq!(
                trie_logits[tok],
                oracle_logits[tok],
                "state #{i}: token {tok} ({:?}) mismatch — trie={} oracle={}",
                String::from_utf8_lossy(&vocab[tok]),
                trie_logits[tok],
                oracle_logits[tok]
            );
        }
    }
    assert!(
        over_cap_states >= 15,
        "corpus should mostly cover the over-cap fallback path (the profiling \
             report measured a 98% fallback rate on this schema shape); got \
             {over_cap_states}/{}",
        corpus.len()
    );
}

#[test]
fn trie_mask_never_over_accepts_vs_oracle() {
    // P0, reported separately from the byte-identical test above: this
    // isolates the over-accept direction (trie allows a token the
    // oracle rejects) so a prune-logic regression that only
    // over-accepts is caught and reported distinctly from an
    // under-accept regression.
    let vocab = trie_diff_vocab();
    let spec = GrammarSpec::JsonSchema(trie_diff_schema());
    let engine = GrammarEngine::new(&spec, vocab.clone()).unwrap();
    let corpus = harvest_trajectory_states(&engine);

    let mut over_accepts: Vec<(usize, Vec<u8>)> = Vec::new();
    for state in &corpus {
        let mut oracle_logits = vec![0.0f32; vocab.len()];
        let mut trie_logits = vec![0.0f32; vocab.len()];
        engine
            .mask_by_simulation(state, &mut oracle_logits)
            .expect("matching vocab length");
        engine
            .mask_by_trie(state, &mut trie_logits)
            .expect("fixture stays inside the stack limits");
        for tok in 0..vocab.len() {
            let oracle_blocked = oracle_logits[tok] == f32::NEG_INFINITY;
            let trie_allowed = trie_logits[tok] != f32::NEG_INFINITY;
            if oracle_blocked && trie_allowed {
                over_accepts.push((tok, vocab[tok].clone()));
            }
        }
    }
    assert!(
        over_accepts.is_empty(),
        "trie over-accepted {} token(s) the oracle rejects (P0 soundness \
             violation): {:?}",
        over_accepts.len(),
        over_accepts
            .iter()
            .take(5)
            .map(|(id, bytes)| (id, String::from_utf8_lossy(bytes).to_string()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn trie_routes_over_cap_states_through_mask_logits() {
    // End-to-end wiring check: the public `mask_logits` entry point
    // (not the private helpers) must actually route over-cap states
    // through the trie and produce the same result as calling the
    // oracle directly — guards against the routing edit in
    // `mask_logits` silently going stale if the fallback call site
    // changes again.
    let vocab = trie_diff_vocab();
    let spec = GrammarSpec::JsonSchema(trie_diff_schema());
    let engine = GrammarEngine::new(&spec, vocab.clone()).unwrap();
    let corpus = harvest_trajectory_states(&engine);
    let deep_state = corpus
        .iter()
        .find(|s| engine.find_state_id(s).is_none())
        .expect("corpus must contain at least one over-cap state");

    let mut via_mask_logits = vec![0.0f32; vocab.len()];
    let mut oracle_logits = vec![0.0f32; vocab.len()];
    let mut state_for_public_call = deep_state.clone();
    engine
        .mask_logits(&mut state_for_public_call, &mut via_mask_logits)
        .expect("matching vocab length");
    engine
        .mask_by_simulation(deep_state, &mut oracle_logits)
        .expect("matching vocab length");

    assert_eq!(
        via_mask_logits, oracle_logits,
        "mask_logits on an over-cap state must match the oracle mask exactly"
    );
}

/// Same differential as `trie_mask_byte_identical_to_oracle_over_corpus` /
/// `trie_mask_never_over_accepts_vs_oracle`, but over the REAL model
/// tokenizer vocabulary instead of the synthetic `trie_diff_vocab`. The
/// synthetic vocab (256 single bytes + ~50 fragments) is a stand-in that
/// exercises trie branching and prefix sharing but is roughly three
/// orders of magnitude smaller than a real ~248K-token vocab, so it
/// cannot rule out a size- or content-dependent trie bug that only
/// surfaces at production scale. `#[ignore]`d because it loads a real
/// tokenizer and simulates the oracle over the full vocab (slow, and
/// requires a model checkout on disk); run explicitly with a `real_vocab`
/// test filter.
#[test]
#[ignore = "loads a real tokenizer + full vocab oracle simulation; run with `real_vocab` filter"]
fn trie_mask_byte_identical_to_oracle_real_vocab() {
    let home = std::env::var("HOME").expect("HOME must be set");
    let tokenizer_dir_str = std::env::var("LATTICE_TOKENIZER_DIR")
        .unwrap_or_else(|_| format!("{home}/.lattice/models/qwen3.5-0.8b"));
    let tokenizer_dir = std::path::Path::new(&tokenizer_dir_str);
    let config_path = tokenizer_dir.join("config.json");
    let tokenizer_path = tokenizer_dir.join("tokenizer.json");
    if !config_path.exists() || !tokenizer_path.exists() {
        panic!(
            "real-vocab differential test skipped: no model checkout at \
                 {tokenizer_dir_str} (expected config.json + tokenizer.json); \
                 set LATTICE_TOKENIZER_DIR to point at one"
        );
    }

    let cfg = crate::model::qwen35_config::Qwen35Config::from_model_dir(tokenizer_dir)
        .expect("config.json load");
    let tokenizer = crate::tokenizer::BpeTokenizer::from_tokenizer_json(&tokenizer_path)
        .expect("tokenizer.json load");
    let vocab = tokenizer
        .vocab_bytes(cfg.vocab_size)
        .expect("vocab_bytes over real tokenizer");
    assert!(
        vocab.len() > 100_000,
        "expected a production-scale vocab (~248K tokens for Qwen3.5), got {}",
        vocab.len()
    );

    let spec = GrammarSpec::JsonSchema(trie_diff_schema());
    let engine = GrammarEngine::new(&spec, vocab.clone()).unwrap();
    assert!(
        engine.exceeds_state_budget(),
        "fixture must exceed the state cap for this differential to exercise \
             the fallback path (the point of this test) — got a real vocab of {} \
             tokens",
        vocab.len()
    );

    let corpus = harvest_trajectory_states(&engine);
    assert!(
        corpus.len() >= 20,
        "need at least 20 distinct harvested states, got {}",
        corpus.len()
    );

    let mut over_cap_states = 0usize;
    let mut mismatches: Vec<(usize, usize)> = Vec::new();
    let mut over_accepts: Vec<(usize, usize)> = Vec::new();
    for (i, state) in corpus.iter().enumerate() {
        let mut oracle_logits = vec![0.0f32; vocab.len()];
        let mut trie_logits = vec![0.0f32; vocab.len()];
        engine
            .mask_by_simulation(state, &mut oracle_logits)
            .expect("matching vocab length");
        engine
            .mask_by_trie(state, &mut trie_logits)
            .expect("fixture stays inside the stack limits");
        if engine.find_state_id(state).is_none() {
            over_cap_states += 1;
        }
        for tok in 0..vocab.len() {
            if trie_logits[tok] != oracle_logits[tok] {
                mismatches.push((i, tok));
            }
            let oracle_blocked = oracle_logits[tok] == f32::NEG_INFINITY;
            let trie_allowed = trie_logits[tok] != f32::NEG_INFINITY;
            if oracle_blocked && trie_allowed {
                over_accepts.push((i, tok));
            }
        }
    }

    assert!(
        over_cap_states >= 10,
        "corpus should mostly cover the over-cap fallback path; got \
             {over_cap_states}/{}",
        corpus.len()
    );

    // Reported separately from the byte-identical check below so a
    // prune-logic regression that only over-accepts (P0 soundness) is
    // distinguishable from a broader mismatch (which could also include
    // under-accepts).
    assert!(
        over_accepts.is_empty(),
        "trie over-accepted {} token(s) the oracle rejects on the real \
             vocab (P0 soundness violation): {:?}",
        over_accepts.len(),
        over_accepts
            .iter()
            .take(5)
            .map(|(state_idx, tok)| (
                state_idx,
                tok,
                String::from_utf8_lossy(&vocab[*tok]).to_string()
            ))
            .collect::<Vec<_>>()
    );

    assert!(
        mismatches.is_empty(),
        "trie mask diverged from oracle mask on {} (state, token) pair(s) \
             over the real vocab: {:?}",
        mismatches.len(),
        mismatches
            .iter()
            .take(5)
            .map(|(state_idx, tok)| (
                state_idx,
                tok,
                String::from_utf8_lossy(&vocab[*tok]).to_string()
            ))
            .collect::<Vec<_>>()
    );

    eprintln!(
        "real_vocab differential: vocab_size={} corpus_states={} \
             over_cap_states={over_cap_states} over_accepts=0 mismatches=0",
        vocab.len(),
        corpus.len(),
    );
}
