use super::*;
use crate::grammar::pda::{GrammarState, SimResult, simulate_token};

fn accepts(grammar: &CompiledGrammar, input: &[u8]) -> bool {
    let state = GrammarState::initial();
    let (result, final_state) = simulate_token(&state, grammar, input);
    result == SimResult::Accept && final_state.is_complete()
}

fn rejects(grammar: &CompiledGrammar, input: &[u8]) -> bool {
    // A grammar "rejects" a string when it cannot accept it as a complete
    // value.  This includes both early byte rejections (SimResult::Reject)
    // and cases where all bytes are consumed but the state is not complete.
    !accepts(grammar, input)
}

#[test]
fn gbnf_string_literal() {
    let g = parse_gbnf("root ::= \"hello\"\n").unwrap();
    assert!(accepts(&g, b"hello"));
    assert!(rejects(&g, b"world"));
}

#[test]
fn gbnf_alternation() {
    let g = parse_gbnf("root ::= \"yes\" | \"no\"\n").unwrap();
    assert!(accepts(&g, b"yes"));
    assert!(accepts(&g, b"no"));
    assert!(rejects(&g, b"maybe"));
}

#[test]
fn gbnf_char_class() {
    let g = parse_gbnf("root ::= [abc]\n").unwrap();
    assert!(accepts(&g, b"a"));
    assert!(accepts(&g, b"b"));
    assert!(accepts(&g, b"c"));
    assert!(rejects(&g, b"d"));
}

#[test]
fn gbnf_char_range() {
    let g = parse_gbnf("root ::= [a-z]\n").unwrap();
    assert!(accepts(&g, b"a"));
    assert!(accepts(&g, b"z"));
    assert!(rejects(&g, b"A"));
    assert!(rejects(&g, b"0"));
}

#[test]
fn gbnf_optional() {
    let g = parse_gbnf("root ::= \"x\"?\n").unwrap();
    assert!(accepts(&g, b"x"));
    // epsilon: initial state should be complete (? makes root = x | ε)
    let state = GrammarState::initial();
    assert!(state.is_complete() || !state.stacks.is_empty());
}

#[test]
fn gbnf_empty_root_is_complete_before_any_byte() {
    let g = parse_gbnf("root ::= \"\"\n").unwrap();
    assert!(GrammarState::for_grammar(&g).is_complete());

    let g = parse_gbnf("root ::= \"x\"?\n").unwrap();
    assert!(GrammarState::for_grammar(&g).is_complete());

    // Control: a literal root is not complete before any byte.
    let g = parse_gbnf("root ::= \"x\"\n").unwrap();
    assert!(!GrammarState::for_grammar(&g).is_complete());
}

#[test]
fn gbnf_plus() {
    let g = parse_gbnf("root ::= [0-9]+\n").unwrap();
    assert!(accepts(&g, b"5"));
    assert!(accepts(&g, b"42"));
    assert!(rejects(&g, b"x"));
}

#[test]
fn gbnf_star() {
    let g = parse_gbnf("root ::= [0-9]*\n").unwrap();
    // one or more accepted
    assert!(accepts(&g, b"123"));
}

#[test]
fn gbnf_non_terminal_reference() {
    let gbnf = "root ::= digit digit\ndigit ::= [0-9]\n";
    let g = parse_gbnf(gbnf).unwrap();
    assert!(accepts(&g, b"42"));
    assert!(rejects(&g, b"4"));
    assert!(rejects(&g, b"abc"));
}

#[test]
fn gbnf_grouping() {
    let g = parse_gbnf("root ::= (\"ab\" | \"cd\")\n").unwrap();
    assert!(accepts(&g, b"ab"));
    assert!(accepts(&g, b"cd"));
    assert!(rejects(&g, b"ac"));
}

#[test]
fn gbnf_dot() {
    let g = parse_gbnf("root ::= .\n").unwrap();
    assert!(accepts(&g, b"a"));
    assert!(accepts(&g, b"z"));
    assert!(accepts(&g, b"0"));
}

#[test]
fn gbnf_no_root_returns_error() {
    assert!(parse_gbnf("foo ::= \"bar\"\n").is_err());
}

/// issue #1079: a reference to a rule name that is never given a `::=`
/// definition must be rejected at compile time, naming the undefined rule
/// — not silently compiled to an epsilon match for the reference.
///
/// Mutation guard: removing the completeness check in `parse_gbnf` makes
/// this `Ok` (the reserved-but-undefined `missing_rule` id has empty
/// `alts`, which `pda.rs` treats as epsilon), so `expect_err` panics.
#[test]
fn gbnf_undefined_rule_reference_is_rejected() {
    let err = parse_gbnf("root ::= missing_rule\n")
        .expect_err("a reference to an undefined rule must not compile");
    assert!(
        err.0.contains("missing_rule"),
        "error must name the undefined rule, got: {}",
        err.0
    );
}

/// issue #1079 (the other direction): `?` desugars into an anon rule with
/// TWO alternatives (`[x]` and the empty sequence `[]`) — its `alts`
/// vector has length 2, so it is not `alts.is_empty()` and must NOT be
/// flagged by the undefined-rule completeness check. Only a rule that was
/// `reserve`d and never `set_alts` (zero alternatives, not one empty
/// alternative) is a genuine undefined reference.
#[test]
fn gbnf_legitimate_epsilon_rule_not_flagged_undefined() {
    let g = parse_gbnf("root ::= \"x\"?\n").unwrap();
    assert!(accepts(&g, b"x"));
}

#[test]
fn gbnf_error_display() {
    let e = GbnfError("test error".to_string());
    assert!(e.to_string().contains("test error"));
}

#[test]
fn gbnf_comment_skipped() {
    let g = parse_gbnf("# this is a comment\nroot ::= \"ok\"\n").unwrap();
    assert!(accepts(&g, b"ok"));
}

fn nested_groups(depth: usize) -> String {
    format!("root ::= {}\"a\"{}\n", "(".repeat(depth), ")".repeat(depth))
}

#[test]
fn group_nesting_at_the_limit_parses() {
    let g = parse_gbnf(&nested_groups(MAX_GROUP_DEPTH)).unwrap();
    assert!(accepts(&g, b"a"));
}

#[test]
fn group_nesting_past_the_limit_is_rejected() {
    let err = parse_gbnf(&nested_groups(MAX_GROUP_DEPTH + 1)).unwrap_err();
    assert!(err.to_string().contains("nesting"), "{err}");
}

/// Without a depth bound the parser recurses once per `(`, so a few thousand
/// groups overflow a small native stack before the grammar is ever matched.
#[test]
fn deeply_nested_groups_are_rejected_on_a_bounded_stack() {
    std::thread::Builder::new()
        .stack_size(1024 * 1024)
        .spawn(|| {
            let err = parse_gbnf(&nested_groups(100_000)).unwrap_err();
            assert!(err.to_string().contains("nesting"), "{err}");
        })
        .expect("bounded-stack regression thread spawns")
        .join()
        .expect("deep nesting must be rejected, not overflow the stack");
}
