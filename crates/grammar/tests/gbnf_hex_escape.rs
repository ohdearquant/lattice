use lattice_grammar::gbnf::parse_gbnf;
use lattice_grammar::pda::{CompiledGrammar, GrammarState, SimResult, simulate_token};

fn accepts(grammar: &CompiledGrammar, input: &[u8]) -> bool {
    let (result, state) = simulate_token(&GrammarState::initial(), grammar, input);
    result == SimResult::Accept && state.is_complete()
}

#[test]
fn hex_escaped_range_is_refused_or_matches_exactly_the_digits() {
    let Ok(grammar) = parse_gbnf("root ::= [\\x30-\\x39]\n") else {
        return;
    };
    for byte in u8::MIN..=u8::MAX {
        assert_eq!(
            accepts(&grammar, &[byte]),
            byte.is_ascii_digit(),
            "wrong language for byte {byte:#04x}"
        );
    }
    assert!(!accepts(&grammar, b""));
    assert!(!accepts(&grammar, b"12"));
}

#[test]
fn hex_escaped_literal_is_refused_or_matches_exactly_a() {
    let Ok(grammar) = parse_gbnf("root ::= \"\\x41\"\n") else {
        return;
    };
    assert!(accepts(&grammar, b"A"));
    assert!(!accepts(&grammar, b"x41"));
    assert!(!accepts(&grammar, b""));
}

#[test]
fn ordinary_range_and_unicode_string_escape_still_work() {
    let range = parse_gbnf("root ::= [0-9]\n").unwrap();
    for byte in u8::MIN..=u8::MAX {
        assert_eq!(accepts(&range, &[byte]), byte.is_ascii_digit());
    }
    let literal = parse_gbnf("root ::= \"\\u0041\"\n").unwrap();
    assert!(accepts(&literal, b"A"));
    assert!(!accepts(&literal, b"x41"));
}
