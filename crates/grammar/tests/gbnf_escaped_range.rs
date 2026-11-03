use lattice_grammar::gbnf::parse_gbnf;
use lattice_grammar::pda::{CompiledGrammar, GrammarState, SimResult, simulate_token};

fn accepts(grammar: &CompiledGrammar, input: &[u8]) -> bool {
    let (result, state) = simulate_token(&GrammarState::initial(), grammar, input);
    result == SimResult::Accept && state.is_complete()
}

#[test]
fn escaped_range_endpoints_match_only_tab_through_carriage_return() {
    let grammar = parse_gbnf("root ::= [\\t-\\r]\n").expect("escaped endpoints");
    for byte in u8::MIN..=u8::MAX {
        assert_eq!(
            accepts(&grammar, &[byte]),
            (b'\t'..=b'\r').contains(&byte),
            "wrong language for byte {byte:#04x}"
        );
    }
    assert!(!accepts(&grammar, b""));
    assert!(!accepts(&grammar, b"\t\n"));
}

#[test]
fn standalone_escapes_and_plain_ranges_keep_their_languages() {
    let escapes = parse_gbnf("root ::= [\\t\\r]\n").unwrap();
    assert!(accepts(&escapes, b"\t"));
    assert!(accepts(&escapes, b"\r"));
    assert!(!accepts(&escapes, b"\n"));
    let range = parse_gbnf("root ::= [0-9]\n").unwrap();
    for byte in u8::MIN..=u8::MAX {
        assert_eq!(accepts(&range, &[byte]), byte.is_ascii_digit());
    }
}
