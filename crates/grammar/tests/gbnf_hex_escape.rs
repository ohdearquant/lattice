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

#[test]
fn ascii_hex_class_and_both_escaped_range_endpoints_are_supported() {
    let literal = parse_gbnf("root ::= [\\x41]\n").unwrap();
    assert!(accepts(&literal, b"A"));
    assert!(!accepts(&literal, b"x"));
    let range = parse_gbnf("root ::= [\\x30-\\x39]\n").unwrap();
    for byte in u8::MIN..=u8::MAX {
        assert_eq!(accepts(&range, &[byte]), byte.is_ascii_digit());
    }
}

#[test]
fn string_hex_escape_decodes_the_scalar_to_utf8() {
    let ascii = parse_gbnf("root ::= \"\\x41\"\n").unwrap();
    assert!(accepts(&ascii, b"A"));
    assert!(!accepts(&ascii, b"x41"));
    let non_ascii = parse_gbnf("root ::= \"\\xC3\"\n").unwrap();
    assert!(accepts(&non_ascii, "Ã".as_bytes()));
    assert!(!accepts(&non_ascii, &[0xc3]));
}

#[test]
fn malformed_or_unsupported_hex_and_unicode_classes_are_refused() {
    for grammar in [
        "root ::= [\\x4]\n",
        "root ::= [\\xGG]\n",
        "root ::= [\\x80]\n",
        "root ::= [\\u0041]\n",
        "root ::= [\\x41-\\xG1]\n",
        "root ::= \"\\x4\"\n",
        "root ::= \"\\xGG\"\n",
    ] {
        assert!(parse_gbnf(grammar).is_err(), "must reject {grammar:?}");
    }
}
