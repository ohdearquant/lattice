//! Regression tests for grammar JSON-schema compiler fixes (issue #310, findings #4 and #6).

use lattice_inference::grammar::json_schema::compile_json_schema;
use lattice_inference::grammar::pda::{CompiledGrammar, GrammarState, StepResult, advance_byte};

/// True iff `g` accepts every byte of `s` and ends in a complete (accepting) state.
fn full_accept(g: &CompiledGrammar, s: &[u8]) -> bool {
    let mut st = GrammarState::initial();
    for &b in s {
        if advance_byte(&mut st, g, b) == StepResult::Rejected {
            return false;
        }
    }
    st.is_complete()
}

// Finding #6: number integer part must reject leading zeros, keep valid forms.
#[test]
fn number_rejects_leading_zero() {
    let g = compile_json_schema(&serde_json::json!({"type":"number"})).unwrap();
    assert!(full_accept(&g, b"0"), "0 must be accepted");
    assert!(full_accept(&g, b"10"), "10 must be accepted");
    assert!(full_accept(&g, b"-5"), "-5 must be accepted");
    assert!(full_accept(&g, b"3.14"), "3.14 must be accepted");
    assert!(full_accept(&g, b"1e10"), "1e10 must be accepted");
    assert!(
        full_accept(&g, b"0.5"),
        "0.5 must be accepted (zero before fraction is legal)"
    );
    assert!(
        !full_accept(&g, b"01"),
        "01 (leading zero) must be rejected"
    );
    assert!(!full_accept(&g, b"00"), "00 must be rejected");
}

#[test]
fn integer_rejects_leading_zero() {
    let g = compile_json_schema(&serde_json::json!({"type":"integer"})).unwrap();
    assert!(full_accept(&g, b"0"), "0 must be accepted");
    assert!(full_accept(&g, b"42"), "42 must be accepted");
    assert!(!full_accept(&g, b"007"), "007 must be rejected");
}

// Finding #4: distinct enums whose join("_") names collide must not share alternatives.
#[test]
fn enum_helper_names_do_not_collide() {
    let g = compile_json_schema(&serde_json::json!({
        "type":"object",
        "properties":{
            "x":{"type":"string","enum":["ab","c_d"]},
            "y":{"type":"string","enum":["ab_c","d"]}
        },
        "required":["x","y"]
    }))
    .unwrap();
    // Each field must accept its OWN enum values.
    assert!(
        full_accept(&g, br#"{"x":"ab","y":"d"}"#),
        "y must accept its own value d"
    );
    assert!(
        full_accept(&g, br#"{"x":"c_d","y":"ab_c"}"#),
        "x=c_d, y=ab_c must be accepted"
    );
}
