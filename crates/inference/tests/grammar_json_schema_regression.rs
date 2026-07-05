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

// Finding #4 ($defs sub-case, raised in review): a generated
// `str_enum_N` helper rule must not overwrite a user `$defs` rule that
// happens to share the same name. Here `a` references a def literally named
// `str_enum_0` (an integer), and `b`'s enum helper would otherwise reserve
// `str_enum_0` and clobber the def's alternatives.
#[test]
fn enum_helper_does_not_clobber_user_defs_rule() {
    let g = compile_json_schema(&serde_json::json!({
        "$defs": {"str_enum_0": {"type":"integer"}},
        "type":"object",
        "properties":{
            "a":{"$ref":"#/$defs/str_enum_0"},
            "b":{"type":"string","enum":["0"]}
        },
        "required":["a","b"]
    }))
    .unwrap();
    assert!(
        full_accept(&g, br#"{"a":7,"b":"0"}"#),
        "a must stay integer-constrained (def rule not clobbered by enum helper)"
    );
    assert!(
        !full_accept(&g, br#"{"a":"0","b":"0"}"#),
        "a must reject a quoted string (still the integer def, not the enum)"
    );
}

// Finding #4 (reverse $defs sub-case, raised in re-review): when an enum
// property compiles BEFORE a `$ref` to a def of a colliding name, the `$ref`
// must resolve to the DEF, not alias to the enum helper. Here `a` (enum, first)
// would claim `str_enum_0`; `b` then references a def named `str_enum_0` that is
// an integer. `b` must stay integer-constrained.
#[test]
fn enum_then_ref_does_not_alias_to_enum_helper() {
    let g = compile_json_schema(&serde_json::json!({
        "$defs": {"str_enum_0": {"type":"integer"}},
        "type":"object",
        "properties":{
            "a":{"type":"string","enum":["0"]},
            "b":{"$ref":"#/$defs/str_enum_0"}
        },
        "required":["a","b"]
    }))
    .unwrap();
    assert!(
        full_accept(&g, br#"{"a":"0","b":7}"#),
        "b must resolve to the integer def, not alias to a's enum helper"
    );
    assert!(
        !full_accept(&g, br#"{"a":"0","b":"0"}"#),
        "b must reject a quoted string (it is the integer def, not the enum)"
    );
}

// A user `$defs` rule must not collide with a builtin rule name either: a def
// named `ws` (whitespace builtin) must keep its own integer constraint.
#[test]
fn user_defs_does_not_clobber_builtin_rule() {
    let g = compile_json_schema(&serde_json::json!({
        "$defs": {"ws": {"type":"integer"}},
        "type":"object",
        "properties":{"a":{"$ref":"#/$defs/ws"}},
        "required":["a"]
    }))
    .unwrap();
    assert!(
        full_accept(&g, br#"{"a":7}"#),
        "def named 'ws' must stay an integer, not alias the whitespace builtin"
    );
}
