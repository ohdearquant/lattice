use lattice_grammar::json_schema::compile_json_schema;
use lattice_grammar::pda::{CompiledGrammar, GrammarState, StepResult, advance_byte};

fn full_accept(grammar: &CompiledGrammar, input: &[u8]) -> bool {
    let mut state = GrammarState::initial();
    for &byte in input {
        if advance_byte(&mut state, grammar, byte) != StepResult::Accepted {
            return false;
        }
    }
    state.is_complete()
}

#[test]
fn integer_typed_integral_float_overlaps_integer_const() {
    let schema = serde_json::json!({
        "oneOf": [
            {"type": "integer", "const": 1.0},
            {"const": 1}
        ]
    });
    let error = compile_json_schema(&schema).expect_err("1.0 and 1 overlap as integers");
    assert!(error.to_string().contains("oneOf"));
}

#[test]
fn integer_type_array_integral_float_overlaps_integer_const() {
    let schema = serde_json::json!({
        "oneOf": [
            {"type": ["integer"], "const": 1.0},
            {"const": 1}
        ]
    });
    let error = compile_json_schema(&schema).expect_err("1.0 and 1 overlap as integers");
    assert!(error.to_string().contains("oneOf"));
}

#[test]
fn integer_type_sibling_accepts_integral_float_ref_target() {
    let schema = serde_json::json!({
        "$defs": {"V": {"const": 1.0}},
        "$ref": "#/$defs/V",
        "type": "integer"
    });
    let grammar = compile_json_schema(&schema)
        .expect("the integer type sibling is redundant for an integral float");
    assert!(full_accept(&grammar, b"1.0"));
    assert!(!full_accept(&grammar, b"1.5"));
}

#[test]
fn integer_typed_integral_float_targets_pass_redundant_ref_guard() {
    for target in [
        serde_json::json!({"type": "integer", "const": 1.0}),
        serde_json::json!({"type": "integer", "enum": [1.0]}),
    ] {
        let schema = serde_json::json!({
            "$defs": {"V": target},
            "$ref": "#/$defs/V",
            "type": "integer"
        });
        let grammar = compile_json_schema(&schema)
            .expect("an integral float satisfies the target and sibling integer types");
        assert!(full_accept(&grammar, b"1.0"));
    }
}

#[test]
fn integer_type_sibling_rejects_fractional_float_ref_target() {
    let schema = serde_json::json!({
        "$defs": {"V": {"const": 1.5}},
        "$ref": "#/$defs/V",
        "type": "integer"
    });
    let error = compile_json_schema(&schema).expect_err("a fractional target must fail closed");
    assert!(error.to_string().contains("$ref"));
}

#[test]
fn integer_typed_fractional_const_does_not_create_overlap() {
    let schema = serde_json::json!({
        "oneOf": [
            {"type": "integer", "const": 1.5},
            {"const": 1.5}
        ]
    });
    compile_json_schema(&schema).expect("the contradictory branch cannot create an overlap");
}

#[test]
fn integer_typed_large_neighbors_remain_disjoint() {
    let schema = serde_json::json!({
        "oneOf": [
            {"type": "integer", "const": 9007199254740993i64},
            {"const": 9007199254740992i64}
        ]
    });
    compile_json_schema(&schema)
        .expect("distinct integers beyond f64 precision must stay distinct");
}

#[test]
fn contradictory_target_type_still_fails_closed() {
    let schema = serde_json::json!({
        "$defs": {"V": {"type": "integer", "enum": [1, "x"]}},
        "$ref": "#/$defs/V",
        "type": "integer"
    });
    let error = compile_json_schema(&schema).expect_err("an inconsistent target must fail closed");
    assert!(error.to_string().contains("$ref"));
}
