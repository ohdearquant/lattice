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

fn assert_integer_literal_outcomes(value: serde_json::Value, expected: bool) {
    for target in [
        serde_json::json!({"const": value}),
        serde_json::json!({"enum": [value]}),
    ] {
        let mut direct = target.clone();
        direct["type"] = serde_json::json!("integer");
        let referenced = serde_json::json!({
            "$defs": {"V": target},
            "$ref": "#/$defs/V",
            "type": "integer"
        });
        for schema in [direct, referenced] {
            let result = compile_json_schema(&schema);
            assert_eq!(
                result.is_ok(),
                expected,
                "schema={schema}, error={:?}",
                result.as_ref().err()
            );
            if let Ok(grammar) = result {
                assert!(full_accept(&grammar, value.to_string().as_bytes()));
                assert!(!full_accept(&grammar, b"42"));
                assert!(!full_accept(&grammar, b"null"));
            }
        }
    }
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

#[test]
fn integer_literals_share_float_bounds_across_direct_and_ref() {
    for (value, expected) in [
        (1.0, true),
        (9_007_199_254_740_991.0, true),
        (-9_007_199_254_740_991.0, true),
        (9_007_199_254_740_992.0, false),
        (-9_007_199_254_740_992.0, false),
        (9_007_199_254_740_994.0, false),
        (-9_007_199_254_740_994.0, false),
        (1e16, false),
        (-1e16, false),
        (1.5, false),
        (-1.5, false),
    ] {
        let number = serde_json::Number::from_f64(value).unwrap();
        assert!(number.is_f64());
        assert!(!number.is_i64() && !number.is_u64());
        assert_integer_literal_outcomes(number.into(), expected);
    }
}

#[test]
fn integer_literals_preserve_exact_integer_representations() {
    for number in [
        serde_json::Number::from(9_007_199_254_740_992_i64),
        serde_json::Number::from(-9_007_199_254_740_992_i64),
        serde_json::Number::from(9_007_199_254_740_993_i64),
        serde_json::Number::from(-9_007_199_254_740_993_i64),
        serde_json::Number::from(9_007_199_254_740_993_u64),
        serde_json::Number::from(i64::MIN),
        serde_json::Number::from(u64::MAX),
    ] {
        assert!(number.is_i64() || number.is_u64());
        assert!(!number.is_f64());
        assert_integer_literal_outcomes(number.into(), true);
    }
}

#[test]
fn integer_literals_reject_nonnumeric_values_without_coercion() {
    for value in [serde_json::json!("1"), serde_json::json!(true)] {
        assert!(!value.is_number());
        assert_integer_literal_outcomes(value, false);
    }
}

#[test]
fn integer_mixed_enums_filter_direct_literals_but_refuse_ref_narrowing() {
    for value in [9_007_199_254_740_992.0, -9_007_199_254_740_992.0] {
        let number = serde_json::Number::from_f64(value).unwrap();
        assert!(number.is_f64());
        assert!(!number.is_i64() && !number.is_u64());
        let grammar = compile_json_schema(&serde_json::json!({
            "type": "integer",
            "enum": [1, number]
        }))
        .expect("a valid enum alternative survives direct integer filtering");
        assert!(full_accept(&grammar, b"1"));
        assert!(!full_accept(&grammar, number.to_string().as_bytes()));
        assert!(!full_accept(&grammar, b"42"));

        let schema = serde_json::json!({
            "$defs": {"V": {"enum": [1, number]}},
            "$ref": "#/$defs/V",
            "type": "integer"
        });
        let error = compile_json_schema(&schema)
            .expect_err("the type sibling cannot narrow a mixed referenced enum");
        assert!(error.to_string().contains("$ref"));
    }
}
