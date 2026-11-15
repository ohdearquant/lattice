//! Typed literal intersections: `enum` and `const` under a string or array-valued `type`, and `type`
//! siblings on `$ref` chain nodes that contribute string values.

use super::*;

#[test]
fn type_array_enum_rejects_incompatible_complete_values() {
    for types in [["string", "null"], ["null", "string"]] {
        let schema = serde_json::json!({"type": types, "enum": ["ok", null, 7]});
        let grammar = compile(&schema).expect("mixed type-array enum must compile");
        assert!(
            rejects(&grammar, b"7"),
            "array-type enum must reject incompatible complete value 7: {schema}"
        );
    }
}

#[test]
fn type_array_enum_preserves_each_compatible_member() {
    for types in [["string", "null"], ["null", "string"]] {
        let schema = serde_json::json!({"type": types, "enum": ["ok", null, 7]});
        let grammar = compile(&schema).unwrap();
        for input in [b"\"ok\"".as_slice(), b"null"] {
            assert!(accepts(&grammar, input), "lost member {input:?}: {schema}");
        }
        for input in [b"\"other\"".as_slice(), b"nul", b"nullx", b"\"ok\"x"] {
            assert!(rejects(&grammar, input), "accepted nonmember {input:?}");
        }
    }
}

#[test]
fn type_array_enum_empty_intersections_report_unsatisfiable() {
    for (schema, forbidden) in [
        (
            serde_json::json!({"type": ["integer", "null"], "enum": ["x"]}),
            b"\"x\"".as_slice(),
        ),
        (
            serde_json::json!({"type": ["string", "null"], "enum": [7]}),
            b"7",
        ),
        (
            serde_json::json!({"type": ["string", "null"], "enum": ["keep", null, 7], "const": "absent"}),
            b"null",
        ),
    ] {
        let result = compile(&schema);
        if let Ok(grammar) = &result {
            assert!(
                rejects(grammar, forbidden),
                "empty array-type enum accepted {forbidden:?}: {schema}"
            );
        }
        let error = result.expect_err("empty array-type enum must fail closed");
        assert!(
            error.0.contains("unsatisfiable"),
            "unexpected empty-intersection error: {error}"
        );
    }
}

#[test]
fn type_array_enum_singleton_preserves_scalar_string_language() {
    for enum_values in [
        serde_json::json!(["ok", 7]),
        serde_json::json!(["foo", "food"]),
    ] {
        let scalar = compile(&serde_json::json!({"type": "string", "enum": enum_values})).unwrap();
        let array = compile(&serde_json::json!({"type": ["string"], "enum": enum_values})).unwrap();
        for input in [
            b"\"ok\"".as_slice(),
            b"\"foo\"",
            b"\"food\"",
            b"\"other\"",
            b"null",
        ] {
            assert_eq!(
                accepts(&array, input),
                accepts(&scalar, input),
                "scalar/array mismatch for {input:?}"
            );
        }
    }
}

#[test]
fn type_array_enum_preserves_sibling_const_intersection() {
    let schema = serde_json::json!({"type": ["string", "null"], "enum": ["keep", "drop", null, 7], "const": "keep"});
    let grammar = compile(&schema).unwrap();
    assert!(accepts(&grammar, b"\"keep\""));
    for input in [b"\"drop\"".as_slice(), b"null", b"7"] {
        assert!(
            rejects(&grammar, input),
            "array-type enum ignored sibling const for {input:?}"
        );
    }
}

#[test]
fn type_array_enum_checks_raw_count_before_type_and_const() {
    for with_const in [false, true] {
        let mut values = vec![Value::String("discarded".into()); 4096];
        values.push(Value::Null);
        let mut schema = serde_json::json!({"type": ["null", "boolean"], "enum": values});
        if with_const {
            schema["const"] = Value::Null;
        }
        let error = compile(&schema).expect_err("raw count must precede narrowing");
        assert_eq!(
            error.0,
            "enum value count (4097) exceeds the supported limit (4096)"
        );
    }
}

#[test]
fn type_array_enum_checks_raw_bytes_before_type_and_const() {
    for with_const in [false, true] {
        let long = "a".repeat(600 * 1024);
        let mut schema = serde_json::json!({"type": ["null", "boolean"], "enum": [null, long, format!("{long}b")]});
        if with_const {
            schema["const"] = Value::Null;
        }
        let error = compile(&schema).expect_err("raw byte budget must precede narrowing");
        assert_eq!(
            error.0,
            "enum value encoded byte length exceeds the supported limit (1048576)"
        );
    }
}

#[test]
fn type_array_enum_preserves_other_literal_kinds() {
    for (schema, allowed, forbidden) in [
        (
            serde_json::json!({"type": ["boolean", "null"], "enum": [true, false, null, "x", 7]}),
            vec![b"true".as_slice(), b"false", b"null"],
            vec![b"\"x\"".as_slice(), b"7"],
        ),
        (
            serde_json::json!({"type": ["array", "object"], "enum": [[1], {"x": 1}, null, 7]}),
            vec![b"[1]".as_slice(), b"{\"x\":1}"],
            vec![b"null".as_slice(), b"7"],
        ),
    ] {
        let grammar = compile(&schema).unwrap();
        for input in allowed {
            assert!(accepts(&grammar, input), "lost {input:?}: {schema}");
        }
        for input in forbidden {
            assert!(
                rejects(&grammar, input),
                "array-type enum retained {input:?}: {schema}"
            );
        }
    }
}

#[test]
fn type_array_enum_preserves_direct_unsatisfiable_union_policy() {
    let schema = serde_json::json!({"anyOf": [
        {"type": ["integer", "null"], "enum": ["x"]}, {"const": true}
    ]});
    let grammar = compile(&schema).unwrap();
    assert!(accepts(&grammar, b"true"));
    assert!(
        rejects(&grammar, b"\"x\""),
        "dead direct array-type enum was not removed"
    );
    let nested = serde_json::json!({"anyOf": [
        {"type": "object", "properties": {"v": {"type": ["integer", "null"], "enum": ["x"]}}, "required": ["v"]},
        {"const": true}
    ]});
    let error =
        compile(&nested).expect_err("nested unsatisfiable literal must not be silently dropped");
    assert!(error.0.contains("unsatisfiable"), "{error}");
}

#[test]
fn type_array_enum_checks_all_string_raw_bytes_before_const() {
    let long = "a".repeat(600 * 1024);
    let schema = serde_json::json!({"type": ["string", "null"], "enum": ["keep", long, format!("{long}b")], "const": "keep"});
    let error = compile(&schema)
        .expect_err("array-type raw bytes must precede const even for all-string input");
    assert_eq!(
        error.0,
        "enum value encoded byte length exceeds the supported limit (1048576)"
    );
    let admitted =
        compile_ok(r#"{"type":["string","null"],"enum":["keep","drop"],"const":"keep"}"#);
    assert!(accepts(&admitted, b"\"keep\""));
    assert!(rejects(&admitted, b"\"drop\""));
}

#[test]
fn type_array_enum_integer_range_refusal_precedes_empty_intersection() {
    for types in [
        serde_json::json!(["integer", "null"]),
        serde_json::json!(["integer", "integer", "null"]),
        serde_json::json!(["integer", "string"]),
        serde_json::json!(["integer", "boolean"]),
    ] {
        let schema = serde_json::json!({"type": types, "enum": [1e300]});
        let error = compile(&schema)
            .expect_err("retained integral float must preserve integer range refusal");
        assert_eq!(
            error.0,
            "unsupported schema: floating-point integer literal is outside the exactly-representable integer range (absolute value must be less than 2^53)"
        );
    }
}

#[test]
fn type_array_enum_number_alternative_covers_large_integral_float() {
    for types in [["integer", "number"], ["number", "integer"]] {
        let schema = serde_json::json!({"type": types, "enum": [9007199254740992.0]});
        let grammar = compile(&schema).expect("number alternative admits the retained float");
        assert!(accepts(&grammar, b"9007199254740992.0"));
        assert!(rejects(&grammar, b"null"));
        assert!(rejects(&grammar, b"9007199254740993"));
    }
}

#[test]
fn type_array_enum_const_excludes_unrepresentable_integer() {
    for types in [
        serde_json::json!(["integer", "null"]),
        serde_json::json!(["integer", "integer", "null"]),
        serde_json::json!(["integer", "string"]),
        serde_json::json!(["integer", "boolean"]),
    ] {
        let schema = serde_json::json!({"type": types, "enum": [1.0, 1e300], "const": 1});
        let grammar =
            compile(&schema).expect("const-excluded out-of-range float must not be checked");
        assert!(
            accepts(&grammar, b"1.0"),
            "shared equality retains the enum's float spelling"
        );
        assert!(
            rejects(&grammar, b"1"),
            "const spelling must not replace enum rendering"
        );
    }
}

#[test]
fn type_array_enum_const_retained_unrepresentable_integer_refuses() {
    let schema =
        serde_json::json!({"type": ["integer", "null"], "enum": [1.0, 1e300], "const": 1e300});
    let error = compile(&schema).expect_err("const-retained out-of-range float must still refuse");
    assert_eq!(
        error.0,
        "unsupported schema: floating-point integer literal is outside the exactly-representable integer range (absolute value must be less than 2^53)"
    );
}

#[test]
fn type_array_enum_integer_membership_keeps_literal_rendering() {
    for types in [
        serde_json::json!(["integer", "null"]),
        serde_json::json!(["null", "integer"]),
        serde_json::json!(["integer", "integer", "null"]),
    ] {
        let schema = serde_json::json!({"type": types, "enum": [1.0, 1.5, null, "x"]});
        let grammar = compile(&schema).unwrap();
        assert!(accepts(&grammar, b"1.0"));
        assert!(accepts(&grammar, b"null"));
        for input in [b"1".as_slice(), b"1.5", b"\"x\""] {
            assert!(
                rejects(&grammar, input),
                "integer type-array retained {input:?}"
            );
        }
    }
}

#[test]
fn type_array_enum_all_string_const_positive_control() {
    let admitted =
        compile_ok(r#"{"type":["string","null"],"enum":["keep","drop"],"const":"keep"}"#);
    assert!(accepts(&admitted, b"\"keep\""));
    assert!(rejects(&admitted, b"\"drop\""));
}

#[test]
fn standalone_const_string_type_rejects_complete_number() {
    let schema = serde_json::json!({"type": "string", "const": 5});
    let result = compile(&schema);
    if let Ok(grammar) = &result {
        assert!(
            rejects(grammar, b"5"),
            "string-typed standalone const accepted complete number 5"
        );
    }
    let error = result.expect_err("incompatible string const must fail closed");
    assert_eq!(
        error.0,
        "unsatisfiable typed non-string literal intersection"
    );
}

#[test]
fn standalone_const_string_type_preserves_full_literal() {
    let grammar = compile_ok(r#"{"type":"string","const":"ok"}"#);
    assert!(accepts(&grammar, b"\"ok\""));
    for input in [b"5".as_slice(), b"\"other\"", b"\"ok", b"\"ok\"x"] {
        assert!(rejects(&grammar, input), "accepted nonliteral {input:?}");
    }
}

#[test]
fn standalone_const_type_array_rejects_incompatible_complete_values() {
    for (schema, forbidden) in [
        (
            serde_json::json!({"type": ["string", "null"], "const": 7}),
            b"7".as_slice(),
        ),
        (
            serde_json::json!({"type": ["integer", "null"], "const": "x"}),
            b"\"x\"",
        ),
        (
            serde_json::json!({"type": ["boolean", "null"], "const": []}),
            b"[]",
        ),
        (
            serde_json::json!({"type": ["array", "object"], "const": false}),
            b"false",
        ),
    ] {
        let result = compile(&schema);
        if let Ok(grammar) = &result {
            assert!(
                rejects(grammar, forbidden),
                "array-typed standalone const accepted {forbidden:?}: {schema}"
            );
        }
        let error = result.expect_err("incompatible type-array const must fail closed");
        assert_eq!(
            error.0,
            "unsatisfiable typed non-string literal intersection"
        );
    }
}

#[test]
fn standalone_const_type_array_preserves_ordinary_members() {
    for (schema, literal) in [
        (
            serde_json::json!({"type": ["string", "null"], "const": "ok"}),
            b"\"ok\"".as_slice(),
        ),
        (
            serde_json::json!({"type": ["null", "string"], "const": null}),
            b"null",
        ),
        (
            serde_json::json!({"type": ["boolean", "null"], "const": true}),
            b"true",
        ),
        (
            serde_json::json!({"type": ["boolean", "null"], "const": false}),
            b"false",
        ),
        (
            serde_json::json!({"type": ["array", "object"], "const": [1]}),
            b"[1]",
        ),
        (
            serde_json::json!({"type": ["object", "array"], "const": {"x": 1}}),
            b"{\"x\":1}",
        ),
    ] {
        let grammar = compile(&schema).unwrap();
        assert!(
            accepts(&grammar, literal),
            "lost compatible constant: {schema}"
        );
        assert!(rejects(&grammar, b"\"other\""));
        let mut trailing = literal.to_vec();
        trailing.push(b'x');
        assert!(rejects(&grammar, &trailing));
    }
}

#[test]
fn standalone_const_type_array_number_covers_large_float() {
    for types in [["integer", "number"], ["number", "integer"]] {
        let schema = serde_json::json!({"type": types, "const": 9007199254740992.0});
        let grammar = compile(&schema).unwrap();
        assert!(accepts(&grammar, b"9007199254740992.0"));
        assert!(rejects(&grammar, b"9007199254740993"));
    }
}

#[test]
fn standalone_const_type_array_integer_range_error_precedes_empty() {
    for types in [
        serde_json::json!(["integer", "null"]),
        serde_json::json!(["integer", "integer", "null"]),
        serde_json::json!(["integer", "string"]),
        serde_json::json!(["integer", "boolean"]),
    ] {
        let schema = serde_json::json!({"type": types, "const": 1e300});
        let error =
            compile(&schema).expect_err("integer-only type set must preserve range refusal");
        assert_eq!(
            error.0,
            "unsupported schema: floating-point integer literal is outside the exactly-representable integer range (absolute value must be less than 2^53)"
        );
    }
}

#[test]
fn standalone_const_type_array_integer_rejects_fraction() {
    let schema = serde_json::json!({"type": ["integer", "null"], "const": 1.5});
    let result = compile(&schema);
    if let Ok(grammar) = &result {
        assert!(
            rejects(grammar, b"1.5"),
            "integer type-array accepted fractional const"
        );
    }
    let error = result.expect_err("fractional integer const is unsatisfiable");
    assert_eq!(
        error.0,
        "unsatisfiable typed non-string literal intersection"
    );
}

#[test]
fn standalone_const_type_array_keeps_integral_float_rendering() {
    for types in [
        serde_json::json!("integer"),
        serde_json::json!(["integer", "null"]),
        serde_json::json!(["null", "integer", "integer"]),
    ] {
        let schema = serde_json::json!({"type": types, "const": 1.0});
        let grammar = compile(&schema).unwrap();
        assert!(accepts(&grammar, b"1.0"));
        assert!(rejects(&grammar, b"1"));
    }
}

#[test]
fn standalone_const_type_array_preserves_direct_and_nested_policy() {
    let schema =
        serde_json::json!({"anyOf": [{"type": ["string", "null"], "const": 7}, {"const": true}]});
    let grammar = compile(&schema).unwrap();
    assert!(accepts(&grammar, b"true"));
    assert!(
        rejects(&grammar, b"7"),
        "direct incompatible array const survived union"
    );
    let nested = serde_json::json!({"anyOf": [
        {"type": "object", "properties": {"v": {"type": ["string", "null"], "const": 7}}, "required": ["v"]},
        {"const": true}
    ]});
    let error = compile(&nested).expect_err("nested incompatible const must propagate");
    assert_eq!(
        error.0,
        "unsatisfiable typed non-string literal intersection"
    );
}

#[test]
fn standalone_const_type_array_keeps_raw_string_guard() {
    let schema = serde_json::json!({"type": ["null"], "const": "x".repeat(1_048_577)});
    let error = compile(&schema).expect_err("raw string guard must precede const type filtering");
    assert_eq!(
        error.0,
        "string literal byte length (1048577) exceeds the supported limit (1048576)"
    );
}

#[test]
fn ref_string_sibling_type_rejects_reported_complete_value() {
    let schema = serde_json::json!({
        "anyOf": [{"$ref": "#/$defs/s", "type": "integer"}, {"const": 7}],
        "$defs": {"s": {"enum": ["x"]}}
    });
    match compile(&schema) {
        Ok(grammar) => {
            assert!(
                rejects(&grammar, b"\"x\""),
                "referenced branch with integer sibling accepted complete string x"
            );
            panic!("incompatible referenced type must use the existing refusal");
        }
        Err(error) => assert_eq!(
            error.0,
            "unsupported schema: `type` sibling alongside `$ref` requires an intersection with the referenced schema that this grammar cannot represent safely"
        ),
    }
}

#[test]
fn ref_string_sibling_type_excluding_string_fails_closed() {
    for union in ["anyOf", "oneOf"] {
        for target in [
            serde_json::json!({"enum": ["x"]}),
            serde_json::json!({"const": "x"}),
            serde_json::json!({"type": "string"}),
        ] {
            for sibling_type in [
                serde_json::json!("boolean"),
                serde_json::json!(["integer"]),
                serde_json::json!(["integer", "null"]),
            ] {
                let mut schema = serde_json::json!({"$defs": {"s": target.clone()}});
                schema[union] = serde_json::json!([
                    {"$ref": "#/$defs/s", "type": sibling_type}, {"const": 7}
                ]);
                let error = compile(&schema).expect_err(
                    "a string-excluding ref type must not be lost during union hoisting",
                );
                assert_eq!(
                    error.0,
                    "unsupported schema: `type` sibling alongside `$ref` requires an intersection with the referenced schema that this grammar cannot represent safely",
                    "schema: {schema}"
                );
            }
        }
    }
}

#[test]
fn ref_string_sibling_type_checks_intermediate_chain_node() {
    for sibling_type in [serde_json::json!("integer"), serde_json::json!(["null"])] {
        let schema = serde_json::json!({
            "$defs": {
                "outer": {"$ref": "#/$defs/s", "type": sibling_type},
                "s": {"enum": ["x"]}
            },
            "anyOf": [{"$ref": "#/$defs/outer"}, {"const": 7}]
        });
        let error = compile(&schema)
            .expect_err("intermediate ref type must reach the existing reference validator");
        assert_eq!(
            error.0,
            "unsupported schema: `type` sibling alongside `$ref` requires an intersection with the referenced schema that this grammar cannot represent safely"
        );
    }
}

#[test]
fn ref_string_sibling_type_keeps_compatible_literals_and_live_peer() {
    for sibling_type in [
        serde_json::json!("string"),
        serde_json::json!(["string", "null"]),
        serde_json::json!(["string", "string"]),
    ] {
        for intermediate in [false, true] {
            let typed_ref = serde_json::json!({"$ref": "#/$defs/s", "type": sibling_type});
            let branch = if intermediate {
                serde_json::json!({"$ref": "#/$defs/outer"})
            } else {
                typed_ref.clone()
            };
            let schema = serde_json::json!({
                "$defs": {"s": {"enum": ["x", "xy"]}, "outer": typed_ref},
                "anyOf": [branch, {"const": 7}]
            });
            let grammar = compile(&schema).expect("compatible referenced strings must stay live");
            for value in [b"\"x\"".as_slice(), b"\"xy\"", b"7"] {
                assert!(accepts(&grammar, value), "missing complete value {value:?}");
            }
            for value in [b"\"z\"".as_slice(), b"\"xyz\"", b"8", b"null", b"\"x"] {
                assert!(
                    rejects(&grammar, value),
                    "unexpected complete value {value:?}"
                );
            }
        }
    }
}

#[test]
fn ref_string_sibling_type_keeps_specialized_string_narrowing() {
    let schema = serde_json::json!({
        "$defs": {
            "s": {"type": "string"},
            "outer": {"$ref": "#/$defs/s", "type": ["string", "null"], "enum": ["x", "xy"]}
        },
        "anyOf": [{"$ref": "#/$defs/outer", "const": "x"}, {"const": 7}]
    });
    let grammar = compile(&schema).expect("specialized string intersections must remain supported");
    assert!(accepts(&grammar, b"\"x\""));
    assert!(accepts(&grammar, b"7"));
    for value in [b"\"xy\"".as_slice(), b"\"z\"", b"null", b"8", b"\"x"] {
        assert!(rejects(&grammar, value), "narrowing lost for {value:?}");
    }
}

#[test]
fn ref_string_sibling_type_keeps_specialized_nonstring_dead_branch() {
    for sibling in [
        serde_json::json!({"type": "string"}),
        serde_json::json!({"type": ["string"]}),
        serde_json::json!({"const": "x"}),
        serde_json::json!({"enum": ["x", "xy"]}),
        serde_json::json!({"type": "integer", "const": "x"}),
    ] {
        let mut branch = sibling;
        branch["$ref"] = serde_json::json!("#/$defs/n");
        let schema = serde_json::json!({
            "$defs": {"n": {"type": "integer"}},
            "anyOf": [branch, {"const": "ok"}]
        });
        let grammar =
            compile(&schema).expect("the existing proven dead-string branch remains droppable");
        assert!(accepts(&grammar, b"\"ok\""));
        for value in [b"\"x\"".as_slice(), b"\"xy\"", b"7", b"null"] {
            assert!(rejects(&grammar, value), "dead branch widened to {value:?}");
        }
    }
}

#[test]
fn ref_string_sibling_type_live_const_is_independently_supported() {
    let grammar = compile_ok(r#"{"const":7}"#);
    assert!(accepts(&grammar, b"7"));
    for value in [b"\"x\"".as_slice(), b"8", b"70", b"null"] {
        assert!(rejects(&grammar, value), "const widened to {value:?}");
    }
}

#[test]
fn ref_string_sibling_type_keeps_standalone_ref_refusal() {
    for sibling_type in [
        serde_json::json!("integer"),
        serde_json::json!(["integer"]),
        serde_json::json!([]),
        serde_json::json!(7),
        serde_json::json!("unknown"),
    ] {
        let schema = serde_json::json!({
            "$defs": {"s": {"enum": ["x"]}},
            "$ref": "#/$defs/s", "type": sibling_type
        });
        let error = compile(&schema).expect_err("standalone reference validation must stay strict");
        assert_eq!(
            error.0,
            "unsupported schema: `type` sibling alongside `$ref` requires an intersection with the referenced schema that this grammar cannot represent safely"
        );
    }
}
