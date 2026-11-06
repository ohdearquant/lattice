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
fn integer_mixed_out_of_range_enums_refuse_direct_and_ref_forms() {
    for value in [9_007_199_254_740_992.0, -9_007_199_254_740_992.0] {
        let number = serde_json::Number::from_f64(value).unwrap();
        assert!(number.is_f64());
        assert!(!number.is_i64() && !number.is_u64());
        let direct_error = compile_json_schema(&serde_json::json!({
            "type": "integer",
            "enum": [1, number]
        }))
        .expect_err("an unsupported integer alternative must not be silently filtered");
        assert!(direct_error.to_string().contains("unsupported schema"));
        assert!(
            direct_error
                .to_string()
                .contains("exactly-representable integer range")
        );

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

#[test]
fn integer_range_refusals_preserve_actual_type_mismatch_filtering() {
    for value in [9_007_199_254_740_992.0, -9_007_199_254_740_992.0] {
        let enum_error = compile_json_schema(&serde_json::json!({
            "type": "integer", "enum": [1, value]
        }))
        .expect_err("unsupported integer enum members must refuse the whole schema");
        let const_error = compile_json_schema(&serde_json::json!({
            "type": "integer", "const": value
        }))
        .expect_err("an unsupported integer constant must name its representation limit");
        assert_eq!(enum_error, const_error);
        let message = enum_error.to_string();
        assert!(message.contains("unsupported schema"), "{message}");
        assert!(
            message.contains("exactly-representable integer range"),
            "{message}"
        );
        assert!(message.contains("2^53"), "{message}");
        assert!(!message.contains("unsatisfiable"), "{message}");
    }

    for values in [
        serde_json::json!([1]),
        serde_json::json!([1, 1.5]),
        serde_json::json!([1, 1.5, "1", true]),
    ] {
        let grammar = compile_json_schema(&serde_json::json!({
            "type": "integer", "enum": values
        }))
        .expect("actual non-integer members may be filtered");
        assert!(full_accept(&grammar, b"1"));
        for input in [b"1.5".as_slice(), b"\"1\"", b"true", b"2"] {
            assert!(!full_accept(&grammar, input));
        }
    }
    for value in [
        serde_json::json!(1.5),
        serde_json::json!("1"),
        serde_json::json!(true),
    ] {
        let error = compile_json_schema(&serde_json::json!({
            "type": "integer", "const": value
        }))
        .expect_err("an actual type contradiction remains unsatisfiable");
        assert!(error.to_string().contains("unsatisfiable"));
    }
}

#[test]
fn integer_const_excludes_out_of_range_float_enum_members() {
    for (constant, excluded) in [
        (1_i64, 9_007_199_254_740_992_i64),
        (9_007_199_254_740_993, 9_007_199_254_740_992),
        (-9_007_199_254_740_993, -9_007_199_254_740_992),
    ] {
        let float = serde_json::Number::from_f64(excluded as f64).unwrap();
        assert!(float.is_f64());
        let grammar = compile_json_schema(&serde_json::json!({
            "type": "integer",
            "const": constant,
            "enum": [float, constant]
        }))
        .expect("const excludes the unsupported float before its representation is checked");
        assert!(full_accept(&grammar, constant.to_string().as_bytes()));
        assert!(!full_accept(&grammar, float.to_string().as_bytes()));
        assert!(!full_accept(&grammar, excluded.to_string().as_bytes()));
        assert!(!full_accept(&grammar, b"null"));
    }
}

#[test]
fn integer_const_retaining_out_of_range_float_enum_member_refuses() {
    for value in [9_007_199_254_740_992.0, -9_007_199_254_740_992.0] {
        let error = compile_json_schema(&serde_json::json!({
            "type": "integer",
            "const": value,
            "enum": [1, value]
        }))
        .expect_err("const keeps this unsupported float in the enum's language");
        let message = error.to_string();
        assert!(message.contains("unsupported schema"), "{message}");
        assert!(
            message.contains("exactly-representable integer range"),
            "{message}"
        );
        assert!(message.contains("2^53"), "{message}");
    }
}

#[test]
fn typed_literal_union_drop_uses_dispatch_result() {
    for type_name in ["number", "integer", "boolean", "null", "array", "object"] {
        for literal in [
            serde_json::json!({"type": type_name, "const": "dead"}),
            serde_json::json!({"type": type_name, "enum": ["dead"]}),
        ] {
            for union in ["anyOf", "oneOf"] {
                let mut schema = serde_json::json!({});
                schema[union] = serde_json::json!([literal, {"const": 7}]);
                let grammar = compile_json_schema(&schema)
                    .unwrap_or_else(|error| panic!("direct branch in {schema}: {error}"));
                assert!(full_accept(&grammar, b"7"), "{schema}");
                for input in [br#""dead""#.as_slice(), b"null", b"true", b"[]", b"{}"] {
                    assert!(!full_accept(&grammar, input), "{schema}: {input:?}");
                }
            }
        }
    }
}

#[test]
fn typed_literal_nested_and_reference_failures_remain_fatal() {
    let dead = serde_json::json!({"type": "boolean", "enum": [1]});
    let mut branches = vec![
        serde_json::json!({
            "type": "object", "properties": {"a": dead}, "required": ["a"]
        }),
        serde_json::json!({"anyOf": [dead], "description": "not a flattened union"}),
        serde_json::json!({"oneOf": [dead]}),
        serde_json::json!({"$ref": "#/$defs/Dead"}),
    ];
    for malformed_ref in [
        serde_json::Value::Null,
        serde_json::json!(false),
        serde_json::json!(3),
        serde_json::json!([]),
        serde_json::json!({}),
    ] {
        branches.push(serde_json::json!({
            "$ref": malformed_ref, "type": "boolean", "const": 1
        }));
    }
    for branch in branches {
        for union in ["anyOf", "oneOf"] {
            let mut schema = serde_json::json!({"$defs": {"Dead": dead}});
            schema[union] = serde_json::json!([branch, {"const": 7}]);
            let error = compile_json_schema(&schema)
                .expect_err("only a direct typed-literal failure may be discarded");
            assert_eq!(
                error,
                lattice_grammar::json_schema::SchemaError(
                    "unsatisfiable typed non-string literal intersection".to_string()
                ),
                "{schema}"
            );
            assert_eq!(
                error.to_string(),
                "JSON Schema compile error: unsatisfiable typed non-string literal intersection"
            );
        }
    }
}

#[test]
fn typed_literal_dispatch_preserves_reference_and_enum_priority() {
    for union in ["anyOf", "oneOf"] {
        let mut schema = serde_json::json!({});
        schema[union] = serde_json::json!([
            {"$ref": "#/$defs/missing", "type": "integer", "const": "dead"},
            {"const": true}
        ]);
        let error = compile_json_schema(&schema).expect_err("$ref precedes a typed literal");
        assert_eq!(error.0, "$ref not found: #/$defs/missing");

        schema[union] = serde_json::json!([
            {"type": "integer", "enum": [1], "const": 7},
            {"const": true}
        ]);
        let grammar = compile_json_schema(&schema).expect("enum and const must both hold");
        assert!(full_accept(&grammar, b"true"));
        assert!(!full_accept(&grammar, b"1"));
        assert!(!full_accept(&grammar, b"7"));

        for (values, message) in [
            (
                vec![serde_json::json!(1); 4097],
                "enum value count (4097) exceeds the supported limit (4096)",
            ),
            (
                vec![serde_json::json!("x".repeat(2048)); 512],
                "enum value encoded byte length exceeds the supported limit (1048576)",
            ),
        ] {
            schema[union] = serde_json::json!([
                {"type": "integer", "enum": values, "const": 7},
                {"const": true}
            ]);
            let error = compile_json_schema(&schema)
                .expect_err("raw literal budgets precede filtering and branch dropping");
            assert_eq!(error.0, message);
        }
    }
}

#[test]
fn numeric_const_enum_intersection_detects_oneof_overlap() {
    let schema = serde_json::json!({
        "oneOf": [
            {"type": "integer", "const": 1, "enum": [1.0]},
            {"const": 1}
        ]
    });
    let error = compile_json_schema(&schema)
        .expect_err("numeric const and enum values agree, so the branches overlap");
    assert!(error.to_string().contains("oneOf"));
}

#[test]
fn float_const_and_integer_enum_detect_oneof_overlap() {
    let schema = serde_json::json!({
        "oneOf": [
            {"const": 1.0, "enum": [1]},
            {"const": 1}
        ]
    });
    let error = compile_json_schema(&schema)
        .expect_err("integer and float spellings of the same value overlap");
    assert!(error.to_string().contains("oneOf"));
}

#[test]
fn distinct_large_numeric_const_enum_values_do_not_create_overlap() {
    let schema = serde_json::json!({
        "oneOf": [
            {"const": 9007199254740993i64, "enum": [9007199254740992.0]},
            {"const": 9007199254740993i64}
        ]
    });
    compile_json_schema(&schema).expect("distinct large values must not be equated through f64");
}

#[test]
fn fractional_const_enum_match_detects_oneof_overlap() {
    let schema = serde_json::json!({
        "oneOf": [
            {"const": 1.5, "enum": [1.5]},
            {"const": 1.5}
        ]
    });
    let error = compile_json_schema(&schema).expect_err("equal fractional values overlap");
    assert!(error.to_string().contains("oneOf"));
}

#[test]
fn different_fractional_and_integer_values_do_not_create_overlap() {
    let schema = serde_json::json!({
        "oneOf": [
            {"const": 1.5, "enum": [1]},
            {"const": 1.5}
        ]
    });
    compile_json_schema(&schema).expect("the contradictory branch cannot overlap");
}

#[test]
fn unsigned_range_integer_and_integral_float_compile() {
    let mut schema: serde_json::Value = serde_json::from_str(
        r#"{"oneOf":[{"const":9223372036854775808},{"const":9.223372036854775808e18}]}"#,
    )
    .unwrap();
    assert_eq!(
        schema["oneOf"][0]["const"].as_u64(),
        Some(9_223_372_036_854_775_808)
    );
    assert!(schema["oneOf"][1]["const"].is_f64());
    assert_eq!(schema["oneOf"][1]["const"].as_f64(), Some(2f64.powi(63)));
    for _ in 0..2 {
        compile_json_schema(&schema)
            .expect("a float outside the exact range cannot prove an integer overlap");
        schema["oneOf"].as_array_mut().unwrap().reverse();
    }
}

#[test]
fn unsigned_range_integer_neighbor_and_integral_float_remain_distinct() {
    let mut schema: serde_json::Value = serde_json::from_str(
        r#"{"oneOf":[{"const":9223372036854775809},{"const":9.223372036854775808e18}]}"#,
    )
    .unwrap();
    assert_eq!(
        schema["oneOf"][0]["const"].as_u64(),
        Some(9_223_372_036_854_775_809)
    );
    assert!(schema["oneOf"][1]["const"].is_f64());
    assert_eq!(schema["oneOf"][1]["const"].as_f64(), Some(2f64.powi(63)));
    for _ in 0..2 {
        compile_json_schema(&schema)
            .expect("the exact u64 neighbor must not round into the integral float's overlap key");
        schema["oneOf"].as_array_mut().unwrap().reverse();
    }
}

#[test]
fn numeric_siblings_accept_equal_ref_targets() {
    for (target_value, sibling_value, accepted) in [
        (serde_json::json!(1), serde_json::json!(1.0), &b"1"[..]),
        (serde_json::json!(1.0), serde_json::json!(1), &b"1.0"[..]),
    ] {
        for target in [
            serde_json::json!({"const": target_value.clone()}),
            serde_json::json!({"enum": [target_value.clone()]}),
        ] {
            for (keyword, sibling) in [
                ("const", sibling_value.clone()),
                ("enum", serde_json::json!([sibling_value.clone(), 3])),
            ] {
                let mut schema = serde_json::json!({
                    "$defs": {"V": target.clone()},
                    "$ref": "#/$defs/V"
                });
                schema[keyword] = sibling;
                let grammar = compile_json_schema(&schema).unwrap_or_else(|error| {
                    panic!("the {keyword} sibling must contain the equal target {target}: {error}")
                });
                assert!(full_accept(&grammar, accepted), "schema: {schema}");
                for rejected in [b"2".as_slice(), b"3", b"null"] {
                    assert!(!full_accept(&grammar, rejected), "schema: {schema}");
                }
            }
        }
    }
}

#[test]
fn nested_numeric_siblings_accept_equal_ref_targets() {
    for (integer_value, float_value, distinct_value) in [
        (
            serde_json::json!([1]),
            serde_json::json!([1.0]),
            serde_json::json!([2]),
        ),
        (
            serde_json::json!({"x": 0}),
            serde_json::json!({"x": 0.0}),
            serde_json::json!({"x": 1}),
        ),
        (
            serde_json::json!({"outer": [{"x": 1}]}),
            serde_json::json!({"outer": [{"x": 1.0}]}),
            serde_json::json!({"outer": [{"x": 2}]}),
        ),
    ] {
        for (target_value, sibling_value) in [
            (&integer_value, &float_value),
            (&float_value, &integer_value),
        ] {
            for target in [
                serde_json::json!({"const": target_value}),
                serde_json::json!({"enum": [target_value]}),
            ] {
                for (keyword, sibling) in [
                    ("const", sibling_value.clone()),
                    ("enum", serde_json::json!([sibling_value])),
                ] {
                    let mut schema = serde_json::json!({
                        "$defs": {"V": target.clone()},
                        "$ref": "#/$defs/V"
                    });
                    schema[keyword] = sibling;
                    let grammar = compile_json_schema(&schema).unwrap_or_else(|error| {
                        panic!("nested numeric sibling must be redundant: {schema}: {error}")
                    });
                    assert!(
                        full_accept(&grammar, target_value.to_string().as_bytes()),
                        "schema: {schema}"
                    );
                    assert!(
                        !full_accept(&grammar, distinct_value.to_string().as_bytes()),
                        "schema: {schema}"
                    );
                    assert!(!full_accept(&grammar, b"null"), "schema: {schema}");
                }
            }
        }
    }
}

#[test]
fn numeric_ref_siblings_preserve_containment_and_target_shape_guards() {
    for (keyword, sibling) in [
        ("const", serde_json::json!(1.0)),
        ("enum", serde_json::json!([1.0])),
    ] {
        for target in [
            serde_json::json!({"const": 2}),
            serde_json::json!({"enum": [1, 2]}),
            serde_json::json!({"const": 1, "enum": [1]}),
        ] {
            let mut schema = serde_json::json!({
                "$defs": {"V": target},
                "$ref": "#/$defs/V"
            });
            schema[keyword] = sibling.clone();
            let error = compile_json_schema(&schema).expect_err(
                "a matching member must not prove containment or a modeled target shape",
            );
            assert_eq!(
                error.0,
                format!(
                    "unsupported schema: `{keyword}` sibling alongside `$ref` requires an intersection with the referenced schema that this grammar cannot represent safely"
                )
            );
        }
    }
}

#[test]
fn rounded_unsigned_neighbor_does_not_prove_oneof_overlap() {
    let mut schema: serde_json::Value = serde_json::from_str(
        r#"{"oneOf":[{"const":9223372036854775808},{"const":9223372036854775809.0}]}"#,
    )
    .unwrap();
    assert_eq!(
        schema["oneOf"][0]["const"].as_u64(),
        Some(9_223_372_036_854_775_808)
    );
    assert!(schema["oneOf"][1]["const"].is_f64());
    assert_eq!(schema["oneOf"][1]["const"].as_f64(), Some(2f64.powi(63)));
    for _ in 0..2 {
        compile_json_schema(&schema)
            .expect("the rounded float neighbor must not prove an overlap with the exact integer");
        schema["oneOf"].as_array_mut().unwrap().reverse();
    }
}

#[test]
fn rounded_exactness_boundary_does_not_prove_oneof_overlap() {
    // 2^53 is the first float that also stands for a different integer, 2^53 + 1,
    // so a float equal to 2^53 cannot show that the two literals are the same value.
    let mut schema: serde_json::Value = serde_json::from_str(
        r#"{"oneOf":[{"const":9007199254740992},{"const":9007199254740992.0}]}"#,
    )
    .unwrap();
    assert_eq!(
        schema["oneOf"][0]["const"].as_u64(),
        Some(9_007_199_254_740_992)
    );
    assert!(schema["oneOf"][1]["const"].is_f64());
    assert_eq!(schema["oneOf"][1]["const"].as_f64(), Some(2f64.powi(53)));
    for _ in 0..2 {
        compile_json_schema(&schema)
            .expect("a float at the exactness boundary cannot prove an integer overlap");
        schema["oneOf"].as_array_mut().unwrap().reverse();
    }
}

#[test]
fn small_integer_and_integral_float_overlap() {
    let mut schema = serde_json::json!({"oneOf": [{"const": 1}, {"const": 1.0}]});
    for _ in 0..2 {
        let error = compile_json_schema(&schema)
            .expect_err("small integer and integral float literals still prove an overlap");
        assert!(
            error
                .0
                .starts_with("oneOf branch 1 overlaps an earlier branch on literal value ")
        );
        assert!(
            error
                .0
                .ends_with("oneOf requires mutually exclusive branches")
        );
        schema["oneOf"].as_array_mut().unwrap().reverse();
    }
}
