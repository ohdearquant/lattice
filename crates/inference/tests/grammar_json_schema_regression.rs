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

// A generated
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

// Reverse $defs case: when an enum
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

#[test]
fn redundant_ref_narrowing_siblings_compile_target_language() {
    let cases: [(&str, serde_json::Value, &[u8], &[u8]); 5] = [
        (
            "identical const",
            serde_json::json!({
                "$defs": { "S": { "const": "a" } },
                "$ref": "#/$defs/S",
                "const": "a"
            }),
            b"\"a\"",
            b"\"b\"",
        ),
        (
            "target enum subset of sibling enum",
            serde_json::json!({
                "$defs": { "S": { "enum": [1, 2] } },
                "$ref": "#/$defs/S",
                "enum": [0, 1, 2, 3]
            }),
            b"2",
            b"3",
        ),
        (
            "identical declared type",
            serde_json::json!({
                "$defs": { "S": { "type": "string" } },
                "$ref": "#/$defs/S",
                "type": "string"
            }),
            b"\"ok\"",
            b"7",
        ),
        (
            "pinned values imply sibling type",
            serde_json::json!({
                "$defs": { "V": { "enum": ["ok", "err"] } },
                "$ref": "#/$defs/V",
                "type": ["string", "null"]
            }),
            b"\"ok\"",
            b"null",
        ),
        (
            "all narrowing siblings independently redundant",
            serde_json::json!({
                "$defs": { "V": { "type": "string", "const": "ok" } },
                "$ref": "#/$defs/V",
                "const": "ok",
                "enum": ["ok", "other"],
                "type": ["string", "null"]
            }),
            b"\"ok\"",
            b"\"other\"",
        ),
    ];

    for (shape, schema, accepted, target_excludes) in cases {
        let grammar = compile_json_schema(&schema)
            .unwrap_or_else(|err| panic!("{shape} should be proven redundant: {err}"));
        assert!(
            full_accept(&grammar, accepted),
            "{shape} must preserve a value accepted by the target"
        );
        assert!(
            !full_accept(&grammar, target_excludes),
            "{shape} must not drop the referenced target"
        );
    }
}

#[test]
fn redundant_ref_narrowing_target_annotations_compile_target_language() {
    let cases = [
        ("comment", "$comment", serde_json::json!("why")),
        ("title", "title", serde_json::json!("Value")),
        ("description", "description", serde_json::json!("A value")),
        ("default", "default", serde_json::json!("fallback")),
        ("examples", "examples", serde_json::json!(["example"])),
        ("deprecated", "deprecated", serde_json::json!(true)),
        ("read only", "readOnly", serde_json::json!(true)),
        ("write only", "writeOnly", serde_json::json!(true)),
    ];

    for (shape, annotation, value) in cases {
        let mut target = serde_json::json!({ "type": "string" });
        target
            .as_object_mut()
            .unwrap()
            .insert(annotation.to_string(), value);
        let schema = serde_json::json!({
            "$defs": { "V": target },
            "$ref": "#/$defs/V",
            "type": "string"
        });

        let grammar = compile_json_schema(&schema)
            .unwrap_or_else(|err| panic!("{shape} annotation should compile: {err}"));
        assert!(
            full_accept(&grammar, br#""accepted""#),
            "{shape} annotation must preserve a value accepted by the target"
        );
        assert!(
            !full_accept(&grammar, b"7"),
            "{shape} annotation must not drop the referenced target"
        );
    }
}

fn assert_unmodeled_target_ref_narrowing_fails_closed(
    shape: &str,
    target: serde_json::Value,
    accepted_by_target: &[u8],
    rejected_by_target: &[u8],
) {
    let target_grammar = compile_json_schema(&target)
        .unwrap_or_else(|err| panic!("{shape} target should compile independently: {err}"));
    assert!(
        full_accept(&target_grammar, accepted_by_target),
        "{shape} must pin the target grammar's accepted language"
    );
    assert!(
        !full_accept(&target_grammar, rejected_by_target),
        "{shape} must pin the target grammar's rejected language"
    );

    let schema = serde_json::json!({
        "$defs": { "V": target },
        "$ref": "#/$defs/V",
        "type": "string"
    });
    let Err(err) = compile_json_schema(&schema) else {
        panic!("{shape} must fail closed");
    };
    assert!(
        err.0.contains("$ref"),
        "{shape} error should identify the unsupported `$ref` intersection: {err}"
    );
}

#[test]
fn ref_narrowing_target_with_any_of_fails_closed() {
    assert_unmodeled_target_ref_narrowing_fails_closed(
        "target anyOf alongside scalar type",
        serde_json::json!({
            "anyOf": [{ "type": "number" }],
            "type": "string"
        }),
        b"1",
        br#""x""#,
    );
}

#[test]
fn ref_narrowing_target_with_one_of_fails_closed() {
    assert_unmodeled_target_ref_narrowing_fails_closed(
        "target oneOf alongside scalar type",
        serde_json::json!({
            "oneOf": [{ "type": "number" }],
            "type": "string"
        }),
        b"1",
        br#""x""#,
    );
}

#[test]
fn ref_narrowing_target_with_all_of_fails_closed() {
    assert_unmodeled_target_ref_narrowing_fails_closed(
        "target allOf alongside scalar type",
        serde_json::json!({
            "allOf": [{ "type": "number" }],
            "type": "string"
        }),
        br#""x""#,
        b"1",
    );
}

#[test]
fn ref_narrowing_target_with_non_schema_type_fails_closed() {
    assert_unmodeled_target_ref_narrowing_fails_closed(
        "target type is neither string nor array",
        serde_json::json!({ "type": 7 }),
        b"1",
        b"not-json",
    );
}

#[test]
fn unproven_ref_narrowing_siblings_fail_closed() {
    let cases = [
        (
            "value sibling without target value bound",
            serde_json::json!({
                "$defs": { "S": { "type": "string" } },
                "$ref": "#/$defs/S",
                "enum": ["a"]
            }),
        ),
        (
            "type sibling without target bound",
            serde_json::json!({
                "$defs": { "V": {} },
                "$ref": "#/$defs/V",
                "type": "string"
            }),
        ),
        (
            "cross-type integer subset number",
            serde_json::json!({
                "$defs": { "N": { "type": "integer" } },
                "$ref": "#/$defs/N",
                "type": "number"
            }),
        ),
        (
            "strictly narrower integer sibling would otherwise widen to number and admit 1.5",
            serde_json::json!({
                "$defs": { "N": { "type": "number" } },
                "$ref": "#/$defs/N",
                "type": "integer"
            }),
        ),
        (
            "required mixed with redundant type",
            serde_json::json!({
                "$defs": {
                    "O": {
                        "type": "object",
                        "properties": { "x": { "type": "integer" } }
                    }
                },
                "$ref": "#/$defs/O",
                "type": "object",
                "required": ["x"]
            }),
        ),
        (
            "empty target type array",
            serde_json::json!({
                "$defs": { "V": { "type": [] } },
                "$ref": "#/$defs/V",
                "type": "string"
            }),
        ),
        (
            "non-empty target type array is not representable by this compiler",
            serde_json::json!({
                "$defs": { "V": { "type": ["string", "null"] } },
                "$ref": "#/$defs/V",
                "type": ["string", "null"]
            }),
        ),
        (
            "target const plus enum dispatches enum before const",
            serde_json::json!({
                "$defs": { "S": { "const": "a", "enum": [1] } },
                "$ref": "#/$defs/S",
                "type": "string"
            }),
        ),
        (
            "target enum dispatch bypasses contradictory scalar type",
            serde_json::json!({
                "$defs": { "V": { "enum": [1, "a"], "type": "integer" } },
                "$ref": "#/$defs/V",
                "type": "integer"
            }),
        ),
        (
            "target const dispatch bypasses contradictory scalar type",
            serde_json::json!({
                "$defs": { "V": { "const": 1, "type": "string" } },
                "$ref": "#/$defs/V",
                "type": "string"
            }),
        ),
        (
            "empty target enum",
            serde_json::json!({
                "$defs": { "V": { "enum": [] } },
                "$ref": "#/$defs/V",
                "enum": ["a"]
            }),
        ),
    ];

    for (shape, schema) in cases {
        let Err(err) = compile_json_schema(&schema) else {
            panic!("{shape} must fail closed");
        };
        assert!(
            err.0.contains("$ref"),
            "{shape} error should identify the unsupported `$ref` intersection: {err}"
        );
    }
}
