use lattice_grammar::json_schema::compile_json_schema;
use lattice_grammar::pda::{CompiledGrammar, GrammarState, StepResult, advance_byte};
use serde_json::{Value, json};

fn full_accept(grammar: &CompiledGrammar, value: &str) -> bool {
    let mut state = GrammarState::initial();
    for byte in value.bytes() {
        if advance_byte(&mut state, grammar, byte) != StepResult::Accepted {
            return false;
        }
    }
    state.is_complete()
}

fn assert_language(schema: Value, strings: &[&str], rejected: &[&str]) {
    let grammar = compile_json_schema(&schema)
        .unwrap_or_else(|error| panic!("union must compile: {error}; schema={schema}"));
    for value in ["true", "false"] {
        assert!(
            full_accept(&grammar, value),
            "boolean peer lost {value:?}; schema={schema}"
        );
    }
    for value in strings {
        assert!(
            full_accept(&grammar, value),
            "missing complete string {value:?}; schema={schema}"
        );
    }
    for value in rejected {
        assert!(
            !full_accept(&grammar, value),
            "unexpected complete value {value:?}; schema={schema}"
        );
    }
}

fn with_bare_target(union: &str, types: &[&str], branch: Value) -> Value {
    let mut schema = json!({"$defs": {"target": {"type": types}}});
    schema[union] = json!([branch, {"type": "boolean"}]);
    schema
}

fn broad_strings_survive(union: &str, types: &[&str]) {
    assert_language(
        with_bare_target(
            union,
            types,
            json!({"$ref": "#/$defs/target", "type": "string"}),
        ),
        &[
            "\"\"",
            "\"ordinary\"",
            "\"unrelated text\"",
            "\"café 🦀\"",
            r#""line\nquote\"slash\\end""#,
            r#""\u0061""#,
        ],
        &[
            "null",
            "7",
            "-2.5",
            "[]",
            "{}",
            "\"",
            "\"unfinished",
            "\"trailing\\",
            "\"raw\nnewline\"",
            r#""\q""#,
        ],
    );
}

#[test]
fn ref_bare_type_array_strings_anyof_string_first() {
    broad_strings_survive("anyOf", &["string", "null"]);
}

#[test]
fn ref_bare_type_array_strings_anyof_null_first() {
    broad_strings_survive("anyOf", &["null", "string"]);
}

#[test]
fn ref_bare_type_array_strings_oneof_string_first() {
    broad_strings_survive("oneOf", &["string", "null"]);
}

#[test]
fn ref_bare_type_array_strings_oneof_null_first() {
    broad_strings_survive("oneOf", &["null", "string"]);
}

fn chain_narrowing_survives(union: &str) {
    for types in [["string", "null"], ["null", "string"]] {
        for (branch, middle) in [
            (
                json!({"$ref": "#/$defs/middle", "type": "string", "const": "keep"}),
                json!({"$ref": "#/$defs/target", "enum": ["keep", "keep-tail"]}),
            ),
            (
                json!({"$ref": "#/$defs/middle"}),
                json!({"$ref": "#/$defs/target", "const": "keep"}),
            ),
            (
                json!({"$ref": "#/$defs/middle", "enum": ["keep"]}),
                json!({"$ref": "#/$defs/target", "enum": ["keep", "keep-tail"]}),
            ),
        ] {
            let mut schema = with_bare_target(union, &types, branch);
            schema["$defs"]["middle"] = middle;
            assert_language(
                schema,
                &["\"keep\""],
                &[
                    "\"\"",
                    "\"keep-tail\"",
                    "\"other\"",
                    "\"keep",
                    "null",
                    "7",
                    "[]",
                    "{}",
                ],
            );
        }
    }
}

#[test]
fn ref_bare_type_array_chain_narrowing_anyof() {
    chain_narrowing_survives("anyOf");
}

#[test]
fn ref_bare_type_array_chain_narrowing_oneof() {
    chain_narrowing_survives("oneOf");
}

fn no_string_target_stays_empty(union: &str) {
    for types in [["integer", "null"], ["null", "integer"]] {
        assert_language(
            with_bare_target(
                union,
                &types,
                json!({"$ref": "#/$defs/target", "type": "string"}),
            ),
            &[],
            &["\"\"", "\"ordinary\"", "null", "7", "-2.5", "[]", "{}"],
        );
    }
}

#[test]
fn ref_bare_no_string_type_array_stays_empty_anyof() {
    no_string_target_stays_empty("anyOf");
}

#[test]
fn ref_bare_no_string_type_array_stays_empty_oneof() {
    no_string_target_stays_empty("oneOf");
}

#[test]
fn ref_bare_type_array_contradictory_chain_stays_empty() {
    for union in ["anyOf", "oneOf"] {
        let mut schema = with_bare_target(
            union,
            &["string", "null"],
            json!({"$ref": "#/$defs/middle", "enum": ["right"]}),
        );
        schema["$defs"]["middle"] = json!({"$ref": "#/$defs/target", "const": "left"});
        assert_language(
            schema,
            &[],
            &["\"\"", "\"left\"", "\"right\"", "null", "7", "[]", "{}"],
        );
    }
}

#[test]
fn ref_terminal_type_array_uses_scalar_string_keyword_policy() {
    for union in ["anyOf", "oneOf"] {
        let mut schema = with_bare_target(
            union,
            &["string", "null"],
            json!({"$ref": "#/$defs/target", "type": "string"}),
        );
        schema["$defs"]["target"]["pattern"] = json!("^a");
        schema["$defs"]["target"]["minLength"] = json!(3);
        schema["$defs"]["target"]["maxLength"] = json!(5);
        schema["$defs"]["target"]["format"] = json!("email");
        assert_language(schema, &["\"z\"", "\"ordinary\""], &["null", "7", "{}"]);
    }
}

#[test]
fn ref_terminal_empty_and_annotation_only_schemas_preserve_string_intersection() {
    for union in ["anyOf", "oneOf"] {
        for terminal in [
            json!({}),
            json!({"description": "any JSON value"}),
            json!({"title": "value", "$comment": "schema annotation"}),
        ] {
            let mut schema = with_bare_target(
                union,
                &["string", "null"],
                json!({"$ref": "#/$defs/target", "type": "string"}),
            );
            schema["$defs"]["target"] = terminal;
            assert_language(
                schema.clone(),
                &["\"\"", "\"ordinary\""],
                &["null", "7", "[]", "{}"],
            );
            schema[union][0]["const"] = json!("keep");
            assert_language(schema, &["\"keep\""], &["\"other\"", "\"keepx\"", "null"]);
        }
    }
}

#[test]
fn ref_terminal_type_array_propagates_unsupported_type() {
    for union in ["anyOf", "oneOf"] {
        let mut schema = with_bare_target(
            union,
            &["string", "vector", "null"],
            json!({"$ref": "#/$defs/target", "type": "string"}),
        );
        schema["$defs"]["target"]["pattern"] = json!("^a");
        let Err(error) = compile_json_schema(&schema) else {
            panic!("unknown type did not surface from {schema}");
        };
        assert_eq!(error.0, "unsupported type: vector");
    }
}

#[test]
fn ref_terminal_does_not_widen_structural_assertions() {
    for union in ["anyOf", "oneOf"] {
        let mut schema = with_bare_target(
            union,
            &["string", "null"],
            json!({"$ref": "#/$defs/target", "type": "string"}),
        );
        schema["$defs"]["target"]["anyOf"] = json!([{"const": "allowed"}]);
        assert_language(schema, &[], &["\"other\"", "null", "7"]);
    }
}

#[test]
fn ref_terminal_does_not_widen_unsupported_reference_siblings() {
    for union in ["anyOf", "oneOf"] {
        for mut middle in [
            json!({"anyOf": [{"const": "allowed"}]}),
            json!({"enum": "invalid"}),
            json!({"type": ["string", "vector"]}),
        ] {
            let mut schema = with_bare_target(
                union,
                &["string", "null"],
                json!({"$ref": "#/$defs/middle", "type": "string"}),
            );
            middle["$ref"] = json!("#/$defs/target");
            schema["$defs"]["middle"] = middle;
            schema["$defs"]["target"] = json!({});
            assert_language(schema, &[], &["\"other\"", "null", "7"]);
        }
    }
}
