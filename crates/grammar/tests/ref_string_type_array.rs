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
