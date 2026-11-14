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

fn assert_language(schema: Value, accepted: &[&str], rejected: &[&str]) {
    let grammar = compile_json_schema(&schema)
        .unwrap_or_else(|error| panic!("union must compile: {error}; schema={schema}"));
    for value in accepted {
        assert!(
            full_accept(&grammar, value),
            "missing complete value {value:?}; schema={schema}"
        );
    }
    for value in rejected {
        assert!(
            !full_accept(&grammar, value),
            "unexpected complete value {value:?}; schema={schema}"
        );
    }
}

fn with_terminal(union: &str, terminal: Value, branch: Value) -> Value {
    let mut schema = json!({"$defs": {"target": terminal}});
    schema[union] = json!([branch, {"type": "boolean"}]);
    schema
}

fn string_ref() -> Value {
    json!({"$ref": "#/$defs/target", "type": "string"})
}

fn type_array_member_survives(union: &str) {
    for types in [json!(["string", "null"]), json!(["null", "string"])] {
        let schema = with_terminal(
            union,
            json!({"type": types, "enum": ["x", null]}),
            string_ref(),
        );
        assert_language(
            schema,
            &["true", "false", "\"x\""],
            &["\"y\"", "\"xy\"", "\"x", "null", "7"],
        );
    }
}

#[test]
fn ref_type_array_enum_string_member_survives_anyof() {
    type_array_member_survives("anyOf");
}

#[test]
fn ref_type_array_enum_string_member_survives_oneof() {
    type_array_member_survives("oneOf");
}

fn mixed_enum_member_survives(union: &str) {
    let schema = with_terminal(union, json!({"enum": ["x", 7]}), string_ref());
    assert_language(
        schema,
        &["true", "false", "\"x\""],
        &["\"y\"", "\"xy\"", "\"x", "null", "7"],
    );
}

#[test]
fn ref_mixed_enum_string_member_survives_anyof() {
    mixed_enum_member_survives("anyOf");
}

#[test]
fn ref_mixed_enum_string_member_survives_oneof() {
    mixed_enum_member_survives("oneOf");
}

#[test]
fn ref_no_string_terminal_stays_empty() {
    for union in ["anyOf", "oneOf"] {
        for terminal in [
            json!({"enum": [7, null]}),
            json!({"type": ["integer", "null"], "enum": ["x", 7, null]}),
        ] {
            assert_language(
                with_terminal(union, terminal, string_ref()),
                &["true", "false"],
                &["\"x\"", "\"y\"", "7", "null"],
            );
        }
    }
}

#[test]
fn ref_mixed_terminal_without_string_forcing_preserves_members() {
    for union in ["anyOf", "oneOf"] {
        assert_language(
            with_terminal(
                union,
                json!({"type": ["string", "null"], "enum": ["x", null]}),
                json!({"$ref": "#/$defs/target"}),
            ),
            &["true", "false", "\"x\"", "null"],
            &["\"y\"", "\"xy\"", "\"x", "7"],
        );
        assert_language(
            with_terminal(
                union,
                json!({"enum": ["x", 7]}),
                json!({"$ref": "#/$defs/target"}),
            ),
            &["true", "false", "\"x\"", "7"],
            &["\"y\"", "\"xy\"", "\"x", "null", "8"],
        );
    }
}

fn chain_narrowing_survives(union: &str) {
    for terminal in [
        json!({"type": ["string", "null"], "enum": ["pre", "prefix", null]}),
        json!({"enum": ["pre", "prefix", 7]}),
    ] {
        let mut schema = with_terminal(
            union,
            terminal,
            json!({"$ref": "#/$defs/middle", "type": "string"}),
        );
        schema["$defs"]["middle"] =
            json!({"$ref": "#/$defs/target", "enum": ["pre", "not-in-target"]});
        assert_language(
            schema,
            &["true", "false", "\"pre\""],
            &[
                "\"prefix\"",
                "\"not-in-target\"",
                "\"pre-extra\"",
                "\"pre",
                "null",
                "7",
            ],
        );
    }
}

#[test]
fn ref_terminal_string_intersection_keeps_chain_narrowing_anyof() {
    chain_narrowing_survives("anyOf");
}

#[test]
fn ref_terminal_string_intersection_keeps_chain_narrowing_oneof() {
    chain_narrowing_survives("oneOf");
}

fn const_and_enum_siblings_force_strings(union: &str) {
    for terminal in [
        json!({"type": ["string", "null"], "enum": ["x", "xy", null]}),
        json!({"enum": ["x", "xy", 7]}),
    ] {
        for mut branch in [
            json!({"const": "x"}),
            json!({"enum": ["x", "not-in-target"]}),
        ] {
            branch["$ref"] = json!("#/$defs/target");
            assert_language(
                with_terminal(union, terminal.clone(), branch),
                &["true", "false", "\"x\""],
                &["\"xy\"", "\"not-in-target\"", "\"x", "null", "7"],
            );
        }
    }
}

#[test]
fn ref_const_and_enum_siblings_force_string_intersection_anyof() {
    const_and_enum_siblings_force_strings("anyOf");
}

#[test]
fn ref_const_and_enum_siblings_force_string_intersection_oneof() {
    const_and_enum_siblings_force_strings("oneOf");
}

#[test]
fn ref_intermediate_contradiction_stays_empty() {
    for union in ["anyOf", "oneOf"] {
        let mut schema = with_terminal(
            union,
            json!({"enum": ["pre", "prefix", 7]}),
            json!({"$ref": "#/$defs/middle", "type": "string"}),
        );
        schema["$defs"]["middle"] =
            json!({"$ref": "#/$defs/target", "const": "prefix", "enum": ["pre"]});
        assert_language(
            schema,
            &["true", "false"],
            &["\"pre\"", "\"prefix\"", "7", "null"],
        );
    }
}

#[test]
fn ref_type_array_terminal_const_conflicts_stay_empty() {
    for union in ["anyOf", "oneOf"] {
        for constant in [Value::Null, json!("y")] {
            assert_language(
                with_terminal(
                    union,
                    json!({"type": ["string", "null"], "enum": ["x", null], "const": constant}),
                    string_ref(),
                ),
                &["true", "false"],
                &["\"x\"", "\"y\"", "null", "7"],
            );
        }
    }
}

fn type_array_terminal_const_keeps_matching_string(union: &str) {
    assert_language(
        with_terminal(
            union,
            json!({"type": ["string", "null"], "enum": ["x", "xy", null], "const": "x"}),
            string_ref(),
        ),
        &["true", "false", "\"x\""],
        &["\"xy\"", "\"y\"", "\"x", "null", "7"],
    );
}

#[test]
fn ref_type_array_terminal_const_keeps_matching_string_anyof() {
    type_array_terminal_const_keeps_matching_string("anyOf");
}

#[test]
fn ref_type_array_terminal_const_keeps_matching_string_oneof() {
    type_array_terminal_const_keeps_matching_string("oneOf");
}

#[test]
fn ref_mixed_terminal_const_intersection_is_preserved() {
    for union in ["anyOf", "oneOf"] {
        for constant in [Value::Null, json!("y"), json!("x")] {
            let retains_x = constant == json!("x");
            let schema = with_terminal(
                union,
                json!({"enum": ["x", 7], "const": constant}),
                string_ref(),
            );
            if retains_x {
                assert_language(
                    schema,
                    &["true", "false", "\"x\""],
                    &["\"y\"", "\"xy\"", "\"x", "null", "7"],
                );
            } else {
                assert_language(schema, &["true", "false"], &["\"x\"", "\"y\"", "null", "7"]);
            }
        }
    }
}

#[test]
fn ref_terminal_raw_enum_count_precedes_filtering() {
    for union in ["anyOf", "oneOf"] {
        for with_type_array in [false, true] {
            let mut values = vec![Value::Null; 4097];
            values[0] = json!("x");
            let mut terminal = json!({"enum": values, "const": "x"});
            if with_type_array {
                terminal["type"] = json!(["string", "null"]);
            }
            let schema = with_terminal(union, terminal, string_ref());
            let error = compile_json_schema(&schema)
                .expect_err("raw terminal enum count must be checked before const filtering");
            assert_eq!(
                error.0, "anyOf string literal count (4097) exceeds the supported limit (4096)",
                "union={union}, with_type_array={with_type_array}"
            );
        }
    }
}

#[test]
fn ref_terminal_raw_string_bytes_precede_filtering() {
    for union in ["anyOf", "oneOf"] {
        for with_type_array in [false, true] {
            let mut terminal = json!({"enum": ["x", "y".repeat(1048577), null], "const": "x"});
            if with_type_array {
                terminal["type"] = json!(["string", "null"]);
            }
            let schema = with_terminal(union, terminal, string_ref());
            let error = compile_json_schema(&schema)
                .expect_err("raw terminal string bytes must be checked before const filtering");
            assert_eq!(
                error.0,
                "string literal byte length (1048577) exceeds the supported limit (1048576)",
                "union={union}, with_type_array={with_type_array}"
            );
        }
    }
}
