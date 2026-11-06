use lattice_grammar::json_schema::compile_json_schema;
use lattice_grammar::pda::{CompiledGrammar, GrammarState, StepResult, Symbol, advance_byte};
use serde_json::{Value, json};

fn impossible() -> Value {
    json!({"type": "string", "enum": [1, 2]})
}

fn compile(schema: &Value) -> CompiledGrammar {
    let grammar = compile_json_schema(schema).expect("productive schema must compile");
    let mut seen = vec![false; grammar.rules.len()];
    let mut pending = vec![0];
    while let Some(id) = pending.pop() {
        if seen[id] {
            continue;
        }
        seen[id] = true;
        let rule = &grammar.rules[id];
        assert!(
            !rule.alts.is_empty(),
            "reachable zero-alternative rule: {}",
            rule.name
        );
        for alt in &rule.alts {
            for symbol in alt {
                if let Symbol::NonTerminal(target) = symbol {
                    assert!(*target < grammar.rules.len());
                    pending.push(*target);
                }
            }
        }
    }
    grammar
}

fn accepts(grammar: &CompiledGrammar, input: &str) -> bool {
    let mut state = GrammarState::initial();
    for byte in input.bytes() {
        match advance_byte(&mut state, grammar, byte) {
            StepResult::Accepted => {}
            StepResult::Rejected => return false,
            StepResult::StackLimitExceeded => panic!("PDA limit while matching {input:?}"),
        }
    }
    state.is_complete()
}

fn assert_empty(schema: &Value) {
    let error = compile_json_schema(schema).expect_err("empty root must not become epsilon");
    assert!(error.0.contains("unsatisfiable"), "{error}");
}

#[test]
fn impossible_roots_fail_compilation() {
    for schema in [
        impossible(),
        json!({"type": "string", "enum": []}),
        json!({"type": "string", "enum": ["x"], "const": "y"}),
        json!({"anyOf": [impossible(), impossible()]}),
    ] {
        assert_empty(&schema);
    }
    let empty_string = compile(&json!({"const": ""}));
    assert!(accepts(&empty_string, r#""""#));
    assert!(!accepts(&empty_string, ""));
    assert!(!accepts(&empty_string, "null"));
}

#[test]
fn required_impossible_values_fail_compilation() {
    assert_empty(&json!({
        "type": "object", "properties": {"a": impossible()}, "required": ["a"]
    }));
    for min_items in [1, 2] {
        assert_empty(&json!({"type": "array", "items": impossible(), "minItems": min_items}));
    }
    assert_empty(&json!({"type": "array", "prefixItems": [{"const": true}, impossible()]}));
}

#[test]
fn optional_impossible_property_preserves_omission() {
    for additional in [false, true] {
        let grammar = compile(&json!({
            "type": "object", "properties": {"a": impossible(), "b": {"const": true}},
            "additionalProperties": additional
        }));
        for input in ["{}", "{ }", r#"{"b":true}"#] {
            assert!(accepts(&grammar, input), "{input}");
        }
        for input in [
            r#"{"a":}"#,
            r#"{"a":null}"#,
            r#"{"a":""}"#,
            r#"{"a":,"b":true}"#,
        ] {
            assert!(!accepts(&grammar, input), "{input}");
        }
    }
}

#[test]
fn impossible_items_preserve_only_empty_arrays() {
    for max_items in [None, Some(0), Some(1), Some(3)] {
        let mut schema = json!({"type": "array", "items": impossible(), "minItems": 0});
        if let Some(max_items) = max_items {
            schema["maxItems"] = json!(max_items);
        }
        let grammar = compile(&schema);
        for input in ["[]", "[ ]"] {
            assert!(accepts(&grammar, input), "{input}");
        }
        for input in ["[,]", "[ , ]", "[null]", r#"[""]"#] {
            assert!(!accepts(&grammar, input), "{input}");
        }
    }
}

#[test]
fn impossible_tuple_tail_preserves_fixed_prefix() {
    let grammar = compile(&json!({
        "type": "array", "prefixItems": [{"const": true}],
        "items": impossible(), "maxItems": 3
    }));
    assert!(accepts(&grammar, "[true]"));
    for input in ["[true,]", "[true, ,]", "[true,null]"] {
        assert!(!accepts(&grammar, input), "{input}");
    }
}

#[test]
fn alternatives_drop_impossible_containers_and_keep_live_values() {
    for keyword in ["anyOf", "oneOf"] {
        let mut schema = json!({});
        schema[keyword] = json!([
            {"type": "object", "properties": {"a": impossible()}, "required": ["a"]},
            {"type": "array", "items": impossible(), "minItems": 2},
            {"const": true}
        ]);
        let grammar = compile(&schema);
        assert!(accepts(&grammar, "true"));
        for input in [r#"{"a":}"#, "[,]", "[]", "false", ""] {
            assert!(!accepts(&grammar, input), "{keyword}: {input}");
        }
    }
    let overlap = json!({"oneOf": [impossible(), {"const": true}, {"const": true}]});
    assert!(
        compile_json_schema(&overlap)
            .unwrap_err()
            .0
            .contains("oneOf")
    );
}

#[test]
fn impossible_refs_and_aliases_preserve_context() {
    for reference in ["#/$defs/Empty", "#/$defs/Alias"] {
        let defs = json!({"Empty": impossible(), "Alias": {"$ref": "#/$defs/Empty"}});
        let field = json!({"$ref": reference});
        assert_empty(&json!({"$defs": defs, "$ref": reference}));
        assert_empty(&json!({
            "$defs": defs, "type": "object", "properties": {"a": field}, "required": ["a"]
        }));
        let optional = compile(&json!({
            "$defs": defs, "type": "object", "properties": {"a": field}
        }));
        assert!(accepts(&optional, "{}"));
        assert!(!accepts(&optional, r#"{"a":}"#));
        let array = compile(&json!({"$defs": defs, "type": "array", "items": field}));
        assert!(accepts(&array, "[]"));
        assert!(!accepts(&array, "[,]"));
    }
}

#[test]
fn shared_empty_definition_keeps_only_optional_branch() {
    let grammar = compile(&json!({
        "$defs": {"Empty": impossible()},
        "anyOf": [
            {"type": "object", "properties": {"optional": {"$ref": "#/$defs/Empty"}}},
            {"type": "object", "properties": {"required": {"$ref": "#/$defs/Empty"}},
             "required": ["required"]}
        ]
    }));
    assert!(accepts(&grammar, "{}"));
    assert!(!accepts(&grammar, r#"{"optional":}"#));
    assert!(!accepts(&grammar, r#"{"required":}"#));
}

#[test]
fn productive_object_and_array_recursion_survives() {
    let object = compile(&json!({
        "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}}},
        "$ref": "#/$defs/Node"
    }));
    for input in ["{}", r#"{"next":{}}"#, r#"{"next":{"next":{}}}"#] {
        assert!(accepts(&object, input), "{input}");
    }
    assert!(!accepts(&object, r#"{"next":}"#));
    let array = compile(&json!({
        "$defs": {"Node": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
        "$ref": "#/$defs/Node"
    }));
    for input in ["[]", "[[]]", "[[],[[]]]"] {
        assert!(accepts(&array, input), "{input}");
    }
    assert!(!accepts(&array, "[,]"));
}

#[test]
fn productive_recursive_alternative_survives() {
    let grammar = compile(&json!({
        "$defs": {"Node": {"anyOf": [
            {"const": null},
            {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}, "required": ["next"]}
        ]}},
        "$ref": "#/$defs/Node"
    }));
    for input in ["null", r#"{"next":null}"#, r#"{"next":{"next":null}}"#] {
        assert!(accepts(&grammar, input), "{input}");
    }
    for input in ["{}", r#"{"next":}"#] {
        assert!(!accepts(&grammar, input), "{input}");
    }
}

#[test]
fn unproductive_recursive_roots_fail_compilation() {
    for defs in [
        json!({"Node": {"$ref": "#/$defs/Node"}}),
        json!({"Node": {"$ref": "#/$defs/Other"}, "Other": {"$ref": "#/$defs/Node"}}),
        json!({"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}, "required": ["next"]}}),
        json!({"Node": {"type": "array", "items": {"$ref": "#/$defs/Node"}, "minItems": 1}}),
    ] {
        assert_empty(&json!({"$defs": defs, "$ref": "#/$defs/Node"}));
    }
}

#[test]
fn repeated_reference_occurrences_remain_productive() {
    let grammar = compile(&json!({
        "$defs": {"Word": {"enum": ["x"]}}, "type": "array",
        "prefixItems": [{"$ref": "#/$defs/Word"}, {"$ref": "#/$defs/Word"}]
    }));
    assert!(accepts(&grammar, r#"["x","x"]"#));
    assert!(accepts(&grammar, r#"[ "x" , "x" ]"#));
    assert!(!accepts(&grammar, r#"["x",]"#));
    assert!(!accepts(&grammar, r#"["x","y"]"#));
}
