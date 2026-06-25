//! Regression harness for GitHub issue #310 (grammar-constrained decoding, ADR-046).
//!
//! Status per finding:
//!   f1 (object optional-member separators)     — DEFERRED, marked #[ignore]
//!   f2 (array cardinality minItems/maxItems)   — FIXED in this session
//!   f3 (prefixItems tuple arrays)              — FIXED in this session
//!   f4 (string enum rule-name collision)       — previously fixed upstream
//!   f5 (single-stack PDA common-prefix)        — Ocean-gated architectural, marked #[ignore]
//!   f6 (leading-zero integer rejection)        — previously fixed upstream
//!   f7 (enum/const string escaping)            — FIXED in this session

use lattice_inference::grammar::json_schema::compile_json_schema;
use lattice_inference::grammar::pda::{
    CompiledGrammar, GrammarBuilder, GrammarState, StepResult, Symbol, advance_byte,
};

fn full_accept(g: &CompiledGrammar, s: &[u8]) -> bool {
    let mut st = GrammarState::initial();
    for &b in s {
        if advance_byte(&mut st, g, b) == StepResult::Rejected {
            return false;
        }
    }
    st.is_complete()
}

#[test]
#[ignore = "issue #310 finding #5 (single-stack PDA cannot backtrack common prefixes) — Ocean-gated architectural redesign"]
fn f5_common_prefix_raw() {
    let mut b = GrammarBuilder::new();
    b.add_rule(
        "root",
        vec![
            vec![Symbol::Terminal(b'a'), Symbol::Terminal(b'b')],
            vec![Symbol::Terminal(b'a'), Symbol::Terminal(b'c')],
        ],
    );
    let g = b.build();
    assert!(
        full_accept(&g, b"ac"),
        "common-prefix: 'ac' must be accepted"
    );
}

#[test]
#[ignore = "issue #310 finding #5 (single-stack PDA cannot backtrack common prefixes) — Ocean-gated architectural redesign"]
fn f5_common_prefix_enum() {
    let g = compile_json_schema(&serde_json::json!({"type":"string","enum":["apple","apricot"]}))
        .unwrap();
    assert!(
        full_accept(&g, br#""apricot""#),
        "'apricot' must be accepted"
    );
}

#[test]
#[ignore = "issue #310 finding #1 (object optional-member separators) — deferred follow-up"]
fn f1_trailing_comma() {
    let g = compile_json_schema(&serde_json::json!({"type":"object",
        "properties":{"a":{"type":"string"},"b":{"type":"string"}},"required":["a"]}))
    .unwrap();
    assert!(
        full_accept(&g, br#"{"a":"x"}"#),
        "no-comma object must be accepted"
    );
    assert!(
        !full_accept(&g, br#"{"a":"x",}"#),
        "trailing comma must be rejected"
    );
}

#[test]
fn f2_max_items() {
    let g = compile_json_schema(
        &serde_json::json!({"type":"array","items":{"type":"integer"},"maxItems":1}),
    )
    .unwrap();
    assert!(
        !full_accept(&g, b"[1,2]"),
        "[1,2] must be rejected (maxItems:1)"
    );
}

#[test]
fn f2_min_items_pins() {
    let g = compile_json_schema(
        &serde_json::json!({"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":3}),
    )
    .unwrap();
    assert!(
        full_accept(&g, b"[1,2]"),
        "[1,2] must be accepted (1<=2<=3)"
    );
}

#[test]
fn f3_prefix_items() {
    let g = compile_json_schema(
        &serde_json::json!({"type":"array","prefixItems":[{"const":1},{"const":2}]}),
    )
    .unwrap();
    assert!(
        !full_accept(&g, b"[2,1]"),
        "[2,1] must be rejected (prefixItems [1,2])"
    );
}

#[test]
fn f6_leading_zero() {
    let g = compile_json_schema(&serde_json::json!({"type":"number"})).unwrap();
    assert!(!full_accept(&g, b"01"), "01 must be rejected");
}

#[test]
fn f7_enum_escaping() {
    let g = compile_json_schema(&serde_json::json!({"type":"string","enum":["a\"b"]})).unwrap();
    assert!(
        full_accept(&g, b"\"a\\\"b\""),
        r#"valid "a\"b" must be accepted"#
    );
}

#[test]
fn f4_enum_collision() {
    let g = compile_json_schema(&serde_json::json!({"type":"object",
        "properties":{"x":{"type":"string","enum":["ab","c_d"]},"y":{"type":"string","enum":["ab_c","d"]}},
        "required":["x","y"]}))
    .unwrap();
    assert!(
        full_accept(&g, br#"{"x":"ab","y":"d"}"#),
        "y must accept its own value 'd'"
    );
}

#[test]
fn f2_max_lt_min_rejected() {
    // issue #310 finding #2: maxItems < minItems is an unsatisfiable schema and
    // must surface as a compile error rather than silently producing a grammar.
    let r = compile_json_schema(
        &serde_json::json!({"type":"array","items":{"type":"integer"},"minItems":3,"maxItems":1}),
    );
    assert!(
        r.is_err(),
        "maxItems(1) < minItems(3) must be a compile error"
    );
}

#[test]
fn f2_empty_only() {
    // issue #310 finding #2: maxItems:0 admits only the empty array.
    let g = compile_json_schema(
        &serde_json::json!({"type":"array","items":{"type":"integer"},"maxItems":0}),
    )
    .unwrap();
    assert!(full_accept(&g, b"[]"), "[] must be accepted (maxItems:0)");
    assert!(
        !full_accept(&g, b"[1]"),
        "[1] must be rejected (maxItems:0)"
    );
}

#[test]
fn f2_multi_array_no_collision() {
    // issue #310 finding #2: two sibling array schemas with distinct item types
    // must each get a unique helper-rule namespace. The pre-fix shared "arr_item"
    // name made the second array reuse the first's item rule, so an integer-typed
    // `a` would wrongly constrain a boolean-typed `b`.
    let g = compile_json_schema(&serde_json::json!({"type":"object",
        "properties":{"a":{"type":"array","items":{"type":"integer"}},
                      "b":{"type":"array","items":{"type":"boolean"}}},
        "required":["a","b"]}))
    .unwrap();
    assert!(
        full_accept(&g, br#"{"a":[1],"b":[true]}"#),
        "b must accept booleans (its own item schema), not integers"
    );
    assert!(
        !full_accept(&g, br#"{"a":[true],"b":[true]}"#),
        "a must reject booleans (its own item schema is integer)"
    );
}

#[test]
fn f2_no_items_cardinality() {
    // codex #1: cardinality must be enforced even without an `items` schema.
    let g0 = compile_json_schema(&serde_json::json!({"type":"array","maxItems":0})).unwrap();
    assert!(
        full_accept(&g0, b"[]"),
        "[] accepted (maxItems:0, no items)"
    );
    assert!(
        !full_accept(&g0, b"[1]"),
        "[1] rejected (maxItems:0, no items)"
    );
    let g2 = compile_json_schema(&serde_json::json!({"type":"array","minItems":2,"maxItems":2}))
        .unwrap();
    assert!(
        full_accept(&g2, b"[1,2]"),
        "[1,2] accepted (2<=2<=2, no items)"
    );
    assert!(!full_accept(&g2, b"[1]"), "[1] rejected (minItems:2)");
    assert!(
        !full_accept(&g2, b"[1,2,3]"),
        "[1,2,3] rejected (maxItems:2)"
    );
}

#[test]
fn f3_prefix_items_maxitems() {
    // critic F1 / codex #2: prefixItems + items must honor maxItems.
    let g = compile_json_schema(&serde_json::json!(
        {"type":"array","prefixItems":[{"type":"integer"}],"items":{"type":"integer"},"maxItems":2}
    ))
    .unwrap();
    assert!(full_accept(&g, b"[1]"), "[1] accepted (1<=2)");
    assert!(full_accept(&g, b"[1,2]"), "[1,2] accepted (2<=2)");
    assert!(
        !full_accept(&g, b"[1,2,3]"),
        "[1,2,3] rejected (maxItems:2)"
    );
}

#[test]
fn f3_prefix_maxitems_lt_prefixlen_error() {
    // maxItems below the fixed prefix length is unsatisfiable -> compile error.
    let r = compile_json_schema(&serde_json::json!(
        {"type":"array","prefixItems":[{"const":1},{"const":2}],"maxItems":1}
    ));
    assert!(
        r.is_err(),
        "maxItems(1) < prefixItems len(2) must be a compile error"
    );
}

#[test]
fn f3_empty_prefix_items_applies_items() {
    // codex #3: empty prefixItems == no prefixItems; `items` must apply.
    let g = compile_json_schema(&serde_json::json!(
        {"type":"array","prefixItems":[],"items":{"type":"integer"}}
    ))
    .unwrap();
    assert!(
        full_accept(&g, b"[1]"),
        "[1] accepted (empty prefixItems, items=integer)"
    );
    assert!(full_accept(&g, b"[]"), "[] accepted");
}
