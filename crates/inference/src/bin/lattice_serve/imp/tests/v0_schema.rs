//! Version-0 JSON-schema admission and validation tests for this binary's response_format path.

use super::*;

// ── admit_v0_schema: golden accept per construct ────────────────────

#[test]
fn admit_v0_schema_accepts_closed_object() {
    admit_v0_schema(&serde_json::from_str(V0_OBJECT_SCHEMA).unwrap())
        .expect("closed all-required object must be admitted");
}

#[test]
fn admit_v0_schema_accepts_uniform_array() {
    let schema = serde_json::json!({"type": "array", "items": {"type": "integer"}});
    admit_v0_schema(&schema).expect("uniform array must be admitted");
}

#[test]
fn admit_v0_schema_accepts_every_primitive() {
    for ty in ["string", "number", "integer", "boolean", "null"] {
        let schema = serde_json::json!({"type": ty});
        admit_v0_schema(&schema).unwrap_or_else(|e| panic!("type '{ty}' must be admitted: {e}"));
    }
}

#[test]
fn admit_v0_schema_accepts_nested_object_and_array() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["items"],
        "additionalProperties": false
    });
    admit_v0_schema(&schema).expect("nesting of admitted constructs must be admitted");
}

// ── admit_v0_schema: one rejection per unsupported keyword ─────────

#[test]
fn admit_v0_schema_rejects_enum() {
    let err = admit_v0_schema(&serde_json::json!({"type": "string", "enum": ["a"]})).unwrap_err();
    assert!(err.contains("enum"), "{err}");
}

#[test]
fn admit_v0_schema_rejects_const() {
    let err = admit_v0_schema(&serde_json::json!({"type": "string", "const": "a"})).unwrap_err();
    assert!(err.contains("const"), "{err}");
}

#[test]
fn admit_v0_schema_rejects_pattern() {
    let err = admit_v0_schema(&serde_json::json!({"type": "string", "pattern": "^a"})).unwrap_err();
    assert!(err.contains("pattern"), "{err}");
}

#[test]
fn admit_v0_schema_rejects_length_and_numeric_constraints() {
    for (kw, schema) in [
        (
            "minLength",
            serde_json::json!({"type": "string", "minLength": 1}),
        ),
        (
            "maxLength",
            serde_json::json!({"type": "string", "maxLength": 1}),
        ),
        (
            "minimum",
            serde_json::json!({"type": "number", "minimum": 1}),
        ),
        (
            "maximum",
            serde_json::json!({"type": "number", "maximum": 1}),
        ),
        (
            "multipleOf",
            serde_json::json!({"type": "number", "multipleOf": 2}),
        ),
    ] {
        let err = admit_v0_schema(&schema).unwrap_err();
        assert!(err.contains(kw), "{kw}: {err}");
    }
}

#[test]
fn admit_v0_schema_rejects_ref() {
    let err = admit_v0_schema(&serde_json::json!({"$ref": "#/$defs/foo"})).unwrap_err();
    assert!(err.contains("$ref"), "{err}");
}

#[test]
fn admit_v0_schema_rejects_composition() {
    for kw in ["anyOf", "oneOf", "allOf", "not"] {
        let schema = serde_json::json!({kw: [{"type": "string"}]});
        let err = admit_v0_schema(&schema).unwrap_err();
        assert!(err.contains(kw), "{kw}: {err}");
    }
}

#[test]
fn admit_v0_schema_rejects_type_union() {
    let err = admit_v0_schema(&serde_json::json!({"type": ["string", "null"]})).unwrap_err();
    assert!(err.contains("union"), "{err}");
}

#[test]
fn admit_v0_schema_rejects_missing_additional_properties() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"]
    });
    let err = admit_v0_schema(&schema).unwrap_err();
    assert!(err.contains("additionalProperties"), "{err}");
}

#[test]
fn admit_v0_schema_rejects_open_additional_properties() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": true
    });
    let err = admit_v0_schema(&schema).unwrap_err();
    assert!(err.contains("additionalProperties"), "{err}");
}

#[test]
fn admit_v0_schema_rejects_missing_required_entry() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": false
    });
    let err = admit_v0_schema(&schema).unwrap_err();
    assert!(err.contains("required"), "{err}");
}

#[test]
fn admit_v0_schema_rejects_prefix_items() {
    let schema = serde_json::json!({
        "type": "array",
        "items": {"type": "string"},
        "prefixItems": [{"type": "string"}]
    });
    let err = admit_v0_schema(&schema).unwrap_err();
    assert!(err.contains("prefixItems"), "{err}");
}

#[test]
fn admit_v0_schema_rejects_format() {
    let err =
        admit_v0_schema(&serde_json::json!({"type": "string", "format": "date-time"})).unwrap_err();
    assert!(err.contains("format"), "{err}");
}

#[test]
fn admit_v0_schema_rejects_unknown_keyword() {
    let err = admit_v0_schema(&serde_json::json!({"type": "string", "wat": true})).unwrap_err();
    assert!(err.contains("wat"), "{err}");
}

/// Mutation-sensitivity check: unlike a plain `.is_err()` check, this asserts the REJECTION
/// SOURCE is specifically `V0_REJECTED_KEYWORDS` (the denylist
/// layer walked before the per-type match in `admit_v0_schema_at`),
/// not the primitive per-type allowlist layer (`ALLOWED` inside the
/// `"string"` arm) that would also reject an unrecognized `enum`
/// key on its own. If `enum` is removed from
/// `V0_REJECTED_KEYWORDS`, this schema still gets rejected -- by the
/// allowlist layer instead -- but with a DIFFERENT message, so this
/// test (unlike a bare `.is_err()` check) fails if the denylist
/// layer's rejection of `enum` regresses.
#[test]
fn admit_v0_schema_mutation_sensitive_to_enum_admission() {
    let schema = serde_json::json!({"type": "string", "enum": ["a", "b"]});
    let err = admit_v0_schema(&schema)
        .expect_err("v0 must reject enum (Q2: deferred from the first patch)");
    assert!(
        err.contains("is not supported by the v0 structured-output schema subset"),
        "expected the denylist layer's own rejection wording, got: {err}"
    );
}

/// Independent companion to the denylist test above: a keyword that `V0_REJECTED_KEYWORDS` never
/// mentions at all (so the denylist layer can never be the one that
/// rejects it) must still be rejected by the primitive per-type
/// `ALLOWED`-keyword layer. Mutation-sensitive to a DIFFERENT
/// mutation than the test above: loosening `ALLOWED` for a
/// primitive type (e.g. adding `"default"`) makes this schema admit
/// successfully, failing this test -- independently of whatever
/// `V0_REJECTED_KEYWORDS` contains.
#[test]
fn admit_v0_schema_mutation_sensitive_to_primitive_allowlist() {
    let schema = serde_json::json!({"type": "boolean", "default": true});
    let err =
        admit_v0_schema(&schema).expect_err("v0 must reject an unrecognized primitive keyword");
    assert!(
        err.contains("is not supported for type 'boolean'"),
        "expected the primitive-allowlist layer's own rejection wording, got: {err}"
    );
}

// ── v0_validate_json: independent evaluator ─────────────────────────

#[test]
fn v0_validate_json_accepts_conforming_object() {
    let schema: Value = serde_json::from_str(V0_OBJECT_SCHEMA).unwrap();
    assert!(v0_validate_json(r#"{"name": "ok"}"#, &schema));
}

#[test]
fn v0_validate_json_rejects_extra_field() {
    let schema: Value = serde_json::from_str(V0_OBJECT_SCHEMA).unwrap();
    assert!(!v0_validate_json(r#"{"name": "ok", "extra": 1}"#, &schema));
}

#[test]
fn v0_validate_json_rejects_missing_field() {
    let schema: Value = serde_json::from_str(V0_OBJECT_SCHEMA).unwrap();
    assert!(!v0_validate_json("{}", &schema));
}

#[test]
fn v0_validate_json_rejects_wrong_type() {
    let schema: Value = serde_json::from_str(V0_OBJECT_SCHEMA).unwrap();
    assert!(!v0_validate_json(r#"{"name": 1}"#, &schema));
}

#[test]
fn v0_validate_json_rejects_malformed_json() {
    let schema: Value = serde_json::from_str(V0_OBJECT_SCHEMA).unwrap();
    assert!(!v0_validate_json(r#"{"name": "ok""#, &schema));
}

#[test]
fn v0_validate_json_rejects_trailing_garbage() {
    let schema: Value = serde_json::from_str(V0_OBJECT_SCHEMA).unwrap();
    assert!(!v0_validate_json(r#"{"name": "ok"} garbage"#, &schema));
}

/// A grammar-conforming integer
/// outside `u64`/`i64` range must validate, not produce a false
/// `validation_failed`.
#[test]
fn v0_validate_json_accepts_integer_outside_u64_range() {
    let schema = serde_json::json!({"type": "integer"});
    assert!(v0_validate_json("18446744073709551616", &schema));
    assert!(v0_validate_json("-18446744073709551616", &schema));
}

#[test]
fn v0_validate_json_accepts_integer_in_u64_range() {
    let schema = serde_json::json!({"type": "integer"});
    assert!(v0_validate_json("42", &schema));
    assert!(v0_validate_json("-1", &schema));
}

/// JSON Schema 2020-12 §6.1.1: a zero fractional part is still an
/// integer.
#[test]
fn v0_validate_json_accepts_integer_with_zero_fraction() {
    let schema = serde_json::json!({"type": "integer"});
    assert!(v0_validate_json("1.0", &schema));
}

#[test]
fn v0_validate_json_rejects_integer_with_nonzero_fraction() {
    let schema = serde_json::json!({"type": "integer"});
    assert!(!v0_validate_json("1.5", &schema));
}

#[test]
fn v0_validate_json_rejects_integer_lexeme_for_string_schema() {
    let schema = serde_json::json!({"type": "string"});
    assert!(!v0_validate_json("18446744073709551616", &schema));
}

/// End-to-end shape: a large
/// integer nested inside an admitted object schema.
#[test]
fn v0_validate_json_accepts_large_integer_nested_in_object() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
        "additionalProperties": false,
    });
    assert!(v0_validate_json(r#"{"n": 18446744073709551616}"#, &schema));
}

/// A valid UTF-16 surrogate pair
/// (the G-clef character, U+1D11E) spelled as its two `\uXXXX`
/// escape halves must validate under a string schema -- it is a
/// grammar-conforming strict completion, and rejecting it turned
/// into a false `validation_failed` 500 before this fix.
#[test]
fn v0_validate_json_accepts_surrogate_pair_under_string_schema() {
    let schema = serde_json::json!({"type": "string"});
    assert!(v0_validate_json(r#""\uD834\uDD1E""#, &schema));
}

/// Same surrogate pair, but as an object property value -- proves
/// the fix applies through `v0_validate_object`'s recursive
/// `v0_validate_at` call, not just the top-level string schema.
#[test]
fn v0_validate_json_accepts_surrogate_pair_as_object_property_value() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"s": {"type": "string"}},
        "required": ["s"],
        "additionalProperties": false,
    });
    assert!(v0_validate_json(r#"{"s": "\uD834\uDD1E"}"#, &schema));
}

/// Grammar-parity: the compiled grammar admits any four-hex
/// `\uXXXX` escape independently of pairing (json_schema.rs
/// `escape_id` alternatives at ~1564), and its own
/// `string_accepts_legal_escapes` test (json_schema.rs ~2762)
/// explicitly accepts lone surrogate halves `\uD800` / `\uDC00`.
/// The validator must not be narrower than the grammar it guards,
/// so a lone (unpaired) surrogate escape must also validate here.
#[test]
fn v0_validate_json_accepts_lone_surrogate_escape_grammar_parity() {
    let schema = serde_json::json!({"type": "string"});
    assert!(v0_validate_json(r#""\uD834""#, &schema));
    assert!(v0_validate_json(r#""\uDD1E""#, &schema));
}

/// RFC 8259 §7
/// requires U+0000-U+001F to be escaped, and the compiled grammar
/// rejects them raw too (json_schema.rs
/// `string_rejects_raw_control_byte`). A literal BEL (0x07) byte
/// embedded unescaped in a string must fail validation.
#[test]
fn v0_validate_json_rejects_raw_control_char() {
    let schema = serde_json::json!({"type": "string"});
    let content = format!("\"{}\"", '\u{7}');
    assert!(!v0_validate_json(&content, &schema));
}
