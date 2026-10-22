use super::*;
use crate::grammar::pda::{GrammarState, SimResult, simulate_token};

#[test]
fn ref_target_annotations_exclude_structural_keys() {
    // `resolve_ref_chain_target` follows `$ref` to a terminal target, and
    // the other keys below do not emit instance values today. This guards
    // a latent proof-widening risk if either behavior changes.
    for key in ["$ref", "$defs", "definitions", "$id", "$schema"] {
        assert!(
            !REF_TARGET_ANNOTATION_KEYS.contains(&key),
            "structural key `{key}` must not be admitted as a target annotation"
        );
    }
}

fn compile_ok(schema_json: &str) -> CompiledGrammar {
    let v: Value = serde_json::from_str(schema_json).unwrap();
    compile(&v).unwrap()
}

fn accepts(grammar: &CompiledGrammar, input: &[u8]) -> bool {
    let state = GrammarState::initial();
    let (result, final_state) = simulate_token(&state, grammar, input);
    result == SimResult::Accept && final_state.is_complete()
}

fn rejects(grammar: &CompiledGrammar, input: &[u8]) -> bool {
    // A grammar "rejects" a string when it cannot accept it as a complete
    // value.  This includes both early byte rejections (SimResult::Reject)
    // and cases where all bytes are consumed but the state is not complete.
    !accepts(grammar, input)
}

#[test]
fn compile_null_schema() {
    let g = compile_ok(r#"{"type":"null"}"#);
    assert!(accepts(&g, b"null"));
    assert!(rejects(&g, b"true"));
}

#[test]
fn compile_boolean_schema() {
    let g = compile_ok(r#"{"type":"boolean"}"#);
    assert!(accepts(&g, b"true"));
    assert!(accepts(&g, b"false"));
    assert!(rejects(&g, b"null"));
}

#[test]
fn compile_integer_schema() {
    let g = compile_ok(r#"{"type":"integer"}"#);
    assert!(accepts(&g, b"42"));
    assert!(accepts(&g, b"-7"));
    assert!(accepts(&g, b"0"));
}

#[test]
fn compile_string_schema() {
    let g = compile_ok(r#"{"type":"string"}"#);
    assert!(accepts(&g, b"\"hello\""));
    assert!(accepts(&g, b"\"\""));
}

/// Regression for issue #931: the built-in `json_string` rule used to
/// accept any raw byte (including the C0 control range) unescaped, and
/// any `\` followed by any byte as a valid escape. Both gaps let the
/// grammar accept output a strict RFC 8259 parser rejects. Before this
/// fix the escape alternative was `'\\' + Symbol::AnyByte` and the
/// plain-byte alternative ranged over all of `0u8..=255`, so the raw
/// 0x07 case, the `\q` case, and the truncated `\u12"` case were all
/// accepted.
#[test]
fn string_rejects_raw_control_byte() {
    let g = compile_ok(r#"{"type":"string"}"#);
    // Raw BEL (0x07) unescaped inside the quotes must be rejected.
    assert!(rejects(&g, b"\"\x07\""));
}

#[test]
fn string_rejects_invalid_escape() {
    let g = compile_ok(r#"{"type":"string"}"#);
    // `\q` is not in the RFC 8259 escape set.
    assert!(rejects(&g, b"\"\\q\""));
}

#[test]
fn string_rejects_malformed_unicode_escape() {
    let g = compile_ok(r#"{"type":"string"}"#);
    // `\u` requires exactly four hex digits. Two- and three-digit runs
    // (closed by the quote) and a non-hex digit anywhere in the run must
    // all reject; a mutation that accepted three digits or any byte would
    // slip through `\u123` or `\u12g4` respectively.
    assert!(rejects(&g, b"\"\\u12\""));
    assert!(rejects(&g, b"\"\\u123\""));
    assert!(rejects(&g, b"\"\\u12g4\""));
}

#[test]
fn string_accepts_multibyte_utf8() {
    let g = compile_ok(r#"{"type":"string"}"#);
    // "好" (U+597D, 3-byte), "é" (U+00E9, 2-byte), "😀" (U+1F600, 4-byte).
    assert!(accepts(&g, "\"好\"".as_bytes()));
    assert!(accepts(&g, "\"é\"".as_bytes()));
    assert!(accepts(&g, "\"😀\"".as_bytes()));
}

#[test]
fn string_rejects_lone_continuation_byte() {
    let g = compile_ok(r#"{"type":"string"}"#);
    // 0x80 alone is a UTF-8 continuation byte with no lead byte.
    assert!(rejects(&g, b"\"\x80\""));
}

#[test]
fn string_rejects_overlong_encoding() {
    let g = compile_ok(r#"{"type":"string"}"#);
    // 0xC0 0xAF is the canonical overlong encoding of '/' (U+002F);
    // 0xC0/0xC1 are excluded from the valid 2-byte lead range entirely.
    assert!(rejects(&g, b"\"\xC0\xAF\""));
}

/// Regression for issue #931: exercises the E0/ED/F0/F4 restricted
/// second-byte sub-rules (`e0_second_id`/`ed_second_id`/`f0_second_id`/
/// `f4_second_id`) end-to-end through the same PDA path constrained
/// decoding uses (`simulate_token` via `accepts`/`rejects`), plus the
/// control-byte and escape-body boundaries in the same string grammar.
#[test]
fn string_utf8_boundary_and_control_escape_table() {
    let g = compile_ok(r#"{"type":"string"}"#);

    // Minimal/maximal well-formed forms for each restricted lead byte.
    let accept_cases: &[(&str, &[u8])] = &[
        ("2-byte min", &[0xC2, 0x80]),
        ("2-byte max", &[0xDF, 0xBF]),
        ("3-byte E0 min", &[0xE0, 0xA0, 0x80]),
        ("3-byte ED max", &[0xED, 0x9F, 0xBF]),
        ("4-byte F0 min", &[0xF0, 0x90, 0x80, 0x80]),
        ("4-byte F4 max", &[0xF4, 0x8F, 0xBF, 0xBF]),
    ];
    for (name, bytes) in accept_cases {
        let mut input = vec![b'"'];
        input.extend_from_slice(bytes);
        input.push(b'"');
        assert!(
            accepts(&g, &input),
            "expected ACCEPT for {name}: {bytes:02X?}"
        );
    }

    // Boundary violations: overlong encodings, a UTF-16 surrogate, and a
    // code point above the U+10FFFF Unicode maximum.
    let reject_cases: &[(&str, &[u8])] = &[
        ("E0 overlong", &[0xE0, 0x80, 0x80]),
        ("F0 overlong", &[0xF0, 0x80, 0x80, 0x80]),
        ("ED surrogate", &[0xED, 0xA0, 0x80]),
        ("F4 above U+10FFFF", &[0xF4, 0x90, 0x80, 0x80]),
        ("C0 overlong 2-byte", &[0xC0, 0x80]),
        ("lone continuation byte", &[0x80]),
    ];
    for (name, bytes) in reject_cases {
        let mut input = vec![b'"'];
        input.extend_from_slice(bytes);
        input.push(b'"');
        assert!(
            rejects(&g, &input),
            "expected REJECT for {name}: {bytes:02X?}"
        );
    }

    // Raw control bytes 0x00-0x1F must be escaped; 0x20 (space) is the
    // first legal unescaped byte.
    for byte in 0x00u8..=0x1F {
        assert!(
            rejects(&g, &[b'"', byte, b'"']),
            "expected REJECT for raw control byte {byte:#04x}"
        );
    }
    assert!(
        accepts(&g, &[b'"', 0x20, b'"']),
        "expected ACCEPT for raw space (0x20)"
    );

    // Escaped controls (valid \u escapes) are accepted.
    assert!(accepts(&g, b"\"\\u0000\""));
    assert!(accepts(&g, b"\"\\u001F\""));

    // Every one-byte escape body outside the RFC 8259 set is rejected.
    let legal_escape_bodies: &[u8] = b"\"\\/bfnrtu";
    for byte in 0x00u8..=0xFF {
        if legal_escape_bodies.contains(&byte) {
            continue;
        }
        assert!(
            rejects(&g, &[b'"', b'\\', byte, b'"']),
            "expected REJECT for illegal escape body {byte:#04x}"
        );
    }
}

#[test]
fn string_accepts_legal_escapes() {
    let g = compile_ok(r#"{"type":"string"}"#);
    assert!(accepts(&g, b"\"\\\"\""));
    assert!(accepts(&g, b"\"\\\\\""));
    assert!(accepts(&g, b"\"\\/\""));
    assert!(accepts(&g, b"\"\\b\""));
    assert!(accepts(&g, b"\"\\f\""));
    assert!(accepts(&g, b"\"\\n\""));
    assert!(accepts(&g, b"\"\\r\""));
    assert!(accepts(&g, b"\"\\t\""));
    assert!(accepts(&g, b"\"\\u0041\""));
    // Surrogate-pair halves are syntactically valid `\uXXXX` per the
    // RFC 8259 ABNF, even though unpaired they are not interoperable text.
    assert!(accepts(&g, b"\"\\uD800\""));
    assert!(accepts(&g, b"\"\\uDC00\""));
    // A full high+low surrogate pair (here: U+1D11E, the G-clef
    // character) is just two of the above alternatives concatenated,
    // so it must accept too -- the compiler/validator inclusion
    // invariant pinned from the grammar side (see the `v0_parse_string`
    // fix and its test-side pin in lattice_serve.rs).
    assert!(accepts(&g, b"\"\\uD834\\uDD1E\""));
}

#[test]
fn compile_string_enum() {
    let g = compile_ok(r#"{"type":"string","enum":["foo","bar"]}"#);
    assert!(accepts(&g, b"\"foo\""));
    assert!(accepts(&g, b"\"bar\""));
    assert!(rejects(&g, b"\"baz\""));
}

#[test]
fn compile_const_value() {
    let g = compile_ok(r#"{"const":"hello"}"#);
    assert!(accepts(&g, b"\"hello\""));
    assert!(rejects(&g, b"\"world\""));
}

#[test]
fn compile_const_integer() {
    let g = compile_ok(r#"{"const":42}"#);
    assert!(accepts(&g, b"42"));
    assert!(rejects(&g, b"43"));
}

#[test]
fn compile_any_of() {
    let g = compile_ok(r#"{"anyOf":[{"type":"boolean"},{"type":"null"}]}"#);
    assert!(accepts(&g, b"true"));
    assert!(accepts(&g, b"false"));
    assert!(accepts(&g, b"null"));
}

#[test]
fn compile_simple_object() {
    let g = compile_ok(
        r#"{
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            }"#,
    );
    assert!(accepts(&g, b"{\"name\":\"Alice\"}"));
}

#[test]
fn compile_empty_object() {
    let g = compile_ok(r#"{"type":"object"}"#);
    assert!(accepts(&g, b"{}"));
}

#[test]
fn empty_object_accepts_interior_whitespace() {
    // A no-properties object schema must accept `{ }` (interior ws), not
    // only the byte-exact `{}`. The None arm previously emitted `{` `}`
    // with no ws rule between the braces.
    let g = compile_ok(r#"{"type":"object"}"#);
    assert!(accepts(&g, b"{}"));
    assert!(
        accepts(&g, b"{ }"),
        "empty no-properties object should accept interior whitespace"
    );
}

#[test]
fn distinct_objects_sharing_a_key_do_not_alias() {
    // Two distinct object schemas (`a` and `b`) each declare a property
    // named `x` with a DIFFERENT value type. Without per-object rule-name
    // namespacing they would share a single `prop_val_x` rule and the first
    // compiled type (string) would silently win for both, rejecting `b.x`
    // as an integer. `a` and `b` are distinct keys, so there is no
    // shared-prefix ambiguity here — the PDA handles it cleanly.
    let g = compile_ok(
        r#"{
              "type":"object",
              "properties":{
                "a":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]},
                "b":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}
              },
              "required":["a","b"]
            }"#,
    );
    assert!(
        accepts(&g, b"{\"a\":{\"x\":\"s\"},\"b\":{\"x\":42}}"),
        "b.x must accept an integer; the string rule for a.x must not alias it"
    );
    assert!(
        rejects(&g, b"{\"a\":{\"x\":42},\"b\":{\"x\":42}}"),
        "a.x must still reject an integer (string-typed)"
    );
}

#[test]
fn compile_array_any() {
    let g = compile_ok(r#"{"type":"array"}"#);
    assert!(accepts(&g, b"[]"));
    // Non-empty untyped arrays must accept too: the any-array tail rule
    // `tail ::= ws ',' ws value tail | ε` reaches the `| ε` alternative
    // after each element so the closing `]` can match.  Regression guard
    // for a mid-alternative-backtrack over-rejection (refs #353).
    assert!(accepts(&g, b"[5]"));
    assert!(accepts(&g, b"[1,2]"));
    assert!(accepts(&g, br#"[{}]"#));
    assert!(accepts(&g, b"[true]"));
}

#[test]
fn any_array_rejects_trailing_comma() {
    // An untyped array (`body ::= any tail | ε`, `tail ::= ws ',' ws any
    // tail | ε`) must reject a trailing comma before `]`.  Previously the
    // PDA's no-rewind parent-backtrack switched the comma-consuming `tail`
    // frame to its `| ε` sibling AFTER the `,` byte was consumed, silently
    // admitting invalid JSON.  The per-frame byte-consumption guard refuses
    // that switch once a byte is committed.  Valid forms must still accept.
    let g = compile_ok(r#"{"type":"array"}"#);
    assert!(accepts(&g, b"[]"));
    assert!(accepts(&g, b"[5]"));
    assert!(accepts(&g, b"[1,2]"));
    assert!(rejects(&g, b"[1,]"));
    assert!(rejects(&g, b"[1,2,]"));
    assert!(rejects(&g, b"[,]"));
}

#[test]
fn typed_array_rejects_trailing_comma() {
    // Same trailing-comma over-acceptance via the bounded-tail rule used
    // for typed arrays (`build_bounded_tail`).  The byte-consumption guard
    // closes both the untyped and typed array paths at once.
    let g = compile_ok(r#"{"type":"array","items":{"type":"integer"}}"#);
    assert!(accepts(&g, b"[]"));
    assert!(accepts(&g, b"[5]"));
    assert!(accepts(&g, b"[1,2]"));
    assert!(rejects(&g, b"[1,]"));
    assert!(rejects(&g, b"[1,2,]"));
}

// -----------------------------------------------------------------------
// Trie-compiled shared-prefix literal alternatives (issue #XXX)
//
// The single-stack no-rewind PDA can only switch alternatives at sym_pos==0.
// A flat encoding of ["foo","food"] commits to "foo" and cannot backtrack
// when it then sees "d" — "food" is over-rejected. The fix compiles string
// enum / anyOf-const alternatives into a trie so that every branch diverges
// on its first byte (the closing `"` is pulled into the trie leaves to make
// sequences prefix-free).
// -----------------------------------------------------------------------

/// string enum ["foo","food"]: both accepted; "foe"/"fooo"/"fo" rejected.
#[test]
fn string_enum_shared_prefix_foo_food() {
    let g = compile_ok(r#"{"type":"string","enum":["foo","food"]}"#);
    assert!(accepts(&g, b"\"foo\""), "\"foo\" must be accepted");
    assert!(accepts(&g, b"\"food\""), "\"food\" must be accepted");
    assert!(rejects(&g, b"\"foe\""), "\"foe\" must be rejected");
    assert!(rejects(&g, b"\"fooo\""), "\"fooo\" must be rejected");
    assert!(rejects(&g, b"\"fo\""), "\"fo\" must be rejected");
}

/// Bare enum WITHOUT `"type":"string"` must also route through the trie via
/// the all-strings predicate in `compile_schema_inner`'s enum branch
/// (`type_is_string || all_strings`), NOT the flat `compile_enum` path —
/// otherwise shared-prefix members are over-rejected again. This is the
/// exact schema the prior known-limitation test used; it locks the routing
/// so a future narrowing to `type_is_string` only cannot silently regress.
#[test]
fn string_enum_bare_no_type_shared_prefix() {
    let g = compile_ok(r#"{"enum":["foo","food"]}"#);
    assert!(accepts(&g, b"\"foo\""), "\"foo\" must be accepted");
    assert!(accepts(&g, b"\"food\""), "\"food\" must be accepted");
    assert!(rejects(&g, b"\"foe\""), "\"foe\" must be rejected");
}

/// anyOf[const "abc", const "abd"]: both accepted; "abe" rejected.
#[test]
fn anyof_const_shared_prefix_abc_abd() {
    let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"const":"abd"}]}"#);
    assert!(accepts(&g, b"\"abc\""), "\"abc\" must be accepted");
    assert!(accepts(&g, b"\"abd\""), "\"abd\" must be accepted");
    assert!(rejects(&g, b"\"abe\""), "\"abe\" must be rejected");
    assert!(rejects(&g, b"\"ab\""), "\"ab\" must be rejected");
    assert!(rejects(&g, b"\"abcd\""), "\"abcd\" must be rejected");
}

/// Regression: a broad `{"type":"string"}` branch alongside shared-prefix
/// string consts must stay reachable. The broad branch compiles to
/// `[NonTerminal(json_string)]`, whose FIRST set contains `"`; #471's
/// unconditional const-trie hoist made the no-rewind PDA consume the opening
/// quote and then fail to fall through, wrongly rejecting an arbitrary string
/// such as `"zzz"`. A present broad string collapses the whole string class
/// to `json_string`, which subsumes the consts. Mutation guard: reverting to
/// the unconditional hoist reintroduces the over-rejection and fails here.
#[test]
fn anyof_broad_string_with_shared_prefix_consts() {
    let g = compile_ok(r#"{"anyOf":[{"type":"string"},{"const":"abc"},{"const":"abd"}]}"#);
    assert!(
        accepts(&g, b"\"zzz\""),
        "broad string branch must accept an arbitrary string"
    );
    assert!(accepts(&g, b"\"abc\""), "\"abc\" must be accepted");
    assert!(accepts(&g, b"\"abd\""), "\"abd\" must be accepted");
    assert!(rejects(&g, b"123"), "a non-string must still be rejected");
}

/// A non-string sibling that cannot begin with `"` (here `integer`) does not
/// block the shared-prefix trie: `"abc"`/`"abd"` are disambiguated by the
/// combined trie and the integer branch stays reachable (a non-`"` input
/// diverges from the trie at sym_pos == 0). Mutation guard: emitting the
/// branches flat instead of one trie over-rejects `"abd"`.
#[test]
fn anyof_shared_prefix_consts_with_integer() {
    let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"const":"abd"},{"type":"integer"}]}"#);
    assert!(accepts(&g, b"\"abc\""), "\"abc\" must be accepted");
    assert!(
        accepts(&g, b"\"abd\""),
        "\"abd\" must be accepted via the trie"
    );
    assert!(
        accepts(&g, b"123"),
        "the integer branch must remain reachable"
    );
    assert!(
        rejects(&g, b"\"abe\""),
        "a non-member string must be rejected"
    );
}

/// Broad string LAST (issue #310 / PR #472): the broad `{"type":"string"}`
/// is listed AFTER the shared-prefix consts, so schema order cannot be
/// relied on to keep it reachable. Classifying the whole string class up
/// front and collapsing it to `json_string` makes position irrelevant — an
/// arbitrary string that diverges from the consts (`"abe"`) is still
/// accepted. Mutation guard: an order-sensitive fix (hoisting the const trie
/// whenever the broad branch is not first) over-rejects `"abe"` here.
#[test]
fn anyof_broad_string_after_consts() {
    let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"const":"abd"},{"type":"string"}]}"#);
    assert!(
        accepts(&g, b"\"abe\""),
        "broad string (listed last) must accept a string that diverges from the consts"
    );
    assert!(accepts(&g, b"\"zzz\""), "arbitrary string must be accepted");
    assert!(accepts(&g, b"\"abc\""), "\"abc\" must be accepted");
    assert!(accepts(&g, b"\"abd\""), "\"abd\" must be accepted");
    assert!(rejects(&g, b"123"), "a non-string must still be rejected");
}

/// A narrower string sibling — a string `enum` rather than a broad string —
/// alongside shared-prefix consts (issue #310 / PR #472). With no broad
/// branch present, every string literal (the two consts and the two enum
/// members) is unioned into ONE trie, so all four diverge inside the quoted
/// region and stay reachable. Mutation guard: keeping the branches flat (or
/// hoisting only the consts) over-rejects `"abd"`; widening the class to
/// `json_string` would over-accept `"abg"`.
#[test]
fn anyof_narrower_string_enum_sibling() {
    let g = compile_ok(
        r#"{"anyOf":[{"const":"abc"},{"const":"abd"},{"type":"string","enum":["abe","abf"]}]}"#,
    );
    assert!(accepts(&g, b"\"abc\""), "const \"abc\" must be accepted");
    assert!(
        accepts(&g, b"\"abd\""),
        "const \"abd\" must be accepted via the combined trie"
    );
    assert!(
        accepts(&g, b"\"abe\""),
        "enum member \"abe\" must be accepted"
    );
    assert!(
        accepts(&g, b"\"abf\""),
        "enum member \"abf\" must be accepted"
    );
    assert!(
        rejects(&g, b"\"abg\""),
        "a string in no branch must be rejected (no over-accept)"
    );
}

/// A non-string `enum` sibling (`[1, 2]`) must classify as a non-string
/// branch, not fold into the string literal trie (issue #310 / PR #472).
/// The consts form the trie; the numeric enum stays reachable because it
/// cannot begin with `"`. Mutation guard: misclassifying the numeric enum as
/// a string literal set drops the `1`/`2` branch; keeping the branches flat
/// over-rejects `"abd"`.
#[test]
fn anyof_numeric_enum_sibling() {
    let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"const":"abd"},{"enum":[1,2]}]}"#);
    assert!(
        accepts(&g, b"\"abd\""),
        "\"abd\" must be accepted via the trie"
    );
    assert!(accepts(&g, b"1"), "numeric enum member 1 must be reachable");
    assert!(accepts(&g, b"2"), "numeric enum member 2 must be reachable");
    assert!(
        rejects(&g, b"\"abe\""),
        "a non-member string must be rejected"
    );
}

/// A typed mixed `enum` sibling `{"type":"string","enum":["abe",1]}`: the
/// `type:"string"` makes the integer member unsatisfiable, so the branch's
/// language is exactly {"abe"} and `string_class_of` folds it into the
/// literal trie (issue #310 / PR #472). The trie
/// then accepts both the const "abc" and the folded "abe". Mutation guard:
/// dropping the typed-mixed fold (classifying the enum as an "other") strands
/// "abe" behind the const trie and over-rejects it.
#[test]
fn anyof_typed_mixed_enum_folded() {
    let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"type":"string","enum":["abe",1]}]}"#);
    assert!(accepts(&g, b"\"abc\""), "const \"abc\" must be accepted");
    assert!(
        accepts(&g, b"\"abe\""),
        "the type:string enum's string member \"abe\" must be folded and accepted"
    );
    assert!(
        rejects(&g, b"\"abd\""),
        "a string in no branch must be rejected (no over-accept)"
    );
    assert!(
        rejects(&g, b"\"zzz\""),
        "the dead integer member must not widen the branch to any-string"
    );
    assert!(
        rejects(&g, b"1"),
        "the integer enum member is unsatisfiable under type:string"
    );
}

/// A `{"type":"string","enum":[1,2]}` branch is unsatisfiable: an instance
/// must be a string AND equal a numeric enum member, so it accepts no string.
/// `string_class_of` must classify it as the empty literal set (a dead
/// branch), NOT `None` — `None` routes it through `compile_string_type`'s
/// `json_string` fallback and over-accepts arbitrary strings (issue #472,
/// finding 2). Mutation guard: returning `None` accepts "zzz".
#[test]
fn anyof_typed_enum_no_string_members_rejected() {
    let g = compile_ok(r#"{"anyOf":[{"type":"integer"},{"type":"string","enum":[1,2]}]}"#);
    assert!(accepts(&g, b"7"), "the integer branch must stay reachable");
    assert!(
        rejects(&g, b"\"zzz\""),
        "the unsatisfiable type:string + numeric-enum branch must not widen to any-string"
    );
}

/// `const` and `enum` are conjunctive, so `{"type":"string","const":"x","enum":["y"]}`
/// accepts no value (nothing equals both "x" and "y"). `string_class_of` must
/// fold the const only when a present enum contains it, classifying this
/// conflicting branch as the empty literal set (issue #472,
/// finding 1). Mutation guard: folding the const before checking the enum
/// accepts "x"; folding the enum before checking the const accepts "y".
#[test]
fn anyof_const_conflicting_enum_rejected() {
    let g =
        compile_ok(r#"{"anyOf":[{"type":"integer"},{"type":"string","const":"x","enum":["y"]}]}"#);
    assert!(accepts(&g, b"7"), "the integer branch must stay reachable");
    assert!(
        rejects(&g, b"\"x\""),
        "const \"x\" conflicts with enum [\"y\"] — empty intersection, reject"
    );
    assert!(
        rejects(&g, b"\"y\""),
        "enum member \"y\" conflicts with const \"x\" — empty intersection, reject"
    );
}

/// `{"type":"string","const":"x","enum":["x","y"]}`: the const narrows the
/// enum to the intersection {"x"}, so only "x" is accepted, NOT "y" (issue
/// #472). Mutation guard: ignoring the const and folding the
/// whole enum accepts "y".
#[test]
fn anyof_const_compatible_enum_folds_to_const() {
    let g = compile_ok(
        r#"{"anyOf":[{"type":"integer"},{"type":"string","const":"x","enum":["x","y"]}]}"#,
    );
    assert!(accepts(&g, b"7"), "the integer branch must stay reachable");
    assert!(
        accepts(&g, b"\"x\""),
        "const \"x\" is in the enum — accepted"
    );
    assert!(
        rejects(&g, b"\"y\""),
        "\"y\" is in the enum but excluded by const \"x\" — reject"
    );
}

/// Standalone `{"type":"string","enum":[1,2]}` is unsatisfiable;
/// `compile_string_type` must emit the empty language, not the `json_string`
/// fallback (standalone path). Mutation guard: the
/// `json_string` fallback accepts "zzz".
#[test]
fn standalone_typed_enum_no_string_members_rejects_all() {
    let g = compile_ok(r#"{"type":"string","enum":[1,2]}"#);
    assert!(
        rejects(&g, b"\"zzz\""),
        "no string satisfies type:string + enum [1,2]"
    );
    assert!(rejects(&g, b"1"), "the numeric member is not a string");
}

/// Standalone const/enum intersection through `compile_string_type`, which
/// `compile_schema_inner` reaches via the `enum` dispatch — without the
/// intersection a sibling `const` would be silently ignored (standalone
/// path). Mutation guard: dropping the const-narrowing
/// accepts "y" in both cases.
#[test]
fn standalone_string_const_enum_intersection() {
    let conflict = compile_ok(r#"{"type":"string","const":"x","enum":["y"]}"#);
    assert!(
        rejects(&conflict, b"\"x\""),
        "const conflicts with enum — reject \"x\""
    );
    assert!(
        rejects(&conflict, b"\"y\""),
        "const conflicts with enum — reject \"y\""
    );
    let compatible = compile_ok(r#"{"type":"string","const":"x","enum":["x","y"]}"#);
    assert!(
        accepts(&compatible, b"\"x\""),
        "const \"x\" is in the enum — accept"
    );
    assert!(
        rejects(&compatible, b"\"y\""),
        "\"y\" excluded by const \"x\" — reject"
    );
}

/// Edge cases for typed string enums: an empty `enum` is the empty language
/// (reject all), and a duplicate member folds to a single trie path without
/// error (issue #472).
#[test]
fn typed_string_enum_empty_and_duplicate_members() {
    let empty = compile_ok(r#"{"type":"string","enum":[]}"#);
    assert!(rejects(&empty, b"\"zzz\""), "empty enum accepts nothing");
    let dups = compile_ok(r#"{"type":"string","enum":["dup","dup"]}"#);
    assert!(
        accepts(&dups, b"\"dup\""),
        "a duplicated member still accepts once"
    );
    assert!(rejects(&dups, b"\"other\""), "no over-accept from dedup");
    let in_anyof = compile_ok(r#"{"anyOf":[{"type":"integer"},{"type":"string","enum":[]}]}"#);
    assert!(accepts(&in_anyof, b"7"), "integer branch reachable");
    assert!(
        rejects(&in_anyof, b"\"zzz\""),
        "empty typed enum contributes no string"
    );
}

// -----------------------------------------------------------------------
// Issue #473 — Tier 1 fixes: a `$ref` to a string schema, a pure nested
// `anyOf`/`oneOf`, and an untyped mixed `enum` now fold into the hoisted
// string entry instead of being shadowed by it. Tier 2 (an untyped `{}`
// and a `{"type":[...]}` union) remains a documented, pinned limitation
// — fixing it needs parallel-stack matching, not a compile-time fold.
// -----------------------------------------------------------------------

/// FIXED (issue #473, shape 1): a `$ref` to a broad-string rule, listed
/// beside string consts, previously had its quoted inputs shadowed by the
/// hoisted const trie (the no-rewind PDA commits to the trie once it
/// consumes the opening `"`). `compile_any_of` now recovers the `$ref`
/// branch's string language via `ref_string_contribution` (terminal string
/// class ∩ chain narrowing), so a `$ref`-to-broad-string sibling with no
/// narrowing widens the whole class to `json_string`, exactly as a literal
/// broad string would. This schema was previously pinned as a known
/// limitation; it now asserts the fix.
///
/// Mutation guard: reverting the `$ref` reclassification in
/// `compile_any_of` (treating every `None` as a plain "other" branch
/// again) reintroduces the shadow and rejects "zzz".
#[test]
fn anyof_ref_to_broad_string_folds_and_accepts() {
    let g = compile_ok(
        r##"{"$defs":{"S":{"type":"string"}},"anyOf":[{"const":"abc"},{"const":"abd"},{"$ref":"#/$defs/S"}]}"##,
    );
    assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
    assert!(accepts(&g, b"\"abd\""), "const \"abd\" still accepted");
    assert!(
        accepts(&g, b"\"zzz\""),
        "issue #473 fix: $ref->broad-string sibling must now be reachable"
    );
    assert!(rejects(&g, b"123"), "a non-string must still be rejected");
}

/// FIXED (issue #473, shape 1, narrow variant): a `$ref` to a schema that
/// reduces to a fixed LITERAL set (not a broad string) folds its literal
/// into the SAME trie as the sibling const, rather than widening to any
/// string. Proves the fold narrows correctly.
///
/// Mutation guard: if the fold incorrectly treated a resolved
/// `StrClass::Literals` target as `Broad`, "abd" (in neither literal set)
/// would be wrongly accepted.
#[test]
fn anyof_ref_to_string_literal_folds_into_trie() {
    let g = compile_ok(
        r##"{"$defs":{"S":{"const":"zzz"}},"anyOf":[{"const":"abc"},{"$ref":"#/$defs/S"}]}"##,
    );
    assert!(accepts(&g, b"\"abc\""), "const \"abc\" accepted");
    assert!(
        accepts(&g, b"\"zzz\""),
        "issue #473 fix: $ref->const literal must be folded into the trie"
    );
    assert!(
        rejects(&g, b"\"abd\""),
        "a string in neither literal must be rejected (no over-accept)"
    );
}

/// A `$ref` to a NON-string schema must stay an "other" branch, unchanged
/// — the design explicitly requires that a `$ref` whose target is
/// non-string keeps its prior behavior (issue #473). The ref'd integer
/// branch stays reachable and the const trie is unaffected.
///
/// Mutation guard: if reclassification incorrectly widened a non-string
/// ref target to accept strings, "zzz" would be wrongly accepted here.
#[test]
fn anyof_ref_to_non_string_stays_other() {
    let g = compile_ok(
        r##"{"$defs":{"N":{"type":"integer"}},"anyOf":[{"const":"abc"},{"$ref":"#/$defs/N"}]}"##,
    );
    assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
    assert!(
        accepts(&g, b"7"),
        "the ref'd integer branch stays reachable"
    );
    assert!(
        rejects(&g, b"\"zzz\""),
        "a non-member string must still be rejected"
    );
}

/// FIXED (issue #473, chain-collect intersection): a branch that is `$ref`
/// to a NON-string target AND carries a STRING-FORCING sibling (a string
/// `const`, `type:"string"`, or an all-string `enum`) is the EMPTY language
/// — no value is both a string and an integer — so the branch must accept
/// NOTHING. `origin/main` OVER-ACCEPTED the sibling string here (it
/// mis-hoisted the sibling into the shared string entry, accepting "y" /
/// "zzz"), while rejecting the number. `ref_string_contribution` resolves
/// the string part to `RefStr::NotString`, and `ref_sub_forces_string`
/// makes `compile_any_of` DROP the branch as an exact empty intersection
/// instead of routing it to the generic fail-closed path.
///
/// Mutation guard: reverting the drop (routing `RefStr::NotString` straight
/// to `other_subs` again) reopens a number over-accept — `7` would be
/// wrongly accepted on every sub-case below. Removing the string guard in
/// `ref_string_contribution` reopens the STRING over-accept — "y"/"zzz"
/// would be wrongly accepted.
#[test]
fn anyof_ref_to_non_string_with_string_forcing_sibling_drops() {
    // string `const` sibling
    let g = compile_ok(
        r##"{"$defs":{"N":{"type":"integer"}},"anyOf":[{"const":"abc"},{"$ref":"#/$defs/N","const":"y"}]}"##,
    );
    assert!(
        accepts(&g, b"\"abc\""),
        "the peer const \"abc\" still accepted"
    );
    assert!(
        rejects(&g, b"\"y\""),
        "issue #473: a string const sibling on a $ref->integer branch must not over-accept the string"
    );
    assert!(
        rejects(&g, b"7"),
        "the string ∧ integer conjunction is empty — the number must be rejected too (no over-accept)"
    );

    // `type:"string"` sibling
    let g = compile_ok(
        r##"{"$defs":{"N":{"type":"integer"}},"anyOf":[{"const":"abc"},{"$ref":"#/$defs/N","type":"string"}]}"##,
    );
    assert!(
        rejects(&g, b"\"zzz\""),
        "type:string sibling on a $ref->integer branch must not over-accept"
    );
    assert!(
        rejects(&g, b"7"),
        "empty conjunction rejects the number too"
    );

    // all-string `enum` sibling
    let g = compile_ok(
        r##"{"$defs":{"N":{"type":"integer"}},"anyOf":[{"const":"abc"},{"$ref":"#/$defs/N","enum":["y","z"]}]}"##,
    );
    assert!(
        rejects(&g, b"\"y\""),
        "all-string enum sibling on a $ref->integer branch must not over-accept"
    );
    assert!(
        rejects(&g, b"7"),
        "empty conjunction rejects the number too"
    );
}

/// FIXED (issue #473): the string-forcing keyword sits on an
/// INTERMEDIATE `$defs` node, not the outer `anyOf` sub. `S` is
/// `{$ref: N-integer, type/const/enum-string}`, so `S`'s language is the
/// empty conjunction (integer ∧ string); the outer branch `{$ref: S}` is a
/// PURE `$ref` with no string-forcing keyword of its own. An outer-sub-only
/// `ref_sub_forces_string(sub)` check misses the forcing and routes the
/// branch to `other_subs`, where `compile_ref` materializes `N` and
/// over-accepts the integer `7`. The chain-level `chain_forces_string` state
/// (accumulated across every walked node) classifies the branch `RefStr::Dead`
/// and drops it, so both the string AND `7` are rejected.
///
/// Mutation guard: reverting to an outer-sub-only forcing check (or dropping
/// the `RefStr::Dead` arm to `other_subs`) reopens the number over-accept —
/// `7` would be wrongly accepted on every sub-case below.
#[test]
fn anyof_ref_chain_intermediate_string_forcing_node_drops() {
    // intermediate `type:"string"`
    let g = compile_ok(
        r##"{"$defs":{"S":{"$ref":"#/$defs/N","type":"string"},"N":{"type":"integer"}},"anyOf":[{"const":"ok"},{"$ref":"#/$defs/S"}]}"##,
    );
    assert!(
        accepts(&g, b"\"ok\""),
        "the peer const \"ok\" still accepted"
    );
    assert!(
        rejects(&g, b"\"zzz\""),
        "issue #473: an intermediate type:string node on a $ref->integer chain must not over-accept the string"
    );
    assert!(
        rejects(&g, b"7"),
        "the intermediate string ∧ integer conjunction is empty — the number must be rejected too"
    );

    // intermediate string `const`
    let g = compile_ok(
        r##"{"$defs":{"S":{"$ref":"#/$defs/N","const":"y"},"N":{"type":"integer"}},"anyOf":[{"const":"ok"},{"$ref":"#/$defs/S"}]}"##,
    );
    assert!(
        rejects(&g, b"\"y\""),
        "intermediate string const on a $ref->integer chain must not over-accept"
    );
    assert!(
        rejects(&g, b"7"),
        "empty conjunction rejects the number too"
    );

    // intermediate all-string `enum`
    let g = compile_ok(
        r##"{"$defs":{"S":{"$ref":"#/$defs/N","enum":["y","z"]},"N":{"type":"integer"}},"anyOf":[{"const":"ok"},{"$ref":"#/$defs/S"}]}"##,
    );
    assert!(
        rejects(&g, b"\"y\""),
        "intermediate all-string enum on a $ref->integer chain must not over-accept"
    );
    assert!(
        rejects(&g, b"7"),
        "empty conjunction rejects the number too"
    );
}

/// FIXED (issue #473): draft-2020-12 §6.1.1 lets `type`
/// be an ARRAY of type names. A string-ONLY type array (`["string"]`, or any
/// non-empty all-`"string"` array) is string-forcing, exactly like the scalar
/// `type:"string"`. On a `$ref`->integer terminal — DIRECT sibling or an
/// intermediate `$defs` node — the conjunction is empty, so the branch must be
/// dropped (`RefStr::Dead`), rejecting both the string AND `7`. A `type`
/// array that admits any non-string type (`["string","null"]`,
/// `["integer"]`) is not representable by this specialized fold and fails
/// closed in the generic path, as covered below.
///
/// Mutation guard: reverting `ref_sub_forces_string` to recognize only the
/// scalar `type:"string"` (dropping the array arm) reopens the number
/// over-accept — `7` would be wrongly accepted on both sub-cases below.
#[test]
fn anyof_ref_type_array_string_only_forces_drop() {
    // DIRECT sibling `type:["string"]`
    let g = compile_ok(
        r##"{"$defs":{"N":{"type":"integer"}},"anyOf":[{"const":"ok"},{"$ref":"#/$defs/N","type":["string"]}]}"##,
    );
    assert!(
        accepts(&g, b"\"ok\""),
        "the peer const \"ok\" still accepted"
    );
    assert!(
        rejects(&g, b"\"zzz\""),
        "a string-only type array on a $ref->integer branch must not over-accept the string"
    );
    assert!(
        rejects(&g, b"7"),
        "the string ∧ integer conjunction is empty — the number must be rejected too"
    );

    // INTERMEDIATE `$defs` node with `type:["string"]`
    let g = compile_ok(
        r##"{"$defs":{"S":{"$ref":"#/$defs/N","type":["string"]},"N":{"type":"integer"}},"anyOf":[{"const":"ok"},{"$ref":"#/$defs/S"}]}"##,
    );
    assert!(
        rejects(&g, b"\"zzz\""),
        "an intermediate string-only type array must not over-accept"
    );
    assert!(
        rejects(&g, b"7"),
        "empty conjunction rejects the number too"
    );
}

/// A mixed sibling `type` array that covers the target's declared type is
/// redundant. The generic `$ref` path may therefore compile the integer
/// target alone, while the peer branch remains intact.
#[test]
fn anyof_ref_type_array_covering_target_preserves_target() {
    let schema = serde_json::json!({
        "$defs": { "N": { "type": "integer" } },
        "anyOf": [
            { "const": "ok" },
            { "$ref": "#/$defs/N", "type": ["string", "integer"] }
        ]
    });
    let g = compile(&schema).expect("the sibling type array covers the target type");
    assert!(accepts(&g, b"7"), "the referenced integer must be accepted");
    assert!(
        accepts(&g, b"\"ok\""),
        "the peer const branch must remain accepted"
    );
    assert!(
        rejects(&g, b"\"other\""),
        "dropping the redundant sibling must not drop the integer target"
    );
}

/// A sibling type array that omits the target's declared type is not
/// redundant and must continue to fail closed.
#[test]
fn anyof_ref_type_array_not_covering_target_fails_closed() {
    let schema = serde_json::json!({
        "$defs": { "N": { "type": "integer" } },
        "anyOf": [
            { "const": "ok" },
            { "$ref": "#/$defs/N", "type": ["string", "null"] }
        ]
    });
    let err = compile(&schema)
        .expect_err("a sibling type array that omits the target type must fail closed");
    assert!(
        err.0.contains("`type` sibling"),
        "error should name the unrepresentable type sibling, got: {}",
        err.0
    );
}

/// A numeric `enum` cannot use the specialized string intersection either.
/// Routing it through the generic `$ref` path must reject the unsupported
/// conjunction instead of compiling the whole integer target and admitting
/// enum non-members such as `7`.
#[test]
fn anyof_ref_to_integer_with_numeric_enum_sibling_fails_closed() {
    let schema = serde_json::json!({
        "$defs": { "N": { "type": "integer" } },
        "anyOf": [
            { "const": "abc" },
            { "$ref": "#/$defs/N", "enum": [1, 2] }
        ]
    });
    let err = compile(&schema)
        .expect_err("an unrepresentable enum sibling inside anyOf must fail closed");
    assert!(
        err.0.contains("`enum` sibling"),
        "error should name the unrepresentable enum sibling, got: {}",
        err.0
    );
}

/// FIXED (issue #473, chain-collect intersection): a DIRECT string `enum`
/// sibling on a branch whose `$ref` resolves to a BROAD string narrows the
/// branch to that enum. The specialized `anyOf` path intersects the
/// terminal's broad string class with the sibling narrowing, folding exactly
/// `{"abc"}` into the shared trie rather than reaching the generic
/// fail-closed intersection path.
///
/// Mutation guard: dropping the sibling narrowing (returning the terminal's
/// broad class unintersected) would accept "zzz" — a re-widening over-accept.
#[test]
fn anyof_ref_to_broad_string_with_enum_sibling_narrows() {
    let g = compile_ok(
        r##"{"$defs":{"T":{"type":"string"}},"anyOf":[{"const":"foo"},{"$ref":"#/$defs/T","enum":["abc"]}]}"##,
    );
    assert!(
        accepts(&g, b"\"foo\""),
        "the peer const \"foo\" still accepted"
    );
    assert!(
        accepts(&g, b"\"abc\""),
        "issue #473: the sibling enum member is reachable (terminal-broad intersect the enum)"
    );
    assert!(
        rejects(&g, b"\"zzz\""),
        "the enum-narrowed branch must reject a non-member string (no over-accept)"
    );
}

/// FIXED (issue #473, chain-collect intersection): the narrowing is
/// collected along the WHOLE `$ref` chain, not just the direct sibling. Here
/// the intermediate `$defs` node `S` states `enum:["abc"]` and `$ref`s `T`
/// (a broad string); the outer branch `$ref`s `S`. `ref_string_contribution`
/// walks sub -> S -> T, intersecting `S`'s `{"abc"}` narrowing with `T`'s
/// broad class, folding `{"abc"}`.
///
/// Mutation guard: if the chain walk stopped collecting narrowing at the
/// first hop (or reset it per hop), the intermediate `enum` would be lost and
/// "zzz" wrongly accepted.
#[test]
fn anyof_ref_chain_collects_intermediate_enum_narrowing() {
    let g = compile_ok(
        r##"{"$defs":{"S":{"$ref":"#/$defs/T","enum":["abc"]},"T":{"type":"string"}},"anyOf":[{"const":"foo"},{"$ref":"#/$defs/S"}]}"##,
    );
    assert!(
        accepts(&g, b"\"foo\""),
        "the peer const \"foo\" still accepted"
    );
    assert!(
        accepts(&g, b"\"abc\""),
        "issue #473: an enum on an intermediate chain node narrows the folded set"
    );
    assert!(
        rejects(&g, b"\"zzz\""),
        "the chain-collected narrowing must reject a non-member string (no over-accept)"
    );
}

/// An unresolvable `$ref` (unknown definition name) inside an `anyOf`
/// must still surface the pre-existing "$ref not found" compile error —
/// the classification-time resolution attempt (issue #473) must not
/// swallow or change this error, only ever supply an ADDITIONAL fold
/// opportunity when resolution succeeds.
///
/// Mutation guard: if `ref_string_contribution` propagated an error instead
/// of returning `RefStr::NotString` on a failed lookup, this would fail with
/// the wrong error (or panic) instead of the expected message.
#[test]
fn anyof_ref_to_missing_def_still_errors() {
    let schema_json = r##"{"anyOf":[{"const":"abc"},{"$ref":"#/$defs/Missing"}]}"##;
    let v: Value = serde_json::from_str(schema_json).unwrap();
    let err = compile(&v).expect_err("an unresolvable $ref must still fail to compile");
    assert!(
        err.0.contains("$ref not found"),
        "expected the pre-existing $ref-not-found error, got: {}",
        err.0
    );
}

/// FIXED (issue #473, shape 2): a nested `anyOf` containing a broad
/// string (`{"anyOf":[{"anyOf":[{"type":"string"}]}]}`) previously
/// classified as an "other" sibling because `string_class_of` does not
/// recurse into nested unions, shadowing its quoted inputs behind the
/// hoisted const trie. `compile_any_of` now flattens a PURE nested union
/// (`flatten_any_of_branches`) before classification, so the inner broad
/// string classifies exactly like a top-level branch. Previously pinned
/// as a known limitation; now asserts the fix.
///
/// Mutation guard: reverting the flatten step (classifying the raw
/// top-level `any_of` again) reintroduces the shadow and rejects "zzz".
#[test]
fn anyof_nested_anyof_string_folds_and_accepts() {
    let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"anyOf":[{"type":"string"}]}]}"#);
    assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
    assert!(
        accepts(&g, b"\"zzz\""),
        "issue #473 fix: nested-anyOf broad string must now be reachable"
    );
}

/// A nested `oneOf` is NOT flattened into the parent's OR-hoist, unlike a
/// nested `anyOf` (`oneOf` is exclusive and exclusivity is not enforced
/// at runtime). Its string branch stays reachable anyway: the matcher
/// keeps every live alternative, so the hoisted `"abc"` trie and the
/// nested string branch both consume the opening quote and `"zzz"`
/// follows the string branch. A matcher that committed to the hoisted
/// trie at the quote used to reject it.
///
/// Mutation guard: keeping only the first matching alternative after a
/// byte (see `alternatives_sharing_a_first_byte_both_stay_alive` in
/// `pda.rs`) rejects "zzz" again.
#[test]
fn anyof_nested_oneof_string_branch_stays_reachable() {
    let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"oneOf":[{"type":"string"}]}]}"#);
    assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
    assert!(
        accepts(&g, b"\"zzz\""),
        "a nested oneOf's string branch must stay reachable next to a sibling const trie"
    );
}

/// A nested `oneOf` with TWO overlapping branches follows `anyOf`
/// (union) semantics at runtime, exactly as a top-level `oneOf` does:
/// `oneOf` is lowered like `anyOf` and only literal overlaps that are
/// decidable at compile time are refused (see `check_one_of_exclusivity`).
/// The broad `type:string` and `const:"zzz"` both match "zzz", so exactly-one
/// semantics would refuse that shared value, but the grammar accepts it.
///
/// Before the matcher kept sibling alternatives alive, "zzz" was rejected
/// only as a side effect of the hoisted `"abc"` trie shadowing the whole
/// nested branch. That shadowing also rejected "qqq", which matches exactly
/// one inner branch and is valid under any reading of `oneOf`. The
/// assertions below pin the values the grammar can and cannot reach now.
#[test]
fn anyof_nested_oneof_overlap_follows_union_semantics() {
    let g =
        compile_ok(r#"{"anyOf":[{"const":"abc"},{"oneOf":[{"type":"string"},{"const":"zzz"}]}]}"#);
    assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
    assert!(
        accepts(&g, b"\"qqq\""),
        "a string matching exactly one inner branch is valid and must be reachable"
    );
    assert!(
        accepts(&g, b"\"zzz\""),
        "the shared value follows union semantics: oneOf exclusivity is not enforced at runtime"
    );
    assert!(
        rejects(&g, b"1"),
        "a number matches no branch and must still be rejected"
    );
}

#[test]
fn oneof_overlapping_literal_branches_rejected() {
    let schema = serde_json::json!({
        "oneOf": [
            {"const": "a"},
            {"enum": ["a", "b"]}
        ]
    });
    let err = compile_json_schema(&schema)
        .expect_err("oneOf branches overlapping on a literal value must be rejected");
    assert!(
        err.0.contains("oneOf"),
        "error should name the oneOf violation, got: {}",
        err.0
    );
}

#[test]
fn oneof_disjoint_literal_branches_still_compiles() {
    let schema = serde_json::json!({
        "oneOf": [
            {"const": "a"},
            {"const": "b"}
        ]
    });
    let g = compile_json_schema(&schema).unwrap();
    assert!(accepts(&g, b"\"a\""));
    assert!(accepts(&g, b"\"b\""));
    assert!(rejects(&g, b"\"c\""));
}

/// `oneOf`'s branches are an unordered set, so an undecidable branch must
/// not hide an overlap that sits on the other side of it. The first two
/// orderings below are the discriminating ones: a check that abandoned the
/// whole schema at the first open-ended branch would compile them, and the
/// same three branches would then reject or compile depending only on the
/// order they were written in.
#[test]
fn oneof_literal_overlap_found_across_an_open_ended_branch() {
    for branches in [
        serde_json::json!([{"const": "a"}, {"type": "object"}, {"const": "a"}]),
        serde_json::json!([{"type": "object"}, {"const": "a"}, {"const": "a"}]),
        serde_json::json!([{"const": "a"}, {"const": "a"}, {"type": "object"}]),
    ] {
        let schema = serde_json::json!({ "oneOf": branches });
        let err = compile_json_schema(&schema).expect_err(
                "two branches overlapping on a literal must be rejected wherever the open-ended branch sits",
            );
        assert!(
            err.0.contains("oneOf"),
            "error should name the oneOf violation, got: {}",
            err.0
        );
    }
}

/// A branch's literal is only the value it matches once the branch's other
/// assertions are taken into account. `{"const":"a","type":"integer"}`
/// asserts a value that is both the string `"a"` and an integer, which no
/// JSON value is, so the branch matches nothing and `"a"` matches exactly
/// one branch — a valid `oneOf`. Crediting the branch with its bare literal
/// manufactures an overlap and rejects a schema that is fine.
#[test]
fn oneof_contradictory_branch_does_not_manufacture_overlap() {
    let schema = serde_json::json!({
        "oneOf": [
            {"const": "a", "type": "integer"},
            {"const": "a"}
        ]
    });
    let g = compile_json_schema(&schema)
        .expect("a branch matching no value cannot overlap another branch");
    assert!(accepts(&g, b"\"a\""), "the reachable branch still matches");
}

/// The other half of the same rule: an assertion this compiler cannot
/// intersect (`minLength` here) may exclude the literal — `"a"` is one
/// character, so the branch matches nothing — and the branch must be
/// skipped rather than credited with a literal it may not match.
#[test]
fn oneof_branch_with_unanalysable_assertion_is_skipped() {
    let schema = serde_json::json!({
        "oneOf": [
            {"const": "a", "minLength": 5},
            {"const": "a"}
        ]
    });
    let g = compile_json_schema(&schema)
        .expect("a branch carrying an un-intersectable assertion cannot prove an overlap");
    assert!(accepts(&g, b"\"a\""), "the analysable branch still matches");
}

/// The same accounting must not go the other way: a `type` that is
/// CONSISTENT with the literal leaves the branch matching it, so a genuine
/// overlap is still found.
#[test]
fn oneof_overlap_still_found_when_type_is_consistent() {
    let schema = serde_json::json!({
        "oneOf": [
            {"const": "a", "type": "string"},
            {"enum": ["a", "b"]}
        ]
    });
    let err = compile_json_schema(&schema)
        .expect_err("consistent type must leave the literal in the branch's value set");
    assert!(
        err.0.contains("oneOf"),
        "error should name the oneOf violation, got: {}",
        err.0
    );
}

/// A repeated member inside ONE `enum` is not two branches. Draft 2020-12
/// says `enum` values SHOULD be unique, not MUST, and a duplicate member
/// does not give the instance a second branch to match, so this schema is
/// genuinely exclusive and must compile. Comparing a branch's raw value
/// list against the shared cross-branch set makes the branch collide with
/// ITSELF, which is both a false rejection and an error message naming an
/// "earlier branch" that does not exist for branch 0.
#[test]
fn oneof_duplicate_enum_member_within_one_branch_is_not_an_overlap() {
    let schema = serde_json::json!({
        "oneOf": [
            {"enum": ["a", "a"]},
            {"const": "b"}
        ]
    });
    let g = compile_json_schema(&schema)
        .expect("a duplicate inside one enum does not give a value a second matching branch");
    assert!(accepts(&g, b"\"a\""), "the duplicated member still matches");
    assert!(accepts(&g, b"\"b\""), "the other branch still matches");
}

/// JSON Schema compares numbers by mathematical value, so `1` and `1.0`
/// are the SAME instance and these two branches both match it. Keying the
/// overlap set by lexical JSON rendering separates them (`"1"` vs `"1.0"`)
/// and lets an ambiguous `oneOf` compile.
#[test]
fn oneof_integer_and_float_spelling_of_one_value_overlap() {
    let schema = serde_json::json!({
        "oneOf": [
            {"const": 1},
            {"const": 1.0}
        ]
    });
    let err = compile_json_schema(&schema)
        .expect_err("1 and 1.0 are the same number, so both branches match it");
    assert!(
        err.0.contains("oneOf"),
        "error should name the oneOf violation, got: {}",
        err.0
    );
}

/// The other direction of the same keying rule, and the reason it must not
/// route through `f64`: these two integers are distinct but land on the
/// same `f64` past 2^53, so a lossy canonicalization would report a false
/// overlap and reject a schema whose branches really are exclusive.
/// Compiling at all is the assertion here.
#[test]
fn oneof_large_integers_indistinguishable_as_f64_stay_distinct() {
    let schema = serde_json::json!({
        "oneOf": [
            {"const": 9007199254740993i64},
            {"const": 9007199254740992i64}
        ]
    });
    let g = compile_json_schema(&schema)
        .expect("integers differing past f64 precision are distinct values");
    assert!(accepts(&g, b"9007199254740993"));
}

/// Exponent notation is a SPELLING, not a distinct value: `1e20` and
/// `100000000000000000000` are the same number, and JSON Schema compares
/// instances by mathematical value. Both exceed `u64::MAX`, so neither
/// takes `overlap_key`'s exact-integer path and both fall to the JSON
/// rendering — which, without `serde_json`'s `arbitrary_precision`
/// feature, has already parsed each to the same `f64` and therefore
/// renders them identically. The overlap is caught by that collapse rather
/// than in spite of it, so this pins the outcome against a future
/// `arbitrary_precision` bump, which would preserve the two spellings and
/// silently split the key.
#[test]
fn oneof_exponent_and_longhand_spelling_of_one_value_overlap() {
    let schema: Value =
        serde_json::from_str(r#"{"oneOf":[{"const":1e20},{"const":100000000000000000000}]}"#)
            .expect("both literals are valid JSON numbers");
    let err = compile_json_schema(&schema).expect_err(
        "1e20 and 100000000000000000000 are the same number, so both branches match it",
    );
    assert!(
        err.0.contains("oneOf"),
        "error should name the oneOf violation, got: {}",
        err.0
    );
}

/// A nested union with an EXTRA sibling key (`"description"`) is NOT a
/// PURE nested union, so `flatten_any_of_branches` deliberately leaves it
/// unflattened (`as_pure_nested_union` only recognizes an object whose
/// ONLY key is `anyOf` / `oneOf`). It falls through to the `other_subs`
/// path, and the matcher keeps that branch alive next to the hoisted
/// const trie, so its broad string is reachable without flattening.
#[test]
fn anyof_nested_anyof_with_sibling_key_stays_reachable_unflattened() {
    let g = compile_ok(
        r#"{"anyOf":[{"const":"abc"},{"anyOf":[{"type":"string"}],"description":"nested"}]}"#,
    );
    assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
    assert!(
        accepts(&g, b"\"zzz\""),
        "an unflattened nested union's broad string must stay reachable"
    );
}

/// FIXED (issue #473, shape 5): an untyped `enum` mixing a string and a
/// non-string member (`{"enum":["abe",1]}`, no `type` keyword) previously
/// classified as `None` and was shadowed entirely — including its string
/// member "abe" — behind the hoisted const trie. `fold_string_members`
/// now folds "abe" into the trie while the branch stays in `other_subs`
/// too, keeping `1` reachable via its disjoint first byte.
///
/// Mutation guard: reverting the shape-5 fold strands "abe" behind the
/// const trie (rejected); reverting keeping the branch in `other_subs`
/// loses reachability of `1`.
#[test]
fn anyof_untyped_mixed_enum_folds_string_member() {
    let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"enum":["abe",1]}]}"#);
    assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
    assert!(
        accepts(&g, b"\"abe\""),
        "issue #473 fix: untyped mixed-enum string member must now be reachable"
    );
    assert!(
        accepts(&g, b"1"),
        "the non-string enum member 1 stays reachable"
    );
    assert!(
        rejects(&g, b"\"abd\""),
        "a string in no branch must still be rejected (no over-accept)"
    );
}

/// Shape 5 with MULTIPLE string members and multiple non-string members:
/// every string member folds into the trie, and every non-string member
/// stays reachable via `other_subs`.
#[test]
fn anyof_untyped_mixed_enum_multiple_members_all_reachable() {
    let g = compile_ok(r#"{"anyOf":[{"enum":["abe","abf",1,null]}]}"#);
    assert!(accepts(&g, b"\"abe\""), "string member \"abe\" reachable");
    assert!(accepts(&g, b"\"abf\""), "string member \"abf\" reachable");
    assert!(accepts(&g, b"1"), "non-string member 1 reachable");
    assert!(accepts(&g, b"null"), "non-string member null reachable");
    assert!(
        rejects(&g, b"\"abg\""),
        "a string not in the enum must be rejected (no over-accept)"
    );
}

/// An untyped `{}` sibling accepts any JSON value, including strings that
/// start the same way as the hoisted const trie: the matcher keeps both
/// alternatives alive after the opening quote.
#[test]
fn anyof_untyped_sibling_accepts_any_string() {
    let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{}]}"#);
    assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
    assert!(
        accepts(&g, b"\"zzz\""),
        "untyped empty-schema string inputs must stay reachable next to the const trie"
    );
}

/// A `{"type":["string","number"]}` sibling accepts both strings and
/// numbers. This compiler does not special-case a `type` ARRAY:
/// `compile_schema_inner` dispatches on `Value::as_str`, which is `None`
/// for a JSON array (same as an absent `type`), so it falls to the
/// untyped `any_value_alts` path, identically to `{}`. There is no fixed
/// literal set to fold (the branch's string language is unbounded); the
/// string inputs are reachable because the matcher keeps this branch alive
/// next to the hoisted const trie.
#[test]
fn anyof_mixed_type_array_sibling_accepts_strings() {
    let g = compile_ok(r#"{"anyOf":[{"const":"abc"},{"type":["string","number"]}]}"#);
    assert!(accepts(&g, b"\"abc\""), "const \"abc\" still accepted");
    assert!(
        accepts(&g, b"42"),
        "a non-string alternative from the mixed-type branch stays reachable"
    );
    assert!(
        accepts(&g, b"\"zzz\""),
        "the mixed-type-array sibling's string inputs must stay reachable next to the const trie"
    );
}

/// DoS bound (issue #473 extends issue #474 finding 2): a small
/// top-level `anyOf` (2 branches) whose SECOND branch is a nested pure
/// union containing MAX_ANYOF_BRANCHES sibling branches must still be
/// rejected by the branch-count cap — proving `flatten_any_of_branches`
/// enforces MAX_ANYOF_BRANCHES on the FLATTENED total, not the raw
/// top-level count (which is only 2 here and would sail under the cap on
/// its own).
///
/// Mutation guard: fires with "anyOf/oneOf branch count". Reverting the
/// flatten-time incremental check (bounding only the raw top-level
/// count, as the pre-issue-473 code did) lets this schema compile
/// successfully instead of being rejected.
#[test]
fn anyof_nested_union_flatten_bypass_rejected_by_branch_cap() {
    let over = MAX_ANYOF_BRANCHES + 1;
    let inner_branches: String = (0..over)
        .map(|_| r#"{"type":"boolean"}"#.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let nested_union = format!("{{\"anyOf\":[{inner_branches}]}}");
    let schema_json = format!("{{\"anyOf\":[{{\"const\":\"abc\"}},{nested_union}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err(
            "flattened branch count over the cap must be rejected even though the raw top-level count is only 2",
        );
    assert!(
        err.0.contains("anyOf/oneOf branch count"),
        "expected the anyOf branch-count cap error, got: {}",
        err.0
    );
}

/// DoS bound (issue #473): `flatten_any_of_branches`' own recursion depth
/// must be capped independent of `MAX_SCHEMA_DEPTH`'s use elsewhere. A
/// `Value` tree is built directly here (not via `serde_json::from_str`,
/// so this is not confounded by serde_json's own JSON-text nesting
/// limit) with `MAX_SCHEMA_DEPTH + 50` nested pure `{"anyOf":[...]}`
/// wrappers, and must be rejected rather than overflow the native stack.
///
/// Mutation guard: fires with "anyOf/oneOf nesting depth". Removing the
/// depth check inside `flatten_any_of_branches` either overflows the
/// stack (a process abort, not a test failure this assertion could
/// catch) or, if the recursion happens to survive, compiles successfully
/// instead of being rejected.
#[test]
fn anyof_nested_union_flatten_depth_capped() {
    let mut inner = {
        let mut m = serde_json::Map::new();
        m.insert("type".to_string(), Value::String("boolean".to_string()));
        Value::Object(m)
    };
    for _ in 0..(MAX_SCHEMA_DEPTH + 50) {
        let mut m = serde_json::Map::new();
        m.insert("anyOf".to_string(), Value::Array(vec![inner]));
        inner = Value::Object(m);
    }
    let err = compile(&inner).expect_err("anyOf/oneOf nesting depth over the cap must be rejected");
    assert!(
        err.0.contains("anyOf/oneOf nesting depth"),
        "expected the flatten depth-cap error, got: {}",
        err.0
    );
}

/// Three-way shared prefix ["foo","food","foot"]: all three accepted.
#[test]
fn string_enum_three_way_shared_prefix() {
    let g = compile_ok(r#"{"type":"string","enum":["foo","food","foot"]}"#);
    assert!(accepts(&g, b"\"foo\""), "\"foo\" must be accepted");
    assert!(accepts(&g, b"\"food\""), "\"food\" must be accepted");
    assert!(accepts(&g, b"\"foot\""), "\"foot\" must be accepted");
    assert!(rejects(&g, b"\"fooz\""), "\"fooz\" must be rejected");
    assert!(rejects(&g, b"\"fo\""), "\"fo\" must be rejected");
}

/// Prefix pair ["a","ab"]: both accepted (quoted → prefix-free).
#[test]
fn string_enum_prefix_pair_a_ab() {
    let g = compile_ok(r#"{"type":"string","enum":["a","ab"]}"#);
    assert!(accepts(&g, b"\"a\""), "\"a\" must be accepted");
    assert!(accepts(&g, b"\"ab\""), "\"ab\" must be accepted");
    assert!(rejects(&g, b"\"abc\""), "\"abc\" must be rejected");
    assert!(rejects(&g, b"\"b\""), "\"b\" must be rejected");
}

/// Escaped values: schema `["a\\b","a\\c"]` (string values `a\b` and `a\c`,
/// each containing a literal backslash) share the JSON-encoded byte prefix
/// `a\\`; the trie must diverge at byte `b` vs `c`.
///
/// The JSON encoding of the literal-backslash string `a\b` is `"a\\b"`,
/// whose byte stream is [34,97,92,92,98,34].  In a Rust byte literal that
/// is `b"\"a\\\\b\""` (four backslash chars → two bytes of 0x5C).
#[test]
fn string_enum_escaped_values() {
    // Schema: string values a\b and a\c (literal backslash in each).
    // Their JSON encodings "a\\b" and "a\\c" share the byte prefix
    // [97, 92, 92] = 'a', '\', '\' and diverge at 'b' vs 'c'.
    let g = compile_ok(r#"{"type":"string","enum":["a\\b","a\\c"]}"#);
    // JSON "a\\b" = bytes [34,97,92,92,98,34] = Rust b"\"a\\\\b\""
    assert!(
        accepts(&g, b"\"a\\\\b\""),
        "literal-backslash value a\\b must be accepted"
    );
    assert!(
        accepts(&g, b"\"a\\\\c\""),
        "literal-backslash value a\\c must be accepted"
    );
    assert!(
        rejects(&g, b"\"a\\\\d\""),
        "a\\d (not in enum) must be rejected"
    );
    assert!(
        rejects(&g, b"\"a\""),
        "bare a (no backslash) must be rejected"
    );
}

/// Single-value enum ["solo"]: accepted normally; no regression.
#[test]
fn string_enum_single_value() {
    let g = compile_ok(r#"{"type":"string","enum":["solo"]}"#);
    assert!(accepts(&g, b"\"solo\""), "\"solo\" must be accepted");
    assert!(rejects(&g, b"\"sol\""), "prefix \"sol\" must be rejected");
    assert!(
        rejects(&g, b"\"soloo\""),
        "extension \"soloo\" must be rejected"
    );
}

/// Disjoint ["cat","dog"]: distinct first bytes; both accepted.
#[test]
fn string_enum_disjoint() {
    let g = compile_ok(r#"{"type":"string","enum":["cat","dog"]}"#);
    assert!(accepts(&g, b"\"cat\""), "\"cat\" must be accepted");
    assert!(accepts(&g, b"\"dog\""), "\"dog\" must be accepted");
    assert!(rejects(&g, b"\"cot\""), "\"cot\" must be rejected");
    assert!(rejects(&g, b"\"dig\""), "\"dig\" must be rejected");
}

/// Mixed anyOf[const "abc", {"type":"object"}]: the string "abc" and any
/// valid object are both accepted; non-member strings are rejected.
#[test]
fn anyof_mixed_string_const_and_object() {
    let g = compile_ok(
        r#"{"anyOf":[{"const":"abc"},{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}]}"#,
    );
    assert!(accepts(&g, b"\"abc\""), "const string must be accepted");
    assert!(accepts(&g, b"{\"x\":1}"), "valid object must be accepted");
    assert!(
        rejects(&g, b"\"abd\""),
        "non-member string must be rejected"
    );
}

/// A single long enum string compiles flat (no per-byte recursion) and
/// accepts itself. Regression guard for `build_trie_node` single-child
/// compression: once a member stops sharing a prefix it is emitted flat, so
/// member length never drives recursion depth. Reverting the compression
/// recurses 20_000 deep and aborts the test binary on a stack overflow.
#[test]
fn string_enum_long_single_value_compiles() {
    let val = "a".repeat(20_000);
    let g = compile_ok(&format!(r#"{{"type":"string","enum":["{val}"]}}"#));
    let mut input = vec![b'"'];
    input.extend(std::iter::repeat_n(b'a', 20_000));
    input.push(b'"');
    assert!(accepts(&g, &input), "the long enum value must be accepted");
    assert!(rejects(&g, b"\"a\""), "a shorter prefix must be rejected");
}

/// Two members sharing a prefix longer than `MAX_TRIE_DEPTH` fail closed
/// with a `SchemaError` naming the depth bound, rather than overflowing the
/// stack. Locks the defense-in-depth recursion cap.
#[test]
fn string_enum_pathological_shared_prefix_fails_closed() {
    let pfx = "a".repeat(MAX_TRIE_DEPTH + 50);
    let schema = format!(r#"{{"enum":["{pfx}b","{pfx}c"]}}"#);
    let v: Value = serde_json::from_str(&schema).unwrap();
    let err = compile(&v).expect_err("an absurd shared prefix must be rejected, not crash");
    assert!(
        err.0.contains("depth"),
        "error should name the depth limit, got: {err:?}"
    );
}

#[test]
fn schema_error_display() {
    let e = SchemaError("test".to_string());
    assert!(e.to_string().contains("test"));
}

#[test]
fn unsupported_type_returns_error() {
    let v = serde_json::json!({"type": "binary"});
    assert!(compile_json_schema(&v).is_err());
}

#[test]
fn ref_not_found_returns_error() {
    let v = serde_json::json!({"$ref": "#/$defs/Missing"});
    assert!(compile(&v).is_err());
}

/// Draft 2020-12 applies supported assertion siblings alongside `$ref`.
/// Until the grammar can represent these intersections, compiling the
/// referenced target alone would widen each schema and must fail closed.
///
/// Mutation guard: removing the narrowing-key check makes every case
/// compile successfully by dropping the sibling.
#[test]
fn ref_narrowing_siblings_fail_closed() {
    for (keyword, schema) in [
        (
            "const",
            serde_json::json!({
                "$defs": { "S": { "type": "string" } },
                "$ref": "#/$defs/S",
                "const": "a"
            }),
        ),
        (
            "enum",
            serde_json::json!({
                "$defs": { "S": { "type": "string" } },
                "$ref": "#/$defs/S",
                "enum": ["a"]
            }),
        ),
        (
            "type",
            serde_json::json!({
                "$defs": { "N": { "type": "number" } },
                "$ref": "#/$defs/N",
                "type": "integer"
            }),
        ),
    ] {
        let err =
            compile(&schema).expect_err("an unrepresentable narrowing sibling must not be dropped");
        assert!(
            err.0.contains(&format!("`{keyword}` sibling")) && err.0.contains("`$ref`"),
            "error should name `{keyword}` and `$ref`, got: {}",
            err.0
        );
    }
}

#[test]
fn defs_resolved() {
    let schema = serde_json::json!({
        "$defs": {
            "Status": {"type": "string", "enum": ["ok", "err"]}
        },
        "type": "object",
        "properties": {
            "status": {"$ref": "#/$defs/Status"}
        },
        "required": ["status"]
    });
    let g = compile(&schema).unwrap();
    assert!(accepts(&g, b"{\"status\":\"ok\"}"));
    assert!(accepts(&g, b"{\"status\":\"err\"}"));
}

#[test]
fn object_required_name_without_properties_is_enforced() {
    let schema = serde_json::json!({
        "type": "object",
        "required": ["y"]
    });
    let g = compile(&schema).expect("an undeclared required name has any-value semantics");

    assert!(rejects(&g, b"{}"), "an object missing y must be rejected");
    for value in ["1", "\"value\"", "true", "null", "{}", "[]"] {
        let instance = format!("{{\"y\":{value}}}");
        assert!(
            accepts(&g, instance.as_bytes()),
            "required y must accept supported JSON value {value}"
        );
    }
}

/// Issue #1078: a `required` sibling alongside `$ref` must be enforced as
/// a conjunction with the target, not silently dropped. `Base` declares
/// `y` as an OPTIONAL property; the sibling makes it required, so an
/// object omitting `y` must now be rejected even though `Base` alone would
/// have allowed it.
///
/// Mutation guard: reverting `compile_schema_inner`'s `$ref` dispatch to
/// the plain early-return `self.compile_ref(ref_str)` (dropping the
/// sibling) makes `rejects(&g, b"{}")` false — `{}` would satisfy `Base`
/// alone — so this assertion fails.
#[test]
fn ref_required_sibling_is_enforced() {
    let schema = serde_json::json!({
        "$defs": {
            "Base": {
                "type": "object",
                "properties": { "y": { "type": "integer" } }
            }
        },
        "$ref": "#/$defs/Base",
        "required": ["y"]
    });
    let g = compile(&schema).unwrap();
    assert!(
        accepts(&g, b"{\"y\":1}"),
        "y present must still be accepted"
    );
    assert!(
        rejects(&g, b"{}"),
        "required sibling on $ref must be enforced, not dropped"
    );
}

/// A `$ref` whose target is itself a `$ref` reaches the object only
/// through the chain. Deciding the merge on the first link alone finds no
/// `"type": "object"` there and falls through to the plain `$ref` compile,
/// which resolves the rest of the chain but has already discarded the
/// `required` sibling — so `{}` would compile as acceptable.
#[test]
fn ref_chain_required_sibling_is_enforced() {
    let schema = serde_json::json!({
        "$defs": {
            "A": { "$ref": "#/$defs/B" },
            "B": {
                "type": "object",
                "properties": { "y": { "type": "integer" } }
            }
        },
        "$ref": "#/$defs/A",
        "required": ["y"]
    });
    let g = compile(&schema).unwrap();
    assert!(
        accepts(&g, b"{\"y\":1}"),
        "y present must still be accepted through the chain"
    );
    assert!(
        rejects(&g, b"{}"),
        "required sibling must survive $ref chain resolution, not be dropped"
    );
}

/// A chain link carrying assertions of its own cannot be merged with the
/// referring schema's siblings, and must say so instead of falling through
/// to the permissive plain-`$ref` path that drops those siblings.
#[test]
fn ref_chain_link_with_own_assertion_is_explicit_error() {
    let schema = serde_json::json!({
        "$defs": {
            "A": { "$ref": "#/$defs/B", "minProperties": 1 },
            "B": {
                "type": "object",
                "properties": { "y": { "type": "integer" } }
            }
        },
        "$ref": "#/$defs/A",
        "required": ["y"]
    });
    let Err(err) = compile(&schema) else {
        panic!("unmergeable chain link must be an explicit error, not a compiled grammar")
    };
    assert!(
        err.0.contains("minProperties"),
        "error should name the unmergeable keyword, got: {}",
        err.0
    );
}

/// A cyclic chain of pure `$ref` links reports the cycle rather than
/// looping while resolving the merge target.
#[test]
fn ref_chain_cycle_is_reported() {
    let schema = serde_json::json!({
        "$defs": {
            "A": { "$ref": "#/$defs/B" },
            "B": { "$ref": "#/$defs/A" }
        },
        "$ref": "#/$defs/A",
        "required": ["y"]
    });
    let Err(err) = compile(&schema) else {
        panic!("cyclic $ref chain must be reported, not compiled")
    };
    assert!(
        err.0.contains("cycles"),
        "error should name the cycle, got: {}",
        err.0
    );
}

/// Draft 2020-12 applies a `$ref` sibling as an intersection, so a
/// property declared on both sides must satisfy both schemas. An integer
/// and a string declaration share no value, so no document satisfies the
/// schema — compiling it to the sibling's `string` (merge-patch
/// precedence) accepts `{"x":"s"}`, which the schema admits for no value
/// of `x` at all.
#[test]
fn ref_sibling_property_collision_is_not_resolved_by_precedence() {
    let schema = serde_json::json!({
        "$defs": {
            "Base": {
                "type": "object",
                "properties": { "x": { "type": "integer" } },
                "required": ["x"]
            }
        },
        "$ref": "#/$defs/Base",
        "properties": { "x": { "type": "string" } }
    });
    match compile(&schema) {
        Err(e) => assert!(
            e.0.contains("`x`"),
            "error should name the colliding property, got: {}",
            e.0
        ),
        Ok(g) => assert!(
            rejects(&g, b"{\"x\":\"s\"}"),
            "colliding property must not compile to the sibling's declaration alone"
        ),
    }
}

/// The representable half of the same rule: an empty sibling declaration
/// admits every value, so the intersection is the target's declaration and
/// the schema still compiles — enforcing the sibling's `required` AND the
/// target's `integer` type. Merge-patch precedence would install the empty
/// schema and accept `{"x":"s"}`.
#[test]
fn ref_sibling_property_collision_conjunction_enforces_both() {
    let schema = serde_json::json!({
        "$defs": {
            "Base": {
                "type": "object",
                "properties": { "x": { "type": "integer" } }
            }
        },
        "$ref": "#/$defs/Base",
        "properties": { "x": {} },
        "required": ["x"]
    });
    let g = compile(&schema).unwrap();
    assert!(accepts(&g, b"{\"x\":1}"), "integer x must be accepted");
    assert!(
        rejects(&g, b"{\"x\":\"s\"}"),
        "target's integer constraint must survive the collision"
    );
    assert!(rejects(&g, b"{}"), "sibling's required must be enforced");
}

/// `required`/`properties` only constrain object instances, so they are
/// vacuous beside a target explicitly pinned to a non-object type. This
/// test pins that case to the unmerged target; unlike a value/type
/// narrowing sibling, dropping these assertions cannot widen the target.
#[test]
fn ref_sibling_on_non_object_target_falls_through() {
    let schema = serde_json::json!({
        "$defs": { "N": { "type": "integer" } },
        "$ref": "#/$defs/N",
        "required": ["y"]
    });
    let g = compile(&schema).unwrap();
    assert!(accepts(&g, b"7"), "non-object $ref target still compiles");
}

/// The merge rebuilds the target from `properties`/`required` alone, so a
/// target carrying any further assertion must NOT take the merge path: the
/// rebuild cannot express that assertion and would drop it. An `enum`
/// target is the sharpest case, because the obvious alternative to
/// rejecting — fall back to the plain `$ref` path, which dispatches `enum`
/// ahead of object compilation and so keeps the target's own assertion —
/// is wrong for a reason that does not show up unless the sibling is
/// chosen carefully.
///
/// The sibling here is deliberately NOT redundant: `enum` admits both `{}`
/// and `{"x":1}`, and `required: ["x"]` excludes the first, so the
/// schema's language is `{"x":1}` alone. Compiling the target by itself
/// would accept `{}` — wider than the conjunction even though it is no
/// wider than the target. A test whose enum members all satisfy the
/// sibling cannot tell the two apart and would pass either way.
#[test]
fn ref_sibling_merge_rejects_a_target_with_an_enum_assertion() {
    let schema = serde_json::json!({
        "$defs": {
            "Base": {
                "type": "object",
                "enum": [{}, { "x": 1 }],
                "properties": { "x": { "type": "integer" } }
            }
        },
        "$ref": "#/$defs/Base",
        "required": ["x"]
    });
    let err = compile(&schema).expect_err(
        "the conjunction of an enum target and a required sibling is not representable",
    );
    assert!(
        err.0.contains("conjunction"),
        "error should name the unrepresentable conjunction, got: {}",
        err.0
    );
}

/// The same rejection for `const`, which `compile_schema_inner` dispatches
/// at the same precedence as `enum`. `const: {}` against `required: ["x"]`
/// is the degenerate case: the conjunction matches NO value at all, so any
/// grammar that compiles and accepts something is wrong by construction.
#[test]
fn ref_sibling_merge_rejects_a_target_with_a_const_assertion() {
    let schema = serde_json::json!({
        "$defs": {
            "Base": {
                "type": "object",
                "const": {},
                "properties": { "x": { "type": "integer" } }
            }
        },
        "$ref": "#/$defs/Base",
        "required": ["x"]
    });
    let err = compile(&schema)
        .expect_err("a const target contradicting its sibling cannot compile to a language");
    assert!(
        err.0.contains("conjunction"),
        "error should name the unrepresentable conjunction, got: {}",
        err.0
    );
}

#[test]
fn ref_required_name_without_properties_is_enforced() {
    let schema = serde_json::json!({
        "$defs": { "Base": { "type": "object" } },
        "$ref": "#/$defs/Base",
        "required": ["y"]
    });
    let g = compile(&schema)
        .expect("a required sibling without a property declaration has any-value semantics");
    assert!(
        accepts(&g, b"{\"y\":1}"),
        "an object carrying the required key must be accepted"
    );
    assert!(
        rejects(&g, b"{}"),
        "a reference result missing y must be rejected"
    );
}

/// The declared case must keep working, and this is the control that says
/// the rejection above is scoped to the undeclared name rather than to
/// `required` siblings generally.
#[test]
fn ref_sibling_merge_accepts_a_required_name_the_target_declares() {
    let schema = serde_json::json!({
        "$defs": {
            "Base": { "type": "object", "properties": { "y": { "type": "integer" } } }
        },
        "$ref": "#/$defs/Base",
        "required": ["y"]
    });
    let g = compile(&schema).expect("a declared required key merges normally");
    assert!(accepts(&g, b"{\"y\":1}"), "the required key is accepted");
    assert!(
        rejects(&g, b"{}"),
        "the merged `required` must actually be enforced"
    );
}

/// The counterpart to the two tests above, and the reason they are not the
/// whole rule. Falling back to the plain `$ref` path is only safe when that
/// path preserves the assertion that blocked the merge — `enum`/`const`,
/// which `compile_schema_inner` dispatches ahead of object compilation.
///
/// `additionalProperties` is not supported by this compiler at all, so the
/// plain path ignores it. Falling back would therefore keep NOTHING from
/// the target and drop the caller's `required: ["y"]` as well, returning
/// `Ok` on a grammar that accepts `{}` — wider than either operand. The
/// conjunction is not representable, so compilation must fail instead.
#[test]
fn ref_sibling_merge_rejects_a_target_whose_assertion_the_plain_path_would_ignore() {
    let schema = serde_json::json!({
        "$defs": {
            "Base": {
                "type": "object",
                "additionalProperties": false,
                "properties": { "x": { "type": "integer" } }
            }
        },
        "$ref": "#/$defs/Base",
        "required": ["y"]
    });
    let err = compile(&schema).expect_err(
        "an unrepresentable conjunction must fail closed, not compile to the wider target",
    );
    assert!(
        err.0.contains("conjunction"),
        "error should name the unrepresentable conjunction, got: {}",
        err.0
    );
}

/// An UNTYPED target is the same hazard reached a different way, and the
/// non-object guard must not absorb it: `type` is absent, so the target is
/// not `"object"`, but an untyped schema still matches object instances —
/// the siblings are live, not vacuous. The plain path compiles an untyped
/// schema to "any JSON value", which admits every object `required: ["y"]`
/// exists to exclude.
#[test]
fn ref_sibling_merge_rejects_an_untyped_target() {
    let schema = serde_json::json!({
        "$defs": { "Any": {} },
        "$ref": "#/$defs/Any",
        "required": ["y"]
    });
    let err = compile(&schema).expect_err("an untyped target cannot carry the sibling conjunction");
    assert!(
        err.0.contains("untyped"),
        "error should name the untyped target, got: {}",
        err.0
    );
}

// -----------------------------------------------------------------------
// Issue #355 — mixed required / optional object properties
// -----------------------------------------------------------------------
//
// The old flat-sequence grammar pushed the "," separator unconditionally
// before every non-first property and then wrapped optional properties in
// an `opt_pair` rule that could collapse to ε.  When the optional collapsed,
// the mandatory comma was already committed → the closing `}` was rejected.
//
// The new tail-chain construction mirrors `build_bounded_tail` for arrays:
// each optional property *owns* its leading comma so that the comma is only
// emitted when the property is present.  The PDA can distinguish the emit
// path (starts with `","`) from the skip path (reaches ε / `}`) at the
// first byte — no backtracking needed.

/// Both properties required — baseline sanity check.
#[test]
fn object_both_required() {
    let g = compile_ok(
        r#"{
                "type":"object",
                "properties":{"r":{"type":"integer"},"s":{"type":"integer"}},
                "required":["r","s"]
            }"#,
    );
    assert!(
        accepts(&g, b"{\"r\":1,\"s\":2}"),
        "fully-present object must be accepted"
    );
    assert!(
        rejects(&g, b"{\"r\":1}"),
        "missing required s must be rejected"
    );
    assert!(
        rejects(&g, b"{}"),
        "empty object must be rejected when both required"
    );
}

/// Single optional property only — object may be empty or have the property.
#[test]
fn object_single_optional_only() {
    let g = compile_ok(
        r#"{
                "type":"object",
                "properties":{"o":{"type":"integer"}}
            }"#,
    );
    assert!(
        accepts(&g, b"{}"),
        "empty object must be accepted when property is optional"
    );
    assert!(
        accepts(&g, b"{\"o\":1}"),
        "object with optional property must be accepted"
    );
    // Trailing comma is invalid JSON — must reject.
    assert!(
        rejects(&g, b"{\"o\":1,}"),
        "trailing comma must be rejected"
    );
}

/// Reproducer from issue #355: one required + one trailing optional.
/// Both `{"r":1}` and `{"r":1,"o":2}` were wrongly rejected before the fix.
#[test]
fn object_required_plus_trailing_optional_issue_355() {
    let g = compile_ok(
        r#"{
                "type":"object",
                "properties":{"r":{"type":"integer"},"o":{"type":"integer"}},
                "required":["r"]
            }"#,
    );
    // These two must now be accepted (were both rejected before the fix).
    assert!(
        accepts(&g, b"{\"r\":1}"),
        "required-only object must be accepted (issue #355)"
    );
    assert!(
        accepts(&g, b"{\"r\":1,\"o\":2}"),
        "fully-present object must be accepted (issue #355)"
    );
    // Invalid inputs must remain rejected.
    assert!(rejects(&g, b"{}"), "missing required r must be rejected");
    assert!(
        rejects(&g, b"{\"o\":2}"),
        "object with only optional and no required must be rejected"
    );
    assert!(
        rejects(&g, b"{\"r\":1,}"),
        "trailing comma must be rejected"
    );
    assert!(
        rejects(&g, b"{\"r\":\"bad\"}"),
        "wrong value type for r must be rejected"
    );
}

/// Two trailing optionals after a required property.
#[test]
fn object_two_trailing_optionals() {
    let g = compile_ok(
        r#"{
                "type":"object",
                "properties":{
                    "r":{"type":"integer"},
                    "o1":{"type":"integer"},
                    "o2":{"type":"integer"}
                },
                "required":["r"]
            }"#,
    );
    // Required only.
    assert!(accepts(&g, b"{\"r\":1}"), "required-only must be accepted");
    // Required + first optional.
    assert!(
        accepts(&g, b"{\"r\":1,\"o1\":2}"),
        "r + o1 must be accepted"
    );
    // All three present.
    assert!(
        accepts(&g, b"{\"r\":1,\"o1\":2,\"o2\":3}"),
        "all three properties must be accepted"
    );
    // Missing required.
    assert!(rejects(&g, b"{}"), "empty object must be rejected");
    assert!(rejects(&g, b"{\"o1\":2}"), "missing r must be rejected");

    // r + o2 only (skip o1): the INTERLEAVED optional case. After the ","
    // the emit-o1 and skip-o1 alternatives are both live; the o2 key
    // matches only the skip branch, so the matcher follows it.
    assert!(
        accepts(&g, b"{\"r\":1,\"o2\":3}"),
        "r + o2 (interleaved optional, skip o1) must be accepted"
    );
    // Trailing comma boundary (see object_three_props_trailing_comma_boundary
    // for the full explanation):
    // - r only with trailing comma: rejects (no optional key follows).
    // - r+o1 with trailing comma: rejects. After the "," every live
    //   alternative needs a key byte, so "}" has no continuation.
    // - r+o1+o2 with trailing comma: rejects (no optional remaining).
    assert!(
        rejects(&g, b"{\"r\":1,}"),
        "trailing comma after r-only must reject"
    );
    assert!(
        rejects(&g, b"{\"r\":1,\"o1\":2,}"),
        "trailing comma after r+o1 must reject (every live alternative needs a key)"
    );
    assert!(
        rejects(&g, b"{\"r\":1,\"o1\":2,\"o2\":3,}"),
        "trailing comma after all three must reject"
    );
}

/// Required-first re-canonicalization (issue #355 reorder).
///
/// The compiler emits object properties in required-first order (required
/// properties in declaration order, then optional ones), not the
/// alphabetical declaration order of the source schema.  This is necessary
/// for PDA safety: with required properties first, the `started=false` phase
/// is a single-alternative chain, so the no-rewind PDA cannot commit to a
/// wrong key terminal (see `compile_object` and `build_object_tail`).
///
/// Consequence: the accepted canonical key order is required-first.  An
/// object written in a different key order is rejected.  This is the safe
/// direction for constrained decoding: JSON objects are unordered (RFC 8259),
/// the model is guided to emit the canonical order, and no invalid output is
/// ever produced.  It also fixes a real over-rejection: the prior flat
/// grammar made comma separators mandatory around omitted optionals, so a
/// required-only object was rejected; required-first construction accepts it.
#[test]
fn object_required_first_reorder() {
    // Schema declares r required; serde orders keys alphabetically
    // (o1, o2, r), so required-first reordering moves r to the front.
    let g = compile_ok(
        r#"{
                "type":"object",
                "properties":{
                    "o1":{"type":"integer"},
                    "o2":{"type":"integer"},
                    "r":{"type":"integer"}
                },
                "required":["r"]
            }"#,
    );
    // Required-only: accepted (the #355 fix; the old flat grammar rejected
    // this because the inter-property commas were mandatory).
    assert!(
        accepts(&g, b"{\"r\":1}"),
        "required-only object must be accepted (the #355 fix)"
    );
    // Canonical required-first order: accepted.
    assert!(
        accepts(&g, b"{\"r\":1,\"o1\":2,\"o2\":3}"),
        "required-first canonical order must be accepted"
    );
    // Alphabetical order (required key last): rejected. The accepted
    // canonical order is required-first; this documents the intentional
    // re-canonicalization (a different valid order, not a capability loss).
    assert!(
        rejects(&g, b"{\"o1\":2,\"o2\":3,\"r\":1}"),
        "non-canonical (alphabetical) key order is rejected (required-first re-canonicalization)"
    );
}

/// No over-acceptance: invalid JSON must still be rejected after the fix.
#[test]
fn object_no_over_acceptance() {
    let g = compile_ok(
        r#"{
                "type":"object",
                "properties":{"r":{"type":"integer"},"o":{"type":"integer"}},
                "required":["r"]
            }"#,
    );
    assert!(
        rejects(&g, b"{\"r\":1,\"o\":\"bad\"}"),
        "wrong type for optional must be rejected"
    );
    assert!(
        rejects(&g, b"{\"r\":1,,\"o\":2}"),
        "double comma must be rejected"
    );
    assert!(rejects(&g, b"{,\"r\":1}"), "leading comma must be rejected");
    assert!(rejects(&g, b"\"r\":1"), "missing braces must be rejected");
    assert!(
        rejects(&g, b"{\"r\":1,\"o\":2,}"),
        "trailing comma after optional must be rejected"
    );
}

/// Regression: the fully-present case (`{"r":1,"o":2}`) must be accepted.
/// This was also broken before the fix — the flat grammar made the comma
/// mandatory before the optional wrapper, so `{"r":1,"o":2}` was accepted
/// BUT `{"r":1}` was not.  Verify both directions are now correct.
#[test]
fn object_fully_present_and_absent_both_correct() {
    let g = compile_ok(
        r#"{
                "type":"object",
                "properties":{"a":{"type":"string"},"b":{"type":"boolean"}},
                "required":["a"]
            }"#,
    );
    assert!(
        accepts(&g, b"{\"a\":\"x\"}"),
        "required-only string must be accepted"
    );
    assert!(
        accepts(&g, b"{\"a\":\"x\",\"b\":true}"),
        "both present must be accepted"
    );
    assert!(
        accepts(&g, b"{\"a\":\"x\",\"b\":false}"),
        "both present (false) must be accepted"
    );
    assert!(
        rejects(&g, b"{\"b\":true}"),
        "missing required a must be rejected"
    );
    assert!(rejects(&g, b"{}"), "empty must be rejected");
}

/// Two fully-optional properties — trailing-comma and skip-first
/// behaviours.
///
/// Trailing comma after the first optional (`{"a":1,}`) rejects: after the
/// "," every live alternative needs the second key, so "}" has no
/// continuation. Skip-first (`{"b":2}` without `a`) is accepted: the
/// emit-a and skip-a alternatives are both live at the opening brace and
/// the `b` key matches only the skip branch.
#[test]
fn object_two_optional_only_skip_first_and_trailing_comma() {
    let g = compile_ok(
        r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}}"#,
    );
    assert!(
        accepts(&g, b"{}"),
        "empty object must be accepted when all optional"
    );
    assert!(
        accepts(&g, b"{\"a\":1}"),
        "first optional alone must be accepted"
    );
    assert!(
        accepts(&g, b"{\"a\":1,\"b\":2}"),
        "both present must be accepted"
    );
    // Skip-first ({"b":2} without a) is a valid object and must be reachable.
    assert!(
        accepts(&g, b"{\"b\":2}"),
        "skip-first optional must be accepted"
    );
    // Trailing comma {"a":1,} rejects: after the "," the second key is
    // required, so "}" has no continuation.
    assert!(
        rejects(&g, b"{\"a\":1,}"),
        "trailing comma after first optional must reject (the second key must follow)"
    );
    // When both properties are present the trailing comma correctly rejects
    // (no further optional to provide an escape route).
    assert!(
        rejects(&g, b"{\"a\":1,\"b\":2,}"),
        "trailing comma after last property must be rejected"
    );
}

/// Three-property schema (1 required + 2 trailing optionals) — documents
/// trailing-comma boundary for the intermediate optional case.
///
/// `{"r":1,"o1":2,}` rejects: after the "," every live alternative needs
/// the o2 key, so "}" has no continuation (the same boundary exercised by
/// `object_two_optional_only_skip_first_and_trailing_comma` above).
/// `{"r":1,}` rejects for the same reason with o1. `{"r":1,"o1":2,"o2":3,}`
/// rejects because no optional remains.
#[test]
fn object_three_props_trailing_comma_boundary() {
    let g = compile_ok(
        r#"{
            "type":"object",
            "properties":{
                "r":{"type":"integer"},
                "o1":{"type":"integer"},
                "o2":{"type":"integer"}
            },
            "required":["r"]
        }"#,
    );
    // Correct accepts.
    assert!(accepts(&g, b"{\"r\":1}"), "required-only must be accepted");
    assert!(accepts(&g, b"{\"r\":1,\"o1\":2}"), "r+o1 must be accepted");
    assert!(
        accepts(&g, b"{\"r\":1,\"o1\":2,\"o2\":3}"),
        "all three must be accepted"
    );
    // Inline fix: trailing comma after required-only correctly rejects.
    assert!(
        rejects(&g, b"{\"r\":1,}"),
        "trailing comma after required-only must be rejected (inline-key fix)"
    );
    // Trailing comma after intermediate optional correctly rejects.
    assert!(
        rejects(&g, b"{\"r\":1,\"o1\":2,}"),
        "trailing comma after r+o1 must reject (every live alternative needs a key)"
    );
    // Trailing comma after last property correctly rejects.
    assert!(
        rejects(&g, b"{\"r\":1,\"o1\":2,\"o2\":3,}"),
        "trailing comma after last property must be rejected"
    );
}

/// Property-count cap (DoS guard): an object schema with more than
/// `MAX_OBJECT_PROPERTIES` properties is rejected at the parse boundary
/// rather than recursing once per property in `build_object_tail` and
/// exhausting the stack at `GrammarEngine::new`. Mirrors the
/// `MAX_ARRAY_CARDINALITY` guard. The cap fires before any per-property rule
/// is materialized, so this test is fast and never deep-recurses; it fails
/// (compile returns `Ok`) when the cap check is reverted.
#[test]
fn object_property_count_cap_rejects() {
    let over = MAX_OBJECT_PROPERTIES + 1;
    let mut props = String::new();
    for i in 0..over {
        if i > 0 {
            props.push(',');
        }
        props.push_str(&format!("\"p{i}\":{{\"type\":\"integer\"}}"));
    }
    let schema_json = format!("{{\"type\":\"object\",\"properties\":{{{props}}}}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err("over-cap object schema must be rejected");
    assert!(
        err.0.contains("exceeds the supported limit"),
        "expected a property-count cap error, got: {}",
        err.0
    );
}

/// A modest object (well under the cap) compiles and accepts a valid value:
/// the cap does not reject ordinary schemas.
#[test]
fn object_under_property_cap_compiles() {
    let mut props = String::new();
    for i in 0..50 {
        if i > 0 {
            props.push(',');
        }
        props.push_str(&format!("\"p{i}\":{{\"type\":\"integer\"}}"));
    }
    let schema_json = format!("{{\"type\":\"object\",\"properties\":{{{props}}}}}");
    let g = compile_ok(&schema_json);
    assert!(
        accepts(&g, b"{}"),
        "empty object (all optional) must accept"
    );
    assert!(accepts(&g, b"{\"p0\":1}"), "single property must accept");
}

/// Literal-count cap via typed string-enum: a schema whose enum members
/// exceed MAX_STRING_LITERALS is rejected at the parse boundary with a
/// SchemaError naming the count limit. Fails (compile returns Ok) when the
/// count guard in compile_trie_literals is reverted.
#[test]
fn string_literal_count_cap_typed_enum_rejects() {
    let values: String = (0..=MAX_STRING_LITERALS)
        .map(|i| format!("\"v{i}\""))
        .collect::<Vec<_>>()
        .join(",");
    let schema_json = format!("{{\"type\":\"string\",\"enum\":[{values}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err("over-cap string enum must be rejected");
    assert!(
        err.0.contains("exceeds the supported limit"),
        "expected a literal-count cap error, got: {}",
        err.0
    );
}

/// Literal-count cap via anyOf const set: the same count guard fires when
/// an anyOf schema aggregates more than MAX_STRING_LITERALS string consts.
/// Proves the shared compile_trie_literals chokepoint is inherited by the
/// anyOf/oneOf entry point, not just compile_string_type.
#[test]
fn string_literal_count_cap_any_of_const_rejects() {
    let branches: String = (0..=MAX_STRING_LITERALS)
        .map(|i| format!("{{\"const\":\"c{i}\"}}"))
        .collect::<Vec<_>>()
        .join(",");
    let schema_json = format!("{{\"anyOf\":[{branches}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err("over-cap anyOf const set must be rejected");
    assert!(
        err.0.contains("exceeds the supported limit"),
        "expected a literal-count cap error, got: {}",
        err.0
    );
}

/// Byte-budget cap: a schema whose literals' total encoded bytes exceed 1 MiB
/// (but whose count is under MAX_STRING_LITERALS) is rejected at the parse
/// boundary. Fails (compile returns Ok) when the byte-budget guard in
/// compile_trie_literals is reverted.
#[test]
fn string_literal_byte_budget_cap_rejects() {
    // Build 2 strings each slightly over 512 KiB so together they clear 1 MiB
    // while staying well under the 4096 count cap.
    let long_val = "a".repeat(600 * 1024);
    let values = format!("\"{long_val}\",\"{long_val}b\"");
    let schema_json = format!("{{\"type\":\"string\",\"enum\":[{values}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err("over-byte-budget string enum must be rejected");
    assert!(
        err.0.contains("exceeds the supported limit"),
        "expected a byte-budget cap error, got: {}",
        err.0
    );
}

/// Raw-count cap fires before dedup in anyOf: a schema whose anyOf branches
/// contain more than MAX_STRING_LITERALS entries — even when all entries are
/// duplicates of a single value — is rejected before deduplication shrinks
/// the set below the limit. Fails (compile returns Ok) when the raw-count
/// guard added before sort/dedup in compile_any_of is reverted (MIN-2).
#[test]
fn string_literal_count_cap_any_of_duplicates_rejected_before_dedup() {
    // All branches are the same const; after dedup only 1 remains, which is
    // far under MAX_STRING_LITERALS. The raw count (MAX_STRING_LITERALS + 1)
    // must still be rejected because the cap fires before dedup.
    let branches: String = (0..=MAX_STRING_LITERALS)
        .map(|_| r#"{"const":"dup"}"#.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let schema_json = format!("{{\"anyOf\":[{branches}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v)
        .expect_err("anyOf with duplicate branches exceeding raw count cap must be rejected");
    assert!(
        err.0.contains("exceeds the supported limit"),
        "expected a literal-count cap error, got: {}",
        err.0
    );
}

/// Happy path: a small string literal set (well under both caps) compiles
/// successfully. Ensures the new guards do not reject valid schemas.
#[test]
fn string_literal_small_set_compiles() {
    let schema = r#"{"type":"string","enum":["alpha","beta","gamma"]}"#;
    let v: Value = serde_json::from_str(schema).unwrap();
    let g = compile(&v).expect("small string enum must compile");
    assert!(accepts(&g, b"\"alpha\""));
    assert!(accepts(&g, b"\"beta\""));
    assert!(rejects(&g, b"\"delta\""));
}

mod cardinality_caps;
