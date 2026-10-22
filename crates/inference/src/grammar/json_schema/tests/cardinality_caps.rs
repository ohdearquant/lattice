//! Cardinality and byte caps applied before allocations sized by untrusted schema input.

use super::*;

// -----------------------------------------------------------------------
// Issue #474 — early cardinality caps before untrusted-size allocations
// -----------------------------------------------------------------------

/// Early string-enum count guard (issue #474, compile_string_type path):
/// a `{"type":"string","enum":[...]}` with more than MAX_STRING_LITERALS
/// DISTINCT one-character members whose total encoded bytes stay under
/// MAX_STRING_LITERAL_BYTES must be rejected by the EARLY guard inserted
/// before the byte pre-pass, not by the late guard in compile_trie_literals.
///
/// Mutation pin: the early guard fires with "string enum literal count" in
/// the message. Reverting the early guard lets the byte pre-pass and
/// Vec::with_capacity run to completion, after which the late
/// compile_trie_literals guard fires with "string literal count" — a
/// different message — so the assertion below fails.
#[test]
fn string_enum_count_cap_early_guard_fires_before_alloc() {
    // Build MAX_STRING_LITERALS + 1 distinct single-character strings.
    // There are only 128 ASCII printable values, so we use two-char strings
    // to stay distinct. Total encoded bytes ≈ (MAX_STRING_LITERALS + 1) * 4
    // (each "XY" encodes as 4 bytes including quotes, minus 1 for the
    // pre-pass formula) — well under the 1 MiB byte budget.
    let over = MAX_STRING_LITERALS + 1;
    let values: String = (0..over)
        .map(|i| format!("\"x{i:04}\""))
        .collect::<Vec<_>>()
        .join(",");
    let schema_json = format!("{{\"type\":\"string\",\"enum\":[{values}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v)
        .expect_err("string enum over MAX_STRING_LITERALS must be rejected before allocation");
    assert!(
        err.0.contains("string enum literal count"),
        "early guard must fire with 'string enum literal count', got: {}",
        err.0
    );
}

/// prefixItems length cap (issue #474): a tuple array schema whose
/// `prefixItems` array has more than MAX_PREFIX_ITEMS entries and no
/// `maxItems` is rejected at the parse boundary before Vec::with_capacity
/// and before any positional compile_schema call. Mutation pin: reverting
/// the cap causes compile to either return Ok (memory exhausted silently) or
/// OOM; the expect_err assertion fails.
#[test]
fn prefix_items_count_cap_rejects() {
    // Build MAX_PREFIX_ITEMS + 1 trivial positional schemas.
    let over = MAX_PREFIX_ITEMS + 1;
    let items: String = (0..over)
        .map(|_| r#"{"type":"string"}"#)
        .collect::<Vec<_>>()
        .join(",");
    let schema_json = format!("{{\"type\":\"array\",\"prefixItems\":[{items}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v)
        .expect_err("prefixItems over MAX_PREFIX_ITEMS with no maxItems must be rejected");
    assert!(
        err.0.contains("prefixItems length"),
        "expected a prefixItems-length cap error, got: {}",
        err.0
    );
}

/// Happy path: a small prefixItems tuple (well under the cap) compiles and
/// accepts the correct value. Ensures the new cap does not reject valid schemas.
#[test]
fn prefix_items_small_tuple_compiles() {
    let schema = r#"{"type":"array","prefixItems":[{"type":"integer"},{"type":"string"}]}"#;
    let v: Value = serde_json::from_str(schema).unwrap();
    let g = compile(&v).expect("small prefixItems tuple must compile");
    assert!(
        accepts(&g, b"[1,\"hi\"]"),
        "a valid tuple value must be accepted"
    );
}

// -----------------------------------------------------------------------
// Issue #478 — adjacent pre-cap allocations (anyOf literal aggregation,
// object property/required collection)
// -----------------------------------------------------------------------

/// Early anyOf string-literal count guard, many-small-branches path
/// (issue #478): an anyOf schema whose foldable string-const branches
/// individually stay tiny but together exceed MAX_STRING_LITERALS must be
/// rejected by the per-extend guard inside compile_any_of's aggregation
/// loop, not by the post-loop backstop or the late compile_trie_literals
/// guard.
///
/// Each branch folds 2 literals (a 2-member untyped string `enum`, not a
/// single `const`) so the BRANCH count stays well under
/// MAX_ANYOF_BRANCHES (issue #474 finding 2) while the aggregated LITERAL
/// count still exceeds MAX_STRING_LITERALS — isolating the per-extend
/// literal guard from the unrelated branch-count guard, which would
/// otherwise fire first on a one-literal-per-branch construction sized to
/// `MAX_STRING_LITERALS + 1` branches.
///
/// Mutation pin: the early guard fires with "anyOf string literal count"
/// in the message. Reverting it lets `literals` accumulate unchecked
/// through the loop; the post-loop backstop then fires with the
/// DIFFERENT "string literal count" message (no "anyOf " prefix), so the
/// assertion below fails.
#[test]
fn anyof_string_literal_count_cap_early_guard_fires_before_alloc() {
    let branch_count = (MAX_STRING_LITERALS / 2) + 2;
    let branches: String = (0..branch_count)
        .map(|i| format!("{{\"enum\":[\"c{i}a\",\"c{i}b\"]}}"))
        .collect::<Vec<_>>()
        .join(",");
    let schema_json = format!("{{\"anyOf\":[{branches}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err =
        compile(&v).expect_err("anyOf branches summing past the literal cap must be rejected");
    assert!(
        err.0.contains("anyOf string literal count"),
        "early per-extend guard must fire with 'anyOf string literal count', got: {}",
        err.0
    );
}

/// Early anyOf string-literal count guard, single-oversized-branch path
/// (issue #478): ONE anyOf branch with a `{"type":"string","enum":[...]}`
/// over MAX_STRING_LITERALS members must be rejected by the guard inside
/// `string_class_of`, before it deep-copies every member into an owned
/// `String`. Distinct from the many-small-branches accumulation path
/// above: here a single `string_class_of` call would itself materialize
/// the oversized Vec if unguarded, before compile_any_of's loop body ever
/// runs again to check the cumulative total.
///
/// Mutation pin: fires with "anyOf string literal count". Reverting just
/// this guard still leaves the per-extend guard in compile_any_of, which
/// independently catches the returned oversized Vec on the same
/// iteration with the same message — so an ALL-STRING enum like this one
/// does not, by itself, distinguish the two guards. See
/// `anyof_single_branch_raw_cardinality_bypasses_filtered_count_guard`
/// below for a test that isolates the `string_class_of` guard alone.
#[test]
fn anyof_single_branch_string_enum_count_cap_rejects() {
    let over = MAX_STRING_LITERALS + 1;
    let values: String = (0..over)
        .map(|i| format!("\"v{i}\""))
        .collect::<Vec<_>>()
        .join(",");
    let schema_json = format!("{{\"anyOf\":[{{\"type\":\"string\",\"enum\":[{values}]}}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err("oversized single anyOf branch enum must be rejected");
    assert!(
        err.0.contains("anyOf string literal count"),
        "expected the anyOf string-literal cap error, got: {}",
        err.0
    );
}

/// Isolates the `string_class_of`-internal guard from compile_any_of's
/// per-extend guard (issue #478): a single anyOf branch declares
/// `{"type":"string","enum":[...]}` with over MAX_STRING_LITERALS
/// members, but all except ONE are non-string (integers). `type:"string"`
/// intersects the enum down to its string members, so the FILTERED count
/// compile_any_of's per-extend guard ever observes is 1 — far under the
/// cap. Only a guard on the RAW `arr.len()` inside `string_class_of`,
/// checked before the filter_map/to_string pass, can catch this: the
/// per-extend guard cannot, because it never sees the discarded members.
///
/// Mutation pin: reverting the `string_class_of`-internal guard makes
/// this schema compile successfully (`Ok`) instead of being rejected —
/// a correctness gap, not just a slower rejection — because nothing else
/// in the call chain re-checks the raw enum cardinality of one branch.
#[test]
fn anyof_single_branch_raw_cardinality_bypasses_filtered_count_guard() {
    let over = MAX_STRING_LITERALS + 1;
    let mut members: Vec<String> = (0..over).map(|i| i.to_string()).collect();
    members.push("\"only-string\"".to_string());
    let values = members.join(",");
    let schema_json = format!("{{\"anyOf\":[{{\"type\":\"string\",\"enum\":[{values}]}}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err(
        "oversized raw enum cardinality must be rejected even with only 1 surviving string",
    );
    assert!(
        err.0.contains("anyOf string literal count"),
        "expected the anyOf string-literal cap error, got: {}",
        err.0
    );
}

/// Early object-property count guard (issue #478): an object schema with
/// more than MAX_OBJECT_PROPERTIES properties is rejected by the guard
/// checked directly on the property map's `.len()` BEFORE
/// `.iter().collect::<Vec<_>>()`, with the "object property count"
/// message proving the pre-collect guard fired.
///
/// Mutation pin: reverting the pre-collect guard removes the only
/// remaining property-count check in `compile_object` (the old post-collect
/// check was replaced, not duplicated), so compile would have to either
/// succeed (wrong) or fail elsewhere with a different message.
#[test]
fn object_property_count_early_guard_fires_before_collect() {
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
        err.0.contains("object property count"),
        "expected the pre-collect object-property-count cap error, got: {}",
        err.0
    );
}

/// `required` cardinality is bounded independent of `properties` (issue
/// #478): an object schema whose `required` array exceeds
/// MAX_OBJECT_PROPERTIES entries is rejected before the
/// filter_map/collect into `Vec<&str>`, even when `properties` itself is
/// tiny. Mutation pin: fires with "object required count"; reverting the
/// guard lets the oversized array collect unchecked (no rejection at
/// all, since nothing downstream re-checks `required`'s cardinality).
#[test]
fn object_required_count_cap_rejects_before_collect() {
    let over = MAX_OBJECT_PROPERTIES + 1;
    let required_list: String = (0..over)
        .map(|i| format!("\"r{i}\""))
        .collect::<Vec<_>>()
        .join(",");
    let schema_json = format!(
        "{{\"type\":\"object\",\"properties\":{{\"p\":{{\"type\":\"integer\"}}}},\"required\":[{required_list}]}}"
    );
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err("over-cap required array must be rejected");
    assert!(
        err.0.contains("object required count"),
        "expected the pre-collect required-count cap error, got: {}",
        err.0
    );
}

// -----------------------------------------------------------------------
// Issue #474 (follow-up) — raw enum cardinality must dominate
// const-narrowing and non-string materialization, not just the
// already-guarded plain-enum paths above.
// -----------------------------------------------------------------------

/// A sibling `const` narrows the surviving string set to a single value,
/// but the RAW enum array is still attacker-controlled cardinality. The
/// guard inside `compile_string_type` must fire on `values.len()` before
/// the const retain runs, not on the post-retain `str_values.len()` (which
/// would always be 0 or 1 once narrowed, and so could never trip).
///
/// Mutation pin: fires with "string enum literal count". Reverting the
/// raw-cardinality guard (restoring a check placed after the const
/// narrowing) lets this schema compile successfully instead of being
/// rejected, because the narrowed set is always under the cap.
#[test]
fn string_const_plus_oversized_enum_rejected_by_raw_guard() {
    let over = MAX_STRING_LITERALS + 1;
    let values: String = (0..over)
        .map(|i| format!("\"v{i}\""))
        .collect::<Vec<_>>()
        .join(",");
    let schema_json = format!("{{\"type\":\"string\",\"const\":\"v0\",\"enum\":[{values}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err(
        "a const that narrows the enum to one survivor must not bypass the raw cardinality cap",
    );
    assert!(
        err.0.contains("string enum literal count"),
        "expected the string-enum literal-count cap error, got: {}",
        err.0
    );
}

/// Same const-narrowing bypass as above, but inside an `anyOf` branch,
/// exercising the `string_class_of` `(Some(c), Some(arr))` arm rather than
/// `compile_string_type`. Before the fix this arm scanned
/// `arr.iter().any(...)` for the const-membership check without ever
/// bounding `arr.len()`, while the sibling `(None, Some(arr))` arm was
/// already guarded.
///
/// Mutation pin: fires with "anyOf string literal count". Reverting the
/// guard hoisted ahead of the `match (const_val, enum_arr)` dispatch lets
/// this schema compile successfully instead of being rejected.
#[test]
fn anyof_string_const_plus_oversized_enum_rejected() {
    let over = MAX_STRING_LITERALS + 1;
    let values: String = (0..over)
        .map(|i| format!("\"v{i}\""))
        .collect::<Vec<_>>()
        .join(",");
    let schema_json =
        format!("{{\"anyOf\":[{{\"type\":\"string\",\"const\":\"v0\",\"enum\":[{values}]}}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err(
        "an anyOf branch's const-narrowed enum must not bypass the raw cardinality cap",
    );
    assert!(
        err.0.contains("anyOf string literal count"),
        "expected the anyOf string-literal cap error, got: {}",
        err.0
    );
}

/// A mixed/non-string `enum` (no `type:"string"`, not all-string members)
/// is routed to `compile_enum`, which allocates one `Alt` per member via
/// `.map(json_value_to_alt).collect()`. That allocation was previously
/// unbounded: only the all-string fast path (`compile_string_type`) had a
/// cardinality guard.
///
/// Mutation pin: fires with "enum value count". Reverting the guard added
/// at the top of `compile_enum` lets this schema compile successfully
/// instead of being rejected.
#[test]
fn non_string_enum_cardinality_capped() {
    let over = MAX_STRING_LITERALS + 1;
    let values: String = (0..over)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let schema_json = format!("{{\"enum\":[{values}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err("a non-string enum over the cap must be rejected");
    assert!(
        err.0.contains("enum value count"),
        "expected the enum-value-count cap error, got: {}",
        err.0
    );
}

// -----------------------------------------------------------------------
// Issue #474 (follow-up) — anyOf branch-count cap and byte
// budgets that were checked late (or not at all) previously.
// -----------------------------------------------------------------------

/// anyOf/oneOf branch-count cap (issue #474, finding 2): a schema whose
/// `anyOf` array declares more than MAX_ANYOF_BRANCHES entries is
/// rejected at the top of `compile_any_of`, before the classification
/// loop runs. anyOf/oneOf branch count was the one cardinality dimension
/// this file never bounded.
///
/// Mutation pin: fires with "anyOf/oneOf branch count". Reverting the
/// guard lets `compile_any_of` push one `other_subs` entry (and later one
/// recursive `compile_schema` call) per branch with no upper bound, so
/// compile returns `Ok` instead of `Err` and the assertion below fails.
#[test]
fn anyof_branch_count_capped() {
    let over = MAX_ANYOF_BRANCHES + 1;
    let branches: String = (0..over)
        .map(|_| r#"{"type":"boolean"}"#.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let schema_json = format!("{{\"anyOf\":[{branches}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err("anyOf branch count over the cap must be rejected");
    assert!(
        err.0.contains("anyOf/oneOf branch count"),
        "expected the anyOf branch-count cap error, got: {}",
        err.0
    );
}

/// Mixed-enum byte-budget cap (issue #474, finding 3): a non-string enum
/// (one oversized string member plus a non-string member, so `type` is
/// absent and members are not all-string, routing to `compile_enum`
/// rather than `compile_string_type`) whose encoded bytes exceed
/// MAX_STRING_LITERAL_BYTES is rejected before the per-member
/// `json_value_to_alt` allocation.
///
/// Backstopped: the schema-wide `guard_schema_string_bytes` pass at the
/// top of `compile_json_schema` now dominates and rejects this input before
/// `compile_enum`'s own byte-budget guard runs, so this test is
/// defense-in-depth for `compile_enum`'s guard, not independently
/// mutation-sensitive to it — see `schema_defs_name_byte_capped` for the
/// guard pinned to the entry-point pass.
///
/// Mutation pin: fires with "byte length" (a substring common to both the
/// entry-point guard's message and `compile_enum`'s own "enum value
/// encoded byte length" message, so this assertion holds regardless of
/// which guard catches it).
#[test]
fn mixed_enum_byte_budget_capped() {
    // One member well over the 1 MiB byte budget by itself, plus a
    // non-string member so this enum is not all-string and is routed to
    // compile_enum (not compile_string_type's own byte guard).
    let huge = "a".repeat(MAX_STRING_LITERAL_BYTES + 16);
    let schema_json = format!("{{\"enum\":[\"{huge}\",1]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err("mixed enum over the byte budget must be rejected");
    assert!(
        err.0.contains("byte length"),
        "expected the byte-length cap error, got: {}",
        err.0
    );
}

/// Mixed-enum CUMULATIVE byte-budget cap (issue #474): a mixed enum (a
/// trailing non-string member keeps it routed to `compile_enum` rather than
/// `compile_string_type`) whose individual string members are each well
/// under MAX_STRING_LITERAL_BYTES — so the schema-wide entry guard passes
/// every one — but whose COMBINED encoded bytes exceed the budget is
/// rejected by `compile_enum`'s incremental byte-budget guard.
///
/// Mutation pin: asserts the incremental guard's exact "enum value encoded
/// byte length" message. The schema-wide `guard_schema_string_bytes` entry
/// pass cannot fire here (no single member exceeds the cap), so reverting
/// `compile_enum`'s incremental guard drops the input to the late
/// `compile_trie_literals` backstop, whose "string literal encoded byte
/// length (N)" message lacks the "enum value" substring — making this the
/// only test that pins the incremental enum guard specifically (its
/// single-oversized-member sibling `mixed_enum_byte_budget_capped` is now
/// dominated by the entry guard).
#[test]
fn mixed_enum_cumulative_bytes_capped() {
    // Three ~400 KiB string members, each under the 1 MiB per-literal cap
    // (so the entry guard passes each), summing past 1 MiB; a trailing
    // non-string member keeps the enum mixed and routed to compile_enum.
    let chunk = "a".repeat(400 * 1024);
    let members: String = (0..3)
        .map(|i| format!(r#""{chunk}{i}""#))
        .collect::<Vec<_>>()
        .join(",");
    let schema_json = format!("{{\"enum\":[{members},1]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v)
        .expect_err("mixed enum whose cumulative bytes exceed the budget must be rejected");
    assert!(
        err.0
            .contains("enum value encoded byte length exceeds the supported limit"),
        "expected the incremental enum byte-budget cap error, got: {}",
        err.0
    );
}

/// anyOf const cumulative byte-budget cap (issue #474, finding 4): an
/// `anyOf` whose individual `const` branches are each modest but whose
/// COMBINED encoded bytes exceed MAX_STRING_LITERAL_BYTES — while the
/// branch COUNT (3) stays far under MAX_STRING_LITERALS — is rejected by
/// the incremental byte-budget guard inside `compile_any_of`'s
/// classification loop, not only by the late backstop in
/// `compile_trie_literals`.
///
/// Mutation pin: the incremental guard's message has no number between
/// "byte length" and "exceeds" ("string literal encoded byte length
/// exceeds the supported limit"); the late `compile_trie_literals`
/// backstop's message embeds the computed total in between ("byte length
/// (N) exceeds"). Reverting the incremental guard still leaves the late
/// backstop to reject (since dedup does not shrink 3 distinct large
/// strings), but with the DIFFERENT embedded-number message, so the exact
/// substring assertion below fails.
#[test]
fn anyof_const_cumulative_bytes_capped() {
    // Each branch is ~400 KiB alone; three of them sum past 1 MiB while
    // the branch count (3) stays far under MAX_STRING_LITERALS.
    let chunk = "a".repeat(400 * 1024);
    let branches: String = (0..3)
        .map(|i| format!(r#"{{"const":"{chunk}{i}"}}"#))
        .collect::<Vec<_>>()
        .join(",");
    let schema_json = format!("{{\"anyOf\":[{branches}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v)
        .expect_err("anyOf consts whose cumulative bytes exceed the budget must be rejected");
    assert!(
        err.0
            .contains("string literal encoded byte length exceeds the supported limit"),
        "expected the incremental anyOf byte-budget cap error, got: {}",
        err.0
    );
}

// -----------------------------------------------------------------------
// Issue #474 (follow-up) — single-oversized-literal byte cap.
// The earlier fixes capped cardinality (many literals); this closes the
// distinct sub-class where ONE literal's own byte length exceeds
// MAX_STRING_LITERAL_BYTES before any prior guard sees it.
// -----------------------------------------------------------------------

/// Top-level const byte cap (issue #474): a top-level `const`
/// whose byte length exceeds MAX_STRING_LITERAL_BYTES is rejected by the
/// `guard_literal_bytes` check in `json_value_to_alt`, which has no other
/// byte guard upstream of it.
///
/// Mutation pin: fires with "byte length". Reverting the guard in
/// `json_value_to_alt` lets `serde_json::to_string` and the per-byte
/// `Symbol::Terminal` collect run to completion, so compile returns `Ok`
/// instead of `Err` and the assertion below fails.
#[test]
fn const_literal_byte_capped() {
    let huge = "a".repeat(MAX_STRING_LITERAL_BYTES + 1);
    let schema_json = format!("{{\"const\":\"{huge}\"}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err("a const literal over the byte cap must be rejected");
    assert!(
        err.0.contains("byte length"),
        "expected the byte-length cap error, got: {}",
        err.0
    );
}

/// Object property key byte cap (issue #474): an object property
/// whose KEY (not value) exceeds MAX_STRING_LITERAL_BYTES is rejected by
/// the `guard_literal_bytes` checks in `compile_object` (raw key) and
/// `json_string_literal` (JSON-encoded key), before the key reaches the
/// `into_bytes` alloc or the per-byte `Symbol::Terminal` expansion.
///
/// Mutation pin: fires with "byte length". Reverting BOTH guards lets the
/// key flow through unbounded, so compile returns `Ok` instead of `Err`
/// and the assertion below fails. (Reverting only one guard leaves the
/// other as a backstop — the pair is what this test pins.)
#[test]
fn object_key_byte_capped() {
    let huge_key = "a".repeat(MAX_STRING_LITERAL_BYTES + 1);
    let schema_json = format!(
        "{{\"type\":\"object\",\"properties\":{{\"{huge_key}\":{{\"type\":\"integer\"}}}}}}"
    );
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err("an object key over the byte cap must be rejected");
    assert!(
        err.0.contains("byte length"),
        "expected the byte-length cap error, got: {}",
        err.0
    );
}

/// anyOf const single-byte cap (issue #474): an `anyOf` branch
/// with a single oversized `const` is rejected by the `guard_literal_bytes`
/// check in `string_class_of`'s const arm.
///
/// Backstopped: the earlier incremental byte-budget guard inside
/// `compile_any_of`'s classification loop (Fix 4) also catches this input,
/// so this test stays green even if this guard alone is reverted.
/// It is defense-in-depth for the transient ~1x copy at the exact line,
/// not independently mutation-sensitive — see
/// `object_key_byte_capped` / `const_literal_byte_capped` for the
/// mutation-pinned guards.
#[test]
fn anyof_const_single_byte_capped() {
    let huge = "a".repeat(MAX_STRING_LITERAL_BYTES + 1);
    let schema_json = format!("{{\"anyOf\":[{{\"const\":\"{huge}\"}}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err("an anyOf const over the byte cap must be rejected");
    assert!(
        err.0.contains("byte length"),
        "expected the byte-length cap error, got: {}",
        err.0
    );
}

/// anyOf enum member single-byte cap (issue #474): an `anyOf`
/// branch with a single oversized `enum` member is rejected by the
/// `guard_literal_bytes` checks in `string_class_of`'s enum arms.
///
/// Backstopped: same as `anyof_const_single_byte_capped` — the earlier
/// incremental byte-budget guard in `compile_any_of` also catches this
/// input, so this test is defense-in-depth, not independently
/// mutation-sensitive.
#[test]
fn anyof_enum_member_byte_capped() {
    let huge = "a".repeat(MAX_STRING_LITERAL_BYTES + 1);
    let schema_json = format!("{{\"anyOf\":[{{\"enum\":[\"{huge}\"]}}]}}");
    let v: Value = serde_json::from_str(&schema_json).unwrap();
    let err = compile(&v).expect_err("an anyOf enum member over the byte cap must be rejected");
    assert!(
        err.0.contains("byte length"),
        "expected the byte-length cap error, got: {}",
        err.0
    );
}

// -----------------------------------------------------------------------
// Issue #474 (entry-point convergent fix) — validate every schema string
// once at `compile`, before any per-site guard runs. Closes sibling sites
// (e.g. a `$defs` name) that have no per-site byte guard of their own.
// -----------------------------------------------------------------------

/// `$defs` name byte cap (issue #474): a `$defs` map whose KEY (the
/// definition name itself, not a `properties` key) exceeds
/// MAX_STRING_LITERAL_BYTES is rejected by the schema-wide
/// `guard_schema_string_bytes` pass at the top of `compile_json_schema`. Unlike a
/// `properties` key (`object_key_byte_capped`), a `$defs` name has no
/// per-site guard: `CompileCtx::new` clones it into the `defs` map
/// unchecked, and nothing in this schema references it via `$ref`, so no
/// downstream lookup ever sees it either.
///
/// Mutation pin: fires with "byte length". Reverting the
/// `guard_schema_string_bytes` call in `compile_json_schema` removes the only
/// guard that ever inspects this key, so compile returns `Ok` instead of `Err`
/// and the assertion below fails.
#[test]
fn schema_defs_name_byte_capped() {
    let huge = "a".repeat(MAX_STRING_LITERAL_BYTES + 1);
    let mut defs = serde_json::Map::new();
    defs.insert(huge, serde_json::json!({"type": "integer"}));
    let mut root = serde_json::Map::new();
    root.insert("$defs".to_string(), Value::Object(defs));
    let v = Value::Object(root);
    let err = compile(&v).expect_err("a $defs name over the byte cap must be rejected");
    assert!(
        err.0.contains("byte length"),
        "expected the byte-length cap error, got: {}",
        err.0
    );
}

/// `compile_json_schema` is itself `pub`, so a direct caller bypasses the
/// `compile` wrapper. The schema-wide byte guard must therefore live in
/// `compile_json_schema`, not only in `compile`: an oversized `$defs` name
/// passed straight to `compile_json_schema` must still be rejected before
/// `CompileCtx::new` clones it into the `defs` map.
///
/// Mutation pin: fires with "byte length". Moving the
/// `guard_schema_string_bytes` call back into `compile` (leaving
/// `compile_json_schema` unguarded) lets the oversized key reach
/// `CompileCtx::new`, so this direct call returns `Ok` and the assertion fails.
#[test]
fn compile_json_schema_direct_defs_name_byte_capped() {
    let huge = "a".repeat(MAX_STRING_LITERAL_BYTES + 1);
    let mut defs = serde_json::Map::new();
    defs.insert(huge, serde_json::json!({"type": "integer"}));
    let mut root = serde_json::Map::new();
    root.insert("$defs".to_string(), Value::Object(defs));
    let v = Value::Object(root);
    let err = compile_json_schema(&v)
        .expect_err("a $defs name over the byte cap must be rejected at the direct entry");
    assert!(
        err.0.contains("byte length"),
        "expected the byte-length cap error, got: {}",
        err.0
    );
}

/// `$defs` name CUMULATIVE byte cap (issue #474): the schema-wide
/// entry guard is per-string, so many definition names each UNDER the
/// per-string cap can still sum past it while `CompileCtx::new` clones every
/// key into the `defs` map (and `$defs` has no cardinality cap). The
/// cumulative budget in `CompileCtx::new` rejects the aggregate.
///
/// Mutation pin: fires with the distinct "schema definition name cumulative
/// byte length" message. Each name here is ~400 KiB (under the 1 MiB
/// per-string cap, so the entry guard passes each one); three of them sum
/// past the cap. Reverting the cumulative guard in `CompileCtx::new` lets all
/// three through, so compile returns `Ok` and the assertion below fails.
#[test]
fn defs_names_cumulative_bytes_capped() {
    let chunk = "a".repeat(400 * 1024);
    let mut defs = serde_json::Map::new();
    for i in 0..3 {
        defs.insert(
            format!("{chunk}{i}"),
            serde_json::json!({"type": "integer"}),
        );
    }
    let mut root = serde_json::Map::new();
    root.insert("$defs".to_string(), Value::Object(defs));
    let v = Value::Object(root);
    let err =
        compile(&v).expect_err("cumulative $defs name bytes over the budget must be rejected");
    assert!(
        err.0
            .contains("schema definition name cumulative byte length"),
        "expected the cumulative $defs-name cap error, got: {}",
        err.0
    );
}

/// Object property key CUMULATIVE byte cap (issue #474): the
/// per-key `guard_literal_bytes` cap and `MAX_OBJECT_PROPERTIES` count cap
/// each bound one dimension, but their product (many near-cap keys) still
/// drives the per-byte `json_string_literal` Symbol expansion and the
/// retained `key_bytes` vectors. The cumulative budget in the property loop
/// rejects the aggregate.
///
/// Mutation pin: fires with the distinct "object property key cumulative
/// byte length" message. Each key here is ~400 KiB (under the per-string
/// cap, so `guard_literal_bytes` passes each one); three of them sum past
/// the cap. Reverting the cumulative guard in the property loop lets all
/// three through, so compile returns `Ok` and the assertion below fails.
#[test]
fn object_keys_cumulative_bytes_capped() {
    let chunk = "a".repeat(400 * 1024);
    let mut props = serde_json::Map::new();
    for i in 0..3 {
        props.insert(
            format!("{chunk}{i}"),
            serde_json::json!({"type": "integer"}),
        );
    }
    let mut root = serde_json::Map::new();
    root.insert("type".to_string(), Value::String("object".to_string()));
    root.insert("properties".to_string(), Value::Object(props));
    let v = Value::Object(root);
    let err = compile(&v)
        .expect_err("cumulative object property key bytes over the budget must be rejected");
    assert!(
        err.0.contains("object property key cumulative byte length"),
        "expected the cumulative object-key cap error, got: {}",
        err.0
    );
}
