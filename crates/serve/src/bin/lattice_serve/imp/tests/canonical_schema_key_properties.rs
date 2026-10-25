use super::*;
use proptest::prelude::*;

/// A string strategy biased toward the characters that make
/// JSON string/object-key escaping hard: quote, backslash,
/// every ASCII control byte incl. NUL/tab/CR/LF/ESC/DEL, and a
/// spread of multi-byte Unicode. `proptest::char::any()`'s
/// `DEFAULT_SPECIAL_CHARS` already includes exactly this set,
/// plus edge cases such as a lone BOM, the Unicode replacement
/// character, an RTL override, and a non-BMP character.
fn arb_json_string() -> impl Strategy<Value = String> {
    proptest::collection::vec(proptest::char::any(), 0..8)
        .prop_map(|chars| chars.into_iter().collect())
}

/// Finite floats that survive a text round trip exactly.
/// Non-finite input would become `Value::Null` through
/// `Value::from(f64)` and skip the float path entirely. Arbitrary
/// finite `f64`s are also unsuitable: without serde_json's
/// `float_roundtrip` feature, parsing is best-effort and can land
/// one ULP away from the printed value, which would fail the
/// round-trip assertion for a reason unrelated to the key.
/// Quarter steps of an `i32` keep the decimal short enough to
/// parse exactly while still covering fractional, integral
/// (`2.0`) and negative floats.
fn arb_finite_f64() -> impl Strategy<Value = f64> {
    any::<i32>().prop_map(|m| f64::from(m) * 0.25)
}

/// Recursive `serde_json::Value` generator covering every JSON
/// shape: null, bool, three numeric spellings (negative-capable
/// `i64`, the `u64` range beyond `i64::MAX`, and finite `f64`),
/// strings (see `arb_json_string`), arrays, and objects with
/// arbitrary string keys.
fn arb_json_value() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(|n| Value::Number(n.into())),
        any::<u64>().prop_map(|n| Value::Number(n.into())),
        arb_finite_f64().prop_map(Value::from),
        arb_json_string().prop_map(Value::String),
    ];
    leaf.prop_recursive(
        4,  // max depth
        32, // desired overall size
        4,  // expected items per collection
        |inner| {
            prop_oneof![
                proptest::collection::vec(inner.clone(), 0..4).prop_map(Value::Array),
                proptest::collection::vec((arb_json_string(), inner), 0..4)
                    .prop_map(|entries| Value::Object(entries.into_iter().collect())),
            ]
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// INJECTIVITY + ROUND-TRIP. Two independently generated
    /// schemas must not share a `canonical_schema_key` unless
    /// they are `serde_json::Value`-equal, and the key itself
    /// must parse back to its exact input.
    ///
    /// The round-trip check is the load-bearing half: it is
    /// what actually runs the `serde_json::to_string(s)
    /// .unwrap_or_default()` calls on the `Value::String` and
    /// object-key paths through a real parser rather than
    /// trusting the printed text by inspection -- an empty
    /// fragment from a would-be serialization failure breaks
    /// the surrounding JSON syntax and fails this parse (or, in
    /// the object-key case, silently merges two entries under
    /// one empty-string key, which the equality check below
    /// then catches instead).
    ///
    /// A note on what "equal" means for numbers: this crate
    /// does not enable serde_json's `arbitrary_precision`
    /// feature, so `serde_json::Number`'s `PartialEq` compares
    /// by *representation* (`PosInt(u64)` / `NegInt(i64)` /
    /// `Float(f64)`), not by mathematical value --
    /// `Value::from(1i64) != Value::from(1.0f64)` even though
    /// `1 == 1.0` as real numbers (see the dedicated MISS test
    /// below). Two values that land on the same representation
    /// compare equal (`Value::from(0i64) == Value::from(0u64)`,
    /// both `PosInt(0)`), and `canonical_schema_key` agrees --
    /// its `to_string()` gives both "0" -- so this is
    /// consistent with injectivity, not an exception to it.
    #[test]
    fn canonical_schema_key_is_injective_and_round_trips(
        a in arb_json_value(),
        b in arb_json_value(),
    ) {
        let key_a = canonical_schema_key(&a);
        let parsed_a: Value = serde_json::from_str(&key_a).unwrap_or_else(|e| {
            panic!(
                "canonical_schema_key output must parse as JSON: {e}\nkey: {key_a:?}"
            )
        });
        prop_assert_eq!(&parsed_a, &a, "key must round-trip back to its input");

        let key_b = canonical_schema_key(&b);
        let parsed_b: Value = serde_json::from_str(&key_b).unwrap_or_else(|e| {
            panic!(
                "canonical_schema_key output must parse as JSON: {e}\nkey: {key_b:?}"
            )
        });
        prop_assert_eq!(&parsed_b, &b, "key must round-trip back to its input");

        if key_a == key_b {
            prop_assert_eq!(
                a, b,
                "two generated schemas share a cache key but are not equal -- \
                 this is the exact defect class #1557 asks a differential test to \
                 catch"
            );
        }
    }

    /// ORDER INVARIANCE. An object's cache key must not depend
    /// on the order its properties were inserted in.
    ///
    /// `serde_json::Map<String, Value>` is `BTreeMap`-backed
    /// unless the `preserve_order` Cargo feature is unified
    /// into the build. Under that configuration `Map::keys()` already yields sorted order
    /// regardless of insertion order, so `forward` and
    /// `backward` below are the exact same `Value` before
    /// `canonical_schema_key` ever runs -- this test cannot, by
    /// itself, tell `canonical_schema_key`'s explicit
    /// `keys.sort()` apart from no sort at all under today's
    /// build. It is written against the function's observable
    /// behaviour rather than against that build detail
    /// specifically, exactly per the risk
    /// `canonical_schema_key`'s own doc comment already names:
    /// the day `preserve_order` gets unified in by some
    /// unrelated dependency bump, `Map::keys()` starts
    /// returning insertion order and this same, unmodified
    /// test starts actually exercising the sort.
    #[test]
    fn canonical_schema_key_is_invariant_to_generated_object_insertion_order(
        entries in proptest::collection::vec(
            (arb_json_string(), arb_json_value()),
            0..6,
        ),
    ) {
        // De-duplicate by key first: a repeated key inserted
        // twice is an overwrite, not a reordering, and would
        // make `forward` and `backward` disagree on CONTENT
        // rather than order.
        let mut deduped: Vec<(String, Value)> = Vec::new();
        for (k, v) in entries {
            if let Some(slot) = deduped.iter_mut().find(|(ek, _)| *ek == k) {
                slot.1 = v;
            } else {
                deduped.push((k, v));
            }
        }

        let forward: serde_json::Map<String, Value> =
            deduped.iter().cloned().collect();
        let mut reversed = deduped;
        reversed.reverse();
        let backward: serde_json::Map<String, Value> =
            reversed.into_iter().collect();

        prop_assert_eq!(
            canonical_schema_key(&Value::Object(forward)),
            canonical_schema_key(&Value::Object(backward)),
            "insertion order must not change the cache key"
        );
    }
}

/// MISS, DOCUMENTED AND DELIBERATE (#1557): `1` and `1.0` are
/// the same number and the same schema constraint, but
/// `serde_json` keeps them as distinct `Value`s (see the
/// numbers note on the injectivity test above), so
/// `canonical_schema_key` necessarily gives them different
/// keys too -- `Number::to_string()` already differs ("1" vs
/// "1.0") before the key is even built. Two schemas that only
/// differ this way get two cache entries and two compiles
/// instead of one, but the compiled grammars accept the same
/// language: the cost of this miss is a redundant recompile,
/// never a wrongly-served grammar. Pinned as a single fixed
/// case, not a property, because it documents one specific,
/// known, and accepted trade-off rather than a general shape.
#[test]
fn canonical_schema_key_distinguishes_equivalent_integer_and_float_number_spellings() {
    let int_schema = serde_json::json!({"type": "number", "minimum": 1});
    let float_schema = serde_json::json!({"type": "number", "minimum": 1.0});
    assert_ne!(
        int_schema, float_schema,
        "serde_json itself already keeps 1 and 1.0 apart as distinct Values"
    );
    assert_ne!(
        canonical_schema_key(&int_schema),
        canonical_schema_key(&float_schema),
        "1 vs 1.0: a documented cache MISS on numerically-equal spellings -- costs \
                     a redundant recompile, never serves the wrong grammar"
    );
}
